import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any

MetricsTracker = None
try:
    from .base import BaseModel
    from .common import (
        Attention,
        FocusedLinearAttention,
        SEBlock,
        ConvBlock,
        C3Module,
        DropPath,
        InvertedResidual,
    )
    from ..training.metrics import MetricsTracker
except:
    from base import BaseModel
    from common import (
        Attention,
        FocusedLinearAttention,
        SEBlock,
        ConvBlock,
        C3Module,
        DropPath,
        InvertedResidual,
    )


class Expert(nn.Module):
    def __init__(self, dim, hidden_dim=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MoEMLP(nn.Module):
    """
    Mixture-of-Experts MLP with efficient batched dispatch.

    Instead of gathering per-pair weights and doing T*K tiny bmms,
    we sort tokens by expert, pad into (E, C, D) tensors, and do
    just 2 batched matmuls for all experts simultaneously.

    Efficiency gains (B=32, N=257, K=6, E=64, D=80, H=160):
      Memory:  2.5GB  → ~15MB   (170× reduction)
      Matmuls: 49,152 tiny bmms → 2 batched bmms  (24,576× fewer launches)
    """

    def __init__(
        self,
        dim,
        num_shared=1,
        num_routed=64,
        num_activated_routed=6,
        expert_ratio=0.5,
        act_layer=nn.GELU,
        drop=0.0,
        balance_factor=0.01,
    ):
        super().__init__()
        self.num_shared = num_shared
        self.num_routed = num_routed
        self.num_activated_routed = num_activated_routed
        self.balance_factor = balance_factor

        expert_hidden = int(dim * 4 * expert_ratio)

        self.shared_experts = nn.ModuleList(
            [Expert(dim, expert_hidden, act_layer, drop) for _ in range(num_shared)]
        )

        self.expert_fc1_weight = nn.Parameter(
            torch.empty(num_routed, expert_hidden, dim)
        )
        self.expert_fc1_bias = nn.Parameter(torch.zeros(num_routed, expert_hidden))
        self.expert_fc2_weight = nn.Parameter(
            torch.empty(num_routed, dim, expert_hidden)
        )
        self.expert_fc2_bias = nn.Parameter(torch.zeros(num_routed, dim))

        nn.init.kaiming_uniform_(self.expert_fc1_weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.expert_fc2_weight, a=5**0.5)

        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.gate = nn.Linear(dim, num_routed, bias=False)

    def forward(self, x):
        B, N, dim = x.shape
        T = B * N
        K = self.num_activated_routed
        E = self.num_routed

        # Shared experts (unchanged)
        shared_out = sum(expert(x) for expert in self.shared_experts)

        # Gating
        gate_logits = self.gate(x)
        gate_scores = F.softmax(gate_logits, dim=-1)  # (B, N, E)
        top_scores, top_indices = torch.topk(gate_scores, K, dim=-1)  # (B, N, K)

        flat_x = x.reshape(T, dim)

        # ============================================================
        # Efficient dispatch: Sort → Pad → Batched BMM → Scatter
        # ============================================================

        # Step 1: Build (token_id, expert_id, score) pairs
        #   Each token has K pairs (one per selected expert)
        pair_token = (
            torch.arange(T, device=x.device).unsqueeze(1).expand(-1, K).reshape(-1)
        )  # (T*K,)
        pair_expert = top_indices.reshape(-1)  # (T*K,)
        pair_score = top_scores.reshape(-1)  # (T*K,)

        # Step 2: Sort all pairs by expert_id
        #   This groups tokens belonging to the same expert together
        order = pair_expert.argsort(stable=True)
        sorted_token = pair_token[order]  # (T*K,)
        sorted_expert = pair_expert[order]  # (T*K,)
        sorted_score = pair_score[order]  # (T*K,)
        sorted_x = flat_x[sorted_token]  # (T*K, dim)

        # Step 3: Compute per-expert token counts and capacity
        expert_counts = torch.bincount(sorted_expert, minlength=E)  # (E,)
        C = expert_counts.max().item()  # capacity = max tokens per expert

        if C == 0:
            # Edge case: no tokens routed (shouldn't happen in practice)
            routed_out = torch.zeros_like(x)
        else:
            # Step 4: Compute each pair's position within its expert group
            #   e.g., sorted_expert = [0,0,0, 1,1, 2,2,2,2, ...]
            #         position       = [0,1,2, 0,1, 0,1,2,3, ...]
            cumsum = torch.zeros(E + 1, device=x.device, dtype=torch.long)
            cumsum[1:] = expert_counts.cumsum(0)
            pos = torch.arange(T * K, device=x.device) - cumsum[sorted_expert]

            # Step 5: Pad into (E, C, dim) tensors for batched computation
            #   Padded positions remain zero → zero input → output zeroed by score
            padded_x = sorted_x.new_zeros(E, C, dim)
            padded_x[sorted_expert, pos] = sorted_x

            padded_score = sorted_score.new_zeros(E, C, 1)
            padded_score[sorted_expert, pos, 0] = sorted_score

            # Step 6: Batched expert forward — just 2 bmm calls!
            #   Shape: (E, C, dim) @ (E, dim, H) → (E, C, H)
            h = torch.bmm(padded_x, self.expert_fc1_weight.transpose(1, 2))
            h = h + self.expert_fc1_bias.unsqueeze(1)
            h = self.act(h)
            h = self.drop(h)

            #   Shape: (E, C, H) @ (E, H, dim) → (E, C, dim)
            out = torch.bmm(h, self.expert_fc2_weight.transpose(1, 2))
            out = out + self.expert_fc2_bias.unsqueeze(1)
            out = self.drop(out)

            # Weight by gate scores (padded positions have score=0 → zeroed)
            out = out * padded_score  # (E, C, dim)

            # Step 7: Extract valid outputs and scatter-add back to tokens
            valid_out = out[sorted_expert, pos]  # (T*K, dim)

            routed_out = flat_x.new_zeros(T, dim)
            routed_out.scatter_add_(
                0,
                sorted_token.unsqueeze(-1).expand(-1, dim),
                valid_out,
            )
            routed_out = routed_out.reshape(B, N, dim)

        # Balance loss (vectorized)
        expert_freq = expert_counts.float() / (T * K)
        expert_prob = gate_scores.reshape(T, -1).mean(dim=0)  # (E,)
        balance_loss = self.balance_factor * (expert_freq * expert_prob).sum()

        return shared_out + routed_out, balance_loss, expert_freq, expert_prob


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        linear_attention=False,
        moe_num_shared=1,
        moe_num_routed=64,
        moe_num_activated_routed=6,
        moe_expert_ratio=0.25,
        moe_balance_factor=0.01,
        drop_path=0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.attn = (
            Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
            if not linear_attention
            else FocusedLinearAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        )
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = MoEMLP(
            dim=dim,
            num_shared=moe_num_shared,
            num_routed=moe_num_routed,
            num_activated_routed=moe_num_activated_routed,
            expert_ratio=moe_expert_ratio,
            act_layer=nn.GELU,
            drop=drop,
            balance_factor=moe_balance_factor,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        mlp_out, aux_loss, expert_freq, expert_prob = self.mlp(self.norm2(x))
        x = x + mlp_out
        return x, aux_loss, expert_freq, expert_prob


class VisionTransformer(nn.Module):
    """Vision Transformer module for processing patch embeddings."""

    def __init__(
        self,
        embed_dim=256,
        num_patches=256,
        depth=8,
        num_heads=8,
        mlp_ratio=4.0,
        drop_rate=0.2,
        linear_attention=False,
        linear_layer_limit=4,
        moe_num_shared=1,
        moe_num_routed=64,
        moe_num_activated_routed=6,
        moe_expert_ratio=0.25,
        moe_balance_factor=0.01,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    linear_attention=linear_attention
                    if i < linear_layer_limit
                    else False,
                    moe_num_shared=moe_num_shared,
                    moe_num_routed=moe_num_routed,
                    moe_num_activated_routed=moe_num_activated_routed,
                    moe_expert_ratio=moe_expert_ratio,
                    moe_balance_factor=moe_balance_factor,
                )
                for i in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        aux_loss = 0
        expert_freq = None
        expert_prob = None
        for block in self.blocks:
            x, block_loss, block_freq, block_prob = block(x)
            aux_loss = aux_loss + block_loss
            if expert_freq is None:
                expert_freq = block_freq
                expert_prob = block_prob
            else:
                expert_freq = expert_freq + block_freq
                expert_prob = expert_prob + block_prob

        x = self.norm(x)

        return x, aux_loss, expert_freq, expert_prob


class MultiScaleVisionTransformer(nn.Module):
    """Vision Transformer with multi-scale token input and scale position encoding."""

    def __init__(
        self,
        embed_dim=256,
        num_scales=3,
        num_patches_per_scale=[1024, 256, 64],
        depth=8,
        num_heads=8,
        mlp_ratio=4.0,
        drop_rate=0.2,
        linear_attention=False,
        linear_layer_limit=4,
        moe_num_shared=1,
        moe_num_routed=64,
        moe_num_activated_routed=6,
        moe_expert_ratio=0.25,
        moe_balance_factor=0.01,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_scales = num_scales
        self.num_patches_per_scale = num_patches_per_scale

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embeds = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
                for num_patches in num_patches_per_scale
            ]
        )

        self.scale_embeds = nn.Parameter(torch.zeros(1, num_scales, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    linear_attention=linear_attention
                    if i < linear_layer_limit
                    else False,
                    moe_num_shared=moe_num_shared,
                    moe_num_routed=moe_num_routed,
                    moe_num_activated_routed=moe_num_activated_routed,
                    moe_expert_ratio=moe_expert_ratio,
                    moe_balance_factor=moe_balance_factor,
                )
                for i in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        for pos_embed in self.pos_embeds:
            nn.init.trunc_normal_(pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.scale_embeds, std=0.02)

    def forward(self, tokens_list):
        B = tokens_list[0].shape[0]

        scale_tokens = []
        for i, (tokens, pos_embed) in enumerate(zip(tokens_list, self.pos_embeds)):
            tokens = tokens + pos_embed[:, 1:, :]
            scale_embed = self.scale_embeds[:, i : i + 1, :].expand(
                B, tokens.shape[1], -1
            )
            tokens = tokens + scale_embed
            scale_tokens.append(tokens)

        x = torch.cat(scale_tokens, dim=1)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_drop(x)

        aux_loss = 0
        expert_freq = None
        expert_prob = None
        for block in self.blocks:
            x, block_loss, block_freq, block_prob = block(x)
            aux_loss = aux_loss + block_loss
            if expert_freq is None:
                expert_freq = block_freq
                expert_prob = block_prob
            else:
                expert_freq = expert_freq + block_freq
                expert_prob = expert_prob + block_prob

        x = self.norm(x)

        return x, aux_loss, expert_freq, expert_prob


class PyramidFeatureExtractor(nn.Module):
    """Concatenation-based Pyramid Feature Extractor with YOLOv5 C3 modules."""

    def __init__(
        self,
        input_channels=3,
        lateral_channels_list=[64, 128, 256],
        out_dim=256,
        num_bottlenecks=3,
        fusion_mode="32x32",
    ):
        super().__init__()
        self.fusion_mode = fusion_mode

        self.stem = ConvBlock(input_channels, lateral_channels_list[0], 3, 2, 1)
        self.layer1 = C3Module(
            lateral_channels_list[0],
            lateral_channels_list[0],
            num_bottlenecks=num_bottlenecks,
            shortcut=True,
        )

        self.down2 = ConvBlock(
            lateral_channels_list[0], lateral_channels_list[1], 3, 2, 1
        )
        self.layer2 = C3Module(
            lateral_channels_list[1],
            lateral_channels_list[1],
            num_bottlenecks=num_bottlenecks,
            shortcut=True,
        )

        self.down3 = ConvBlock(
            lateral_channels_list[1], lateral_channels_list[2], 3, 2, 1
        )
        self.layer3 = C3Module(
            lateral_channels_list[2],
            lateral_channels_list[2],
            num_bottlenecks=num_bottlenecks,
            shortcut=True,
        )

        if fusion_mode == "32x32":
            self.lateral1 = nn.Conv2d(lateral_channels_list[0], out_dim, 1)
            self.lateral2 = nn.Sequential(
                nn.Conv2d(lateral_channels_list[1], out_dim, 1),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            )
            self.lateral3 = nn.Sequential(
                nn.Conv2d(lateral_channels_list[2], out_dim, 1),
                nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            )
        else:
            self.lateral1 = nn.Conv2d(lateral_channels_list[0], out_dim, 1)
            self.lateral2 = nn.Conv2d(lateral_channels_list[1], out_dim, 1)
            self.lateral3 = nn.Conv2d(lateral_channels_list[2], out_dim, 1)

        self.se1 = SEBlock(out_dim)
        self.se2 = SEBlock(out_dim)
        self.se3 = SEBlock(out_dim)

        if fusion_mode == "32x32":
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(out_dim * 3, out_dim, 1),
                nn.BatchNorm2d(out_dim),
                nn.SiLU(inplace=True),
            )

    def forward(self, x):
        x = self.stem(x)
        c1 = self.layer1(x)
        x = self.down2(c1)
        c2 = self.layer2(x)
        x = self.down3(c2)
        c3 = self.layer3(x)

        p1 = self.lateral1(c1)
        p2 = self.lateral2(c2)
        p3 = self.lateral3(c3)

        p1 = self.se1(p1)
        p2 = self.se2(p2)
        p3 = self.se3(p3)

        if self.fusion_mode == "32x32":
            fused = torch.cat([p1, p2, p3], dim=1)
            out = self.fusion_conv(fused)
            return out
        else:
            return p1, p2, p3


class PyramidFeatureExtractorInvertedResidual(nn.Module):
    """InvertedResidual-based Pyramid（替换 C3）"""

    def __init__(
        self,
        input_channels=24,
        lateral_channels_list=[32, 64, 128],
        out_dim=64,
        fusion_mode="32x32",
    ):
        super().__init__()
        self.fusion_mode = fusion_mode

        # Stem: 轻量入口
        self.stem = nn.Sequential(
            nn.Conv2d(
                input_channels,
                lateral_channels_list[0],
                3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(lateral_channels_list[0]),
            nn.SiLU(inplace=True),
        )

        # Stage 1: 32×32
        ch0 = lateral_channels_list[0]
        self.stage1 = nn.Sequential(
            InvertedResidual(ch0, ch0, stride=1, expand_ratio=2.0),
            InvertedResidual(ch0, ch0, stride=1, expand_ratio=2.0),
        )

        # Stage 2: 16×16
        ch1 = lateral_channels_list[1]
        self.stage2 = nn.Sequential(
            InvertedResidual(ch0, ch1, stride=2, expand_ratio=2.5),
            InvertedResidual(ch1, ch1, stride=1, expand_ratio=2.5),
        )

        # Stage 3: 8×8
        ch2 = lateral_channels_list[2]
        self.stage3 = nn.Sequential(
            InvertedResidual(ch1, ch2, stride=2, expand_ratio=3.0),
            InvertedResidual(ch2, ch2, stride=1, expand_ratio=3.0),
        )

        # Lateral projections (不需要额外 SE，InvertedResidual 内部已有)
        if fusion_mode == "32x32":
            self.lateral1 = nn.Conv2d(ch0, out_dim, 1, bias=False)
            self.lateral2 = nn.Sequential(
                nn.Conv2d(ch1, out_dim, 1, bias=False),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            )
            self.lateral3 = nn.Sequential(
                nn.Conv2d(ch2, out_dim, 1, bias=False),
                nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            )
            self.fusion = nn.Sequential(
                nn.Conv2d(out_dim * 3, out_dim, 1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.SiLU(inplace=True),
            )
        else:
            self.lateral1 = nn.Conv2d(ch0, out_dim, 1, bias=False)
            self.lateral2 = nn.Conv2d(ch1, out_dim, 1, bias=False)
            self.lateral3 = nn.Conv2d(ch2, out_dim, 1, bias=False)

    def forward(self, x):
        x = self.stem(x)
        c1 = self.stage1(x)  # 32×32
        c2 = self.stage2(c1)  # 16×16
        c3 = self.stage3(c2)  # 8×8

        p1 = self.lateral1(c1)
        p2 = self.lateral2(c2)
        p3 = self.lateral3(c3)

        if self.fusion_mode == "32x32":
            return self.fusion(torch.cat([p1, p2, p3], dim=1)), c2
            #                                                    ↑
            #                                    返回中间特征供局部池化
        else:
            return (p1, p2, p3), c2


class FeaturePyramidMoEViT(BaseModel):
    """Vision Transformer with MoE MLP for Chinese Character Recognition."""

    @property
    def model_type(self) -> str:
        return "classification"

    @property
    def has_aux_loss(self) -> bool:
        return True

    @property
    def arch_type(self) -> str:
        return "moe"

    @classmethod
    def get_criterion(cls, **kwargs) -> nn.Module:
        return nn.CrossEntropyLoss()

    @classmethod
    def get_metrics_tracker(cls, **kwargs) -> Any:
        assert MetricsTracker is not None, "Cannot import MetricsTracker"
        return MetricsTracker()

    def __init__(
        self,
        img_size=64,
        preprocess_channels=32,
        fpn_out_channels=128,
        embed_dim=128,
        patch_size=16,
        input_channels=3,
        num_classes=631,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        drop_rate=0.2,
        lateral_channels_list=None,
        num_bottlenecks=3,
        fpn_mode="multiscale",
        linear_attention=True,
        linear_layer_limit=4,
        moe_num_shared=1,
        moe_num_routed=64,
        moe_num_activated_routed=6,
        moe_expert_ratio=0.25,
        moe_balance_factor=0.01,
        num_patches_per_scale=[1024, 256, 64],
        **kwargs,
    ):
        super().__init__(
            num_classes=num_classes, input_channels=input_channels, **kwargs
        )

        self.embed_dim = embed_dim
        self.fpn_mode = fpn_mode
        self.moe_num_routed = moe_num_routed
        self.input_size = img_size

        if lateral_channels_list is None:
            lateral_channels_list = [64, 128, 256]

        self.image_preprocess = nn.Sequential(
            nn.Conv2d(input_channels, preprocess_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(preprocess_channels),
            nn.SiLU(inplace=True),
        )

        self.pyramid_extractor = PyramidFeatureExtractorInvertedResidual(
            input_channels=preprocess_channels,
            lateral_channels_list=lateral_channels_list,
            out_dim=fpn_out_channels,
            # num_bottlenecks=num_bottlenecks,
            fusion_mode=fpn_mode,
        )

        if fpn_mode == "32x32":
            stride = (img_size // 2) // patch_size
            num_patches = ((img_size // 2) // stride) ** 2
            self.conv_bottleneck = nn.Sequential(
                nn.Conv2d(
                    fpn_out_channels,
                    self.embed_dim,
                    kernel_size=3,
                    padding=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(self.embed_dim),
                nn.SiLU(inplace=True),
            )

            self.vit = VisionTransformer(
                embed_dim=self.embed_dim,
                num_patches=num_patches,
                depth=depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                linear_attention=linear_attention,
                linear_layer_limit=linear_layer_limit,
                moe_num_shared=moe_num_shared,
                moe_num_routed=moe_num_routed,
                moe_num_activated_routed=moe_num_activated_routed,
                moe_expert_ratio=moe_expert_ratio,
                moe_balance_factor=moe_balance_factor,
            )
        else:
            self.scale_projectors = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(fpn_out_channels, embed_dim, 1),
                        nn.BatchNorm2d(embed_dim),
                        nn.SiLU(inplace=True),
                    )
                    for _ in range(3)
                ]
            )

            self.vit = MultiScaleVisionTransformer(
                embed_dim=self.embed_dim,
                num_scales=3,
                num_patches_per_scale=num_patches_per_scale,
                depth=depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                linear_attention=linear_attention,
                linear_layer_limit=linear_layer_limit,
                moe_num_shared=moe_num_shared,
                moe_num_routed=moe_num_routed,
                moe_num_activated_routed=moe_num_activated_routed,
                moe_expert_ratio=moe_expert_ratio,
                moe_balance_factor=moe_balance_factor,
            )
            self.conv_bottleneck = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(self.embed_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        self.apply(self._init_weights_layer)

    def _init_weights_layer(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.image_preprocess(x)

        features, _ = self.pyramid_extractor(x)

        if self.fpn_mode == "32x32":
            x = self.conv_bottleneck(features)
            x = x.flatten(2).transpose(1, 2)

            x, aux_loss, expert_freq, expert_prob = self.vit(x)
            return self.head(x[:, 0]), aux_loss, expert_freq, expert_prob
        else:
            p1, p2, p3 = features

            p1 = self.scale_projectors[0](p1)
            p2 = self.scale_projectors[1](p2)
            p3 = self.scale_projectors[2](p3)

            tokens1 = p1.flatten(2).transpose(1, 2)
            tokens2 = p2.flatten(2).transpose(1, 2)
            tokens3 = p3.flatten(2).transpose(1, 2)

            x, aux_loss, expert_freq, expert_prob = self.vit(
                [tokens1, tokens2, tokens3]
            )
            return self.head(x[:, 0]), aux_loss, expert_freq, expert_prob


class SiameseFPNMoEViT(BaseModel):
    """Siamese FPN-ViT with MoE for metric learning."""

    @property
    def model_type(self) -> str:
        return "siamese"

    @property
    def has_aux_loss(self) -> bool:
        return True

    @property
    def arch_type(self) -> str:
        return "moe"

    @classmethod
    def get_criterion(cls, margin: float = 1.0, **kwargs) -> nn.Module:
        from .siamese import TripletLoss

        return TripletLoss(margin=margin)

    @classmethod
    def get_metrics_tracker(cls, margin: float = 1.0, **kwargs) -> Any:
        from ..training.metrics import TripletMetricsTracker

        return TripletMetricsTracker(margin=margin)

    def __init__(
        self,
        img_size=64,
        preprocess_channels=32,
        fpn_out_channels=128,
        embed_dim=128,
        embedding_dim=256,
        patch_size=16,
        input_channels=3,
        num_classes=631,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        drop_rate=0.2,
        lateral_channels_list=None,
        num_bottlenecks=3,
        fpn_mode="32x32",
        linear_attention=True,
        linear_layer_limit=4,
        moe_num_shared=1,
        moe_num_routed=64,
        moe_num_activated_routed=6,
        moe_expert_ratio=0.25,
        moe_balance_factor=0.01,
        **kwargs,
    ):
        super().__init__(
            num_classes=num_classes, input_channels=input_channels, **kwargs
        )

        self.embed_dim = embed_dim
        self.embedding_dim = embedding_dim
        self.fpn_mode = fpn_mode
        self.moe_num_routed = moe_num_routed

        if lateral_channels_list is None:
            lateral_channels_list = [64, 128, 256]

        self.image_preprocess = nn.Sequential(
            nn.Conv2d(input_channels, preprocess_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(preprocess_channels),
            nn.SiLU(inplace=True),
        )

        self.pyramid_extractor = PyramidFeatureExtractorInvertedResidual(
            input_channels=preprocess_channels,
            lateral_channels_list=lateral_channels_list,
            out_dim=fpn_out_channels,
            # num_bottlenecks=num_bottlenecks,
            fusion_mode=fpn_mode,
        )

        if fpn_mode == "32x32":
            stride = (img_size // 2) // patch_size
            num_patches = ((img_size // 2) // stride) ** 2
            self.conv_bottleneck = nn.Sequential(
                nn.Conv2d(
                    fpn_out_channels,
                    self.embed_dim,
                    kernel_size=3,
                    padding=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(self.embed_dim),
                nn.SiLU(inplace=True),
            )

            self.vit = VisionTransformer(
                embed_dim=self.embed_dim,
                num_patches=num_patches,
                depth=depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                linear_attention=linear_attention,
                linear_layer_limit=linear_layer_limit,
                moe_num_shared=moe_num_shared,
                moe_num_routed=moe_num_routed,
                moe_num_activated_routed=moe_num_activated_routed,
                moe_expert_ratio=moe_expert_ratio,
                moe_balance_factor=moe_balance_factor,
            )
        else:
            self.scale_projectors = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(fpn_out_channels, embed_dim, 1),
                        nn.BatchNorm2d(embed_dim),
                        nn.SiLU(inplace=True),
                    )
                    for _ in range(3)
                ]
            )

            self.vit = MultiScaleVisionTransformer(
                embed_dim=self.embed_dim,
                num_scales=3,
                num_patches_per_scale=[1024, 256, 64],
                depth=depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                linear_attention=linear_attention,
                linear_layer_limit=linear_layer_limit,
                moe_num_shared=moe_num_shared,
                moe_num_routed=moe_num_routed,
                moe_num_activated_routed=moe_num_activated_routed,
                moe_expert_ratio=moe_expert_ratio,
                moe_balance_factor=moe_balance_factor,
            )
            self.conv_bottleneck = nn.Identity()

        self.projection = nn.Sequential(
            nn.Linear(self.embed_dim, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
        )

        self._init_weights()

    def _init_weights(self):
        self.apply(self._init_weights_layer)

    def _init_weights_layer(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.image_preprocess(x)

        features, _ = self.pyramid_extractor(x)

        if self.fpn_mode == "32x32":
            x = self.conv_bottleneck(features)
            x = x.flatten(2).transpose(1, 2)

            x, aux_loss, expert_freq, expert_prob = self.vit(x)
        else:
            p1, p2, p3 = features

            p1 = self.scale_projectors[0](p1)
            p2 = self.scale_projectors[1](p2)
            p3 = self.scale_projectors[2](p3)

            tokens1 = p1.flatten(2).transpose(1, 2)
            tokens2 = p2.flatten(2).transpose(1, 2)
            tokens3 = p3.flatten(2).transpose(1, 2)

            x, aux_loss, expert_freq, expert_prob = self.vit(
                [tokens1, tokens2, tokens3]
            )

        cls_token = x[:, 0]
        embedding = self.projection(cls_token)

        if self.training:
            embedding = F.normalize(embedding, p=2, dim=1)

        return embedding, aux_loss, expert_freq, expert_prob


class ModelVariant:
    """Model variant configurations for different parameter budgets."""

    TINY = {
        "preprocess_channels": 16,
        "fpn_out_channels": 48,
        "embed_dim": 96,
        "depth": 2,
        "num_heads": 4,
        "lateral_channels": [24, 48, 96],
        "mlp_ratio": 2.0,
        "num_bottlenecks": 2,
        "moe_num_shared": 1,
        "moe_num_routed": 4,
        "moe_num_activated_routed": 1,
        "moe_expert_ratio": 0.5,
        "moe_balance_factor": 0.01,
    }

    SMALL = {
        # ===== 保持 TINY 的计算骨架 =====
        "preprocess_channels": 16,
        "embed_dim": 128,
        "depth": 2,
        "num_heads": 2,  # ← 修复! head_dim=32
        "lateral_channels": [24, 48, 96],
        "num_bottlenecks": 2,
        # ===== MoE 杠杆：用参数换容量 =====
        "moe_num_shared": 1,  # 减少 shared 省计算
        "moe_num_routed": 16,  # ← 关键：从 4 → 48
        "moe_num_activated_routed": 2,
        "moe_expert_ratio": 0.5,
        "moe_balance_factor": 0.05,
        # ===== 减少 tokens 进一步省计算 =====
        "fpn_mode": "multiscale",
        # 去掉 32×32 scale，只保留 16×16 + 8×8
        "num_patches_per_scale": [256, 64],  # 320 tokens
    }

    BASE = {
        "preprocess_channels": 24,
        "fpn_out_channels": 112,
        "embed_dim": 128,
        "depth": 6,
        "num_heads": 8,
        "lateral_channels": [56, 112, 224],
        "mlp_ratio": 3.0,
        "num_bottlenecks": 3,
        "moe_num_shared": 2,
        "moe_num_routed": 24,
        "moe_num_activated_routed": 4,
        "moe_expert_ratio": 0.5,
        "moe_balance_factor": 0.1,
    }

    LARGE = {
        # ===== 保持 TINY 的计算骨架 =====
        "preprocess_channels": 32,
        "embed_dim": 192,
        "depth": 2,
        "num_heads": 2,  # ← 修复! head_dim=32
        "lateral_channels": [24, 48, 96],
        "num_bottlenecks": 2,
        # ===== MoE 杠杆：用参数换容量 =====
        "moe_num_shared": 1,  # 减少 shared 省计算
        "moe_num_routed": 48,  # ← 关键：从 4 → 48
        "moe_num_activated_routed": 2,
        "moe_expert_ratio": 0.5,
        "moe_balance_factor": 0.05,
        # ===== 减少 tokens 进一步省计算 =====
        "fpn_mode": "multiscale",
        # 去掉 32×32 scale，只保留 16×16 + 8×8
        "num_patches_per_scale": [256, 64],  # 320 tokens
    }


def create_fpn_moe_vit(variant="base", num_classes=631, **kwargs):
    """Factory function to create FPN-MoE-ViT with specified variant."""
    config = getattr(ModelVariant, variant.upper(), ModelVariant.BASE).copy()
    config.update(kwargs)

    return FeaturePyramidMoEViT(
        preprocess_channels=config["preprocess_channels"],
        fpn_out_channels=config["fpn_out_channels"],
        embed_dim=config["embed_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        mlp_ratio=config.get("mlp_ratio", 4.0),
        lateral_channels_list=config.get("lateral_channels"),
        num_bottlenecks=config.get("num_bottlenecks", 3),
        num_classes=num_classes,
        fpn_mode=config.get("fpn_mode", "multiscale"),
        num_patches_per_scale=config.get("num_patches_per_scale", None),
        moe_num_shared=config.get("moe_num_shared", 1),
        moe_num_routed=config.get("moe_num_routed", 8),
        moe_num_activated_routed=config.get("moe_num_activated_routed", 2),
        moe_expert_ratio=config.get("moe_expert_ratio", 0.25),
        moe_balance_factor=config.get("moe_balance_factor", 0.01),
    )


def create_siamese_fpn_moe_vit(variant="base", embedding_dim=256, **kwargs):
    """Factory function to create Siamese FPN-MoE-ViT with specified variant."""
    config = getattr(ModelVariant, variant.upper(), ModelVariant.BASE).copy()
    config.update(kwargs)

    return SiameseFPNMoEViT(
        preprocess_channels=config["preprocess_channels"],
        fpn_out_channels=config["fpn_out_channels"],
        embed_dim=config["embed_dim"],
        embedding_dim=embedding_dim,
        depth=config["depth"],
        num_heads=config["num_heads"],
        mlp_ratio=config.get("mlp_ratio", 4.0),
        lateral_channels_list=config.get("lateral_channels"),
        num_bottlenecks=config.get("num_bottlenecks", 3),
        num_classes=631,
        fpn_mode=config.get("fpn_mode", "32x32"),
        moe_num_shared=config.get("moe_num_shared", 1),
        moe_num_routed=config.get("moe_num_routed", 8),
        moe_num_activated_routed=config.get("moe_num_activated_routed", 2),
        moe_expert_ratio=config.get("moe_expert_ratio", 0.25),
        moe_balance_factor=config.get("moe_balance_factor", 0.01),
    )


class FeaturePyramidMoEViTTiny(FeaturePyramidMoEViT):
    """FPN-MoE-ViT Tiny variant."""

    def __init__(self, num_classes=631, **kwargs):
        config = ModelVariant.TINY.copy()
        config.update(kwargs)
        super().__init__(
            input_channels=config.get("input_channels", 3),
            preprocess_channels=config["preprocess_channels"],
            fpn_out_channels=config["fpn_out_channels"],
            embed_dim=config["embed_dim"],
            depth=config["depth"],
            num_heads=config["num_heads"],
            mlp_ratio=config.get("mlp_ratio", 4.0),
            lateral_channels_list=config.get("lateral_channels"),
            num_bottlenecks=config.get("num_bottlenecks", 3),
            num_classes=num_classes,
            fpn_mode=config.get("fpn_mode", "32x32"),
            moe_num_shared=config.get("moe_num_shared", 1),
            moe_num_routed=config.get("moe_num_routed", 4),
            moe_num_activated_routed=config.get("moe_num_activated_routed", 2),
            moe_expert_ratio=config.get("moe_expert_ratio", 0.25),
            moe_balance_factor=config.get("moe_balance_factor", 0.01),
        )


class FeaturePyramidMoEViTSmall(FeaturePyramidMoEViT):
    """FPN-MoE-ViT Small variant."""

    def __init__(self, num_classes=631, **kwargs):
        config = ModelVariant.SMALL.copy()
        config.update(kwargs)
        super().__init__(
            input_channels=config.get("input_channels", 3),
            preprocess_channels=config["preprocess_channels"],
            # fpn_out_channels=config["fpn_out_channels"],
            embed_dim=config["embed_dim"],
            depth=config["depth"],
            num_heads=config["num_heads"],
            mlp_ratio=config.get("mlp_ratio", 4.0),
            lateral_channels_list=config.get("lateral_channels"),
            num_bottlenecks=config.get("num_bottlenecks", 3),
            num_classes=num_classes,
            fpn_mode=config.get("fpn_mode", "32x32"),
            moe_num_shared=config.get("moe_num_shared", 1),
            moe_num_routed=config.get("moe_num_routed", 8),
            moe_num_activated_routed=config.get("moe_num_activated_routed", 2),
            moe_expert_ratio=config.get("moe_expert_ratio", 0.25),
            moe_balance_factor=config.get("moe_balance_factor", 0.01),
        )


class FeaturePyramidMoEViTLarge(FeaturePyramidMoEViT):
    """FPN-MoE-ViT Large variant."""

    def __init__(self, num_classes=631, **kwargs):
        config = ModelVariant.LARGE.copy()
        config.update(kwargs)
        super().__init__(
            input_channels=config.get("input_channels", 3),
            preprocess_channels=config["preprocess_channels"],
            fpn_out_channels=config["fpn_out_channels"],
            embed_dim=config["embed_dim"],
            depth=config["depth"],
            num_heads=config["num_heads"],
            mlp_ratio=config.get("mlp_ratio", 4.0),
            lateral_channels_list=config.get("lateral_channels"),
            num_bottlenecks=config.get("num_bottlenecks", 3),
            num_classes=num_classes,
            fpn_mode=config.get("fpn_mode", "32x32"),
            moe_num_shared=config.get("moe_num_shared", 2),
            moe_num_routed=config.get("moe_num_routed", 32),
            moe_num_activated_routed=config.get("moe_num_activated_routed", 6),
            moe_expert_ratio=config.get("moe_expert_ratio", 0.25),
            moe_balance_factor=config.get("moe_balance_factor", 0.01),
        )


class SiameseFPNMoEViTTiny(SiameseFPNMoEViT):
    """Siamese FPN-MoE-ViT Tiny variant."""

    def __init__(self, embedding_dim=256, num_classes=631, **kwargs):
        config = ModelVariant.TINY.copy()
        config.update(kwargs)
        super().__init__(
            input_channels=config.get("input_channels", 3),
            preprocess_channels=config["preprocess_channels"],
            fpn_out_channels=config["fpn_out_channels"],
            embed_dim=config["embed_dim"],
            embedding_dim=embedding_dim,
            depth=config["depth"],
            num_heads=config["num_heads"],
            mlp_ratio=config.get("mlp_ratio", 4.0),
            lateral_channels_list=config.get("lateral_channels"),
            num_bottlenecks=config.get("num_bottlenecks", 3),
            num_classes=num_classes,
            fpn_mode=config.get("fpn_mode", "32x32"),
            moe_num_shared=config.get("moe_num_shared", 1),
            moe_num_routed=config.get("moe_num_routed", 4),
            moe_num_activated_routed=config.get("moe_num_activated_routed", 2),
            moe_expert_ratio=config.get("moe_expert_ratio", 0.25),
            moe_balance_factor=config.get("moe_balance_factor", 0.01),
        )


class SiameseFPNMoEViTSmall(SiameseFPNMoEViT):
    """Siamese FPN-MoE-ViT Small variant."""

    def __init__(self, embedding_dim=256, num_classes=631, **kwargs):
        config = ModelVariant.SMALL.copy()
        config.update(kwargs)
        super().__init__(
            input_channels=config.get("input_channels", 3),
            preprocess_channels=config["preprocess_channels"],
            fpn_out_channels=config["fpn_out_channels"],
            embed_dim=config["embed_dim"],
            embedding_dim=embedding_dim,
            depth=config["depth"],
            num_heads=config["num_heads"],
            mlp_ratio=config.get("mlp_ratio", 4.0),
            lateral_channels_list=config.get("lateral_channels"),
            num_bottlenecks=config.get("num_bottlenecks", 3),
            num_classes=num_classes,
            fpn_mode=config.get("fpn_mode", "32x32"),
            moe_num_shared=config.get("moe_num_shared", 1),
            moe_num_routed=config.get("moe_num_routed", 8),
            moe_num_activated_routed=config.get("moe_num_activated_routed", 2),
            moe_expert_ratio=config.get("moe_expert_ratio", 0.25),
            moe_balance_factor=config.get("moe_balance_factor", 0.01),
        )


class SiameseFPNMoEViTLarge(SiameseFPNMoEViT):
    """Siamese FPN-MoE-ViT Large variant."""

    def __init__(self, embedding_dim=256, num_classes=631, **kwargs):
        config = ModelVariant.LARGE.copy()
        config.update(kwargs)
        super().__init__(
            input_channels=config.get("input_channels", 3),
            preprocess_channels=config["preprocess_channels"],
            fpn_out_channels=config["fpn_out_channels"],
            embed_dim=config["embed_dim"],
            embedding_dim=embedding_dim,
            depth=config["depth"],
            num_heads=config["num_heads"],
            mlp_ratio=config.get("mlp_ratio", 4.0),
            lateral_channels_list=config.get("lateral_channels"),
            num_bottlenecks=config.get("num_bottlenecks", 3),
            num_classes=num_classes,
            fpn_mode=config.get("fpn_mode", "32x32"),
            moe_num_shared=config.get("moe_num_shared", 2),
            moe_num_routed=config.get("moe_num_routed", 32),
            moe_num_activated_routed=config.get("moe_num_activated_routed", 6),
            moe_expert_ratio=config.get("moe_expert_ratio", 0.25),
            moe_balance_factor=config.get("moe_balance_factor", 0.01),
        )


if __name__ == "__main__":
    from torchinfo import summary

    print("=" * 80)
    print("Testing FPN-MoE-ViT:")
    print("=" * 80)
    model = FeaturePyramidMoEViT(moe_num_routed=8, moe_num_activated_routed=2)
    summary(
        model,
        input_size=(1, 3, 64, 64),
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
    )

    print("\n" + "=" * 80)
    print("Testing SiameseFPNMoEViT:")
    print("=" * 80)
    # model = SiameseFPNMoEViT(moe_num_routed=8, moe_num_activated_routed=2)
    # summary(
    #     model,
    #     input_size=(1, 3, 64, 64),
    #     col_names=["input_size", "output_size", "num_params", "mult_adds"],
    # )

    # print("\n" + "=" * 80)
    # print("Testing forward pass (returns aux_loss):")
    # print("=" * 80)
    # model = FeaturePyramidMoEViT(moe_num_routed=8, moe_num_activated_routed=2)
    # x = torch.randn(2, 3, 64, 64)
    # output, aux_loss, expert_freq, expert_prob = model(x)
    # print(f"Output shape: {output.shape}")
    # print(f"Aux loss: {aux_loss.item():.6f}")
    # print(f"Expert freq: {expert_freq}")
    # print(f"Expert prob: {expert_prob}")
