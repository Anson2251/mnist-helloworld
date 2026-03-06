import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any

MetricsTracker = None
try:
    from .base import BaseModel
    from ..training.metrics import MetricsTracker
except ImportError:
    from base import BaseModel


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
    def __init__(self, dim, num_shared=1, num_routed=64,
                 num_activated_routed=6, expert_ratio=0.5,
                 act_layer=nn.GELU, drop=0.0, balance_factor=0.01):
        super().__init__()
        self.num_shared = num_shared
        self.num_routed = num_routed
        self.num_activated_routed = num_activated_routed
        self.balance_factor = balance_factor

        expert_hidden = int(dim * 4 * expert_ratio)

        self.shared_experts = nn.ModuleList(
            [Expert(dim, expert_hidden, act_layer, drop) for _ in range(num_shared)]
        )

        # ---- KEY CHANGE: Stacked expert weights for parallel computation ----
        self.expert_fc1_weight = nn.Parameter(
            torch.empty(num_routed, expert_hidden, dim))
        self.expert_fc1_bias = nn.Parameter(torch.zeros(num_routed, expert_hidden))
        self.expert_fc2_weight = nn.Parameter(
            torch.empty(num_routed, dim, expert_hidden))
        self.expert_fc2_bias = nn.Parameter(torch.zeros(num_routed, dim))

        nn.init.kaiming_uniform_(self.expert_fc1_weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.expert_fc2_weight, a=5**0.5)

        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.gate = nn.Linear(dim, num_routed, bias=False)

    def forward(self, x):
        B, N, dim = x.shape
        T = B * N  # total tokens

        # Shared experts (unchanged)
        shared_out = sum(expert(x) for expert in self.shared_experts)

        # Gating
        gate_logits = self.gate(x)
        gate_scores = F.softmax(gate_logits, dim=-1)  # (B, N, E)
        top_scores, top_indices = torch.topk(
            gate_scores, self.num_activated_routed, dim=-1
        )  # (B, N, K)

        # ---- PARALLEL DISPATCH (no for-loop) ----
        flat_x = x.reshape(T, dim)                          # (T, D)
        flat_indices = top_indices.reshape(T, -1)            # (T, K)
        flat_scores = top_scores.reshape(T, -1)              # (T, K)

        # Gather expert weights for each (token, top-k) pair
        # expert_ids: (T*K,)
        expert_ids = flat_indices.reshape(-1)
        # Repeat each token K times
        token_x = flat_x.unsqueeze(1).expand(-1, self.num_activated_routed, -1)
        token_x = token_x.reshape(T * self.num_activated_routed, dim)  # (T*K, D)

        # Batched expert forward: gather per-expert weights
        w1 = self.expert_fc1_weight[expert_ids]   # (T*K, H, D)
        b1 = self.expert_fc1_bias[expert_ids]      # (T*K, H)
        w2 = self.expert_fc2_weight[expert_ids]    # (T*K, D, H)
        b2 = self.expert_fc2_bias[expert_ids]      # (T*K, D)

        # Forward: two linear layers with activation
        h = torch.bmm(w1, token_x.unsqueeze(-1)).squeeze(-1) + b1  # (T*K, H)
        h = self.act(h)
        h = self.drop(h)
        out = torch.bmm(w2, h.unsqueeze(-1)).squeeze(-1) + b2      # (T*K, D)
        out = self.drop(out)

        # Weight by gate scores and scatter-add back
        scores = flat_scores.reshape(-1, 1)        # (T*K, 1)
        out = out * scores                          # (T*K, D)

        # Scatter back to token positions
        token_indices = torch.arange(T, device=x.device).unsqueeze(1)
        token_indices = token_indices.expand(-1, self.num_activated_routed)
        token_indices = token_indices.reshape(-1)   # (T*K,)

        routed_out = torch.zeros(T, dim, device=x.device)
        routed_out.scatter_add_(0, token_indices.unsqueeze(-1).expand(-1, dim), out)
        routed_out = routed_out.reshape(B, N, dim)

        # Balance loss (vectorized)
        expert_freq = torch.zeros(self.num_routed, device=x.device)
        expert_freq.scatter_add_(0, expert_ids, torch.ones_like(expert_ids, dtype=torch.float))
        expert_freq = expert_freq / (T * self.num_activated_routed)

        expert_prob = gate_scores.reshape(T, -1).mean(dim=0)  # (E,)

        balance_loss = self.balance_factor * (expert_freq * expert_prob).sum()

        return shared_out + routed_out, balance_loss, expert_freq, expert_prob


class PatchEmbed(nn.Module):
    """将 2D 图片展平为 Patch Embeddings"""

    def __init__(self, img_size=64, patch_size=8, in_chans=3, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size**2
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LinearAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.kernel_fn = nn.ELU()

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.kernel_fn(q) + 1
        k = self.kernel_fn(k) + 1

        q = q * self.scale
        kv = k.transpose(-2, -1) @ v
        x = q @ kv

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FocusedLinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        focusing_factor=3,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.focusing_factor = focusing_factor

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.dwc = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        q = F.relu(q) + 1e-6
        k = F.relu(k) + 1e-6

        q = q / q.sum(dim=-1, keepdim=True)
        k = k / k.sum(dim=-1, keepdim=True)
        q = q**self.focusing_factor
        k = k**self.focusing_factor
        q = q / q.sum(dim=-1, keepdim=True)
        k = k / k.sum(dim=-1, keepdim=True)

        kv = torch.einsum("bhnd,bhne->bhde", k, v)
        z = 1.0 / (torch.einsum("bhnd,bhd->bhn", q, k.sum(dim=2)) + 1e-6)
        out = torch.einsum("bhnd,bhde,bhn->bhne", q, kv, z)

        out = out.transpose(1, 2).reshape(B, N, C)
        out = out + self.dwc(out.transpose(1, 2)).transpose(1, 2)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def drop_path(x: torch.Tensor, keep_prob: float = 1.0, inplace: bool = False) -> torch.Tensor:
    mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    # remember tuples have the * operator -> (1,) * 3 = (1,1,1)
    mask: torch.Tensor = x.new_empty(mask_shape).bernoulli_(keep_prob)
    mask.div_(keep_prob)
    if inplace:
        x.mul_(mask)
    else:
        x = x * mask
    return x

class DropPath(nn.Module):
    """
    DropPath (Stochastic Depth) regularization layer.
    During training, randomly drops entire residual paths with probability `p`.
     - `p=0` means no dropping (always keep).
     - `p=1` means always drop (never keep).
    The remaining paths are scaled by `1/(1-p)` to maintain expected feature magnitudes.
    This encourages the model to learn more robust representations and can improve generalization.
    Reference: https://arxiv.org/abs/1603.09382

    Implementation from: https://github.com/FrancescoSaverioZuppichini/DropPath
    """
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.p > 0:
            x = drop_path(x, self.p, self.inplace)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"


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
        drop_path=0.1
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


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ConvBlock(nn.Module):
    """Standard Conv Block (Conv + BatchNorm + Activation)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        groups=1,
        act=True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """YOLOv5 Bottleneck block with optional shortcut."""

    def __init__(
        self, in_channels, out_channels, shortcut=True, groups=1, expansion=0.5
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = ConvBlock(in_channels, hidden_channels, 1, 1)
        self.cv2 = ConvBlock(hidden_channels, out_channels, 3, 1, 1, groups=groups)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3Module(nn.Module):
    """YOLOv5 C3 Module (Cross Stage Partial Network)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        num_bottlenecks=3,
        shortcut=True,
        groups=1,
        expansion=0.5,
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.cv1 = ConvBlock(in_channels, hidden_channels, 1, 1)
        self.cv2 = ConvBlock(in_channels, hidden_channels, 1, 1)
        self.cv3 = ConvBlock(2 * hidden_channels, out_channels, 1, 1)
        self.m = nn.Sequential(
            *[
                Bottleneck(
                    hidden_channels, hidden_channels, shortcut, groups, expansion
                )
                for _ in range(num_bottlenecks)
            ]
        )

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


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

        self.pyramid_extractor = PyramidFeatureExtractor(
            input_channels=preprocess_channels,
            lateral_channels_list=lateral_channels_list,
            out_dim=fpn_out_channels,
            num_bottlenecks=num_bottlenecks,
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

        features = self.pyramid_extractor(x)

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

        self.pyramid_extractor = PyramidFeatureExtractor(
            input_channels=preprocess_channels,
            lateral_channels_list=lateral_channels_list,
            out_dim=fpn_out_channels,
            num_bottlenecks=num_bottlenecks,
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

        features = self.pyramid_extractor(x)

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
        "embed_dim": 64,
        "depth": 2,
        "num_heads": 4,
        "lateral_channels": [24, 48, 96],
        "mlp_ratio": 2.0,
        "num_bottlenecks": 2,
        "moe_num_shared": 2,
        "moe_num_routed": 4,
        "moe_num_activated_routed": 2,
        "moe_expert_ratio": 0.5,
        "moe_balance_factor": 0.01,
    }

    SMALL = {
        "preprocess_channels": 32,
        "fpn_out_channels": 112,
        "embed_dim": 128,
        "depth": 5,
        "num_heads": 8,
        "lateral_channels": [56, 112, 224],
        "mlp_ratio": 3.0,
        "num_bottlenecks": 2,
        "moe_num_shared": 2,
        "moe_num_routed": 8,
        "moe_num_activated_routed": 2,
        "moe_expert_ratio": 0.5,
        "moe_balance_factor": 0.1,
    }

    BASE = {
        "preprocess_channels": 32,
        "fpn_out_channels": 112,
        "embed_dim": 128,
        "depth": 6,
        "num_heads": 8,
        "lateral_channels": [56, 112, 224],
        "mlp_ratio": 3.0,
        "num_bottlenecks": 3,
        "moe_num_shared": 2,
        "moe_num_routed": 16,
        "moe_num_activated_routed": 4,
        "moe_expert_ratio": 0.5,
        "moe_balance_factor": 0.1,
    }

    LARGE = {
        "preprocess_channels": 40,
        "fpn_out_channels": 144,
        "embed_dim": 160,
        "depth": 8,
        "num_heads": 8,
        "lateral_channels": [72, 144, 288],
        "mlp_ratio": 3.0,
        "num_bottlenecks": 3,
        "moe_num_shared": 4,
        "moe_num_routed": 32,
        "moe_num_activated_routed": 6,
        "moe_expert_ratio": 0.5,
        "moe_balance_factor": 0.1,
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
        fpn_mode=config.get("fpn_mode", "32x32"),
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
    summary(model, input_size=(1, 3, 64, 64))

    print("\n" + "=" * 80)
    print("Testing SiameseFPNMoEViT:")
    print("=" * 80)
    model = SiameseFPNMoEViT(moe_num_routed=8, moe_num_activated_routed=2)
    summary(model, input_size=(1, 3, 64, 64))

    print("\n" + "=" * 80)
    print("Testing forward pass (returns aux_loss):")
    print("=" * 80)
    model = FeaturePyramidMoEViT(moe_num_routed=8, moe_num_activated_routed=2)
    x = torch.randn(2, 3, 64, 64)
    output, aux_loss, expert_freq, expert_prob = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Aux loss: {aux_loss.item():.6f}")
    print(f"Expert freq: {expert_freq}")
    print(f"Expert prob: {expert_prob}")
