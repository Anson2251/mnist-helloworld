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
        # kernel function：elu+1（确保非负）
        self.kernel_fn = nn.ELU()

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, heads, N, head_dim]

        q = self.kernel_fn(q) + 1
        k = self.kernel_fn(k) + 1

        # (Q @ (K^T @ V)) / scaling
        q = q * self.scale
        # K^T @ V  [B, heads, head_dim, head_dim]
        kv = k.transpose(-2, -1) @ v
        # Q @ KV  [B, heads, N, head_dim]
        x = q @ kv

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FocusedLinearAttention(nn.Module):
    """
    Focused Linear Attention (ICCV 2023)
    核心改进：加入 focusing factor 让注意力分布更尖锐
    """

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

        # learnable focusing param
        self.dwc = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # (B, heads, N, d)

        # 1. ReLU kernel mapping (ensure non-negative)
        q = F.relu(q) + 1e-6
        k = F.relu(k) + 1e-6

        # 2. L2 norm + focusing factor (make distribution sharper)
        q = q / q.sum(dim=-1, keepdim=True)  # normalize along feature dim
        k = k / k.sum(dim=-1, keepdim=True)
        q = q**self.focusing_factor  # power operation to make distribution sharper
        k = k**self.focusing_factor
        q = q / q.sum(dim=-1, keepdim=True)  # re-normalize
        k = k / k.sum(dim=-1, keepdim=True)

        # 3. linear attention: first calc K^T @ V, complexity O(Nd^2)
        kv = torch.einsum("bhnd,bhne->bhde", k, v)  # (B, h, d, d)
        z = 1.0 / (
            torch.einsum("bhnd,bhd->bhn", q, k.sum(dim=2)) + 1e-6
        )  # normalization factor
        out = torch.einsum("bhnd,bhde,bhn->bhne", q, kv, z)  # (B, h, N, d)

        # 4. Local feature enhancement (compensate for lost local info in linear attention)
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
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
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
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


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
    """YOLOv5 C3 Module (Cross Stage Partial Network).

    Args:
        in_channels: Input channel count
        out_channels: Output channel count
        num_bottlenecks: Number of bottleneck blocks
        shortcut: Whether to use shortcut connections in bottlenecks
        groups: Number of groups for grouped convolutions
        expansion: Channel expansion factor for bottleneck hidden layers
    """

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
        x = self.blocks(x)
        x = self.norm(x)
        return x


class MultiScaleVisionTransformer(nn.Module):
    """Vision Transformer with multi-scale token input and scale position encoding.

    Accepts tokens from multiple scales (e.g., 32x32, 16x16, 8x8), adds scale-specific
    position encoding, and processes them together.
    """

    def __init__(
        self,
        embed_dim=256,
        num_scales=3,
        num_patches_per_scale=[1024, 256, 64],  # 32x32, 16x16, 8x8
        depth=8,
        num_heads=8,
        mlp_ratio=4.0,
        drop_rate=0.2,
        linear_attention=False,
        linear_layer_limit=4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_scales = num_scales
        self.num_patches_per_scale = num_patches_per_scale

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Spatial position embeddings for each scale
        self.pos_embeds = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
                for num_patches in num_patches_per_scale
            ]
        )

        # Scale embeddings to distinguish tokens from different scales
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
        """
        Args:
            tokens_list: List of 3 tensors [tokens_scale1, tokens_scale2, tokens_scale3]
                Each tensor: (B, N_i, embed_dim) where N_i is num_patches for that scale
        """
        B = tokens_list[0].shape[0]

        # Add spatial position encoding and scale encoding to each scale
        scale_tokens = []
        for i, (tokens, pos_embed) in enumerate(zip(tokens_list, self.pos_embeds)):
            # Add spatial position embedding (skip CLS position)
            tokens = tokens + pos_embed[:, 1:, :]
            # Add scale embedding
            scale_embed = self.scale_embeds[:, i : i + 1, :].expand(
                B, tokens.shape[1], -1
            )
            tokens = tokens + scale_embed
            scale_tokens.append(tokens)

        # Concatenate all scale tokens
        x = torch.cat(scale_tokens, dim=1)  # (B, total_patches, embed_dim)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x


class PyramidFeatureExtractor(nn.Module):
    """Concatenation-based Pyramid Feature Extractor with YOLOv5 C3 modules.

    Supports two modes:
    1. fusion_mode='32x32': Align all scales to 32x32 and fuse (1024 tokens)
    2. fusion_mode='multiscale': Keep original resolutions, return multi-scale tokens
    """

    def __init__(
        self,
        input_channels=3,
        lateral_channels_list=[64, 128, 256],
        out_dim=256,
        num_bottlenecks=3,
        fusion_mode="32x32",  # '32x32' or 'multiscale'
    ):
        super().__init__()
        self.fusion_mode = fusion_mode

        # Stage 1: Conv + C3 (downsample from 64x64 to 32x32)
        self.stem = ConvBlock(
            input_channels, lateral_channels_list[0], 3, 2, 1
        )  # 64x64 -> 32x32
        self.layer1 = C3Module(
            lateral_channels_list[0],
            lateral_channels_list[0],
            num_bottlenecks=num_bottlenecks,
            shortcut=True,
        )

        # Stage 2: Conv + C3 (downsample from 32x32 to 16x16)
        self.down2 = ConvBlock(
            lateral_channels_list[0], lateral_channels_list[1], 3, 2, 1
        )  # 32x32 -> 16x16
        self.layer2 = C3Module(
            lateral_channels_list[1],
            lateral_channels_list[1],
            num_bottlenecks=num_bottlenecks,
            shortcut=True,
        )

        # Stage 3: Conv + C3 (downsample from 16x16 to 8x8)
        self.down3 = ConvBlock(
            lateral_channels_list[1], lateral_channels_list[2], 3, 2, 1
        )  # 16x16 -> 8x8
        self.layer3 = C3Module(
            lateral_channels_list[2],
            lateral_channels_list[2],
            num_bottlenecks=num_bottlenecks,
            shortcut=True,
        )

        if fusion_mode == "32x32":
            # Mode 1: Align all scales to 32x32
            self.lateral1 = nn.Conv2d(
                lateral_channels_list[0], out_dim, 1
            )  # 32x32 stays
            self.lateral2 = nn.Sequential(
                nn.Conv2d(lateral_channels_list[1], out_dim, 1),
                nn.Upsample(
                    scale_factor=2, mode="bilinear", align_corners=False
                ),  # 16x16 -> 32x32
            )
            self.lateral3 = nn.Sequential(
                nn.Conv2d(lateral_channels_list[2], out_dim, 1),
                nn.Upsample(
                    scale_factor=4, mode="bilinear", align_corners=False
                ),  # 8x8 -> 32x32
            )
        else:  # multiscale
            # Mode 2: Keep original resolutions
            self.lateral1 = nn.Conv2d(lateral_channels_list[0], out_dim, 1)  # 32x32
            self.lateral2 = nn.Conv2d(lateral_channels_list[1], out_dim, 1)  # 16x16
            self.lateral3 = nn.Conv2d(lateral_channels_list[2], out_dim, 1)  # 8x8

        # Scale-specific SE (optional shared weights)
        self.se1 = SEBlock(out_dim)
        self.se2 = SEBlock(out_dim)
        self.se3 = SEBlock(out_dim)

        if fusion_mode == "32x32":
            # Fusion conv: reduce channels after concatenation
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(out_dim * 3, out_dim, 1),
                nn.BatchNorm2d(out_dim),
                nn.SiLU(inplace=True),
            )

    def forward(self, x):
        # Bottom-up pathway with C3 modules
        x = self.stem(x)  # 32x32, 64ch
        c1 = self.layer1(x)  # 32x32, 64ch
        x = self.down2(c1)  # 16x16, 128ch
        c2 = self.layer2(x)  # 16x16, 128ch
        x = self.down3(c2)  # 8x8, 256ch
        c3 = self.layer3(x)  # 8x8, 256ch

        # 1. Unify channels
        p1 = self.lateral1(c1)  # 32x32, out_dim
        p2 = self.lateral2(c2)  # 16x16 or 32x32, out_dim
        p3 = self.lateral3(c3)  # 8x8 or 32x32, out_dim

        # 2. Apply SE independently
        p1 = self.se1(p1)
        p2 = self.se2(p2)
        p3 = self.se3(p3)

        if self.fusion_mode == "32x32":
            # 3. Concatenate and fuse at 32x32 (1024 tokens after flattening)
            fused = torch.cat([p1, p2, p3], dim=1)  # (B, out_dim*3, 32, 32)
            out = self.fusion_conv(fused)  # (B, out_dim, 32, 32)
            return out
        else:
            # 3. Return multi-scale features without fusion
            # p1: (B, out_dim, 32, 32), p2: (B, out_dim, 16, 16), p3: (B, out_dim, 8, 8)
            return p1, p2, p3


class FeaturePyramidViT(BaseModel):
    """Vision Transformer with Bottleneck for Chinese Character Recognition.

    Structure: Conv feature extraction -> Conv bottleneck -> ViT blocks -> 2x FC classification
    """

    @property
    def model_type(self) -> str:
        return "classification"

    @classmethod
    def get_criterion(cls, **kwargs) -> nn.Module:
        """Return CrossEntropyLoss for classification."""
        return nn.CrossEntropyLoss()

    @classmethod
    def get_metrics_tracker(cls, **kwargs) -> Any:
        """Return standard metrics tracker for classification."""
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
        fpn_mode="multiscale",  # '32x32' or 'multiscale'
        **kwargs,
    ):
        super().__init__(
            num_classes=num_classes, input_channels=input_channels, **kwargs
        )

        self.embed_dim = embed_dim
        self.fpn_mode = fpn_mode
        self.input_size = img_size

        if lateral_channels_list is None:
            lateral_channels_list = [64, 128, 256]

        self.image_preprocess = nn.Sequential(
            nn.Conv2d(input_channels, preprocess_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(preprocess_channels),
            nn.SiLU(inplace=True),
        )

        # Stage 1: Pyramid Feature Extractor & Channel Reduction
        self.pyramid_extractor = PyramidFeatureExtractor(
            input_channels=preprocess_channels,
            lateral_channels_list=lateral_channels_list,
            out_dim=fpn_out_channels,
            num_bottlenecks=num_bottlenecks,
            fusion_mode=fpn_mode,
        )

        if fpn_mode == "32x32":
            # Stage 2: 1024 patches for ViT (32x32)
            stride = (img_size // 2) // patch_size  # 32 / patch_size
            num_patches = ((img_size // 2) // stride) ** 2  # 32x32 = 1024
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

            # Stage 3: Vision Transformer with 1024 patches
            self.vit = VisionTransformer(
                embed_dim=self.embed_dim,
                num_patches=num_patches,
                depth=depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                linear_attention=True,
                linear_layer_limit=4,
            )
        else:  # multiscale
            # Stage 2: Project each scale to embed_dim
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

            # Stage 3: Multi-scale Vision Transformer
            # Tokens: 32x32=1024, 16x16=256, 8x8=64, total=1344
            self.vit = MultiScaleVisionTransformer(
                embed_dim=self.embed_dim,
                num_scales=3,
                num_patches_per_scale=[1024, 256, 64],
                depth=depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                linear_attention=True,
                linear_layer_limit=4,
            )
            # Placeholder for conv_bottleneck in multiscale mode
            self.conv_bottleneck = nn.Identity()

        # Stage 5: Classification Head (2x FC with more capacity)
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

        # Stage 1: FPN Feature Extraction
        features = self.pyramid_extractor(x)

        if self.fpn_mode == "32x32":
            # Stage 2: Conv Bottleneck
            x = self.conv_bottleneck(features)

            # Stage 3: Reshape to patches for ViT (B, C, H, W) -> (B, H*W, C)
            x = x.flatten(2).transpose(1, 2)

            # Stage 4: ViT Processing
            x = self.vit(x)
        else:  # multiscale
            # features is a tuple of (p1, p2, p3)
            p1, p2, p3 = features

            # Stage 2: Project each scale to embed_dim
            p1 = self.scale_projectors[0](p1)  # (B, embed_dim, 32, 32)
            p2 = self.scale_projectors[1](p2)  # (B, embed_dim, 16, 16)
            p3 = self.scale_projectors[2](p3)  # (B, embed_dim, 8, 8)

            # Stage 3: Flatten to tokens
            tokens1 = p1.flatten(2).transpose(1, 2)  # (B, 1024, embed_dim)
            tokens2 = p2.flatten(2).transpose(1, 2)  # (B, 256, embed_dim)
            tokens3 = p3.flatten(2).transpose(1, 2)  # (B, 64, embed_dim)

            # Stage 4: Multi-scale ViT Processing
            x = self.vit([tokens1, tokens2, tokens3])

        # Stage 5: Classification
        return self.head(x[:, 0])


class SiameseFPNViT(BaseModel):
    """
    Siamese FPN-ViT for metric learning with triplet loss.

    Same architecture as FeaturePyramidViT but without classification head.
    Returns L2-normalized embeddings for metric learning.
    """

    @property
    def model_type(self) -> str:
        return "siamese"

    @classmethod
    def get_criterion(cls, margin: float = 1.0, **kwargs) -> nn.Module:
        """Return TripletLoss for siamese network."""
        from .siamese import TripletLoss

        return TripletLoss(margin=margin)

    @classmethod
    def get_metrics_tracker(cls, margin: float = 1.0, **kwargs) -> Any:
        """Return triplet metrics tracker."""
        from ..training.metrics import TripletMetricsTracker

        return TripletMetricsTracker(margin=margin)

    def __init__(
        self,
        img_size=64,
        preprocess_channels=32,
        fpn_out_channels=128,
        embed_dim=128,
        embedding_dim=256,  # Final embedding dimension (can differ from embed_dim)
        patch_size=16,
        input_channels=3,
        num_classes=631,  # Kept for compatibility but not used
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        drop_rate=0.2,
        lateral_channels_list=None,
        num_bottlenecks=3,
        fpn_mode="32x32",  # '32x32' or 'multiscale'
        **kwargs,
    ):
        super().__init__(
            num_classes=num_classes, input_channels=input_channels, **kwargs
        )

        self.embed_dim = embed_dim
        self.embedding_dim = embedding_dim
        self.fpn_mode = fpn_mode

        if lateral_channels_list is None:
            lateral_channels_list = [64, 128, 256]

        self.image_preprocess = nn.Sequential(
            nn.Conv2d(input_channels, preprocess_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(preprocess_channels),
            nn.SiLU(inplace=True),
        )

        # Stage 1: Pyramid Feature Extractor & Channel Reduction
        self.pyramid_extractor = PyramidFeatureExtractor(
            input_channels=preprocess_channels,
            lateral_channels_list=lateral_channels_list,
            out_dim=fpn_out_channels,
            num_bottlenecks=num_bottlenecks,
            fusion_mode=fpn_mode,
        )

        if fpn_mode == "32x32":
            # Stage 2: 1024 patches for ViT (32x32)
            stride = (img_size // 2) // patch_size  # 32 / patch_size
            num_patches = ((img_size // 2) // stride) ** 2  # 32x32 = 1024
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

            # Stage 3: Vision Transformer
            self.vit = VisionTransformer(
                embed_dim=self.embed_dim,
                num_patches=num_patches,
                depth=depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                linear_attention=True,
                linear_layer_limit=4,
            )
        else:  # multiscale
            # Stage 2: Project each scale to embed_dim
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

            # Stage 3: Multi-scale Vision Transformer
            self.vit = MultiScaleVisionTransformer(
                embed_dim=self.embed_dim,
                num_scales=3,
                num_patches_per_scale=[1024, 256, 64],
                depth=depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                linear_attention=True,
                linear_layer_limit=4,
            )
            self.conv_bottleneck = nn.Identity()

        # Stage 4: Projection head - decouples embedding_dim from embed_dim
        # Deeper head without dropout for stable metric learning
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

        # Stage 1: FPN Feature Extraction
        features = self.pyramid_extractor(x)

        if self.fpn_mode == "32x32":
            # Stage 2: Conv Bottleneck
            x = self.conv_bottleneck(features)

            # Stage 3: Reshape to patches for ViT
            x = x.flatten(2).transpose(1, 2)

            # Stage 4: ViT Processing
            x = self.vit(x)
        else:  # multiscale
            # features is a tuple of (p1, p2, p3)
            p1, p2, p3 = features

            # Stage 2: Project each scale to embed_dim
            p1 = self.scale_projectors[0](p1)  # (B, embed_dim, 32, 32)
            p2 = self.scale_projectors[1](p2)  # (B, embed_dim, 16, 16)
            p3 = self.scale_projectors[2](p3)  # (B, embed_dim, 8, 8)

            # Stage 3: Flatten to tokens
            tokens1 = p1.flatten(2).transpose(1, 2)  # (B, 1024, embed_dim)
            tokens2 = p2.flatten(2).transpose(1, 2)  # (B, 256, embed_dim)
            tokens3 = p3.flatten(2).transpose(1, 2)  # (B, 64, embed_dim)

            # Stage 4: Multi-scale ViT Processing
            x = self.vit([tokens1, tokens2, tokens3])

        # Stage 5: Projection head
        # Extract CLS token and project to embedding dimension
        cls_token = x[:, 0]
        embedding = self.projection(cls_token)

        # L2 normalize during training
        if self.training:
            embedding = F.normalize(embedding, p=2, dim=1)

        return embedding


class ModelVariant:
    """Model variant configurations for different parameter budgets."""

    # Tiny: 0.5M - 1M params
    TINY = {
        "preprocess_channels": 16,
        "fpn_out_channels": 48,
        "embed_dim": 64,
        "depth": 2,
        "num_heads": 4,
        "lateral_channels": [24, 48, 96],
        "mlp_ratio": 2.0,
        "num_bottlenecks": 2,
    }

    # Small: 1.7M - 2.5M params
    SMALL = {
        "preprocess_channels": 32,
        "fpn_out_channels": 112,
        "embed_dim": 128,
        "depth": 5,
        "num_heads": 8,
        "lateral_channels": [56, 112, 224],
        "mlp_ratio": 3.0,
        "num_bottlenecks": 2,
    }

    # Base: ~2.5M params (current default)
    BASE = {
        "preprocess_channels": 32,
        "fpn_out_channels": 112,
        "embed_dim": 128,
        "depth": 6,
        "num_heads": 8,
        "lateral_channels": [56, 112, 224],
        "mlp_ratio": 3.0,
        "num_bottlenecks": 3,
    }

    # Large: 3.5M+ params
    LARGE = {
        "preprocess_channels": 40,
        "fpn_out_channels": 144,
        "embed_dim": 160,
        "depth": 8,
        "num_heads": 8,
        "lateral_channels": [72, 144, 288],
        "mlp_ratio": 3.0,
        "num_bottlenecks": 3,
    }


def create_fpn_vit(variant="base", num_classes=631, **kwargs):
    """Factory function to create FPN-ViT with specified variant.

    Args:
        variant: 'tiny', 'small', 'base', or 'large'
        num_classes: number of output classes
        **kwargs: override any config parameters
    """
    config = getattr(ModelVariant, variant.upper(), ModelVariant.BASE).copy()
    config.update(kwargs)

    return FeaturePyramidViT(
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
    )


def create_siamese_fpn_vit(variant="base", embedding_dim=256, **kwargs):
    """Factory function to create Siamese FPN-ViT with specified variant.

    Args:
        variant: 'tiny', 'small', 'base', or 'large'
        embedding_dim: final embedding dimension
        **kwargs: override any config parameters
    """
    config = getattr(ModelVariant, variant.upper(), ModelVariant.BASE).copy()
    config.update(kwargs)

    return SiameseFPNViT(
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
        fpn_mode=config.get("fpn_mode", "32x32"),
    )


# Model variant wrapper classes for registry
class FeaturePyramidViTTiny(FeaturePyramidViT):
    """FPN-ViT Tiny variant (~0.9M params)."""

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
        )


class FeaturePyramidViTSmall(FeaturePyramidViT):
    """FPN-ViT Small variant (~1.7M params)."""

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
        )


class FeaturePyramidViTLarge(FeaturePyramidViT):
    """FPN-ViT Large variant (~3.8M params)."""

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
        )


class SiameseFPNViTTiny(SiameseFPNViT):
    """Siamese FPN-ViT Tiny variant (~1.0M params)."""

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
        )


class SiameseFPNViTSmall(SiameseFPNViT):
    """Siamese FPN-ViT Small variant (~1.7M params)."""

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
        )


class SiameseFPNViTLarge(SiameseFPNViT):
    """Siamese FPN-ViT Large variant (~3.8M params)."""

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
        )


if __name__ == "__main__":
    from torchinfo import summary

    print("=" * 80)
    print("Testing 32x32 fusion mode (1024 tokens):")
    print("=" * 80)
    model = FeaturePyramidViTTiny(fpn_mode="32x32")
    summary(model, input_size=(1, 3, 64, 64))
    model = FeaturePyramidViTSmall(fpn_mode="32x32")
    summary(model, input_size=(1, 3, 64, 64))
    model = FeaturePyramidViT(fpn_mode="32x32")
    summary(model, input_size=(1, 3, 64, 64))
    model = FeaturePyramidViTLarge(fpn_mode="32x32")
    summary(model, input_size=(1, 3, 64, 64))

    print("\n" + "=" * 80)
    print("Testing multiscale token mode (1344 tokens total):")
    print("=" * 80)
    model = FeaturePyramidViTTiny(fpn_mode="multiscale")
    summary(model, input_size=(1, 3, 64, 64))
    model = FeaturePyramidViTSmall(fpn_mode="multiscale")
    summary(model, input_size=(1, 3, 64, 64))
    model = FeaturePyramidViT(fpn_mode="multiscale")
    summary(model, input_size=(1, 3, 64, 64))
    model = FeaturePyramidViTLarge(fpn_mode="multiscale")
    summary(model, input_size=(1, 3, 64, 64))
