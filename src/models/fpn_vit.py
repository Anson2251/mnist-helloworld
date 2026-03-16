import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any

from .base import BaseModel
from .common import (
    Attention,
    FocusedLinearAttention,
    Mlp,
    SEBlock,
    ConvBlock,
    C3Module,
)
from ..training.metrics import MetricsTracker


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
        x = self.blocks(x)
        x = self.norm(x)
        return x


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


class FeaturePyramidViT(BaseModel):
    """Vision Transformer with Bottleneck for Chinese Character Recognition."""

    @property
    def model_type(self) -> str:
        return "classification"

    @classmethod
    def get_criterion(cls, **kwargs) -> nn.Module:
        return nn.CrossEntropyLoss()

    @classmethod
    def get_metrics_tracker(cls, **kwargs) -> Any:
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
                linear_attention=True,
                linear_layer_limit=4,
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
                linear_attention=True,
                linear_layer_limit=4,
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
            x = self.vit(x)
        else:
            p1, p2, p3 = features
            p1 = self.scale_projectors[0](p1)
            p2 = self.scale_projectors[1](p2)
            p3 = self.scale_projectors[2](p3)
            tokens1 = p1.flatten(2).transpose(1, 2)
            tokens2 = p2.flatten(2).transpose(1, 2)
            tokens3 = p3.flatten(2).transpose(1, 2)
            x = self.vit([tokens1, tokens2, tokens3])

        return self.head(x[:, 0])


class SiameseFPNViT(BaseModel):
    """Siamese FPN-ViT for metric learning with triplet loss."""

    @property
    def model_type(self) -> str:
        return "siamese"

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
                linear_attention=True,
                linear_layer_limit=4,
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
                linear_attention=True,
                linear_layer_limit=4,
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
            x = self.vit(x)
        else:
            p1, p2, p3 = features
            p1 = self.scale_projectors[0](p1)
            p2 = self.scale_projectors[1](p2)
            p3 = self.scale_projectors[2](p3)
            tokens1 = p1.flatten(2).transpose(1, 2)
            tokens2 = p2.flatten(2).transpose(1, 2)
            tokens3 = p3.flatten(2).transpose(1, 2)
            x = self.vit([tokens1, tokens2, tokens3])

        cls_token = x[:, 0]
        embedding = self.projection(cls_token)

        if self.training:
            embedding = F.normalize(embedding, p=2, dim=1)

        return embedding


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
    }


def create_fpn_vit(variant="base", num_classes=631, **kwargs):
    """Factory function to create FPN-ViT with specified variant."""
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
    """Factory function to create Siamese FPN-ViT with specified variant."""
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
    summary(
        model,
        input_size=(1, 3, 64, 64),
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
    )
    model = FeaturePyramidViTSmall(fpn_mode="32x32")
    summary(
        model,
        input_size=(1, 3, 64, 64),
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
    )
    model = FeaturePyramidViT(fpn_mode="32x32")
    summary(
        model,
        input_size=(1, 3, 64, 64),
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
    )
    model = FeaturePyramidViTLarge(fpn_mode="32x32")
    summary(
        model,
        input_size=(1, 3, 64, 64),
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
    )

    print("\n" + "=" * 80)
    print("Testing multiscale token mode (1344 tokens total):")
    print("=" * 80)
    model = FeaturePyramidViTTiny(fpn_mode="multiscale")
    summary(
        model,
        input_size=(1, 3, 64, 64),
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
    )
    model = FeaturePyramidViTSmall(fpn_mode="multiscale")
    summary(
        model,
        input_size=(1, 3, 64, 64),
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
    )
    model = FeaturePyramidViT(fpn_mode="multiscale")
    summary(
        model,
        input_size=(1, 3, 64, 64),
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
    )
    model = FeaturePyramidViTLarge(fpn_mode="multiscale")
    summary(
        model,
        input_size=(1, 3, 64, 64),
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
    )
