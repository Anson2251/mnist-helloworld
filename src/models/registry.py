from typing import Dict, Type
from .base import BaseModel

# Auto-register models
from .lenet import LeNet
from .mynet import MyNet
from .bottleneck_vit import BottleneckViT
from .fpn_vit import (
    FeaturePyramidViT,
    SiameseFPNViT,
    FeaturePyramidViTTiny,
    FeaturePyramidViTSmall,
    FeaturePyramidViTLarge,
    SiameseFPNViTTiny,
    SiameseFPNViTSmall,
    SiameseFPNViTLarge,
)
from .fpn_moe_vit import (
    FeaturePyramidMoEViT,
    SiameseFPNMoEViT,
    FeaturePyramidMoEViTTiny,
    FeaturePyramidMoEViTSmall,
    FeaturePyramidMoEViTLarge,
    SiameseFPNMoEViTTiny,
    SiameseFPNMoEViTSmall,
    SiameseFPNMoEViTLarge,
)
from .siamese import SiameseNetwork
from .alexnet import AlexNet


class ModelRegistry:
    """Registry for model implementations."""

    _models: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str, model_class: Type[BaseModel]):
        """Register a model class."""
        cls._models[name.lower()] = model_class

    @classmethod
    def get(cls, name: str) -> Type[BaseModel]:
        """Get a model class by name."""
        if name.lower() not in cls._models:
            available = list(cls._models.keys())
            raise ValueError(f"Model '{name}' not found. Available: {available}")
        return cls._models[name.lower()]

    @classmethod
    def list_available(cls) -> list:
        """List all available models."""
        return list(cls._models.keys())

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseModel:
        """Create a model instance."""
        model_class = cls.get(name)
        return model_class(**kwargs)

ModelRegistry.register("lenet", LeNet)
ModelRegistry.register("mynet", MyNet)
ModelRegistry.register("bottleneck_vit", BottleneckViT)
ModelRegistry.register("fpn_vit", FeaturePyramidViT)
ModelRegistry.register("fpn_vit_tiny", FeaturePyramidViTTiny)
ModelRegistry.register("fpn_vit_small", FeaturePyramidViTSmall)
ModelRegistry.register("fpn_vit_large", FeaturePyramidViTLarge)
ModelRegistry.register("siamese", SiameseNetwork)
ModelRegistry.register("siamese_fpn_vit", SiameseFPNViT)
ModelRegistry.register("siamese_fpn_vit_tiny", SiameseFPNViTTiny)
ModelRegistry.register("siamese_fpn_vit_small", SiameseFPNViTSmall)
ModelRegistry.register("siamese_fpn_vit_large", SiameseFPNViTLarge)
ModelRegistry.register("fpn_moe_vit", FeaturePyramidMoEViT)
ModelRegistry.register("fpn_moe_vit_tiny", FeaturePyramidMoEViTTiny)
ModelRegistry.register("fpn_moe_vit_small", FeaturePyramidMoEViTSmall)
ModelRegistry.register("fpn_moe_vit_large", FeaturePyramidMoEViTLarge)
ModelRegistry.register("siamese_fpn_moe_vit", SiameseFPNMoEViT)
ModelRegistry.register("siamese_fpn_moe_vit_tiny", SiameseFPNMoEViTTiny)
ModelRegistry.register("siamese_fpn_moe_vit_small", SiameseFPNMoEViTSmall)
ModelRegistry.register("siamese_fpn_moe_vit_large", SiameseFPNMoEViTLarge)
ModelRegistry.register("alexnet", AlexNet)
