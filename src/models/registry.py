from typing import Dict, Type
from .base import BaseModel

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

# Auto-register models
from .lenet import LeNet
from .mynet import MyNet

ModelRegistry.register('lenet', LeNet)
ModelRegistry.register('mynet', MyNet)