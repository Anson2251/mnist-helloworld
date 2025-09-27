from abc import ABC, abstractmethod
import torch.nn as nn
from typing import Any

class BaseModel(nn.Module, ABC):
    """Abstract base class for all models."""

    def __init__(self, num_classes: int = 10, input_channels: int = 1, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels

    @abstractmethod
    def forward(self, x):
        """Forward pass."""
        pass

    def get_model_info(self) -> dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'name': self.__class__.__name__,
            'num_classes': self.num_classes,
            'input_channels': self.input_channels,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        }
