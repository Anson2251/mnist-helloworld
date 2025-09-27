import torch.nn as nn
from .base import BaseModel

class LeNet(BaseModel):
    """Classic LeNet-5 implementation."""
    
    def __init__(self, num_classes: int = 10, input_channels: int = 1, input_size: tuple = (28, 28), **kwargs):
        super().__init__(num_classes, input_channels)
        
        self.input_size = input_size
        
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 6, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # Calculate the size of the flattened features
        # For 28x28 input: (28-5+1)/2 = 12, (12-5+1)/2 = 4, so 16*4*4 = 256
        # For 32x32 input: (32-5+1)/2 = 14, (14-5+1)/2 = 5, so 16*5*5 = 400
        if input_size == (28, 28):
            self.flattened_size = 16 * 4 * 4  # 256
        elif input_size == (32, 32):
            self.flattened_size = 16 * 5 * 5  # 400
        else:
            # Calculate for arbitrary input size
            h, w = input_size
            h = (h - 5 + 1) // 2  # After first conv and pool
            h = (h - 5 + 1) // 2  # After second conv and pool
            w = (w - 5 + 1) // 2  # After first conv and pool
            w = (w - 5 + 1) // 2  # After second conv and pool
            self.flattened_size = 16 * h * w
        
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten dynamically
        x = self.classifier(x)
        return x