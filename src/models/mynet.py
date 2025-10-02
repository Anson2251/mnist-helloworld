import torch.nn as nn
from .base import BaseModel

class Conv(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, kernel_size: tuple = (3, 3), 
                 act: nn.Module = None, bn: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size)
        self.bn = nn.BatchNorm2d(ch_out) if bn else nn.Identity()
        self.act = act if act else nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Linear(nn.Module):
    def __init__(self, feat_in: int, feat_out: int, bias: bool = True, 
                 act: nn.Module = None, dropout: float = 0.5):
        super().__init__()
        self.linear = nn.Linear(feat_in, feat_out, bias)
        self.act = act if act else nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(self.linear(x)))

class MyNet(BaseModel):
    """Custom network similar to the original implementation."""
    
    def __init__(self, num_classes: int = 10, input_channels: int = 1, input_size: tuple = (28, 28), **kwargs):
        super().__init__(num_classes, input_channels)
        
        self.channels = 16
        self.input_size = input_size
        
        self.features = nn.Sequential(
            Conv(input_channels, self.channels, (5, 5)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv(self.channels, self.channels, (5, 1)),
            Conv(self.channels, self.channels, (1, 5)),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Calculate the flattened feature size based on input size
        if input_size == (28, 28):
            # For 28x28: (28-5+1)/2 = 12, (12-5+1)/2 = 4, so 16*4*4 = 256
            self.feature_size = self.channels * 4 * 4
        elif input_size == (32, 32):
            # For 32x32: (32-5+1)/2 = 14, (14-5+1)/2 = 5, so 16*5*5 = 400
            self.feature_size = self.channels * 5 * 5
        else:
            # Calculate for arbitrary input size
            h, w = input_size
            h = (h - 5 + 1) // 2  # After first conv and pool with 5x5 kernel
            h = (h - 5 + 1) // 2  # After asymmetric convolutions and second pool
            w = (w - 5 + 1) // 2  # After first conv and pool with 5x5 kernel
            w = (w - 5 + 1) // 2  # After asymmetric convolutions and second pool
            self.feature_size = self.channels * h * w
        
        self.classifier = nn.Sequential(
            Linear(self.feature_size, self.channels * 4 * 2, dropout=0.2),
            Linear(self.channels * 4 * 2, self.channels * 4, dropout=0.3),
            nn.Linear(self.channels * 4, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten dynamically
        x = self.classifier(x)
        return x