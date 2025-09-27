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
    
    def __init__(self, num_classes: int = 10, input_channels: int = 1, **kwargs):
        super().__init__(num_classes, input_channels)
        
        self.channels = 16
        self.features = nn.Sequential(
            Conv(input_channels, self.channels, (5, 5)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv(self.channels, self.channels, (5, 1)),
            Conv(self.channels, self.channels, (1, 5)),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            Linear(self.channels * 4 * 4, self.channels * 4 * 2, dropout=0.2),
            Linear(self.channels * 4 * 2, self.channels * 4, dropout=0.3),
            nn.Linear(self.channels * 4, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.channels * 4 * 4)
        x = self.classifier(x)
        return x