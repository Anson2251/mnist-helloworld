import torch
from typing import Dict, List, Optional
import time

class MetricsTracker:
    """Track training and validation metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.losses: List[float] = []
        self.correct = 0
        self.total = 0
        self.start_time = time.time()
    
    def update(self, loss: float, outputs: torch.Tensor, targets: torch.Tensor):
        """Update metrics with batch results."""
        self.losses.append(loss)
        
        _, predicted = torch.max(outputs.data, 1)
        self.total += targets.size(0)
        self.correct += (predicted == targets).sum().item()
    
    def get_accuracy(self) -> float:
        """Get current accuracy."""
        if self.total == 0:
            return 0.0
        return 100.0 * self.correct / self.total
    
    def get_average_loss(self) -> float:
        """Get average loss."""
        if not self.losses:
            return 0.0
        return sum(self.losses) / len(self.losses)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get all metrics as dictionary."""
        return {
            'loss': self.get_average_loss(),
            'accuracy': self.get_accuracy(),
            'total_samples': self.total
        }