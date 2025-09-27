import torch
import torch.nn as nn
import torch.optim as optim
import os
from typing import Optional, Dict, Any

class CheckpointManager:
    """Manage model checkpoints."""
    
    def __init__(self, checkpoint_dir: str = 'checkpoints'):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.best_accuracy = 0.0
    
    def save_checkpoint(self, 
                       model: nn.Module, 
                       optimizer: optim.Optimizer,
                       epoch: int,
                       loss: float,
                       accuracy: float,
                       filepath: str,
                       additional_info: Optional[Dict[str, Any]] = None):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'accuracy': accuracy
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, 
                       filepath: str, 
                       model: nn.Module, 
                       optimizer: Optional[optim.Optimizer] = None) -> Dict[str, Any]:
        """Load model checkpoint."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return {
            'epoch': checkpoint.get('epoch', 0),
            'loss': checkpoint.get('loss', 0.0),
            'accuracy': checkpoint.get('accuracy', 0.0)
        }
    
    def save_best_model(self, 
                       model: nn.Module, 
                       optimizer: optim.Optimizer,
                       epoch: int,
                       loss: float,
                       accuracy: float,
                       additional_info: Optional[Dict[str, Any]] = None):
        """Save best model if current accuracy is better."""
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            self.save_checkpoint(model, optimizer, epoch, loss, accuracy, 
                               best_path, additional_info)
            return True
        return False
    
    def get_latest_checkpoint_path(self) -> str:
        """Get path for latest checkpoint."""
        return os.path.join(self.checkpoint_dir, 'latest_checkpoint.pt')
    
    def save_latest_checkpoint(self, 
                              model: nn.Module, 
                              optimizer: optim.Optimizer,
                              epoch: int,
                              loss: float,
                              accuracy: float,
                              additional_info: Optional[Dict[str, Any]] = None):
        """Save latest checkpoint."""
        latest_path = self.get_latest_checkpoint_path()
        self.save_checkpoint(model, optimizer, epoch, loss, accuracy, 
                           latest_path, additional_info)