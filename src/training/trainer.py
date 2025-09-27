import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from typing import Optional, Dict, Any, Callable
from .metrics import MetricsTracker
from .checkpoint import CheckpointManager

class Trainer:
    """Modular training framework."""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 device: torch.device,
                 checkpoint_manager: Optional[CheckpointManager] = None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} - Training')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            self.train_metrics.update(loss.item(), outputs, labels)
            
            # Update progress bar
            metrics = self.train_metrics.get_metrics()
            pbar.set_postfix({
                'Loss': f'{metrics["loss"]:.4f}',
                'Acc': f'{metrics["accuracy"]:.2f}%'
            })
        
        return self.train_metrics.get_metrics()
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        self.val_metrics.reset()
        
        pbar = tqdm(self.val_loader, desc='Validating')
        
        with torch.no_grad():
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                self.val_metrics.update(loss.item(), outputs, labels)
                
                # Update progress bar
                metrics = self.val_metrics.get_metrics()
                pbar.set_postfix({
                    'Loss': f'{metrics["loss"]:.4f}',
                    'Acc': f'{metrics["accuracy"]:.2f}%'
                })
        
        return self.val_metrics.get_metrics()
    
    def train(self, epochs: int, start_epoch: int = 0) -> Dict[str, Any]:
        """Train the model for specified epochs."""
        print(f'Starting training for {epochs} epochs')
        start_time = time.time()
        
        best_accuracy = 0.0
        training_history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        for epoch in range(start_epoch, epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Update history
            training_history['train_loss'].append(train_metrics['loss'])
            training_history['train_accuracy'].append(train_metrics['accuracy'])
            training_history['val_loss'].append(val_metrics['loss'])
            training_history['val_accuracy'].append(val_metrics['accuracy'])
            
            # Print epoch summary
            print(f'Epoch {epoch+1}/{epochs} - '
                  f'Train Loss: {train_metrics["loss"]:.4f}, '
                  f'Train Acc: {train_metrics["accuracy"]:.2f}% - '
                  f'Val Loss: {val_metrics["loss"]:.4f}, '
                  f'Val Acc: {val_metrics["accuracy"]:.2f}%')
            
            # Save checkpoints
            if self.checkpoint_manager:
                # Save latest
                self.checkpoint_manager.save_latest_checkpoint(
                    self.model, self.optimizer, epoch+1,
                    val_metrics['loss'], val_metrics['accuracy']
                )
                
                # Save best
                is_best = self.checkpoint_manager.save_best_model(
                    self.model, self.optimizer, epoch+1,
                    val_metrics['loss'], val_metrics['accuracy']
                )
                
                if is_best:
                    best_accuracy = val_metrics['accuracy']
                    print(f'New best model saved with accuracy: {best_accuracy:.2f}%')
        
        training_time = time.time() - start_time
        
        return {
            'epochs_trained': epochs,
            'best_accuracy': best_accuracy,
            'training_time': training_time,
            'history': training_history
        }