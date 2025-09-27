import argparse
from typing import Dict, Any, Optional
from pathlib import Path

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

class Config:
    """Configuration management."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
        self.dataset = config_dict.get('dataset', {})
        self.model = config_dict.get('model', {})
        self.training = config_dict.get('training', {})
        self.optimization = config_dict.get('optimization', {})
        self.checkpointing = config_dict.get('checkpointing', {})
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is required for YAML configuration files. Install with: pip install pyyaml")
        
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'Config':
        """Create configuration from command line arguments."""
        config_dict = {
            'dataset': {
                'name': args.dataset,
                'root': args.data_root,
                'download': True
            },
            'model': {
                'name': args.model,
                'num_classes': args.num_classes
            },
            'training': {
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'num_workers': args.num_workers
            },
            'optimization': {
                'learning_rate': args.learning_rate,
                'optimizer': args.optimizer
            },
            'checkpointing': {
                'checkpoint_dir': args.checkpoint_dir,
                'save_frequency': args.save_frequency
            }
        }
        return cls(config_dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self._config.copy()

def get_default_config() -> Dict[str, Any]:
    """Get default configuration."""
    return {
        'dataset': {
            'name': 'mnist',
            'root': './data',
            'download': True
        },
        'model': {
            'name': 'mynet',
            'num_classes': 10,
            'input_channels': 1
        },
        'training': {
            'epochs': 20,
            'batch_size': 64,
            'num_workers': 4,
            'shuffle_train': True
        },
        'optimization': {
            'learning_rate': 1e-3,
            'optimizer': 'adamw',
            'weight_decay': 0.01
        },
        'checkpointing': {
            'checkpoint_dir': 'checkpoints',
            'save_frequency': 1
        }
    }

def create_config_parser() -> argparse.ArgumentParser:
    """Create argument parser for configuration."""
    parser = argparse.ArgumentParser(description='Train a neural network')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'cifar10'],
                       help='Dataset to use')
    parser.add_argument('--data-root', type=str, default='./data',
                       help='Root directory for datasets')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='mynet',
                       choices=['lenet', 'mynet'],
                       help='Model architecture')
    parser.add_argument('--num-classes', type=int, default=10,
                       help='Number of classes')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Optimization arguments
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adamw', 'adam', 'sgd'],
                       help='Optimizer')
    
    # Checkpointing arguments
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory for checkpoints')
    parser.add_argument('--save-frequency', type=int, default=1,
                       help='Save frequency in epochs')
    
    # Configuration file
    parser.add_argument('--config', type=str,
                       help='Path to YAML configuration file')
    
    return parser

def load_config(config_path: Optional[str] = None, args: Optional[argparse.Namespace] = None) -> Config:
    """Load configuration from file or arguments."""
    if config_path:
        return Config.from_yaml(config_path)
    elif args:
        return Config.from_args(args)
    else:
        return Config(get_default_config())