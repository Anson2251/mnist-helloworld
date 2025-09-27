from typing import Dict, Type
from .base import BaseDataset

class DatasetRegistry:
    """Registry for dataset implementations."""
    
    _datasets: Dict[str, Type[BaseDataset]] = {}
    
    @classmethod
    def register(cls, name: str, dataset_class: Type[BaseDataset]):
        """Register a dataset class."""
        cls._datasets[name.lower()] = dataset_class
    
    @classmethod
    def get(cls, name: str) -> Type[BaseDataset]:
        """Get a dataset class by name."""
        if name.lower() not in cls._datasets:
            available = list(cls._datasets.keys())
            raise ValueError(f"Dataset '{name}' not found. Available: {available}")
        return cls._datasets[name.lower()]
    
    @classmethod
    def list_available(cls) -> list:
        """List all available datasets."""
        return list(cls._datasets.keys())
    
    @classmethod
    def create(cls, name: str, **kwargs) -> BaseDataset:
        """Create a dataset instance."""
        dataset_class = cls.get(name)
        return dataset_class(**kwargs)

# Auto-register datasets
from .mnist import MNISTDataset
from .cifar import CIFARDataset

DatasetRegistry.register('mnist', MNISTDataset)
DatasetRegistry.register('cifar10', CIFARDataset)