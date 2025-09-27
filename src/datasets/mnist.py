import torchvision
import torchvision.transforms as transforms
from .base import BaseDataset

class MNISTDataset(BaseDataset):
    """MNIST dataset implementation."""
    
    def get_train_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=3),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    def get_test_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=3),
            transforms.ToTensor(),
            transforms.Normalize((0.1326,), (0.3106,))
        ])
    
    def load_data(self):
        """Load MNIST dataset."""
        self._train_dataset = torchvision.datasets.MNIST(
            root=self.root, train=True, download=self.download, transform=self._train_transform
        )
        self._test_dataset = torchvision.datasets.MNIST(
            root=self.root, train=False, download=self.download, transform=self._test_transform
        )
    
    @property
    def num_classes(self) -> int:
        return 10
    
    @property
    def input_channels(self) -> int:
        return 1
    
    @property
    def input_size(self) -> tuple:
        return (28, 28)