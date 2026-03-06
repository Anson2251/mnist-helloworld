import torchvision
import torchvision.transforms as transforms
from .base import ClassificationDataset
from .utils import ResizePad


class MNISTDataset(ClassificationDataset):
    """MNIST dataset implementation."""

    def __init__(
        self,
        root: str = "./data",
        download: bool = True,
        reapply_transforms: bool = False,
        image_size: int = 28,
        output_channels: int = 1,
    ):
        super().__init__(
            root, download, reapply_transforms, image_size, output_channels
        )

    def get_train_transform(
        self, image_size: int = 28, output_channels: int = 1
    ) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=output_channels),
                ResizePad(image_size, pad_value=0),
                transforms.RandomAffine(
                    degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=3
                ),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def get_test_transform(
        self, image_size: int = 28, output_channels: int = 1
    ) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=output_channels),
                ResizePad(image_size, pad_value=0),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def load_data(self):
        """Load MNIST dataset."""
        self._train_dataset = torchvision.datasets.MNIST(
            root=self.root,
            train=True,
            download=self.download,
            transform=self._train_transform,
        )
        self._test_dataset = torchvision.datasets.MNIST(
            root=self.root,
            train=False,
            download=self.download,
            transform=self._test_transform,
        )

    def _reload_train_data(self):
        """Reload training data with current transforms."""
        self._train_dataset = torchvision.datasets.MNIST(
            root=self.root, train=True, download=False, transform=self._train_transform
        )

    @property
    def num_classes(self) -> int:
        return 10

    @property
    def input_channels(self) -> int:
        return self.output_channels

    @property
    def input_size(self) -> tuple:
        return (self.image_size, self.image_size)
