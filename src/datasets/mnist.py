import torchvision
import torchvision.transforms as transforms
from .base import ClassificationDataset


def _mnist_normalize(output_channels: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    mean = (0.1307,) * output_channels
    std = (0.3081,) * output_channels
    return mean, std


def _build_mnist_transform(
    image_size: int, output_channels: int, train: bool
) -> transforms.Compose:
    steps: list[object] = []

    if output_channels != 1:
        steps.append(transforms.Grayscale(num_output_channels=output_channels))

    if image_size != 28:
        steps.append(transforms.Resize((image_size, image_size), antialias=True))

    if train:
        steps.append(
            transforms.RandomAffine(
                degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=3
            )
        )

    mean, std = _mnist_normalize(output_channels)
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    return transforms.Compose(steps)


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
        return _build_mnist_transform(image_size, output_channels, train=True)

    def get_test_transform(
        self, image_size: int = 28, output_channels: int = 1
    ) -> transforms.Compose:
        return _build_mnist_transform(image_size, output_channels, train=False)

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
