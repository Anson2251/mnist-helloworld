import torchvision
import torchvision.transforms as transforms
import random
from torch.utils.data import Dataset
from .base import TripletDatasetBase, BalancedTripletDataset, FixedTripletDataset
from .utils import ResizePad


class TripletMNISTDataset(TripletDatasetBase):
    """
    MNIST dataset for triplet learning.

    Returns triplets (anchor, positive, negative) for metric learning.
    Anchor and positive have the same label.
    Anchor and negative have different labels.
    """

    def __init__(
        self,
        root: str = "./data",
        download: bool = True,
        reapply_transforms: bool = False,
        image_size: int = 28,
        output_channels: int = 1,
        num_triplets: int | None = None,
    ):
        super().__init__(
            root, download, reapply_transforms, image_size, output_channels
        )
        self.num_triplets = num_triplets

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
        """Load MNIST dataset and organize by label."""
        # Load base MNIST dataset
        base_train = torchvision.datasets.MNIST(
            root=self.root,
            train=True,
            download=self.download,
            transform=None,  # We'll apply transforms later
        )

        base_test = torchvision.datasets.MNIST(
            root=self.root,
            train=False,
            download=self.download,
            transform=None,
        )

        # Organize by label for triplet sampling
        self.train_data_by_label = {i: [] for i in range(10)}
        self.test_data_by_label = {i: [] for i in range(10)}

        for idx, (img, label) in enumerate(base_train):  # pyright: ignore[ reportArgumentType]
            self.train_data_by_label[label].append(idx)

        for idx, (img, label) in enumerate(base_test):  # pyright: ignore[ reportArgumentType]
            self.test_data_by_label[label].append(idx)

        # Create triplet datasets
        self._train_dataset = TripletDataset(
            base_dataset=base_train,
            data_by_label=self.train_data_by_label,
            transform=self._train_transform,
            num_triplets=self.num_triplets,
        )

        self._test_dataset = TripletDataset(
            base_dataset=base_test,
            data_by_label=self.test_data_by_label,
            transform=self._test_transform,
            num_triplets=self.num_triplets
            if self.num_triplets
            else len(base_test) // 10,
        )

    def _reload_train_data(self):
        """Reload training data with current transforms."""
        assert self.num_triplets is not None, "Number of triplets must be specified"

        base_train = torchvision.datasets.MNIST(
            root=self.root,
            train=True,
            download=False,
            transform=None,
        )

        self._train_dataset = TripletDataset(
            base_dataset=base_train,
            data_by_label=self.train_data_by_label,
            transform=self._train_transform,
            num_triplets=self.num_triplets,
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


class TripletDataset(Dataset):
    """
    Dataset that generates triplets on-the-fly.
    """

    def __init__(
        self,
        base_dataset,
        data_by_label: dict,
        transform=None,
        num_triplets: int | None = None,
    ):
        self.base_dataset = base_dataset
        self.data_by_label = data_by_label
        self.transform = transform
        self.num_triplets = num_triplets or len(base_dataset)

        # Get all labels
        self.labels = list(data_by_label.keys())

    def __len__(self):
        return self.num_triplets

    def __getitem__(self, idx):
        """
        Returns a triplet: (anchor, positive, negative) and anchor label.

        Returns:
            anchor_img: Tensor of shape (C, H, W)
            positive_img: Tensor of shape (C, H, W)
            negative_img: Tensor of shape (C, H, W)
            anchor_label: int
        """
        # Randomly select anchor label
        anchor_label = random.choice(self.labels)

        # Get indices for anchor label
        anchor_indices = self.data_by_label[anchor_label]

        # Sample anchor and positive (same label)
        anchor_idx = random.choice(anchor_indices)
        positive_idx = random.choice(anchor_indices)

        # Ensure anchor != positive
        while positive_idx == anchor_idx:
            positive_idx = random.choice(anchor_indices)

        # Sample negative (different label)
        negative_label = random.choice(
            [label for label in self.labels if label != anchor_label]
        )
        negative_idx = random.choice(self.data_by_label[negative_label])

        # Load images
        anchor_img, _ = self.base_dataset[anchor_idx]
        positive_img, _ = self.base_dataset[positive_idx]
        negative_img, _ = self.base_dataset[negative_idx]

        # Apply transforms
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img, anchor_label


class BalancedTripletMNISTDataset(BalancedTripletDataset):
    """
    Balanced MNIST triplet dataset with offline triplet generation.

    Pre-generates a fixed set of triplets to ensure balanced sampling.
    """

    def __init__(
        self,
        root: str = "./data",
        download: bool = True,
        reapply_transforms: bool = False,
        image_size: int = 28,
        output_channels: int = 1,
        triplets_per_class: int = 1000,
    ):
        super().__init__(
            root,
            download,
            reapply_transforms,
            image_size,
            output_channels,
            triplets_per_class,
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
        """Load MNIST and generate triplets."""
        # Load base MNIST dataset
        base_train = torchvision.datasets.MNIST(
            root=self.root,
            train=True,
            download=self.download,
            transform=None,
        )

        base_test = torchvision.datasets.MNIST(
            root=self.root,
            train=False,
            download=self.download,
            transform=None,
        )

        # Organize by label
        train_by_label = {i: [] for i in range(10)}
        test_by_label = {i: [] for i in range(10)}

        for idx, (img, label) in enumerate(base_train):  # pyright: ignore[ reportArgumentType]
            train_by_label[label].append(idx)

        for idx, (img, label) in enumerate(base_test):  # pyright: ignore[ reportArgumentType]
            test_by_label[label].append(idx)

        # Generate triplets
        train_triplets = self._generate_triplets(
            train_by_label, self.triplets_per_class
        )
        test_triplets = self._generate_triplets(
            test_by_label, self.triplets_per_class // 10
        )

        # Create datasets
        self._train_dataset = FixedTripletDataset(
            base_dataset=base_train,
            triplets=train_triplets,
            transform=self._train_transform,
        )

        self._test_dataset = FixedTripletDataset(
            base_dataset=base_test,
            triplets=test_triplets,
            transform=self._test_transform,
        )

    def _reload_train_data(self):
        """Reload with new transforms."""
        base_train = torchvision.datasets.MNIST(
            root=self.root,
            train=True,
            download=False,
            transform=None,
        )

        train_by_label = {i: [] for i in range(10)}
        for idx, (_, label) in enumerate(base_train):  # pyright: ignore[ reportArgumentType]
            train_by_label[label].append(idx)

        train_triplets = self._generate_triplets(
            train_by_label, self.triplets_per_class
        )

        self._train_dataset = FixedTripletDataset(
            base_dataset=base_train,
            triplets=train_triplets,
            transform=self._train_transform,
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
