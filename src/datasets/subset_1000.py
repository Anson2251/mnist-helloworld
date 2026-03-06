import os
import torch
import torchvision
from torch.utils.data import random_split
import torchvision.transforms as transforms
from .base import ClassificationDataset, BalancedTripletDataset, FixedTripletDataset
from .utils import (
    get_character_train_transform,
    get_character_test_transform,
    export_index_label_json,
)
from ..utils import setup_logger

logger = setup_logger("subset_1000")


class Subset1000Dataset(ClassificationDataset):
    """Subset 1000 Chinese character dataset."""

    def __init__(
        self,
        root: str = "./data",
        download: bool = True,
        reapply_transforms: bool = False,
        image_size: int = 64,
        output_channels: int = 1,
    ):
        super().__init__(
            root, download, reapply_transforms, image_size, output_channels
        )
        self._train_indices = None
        self._test_indices = None

    def get_train_transform(
        self, image_size: int = 64, output_channels: int = 1
    ) -> transforms.Compose:
        return get_character_train_transform(
            image_size=image_size, output_channels=output_channels
        )

    def get_test_transform(
        self, image_size: int = 64, output_channels: int = 1
    ) -> transforms.Compose:
        return get_character_test_transform(
            image_size=image_size, output_channels=output_channels
        )

    def load_data(self):
        """Load Subset 1000 dataset."""

        data_path = os.path.join(self.root, "subset_1000")

        full_dataset = torchvision.datasets.ImageFolder(
            root=data_path, transform=self._train_transform
        )

        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size

        generator = torch.Generator().manual_seed(42)
        train_dataset, test_dataset = random_split(
            full_dataset, [train_size, test_size], generator=generator
        )

        # Store indices to ensure consistent splits across reloads
        self._train_indices = train_dataset.indices
        self._test_indices = test_dataset.indices

        self._train_dataset = train_dataset
        self._test_dataset = test_dataset

    @property
    def num_classes(self) -> int:
        return 1000

    @property
    def input_channels(self) -> int:
        return self.output_channels

    def get_index_label_mapping(self) -> dict:
        """Get mapping from class index to character label.

        Returns:
            dict: Mapping from integer index to character string.
        """
        if self._train_dataset is None:
            self.load_data()

        assert self._train_dataset is not None, "Train dataset is not loaded"

        # ImageFolder stores class_to_idx mapping
        full_dataset = self._train_dataset.dataset  # pyright: ignore[reportAttributeAccessIssue]
        idx_to_class = {v: k for k, v in full_dataset.class_to_idx.items()}
        return idx_to_class

    def export_index_label_json(self, output_path: str = "index_label_mapping.json"):
        """Export index-label mapping to JSON file.

        Args:
            output_path: Path to save the JSON file.
        """
        mapping = self.get_index_label_mapping()
        return export_index_label_json(mapping, output_path)

    @property
    def input_size(self) -> tuple:
        return (self.image_size, self.image_size)

    def _reload_train_data(self):
        """Reload training data with current transforms."""
        data_path = os.path.join(self.root, "subset_1000")

        full_dataset = torchvision.datasets.ImageFolder(
            root=data_path, transform=self._train_transform
        )

        # Use stored indices to maintain same train/test split
        if self._train_indices is not None:
            from torch.utils.data import Subset

            self._train_dataset = Subset(full_dataset, self._train_indices)
        else:
            # Fallback: create new split (shouldn't happen in normal usage)
            train_size = int(0.8 * len(full_dataset))
            test_size = len(full_dataset) - train_size

            generator = torch.Generator().manual_seed(42)
            self._train_dataset, _ = random_split(
                full_dataset, [train_size, test_size], generator=generator
            )


class TripletSubset1000Dataset(BalancedTripletDataset):
    """Subset 1000 dataset for triplet learning."""

    def __init__(
        self,
        root: str = "./data",
        download: bool = True,
        reapply_transforms: bool = False,
        image_size: int = 64,
        output_channels: int = 1,
        triplets_per_class: int = 100,
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
        self, image_size: int = 64, output_channels: int = 1
    ) -> transforms.Compose:
        return get_character_train_transform(
            image_size=image_size, output_channels=output_channels
        )

    def get_test_transform(
        self, image_size: int = 64, output_channels: int = 1
    ) -> transforms.Compose:
        return get_character_test_transform(
            image_size=image_size, output_channels=output_channels
        )

    def load_data(self):
        """Load Subset 1000 dataset and generate triplets."""
        data_path = os.path.join(self.root, "subset_1000")

        # Load full dataset without transform first to organize by label
        logger.info("Loading dataset from disk...")
        full_dataset = torchvision.datasets.ImageFolder(root=data_path, transform=None)

        # Organize by label with progress bar
        logger.info(f"Organizing {len(full_dataset)} samples by label...")
        data_by_label = {}
        # Use imgs attribute directly for faster access (avoids __getitem__ overhead)
        for idx, (_, label) in enumerate(full_dataset.imgs):
            if label not in data_by_label:
                data_by_label[label] = []
            data_by_label[label].append(idx)

        # Split into train/test (80/20)
        train_indices = []
        test_indices = []
        for label, indices in data_by_label.items():
            train_size = int(0.8 * len(indices))
            train_indices.extend(indices[:train_size])
            test_indices.extend(indices[train_size:])

        # Create datasets with transforms

        logger.info(f"Generating triplets for {len(data_by_label)} classes...")
        train_triplets = self._generate_triplets(
            data_by_label,
            self.triplets_per_class,
            available_indices=train_indices,
        )
        test_triplets = self._generate_triplets(
            data_by_label,
            self.triplets_per_class // 10,
            available_indices=test_indices,
        )

        self._train_dataset = FixedTripletDataset(
            base_dataset=full_dataset,
            triplets=train_triplets,
            transform=self._train_transform,
        )
        self._test_dataset = FixedTripletDataset(
            base_dataset=full_dataset,
            triplets=test_triplets,
            transform=self._test_transform,
        )

        # Store train indices for reload
        train_set = set(train_indices)

        logger.info("Filtering train indices by label...")
        self._train_indices_by_label = {
            label: [idx for idx in indices if idx in train_set]
            for label, indices in data_by_label.items()
        }

    def _reload_train_data(self):
        """Reload training data with current transforms."""
        data_path = os.path.join(self.root, "subset_1000")
        full_dataset = torchvision.datasets.ImageFolder(root=data_path, transform=None)

        logger.info("Regenerating training triplets...")
        all_train_indices = [
            idx for indices in self._train_indices_by_label.values() for idx in indices
        ]
        train_triplets = self._generate_triplets(
            self._train_indices_by_label,
            self.triplets_per_class,
            available_indices=all_train_indices,
        )

        self._train_dataset = FixedTripletDataset(
            base_dataset=full_dataset,
            triplets=train_triplets,
            transform=self._train_transform,
        )

    @property
    def num_classes(self) -> int:
        return 1000

    @property
    def input_channels(self) -> int:
        return self.output_channels

    def get_index_label_mapping(self) -> dict:
        """Get mapping from class index to character label.

        Returns:
            dict: Mapping from integer index to character string.
        """
        if self._train_dataset is None:
            self.load_data()

        assert self._train_dataset is not None, "Cannot load training dataset"

        # ImageFolder stores class_to_idx mapping
        full_dataset = self._train_dataset.dataset  # pyright: ignore[reportAttributeAccessIssue]
        idx_to_class = {v: k for k, v in full_dataset.class_to_idx.items()}
        return idx_to_class

    def export_index_label_json(self, output_path: str = "index_label_mapping.json"):
        """Export index-label mapping to JSON file.

        Args:
            output_path: Path to save the JSON file.
        """
        mapping = self.get_index_label_mapping()
        return export_index_label_json(mapping, output_path)

    @property
    def input_size(self) -> tuple:
        return (self.image_size, self.image_size)
