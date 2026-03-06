import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, random_split

from .base import ClassificationDataset
from .utils import (
    get_character_train_transform,
    get_character_test_transform,
    export_index_label_json,
)


class Subset631Dataset(ClassificationDataset):
    """Subset 631 Chinese character dataset."""

    def __init__(
        self,
        root: str = "./data",
        download: bool = True,
        reapply_transforms: bool = True,
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
        """Load Subset 631 dataset."""

        data_path = os.path.join(self.root, "subset_631")

        full_dataset = torchvision.datasets.ImageFolder(
            root=data_path, transform=self._train_transform
        )

        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size

        # Use fixed seed for reproducible split
        import torch

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
        return 631

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

        assert self._train_dataset is not None

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
        data_path = os.path.join(self.root, "subset_631")

        full_dataset = torchvision.datasets.ImageFolder(
            root=data_path, transform=self._train_transform
        )

        # Use stored indices to maintain same train/test split
        if self._train_indices is not None:
            self._train_dataset = Subset(full_dataset, self._train_indices)
        else:
            # Fallback: create new split (shouldn't happen in normal usage)
            train_size = int(0.8 * len(full_dataset))
            test_size = len(full_dataset) - train_size

            generator = torch.Generator().manual_seed(42)
            self._train_dataset, _ = random_split(
                full_dataset, [train_size, test_size], generator=generator
            )
