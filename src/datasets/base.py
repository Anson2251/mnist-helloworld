import json
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Any
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from ..utils import setup_logger

logger = setup_logger("datasets")


class BaseDataset(ABC):
    """Abstract base class for all datasets."""

    DATASET_TYPES = ["standard", "pairwise", "triplet"]

    def __init__(
        self,
        root: str = "./data",
        download: bool = True,
        reapply_transforms: bool = False,
        image_size: int = 64,
        output_channels: int = 1,
    ):
        self.root = root
        self.download = download
        self.reapply_transforms = reapply_transforms
        self.image_size = image_size
        self.output_channels = output_channels
        self._train_dataset: Optional[Dataset[Any]] = None
        self._test_dataset: Optional[Dataset[Any]] = None
        self._train_transform = self.get_train_transform(image_size, output_channels)
        self._test_transform = self.get_test_transform(image_size, output_channels)

    @property
    @abstractmethod
    def dataset_type(self) -> str:
        """Dataset paradigm type. Must be one of DATASET_TYPES."""
        pass

    @abstractmethod
    def get_train_transform(
        self, image_size: int, output_channels: int
    ) -> transforms.Compose:
        """Get training data transforms.

        Args:
            image_size: Target image size
            output_channels: Number of output channels
        """
        pass

    @abstractmethod
    def get_test_transform(
        self, image_size: int, output_channels: int
    ) -> transforms.Compose:
        """Get testing data transforms.

        Args:
            image_size: Target image size
            output_channels: Number of output channels
        """
        pass

    @abstractmethod
    def load_data(self) -> Any:
        """Load the dataset."""
        pass

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Number of classes in the dataset."""
        pass

    @property
    @abstractmethod
    def input_channels(self) -> int:
        """Number of input channels."""
        pass

    @property
    @abstractmethod
    def input_size(self) -> Tuple[int, int]:
        """Input image size (height, width)."""
        pass

    def get_index_label_mapping(self) -> dict:
        """Get mapping from class index to label.

        Returns:
            dict: Mapping from integer index to label string.
        """
        raise NotImplementedError("Subclasses must implement get_index_label_mapping()")

    def export_index_label_json(self, output_path: str = "index_label_mapping.json"):
        """Export index-label mapping to JSON file.

        Args:
            output_path: Path to save the JSON file.
        """

        mapping = self.get_index_label_mapping()
        # Convert int keys to strings for JSON
        mapping_str_keys = {str(k): v for k, v in mapping.items()}

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(mapping_str_keys, f, ensure_ascii=False, indent=2)

        logger.info(f"Index-label mapping exported to {output_path}")
        return output_path

    def get_dataloaders(
        self,
        batch_size: int = 64,
        num_workers: int = 4,
        shuffle_train: bool = True,
        val_num_workers: Optional[int] = None,
    ) -> Tuple[DataLoader, DataLoader]:
        """Get train and test DataLoaders."""
        if self._train_dataset is None or self._test_dataset is None:
            self.load_data()

        assert self._train_dataset is not None, "Train dataset is not loaded"
        assert self._test_dataset is not None, "Test dataset is not loaded"

        train_loader = DataLoader(
            self._train_dataset,
            **self._build_loader_kwargs(
                batch_size=batch_size,
                shuffle=shuffle_train,
                num_workers=num_workers,
            ),
        )

        test_loader = DataLoader(
            self._test_dataset,
            **self._build_loader_kwargs(
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers if val_num_workers is None else val_num_workers,
            ),
        )

        return train_loader, test_loader

    def _build_loader_kwargs(
        self, batch_size: int, shuffle: bool, num_workers: int
    ) -> dict[str, Any]:
        """Build DataLoader kwargs with sensible defaults for throughput."""
        worker_count = max(0, num_workers)
        loader_kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": worker_count,
            "pin_memory": torch.cuda.is_available(),
        }

        if worker_count > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 4 if worker_count >= 4 else 2

        return loader_kwargs

    def get_info(self) -> dict:
        """Get dataset information."""
        return {
            "name": self.__class__.__name__,
            "type": self.dataset_type,
            "num_classes": self.num_classes,
            "input_channels": self.input_channels,
            "input_size": self.input_size,
            "root": self.root,
        }

    def reset_train_transforms(self):
        """Reset training transforms to get new random augmentations.

        This should be called after each epoch when reapply_transforms is True
        to get different random augmentations for the training data.
        """
        if not self.reapply_transforms:
            return

        # Regenerate transforms (this creates new random state for random transforms)
        self._train_transform = self.get_train_transform(
            self.image_size, self.output_channels
        )

        # Reload the training dataset with new transforms
        self._reload_train_data()

    @abstractmethod
    def _reload_train_data(self):
        """Reload training data with current transforms.

        Subclasses should implement this to reload the training dataset
        with the current self._train_transform.
        """
        pass


class ClassificationDataset(BaseDataset):
    """Base class for standard classification datasets.

    Returns (image, label) tuples where label is the class index.
    """

    @property
    def dataset_type(self) -> str:
        return "standard"


class TripletDatasetBase(BaseDataset):
    """Base class for triplet-based datasets.

    Returns (anchor, positive, negative, anchor_label) tuples.
    """

    @property
    def dataset_type(self) -> str:
        return "triplet"

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


class BalancedTripletDataset(TripletDatasetBase):
    """Base class for balanced triplet datasets.

    Generates equal number of triplets per class for balanced training.
    Child classes must implement load_data() and _reload_train_data().
    """

    def __init__(
        self,
        root: str = "./data",
        download: bool = True,
        reapply_transforms: bool = False,
        image_size: int = 64,
        output_channels: int = 1,
        triplets_per_class: int = 1000,
    ):
        super().__init__(
            root, download, reapply_transforms, image_size, output_channels
        )
        self.triplets_per_class = triplets_per_class

    def _generate_triplets(
        self,
        data_by_label: dict,
        per_class: int,
        available_indices: list | None = None,
    ) -> list:
        """Generate balanced triplets with equal samples per class.

        Args:
            data_by_label: Dict mapping labels to lists of indices
            per_class: Number of triplets to generate per class
            available_indices: Optional list of available indices to filter by
            desc: Description for progress bar (if tqdm available)

        Returns:
            List of (anchor_idx, positive_idx, negative_idx, anchor_label) tuples
        """
        import random

        triplets = []

        # Filter by available indices if provided
        if available_indices is not None:
            available_set = set(available_indices)
            data_by_label = {
                label: [idx for idx in indices if idx in available_set]
                for label, indices in data_by_label.items()
            }

        labels = list(data_by_label.keys())

        for anchor_label in labels:
            anchor_indices = data_by_label[anchor_label]
            if len(anchor_indices) < 2:
                continue  # Skip if not enough samples for anchor + positive

            for _ in range(per_class):
                # Sample anchor and positive (same label, different samples)
                anchor_idx = random.choice(anchor_indices)
                positive_idx = random.choice(anchor_indices)
                while positive_idx == anchor_idx:
                    positive_idx = random.choice(anchor_indices)

                # Sample negative (different label)
                negative_label = random.choice(
                    [label for label in labels if label != anchor_label]
                )
                negative_indices = data_by_label[negative_label]
                if not negative_indices:
                    continue  # Skip if no negative samples available
                negative_idx = random.choice(negative_indices)

                triplets.append((anchor_idx, positive_idx, negative_idx, anchor_label))

        return triplets


class FixedTripletDataset(Dataset):
    """Dataset with pre-generated triplets."""

    def __init__(self, base_dataset, triplets: list, transform=None):
        self.base_dataset = base_dataset
        self.triplets = triplets
        self.transform = transform

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_idx, positive_idx, negative_idx, anchor_label = self.triplets[idx]

        anchor_img, _ = self.base_dataset[anchor_idx]
        positive_img, _ = self.base_dataset[positive_idx]
        negative_img, _ = self.base_dataset[negative_idx]

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img, anchor_label
