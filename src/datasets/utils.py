"""Dataset utility functions and transformations."""

import json
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms
from PIL import Image

from ..utils import setup_logger

logger = setup_logger("datasets_utils")


class ResizePad:
    """Resize image while keeping aspect ratio and pad to square."""

    def __init__(self, target_size, pad_value=0):
        self.target_size = target_size
        self.pad_value = pad_value

    def __call__(self, img):
        # Calculate aspect ratio
        width, height = img.size
        scale = self.target_size / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        # Handle PIL resampling
        img = img.resize((new_width, new_height), Image.Resampling.BILINEAR)

        # Create new image with pad_value
        new_img = Image.new(
            img.mode, (self.target_size, self.target_size), self.pad_value
        )
        # Paste centered
        left = (self.target_size - new_width) // 2
        top = (self.target_size - new_height) // 2
        new_img.paste(img, (left, top))
        return new_img


def decode_label(label):
    """Decode label to proper Chinese character.

    Handles various encoding formats:
    - #UXXXX format (common in some datasets)
    - Unicode escape sequences like \u4e14
    - Already decoded characters

    Args:
        label: The label string to decode

    Returns:
        Decoded string
    """
    if isinstance(label, str):
        # Check for #UXXXX format (common in some datasets)
        if label.startswith("#U") or label.startswith("#u"):
            try:
                hex_code = label[2:]  # Remove #U or #u prefix
                return chr(int(hex_code, 16))
            except (ValueError, OverflowError):
                return label
        # Check for Unicode escape sequences like \u4e14
        if "\\u" in label or "\\U" in label:
            try:
                return label.encode("utf-8").decode("unicode-escape")
            except (UnicodeDecodeError, UnicodeEncodeError):
                return label
        # Already a proper character
        return label
    return str(label)


def export_index_label_json(
    mapping: dict, output_path: str = "index_label_mapping.json"
) -> str:
    """Export index-label mapping to JSON file with Chinese character decoding.

    Args:
        mapping: Dictionary mapping from integer index to label string
        output_path: Path to save the JSON file

    Returns:
        Path to the exported file
    """
    # Convert int keys to strings and decode labels
    mapping_str_keys = {str(k): decode_label(v) for k, v in mapping.items()}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mapping_str_keys, f, ensure_ascii=False, indent=2)

    logger.info(f"Index-label mapping exported to {output_path}")
    return output_path


class AlbumentationsTransform:
    """Wrapper to use Albumentations transforms with torchvision datasets."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        # Convert PIL Image to numpy array
        img_np = np.array(img)
        # Apply Albumentations transform with named argument
        transformed = self.transform(image=img_np)
        return transformed["image"]


def get_character_train_transform(
    image_size: int = 64, output_channels: int = 1
) -> transforms.Compose:
    """Get training transforms for character/digit recognition datasets.

    Uses Albumentations for aggressive augmentation suitable for handwritten
    characters and digits.

    Args:
        image_size: Target image size (default: 64x64)
        output_channels: Number of output channels (default: 1 for grayscale)

    Returns:
        torchvision.transforms.Compose: Composed transform pipeline
    """
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=output_channels),
            AlbumentationsTransform(
                A.Compose(
                    [
                        # Geometric transforms: simulate natural handwriting variations
                        A.Affine(
                            scale=(0.8, 1.2),  # Random scale ±20%
                            translate_percent=(-0.1, 0.1),  # Random translation ±10%
                            rotate=(-15, 15),  # Random rotation ±15°
                            shear=(-10, 10),  # Random shear ±10°
                            p=0.8,
                        ),
                        # Elastic deformation: simulate natural stroke curvature
                        A.ElasticTransform(
                            alpha=1,
                            sigma=50,
                            p=0.3,
                        ),
                        # Stroke thickness variation
                        A.RandomGridShuffle(
                            grid=(2, 2), p=0.2
                        ),  # Random local block shuffle (force structure learning)
                        # Local occlusion: simulate stains or missing strokes
                        A.CoarseDropout(
                            num_holes_range=(1, 4),  # Number of holes
                            hole_height_range=(
                                0.05,
                                0.1,
                            ),  # Hole height as ratio of image height
                            hole_width_range=(
                                0.05,
                                0.1,
                            ),  # Hole width as ratio of image width
                            fill=0,  # Fill value
                            p=0.3,
                        ),
                        # Contrast/brightness: simulate different paper/scanners
                        A.RandomBrightnessContrast(
                            brightness_limit=(-0.2, 0.2),
                            contrast_limit=(-0.2, 0.2),
                            p=0.5,
                        ),
                        # Gaussian noise: simulate sensor noise
                        A.GaussNoise(p=0.2),
                        # Final processing: resize preserving aspect ratio, pad to square
                        A.LongestMaxSize(max_size=image_size, p=1.0),
                        A.PadIfNeeded(
                            min_height=image_size,
                            min_width=image_size,
                            border_mode=0,
                            fill=0,
                            p=1.0,
                        ),
                        A.Normalize(mean=0.5, std=0.5),
                        ToTensorV2(),
                    ]
                )
            ),
        ]
    )


def get_character_test_transform(
    image_size: int = 64, output_channels: int = 1
) -> transforms.Compose:
    """Get test transforms for character/digit recognition datasets.

    Minimal augmentation for validation/testing.

    Args:
        image_size: Target image size (default: 64x64)
        output_channels: Number of output channels (default: 1 for grayscale)

    Returns:
        torchvision.transforms.Compose: Composed transform pipeline
    """
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=output_channels),
            AlbumentationsTransform(
                A.Compose(
                    [
                        A.LongestMaxSize(max_size=image_size, p=1.0),
                        A.PadIfNeeded(
                            min_height=image_size,
                            min_width=image_size,
                            border_mode=0,
                            fill=0,
                            p=1.0,
                        ),
                        A.Normalize(mean=0.5, std=0.5),
                        ToTensorV2(),
                    ]
                )
            ),
        ]
    )


def get_simple_train_transform(
    image_size: int = 64, output_channels: int = 1
) -> transforms.Compose:
    """Get simple training transforms for character datasets.

    Lighter augmentation compared to get_character_train_transform.

    Args:
        image_size: Target image size (default: 64x64)
        output_channels: Number of output channels (default: 1 for grayscale)

    Returns:
        torchvision.transforms.Compose: Composed transform pipeline
    """
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=output_channels),
            ResizePad(image_size, pad_value=0),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def get_simple_test_transform(
    image_size: int = 64, output_channels: int = 1
) -> transforms.Compose:
    """Get simple test transforms for character datasets.

    Args:
        image_size: Target image size (default: 64x64)
        output_channels: Number of output channels (default: 1 for grayscale)

    Returns:
        torchvision.transforms.Compose: Composed transform pipeline
    """
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=output_channels),
            ResizePad(image_size, pad_value=0),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
