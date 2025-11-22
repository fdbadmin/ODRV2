from typing import Any

import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A


def build_training_augmentations(image_size: tuple[int, int]) -> A.Compose:
    """Create a strong augmentation pipeline for fundus images."""
    height, width = image_size
    return A.Compose([
        A.RandomResizedCrop(size=(height, width), scale=(0.9, 1.0), ratio=(0.95, 1.05)),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            scale=(0.95, 1.05),
            translate_percent={"x": (-0.04, 0.04), "y": (-0.04, 0.04)},
            rotate=(-180, 180),
            border_mode=0,
            keep_ratio=True,
            p=0.7,
        ),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.15, p=0.7),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(std_range=(0.01, 0.08), mean_range=(0.0, 0.0), p=0.15),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1.0),
    ])


def build_validation_transforms(image_size: tuple[int, int]) -> A.Compose:
    """Create validation transforms with deterministic resizing."""
    height, width = image_size
    return A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1.0),
    ])


def to_tensor_transform() -> list[Any]:
    """Return conversion steps for PyTorch datamodules."""
    from albumentations.pytorch import ToTensorV2  # type: ignore[import]
    return [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1.0),
        ToTensorV2(),
    ]
