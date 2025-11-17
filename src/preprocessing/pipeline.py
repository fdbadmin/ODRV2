from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def load_fundus_image(path: Path) -> np.ndarray:
    """Load an RGB fundus image using OpenCV."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def apply_center_crop(image: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    """Scale and crop centrally so the retina fills the frame."""
    height, width = image.shape[:2]
    target_h, target_w = output_size
    if height == 0 or width == 0:
        raise ValueError("Fundus image has invalid dimensions")

    scale = max(target_h / height, target_w / width)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    start_y = max(0, (new_h - target_h) // 2)
    start_x = max(0, (new_w - target_w) // 2)
    end_y = start_y + target_h
    end_x = start_x + target_w
    return resized[start_y:end_y, start_x:end_x]


def apply_circular_mask(image: np.ndarray) -> np.ndarray:
    """Apply a soft circular vignette to de-emphasize frame corners."""
    height, width = image.shape[:2]
    center_y, center_x = height / 2.0, width / 2.0
    max_radius = min(center_x, center_y)
    inner_radius = max_radius * 0.98
    feather = max_radius * 0.07

    y_indices, x_indices = np.ogrid[:height, :width]
    dist = np.sqrt((x_indices - center_x) ** 2 + (y_indices - center_y) ** 2)

    mask = np.ones_like(dist, dtype=np.float32)
    outside = dist >= inner_radius
    mask[outside] = np.clip((inner_radius + feather - dist[outside]) / max(feather, 1e-6), 0.0, 1.0)

    masked = image.astype(np.float32) * mask[..., None]
    return masked.astype(image.dtype)


def preprocess_batch(paths: Iterable[Path], output_size: tuple[int, int]) -> list[np.ndarray]:
    """Preprocess a batch of images with deterministic transforms."""
    outputs: list[np.ndarray] = []
    for path in paths:
        image = load_fundus_image(path)
        cropped = apply_center_crop(image, output_size)
        masked = apply_circular_mask(cropped)
        outputs.append(masked.astype(np.float32) / 255.0)
    return outputs
