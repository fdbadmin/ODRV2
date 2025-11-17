from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd  # type: ignore[import]
import torch  # type: ignore[import]
from torch.utils.data import DataLoader, Dataset  # type: ignore[import]

from src.preprocessing.pipeline import preprocess_batch
from src.utils.label_parser import CLASS_CODES, keywords_to_multihot


@dataclass
class FundusRecord:
    left_path: Path
    right_path: Path
    labels: torch.Tensor
    age: float
    sex: int


class ODIRDataset(Dataset[dict[str, Any]]):
    """Dataset wrapping preprocessed ODIR samples."""

    def __init__(self, dataframe: pd.DataFrame, root: Path, image_size: tuple[int, int], transform: Any | None = None) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.root = root
        self.image_size = image_size
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        left_path = self.root / row["Left-Fundus"]
        right_path = self.root / row["Right-Fundus"]
        images = preprocess_batch([left_path, right_path], self.image_size)
        left_cols = [f"Left_{cls}" for cls in CLASS_CODES]
        right_cols = [f"Right_{cls}" for cls in CLASS_CODES]
        if set(left_cols).issubset(row.index) and set(right_cols).issubset(row.index):
            left_eye_labels = torch.tensor(row[left_cols].values.astype("float32"))
            right_eye_labels = torch.tensor(row[right_cols].values.astype("float32"))
        else:
            left_eye_labels = torch.from_numpy(keywords_to_multihot(row.get("Left-Diagnostic Keywords"), CLASS_CODES))
            right_eye_labels = torch.from_numpy(keywords_to_multihot(row.get("Right-Diagnostic Keywords"), CLASS_CODES))
        labels = torch.maximum(left_eye_labels, right_eye_labels)
        age = torch.tensor(row["Patient Age"], dtype=torch.float32)
        sex = torch.tensor(0 if row["Patient Sex"].lower().startswith("m") else 1, dtype=torch.float32)
        left_np = images[0]
        right_np = images[1]
        if self.transform is not None:
            left_np = self.transform(image=left_np)["image"]
            right_np = self.transform(image=right_np)["image"]
        left = torch.from_numpy(left_np.astype("float32")).permute(2, 0, 1)
        right = torch.from_numpy(right_np.astype("float32")).permute(2, 0, 1)
        return {
            "left_image": left,
            "right_image": right,
            "labels": labels,
            "left_labels": left_eye_labels,
            "right_labels": right_eye_labels,
            "age": age,
            "sex": sex,
        }


def create_dataloader(
    dataset: Dataset[Any],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    *,
    pin_memory: bool,
    persistent_workers: bool,
) -> DataLoader[Any]:
    """Build a dataloader with reproducible defaults."""
    effective_persistent = persistent_workers and num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=effective_persistent,
    )


def split_dataframe(df: pd.DataFrame, folds: Sequence[int], fold_index: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataframe into train and validation folds."""
    train_df = df[df["fold"] != folds[fold_index]]
    val_df = df[df["fold"] == folds[fold_index]]
    return train_df, val_df
