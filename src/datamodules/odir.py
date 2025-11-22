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
    """Dataset wrapping single-eye ODIR samples."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        root: Path,
        image_size: tuple[int, int],
        transform: Any | None = None,
        single_eye: bool = True,
        use_eye_level_labels: bool = False,
    ) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.root = root
        self.image_size = image_size
        self.transform = transform
        self.single_eye = single_eye
        self.use_eye_level_labels = use_eye_level_labels
        self.eye_keys = ("Left", "Right")

    def __len__(self) -> int:
        if self.use_eye_level_labels:
            return len(self.df)
        if self.single_eye:
            return len(self.df) * len(self.eye_keys)
        return len(self.df)

    def _select_row(self, index: int) -> tuple[pd.Series, str]:
        if self.use_eye_level_labels:
            # In eye-level mode, the row already corresponds to a specific eye.
            # We infer the eye side from the filename for metadata consistency if needed.
            row = self.df.iloc[index]
            fname = str(row.get("filename", "")).lower()
            eye = "Left" if "left" in fname else "Right"
            return row, eye
            
        if not self.single_eye:
            return self.df.iloc[index], "Left"
        patient_idx = index // len(self.eye_keys)
        eye_idx = index % len(self.eye_keys)
        return self.df.iloc[patient_idx], self.eye_keys[eye_idx]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row, eye = self._select_row(index)
        
        if self.use_eye_level_labels:
            image_path = self.root / row["filename"]
            # Use pre-calculated one-hot columns if available
            label_cols = [f"Label_{c}" for c in CLASS_CODES]
            if set(label_cols).issubset(row.index):
                label_tensor = torch.tensor(row[label_cols].values.astype("float32"))
            else:
                # Fallback to parsing target string or keywords if needed
                # But we expect the processed CSV to have these columns
                keywords = row.get(f"{eye}-Diagnostic Keywords")
                label_tensor = torch.from_numpy(keywords_to_multihot(keywords, CLASS_CODES))
        else:
            image_column = f"{eye}-Fundus"
            eye_prefix = f"{eye}_"
            image_path = self.root / row[image_column]
            image_np = preprocess_batch([image_path], self.image_size)[0]
            class_cols = [f"{eye_prefix}{cls}" for cls in CLASS_CODES]
            if set(class_cols).issubset(row.index):
                label_tensor = torch.tensor(row[class_cols].values.astype("float32"))
            else:
                keywords = row.get(f"{eye}-Diagnostic Keywords")
                label_tensor = torch.from_numpy(keywords_to_multihot(keywords, CLASS_CODES))

        # Load image if not already loaded (use_eye_level_labels path)
        if self.use_eye_level_labels:
             image_np = preprocess_batch([image_path], self.image_size)[0]

        if self.transform is not None:
            transformed = self.transform(image=image_np)
            image_np = transformed["image"]

        if not isinstance(image_np, torch.Tensor):
            image_tensor = torch.from_numpy(image_np.astype("float32"))
            image_tensor = image_tensor.permute(2, 0, 1)
        else:
            image_tensor = image_np

        image_tensor = image_tensor.to(dtype=torch.float32)

        age = torch.tensor(row["Patient Age"], dtype=torch.float32)
        sex = torch.tensor(0 if row["Patient Sex"].lower().startswith("m") else 1, dtype=torch.float32)
        eye_idx = torch.tensor(0 if eye == "Left" else 1, dtype=torch.float32)

        return {
            "image": image_tensor,
            "labels": label_tensor,
            "age": age,
            "sex": sex,
            "eye": eye_idx,
        }


def create_dataloader(
    dataset: Dataset[Any],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    *,
    pin_memory: bool,
    persistent_workers: bool,
    sampler: Any | None = None,
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
        sampler=sampler,
    )


def split_dataframe(df: pd.DataFrame, folds: Sequence[int], fold_index: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataframe into train and validation folds."""
    train_df = df[df["fold"] != folds[fold_index]]
    val_df = df[df["fold"] == folds[fold_index]]
    return train_df, val_df
