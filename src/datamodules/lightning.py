from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import]
from lightning import LightningDataModule  # type: ignore[import]

from src.augmentation.fundus import build_training_augmentations, build_validation_transforms
from src.datamodules.odir import ODIRDataset, create_dataloader


class FundusDataModule(LightningDataModule):
    """LightningDataModule wrapping ODIR dataset with augmentations."""

    def __init__(
        self,
        csv_path: Path,
        image_root: Path,
        image_size: tuple[int, int],
        batch_size: int,
        num_workers: int = 8,
        seed: int = 42,
        num_folds: int = 5,
        val_fold: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
    ) -> None:
        super().__init__()
        self.csv_path = csv_path
        self.image_root = image_root
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.seed = seed
        self.num_folds = max(1, num_folds)
        self.val_fold = val_fold
        self.train_df: pd.DataFrame | None = None
        self.val_df: pd.DataFrame | None = None
        self.train_transform = build_training_augmentations(image_size)  # type: ignore[arg-type]
        self.val_transform = build_validation_transforms(image_size)  # type: ignore[arg-type]

    def setup(self, stage: str | None = None) -> None:  # type: ignore[override]
        if self.val_fold >= self.num_folds:
            raise ValueError("val_fold must be less than num_folds")
        df = pd.read_csv(self.csv_path)
        if "fold" not in df.columns:
            df = self._assign_folds(df)
        self.train_df = df[df["fold"] != self.val_fold].copy()
        self.val_df = df[df["fold"] == self.val_fold].copy()

    def train_dataloader(self):  # type: ignore[override]
        assert self.train_df is not None
        dataset = ODIRDataset(self.train_df, self.image_root, self.image_size, transform=self.train_transform)
        return create_dataloader(
            dataset,
            self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):  # type: ignore[override]
        assert self.val_df is not None
        dataset = ODIRDataset(self.val_df, self.image_root, self.image_size, transform=self.val_transform)
        return create_dataloader(
            dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def _assign_folds(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a fold column if metadata only contains eye-level labels."""
        shuffled = df.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)
        fold_size = max(1, len(shuffled) // self.num_folds)
        shuffled["fold"] = 0
        for fold in range(self.num_folds):
            start = fold * fold_size
            end = len(shuffled) if fold == self.num_folds - 1 else (fold + 1) * fold_size
            shuffled.loc[start:end - 1, "fold"] = fold
        return shuffled
