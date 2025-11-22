from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import]
import torch  # type: ignore[import]
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
        single_eye: bool = True,
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
        self.single_eye = single_eye
        self.train_df: pd.DataFrame | None = None
        self.val_df: pd.DataFrame | None = None
        self.train_transform = build_training_augmentations(image_size)  # type: ignore[arg-type]
        self.val_transform = build_validation_transforms(image_size)  # type: ignore[arg-type]

    def setup(self, stage: str | None = None) -> None:  # type: ignore[override]
        if self.val_fold >= self.num_folds:
            raise ValueError("val_fold must be less than num_folds")
        df = pd.read_csv(self.csv_path)
        
        # Assign folds by Patient ID to prevent data leakage between train/val
        if "fold" not in df.columns:
            patient_ids = df[["ID"]].drop_duplicates().reset_index(drop=True)
            patient_ids = self._assign_folds(patient_ids)
            df = df.merge(patient_ids[["ID", "fold"]], on="ID", how="left")
            
        self.train_df = df[df["fold"] != self.val_fold].copy()
        self.val_df = df[df["fold"] == self.val_fold].copy()

    def train_dataloader(self):  # type: ignore[override]
        assert self.train_df is not None
        dataset = ODIRDataset(
            self.train_df,
            self.image_root,
            self.image_size,
            transform=self.train_transform,
            single_eye=self.single_eye,
            use_eye_level_labels=True,
        )
        
        # Calculate sample weights for WeightedRandomSampler
        # We use the max weight of the active classes for each sample
        # Class weights: [1.0, 5.7, 5.6, 7.1, 11.0, 7.7, 1.5] (D, G, C, A, H, M, O)
        # Normal (N) is implicit, we give it a base weight of 1.0
        class_weights = [1.0, 5.7, 5.6, 7.1, 11.0, 7.7, 1.5]
        class_cols = [f"Label_{c}" for c in ["D", "G", "C", "A", "H", "M", "O"]]
        
        # Ensure columns exist
        available_cols = [c for c in class_cols if c in self.train_df.columns]
        if len(available_cols) != 7:
             # Fallback if columns are missing (should not happen with processed csv)
             return create_dataloader(
                dataset,
                self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
            )

        # Vectorized weight calculation
        # weights_matrix: [num_samples, num_classes]
        weights_matrix = self.train_df[class_cols].values * class_weights
        # sample_weights: [num_samples] -> max weight across classes, default 1.0 for Normal
        sample_weights = weights_matrix.max(axis=1)
        sample_weights = torch.tensor(sample_weights, dtype=torch.double)
        # Replace 0.0 (Normal) with 1.0
        sample_weights = torch.where(sample_weights == 0, torch.tensor(1.0, dtype=torch.double), sample_weights)
        
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        return create_dataloader(
            dataset,
            self.batch_size,
            shuffle=False, # Must be False when using sampler
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            sampler=sampler
        )

    def val_dataloader(self):  # type: ignore[override]
        assert self.val_df is not None
        dataset = ODIRDataset(
            self.val_df,
            self.image_root,
            self.image_size,
            transform=self.val_transform,
            single_eye=self.single_eye,
            use_eye_level_labels=True,
        )
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
