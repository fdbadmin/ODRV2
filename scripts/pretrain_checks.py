from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure headless-friendly backend and silence Albumentations' update ping.
os.environ.setdefault("MPLCONFIGDIR", str(Path.home() / ".matplotlib"))
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
plt.switch_backend("Agg")

from src.datamodules.lightning import FundusDataModule
from src.training.cli import _load_with_defaults
from scripts.build_labels import generate_eye_level_labels
from scripts.prepare_data import create_patient_split

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def unnormalize(image: np.ndarray) -> np.ndarray:
    return np.clip((image * STD) + MEAN, 0.0, 1.0)


def save_label_qc(labels_df: pd.DataFrame, folds_df: pd.DataFrame | None, qc_dir: Path) -> None:
    qc_dir.mkdir(parents=True, exist_ok=True)
    class_cols = [col for col in labels_df.columns if col.startswith("Label_")]
    class_counts = labels_df[class_cols].sum().sort_values(ascending=False)

    plt.figure(figsize=(8, 4))
    class_counts.plot(kind="bar")
    plt.ylabel("Number of samples")
    plt.title("Union Label Distribution")
    plt.tight_layout()
    plt.savefig(qc_dir / "label_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    labels_df["Patient Age"].hist(bins=30)
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.title("Patient Age Distribution")
    plt.tight_layout()
    plt.savefig(qc_dir / "age_distribution.png", dpi=200)
    plt.close()

    sex_counts = labels_df["Patient Sex"].value_counts()
    plt.figure(figsize=(4, 4))
    sex_counts.plot(kind="bar")
    plt.ylabel("Number of patients")
    plt.title("Patient Sex Distribution")
    plt.tight_layout()
    plt.savefig(qc_dir / "sex_distribution.png", dpi=200)
    plt.close()

    summary = {
        "class_counts": class_counts,
        "sex_counts": sex_counts,
    }
    if folds_df is not None and "fold" in folds_df.columns:
        fold_counts = folds_df["fold"].value_counts().sort_index()
        plt.figure(figsize=(6, 4))
        fold_counts.plot(kind="bar")
        plt.xlabel("Fold")
        plt.ylabel("Number of samples")
        plt.title("Fold Size Distribution")
        plt.tight_layout()
        plt.savefig(qc_dir / "fold_distribution.png", dpi=200)
        plt.close()
        summary["fold_counts"] = fold_counts

    summary_df = pd.concat(summary, axis=1)
    summary_df.to_csv(qc_dir / "summary_metrics.csv")


def save_preprocessed_examples(cfg, qc_dir: Path) -> None:
    dm = FundusDataModule(
        csv_path=Path(cfg.datamodule.metadata_csv),
        image_root=Path(cfg.paths.raw_data),
        image_size=tuple(cfg.datamodule.image_size),
        batch_size=cfg.datamodule.batch_size,
        num_workers=0,
        seed=cfg.project.seed,
        num_folds=cfg.datamodule.num_folds,
        val_fold=cfg.datamodule.val_fold,
        pin_memory=False,
        persistent_workers=False,
    )
    dm.setup()
    batch = next(iter(dm.val_dataloader()))
    left = batch["left_image"][:4].permute(0, 2, 3, 1).cpu().numpy()
    right = batch["right_image"][:4].permute(0, 2, 3, 1).cpu().numpy()

    fig, axes = plt.subplots(4, 2, figsize=(6, 10))
    for idx in range(4):
        axes[idx, 0].imshow(unnormalize(left[idx]))
        axes[idx, 0].axis("off")
        axes[idx, 0].set_title(f"Sample {idx} Left")
        axes[idx, 1].imshow(unnormalize(right[idx]))
        axes[idx, 1].axis("off")
        axes[idx, 1].set_title(f"Sample {idx} Right")
    plt.tight_layout()
    fig.savefig(qc_dir / "preprocessed_examples.png", dpi=200)
    plt.close(fig)


def main() -> None:
    cfg = _load_with_defaults(Path("configs/training.yaml"))
    labels_csv = Path(cfg.paths.processed_data) / "odir_eye_labels.csv"
    folds_csv = Path(cfg.paths.processed_data) / "odir_folds.csv"
    qc_dir = Path(cfg.paths.processed_data) / "qc"

    print("[1/3] Generating eye-level labels...")
    labels_df = generate_eye_level_labels(
        input_csv=Path(cfg.paths.raw_data) / "full_df.csv",
        output_csv=labels_csv,
    )

    print("[2/3] Creating fold assignments...")
    create_patient_split(labels_csv, folds_csv, seed=cfg.project.seed)
    folds_df = pd.read_csv(folds_csv)

    print("[3/3] Building QC artifacts...")
    save_label_qc(labels_df, folds_df, qc_dir)
    save_preprocessed_examples(cfg, qc_dir)
    print("QC artifacts saved to", qc_dir)


if __name__ == "__main__":
    main()
