from __future__ import annotations

from pathlib import Path

import pandas as pd  # type: ignore[import]


def create_patient_split(csv_path: Path, output_path: Path, seed: int = 42) -> None:
    """Create patient-level train/val/test split and persist with fold column."""
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    num_patients = df.shape[0]
    fold_size = num_patients // 5
    df["fold"] = 0
    for fold in range(5):
        start = fold * fold_size
        end = num_patients if fold == 4 else (fold + 1) * fold_size
        df.loc[start:end - 1, "fold"] = fold
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    root = Path("RAW DATA FULL/full_df.csv")
    output = Path("data/processed/odir_folds.csv")
    create_patient_split(root, output)
