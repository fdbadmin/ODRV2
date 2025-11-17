from __future__ import annotations

from pathlib import Path

import numpy as np  # type: ignore[import]
import pandas as pd  # type: ignore[import]

from src.utils.label_parser import CLASS_CODES, keywords_to_multihot

RAW_CSV = Path("RAW DATA FULL/full_df.csv")
OUTPUT_CSV = Path("data/processed/odir_eye_labels.csv")


def _stack_vectors(series: pd.Series) -> np.ndarray:
    return np.vstack(series.apply(lambda value: keywords_to_multihot(value, CLASS_CODES)))


def generate_eye_level_labels(input_csv: Path = RAW_CSV, output_csv: Path = OUTPUT_CSV) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    left_vectors = _stack_vectors(df["Left-Diagnostic Keywords"].fillna(""))
    right_vectors = _stack_vectors(df["Right-Diagnostic Keywords"].fillna(""))
    union_vectors = np.maximum(left_vectors, right_vectors)

    left_cols = [f"Left_{cls}" for cls in CLASS_CODES]
    right_cols = [f"Right_{cls}" for cls in CLASS_CODES]
    union_cols = [f"Label_{cls}" for cls in CLASS_CODES]

    df[left_cols] = left_vectors
    df[right_cols] = right_vectors
    df[union_cols] = union_vectors

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def main() -> None:
    generate_eye_level_labels()


if __name__ == "__main__":
    main()
