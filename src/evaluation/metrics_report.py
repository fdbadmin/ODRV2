from __future__ import annotations

from pathlib import Path

import numpy as np  # type: ignore[import]
import pandas as pd  # type: ignore[import]

from src.utils.metrics import compute_macro_auc, compute_macro_average_precision, optimal_thresholds


def generate_report(predictions_path: Path, targets_path: Path) -> dict[str, float]:
    """Generate evaluation report from stored predictions and targets."""
    preds = np.load(predictions_path)
    targets = np.load(targets_path)
    auc = compute_macro_auc(targets, preds)
    ap = compute_macro_average_precision(targets, preds)
    thresholds = optimal_thresholds(preds, targets)
    report = {
        "macro_auc": auc,
        "macro_average_precision": ap,
    }
    for idx, threshold in enumerate(thresholds):
        report[f"threshold_class_{idx}"] = threshold
    return report


def save_report(report: dict[str, float], output_path: Path) -> None:
    """Persist metrics report as CSV."""
    df = pd.DataFrame(report.items(), columns=["metric", "value"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
