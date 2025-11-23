from __future__ import annotations

from pathlib import Path

from src.evaluation.metrics_report import generate_report, save_report


if __name__ == "__main__":
    predictions = Path("data/processed/predictions.npy")
    targets = Path("data/processed/targets.npy")
    report = generate_report(predictions, targets)
    save_report(report, Path("logs/metrics_report.csv"))
