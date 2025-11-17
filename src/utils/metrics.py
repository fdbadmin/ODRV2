from __future__ import annotations

from typing import Iterable

import numpy as np  # type: ignore[import]
from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore[import]


def compute_macro_auc(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Compute macro AUC across disease classes."""
    scores = []
    for idx in range(targets.shape[1]):
        try:
            scores.append(roc_auc_score(targets[:, idx], predictions[:, idx]))
        except ValueError:
            continue
    return float(np.mean(scores))


def compute_macro_average_precision(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Compute macro average precision across classes."""
    scores = []
    for idx in range(targets.shape[1]):
        try:
            scores.append(average_precision_score(targets[:, idx], predictions[:, idx]))
        except ValueError:
            continue
    return float(np.mean(scores))


def optimal_thresholds(predictions: np.ndarray, targets: np.ndarray, grid: Iterable[float] | None = None) -> list[float]:
    """Search class-specific thresholds maximizing F1."""
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    thresholds: list[float] = []
    for idx in range(targets.shape[1]):
        best_threshold = 0.5
        best_score = -1.0
        for threshold in grid:
            preds = (predictions[:, idx] >= threshold).astype(int)
            tp = np.logical_and(preds == 1, targets[:, idx] == 1).sum()
            fp = np.logical_and(preds == 1, targets[:, idx] == 0).sum()
            fn = np.logical_and(preds == 0, targets[:, idx] == 1).sum()
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)
            if f1 > best_score:
                best_score = f1
                best_threshold = float(threshold)
        thresholds.append(best_threshold)
    return thresholds
