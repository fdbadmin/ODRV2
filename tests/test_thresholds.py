import numpy as np

from src.utils.metrics import optimal_thresholds


def test_thresholds_shape() -> None:
    predictions = np.random.rand(8, 7)
    targets = (predictions > 0.5).astype(int)
    thresholds = optimal_thresholds(predictions, targets)
    assert len(thresholds) == 7
