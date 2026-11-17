import numpy as np

from surgphase.config import UNLABELLED
from surgphase.eval.metrics import MetricsResult, compute_metrics


def score(labels: np.ndarray, predictions: np.ndarray) -> MetricsResult:
    if labels.shape != predictions.shape:
        raise ValueError(f"labels {labels.shape} and predictions {predictions.shape} differ")
    labelled = labels != UNLABELLED
    return compute_metrics(labels[labelled], predictions[labelled])
