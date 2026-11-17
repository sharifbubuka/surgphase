from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
)


@dataclass
class MetricsResult:
    accuracy: float
    f1_macro: float
    iou: float
    precision: float
    recall: float
    transition_error: float
    temporal_stability: float
    edit_score: float
    f1_at_10: float
    f1_at_25: float
    f1_at_50: float

    def __repr__(self) -> str:
        return (
            f"MetricsResult(\n"
            f"  accuracy={self.accuracy:.4f},\n"
            f"  f1_macro={self.f1_macro:.4f},\n"
            f"  iou={self.iou:.4f},\n"
            f"  precision={self.precision:.4f},\n"
            f"  recall={self.recall:.4f},\n"
            f"  transition_error={self.transition_error:.4f},\n"
            f"  temporal_stability={self.temporal_stability:.4f},\n"
            f"  edit_score={self.edit_score:.4f},\n"
            f"  f1_at_10={self.f1_at_10:.4f},\n"
            f"  f1_at_25={self.f1_at_25:.4f},\n"
            f"  f1_at_50={self.f1_at_50:.4f}\n"
            f")"
        )


class Metric(Protocol):
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float: ...


class Accuracy:
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return accuracy_score(y_true, y_pred)


class Precision:
    def __init__(self, average: str = "macro"):
        self.average = average

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return precision_score(y_true, y_pred, average=self.average, zero_division=0)


class Recall:
    def __init__(self, average: str = "macro"):
        self.average = average

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return recall_score(y_true, y_pred, average=self.average, zero_division=0)


class F1Macro:
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return f1_score(y_true, y_pred, average="macro", zero_division=0)


class IntersectionOverUnion:
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return jaccard_score(y_true, y_pred, average="macro", zero_division=0)


class TransitionError:
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        true_transitions = np.sum(np.diff(y_true) != 0)
        pred_transitions = np.sum(np.diff(y_pred) != 0)

        if true_transitions == 0:
            return 0.0 if pred_transitions == 0 else 1.0

        transition_diff = np.abs(true_transitions - pred_transitions)
        return float(transition_diff / true_transitions)


class TemporalStability:
    def __init__(self, window_size: int = 5):
        self.window_size = window_size

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if len(y_pred) < self.window_size:
            return float(np.mean(y_true == y_pred))

        stabilities = []
        for i in range(len(y_pred) - self.window_size + 1):
            window = y_pred[i : i + self.window_size]
            window_true = y_true[i : i + self.window_size]
            matches = np.sum(window == window_true)
            stabilities.append(matches / self.window_size)

        return float(np.mean(stabilities))


def segments(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    if labels.size == 0:
        empty = np.array([], dtype=int)
        return labels[:0], empty, empty
    starts = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
    ends = np.r_[starts[1:], labels.size]
    return labels[starts], starts, ends


class EditScore:
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        true_segments, _, _ = segments(y_true)
        pred_segments, _, _ = segments(y_pred)
        longest = max(len(true_segments), len(pred_segments))
        if longest == 0:
            return 1.0
        previous = np.arange(len(true_segments) + 1)
        for i, pred_label in enumerate(pred_segments, start=1):
            current = np.empty_like(previous)
            current[0] = i
            substitution = previous[:-1] + (true_segments != pred_label)
            for j in range(1, len(previous)):
                current[j] = min(previous[j] + 1, current[j - 1] + 1, substitution[j - 1])
            previous = current
        return float(1 - previous[-1] / longest)


class SegmentalF1:
    def __init__(self, overlap: float):
        self.overlap = overlap

    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        true_labels, true_starts, true_ends = segments(y_true)
        pred_labels, pred_starts, pred_ends = segments(y_pred)
        if len(true_labels) == 0 and len(pred_labels) == 0:
            return 1.0
        matched = np.zeros(len(true_labels), dtype=bool)
        true_positives = 0
        for label, start, end in zip(pred_labels, pred_starts, pred_ends, strict=True):
            intersection = np.minimum(end, true_ends) - np.maximum(start, true_starts)
            union = np.maximum(end, true_ends) - np.minimum(start, true_starts)
            iou = np.where(true_labels == label, intersection.clip(min=0) / union, 0.0)
            best = int(iou.argmax()) if iou.size else -1
            if best >= 0 and iou[best] >= self.overlap and not matched[best]:
                matched[best] = True
                true_positives += 1
        false_positives = len(pred_labels) - true_positives
        false_negatives = len(true_labels) - true_positives
        denominator = 2 * true_positives + false_positives + false_negatives
        return float(2 * true_positives / denominator) if denominator else 0.0


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    temporal_stability_window: int = 5,
) -> MetricsResult:
    return MetricsResult(
        accuracy=Accuracy().compute(y_true, y_pred),
        f1_macro=F1Macro().compute(y_true, y_pred),
        iou=IntersectionOverUnion().compute(y_true, y_pred),
        precision=Precision().compute(y_true, y_pred),
        recall=Recall().compute(y_true, y_pred),
        transition_error=TransitionError().compute(y_true, y_pred),
        temporal_stability=TemporalStability(temporal_stability_window).compute(y_true, y_pred),
        edit_score=EditScore().compute(y_true, y_pred),
        f1_at_10=SegmentalF1(0.10).compute(y_true, y_pred),
        f1_at_25=SegmentalF1(0.25).compute(y_true, y_pred),
        f1_at_50=SegmentalF1(0.50).compute(y_true, y_pred),
    )
