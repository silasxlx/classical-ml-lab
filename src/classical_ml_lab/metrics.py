"""Validated classification metrics used by every experiment."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from classical_ml_lab.errors import MetricValidationError
from classical_ml_lab.models import MetricResult


def _arrays(
    y_true: npt.ArrayLike, y_pred: npt.ArrayLike, y_score: npt.ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    scores = np.asarray(y_score, dtype=float)
    if truth.ndim != 1 or predicted.ndim != 1 or len(truth) != len(predicted):
        raise MetricValidationError("Targets and predictions must be equal-length vectors.")
    if len(truth) == 0 or scores.shape[0] != len(truth):
        raise MetricValidationError("Scores must contain one row per target.")
    if not np.isfinite(scores).all():
        raise MetricValidationError("Scores must be finite; NaN and Infinity are not allowed.")
    return truth, predicted, scores


def compute_binary_metrics(
    y_true: npt.ArrayLike, y_pred: npt.ArrayLike, y_score: npt.ArrayLike
) -> MetricResult:
    """Compute binary metrics from labels plus a continuous positive-class score."""

    truth, predicted, scores = _arrays(y_true, y_pred, y_score)
    if scores.ndim != 1:
        raise MetricValidationError("Binary scores must be a one-dimensional vector.")
    if set(np.unique(truth).tolist()) != {0, 1}:
        raise MetricValidationError("Binary metrics require both target classes 0 and 1.")
    values = {
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "f1": float(f1_score(truth, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, scores)),
        "pr_auc": float(average_precision_score(truth, scores)),
    }
    matrix = confusion_matrix(truth, predicted, labels=[0, 1]).astype(int).tolist()
    return MetricResult(values=values, confusion_matrix=matrix)


def compute_multiclass_metrics(
    y_true: npt.ArrayLike, y_pred: npt.ArrayLike, y_score: npt.ArrayLike
) -> MetricResult:
    """Compute macro/weighted multiclass metrics from class probabilities."""

    truth, predicted, scores = _arrays(y_true, y_pred, y_score)
    classes = np.unique(truth)
    if len(classes) < 3 or scores.ndim != 2 or scores.shape[1] != len(classes):
        raise MetricValidationError("Multiclass scores must provide one column per observed class.")
    encoded = label_binarize(truth, classes=classes)
    values = {
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "precision_macro": float(
            precision_score(truth, predicted, average="macro", zero_division=0)
        ),
        "precision_weighted": float(
            precision_score(truth, predicted, average="weighted", zero_division=0)
        ),
        "recall_macro": float(recall_score(truth, predicted, average="macro", zero_division=0)),
        "recall_weighted": float(
            recall_score(truth, predicted, average="weighted", zero_division=0)
        ),
        "f1_macro": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(truth, predicted, average="weighted", zero_division=0)),
        "roc_auc_ovr_macro": float(
            roc_auc_score(truth, scores, multi_class="ovr", average="macro")
        ),
        "pr_auc_macro": float(average_precision_score(encoded, scores, average="macro")),
    }
    matrix = confusion_matrix(truth, predicted, labels=classes).astype(int).tolist()
    return MetricResult(values=values, confusion_matrix=matrix)
