"""Typed data contracts shared by loaders, experiments, and artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

TaskType = Literal["binary_classification", "multiclass_classification"]


@dataclass(frozen=True)
class DatasetBundle:
    """A validated feature matrix and target with audit metadata."""

    features: pd.DataFrame
    target: pd.Series
    name: str
    source: str
    fingerprint: str
    task: TaskType


@dataclass(frozen=True)
class MetricResult:
    """Validated classification metrics and confusion matrix."""

    values: dict[str, float]
    confusion_matrix: list[list[int]]


@dataclass(frozen=True)
class ExperimentResult:
    """Serializable result returned by every experiment runner."""

    experiment: str
    task: TaskType
    dataset_name: str
    dataset_source: str
    dataset_fingerprint: str
    split: dict[str, Any]
    metrics: dict[str, float]
    confusion_matrix: list[list[int]]
    cross_validation: dict[str, Any] | None
    config: dict[str, Any]
    figures: tuple[str, ...]

    def to_metrics_payload(self) -> dict[str, Any]:
        """Return the versioned JSON-ready metrics contract."""

        return {
            "schema_version": "1.0",
            "experiment": self.experiment,
            "task": self.task,
            "dataset": {
                "name": self.dataset_name,
                "source": self.dataset_source,
                "fingerprint": self.dataset_fingerprint,
            },
            "split": self.split,
            "metrics": self.metrics,
            "confusion_matrix": self.confusion_matrix,
            "cross_validation": self.cross_validation,
        }
