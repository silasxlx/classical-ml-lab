"""Typed data contracts shared by loaders, experiments, and artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

TaskType = Literal["binary_classification", "multiclass_classification", "regression"]
TargetUsage = Literal["training", "evaluation_only"]


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
class RegressionMetricResult:
    """Validated single-target regression metrics."""

    values: dict[str, float]


@dataclass(frozen=True)
class DataQualityArtifact:
    """Aggregate-only data-quality payload and its rendered figure."""

    payload: dict[str, Any]
    figure_path: str


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
    data_quality: dict[str, Any]

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
            "config": self.config,
        }


@dataclass(frozen=True)
class ClusteringExperimentResult:
    """Serializable result for a true unsupervised clustering experiment."""

    experiment: str
    dataset_name: str
    dataset_source: str
    dataset_fingerprint: str
    fit: dict[str, Any]
    metrics: dict[str, float]
    cluster_sizes: list[int]
    config: dict[str, Any]
    figures: tuple[str, ...]
    data_quality: dict[str, Any]
    task: Literal["clustering"] = field(default="clustering", init=False)

    def to_metrics_payload(self) -> dict[str, Any]:
        """Return the clustering-specific JSON-ready metrics contract."""

        return {
            "schema_version": "1.0",
            "experiment": self.experiment,
            "task": self.task,
            "dataset": {
                "name": self.dataset_name,
                "source": self.dataset_source,
                "fingerprint": self.dataset_fingerprint,
            },
            "fit": self.fit,
            "metrics": self.metrics,
            "cluster_sizes": self.cluster_sizes,
            "config": self.config,
        }


@dataclass(frozen=True)
class RegressionExperimentResult:
    """Serializable result for one single-target regression experiment."""

    experiment: str
    dataset_name: str
    dataset_source: str
    dataset_fingerprint: str
    split: dict[str, Any]
    metrics: dict[str, float]
    config: dict[str, Any]
    figures: tuple[str, ...]
    data_quality: dict[str, Any]
    explanation: dict[str, Any] | None = None
    explanation_path: str | None = None
    task: Literal["regression"] = field(default="regression", init=False)

    def to_metrics_payload(self) -> dict[str, Any]:
        """Return the regression-specific JSON-ready metrics contract."""

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
            "config": self.config,
        }


RunResult = ExperimentResult | ClusteringExperimentResult | RegressionExperimentResult
