"""Validated metrics for unsupervised clustering experiments."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from classical_ml_lab.errors import MetricValidationError


def compute_clustering_metrics(
    features: npt.ArrayLike,
    cluster_labels: npt.ArrayLike,
    reference_labels: npt.ArrayLike,
    *,
    inertia: float,
) -> dict[str, float]:
    """Compute internal and external clustering metrics after validating inputs."""

    matrix = np.asarray(features, dtype=float)
    assigned = np.asarray(cluster_labels)
    reference = np.asarray(reference_labels)
    if matrix.ndim != 2 or assigned.ndim != 1 or reference.ndim != 1:
        raise MetricValidationError("Clustering inputs must be a feature matrix and label vectors.")
    if len(matrix) != len(assigned) or len(assigned) != len(reference):
        raise MetricValidationError("Features and clustering labels must have equal lengths.")
    if len(assigned) == 0:
        raise MetricValidationError("Clustering metrics require at least one sample.")
    if not np.isfinite(matrix).all():
        raise MetricValidationError("Clustering features must be finite.")
    if not np.isfinite(inertia) or inertia < 0:
        raise MetricValidationError("Clustering inertia must be finite and non-negative.")
    cluster_count = len(np.unique(assigned))
    if cluster_count < 2:
        raise MetricValidationError("Clustering metrics require at least two clusters.")
    if cluster_count >= len(assigned):
        raise MetricValidationError("Clustering metrics require fewer clusters than samples.")
    values = {
        "silhouette": float(silhouette_score(matrix, assigned)),
        "adjusted_rand_index": float(adjusted_rand_score(reference, assigned)),
        "normalized_mutual_info": float(normalized_mutual_info_score(reference, assigned)),
        "inertia": float(inertia),
    }
    if not np.isfinite(list(values.values())).all():
        raise MetricValidationError("Clustering metrics must be finite.")
    return values
