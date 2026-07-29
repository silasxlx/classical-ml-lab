from __future__ import annotations

import numpy as np
import pytest

from classical_ml_lab.clustering_metrics import compute_clustering_metrics
from classical_ml_lab.errors import MetricValidationError


@pytest.mark.unit
def test_clustering_metrics_are_finite_and_match_known_clusters() -> None:
    features = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]])
    labels = np.array([0, 0, 1, 1])
    result = compute_clustering_metrics(features, labels, labels, inertia=0.02)
    assert set(result) == {
        "silhouette",
        "adjusted_rand_index",
        "normalized_mutual_info",
        "inertia",
    }
    assert result["silhouette"] > 0.9
    assert result["adjusted_rand_index"] == pytest.approx(1.0)
    assert result["normalized_mutual_info"] == pytest.approx(1.0)
    assert result["inertia"] == pytest.approx(0.02)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("features", "labels", "reference", "inertia", "message"),
    [
        ([[0.0], [1.0]], [0, 0], [0, 1], 1.0, "at least two clusters"),
        ([[0.0], [1.0]], [0, 1], [0, 1], 1.0, "fewer clusters than samples"),
        ([[0.0], [1.0]], [0], [0, 1], 1.0, "equal lengths"),
        ([[0.0], [np.nan]], [0, 1], [0, 1], 1.0, "finite"),
        ([[0.0], [1.0]], [0, 1], [0, 1], -1.0, "non-negative"),
    ],
)
def test_clustering_metrics_reject_invalid_inputs(
    features: list[list[float]],
    labels: list[int],
    reference: list[int],
    inertia: float,
    message: str,
) -> None:
    with pytest.raises(MetricValidationError, match=message):
        compute_clustering_metrics(features, labels, reference, inertia=inertia)
