from __future__ import annotations

import numpy as np
import pytest

from classical_ml_lab.errors import MetricValidationError
from classical_ml_lab.metrics import (
    compute_binary_metrics,
    compute_multiclass_metrics,
    compute_regression_metrics,
)


@pytest.mark.unit
def test_binary_metrics_have_required_keys_and_use_continuous_scores() -> None:
    truth = np.array([0, 0, 1, 1])
    predicted = np.array([0, 1, 0, 1])
    weak_scores = np.array([0.1, 0.9, 0.8, 0.7])
    strong_scores = np.array([0.1, 0.2, 0.8, 0.9])
    weak = compute_binary_metrics(truth, predicted, weak_scores)
    strong = compute_binary_metrics(truth, predicted, strong_scores)
    assert set(strong.values) == {
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    }
    assert strong.values["roc_auc"] > weak.values["roc_auc"]
    assert sum(map(sum, strong.confusion_matrix)) == len(truth)


@pytest.mark.unit
def test_multiclass_metrics_have_macro_weighted_and_ovr_scores() -> None:
    truth = np.array([0, 0, 1, 1, 2, 2])
    predicted = np.array([0, 0, 1, 2, 2, 2])
    scores = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.4, 0.5],
            [0.05, 0.05, 0.9],
            [0.1, 0.1, 0.8],
        ]
    )
    result = compute_multiclass_metrics(truth, predicted, scores)
    assert "roc_auc_ovr_macro" in result.values
    assert "pr_auc_macro" in result.values
    assert "f1_weighted" in result.values
    assert sum(map(sum, result.confusion_matrix)) == len(truth)


@pytest.mark.unit
def test_binary_metrics_reject_single_class() -> None:
    with pytest.raises(MetricValidationError, match="both target classes"):
        compute_binary_metrics([0, 0], [0, 0], [0.1, 0.2])


@pytest.mark.unit
@pytest.mark.parametrize("bad", [[0.1, np.nan], [0.1, np.inf]])
def test_metrics_reject_non_finite_scores(bad: list[float]) -> None:
    with pytest.raises(MetricValidationError, match="finite"):
        compute_binary_metrics([0, 1], [0, 1], bad)


@pytest.mark.unit
def test_metrics_reject_shape_mismatch() -> None:
    with pytest.raises(MetricValidationError, match="one row per target"):
        compute_binary_metrics([0, 1], [0, 1], [0.2])


@pytest.mark.unit
def test_regression_metrics_have_expected_values_and_allow_negative_r2() -> None:
    result = compute_regression_metrics([1.0, 2.0, 3.0], [3.0, 3.0, 3.0])
    assert set(result.values) == {"mae", "rmse", "r2"}
    assert result.values["mae"] == pytest.approx(1.0)
    assert result.values["rmse"] == pytest.approx((5.0 / 3.0) ** 0.5)
    assert result.values["r2"] < 0.0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("truth", "predicted", "message"),
    [
        ([], [], "at least two"),
        ([1.0, 2.0], [1.0], "equal-length"),
        ([1.0, np.nan], [1.0, 2.0], "finite"),
        ([1.0, 2.0], [1.0, np.inf], "finite"),
        ([1.0, 1.0], [1.0, 1.0], "constant"),
    ],
)
def test_regression_metrics_reject_invalid_inputs(
    truth: list[float], predicted: list[float], message: str
) -> None:
    with pytest.raises(MetricValidationError, match=message):
        compute_regression_metrics(truth, predicted)
