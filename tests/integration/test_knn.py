from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classical_ml_lab.experiments.knn import build_knn_pipeline, run_knn
from classical_ml_lab.models import MetricResult


@pytest.mark.integration
def test_knn_pipeline_has_fixed_scaled_configuration() -> None:
    pipeline = build_knn_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["scale", "classifier"]
    assert isinstance(pipeline.named_steps["scale"], StandardScaler)
    classifier = pipeline.named_steps["classifier"]
    assert isinstance(classifier, KNeighborsClassifier)
    assert classifier.n_neighbors == 5
    assert classifier.weights == "uniform"
    assert classifier.metric == "minkowski"
    assert classifier.p == 2


@pytest.mark.integration
def test_knn_scaler_statistics_only_use_fit_data() -> None:
    train_x = pd.DataFrame(
        {
            "a": [0.0, 1.0, 2.0] * 12,
            "b": [2.0, 1.0, 0.0] * 12,
            "c": [1.0, 2.0, 3.0] * 12,
            "d": [3.0, 2.0, 1.0] * 12,
        }
    )
    train_y = pd.Series([0, 1, 2] * 12)
    pipeline = build_knn_pipeline()
    pipeline.fit(train_x, train_y)
    scaler: StandardScaler = pipeline.named_steps["scale"]
    assert scaler.mean_ == pytest.approx(train_x.mean().to_numpy())


@pytest.mark.integration
def test_knn_uses_probabilities_and_writes_metrics_and_figure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from classical_ml_lab.experiments import knn

    real_compute = knn.compute_multiclass_metrics
    captured: dict[str, np.ndarray] = {}

    def capture_scores(
        truth: pd.Series, predicted: np.ndarray, scores: np.ndarray
    ) -> MetricResult:
        captured["predicted"] = np.asarray(predicted)
        captured["scores"] = np.asarray(scores)
        return real_compute(truth, predicted, scores)

    monkeypatch.setattr(knn, "compute_multiclass_metrics", capture_scores)
    result = run_knn(seed=42, output_dir=tmp_path)
    assert result.experiment == "knn"
    assert result.split["test_samples"] == 45
    assert sum(map(sum, result.confusion_matrix)) == 45
    assert result.metrics["accuracy"] >= 0.85
    assert result.metrics["roc_auc_ovr_macro"] >= 0.95
    assert captured["scores"].shape == (45, 3)
    assert not np.array_equal(captured["scores"], captured["predicted"])
    assert result.figures == (
        "figures/confusion-matrix.png",
        "figures/data-quality.png",
    )
    assert (tmp_path / result.figures[0]).stat().st_size > 0
