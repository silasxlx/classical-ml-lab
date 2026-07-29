from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.naive_bayes import GaussianNB

from classical_ml_lab.experiments.naive_bayes import build_naive_bayes, run_naive_bayes
from classical_ml_lab.models import MetricResult


@pytest.mark.integration
def test_naive_bayes_has_one_fixed_gaussian_configuration() -> None:
    model = build_naive_bayes()
    assert isinstance(model, GaussianNB)
    assert model.var_smoothing == pytest.approx(1e-9)


@pytest.mark.integration
def test_naive_bayes_uses_probabilities_and_writes_metrics_and_figure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from classical_ml_lab.experiments import naive_bayes

    real_compute = naive_bayes.compute_multiclass_metrics
    captured: dict[str, np.ndarray] = {}

    def capture_scores(
        truth: pd.Series, predicted: np.ndarray, scores: np.ndarray
    ) -> MetricResult:
        captured["predicted"] = np.asarray(predicted)
        captured["scores"] = np.asarray(scores)
        return real_compute(truth, predicted, scores)

    monkeypatch.setattr(naive_bayes, "compute_multiclass_metrics", capture_scores)
    result = run_naive_bayes(seed=42, output_dir=tmp_path)
    assert result.experiment == "naive-bayes"
    assert result.split["test_samples"] == 45
    assert sum(map(sum, result.confusion_matrix)) == 45
    assert result.metrics["accuracy"] >= 0.85
    assert result.metrics["roc_auc_ovr_macro"] >= 0.95
    assert captured["scores"].shape == (45, 3)
    assert not np.array_equal(captured["scores"], captured["predicted"])
    assert result.figures == ("figures/confusion-matrix.png",)
    assert (tmp_path / result.figures[0]).stat().st_size > 0
