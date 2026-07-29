from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classical_ml_lab.experiments.ridge_regression import (
    build_ridge_regression_pipeline,
    run_ridge_regression,
)


@pytest.mark.integration
def test_ridge_pipeline_has_fixed_scaled_configuration() -> None:
    pipeline = build_ridge_regression_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["scale", "regressor"]
    assert isinstance(pipeline.named_steps["scale"], StandardScaler)
    regressor = pipeline.named_steps["regressor"]
    assert isinstance(regressor, Ridge)
    assert regressor.alpha == pytest.approx(1.0)


@pytest.mark.integration
def test_ridge_scaler_statistics_only_use_fit_data() -> None:
    train_x = pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [3.0, 4.0, 5.0]})
    train_y = pd.Series([10.0, 20.0, 30.0])
    pipeline = build_ridge_regression_pipeline()
    pipeline.fit(train_x, train_y)
    scaler: StandardScaler = pipeline.named_steps["scale"]
    assert scaler.mean_ == pytest.approx(train_x.mean().to_numpy())


@pytest.mark.integration
def test_ridge_scaler_is_fitted_on_only_309_training_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed_rows: list[int] = []
    original_fit = StandardScaler.fit

    def capture_fit(
        scaler: StandardScaler,
        features: pd.DataFrame,
        target: pd.Series | None = None,
        sample_weight: object = None,
    ) -> StandardScaler:
        observed_rows.append(len(features))
        return original_fit(scaler, features, target, sample_weight=sample_weight)

    monkeypatch.setattr(StandardScaler, "fit", capture_fit)
    run_ridge_regression(seed=42, output_dir=tmp_path)
    assert observed_rows == [309]


@pytest.mark.integration
def test_ridge_regression_meets_contract_and_writes_figures(tmp_path: Path) -> None:
    result = run_ridge_regression(seed=42, output_dir=tmp_path)
    assert result.experiment == "ridge-regression"
    assert result.task == "regression"
    assert result.split == {
        "strategy": "random_holdout",
        "train_samples": 309,
        "test_samples": 133,
        "test_size": 0.3,
    }
    assert result.metrics["mae"] <= 50.0
    assert result.metrics["rmse"] <= 60.0
    assert result.metrics["r2"] >= 0.35
    assert result.figures == ("figures/predictions.png", "figures/data-quality.png")
    for relative in result.figures:
        assert (tmp_path / relative).stat().st_size > 0
    assert result.data_quality["shape"] == {"samples": 442, "features": 10}
