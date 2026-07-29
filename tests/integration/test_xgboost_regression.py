from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import pytest

from classical_ml_lab.errors import InputValidationError
from classical_ml_lab.experiments.xgboost_regression import (
    build_xgboost_regressor,
    run_xgboost_regression,
)
from classical_ml_lab.runner import run_experiments

HAS_XGBOOST = find_spec("xgboost") is not None and find_spec("shap") is not None


@pytest.mark.integration
@pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost extra is not installed")
def test_xgboost_has_fixed_cpu_configuration() -> None:
    model = build_xgboost_regressor(42)
    params = model.get_params()
    assert params["objective"] == "reg:squarederror"
    assert params["tree_method"] == "hist"
    assert params["n_estimators"] == 200
    assert params["max_depth"] == 3
    assert params["learning_rate"] == pytest.approx(0.05)
    assert params["subsample"] == pytest.approx(0.8)
    assert params["colsample_bytree"] == pytest.approx(0.8)
    assert params["reg_lambda"] == pytest.approx(1.0)
    assert params["random_state"] == 42
    assert params["n_jobs"] == 1


@pytest.mark.integration
@pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost extra is not installed")
def test_xgboost_regression_is_reproducible_and_meets_contract(tmp_path: Path) -> None:
    first = run_xgboost_regression(seed=42, output_dir=tmp_path / "first")
    second = run_xgboost_regression(seed=42, output_dir=tmp_path / "second")
    assert first.metrics == pytest.approx(second.metrics)
    assert first.dataset_fingerprint == second.dataset_fingerprint
    assert first.metrics["r2"] >= 0.30
    assert first.split["test_samples"] == 133
    assert first.figures[:2] == ("figures/predictions.png", "figures/data-quality.png")


@pytest.mark.integration
def test_xgboost_missing_dependency_is_an_actionable_input_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from classical_ml_lab.experiments import xgboost_regression

    monkeypatch.setattr(xgboost_regression, "_import_xgboost", lambda: None)
    with pytest.raises(InputValidationError, match=r"--extra xgboost"):
        build_xgboost_regressor(42)


@pytest.mark.integration
def test_xgboost_missing_dependency_does_not_commit_partial_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from classical_ml_lab.experiments import xgboost_regression

    monkeypatch.setattr(xgboost_regression, "_import_xgboost", lambda: None)
    output_dir = tmp_path / "artifacts"
    with pytest.raises(InputValidationError, match=r"--extra xgboost"):
        run_experiments("xgboost-regression", output_dir=output_dir)
    assert output_dir.is_dir()
    assert list(output_dir.iterdir()) == []
