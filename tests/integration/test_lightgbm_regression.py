from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import pytest

from classical_ml_lab.errors import InputValidationError
from classical_ml_lab.experiments.lightgbm_regression import (
    build_lightgbm_regressor,
    run_lightgbm_regression,
)

HAS_LIGHTGBM = find_spec("lightgbm") is not None and find_spec("shap") is not None


@pytest.mark.integration
@pytest.mark.skipif(not HAS_LIGHTGBM, reason="lightgbm extra is not installed")
def test_lightgbm_has_fixed_cpu_configuration() -> None:
    model = build_lightgbm_regressor(42)
    params = model.get_params()
    assert params["objective"] == "regression"
    assert params["n_estimators"] == 200
    assert params["max_depth"] == 3
    assert params["learning_rate"] == pytest.approx(0.05)
    assert params["subsample"] == pytest.approx(0.8)
    assert params["subsample_freq"] == 1
    assert params["colsample_bytree"] == pytest.approx(0.8)
    assert params["reg_lambda"] == pytest.approx(1.0)
    assert params["random_state"] == 42
    assert params["n_jobs"] == 1


@pytest.mark.integration
@pytest.mark.skipif(not HAS_LIGHTGBM, reason="lightgbm extra is not installed")
def test_lightgbm_regression_is_reproducible_and_explained(tmp_path: Path) -> None:
    first = run_lightgbm_regression(seed=42, output_dir=tmp_path / "first")
    second = run_lightgbm_regression(seed=42, output_dir=tmp_path / "second")
    assert first.metrics == pytest.approx(second.metrics)
    assert first.metrics["r2"] >= 0.30
    assert first.split["test_samples"] == 133
    assert first.explanation is not None
    assert first.explanation["experiment"] == "lightgbm-regression"
    assert abs(first.explanation["local_explanation"]["additivity_residual"]) <= 1e-3
    assert first.figures[-2:] == ("figures/shap-global.png", "figures/shap-local.png")


@pytest.mark.integration
def test_lightgbm_missing_dependency_is_an_actionable_input_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from classical_ml_lab.experiments import lightgbm_regression

    monkeypatch.setattr(lightgbm_regression, "_import_lightgbm", lambda: None)
    with pytest.raises(InputValidationError, match=r"--extra lightgbm"):
        build_lightgbm_regressor(42)
