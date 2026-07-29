from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import pytest

from classical_ml_lab.errors import InputValidationError
from classical_ml_lab.experiments.catboost_regression import (
    build_catboost_regressor,
    run_catboost_regression,
)

HAS_CATBOOST = find_spec("catboost") is not None and find_spec("shap") is not None


@pytest.mark.integration
@pytest.mark.skipif(not HAS_CATBOOST, reason="catboost extra is not installed")
def test_catboost_has_fixed_cpu_configuration() -> None:
    model = build_catboost_regressor(42)
    params = model.get_params()
    assert params["loss_function"] == "RMSE"
    assert params["iterations"] == 200
    assert params["depth"] == 3
    assert params["learning_rate"] == pytest.approx(0.05)
    assert params["bootstrap_type"] == "Bernoulli"
    assert params["subsample"] == pytest.approx(0.8)
    assert params["l2_leaf_reg"] == pytest.approx(1.0)
    assert params["random_seed"] == 42
    assert params["thread_count"] == 1
    assert params["allow_writing_files"] is False
    assert params["boost_from_average"] is False


@pytest.mark.integration
@pytest.mark.skipif(not HAS_CATBOOST, reason="catboost extra is not installed")
def test_catboost_regression_is_reproducible_and_explained(tmp_path: Path) -> None:
    first = run_catboost_regression(seed=42, output_dir=tmp_path / "first")
    second = run_catboost_regression(seed=42, output_dir=tmp_path / "second")
    assert first.metrics == pytest.approx(second.metrics)
    assert first.metrics["r2"] >= 0.30
    assert first.split["test_samples"] == 133
    assert first.explanation is not None
    assert first.explanation["experiment"] == "catboost-regression"
    assert abs(first.explanation["local_explanation"]["additivity_residual"]) <= 1e-3
    assert first.figures[-2:] == ("figures/shap-global.png", "figures/shap-local.png")


@pytest.mark.integration
def test_catboost_missing_dependency_is_an_actionable_input_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from classical_ml_lab.experiments import catboost_regression

    monkeypatch.setattr(catboost_regression, "_import_catboost", lambda: None)
    with pytest.raises(InputValidationError, match=r"--extra catboost"):
        build_catboost_regressor(42)
