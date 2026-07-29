"""Optional deterministic CPU LightGBM regression on Diabetes."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from classical_ml_lab.data import load_diabetes_regression
from classical_ml_lab.data_quality import create_data_quality_artifact
from classical_ml_lab.errors import InputValidationError
from classical_ml_lab.experiments.common import random_split
from classical_ml_lab.experiments.explanations import create_tree_explanation
from classical_ml_lab.experiments.regression_visualization import save_regression_diagnostics
from classical_ml_lab.metrics import compute_regression_metrics
from classical_ml_lab.models import RegressionExperimentResult

_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "n_jobs": 1,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
}


def _import_lightgbm() -> ModuleType | None:
    try:
        return import_module("lightgbm")
    except ModuleNotFoundError:
        return None


def build_lightgbm_regressor(seed: int) -> Any:
    """Build the fixed LGBMRegressor or raise an actionable dependency error."""

    module = _import_lightgbm()
    if module is None:
        raise InputValidationError(
            "LightGBM is optional. Run with: uv run --extra lightgbm ml-lab run "
            "lightgbm-regression"
        )
    return module.LGBMRegressor(random_state=seed, **_PARAMS)


def run_lightgbm_regression(*, seed: int, output_dir: Path) -> RegressionExperimentResult:
    """Train, evaluate, and explain optional LightGBM on the Diabetes holdout."""

    bundle = load_diabetes_regression()
    train_x, test_x, train_y, test_y, split = random_split(bundle, seed)
    model = build_lightgbm_regressor(seed)
    model.fit(train_x, train_y)
    predicted = np.asarray(model.predict(test_x), dtype=float)
    evaluated = compute_regression_metrics(test_y, predicted)
    quality = create_data_quality_artifact(bundle, output_dir=output_dir)
    figure_path = save_regression_diagnostics(
        test_y, predicted, output_dir=output_dir, title="LightGBM regression"
    )
    explanation, explanation_figures = create_tree_explanation(
        model,
        train_x,
        test_x,
        seed=seed,
        output_dir=output_dir,
        experiment="lightgbm-regression",
    )
    return RegressionExperimentResult(
        experiment="lightgbm-regression",
        dataset_name=bundle.name,
        dataset_source=bundle.source,
        dataset_fingerprint=bundle.fingerprint,
        split=split,
        metrics=evaluated.values,
        config={**_PARAMS, "random_state": seed},
        figures=(figure_path, quality.figure_path, *explanation_figures),
        data_quality=quality.payload,
        explanation=explanation,
        explanation_path="explanations.json",
    )
