"""Optional deterministic CPU CatBoost regression on Diabetes."""

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
    "loss_function": "RMSE",
    "iterations": 200,
    "depth": 3,
    "learning_rate": 0.05,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.8,
    "l2_leaf_reg": 1.0,
    "thread_count": 1,
    "verbose": False,
    "allow_writing_files": False,
    "boost_from_average": False,
}


def _import_catboost() -> ModuleType | None:
    try:
        return import_module("catboost")
    except ModuleNotFoundError:
        return None


def build_catboost_regressor(seed: int) -> Any:
    """Build the fixed CatBoostRegressor or raise an actionable dependency error."""

    module = _import_catboost()
    if module is None:
        raise InputValidationError(
            "CatBoost is optional. Run with: uv run --extra catboost ml-lab run "
            "catboost-regression"
        )
    return module.CatBoostRegressor(random_seed=seed, **_PARAMS)


def run_catboost_regression(*, seed: int, output_dir: Path) -> RegressionExperimentResult:
    """Train, evaluate, and explain optional CatBoost on the Diabetes holdout."""

    bundle = load_diabetes_regression()
    train_x, test_x, train_y, test_y, split = random_split(bundle, seed)
    model = build_catboost_regressor(seed)
    model.fit(train_x, train_y)
    predicted = np.asarray(model.predict(test_x), dtype=float)
    evaluated = compute_regression_metrics(test_y, predicted)
    quality = create_data_quality_artifact(bundle, output_dir=output_dir)
    figure_path = save_regression_diagnostics(
        test_y, predicted, output_dir=output_dir, title="CatBoost regression"
    )
    explanation, explanation_figures = create_tree_explanation(
        model,
        train_x,
        test_x,
        seed=seed,
        output_dir=output_dir,
        experiment="catboost-regression",
    )
    return RegressionExperimentResult(
        experiment="catboost-regression",
        dataset_name=bundle.name,
        dataset_source=bundle.source,
        dataset_fingerprint=bundle.fingerprint,
        split=split,
        metrics=evaluated.values,
        config={**_PARAMS, "random_seed": seed},
        figures=(figure_path, quality.figure_path, *explanation_figures),
        data_quality=quality.payload,
        explanation=explanation,
        explanation_path="explanations.json",
    )
