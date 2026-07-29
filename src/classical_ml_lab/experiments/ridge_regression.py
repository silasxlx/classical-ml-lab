"""Leakage-safe Ridge regression on the bundled Diabetes dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classical_ml_lab.data import load_diabetes_regression
from classical_ml_lab.data_quality import create_data_quality_artifact
from classical_ml_lab.experiments.common import random_split
from classical_ml_lab.experiments.regression_visualization import save_regression_diagnostics
from classical_ml_lab.metrics import compute_regression_metrics
from classical_ml_lab.models import RegressionExperimentResult


def build_ridge_regression_pipeline() -> Pipeline:
    """Build the public fixed-parameter Ridge Pipeline."""

    return Pipeline(steps=[("scale", StandardScaler()), ("regressor", Ridge(alpha=1.0))])


def run_ridge_regression(*, seed: int, output_dir: Path) -> RegressionExperimentResult:
    """Train and evaluate Ridge on one untouched Diabetes holdout."""

    bundle = load_diabetes_regression()
    train_x, test_x, train_y, test_y, split = random_split(bundle, seed)
    model = build_ridge_regression_pipeline()
    model.fit(train_x, train_y)
    predicted = np.asarray(model.predict(test_x), dtype=float)
    evaluated = compute_regression_metrics(test_y, predicted)
    quality = create_data_quality_artifact(bundle, output_dir=output_dir)

    figure_path = save_regression_diagnostics(
        test_y, predicted, output_dir=output_dir, title="Ridge regression"
    )

    return RegressionExperimentResult(
        experiment="ridge-regression",
        dataset_name=bundle.name,
        dataset_source=bundle.source,
        dataset_fingerprint=bundle.fingerprint,
        split=split,
        metrics=evaluated.values,
        config={
            "pipeline": ["StandardScaler", "Ridge"],
            "alpha": 1.0,
            "dataset_scaled_by_loader": False,
            "random_state": seed,
        },
        figures=(figure_path, quality.figure_path),
        data_quality=quality.payload,
    )
