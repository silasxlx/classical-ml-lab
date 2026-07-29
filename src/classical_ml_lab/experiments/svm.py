"""Support-vector-machine experiment with scaling inside the Pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from classical_ml_lab.data import load_iris_binary
from classical_ml_lab.experiments.common import stratified_split
from classical_ml_lab.metrics import compute_binary_metrics
from classical_ml_lab.models import ExperimentResult


def build_svm_pipeline(seed: int) -> Pipeline:
    """Build the public, leakage-safe SVM Pipeline."""

    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("classifier", SVC(C=1.0, kernel="rbf", gamma="scale", random_state=seed)),
        ]
    )


def run_svm(*, seed: int, output_dir: Path) -> ExperimentResult:
    """Train and evaluate the scaled binary SVM experiment."""

    bundle = load_iris_binary()
    train_x, test_x, train_y, test_y, split = stratified_split(bundle, seed)
    model = build_svm_pipeline(seed)
    model.fit(train_x, train_y)
    predicted = model.predict(test_x)
    scores = model.decision_function(test_x)
    evaluated = compute_binary_metrics(test_y, predicted, scores)

    first, second = bundle.features.columns
    x_min, x_max = float(bundle.features[first].min()), float(bundle.features[first].max())
    y_min, y_max = float(bundle.features[second].min()), float(bundle.features[second].max())
    grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, 160), np.linspace(y_min, y_max, 160))
    grid = pd.DataFrame({first: grid_x.ravel(), second: grid_y.ravel()}, columns=[first, second])
    boundary = model.predict(grid).reshape(grid_x.shape)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / "decision-boundary.png"
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.contourf(grid_x, grid_y, boundary, alpha=0.25, cmap="coolwarm")
    axis.scatter(
        bundle.features[first],
        bundle.features[second],
        c=bundle.target,
        edgecolors="black",
        cmap="coolwarm",
    )
    axis.set_xlabel(str(first))
    axis.set_ylabel(str(second))
    axis.set_title("Scaled SVM decision boundary")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=140)
    plt.close(figure)

    return ExperimentResult(
        experiment="svm",
        task=bundle.task,
        dataset_name=bundle.name,
        dataset_source=bundle.source,
        dataset_fingerprint=bundle.fingerprint,
        split=split,
        metrics=evaluated.values,
        confusion_matrix=evaluated.confusion_matrix,
        cross_validation=None,
        config={"pipeline": ["StandardScaler", "SVC"], "C": 1.0, "kernel": "rbf"},
        figures=("figures/decision-boundary.png",),
    )
