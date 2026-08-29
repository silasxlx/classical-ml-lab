"""Leakage-safe logistic-regression tuning on synthetic or local credit data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classical_ml_lab.data_quality import create_data_quality_artifact
from classical_ml_lab.experiments.common import load_credit_dataset, stratified_split
from classical_ml_lab.metrics import compute_binary_metrics
from classical_ml_lab.models import ExperimentResult

_C_GRID = [0.1, 1.0, 10.0]


def _build_l2_classifier(seed: int) -> LogisticRegression:
    """Build an L2 classifier across scikit-learn's penalty API transition.

    scikit-learn 1.8+ deprecates passing ``penalty="l2"`` explicitly and
    emits a ``FutureWarning`` that this repository treats as an error.  The
    new default plus ``l1_ratio=0.0`` has the same L2 semantics, while older
    supported releases still need the explicit penalty argument.
    """

    params: dict[str, Any] = {
        "solver": "liblinear",
        "class_weight": "balanced",
        "max_iter": 1000,
        "random_state": seed,
    }
    if LogisticRegression().get_params()["penalty"] == "deprecated":
        params["l1_ratio"] = 0.0
    else:
        params["penalty"] = "l2"
    return LogisticRegression(**params)


def build_logistic_regression_search(seed: int) -> GridSearchCV:
    """Build the public, deterministic search with preprocessing inside CV."""

    pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "classifier",
                _build_l2_classifier(seed),
            ),
        ]
    )
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    return GridSearchCV(
        pipeline,
        param_grid={"classifier__C": _C_GRID},
        scoring="roc_auc",
        cv=folds,
        n_jobs=1,
        refit=True,
    )


def run_logistic_regression(
    *, seed: int, output_dir: Path, dataset: str = "synthetic", data_path: Path | None = None
) -> ExperimentResult:
    """Tune and evaluate logistic regression on one untouched stratified holdout."""

    bundle = load_credit_dataset(dataset, data_path, seed)
    quality = create_data_quality_artifact(bundle, output_dir=output_dir)
    train_x, test_x, train_y, test_y, split = stratified_split(bundle, seed)
    search = build_logistic_regression_search(seed)
    search.fit(train_x, train_y)
    predicted = search.predict(test_x)
    scores = search.predict_proba(test_x)[:, 1]
    evaluated = compute_binary_metrics(test_y, predicted, scores)

    fitted_pipeline: Pipeline = search.best_estimator_
    classifier: LogisticRegression = fitted_pipeline.named_steps["classifier"]
    coefficients = classifier.coef_[0]
    order = np.argsort(coefficients)
    labels = bundle.features.columns.to_numpy()[order]
    values = coefficients[order]
    colors = np.where(values >= 0, "#2e7d32", "#c62828")

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / "coefficients.png"
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.barh(labels, values, color=colors)
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_title("Logistic regression standardized feature coefficients")
    axis.set_xlabel("Signed coefficient")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=140)
    plt.close(figure)

    best_params: dict[str, Any] = {
        key.removeprefix("classifier__"): value for key, value in search.best_params_.items()
    }
    selected_c = float(best_params["C"])
    return ExperimentResult(
        experiment="logistic-regression",
        task=bundle.task,
        dataset_name=bundle.name,
        dataset_source=bundle.source,
        dataset_fingerprint=bundle.fingerprint,
        split=split,
        metrics=evaluated.values,
        confusion_matrix=evaluated.confusion_matrix,
        cross_validation={
            "strategy": "StratifiedKFold",
            "folds": 3,
            "scoring": "roc_auc",
            "best_score": float(search.best_score_),
            "best_params": {"C": selected_c},
        },
        config={
            "dataset": dataset,
            "pipeline": ["SimpleImputer", "StandardScaler", "LogisticRegression"],
            "imputer": {"strategy": "median"},
            "classifier": {
                "solver": "liblinear",
                "penalty": "l2",
                "class_weight": "balanced",
                "max_iter": 1000,
                "random_state": seed,
            },
            "grid": {"C": _C_GRID},
            "selected": {"C": selected_c},
        },
        figures=("figures/coefficients.png", quality.figure_path),
        data_quality=quality.payload,
    )
