"""Leakage-safe random-forest tuning on synthetic or local credit data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from classical_ml_lab.data import load_credit_csv, make_synthetic_credit
from classical_ml_lab.errors import InputValidationError
from classical_ml_lab.experiments.common import stratified_split
from classical_ml_lab.metrics import compute_binary_metrics
from classical_ml_lab.models import DatasetBundle, ExperimentResult


def build_random_forest_search(seed: int) -> GridSearchCV:
    """Build a small, deterministic search whose preprocessing stays inside CV."""

    pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=80,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=1,
                ),
            ),
        ]
    )
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    return GridSearchCV(
        pipeline,
        param_grid={
            "classifier__max_depth": [5, None],
            "classifier__min_samples_leaf": [1, 4],
        },
        scoring="roc_auc",
        cv=folds,
        n_jobs=1,
        refit=True,
    )


def _dataset(dataset: str, data_path: Path | None, seed: int) -> DatasetBundle:
    if dataset == "synthetic":
        if data_path is not None:
            raise InputValidationError("--data-path is only valid with --dataset credit.")
        return make_synthetic_credit(seed)
    if dataset == "credit":
        if data_path is None:
            raise InputValidationError("--data-path is required with --dataset credit.")
        return load_credit_csv(data_path)
    raise InputValidationError("Random-forest dataset must be 'synthetic' or 'credit'.")


def run_random_forest(
    *, seed: int, output_dir: Path, dataset: str = "synthetic", data_path: Path | None = None
) -> ExperimentResult:
    """Tune and evaluate random forest without exposing the final holdout to CV."""

    bundle = _dataset(dataset, data_path, seed)
    train_x, test_x, train_y, test_y, split = stratified_split(bundle, seed)
    search = build_random_forest_search(seed)
    search.fit(train_x, train_y)
    predicted = search.predict(test_x)
    scores = search.predict_proba(test_x)[:, 1]
    evaluated = compute_binary_metrics(test_y, predicted, scores)

    fitted_pipeline: Pipeline = search.best_estimator_
    classifier: RandomForestClassifier = fitted_pipeline.named_steps["classifier"]
    order = np.argsort(classifier.feature_importances_)[-10:]
    labels = bundle.features.columns.to_numpy()[order]
    values = classifier.feature_importances_[order]
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / "feature-importance.png"
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.barh(labels, values)
    axis.set_title("Random forest feature importances")
    axis.set_xlabel("Importance")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=140)
    plt.close(figure)

    best_params: dict[str, Any] = {
        key.removeprefix("classifier__"): value for key, value in search.best_params_.items()
    }
    return ExperimentResult(
        experiment="random-forest",
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
            "best_params": best_params,
        },
        config={"dataset": dataset, "n_estimators": 80, "grid": best_params},
        figures=("figures/feature-importance.png",),
    )
