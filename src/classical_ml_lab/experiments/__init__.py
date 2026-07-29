"""Experiment registry for the public CLI."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from classical_ml_lab.models import ExperimentResult, RunResult

from .adaboost import run_adaboost
from .decision_tree import run_decision_tree
from .kmeans import run_kmeans
from .knn import run_knn
from .logistic_regression import run_logistic_regression
from .naive_bayes import run_naive_bayes
from .random_forest import run_random_forest
from .ridge_regression import run_ridge_regression
from .svm import run_svm

ExperimentRunner = Callable[..., RunResult]

EXPERIMENT_DESCRIPTIONS = {
    "decision-tree": "Iris multiclass classification and tree visualization",
    "random-forest": "Imbalanced classification with leakage-safe tuning",
    "adaboost": "Breast-cancer classification with boosted decision stumps",
    "svm": "Scaled binary classification and decision boundary",
    "logistic-regression": "Scaled credit classification with leakage-safe tuning",
    "knn": "Scaled Iris classification with five nearest neighbors",
    "naive-bayes": "Gaussian Naive Bayes classification on Iris",
    "kmeans": "Scaled unsupervised KMeans clustering on Iris",
    "ridge-regression": "Scaled Ridge regression on the bundled Diabetes dataset",
    "xgboost-regression": "Optional CPU XGBoost regression on Diabetes",
    "lightgbm-regression": "Optional CPU LightGBM regression on Diabetes",
    "catboost-regression": "Optional CPU CatBoost regression on Diabetes",
}


def _run_xgboost_regression(*, seed: int, output_dir: Path) -> RunResult:
    from .xgboost_regression import run_xgboost_regression

    return run_xgboost_regression(seed=seed, output_dir=output_dir)


def _run_lightgbm_regression(*, seed: int, output_dir: Path) -> RunResult:
    from .lightgbm_regression import run_lightgbm_regression

    return run_lightgbm_regression(seed=seed, output_dir=output_dir)


def _run_catboost_regression(*, seed: int, output_dir: Path) -> RunResult:
    from .catboost_regression import run_catboost_regression

    return run_catboost_regression(seed=seed, output_dir=output_dir)

EXPERIMENT_RUNNERS: dict[str, ExperimentRunner] = {
    "decision-tree": run_decision_tree,
    "random-forest": run_random_forest,
    "adaboost": run_adaboost,
    "svm": run_svm,
    "logistic-regression": run_logistic_regression,
    "knn": run_knn,
    "naive-bayes": run_naive_bayes,
    "kmeans": run_kmeans,
    "ridge-regression": run_ridge_regression,
    "xgboost-regression": _run_xgboost_regression,
    "lightgbm-regression": _run_lightgbm_regression,
    "catboost-regression": _run_catboost_regression,
}

EXPERIMENT_IDS = tuple(EXPERIMENT_RUNNERS)
CORE_EXPERIMENT_IDS = EXPERIMENT_IDS[:9]
OPTIONAL_EXPERIMENT_IDS = EXPERIMENT_IDS[9:]
DATASET_EXPERIMENT_IDS = frozenset({"random-forest", "logistic-regression"})

__all__ = [
    "CORE_EXPERIMENT_IDS",
    "DATASET_EXPERIMENT_IDS",
    "EXPERIMENT_DESCRIPTIONS",
    "EXPERIMENT_IDS",
    "EXPERIMENT_RUNNERS",
    "OPTIONAL_EXPERIMENT_IDS",
    "ExperimentResult",
]
