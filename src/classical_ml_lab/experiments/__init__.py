"""Experiment registry for the public CLI."""

from __future__ import annotations

from collections.abc import Callable

from classical_ml_lab.models import ExperimentResult

from .adaboost import run_adaboost
from .decision_tree import run_decision_tree
from .logistic_regression import run_logistic_regression
from .random_forest import run_random_forest
from .svm import run_svm

ExperimentRunner = Callable[..., ExperimentResult]

EXPERIMENT_DESCRIPTIONS = {
    "decision-tree": "Iris multiclass classification and tree visualization",
    "random-forest": "Imbalanced classification with leakage-safe tuning",
    "adaboost": "Breast-cancer classification with boosted decision stumps",
    "svm": "Scaled binary classification and decision boundary",
    "logistic-regression": "Scaled credit classification with leakage-safe tuning",
}

EXPERIMENT_RUNNERS: dict[str, ExperimentRunner] = {
    "decision-tree": run_decision_tree,
    "random-forest": run_random_forest,
    "adaboost": run_adaboost,
    "svm": run_svm,
    "logistic-regression": run_logistic_regression,
}

EXPERIMENT_IDS = tuple(EXPERIMENT_RUNNERS)
DATASET_EXPERIMENT_IDS = frozenset({"random-forest", "logistic-regression"})

__all__ = [
    "DATASET_EXPERIMENT_IDS",
    "EXPERIMENT_DESCRIPTIONS",
    "EXPERIMENT_IDS",
    "EXPERIMENT_RUNNERS",
    "ExperimentResult",
]
