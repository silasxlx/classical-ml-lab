"""Experiment registry for the public CLI."""

from __future__ import annotations

from collections.abc import Callable

from classical_ml_lab.models import ExperimentResult

from .adaboost import run_adaboost
from .decision_tree import run_decision_tree
from .random_forest import run_random_forest
from .svm import run_svm

ExperimentRunner = Callable[..., ExperimentResult]

EXPERIMENT_DESCRIPTIONS = {
    "decision-tree": "Iris multiclass classification and tree visualization",
    "random-forest": "Imbalanced classification with leakage-safe tuning",
    "adaboost": "Breast-cancer classification with boosted decision stumps",
    "svm": "Scaled binary classification and decision boundary",
}

EXPERIMENT_RUNNERS: dict[str, ExperimentRunner] = {
    "decision-tree": run_decision_tree,
    "random-forest": run_random_forest,
    "adaboost": run_adaboost,
    "svm": run_svm,
}

EXPERIMENT_IDS = tuple(EXPERIMENT_RUNNERS)

__all__ = [
    "EXPERIMENT_DESCRIPTIONS",
    "EXPERIMENT_IDS",
    "EXPERIMENT_RUNNERS",
    "ExperimentResult",
]
