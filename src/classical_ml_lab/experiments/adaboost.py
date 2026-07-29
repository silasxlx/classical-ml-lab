"""AdaBoost experiment using the current scikit-learn estimator API."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from classical_ml_lab.data import load_breast_cancer_binary
from classical_ml_lab.data_quality import create_data_quality_artifact
from classical_ml_lab.experiments.common import stratified_split
from classical_ml_lab.metrics import compute_binary_metrics
from classical_ml_lab.models import ExperimentResult


def build_adaboost(seed: int) -> AdaBoostClassifier:
    """Build AdaBoost with a current ``estimator`` parameter."""

    stump = DecisionTreeClassifier(max_depth=1, random_state=seed)
    return AdaBoostClassifier(
        estimator=stump,
        n_estimators=50,
        learning_rate=0.5,
        random_state=seed,
    )


def run_adaboost(*, seed: int, output_dir: Path) -> ExperimentResult:
    """Train and evaluate AdaBoost on the built-in breast-cancer data."""

    bundle = load_breast_cancer_binary()
    quality = create_data_quality_artifact(bundle, output_dir=output_dir)
    train_x, test_x, train_y, test_y, split = stratified_split(bundle, seed)
    model = build_adaboost(seed)
    model.fit(train_x, train_y)
    predicted = model.predict(test_x)
    scores = model.predict_proba(test_x)[:, 1]
    evaluated = compute_binary_metrics(test_y, predicted, scores)

    order = np.argsort(model.feature_importances_)[-10:]
    labels = bundle.features.columns.to_numpy()[order]
    values = model.feature_importances_[order]
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / "feature-importance.png"
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.barh(labels, values)
    axis.set_title("AdaBoost top feature importances")
    axis.set_xlabel("Importance")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=140)
    plt.close(figure)

    return ExperimentResult(
        experiment="adaboost",
        task=bundle.task,
        dataset_name=bundle.name,
        dataset_source=bundle.source,
        dataset_fingerprint=bundle.fingerprint,
        split=split,
        metrics=evaluated.values,
        confusion_matrix=evaluated.confusion_matrix,
        cross_validation=None,
        config={"estimator": "DecisionTreeClassifier(max_depth=1)", "n_estimators": 50},
        figures=("figures/feature-importance.png", quality.figure_path),
        data_quality=quality.payload,
    )
