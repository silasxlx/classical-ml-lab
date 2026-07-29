"""Decision-tree experiment on the built-in Iris dataset."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

from classical_ml_lab.data import load_iris_multiclass
from classical_ml_lab.data_quality import create_data_quality_artifact
from classical_ml_lab.experiments.common import stratified_split
from classical_ml_lab.metrics import compute_multiclass_metrics
from classical_ml_lab.models import ExperimentResult


def run_decision_tree(*, seed: int, output_dir: Path) -> ExperimentResult:
    """Train and evaluate a deterministic Iris decision tree."""

    bundle = load_iris_multiclass()
    quality = create_data_quality_artifact(bundle, output_dir=output_dir)
    train_x, test_x, train_y, test_y, split = stratified_split(bundle, seed)
    config = {"max_depth": 4, "criterion": "gini"}
    model = DecisionTreeClassifier(max_depth=4, random_state=seed)
    model.fit(train_x, train_y)
    predicted = model.predict(test_x)
    scores = model.predict_proba(test_x)
    evaluated = compute_multiclass_metrics(test_y, predicted, scores)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / "tree.png"
    figure, axis = plt.subplots(figsize=(14, 8))
    plot_tree(
        model,
        feature_names=list(bundle.features.columns),
        class_names=["setosa", "versicolor", "virginica"],
        filled=True,
        rounded=True,
        ax=axis,
    )
    figure.tight_layout()
    figure.savefig(figure_path, dpi=140)
    plt.close(figure)

    return ExperimentResult(
        experiment="decision-tree",
        task=bundle.task,
        dataset_name=bundle.name,
        dataset_source=bundle.source,
        dataset_fingerprint=bundle.fingerprint,
        split=split,
        metrics=evaluated.values,
        confusion_matrix=evaluated.confusion_matrix,
        cross_validation=None,
        config=config,
        figures=("figures/tree.png", quality.figure_path),
        data_quality=quality.payload,
    )
