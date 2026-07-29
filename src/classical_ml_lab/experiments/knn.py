"""Scaled K-nearest-neighbors experiment on the multiclass Iris dataset."""

from __future__ import annotations

from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classical_ml_lab.data import load_iris_multiclass
from classical_ml_lab.experiments.common import stratified_split
from classical_ml_lab.experiments.visualization import save_confusion_matrix_figure
from classical_ml_lab.metrics import compute_multiclass_metrics
from classical_ml_lab.models import ExperimentResult

_IRIS_CLASS_NAMES = ("setosa", "versicolor", "virginica")


def build_knn_pipeline() -> Pipeline:
    """Build the public fixed-parameter KNN Pipeline."""

    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "classifier",
                KNeighborsClassifier(
                    n_neighbors=5,
                    weights="uniform",
                    metric="minkowski",
                    p=2,
                ),
            ),
        ]
    )


def run_knn(*, seed: int, output_dir: Path) -> ExperimentResult:
    """Train and evaluate the fixed scaled KNN experiment."""

    bundle = load_iris_multiclass()
    train_x, test_x, train_y, test_y, split = stratified_split(bundle, seed)
    model = build_knn_pipeline()
    model.fit(train_x, train_y)
    predicted = model.predict(test_x)
    scores = model.predict_proba(test_x)
    evaluated = compute_multiclass_metrics(test_y, predicted, scores)
    figure_path = save_confusion_matrix_figure(
        evaluated.confusion_matrix,
        class_names=_IRIS_CLASS_NAMES,
        output_dir=output_dir,
        title="KNN confusion matrix",
    )
    return ExperimentResult(
        experiment="knn",
        task=bundle.task,
        dataset_name=bundle.name,
        dataset_source=bundle.source,
        dataset_fingerprint=bundle.fingerprint,
        split=split,
        metrics=evaluated.values,
        confusion_matrix=evaluated.confusion_matrix,
        cross_validation=None,
        config={
            "pipeline": ["StandardScaler", "KNeighborsClassifier"],
            "n_neighbors": 5,
            "weights": "uniform",
            "metric": "minkowski",
            "p": 2,
        },
        figures=(figure_path,),
    )
