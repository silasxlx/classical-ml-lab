"""Fixed Gaussian Naive Bayes experiment on the multiclass Iris dataset."""

from __future__ import annotations

from pathlib import Path

from sklearn.naive_bayes import GaussianNB

from classical_ml_lab.data import load_iris_multiclass
from classical_ml_lab.data_quality import create_data_quality_artifact
from classical_ml_lab.experiments.common import stratified_split
from classical_ml_lab.experiments.visualization import save_confusion_matrix_figure
from classical_ml_lab.metrics import compute_multiclass_metrics
from classical_ml_lab.models import ExperimentResult

_IRIS_CLASS_NAMES = ("setosa", "versicolor", "virginica")


def build_naive_bayes() -> GaussianNB:
    """Build the public fixed-parameter Gaussian Naive Bayes model."""

    return GaussianNB(var_smoothing=1e-9)


def run_naive_bayes(*, seed: int, output_dir: Path) -> ExperimentResult:
    """Train and evaluate Gaussian Naive Bayes on one stratified holdout."""

    bundle = load_iris_multiclass()
    quality = create_data_quality_artifact(bundle, output_dir=output_dir)
    train_x, test_x, train_y, test_y, split = stratified_split(bundle, seed)
    model = build_naive_bayes()
    model.fit(train_x, train_y)
    predicted = model.predict(test_x)
    scores = model.predict_proba(test_x)
    evaluated = compute_multiclass_metrics(test_y, predicted, scores)
    figure_path = save_confusion_matrix_figure(
        evaluated.confusion_matrix,
        class_names=_IRIS_CLASS_NAMES,
        output_dir=output_dir,
        title="Gaussian Naive Bayes confusion matrix",
    )
    return ExperimentResult(
        experiment="naive-bayes",
        task=bundle.task,
        dataset_name=bundle.name,
        dataset_source=bundle.source,
        dataset_fingerprint=bundle.fingerprint,
        split=split,
        metrics=evaluated.values,
        confusion_matrix=evaluated.confusion_matrix,
        cross_validation=None,
        config={"model": "GaussianNB", "var_smoothing": 1e-9},
        figures=(figure_path, quality.figure_path),
        data_quality=quality.payload,
    )
