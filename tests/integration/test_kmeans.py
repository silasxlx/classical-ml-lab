from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from jsonschema import Draft202012Validator, ValidationError
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classical_ml_lab.experiments.kmeans import build_kmeans_pipeline, run_kmeans
from classical_ml_lab.runner import run_experiments

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
def test_kmeans_pipeline_has_fixed_scaled_configuration() -> None:
    pipeline = build_kmeans_pipeline(42)
    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["scale", "clusterer"]
    assert isinstance(pipeline.named_steps["scale"], StandardScaler)
    clusterer = pipeline.named_steps["clusterer"]
    assert isinstance(clusterer, KMeans)
    assert clusterer.n_clusters == 3
    assert clusterer.n_init == 10
    assert clusterer.random_state == 42


@pytest.mark.integration
def test_kmeans_fit_predict_never_receives_reference_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from classical_ml_lab.experiments import kmeans

    real_pipeline = build_kmeans_pipeline(42)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class PipelineSpy:
        def fit_predict(self, *args: object, **kwargs: object) -> object:
            calls.append((args, kwargs))
            return real_pipeline.fit_predict(*args, **kwargs)

        def __getattr__(self, name: str) -> object:
            return getattr(real_pipeline, name)

    monkeypatch.setattr(kmeans, "build_kmeans_pipeline", lambda _: PipelineSpy())
    run_kmeans(seed=42, output_dir=tmp_path)
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert len(args) == 1
    assert isinstance(args[0], pd.DataFrame)
    assert kwargs == {}


@pytest.mark.integration
def test_kmeans_writes_true_clustering_metrics_sizes_and_projection(tmp_path: Path) -> None:
    result = run_kmeans(seed=42, output_dir=tmp_path)
    assert result.experiment == "kmeans"
    assert result.task == "clustering"
    assert result.fit == {"strategy": "full_dataset", "samples": 150, "features": 4}
    assert result.metrics["silhouette"] >= 0.40
    assert result.metrics["adjusted_rand_index"] >= 0.55
    assert result.metrics["normalized_mutual_info"] >= 0.60
    assert result.metrics["inertia"] > 0.0
    assert len(result.cluster_sizes) == 3
    assert min(result.cluster_sizes) > 0
    assert sum(result.cluster_sizes) == 150
    assert result.config["reference_labels_usage"] == "evaluation_only"
    assert result.figures == ("figures/clusters.png", "figures/data-quality.png")
    assert (tmp_path / result.figures[0]).stat().st_size > 0


@pytest.mark.integration
def test_kmeans_artifact_uses_dedicated_schema_and_is_reproducible(tmp_path: Path) -> None:
    first = run_experiments("kmeans", seed=42, output_dir=tmp_path)
    second = run_experiments("kmeans", seed=42, output_dir=tmp_path)
    first_manifest = json.loads((first / "run.json").read_text(encoding="utf-8"))
    second_manifest = json.loads((second / "run.json").read_text(encoding="utf-8"))
    assert first_manifest["config_hash"] == second_manifest["config_hash"]
    first_metrics = json.loads(
        (first / "experiments" / "kmeans" / "metrics.json").read_text(encoding="utf-8")
    )
    second_metrics = json.loads(
        (second / "experiments" / "kmeans" / "metrics.json").read_text(encoding="utf-8")
    )
    assert first_metrics == second_metrics
    assert "split" not in first_metrics
    assert "confusion_matrix" not in first_metrics
    schema = json.loads(
        (PROJECT_ROOT / "schemas" / "clustering.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(first_metrics)
    classification_schema = json.loads(
        (PROJECT_ROOT / "schemas" / "metrics.schema.json").read_text(encoding="utf-8")
    )
    with pytest.raises(ValidationError):
        Draft202012Validator(classification_schema).validate(first_metrics)
