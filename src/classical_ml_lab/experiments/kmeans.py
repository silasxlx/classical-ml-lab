"""True unsupervised KMeans experiment on the Iris feature matrix."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classical_ml_lab.clustering_metrics import compute_clustering_metrics
from classical_ml_lab.data import load_iris_multiclass
from classical_ml_lab.models import ClusteringExperimentResult

_N_CLUSTERS = 3


def build_kmeans_pipeline(seed: int) -> Pipeline:
    """Build the public fixed-parameter scaled KMeans Pipeline."""

    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "clusterer",
                KMeans(n_clusters=_N_CLUSTERS, n_init=10, random_state=seed),
            ),
        ]
    )


def run_kmeans(*, seed: int, output_dir: Path) -> ClusteringExperimentResult:
    """Fit KMeans without target labels and evaluate the resulting partition."""

    bundle = load_iris_multiclass()
    model = build_kmeans_pipeline(seed)
    assigned = np.asarray(model.fit_predict(bundle.features), dtype=int)
    scaler: StandardScaler = model.named_steps["scale"]
    clusterer: KMeans = model.named_steps["clusterer"]
    scaled_features = scaler.transform(bundle.features)
    metrics = compute_clustering_metrics(
        scaled_features,
        assigned,
        bundle.target,
        inertia=float(clusterer.inertia_),
    )
    cluster_sizes = np.bincount(assigned, minlength=_N_CLUSTERS).astype(int).tolist()

    centers = scaler.inverse_transform(clusterer.cluster_centers_)
    first = "petal length (cm)"
    second = "petal width (cm)"
    first_index = bundle.features.columns.get_loc(first)
    second_index = bundle.features.columns.get_loc(second)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / "clusters.png"
    figure, axis = plt.subplots(figsize=(8, 6))
    points = axis.scatter(
        bundle.features[first],
        bundle.features[second],
        c=assigned,
        cmap="viridis",
        edgecolors="black",
        alpha=0.8,
    )
    axis.scatter(
        centers[:, first_index],
        centers[:, second_index],
        color="red",
        marker="X",
        s=180,
        edgecolors="white",
        label="Cluster center",
    )
    figure.colorbar(points, ax=axis, label="Predicted cluster")
    axis.set_xlabel(first)
    axis.set_ylabel(second)
    axis.set_title("KMeans clusters (2D projection of 4-feature fit)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(figure_path, dpi=140)
    plt.close(figure)

    return ClusteringExperimentResult(
        experiment="kmeans",
        dataset_name=bundle.name,
        dataset_source=bundle.source,
        dataset_fingerprint=bundle.fingerprint,
        fit={
            "strategy": "full_dataset",
            "samples": len(bundle.features),
            "features": bundle.features.shape[1],
        },
        metrics=metrics,
        cluster_sizes=cluster_sizes,
        config={
            "pipeline": ["StandardScaler", "KMeans"],
            "n_clusters": _N_CLUSTERS,
            "n_init": 10,
            "random_state": seed,
            "reference_labels_usage": "evaluation_only",
        },
        figures=("figures/clusters.png",),
    )
