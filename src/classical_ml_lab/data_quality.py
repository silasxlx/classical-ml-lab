"""Aggregate-only, task-aware data-quality artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from classical_ml_lab.models import DataQualityArtifact, DatasetBundle, TargetUsage


def _numeric_summary(series: pd.Series) -> dict[str, float] | None:
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        return None
    quantiles = numeric.quantile([0.25, 0.5, 0.75])
    return {
        "min": float(numeric.min()),
        "q1": float(quantiles.loc[0.25]),
        "median": float(quantiles.loc[0.5]),
        "q3": float(quantiles.loc[0.75]),
        "max": float(numeric.max()),
    }


def _iqr_outliers(series: pd.Series) -> int:
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if numeric.empty:
        return 0
    q1, q3 = numeric.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return 0
    return int(((numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)).sum())


def _target_payload(bundle: DatasetBundle, usage: TargetUsage) -> dict[str, Any]:
    target = bundle.target
    payload: dict[str, Any] = {
        "name": str(target.name or "target"),
        "dtype": str(target.dtype),
        "usage": usage,
        "missing_count": int(target.isna().sum()),
        "unique_count": int(target.nunique(dropna=True)),
    }
    if bundle.task == "regression":
        payload["summary"] = _numeric_summary(target)
    else:
        payload["class_counts"] = {
            str(key): int(value)
            for key, value in target.value_counts(dropna=False).sort_index().items()
        }
    return payload


def _render_quality_figure(
    bundle: DatasetBundle, payload: dict[str, Any], output_dir: Path, usage: TargetUsage
) -> str:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / "data-quality.png"
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    columns = payload["columns"]
    names = [str(item["name"]) for item in columns]
    missing_rates = [float(item["missing_rate"]) for item in columns]
    axes[0].barh(names, missing_rates, color="steelblue")
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_xlabel("Missing rate")
    axes[0].set_title("Feature completeness")
    if bundle.task == "regression":
        axes[1].hist(
            bundle.target.to_numpy(dtype=float),
            bins=18,
            color="slateblue",
            edgecolor="white",
        )
        axes[1].set_ylabel("Samples")
    else:
        counts = bundle.target.value_counts().sort_index()
        axes[1].bar([str(item) for item in counts.index], counts.to_numpy(), color="slateblue")
        axes[1].set_ylabel("Samples")
    axes[1].set_xlabel(str(bundle.target.name or "target"))
    axes[1].set_title(f"Target summary ({usage})")
    figure.suptitle(f"Data quality: {bundle.name}")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=140)
    plt.close(figure)
    return "figures/data-quality.png"


def create_data_quality_artifact(
    bundle: DatasetBundle,
    *,
    output_dir: Path,
    target_usage: TargetUsage = "training",
) -> DataQualityArtifact:
    """Build a sanitized aggregate report and one headless overview figure."""

    columns: list[dict[str, Any]] = []
    warnings: set[str] = set()
    for name in bundle.features.columns:
        series = bundle.features[name]
        numeric = pd.to_numeric(series, errors="coerce")
        missing_count = int(series.isna().sum())
        finite_mask = np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))
        non_finite_count = int((~finite_mask & series.notna().to_numpy()).sum())
        unique_count = int(series.nunique(dropna=True))
        outliers = _iqr_outliers(series)
        constant = unique_count <= 1
        if missing_count:
            warnings.add("missing_values")
        if non_finite_count:
            warnings.add("non_finite_values")
        if constant:
            warnings.add("constant_columns")
        if outliers:
            warnings.add("iqr_outliers")
        columns.append(
            {
                "name": str(name),
                "dtype": str(series.dtype),
                "missing_count": missing_count,
                "missing_rate": float(missing_count / len(series)) if len(series) else 0.0,
                "non_finite_count": non_finite_count,
                "unique_count": unique_count,
                "constant": constant,
                "summary": _numeric_summary(series),
                "iqr_outlier_count": outliers,
            }
        )
    duplicates = int(bundle.features.duplicated().sum())
    if duplicates:
        warnings.add("duplicate_rows")
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "dataset": {
            "name": bundle.name,
            "source": bundle.source,
            "fingerprint": bundle.fingerprint,
        },
        "task": "clustering" if target_usage == "evaluation_only" else bundle.task,
        "shape": {"samples": len(bundle.features), "features": bundle.features.shape[1]},
        "duplicate_rows": duplicates,
        "columns": columns,
        "target": _target_payload(bundle, target_usage),
        "warnings": sorted(warnings),
    }
    figure_path = _render_quality_figure(bundle, payload, output_dir, target_usage)
    return DataQualityArtifact(payload=payload, figure_path=figure_path)
