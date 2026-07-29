"""Deterministic, aggregate-safe Tree SHAP artifacts for optional regressors."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from classical_ml_lab.errors import InputValidationError


def _import_shap() -> ModuleType | None:
    try:
        return import_module("shap")
    except ModuleNotFoundError:
        return None


def select_explanation_data(
    train_x: pd.DataFrame, test_x: pd.DataFrame, *, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select a deterministic training-only background and preserve the test order."""

    background_size = min(100, len(train_x))
    background = train_x.sample(n=background_size, random_state=seed).copy()
    return background, test_x.copy()


def create_tree_explanation(
    model: Any,
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    *,
    seed: int,
    output_dir: Path,
    experiment: str,
) -> tuple[dict[str, Any], tuple[str, str]]:
    """Create versioned global and local Tree SHAP explanations."""

    shap = _import_shap()
    if shap is None:
        extra = experiment.removesuffix("-regression")
        raise InputValidationError(
            f"SHAP is optional. Run this experiment with its extra: uv run --extra {extra}"
        )
    background, explained = select_explanation_data(train_x, test_x, seed=seed)
    explainer = shap.TreeExplainer(
        model,
        data=background,
        feature_perturbation="interventional",
        model_output="raw",
    )
    # SHAP's internal batch-level check is stricter than its public API permits us
    # to configure. We calculate and persist the stable local residual below.
    explanation = explainer(explained, check_additivity=False)
    values = np.asarray(explanation.values, dtype=float)
    base_values = np.asarray(explanation.base_values, dtype=float).reshape(-1)
    if values.ndim != 2 or values.shape != explained.shape:
        raise RuntimeError("Unexpected Tree SHAP output shape.")
    model_output = float(np.asarray(model.predict(explained.iloc[[0]])).reshape(-1)[0])
    base_value = float(base_values[0])
    local_values = values[0]
    residual = float(model_output - (base_value + local_values.sum()))

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    global_path = figures_dir / "shap-global.png"
    shap.plots.bar(explanation, max_display=len(explained.columns), show=False)
    global_figure = plt.gcf()
    global_figure.suptitle("Global mean absolute SHAP values")
    global_figure.tight_layout()
    global_figure.savefig(global_path, dpi=140, bbox_inches="tight")
    plt.close(global_figure)

    local_path = figures_dir / "shap-local.png"
    shap.plots.waterfall(explanation[0], max_display=len(explained.columns), show=False)
    local_figure = plt.gcf()
    local_figure.suptitle("Local Tree SHAP explanation")
    local_figure.tight_layout()
    local_figure.savefig(local_path, dpi=140, bbox_inches="tight")
    plt.close(local_figure)

    raw_index = explained.index[0]
    sample_index: int | str = (
        int(raw_index) if isinstance(raw_index, (int, np.integer)) else str(raw_index)
    )
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "experiment": experiment,
        "task": "regression",
        "method": "TreeExplainer",
        "model_output": "raw",
        "background_source": "training_only",
        "background_samples": len(background),
        "explained_samples": len(explained),
        "feature_names": [str(name) for name in explained.columns],
        "local_explanation": {
            "sample_index": sample_index,
            "base_value": base_value,
            "model_output": model_output,
            "shap_values": local_values.astype(float).tolist(),
            "additivity_residual": residual,
        },
    }
    return payload, ("figures/shap-global.png", "figures/shap-local.png")
