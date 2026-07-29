"""Headless diagnostics shared by single-target regression experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def save_regression_diagnostics(
    truth: npt.ArrayLike,
    predicted: npt.ArrayLike,
    *,
    output_dir: Path,
    title: str,
) -> str:
    """Write observed/predicted and residual panels."""

    observed = np.asarray(truth, dtype=float)
    estimates = np.asarray(predicted, dtype=float)
    residuals = observed - estimates
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / "predictions.png"
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(observed, estimates, alpha=0.75, edgecolors="black")
    lower = float(min(observed.min(), estimates.min()))
    upper = float(max(observed.max(), estimates.max()))
    axes[0].plot([lower, upper], [lower, upper], color="red", linestyle="--")
    axes[0].set_xlabel("Observed disease progression")
    axes[0].set_ylabel("Predicted disease progression")
    axes[0].set_title("Observed vs predicted")
    axes[1].scatter(estimates, residuals, alpha=0.75, edgecolors="black")
    axes[1].axhline(0.0, color="red", linestyle="--")
    axes[1].set_xlabel("Predicted disease progression")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residual diagnostic")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=140)
    plt.close(figure)
    return "figures/predictions.png"
