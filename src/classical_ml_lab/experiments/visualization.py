"""Small, headless plotting helpers shared by new experiment modules."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def save_confusion_matrix_figure(
    matrix: list[list[int]], *, class_names: tuple[str, ...], output_dir: Path, title: str
) -> str:
    """Write a compact confusion-matrix PNG and return its relative artifact path."""

    values = np.asarray(matrix, dtype=int)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / "confusion-matrix.png"
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(values, cmap="Blues")
    figure.colorbar(image, ax=axis)
    ticks = range(len(class_names))
    axis.set_xticks(ticks, labels=class_names, rotation=30, ha="right")
    axis.set_yticks(ticks, labels=class_names)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title(title)
    threshold = float(values.max()) / 2.0
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            color = "white" if values[row, column] > threshold else "black"
            axis.text(column, row, str(values[row, column]), ha="center", va="center", color=color)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=140)
    plt.close(figure)
    return "figures/confusion-matrix.png"
