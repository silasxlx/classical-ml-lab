"""Shared, leakage-safe experiment helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from classical_ml_lab.models import DatasetBundle


def stratified_split(
    bundle: DatasetBundle, seed: int, test_size: float = 0.3
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict[str, Any]]:
    """Create one stratified holdout split and its audit metadata."""

    train_x, test_x, train_y, test_y = train_test_split(
        bundle.features,
        bundle.target,
        test_size=test_size,
        random_state=seed,
        stratify=bundle.target,
    )
    split = {
        "strategy": "stratified_holdout",
        "train_samples": len(train_y),
        "test_samples": len(test_y),
        "test_size": test_size,
    }
    return train_x, test_x, train_y, test_y, split
