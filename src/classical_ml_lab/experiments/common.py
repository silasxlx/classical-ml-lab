"""Shared, leakage-safe experiment helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from classical_ml_lab.data import load_credit_csv, make_synthetic_credit
from classical_ml_lab.errors import InputValidationError
from classical_ml_lab.models import DatasetBundle


def load_credit_dataset(dataset: str, data_path: Path | None, seed: int) -> DatasetBundle:
    """Resolve a supported offline or explicitly supplied local credit dataset."""

    if dataset == "synthetic":
        if data_path is not None:
            raise InputValidationError("--data-path is only valid with --dataset credit.")
        return make_synthetic_credit(seed)
    if dataset == "credit":
        if data_path is None:
            raise InputValidationError("--data-path is required with --dataset credit.")
        return load_credit_csv(data_path)
    raise InputValidationError("Dataset must be 'synthetic' or 'credit'.")


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


def random_split(
    bundle: DatasetBundle, seed: int, test_size: float = 0.3
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict[str, Any]]:
    """Create one deterministic non-stratified holdout for regression."""

    train_x, test_x, train_y, test_y = train_test_split(
        bundle.features,
        bundle.target,
        test_size=test_size,
        random_state=seed,
    )
    split = {
        "strategy": "random_holdout",
        "train_samples": len(train_y),
        "test_samples": len(test_y),
        "test_size": test_size,
    }
    return train_x, test_x, train_y, test_y, split
