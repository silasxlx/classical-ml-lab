"""Shared synthetic fixtures; no real customer or competition data is used."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from classical_ml_lab.data import CREDIT_FEATURES, CREDIT_TARGET


@pytest.fixture
def valid_credit_frame() -> pd.DataFrame:
    """Return deterministic credit-shaped data with enough minority samples."""

    generator = np.random.default_rng(42)
    frame = pd.DataFrame(
        generator.normal(size=(100, len(CREDIT_FEATURES))), columns=CREDIT_FEATURES
    )
    frame[CREDIT_TARGET] = np.array([0] * 80 + [1] * 20, dtype=int)
    frame.loc[[2, 5], "MonthlyIncome"] = np.nan
    return frame


@pytest.fixture
def credit_csv(tmp_path: Path, valid_credit_frame: pd.DataFrame) -> Path:
    """Write a valid synthetic credit CSV into pytest's temporary directory."""

    path = tmp_path / "credit.csv"
    valid_credit_frame.to_csv(path, index=False)
    return path
