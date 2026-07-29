"""Offline datasets and strict validation for optional credit data."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, make_classification

from classical_ml_lab.errors import InputValidationError
from classical_ml_lab.models import DatasetBundle

CREDIT_TARGET = "SeriousDlqin2yrs"
CREDIT_FEATURES = (
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
)


def _fingerprint(features: pd.DataFrame, target: pd.Series) -> str:
    combined = features.copy()
    combined["__target__"] = target.to_numpy()
    payload = combined.to_csv(index=False, float_format="%.17g").encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def load_iris_multiclass() -> DatasetBundle:
    """Load the built-in Iris dataset without network access."""

    raw = load_iris(as_frame=True)
    features = pd.DataFrame(raw.data).copy()
    target = pd.Series(raw.target, name="target", dtype="int64")
    return DatasetBundle(
        features=features,
        target=target,
        name="iris",
        source="sklearn",
        fingerprint=_fingerprint(features, target),
        task="multiclass_classification",
    )


def load_iris_binary() -> DatasetBundle:
    """Load two Iris classes and two petal features for SVM visualization."""

    raw = load_iris(as_frame=True)
    frame = pd.DataFrame(raw.data).copy()
    target = pd.Series(raw.target, name="target", dtype="int64")
    mask = target < 2
    features = frame.loc[mask, ["petal length (cm)", "petal width (cm)"]].reset_index(drop=True)
    binary_target = target.loc[mask].reset_index(drop=True)
    return DatasetBundle(
        features=features,
        target=binary_target,
        name="iris-binary",
        source="sklearn",
        fingerprint=_fingerprint(features, binary_target),
        task="binary_classification",
    )


def load_breast_cancer_binary() -> DatasetBundle:
    """Load the built-in breast-cancer dataset for AdaBoost."""

    raw = load_breast_cancer(as_frame=True)
    features = pd.DataFrame(raw.data).copy()
    target = pd.Series(raw.target, name="target", dtype="int64")
    return DatasetBundle(
        features=features,
        target=target,
        name="breast-cancer-wisconsin-diagnostic",
        source="sklearn",
        fingerprint=_fingerprint(features, target),
        task="binary_classification",
    )


def make_synthetic_credit(seed: int) -> DatasetBundle:
    """Create deterministic, imbalanced, credit-shaped demonstration data."""

    values, labels = make_classification(
        n_samples=1_200,
        n_features=len(CREDIT_FEATURES),
        n_informative=6,
        n_redundant=2,
        weights=[0.9, 0.1],
        class_sep=1.1,
        random_state=seed,
    )
    features = pd.DataFrame(values, columns=CREDIT_FEATURES)
    target = pd.Series(labels, name=CREDIT_TARGET, dtype="int64")
    return DatasetBundle(
        features=features,
        target=target,
        name="synthetic-credit-like",
        source="generated-locally",
        fingerprint=_fingerprint(features, target),
        task="binary_classification",
    )


def _read_header(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return next(csv.reader(handle))
    except StopIteration as exc:
        raise InputValidationError("Credit CSV is empty.") from exc
    except (OSError, UnicodeError) as exc:
        raise InputValidationError(f"Cannot read credit CSV: {path.name}") from exc


def _validate_header(header: list[str]) -> None:
    normalized = [name.strip() for name in header]
    named = [name for name in normalized if name]
    duplicates = sorted(name for name, count in Counter(named).items() if count > 1)
    if duplicates:
        raise InputValidationError(f"Duplicate credit columns: {', '.join(duplicates)}")
    missing = [name for name in (*CREDIT_FEATURES, CREDIT_TARGET) if name not in named]
    if missing:
        raise InputValidationError(f"Missing credit columns: {', '.join(missing)}")


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    raw = frame[column]
    converted = pd.to_numeric(raw, errors="coerce")
    invalid = raw.notna() & converted.isna()
    if invalid.any():
        row_number = int(np.flatnonzero(invalid.to_numpy())[0]) + 2
        raise InputValidationError(
            f"Column {column!r} contains a non-numeric value at CSV row {row_number}."
        )
    return converted.astype("float64")


def load_credit_csv(path: Path) -> DatasetBundle:
    """Load a local Give Me Some Credit-shaped CSV after strict validation."""

    if not path.is_file():
        raise InputValidationError(f"Credit CSV does not exist: {path.name}")
    _validate_header(_read_header(path))
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        raise InputValidationError(f"Cannot parse credit CSV: {path.name}") from exc
    if len(frame) < 50:
        raise InputValidationError("Credit CSV must contain at least 50 data rows.")

    features = pd.DataFrame({name: _numeric_column(frame, name) for name in CREDIT_FEATURES})
    raw_target = frame[CREDIT_TARGET]
    numeric_target = pd.to_numeric(raw_target, errors="coerce")
    invalid_target = raw_target.isna() | numeric_target.isna()
    if invalid_target.any():
        row_number = int(np.flatnonzero(invalid_target.to_numpy())[0]) + 2
        raise InputValidationError(f"Target {CREDIT_TARGET!r} is invalid at CSV row {row_number}.")
    target = numeric_target.astype("int64")
    classes = set(target.unique().tolist())
    if classes != {0, 1}:
        raise InputValidationError(f"Target {CREDIT_TARGET!r} must contain both 0 and 1 only.")
    if int(target.value_counts().min()) < 5:
        raise InputValidationError("Each credit target class must contain at least five rows.")

    return DatasetBundle(
        features=features,
        target=pd.Series(target, name=CREDIT_TARGET),
        name="give-me-some-credit-local",
        source="user-provided-local-file",
        fingerprint=_fingerprint(features, target),
        task="binary_classification",
    )
