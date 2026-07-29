from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from classical_ml_lab.data import (
    CREDIT_FEATURES,
    CREDIT_TARGET,
    load_breast_cancer_binary,
    load_credit_csv,
    load_iris_binary,
    load_iris_multiclass,
    make_synthetic_credit,
)
from classical_ml_lab.errors import InputValidationError


@pytest.mark.unit
def test_builtin_and_synthetic_datasets_are_available_offline() -> None:
    assert load_iris_multiclass().features.shape == (150, 4)
    assert load_iris_binary().features.shape == (100, 2)
    assert load_breast_cancer_binary().features.shape[1] == 30
    assert make_synthetic_credit(42).features.shape == (1200, 10)


@pytest.mark.unit
def test_credit_loader_accepts_valid_data_and_missing_predictors(credit_csv: Path) -> None:
    bundle = load_credit_csv(credit_csv)
    assert tuple(bundle.features.columns) == CREDIT_FEATURES
    assert set(bundle.target.unique()) == {0, 1}
    assert bundle.features["MonthlyIncome"].isna().sum() == 2
    assert bundle.fingerprint.startswith("sha256:")


@pytest.mark.unit
def test_credit_loader_ignores_one_unnamed_index(
    tmp_path: Path, valid_credit_frame: pd.DataFrame
) -> None:
    path = tmp_path / "indexed.csv"
    valid_credit_frame.to_csv(path, index=True)
    bundle = load_credit_csv(path)
    assert tuple(bundle.features.columns) == CREDIT_FEATURES


@pytest.mark.unit
@pytest.mark.parametrize("missing", [*CREDIT_FEATURES, CREDIT_TARGET])
def test_credit_loader_rejects_each_missing_required_column(
    tmp_path: Path, valid_credit_frame: pd.DataFrame, missing: str
) -> None:
    path = tmp_path / "missing.csv"
    valid_credit_frame.drop(columns=[missing]).to_csv(path, index=False)
    with pytest.raises(InputValidationError, match=missing):
        load_credit_csv(path)


@pytest.mark.unit
def test_credit_loader_rejects_duplicate_header(
    tmp_path: Path, valid_credit_frame: pd.DataFrame
) -> None:
    path = tmp_path / "duplicate.csv"
    valid_credit_frame.to_csv(path, index=False)
    text = path.read_text(encoding="utf-8")
    first, remainder = text.split("\n", maxsplit=1)
    names = first.split(",")
    names[1] = names[0]
    path.write_text(",".join(names) + "\n" + remainder, encoding="utf-8")
    with pytest.raises(InputValidationError, match="Duplicate credit columns"):
        load_credit_csv(path)


@pytest.mark.unit
@pytest.mark.parametrize("bad_target", [2, "not-a-label", None])
def test_credit_loader_rejects_invalid_target(
    tmp_path: Path, valid_credit_frame: pd.DataFrame, bad_target: object
) -> None:
    path = tmp_path / "target.csv"
    valid_credit_frame[CREDIT_TARGET] = valid_credit_frame[CREDIT_TARGET].astype("object")
    valid_credit_frame.loc[0, CREDIT_TARGET] = bad_target
    valid_credit_frame.to_csv(path, index=False)
    with pytest.raises(InputValidationError, match="Target"):
        load_credit_csv(path)


@pytest.mark.unit
def test_credit_loader_reports_non_numeric_field_and_row(
    tmp_path: Path, valid_credit_frame: pd.DataFrame
) -> None:
    path = tmp_path / "invalid-number.csv"
    valid_credit_frame["age"] = valid_credit_frame["age"].astype("object")
    valid_credit_frame.loc[3, "age"] = "secret-free-invalid-value"
    valid_credit_frame.to_csv(path, index=False)
    with pytest.raises(InputValidationError, match=r"'age'.*row 5"):
        load_credit_csv(path)


@pytest.mark.unit
def test_credit_loader_rejects_small_dataset(
    tmp_path: Path, valid_credit_frame: pd.DataFrame
) -> None:
    path = tmp_path / "small.csv"
    valid_credit_frame.iloc[:20].to_csv(path, index=False)
    with pytest.raises(InputValidationError, match="at least 50"):
        load_credit_csv(path)


@pytest.mark.unit
def test_credit_loader_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(InputValidationError, match="does not exist"):
        load_credit_csv(tmp_path / "missing.csv")
