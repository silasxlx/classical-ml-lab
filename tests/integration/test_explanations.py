from __future__ import annotations

import json
from importlib.util import find_spec
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from classical_ml_lab.data import load_diabetes_regression
from classical_ml_lab.experiments.common import random_split
from classical_ml_lab.experiments.explanations import select_explanation_data
from classical_ml_lab.experiments.xgboost_regression import run_xgboost_regression

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HAS_XGBOOST = find_spec("xgboost") is not None and find_spec("shap") is not None


@pytest.mark.integration
def test_explanation_background_only_uses_training_rows() -> None:
    bundle = load_diabetes_regression()
    train_x, test_x, *_ = random_split(bundle, 42)
    background, explained = select_explanation_data(train_x, test_x, seed=42)
    assert len(background) == 100
    assert len(explained) == len(test_x)
    assert set(background.index).issubset(train_x.index)
    assert set(background.index).isdisjoint(test_x.index)
    assert explained.index.equals(test_x.index)


@pytest.mark.integration
@pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost extra is not installed")
def test_xgboost_writes_schema_valid_shap_artifacts(tmp_path: Path) -> None:
    result = run_xgboost_regression(seed=42, output_dir=tmp_path)
    assert result.explanation_path == "explanations.json"
    assert result.explanation is not None
    payload = result.explanation
    schema = json.loads(
        (PROJECT_ROOT / "schemas" / "explanation.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(payload)
    assert payload["method"] == "TreeExplainer"
    assert payload["background_samples"] == 100
    assert payload["explained_samples"] == 133
    assert len(payload["feature_names"]) == 10
    assert len(payload["local_explanation"]["shap_values"]) == 10
    assert abs(payload["local_explanation"]["additivity_residual"]) <= 1e-3
    assert "feature_values" not in payload["local_explanation"]
    assert result.figures[-2:] == ("figures/shap-global.png", "figures/shap-local.png")
    for relative in result.figures[-2:]:
        assert (tmp_path / relative).stat().st_size > 0
