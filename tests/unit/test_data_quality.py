from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from classical_ml_lab.data_quality import create_data_quality_artifact
from classical_ml_lab.models import DatasetBundle


@pytest.mark.unit
def test_data_quality_reports_aggregates_warnings_and_no_raw_rows(tmp_path: Path) -> None:
    features = pd.DataFrame(
        {
            "missing": [1.0, np.nan, 3.0, 100.0],
            "constant": [2.0, 2.0, 2.0, 2.0],
            "infinite": [0.0, 1.0, np.inf, 3.0],
        }
    )
    target = pd.Series([10.0, 20.0, 30.0, 40.0], name="target")
    bundle = DatasetBundle(
        features=features,
        target=target,
        name="quality-fixture",
        source="test",
        fingerprint="sha256:" + "a" * 64,
        task="regression",
    )
    artifact = create_data_quality_artifact(bundle, output_dir=tmp_path)
    payload = artifact.payload
    assert payload["shape"] == {"samples": 4, "features": 3}
    assert payload["duplicate_rows"] == 0
    assert set(payload["warnings"]) >= {"missing_values", "constant_columns", "non_finite_values"}
    columns = {item["name"]: item for item in payload["columns"]}
    assert columns["missing"]["missing_count"] == 1
    assert columns["constant"]["constant"] is True
    assert columns["infinite"]["non_finite_count"] == 1
    assert payload["target"]["usage"] == "training"
    assert artifact.figure_path == "figures/data-quality.png"
    assert (tmp_path / artifact.figure_path).stat().st_size > 0
    serialized = json.dumps(payload)
    assert "records" not in serialized
    assert all("values" not in item for item in payload["columns"])
    assert str(tmp_path.resolve()) not in serialized


@pytest.mark.unit
def test_clustering_quality_marks_reference_target_as_evaluation_only(tmp_path: Path) -> None:
    bundle = DatasetBundle(
        features=pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [2.0, 1.0, 0.0]}),
        target=pd.Series([0, 1, 1], name="species"),
        name="cluster-fixture",
        source="test",
        fingerprint="sha256:" + "b" * 64,
        task="multiclass_classification",
    )
    artifact = create_data_quality_artifact(
        bundle, output_dir=tmp_path, target_usage="evaluation_only"
    )
    assert artifact.payload["target"]["usage"] == "evaluation_only"
