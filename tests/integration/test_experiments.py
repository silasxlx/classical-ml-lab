from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from classical_ml_lab.experiments.adaboost import build_adaboost, run_adaboost
from classical_ml_lab.experiments.decision_tree import run_decision_tree
from classical_ml_lab.experiments.random_forest import (
    build_random_forest_search,
    run_random_forest,
)
from classical_ml_lab.experiments.svm import build_svm_pipeline, run_svm


@pytest.mark.integration
def test_decision_tree_runs_without_graphviz(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", "")
    result = run_decision_tree(seed=42, output_dir=tmp_path)
    assert result.experiment == "decision-tree"
    assert "roc_auc_ovr_macro" in result.metrics
    assert (tmp_path / result.figures[0]).stat().st_size > 0
    assert not any("graphviz" in key.lower() for key in result.config)


@pytest.mark.integration
def test_svm_pipeline_scales_before_classification_and_uses_holdout(tmp_path: Path) -> None:
    pipeline = build_svm_pipeline(42)
    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == ["scale", "classifier"]
    assert isinstance(pipeline.named_steps["scale"], StandardScaler)
    assert isinstance(pipeline.named_steps["classifier"], SVC)
    result = run_svm(seed=42, output_dir=tmp_path)
    assert result.split["test_samples"] == 30
    assert sum(map(sum, result.confusion_matrix)) == 30


@pytest.mark.integration
def test_adaboost_uses_current_estimator_api(tmp_path: Path) -> None:
    model = build_adaboost(42)
    assert isinstance(model, AdaBoostClassifier)
    assert model.estimator is not None
    assert "base_estimator" not in model.get_params(deep=False)
    result = run_adaboost(seed=42, output_dir=tmp_path)
    assert result.metrics["roc_auc"] > 0.9


@pytest.mark.integration
def test_random_forest_keeps_imputation_inside_cross_validation(tmp_path: Path) -> None:
    search = build_random_forest_search(42)
    assert isinstance(search.estimator, Pipeline)
    assert isinstance(search.estimator.named_steps["impute"], SimpleImputer)
    assert isinstance(search.estimator.named_steps["classifier"], RandomForestClassifier)
    result = run_random_forest(seed=42, output_dir=tmp_path)
    assert result.cross_validation is not None
    assert result.cross_validation["strategy"] == "StratifiedKFold"
    assert result.metrics["roc_auc"] > 0.7


@pytest.mark.integration
def test_random_forest_imputer_statistics_come_only_from_fit_data() -> None:
    train_x = pd.DataFrame(
        {
            "a": [1.0, 1.0, np.nan] * 10,
            "b": [0.0, 1.0, 2.0] * 10,
        }
    )
    train_y = pd.Series([0, 1, 0] * 10)
    search = build_random_forest_search(42)
    search.fit(train_x, train_y)
    fitted: Pipeline = search.best_estimator_
    imputer: SimpleImputer = fitted.named_steps["impute"]
    assert imputer.statistics_[0] == pytest.approx(1.0)
