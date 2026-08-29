from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classical_ml_lab.data import CREDIT_FEATURES, make_synthetic_credit
from classical_ml_lab.experiments.common import stratified_split
from classical_ml_lab.experiments.logistic_regression import (
    build_logistic_regression_search,
    run_logistic_regression,
)
from classical_ml_lab.runner import run_experiments


@pytest.mark.integration
def test_req_exp_lr_001_builds_the_exact_leakage_safe_search() -> None:
    search = build_logistic_regression_search(42)
    assert isinstance(search, GridSearchCV)
    assert isinstance(search.estimator, Pipeline)
    assert list(search.estimator.named_steps) == ["impute", "scale", "classifier"]
    assert isinstance(search.estimator.named_steps["impute"], SimpleImputer)
    assert search.estimator.named_steps["impute"].strategy == "median"
    assert isinstance(search.estimator.named_steps["scale"], StandardScaler)
    classifier = search.estimator.named_steps["classifier"]
    assert isinstance(classifier, LogisticRegression)
    assert classifier.solver == "liblinear"
    # scikit-learn 1.8+ represents the default L2 penalty as ``deprecated``
    # and uses l1_ratio=0.0; older supported versions expose ``penalty=l2``.
    assert classifier.penalty in {"l2", "deprecated"}
    if classifier.penalty == "deprecated":
        assert classifier.l1_ratio == 0.0
    assert classifier.class_weight == "balanced"
    assert classifier.max_iter == 1000
    assert classifier.random_state == 42
    assert search.param_grid == {"classifier__C": [0.1, 1.0, 10.0]}
    assert isinstance(search.cv, StratifiedKFold)
    assert search.cv.n_splits == 3
    assert search.cv.shuffle is True
    assert search.cv.random_state == 42
    assert search.scoring == "roc_auc"
    assert search.refit is True
    assert search.n_jobs == 1


@pytest.mark.integration
def test_req_exp_lr_001_preprocessing_statistics_only_use_fit_data() -> None:
    generator = np.random.default_rng(42)
    train_x = pd.DataFrame(generator.normal(size=(60, 10)), columns=CREDIT_FEATURES)
    train_x.loc[[0, 1], CREDIT_FEATURES[0]] = np.nan
    train_y = pd.Series([0, 1] * 30)
    search = build_logistic_regression_search(42)
    search.fit(train_x, train_y)
    fitted: Pipeline = search.best_estimator_
    imputer: SimpleImputer = fitted.named_steps["impute"]
    scaler: StandardScaler = fitted.named_steps["scale"]
    expected_median = float(train_x[CREDIT_FEATURES[0]].median())
    expected_mean = float(train_x[CREDIT_FEATURES[0]].fillna(expected_median).mean())
    assert imputer.statistics_[0] == pytest.approx(expected_median)
    assert scaler.mean_[0] == pytest.approx(expected_mean)


@pytest.mark.integration
def test_req_exp_lr_002_search_never_receives_the_final_holdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from classical_ml_lab.experiments import logistic_regression

    bundle = make_synthetic_credit(42)
    expected_train, expected_test, _, _, _ = stratified_split(bundle, 42)
    real_search = build_logistic_regression_search(42)
    fitted_indices: list[int] = []

    class SearchSpy:
        def fit(self, features: pd.DataFrame, target: pd.Series) -> SearchSpy:
            fitted_indices.extend(int(index) for index in features.index)
            real_search.fit(features, target)
            return self

        def __getattr__(self, name: str) -> object:
            return getattr(real_search, name)

    spy = SearchSpy()
    monkeypatch.setattr(logistic_regression, "build_logistic_regression_search", lambda _: spy)
    run_logistic_regression(seed=42, output_dir=tmp_path)
    assert set(fitted_indices) == set(expected_train.index)
    assert set(fitted_indices).isdisjoint(expected_test.index)


@pytest.mark.integration
def test_req_exp_lr_003_generates_metrics_cv_config_and_coefficients(tmp_path: Path) -> None:
    result = run_logistic_regression(seed=42, output_dir=tmp_path)
    assert result.experiment == "logistic-regression"
    assert result.task == "binary_classification"
    assert result.metrics["roc_auc"] >= 0.75
    assert result.metrics["pr_auc"] > 0.0
    assert sum(map(sum, result.confusion_matrix)) == result.split["test_samples"]
    assert result.cross_validation is not None
    assert result.cross_validation["strategy"] == "StratifiedKFold"
    assert result.cross_validation["folds"] == 3
    assert result.cross_validation["scoring"] == "roc_auc"
    assert result.cross_validation["best_params"]["C"] in [0.1, 1.0, 10.0]
    assert result.config["grid"] == {"C": [0.1, 1.0, 10.0]}
    assert result.config["selected"]["C"] in [0.1, 1.0, 10.0]
    figure = tmp_path / result.figures[0]
    assert result.figures == ("figures/coefficients.png", "figures/data-quality.png")
    assert figure.is_file()
    assert figure.stat().st_size > 0


@pytest.mark.integration
def test_req_exp_lr_003_passes_continuous_probabilities_to_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from classical_ml_lab.experiments import logistic_regression

    real_compute = logistic_regression.compute_binary_metrics
    captured: dict[str, np.ndarray] = {}

    def capture_scores(
        truth: pd.Series, predicted: np.ndarray, scores: np.ndarray
    ) -> object:
        captured["predicted"] = np.asarray(predicted)
        captured["scores"] = np.asarray(scores)
        return real_compute(truth, predicted, scores)

    monkeypatch.setattr(logistic_regression, "compute_binary_metrics", capture_scores)
    run_logistic_regression(seed=42, output_dir=tmp_path)
    assert len(np.unique(captured["scores"])) > 2
    assert np.all((captured["scores"] >= 0.0) & (captured["scores"] <= 1.0))
    assert not np.array_equal(captured["scores"], captured["predicted"])


@pytest.mark.integration
def test_req_data_lr_001_accepts_valid_local_credit_csv(
    tmp_path: Path, credit_csv: Path
) -> None:
    result = run_logistic_regression(
        seed=42,
        output_dir=tmp_path / "experiment",
        dataset="credit",
        data_path=credit_csv,
    )
    assert result.dataset_name == "give-me-some-credit-local"
    assert result.dataset_source == "user-provided-local-file"


@pytest.mark.integration
def test_req_art_lr_001_same_seed_is_reproducible(tmp_path: Path) -> None:
    first = run_experiments("logistic-regression", seed=42, output_dir=tmp_path)
    second = run_experiments("logistic-regression", seed=42, output_dir=tmp_path)
    changed = run_experiments("logistic-regression", seed=43, output_dir=tmp_path)
    first_manifest = json.loads((first / "run.json").read_text(encoding="utf-8"))
    second_manifest = json.loads((second / "run.json").read_text(encoding="utf-8"))
    changed_manifest = json.loads((changed / "run.json").read_text(encoding="utf-8"))
    assert first_manifest["config_hash"] == second_manifest["config_hash"]
    assert first_manifest["config_hash"] != changed_manifest["config_hash"]
    first_metrics = json.loads(
        (first / "experiments" / "logistic-regression" / "metrics.json").read_text(
            encoding="utf-8"
        )
    )
    second_metrics = json.loads(
        (second / "experiments" / "logistic-regression" / "metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert first_metrics == second_metrics
    assert first_metrics["config"]["pipeline"] == [
        "SimpleImputer",
        "StandardScaler",
        "LogisticRegression",
    ]
    assert first_metrics["config"]["grid"] == {"C": [0.1, 1.0, 10.0]}
