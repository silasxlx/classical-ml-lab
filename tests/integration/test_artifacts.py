from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from classical_ml_lab.runner import run_experiments

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.integration
def test_all_experiments_produce_schema_valid_auditable_artifacts(tmp_path: Path) -> None:
    run_dir = run_experiments("all", seed=42, output_dir=tmp_path / "结果 artifacts")
    manifest = _load_json(run_dir / "run.json")
    run_schema = _load_json(PROJECT_ROOT / "schemas" / "run.schema.json")
    metrics_schema = _load_json(PROJECT_ROOT / "schemas" / "metrics.schema.json")
    clustering_schema = _load_json(PROJECT_ROOT / "schemas" / "clustering.schema.json")
    Draft202012Validator(run_schema).validate(manifest)
    assert manifest["status"] == "success"
    experiments = manifest["experiments"]
    assert isinstance(experiments, list)
    assert [item["id"] for item in experiments] == [
        "decision-tree",
        "random-forest",
        "adaboost",
        "svm",
        "logistic-regression",
        "knn",
        "naive-bayes",
        "kmeans",
    ]
    for experiment in experiments:
        assert isinstance(experiment, dict)
        metrics_path = run_dir / str(experiment["metrics_path"])
        metrics = _load_json(metrics_path)
        schema = clustering_schema if metrics["task"] == "clustering" else metrics_schema
        Draft202012Validator(schema).validate(metrics)
        for relative in experiment["figures"]:
            figure = run_dir / str(relative)
            assert figure.is_file()
            assert figure.stat().st_size > 0
            assert figure.resolve().is_relative_to(run_dir.resolve())
    random_forest = _load_json(
        run_dir / "experiments" / "random-forest" / "metrics.json"
    )
    logistic_regression = _load_json(
        run_dir / "experiments" / "logistic-regression" / "metrics.json"
    )
    assert random_forest["dataset"] == logistic_regression["dataset"]


@pytest.mark.integration
def test_v1_metrics_without_the_v11_config_addition_remain_schema_valid(tmp_path: Path) -> None:
    run_dir = run_experiments("svm", seed=42, output_dir=tmp_path)
    metrics = _load_json(run_dir / "experiments" / "svm" / "metrics.json")
    metrics.pop("config")
    metrics_schema = _load_json(PROJECT_ROOT / "schemas" / "metrics.schema.json")
    Draft202012Validator(metrics_schema).validate(metrics)


@pytest.mark.integration
def test_all_uses_the_same_local_credit_data_for_both_supported_models(
    tmp_path: Path, credit_csv: Path
) -> None:
    run_dir = run_experiments(
        "all",
        seed=42,
        output_dir=tmp_path / "artifacts",
        dataset="credit",
        data_path=credit_csv,
    )
    random_forest = _load_json(
        run_dir / "experiments" / "random-forest" / "metrics.json"
    )
    logistic_regression = _load_json(
        run_dir / "experiments" / "logistic-regression" / "metrics.json"
    )
    assert random_forest["dataset"] == logistic_regression["dataset"]


@pytest.mark.integration
def test_repeated_run_never_overwrites_and_is_reproducible(tmp_path: Path) -> None:
    first = run_experiments("svm", seed=42, output_dir=tmp_path)
    second = run_experiments("svm", seed=42, output_dir=tmp_path)
    assert first != second
    first_manifest = _load_json(first / "run.json")
    second_manifest = _load_json(second / "run.json")
    assert first_manifest["config_hash"] == second_manifest["config_hash"]
    first_metrics = _load_json(first / "experiments" / "svm" / "metrics.json")
    second_metrics = _load_json(second / "experiments" / "svm" / "metrics.json")
    assert first_metrics == second_metrics


@pytest.mark.integration
def test_unexpected_failure_writes_only_a_sanitized_failure_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from classical_ml_lab import runner

    def fail(**_: object) -> object:
        raise RuntimeError(f"do not leak {tmp_path.resolve()} private-value")

    monkeypatch.setitem(runner.EXPERIMENT_RUNNERS, "decision-tree", fail)
    with pytest.raises(RuntimeError):
        run_experiments("decision-tree", seed=42, output_dir=tmp_path)
    run_dirs = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    manifest_text = (run_dirs[0] / "run.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["status"] == "failed"
    assert not (run_dirs[0] / "experiments").exists()
    assert "private-value" not in manifest_text
    assert str(tmp_path.resolve()) not in manifest_text
