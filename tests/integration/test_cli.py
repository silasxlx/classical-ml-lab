from __future__ import annotations

import json
from pathlib import Path

import pytest

from classical_ml_lab.cli import main
from classical_ml_lab.experiments import DATASET_EXPERIMENT_IDS


@pytest.mark.integration
def test_list_prints_core_and_optional_experiments_in_stable_order(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["list"]) == 0
    ids = [line.split("\t", maxsplit=1)[0] for line in capsys.readouterr().out.splitlines()]
    assert ids == [
        "decision-tree",
        "random-forest",
        "adaboost",
        "svm",
        "logistic-regression",
        "knn",
        "naive-bayes",
        "kmeans",
        "ridge-regression",
        "xgboost-regression",
        "lightgbm-regression",
        "catboost-regression",
    ]


@pytest.mark.integration
def test_single_run_returns_success_and_writes_only_selected_experiment(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["run", "decision-tree", "--output-dir", str(tmp_path)]) == 0
    run_dir = Path(capsys.readouterr().out.strip())
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert [item["id"] for item in manifest["experiments"]] == ["decision-tree"]
    assert not (run_dir / "experiments" / "svm").exists()


@pytest.mark.integration
def test_credit_options_are_rejected_for_other_experiments(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "run",
            "svm",
            "--dataset",
            "credit",
            "--data-path",
            str(tmp_path / "credit.csv"),
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert code == 2
    assert "only valid for random-forest or logistic-regression" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


@pytest.mark.integration
@pytest.mark.parametrize(
    "experiment",
    [
        "knn",
        "naive-bayes",
        "kmeans",
        "ridge-regression",
        "xgboost-regression",
        "lightgbm-regression",
        "catboost-regression",
    ],
)
def test_new_experiments_do_not_expand_credit_data_options(
    experiment: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_dir = tmp_path / experiment
    code = main(
        [
            "run",
            experiment,
            "--dataset",
            "synthetic",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert code == 2
    assert "only valid for random-forest or logistic-regression" in capsys.readouterr().err
    assert not output_dir.exists()


@pytest.mark.integration
def test_credit_data_capability_remains_limited_to_the_existing_two_experiments() -> None:
    assert frozenset({"random-forest", "logistic-regression"}) == DATASET_EXPERIMENT_IDS


@pytest.mark.integration
def test_logistic_regression_cli_accepts_local_credit_data(
    tmp_path: Path, credit_csv: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_dir = tmp_path / "artifacts"
    code = main(
        [
            "run",
            "logistic-regression",
            "--dataset",
            "credit",
            "--data-path",
            str(credit_csv),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert code == 0
    run_dir = Path(capsys.readouterr().out.strip())
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert [item["id"] for item in manifest["experiments"]] == ["logistic-regression"]


@pytest.mark.integration
def test_credit_dataset_requires_a_path_before_output_is_created(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_dir = tmp_path / "artifacts"
    code = main(
        [
            "run",
            "logistic-regression",
            "--dataset",
            "credit",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert code == 2
    assert "--data-path is required" in capsys.readouterr().err
    assert not output_dir.exists()


@pytest.mark.integration
def test_seed_range_is_validated_before_output_is_created(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["run", "svm", "--seed", "-1", "--output-dir", str(tmp_path)]) == 2
    assert "between 0 and 4294967295" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []
