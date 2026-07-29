from __future__ import annotations

import json
from pathlib import Path

import pytest

from classical_ml_lab.cli import main


@pytest.mark.integration
def test_list_prints_four_experiments_in_stable_order(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list"]) == 0
    ids = [line.split("\t", maxsplit=1)[0] for line in capsys.readouterr().out.splitlines()]
    assert ids == ["decision-tree", "random-forest", "adaboost", "svm"]


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
    assert "only valid for random-forest" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


@pytest.mark.integration
def test_seed_range_is_validated_before_output_is_created(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["run", "svm", "--seed", "-1", "--output-dir", str(tmp_path)]) == 2
    assert "between 0 and 4294967295" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []
