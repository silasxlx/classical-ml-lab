"""Application service that runs experiments and commits their artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from classical_ml_lab.artifacts import (
    config_hash,
    create_staging_directory,
    discard_staging,
    failed_payload,
    finalize_staging,
    make_run_id,
    success_payload,
    utc_now,
    write_json,
)
from classical_ml_lab.errors import InputValidationError, LabError
from classical_ml_lab.experiments import (
    DATASET_EXPERIMENT_IDS,
    EXPERIMENT_IDS,
    EXPERIMENT_RUNNERS,
)
from classical_ml_lab.models import RunResult


def _selection(experiment: str) -> tuple[str, ...]:
    if experiment == "all":
        return EXPERIMENT_IDS
    if experiment not in EXPERIMENT_RUNNERS:
        allowed = ", ".join((*EXPERIMENT_IDS, "all"))
        raise InputValidationError(f"Unknown experiment {experiment!r}. Choose one of: {allowed}.")
    return (experiment,)


def _validate_options(
    experiment: str, dataset: str | None, data_path: Path | None, seed: int
) -> None:
    if not 0 <= seed <= 4_294_967_295:
        raise InputValidationError("Seed must be between 0 and 4294967295.")
    if (
        experiment not in DATASET_EXPERIMENT_IDS
        and experiment != "all"
        and (dataset is not None or data_path is not None)
    ):
        raise InputValidationError(
            "Credit dataset options are only valid for random-forest or logistic-regression."
        )
    if dataset == "credit" and data_path is None:
        raise InputValidationError("--data-path is required with --dataset credit.")
    if dataset not in {None, "synthetic", "credit"}:
        raise InputValidationError("Dataset must be 'synthetic' or 'credit'.")


def _run_one(
    experiment: str,
    *,
    seed: int,
    output_dir: Path,
    dataset: str | None,
    data_path: Path | None,
) -> RunResult:
    runner = EXPERIMENT_RUNNERS[experiment]
    if experiment in DATASET_EXPERIMENT_IDS:
        return runner(
            seed=seed,
            output_dir=output_dir,
            dataset=dataset or "synthetic",
            data_path=data_path,
        )
    return runner(seed=seed, output_dir=output_dir)


def _hash_input(
    *, seed: int, selection: tuple[str, ...], results: list[RunResult]
) -> dict[str, Any]:
    return {
        "seed": seed,
        "experiments": list(selection),
        "results": [
            {
                "id": result.experiment,
                "dataset_fingerprint": result.dataset_fingerprint,
                "config": result.config,
            }
            for result in results
        ],
    }


def run_experiments(
    experiment: str,
    *,
    seed: int = 42,
    output_dir: Path = Path("artifacts"),
    dataset: str | None = None,
    data_path: Path | None = None,
) -> Path:
    """Run one or all experiments and return the finalized artifact directory."""

    selected = _selection(experiment)
    _validate_options(experiment, dataset, data_path, seed)
    staging = create_staging_directory(output_dir)
    started_at = utc_now()
    try:
        results: list[RunResult] = []
        for experiment_id in selected:
            experiment_dir = staging / "experiments" / experiment_id
            result = _run_one(
                experiment_id,
                seed=seed,
                output_dir=experiment_dir,
                dataset=dataset,
                data_path=data_path,
            )
            write_json(experiment_dir / "metrics.json", result.to_metrics_payload())
            results.append(result)

        digest = config_hash(_hash_input(seed=seed, selection=selected, results=results))
        completed_at = utc_now()
        run_id = make_run_id(completed_at, digest)
        manifest = success_payload(
            run_id=run_id,
            seed=seed,
            digest=digest,
            started_at=started_at,
            completed_at=completed_at,
            results=results,
        )
        write_json(staging / "run.json", manifest)
        return finalize_staging(staging, output_dir, run_id)
    except LabError:
        discard_staging(staging)
        raise
    except Exception:
        discard_staging(staging)
        staging = create_staging_directory(output_dir)
        completed_at = utc_now()
        digest = config_hash({"seed": seed, "experiments": list(selected), "status": "failed"})
        run_id = make_run_id(completed_at, digest)
        write_json(
            staging / "run.json",
            failed_payload(
                run_id=run_id,
                seed=seed,
                digest=digest,
                started_at=started_at,
                completed_at=completed_at,
            ),
        )
        finalize_staging(staging, output_dir, run_id)
        raise
