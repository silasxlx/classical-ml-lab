"""Versioned, auditable run artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sklearn

from classical_ml_lab import __version__
from classical_ml_lab.errors import InputValidationError
from classical_ml_lab.models import RunResult


def utc_now() -> datetime:
    """Return the current timezone-aware UTC time."""

    return datetime.now(UTC)


def isoformat(value: datetime) -> str:
    """Serialize a datetime as seconds-precision UTC."""

    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def config_hash(payload: dict[str, Any]) -> str:
    """Hash deterministic configuration and dataset metadata."""

    encoded = json.dumps(
        payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write UTF-8 JSON atomically within an existing run directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def create_staging_directory(output_dir: Path) -> Path:
    """Create a unique staging directory or raise an actionable validation error."""

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        staging = output_dir / f".tmp-{uuid.uuid4().hex}"
        staging.mkdir()
        return staging
    except OSError as exc:
        raise InputValidationError("Output directory cannot be created or written.") from exc


def discard_staging(staging: Path) -> None:
    """Remove an incomplete staging directory."""

    if staging.exists():
        shutil.rmtree(staging)


def _unique_destination(output_dir: Path, base_name: str) -> Path:
    candidate = output_dir / base_name
    suffix = 1
    while candidate.exists():
        candidate = output_dir / f"{base_name}-{suffix:02d}"
        suffix += 1
    return candidate


def finalize_staging(staging: Path, output_dir: Path, run_id: str) -> Path:
    """Move a complete staging directory without overwriting an earlier run."""

    destination = _unique_destination(output_dir, run_id)
    staging.rename(destination)
    return destination


def make_run_id(timestamp: datetime, digest: str) -> str:
    """Create a readable UTC run ID containing the deterministic hash prefix."""

    stamp = timestamp.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{digest.removeprefix('sha256:')[:12]}"


def success_payload(
    *,
    run_id: str,
    seed: int,
    digest: str,
    started_at: datetime,
    completed_at: datetime,
    results: list[RunResult],
) -> dict[str, Any]:
    """Build the public success run manifest."""

    experiments = []
    for result in results:
        prefix = Path("experiments") / result.experiment
        experiments.append(
            {
                "id": result.experiment,
                "status": "success",
                "metrics_path": (prefix / "metrics.json").as_posix(),
                "figures": [(prefix / item).as_posix() for item in result.figures],
            }
        )
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "status": "success",
        "seed": seed,
        "config_hash": digest,
        "started_at": isoformat(started_at),
        "completed_at": isoformat(completed_at),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.system().lower(),
            "package_version": __version__,
            "scikit_learn": sklearn.__version__,
        },
        "experiments": experiments,
        "error": None,
    }


def failed_payload(
    *, run_id: str, seed: int, digest: str, started_at: datetime, completed_at: datetime
) -> dict[str, Any]:
    """Build a deliberately minimal and sanitized failure manifest."""

    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "status": "failed",
        "seed": seed,
        "config_hash": digest,
        "started_at": isoformat(started_at),
        "completed_at": isoformat(completed_at),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.system().lower(),
            "package_version": __version__,
            "scikit_learn": sklearn.__version__,
        },
        "experiments": [],
        "error": {
            "code": "experiment_runtime_error",
            "message": "Experiment execution failed. Inspect local console output.",
        },
    }
