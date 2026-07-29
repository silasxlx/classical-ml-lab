"""Public command-line interface for classical-ml-lab."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from classical_ml_lab.errors import LabError
from classical_ml_lab.experiments import EXPERIMENT_DESCRIPTIONS, EXPERIMENT_IDS
from classical_ml_lab.runner import run_experiments


def build_parser() -> argparse.ArgumentParser:
    """Build the stable public argument parser."""

    parser = argparse.ArgumentParser(
        prog="ml-lab",
        description="Run reproducible classical machine-learning experiments.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List available experiments.")

    run_parser = commands.add_parser(
        "run",
        help=(
            "Run one experiment or all experiments. Educational use only; not for credit decisions."
        ),
    )
    run_parser.add_argument("experiment", choices=(*EXPERIMENT_IDS, "all"))
    run_parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    run_parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts"), help="Artifact root directory."
    )
    run_parser.add_argument(
        "--dataset", choices=("synthetic", "credit"), help="Random-forest dataset."
    )
    run_parser.add_argument(
        "--data-path", type=Path, help="Explicit local credit CSV; no data is downloaded."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return its documented process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "list":
        for experiment_id, description in EXPERIMENT_DESCRIPTIONS.items():
            print(f"{experiment_id}\t{description}")
        return 0
    try:
        result_dir = run_experiments(
            args.experiment,
            seed=args.seed,
            output_dir=args.output_dir,
            dataset=args.dataset,
            data_path=args.data_path,
        )
    except LabError as exc:
        print(f"error[{exc.code}]: {exc}", file=sys.stderr)
        return 2
    except Exception:
        print(
            "error[experiment_runtime_error]: Experiment execution failed; "
            "a sanitized failure manifest was written.",
            file=sys.stderr,
        )
        return 1
    print(result_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
