# Public API

The command line is the primary stable interface. Python functions are public where documented below; other modules are implementation details and may change within a minor release.

## CLI

### `ml-lab list`

Lists the four experiment IDs in stable order. Exit code `0` indicates success.

### `ml-lab run`

```text
ml-lab run <decision-tree|random-forest|adaboost|svm|all>
           [--seed INTEGER]
           [--output-dir PATH]
           [--dataset synthetic|credit]
           [--data-path CSV]
```

- `--seed`: integer from `0` through `4294967295`; default `42`.
- `--output-dir`: artifact root; default `artifacts`.
- `--dataset` and `--data-path`: random-forest only. `credit` requires an explicit local path.
- Exit `0`: success. Exit `2`: argument or data validation failure. Exit `1`: unexpected execution failure with a sanitized failure manifest.

## Python API

### `classical_ml_lab.runner.run_experiments`

```python
run_experiments(
    experiment: str,
    *,
    seed: int = 42,
    output_dir: Path = Path("artifacts"),
    dataset: str | None = None,
    data_path: Path | None = None,
) -> Path
```

Runs one experiment or `all` and returns the finalized artifact directory. Raises `InputValidationError` for expected invalid input and preserves unexpected failures in a sanitized `run.json`.

### Dataset loaders

- `load_iris_multiclass() -> DatasetBundle`
- `load_iris_binary() -> DatasetBundle`
- `load_breast_cancer_binary() -> DatasetBundle`
- `make_synthetic_credit(seed: int) -> DatasetBundle`
- `load_credit_csv(path: Path) -> DatasetBundle`

All default loaders work offline. `load_credit_csv` validates the external file before training.

### Metric functions

```python
compute_binary_metrics(y_true, y_pred, y_score) -> MetricResult
compute_multiclass_metrics(y_true, y_pred, y_score) -> MetricResult
```

`y_score` must be continuous probability or decision-score data. The functions reject invalid shapes, missing classes and non-finite scores instead of writing undefined JSON values.

### Experiment runners

```python
run_decision_tree(*, seed: int, output_dir: Path) -> ExperimentResult
run_svm(*, seed: int, output_dir: Path) -> ExperimentResult
run_adaboost(*, seed: int, output_dir: Path) -> ExperimentResult
run_random_forest(
    *, seed: int, output_dir: Path, dataset: str = "synthetic", data_path: Path | None = None
) -> ExperimentResult
```

Experiment runners write only their figures into `output_dir` and return typed results. The application runner owns final JSON serialization and run-directory commit semantics.

## Artifact schemas

- [`schemas/run.schema.json`](../schemas/run.schema.json)
- [`schemas/metrics.schema.json`](../schemas/metrics.schema.json)

Schema version `1.0` is part of the public compatibility contract. Removing or changing required fields requires a major version.
