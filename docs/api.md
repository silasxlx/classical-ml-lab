# Public API

The command line is the primary stable interface. Python functions are public where documented below; other modules are implementation details and may change within a minor release.

## CLI

### `ml-lab list`

Lists nine core experiment IDs followed by three optional Boosting experiment IDs in stable order. Exit code `0` indicates success.

### `ml-lab run`

```text
ml-lab run <decision-tree|random-forest|adaboost|svm|logistic-regression|knn|naive-bayes|kmeans|ridge-regression|xgboost-regression|lightgbm-regression|catboost-regression|all|boosting-all>
           [--seed INTEGER]
           [--output-dir PATH]
           [--dataset synthetic|credit]
           [--data-path CSV]
```

- `--seed`: integer from `0` through `4294967295`; default `42`.
- `--output-dir`: artifact root; default `artifacts`.
- `--dataset` and `--data-path`: supported by random forest, Logistic Regression, and `all`. `credit` requires an explicit local path; with `all`, both credit-shaped experiments use the same selected data.
- `all` runs the nine dependency-light core experiments. `boosting-all` runs only the three optional Boosting regressors and requires `--all-extras`.
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
- `load_diabetes_regression() -> DatasetBundle`

All default loaders work offline. `load_credit_csv` validates the external file before training.

### Metric functions

```python
compute_binary_metrics(y_true, y_pred, y_score) -> MetricResult
compute_multiclass_metrics(y_true, y_pred, y_score) -> MetricResult
compute_clustering_metrics(features, cluster_labels, reference_labels, *, inertia: float) -> dict[str, float]
compute_regression_metrics(y_true, y_pred) -> RegressionMetricResult
```

`y_score` must be continuous probability or decision-score data. The functions reject invalid shapes, missing classes and non-finite scores instead of writing undefined JSON values.

`compute_clustering_metrics` validates an unsupervised partition and reports silhouette, adjusted Rand index, normalized mutual information, and inertia. Reference labels are evaluation-only and are never supplied to KMeans fitting.

`compute_regression_metrics` validates equal-length, finite, non-constant targets and reports MAE, RMSE, and R². MAPE is deliberately excluded because targets near zero make it unstable.

### Data quality

```python
create_data_quality_artifact(
    bundle: DatasetBundle,
    *,
    output_dir: Path,
    target_usage: Literal["training", "evaluation_only"] = "training",
) -> DataQualityArtifact
```

Returns aggregate column and target statistics plus a headless PNG. It records no raw rows, patient records, credentials, or absolute paths. IQR outliers are warnings and are not removed automatically.

### Experiment runners

```python
run_decision_tree(*, seed: int, output_dir: Path) -> ExperimentResult
run_svm(*, seed: int, output_dir: Path) -> ExperimentResult
run_adaboost(*, seed: int, output_dir: Path) -> ExperimentResult
run_random_forest(
    *, seed: int, output_dir: Path, dataset: str = "synthetic", data_path: Path | None = None
) -> ExperimentResult
build_logistic_regression_search(seed: int) -> GridSearchCV
run_logistic_regression(
    *, seed: int, output_dir: Path, dataset: str = "synthetic", data_path: Path | None = None
) -> ExperimentResult
build_knn_pipeline() -> Pipeline
run_knn(*, seed: int, output_dir: Path) -> ExperimentResult
build_naive_bayes() -> GaussianNB
run_naive_bayes(*, seed: int, output_dir: Path) -> ExperimentResult
build_kmeans_pipeline(seed: int) -> Pipeline
run_kmeans(*, seed: int, output_dir: Path) -> ClusteringExperimentResult
build_ridge_regression_pipeline() -> Pipeline
run_ridge_regression(*, seed: int, output_dir: Path) -> RegressionExperimentResult
build_xgboost_regressor(seed: int) -> XGBRegressor
run_xgboost_regression(*, seed: int, output_dir: Path) -> RegressionExperimentResult
build_lightgbm_regressor(seed: int) -> LGBMRegressor
run_lightgbm_regression(*, seed: int, output_dir: Path) -> RegressionExperimentResult
build_catboost_regressor(seed: int) -> CatBoostRegressor
run_catboost_regression(*, seed: int, output_dir: Path) -> RegressionExperimentResult
```

Experiment runners write only their figures into `output_dir` and return typed results. The application runner owns final JSON serialization and run-directory commit semantics.

`build_logistic_regression_search` returns the documented median-imputation, standardization, and Logistic Regression Pipeline wrapped in a deterministic three-fold ROC-AUC grid search. `run_logistic_regression` keeps the final stratified holdout outside that search and evaluates it with continuous probabilities.

The Ridge API uses only core dependencies. Boosting APIs are lazy optional interfaces: invoke them with `uv run --extra xgboost`, `--extra lightgbm`, or `--extra catboost`. A missing extra is an input error with installation guidance, not a partial run.

## Artifact schemas

- [`schemas/run.schema.json`](../schemas/run.schema.json)
- [`schemas/metrics.schema.json`](../schemas/metrics.schema.json)
- [`schemas/clustering.schema.json`](../schemas/clustering.schema.json)
- [`schemas/regression.schema.json`](../schemas/regression.schema.json)
- [`schemas/data-quality.schema.json`](../schemas/data-quality.schema.json)
- [`schemas/explanation.schema.json`](../schemas/explanation.schema.json)

Schema version `1.0` is part of the public compatibility contract. Removing or changing required fields requires a major version.

New v1.1 metrics artifacts include an optional `config` object with the auditable model and search setup. It remains optional in the schema so artifacts produced by v1.0 continue to validate.

KNN and Naive Bayes use the classification schema. KMeans uses the dedicated clustering schema and intentionally omits classification-only split and confusion-matrix fields.

Ridge and the three Boosting models use the dedicated regression schema. Every successful experiment references its data-quality artifact from `run.json`; optional Boosting models also reference a Tree SHAP explanation artifact. Existing v1.0–v1.2 classification and clustering payloads remain valid under their schemas.
