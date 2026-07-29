# Architecture, data, and evaluation design

This document records the small set of decisions that define the project's public behavior, reproducibility, and safety boundaries.

## Package and CLI as the executable core

The installable `src/classical_ml_lab` package owns data loading, metrics, experiments, artifacts, and orchestration. The `ml-lab` CLI exposes stable behavior, while notebooks call the same package instead of duplicating training logic.

This keeps execution independent of notebook state and local paths. Public CLI commands, documented Python symbols, and artifact schemas follow SemVer compatibility rules.

## Offline datasets by default

| Experiment | Dataset | Source | Network |
| --- | --- | --- | --- |
| Decision Tree | Iris | scikit-learn bundled dataset | Never |
| SVM | First two Iris classes and petal features | scikit-learn bundled dataset | Never |
| AdaBoost | Breast Cancer Wisconsin Diagnostic | scikit-learn bundled dataset | Never |
| Random Forest | Deterministic credit-shaped synthetic data | Generated locally | Never |

Dataset fingerprints are calculated from normalized feature and target values and recorded in `metrics.json`. Default experiments never download data or depend on the current working directory.

## Optional local credit data

The random-forest experiment accepts an explicitly supplied local CSV shaped like the historical Give Me Some Credit training data. The project does not download or redistribute that dataset. Users are responsible for obtaining data lawfully and complying with its terms.

Required target: `SeriousDlqin2yrs`.

Required predictors:

- `RevolvingUtilizationOfUnsecuredLines`
- `age`
- `NumberOfTime30-59DaysPastDueNotWorse`
- `DebtRatio`
- `MonthlyIncome`
- `NumberOfOpenCreditLinesAndLoans`
- `NumberOfTimes90DaysLate`
- `NumberRealEstateLoansOrLines`
- `NumberOfTime60-89DaysPastDueNotWorse`
- `NumberOfDependents`

The loader permits missing predictor values because imputation occurs inside the training Pipeline. It rejects missing targets, non-binary targets, missing or duplicate required columns, non-numeric predictor content, fewer than 50 rows, or fewer than five rows in either class.

## Isolated preprocessing, tuning, and evaluation

Each experiment creates one stratified final holdout. Imputation and scaling are fitted inside scikit-learn Pipelines. Hyperparameter search uses only training data with stratified cross-validation. Final metrics are computed once on the untouched holdout, using probabilities or decision scores for AUC.

Tests verify split isolation and Pipeline placement rather than accepting a model only because it reaches a high score.

## Versioned and auditable artifacts

Every CLI run writes a versioned `run.json`, per-experiment `metrics.json`, and referenced PNG figures in a unique directory. Configuration hashes cover the seed, model configuration, and dataset fingerprint while excluding timestamps and output paths.

Failure manifests are sanitized and do not contain raw input data, credentials, or local absolute paths. JSON contracts live in [`schemas/`](../schemas/).

## Safety limitations

- No public demo may contain real customer or employer data.
- Included datasets do not establish fairness, calibration, stability, or regulatory suitability.
- Metrics are educational and must not be interpreted as approval for production lending, medical use, or another high-risk decision.
- Real-world use would require independent review of data rights, representativeness, fairness, calibration, drift, explainability, human oversight, and monitoring.
