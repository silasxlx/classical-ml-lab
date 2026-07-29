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
| Logistic Regression | Deterministic credit-shaped synthetic data | Generated locally | Never |
| KNN | Iris | scikit-learn bundled dataset | Never |
| Gaussian Naive Bayes | Iris | scikit-learn bundled dataset | Never |
| KMeans | Iris features | scikit-learn bundled dataset | Never |
| Ridge Regression | Diabetes, raw feature scale | scikit-learn bundled dataset | Never |
| Optional Boosting regressors | Diabetes, raw feature scale | scikit-learn bundled dataset | Never |

Dataset fingerprints are calculated from normalized feature and target values and recorded in `metrics.json`. Default experiments never download data or depend on the current working directory. Diabetes is loaded with `scaled=False`; Ridge performs scaling inside its training Pipeline instead of consuming a full-dataset pre-scaled result.

## Optional local credit data

The random-forest and Logistic Regression experiments accept an explicitly supplied local CSV shaped like the historical Give Me Some Credit training data. The project does not download or redistribute that dataset. Users are responsible for obtaining data lawfully and complying with its terms. When `ml-lab run all` receives local credit-data options, both experiments use the same validated dataset and fingerprint.

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

Classification experiments create one stratified final holdout. Regression experiments create one deterministic random holdout. Imputation and scaling are fitted inside scikit-learn Pipelines. Hyperparameter search uses only training data with stratified cross-validation. Final metrics are computed once on the untouched holdout, using probabilities or decision scores for classification AUC.

Logistic Regression uses median imputation followed by standardization and an L2-regularized, class-balanced classifier. Its three-fold search selects `C` using training ROC-AUC. The coefficient figure describes associations in standardized feature space; it is not a causal explanation or evidence of production suitability.

Tests verify split isolation and Pipeline placement rather than accepting a model only because it reaches a high score.

## Classification, clustering, and regression remain distinct

KNN uses scaling inside its Pipeline because neighbor distance is scale-sensitive. Gaussian Naive Bayes uses its fixed continuous-feature model directly. Both reuse the established Iris holdout and multiclass probability metrics.

KMeans fits a standardized four-feature Iris matrix without receiving target labels. The known species labels are used only after fitting for ARI and NMI; silhouette and inertia remain feature/partition metrics. Its artifact has a dedicated schema, a full-dataset fit summary, cluster sizes, and a two-feature visualization projection rather than a fabricated classification split or confusion matrix.

Diabetes regression predicts the continuous disease-progression target one year after baseline measurement. Ridge uses `StandardScaler → Ridge(alpha=1.0)` and reports MAE, RMSE, and R² on 133 held-out rows. It does not report classification metrics or MAPE. The dataset and model are teaching examples and are not medical decision tools.

## Aggregate data-quality reporting

Every core experiment produces the same versioned quality contract before modeling: shape and fingerprint; column types, missing/non-finite counts, uniqueness and constant flags; numeric five-number summaries; duplicate-row counts; and IQR outlier warnings. Classification targets use distributions, regression targets use aggregate statistics, and KMeans reference labels are marked `evaluation_only`.

Reports never store raw rows, complete patient records, credentials, or local absolute paths. Warnings are descriptive: the report does not silently drop outliers or alter training data.

## Optional Boosting and model explanations

XGBoost, LightGBM, and CatBoost are optional CPU-only, single-threaded regressors. They are excluded from core `run all`; `boosting-all` makes their larger dependencies explicit. If an extra is absent, validation fails before the staging directory is committed and gives the exact `uv run --extra ...` command.

Tree SHAP uses a deterministic background sampled only from the training split and explains held-out test rows. The global figure uses mean absolute SHAP values; the local figure uses a stable test index. The JSON stores feature names, SHAP values, base value, model output, and additivity residual without storing the sample's feature values. SHAP describes model behavior, not causality, medical evidence, or clinical safety.

## Versioned and auditable artifacts

Every CLI run writes a versioned `run.json`, per-experiment `metrics.json`, and referenced PNG figures in a unique directory. Metrics artifacts include the model configuration, while configuration hashes cover the seed, model configuration, and dataset fingerprint and exclude timestamps and output paths. The additive `config` field is optional in schema version `1.0`, so existing v1.0 artifacts remain valid.

Failure manifests are sanitized and do not contain raw input data, credentials, or local absolute paths. Classification, clustering, regression, data-quality, and explanation JSON contracts live in [`schemas/`](../schemas/) and remain separate so their scientific meanings are explicit. Optional runs also record the installed Boosting and SHAP versions.

## Safety limitations

- No public demo may contain real customer or employer data.
- Included datasets do not establish fairness, calibration, stability, or regulatory suitability.
- Metrics are educational and must not be interpreted as approval for production lending, medical use, or another high-risk decision.
- Real-world use would require independent review of data rights, representativeness, fairness, calibration, drift, explainability, human oversight, and monitoring.
