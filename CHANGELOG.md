# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and releases follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Authentic README previews for clustering, regression diagnostics, data quality, and global SHAP output.

### Changed

- Updated package metadata links to the active `classical-ml-lab` repository.
- Switched Dependabot from generic pip updates to native uv support so `pyproject.toml` and `uv.lock` stay synchronized; automated version updates are limited to compatible minor and patch releases.

## [1.7.0] - 2026-07-29

### Added

- Leakage-safe Ridge regression on scikit-learn's bundled Diabetes dataset.
- Dedicated regression metrics and artifact schema for MAE, RMSE, and R².
- Versioned aggregate data-quality JSON and PNG reports for every core experiment.
- Optional, deterministic CPU XGBoost, LightGBM, and CatBoost regression on Diabetes.
- Explicit `boosting-all` command for the three optional Boosting regressors.
- Training-background-isolated global and local Tree SHAP explanations.
- Versioned, aggregate-safe explanation JSON with an audited additivity residual.

### Changed

- Core `ml-lab run all` now executes nine offline experiments.
- Package architecture now keeps classification, clustering, and regression contracts separate.

### Security

- Optional dependency failures return an actionable input error without partial artifacts.
- Explanation artifacts omit complete patient rows, feature values, and local paths.

## [1.2.0] - 2026-07-29

### Added

- Fixed-parameter KNN and Gaussian Naive Bayes experiments on the Iris dataset.
- True unsupervised KMeans experiment with a dedicated clustering artifact schema.
- Validated silhouette, adjusted Rand index, normalized mutual information, and inertia metrics.

### Changed

- `ml-lab list` and `ml-lab run all` now include eight experiments.
- Package version advanced to `1.2.0` without changing existing classification experiment behavior.

## [1.1.0] - 2026-07-29

### Added

- Leakage-safe Logistic Regression experiment with median imputation, feature scaling, and training-only cross-validation.
- Standardized coefficient visualization and auditable hyperparameter-search metadata.
- Synthetic and explicitly supplied local credit-data support for both credit-shaped experiments.

### Changed

- `ml-lab list` and `ml-lab run all` now include five experiments.
- Package version advanced to `1.1.0` without changing artifact schema version `1.0`.

## [1.0.0] - 2026-07-29

### Added

- Reproducible package and `ml-lab` CLI for four classical ML experiments.
- Versioned metrics and run artifact schemas.
- Typed data validation, leakage-safe Pipelines, tests, CI, API docs, and ADRs.

### Changed

- Replaced path-dependent 2019 scripts and Notebook-only logic with reusable modules.
- Replaced committed raw credit data with an optional, explicitly supplied local-data path.

### Security

- Added dependency, code, and secret scanning configuration.
- Added explicit prohibitions on real customer data and high-risk decision use.

[Unreleased]: https://github.com/silasxlx/classical-ml-lab/compare/v1.7.0...HEAD
[1.7.0]: https://github.com/silasxlx/classical-ml-lab/compare/v1.2.0...v1.7.0
[1.2.0]: https://github.com/silasxlx/classical-ml-lab/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/silasxlx/classical-ml-lab/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/silasxlx/classical-ml-lab/releases/tag/v1.0.0
