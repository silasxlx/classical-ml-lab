# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and releases follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

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

[Unreleased]: https://github.com/silasxlx/classical-ml-lab/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/silasxlx/classical-ml-lab/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/silasxlx/classical-ml-lab/releases/tag/v1.0.0
