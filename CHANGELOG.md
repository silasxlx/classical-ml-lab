# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and releases follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

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

[Unreleased]: https://github.com/silasxlx/classical-ml-lab/compare/v1.0.0...HEAD
