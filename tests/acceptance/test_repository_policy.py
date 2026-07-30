from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest

from classical_ml_lab import __version__
from classical_ml_lab.cli import build_parser

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.acceptance
def test_readme_quickstart_has_exactly_three_real_commands() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    block = readme.split("<!-- quickstart:start -->", maxsplit=1)[1].split(
        "<!-- quickstart:end -->", maxsplit=1
    )[0]
    commands = [
        line.strip()
        for line in block.splitlines()
        if line.strip() and not line.strip().startswith("```")
    ]
    assert commands == [
        "git clone https://github.com/silasxlx/classical-ml-lab.git",
        "cd classical-ml-lab",
        "uv run ml-lab run all",
    ]


@pytest.mark.acceptance
def test_readme_answers_three_questions_and_states_high_risk_boundary() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    for phrase in ("是什么", "为什么", "怎么用", "不得直接用于真实授信", "Quick Start"):
        assert phrase in readme


@pytest.mark.acceptance
def test_api_docs_cover_public_commands_and_functions() -> None:
    api = (PROJECT_ROOT / "docs" / "api.md").read_text(encoding="utf-8")
    for symbol in (
        "ml-lab list",
        "ml-lab run",
        "run_experiments",
        "load_credit_csv",
        "compute_binary_metrics",
        "compute_multiclass_metrics",
        "run_decision_tree",
        "run_random_forest",
        "run_adaboost",
        "run_svm",
        "logistic-regression",
        "build_logistic_regression_search",
        "run_logistic_regression",
        "build_knn_pipeline",
        "run_knn",
        "build_naive_bayes",
        "run_naive_bayes",
        "build_kmeans_pipeline",
        "run_kmeans",
        "compute_clustering_metrics",
        "load_diabetes_regression",
        "compute_regression_metrics",
        "create_data_quality_artifact",
        "build_ridge_regression_pipeline",
        "run_ridge_regression",
        "xgboost-regression",
        "lightgbm-regression",
        "catboost-regression",
        "boosting-all",
    ):
        assert symbol in api
    assert "Educational use only" in build_parser().format_help() or "machine-learning" in api


@pytest.mark.acceptance
def test_design_document_preserves_required_architecture_decisions() -> None:
    design = (PROJECT_ROOT / "docs" / "design.md").read_text(encoding="utf-8")
    for heading in (
        "Package and CLI as the executable core",
        "Offline datasets by default",
        "Isolated preprocessing, tuning, and evaluation",
        "Versioned and auditable artifacts",
        "Safety limitations",
    ):
        assert f"## {heading}" in design


@pytest.mark.acceptance
def test_required_engineering_and_community_files_exist() -> None:
    required = (
        ".gitignore",
        "LICENSE",
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        "SECURITY.md",
        "CHANGELOG.md",
        "pyproject.toml",
        ".github/workflows/ci.yml",
        ".github/pull_request_template.md",
    )
    for relative in required:
        path = PROJECT_ROOT / relative
        assert path.is_file(), relative
        assert path.stat().st_size > 20


@pytest.mark.acceptance
def test_version_and_release_policy_use_semver() -> None:
    assert re.fullmatch(r"0|[1-9]\d*\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)", __version__)
    changelog = (PROJECT_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    contributing = (PROJECT_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    lockfile = (PROJECT_ROOT / "uv.lock").read_text(encoding="utf-8")
    assert "## [Unreleased]" in changelog
    assert "SemVer" in contributing
    assert 'name = "cz_conventional_commits"' in pyproject
    assert __version__ == "1.7.0"
    assert pyproject.count('version = "1.7.0"') == 2
    assert re.search(
        r'\[\[package\]\]\s+name = "classical-ml-lab"\s+version = "1.7.0"', lockfile
    )
    assert "## [1.7.0]" in changelog
    assert "## [1.2.0]" in changelog
    assert "## [1.1.0]" in changelog


@pytest.mark.acceptance
def test_readme_documents_core_and_optional_experiments() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    for phrase in (
        "九个核心实验",
        "Logistic Regression",
        "KNN",
        "Naive Bayes",
        "KMeans",
        "knn",
        "naive-bayes",
        "kmeans",
        "Ridge Regression",
        "ridge-regression",
        "xgboost-regression",
        "lightgbm-regression",
        "catboost-regression",
        "data-quality.json",
        "SHAP 只描述模型",
    ):
        assert phrase in readme


@pytest.mark.acceptance
def test_ci_runs_lint_type_check_tests_and_security_scans() -> None:
    workflows = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (PROJECT_ROOT / ".github" / "workflows").glob("*.yml")
    )
    for command in ("ruff check", "mypy src", "pytest", "pip-audit", "gitleaks", "codeql"):
        assert command in workflows


@pytest.mark.acceptance
def test_source_has_no_untracked_work_markers_or_obvious_comment_blocks() -> None:
    work_markers = ("TO" + "DO", "FIX" + "ME", "HA" + "CK")
    marker = re.compile(r"\b(?:" + "|".join(work_markers) + r")\b")
    commented_code = re.compile(r"^\s*#\s*(?:def|class|import|from|if|for|while)\b")
    for path in (PROJECT_ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert marker.search(text) is None, path
        assert commented_code.search(text) is None, path


@pytest.mark.acceptance
def test_repository_contains_no_large_data_or_model_artifacts() -> None:
    forbidden_suffixes = {".csv", ".joblib", ".pkl", ".onnx"}
    ignored_roots = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "artifacts",
        "htmlcov",
    }
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file() or any(part in ignored_roots for part in path.parts):
            continue
        assert path.suffix.lower() not in forbidden_suffixes, path
        assert path.stat().st_size < 1_000_000, path


@pytest.mark.acceptance
def test_json_schemas_are_parseable_and_versioned() -> None:
    for path in (PROJECT_ROOT / "schemas").glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["$schema"].endswith("2020-12/schema")
        assert payload["type"] == "object"


@pytest.mark.acceptance
def test_project_urls_reference_the_active_repository() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as pyproject_file:
        project_urls = tomllib.load(pyproject_file)["project"]["urls"]

    assert project_urls == {
        "Homepage": "https://github.com/silasxlx/classical-ml-lab",
        "Documentation": "https://github.com/silasxlx/classical-ml-lab/tree/main/docs",
        "Issues": "https://github.com/silasxlx/classical-ml-lab/issues",
    }


@pytest.mark.acceptance
def test_dependabot_uses_native_uv_lockfile_updates() -> None:
    config = (PROJECT_ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")

    assert "package-ecosystem: uv" in config
    assert "versioning-strategy: increase-if-necessary" in config
    assert 'update-types: ["version-update:semver-major"]' in config
    assert 'patterns: ["github/codeql-action/*"]' in config
    assert "package-ecosystem: pip" not in config


@pytest.mark.acceptance
def test_readme_preview_images_are_versioned_and_lightweight() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    preview_paths = (
        "docs/assets/preview-kmeans-clusters.png",
        "docs/assets/preview-ridge-predictions.png",
        "docs/assets/preview-data-quality.png",
        "docs/assets/preview-shap-global.png",
    )

    for relative_path in preview_paths:
        preview = PROJECT_ROOT / relative_path
        assert relative_path in readme
        assert preview.is_file()
        assert 0 < preview.stat().st_size < 1_000_000
