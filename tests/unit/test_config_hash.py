from __future__ import annotations

import pytest

from classical_ml_lab.artifacts import config_hash


@pytest.mark.unit
def test_config_hash_ignores_dictionary_order() -> None:
    assert config_hash({"seed": 42, "model": "svm"}) == config_hash({"model": "svm", "seed": 42})


@pytest.mark.unit
def test_config_hash_changes_with_seed_model_or_data() -> None:
    baseline = config_hash({"seed": 42, "model": "svm", "data": "a"})
    assert baseline != config_hash({"seed": 43, "model": "svm", "data": "a"})
    assert baseline != config_hash({"seed": 42, "model": "tree", "data": "a"})
    assert baseline != config_hash({"seed": 42, "model": "svm", "data": "b"})
