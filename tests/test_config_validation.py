"""Tests for config validation and utilities."""

import json
import os
from pathlib import Path

import pytest
import yaml

# Add project root to path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import (
    CONFIG_DIR,
    SCHEMA_DIR,
    apply_patch,
    diff_configs,
    hash_config,
    load_yaml,
    merge_configs,
    validate_augmentations,
    validate_tuning,
)


class TestConfigLoading:
    def test_load_base_config(self):
        config = load_yaml(CONFIG_DIR / "base.yaml")
        assert "project" in config
        assert "model" in config
        assert "dataset" in config

    def test_load_augmentations_config(self):
        config = load_yaml(CONFIG_DIR / "augmentations.yaml")
        assert "pipeline" in config

    def test_load_tuning_config(self):
        config = load_yaml(CONFIG_DIR / "tuning.yaml")
        assert "optimizer" in config
        assert "scheduler" in config
        assert "batch" in config

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_yaml("/nonexistent/path.yaml")


class TestConfigValidation:
    def test_valid_augmentation_config(self):
        config = load_yaml(CONFIG_DIR / "augmentations.yaml")
        errors = validate_augmentations(config)
        assert errors == [], f"Default augmentation config should be valid: {errors}"

    def test_valid_tuning_config(self):
        config = load_yaml(CONFIG_DIR / "tuning.yaml")
        errors = validate_tuning(config)
        assert errors == [], f"Default tuning config should be valid: {errors}"

    def test_invalid_learning_rate_too_high(self):
        config = {
            "optimizer": {"name": "adamw", "learning_rate": 1.0},  # Way too high
            "scheduler": {"name": "cosine", "warmup_steps": 50},
            "batch": {"train_batch_size": 8, "eval_batch_size": 16},
            "regularization": {},
        }
        errors = validate_tuning(config)
        assert len(errors) > 0, "Learning rate 1.0 should be rejected"

    def test_invalid_optimizer_name(self):
        config = {
            "optimizer": {"name": "invalid_opt", "learning_rate": 5e-5},
            "scheduler": {"name": "cosine", "warmup_steps": 50},
            "batch": {"train_batch_size": 8, "eval_batch_size": 16},
            "regularization": {},
        }
        errors = validate_tuning(config)
        assert len(errors) > 0, "Invalid optimizer name should be rejected"

    def test_invalid_batch_size(self):
        config = {
            "optimizer": {"name": "adamw", "learning_rate": 5e-5},
            "scheduler": {"name": "cosine", "warmup_steps": 50},
            "batch": {"train_batch_size": 7, "eval_batch_size": 16},  # 7 not in enum
            "regularization": {},
        }
        errors = validate_tuning(config)
        assert len(errors) > 0, "Batch size 7 should be rejected (not in enum)"

    def test_invalid_augmentation_probability(self):
        config = {
            "pipeline": {
                "enabled": True,
                "text_augmentations": [
                    {
                        "name": "synonym_replacement",
                        "enabled": True,
                        "probability": 1.5,  # > 1.0
                    }
                ],
            }
        }
        errors = validate_augmentations(config)
        assert len(errors) > 0, "Probability > 1.0 should be rejected"


class TestConfigDiff:
    def test_no_changes(self):
        config = {"a": 1, "b": 2}
        assert diff_configs(config, config) == []

    def test_value_change(self):
        old = {"optimizer": {"learning_rate": 5e-5}}
        new = {"optimizer": {"learning_rate": 1e-4}}
        diffs = diff_configs(old, new)
        assert len(diffs) == 1
        assert "learning_rate" in diffs[0]

    def test_addition(self):
        old = {"a": 1}
        new = {"a": 1, "b": 2}
        diffs = diff_configs(old, new)
        assert len(diffs) == 1
        assert diffs[0].startswith("+")

    def test_removal(self):
        old = {"a": 1, "b": 2}
        new = {"a": 1}
        diffs = diff_configs(old, new)
        assert len(diffs) == 1
        assert diffs[0].startswith("-")


class TestConfigMerge:
    def test_simple_merge(self):
        a = {"x": 1, "y": 2}
        b = {"y": 3, "z": 4}
        result = merge_configs(a, b)
        assert result == {"x": 1, "y": 3, "z": 4}

    def test_deep_merge(self):
        a = {"optimizer": {"name": "adamw", "lr": 5e-5}}
        b = {"optimizer": {"lr": 1e-4}}
        result = merge_configs(a, b)
        assert result["optimizer"]["name"] == "adamw"
        assert result["optimizer"]["lr"] == 1e-4

    def test_apply_patch(self):
        base = {"a": {"b": 1, "c": 2}, "d": 3}
        patch = {"a": {"b": 10}}
        result = apply_patch(base, patch)
        assert result["a"]["b"] == 10
        assert result["a"]["c"] == 2
        assert result["d"] == 3


class TestConfigHash:
    def test_deterministic(self):
        config = {"optimizer": {"lr": 5e-5}, "batch_size": 8}
        hash1 = hash_config(config)
        hash2 = hash_config(config)
        assert hash1 == hash2

    def test_different_configs(self):
        config1 = {"lr": 5e-5}
        config2 = {"lr": 1e-4}
        assert hash_config(config1) != hash_config(config2)
