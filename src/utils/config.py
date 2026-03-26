"""Configuration loading, validation, and diffing utilities."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft7Validator, ValidationError
from rich.console import Console

console = Console()

# Project root is 2 levels up from src/utils/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
SCHEMA_DIR = CONFIG_DIR / "schema"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file with environment variable interpolation.

    Supports ${VAR} and ${VAR:-default} syntax in string values.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = f.read()

    # Interpolate environment variables: ${VAR} and ${VAR:-default}
    import re

    def _replace_env(match: re.Match) -> str:
        var = match.group(1)
        if ":-" in var:
            name, default = var.split(":-", 1)
            return os.environ.get(name, default)
        return os.environ.get(var, match.group(0))

    interpolated = re.sub(r"\$\{([^}]+)\}", _replace_env, raw)
    return yaml.safe_load(interpolated) or {}


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    """Save a dictionary as a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def load_config() -> dict[str, Any]:
    """Load and merge all configuration files (base + augmentations + tuning)."""
    base = load_yaml(CONFIG_DIR / "base.yaml")
    augmentations = load_yaml(CONFIG_DIR / "augmentations.yaml")
    tuning = load_yaml(CONFIG_DIR / "tuning.yaml")
    return merge_configs(base, augmentations, tuning)


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    """Deep merge multiple config dictionaries. Later configs override earlier ones."""
    result: dict[str, Any] = {}
    for config in configs:
        result = _deep_merge(result, config)
    return result


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def validate_config(
    config: dict[str, Any],
    schema_name: str,
) -> list[str]:
    """Validate a config dict against a JSON schema.

    Args:
        config: The configuration dictionary to validate.
        schema_name: Schema filename (e.g., 'augmentations_schema.json').

    Returns:
        List of validation error messages. Empty list means valid.
    """
    schema_path = SCHEMA_DIR / schema_name
    if not schema_path.exists():
        return [f"Schema file not found: {schema_path}"]

    with open(schema_path) as f:
        schema = json.load(f)

    validator = Draft7Validator(schema)
    errors = []
    for error in sorted(validator.iter_errors(config), key=lambda e: list(e.path)):
        path = " → ".join(str(p) for p in error.path) or "(root)"
        errors.append(f"[{path}] {error.message}")
    return errors


def validate_augmentations(config: dict[str, Any]) -> list[str]:
    """Validate augmentation config against its schema."""
    return validate_config(config, "augmentations_schema.json")


def validate_tuning(config: dict[str, Any]) -> list[str]:
    """Validate tuning config against its schema."""
    return validate_config(config, "tuning_schema.json")


def diff_configs(
    old: dict[str, Any],
    new: dict[str, Any],
    prefix: str = "",
) -> list[str]:
    """Generate a human-readable diff between two config dicts.

    Returns:
        List of change descriptions, e.g.:
        - "optimizer.learning_rate: 5e-05 → 1e-04"
        - "+ augmentations.synonym_replacement.enabled: true"
    """
    changes: list[str] = []
    all_keys = set(list(old.keys()) + list(new.keys()))

    for key in sorted(all_keys):
        full_key = f"{prefix}{key}" if prefix else key
        old_val = old.get(key)
        new_val = new.get(key)

        if old_val == new_val:
            continue

        if isinstance(old_val, dict) and isinstance(new_val, dict):
            changes.extend(diff_configs(old_val, new_val, f"{full_key}."))
        elif key not in old:
            changes.append(f"+ {full_key}: {_format_val(new_val)}")
        elif key not in new:
            changes.append(f"- {full_key}: {_format_val(old_val)}")
        else:
            changes.append(f"  {full_key}: {_format_val(old_val)} → {_format_val(new_val)}")

    return changes


def _format_val(val: Any) -> str:
    """Format a value for display in diffs."""
    if isinstance(val, float):
        return f"{val:.2e}" if val < 0.001 or val > 1000 else f"{val}"
    if isinstance(val, dict):
        return "{...}"
    if isinstance(val, list):
        if len(val) <= 3:
            return str(val)
        return f"[{len(val)} items]"
    return str(val)


def hash_config(config: dict[str, Any]) -> str:
    """Generate a deterministic SHA256 hash of a config dictionary."""
    serialized = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def apply_patch(
    base_config: dict[str, Any],
    patch: dict[str, Any],
) -> dict[str, Any]:
    """Apply a partial config patch to a base config.

    The patch is a nested dict where only changed keys are present.
    """
    return _deep_merge(base_config, patch)
