from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(experiment_config: str | Path) -> dict[str, Any]:
    experiment_path = Path(experiment_config).resolve()
    base_path = PROJECT_ROOT / "configs" / "base.yaml"
    with base_path.open() as stream:
        base = yaml.safe_load(stream) or {}
    with experiment_path.open() as stream:
        experiment = yaml.safe_load(stream) or {}
    config = _merge(base, experiment)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    feature_mode = config["model"].get("input_features")
    if feature_mode not in {"rgb", "rgb_sobel"}:
        raise ValueError("model.input_features must be 'rgb' or 'rgb_sobel'.")
    if config["model"]["name"] != "efficientnet_b0":
        raise ValueError("This initial repository supports model.name=efficientnet_b0.")
    if not config["experiment"].get("id"):
        raise ValueError("experiment.id is required.")
    if not config["data"].get("classes"):
        raise ValueError("data.classes must not be empty.")


def input_channels(config: dict[str, Any]) -> int:
    return 4 if config["model"]["input_features"] == "rgb_sobel" else 3


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path
