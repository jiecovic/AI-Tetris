# src/tetris_rl/core/config/io.py
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

from tetris_rl.core.config.root import DataGenConfig, ExperimentConfig, ImitationExperimentConfig


def _strip_hydra_key(data: dict[Any, Any]) -> dict[str, Any]:
    out = dict(data)
    out.pop("hydra", None)
    return cast(dict[str, Any], out)


def to_plain_dict(cfg: Any) -> dict[str, Any]:
    if isinstance(cfg, BaseModel):
        return cfg.model_dump(mode="json")
    if isinstance(cfg, DictConfig):
        data = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(data, dict):
            raise TypeError("config must resolve to a mapping")
        return _strip_hydra_key(cast(dict[Any, Any], data))
    raise TypeError(f"unsupported config type: {type(cfg).__name__}")


def load_yaml(path: Path) -> dict[str, Any]:
    cfg_path = Path(path)
    cfg = OmegaConf.load(cfg_path)
    data = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(data, dict):
        raise TypeError(f"config({path}) must be a mapping")
    return _strip_hydra_key(cast(dict[Any, Any], data))


def _normalize_legacy_policy_selector_fields(data: dict[str, Any]) -> dict[str, Any]:
    # Persisted run configs may still carry removed bootstrap fields under learn.
    # Keep authored-config validation strict elsewhere and normalize only at load time.
    out = dict(data)
    learn_obj = out.get("learn", None)
    if not isinstance(learn_obj, dict):
        return out
    learn = dict(learn_obj)
    learn.pop("policy_init", None)
    learn.pop("resume", None)
    out["learn"] = learn
    out.pop("policy_init", None)
    return out


def load_experiment_config(path: Path, *, allow_legacy_policy_selector: bool = False) -> ExperimentConfig:
    data = load_yaml(path)
    if allow_legacy_policy_selector:
        data = _normalize_legacy_policy_selector_fields(data)
    return ExperimentConfig.model_validate(data)


def load_datagen_config(path: Path) -> DataGenConfig:
    return DataGenConfig.model_validate(load_yaml(path))


def load_imitation_config(path: Path, *, allow_legacy_policy_selector: bool = False) -> ImitationExperimentConfig:
    data = load_yaml(path)
    if allow_legacy_policy_selector:
        data = _normalize_legacy_policy_selector_fields(data)
    return ImitationExperimentConfig.model_validate(data)


__all__ = [
    "to_plain_dict",
    "load_yaml",
    "load_experiment_config",
    "load_datagen_config",
    "load_imitation_config",
]
