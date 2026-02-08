# src/tetris_rl/core/config/io.py
from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

from tetris_rl.core.config.root import DataGenConfig, ExperimentConfig, ImitationExperimentConfig


def _strip_hydra_key(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    out.pop("hydra", None)
    return out


def to_plain_dict(cfg: Any) -> dict[str, Any]:
    if isinstance(cfg, BaseModel):
        return cfg.model_dump(mode="json")
    if isinstance(cfg, DictConfig):
        data = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(data, dict):
            raise TypeError("config must resolve to a mapping")
        return _strip_hydra_key(data)
    raise TypeError(f"unsupported config type: {type(cfg).__name__}")


def load_yaml(path: Path) -> dict[str, Any]:
    cfg_path = Path(path)
    cfg = OmegaConf.load(cfg_path)
    data = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(data, dict):
        raise TypeError(f"config({path}) must be a mapping")
    return _strip_hydra_key(data)


def load_experiment_config(path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(load_yaml(path))


def load_datagen_config(path: Path) -> DataGenConfig:
    return DataGenConfig.model_validate(load_yaml(path))


def load_imitation_config(path: Path) -> ImitationExperimentConfig:
    return ImitationExperimentConfig.model_validate(load_yaml(path))


__all__ = [
    "to_plain_dict",
    "load_yaml",
    "load_experiment_config",
    "load_datagen_config",
    "load_imitation_config",
]
