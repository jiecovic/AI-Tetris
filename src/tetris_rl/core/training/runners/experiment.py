# src/tetris_rl/core/training/runners/experiment.py
from __future__ import annotations

from omegaconf import DictConfig

from tetris_rl.core.training.runners.ga import run_ga_experiment
from tetris_rl.core.training.runners.rl import run_rl_experiment


def run_experiment(cfg: DictConfig) -> int:
    cfg_dict = cfg if isinstance(cfg, DictConfig) else DictConfig(cfg)
    keys = set(cfg_dict.keys())

    if "algo" in keys:
        algo_block = cfg_dict.get("algo", {}) or {}
        algo_type = str(algo_block.get("type", "")).strip().lower()
        if algo_type == "ga":
            return run_ga_experiment(cfg)
        return run_rl_experiment(cfg)

    raise ValueError("unknown config shape: expected config with algo.type")


__all__ = ["run_experiment"]
