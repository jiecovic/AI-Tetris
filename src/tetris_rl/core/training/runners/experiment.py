# src/tetris_rl/core/training/runners/experiment.py
from __future__ import annotations

from omegaconf import DictConfig

from tetris_rl.core.training.runners.ga import run_ga_experiment
from tetris_rl.core.training.runners.imitation import run_imitation_experiment
from tetris_rl.core.training.runners.ppo import run_ppo_experiment


def run_experiment(cfg: DictConfig) -> int:
    cfg_dict = cfg if isinstance(cfg, DictConfig) else DictConfig(cfg)
    keys = set(cfg_dict.keys())

    if "algo" in keys:
        algo_block = cfg_dict.get("algo", {}) or {}
        algo_type = str(algo_block.get("type", "")).strip().lower()
        if algo_type == "imitation":
            return run_imitation_experiment(cfg)
        if algo_type == "ga":
            return run_ga_experiment(cfg)
        return run_ppo_experiment(cfg)

    raise ValueError("unknown config shape: expected imitation or algo.type")


__all__ = ["run_experiment"]
