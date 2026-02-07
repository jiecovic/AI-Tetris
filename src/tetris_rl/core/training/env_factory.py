# src/tetris_rl/core/training/env_factory.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from tetris_rl.core.runs.config import RunConfig
from tetris_rl.core.envs.factory import make_env_from_cfg


@dataclass(frozen=True)
class BuiltVecEnv:
    vec_env: VecEnv


def _spawn_env_seeds(base_seed: int, n_envs: int) -> list[int]:
    ss = np.random.SeedSequence(int(base_seed))
    children = ss.spawn(int(n_envs))
    return [int(c.generate_state(1, dtype=np.uint32)[0]) for c in children]


def make_vec_env_from_cfg(*, cfg: dict[str, Any], run_cfg: RunConfig) -> BuiltVecEnv:
    """
    Build VecEnv using:
      - raw cfg for env wiring (cfg.env + cfg.env.game)
      - RunConfig for runtime wiring (seed, n_envs, vec backend)

    IMPORTANT (Rust engine + Windows spawn):
      - Do not construct/hold a PyO3 engine in the parent.
      - Each worker must construct its own engine inside make_env_from_cfg().
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a mapping, got {type(cfg)!r}")
    root = cfg

    base_seed = int(run_cfg.seed)
    n_envs = max(1, int(run_cfg.n_envs))
    vec_kind = str(run_cfg.vec).strip().lower()

    set_random_seed(int(base_seed))
    env_seeds = _spawn_env_seeds(int(base_seed), int(n_envs))

    def make_one(rank: int):
        def _thunk():
            # Engine is created inside make_env_from_cfg() (per-process).
            # Pass per-rank seed so the initial engine instance differs across workers.
            built = make_env_from_cfg(cfg=root, seed=int(env_seeds[int(rank)]))

            # Monitor expects a Gym Env, not your wrapper object.
            env = Monitor(built.env)

            # IMPORTANT: do NOT reset here.
            # VecEnv/SB3 will reset when needed; manual reset can double-reset
            # and create confusing Monitor episode boundaries.
            return env

        return _thunk

    env_fns = [make_one(i) for i in range(int(n_envs))]
    if vec_kind == "subproc":
        vec_env: VecEnv = SubprocVecEnv(env_fns, start_method="spawn")
    elif vec_kind == "dummy":
        vec_env = DummyVecEnv(env_fns)
    else:
        raise ValueError(f"run_cfg.vec must be 'subproc' or 'dummy' (got {run_cfg.vec!r})")

    return BuiltVecEnv(vec_env=vec_env)


__all__ = ["BuiltVecEnv", "make_vec_env_from_cfg"]
