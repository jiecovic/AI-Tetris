# src/tetris_rl/training/env_factory.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from tetris_rl.config.run_spec import RunSpec
from tetris_rl.config.schema_types import require_mapping
from tetris_rl.envs.factory import make_env_from_cfg
from tetris_rl.game.factory import make_game_from_cfg


@dataclass(frozen=True)
class BuiltVecEnv:
    vec_env: VecEnv


def _spawn_env_seeds(base_seed: int, n_envs: int) -> list[int]:
    ss = np.random.SeedSequence(int(base_seed))
    children = ss.spawn(int(n_envs))
    return [int(c.generate_state(1, dtype=np.uint32)[0]) for c in children]


def make_vec_env_from_cfg(*, cfg: dict[str, Any], run_spec: RunSpec) -> BuiltVecEnv:
    """
    Build VecEnv using:
      - raw cfg for env/game wiring (cfg.env + cfg.game)
      - RunSpec for runtime wiring (seed, n_envs, vec backend)

    Training semantics (RL vs imitation vs eval) are NOT handled here.
    This factory is strictly responsible for environment construction.
    """
    root = require_mapping(cfg, where="cfg")

    base_seed = int(run_spec.seed)
    n_envs = max(1, int(run_spec.n_envs))
    vec_kind = str(run_spec.vec).strip().lower()

    set_random_seed(int(base_seed))

    env_seeds = _spawn_env_seeds(int(base_seed), int(n_envs))

    def make_one(rank: int):
        def _thunk():
            game = make_game_from_cfg(root)
            built = make_env_from_cfg(cfg=root, game=game)
            env = Monitor(built.env)
            env.reset(seed=int(env_seeds[int(rank)]))
            return env

        return _thunk

    env_fns = [make_one(i) for i in range(int(n_envs))]
    if vec_kind == "subproc":
        vec_env: VecEnv = SubprocVecEnv(env_fns, start_method="spawn")
    elif vec_kind == "dummy":
        vec_env = DummyVecEnv(env_fns)
    else:
        raise ValueError(f"run_spec.vec must be 'subproc' or 'dummy' (got {run_spec.vec!r})")

    return BuiltVecEnv(vec_env=vec_env)


__all__ = ["BuiltVecEnv", "make_vec_env_from_cfg"]
