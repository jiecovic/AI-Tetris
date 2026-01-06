# src/tetris_rl/runs/action_source.py
from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np


def as_action_scalar(action: Any) -> int:
    if np.isscalar(action):
        return int(action)
    arr = np.asarray(action).reshape(-1)
    return int(arr[0]) if arr.size > 0 else 0


def as_action_pair(action: Any) -> Tuple[int, int]:
    if isinstance(action, (tuple, list)) and len(action) >= 2:
        return int(action[0]), int(action[1])
    arr = np.asarray(action).reshape(-1)
    if arr.size < 2:
        raise TypeError(f"invalid multidiscrete action: {action!r}")
    return int(arr[0]), int(arr[1])


def get_action_mask(env: Any) -> Optional[np.ndarray]:
    if not hasattr(env, "action_masks"):
        return None
    try:
        m = env.action_masks()
    except Exception:
        return None
    if m is None:
        return None
    return np.asarray(m, dtype=bool).reshape(-1)


def sample_masked_discrete(env: Any) -> int:
    mask = get_action_mask(env)
    if mask is None:
        return int(env.action_space.sample())
    idx = np.flatnonzero(mask)
    if idx.size <= 0:
        return int(env.action_space.sample())
    rng = getattr(env, "np_random", None)
    if rng is None:
        return int(np.random.choice(idx))
    try:
        j = int(rng.integers(0, int(idx.size)))
        return int(idx[j])
    except Exception:
        return int(np.random.choice(idx))


def predict_action(*, algo_type: str, model: Any, obs: Any, env: Any) -> Any:
    if algo_type == "maskable_ppo" and str(getattr(env, "action_mode", "discrete")) == "discrete":
        mask = get_action_mask(env)
        action, _ = model.predict(obs, deterministic=True, action_masks=mask)
        return action
    action, _ = model.predict(obs, deterministic=True)
    return action
