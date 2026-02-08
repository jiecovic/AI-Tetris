# src/planning_rl/ga/utils.py
from __future__ import annotations

from typing import Any

import numpy as np

from planning_rl.utils.seed import seed32_from


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    if weights.ndim == 1:
        norm = float(np.linalg.norm(weights))
        if norm <= 0.0:
            return weights
        return weights / norm
    norms = np.linalg.norm(weights, axis=1, keepdims=True)
    norms = np.where(norms <= 0.0, 1.0, norms)
    return weights / norms


def episode_seeds(*, base_seed: int, episodes: int) -> list[int]:
    return [int(seed32_from(base_seed=int(base_seed), stream_id=int(i))) for i in range(episodes)]


__all__ = ["episode_seeds", "normalize_weights", "to_jsonable"]
