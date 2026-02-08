# src/planning_rl/td/utils.py
from __future__ import annotations

from planning_rl.utils.seed import seed32_from


def episode_seed(*, base_seed: int, env_idx: int, episode_idx: int) -> int:
    stream_id = (int(env_idx) << 20) ^ int(episode_idx)
    return int(seed32_from(base_seed=int(base_seed), stream_id=int(stream_id)))


__all__ = ["episode_seed"]
