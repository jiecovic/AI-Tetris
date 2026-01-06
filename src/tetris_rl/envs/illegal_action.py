# src/tetris_rl/envs/illegal_action.py
from __future__ import annotations

from typing import Literal, Optional

import numpy as np

IllegalActionPolicy = Literal[
    "noop",          # do nothing (no hard_drop)
    "closest_legal", # map to a deterministic legal action
    "random_legal",  # sample a random legal action
    "terminate",     # end episode immediately
]


def pick_first_legal_action_id(mask: np.ndarray) -> Optional[int]:
    m = np.asarray(mask, dtype=bool).reshape(-1)
    idx = np.flatnonzero(m)
    if idx.size == 0:
        return None
    return int(idx[0])


def pick_random_legal_action_id(mask: np.ndarray, *, rng: np.random.Generator) -> Optional[int]:
    m = np.asarray(mask, dtype=bool).reshape(-1)
    idx = np.flatnonzero(m)
    if idx.size == 0:
        return None
    j = int(rng.integers(0, idx.size))
    return int(idx[j])


def pick_closest_legal_action_id(
    mask: np.ndarray,
    *,
    requested_rot: int,
    requested_col: int,
    board_w: int,
    max_rots: int,
) -> Optional[int]:
    """
    Deterministic "closest legal":
      1) try same rot, nearest col (min abs distance)
      2) else fallback to first legal anywhere
    """
    m = np.asarray(mask, dtype=bool).reshape(-1)
    bw = int(board_w)
    mr = int(max_rots)
    if bw <= 0 or mr <= 0:
        return pick_first_legal_action_id(m)

    action_dim = int(mr * bw)
    if int(m.shape[0]) != action_dim:
        return pick_first_legal_action_id(m)

    r = int(requested_rot)
    c = int(requested_col)

    if 0 <= r < mr:
        base = int(r * bw)
        best_aid: Optional[int] = None
        best_dist = 10**9
        for col in range(bw):
            aid = int(base + col)
            if bool(m[aid]):
                d = abs(int(col) - c)
                if d < best_dist:
                    best_dist = int(d)
                    best_aid = int(aid)
                    if d == 0:
                        break
        if best_aid is not None:
            return int(best_aid)

    return pick_first_legal_action_id(m)


__all__ = [
    "IllegalActionPolicy",
    "pick_first_legal_action_id",
    "pick_random_legal_action_id",
    "pick_closest_legal_action_id",
]
