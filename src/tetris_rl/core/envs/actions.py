# src/tetris_rl/core/envs/actions.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

ActionMode = Literal["discrete", "multidiscrete"]

InvalidActionPolicy = Literal[
    "noop",       # do nothing (no engine step)
    "terminate",  # end episode immediately
]


@dataclass(frozen=True)
class ActionRequest:
    requested_rot: int
    requested_col: int
    requested_action_id: int


@dataclass(frozen=True)
class MaskStats:
    masked_action: bool
    masked_action_count: int
    action_dim: int


def action_mask_bool(game: Any) -> np.ndarray:
    m_u8 = np.asarray(game.action_mask(), dtype=np.uint8)
    return (m_u8.astype(np.uint8, copy=False) != 0).reshape(-1)


def resolve_action_request(*, action: Any, action_mode: str, game: Any) -> ActionRequest:
    mode = str(action_mode).strip().lower()
    if mode == "discrete":
        requested_action_id = int(action)
        try:
            rot, col = game.decode_action_id(int(requested_action_id))
            return ActionRequest(int(rot), int(col), int(requested_action_id))
        except Exception:
            return ActionRequest(-1, -1, int(requested_action_id))

    if isinstance(action, (tuple, list)) and len(action) == 2:
        rot, col = action
        requested_rot = int(rot)
        requested_col = int(col)
    else:
        arr = np.asarray(action).reshape(-1)
        if arr.size != 2:
            raise TypeError(f"invalid action for action_mode={mode!r}: {action!r}")
        requested_rot = int(arr[0])
        requested_col = int(arr[1])

    requested_action_id = int(game.encode_action_id(int(requested_rot), int(requested_col)))
    return ActionRequest(int(requested_rot), int(requested_col), int(requested_action_id))


def mask_stats_for_action_id(*, action_id: int, mask: np.ndarray) -> MaskStats:
    aid = int(action_id)
    masked = bool(aid < 0 or aid >= int(mask.size) or (not bool(mask[aid])))
    masked_count = int((~mask).sum())
    return MaskStats(masked_action=bool(masked), masked_action_count=int(masked_count), action_dim=int(mask.size))


__all__ = [
    "ActionMode",
    "InvalidActionPolicy",
    "ActionRequest",
    "MaskStats",
    "action_mask_bool",
    "resolve_action_request",
    "mask_stats_for_action_id",
]
