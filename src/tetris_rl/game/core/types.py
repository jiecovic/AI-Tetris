# src/tetris_rl/game/core/types.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np


class Action(Enum):
    LEFT = auto()
    RIGHT = auto()
    SOFT_DROP = auto()
    HARD_DROP = auto()
    ROT_CW = auto()
    ROT_CCW = auto()
    HOLD = auto()


@dataclass(frozen=True)
class ActivePiece:
    kind: str
    rot: int
    x: int
    y: int


@dataclass(frozen=True)
class State:
    """
    Render-/env-facing snapshot.

    Locked contracts (no legacy support):

    Board (grid):
      - grid is the LOCKED board (no active overlay).
      - grid cell values are "board ids".
      - board ids are configured by env obs mode:
          * binary board => cells in {0,1}
          * categorical board => cells in {0..K} where 0=empty and 1..K encode kind_idx+1

    Specials (ACTIVE/NEXT):
      - ALWAYS tetromino kind indices:
          active_kind_idx, next_kind_idx ∈ {0..K-1}
      - no empty token
      - no piece_id/board_id representation
      - engine must expose these explicitly
    """

    grid: np.ndarray
    score: int
    lines: int
    level: int
    game_over: bool

    active: ActivePiece
    next_kind: str

    # Strict special-token contract for tokenizer/model:
    active_kind_idx: int
    next_kind_idx: int

    # Convenience mirror (renderer/UI often wants this directly)
    active_rot: int
