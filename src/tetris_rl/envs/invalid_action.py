# src/tetris_rl/envs/invalid_action.py
from __future__ import annotations

from typing import Literal

InvalidActionPolicy = Literal[
    "noop",       # do nothing (no engine step)
    "terminate",  # end episode immediately
]

__all__ = [
    "InvalidActionPolicy",
]
