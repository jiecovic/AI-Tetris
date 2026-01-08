# src/tetris_rl/env_bundles/macro_actions.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

ActionMode = Literal["discrete", "multidiscrete"]


@dataclass(frozen=True)
class MacroAction:
    rot: int
    col: int
