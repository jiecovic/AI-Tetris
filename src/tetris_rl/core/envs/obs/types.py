# src/tetris_rl/core/envs/obs/types.py
from __future__ import annotations

from typing import Literal

# Board cell encoding mode (env observation contract)
# - "binary":       board cells are {0,1} (empty / occupied)
# - "categorical":  board cells are {0..K} where 0=empty and 1..K encode tetromino kind+1
CellMode = Literal["binary", "categorical"]
