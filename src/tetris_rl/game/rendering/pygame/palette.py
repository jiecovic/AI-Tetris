# src/tetris_rl/game/rendering/pygame/palette.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

Color = Tuple[int, int, int]


@dataclass(frozen=True)
class Palette:
    bg: Color = (20, 20, 24)
    panel_bg: Color = (26, 26, 30)
    empty: Color = (30, 30, 34)
    grid: Color = (45, 45, 52)
    border: Color = (90, 90, 105)

    text: Color = (220, 220, 230)
    muted: Color = (170, 170, 185)
    warn: Color = (240, 160, 90)

    matrix: Color = (0, 255, 120)
    fallback_piece: Color = (180, 180, 200)

    hidden_overlay_rgba: Tuple[int, int, int, int] = (0, 0, 0, 120)
    hidden_rows: int = 2

    hud_bg: Color = (15, 15, 17)
    hud_border: Color = (70, 70, 80)
