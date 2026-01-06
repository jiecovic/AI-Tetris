# src/tetris_rl/game/rendering/pygame/window.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pygame

from tetris_rl.game.rendering.pygame.sidebar import SIDEBAR_W


@dataclass(frozen=True)
class WindowSpec:
    width: int
    height: int
    title: str = "RL-Tetris | manual play"


@dataclass(frozen=True)
class Layout:
    origin: Tuple[int, int]
    margin: int
    sidebar_x: int
    sidebar_y: int
    sidebar_w: int
    window_w: int
    window_h: int
    window: WindowSpec


def create_window(spec: WindowSpec) -> pygame.Surface:
    pygame.display.set_caption(spec.title)
    return pygame.display.set_mode((int(spec.width), int(spec.height)))


def compute_layout(
        *,
        board_h: int,
        board_w: int,
        cell: int,
        hud_h: int = 0,
        sidebar_w: int = SIDEBAR_W,
        right_pad: int = 8,  # keep tiny breathing room
        bottom_pad: int = 64,
) -> Layout:
    """
    Single source of truth for window geometry.

    sidebar_w defaults to sidebar.SIDEBAR_W so changing that constant actually changes the window.
    """
    ox = 24
    oy = 24 + int(hud_h)

    margin = 6
    sidebar_x = ox + int(board_w) * int(cell) + 24
    sidebar_y = oy - margin

    # Prevent 2px borders/text from touching the very edge.
    safety = 6

    window_w = int(sidebar_x) + int(sidebar_w) + int(right_pad) + int(safety)
    window_h = int(oy) + int(board_h) * int(cell) + int(bottom_pad)

    win = WindowSpec(width=int(window_w), height=int(window_h))
    return Layout(
        origin=(int(ox), int(oy)),
        margin=int(margin),
        sidebar_x=int(sidebar_x),
        sidebar_y=int(sidebar_y),
        sidebar_w=int(sidebar_w),
        window_w=int(window_w),
        window_h=int(window_h),
        window=win,
    )
