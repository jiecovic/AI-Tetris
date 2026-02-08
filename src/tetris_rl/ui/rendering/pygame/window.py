# src/tetris_rl/ui/rendering/pygame/window.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pygame

from tetris_rl.ui.rendering.pygame.sidebar import SIDEBAR_W


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
    footer_y: int
    footer_h: int
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
    right_pad: int = 8,   # breathing room on the right
    bottom_pad: int = 8,  # breathing room BETWEEN board+sidebar outer frame and footer bar
    footer_h: int = 0,    # reserved footer bar height (below board+sidebar)
) -> Layout:
    """
    Authoritative layout for window geometry.

    IMPORTANT:
    - The board is typically rendered with an outer frame that includes `margin` pixels
      on all sides (see draw_grid usage of origin+margin).
    - Therefore the "content bottom" must include that margin, otherwise the footer
      will overlap the board frame.
    """
    ox = 24
    oy = 24 + int(hud_h)

    margin = 6

    sidebar_x = ox + int(board_w) * int(cell) + 24
    sidebar_y = oy - margin

    safety = 6
    window_w = int(sidebar_x) + int(sidebar_w) + int(right_pad) + int(safety)

    # Board outer rect is usually:
    #   top    = oy - margin
    #   height = board_h*cell + 2*margin
    # => bottom = (oy - margin) + (board_h*cell + 2*margin) = oy + board_h*cell + margin
    board_outer_bottom = int(oy) + int(board_h) * int(cell) + int(margin)

    footer_h_i = max(0, int(footer_h))
    window_h = int(board_outer_bottom) + int(bottom_pad) + int(footer_h_i)

    footer_y = int(window_h) - int(footer_h_i)

    win = WindowSpec(width=int(window_w), height=int(window_h))
    return Layout(
        origin=(int(ox), int(oy)),
        margin=int(margin),
        sidebar_x=int(sidebar_x),
        sidebar_y=int(sidebar_y),
        sidebar_w=int(sidebar_w),
        footer_y=int(footer_y),
        footer_h=int(footer_h_i),
        window_w=int(window_w),
        window_h=int(window_h),
        window=win,
    )
