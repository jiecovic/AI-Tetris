# src/tetris_rl/game/rendering/pygame/surf.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pygame

Color = Tuple[int, int, int]


@dataclass
class SurfaceCache:
    """
    Cache small surfaces (e.g., cell-sized blocks) by (size, color).
    This avoids re-allocating surfaces every frame.
    """

    _cells: Dict[Tuple[int, Color], pygame.Surface]

    def __init__(self) -> None:
        self._cells = {}

    def cell(self, *, size: int, color: Color) -> pygame.Surface:
        key = (int(size), color)
        surf = self._cells.get(key)
        if surf is None:
            s = int(size)
            surf = pygame.Surface((s, s), flags=pygame.SRCALPHA)
            surf.fill(color)
            self._cells[key] = surf
        return surf


def blit_text(
        *,
        screen: pygame.Surface,
        font: pygame.font.Font,
        text: str,
        pos: Tuple[int, int],
        color: Color,
) -> None:
    img = font.render(text, True, color)
    screen.blit(img, pos)
