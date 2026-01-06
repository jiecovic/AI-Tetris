# src/tetris_rl/game/rendering/pygame/renderer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pygame

from tetris_rl.game.core.metrics import board_snapshot_metrics_from_grid
from tetris_rl.game.core.pieceset import PieceSet
from tetris_rl.game.rendering.pygame.grid import draw_grid
from tetris_rl.game.rendering.pygame.hud_panel import HudFonts, draw_hud_panel, hud_panel_height_for_lines
from tetris_rl.game.rendering.pygame.palette import Palette, Color
from tetris_rl.game.rendering.pygame.sidebar import SIDEBAR_W, draw_sidebar
from tetris_rl.game.rendering.pygame.surf import SurfaceCache, blit_text
from tetris_rl.game.rendering.pygame.window import Layout, WindowSpec, compute_layout, create_window

__all__ = ["Color", "Palette", "TetrisRenderer"]


@dataclass(frozen=True)
class Fonts:
    main: pygame.font.Font
    small: pygame.font.Font
    tiny: pygame.font.Font


class TetrisRenderer:
    def __init__(
            self,
            *,
            cell: int,
            show_grid_lines: bool,
            pieces: PieceSet,
            palette: Optional[Palette] = None,
            hud_height: int = 0,
    ) -> None:
        self.cell = int(cell)
        self.show_grid_lines = bool(show_grid_lines)
        self.palette = palette or Palette()
        self.pieces = pieces
        self.hud_height = int(hud_height)

        main = pygame.font.SysFont("consolas", 18) or pygame.font.SysFont(None, 18)
        small = pygame.font.SysFont("consolas", 16) or pygame.font.SysFont(None, 16)
        tiny = pygame.font.SysFont("consolas", 14) or pygame.font.SysFont(None, 14)
        self.fonts = Fonts(main=main, small=small, tiny=tiny)

        self.cache = SurfaceCache()

    def init_window(
            self,
            *,
            board_h: int,
            board_w: int,
            hud_text: Optional[str],
            title: str = "RL-Tetris | manual play",
            sidebar_w: int = SIDEBAR_W,
    ) -> tuple[pygame.Surface, Layout]:
        """
        Renderer-owned window/layout creation.
        Callers (watch/manual) should not do any geometry.
        """
        n_lines = 0
        if hud_text:
            raw = hud_text.splitlines()
            while raw and raw[-1].strip() == "":
                raw.pop()
            n_lines = len(raw)

        self.hud_height = self.hud_height_for_lines(n_lines=n_lines) if n_lines > 0 else 0

        layout = compute_layout(
            board_h=int(board_h),
            board_w=int(board_w),
            cell=int(self.cell),
            hud_h=int(self.hud_height),
            sidebar_w=int(sidebar_w),
        )

        spec = WindowSpec(width=layout.window.width, height=layout.window.height, title=str(title))
        screen = create_window(spec)
        return screen, layout

    def render(
            self,
            *,
            screen: pygame.Surface,
            state: Any,
            reward: float,
            done: bool,
            layout: Layout,
            hud_text: Optional[str] = None,
            env_info: Optional[dict[str, Any]] = None,
            ghost: Optional[dict[str, Any]] = None,
    ) -> None:
        grid = getattr(state, "grid", None)
        if grid is None:
            screen.fill(self.palette.bg)
            blit_text(
                screen=screen,
                font=self.fonts.main,
                text="State has no grid (state.grid expected).",
                pos=(20, 20),
                color=(240, 90, 90),
            )
            return

        # GAME metrics are derived from the LOCKED board grid (env-independent).
        try:
            m = board_snapshot_metrics_from_grid(grid)
            game_metrics = {"holes": int(m.holes), "max_height": int(m.max_height)}
        except Exception:
            game_metrics = {"holes": None, "max_height": None}

        screen.fill(self.palette.bg)

        if self.hud_height > 0 and hud_text:
            draw_hud_panel(
                screen=screen,
                palette=self.palette,
                fonts=HudFonts(header=self.fonts.small, body=self.fonts.tiny),
                hud_height=self.hud_height,
                text=hud_text,
            )

        draw_grid(
            screen=screen,
            grid=grid,
            state=state,
            ghost=ghost,
            origin=layout.origin,
            margin=layout.margin,
            cell=self.cell,
            show_grid_lines=self.show_grid_lines,
            palette=self.palette,
            pieces=self.pieces,
            cache=self.cache,
        )

        draw_sidebar(
            screen=screen,
            state=state,
            env_info=env_info,
            game_metrics=game_metrics,
            reward=reward,
            done=done,
            x=layout.sidebar_x,
            y=layout.sidebar_y,
            w=layout.sidebar_w,
            origin=layout.origin,
            margin=layout.margin,
            grid=grid,
            cell=self.cell,
            palette=self.palette,
            pieces=self.pieces,
            cache=self.cache,
            font_small=self.fonts.small,
            font_tiny=self.fonts.tiny,
        )

    def hud_height_for_lines(self, *, n_lines: int, pad_y: int = 10, gap_y: int = 3) -> int:
        return hud_panel_height_for_lines(
            fonts=HudFonts(header=self.fonts.small, body=self.fonts.tiny),
            n_lines=n_lines,
            pad_y=pad_y,
            gap_y=gap_y,
        )
