# src/tetris_rl/game/rendering/pygame/renderer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pygame

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


def _state_get(state: Any, key: str, default: Any = None) -> Any:
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def _extract_game_metrics(env_info: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """
    Renderer must NOT compute metrics.

    With the new env_info schema (macro_info.build_step_info_update), we expect:
      env_info["tf"] contains:
        - holes
        - max_height
        - bumpiness
        - agg_height
      (plus deltas, etc.)

    Sidebar expects game_metrics dict (optional). We provide a minimal stable subset.
    """
    if not isinstance(env_info, dict):
        return None

    tf = env_info.get("tf", None)
    if not isinstance(tf, dict):
        return None

    return {
        "holes": tf.get("holes", None),
        "max_height": tf.get("max_height", None),
        "bumpiness": tf.get("bumpiness", None),
        "agg_height": tf.get("agg_height", None),
    }


class TetrisRenderer:
    def __init__(
        self,
        *,
        cell: int,
        show_grid_lines: bool,
        palette: Optional[Palette] = None,
        hud_height: int = 0,
    ) -> None:
        self.cell = int(cell)
        self.show_grid_lines = bool(show_grid_lines)
        self.palette = palette or Palette()
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
        title: str = "RL-Tetris | watch",
        sidebar_w: int = SIDEBAR_W,
    ) -> tuple[pygame.Surface, Layout]:
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
        engine: Any = None,  # Rust engine (PyO3), used for UI-only preview masks
    ) -> None:
        grid = _state_get(state, "grid", None)
        if grid is None:
            screen.fill(self.palette.bg)
            blit_text(
                screen=screen,
                font=self.fonts.main,
                text='State has no grid (state["grid"] expected).',
                pos=(20, 20),
                color=(240, 90, 90),
            )
            return

        # Renderer decides whether to show the "active piece preview":
        # - when NOT paused => ghost is None => show active preview
        # - when paused     => ghost exists  => suppress active preview (ghost replaces it)
        show_active_preview = ghost is None

        game_metrics = _extract_game_metrics(env_info)

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
            cache=self.cache,
            engine=engine,
            show_active=bool(show_active_preview),
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
            cache=self.cache,
            font_small=self.fonts.small,
            font_tiny=self.fonts.tiny,
            engine=engine,
        )

    def hud_height_for_lines(self, *, n_lines: int, pad_y: int = 10, gap_y: int = 3) -> int:
        return hud_panel_height_for_lines(
            fonts=HudFonts(header=self.fonts.small, body=self.fonts.tiny),
            n_lines=n_lines,
            pad_y=pad_y,
            gap_y=gap_y,
        )
