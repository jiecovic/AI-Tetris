# src/tetris_rl/game/rendering/pygame/app.py
from __future__ import annotations

from typing import Any, Dict

import pygame

from tetris_rl.game.rendering.pygame.renderer import TetrisRenderer
from tetris_rl.game.rendering.pygame.window import WindowSpec, compute_layout, create_window


def run_manual_play(
        *,
        game: Any,
        seed: int,
        cell: int,
        fps: int,
        gravity_ms: int,
        show_grid: bool,
        no_gravity: bool,
        no_repeat: bool,
        action_resolver,
) -> int:
    try:
        state = game.reset(seed=seed)
    except TypeError:
        state = game.reset()

    grid = getattr(state, "grid", None)
    if grid is None:
        raise RuntimeError("State must expose state.grid for manual play.")

    board_h = len(grid)
    board_w = len(grid[0]) if board_h > 0 else 0

    hud_h = 0

    origin, margin, sidebar_x, sidebar_y = compute_layout(board_h=board_h, board_w=board_w, cell=cell, hud_h=hud_h)

    sidebar_w = 220
    sidebar_pad = 24
    right_margin = 32
    width = origin[0] + board_w * cell + sidebar_w + sidebar_pad + right_margin

    bottom_margin = 64
    height = origin[1] + board_h * cell + bottom_margin

    pygame.init()
    screen = create_window(WindowSpec(width=width, height=height))
    clock = pygame.time.Clock()

    if not no_repeat:
        pygame.key.set_repeat(120, 35)

    renderer = TetrisRenderer(cell=cell, show_grid_lines=show_grid, pieces=game.pieces, hud_height=hud_h)

    reward = 0.0
    done = False
    info: Dict[str, object] = {}

    KEYMAP = {
        pygame.K_a: "left",
        pygame.K_d: "right",
        pygame.K_LEFT: "left",
        pygame.K_RIGHT: "right",
        pygame.K_s: "soft_drop",
        pygame.K_DOWN: "soft_drop",
        pygame.K_w: "hard_drop",
        pygame.K_SPACE: "hard_drop",
        pygame.K_q: "rot_ccw",
        pygame.K_e: "rot_cw",
        pygame.K_z: "rot_ccw",
        pygame.K_x: "rot_cw",
        pygame.K_UP: "rot_cw",
        pygame.K_c: "hold",
        pygame.K_r: "reset",
        pygame.K_ESCAPE: "quit",
    }

    def do_step(action_name: str) -> None:
        nonlocal state, reward, done, info

        if action_name == "reset":
            try:
                state = game.reset(seed=seed)
            except TypeError:
                state = game.reset()
            reward = 0.0
            done = False
            info = {}
            return

        if done:
            return

        action = action_resolver(action_name)
        out = game.step(action)

        if isinstance(out, tuple) and len(out) >= 3:
            state = out[0]
            reward = float(out[1])
            done = bool(out[2])
            if len(out) >= 4 and isinstance(out[3], dict):
                info = out[3]
        else:
            state = out
            reward = 0.0
            done = False

    last_drop_ms = pygame.time.get_ticks()

    running = True
    while running:
        clock.tick(fps)
        now_ms = pygame.time.get_ticks()

        if (not no_gravity) and (not done):
            if now_ms - last_drop_ms >= gravity_ms:
                do_step("soft_drop")
                last_drop_ms = now_ms

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.KEYDOWN:
                a = KEYMAP.get(event.key)
                if not a:
                    continue
                if a == "quit":
                    running = False
                    break
                do_step(a)
                last_drop_ms = pygame.time.get_ticks()

        renderer.render(
            screen=screen,
            state=state,
            reward=reward,
            done=done,
            origin=origin,
            margin=margin,
            sidebar_x=sidebar_x,
            sidebar_y=sidebar_y,
            hud_text=None,
        )
        pygame.display.flip()

    pygame.quit()
    return 0
