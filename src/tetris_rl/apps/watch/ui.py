# src/tetris_rl/apps/watch/ui.py
from __future__ import annotations

import time
from typing import Any

from tetris_rl.core.agents.actions import choose_action
from tetris_rl.core.runtime.hud_adapter import env_info_for_renderer
from tetris_rl.core.runtime.hud_adapter import from_info as hud_from_info
from tetris_rl.core.runtime.run_context import RunContext
from tetris_rl.ui.runtime.hud_text import HudFormatter, HudSnapshot
from tetris_rl.ui.runtime.live_stats import StepWindow
from tetris_rl.ui.runtime.manual_cursor import ManualMacroCursor
from tetris_rl.ui.runtime.speed_control import RateMeter, SpeedControl


def run_watch_loop(*, args: Any, ctx: RunContext) -> int:
    env = ctx.env
    game = ctx.game
    ckpt = ctx.ckpt
    model = ctx.model
    planning_policy = ctx.planning_policy
    expert_policy = ctx.expert_policy
    poller = ctx.poller
    planning_poller = ctx.planning_poller
    algo_type = ctx.algo_type

    window = StepWindow(capacity=max(0, int(args.window_steps)))
    hud = HudFormatter(window_steps=int(args.window_steps))

    import pygame

    from tetris_rl.ui.rendering.pygame.renderer import TetrisRenderer

    pygame.init()
    clock = pygame.time.Clock()

    speed = SpeedControl(render_fps_cap=int(args.fps), step_ms=int(args.step_ms))
    speed.clamp()

    obs, info = env.reset(seed=int(args.seed))
    window.reset_episode()
    state: Any = game.snapshot(include_grid=True, visible=False)

    renderer = TetrisRenderer(
        cell=int(args.cell),
        show_grid_lines=bool(args.show_grid),
        hud_height=0,
    )

    demo_hud_text = hud.format_text(
        HudSnapshot(
            run_name=str(args.run),
            mode=str(getattr(env, "action_mode", "?")),
            ckpt_name=str(getattr(ckpt, "stem", str(ckpt)) if ckpt is not None else "none"),
            paused=False,
            seed=int(args.seed),
            reload_every_s=float(args.reload),
            reloads=0,
            last_reload_age_s=0.0,
            episode_idx=0,
            episode_step=0,
            episode_reward=0.0,
            last_step_reward=0.0,
            next_kind="?",
            piece_rule="?",
            win_capacity=int(args.window_steps),
            win_steps=0,
            win_avg_r=0.0,
            win_avg_lines=0.0,
            win_invalid_pct=0.0,
            win_avg_score=0.0,
            win_avg_ep_len=0.0,
            win_action_entropy=0.0,
        )
    )

    state = game.snapshot(include_grid=True, visible=False)
    grid = state["grid"]
    board_h = len(grid)
    board_w = len(grid[0]) if board_h else 10

    screen, layout = renderer.init_window(
        board_h=board_h,
        board_w=board_w,
        hud_text=demo_hud_text,
        title="RL-Tetris | watch",
    )

    if not bool(args.no_repeat):
        pygame.key.set_repeat(140, 35)

    cursor = ManualMacroCursor(game=game, env=env)
    if isinstance(state, dict):
        cursor.sync_from_snapshot(state)

    running = True
    paused = False

    last_reload_at_s: float | None = time.time()

    # meters (actual rates)
    sps_meter = RateMeter(window=60)
    fps_meter = RateMeter(window=60)

    # fixed-step accumulator for ms-mode
    acc_s = 0.0
    last_frame_t = time.perf_counter()
    last_speed_adjust_s = time.time()

    SPEED_ADJUST_REPEAT_S = 0.08

    # cap sim steps per frame so input stays responsive in uncapped mode
    MAX_STEPS_PER_FRAME = 500

    def _maybe_reload() -> None:
        nonlocal model, ckpt, algo_type, last_reload_at_s, planning_policy
        now_s = time.time()
        if poller is not None and model is not None:
            maybe = poller.maybe_reload(now_s=now_s)
            if maybe is not None:
                model, ckpt, algo_type = maybe
                last_reload_at_s = now_s
        if planning_poller is not None:
            maybe_plan = planning_poller.maybe_reload(now_s=now_s)
            if maybe_plan is not None:
                planning_policy, ckpt = maybe_plan
                algo_type = "td"
                last_reload_at_s = now_s

    def _advance_one() -> None:
        nonlocal obs, info, state

        a = choose_action(
            args=args,
            algo_type=str(algo_type),
            model=model,
            obs=obs,
            env=env,
            game=game,
            expert_policy=expert_policy,
            planning_policy=planning_policy,
        )

        obs2, r, terminated, truncated, info2 = env.step(a)
        obs = obs2
        info = info2
        state = game.snapshot(include_grid=True, visible=False)

        h2 = hud_from_info(info2)
        window.push(
            step_reward=float(r),
            cleared_lines=int(h2.cleared_lines),
            invalid=int(h2.invalid_action),
            masked=int(h2.masked_action),
            score_delta=float(h2.delta_score),
            action_id=h2.action_id,
            action_dim=h2.action_dim,
            episode_done=bool(terminated or truncated),
        )

        if terminated or truncated:
            obs_r, info_r = env.reset()
            obs = obs_r
            info = info_r
            window.reset_episode()
            state = game.snapshot(include_grid=True, visible=False)

        if isinstance(state, dict):
            cursor.sync_from_snapshot(state)

        sps_meter.tick()

    def _overlay_text() -> str:
        fps_cap = max(1, int(speed.render_fps_cap))
        fps_act = fps_meter.rate_hz()
        sps_act = sps_meter.rate_hz()
        sps_tgt = speed.target_sps()
        tgt = "max" if sps_tgt is None else f"{sps_tgt:.1f}"
        return f"SPS {sps_act:5.1f}/{tgt}   FPS {fps_act:5.1f}/{fps_cap}   mode={speed.label()}"

    def _render_once() -> None:
        h = hud_from_info(info)
        ws = window.summary()
        denom = float(max(1, int(ws.steps)))

        avg_lines = float(ws.sum_lines) / denom
        invalid_pct = 100.0 * float(ws.avg_invalid)

        now_s = time.time()
        last_reload_age_s = float("inf") if last_reload_at_s is None else max(0.0, now_s - float(last_reload_at_s))

        reloads = int(getattr(poller, "reload_count", 0))
        if planning_poller is not None:
            reloads = max(reloads, int(getattr(planning_poller, "reload_count", 0)))

        snap = HudSnapshot(
            run_name=str(args.run),
            mode=str(h.action_mode),
            ckpt_name=str(getattr(ckpt, "stem", str(ckpt)) if ckpt is not None else "none"),
            paused=bool(paused),
            seed=int(args.seed),
            reload_every_s=float(args.reload),
            reloads=reloads,
            last_reload_age_s=float(last_reload_age_s),
            episode_idx=int(h.episode_idx),
            episode_step=int(h.episode_step),
            episode_reward=float(ws.cur_episode_reward),
            last_step_reward=float(ws.last_step_reward),
            next_kind=(str(h.next_kind)[:1] if str(h.next_kind) else "?"),
            piece_rule=str(h.piece_rule),
            win_capacity=int(args.window_steps),
            win_steps=int(ws.steps),
            win_avg_r=float(ws.avg_reward),
            win_avg_lines=float(avg_lines),
            win_invalid_pct=float(invalid_pct),
            win_avg_score=float(ws.avg_score_delta),
            win_avg_ep_len=float(ws.avg_episode_len),
            win_action_entropy=float(ws.action_entropy),
        )
        hud_text = hud.format_text(snap)

        env_info = env_info_for_renderer(info)
        ghost = cursor.ghost_for_render(True) if bool(paused) else None

        renderer.render(
            screen=screen,
            state=state,
            ghost=ghost,
            env_info=env_info,
            reward=float(ws.last_step_reward),
            done=bool(h.game_over),
            layout=layout,
            hud_text=hud_text,
            engine=game,
            overlay_text=_overlay_text(),
        )
        pygame.display.flip()
        fps_meter.tick()

    while running:
        # Render loop throttle (UI loop)
        clock.tick(int(speed.render_fps_cap))

        # frame timing for accumulator
        now_t = time.perf_counter()
        frame_dt = now_t - last_frame_t
        last_frame_t = now_t

        # events first (keep window responsive even if sim is heavy)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break

                if event.key == pygame.K_LEFTBRACKET:  # [
                    speed.handle_slower()
                    speed.clamp()
                    last_speed_adjust_s = time.time()
                    sps_meter.trim_to_last(2)
                    fps_meter.trim_to_last(2)
                    continue

                if event.key == pygame.K_RIGHTBRACKET:  # ]
                    speed.handle_faster()
                    speed.clamp()
                    last_speed_adjust_s = time.time()
                    sps_meter.trim_to_last(2)
                    fps_meter.trim_to_last(2)
                    continue

                if event.key == pygame.K_EQUALS:  # =
                    speed.handle_max()
                    speed.clamp()
                    last_speed_adjust_s = time.time()
                    sps_meter.trim_to_last(2)
                    fps_meter.trim_to_last(2)
                    continue

                if event.key == pygame.K_p:
                    paused = not paused
                    continue

                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    # Manual reset should also clear rolling HUD stats/history.
                    window.clear()
                    state = game.snapshot(include_grid=True, visible=False)
                    acc_s = 0.0
                    sps_meter.trim_to_last(2)
                    fps_meter.trim_to_last(2)
                    continue

                if event.key == pygame.K_n:
                    _maybe_reload()
                    _advance_one()
                    continue

                if paused:
                    if event.key == pygame.K_a:
                        cursor.move_col(dx=-1)
                    elif event.key == pygame.K_d:
                        cursor.move_col(dx=+1)
                    elif event.key == pygame.K_q:
                        cursor.move_rot(dr=-1)
                    elif event.key == pygame.K_e:
                        cursor.move_rot(dr=+1)
                    elif event.key == pygame.K_SPACE:
                        # commit cursor action as a *real* step
                        a = cursor.action_for_commit()
                        obs2, r, terminated, truncated, info2 = env.step(a)
                        obs, info = obs2, info2
                        state = game.snapshot(include_grid=True, visible=False)
                        if isinstance(state, dict):
                            cursor.sync_from_snapshot(state)
                        sps_meter.tick()

        if not running:
            break

        # apply speed adjustments while key is held (more responsive)
        keys = pygame.key.get_pressed()
        now_s = time.time()
        if (now_s - last_speed_adjust_s) >= SPEED_ADJUST_REPEAT_S:
            if keys[pygame.K_LEFTBRACKET]:
                speed.handle_slower()
                speed.clamp()
                last_speed_adjust_s = now_s
                sps_meter.trim_to_last(2)
                fps_meter.trim_to_last(2)
            elif keys[pygame.K_RIGHTBRACKET]:
                speed.handle_faster()
                speed.clamp()
                last_speed_adjust_s = now_s
                sps_meter.trim_to_last(2)
                fps_meter.trim_to_last(2)
            elif keys[pygame.K_EQUALS]:
                speed.handle_max()
                speed.clamp()
                last_speed_adjust_s = now_s
                sps_meter.trim_to_last(2)
                fps_meter.trim_to_last(2)

        # simulation updates (decoupled from render)
        if not paused:
            _maybe_reload()

            interval_ms = speed.interval_ms()
            if interval_ms == 0:
                # uncapped: run multiple steps this frame (but cap to keep input responsive)
                for _ in range(MAX_STEPS_PER_FRAME):
                    _advance_one()
            elif interval_ms is None:
                # frame-locked: exactly one sim step per frame
                _advance_one()
            else:
                # ms-mode: fixed timestep accumulator
                acc_s += frame_dt
                step_s = float(interval_ms) / 1000.0
                # avoid spiral: cap steps per frame
                steps = 0
                while acc_s >= step_s and steps < MAX_STEPS_PER_FRAME:
                    _advance_one()
                    acc_s -= step_s
                    steps += 1

        # render exactly once per frame (decoupled)
        _render_once()

    pygame.quit()
    env.close()
    return 0


__all__ = ["run_watch_loop"]
