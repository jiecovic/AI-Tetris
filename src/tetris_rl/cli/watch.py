# src/tetris_rl/cli/watch.py
from __future__ import annotations

import argparse
import time
from typing import Any, Optional

from tetris_rl.config.snapshot import load_yaml
from tetris_rl.config.train_spec import parse_train_spec
from tetris_rl.envs.factory import make_env_from_cfg
from tetris_rl.runs.action_source import (
    as_action_pair,
    as_action_scalar,
    predict_action,
    sample_masked_discrete,
)
from tetris_rl.runs.checkpoint_manager import CheckpointPaths, resolve_checkpoint_path
from tetris_rl.runs.checkpoint_poll import CheckpointPoller
from tetris_rl.runs.hud_adapter import env_info_for_renderer, from_info as hud_from_info
from tetris_rl.runs.hud_text import HudFormatter, HudSnapshot
from tetris_rl.runs.live_stats import StepWindow
from tetris_rl.runs.manual_cursor import ManualMacroCursor
from tetris_rl.runs.run_io import choose_config_path
from tetris_rl.runs.speed_control import RateMeter, SpeedControl
from tetris_rl.training.model_io import load_model_from_spec, warn_if_maskable_with_multidiscrete
from tetris_rl.utils.paths import repo_root, resolve_run_dir
from tetris_rl.config.resolve import resolve_config

from tetris_rl.utils.config_merge import merge_env_for_eval  # type: ignore[import-not-found]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Watch a trained PPO agent play RL-Tetris (pygame).")
    ap.add_argument("--run", type=str, required=True)
    ap.add_argument(
        "--which",
        type=str,
        default="latest",
        choices=["latest", "best", "reward", "score", "lines", "level", "survival", "final"],
    )
    ap.add_argument("--device", type=str, default="auto")

    # --- runtime / UI ---
    ap.add_argument("--reload", type=float, default=3.0, help="poll for newer checkpoint every N seconds (0 disables)")
    ap.add_argument("--fps", type=int, default=60, help="render FPS cap (UI loop)")
    ap.add_argument(
        "--step-ms",
        type=int,
        default=120,
        help="simulation stepping: >0 ms between steps | 0 => 1 step per frame | <0 => uncapped",
    )
    ap.add_argument("--cell", type=int, default=26)
    ap.add_argument("--show-grid", action="store_true")
    ap.add_argument("--no-repeat", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window-steps", type=int, default=500, help="step-window size (0 disables)")

    # --- env selection ---
    ap.add_argument(
        "--env",
        type=str,
        default="eval",
        choices=["eval", "train"],
        help="Which env config to use: eval applies cfg.train.eval.env_override; train uses cfg.env as-is.",
    )

    # --- action sources (agent) ---
    ap.add_argument("--random-action", action="store_true", help="use random actions instead of PPO policy")

    # Rust expert / heuristic policy (PyO3)
    ap.add_argument("--heuristic-agent", action="store_true", help="use Rust expert policy (no PPO)")
    ap.add_argument(
        "--heuristic-policy",
        type=str,
        default="auto",
        choices=["auto", "codemy0", "codemy1", "codemy2", "codemy2fast"],
        help="Rust expert policy to use. auto maps lookahead=0->codemy0, lookahead=1->codemy1.",
    )
    ap.add_argument("--heuristic-lookahead", type=int, default=1, choices=[0, 1])
    ap.add_argument("--heuristic-beam-width", type=int, default=10)
    ap.add_argument("--heuristic-beam-from-depth", type=int, default=1)
    ap.add_argument("--heuristic-tail-weight", type=float, default=0.5)

    ap.add_argument(
        "--piece-rule",
        type=str,
        default=None,
        choices=["uniform", "bag7"],
        help="Override cfg.game.piece_rule for watch (Rust engine).",
    )

    return ap.parse_args()


def _build_watch_cfg(*, cfg: dict[str, Any], train_spec: Any, which: str) -> dict[str, Any]:
    w = str(which).strip().lower()
    if w == "train":
        return cfg
    cfg_watch: dict[str, Any] = dict(cfg)
    override = getattr(getattr(train_spec, "eval", None), "env_override", {}) or {}
    if not isinstance(override, dict):
        override = {}
    return merge_env_for_eval(cfg=cfg_watch, env_override=override)


def _engine_board_h(env: Any, fallback: int) -> int:
    try:
        g = getattr(env, "game", None)
        if g is not None and hasattr(g, "visible_h"):
            return int(g.visible_h())
    except Exception:
        pass
    return int(fallback)


def _engine_board_w(env: Any, fallback: int) -> int:
    try:
        g = getattr(env, "game", None)
        if g is not None and hasattr(g, "board_w"):
            return int(g.board_w())
    except Exception:
        pass
    return int(fallback)


def _resolve_expert_policy_class(*, engine: Any) -> Any:
    from tetris_rl_engine import ExpertPolicy
    return ExpertPolicy


def _make_expert_policy(*, args: argparse.Namespace, engine: Any) -> Any:
    ExpertPolicy = _resolve_expert_policy_class(engine=engine)

    name = str(args.heuristic_policy).strip().lower()
    if name == "auto":
        name = "codemy0" if int(args.heuristic_lookahead) <= 0 else "codemy1"

    beam_w = max(1, int(args.heuristic_beam_width))
    beam_from_depth = int(args.heuristic_beam_from_depth)

    if name == "codemy0":
        return ExpertPolicy.codemy0(beam_width=int(beam_w), beam_from_depth=int(beam_from_depth))
    if name == "codemy1":
        return ExpertPolicy.codemy1(beam_width=int(beam_w), beam_from_depth=int(beam_from_depth))
    if name == "codemy2":
        return ExpertPolicy.codemy2(beam_width=int(beam_w), beam_from_depth=int(beam_from_depth))
    if name == "codemy2fast":
        return ExpertPolicy.codemy2fast(tail_weight=float(args.heuristic_tail_weight))

    raise RuntimeError(f"unknown heuristic policy: {name}")


def _choose_action(
        *,
        args: argparse.Namespace,
        algo_type: str,
        model: Any,
        obs: Any,
        env: Any,
        game: Any,
        expert_policy: Any,
) -> Any:
    action_mode = str(getattr(env, "action_mode", "discrete")).strip().lower()

    if bool(args.heuristic_agent):
        if expert_policy is None:
            raise RuntimeError("--heuristic-agent set but expert_policy is None")
        aid = expert_policy.action_id(game)
        if aid is None:
            aid = 0
        if action_mode == "discrete":
            return int(aid)
        rot_u, col_u = game.decode_action_id(int(aid))
        return (int(rot_u), int(col_u))

    if bool(args.random_action):
        if action_mode == "discrete":
            return int(sample_masked_discrete(env))
        return as_action_pair(env.action_space.sample())

    if model is None:
        raise RuntimeError("model is not loaded")

    pred = predict_action(algo_type=str(algo_type), model=model, obs=obs, env=env)
    if action_mode == "discrete":
        return as_action_scalar(pred)
    return as_action_pair(pred)


def main() -> int:
    args = parse_args()

    repo = repo_root()
    run_dir = resolve_run_dir(repo, str(args.run))

    cfg_path = choose_config_path(run_dir)
    cfg = load_yaml(cfg_path)
    cfg = resolve_config(cfg=cfg, cfg_path=cfg_path)
    train_spec = parse_train_spec(cfg=cfg)

    cfg_watch = _build_watch_cfg(cfg=cfg, train_spec=train_spec, which=str(args.env))

    if args.piece_rule is not None:
        cfg_watch = dict(cfg_watch)
        game_cfg = cfg_watch.get("game", {}) or {}
        if not isinstance(game_cfg, dict):
            game_cfg = {}
        game_cfg = dict(game_cfg)
        game_cfg["piece_rule"] = str(args.piece_rule).strip().lower()
        cfg_watch["game"] = game_cfg

    built = make_env_from_cfg(cfg=cfg_watch, seed=int(args.seed))
    env = built.env

    game = getattr(env, "game", None)
    if game is None:
        raise RuntimeError("env must expose .game (rust engine wrapper) for watch UI")

    algo_type = str(train_spec.rl.algo.type).strip().lower()

    expert_policy: Optional[Any] = None
    if bool(args.heuristic_agent):
        expert_policy = _make_expert_policy(args=args, engine=game)

    ckpt_dir = run_dir / "checkpoints"
    paths = CheckpointPaths(checkpoint_dir=ckpt_dir)

    which = str(args.which).strip().lower()
    if which == "final":
        ckpt = ckpt_dir / "final.zip"
    else:
        ckpt = resolve_checkpoint_path(paths, which)
    if not ckpt.is_file() and which in {"best", "reward"} and paths.latest.is_file():
        ckpt = paths.latest

    model = None
    if (not bool(args.heuristic_agent)) and (not bool(args.random_action)):
        loaded = load_model_from_spec(train_spec=train_spec, ckpt=ckpt, device=str(args.device))
        model = loaded.model
        algo_type = loaded.algo_type
        ckpt = loaded.ckpt
        if algo_type == "maskable_ppo":
            warn_if_maskable_with_multidiscrete(train_spec=train_spec, env=env)

    print(f"[watch] run_dir={run_dir}")
    print(f"[watch] cfg={cfg_path.name}")
    print(f"[watch] env={str(args.env).strip().lower()}")
    print(f"[watch] algo.type={algo_type}")
    if ckpt.is_file():
        print(f"[watch] loaded ckpt={ckpt.name} (mtime={int(ckpt.stat().st_mtime)})")
    else:
        print(f"[watch] loaded ckpt={ckpt.name} (missing on disk)")

    agent_name = "rust_expert" if bool(args.heuristic_agent) else ("random" if bool(args.random_action) else algo_type)
    if bool(args.heuristic_agent):
        agent_name = f"{agent_name}({str(args.heuristic_policy).strip().lower()})"
    print(f"[watch] agent={agent_name}")
    print("[watch] controls: P pause | N step | R reset | ESC quit | [ slower | ] faster | = max-speed")

    poller = CheckpointPoller(
        run_dir=run_dir,
        which=str(args.which),
        train_spec=train_spec,
        device=str(args.device),
        reload_every_s=float(args.reload),
    )
    if model is not None:
        poller.set_current(ckpt=ckpt, model=model, algo_type=str(algo_type))

    window = StepWindow(capacity=max(0, int(args.window_steps)))
    hud = HudFormatter(window_steps=int(args.window_steps))

    import pygame
    from tetris_rl.game.rendering.pygame.renderer import TetrisRenderer

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
            ckpt_name=str(getattr(ckpt, "stem", str(ckpt))),
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
            win_illegal_pct=0.0,
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

    # cap sim steps per frame so input stays responsive in uncapped mode
    MAX_STEPS_PER_FRAME = 500

    def _maybe_reload() -> None:
        nonlocal model, ckpt, algo_type, last_reload_at_s
        if model is None:
            return
        now_s = time.time()
        maybe = poller.maybe_reload(now_s=now_s)
        if maybe is None:
            return
        model, ckpt, algo_type = maybe
        last_reload_at_s = now_s

    def _advance_one() -> None:
        nonlocal obs, info, state

        a = _choose_action(
            args=args,
            algo_type=str(algo_type),
            model=model,
            obs=obs,
            env=env,
            game=game,
            expert_policy=expert_policy,
        )

        obs2, r, terminated, truncated, info2 = env.step(a)
        obs = obs2
        info = info2
        state = game.snapshot(include_grid=True, visible=False)

        h2 = hud_from_info(info2)
        window.push(
            step_reward=float(r),
            cleared_lines=int(h2.cleared_lines),
            illegal=int(h2.invalid_action),
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
        illegal_pct = 100.0 * float(ws.avg_illegal)

        now_s = time.time()
        last_reload_age_s = float("inf") if last_reload_at_s is None else max(0.0, now_s - float(last_reload_at_s))

        snap = HudSnapshot(
            run_name=str(args.run),
            mode=str(h.action_mode),
            ckpt_name=str(getattr(ckpt, "stem", str(ckpt))),
            paused=bool(paused),
            seed=int(args.seed),
            reload_every_s=float(args.reload),
            reloads=int(getattr(poller, "reload_count", 0)),
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
            win_illegal_pct=float(illegal_pct),
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
                    continue

                if event.key == pygame.K_RIGHTBRACKET:  # ]
                    speed.handle_faster()
                    speed.clamp()
                    continue

                if event.key == pygame.K_EQUALS:  # =
                    speed.handle_max()
                    speed.clamp()
                    continue

                if event.key == pygame.K_p:
                    paused = not paused
                    continue

                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    window.reset_episode()
                    state = game.snapshot(include_grid=True, visible=False)
                    acc_s = 0.0
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


if __name__ == "__main__":
    raise SystemExit(main())
