# src/tetris_rl/cli/watch.py
from __future__ import annotations

import argparse
import time
from typing import Any

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
from tetris_rl.training.model_io import load_model_from_spec, warn_if_maskable_with_multidiscrete
from tetris_rl.utils.paths import repo_root, resolve_run_dir

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
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--step-ms", type=int, default=120, help="milliseconds between agent steps")
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
    ap.add_argument("--heuristic-agent", action="store_true", help="use heuristic macro-placement agent (no PPO)")
    ap.add_argument(
        "--heuristic-lookahead",
        type=int,
        default=1,
        choices=[0, 1],
        help="heuristic agent lookahead depth: 0=fast (no next piece), 1=one-piece lookahead",
    )
    ap.add_argument(
        "--heuristic-beam-width",
        type=int,
        default=10,
        help="heuristic agent beam width (only used when lookahead=1)",
    )
    return ap.parse_args()


def _choose_action(
    *,
    args: argparse.Namespace,
    algo_type: str,
    model: Any,
    obs: Any,
    env: Any,
    game: Any,
    heuristic_agent: Any,
) -> Any:
    action_mode = str(getattr(env, "action_mode", "discrete")).strip().lower()

    # --- heuristic agent (macro placement) ---
    if bool(args.heuristic_agent):
        if heuristic_agent is None:
            raise RuntimeError("--heuristic-agent set but agent not initialized")

        # NOTE: heuristic agent API may still want a "game state" object.
        # If you later migrate it to snapshots, switch this to env.last_state.
        st = game.state() if hasattr(game, "state") else game._state()  # type: ignore[attr-defined]

        rot, col = heuristic_agent.best_macro_action(st)
        if action_mode == "discrete":
            board_w = int(getattr(game, "w", 10))
            return int(rot) * int(board_w) + int(col)
        return (int(rot), int(col))

    # --- random baseline ---
    if bool(args.random_action):
        if action_mode == "discrete":
            return int(sample_masked_discrete(env))
        return as_action_pair(env.action_space.sample())

    # --- policy (SB3) ---
    if model is None:
        raise RuntimeError("model is not loaded")

    pred = predict_action(algo_type=str(algo_type), model=model, obs=obs, env=env)
    if action_mode == "discrete":
        return as_action_scalar(pred)
    return as_action_pair(pred)


def _build_watch_cfg(*, cfg: dict[str, Any], train_spec: Any, which: str) -> dict[str, Any]:
    """
    Build the config used to construct the watch env.

    - "train": uses cfg as-is.
    - "eval": patches cfg.env with cfg.train.eval.env_override (eval-only semantics).
    """
    w = str(which).strip().lower()
    if w == "train":
        return cfg

    cfg_watch: dict[str, Any] = dict(cfg)

    override = getattr(getattr(train_spec, "eval", None), "env_override", {}) or {}
    if not isinstance(override, dict):
        override = {}

    cfg_watch = merge_env_for_eval(cfg=cfg_watch, env_override=override)
    return cfg_watch


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


def main() -> int:
    args = parse_args()

    # -----------------------------------------------------------------------------
    # Resolve run + load config/spec
    # -----------------------------------------------------------------------------
    repo = repo_root()
    run_dir = resolve_run_dir(repo, str(args.run))

    cfg_path = choose_config_path(run_dir)
    cfg = load_yaml(cfg_path)
    train_spec = parse_train_spec(cfg=cfg)

    # -----------------------------------------------------------------------------
    # Build env (watch defaults to eval env semantics)
    # -----------------------------------------------------------------------------
    cfg_watch = _build_watch_cfg(cfg=cfg, train_spec=train_spec, which=str(args.env))

    built = make_env_from_cfg(cfg=cfg_watch, seed=int(args.seed))
    env = built.env

    game = getattr(env, "game", None)
    if game is None:
        raise RuntimeError("env must expose .game (rust engine wrapper) for watch UI")

    # -----------------------------------------------------------------------------
    # Select agent type
    # -----------------------------------------------------------------------------
    algo_type = str(train_spec.rl.algo.type).strip().lower()

    heuristic_agent = None
    if bool(args.heuristic_agent):
        from tetris_rl.agents.heuristic_agent import HeuristicAgent

        heuristic_agent = HeuristicAgent(
            game=game,
            lookahead=int(args.heuristic_lookahead),
            beam_width=int(args.heuristic_beam_width),
        )

    # -----------------------------------------------------------------------------
    # Resolve checkpoint + load model (unless heuristic/random)
    # -----------------------------------------------------------------------------
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

    # -----------------------------------------------------------------------------
    # Print run header
    # -----------------------------------------------------------------------------
    print(f"[watch] run_dir={run_dir}")
    print(f"[watch] cfg={cfg_path.name}")
    print(f"[watch] env={str(args.env).strip().lower()}")
    print(f"[watch] algo.type={algo_type}")
    if ckpt.is_file():
        print(f"[watch] loaded ckpt={ckpt.name} (mtime={int(ckpt.stat().st_mtime)})")
    else:
        print(f"[watch] loaded ckpt={ckpt.name} (missing on disk)")

    agent_name = "heuristic" if bool(args.heuristic_agent) else ("random" if bool(args.random_action) else algo_type)
    if bool(args.heuristic_agent):
        agent_name = f"{agent_name}(lookahead={int(args.heuristic_lookahead)})"
    print(f"[watch] agent={agent_name}")

    # -----------------------------------------------------------------------------
    # Live reload machinery
    # -----------------------------------------------------------------------------
    poller = CheckpointPoller(
        run_dir=run_dir,
        which=str(args.which),
        train_spec=train_spec,
        device=str(args.device),
        reload_every_s=float(args.reload),
    )
    if model is not None:
        poller.set_current(ckpt=ckpt, model=model, algo_type=str(algo_type))

    # -----------------------------------------------------------------------------
    # HUD + window stats
    # -----------------------------------------------------------------------------
    window = StepWindow(capacity=max(0, int(args.window_steps)))
    hud = HudFormatter(window_steps=int(args.window_steps))

    import pygame
    from tetris_rl.game.rendering.pygame.renderer import TetrisRenderer

    obs, info = env.reset(seed=int(args.seed))
    window.reset_episode()

    # IMPORTANT: for watch UI, always use engine snapshot (full grid)
    state: Any = game.snapshot(include_grid=True, visible=False)

    pygame.init()
    fps_font = pygame.font.SysFont("consolas", 16)

    renderer = TetrisRenderer(
        cell=int(args.cell),
        show_grid_lines=bool(args.show_grid),
        hud_height=0,
    )

    last_reload_at_s: float | None = time.time()

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

    screen, layout = renderer.init_window(
        board_h=_engine_board_h(env, int(getattr(game, "h", 20))),
        board_w=_engine_board_w(env, int(getattr(game, "w", 10))),
        hud_text=demo_hud_text,
        title="RL-Tetris | watch",
    )

    clock = pygame.time.Clock()

    if not bool(args.no_repeat):
        pygame.key.set_repeat(140, 35)

    # -----------------------------------------------------------------------------
    # Manual cursor (paused placement)
    # -----------------------------------------------------------------------------
    cursor = ManualMacroCursor(game=game, env=env)
    if isinstance(state, dict):
        cursor.sync_from_snapshot(state)

    running = True
    paused = False
    last_step_ms = pygame.time.get_ticks()

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
        print(f"[watch] reloaded ckpt={ckpt.name} (mtime={int(ckpt.stat().st_mtime)}) algo.type={algo_type}")

    def _advance(a: Any) -> None:
        nonlocal obs, info, last_step_ms, state

        obs2, r, terminated, truncated, info2 = env.step(a)
        obs = obs2
        info = info2

        # IMPORTANT: refresh from engine snapshot (full grid)
        state = game.snapshot(include_grid=True, visible=False)

        h = hud_from_info(info2)

        window.push(
            step_reward=float(r),
            cleared_lines=int(h.cleared_lines),
            illegal=int(h.illegal_action),
            masked=int(h.masked_action),
            redrot=int(h.redundant_rotation),
            score_delta=float(h.delta_score),
            action_id=h.action_id,
            action_dim=h.action_dim,
            episode_done=bool(terminated or truncated),
        )

        if terminated or truncated:
            obs_r, info_r = env.reset()
            obs = obs_r
            info = info_r
            window.reset_episode()

            # IMPORTANT: snapshot again after reset
            state = game.snapshot(include_grid=True, visible=False)

        last_step_ms = pygame.time.get_ticks()
        if isinstance(state, dict):
            cursor.sync_from_snapshot(state)

    # -----------------------------------------------------------------------------
    # Main pygame loop
    # -----------------------------------------------------------------------------
    while running:
        clock.tick(int(args.fps))
        now_ms = pygame.time.get_ticks()

        if (not paused) and (now_ms - last_step_ms >= int(args.step_ms)):
            _maybe_reload()
            a = _choose_action(
                args=args,
                algo_type=str(algo_type),
                model=model,
                obs=obs,
                env=env,
                game=game,
                heuristic_agent=heuristic_agent,
            )
            _advance(a)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break

                if event.key == pygame.K_p:
                    paused = not paused
                    if paused:
                        if isinstance(state, dict):
                            cursor.sync_from_snapshot(state)
                        cursor.recenter_for_pause()
                    continue

                if event.key == pygame.K_r:
                    obs, info = env.reset()
                    window.reset_episode()

                    # IMPORTANT: snapshot after reset (full grid)
                    state = game.snapshot(include_grid=True, visible=False)
                    if isinstance(state, dict):
                        cursor.sync_from_snapshot(state)

                    last_step_ms = pygame.time.get_ticks()
                    continue

                if event.key == pygame.K_n:
                    _maybe_reload()
                    a = _choose_action(
                        args=args,
                        algo_type=str(algo_type),
                        model=model,
                        obs=obs,
                        env=env,
                        game=game,
                        heuristic_agent=heuristic_agent,
                    )
                    _advance(a)
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
                        _advance(cursor.action_for_commit())

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

        # IMPORTANT: ghost only when paused
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
            engine=game,  # pass engine for UI-only masks (NEXT, active preview, etc.)
        )

        fps = float(clock.get_fps())
        fps_surf = fps_font.render(f"FPS: {fps:5.1f}", True, (180, 180, 180))
        screen.blit(
            fps_surf,
            (screen.get_width() - fps_surf.get_width() - 12, screen.get_height() - fps_surf.get_height() - 10),
        )

        pygame.display.flip()

    pygame.quit()
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
