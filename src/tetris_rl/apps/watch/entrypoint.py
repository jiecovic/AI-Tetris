# src/tetris_rl/apps/watch/entrypoint.py
from __future__ import annotations

import argparse

from tetris_rl.apps.watch.ui import run_watch_loop
from tetris_rl.core.runtime.run_context import build_run_context
from tetris_rl.core.utils.logging import setup_logger


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Watch a trained PPO agent play RL-Tetris (pygame).")
    ap.add_argument("--run", type=str, required=True)
    ap.add_argument(
        "--which",
        type=str,
        default="latest",
        choices=["latest", "best", "reward", "lines", "survival", "final"],
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
        help="Which env config to use: env_train or env_eval from the experiment config.",
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
        help="Override cfg.env.game.piece_rule for watch (Rust engine).",
    )

    return ap.parse_args()


def run_watch(args: argparse.Namespace) -> int:
    ctx = build_run_context(
        run=str(args.run),
        which=str(args.which),
        which_env=str(args.env),
        seed=int(args.seed),
        device=str(args.device),
        piece_rule=args.piece_rule,
        reload_every_s=float(args.reload),
        use_expert=bool(args.heuristic_agent),
        random_action=bool(args.random_action),
        expert_args=args,
    )

    logger = setup_logger(name="tetris_rl.apps.watch", use_rich=True, level="info")
    logger.info("[watch] run_dir=%s", str(ctx.spec.run_dir))
    logger.info("[watch] cfg=%s", str(ctx.spec.cfg_path.name))
    logger.info("[watch] env=%s", str(args.env).strip().lower())
    logger.info("[watch] algo.type=%s", str(ctx.algo_type))
    if ctx.ckpt is not None:
        if ctx.ckpt.is_file():
            logger.info("[watch] loaded ckpt=%s (mtime=%s)", str(ctx.ckpt.name), int(ctx.ckpt.stat().st_mtime))
        else:
            logger.warning("[watch] loaded ckpt=%s (missing on disk)", str(ctx.ckpt.name))
    if ctx.artifact.note:
        logger.info("[watch] note=%s", str(ctx.artifact.note))

    agent_name = "rust_expert" if bool(args.heuristic_agent) else ("random" if bool(args.random_action) else ctx.algo_type)
    if bool(args.heuristic_agent):
        agent_name = f"{agent_name}({str(args.heuristic_policy).strip().lower()})"
    if ctx.planning_policy is not None and (not bool(args.heuristic_agent)) and (not bool(args.random_action)):
        agent_name = "ga_heuristic" if ctx.algo_type == "ga" else "td_heuristic"
    logger.info("[watch] agent=%s", str(agent_name))

    return run_watch_loop(args=args, ctx=ctx)


__all__ = ["parse_args", "run_watch"]
