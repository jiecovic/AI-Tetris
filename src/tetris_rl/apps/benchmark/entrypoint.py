# src/tetris_rl/apps/benchmark/entrypoint.py
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import Any

# Prefer Rich progress bar if installed; fall back gracefully.
try:
    from tqdm.rich import tqdm  # type: ignore
except Exception:  # pragma: no cover
    from tqdm.auto import tqdm  # type: ignore

from tetris_rl.core.agents.actions import choose_action
from tetris_rl.core.runtime.hud_adapter import from_info as hud_from_info
from tetris_rl.core.runtime.run_context import build_run_context
from tetris_rl.core.utils.logging import setup_logger


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Benchmark a trained PPO (or heuristic/random) agent on RL-Tetris without rendering."
    )
    ap.add_argument("--run", type=str, required=True)
    ap.add_argument(
        "--which",
        type=str,
        default="latest",
        choices=["latest", "best", "reward", "lines", "survival", "final"],
    )
    ap.add_argument("--device", type=str, default="auto")

    # --- runtime / reload ---
    ap.add_argument("--reload", type=float, default=0.0, help="poll for newer checkpoint every N seconds (0 disables)")

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
        help="Override cfg.env.game.piece_rule for benchmark (Rust engine).",
    )

    # --- benchmark controls ---
    ap.add_argument(
        "--steps",
        type=int,
        default=0,
        help="max env steps to run (0 disables; use with --episodes for a safety cap)",
    )
    ap.add_argument("--episodes", type=int, default=10, help="collect N finished episodes (0 disables)")
    ap.add_argument("--max-episodes", type=int, default=0, help="alias for --episodes (deprecated)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--print-every", type=int, default=20_000, help="print interim stats every N steps (0 disables)")
    ap.add_argument("--json", action="store_true", help="print final stats as JSON only")

    # --- progress bar ---
    ap.add_argument("--progress", action="store_true", help="force progress bar on (default)")
    ap.add_argument("--no-progress", action="store_true", help="disable progress bar")

    return ap.parse_args()


@dataclass
class BenchTotals:
    steps: int = 0

    # Episodes
    episodes_started: int = 0
    episodes_done: int = 0

    # Per-step sums
    sum_reward: float = 0.0
    sum_lines: float = 0.0
    sum_score_delta: float = 0.0

    # Episode lengths
    cur_ep_len: int = 0
    sum_ep_len_done: float = 0.0

    # Optional extras
    illegal_steps: int = 0
    masked_steps: int = 0

    def start_episode(self) -> None:
        self.episodes_started += 1
        self.cur_ep_len = 0

    def push_step(self, *, r: float, lines: int, score_delta: float, illegal: bool, masked: bool) -> None:
        self.steps += 1
        self.cur_ep_len += 1
        self.sum_reward += float(r)
        self.sum_lines += float(lines)
        self.sum_score_delta += float(score_delta)
        if bool(illegal):
            self.illegal_steps += 1
        if bool(masked):
            self.masked_steps += 1

    def finish_episode(self) -> None:
        self.episodes_done += 1
        self.sum_ep_len_done += float(self.cur_ep_len)
        self.cur_ep_len = 0

    def to_dict(self) -> dict[str, Any]:
        denom_steps = float(max(1, self.steps))

        avg_ep_len_done_only = (
            (self.sum_ep_len_done / float(self.episodes_done)) if self.episodes_done > 0 else 0.0
        )
        # Option A: include the partial/in-progress episode at the end of the benchmark
        denom_eps_started = float(max(1, self.episodes_started))
        avg_ep_len_including_partial = (self.sum_ep_len_done + float(self.cur_ep_len)) / denom_eps_started

        return {
            "steps": int(self.steps),
            "episodes_started": int(self.episodes_started),
            "episodes_done": int(self.episodes_done),
            "avg_reward_per_step": float(self.sum_reward / denom_steps),
            "avg_lines_per_step": float(self.sum_lines / denom_steps),
            "avg_score_per_step": float(self.sum_score_delta / denom_steps),
            "avg_episode_len_done_only": float(avg_ep_len_done_only),
            "avg_episode_len_including_partial": float(avg_ep_len_including_partial),
            "illegal_step_pct": float(100.0 * float(self.illegal_steps) / denom_steps),
            "masked_step_pct": float(100.0 * float(self.masked_steps) / denom_steps),
        }


def _format_report(*, meta: dict[str, Any], stats: dict[str, Any]) -> str:
    # Keep this stable/easy to scan.
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 86)
    lines.append("[bench] RESULT")
    lines.append("-" * 86)
    lines.append(
        f"run={meta['run']}   env={meta['env']}   which={meta['which']}   agent={meta['agent']}   ckpt={meta['ckpt']}"
    )
    lines.append(
        f"steps={stats['steps']}   episodes_started={stats['episodes_started']}   episodes_done={stats['episodes_done']}"
    )
    lines.append("")
    lines.append("Per-step averages")
    lines.append(f"  reward/step : {stats['avg_reward_per_step']:.6f}")
    lines.append(f"  lines/step  : {stats['avg_lines_per_step']:.6f}")
    lines.append(f"  score/step  : {stats['avg_score_per_step']:.6f}")
    lines.append("")
    lines.append("Episode length")
    lines.append(f"  avg (done-only)         : {stats['avg_episode_len_done_only']:.1f}")
    lines.append(f"  avg (including partial) : {stats['avg_episode_len_including_partial']:.1f}")
    lines.append("")
    lines.append("Quality / safety")
    lines.append(f"  illegal steps : {stats['illegal_step_pct']:.3f}%")
    lines.append(f"  masked steps  : {stats['masked_step_pct']:.3f}%")
    lines.append("=" * 86)
    return "\n".join(lines)


def _render_report_table(*, meta: dict[str, Any], stats: dict[str, Any]) -> Any | None:
    try:
        from rich import box
        from rich.table import Table
    except Exception:
        return None

    table = Table(title="[bench] RESULT", box=box.SIMPLE_HEAVY)
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    def add_row(label: str, value: Any) -> None:
        table.add_row(str(label), str(value))

    add_row("run", meta.get("run", "-"))
    add_row("env", meta.get("env", "-"))
    add_row("which", meta.get("which", "-"))
    add_row("agent", meta.get("agent", "-"))
    add_row("ckpt", meta.get("ckpt", "-"))
    add_row("seed", meta.get("seed", "-"))
    steps_cap = meta.get("steps_cap", 0)
    episodes_target = meta.get("episodes_target", 0)
    if int(steps_cap) > 0:
        add_row("steps_cap", int(steps_cap))
    if int(episodes_target) > 0:
        add_row("episodes_target", int(episodes_target))

    table.add_section()
    add_row("steps", int(stats["steps"]))
    add_row("episodes_started", int(stats["episodes_started"]))
    add_row("episodes_done", int(stats["episodes_done"]))

    table.add_section()
    add_row("reward/step", f"{stats['avg_reward_per_step']:.6f}")
    add_row("lines/step", f"{stats['avg_lines_per_step']:.6f}")
    add_row("score/step", f"{stats['avg_score_per_step']:.6f}")

    table.add_section()
    add_row("avg ep len (done-only)", f"{stats['avg_episode_len_done_only']:.1f}")
    add_row("avg ep len (+partial)", f"{stats['avg_episode_len_including_partial']:.1f}")

    table.add_section()
    add_row("illegal steps", f"{stats['illegal_step_pct']:.3f}%")
    add_row("masked steps", f"{stats['masked_step_pct']:.3f}%")

    return table


def _emit_table(*, logger, table: Any) -> None:
    if logger is None:
        from rich.console import Console  # type: ignore

        Console().print(table)
        return
    console = None
    for handler in getattr(logger, "handlers", []):
        console = getattr(handler, "console", None)
        if console is not None:
            break
    if console is None:
        from rich.console import Console  # type: ignore

        Console().print(table)
        return
    console.print(table)


class _BenchInterimTable:
    def __init__(self, *, emit) -> None:
        self._emit = emit
        self._header_emitted = False

    def _emit_header(self) -> None:
        self._emit(
            "[bench]    steps  eps_done    r/step  lines/step  score/step  ep_len(done)  ep_len(+partial)"
        )
        self._emit(
            "[bench] -------- -------- -------- ---------- ---------- ------------ ----------------"
        )
        self._header_emitted = True

    def emit_row(self, stats: dict[str, Any]) -> None:
        if not self._header_emitted:
            self._emit_header()
        self._emit(
            "[bench] "
            f"{int(stats['steps']):>8} "
            f"{int(stats['episodes_done']):>8} "
            f"{float(stats['avg_reward_per_step']):>8.4f} "
            f"{float(stats['avg_lines_per_step']):>10.4f} "
            f"{float(stats['avg_score_per_step']):>10.2f} "
            f"{float(stats['avg_episode_len_done_only']):>12.1f} "
            f"{float(stats['avg_episode_len_including_partial']):>16.1f}"
        )


def run_benchmark(args: argparse.Namespace) -> int:
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
    spec = ctx.spec
    env = ctx.env
    game = ctx.game
    model = ctx.model
    planning_policy = ctx.planning_policy
    expert_policy = ctx.expert_policy
    algo_type = str(ctx.algo_type)
    ckpt = ctx.ckpt

    agent_name = "rust_expert" if bool(args.heuristic_agent) else ("random" if bool(args.random_action) else algo_type)
    if bool(args.heuristic_agent):
        agent_name = f"{agent_name}({str(args.heuristic_policy).strip().lower()})"
    if planning_policy is not None and (not bool(args.heuristic_agent)) and (not bool(args.random_action)):
        agent_name = "ga_heuristic" if algo_type == "ga" else "td_heuristic"

    logger = None
    if not bool(args.json):
        logger = setup_logger(name="tetris_rl.apps.benchmark", use_rich=True, level="info")
        logger.info("[bench] run_dir=%s", str(spec.run_dir))
        logger.info("[bench] cfg=%s", str(spec.cfg_path.name))
        logger.info("[bench] env=%s", str(args.env).strip().lower())
        logger.info("[bench] algo.type=%s", str(algo_type))
        if ckpt.is_file():
            logger.info("[bench] loaded ckpt=%s (mtime=%s)", str(ckpt.name), int(ckpt.stat().st_mtime))
        else:
            logger.info("[bench] loaded ckpt=%s (missing on disk)", str(ckpt.name))
        logger.info("[bench] agent=%s", str(agent_name))

    poller = ctx.poller
    interim_table = None
    if not bool(args.json):
        emit = logger.info if logger is not None else print
        interim_table = _BenchInterimTable(emit=emit)

    totals = BenchTotals()

    obs, info = env.reset(seed=int(args.seed))
    totals.start_episode()

    step_budget = max(0, int(args.steps))
    episodes_target = max(0, int(args.episodes), int(args.max_episodes))
    print_every = max(0, int(args.print_every))

    last_reload_at_s: float | None = time.time()

    def _maybe_reload() -> None:
        nonlocal model, ckpt, algo_type, last_reload_at_s
        if model is None or poller is None:
            return
        if float(args.reload) <= 0.0:
            return
        now_s = time.time()
        maybe = poller.maybe_reload(now_s=now_s)
        if maybe is None:
            return
        model, ckpt, algo_type = maybe
        last_reload_at_s = now_s

    use_steps = step_budget > 0
    use_episodes = episodes_target > 0
    if not use_steps and not use_episodes:
        raise ValueError("benchmark requires --steps > 0 or --episodes > 0")

    use_bar = (not bool(args.no_progress)) and (not bool(args.json)) and (use_steps or use_episodes)
    pbar_total = step_budget if use_steps else episodes_target
    pbar_unit = "step" if use_steps else "episode"
    pbar = tqdm(total=pbar_total, unit=pbar_unit, dynamic_ncols=True) if use_bar else None

    # Throttle these to keep benchmark fast.
    POSTFIX_EVERY = 2000

    try:
        while True:
            if use_episodes and totals.episodes_done >= episodes_target:
                break
            if use_steps and totals.steps >= step_budget:
                if use_episodes and totals.cur_ep_len > 0:
                    # Finish the in-progress episode when collecting full episodes.
                    pass
                else:
                    break

            _maybe_reload()

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

            h2 = hud_from_info(info2)

            totals.push_step(
                r=float(r),
                lines=int(h2.cleared_lines),
                score_delta=float(h2.delta_score),
                illegal=bool(h2.invalid_action),
                masked=bool(h2.masked_action),
            )

            if pbar is not None:
                if use_steps:
                    pbar.update(1)
                if (totals.steps % POSTFIX_EVERY) == 0:
                    d = totals.to_dict()
                    pbar.set_postfix(
                        r_step=f"{d['avg_reward_per_step']:.3f}",
                        l_step=f"{d['avg_lines_per_step']:.3f}",
                        s_step=f"{d['avg_score_per_step']:.1f}",
                        eps_done=str(d["episodes_done"]),
                        ep_len=f"{d['avg_episode_len_including_partial']:.0f}",
                    )

            if print_every > 0 and (totals.steps % print_every) == 0 and (not bool(args.json)):
                d = totals.to_dict()
                if interim_table is not None:
                    interim_table.emit_row(d)

            if bool(terminated or truncated):
                totals.finish_episode()
                obs, info = env.reset()
                totals.start_episode()
                if pbar is not None and (not use_steps):
                    pbar.update(1)

    finally:
        if pbar is not None:
            pbar.close()
        env.close()

    stats = totals.to_dict()
    meta = {
        "run": str(args.run),
        "env": str(args.env).strip().lower(),
        "which": str(args.which).strip().lower(),
        "agent": agent_name,
        "ckpt": str(getattr(ckpt, "name", str(ckpt))),
        "seed": int(args.seed),
        "steps_cap": int(step_budget),
        "episodes_target": int(episodes_target),
    }

    out = {**meta, **stats}

    if bool(args.json):
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        report_table = _render_report_table(meta=meta, stats=stats)
        if report_table is None:
            report = _format_report(meta=meta, stats=stats)
            if logger is None:
                print(report)
            else:
                for line in report.splitlines():
                    logger.info(line)
        else:
            _emit_table(logger=logger, table=report_table)
        # Still print JSON as a single line for copy/paste and quick diffing:
        print("[bench] json:", json.dumps(out, sort_keys=True))

    return 0


__all__ = ["parse_args", "run_benchmark"]
