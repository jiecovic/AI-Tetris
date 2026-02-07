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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Benchmark a trained PPO (or heuristic/random) agent on RL-Tetris without rendering."
    )
    ap.add_argument("--run", type=str, required=True)
    ap.add_argument(
        "--which",
        type=str,
        default="latest",
        choices=["latest", "best", "reward", "score", "lines", "level", "survival", "final"],
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
    ap.add_argument("--steps", type=int, default=200_000, help="total env steps to run (across many episodes)")
    ap.add_argument("--max-episodes", type=int, default=0, help="stop after N finished episodes (0 disables)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--print-every", type=int, default=20_000, help="print interim stats every N steps (0 disables)")
    ap.add_argument("--json", action="store_true", help="print final stats as JSON only")

    # --- progress bar ---
    ap.add_argument("--progress", action="store_true", help="show progress bar")
    ap.add_argument("--no-progress", action="store_true", help="disable progress bar (overrides --progress)")

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
    ga_policy = ctx.ga_policy
    expert_policy = ctx.expert_policy
    algo_type = str(ctx.algo_type)
    ckpt = ctx.ckpt

    agent_name = "rust_expert" if bool(args.heuristic_agent) else ("random" if bool(args.random_action) else algo_type)
    if bool(args.heuristic_agent):
        agent_name = f"{agent_name}({str(args.heuristic_policy).strip().lower()})"
    if ga_policy is not None and (not bool(args.heuristic_agent)) and (not bool(args.random_action)):
        agent_name = "ga_heuristic"

    if not bool(args.json):
        print(f"[bench] run_dir={spec.run_dir}")
        print(f"[bench] cfg={spec.cfg_path.name}")
        print(f"[bench] env={str(args.env).strip().lower()}")
        print(f"[bench] algo.type={algo_type}")
        if ckpt.is_file():
            print(f"[bench] loaded ckpt={ckpt.name} (mtime={int(ckpt.stat().st_mtime)})")
        else:
            print(f"[bench] loaded ckpt={ckpt.name} (missing on disk)")
        print(f"[bench] agent={agent_name}")

    poller = ctx.poller

    totals = BenchTotals()

    obs, info = env.reset(seed=int(args.seed))
    totals.start_episode()

    step_budget = max(0, int(args.steps))
    max_eps_done = max(0, int(args.max_episodes))
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

    use_bar = bool(args.progress) and (not bool(args.no_progress)) and (not bool(args.json)) and step_budget > 0
    pbar = tqdm(total=step_budget, unit="step", dynamic_ncols=True) if use_bar else None

    # Throttle these to keep benchmark fast.
    POSTFIX_EVERY = 2000

    try:
        while totals.steps < step_budget:
            if max_eps_done > 0 and totals.episodes_done >= max_eps_done:
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
                ga_policy=ga_policy,
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
                print(
                    f"[bench] interim steps={d['steps']} eps_done={d['episodes_done']} "
                    f"r/step={d['avg_reward_per_step']:.4f} "
                    f"lines/step={d['avg_lines_per_step']:.4f} "
                    f"score/step={d['avg_score_per_step']:.2f} "
                    f"avg_ep_len(done)={d['avg_episode_len_done_only']:.1f} "
                    f"avg_ep_len(+partial)={d['avg_episode_len_including_partial']:.1f}"
                )

            if bool(terminated or truncated):
                totals.finish_episode()
                obs, info = env.reset()
                totals.start_episode()

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
    }

    out = {**meta, **stats}

    if bool(args.json):
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print(_format_report(meta=meta, stats=stats))
        # Still print JSON as a single line for copy/paste and quick diffing:
        print("[bench] json:", json.dumps(out, sort_keys=True))

    return 0


__all__ = ["parse_args", "run_benchmark"]
