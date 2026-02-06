# src/tetris_rl/cli/benchmark.py
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import Any, Optional

# Prefer Rich progress bar if installed; fall back gracefully.
try:
    from tqdm.rich import tqdm  # type: ignore
except Exception:  # pragma: no cover
    from tqdm.auto import tqdm  # type: ignore

from tetris_rl.config.io import load_experiment_config, to_plain_dict
from tetris_rl.envs.factory import make_env_from_cfg
from tetris_rl.runs.action_source import (
    as_action_pair,
    as_action_scalar,
    predict_action,
    sample_masked_discrete,
)
from tetris_rl.runs.checkpoint_manifest import resolve_checkpoint_from_manifest
from tetris_rl.runs.checkpoint_poll import CheckpointPoller
from tetris_rl.runs.hud_adapter import from_info as hud_from_info
from tetris_rl.runs.run_io import choose_config_path
from tetris_rl.training.model_io import load_model_from_train_config, warn_if_maskable_with_multidiscrete
from tetris_rl.utils.config_merge import merge_cfg_for_eval
from tetris_rl.utils.paths import repo_root, resolve_run_dir


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
        help="Override cfg.game.piece_rule for benchmark (Rust engine).",
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


def _build_eval_cfg(*, cfg: dict[str, Any], train_cfg: Any, which_env: str) -> dict[str, Any]:
    w = str(which_env).strip().lower()
    if w == "train":
        return cfg
    cfg_eval: dict[str, Any] = dict(cfg)
    override = getattr(getattr(train_cfg, "eval", None), "env_override", {}) or {}
    if not isinstance(override, dict):
        override = {}
    return merge_cfg_for_eval(cfg=cfg_eval, env_override=override)


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


def main() -> int:
    args = parse_args()

    repo = repo_root()
    run_dir = resolve_run_dir(repo, str(args.run))

    cfg_path = choose_config_path(run_dir)
    exp_cfg = load_experiment_config(cfg_path)
    cfg = to_plain_dict(exp_cfg)
    train_cfg = exp_cfg.train

    cfg_bench = _build_eval_cfg(cfg=cfg, train_cfg=train_cfg, which_env=str(args.env))

    if args.piece_rule is not None:
        cfg_bench = dict(cfg_bench)
        game_cfg = cfg_bench.get("game", {}) or {}
        if not isinstance(game_cfg, dict):
            game_cfg = {}
        game_cfg = dict(game_cfg)
        game_cfg["piece_rule"] = str(args.piece_rule).strip().lower()
        cfg_bench["game"] = game_cfg

    built = make_env_from_cfg(cfg=cfg_bench, seed=int(args.seed))
    env = built.env

    game = getattr(env, "game", None)
    if game is None:
        raise RuntimeError("env must expose .game (rust engine wrapper) for benchmark")

    algo_type = str(train_cfg.rl.algo.type).strip().lower()

    expert_policy: Optional[Any] = None
    if bool(args.heuristic_agent):
        expert_policy = _make_expert_policy(args=args, engine=game)

    which = str(args.which).strip().lower()
    ckpt = resolve_checkpoint_from_manifest(run_dir=run_dir, which=which)

    model = None
    if (not bool(args.heuristic_agent)) and (not bool(args.random_action)):
        loaded = load_model_from_train_config(train_cfg=train_cfg, ckpt=ckpt, device=str(args.device))
        model = loaded.model
        algo_type = loaded.algo_type
        ckpt = loaded.ckpt
        if algo_type == "maskable_ppo":
            warn_if_maskable_with_multidiscrete(train_cfg=train_cfg, env=env)

    agent_name = "rust_expert" if bool(args.heuristic_agent) else ("random" if bool(args.random_action) else algo_type)
    if bool(args.heuristic_agent):
        agent_name = f"{agent_name}({str(args.heuristic_policy).strip().lower()})"

    if not bool(args.json):
        print(f"[bench] run_dir={run_dir}")
        print(f"[bench] cfg={cfg_path.name}")
        print(f"[bench] env={str(args.env).strip().lower()}")
        print(f"[bench] algo.type={algo_type}")
        if ckpt.is_file():
            print(f"[bench] loaded ckpt={ckpt.name} (mtime={int(ckpt.stat().st_mtime)})")
        else:
            print(f"[bench] loaded ckpt={ckpt.name} (missing on disk)")
        print(f"[bench] agent={agent_name}")

    poller = CheckpointPoller(
        run_dir=run_dir,
        which=str(args.which),
        train_cfg=train_cfg,
        device=str(args.device),
        reload_every_s=float(args.reload),
    )
    if model is not None and float(args.reload) > 0.0:
        poller.set_current(ckpt=ckpt, model=model, algo_type=str(algo_type))

    totals = BenchTotals()

    obs, info = env.reset(seed=int(args.seed))
    totals.start_episode()

    step_budget = max(0, int(args.steps))
    max_eps_done = max(0, int(args.max_episodes))
    print_every = max(0, int(args.print_every))

    last_reload_at_s: float | None = time.time()

    def _maybe_reload() -> None:
        nonlocal model, ckpt, algo_type, last_reload_at_s
        if model is None:
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


if __name__ == "__main__":
    raise SystemExit(main())
