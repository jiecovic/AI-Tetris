# src/tetris_rl/cli/smoke_rust_engine.py
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

from tetris_rl_engine import ExpertPolicy, TetrisEngine, WarmupSpec


@dataclass
class EpStats:
    episodes_finished: int = 0
    ep_len_sum: int = 0
    ep_len_max: int = 0
    current_ep_len: int = 0

    def on_step(self) -> None:
        self.current_ep_len += 1

    def on_episode_end(self) -> None:
        self.episodes_finished += 1
        self.ep_len_sum += self.current_ep_len
        self.ep_len_max = max(self.ep_len_max, self.current_ep_len)
        self.current_ep_len = 0

    def avg_ep_len(self) -> float:
        if self.episodes_finished == 0:
            return 0.0
        return self.ep_len_sum / self.episodes_finished


def make_expert(args: argparse.Namespace) -> ExpertPolicy:
    if args.expert == "codemy0":
        return ExpertPolicy.codemy0(
            beam_width=args.beam_width, beam_from_depth=args.beam_from_depth
        )
    if args.expert == "codemy1":
        return ExpertPolicy.codemy1(
            beam_width=args.beam_width, beam_from_depth=args.beam_from_depth
        )
    if args.expert == "codemy2":
        return ExpertPolicy.codemy2(
            beam_width=args.beam_width, beam_from_depth=args.beam_from_depth
        )
    if args.expert == "codemy2fast":
        return ExpertPolicy.codemy2fast(tail_weight=args.tail_weight)
    raise ValueError(f"unknown --expert {args.expert!r}")


def build_warmup(args: argparse.Namespace) -> WarmupSpec | None:
    """
    Build WarmupSpec for Rust engine.

    Priority:
    - If --warmup-mode is 'none', returns None.
    - Otherwise uses the distribution flags.
    - --warmup-rows/--warmup-holes remain as a convenience for fixed mode.
    """
    if args.warmup_mode == "none":
        return None

    spawn_buffer = args.spawn_buffer

    # Back-compat convenience: fixed mode can be driven by --warmup-rows/--warmup-holes.
    if args.warmup_mode == "fixed":
        if args.warmup_rows <= 0:
            return None
        return WarmupSpec.fixed(
            rows=int(args.warmup_rows),
            holes=int(args.warmup_holes),
            spawn_buffer=spawn_buffer,
        )

    if args.warmup_mode == "uniform_rows":
        if args.rows_min is None or args.rows_max is None:
            raise ValueError("--warmup-mode=uniform_rows requires --rows-min and --rows-max")
        w = WarmupSpec.uniform_rows(
            min_rows=int(args.rows_min),
            max_rows=int(args.rows_max),
            holes=int(args.warmup_holes),
            spawn_buffer=spawn_buffer,
        )
        return w

    if args.warmup_mode == "poisson":
        if args.lambda_ is None or args.rows_cap is None:
            raise ValueError("--warmup-mode=poisson requires --lambda and --rows-cap")
        w = WarmupSpec.poisson(
            lambda_=float(args.lambda_),
            cap=int(args.rows_cap),
            holes=int(args.warmup_holes),
            spawn_buffer=spawn_buffer,
        )
        return w

    if args.warmup_mode == "base_plus_poisson":
        if args.rows_base is None or args.lambda_ is None or args.rows_cap is None:
            raise ValueError(
                "--warmup-mode=base_plus_poisson requires --rows-base, --lambda and --rows-cap"
            )
        w = WarmupSpec.base_plus_poisson(
            base=int(args.rows_base),
            lambda_=float(args.lambda_),
            cap=int(args.rows_cap),
            holes=int(args.warmup_holes),
            spawn_buffer=spawn_buffer,
        )
        return w

    raise ValueError(f"unknown --warmup-mode {args.warmup_mode!r}")


def maybe_apply_uniform_holes(warmup: WarmupSpec | None, args: argparse.Namespace) -> WarmupSpec | None:
    if warmup is None:
        return None
    if args.holes_min is None and args.holes_max is None:
        return warmup
    if args.holes_min is None or args.holes_max is None:
        raise ValueError("If using uniform holes, set both --holes-min and --holes-max")
    return warmup.with_uniform_holes(int(args.holes_min), int(args.holes_max))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke + speed test Rust Tetris engine (Python bindings)"
    )
    parser.add_argument("--steps", type=int, default=200_000, help="Total placement steps to execute.")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--piece-rule", type=str, default="uniform", choices=["uniform", "bag7"])

    # Warmup (Option B)
    parser.add_argument(
        "--warmup-mode",
        type=str,
        default="fixed",
        choices=["none", "fixed", "uniform_rows", "poisson", "base_plus_poisson"],
        help="Warmup distribution for garbage rows. 'fixed' uses --warmup-rows/--warmup-holes for backwards compatibility.",
    )
    parser.add_argument("--spawn-buffer", type=int, default=None, help="Override spawn buffer (rows kept empty at top).")

    # Back-compat / convenience (fixed warmup)
    parser.add_argument("--warmup-rows", type=int, default=0)
    parser.add_argument("--warmup-holes", type=int, default=1)

    # Distribution params
    parser.add_argument("--rows-min", type=int, default=None)
    parser.add_argument("--rows-max", type=int, default=None)
    parser.add_argument("--rows-base", type=int, default=None)
    parser.add_argument("--rows-cap", type=int, default=None)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=None)

    # Optional uniform holes
    parser.add_argument("--holes-min", type=int, default=None)
    parser.add_argument("--holes-max", type=int, default=None)

    # Expert choice
    parser.add_argument(
        "--expert",
        type=str,
        default="codemy1",
        choices=["codemy0", "codemy1", "codemy2", "codemy2fast"],
    )

    # Codemy0/1/2 knobs
    parser.add_argument("--beam-width", type=int, default=None)
    parser.add_argument("--beam-from-depth", type=int, default=0)

    # Codemy2fast knob
    parser.add_argument("--tail-weight", type=float, default=0.5)

    # Reporting
    parser.add_argument("--stats-every", type=int, default=50_000, help="Print progress every N steps (0 disables).")
    parser.add_argument("--verify-legal", action="store_true", help="Occasionally verify expert actions are legal.")
    parser.add_argument("--verify-every", type=int, default=10_000, help="Verification cadence in steps.")
    args = parser.parse_args()

    warmup = build_warmup(args)
    warmup = maybe_apply_uniform_holes(warmup, args)

    g = TetrisEngine(
        seed=int(args.seed),
        piece_rule=args.piece_rule,
        warmup=warmup,
    )
    expert = make_expert(args)

    steps_done = 0
    stats = EpStats()

    t0 = time.perf_counter()

    while steps_done < args.steps:
        terminated, _cleared_step, illegal, aid = g.step_expert(expert)

        stats.on_step()
        steps_done += 1

        if illegal:
            raise RuntimeError(f"Expert produced illegal action at step={steps_done}, aid={aid}")

        if args.verify_legal and (args.verify_every > 0) and (steps_done % args.verify_every == 0):
            if aid is not None:
                mask = g.action_mask()
                if int(mask[int(aid)]) != 1:
                    raise RuntimeError(f"Verify failed: aid={aid} not legal per mask at step={steps_done}")

        if terminated or g.game_over():
            stats.on_episode_end()

            # deterministic episode seed stream (matches your Rust CLI pattern)
            ep_seed = int(args.seed) + stats.episodes_finished

            # IMPORTANT: reset requires explicit seed (Rust binding raises if None)
            g.reset(
                seed=ep_seed,
                piece_rule=args.piece_rule,
                warmup=warmup,
            )

        if args.stats_every and (steps_done % args.stats_every == 0):
            elapsed = time.perf_counter() - t0
            sps = steps_done / max(elapsed, 1e-12)
            print(
                f"PROGRESS: expert={args.expert} steps_done={steps_done} "
                f"elapsed={elapsed:.3f}s steps/s={sps:.1f} "
                f"episodes_finished={stats.episodes_finished} avg_ep_len={stats.avg_ep_len():.2f} "
                f"max_ep_len={stats.ep_len_max}"
            )

    elapsed = time.perf_counter() - t0
    sps = steps_done / max(elapsed, 1e-12)

    print(
        "DONE: "
        f"expert={args.expert} piece_rule={args.piece_rule} warmup_mode={args.warmup_mode} "
        f"steps_done={steps_done} elapsed={elapsed:.3f}s steps/s={sps:.1f} "
        f"episodes_finished={stats.episodes_finished} avg_ep_len={stats.avg_ep_len():.2f} "
        f"max_ep_len={stats.ep_len_max}"
    )


if __name__ == "__main__":
    main()
