# src/tetris_rl/cli/smoke_rust_engine.py
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

from tetris_rl_engine import ExpertPolicy, TetrisEngine


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
        return ExpertPolicy.codemy0(beam_width=args.beam_width, beam_from_depth=args.beam_from_depth)
    if args.expert == "codemy1":
        return ExpertPolicy.codemy1(beam_width=args.beam_width, beam_from_depth=args.beam_from_depth)
    if args.expert == "codemy2":
        return ExpertPolicy.codemy2(beam_width=args.beam_width, beam_from_depth=args.beam_from_depth)
    if args.expert == "codemy2fast":
        return ExpertPolicy.codemy2fast(tail_weight=args.tail_weight)
    raise ValueError(f"unknown --expert {args.expert!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke + speed test Rust Tetris engine (Python bindings)")
    parser.add_argument("--steps", type=int, default=200_000, help="Total placement steps to execute.")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--piece-rule", type=str, default="uniform", choices=["uniform", "bag7"])
    parser.add_argument("--warmup-rows", type=int, default=0)
    parser.add_argument("--warmup-holes", type=int, default=1)

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

    g = TetrisEngine(
        seed=args.seed,
        piece_rule=args.piece_rule,
        warmup_rows=int(args.warmup_rows),
        warmup_holes=int(args.warmup_holes),
    )
    expert = make_expert(args)

    # Totals
    steps_done = 0
    total_lines_before = int(g.lines_cleared())
    total_score_before = int(g.score())
    stats = EpStats()

    t0 = time.perf_counter()

    # Main loop (avoid calling grid() each step for speed)
    while steps_done < args.steps:
        terminated, _cleared_step, illegal, aid = g.step_expert(expert)

        # step_expert returns aid=None when expert has no legal action (should be rare)
        # We treat this as terminated for stats purposes.
        stats.on_step()
        steps_done += 1

        if illegal:
            # This should never happen if the expert uses the engine’s legality.
            # If it does, stop loudly.
            raise RuntimeError(f"Expert produced illegal action at step={steps_done}, aid={aid}")

        if args.verify_legal and (args.verify_every > 0) and (steps_done % args.verify_every == 0):
            if aid is not None:
                mask = g.action_mask()
                if int(mask[int(aid)]) != 1:
                    raise RuntimeError(f"Verify failed: aid={aid} not legal per mask at step={steps_done}")

        if terminated or g.game_over():
            stats.on_episode_end()
            # reset with deterministic episode seed stream (matches your Rust CLI pattern)
            ep_seed = int(args.seed) + stats.episodes_finished
            g.reset(
                seed=ep_seed,
                piece_rule=args.piece_rule,
                warmup_rows=int(args.warmup_rows),
                warmup_holes=int(args.warmup_holes),
            )

        if args.stats_every and (steps_done % args.stats_every == 0):
            elapsed = time.perf_counter() - t0
            total_lines = int(g.lines_cleared()) + (stats.episodes_finished * 0)  # current engine only tracks current ep
            total_score = int(g.score()) + (stats.episodes_finished * 0)

            # Since we reset the engine each episode, the per-episode score/lines are not cumulative in-engine.
            # For live stats we can’t use g.lines_cleared()/g.score() as totals unless we accumulate ourselves.
            # So we only print SPS + episode lengths live here.
            sps = steps_done / max(elapsed, 1e-12)
            print(
                f"PROGRESS: expert={args.expert} steps_done={steps_done} "
                f"elapsed={elapsed:.3f}s steps/s={sps:.1f} "
                f"episodes_finished={stats.episodes_finished} avg_ep_len={stats.avg_ep_len():.2f} "
                f"max_ep_len={stats.ep_len_max}"
            )

    elapsed = time.perf_counter() - t0
    sps = steps_done / max(elapsed, 1e-12)

    # Final: since we reset every episode, compute totals by running deltas from the *last* episode only is meaningless.
    # If you want totals (lines/score) across episodes, we should accumulate them explicitly in this script.
    # For now, print speed + episode length stats (what you asked).
    print(
        "DONE: "
        f"expert={args.expert} piece_rule={args.piece_rule} warmup_rows={args.warmup_rows} "
        f"steps_done={steps_done} elapsed={elapsed:.3f}s steps/s={sps:.1f} "
        f"episodes_finished={stats.episodes_finished} avg_ep_len={stats.avg_ep_len():.2f} "
        f"max_ep_len={stats.ep_len_max}"
    )


if __name__ == "__main__":
    main()
