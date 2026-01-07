# src/tetris_rl/cli/smoke_rust_engine.py
from __future__ import annotations

import argparse
import numpy as np

from tetris_rl_engine import TetrisEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test Rust Tetris engine")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    print("Creating PyGame...")
    g = TetrisEngine(seed=args.seed)

    print("Resetting...")
    g.reset()

    print("Initial state:")
    grid = g.grid()
    print("  grid shape:", grid.shape, grid.dtype)
    print("  score:", g.score())
    print("  game_over:", g.game_over())
    print("  active:", g.active_kind(), "next:", g.next_kind())

    for step in range(args.steps):
        mask = g.action_mask()
        assert mask.ndim == 1

        legal = np.nonzero(mask)[0]
        if len(legal) == 0:
            print("No legal actions!")
            break

        action = int(legal[0])  # deterministic
        terminated, cleared, illegal = g.step_action_id(action)

        print(
            f"step={step:03d} "
            f"action={action:03d} "
            f"illegal={int(illegal)} "
            f"cleared={cleared} "
            f"score={g.score()} "
            f"game_over={g.game_over()}"
        )

        # This should never happen in this smoke test because we select from the mask.
        if illegal:
            print("Illegal action reported (unexpected with mask-based selection).")

        if terminated:
            print("Terminated.")
            break

    print("Final grid sum:", int(g.grid().sum()))
    print("Smoke test OK.")


if __name__ == "__main__":
    main()
