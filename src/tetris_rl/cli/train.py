# src/tetris_rl/cli/train.py
from __future__ import annotations

from tetris_rl.apps.train.entrypoint import parse_args, run_train


def main() -> int:
    return run_train(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
