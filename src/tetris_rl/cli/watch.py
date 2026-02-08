# src/tetris_rl/cli/watch.py
from __future__ import annotations

from tetris_rl.apps.watch.entrypoint import parse_args, run_watch


def main() -> int:
    return run_watch(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
