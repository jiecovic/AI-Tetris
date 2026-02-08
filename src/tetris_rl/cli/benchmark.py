# src/tetris_rl/cli/benchmark.py
from __future__ import annotations

from tetris_rl.apps.benchmark.entrypoint import parse_args, run_benchmark


def main() -> int:
    return run_benchmark(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
