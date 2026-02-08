# src/tetris_rl/cli/engine_speedtest.py
from __future__ import annotations

from tetris_rl.apps.engine_speedtest.entrypoint import parse_args, run_speedtest


def main() -> int:
    return run_speedtest(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
