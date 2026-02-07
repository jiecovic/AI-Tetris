# src/tetris_rl/cli/datagen.py
from __future__ import annotations

from tetris_rl.apps.datagen.entrypoint import parse_args, run_datagen_job


def main() -> int:
    return run_datagen_job(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
