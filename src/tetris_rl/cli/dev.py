# src/tetris_rl/cli/dev.py
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]


def _run(cmd: Sequence[str]) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=REPO_ROOT, check=False).returncode


def _run_all(commands: Sequence[Sequence[str]]) -> int:
    for cmd in commands:
        code = _run(cmd)
        if code != 0:
            return code
    return 0


def _py_module(module: str, *args: str) -> list[str]:
    return [sys.executable, "-m", module, *args]


def _cmd_check() -> int:
    return _run_all(
        [
            _py_module("ruff", "check", "."),
            _py_module("ruff", "format", "--check", "."),
            _py_module("pyright"),
        ]
    )


def _cmd_fix() -> int:
    return _run_all(
        [
            _py_module("ruff", "check", "--fix", "."),
            _py_module("ruff", "format", "."),
        ]
    )


def _cmd_hooks(install: bool) -> int:
    if install:
        code = _run(_py_module("pre_commit", "install"))
        if code != 0:
            return code
    return _run(_py_module("pre_commit", "run", "--all-files"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RL-Tetris dev tooling commands.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="Run lint, format check, and pyright.")
    sub.add_parser("fix", help="Auto-fix lint and format issues.")

    hooks = sub.add_parser("hooks", help="Run pre-commit hooks for all files.")
    hooks.add_argument("--install", action="store_true", help="Install git pre-commit hook before running.")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "check":
        return _cmd_check()
    if args.command == "fix":
        return _cmd_fix()
    if args.command == "hooks":
        return _cmd_hooks(install=args.install)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
