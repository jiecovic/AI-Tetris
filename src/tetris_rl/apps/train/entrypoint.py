# src/tetris_rl/apps/train/entrypoint.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from hydra import compose, initialize, initialize_config_dir
from omegaconf import DictConfig

from tetris_rl.core.training.runners import run_experiment


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Train an RL-Tetris agent.",
        allow_abbrev=False,
    )
    ap.add_argument("-cfg", "--config-file", dest="config_file", default=None, help="path to a YAML config file")
    ap.add_argument("-c", "--config-name", dest="config_name", help="config name (no .yaml)")
    ap.add_argument("-p", "--config-path", dest="config_path", default="configs", help="config directory")
    ap.add_argument("overrides", nargs=argparse.REMAINDER, help="Hydra overrides (after --)")
    return ap.parse_args(argv)


def _resolve_config_selection(args: argparse.Namespace) -> tuple[Path, str]:
    if args.config_file:
        p = Path(str(args.config_file)).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"--config-file not found: {p}")
        # If the file lives under a "configs" dir, use that as config root so defaults work.
        for parent in p.parents:
            if parent.name == "configs":
                rel = p.relative_to(parent).with_suffix("")
                return parent, rel.as_posix()
        return p.parent, p.stem
    base = Path(str(args.config_path)).expanduser()
    if not base.is_absolute():
        base = (Path.cwd() / base).resolve()
    return base, str(args.config_name)


def _normalize_overrides(overrides: Sequence[str]) -> list[str]:
    if not overrides:
        return []
    if overrides[0] == "--":
        return list(overrides[1:])
    return list(overrides)


def run_train(args: argparse.Namespace) -> int:
    cfg_path, cfg_name = _resolve_config_selection(args)
    overrides = _normalize_overrides(args.overrides)

    if cfg_path.is_absolute():
        with initialize_config_dir(version_base=None, config_dir=str(cfg_path)):
            cfg: DictConfig = compose(config_name=str(cfg_name), overrides=overrides)
    else:
        with initialize(version_base=None, config_path=str(cfg_path)):
            cfg = compose(config_name=str(cfg_name), overrides=overrides)

    return run_experiment(cfg)


__all__ = ["parse_args", "run_train"]
