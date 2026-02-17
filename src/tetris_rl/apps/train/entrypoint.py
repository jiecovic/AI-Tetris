# src/tetris_rl/apps/train/entrypoint.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from hydra import compose, initialize, initialize_config_dir
from omegaconf import DictConfig

from tetris_rl.core.training.runners import run_experiment


def _looks_like_path_or_yaml(arg: str) -> bool:
    s = str(arg).strip().strip('"').strip("'")
    if not s:
        return False
    if s.endswith((".yaml", ".yml")):
        return True
    # Windows/relative path hints. Note: config names may include "/" (e.g. "ppo/ppo_cnn"),
    # so we do NOT treat a plain "/" as a filesystem path hint.
    if s.startswith((".", "/", "\\")):
        return True
    if "\\" in s:
        return True
    # Drive letter path: C:\...
    if len(s) >= 3 and s[1:3] == ":\\":
        return True
    return False


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Train an RL-Tetris agent.",
        allow_abbrev=False,
    )
    ap.add_argument("-cfg", "--config-file", dest="config_file", default=None, help="path to a YAML config file")
    ap.add_argument("-c", "--config-name", dest="config_name", help="config name (no .yaml)")
    ap.add_argument("-p", "--config-path", dest="config_path", default="configs", help="config directory")

    # Convenience: allow passing a config file as the first positional argument, e.g.
    #   tetris-train .\\configs\\ppo\\ppo_cnn.yaml
    # If -cfg/--config-file is provided, this is ignored.
    ap.add_argument("config_pos", nargs="?", default=None, help="optional config file path (equivalent to --config-file)")
    ap.add_argument("overrides", nargs=argparse.REMAINDER, help="Hydra overrides (after --)")
    args = ap.parse_args(argv)

    # Re-interpret the optional positional:
    # - if config is already provided via flags, treat it as an override
    # - else if it looks like a YAML/path, treat as --config-file
    # - else treat as --config-name
    if args.config_pos is not None:
        cp = str(args.config_pos)
        if args.config_file is not None or args.config_name is not None:
            args.overrides = [cp, *list(args.overrides or [])]
            args.config_pos = None
        elif cp == "--":
            args.overrides = [cp, *list(args.overrides or [])]
            args.config_pos = None
        elif _looks_like_path_or_yaml(cp):
            args.config_file = cp
            args.config_pos = None
        else:
            args.config_name = cp
            args.config_pos = None

    return args


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


def _maybe_add_conf_searchpath(cfg_path: Path, overrides: list[str]) -> list[str]:
    if any(str(o).startswith("hydra.searchpath") for o in overrides):
        return list(overrides)
    if cfg_path.name != "configs":
        return list(overrides)
    conf_dir = cfg_path.parent / "conf"
    if not conf_dir.is_dir():
        return list(overrides)
    conf_uri = f"file://{conf_dir.as_posix()}"
    return [f"hydra.searchpath=[{conf_uri}]", *overrides]


def run_train(args: argparse.Namespace) -> int:
    cfg_path, cfg_name = _resolve_config_selection(args)
    overrides = _normalize_overrides(args.overrides)
    overrides = _maybe_add_conf_searchpath(cfg_path, overrides)

    if cfg_path.is_absolute():
        with initialize_config_dir(version_base=None, config_dir=str(cfg_path)):
            cfg: DictConfig = compose(config_name=str(cfg_name), overrides=overrides)
    else:
        with initialize(version_base=None, config_path=str(cfg_path)):
            cfg = compose(config_name=str(cfg_name), overrides=overrides)

    return run_experiment(cfg)


__all__ = ["parse_args", "run_train"]
