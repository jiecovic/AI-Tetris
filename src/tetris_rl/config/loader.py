# src/tetris_rl/config/loader.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

from tetris_rl.config.datagen_spec import DataGenSpec, parse_datagen_spec
from tetris_rl.config.game_spec import GameSpec, parse_game_spec
from tetris_rl.config.resolve import resolve_config
from tetris_rl.config.snapshot import ConfigSnapshotPaths, load_yaml, write_config_snapshot
from tetris_rl.config.train_spec import TrainSpec, parse_train_spec
from tetris_rl.config.schema_types import require_mapping_strict

ConfigDomain = Literal["train", "datagen"]


@dataclass(frozen=True)
class LoadedConfig:
    """
    Result of loading a YAML config.

    - cfg: RESOLVED mapping (after applying specs.* includes)
    - domain: which schema we parsed ("train" or "datagen")
    - spec: typed spec object (TrainSpec or DataGenSpec)
    - game: typed GameSpec parsed from cfg.game (train domain only)
    - snapshot: snapshot paths written into run_dir (raw + optionally resolved)
    """
    cfg: Dict[str, Any]
    domain: ConfigDomain
    spec: TrainSpec | DataGenSpec
    game: Optional[GameSpec] = None
    snapshot: Optional[ConfigSnapshotPaths] = None


def _infer_domain(cfg: Dict[str, Any]) -> ConfigDomain:
    """
    Clean heuristic: datagen YAML has top-level 'dataset' + 'generation' + 'expert'.
    Train YAML has top-level 'train' (and typically run/env/game/model).
    """
    has_train = isinstance(cfg.get("train", None), dict)
    has_dataset = isinstance(cfg.get("dataset", None), dict)
    has_generation = isinstance(cfg.get("generation", None), dict)
    has_expert = isinstance(cfg.get("expert", None), dict)

    if has_train and not (has_dataset or has_generation or has_expert):
        return "train"

    if has_dataset and has_generation and has_expert and not has_train:
        return "datagen"

    keys = ", ".join(sorted(list(cfg.keys())))
    raise ValueError(
        "Could not infer config domain (train vs datagen) from YAML top-level keys.\n"
        "Rules:\n"
        "  - train configs must have top-level 'train' and must NOT have 'dataset'/'generation'/'expert'\n"
        "  - datagen configs must have top-level 'dataset'+'generation'+'expert' and must NOT have 'train'\n"
        f"Top-level keys seen: [{keys}]"
    )


def parse_loaded_config(
    *,
    cfg: Dict[str, Any],
    domain: Optional[ConfigDomain] = None,
) -> Tuple[ConfigDomain, TrainSpec | DataGenSpec, Optional[GameSpec]]:
    """
    Parse a RESOLVED YAML mapping into typed specs.

    Returns:
      (domain, spec, game_spec)
    """
    root = require_mapping_strict(cfg, where="cfg")

    if domain is None:
        dom = _infer_domain(root)
    else:
        dom = str(domain).strip().lower()  # type: ignore[assignment]
        if dom not in ("train", "datagen"):
            raise ValueError(f"domain must be 'train' or 'datagen' (got {domain!r})")

    if dom == "train":
        train_spec = parse_train_spec(cfg=root)
        game_spec = parse_game_spec(cfg=root)
        return "train", train_spec, game_spec

    return "datagen", parse_datagen_spec(cfg=root), None


def load_config(
    *,
    path: Path,
    domain: Optional[ConfigDomain] = None,
    snapshot_dir: Optional[Path] = None,
    write_resolved: bool = False,
) -> LoadedConfig:
    """
    Load a YAML config from disk, resolve specs.* includes, parse into typed spec(s),
    and optionally snapshot.
    """
    path = Path(path)

    cfg_raw = load_yaml(path)
    cfg_resolved = resolve_config(cfg=cfg_raw, cfg_path=path)

    dom, spec, game = parse_loaded_config(cfg=cfg_resolved, domain=domain)

    snapshot: Optional[ConfigSnapshotPaths] = None
    if snapshot_dir is not None:
        resolved_cfg = dict(cfg_resolved) if bool(write_resolved) else None
        snapshot = write_config_snapshot(src_path=path, run_dir=Path(snapshot_dir), resolved_cfg=resolved_cfg)

    return LoadedConfig(cfg=cfg_resolved, domain=dom, spec=spec, game=game, snapshot=snapshot)


__all__ = [
    "ConfigDomain",
    "LoadedConfig",
    "parse_loaded_config",
    "load_config",
]
