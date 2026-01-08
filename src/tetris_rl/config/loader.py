# src/tetris_rl/config/loader.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

from tetris_rl.config.datagen_spec import DataGenSpec, parse_datagen_spec
from tetris_rl.config.game_spec import GameSpec, parse_game_spec
from tetris_rl.config.snapshot import ConfigSnapshotPaths, load_yaml, write_config_snapshot
from tetris_rl.config.train_spec import TrainSpec, parse_train_spec
from tetris_rl.config.schema_types import require_mapping_strict

ConfigDomain = Literal["train", "datagen"]


@dataclass(frozen=True)
class LoadedConfig:
    """
    Result of loading a YAML config.

    - cfg: raw YAML mapping (materialized dict)
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
    Train YAML has top-level 'train' (and typically run/game/env/model).
    """
    has_train = isinstance(cfg.get("train", None), dict)
    has_dataset = isinstance(cfg.get("dataset", None), dict)
    has_generation = isinstance(cfg.get("generation", None), dict)
    has_expert = isinstance(cfg.get("expert", None), dict)

    if has_train and not (has_dataset or has_generation or has_expert):
        return "train"

    if has_dataset and has_generation and has_expert and not has_train:
        return "datagen"

    # Ambiguous / mixed is an error (north star: two domains never mixed)
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
    Parse a raw YAML mapping into typed specs.

    Returns:
      (domain, spec, game_spec)

    Notes:
      - game_spec is only parsed for train domain.
      - We parse GameSpec here so all callers get the same validation + defaults.
    """
    root = require_mapping_strict(cfg, where="cfg")

    dom: ConfigDomain
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
    Load a YAML config from disk, parse into typed spec(s), and optionally snapshot.

    - domain=None: infer train vs datagen from top-level keys (strict)
    - snapshot_dir: if provided, writes a copy of the raw yaml (and optionally resolved) into snapshot_dir
    - write_resolved: if True, also write resolved_cfg (currently just the raw mapping; future: fully resolved)
    """
    path = Path(path)
    cfg = load_yaml(path)

    dom, spec, game = parse_loaded_config(cfg=cfg, domain=domain)

    snapshot: Optional[ConfigSnapshotPaths] = None
    if snapshot_dir is not None:
        resolved_cfg = dict(cfg) if bool(write_resolved) else None
        snapshot = write_config_snapshot(src_path=path, run_dir=Path(snapshot_dir), resolved_cfg=resolved_cfg)

    return LoadedConfig(cfg=cfg, domain=dom, spec=spec, game=game, snapshot=snapshot)


__all__ = [
    "ConfigDomain",
    "LoadedConfig",
    "parse_loaded_config",
    "load_config",
]
