# src/tetris_rl/config/snapshot.py
from __future__ import annotations

from dataclasses import dataclass, asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from tetris_rl.config.schema_types import require_mapping_strict


@dataclass(frozen=True)
class ConfigSnapshotPaths:
    raw: Path
    resolved: Optional[Path] = None


def load_yaml(path: Path) -> dict[str, Any]:
    """
    Load a YAML file into a dict[str, Any].

    Contract:
      - top-level MUST be a mapping
      - no schema validation here (pure I/O)
    """
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    return require_mapping_strict(obj, where=f"config({path})")


# ---------------------------------------------------------------------
# YAML safety helpers
# ---------------------------------------------------------------------


def _to_yaml_safe(obj: Any) -> Any:
    """
    Recursively convert dataclasses and containers into YAML-safe objects.

    Rules:
      - dataclass -> dict (via asdict)
      - dict -> dict (values sanitized)
      - list/tuple -> list (values sanitized)
      - everything else -> unchanged
    """
    if is_dataclass(obj):
        return _to_yaml_safe(asdict(obj))

    if isinstance(obj, dict):
        return {str(k): _to_yaml_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_to_yaml_safe(v) for v in obj]

    return obj


def write_config_snapshot(
        *,
        src_path: Path,
        run_dir: Path,
        resolved_cfg: Optional[dict[str, Any]] = None,
        raw_name: str = "config.yaml",
        resolved_name: str = "config.resolved.yaml",
) -> ConfigSnapshotPaths:
    """
    Save an exact copy of the input YAML plus an optional resolved snapshot.

    Notes:
      - raw snapshot is a byte-for-byte copy of src_path
      - resolved snapshot is YAML-dumped after converting dataclasses
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- raw snapshot (exact copy) ---
    raw_dst = run_dir / str(raw_name)
    raw_dst.write_bytes(Path(src_path).read_bytes())

    # --- resolved snapshot (YAML-safe) ---
    resolved_dst: Optional[Path] = None
    if resolved_cfg is not None:
        resolved_dst = run_dir / str(resolved_name)
        safe_cfg = _to_yaml_safe(resolved_cfg)
        with resolved_dst.open("w", encoding="utf-8") as f:
            yaml.safe_dump(safe_cfg, f, sort_keys=False)

    return ConfigSnapshotPaths(raw=raw_dst, resolved=resolved_dst)


__all__ = ["ConfigSnapshotPaths", "load_yaml", "write_config_snapshot"]
