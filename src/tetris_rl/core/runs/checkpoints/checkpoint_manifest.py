# src/tetris_rl/core/runs/checkpoints/checkpoint_manifest.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

from tetris_rl.core.config.base import ConfigBase
from tetris_rl.core.utils.file_io import read_json, write_json


class CheckpointEntry(ConfigBase):
    path: str
    timesteps: Optional[int] = None
    metric: Optional[str] = None
    value: Optional[float] = None


class CheckpointManifest(ConfigBase):
    latest: Optional[CheckpointEntry] = None
    best_reward: Optional[CheckpointEntry] = None
    best_lines: Optional[CheckpointEntry] = None
    best_survival: Optional[CheckpointEntry] = None
    final: Optional[CheckpointEntry] = None


def load_checkpoint_manifest(path: Path) -> CheckpointManifest:
    data = read_json(Path(path))
    if not isinstance(data, dict):
        raise TypeError(f"checkpoint manifest must be a mapping: {path}")
    data.pop("best_score", None)
    data.pop("best_level", None)
    return CheckpointManifest.model_validate(data)


def save_checkpoint_manifest(path: Path, manifest: CheckpointManifest) -> None:
    write_json(Path(path), manifest.model_dump(mode="json"))


def update_checkpoint_manifest(*, manifest_path: Path, field: str, entry: CheckpointEntry) -> None:
    try:
        cur = load_checkpoint_manifest(manifest_path)
    except Exception:
        cur = CheckpointManifest()

    updated = cur.model_copy(update={str(field): entry})
    save_checkpoint_manifest(manifest_path, updated)


def resolve_checkpoint_from_manifest(*, run_dir: Path, which: str) -> Path:
    ckpt_dir = Path(run_dir) / "checkpoints"
    manifest_path = ckpt_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"checkpoint manifest not found: {manifest_path}")

    manifest = load_checkpoint_manifest(manifest_path)
    w = str(which).strip().lower()

    if w == "latest":
        entry = manifest.latest
    elif w in {"best", "reward"}:
        entry = manifest.best_reward
    elif w == "lines":
        entry = manifest.best_lines
    elif w in {"survival", "len", "length", "time"}:
        entry = manifest.best_survival
    elif w == "final":
        entry = manifest.final
    else:
        raise ValueError(f"unknown checkpoint selector: {which!r}")

    if entry is None or not str(entry.path).strip():
        raise FileNotFoundError(f"checkpoint '{which}' not recorded in manifest: {manifest_path}")

    p = Path(entry.path)
    if not p.is_absolute():
        p = ckpt_dir / p
    return p


__all__ = [
    "CheckpointEntry",
    "CheckpointManifest",
    "load_checkpoint_manifest",
    "save_checkpoint_manifest",
    "update_checkpoint_manifest",
    "resolve_checkpoint_from_manifest",
]
