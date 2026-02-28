# src/tetris_rl/core/datagen/io/writer.py
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, cast

import numpy as np

from tetris_rl.core.datagen.io.schema import DatasetManifest, ShardInfo

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _json_default(o: Any) -> Any:
    if is_dataclass(o) and not isinstance(o, type):
        return asdict(o)
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    _ensure_parent_dir(path)
    tmp_path: Optional[Path] = None
    try:
        fd, tmp_name = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=path.name + ".",
            suffix=".tmp",
        )
        tmp_path = Path(tmp_name)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(str(tmp_path), str(path))
        tmp_path = None
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except Exception:
                pass


# -----------------------------------------------------------------------------
# manifest (BC-minimal)
# -----------------------------------------------------------------------------


def init_manifest(
    *,
    name: str,
    board_h: int,
    board_w: int,
    num_kinds: int,
    action_dim: int,
    compression: bool,
) -> DatasetManifest:
    return DatasetManifest(
        name=str(name),
        created_utc=_utc_now_iso(),
        board_h=int(board_h),
        board_w=int(board_w),
        num_kinds=int(num_kinds),
        action_dim=int(action_dim),
        compression=bool(compression),
        keys_required=["grid", "active_kind", "next_kind", "action"],
        dtypes={
            "grid": "uint8",
            "active_kind": "uint8",
            "next_kind": "uint8",
            "action": "uint8",
        },
        shards=[],
    )


def write_manifest(*, dataset_dir: Path, manifest: DatasetManifest, overwrite: bool = False) -> Path:
    """
    Write manifest.json atomically.

    Safety: by default, refuses to overwrite an existing manifest.json.
    Callers that intentionally update the manifest must pass overwrite=True.
    """
    path = Path(dataset_dir) / "manifest.json"
    if path.exists() and not bool(overwrite):
        raise FileExistsError(f"refusing to overwrite existing manifest.json: {path}")

    payload = json.dumps(
        manifest,
        default=_json_default,
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")
    _atomic_write_bytes(path, payload)
    return path


def read_manifest(*, dataset_dir: Path) -> DatasetManifest:
    path = Path(dataset_dir) / "manifest.json"
    with open(path, "rb") as f:
        raw = json.loads(f.read().decode("utf-8"))
    if not isinstance(raw, dict):
        raise FileNotFoundError(f"missing or invalid manifest.json: {path}")
    raw_dict = cast(dict[str, Any], raw)
    return DatasetManifest(**raw_dict)


def _coerce_int(value: Any) -> int:
    return int(value)


def _to_shard_entry_dict(obj: Any) -> Optional[Dict[str, Any]]:
    if isinstance(obj, dict):
        sid_raw = obj.get("shard_id")
        file_raw = obj.get("file")
        num_samples_raw = obj.get("num_samples")
        seed_raw = obj.get("seed")
        episode_max_steps = obj.get("episode_max_steps", None)
    else:
        sid_raw = getattr(obj, "shard_id", None)
        file_raw = getattr(obj, "file", None)
        num_samples_raw = getattr(obj, "num_samples", None)
        seed_raw = getattr(obj, "seed", None)
        episode_max_steps = getattr(obj, "episode_max_steps", None)

    try:
        sid = _coerce_int(sid_raw)
        file_rel = str(file_raw)
        num_samples = _coerce_int(num_samples_raw)
        seed = _coerce_int(seed_raw)
    except Exception:
        return None

    out: Dict[str, Any] = {
        "shard_id": int(sid),
        "file": str(file_rel),
        "num_samples": int(num_samples),
        "seed": int(seed),
    }

    if episode_max_steps is not None:
        try:
            out["episode_max_steps"] = int(episode_max_steps)
        except Exception:
            pass

    return out


def merge_shards_into_manifest(*, dataset_dir: Path, shards: list[ShardInfo | Dict[str, Any]]) -> Path:
    ds = Path(dataset_dir)
    manifest = read_manifest(dataset_dir=ds)

    by_sid: Dict[int, Dict[str, Any]] = {}

    for s in list(getattr(manifest, "shards", []) or []):
        entry = _to_shard_entry_dict(s)
        if entry is None:
            continue
        by_sid[int(entry["shard_id"])] = entry

    for s in shards:
        entry = _to_shard_entry_dict(s)
        if entry is None:
            raise ValueError(f"invalid shard entry: {s!r}")
        by_sid[int(entry["shard_id"])] = entry

    merged = [by_sid[sid] for sid in sorted(by_sid.keys())]
    m_dict = asdict(manifest)
    m_dict["shards"] = merged
    updated = DatasetManifest(**m_dict)
    return write_manifest(dataset_dir=ds, manifest=updated, overwrite=True)


__all__ = [
    "init_manifest",
    "write_manifest",
    "read_manifest",
    "merge_shards_into_manifest",
]
