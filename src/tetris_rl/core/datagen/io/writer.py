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


def append_shard_to_manifest(*, dataset_dir: Path, shard: ShardInfo) -> Path:
    ds = Path(dataset_dir)
    manifest = read_manifest(dataset_dir=ds)

    shards_any = list(getattr(manifest, "shards", []) or [])

    entry: Dict[str, Any] = {
        "shard_id": int(shard.shard_id),
        "file": str(shard.file),
        "num_samples": int(shard.num_samples),
        "seed": int(shard.seed),
    }

    sid = int(shard.shard_id)
    out: list[Dict[str, Any]] = []
    for s in shards_any:
        if isinstance(s, dict):
            try:
                if int(s.get("shard_id", -1)) == sid:
                    continue
            except Exception:
                pass
            out.append(s)
        else:
            try:
                if int(getattr(s, "shard_id")) == sid:
                    continue
                out.append(
                    {
                        "shard_id": int(getattr(s, "shard_id")),
                        "file": str(getattr(s, "file")),
                        "num_samples": int(getattr(s, "num_samples")),
                        "seed": int(getattr(s, "seed")),
                    }
                )
            except Exception:
                continue

    out.append(entry)
    out.sort(key=lambda d: int(d.get("shard_id", 0)))

    m_dict = asdict(manifest)
    m_dict["shards"] = out

    updated = DatasetManifest(**m_dict)
    return write_manifest(dataset_dir=ds, manifest=updated, overwrite=True)


__all__ = [
    "init_manifest",
    "write_manifest",
    "read_manifest",
    "append_shard_to_manifest",
]
