# src/tetris_rl/core/datagen/io/writer.py
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from tetris_rl.core.datagen.io.schema import DatasetManifest, ShardInfo, validate_shard_arrays


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _json_default(o: Any) -> Any:
    if is_dataclass(o):
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
    return DatasetManifest(**raw)


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


# -----------------------------------------------------------------------------
# shard writer (BC-only)
# -----------------------------------------------------------------------------

class ShardWriter:
    def __init__(
        self,
        *,
        dataset_dir: Path,
        shard_id: int,
        compression: bool,
        board_h: int,
        board_w: int,
        num_kinds: int,
        action_dim: int,
        seed: int,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.shard_id = int(shard_id)
        self.compression = bool(compression)
        self.board_h = int(board_h)
        self.board_w = int(board_w)
        self.num_kinds = int(num_kinds)
        self.action_dim = int(action_dim)
        self.seed = int(seed)

        (self.dataset_dir / "shards").mkdir(parents=True, exist_ok=True)

    def _rel_file(self) -> str:
        return f"shards/shard_{self.shard_id:04d}.npz"

    def _abs_file(self) -> Path:
        return self.dataset_dir / self._rel_file()

    def write(
        self,
        *,
        grid: np.ndarray,
        active_kind: np.ndarray,
        next_kind: np.ndarray,
        action: np.ndarray,
    ) -> ShardInfo:
        validate_shard_arrays(
            grid=np.asarray(grid),
            active_kind=np.asarray(active_kind),
            next_kind=np.asarray(next_kind),
            action=np.asarray(action),
            board_h=self.board_h,
            board_w=self.board_w,
            num_kinds=self.num_kinds,
            action_dim=self.action_dim,
        )

        payload: Dict[str, Any] = {
            "grid": np.asarray(grid, dtype=np.uint8),
            "active_kind": np.asarray(active_kind, dtype=np.uint8),
            "next_kind": np.asarray(next_kind, dtype=np.uint8),
            "action": np.asarray(action, dtype=np.uint8),
        }

        out_path = self._abs_file()
        _ensure_parent_dir(out_path)

        tmp_path: Optional[Path] = None
        try:
            fd, tmp_name = tempfile.mkstemp(
                dir=str(out_path.parent),
                prefix=out_path.name + ".",
                suffix=".tmp.npz",
            )
            os.close(fd)
            tmp_path = Path(tmp_name)

            if self.compression:
                np.savez_compressed(str(tmp_path), **payload)
            else:
                np.savez(str(tmp_path), **payload)

            os.replace(str(tmp_path), str(out_path))
            tmp_path = None
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

        return ShardInfo(
            shard_id=self.shard_id,
            file=self._rel_file(),
            num_samples=int(payload["grid"].shape[0]),
            seed=self.seed,
        )


__all__ = [
    "init_manifest",
    "write_manifest",
    "read_manifest",
    "append_shard_to_manifest",
    "ShardWriter",
]
