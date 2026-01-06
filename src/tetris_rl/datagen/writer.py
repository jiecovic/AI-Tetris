# src/tetris_rl/datagen/writer.py
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from tetris_rl.datagen.schema import (
    DatasetManifest,
    ShardInfo,
    shard_dtypes,
    validate_shard_arrays,
    FEATURE_NAMES,
)


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
# manifest
# -----------------------------------------------------------------------------

def init_manifest(
        *,
        name: str,
        board_h: int,
        board_w: int,
        num_kinds: int,
        action_dim: int,
        max_rots: int,
        pieces: str,
        piece_rule: str,
        compression: bool,
        datagen_spec: Dict[str, Any],
) -> DatasetManifest:
    """
    Initialize a dataset manifest.

    IMPORTANT:
      - Manifest describes schema + environment compatibility.
      - Reward-fit presence is inferred from shard contents, NOT declared here.
    """
    dtypes = {k: str(v) for k, v in shard_dtypes().items()}

    return DatasetManifest(
        name=str(name),
        created_utc=_utc_now_iso(),
        board_h=int(board_h),
        board_w=int(board_w),
        num_kinds=int(num_kinds),
        action_dim=int(action_dim),
        max_rots=int(max_rots),
        pieces=str(pieces),
        piece_rule=str(piece_rule),
        compression=bool(compression),
        dtypes=dict(dtypes),
        feature_names=list(FEATURE_NAMES),
        datagen_spec=dict(datagen_spec),
    )


def write_manifest(*, dataset_dir: Path, manifest: DatasetManifest) -> Path:
    path = Path(dataset_dir) / "manifest.json"
    payload = json.dumps(
        manifest,
        default=_json_default,
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")
    _atomic_write_bytes(path, payload)
    return path


def read_manifest(*, dataset_dir: Path) -> DatasetManifest:
    """
    Load manifest.json and parse it into DatasetManifest.
    """
    path = Path(dataset_dir) / "manifest.json"
    with open(path, "rb") as f:
        raw = json.loads(f.read().decode("utf-8"))
    if not isinstance(raw, dict):
        raise FileNotFoundError(f"missing or invalid manifest.json: {path}")
    return DatasetManifest(**raw)


def append_shard_to_manifest(*, dataset_dir: Path, shard: ShardInfo) -> Path:
    """
    Atomically append/replace a shard entry in manifest.json.

    Idempotent for shard_id:
      - if shard_id exists, it is replaced
      - otherwise it is appended

    This enables partial dataset consumption while datagen is still running.
    """
    ds = Path(dataset_dir)
    manifest = read_manifest(dataset_dir=ds)

    shards_any = list(getattr(manifest, "shards", []) or [])

    entry: Dict[str, Any] = {
        "shard_id": int(shard.shard_id),
        "file": str(shard.file),
        "num_samples": int(shard.num_samples),
        "seed": int(shard.seed),
        "episode_max_steps": None if shard.episode_max_steps is None else int(shard.episode_max_steps),
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
            # tolerate typed objects/dataclasses
            try:
                if int(getattr(s, "shard_id")) == sid:
                    continue
                out.append(
                    {
                        "shard_id": int(getattr(s, "shard_id")),
                        "file": str(getattr(s, "file")),
                        "num_samples": int(getattr(s, "num_samples")),
                        "seed": int(getattr(s, "seed")),
                        "episode_max_steps": getattr(s, "episode_max_steps", None),
                    }
                )
            except Exception:
                # if something is malformed, drop it rather than poisoning the manifest
                continue

    out.append(entry)
    out.sort(key=lambda d: int(d.get("shard_id", 0)))

    m_dict = asdict(manifest)
    m_dict["shards"] = out

    updated = DatasetManifest(**m_dict)
    return write_manifest(dataset_dir=ds, manifest=updated)


# -----------------------------------------------------------------------------
# shard writer
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
            episode_max_steps: Optional[int],
            seed: int,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.shard_id = int(shard_id)
        self.compression = bool(compression)
        self.board_h = int(board_h)
        self.board_w = int(board_w)
        self.num_kinds = int(num_kinds)
        self.action_dim = int(action_dim)
        self.episode_max_steps = None if episode_max_steps is None else int(episode_max_steps)
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
            placed_cells_cleared: Optional[np.ndarray] = None,
            placed_cells_all_cleared: Optional[np.ndarray] = None,
            legal_mask: Optional[np.ndarray] = None,
            phi: Optional[np.ndarray] = None,
            delta: Optional[np.ndarray] = None,
            feature_names: Optional[np.ndarray] = None,
    ) -> ShardInfo:
        # ------------------------------------------------------------
        # Enforce reward-fit invariants: ALL or NONE
        # ------------------------------------------------------------
        rewardfit = (legal_mask, phi, delta)
        any_rf = any(x is not None for x in rewardfit)
        all_rf = all(x is not None for x in rewardfit)

        if any_rf and not all_rf:
            raise ValueError(
                "ShardWriter.write(): reward-fit arrays must be ALL present or ALL absent "
                "(legal_mask, phi, delta)."
            )

        if all_rf:
            if feature_names is None:
                feature_names = np.asarray(FEATURE_NAMES, dtype=np.str_)
            else:
                fn = [str(x) for x in list(feature_names)]
                if fn != list(FEATURE_NAMES):
                    raise ValueError(f"feature_names mismatch: got={fn}, expected={list(FEATURE_NAMES)}")
                feature_names = np.asarray(fn, dtype=np.str_)

        validate_shard_arrays(
            grid=grid,
            active_kind=active_kind,
            next_kind=next_kind,
            action=action,
            board_h=self.board_h,
            board_w=self.board_w,
            num_kinds=self.num_kinds,
            placed_cells_cleared=placed_cells_cleared,
            placed_cells_all_cleared=placed_cells_all_cleared,
            legal_mask=legal_mask,
            phi=phi,
            delta=delta,
            action_dim=self.action_dim,
            feature_names=feature_names,
        )

        payload: Dict[str, Any] = {
            "grid": np.asarray(grid, dtype=np.uint8),
            "active_kind": np.asarray(active_kind, dtype=np.uint8),
            "next_kind": np.asarray(next_kind, dtype=np.uint8),
            "action": np.asarray(action, dtype=np.int64),
        }

        if placed_cells_cleared is not None:
            payload["placed_cells_cleared"] = np.asarray(placed_cells_cleared, dtype=np.uint8)
        if placed_cells_all_cleared is not None:
            payload["placed_cells_all_cleared"] = np.asarray(placed_cells_all_cleared, dtype=np.bool_)

        if all_rf:
            payload.update(
                {
                    "legal_mask": np.asarray(legal_mask, dtype=np.bool_),
                    "phi": np.asarray(phi, dtype=np.float32),
                    "delta": np.asarray(delta, dtype=np.float32),
                    "action_dim": np.asarray([self.action_dim], dtype=np.int32),
                    "feature_names": np.asarray(feature_names, dtype=np.str_),
                }
            )

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
            episode_max_steps=self.episode_max_steps,
        )
