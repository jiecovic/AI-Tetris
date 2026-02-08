# src/tetris_rl/core/datagen/io/shard_reader.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple

import numpy as np

from tetris_rl.core.datagen.io.schema import (
    REQUIRED_KEYS,
    DatasetManifest,
    validate_shard_arrays,
)
from tetris_rl.core.utils.file_io import read_json


@dataclass(frozen=True)
class ShardRef:
    shard_id: int
    path: Path
    num_samples: int
    seed: int


def _get_field(obj: Any, key: str) -> Any:
    # manifest.shards entries may be dicts (json) or typed objects
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _req_int(obj: Any, key: str, *, where: str) -> int:
    v = _get_field(obj, key)
    if v is None:
        raise ValueError(f"missing {where}.{key}")
    try:
        return int(v)
    except Exception as e:
        raise ValueError(f"invalid int for {where}.{key}: {v!r}") from e


def _req_str(obj: Any, key: str, *, where: str) -> str:
    v = _get_field(obj, key)
    if v is None:
        raise ValueError(f"missing {where}.{key}")
    return str(v)


def _raise_missing_manifest(dataset_dir: Path, manifest_path: Path) -> None:
    # Common footgun: passing a config yaml path as dataset_dir
    if str(dataset_dir).lower().endswith((".yaml", ".yml")):
        raise FileNotFoundError(
            "dataset_dir points to a YAML file; expected a dataset folder.\n"
            f"  got: {dataset_dir}\n"
            "  expected: <repo>/datasets/bc/<dataset_name>\n"
            f"  missing: {manifest_path}"
        )
    raise FileNotFoundError(f"missing or invalid manifest.json: {manifest_path}")


class ShardDataset:
    """
    BC-only schema reader for datagen datasets.

    Responsibilities:
      - load manifest.json
      - enumerate shards
      - load shard NPZ files
      - validate arrays (BC-only)
      - return raw numpy arrays exactly as stored (after dtype casts)

    Non-responsibilities:
      - batching/shuffling
      - torch/tensors
      - BC loss/masking logic
    """

    def __init__(self, *, dataset_dir: Path) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.shards_dir = self.dataset_dir / "shards"
        self.manifest_path = self.dataset_dir / "manifest.json"

        manifest_raw = read_json(self.manifest_path)
        if not isinstance(manifest_raw, dict):
            _raise_missing_manifest(self.dataset_dir, self.manifest_path)

        self.manifest = DatasetManifest(**manifest_raw)

        # Basic sanity
        if int(self.manifest.board_h) <= 0 or int(self.manifest.board_w) <= 0:
            raise ValueError("invalid board dimensions in manifest")
        if int(self.manifest.num_kinds) <= 0:
            raise ValueError("invalid num_kinds in manifest")
        if int(self.manifest.action_dim) <= 0:
            raise ValueError("invalid action_dim in manifest")

        # Parse shard list (dict-safe + attr-safe)
        shards_any = list(getattr(self.manifest, "shards", []) or [])
        shards: list[ShardRef] = []
        for i, s in enumerate(shards_any):
            where = f"manifest.shards[{i}]"
            shard_id = _req_int(s, "shard_id", where=where)
            file_rel = _req_str(s, "file", where=where)
            num_samples = _req_int(s, "num_samples", where=where)
            seed = _req_int(s, "seed", where=where)

            p = (self.dataset_dir / file_rel).resolve()
            shards.append(
                ShardRef(
                    shard_id=int(shard_id),
                    path=p,
                    num_samples=int(num_samples),
                    seed=int(seed),
                )
            )

        # allow empty shards list (partially generated datasets)
        shards.sort(key=lambda r: int(r.shard_id))
        self.shards: Tuple[ShardRef, ...] = tuple(shards)

    def shard_ids(self) -> Tuple[int, ...]:
        return tuple(s.shard_id for s in self.shards)

    def get_shard(self, shard_id: int) -> Dict[str, np.ndarray]:
        ref = self._find_shard(shard_id)
        return self._load_and_validate(ref)

    def iter_shards(
        self,
        *,
        shard_ids: Optional[Iterable[int]] = None,
    ) -> Iterator[Tuple[int, Dict[str, np.ndarray]]]:
        wanted = None
        if shard_ids is not None:
            wanted = {int(s) for s in shard_ids}

        for ref in self.shards:
            if wanted is not None and int(ref.shard_id) not in wanted:
                continue
            yield int(ref.shard_id), self._load_and_validate(ref)

    def _find_shard(self, shard_id: int) -> ShardRef:
        for r in self.shards:
            if int(r.shard_id) == int(shard_id):
                return r
        raise KeyError(f"shard_id not found: {shard_id}")

    def _load_and_validate(self, ref: ShardRef) -> Dict[str, np.ndarray]:
        if not ref.path.is_file():
            raise FileNotFoundError(f"missing shard file: {ref.path}")

        with np.load(str(ref.path), allow_pickle=False) as z:
            arrays: Dict[str, np.ndarray] = {}

            # Required BC keys only
            for k in REQUIRED_KEYS:
                if k not in z:
                    raise RuntimeError(f"shard {ref.path} missing required key {k!r}")
                arrays[k] = np.asarray(z[k])

        # Enforce BC dtypes (compact)
        arrays["grid"] = np.asarray(arrays["grid"], dtype=np.uint8)
        arrays["active_kind"] = np.asarray(arrays["active_kind"], dtype=np.uint8).reshape(-1)
        arrays["next_kind"] = np.asarray(arrays["next_kind"], dtype=np.uint8).reshape(-1)
        arrays["action"] = np.asarray(arrays["action"], dtype=np.uint8).reshape(-1)

        validate_shard_arrays(
            grid=arrays["grid"],
            active_kind=arrays["active_kind"],
            next_kind=arrays["next_kind"],
            action=arrays["action"],
            board_h=int(self.manifest.board_h),
            board_w=int(self.manifest.board_w),
            num_kinds=int(self.manifest.num_kinds),
            action_dim=int(self.manifest.action_dim),
        )

        return arrays


__all__ = [
    "ShardRef",
    "ShardDataset",
]
