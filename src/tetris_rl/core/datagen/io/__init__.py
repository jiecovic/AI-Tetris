# src/tetris_rl/core/datagen/io/__init__.py
from .schema import DatasetManifest, ShardInfo, validate_shard_arrays
from .shard_reader import ShardDataset
from .writer import append_shard_to_manifest, init_manifest, read_manifest, write_manifest

__all__ = [
    "DatasetManifest",
    "ShardInfo",
    "ShardDataset",
    "append_shard_to_manifest",
    "init_manifest",
    "read_manifest",
    "validate_shard_arrays",
    "write_manifest",
]
