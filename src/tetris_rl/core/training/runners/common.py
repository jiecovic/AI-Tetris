# src/tetris_rl/core/training/runners/common.py
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from omegaconf import OmegaConf

from tetris_rl.core.runs.checkpoints.checkpoint_manifest import (
    CheckpointManifest,
    save_checkpoint_manifest,
)
from tetris_rl.core.runs.config import RunConfig
from tetris_rl.core.runs.run_io import RunPaths, make_run_paths, materialize_run_paths
from tetris_rl.core.runs.run_manifest import write_run_manifest


@dataclass(frozen=True)
class RunArtifacts:
    paths: RunPaths
    config_path: Path
    timing_s: float


def init_run_artifacts(
    *,
    cfg_dict: Mapping[str, Any],
    run_cfg: RunConfig,
    logger: Any,
) -> RunArtifacts:
    paths = make_run_paths(run_cfg=run_cfg)
    logger.info(f"[run] dir: {paths.run_dir}")

    t0 = time.perf_counter()
    materialize_run_paths(paths=paths)
    config_path = paths.run_dir / "config.yaml"
    OmegaConf.save(config=OmegaConf.create(dict(cfg_dict)), f=config_path)
    write_run_manifest(run_dir=paths.run_dir, config_path=config_path)
    timing_s = time.perf_counter() - t0
    logger.info(f"[timing] paths+snapshot: {timing_s:.2f}s")

    return RunArtifacts(paths=paths, config_path=config_path, timing_s=timing_s)


def ensure_checkpoint_manifest(*, ckpt_dir: Path) -> Path:
    manifest_path = Path(ckpt_dir) / "manifest.json"
    if not manifest_path.exists():
        save_checkpoint_manifest(manifest_path, CheckpointManifest())
    return manifest_path


def with_env_cfg(
    *,
    cfg: Mapping[str, Any],
    env_cfg: Mapping[str, Any],
    max_steps_per_episode: int | None = None,
) -> dict[str, Any]:
    out = dict(cfg)
    env_out = dict(env_cfg)
    if max_steps_per_episode is not None:
        params = env_out.get("params", {}) or {}
        if not isinstance(params, dict):
            params = {}
        params = dict(params)
        params["max_steps"] = int(max_steps_per_episode)
        env_out["params"] = params
    out["env"] = env_out
    return out


__all__ = [
    "RunArtifacts",
    "ensure_checkpoint_manifest",
    "init_run_artifacts",
    "with_env_cfg",
]
