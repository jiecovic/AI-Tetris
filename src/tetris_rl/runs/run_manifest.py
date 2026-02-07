# src/tetris_rl/runs/run_manifest.py
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from tetris_rl.config.base import ConfigBase
from tetris_rl.utils.file_io import write_json


class RunManifest(ConfigBase):
    created_at_utc: str
    run_dir: str
    config_path: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_run_manifest(*, run_dir: Path, config_path: Path) -> None:
    manifest = RunManifest(
        created_at_utc=_utc_now_iso(),
        run_dir=str(Path(run_dir).resolve()),
        config_path=str(Path(config_path).resolve()),
    )
    write_json(Path(run_dir) / "run.json", manifest.model_dump(mode="json"))


__all__ = ["RunManifest", "write_run_manifest"]
