# src/tetris_rl/runs/run_io.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tetris_rl.runs.config import RunConfig


def choose_config_path(run_dir: Path) -> Path:
    """
    Resolve the authoritative config artifact inside an existing run directory.

    Priority:
      1) config.yaml

    Note:
      - This is NOT YAML parsing and NOT schema logic.
      - It is "run artifact selection" used by watch/eval tooling.
    """
    run_dir = Path(run_dir)

    p_cfg = run_dir / "config.yaml"
    if p_cfg.is_file():
        return p_cfg

    found = sorted([p.name for p in run_dir.iterdir() if p.is_file()])
    raise FileNotFoundError(
        "Could not find config in run dir. Expected config.yaml.\n"
        f"run_dir={run_dir}\n"
        f"files={found}"
    )


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _is_nonempty_dir(p: Path) -> bool:
    try:
        if not p.exists() or not p.is_dir():
            return False
        return any(p.iterdir())
    except Exception:
        # be conservative: treat as non-empty to avoid clobbering something odd
        return True


def _pick_run_dir(out_root: Path, name: str) -> Path:
    """
    Choose a run directory without creating it.

    Policy: ALWAYS suffix.
      - first run is <out_root>/<name>_001
      - then _002, _003, ...

    Reuse rule:
      - if a candidate exists but is empty, we may reuse it
        (helps after interrupted "mkdir-only" or partially-materialized runs).
    """
    out_root = Path(out_root)
    for i in range(1, 10_000):
        cand = out_root / f"{name}_{i:03d}"

        if not cand.exists():
            return cand

        if cand.is_dir() and not _is_nonempty_dir(cand):
            return cand

    raise RuntimeError(f"could not find a free run dir under {out_root} for name={name!r}")


@dataclass(frozen=True)
class RunPaths:
    """
    Concrete filesystem layout for a training run directory.

    This is *runtime plumbing*, not config semantics.
    """
    run_dir: Path
    tb_dir: Optional[Path]
    ckpt_dir: Path
    latest_ckpt: Path


def make_run_paths(*, run_cfg: RunConfig) -> RunPaths:
    """
    Determine run output locations from RunConfig (wiring only).

    Responsibilities:
      - TrainConfig/DataGenConfig own training/datagen semantics.
      - RunConfig owns runtime wiring + filesystem/logging.
      - No other module reads cfg.run directly.
    """
    run_dir = _pick_run_dir(Path(run_cfg.out_root), str(run_cfg.name))
    ckpt_dir = run_dir / "checkpoints"
    tb_dir = (run_dir / "tb") if bool(run_cfg.tensorboard) else None
    latest_ckpt = ckpt_dir / "latest.zip"

    return RunPaths(
        run_dir=run_dir,
        tb_dir=tb_dir,
        ckpt_dir=ckpt_dir,
        latest_ckpt=latest_ckpt,
    )


def materialize_run_paths(*, paths: RunPaths) -> None:
    """
    Create directories only once everything is successfully built.
    """
    _ensure_dir(paths.run_dir)
    _ensure_dir(paths.ckpt_dir)
    if paths.tb_dir is not None:
        _ensure_dir(paths.tb_dir)


__all__ = [
    "RunPaths",
    "choose_config_path",
    "make_run_paths",
    "materialize_run_paths",
]
