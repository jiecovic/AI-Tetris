# src/tetris_rl/config/run_spec.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal

from tetris_rl.config.schema_types import (
    get_bool,
    get_int,
    get_mapping,
    get_str,
    require_mapping,
)

VecKind = Literal["subproc", "dummy"]


@dataclass(frozen=True)
class RunSpec:
    """
    RunSpec owns ALL run-time wiring + filesystem/logging semantics.

    This is intentionally NOT "training semantics" (those live in TrainSpec).
    """
    name: str
    out_root: Path
    seed: int
    device: str
    tensorboard: bool
    n_envs: int
    vec: VecKind


def _non_empty_str(s: str, *, where: str) -> str:
    v = str(s).strip()
    if not v:
        raise ValueError(f"{where} must be a non-empty string")
    return v


def _parse_vec_kind(v: str, *, where: str) -> VecKind:
    s = str(v).strip().lower()
    if s not in ("subproc", "dummy"):
        raise ValueError(f"{where} must be 'subproc' or 'dummy' (got {v!r})")
    return s  # type: ignore[return-value]


def parse_run_spec(*, cfg: Dict[str, Any]) -> RunSpec:
    """
    Canonical layout ONLY:
      run:
        name: str
        out_root: str
        seed: int
        device: str
        tensorboard: bool
        n_envs: int
        vec: "subproc" | "dummy"

    Strictness:
      - No alternate spellings
      - Types coerced via schema_types getters (project convention)
      - Validations: name/out_root non-empty, n_envs>=1, vec enum
    """
    root = require_mapping(cfg, where="cfg")
    run = get_mapping(root, "run", default={}, where="cfg.run")

    # Defaults MUST match current stable behavior (keep existing semantics).
    name = get_str(run, "name", default="run", where="cfg.run.name")
    out_root = get_str(run, "out_root", default="experiments", where="cfg.run.out_root")
    seed = get_int(run, "seed", default=0, where="cfg.run.seed")
    device = get_str(run, "device", default="auto", where="cfg.run.device")
    tensorboard = get_bool(run, "tensorboard", default=True, where="cfg.run.tensorboard")
    n_envs = get_int(run, "n_envs", default=8, where="cfg.run.n_envs")
    vec_raw = get_str(run, "vec", default="subproc", where="cfg.run.vec")

    name = _non_empty_str(name, where="cfg.run.name")
    out_root_s = _non_empty_str(out_root, where="cfg.run.out_root")

    n_envs_i = int(n_envs)
    if n_envs_i < 1:
        raise ValueError(f"cfg.run.n_envs must be >= 1 (got {n_envs!r})")

    vec = _parse_vec_kind(vec_raw, where="cfg.run.vec")

    device_s = str(device).strip()
    if not device_s:
        device_s = "auto"

    return RunSpec(
        name=str(name),
        out_root=Path(out_root_s),
        seed=int(seed),
        device=str(device_s),
        tensorboard=bool(tensorboard),
        n_envs=int(n_envs_i),
        vec=vec,
    )


__all__ = ["RunSpec", "VecKind", "parse_run_spec"]
