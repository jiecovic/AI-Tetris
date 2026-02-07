# src/tetris_rl/runs/config.py
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator

from tetris_rl.config.base import ConfigBase

VecKind = Literal["subproc", "dummy"]


class RunConfig(ConfigBase):
    """
    RunConfig owns ALL run-time wiring + filesystem/logging semantics.

    This is intentionally NOT "training semantics" (those live in ExperimentConfig).
    """

    name: str = "run"
    out_root: Path = Path("experiments")
    seed: int = 0
    device: str = "auto"
    tensorboard: bool = True
    n_envs: int = Field(default=8, ge=1)
    workers: int = Field(default=1, ge=1)
    vec: VecKind = "subproc"

    @field_validator("name")
    @classmethod
    def _name_nonempty(cls, v: str) -> str:
        s = str(v).strip()
        if not s:
            raise ValueError("run.name must be a non-empty string")
        return s

    @field_validator("out_root")
    @classmethod
    def _out_root_nonempty(cls, v: Path) -> Path:
        s = str(v).strip()
        if not s:
            raise ValueError("run.out_root must be a non-empty string")
        return Path(s)

    @field_validator("device", mode="before")
    @classmethod
    def _device_default(cls, v: object) -> str:
        if v is None:
            return "auto"
        s = str(v).strip()
        return s if s else "auto"

    @field_validator("vec", mode="before")
    @classmethod
    def _vec_lower(cls, v: object) -> str:
        return str(v).strip().lower()


__all__ = ["RunConfig", "VecKind"]
