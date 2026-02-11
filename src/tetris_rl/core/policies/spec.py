# src/tetris_rl/core/policies/spec.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, Field, ValidationInfo, field_validator

from tetris_rl.core.config.io import load_yaml


class HeuristicSearch(BaseModel):
    plies: int = 1
    beam_width: int | None = None
    beam_from_depth: int = 0
    feature_clear_mode: str = "post"

    @field_validator("plies")
    @classmethod
    def _validate_plies(cls, v: int) -> int:
        if int(v) < 1:
            raise ValueError("search.plies must be >= 1")
        return v

    @field_validator("beam_width")
    @classmethod
    def _validate_beam_width(cls, v: int | None) -> int | None:
        if v is not None and int(v) < 1:
            raise ValueError("search.beam_width must be >= 1")
        return v

    @field_validator("beam_from_depth")
    @classmethod
    def _validate_beam_from_depth(cls, v: int) -> int:
        if int(v) < 0:
            raise ValueError("search.beam_from_depth must be >= 0")
        return v

    @field_validator("feature_clear_mode", mode="before")
    @classmethod
    def _validate_feature_clear_mode(cls, v: object) -> str:
        mode = str(v).strip().lower()
        if mode not in {"lock", "pre", "post", "clear"}:
            raise ValueError("search.feature_clear_mode must be pre|lock|post|clear")
        return mode


class HeuristicSpec(BaseModel):
    type: Literal["heuristic"] = "heuristic"
    name: str | None = None
    features: list[str] = Field(default_factory=list)
    weights: list[float] = Field(default_factory=list)
    search: HeuristicSearch = Field(default_factory=HeuristicSearch)

    @field_validator("features")
    @classmethod
    def _validate_features(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("features must be non-empty")
        return v

    @field_validator("weights", mode="after")
    @classmethod
    def _validate_weights(cls, v: list[float], info: ValidationInfo) -> list[float]:
        features = info.data.get("features", [])
        if not features:
            raise ValueError("features must be non-empty")
        if len(features) != len(v):
            raise ValueError("features/weights length mismatch")
        return v


def load_heuristic_spec(path: Path) -> HeuristicSpec:
    return HeuristicSpec.model_validate(load_yaml(path))


def save_heuristic_spec(path: Path, spec: HeuristicSpec) -> Path:
    data: dict[str, Any] = spec.model_dump(mode="json")
    cfg = OmegaConf.create(data)
    out_path = Path(path)
    OmegaConf.save(cfg, out_path)
    return out_path


__all__ = [
    "HeuristicSearch",
    "HeuristicSpec",
    "load_heuristic_spec",
    "save_heuristic_spec",
]
