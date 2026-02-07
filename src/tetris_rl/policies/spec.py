# src/tetris_rl/policies/spec.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, Field, model_validator

from tetris_rl.config.io import load_yaml


class HeuristicSearch(BaseModel):
    plies: int = 1
    beam_width: int | None = None
    beam_from_depth: int = 0

    @model_validator(mode="after")
    def _validate(self) -> "HeuristicSearch":
        if self.plies < 1:
            raise ValueError("search.plies must be >= 1")
        if self.beam_width is not None and self.beam_width < 1:
            raise ValueError("search.beam_width must be >= 1")
        if self.beam_from_depth < 0:
            raise ValueError("search.beam_from_depth must be >= 0")
        return self


class HeuristicSpec(BaseModel):
    type: Literal["heuristic"] = "heuristic"
    name: str | None = None
    features: list[str] = Field(default_factory=list)
    weights: list[float] = Field(default_factory=list)
    search: HeuristicSearch = Field(default_factory=HeuristicSearch)

    @model_validator(mode="after")
    def _validate(self) -> "HeuristicSpec":
        if not self.features:
            raise ValueError("features must be non-empty")
        if len(self.features) != len(self.weights):
            raise ValueError("features/weights length mismatch")
        return self


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
