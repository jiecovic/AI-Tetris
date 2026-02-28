# src/tetris_rl/core/game/warmup_params.py
from __future__ import annotations

from typing import Mapping

from pydantic import Field, field_validator, model_validator

from tetris_rl.core.config.base import ConfigBase


def _as_int(value: object, *, where: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{where} must be an int, got bool")
    if not isinstance(value, (int, float, str, bytes, bytearray)):
        raise TypeError(f"{where} must be an int-like value, got {type(value)!r}")
    try:
        return int(value)
    except Exception as e:
        raise TypeError(f"{where} must be an int-like value, got {type(value)!r}") from e


def _parse_hole_range(value: object, *, where: str) -> dict[str, int]:
    if isinstance(value, Mapping):
        if "min" not in value or "max" not in value:
            raise KeyError(f"{where} dict must contain keys {{min,max}}")
        lo = _as_int(value["min"], where=f"{where}.min")
        hi = _as_int(value["max"], where=f"{where}.max")
    elif isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"{where} must have exactly 2 values [min,max]")
        lo = _as_int(value[0], where=f"{where}[0]")
        hi = _as_int(value[1], where=f"{where}[1]")
    elif isinstance(value, str):
        s = value.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1].strip()
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 2:
            raise ValueError(f"{where} string range must be 'min,max'")
        lo = _as_int(parts[0], where=f"{where}.min")
        hi = _as_int(parts[1], where=f"{where}.max")
    else:
        raise TypeError(f"{where} must be [min,max], 'min,max', or {{min,max}}")
    return {"min": int(lo), "max": int(hi)}


def parse_hole_range(value: object, *, where: str) -> tuple[int, int]:
    parsed = _parse_hole_range(value, where=where)
    return int(parsed["min"]), int(parsed["max"])


def extract_holes_config(params: Mapping[str, object]) -> tuple[int, tuple[int, int] | None]:
    """
    Normalize warmup holes config from params mapping.

    Supports:
      - holes: int
      - holes: [min,max] (or "min,max" / {min,max})
      - uniform_holes: {min,max}  (legacy key)
    """
    holes_raw = params.get("holes", 1)
    holes_range: tuple[int, int] | None = None
    fixed_holes: int

    if isinstance(holes_raw, (list, tuple, Mapping)):
        holes_range = parse_hole_range(holes_raw, where="warmup.params.holes")
        fixed_holes = int(holes_range[0])
    elif isinstance(holes_raw, str) and "," in holes_raw:
        holes_range = parse_hole_range(holes_raw, where="warmup.params.holes")
        fixed_holes = int(holes_range[0])
    else:
        fixed_holes = _as_int(holes_raw, where="warmup.params.holes")

    uh = params.get("uniform_holes", None)
    if uh is not None:
        legacy_range = parse_hole_range(uh, where="warmup.params.uniform_holes")
        if holes_range is None:
            holes_range = legacy_range
        elif holes_range != legacy_range:
            raise ValueError(
                "warmup.params.holes range and warmup.params.uniform_holes disagree; use one or make them equal"
            )

    return int(fixed_holes), holes_range


class HoleRangeConfig(ConfigBase):
    min: int = Field(ge=0)
    max: int = Field(ge=0)

    @model_validator(mode="before")
    @classmethod
    def _validate_order(cls, data: object) -> object:
        if not isinstance(data, Mapping):
            return data
        out = dict(data)
        if "min" in out and "max" in out:
            lo = _as_int(out["min"], where="range.min")
            hi = _as_int(out["max"], where="range.max")
            if lo > hi:
                raise ValueError(f"range min must be <= max (got {lo}>{hi})")
        return out


class WarmupHolesParams(ConfigBase):
    holes: int | HoleRangeConfig = 1
    uniform_holes: HoleRangeConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_hole_keys(cls, data: object) -> object:
        if not isinstance(data, Mapping):
            return data
        out = dict(data)
        if "holes" in out:
            h = out.get("holes")
            if isinstance(h, (list, tuple, Mapping)) or (isinstance(h, str) and "," in h):
                out["holes"] = _parse_hole_range(h, where="warmup.params.holes")
        if "uniform_holes" in out and out.get("uniform_holes") is not None:
            uh = out.get("uniform_holes")
            if isinstance(uh, (list, tuple, Mapping)) or (isinstance(uh, str) and "," in uh):
                out["uniform_holes"] = _parse_hole_range(uh, where="warmup.params.uniform_holes")
        return out

    @field_validator("holes", mode="before")
    @classmethod
    def _holes_non_bool(cls, v: object) -> object:
        if isinstance(v, bool):
            raise TypeError("warmup.params.holes must not be bool")
        return v

    @field_validator("holes", mode="after")
    @classmethod
    def _holes_non_negative(cls, v: int | HoleRangeConfig) -> int | HoleRangeConfig:
        if isinstance(v, int) and int(v) < 0:
            raise ValueError("warmup.params.holes must be >= 0")
        return v

    @model_validator(mode="before")
    @classmethod
    def _validate_hole_range_consistency(cls, data: object) -> object:
        if not isinstance(data, Mapping):
            return data
        out = dict(data)
        holes = out.get("holes", 1)
        uniform_holes = out.get("uniform_holes", None)

        holes_range: dict[str, int] | None = None
        if isinstance(holes, Mapping):
            if "min" in holes and "max" in holes:
                holes_range = _parse_hole_range(holes, where="warmup.params.holes")
        elif isinstance(holes, (list, tuple)) or (isinstance(holes, str) and "," in holes):
            holes_range = _parse_hole_range(holes, where="warmup.params.holes")

        uniform_range: dict[str, int] | None = None
        if uniform_holes is not None:
            if isinstance(uniform_holes, Mapping):
                if "min" in uniform_holes and "max" in uniform_holes:
                    uniform_range = _parse_hole_range(uniform_holes, where="warmup.params.uniform_holes")
            elif isinstance(uniform_holes, (list, tuple)) or (isinstance(uniform_holes, str) and "," in uniform_holes):
                uniform_range = _parse_hole_range(uniform_holes, where="warmup.params.uniform_holes")

        if holes_range is not None and uniform_range is not None:
            if holes_range["min"] != uniform_range["min"] or holes_range["max"] != uniform_range["max"]:
                raise ValueError(
                    "warmup.params.holes range and warmup.params.uniform_holes disagree; use one or make them equal"
                )
        return out


class WarmupNoneParams(WarmupHolesParams):
    pass


class WarmupFixedParams(WarmupHolesParams):
    rows: int = Field(ge=0)


class WarmupUniformRowsParams(WarmupHolesParams):
    min_rows: int = Field(ge=0)
    max_rows: int = Field(ge=0)

    @model_validator(mode="before")
    @classmethod
    def _validate_rows(cls, data: object) -> object:
        if not isinstance(data, Mapping):
            return data
        out = dict(data)
        if "min_rows" in out and "max_rows" in out:
            min_rows = _as_int(out["min_rows"], where="warmup.params.min_rows")
            max_rows = _as_int(out["max_rows"], where="warmup.params.max_rows")
            if min_rows > max_rows:
                raise ValueError(f"warmup.params.min_rows must be <= max_rows (got {min_rows}>{max_rows})")
        return out


class WarmupPoissonParams(WarmupHolesParams):
    lambda_: float
    cap: int = Field(ge=0)

    @model_validator(mode="before")
    @classmethod
    def _normalize_lambda(cls, data: object) -> object:
        if not isinstance(data, Mapping):
            return data
        out = dict(data)
        if "lambda_" not in out and "lambda" in out:
            out["lambda_"] = out.get("lambda")
        return out

    @field_validator("lambda_", mode="after")
    @classmethod
    def _validate_lambda(cls, v: float) -> float:
        if float(v) <= 0.0:
            raise ValueError("warmup.params.lambda must be > 0")
        return float(v)


class WarmupBasePlusPoissonParams(WarmupHolesParams):
    base: int = Field(ge=0)
    lambda_: float
    cap: int = Field(ge=0)

    @model_validator(mode="before")
    @classmethod
    def _normalize_lambda(cls, data: object) -> object:
        if not isinstance(data, Mapping):
            return data
        out = dict(data)
        if "lambda_" not in out and "lambda" in out:
            out["lambda_"] = out.get("lambda")
        return out

    @field_validator("lambda_", mode="after")
    @classmethod
    def _validate_lambda(cls, v: float) -> float:
        if float(v) <= 0.0:
            raise ValueError("warmup.params.lambda must be > 0")
        return float(v)


WarmupParams = (
    WarmupNoneParams | WarmupFixedParams | WarmupUniformRowsParams | WarmupPoissonParams | WarmupBasePlusPoissonParams
)


__all__ = [
    "extract_holes_config",
    "HoleRangeConfig",
    "parse_hole_range",
    "WarmupHolesParams",
    "WarmupNoneParams",
    "WarmupFixedParams",
    "WarmupUniformRowsParams",
    "WarmupPoissonParams",
    "WarmupBasePlusPoissonParams",
    "WarmupParams",
]
