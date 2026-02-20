# src/tetris_rl/core/envs/rewards/params.py
from __future__ import annotations

from typing import Mapping

from pydantic import model_validator

from tetris_rl.core.config.base import ConfigBase


class InvalidPenaltyParams(ConfigBase):
    invalid_penalty: float = 10.0

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_penalty_key(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        if "illegal_penalty" not in data:
            return data

        out = dict(data)
        legacy = out.pop("illegal_penalty")
        if "invalid_penalty" not in out:
            out["invalid_penalty"] = legacy
            return out

        try:
            if float(out["invalid_penalty"]) != float(legacy):
                raise ValueError("invalid_penalty and illegal_penalty disagree; use invalid_penalty only")
        except Exception:
            raise ValueError("invalid_penalty and illegal_penalty disagree; use invalid_penalty only")
        return out

    @property
    def illegal_penalty(self) -> float:
        # Back-compat attribute shim for older call sites.
        return float(self.invalid_penalty)


class LinesRewardParams(InvalidPenaltyParams):
    terminal_penalty: float = 10.0
    survival_bonus: float = 0.0


class HeuristicDeltaRewardParams(InvalidPenaltyParams):
    terminal_penalty: float = 10.0
    w_lines: float = 0.760666
    w_holes: float = -0.35663
    w_bumpiness: float = -0.184483
    w_agg_height: float = -0.510066
    tetris_bonus: float = 0.0
    clip_term_lines: float | None = None
    clip_term_holes: float | None = None
    clip_term_bumpiness: float | None = None
    clip_term_agg_height: float | None = None
    survival_bonus: float = 0.0
    survival_scale_mode: str = "none"
    survival_scale_floor: float = 0.0
    survival_scale_power: float = 1.0

    @model_validator(mode="before")
    @classmethod
    def _validate_clips(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        for name in ("clip_term_lines", "clip_term_holes", "clip_term_bumpiness", "clip_term_agg_height"):
            value = out.get(name)
            if value is None:
                continue
            if float(value) < 0.0:
                raise ValueError(f"{name} must be >= 0 when set")
        mode = str(out.get("survival_scale_mode", "none")).strip().lower()
        if mode not in {"none", "max_h", "agg_h"}:
            raise ValueError("survival_scale_mode must be one of: none|max_h|agg_h")
        out["survival_scale_mode"] = mode
        floor = float(out.get("survival_scale_floor", 0.0))
        if not (0.0 <= floor <= 1.0):
            raise ValueError("survival_scale_floor must be in [0,1]")
        power = float(out.get("survival_scale_power", 1.0))
        if power < 1.0:
            raise ValueError("survival_scale_power must be >= 1")
        return out


class LinesCleanRewardParams(InvalidPenaltyParams):
    terminal_penalty: float = 10.0
    survival_bonus: float = 0.0
    no_new_holes_bonus: float = 1.0


class LinesShapeRewardParams(InvalidPenaltyParams):
    terminal_penalty: float = 10.0
    survival_bonus: float = 0.0
    line_cleared_bonus: float = 1.0
    holes_increase_reward: float = -1.0
    holes_same_reward: float = 0.1
    holes_decrease_reward: float = 1.1
    bumpiness_increase_reward: float = 0.0
    bumpiness_same_reward: float = 0.0
    bumpiness_decrease_reward: float = 0.0
    max_height_increase_reward: float = 0.0
    max_height_same_reward: float = 0.0
    max_height_decrease_reward: float = 0.0
    agg_height_increase_reward: float = 0.0
    agg_height_same_reward: float = 0.0
    agg_height_decrease_reward: float = 0.0
    tetris_bonus: float = 1.0

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_hole_keys(cls, data: object) -> object:
        """
        Back-compat for older lines_shape configs:
          - hole_added_penalty
          - no_new_holes_bonus
          - hole_removed_bonus
        """
        if not isinstance(data, dict):
            return data

        out = dict(data)
        old_inc_pen = out.pop("hole_added_penalty", None)
        old_same_bonus = out.pop("no_new_holes_bonus", None)
        old_dec_extra = out.pop("hole_removed_bonus", None)

        if old_inc_pen is None and old_same_bonus is None and old_dec_extra is None:
            return data

        def _set_or_check(name: str, value: float) -> None:
            if name not in out:
                out[name] = float(value)
                return
            if float(out[name]) != float(value):
                raise ValueError(
                    f"legacy hole key conflicts with {name}; use {name} only"
                )

        if old_inc_pen is not None:
            _set_or_check("holes_increase_reward", -float(old_inc_pen))
        if old_same_bonus is not None:
            _set_or_check("holes_same_reward", float(old_same_bonus))
        if old_dec_extra is not None:
            base_same = float(old_same_bonus) if old_same_bonus is not None else 0.0
            _set_or_check("holes_decrease_reward", base_same + float(old_dec_extra))

        return out


RewardParams = (
    LinesRewardParams
    | HeuristicDeltaRewardParams
    | LinesCleanRewardParams
    | LinesShapeRewardParams
)

REWARD_PARAMS_REGISTRY: Mapping[str, type[ConfigBase]] = {
    "lines": LinesRewardParams,
    "heuristic_delta": HeuristicDeltaRewardParams,
    "lines_clean": LinesCleanRewardParams,
    "lines_shape": LinesShapeRewardParams,
}

__all__ = [
    "LinesRewardParams",
    "HeuristicDeltaRewardParams",
    "LinesCleanRewardParams",
    "LinesShapeRewardParams",
    "RewardParams",
    "REWARD_PARAMS_REGISTRY",
]
