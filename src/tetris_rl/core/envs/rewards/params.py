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
    survival_bonus: float = 0.0


class LinesCleanRewardParams(InvalidPenaltyParams):
    terminal_penalty: float = 10.0
    survival_bonus: float = 0.0
    no_new_holes_bonus: float = 1.0


class LinesShapeRewardParams(InvalidPenaltyParams):
    terminal_penalty: float = 10.0
    survival_bonus: float = 0.0
    line_cleared_bonus: float = 1.0
    hole_added_penalty: float = 1.0
    no_new_holes_bonus: float = 0.1
    hole_removed_bonus: float = 1.0
    tetris_bonus: float = 1.0


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
