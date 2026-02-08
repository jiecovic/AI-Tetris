# src/tetris_rl/core/envs/api.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Protocol, runtime_checkable


@dataclass(frozen=True)
class TransitionFeatures:
    """
    Extra signals computed during env.step() for reward shaping + logging.

    Core outcome:
      - cleared_lines: number of cleared lines on this step.
      - game_over: whether the episode terminated after this step (engine game-over / lock-out).
      - placed_kind: tetromino kind that was active when the action was chosen.

    Action identity (requested vs actually executed):
      - requested_rotation / requested_column: what the policy/agent asked for (macro action).
      - used_rotation / used_column: the action actually executed (same as requested unless engine remaps).

    Strict legality (engine truth):
      - applied: True iff we actually applied *some* placement to the active piece and executed hard_drop.
                (False for noop/terminate or invalid actions.)
      - invalid_action: True iff the originally requested action was illegal under strict rules.
      - invalid_action_policy: env-level policy for illegal actions ("noop" | "terminate").

    Mask/debug-only signals (mainly for MaskablePPO / sanity checks):
      - masked_action: whether the requested joint Discrete(rot×col) action was masked out as illegal.
                       In discrete mode, this should agree with invalid_action.

    Feature mode:
      - feature_clear_mode: "post" (after line clear) or "lock" (pre-clear).

    Optional shaping/diagnostics:
      - delta_holes / delta_max_height / delta_bumpiness / delta_agg_height: board metric deltas vs previous step.
      - agg_height: current aggregate height (after step), if computed.
      - max_height / holes / bumpiness: absolute metrics after step, if computed.
      - delta_* + (holes/max_height/bumpiness/agg_height) follow env.feature_clear_mode (post by default).
    """

    cleared_lines: int
    game_over: bool
    placed_kind: str

    requested_rotation: int
    requested_column: int
    used_rotation: int
    used_column: int
    used_action_id: int

    applied: bool
    invalid_action: bool
    invalid_action_policy: str | None

    masked_action: bool

    feature_clear_mode: Literal["post", "lock"]

    delta_holes: int | None = None
    delta_max_height: int | None = None
    delta_bumpiness: int | None = None
    delta_agg_height: int | None = None
    agg_height: int | None = None
    max_height: int | None = None
    holes: int | None = None
    bumpiness: int | None = None


@runtime_checkable
class RewardFn(Protocol):
    def __call__(
            self,
            *,
            prev_state: Any,
            action: Any,
            next_state: Any,
            features: TransitionFeatures,
            info: Dict[str, Any],
    ) -> float: ...


__all__ = ["TransitionFeatures", "RewardFn"]
