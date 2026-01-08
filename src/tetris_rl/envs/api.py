# src/tetris_rl/env_bundles/api.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol, runtime_checkable

import numpy as np


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
      - used_rotation / used_column: the action actually executed (may differ if remapped).

    Strict legality (North Star: invalid_rot OR oob OR collision at current py):
      - applied: True iff we actually applied *some* placement to the active piece and executed hard_drop.
                (False for noop/terminate, or if remap fails / no legal actions exist.)
      - invalid_action: True iff the originally requested action was illegal under strict rules.
      - illegal_reason: why it was illegal ("invalid_rotation" | "oob" | "collision" | ...), else None.
      - remapped: True iff we executed a different (rot,col) than requested due to invalid_action_policy.
      - invalid_action_policy: "closest_legal" | "random_legal" if remapped, else None.

    Mask/debug-only signals (mainly for MaskablePPO / sanity checks):
      - masked_action: whether the requested joint Discrete(rot×col) action was masked out as illegal.
                       Under “one truth”, this should agree with invalid_action in discrete mode.

    Optional shaping/diagnostics:
      - delta_holes / delta_max_height / delta_bumpiness / delta_agg_height: board metric deltas vs previous step.
      - agg_height: current aggregate height (after step), if computed.
      - board_before / board_after: optional raw grids (expensive; usually None).
      - max_height / holes / bumpiness: absolute metrics after step, if computed.

    New “piece vanished” signals (line-clear interaction):
      - placed_cells_cleared: how many of the 4 placed tetromino cells vanished due to cleared lines (0..4).
      - placed_all_cells_cleared: True iff all 4 placed cells vanished (strong “perfect fit” signal).
    """

    cleared_lines: int
    game_over: bool
    placed_kind: str

    requested_rotation: int
    requested_column: int
    used_rotation: int
    used_column: int

    applied: bool
    invalid_action: bool
    invalid_action_policy: str | None

    masked_action: bool

    delta_holes: int | None = None
    delta_max_height: int | None = None
    delta_bumpiness: int | None = None
    delta_agg_height: int | None = None
    agg_height: int | None = None

    placed_cells_cleared: int | None = None
    placed_all_cells_cleared: bool | None = None

    board_before: np.ndarray | None = None
    board_after: np.ndarray | None = None
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


@runtime_checkable
class WarmupFn(Protocol):
    """
    Post-reset mutator hook (shared by RL + imitation + datagen).

    Called by env.reset() AFTER game.reset() and BEFORE the first obs is emitted.
    Must only mutate game state (typically game.board.grid).
    """

    def __call__(self, *, game: Any, rng: np.random.Generator) -> None: ...


__all__ = ["TransitionFeatures", "RewardFn", "WarmupFn"]
