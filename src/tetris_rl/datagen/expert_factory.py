# src/tetris_rl/datagen/expert_factory.py
from __future__ import annotations

from tetris_rl.agents.heuristic_agent import HeuristicAgent, HeuristicWeights
from tetris_rl.config.datagen_spec import DataGenExpertSpec
from tetris_rl.game.core.game import TetrisGame


def make_expert_from_spec(
        *,
        game: TetrisGame,
        expert_spec: DataGenExpertSpec,
) -> HeuristicAgent:
    """
    Typed factory for DataGen experts.

    Policy:
      - Pure wiring only (no geometry / asset conventions here).
      - All action-space geometry is derived from the game assets themselves.
    """
    expert_type = str(expert_spec.type).strip().lower()
    if expert_type not in {"heuristic_agent", "heuristic"}:
        raise ValueError(f"unknown datagen expert.type: {expert_type!r}")

    hp = expert_spec.heuristic

    lookahead = int(hp.lookahead)
    if lookahead not in (0, 1):
        raise ValueError(f"expert.heuristic.lookahead must be 0 or 1, got {lookahead}")

    beam_width = int(hp.beam_width)
    if beam_width <= 0:
        raise ValueError(f"expert.heuristic.beam_width must be > 0, got {beam_width}")

    w = hp.weights
    weights = HeuristicWeights(
        a_agg_height=float(w.a_agg_height),
        b_lines=float(w.b_lines),
        c_holes=float(w.c_holes),
        d_bumpiness=float(w.d_bumpiness),
    )

    return HeuristicAgent(
        game=game,
        weights=weights,
        lookahead=lookahead,
        beam_width=beam_width,
    )


__all__ = ["make_expert_from_spec"]
