# src/tetris_rl/env_bundles/rewards/heuristics.py

"""
Heuristic reward shaping based on a linear Tetris board evaluation function.

Based on the Stanford CS231N (2016) student project on Tetris, which defines
a hand-tuned linear fitness Φ(s) and uses the per-step reward ΔΦ = Φ(s_{t+1}) − Φ(s_t).

Reference:
  https://cs231n.stanford.edu/reports/2016/pdfs/121_Report.pdf
"""

from __future__ import annotations

from dataclasses import dataclass

from tetris_rl.envs.api import TransitionFeatures


@dataclass(frozen=True)
class DeltaHeuristicWeights:
    """
    Weights for delta-based board quality scoring.

    Convention:
      score = gains - damages
      - positive score => good placement
      - negative score => bad placement

    CS231N-style note:
      This helper is meant to represent the *delta of a linear board evaluation function* Φ,
      i.e. score = Φ(s') - Φ(s). For a linear Φ, this is exactly a weighted sum of deltas.
      Therefore we:
        - use deltas symmetrically (no pos(), no improve_frac),
        - treat cleared_lines as the per-step Δcomplete_lines term,
        - keep a single linear weight for line clears (NOT a nonlinear schedule).
    """

    # Linear line-clear term (treat cleared_lines as Δcomplete_lines for this placement).
    # CS231N: +0.76 * complete_lines
    w_lines: float = 0.76

    # Structural terms (weights apply symmetrically to deltas).
    # CS231N fitness function:
    #   Φ(s) = -0.51 * aggregate_height(s) + 0.76 * complete_lines(s)
    #          -0.36 * holes(s) -0.18 * bumpiness(s)
    #
    # Therefore:
    #   ΔΦ = +0.76 * cleared_lines
    #        -0.36 * delta_holes
    #        -0.51 * delta_agg_height
    #        -0.18 * delta_bumpiness
    #
    # CS231N does not include a max-height term.
    w_holes: float = 0.86
    w_agg_height: float = 0.31
    w_bumpiness: float = 0.10
    w_max_height: float = 0.001


def delta_heuristic_score(*, features: TransitionFeatures, w: DeltaHeuristicWeights) -> float:
    """
    Convert TransitionFeatures deltas into a single scalar "quality score".

    Returns:
      score: float where higher is better.

    Uses ONLY deltas (and cleared_lines) -> no absolute board peeking.

    CS231N-style:
      score = Φ(s') - Φ(s) for a linear board evaluation Φ.
      Implemented as a simple weighted sum of deltas (+ linear cleared_lines term).
    """
    score = 0.0

    # ---- line clears (linear gain) ----
    cl = int(getattr(features, "cleared_lines", 0) or 0)
    cl = max(0, min(cl, 4))  # hard cap: never error out (supports alt rules / bug tolerance)
    score += float(w.w_lines) * float(cl)

    # ---- deltas: symmetric (linear potential difference) ----

    dh = getattr(features, "delta_holes", None)
    if dh is not None:
        score -= float(w.w_holes) * float(int(dh)-1)

    dah = getattr(features, "delta_agg_height", None)
    if dah is not None:
        score -= float(w.w_agg_height) * float(int(dah))

    db = getattr(features, "delta_bumpiness", None)
    if db is not None:
        score -= float(w.w_bumpiness) * float(int(db))

    dmh = getattr(features, "delta_max_height", None)
    if dmh is not None:
        score -= float(w.w_max_height) * float(int(dmh))

    return float(score)
