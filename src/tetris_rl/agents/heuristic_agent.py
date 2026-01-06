# src/tetris_rl/agents/heuristic_agent.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple

import numpy as np

from tetris_rl.datagen.schema import FEATURE_NAMES
from tetris_rl.game.core.game import TetrisGame
from tetris_rl.game.core.macro_step import decode_discrete_action_id, encode_discrete_action_id
from tetris_rl.game.core.metrics import BoardSnapshotMetrics, board_snapshot_metrics_from_grid
from tetris_rl.game.core.placement_cache import StaticPlacementCache
from tetris_rl.game.core.simulate import MacroPlacementSimulator, SimPlacementResult
from tetris_rl.game.core.types import State


@dataclass(frozen=True)
class HeuristicWeights:
    # CodemyRoad: a*agg_height + b*complete_lines + c*holes + d*bumpiness
    a_agg_height: float = -0.510066
    b_lines: float = 0.760666
    c_holes: float = -0.35663
    d_bumpiness: float = -0.184483


class HeuristicAgent:
    """
    CodemyRoad-style heuristic placement agent Φ with optional 1-piece lookahead.

    Canonical output is a Discrete action_id under bbox-left semantics:
      action_id = encode_discrete_action_id(rot, col, board_w)

    Candidates hook:
      - best_action_id(..., candidates=iterable[action_id]) restricts search to those ids.

    Reward-fit labels:
      - evaluate_action_set(...) computes dense per-action phi + delta for the
        provided candidate ids, aligned to global action_id order.

        IMPORTANT:
          - If lookahead=0: phi is immediate Φ(s') of the placement.
          - If lookahead=1: phi reflects PLANNING using next_kind (part of state):
              * compute cheap=Φ(s1) for all candidates
              * select beam = top-k by cheap (k=beam_width)
              * for actions in beam: phi = max_a2 Φ(s2) after placing next_kind
              * for actions outside beam: phi = cheap
          - delta is ALWAYS immediate (from s -> s1) in schema.FEATURE_NAMES order.

    Terminal/lock-out handling (CRITICAL for reward-fit):
      - For reward-fit: terminal placements MUST remain phi=-inf so the fitter can drop them.
      - Therefore: evaluate_action_set() SKIPS r1.game_over completely (no cheap score, no delta).
      - For action selection: best_action_id() also skips r1.game_over (treat as invalid move).
    """

    def __init__(
            self,
            *,
            game: TetrisGame,
            weights: HeuristicWeights = HeuristicWeights(),
            lookahead: int = 1,  # 0 = immediate only, 1 = 1-piece lookahead
            beam_width: int = 10,  # used when lookahead=1
    ) -> None:
        self.game = game
        self.w = weights
        self.lookahead = int(lookahead)
        self.beam_width = int(beam_width)

        self.board_w = int(game.w)
        self.board_h = int(game.h)

        pieces = getattr(game, "pieces", None)
        if pieces is None:
            raise RuntimeError("HeuristicAgent requires game.pieces (PieceSet)")
        if not hasattr(pieces, "max_rotations"):
            raise RuntimeError("HeuristicAgent requires PieceSet.max_rotations() derived from YAML rotations")

        # Single source of truth: derived from assets via PieceSet
        self.max_rots = int(pieces.max_rotations())
        if self.max_rots <= 0:
            raise RuntimeError(f"invalid max_rots derived from pieces: {self.max_rots}")

        self.sim = MacroPlacementSimulator(
            pieces=pieces,
            board_w=int(game.w),
            board_h=int(game.h),
            spawn_rows=int(game.spawn_rows),
            empty_cell=0,
        )

        # Geometry cache for candidate generation (no collision checks here).
        self._legal = StaticPlacementCache.build(
            pieces=pieces,
            board_w=int(game.w),
        )

        # kind -> tuple[(rot, col)] geometry-legal under bbox-left semantics
        self._actions_cache: dict[str, Tuple[Tuple[int, int], ...]] = {}

    def _phi(self, *, cleared_lines: int, metrics: Any) -> float:
        return (
                self.w.a_agg_height * float(metrics.agg_height)
                + self.w.b_lines * float(cleared_lines)
                + self.w.c_holes * float(metrics.holes)
                + self.w.d_bumpiness * float(metrics.bumpiness)
        )

    def _precomputed_actions(self, kind: str) -> Tuple[Tuple[int, int], ...]:
        """
        All geometry-legal (rot, col) actions for this kind:
          - rot must be asset-valid for kind
          - col is bbox-left column in [0..bbox_left_max(rot)]
        """
        k = str(kind)
        hit = self._actions_cache.get(k)
        if hit is not None:
            return hit

        out: list[tuple[int, int]] = []
        for rot in range(int(self.max_rots)):
            if not self._legal.is_valid_rotation(k, int(rot)):
                continue
            _minx, _bbox_w, bbox_left_max = self._legal.bbox_params(k, int(rot))
            for col in range(int(bbox_left_max) + 1):
                out.append((int(rot), int(col)))

        tup = tuple(out) if out else ((0, 0),)
        self._actions_cache[k] = tup
        return tup

    def _metrics(self, grid: np.ndarray) -> BoardSnapshotMetrics:
        return board_snapshot_metrics_from_grid(np.asarray(grid))

    def _delta_vec(
            self,
            *,
            before: BoardSnapshotMetrics,
            after: BoardSnapshotMetrics,
            cleared_lines: int,
            placed_cells_cleared: int,
            placed_cells_all_cleared: bool,
    ) -> np.ndarray:
        """
        Delta vector in EXACT schema.FEATURE_NAMES order.

        This is name-driven so FEATURE_NAMES can grow without silently mis-indexing.
        Any unknown feature name stays 0.0 (caller can detect via list-features).
        """
        f = int(len(FEATURE_NAMES))
        out = np.zeros((f,), dtype=np.float32)

        vals: dict[str, float] = {
            # legacy deltas
            "cleared_lines": float(cleared_lines),
            "delta_holes": float(after.holes) - float(before.holes),
            "delta_max_height": float(after.max_height) - float(before.max_height),
            "delta_bumpiness": float(after.bumpiness) - float(before.bumpiness),
            "delta_agg_height": float(after.agg_height) - float(before.agg_height),
            # new per-action signals (must be produced per simulated action)
            "placed_cells_cleared": float(int(placed_cells_cleared)),
            "placed_cells_all_cleared": float(1.0 if bool(placed_cells_all_cleared) else 0.0),
        }

        for i, name in enumerate(FEATURE_NAMES):
            v = vals.get(str(name))
            if v is not None:
                out[int(i)] = np.float32(v)

        return out

    def evaluate_action_set(
            self,
            state: State,
            *,
            candidates: Iterable[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return dense per-action arrays aligned to global action_id order.

        - phi:   (A,) float32, default -inf for non-candidates
        - delta: (A,F) float32, default 0 for non-candidates

        Terminal placements:
          - skipped entirely so phi stays -inf (lets reward-fit drop them cleanly)
        """
        A = int(self.max_rots * self.board_w)
        F = int(len(FEATURE_NAMES))

        phi = np.full((A,), -np.inf, dtype=np.float32)
        delta = np.zeros((A, F), dtype=np.float32)

        cur_kind = str(state.active.kind)
        next_kind = str(state.next_kind)
        start_y = int(state.active.y)

        m0 = self._metrics(state.grid)

        # Materialize + de-dupe candidate ids in-range (preserve order best-effort)
        cand_ids: list[int] = []
        seen: set[int] = set()
        for aid0 in candidates:
            aid = int(aid0)
            if 0 <= aid < A and aid not in seen:
                seen.add(aid)
                cand_ids.append(aid)

        if not cand_ids:
            return phi, delta

        # Pass 1: simulate immediate result for all candidates -> cheap score + delta
        r1_by_aid: dict[int, SimPlacementResult] = {}
        cheap_by_aid: dict[int, float] = {}

        for aid in cand_ids:
            rot, col = decode_discrete_action_id(action_id=int(aid), board_w=int(self.board_w))
            r1 = self.sim.simulate(
                grid=state.grid,
                kind=cur_kind,
                rot=int(rot),
                col=int(col),
                start_y=int(start_y),
            )

            # Candidates come from legality mask; illegal should not happen.
            if r1.illegal_reason is not None:
                continue

            # CRITICAL: skip terminal/lock-out placements for reward-fit
            if bool(r1.game_over):
                continue

            r1_by_aid[int(aid)] = r1
            cheap = float(self._phi(cleared_lines=int(r1.cleared_lines), metrics=r1.metrics_after))
            cheap_by_aid[int(aid)] = cheap

            # These must be produced by the simulator per action.
            # Keep getattr fallback so code runs even before simulate.py is updated.
            pcc = int(getattr(r1, "placed_cells_cleared", 0))
            pac = bool(getattr(r1, "placed_cells_all_cleared", False))

            delta[int(aid), :] = self._delta_vec(
                before=m0,
                after=r1.metrics_after,
                cleared_lines=int(r1.cleared_lines),
                placed_cells_cleared=int(pcc),
                placed_cells_all_cleared=bool(pac),
            )

        if not cheap_by_aid:
            return phi, delta

        # lookahead=0 -> phi is immediate cheap for all valid candidates
        if int(self.lookahead) <= 0:
            for aid, cheap in cheap_by_aid.items():
                phi[int(aid)] = np.float32(cheap)
            return phi, delta

        # lookahead=1 (beam semantics): expand only top-k by cheap
        k_beam = max(1, int(self.beam_width))
        ranked = sorted(cheap_by_aid.items(), key=lambda kv: kv[1], reverse=True)
        beam_aids = {aid for (aid, _s) in ranked[: min(k_beam, len(ranked))]}

        # Default phi for all valid candidates is cheap (since non-beam are not expanded)
        for aid, cheap in cheap_by_aid.items():
            phi[int(aid)] = np.float32(cheap)

        # Expand beam actions with next_kind planning value
        next_actions = self._precomputed_actions(next_kind)
        next_start_y = 0

        for aid in beam_aids:
            r1 = r1_by_aid.get(int(aid))
            if r1 is None:
                continue

            best2 = -float("inf")
            for rot2, col2 in next_actions:
                r2 = self.sim.simulate(
                    grid=r1.grid_after,
                    kind=next_kind,
                    rot=int(rot2),
                    col=int(col2),
                    start_y=int(next_start_y),
                )
                if r2.illegal_reason is not None or bool(r2.game_over):
                    continue
                s2 = float(self._phi(cleared_lines=int(r2.cleared_lines), metrics=r2.metrics_after))
                if s2 > best2:
                    best2 = s2

            # If no valid r2 found, fall back to immediate cheap (already set)
            if np.isfinite(best2):
                phi[int(aid)] = np.float32(best2)

        return phi, delta

    def best_action_id(
            self,
            state: State,
            *,
            candidates: Optional[Iterable[int]] = None,
    ) -> int:
        """
        Return best Discrete action_id.

        IMPORTANT:
          - terminal/lock-out placements are treated as invalid (skipped)
        """
        cur_kind = str(state.active.kind)
        next_kind = str(state.next_kind)
        start_y = int(state.active.y)

        # Build candidate (rot,col) pairs
        if candidates is None:
            actions: List[Tuple[int, int]] = list(self._precomputed_actions(cur_kind))
        else:
            actions = [
                decode_discrete_action_id(action_id=int(aid), board_w=int(self.board_w))
                for aid in candidates
            ]

        if not actions:
            actions = [(0, 0)]

        # ------------------------------------------------------------
        # lookahead = 0 (immediate)
        # ------------------------------------------------------------
        if int(self.lookahead) <= 0:
            best_score = -float("inf")
            best_rot_col: Tuple[int, int] = (actions[0][0], actions[0][1])
            found = False

            for rot, col in actions:
                r1 = self.sim.simulate(
                    grid=state.grid,
                    kind=cur_kind,
                    rot=int(rot),
                    col=int(col),
                    start_y=int(start_y),
                )
                if r1.illegal_reason is not None or bool(r1.game_over):
                    continue
                s = float(self._phi(cleared_lines=int(r1.cleared_lines), metrics=r1.metrics_after))
                if (not found) or (s > best_score):
                    found = True
                    best_score = s
                    best_rot_col = (int(rot), int(col))

            return encode_discrete_action_id(
                rot=int(best_rot_col[0]),
                col=int(best_rot_col[1]),
                board_w=int(self.board_w),
            )

        # ------------------------------------------------------------
        # lookahead = 1 (beam)
        # ------------------------------------------------------------
        scored_r1: List[Tuple[float, int, int, SimPlacementResult]] = []
        for rot, col in actions:
            r1 = self.sim.simulate(
                grid=state.grid,
                kind=cur_kind,
                rot=int(rot),
                col=int(col),
                start_y=int(start_y),
            )
            if r1.illegal_reason is not None or bool(r1.game_over):
                continue
            cheap = float(self._phi(cleared_lines=int(r1.cleared_lines), metrics=r1.metrics_after))
            scored_r1.append((cheap, int(rot), int(col), r1))

        if not scored_r1:
            # deterministic fallback
            return encode_discrete_action_id(rot=int(actions[0][0]), col=int(actions[0][1]), board_w=int(self.board_w))

        k_beam = max(1, int(self.beam_width))
        scored_r1.sort(key=lambda t: t[0], reverse=True)
        beam = scored_r1[: min(k_beam, len(scored_r1))]

        best_score = -float("inf")
        best_rot_col: Tuple[int, int] = (beam[0][1], beam[0][2])

        next_actions = self._precomputed_actions(next_kind)
        next_start_y = 0

        for cheap1, rot, col, r1 in beam:
            best2 = -float("inf")
            for rot2, col2 in next_actions:
                r2 = self.sim.simulate(
                    grid=r1.grid_after,
                    kind=next_kind,
                    rot=int(rot2),
                    col=int(col2),
                    start_y=int(next_start_y),
                )
                if r2.illegal_reason is not None or bool(r2.game_over):
                    continue
                s2 = float(self._phi(cleared_lines=int(r2.cleared_lines), metrics=r2.metrics_after))
                if s2 > best2:
                    best2 = s2

            if not np.isfinite(best2):
                best2 = float(cheap1)

            if best2 > best_score:
                best_score = float(best2)
                best_rot_col = (int(rot), int(col))

        return encode_discrete_action_id(
            rot=int(best_rot_col[0]),
            col=int(best_rot_col[1]),
            board_w=int(self.board_w),
        )

    def best_macro_action(
            self,
            state: State,
            *,
            candidates: Optional[Iterable[Tuple[int, int]]] = None,
    ) -> Tuple[int, int]:
        """
        Compatibility wrapper.

        Returns (rot, col) using bbox-left semantics.
        Implemented by calling the canonical best_action_id() and decoding it.
        """
        if candidates is not None:
            cand_ids = (
                encode_discrete_action_id(rot=int(r), col=int(c), board_w=int(self.board_w))
                for (r, c) in candidates
            )
        else:
            cand_ids = None

        aid = self.best_action_id(state, candidates=cand_ids)
        rot, col = decode_discrete_action_id(action_id=int(aid), board_w=int(self.board_w))
        return int(rot), int(col)


__all__ = ["HeuristicWeights", "HeuristicAgent"]
