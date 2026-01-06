# src/tetris_rl/cli/verify_rewardfit_shards.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from tetris_rl.agents.heuristic_agent import HeuristicAgent, HeuristicWeights
from tetris_rl.datagen.shard_reader import ShardDataset
from tetris_rl.game.core.game import TetrisGame
from tetris_rl.game.core.types import ActivePiece, State
from tetris_rl.utils.logging import setup_logger
from tetris_rl.utils.paths import repo_root


@dataclass(frozen=True)
class StateCheck:
    shard_id: int
    state_idx: int
    legal_count: int
    top1_match: bool
    max_abs_phi: float
    mean_abs_phi: float
    max_abs_delta: Optional[float]
    mean_abs_delta: Optional[float]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Verify rewardfit shards by recomputing heuristic phi/delta using HeuristicAgent.\n\n"
            "Requirements per shard:\n"
            "  legal_mask: (N, A)\n"
            "  phi:        (N, A)\n"
            "  delta:      (N, A, F)   (optional unless --check-delta)\n\n"
            "And for reconstruction (strict):\n"
            "  grid:            (N, H, W)\n"
            "  active_kind:     (N,)   kind_idx in 0..K-1\n"
            "  next_kind:       (N,)   kind_idx in 0..K-1\n"
            "Optionally:\n"
            "  active_y:        (N,)   (defaults to 0)\n"
            "  active_rot:      (N,)   (defaults to 0)\n"
            "Notes:\n"
            "  Some datasets may store active_kind_idx/next_kind_idx; this script accepts either naming.\n"
        )
    )

    ap.add_argument("--dataset", "-d", type=str, required=True, help="dataset directory (contains manifest.json)")
    ap.add_argument("--shards", type=str, default="", help="comma-separated shard ids to use (default: all)")
    ap.add_argument("--states-per-shard", type=int, default=50, help="random states sampled per shard")
    ap.add_argument("--seed", type=int, default=0, help="rng seed for sampling")

    ap.add_argument("--lookahead", type=int, default=1, choices=[0, 1], help="heuristic lookahead (0 or 1)")
    ap.add_argument("--beam-width", type=int, default=10, help="beam width when lookahead=1")

    ap.add_argument("--check-delta", action="store_true", help="also compare delta (requires 'delta' arrays)")
    ap.add_argument("--atol", type=float, default=1e-6, help="absolute tolerance for reporting mismatches")
    ap.add_argument("--max-report", type=int, default=10, help="max worst states to print")

    ap.add_argument("--log-level", type=str, default="info", help="debug|info|warning|error")
    ap.add_argument("--no-rich", action="store_true", help="disable Rich logging")
    return ap.parse_args()


def _parse_shards_arg(s: str) -> Optional[List[int]]:
    if not str(s).strip():
        return None
    out: List[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out if out else None


def _require(arrays: Dict[str, np.ndarray], key: str, *, shard_id: int) -> np.ndarray:
    if key not in arrays:
        raise RuntimeError(f"shard_{int(shard_id):04d} missing array {key!r} (has: {sorted(arrays.keys())})")
    return arrays[key]


def _require_any(arrays: Dict[str, np.ndarray], keys: Sequence[str], *, shard_id: int) -> np.ndarray:
    for k in keys:
        if k in arrays:
            return arrays[k]
    raise RuntimeError(f"shard_{int(shard_id):04d} missing arrays {list(keys)!r} (has: {sorted(arrays.keys())})")


def _optional(arrays: Dict[str, np.ndarray], key: str) -> Optional[np.ndarray]:
    return arrays.get(key, None)


def _build_state_from_arrays(
        *,
        game: TetrisGame,
        grid: np.ndarray,
        active_kind_idx: int,
        next_kind_idx: int,
        active_y: int,
        active_rot: int,
) -> State:
    cur_kind = game.pieces.idx_to_kind(int(active_kind_idx))
    nxt_kind = game.pieces.idx_to_kind(int(next_kind_idx))

    ap = ActivePiece(kind=str(cur_kind), rot=int(active_rot), x=0, y=int(active_y))

    return State(
        grid=np.asarray(grid),
        score=0,
        lines=0,
        level=0,
        game_over=False,
        active=ap,
        next_kind=str(nxt_kind),
        active_kind_idx=int(active_kind_idx),
        next_kind_idx=int(next_kind_idx),
        active_rot=int(active_rot),
    )


def main() -> int:
    args = parse_args()
    logger = setup_logger(name="verify_rewardfit", use_rich=(not bool(args.no_rich)), level=str(args.log_level))

    repo = repo_root()
    dataset_dir = Path(args.dataset)
    if not dataset_dir.is_absolute():
        dataset_dir = (repo / dataset_dir).resolve()
    if not dataset_dir.is_dir():
        logger.error("dataset dir not found: %s", dataset_dir)
        return 2

    ds = ShardDataset(dataset_dir=dataset_dir)
    wanted = _parse_shards_arg(str(args.shards))
    shard_ids = ds.shard_ids()
    if wanted is not None:
        wanted_set = {int(x) for x in wanted}
        shard_ids = [int(s) for s in shard_ids if int(s) in wanted_set]
    shard_ids = sorted([int(s) for s in shard_ids])

    if not shard_ids:
        logger.error("no shards selected")
        return 2

    game = TetrisGame()
    agent = HeuristicAgent(
        game=game,
        weights=HeuristicWeights(),
        lookahead=int(args.lookahead),
        beam_width=int(args.beam_width),
    )

    rng = np.random.default_rng(int(args.seed))

    checks: List[StateCheck] = []
    total_states = 0
    total_top1_match = 0

    for sid in shard_ids:
        arrays = ds.get_shard(int(sid))

        # rewardfit arrays
        phi_rec = _require(arrays, "phi", shard_id=sid)  # (N,A)
        legal = _require(arrays, "legal_mask", shard_id=sid)  # (N,A)
        delta_rec = _optional(arrays, "delta")  # (N,A,F)

        # state reconstruction arrays (accept either naming)
        grids = _require(arrays, "grid", shard_id=sid)  # (N,H,W)
        ak = _require_any(arrays, ("active_kind", "active_kind_idx"), shard_id=sid)  # (N,)
        nk = _require_any(arrays, ("next_kind", "next_kind_idx"), shard_id=sid)  # (N,)

        ay_arr = _optional(arrays, "active_y")
        ar_arr = _optional(arrays, "active_rot")

        phi_rec = np.asarray(phi_rec, dtype=np.float32)
        legal = np.asarray(legal, dtype=bool)
        grids = np.asarray(grids)

        if phi_rec.ndim != 2 or legal.ndim != 2:
            raise RuntimeError(f"bad shapes in shard_{sid:04d}: phi{phi_rec.shape} legal{legal.shape}")
        if phi_rec.shape != legal.shape:
            raise RuntimeError(f"phi/legal mismatch in shard_{sid:04d}: {phi_rec.shape} vs {legal.shape}")
        if grids.ndim != 3 or grids.shape[0] != phi_rec.shape[0]:
            raise RuntimeError(f"grid mismatch in shard_{sid:04d}: grid{grids.shape} vs phi{phi_rec.shape}")

        N, A = phi_rec.shape

        # sanity: active/next kind arrays length
        ak = np.asarray(ak)
        nk = np.asarray(nk)
        if ak.ndim != 1 or nk.ndim != 1 or ak.shape[0] != N or nk.shape[0] != N:
            raise RuntimeError(
                f"kind arrays mismatch in shard_{sid:04d}: "
                f"active_kind{ak.shape} next_kind{nk.shape} vs N={N}"
            )

        sps = max(0, int(args.states_per_shard))
        if sps == 0:
            continue

        take = min(int(sps), int(N))
        idxs = rng.choice(np.arange(N, dtype=np.int64), size=int(take), replace=False)

        for i in idxs:
            mask = legal[int(i)]
            cand = np.flatnonzero(mask)
            if cand.size == 0:
                continue

            active_y = int(ay_arr[int(i)]) if ay_arr is not None else 0
            active_rot = int(ar_arr[int(i)]) if ar_arr is not None else 0

            st = _build_state_from_arrays(
                game=game,
                grid=grids[int(i)],
                active_kind_idx=int(ak[int(i)]),
                next_kind_idx=int(nk[int(i)]),
                active_y=int(active_y),
                active_rot=int(active_rot),
            )

            phi_new, delta_new = agent.evaluate_action_set(st, candidates=(int(aid) for aid in cand))

            # Compare phi on legal actions only.
            phi0 = phi_rec[int(i)]
            dphi = np.abs(np.asarray(phi_new, dtype=np.float32)[cand] - np.asarray(phi0, dtype=np.float32)[cand])
            max_abs_phi = float(np.max(dphi))
            mean_abs_phi = float(np.mean(dphi))

            # Top-1 agreement among legal actions.
            best_rec = int(cand[int(np.argmax(phi0[cand]))])
            best_new = int(cand[int(np.argmax(phi_new[cand]))])
            top1_match = bool(best_rec == best_new)

            max_abs_delta: Optional[float] = None
            mean_abs_delta: Optional[float] = None
            if bool(args.check_delta):
                if delta_rec is None:
                    raise RuntimeError(f"--check-delta set but shard_{sid:04d} has no 'delta' array")
                d0 = np.asarray(delta_rec[int(i)], dtype=np.float32)  # (A,F)
                d1 = np.asarray(delta_new, dtype=np.float32)  # (A,F)
                if d0.shape != d1.shape:
                    raise RuntimeError(f"delta shape mismatch shard_{sid:04d}: {d0.shape} vs {d1.shape}")
                dd = np.abs(d1[cand, :] - d0[cand, :])
                max_abs_delta = float(np.max(dd))
                mean_abs_delta = float(np.mean(dd))

            checks.append(
                StateCheck(
                    shard_id=int(sid),
                    state_idx=int(i),
                    legal_count=int(cand.size),
                    top1_match=bool(top1_match),
                    max_abs_phi=float(max_abs_phi),
                    mean_abs_phi=float(mean_abs_phi),
                    max_abs_delta=max_abs_delta,
                    mean_abs_delta=mean_abs_delta,
                )
            )

            total_states += 1
            total_top1_match += int(top1_match)

    if total_states == 0:
        logger.error("no comparable states found (maybe missing arrays or no legal actions?)")
        return 3

    frac = float(total_top1_match) / float(total_states)
    logger.info(
        "[verify_rewardfit] checked_states=%d top1_match=%d (%.2f%%)",
        total_states,
        total_top1_match,
        100.0 * frac,
    )

    atol = float(args.atol)
    worst = sorted(checks, key=lambda c: c.max_abs_phi, reverse=True)
    bad = [c for c in worst if float(c.max_abs_phi) > atol]

    if not bad:
        logger.info("[verify_rewardfit] phi OK under atol=%g (max_abs over checked states <= atol)", atol)
        return 0

    k = min(int(args.max_report), len(bad))
    logger.warning(
        "[verify_rewardfit] phi mismatches above atol=%g: %d states (showing %d worst)",
        atol,
        len(bad),
        k,
    )
    for c in bad[:k]:
        if c.max_abs_delta is None:
            logger.warning(
                "  shard_%04d state=%d legal=%d top1_match=%s  phi: max=%g mean=%g",
                c.shard_id,
                c.state_idx,
                c.legal_count,
                str(bool(c.top1_match)),
                float(c.max_abs_phi),
                float(c.mean_abs_phi),
            )
        else:
            logger.warning(
                "  shard_%04d state=%d legal=%d top1_match=%s  phi: max=%g mean=%g  delta: max=%g mean=%g",
                c.shard_id,
                c.state_idx,
                c.legal_count,
                str(bool(c.top1_match)),
                float(c.max_abs_phi),
                float(c.mean_abs_phi),
                float(c.max_abs_delta),
                float(c.mean_abs_delta),
            )

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
