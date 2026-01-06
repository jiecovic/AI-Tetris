# src/tetris_rl/datagen/worker.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np

from tetris_rl.config.datagen_spec import DataGenSpec
from tetris_rl.config.instantiate import instantiate
from tetris_rl.datagen.interleave_noise import InterleaveNoiseSpec, maybe_interleave_noise
from tetris_rl.datagen.schema import FEATURE_NAMES, ShardInfo
from tetris_rl.envs.api import WarmupFn
from tetris_rl.envs.catalog import WARMUP_REGISTRY
from tetris_rl.game.core.macro_legality import discrete_action_mask, macro_illegal_reason_bbox_left
from tetris_rl.game.core.macro_step import (
    apply_discrete_action_id_no_reward,
    apply_discrete_action_id_no_reward_with_diag,
    decode_discrete_action_id,
)
from tetris_rl.game.core.placement_cache import StaticPlacementCache
from tetris_rl.utils.seed import seed32_from


@dataclass(frozen=True)
class WorkerResult:
    worker_id: int
    shards: list[ShardInfo]
    shards_written: int
    samples_written: int


# Injected once per spawned worker process via ProcessPoolExecutor(initializer=..., initargs=...)
_WORKER_PROGRESS_QUEUE: Any = None


def _set_worker_progress_queue(q: Any) -> None:
    global _WORKER_PROGRESS_QUEUE
    _WORKER_PROGRESS_QUEUE = q


def _get_progress_queue(q: Any) -> Any:
    return q if q is not None else _WORKER_PROGRESS_QUEUE


def _q_put(q: Any, ev: tuple[Any, ...]) -> None:
    if q is None:
        return
    try:
        q.put_nowait(ev)
    except Exception:
        pass


# -----------------------------------------------------------------------------
# small helpers
# -----------------------------------------------------------------------------

def _apply_action_and_get_terminated(
        *,
        game: Any,
        legal_cache: StaticPlacementCache,
        action_id: int,
        board_w: int,
) -> bool:
    _state, _cleared, terminated = apply_discrete_action_id_no_reward(
        game=game,
        legal_cache=legal_cache,
        action_id=int(action_id),
        board_w=int(board_w),
    )
    return bool(terminated)


def _extract_sample_from_state(*, state: Any) -> tuple[np.ndarray, int, int]:
    # Copy: dataset stores locked board snapshot; engine owns authoritative grid.
    grid = np.asarray(state.grid, dtype=np.uint8).copy()
    ak = int(state.active_kind_idx)
    nk = int(state.next_kind_idx)
    return grid, ak, nk


def _mask_for_game(*, game: Any, legal_cache: StaticPlacementCache) -> np.ndarray:
    ap = game.active
    return (
        discrete_action_mask(
            board=game.board,
            pieces=game.pieces,
            cache=legal_cache,
            kind=str(ap.kind),
            py=int(ap.y),
        )
        .astype(bool, copy=False)
        .reshape(-1)
    )


def _raise_illegal_action_diag(
        *,
        game: Any,
        legal_cache: StaticPlacementCache,
        aid: int,
        action_dim: int,
) -> None:
    bw = int(legal_cache.board_w)
    rot, col = decode_discrete_action_id(action_id=int(aid), board_w=bw)
    ap = game.active

    reason = macro_illegal_reason_bbox_left(
        board=game.board,
        pieces=game.pieces,
        cache=legal_cache,
        kind=str(ap.kind),
        rot=int(rot),
        py=int(ap.y),
        bbox_left_col=int(col),
    )

    raise RuntimeError(
        "[datagen] expert chose illegal action\n"
        f"  aid={aid} decoded=(rot={rot}, col={col}) action_dim={action_dim}\n"
        f"  illegal_reason={reason!r}"
    )


# -----------------------------------------------------------------------------
# worker entrypoint
# -----------------------------------------------------------------------------

def worker_generate_shards(
        *,
        worker_id: int,
        shard_ids: list[int],
        spec: DataGenSpec,
        dataset_dir: str,
        progress_queue: Any = None,
        progress_every: int = 0,
) -> WorkerResult:
    """
    Generate a list of shards inside one worker process.

    Key contracts:
      - active_kind_idx / next_kind_idx are strict kind indices in [0..K-1]
      - legality masks are computed via the same macro legality logic as the env
      - warmup is a post-reset mutator hook: warmup(game=<game>, rng=<rng>) -> None
    """
    from tetris_rl.datagen.expert_factory import make_expert_from_spec
    from tetris_rl.datagen.writer import ShardWriter
    from tetris_rl.game.factory import make_game_from_spec

    # -------------------------------------------------------------------------
    # construct game + derived geometry
    # -------------------------------------------------------------------------
    out_dir = Path(dataset_dir)
    game = make_game_from_spec(spec.game)

    pieces = game.pieces
    num_kinds = int(len(list(pieces.kinds())))
    board_h = int(game.h)
    board_w = int(game.w)

    legal_cache = StaticPlacementCache.build(
        pieces=pieces,
        board_w=board_w,
    )

    try:
        max_rots = int(pieces.max_rotations())
    except Exception as e:
        raise RuntimeError("pieces.max_rotations() failed; PieceSet must define this") from e
    if max_rots <= 0:
        raise RuntimeError(f"invalid max_rots derived from pieces: {max_rots}")

    action_dim = int(max_rots * board_w)

    # Expert (policy used to pick actions and optionally emit reward-fit labels)
    expert = make_expert_from_spec(
        game=game,
        expert_spec=spec.expert,
    )

    record_rewardfit = bool(spec.generation.labels.record_rewardfit)

    # -------------------------------------------------------------------------
    # warmup (component spec) — type it as WarmupFn|None so narrowing works
    # -------------------------------------------------------------------------
    warmup_fn: WarmupFn | None = None
    if spec.generation.warmup is not None:
        obj = instantiate(
            spec_obj={"type": spec.generation.warmup.type, "params": dict(spec.generation.warmup.params)},
            registry=WARMUP_REGISTRY,
            where="cfg.generation.warmup",
            injected={},
        )
        if obj is None:
            raise RuntimeError(
                "[datagen] warmup instantiate() returned None. "
                "Warmup registry constructors must return a callable WarmupFn object."
            )
        if not callable(obj):
            raise TypeError(
                f"[datagen] warmup object is not callable: type={type(obj).__name__}. "
                "Warmup components must be callable procedures."
            )
        warmup_fn = cast(WarmupFn, obj)

    def do_warmup(*, rng: np.random.Generator) -> None:
        w = warmup_fn
        if w is None:
            return
        # IMPORTANT: warmups use keyword-only __call__(*, game, rng)
        ret = w(game=game, rng=rng)
        if ret is not None:
            raise TypeError(
                f"[datagen] warmup {type(w).__name__} returned {type(ret).__name__}; expected None. "
                "Warmup components must be procedures that mutate the game and return None."
            )

    # -------------------------------------------------------------------------
    # interleave noise (episode-time perturbation)
    # -------------------------------------------------------------------------
    interleave_spec = InterleaveNoiseSpec(
        enabled=bool(spec.generation.noise.enabled),
        interleave_prob=spec.generation.noise.interleave_prob,
        interleave_max_steps=spec.generation.noise.interleave_max_steps,
        require_masks=True,
    )

    # -------------------------------------------------------------------------
    # generation loop
    # -------------------------------------------------------------------------
    shard_infos: list[ShardInfo] = []
    shards_written = 0
    samples_written = 0

    k = int(progress_every) if progress_every is not None else 0
    if k < 0:
        k = 0

    for sid in shard_ids:
        s32 = seed32_from(base_seed=spec.run.seed, stream_id=sid)
        rng = np.random.default_rng(s32)

        def do_reset() -> None:
            game.reset()

        do_reset()
        do_warmup(rng=rng)
        ep_steps = 0

        N = int(spec.dataset.shards.shard_steps)
        if N <= 0:
            raise ValueError(f"invalid shard_steps: {N}")

        pq = _get_progress_queue(progress_queue)
        _q_put(pq, ("start", int(worker_id), int(sid), int(N)))

        # Buffers (we fill sequentially via `filled`; never leave uninitialized rows)
        grid_buf = np.empty((N, board_h, board_w), dtype=np.uint8)
        ak_buf = np.empty((N,), dtype=np.uint8)
        nk_buf = np.empty((N,), dtype=np.uint8)
        act_buf = np.empty((N,), dtype=np.int64)

        # New per-step engine features (always recorded)
        placed_cells_cleared_buf = np.empty((N,), dtype=np.uint8)  # 0..4
        placed_cells_all_cleared_buf = np.empty((N,), dtype=bool)

        mask_buf: Optional[np.ndarray] = None
        phi_buf: Optional[np.ndarray] = None
        delta_buf: Optional[np.ndarray] = None

        if record_rewardfit:
            mask_buf = np.empty((N, action_dim), dtype=bool)
            phi_buf = np.empty((N, action_dim), dtype=np.float32)
            delta_buf = np.empty((N, action_dim, len(FEATURE_NAMES)), dtype=np.float32)

        filled = 0
        attempts = 0

        # Bound rejections so "always-reject" bugs don't spin forever.
        max_attempts = int(N * 50)

        while filled < N:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    f"[datagen] too many rejected samples for shard {sid}: "
                    f"filled={filled}/{N} attempts={attempts} "
                    f"(check masks / expert / rewardfit filtering)"
                )

            # Optional random perturbation step(s) between recorded samples
            maybe_interleave_noise(
                rng=rng,
                spec=interleave_spec,
                action_dim=action_dim,
                get_mask=lambda: _mask_for_game(game=game, legal_cache=legal_cache),
                apply_action_no_record=lambda aid: _apply_action_and_get_terminated(
                    game=game,
                    legal_cache=legal_cache,
                    action_id=int(aid),
                    board_w=board_w,
                ),
                do_reset=do_reset,
            )

            st = game.state()
            grid, ak, nk = _extract_sample_from_state(state=st)

            # If this ever fires, the bug is in engine/state construction.
            if not (0 <= ak < num_kinds and 0 <= nk < num_kinds):
                raise RuntimeError(f"[datagen] kind_idx out of range at source: ak={ak} nk={nk} K={num_kinds}")

            mask = _mask_for_game(game=game, legal_cache=legal_cache)
            candidates = np.flatnonzero(mask).tolist()

            # If there are no legal macro placements, treat as terminal-like and restart.
            if not candidates:
                do_reset()
                do_warmup(rng=rng)
                ep_steps = 0
                continue

            # Pick expert action (and optionally compute reward-fit labels)
            if record_rewardfit:
                phi, delta = expert.evaluate_action_set(st, candidates=candidates)

                cand_ids = np.asarray(candidates, dtype=np.int64)
                cand_phi = phi[cand_ids]
                finite = np.isfinite(cand_phi)

                # If expert assigns no finite score to any legal move, restart.
                if not bool(finite.any()):
                    do_reset()
                    do_warmup(rng=rng)
                    ep_steps = 0
                    continue

                best_local = int(np.argmax(cand_phi[finite]))
                aid = int(cand_ids[finite][best_local])

                assert mask_buf is not None and phi_buf is not None and delta_buf is not None
                mask_buf[filled] = mask
                phi_buf[filled] = phi
                delta_buf[filled] = delta
            else:
                aid = int(expert.best_action_id(st, candidates=candidates))

            # Sanity: expert must choose a legal action under the same mask
            if not mask[aid]:
                _raise_illegal_action_diag(
                    game=game,
                    legal_cache=legal_cache,
                    aid=aid,
                    action_dim=action_dim,
                )

            # Record sample (pre-action state)
            grid_buf[filled] = grid
            ak_buf[filled] = ak
            nk_buf[filled] = nk
            act_buf[filled] = aid

            # Apply chosen macro placement (engine emits new features via info_engine)
            r = apply_discrete_action_id_no_reward_with_diag(
                game=game,
                legal_cache=legal_cache,
                action_id=int(aid),
                board_w=int(board_w),
            )
            terminated = bool(r.terminated)

            info_engine = r.info_engine or {}
            try:
                placed_cells_cleared_buf[filled] = np.uint8(int(info_engine.get("placed_cells_cleared", 0)))
            except Exception:
                placed_cells_cleared_buf[filled] = np.uint8(0)
            placed_cells_all_cleared_buf[filled] = bool(info_engine.get("placed_cells_all_cleared", False))

            ep_steps += 1
            if terminated or (
                    spec.generation.episode_max_steps is not None and ep_steps >= spec.generation.episode_max_steps
            ):
                do_reset()
                do_warmup(rng=rng)
                ep_steps = 0

            filled += 1

            if k and (filled % k == 0):
                _q_put(pq, ("progress", int(worker_id), int(sid), int(filled), int(N)))

        _q_put(pq, ("progress", int(worker_id), int(sid), int(N), int(N)))

        # ---------------------------------------------------------------------
        # write shard
        # ---------------------------------------------------------------------
        writer = ShardWriter(
            dataset_dir=out_dir,
            shard_id=sid,
            compression=spec.dataset.compression,
            board_h=board_h,
            board_w=board_w,
            num_kinds=num_kinds,
            action_dim=action_dim,
            episode_max_steps=spec.generation.episode_max_steps,
            seed=s32,
        )

        info = writer.write(
            grid=grid_buf,
            active_kind=ak_buf,
            next_kind=nk_buf,
            action=act_buf,
            placed_cells_cleared=placed_cells_cleared_buf,
            placed_cells_all_cleared=placed_cells_all_cleared_buf,
            legal_mask=mask_buf,
            phi=phi_buf,
            delta=delta_buf,
            feature_names=list(FEATURE_NAMES) if record_rewardfit else None,
        )

        shard_infos.append(info)
        shards_written += 1
        samples_written += N

        _q_put(pq, ("done", int(worker_id), int(sid), int(N)))

    return WorkerResult(
        worker_id=int(worker_id),
        shards=sorted(shard_infos, key=lambda s: s.shard_id),
        shards_written=int(shards_written),
        samples_written=int(samples_written),
    )
