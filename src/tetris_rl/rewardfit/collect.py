# src/tetris_rl/rewardfit/collect.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from tetris_rl.datagen.shard_reader import ShardDataset
from tetris_rl.utils.seed import seed32_from


@dataclass(frozen=True)
class Collected:
    X: np.ndarray  # (R, D)
    y: np.ndarray  # (R,)
    row_state_keys: np.ndarray  # (R, 2) int64: [shard_id, state_index] per row

    states_used: int
    rows_used: int
    shards_seen: int
    shards_used: int
    shards_missing_rewardfit: int
    states_no_legal: int

    # diagnostics (helps debug “R²=0 / MSE huge” issues)
    states_phi_all_nonfinite: int
    rows_dropped_nonfinite_phi: int


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return x

    if not np.isfinite(x).any():
        return np.full_like(x, -np.inf, dtype=np.float64)

    xf = np.where(np.isfinite(x), x, -np.inf)

    m = float(np.max(xf))
    z = xf - m
    lse = m + float(np.log(np.sum(np.exp(z))))
    return xf - lse


def _collect_one_shard(
        *,
        ds: ShardDataset,
        shard_id: int,
        feature_dim: int,
        feature_idx: Optional[np.ndarray],
        tau: float,
        topk: int,
        max_states_local: int,
        max_rows_local: int,
        seed: int,
) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], int, int, int, int, int]:
    arrays = ds.get_shard(int(shard_id))

    if ("phi" not in arrays) or ("delta" not in arrays) or ("legal_mask" not in arrays):
        return int(shard_id), None, None, None, 0, 0, 0, 0, 0

    phi = np.asarray(arrays["phi"], dtype=np.float64)  # (N, A)
    delta = np.asarray(arrays["delta"], dtype=np.float32)  # (N, A, F_full)
    legal = np.asarray(arrays["legal_mask"], dtype=bool)  # (N, A)

    if phi.ndim != 2 or legal.ndim != 2 or delta.ndim != 3:
        raise RuntimeError(
            f"bad shapes in shard_{int(shard_id):04d}: phi{phi.shape} legal{legal.shape} delta{delta.shape}"
        )
    if phi.shape != legal.shape:
        raise RuntimeError(f"phi/legal mismatch in shard_{int(shard_id):04d}: {phi.shape} vs {legal.shape}")
    if delta.shape[0] != phi.shape[0] or delta.shape[1] != phi.shape[1]:
        raise RuntimeError(f"delta mismatch in shard_{int(shard_id):04d}: {delta.shape} vs {phi.shape}")

    if feature_idx is not None:
        feature_idx = np.asarray(feature_idx, dtype=np.int64).reshape(-1)
        if feature_idx.size != int(feature_dim):
            raise RuntimeError(f"feature_idx size mismatch: got {int(feature_idx.size)} expected {int(feature_dim)}")
        delta = delta[:, :, feature_idx]
    else:
        if int(delta.shape[2]) != int(feature_dim):
            raise RuntimeError(
                f"feature dim mismatch in shard_{int(shard_id):04d}: "
                f"delta F={int(delta.shape[2])} expected F={int(feature_dim)}"
            )

    N, _A = phi.shape
    rng = np.random.default_rng(int(seed))
    state_indices = np.arange(N, dtype=np.int64)

    if max_states_local and N > max_states_local:
        state_indices = rng.choice(state_indices, size=int(max_states_local), replace=False)
        state_indices = np.asarray(state_indices, dtype=np.int64)

    X_rows: List[np.ndarray] = []
    y_rows: List[float] = []
    key_rows: List[Tuple[int, int]] = []

    used_states = 0
    used_rows = 0
    states_no_legal = 0

    states_phi_all_nonfinite = 0
    rows_dropped_nonfinite_phi = 0

    sid = int(shard_id)

    for i in state_indices:
        if max_states_local and used_states >= max_states_local:
            break
        if max_rows_local and used_rows >= max_rows_local:
            break

        si = int(i)

        mask = legal[si]
        cand = np.flatnonzero(mask)
        if cand.size == 0:
            states_no_legal += 1
            continue

        phi_i = np.asarray(phi[si, cand], dtype=np.float64)
        finite = np.isfinite(phi_i)

        if not finite.any():
            states_phi_all_nonfinite += 1
            continue

        if not finite.all():
            rows_dropped_nonfinite_phi += int((~finite).sum())
            cand = cand[finite]
            phi_i = phi_i[finite]

        if cand.size < 2:
            continue

        if topk and cand.size > topk:
            order = np.argsort(-phi_i)[: int(topk)]
            cand = cand[order]
            phi_i = phi_i[order]

        logp = _log_softmax(phi_i / float(tau))
        if (logp.size == 0) or (not np.isfinite(logp).all()):
            states_phi_all_nonfinite += 1
            continue

        logp_centered = logp - float(np.mean(logp))

        d_i = delta[si, cand, :]  # (K, D)
        for k in range(int(d_i.shape[0])):
            if max_rows_local and used_rows >= max_rows_local:
                break
            X_rows.append(np.asarray(d_i[k], dtype=np.float32))
            y_rows.append(float(logp_centered[k]))
            key_rows.append((sid, si))
            used_rows += 1

        used_states += 1

    if not X_rows:
        return (
            sid,
            np.empty((0, feature_dim), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0, 2), dtype=np.int64),
            int(used_states),
            0,
            int(states_no_legal),
            int(states_phi_all_nonfinite),
            int(rows_dropped_nonfinite_phi),
        )

    Xp = np.stack(X_rows, axis=0).astype(np.float32, copy=False)
    yp = np.asarray(y_rows, dtype=np.float32)
    kp = np.asarray(key_rows, dtype=np.int64).reshape(-1, 2)

    return (
        sid,
        Xp,
        yp,
        kp,
        int(used_states),
        int(Xp.shape[0]),
        int(states_no_legal),
        int(states_phi_all_nonfinite),
        int(rows_dropped_nonfinite_phi),
    )


def collect_xy_from_dataset(
        *,
        ds: ShardDataset,
        feature_names: Sequence[str],
        feature_idx: Optional[Sequence[int]],
        tau: float,
        topk: int,
        max_states: int,
        max_rows: int,
        seed: int,
        shard_ids: Optional[Sequence[int]],
        progress_cb: Any,
        logger: Any,
) -> Collected:
    if not (float(tau) > 0.0):
        raise ValueError(f"tau must be > 0, got {tau}")
    if int(topk) < 0:
        raise ValueError(f"topk must be >= 0, got {topk}")
    if int(max_states) < 0:
        raise ValueError(f"max_states must be >= 0, got {max_states}")
    if int(max_rows) < 0:
        raise ValueError(f"max_rows must be >= 0, got {max_rows}")

    wanted = None if shard_ids is None else {int(x) for x in shard_ids}
    refs = [r for r in ds.shards if (wanted is None or int(r.shard_id) in wanted)]
    refs.sort(key=lambda r: int(r.shard_id))

    total_shards = len(refs)
    if total_shards == 0:
        raise RuntimeError("no shards selected")

    def wrap_shards(it: Iterable[Any], total: int) -> Iterable[Any]:
        if progress_cb is None:
            return it
        return progress_cb.wrap_shards(it, total=total)

    feature_idx_arr: Optional[np.ndarray]
    if feature_idx is None:
        feature_idx_arr = None
        feature_dim = int(len(feature_names))
    else:
        feature_idx_arr = np.asarray(list(feature_idx), dtype=np.int64).reshape(-1)
        feature_dim = int(feature_idx_arr.size)

    states_left = int(max_states) if int(max_states) else 0
    rows_left = int(max_rows) if int(max_rows) else 0

    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    k_parts: List[np.ndarray] = []

    used_states = 0
    used_rows = 0
    shards_seen = 0
    shards_used = 0
    shards_missing_rf = 0
    states_no_legal = 0

    states_phi_all_nonfinite = 0
    rows_dropped_nonfinite_phi = 0

    for ref in wrap_shards(refs, total=total_shards):
        sid = int(ref.shard_id)
        shards_seen += 1

        max_states_local = int(states_left) if states_left else 0
        max_rows_local = int(rows_left) if rows_left else 0
        shard_seed = seed32_from(base_seed=int(seed), stream_id=int(sid))

        (
            _sid,
            Xp,
            yp,
            kp,
            s_used,
            r_used,
            s_no_legal,
            s_phi_bad,
            r_drop_nf,
        ) = _collect_one_shard(
            ds=ds,
            shard_id=sid,
            feature_dim=feature_dim,
            feature_idx=feature_idx_arr,
            tau=float(tau),
            topk=int(topk),
            max_states_local=max_states_local,
            max_rows_local=max_rows_local,
            seed=int(shard_seed),
        )

        states_phi_all_nonfinite += int(s_phi_bad)
        rows_dropped_nonfinite_phi += int(r_drop_nf)

        if Xp is None or yp is None or kp is None:
            shards_missing_rf += 1
            continue

        if int(r_used) > 0:
            shards_used += 1
            X_parts.append(Xp)
            y_parts.append(yp)
            k_parts.append(kp)

        used_states += int(s_used)
        used_rows += int(r_used)
        states_no_legal += int(s_no_legal)

        if states_left:
            states_left = max(0, states_left - int(s_used))
        if rows_left:
            rows_left = max(0, rows_left - int(r_used))

        if (max_states and used_states >= max_states) or (max_rows and used_rows >= max_rows):
            break

    if not X_parts:
        msg = (
            "no rows collected; nothing to fit.\n"
            f"shards_seen={shards_seen} shards_used={shards_used} "
            f"shards_missing_rewardfit={shards_missing_rf} states_no_legal={states_no_legal}\n"
            f"states_phi_all_nonfinite={states_phi_all_nonfinite} rows_dropped_nonfinite_phi={rows_dropped_nonfinite_phi}\n"
            "hint: dataset must be generated with generation.labels.record_rewardfit=true\n"
        )
        raise RuntimeError(msg.rstrip())

    X = np.concatenate(X_parts, axis=0).astype(np.float32, copy=False)
    y = np.concatenate(y_parts, axis=0).astype(np.float32, copy=False)
    keys = np.concatenate(k_parts, axis=0).astype(np.int64, copy=False)

    if int(max_rows) and X.shape[0] > int(max_rows):
        X = X[: int(max_rows)]
        y = y[: int(max_rows)]
        keys = keys[: int(max_rows)]

    try:
        logger.info(
            "[fit_reward] phi_sanity: states_phi_all_nonfinite=%d rows_dropped_nonfinite_phi=%d",
            int(states_phi_all_nonfinite),
            int(rows_dropped_nonfinite_phi),
        )
    except Exception:
        pass

    return Collected(
        X=X,
        y=y,
        row_state_keys=keys,
        states_used=int(used_states),
        rows_used=int(X.shape[0]),
        shards_seen=int(shards_seen),
        shards_used=int(shards_used),
        shards_missing_rewardfit=int(shards_missing_rf),
        states_no_legal=int(states_no_legal),
        states_phi_all_nonfinite=int(states_phi_all_nonfinite),
        rows_dropped_nonfinite_phi=int(rows_dropped_nonfinite_phi),
    )
