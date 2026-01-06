# src/tetris_rl/rewardfit/api.py
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from tetris_rl.datagen.shard_reader import ShardDataset
from tetris_rl.rewardfit.collect import Collected, collect_xy_from_dataset
from tetris_rl.rewardfit.models import fit_models, get_adapter_for_fit
from tetris_rl.rewardfit.progress import RichProgress
from tetris_rl.rewardfit.report import log_fit
from tetris_rl.rewardfit.types import FitResult, RewardFitConfig, RewardFitOutput
from tetris_rl.utils.seed import splitmix64_np


def _extra_float(extra: Mapping[str, Any], key: str) -> float:
    v = extra.get(key, None)
    if isinstance(v, (int, float, np.floating)):
        return float(v)
    return float("nan")


def _normalize_fit(
        *,
        fit: FitResult,
        X_base: np.ndarray,
        kind: str,
        base_feature_dim: int,
) -> Tuple[FitResult, float]:
    k = str(kind).lower().strip()
    w = np.asarray(fit.coef, dtype=np.float64).reshape(-1)
    b = float(fit.bias)

    scale = 1.0
    eps = 1e-12

    if k == "none":
        scale = 1.0
    elif k == "maxabs":
        denom = float(np.max(np.abs(w))) if w.size else 0.0
        scale = 1.0 if denom <= eps else (1.0 / denom)
    elif k == "l1":
        denom = float(np.sum(np.abs(w))) if w.size else 0.0
        scale = 1.0 if denom <= eps else (1.0 / denom)
    elif k == "std":
        ad = get_adapter_for_fit(fit=fit, fallback_kind="linear")
        if ad.predict is not None:
            r = ad.predict(np.asarray(X_base, dtype=np.float64), fit, int(base_feature_dim))
        else:
            r = (np.asarray(X_base, dtype=np.float64) @ w) + b
        denom = float(np.std(r))
        scale = 1.0 if denom <= eps else (1.0 / denom)
    else:
        raise ValueError(f"unknown normalize kind: {kind!r}")

    w2 = (w * float(scale)).astype(np.float64, copy=False)
    b2 = float(b * float(scale))

    ad2 = get_adapter_for_fit(fit=fit, fallback_kind="linear")
    extra2 = dict(fit.extra)
    if ad2.scale_extra is not None:
        extra2 = ad2.scale_extra(extra2, float(scale))

    name = fit.name if k == "none" else f"{fit.name} | normalized={k}"
    return (
        FitResult(
            name=name,
            coef=w2,
            bias=b2,
            r2=float(fit.r2),
            feature_names=list(fit.feature_names),
            extra=extra2,
        ),
        float(scale),
    )


def _resolve_feature_selection(
        *,
        base_feature_names: list[str],
        include: Optional[Sequence[str]],
        drop: Optional[Sequence[str]],
) -> Tuple[list[str], np.ndarray]:
    name_to_idx = {str(n): int(i) for i, n in enumerate(base_feature_names)}

    inc: Optional[list[int]] = None
    if include is not None:
        inc_list = [str(x).strip() for x in include if str(x).strip()]
        if inc_list:
            bad = [x for x in inc_list if x not in name_to_idx]
            if bad:
                raise ValueError(f"unknown --features entries: {bad!r}. available: {base_feature_names!r}")
            inc = [name_to_idx[x] for x in inc_list]

    drop_list = [str(x).strip() for x in (drop or []) if str(x).strip()]
    drop_idx: set[int] = set()
    if drop_list:
        bad = [x for x in drop_list if x not in name_to_idx]
        if bad:
            raise ValueError(f"unknown --drop-features entries: {bad!r}. available: {base_feature_names!r}")
        drop_idx = {name_to_idx[x] for x in drop_list}

    if inc is None:
        idx = [i for i in range(len(base_feature_names)) if i not in drop_idx]
    else:
        idx = [i for i in inc if i not in drop_idx]

    if not idx:
        raise ValueError("feature selection produced 0 features (check --features/--drop-features)")

    idx_arr = np.asarray(idx, dtype=np.int64)
    sel_names = [base_feature_names[int(i)] for i in idx_arr.tolist()]
    return sel_names, idx_arr


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.size == 0:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - (ss_res / ss_tot)


def _split_rows_by_state(
        *,
        keys: np.ndarray,  # (R,2) int64
        seed: int,
        eval_frac: float,
        test_frac: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not (0.0 <= float(eval_frac) < 1.0):
        raise ValueError(f"eval_frac must be in [0,1), got {eval_frac}")
    if not (0.0 <= float(test_frac) < 1.0):
        raise ValueError(f"test_frac must be in [0,1), got {test_frac}")
    if float(eval_frac) + float(test_frac) >= 1.0:
        raise ValueError("eval_frac + test_frac must be < 1.0")

    keys = np.asarray(keys, dtype=np.int64).reshape(-1, 2)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)

    sid = uniq[:, 0].astype(np.uint64)
    si = uniq[:, 1].astype(np.uint64)
    s = np.uint64(int(seed) & 0xFFFFFFFF)

    x = (sid << np.uint64(32)) ^ si ^ (s << np.uint64(1))
    h = splitmix64_np(x)
    u = h.astype(np.float64) / 18446744073709551616.0  # 2**64

    t = float(test_frac)
    e = float(eval_frac)

    state_is_test = u < t
    state_is_eval = (u >= t) & (u < (t + e))
    state_is_train = ~(state_is_test | state_is_eval)

    row_train = state_is_train[inv]
    row_eval = state_is_eval[inv]
    row_test = state_is_test[inv]
    return row_train, row_eval, row_test


def fit_from_dataset(*, dataset_dir, cfg: RewardFitConfig, logger: Any, use_rich: bool) -> RewardFitOutput:
    if not (float(cfg.tau) > 0.0):
        raise ValueError(f"tau must be > 0, got {cfg.tau}")
    if int(cfg.topk_actions) < 0:
        raise ValueError(f"topk_actions must be >= 0, got {cfg.topk_actions}")
    if int(cfg.max_states) < 0:
        raise ValueError(f"max_states must be >= 0, got {cfg.max_states}")
    if int(cfg.max_rows) < 0:
        raise ValueError(f"max_rows must be >= 0, got {cfg.max_rows}")

    ds = ShardDataset(dataset_dir=dataset_dir)

    base_feature_names = list(getattr(ds.manifest, "feature_names", []) or [])
    if not base_feature_names:
        raise RuntimeError("manifest.feature_names missing/empty (did you record_rewardfit=true?)")

    if len(ds.shard_ids()) == 0:
        raise RuntimeError("dataset has 0 shards registered in manifest.json (datagen still running?)")

    selected_feature_names, selected_feature_idx = _resolve_feature_selection(
        base_feature_names=base_feature_names,
        include=cfg.features,
        drop=cfg.drop_features,
    )

    progress_mode = str(cfg.progress).lower().strip()
    enable_progress = bool(use_rich) and progress_mode in {"shards", "states"}

    shard_ids = None if cfg.shards is None else [int(x) for x in cfg.shards]

    def _collect(*, progress_cb: Any) -> Collected:
        return collect_xy_from_dataset(
            ds=ds,
            feature_names=base_feature_names,  # full list for validation / error messages
            feature_idx=selected_feature_idx.tolist(),  # selection applied inside collector
            tau=float(cfg.tau),
            topk=int(cfg.topk_actions),
            max_states=int(cfg.max_states),
            max_rows=int(cfg.max_rows),
            seed=int(cfg.seed),
            shard_ids=shard_ids,
            progress_cb=progress_cb,
            logger=logger,
        )

    if enable_progress:
        with RichProgress(enabled=True, mode=progress_mode) as pcb:
            collected = _collect(progress_cb=pcb)
    else:
        collected = _collect(progress_cb=None)

    X = np.asarray(collected.X, dtype=np.float32)
    y = np.asarray(collected.y, dtype=np.float32)

    logger.info(
        "[fit_reward] states_used=%d rows_used=%d features=%d",
        int(collected.states_used),
        int(collected.rows_used),
        int(X.shape[1]),
    )
    if selected_feature_names != base_feature_names:
        logger.info(
            "[fit_reward] feature_selection: %d/%d features",
            int(len(selected_feature_names)),
            int(len(base_feature_names)),
        )
        logger.info("[fit_reward] features: %s", ", ".join(selected_feature_names))

    split_kind = str(getattr(cfg, "split", "state")).lower().strip()
    eval_frac = float(getattr(cfg, "eval_frac", 0.10))
    test_frac = float(getattr(cfg, "test_frac", 0.0))

    if split_kind == "none" or (eval_frac <= 0.0 and test_frac <= 0.0):
        row_train = np.ones((X.shape[0],), dtype=bool)
        row_eval = np.zeros((X.shape[0],), dtype=bool)
        row_test = np.zeros((X.shape[0],), dtype=bool)
    elif split_kind == "state":
        row_train, row_eval, row_test = _split_rows_by_state(
            keys=collected.row_state_keys,
            seed=int(cfg.seed),
            eval_frac=eval_frac,
            test_frac=test_frac,
        )
    elif split_kind == "shard":
        sid = np.asarray(collected.row_state_keys[:, 0], dtype=np.int64)
        uniq = np.unique(sid)
        rng = np.random.default_rng(int(cfg.seed))
        rng.shuffle(uniq)

        n = int(uniq.size)
        n_test = int(round(float(test_frac) * n))
        n_eval = int(round(float(eval_frac) * n))

        test_shards = uniq[:n_test]
        eval_shards = uniq[n_test: n_test + n_eval]

        row_test = np.isin(sid, test_shards)
        row_eval = np.isin(sid, eval_shards)
        row_train = ~(row_test | row_eval)
    else:
        raise ValueError(f"unknown split kind: {split_kind!r}")

    X_train, y_train = X[row_train], y[row_train]
    X_eval, y_eval = X[row_eval], y[row_eval]
    X_test, y_test = X[row_test], y[row_test]

    logger.info(
        "[fit_reward] split=%s train=%d eval=%d test=%d",
        split_kind,
        int(X_train.shape[0]),
        int(X_eval.shape[0]),
        int(X_test.shape[0]),
    )

    # IMPORTANT: forward cfg.fit_intercept so --no-bias actually takes effect.
    fits = fit_models(
        X_train,
        y_train,
        logger=logger,
        which=str(cfg.model),
        feature_names=selected_feature_names,
        fit_intercept=bool(getattr(cfg, "fit_intercept", True)),
    )
    base_feature_dim = int(X_train.shape[1])

    def add_split_scores(fr: FitResult) -> FitResult:
        ad = get_adapter_for_fit(fit=fr, fallback_kind="linear")
        if ad.predict is None:
            return fr

        extra = dict(fr.extra)

        if X_eval.shape[0] > 0:
            yhat = ad.predict(np.asarray(X_eval, dtype=np.float64), fr, int(base_feature_dim))
            extra["r2_eval"] = float(_r2_score(y_eval, yhat))

        if X_test.shape[0] > 0:
            yhat = ad.predict(np.asarray(X_test, dtype=np.float64), fr, int(base_feature_dim))
            extra["r2_test"] = float(_r2_score(y_test, yhat))

        return FitResult(
            name=fr.name,
            coef=fr.coef,
            bias=float(fr.bias),
            r2=float(fr.r2),
            feature_names=list(fr.feature_names),
            extra=extra,
        )

    fits2 = [add_split_scores(fr) for fr in fits]

    def sort_key(fr: FitResult) -> float:
        r2ev = _extra_float(fr.extra, "r2_eval")
        return r2ev if np.isfinite(r2ev) else float(fr.r2)

    fits_sorted = sorted(fits2, key=sort_key, reverse=True)

    if len(fits_sorted) > 1:
        logger.info("")
        logger.info("[fit_reward] model comparison (R²):")
        for fr in fits_sorted:
            r2tr = float(fr.r2)
            r2ev = _extra_float(fr.extra, "r2_eval")
            if np.isfinite(r2ev):
                logger.info("  eval=%8.4f train=%8.4f  %s", float(r2ev), r2tr, fr.name)
            else:
                logger.info("  train=%8.4f  %s", r2tr, fr.name)

    best_raw = fits_sorted[0]

    norm_kind = str(getattr(cfg, "normalize", "std")).lower().strip()
    best, scale = _normalize_fit(
        fit=best_raw,
        X_base=X_train.astype(np.float64, copy=False),
        kind=norm_kind,
        base_feature_dim=int(X_train.shape[1]),
    )

    if norm_kind != "none":
        logger.info("[fit_reward] normalization=%s scale=%0.6g", norm_kind, float(scale))

    log_fit(
        logger=logger,
        feature_names=list(best.feature_names) if best.feature_names else list(selected_feature_names),
        fit=best,
        print_weights=bool(cfg.print_weights),
        print_snippet=bool(cfg.print_snippet),
        sort_weights=str(cfg.sort_weights),
    )

    save_dict: Dict[str, object] = {
        "X": X,
        "y": y,
        "row_state_keys": np.asarray(collected.row_state_keys, dtype=np.int64),
        "feature_names": np.asarray(base_feature_names, dtype=object),
        "selected_feature_names": np.asarray(selected_feature_names, dtype=object),
        "selected_feature_idx": np.asarray(selected_feature_idx, dtype=np.int64),
        "tau": np.asarray([float(cfg.tau)], dtype=np.float32),
        "topk_actions": np.asarray([int(cfg.topk_actions)], dtype=np.int64),
        "seed": np.asarray([int(cfg.seed)], dtype=np.int64),
        "max_states": np.asarray([int(cfg.max_states)], dtype=np.int64),
        "max_rows": np.asarray([int(cfg.max_rows)], dtype=np.int64),
        "rows_used": np.asarray([int(collected.rows_used)], dtype=np.int64),
        "states_used": np.asarray([int(collected.states_used)], dtype=np.int64),
        "split": np.asarray([split_kind], dtype=object),
        "eval_frac": np.asarray([float(eval_frac)], dtype=np.float64),
        "test_frac": np.asarray([float(test_frac)], dtype=np.float64),
        "row_is_train": np.asarray(row_train, dtype=bool),
        "row_is_eval": np.asarray(row_eval, dtype=bool),
        "row_is_test": np.asarray(row_test, dtype=bool),
        "selected_model": np.asarray([str(cfg.model)], dtype=object),
        "dataset_dir": np.asarray([str(dataset_dir)], dtype=object),
        "shards": np.asarray([-1] if cfg.shards is None else [int(x) for x in cfg.shards], dtype=np.int64),
        "normalize": np.asarray([norm_kind], dtype=object),
        "scale": np.asarray([float(scale)], dtype=np.float64),
        "fit_intercept": np.asarray([bool(getattr(cfg, "fit_intercept", True))], dtype=bool),
        "best_name_raw": np.asarray([best_raw.name], dtype=object),
        "best_name_scaled": np.asarray([best.name], dtype=object),
        "best_r2_train": np.asarray([float(best_raw.r2)], dtype=np.float64),
        "best_r2_eval": np.asarray([_extra_float(best_raw.extra, "r2_eval")], dtype=np.float64),
        "best_r2_test": np.asarray([_extra_float(best_raw.extra, "r2_test")], dtype=np.float64),
        "best_feature_names": np.asarray(
            list(best.feature_names) if best.feature_names else list(selected_feature_names),
            dtype=object,
        ),
        "best_coef_raw": np.asarray(best_raw.coef, dtype=np.float64),
        "best_bias_raw": np.asarray([float(best_raw.bias)], dtype=np.float64),
        "best_coef_scaled": np.asarray(best.coef, dtype=np.float64),
        "best_bias_scaled": np.asarray([float(best.bias)], dtype=np.float64),
    }

    if best_raw.extra:
        ad = get_adapter_for_fit(fit=best_raw, fallback_kind=str(cfg.model))
        if ad.save_extra is not None:
            ad.save_extra(save_dict, best_raw, best)
        else:
            save_dict["best_extra_kind"] = np.asarray([str(best_raw.extra.get("kind", ""))], dtype=object)

    return RewardFitOutput(
        best=best,
        fits_sorted=fits_sorted,
        feature_names=base_feature_names,
        X=X,
        y=y,
        states_used=int(collected.states_used),
        rows_used=int(collected.rows_used),
        save_dict=save_dict,
    )
