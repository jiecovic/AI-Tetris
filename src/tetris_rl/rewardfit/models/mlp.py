# src/tetris_rl/rewardfit/models_old/mlp.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from tetris_rl.rewardfit.models.adapters import ModelAdapter
from tetris_rl.rewardfit.types import FitResult

# Public "kind" string used by registry/normalization/save hooks.
MLP_KIND = "mlp"


@dataclass(frozen=True)
class MLPSpec:
    """
    Small "neutral" MLP regressor over delta-features.

    Goal: capture mild non-linear interactions without turning this into a full DL project.

    Architecture:
      - input_dim = F
      - hidden = (h1, h2, ...)
      - activation = gelu or tanh
      - output_dim = 1

    Training:
      - AdamW with weight decay
      - few epochs, large batches
      - optional validation + early stopping
    """
    hidden: Tuple[int, ...] = (32, 32)
    activation: str = "gelu"  # "gelu" | "tanh"
    dropout: float = 0.0

    epochs: int = 6
    batch_size: int = 8192
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # val split for early stopping (0 disables)
    val_frac: float = 0.02
    patience: int = 2  # epochs without improvement before stopping (only if val enabled)

    # deterministic-ish shuffling
    seed: int = 0

    # compute device: "auto" uses cuda if available
    device: str = "auto"


def _device_from_spec(spec: MLPSpec) -> str:
    d = str(spec.device).lower().strip()
    if d in {"cpu", "cuda"}:
        return d
    # auto
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float32)
    mu = np.mean(X, axis=0).astype(np.float32, copy=False)
    sigma = np.std(X, axis=0).astype(np.float32, copy=False)
    sigma = np.where(sigma > 1e-12, sigma, np.float32(1.0))
    return mu, sigma


def _standardize_apply(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    mu = np.asarray(mu, dtype=np.float32).reshape(1, -1)
    sigma = np.asarray(sigma, dtype=np.float32).reshape(1, -1)
    return (X - mu) / sigma


def _build_mlp(*, in_dim: int, spec: MLPSpec):
    import torch
    from torch import nn

    act = str(spec.activation).lower().strip()
    if act == "tanh":
        Act = nn.Tanh
    else:
        # default: gelu
        Act = nn.GELU

    layers = []
    prev = int(in_dim)
    for h in tuple(int(x) for x in spec.hidden):
        layers.append(nn.Linear(prev, h))
        layers.append(Act())
        if float(spec.dropout) > 0.0:
            layers.append(nn.Dropout(p=float(spec.dropout)))
        prev = int(h)

    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def _to_state_dict_numpy(state_dict: Dict[str, Any]) -> Dict[str, object]:
    """
    Convert torch state_dict tensors to numpy for JSON/npz friendliness.
    """
    out: Dict[str, object] = {}
    for k, v in state_dict.items():
        try:
            import torch
            if isinstance(v, torch.Tensor):
                out[k] = v.detach().cpu().numpy()
            else:
                out[k] = v
        except Exception:
            out[k] = v
    return out


def _from_state_dict_numpy(sd_np: Dict[str, object]) -> Dict[str, Any]:
    """
    Convert numpy arrays back to torch tensors.
    """
    import torch

    out: Dict[str, Any] = {}
    for k, v in sd_np.items():
        if isinstance(v, np.ndarray):
            out[k] = torch.from_numpy(v)
        else:
            out[k] = v
    return out


def _split_train_val(
        X: np.ndarray,
        y: np.ndarray,
        *,
        val_frac: float,
        seed: int,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1)

    n = int(X.shape[0])
    frac = float(val_frac)
    if frac <= 0.0 or n < 10:
        return X, y, None, None

    n_val = int(max(1, round(frac * n)))
    n_val = min(n_val, n - 1)

    rng = np.random.default_rng(int(seed))
    idx = np.arange(n, dtype=np.int64)
    rng.shuffle(idx)

    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    Xtr = X[tr_idx]
    ytr = y[tr_idx]
    Xva = X[val_idx]
    yva = y[val_idx]
    return Xtr, ytr, Xva, yva


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.size == 0:
        return 0.0
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    return 0.0 if ss_tot <= 0 else (1.0 - ss_res / ss_tot)


def fit_mlp(
        X: np.ndarray,
        y: np.ndarray,
        *,
        feature_names: Sequence[str],
        logger: Any,
        spec: MLPSpec,
) -> FitResult:
    """
    Fit a small MLP on standardized X.

    Returns FitResult with:
      - coef empty (nonlinear)
      - bias stored separately as fit.bias (final layer bias is included in state)
      - fit.extra contains everything needed for prediction/export (x_mean/x_std/state_dict/spec)
    """
    try:
        import torch
        from torch import nn
    except Exception as e:
        raise RuntimeError("PyTorch is required for mlp model. Install with: pip install torch") from e

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X/y mismatch: {X.shape} vs {y.shape}")
    if X.shape[0] < 2:
        raise ValueError("need at least 2 rows to fit")

    n, f = X.shape
    if len(feature_names) != int(f):
        raise ValueError(f"feature_names mismatch: {len(feature_names)} vs F={int(f)}")

    # standardize features
    x_mean, x_std = _standardize_fit(X)
    Xs = _standardize_apply(X, x_mean, x_std)

    # train/val split (on standardized X)
    Xtr, ytr, Xva, yva = _split_train_val(
        Xs,
        y,
        val_frac=float(spec.val_frac),
        seed=int(spec.seed),
    )

    device = _device_from_spec(spec)
    torch.manual_seed(int(spec.seed))

    model = _build_mlp(in_dim=int(f), spec=spec).to(device=device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(spec.lr),
        weight_decay=float(spec.weight_decay),
    )
    loss_fn = nn.MSELoss()

    # data tensors
    Xtr_t = torch.from_numpy(np.asarray(Xtr, dtype=np.float32)).to(device=device)
    ytr_t = torch.from_numpy(np.asarray(ytr, dtype=np.float32).reshape(-1, 1)).to(device=device)

    Xva_t = None
    yva_t = None
    if Xva is not None and yva is not None:
        Xva_t = torch.from_numpy(np.asarray(Xva, dtype=np.float32)).to(device=device)
        yva_t = torch.from_numpy(np.asarray(yva, dtype=np.float32).reshape(-1, 1)).to(device=device)

    bs = int(max(1, spec.batch_size))
    epochs = int(max(1, spec.epochs))
    patience = int(max(0, spec.patience))

    best_val = None
    best_state = None
    bad_epochs = 0

    t0 = time.perf_counter()

    for ep in range(1, epochs + 1):
        model.train()
        # shuffle indices each epoch (deterministic via torch RNG seed above)
        perm = torch.randperm(Xtr_t.shape[0], device=device)

        total_loss = 0.0
        seen = 0

        for start in range(0, int(Xtr_t.shape[0]), bs):
            idx = perm[start: start + bs]
            xb = Xtr_t.index_select(0, idx)
            yb = ytr_t.index_select(0, idx)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += float(loss.detach().item()) * int(xb.shape[0])
            seen += int(xb.shape[0])

        train_mse = total_loss / max(1, seen)

        # validation
        val_mse = None
        if Xva_t is not None and yva_t is not None:
            model.eval()
            with torch.no_grad():
                pred = model(Xva_t)
                val_mse = float(loss_fn(pred, yva_t).detach().item())

            # early stopping bookkeeping
            if best_val is None or val_mse < best_val:
                best_val = float(val_mse)
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1

        if val_mse is None:
            logger.info("[fit_reward] mlp epoch %d/%d train_mse=%0.6g", ep, epochs, float(train_mse))
        else:
            logger.info(
                "[fit_reward] mlp epoch %d/%d train_mse=%0.6g val_mse=%0.6g",
                ep, epochs, float(train_mse), float(val_mse)
            )

        if val_mse is not None and patience > 0 and bad_epochs >= patience:
            logger.info("[fit_reward] mlp early stop (patience=%d)", patience)
            break

    # restore best state if we tracked it
    if best_state is not None:
        model.load_state_dict(best_state)

    dt = time.perf_counter() - t0

    # compute train R2 on full standardized X
    model.eval()
    with torch.no_grad():
        Xall_t = torch.from_numpy(np.asarray(Xs, dtype=np.float32)).to(device=device)
        pred = model(Xall_t).detach().cpu().numpy().reshape(-1)
    r2 = _r2_score(y, pred)

    logger.info(
        "[fit_reward] done MLP(hidden=%s,act=%s) in %0.2fs (train R²=%0.4f)",
        str(tuple(int(x) for x in spec.hidden)),
        str(spec.activation),
        dt,
        float(r2),
    )

    extra: Dict[str, object] = {
        "kind": MLP_KIND,
        "mlp_hidden": tuple(int(x) for x in spec.hidden),
        "mlp_activation": str(spec.activation),
        "mlp_dropout": float(spec.dropout),
        "mlp_epochs": int(spec.epochs),
        "mlp_batch_size": int(spec.batch_size),
        "mlp_lr": float(spec.lr),
        "mlp_weight_decay": float(spec.weight_decay),
        "mlp_val_frac": float(spec.val_frac),
        "mlp_patience": int(spec.patience),
        "mlp_seed": int(spec.seed),
        "mlp_device": str(device),
        "x_mean": np.asarray(x_mean, dtype=np.float32),
        "x_std": np.asarray(x_std, dtype=np.float32),
        "state_dict": _to_state_dict_numpy(model.state_dict()),
        "predictor": "mlp_torch_state_dict",
    }

    # coef/bias: keep coef empty; bias is not used (state_dict includes biases),
    # but bias field must exist; keep 0.0 to match centered targets.
    return FitResult(
        name=f"MLP(hidden={tuple(int(x) for x in spec.hidden)},act={str(spec.activation)})",
        coef=np.zeros((0,), dtype=np.float64),
        bias=0.0,
        r2=float(r2),
        feature_names=list(feature_names),
        extra=extra,
    )


def mlp_predict_from_fit(*, X: np.ndarray, fit: FitResult, base_feature_dim: int) -> np.ndarray:
    """
    Adapter predict hook for normalize=std:
      r(X) = model( (X - mean)/std )
    """
    if not fit.extra or str(fit.extra.get("kind", "")).lower().strip() != MLP_KIND:
        raise ValueError(f"fit is not {MLP_KIND!r}")

    x_mean = np.asarray(fit.extra["x_mean"], dtype=np.float32).reshape(-1)
    x_std = np.asarray(fit.extra["x_std"], dtype=np.float32).reshape(-1)
    if int(x_mean.size) != int(base_feature_dim) or int(x_std.size) != int(base_feature_dim):
        raise ValueError("invalid x_mean/x_std in fit.extra")

    spec = MLPSpec(
        hidden=tuple(int(x) for x in fit.extra.get("mlp_hidden", (64, 64))),
        activation=str(fit.extra.get("mlp_activation", "gelu")),
        dropout=float(fit.extra.get("mlp_dropout", 0.0)),
        device="cpu",
    )

    # rebuild model on CPU for inference
    import torch
    model = _build_mlp(in_dim=int(base_feature_dim), spec=spec).to(device="cpu")
    sd_np = fit.extra["state_dict"]
    if not isinstance(sd_np, dict):
        raise ValueError("invalid state_dict in fit.extra")
    model.load_state_dict(_from_state_dict_numpy(sd_np))
    model.eval()

    Xs = _standardize_apply(np.asarray(X, dtype=np.float32), x_mean, x_std)
    with torch.no_grad():
        Xt = torch.from_numpy(Xs).to(device="cpu")
        pred = model(Xt).detach().cpu().numpy().reshape(-1)
    return np.asarray(pred, dtype=np.float64)


def mlp_scale_extra(extra: Dict[str, object], scale: float) -> Dict[str, object]:
    """
    Keep MLP compact params consistent with scaling.

    Scaling semantics in api._normalize_fit():
      r_scaled(X) = scale * r_raw(X)

    For an MLP, easiest is to scale the FINAL linear layer (weight + bias) by scale.
    That preserves exactly: model_scaled(x) == scale * model_raw(x).

    We implement this by:
      - finding last Linear layer params in state_dict (heuristic: highest index in "N.weight"/"N.bias")
      - multiplying them by scale
    """
    out = dict(extra)
    if str(out.get("kind", "")).lower().strip() != MLP_KIND:
        return out

    sd = out.get("state_dict", None)
    if not isinstance(sd, dict):
        return out

    # identify final layer keys
    weight_keys = [k for k in sd.keys() if str(k).endswith(".weight")]
    bias_keys = [k for k in sd.keys() if str(k).endswith(".bias")]

    def _idx(k: str) -> int:
        # sequential uses "0.weight", "0.bias", "2.weight", ...
        base = str(k).split(".")[0]
        try:
            return int(base)
        except Exception:
            return -1

    if not weight_keys:
        return out

    last_w = max(weight_keys, key=_idx)
    last_b = None
    if bias_keys:
        # pick bias with same prefix if exists, else max index bias
        pref = str(last_w).split(".")[0] + ".bias"
        last_b = pref if pref in sd else max(bias_keys, key=_idx)

    w = sd.get(last_w, None)
    if isinstance(w, np.ndarray):
        sd[last_w] = (np.asarray(w, dtype=np.float64) * float(scale)).astype(w.dtype, copy=False)

    if last_b is not None:
        b = sd.get(last_b, None)
        if isinstance(b, np.ndarray):
            sd[last_b] = (np.asarray(b, dtype=np.float64) * float(scale)).astype(b.dtype, copy=False)

    out["state_dict"] = sd
    return out


def mlp_save_extra_npz(*, save_dict: Dict[str, object], best_raw: FitResult, best_scaled: FitResult) -> None:
    """
    Model-specific np.savez payload for MLP.
    Store:
      - x_mean/x_std
      - state_dict raw/scaled (object arrays of (key, value))
    """
    save_dict["best_extra_kind"] = np.asarray([str(best_raw.extra.get("kind", ""))], dtype=object)

    save_dict["mlp_hidden"] = np.asarray(list(best_raw.extra.get("mlp_hidden", (64, 64))), dtype=np.int64)
    save_dict["mlp_activation"] = np.asarray([str(best_raw.extra.get("mlp_activation", "gelu"))], dtype=object)
    save_dict["mlp_dropout"] = np.asarray([float(best_raw.extra.get("mlp_dropout", 0.0))], dtype=np.float64)

    save_dict["mlp_x_mean"] = np.asarray(best_raw.extra["x_mean"], dtype=np.float32)
    save_dict["mlp_x_std"] = np.asarray(best_raw.extra["x_std"], dtype=np.float32)

    def _sd_items(sd: Dict[str, object]) -> np.ndarray:
        # store as object array of tuples (k, v)
        items = []
        for k in sorted(sd.keys()):
            items.append((str(k), sd[k]))
        return np.asarray(items, dtype=object)

    sd_raw = best_raw.extra.get("state_dict", {})
    sd_scl = best_scaled.extra.get("state_dict", {})
    if not isinstance(sd_raw, dict) or not isinstance(sd_scl, dict):
        raise ValueError("mlp state_dict missing/invalid in fit.extra")

    save_dict["mlp_state_dict_raw"] = _sd_items(sd_raw)
    save_dict["mlp_state_dict_scaled"] = _sd_items(sd_scl)


def _fit_mlp_dispatch(
        X: np.ndarray,
        y: np.ndarray,
        *,
        logger: Any,
        which: str,
        feature_names: Sequence[str],
) -> list[FitResult]:
    # which is currently unused; keep it for interface compatibility
    _ = which
    logger.info("[fit_reward] fitting MLP")
    fit = fit_mlp(X, y, feature_names=feature_names, logger=logger, spec=MLPSpec())
    return [fit]


def make_mlp_adapter() -> ModelAdapter:
    return ModelAdapter(
        kind=MLP_KIND,
        fit=_fit_mlp_dispatch,
        predict=lambda X, fit, base_feature_dim: mlp_predict_from_fit(
            X=np.asarray(X, dtype=np.float64),
            fit=fit,
            base_feature_dim=int(base_feature_dim),
        ),
        scale_extra=lambda extra, scale: mlp_scale_extra(extra, float(scale)),
        save_extra=lambda save_dict, best_raw, best_scaled: mlp_save_extra_npz(
            save_dict=save_dict,
            best_raw=best_raw,
            best_scaled=best_scaled,
        ),
    )
