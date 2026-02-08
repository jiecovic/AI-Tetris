# src/tetris_rl/core/training/imitation/bc.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class BCTrainSpec:
    learning_rate: float = 3e-4
    max_grad_norm: float = 1.0

    # How often to call on_update (in optimizer updates).
    # 1 = every batch/update (recommended for live tqdm display).
    log_every_updates: int = 1


def _policy_obs_keys(policy: Any) -> Optional[Tuple[str, ...]]:
    try:
        space = getattr(policy, "observation_space", None)
        spaces = getattr(space, "spaces", None)
        if isinstance(spaces, dict) and spaces:
            return tuple(str(k) for k in spaces.keys())
    except Exception:
        pass
    return None


def _split_batch(policy: Any, batch: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
    legal_mask = None
    if "legal_mask" in batch:
        try:
            legal_mask = np.asarray(batch.get("legal_mask"))
        except Exception:
            legal_mask = None

    candidate_obs: Dict[str, np.ndarray] = {}
    for k, v in batch.items():
        if k in {"action", "legal_mask"}:
            continue
        candidate_obs[str(k)] = np.asarray(v)

    keys = _policy_obs_keys(policy)
    if keys is None:
        return candidate_obs, legal_mask

    obs_b: Dict[str, np.ndarray] = {}
    missing: list[str] = []
    for k in keys:
        if k in candidate_obs:
            obs_b[k] = candidate_obs[k]
        else:
            missing.append(k)

    if missing:
        raise KeyError(
            "BC batch missing observation keys required by policy.observation_space: "
            f"{missing}. Present keys: {sorted(candidate_obs.keys())}"
        )

    return obs_b, legal_mask


def _to_torch_obs(policy: Any, obs_batch: Dict[str, np.ndarray], device: torch.device) -> Any:
    if hasattr(policy, "obs_to_tensor"):
        t, _ = policy.obs_to_tensor(obs_batch)  # type: ignore[misc]
        if isinstance(t, dict):
            return {k: v.to(device) for k, v in t.items()}
        return t.to(device)

    return {k: torch.as_tensor(v, device=device) for k, v in obs_batch.items()}


def _policy_log_prob(policy: Any, obs_t: Any, actions_t: torch.Tensor) -> torch.Tensor:
    if hasattr(policy, "get_distribution"):
        dist = policy.get_distribution(obs_t)  # type: ignore[misc]
        return dist.log_prob(actions_t)

    if hasattr(policy, "evaluate_actions"):
        out = policy.evaluate_actions(obs_t, actions_t)  # type: ignore[misc]
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            return out[1]
        raise RuntimeError("policy.evaluate_actions returned unexpected output")

    raise RuntimeError("policy has neither get_distribution nor evaluate_actions")


def _policy_entropy_and_acc(policy: Any, obs_t: Any, actions_t: torch.Tensor) -> Tuple[Optional[float], Optional[float]]:
    """
    Best-effort diagnostics:
      - entropy: mean(dist.entropy())
      - acc_top1: argmax(probs) == action

    Returns (entropy_mean, acc_top1) or (None, None) if unavailable.
    """
    try:
        if not hasattr(policy, "get_distribution"):
            return None, None

        dist = policy.get_distribution(obs_t)  # type: ignore[misc]

        ent_v: Optional[float] = None
        try:
            ent = dist.entropy()
            ent_v = float(ent.mean().detach().cpu().item())
        except Exception:
            ent_v = None

        acc_v: Optional[float] = None
        try:
            probs = getattr(dist, "distribution", None)
            probs = getattr(probs, "probs", None)
            if probs is not None:
                pred = probs.argmax(dim=-1)
                correct = torch.eq(pred, actions_t)
                acc_v = float(correct.float().mean().detach().cpu().item())
        except Exception:
            acc_v = None

        return ent_v, acc_v
    except Exception:
        return None, None


def bc_train_stream(
        *,
        model: Any,
        batch_iter: Any,  # Iterator[Dict[str, np.ndarray]]
        spec: BCTrainSpec,
        device: str | torch.device = "cpu",
        on_update: Optional[Callable[[int, Dict[str, float]], None]] = None,
) -> Dict[str, float]:
    """
    Behavior cloning on a stream of batches.

    Required batch fields:
      - action: (B,) int64
      - observation fields matching model.policy.observation_space

    Optional batch fields:
      - legal_mask: (B,A) bool  (metadata; NOT part of observation)
    """
    policy = getattr(model, "policy", None)
    if policy is None:
        raise TypeError("model has no .policy (expected SB3 model with policy)")

    lr = float(spec.learning_rate)
    max_gn = float(spec.max_grad_norm)
    log_every = max(1, int(spec.log_every_updates))

    dev = torch.device(device) if not isinstance(device, torch.device) else device
    policy = policy.to(dev)

    params = [p for p in policy.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("no trainable parameters found on policy")

    opt = torch.optim.Adam(params, lr=lr)

    total_updates = 0
    total_samples = 0
    last_loss = 0.0

    last_entropy: Optional[float] = None
    last_acc: Optional[float] = None

    _ = _policy_obs_keys(policy)  # cache introspection once

    for batch in batch_iter:
        if batch is None:
            continue

        act = np.asarray(batch.get("action"))
        if act.ndim != 1:
            raise ValueError(f"batch['action'] must be (B,), got {act.shape}")
        bsz = int(act.shape[0])
        if bsz <= 0:
            continue

        obs_b, _legal_mask = _split_batch(policy, batch)

        actions_t = torch.as_tensor(act.astype(np.int64, copy=False), device=dev, dtype=torch.long)
        obs_t = _to_torch_obs(policy, obs_b, dev)

        logp = _policy_log_prob(policy, obs_t, actions_t)
        loss = (-logp).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if max_gn > 0:
            nn.utils.clip_grad_norm_(params, max_norm=max_gn)
        opt.step()

        total_updates += 1
        total_samples += int(bsz)
        last_loss = float(loss.detach().cpu().item())

        last_entropy, last_acc = _policy_entropy_and_acc(policy, obs_t, actions_t)

        if on_update is not None and (total_updates % log_every) == 0:
            stats: Dict[str, float] = {
                # keep the "bc/..." names (runner expects these), but also include flat aliases
                "bc/loss": float(last_loss),
                "bc/updates": float(total_updates),
                "bc/samples": float(total_samples),
                "bc_loss": float(last_loss),
                "bc_updates": float(total_updates),
                "bc_samples": float(total_samples),
            }
            if last_entropy is not None:
                stats["bc/entropy"] = float(last_entropy)
                stats["bc_entropy"] = float(last_entropy)
            if last_acc is not None:
                stats["bc/acc_top1"] = float(last_acc)
                stats["bc_acc_top1"] = float(last_acc)

            # DO NOT swallow errors here; if this fails, you want to know.
            on_update(int(total_updates), stats)

    out: Dict[str, float] = {
        "bc_loss": float(last_loss),
        "bc_updates": float(total_updates),
        "bc_samples": float(total_samples),
    }
    if last_entropy is not None:
        out["bc_entropy"] = float(last_entropy)
    if last_acc is not None:
        out["bc_acc_top1"] = float(last_acc)
    return out


def bc_eval_stream(
        *,
        model: Any,
        batch_iter: Any,  # Iterator[Dict[str, np.ndarray]]
        device: str | torch.device = "cpu",
        max_samples: int = 0,
) -> Dict[str, float]:
    """
    Offline (dataset-only) validation pass.

    Returns ALWAYS the same keys (stable table columns):
      - bc_val_loss
      - bc_val_acc_top1
      - bc_val_entropy
      - bc_val_samples

    Missing diagnostics are reported as NaN.
    """
    policy = getattr(model, "policy", None)
    if policy is None:
        raise TypeError("model has no .policy (expected SB3 model with policy)")

    dev = torch.device(device) if not isinstance(device, torch.device) else device
    policy = policy.to(dev)
    policy.eval()

    limit = int(max_samples) if int(max_samples) > 0 else 0

    total_n = 0
    sum_loss = 0.0

    sum_ent = 0.0
    sum_acc = 0.0
    have_ent = False
    have_acc = False

    _ = _policy_obs_keys(policy)  # cache introspection once

    with torch.no_grad():
        for batch in batch_iter:
            if batch is None:
                continue

            act = np.asarray(batch.get("action"))
            if act.ndim != 1:
                raise ValueError(f"batch['action'] must be (B,), got {act.shape}")
            bsz = int(act.shape[0])
            if bsz <= 0:
                continue

            obs_b, _legal_mask = _split_batch(policy, batch)

            actions_t = torch.as_tensor(act.astype(np.int64, copy=False), device=dev, dtype=torch.long)
            obs_t = _to_torch_obs(policy, obs_b, dev)

            logp = _policy_log_prob(policy, obs_t, actions_t)
            loss = (-logp).mean()

            total_n += int(bsz)
            sum_loss += float(loss.detach().cpu().item()) * float(bsz)

            ent_v, acc_v = _policy_entropy_and_acc(policy, obs_t, actions_t)
            if ent_v is not None:
                have_ent = True
                sum_ent += float(ent_v) * float(bsz)
            if acc_v is not None:
                have_acc = True
                sum_acc += float(acc_v) * float(bsz)

            if limit and total_n >= limit:
                break

    denom = float(max(1, total_n))
    loss_mean = float(sum_loss / denom)
    ent_mean = float(sum_ent / denom) if have_ent else float("nan")
    acc_mean = float(sum_acc / denom) if have_acc else float("nan")

    return {
        "bc_val_loss": float(loss_mean),
        "bc_val_acc_top1": float(acc_mean),
        "bc_val_entropy": float(ent_mean),
        "bc_val_samples": float(total_n),
    }


__all__ = [
    "BCTrainSpec",
    "bc_train_stream",
    "bc_eval_stream",
]
