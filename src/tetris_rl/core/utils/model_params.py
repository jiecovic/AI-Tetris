# src/tetris_rl/core/utils/model_params.py
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Type

import torch
from torch import nn


@dataclass(frozen=True)
class ParamCount:
    total: int
    trainable: int


def _count_params(module: torch.nn.Module) -> ParamCount:
    total = 0
    trainable = 0
    for p in module.parameters():
        n = int(p.numel())
        total += n
        if p.requires_grad:
            trainable += n
    return ParamCount(total=total, trainable=trainable)


def _get_attr(obj: Any, name: str) -> Optional[Any]:
    return getattr(obj, name, None)


def _format_count(n: int) -> str:
    # 20 -> "20", 1200 -> "1.2K", 2_340_000 -> "2.34M", 1_200_000_000 -> "1.20B"
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


# -----------------------------------------------------------------------------
# SB3 helpers (policy/algo plumbing)
# -----------------------------------------------------------------------------

def parse_net_arch(obj: Any) -> list[int] | dict[str, list[int]]:
    """
    SB3 supports:
      - list[int]
      - dict(pi=[...], vf=[...])  (ActorCriticPolicy)
    """
    if obj is None:
        return {"pi": [], "vf": []}

    if isinstance(obj, (list, tuple)):
        return [int(x) for x in obj if x is not None]

    if isinstance(obj, dict):
        pi = obj.get("pi", [])
        vf = obj.get("vf", [])
        if not isinstance(pi, (list, tuple)) or not isinstance(vf, (list, tuple)):
            raise TypeError("net_arch dict must have list values for keys 'pi' and 'vf'")
        return {
            "pi": [int(x) for x in pi if x is not None],
            "vf": [int(x) for x in vf if x is not None],
        }

    raise TypeError("net_arch must be list[int] or dict(pi=[...], vf=[...])")


def parse_activation_fn(name: str) -> Type[nn.Module]:
    """
    Map config strings to torch activation *classes* (SB3 expects a class).
    """
    n = str(name).strip().lower()
    if n == "gelu":
        return nn.GELU
    if n == "relu":
        return nn.ReLU
    if n in {"silu", "swish"}:
        return nn.SiLU
    if n == "tanh":
        return nn.Tanh
    if n in {"identity", "none"}:
        return nn.Identity
    raise ValueError(f"unknown activation_fn: {name!r}")


def build_algo_kwargs(*, algo_cls: Type[Any], raw: Mapping[str, Any], seed: int, where: str) -> Dict[str, Any]:
    """
    Shove YAML params into SB3 algo ctor.

    Rules:
      - YAML already provides correct python types (no coercion here)
      - seed is injected if missing
      - unknown keys raise (typos fail fast)
    """
    sig = inspect.signature(algo_cls.__init__)
    valid = set(sig.parameters.keys())
    valid.discard("self")

    user_kwargs: Dict[str, Any] = {str(k): v for k, v in dict(raw).items()}

    if "seed" not in user_kwargs:
        user_kwargs["seed"] = int(seed)

    unknown = sorted(set(user_kwargs.keys()) - valid)
    if unknown:
        raise TypeError(f"{where} contains unknown kwargs for {algo_cls.__name__}: {unknown!r}")

    return user_kwargs


# -----------------------------------------------------------------------------
# Param reports (existing)
# -----------------------------------------------------------------------------

def format_sb3_param_summary(model: Any) -> str:
    """
    One-line, human readable summary.
    Example: "[params] total=79.7K (trainable=79.7K)"
    """
    policy = _get_attr(model, "policy")
    if policy is None or not isinstance(policy, torch.nn.Module):
        return "[params] total=? (trainable=?)"

    pc = _count_params(policy)
    return f"[params] total={_format_count(pc.total)} (trainable={_format_count(pc.trainable)})"


def format_sb3_param_report(model: Any) -> str:
    """
    Detailed-ish report for SB3 algorithms that expose `.policy` (PPO/A2C/SAC/TD3/...).

    Extra handling for SB3 ActorCriticPolicy which may create:
      - features_extractor
      - pi_features_extractor / vf_features_extractor (sometimes shared)
      - mlp_extractor with policy_net/value_net
      - action_net / value_net
    """
    policy = _get_attr(model, "policy")
    if policy is None:
        return "[params] model has no .policy attribute"
    if not isinstance(policy, torch.nn.Module):
        return "[params] model.policy is not a torch.nn.Module"

    lines: list[str] = []

    pc_policy = _count_params(policy)
    lines.append(f"[params] policy: total={pc_policy.total:,} trainable={pc_policy.trainable:,}")

    # --- feature extractor(s) ---
    feat = _get_attr(policy, "features_extractor")
    pi_feat = _get_attr(policy, "pi_features_extractor")
    vf_feat = _get_attr(policy, "vf_features_extractor")

    if isinstance(feat, torch.nn.Module):
        pc = _count_params(feat)
        lines.append(f"[params] features_extractor({type(feat).__name__}): total={pc.total:,} trainable={pc.trainable:,}")

    if isinstance(pi_feat, torch.nn.Module):
        if pi_feat is feat:
            lines.append("[params] pi_features_extractor: <shared with features_extractor>")
        else:
            pc = _count_params(pi_feat)
            lines.append(f"[params] pi_features_extractor({type(pi_feat).__name__}): total={pc.total:,} trainable={pc.trainable:,}")

    if isinstance(vf_feat, torch.nn.Module):
        if vf_feat is feat:
            lines.append("[params] vf_features_extractor: <shared with features_extractor>")
        elif vf_feat is pi_feat:
            lines.append("[params] vf_features_extractor: <shared with pi_features_extractor>")
        else:
            pc = _count_params(vf_feat)
            lines.append(f"[params] vf_features_extractor({type(vf_feat).__name__}): total={pc.total:,} trainable={pc.trainable:,}")

    # --- mlp_extractor (SB3 common) ---
    mlp_extractor = _get_attr(policy, "mlp_extractor")
    if isinstance(mlp_extractor, torch.nn.Module):
        pc_mlp = _count_params(mlp_extractor)
        lines.append(f"[params] mlp_extractor({type(mlp_extractor).__name__}): total={pc_mlp.total:,} trainable={pc_mlp.trainable:,}")

        policy_net = _get_attr(mlp_extractor, "policy_net")
        if isinstance(policy_net, torch.nn.Module):
            pc = _count_params(policy_net)
            lines.append(f"[params]  ├─ policy_net: total={pc.total:,} trainable={pc.trainable:,}")

        value_net = _get_attr(mlp_extractor, "value_net")
        if isinstance(value_net, torch.nn.Module):
            pc = _count_params(value_net)
            lines.append(f"[params]  └─ value_net : total={pc.total:,} trainable={pc.trainable:,}")

    # --- heads / actor-critic bits ---
    for name in ("action_net", "value_net", "actor", "critic", "qf0", "qf1"):
        m = _get_attr(policy, name)
        if isinstance(m, torch.nn.Module):
            pc = _count_params(m)
            lines.append(f"[params] {name}({type(m).__name__}): total={pc.total:,} trainable={pc.trainable:,}")

    return "\n".join(lines)


__all__ = [
    "parse_net_arch",
    "parse_activation_fn",
    "build_algo_kwargs",
    "format_sb3_param_summary",
    "format_sb3_param_report",
]
