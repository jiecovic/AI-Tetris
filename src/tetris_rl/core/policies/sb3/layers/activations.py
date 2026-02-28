# src/tetris_rl/core/policies/sb3/layers/activations.py
from __future__ import annotations

from typing import Type

from torch import nn

from tetris_rl.core.policies.sb3.types import LayerActivationName, PolicyActivationName

# Back-compat alias; historically this module exported ActivationName.
ActivationName = PolicyActivationName


class GELUNone(nn.GELU):
    """No-arg GELU class for SB3 activation_fn wiring (approximate='none')."""

    def __init__(self) -> None:
        super().__init__(approximate="none")


class GELUTanh(nn.GELU):
    """No-arg GELU class for SB3 activation_fn wiring (approximate='tanh')."""

    def __init__(self) -> None:
        super().__init__(approximate="tanh")


def normalize_activation_name(name: PolicyActivationName | str) -> str:
    """
    Normalize user-facing activation names to canonical internal tags.

    Canonical tags:
      - gelu        -> exact/default PyTorch GELU (approximate='none')
      - gelu_tanh   -> tanh-approx GELU
      - relu, silu, tanh, identity
    """
    n = str(name).strip().lower()
    alias_map = {
        "gelu_none": "gelu",
        "gelu_exact": "gelu",
        "gelu_approx": "gelu_tanh",
        "gelu_fast": "gelu_tanh",
        "swish": "silu",
        "none": "identity",
    }
    return alias_map.get(n, n)


def make_activation(name: LayerActivationName | str) -> nn.Module:
    """
    Map a string to an activation module.

    Conventions:
    - ReLU is non-inplace (safer for residual-style models + debugging).
    - GELU supports both exact/default ("gelu") and tanh-approx ("gelu_tanh").
    - Uses the same canonical mapping path as SB3 policy activation_fn.
    """
    return activation_class(name)()


def activation_class(name: PolicyActivationName | str) -> Type[nn.Module]:
    """
    Map a string to an activation module class (for SB3 policy_kwargs.activation_fn).
    """
    n = normalize_activation_name(name)
    if n == "gelu":
        return GELUNone
    if n == "gelu_tanh":
        return GELUTanh
    if n == "relu":
        return nn.ReLU
    if n == "silu":
        return nn.SiLU
    if n == "tanh":
        return nn.Tanh
    if n == "identity":
        return nn.Identity
    raise ValueError(f"unknown activation: {name!r}")


__all__ = [
    "ActivationName",
    "GELUNone",
    "GELUTanh",
    "normalize_activation_name",
    "make_activation",
    "activation_class",
]
