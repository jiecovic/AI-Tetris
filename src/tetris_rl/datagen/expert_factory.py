# src/tetris_rl/datagen/expert_factory.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from tetris_rl.config.datagen_spec import DataGenExpertSpec


@dataclass(frozen=True)
class RustExpertBundle:
    """
    Small wrapper around the Rust binding expert.

    We keep this as a tiny "bundle" so datagen code can:
      - pass around a single object
      - keep the Rust binding import localized
      - extend later (e.g. expose name/knobs) without touching call sites
    """
    policy: Any  # tetris_rl_engine.ExpertPolicy


def _opt_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    return int(x)


def _opt_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    return float(x)


def make_expert_from_spec(*, expert_spec: DataGenExpertSpec) -> RustExpertBundle:
    """
    Build a Rust ExpertPolicy from DataGenExpertSpec.

    Supported expert.type:
      - "codemy0"
      - "codemy1"
      - "codemy2"
      - "codemy2fast"

    Notes:
      - This factory is intentionally thin: config -> Rust binding call.
      - Any "defaults" should live in the config schema, not here.
    """
    from tetris_rl_engine import ExpertPolicy  # local import keeps optional dep tight

    t = str(expert_spec.type).strip().lower()

    # ---------------------------
    # codemy0/1/2 share knobs
    # ---------------------------
    beam_width = _opt_int(getattr(expert_spec, "beam_width", None))
    beam_from_depth = int(_opt_int(getattr(expert_spec, "beam_from_depth", None)) or 0)

    if t == "codemy0":
        return RustExpertBundle(
            policy=ExpertPolicy.codemy0(beam_width=beam_width, beam_from_depth=beam_from_depth)
        )
    if t == "codemy1":
        return RustExpertBundle(
            policy=ExpertPolicy.codemy1(beam_width=beam_width, beam_from_depth=beam_from_depth)
        )
    if t == "codemy2":
        return RustExpertBundle(
            policy=ExpertPolicy.codemy2(beam_width=beam_width, beam_from_depth=beam_from_depth)
        )

    # ---------------------------
    # codemy2fast has its own knob
    # ---------------------------
    if t == "codemy2fast":
        tail_weight = float(_opt_float(getattr(expert_spec, "tail_weight", None)) or 0.5)
        return RustExpertBundle(policy=ExpertPolicy.codemy2fast(tail_weight=tail_weight))

    raise ValueError(f"unknown datagen expert.type: {t!r}")


__all__ = ["RustExpertBundle", "make_expert_from_spec"]
