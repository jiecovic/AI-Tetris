# src/tetris_rl/config/game_spec.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from tetris_rl.config.schema_types import (
    get_float,
    get_mapping,
    get_str,
    require_mapping,
)


@dataclass(frozen=True)
class WarmupSpecCfg:
    """
    Pure-Python warmup config (no PyO3 objects).

    Canonical YAML (recommended):

      game:
        warmup:
          prob: 0.9
          spec:
            type: poisson
            params:
              lambda: 16
              cap: 18
              holes: 1
              spawn_buffer: 2
            uniform_holes: { min: 1, max: 3 }

    Back-compat accepted forms:
      - game.warmup: null
      - game.warmup: {type: "...", ...}                # treated as spec (prob=1.0)
      - game.warmup: {prob: 0.9, type: "...", ...}     # treated as spec with prob
      - game.warmup: {prob: 0.9, spec: {type: "..."}}
    """

    enabled: bool = True
    prob: float = 1.0

    # A normalized warmup "spec dict" that game.factory will convert to PyO3 WarmupSpec.
    # Expected shape:
    #   {"type": "...", "params": {...}, "uniform_holes": {...?}}
    spec: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class GameSpec:
    """
    Game-level config (engine-facing).

    Keep this as the single home for things that conceptually belong to the engine:
      - piece rule
      - warmup noise (board initialization)
      - future: scoring rulesets, gravity variants, etc.
    """

    piece_rule: str = "uniform"  # "uniform" | "bag7"
    warmup: WarmupSpecCfg = field(default_factory=WarmupSpecCfg)


def _normalize_warmup(*, warmup_obj: Any, where: str) -> WarmupSpecCfg:
    """
    Normalize the various accepted warmup forms into WarmupSpecCfg.

    Returns:
      WarmupSpecCfg(enabled=False, spec=None) if disabled/null.
    """
    if warmup_obj is None:
        return WarmupSpecCfg(enabled=False, prob=0.0, spec=None)

    if not isinstance(warmup_obj, dict):
        raise TypeError(f"{where} must be a mapping or null, got {type(warmup_obj)!r}")

    # If user supplies: { enabled: false } or { type: null } => disable.
    enabled_raw = warmup_obj.get("enabled", True)
    enabled = bool(enabled_raw)

    # prob can appear at top-level in any form
    prob = float(warmup_obj.get("prob", 1.0))
    if prob < 0.0 or prob > 1.0:
        raise ValueError(f"{where}.prob must be in [0,1], got {prob}")

    # canonical nested form: warmup: { prob, spec: {...} }
    spec_obj = warmup_obj.get("spec", None)

    # back-compat: warmup dict itself is the spec (contains "type")
    if spec_obj is None and ("type" in warmup_obj or "params" in warmup_obj or "uniform_holes" in warmup_obj):
        spec_obj = dict(warmup_obj)
        # strip non-spec keys
        spec_obj.pop("enabled", None)
        spec_obj.pop("prob", None)
        spec_obj.pop("spec", None)

    # still None => treat as disabled (or "enabled true but no spec" => no warmup)
    if spec_obj is None:
        return WarmupSpecCfg(enabled=False, prob=0.0, spec=None)

    if not isinstance(spec_obj, dict):
        raise TypeError(f"{where}.spec must be a mapping, got {type(spec_obj)!r}")

    t = spec_obj.get("type", "none")
    if t is None:
        return WarmupSpecCfg(enabled=False, prob=0.0, spec=None)

    t = str(t).strip().lower()
    params = spec_obj.get("params", {}) or {}
    if not isinstance(params, dict):
        raise TypeError(f"{where}.spec.params must be a mapping, got {type(params)!r}")

    # Keep everything as raw dict; deeper validation happens in game.factory
    normalized: Dict[str, Any] = {"type": t, "params": dict(params)}

    # optional post-transform
    uh = spec_obj.get("uniform_holes", None)
    if uh is not None:
        if not isinstance(uh, dict):
            raise TypeError(f"{where}.spec.uniform_holes must be a mapping, got {type(uh)!r}")
        normalized["uniform_holes"] = dict(uh)

    # allow explicit "none" spec to mean disabled
    if t in {"none", "off", "disabled", "null"}:
        return WarmupSpecCfg(enabled=False, prob=0.0, spec=None)

    if not enabled:
        return WarmupSpecCfg(enabled=False, prob=0.0, spec=None)

    # enabled + spec => return it (prob may be 0 which effectively disables)
    if prob <= 0.0:
        return WarmupSpecCfg(enabled=False, prob=0.0, spec=None)

    return WarmupSpecCfg(enabled=True, prob=prob, spec=normalized)


def parse_game_spec(*, cfg: Dict[str, Any]) -> GameSpec:
    """
    Parse cfg.game into a typed GameSpec.

    Expected (canonical) layout:

      game:
        piece_rule: uniform | bag7
        warmup:
          prob: 0.9
          spec:
            type: poisson | fixed | uniform_rows | base_plus_poisson | none
            params: {...}
            uniform_holes: {min,max}  # optional

    Back-compat:
      - game.warmup may be a spec dict directly (contains "type"/"params")
      - game.warmup may be null
    """
    root = require_mapping(cfg, where="cfg")
    game = get_mapping(root, "game", default={}, where="cfg.game")

    piece_rule = get_str(game, "piece_rule", default=GameSpec.piece_rule, where="cfg.game.piece_rule").strip().lower()
    if piece_rule not in {"uniform", "bag7"}:
        raise ValueError(f"cfg.game.piece_rule must be 'uniform'|'bag7', got {piece_rule!r}")

    warmup_obj = game.get("warmup", None)
    warmup = _normalize_warmup(warmup_obj=warmup_obj, where="cfg.game.warmup")

    return GameSpec(piece_rule=str(piece_rule), warmup=warmup)


__all__ = ["GameSpec", "WarmupSpecCfg", "parse_game_spec"]
