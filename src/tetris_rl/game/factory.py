# src/tetris_rl/game/factory.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from tetris_rl_engine import TetrisEngine as _TetrisEngine  # type: ignore[import-not-found]
    from tetris_rl_engine import WarmupSpec as _WarmupSpec      # type: ignore[import-not-found]


def _import_engine() -> tuple[Any, Any]:
    """
    Runtime import for the PyO3 extension.
    Keeps static analyzers from hard-failing when the extension isn't built.
    """
    try:
        from tetris_rl_engine import TetrisEngine, WarmupSpec  # type: ignore[import-not-found]
        return TetrisEngine, WarmupSpec
    except Exception as e:
        raise ImportError(
            "Failed to import 'tetris_rl_engine' (PyO3 extension). "
            "Build/install it into this interpreter."
        ) from e


@dataclass(frozen=True)
class GameBundle:
    """
    Engine + warmup gating info.

    warmup_prob:
      - probability in [0,1] that env should apply warmup on each reset
      - if 0, env should pass warmup=None (disabled)

    warmup_spec:
      - PyO3 WarmupSpec instance OR None
      - when env decides to apply warmup, it passes this object to engine.reset(..., warmup=...)
    """

    game: Any
    warmup_prob: float
    warmup_spec: Optional[Any]


def _parse_warmup_spec(obj: Any) -> Optional[Any]:
    """
    Convert config -> tetris_rl_engine.WarmupSpec.

    Accepts:
      - None
      - already a WarmupSpec instance
      - dict forms (spec dict):
          {"type":"none"}
          {"type":"fixed", "rows":18, "holes":1, "spawn_buffer":2}
          {"type":"uniform_rows", "min_rows":10, "max_rows":18, "holes":1, "spawn_buffer":2}
          {"type":"poisson", "lambda":12.0, "cap":18, "holes":1, "spawn_buffer":2}
          {"type":"base_plus_poisson", "base":8, "lambda":6.0, "cap":18, "holes":1, "spawn_buffer":2}

        Optional post-transform:
          "uniform_holes": {"min": 1, "max": 3}
    """
    if obj is None:
        return None

    _, WarmupSpec = _import_engine()

    # already bound object
    if isinstance(obj, WarmupSpec):
        return obj

    if not isinstance(obj, dict):
        raise TypeError(f"game.warmup.spec must be None|WarmupSpec|dict, got {type(obj)!r}")

    # spawn_buffer is now engine-internal; accept it in configs for back-compat but ignore it.
    # (Do NOT forward to WarmupSpec constructors.)
    _ = obj.get("spawn_buffer", None)

    t = str(obj.get("type", "none")).strip().lower()

    if t in {"none", "off", "disabled", "null"}:
        spec = WarmupSpec.none()

    elif t == "fixed":
        rows = int(obj["rows"])
        holes = int(obj.get("holes", 1))
        spec = WarmupSpec.fixed(rows, holes=holes)

    elif t in {"uniform_rows", "uniform"}:
        min_rows = int(obj["min_rows"])
        max_rows = int(obj["max_rows"])
        holes = int(obj.get("holes", 1))
        spec = WarmupSpec.uniform_rows(min_rows, max_rows, holes=holes)

    elif t == "poisson":
        lam_raw = obj.get("lambda", obj.get("lambda_", None))
        if lam_raw is None:
            raise KeyError("warmup.type=poisson requires key 'lambda' (or 'lambda_')")
        lam = float(lam_raw)
        cap = int(obj["cap"])
        holes = int(obj.get("holes", 1))
        spec = WarmupSpec.poisson(lam, cap, holes=holes)

    elif t in {"base_plus_poisson", "base+poisson"}:
        base = int(obj["base"])
        lam_raw = obj.get("lambda", obj.get("lambda_", None))
        if lam_raw is None:
            raise KeyError("warmup.type=base_plus_poisson requires key 'lambda' (or 'lambda_')")
        lam = float(lam_raw)
        cap = int(obj["cap"])
        holes = int(obj.get("holes", 1))
        spec = WarmupSpec.base_plus_poisson(base, lam, cap, holes=holes)

    else:
        raise ValueError(f"unknown warmup.type={t!r}")

    # optional: make holes uniform after base spec was built
    uh = obj.get("uniform_holes", None)
    if uh is not None:
        if not isinstance(uh, dict):
            raise TypeError("warmup.uniform_holes must be a dict {min,max}")
        spec = spec.with_uniform_holes(int(uh["min"]), int(uh["max"]))

    return spec



def _parse_warmup_cfg(obj: Any) -> Tuple[float, Optional[Any]]:
    """
    Parse cfg.game.warmup in canonical form:

      warmup: null
      warmup:
        prob: 0.9
        spec:
          type: poisson
          params: {...}     # accepted, but we also allow flattened spec dict for back-compat

    Accepted back-compat:
      warmup: {type: "...", ...}                 -> prob=1.0, spec=that dict
      warmup: {prob: 0.9, type: "...", ...}      -> prob from dict, spec from remaining keys
      warmup: {enabled: false, ...}              -> disabled
      warmup: {prob: 0.0, ...}                   -> disabled
    """
    if obj is None:
        return 0.0, None

    # If someone already passed a WarmupSpec instance here, treat as prob=1.
    _, WarmupSpec = _import_engine()
    if isinstance(obj, WarmupSpec):
        return 1.0, obj

    if not isinstance(obj, dict):
        raise TypeError(f"cfg.game.warmup must be a mapping or null, got {type(obj)!r}")

    enabled = bool(obj.get("enabled", True))
    prob = float(obj.get("prob", 1.0))

    if (not enabled) or prob <= 0.0:
        return 0.0, None

    if prob > 1.0:
        raise ValueError(f"cfg.game.warmup.prob must be in [0,1], got {prob}")

    spec_obj = obj.get("spec", None)

    # back-compat: treat warmup dict itself as spec if it looks like a spec
    if spec_obj is None and ("type" in obj or "uniform_holes" in obj or "rows" in obj or "min_rows" in obj):
        spec_obj = dict(obj)
        # strip non-spec keys
        spec_obj.pop("enabled", None)
        spec_obj.pop("prob", None)
        spec_obj.pop("spec", None)

        # if "params" exists, allow the canonical (type, params) shape too
        params = spec_obj.get("params", None)
        if isinstance(params, dict):
            t = str(spec_obj.get("type", "none")).strip().lower()
            # expand canonical params into flat keys expected by _parse_warmup_spec
            merged = {"type": t, **dict(params)}
            if "uniform_holes" in spec_obj:
                merged["uniform_holes"] = spec_obj["uniform_holes"]
            spec_obj = merged

    # canonical: spec is mapping with (type, params) or already flattened
    if spec_obj is None:
        return 0.0, None
    if not isinstance(spec_obj, dict):
        raise TypeError(f"cfg.game.warmup.spec must be a mapping or null, got {type(spec_obj)!r}")

    # If canonical {type, params}, expand params into flat dict.
    if "params" in spec_obj and isinstance(spec_obj.get("params"), dict):
        t = str(spec_obj.get("type", "none")).strip().lower()
        merged = {"type": t, **dict(spec_obj["params"])}
        if "uniform_holes" in spec_obj:
            merged["uniform_holes"] = spec_obj["uniform_holes"]
        spec_obj = merged

    warmup_spec = _parse_warmup_spec(spec_obj)
    return float(prob), warmup_spec


def make_game_bundle_from_cfg(cfg: Dict[str, Any]) -> GameBundle:
    """
    Construct the Rust engine wrapper + warmup gating info.

    Notes:
      - Engine stores (piece_rule, warmup) as defaults for reset() reuse.
      - We still return (prob, warmup_spec) so Python env can decide per-episode:
          engine.reset(seed=..., warmup=spec_or_none)
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a mapping, got {type(cfg)!r}")

    game_cfg = cfg.get("game", {}) or {}
    if not isinstance(game_cfg, dict):
        game_cfg = {}

    TetrisEngine, _ = _import_engine()

    # default seed here is fine; env.reset(seed=episode_seed) should override per episode
    seed = int(game_cfg.get("seed", 12345))

    # Rust expects "uniform" | "bag7"
    piece_rule = str(game_cfg.get("piece_rule", "uniform")).strip().lower()

    warmup_prob, warmup_spec = _parse_warmup_cfg(game_cfg.get("warmup", None))

    # IMPORTANT: set engine default warmup to NONE.
    # We want the env to control whether warmup is applied each episode by passing warmup=...
    game = TetrisEngine(seed=seed, piece_rule=piece_rule, warmup=None)

    return GameBundle(game=game, warmup_prob=warmup_prob, warmup_spec=warmup_spec)


def make_game_from_cfg(cfg: Dict[str, Any]) -> Any:
    """
    Backward-compat helper returning only the engine.

    Prefer make_game_bundle_from_cfg() for training/env construction (so warmup prob can be used).
    """
    return make_game_bundle_from_cfg(cfg).game


__all__ = ["GameBundle", "make_game_bundle_from_cfg", "make_game_from_cfg"]
