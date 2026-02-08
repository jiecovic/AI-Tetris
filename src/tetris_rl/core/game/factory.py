# src/tetris_rl/core/game/factory.py
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    pass  # type: ignore[import-not-found]


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
    Engine + warmup spec (engine-owned gating).

    warmup_spec:
      - PyO3 WarmupSpec instance OR None
      - passed to engine at construction time
    """

    game: Any
    warmup_spec: Optional[Any]


def _parse_warmup_spec(obj: Any) -> Optional[Any]:
    """
    Convert config -> tetris_rl_engine.WarmupSpec.

    Canonical spec shape:
      {"type": "...", "params": {...}}
    """
    if obj is None:
        return None

    _, WarmupSpec = _import_engine()

    # already bound object
    if isinstance(obj, WarmupSpec):
        return obj

    if not isinstance(obj, dict):
        raise TypeError(f"game.warmup.spec must be None|WarmupSpec|dict, got {type(obj)!r}")

    t = str(obj.get("type", "")).strip().lower()
    if not t:
        raise ValueError("game.warmup.spec.type must be a non-empty string")

    params = obj.get("params", {}) or {}
    if not isinstance(params, dict):
        raise TypeError("game.warmup.spec.params must be a mapping")

    if t in {"none", "off", "disabled", "null"}:
        spec = WarmupSpec.none()

    elif t == "fixed":
        rows = int(params["rows"])
        holes = int(params.get("holes", 1))
        spec = WarmupSpec.fixed(rows, holes=holes)

    elif t in {"uniform_rows", "uniform"}:
        min_rows = int(params["min_rows"])
        max_rows = int(params["max_rows"])
        holes = int(params.get("holes", 1))
        spec = WarmupSpec.uniform_rows(min_rows, max_rows, holes=holes)

    elif t == "poisson":
        lam_raw = params.get("lambda", params.get("lambda_", None))
        if lam_raw is None:
            raise KeyError("warmup.type=poisson requires key 'lambda' (or 'lambda_')")
        lam = float(lam_raw)
        cap = int(params["cap"])
        holes = int(params.get("holes", 1))
        spec = WarmupSpec.poisson(lam, cap, holes=holes)

    elif t in {"base_plus_poisson", "base+poisson"}:
        base = int(params["base"])
        lam_raw = params.get("lambda", params.get("lambda_", None))
        if lam_raw is None:
            raise KeyError("warmup.type=base_plus_poisson requires key 'lambda' (or 'lambda_')")
        lam = float(lam_raw)
        cap = int(params["cap"])
        holes = int(params.get("holes", 1))
        spec = WarmupSpec.base_plus_poisson(base, lam, cap, holes=holes)

    else:
        raise ValueError(f"unknown warmup.type={t!r}")

    # optional: make holes uniform after base spec was built
    uh = params.get("uniform_holes", None)
    if uh is not None:
        if not isinstance(uh, dict):
            raise TypeError("warmup.params.uniform_holes must be a dict {min,max}")
        spec = spec.with_uniform_holes(int(uh["min"]), int(uh["max"]))

    return spec



def _parse_warmup_cfg(obj: Any) -> Optional[Any]:
    """
    Parse cfg.env.game.warmup in canonical form:

      warmup: null
      warmup:
        prob: 0.9
        spec:
          type: poisson
          params: {...}
    """
    if obj is None:
        return None

    # If someone already passed a WarmupSpec instance here, treat as prob=1.
    _, WarmupSpec = _import_engine()
    if isinstance(obj, WarmupSpec):
        return obj

    if not isinstance(obj, dict):
        raise TypeError(f"cfg.env.game.warmup must be a mapping or null, got {type(obj)!r}")

    prob = float(obj.get("prob", 1.0))
    if prob <= 0.0:
        return None
    if prob > 1.0:
        raise ValueError(f"cfg.env.game.warmup.prob must be in [0,1], got {prob}")

    spec_obj = obj.get("spec", None)
    if spec_obj is None:
        raise KeyError("cfg.env.game.warmup.spec is required when warmup is enabled")
    if not isinstance(spec_obj, dict):
        raise TypeError(f"cfg.env.game.warmup.spec must be a mapping, got {type(spec_obj)!r}")

    warmup_spec = _parse_warmup_spec(spec_obj)
    if warmup_spec is None:
        return None
    if prob < 1.0:
        warmup_spec = warmup_spec.with_prob(prob)
    return warmup_spec


def make_game_bundle_from_cfg(cfg: Dict[str, Any]) -> GameBundle:
    """
    Construct the Rust engine wrapper + warmup spec.

    Notes:
      - Engine stores (piece_rule, warmup) as defaults for reset() reuse.
      - Warmup gating probability is handled inside Rust.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a mapping, got {type(cfg)!r}")

    env_cfg = cfg.get("env", {}) or {}
    if not isinstance(env_cfg, dict):
        env_cfg = {}
    game_cfg = env_cfg.get("game", {}) or {}
    if not isinstance(game_cfg, dict):
        game_cfg = {}

    TetrisEngine, _ = _import_engine()

    # default seed here is fine; env.reset(seed=episode_seed) should override per episode
    seed = int(game_cfg.get("seed", 12345))

    # Rust expects "uniform" | "bag7"
    piece_rule = str(game_cfg.get("piece_rule", "uniform")).strip().lower()

    warmup_spec = _parse_warmup_cfg(game_cfg.get("warmup", None))

    # Engine owns warmup gating; pass the spec at construction time.
    game = TetrisEngine(seed=seed, piece_rule=piece_rule, warmup=warmup_spec)

    return GameBundle(game=game, warmup_spec=warmup_spec)


def make_game_from_cfg(cfg: Dict[str, Any]) -> Any:
    """
    Backward-compat helper returning only the engine.

    Prefer make_game_bundle_from_cfg() for training/env construction (so warmup prob can be used).
    """
    return make_game_bundle_from_cfg(cfg).game


__all__ = ["GameBundle", "make_game_bundle_from_cfg", "make_game_from_cfg"]
