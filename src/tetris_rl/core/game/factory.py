# src/tetris_rl/core/game/factory.py
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

from tetris_rl.core.game.config import GameConfig, WarmupConfig
from tetris_rl.core.game.warmup_params import extract_holes_config


@lru_cache(maxsize=1)
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
            "Failed to import 'tetris_rl_engine' (PyO3 extension). Build/install it into this interpreter."
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

    _, WarmupSpec = _import_engine()
    if isinstance(obj, WarmupSpec):
        return obj

    if isinstance(obj, WarmupConfig):
        warmup_cfg = obj
    elif isinstance(obj, dict):
        warmup_cfg = WarmupConfig.model_validate(obj)
    else:
        raise TypeError(f"cfg.env.game.warmup must be WarmupConfig|mapping|null, got {type(obj)!r}")

    prob = float(warmup_cfg.prob)
    if prob <= 0.0:
        return None
    if prob > 1.0:
        raise ValueError(f"cfg.env.game.warmup.prob must be in [0,1], got {prob}")

    spec_cfg = warmup_cfg.spec
    if spec_cfg is None:
        raise KeyError("cfg.env.game.warmup.spec is required when warmup is enabled")

    t = str(spec_cfg.type).strip().lower()
    params_obj = spec_cfg.params
    if isinstance(params_obj, dict):
        params = dict(params_obj)
    else:
        params = params_obj.model_dump(mode="json", by_alias=True)
    holes_fixed, holes_range = extract_holes_config(params)

    if t in {"none", "off", "disabled", "null"}:
        warmup_spec = WarmupSpec.none()
    elif t == "fixed":
        rows = int(params["rows"])
        warmup_spec = WarmupSpec.fixed(rows, holes=holes_fixed)
    elif t in {"uniform_rows", "uniform"}:
        min_rows = int(params["min_rows"])
        max_rows = int(params["max_rows"])
        warmup_spec = WarmupSpec.uniform_rows(min_rows, max_rows, holes=holes_fixed)
    elif t == "poisson":
        lam_raw = params.get("lambda", params.get("lambda_", None))
        if lam_raw is None:
            raise KeyError("warmup.type=poisson requires key 'lambda' (or 'lambda_')")
        lam = float(lam_raw)
        cap = int(params["cap"])
        warmup_spec = WarmupSpec.poisson(lam, cap, holes=holes_fixed)
    elif t in {"base_plus_poisson", "base+poisson"}:
        base = int(params["base"])
        lam_raw = params.get("lambda", params.get("lambda_", None))
        if lam_raw is None:
            raise KeyError("warmup.type=base_plus_poisson requires key 'lambda' (or 'lambda_')")
        lam = float(lam_raw)
        cap = int(params["cap"])
        warmup_spec = WarmupSpec.base_plus_poisson(base, lam, cap, holes=holes_fixed)
    else:
        raise ValueError(f"unknown warmup.type={t!r}")

    if holes_range is not None:
        warmup_spec = warmup_spec.with_uniform_holes(int(holes_range[0]), int(holes_range[1]))

    if prob < 1.0:
        warmup_spec = warmup_spec.with_prob(prob)
    return warmup_spec


def make_game_bundle_from_cfg(cfg: dict[str, Any]) -> GameBundle:
    """
    Construct the Rust engine wrapper + warmup spec.

    Notes:
      - Engine stores (piece_rule, warmup) as defaults for reset() reuse.
      - Warmup gating probability is handled inside Rust.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a mapping, got {type(cfg)!r}")

    env_cfg = cfg.get("env", {})
    if env_cfg is None:
        env_cfg = {}
    if not isinstance(env_cfg, dict):
        raise TypeError(f"cfg.env must be a mapping when provided, got {type(env_cfg)!r}")

    game_cfg_raw = env_cfg.get("game", {})
    if game_cfg_raw is None:
        game_cfg_raw = {}
    if isinstance(game_cfg_raw, GameConfig):
        game_cfg = game_cfg_raw
    elif isinstance(game_cfg_raw, dict):
        game_cfg = GameConfig.model_validate(game_cfg_raw)
    else:
        raise TypeError(f"cfg.env.game must be GameConfig|mapping when provided, got {type(game_cfg_raw)!r}")

    TetrisEngine, _ = _import_engine()

    # default seed here is fine; env.reset(seed=episode_seed) should override per episode
    seed = int(game_cfg.seed)

    # Rust expects "uniform" | "bag7"
    piece_rule = str(game_cfg.piece_rule).strip().lower()

    warmup_spec = _parse_warmup_cfg(game_cfg.warmup)

    # Engine owns warmup gating; pass the spec at construction time.
    game = TetrisEngine(seed=seed, piece_rule=piece_rule, warmup=warmup_spec)

    return GameBundle(game=game, warmup_spec=warmup_spec)


def make_game_from_cfg(cfg: dict[str, Any]) -> Any:
    """
    Backward-compat helper returning only the engine.

    Prefer make_game_bundle_from_cfg() for training/env construction (so warmup prob can be used).
    """
    return make_game_bundle_from_cfg(cfg).game


__all__ = ["GameBundle", "make_game_bundle_from_cfg", "make_game_from_cfg"]
