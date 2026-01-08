# src/tetris_rl/game/factory.py
from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

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


def _parse_warmup_spec(obj: Any) -> Optional[Any]:
    """
    Convert config -> tetris_rl_engine.WarmupSpec.

    Accepts:
      - None
      - already a WarmupSpec instance
      - dict forms:
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
    if obj.__class__.__name__ == "WarmupSpec":
        return obj

    if not isinstance(obj, dict):
        raise TypeError(f"game.warmup must be None|WarmupSpec|dict, got {type(obj)!r}")

    t = str(obj.get("type", "none")).strip().lower()

    spawn_buffer = obj.get("spawn_buffer", None)
    spawn_buffer_i = None if spawn_buffer is None else int(spawn_buffer)

    if t in {"none", "off", "disabled"}:
        spec = WarmupSpec.none()

    elif t == "fixed":
        rows = int(obj["rows"])
        holes = int(obj.get("holes", 1))
        spec = WarmupSpec.fixed(rows, holes=holes, spawn_buffer=spawn_buffer_i)

    elif t in {"uniform_rows", "uniform"}:
        min_rows = int(obj["min_rows"])
        max_rows = int(obj["max_rows"])
        holes = int(obj.get("holes", 1))
        spec = WarmupSpec.uniform_rows(min_rows, max_rows, holes=holes, spawn_buffer=spawn_buffer_i)

    elif t == "poisson":
        lam_raw = obj.get("lambda", obj.get("lambda_", None))
        if lam_raw is None:
            raise KeyError("warmup.type=poisson requires key 'lambda' (or 'lambda_')")
        lam = float(lam_raw)
        cap = int(obj["cap"])
        holes = int(obj.get("holes", 1))
        spec = WarmupSpec.poisson(lam, cap, holes=holes, spawn_buffer=spawn_buffer_i)

    elif t in {"base_plus_poisson", "base+poisson"}:
        base = int(obj["base"])
        lam_raw = obj.get("lambda", obj.get("lambda_", None))
        if lam_raw is None:
            raise KeyError("warmup.type=base_plus_poisson requires key 'lambda' (or 'lambda_')")
        lam = float(lam_raw)
        cap = int(obj["cap"])
        holes = int(obj.get("holes", 1))
        spec = WarmupSpec.base_plus_poisson(base, lam, cap, holes=holes, spawn_buffer=spawn_buffer_i)

    else:
        raise ValueError(f"unknown warmup.type={t!r}")

    # optional: make holes uniform after base spec was built
    uh = obj.get("uniform_holes", None)
    if uh is not None:
        if not isinstance(uh, dict):
            raise TypeError("warmup.uniform_holes must be a dict {min,max}")
        spec = spec.with_uniform_holes(int(uh["min"]), int(uh["max"]))

    return spec


def make_game_from_cfg(cfg: Dict[str, Any]) -> Any:
    """
    Construct the Rust engine wrapper.

    NOTE: Engine stores (piece_rule, warmup) as defaults and reuses them on reset()
    unless reset() is called with explicit overrides. So the env only needs to pass
    seed on each reset.
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

    warmup_spec = _parse_warmup_spec(game_cfg.get("warmup", None))

    return TetrisEngine(seed=seed, piece_rule=piece_rule, warmup=warmup_spec)
