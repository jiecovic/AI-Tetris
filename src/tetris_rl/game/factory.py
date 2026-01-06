# src/tetris_rl/game/factory.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

from tetris_rl.game.core.game import TetrisGame
from tetris_rl.game.core.piece_rules import (
    BagPieceRule,
    GameBoyOrPieceRule,
    PieceRule,
    UniformPieceRule,
)
from tetris_rl.game.core.pieceset import PieceSet
from tetris_rl.utils.paths import repo_root
from tetris_rl.config.datagen_spec import DataGenGameSpec


def _as_mapping(x: Any) -> Mapping[str, Any] | None:
    return x if isinstance(x, Mapping) else None


def _resolve_pieces_path(spec: Any) -> Path:
    """
    Resolve cfg.game.pieces into an on-disk YAML path.
    """
    if spec is None:
        p = Path(PieceSet.default_classic7_path())
        if not p.is_file():
            raise FileNotFoundError(f"default classic7 pieceset missing: {p}")
        return p

    m = _as_mapping(spec)
    if m is not None:
        if "path" in m:
            return _resolve_pieces_path(m.get("path"))
        if "name" in m:
            return _resolve_pieces_path(m.get("name"))
        raise TypeError("cfg.game.pieces dict must contain 'path' or 'name'")

    s = str(spec).strip()
    if not s:
        p = Path(PieceSet.default_classic7_path())
        if not p.is_file():
            raise FileNotFoundError(f"default classic7 pieceset missing: {p}")
        return p

    key = s.lower()
    if key in {"classic7", "default", "classic"}:
        p = Path(PieceSet.default_classic7_path())
        if not p.is_file():
            raise FileNotFoundError(f"default classic7 pieceset missing: {p}")
        return p

    rr = repo_root()

    looks_like_short_name = ("/" not in s) and ("\\" not in s) and (not s.endswith((".yaml", ".yml")))
    if looks_like_short_name:
        p = (rr / "assets" / "pieces" / f"{s}.yaml").resolve()
    else:
        p0 = Path(s)
        p = (p0 if p0.is_absolute() else (rr / p0)).resolve()

    if not p.is_file():
        raise FileNotFoundError(f"pieceset yaml not found: {p} (from cfg.game.pieces={spec!r})")
    return p


def _make_piece_rule(spec: Any) -> PieceRule:
    """
    Resolve cfg.game.piece_rule to a fresh PieceRule instance.

    Supported rules (canonical names only):
      - "uniform"
      - "k-bag"
      - "gameboy_or"
    """
    if spec is None:
        return UniformPieceRule()

    m = _as_mapping(spec)
    if m is not None:
        name = str(m.get("type", "uniform")).strip().lower()
    else:
        name = str(spec).strip().lower()

    if name == "uniform":
        return UniformPieceRule()

    if name == "k-bag":
        if m is None:
            return BagPieceRule(bag_copies=1)
        bag_copies = int(m.get("bag_copies", 1))
        return BagPieceRule(bag_copies=bag_copies)

    if name == "gameboy_or":
        return GameBoyOrPieceRule()

    raise ValueError(
        f"unknown game.piece_rule: {name!r} "
        "(expected 'uniform', 'k-bag', or 'gameboy_or')"
    )


def make_game_from_cfg(cfg: Dict[str, Any]) -> TetrisGame:
    """
    Canonical cfg -> TetrisGame builder.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a mapping, got {type(cfg)!r}")

    game_cfg = cfg.get("game", {}) or {}
    if not isinstance(game_cfg, dict):
        raise TypeError("cfg.game must be a mapping")

    pieces_path = _resolve_pieces_path(game_cfg.get("pieces", None))
    piece_set = PieceSet.from_yaml(pieces_path)

    piece_rule = _make_piece_rule(game_cfg.get("piece_rule", None))

    kwargs: Dict[str, Any] = {
        "piece_set": piece_set,
        "piece_rule": piece_rule,
    }

    if "visible_height" in game_cfg and game_cfg["visible_height"] is not None:
        kwargs["visible_height"] = int(game_cfg["visible_height"])
    if "spawn_rows" in game_cfg and game_cfg["spawn_rows"] is not None:
        kwargs["spawn_rows"] = int(game_cfg["spawn_rows"])
    if "width" in game_cfg and game_cfg["width"] is not None:
        kwargs["width"] = int(game_cfg["width"])

    return TetrisGame(**kwargs)


def make_game_from_spec(spec: DataGenGameSpec) -> TetrisGame:
    """
    Typed builder (used by datagen and any spec-only pipelines).
    """
    pieces_path = _resolve_pieces_path(spec.pieces)
    piece_set = PieceSet.from_yaml(pieces_path)

    piece_rule = _make_piece_rule(spec.piece_rule)

    kwargs: Dict[str, Any] = {
        "piece_set": piece_set,
        "piece_rule": piece_rule,
    }
    # DataGenGameSpec currently only has pieces + piece_rule.
    # If you later add width/spawn_rows/etc to the spec, wire them here.
    return TetrisGame(**kwargs)