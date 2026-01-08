# src/tetris_rl/config/env_spec.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from tetris_rl.config.schema_types import require_mapping_strict


@dataclass(frozen=True)
class EnvSpec:
    """
    Env-bundle spec loaded from specs.env.

    New-only contract:
      - must contain top-level keys: env, game
      - env and game are separate namespaces in the composed root cfg

    This parser only validates presence/types. Deeper validation is done by
    env/game factories and typed spec parsers.
    """
    env: Dict[str, Any]
    game: Dict[str, Any]


def parse_env_spec(*, obj: Dict[str, Any], where: str = "specs.env") -> EnvSpec:
    root = require_mapping_strict(obj, where=where, allowed_keys={"env", "game"})

    env = root.get("env", None)
    if not isinstance(env, dict):
        raise TypeError(f"{where}.env must be a mapping")

    game = root.get("game", None)
    if not isinstance(game, dict):
        raise TypeError(f"{where}.game must be a mapping")

    return EnvSpec(env=dict(env), game=dict(game))


__all__ = ["EnvSpec", "parse_env_spec"]
