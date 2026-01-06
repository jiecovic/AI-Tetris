# src/tetris_rl/runs/hud_adapter.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class HudStep:
    cleared_lines: int
    score: int
    delta_score: float
    game_over: bool
    illegal_action: bool

    masked_action: bool
    redundant_rotation: bool

    # entropy/hist input: requested joint id and its dimension
    action_id: Optional[int]
    action_dim: Optional[int]

    action_mode: str
    next_kind: str
    piece_rule: str
    episode_idx: int
    episode_step: int


def _as_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _as_nested(d: Dict[str, Any], key: str) -> Dict[str, Any]:
    v = d.get(key)
    return v if isinstance(v, dict) else {}


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _safe_int_opt(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _safe_bool(x: Any) -> bool:
    return bool(x)


def _safe_str(x: Any, default: str = "?") -> str:
    s = str(x).strip() if x is not None else ""
    return s if s else default


def from_info(info: Any) -> HudStep:
    d = _as_dict(info)

    tf = _as_nested(d, "tf")
    game = _as_nested(d, "game")
    ui = _as_nested(d, "ui")

    if tf and game and ui:
        # IMPORTANT: env emits requested_action_id (not action_id)
        action_id = _safe_int_opt(ui.get("requested_action_id"))
        action_dim = _safe_int_opt(ui.get("action_dim"))

        if action_dim is not None and action_dim <= 0:
            action_dim = None

        if action_id is not None and action_dim is not None:
            if action_id < 0 or action_id >= action_dim:
                action_id = None

        return HudStep(
            cleared_lines=_safe_int(tf.get("cleared_lines", 0), 0),
            score=_safe_int(game.get("score", 0), 0),
            delta_score=_safe_float(game.get("delta_score", 0.0), 0.0),
            game_over=_safe_bool(tf.get("game_over", False)),
            illegal_action=_safe_bool(tf.get("illegal_action", False)),
            masked_action=_safe_bool(ui.get("masked_action", False)),
            redundant_rotation=_safe_bool(tf.get("redundant_rotation", False)),
            action_id=action_id,
            action_dim=action_dim,
            action_mode=_safe_str(ui.get("action_mode", "?"), "?"),
            next_kind=_safe_str(ui.get("next_kind", "?"), "?"),
            piece_rule=_safe_str(ui.get("piece_rule", "?"), "?"),
            episode_idx=_safe_int(ui.get("episode_idx", 0), 0),
            episode_step=_safe_int(ui.get("episode_step", 0), 0),
        )

    return HudStep(
        cleared_lines=0,
        score=_safe_int(d.get("score", 0), 0),
        delta_score=0.0,
        game_over=_safe_bool(d.get("game_over", False)),
        illegal_action=False,
        masked_action=False,
        redundant_rotation=False,
        action_id=None,
        action_dim=None,
        action_mode=_safe_str(d.get("action_mode", "?"), "?"),
        next_kind=_safe_str(d.get("next_kind", "?"), "?"),
        piece_rule=_safe_str(d.get("piece_rule", "?"), "?"),
        episode_idx=_safe_int(d.get("episode_idx", 0), 0),
        episode_step=_safe_int(d.get("episode_step", 0), 0),
    )


def env_info_for_renderer(info: Any) -> Optional[Dict[str, Any]]:
    d = _as_dict(info)
    ui = d.get("ui")
    if isinstance(ui, dict):
        return ui
    return d if d else None
