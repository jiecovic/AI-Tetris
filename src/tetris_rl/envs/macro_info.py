# src/tetris_rl/env_bundles/macro_info.py
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from tetris_rl.envs.api import TransitionFeatures

# Env-side HUD rows (right column of sidebar STATS panel).
ENV_SIDEBAR_ROWS_ORDER = (
    "Invalid",
    "ΔHoles",
    "ΔMaxH",
    "ΔBump",
    "Bumpy",
)


def _is_mapping(x: Any) -> bool:
    return isinstance(x, Mapping)


def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    if _is_mapping(obj):
        return obj.get(key, default)  # type: ignore[call-arg]
    return getattr(obj, key, default)


def sidebar_env_rows(
    *,
    invalid_action: bool,
    delta_holes: Optional[int],
    delta_max_height: Optional[int],
    delta_bumpiness: Optional[int],
    bumpiness: Optional[int],
) -> list[tuple[str, Any]]:
    rows_map: Dict[str, Any] = {
        "Invalid": bool(invalid_action),
        "ΔHoles": delta_holes,
        "ΔMaxH": delta_max_height,
        "ΔBump": delta_bumpiness,
        "Bumpy": bumpiness,
    }
    return [(k, rows_map[k]) for k in ENV_SIDEBAR_ROWS_ORDER if k in rows_map]


def build_reset_info(
    *,
    state: Any,
    episode_idx: int,
    episode_step: int,
    action_mode: str,
    piece_rule: str,
) -> Dict[str, Any]:
    # Works with both legacy State objects and Rust snapshot dicts.
    return {
        "score": int(_get_field(state, "score", 0)),
        "lines": int(_get_field(state, "lines", 0)),
        "level": int(_get_field(state, "level", 0)),
        "active_kind": _get_field(state, "active_kind", None),
        "next_kind": _get_field(state, "next_kind", None),
        "active_kind_idx": int(_get_field(state, "active_kind_idx", 0)),
        "next_kind_idx": int(_get_field(state, "next_kind_idx", 0)),
        "episode_idx": int(episode_idx),
        "episode_step": int(episode_step),
        "action_mode": str(action_mode),
        "piece_rule": str(piece_rule),
    }


def build_transition_features(
    *,
    cleared: int,
    terminated: bool,
    placed_kind: str,
    requested_rot: int,
    requested_col: int,
    used_rot: int,
    used_col: int,
    applied: bool,
    invalid_action: bool,
    invalid_action_policy: Optional[str],
    masked_action: bool,
    delta_holes: Optional[int],
    delta_max_height: Optional[int],
    delta_bumpiness: Optional[int],
    delta_agg_height: Optional[int],
    holes_after: Optional[int],
    max_height_after: Optional[int],
    bumpiness_after: Optional[int],
    agg_height_after: Optional[int],
) -> TransitionFeatures:
    return TransitionFeatures(
        cleared_lines=int(cleared),
        game_over=bool(terminated),
        placed_kind=str(placed_kind),
        requested_rotation=int(requested_rot),
        requested_column=int(requested_col),
        used_rotation=int(used_rot),
        used_column=int(used_col),
        applied=bool(applied),
        invalid_action=bool(invalid_action),
        invalid_action_policy=str(invalid_action_policy) if invalid_action_policy is not None else None,
        masked_action=bool(masked_action),
        delta_holes=delta_holes,
        delta_max_height=delta_max_height,
        delta_bumpiness=delta_bumpiness,
        delta_agg_height=delta_agg_height,
        holes=holes_after,
        max_height=max_height_after,
        bumpiness=bumpiness_after,
        agg_height=agg_height_after,
    )


def build_step_info_update(
    *,
    # --- env truth (must be passed in) ---
    invalid_action: bool,
    remapped: bool,
    invalid_action_policy: Optional[str],
    applied: bool,
    mask_mismatch: bool,
    game_over: bool,
    delta_score: Optional[float],
    # --- state / presentation ---
    state: Any,
    cleared: int,
    action_mode: str,
    requested_rot: int,
    requested_col: int,
    requested_action_id: int,
    used_rot: int,
    used_col: int,
    masked_action: bool,
    action_dim: Optional[int] = None,
    masked_action_count: Optional[int] = None,
    episode_idx: int,
    episode_step: int,
    piece_rule: str,
    holes_after: Optional[int] = None,
    delta_holes: Optional[int] = None,
    max_height_after: Optional[int] = None,
    delta_max_height: Optional[int] = None,
    bumpiness_after: Optional[int] = None,
    delta_bumpiness: Optional[int] = None,
    agg_height_after: Optional[int] = None,
    delta_agg_height: Optional[int] = None,
    sidebar_env: Optional[list[tuple[str, Any]]] = None,
    engine_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    sidebar_env = sidebar_env or []
    engine_info = engine_info or {}

    tf: Dict[str, Any] = {
        "cleared_lines": int(cleared),
        "invalid_action": bool(invalid_action),
        "game_over": bool(game_over),
        "holes": holes_after,
        "delta_holes": delta_holes,
        "max_height": max_height_after,
        "delta_max_height": delta_max_height,
        "bumpiness": bumpiness_after,
        "delta_bumpiness": delta_bumpiness,
        "agg_height": agg_height_after,
        "delta_agg_height": delta_agg_height,
    }

    # GAME panel: store "game-ish" metrics here so the sidebar can show them
    # without needing an extra game_metrics arg plumbed through the renderer.
    game: Dict[str, Any] = {
        "score": int(_get_field(state, "score", 0)),
        "lines_total": int(_get_field(state, "lines", 0)),
        "level": int(_get_field(state, "level", 0)),
        "holes": holes_after,
        "max_height": max_height_after,
        # Optional (keep commented unless you want them in GAME column too):
        # "bumpiness": bumpiness_after,
        # "agg_height": agg_height_after,
    }
    if delta_score is not None:
        game["delta_score"] = float(delta_score)

    ui: Dict[str, Any] = {
        "sidebar_env": sidebar_env,
        "active_kind": _get_field(state, "active_kind", None),
        "next_kind": _get_field(state, "next_kind", None),
        "active_kind_idx": int(_get_field(state, "active_kind_idx", 0)),
        "next_kind_idx": int(_get_field(state, "next_kind_idx", 0)),
        "action_mode": str(action_mode),
        "piece_rule": str(piece_rule),
        "episode_idx": int(episode_idx),
        "episode_step": int(episode_step),
        "requested_rotation": int(requested_rot),
        "requested_column": int(requested_col),
        "requested_action_id": int(requested_action_id) if requested_action_id is not None else None,
        "used_rotation": int(used_rot),
        "used_column": int(used_col),
        "masked_action": bool(masked_action),
        "mask_mismatch": bool(mask_mismatch),
        "invalid_action_policy": str(invalid_action_policy),
        "remapped": bool(remapped),
        "applied": bool(applied),
    }
    if action_dim is not None:
        ui["action_dim"] = int(action_dim)
    if masked_action_count is not None:
        ui["masked_action_count"] = int(masked_action_count)

    info: Dict[str, Any] = {}
    info.update(engine_info)
    info["tf"] = tf
    info["game"] = game
    info["ui"] = ui
    return info
