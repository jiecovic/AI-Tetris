# src/tetris_rl/envs/macro_info.py
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from tetris_rl.envs.api import TransitionFeatures

ENV_SIDEBAR_ROWS_ORDER = (
    "Illegal",
    "RedRot",
    "PClear",  # placed cells cleared (0..4)
    "AllClr",  # whole placed tetromino vanished
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
        illegal_action: bool,
        redundant_rotation: bool,
        placed_cells_cleared: int,
        placed_all_cells_cleared: bool,
        delta_holes: Optional[int],
        delta_max_height: Optional[int],
        delta_bumpiness: Optional[int],
        bumpiness: Optional[int],
) -> list[tuple[str, Any]]:
    rows_map: Dict[str, Any] = {
        "Illegal": bool(illegal_action),
        "RedRot": bool(redundant_rotation),
        "PClear": int(placed_cells_cleared),
        "AllClr": bool(placed_all_cells_cleared),
        "ΔHoles": delta_holes,
        "ΔMaxH": delta_max_height,
        "ΔBump": delta_bumpiness,
        "Bumpy": bumpiness,
    }
    out: list[tuple[str, Any]] = []
    for k in ENV_SIDEBAR_ROWS_ORDER:
        if k in rows_map:
            out.append((k, rows_map[k]))
    return out


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
        placed_cells_cleared: int,
        placed_all_cells_cleared: bool,
        terminated: bool,
        placed_kind: str,
        requested_rot: int,
        requested_col: int,
        used_rot: int,
        used_col: int,
        applied: bool,
        illegal_action: bool,
        illegal_reason: Optional[str],
        remapped: bool,
        remap_policy: Optional[str],
        masked_action: bool,
        redundant_rotation: bool,
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
        placed_cells_cleared=int(placed_cells_cleared),
        placed_all_cells_cleared=bool(placed_all_cells_cleared),
        game_over=bool(terminated),
        placed_kind=str(placed_kind),
        requested_rotation=int(requested_rot),
        requested_column=int(requested_col),
        used_rotation=int(used_rot),
        used_column=int(used_col),
        applied=bool(applied),
        illegal_action=bool(illegal_action),
        illegal_reason=str(illegal_reason) if illegal_reason is not None else None,
        remapped=bool(remapped),
        remap_policy=str(remap_policy) if remap_policy is not None else None,
        masked_action=bool(masked_action),
        redundant_rotation=bool(redundant_rotation),
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
        illegal_action: bool,
        illegal_reason: Optional[str],
        illegal_action_policy: str,
        remapped: bool,
        remap_policy: Optional[str],
        applied: bool,
        mask_mismatch: bool,
        game_over: bool,
        delta_score: Optional[float],
        # --- state / presentation ---
        state: Any,
        cleared: int,
        placed_cells_cleared: int,
        placed_all_cells_cleared: bool,
        action_mode: str,
        requested_rot: int,
        requested_col: int,
        requested_action_id: int,
        used_rot: int,
        used_col: int,
        masked_action: bool,
        redundant_rotation: bool,
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
        "placed_cells_cleared": int(placed_cells_cleared),
        "placed_all_cells_cleared": bool(placed_all_cells_cleared),
        "illegal_action": bool(illegal_action),
        "redundant_rotation": bool(redundant_rotation),
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

    game: Dict[str, Any] = {
        "score": int(_get_field(state, "score", 0)),
        "lines_total": int(_get_field(state, "lines", 0)),
        "level": int(_get_field(state, "level", 0)),
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
        "illegal_reason": str(illegal_reason) if illegal_reason is not None else None,
        "illegal_action_policy": str(illegal_action_policy),
        "remapped": bool(remapped),
        "remap_policy": str(remap_policy) if remap_policy is not None else None,
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
