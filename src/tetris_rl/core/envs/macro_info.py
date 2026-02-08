# src/tetris_rl/core/envs/macro_info.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from tetris_rl.core.envs.actions import ActionRequest, MaskStats
from tetris_rl.core.envs.api import TransitionFeatures

# Env-side HUD rows (right column of sidebar STATS panel).
ENV_SIDEBAR_ROWS_ORDER = (
    "Invalid",
    "ΔHoles",
    "ΔMaxH",
    "ΔBump",
    "Bumpy",
)


@dataclass(frozen=True)
class FeatureBlock:
    holes: Optional[int] = None
    max_height: Optional[int] = None
    bumpiness: Optional[int] = None
    agg_height: Optional[int] = None
    delta_holes: Optional[int] = None
    delta_max_height: Optional[int] = None
    delta_bumpiness: Optional[int] = None
    delta_agg_height: Optional[int] = None

    @staticmethod
    def _get_int(obj: Mapping[str, Any] | None, key: str) -> Optional[int]:
        if obj is None:
            return None
        try:
            return int(obj.get(key))  # type: ignore[call-arg]
        except Exception:
            return None

    @classmethod
    def from_step(
        cls,
        cur: Mapping[str, Any] | None,
        delta: Mapping[str, Any] | None,
    ) -> "FeatureBlock":
        return cls(
            holes=cls._get_int(cur, "holes"),
            max_height=cls._get_int(cur, "max_h"),
            bumpiness=cls._get_int(cur, "bump"),
            agg_height=cls._get_int(cur, "agg_h"),
            delta_holes=cls._get_int(delta, "d_holes"),
            delta_max_height=cls._get_int(delta, "d_max_h"),
            delta_bumpiness=cls._get_int(delta, "d_bump"),
            delta_agg_height=cls._get_int(delta, "d_agg_h"),
        )

    def as_prev_tuple(self) -> Optional[tuple[int, int, int, int]]:
        if (
            self.max_height is None
            or self.agg_height is None
            or self.holes is None
            or self.bumpiness is None
        ):
            return None
        return (
            int(self.max_height),
            int(self.agg_height),
            int(self.holes),
            int(self.bumpiness),
        )


@dataclass
class TransitionFeaturesBuilder:
    cleared: int
    terminated: bool
    placed_kind: str
    requested_rot: int
    requested_col: int
    used_rot: int
    used_col: int
    used_action_id: int
    applied: bool
    invalid_action: bool
    invalid_action_policy: Optional[str]
    masked_action: bool
    feature_clear_mode: str

    delta_holes: Optional[int] = None
    delta_max_height: Optional[int] = None
    delta_bumpiness: Optional[int] = None
    delta_agg_height: Optional[int] = None
    holes_after: Optional[int] = None
    max_height_after: Optional[int] = None
    bumpiness_after: Optional[int] = None
    agg_height_after: Optional[int] = None

    def with_post_features(
        self,
        *,
        holes: Optional[int],
        max_height: Optional[int],
        bumpiness: Optional[int],
        agg_height: Optional[int],
        delta_holes: Optional[int],
        delta_max_height: Optional[int],
        delta_bumpiness: Optional[int],
        delta_agg_height: Optional[int],
    ) -> "TransitionFeaturesBuilder":
        self.holes_after = holes
        self.max_height_after = max_height
        self.bumpiness_after = bumpiness
        self.agg_height_after = agg_height
        self.delta_holes = delta_holes
        self.delta_max_height = delta_max_height
        self.delta_bumpiness = delta_bumpiness
        self.delta_agg_height = delta_agg_height
        return self

    def with_post_block(self, block: FeatureBlock | None) -> "TransitionFeaturesBuilder":
        if block is None:
            return self
        return self.with_post_features(
            holes=block.holes,
            max_height=block.max_height,
            bumpiness=block.bumpiness,
            agg_height=block.agg_height,
            delta_holes=block.delta_holes,
            delta_max_height=block.delta_max_height,
            delta_bumpiness=block.delta_bumpiness,
            delta_agg_height=block.delta_agg_height,
        )

    def build(self) -> TransitionFeatures:
        return build_transition_features(**self.__dict__)


@dataclass
class StepInfoBuilder:
    invalid_action: bool
    invalid_action_policy: Optional[str]
    applied: bool
    mask_mismatch: bool
    game_over: bool
    delta_score: Optional[float]
    state: Any
    cleared: int
    action_mode: str
    requested_rot: int
    requested_col: int
    requested_action_id: int
    used_rot: int
    used_col: int
    masked_action: bool
    action_dim: Optional[int] = None
    masked_action_count: Optional[int] = None
    episode_idx: int = 0
    episode_step: int = 0
    piece_rule: str = ""

    holes_after: Optional[int] = None
    delta_holes: Optional[int] = None
    max_height_after: Optional[int] = None
    delta_max_height: Optional[int] = None
    bumpiness_after: Optional[int] = None
    delta_bumpiness: Optional[int] = None
    agg_height_after: Optional[int] = None
    delta_agg_height: Optional[int] = None

    sidebar_env: Optional[list[tuple[str, Any]]] = None
    engine_info: Optional[Dict[str, Any]] = None

    def with_post_features(
        self,
        *,
        holes: Optional[int],
        max_height: Optional[int],
        bumpiness: Optional[int],
        agg_height: Optional[int],
        delta_holes: Optional[int],
        delta_max_height: Optional[int],
        delta_bumpiness: Optional[int],
        delta_agg_height: Optional[int],
    ) -> "StepInfoBuilder":
        self.holes_after = holes
        self.max_height_after = max_height
        self.bumpiness_after = bumpiness
        self.agg_height_after = agg_height
        self.delta_holes = delta_holes
        self.delta_max_height = delta_max_height
        self.delta_bumpiness = delta_bumpiness
        self.delta_agg_height = delta_agg_height
        return self

    def with_post_block(self, block: FeatureBlock | None) -> "StepInfoBuilder":
        if block is None:
            return self
        return self.with_post_features(
            holes=block.holes,
            max_height=block.max_height,
            bumpiness=block.bumpiness,
            agg_height=block.agg_height,
            delta_holes=block.delta_holes,
            delta_max_height=block.delta_max_height,
            delta_bumpiness=block.delta_bumpiness,
            delta_agg_height=block.delta_agg_height,
        )

    def build(self) -> Dict[str, Any]:
        return build_step_info_update(**self.__dict__)


@dataclass(frozen=True)
class StepPayload:
    features: TransitionFeatures
    info: Dict[str, Any]
    prev_feat: Optional[tuple[int, int, int, int]]


@dataclass(frozen=True)
class ActionContext:
    requested_rot: int
    requested_col: int
    requested_action_id: int
    used_rot: int
    used_col: int
    used_action_id: int
    masked_action: bool
    mask_mismatch: bool
    masked_action_count: int
    action_dim: int
    action_mode: str

    @classmethod
    def from_env(
        cls,
        env: Any,
        *,
        requested_rot: int,
        requested_col: int,
        requested_action_id: int,
        used_rot: int,
        used_col: int,
        used_action_id: int,
        masked_action: bool,
        mask_mismatch: bool,
        masked_action_count: int,
        action_dim: Optional[int] = None,
    ) -> "ActionContext":
        return cls(
            requested_rot=int(requested_rot),
            requested_col=int(requested_col),
            requested_action_id=int(requested_action_id),
            used_rot=int(used_rot),
            used_col=int(used_col),
            used_action_id=int(used_action_id),
            masked_action=bool(masked_action),
            mask_mismatch=bool(mask_mismatch),
            masked_action_count=int(masked_action_count),
            action_dim=int(action_dim if action_dim is not None else getattr(env, "action_dim", 0)),
            action_mode=str(getattr(env, "action_mode", "")),
        )


@dataclass(frozen=True)
class StepContext:
    applied: bool
    invalid_action: bool
    terminated: bool
    cleared: int
    placed_kind: str
    state: Any
    delta_score: float
    sidebar_env: list[tuple[str, Any]]
    invalid_action_policy: str
    episode_idx: int
    episode_step: int
    action_mode: str
    piece_rule: str
    feature_clear_mode: str

    @classmethod
    def from_env(
        cls,
        env: Any,
        *,
        applied: bool,
        invalid_action: bool,
        terminated: bool,
        cleared: int,
        placed_kind: str,
        state: Any,
        delta_score: float,
        sidebar_env: list[tuple[str, Any]],
    ) -> "StepContext":
        return cls(
            applied=bool(applied),
            invalid_action=bool(invalid_action),
            terminated=bool(terminated),
            cleared=int(cleared),
            placed_kind=str(placed_kind),
            state=state,
            delta_score=float(delta_score),
            sidebar_env=list(sidebar_env),
            invalid_action_policy=str(getattr(env, "invalid_action_policy", "")),
            episode_idx=int(getattr(env, "_episode_idx", 0)),
            episode_step=int(getattr(env, "_steps", 0)),
            action_mode=str(getattr(env, "action_mode", "")),
            piece_rule=str(getattr(env, "_piece_rule_name", lambda: "")()),
            feature_clear_mode=str(getattr(env, "feature_clear_mode", "post")).strip().lower(),
        )


def build_step_payload(
    *,
    sf: Mapping[str, Any],
    action: ActionContext,
    step: StepContext,
    engine_info: Dict[str, Any],
) -> StepPayload:
    cur = _get_field(sf, "cur", {}) or {}
    delta = _get_field(sf, "delta", {}) or {}

    post_block = FeatureBlock.from_step(cur, delta)
    prev_tuple = post_block.as_prev_tuple()

    features = (
        TransitionFeaturesBuilder(
            cleared=int(step.cleared),
            terminated=bool(step.terminated),
            placed_kind=str(step.placed_kind),
            requested_rot=int(action.requested_rot),
            requested_col=int(action.requested_col),
            used_rot=int(action.used_rot),
            used_col=int(action.used_col),
            used_action_id=int(action.used_action_id),
            applied=bool(step.applied),
            invalid_action=bool(step.invalid_action),
            invalid_action_policy=str(step.invalid_action_policy),
            masked_action=bool(action.masked_action),
            feature_clear_mode=str(step.feature_clear_mode),
        )
        .with_post_block(post_block)
        .build()
    )

    info = (
        StepInfoBuilder(
            invalid_action=bool(step.invalid_action),
            invalid_action_policy=str(step.invalid_action_policy),
            applied=bool(step.applied),
            mask_mismatch=bool(action.mask_mismatch),
            game_over=bool(step.terminated),
            delta_score=float(step.delta_score),
            state=step.state,
            cleared=int(step.cleared),
            action_mode=str(step.action_mode),
            requested_rot=int(action.requested_rot),
            requested_col=int(action.requested_col),
            requested_action_id=int(action.requested_action_id),
            used_rot=int(action.used_rot),
            used_col=int(action.used_col),
            masked_action=bool(action.masked_action),
            action_dim=int(action.action_dim),
            masked_action_count=int(action.masked_action_count),
            episode_idx=int(step.episode_idx),
            episode_step=int(step.episode_step),
            piece_rule=str(step.piece_rule),
            sidebar_env=step.sidebar_env,
            engine_info=dict(engine_info),
        )
        .with_post_block(post_block)
        .build()
    )

    return StepPayload(features=features, info=info, prev_feat=prev_tuple)


def build_step_payload_for_env(
    *,
    env: Any,
    sf: Mapping[str, Any],
    requested: ActionRequest,
    mask_stats: MaskStats,
    used_action_id: int,
    used_rot: int,
    used_col: int,
    invalid_action: bool,
    terminated: bool,
    cleared: int,
    prev_state: Mapping[str, Any],
    next_state: Mapping[str, Any],
) -> StepPayload:
    cur = _get_field(sf, "cur", {}) or {}
    delta = _get_field(sf, "delta", {}) or {}
    mask_mismatch = False
    if str(getattr(env, "action_mode", "")) == "discrete":
        mask_mismatch = bool(mask_stats.masked_action) != bool(invalid_action)

    post_block = FeatureBlock.from_step(cur, delta)

    sidebar_env = sidebar_env_rows(
        invalid_action=bool(invalid_action),
        delta_holes=post_block.delta_holes,
        delta_max_height=post_block.delta_max_height,
        delta_bumpiness=post_block.delta_bumpiness,
        bumpiness=post_block.bumpiness,
    )

    prev_score = float(_get_field(prev_state, "score", 0.0))
    next_score = float(_get_field(next_state, "score", 0.0))
    delta_score = float(next_score - prev_score)
    placed_kind = str(_get_field(prev_state, "active_kind", "?"))
    applied = bool((not mask_stats.masked_action) and (not bool(invalid_action)))

    action_ctx = ActionContext.from_env(
        env,
        requested_rot=int(requested.requested_rot),
        requested_col=int(requested.requested_col),
        requested_action_id=int(requested.requested_action_id),
        used_rot=int(used_rot),
        used_col=int(used_col),
        used_action_id=int(used_action_id),
        masked_action=bool(mask_stats.masked_action),
        mask_mismatch=bool(mask_mismatch),
        masked_action_count=int(mask_stats.masked_action_count),
        action_dim=int(mask_stats.action_dim),
    )
    step_ctx = StepContext.from_env(
        env,
        applied=bool(applied),
        invalid_action=bool(invalid_action),
        terminated=bool(terminated),
        cleared=int(cleared),
        placed_kind=str(placed_kind),
        state=next_state,
        delta_score=float(delta_score),
        sidebar_env=sidebar_env,
    )
    return build_step_payload(
        sf=sf,
        action=action_ctx,
        step=step_ctx,
        engine_info={},
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
    used_action_id: int,
    applied: bool,
    invalid_action: bool,
    invalid_action_policy: Optional[str],
    masked_action: bool,
    feature_clear_mode: str,
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
        used_action_id=int(used_action_id),
        applied=bool(applied),
        invalid_action=bool(invalid_action),
        invalid_action_policy=str(invalid_action_policy) if invalid_action_policy is not None else None,
        masked_action=bool(masked_action),
        feature_clear_mode=str(feature_clear_mode),
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
        "applied": bool(applied),
    }
    if action_dim is not None:
        ui["action_dim"] = int(action_dim)
    if masked_action_count is not None:
        ui["masked_action_count"] = int(masked_action_count)

    info: Dict[str, Any] = {}
    info["engine_info"] = dict(engine_info)
    info["tf"] = tf
    info["game"] = game
    info["ui"] = ui
    return info
