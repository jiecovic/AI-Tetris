# src/tetris_rl/config/datagen_spec.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from tetris_rl.config.schema_types import (
    clamp_prob,
    get_bool,
    get_float,
    get_int,
    get_mapping,
    get_str,
    require_mapping_strict,
)


# -----------------------------------------------------------------------------
# Spec dataclasses
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class DataGenShardsSpec:
    shard_steps: int = 50_000
    num_shards: int = 1


@dataclass(frozen=True)
class DataGenDatasetSpec:
    name: str = "bc_dataset"
    out_root: str = "datasets/bc"
    shards: DataGenShardsSpec = field(default_factory=DataGenShardsSpec)
    compression: bool = False


@dataclass(frozen=True)
class DataGenRunSpec:
    seed: int = 0
    num_workers: int = 1
    progress_update_every_k: int = 2000  # only relevant when num_workers > 1


@dataclass(frozen=True)
class DataGenGameSpec:
    pieces: str = "classic7"
    piece_rule: str = "k-bag"
    # Action-space geometry is derived from assets (PieceSet.max_rotations()) and board_w at runtime.


# ---------------------------
# Warmup component (datagen)
# ---------------------------

@dataclass(frozen=True)
class DataGenWarmupSpec:
    """
    Warmup policy applied after each reset.

    This is a component spec that maps 1:1 to a WarmupFn via registry instantiation:
      { type: ..., params: {...} }

    Policy:
      - STRICT: only these keys exist
      - warmup is independent of "noise" interleave
    """
    type: str = "poisson_init_rows"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DataGenNoiseSpec:
    """
    Noise injected DURING episode generation (between labeled steps).
    Warmup is not part of this anymore.
    """
    enabled: bool = False
    interleave_prob: float = 0.0
    interleave_max_steps: int = 1


@dataclass(frozen=True)
class DataGenLabelsSpec:
    """
    Controls optional label recording.

    Invariant:
      record_rewardfit == True
        -> legal_mask + phi + delta must ALL be recorded
      record_rewardfit == False
        -> none of them may be present
    """
    record_rewardfit: bool = False


@dataclass(frozen=True)
class DataGenGenerationSpec:
    episode_max_steps: Optional[int] = None
    warmup: Optional[DataGenWarmupSpec] = None
    noise: DataGenNoiseSpec = field(default_factory=DataGenNoiseSpec)
    labels: DataGenLabelsSpec = field(default_factory=DataGenLabelsSpec)


# -----------------------------------------------------------------------------
# Expert specs (typed; NO ad-hoc params parsing in factories)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class DataGenHeuristicWeightsSpec:
    # Mirrors agents.heuristic_agent.HeuristicWeights defaults.
    a_agg_height: float = -0.510066
    b_lines: float = 0.760666
    c_holes: float = -0.35663
    d_bumpiness: float = -0.184483


@dataclass(frozen=True)
class DataGenHeuristicExpertParams:
    # 0 = immediate only, 1 = 1-piece lookahead
    lookahead: int = 1
    # used only when lookahead=1
    beam_width: int = 10
    weights: DataGenHeuristicWeightsSpec = field(default_factory=DataGenHeuristicWeightsSpec)


@dataclass(frozen=True)
class DataGenExpertSpec:
    """
    DataGen expert configuration.

    Policy:
      - expert_spec is the ONLY place where expert semantics are defined.
      - factories are pure wiring; no coercion rules or fallback inference.

    Supported types:
      - "heuristic_agent" / "heuristic"
    """

    type: str = "heuristic_agent"
    heuristic: DataGenHeuristicExpertParams = field(default_factory=DataGenHeuristicExpertParams)


@dataclass(frozen=True)
class DataGenSpec:
    dataset: DataGenDatasetSpec = field(default_factory=DataGenDatasetSpec)
    run: DataGenRunSpec = field(default_factory=DataGenRunSpec)
    game: DataGenGameSpec = field(default_factory=DataGenGameSpec)
    generation: DataGenGenerationSpec = field(default_factory=DataGenGenerationSpec)
    expert: DataGenExpertSpec = field(default_factory=DataGenExpertSpec)


# -----------------------------------------------------------------------------
# Parser (STRICT, canonical-only)
# -----------------------------------------------------------------------------


def _parse_optional_component_spec(
        *,
        parent: Dict[str, Any],
        key: str,
        where: str,
        default_type: str,
) -> Optional[DataGenWarmupSpec]:
    """
    Parse optional component spec of the form:
      key:
        type: <str>
        params: <mapping>

    Returns None if key missing or explicitly null.
    """
    if key not in parent:
        return None

    obj_raw = parent.get(key, None)
    if obj_raw is None:
        return None

    obj = require_mapping_strict(
        get_mapping(parent, key, default={}, where=f"{where}.{key}"),
        where=f"{where}.{key}",
        allowed_keys={"type", "params"},
    )

    typ = get_str(obj, "type", default=default_type, where=f"{where}.{key}.type").strip().lower()
    params = get_mapping(obj, "params", default={}, where=f"{where}.{key}.params")
    # params can be any mapping; we keep it as raw dict to feed instantiate()
    return DataGenWarmupSpec(type=str(typ), params=dict(params))


def parse_datagen_spec(*, cfg: Dict[str, Any]) -> DataGenSpec:
    root = require_mapping_strict(
        cfg,
        where="cfg",
        allowed_keys={"dataset", "run", "game", "generation", "expert"},
    )

    # ---------------------------
    # dataset + shards
    # ---------------------------
    dataset_obj = require_mapping_strict(
        get_mapping(root, "dataset", default={}, where="cfg.dataset"),
        where="cfg.dataset",
        allowed_keys={"name", "out_root", "shards", "compression"},
    )

    shards_obj = require_mapping_strict(
        get_mapping(dataset_obj, "shards", default={}, where="cfg.dataset.shards"),
        where="cfg.dataset.shards",
        allowed_keys={"shard_steps", "num_shards"},
    )

    shards = DataGenShardsSpec(
        shard_steps=max(
            1,
            get_int(
                shards_obj,
                "shard_steps",
                default=DataGenShardsSpec.shard_steps,
                where="cfg.dataset.shards.shard_steps",
            ),
        ),
        num_shards=max(
            1,
            get_int(
                shards_obj,
                "num_shards",
                default=DataGenShardsSpec.num_shards,
                where="cfg.dataset.shards.num_shards",
            ),
        ),
    )

    dataset = DataGenDatasetSpec(
        name=get_str(dataset_obj, "name", default=DataGenDatasetSpec.name, where="cfg.dataset.name"),
        out_root=get_str(dataset_obj, "out_root", default=DataGenDatasetSpec.out_root, where="cfg.dataset.out_root"),
        shards=shards,
        compression=get_bool(
            dataset_obj,
            "compression",
            default=DataGenDatasetSpec.compression,
            where="cfg.dataset.compression",
        ),
    )

    # ---------------------------
    # run
    # ---------------------------
    run_obj = require_mapping_strict(
        get_mapping(root, "run", default={}, where="cfg.run"),
        where="cfg.run",
        allowed_keys={"seed", "num_workers", "progress_update_every_k"},
    )

    run = DataGenRunSpec(
        seed=get_int(run_obj, "seed", default=DataGenRunSpec.seed, where="cfg.run.seed"),
        num_workers=max(
            1,
            get_int(run_obj, "num_workers", default=DataGenRunSpec.num_workers, where="cfg.run.num_workers"),
        ),
        progress_update_every_k=max(
            1,
            get_int(
                run_obj,
                "progress_update_every_k",
                default=DataGenRunSpec.progress_update_every_k,
                where="cfg.run.progress_update_every_k",
            ),
        ),
    )

    # ---------------------------
    # game
    # ---------------------------
    game_obj = require_mapping_strict(
        get_mapping(root, "game", default={}, where="cfg.game"),
        where="cfg.game",
        allowed_keys={"pieces", "piece_rule"},
    )

    game = DataGenGameSpec(
        pieces=get_str(game_obj, "pieces", default=DataGenGameSpec.pieces, where="cfg.game.pieces"),
        piece_rule=get_str(game_obj, "piece_rule", default=DataGenGameSpec.piece_rule, where="cfg.game.piece_rule"),
    )

    # ---------------------------
    # generation + noise + labels
    # ---------------------------
    gen_obj = require_mapping_strict(
        get_mapping(root, "generation", default={}, where="cfg.generation"),
        where="cfg.generation",
        allowed_keys={"episode_max_steps", "warmup", "noise", "labels"},
    )

    episode_max_steps_raw = gen_obj.get("episode_max_steps", None)
    if episode_max_steps_raw is None:
        episode_max_steps: Optional[int] = None
    else:
        v = int(get_int(gen_obj, "episode_max_steps", default=0, where="cfg.generation.episode_max_steps"))
        episode_max_steps = v if v > 0 else None

    warmup = _parse_optional_component_spec(
        parent=gen_obj,
        key="warmup",
        where="cfg.generation",
        default_type=DataGenWarmupSpec.type,
    )

    noise_obj = require_mapping_strict(
        get_mapping(gen_obj, "noise", default={}, where="cfg.generation.noise"),
        where="cfg.generation.noise",
        allowed_keys={"enabled", "interleave_prob", "interleave_max_steps"},
    )

    noise = DataGenNoiseSpec(
        enabled=get_bool(noise_obj, "enabled", default=False, where="cfg.generation.noise.enabled"),
        interleave_prob=clamp_prob(
            get_float(
                noise_obj,
                "interleave_prob",
                default=DataGenNoiseSpec.interleave_prob,
                where="cfg.generation.noise.interleave_prob",
            )
        ),
        interleave_max_steps=max(
            1,
            get_int(
                noise_obj,
                "interleave_max_steps",
                default=DataGenNoiseSpec.interleave_max_steps,
                where="cfg.generation.noise.interleave_max_steps",
            ),
        ),
    )

    labels_obj = require_mapping_strict(
        get_mapping(gen_obj, "labels", default={}, where="cfg.generation.labels"),
        where="cfg.generation.labels",
        allowed_keys={"record_rewardfit"},
    )

    labels = DataGenLabelsSpec(
        record_rewardfit=get_bool(
            labels_obj,
            "record_rewardfit",
            default=DataGenLabelsSpec.record_rewardfit,
            where="cfg.generation.labels.record_rewardfit",
        )
    )

    generation = DataGenGenerationSpec(
        episode_max_steps=episode_max_steps,
        warmup=warmup,
        noise=noise,
        labels=labels,
    )

    # ---------------------------
    # expert (typed)
    # ---------------------------
    expert_obj = require_mapping_strict(
        get_mapping(root, "expert", default={}, where="cfg.expert"),
        where="cfg.expert",
        allowed_keys={"type", "heuristic"},
    )

    expert_type = get_str(expert_obj, "type", default=DataGenExpertSpec.type, where="cfg.expert.type").strip().lower()
    heuristic_obj = require_mapping_strict(
        get_mapping(expert_obj, "heuristic", default={}, where="cfg.expert.heuristic"),
        where="cfg.expert.heuristic",
        allowed_keys={"lookahead", "beam_width", "weights"},
    )

    weights_obj = require_mapping_strict(
        get_mapping(heuristic_obj, "weights", default={}, where="cfg.expert.heuristic.weights"),
        where="cfg.expert.heuristic.weights",
        allowed_keys={"a_agg_height", "b_lines", "c_holes", "d_bumpiness"},
    )

    lookahead_raw = get_int(
        heuristic_obj,
        "lookahead",
        default=DataGenHeuristicExpertParams.lookahead,
        where="cfg.expert.heuristic.lookahead",
    )
    lookahead = 0 if lookahead_raw < 0 else (1 if lookahead_raw > 1 else int(lookahead_raw))

    beam_width = max(
        1,
        get_int(
            heuristic_obj,
            "beam_width",
            default=DataGenHeuristicExpertParams.beam_width,
            where="cfg.expert.heuristic.beam_width",
        ),
    )

    weights = DataGenHeuristicWeightsSpec(
        a_agg_height=get_float(
            weights_obj,
            "a_agg_height",
            default=DataGenHeuristicWeightsSpec.a_agg_height,
            where="cfg.expert.heuristic.weights.a_agg_height",
        ),
        b_lines=get_float(
            weights_obj,
            "b_lines",
            default=DataGenHeuristicWeightsSpec.b_lines,
            where="cfg.expert.heuristic.weights.b_lines",
        ),
        c_holes=get_float(
            weights_obj,
            "c_holes",
            default=DataGenHeuristicWeightsSpec.c_holes,
            where="cfg.expert.heuristic.weights.c_holes",
        ),
        d_bumpiness=get_float(
            weights_obj,
            "d_bumpiness",
            default=DataGenHeuristicWeightsSpec.d_bumpiness,
            where="cfg.expert.heuristic.weights.d_bumpiness",
        ),
    )

    expert = DataGenExpertSpec(
        type=str(expert_type),
        heuristic=DataGenHeuristicExpertParams(
            lookahead=int(lookahead),
            beam_width=int(beam_width),
            weights=weights,
        ),
    )

    return DataGenSpec(
        dataset=dataset,
        run=run,
        game=game,
        generation=generation,
        expert=expert,
    )
