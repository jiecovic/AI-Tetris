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
# Spec dataclasses (NEW-ONLY)
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
class DataGenNoiseSpec:
    enabled: bool = False
    interleave_prob: float = 0.0
    interleave_max_steps: int = 1


@dataclass(frozen=True)
class DataGenLabelsSpec:
    record_rewardfit: bool = False


@dataclass(frozen=True)
class DataGenGenerationSpec:
    episode_max_steps: Optional[int] = None
    noise: DataGenNoiseSpec = field(default_factory=DataGenNoiseSpec)
    labels: DataGenLabelsSpec = field(default_factory=DataGenLabelsSpec)


# -----------------------------------------------------------------------------
# Expert specs (NEW-ONLY)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class DataGenRustExpertParams:
    """
    Rust heuristic policy params (codemy* family).
    Kept flat in YAML under expert: ...
    """
    beam_from_depth: int = 0
    beam_width: Optional[int] = None  # None => use Rust defaults
    tail_weight: float = 0.5          # only meaningful for codemy2fast


@dataclass(frozen=True)
class DataGenExpertSpec:
    """
    New-only:
      expert:
        type: codemy0|codemy1|codemy2|codemy2fast
        beam_from_depth: 0
        beam_width: 10
        tail_weight: 0.5
    """
    type: str = "codemy1"
    rust: DataGenRustExpertParams = field(default_factory=DataGenRustExpertParams)


@dataclass(frozen=True)
class DataGenSpec:
    """
    New-only datagen spec.

    NOTE:
      - Composition keys (specs/env/game) are handled by config.resolve and are NOT part of DataGenSpec.
      - Datagen runtime may still consume cfg.env/cfg.game directly (resolved root), but this
        typed spec is strictly the datagen-owned blocks below.
    """
    dataset: DataGenDatasetSpec = field(default_factory=DataGenDatasetSpec)
    run: DataGenRunSpec = field(default_factory=DataGenRunSpec)
    generation: DataGenGenerationSpec = field(default_factory=DataGenGenerationSpec)
    expert: DataGenExpertSpec = field(default_factory=DataGenExpertSpec)


# -----------------------------------------------------------------------------
# Parser (STRICT at top level; allows composition keys)
# -----------------------------------------------------------------------------


def parse_datagen_spec(*, cfg: Dict[str, Any]) -> DataGenSpec:
    # After resolve_config(), datagen configs may contain:
    #   - specs: { env: ... }
    #   - env:   {...}
    #   - game:  {...}
    # These are composition/runtime wiring keys and are intentionally ignored here.
    root = require_mapping_strict(
        cfg,
        where="cfg",
        allowed_keys={"dataset", "run", "generation", "expert", "specs", "env", "game"},
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
        compression=get_bool(dataset_obj, "compression", default=DataGenDatasetSpec.compression, where="cfg.dataset.compression"),
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
        num_workers=max(1, get_int(run_obj, "num_workers", default=DataGenRunSpec.num_workers, where="cfg.run.num_workers")),
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
    # generation + noise + labels
    # ---------------------------
    gen_obj = require_mapping_strict(
        get_mapping(root, "generation", default={}, where="cfg.generation"),
        where="cfg.generation",
        allowed_keys={"episode_max_steps", "noise", "labels"},
    )

    episode_max_steps_raw = gen_obj.get("episode_max_steps", None)
    if episode_max_steps_raw is None:
        episode_max_steps: Optional[int] = None
    else:
        v = int(get_int(gen_obj, "episode_max_steps", default=0, where="cfg.generation.episode_max_steps"))
        episode_max_steps = v if v > 0 else None

    noise_obj = require_mapping_strict(
        get_mapping(gen_obj, "noise", default={}, where="cfg.generation.noise"),
        where="cfg.generation.noise",
        allowed_keys={"enabled", "interleave_prob", "interleave_max_steps"},
    )

    noise = DataGenNoiseSpec(
        enabled=get_bool(noise_obj, "enabled", default=False, where="cfg.generation.noise.enabled"),
        interleave_prob=clamp_prob(
            get_float(noise_obj, "interleave_prob", default=DataGenNoiseSpec.interleave_prob, where="cfg.generation.noise.interleave_prob")
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
        ),
    )

    generation = DataGenGenerationSpec(
        episode_max_steps=episode_max_steps,
        noise=noise,
        labels=labels,
    )

    # ---------------------------
    # expert (NEW-ONLY flat rust)
    # ---------------------------
    expert_obj = require_mapping_strict(
        get_mapping(root, "expert", default={}, where="cfg.expert"),
        where="cfg.expert",
        allowed_keys={"type", "beam_from_depth", "beam_width", "tail_weight"},
    )

    expert_type = get_str(expert_obj, "type", default=DataGenExpertSpec.type, where="cfg.expert.type").strip().lower()

    beam_from_depth = get_int(
        expert_obj,
        "beam_from_depth",
        default=DataGenRustExpertParams.beam_from_depth,
        where="cfg.expert.beam_from_depth",
    )

    beam_width_raw = expert_obj.get("beam_width", None)
    if beam_width_raw is None:
        beam_width_r: Optional[int] = None
    else:
        beam_width_r = max(1, get_int(expert_obj, "beam_width", default=10, where="cfg.expert.beam_width"))

    tail_weight = float(get_float(expert_obj, "tail_weight", default=DataGenRustExpertParams.tail_weight, where="cfg.expert.tail_weight"))

    expert = DataGenExpertSpec(
        type=str(expert_type),
        rust=DataGenRustExpertParams(
            beam_from_depth=int(beam_from_depth),
            beam_width=beam_width_r,
            tail_weight=float(tail_weight),
        ),
    )

    return DataGenSpec(
        dataset=dataset,
        run=run,
        generation=generation,
        expert=expert,
    )


__all__ = ["DataGenSpec", "parse_datagen_spec"]
