// rust/engine/src/lib.rs
#![forbid(unsafe_code)]

/*!
Stable public API contract for `tetris_engine`.

This crate intentionally exposes a curated root-level API only. Internal module
layout is private implementation detail and may change without notice.

Public engine surface:
- Core state/types: `Game`, `SimPlacement`, `StepResult`
- Geometry/constants: `H`, `W`, `HIDDEN_ROWS`, `ACTION_DIM`
- Piece/warmup config: `Kind`, `PieceRuleKind`, `WarmupSpec`, `RowCountDist`, `HoleCount`
- Feature types/functions:
  - `GridFeatures`, `GridDelta`, `StepFeatures`
  - `compute_grid_features`, `compute_grid_features_visible`, `compute_step_features`
- Action-id helpers: `encode_action_id`, `decode_action_id`
- UI helper: `preview_mask_4x4`

Public policy surface:
- Trait/config: `Policy`, `BeamConfig`
- Policy entrypoints: `RandomPolicy`, `HeuristicPolicy`, `CodemyPolicy`, `Codemy0/1/2`, `Codemy2FastPolicy`
- Heuristic support: `HeuristicFeature`, `HeuristicBuildError`, `compute_feature_values`

Compatibility guidance:
- Depend on items re-exported from crate root (`tetris_engine::...`).
- Do not depend on internal module paths or private types.
*/
mod engine;
mod policy;

pub use engine::{
    ACTION_DIM, Game, GridDelta, GridFeatures, H, HIDDEN_ROWS, HoleCount, Kind, PieceRuleKind,
    RowCountDist, SimPlacement, StepFeatures, StepResult, W, WarmupSpec, compute_grid_features,
    compute_grid_features_visible, compute_step_features, decode_action_id, encode_action_id,
    preview_mask_4x4,
};

pub use policy::{
    BeamConfig, Codemy0, Codemy1, Codemy2, Codemy2FastPolicy, CodemyPolicy, HeuristicBuildError,
    HeuristicFeature, HeuristicPolicy, Policy, RandomPolicy, compute_feature_values,
};
