// rust/engine/src/engine/mod.rs
#![forbid(unsafe_code)]

mod constants;
mod features;
mod game;
mod geometry;
mod grid;
mod piece_rule;
mod pieces;
mod warmup;

/**
 * Curated engine public API.
 *
 * Internal implementation modules remain private; only stable items are re-exported here.
 */
pub use constants::{
    ACTION_DIM, DEFAULT_SPAWN_BUFFER, H, HIDDEN_ROWS, MAX_ROTS, VISIBLE_H, W, decode_action_id,
    encode_action_id,
};
pub use features::{
    GridDelta, GridFeatures, StepFeatures, compute_grid_features, compute_grid_features_visible,
    compute_step_features,
};
pub use game::{Game, SimPlacement, StepResult};
pub use piece_rule::{PieceRule, PieceRuleKind};
pub use pieces::{Kind, preview_mask_4x4};
pub use warmup::{HoleCount, RowCountDist, WarmupSpec};
