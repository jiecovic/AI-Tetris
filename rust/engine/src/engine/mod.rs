// rust/engine/src/engine/mod.rs
#![forbid(unsafe_code)]

pub mod constants;
pub mod features;
pub mod game;
pub mod geometry;
pub mod grid;
pub mod piece_rule;
pub mod pieces;
pub mod warmup;

// Re-exports (engine public API)
pub use constants::*;
pub use features::{compute_grid_features, compute_step_features, GridDelta, GridFeatures, StepFeatures};
pub use game::*;
pub use piece_rule::*;
pub use pieces::*; // includes Kind, rotations, preview_mask_4x4, etc.
pub use warmup::{HoleCount, RowCountDist, WarmupSpec};
