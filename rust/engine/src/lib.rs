// rust/engine/src/lib.rs
#![forbid(unsafe_code)]

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
