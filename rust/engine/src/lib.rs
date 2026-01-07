// rust/engine/src/lib.rs
#![forbid(unsafe_code)]

pub mod engine;
pub mod policy;

// Re-export the bits the Python bindings need:
pub use engine::{
    decode_action_id, encode_action_id, Game, PieceRuleKind, ACTION_DIM, H, HIDDEN_ROWS, W,
};

// Re-export expert policy types for Python bindings / downstream tooling:
pub use policy::{BeamConfig, Codemy0, Codemy1, Codemy2, Codemy2FastPolicy, Policy};
