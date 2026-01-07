// rust_engine/src/lib.rs
#![forbid(unsafe_code)]

pub mod engine;
pub mod policy;

// Re-export the bits the Python bindings need:
pub use engine::{
    Game, PieceRuleKind, ACTION_DIM, H, W, HIDDEN_ROWS,
    decode_action_id, encode_action_id,
};
