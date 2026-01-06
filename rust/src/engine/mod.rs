// src/engine/mod.rs
#![forbid(unsafe_code)]

pub mod constants;
pub mod game;
pub mod geometry;
pub mod grid;

pub mod pieces;
pub mod piece_rule;

pub use constants::{decode_action_id, encode_action_id, ACTION_DIM, H, MAX_ROTS, W};
pub use game::{Game, SimPlacement, StepResult};
pub use pieces::{rotations, Kind};
pub use piece_rule::{PieceRule, PieceRuleKind};
