// src/policy/base.rs
#![forbid(unsafe_code)]

use crate::engine::Game;

/// Policy chooses a placement for the current state.
/// Returns (rot, bbox-left column), or None if no legal action exists.
pub trait Policy {
    fn choose_action(&mut self, g: &Game) -> Option<(usize, i32)>;
}
