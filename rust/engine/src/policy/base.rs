// rust/engine/src/policy/base.rs
#![forbid(unsafe_code)]

use crate::engine::Game;

/// Policy chooses a placement for the current state.
///
/// Returns `action_id` in `[0, ACTION_DIM)`, or `None` if no legal action exists.
///
/// Object-safe so it can be used as `Box<dyn Policy>`.
pub trait Policy {
    fn choose_action(&mut self, g: &Game) -> Option<usize>;
}
