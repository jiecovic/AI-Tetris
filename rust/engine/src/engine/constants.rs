// src/engine/constants.rs
#![forbid(unsafe_code)]

pub const H: usize = 22;
pub const W: usize = 10;

pub const HIDDEN_ROWS: usize = 2;
pub const VISIBLE_H: usize = H - HIDDEN_ROWS;

/**
 * Default number of top rows that must remain empty at episode start (warmup headroom).
 * - Must be >= HIDDEN_ROWS to guarantee spawn rows are empty.
 * - Using a small extra margin (e.g. +2) makes starts less "immediately cramped".
 */
pub const DEFAULT_SPAWN_BUFFER: usize = HIDDEN_ROWS + 2;

pub const MAX_ROTS: usize = 4;
pub const ACTION_DIM: usize = MAX_ROTS * W;

#[inline]
pub fn encode_action_id(rot: usize, col: usize) -> usize {
    debug_assert!(rot < MAX_ROTS);
    debug_assert!(col < W);
    rot * W + col
}

#[inline]
pub fn decode_action_id(aid: usize) -> (usize, usize) {
    (aid / W, aid % W)
}
