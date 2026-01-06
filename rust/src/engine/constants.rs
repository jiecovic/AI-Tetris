// src/engine/constants.rs
#![forbid(unsafe_code)]

pub const H: usize = 20;
pub const W: usize = 10;

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
