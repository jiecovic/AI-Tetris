// rust/engine/src/engine/geometry.rs
#![forbid(unsafe_code)]

use crate::engine::constants::W;
use crate::engine::pieces::{rotations, Kind};

/// Return (min_dx, max_dx) across the 4 blocks of the rotated piece.
///
/// NOTE:
/// - `rot` must be a *distinct* rotation index for this `kind`:
///   0 <= rot < kind.num_rots()
/// - Rotation-slot validity is enforced at call sites (mask/kernel/step).
#[inline]
pub fn dx_range(kind: Kind, rot: usize) -> (i32, i32) {
    debug_assert!(
        rot < rotations(kind).len(),
        "rot out of range for kind {:?}: rot={} num_rots={}",
        kind,
        rot,
        rotations(kind).len()
    );

    let cells = rotations(kind)[rot];
    let mut mn = i32::MAX;
    let mut mx = i32::MIN;
    for &(dx, _dy) in cells {
        mn = mn.min(dx);
        mx = mx.max(dx);
    }
    (mn, mx)
}

/// Returns (min_dx, bbox_w, bbox_left_max).
#[inline]
pub fn bbox_params(kind: Kind, rot: usize) -> (i32, i32, i32) {
    let (min_dx, max_dx) = dx_range(kind, rot);
    let bbox_w = max_dx - min_dx + 1;
    let bbox_left_max = (W as i32) - bbox_w;
    (min_dx, bbox_w, bbox_left_max)
}

/// Convert bbox-left column coordinate into the anchor x coordinate used by rotations().
#[inline]
pub fn bbox_left_to_anchor_x(kind: Kind, rot: usize, bbox_left_col: i32) -> i32 {
    let (min_dx, _bbox_w, _bbox_left_max) = bbox_params(kind, rot);
    bbox_left_col - min_dx
}
