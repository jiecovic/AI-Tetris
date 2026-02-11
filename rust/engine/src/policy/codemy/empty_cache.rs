// rust/engine/src/policy/codemy/empty_cache.rs
#![forbid(unsafe_code)]

use std::sync::OnceLock;

use crate::engine::{Game, Kind, H, W};

#[inline]
fn kind_idx0(k: Kind) -> usize {
    match k {
        Kind::I => 0,
        Kind::O => 1,
        Kind::T => 2,
        Kind::S => 3,
        Kind::Z => 4,
        Kind::J => 5,
        Kind::L => 6,
    }
}

static EMPTY_VALID_AIDS: OnceLock<[Vec<usize>; 7]> = OnceLock::new();

/// Action ids that are valid for `kind` on an empty grid.
///
/// Notes:
/// - This is a *superset* of valid moves on non-empty grids (collisions can still invalidate).
/// - Redundant rotation slots are already excluded by `Game::action_mask_for_grid`
///   because the engine enforces `rot_u < kind.num_rots()`.
pub(crate) fn empty_valid_action_ids(kind: Kind) -> &'static [usize] {
    let arr = EMPTY_VALID_AIDS.get_or_init(|| {
        let empty = [[0u8; W]; H];

        let mut out: [Vec<usize>; 7] = [
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ];

        for &k in Kind::all() {
            let mask = Game::action_mask_for_grid(&empty, k);

            // Tighter upper bound than ACTION_DIM; still safe.
            let mut v = Vec::with_capacity(k.num_rots() * W);

            for (aid, is_valid) in mask.iter().enumerate() {
                if *is_valid {
                    v.push(aid);
                }
            }

            out[kind_idx0(k)] = v;
        }

        out
    });

    &arr[kind_idx0(kind)]
}

pub(crate) fn kind_idx0_u8(kind: Kind) -> u8 {
    kind_idx0(kind) as u8
}
