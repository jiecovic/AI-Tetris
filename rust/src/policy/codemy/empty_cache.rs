// src/policy/codemy/empty_cache.rs
#![forbid(unsafe_code)]

use std::sync::OnceLock;

use crate::engine::{Game, Kind, ACTION_DIM, H, W};

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

static EMPTY_LEGAL_AIDS: OnceLock<[Vec<usize>; 7]> = OnceLock::new();

pub(crate) fn empty_legal_action_ids(kind: Kind) -> &'static [usize] {
    let arr = EMPTY_LEGAL_AIDS.get_or_init(|| {
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
            let mut v = Vec::new();
            v.reserve(ACTION_DIM);

            for aid in 0..ACTION_DIM {
                if mask[aid] {
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
