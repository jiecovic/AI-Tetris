// src/policy/codemy/fast.rs
#![forbid(unsafe_code)]

use rustc_hash::FxHashMap;

use crate::engine::{Game, ACTION_DIM, H, W};
use crate::policy::base::Policy;

use super::core::CodemyCore;

/// "codemy2fast": codemy1 exact best-response to known next piece,
/// plus a cheap one-step tail estimate for the unknown next-next piece.
/// No beam needed; uses per-decision caching.
pub struct Codemy2FastPolicy {
    tail_weight: f64,
}

impl Codemy2FastPolicy {
    pub fn new(tail_weight: f64) -> Self {
        Self { tail_weight }
    }
}

impl Policy for Codemy2FastPolicy {
    fn choose_action(&mut self, g: &Game) -> Option<usize> {
        let core = CodemyCore::new(None);

        // Per-decision caches:
        //  - tail_cache: grid_hash -> best leaf scores for each kind (7)
        //  - br_cache: (grid_hash, kind) -> (best_aid, best_score_lock, best_grid_after_clear)
        let mut tail_cache: FxHashMap<u64, [f64; 7]> = FxHashMap::default();
        let mut br_cache: FxHashMap<(u64, u8), (usize, f64, [[u8; W]; H])> =
            FxHashMap::default();

        let mask1 = g.action_mask();
        let mut best: Option<(usize, f64)> = None;

        for aid0 in 0..ACTION_DIM {
            if !mask1[aid0] {
                continue;
            }
            let sim1 = g.simulate_action_id_active(aid0);
            if sim1.invalid {
                continue;
            }

            let grid1 = &sim1.grid_after_clear;

            let Some((_aid1_star, best1_score_lock, grid2)) =
                core.best_response_for_known_piece_cached(grid1, g.next, &mut br_cache)
            else {
                continue;
            };

            let tail = core.tail_uniform_cached(&grid2, &mut tail_cache);
            let v0 = best1_score_lock + self.tail_weight * tail;

            match best {
                None => best = Some((aid0, v0)),
                Some((_ba, bv)) if v0 > bv => best = Some((aid0, v0)),
                _ => {}
            }
        }

        best.map(|(aid, _)| aid)
    }
}
