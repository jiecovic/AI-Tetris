// src/policy/codemy/core.rs
#![forbid(unsafe_code)]

use std::collections::HashMap;

use crate::engine::{Game, Kind, ACTION_DIM, H, W};

use crate::policy::beam::{prune_top_n_scores, BeamConfig};

use super::empty_cache::{empty_legal_action_ids, kind_idx0_u8};
use super::score::score_grid;
use super::unknown::UnknownModel;

/// Core implementation shared by dynamic + static policy wrappers.
/// Holds all knobs (currently only beam pruning).
#[derive(Clone, Copy, Debug)]
pub struct CodemyCore {
    pub(crate) beam: Option<BeamConfig>,
}

impl CodemyCore {
    pub(crate) fn new(beam: Option<BeamConfig>) -> Self {
        Self { beam }
    }

    #[inline]
    fn should_prune(&self, depth: u8) -> Option<usize> {
        let b = self.beam?;
        if depth >= b.beam_from_depth {
            Some(b.beam_width.max(1))
        } else {
            None
        }
    }

    /// Fast path: maximize score_grid(grid_after_lock) for a known piece on a grid.
    /// Single simulation per candidate aid. No allocations.
    fn best_leaf_score_for_known_piece(&self, grid: &[[u8; W]; H], kind: Kind) -> f64 {
        let mut best = f64::NEG_INFINITY;

        for &aid in empty_legal_action_ids(kind) {
            let sim = Game::apply_action_id_to_grid(grid, kind, aid);
            if sim.terminated {
                continue;
            }
            let s = score_grid(&sim.grid_after_lock);
            if s > best {
                best = s;
            }
        }

        best
    }

    /// Leaf with beam pruning:
    /// compute scores for all candidate actions, select top-N, then return the max score among them.
    fn best_leaf_score_for_known_piece_beam(
        &self,
        grid: &[[u8; W]; H],
        kind: Kind,
        n: usize,
    ) -> f64 {
        let mut scores: Vec<(usize, f64)> = Vec::new();

        for &aid in empty_legal_action_ids(kind) {
            let sim = Game::apply_action_id_to_grid(grid, kind, aid);
            if sim.terminated {
                continue;
            }
            scores.push((aid, score_grid(&sim.grid_after_lock)));
        }

        if scores.is_empty() {
            return f64::NEG_INFINITY;
        }

        let kept = prune_top_n_scores(scores, n);

        let mut best = f64::NEG_INFINITY;
        for (_aid, s) in kept {
            if s > best {
                best = s;
            }
        }
        best
    }

    /// Value when the next piece is known: maximize over placements of `kind` on `grid`.
    pub(crate) fn value_known_piece<M: UnknownModel>(
        &self,
        grid: &[[u8; W]; H],
        kind: Kind,
        plies_left: u8,
        depth: u8,
    ) -> f64 {
        debug_assert!(plies_left >= 1);

        // Leaf
        if plies_left == 1 {
            if let Some(n) = self.should_prune(depth) {
                return self.best_leaf_score_for_known_piece_beam(grid, kind, n);
            }
            return self.best_leaf_score_for_known_piece(grid, kind);
        }

        // No pruning => single pass
        let Some(_n) = self.should_prune(depth) else {
            let mut best = f64::NEG_INFINITY;

            for &aid in empty_legal_action_ids(kind) {
                let sim = Game::apply_action_id_to_grid(grid, kind, aid);
                if sim.terminated {
                    continue;
                }
                let v = self.value_after_clear::<M>(&sim.grid_after_clear, plies_left - 1, depth + 1);
                if v > best {
                    best = v;
                }
            }

            return best;
        };

        // Pruned non-leaf => two phase
        let n = self.should_prune(depth).unwrap_or(ACTION_DIM);

        let mut proxies: Vec<(usize, f64)> = Vec::new();
        for &aid in empty_legal_action_ids(kind) {
            let sim = Game::apply_action_id_to_grid(grid, kind, aid);
            if sim.terminated {
                continue;
            }
            proxies.push((aid, score_grid(&sim.grid_after_lock)));
        }

        if proxies.is_empty() {
            return f64::NEG_INFINITY;
        }

        let kept = prune_top_n_scores(proxies, n);

        let mut best = f64::NEG_INFINITY;
        for (aid, _proxy) in kept {
            let sim = Game::apply_action_id_to_grid(grid, kind, aid);
            if sim.terminated {
                continue;
            }
            let v = self.value_after_clear::<M>(&sim.grid_after_clear, plies_left - 1, depth + 1);
            if v > best {
                best = v;
            }
        }

        best
    }

    #[inline]
    fn value_after_clear<M: UnknownModel>(
        &self,
        grid: &[[u8; W]; H],
        plies_left: u8,
        depth: u8,
    ) -> f64 {
        debug_assert!(plies_left >= 1);
        M::expected_value(self, grid, plies_left, depth)
    }

    /// Build aid0 candidates for the active piece with a cheap proxy to rank for pruning.
    /// (Uses Game::action_mask() for the current grid.)
    pub(crate) fn aid0_candidates_with_proxy(&self, g: &Game) -> Vec<(usize, f64)> {
        let mask = g.action_mask();
        let mut out: Vec<(usize, f64)> = Vec::new();

        for aid0 in 0..ACTION_DIM {
            if !mask[aid0] {
                continue;
            }
            let sim1 = g.simulate_action_id_active(aid0);
            if sim1.terminated {
                continue;
            }
            out.push((aid0, score_grid(&sim1.grid_after_lock)));
        }

        if let Some(n0) = self.should_prune(0) {
            out = prune_top_n_scores(out, n0);
        }

        out
    }

    // -------------------------------------------------------------------------
    // Caching helpers (used by Codemy2FastPolicy)
    // -------------------------------------------------------------------------

    /// Cheap 64-bit hash of the grid for per-decision caching (FNV-1a).
    #[inline]
    pub(crate) fn hash_grid(grid: &[[u8; W]; H]) -> u64 {
        let mut h: u64 = 1469598103934665603;
        for r in 0..H {
            for c in 0..W {
                h ^= grid[r][c] as u64;
                h = h.wrapping_mul(1099511628211);
            }
        }
        h
    }

    /// Same as "best response for known piece", but cached:
    /// (grid_hash, kind) -> (best_aid, best_score_lock, best_grid_after_clear)
    pub(crate) fn best_response_for_known_piece_cached(
        &self,
        grid: &[[u8; W]; H],
        kind: Kind,
        cache: &mut HashMap<(u64, u8), (usize, f64, [[u8; W]; H])>,
    ) -> Option<(usize, f64, [[u8; W]; H])> {
        let key = (Self::hash_grid(grid), kind_idx0_u8(kind));
        if let Some(v) = cache.get(&key) {
            return Some(*v);
        }

        let mut best_aid: Option<usize> = None;
        let mut best_score = f64::NEG_INFINITY;
        let mut best_clear = *grid;

        for &aid in empty_legal_action_ids(kind) {
            let sim = Game::apply_action_id_to_grid(grid, kind, aid);
            if sim.terminated {
                continue;
            }
            let s = score_grid(&sim.grid_after_lock);
            if s > best_score {
                best_score = s;
                best_aid = Some(aid);
                best_clear = sim.grid_after_clear;
            }
        }

        let aid = best_aid?;
        let val = (aid, best_score, best_clear);
        cache.insert(key, val);
        Some(val)
    }

    /// Compute best leaf scores for all 7 kinds on a given grid, with per-decision cache.
    pub(crate) fn best_leaf_scores_all_kinds_cached(
        &self,
        grid: &[[u8; W]; H],
        cache: &mut HashMap<u64, [f64; 7]>,
    ) -> [f64; 7] {
        let key = Self::hash_grid(grid);
        if let Some(v) = cache.get(&key) {
            return *v;
        }

        let mut arr = [f64::NEG_INFINITY; 7];
        for (i, &k) in Kind::all().iter().enumerate() {
            arr[i] = self.best_leaf_score_for_known_piece(grid, k);
        }

        cache.insert(key, arr);
        arr
    }

    /// Tail value: expected best leaf score under UniformIID next piece.
    pub(crate) fn tail_uniform_cached(
        &self,
        grid: &[[u8; W]; H],
        cache: &mut HashMap<u64, [f64; 7]>,
    ) -> f64 {
        let arr = self.best_leaf_scores_all_kinds_cached(grid, cache);
        let mut sum = 0.0;
        for v in arr {
            sum += v;
        }
        sum / 7.0
    }
}
