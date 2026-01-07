// src/policy/heuristic.rs
#![forbid(unsafe_code)]

use core::marker::PhantomData;
use std::collections::HashMap;

use crate::engine::{decode_action_id, features::compute_grid_features, Game, Kind, ACTION_DIM, H, W};

use super::base::Policy;

#[derive(Clone, Copy, Debug)]
pub struct BeamConfig {
    /// Start pruning from this depth onward:
    /// 0 => prune aid0 candidates
    /// 1 => prune aid1 candidates
    /// 2 => prune aid2+ candidates (inner searches)
    pub beam_from_depth: u8,
    /// Keep top-N candidates at pruned depths.
    pub beam_width: usize,
}

impl BeamConfig {
    pub fn new(beam_from_depth: u8, beam_width: usize) -> Self {
        Self {
            beam_from_depth,
            beam_width: beam_width.max(1),
        }
    }
}

impl Default for BeamConfig {
    fn default() -> Self {
        // Default behavior: don't effectively prune aid0/aid1.
        // (beam_from_depth=2 means pruning only affects inner searches,
        // and with width=ACTION_DIM this is effectively exhaustive.)
        Self {
            beam_from_depth: 2,
            beam_width: ACTION_DIM,
        }
    }
}

/// Core implementation shared by dynamic + static policy wrappers.
/// Holds all knobs (currently only beam pruning).
#[derive(Clone, Copy, Debug)]
struct CodemyCore {
    beam: Option<BeamConfig>,
}

impl CodemyCore {
    fn new(beam: Option<BeamConfig>) -> Self {
        Self { beam }
    }

    #[inline]
    fn complete_lines(grid: &[[u8; W]; H]) -> u32 {
        grid.iter()
            .filter(|row| row.iter().all(|&c| c != 0))
            .count() as u32
    }

    #[inline]
    fn score_grid(grid_after_lock: &[[u8; W]; H]) -> f64 {
        let f = compute_grid_features(grid_after_lock);
        let complete_lines = Self::complete_lines(grid_after_lock);

        // CodemyRoad GA weights
        -0.510066 * (f.agg_h as f64)
            + 0.760666 * (complete_lines as f64)
            - 0.35663 * (f.holes as f64)
            - 0.184483 * (f.bump as f64)
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

    /// Deterministic top-N selection for (aid, score) without sorting the entire list.
    /// - O(n) partition + O(k log k) sort of the kept prefix.
    fn prune_top_n_scores(mut xs: Vec<(usize, f64)>, n: usize) -> Vec<(usize, f64)> {
        if xs.is_empty() {
            return xs;
        }
        if n >= xs.len() {
            // No effective pruning => don't pay sorting cost.
            return xs;
        }

        // Partition so that [0..n] are the n best elements in *some* order.
        let nth = n - 1;
        xs.select_nth_unstable_by(nth, |a, b| b.1.total_cmp(&a.1));
        xs.truncate(n);

        // Deterministic iteration order among kept elements.
        xs.sort_by(|a, b| b.1.total_cmp(&a.1));
        xs
    }

    /// Fast path: maximize score_grid(grid_after_lock) for a known piece on a grid.
    /// Single simulation per legal aid. No allocations.
    fn best_leaf_score_for_known_piece(&self, grid: &[[u8; W]; H], kind: Kind) -> f64 {
        let mask = Game::action_mask_for_grid(grid, kind);
        let mut best = f64::NEG_INFINITY;

        for aid in 0..ACTION_DIM {
            if !mask[aid] {
                continue;
            }
            let sim = Game::apply_action_id_to_grid(grid, kind, aid);
            if sim.terminated {
                continue;
            }
            let s = Self::score_grid(&sim.grid_after_lock);
            if s > best {
                best = s;
            }
        }

        best
    }

    /// Leaf with beam pruning:
    /// compute scores for all legal actions, select top-N, then return the max score among them.
    fn best_leaf_score_for_known_piece_beam(
        &self,
        grid: &[[u8; W]; H],
        kind: Kind,
        n: usize,
    ) -> f64 {
        let mask = Game::action_mask_for_grid(grid, kind);
        let mut scores: Vec<(usize, f64)> = Vec::new();

        for aid in 0..ACTION_DIM {
            if !mask[aid] {
                continue;
            }
            let sim = Game::apply_action_id_to_grid(grid, kind, aid);
            if sim.terminated {
                continue;
            }
            let s = Self::score_grid(&sim.grid_after_lock);
            scores.push((aid, s));
        }

        if scores.is_empty() {
            return f64::NEG_INFINITY;
        }

        let kept = Self::prune_top_n_scores(scores, n);

        let mut best = f64::NEG_INFINITY;
        for (_aid, s) in kept {
            if s > best {
                best = s;
            }
        }
        best
    }

    /// Value when the next piece is known: maximize over placements of `kind` on `grid`.
    ///
    /// Semantics:
    /// - plies_left == 1: leaf => score on locked (pre-clear) grid
    /// - plies_left  > 1: after placing known piece, remaining future plies are handled by `M`
    ///
    /// PERFORMANCE RULES:
    /// - If no pruning at this depth: do a single-pass simulate+recurse, no allocations.
    /// - If pruning: two-phase
    ///     1) score-only pass to pick top-N (cheap proxy)
    ///     2) re-simulate only top-N to recurse
    fn value_known_piece<M: UnknownModel>(
        &self,
        grid: &[[u8; W]; H],
        kind: Kind,
        plies_left: u8,
        depth: u8,
    ) -> f64 {
        debug_assert!(plies_left >= 1);

        // Leaf: avoid all allocations/copies.
        if plies_left == 1 {
            if let Some(n) = self.should_prune(depth) {
                return self.best_leaf_score_for_known_piece_beam(grid, kind, n);
            }
            return self.best_leaf_score_for_known_piece(grid, kind);
        }

        // Non-leaf:
        // If no pruning, do a single pass: simulate once per aid, recurse immediately.
        let Some(_n) = self.should_prune(depth) else {
            let mask = Game::action_mask_for_grid(grid, kind);
            let mut best = f64::NEG_INFINITY;

            for aid in 0..ACTION_DIM {
                if !mask[aid] {
                    continue;
                }
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

        // NOTE: For simplicity and to keep behavior identical to your currently-good code,
        // we keep the two-phase pruning path as-is (beam mode).
        // (Your "2fast" variant below does not rely on beam.)
        let n = self.should_prune(depth).unwrap_or(ACTION_DIM);

        // Pruned non-leaf: two-phase.
        // Phase 1: compute proxy scores to select top-N.
        let mask = Game::action_mask_for_grid(grid, kind);
        let mut scores: Vec<(usize, f64)> = Vec::new();

        for aid in 0..ACTION_DIM {
            if !mask[aid] {
                continue;
            }
            let sim = Game::apply_action_id_to_grid(grid, kind, aid);
            if sim.terminated {
                continue;
            }
            let s = Self::score_grid(&sim.grid_after_lock);
            scores.push((aid, s));
        }

        if scores.is_empty() {
            return f64::NEG_INFINITY;
        }

        let kept = Self::prune_top_n_scores(scores, n);

        // Phase 2: re-simulate only kept aids to recurse.
        let mut best = f64::NEG_INFINITY;
        for (aid, _s) in kept {
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
    fn aid0_candidates_with_proxy(&self, g: &Game) -> Vec<(usize, f64)> {
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
            let proxy0 = Self::score_grid(&sim1.grid_after_lock);
            out.push((aid0, proxy0));
        }

        if let Some(n0) = self.should_prune(0) {
            out = Self::prune_top_n_scores(out, n0);
        }

        out
    }

    // -------------------------------------------------------------------------
    // Codemy1 helpers (used by "codemy2fast")
    // -------------------------------------------------------------------------

    /// For a *known* next piece: find the best response action id and its leaf score.
    /// Returns (best_aid, best_leaf_score). If no legal move, returns None.
    fn best_response_for_known_piece(&self, grid: &[[u8; W]; H], kind: Kind) -> Option<(usize, f64)> {
        let mask = Game::action_mask_for_grid(grid, kind);
        let mut best_aid: Option<usize> = None;
        let mut best_score = f64::NEG_INFINITY;

        for aid in 0..ACTION_DIM {
            if !mask[aid] {
                continue;
            }
            let sim = Game::apply_action_id_to_grid(grid, kind, aid);
            if sim.terminated {
                continue;
            }
            let s = Self::score_grid(&sim.grid_after_lock);
            if s > best_score {
                best_score = s;
                best_aid = Some(aid);
            }
        }

        best_aid.map(|aid| (aid, best_score))
    }

    /// Cheap 64-bit hash of the grid for per-decision caching (FNV-1a).
    #[inline]
    fn hash_grid(grid: &[[u8; W]; H]) -> u64 {
        let mut h: u64 = 1469598103934665603;
        for r in 0..H {
            for c in 0..W {
                h ^= grid[r][c] as u64;
                h = h.wrapping_mul(1099511628211);
            }
        }
        h
    }

    /// Compute best leaf scores for all 7 kinds on a given grid, with per-decision cache.
    #[inline]
    fn best_leaf_scores_all_kinds_cached(
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
    #[inline]
    fn tail_uniform_cached(&self, grid: &[[u8; W]; H], cache: &mut HashMap<u64, [f64; 7]>) -> f64 {
        let arr = self.best_leaf_scores_all_kinds_cached(grid, cache);
        let mut sum = 0.0;
        for v in arr {
            sum += v;
        }
        sum / 7.0
    }
}

/// Unknown future model (type-level), used for inlining in static policy variants.
/// Kept crate-visible to avoid public API warnings while not leaking as public API.
pub(crate) trait UnknownModel {
    fn expected_value(core: &CodemyCore, grid: &[[u8; W]; H], plies_left: u8, depth: u8) -> f64;
}

/// IID-uniform over the 7 kinds.
#[derive(Clone, Copy, Debug)]
pub struct UniformIID;

impl UnknownModel for UniformIID {
    #[inline]
    fn expected_value(core: &CodemyCore, grid: &[[u8; W]; H], plies_left: u8, depth: u8) -> f64 {
        let mut sum = 0.0;
        let mut n = 0.0;

        for &k in Kind::all() {
            sum += core.value_known_piece::<UniformIID>(grid, k, plies_left, depth);
            n += 1.0;
        }

        if n > 0.0 { sum / n } else { f64::NEG_INFINITY }
    }
}

/// Dynamic (runtime plies) policy.
pub struct CodemyPolicyDynamic {
    core: CodemyCore,
    plies: u8,
}

impl CodemyPolicyDynamic {
    pub fn new(plies: u8, beam: Option<BeamConfig>) -> Self {
        Self {
            core: CodemyCore::new(beam),
            plies: plies.max(1),
        }
    }
}

impl Policy for CodemyPolicyDynamic {
    fn choose_action(&mut self, g: &Game) -> Option<(usize, i32)> {
        let aid0_cands = self.core.aid0_candidates_with_proxy(g);
        if aid0_cands.is_empty() {
            return None;
        }

        let mut best: Option<(usize, f64)> = None;

        for (aid0, _proxy0) in aid0_cands {
            let sim1 = g.simulate_action_id_active(aid0);
            if sim1.terminated {
                continue;
            }

            let v0 = if self.plies == 1 {
                CodemyCore::score_grid(&sim1.grid_after_lock)
            } else {
                // Next ply is the known next piece.
                self.core
                    .value_known_piece::<UniformIID>(&sim1.grid_after_clear, g.next, self.plies - 1, 1)
            };

            match best {
                None => best = Some((aid0, v0)),
                Some((_ba, bv)) if v0 > bv => best = Some((aid0, v0)),
                _ => {}
            }
        }

        best.map(|(aid, _)| {
            let (rot, col) = decode_action_id(aid);
            (rot, col as i32)
        })
    }
}

/// Static (compile-time plies + unknown model) policy.
/// This is the "Rust templates" fast-path: monomorphized for each (M, PLIES).
pub struct CodemyPolicyStatic<M: UnknownModel, const PLIES: u8> {
    core: CodemyCore,
    _m: PhantomData<M>,
}

impl<M: UnknownModel, const PLIES: u8> CodemyPolicyStatic<M, PLIES> {
    pub fn new(beam: Option<BeamConfig>) -> Self {
        debug_assert!(PLIES >= 1);
        Self {
            core: CodemyCore::new(beam),
            _m: PhantomData,
        }
    }
}

impl<M: UnknownModel, const PLIES: u8> Policy for CodemyPolicyStatic<M, PLIES> {
    fn choose_action(&mut self, g: &Game) -> Option<(usize, i32)> {
        debug_assert!(PLIES >= 1);

        let aid0_cands = self.core.aid0_candidates_with_proxy(g);
        if aid0_cands.is_empty() {
            return None;
        }

        let mut best: Option<(usize, f64)> = None;

        for (aid0, _proxy0) in aid0_cands {
            let sim1 = g.simulate_action_id_active(aid0);
            if sim1.terminated {
                continue;
            }

            // PLIES is const => compiler folds these branches per instantiation.
            let v0 = if PLIES == 1 {
                CodemyCore::score_grid(&sim1.grid_after_lock)
            } else {
                self.core
                    .value_known_piece::<M>(&sim1.grid_after_clear, g.next, PLIES - 1, 1)
            };

            match best {
                None => best = Some((aid0, v0)),
                Some((_ba, bv)) if v0 > bv => best = Some((aid0, v0)),
                _ => {}
            }
        }

        best.map(|(aid, _)| {
            let (rot, col) = decode_action_id(aid);
            (rot, col as i32)
        })
    }
}

/// "codemy2fast": codemy1 exact best-response to known next piece,
/// plus a cheap one-step tail estimate for the unknown next-next piece.
/// No beam needed; uses per-decision caching.
///
/// How it scores a candidate aid0:
/// 1) simulate aid0 -> grid1 (post-clear)
/// 2) choose best aid1* for g.next on grid1 (codemy1 rule)
/// 3) simulate aid1* -> grid2 (post-clear)
/// 4) tail(grid2) = E_k[ best_leaf_score(grid2, k) ] (cached)
///
/// Final value = codemy1_best_score + tail_weight * tail
/// (you can tune tail_weight; start small to avoid destabilizing codemy1).
pub struct Codemy2FastPolicy {
    tail_weight: f64,
}

impl Codemy2FastPolicy {
    pub fn new(tail_weight: f64) -> Self {
        Self { tail_weight }
    }
}

impl Policy for Codemy2FastPolicy {
    fn choose_action(&mut self, g: &Game) -> Option<(usize, i32)> {
        let core = CodemyCore::new(None);

        // Per-decision cache: grid_hash -> best leaf scores for each kind (7).
        let mut cache: HashMap<u64, [f64; 7]> = HashMap::new();

        let mask1 = g.action_mask();
        let mut best: Option<(usize, f64)> = None;

        for aid0 in 0..ACTION_DIM {
            if !mask1[aid0] {
                continue;
            }
            let sim1 = g.simulate_action_id_active(aid0);
            if sim1.terminated {
                continue;
            }

            let grid1 = &sim1.grid_after_clear;

            // codemy1: best response to known next piece.
            let Some((aid1_star, best1_score_lock)) = core.best_response_for_known_piece(grid1, g.next) else {
                continue;
            };

            // Apply the chosen best response to get grid2 (post-clear).
            let sim2 = Game::apply_action_id_to_grid(grid1, g.next, aid1_star);
            if sim2.terminated {
                continue;
            }
            let grid2 = &sim2.grid_after_clear;

            // Cheap tail: expectation over unknown next-next piece.
            let tail = core.tail_uniform_cached(grid2, &mut cache);

            let v0 = best1_score_lock + self.tail_weight * tail;

            match best {
                None => best = Some((aid0, v0)),
                Some((_ba, bv)) if v0 > bv => best = Some((aid0, v0)),
                _ => {}
            }
        }

        best.map(|(aid, _)| {
            let (rot, col) = decode_action_id(aid);
            (rot, col as i32)
        })
    }
}

// -----------------------------------------------------------------------------
// Public aliases (convenient "named" policies)
// -----------------------------------------------------------------------------

/// Backwards-friendly default name: dynamic policy.
pub type CodemyPolicy = CodemyPolicyDynamic;

/// Fast monomorphized presets (UniformIID unknown model).
pub type Codemy0 = CodemyPolicyStatic<UniformIID, 1>;
pub type Codemy1 = CodemyPolicyStatic<UniformIID, 2>;
pub type Codemy2 = CodemyPolicyStatic<UniformIID, 3>;
