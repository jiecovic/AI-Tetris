// src/policy/beam.rs
#![forbid(unsafe_code)]

/**
 * Beam pruning configuration for Codemy-style lookahead.
 *
 * Depth meaning:
 * - 0 = choosing aid0 (current piece)
 * - 1 = choosing aid1 (next piece)
 * - 2+ = deeper recursion
 */
#[derive(Clone, Copy, Debug)]
pub struct BeamConfig {
    /// Start pruning from this depth onward.
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

/**
 * Deterministic top-N selection for (aid, score) without sorting the entire list.
 * - O(n) partition + O(k log k) sort of the kept prefix.
 */
pub(crate) fn prune_top_n_scores(mut xs: Vec<(usize, f64)>, n: usize) -> Vec<(usize, f64)> {
    if xs.is_empty() {
        return xs;
    }
    if n >= xs.len() {
        // No effective pruning => don't pay sorting cost.
        return xs;
    }

    let nth = n - 1;
    xs.select_nth_unstable_by(nth, |a, b| b.1.total_cmp(&a.1));
    xs.truncate(n);

    // Deterministic iteration order among kept elements.
    xs.sort_by(|a, b| b.1.total_cmp(&a.1));
    xs
}

/**
 * In-place top-N pruning on a preallocated slice.
 * Returns the number of kept elements (prefix length).
 */
pub(crate) fn prune_top_n_scores_inplace(xs: &mut [(usize, f64)], len: usize, n: usize) -> usize {
    if len == 0 {
        return 0;
    }

    let keep = n.min(len);
    if keep == len {
        // No effective pruning => don't pay sorting cost.
        return len;
    }

    let nth = keep - 1;
    let slice = &mut xs[..len];
    slice.select_nth_unstable_by(nth, |a, b| b.1.total_cmp(&a.1));

    // Deterministic iteration order among kept elements.
    slice[..keep].sort_by(|a, b| b.1.total_cmp(&a.1));
    keep
}
