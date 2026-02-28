// src/policy/codemy/score.rs
#![forbid(unsafe_code)]

use crate::engine::{H, HIDDEN_ROWS, W, compute_grid_features};

#[inline]
pub(crate) fn complete_lines_visible_only(grid: &[[u8; W]; H]) -> u32 {
    // With H=22 and HIDDEN_ROWS=2, treat the bottom (H - HIDDEN_ROWS) rows as the playfield.
    // So we ignore full rows in the hidden spawn buffer.
    grid[HIDDEN_ROWS..]
        .iter()
        .filter(|row| row.iter().all(|&c| c != 0))
        .count() as u32
}

#[inline]
pub(crate) fn score_grid(grid_after_lock: &[[u8; W]; H]) -> f64 {
    // NOTE: compute_grid_features currently scans the FULL grid height (including hidden rows).
    // This is consistent, but if you want strict CodemyRoad (20-row) comparability,
    // you'll want a visible-only feature variant later.
    let f = compute_grid_features(grid_after_lock);
    let complete_lines = complete_lines_visible_only(grid_after_lock);

    // CodemyRoad GA weights
    -0.510066 * (f.agg_h as f64) + 0.760666 * (complete_lines as f64)
        - 0.35663 * (f.holes as f64)
        - 0.184483 * (f.bump as f64)
}
