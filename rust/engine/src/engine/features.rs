// src/engine/features.rs
#![forbid(unsafe_code)]

use super::constants::{H, HIDDEN_ROWS, W};

#[derive(Clone, Copy, Debug, Default)]
pub struct GridFeatures {
    pub max_h: u32,
    pub agg_h: u32,
    pub holes: u32,
    pub bump: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GridDelta {
    pub d_max_h: i32,
    pub d_agg_h: i32,
    pub d_holes: i32,
    pub d_bump: i32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct StepFeatures {
    pub cur: GridFeatures,
    pub delta: GridDelta,
}

/// Compute classic Tetris grid features on a *locked* grid.
/// Complexity: O(H*W), no allocations.
///
/// NOTE (hidden spawn rows):
/// With `H=22` (and a conceptual hidden top area), this implementation currently computes
/// heights/holes over the FULL 22-row grid (i.e., hidden rows are included).
/// If you later want "visible-only" features, adjust the scans in `column_heights` and
/// `count_holes` to start from `HIDDEN_ROWS` instead of `0`.
pub fn compute_grid_features(grid: &[[u8; W]; H]) -> GridFeatures {
    let heights = column_heights(grid);

    let mut max_h = 0u32;
    let mut agg_h = 0u32;
    for &h in &heights {
        max_h = max_h.max(h);
        agg_h += h;
    }

    let holes = count_holes(grid, &heights);
    let bump = bumpiness(&heights);

    GridFeatures {
        max_h,
        agg_h,
        holes,
        bump,
    }
}

/// Compute classic Tetris grid features on the VISIBLE grid only
/// (hidden spawn rows are zeroed before feature extraction).
pub fn compute_grid_features_visible(grid: &[[u8; W]; H]) -> GridFeatures {
    let mut g = *grid;
    for r in 0..HIDDEN_ROWS {
        g[r] = [0u8; W];
    }
    compute_grid_features(&g)
}

/// Convenience: compute current features and delta vs previous features.
/// If `prev` is None, deltas are reported as 0.
pub fn compute_step_features(grid: &[[u8; W]; H], prev: Option<GridFeatures>) -> StepFeatures {
    let cur = compute_grid_features(grid);

    let delta = match prev {
        None => GridDelta::default(),
        Some(p) => GridDelta {
            d_max_h: cur.max_h as i32 - p.max_h as i32,
            d_agg_h: cur.agg_h as i32 - p.agg_h as i32,
            d_holes: cur.holes as i32 - p.holes as i32,
            d_bump: cur.bump as i32 - p.bump as i32,
        },
    };

    StepFeatures { cur, delta }
}

// ---------------- internal helpers ----------------

fn column_heights(grid: &[[u8; W]; H]) -> [u32; W] {
    let mut h = [0u32; W];
    for c in 0..W {
        for r in 0..H {
            if grid[r][c] != 0 {
                h[c] = (H - r) as u32;
                break;
            }
        }
    }
    h
}

fn count_holes(grid: &[[u8; W]; H], heights: &[u32; W]) -> u32 {
    let mut holes = 0u32;
    for c in 0..W {
        let col_h = heights[c] as i32;
        if col_h <= 0 {
            continue;
        }
        let top_r = (H as i32) - col_h;
        for r in top_r..(H as i32) {
            if grid[r as usize][c] == 0 {
                holes += 1;
            }
        }
    }
    holes
}

fn bumpiness(heights: &[u32; W]) -> u32 {
    let mut b = 0u32;
    for i in 0..(W - 1) {
        let a = heights[i] as i32;
        let c = heights[i + 1] as i32;
        b += (a - c).abs() as u32;
    }
    b
}
