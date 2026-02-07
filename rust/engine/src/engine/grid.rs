// src/engine/grid.rs
#![forbid(unsafe_code)]

use rand::prelude::*;

use crate::engine::constants::{H, HIDDEN_ROWS, W};
use crate::engine::pieces::{Kind, rotations};

pub fn fits_on_grid(grid: &[[u8; W]; H], kind: Kind, rot: usize, x: i32, y: i32) -> bool {
    let cells = rotations(kind)[rot];
    for &(dx, dy) in cells {
        let gx = x + dx;
        let gy = y + dy;
        if gx < 0 || gx >= W as i32 || gy < 0 || gy >= H as i32 {
            return false;
        }
        if grid[gy as usize][gx as usize] != 0 {
            return false;
        }
    }
    true
}

pub fn lock_on_grid(grid: &mut [[u8; W]; H], kind: Kind, rot: usize, x: i32, y: i32) {
    let v = kind.idx();
    for &(dx, dy) in rotations(kind)[rot] {
        let gx = (x + dx) as usize;
        let gy = (y + dy) as usize;
        grid[gy][gx] = v;
    }
}

pub fn clear_lines_grid(grid: &[[u8; W]; H]) -> ([[u8; W]; H], u32) {
    let mut cleared = 0u32;
    let mut new_grid = [[0u8; W]; H];
    let mut write_row: i32 = (H as i32) - 1;

    for r in (0..H).rev() {
        let full = grid[r].iter().all(|&c| c != 0);
        if full {
            cleared += 1;
            continue;
        }
        new_grid[write_row as usize] = grid[r];
        write_row -= 1;
    }

    (new_grid, cleared)
}

/// Clear any full lines in-place.
/// Returns (cleared_lines, spawn_rows_occupied).
pub fn clear_lines_inplace(grid: &mut [[u8; W]; H]) -> (u32, bool) {
    let mut cleared = 0u32;
    let mut write_row: i32 = (H as i32) - 1;

    for r in (0..H).rev() {
        let full = grid[r].iter().all(|&c| c != 0);
        if full {
            cleared += 1;
            continue;
        }
        if write_row != r as i32 {
            grid[write_row as usize] = grid[r];
        }
        write_row -= 1;
    }

    if write_row >= 0 {
        for r in 0..=write_row {
            grid[r as usize] = [0u8; W];
        }
    }

    let mut spawn_occupied = false;
    for r in 0..HIDDEN_ROWS {
        if grid[r].iter().any(|&c| c != 0) {
            spawn_occupied = true;
            break;
        }
    }

    (cleared, spawn_occupied)
}

/// Fill the bottom `rows` with "garbage": each row is filled except for `holes` empty cells.
/// Uses a derived RNG from `seed` so warmup does not perturb the piece stream.
///
/// Preconditions (enforced with debug_assert):
/// - rows <= H
/// - 1 <= holes <= W-1
pub fn apply_warmup_garbage(grid: &mut [[u8; W]; H], seed: u64, rows: u8, holes: u8) {
    debug_assert!((rows as usize) <= H);
    debug_assert!(holes >= 1);
    debug_assert!((holes as usize) < W);

    let mut rng = StdRng::seed_from_u64(seed ^ 0x9E3779B97F4A7C15);

    let rows_usize = rows as usize;
    let holes_usize = holes as usize;

    // Reused scratch buffer for sampling distinct hole columns without allocations.
    let mut cols: [usize; W] = core::array::from_fn(|i| i);

    for i in 0..rows_usize {
        let r = H - 1 - i;

        // Fill row
        for c in 0..W {
            grid[r][c] = 1;
        }

        // Sample `holes` distinct columns: shuffle prefix of [0..W)
        cols.shuffle(&mut rng);
        for j in 0..holes_usize {
            let hole_c = cols[j];
            grid[r][hole_c] = 0;
        }
    }
}

/// Column height: number of filled cells from bottom (0..H).
///
/// NOTE: With `H=22` and `HIDDEN_ROWS=2`, this currently counts occupancy in the hidden spawn
/// rows as part of the height. That is intentional for now (simple + consistent), but if you
/// want "visible-only" metrics you should start scanning from `HIDDEN_ROWS` instead of `0`.
#[inline]
pub fn col_height(grid: &[[u8; W]; H], c: usize) -> u32 {
    for r in 0..H {
        if grid[r][c] != 0 {
            return (H - r) as u32;
        }
    }
    0
}

/// Returns (max_height, avg_height).
pub fn height_metrics(grid: &[[u8; W]; H]) -> (u32, f32) {
    let mut max_h: u32 = 0;
    let mut sum: u32 = 0;
    for c in 0..W {
        let h = col_height(grid, c);
        if h > max_h {
            max_h = h;
        }
        sum += h;
    }
    let avg = sum as f32 / (W as f32);
    (max_h, avg)
}

// Silence "unused import" warnings for now while we keep the note above.
// (If you later switch to visible-only metrics, you'll use HIDDEN_ROWS in code.)
#[allow(dead_code)]
const _HIDDEN_ROWS_NOTE_ONLY: usize = HIDDEN_ROWS;
