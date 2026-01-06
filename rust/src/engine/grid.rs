// src/engine/grid.rs
#![forbid(unsafe_code)]

use crate::engine::constants::{H, W};
use crate::engine::pieces::{rotations, Kind};

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

/// Column height: number of filled cells from bottom (0..H).
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
