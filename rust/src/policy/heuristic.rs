// src/policy/heuristic.rs
#![forbid(unsafe_code)]

use crate::engine::{decode_action_id, Game, ACTION_DIM, H, W};

use super::base::Policy;

#[derive(Clone, Copy, Debug)]
pub enum Lookahead {
    D0,
    D1,
}

pub struct CodemyPolicy {
    lookahead: Lookahead,
}

impl CodemyPolicy {
    pub fn new(lookahead: Lookahead) -> Self {
        Self { lookahead }
    }

    fn score_grid(grid: &[[u8; W]; H]) -> f64 {
        let heights = column_heights(grid);
        let agg_height: i32 = heights.iter().sum();
        let complete_lines: i32 = grid
            .iter()
            .filter(|row| row.iter().all(|&c| c != 0))
            .count() as i32;
        let holes: i32 = count_holes(grid, &heights);
        let bumpiness: i32 = heights.windows(2).map(|w| (w[0] - w[1]).abs()).sum();

        // CodemyRoad GA weights
        -0.510066 * agg_height as f64
            + 0.760666 * complete_lines as f64
            - 0.35663 * holes as f64
            - 0.184483 * bumpiness as f64
    }
}

impl Policy for CodemyPolicy {
    fn choose_action(&mut self, g: &Game) -> Option<(usize, i32)> {
        // Iterate full fixed action space but filter by legality
        let mask1 = g.action_mask();

        let mut best: Option<(usize, f64)> = None; // (aid, value)

        for aid in 0..ACTION_DIM {
            if !mask1[aid] {
                continue;
            }

            let sim1 = g.simulate_action_id_active(aid);
            if sim1.terminated {
                continue;
            }

            let v = match self.lookahead {
                Lookahead::D0 => {
                    // Score on pre-clear grid (after lock, before clear)
                    Self::score_grid(&sim1.grid_after_lock)
                }
                Lookahead::D1 => {
                    // Roll forward on post-clear grid, then choose best next placement for known next piece.
                    let mask2 = Game::action_mask_for_grid(&sim1.grid_after_clear, g.next);

                    let mut best2 = f64::NEG_INFINITY;
                    for aid2 in 0..ACTION_DIM {
                        if !mask2[aid2] {
                            continue;
                        }
                        let sim2 =
                            Game::apply_action_id_to_grid(&sim1.grid_after_clear, g.next, aid2);
                        if sim2.terminated {
                            continue;
                        }
                        let s2 = Self::score_grid(&sim2.grid_after_lock);
                        if s2 > best2 {
                            best2 = s2;
                        }
                    }
                    best2
                }
            };

            match best {
                None => best = Some((aid, v)),
                Some((_ba, bv)) if v > bv => best = Some((aid, v)),
                _ => {}
            }
        }

        // Convert best action_id back to (rot, col) for the current Policy trait.
        best.map(|(aid, _)| {
            let (rot, col) = decode_action_id(aid);
            (rot, col as i32)
        })
    }
}

// ---------------- feature helpers ----------------

fn column_heights(grid: &[[u8; W]; H]) -> [i32; W] {
    let mut h = [0; W];
    for c in 0..W {
        for r in 0..H {
            if grid[r][c] != 0 {
                h[c] = (H - r) as i32;
                break;
            }
        }
    }
    h
}

fn count_holes(grid: &[[u8; W]; H], heights: &[i32; W]) -> i32 {
    let mut holes = 0;
    for c in 0..W {
        let col_h = heights[c];
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
