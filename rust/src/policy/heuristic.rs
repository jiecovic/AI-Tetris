// src/policy/heuristic.rs
#![forbid(unsafe_code)]

use crate::engine::{decode_action_id, Game, Kind, ACTION_DIM, H, W};

use super::base::Policy;

#[derive(Clone, Copy, Debug)]
pub enum Lookahead {
    /// One-ply greedy on current piece (score after lock).
    D0,
    /// Two-ply with known next piece (choose best next placement).
    D1,
    /// Three-ply where the 3rd piece is unknown; assume IID-uniform over 7 kinds.
    /// This is intended for the `uniform` piece rule; it will still run under bag7,
    /// but the assumption is then mismatched.
    D2Uniform,
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

    /// Best heuristic score achievable by placing `kind` onto `grid` (post-clear grid),
    /// scoring the *pre-clear* grid after lock.
    #[inline]
    fn best_score_for_piece_on_grid(grid: &[[u8; W]; H], kind: Kind) -> f64 {
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

    /// Expected best score for an unknown piece drawn IID-uniform from the 7 kinds.
    #[inline]
    fn expected_best_score_uniform_next_piece(grid: &[[u8; W]; H]) -> f64 {
        // Kind::all() is a static slice of the 7 tetromino kinds.
        let mut sum = 0.0;
        let mut n = 0.0;

        for &k in Kind::all() {
            let best_k = Self::best_score_for_piece_on_grid(grid, k);
            // If a piece has no legal move, treat it as very bad.
            // (This can happen if the grid is already dead / near-dead.)
            let v = if best_k.is_finite() { best_k } else { f64::NEG_INFINITY };
            sum += v;
            n += 1.0;
        }

        // n is always 7.0 here, but keep it robust.
        if n > 0.0 { sum / n } else { f64::NEG_INFINITY }
    }
}

impl Policy for CodemyPolicy {
    fn choose_action(&mut self, g: &Game) -> Option<(usize, i32)> {
        // Iterate fixed action space but filter by legality.
        let mask1 = g.action_mask();
        let mut best: Option<(usize, f64)> = None; // (aid, value)

        for aid0 in 0..ACTION_DIM {
            if !mask1[aid0] {
                continue;
            }

            let sim1 = g.simulate_action_id_active(aid0);
            if sim1.terminated {
                continue;
            }

            let v0 = match self.lookahead {
                Lookahead::D0 => {
                    // Score on pre-clear grid (after lock, before clear).
                    Self::score_grid(&sim1.grid_after_lock)
                }

                Lookahead::D1 => {
                    // Choose best next placement for the known next piece.
                    Self::best_score_for_piece_on_grid(&sim1.grid_after_clear, g.next)
                }

                Lookahead::D2Uniform => {
                    // Choose best next placement for known next piece,
                    // then evaluate expected best score under IID-uniform next-next piece.
                    let grid1 = &sim1.grid_after_clear;
                    let mask2 = Game::action_mask_for_grid(grid1, g.next);

                    let mut best2 = f64::NEG_INFINITY;

                    for aid1 in 0..ACTION_DIM {
                        if !mask2[aid1] {
                            continue;
                        }
                        let sim2 = Game::apply_action_id_to_grid(grid1, g.next, aid1);
                        if sim2.terminated {
                            continue;
                        }

                        let grid2 = &sim2.grid_after_clear;
                        let exp3 = Self::expected_best_score_uniform_next_piece(grid2);

                        if exp3 > best2 {
                            best2 = exp3;
                        }
                    }

                    best2
                }
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
