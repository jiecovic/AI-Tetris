// src/policy/heuristic.rs
#![forbid(unsafe_code)]

use crate::engine::{decode_action_id, features::compute_grid_features, Game, Kind, ACTION_DIM, H, W};

use super::base::Policy;

#[derive(Clone, Copy, Debug)]
pub struct BeamConfig {
    /// Start pruning from this depth onward:
    /// 0 => prune aid0 candidates
    /// 1 => prune aid1 candidates
    /// 2 => prune aid2 candidates (inner "best placement" searches)
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
        // (beam_from_depth=2 means pruning only affects inner aid2 searches,
        // and with width=ACTION_DIM this is effectively exhaustive.)
        Self {
            beam_from_depth: 2,
            beam_width: ACTION_DIM,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Lookahead {
    /// One-ply greedy on current piece (score after lock).
    D0,
    /// Two-ply with known next piece (choose best next placement).
    D1,
    /// Three-ply where the 3rd piece is unknown; assume IID-uniform over 7 kinds.
    D2Uniform,
    /// Same as D2Uniform, but with configurable beam pruning.
    D2UniformBeam(BeamConfig),
}

pub struct CodemyPolicy {
    lookahead: Lookahead,
}

impl CodemyPolicy {
    pub fn new(lookahead: Lookahead) -> Self {
        Self { lookahead }
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
    fn should_prune(beam: Option<BeamConfig>, depth: u8) -> Option<usize> {
        let b = beam?;
        if depth >= b.beam_from_depth {
            Some(b.beam_width.max(1))
        } else {
            None
        }
    }

    /// Keep top-N by score descending. Deterministic ordering.
    fn top_n_by_score(mut cands: Vec<(usize, f64)>, n: usize) -> Vec<(usize, f64)> {
        if cands.is_empty() {
            return cands;
        }
        cands.sort_by(|a, b| b.1.total_cmp(&a.1));
        if n >= cands.len() {
            cands
        } else {
            cands.truncate(n);
            cands
        }
    }

    /// Candidate list for (grid, kind) scored by "local" heuristic on locked grid.
    fn candidates_local(grid: &[[u8; W]; H], kind: Kind) -> Vec<(usize, f64)> {
        let mask = Game::action_mask_for_grid(grid, kind);
        let mut out: Vec<(usize, f64)> = Vec::new();

        for aid in 0..ACTION_DIM {
            if !mask[aid] {
                continue;
            }
            let sim = Game::apply_action_id_to_grid(grid, kind, aid);
            if sim.terminated {
                continue;
            }
            let s = Self::score_grid(&sim.grid_after_lock);
            out.push((aid, s));
        }
        out
    }

    /// Best heuristic score achievable by placing `kind` onto `grid` (post-clear grid),
    /// scoring the *pre-clear* grid after lock.
    fn best_score_for_piece_on_grid(
        grid: &[[u8; W]; H],
        kind: Kind,
        beam: Option<BeamConfig>,
        depth: u8,
    ) -> f64 {
        let mut cands = Self::candidates_local(grid, kind);

        if let Some(n) = Self::should_prune(beam, depth) {
            cands = Self::top_n_by_score(cands, n);
        }

        let mut best = f64::NEG_INFINITY;
        for (_aid, s) in cands {
            if s > best {
                best = s;
            }
        }
        best
    }

    /// Expected best score for an unknown piece drawn IID-uniform from the 7 kinds.
    fn expected_best_score_uniform_next_piece(
        grid: &[[u8; W]; H],
        beam: Option<BeamConfig>,
        depth: u8,
    ) -> f64 {
        let mut sum = 0.0;
        let mut n = 0.0;

        for &k in Kind::all() {
            let best_k = Self::best_score_for_piece_on_grid(grid, k, beam, depth);
            let v = if best_k.is_finite() { best_k } else { f64::NEG_INFINITY };
            sum += v;
            n += 1.0;
        }

        if n > 0.0 { sum / n } else { f64::NEG_INFINITY }
    }
}

impl Policy for CodemyPolicy {
    fn choose_action(&mut self, g: &Game) -> Option<(usize, i32)> {
        let beam = match self.lookahead {
            Lookahead::D2UniformBeam(b) => Some(b),
            _ => None,
        };

        // Legal actions for current piece.
        let mask1 = g.action_mask();
        let mut aid0_cands: Vec<(usize, f64)> = Vec::new();

        // For D2 beam pruning at depth0, we need a cheap proxy to rank aid0.
        // Use local heuristic on grid_after_lock for the active piece placement.
        for aid0 in 0..ACTION_DIM {
            if !mask1[aid0] {
                continue;
            }
            let sim1 = g.simulate_action_id_active(aid0);
            if sim1.terminated {
                continue;
            }
            let proxy0 = Self::score_grid(&sim1.grid_after_lock);
            aid0_cands.push((aid0, proxy0));
        }

        if aid0_cands.is_empty() {
            return None;
        }

        // Optionally prune aid0 candidates (depth 0) before computing deeper values.
        if let Some(n0) = Self::should_prune(beam, 0) {
            aid0_cands = Self::top_n_by_score(aid0_cands, n0);
        }

        let mut best: Option<(usize, f64)> = None;

        for (aid0, _proxy0) in aid0_cands {
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
                    Self::best_score_for_piece_on_grid(&sim1.grid_after_clear, g.next, None, 2)
                }

                Lookahead::D2Uniform | Lookahead::D2UniformBeam(_) => {
                    // D2: choose best next placement for known next piece,
                    // then evaluate expected best score under IID-uniform next-next piece.
                    let grid1 = &sim1.grid_after_clear;
                    let mut aid1_cands = Self::candidates_local(grid1, g.next);

                    // Optionally prune aid1 candidates (depth 1) before doing expensive expectation.
                    if let Some(n1) = Self::should_prune(beam, 1) {
                        aid1_cands = Self::top_n_by_score(aid1_cands, n1);
                    }

                    let mut best2 = f64::NEG_INFINITY;

                    for (aid1, _proxy1) in aid1_cands {
                        let sim2 = Game::apply_action_id_to_grid(grid1, g.next, aid1);
                        if sim2.terminated {
                            continue;
                        }

                        let grid2 = &sim2.grid_after_clear;

                        // Depth 2 is inside the unknown-piece "best placement" searches.
                        let exp3 = Self::expected_best_score_uniform_next_piece(grid2, beam, 2);

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
