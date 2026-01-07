// src/engine/game.rs
#![forbid(unsafe_code)]

use crate::engine::piece_rule::{PieceRule, PieceRuleKind};
use crate::engine::pieces::Kind;

use crate::engine::constants::{
    decode_action_id, encode_action_id, ACTION_DIM, H, HIDDEN_ROWS, MAX_ROTS, W,
};
use crate::engine::geometry::{bbox_left_to_anchor_x, bbox_params};
use crate::engine::grid::{
    apply_warmup_garbage, clear_lines_grid, fits_on_grid, height_metrics as grid_height_metrics,
    lock_on_grid,
};

#[derive(Clone, Copy, Debug)]
pub struct SimPlacement {
    pub grid_after_lock: [[u8; W]; H],
    pub grid_after_clear: [[u8; W]; H],
    pub cleared_lines: u32,
    pub terminated: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct StepResult {
    pub terminated: bool,
    pub cleared_lines: u32,
    pub grid_after_lock: Option<[[u8; W]; H]>,
}

#[derive(Clone)]
pub struct Game {
    pub grid: [[u8; W]; H],

    piece_rule: PieceRule,

    pub active: Kind,
    pub next: Kind,

    pub score: u64,
    pub lines_cleared: u64,
    pub steps: u64,
    pub game_over: bool,
}

impl Game {
    /// Default: uniform IID stream.
    pub fn new(seed: u64) -> Self {
        Self::new_with_rule(seed, PieceRuleKind::Uniform)
    }

    pub fn new_with_rule(seed: u64, rule_kind: PieceRuleKind) -> Self {
        Self::new_with_rule_and_warmup(seed, rule_kind, 0, 1)
    }

    pub fn new_with_rule_and_warmup(
        seed: u64,
        rule_kind: PieceRuleKind,
        warmup_rows: u8,
        warmup_holes: u8,
    ) -> Self {
        let mut piece_rule = PieceRule::new(seed, rule_kind);

        // Draw pieces first (keeps piece stream semantics identical regardless of warmup mode).
        let active = piece_rule.draw();
        let next = piece_rule.draw();

        let mut grid = [[0u8; W]; H];

        if warmup_rows > 0 {
            // Keep top rows empty to avoid blocking spawn.
            // With hidden rows: reserve them + a little extra headroom.
            let spawn_buffer: usize = HIDDEN_ROWS + 2;
            let max_rows = H.saturating_sub(spawn_buffer).min(u8::MAX as usize) as u8;

            let rows = warmup_rows.min(max_rows);

            // Ensure at least 1 hole and at most W-1 holes.
            let max_holes = (W.saturating_sub(1)).min(u8::MAX as usize) as u8;
            let holes = warmup_holes.clamp(1, max_holes);

            apply_warmup_garbage(&mut grid, seed, rows, holes);
        }

        Self {
            grid,
            piece_rule,
            active,
            next,
            score: 0,
            lines_cleared: 0,
            steps: 0,
            game_over: false,
        }
    }

    pub fn piece_rule_kind(&self) -> PieceRuleKind {
        self.piece_rule.kind()
    }

    pub fn spawn_next(&mut self) {
        self.active = self.next;
        self.next = self.piece_rule.draw();
    }

    // -------------------------------------------------------------------------
    // Height metrics (locked grid only)
    // -------------------------------------------------------------------------

    /// Returns (max_height, avg_height).
    pub fn height_metrics(&self) -> (u32, f32) {
        grid_height_metrics(&self.grid)
    }

    // -------------------------------------------------------------------------
    // Fixed action space mask
    // -------------------------------------------------------------------------

    pub fn action_mask(&self) -> [bool; ACTION_DIM] {
        Self::action_mask_for_grid(&self.grid, self.active)
    }

    pub fn action_mask_for_grid(grid: &[[u8; W]; H], kind: Kind) -> [bool; ACTION_DIM] {
        let mut m = [false; ACTION_DIM];
        for aid in 0..ACTION_DIM {
            let (rot_u, col_u) = decode_action_id(aid);
            let rot = rot_u % MAX_ROTS;
            let col_left = col_u as i32;

            let (_min_dx, _bbox_w, bbox_left_max) = bbox_params(kind, rot);
            if bbox_left_max < 0 {
                continue;
            }
            if col_left < 0 || col_left > bbox_left_max {
                continue;
            }

            let x = bbox_left_to_anchor_x(kind, rot, col_left);
            if fits_on_grid(grid, kind, rot, x, 0) {
                m[aid] = true;
            }
        }
        m
    }

    pub fn legal_action_ids(&self) -> Vec<usize> {
        let m = self.action_mask();
        m.iter()
            .enumerate()
            .filter_map(|(aid, &ok)| ok.then_some(aid))
            .collect()
    }

    // -------------------------------------------------------------------------
    // Pure transition kernel
    // -------------------------------------------------------------------------

    pub fn apply_action_id_to_grid(
        grid_in: &[[u8; W]; H],
        kind: Kind,
        action_id: usize,
    ) -> SimPlacement {
        let (rot_u, col_u) = decode_action_id(action_id);
        let rot = rot_u % MAX_ROTS;
        let col_left = col_u as i32;

        let (_min_dx, _bbox_w, bbox_left_max) = bbox_params(kind, rot);
        if bbox_left_max < 0 || col_left < 0 || col_left > bbox_left_max {
            return SimPlacement {
                grid_after_lock: *grid_in,
                grid_after_clear: *grid_in,
                cleared_lines: 0,
                terminated: true,
            };
        }

        let x = bbox_left_to_anchor_x(kind, rot, col_left);

        let mut y: i32 = 0;
        if !fits_on_grid(grid_in, kind, rot, x, y) {
            return SimPlacement {
                grid_after_lock: *grid_in,
                grid_after_clear: *grid_in,
                cleared_lines: 0,
                terminated: true,
            };
        }

        while fits_on_grid(grid_in, kind, rot, x, y + 1) {
            y += 1;
        }

        let mut grid_lock = *grid_in;
        lock_on_grid(&mut grid_lock, kind, rot, x, y);

        let (grid_clear, cleared) = clear_lines_grid(&grid_lock);

        SimPlacement {
            grid_after_lock: grid_lock,
            grid_after_clear: grid_clear,
            cleared_lines: cleared,
            terminated: false,
        }
    }

    pub fn simulate_action_id(&self, kind: Kind, action_id: usize) -> SimPlacement {
        Self::apply_action_id_to_grid(&self.grid, kind, action_id)
    }

    pub fn simulate_action_id_active(&self, action_id: usize) -> SimPlacement {
        Self::apply_action_id_to_grid(&self.grid, self.active, action_id)
    }

    // -------------------------------------------------------------------------
    // Mutating step
    // -------------------------------------------------------------------------

    pub fn step_action_id(&mut self, action_id: usize, return_grid_after_lock: bool) -> StepResult {
        if self.game_over {
            return StepResult {
                terminated: true,
                cleared_lines: 0,
                grid_after_lock: None,
            };
        }
        if action_id >= ACTION_DIM {
            self.game_over = true;
            return StepResult {
                terminated: true,
                cleared_lines: 0,
                grid_after_lock: None,
            };
        }

        let sim = Self::apply_action_id_to_grid(&self.grid, self.active, action_id);

        if sim.terminated {
            self.game_over = true;
            return StepResult {
                terminated: true,
                cleared_lines: 0,
                grid_after_lock: if return_grid_after_lock {
                    Some(sim.grid_after_lock)
                } else {
                    None
                },
            };
        }

        self.grid = sim.grid_after_clear;

        self.lines_cleared += sim.cleared_lines as u64;
        self.score += 100 * sim.cleared_lines as u64;
        self.steps += 1;

        self.spawn_next();

        StepResult {
            terminated: self.game_over,
            cleared_lines: sim.cleared_lines,
            grid_after_lock: if return_grid_after_lock {
                Some(sim.grid_after_lock)
            } else {
                None
            },
        }
    }

    pub fn step_macro(&mut self, rot: usize, col: i32) -> (bool, u32) {
        if col < 0 || col >= W as i32 {
            self.game_over = true;
            return (true, 0);
        }
        let aid = encode_action_id(rot % MAX_ROTS, col as usize);
        let r = self.step_action_id(aid, false);
        (r.terminated, r.cleared_lines)
    }

    pub fn render_ascii(&self) -> String {
        let mut s = String::new();
        s.push_str("+----------+\n");
        for r in HIDDEN_ROWS..H {
            s.push('|');
            for c in 0..W {
                let v = self.grid[r][c];
                s.push(if v == 0 { ' ' } else { '#' });
            }
            s.push_str("|\n");
        }
        s.push_str("+----------+\n");
        s.push_str(&format!(
            "rule={:?} active={} next={} score={} lines={} steps={} over={}\n",
            self.piece_rule.kind(),
            self.active.glyph(),
            self.next.glyph(),
            self.score,
            self.lines_cleared,
            self.steps,
            self.game_over
        ));
        s
    }
}
