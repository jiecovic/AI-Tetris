// src/game.rs
use crate::piece_rule::{PieceRule, PieceRuleKind};
use crate::pieces::{rotations, Kind};

pub const H: usize = 20;
pub const W: usize = 10;

pub const MAX_ROTS: usize = 4;
pub const ACTION_DIM: usize = MAX_ROTS * W;

#[inline]
pub fn encode_action_id(rot: usize, col: usize) -> usize {
    debug_assert!(rot < MAX_ROTS);
    debug_assert!(col < W);
    rot * W + col
}

#[inline]
pub fn decode_action_id(aid: usize) -> (usize, usize) {
    (aid / W, aid % W)
}

// -----------------------------------------------------------------------------
// bbox-left semantics helpers
// -----------------------------------------------------------------------------

#[inline]
fn dx_range(kind: Kind, rot: usize) -> (i32, i32) {
    let cells = rotations(kind)[rot];
    let mut mn = i32::MAX;
    let mut mx = i32::MIN;
    for &(dx, _dy) in cells {
        mn = mn.min(dx);
        mx = mx.max(dx);
    }
    (mn, mx)
}

#[inline]
fn bbox_params(kind: Kind, rot: usize) -> (i32, i32, i32) {
    let (min_dx, max_dx) = dx_range(kind, rot);
    let bbox_w = max_dx - min_dx + 1;
    let bbox_left_max = (W as i32) - bbox_w;
    (min_dx, bbox_w, bbox_left_max)
}

#[inline]
fn bbox_left_to_anchor_x(kind: Kind, rot: usize, bbox_left_col: i32) -> i32 {
    let (min_dx, _bbox_w, _bbox_left_max) = bbox_params(kind, rot);
    bbox_left_col - min_dx
}

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
        let mut piece_rule = PieceRule::new(seed, rule_kind);

        let active = piece_rule.draw();
        let next = piece_rule.draw();

        Self {
            grid: [[0; W]; H],
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

    /// Column height: number of filled cells from bottom (0..H).
    #[inline]
    fn col_height(grid: &[[u8; W]; H], c: usize) -> u32 {
        for r in 0..H {
            if grid[r][c] != 0 {
                return (H - r) as u32;
            }
        }
        0
    }

    /// Returns (max_height, avg_height).
    pub fn height_metrics(&self) -> (u32, f32) {
        let mut max_h: u32 = 0;
        let mut sum: u32 = 0;
        for c in 0..W {
            let h = Self::col_height(&self.grid, c);
            if h > max_h {
                max_h = h;
            }
            sum += h;
        }
        let avg = sum as f32 / (W as f32);
        (max_h, avg)
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
        for r in 0..H {
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

// -----------------------------------------------------------------------------
// Pure helpers
// -----------------------------------------------------------------------------

fn fits_on_grid(grid: &[[u8; W]; H], kind: Kind, rot: usize, x: i32, y: i32) -> bool {
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

fn lock_on_grid(grid: &mut [[u8; W]; H], kind: Kind, rot: usize, x: i32, y: i32) {
    let v = kind.idx();
    for &(dx, dy) in rotations(kind)[rot] {
        let gx = (x + dx) as usize;
        let gy = (y + dy) as usize;
        grid[gy][gx] = v;
    }
}

fn clear_lines_grid(grid: &[[u8; W]; H]) -> ([[u8; W]; H], u32) {
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
