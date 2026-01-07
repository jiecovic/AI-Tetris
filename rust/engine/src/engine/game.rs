// rust/engine/src/engine/game.rs
#![forbid(unsafe_code)]

use crate::engine::piece_rule::{PieceRule, PieceRuleKind};
use crate::engine::pieces::Kind;

use crate::engine::constants::{
    decode_action_id, encode_action_id, ACTION_DIM, H, HIDDEN_ROWS, MAX_ROTS, W,
};
use crate::engine::geometry::{bbox_left_to_anchor_x, bbox_params};
use crate::engine::grid::{
    clear_lines_grid, fits_on_grid, height_metrics as grid_height_metrics, lock_on_grid,
};
use crate::engine::warmup::{apply_warmup, HoleCount, RowCountDist, WarmupSpec};

#[derive(Clone, Copy, Debug)]
pub struct SimPlacement {
    pub grid_after_lock: [[u8; W]; H],
    pub grid_after_clear: [[u8; W]; H],
    pub cleared_lines: u32,
    /// True iff the placement is invalid for (grid, kind, action_id).
    pub invalid: bool,
}


#[derive(Clone, Copy, Debug)]
pub struct StepResult {
    /// True game over (spawn rows occupied) OR engine already in game_over.
    pub terminated: bool,
    pub cleared_lines: u32,
    /// True iff the provided action was illegal; in that case the transition is a no-op.
    pub illegal_action: bool,
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

    /// Backwards-compatible convenience constructor.
    /// Uses fixed warmup rows/holes if warmup_rows > 0, otherwise no warmup.
    pub fn new_with_rule_and_warmup(
        seed: u64,
        rule_kind: PieceRuleKind,
        warmup_rows: u8,
        warmup_holes: u8,
    ) -> Self {
        let spec = if warmup_rows == 0 {
            WarmupSpec::none()
        } else {
            WarmupSpec {
                rows: RowCountDist::Fixed(warmup_rows),
                holes: HoleCount::Fixed(warmup_holes),
                spawn_buffer: (HIDDEN_ROWS as u8) + 2,
                seed_salt: WarmupSpec::DEFAULT_SEED_SALT,
            }
        };
        Self::new_with_rule_and_warmup_spec(seed, rule_kind, spec)
    }

    /// New constructor: warmup is configured via a documented spec.
    pub fn new_with_rule_and_warmup_spec(
        seed: u64,
        rule_kind: PieceRuleKind,
        warmup: WarmupSpec,
    ) -> Self {
        let mut piece_rule = PieceRule::new(seed, rule_kind);

        // Draw pieces first (keeps piece stream semantics identical regardless of warmup mode).
        let active = piece_rule.draw();
        let next = piece_rule.draw();

        let mut grid = [[0u8; W]; H];
        apply_warmup(&mut grid, seed, &warmup);

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

    #[inline]
    fn spawn_rows_occupied(grid: &[[u8; W]; H]) -> bool {
        for r in 0..HIDDEN_ROWS {
            for c in 0..W {
                if grid[r][c] != 0 {
                    return true;
                }
            }
        }
        false
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
                invalid: true,
            };
        }

        let x = bbox_left_to_anchor_x(kind, rot, col_left);

        let mut y: i32 = 0;
        if !fits_on_grid(grid_in, kind, rot, x, y) {
            return SimPlacement {
                grid_after_lock: *grid_in,
                grid_after_clear: *grid_in,
                cleared_lines: 0,
                invalid: true,
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
            invalid: false,
        }
    }

    pub fn simulate_action_id(&self, kind: Kind, action_id: usize) -> SimPlacement {
        Self::apply_action_id_to_grid(&self.grid, kind, action_id)
    }

    pub fn simulate_action_id_active(&self, action_id: usize) -> SimPlacement {
        Self::apply_action_id_to_grid(&self.grid, self.active, action_id)
    }

    // -------------------------------------------------------------------------
    // Mutating step (FAST PATH)
    // -------------------------------------------------------------------------

    /// Applies a placement action for the current active piece.
    ///
    /// Engine semantics:
    /// - Illegal actions are a no-op and return `illegal_action=true` without terminating.
    /// - True game over is detected iff the *post-clear* locked grid occupies any spawn row
    ///   (`r < HIDDEN_ROWS`).
    pub fn step_action_id(&mut self, action_id: usize) -> StepResult {
        if self.game_over {
            return StepResult {
                terminated: true,
                cleared_lines: 0,
                illegal_action: false,
            };
        }

        // Out-of-range action id => illegal no-op (engine does not terminate).
        if action_id >= ACTION_DIM {
            return StepResult {
                terminated: false,
                cleared_lines: 0,
                illegal_action: true,
            };
        }

        let sim = Self::apply_action_id_to_grid(&self.grid, self.active, action_id);

        // Illegal placement => no-op.
        if sim.invalid {
            return StepResult {
                terminated: false,
                cleared_lines: 0,
                illegal_action: true,
            };
        }


        // Valid placement: commit post-clear grid.
        self.grid = sim.grid_after_clear;

        self.lines_cleared += sim.cleared_lines as u64;
        self.score += 100 * sim.cleared_lines as u64;
        self.steps += 1;

        // True game over check: locked blocks in spawn rows.
        if Self::spawn_rows_occupied(&self.grid) {
            self.game_over = true;
            return StepResult {
                terminated: true,
                cleared_lines: sim.cleared_lines,
                illegal_action: false,
            };
        }

        self.spawn_next();

        StepResult {
            terminated: false,
            cleared_lines: sim.cleared_lines,
            illegal_action: false,
        }
    }

    pub fn step_macro(&mut self, rot: usize, col: i32) -> (bool, u32) {
        if col < 0 || col >= W as i32 {
            // Macro convenience: treat as illegal no-op.
            return (false, 0);
        }
        let aid = encode_action_id(rot % MAX_ROTS, col as usize);
        let r = self.step_action_id(aid);
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
