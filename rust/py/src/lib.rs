// rust/py/src/lib.rs
#![forbid(unsafe_code)]
#![allow(unsafe_op_in_unsafe_fn)] // pyo3 macro-generated glue triggers this on Rust 2024

use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;

use tetris_engine::engine::{Game, PieceRuleKind, H, W};

#[pyclass]
pub struct TetrisEngine {
    g: Game,
}

#[pymethods]
impl TetrisEngine {
    #[new]
    #[pyo3(signature = (seed=12345, piece_rule="uniform", warmup_rows=0, warmup_holes=1))]
    fn new(seed: u64, piece_rule: &str, warmup_rows: u8, warmup_holes: u8) -> Self {
        let rule = PieceRuleKind::from_cli(piece_rule);
        let g = Game::new_with_rule_and_warmup(seed, rule, warmup_rows, warmup_holes);
        Self { g }
    }

    #[pyo3(signature = (seed=None, piece_rule=None, warmup_rows=None, warmup_holes=None))]
    fn reset(
        &mut self,
        seed: Option<u64>,
        piece_rule: Option<&str>,
        warmup_rows: Option<u8>,
        warmup_holes: Option<u8>,
    ) {
        let seed = seed.unwrap_or(12345);
        let rule = PieceRuleKind::from_cli(piece_rule.unwrap_or("uniform"));
        let wr = warmup_rows.unwrap_or(0);
        let wh = warmup_holes.unwrap_or(1);
        self.g = Game::new_with_rule_and_warmup(seed, rule, wr, wh);
    }

    /// Returns (terminated, cleared_lines, illegal_action).
    fn step_action_id(&mut self, action_id: usize) -> (bool, u32, bool) {
        let r = self.g.step_action_id(action_id);
        (r.terminated, r.cleared_lines, r.illegal_action)
    }

    /// Returns mask as uint8 array of shape (ACTION_DIM,): 1 = legal, 0 = illegal.
    fn action_mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        let m = self.g.action_mask();
        let v: Vec<u8> = m.into_iter().map(|b| if b { 1 } else { 0 }).collect();

        // numpy 0.22 API:
        PyArray1::from_vec_bound(py, v)
    }

    fn legal_action_ids(&self) -> Vec<usize> {
        self.g.legal_action_ids()
    }

    /// Returns grid as uint8 array of shape (H, W).
    fn grid<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u8>> {
        let mut rows: Vec<Vec<u8>> = Vec::with_capacity(H);
        for r in 0..H {
            let mut row: Vec<u8> = Vec::with_capacity(W);
            for c in 0..W {
                row.push(self.g.grid[r][c]);
            }
            rows.push(row);
        }

        PyArray2::from_vec2_bound(py, &rows).expect("from_vec2 failed")
    }

    fn score(&self) -> u64 {
        self.g.score
    }
    fn lines_cleared(&self) -> u64 {
        self.g.lines_cleared
    }
    fn steps(&self) -> u64 {
        self.g.steps
    }
    fn game_over(&self) -> bool {
        self.g.game_over
    }

    fn active_kind(&self) -> String {
        self.g.active.glyph().to_string()
    }
    fn next_kind(&self) -> String {
        self.g.next.glyph().to_string()
    }
}

#[pymodule]
fn tetris_rl_engine(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TetrisEngine>()?;
    Ok(())
}
