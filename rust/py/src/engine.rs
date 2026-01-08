// rust/py/src/engine.rs
#![forbid(unsafe_code)]

use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use tetris_engine::engine::{
    compute_grid_features, compute_step_features, Game, GridFeatures, PieceRuleKind, SimPlacement,
    WarmupSpec, ACTION_DIM, H, HIDDEN_ROWS, MAX_ROTS, W,
};

// SSOT: bind the engine's canonical action-id helpers (do NOT reimplement).
use tetris_engine::engine::constants::{decode_action_id, encode_action_id};

// UI helpers (SSOT) for piece layout preview.
// NOTE: pieces live under tetris_engine::engine::pieces (not crate-root).
use tetris_engine::engine::pieces::{preview_mask_4x4, Kind};

use crate::engine_dicts::{grid_features_to_dict, step_features_to_dict};
use crate::engine_helpers::{mask4_to_pyarray2, visible_grid_as_full_h};
use crate::expert_policy::ExpertPolicy;
use crate::util::{grid_rows_to_pyarray2, kind_glyph_to_idx};
use crate::warmup_spec::PyWarmupSpec;

#[pyclass]
pub struct TetrisEngine {
    pub(crate) g: Game,
    // Stored defaults for reset() reuse:
    rule: PieceRuleKind,
    warmup: WarmupSpec,
}

#[pymethods]
impl TetrisEngine {
    /// TetrisEngine(seed=12345, piece_rule="uniform", warmup=None)
    ///
    /// Seeding notes:
    /// - The Rust engine is deterministic given (seed, piece_rule, warmup).
    /// - For RL training, the *Python* environment should generate a fresh episode seed
    ///   each reset and pass it explicitly.
    #[new]
    #[pyo3(signature = (seed=12345, piece_rule="uniform", warmup=None))]
    fn new(seed: u64, piece_rule: &str, warmup: Option<PyWarmupSpec>) -> Self {
        let rule = PieceRuleKind::from_cli(piece_rule);
        let warmup = warmup.map(|w| w.into_inner()).unwrap_or_else(WarmupSpec::none);
        let g = Game::new_with_rule_and_warmup_spec(seed, rule, warmup);
        Self { g, rule, warmup }
    }

    // ---------------------------------------------------------------------
    // Constants / geometry getters
    // ---------------------------------------------------------------------

    /// Board width (W).
    fn board_w(&self) -> usize {
        W
    }

    /// Board height including hidden spawn rows (H).
    fn board_h(&self) -> usize {
        H
    }

    /// Number of hidden spawn rows at the top.
    fn hidden_rows(&self) -> usize {
        HIDDEN_ROWS
    }

    /// Visible board height (H - HIDDEN_ROWS).
    fn visible_h(&self) -> usize {
        H.saturating_sub(HIDDEN_ROWS)
    }

    /// Fixed rotation slots used by action-id encoding.
    ///
    /// NOTE:
    /// This is NOT "number of distinct rotations of the current piece".
    /// Distinct rotations are enforced by the engine's validity/mask logic.
    fn max_rots(&self) -> usize {
        MAX_ROTS
    }

    /// Fixed action-space dimension (MAX_ROTS * W).
    fn action_dim(&self) -> usize {
        ACTION_DIM
    }

    // ---------------------------------------------------------------------
    // Piece rule reporting (UI/logging convenience)
    // ---------------------------------------------------------------------

    /// piece_rule() -> "uniform" | "bag7"
    ///
    /// Convenience getter so Python code doesn't have to depend on snapshot keys.
    fn piece_rule(&self) -> &'static str {
        match self.g.piece_rule_kind() {
            PieceRuleKind::Uniform => "uniform",
            PieceRuleKind::Bag7 => "bag7",
        }
    }

    // ---------------------------------------------------------------------
    // Action-id helpers (SSOT: bind constants::{encode_action_id, decode_action_id})
    // ---------------------------------------------------------------------

    /// encode_action_id(rot, col) -> action_id
    ///
    /// Fixed action encoding, authoritative in Rust engine.
    /// - rot is wrapped by MAX_ROTS
    /// - col must be in [0, W)
    fn encode_action_id(&self, rot: usize, col: usize) -> PyResult<usize> {
        if col >= W {
            return Err(PyValueError::new_err(format!(
                "encode_action_id: col out of range: {col} (expected 0..{W})"
            )));
        }
        Ok(encode_action_id(rot % MAX_ROTS, col))
    }

    /// decode_action_id(action_id) -> (rot, col)
    ///
    /// Inverse of encode_action_id, authoritative in Rust engine.
    fn decode_action_id(&self, action_id: usize) -> PyResult<(usize, usize)> {
        if action_id >= ACTION_DIM {
            return Err(PyValueError::new_err(format!(
                "decode_action_id: action_id out of range: {action_id} (expected 0..{ACTION_DIM})"
            )));
        }
        Ok(decode_action_id(action_id))
    }

    // ---------------------------------------------------------------------
    // UI-only helpers (piece preview layout)
    // ---------------------------------------------------------------------

    /// kind_preview_mask(kind_idx, rot=0) -> uint8[4,4]
    ///
    /// UI-only helper:
    /// - Returns a 4x4 mask for the given piece kind and rotation.
    /// - Filled cells contain the piece id (1..=7), empty cells are 0.
    ///
    /// Notes:
    /// - `kind_idx` matches the engine grid encoding (0 = empty, 1..=7 = piece kind).
    /// - `rot` is clamped to the last *distinct* rotation for that kind.
    /// - This method must only be called by rendering / watch-mode code, never in the env step path.
    #[pyo3(signature = (kind_idx, rot=0))]
    fn kind_preview_mask<'py>(
        &self,
        py: Python<'py>,
        kind_idx: u8,
        rot: usize,
    ) -> PyResult<Bound<'py, PyArray2<u8>>> {
        let kind = Kind::from_idx(kind_idx).ok_or_else(|| {
            PyValueError::new_err(format!(
                "kind_preview_mask: invalid kind_idx {kind_idx} (expected 1..=7)"
            ))
        })?;

        let m = preview_mask_4x4(kind, rot, kind.idx());
        Ok(mask4_to_pyarray2(py, m))
    }

    // ---------------------------------------------------------------------
    // Episode control
    // ---------------------------------------------------------------------

    /// reset(seed=None, piece_rule=None, warmup=None)
    ///
    /// IMPORTANT:
    /// - `seed` MUST be provided. If seed is None, this raises ValueError.
    /// - This is intentional: Gymnasium envs should generate an episode seed in Python and pass it in.
    ///
    /// Determinism:
    /// - Same (seed, piece_rule, warmup) => identical episode (piece stream + warmup noise).
    #[pyo3(signature = (seed=None, piece_rule=None, warmup=None))]
    fn reset(
        &mut self,
        seed: Option<u64>,
        piece_rule: Option<&str>,
        warmup: Option<PyWarmupSpec>,
    ) -> PyResult<()> {
        let seed = seed.ok_or_else(|| {
            PyValueError::new_err(
                "reset(seed=None) is not allowed. Pass an explicit episode seed (generated by the Python env).",
            )
        })?;

        if let Some(pr) = piece_rule {
            self.rule = PieceRuleKind::from_cli(pr);
        }
        if let Some(w) = warmup {
            self.warmup = w.into_inner();
        }

        self.g = Game::new_with_rule_and_warmup_spec(seed, self.rule, self.warmup);
        Ok(())
    }

    /// Returns (terminated, cleared_lines, invalid_action).
    fn step_action_id(&mut self, action_id: usize) -> (bool, u32, bool) {
        let r = self.g.step_action_id(action_id);
        (r.terminated, r.cleared_lines, r.invalid_action)
    }

    /// Convenience helper: compute expert action + step once.
    /// Returns (terminated, cleared_lines, invalid_action, action_id).
    ///
    /// If the expert has no valid action, returns (true, 0, false, None).
    fn step_expert(&mut self, policy: &mut ExpertPolicy) -> (bool, u32, bool, Option<usize>) {
        let Some(aid) = policy.inner.action_id(&self.g) else {
            return (true, 0, false, None);
        };
        let (terminated, cleared, invalid) = self.step_action_id(aid);
        (terminated, cleared, invalid, Some(aid))
    }

    // ---------------------------------------------------------------------
    // Mask / valid actions
    // ---------------------------------------------------------------------

    /// Returns mask as uint8 array of shape (ACTION_DIM,): 1 = valid, 0 = invalid.
    fn action_mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        let m = self.g.action_mask();
        let v: Vec<u8> = m.into_iter().map(|b| if b { 1 } else { 0 }).collect();
        PyArray1::from_vec_bound(py, v)
    }

    /// valid_action_ids() -> list[int]
    fn valid_action_ids(&self) -> Vec<usize> {
        self.g.valid_action_ids()
    }

    // ---------------------------------------------------------------------
    // Grids
    // ---------------------------------------------------------------------

    /// Returns grid as uint8 array of shape (H, W).
    fn grid<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u8>> {
        grid_rows_to_pyarray2(py, &self.g.grid, 0, H)
    }

    /// Returns the *visible* grid as uint8 array of shape (H - HIDDEN_ROWS, W).
    fn visible_grid<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u8>> {
        grid_rows_to_pyarray2(py, &self.g.grid, HIDDEN_ROWS, H)
    }

    // ---------------------------------------------------------------------
    // Snapshot (dict)
    // ---------------------------------------------------------------------

    /// snapshot(include_grid=False, visible=True) -> dict
    ///
    /// Returns a small Python dict containing the engine state.
    /// If include_grid is True, includes "grid" (visible or full depending on `visible`).
    #[pyo3(signature = (include_grid=false, visible=true))]
    fn snapshot<'py>(
        &self,
        py: Python<'py>,
        include_grid: bool,
        visible: bool,
    ) -> PyResult<PyObject> {
        let d = PyDict::new_bound(py);

        let active = self.g.active.glyph().to_string();
        let next = self.g.next.glyph().to_string();

        d.set_item("score", self.g.score)?;
        d.set_item("lines", self.g.lines_cleared)?;
        d.set_item("steps", self.g.steps)?;
        d.set_item("game_over", self.g.game_over)?;

        d.set_item("active_kind", &active)?;
        d.set_item("next_kind", &next)?;

        d.set_item("active_kind_idx", kind_glyph_to_idx(&active)?)?;
        d.set_item("next_kind_idx", kind_glyph_to_idx(&next)?)?;

        // Piece rule reporting for env logging/UI.
        d.set_item("piece_rule", self.piece_rule())?;

        if include_grid {
            let g = if visible { self.visible_grid(py) } else { self.grid(py) };
            d.set_item("grid", g)?;
        }

        Ok(d.into_py(py))
    }

    // ---------------------------------------------------------------------
    // Features (VISIBLE GRID ONLY)
    // ---------------------------------------------------------------------

    /// grid_features() -> dict
    ///
    /// Computes classic locked-grid features (max_h, agg_h, holes, bump)
    /// on the VISIBLE rows only (HIDDEN_ROWS..H).
    fn grid_features<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let grid_vis = visible_grid_as_full_h(&self.g.grid);
        let f = compute_grid_features(&grid_vis);
        Ok(grid_features_to_dict(py, f).into_py(py))
    }

    /// step_features(prev=None) -> dict
    ///
    /// Computes current grid features plus deltas vs `prev` on the VISIBLE rows only.
    /// `prev` is either None or a 4-tuple: (max_h, agg_h, holes, bump).
    #[pyo3(signature = (prev=None))]
    fn step_features<'py>(
        &self,
        py: Python<'py>,
        prev: Option<(u32, u32, u32, u32)>,
    ) -> PyResult<PyObject> {
        let prev_f = prev.map(|(max_h, agg_h, holes, bump)| GridFeatures {
            max_h,
            agg_h,
            holes,
            bump,
        });

        let grid_vis = visible_grid_as_full_h(&self.g.grid);
        let sf = compute_step_features(&grid_vis, prev_f);
        Ok(step_features_to_dict(py, sf).into_py(py))
    }

    // ---------------------------------------------------------------------
    // Optional: simulation (pure, no mutation)
    // ---------------------------------------------------------------------

    /// simulate_active(action_id, include_grids=False, visible=True) -> dict
    #[pyo3(signature = (action_id, include_grids=false, visible=true))]
    fn simulate_active<'py>(
        &self,
        py: Python<'py>,
        action_id: usize,
        include_grids: bool,
        visible: bool,
    ) -> PyResult<PyObject> {
        let sim: SimPlacement = self.g.simulate_action_id_active(action_id);

        let d = PyDict::new_bound(py);
        d.set_item("cleared_lines", sim.cleared_lines)?;
        d.set_item("invalid", sim.invalid)?;

        if include_grids {
            let (start, end) = if visible { (HIDDEN_ROWS, H) } else { (0, H) };

            let g_lock = grid_rows_to_pyarray2(py, &sim.grid_after_lock, start, end);
            let g_clear = grid_rows_to_pyarray2(py, &sim.grid_after_clear, start, end);
            d.set_item("grid_after_lock", g_lock)?;
            d.set_item("grid_after_clear", g_clear)?;
        }

        Ok(d.into_py(py))
    }

    // ---------------------------------------------------------------------
    // Scalars
    // ---------------------------------------------------------------------

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
