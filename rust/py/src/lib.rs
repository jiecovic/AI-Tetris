// rust/py/src/lib.rs
#![forbid(unsafe_code)]
#![allow(unsafe_op_in_unsafe_fn)] // pyo3 macro-generated glue triggers this on Rust 2024

use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use tetris_engine::{Game, HoleCount, PieceRuleKind, RowCountDist, WarmupSpec, H, W};
use tetris_engine::policy::{BeamConfig, Codemy0, Codemy1, Codemy2, Codemy2FastPolicy, Policy};

#[pyclass(name = "WarmupSpec")]
#[derive(Clone)]
pub struct PyWarmupSpec {
    inner: WarmupSpec,
}

#[pymethods]
impl PyWarmupSpec {
    /// WarmupSpec.none()
    ///
    /// No warmup noise (clean board).
    #[staticmethod]
    fn none() -> Self {
        Self {
            inner: WarmupSpec::none(),
        }
    }

    /// WarmupSpec.fixed(rows, holes=1, spawn_buffer=None)
    ///
    /// Fixed number of garbage rows and fixed holes per row.
    #[staticmethod]
    #[pyo3(signature = (rows, holes=1, spawn_buffer=None))]
    fn fixed(rows: u8, holes: u8, spawn_buffer: Option<u8>) -> Self {
        let mut s = WarmupSpec {
            rows: RowCountDist::Fixed(rows),
            holes: HoleCount::Fixed(holes),
            ..WarmupSpec::none()
        };
        if let Some(sb) = spawn_buffer {
            s.spawn_buffer = sb;
        }
        Self { inner: s }
    }

    /// WarmupSpec.uniform_rows(min_rows, max_rows, holes=1, spawn_buffer=None)
    ///
    /// Uniformly sample number of garbage rows from [min_rows, max_rows] (inclusive).
    #[staticmethod]
    #[pyo3(signature = (min_rows, max_rows, holes=1, spawn_buffer=None))]
    fn uniform_rows(min_rows: u8, max_rows: u8, holes: u8, spawn_buffer: Option<u8>) -> Self {
        let mut s = WarmupSpec {
            rows: RowCountDist::Uniform {
                min: min_rows,
                max: max_rows,
            },
            holes: HoleCount::Fixed(holes),
            ..WarmupSpec::none()
        };
        if let Some(sb) = spawn_buffer {
            s.spawn_buffer = sb;
        }
        Self { inner: s }
    }

    /// WarmupSpec.poisson(lambda_, cap, holes=1, spawn_buffer=None)
    ///
    /// Sample rows ~ Poisson(lambda_) and clamp to cap (then engine also clamps to spawn-safe max).
    /// NOTE: Poisson is skewed (more mass below lambda_).
    #[staticmethod]
    #[pyo3(signature = (lambda_, cap, holes=1, spawn_buffer=None))]
    fn poisson(lambda_: f64, cap: u8, holes: u8, spawn_buffer: Option<u8>) -> Self {
        let mut s = WarmupSpec {
            rows: RowCountDist::Poisson { lambda: lambda_, cap },
            holes: HoleCount::Fixed(holes),
            ..WarmupSpec::none()
        };
        if let Some(sb) = spawn_buffer {
            s.spawn_buffer = sb;
        }
        Self { inner: s }
    }

    /// WarmupSpec.base_plus_poisson(base, lambda_, cap, holes=1, spawn_buffer=None)
    ///
    /// rows = base + Poisson(lambda_), clamped to cap (then engine also clamps to spawn-safe max).
    /// Good for "hard baseline + jitter": mostly hard starts, sometimes harder.
    #[staticmethod]
    #[pyo3(signature = (base, lambda_, cap, holes=1, spawn_buffer=None))]
    fn base_plus_poisson(
        base: u8,
        lambda_: f64,
        cap: u8,
        holes: u8,
        spawn_buffer: Option<u8>,
    ) -> Self {
        let mut s = WarmupSpec {
            rows: RowCountDist::BasePlusPoisson {
                base,
                lambda: lambda_,
                cap,
            },
            holes: HoleCount::Fixed(holes),
            ..WarmupSpec::none()
        };
        if let Some(sb) = spawn_buffer {
            s.spawn_buffer = sb;
        }
        Self { inner: s }
    }

    /// Return a copy with holes sampled uniformly from [min_holes, max_holes] (inclusive).
    #[pyo3(signature = (min_holes, max_holes))]
    fn with_uniform_holes(&self, min_holes: u8, max_holes: u8) -> Self {
        let mut s = self.inner;
        s.holes = HoleCount::Uniform {
            min: min_holes,
            max: max_holes,
        };
        Self { inner: s }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

impl PyWarmupSpec {
    fn into_inner(self) -> WarmupSpec {
        self.inner
    }
}

#[pyclass]
pub struct TetrisEngine {
    g: Game,
    rule: PieceRuleKind,
    warmup: WarmupSpec,
}

#[pyclass]
pub struct ExpertPolicy {
    inner: ExpertPolicyInner,
}

enum ExpertPolicyInner {
    Codemy0(Codemy0),
    Codemy1(Codemy1),
    Codemy2(Codemy2),
    Codemy2Fast(Codemy2FastPolicy),
}

impl ExpertPolicyInner {
    fn action_id(&mut self, g: &Game) -> Option<usize> {
        match self {
            ExpertPolicyInner::Codemy0(p) => Policy::choose_action(p, g),
            ExpertPolicyInner::Codemy1(p) => Policy::choose_action(p, g),
            ExpertPolicyInner::Codemy2(p) => Policy::choose_action(p, g),
            ExpertPolicyInner::Codemy2Fast(p) => Policy::choose_action(p, g),
        }
    }
}

#[pymethods]
impl ExpertPolicy {
    /// Codemy0 (1-ply) expert. Optional beam pruning.
    #[staticmethod]
    #[pyo3(signature = (beam_width=None, beam_from_depth=0))]
    fn codemy0(beam_width: Option<usize>, beam_from_depth: u8) -> Self {
        let beam = beam_width.map(|w| BeamConfig::new(beam_from_depth, w));
        Self {
            inner: ExpertPolicyInner::Codemy0(Codemy0::new(beam)),
        }
    }

    /// Codemy1 (2-ply) expert. Optional beam pruning.
    #[staticmethod]
    #[pyo3(signature = (beam_width=None, beam_from_depth=0))]
    fn codemy1(beam_width: Option<usize>, beam_from_depth: u8) -> Self {
        let beam = beam_width.map(|w| BeamConfig::new(beam_from_depth, w));
        Self {
            inner: ExpertPolicyInner::Codemy1(Codemy1::new(beam)),
        }
    }

    /// Codemy2 (3-ply) expert. Optional beam pruning.
    #[staticmethod]
    #[pyo3(signature = (beam_width=None, beam_from_depth=0))]
    fn codemy2(beam_width: Option<usize>, beam_from_depth: u8) -> Self {
        let beam = beam_width.map(|w| BeamConfig::new(beam_from_depth, w));
        Self {
            inner: ExpertPolicyInner::Codemy2(Codemy2::new(beam)),
        }
    }

    /// Codemy2Fast expert (codemy1 + cheap one-step tail).
    #[staticmethod]
    #[pyo3(signature = (tail_weight=0.5))]
    fn codemy2fast(tail_weight: f64) -> Self {
        Self {
            inner: ExpertPolicyInner::Codemy2Fast(Codemy2FastPolicy::new(tail_weight)),
        }
    }

    /// Compute an expert action id for the current state of `engine`.
    ///
    /// NOTE: This borrows the engine state (no copies).
    fn action_id(&mut self, engine: &TetrisEngine) -> Option<usize> {
        self.inner.action_id(&engine.g)
    }
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

    /// Returns (terminated, cleared_lines, illegal_action).
    fn step_action_id(&mut self, action_id: usize) -> (bool, u32, bool) {
        let r = self.g.step_action_id(action_id);
        (r.terminated, r.cleared_lines, r.illegal_action)
    }

    /// Convenience helper: compute expert action + step once.
    /// Returns (terminated, cleared_lines, illegal_action, action_id).
    ///
    /// If the expert has no legal action, returns (true, 0, false, None).
    fn step_expert(&mut self, policy: &mut ExpertPolicy) -> (bool, u32, bool, Option<usize>) {
        let Some(aid) = policy.inner.action_id(&self.g) else {
            return (true, 0, false, None);
        };
        let (terminated, cleared, illegal) = self.step_action_id(aid);
        (terminated, cleared, illegal, Some(aid))
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
    m.add_class::<ExpertPolicy>()?;
    m.add_class::<PyWarmupSpec>()?;
    Ok(())
}
