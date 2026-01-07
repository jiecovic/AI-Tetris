// rust/py/src/expert_policy.rs
#![forbid(unsafe_code)]

use pyo3::prelude::*;

use tetris_engine::policy::{BeamConfig, Codemy0, Codemy1, Codemy2, Codemy2FastPolicy, Policy};
use tetris_engine::Game;

use crate::engine::TetrisEngine;

#[pyclass]
pub struct ExpertPolicy {
    pub(crate) inner: ExpertPolicyInner,
}

pub(crate) enum ExpertPolicyInner {
    Codemy0(Codemy0),
    Codemy1(Codemy1),
    Codemy2(Codemy2),
    Codemy2Fast(Codemy2FastPolicy),
}

impl ExpertPolicyInner {
    pub(crate) fn action_id(&mut self, g: &Game) -> Option<usize> {
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
