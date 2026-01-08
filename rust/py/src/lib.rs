// rust/py/src/lib.rs
#![forbid(unsafe_code)]
#![allow(unsafe_op_in_unsafe_fn)] // pyo3 macro-generated glue triggers this on Rust 2024

mod engine;
mod engine_dicts;
mod engine_helpers;
mod expert_policy;
mod util;
mod warmup_spec;

use pyo3::prelude::*;

use crate::engine::TetrisEngine;
use crate::expert_policy::ExpertPolicy;
use crate::warmup_spec::PyWarmupSpec;

#[pymodule]
fn tetris_rl_engine(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Export order doesn't matter, but keep it explicit.
    m.add_class::<TetrisEngine>()?;
    m.add_class::<ExpertPolicy>()?;
    m.add_class::<PyWarmupSpec>()?;
    Ok(())
}
