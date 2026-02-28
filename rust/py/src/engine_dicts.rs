// rust/py/src/engine_dicts.rs
#![forbid(unsafe_code)]

use pyo3::prelude::*;
use pyo3::types::PyDict;

use tetris_engine::engine::{GridDelta, GridFeatures, StepFeatures};

pub(crate) fn grid_features_to_dict<'py>(
    py: Python<'py>,
    f: GridFeatures,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("max_h", f.max_h)?;
    d.set_item("agg_h", f.agg_h)?;
    d.set_item("holes", f.holes)?;
    d.set_item("bump", f.bump)?;
    Ok(d)
}

pub(crate) fn step_features_to_dict<'py>(
    py: Python<'py>,
    sf: StepFeatures,
    lock: Option<GridFeatures>,
    lock_delta: Option<GridDelta>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new_bound(py);

    let cur = PyDict::new_bound(py);
    cur.set_item("max_h", sf.cur.max_h)?;
    cur.set_item("agg_h", sf.cur.agg_h)?;
    cur.set_item("holes", sf.cur.holes)?;
    cur.set_item("bump", sf.cur.bump)?;

    let delta = PyDict::new_bound(py);
    delta.set_item("d_max_h", sf.delta.d_max_h)?;
    delta.set_item("d_agg_h", sf.delta.d_agg_h)?;
    delta.set_item("d_holes", sf.delta.d_holes)?;
    delta.set_item("d_bump", sf.delta.d_bump)?;

    d.set_item("cur", cur)?;
    d.set_item("delta", delta)?;

    if let Some(lock_f) = lock {
        let lock_d = PyDict::new_bound(py);
        lock_d.set_item("max_h", lock_f.max_h)?;
        lock_d.set_item("agg_h", lock_f.agg_h)?;
        lock_d.set_item("holes", lock_f.holes)?;
        lock_d.set_item("bump", lock_f.bump)?;
        d.set_item("lock", lock_d)?;
    }

    if let Some(lock_d) = lock_delta {
        let lock_delta_d = PyDict::new_bound(py);
        lock_delta_d.set_item("d_max_h", lock_d.d_max_h)?;
        lock_delta_d.set_item("d_agg_h", lock_d.d_agg_h)?;
        lock_delta_d.set_item("d_holes", lock_d.d_holes)?;
        lock_delta_d.set_item("d_bump", lock_d.d_bump)?;
        d.set_item("lock_delta", lock_delta_d)?;
    }

    Ok(d)
}
