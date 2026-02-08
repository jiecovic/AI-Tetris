// rust/py/src/engine_dicts.rs
#![forbid(unsafe_code)]

use pyo3::prelude::*;
use pyo3::types::PyDict;

use tetris_engine::engine::{GridDelta, GridFeatures, StepFeatures};

pub(crate) fn grid_features_to_dict<'py>(py: Python<'py>, f: GridFeatures) -> Bound<'py, PyDict> {
    let d = PyDict::new_bound(py);
    d.set_item("max_h", f.max_h).unwrap();
    d.set_item("agg_h", f.agg_h).unwrap();
    d.set_item("holes", f.holes).unwrap();
    d.set_item("bump", f.bump).unwrap();
    d
}

pub(crate) fn step_features_to_dict<'py>(
    py: Python<'py>,
    sf: StepFeatures,
    lock: Option<GridFeatures>,
    lock_delta: Option<GridDelta>,
) -> Bound<'py, PyDict> {
    let d = PyDict::new_bound(py);

    let cur = PyDict::new_bound(py);
    cur.set_item("max_h", sf.cur.max_h).unwrap();
    cur.set_item("agg_h", sf.cur.agg_h).unwrap();
    cur.set_item("holes", sf.cur.holes).unwrap();
    cur.set_item("bump", sf.cur.bump).unwrap();

    let delta = PyDict::new_bound(py);
    delta.set_item("d_max_h", sf.delta.d_max_h).unwrap();
    delta.set_item("d_agg_h", sf.delta.d_agg_h).unwrap();
    delta.set_item("d_holes", sf.delta.d_holes).unwrap();
    delta.set_item("d_bump", sf.delta.d_bump).unwrap();

    d.set_item("cur", cur).unwrap();
    d.set_item("delta", delta).unwrap();

    if let Some(lock_f) = lock {
        let lock_d = PyDict::new_bound(py);
        lock_d.set_item("max_h", lock_f.max_h).unwrap();
        lock_d.set_item("agg_h", lock_f.agg_h).unwrap();
        lock_d.set_item("holes", lock_f.holes).unwrap();
        lock_d.set_item("bump", lock_f.bump).unwrap();
        d.set_item("lock", lock_d).unwrap();
    }

    if let Some(lock_d) = lock_delta {
        let lock_delta_d = PyDict::new_bound(py);
        lock_delta_d.set_item("d_max_h", lock_d.d_max_h).unwrap();
        lock_delta_d.set_item("d_agg_h", lock_d.d_agg_h).unwrap();
        lock_delta_d.set_item("d_holes", lock_d.d_holes).unwrap();
        lock_delta_d.set_item("d_bump", lock_d.d_bump).unwrap();
        d.set_item("lock_delta", lock_delta_d).unwrap();
    }

    d
}
