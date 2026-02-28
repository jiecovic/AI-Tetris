// rust/py/src/util.rs
#![forbid(unsafe_code)]

use numpy::PyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use tetris_engine::{H, W};

/**
 * Convert a Rust grid `[[u8; W]; H]` into a NumPy array with row range [r0, r1).
 *
 * This intentionally copies into a Vec<Vec<u8>> for a simple, safe PyO3 surface.
 * (We can optimize later with a contiguous buffer if it becomes a bottleneck.)
 */
pub(crate) fn grid_rows_to_pyarray2<'py>(
    py: Python<'py>,
    grid: &[[u8; W]; H],
    r0: usize,
    r1: usize,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let r0 = r0.min(H);
    let r1 = r1.min(H);
    let r0 = r0.min(r1);

    let mut rows: Vec<Vec<u8>> = Vec::with_capacity(r1 - r0);
    for row in grid.iter().take(r1).skip(r0) {
        rows.push(row.to_vec());
    }

    PyArray2::from_vec2_bound(py, &rows).map_err(|e| {
        PyValueError::new_err(format!(
            "grid_rows_to_pyarray2: failed to build numpy array from grid rows [{r0},{r1}): {e}"
        ))
    })
}

/// Map a Kind glyph ("I","O","T","S","Z","J","L") to strict kind_idx (0..6).
pub(crate) fn kind_glyph_to_idx(glyph: &str) -> PyResult<u8> {
    match glyph {
        "I" => Ok(0),
        "O" => Ok(1),
        "T" => Ok(2),
        "S" => Ok(3),
        "Z" => Ok(4),
        "J" => Ok(5),
        "L" => Ok(6),
        _ => Err(PyValueError::new_err(format!(
            "unknown kind glyph: {glyph:?} (expected one of I,O,T,S,Z,J,L)"
        ))),
    }
}
