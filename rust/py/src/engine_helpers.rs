// rust/py/src/engine_helpers.rs
#![forbid(unsafe_code)]

use numpy::PyArray2;
use pyo3::prelude::*;

use tetris_engine::{H, HIDDEN_ROWS, W};

/// Build a full-H grid buffer that is identical to the engine grid on visible rows,
/// and zero on hidden rows. This allows reuse of feature functions that expect [[u8; W]; H].
pub(crate) fn visible_grid_as_full_h(grid: &[[u8; W]; H]) -> [[u8; W]; H] {
    let mut out = [[0u8; W]; H];
    for r in HIDDEN_ROWS..H {
        out[r] = grid[r];
    }
    out
}

/// Convert a 4x4 mask ([[u8;4];4]) into a numpy uint8 array (4,4).
///
/// This keeps `#![forbid(unsafe_code)]` compatible by using a Vec-of-Vec
/// and numpy's safe constructors.
pub(crate) fn mask4_to_pyarray2<'py>(
    py: Python<'py>,
    m: [[u8; 4]; 4],
) -> Bound<'py, PyArray2<u8>> {
    let rows: Vec<Vec<u8>> = (0..4).map(|y| m[y].to_vec()).collect();
    PyArray2::from_vec2_bound(py, &rows).expect("mask4_to_pyarray2: shape must be 4x4")
}
