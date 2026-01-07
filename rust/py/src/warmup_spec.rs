// rust/py/src/warmup_spec.rs
#![forbid(unsafe_code)]

use pyo3::prelude::*;

use tetris_engine::{HoleCount, RowCountDist, WarmupSpec};

/// Python-facing warmup specification.
///
/// This wraps the Rust engine's `WarmupSpec` without duplicating engine logic.
/// Use the static constructors to build commonly used distributions.
#[pyclass(name = "WarmupSpec")]
#[derive(Clone)]
pub struct PyWarmupSpec {
    pub(crate) inner: WarmupSpec,
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
    pub(crate) fn into_inner(self) -> WarmupSpec {
        self.inner
    }
}
