// src/policy/codemy/unknown.rs
#![forbid(unsafe_code)]

use crate::engine::{H, Kind, W};

use super::core::{GridScorer, SearchCore};

/// Unknown future model (type-level), used for inlining in static policy variants.
pub trait UnknownModel<S: GridScorer> {
    fn expected_value(core: &SearchCore<S>, grid: &[[u8; W]; H], plies_left: u8, depth: u8) -> f64;
}

/// IID-uniform over the 7 kinds.
#[derive(Clone, Copy, Debug)]
pub struct UniformIID;

impl<S: GridScorer> UnknownModel<S> for UniformIID {
    #[inline]
    fn expected_value(core: &SearchCore<S>, grid: &[[u8; W]; H], plies_left: u8, depth: u8) -> f64 {
        let mut sum = 0.0;
        let mut n = 0.0;

        for &k in Kind::all() {
            sum += core.value_known_piece::<UniformIID>(grid, k, plies_left, depth);
            n += 1.0;
        }

        if n > 0.0 { sum / n } else { f64::NEG_INFINITY }
    }
}
