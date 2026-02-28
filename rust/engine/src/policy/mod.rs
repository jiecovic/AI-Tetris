// src/policy/mod.rs
#![forbid(unsafe_code)]

mod base;
mod beam;
mod codemy;
mod heuristic;
mod random;

/**
 * Curated policy public API.
 *
 * Internal implementation modules remain private; only stable policy entrypoints are re-exported.
 */
pub use base::Policy;
pub use beam::BeamConfig;
pub use codemy::{Codemy0, Codemy1, Codemy2, Codemy2FastPolicy, CodemyPolicy};
pub use heuristic::{
    HeuristicBuildError, HeuristicFeature, HeuristicPolicy, compute_feature_values,
};
pub use random::RandomPolicy;
