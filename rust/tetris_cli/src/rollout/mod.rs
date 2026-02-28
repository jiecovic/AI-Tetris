// src/rollout/mod.rs
#![forbid(unsafe_code)]

pub mod runner;
pub mod sinks;
pub mod stats;

pub use runner::{Runner, RunnerConfig};
pub use sinks::{NoopSink, RolloutSink, TableSink};
// pub use stats::{FinalReport, RolloutStats};
