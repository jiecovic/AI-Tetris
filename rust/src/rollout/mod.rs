// src/rollout/mod.rs
#![forbid(unsafe_code)]

pub mod runner;
pub mod stats;
pub mod sinks;

pub use runner::{Runner, RunnerConfig};
pub use sinks::{NoopSink, RolloutSink, TableSink};
pub use stats::{FinalReport, RolloutStats};
