// src/rollout/mod.rs
#![forbid(unsafe_code)]

pub mod features;
pub mod runner;
pub mod sink;
pub mod stats;
pub mod table;

pub use runner::{Runner, RunnerConfig};
pub use sink::{NoopSink, RolloutSink, TableSink};
