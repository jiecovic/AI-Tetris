// src/rollout/mod.rs
#![forbid(unsafe_code)]

pub mod runner;
pub mod stats;

pub use runner::{Runner, RunnerConfig};
pub use stats::{FinalReport, LiveMsg, RolloutStats};
