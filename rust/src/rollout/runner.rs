// src/rollout/runner.rs
#![forbid(unsafe_code)]

use std::time::Duration;

use indicatif::{ProgressBar, ProgressStyle};

use crate::engine::{Game, PieceRuleKind};
use crate::policy::Policy;

use super::features::{compute_step_features, GridFeatures};
use super::sink::RolloutSink;
use super::stats::{FinalReport, RolloutStats};
use super::table::ReportRow;

#[derive(Clone, Debug)]
pub struct RunnerConfig {
    // ---------------- core rollout ----------------
    /// Total placements to execute across episodes.
    pub steps: u64,
    /// Base seed; each episode uses base_seed + episode_id.
    pub base_seed: u64,
    pub rule_kind: PieceRuleKind,

    // ---------------- optional rendering ----------------
    pub render: bool,
    pub sleep_ms: u64,

    // ---------------- live status ----------------
    pub perf: bool,
    pub progress: bool,
    /// Update progress message/perf line every N steps.
    pub stats_every: u64,

    // ---------------- periodic table reporting ----------------
    /// Print a stats row every N steps. 0 disables reporting completely.
    pub report_every: u64,
    /// If true, compute and include heavier grid features (holes/bumpiness/agg height + deltas).
    /// These are computed ONLY on report cadence, not every step.
    pub report_features: bool,
    /// Reprint the table header every N printed rows.
    pub report_header_every: u64,

    /// Used only for the final report string.
    pub policy_name: String,
}

pub struct Runner {
    cfg: RunnerConfig,
    sink: Box<dyn RolloutSink>,
}

impl Runner {
    pub fn new(cfg: RunnerConfig, sink: Box<dyn RolloutSink>) -> Self {
        Self { cfg, sink }
    }

    pub fn run(&mut self, policy: &mut dyn Policy) -> FinalReport {
        let cfg = self.cfg.clone();

        // Progress bar is purely UI; runner logic works without it.
        let pb = if cfg.progress {
            let pb = ProgressBar::new(cfg.steps);
            pb.set_style(
                ProgressStyle::with_template(
                    "{bar:40.cyan/blue} {pos:>9}/{len:<9}  {percent:>3}%  {elapsed_precise}  {msg}",
                )
                .unwrap()
                .progress_chars("=>-"),
            );
            Some(pb)
        } else {
            None
        };

        let mut stats = RolloutStats::new();

        // Episode state.
        let mut episode_id: u64 = 0;
        let mut game = Game::new_with_rule(cfg.base_seed.wrapping_add(episode_id), cfg.rule_kind);

        // Totals across *completed* episodes (live totals include current episode too).
        let mut total_lines_finished: u64 = 0;
        let mut total_score_finished: u64 = 0;

        // For delta features: last reported (not last step) features.
        let mut prev_features: Option<GridFeatures> = None;

        if cfg.render {
            print!("{}", game.render_ascii());
        }

        let stats_every = cfg.stats_every.max(1);

        while stats.steps_done < cfg.steps {
            // ------------------------------------------------------------
            // Episode boundary: finalize counters, then reset environment.
            // ------------------------------------------------------------
            if game.game_over {
                stats.on_episode_end(game.lines_cleared, game.score);
                total_lines_finished += game.lines_cleared;
                total_score_finished += game.score;

                episode_id += 1;
                game = Game::new_with_rule(cfg.base_seed.wrapping_add(episode_id), cfg.rule_kind);
                prev_features = None; // don't carry deltas across episodes

                if cfg.render {
                    println!(
                        "=== reset: episodes_finished={} avg_ep_len={:.2} max_ep_len={} ===",
                        stats.episodes_finished,
                        stats.avg_ep_len(),
                        stats.episode_len_max
                    );
                    print!("{}", game.render_ascii());
                }
                continue;
            }

            // ------------------------------------------------------------
            // One placement: policy chooses (rot, bbox-left col).
            // ------------------------------------------------------------
            let (rot, col) = match policy.choose_action(&game) {
                Some(x) => x,
                None => {
                    game.game_over = true;
                    continue;
                }
            };

            let (_term, cleared) = game.step_macro(rot, col);

            // Light-weight per-step stats (always on; cheap).
            let (mh, ah) = game.height_metrics();
            stats.on_step(cleared, mh, ah);

            if let Some(ref pb) = pb {
                pb.inc(1);
            }

            if cfg.render {
                println!(
                    "step={} action=(rot={}, col={}) cleared={}",
                    stats.steps_done, rot, col, cleared
                );
                print!("{}", game.render_ascii());
                std::thread::sleep(Duration::from_millis(cfg.sleep_ms));
            }

            // ------------------------------------------------------------
            // Periodic table report: build a ReportRow and hand it to sink.
            // This is off by default; when enabled, heavier features are
            // computed ONLY on this cadence.
            // ------------------------------------------------------------
            if cfg.report_every > 0 && (stats.steps_done % cfg.report_every == 0) {
                let live_total_lines = total_lines_finished + game.lines_cleared;
                let live_total_score = total_score_finished + game.score;

                let features_opt = if cfg.report_features {
                    let sf = compute_step_features(&game.grid, prev_features);
                    prev_features = Some(sf.cur);
                    Some(sf)
                } else {
                    None
                };

                let row = ReportRow {
                    step: stats.steps_done,
                    steps_total: cfg.steps,
                    sps: stats.steps_per_sec(),
                    episodes_finished: stats.episodes_finished,
                    avg_ep_len: stats.avg_ep_len(),
                    max_ep_len: stats.episode_len_max,
                    lines_per_step: if stats.steps_done > 0 {
                        live_total_lines as f64 / stats.steps_done as f64
                    } else {
                        0.0
                    },
                    score_per_step: if stats.steps_done > 0 {
                        live_total_score as f64 / stats.steps_done as f64
                    } else {
                        0.0
                    },
                    avg_max_h: stats.avg_max_h(),
                    avg_h: stats.avg_avg_h(),
                    total_lines: live_total_lines,
                    total_score: live_total_score,
                    features: features_opt,
                };

                self.sink.on_report(&row);
            }

            // ------------------------------------------------------------
            // Live progress message/perf line cadence.
            // ------------------------------------------------------------
            if (cfg.perf || cfg.progress) && (stats.steps_done % stats_every == 0) {
                let live_total_lines = total_lines_finished + game.lines_cleared;
                let live_total_score = total_score_finished + game.score;

                let live = stats.live_msg(cfg.rule_kind, live_total_lines, live_total_score);

                if let Some(ref pb) = pb {
                    pb.set_message(live.msg);
                } else if cfg.perf {
                    println!(
                        "stats: steps_done={}/{} {}",
                        stats.steps_done, cfg.steps, live.msg
                    );
                }
            }
        }

        // Include current in-progress episode in totals.
        let total_lines = total_lines_finished + game.lines_cleared;
        let total_score = total_score_finished + game.score;

        if let Some(pb) = pb {
            pb.finish_with_message("done");
        }

        stats.final_report(
            &cfg.policy_name,
            cfg.rule_kind,
            total_lines,
            total_score,
            stats.ep_len,
            game.game_over,
        )
    }
}
