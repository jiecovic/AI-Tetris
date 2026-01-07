// src/rollout/runner.rs
#![forbid(unsafe_code)]

use std::time::Duration;

use indicatif::{ProgressBar, ProgressStyle};

use tetris_engine::engine::Game;
use tetris_engine::policy::Policy;

use super::sinks::{ReportRow, RolloutSink};
use super::stats::{FinalReport, RolloutStats};

/// Fixed internal cadence for progress-bar live message updates.
/// (No CLI knob on purpose.)
const LIVE_EVERY: u64 = 200;

#[derive(Clone, Debug)]
pub struct RunnerConfig {
    // ---------------- core rollout ----------------
    /// Total placements to execute across episodes.
    pub steps: u64,
    /// Base seed; each episode uses base_seed + episode_id.
    pub base_seed: u64,
    pub rule_kind: tetris_engine::engine::PieceRuleKind,

    /// Used only for the final report string.
    pub policy_name: String,

    // ---------------- warmup ----------------
    /// Number of bottom rows to fill with warmup garbage on episode reset.
    /// 0 disables warmup.
    pub warmup_rows: u8,
    /// Number of holes per warmed row (clamped in engine).
    pub warmup_holes: u8,

    // ---------------- output ----------------
    /// 0 = final summary only
    /// 1 = progress bar
    /// 2 = progress bar + periodic table (via sink)
    pub verbosity: u8,

    /// Print a table row every N steps (only used when verbosity == 2).
    /// 0 disables table reporting.
    pub report_every: u64,

    // ---------------- rendering ----------------
    /// If Some(ms): render every step; sleep ms between frames (0 = no sleep).
    pub render_ms: Option<u64>,
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

        // Progress bar is UI only; runner logic does not depend on it.
        let pb = if cfg.verbosity >= 1 {
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
        let mut game = Game::new_with_rule_and_warmup(
            cfg.base_seed.wrapping_add(episode_id),
            cfg.rule_kind,
            cfg.warmup_rows,
            cfg.warmup_holes,
        );

        // Totals across completed episodes (live totals include current episode too).
        let mut total_lines_finished: u64 = 0;
        let mut total_score_finished: u64 = 0;

        // Rendering is a separate axis from verbosity.
        if cfg.render_ms.is_some() {
            print!("{}", game.render_ascii());
        }

        while stats.steps_done < cfg.steps {
            // ------------------------------------------------------------
            // Episode boundary: finalize counters, then reset.
            // ------------------------------------------------------------
            if game.game_over {
                // Stats: finalize episode length counters + reset per-episode delta baseline.
                stats.on_episode_end();

                // Runner: accumulate totals from finished episode.
                total_lines_finished += game.lines_cleared;
                total_score_finished += game.score;

                // Reset env
                episode_id += 1;
                game = Game::new_with_rule_and_warmup(
                    cfg.base_seed.wrapping_add(episode_id),
                    cfg.rule_kind,
                    cfg.warmup_rows,
                    cfg.warmup_holes,
                );

                if cfg.render_ms.is_some() {
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
            // One placement: policy chooses action_id.
            // ------------------------------------------------------------
            let aid = match policy.choose_action(&game) {
                Some(aid) => aid,
                None => {
                    game.game_over = true;
                    continue;
                }
            };

            let r = game.step_action_id(aid);
            let _ = r.terminated; // game.game_over is the canonical termination flag.


            // Stats update (includes heavy features + deltas internally).
            let (mh, ah) = game.height_metrics();
            stats.on_step(&game.grid, mh, ah);

            if let Some(ref pb) = pb {
                pb.inc(1);
            }

            // Rendering (ASCII) every step when enabled.
            if let Some(ms) = cfg.render_ms {
                println!("step={} action_id={} lines={}", stats.steps_done, aid, game.lines_cleared);
                print!("{}", game.render_ascii());
                if ms > 0 {
                    std::thread::sleep(Duration::from_millis(ms));
                }
            }

            // ------------------------------------------------------------
            // Periodic table report (verbosity == 2 only).
            // IMPORTANT: the table prints only AGGREGATE stats.
            // ------------------------------------------------------------
            if cfg.verbosity == 2
                && cfg.report_every > 0
                && (stats.steps_done % cfg.report_every == 0)
            {
                let live_total_lines = total_lines_finished + game.lines_cleared;
                let live_total_score = total_score_finished + game.score;

                let row = ReportRow {
                    step: stats.steps_done,
                    steps_total: cfg.steps,
                    sps: stats.steps_per_sec(),

                    episodes_finished: stats.episodes_finished,
                    avg_ep_len: stats.avg_ep_len(),
                    max_ep_len: stats.episode_len_max,

                    lines_per_step: stats.lines_per_step(live_total_lines),
                    score_per_step: stats.score_per_step(live_total_score),

                    // NOTE: these fields must exist in ReportRow + be filled from stats getters,
                    // otherwise you'll get a "missing fields" error.
                    max_h_worst: stats.max_h_worst,
                    avg_max_h: stats.avg_max_h(),
                    avg_avg_h: stats.avg_avg_h(),

                    avg_agg_h: stats.avg_agg_h(),
                    avg_holes: stats.avg_holes(),
                    avg_bump: stats.avg_bump(),

                    avg_d_max_h: stats.avg_d_max_h(),
                    avg_d_agg_h: stats.avg_d_agg_h(),
                    avg_d_holes: stats.avg_d_holes(),
                    avg_d_bump: stats.avg_d_bump(),
                };

                self.sink.on_report_row(&row, pb.as_ref());
            }

            // ------------------------------------------------------------
            // Live progress message cadence (fixed internal cadence).
            // ------------------------------------------------------------
            if cfg.verbosity >= 1 && (stats.steps_done % LIVE_EVERY == 0) {
                let live_total_lines = total_lines_finished + game.lines_cleared;
                let live_total_score = total_score_finished + game.score;

                let lps = stats.lines_per_step(live_total_lines);
                let sps = stats.score_per_step(live_total_score);

                let msg = stats.live_msg(cfg.rule_kind, lps, sps);

                if let Some(ref pb) = pb {
                    pb.set_message(msg);
                }
            }
        }

        // Include current in-progress episode in totals.
        let total_lines = total_lines_finished + game.lines_cleared;
        let total_score = total_score_finished + game.score;

        if let Some(pb) = pb {
            pb.finish_with_message("done");
        }

        // Final report is still created by stats (stable end-of-run struct).
        stats.final_report(
            &cfg.policy_name,
            cfg.rule_kind,
            cfg.warmup_rows,
            cfg.warmup_holes,
            total_lines,
            total_score,
            stats.ep_len,
            game.game_over,
        )
    }
}
