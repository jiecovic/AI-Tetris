// src/rollout/runner.rs
#![forbid(unsafe_code)]

use std::time::Duration;

use indicatif::{ProgressBar, ProgressStyle};

use crate::engine::Game;
use crate::engine::PieceRuleKind;
use crate::policy::Policy;

use super::stats::{FinalReport, RolloutStats};

#[derive(Clone, Debug)]
pub struct RunnerConfig {
    pub steps: u64,
    pub base_seed: u64,
    pub rule_kind: PieceRuleKind,

    pub render: bool,
    pub sleep_ms: u64,

    pub perf: bool,
    pub progress: bool,
    pub stats_every: u64,

    /// Used only for the final report string.
    pub policy_name: String,
}

pub struct Runner {
    cfg: RunnerConfig,
}

impl Runner {
    pub fn new(cfg: RunnerConfig) -> Self {
        Self { cfg }
    }

    pub fn run(&mut self, policy: &mut dyn Policy) -> FinalReport {
        let cfg = self.cfg.clone();

        // Progress bar
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

        // Episode management
        let mut episode_id: u64 = 0;
        let mut game = Game::new_with_rule(cfg.base_seed.wrapping_add(episode_id), cfg.rule_kind);

        // Totals across completed episodes
        let mut total_lines_finished: u64 = 0;
        let mut total_score_finished: u64 = 0;

        if cfg.render {
            print!("{}", game.render_ascii());
        }

        let stats_every = cfg.stats_every.max(1);
        let mut pending_pb_inc: u64 = 0;

        while stats.steps_done < cfg.steps {
            // Episode ended? finalize + reset
            if game.game_over {
                stats.on_episode_end(game.lines_cleared, game.score);
                total_lines_finished += game.lines_cleared;
                total_score_finished += game.score;

                episode_id += 1;
                game = Game::new_with_rule(cfg.base_seed.wrapping_add(episode_id), cfg.rule_kind);

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

            // Choose action; if policy cannot act, terminate episode (reset next loop)
            let a = policy.choose_action(&game);
            let (rot, col) = match a {
                Some(x) => x,
                None => {
                    game.game_over = true;
                    continue;
                }
            };

            let (_term, cleared) = game.step_macro(rot, col);

            // Height metrics after the step
            let (mh, ah) = game.height_metrics();
            stats.on_step(cleared, mh, ah);

            if pb.is_some() {
                pending_pb_inc += 1;
            }

            if cfg.render {
                println!(
                    "step={} action=(rot={}, col={}) cleared={}",
                    stats.steps_done, rot, col, cleared
                );
                print!("{}", game.render_ascii());
                std::thread::sleep(Duration::from_millis(cfg.sleep_ms));
            }

            // Live stats flush
            if (cfg.perf || cfg.progress) && (stats.steps_done % stats_every == 0) {
                let live_total_lines = total_lines_finished + game.lines_cleared;
                let live_total_score = total_score_finished + game.score;

                let live = stats.live_msg(cfg.rule_kind, live_total_lines, live_total_score);

                if let Some(ref pb) = pb {
                    if pending_pb_inc > 0 {
                        pb.inc(pending_pb_inc);
                        pending_pb_inc = 0;
                    }
                    pb.set_message(live.msg);
                } else if cfg.perf {
                    println!(
                        "stats: steps_done={}/{} {}",
                        stats.steps_done, cfg.steps, live.msg
                    );
                }
            }
        }

        // Finalize totals including in-progress episode
        let total_lines = total_lines_finished + game.lines_cleared;
        let total_score = total_score_finished + game.score;

        if let Some(pb) = pb {
            if pending_pb_inc > 0 {
                pb.inc(pending_pb_inc);
            }
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
