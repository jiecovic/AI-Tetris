// src/main.rs
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use std::time::{Duration, Instant};

mod game;
mod piece_rule;
mod pieces;
mod policy;
mod policy_codemy;

use piece_rule::PieceRuleKind;
use policy::{Policy, RandomPolicy};
use policy_codemy::{CodemyPolicy, Lookahead};

#[derive(Parser, Debug)]
#[command(name = "tetris_cli")]
struct Args {
    /// Render board as ASCII each step
    #[arg(long)]
    render: bool,

    /// How many placements to run TOTAL (across episodes)
    #[arg(long, default_value_t = 200)]
    steps: usize,

    /// RNG seed (optional)
    #[arg(long)]
    seed: Option<u64>,

    /// Sleep ms between renders
    #[arg(long, default_value_t = 50)]
    sleep_ms: u64,

    /// Policy: random | codemy0 | codemy1
    #[arg(long, default_value = "random")]
    policy: String,

    /// Piece rule: uniform | bag7
    #[arg(long, default_value = "uniform")]
    piece_rule: String,

    /// Print periodic performance stats (steps/sec)
    #[arg(long)]
    perf: bool,

    /// Progress bar
    #[arg(long)]
    progress: bool,

    /// Update live stats every N steps (and progress bar increments will be flushed on this cadence)
    #[arg(long, default_value_t = 10_000)]
    stats_every: u64,
}

fn main() {
    let args = Args::parse();

    let base_seed = args.seed.unwrap_or(12345);
    let rule_kind = PieceRuleKind::from_cli(&args.piece_rule);

    // Choose policy (Policy trait returns (rot, col_left))
    let mut policy: Box<dyn Policy> = match args.policy.as_str() {
        "codemy0" => Box::new(CodemyPolicy::new(Lookahead::D0)),
        "codemy1" => Box::new(CodemyPolicy::new(Lookahead::D1)),
        _ => Box::new(RandomPolicy::new(base_seed.wrapping_add(999))),
    };

    // Progress bar setup
    let pb = if args.progress {
        let pb = ProgressBar::new(args.steps as u64);
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

    // Episode management / stats
    let mut episode_id: u64 = 0;
    let mut game = game::Game::new_with_rule(base_seed.wrapping_add(episode_id), rule_kind);

    let mut ep_len: u64 = 0;
    let mut episodes_finished: u64 = 0;
    let mut episode_len_sum: u64 = 0;
    let mut episode_len_max: u64 = 0;

    // Aggregate across all episodes (for totals + per-step averages)
    let mut total_lines: u64 = 0;
    let mut total_score: u64 = 0;

    // Height tracking (per step, locked-grid metrics)
    let mut sum_max_h: f64 = 0.0;
    let mut sum_avg_h: f64 = 0.0;

    if args.render {
        print!("{}", game.render_ascii());
    }

    let t0 = Instant::now();
    let mut steps_done: u64 = 0;

    let stats_every: u64 = args.stats_every.max(1);
    let mut pending_pb_inc: u64 = 0;

    while (steps_done as usize) < args.steps {
        // If episode ended, finalize stats and reset
        if game.game_over {
            episodes_finished += 1;
            episode_len_sum += ep_len;
            episode_len_max = episode_len_max.max(ep_len);

            total_lines += game.lines_cleared;
            total_score += game.score;

            episode_id += 1;
            game = game::Game::new_with_rule(base_seed.wrapping_add(episode_id), rule_kind);
            ep_len = 0;

            if args.render {
                println!(
                    "=== reset: episodes_finished={} avg_ep_len={:.2} max_ep_len={} ===",
                    episodes_finished,
                    if episodes_finished > 0 {
                        episode_len_sum as f64 / episodes_finished as f64
                    } else {
                        0.0
                    },
                    episode_len_max
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

        steps_done += 1;
        ep_len += 1;

        // Height metrics after the step (locked grid)
        let (mh, ah) = game.height_metrics();
        sum_max_h += mh as f64;
        sum_avg_h += ah as f64;

        if pb.is_some() {
            pending_pb_inc += 1;
        }

        if args.render {
            println!(
                "step={} action=(rot={}, col={}) cleared={}",
                steps_done, rot, col, cleared
            );
            print!("{}", game.render_ascii());
            std::thread::sleep(Duration::from_millis(args.sleep_ms));
        }

        // Live stats (and progress bar flush) on cadence
        if (args.perf || args.progress) && (steps_done % stats_every == 0) {
            let dt = t0.elapsed().as_secs_f64();
            let sps = if dt > 0.0 { steps_done as f64 / dt } else { 0.0 };

            let avg_ep_len = if episodes_finished > 0 {
                episode_len_sum as f64 / episodes_finished as f64
            } else {
                0.0
            };

            // totals don’t include the current in-progress episode until it ends,
            // so include current episode's counters for "live" per-step numbers
            let lines_per_step = if steps_done > 0 {
                (total_lines + game.lines_cleared) as f64 / steps_done as f64
            } else {
                0.0
            };

            let score_per_step = if steps_done > 0 {
                (total_score + game.score) as f64 / steps_done as f64
            } else {
                0.0
            };

            let avg_max_h = if steps_done > 0 {
                sum_max_h / steps_done as f64
            } else {
                0.0
            };

            let avg_avg_h = if steps_done > 0 {
                sum_avg_h / steps_done as f64
            } else {
                0.0
            };

            let msg = format!(
                "rule={:?} sps={:.1} eps_done={} avg_ep_len={:.1} max_ep_len={} l/step={:.3} score/step={:.2} avg_max_h={:.2} avg_h={:.2}",
                rule_kind,
                sps,
                episodes_finished,
                avg_ep_len,
                episode_len_max,
                lines_per_step,
                score_per_step,
                avg_max_h,
                avg_avg_h
            );

            if let Some(ref pb) = pb {
                if pending_pb_inc > 0 {
                    pb.inc(pending_pb_inc);
                    pending_pb_inc = 0;
                }
                pb.set_message(msg);
            } else if args.perf {
                println!("stats: steps_done={}/{} {}", steps_done, args.steps, msg);
            }
        }
    }

    // Add current in-progress episode to totals for final report
    total_lines += game.lines_cleared;
    total_score += game.score;

    let dt = t0.elapsed().as_secs_f64();
    let sps = if dt > 0.0 { steps_done as f64 / dt } else { 0.0 };

    let avg_ep_len = if episodes_finished > 0 {
        episode_len_sum as f64 / episodes_finished as f64
    } else {
        0.0
    };

    let lines_per_step = if steps_done > 0 {
        total_lines as f64 / steps_done as f64
    } else {
        0.0
    };

    let score_per_step = if steps_done > 0 {
        total_score as f64 / steps_done as f64
    } else {
        0.0
    };

    let avg_max_h = if steps_done > 0 {
        sum_max_h / steps_done as f64
    } else {
        0.0
    };

    let avg_avg_h = if steps_done > 0 {
        sum_avg_h / steps_done as f64
    } else {
        0.0
    };

    if let Some(pb) = pb {
        if pending_pb_inc > 0 {
            pb.inc(pending_pb_inc);
        }
        pb.finish_with_message("done");
    }

    println!(
        "DONE: policy={} piece_rule={:?} steps_done={} elapsed={:.3}s steps/s={:.1} episodes_finished={} avg_ep_len={:.2} max_ep_len={} lines/step={:.3} score/step={:.2} avg_max_h={:.2} avg_h={:.2} total_score={} total_lines={} (last_ep_len={} last_game_over={})",
        args.policy,
        rule_kind,
        steps_done,
        dt,
        sps,
        episodes_finished,
        avg_ep_len,
        episode_len_max,
        lines_per_step,
        score_per_step,
        avg_max_h,
        avg_avg_h,
        total_score,
        total_lines,
        ep_len,
        game.game_over
    );
}
