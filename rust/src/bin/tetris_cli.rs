// src/bin/tetris_cli.rs
#![forbid(unsafe_code)]

use clap::Parser;

use tetris_cli::engine::PieceRuleKind;
use tetris_cli::policy::{CodemyPolicy, Lookahead, Policy, RandomPolicy};
use tetris_cli::rollout::{NoopSink, RolloutSink, Runner, RunnerConfig, TableSink};

#[derive(Parser, Debug)]
#[command(name = "tetris_cli")]
struct Args {
    // ---------------- rollout sizing ----------------
    /// Total placements to execute across episodes.
    #[arg(long, default_value_t = 200)]
    steps: u64,

    /// Base RNG seed (episodes use base_seed + episode_id). If omitted, a fixed default is used.
    #[arg(long)]
    seed: Option<u64>,

    /// Policy: random | codemy0 | codemy1 | codemy2
    #[arg(long, default_value = "random")]
    policy: String,

    /// Piece rule: uniform | bag7
    #[arg(long, default_value = "uniform")]
    piece_rule: String,

    // ---------------- visualization ----------------
    /// Render board as ASCII every step; value is sleep in ms (e.g. 30). Omit to disable rendering.
    /// Examples:
    ///   --render 0    (render as fast as possible)
    ///   --render 30   (sleep 30ms between frames)
    #[arg(long, value_name = "ms")]
    render: Option<u64>,

    // ---------------- output / reporting ----------------
    /// Verbosity: 0=silent (final summary only), 1=progress bar, 2=progress bar + periodic table.
    #[arg(long, default_value_t = 1)]
    verbosity: u8,

    /// Print a table row every N steps (only used with --verbosity 2).
    #[arg(long, default_value_t = 2000)]
    report_every: u64,
}

fn main() {
    let args = Args::parse();

    // Episode seeds are derived from this base seed.
    let base_seed = args.seed.unwrap_or(12345);
    let rule_kind = PieceRuleKind::from_cli(&args.piece_rule);

    // Policy instance (boxed so the CLI can switch implementations at runtime).
    let mut policy: Box<dyn Policy> = match args.policy.as_str() {
        "codemy0" => Box::new(CodemyPolicy::new(Lookahead::D0)),
        "codemy1" => Box::new(CodemyPolicy::new(Lookahead::D1)),
        "codemy2" => Box::new(CodemyPolicy::new(Lookahead::D2Uniform)),
        _ => Box::new(RandomPolicy::new(base_seed.wrapping_add(999))),
    };

    // Rollout configuration (data only; no logic).
    let cfg = RunnerConfig {
        steps: args.steps,
        base_seed,
        rule_kind,

        render_ms: args.render,

        verbosity: args.verbosity,
        report_every: args.report_every,

        policy_name: args.policy.clone(),
    };

    // Reporting sink:
    // - verbosity 2 => periodic table (unless report_every == 0)
    // - otherwise   => no-op
    let sink: Box<dyn RolloutSink> = if cfg.verbosity >= 2 && cfg.report_every > 0 {
        // Header cadence is a formatting detail; cadence in *steps* is handled by Runner.
        Box::new(TableSink::new(20))
    } else {
        Box::new(NoopSink::default())
    };

    let mut runner = Runner::new(cfg, sink);
    let report = runner.run(&mut *policy);

    // Final one-line summary (useful for logs / grep).
    println!(
        "DONE: policy={} piece_rule={:?} steps_done={} elapsed={:.3}s steps/s={:.1} episodes_finished={} avg_ep_len={:.2} max_ep_len={} lines/step={:.3} score/step={:.2} avg_max_h={:.2} avg_h={:.2} total_score={} total_lines={} (last_ep_len={} last_game_over={})",
        report.policy,
        report.piece_rule,
        report.steps_done,
        report.elapsed_s,
        report.steps_per_s,
        report.episodes_finished,
        report.avg_ep_len,
        report.max_ep_len,
        report.lines_per_step,
        report.score_per_step,
        report.avg_max_h,
        report.avg_h,
        report.total_score,
        report.total_lines,
        report.last_ep_len,
        report.last_game_over,
    );
}
