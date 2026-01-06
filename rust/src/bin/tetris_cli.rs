// src/bin/tetris_cli.rs
use clap::Parser;

use tetris_cli::engine::PieceRuleKind;
use tetris_cli::policy::{Policy, RandomPolicy, CodemyPolicy, Lookahead};
use tetris_cli::rollout::{Runner, RunnerConfig};


#[derive(Parser, Debug)]
#[command(name = "tetris_cli")]
struct Args {
    /// Render board as ASCII each step
    #[arg(long)]
    render: bool,

    /// How many placements to run TOTAL (across episodes)
    #[arg(long, default_value_t = 200)]
    steps: u64,

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

    let cfg = RunnerConfig {
        steps: args.steps,
        base_seed,
        rule_kind,
        render: args.render,
        sleep_ms: args.sleep_ms,
        perf: args.perf,
        progress: args.progress,
        stats_every: args.stats_every,
        policy_name: args.policy.clone(),
    };

    let mut runner = Runner::new(cfg);
    let report = runner.run(&mut *policy);

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
