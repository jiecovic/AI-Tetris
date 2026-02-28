// src/bin/main.rs
#![forbid(unsafe_code)]

mod rollout;

use clap::Parser;

use crate::rollout::{NoopSink, RolloutSink, Runner, RunnerConfig, TableSink};
use tetris_engine::{
    BeamConfig, Codemy0, Codemy1, Codemy2, Codemy2FastPolicy, CodemyPolicy, PieceRuleKind, Policy,
    RandomPolicy,
};

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

    /// Policy: random | codemy | codemy0 | codemy1 | codemy2 | codemy2fast
    #[arg(long, default_value = "random")]
    policy: String,

    /**
     * For --policy codemy: number of plies (1..N). Defaults to 3.
     * Examples:
     *   --policy codemy --lookahead 1   (equiv to codemy0)
     *   --policy codemy --lookahead 2   (equiv to codemy1)
     *   --policy codemy --lookahead 3   (equiv to codemy2)
     *   --policy codemy --lookahead 6   (dynamic fallback)
     */
    #[arg(long)]
    lookahead: Option<u8>,

    /// Beam width (top-N). If omitted, no pruning is applied.
    #[arg(long)]
    beam_width: Option<usize>,

    /**
     * Start pruning from this decision depth onward (0=current, 1=next, 2=deeper...).
     * Only used if --beam-width is provided.
     */
    #[arg(long, default_value_t = 0)]
    beam_from_depth: u8,

    /**
     * Tail weight for --policy codemy2fast.
     * 0.0 => behaves like codemy1 (ignoring the tail); higher values weigh the tail more.
     */
    #[arg(long, default_value_t = 0.5)]
    tail_weight: f64,

    /// Piece rule: uniform | bag7
    #[arg(long, default_value = "uniform")]
    piece_rule: String,

    // ---------------- warmup ----------------
    /// Fill the bottom N rows with garbage on episode reset (0 disables warmup).
    #[arg(long, default_value_t = 0)]
    warmup_rows: u8,

    /// Holes per warmed row (clamped in engine). Only relevant if --warmup-rows > 0.
    #[arg(long, default_value_t = 1)]
    warmup_holes: u8,

    // ---------------- visualization ----------------
    /**
     * Render board as ASCII every step; value is sleep in ms (e.g. 30). Omit to disable rendering.
     * Examples:
     *   --render 0    (render as fast as possible)
     *   --render 30   (sleep 30ms between frames)
     */
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

    let beam: Option<BeamConfig> = args
        .beam_width
        .map(|w| BeamConfig::new(args.beam_from_depth, w));

    // Policy instance (boxed so the CLI can switch implementations at runtime).
    let mut policy: Box<dyn Policy> = match args.policy.as_str() {
        // Static fast-paths (monomorphized).
        "codemy0" => Box::new(Codemy0::new(beam)),
        "codemy1" => Box::new(Codemy1::new(beam)),
        "codemy2" => Box::new(Codemy2::new(beam)),

        // codemy1 + cheap one-step tail.
        // Note: beam is intentionally ignored here.
        "codemy2fast" => Box::new(Codemy2FastPolicy::new(args.tail_weight)),

        // General codemy entrypoint: use --lookahead N (defaults to 3).
        "codemy" => {
            let n = args.lookahead.unwrap_or(3).max(1);
            match n {
                1 => Box::new(Codemy0::new(beam)),
                2 => Box::new(Codemy1::new(beam)),
                3 => Box::new(Codemy2::new(beam)),
                _ => Box::new(CodemyPolicy::new(n, beam)),
            }
        }

        _ => Box::new(RandomPolicy::new(base_seed.wrapping_add(999))),
    };

    // Rollout configuration (data only; no logic).
    let cfg = RunnerConfig {
        steps: args.steps,
        base_seed,
        rule_kind,

        warmup_rows: args.warmup_rows,
        warmup_holes: args.warmup_holes,

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
        Box::new(NoopSink)
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
