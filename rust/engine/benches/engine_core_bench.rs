// rust/engine/benches/engine_core_bench.rs
#![forbid(unsafe_code)]

/**
 * Core engine micro-benchmarks.
 *
 * Focus:
 * - Transition kernel (`step_action_id`)
 * - Pure simulation (`simulate_action_id_active`)
 * - Policy decision latency on fixed board states
 */
use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
use tetris_engine::{
    Codemy2FastPolicy, Game, HeuristicFeature, HeuristicPolicy, PieceRuleKind, Policy,
};

fn build_nontrivial_game(seed: u64) -> Game {
    let mut g = Game::new_with_rule(seed, PieceRuleKind::Uniform);
    for i in 0usize..32 {
        if g.game_over {
            break;
        }
        let ids = g.valid_action_ids();
        if ids.is_empty() {
            break;
        }
        let aid = ids[(i * 11) % ids.len()];
        let _ = g.step_action_id(aid);
    }
    g
}

fn bench_step_action_id(c: &mut Criterion) {
    c.bench_function("engine.step_action_id.valid_path", |b| {
        b.iter_batched(
            || Game::new_with_rule(20260228, PieceRuleKind::Uniform),
            |mut g| {
                for i in 0usize..256 {
                    if g.game_over {
                        break;
                    }
                    let ids = g.valid_action_ids();
                    if ids.is_empty() {
                        break;
                    }
                    let aid = ids[i % ids.len()];
                    black_box(g.step_action_id(aid));
                }
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_simulate_action_id_active(c: &mut Criterion) {
    c.bench_function("engine.simulate_action_id_active", |b| {
        b.iter_batched(
            || build_nontrivial_game(777),
            |g| {
                let ids = g.valid_action_ids();
                if !ids.is_empty() {
                    let aid = ids[ids.len() / 2];
                    black_box(g.simulate_action_id_active(aid));
                }
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_policy_choose_action(c: &mut Criterion) {
    c.bench_function("policy.codemy2fast.choose_action", |b| {
        b.iter_batched(
            || (build_nontrivial_game(1234), Codemy2FastPolicy::new(0.35)),
            |(g, mut p)| {
                black_box(p.choose_action(&g));
            },
            BatchSize::SmallInput,
        );
    });

    c.bench_function("policy.heuristic.choose_action", |b| {
        b.iter_batched(
            || {
                let g = build_nontrivial_game(5678);
                let p = HeuristicPolicy::try_new(
                    vec![
                        HeuristicFeature::AggHeight,
                        HeuristicFeature::Holes,
                        HeuristicFeature::Bumpiness,
                        HeuristicFeature::CompleteLines,
                    ],
                    vec![-0.510066, -0.35663, -0.184483, 0.760666],
                    2,
                    None,
                    false,
                )
                .expect("valid heuristic benchmark config");
                (g, p)
            },
            |(g, mut p)| {
                black_box(p.choose_action(&g));
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(
    engine_core_benches,
    bench_step_action_id,
    bench_simulate_action_id_active,
    bench_policy_choose_action
);
criterion_main!(engine_core_benches);
