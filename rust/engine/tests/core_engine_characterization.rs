// rust/engine/tests/core_engine_characterization.rs
#![forbid(unsafe_code)]

/**
 * Core engine characterization tests.
 *
 * Purpose:
 * - Lock in current observable engine behavior before deeper refactors.
 * - Catch behavioral regressions in the transition kernel, seeding rules,
 *   and piece generation semantics.
 *
 * What is tested:
 * - Deterministic trajectories for identical `(seed, piece_rule, warmup)` inputs.
 * - Warmup isolation: warmup must not alter initial `active/next` piece draws.
 * - Action-id encoding/decoding roundtrip and invalid out-of-range no-op behavior.
 * - Simulation parity: `simulate_action_id_active` must match `step_action_id`
 *   for valid actions on grid and scalar counters.
 * - Piece-rule invariants:
 *   - `Bag7`: each bag contains every kind exactly once.
 *   - `Uniform`: emitted kind ids are always valid.
 * - Game-over latch semantics when spawn rows are occupied.
 *
 * How the tests work:
 * - They primarily compare full-state snapshots (`grid`, piece kinds, counters,
 *   `game_over`, `last_lock_features`) rather than isolated fields.
 * - They use deterministic fixture seeds and bounded rollout loops to keep tests
 *   fast while still exercising realistic transitions.
 * - They assert public API contracts (not private implementation details), so
 *   internals can be refactored without rewriting tests as long as behavior holds.
 */
use tetris_engine::{
    ACTION_DIM, Game, HoleCount, PieceRuleKind, RowCountDist, W, WarmupSpec, decode_action_id,
    encode_action_id,
};

fn score_delta_for_clear(cleared_lines: u32, level_before: u32) -> u64 {
    let base = match cleared_lines {
        1 => 100u64,
        2 => 300u64,
        3 => 500u64,
        4 => 800u64,
        _ => 0u64,
    };
    base.saturating_mul(u64::from(level_before) + 1)
}

fn lock_features_tuple(g: &Game) -> Option<(u32, u32, u32, u32)> {
    g.last_lock_features()
        .map(|f| (f.max_h, f.agg_h, f.holes, f.bump))
}

fn assert_state_equal(lhs: &Game, rhs: &Game) {
    assert_eq!(lhs.grid, rhs.grid);
    assert_eq!(lhs.active.idx(), rhs.active.idx());
    assert_eq!(lhs.next.idx(), rhs.next.idx());
    assert_eq!(lhs.score, rhs.score);
    assert_eq!(lhs.lines_cleared, rhs.lines_cleared);
    assert_eq!(lhs.level, rhs.level);
    assert_eq!(lhs.steps, rhs.steps);
    assert_eq!(lhs.game_over, rhs.game_over);
    assert_eq!(lock_features_tuple(lhs), lock_features_tuple(rhs));
}

#[test]
fn deterministic_episode_for_same_seed_rule_and_warmup() {
    let warmup = WarmupSpec {
        rows: RowCountDist::Uniform { min: 2, max: 6 },
        holes: HoleCount::Uniform { min: 1, max: 3 },
        prob: 1.0,
        seed_salt: 0x1234_ABCD_4321_DCBA,
    };

    let mut g1 = Game::new_with_rule_and_warmup_spec(20260228, PieceRuleKind::Bag7, warmup);
    let mut g2 = Game::new_with_rule_and_warmup_spec(20260228, PieceRuleKind::Bag7, warmup);

    for step in 0usize..80 {
        assert_state_equal(&g1, &g2);

        let valid1 = g1.valid_action_ids();
        let valid2 = g2.valid_action_ids();
        assert_eq!(valid1, valid2);
        if valid1.is_empty() {
            break;
        }

        let aid = valid1[step % valid1.len()];
        let r1 = g1.step_action_id(aid);
        let r2 = g2.step_action_id(aid);

        assert_eq!(r1.terminated, r2.terminated);
        assert_eq!(r1.cleared_lines, r2.cleared_lines);
        assert_eq!(r1.invalid_action, r2.invalid_action);

        if r1.terminated {
            assert_state_equal(&g1, &g2);
            break;
        }
    }
}

#[test]
fn warmup_does_not_change_initial_piece_stream() {
    let warmup = WarmupSpec {
        rows: RowCountDist::Fixed(8),
        holes: HoleCount::Fixed(2),
        prob: 1.0,
        seed_salt: 7,
    };

    for rule in [PieceRuleKind::Uniform, PieceRuleKind::Bag7] {
        let plain = Game::new_with_rule_and_warmup_spec(1337, rule, WarmupSpec::none());
        let noisy = Game::new_with_rule_and_warmup_spec(1337, rule, warmup);
        assert_eq!(plain.active.idx(), noisy.active.idx());
        assert_eq!(plain.next.idx(), noisy.next.idx());
    }
}

#[test]
fn action_id_roundtrip_and_out_of_range_is_invalid_noop() {
    let max_rots = ACTION_DIM / W;
    for rot in 0..max_rots {
        for col in 0..W {
            let aid = encode_action_id(rot, col);
            let (rr, cc) = decode_action_id(aid);
            assert_eq!(rr, rot);
            assert_eq!(cc, col);
        }
    }

    let mut g = Game::new(4242);
    let before = g.clone();

    let r = g.step_action_id(ACTION_DIM);
    assert!(!r.terminated);
    assert_eq!(r.cleared_lines, 0);
    assert!(r.invalid_action);

    assert_state_equal(&g, &before);
}

#[test]
fn simulate_active_matches_step_for_valid_actions() {
    let mut g = Game::new_with_rule(5150, PieceRuleKind::Uniform);

    for i in 0usize..40 {
        let valid = g.valid_action_ids();
        if valid.is_empty() {
            break;
        }

        let aid = valid[i % valid.len()];
        let before = g.clone();
        let sim = before.simulate_action_id_active(aid);
        assert!(!sim.invalid);

        let level_before = before.level;
        let lines_before = before.lines_cleared;
        let score_before = before.score;

        let r = g.step_action_id(aid);
        assert!(!r.invalid_action);
        assert_eq!(r.cleared_lines, sim.cleared_lines);
        assert_eq!(g.grid, sim.grid_after_clear);
        assert_eq!(
            g.score,
            score_before + score_delta_for_clear(r.cleared_lines, level_before)
        );
        assert_eq!(g.lines_cleared, lines_before + u64::from(r.cleared_lines));

        if r.terminated {
            break;
        }
        assert_eq!(g.active.idx(), before.next.idx());
    }
}

#[test]
fn bag7_emits_each_kind_exactly_once_per_bag() {
    let mut g = Game::new_with_rule(9001, PieceRuleKind::Bag7);
    let mut draws = Vec::with_capacity(28);
    draws.push(g.active.idx());
    for _ in 1..28 {
        g.spawn_next();
        draws.push(g.active.idx());
    }

    for bag in 0..4 {
        let mut seen = [false; 7];
        for idx in draws[(bag * 7)..((bag + 1) * 7)].iter() {
            let idx0 = (*idx - 1) as usize;
            assert!(!seen[idx0]);
            seen[idx0] = true;
        }
        assert!(seen.into_iter().all(|v| v));
    }
}

#[test]
fn uniform_rule_emits_only_valid_kind_ids() {
    let mut g = Game::new_with_rule(123456, PieceRuleKind::Uniform);
    for i in 0..200 {
        if i > 0 {
            g.spawn_next();
        }
        let idx = g.active.idx();
        assert!((1..=7).contains(&idx));
    }
}

#[test]
fn occupied_spawn_rows_trigger_termination_and_latch_game_over() {
    let mut g = Game::new(77);
    g.grid[0][0] = 1;

    let valid = g.valid_action_ids();
    assert!(!valid.is_empty());

    let r1 = g.step_action_id(valid[0]);
    assert!(r1.terminated);
    assert!(!r1.invalid_action);
    assert!(g.game_over);

    let steps_before = g.steps;
    let r2 = g.step_action_id(0);
    assert!(r2.terminated);
    assert_eq!(r2.cleared_lines, 0);
    assert!(!r2.invalid_action);
    assert_eq!(g.steps, steps_before);
}
