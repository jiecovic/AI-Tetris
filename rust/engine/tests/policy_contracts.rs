// rust/engine/tests/policy_contracts.rs
#![forbid(unsafe_code)]

/**
 * Cross-policy contract tests for non-codemy policies.
 *
 * Purpose:
 * - Enforce shared behavior contracts for policy implementations:
 *   legal action selection, determinism (where applicable), and input-state purity.
 *
 * Covered policy families:
 * - `RandomPolicy` (seeded deterministic RNG path)
 * - `HeuristicPolicy` (deterministic scorer/search path)
 */
use tetris_engine::{
    Game, H, HeuristicFeature, HeuristicPolicy, Kind, PieceRuleKind, Policy, RandomPolicy, W,
};

#[derive(Clone, Debug, Eq, PartialEq)]
struct GameSnapshot {
    grid: [[u8; W]; H],
    active_idx: u8,
    next_idx: u8,
    score: u64,
    lines_cleared: u64,
    level: u32,
    steps: u64,
    game_over: bool,
    last_lock_features: Option<(u32, u32, u32, u32)>,
}

fn snapshot(g: &Game) -> GameSnapshot {
    GameSnapshot {
        grid: g.grid,
        active_idx: g.active.idx(),
        next_idx: g.next.idx(),
        score: g.score,
        lines_cleared: g.lines_cleared,
        level: g.level,
        steps: g.steps,
        game_over: g.game_over,
        last_lock_features: g
            .last_lock_features()
            .map(|f| (f.max_h, f.agg_h, f.holes, f.bump)),
    }
}

fn fixture_game() -> Game {
    let mut g = Game::new_with_rule(987654, PieceRuleKind::Uniform);

    for c in 0..W {
        if c != 3 {
            g.grid[H - 1][c] = 1;
        }
        if c != 6 {
            g.grid[H - 2][c] = 2;
        }
    }
    for c in [0usize, 2, 7, 9] {
        g.grid[H - 3][c] = 3;
    }

    g.active = Kind::L;
    g.next = Kind::T;
    g
}

fn blocked_game() -> Game {
    let mut g = fixture_game();
    for r in 0..H {
        for c in 0..W {
            g.grid[r][c] = 1;
        }
    }
    g
}

fn assert_valid_action_or_none(policy: &mut dyn Policy, g: &Game) {
    let valid = g.valid_action_ids();
    match policy.choose_action(g) {
        Some(aid) => assert!(valid.contains(&aid)),
        None => assert!(valid.is_empty()),
    }
}

#[test]
fn random_policy_returns_legal_actions() {
    let g = fixture_game();
    let mut p = RandomPolicy::new(123);
    for _ in 0..20 {
        assert_valid_action_or_none(&mut p, &g);
    }
}

#[test]
fn random_policy_is_seed_deterministic_for_fixed_state() {
    let g = fixture_game();
    let mut p1 = RandomPolicy::new(42);
    let mut p2 = RandomPolicy::new(42);
    for _ in 0..12 {
        assert_eq!(p1.choose_action(&g), p2.choose_action(&g));
    }
}

#[test]
fn random_policy_does_not_mutate_game() {
    let g = fixture_game();
    let before = snapshot(&g);
    let mut p = RandomPolicy::new(99);
    let _ = p.choose_action(&g);
    assert_eq!(before, snapshot(&g));
}

#[test]
fn random_policy_returns_none_on_blocked_board() {
    let g = blocked_game();
    let mut p = RandomPolicy::new(1);
    assert_eq!(p.choose_action(&g), None);
}

fn heuristic_policy() -> HeuristicPolicy {
    let features = vec![
        HeuristicFeature::AggHeight,
        HeuristicFeature::Holes,
        HeuristicFeature::Bumpiness,
        HeuristicFeature::CompleteLines,
    ];
    let weights = vec![-0.510066, -0.35663, -0.184483, 0.760666];
    HeuristicPolicy::new(features, weights, 2, None, false)
        .expect("heuristic policy fixture should build")
}

#[test]
fn heuristic_policy_returns_legal_actions() {
    let g = fixture_game();
    let mut p = heuristic_policy();
    for _ in 0..8 {
        assert_valid_action_or_none(&mut p, &g);
    }
}

#[test]
fn heuristic_policy_is_deterministic_for_fixed_state() {
    let g = fixture_game();
    let mut p = heuristic_policy();
    let a1 = p.choose_action(&g);
    let a2 = p.choose_action(&g);
    assert_eq!(a1, a2);
}

#[test]
fn heuristic_policy_does_not_mutate_game() {
    let g = fixture_game();
    let before = snapshot(&g);
    let mut p = heuristic_policy();
    let _ = p.choose_action(&g);
    assert_eq!(before, snapshot(&g));
}

#[test]
fn heuristic_policy_returns_none_on_blocked_board() {
    let g = blocked_game();
    let mut p = heuristic_policy();
    assert_eq!(p.choose_action(&g), None);
}
