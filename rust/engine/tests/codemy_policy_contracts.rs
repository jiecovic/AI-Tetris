// rust/engine/tests/codemy_policy_contracts.rs
#![forbid(unsafe_code)]

/**
 * Codemy policy contract and regression tests.
 *
 * Purpose:
 * - Define stable behavioral contracts for codemy family policies independent
 *   of internal search/caching implementation details.
 * - Provide a compact regression harness so refactors can be validated quickly.
 *
 * What is tested:
 * - Contract: when legal actions exist, each policy returns a legal action id.
 * - Contract: when no legal actions exist, each policy returns `None`.
 * - Determinism: repeated `choose_action` calls on the same immutable state
 *   produce the same result.
 * - Purity: `choose_action` does not mutate the input `Game` state.
 * - Regression fixture: a deterministic handcrafted board + fixed active/next
 *   piece pair yields a fixed action tuple across codemy variants.
 *
 * How the tests work:
 * - A shared fixture builder creates:
 *   - one non-trivial board with multiple decision options, and
 *   - one fully blocked board with zero legal actions.
 * - A `with_each_policy` helper runs the same assertions over all target
 *   codemy variants to keep coverage broad and consistent.
 * - The regression test stores expected actions as explicit constants so
 *   behavior deltas during refactors are surfaced immediately.
 */
use std::collections::HashSet;

use tetris_engine::{
    Codemy0, Codemy1, Codemy2, Codemy2FastPolicy, CodemyPolicy, Game, H, Kind, PieceRuleKind,
    Policy, W,
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
    let last_lock_features = g
        .last_lock_features()
        .map(|f| (f.max_h, f.agg_h, f.holes, f.bump));
    GameSnapshot {
        grid: g.grid,
        active_idx: g.active.idx(),
        next_idx: g.next.idx(),
        score: g.score,
        lines_cleared: g.lines_cleared,
        level: g.level,
        steps: g.steps,
        game_over: g.game_over,
        last_lock_features,
    }
}

fn policy_fixture_game() -> Game {
    let mut g = Game::new_with_rule(31415926, PieceRuleKind::Uniform);

    // Create a deterministic, non-trivial board profile in visible rows.
    for c in 0..W {
        if c != 4 {
            g.grid[H - 1][c] = 1;
        }
        if c != 2 && c != 7 {
            g.grid[H - 2][c] = 2;
        }
    }
    for c in [0usize, 1, 5, 8] {
        g.grid[H - 3][c] = 3;
    }
    for c in [2usize, 6, 9] {
        g.grid[H - 4][c] = 4;
    }

    g.active = Kind::T;
    g.next = Kind::I;
    g
}

fn blocked_game_with_no_valid_actions() -> Game {
    let mut g = policy_fixture_game();
    for r in 0..H {
        for c in 0..W {
            g.grid[r][c] = 1;
        }
    }
    g
}

fn policy_fixture_game_b() -> Game {
    let mut g = Game::new_with_rule(27182818, PieceRuleKind::Uniform);

    for c in 0..W {
        if c != 1 && c != 8 {
            g.grid[H - 1][c] = 1;
        }
        if c != 4 {
            g.grid[H - 2][c] = 2;
        }
    }
    for c in [0usize, 3, 5, 9] {
        g.grid[H - 3][c] = 3;
    }
    for c in [2usize, 6] {
        g.grid[H - 4][c] = 4;
    }

    g.active = Kind::S;
    g.next = Kind::J;
    g
}

fn with_each_policy(mut f: impl FnMut(&str, &mut dyn Policy)) {
    let mut p0 = Codemy0::new(None);
    f("codemy0", &mut p0);

    let mut p1 = Codemy1::new(None);
    f("codemy1", &mut p1);

    let mut p2 = Codemy2::new(None);
    f("codemy2", &mut p2);

    let mut p2_fast = Codemy2FastPolicy::new(0.35);
    f("codemy2fast", &mut p2_fast);

    let mut p_dynamic = CodemyPolicy::new(3, None);
    f("codemy_dynamic_p3", &mut p_dynamic);
}

type RegressionActions = (
    Option<usize>,
    Option<usize>,
    Option<usize>,
    Option<usize>,
    Option<usize>,
);

#[test]
fn codemy_policies_return_valid_action_when_available() {
    let g = policy_fixture_game();
    let valid: HashSet<usize> = g.valid_action_ids().into_iter().collect();
    assert!(!valid.is_empty());

    with_each_policy(|name, policy| {
        let aid = policy
            .choose_action(&g)
            .unwrap_or_else(|| panic!("{name} returned None despite valid actions"));
        assert!(
            valid.contains(&aid),
            "{name} returned invalid action_id={aid}"
        );
    });
}

#[test]
fn codemy_policies_return_none_when_no_actions_exist() {
    let g = blocked_game_with_no_valid_actions();
    assert!(g.valid_action_ids().is_empty());

    with_each_policy(|name, policy| {
        let aid = policy.choose_action(&g);
        assert!(aid.is_none(), "{name} should return None on blocked board");
    });
}

#[test]
fn codemy_policies_are_deterministic_for_fixed_state() {
    let g = policy_fixture_game();

    with_each_policy(|name, policy| {
        let a1 = policy.choose_action(&g);
        let a2 = policy.choose_action(&g);
        assert_eq!(
            a1, a2,
            "{name} produced different actions across repeated calls"
        );
    });
}

#[test]
fn codemy_policies_do_not_mutate_game_state() {
    let g = policy_fixture_game();
    let before = snapshot(&g);

    with_each_policy(|name, policy| {
        let _ = policy.choose_action(&g);
        let after = snapshot(&g);
        assert_eq!(before, after, "{name} mutated input game state");
    });
}

#[test]
fn codemy_regression_fixture_actions_are_stable() {
    let g = policy_fixture_game();

    const EXPECTED: RegressionActions = (Some(17), Some(18), Some(18), Some(18), Some(18));

    let mut p0 = Codemy0::new(None);
    let mut p1 = Codemy1::new(None);
    let mut p2 = Codemy2::new(None);
    let mut p2_fast = Codemy2FastPolicy::new(0.35);
    let mut p_dynamic = CodemyPolicy::new(3, None);

    let actual = (
        p0.choose_action(&g),
        p1.choose_action(&g),
        p2.choose_action(&g),
        p2_fast.choose_action(&g),
        p_dynamic.choose_action(&g),
    );
    assert_eq!(actual, EXPECTED);
}

#[test]
fn codemy_regression_fixture_b_actions_are_stable() {
    let g = policy_fixture_game_b();

    const EXPECTED: RegressionActions = (Some(10), Some(10), Some(12), Some(10), Some(12));

    let mut p0 = Codemy0::new(None);
    let mut p1 = Codemy1::new(None);
    let mut p2 = Codemy2::new(None);
    let mut p2_fast = Codemy2FastPolicy::new(0.35);
    let mut p_dynamic = CodemyPolicy::new(3, None);

    let actual = (
        p0.choose_action(&g),
        p1.choose_action(&g),
        p2.choose_action(&g),
        p2_fast.choose_action(&g),
        p_dynamic.choose_action(&g),
    );
    assert_eq!(actual, EXPECTED);
}
