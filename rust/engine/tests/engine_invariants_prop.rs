// rust/engine/tests/engine_invariants_prop.rs
#![forbid(unsafe_code)]

/**
 * Property/invariant tests for the core transition kernel.
 *
 * Purpose:
 * - Provide fuzz-like coverage using generated seeds and rollout lengths.
 * - Lock core invariants that must hold regardless of policy logic.
 *
 * Invariants covered:
 * - `action_mask` and `valid_action_ids` stay equivalent.
 * - Every listed valid action simulates as valid.
 * - Applying a valid action keeps counters monotonic and level consistent.
 * - `step_action_id` agrees with `simulate_action_id_active` on cleared lines
 *   and resulting cleared-grid state.
 * - On true termination from a step, spawn rows are occupied.
 */
use proptest::prelude::*;
use tetris_engine::{
    ACTION_DIM, Game, HIDDEN_ROWS, PieceRuleKind, W, decode_action_id, encode_action_id,
};

fn assert_action_set_consistent(g: &Game) {
    let mask = g.action_mask();
    let valid = g.valid_action_ids();

    let mut from_mask = Vec::new();
    for (aid, ok) in mask.iter().enumerate() {
        if *ok {
            from_mask.push(aid);
        }
    }

    assert_eq!(from_mask, valid);
    for aid in valid {
        assert!(aid < ACTION_DIM);
        assert!(mask[aid]);
    }
}

#[test]
fn action_mask_and_valid_ids_stay_consistent_over_rollout() {
    let mut g = Game::new_with_rule(20260228, PieceRuleKind::Uniform);

    for i in 0..100usize {
        if g.game_over {
            break;
        }
        assert_action_set_consistent(&g);
        let valid = g.valid_action_ids();
        if valid.is_empty() {
            break;
        }
        let aid = valid[i % valid.len()];
        let sim = g.simulate_action_id_active(aid);
        assert!(!sim.invalid);
        let r = g.step_action_id(aid);
        assert!(!r.invalid_action);
    }
}

#[test]
fn encode_decode_roundtrip_for_all_slots() {
    for aid in 0..ACTION_DIM {
        let (rot, col) = decode_action_id(aid);
        let rt = encode_action_id(rot, col);
        assert_eq!(aid, rt);
        assert!(col < W);
    }
}

proptest! {
    #[test]
    fn generated_rollout_respects_core_invariants(
        seed in any::<u64>(),
        steps in 1usize..80,
        use_bag7 in any::<bool>(),
    ) {
        let rule = if use_bag7 { PieceRuleKind::Bag7 } else { PieceRuleKind::Uniform };
        let mut g = Game::new_with_rule(seed, rule);

        for i in 0..steps {
            if g.game_over {
                break;
            }

            assert_action_set_consistent(&g);

            let valid = g.valid_action_ids();
            if valid.is_empty() {
                break;
            }

            let idx = ((seed as usize).wrapping_add(i * 31)) % valid.len();
            let aid = valid[idx];

            let sim = g.simulate_action_id_active(aid);
            prop_assert!(!sim.invalid);
            prop_assert!(g.simulate_action_id_active_lock_only(aid).is_some());

            let before = g.clone();
            let r = g.step_action_id(aid);

            prop_assert!(!r.invalid_action);
            prop_assert_eq!(r.cleared_lines, sim.cleared_lines);
            prop_assert_eq!(g.grid, sim.grid_after_clear);
            prop_assert!(g.score >= before.score);
            prop_assert!(g.lines_cleared >= before.lines_cleared);
            prop_assert_eq!(g.level, (g.lines_cleared / 10) as u32);

            if r.terminated {
                prop_assert!(g.game_over);
                let spawn_occupied = g.grid[..HIDDEN_ROWS]
                    .iter()
                    .any(|row| row.iter().any(|&v| v != 0));
                prop_assert!(spawn_occupied);
                break;
            }

            prop_assert_eq!(g.active.idx(), before.next.idx());
        }
    }
}
