# Rust Engine Architecture

This document is the primary structure map for the Rust part of the repo.

## Workspace Layout

The Rust workspace is defined in `rust/Cargo.toml` and has three crates:

- `rust/engine` (`tetris_engine`): core game engine + policy logic.
- `rust/py` (`tetris_rl_engine`): PyO3 bridge exposing engine APIs to Python.
- `rust/tetris_cli` (`tetris_cli`): Rust-native CLI binary using `tetris_engine`.

## Crate Boundaries

### `tetris_engine` (core)

`rust/engine/src/lib.rs` is the public API boundary. It keeps internal modules private and re-exports only curated stable items.

High-level modules:

- `engine/`: game state, transition kernel, geometry, grid ops, piece stream, warmup, feature extraction.
- `policy/`: policy trait and implementations (`random`, `heuristic`, `codemy`, beam helpers).

### `tetris_rl_engine` (Python bridge)

Depends on `tetris_engine` and exposes selected functions/types to Python through PyO3 and NumPy.

Rule of thumb:

- game rules, invariants, and core algorithms belong in `tetris_engine`.
- Python-specific conversion and API shape belong in `rust/py`.

### `tetris_cli` (Rust binary)

Thin consumer of `tetris_engine` for local Rust-side execution and debugging.

## `engine/` Module Responsibilities

- `constants.rs`: board/action dimensions and action-id encoding (`encode_action_id`/`decode_action_id`).
- `pieces.rs`: tetromino definitions and rotation data.
- `geometry.rs`: bounding-box and placement math.
- `grid.rs`: pure grid operations (fit checks, lock, clear lines, warmup garbage helpers, metrics).
- `piece_rule.rs`: piece stream RNG (`Uniform` and `Bag7`).
- `warmup.rs`: deterministic warmup distribution sampling and application.
- `features.rs`: grid feature extraction (`GridFeatures`, `GridDelta`, `StepFeatures`).
- `game.rs`: stateful game object (`Game`), pure simulation helpers, and mutating `step_action_id`.

## `policy/` Module Responsibilities

- `base.rs`: `Policy` trait (`choose_action(&Game) -> Option<usize>`).
- `random.rs`: random valid-action policy baseline.
- `heuristic.rs`: weighted-feature heuristic policy + feature parser/validation.
- `beam.rs`: deterministic top-N pruning utilities and `BeamConfig`.
- `codemy/`: lookahead family (dynamic/static variants, fast path, scoring/search internals).

## Core Data Model

Main runtime state lives in `Game`:

- `grid: [[u8; W]; H]`
- piece stream state (`active`, `next`, internal `PieceRule`)
- score/lines/level/steps/game_over counters
- `last_lock_features` cache used by reward/analysis paths

Transition outputs:

- `SimPlacement`: pure simulation result (`grid_after_lock`, `grid_after_clear`, `cleared_lines`, `invalid`).
- `StepResult`: mutating-step result (`terminated`, `cleared_lines`, `invalid_action`).

## Step Lifecycle

`Game::step_action_id` is the canonical mutating transition:

1. Validate action id and resolve placement.
2. Invalid action => no-op (`invalid_action=true`, not terminated).
3. Valid action => lock piece, compute lock features, clear lines.
4. Update score/line/level/step counters.
5. If spawn rows are occupied post-clear => `game_over=true`, `terminated=true`.
6. Otherwise spawn next piece and continue.

Pure, non-mutating alternatives:

- `simulate_action_id(...)`
- `apply_action_id_to_grid(...)`
- `*_lock_only(...)` fast paths

## Important Invariants

- `ACTION_DIM = MAX_ROTS * W`; not every rotation slot is valid for every piece.
- Action mask marks only legal slots (`rot < kind.num_rots()` and valid spawn fit).
- Invalid actions are explicitly represented as no-op transitions.
- Spawn/game-over logic is based on hidden rows (`HIDDEN_ROWS`).
- Warmup must not affect piece stream RNG consumption.
  - `active` and `next` are drawn before warmup.
  - warmup uses a separate deterministic RNG stream (`seed ^ seed_salt`).
- Unsafe Rust is forbidden crate-wide (`#![forbid(unsafe_code)]`).

## Public API Contract

External users should import only from crate root (`tetris_engine::...`), not internal module paths.

Stable surface is curated by `rust/engine/src/lib.rs` and includes:

- engine core types (`Game`, `SimPlacement`, `StepResult`)
- constants/action helpers
- warmup/piece rule configs
- feature extractors
- policy trait/config/entrypoints

## Testing and Quality Gates

Current engine test suites:

- `rust/engine/tests/core_engine_characterization.rs`
- `rust/engine/tests/engine_invariants_prop.rs`
- `rust/engine/tests/policy_contracts.rs`
- `rust/engine/tests/codemy_policy_contracts.rs`

Recommended commands:

- `cargo test -p tetris_engine`
- `cargo clippy -p tetris_engine -- -D warnings`
- `cargo fmt --all -- --check`

## Extension Guidance

When adding new behavior:

1. Add logic in internal module first.
2. Expose only intentional API via `engine/mod.rs` or `policy/mod.rs`.
3. Re-export from `lib.rs` only if part of stable external surface.
4. Add characterization/invariant tests before or with behavior changes.
5. Keep Python binding crate as adapter only; avoid duplicating engine logic there.
