# Release Checklist

## Pre-Release

- [ ] `cargo fmt --all -- --check`
- [ ] `cargo clippy --workspace --all-targets -- -D warnings`
- [ ] `cargo test --workspace`
- [ ] `cargo doc --workspace --no-deps`
- [ ] `pre-commit run --all-files`
- [ ] Bench sanity run: `cargo bench -p tetris_engine --bench engine_core_bench`

## Behavior Safety

- [ ] Core characterization tests pass.
- [ ] Invariant/property tests pass.
- [ ] Policy regression fixtures pass (codemy + heuristic/random contracts).

## Packaging / API

- [ ] Confirm intended public API surface in `rust/engine/src/lib.rs`.
- [ ] Ensure fallible config paths have typed or documented errors.
- [ ] Update docs/changelog notes for any behavior change.

## Optional Deep Checks

- [ ] Evaluate and run sanitizer/miri job (nightly-only, optional).

## Ship

- [ ] Tag version.
- [ ] Publish release notes with benchmark deltas and notable behavior changes.
