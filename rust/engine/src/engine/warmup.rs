// rust/engine/src/engine/warmup.rs
#![forbid(unsafe_code)]

/*
Warmup noise (initial grid perturbation)

Goal
----
Provide a well-documented, deterministic way to start episodes from a distribution
over "messy" boards, without affecting the piece stream.

Key principle
-------------
Piece stream semantics must remain identical regardless of warmup mode.
Therefore:
- Game draws `active` and `next` first (piece RNG consumption is fixed).
- Warmup uses a separate RNG stream derived from the same episode seed + `seed_salt`.

Distributions
-------------
RowCountDist:
- Fixed(n): exact number of warmup rows.
- Uniform{min,max}: bounded variation (coverage/fairness).
- Poisson{lambda,cap}: rows ~ Poisson(lambda), then clamped to cap (mostly smaller values).
- BasePlusPoisson{base,lambda,cap}: rows = base + Poisson(lambda), then clamped (hard baseline + jitter).

HoleCount:
- Fixed(k) or Uniform{min,max}, clamped to [1, W-1].

Spawn safety
------------
We reserve `DEFAULT_SPAWN_BUFFER` top rows to avoid blocking spawn. The effective maximum
warmup rows is `H - DEFAULT_SPAWN_BUFFER`.

Notes
-----
- Spawn buffer MUST be >= HIDDEN_ROWS to guarantee spawn rows remain empty.
- This module enforces that invariant by clamping, but the value itself is not configurable.
*/

use rand::rngs::StdRng;
use rand::{Rng, RngCore, SeedableRng};

use crate::engine::constants::{DEFAULT_SPAWN_BUFFER, H, HIDDEN_ROWS, W};
use crate::engine::grid::apply_warmup_garbage;

#[derive(Clone, Copy, Debug)]
pub enum RowCountDist {
    Fixed(u8),
    /// Inclusive bounds.
    Uniform { min: u8, max: u8 },
    /// Sample rows ~ Poisson(lambda), then clamp to cap.
    /// NOTE: This is a *skewed* distribution (more mass below lambda).
    Poisson { lambda: f64, cap: u8 },
    /// rows = base + Poisson(lambda), then clamp to cap.
    /// Good for "mostly hard starts, sometimes harder".
    BasePlusPoisson { base: u8, lambda: f64, cap: u8 },
}

#[derive(Clone, Copy, Debug)]
pub enum HoleCount {
    Fixed(u8),
    /// Inclusive bounds.
    Uniform { min: u8, max: u8 },
}

#[derive(Clone, Copy, Debug)]
pub struct WarmupSpec {
    pub rows: RowCountDist,
    pub holes: HoleCount,
    /// Probability in [0,1] to apply warmup on each reset.
    pub prob: f64,
    /// Salt mixed into the episode seed to create an independent warmup RNG stream.
    pub seed_salt: u64,
}

impl WarmupSpec {
    pub const DEFAULT_SEED_SALT: u64 = 0xA5A5_A5A5_5A5A_5A5A;

    pub fn none() -> Self {
        Self {
            rows: RowCountDist::Fixed(0),
            holes: HoleCount::Fixed(1),
            prob: 1.0,
            seed_salt: Self::DEFAULT_SEED_SALT,
        }
    }
}

impl Default for WarmupSpec {
    fn default() -> Self {
        Self::none()
    }
}

/// Apply warmup to the provided grid.
/// Deterministic w.r.t. (seed, spec), but does not consume piece RNG.
pub fn apply_warmup(grid: &mut [[u8; W]; H], episode_seed: u64, spec: &WarmupSpec) {
    let max_rows = max_warmup_rows();

    // Derive an independent RNG stream for warmup.
    let mut rng = StdRng::seed_from_u64(episode_seed ^ spec.seed_salt);

    let p = spec.prob.clamp(0.0, 1.0);
    if p <= 0.0 {
        return;
    }
    if p < 1.0 {
        let u: f64 = rng.r#gen();
        if u >= p {
            return;
        }
    }

    let rows = sample_rows(&mut rng, spec.rows).min(max_rows);
    if rows == 0 {
        return;
    }

    // We want holes to be independently configurable but still deterministic.
    let holes_raw = sample_holes(&mut rng, spec.holes);
    let holes = clamp_holes(holes_raw);

    // Use a fresh deterministic seed for the actual garbage filling (keeps behavior stable
    // even if we later change sampling order).
    let fill_seed = rng.next_u64() ^ 0xD1B5_4A32_D192_ED03;

    apply_warmup_garbage(grid, fill_seed, rows, holes);
}

fn max_warmup_rows() -> u8 {
    // Enforce invariant: DEFAULT_SPAWN_BUFFER >= HIDDEN_ROWS
    let sb = DEFAULT_SPAWN_BUFFER.max(HIDDEN_ROWS);

    let max_rows_usize = H.saturating_sub(sb).min(u8::MAX as usize);
    max_rows_usize as u8
}

fn clamp_holes(holes: u8) -> u8 {
    // Ensure at least 1 hole and at most W-1 holes.
    let max_holes = (W.saturating_sub(1)).min(u8::MAX as usize) as u8;
    holes.clamp(1, max_holes.max(1))
}

fn sample_holes(rng: &mut StdRng, dist: HoleCount) -> u8 {
    match dist {
        HoleCount::Fixed(v) => v,
        HoleCount::Uniform { min, max } => {
            if min >= max {
                min
            } else {
                rng.gen_range(min..=max)
            }
        }
    }
}

fn sample_rows(rng: &mut StdRng, dist: RowCountDist) -> u8 {
    match dist {
        RowCountDist::Fixed(v) => v,
        RowCountDist::Uniform { min, max } => {
            if min >= max {
                min
            } else {
                rng.gen_range(min..=max)
            }
        }
        RowCountDist::Poisson { lambda, cap } => {
            if lambda <= 0.0 {
                0
            } else {
                let k = poisson_knuth(rng, lambda);
                (k.min(cap as u32)) as u8
            }
        }
        RowCountDist::BasePlusPoisson { base, lambda, cap } => {
            if lambda <= 0.0 {
                base.min(cap)
            } else {
                let k = poisson_knuth(rng, lambda);
                let v = (base as u32).saturating_add(k);
                (v.min(cap as u32)) as u8
            }
        }
    }
}

/// Poisson sampler (Knuth).
/// Good for small/moderate lambdas; no extra dependency (`rand_distr`) required.
fn poisson_knuth(rng: &mut StdRng, lambda: f64) -> u32 {
    if lambda <= 0.0 {
        return 0;
    }
    let l = (-lambda).exp();
    let mut k: u32 = 0;
    let mut p: f64 = 1.0;
    loop {
        k += 1;
        p *= rng.r#gen::<f64>();
        if p <= l {
            return k - 1;
        }
    }
}
