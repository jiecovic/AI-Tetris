// rust/engine/src/policy/codemy/mod.rs
#![forbid(unsafe_code)]

use ::core::marker::PhantomData;

use crate::engine::Game;

use crate::policy::base::Policy;
use crate::policy::beam::BeamConfig;

mod core;
mod empty_cache;
mod fast;
mod score;
mod unknown;

pub use fast::Codemy2FastPolicy;
pub use unknown::UniformIID;

// Keep these internal unless you actually want them public later.
use core::CodemyCore;
use unknown::UnknownModel;

/// Dynamic (runtime plies) policy.
pub struct CodemyPolicyDynamic {
    core: CodemyCore,
    plies: u8,
}

impl CodemyPolicyDynamic {
    pub fn new(plies: u8, beam: Option<BeamConfig>) -> Self {
        Self {
            core: CodemyCore::new(beam),
            plies: plies.max(1),
        }
    }
}

impl Policy for CodemyPolicyDynamic {
    fn choose_action(&mut self, g: &Game) -> Option<usize> {
        let aid0_cands = self.core.aid0_candidates_with_proxy(g);
        if aid0_cands.is_empty() {
            return None;
        }

        let mut best: Option<(usize, f64)> = None;

        for (aid0, _proxy0) in aid0_cands {
            let sim1 = g.simulate_action_id_active(aid0);
            if sim1.invalid {
                continue;
            }

            let v0 = if self.plies == 1 {
                score::score_grid(&sim1.grid_after_lock)
            } else {
                self.core.value_known_piece::<UniformIID>(
                    &sim1.grid_after_clear,
                    g.next,
                    self.plies - 1,
                    1,
                )
            };

            match best {
                None => best = Some((aid0, v0)),
                Some((_ba, bv)) if v0 > bv => best = Some((aid0, v0)),
                _ => {}
            }
        }

        best.map(|(aid, _)| aid)
    }
}

/// Static (compile-time plies + unknown model) policy.
/// This is the "Rust templates" fast-path: monomorphized for each (M, PLIES).
pub struct CodemyPolicyStatic<M: UnknownModel, const PLIES: u8> {
    core: CodemyCore,
    _m: PhantomData<M>,
}

impl<M: UnknownModel, const PLIES: u8> CodemyPolicyStatic<M, PLIES> {
    pub fn new(beam: Option<BeamConfig>) -> Self {
        debug_assert!(PLIES >= 1);
        Self {
            core: CodemyCore::new(beam),
            _m: PhantomData,
        }
    }
}

impl<M: UnknownModel, const PLIES: u8> Policy for CodemyPolicyStatic<M, PLIES> {
    fn choose_action(&mut self, g: &Game) -> Option<usize> {
        debug_assert!(PLIES >= 1);

        let aid0_cands = self.core.aid0_candidates_with_proxy(g);
        if aid0_cands.is_empty() {
            return None;
        }

        let mut best: Option<(usize, f64)> = None;

        for (aid0, _proxy0) in aid0_cands {
            let sim1 = g.simulate_action_id_active(aid0);
            if sim1.invalid {
                continue;
            }

            let v0 = if PLIES == 1 {
                score::score_grid(&sim1.grid_after_lock)
            } else {
                self.core
                    .value_known_piece::<M>(&sim1.grid_after_clear, g.next, PLIES - 1, 1)
            };

            match best {
                None => best = Some((aid0, v0)),
                Some((_ba, bv)) if v0 > bv => best = Some((aid0, v0)),
                _ => {}
            }
        }

        best.map(|(aid, _)| aid)
    }
}

// -----------------------------------------------------------------------------
// Public aliases (convenient "named" policies)
// -----------------------------------------------------------------------------

/// Backwards-friendly default name: dynamic policy.
pub type CodemyPolicy = CodemyPolicyDynamic;

/// Fast monomorphized presets (UniformIID unknown model).
pub type Codemy0 = CodemyPolicyStatic<UniformIID, 1>;
pub type Codemy1 = CodemyPolicyStatic<UniformIID, 2>;
pub type Codemy2 = CodemyPolicyStatic<UniformIID, 3>;
