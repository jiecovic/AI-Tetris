// rust/engine/src/policy/random.rs
#![forbid(unsafe_code)]

use rand::prelude::*;

use crate::engine::Game;

use super::base::Policy;

pub struct RandomPolicy {
    rng: StdRng,
}

impl RandomPolicy {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl Policy for RandomPolicy {
    fn choose_action(&mut self, g: &Game) -> Option<usize> {
        let ids = g.valid_action_ids();
        let &aid = ids.choose(&mut self.rng)?;
        Some(aid)
    }
}
