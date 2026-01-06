// src/policy.rs
use rand::prelude::*;

use crate::game;
use crate::game::Game;

pub trait Policy {
    fn choose_action(&mut self, g: &Game) -> Option<(usize, i32)>;
}

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
    fn choose_action(&mut self, g: &Game) -> Option<(usize, i32)> {
        let ids = g.legal_action_ids();
        let &aid = ids.choose(&mut self.rng)?;
        let (rot, col_u) = game::decode_action_id(aid);
        Some((rot, col_u as i32)) // col is bbox-left col
    }
}
