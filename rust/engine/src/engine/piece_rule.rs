// src/piece_rule.rs
use rand::prelude::*;

use crate::engine::pieces::Kind;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PieceRuleKind {
    Uniform,
    Bag7,
}

impl PieceRuleKind {
    pub fn from_cli(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "bag7" | "7bag" | "bag" => PieceRuleKind::Bag7,
            _ => PieceRuleKind::Uniform,
        }
    }
}

#[derive(Clone)]
pub(crate) struct PieceRule {
    kind: PieceRuleKind,

    // RNG lives here (spawn stream responsibility)
    rng: StdRng,

    // 7-bag state (only used if kind == Bag7)
    bag: [Kind; 7],
    bag_idx: usize,
}

impl PieceRule {
    pub(crate) fn new(seed: u64, kind: PieceRuleKind) -> Self {
        Self {
            kind,
            rng: StdRng::seed_from_u64(seed),
            bag: [
                Kind::I,
                Kind::O,
                Kind::T,
                Kind::S,
                Kind::Z,
                Kind::J,
                Kind::L,
            ],
            bag_idx: 7, // force refill on first Bag7 draw
        }
    }

    pub(crate) fn kind(&self) -> PieceRuleKind {
        self.kind
    }

    fn refill_bag7(&mut self) {
        self.bag = [
            Kind::I,
            Kind::O,
            Kind::T,
            Kind::S,
            Kind::Z,
            Kind::J,
            Kind::L,
        ];
        self.bag.shuffle(&mut self.rng);
        self.bag_idx = 0;
    }

    pub(crate) fn draw(&mut self) -> Kind {
        match self.kind {
            PieceRuleKind::Uniform => {
                let all = Kind::all();
                let idx = self.rng.gen_range(0..all.len());
                all[idx]
            }
            PieceRuleKind::Bag7 => {
                if self.bag_idx >= 7 {
                    self.refill_bag7();
                }
                let k = self.bag[self.bag_idx];
                self.bag_idx += 1;
                k
            }
        }
    }
}
