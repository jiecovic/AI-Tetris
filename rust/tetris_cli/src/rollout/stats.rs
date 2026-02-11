// src/rollout/stats.rs
#![forbid(unsafe_code)]

use std::time::Instant;

use tetris_engine::engine::{compute_grid_features, GridDelta, GridFeatures, PieceRuleKind};

#[derive(Clone, Debug)]
pub struct RolloutStats {
    pub episodes_finished: u64,
    pub ep_len: u64,
    pub episode_len_sum: u64,
    pub episode_len_max: u64,

    pub steps_done: u64,

    // height aggregates
    pub sum_max_h: f64,
    pub sum_avg_h: f64,
    pub max_h_worst: u32,

    // grid feature aggregates (absolute, across ALL steps)
    pub sum_agg_h: f64,
    pub sum_holes: f64,
    pub sum_bump: f64,

    // delta aggregates (difference between consecutive steps)
    pub sum_d_max_h: f64,
    pub sum_d_agg_h: f64,
    pub sum_d_holes: f64,
    pub sum_d_bump: f64,

    // internal
    prev_grid_features: Option<GridFeatures>,
    t0: Instant,
}

impl RolloutStats {
    pub fn new() -> Self {
        Self {
            episodes_finished: 0,
            ep_len: 0,
            episode_len_sum: 0,
            episode_len_max: 0,
            steps_done: 0,
            sum_max_h: 0.0,
            sum_avg_h: 0.0,
            max_h_worst: 0,
            sum_agg_h: 0.0,
            sum_holes: 0.0,
            sum_bump: 0.0,
            sum_d_max_h: 0.0,
            sum_d_agg_h: 0.0,
            sum_d_holes: 0.0,
            sum_d_bump: 0.0,
            prev_grid_features: None,
            t0: Instant::now(),
        }
    }

    /// Call once per placement.
    ///
    /// `max_h` and `avg_h` are from your cheap height metrics.
    /// We ALSO compute grid features here (agg/holes/bump) to maintain running averages.
    pub fn on_step(
        &mut self,
        grid: &[[u8;tetris_engine::engine::W]; tetris_engine::engine::H],
        max_h: u32,
        avg_h: f32,
    ) {
        self.steps_done += 1;
        self.ep_len += 1;

        self.sum_max_h += max_h as f64;
        self.sum_avg_h += avg_h as f64;
        self.max_h_worst = self.max_h_worst.max(max_h);

        // heavier grid features
        let cur = compute_grid_features(grid);
        self.sum_agg_h += cur.agg_h as f64;
        self.sum_holes += cur.holes as f64;
        self.sum_bump += cur.bump as f64;

        // deltas vs previous step (within episode)
        let d: GridDelta = match self.prev_grid_features {
            None => GridDelta::default(),
            Some(p) => GridDelta {
                d_max_h: cur.max_h as i32 - p.max_h as i32,
                d_agg_h: cur.agg_h as i32 - p.agg_h as i32,
                d_holes: cur.holes as i32 - p.holes as i32,
                d_bump: cur.bump as i32 - p.bump as i32,
            },
        };

        self.sum_d_max_h += d.d_max_h as f64;
        self.sum_d_agg_h += d.d_agg_h as f64;
        self.sum_d_holes += d.d_holes as f64;
        self.sum_d_bump += d.d_bump as f64;

        self.prev_grid_features = Some(cur);
    }

    /// Call when an episode terminates (game_over), before resetting the game.
    pub fn on_episode_end(&mut self) {
        self.episodes_finished += 1;
        self.episode_len_sum += self.ep_len;
        self.episode_len_max = self.episode_len_max.max(self.ep_len);

        self.ep_len = 0;
        self.prev_grid_features = None; // don't carry deltas across episodes
    }

    pub fn elapsed_secs(&self) -> f64 {
        self.t0.elapsed().as_secs_f64()
    }

    pub fn steps_per_sec(&self) -> f64 {
        let dt = self.elapsed_secs();
        if dt > 0.0 {
            self.steps_done as f64 / dt
        } else {
            0.0
        }
    }

    pub fn avg_ep_len(&self) -> f64 {
        if self.episodes_finished > 0 {
            self.episode_len_sum as f64 / self.episodes_finished as f64
        } else {
            0.0
        }
    }

    pub fn avg_max_h(&self) -> f64 {
        if self.steps_done > 0 {
            self.sum_max_h / self.steps_done as f64
        } else {
            0.0
        }
    }

    pub fn avg_avg_h(&self) -> f64 {
        if self.steps_done > 0 {
            self.sum_avg_h / self.steps_done as f64
        } else {
            0.0
        }
    }

    pub fn avg_agg_h(&self) -> f64 {
        if self.steps_done > 0 {
            self.sum_agg_h / self.steps_done as f64
        } else {
            0.0
        }
    }

    pub fn avg_holes(&self) -> f64 {
        if self.steps_done > 0 {
            self.sum_holes / self.steps_done as f64
        } else {
            0.0
        }
    }

    pub fn avg_bump(&self) -> f64 {
        if self.steps_done > 0 {
            self.sum_bump / self.steps_done as f64
        } else {
            0.0
        }
    }

    pub fn avg_d_max_h(&self) -> f64 {
        if self.steps_done > 0 {
            self.sum_d_max_h / self.steps_done as f64
        } else {
            0.0
        }
    }
    pub fn avg_d_agg_h(&self) -> f64 {
        if self.steps_done > 0 {
            self.sum_d_agg_h / self.steps_done as f64
        } else {
            0.0
        }
    }
    pub fn avg_d_holes(&self) -> f64 {
        if self.steps_done > 0 {
            self.sum_d_holes / self.steps_done as f64
        } else {
            0.0
        }
    }
    pub fn avg_d_bump(&self) -> f64 {
        if self.steps_done > 0 {
            self.sum_d_bump / self.steps_done as f64
        } else {
            0.0
        }
    }

    pub fn lines_per_step(&self, live_total_lines: u64) -> f64 {
        if self.steps_done > 0 {
            live_total_lines as f64 / self.steps_done as f64
        } else {
            0.0
        }
    }

    pub fn score_per_step(&self, live_total_score: u64) -> f64 {
        if self.steps_done > 0 {
            live_total_score as f64 / self.steps_done as f64
        } else {
            0.0
        }
    }

    pub fn live_msg(&self, rule_kind: PieceRuleKind, lps: f64, spscore: f64) -> String {
        format!(
            "rule={:?} sps={:.1} eps={} avg_ep={:.1} max_ep={} l/step={:.3} score/step={:.2} maxH={} avgAgg={:.1} avgHol={:.2} avgBum={:.2}",
            rule_kind,
            self.steps_per_sec(),
            self.episodes_finished,
            self.avg_ep_len(),
            self.episode_len_max,
            lps,
            spscore,
            self.max_h_worst,
            self.avg_agg_h(),
            self.avg_holes(),
            self.avg_bump(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn final_report(
        &self,
        policy_name: &str,
        rule_kind: PieceRuleKind,
        warmup_rows: u8,
        warmup_holes: u8,
        total_lines: u64,
        total_score: u64,
        last_ep_len: u64,
        last_game_over: bool,
    ) -> FinalReport {
        FinalReport {
            policy: policy_name.to_string(),
            piece_rule: rule_kind,

            warmup_rows,
            warmup_holes,

            steps_done: self.steps_done,
            elapsed_s: self.elapsed_secs(),
            steps_per_s: self.steps_per_sec(),
            episodes_finished: self.episodes_finished,
            avg_ep_len: self.avg_ep_len(),
            max_ep_len: self.episode_len_max,
            lines_per_step: self.lines_per_step(total_lines),
            score_per_step: self.score_per_step(total_score),
            max_h_worst: self.max_h_worst,
            avg_max_h: self.avg_max_h(),
            avg_h: self.avg_avg_h(),
            avg_agg_h: self.avg_agg_h(),
            avg_holes: self.avg_holes(),
            avg_bump: self.avg_bump(),
            avg_d_max_h: self.avg_d_max_h(),
            avg_d_agg_h: self.avg_d_agg_h(),
            avg_d_holes: self.avg_d_holes(),
            avg_d_bump: self.avg_d_bump(),
            total_score,
            total_lines,
            last_ep_len,
            last_game_over,
        }
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct FinalReport {
    pub policy: String,
    pub piece_rule: PieceRuleKind,

    pub warmup_rows: u8,
    pub warmup_holes: u8,

    pub steps_done: u64,
    pub elapsed_s: f64,
    pub steps_per_s: f64,

    pub episodes_finished: u64,
    pub avg_ep_len: f64,
    pub max_ep_len: u64,

    pub lines_per_step: f64,
    pub score_per_step: f64,

    pub max_h_worst: u32,
    pub avg_max_h: f64,
    pub avg_h: f64,

    pub avg_agg_h: f64,
    pub avg_holes: f64,
    pub avg_bump: f64,

    pub avg_d_max_h: f64,
    pub avg_d_agg_h: f64,
    pub avg_d_holes: f64,
    pub avg_d_bump: f64,

    pub total_score: u64,
    pub total_lines: u64,

    pub last_ep_len: u64,
    pub last_game_over: bool,
}
