// src/rollout/stats.rs
#![forbid(unsafe_code)]

use std::time::Instant;

use crate::engine::PieceRuleKind;

#[derive(Clone, Debug)]
pub struct RolloutStats {
    pub episodes_finished: u64,
    pub ep_len: u64,
    pub episode_len_sum: u64,
    pub episode_len_max: u64,

    pub total_lines_finished: u64,
    pub total_score_finished: u64,

    pub steps_done: u64,
    pub sum_max_h: f64,
    pub sum_avg_h: f64,

    pub t0: Instant,
}

impl RolloutStats {
    pub fn new() -> Self {
        Self {
            episodes_finished: 0,
            ep_len: 0,
            episode_len_sum: 0,
            episode_len_max: 0,
            total_lines_finished: 0,
            total_score_finished: 0,
            steps_done: 0,
            sum_max_h: 0.0,
            sum_avg_h: 0.0,
            t0: Instant::now(),
        }
    }

    pub fn on_step(&mut self, cleared_lines: u32, max_h: u32, avg_h: f32) {
        let _ = cleared_lines;
        self.steps_done += 1;
        self.ep_len += 1;
        self.sum_max_h += max_h as f64;
        self.sum_avg_h += avg_h as f64;
    }

    pub fn on_episode_end(&mut self, lines: u64, score: u64) {
        self.episodes_finished += 1;
        self.episode_len_sum += self.ep_len;
        self.episode_len_max = self.episode_len_max.max(self.ep_len);

        self.total_lines_finished += lines;
        self.total_score_finished += score;

        self.ep_len = 0;
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

    pub fn live_msg(
        &self,
        rule_kind: PieceRuleKind,
        live_total_lines: u64,
        live_total_score: u64,
    ) -> LiveMsg {
        let sps = self.steps_per_sec();
        let lines_per_step = if self.steps_done > 0 {
            live_total_lines as f64 / self.steps_done as f64
        } else {
            0.0
        };
        let score_per_step = if self.steps_done > 0 {
            live_total_score as f64 / self.steps_done as f64
        } else {
            0.0
        };

        let msg = format!(
            "rule={:?} sps={:.1} eps_done={} avg_ep_len={:.1} max_ep_len={} l/step={:.3} score/step={:.2} avg_max_h={:.2} avg_h={:.2}",
            rule_kind,
            sps,
            self.episodes_finished,
            self.avg_ep_len(),
            self.episode_len_max,
            lines_per_step,
            score_per_step,
            self.avg_max_h(),
            self.avg_avg_h(),
        );

        LiveMsg { msg }
    }

    pub fn final_report(
        &self,
        policy_name: &str,
        rule_kind: PieceRuleKind,
        total_lines: u64,
        total_score: u64,
        last_ep_len: u64,
        last_game_over: bool,
    ) -> FinalReport {
        let dt = self.elapsed_secs();
        let sps = self.steps_per_sec();

        let lines_per_step = if self.steps_done > 0 {
            total_lines as f64 / self.steps_done as f64
        } else {
            0.0
        };
        let score_per_step = if self.steps_done > 0 {
            total_score as f64 / self.steps_done as f64
        } else {
            0.0
        };

        FinalReport {
            policy: policy_name.to_string(),
            piece_rule: rule_kind,
            steps_done: self.steps_done,
            elapsed_s: dt,
            steps_per_s: sps,
            episodes_finished: self.episodes_finished,
            avg_ep_len: self.avg_ep_len(),
            max_ep_len: self.episode_len_max,
            lines_per_step,
            score_per_step,
            avg_max_h: self.avg_max_h(),
            avg_h: self.avg_avg_h(),
            total_score,
            total_lines,
            last_ep_len,
            last_game_over,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LiveMsg {
    pub msg: String,
}

#[derive(Clone, Debug)]
pub struct FinalReport {
    pub policy: String,
    pub piece_rule: PieceRuleKind,
    pub steps_done: u64,
    pub elapsed_s: f64,
    pub steps_per_s: f64,
    pub episodes_finished: u64,
    pub avg_ep_len: f64,
    pub max_ep_len: u64,
    pub lines_per_step: f64,
    pub score_per_step: f64,
    pub avg_max_h: f64,
    pub avg_h: f64,
    pub total_score: u64,
    pub total_lines: u64,
    pub last_ep_len: u64,
    pub last_game_over: bool,
}
