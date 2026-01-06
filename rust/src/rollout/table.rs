// src/rollout/table.rs
#![forbid(unsafe_code)]

use crate::rollout::features::StepFeatures;

/// One periodic row printed by the table reporter.
///
/// Keep this as a "transport struct": Runner/Stats compute everything,
/// TablePrinter just formats.
#[derive(Clone, Debug)]
pub struct ReportRow {
    pub step: u64,
    pub steps_total: u64,

    pub sps: f64,

    pub episodes_finished: u64,
    pub avg_ep_len: f64,
    pub max_ep_len: u64,

    pub lines_per_step: f64,
    pub score_per_step: f64,

    pub avg_max_h: f64,
    pub avg_h: f64,

    pub total_lines: u64,
    pub total_score: u64,

    /// Optional heavier grid features + deltas.
    pub features: Option<StepFeatures>,
}

pub struct TablePrinter {
    every: u64,
    header_every: u64,
    rows_printed: u64,
}

impl TablePrinter {
    /// `every`: print a row every N steps. If 0 => disabled.
    /// `header_every`: re-print header every N rows.
    pub fn new(every: u64, header_every: u64) -> Self {
        Self {
            every,
            header_every: header_every.max(1),
            rows_printed: 0,
        }
    }

    pub fn enabled(&self) -> bool {
        self.every > 0
    }

    pub fn maybe_print(&mut self, row: &ReportRow) {
        if self.every == 0 {
            return;
        }
        if row.step == 0 || (row.step % self.every != 0) {
            return;
        }

        if self.rows_printed % self.header_every == 0 {
            self.print_header(row.features.is_some());
        }

        self.print_row(row);
        self.rows_printed += 1;
    }

    fn print_header(&self, with_features: bool) {
        if with_features {
            println!(
                "{:>10} {:>9} {:>7} {:>10} {:>10} {:>10} {:>11} {:>9} {:>9} \
                 {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}",
                "step",
                "sps",
                "eps",
                "avg_ep",
                "max_ep",
                "l/step",
                "score/step",
                "avgMaxH",
                "avgH",
                "maxH",
                "aggH",
                "holes",
                "bump",
                "dMax",
                "dAgg",
                "dHol",
                "dBum",
            );
        } else {
            println!(
                "{:>10} {:>9} {:>7} {:>10} {:>10} {:>10} {:>11} {:>9} {:>9} {:>12} {:>12}",
                "step",
                "sps",
                "eps",
                "avg_ep",
                "max_ep",
                "l/step",
                "score/step",
                "avgMaxH",
                "avgH",
                "total_lines",
                "total_score",
            );
        }
    }

    fn print_row(&self, r: &ReportRow) {
        if let Some(f) = r.features {
            println!(
                "{:>10}/{:<10} {:>9.1} {:>7} {:>10.1} {:>10} {:>10.3} {:>11.2} {:>9.2} {:>9.2} \
                 {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}",
                r.step,
                r.steps_total,
                r.sps,
                r.episodes_finished,
                r.avg_ep_len,
                r.max_ep_len,
                r.lines_per_step,
                r.score_per_step,
                r.avg_max_h,
                r.avg_h,
                f.cur.max_h,
                f.cur.agg_h,
                f.cur.holes,
                f.cur.bump,
                f.delta.d_max_h,
                f.delta.d_agg_h,
                f.delta.d_holes,
                f.delta.d_bump,
            );
        } else {
            println!(
                "{:>10}/{:<10} {:>9.1} {:>7} {:>10.1} {:>10} {:>10.3} {:>11.2} {:>9.2} {:>9.2} {:>12} {:>12}",
                r.step,
                r.steps_total,
                r.sps,
                r.episodes_finished,
                r.avg_ep_len,
                r.max_ep_len,
                r.lines_per_step,
                r.score_per_step,
                r.avg_max_h,
                r.avg_h,
                r.total_lines,
                r.total_score,
            );
        }
    }
}
