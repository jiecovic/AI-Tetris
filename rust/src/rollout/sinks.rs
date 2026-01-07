// src/rollout/sinks.rs
#![forbid(unsafe_code)]

/// One periodic row emitted by the runner.
///
/// Transport struct: runner/stats compute fields, sinks only format/emit.
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

    /// Worst (max) column height observed so far (over all steps).
    pub max_h_worst: u32,

    /// Averages over all steps so far.
    pub avg_max_h: f64,
    pub avg_avg_h: f64,

    pub avg_agg_h: f64,
    pub avg_holes: f64,
    pub avg_bump: f64,

    /// Average deltas per step (within episodes; resets on episode reset).
    pub avg_d_max_h: f64,
    pub avg_d_agg_h: f64,
    pub avg_d_holes: f64,
    pub avg_d_bump: f64,
}

/// Sink interface for periodic reporting (table/logging/dataset emission later).
pub trait RolloutSink {
    fn on_report_row(&mut self, row: &ReportRow, pb: Option<&indicatif::ProgressBar>);
}

/// Default sink: does nothing.
#[derive(Default)]
pub struct NoopSink;

impl RolloutSink for NoopSink {
    fn on_report_row(&mut self, _row: &ReportRow, _pb: Option<&indicatif::ProgressBar>) {}
}

/// Human-readable periodic table sink.
///
/// Cadence (every N steps) is handled by Runner. This sink prints whenever called.
pub struct TableSink {
    header_every: u64,
    rows_printed: u64,
}

impl TableSink {
    const DEFAULT_HEADER_EVERY: u64 = 20;

    /// If `header_every == 0`, a reasonable default is used.
    pub fn new(header_every: u64) -> Self {
        Self {
            header_every: if header_every == 0 {
                Self::DEFAULT_HEADER_EVERY
            } else {
                header_every
            },
            rows_printed: 0,
        }
    }

    fn header_line(&self) -> String {
        // Note: keep widths aligned with row_line() below.
        // The Δ-columns are printed in scientific notation, so we give them extra width.
        format!(
            "{:>21} {:>9} {:>5} {:>9} {:>9} {:>8} {:>10} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8} {:>12} {:>12} {:>12} {:>12}",
            "step/total",
            "sps",
            "eps",
            "avg_ep",
            "max_ep",
            "l/step",
            "score/st",
            "maxH",
            "avgMaxH",
            "avgH",
            "avgAgg",
            "avgHol",
            "avgBum",
            "ΔMaxH̄",
            "ΔAggH̄",
            "ΔHol̄",
            "ΔBum̄",
        )
    }

    fn sep_line(&self) -> String {
        "-".repeat(self.header_line().len())
    }

    fn row_line(&self, r: &ReportRow) -> String {
        format!(
            "{:>10}/{:<10} {:>9.1} {:>5} {:>9.1} {:>9} {:>8.3} {:>10.2} {:>6} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>12.2e} {:>12.2e} {:>12.2e} {:>12.2e}",
            r.step,
            r.steps_total,
            r.sps,
            r.episodes_finished,
            r.avg_ep_len,
            r.max_ep_len,
            r.lines_per_step,
            r.score_per_step,
            r.max_h_worst,
            r.avg_max_h,
            r.avg_avg_h,
            r.avg_agg_h,
            r.avg_holes,
            r.avg_bump,
            r.avg_d_max_h,
            r.avg_d_agg_h,
            r.avg_d_holes,
            r.avg_d_bump,
        )
    }
}

impl RolloutSink for TableSink {
    fn on_report_row(&mut self, row: &ReportRow, pb: Option<&indicatif::ProgressBar>) {
        let mut lines: Vec<String> = Vec::new();

        if self.rows_printed == 0 || (self.rows_printed % self.header_every == 0) {
            lines.push(self.header_line());
            lines.push(self.sep_line());
        }

        lines.push(self.row_line(row));
        self.rows_printed += 1;

        if let Some(pb) = pb {
            for l in lines {
                pb.println(l);
            }
        } else {
            for l in lines {
                println!("{l}");
            }
        }
    }
}
