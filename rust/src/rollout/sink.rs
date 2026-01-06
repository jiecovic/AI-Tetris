// src/rollout/sink.rs
#![forbid(unsafe_code)]

use crate::rollout::table::{ReportRow, TablePrinter};

pub trait RolloutSink {
    /// Called on periodic cadence with a fully-prepared report row.
    fn on_report(&mut self, _row: &ReportRow) {}
}

/// Default: absolutely minimal overhead.
#[derive(Default)]
pub struct NoopSink;

impl RolloutSink for NoopSink {}

/// Prints periodic rows to stdout.
pub struct TableSink {
    printer: TablePrinter,
}

impl TableSink {
    pub fn new(every_steps: u64, header_every_rows: u64) -> Self {
        Self {
            printer: TablePrinter::new(every_steps, header_every_rows),
        }
    }

    pub fn enabled(&self) -> bool {
        self.printer.enabled()
    }
}

impl RolloutSink for TableSink {
    fn on_report(&mut self, row: &ReportRow) {
        self.printer.maybe_print(row);
    }
}
