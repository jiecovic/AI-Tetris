// rust/engine/src/policy/heuristic.rs
#![forbid(unsafe_code)]

use crate::engine::{Game, H, HIDDEN_ROWS, W};
use crate::policy::base::Policy;
use crate::policy::beam::BeamConfig;
use crate::policy::codemy::{GridScorer, SearchCore, UniformIID};

#[derive(Clone, Copy, Debug)]
pub enum HeuristicFeature {
    AggHeight,
    MaxHeight,
    MeanHeight,
    StdHeight,
    VarHeight,
    Holes,
    HolesStdCol,
    HolesStdRow,
    Bumpiness,
    CompleteLines,
    AggHeightNorm,
    HolesNorm,
    BumpinessNorm,
    CompleteLinesNorm,
}

impl HeuristicFeature {
    pub fn parse(name: &str) -> Option<Self> {
        let k = name.trim().to_ascii_lowercase();
        match k.as_str() {
            "agg_h" | "agg_height" | "aggregate_height" => Some(Self::AggHeight),
            "max_h" | "max_height" => Some(Self::MaxHeight),
            "mean_h" | "mean_height" | "avg_h" | "avg_height" => Some(Self::MeanHeight),
            "std_h" | "std_height" => Some(Self::StdHeight),
            "var_h" | "var_height" | "variance_height" => Some(Self::VarHeight),
            "holes" | "holes_total" => Some(Self::Holes),
            "holes_std_col" | "holes_std_cols" | "holes_std_column" => Some(Self::HolesStdCol),
            "holes_std_row" | "holes_std_rows" | "holes_std_rowwise" => Some(Self::HolesStdRow),
            "bump" | "bumpiness" => Some(Self::Bumpiness),
            "complete_lines" | "lines" => Some(Self::CompleteLines),
            "agg_h_norm" | "agg_height_norm" | "aggregate_height_norm" => Some(Self::AggHeightNorm),
            "holes_norm" | "holes_total_norm" => Some(Self::HolesNorm),
            "bump_norm" | "bumpiness_norm" => Some(Self::BumpinessNorm),
            "complete_lines_norm" | "lines_norm" => Some(Self::CompleteLinesNorm),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct HeuristicScorer {
    features: Vec<HeuristicFeature>,
    weights: Vec<f64>,
}

impl HeuristicScorer {
    pub fn new(features: Vec<HeuristicFeature>, weights: Vec<f64>) -> Result<Self, String> {
        if features.is_empty() {
            return Err("features must not be empty".to_string());
        }
        if features.len() != weights.len() {
            return Err(format!(
                "features/weights length mismatch: {} vs {}",
                features.len(),
                weights.len()
            ));
        }
        Ok(Self { features, weights })
    }

    fn score_grid(&self, grid: &[[u8; W]; H]) -> f64 {
        let mut need_heights = false;
        let mut need_holes_col = false;
        let mut need_holes_row = false;
        let mut need_bump = false;
        let mut need_holes_std_col = false;
        let mut need_holes_std_row = false;
        let mut need_complete_lines = false;

        for f in &self.features {
            match f {
                HeuristicFeature::AggHeight
                | HeuristicFeature::MaxHeight
                | HeuristicFeature::MeanHeight
                | HeuristicFeature::StdHeight
                | HeuristicFeature::VarHeight
                | HeuristicFeature::Bumpiness
                | HeuristicFeature::AggHeightNorm
                | HeuristicFeature::BumpinessNorm => {
                    need_heights = true;
                    need_bump = true;
                }
                HeuristicFeature::Holes | HeuristicFeature::HolesNorm => {
                    need_holes_col = true;
                    need_heights = true;
                }
                HeuristicFeature::HolesStdCol => {
                    need_holes_col = true;
                    need_heights = true;
                    need_holes_std_col = true;
                }
                HeuristicFeature::HolesStdRow => {
                    need_holes_row = true;
                    need_heights = true;
                    need_holes_std_row = true;
                }
                HeuristicFeature::CompleteLines | HeuristicFeature::CompleteLinesNorm => {
                    need_complete_lines = true;
                }
            }
        }

        let mut heights = [0u32; W];
        let mut holes_col = [0u32; W];
        let mut holes_row = [0u32; H];

        if need_heights || need_holes_col || need_holes_row {
            for c in 0..W {
                let mut seen_block = false;
                for r in 0..H {
                    let filled = grid[r][c] != 0;
                    if filled {
                        if !seen_block {
                            heights[c] = (H - r) as u32;
                            seen_block = true;
                        }
                    } else if seen_block {
                        if need_holes_col {
                            holes_col[c] += 1;
                        }
                        if need_holes_row {
                            holes_row[r] += 1;
                        }
                    }
                }
            }
        }

        let mut agg_h: f64 = 0.0;
        let mut max_h: f64 = 0.0;
        let mut sumsq_h: f64 = 0.0;
        if need_heights {
            for h in heights {
                let hf = h as f64;
                agg_h += hf;
                sumsq_h += hf * hf;
                if hf > max_h {
                    max_h = hf;
                }
            }
        }
        let mean_h = if need_heights {
            agg_h / (W as f64)
        } else {
            0.0
        };
        let var_h = if need_heights {
            let v = (sumsq_h / (W as f64)) - (mean_h * mean_h);
            if v < 0.0 { 0.0 } else { v }
        } else {
            0.0
        };
        let std_h = if need_heights { var_h.sqrt() } else { 0.0 };

        let mut bump: f64 = 0.0;
        if need_bump {
            for i in 0..(W - 1) {
                let a = heights[i] as i32;
                let b = heights[i + 1] as i32;
                bump += (a - b).abs() as f64;
            }
        }

        let holes_total = if need_holes_col {
            holes_col.iter().map(|&v| v as f64).sum::<f64>()
        } else {
            0.0
        };

        let holes_std_col = if need_holes_std_col {
            let mean = holes_total / (W as f64);
            let mut sumsq = 0.0;
            for v in holes_col {
                let dv = (v as f64) - mean;
                sumsq += dv * dv;
            }
            (sumsq / (W as f64)).sqrt()
        } else {
            0.0
        };

        let holes_std_row = if need_holes_std_row {
            let rows = H as f64;
            let mean = holes_row.iter().map(|&v| v as f64).sum::<f64>() / rows;
            let mut sumsq = 0.0;
            for v in holes_row {
                let dv = (v as f64) - mean;
                sumsq += dv * dv;
            }
            (sumsq / rows).sqrt()
        } else {
            0.0
        };

        let complete_lines = if need_complete_lines {
            grid[HIDDEN_ROWS..]
                .iter()
                .filter(|row| row.iter().all(|&c| c != 0))
                .count() as f64
        } else {
            0.0
        };

        let max_cells = (H * W) as f64;
        let max_bump = (H * (W - 1)) as f64;
        let max_lines = 4.0; // max cleared lines per placement for a tetromino

        let mut score = 0.0;
        for (feat, w) in self.features.iter().zip(self.weights.iter()) {
            let v = match feat {
                HeuristicFeature::AggHeight => agg_h,
                HeuristicFeature::MaxHeight => max_h,
                HeuristicFeature::MeanHeight => mean_h,
                HeuristicFeature::StdHeight => std_h,
                HeuristicFeature::VarHeight => var_h,
                HeuristicFeature::Holes => holes_total,
                HeuristicFeature::HolesStdCol => holes_std_col,
                HeuristicFeature::HolesStdRow => holes_std_row,
                HeuristicFeature::Bumpiness => bump,
                HeuristicFeature::CompleteLines => complete_lines,
                HeuristicFeature::AggHeightNorm => {
                    if max_cells > 0.0 {
                        agg_h / max_cells
                    } else {
                        0.0
                    }
                }
                HeuristicFeature::HolesNorm => {
                    if max_cells > 0.0 {
                        holes_total / max_cells
                    } else {
                        0.0
                    }
                }
                HeuristicFeature::BumpinessNorm => {
                    if max_bump > 0.0 {
                        bump / max_bump
                    } else {
                        0.0
                    }
                }
                HeuristicFeature::CompleteLinesNorm => {
                    if max_lines > 0.0 {
                        complete_lines / max_lines
                    } else {
                        0.0
                    }
                }
            };
            score += w * v;
        }

        score
    }
}

pub fn compute_feature_values(grid: &[[u8; W]; H], features: &[HeuristicFeature]) -> Vec<f64> {
    if features.is_empty() {
        return Vec::new();
    }

    let mut need_heights = false;
    let mut need_holes_col = false;
    let mut need_holes_row = false;
    let mut need_bump = false;
    let mut need_holes_std_col = false;
    let mut need_holes_std_row = false;
    let mut need_complete_lines = false;

    for f in features {
        match f {
            HeuristicFeature::AggHeight
            | HeuristicFeature::MaxHeight
            | HeuristicFeature::MeanHeight
            | HeuristicFeature::StdHeight
            | HeuristicFeature::VarHeight
            | HeuristicFeature::Bumpiness
            | HeuristicFeature::AggHeightNorm
            | HeuristicFeature::BumpinessNorm => {
                need_heights = true;
                need_bump = true;
            }
            HeuristicFeature::Holes | HeuristicFeature::HolesNorm => {
                need_holes_col = true;
                need_heights = true;
            }
            HeuristicFeature::HolesStdCol => {
                need_holes_col = true;
                need_heights = true;
                need_holes_std_col = true;
            }
            HeuristicFeature::HolesStdRow => {
                need_holes_row = true;
                need_heights = true;
                need_holes_std_row = true;
            }
            HeuristicFeature::CompleteLines | HeuristicFeature::CompleteLinesNorm => {
                need_complete_lines = true;
            }
        }
    }

    let mut heights = [0u32; W];
    let mut holes_col = [0u32; W];
    let mut holes_row = [0u32; H];

    if need_heights || need_holes_col || need_holes_row {
        for c in 0..W {
            let mut seen_block = false;
            for r in 0..H {
                let filled = grid[r][c] != 0;
                if filled {
                    if !seen_block {
                        heights[c] = (H - r) as u32;
                        seen_block = true;
                    }
                } else if seen_block {
                    if need_holes_col {
                        holes_col[c] += 1;
                    }
                    if need_holes_row {
                        holes_row[r] += 1;
                    }
                }
            }
        }
    }

    let mut agg_h: f64 = 0.0;
    let mut max_h: f64 = 0.0;
    let mut sumsq_h: f64 = 0.0;
    if need_heights {
        for h in heights {
            let hf = h as f64;
            agg_h += hf;
            sumsq_h += hf * hf;
            if hf > max_h {
                max_h = hf;
            }
        }
    }
    let mean_h = if need_heights {
        agg_h / (W as f64)
    } else {
        0.0
    };
    let var_h = if need_heights {
        let v = (sumsq_h / (W as f64)) - (mean_h * mean_h);
        if v < 0.0 { 0.0 } else { v }
    } else {
        0.0
    };
    let std_h = if need_heights { var_h.sqrt() } else { 0.0 };

    let mut bump: f64 = 0.0;
    if need_bump {
        for i in 0..(W - 1) {
            let a = heights[i] as i32;
            let b = heights[i + 1] as i32;
            bump += (a - b).abs() as f64;
        }
    }

    let holes_total = if need_holes_col {
        holes_col.iter().map(|&v| v as f64).sum::<f64>()
    } else {
        0.0
    };

    let holes_std_col = if need_holes_std_col {
        let mean = holes_total / (W as f64);
        let mut sumsq = 0.0;
        for v in holes_col {
            let dv = (v as f64) - mean;
            sumsq += dv * dv;
        }
        (sumsq / (W as f64)).sqrt()
    } else {
        0.0
    };

    let holes_std_row = if need_holes_std_row {
        let rows = H as f64;
        let mean = holes_row.iter().map(|&v| v as f64).sum::<f64>() / rows;
        let mut sumsq = 0.0;
        for v in holes_row {
            let dv = (v as f64) - mean;
            sumsq += dv * dv;
        }
        (sumsq / rows).sqrt()
    } else {
        0.0
    };

    let complete_lines = if need_complete_lines {
        grid[HIDDEN_ROWS..]
            .iter()
            .filter(|row| row.iter().all(|&c| c != 0))
            .count() as f64
    } else {
        0.0
    };

    let max_cells = (H * W) as f64;
    let max_bump = (H * (W - 1)) as f64;
    let max_lines = 4.0; // max cleared lines per placement for a tetromino

    let mut out = Vec::with_capacity(features.len());
    for feat in features {
        let v = match feat {
            HeuristicFeature::AggHeight => agg_h,
            HeuristicFeature::MaxHeight => max_h,
            HeuristicFeature::MeanHeight => mean_h,
            HeuristicFeature::StdHeight => std_h,
            HeuristicFeature::VarHeight => var_h,
            HeuristicFeature::Holes => holes_total,
            HeuristicFeature::HolesStdCol => holes_std_col,
            HeuristicFeature::HolesStdRow => holes_std_row,
            HeuristicFeature::Bumpiness => bump,
            HeuristicFeature::CompleteLines => complete_lines,
            HeuristicFeature::AggHeightNorm => {
                if max_cells > 0.0 {
                    agg_h / max_cells
                } else {
                    0.0
                }
            }
            HeuristicFeature::HolesNorm => {
                if max_cells > 0.0 {
                    holes_total / max_cells
                } else {
                    0.0
                }
            }
            HeuristicFeature::BumpinessNorm => {
                if max_bump > 0.0 {
                    bump / max_bump
                } else {
                    0.0
                }
            }
            HeuristicFeature::CompleteLinesNorm => {
                if max_lines > 0.0 {
                    complete_lines / max_lines
                } else {
                    0.0
                }
            }
        };
        out.push(v);
    }

    out
}

impl GridScorer for HeuristicScorer {
    fn score(&self, grid: &[[u8; W]; H]) -> f64 {
        self.score_grid(grid)
    }
}

pub struct HeuristicPolicy {
    core: SearchCore<HeuristicScorer>,
    plies: u8,
    score_after_clear: bool,
}

impl HeuristicPolicy {
    pub fn new(
        features: Vec<HeuristicFeature>,
        weights: Vec<f64>,
        plies: u8,
        beam: Option<BeamConfig>,
        score_after_clear: bool,
    ) -> Result<Self, String> {
        let scorer = HeuristicScorer::new(features, weights)?;
        Ok(Self {
            core: SearchCore::new_with_scorer(scorer, beam, score_after_clear),
            plies: plies.max(1),
            score_after_clear,
        })
    }
}

impl Policy for HeuristicPolicy {
    fn choose_action(&mut self, g: &Game) -> Option<usize> {
        let aid0_cands = self.core.aid0_candidates_with_proxy(g);
        if aid0_cands.is_empty() {
            return None;
        }

        let mut best: Option<(usize, f64)> = None;

        for (aid0, _proxy0) in aid0_cands {
            let v0 = if self.plies == 1 {
                if self.score_after_clear {
                    let sim = g.simulate_action_id_active(aid0);
                    if sim.invalid {
                        continue;
                    }
                    self.core.scorer().score(&sim.grid_after_clear)
                } else {
                    let Some(grid_lock) = g.simulate_action_id_active_lock_only(aid0) else {
                        continue;
                    };
                    self.core.scorer().score(&grid_lock)
                }
            } else {
                let sim1 = g.simulate_action_id_active(aid0);
                if sim1.invalid {
                    continue;
                }
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
