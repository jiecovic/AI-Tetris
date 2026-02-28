// rust/engine/src/policy/heuristic.rs
#![forbid(unsafe_code)]

use std::fmt;

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

#[derive(Clone, Copy, Debug, Default)]
struct FeatureNeeds {
    need_heights: bool,
    need_holes_col: bool,
    need_holes_row: bool,
    need_bump: bool,
    need_holes_std_col: bool,
    need_holes_std_row: bool,
    need_complete_lines: bool,
}

fn feature_needs(features: &[HeuristicFeature]) -> FeatureNeeds {
    let mut needs = FeatureNeeds::default();

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
                needs.need_heights = true;
                needs.need_bump = true;
            }
            HeuristicFeature::Holes | HeuristicFeature::HolesNorm => {
                needs.need_holes_col = true;
                needs.need_heights = true;
            }
            HeuristicFeature::HolesStdCol => {
                needs.need_holes_col = true;
                needs.need_heights = true;
                needs.need_holes_std_col = true;
            }
            HeuristicFeature::HolesStdRow => {
                needs.need_holes_row = true;
                needs.need_heights = true;
                needs.need_holes_std_row = true;
            }
            HeuristicFeature::CompleteLines | HeuristicFeature::CompleteLinesNorm => {
                needs.need_complete_lines = true;
            }
        }
    }

    needs
}

#[derive(Clone, Copy, Debug, Default)]
struct FeatureStats {
    agg_h: f64,
    max_h: f64,
    mean_h: f64,
    std_h: f64,
    var_h: f64,
    holes_total: f64,
    holes_std_col: f64,
    holes_std_row: f64,
    bump: f64,
    complete_lines: f64,
}

impl FeatureStats {
    #[inline]
    fn value(self, feat: HeuristicFeature) -> f64 {
        let max_cells = (H * W) as f64;
        let max_bump = (H * (W - 1)) as f64;
        let max_lines = 4.0; // max cleared lines per placement for a tetromino

        match feat {
            HeuristicFeature::AggHeight => self.agg_h,
            HeuristicFeature::MaxHeight => self.max_h,
            HeuristicFeature::MeanHeight => self.mean_h,
            HeuristicFeature::StdHeight => self.std_h,
            HeuristicFeature::VarHeight => self.var_h,
            HeuristicFeature::Holes => self.holes_total,
            HeuristicFeature::HolesStdCol => self.holes_std_col,
            HeuristicFeature::HolesStdRow => self.holes_std_row,
            HeuristicFeature::Bumpiness => self.bump,
            HeuristicFeature::CompleteLines => self.complete_lines,
            HeuristicFeature::AggHeightNorm => {
                if max_cells > 0.0 {
                    self.agg_h / max_cells
                } else {
                    0.0
                }
            }
            HeuristicFeature::HolesNorm => {
                if max_cells > 0.0 {
                    self.holes_total / max_cells
                } else {
                    0.0
                }
            }
            HeuristicFeature::BumpinessNorm => {
                if max_bump > 0.0 {
                    self.bump / max_bump
                } else {
                    0.0
                }
            }
            HeuristicFeature::CompleteLinesNorm => {
                if max_lines > 0.0 {
                    self.complete_lines / max_lines
                } else {
                    0.0
                }
            }
        }
    }
}

fn compute_feature_stats(grid: &[[u8; W]; H], needs: FeatureNeeds) -> FeatureStats {
    let mut heights = [0u32; W];
    let mut holes_col = [0u32; W];
    let mut holes_row = [0u32; H];

    if needs.need_heights || needs.need_holes_col || needs.need_holes_row {
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
                    if needs.need_holes_col {
                        holes_col[c] += 1;
                    }
                    if needs.need_holes_row {
                        holes_row[r] += 1;
                    }
                }
            }
        }
    }

    let mut agg_h = 0.0;
    let mut max_h = 0.0;
    let mut sumsq_h = 0.0;
    if needs.need_heights {
        for h in heights {
            let hf = h as f64;
            agg_h += hf;
            sumsq_h += hf * hf;
            if hf > max_h {
                max_h = hf;
            }
        }
    }
    let mean_h = if needs.need_heights {
        agg_h / (W as f64)
    } else {
        0.0
    };
    let var_h = if needs.need_heights {
        let v = (sumsq_h / (W as f64)) - (mean_h * mean_h);
        if v < 0.0 { 0.0 } else { v }
    } else {
        0.0
    };
    let std_h = if needs.need_heights {
        var_h.sqrt()
    } else {
        0.0
    };

    let mut bump = 0.0;
    if needs.need_bump {
        for i in 0..(W - 1) {
            let a = heights[i] as i32;
            let b = heights[i + 1] as i32;
            bump += (a - b).abs() as f64;
        }
    }

    let holes_total = if needs.need_holes_col {
        holes_col.iter().map(|&v| v as f64).sum::<f64>()
    } else {
        0.0
    };

    let holes_std_col = if needs.need_holes_std_col {
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

    let holes_std_row = if needs.need_holes_std_row {
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

    let complete_lines = if needs.need_complete_lines {
        grid[HIDDEN_ROWS..]
            .iter()
            .filter(|row| row.iter().all(|&c| c != 0))
            .count() as f64
    } else {
        0.0
    };

    FeatureStats {
        agg_h,
        max_h,
        mean_h,
        std_h,
        var_h,
        holes_total,
        holes_std_col,
        holes_std_row,
        bump,
        complete_lines,
    }
}

#[derive(Clone, Debug)]
pub struct HeuristicScorer {
    features: Vec<HeuristicFeature>,
    weights: Vec<f64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HeuristicBuildError {
    EmptyFeatures,
    LengthMismatch { features: usize, weights: usize },
}

impl fmt::Display for HeuristicBuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HeuristicBuildError::EmptyFeatures => write!(f, "features must not be empty"),
            HeuristicBuildError::LengthMismatch { features, weights } => write!(
                f,
                "features/weights length mismatch: {features} vs {weights}"
            ),
        }
    }
}

impl std::error::Error for HeuristicBuildError {}

impl HeuristicScorer {
    /**
     * Typed constructor for heuristic scorer config validation.
     */
    pub fn try_new(
        features: Vec<HeuristicFeature>,
        weights: Vec<f64>,
    ) -> Result<Self, HeuristicBuildError> {
        if features.is_empty() {
            return Err(HeuristicBuildError::EmptyFeatures);
        }
        if features.len() != weights.len() {
            return Err(HeuristicBuildError::LengthMismatch {
                features: features.len(),
                weights: weights.len(),
            });
        }
        Ok(Self { features, weights })
    }

    fn score_grid(&self, grid: &[[u8; W]; H]) -> f64 {
        let needs = feature_needs(&self.features);
        let stats = compute_feature_stats(grid, needs);
        self.features
            .iter()
            .zip(self.weights.iter())
            .map(|(feat, w)| *w * stats.value(*feat))
            .sum()
    }
}

pub fn compute_feature_values(grid: &[[u8; W]; H], features: &[HeuristicFeature]) -> Vec<f64> {
    if features.is_empty() {
        return Vec::new();
    }
    let needs = feature_needs(features);
    let stats = compute_feature_stats(grid, needs);
    features.iter().map(|f| stats.value(*f)).collect()
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
    /**
     * Typed constructor for policy config validation.
     */
    pub fn try_new(
        features: Vec<HeuristicFeature>,
        weights: Vec<f64>,
        plies: u8,
        beam: Option<BeamConfig>,
        score_after_clear: bool,
    ) -> Result<Self, HeuristicBuildError> {
        let scorer = HeuristicScorer::try_new(features, weights)?;
        Ok(Self {
            core: SearchCore::new_with_scorer(scorer, beam, score_after_clear),
            plies: plies.max(1),
            score_after_clear,
        })
    }

    pub fn new(
        features: Vec<HeuristicFeature>,
        weights: Vec<f64>,
        plies: u8,
        beam: Option<BeamConfig>,
        score_after_clear: bool,
    ) -> Result<Self, String> {
        Self::try_new(features, weights, plies, beam, score_after_clear).map_err(|e| e.to_string())
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
