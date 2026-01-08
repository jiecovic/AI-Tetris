// rust/engine/src/pieces.rs
#![forbid(unsafe_code)]

#[derive(Clone, Copy, Debug)]
pub enum Kind {
    I,
    O,
    T,
    S,
    Z,
    J,
    L,
}

impl Kind {
    pub fn all() -> &'static [Kind] {
        use Kind::*;
        &[I, O, T, S, Z, J, L]
    }

    /// Strict 1..=7 id used for grid cell encoding (0 = empty).
    pub fn idx(self) -> u8 {
        use Kind::*;
        match self {
            I => 1,
            O => 2,
            T => 3,
            S => 4,
            Z => 5,
            J => 6,
            L => 7,
        }
    }

    /// Inverse of `idx()` (1..=7). Returns None for invalid ids.
    pub fn from_idx(idx: u8) -> Option<Self> {
        use Kind::*;
        match idx {
            1 => Some(I),
            2 => Some(O),
            3 => Some(T),
            4 => Some(S),
            5 => Some(Z),
            6 => Some(J),
            7 => Some(L),
            _ => None,
        }
    }

    pub fn glyph(self) -> char {
        use Kind::*;
        match self {
            I => 'I',
            O => 'O',
            T => 'T',
            S => 'S',
            Z => 'Z',
            J => 'J',
            L => 'L',
        }
    }

    /// Number of *distinct* rotations for this piece (Classic7 semantics).
    ///
    /// This is SSOT for "redundant rotation slots are invalid".
    #[inline]
    pub fn num_rots(self) -> usize {
        rotations(self).len()
    }
}

/// Rotations are represented as (dx, dy) offsets.
/// `rotations(kind)[rot]` returns a slice of 4 blocks.
///
/// IMPORTANT:
/// - This returns ONLY distinct rotations (Classic7), matching the old YAML.
/// - Fixed action encoding still uses MAX_ROTS=4 "rotation slots"; slots with
///   rot_u >= kind.num_rots() must be treated as invalid by the engine.
pub fn rotations(kind: Kind) -> &'static [&'static [(i32, i32)]] {
    use Kind::*;
    match kind {
        // 1 rotation
        O => &[&[(0, 0), (1, 0), (0, 1), (1, 1)]],

        // 2 rotations
        I => &[
            &[(0, 0), (1, 0), (2, 0), (3, 0)],
            &[(1, 0), (1, 1), (1, 2), (1, 3)],
        ],
        S => &[
            &[(1, 0), (2, 0), (0, 1), (1, 1)],
            &[(1, 0), (1, 1), (2, 1), (2, 2)],
        ],
        Z => &[
            &[(0, 0), (1, 0), (1, 1), (2, 1)],
            &[(2, 0), (1, 1), (2, 1), (1, 2)],
        ],

        // 4 rotations
        T => &[
            &[(1, 0), (0, 1), (1, 1), (2, 1)],
            &[(1, 0), (1, 1), (2, 1), (1, 2)],
            &[(0, 1), (1, 1), (2, 1), (1, 2)],
            &[(1, 0), (0, 1), (1, 1), (1, 2)],
        ],
        J => &[
            &[(0, 0), (0, 1), (1, 1), (2, 1)],
            &[(1, 0), (2, 0), (1, 1), (1, 2)],
            &[(0, 1), (1, 1), (2, 1), (2, 2)],
            &[(1, 0), (1, 1), (0, 2), (1, 2)],
        ],
        L => &[
            &[(2, 0), (0, 1), (1, 1), (2, 1)],
            &[(1, 0), (1, 1), (1, 2), (2, 2)],
            &[(0, 1), (1, 1), (2, 1), (0, 2)],
            &[(0, 0), (1, 0), (1, 1), (1, 2)],
        ],
    }
}

/// UI helper: rasterize a piece rotation into a 4x4 mask.
///
/// - `rot` is clamped to the last distinct rotation for `kind`.
/// - `fill` is the value written into occupied cells (recommended: `kind.idx()`).
///
/// Returns a 4x4 grid in row-major order: m[y][x].
pub fn preview_mask_4x4(kind: Kind, rot: usize, fill: u8) -> [[u8; 4]; 4] {
    let mut m = [[0u8; 4]; 4];

    let rots = rotations(kind);
    let r = if rots.is_empty() {
        0
    } else {
        rot.min(rots.len() - 1)
    };

    for &(dx, dy) in rots[r] {
        // The hardcoded rotation tables are designed to fit a 4x4 preview.
        // Still guard bounds for robustness.
        if dx < 0 || dy < 0 {
            continue;
        }
        let x = dx as usize;
        let y = dy as usize;
        if x < 4 && y < 4 {
            m[y][x] = fill;
        }
    }

    m
}
