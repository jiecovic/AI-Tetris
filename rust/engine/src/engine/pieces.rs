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
