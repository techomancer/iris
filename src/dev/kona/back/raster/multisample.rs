//! Multisample: the sample patterns, the sample mask's coverage words, and the state a context keeps.
//!
//! The definitions table's section on multisampling gives the sample count (a layout choice:
//! 4 and 8 are covered), the three enables and the sample-mask enable, the pattern token's
//! seven enumerants with a pair of location words for each count, and the 33 coverage words
//! each count's sample mask indexes. Sample locations are on an eighth-of-a-pixel grid, a
//! three-bit x and a three-bit y; the first location register holds samples 0 to 3 and the
//! second 4 to 7. The table grades the location words and the coverage words established
//! and their reading (the positions, one nibble or byte a pixel) inferred.
//!
//! What the table leaves unknown, and this model decides, each stated here and counted in `RasterCounts` (multisampled draws, with alpha to mask, with a sample mask):
//!
//! - A location is measured from the pixel's integer corner, the same corner the single-sample
//!   path's one sample sits at (the table does not say whether the origin is a corner or the
//!   centre). Edges and ties are as set-up has them, with the sample moved.
//! - The samples live in the layout's own words, whose sizes say how: a multisampled window
//!   reserves 3 words a 4 samples for colour (24 bits a sample, red, green and blue; alpha is
//!   not kept), 3 words a 4 for 24-bit depth (S/2 at 16 bits) and 1 a 4 for 8-bit stencil, in place
//!   of the plain depth and stencil words. The model packs a sample's value at `k` times its
//!   width from the run's first bit, across word boundaries (the table gives no layout in memory;
//!   the sizes are the only evidence, so the reading is inferred). A fragment is shaded once,
//!   at the pixel's corner, and run through the stencil and depth tests and the colour operations
//!   once a covered sample, against that sample's values, with the depth at the sample's own
//!   position. The pixel's colour word is then the mean of its samples, so two triangles that
//!   share an edge cover a straddling pixel between them without a seam. The word's alpha is the
//!   last fragment's (samples keep none).
//!   The pipe has no resolve step: a read, a copy or a draw-buffer selection uses the colour and
//!   auxiliary banks and never reads the sample count, so a colour read most likely takes the
//!   maintained per-pixel colour (inferred), which is the mean word here. What a depth or
//!   stencil read returns, and how accumulation and a blended copy use the samples, is not
//!   readable from software (unknown).
//! - A clear sets every sample (the firmware has no multisample branch in a clear; which sample
//!   banks the hardware reaches is unknown).
//! - The sample mask's coverage word picks a pixel's nibble (4 samples) or byte (8) by the
//!   pixel's x modulo the number of them, and the bits are samples 0 up. The invert flag is a plain
//!   complement (established). That the word is ANDed with the primitive's own coverage is
//!   inferred from the table's shape and the vendor test's pairing.
//! - The board's registers for all this, the control word (0x1c03), the primitive sampling word
//!   (0x0806) and the depth clear word (0x1c09), are in `defs::multisample`. The model draws from
//!   its context state and not from those words, so the optimisation the firmware turns off under
//!   blending, a logic operation, a colour mask and a masked clear (values established, meaning
//!   inferred) has no effect on a pixel here.
//! - A mask value becomes a coverage index by rounding its product with 32 (the table says
//!   the float-to-index rounding is not recorded).
//! - Alpha to mask covers the first `round(alpha * samples)` samples (the table has no rule),
//!   and alpha to one sets the fragment's alpha to one afterwards.
//! - Lines and points go through the same triangles, so they are multisampled the same way
//!   (the table has no rule for them).

use crate::dev::kona::stream::Stream;

/// The sample patterns, 0x80a1 to 0x80a7, as a pair of location words for 4 samples and one for 8.
const PATTERN_FIRST: u32 = 0x80a1;
const WORDS4: [[u32; 2]; 7] = [[0xd94880; 2], [0xf92940; 2], [0xb19f80; 2], [0x4eaf40; 2], [0x793a40; 2], [0xaddc80; 2], [0xdad6c0; 2]];
const WORDS8: [[u32; 2]; 7] = [[0x655f80, 0xb0b9f2], [0xf77380, 0x913a99], [0xf32840, 0xad774e], [0x2d1fc0, 0x8b476e], [0x8ab240, 0x7fe535], [0x3b3e40, 0xb6251f], [0x5ede40, 0x2e27b4]];

/// The coverage word for a sample-mask index `n` (0 to 32).
const COVER4: [u32; 17] = [0, 0x11, 0x11000011, 0x11001111, 0x11111111, 0x11111133, 0x33111133, 0x33113333, 0x33333333, 0x33333377, 0x77333377, 0x77337777, 0x77777777, 0x777777ff, 0xff7777ff, 0xff77ffff, 0xffffffff];
const COVER8: [u32; 33] = [
    0x00000000, 0x00000001, 0x01000001, 0x01000101, 0x01010101, 0x01010111, 0x11010111, 0x11011111, 0x11111111, 0x11111113, 0x13111113, 0x13111313, 0x13131313, 0x13131333, 0x33131333, 0x33133333, 0x33333333, 0x33333337, 0x37333337, 0x37333737, 0x37373737, 0x37373777, 0x77373777, 0x77377777, 0x77777777, 0x7777777f,
    0x7f77777f, 0x7f777f7f, 0x7f7f7f7f, 0x7f7f7fff, 0xff7f7fff, 0xff7fffff, 0xffffffff,
];

/// What a context keeps of multisampling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Multisample {
    /// The sample count of the window's layout: 0 (none), 4 or 8.
    pub samples: u32,
    pub enable: bool,
    pub alpha_to_mask: bool,
    pub alpha_to_one: bool,
    pub mask_on: bool,
    pub pattern: u32,
    pub mask_value: f32,
    pub mask_invert: bool,
}

impl Default for Multisample {
    /// What a buffer configuration leaves: the pattern one-pass, the mask 1.0, the alpha enables off,
    /// and multisample enabled whatever the count.
    fn default() -> Self {
        Multisample { samples: 0, enable: true, alpha_to_mask: false, alpha_to_one: false, mask_on: false, pattern: PATTERN_FIRST, mask_value: 1.0, mask_invert: false }
    }
}

/// What a draw runs under: the count, the sample locations, the coverage word of the mask and the
/// alpha options.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Samples {
    pub count: u8,
    pub at: [[u8; 2]; 8],
    /// The sample mask's coverage word, or all ones with the mask off.
    pub cover: u32,
    pub alpha_to_mask: bool,
    pub alpha_to_one: bool,
    /// The layout's words that hold the samples' colour, depth and stencil: the first word and how many.
    pub colour: (u32, u32),
    pub depth: (u32, u32),
    pub stencil: (u32, u32),
}

impl Multisample {
    /// A new buffer configuration: the count, and everything else as the start-up values.
    pub fn configure(&mut self, samples: u32) {
        *self = Multisample { samples: if matches!(samples, 4 | 8) { samples } else { 0 }, ..Multisample::default() };
    }

    /// The pattern token: false (an invalid enumerant) outside 0x80a1 to 0x80a7.
    pub fn set_pattern(&mut self, pattern: u32) -> bool {
        let ok = (PATTERN_FIRST..PATTERN_FIRST + 7).contains(&pattern);
        if ok {
            self.pattern = pattern;
        }
        ok
    }

    /// The coverage index of the mask value, rounded, and its word with the invert flag.
    pub fn coverage_word(&self) -> u32 {
        let n = (self.mask_value.clamp(0.0, 1.0) * 32.0 + 0.5) as usize;
        let w = if self.samples == 8 { COVER8[n.min(32)] } else { COVER4[(n.min(32) + 1) / 2] };
        if self.mask_invert { !w } else { w }
    }

    /// The draw's sample state, or None when multisampling does not apply: no multisample buffer, or
    /// the enable is off.
    pub fn draw(&self, colour: (u32, u32), depth: (u32, u32), stencil: (u32, u32)) -> Option<Samples> {
        if self.samples == 0 || !self.enable {
            return None;
        }
        let words = if self.samples == 8 { WORDS8 } else { WORDS4 }[(self.pattern - PATTERN_FIRST) as usize];
        let mut at = [[0u8; 2]; 8];
        for (n, loc) in at.iter_mut().enumerate().take(self.samples as usize) {
            let (w, shift) = (words[n / 4], 6 * (n % 4));
            *loc = [(w >> shift & 7) as u8, (w >> (shift + 3) & 7) as u8];
        }
        Some(Samples { count: self.samples as u8, at, cover: if self.mask_on { self.coverage_word() } else { !0 }, alpha_to_mask: self.alpha_to_mask, alpha_to_one: self.alpha_to_one, colour, depth, stencil })
    }

    pub fn visit<S: Stream>(&mut self, s: &mut S) {
        // Every field is named, so a new one fails to compile until it is saved here or left out on purpose.
        let Self { samples: _, enable: _, alpha_to_mask: _, alpha_to_one: _, mask_on: _, pattern: _, mask_value: _, mask_invert: _ } = self;
        s.u32(&mut self.samples);
        for b in [&mut self.enable, &mut self.alpha_to_mask, &mut self.alpha_to_one, &mut self.mask_on, &mut self.mask_invert] {
            s.bool(b);
        }
        s.u32(&mut self.pattern);
        s.f32(&mut self.mask_value);
        // A pattern read from a damaged snapshot would index past the tables.
        if !(PATTERN_FIRST..PATTERN_FIRST + 7).contains(&self.pattern) {
            self.pattern = PATTERN_FIRST;
        }
    }
}

impl Samples {
    /// The samples of pixel column `x` a mask allows: the nibble or byte its column picks.
    pub fn allowed(&self, x: i32) -> u32 {
        if self.count == 8 { self.cover >> (8 * (x & 3)) & 0xff } else { self.cover >> (4 * (x & 7)) & 0xf }
    }

    /// The samples alpha lets through: the first `round(alpha * count)` of them. `alpha` is 13 bits.
    pub fn alpha_samples(&self, alpha: u32) -> u32 {
        let k = (alpha.min(8191) * self.count as u32 + 4095) / 8191;
        (1u32 << k) - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The table's worked words give the table's positions: pattern one-pass at 4 samples is (0,0) (2,4) (4,2) (6,6),
    /// at 8 samples (0,0) (6,7) (5,2) (1,3) then (2,6) (7,4) (3,1) (4,5).
    #[test]
    fn the_location_words_decode_to_the_tables_positions() {
        let mut m = Multisample::default();
        m.configure(4);
        let s = m.draw((1, 3), (4, 3), (7, 1)).expect("multisampled");
        assert_eq!(&s.at[..4], &[[0, 0], [2, 4], [4, 2], [6, 6]]);
        m.configure(8);
        let s = m.draw((1, 3), (4, 3), (7, 1)).expect("multisampled");
        assert_eq!(s.at, [[0, 0], [6, 7], [5, 2], [1, 3], [2, 6], [7, 4], [3, 1], [4, 5]]);
        assert!(m.set_pattern(0x80a7));
        assert_eq!(&m.draw((1, 3), (4, 3), (7, 1)).unwrap().at[..4], &[[0, 0], [1, 7], [5, 5], [7, 2]]);
        assert!(!m.set_pattern(0x80a8));
        m.configure(4);
        assert!(m.set_pattern(0x80a2));
        assert_eq!(&m.draw((1, 3), (4, 3), (7, 1)).unwrap().at[..4], &[[0, 0], [5, 4], [2, 2], [6, 7]]);
    }

    /// A configuration with no multisample buffer, or any other count, draws without samples; the enable turns them off.
    #[test]
    fn only_a_multisampled_layout_with_the_enable_on_has_samples() {
        let mut m = Multisample::default();
        assert!(m.draw((1, 3), (4, 3), (7, 1)).is_none());
        m.configure(2);
        assert_eq!(m.samples, 0);
        m.configure(4);
        assert!(m.draw((1, 3), (4, 3), (7, 1)).is_some());
        m.enable = false;
        assert!(m.draw((1, 3), (4, 3), (7, 1)).is_none());
    }

    /// The mask value indexes the coverage tables (rounded), the invert flag complements, and a pixel's column picks its share.
    #[test]
    fn the_sample_mask_picks_a_coverage_word_and_a_share_per_column() {
        let mut m = Multisample::default();
        m.configure(4);
        m.mask_on = true;
        m.mask_value = 0.0;
        assert_eq!(m.coverage_word(), 0);
        m.mask_value = 0.5;
        assert_eq!(m.coverage_word(), 0x33333333, "index 16 of 32 is the table's 15-16 entry");
        m.mask_invert = true;
        assert_eq!(m.coverage_word(), !0x33333333);
        m.configure(8);
        m.mask_on = true;
        m.mask_value = 1.0;
        assert_eq!(m.coverage_word(), 0xffffffff);
        m.mask_value = 0.5;
        let s = m.draw((1, 3), (4, 3), (7, 1)).unwrap();
        assert_eq!(s.cover, 0x33333333);
        assert_eq!([s.allowed(0), s.allowed(3), s.allowed(4)], [0x33, 0x33, 0x33]);
        // The first sample of an index-1 word only.
        m.mask_value = 1.0 / 32.0;
        let s = m.draw((1, 3), (4, 3), (7, 1)).unwrap();
        assert_eq!([s.allowed(0), s.allowed(1)], [1, 0]);
        assert_eq!(s.alpha_samples(8191), 0xff);
        assert_eq!(s.alpha_samples(0), 0);
        assert_eq!(s.alpha_samples(4096).count_ones(), 4);
    }

    #[test]
    fn the_state_round_trips_through_a_stream() {
        use crate::dev::kona::stream::{Reader, Writer};
        let mut m = Multisample::default();
        m.configure(8);
        m.mask_on = true;
        m.mask_value = 0.25;
        m.mask_invert = true;
        m.set_pattern(0x80a5);
        let mut w = Writer::default();
        m.visit(&mut w);
        let mut n = Multisample::default();
        let mut r = Reader::new(&w.out);
        n.visit(&mut r);
        assert!(r.finish().is_ok());
        assert_eq!(n, m);
    }
}
