//! Where a window's planes sit inside a pixel's words.
//!
//! The buffer-configuration command carries ten words and no raster manager
//! count: level, colour size, index size, double buffer, stereo, sample
//! count, depth bits, stencil bits, accumulation bits and auxiliary buffers.
//! From them the planes follow one another in a pixel's record in this order:
//! colour buffers, depth, 8-bit stencil, multisample colour, accumulation.
//! Sizes are in words; `S` is the sample count. The layout is the same for 1,
//! 2 and 4 raster managers.
//!
//! Grades. The order, the word counts per plane and the worked check below
//! are established from the pipe's firmware as software expectation; that the
//! counts are 32-bit words and that word 0 is the overlay level's colour word
//! are inferred. The unit of the window offset and what sets it beyond its
//! start-up value are unknown; here it is a plain word count that moves every
//! plane's base together.
//!
//! Two readings are the model's own, graded inferred. A sample count of 1
//! means multisampling is off, the same as 0 (the documents speak of "no
//! multisampling" and of `S` samples, and 1, 4 or 8 are the published
//! counts). A colour-index visual carries no colour size, so it reads as
//! under 12 bits and takes one word per buffer.
//!
//! What the hardware documents do not cover is refused with a reason, not
//! guessed: a depth width other than 16, 24 or 32 bits once rounded up to a
//! multiple of 8, a sample count other than 0, 1, 4 or 8, a stencil width
//! other than 0, 1 or 8, a colour size above 11 bits other than 12 or 16, a
//! level below 0, an auxiliary buffer, and an accumulation buffer on a 16-bit
//! colour size (its words depend on a firmware mode value nothing defines).

use std::ops::Range;

/// The first colour word of a level-0 layout at window offset 0. The textport,
/// the X path, the GL path's draw target and the display's compose all use this
/// word and the one after it (the second buffer of a double-buffered window), so
/// that what is drawn is what is shown.
pub const MAIN_COLOUR_WORD: u32 = 1;

/// The ten words of the command.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferConfig {
    /// 0 for the main planes, above 0 an overlay.
    pub level: i32,
    /// Bits per colour component; 0 for a colour-index visual.
    pub colour_bits: u32,
    /// Bits of colour index; 0 for an RGB visual.
    pub index_bits: u32,
    pub double_buffer: bool,
    pub stereo: bool,
    pub samples: u32,
    pub depth_bits: u32,
    pub stencil_bits: u32,
    pub accum_bits: u32,
    pub aux: u32,
}

impl BufferConfig {
    pub const WORDS: usize = 10;

    pub fn from_words(w: &[u32; Self::WORDS]) -> BufferConfig {
        BufferConfig {
            level: w[0] as i32,
            colour_bits: w[1],
            index_bits: w[2],
            double_buffer: w[3] != 0,
            stereo: w[4] != 0,
            samples: w[5],
            depth_bits: w[6],
            stencil_bits: w[7],
            accum_bits: w[8],
            aux: w[9],
        }
    }
}

/// Why a configuration has no layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// A level below 0.
    Underlay,
    /// A sample count other than 0, 1, 4 or 8.
    Samples,
    /// A depth width that rounds up to something other than 16, 24 or 32 bits.
    DepthWidth,
    /// A stencil width other than 0, 1 or 8.
    StencilWidth,
    /// An auxiliary buffer: the hardware documents give it no words.
    Auxiliary,
    /// An accumulation buffer on a 16-bit colour size.
    Accumulation16,
    /// A colour size of 13 to 15 bits, or above 16.
    ColourWidth,
    /// More words than the host's budget gives a pixel.
    OverBudget,
}

impl Refusal {
    pub const ALL: [Refusal; 8] = [
        Refusal::Underlay,
        Refusal::Samples,
        Refusal::DepthWidth,
        Refusal::StencilWidth,
        Refusal::Auxiliary,
        Refusal::Accumulation16,
        Refusal::ColourWidth,
        Refusal::OverBudget,
    ];
}

// The most words a pixel's record may take, from the host's own budget
// arithmetic.
//
// The host driver, not the board, picks a window's per-pixel class: it counts
// the managed area's pixels (the width rounded up to a multiple of 160, times
// the height), divides a budget of 32-bit words by that count and maps the
// quotient to a class of 64, 32, 16, 8 or 4 words. The budget is
// `rm_count * 5 * 2^22` words on an InfiniteReality. That the budget counts
// 32-bit words is inferred; the worked case agrees with it: one raster manager
// at 1280 x 1024 gives 20,971,520 words over 1,310,720 pixels, 16 words, the
// published 512 bits per pixel. The back end does not report a class (the guest
// derives it from the raster manager count and the managed size); the bound only
// limits the model: a layout that needs more words than it allows is refused.

/// The widths the host rounds the managed area's width up to.
const WIDTH_UNIT: u32 = 160;

/// The classes, in words per pixel, largest first.
const CLASSES: [u32; 5] = [64, 32, 16, 8, 4];

/// The budget in 32-bit words for `rm_count` raster managers.
fn budget_words(rm_count: u32) -> u64 {
    (5u64 << 22) * rm_count as u64
}

/// The managed area's pixels as the host counts them.
fn managed_pixels(width: u32, height: u32) -> u64 {
    width.div_ceil(WIDTH_UNIT) as u64 * WIDTH_UNIT as u64 * height as u64
}

/// The words a pixel's record may take, or 0 when the budget gives less than the
/// smallest class (which the hardware documents do not cover).
pub fn record_words(rm_count: u32, width: u32, height: u32) -> u32 {
    let pixels = managed_pixels(width, height);
    if pixels == 0 {
        return 0;
    }
    let quotient = budget_words(rm_count) / pixels;
    CLASSES.into_iter().find(|&c| quotient >= c as u64).unwrap_or(0)
}

/// A window's planes as word ranges in a pixel's record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub colour: Range<u32>,
    pub depth: Range<u32>,
    pub stencil: Range<u32>,
    pub multisample: Range<u32>,
    pub accumulation: Range<u32>,
    /// The last word the window uses.
    pub last: u32,
}

impl Layout {
    /// Words a pixel's record needs, counting word 0.
    pub fn words(&self) -> u32 {
        self.last + 1
    }

    /// The layout of `cfg` for a window at `window_offset` words, or why not.
    /// `bound` is the most words a pixel may take (`record_words`):
    /// a layout needing more is refused.
    pub fn new(cfg: &BufferConfig, window_offset: u32, bound: u32) -> Result<Layout, Refusal> {
        if cfg.level < 0 {
            return Err(Refusal::Underlay);
        }
        if cfg.aux != 0 {
            return Err(Refusal::Auxiliary);
        }
        if cfg.colour_bits >= 12 && !matches!(cfg.colour_bits, 12 | 16) {
            return Err(Refusal::ColourWidth);
        }
        let s = cfg.samples;
        if !matches!(s, 0 | 1 | 4 | 8) {
            return Err(Refusal::Samples);
        }
        let multisampled = s > 1;
        let depth = if cfg.depth_bits == 0 { 0 } else { cfg.depth_bits.div_ceil(8) * 8 };
        if !matches!(depth, 0 | 16 | 24 | 32) {
            return Err(Refusal::DepthWidth);
        }
        if !matches!(cfg.stencil_bits, 0 | 1 | 8) {
            return Err(Refusal::StencilWidth);
        }

        // Colour: one word per buffer under 12 bits or at exactly 16, one and a
        // half, rounded up, at 12 (other sizes are refused above). Index
        // visuals carry no component size and so take one word.
        let buffers = 1 + cfg.double_buffer as u32 + 2 * cfg.stereo as u32;
        let wide = cfg.colour_bits == 12;
        let colour_words = if wide { (buffers * 3).div_ceil(2) } else { buffers };
        let colour_start = if cfg.level == 0 { window_offset + MAIN_COLOUR_WORD } else { 0 };
        let colour = colour_start..colour_start + colour_words;

        // Depth: none without depth; one word without multisampling; with it
        // S/2 for 16 bits and 3S/4 for 24 or 32. A 1-bit stencil with no depth
        // takes one word.
        let depth_words = match (depth, multisampled) {
            (0, _) => (cfg.stencil_bits == 1) as u32,
            (_, false) => 1,
            (16, true) => s / 2,
            (_, true) => 3 * s / 4,
        };
        let depth_range = colour.end..colour.end + depth_words;

        // 8-bit stencil: max(1, S/4) words. A 1-bit stencil shares the depth word.
        let stencil_words = if cfg.stencil_bits == 8 { (s / 4).max(1) } else { 0 };
        let stencil = depth_range.end..depth_range.end + stencil_words;

        // Multisample colour: 3S/4 words under 12 bits or at 16, 3S/2 at 12.
        let ms_words = match (multisampled, wide) {
            (false, _) => 0,
            (true, false) => 3 * s / 4,
            (true, true) => 3 * s / 2,
        };
        let multisample = stencil.end..stencil.end + ms_words;

        // Accumulation: none without, none in colour-index mode; else 3 words
        // under 12 bits, 4 at 12; 16 bits is 2 or 4 by a mode the hardware documents do
        // not define.
        let accum_words = if cfg.accum_bits == 0 || cfg.index_bits != 0 {
            0
        } else if cfg.colour_bits == 16 {
            return Err(Refusal::Accumulation16);
        } else if cfg.colour_bits < 12 {
            3
        } else {
            4
        };
        let accumulation = multisample.end..multisample.end + accum_words;

        let last = accumulation.end - 1;
        let layout = Layout { colour, depth: depth_range, stencil, multisample, accumulation, last };
        if layout.words() > bound {
            return Err(Refusal::OverBudget);
        }
        Ok(layout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The worked case: one raster manager, 1280 x 1024, 16 words, 512 bits.
    #[test]
    fn one_raster_manager_at_1280_by_1024_is_sixteen_words() {
        assert_eq!(budget_words(1), 20_971_520);
        assert_eq!(managed_pixels(1280, 1024), 1_310_720);
        assert_eq!(record_words(1, 1280, 1024), 16);
        assert_eq!(record_words(2, 1280, 1024), 32);
        assert_eq!(record_words(4, 1280, 1024), 64);
    }

    /// The width rounds up to a multiple of 160 before the division, and a
    /// quotient between two classes falls to the smaller.
    #[test]
    fn the_width_rounds_up_and_a_quotient_takes_the_lower_class() {
        assert_eq!(managed_pixels(1, 1), 160);
        assert_eq!(managed_pixels(161, 1), 320);
        assert_eq!(managed_pixels(1920, 1080), 1920 * 1080);
        assert_eq!(managed_pixels(1921, 1080), 2080 * 1080);
        // 2048 rounds up to 2080: 20,971,520 / 4,259,840 = 4.9, the 4-word class.
        assert_eq!(record_words(1, 2048, 2048), 4);
        // 1920 x 1200: 20,971,520 / 2,304,000 = 9, so the 8-word class.
        assert_eq!(record_words(1, 1920, 1200), 8);
        assert_eq!(record_words(1, 0, 1024), 0);
    }

    fn cfg(w: [u32; 10]) -> BufferConfig {
        BufferConfig::from_words(&w)
    }

    /// The hardware document's consistency check: an RGB10 double-buffered pixel with 4
    /// samples, 24 bits of depth, 8 of stencil and accumulation takes colour 2
    /// words (1 and 2), depth 3, stencil 1, multisample colour 3 and
    /// accumulation 3, and ends at word 12: 13 words counting word 0.
    #[test]
    fn the_worked_check_ends_at_word_12() {
        let l = Layout::new(&cfg([0, 10, 0, 1, 0, 4, 24, 8, 25, 0]), 0, 16).unwrap();
        assert_eq!(l.colour, 1..3);
        assert_eq!(l.depth, 3..6);
        assert_eq!(l.stencil, 6..7);
        assert_eq!(l.multisample, 7..10);
        assert_eq!(l.accumulation, 10..13);
        assert_eq!(l.last, 12);
        assert_eq!(l.words(), 13);
    }

    /// Colour: one word per buffer under 12 bits and at 16, one and a half
    /// (rounded up) otherwise; buffers are 1, 2, 3 or 4.
    #[test]
    fn colour_words_follow_the_size_and_the_buffer_count() {
        let words = |rgb, double, stereo| Layout::new(&cfg([0, rgb, 0, double, stereo, 0, 0, 0, 0, 0]), 0, 64).unwrap().colour.len();
        assert_eq!([words(8, 0, 0), words(8, 1, 0), words(8, 0, 1), words(8, 1, 1)], [1, 2, 3, 4]);
        assert_eq!([words(16, 0, 0), words(16, 1, 1)], [1, 4]);
        assert_eq!([words(12, 0, 0), words(12, 1, 0), words(12, 0, 1), words(12, 1, 1)], [2, 3, 5, 6]);
        assert_eq!(words(0, 0, 0), 1, "an index visual takes one word a buffer");
    }

    /// A visual of the kind a posted capture gives: 12-bit colour, double
    /// buffered, 23 bits of depth (24 once rounded), 1-bit stencil.
    #[test]
    fn a_one_bit_stencil_shares_the_depth_word() {
        let l = Layout::new(&cfg([0, 12, 0, 1, 0, 0, 23, 1, 0, 0]), 0, 16).unwrap();
        assert_eq!((l.colour.clone(), l.depth.clone(), l.stencil.clone()), (1..4, 4..5, 5..5));
        assert_eq!(l.words(), 5);
        // With no depth, the 1-bit stencil takes the word.
        let l = Layout::new(&cfg([0, 8, 0, 0, 0, 0, 0, 1, 0, 0]), 0, 16).unwrap();
        assert_eq!((l.colour.clone(), l.depth.clone()), (1..2, 2..3));
    }

    #[test]
    fn multisample_words_follow_the_sample_count() {
        let l4 = Layout::new(&cfg([0, 8, 0, 0, 0, 4, 16, 8, 0, 0]), 0, 64).unwrap();
        assert_eq!((l4.depth.len(), l4.stencil.len(), l4.multisample.len()), (2, 1, 3));
        let l8 = Layout::new(&cfg([0, 8, 0, 0, 0, 8, 24, 8, 0, 0]), 0, 64).unwrap();
        assert_eq!((l8.depth.len(), l8.stencil.len(), l8.multisample.len()), (6, 2, 6));
        let l12 = Layout::new(&cfg([0, 12, 0, 0, 0, 4, 24, 0, 0, 0]), 0, 64).unwrap();
        assert_eq!(l12.multisample.len(), 6);
    }

    /// Accumulation: 3 words under 12 bits, 4 at 12, none in index mode.
    #[test]
    fn accumulation_words_follow_the_colour_size() {
        let acc = |rgb, index| Layout::new(&cfg([0, rgb, index, 0, 0, 0, 0, 0, 25, 0]), 0, 64).unwrap().accumulation.len();
        assert_eq!([acc(8, 0), acc(12, 0), acc(0, 8)], [3, 4, 0]);
    }

    /// The window offset moves every plane's base together; above level 0 the
    /// colour planes start at word 0.
    #[test]
    fn the_window_offset_moves_the_planes_together() {
        let c = cfg([0, 8, 0, 1, 0, 0, 24, 8, 0, 0]);
        let a = Layout::new(&c, 0, 64).unwrap();
        let b = Layout::new(&c, 5, 64).unwrap();
        assert_eq!(b.colour.start, a.colour.start + 5);
        assert_eq!((b.depth.start, b.stencil.start, b.last), (a.depth.start + 5, a.stencil.start + 5, a.last + 5));
        let overlay = Layout::new(&cfg([1, 0, 8, 0, 0, 0, 0, 0, 0, 0]), 0, 64).unwrap();
        assert_eq!(overlay.colour, 0..1);
        assert_eq!(overlay.words(), 1);
    }

    #[test]
    fn what_the_documents_do_not_cover_is_refused() {
        let r = |w: [u32; 10]| Layout::new(&cfg(w), 0, 64).unwrap_err();
        assert_eq!(r([u32::MAX, 8, 0, 0, 0, 0, 0, 0, 0, 0]), Refusal::Underlay);
        assert_eq!(r([0, 8, 0, 0, 0, 2, 0, 0, 0, 0]), Refusal::Samples);
        assert_eq!(r([0, 8, 0, 0, 0, 0, 8, 0, 0, 0]), Refusal::DepthWidth);
        assert_eq!(r([0, 8, 0, 0, 0, 0, 0, 4, 0, 0]), Refusal::StencilWidth);
        assert_eq!(r([0, 8, 0, 0, 0, 0, 0, 0, 0, 1]), Refusal::Auxiliary);
        assert_eq!(r([0, 16, 0, 0, 0, 0, 0, 0, 25, 0]), Refusal::Accumulation16);
        for bits in [13, 15, 17, 32] {
            assert_eq!(r([0, bits, 0, 0, 0, 0, 0, 0, 0, 0]), Refusal::ColourWidth, "{bits}");
        }
    }

    /// The host's budget bounds the record: the worked check needs 13 words,
    /// which the 16-word class holds and the 8-word class does not.
    #[test]
    fn a_layout_over_the_budget_is_refused() {
        let c = cfg([0, 10, 0, 1, 0, 4, 24, 8, 25, 0]);
        assert!(Layout::new(&c, 0, 16).is_ok());
        assert_eq!(Layout::new(&c, 0, 12), Err(Refusal::OverBudget));
        assert_eq!(Layout::new(&c, 0, 0), Err(Refusal::OverBudget));
    }
}
