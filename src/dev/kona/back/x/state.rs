//! The X server's drawing state: graphics context, position, clip and
//! scissor.
//!
//! The X server sends integer drawing in its own command stream, onto the same
//! raster state the GL path uses. These commands load the state that fills
//! then draw under:
//!
//! - 0x129 (`x_gc`) the thirteen words of the graphics context, decoded by
//!   `gc` when a fill needs them;
//! - 0x12a (`x_position`) the drawable's x, y, width and height, the origin of
//!   rectangle coordinates;
//! - 0x145 (`x_bdata4`) the leading words of a clip list longer than one
//!   rectangle, kept until the 0x12b (`x_clip`) that ends it: a count `n` and
//!   `n` rectangles `x1, y1, x2, y2`;
//! - 0x12d (`x_scissor`) a clip box (x1, y1, width, height);
//! - 0x13c, 0x13d, 0x13e the stipple cache (see `stipple`).
//!
//! Grades. The word layouts are software expectations from the senders. That
//! each clip rectangle is absolute with an exclusive far corner, and that
//! drawing clips to their union, is inferred. A list whose length does not
//! match its count, or a count of 0, leaves drawing clipped only to the
//! screen, since what the pipe makes of one is unknown. The scissor is applied
//! as a second rectangle with the clip and cleared at the next clip load: that
//! the pipe does either is inferred and how long it lasts is unknown. The
//! model keeps at most 64 clip rectangles, its own bound.

use crate::dev::kona::defs::*;
use super::gc::{Gc, Status};

pub const X_GC: u32 = WINDOW_SYSTEM_GRAPHICS_CONTEXT_LOAD;
pub const X_POSITION: u32 = WINDOW_SYSTEM_DRAWABLE_ORIGIN_AND_SIZE_LOAD;
pub const X_CLIP: u32 = WINDOW_SYSTEM_CLIP_LOAD;
pub const X_SCISSOR: u32 = WINDOW_SYSTEM_SCISSOR_LOAD;
pub const X_BDATA4: u32 = WINDOW_SYSTEM_FOUR_WORD_DATA_PRELOAD_BYTE_FORM;

/// Clip rectangles the model keeps.
pub const CLIP_MAX_RECTS: usize = 64;
const CLIP_WORDS: usize = 5;
const BDATA4_WORDS: usize = 4;
const CLIP_LIST_WORDS: usize = 1 + 4 * CLIP_MAX_RECTS;

/// The rectangle drawing lands in, far corner exclusive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bounds {
    pub x1: i64,
    pub y1: i64,
    pub x2: i64,
    pub y2: i64,
}

#[derive(Clone, Debug)]
pub struct XState {
    pub(super) gc: [u32; GC_WORDS],
    position: [u32; 4],
    clip: Vec<[i32; 4]>,
    scissor: Option<[u32; 4]>,
    /// Words of a clip list's leading data tokens, waiting for the clip.
    clip_run: Vec<u32>,
    /// Words of data tokens waiting for the box, rectangle or point that uses
    /// them.
    pub(super) run: Vec<u32>,
    /// Buffers kept between requests, so a fill allocates nothing: the boxes
    /// of a request and the spans of a row.
    pub(super) boxes: Vec<super::fill::Box4>,
    pub(super) spans: Vec<(i64, i64)>,
    pub fills: super::fill::FillCounts,
    pub stipple: super::stipple::Stipple,
    pub(super) pixel: super::pixel::Pixel,
    /// Clip lists refused (a count that does not match, or too long) and runs
    /// dropped by another command.
    pub clips_refused: u64,
    pub runs_dropped: u64,
    /// State commands taken: zero for a copy that was only ever selected.
    pub loads: u64,
}

impl Default for XState {
    fn default() -> Self {
        XState { gc: [0; GC_WORDS], position: [0; 4], clip: Vec::new(), scissor: None, clip_run: Vec::new(), run: Vec::new(), boxes: Vec::new(), spans: Vec::new(), fills: Default::default(), stipple: Default::default(), pixel: Default::default(), clips_refused: 0, runs_dropped: 0, loads: 0 }
    }
}

impl XState {
    /// Whether a pixel transfer is part way, which a snapshot does not carry.
    pub fn busy(&self) -> bool {
        self.pixel.busy()
    }

    /// The graphics context, position, clip, scissor, waiting data, stipple cache and
    /// pixel set-up, as the stream module reads and writes them. Counters, the scratch
    /// buffers and the refused counts are not saved.
    pub fn visit<S: crate::dev::kona::stream::Stream>(&mut self, s: &mut S) {
        // Every field is named, so a new one fails to compile until it is saved here or left out on purpose.
        let Self { gc: _, position: _, clip: _, scissor: _, clip_run: _, run: _, boxes: _, spans: _, fills: _, stipple: _, pixel: _, clips_refused: _, runs_dropped: _, loads: _ } = self;
        s.u32s(&mut self.gc);
        s.u32s(&mut self.position);
        let n = s.len(self.clip.len());
        if s.loading() {
            self.clip.clear();
            self.clip.resize(n, [0; 4]);
        }
        for r in self.clip.iter_mut() {
            s.i32s(r);
        }
        s.opt(&mut self.scissor, |s, v| s.u32s(v));
        s.vec_u32(&mut self.clip_run);
        s.vec_u32(&mut self.run);
        self.stipple.visit(s);
        self.pixel.visit(s);
        s.u64(&mut self.loads);
    }

    #[cfg(test)]
    /// Power-on state.
    pub fn reset(&mut self) {
        *self = XState::default();
    }

    /// The graphics context as last loaded, decoded for a fill.
    pub fn gc(&self) -> Result<Gc, Status> {
        Gc::decode_for_fill(&self.gc)
    }

    #[cfg(test)]
    pub fn gc_words(&self) -> &[u32; GC_WORDS] {
        &self.gc
    }

    /// The drawable's x and y, the origin of rectangle coordinates.
    pub fn origin(&self) -> (i64, i64) {
        (self.position[0] as i32 as i64, self.position[1] as i32 as i64)
    }

    pub fn clip_rects(&self) -> &[[i32; 4]] {
        &self.clip
    }

    pub fn scissor(&self) -> Option<[u32; 4]> {
        self.scissor
    }

    /// The screen of `width` x `height`, narrowed to the bounds of the clip
    /// list's rectangles when one is loaded and to the scissor when one is.
    pub fn bounds(&self, width: u32, height: u32) -> Bounds {
        let mut b = Bounds { x1: 0, y1: 0, x2: width as i64, y2: height as i64 };
        if !self.clip.is_empty() {
            let (mut x1, mut y1, mut x2, mut y2) = (i64::MAX, i64::MAX, i64::MIN, i64::MIN);
            for c in &self.clip {
                x1 = x1.min(c[0] as i64);
                y1 = y1.min(c[1] as i64);
                x2 = x2.max(c[2] as i64);
                y2 = y2.max(c[3] as i64);
            }
            b = Bounds { x1: b.x1.max(x1), y1: b.y1.max(y1), x2: b.x2.min(x2), y2: b.y2.min(y2) };
        }
        if let Some(s) = self.scissor {
            let (x1, y1) = (s[0] as i32 as i64, s[1] as i32 as i64);
            b = Bounds { x1: b.x1.max(x1), y1: b.y1.max(y1), x2: b.x2.min(x1 + s[2] as i64), y2: b.y2.min(y1 + s[3] as i64) };
        }
        b
    }

    #[cfg(test)]
    /// Whether (`x`, `y`), inside `bounds`, is in one of the clip list's
    /// rectangles. One rectangle (or none) is the bounds themselves.
    pub fn clip_contains(&self, x: i64, y: i64) -> bool {
        self.clip.len() <= 1 || self.clip.iter().any(|c| x >= c[0] as i64 && y >= c[1] as i64 && x < c[2] as i64 && y < c[3] as i64)
    }

    /// Every kernel command ends a waiting clip list's words except the two
    /// that belong to it, and a waiting data run's except the tokens that use
    /// or extend it. Called for every kernel-grammar command before it runs.
    ///
    /// The model this was taken from keeps a held run across words another
    /// context sends (the run belongs to the context that began it; inferred
    /// there). This model drops it: the context-ownership rule is not modelled,
    /// to be revisited with the context-switch step.
    pub fn end_runs(&mut self, selector: u32) {
        if selector != X_BDATA4 && selector != X_CLIP && !self.clip_run.is_empty() {
            self.clip_run.clear();
            self.runs_dropped += 1;
        }
        if !super::fill::keeps_run(selector) && !self.run.is_empty() {
            self.run.clear();
            self.runs_dropped += 1;
        }
    }

    /// An X state command. False if `selector` is not one of them.
    pub fn command(&mut self, selector: u32, args: &[u32]) -> bool {
        self.end_runs(selector);
        match selector {
            X_GC => {
                if let Ok(w) = <[u32; GC_WORDS]>::try_from(args) {
                    self.gc = w;
                }
            }
            X_POSITION => {
                if let Ok(w) = <[u32; 4]>::try_from(args) {
                    self.position = w;
                }
            }
            X_BDATA4 => {
                if self.clip_run.len() + BDATA4_WORDS + CLIP_WORDS > CLIP_LIST_WORDS {
                    self.clip_run.clear();
                    self.runs_dropped += 1;
                } else if args.len() == BDATA4_WORDS {
                    self.clip_run.extend_from_slice(args);
                }
            }
            X_CLIP => self.load_clip(args),
            super::stipple::X_LOAD_STIPPLE => self.stipple.load(args),
            super::stipple::X_STORE_STIPPLE => {
                self.stipple.begin_store(args);
            }
            super::stipple::X_ROTATE_STIPPLE => self.stipple.rotate(args),
            X_SCISSOR => {
                if let Ok(w) = <[u32; 4]>::try_from(args) {
                    self.scissor = Some(w);
                }
            }
            _ => return false,
        }
        self.loads += 1;
        true
    }

    fn load_clip(&mut self, args: &[u32]) {
        let mut list = std::mem::take(&mut self.clip_run);
        self.scissor = None;
        if args.len() != CLIP_WORDS {
            self.clips_refused += 1;
            self.clip.clear();
            return;
        }
        list.extend_from_slice(args);
        let count = list[0] as usize;
        if count == 0 || list.len() != 1 + 4 * count || count > CLIP_MAX_RECTS {
            self.clips_refused += 1;
            self.clip.clear();
            return;
        }
        self.clip = list[1..].chunks_exact(4).map(|r| [r[0] as i32, r[1] as i32, r[2] as i32, r[3] as i32]).collect();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clip(x: &mut XState, rects: &[[i32; 4]]) {
        let flat: Vec<u32> = std::iter::once(rects.len() as u32).chain(rects.iter().flatten().map(|v| *v as u32)).collect();
        // The first rectangle's words ride in the clip token; the rest go
        // ahead of it as bdata4 tokens.
        let (lead, last) = flat.split_at(flat.len() - CLIP_WORDS);
        // bdata4 carries words in fours, from the list's first word.
        for c in lead.chunks(BDATA4_WORDS) {
            assert_eq!(c.len(), BDATA4_WORDS, "the test builds lists the senders would");
            x.command(X_BDATA4, c);
        }
        x.command(X_CLIP, last);
    }

    #[test]
    fn a_one_rectangle_clip_narrows_the_screen() {
        let mut x = XState::default();
        clip(&mut x, &[[10, 20, 110, 220]]);
        assert_eq!(x.bounds(1280, 1024), Bounds { x1: 10, y1: 20, x2: 110, y2: 220 });
        assert!(x.clip_contains(0, 0), "one rectangle is the bounds themselves");
    }

    /// Two rectangles: the bounds are their union's box and a pixel must be in
    /// one of them. The first four words ride in a bdata4 token.
    #[test]
    fn a_longer_list_arrives_through_bdata4() {
        let mut x = XState::default();
        // count 2: 1 + 8 = 9 words: bdata4 takes four, a second four, clip the last five? 9 = 4 + 5.
        x.command(X_BDATA4, &[2, 0, 0, 10]);
        x.command(X_CLIP, &[10, 50, 50, 60, 60]);
        assert_eq!(x.clip_rects(), [[0, 0, 10, 10], [50, 50, 60, 60]]);
        assert_eq!(x.bounds(100, 100), Bounds { x1: 0, y1: 0, x2: 60, y2: 60 });
        assert!(x.clip_contains(5, 5) && x.clip_contains(55, 55));
        assert!(!x.clip_contains(30, 30), "between the rectangles");
    }

    /// A count that does not match the words, or zero, leaves the screen.
    #[test]
    fn a_clip_list_that_does_not_match_its_count_clips_to_the_screen() {
        let mut x = XState::default();
        x.command(X_CLIP, &[1, 5, 5, 9, 9]);
        assert_eq!(x.clip_rects().len(), 1);
        x.command(X_CLIP, &[0, 0, 0, 0, 0]);
        assert!(x.clip_rects().is_empty());
        assert_eq!(x.bounds(8, 8), Bounds { x1: 0, y1: 0, x2: 8, y2: 8 });
        x.command(X_CLIP, &[3, 0, 0, 1, 1]);
        assert!(x.clip_rects().is_empty());
        assert_eq!(x.clips_refused, 2);
    }

    #[test]
    fn a_scissor_narrows_the_bounds_until_the_next_clip() {
        let mut x = XState::default();
        x.command(X_SCISSOR, &[100, 200, 30, 40]);
        assert_eq!(x.bounds(1280, 1024), Bounds { x1: 100, y1: 200, x2: 130, y2: 240 });
        x.command(X_CLIP, &[1, 0, 0, 1280, 1024]);
        assert_eq!(x.scissor(), None);
        x.command(X_SCISSOR, &[0, 0, 0, 0]);
        let b = x.bounds(16, 16);
        assert!(b.x1 >= b.x2, "a scissor of no width clips everything");
    }

    /// Another command ends a waiting clip list; the clip and bdata4 do not.
    #[test]
    fn another_command_ends_a_waiting_clip_list() {
        let mut x = XState::default();
        x.command(X_BDATA4, &[2, 0, 0, 10]);
        x.command(X_BDATA4, &[10, 1, 1, 2]);
        assert_eq!(x.runs_dropped, 0);
        x.command(X_POSITION, &[1, 2, 3, 4]);
        assert_eq!(x.runs_dropped, 1);
        x.command(X_CLIP, &[1, 0, 0, 4, 4]);
        assert_eq!(x.clip_rects(), [[0, 0, 4, 4]], "the dropped words are not in the list");
    }

    /// The run is bounded: a list past 64 rectangles is dropped.
    #[test]
    fn a_clip_list_is_bounded() {
        let mut x = XState::default();
        for _ in 0..CLIP_LIST_WORDS / BDATA4_WORDS + 1 {
            x.command(X_BDATA4, &[0; 4]);
        }
        assert_eq!(x.runs_dropped, 1);
    }

    /// A clip token of the wrong length is refused, not indexed.
    #[test]
    fn a_clip_of_the_wrong_length_is_refused() {
        let mut x = XState::default();
        x.command(X_CLIP, &[]);
        x.command(X_CLIP, &[1, 2]);
        assert_eq!(x.clips_refused, 2);
        assert!(x.clip_rects().is_empty());
    }

    #[test]
    fn state_loads_keep_their_words() {
        let mut x = XState::default();
        let gc = [0x5a000, 0x1ffe, 0xa5000, 0x1ffe, 0xf00fff, 0, 0x180c, 0x40001, 1, 6, 5, 1, 1];
        assert!(x.command(X_GC, &gc));
        assert_eq!(x.gc_words(), &gc);
        assert_eq!(x.gc().unwrap().foreground, 0x5a00);
        x.command(X_POSITION, &[(-3i32) as u32, 7, 100, 50]);
        assert_eq!(x.origin(), (-3, 7));
        assert!(!x.command(0x1, &[]));
        // A load of the wrong length is ignored.
        x.command(X_GC, &gc[..5]);
        assert_eq!(x.gc_words(), &gc);
        x.reset();
        assert_eq!(x.origin(), (0, 0));
    }
}
