//! The X server's fills: boxes and rectangles, and the data tokens that carry
//! their words.
//!
//! A box (0x154, `x_box`; 0x155 with six) is two words, `x1 << 16 | y1` and
//! `x2 << 16 | y2` with signed halves, far corner exclusive, both corners added
//! to the drawable's position like a rectangle's origin and a point. That the pipe
//! adds the position to a box is inferred: the senders in the recordings zero the
//! position before 18 of the 19 boxes, so they cannot tell a position-relative box
//! from an absolute one, and of the boxes that follow a nonzero origin there are
//! four (three of 0x154 and one of 0x155, against 201 rectangles of 0x152 and 4 of
//! 0x153), which the rectangles' reading and the points' make the likelier. A
//! rectangle (0x152, 0x153) is `x << 16 | y` and `width << 16 | height`, the X
//! protocol's, relative to the drawable the position names. When a request has
//! more boxes than a token holds, the leading words arrive as data tokens
//! (0x141 to 0x14b) and are kept until the box or rectangle token that ends
//! them, which draws them all.
//!
//! Spans (0x156, 0x157, 0x158 with one, two or six) are a count `n`, `n` points
//! `x << 16 | y` and `n` widths: each a one-pixel-high run of `width` pixels from
//! its point, absolute (no position added). Points (0x14c, 0x14d, 0x14e with
//! one, six or twenty-four) are single pixels `x << 16 | y`, added to the
//! drawable's position like a rectangle's origin and a relative point's start, and like boxes
//! they take the words of a waiting data run first. The senders do not zero the position: the
//! X server loads the drawable's screen x and y, and 386 of the 1192 position loads in the desktop
//! recording are nonzero. The recorded position loads are established; that the pipe adds the
//! position to a point is inferred. A span token whose count
//! is not what its length implies is dropped, counted. Every fill is masked by the
//! loaded stipple (see `stipple`).
//!
//! Background points (0x15f, 0x160, 0x161 with one, six or twenty-four) are
//! plain points in the context's background colour (graphics-context words 2
//! and 3). They take a waiting data run first and go through the clip,
//! scissor, write mask and logic operation, but not the stipple: the stipple
//! masks the fills of the shape tokens above, and a background point is
//! drawn whole. The token rows are established; that the colour is the
//! context's background, from words 2 and 3 of the context load, is inferred,
//! and so is the stipple not applying (no source states either). Ported from sgi_kona_x_points in hw/misc/sgi-kona.c in
//! jamesbraid/qemu-sgi-origami.
//!
//! Relative points (0x14f, 0x150, 0x151 with one, six or twenty-four) are
//! points whose words are steps: each is relative to the previous point, the
//! first to the drawable's origin (`x << 16 | y`, signed halves), and the walk
//! runs on through a waiting data run and the burst's words and starts again
//! at the origin with the next token. The token rows are established; the
//! anchor rules are inferred, and no source states them. Like absolute points
//! they go through the stipple.
//!
//! Segments and polylines (0x159 to 0x15e) are in `line`. A waiting data run is
//! dropped, counted, when one of them arrives.
//!
//! X shapes draw the pixels the request names: they do not go through the
//! triangle sample rule (approximated; the architecture says X is a different
//! stream onto the same raster state, not that the board applies that rule to
//! it). A fill is clipped to the screen, the clip list and the scissor, takes
//! the context's colour through its write mask and logic operation, and is
//! refused, counted, if the context is not decoded, targets planes the model
//! does not yet hold, or finds a destination word wider than the target. The
//! planes not held are the overlays, the popup planes and the display-ID plane,
//! which the display step takes, and the two-buffer RGB form (configuration 4,
//! buffer selector 3), which the model's source draws into both buffers while
//! this model has one colour word and refuses it rather than draw half. A
//! request is checked whole before any pixel is written, so a refusal draws
//! nothing. That a data run is the next box's or rectangle's words, and that a
//! run of odd length is dropped, are inferred from the senders.
//!
//! The model carries no per-pixel tag saying what a word means (index or
//! direct); which colour map resolves a word is the display's to settle.
//!
//! Y: the X server's coordinates run down from the top of the screen, the
//! textport's up from the bottom. The store is bottom-origin, so a fill's row is
//! `height - 1 - y` (established by comparing the store rendered from the
//! recorded greeter with its capture, whose toolchest and icons land at the top
//! only this way). Clip, scissor and stipple use the X coordinates.
//!
//! # Cost
//!
//! A request is planned once: the context is decoded, the bounds, clip
//! rectangles and stipple are looked up, and the word array is resolved. Each
//! box is then clipped to rectangles up front and drawn a row of spans at a
//! time, with the stipple stepped by a column counter and the logic operation a
//! constant of the loop (`rows`), so nothing in the pixel loop looks anything up
//! or branches on the state. A solid, unmasked, clip-free fill is a plain run of
//! stores.

use crate::dev::kona::defs::*;
use super::gc::{Gc, Target};
use crate::dev::kona::back::raster::ops::LogicOp;
use super::state::XState;
use crate::dev::kona::back::raster::Surface;
use crate::dev::kona::back::store::{WordArray, DID_PLANE, OVERLAY0_PLANE, OVERLAY1_PLANE, POPUP_PLANE};

/// The two-word data token (tests).
#[cfg(test)]
pub const X_DATA2: u32 = WINDOW_SYSTEM_2_WORD_DATA_PRELOAD;
pub const X_RECT: u32 = FILL_RECTANGLES_ORIGIN_WIDTH_HEIGHT;
pub const X_RECT6: u32 = FILL_RECTANGLES_ORIGIN_WIDTH_HEIGHT_BURST_OF_6;
pub const X_BOX: u32 = FILL_BOXES_TWO_CORNERS;
pub const X_BOX6: u32 = FILL_BOXES_TWO_CORNERS_BURST_OF_6;
pub const X_POINT: u32 = DRAW_ABSOLUTE_POINTS;
pub const X_POINT6: u32 = DRAW_ABSOLUTE_POINTS_BURST_OF_6;
pub const X_POINT24: u32 = DRAW_ABSOLUTE_POINTS_BURST_OF_24;
pub const X_RPOINT: u32 = DRAW_RELATIVE_POINTS;
pub const X_RPOINT6: u32 = DRAW_RELATIVE_POINTS_BURST_OF_6;
pub const X_RPOINT24: u32 = DRAW_RELATIVE_POINTS_BURST_OF_24;
pub const X_BGPOINT: u32 = DRAW_POINTS_IN_BACKGROUND_COLOUR;
pub const X_BGPOINT6: u32 = DRAW_POINTS_IN_BACKGROUND_COLOUR_BURST_OF_6;
pub const X_BGPOINT24: u32 = DRAW_POINTS_IN_BACKGROUND_COLOUR_BURST_OF_24;
pub const X_SPANS: u32 = FILL_SPANS;
pub const X_SPANS2: u32 = FILL_SPANS_BURST_OF_2;
pub const X_SPANS6: u32 = FILL_SPANS_BURST_OF_6;
/// The data tokens, 0x141 to 0x14b: how many words each carries.
const DATA: [(u32, usize); 8] = [(WINDOW_SYSTEM_1_WORD_DATA_PRELOAD, 1), (WINDOW_SYSTEM_2_WORD_DATA_PRELOAD, 2), (WINDOW_SYSTEM_3_WORD_DATA_PRELOAD, 3), (WINDOW_SYSTEM_4_WORD_DATA_PRELOAD, 4), (WINDOW_SYSTEM_5_WORD_DATA_PRELOAD, 5), (WINDOW_SYSTEM_6_WORD_DATA_PRELOAD, 6), (WINDOW_SYSTEM_8_WORD_DATA_PRELOAD, 8), (WINDOW_SYSTEM_10_WORD_DATA_PRELOAD, 10)];
/// The most words a data run may hold.
const RUN_WORDS: usize = 32;

/// What fills did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FillCounts {
    /// Fills that drew at least a pixel's worth of box.
    pub drawn: u64,
    /// Requests refused because the context did not decode (or was opaque with
    /// no stipple), because it targets planes not held, or because a
    /// destination word did not fit the target.
    pub bad_context: u64,
    pub other_target: u64,
    pub bad_destination: u64,
    /// Tokens whose word count is not what they carry: a span token whose
    /// count disagrees with its length, a data token of the wrong size.
    pub malformed: u64,
}

/// Whether `selector` is a data token, and its word count.
pub fn data_words(selector: u32) -> Option<usize> {
    DATA.iter().find(|(s, _)| *s == selector).map(|&(_, n)| n)
}

/// Whether `selector` is a token that writes pixels when it arrives: a box,
/// rectangle, point (absolute, relative or background) or span. The data tokens only collect words for one.
pub fn paints(selector: u32) -> bool {
    super::line::is_line(selector) || matches!(selector, X_RECT | X_RECT6 | X_BOX | X_BOX6 | X_POINT | X_POINT6 | X_POINT24 | X_BGPOINT | X_BGPOINT6 | X_BGPOINT24 | X_RPOINT | X_RPOINT6 | X_RPOINT24 | X_SPANS | X_SPANS2 | X_SPANS6)
}

/// Whether `selector` is a token `draw_command` takes.
pub fn is_draw(selector: u32) -> bool {
    keeps_run(selector) || super::line::is_line(selector) || matches!(selector, X_SPANS | X_SPANS2 | X_SPANS6)
}

/// A token that keeps a waiting data run: the data tokens, boxes, rectangles,
/// points and the four polylines, whose points a run precedes.
pub fn keeps_run(selector: u32) -> bool {
    data_words(selector).is_some() || super::line::is_polyline(selector) || matches!(selector, X_RECT | X_RECT6 | X_BOX | X_BOX6 | X_POINT | X_POINT6 | X_POINT24 | X_BGPOINT | X_BGPOINT6 | X_BGPOINT24 | X_RPOINT | X_RPOINT6 | X_RPOINT24)
}

/// Why a request was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Refused {
    Context,
    Target,
    Destination,
}

/// A box, far corner exclusive, in the X server's coordinates.
pub(super) type Box4 = (i64, i64, i64, i64);

/// The stipple pattern of a request: its rows, size and the point it is
/// measured from.
#[derive(Clone, Copy, Debug)]
pub struct StipplePattern {
    pub rows: [u32; 32],
    pub width: i64,
    pub height: i64,
    pub origin: (i64, i64),
}

/// Everything about a request, looked up once.
pub(super) struct Plan<'a> {
    pub(super) gc: Gc,
    pub(super) array: WordArray<'a>,
    pub(super) pitch: usize,
    pub(super) height: i64,
    pub(super) bounds: super::state::Bounds,
    /// The clip rectangles when there are several (one is the bounds).
    clips: &'a [[i32; 4]],
    pub(super) stipple: Option<StipplePattern>,
    pub(super) needs: bool,
}

impl<'a> Plan<'a> {
    pub(super) fn new(x: &'a XState, surface: &Surface<'a>) -> Result<Option<Plan<'a>>, Refused> {
        let stipple = x.stipple_pattern();
        // An opaque or tiled context carries both colours and needs a stipple.
        let gc = match Gc::decode_for_fill(&x.gc) {
            Ok(gc) if stipple.is_some() || !gc.opaque_or_tiled => gc,
            _ => return Err(Refused::Context),
        };
        Plan::with_gc(x, surface, gc, stipple)
    }

    /// The plan for background points: the context's background colour as
    /// the colour, and no stipple.
    fn background(x: &'a XState, surface: &Surface<'a>) -> Result<Option<Plan<'a>>, Refused> {
        let Ok(mut gc) = Gc::decode_for_fill(&x.gc) else { return Err(Refused::Context) };
        gc.foreground = gc.background;
        Plan::with_gc(x, surface, gc, None)
    }

    /// The plan for a request drawn under `gc`, which the caller decoded:
    /// refused if it targets planes the model does not hold. Inlined, so the
    /// fills' own plan is the code it was before this had a second caller.
    #[inline(always)]
    pub(super) fn with_gc(x: &'a XState, surface: &Surface<'a>, gc: Gc, stipple: Option<StipplePattern>) -> Result<Option<Plan<'a>>, Refused> {
        let two_buffers = gc.target == Target::MainRgb8 && gc.raw[GC_FRAMEBUFFER_CONFIG] == 4 && gc.first_selector == 3;
        if two_buffers || gc.target == Target::OverlayBoth {
            return Err(Refused::Target);
        }
        if gc.is_noop() {
            return Ok(None);
        }
        let (width, height) = surface.size();
        // The colour word, or the plane the target names.
        let array = match gc.target {
            Target::MainCi8 | Target::MainIndex16 | Target::MainRgb8 => surface.array(),
            Target::Did => surface.sibling(DID_PLANE).array(),
            Target::Overlay0 => surface.sibling(OVERLAY0_PLANE).array(),
            Target::Overlay1 => surface.sibling(OVERLAY1_PLANE).array(),
            Target::Popup => surface.sibling(POPUP_PLANE).array(),
            Target::OverlayBoth => unreachable!("refused above"),
        };
        let needs = gc.needs_destination();
        let clips = if x.clip_rects().len() > 1 { x.clip_rects() } else { &[] };
        Ok(Some(Plan { gc, pitch: array.pitch(), array, height: height as i64, bounds: x.bounds(width, height), clips, stipple, needs }))
    }

    /// `b` clipped to the bounds; None if nothing is left.
    pub(super) fn clip(&self, b: Box4) -> Option<Box4> {
        let (x1, y1, x2, y2) = (b.0.max(self.bounds.x1), b.1.max(self.bounds.y1), b.2.min(self.bounds.x2), b.3.min(self.bounds.y2));
        (x1 < x2 && y1 < y2).then_some((x1, y1, x2, y2))
    }

    /// The runs of row `y` between `x1` and `x2` that the clip list lets
    /// through, sorted and not overlapping (a pixel in two rectangles is
    /// painted once: a logic operation twice would undo itself).
    pub(super) fn row_spans(&self, y: i64, x1: i64, x2: i64, out: &mut Vec<(i64, i64)>) {
        out.clear();
        if self.clips.is_empty() {
            out.push((x1, x2));
            return;
        }
        for c in self.clips.iter().filter(|c| (c[1] as i64..c[3] as i64).contains(&y)) {
            let (a, b) = (x1.max(c[0] as i64), x2.min(c[2] as i64));
            if a < b {
                out.push((a, b));
            }
        }
        out.sort_unstable();
        let mut merged = 0;
        for i in 1..out.len() {
            if out[i].0 <= out[merged].1 {
                out[merged].1 = out[merged].1.max(out[i].1);
            } else {
                merged += 1;
                out[merged] = out[i];
            }
        }
        out.truncate(if out.is_empty() { 0 } else { merged + 1 });
    }

    /// The stipple's row for `y`, and the column its first pixel at `x` falls
    /// in: bits of the row, the pattern width and the starting column.
    pub(super) fn stipple_start(&self, p: &StipplePattern, y: i64, x: i64) -> (u32, i64, i64) {
        let row = p.rows[(y - p.origin.1).rem_euclid(p.height) as usize];
        (row, p.width, (x - p.origin.0).rem_euclid(p.width))
    }

    /// Whether every pixel this box would change holds a word that fits the
    /// target: only a fill that reads its destination can fail.
    fn destination_fits(&self, b: Box4, spans: &mut Vec<(i64, i64)>) -> bool {
        let Some((x1, y1, x2, y2)) = self.clip(b) else { return true };
        for y in y1..y2 {
            self.row_spans(y, x1, x2, spans);
            let base = (self.height - 1 - y) as usize * self.pitch;
            for &(a, c) in spans.iter() {
                let (mut bits, mut width, mut col) = (0, 1, 0);
                if let Some(p) = &self.stipple {
                    (bits, width, col) = self.stipple_start(p, y, a);
                }
                for x in a..c {
                    let set = self.stipple.is_none() || bits & (0x8000_0000 >> col) != 0;
                    col += 1;
                    if col == width {
                        col = 0;
                    }
                    if !set && !self.gc.opaque_or_tiled {
                        continue;
                    }
                    // SAFETY: the box is inside the bounds, which are inside the managed area.
                    if unsafe { self.array.load(base + x as usize) } > self.gc.width_mask {
                        return false;
                    }
                }
            }
        }
        true
    }
}

/// The word written for `source` over `destination`, the operation a
/// constant of the caller.
#[inline(always)]
pub(super) fn apply<const GX: u32>(gc: &Gc, source: u32, destination: u32) -> u32 {
    ((LogicOp::from_index(GX).apply(source, destination) & gc.mask) | (destination & !gc.mask)) & gc.width_mask
}

/// Draw the clipped box `b`, a row of spans at a time. `NEEDS`: the write
/// reads the destination; `STIP`: a stipple masks it; `GX`: the logic
/// operation, used only when the destination is read.
fn rows<const GX: u32, const NEEDS: bool, const STIP: bool>(p: &Plan<'_>, b: Box4, spans: &mut Vec<(i64, i64)>) {
    let (x1, y1, x2, y2) = b;
    let gc = &p.gc;
    // With no destination read the colour alone makes each word, so the two
    // results are made once.
    let (fg, bg) = if NEEDS { (gc.foreground, gc.background) } else { (gc.apply(gc.foreground, 0), gc.apply(gc.background, 0)) };
    for y in y1..y2 {
        p.row_spans(y, x1, x2, spans);
        let base = (p.height - 1 - y) as usize * p.pitch;
        for &(a, c) in spans.iter() {
            if !NEEDS && !STIP {
                for x in a..c {
                    // SAFETY: the box is inside the bounds, which are inside the managed area.
                    unsafe { p.array.store(base + x as usize, fg) };
                }
                continue;
            }
            let (mut bits, mut width, mut col) = (0, 1, 0);
            if STIP {
                if let Some(s) = &p.stipple {
                    (bits, width, col) = p.stipple_start(s, y, a);
                }
            }
            for x in a..c {
                let set = !STIP || bits & (0x8000_0000 >> col) != 0;
                col += 1;
                if col == width {
                    col = 0;
                }
                if !set && !gc.opaque_or_tiled {
                    continue;
                }
                let i = base + x as usize;
                let value = if NEEDS {
                    // SAFETY: as above.
                    let old = unsafe { p.array.load(i) };
                    apply::<GX>(gc, if set { fg } else { bg }, old)
                } else if set {
                    fg
                } else {
                    bg
                };
                // SAFETY: as above.
                unsafe { p.array.store(i, value) };
            }
        }
    }
}

/// Draw `b` with the loop that fits the request: chosen once, not per pixel.
fn draw(p: &Plan<'_>, b: Box4, spans: &mut Vec<(i64, i64)>) {
    macro_rules! by_op {
        ($stip:literal, $($gx:literal),*) => {
            match p.gc.gx as u32 {
                $($gx => rows::<$gx, true, $stip>(p, b, spans),)*
                _ => {
                    debug_assert!(false, "a decoded context's logic operation is 0 to 15");
                    rows::<3, true, $stip>(p, b, spans)
                }
            }
        };
    }
    let stip = p.stipple.is_some();
    match (p.needs, stip) {
        (false, false) => rows::<0, false, false>(p, b, spans),
        (false, true) => rows::<0, false, true>(p, b, spans),
        (true, false) => by_op!(false, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
        (true, true) => by_op!(true, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
    }
}

/// What a request did: boxes drawn, and why it was refused if it was.
struct Outcome {
    drawn: u64,
    refused: Option<Refused>,
}

impl XState {
    /// The stipple pattern a fill uses, if one with a pattern is loaded.
    pub(super) fn stipple_pattern(&self) -> Option<StipplePattern> {
        let (rows, width, height, origin) = self.stipple.loaded_pattern()?;
        let (ox, oy) = self.origin();
        Some(StipplePattern { rows, width, height, origin: (ox + origin.0, oy + origin.1) })
    }

    /// Draw `boxes` under the context: planned once, every box checked before
    /// any is drawn when the write reads its destination.
    fn request(&self, surface: &Surface<'_>, boxes: impl Iterator<Item = Box4> + Clone, background: bool, spans: &mut Vec<(i64, i64)>) -> Outcome {
        let plan = match if background { Plan::background(self, surface) } else { Plan::new(self, surface) } {
            Ok(Some(plan)) => plan,
            Ok(None) => return Outcome { drawn: 0, refused: None },
            Err(why) => return Outcome { drawn: 0, refused: Some(why) },
        };
        if plan.needs && !boxes.clone().all(|b| plan.destination_fits(b, spans)) {
            return Outcome { drawn: 0, refused: Some(Refused::Destination) };
        }
        let mut drawn = 0;
        for b in boxes {
            if let Some(b) = plan.clip(b) {
                draw(&plan, b, spans);
                drawn += 1;
            }
        }
        Outcome { drawn, refused: None }
    }

    /// Run `boxes` and count the outcome. The spans buffer is kept between
    /// requests, so a request allocates nothing.
    pub(super) fn run_request(&mut self, surface: &Surface<'_>, boxes: impl Iterator<Item = Box4> + Clone, background: bool) {
        let mut spans = std::mem::take(&mut self.spans);
        let out = self.request(surface, boxes, background, &mut spans);
        self.spans = spans;
        self.fills.drawn += out.drawn;
        match out.refused {
            Some(Refused::Context) => self.fills.bad_context += 1,
            Some(Refused::Target) => self.fills.other_target += 1,
            Some(Refused::Destination) => self.fills.bad_destination += 1,
            None => {}
        }
    }

    /// A data token, which only collects words for the paint that follows. False if
    /// `selector` is not one.
    pub fn collect_command(&mut self, selector: u32, args: &[u32]) -> bool {
        if let Some(n) = data_words(selector) {
            if args.len() != n {
                self.fills.malformed += 1;
            } else if self.run.len() + n > RUN_WORDS {
                self.run.clear();
                self.runs_dropped += 1;
            } else {
                self.run.extend_from_slice(args);
            }
            return true;
        }
        false
    }

    /// A data token, a box, a rectangle, points or spans. False if
    /// `selector` is none of them.
    pub fn draw_command(&mut self, surface: &Surface<'_>, selector: u32, args: &[u32]) -> bool {
        if self.collect_command(selector, args) {
            return true;
        }
        match selector {
            X_BOX | X_BOX6 => self.boxes(surface, false, args),
            X_RECT | X_RECT6 => self.boxes(surface, true, args),
            X_POINT | X_POINT6 | X_POINT24 => self.points(surface, args, false),
            X_BGPOINT | X_BGPOINT6 | X_BGPOINT24 => self.points(surface, args, true),
            X_RPOINT | X_RPOINT6 | X_RPOINT24 => self.relative_points(surface, args),
            X_SPANS | X_SPANS2 | X_SPANS6 => self.spans(surface, args),
            t if super::line::is_line(t) => self.lines(surface, t, args),
            _ => return false,
        }
        true
    }

    /// The waiting run's words, then the token's, as boxes or rectangles.
    fn boxes(&mut self, surface: &Surface<'_>, rects: bool, args: &[u32]) {
        if self.run.len() % 2 != 0 {
            self.run.clear();
            self.runs_dropped += 1;
        }
        let (ox, oy) = self.origin();
        let words = self.run.iter().chain(args).copied();
        let pairs = words.clone().step_by(2).zip(words.skip(1).step_by(2));
        let boxes = pairs.map(move |(w0, w1)| {
            let (x, y) = ((w0 >> 16) as u16 as i16 as i64, w0 as u16 as i16 as i64);
            if rects {
                (x + ox, y + oy, x + ox + (w1 >> 16) as i64, y + oy + (w1 & 0xffff) as i64)
            } else {
                (x + ox, y + oy, ((w1 >> 16) as u16 as i16 as i64) + ox, (w1 as u16 as i16 as i64) + oy)
            }
        });
        // The boxes go in a buffer the state keeps; the run is cleared after,
        // keeping its capacity.
        let mut buf = std::mem::take(&mut self.boxes);
        buf.clear();
        buf.extend(boxes);
        self.run.clear();
        self.run_request(surface, buf.iter().copied(), false);
        self.boxes = buf;
    }

    /// Points: the waiting run's words, then the token's, one pixel each, in
    /// the foreground colour or (`background`) the background colour, each
    /// at the drawable's position plus its own.
    fn points(&mut self, surface: &Surface<'_>, args: &[u32], background: bool) {
        let (ox, oy) = self.origin();
        let mut buf = std::mem::take(&mut self.boxes);
        buf.clear();
        buf.extend(self.run.iter().chain(args).map(|&w| {
            let (x, y) = (((w >> 16) as u16 as i16 as i64) + ox, (w as u16 as i16 as i64) + oy);
            (x, y, x + 1, y + 1)
        }));
        self.run.clear();
        self.run_request(surface, buf.iter().copied(), background);
        self.boxes = buf;
    }

    /// Relative points: each word is a step from the previous point, the
    /// first from the drawable's origin, and the walk runs through the
    /// waiting run's words and then the token's. The next token starts again
    /// at the origin.
    fn relative_points(&mut self, surface: &Surface<'_>, args: &[u32]) {
        let (mut x, mut y) = self.origin();
        let mut buf = std::mem::take(&mut self.boxes);
        buf.clear();
        buf.extend(self.run.iter().chain(args).map(|&w| {
            x += (w >> 16) as u16 as i16 as i64;
            y += w as u16 as i16 as i64;
            (x, y, x + 1, y + 1)
        }));
        self.run.clear();
        self.run_request(surface, buf.iter().copied(), false);
        self.boxes = buf;
    }

    /// Spans: a count, that many points, that many widths.
    fn spans(&mut self, surface: &Surface<'_>, args: &[u32]) {
        let n = args.len().saturating_sub(1) / 2;
        if args.is_empty() || args[0] as usize != n {
            self.fills.malformed += 1;
            return;
        }
        let mut buf = std::mem::take(&mut self.boxes);
        buf.clear();
        buf.extend((0..n).map(|i| {
            let w = args[1 + i];
            let (x, y) = ((w >> 16) as u16 as i16 as i64, w as u16 as i16 as i64);
            (x, y, x + args[1 + n + i] as i64, y + 1)
        }));
        self.run_request(surface, buf.iter().copied(), false);
        self.boxes = buf;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dev::kona::back::store::PixelStore;

    /// An index-16 context: foreground 0x5a00, background 0xa500, full mask.
    const GC: [u32; 13] = [0x5a000, 0x1ffe, 0xa5000, 0x1ffe, 0xf00fff, 0, 0x180c, 0x40001, 1, 6, 5, 1, 1];

    fn rig() -> (XState, PixelStore) {
        let mut x = XState::default();
        x.command(0x129, &GC);
        (x, PixelStore::new(32, 32))
    }

    /// The word at X coordinates (`x`, `y`): the store is bottom-origin.
    fn at(s: &PixelStore, x: u32, y: u32) -> u32 {
        s.read(x, 31 - y, 1)
    }

    fn lit(s: &PixelStore) -> usize {
        (0..32).flat_map(|y| (0..32).map(move |x| (x, y))).filter(|&(x, y)| s.read(x, y, 1) != 0).count()
    }

    fn pack(a: i32, b: i32) -> u32 {
        (a as u32) << 16 | (b as u32 & 0xffff)
    }

    /// A box has an exclusive far corner: it draws the pixels the request names,
    /// not by the textport's fill rule.
    #[test]
    fn a_box_draws_the_pixels_it_names() {
        let (mut x, s) = rig();
        let d = Surface::new(&s, 1);
        assert!(x.draw_command(&d, X_BOX, &[pack(2, 3), pack(5, 7)]));
        assert_eq!(lit(&s), 3 * 4);
        assert_eq!(at(&s, 2, 3), 0x5a00);
        assert_eq!(at(&s, 4, 6), 0x5a00);
        assert_eq!(at(&s, 5, 3), 0, "the far corner is exclusive");
        assert_eq!(x.fills.drawn, 1);
    }

    /// A box takes the drawable's position at both corners, as a rectangle's origin and a point do,
    /// in the token and in its six-box burst, and a position loaded back to zero takes it away.
    #[test]
    fn the_position_is_added_to_both_corners_of_a_box() {
        let mut x = XState::default();
        x.command(0x129, &GC);
        // Tall and wide enough for position 100,50 and a box beyond it.
        let s = PixelStore::new(128, 64);
        let d = Surface::new(&s, 1);
        let at = |x: u32, y: u32| s.read(x, 63 - y, 1);
        let lit = || (0..64).flat_map(|y| (0..128).map(move |x| (x, y))).filter(|&(x, y)| s.read(x, y, 1) != 0).count();
        x.command(0x12a, &[100, 50, 100, 100]);
        x.draw_command(&d, X_BOX, &[pack(2, 3), pack(5, 7)]);
        assert_eq!(lit(), 3 * 4);
        assert_eq!((at(102, 53), at(104, 56), at(105, 53), at(102, 57), at(2, 3)), (0x5a00, 0x5a00, 0, 0, 0));
        // Six boxes of one pixel, two columns apart.
        let mut six = [0u32; 12];
        for i in 0..6 {
            six[2 * i] = pack(10 + 2 * i as i32, 0);
            six[2 * i + 1] = pack(11 + 2 * i as i32, 1);
        }
        x.draw_command(&d, X_BOX6, &six);
        assert_eq!(lit(), 3 * 4 + 6);
        assert!((0..6).all(|i| at(110 + 2 * i, 50) == 0x5a00) && at(10, 0) == 0);
        // Back at position zero a box is where its words put it.
        x.command(0x12a, &[0, 0, 100, 100]);
        x.draw_command(&d, X_BOX, &[pack(20, 20), pack(22, 22)]);
        assert_eq!((lit(), at(20, 20), at(21, 21), at(22, 22)), (3 * 4 + 6 + 4, 0x5a00, 0x5a00, 0));
    }

    /// A rectangle is relative to the drawable's position, with a size.
    #[test]
    fn a_rectangle_is_relative_to_the_position() {
        let (mut x, s) = rig();
        x.command(0x12a, &[10, 20, 100, 100]);
        let d = Surface::new(&s, 1);
        x.draw_command(&d, X_RECT, &[pack(1, 2), 4 << 16 | 3]);
        assert_eq!(lit(&s), 12);
        assert_eq!(at(&s, 11, 22), 0x5a00);
        assert_eq!(at(&s, 14, 24), 0x5a00);
        assert_eq!(at(&s, 15, 22), 0);
    }

    /// Data tokens carry the leading boxes; the box token draws them all, and
    /// an odd run is dropped.
    #[test]
    fn data_tokens_feed_the_next_box() {
        let (mut x, s) = rig();
        let d = Surface::new(&s, 1);
        x.draw_command(&d, 0x144, &[pack(0, 0), pack(2, 2), pack(10, 10), pack(12, 12)]);
        x.draw_command(&d, X_BOX, &[pack(20, 20), pack(22, 22)]);
        assert_eq!(lit(&s), 4 + 4 + 4);
        assert_eq!(x.fills.drawn, 3);
        // Three words are not boxes: the run is dropped, the token's own box draws.
        x.draw_command(&d, 0x143, &[pack(1, 1), pack(2, 2), pack(3, 3)]);
        x.draw_command(&d, X_BOX, &[pack(25, 25), pack(26, 26)]);
        assert_eq!(x.runs_dropped, 1);
        assert_eq!(at(&s, 1, 1), 0x5a00, "the first run's box 0,0-2,2 covers it");
        assert_eq!(at(&s, 25, 25), 0x5a00);
        assert_eq!(at(&s, 3, 3), 0);
        // A run that grows past its bound is dropped.
        for _ in 0..4 {
            x.draw_command(&d, 0x149, &[0; 8]);
        }
        x.draw_command(&d, 0x141, &[0]);
        assert_eq!(x.runs_dropped, 2);
    }

    #[test]
    fn a_fill_clips_to_the_screen_the_clip_and_the_scissor() {
        let (mut x, s) = rig();
        let d = Surface::new(&s, 1);
        x.draw_command(&d, X_BOX, &[pack(-10, -10), pack(4, 4)]);
        assert_eq!(lit(&s), 16);
        x.command(0x12b, &[1, 10, 10, 20, 20]);
        x.draw_command(&d, X_BOX, &[pack(0, 0), pack(31, 31)]);
        assert_eq!(at(&s, 10, 10), 0x5a00);
        assert_eq!(at(&s, 9, 10), 0);
        assert_eq!(at(&s, 20, 15), 0);
        x.command(0x12d, &[12, 12, 2, 2]);
        let before = lit(&s);
        x.draw_command(&d, X_BOX, &[pack(0, 0), pack(1, 1)]);
        assert_eq!(lit(&s), before, "nothing in the scissor");
    }

    /// With several clip rectangles a pixel between them stays clear.
    #[test]
    fn a_clip_list_masks_the_fill() {
        let (mut x, s) = rig();
        x.command(0x145, &[2, 0, 0, 8]);
        x.command(0x12b, &[8, 20, 20, 30, 30]);
        let d = Surface::new(&s, 1);
        x.draw_command(&d, X_BOX, &[pack(0, 0), pack(31, 31)]);
        assert_eq!(lit(&s), 64 + 100);
        assert_eq!(at(&s, 15, 15), 0);
    }

    /// A write mask keeps the destination's other bits.
    #[test]
    fn a_write_mask_and_operation_apply_to_the_destination() {
        let (mut x, s) = rig();
        let d = Surface::new(&s, 1);
        x.draw_command(&d, X_BOX, &[pack(0, 0), pack(2, 2)]);
        let mut gc = GC;
        gc[0] = 0xff000; // foreground 0xff00
        gc[4] = 0xf0000f; // mask: only the low nibble of the high byte? index mask 0xf000
        x.command(0x129, &gc);
        x.draw_command(&d, X_BOX, &[pack(0, 0), pack(1, 1)]);
        let g = x.gc().unwrap();
        assert_eq!(at(&s, 0, 0), g.apply(g.foreground, 0x5a00));
        assert_eq!(at(&s, 1, 1), 0x5a00);
        // The no-op operation changes nothing.
        let mut gc = GC;
        gc[9] = 10;
        x.command(0x129, &gc);
        x.draw_command(&d, X_BOX, &[pack(0, 0), pack(2, 2)]);
        assert_eq!(at(&s, 0, 0), g.apply(g.foreground, 0x5a00));
    }

    /// What cannot be drawn is refused whole and counted: an undecoded
    /// context, a target held elsewhere, a destination wider than the target,
    /// an opaque context with no stipple.
    #[test]
    fn what_cannot_be_drawn_is_refused_and_counted() {
        let (mut x, s) = rig();
        let d = Surface::new(&s, 1);
        let mut bad = GC;
        bad[9] = 1;
        x.command(0x129, &bad);
        x.draw_command(&d, X_BOX, &[pack(0, 0), pack(4, 4)]);
        assert_eq!((x.fills.bad_context, lit(&s)), (1, 0));
        // The two-buffer RGB form draws two colour buffers at once, which the model has one word for.
        let two_buffers = [0, 0, 0, 0, 0xff_ffff, 0xff_ffff, 4, 3, 1, 6, 0, 0, 0];
        x.command(0x129, &two_buffers);
        x.draw_command(&d, X_BOX, &[pack(0, 0), pack(4, 4)]);
        assert_eq!((x.fills.other_target, lit(&s)), (1, 0));
        // An opaque stippled context with no stipple loaded.
        let mut opaque = GC;
        opaque[4] |= 0x8000_0000;
        x.command(0x129, &opaque);
        x.draw_command(&d, X_BOX, &[pack(0, 0), pack(4, 4)]);
        assert_eq!((x.fills.bad_context, lit(&s)), (2, 0));
        // A destination wider than the 16-bit target (a direct pixel there),
        // with a mask that needs the destination.
        let mut masked = GC;
        masked[4] = 0xf0053f;
        x.command(0x129, &masked);
        s.write(10, 31 - 10, 1, 0x12_3456);
        x.draw_command(&d, X_BOX, &[pack(8, 8), pack(12, 12)]);
        assert_eq!((x.fills.bad_destination, at(&s, 8, 8)), (1, 0), "nothing of the request drew");
        // One bad box in a request stops the whole request: the first box is
        // fine, the second reaches the bad destination.
        x.draw_command(&d, 0x142, &[pack(0, 0), pack(2, 2)]);
        x.draw_command(&d, X_BOX, &[pack(9, 9), pack(11, 11)]);
        assert_eq!(x.fills.bad_destination, 2);
        assert_eq!(at(&s, 0, 0), 0, "the first box of the refused request did not draw");
    }

    /// The display-ID plane is drawn by its own context into a plane of its
    /// own: only the bits the mask names change, and the colour word does not.
    #[test]
    fn the_display_id_plane_takes_its_own_fills() {
        use crate::dev::kona::back::store::DID_PLANE;
        let (mut x, s) = rig();
        let d = Surface::new(&s, 1);
        // The main ID field, bits 7:2, under mask 0xfc, written as ID 0x2b.
        let id = |value: u32| [0, value << 9 & 0x3ffde00, 0, 0, 0, 0xfc000, 8, 1, 0, 6, 0, 1, 1];
        s.write(1, 31 - 1, DID_PLANE, 0x1f03);
        x.command(0x129, &id(0));
        x.draw_command(&d, X_BOX, &[pack(0, 0), pack(2, 2)]);
        assert_eq!(s.read(1, 31 - 1, DID_PLANE), 0x1f03, "ID 0 into bits 7:2 leaves them zero");
        s.write(1, 31 - 1, DID_PLANE, 0x1fff);
        x.draw_command(&d, X_BOX, &[pack(0, 0), pack(2, 2)]);
        assert_eq!(s.read(1, 31 - 1, DID_PLANE), 0x1f03, "the ID bits cleared, the overlay field kept");
        assert_eq!(lit(&s), 0, "the colour word is untouched");
        assert_eq!(x.fills.other_target, 0);
    }

    /// Spans are absolute one-pixel-high runs; a count that disagrees with the
    /// length drops the token.
    #[test]
    fn spans_draw_runs_and_a_bad_count_is_dropped() {
        let (mut x, s) = rig();
        x.command(0x12a, &[100, 100, 5, 5]);
        let d = Surface::new(&s, 1);
        x.draw_command(&d, X_SPANS2, &[2, pack(1, 1), pack(4, 3), 5, 2]);
        assert_eq!(lit(&s), 7);
        assert_eq!(at(&s, 1, 1), 0x5a00);
        assert_eq!(at(&s, 5, 1), 0x5a00);
        assert_eq!(at(&s, 6, 1), 0);
        assert_eq!(at(&s, 5, 3), 0x5a00);
        x.draw_command(&d, X_SPANS, &[2, pack(0, 9), 3]);
        assert_eq!((x.fills.malformed, lit(&s)), (1, 7));
        // A zero width draws nothing; spans end a waiting data run.
        x.draw_command(&d, 0x141, &[pack(20, 20)]);
        // (the back end offers every kernel command to `command` first, which
        // ends the run)
        assert!(!x.command(X_SPANS, &[]));
        x.draw_command(&d, X_SPANS, &[1, pack(0, 9), 0]);
        assert_eq!((x.runs_dropped, lit(&s)), (1, 7));
    }

    /// Points take a waiting run's words first, and are single pixels.
    #[test]
    fn points_draw_single_pixels_including_a_waiting_run() {
        let (mut x, s) = rig();
        let d = Surface::new(&s, 1);
        x.draw_command(&d, 0x142, &[pack(1, 1), pack(2, 2)]);
        x.draw_command(&d, X_POINT, &[pack(3, 3)]);
        assert_eq!(lit(&s), 3);
        x.draw_command(&d, X_POINT6, &[pack(10, 1), pack(11, 1), pack(12, 1), pack(13, 1), pack(14, 1), pack(-1, 1)]);
        assert_eq!(lit(&s), 3 + 5);
    }

    /// An absolute point is at the drawable's position plus its own coordinates, in the run and in the token,
    /// foreground or background.
    #[test]
    fn the_position_is_added_to_absolute_points() {
        let (mut x, s) = rig();
        x.command(0x12a, &[10, 20, 100, 100]);
        let d = Surface::new(&s, 1);
        x.draw_command(&d, 0x141, &[pack(1, 1)]);
        x.draw_command(&d, X_POINT, &[pack(2, 2)]);
        assert_eq!((lit(&s), at(&s, 11, 21), at(&s, 12, 22), at(&s, 1, 1)), (2, 0x5a00, 0x5a00, 0));
        x.draw_command(&d, X_BGPOINT, &[pack(-1, -1)]);
        assert_eq!(at(&s, 9, 19), 0xa500);
    }

    /// Relative points step from the previous point, the first from the
    /// drawable's origin, through a waiting run and a burst, and start again
    /// at the origin with the next token.
    #[test]
    fn relative_points_step_from_the_previous_point() {
        let (mut x, s) = rig();
        x.command(0x12a, &[10, 20, 100, 100]);
        let d = Surface::new(&s, 1);
        x.draw_command(&d, X_RPOINT, &[pack(1, 2)]);
        assert_eq!((lit(&s), at(&s, 11, 22)), (1, 0x5a00));
        // The run's words come first, and the burst continues the walk.
        x.draw_command(&d, 0x142, &[pack(0, 0), pack(2, 0)]);
        x.draw_command(&d, X_RPOINT6, &[pack(0, 1), pack(-1, 0), pack(0, 0), pack(5, 1), pack(0, 0), pack(0, 0)]);
        let want = [(10, 20), (12, 20), (12, 21), (11, 21), (11, 21), (16, 22), (16, 22), (16, 22)];
        for &(px, py) in &want {
            assert_eq!(at(&s, px, py), 0x5a00, "({px}, {py})");
        }
        assert_eq!(lit(&s), 1 + 5, "a pixel named twice is one pixel");
        // A step of -1 in both axes from the origin goes up and left of it.
        x.draw_command(&d, X_RPOINT, &[pack(-1, -1)]);
        assert_eq!(at(&s, 9, 19), 0x5a00);
    }

    /// Background points are points in the context's background colour, and
    /// take a waiting run's words like the foreground ones.
    #[test]
    fn background_points_draw_in_the_background_colour() {
        let (mut x, s) = rig();
        let d = Surface::new(&s, 1);
        x.draw_command(&d, 0x141, &[pack(1, 1)]);
        x.draw_command(&d, X_BGPOINT, &[pack(2, 2)]);
        assert_eq!((lit(&s), at(&s, 1, 1), at(&s, 2, 2)), (2, 0xa500, 0xa500));
        x.draw_command(&d, X_BGPOINT6, &[pack(10, 1), pack(11, 1), pack(12, 1), pack(13, 1), pack(14, 1), pack(-1, 1)]);
        assert_eq!((lit(&s), at(&s, 14, 1)), (2 + 5, 0xa500));
        let words: Vec<u32> = (0..24).map(|i| pack(i, 20)).collect();
        x.draw_command(&d, X_BGPOINT24, &words);
        assert_eq!((lit(&s), at(&s, 23, 20)), (2 + 5 + 24, 0xa500));
        // The foreground token still draws the foreground.
        x.draw_command(&d, X_POINT, &[pack(5, 5)]);
        assert_eq!(at(&s, 5, 5), 0x5a00);
    }

    /// A background point is not masked by the stipple: it draws where the
    /// pattern is clear, and a plain point in the same place is masked.
    #[test]
    fn background_points_ignore_the_stipple() {
        let (mut x, s) = rig();
        let d = Surface::new(&s, 1);
        // An 8x1 pattern of alternating columns, the even ones set.
        x.command(0x13d, &[1, 8, 1]);
        x.stipple.rows(&[0xaaaa_aaaa]);
        x.command(0x13c, &[1]);
        x.draw_command(&d, X_BGPOINT6, &[pack(0, 0), pack(1, 0), pack(2, 0), pack(3, 0), pack(4, 0), pack(5, 0)]);
        assert_eq!((0..6).map(|c| at(&s, c, 0)).collect::<Vec<_>>(), [0xa500; 6]);
        x.draw_command(&d, X_POINT6, &[pack(0, 1), pack(1, 1), pack(2, 1), pack(3, 1), pack(4, 1), pack(5, 1)]);
        assert_eq!((0..6).map(|c| at(&s, c, 1)).collect::<Vec<_>>(), [0x5a00, 0, 0x5a00, 0, 0x5a00, 0]);
    }

    /// The two-buffer RGB form (configuration 4, buffer selector 3) is
    /// refused, not drawn into one buffer.
    #[test]
    fn the_two_buffer_rgb_form_is_refused() {
        let (mut x, s) = rig();
        let d = Surface::new(&s, 1);
        let rgb = [0, 0, 0, 0, 0xffffff, 0xffffff, 4, 3, 1, 6, 0, 0, 0];
        x.command(0x129, &rgb);
        assert!(Gc::decode_for_fill(&rgb).is_ok());
        x.draw_command(&d, X_BOX, &[pack(0, 0), pack(4, 4)]);
        assert_eq!((x.fills.other_target, lit(&s)), (1, 0));
        // The same form with selector 1 draws.
        let mut one = rgb;
        one[7] = 1;
        x.command(0x129, &one);
        x.draw_command(&d, X_BOX, &[pack(0, 0), pack(4, 4)]);
        assert_eq!(x.fills.other_target, 1);
    }

    /// A pixel in two overlapping clip rectangles is painted once: under xor a
    /// second painting would undo the first.
    #[test]
    fn overlapping_clip_rectangles_paint_a_pixel_once() {
        let (mut x, s) = rig();
        let mut gc = GC;
        gc[9] = 12; // xor
        x.command(0x129, &gc);
        x.command(0x145, &[2, 0, 0, 20]);
        x.command(0x12b, &[10, 10, 0, 30, 10]);
        let d = Surface::new(&s, 1);
        x.draw_command(&d, X_BOX, &[pack(0, 0), pack(32, 10)]);
        let v = x.gc().unwrap().foreground;
        for px in [0, 15, 19, 25, 29] {
            assert_eq!(at(&s, px, 5), v, "x {px}");
        }
        assert_eq!(at(&s, 30, 5), 0);
    }

    /// A data token with the wrong word count is counted, not taken.
    #[test]
    fn a_data_token_of_the_wrong_size_is_counted() {
        let (mut x, s) = rig();
        let d = Surface::new(&s, 1);
        x.draw_command(&d, 0x142, &[1]);
        x.draw_command(&d, 0x141, &[1, 2]);
        assert_eq!((x.fills.malformed, x.runs_dropped), (2, 0));
        x.draw_command(&d, X_BOX, &[pack(0, 0), pack(1, 1)]);
        assert_eq!(x.fills.drawn, 1, "the box alone");
    }

    /// The optimised fills against a per-pixel reference that works from what
    /// this test wrote, not from the code under test: its own table of the 16
    /// logic operations, the raw words it chose for the colours and the mask, the
    /// clip rectangles, scissor, stipple rows, origin and position it sent, and
    /// the clip rule stated once below. Nothing on the expected-value path calls
    /// `XState`, `Gc` or `Stipple`; only the loads and the draw go through them.
    ///
    /// The generator keeps most requests on the screen (clip rectangles
    /// overlapping it, few empty masks), and the test counts, from the
    /// reference, the requests that change a pixel and asserts at least half of
    /// them do, and that every operation that writes changed something.
    ///
    /// The clip rule, as the model this was taken from has it (inferred there,
    /// not a hardware measurement): drawing is limited to the screen; to the box
    /// from the smallest x1 and y1 to the largest x2 and y2 of the clip
    /// rectangles as sent (a reversed rectangle contributes its corners as given,
    /// so it can make that box empty); when more than one rectangle was sent, to
    /// pixels inside at least one (x1 <= x < x2 and y1 <= y < y2, so a reversed
    /// rectangle holds none); and to the scissor box (x1, y1, width, height) when
    /// one was loaded after the clip. A stipple bit is the row at `(y - oy) mod
    /// height`, bit `(x - ox) mod width` from the most significant, where the
    /// origin is the position plus the entry's origin.
    #[test]
    fn fills_match_a_per_pixel_reference() {
        const W: i64 = 64;
        const H: i64 = 48;
        // The 16 operations in the GL numbering, written out here.
        let ops: [fn(u32, u32) -> u32; 16] = [
            |_, _| 0,
            |s, d| s & d,
            |s, d| s & !d,
            |s, _| s,
            |s, d| !s & d,
            |_, d| d,
            |s, d| s ^ d,
            |s, d| s | d,
            |s, d| !(s | d),
            |s, d| !(s ^ d),
            |_, d| !d,
            |s, d| s | !d,
            |s, _| !s,
            |s, d| !s | d,
            |s, d| !(s & d),
            |_, _| u32::MAX,
        ];
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = move |n: u32| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as u32) % n
        };
        let store = PixelStore::new(W as u32, H as u32);
        let surface = Surface::new(&store, 1);
        // The reference's own copy of the screen, by X row (y down), and the store
        // is bottom-origin: store row = H - 1 - y.
        let mut model = vec![0u32; (W * H) as usize];
        for y in 0..H {
            for x in 0..W {
                let v = rnd(0x1_0000);
                model[(y * W + x) as usize] = v;
                store.write(x as u32, (H - 1 - y) as u32, 1, v);
            }
        }
        // A rectangle mostly over the screen; now and then reversed.
        let rect = |rnd: &mut dyn FnMut(u32) -> u32| -> [i64; 4] {
            let (x1, y1) = (rnd(70) as i64 - 8, rnd(54) as i64 - 6);
            let (w, h) = (6 + rnd(60) as i64, 6 + rnd(44) as i64);
            if rnd(12) == 0 { [x1 + w, y1 + h, x1, y1] } else { [x1, y1, x1 + w, y1 + h] }
        };
        let (mut changing, mut by_op) = (0u32, [0u32; 16]);
        const REQUESTS: u32 = 600;
        for request in 0..REQUESTS {
            let mut x = XState::default();
            // Stipple: none, or one of several sizes at a random origin.
            let stippled = rnd(3) != 0;
            let (mut rows, mut sw, mut sh, mut so) = (Vec::new(), 1i64, 1i64, (0i64, 0i64));
            if stippled {
                let sizes = [1u32, 2, 3, 5, 8, 17, 32];
                let (w, h) = (sizes[rnd(7) as usize], sizes[rnd(7) as usize]);
                x.command(0x13d, &[1, w, h]);
                rows = (0..h).map(|_| rnd(u32::MAX)).collect();
                x.stipple.rows(&rows);
                so = (rnd(80) as i64 - 40, rnd(80) as i64 - 40);
                x.command(0x13e, &[1, so.0 as i32 as u32, so.1 as i32 as u32]);
                x.command(0x13c, &[1]);
                (sw, sh) = (w as i64, h as i64);
            }
            let pos = (rnd(40) as i64 - 20, rnd(40) as i64 - 20);
            x.command(0x12a, &[pos.0 as i32 as u32, pos.1 as i32 as u32, 64, 48]);
            // The context: the words are built from the values chosen here.
            let gx = rnd(16);
            let mask = match rnd(5) {
                0 | 1 => 0xffff,
                _ => 1 + rnd(0xffff),
            };
            let (fg, bg) = (rnd(0x1_0000), rnd(0x1_0000));
            let opaque = stippled && rnd(2) == 0;
            let word = |v: u32| (v & 0xfff0) << 4 | (v & 0xf) << 3;
            let mut gc = [0, 0x1ffe, 0, 0x1ffe, 0, 0, 0x180c, 0x40001, 1, 2 * gx, 5, 1, 1];
            gc[0] = word(fg);
            gc[2] = word(bg);
            gc[4] = (mask >> 4) & 0xfff | (mask & 0xf) << 20 | if opaque { 0x8000_0000 } else { 0 };
            x.command(0x129, &gc);
            // The clip: none, one rectangle, or several that overlap.
            let mut clips: Vec<[i64; 4]> = Vec::new();
            match rnd(3) {
                0 => {}
                1 => {
                    clips.push(rect(&mut rnd));
                    let r = clips[0];
                    x.command(0x12b, &[1, r[0] as i32 as u32, r[1] as i32 as u32, r[2] as i32 as u32, r[3] as i32 as u32]);
                }
                _ => {
                    let n = 2 + rnd(3) as usize;
                    let mut flat = vec![n as u32];
                    for _ in 0..n {
                        let r = rect(&mut rnd);
                        clips.push(r);
                        flat.extend(r.iter().map(|&v| v as i32 as u32));
                    }
                    let (lead, last) = flat.split_at(flat.len() - 5);
                    for c in lead.chunks(4) {
                        x.command(0x145, c);
                    }
                    x.command(0x12b, last);
                }
            }
            let scissor = (rnd(4) == 0).then(|| [rnd(50) as i64 - 10, rnd(36) as i64 - 8, 10 + rnd(60) as i64, 8 + rnd(40) as i64]);
            if let Some(s) = scissor {
                x.command(0x12d, &[s[0] as i32 as u32, s[1] as i32 as u32, s[2] as u32, s[3] as u32]);
            }
            // `b` is where the box lands: it is sent relative to the position, which both corners take.
            let b = rect(&mut rnd);
            x.draw_command(&surface, X_BOX, &[pack((b[0] - pos.0) as i32, (b[1] - pos.1) as i32), pack((b[2] - pos.0) as i32, (b[3] - pos.1) as i32)]);

            // The reference.
            let mut limit = [0i64, 0, W, H];
            if !clips.is_empty() {
                limit = [
                    limit[0].max(clips.iter().map(|c| c[0]).min().unwrap()),
                    limit[1].max(clips.iter().map(|c| c[1]).min().unwrap()),
                    limit[2].min(clips.iter().map(|c| c[2]).max().unwrap()),
                    limit[3].min(clips.iter().map(|c| c[3]).max().unwrap()),
                ];
            }
            if let Some(s) = scissor {
                limit = [limit[0].max(s[0]), limit[1].max(s[1]), limit[2].min(s[0] + s[2]), limit[3].min(s[1] + s[3])];
            }
            let writes = mask != 0 && gx != 5;
            let mut changed = false;
            if writes {
                for py in 0..H {
                    for px in 0..W {
                        let inside = |r: &[i64; 4]| px >= r[0] && px < r[2] && py >= r[1] && py < r[3];
                        if !inside(&b) || !inside(&limit) || (clips.len() > 1 && !clips.iter().any(inside)) {
                            continue;
                        }
                        let set = !stippled || rows[(py - (pos.1 + so.1)).rem_euclid(sh) as usize] & (0x8000_0000u32 >> (px - (pos.0 + so.0)).rem_euclid(sw)) != 0;
                        if !set && !opaque {
                            continue;
                        }
                        let at = (py * W + px) as usize;
                        let old = model[at];
                        let new = ((ops[gx as usize](if set { fg } else { bg }, old) & mask) | (old & !mask)) & 0xffff;
                        changed |= new != old;
                        model[at] = new;
                    }
                }
            }
            for py in 0..H {
                for px in 0..W {
                    let (a, r) = (store.read(px as u32, (H - 1 - py) as u32, 1), model[(py * W + px) as usize]);
                    assert_eq!(a, r, "request {request}: pixel ({px}, {py}) after gx {gx}, mask {mask:#x}, box {b:?}, clips {clips:?}, scissor {scissor:?}, stipple {stippled} (opaque {opaque})");
                }
            }
            if changed {
                changing += 1;
                by_op[gx as usize] += 1;
            }
        }
        assert!(changing >= REQUESTS / 2, "only {changing} of {REQUESTS} requests changed a pixel");
        for (gx, n) in by_op.iter().enumerate() {
            // The no-op writes nothing; the others must have changed pixels.
            assert!(gx == 5 || *n >= 5, "operation {gx} changed pixels in only {n} requests");
        }
    }

    #[test]
    fn the_run_rule_names_the_tokens_that_keep_a_run() {
        for t in [0x141, 0x142, 0x143, 0x144, 0x146, 0x147, 0x149, 0x14b, X_BOX, X_RECT, X_BOX6, X_RECT6] {
            assert!(keeps_run(t), "{t:#x}");
        }
        for t in [X_POINT, X_POINT6, X_POINT24, X_BGPOINT, X_BGPOINT6, X_BGPOINT24] {
            assert!(keeps_run(t) && paints(t) && is_draw(t), "{t:#x}");
        }
        assert!(!keeps_run(0x129) && !keeps_run(0x145) && !keeps_run(crate::dev::kona::back::x::line::X_SEGMENT));
        for t in [crate::dev::kona::back::x::line::X_LINE, crate::dev::kona::back::x::line::X_LINE6, crate::dev::kona::back::x::line::X_RLINE, crate::dev::kona::back::x::line::X_RLINE6] {
            assert!(keeps_run(t) && paints(t) && is_draw(t), "{t:#x}");
        }
        assert_eq!(data_words(0x14b), Some(10));
        assert_eq!(data_words(0x145), None);
    }
}
