//! The X server's pixel transfers: setup, write, read, copy and finish.
//!
//! - 0x136 (`x_pixel_setup`) seven words that the next transfer reads: mode,
//!   source packing, destination packing, shift, width, height and pad. The
//!   packing and shift words are table entries whose meaning to the pipe is
//!   unknown; this model reads the forms the X server is known to send and
//!   refuses the rest, counted.
//! - 0x139 (`x_pixel_write`) the position of an image whose rows follow, either
//!   behind the command as a data run the back end sizes, or, with the DMA bit
//!   of the mode word set, in a DMA packet that arrives next. Rows run down from
//!   the position, each `ceil((pad + bits * width) / 32)` words, pixels packed
//!   most significant bit first (inferred from the senders' packing tables).
//! - 0x13a (`x_pixel_finish`) ends the write.
//! - 0x137 (`x_pixel_copy`) moves a rectangle inside the store.
//! - 0x138 (`x_pixel_read`) returns a rectangle on the return page.
//!
//! Grades. The word layouts, the pixel widths a source packing names and the
//! forms of context each transfer accepts are software expectations from the
//! senders, ported from hw/display/sgi/x-gc.c and hw/misc/sgi-kona.c in
//! jamesbraid/qemu-sgi-origami. What a clear bit of a one-bit image paints,
//! and the conversion of a packed pixel to the plane's word, are inferred
//! there and here.
//!
//! Approximated: the model applies each row as it arrives, where the model
//! this was taken from stages a batch of writes and commits it at the finish.
//! The two agree because a logic operation acts on each bit alone, so applying
//! operations in order gives the staged result. The one difference is that
//! this model checks the destination before any pixel is written only for
//! the write in hand, not for the whole batch.
//!
//! Refused, counted and drawn nowhere: the window-copy pair (source and
//! destination packing 3), a write or copy aimed at the overlay, popup or
//! display-ID planes (fills draw those; transfers do not) and the two-buffer RGB
//! form, and a frame-buffer word read, which tells an index from a direct colour by a tag
//! this store does not keep. A read that is refused still completes, with the
//! whole usable page zeroed: the X server waits on the page's first word with
//! no timeout, and what the pipe answers to a read it cannot perform is
//! unknown (approximated, as the model this was taken from does).
//!
//! # Cost
//!
//! A write is planned once, as a fill is. The pixel loop is chosen at compile
//! time by the packing kind (packed, one-bit, opaque one-bit), whether a
//! stipple masks it and the logic operation, and does an integer window of two
//! words, a shift and a store per pixel; the row's words are copied once with
//! a zero after them, so the window never reads past the row. What it still
//! does per pixel is two bounds-checked word loads and, for a one-bit image, a
//! skip of each clear bit, which is data and cannot be hoisted. A copy checks
//! its source (and, for a write that reads its destination, its destination)
//! in loops of their own and then stores under a logic operation that is a
//! constant of the row loop. A read hoists the packing out of the pixel loop:
//! one loop for each of one, two and four bytes. Only `WordArray::store` writes
//! a pixel.

use crate::dev::kona::defs::*;
use super::gc::{Gc, Status, Target};
use super::state::XState;
use super::fill::{apply, Plan, Refused};
use crate::dev::kona::back::raster::Surface;

pub const X_PIXEL_SETUP: u32 = WINDOW_SYSTEM_PIXEL_PATH_SET_UP;
pub const X_PIXEL_COPY: u32 = WINDOW_SYSTEM_PIXEL_COPY;
pub const X_PIXEL_READ: u32 = WINDOW_SYSTEM_PIXEL_READ;
pub const X_PIXEL_WRITE: u32 = WINDOW_SYSTEM_PIXEL_WRITE;
pub const X_PIXEL_FINISH: u32 = WINDOW_SYSTEM_PIXEL_PATH_FINISH;

/// The setup words in send order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Setup {
    pub mode: u32,
    pub src: u32,
    pub dst: u32,
    pub shift: u32,
    pub width: u32,
    pub height: u32,
    pub pad: u32,
}

impl Setup {
    /// The mode word's bit that sends a write's rows by DMA.
    pub const MODE_DMA: u32 = PIXEL_SETUP_DIRECT_WRITE_FLAG;

    fn from_words(w: &[u32; PIXEL_SETUP_WORDS]) -> Setup {
        Setup {
            mode: w[PIXEL_SETUP_TRANSFER_KIND],
            src: w[PIXEL_SETUP_OUTPUT_MODULE],
            dst: w[PIXEL_SETUP_CONVERSION_OR_PACKER],
            shift: w[PIXEL_SETUP_ENGINE_VALUE],
            width: w[PIXEL_SETUP_WIDTH],
            height: w[PIXEL_SETUP_HEIGHT],
            pad: w[PIXEL_SETUP_DISTRIBUTOR_VALUE],
        }
    }

    /// Bits a source pixel takes, or None for a packing this model does not
    /// know. 4 and 5 are the one-bit forms.
    fn bits(&self) -> Option<u32> {
        match self.src {
            0 => Some(8),
            1 => Some(16),
            2 => Some(32),
            4 | 5 => Some(1),
            _ => None,
        }
    }

    /// Words a row takes, and words an image takes; None past what the model
    /// accepts or for an unknown packing.
    fn words(&self) -> Option<(usize, usize)> {
        let bits = self.bits()? as u64;
        let stride = (self.pad as u64 + bits * self.width as u64).div_ceil(32);
        let all = stride.checked_mul(self.height as u64)?;
        (all <= MAX_WRITE_WORDS as u64).then_some((stride as usize, all as usize))
    }
}

/// The most words one write may carry: 64 MiB, the model's own bound.
pub const MAX_WRITE_WORDS: usize = 16 << 20;

/// The largest reply a read can post: the return page less its first two words.
const READ_LIMIT: usize = 0x4000 - 8;

/// A write that has announced its rows and waits for them.
#[derive(Clone, Copy, Debug)]
struct Pending {
    x: i32,
    y: i32,
    words: usize,
    dma: bool,
}

/// What the pixel path did, for `kona status`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PixelCounts {
    pub setups: u64,
    pub writes: u64,
    pub dma_writes: u64,
    pub copies: u64,
    pub reads: u64,
    pub finishes: u64,
    /// Writes and copies refused: the context or setup was not understood,
    /// the target has no store, a destination word did not fit, or the
    /// window-copy pair.
    pub bad_context: u64,
    pub other_target: u64,
    pub bad_destination: u64,
    pub window_copies: u64,
    /// Reads answered with zeros, and of them frame-buffer word reads.
    pub reads_refused: u64,
    pub fb_word_reads: u64,
    /// Writes whose rows did not arrive whole, rows that arrived with none
    /// waiting, and writes that ended with rows still owed.
    pub short: u64,
    pub unexpected: u64,
    /// Read brackets that finished refused or without a read, and those whose
    /// host buffer could not be written or whose reason page was missing, which
    /// post no completion.
    pub bracket_refused: u64,
    pub bracket_unposted: u64,
}

/// How a transfer ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verdict {
    /// Drawn, or nothing to draw.
    Done,
    /// Drawn and a copy: counted as one.
    Copied,
    Context,
    Target,
    Destination,
}

/// A transfer's state, one per context.
#[derive(Clone, Debug, Default)]
pub struct Pixel {
    setup: Option<Setup>,
    pending: Option<Pending>,
    read_dma: Option<ReadDma>,
    pub counts: PixelCounts,
    /// Scratch for a copy's source row, kept between copies.
    row: Vec<u32>,
}

impl Pixel {
    /// Whether a write waits for its rows or a read bracket is open, neither of which a
    /// snapshot carries.
    pub fn busy(&self) -> bool {
        self.pending.is_some() || self.read_dma.is_some()
    }

    /// The set-up, as the stream module reads and writes it (the counters are not saved).
    pub fn visit<S: crate::dev::kona::stream::Stream>(&mut self, s: &mut S) {
        // Every field is named, so a new one fails to compile until it is saved here or left out on purpose.
        let Self { setup: _, pending: _, read_dma: _, counts: _, row: _ } = self;
        s.opt(&mut self.setup, |s, v| {
            for w in [&mut v.mode, &mut v.src, &mut v.dst, &mut v.shift, &mut v.width, &mut v.height, &mut v.pad] {
                s.u32(w);
            }
        });
        if s.loading() {
            self.pending = None;
            self.read_dma = None;
        }
    }

    fn count(&mut self, v: Verdict) {
        match v {
            Verdict::Done => {}
            Verdict::Copied => self.counts.copies += 1,
            Verdict::Context => self.counts.bad_context += 1,
            Verdict::Target => self.counts.other_target += 1,
            Verdict::Destination => self.counts.bad_destination += 1,
        }
    }
}

/// A read bracket the kernel opened with select-feed start: where the host
/// buffer is and how its lines lie, and, once the read inside it has run, the
/// bytes waiting for the bracket's finish.
#[derive(Clone, Debug)]
struct ReadDma {
    descriptor: [u32; 6],
    refused: bool,
    header: [u32; 7],
    data: Option<Vec<u8>>,
    /// The bytes are a texel read-back's, whose descriptor length is the packed image's own (`feed_texels`).
    texels: bool,
}

impl ReadDma {
    /// The host buffer: device address, line pitch and line count, for the
    /// descriptor forms the X server sends (address tagged 0x9000 in the low
    /// word, a byte count that is a whole number of lines, a non-negative gap).
    fn buffer(&self) -> Option<(u64, u64, u32)> {
        let a = &self.descriptor;
        if a[1] & !0xfff != 0x9000 || a[2] == 0 || a[2] > 64 << 20 || a[3] == 0 || a[3] > a[2] || a[2] % a[3] != 0 || (a[4] as i32) < 0 || a[5] != 0 {
            return None;
        }
        let io = (a[0] as u64) << 12 | (a[1] & 0xfff) as u64;
        let lines = a[2] / a[3];
        let pitch = a[3] as u64 + a[4] as u64;
        let span = (lines as u64 - 1) * pitch + a[3] as u64;
        (span <= 64 << 20).then_some((io, pitch, lines))
    }

    /// Whether the descriptor fits the read `header` names: the rows laid
    /// end to end, or one line, with the bytes a row of the read takes.
    fn fits(&self, h: &[u32; 7]) -> bool {
        if !matches!(h[4], 1 | 2 | 4) || h[5] != 0 || h[6] != 0 {
            return false;
        }
        let row = h[2] as u64 * h[4] as u64;
        if row == 0 || (h[2] as i32) <= 0 || (h[3] as i32) <= 0 || h[2] > 8192 || h[3] > 8192 || self.buffer().is_none() {
            return false;
        }
        let total = row * h[3] as u64;
        self.descriptor[2] as u64 == total && ((self.descriptor[3] as u64 == total && self.descriptor[4] == 0) || self.descriptor[3] as u64 == row)
    }
}

/// What a finished read bracket writes to the host and posts.
#[derive(Debug, PartialEq, Eq)]
pub struct FeedDone {
    /// Device address of the first line, the line pitch and the bytes of a line.
    pub io: u64,
    pub pitch: u64,
    pub line: usize,
    pub data: Vec<u8>,
    /// The second word of the finish, which the completion repeats.
    pub echo: u32,
}

/// What a pixel command asks of the back end next.
#[derive(Debug, PartialEq, Eq)]
pub enum Next {
    Nothing,
    /// The write's rows follow as this many data words.
    Words(u32),
    /// The write's rows arrive in a DMA packet.
    Dma,
    /// Post these bytes on the return page.
    Return(Vec<u8>),
}

/// Whether `selector` is one of the pixel transfer's commands.
pub fn is_pixel(selector: u32) -> bool {
    matches!(selector, X_PIXEL_SETUP | X_PIXEL_COPY | X_PIXEL_READ | X_PIXEL_WRITE | X_PIXEL_FINISH)
}

/// The kinds of source the pixel loop tells apart.
const PACKED: u8 = 0;
const BITMAP: u8 = 1;
const OPAQUE: u8 = 2;

/// How a source pixel becomes the word a plane takes.
#[derive(Clone, Copy)]
struct Conv {
    fg: u32,
    bg: u32,
    shift: u32,
    mask: u32,
    bits: u32,
    pad: u32,
}

/// Where an image lands and how its rows lie.
#[derive(Clone, Copy)]
struct Geo {
    x: i64,
    y: i64,
    width: i64,
    height: i64,
    stride: usize,
}

/// The context the write draws under, from the context words and the setup:
/// the forms the X server sends for images and glyphs. Ported from the
/// write-decode of the model named above; every branch keeps its conditions.
fn decode_write(raw: &[u32; GC_WORDS], s: &Setup) -> Result<Gc, Status> {
    let mut raw = *raw;
    let bad = Status::Deferred;
    if s.mode & !Setup::MODE_DMA != 0 {
        return Err(bad);
    }
    let bitmap = s.src == 4 || s.src == 5;
    match s.src {
        1 => {
            if s.dst != 1 || s.shift != 4 || raw[GC_PIXEL_PACK_CODE] != 5 {
                return Err(bad);
            }
            let gc = Gc::decode_main_index(&raw)?;
            return if gc.opaque_or_tiled { Err(bad) } else { Ok(gc) };
        }
        2 => {
            if s.dst != 2 || s.shift != 8 || raw[GC_PIXEL_PACK_CODE] != 8 {
                return Err(bad);
            }
            let gc = Gc::decode(&raw)?;
            return if gc.target == Target::MainRgb8 && !gc.opaque_or_tiled { Ok(gc) } else { Err(bad) };
        }
        _ => {}
    }
    // A glyph into a 24-bit or 30-bit TrueColor window: the colour bytes ride
    // in words 0 and 1 widened to twelve bits; rewrite them into the form the
    // solid fills use, alpha all ones.
    if s.src == 4 && (raw[GC_FRAMEBUFFER_CONFIG] == 0x2804 || raw[GC_FRAMEBUFFER_CONFIG] == 0x3804 || (raw[GC_FRAMEBUFFER_CONFIG] == 0 && raw[GC_OPERATION] == 0x64001)) {
        if s.dst != 2 || s.shift != 0 || (raw[GC_FOREGROUND_HIGH] & !0xf) != 0xfff0 || raw[GC_BACKGROUND_HIGH] != 0 || raw[GC_BACKGROUND_LOW] != 0 || raw[GC_PIXEL_PACK_CODE] != 0 {
            return Err(bad);
        }
        let rgb30 = raw[GC_FRAMEBUFFER_CONFIG] == 0;
        let red = (raw[GC_FOREGROUND_LOW] & 0xfff) as u64;
        let green = ((raw[GC_FOREGROUND_LOW] >> 12) & 0xfff) as u64;
        let blue = ((raw[GC_FOREGROUND_HIGH] & 0xf) as u64) << 8 | (raw[GC_FOREGROUND_LOW] >> 24) as u64;
        let colour = red << 40 | green << 27 | blue << 14 | 0x1ffe;
        raw[GC_FOREGROUND_HIGH] = (colour >> 32) as u32;
        raw[GC_FOREGROUND_LOW] = colour as u32;
        raw[GC_PIXEL_PACK_CODE] = if rgb30 { 9 } else { 8 };
        let gc = Gc::decode(&raw)?;
        return if gc.target == Target::MainRgb8 && !gc.opaque_or_tiled { Ok(gc) } else { Err(bad) };
    }
    // An 8-bit image into the popup planes.
    if s.src == 0 && raw[GC_FRAMEBUFFER_CONFIG] == 0x1020 {
        if s.dst != 1 || s.shift != 0 {
            return Err(bad);
        }
        let gc = Gc::decode(&raw)?;
        return if gc.target == Target::Popup && !gc.opaque_or_tiled { Ok(gc) } else { Err(bad) };
    }
    if s.dst != 1 || s.shift != if bitmap { 0 } else { 8 } || (!bitmap && s.src != 0) {
        return Err(bad);
    }
    // The one-bit image sender sends the glyph form's words with the geometry
    // configuration; a glyph into the popup planes uses configuration 0x1820.
    let geometry = s.src == 4 && raw[GC_FRAMEBUFFER_CONFIG] == 0xc;
    let popup = s.src == 4 && raw[GC_FRAMEBUFFER_CONFIG] == 0x1820;
    if bitmap {
        if (raw[GC_FRAMEBUFFER_CONFIG] != 0x180c && !geometry && !popup) || raw[GC_PIXEL_PACK_CODE] != if s.src == 4 { 0 } else { 5 } {
            return Err(bad);
        }
        if s.src == 4 {
            if raw[GC_FOREGROUND_HIGH] != 0xfff0 || raw[GC_FOREGROUND_LOW] & !0xf0_0fff != 0 || raw[GC_BACKGROUND_HIGH] != 0 || raw[GC_BACKGROUND_LOW] != 0 {
                return Err(bad);
            }
            let value = ((raw[GC_FOREGROUND_LOW] & 0xfff) << 4) | (raw[GC_FOREGROUND_LOW] >> 20);
            raw[GC_FOREGROUND_HIGH] = ((value & 0xfff0) << 4) | ((value & 15) << 3);
            raw[GC_BACKGROUND_HIGH] = 0;
        } else {
            if (raw[GC_FOREGROUND_HIGH] | raw[GC_BACKGROUND_HIGH]) & !0xffff != 0 || raw[GC_FOREGROUND_LOW] != 0 || raw[GC_BACKGROUND_LOW] != 0 {
                return Err(bad);
            }
            raw[GC_FOREGROUND_HIGH] = ((raw[GC_FOREGROUND_HIGH] & 0xfff0) << 4) | ((raw[GC_FOREGROUND_HIGH] & 15) << 3);
            raw[GC_BACKGROUND_HIGH] = ((raw[GC_BACKGROUND_HIGH] & 0xfff0) << 4) | ((raw[GC_BACKGROUND_HIGH] & 15) << 3);
        }
        raw[GC_FOREGROUND_LOW] = 0;
        raw[GC_BACKGROUND_LOW] = 0;
        raw[GC_PIXEL_PACK_CODE] = if geometry { 0 } else { 5 };
    }
    if popup {
        let gc = Gc::decode(&raw)?;
        return if gc.target == Target::Popup && !gc.opaque_or_tiled { Ok(gc) } else { Err(bad) };
    }
    let gc = Gc::decode_main_index(&raw)?;
    if gc.opaque_or_tiled { Err(bad) } else { Ok(gc) }
}

/// Walk an image: call `emit(index, source)` for every pixel that lands on
/// the store, in row order, with the store index it lands on and the word its
/// bits convert to. A clear bit of a one-bit image (`BITMAP`) emits nothing.
fn walk<const KIND: u8, const STIP: bool>(p: &Plan<'_>, c: &Conv, g: &Geo, words: &[u32], spans: &mut Vec<(i64, i64)>, pad: &mut Vec<u32>, mut emit: impl FnMut(usize, u32)) {
    let mask = if c.bits == 32 { u32::MAX as u64 } else { (1u64 << c.bits) - 1 };
    let (xa, xb) = (g.x.max(p.bounds.x1), (g.x + g.width).min(p.bounds.x2));
    for r in 0..g.height {
        let y = g.y + r;
        if y < p.bounds.y1 || y >= p.bounds.y2 || xa >= xb {
            continue;
        }
        // The row's words and a zero one after, so a pixel that ends in the last
        // word has a word to read its low bits from without a test.
        pad.clear();
        pad.extend_from_slice(&words[r as usize * g.stride..(r as usize + 1) * g.stride]);
        pad.push(0);
        let row = &pad[..];
        p.row_spans(y, xa, xb, spans);
        let base = (p.height - 1 - y) as usize * p.pitch;
        for &(a, b) in spans.iter() {
            let (mut sbits, mut swidth, mut col) = (0, 1, 0);
            if STIP {
                if let Some(s) = &p.stipple {
                    (sbits, swidth, col) = p.stipple_start(s, y, a);
                }
            }
            for x in a..b {
                let set = !STIP || sbits & (0x8000_0000 >> col) != 0;
                col += 1;
                if col == swidth {
                    col = 0;
                }
                if !set {
                    continue;
                }
                let bit = c.pad as u64 + c.bits as u64 * (x - g.x) as u64;
                let (w, o) = ((bit / 32) as usize, (bit % 32) as u32);
                // The pixel's bits sit in this word and, if it runs over, the next.
                let window = (row[w] as u64) << 32 | row[w + 1] as u64;
                let v = ((window >> (64 - o - c.bits)) & mask) as u32;
                let src = match KIND {
                    PACKED => (v << c.shift) & c.mask,
                    BITMAP => {
                        if v == 0 {
                            continue;
                        }
                        c.fg
                    }
                    _ => {
                        if v != 0 {
                            c.fg
                        } else {
                            c.bg
                        }
                    }
                };
                emit(base + x as usize, src);
            }
        }
    }
}

/// `walk` with the kind and stipple chosen once.
fn walk_kind(p: &Plan<'_>, kind: u8, c: &Conv, g: &Geo, words: &[u32], spans: &mut Vec<(i64, i64)>, pad: &mut Vec<u32>, emit: impl FnMut(usize, u32)) {
    let stip = p.stipple.is_some();
    match (kind, stip) {
        (PACKED, false) => walk::<PACKED, false>(p, c, g, words, spans, pad, emit),
        (PACKED, true) => walk::<PACKED, true>(p, c, g, words, spans, pad, emit),
        (BITMAP, false) => walk::<BITMAP, false>(p, c, g, words, spans, pad, emit),
        (BITMAP, true) => walk::<BITMAP, true>(p, c, g, words, spans, pad, emit),
        (_, false) => walk::<OPAQUE, false>(p, c, g, words, spans, pad, emit),
        (_, true) => walk::<OPAQUE, true>(p, c, g, words, spans, pad, emit),
    }
}

/// Draw an image under the logic operation, a constant of the loop.
fn draw_image<const GX: u32, const NEEDS: bool>(p: &Plan<'_>, kind: u8, c: &Conv, g: &Geo, words: &[u32], spans: &mut Vec<(i64, i64)>, pad: &mut Vec<u32>) {
    let gc = &p.gc;
    walk_kind(p, kind, c, g, words, spans, pad, |i, src| {
        let old = if NEEDS {
            // SAFETY: the walk clips to the bounds, which are inside the managed area.
            unsafe { p.array.load(i) }
        } else {
            0
        };
        // SAFETY: as above.
        unsafe { p.array.store(i, apply::<GX>(gc, src, old)) };
    });
}

fn draw_by_op(p: &Plan<'_>, kind: u8, c: &Conv, g: &Geo, words: &[u32], spans: &mut Vec<(i64, i64)>, pad: &mut Vec<u32>) {
    macro_rules! by_op {
        ($needs:literal, $($gx:literal),*) => {
            match p.gc.gx as u32 {
                $($gx => draw_image::<$gx, $needs>(p, kind, c, g, words, spans, pad),)*
                _ => {
                    debug_assert!(false, "a decoded context's logic operation is 0 to 15");
                    draw_image::<3, true>(p, kind, c, g, words, spans, pad)
                }
            }
        };
    }
    if p.needs {
        by_op!(true, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    } else {
        // Only the operations that ignore the destination get here.
        by_op!(false, 0, 3, 12, 15)
    }
}

/// Store one row of source words under the logic operation, a constant of the
/// loop: `row[0]` lands at column `x0`, the spans are the columns the clip lets
/// through, `dbase` the destination row's first index.
fn copy_row<const GX: u32>(plan: &Plan<'_>, row: &[u32], spans: &[(i64, i64)], dbase: usize, x0: i64) {
    for &(a, c) in spans {
        for x in a..c {
            let d = dbase + x as usize;
            // SAFETY: the spans are inside the bounds, which are inside the managed area.
            unsafe { plan.array.store(d, apply::<GX>(&plan.gc, row[(x - x0) as usize], plan.array.load(d))) };
        }
    }
}

fn copy_by_op(plan: &Plan<'_>, row: &[u32], spans: &[(i64, i64)], dbase: usize, x0: i64) {
    macro_rules! by_op {
        ($($gx:literal),*) => {
            match plan.gc.gx as u32 {
                $($gx => copy_row::<$gx>(plan, row, spans, dbase, x0),)*
                _ => {
                    debug_assert!(false, "a decoded context's logic operation is 0 to 15");
                    copy_row::<3>(plan, row, spans, dbase, x0)
                }
            }
        };
    }
    by_op!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
}

/// Copy `s.width` x `s.height` pixels from (`sx`, `sy`) to (`dx`, `dy`) under
/// the plan's context, clipped. `row` holds one source row at a time.
fn copy_rows(plan: &Plan<'_>, s: &Setup, [sx, sy, dx, dy]: [i64; 4], spans: &mut Vec<(i64, i64)>, row: &mut Vec<u32>) -> Verdict {
    let (w, h) = (s.width as i64, s.height as i64);
    let array = plan.array;
    let pitch = plan.pitch;
    let top = plan.height - 1;
    let width = pitch as i64;
    let b = plan.bounds;
    let (i0, i1) = ((-sx).max(0).max(b.x1 - dx), w.min(width - sx).min(b.x2 - dx));
    let (j0, j1) = ((-sy).max(0).max(b.y1 - dy), h.min(plan.height - sy).min(b.y2 - dy));
    if i0 >= i1 || j0 >= j1 {
        return Verdict::Done;
    }
    let mask = plan.gc.width_mask;
    // Every source word, and every destination word a write reads, must fit.
    for j in j0..j1 {
        let sbase = (top - (sy + j)) as usize * pitch;
        for i in i0..i1 {
            // SAFETY: the rectangle is inside the managed area; i0..i1 and j0..j1 clip it.
            if unsafe { array.load(sbase + (sx + i) as usize) } > mask {
                return Verdict::Destination;
            }
        }
    }
    if plan.needs {
        for j in j0..j1 {
            let dbase = (top - (dy + j)) as usize * pitch;
            for i in i0..i1 {
                // SAFETY: as above.
                if unsafe { array.load(dbase + (dx + i) as usize) } > mask {
                    return Verdict::Destination;
                }
            }
        }
    }
    // Rows run away from where an overlapping destination lies.
    for k in j0..j1 {
        let j = if dy > sy { j1 - 1 - (k - j0) } else { k };
        let (sbase, dbase) = ((top - (sy + j)) as usize * pitch, (top - (dy + j)) as usize * pitch);
        row.clear();
        // SAFETY: as above.
        row.extend((i0..i1).map(|i| unsafe { array.load(sbase + (sx + i) as usize) }));
        plan.row_spans(dy + j, dx + i0, dx + i1, spans);
        copy_by_op(plan, row, spans, dbase, dx + i0);
    }
    Verdict::Copied
}

impl XState {
    /// A pixel command. `surface` is the plane drawn into.
    /// `answered`: a context is current and a return page is registered, so a
    /// read has somewhere to reply.
    pub fn pixel_command(&mut self, surface: &Surface<'_>, selector: u32, args: &[u32], answered: bool) -> Next {
        match selector {
            X_PIXEL_SETUP => {
                if let Ok(w) = <&[u32; PIXEL_SETUP_WORDS]>::try_from(args) {
                    self.pixel.setup = Some(Setup::from_words(w));
                    self.pixel.counts.setups += 1;
                }
                Next::Nothing
            }
            X_PIXEL_WRITE => self.pixel_write(args),
            X_PIXEL_READ => self.pixel_read(surface, args, answered),
            X_PIXEL_COPY => {
                self.pixel_copy(surface, args);
                Next::Nothing
            }
            X_PIXEL_FINISH => {
                self.pixel.counts.finishes += 1;
                if self.pixel.pending.take().is_some() {
                    self.pixel.counts.short += 1;
                }
                Next::Nothing
            }
            _ => Next::Nothing,
        }
    }

    /// 0x139: where the image goes, and how its rows will come.
    fn pixel_write(&mut self, args: &[u32]) -> Next {
        let Ok(&[x, y]) = <&[u32; 2]>::try_from(args) else {
            self.pixel.counts.bad_context += 1;
            return Next::Nothing;
        };
        self.pixel.pending = None;
        let Some(setup) = self.pixel.setup else {
            self.pixel.counts.bad_context += 1;
            return Next::Nothing;
        };
        let Some((_, words)) = setup.words() else {
            self.pixel.counts.bad_context += 1;
            return Next::Nothing;
        };
        if words == 0 {
            return Next::Nothing;
        }
        let dma = setup.mode & Setup::MODE_DMA != 0;
        self.pixel.pending = Some(Pending { x: x as i32, y: y as i32, words, dma });
        if dma {
            Next::Dma
        } else {
            Next::Words(words as u32)
        }
    }

    /// Whether a write waits for its rows by DMA, and how many words it owes.
    pub fn dma_write_words(&self) -> Option<usize> {
        self.pixel.pending.filter(|p| p.dma).map(|p| p.words)
    }

    /// A DMA packet that carried no write's rows. Counted.
    pub fn pixel_unexpected(&mut self) {
        self.pixel.counts.unexpected += 1;
    }

    /// The rows of the open write, as the words of the data run or the DMA
    /// packet. A count that is not what the setup announced draws nothing.
    pub fn pixel_rows(&mut self, surface: &Surface<'_>, words: &[u32]) {
        let Some(p) = self.pixel.pending.take() else {
            self.pixel.counts.unexpected += 1;
            return;
        };
        if words.len() != p.words {
            self.pixel.counts.short += 1;
            return;
        }
        self.pixel.counts.writes += 1;
        self.pixel.counts.dma_writes += p.dma as u64;
        let Some(setup) = self.pixel.setup else { return };
        let Some((stride, _)) = setup.words() else { return };
        let gc = match decode_write(&self.gc, &setup) {
            Ok(gc) if gc.target != Target::Popup => gc,
            Ok(_) => {
                self.pixel.counts.other_target += 1;
                return;
            }
            Err(_) => {
                self.pixel.counts.bad_context += 1;
                return;
            }
        };
        let stipple = self.stipple_pattern();
        let (kind, bits) = match setup.src {
            4 => (BITMAP, 1),
            5 => (OPAQUE, 1),
            _ => (PACKED, setup.bits().unwrap_or(8)),
        };
        let conv = Conv {
            fg: gc.foreground,
            bg: gc.background,
            shift: if gc.target == Target::MainRgb8 { 0 } else { setup.shift },
            mask: if gc.target == Target::MainRgb8 { 0xff_ffff } else { 0xffff },
            bits,
            pad: setup.pad,
        };
        let geo = Geo { x: p.x as i64, y: p.y as i64, width: setup.width as i64, height: setup.height as i64, stride };
        let mut spans = std::mem::take(&mut self.spans);
        let mut pad = std::mem::take(&mut self.pixel.row);
        let verdict = match Plan::with_gc(self, surface, gc, stipple) {
            Ok(Some(plan)) => {
                let fits = !plan.needs || {
                    let mut ok = true;
                    let mask = plan.gc.width_mask;
                    walk_kind(&plan, kind, &conv, &geo, words, &mut spans, &mut pad, |i, _| {
                        // SAFETY: the walk clips to the bounds, which are inside the managed area.
                        ok &= unsafe { plan.array.load(i) } <= mask;
                    });
                    ok
                };
                if fits {
                    draw_by_op(&plan, kind, &conv, &geo, words, &mut spans, &mut pad);
                    Verdict::Done
                } else {
                    Verdict::Destination
                }
            }
            Ok(None) => Verdict::Done,
            Err(Refused::Target) => Verdict::Target,
            Err(_) => Verdict::Context,
        };
        self.spans = spans;
        self.pixel.row = pad;
        self.pixel.count(verdict);
    }

    /// 0x138: a rectangle of the main plane, as the packing the setup names.
    /// Inside a read bracket the bytes wait for its finish; otherwise they go
    /// on the return page. A read that cannot be answered from the store is
    /// answered with zeros all the same, if there is somewhere to answer.
    fn pixel_read(&mut self, surface: &Surface<'_>, args: &[u32], answered: bool) -> Next {
        self.pixel.counts.reads += 1;
        // A bracket's buffer takes up to 64 MiB; the return page, what is left of it.
        let limit = if self.pixel.read_dma.is_some() { 64 << 20 } else { READ_LIMIT };
        let data = self.read_payload(surface, args, limit).ok();
        if data.is_none() {
            self.pixel.counts.reads_refused += 1;
        }
        let Some(mut dma) = self.pixel.read_dma.take() else {
            return Next::Return(data.unwrap_or_else(|| vec![0; READ_LIMIT]));
        };
        let setup = self.pixel.setup.unwrap_or_default();
        let (x, y) = (args.first().copied().unwrap_or(0), args.get(1).copied().unwrap_or(0));
        let header = [x, y, setup.width, setup.height, [1, 2, 4].get(setup.dst as usize).copied().unwrap_or(0), 0, 0];
        if dma.refused || dma.data.is_some() || !dma.fits(&header) {
            dma.refused = true;
            dma.data = None;
        } else if let Some(d) = data {
            dma.header = header;
            dma.data = Some(d);
        } else if answered {
            // The descriptor fixes the byte count: a read the store cannot answer completes as zeros.
            dma.header = header;
            dma.data = Some(vec![0; dma.descriptor[2] as usize]);
        } else {
            dma.refused = true;
        }
        self.pixel.read_dma = Some(dma);
        Next::Nothing
    }

    /// Whether a select-feed bracket is open.
    pub fn bracket_open(&self) -> bool {
        self.pixel.read_dma.is_some()
    }

    /// A texel read-back's bytes (token 0x052) for the open bracket: the packed image, `rows` rows of
    /// `row` bytes. The client's descriptor names the image's padded length as one line with no stride
    /// (the table), and the bracket's finish writes the bytes there from its first byte. A descriptor of
    /// `rows` lines of `row` is taken too. Any other length refuses the bracket, counted, as an X read's
    /// would be. False when no bracket is open.
    pub fn feed_texels(&mut self, row: u32, rows: u32, mut data: Vec<u8>) -> bool {
        let Some(mut dma) = self.pixel.read_dma.take() else { return false };
        let header = [0, 0, row, rows, 1, 0, 0];
        let d = dma.descriptor;
        if dma.refused || dma.data.is_some() {
            dma.refused = true;
            dma.data = None;
        } else if d[2] as usize == data.len() && d[3] == d[2] && d[4] == 0 && dma.buffer().is_some() {
            dma.texels = true;
            dma.data = Some(data);
        } else if !dma.fits(&header) {
            dma.refused = true;
            dma.data = None;
        } else {
            data.resize(row as usize * rows as usize, 0);
            dma.header = header;
            dma.data = Some(data);
        }
        self.pixel.read_dma = Some(dma);
        true
    }

    /// Select-feed start (0x1f5): a bracket whose read will be written to a
    /// host buffer. `owned`: a context is current.
    pub fn feed_start(&mut self, args: &[u32], owned: bool) {
        let Ok(&descriptor) = <&[u32; 6]>::try_from(args) else { return };
        let mut dma = ReadDma { descriptor, refused: !owned, header: [0; 7], data: None, texels: false };
        dma.refused |= dma.buffer().is_none();
        self.pixel.read_dma = Some(dma);
    }

    /// Select-feed finish (0x1f6): the read's bytes for the host buffer and
    /// the completion to post, or None, counted, for a bracket that is not a
    /// whole read of a kind this model takes (a feedback or selection pass's
    /// bracket holds none).
    pub fn feed_finish(&mut self, args: &[u32]) -> Option<FeedDone> {
        let dma = self.pixel.read_dma.take()?;
        let &[mark, echo] = <&[u32; 2]>::try_from(args).ok()?;
        if dma.refused || mark != 0x100 || !(dma.texels || dma.fits(&dma.header)) {
            self.pixel.counts.bracket_refused += 1;
            return None;
        }
        let (io, pitch, _) = dma.buffer()?;
        let data = dma.data?;
        Some(FeedDone { io, pitch, line: dma.descriptor[3] as usize, data, echo })
    }

    /// A select-feed bracket's host buffer, taking the bracket: the device address of the first
    /// line, the line pitch, the bytes in a line and in all. None for a bracket the descriptor
    /// does not describe a buffer for. A feedback or select pass writes there.
    pub fn feed_buffer(&mut self) -> Option<(u64, u64, usize, usize)> {
        let dma = self.pixel.read_dma.take()?;
        if dma.refused {
            return None;
        }
        let (io, pitch, _) = dma.buffer()?;
        Some((io, pitch, dma.descriptor[3] as usize, dma.descriptor[2] as usize))
    }

    /// The bytes of the rectangle at `args`, or why the store cannot answer
    /// it: the setup or context is not a form the X server's reads send, the
    /// rectangle is off the screen or too big for the page, or a source word
    /// is not a main-plane colour of the width the packing takes.
    fn read_payload(&mut self, surface: &Surface<'_>, args: &[u32], limit: usize) -> Result<Vec<u8>, &'static str> {
        let &[x, y] = <&[u32; 2]>::try_from(args).map_err(|_| "not two words")?;
        let s = self.pixel.setup.ok_or("no setup loaded")?;
        let (x, y) = (x as i32 as i64, y as i32 as i64);
        let bytes: usize = match (s.dst, s.shift as i32) {
            (0, 8) => 1,
            (1, 12) => 2,
            (2, -8) | (2, 0) => 4,
            _ => 0,
        };
        if bytes == 4 && s.shift == 0 {
            // Told apart by a per-pixel tag this store does not keep.
            self.pixel.counts.fb_word_reads += 1;
            return Err("a frame-buffer word read needs a pixel tag");
        }
        let rgb = bytes == 4;
        let (w6, rest) = (self.gc[6], &self.gc[7..]);
        let gc_ok = if rgb { w6 == 0x2804 && rest == [0x60001, 1, 6, 8, 0, 0] } else { (w6 == 0x180c && rest == [0x40001, 1, 6, 5, 1, 1]) || (w6 == 0xc && rest == [1, 1, 6, 0, 1, 1]) };
        if bytes == 0 {
            return Err("unknown packing");
        }
        if !gc_ok {
            return Err("not a main-plane read context");
        }
        if s.mode != 2 || s.src != if rgb { 2 } else { 1 } || s.pad != 0 {
            return Err("not a read setup");
        }
        let (width, height) = (s.width as i32 as i64, s.height as i32 as i64);
        let (sw, sh) = surface.size();
        if width <= 0 || height <= 0 || x < 0 || y < 0 || x + width > sw as i64 || y + height > sh as i64 {
            return Err("rectangle outside the main plane");
        }
        let pitch = width as usize * bytes;
        if pitch > limit || height as usize > limit / pitch {
            return Err("payload exceeds the room for it");
        }
        let array = surface.array();
        let limit = if rgb { 0xff_ffff } else { 0xffff };
        let at = |c: i64, r: i64| (sh as i64 - 1 - y - r) as usize * array.pitch() + (x + c) as usize;
        // Every source is checked before any of the reply is made.
        for r in 0..height {
            for c in 0..width {
                // SAFETY: the rectangle is inside the managed area.
                if unsafe { array.load(at(c, r)) } > limit {
                    return Err("source is not main-plane colour");
                }
            }
        }
        let mut data = vec![0u8; pitch * height as usize];
        macro_rules! rows {
            ($b:literal, $pack:expr) => {
                for r in 0..height {
                    let base = (sh as i64 - 1 - y - r) as usize * array.pitch() + x as usize;
                    let out = &mut data[r as usize * pitch..(r as usize + 1) * pitch];
                    for (c, px) in out.chunks_exact_mut($b).enumerate() {
                        // SAFETY: the rectangle is inside the managed area.
                        let word = unsafe { array.load(base + c) };
                        $pack(word, px);
                    }
                }
            };
        }
        match bytes {
            4 => rows!(4, |word: u32, px: &mut [u8]| px.copy_from_slice(&word.to_be_bytes())),
            2 => rows!(2, |word: u32, px: &mut [u8]| px.copy_from_slice(&(((word >> 4) & 0xfff) as u16).to_be_bytes())),
            _ => rows!(1, |word: u32, px: &mut [u8]| px[0] = (word >> 8) as u8),
        }
        Ok(data)
    }

    /// 0x137: copy a rectangle inside the store under the context's logic
    /// operation, clipped to the clip list and scissor. Every source word is
    /// checked first, and rows run in the order that keeps an overlapping
    /// source whole.
    fn pixel_copy(&mut self, surface: &Surface<'_>, args: &[u32]) {
        let Ok(&[sx, sy, dx, dy]) = <&[u32; 4]>::try_from(args) else {
            self.pixel.counts.bad_context += 1;
            return;
        };
        let Some(s) = self.pixel.setup else {
            self.pixel.counts.bad_context += 1;
            return;
        };
        if s.src == 3 && s.dst == 3 {
            self.pixel.counts.window_copies += 1;
            return;
        }
        let gc = match Gc::decode_for_fill(&self.gc) {
            Ok(gc) => gc,
            Err(_) => {
                self.pixel.counts.bad_context += 1;
                return;
            }
        };
        let rgb = gc.target == Target::MainRgb8;
        let form_ok = if rgb { gc.raw[GC_PIXEL_PACK_CODE] == 8 || gc.raw[GC_PIXEL_PACK_CODE] == 9 } else { gc.target == Target::MainIndex16 && gc.raw[GC_PIXEL_PACK_CODE] == 5 };
        let pack = if rgb { 2 } else { 1 };
        if !form_ok || s.mode != 1 || s.src != pack || s.dst != pack || s.shift != 0 || s.pad != 0 {
            self.pixel.counts.bad_context += 1;
            return;
        }
        if gc.is_noop() {
            return;
        }
        let (mut spans, mut row) = (std::mem::take(&mut self.spans), std::mem::take(&mut self.pixel.row));
        let verdict = match Plan::with_gc(self, surface, gc, None) {
            Ok(Some(plan)) => copy_rows(&plan, &s, [sx as i32 as i64, sy as i32 as i64, dx as i32 as i64, dy as i32 as i64], &mut spans, &mut row),
            Ok(None) => Verdict::Done,
            Err(Refused::Target) => Verdict::Target,
            Err(_) => Verdict::Context,
        };
        self.spans = spans;
        self.pixel.row = row;
        self.pixel.count(verdict);
    }

    /// A read bracket's bytes could not be written, or there was no reason
    /// page: no completion was posted.
    pub fn bracket_unposted(&mut self) {
        self.pixel.counts.bracket_unposted += 1;
    }

    pub fn pixel_counts(&self) -> &PixelCounts {
        &self.pixel.counts
    }

    /// A destroyed window ends the transfer open on it.
    pub fn pixel_release(&mut self) {
        if self.pixel.pending.take().is_some() {
            self.pixel.counts.short += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dev::kona::back::store::PixelStore;

    /// An index context, native 16-bit index, full mask.
    const INDEX: [u32; 13] = [0x5a000, 0x1ffe, 0xa5000, 0x1ffe, 0xf00fff, 0, 0x180c, 0x40001, 1, 6, 5, 1, 1];
    /// 24-bit RGB, copy, full mask.
    const RGB: [u32; 13] = [0, 0, 0, 0, 0xff_ffff, 0xff_ffff, 0x2804, 0x60001, 1, 6, 8, 0, 0];
    /// A glyph colour, index 0x1234, in the one-bit senders' form.
    const GLYPH: [u32; 13] = [0xfff0, 0x123 | 4 << 20, 0, 0, 0xf00fff, 0, 0x180c, 0x40001, 1, 6, 0, 1, 1];
    /// An opaque one-bit image: foreground index 0x1230, background 0x10.
    const OPAQUE_GC: [u32; 13] = [0x1230, 0, 0x10, 0, 0xf00fff, 0, 0x180c, 0x40001, 1, 6, 5, 1, 1];

    fn rig(gc: &[u32; 13]) -> (XState, PixelStore) {
        let mut x = XState::default();
        x.command(0x129, gc);
        (x, PixelStore::new(32, 32))
    }

    /// The word at X coordinates (`x`, `y`): the store is bottom-origin.
    fn at(s: &PixelStore, x: u32, y: u32) -> u32 {
        s.read(x, 31 - y, 1)
    }

    fn row(s: &PixelStore, y: u32, n: u32) -> Vec<u32> {
        (0..n).map(|c| at(s, c, y)).collect()
    }

    fn cmd(x: &mut XState, s: &PixelStore, selector: u32, args: &[u32]) -> Next {
        x.pixel_command(&Surface::new(s, 1), selector, args, true)
    }

    /// Set up, open a write at `pos` and send its rows as the data run.
    fn write(x: &mut XState, s: &PixelStore, w: [u32; 7], pos: (u32, u32), rows: &[u32]) -> Next {
        cmd(x, s, X_PIXEL_SETUP, &w);
        let next = cmd(x, s, X_PIXEL_WRITE, &[pos.0, pos.1]);
        if let Next::Words(n) = next {
            assert_eq!(n as usize, rows.len(), "the run the back end asks for");
            x.pixel_rows(&Surface::new(s, 1), rows);
        }
        next
    }

    /// An 8-bit image lands shifted into the index word, its first row at the
    /// position and each row below the last; pixels pack most significant first.
    #[test]
    fn an_eight_bit_image_lands_shifted_top_row_first() {
        let (mut x, s) = rig(&INDEX);
        assert_eq!(write(&mut x, &s, [0, 0, 1, 8, 3, 2, 0], (4, 2), &[0x0102_0300, 0x0405_0600]), Next::Words(2));
        assert_eq!([at(&s, 4, 2), at(&s, 5, 2), at(&s, 6, 2)], [0x100, 0x200, 0x300]);
        assert_eq!([at(&s, 4, 3), at(&s, 5, 3), at(&s, 6, 3)], [0x400, 0x500, 0x600]);
        assert_eq!(at(&s, 7, 2), 0);
        assert_eq!(x.pixel_counts().writes, 1);
    }

    /// A pad of bits starts the row's first pixel that far in, and a pixel may
    /// run over into the next word of the row.
    #[test]
    fn a_row_starts_after_its_pad_and_pixels_cross_words() {
        let (mut x, s) = rig(&INDEX);
        write(&mut x, &s, [0, 0, 1, 8, 2, 1, 8], (0, 0), &[0x0011_2200]);
        assert_eq!(row(&s, 0, 2), [0x1100, 0x2200]);
        // Ten bits of pad, then 0xab 0xcd 0xef: the last pixel ends in the second word.
        let next = write(&mut x, &s, [0, 0, 1, 8, 3, 1, 10], (0, 1), &[0x002a_f37b, 0xc000_0000]);
        assert_eq!(next, Next::Words(2));
        assert_eq!(row(&s, 1, 3), [0xab00, 0xcd00, 0xef00]);
    }

    /// A 16-bit image takes the setup's shift of 4, a 32-bit one into an RGB
    /// window keeps the low 24 bits.
    #[test]
    fn sixteen_and_thirty_two_bit_images_convert_to_the_plane_word() {
        let (mut x, s) = rig(&INDEX);
        write(&mut x, &s, [0, 1, 1, 4, 2, 1, 0], (0, 0), &[0x1234_0056]);
        assert_eq!(row(&s, 0, 2), [0x2340, 0x0560]);
        let (mut x, s) = rig(&RGB);
        write(&mut x, &s, [0, 2, 2, 8, 2, 1, 0], (3, 3), &[0x01ff_8040, 0xaa11_2233]);
        assert_eq!([at(&s, 3, 3), at(&s, 4, 3)], [0xff_8040, 0x11_2233]);
    }

    /// A one-bit image paints the context's colour where a bit is set and
    /// leaves the rest; the opaque form paints its background there too.
    #[test]
    fn one_bit_images_paint_the_foreground_and_the_opaque_form_the_background() {
        let (mut x, s) = rig(&GLYPH);
        s.write(1, 31, 1, 0x7777);
        write(&mut x, &s, [0, 4, 1, 0, 5, 2, 0], (0, 0), &[0xb000_0000, 0x4000_0000]);
        assert_eq!(row(&s, 0, 5), [0x1234, 0x7777, 0x1234, 0x1234, 0], "a clear bit leaves its pixel");
        assert_eq!(row(&s, 1, 5), [0, 0x1234, 0, 0, 0]);
        let (mut x, s) = rig(&OPAQUE_GC);
        write(&mut x, &s, [0, 5, 1, 0, 3, 1, 0], (0, 0), &[0xa000_0000]);
        assert_eq!(row(&s, 0, 3), [0x1230, 0x10, 0x1230]);
    }

    /// The clip list and the scissor narrow a write, and a loaded stipple
    /// masks it.
    #[test]
    fn a_write_is_clipped_scissored_and_stippled() {
        let (mut x, s) = rig(&INDEX);
        x.command(0x12b, &[1, 2, 0, 4, 1]);
        write(&mut x, &s, [0, 0, 1, 8, 4, 2, 0], (0, 0), &[0x0102_0304, 0x0506_0708]);
        assert_eq!(row(&s, 0, 4), [0, 0, 0x300, 0x400], "the clip is x 2 to 3, y 0");
        assert_eq!(row(&s, 1, 4), [0; 4]);
        x.command(0x12d, &[3, 0, 1, 1]);
        write(&mut x, &s, [0, 0, 1, 8, 4, 1, 0], (0, 0), &[0x0a0b_0c0d]);
        assert_eq!(row(&s, 0, 4), [0, 0, 0x300, 0xd00], "the scissor leaves column 3 only");
        // A 2 x 1 stipple, left bit set: every other column of a fresh clip.
        x.command(0x12b, &[1, 0, 0, 8, 8]);
        x.command(0x13d, &[1, 2, 1]);
        x.stipple.rows(&[0x8000_0000]);
        x.command(0x13c, &[1]);
        write(&mut x, &s, [0, 0, 1, 8, 4, 1, 0], (0, 4), &[0x0102_0304]);
        assert_eq!(row(&s, 4, 4), [0x100, 0, 0x300, 0]);
    }

    /// A write under a partial mask or a logic operation combines with the
    /// pixel it covers; a destination word too wide for the target refuses the
    /// whole write.
    #[test]
    fn a_masked_write_combines_and_a_wide_destination_refuses_it() {
        // Mask 0x00ff: the high byte of the destination stays.
        let mut gc = INDEX;
        gc[4] = 0xf0000f;
        let (mut x, s) = rig(&gc);
        s.write(0, 31, 1, 0xabcd);
        write(&mut x, &s, [0, 0, 1, 8, 1, 1, 0], (0, 0), &[0x0100_0000]);
        assert_eq!(at(&s, 0, 0), 0xab00);
        // Pixel 1 holds a word wider than the 16-bit target: nothing is drawn.
        s.write(1, 31, 1, 0x1_0000);
        write(&mut x, &s, [0, 0, 1, 8, 2, 1, 0], (0, 0), &[0x0202_0000]);
        assert_eq!(row(&s, 0, 2), [0xab00, 0x1_0000]);
        assert_eq!(x.pixel_counts().bad_destination, 1);
        // XOR (operation 6) with the full mask reads the destination too.
        let mut gc = INDEX;
        gc[9] = 12;
        let (mut x, s) = rig(&gc);
        s.write(0, 31, 1, 0x0f0f);
        write(&mut x, &s, [0, 0, 1, 8, 1, 1, 0], (0, 0), &[0xff00_0000]);
        assert_eq!(at(&s, 0, 0), 0xff00 ^ 0x0f0f);
    }

    /// A write whose rows do not number what the setup announced draws nothing
    /// and is counted; so is one with no setup, a packing that is not known and
    /// a target with no store (the popup planes).
    #[test]
    fn writes_that_cannot_be_drawn_are_counted() {
        let (mut x, s) = rig(&INDEX);
        assert_eq!(cmd(&mut x, &s, X_PIXEL_WRITE, &[0, 0]), Next::Nothing, "no setup");
        cmd(&mut x, &s, X_PIXEL_SETUP, &[0, 0, 1, 8, 2, 1, 0]);
        cmd(&mut x, &s, X_PIXEL_WRITE, &[0, 0]);
        x.pixel_rows(&Surface::new(&s, 1), &[1, 2]);
        assert_eq!(at(&s, 0, 0), 0);
        let c = x.pixel_counts().clone();
        assert_eq!((c.bad_context, c.short, c.writes), (1, 1, 0));
        assert_eq!(write(&mut x, &s, [0, 3, 1, 8, 2, 1, 0], (0, 0), &[1]), Next::Nothing, "packing 3 has no framing");
        assert_eq!(x.pixel_counts().bad_context, 2);
        let mut popup = INDEX;
        popup[0] = 0x78;
        popup[2] = 0x38;
        popup[4] = 0x30_0000;
        popup[6] = 0x1020;
        popup[7] = 1;
        popup[8] = 0;
        let (mut x, s) = rig(&popup);
        write(&mut x, &s, [0, 0, 1, 0, 1, 1, 0], (0, 0), &[0x0100_0000]);
        assert_eq!((x.pixel_counts().other_target, at(&s, 0, 0)), (1, 0), "{:?}", x.pixel_counts());
        // A rows packet with no write waiting is counted too.
        x.pixel_rows(&Surface::new(&s, 1), &[0]);
        assert_eq!(x.pixel_counts().unexpected, 1);
    }

    /// The DMA bit of the mode word sends the rows by DMA: the write waits for
    /// them, a finish ends it, and a destroyed window drops it.
    #[test]
    fn a_dma_write_waits_for_its_rows() {
        let (mut x, s) = rig(&INDEX);
        assert_eq!(write(&mut x, &s, [0x10, 0, 1, 8, 3, 2, 0], (1, 1), &[]), Next::Dma);
        assert_eq!(x.dma_write_words(), Some(2));
        x.pixel_rows(&Surface::new(&s, 1), &[0x0102_0300, 0x0405_0600]);
        assert_eq!((at(&s, 1, 1), at(&s, 3, 2)), (0x100, 0x600));
        assert_eq!((x.dma_write_words(), x.pixel_counts().dma_writes), (None, 1));
        // An open write ended by the finish, or by its window going, was short.
        write(&mut x, &s, [0x10, 0, 1, 8, 1, 1, 0], (0, 0), &[]);
        cmd(&mut x, &s, X_PIXEL_FINISH, &[0]);
        write(&mut x, &s, [0x10, 0, 1, 8, 1, 1, 0], (0, 0), &[]);
        x.pixel_release();
        assert_eq!((x.pixel_counts().short, x.pixel_counts().finishes), (2, 1));
        assert_eq!(x.dma_write_words(), None);
    }

    /// A copy moves a rectangle, and an overlapping one moves as if the source
    /// were held apart, whichever way it runs.
    #[test]
    fn a_copy_handles_an_overlap_in_either_direction() {
        let (mut x, s) = rig(&INDEX);
        for c in 0..8 {
            for r in 0..4 {
                s.write(c, 31 - r, 1, (c + 10 * r) as u32);
            }
        }
        cmd(&mut x, &s, X_PIXEL_SETUP, &[1, 1, 1, 0, 6, 1, 0]);
        // Right by two, within the row.
        cmd(&mut x, &s, X_PIXEL_COPY, &[0, 0, 2, 0]);
        assert_eq!(row(&s, 0, 8), [0, 1, 0, 1, 2, 3, 4, 5]);
        // Two rows down by one, over the row the second reads.
        cmd(&mut x, &s, X_PIXEL_SETUP, &[1, 1, 1, 0, 3, 2, 0]);
        cmd(&mut x, &s, X_PIXEL_COPY, &[0, 0, 0, 1]);
        assert_eq!(row(&s, 1, 3), [0, 1, 0]);
        assert_eq!(row(&s, 2, 3), [10, 11, 12], "row 1 was read before it was written over");
        // And two rows up by one.
        cmd(&mut x, &s, X_PIXEL_COPY, &[0, 1, 0, 0]);
        assert_eq!(row(&s, 0, 3), [0, 1, 0]);
        assert_eq!(row(&s, 1, 3), [10, 11, 12]);
        assert_eq!(x.pixel_counts().copies, 3);
    }

    /// A copy is clipped, refuses the window-copy pair and a source word too
    /// wide for the target, and an RGB window copies as 24-bit words.
    #[test]
    fn a_copy_is_clipped_and_refuses_what_it_cannot_draw() {
        let (mut x, s) = rig(&INDEX);
        s.write(0, 31, 1, 7);
        s.write(1, 31, 1, 8);
        cmd(&mut x, &s, X_PIXEL_SETUP, &[1, 1, 1, 0, 2, 1, 0]);
        x.command(0x12b, &[1, 11, 0, 12, 1]);
        cmd(&mut x, &s, X_PIXEL_COPY, &[0, 0, 10, 0]);
        assert_eq!(row(&s, 0, 12)[10..], [0, 8], "only x 11 is inside the clip");
        x.command(0x12b, &[1, 0, 0, 32, 32]);
        s.write(0, 31, 1, 0x1_0000);
        cmd(&mut x, &s, X_PIXEL_COPY, &[0, 0, 4, 0]);
        assert_eq!((at(&s, 4, 0), x.pixel_counts().bad_destination), (0, 1));
        cmd(&mut x, &s, X_PIXEL_SETUP, &[1, 3, 3, 0, 2, 1, 0]);
        cmd(&mut x, &s, X_PIXEL_COPY, &[0, 0, 4, 0]);
        assert_eq!(x.pixel_counts().window_copies, 1);
        let (mut x, s) = rig(&RGB);
        s.write(0, 31, 1, 0xab_cdef);
        cmd(&mut x, &s, X_PIXEL_SETUP, &[1, 2, 2, 0, 1, 1, 0]);
        cmd(&mut x, &s, X_PIXEL_COPY, &[0, 0, 5, 5]);
        assert_eq!(at(&s, 5, 5), 0xab_cdef);
    }

    /// A read packs the words as the setup asks, and a read that cannot be
    /// answered from the store completes with the whole usable page of zeros.
    #[test]
    fn a_read_packs_the_plane_words_and_a_refused_one_answers_zeros() {
        let (mut x, s) = rig(&INDEX);
        // The native read context.
        let mut gc = [0u32; 13];
        gc[6..].copy_from_slice(&[0x180c, 0x40001, 1, 6, 5, 1, 1]);
        x.command(0x129, &gc);
        s.write(2, 31 - 3, 1, 0xabcd);
        s.write(3, 31 - 3, 1, 0x1234);
        cmd(&mut x, &s, X_PIXEL_SETUP, &[2, 1, 0, 8, 2, 1, 0]);
        assert_eq!(cmd(&mut x, &s, X_PIXEL_READ, &[2, 3]), Next::Return(vec![0xab, 0x12]), "the high byte of each index");
        cmd(&mut x, &s, X_PIXEL_SETUP, &[2, 1, 1, 12, 2, 1, 0]);
        assert_eq!(cmd(&mut x, &s, X_PIXEL_READ, &[2, 3]), Next::Return(vec![0x0a, 0xbc, 0x01, 0x23]), "twelve bits each");
        // 24-bit RGB words.
        let mut gc = [0u32; 13];
        gc[6..].copy_from_slice(&[0x2804, 0x60001, 1, 6, 8, 0, 0]);
        x.command(0x129, &gc);
        s.write(0, 31, 1, 0x00_0102_03);
        cmd(&mut x, &s, X_PIXEL_SETUP, &[2, 2, 2, (-8i32) as u32, 1, 1, 0]);
        assert_eq!(cmd(&mut x, &s, X_PIXEL_READ, &[0, 0]), Next::Return(vec![0, 1, 2, 3]));
        assert_eq!(x.pixel_counts().reads_refused, 0);
        // A frame-buffer word read has no tag to answer from: zeros, counted.
        cmd(&mut x, &s, X_PIXEL_SETUP, &[2, 2, 2, 0, 1, 1, 0]);
        assert_eq!(cmd(&mut x, &s, X_PIXEL_READ, &[0, 0]), Next::Return(vec![0; READ_LIMIT]));
        // So is a rectangle off the screen, and a source too wide for the packing.
        cmd(&mut x, &s, X_PIXEL_SETUP, &[2, 2, 2, (-8i32) as u32, 2, 1, 0]);
        assert_eq!(cmd(&mut x, &s, X_PIXEL_READ, &[31, 0]), Next::Return(vec![0; READ_LIMIT]));
        s.write(0, 31, 1, 0x100_0000);
        cmd(&mut x, &s, X_PIXEL_SETUP, &[2, 2, 2, (-8i32) as u32, 1, 1, 0]);
        assert_eq!(cmd(&mut x, &s, X_PIXEL_READ, &[0, 0]), Next::Return(vec![0; READ_LIMIT]));
        let c = x.pixel_counts();
        assert_eq!((c.reads, c.reads_refused, c.fb_word_reads), (6, 3, 1));
    }

    /// A read inside a bracket waits for the finish, which hands back its
    /// lines for the host buffer; a bracket whose descriptor or read does not
    /// fit is refused and writes nothing.
    #[test]
    fn a_read_bracket_hands_back_its_lines_at_the_finish() {
        let mut gc = [0u32; 13];
        gc[6..].copy_from_slice(&[0x2804, 0x60001, 1, 6, 8, 0, 0]);
        let (mut x, s) = rig(&gc);
        s.write(0, 31, 1, 0x11);
        s.write(1, 31, 1, 0x22);
        s.write(0, 30, 1, 0x33);
        s.write(1, 30, 1, 0x44);
        cmd(&mut x, &s, X_PIXEL_SETUP, &[2, 2, 2, (-8i32) as u32, 2, 2, 0]);
        // Two lines of eight bytes, a gap of four between them.
        x.feed_start(&[0x1c0, 0x9000 | 0x10, 16, 8, 4, 0], true);
        assert_eq!(cmd(&mut x, &s, X_PIXEL_READ, &[0, 0]), Next::Nothing);
        let done = x.feed_finish(&[0x100, 0x55]).expect("a finished read");
        assert_eq!((done.io, done.pitch, done.line, done.echo), (0x1c0 << 12 | 0x10, 12, 8, 0x55));
        assert_eq!(done.data, [0, 0, 0, 0x11, 0, 0, 0, 0x22, 0, 0, 0, 0x33, 0, 0, 0, 0x44]);
        // One line the size of the whole read is the same bytes.
        x.feed_start(&[0x1c0, 0x9000, 16, 16, 0, 0], true);
        cmd(&mut x, &s, X_PIXEL_READ, &[0, 0]);
        assert_eq!(x.feed_finish(&[0x100, 0]).map(|d| (d.line, d.data.len())), Some((16, 16)));
        // A count that is not the read's, a bad tag, no context and a finish that is not 0x100 refuse.
        for (desc, owned, mark) in [([0x1c0, 0x9000, 20, 20, 0, 0], true, 0x100), ([0x1c0, 0x8000, 16, 16, 0, 0], true, 0x100), ([0x1c0, 0x9000, 16, 16, 0, 0], false, 0x100), ([0x1c0, 0x9000, 16, 16, 0, 0], true, 0x101)] {
            x.feed_start(&desc, owned);
            cmd(&mut x, &s, X_PIXEL_READ, &[0, 0]);
            assert_eq!(x.feed_finish(&[mark, 0]), None, "{desc:x?} {owned} {mark:#x}");
        }
        assert_eq!(x.pixel_counts().bracket_refused, 4);
        // A bracket with no read in it has nothing to write (a feedback pass's).
        x.feed_start(&[0x1c0, 0x9000, 16, 16, 0, 0], true);
        assert_eq!(x.feed_finish(&[0x100, 0]), None);
    }
}
