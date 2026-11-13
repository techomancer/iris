//! Display composition in two steps, the shape of the IMPACT model's
//! (`mgras/frame.rs`):
//!
//! 1. `Frame::snapshot`, taken by the display thread after the retrace: the
//!    PB&J scan-out registers (pixel and overlay/window-id bases, size,
//!    stride in tiles, frame-buffer depth) become two surfaces over board
//!    memory, and the display modes, overlay modes, colour lookup RAM, gamma
//!    table and the displayed-buffer bits become per-window-id decisions
//!    (`ResolveCtx`).
//! 2. `Frame::compose`: one pass over the whole screen. Each pixel's
//!    window-id byte picks its window's decisions, the overlay byte shows
//!    over it when its overlay id is enabled, and the main pixel is the
//!    frame-buffer field of the buffer the window displays.
//!
//! The pass walks each row a tile at a time: a tile's row of 16 pixels is
//! contiguous in both buffers, so its window-id and overlay bytes are four
//! memory words and its frame-buffer pixels sit at a fixed step. Each window
//! id's shown field is planned once a frame (`Plan`): a reader monomorphised
//! over the field's codec and over RGB or colour index. A span of 16 pixels
//! of one window id with no overlay byte, nearly every span on a desktop,
//! goes through one call of the span reader; any other span goes pixel by
//! pixel through the per-pixel reader. `Frame::pixel` is the reference both
//! must match (`compose_matches_the_pixel_reference`).
//!
//! No dirty lines: every frame is composed whole from memory, as the board
//! scans it out.

use super::display::{IdShow, ResolveCtx};
use super::field::{lanes_from, texel_unpack, word_of, Codec, Field, Layout, FIELD_FOUR, FIELD_INDEX, FIELD_PACKED, PIXEL_FMT_B};
use super::reader::DisplayRegs;
use super::vram::{Format, Surface, VramPtr, TILE};
use super::consts::{dispmode, lut, ODSY_DID_MAIN_SHIFT, ODSY_DID_OVERLAY_MASK};

/// The PB&J scan-out entries of the backend register file, in the order the
/// frame-buffer set-up writes them (traced in IRIS, `ignore/odsy/docs/measured.md`
/// section 4).
pub(super) mod pbj {
    pub const BASE: u32 = 0x2800;
    /// The overlay and window-id buffer's base, in 16-byte units.
    pub const OVERLAY_BASE: usize = 0;
    /// The frame buffer's base, in 16-byte units.
    pub const PIXEL_BASE: usize = 1;
    pub const WIDTH: usize = 2;
    pub const HEIGHT: usize = 3;
    /// Tiles in a row of tiles, shared by both buffers.
    pub const STRIDE: usize = 7;
    /// Frame-buffer bytes a pixel, 2 << code.
    pub const DEPTH: usize = 9;
}

/// The fields of the main display mode word (backend entries 0x2900 + id)
/// that place and decode a window's colour: bits 24:13 the pixel field
/// format (window pixel size, colour field size in 2-byte units, then the
/// offsets of buffer A's and buffer B's colour field in 2-byte units, three
/// bits each), bits 12:9 the display format. Decoded from the X server's
/// per-visual words, each matching the GL library's field offsets
/// (`ignore/odsy/docs/measured.md` section 9).
pub(super) mod mode {
    pub const FIELD_UNITS_SHIFT: u32 = 19;
    pub const A_SHIFT: u32 = 16;
    pub const B_SHIFT: u32 = 13;
    pub const FORMAT_SHIFT: u32 = 9;
    pub const FORMAT_MASK: u32 = 0xf;
    pub const RGB4: u32 = 0xc;
    pub const RGB5: u32 = 0xb;
    pub const RGB8: u32 = 0xa;
    pub const RGB10: u32 = 0x9;
    pub const RGB12: u32 = 0x6;
}

/// Where window `mode` keeps its colour, buffer A then buffer B (buffer A
/// shows while the window's displayed bit is clear), and how it reads.
pub(super) fn mode_fields(m: u32) -> [Field; 2] {
    let units = ((m >> mode::FIELD_UNITS_SHIFT) & 7) as u8;
    let codec = match (m >> mode::FORMAT_SHIFT) & mode::FORMAT_MASK {
        mode::RGB4 => Codec::Texel { bytes: 2, field: FIELD_FOUR },
        mode::RGB5 => Codec::Texel { bytes: 2, field: FIELD_PACKED },
        mode::RGB8 => Codec::Texel { bytes: 4, field: FIELD_FOUR },
        mode::RGB10 => Codec::Texel { bytes: 4, field: FIELD_PACKED },
        mode::RGB12 => Codec::Lanes,
        // Colour index, in the field's low bits.
        _ => Codec::Texel { bytes: (2 * units).clamp(2, 4), field: FIELD_INDEX },
    };
    let at = |shift: u32| 2 * ((m >> shift) & 7) as u8;
    [Field { at: at(mode::A_SHIFT), codec }, Field { at: at(mode::B_SHIFT), codec }]
}

/// Frame::parts().
pub(super) struct PixelParts {
    pub did: u32,
    pub overlay: u32,
    pub id: u32,
    pub fields: [u32; 2],
    pub shown: usize,
    pub rgb: u32,
    /// The frame-buffer pixel's bytes.
    pub bytes: Vec<u8>,
}

pub(super) struct Frame {
    pub width: u32,
    pub height: u32,
    pub(super) fb: Surface,
    /// Per main window id, its two colour fields (mode_fields()).
    pub(super) fields: [[Field; 2]; dispmode::COUNT as usize],
    pub(super) wid: Surface,
    pub(super) wid_layout: Layout,
    pub(super) ctx: ResolveCtx,
    /// Per main window id, how its shown field is read (`compose`).
    plans: [Plan; dispmode::COUNT as usize],
    /// The gamma table with each RGB mode's channel fold (`FOLDS`) applied:
    /// [fold][red, green, blue][channel byte], in the output's order.
    gamma: Box<[[[u32; 256]; 3]; 3]>,
    /// The lookup RAM's colours, in the output's order.
    lut: Box<[u32; lut::SIZE as usize]>,
    /// Whether `compose` writes the screen's order, 0xAABBGGRR with alpha
    /// set (`Frame::snapshot_for_screen`), not 0x00RRGGBB.
    abgr: bool,
}

/// 0x00RRGGBB as the screen's 0xAABBGGRR, opaque.
#[inline(always)]
pub(super) fn to_screen(rgb: u32) -> u32 {
    0xff00_0000 | rgb.swap_bytes() >> 8
}

/// A tile row of one window id, straight from memory: the frame (its
/// tables), the window's plan, the tile row's frame-buffer words (the first
/// pixel at the first word's top byte), and the 16 colours out.
type SpanFn = fn(&Frame, &Plan, &[u64], &mut [u32]);
/// Pixel i of a tile row: as SpanFn, for one pixel.
type PixelFn = fn(&Frame, &Plan, &[u64], usize) -> u32;

/// How one window id's shown colour field is read and shown, prepared once
/// a frame: the readers monomorphised for its codec, RGB or index, and the
/// frame buffer's pixel size; where in the pixel the field is; and what a
/// value becomes.
#[derive(Clone, Copy)]
struct Plan {
    span: SpanFn,
    one: PixelFn,
    /// The field's byte offset in the pixel.
    at: u64,
    /// RGB: which of `Frame::gamma` (the mode's channel width folded in).
    table: usize,
    /// Index: lookup entry = base + (value & mask).
    base: u32,
    mask: u32,
    /// The field, for the generic reader.
    field: Field,
}

/// The field readers by codec: the display modes `mode_fields` produces,
/// a field that lies past the end of the pixel (black), and any other field
/// through `Field::load`.
const K_RGB4: u32 = 0;
const K_RGB5: u32 = 1;
const K_RGB8: u32 = 2;
const K_RGB10: u32 = 3;
const K_LANES: u32 = 4;
const K_INDEX2: u32 = 5;
const K_INDEX4: u32 = 6;
const K_OUTSIDE: u32 = 7;
const K_GENERIC: u32 = 8;
const KINDS: usize = 9;

/// The value of the n bytes at byte offset b of `words`: `VramPtr::load`
/// on words already in hand, the lowest address highest. A field that runs
/// past its word reads as the board-memory load does in a release build.
#[inline(always)]
fn value(words: &[u64], b: u64, n: u32) -> u64 {
    let w = words[(b >> 3) as usize];
    let shift = 64u32.wrapping_sub(8 * ((b & 7) as u32 + n));
    if n == 8 { w } else { w.wrapping_shr(shift) & ((1u64 << (8 * n)) - 1) }
}

/// The colour of pixel i of a tile row whose pixels are 1 << LOG2 bytes
/// (LOG2 0: the frame's own size), read as codec K and shown RGB or through
/// the lookup RAM.
#[inline(always)]
fn colour<const K: u32, const RGB: bool, const LOG2: u32>(f: &Frame, p: &Plan, words: &[u64], i: usize) -> u32 {
    let log2 = if LOG2 == 0 { f.fb.bpp_log2 as u32 } else { LOG2 };
    let b = ((i as u64) << log2) + p.at;
    let word = match K {
        K_RGB4 => texel_unpack(value(words, b, 2), 2, FIELD_FOUR),
        K_RGB5 => texel_unpack(value(words, b, 2), 2, FIELD_PACKED),
        K_RGB8 => value(words, b, 4) as u32,
        K_RGB10 => texel_unpack(value(words, b, 4), 4, FIELD_PACKED),
        K_LANES => word_of(lanes_from(value(words, b, 6))),
        K_INDEX2 => value(words, b, 2) as u32,
        K_INDEX4 => value(words, b, 4) as u32,
        K_OUTSIDE => return if f.abgr { to_screen(0) } else { 0 },
        _ => p.field.load(value(words, b, p.field.value_len())),
    };
    if RGB {
        let [red, green, blue] = &f.gamma[p.table];
        red[(word >> 24) as usize] | green[((word >> 16) & 0xff) as usize] | blue[((word >> 8) & 0xff) as usize]
    } else {
        f.lut[(p.base + (word & p.mask)) as usize & (lut::SIZE as usize - 1)]
    }
}

fn span_k<const K: u32, const RGB: bool, const LOG2: u32>(f: &Frame, p: &Plan, words: &[u64], out: &mut [u32]) {
    let out: &mut [u32; TILE as usize] = out.try_into().unwrap();
    for (i, o) in out.iter_mut().enumerate() {
        *o = colour::<K, RGB, LOG2>(f, p, words, i);
    }
}

fn one_k<const K: u32, const RGB: bool, const LOG2: u32>(f: &Frame, p: &Plan, words: &[u64], i: usize) -> u32 {
    colour::<K, RGB, LOG2>(f, p, words, i)
}

/// The readers, [pixel size][codec][RGB]: pixel sizes 2, 4, 8 and 16 bytes
/// monomorphised, any other read at the frame's own.
macro_rules! readers_for {
    ($t:ident, $fty:ty, $log2:expr) => {
        [
            [$t::<K_RGB4, false, { $log2 }> as $fty, $t::<K_RGB4, true, { $log2 }>],
            [$t::<K_RGB5, false, { $log2 }>, $t::<K_RGB5, true, { $log2 }>],
            [$t::<K_RGB8, false, { $log2 }>, $t::<K_RGB8, true, { $log2 }>],
            [$t::<K_RGB10, false, { $log2 }>, $t::<K_RGB10, true, { $log2 }>],
            [$t::<K_LANES, false, { $log2 }>, $t::<K_LANES, true, { $log2 }>],
            [$t::<K_INDEX2, false, { $log2 }>, $t::<K_INDEX2, true, { $log2 }>],
            [$t::<K_INDEX4, false, { $log2 }>, $t::<K_INDEX4, true, { $log2 }>],
            [$t::<K_OUTSIDE, false, { $log2 }>, $t::<K_OUTSIDE, true, { $log2 }>],
            [$t::<K_GENERIC, false, { $log2 }>, $t::<K_GENERIC, true, { $log2 }>],
        ]
    };
}
static SPANS: [[[SpanFn; 2]; KINDS]; 5] =
    [readers_for!(span_k, SpanFn, 0), readers_for!(span_k, SpanFn, 1), readers_for!(span_k, SpanFn, 2), readers_for!(span_k, SpanFn, 3), readers_for!(span_k, SpanFn, 4)];
static ONES: [[[PixelFn; 2]; KINDS]; 5] =
    [readers_for!(one_k, PixelFn, 0), readers_for!(one_k, PixelFn, 1), readers_for!(one_k, PixelFn, 2), readers_for!(one_k, PixelFn, 3), readers_for!(one_k, PixelFn, 4)];

/// The channel folds of the RGB display modes (`ResolveCtx`): (keep, down).
const FOLDS: [(u32, u32); 3] = [(0xff, 8), (0xf0, 4), (0xf8, 5)];

impl Plan {
    /// Window `field` shown through `show`, in pixels of 1 << log2 bytes.
    fn of(field: Field, show: &IdShow, log2: u8) -> Plan {
        let k = if field.at as u32 + 2 > 1 << log2 {
            K_OUTSIDE
        } else {
            match field.codec {
                Codec::Lanes => K_LANES,
                Codec::Texel { bytes: 2, field: FIELD_FOUR } => K_RGB4,
                Codec::Texel { bytes: 2, field: FIELD_PACKED } => K_RGB5,
                Codec::Texel { bytes: 4, field: FIELD_FOUR } => K_RGB8,
                Codec::Texel { bytes: 4, field: FIELD_PACKED } => K_RGB10,
                Codec::Texel { bytes: 2, field: FIELD_INDEX } => K_INDEX2,
                Codec::Texel { bytes: 4, field: FIELD_INDEX } => K_INDEX4,
                _ => K_GENERIC,
            }
        } as usize;
        let size = if (1..=4).contains(&log2) { log2 as usize } else { 0 };
        let rgb = show.rgb as usize;
        let table = FOLDS.iter().position(|&f| f == (show.keep, show.down)).unwrap_or(0);
        Plan { span: SPANS[size][k][rgb], one: ONES[size][k][rgb], at: field.at as u64, table, base: show.base, mask: show.mask, field }
    }
}

impl Frame {
    /// The frame the registers describe, with buffer (displayed >> id) & 1
    /// showing for each window; None before the scan-out is set up.
    pub fn snapshot(regs: &DisplayRegs, displayed: u32) -> Option<Frame> {
        Self::snapshot_in(regs, displayed, false)
    }

    /// As `snapshot`, composing in the screen's order (`Rex3Screen::rgba`,
    /// 0xAABBGGRR): the display thread's frames, which then go to the
    /// screen as they are.
    pub fn snapshot_for_screen(regs: &DisplayRegs, displayed: u32) -> Option<Frame> {
        Self::snapshot_in(regs, displayed, true)
    }

    fn snapshot_in(regs: &DisplayRegs, displayed: u32, abgr: bool) -> Option<Frame> {
        let r = &regs.pbj;
        let (width, height, stride) = (r[pbj::WIDTH], r[pbj::HEIGHT], r[pbj::STRIDE]);
        if width == 0 || height == 0 || stride == 0 || width > stride * TILE {
            return None;
        }
        let fb_log2 = (r[pbj::DEPTH] & 7) as u8 + 1;
        let wid_format = Format { fb_log2: 1, win_log2: 1, pixel_fmt: PIXEL_FMT_B, ..Default::default() };
        let fb = Surface::new(r[pbj::PIXEL_BASE], stride, fb_log2);
        let fields: [[Field; 2]; dispmode::COUNT as usize] = std::array::from_fn(|id| mode_fields(regs.dispmode[id]));
        let ctx = ResolveCtx::new(regs, displayed);
        let plans = std::array::from_fn(|id| {
            let show = &ctx.ids[id];
            Plan::of(fields[id][show.buf & 1], show, fb.bpp_log2)
        });
        let mut gamma = Box::new([[[0u32; 256]; 3]; 3]);
        for (t, &(keep, down)) in gamma.iter_mut().zip(&FOLDS) {
            for c in 0..256u32 {
                let folded = ((c & keep) | (c >> down)) as usize & 0xff;
                for ch in 0..3 {
                    // Each channel's entry alone in its byte; alpha is set in
                    // all three, so their OR is opaque.
                    let v = ctx.gamma[ch][folded];
                    t[ch][c as usize] = if abgr { to_screen(v) } else { v };
                }
            }
        }
        Some(Frame {
            width,
            height,
            fb,
            fields,
            wid: Surface::new(r[pbj::OVERLAY_BASE], stride, 1),
            wid_layout: Layout::of(wid_format),
            lut: Box::new(std::array::from_fn(|i| if abgr { to_screen(ctx.rgb[i]) } else { ctx.rgb[i] })),
            ctx,
            plans,
            gamma,
            abgr,
        })
    }

    /// Whether `compose` writes the screen's order (`snapshot_for_screen`).
    pub fn is_screen_order(&self) -> bool {
        self.abgr
    }

    /// The frame buffer and the window-id buffer the scan-out reads.
    pub fn surfaces(&self) -> (Surface, Surface) {
        (self.fb, self.wid)
    }

    /// What pixel (x, y) is made of, for the monitor: its window-id and
    /// overlay bytes, its main window id, the two colour fields' words, the
    /// one the window shows and the colour shown (before the cursor).
    pub fn parts(&self, mem: VramPtr, x: u32, y: u32) -> Option<PixelParts> {
        let (w, p) = (self.wid.offset(x, y), self.fb.offset(x, y));
        if w.max(p) >= mem.size() {
            return None;
        }
        let did = field_word(mem, w, &self.wid_layout.front);
        let id = (did >> ODSY_DID_MAIN_SHIFT) as usize & (dispmode::COUNT as usize - 1);
        let f = &self.fields[id];
        let n = self.fb.bytes_per_pixel();
        let load = |f: &Field| if f.at as u32 + 2 <= n { field_word(mem, p, f) } else { 0 };
        Some(PixelParts {
            did,
            overlay: field_word(mem, w, &self.wid_layout.back),
            id: id as u32,
            fields: [load(&f[0]), load(&f[1])],
            shown: self.ctx.ids[id].buf & 1,
            rgb: self.pixel(mem, x, y),
            bytes: (0..n as u64).map(|i| mem.load(p + i, 1) as u8).collect(),
        })
    }

    /// One displayed pixel as 0x00RRGGBB, the reference `compose` is held
    /// to.
    #[inline]
    pub(super) fn pixel(&self, mem: VramPtr, x: u32, y: u32) -> u32 {
        let (w, p) = (self.wid.offset(x, y), self.fb.offset(x, y));
        let did = field_word(mem, w, &self.wid_layout.front);
        let ov = field_word(mem, w, &self.wid_layout.back);
        let (enabled, obase) = self.ctx.overlay[(did & ODSY_DID_OVERLAY_MASK) as usize];
        if enabled && ov != 0 {
            return self.ctx.rgb[(obase + ov) as usize & (lut::SIZE as usize - 1)];
        }
        let id = (did >> ODSY_DID_MAIN_SHIFT) as usize & (dispmode::COUNT as usize - 1);
        let show: &IdShow = &self.ctx.ids[id];
        let field = self.fields[id][show.buf & 1];
        if field.at as u32 + 2 > self.fb.bytes_per_pixel() {
            return 0;
        }
        let word = field_word(mem, p, &field);
        if show.rgb {
            let [red, green, blue] = &*self.ctx.gamma;
            let f = |c: u32| ((c & show.keep) | (c >> show.down)) as usize & 0xff;
            red[f(word >> 24)] | green[f((word >> 16) & 0xff)] | blue[f((word >> 8) & 0xff)]
        } else {
            self.ctx.rgb[(show.base + (word & show.mask)) as usize & (lut::SIZE as usize - 1)]
        }
    }

    /// The whole frame into `out` (rows of `stride` pixels, at most
    /// `rows` of them), the cursor over it.
    pub fn compose(&self, mem: VramPtr, regs: &DisplayRegs, out: &mut [u32], stride: usize, rows: usize) {
        let w = (self.width as usize).min(stride);
        let h = (self.height as usize).min(rows);
        // A band of 16 lines at a time, tile by tile: a tile is contiguous
        // in memory and so is a row of tiles, so the band reads memory in
        // order. Line by line, each tile row would be a jump to the next
        // tile, a page apart in a 16-byte frame buffer, which the hardware
        // prefetcher does not follow.
        for band in (0..h).step_by(TILE as usize) {
            let lines = (h - band).min(TILE as usize);
            for x in (0..w).step_by(TILE as usize) {
                let n = (w - x).min(TILE as usize);
                for y in band..band + lines {
                    let at = y * stride + x;
                    self.tile_row(mem, x as u32, y as u32, &mut out[at..at + n]);
                }
            }
            for y in band..band + lines {
                regs.cursor_row_in(y as u32, &mut out[y * stride..y * stride + w], w as u32, self.abgr);
            }
        }
    }

    /// Up to 16 pixels of line y from x, a tile row, before the cursor,
    /// read from board memory as words.
    #[inline(always)]
    fn tile_row(&self, mem: VramPtr, x: u32, y: u32, seg: &mut [u32]) {
        // The window-id buffer's pixels are 2 bytes: the id byte, then the
        // overlay byte (`Layout::of`, the B format), so a word holds four
        // pixels, the first in its top bytes. A span of one id and no
        // overlay byte is four copies of one pattern.
        const SPLAT: u64 = 0x0100_0100_0100_0100;
        const OVERLAYS: u64 = 0x00ff_00ff_00ff_00ff;
        // A tile row of the frame buffer: 16 pixels, whole words (a tile row
        // starts on a 32-byte boundary).
        let fb_words = ((TILE as usize) << self.fb.bpp_log2).div_ceil(8);
        let (wa, pa) = (self.wid.offset(x, y), self.fb.offset(x, y));
        let (Some(ids), Some(px)) = (mem.words(wa, 4), mem.words(pa, fb_words)) else {
            // A tile row that wraps the end of memory.
            for (i, o) in seg.iter_mut().enumerate() {
                let rgb = self.pixel(mem, x + i as u32, y);
                *o = if self.abgr { to_screen(rgb) } else { rgb };
            }
            return;
        };
        let lead = ids[0] >> 56;
        if seg.len() == TILE as usize && ids.iter().all(|&w| w & !OVERLAYS == lead * SPLAT) {
            // One window id: its reader for the whole tile row, then the
            // overlay bytes it shows over it, if any.
            let plan = &self.plans[(lead as u32 >> ODSY_DID_MAIN_SHIFT) as usize & (dispmode::COUNT as usize - 1)];
            (plan.span)(self, plan, px, seg);
            let (enabled, obase) = self.ctx.overlay[(lead as u32 & ODSY_DID_OVERLAY_MASK) as usize];
            if enabled && ids.iter().any(|&w| w & OVERLAYS != 0) {
                for (i, o) in seg.iter_mut().enumerate() {
                    let ov = (ids[i / 4] >> (48 - 16 * (i % 4))) as u32 & 0xff;
                    if ov != 0 {
                        *o = self.lut[(obase + ov) as usize & (lut::SIZE as usize - 1)];
                    }
                }
            }
            return;
        }
        for (i, o) in seg.iter_mut().enumerate() {
            let pair = (ids[i / 4] >> (48 - 16 * (i % 4))) as u32 & 0xffff;
            let (did, ov) = (pair >> 8, pair & 0xff);
            let (enabled, obase) = self.ctx.overlay[(did & ODSY_DID_OVERLAY_MASK) as usize];
            if enabled && ov != 0 {
                *o = self.lut[(obase + ov) as usize & (lut::SIZE as usize - 1)];
                continue;
            }
            let plan = &self.plans[(did >> ODSY_DID_MAIN_SHIFT) as usize & (dispmode::COUNT as usize - 1)];
            *o = (plan.one)(self, plan, px, i);
        }
    }
}

/// What the scan-out reads, saved for composing offline (`vpro scanout
/// save`, the compose bench): the registers, the displayed-buffer bits and
/// the memory under the frame buffer and the window-id buffer, as words.
/// Little-endian throughout; a format of this model's, not the board's.
#[cfg(test)]
pub(super) struct Capture {
    pub regs: DisplayRegs,
    pub displayed: u32,
    /// Board memory's size in bytes.
    pub size: u64,
    /// Runs of memory: the byte address of the first word, and the words.
    pub runs: Vec<(u64, Vec<u64>)>,
}

const CAPTURE_MAGIC: &[u8; 8] = b"VPSCAN01";

/// `Capture`'s bytes for the frame `frame` of memory `mem` and registers
/// `regs` with displayed-buffer bits `displayed` (`vpro scanout save`).
pub(super) fn save_capture(frame: &Frame, mem: VramPtr, regs: &DisplayRegs, displayed: u32) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    let u32s = |out: &mut Vec<u8>, v: &[u32]| {
        out.extend_from_slice(&(v.len() as u32).to_le_bytes());
        v.iter().for_each(|w| out.extend_from_slice(&w.to_le_bytes()));
    };
    out.extend_from_slice(CAPTURE_MAGIC);
    out.extend_from_slice(&mem.size().to_le_bytes());
    out.extend_from_slice(&displayed.to_le_bytes());
    u32s(&mut out, &regs.lut);
    u32s(&mut out, &regs.dispmode);
    u32s(&mut out, &regs.overlay_mode);
    u32s(&mut out, &regs.cursor);
    u32s(&mut out, &regs.pbj);
    let surfaces = [frame.fb, frame.wid];
    out.extend_from_slice(&(surfaces.len() as u32).to_le_bytes());
    for f in surfaces {
        // Every row of tiles the frame's rows touch.
        let start = f.offset(0, 0) & !7;
        let bytes = (frame.height.div_ceil(TILE) as u64 * f.pitch as u64 * (TILE * TILE) as u64) << f.bpp_log2;
        let n = bytes.div_ceil(8) as usize;
        let words = mem.words(start, n).ok_or("the surface wraps the end of board memory")?;
        out.extend_from_slice(&start.to_le_bytes());
        out.extend_from_slice(&(n as u64).to_le_bytes());
        words.iter().for_each(|w| out.extend_from_slice(&w.to_le_bytes()));
    }
    Ok(out)
}

#[cfg(test)]
impl Capture {
    pub fn load(b: &[u8]) -> Result<Capture, String> {
        struct Cursor<'a>(&'a [u8]);
        impl<'a> Cursor<'a> {
            fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
                let (head, rest) = self.0.split_at_checked(n).ok_or("the capture is cut short")?;
                self.0 = rest;
                Ok(head)
            }
            fn u32(&mut self) -> Result<u32, String> {
                Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
            }
            fn u64(&mut self) -> Result<u64, String> {
                Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
            }
            fn u32s(&mut self) -> Result<Vec<u32>, String> {
                let n = self.u32()? as usize;
                Ok(self.take(4 * n)?.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect())
            }
        }
        let mut c = Cursor(b);
        if c.take(8)? != CAPTURE_MAGIC {
            return Err("not a VPro scan-out capture".into());
        }
        let size = c.u64()?;
        let displayed = c.u32()?;
        let mut regs = DisplayRegs::default();
        regs.lut = c.u32s()?;
        let into = |dst: &mut [u32], v: Vec<u32>| dst.iter_mut().zip(v).for_each(|(d, s)| *d = s);
        into(&mut regs.dispmode, c.u32s()?);
        into(&mut regs.overlay_mode, c.u32s()?);
        into(&mut regs.cursor, c.u32s()?);
        into(&mut regs.pbj, c.u32s()?);
        let mut runs = Vec::new();
        for _ in 0..c.u32()? {
            let start = c.u64()?;
            let n = c.u64()? as usize;
            runs.push((start, c.take(8 * n)?.chunks_exact(8).map(|w| u64::from_le_bytes(w.try_into().unwrap())).collect()));
        }
        Ok(Capture { regs, displayed, size, runs })
    }
}

/// The word of field f in the pixel at byte address p.
#[inline(always)]
fn field_word(mem: VramPtr, p: u64, f: &Field) -> u32 {
    f.load(mem.load(p + f.at as u64, f.value_len()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::consts::ODSY_GAMMA_GROUP;
    use super::super::vram::Vram;

    #[test]
    fn composes_the_frame_buffer_through_its_window_id() {
        let mut regs = DisplayRegs::default();
        // 32 x 16, 16-byte pixels at 0x100, the id buffer at 0x2000.
        regs.pbj[pbj::OVERLAY_BASE] = 0x2000;
        regs.pbj[pbj::PIXEL_BASE] = 0x100;
        regs.pbj[pbj::WIDTH] = 32;
        regs.pbj[pbj::HEIGHT] = 16;
        regs.pbj[pbj::STRIDE] = 2;
        regs.pbj[pbj::DEPTH] = 3;
        // Window 1: RGB, an identity gamma; window 2 shows buffer 1.
        // The X server's 8:8:8:8 modes: single-buffered (colour at unit 2)
        // and double-buffered with depth (A at unit 6, B at unit 2).
        regs.dispmode[1] = 0x005215a6;
        regs.dispmode[2] = 0x009655a6;
        for id in [1, 2] {
            assert_eq!(regs.dispmode[id] >> dispmode::GROUP_SHIFT & dispmode::GROUP_MASK, ODSY_GAMMA_GROUP);
        }
        for t in 0..0x600u32 {
            let v = if t < 0x200 { t } else if t < 0x400 { (t - 0x100) * 2 } else { (t - 0x200) * 4 };
            let c = (v >> 4).min(255);
            regs.lut[(super::super::consts::ODSY_GAMMA_BASE + t) as usize] = c << lut::RED_SHIFT | c << lut::GREEN_SHIFT | c << lut::BLUE_SHIFT;
        }
        let frame = Frame::snapshot(&regs, 1 << 2).unwrap();
        let mut m = Vram::new(1 << 20);
        let fb = Surface::new(0x100, 2, 4);
        let wid = Surface::new(0x2000, 2, 1);
        // Drawn as the pixel mode of each window places it.
        let single = Layout::of(Format::decode(0x6950));
        let double = Layout::of(Format::decode(0x20406d50));
        let mem = m.ptr();
        let put = |f: &Field, x, y, w| mem.store(fb.offset(x, y) + f.at as u64, f.value_len(), f.store(w));
        put(&single.front, 3, 2, 0x40_80_c0_ff);
        put(&double.back, 20, 9, 0x10_20_30_ff);
        assert_eq!(mode_fields(0x009655a6)[1], double.back);
        assert_eq!(mode_fields(0x005215a6)[0], single.front);
        mem.store(wid.offset(3, 2), 1, 1 << ODSY_DID_MAIN_SHIFT);
        mem.store(wid.offset(20, 9), 1, 2 << ODSY_DID_MAIN_SHIFT);
        let mut out = vec![0u32; 32 * 16];
        frame.compose(mem, &regs, &mut out, 32, 16);
        assert_eq!(out[2 * 32 + 3], 0x40_80_c0);
        assert_eq!(out[9 * 32 + 20], 0x10_20_30);
    }
}
