//! HQ2 command engine: CPU-visible registers, microcode RAM, and the HLE
//! command interpreter that runs on the HQ2 thread.
//!
//! The HQ2 and GE7 microcode is stored and read back but never executed.
//! Each FIFO word index is the HQ2 microcode entry address (GR2.h), so the
//! interpreter dispatches on it directly and turns each command into RE3
//! register writes. Only the textport commands are implemented so far.

use std::sync::atomic::{AtomicU32, Ordering};

use super::ge7::{Ge7, LOADUCODE_MASK};
use super::re3;

#[path = "gl.rs"]
mod gl;

// Register byte offsets from the HQ2 base (0x6a000).
pub const HQ_ATTRJMP: u32 = 0x00; // 16 words
pub const HQ_VERSION: u32 = 0x40;
pub const HQ_NUMGE: u32 = 0x44;
pub const HQ_FIN1: u32 = 0x48;
pub const HQ_FIN2: u32 = 0x4c;
pub const HQ_DMASYNC: u32 = 0x50;
pub const HQ_FIFO_FULL_TIMEOUT: u32 = 0x54;
pub const HQ_FIFO_EMPTY_TIMEOUT: u32 = 0x58;
pub const HQ_FIFO_FULL: u32 = 0x5c;
pub const HQ_FIFO_EMPTY: u32 = 0x60;
pub const HQ_GE7LOADUCODE: u32 = 0x64;
pub const HQ_GEDMA: u32 = 0x68;
pub const HQ_HQ_GEPC: u32 = 0x6c;
pub const HQ_GEPC: u32 = 0x70;
pub const HQ_INTR: u32 = 0x74;
pub const HQ_UNSTALL: u32 = 0x78;
pub const HQ_MYSTERY: u32 = 0x7c;
pub const HQ_REFRESH: u32 = 0x80;
pub const HQ_FIN3: u32 = 0x100;

pub const HQ2_MYSTERY: u32 = 0xdead_beef;
/// HQ2 revision, reported in version[31:23] and hq_gepc[31:16].
pub const HQ2_REV: u32 = 2;
pub const HQ_UCODE_WORDS: usize = 8192;

/// FIFO low-watermark flag in `version`.
const VERSION_LOWATER: u32 = 1 << 13;
/// Finish flags in `version`: FIN1 = bit 2, FIN2 = bit 1, FIN3 = bit 0.
pub const FIN1: usize = 0;
pub const FIN2: usize = 1;
pub const FIN3: usize = 2;

// Textport command indices (FIFO word index = HQ2 microcode entry point).
pub const PUC_INIT: u32 = 401;
pub const PUC_COLOR: u32 = 402;
pub const PUC_FINISH: u32 = 403;
pub const PUC_PNT2I: u32 = 404;
pub const PUC_RECTI2D: u32 = 405;
pub const PUC_CMOV2I: u32 = 406;
pub const PUC_LINE2I: u32 = 407;
pub const PUC_DRAWCHAR: u32 = 408;
pub const PUC_RECTCOPY: u32 = 409;
pub const PUC_DATA: u32 = 479;

/// Pseudo FIFO index for words the CPU or VDMA writes to `HQ2_GEDMA`
/// (0x6a068). Real FIFO indices are 0..0x7fff; this marks the DMA data stream.
pub const HQ_TOKEN_GEDMA: u32 = 0x1_0000;
/// Pseudo FIFO index queued when the CPU writes `unstall`: the microcode
/// restarts, and its reset path takes one DATA word (the start argument).
pub const HQ_TOKEN_UNSTALL: u32 = 0x1_0001;
/// Context switch (GE_HQMSAV, 0x1f0): context id, then 2 DATA words.
pub const GE_HQMSAV: u32 = 496;
/// Context save / restore through HQ2_GEDMA (kernel _Gr2CXSaveRestore).
pub const GE_CX_SAVE_MAIN: u32 = 0x1e1;
pub const GE_CX_RESTORE_MAIN: u32 = 0x1e2;
pub const GE_CX_SAVE_EXT: u32 = 0x1e3;
pub const GE_CX_RESTORE_EXT: u32 = 0x1e4;
/// shram words the microcode fills in for Gr2PcxSwap: main context size in
/// words (0 = nothing to save), the owner (GE_HQMSAV id) of that state, and
/// the extended context size.
pub const SHRAM_CX_SIZE_MAIN: usize = 0x302;
pub const SHRAM_CX_OWNER: usize = 0x303;
pub const SHRAM_CX_SIZE_EXT: usize = 0x304;
/// Words of one saved GL context (the image streamed through HQ2_GEDMA).
pub const CX_WORDS: usize = gl::CX_WORDS;
/// Undocumented kernel-to-microcode requests used by Gr2PcxSwap (one word
/// each, fifo offsets 0x780 / 0x798); both finish with FIN2.
pub const HQ_PCX_1E0: u32 = 0x1e0;
pub const HQ_PCX_1E6: u32 = 0x1e6;
/// GL microcode request that completes by raising FIN3: Xsgi sends it and
/// spins on version bit 0, then clears FIN3 via 0x6b000 (see HQ2.h).
pub const HQ_GL_FIN3: u32 = 0x155;
/// OpenGL tokens (libglcore EXPRESS `gr2_hq2_fifo`: struct offset X is FIFO
/// index (X - 0x2000) / 4).
/// glFinish / SwapBuffers: `readback_trigger` (0x228C); the client then spins
/// on version bit 0 (FIN3) and acks at the FIN3 port.
pub const GL_FINISH: u32 = 0xa3;
/// glFlush / SwapBuffers: `swap_sync` (0x2018).
pub const GL_FLUSH: u32 = 0x006;
/// SwapBuffers: `swap_buffers` (0x279C) = swap interval, then ioctl 1005.
pub const GL_SWAP_BUFFERS: u32 = 0x1e7;

// ── 2D (Xsgi DDX) commands, GL microcode (HQ2.h section 8) ─────────────────
pub const HQ2_2D_BEGIN: u32 = 0x12c; // 0x404b0, sent first by expInitHW (1 word)
pub const HQ2_2D_SOLID_RECT: u32 = 304; // stream: 4 DATA words per box
pub const HQ2_2D_COLOR_AUX: u32 = 309; // aux-plane write mask
/// Colour for CLEAR bits of glyphs/mono images/stipples when drawn opaque
/// (HQ2.h called it FG_COLOR; the DDX writes the GC background here).
pub const HQ2_2D_COLOR_OFF: u32 = 311;
/// Colour for SET bits of glyphs/mono images/stipples (HQ2.h called it
/// BG_COLOR; the DDX writes the text foreground here).
pub const HQ2_2D_COLOR_ON: u32 = 312;
pub const HQ2_2D_MODE: u32 = 331; // visual/plane select (| 0x1000)
pub const HQ2_2D_ROP: u32 = 332; // fg; DATA planemask, alu, flag
pub const HQ2_2D_CID_WRITE: u32 = 336; // (cid << 8) | 0xF000, 0 = off
/// Pixel format and alignment for the next image transfer: [format (2 =
/// 8-bit, 1 = 16-bit, 0 = 32-bit per pixel, as TILE_SETUP), first pixel's
/// slot in the first word of each row] (expReadImage*, expDrawImage*).
pub const HQ2_2D_BUF_SELECT: u32 = 337;
/// Screen-to-host readback: words per row; DATA x, y (X-style, top-down),
/// width, rows. The microcode packs the rows into shram from word
/// READ_IMAGE_SHRAM (format and alignment from BUF_SELECT, MSB = first
/// pixel) and raises FIN3 (expReadImage).
pub const HQ2_2D_READ_IMAGE: u32 = 344;
/// First shram word of the READ_IMAGE result (Xsgi reads base + 0x1B000).
pub const READ_IMAGE_SHRAM: usize = 0x6C00;
pub const HQ2_2D_END_PRIMITIVE: u32 = 490;
/// Tile setup: size; DATA width, height, format (2), 0xD022 (expTileRects).
pub const HQ2_2D_TILE_SETUP: u32 = 314;
/// Tile data chunks (64/32/16/8-word loaders): words append to the tile.
pub const HQ2_2D_TILE_DATA_FIRST: u32 = 315;
pub const HQ2_2D_TILE_DATA_LAST: u32 = 318;
/// Tiled fill: DATA origin x, origin y, then boxes (x1, y1, x2, y2).
pub const HQ2_2D_TILE_RECT: u32 = 321;
/// Same, used when the tile width is not a whole number of words (expTileRects).
pub const HQ2_2D_TILE_RECT_ODD: u32 = 322;
/// 1-bpp images: x; DATA y, width, rows; then a fixed data block.
pub const HQ2_2D_MONO_IMAGE_8: u32 = 349;
pub const HQ2_2D_MONO_IMAGE_16: u32 = 350;
pub const HQ2_2D_MONO_IMAGE_32: u32 = 351;
pub const HQ2_2D_MONO_IMAGE_64: u32 = 352;
/// Host-to-screen image (expDrawImage): x; DATA y, width, rows, words/row,
/// format, left skip; then rows*words/row pixel words, padded with
/// 0xDEADBEEF to the chunk size. Streamed.
pub const HQ2_2D_DRAW_IMAGE: u32 = 342;
pub const HQ2_2D_DRAW_IMAGE_SMALL: u32 = 343;
/// Host-to-screen pixel DMA (kernel _Gr2DMAtrigger / _Gr2MCDMAtrigger):
/// fifo[0x147] = x; then 6 words through FIFO index 0: y, width (pixels),
/// height, words per row, flag, 0; then height * words_per_row pixel words
/// through HQ2_GEDMA (VDMA). The kernel waits for FIN2 after the transfer.
pub const HQ2_DMA_WRITE_PIXELS: u32 = 327;
/// GL-microcode pixel DMA (IRIS GL lrectwrite, libgl gl_gr2dma_lrectwrite):
/// 0x0B5 at pixel zoom 1, 0x0B8 zoomed. Same kernel protocol and header as
/// 0x147, drawn in the current GL context (window-relative, GL y up).
pub const HQ2_GL_DMA_WRITE: u32 = 0x0b5;
pub const HQ2_GL_DMA_WRITE_ZOOM: u32 = 0x0b8;
/// Pixel DMA reads (HQ2.h "PIXEL DMA DIRECTION"): the same kernel ioctl and
/// header as the writes, but the VDMA reads height * words/row words from
/// HQ2_GEDMA; FIN2 when done. 0x152 = Xsgi expReadImage* (screen, X-style
/// y = top row, format from BUF_SELECT); 0x0AC = GL lrectread /
/// glReadPixels (window-relative, y = bottom row, source from 0x10A).
pub const HQ2_2D_DMA_READ_PIXELS: u32 = 0x152;
pub const HQ2_GL_DMA_READ: u32 = 0x0ac;

/// Pixel rectangles streamed by the kernel's pixel DMA (_Gr2DMAtrigger):
/// x on the token; y, width, height, words/row, flag, 0 on GE_DATA; then
/// the pixel words through HQ2_GEDMA; FIN2 when done.
fn is_pixel_dma(cmd: u32) -> bool {
    matches!(cmd, HQ2_DMA_WRITE_PIXELS | HQ2_GL_DMA_WRITE | HQ2_GL_DMA_WRITE_ZOOM)
}
/// Screen-to-screen copy: pitch; DATA chunk, srcX, srcY, w, h, dstX, dstY
/// (X-style y). Layout from HQ2.h section 8 (unverified against a trace).
pub const HQ2_2D_COPY_RECT: u32 = 340;
/// Glyphs (TE/CW text): x; DATA y, width, rows; then 8/16/32/64 data words.
pub const HQ2_2D_GLYPH_8: u32 = 323;
pub const HQ2_2D_GLYPH_64: u32 = 326;
/// Fill spans: stream of (x, y, width) triples, padded with (1280, 1024, 1).
pub const HQ2_2D_POLY_SPAN: u32 = 305;
/// Line segments: 0; then a stream of (x1, y1, x2, y2) quads until the next
/// command (END_PRIMITIVE); padding quads have x1 = x2 = 1280 (expLineSS,
/// expPolySegment; 4Dwm's move/resize frame is 16 of them).
pub const HQ2_2D_LINE_SEG: u32 = 301;
/// Line mode word (expLineSS writes 0xA).
pub const HQ2_2D_LINE_MODE: u32 = 328;
/// Line clip box: x1; DATA x2, y1, y2 (inclusive).
pub const HQ2_2D_LINE_CLIP: u32 = 330;
/// Polyline: stream of (x, y) points (0x165 on some configurations).
pub const HQ2_2D_POLYLINE: u32 = 302;
pub const HQ2_2D_POLYLINE_ALT: u32 = 357;
/// Segments: stream of (x1, y1, x2, y2), padded with (1280, 1024) pairs.
pub const HQ2_2D_SEGMENTS: u32 = 345;
/// Stipple opaque flag (1 = zero bits get bg).
pub const HQ2_2D_STIPPLE_AUX: u32 = 313;
/// Stippled box: 0; DATA origin x, x1, origin y, y1, x2, y2.
pub const HQ2_2D_STIPPLE_RECT: u32 = 348;
/// Stippled boxes from expStippledFillRects, same 7 words as STIPPLE_RECT,
/// stipple loaded through TILE_SETUP (format 3) + TILE_DATA. 319 is used for
/// power-of-two stipple widths, 320 for the rest.
pub const HQ2_2D_STIPPLE_BOX_POW2: u32 = 319;
pub const HQ2_2D_STIPPLE_BOX: u32 = 320;
/// Stippled spans (expStippledFillSpans): 0; stream of (x, y, w, pattern)
/// quads, y top-down. The pattern is pre-rotated so its MSB is pixel x and
/// repeats every 32 pixels. 346 for stipples narrower than 32 / multi-word,
/// 347 for 32-wide stipples. COLOR_ON = set bits, STIPPLE_AUX 1 = opaque
/// with COLOR_OFF.
pub const HQ2_2D_STIPPLED_SPAN_A: u32 = 346;
pub const HQ2_2D_STIPPLED_SPAN_B: u32 = 347;
/// Off-screen sentinel coordinates the DDX pads streams with.
const PAD_X: u32 = 1280;
const PAD_Y: u32 = 1024;
/// Max pixel words per image row kept (1280 8-bit pixels = 320 words).
const IMG_ROW_WORDS: usize = 1280;
/// Max tile words kept (a 64x64 8-bit tile is 1024 words; larger is clipped).
/// Tile/stipple store. expTileRects sends tiles of up to 0x1300 words
/// (words per row * rows <= 0x1300; 4Dwm's 85x67 8-bit icon images need
/// 1474), padded to a multiple of 8 words.
const TILE_WORDS: usize = 0x1308;

/// 2D drawing state set by the MODE/ROP/COLOR_AUX/... commands.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct State2d {
    pub mode: u32,
    pub aux_mask: u32,
    /// ROP fg: colour for solid fills, spans, lines.
    pub fg: u32,
    /// 0x138: set bits of 1-bpp primitives.
    pub color_on: u32,
    /// 0x137: clear bits of 1-bpp primitives when opaque.
    pub color_off: u32,
    pub planemask: u32,
    pub alu: u32,
    pub flag: u32,
    pub cid_write: u32,
    pub buf_select: u32,
    pub buf_offset: u32,
}

/// How a screen-to-host read turns a VRAM word into a pixel value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ReadDecode {
    /// Xsgi: the plane group selected by the 2D MODE (`read_value`).
    Planes2d,
    /// GL 24-bit RGB: 0xAABBGGRR, alpha 0xFF (no alpha planes).
    Rgb24,
    /// GL 12-bit RGB in `bank` (0 = bits 11:0, 1 = bits 23:12), nibbles
    /// widened to 8 bits (x 17), as Rgb24.
    Rgb12 { bank: u32 },
    /// GL colour index in 12-bit `bank`.
    Ci12 { bank: u32 },
    /// GL depth: Z right-justified (libglcore shifts it up by 32 - bits).
    Depth,
}

/// Where a screen-to-host read goes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ReadDest {
    /// READ_IMAGE: shram from READ_IMAGE_SHRAM, then FIN3.
    Shram,
    /// Pixel DMA read: the HQ2_GEDMA read port, then FIN2.
    Gedma,
}

/// A screen-to-host read (READ_IMAGE, 0x152, 0x0AC), as the sink needs it.
#[derive(Clone, Copy)]
pub struct ReadImage {
    /// Left x and GL row of the top line (y = 0 at the bottom); rows go
    /// down from there.
    pub x: i32,
    pub top: i32,
    pub w: u32,
    pub rows: u32,
    pub words_per_row: u32,
    /// Pixel format (buf_select: 2 = 8-bit, 1 = 16-bit, 0 = 32-bit) and
    /// first-pixel slot (buf_offset); the planes for Planes2d.
    pub s2d: State2d,
    pub decode: ReadDecode,
}

impl ReadImage {
    pub fn per_word(&self) -> u32 {
        match self.s2d.buf_select { 2 => 4, 1 => 2, _ => 1 }
    }

    /// Pack the rectangle (as `read` returns VRAM words) into `dst`:
    /// rows of `words_per_row` words, MSB = first pixel, the first pixel
    /// of each row in slot `buf_offset`.
    pub fn pack(&self, read: impl Fn(i32, i32) -> u32, dst: &mut [u32]) {
        let per_word = self.per_word();
        let bits = 32 / per_word;
        let mask = if bits == 32 { u32::MAX } else { (1 << bits) - 1 };
        let off = self.s2d.buf_offset.min(per_word - 1) as i32;
        let wpr = self.words_per_row as usize;
        for r in 0..self.rows as usize {
            let gy = self.top - r as i32;
            for k in 0..wpr {
                let Some(d) = dst.get_mut(r * wpr + k) else { return };
                let mut word = 0;
                for p in 0..per_word {
                    let i = (k as u32 * per_word + p) as i32 - off;
                    if i >= 0 && (i as u32) < self.w {
                        let v = self.value(read(self.x + i, gy)) & mask;
                        word |= v << (32 - bits * (p + 1));
                    }
                }
                *d = word;
            }
        }
    }
}

impl ReadImage {
    /// The pixel value of VRAM word `vram` (for Depth: the Z buffer word).
    pub fn value(&self, vram: u32) -> u32 {
        let wide = |n: u32| (n & 0xf) * 17;
        match self.decode {
            ReadDecode::Planes2d => read_value(&self.s2d, vram),
            ReadDecode::Rgb24 => 0xff00_0000 | (vram & 0x00ff_ffff),
            ReadDecode::Rgb12 { bank } => {
                let v = (vram >> (12 * (bank & 1))) & 0xfff;
                0xff00_0000 | wide(v) | (wide(v >> 4) << 8) | (wide(v >> 8) << 16)
            }
            ReadDecode::Ci12 { bank } => (vram >> (12 * (bank & 1))) & 0xfff,
            ReadDecode::Depth => vram & 0x00ff_ffff,
        }
    }
}

/// The value a VRAM word holds for the active plane group (the inverse of
/// `value2d`): 4-bit overlay (MODE 8), 2-bit overlay or popup (MODE 0xB),
/// or the colour planes (R in bits 7:0).
pub fn read_value(s: &State2d, vram: u32) -> u32 {
    let aux = (vram >> 24) & 0xf;
    match s.mode & 0xff {
        8 => aux,
        0xb => if s.aux_mask & 0xc != 0 { (aux >> 2) & 3 } else { aux & 3 },
        // 12-bit RGB (expReadImage12TC: MODE 0x1002, ROP flag = the
        // window's buffer * 8): the bank the flag selects, R 3:0, G 7:4,
        // B 11:8, as 8:8:8 with each nibble replicated. The DDX takes the
        // low nibble of each byte back ((w & 0xF) | (w & 0xF00) >> 4 |
        // (w & 0xF0000) >> 8); expDrawImage12TC sends the high nibbles.
        2 => {
            let v = (vram >> (12 * ((s.flag >> 3) & 1))) & 0xfff;
            let wide = |n: u32| (n & 0xf) * 17;
            wide(v) | (wide(v >> 4) << 8) | (wide(v >> 8) << 16)
        }
        _ => vram & 0x00ff_ffff,
    }
}

/// Name of a FIFO word index: textport (PUC_*), GL/X (GE_*) or context-switch
/// tokens from HQ2.h. Indices are microcode entry points, so the meaning
/// depends on the loaded microcode; these are the documented ones.
pub fn token_name(index: u32) -> Option<&'static str> {
    Some(match index {
        PUC_INIT => "PUC_INIT", PUC_COLOR => "PUC_COLOR", PUC_FINISH => "PUC_FINISH",
        PUC_PNT2I => "PUC_PNT2I", PUC_RECTI2D => "PUC_RECTI2D", PUC_CMOV2I => "PUC_CMOV2I",
        PUC_LINE2I => "PUC_LINE2I", PUC_DRAWCHAR => "PUC_DRAWCHAR", PUC_RECTCOPY => "PUC_RECTCOPY",
        PUC_DATA => "DATA", HQ_TOKEN_GEDMA => "GEDMA", HQ_TOKEN_UNSTALL => "UNSTALL",
        HQ2_DMA_WRITE_PIXELS => "DMA_WRITE_PIXELS",
        HQ_PCX_1E0 => "PCX_1E0", HQ_PCX_1E6 => "PCX_1E6", HQ_GL_FIN3 => "2D_SYNC",
        GL_FINISH => "GL_FINISH", GL_FLUSH => "GL_FLUSH", GL_SWAP_BUFFERS => "GL_SWAP_BUFFERS",
        HQ2_2D_BEGIN => "2D_BEGIN", HQ2_2D_LINE_SEG => "2D_LINE_SEG", 303 => "2D_POINTS",
        HQ2_2D_SOLID_RECT => "2D_SOLID_RECT", 305 => "2D_POLY_SPAN", 307 => "2D_LINE",
        HQ2_2D_COLOR_AUX => "2D_COLOR_AUX", HQ2_2D_COLOR_OFF => "2D_COLOR_OFF", HQ2_2D_COLOR_ON => "2D_COLOR_ON",
        313 => "2D_STIPPLE_AUX", HQ2_2D_TILE_SETUP => "2D_TILE_SETUP", 315..=318 => "2D_TILE_DATA",
        HQ2_2D_STIPPLE_BOX_POW2 | HQ2_2D_STIPPLE_BOX => "2D_STIPPLE_BOX", HQ2_2D_TILE_RECT | HQ2_2D_TILE_RECT_ODD => "2D_TILE_RECT",
        HQ2_2D_POLYLINE => "2D_POLYLINE", HQ2_2D_POLYLINE_ALT => "2D_POLYLINE", HQ2_2D_SEGMENTS => "2D_SEGMENTS",
        323 => "2D_GLYPH_8", 324 => "2D_GLYPH_16", 325 => "2D_GLYPH_32", 326 => "2D_GLYPH_64",
        HQ2_2D_LINE_MODE => "2D_LINE_MODE", HQ2_2D_LINE_CLIP => "2D_LINE_CLIP", HQ2_2D_MODE => "2D_MODE", HQ2_2D_ROP => "2D_ROP",
        HQ2_2D_CID_WRITE => "2D_CID_WRITE", HQ2_2D_BUF_SELECT => "2D_BUF_SELECT",
        HQ2_2D_DMA_READ_PIXELS => "2D_DMA_READ_PIXELS", HQ2_GL_DMA_READ => "GL_DMA_READ",
        340 => "2D_COPY_RECT", 342 => "2D_DRAW_IMAGE", 343 => "2D_DRAW_IMAGE_SMALL",
        HQ2_2D_READ_IMAGE => "2D_READ_IMAGE", 346 | 347 => "2D_STIPPLED_SPAN", HQ2_2D_STIPPLE_RECT => "2D_STIPPLE_RECT",
        349 => "2D_MONO_IMAGE_8", 350 => "2D_MONO_IMAGE_16", 351 => "2D_MONO_IMAGE_32",
        352 => "2D_MONO_IMAGE_64", 354 => "2D_FAST_LINE_AUX", HQ2_2D_END_PRIMITIVE => "2D_END_PRIMITIVE",
        0 => "GE_DATA", 1 => "GE_INIT", 10 => "GE_COLOR", 11 => "GE_FINISH0", 12 => "GE_PNT2I",
        13 => "GE_SBOXI", 14 => "GE_CMOV2I", 15 => "GE_DRAWCHAR", 20 => "GE_READPIXELS",
        21 => "GE_WRITEPIXELS", 22 => "GE_LOADRE", 26 => "GE_PIXWRITEMASK", 29 => "GE_READBLOCK",
        30 => "GE_RECTREAD", 32 => "GE_READPIXDMA", 35 => "GE_RECTCOPY", 36 => "GE_RGBCOLOR",
        38 => "GE_RWMODE", 39 => "GE_SCREENCLEAR", 42 => "GE_WRITEBLOCK", 43 => "GE_RECTWRITE",
        44 => "GE_WRITEPIXDMA", 46 => "GE_ZBUFFER", 47 => "GE_ZCLEAR", 48 => "GE_ZOOMFACTOR",
        49 => "GE_READSOURCE", 50 => "GE_DRAWMODE", 51 => "GE_CZCLEAR", 61 => "GE_POLYGON",
        62 => "GE_ENDPOLYGON", 82 => "GE_LINESTYLE", 83 => "GE_LINEWIDTH", 84 => "GE_VERTEX2I",
        86 => "GE_VERTEX3I", 87 => "GE_VERTEX3F", 90 => "GE_MOVE2I", 92 => "GE_DRAW2I",
        94 => "GE_RDRAW2I", 96 => "GE_PATTERN", 101 => "GE_SBOXF", 102 => "GE_SBOXFI",
        105 => "GE_SBOX", 108 => "GE_SCRMASK", 109 => "GE_RASTEROP",
        481 => "GE_CX_SAVE_MAIN", 482 => "GE_CX_RESTORE_MAIN", 483 => "GE_CX_SAVE_EXT",
        484 => "GE_CX_RESTORE_EXT", 496 => "GE_HQMSAV", 511 => "DIAG_CTXSW",
        _ => return None,
    })
}

/// OpenGL microcode tokens (FIFO index bits 8:0), named after their libglcore
/// EXPRESS emitters (HQ2.h "OpenGL token addressing").
pub fn gl_token_name(tok: u32) -> Option<&'static str> {
    Some(match tok {
        0x003 => "GL_PIXWRITEMASK", 0x004 => "GL_MAKECURRENT", 0x006 => "GL_FLUSH",
        0x00a => "GL_DEPTH_MASK", 0x00e => "GL_STENCIL_CONFIG", 0x00f => "GL_STENCIL_MODE",
        0x014 => "GL_DEPTH_TEST", 0x01a => "GL_LOGIC_OP", 0x01e => "GL_POLYGON_STIPPLE_OFF",
        0x01f => "GL_POLYGON_STIPPLE", 0x024 => "GL_DEPTH_FUNC", 0x026 => "GL_BLEND_MODE",
        0x02b => "GL_FRAGMENT", 0x0a1 => "GL_CLEAR_STENCIL", 0x0bd => "GL_READ_DONE",
        0x0e9 => "GL_GET_COLOR", 0x0ea => "GL_GET_NORMAL", 0x107 => "GL_GET_RASTERPOS",
        0x010 => "GL_STENCIL_WMASK", 0x011 => "GL_DITHER", 0x013 => "GL_SHADE_MODEL",
        0x017 => "GL_LINE_WIDTH", 0x018 => "GL_LINE_STIPPLE", 0x01b => "GL_CULL_A", 0x01c => "GL_CULL_B",
        0x021 => "GL_POINT_SMOOTH", 0x022 => "GL_LINE_SMOOTH", 0x025 => "GL_BLEND_FACTOR",
        0x028 => "GL_FOG", 0x037 => "GL_MODELVIEW", 0x038 => "GL_PROJECTION", 0x039 => "GL_NORMAL_MATRIX",
        0x03a => "GL_TEXTURE_MATRIX", 0x03b => "GL_VIEWPORT", 0x03c => "GL_SCISSOR",
        0x046 => "GL_BEGIN_TSTRIP_TFAN", 0x047 => "GL_VTX_TSTRIP", 0x04a => "GL_END_TSTRIP_TFAN",
        0x04c => "GL_BEGIN_QSTRIP", 0x04d => "GL_VTX_QSTRIP", 0x051 => "GL_END_QSTRIP",
        0x056 => "GL_VTX_LSTRIP", 0x057 => "GL_END_LINES", 0x058 => "GL_BEGIN_LLOOP",
        0x059 => "GL_VTX_LLOOP", 0x05a => "GL_END_LLOOP", 0x063 => "VERTEX", 0x065 => "GL_VTX_OUTSIDE",
        0x074 => "GL_SPEC_LUT", 0x076..=0x07d => "GL_MATERIAL", 0x07f..=0x085 => "GL_LIGHT",
        0x0a0 => "GL_CLEAR_COLOR_DEPTH", 0x0a3 => "GL_FINISH", 0x0d4 => "GL_LIGHTPATH", 0x0e2 => "GL_POLYGON_MODE",
        0x0e3 => "GL_NORMALIZE", 0x0e4 => "GL_POLYGON_OFFSET", 0x0e6 => "GL_CONTEXT_FLUSH",
        0x0eb => "GL_VTX_LINES", 0x0ed => "GL_VTX_TFAN",
        0x0f0 => "GL_BEGIN_TRIANGLES", 0x0f1 => "GL_END_TRIANGLES", 0x0f2 => "GL_VTX_TRIANGLES",
        0x0f3 => "GL_BEGIN_QUADS", 0x0f4 => "GL_END_QUADS", 0x0f5 => "GL_VTX_QUADS",
        0x0f6 => "GL_EDGE_FLAG", 0x0f7 => "GL_BEGIN_POLYGON", 0x0f8 => "GL_VTX_POLYGON",
        0x0fc | 0x0fd => "GL_END_POLYGON", 0x104 => "GL_CLEAR_COLOR", 0x108 => "GL_FRONT_FACE",
        0x10a => "GL_READ_BUFFER", 0x10b => "GL_COLOR_WRITEMASK", 0x10d => "GL_NORMAL",
        0x10e => "GL_LIGHTPATH", 0x11d => "GL_BEGIN_POINTS", 0x123 => "GL_VTX_POINTS",
        0x168 => "GL_END_POINTS", 0x17c => "GL_BEGIN_LINES", 0x182 => "COLOR",
        0x1e5 => "CX_RESTORE_1E5", 0x1e7 => "GL_SWAP_BUFFERS",
        _ => return None,
    })
}

/// Trace label for any FIFO index: a documented name, or the GL token with
/// its HQ2 modifier bits (ITOF, conversion, LOADV, USEV) decoded.
pub fn index_label(index: u32) -> String {
    // GL names first: the IrisGL-era GE_* names HQ2.h lists for low indices
    // do not apply to the OpenGL microcode and are only used above 0x12B
    // (2D, textport, kernel tokens) or for the data ports.
    if let Some(n) = gl_token_name(index) {
        return n.to_string();
    }
    if let Some(n) = token_name(index).filter(|_| index >= 0x12c || index == 0) {
        return n.to_string();
    }
    let tok = index & 0x1ff;
    let mut mods: Vec<&str> = Vec::new();
    if index & 0x4000 != 0 { mods.push("ITOF"); }
    match (index >> 11) & 7 {
        1 => mods.push("V3"), 2 => mods.push("V2"), 3 => mods.push("C3"), 4 => mods.push("C4"),
        5 => mods.push("CP"), 6 => mods.push("C1"), 7 => mods.push("CONV7"), _ => {}
    }
    if index & 0x400 != 0 { mods.push("LOADV"); }
    if index & 0x200 != 0 { mods.push("USEV"); }
    let base = gl_token_name(tok).or_else(|| token_name(tok).filter(|_| tok >= 0x12c || tok == 0))
        .map(str::to_string).unwrap_or_else(|| format!("tok{tok:#x}"));
    if mods.is_empty() { base } else { format!("{}:{base}", mods.join("|")) }
}

/// Screen height the textport commands assume (1280x1024 CRT).
const SCREEN_H: i32 = 1024;
const MAX_ARGS: usize = 72;
/// `need` value for an open (variable-length, unknown) command.
const OPEN: u32 = u32::MAX;
/// `need` value for a streaming command: DATA words are consumed as they
/// arrive (e.g. 4 per box for SOLID_RECT) until the next command.
const STREAM: u32 = u32::MAX - 1;
const FIFO_WORDS: usize = 0x8000;

/// CPU-visible HQ2 registers and microcode RAM. Plain data, zero-valid.
#[repr(C)]
pub struct Hq2Regs {
    pub attrjmp: [u32; 16],
    pub numge: u32,
    /// Finish flags FIN1..FIN3. The CPU clears them; the HQ2 thread sets
    /// them when the microcode would, so they are atomics.
    pub fin: [AtomicU32; 3],
    pub dmasync: u32,
    pub fifo_full_timeout: u32,
    pub fifo_empty_timeout: u32,
    pub fifo_full: u32,
    pub fifo_empty: u32,
    /// Readback latch for `ge7loaducode` (reloaded by a `gepc` write).
    pub loaducode: u32,
    pub gepc: u32,
    pub intr: u32,
    pub refresh: u32,
    /// Set by a write to `unstall`: the engine is running microcode.
    pub running: u32,
    pub ucode: [u32; HQ_UCODE_WORDS],
}

impl Hq2Regs {
    pub fn read(&self, off: u32) -> u32 {
        match off {
            o if o < HQ_VERSION => self.attrjmp[(o >> 2) as usize & 15],
            HQ_VERSION => {
                let f = |i: usize| (self.fin[i].load(Ordering::Acquire) != 0) as u32;
                (HQ2_REV << 23) | VERSION_LOWATER | (f(FIN1) << 2) | (f(FIN2) << 1) | f(FIN3)
            }
            HQ_NUMGE => self.numge,
            HQ_FIN1 => self.fin[FIN1].load(Ordering::Acquire),
            HQ_FIN2 => self.fin[FIN2].load(Ordering::Acquire),
            HQ_DMASYNC => self.dmasync,
            HQ_FIFO_FULL_TIMEOUT => self.fifo_full_timeout,
            HQ_FIFO_EMPTY_TIMEOUT => self.fifo_empty_timeout,
            HQ_FIFO_FULL => self.fifo_full,
            HQ_FIFO_EMPTY => self.fifo_empty,
            HQ_GE7LOADUCODE => self.loaducode & LOADUCODE_MASK,
            // The kernel reads the HQ2 revision from hq_gepc[31:16] (HQ2.h).
            HQ_HQ_GEPC => (HQ2_REV << 16) | (self.gepc & 0xffff),
            HQ_GEPC => self.gepc,
            HQ_INTR => self.intr,
            HQ_MYSTERY => HQ2_MYSTERY,
            HQ_REFRESH => self.refresh,
            HQ_FIN3 => self.fin[FIN3].load(Ordering::Acquire),
            _ => 0,
        }
    }

    pub fn write(&mut self, off: u32, val: u32, ge: &mut Ge7) {
        match off {
            o if o < HQ_VERSION => self.attrjmp[(o >> 2) as usize & 15] = val,
            HQ_NUMGE => self.numge = val,
            HQ_FIN1 => self.fin[FIN1].store(val, Ordering::Release),
            HQ_FIN2 => self.fin[FIN2].store(val, Ordering::Release),
            HQ_DMASYNC => self.dmasync = val,
            HQ_FIFO_FULL_TIMEOUT => self.fifo_full_timeout = val,
            HQ_FIFO_EMPTY_TIMEOUT => self.fifo_empty_timeout = val,
            HQ_FIFO_FULL => self.fifo_full = val,
            HQ_FIFO_EMPTY => self.fifo_empty = val,
            HQ_GE7LOADUCODE => {
                self.loaducode = val;
                ge.commit(self.gepc, val);
            }
            HQ_GEPC => {
                self.gepc = val & 0xffff;
                self.loaducode = ge.reload(self.gepc);
            }
            HQ_INTR => self.intr = val,
            HQ_UNSTALL => self.running = 1,
            HQ_REFRESH => self.refresh = val,
            HQ_FIN3 => self.fin[FIN3].store(val, Ordering::Release),
            _ => {}
        }
    }
}

/// Where the interpreter's output goes: RE3 register writes and private ops.
pub trait Re3Sink {
    fn reg(&mut self, reg: usize, val: u32);
    fn copy(&mut self, sx: i32, sy: i32, w: i32, h: i32, dx: i32, dy: i32);
    /// Raise a finish flag (FIN1/FIN2/FIN3), as the microcode does when it
    /// completes a request the host is waiting on.
    fn finish(&mut self, flag: usize);
    /// Screen-to-host read: once all drawing is done, pack the rectangle
    /// into shram and raise FIN3 (READ_IMAGE), or feed it to the HQ2_GEDMA
    /// read port and raise FIN2 (pixel DMA reads).
    fn read_image(&mut self, req: &ReadImage, dest: ReadDest);
    /// Emulator-private RE3 operation (re3::RE3_OP_*).
    fn op(&mut self, op: u32, val: u64);
    /// Store a word into shared RAM, as the microcode does for readbacks.
    fn shram(&mut self, word: usize, val: u32);
    /// Start a new transfer on the HQ2_GEDMA read port and queue `words`
    /// (a context image, 0x1E1). Words a previous transfer left unread are
    /// dropped. Blocks while the port is full.
    fn gedma_out(&mut self, words: &[u32]);
}

/// HLE command interpreter state. Owned by the HQ2 thread. Plain data.
#[repr(C)]
pub struct Hq2Engine {
    cmd: u32,
    need: u32,
    nargs: u32,
    last_nargs: u32,
    /// Total words of the last command, including any beyond MAX_ARGS.
    last_total: u32,
    args: [u32; MAX_ARGS],
    /// Character raster position (GL coordinates, y = 0 at the bottom).
    cx: i32,
    cy: i32,
    pattern_on: u32,
    /// Set by UNSTALL: the next stray DATA word is the start argument.
    restarted: u32,
    /// Streaming command state: words collected for the current item, items done.
    stream_words: u32,
    stream_items: u32,
    s2d: State2d,
    /// Tile for TILE_RECT: geometry from TILE_SETUP, words from TILE_DATA.
    tile_w: u32,
    tile_h: u32,
    tile_len: u32,
    /// Tile pixel format from TILE_SETUP: 2 = 8-bit pixels, 3 = 1-bit stipple.
    tile_fmt: u32,
    tile: [u32; TILE_WORDS],
    stipple_opaque: u32,
    /// Line clip box (inclusive), set by LINE_CLIP; full screen after BEGIN.
    line_clip: [i32; 4],
    /// Previous polyline point, and whether one is pending.
    line_prev: [i32; 2],
    line_have_prev: u32,
    /// DRAW_IMAGE: header, current row buffer, rows done.
    img_hdr: [u32; 7],
    img_row: [u32; IMG_ROW_WORDS],
    img_rows_done: u32,
    /// One bit per FIFO index already reported as unimplemented.
    warned: [u64; FIFO_WORDS / 64],
    /// OpenGL (GE7 geometry) state.
    gl: gl::GlState,
    /// GL context switch bookkeeping and restore stream (GE_HQMSAV).
    gl_ctx: gl::GlCx,
    /// GE_CX_RESTORE_EXT words still to consume (u32::MAX: count not yet in).
    cx_ext_left: u32,
}

impl Hq2Engine {
    /// Total words (command word + PUC_DATA words) each command consumes.
    fn arg_count(cmd: u32) -> u32 {
        match cmd {
            PUC_INIT | PUC_COLOR | PUC_FINISH => 1,
            PUC_PNT2I | PUC_CMOV2I => 2,
            PUC_RECTI2D | PUC_LINE2I => 4,
            PUC_DRAWCHAR => 25,
            PUC_RECTCOPY => 8,
            GE_HQMSAV => 3,
            GE_CX_SAVE_MAIN | GE_CX_SAVE_EXT => 1,
            // Restore main: the token carries the word count; the image
            // follows on HQ2_GEDMA. Restore ext: token, DATA count, image.
            GE_CX_RESTORE_MAIN | GE_CX_RESTORE_EXT => STREAM,
            HQ_GL_FIN3 | GL_FINISH | GL_FLUSH => 1,
            HQ_PCX_1E0 | HQ_PCX_1E6 => 1,
            HQ2_2D_BEGIN | HQ2_2D_COLOR_AUX | HQ2_2D_COLOR_OFF | HQ2_2D_COLOR_ON | HQ2_2D_MODE
            | HQ2_2D_CID_WRITE | HQ2_2D_END_PRIMITIVE => 1,
            HQ2_2D_BUF_SELECT => 2,
            HQ2_2D_READ_IMAGE => 5,
            // Token x, then y, width, height, words/row, flag, 0 on GE_DATA.
            HQ2_2D_DMA_READ_PIXELS | HQ2_GL_DMA_READ => 7,
            HQ2_2D_ROP => 4,
            HQ2_2D_TILE_SETUP => 5,
            HQ2_2D_MONO_IMAGE_8 => 8,
            HQ2_2D_MONO_IMAGE_16 => 12,
            HQ2_2D_MONO_IMAGE_32 => 20,
            HQ2_2D_MONO_IMAGE_64 => 36,
            HQ2_2D_GLYPH_8 => 12,
            324 => 20,
            325 => 36,
            HQ2_2D_GLYPH_64 => 68,
            HQ2_2D_LINE_MODE | HQ2_2D_STIPPLE_AUX => 1,
            HQ2_2D_LINE_CLIP => 4,
            HQ2_2D_STIPPLE_RECT | HQ2_2D_STIPPLE_BOX_POW2 | HQ2_2D_STIPPLE_BOX => 7,
            HQ2_2D_STIPPLED_SPAN_A | HQ2_2D_STIPPLED_SPAN_B => STREAM,
            HQ2_2D_POLY_SPAN | HQ2_2D_POLYLINE | HQ2_2D_POLYLINE_ALT | HQ2_2D_SEGMENTS | HQ2_2D_LINE_SEG => STREAM,
            HQ2_2D_SOLID_RECT => STREAM,
            HQ2_2D_TILE_RECT | HQ2_2D_TILE_RECT_ODD => 7,
            HQ2_2D_DRAW_IMAGE | HQ2_2D_DRAW_IMAGE_SMALL | HQ2_DMA_WRITE_PIXELS
            | HQ2_GL_DMA_WRITE | HQ2_GL_DMA_WRITE_ZOOM => STREAM,
            HQ2_2D_COPY_RECT => 8,
            HQ2_2D_TILE_DATA_FIRST..=HQ2_2D_TILE_DATA_LAST => STREAM,
            _ => Self::gl_arg_count(cmd).unwrap_or(OPEN),
        }
    }

    /// Feed one FIFO entry (word index, data). `done` is called with a
    /// decoded description of every command that completes (tracing only).
    ///
    /// Known commands have a fixed word count. Unknown (not yet implemented)
    /// commands are *open*: they collect every following DATA word until the
    /// next command index arrives, so traces show each unknown command with
    /// all its arguments.
    pub fn push(&mut self, index: u32, val: u32, out: &mut dyn Re3Sink, done: Option<&mut dyn FnMut(String)>) {
        let mut done = done;
        if index == HQ_TOKEN_UNSTALL {
            if self.need == OPEN {
                self.complete(out, &mut done);
            }
            self.need = 0;
            self.nargs = 0;
            self.restarted = 1;
            return;
        }
        // FIFO index 0 (GE_DATA) is a data port too: the kernel's pixel-DMA
        // trigger streams its arguments there.
        // Any index whose token bits are PUC_DATA is a data port: the ITOF
        // variant (0x41DF, integer DATA) carries e.g. software fragments.
        if index == PUC_DATA || index == HQ_TOKEN_GEDMA || index == 0
            || (index < FIFO_WORDS as u32 && index & 0x1ff == PUC_DATA) {
            if self.need == 0 && self.gl_port_wants_data(val, out, &mut done) {
                return;
            }
            if self.need == 0 {
                if self.restarted != 0 && index == PUC_DATA {
                    // Microcode reset path (kernel Gr2Start): one argument
                    // (0/1/2 by board rev), then FIN2.
                    self.restarted = 0;
                    self.cmd = HQ_TOKEN_UNSTALL;
                    self.args[0] = val;
                    self.nargs = 1;
                    self.complete(out, &mut done);
                }
                // Stray argument: the last command took fewer words than
                // the host sent (usually a stream mistaken for fixed-size).
                if let Some(f) = done.as_mut() {
                    f(format!("stray DATA {val:#x} after {}", token_name(self.cmd).map(str::to_string)
                        .unwrap_or_else(|| format!("CMD{:#x}", self.cmd))));
                }
                return;
            }
            if self.need == STREAM {
                self.stream_word(val, out, &mut done);
                return;
            }
            if (self.nargs as usize) < MAX_ARGS {
                self.args[self.nargs as usize] = val;
            }
            self.nargs += 1;
        } else {
            // GL self-streaming ports (matrix, vertex, colour, normal): every
            // word goes to the same index; they close any open command.
            let mid_fixed = self.need != 0 && self.need != OPEN && self.need != STREAM && self.nargs < self.need;
            if !mid_fixed && self.gl_port_ready(index) {
                if self.need == STREAM {
                    self.end_stream(&mut done);
                } else if self.need == OPEN {
                    self.complete(out, &mut done);
                }
                self.need = 0;
                self.nargs = 0;
                self.gl_port(index, val, out, &mut done);
                return;
            }
            // Commands whose arguments all go to the command's own index
            // (0x02E enable, plane; 0x008 IRIS GL buffer): a repeat of the
            // index mid-command is the next argument, not a new command.
            if index == self.cmd && self.need != 0 && self.need != OPEN && self.need != STREAM
                && self.nargs < self.need
            {
                if (self.nargs as usize) < MAX_ARGS {
                    self.args[self.nargs as usize] = val;
                }
                self.nargs += 1;
                if self.nargs >= self.need {
                    self.complete(out, &mut done);
                }
                return;
            }
            if self.need == STREAM {
                self.end_stream(&mut done);
            } else if self.need == OPEN {
                self.complete(out, &mut done);
            } else if self.need != 0 {
                crate::dlog_dev!(crate::devlog::LogModule::Gr2,
                    "HQ2: command {} dropped after {}/{} words", self.cmd, self.nargs, self.need);
            }
            self.restarted = 0;
            self.cmd = index;
            self.need = Self::arg_count(index);
            self.args[0] = val;
            self.nargs = 1;
            if self.need == STREAM {
                self.stream_words = 0;
                self.stream_items = 0;
                // Tile data loaders carry a tile word in the command word too.
                if (HQ2_2D_TILE_DATA_FIRST..=HQ2_2D_TILE_DATA_LAST).contains(&index) {
                    self.tile_push(val);
                }
                if index == HQ2_2D_POLYLINE || index == HQ2_2D_POLYLINE_ALT {
                    self.line_have_prev = 0;
                }
                if index == HQ2_2D_DRAW_IMAGE || index == HQ2_2D_DRAW_IMAGE_SMALL || is_pixel_dma(index) {
                    self.img_hdr[0] = val;
                    self.img_rows_done = 0;
                }
                if index == GE_CX_RESTORE_MAIN {
                    self.gl_cx_restore_begin(val);
                    if val == 0 {
                        self.need = 0;
                        out.finish(FIN2);
                    }
                }
                if index == GE_CX_RESTORE_EXT {
                    // The word count arrives as the first DATA word.
                    self.cx_ext_left = u32::MAX;
                }
                return;
            }
        }
        if self.need != OPEN && self.nargs >= self.need {
            self.complete(out, &mut done);
        }
    }

    fn tile_push(&mut self, val: u32) {
        if (self.tile_len as usize) < TILE_WORDS {
            self.tile[self.tile_len as usize] = val;
        }
        self.tile_len += 1;
    }

    /// One DATA word for the active streaming command.
    fn stream_word(&mut self, val: u32, out: &mut dyn Re3Sink, done: &mut Option<&mut dyn FnMut(String)>) {
        if self.cmd == GE_CX_RESTORE_MAIN {
            if self.gl_cx_restore_word(val) {
                self.need = 0;
                out.finish(FIN2);
                if let Some(f) = done.as_mut() {
                    let why = match self.gl_ctx.restore_reject {
                        0 => "restored".to_string(),
                        1 => format!("REJECTED: {} words, image is {}: previous context's state stays live", self.args[0], CX_WORDS),
                        2 => "REJECTED: bad magic: previous context's state stays live".to_string(),
                        _ => "REJECTED: GlState size differs: previous context's state stays live".to_string(),
                    };
                    f(format!("GE_CX_RESTORE_MAIN {} words {} -> FIN2", self.args[0], why));
                }
            }
            return;
        }
        if self.cmd == GE_CX_RESTORE_EXT {
            // The HLE reports no extended context (shram 0x304 = 0); if the
            // kernel restores one anyway, consume its words and ack.
            if self.cx_ext_left == u32::MAX {
                self.cx_ext_left = val;
            } else if self.cx_ext_left > 0 {
                self.cx_ext_left -= 1;
            }
            if self.cx_ext_left == 0 {
                self.need = 0;
                out.finish(FIN2);
            }
            return;
        }
        if (HQ2_2D_TILE_DATA_FIRST..=HQ2_2D_TILE_DATA_LAST).contains(&self.cmd) {
            self.tile_push(val);
            return;
        }
        if self.cmd == HQ2_2D_DRAW_IMAGE || self.cmd == HQ2_2D_DRAW_IMAGE_SMALL || is_pixel_dma(self.cmd) {
            self.image_word(val, out, done);
            return;
        }
        let group = match self.cmd {
            HQ2_2D_POLY_SPAN => 3,
            HQ2_2D_POLYLINE | HQ2_2D_POLYLINE_ALT => 2,
            HQ2_2D_SEGMENTS | HQ2_2D_LINE_SEG | HQ2_2D_STIPPLED_SPAN_A | HQ2_2D_STIPPLED_SPAN_B => 4,
            _ => 0,
        };
        if group != 0 {
            let k = self.stream_words as usize;
            self.args[1 + k] = val;
            self.stream_words += 1;
            if self.stream_words as usize == group {
                self.stream_words = 0;
                self.stream_items += 1;
                let a = self.args;
                let stippled = matches!(self.cmd, HQ2_2D_STIPPLED_SPAN_A | HQ2_2D_STIPPLED_SPAN_B);
                let fg = if stippled { self.s2d.color_on } else { self.s2d.fg };
                if self.stream_items == 1 {
                    self.setup2d(fg, out);
                }
                match self.cmd {
                    HQ2_2D_POLY_SPAN => {
                        if a[1] < PAD_X && a[2] < PAD_Y {
                            let (x, y, w) = (a[1] as i32, a[2] as i32, a[3] as i32);
                            let (x1, x2) = (x.max(0), (x + w).min(re3::FB_W as i32));
                            if x1 < x2 {
                                self.span(x1, SCREEN_H - 1 - y, (x2 - x1) as u32, None, out);
                            }
                        }
                    }
                    HQ2_2D_STIPPLED_SPAN_A | HQ2_2D_STIPPLED_SPAN_B => {
                        self.stippled_span(a[1] as i32, a[2] as i32, a[3] as i32, a[4], out);
                        if let Some(f) = done.as_mut() {
                            f(format!("2D_STIPPLED_SPAN ({}, {}) w={} pattern={:#010x} opaque={} on={:#x} off={:#x}",
                                a[1] as i32, a[2] as i32, a[3] as i32, a[4], self.stipple_opaque, self.s2d.color_on, self.s2d.color_off));
                        }
                    }
                    HQ2_2D_SEGMENTS | HQ2_2D_LINE_SEG => {
                        // Padding: (1280, 1024) pairs (SEGMENTS) or x1 = x2
                        // = 1280 with y 0 (LINE_SEG, 4Dwm frames).
                        let pad = (a[1] == PAD_X && a[2] == PAD_Y) || (a[1] == PAD_X && a[3] == PAD_X);
                        if !pad {
                            // SEGMENTS draws both endpoints; LINE_SEG is the
                            // CapNotLast form and stops before (x2, y2)
                            // (expSegmentSS picks it by cap style and swaps
                            // reversed segments with +1 to keep the pixels).
                            let last = self.cmd == HQ2_2D_SEGMENTS;
                            self.line2d([a[1] as i32, a[2] as i32], [a[3] as i32, a[4] as i32], last, out);
                        }
                        if let Some(f) = done.as_mut() {
                            f(format!("{} ({}, {})-({}, {}){} fg={:#x}", token_name(self.cmd).unwrap_or("?"),
                                a[1] as i32, a[2] as i32, a[3] as i32, a[4] as i32, if pad { " pad" } else { "" }, self.s2d.fg));
                        }
                    }
                    _ => {
                        let p = [a[1] as i32, a[2] as i32];
                        if a[1] == PAD_X && a[2] == PAD_Y {
                            self.line_have_prev = 0;
                        } else if self.line_have_prev != 0 {
                            // Each segment stops before its end point, so
                            // joints are drawn once and the final point is
                            // not drawn: expLineSS sends an extra (x + 1, y)
                            // point when the cap style wants it.
                            self.line2d(self.line_prev, p, false, out);
                            self.line_prev = p;
                        } else {
                            self.line_prev = p;
                            self.line_have_prev = 1;
                        }
                    }
                }
            }
            return;
        }
        let k = self.stream_words as usize;
        self.args[1 + k] = val;
        self.stream_words += 1;
        if self.cmd == HQ2_2D_SOLID_RECT && self.stream_words == 4 {
            self.stream_words = 0;
            self.stream_items += 1;
            let b = [self.args[1], self.args[2], self.args[3], self.args[4]].map(|v| v as i32);
            self.rect2d(b[0], b[1], b[2], b[3], out);
            if let Some(f) = done.as_mut() {
                let s = self.s2d;
                f(format!("2D_SOLID_RECT box {} ({}, {})-({}, {}) mode={:#x} fg={:#x} pm={:#x} aux={:#x} alu={} cid={:#x}",
                    self.stream_items, b[0], b[1], b[2], b[3], s.mode, s.fg, s.planemask, s.aux_mask, s.alu, s.cid_write));
            }
        }
    }

    fn end_stream(&mut self, done: &mut Option<&mut dyn FnMut(String)>) {
        if self.cmd == HQ2_2D_POLYLINE || self.cmd == HQ2_2D_POLYLINE_ALT {
            self.line_have_prev = 0;
        }
        let leftover = if self.cmd == HQ2_2D_DRAW_IMAGE || self.cmd == HQ2_2D_DRAW_IMAGE_SMALL {
            self.img_rows_done < self.img_hdr[3]
        } else { self.stream_words != 0 };
        if leftover {
            if let Some(f) = done.as_mut() {
                f(format!("{} ended with {} stray words", token_name(self.cmd).unwrap_or("?"), self.stream_words));
            }
        }
        self.need = 0;
        self.nargs = 0;
        self.stream_words = 0;
    }

    /// 2D solid rectangle: X box (x2/y2 exclusive, y down from the top of
    /// the screen) into the planes selected by MODE / CID_WRITE.
    fn rect2d(&mut self, x1: i32, y1: i32, x2: i32, y2: i32, out: &mut dyn Re3Sink) {
        let (x1, x2) = (x1.max(0), x2.min(re3::FB_W as i32));
        let (y1, y2) = (y1.max(0), y2.min(SCREEN_H));
        if x1 >= x2 || y1 >= y2 {
            return;
        }
        let fg = self.s2d.fg;
        self.setup2d(fg, out);
        for y in y1..y2 {
            self.span(x1, SCREEN_H - 1 - y, (x2 - x1) as u32, None, out);
        }
    }

    /// Set the value the next spans write, for the active plane group:
    /// CID (from CID_WRITE, fixed), 4-bit overlay (MODE 8), 2-bit overlay or
    /// popup (MODE 0xB), or the colour planes. Used once per primitive by
    /// `setup2d` and again for every colour run of tiles/images/stipples.
    fn value2d(&self, v: u32, out: &mut dyn Re3Sink) {
        let s = self.s2d;
        if s.cid_write & 0xf000 != 0 {
            return;
        }
        match s.mode & 0xff {
            8 => out.reg(re3::REG_UAUXDATA, v & 0xf),
            0xb => out.reg(re3::REG_UAUXDATA, if s.aux_mask & 0xc != 0 { (v << 2) & 0xc } else { v & 3 }),
            _ => self.color2d(v, out),
        }
    }

    /// Set the RE3 colour for a GC pixel (fg, COLOR_ON / OFF; colour planes:
    /// R in bits 7:0). In 12-bit mode (MODE 2) the pixel is the X visual's
    /// 12-bit value (R 3:0, G 7:4, B 11:8): the DDX passes GC pixels
    /// unconverted (expDrawPoints), so widen it here (inferred).
    fn color2d(&self, pixel: u32, out: &mut dyn Re3Sink) {
        if self.s2d.mode & 0xff == 2 {
            let wide = |n: u32| (n & 0xf) * 17;
            return self.rgb2d(wide(pixel) | (wide(pixel >> 4) << 8) | (wide(pixel >> 8) << 16), out);
        }
        self.rgb2d(pixel, out);
    }

    /// Colour of an image / tile pixel word: in 12-bit mode the DDX has
    /// already widened it to 8:8:8 (expDrawImage12TC, expTileRects: the
    /// nibbles in the high halves), which the RE3 12-bit packer takes as is.
    fn image2d(&self, pixel: u32, out: &mut dyn Re3Sink) {
        if self.s2d.mode & 0xff == 2 && self.s2d.cid_write & 0xf000 == 0 {
            return self.rgb2d(pixel, out);
        }
        self.value2d(pixel, out);
    }

    /// RE3 colour registers from an 8:8:8 value (R in bits 7:0).
    fn rgb2d(&self, pixel: u32, out: &mut dyn Re3Sink) {
        out.reg(re3::REG_R, (pixel & 0xff) << 11);
        out.reg(re3::REG_G, ((pixel >> 8) & 0xff) << 11);
        out.reg(re3::REG_B, ((pixel >> 16) & 0xff) << 11);
    }

    /// 1-bpp bitmap (MONO_IMAGE_* and GLYPH_*), transparent: fg where a bit
    /// is 1. Header x; y, width, rows; `words` data words follow. Rows are
    /// MSB-leftmost; width <= 16 packs two 16-bit rows per word (first row
    /// high), <= 32 one row per word, wider two words per row.
    fn mono_image(&mut self, words: usize, out: &mut dyn Re3Sink) {
        let a = self.args;
        let (x, y, w) = (a[0] as i32, a[1] as i32, a[2].min(64));
        let cap = if w <= 16 { words * 2 } else if w <= 32 { words } else { words / 2 };
        let rows = (a[3] as usize).min(cap);
        let on = self.s2d.color_on;
        self.setup2d(on, out);
        for r in 0..rows {
            let gy = SCREEN_H - 1 - (y + r as i32);
            let word = |i: usize| a.get(4 + i).copied().unwrap_or(0);
            // Up to two 32-pixel halves per row.
            let halves: [u32; 2] = if w <= 16 {
                let wd = word(r / 2);
                [if r % 2 == 0 { wd & 0xffff_0000 } else { wd << 16 }, 0]
            } else if w <= 32 {
                [word(r), 0]
            } else {
                [word(2 * r), word(2 * r + 1)]
            };
            for (h, &bits) in halves.iter().enumerate() {
                let hw = w.saturating_sub(32 * h as u32).min(32);
                if hw == 0 {
                    break;
                }
                let pat = if hw < 32 { bits & !(u32::MAX >> hw) } else { bits };
                if pat != 0 {
                    self.span(x + 32 * h as i32, gy, hw, Some(pat), out);
                }
            }
        }
    }

    /// One DRAW_IMAGE data word: 6 header words after x, then pixel rows;
    /// anything after the last row (0xDEADBEEF padding) is ignored.
    fn image_word(&mut self, val: u32, out: &mut dyn Re3Sink, done: &mut Option<&mut dyn FnMut(String)>) {
        let k = self.stream_words as usize;
        self.stream_words += 1;
        if k < 6 {
            self.img_hdr[1 + k] = val;
            if k == 5 {
                if is_pixel_dma(self.cmd) {
                    // DMA header is y, width, height, words/row, flag, 0:
                    // derive the pixel format from pixels per word.
                    let (w, wpr) = (self.img_hdr[2], self.img_hdr[4].max(1));
                    let per_word = (w + wpr - 1) / wpr;
                    self.img_hdr[5] = match per_word { 4 => 2, 2 => 1, _ => 0 };
                    self.img_hdr[6] = 0;
                }
                if self.cmd == HQ2_DMA_WRITE_PIXELS || !is_pixel_dma(self.cmd) {
                    let fg = self.s2d.fg;
                    self.setup2d(fg, out);
                }
            }
            return;
        }
        let [x, y, w, rows, wpr, fmt, skip] = self.img_hdr;
        let wpr = wpr.max(1) as usize;
        if self.img_rows_done >= rows {
            return; // padding
        }
        let i = (k - 6) % wpr;
        if i < IMG_ROW_WORDS {
            self.img_row[i] = val;
        }
        if i + 1 == wpr {
            let row = self.img_rows_done;
            if self.cmd == HQ2_GL_DMA_WRITE || self.cmd == HQ2_GL_DMA_WRITE_ZOOM {
                let words = self.img_row;
                let zoomed = self.cmd == HQ2_GL_DMA_WRITE_ZOOM;
                self.gl_pixel_row(x as i32, y as i32, row, w, rows, fmt, &words[..wpr.min(IMG_ROW_WORDS)], zoomed, out);
            } else {
                self.image_row(x as i32, y as i32 + row as i32, w, fmt, skip, wpr.min(IMG_ROW_WORDS), out);
            }
            self.img_rows_done += 1;
            if self.img_rows_done == rows {
                if is_pixel_dma(self.cmd) {
                    // The kernel polls FIN2 for completion of the transfer.
                    out.finish(FIN2);
                }
                if let Some(f) = done.as_mut() {
                    f(format!("{} at ({}, {}) {}x{} words/row={} format={} skip={} fg={:#x} mode={:#x}",
                        token_name(self.cmd).unwrap_or("?"), x as i32, y as i32, w, rows, wpr, fmt, skip,
                        self.s2d.fg, self.s2d.mode));
                }
            }
        }
    }

    /// Draw one image row as runs of equal pixels. Format 2 = 8-bit pixels
    /// (4 per word, MSB first; seen from the DDX). 1 = 16-bit and 0 = 32-bit
    /// pixels are (unverified) guesses.
    fn image_row(&mut self, x: i32, y: i32, w: u32, fmt: u32, skip: u32, wpr: usize, out: &mut dyn Re3Sink) {
        if y < 0 || y >= SCREEN_H {
            return;
        }
        let row = self.img_row;
        let (per_word, bits) = match fmt { 2 => (4usize, 8u32), 1 => (2, 16), _ => (1, 32) };
        let px = |n: usize| -> u32 {
            let wi = n / per_word;
            if wi >= wpr { return 0; }
            let sh = 32 - bits * (1 + (n % per_word) as u32);
            if bits == 32 { row[wi] } else { (row[wi] >> sh) & ((1 << bits) - 1) }
        };
        let gy = SCREEN_H - 1 - y;
        let (start, end) = (skip as usize, skip as usize + w as usize);
        let mut n = start;
        while n < end {
            let c = px(n);
            let mut e = n + 1;
            while e < end && px(e) == c {
                e += 1;
            }
            let sx = x + (n - start) as i32;
            if sx < re3::FB_W as i32 && sx + (e - n) as i32 > 0 {
                self.image2d(c, out);
                self.span(sx, gy, (e - n) as u32, None, out);
            }
            n = e;
        }
    }

    /// Zero-width line with Bresenham, clipped to the line clip box and the
    /// screen, emitted as horizontal runs. The start point is always drawn;
    /// `last` also draws the end point.
    fn line2d(&mut self, p0: [i32; 2], p1: [i32; 2], last: bool, out: &mut dyn Re3Sink) {
        let c = self.line_clip;
        let (cx1, cx2) = (c[0].max(0), c[1].min(re3::FB_W as i32 - 1));
        let (cy1, cy2) = (c[2].max(0), c[3].min(SCREEN_H - 1));
        let (mut x, mut y) = (p0[0], p0[1]);
        let (dx, dy) = ((p1[0] - x).abs(), -(p1[1] - y).abs());
        let (sx, sy) = (if p1[0] >= x { 1 } else { -1 }, if p1[1] >= y { 1 } else { -1 });
        let mut err = dx + dy;
        let mut run: Option<(i32, i32, i32)> = None; // (y, xmin, xmax)
        loop {
            let end = x == p1[0] && y == p1[1];
            if end && !last {
                break;
            }
            if x >= cx1 && x <= cx2 && y >= cy1 && y <= cy2 {
                run = match run {
                    Some((ry, a, b)) if ry == y && (x == b + 1 || x == a - 1) => Some((ry, a.min(x), b.max(x))),
                    Some((ry, a, b)) => {
                        self.span(a, SCREEN_H - 1 - ry, (b - a + 1) as u32, None, out);
                        Some((y, x, x))
                    }
                    None => Some((y, x, x)),
                };
            }
            if end {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                err += dx;
                y += sy;
            }
        }
        if let Some((ry, a, b)) = run {
            self.span(a, SCREEN_H - 1 - ry, (b - a + 1) as u32, None, out);
        }
    }

    /// Stippled box (STIPPLE_RECT): 1-bit tile anchored at the origin; bits
    /// set draw fg, clear bits draw bg when STIPPLE_AUX is 1 (opaque).
    /// One stippled span: `w` pixels from (x, y) (y top-down); `pat` MSB is
    /// pixel x and repeats every 32 pixels. The colour currently loaded is
    /// COLOR_ON (setup2d at the start of the stream).
    fn stippled_span(&mut self, x: i32, y: i32, w: i32, pat: u32, out: &mut dyn Re3Sink) {
        if y < 0 || y >= SCREEN_H || w <= 0 || x >= re3::FB_W as i32 {
            return;
        }
        let gy = SCREEN_H - 1 - y;
        let (on, off, opaque) = (self.s2d.color_on, self.s2d.color_off, self.stipple_opaque != 0);
        if opaque {
            // Background first, then the set bits on top.
            self.value2d(off, out);
            self.span(x, gy, w as u32, None, out);
            self.value2d(on, out);
        }
        let mut i = 0;
        while i < w {
            let n = (w - i).min(32);
            let p = if n < 32 { pat & !(u32::MAX >> n) } else { pat };
            if p != 0 {
                self.span(x + i, gy, n as u32, Some(p), out);
            }
            i += 32;
        }
    }

    fn stipple_rect(&mut self, ox: i32, oy: i32, b: [i32; 4], out: &mut dyn Re3Sink) {
        let (x1, x2) = (b[0].max(0), b[2].min(re3::FB_W as i32));
        let (y1, y2) = (b[1].max(0), b[3].min(SCREEN_H));
        let (tw, th) = (self.tile_w.max(1) as i32, self.tile_h.max(1) as i32);
        if x1 >= x2 || y1 >= y2 {
            return;
        }
        let stride = ((tw + 31) / 32) as usize;
        let tile = self.tile;
        let bit = |tx: i32, ty: i32| -> bool {
            let i = ty as usize * stride + tx as usize / 32;
            let wd = if i < TILE_WORDS { tile[i] } else { 0 };
            (wd >> (31 - tx % 32)) & 1 != 0
        };
        let (fg, bg, opaque) = (self.s2d.color_on, self.s2d.color_off, self.stipple_opaque != 0);
        self.setup2d(fg, out);
        let mut cur = fg;
        for y in y1..y2 {
            let ty = (y - oy).rem_euclid(th);
            let gy = SCREEN_H - 1 - y;
            let mut x = x1;
            while x < x2 {
                let on = bit((x - ox).rem_euclid(tw), ty);
                let mut e = x + 1;
                while e < x2 && bit((e - ox).rem_euclid(tw), ty) == on {
                    e += 1;
                }
                if on || opaque {
                    let c = if on { fg } else { bg };
                    if c != cur {
                        self.value2d(c, out);
                        cur = c;
                    }
                    self.span(x, gy, (e - x) as u32, None, out);
                }
                x = e;
            }
        }
    }

    /// Tiled fill of one X box. The tile is 8-bit pixels, 4 per word, MSB
    /// first, rows of ceil(width/4) words; the pattern is anchored at the
    /// origin. (Format inferred from the root-window trace: (unverified).)
    fn tile_rect(&mut self, ox: i32, oy: i32, b: [i32; 4], out: &mut dyn Re3Sink) {
        let (x1, x2) = (b[0].max(0), b[2].min(re3::FB_W as i32));
        let (y1, y2) = (b[1].max(0), b[3].min(SCREEN_H));
        let (tw, th) = (self.tile_w.max(1) as i32, self.tile_h.max(1) as i32);
        if x1 >= x2 || y1 >= y2 {
            return;
        }
        // Pixel format (TILE_SETUP word 3, expTileRects): 2 = 8-bit (4 per
        // word), 1 = 16-bit (2 per word), 0 = 24/32-bit (1 per word); MSB
        // first; rows are ceil(width * bpp / 32) words.
        let per_word: i32 = match self.tile_fmt { 2 => 4, 1 => 2, _ => 1 };
        let bits = 32 / per_word as u32;
        let stride = ((tw + per_word - 1) / per_word) as usize;
        let tile = self.tile;
        let tile_px = |tx: i32, ty: i32| -> u32 {
            let i = ty as usize * stride + (tx / per_word) as usize;
            let wd = if i < TILE_WORDS { tile[i] } else { 0 };
            if bits == 32 { wd } else { (wd >> (32 - bits * (1 + (tx % per_word) as u32))) & ((1 << bits) - 1) }
        };
        let fg = self.s2d.fg;
        self.setup2d(fg, out);
        for y in y1..y2 {
            let ty = (y - oy).rem_euclid(th);
            let gy = SCREEN_H - 1 - y;
            // Emit runs of equal colour.
            let mut x = x1;
            while x < x2 {
                let c = tile_px((x - ox).rem_euclid(tw), ty);
                let mut e = x + 1;
                while e < x2 && tile_px((e - ox).rem_euclid(tw), ty) == c {
                    e += 1;
                }
                self.image2d(c, out);
                self.span(x, gy, (e - x) as u32, None, out);
                x = e;
            }
        }
    }

    /// RE3 state for a 2D primitive: raster op, colour and the plane group
    /// selected by MODE / CID_WRITE.
    fn setup2d(&mut self, fg: u32, out: &mut dyn Re3Sink) {
        let s = self.s2d;
        out.reg(re3::REG_FUNC, s.alu & 0xf);
        self.color2d(fg, out);
        out.reg(re3::REG_NOPUP, 1);
        if s.cid_write & 0xf000 != 0 {
            // CID planes (bits 31:28) only.
            out.reg(re3::REG_WIDDATA, (s.cid_write >> 8) & 0xf);
            out.reg(re3::REG_UAUXDATA, 0);
            out.reg(re3::REG_AUXMASK, 0xf0);
            out.reg(re3::REG_PIXMASK, 0);
        } else {
            match s.mode & 0xff {
                // 4-bit overlay (MODE 8) and 2-bit overlay / popup (MODE 0xB):
                // mask COLOR_AUX, value set by value2d (popup value shifted
                // up, inferred).
                8 | 0xb => {
                    out.reg(re3::REG_AUXMASK, s.aux_mask & 0xf);
                    out.reg(re3::REG_PIXMASK, 0);
                    self.value2d(fg, out);
                }
                // 12-bit RGB (MODE 2, 12-bit TrueColor windows): the RE3
                // writes the value to both 12-bit buffers and the plane mask
                // picks one (RE3.h). The buffers never move; ROP flag bit 3
                // (the window's dbc buffer * 8) names buffer 1. A mask the
                // DDX already moved or doubled (dbc -3 / -1, expTileRects)
                // is used as it is.
                2 => {
                    let pm = s.planemask & 0x00ff_ffff;
                    let mask = if s.flag & 8 == 0 {
                        pm & 0xfff
                    } else if pm & 0xfff000 != 0 {
                        pm
                    } else {
                        pm << 12
                    };
                    out.reg(re3::REG_UAUXDATA, 0);
                    out.reg(re3::REG_AUXMASK, 0);
                    out.reg(re3::REG_PIXMASK, mask);
                    out.op(re3::RE3_OP_PIXFMT, re3::PIXFMT_RGB12 as u64);
                    out.reg(re3::REG_ENABDITH, 0);
                }
                // Colour planes.
                _ => {
                    out.reg(re3::REG_UAUXDATA, 0);
                    out.reg(re3::REG_AUXMASK, 0);
                    out.reg(re3::REG_PIXMASK, s.planemask & 0x00ff_ffff);
                    out.op(re3::RE3_OP_PIXFMT, 0);
                }
            }
        }
    }

    fn complete(&mut self, out: &mut dyn Re3Sink, done: &mut Option<&mut dyn FnMut(String)>) {
        let cmd = self.cmd;
        self.last_nargs = self.nargs.min(MAX_ARGS as u32);
        self.last_total = self.nargs;
        self.need = 0;
        self.nargs = 0;
        self.execute(cmd, out);
        if let Some(f) = done.as_mut() {
            f(self.describe(cmd));
        }
    }

    /// Decoded form of the command that just completed (for traces).
    pub fn describe(&self, cmd: u32) -> String {
        let a = &self.args[..self.last_nargs as usize];
        let name = index_label(cmd);
        let i = |k: usize| a.get(k).copied().unwrap_or(0) as i32;
        match cmd {
            PUC_INIT => format!("{name} arg={}", i(0)),
            HQ_TOKEN_UNSTALL => format!("microcode start arg={} -> FIN2", i(0)),
            GE_HQMSAV => {
                let prev = self.gl_ctx.prev_owner;
                let live = if prev == u32::MAX { "none".to_string() } else { format!("{prev:#x}") };
                // State 1 = the kernel believes this context's state is still
                // in the GE: no restore follows. If the HLE's live state
                // belongs to another context, that one's state leaks.
                let leak = if i(1) == 1 && prev != u32::MAX && prev != a[0] { "  LEAK? state 1 but live state is another context's" } else { "" };
                format!("{name} ctx={:#x} state={} mode={} (live was {live}){leak} -> FIN2", i(0), i(1), i(2))
            }
            HQ_PCX_1E0 | HQ_PCX_1E6 => format!("{name} {} -> FIN2", i(0)),
            HQ_GL_FIN3 | GL_FINISH => format!("{name} {} -> FIN3", i(0)),
            HQ2_2D_ROP => format!("{name} fg={:#x} planemask={:#x} alu={} flag={:#x}", a[0], a[1], i(2), a[3]),
            HQ2_2D_TILE_SETUP => format!("{name} size={:#x} {}x{} format={} {:#x}", a[0], i(1), i(2), i(3), a[4]),
            HQ2_2D_COPY_RECT => format!("{name} pitch={} chunk={} src=({}, {}) {}x{} dst=({}, {})",
                i(0), i(1), i(2), i(3), i(4), i(5), i(6), i(7)),
            HQ2_2D_MONO_IMAGE_8..=HQ2_2D_MONO_IMAGE_64 | HQ2_2D_GLYPH_8..=HQ2_2D_GLYPH_64 =>
                format!("{name} at ({}, {}) {}x{} on={:#x}", i(0), i(1), i(2), i(3), self.s2d.color_on),
            HQ2_2D_LINE_CLIP => format!("{name} x {}..{} y {}..{}", i(0), i(1), i(2), i(3)),
            HQ2_2D_LINE_MODE | HQ2_2D_STIPPLE_AUX => format!("{name} {:#x}", a[0]),
            HQ2_2D_TILE_RECT | HQ2_2D_TILE_RECT_ODD => format!("{name} ({}, {})-({}, {}) origin ({}, {}) tile {}x{} format={} ({} words)",
                i(2), i(4), i(5), i(6), i(1), i(3), self.tile_w, self.tile_h, self.tile_fmt, self.tile_len),
            HQ2_2D_STIPPLE_RECT | HQ2_2D_STIPPLE_BOX_POW2 | HQ2_2D_STIPPLE_BOX => format!("{name} ({}, {})-({}, {}) origin ({}, {}) stipple {}x{} opaque={} fg={:#x} bg={:#x}",
                i(2), i(4), i(5), i(6), i(1), i(3), self.tile_w, self.tile_h, self.stipple_opaque, self.s2d.color_on, self.s2d.color_off),
            HQ2_2D_MODE | HQ2_2D_COLOR_AUX | HQ2_2D_COLOR_OFF | HQ2_2D_COLOR_ON | HQ2_2D_CID_WRITE
            | HQ2_2D_BEGIN | HQ2_2D_END_PRIMITIVE => format!("{name} {:#x}", a[0]),
            HQ2_2D_BUF_SELECT => format!("{name} format={} offset={}", i(0), i(1)),
            HQ2_2D_DMA_READ_PIXELS => format!("{name} ({}, {}) {}x{} words/row={} flag={} format={} offset={} mode={:#x} -> GEDMA, FIN2",
                i(0), i(1), i(2), i(3), i(4), i(5), self.s2d.buf_select, self.s2d.buf_offset, self.s2d.mode),
            HQ2_GL_DMA_READ => format!("{name} ({}, {}) {}x{} words/row={} flag={} {} -> GEDMA, FIN2",
                i(0), i(1), i(2), i(3), i(4), i(5), self.gl_read_desc()),
            HQ2_2D_READ_IMAGE => format!("{name} ({}, {}) {}x{} words/row={} format={} offset={} mode={:#x} -> shram[{:#x}], FIN3",
                i(1), i(2), i(3), i(4), i(0), self.s2d.buf_select, self.s2d.buf_offset, self.s2d.mode, READ_IMAGE_SHRAM),
            PUC_COLOR => format!("{name} ci={}", i(0)),
            PUC_PNT2I | PUC_CMOV2I => format!("{name} ({}, {})", i(0), i(1)),
            PUC_RECTI2D => format!("{name} ({}, {})-({}, {})", i(0), i(1), i(2), i(3)),
            PUC_DRAWCHAR => {
                let rows: Vec<String> = a.iter().skip(7).take(i(1).clamp(0, 18) as usize)
                    .map(|r| format!("{r:04x}")).collect();
                format!("{name} {}x{} mode={} orig=({}, {}) move=({}, {}) raster now ({}, {}) rows=[{}]",
                    i(0), i(1), i(2), i(3), i(4), i(5), i(6), self.cx, self.cy, rows.join(" "))
            }
            PUC_RECTCOPY => format!("{name} src=({}, {}) {}x{} dst=({}, {}) [X-style y] word_len={} max_lines={}",
                i(2), i(3), i(4), i(5), i(6), i(7), i(0), i(1)),
            _ if self.gl_describe(cmd).is_some() => self.gl_describe(cmd).unwrap_or_default(),
            _ => {
                let words: Vec<String> = a.iter().map(|w| format!("{w:#x}")).collect();
                let more = self.last_total.saturating_sub(a.len() as u32);
                let more = if more > 0 { format!(" +{more} more") } else { String::new() };
                format!("{name} [{}{more}] ({} words, not implemented)", words.join(", "), self.last_total)
            }
        }
    }

    /// Current interpreter state, for the monitor.
    pub fn summary(&self) -> String {
        format!("pending={} {}/{} words  raster=({}, {})  pattern_on={}",
            if self.need != 0 { token_name(self.cmd).unwrap_or("?").to_string() } else { "-".into() },
            self.nargs, self.need, self.cx, self.cy, self.pattern_on)
    }

    fn execute(&mut self, cmd: u32, out: &mut dyn Re3Sink) {
        let a = self.args;
        match cmd {
            PUC_INIT => self.init(out),
            PUC_COLOR => {
                out.reg(re3::REG_R, (a[0] & 0xfff) << 11);
                out.reg(re3::REG_G, 0);
                out.reg(re3::REG_B, 0);
            }
            PUC_FINISH => {}
            // Context switch (Gr2PcxSwap): swap the HLE's GL state; the
            // kernel waits for FIN2.
            GE_HQMSAV => {
                self.gl_switch_context(a[0], a[1], out);
                out.finish(FIN2);
            }
            // Save main: hand out the image taken at GE_HQMSAV; the kernel
            // reads it from HQ2_GEDMA after FIN2. Save ext: nothing to save.
            GE_CX_SAVE_MAIN => {
                self.gl_cx_save(out);
                out.finish(FIN2);
            }
            GE_CX_SAVE_EXT => {
                out.gedma_out(&[]);
                out.finish(FIN2);
            }
            HQ_PCX_1E0 | HQ_PCX_1E6 => out.finish(FIN2),
            HQ_GL_FIN3 | GL_FINISH => out.finish(FIN3),
            // Nothing is buffered beyond the FIFO.
            GL_FLUSH => {}
            // First word of the DDX's hardware init: make sure the raster
            // engine is in a known full-screen state for 2D drawing.
            HQ2_2D_BEGIN => {
                for (reg, val) in [
                    (re3::REG_XMIN, (0)),
                    (re3::REG_XMAX, (re3::FB_W as u32 - 1)),
                    (re3::REG_YMIN, 0),
                    (re3::REG_YMAX, re3::FB_H as u32 - 1),
                    (re3::REG_ENABPAT, 0),
                    (re3::REG_ENABWID, 0),
                    (re3::REG_RWMODE, re3::RWMODE_FB),
                ] {
                    out.reg(reg, val);
                }
                self.pattern_on = 0;
                self.line_clip = [0, re3::FB_W as i32 - 1, 0, SCREEN_H - 1];
            }
            HQ2_2D_END_PRIMITIVE => {}
            HQ2_2D_MODE => self.s2d.mode = a[0],
            HQ2_2D_COLOR_AUX => self.s2d.aux_mask = a[0],
            HQ2_2D_COLOR_OFF => self.s2d.color_off = a[0],
            HQ2_2D_COLOR_ON => self.s2d.color_on = a[0],
            HQ2_2D_CID_WRITE => self.s2d.cid_write = a[0],
            HQ2_2D_BUF_SELECT => {
                self.s2d.buf_select = a[0];
                self.s2d.buf_offset = a[1];
            }
            HQ2_2D_READ_IMAGE => {
                let req = ReadImage {
                    x: a[1] as i32,
                    top: SCREEN_H - 1 - a[2] as i32,
                    w: a[3],
                    rows: a[4],
                    words_per_row: a[0],
                    s2d: self.s2d,
                    decode: ReadDecode::Planes2d,
                };
                out.read_image(&req, ReadDest::Shram);
            }
            HQ2_2D_DMA_READ_PIXELS => {
                let req = ReadImage {
                    x: a[0] as i32,
                    top: SCREEN_H - 1 - a[1] as i32,
                    w: a[2],
                    rows: a[3],
                    words_per_row: a[4],
                    s2d: self.s2d,
                    decode: ReadDecode::Planes2d,
                };
                out.read_image(&req, ReadDest::Gedma);
            }
            HQ2_GL_DMA_READ => self.gl_dma_read(&a, out),
            HQ2_2D_TILE_SETUP => {
                self.tile_w = a[1];
                self.tile_h = a[2];
                self.tile_fmt = a[3];
                self.tile_len = 0;
            }
            HQ2_2D_MONO_IMAGE_8..=HQ2_2D_MONO_IMAGE_64 => {
                self.mono_image(4 << (cmd - HQ2_2D_MONO_IMAGE_8), out)
            }
            HQ2_2D_GLYPH_8..=HQ2_2D_GLYPH_64 => self.mono_image(8 << (cmd - HQ2_2D_GLYPH_8), out),
            HQ2_2D_LINE_MODE => {}
            HQ2_2D_LINE_CLIP => self.line_clip = [a[0] as i32, a[1] as i32, a[2] as i32, a[3] as i32],
            HQ2_2D_STIPPLE_AUX => self.stipple_opaque = a[0] & 1,
            // One box per command: 0; DATA origin x, x1, origin y, y1, x2, y2
            // (same interleaved order as STIPPLE_RECT; expTileRects).
            HQ2_2D_TILE_RECT | HQ2_2D_TILE_RECT_ODD => {
                let (ox, oy) = (a[1] as i32, a[3] as i32);
                if self.tile_fmt == 3 {
                    self.stipple_rect(ox, oy, [a[2] as i32, a[4] as i32, a[5] as i32, a[6] as i32], out);
                } else {
                    self.tile_rect(ox, oy, [a[2] as i32, a[4] as i32, a[5] as i32, a[6] as i32], out);
                }
            }
            HQ2_2D_STIPPLE_RECT | HQ2_2D_STIPPLE_BOX_POW2 | HQ2_2D_STIPPLE_BOX => {
                let (ox, oy) = (a[1] as i32, a[3] as i32);
                self.stipple_rect(ox, oy, [a[2] as i32, a[4] as i32, a[5] as i32, a[6] as i32], out);
            }
            HQ2_2D_COPY_RECT => {
                // X-style y (top-down); RE3 copy takes GL bottom-row y.
                let (sx, sy, w, h, dx, dy) = (a[2] as i32, a[3] as i32, a[4] as i32, a[5] as i32, a[6] as i32, a[7] as i32);
                let fg = self.s2d.fg;
                self.setup2d(fg, out);
                out.copy(sx, SCREEN_H - sy - h, w, h, dx, SCREEN_H - dy - h);
            }
            HQ2_2D_ROP => {
                self.s2d.fg = a[0];
                self.s2d.planemask = a[1];
                self.s2d.alu = a[2];
                self.s2d.flag = a[3];
            }
            // Microcode reset path after unstall (kernel Gr2Start): the start
            // argument (0/1/2 by board rev) resets drawing state, then FIN2.
            HQ_TOKEN_UNSTALL => {
                self.init(out);
                out.finish(FIN2);
            }
            PUC_PNT2I => self.span(a[0] as i32, a[1] as i32, 1, None, out),
            PUC_RECTI2D => {
                let (x1, y1, x2, y2) = (a[0] as i32, a[1] as i32, a[2] as i32, a[3] as i32);
                let (xl, xr) = (x1.min(x2), x1.max(x2));
                for y in y1.min(y2)..=y1.max(y2) {
                    self.span(xl, y, (xr - xl + 1) as u32, None, out);
                }
            }
            PUC_CMOV2I => {
                self.cx = a[0] as i32;
                self.cy = a[1] as i32;
            }
            PUC_DRAWCHAR => self.drawchar(out),
            PUC_RECTCOPY => {
                let (sx, sy, w, h, dx, dy) =
                    (a[2] as i32, a[3] as i32, a[4] as i32, a[5] as i32, a[6] as i32, a[7] as i32);
                // RECTCOPY takes X-style y (top edge, top-down); VRAM is bottom-up.
                out.copy(sx, SCREEN_H - sy - h, w, h, dx, SCREEN_H - dy - h);
            }
            _ if self.gl_execute(cmd, out) => {}
            _ => {
                let i = cmd as usize & (FIFO_WORDS - 1);
                if self.warned[i / 64] & (1 << (i % 64)) == 0 {
                    self.warned[i / 64] |= 1 << (i % 64);
                    crate::dlog_dev!(crate::devlog::LogModule::Gr2,
                        "HQ2: unimplemented command {} ({:#x}) data {:#010x}", cmd, cmd, a[0]);
                }
            }
        }
    }

    /// PUC_INIT: put the RE3 into the textport's drawing state.
    fn init(&mut self, out: &mut dyn Re3Sink) {
        self.cx = 0;
        self.cy = 0;
        self.pattern_on = 0;
        for (reg, val) in [
            (re3::REG_FUNC, re3::ROP_COPY),
            (re3::REG_ENABRGB, 0),
            (re3::REG_NOPUP, 0),
            (re3::REG_PUPDATA, 0),
            (re3::REG_PIXMASK, 0x00ff_ffff),
            (re3::REG_AUXMASK, 0),
            (re3::REG_WIDDATA, 0),
            (re3::REG_UAUXDATA, 0),
            (re3::REG_RWMODE, re3::RWMODE_FB),
            (re3::REG_ENABPAT, 0),
            (re3::REG_ALIGNPAT, 0),
            (re3::REG_ENABWID, 0),
            (re3::REG_XMIN, (0)),
            (re3::REG_XMAX, (re3::FB_W as u32 - 1)),
            (re3::REG_YMIN, 0),
            (re3::REG_YMAX, re3::FB_H as u32 - 1),
            (re3::REG_DX, 0),
            (re3::REG_DY, 0),
            (re3::REG_DR, 0),
            (re3::REG_DG, 0),
            (re3::REG_DB, 0),
            (re3::REG_R, 0),
            (re3::REG_G, 0),
            (re3::REG_B, 0),
        ] {
            out.reg(reg, val);
        }
    }

    /// One horizontal flat span, optionally masked by a 32-bit pattern
    /// (bit 31 = leftmost pixel).
    fn span(&mut self, x: i32, y: i32, n: u32, pattern: Option<u32>, out: &mut dyn Re3Sink) {
        let (mut x, mut n, mut pat) = (x, n as i32, pattern.unwrap_or(0));
        if x < 0 {
            // Bank-coded X can't express negative columns: trim the span.
            n += x;
            if pattern.is_some() {
                pat = if -x >= 32 { 0 } else { pat << -x };
            }
            x = 0;
        }
        if n <= 0 || y < 0 || y >= re3::FB_H as i32 || x >= re3::FB_W as i32 {
            return;
        }
        let want_pat = pattern.is_some() as u32;
        if want_pat != self.pattern_on {
            out.reg(re3::REG_ENABPAT, want_pat);
            self.pattern_on = want_pat;
        }
        if pattern.is_some() {
            out.reg(re3::REG_PATH, pat >> 16);
            out.reg(re3::REG_PATL, pat & 0xffff);
        }
        out.reg(re3::REG_X, x as u32);
        out.reg(re3::REG_Y, y as u32);
        out.reg(re3::REG_NUMPIX, n as u32);
        out.reg(re3::REG_IR, re3::IR_FLAT);
    }

    /// PUC_DRAWCHAR (GR2.h): xsize; ysize, mode, xorig, yorig, xmove, ymove,
    /// then 18 rows, bottom row first, MSB = leftmost pixel.
    fn drawchar(&mut self, out: &mut dyn Re3Sink) {
        let a = self.args;
        let xsize = a[0].min(32);
        let ysize = a[1].min(18);
        let mode = a[2];
        let (xorig, yorig) = (a[3] as i32, a[4] as i32);
        let (xmove, ymove) = (a[5] as i32, a[6] as i32);
        let left = self.cx - xorig;
        let bottom = self.cy - yorig;
        for k in 0..ysize as usize {
            let row = a[7 + k];
            let mut pat = if mode == 2 { row } else { (row & 0xffff) << 16 };
            if xsize < 32 {
                pat &= !(u32::MAX >> xsize); // keep only the leftmost xsize bits
            }
            if pat != 0 {
                self.span(left, bottom + k as i32, xsize, Some(pat), out);
            }
        }
        self.cx += xmove;
        self.cy += ymove;
    }
}
