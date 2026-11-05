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
//! No dirty lines: every frame is composed whole from memory, as the board
//! scans it out.

use super::display::{IdShow, ResolveCtx};
use super::field::{Codec, Field, Layout, FIELD_FOUR, FIELD_INDEX, FIELD_PACKED, PIXEL_FMT_B};
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
    fb: Surface,
    /// Per main window id, its two colour fields (mode_fields()).
    fields: [[Field; 2]; dispmode::COUNT as usize],
    wid: Surface,
    wid_layout: Layout,
    ctx: ResolveCtx,
}

impl Frame {
    /// The frame the registers describe, with buffer (displayed >> id) & 1
    /// showing for each window; None before the scan-out is set up.
    pub fn snapshot(regs: &DisplayRegs, displayed: u32) -> Option<Frame> {
        let r = &regs.pbj;
        let (width, height, stride) = (r[pbj::WIDTH], r[pbj::HEIGHT], r[pbj::STRIDE]);
        if width == 0 || height == 0 || stride == 0 || width > stride * TILE {
            return None;
        }
        let fb_log2 = (r[pbj::DEPTH] & 7) as u8 + 1;
        let wid_format = Format { fb_log2: 1, win_log2: 1, pixel_fmt: PIXEL_FMT_B, ..Default::default() };
        Some(Frame {
            width,
            height,
            fb: Surface::new(r[pbj::PIXEL_BASE], stride, fb_log2),
            fields: std::array::from_fn(|id| mode_fields(regs.dispmode[id])),
            wid: Surface::new(r[pbj::OVERLAY_BASE], stride, 1),
            wid_layout: Layout::of(wid_format),
            ctx: ResolveCtx::new(regs, displayed),
        })
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

    /// One displayed pixel as 0x00RRGGBB.
    #[inline]
    fn pixel(&self, mem: VramPtr, x: u32, y: u32) -> u32 {
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
        for y in 0..(self.height as usize).min(rows) as u32 {
            let row = &mut out[y as usize * stride..y as usize * stride + w];
            for (x, o) in row.iter_mut().enumerate() {
                *o = self.pixel(mem, x as u32, y);
            }
            regs.cursor_row(y, row, w as u32);
        }
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
