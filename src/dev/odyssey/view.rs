//! One field of a surface in board memory, as a drawing reaches it: the
//! colour of the front or back buffer, depth, stencil, the window-id byte or
//! the overlay byte. Built from the registers that name it (the pixel mode,
//! block 0x017; the scan-out bases for the window-id buffer) and nothing
//! else; `Rows` adds the memory.
//!
//! The accessors trade in the model's words: a colour as 0xRRGGBBAA, depth
//! as its 24-bit value, a byte as itself; a field stored any other way
//! converts through its codec (field.rs). Memory is words (vram.rs): a
//! field is a bit range of one.

use super::field::{Codec, Field, Layout, FIELD_FOUR, PIXEL_FMT_B};
use super::vram::{Format, Surface, VramPtr, PITCH_MASK, TILE, UNIT};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Kind {
    /// No such field: loads find nothing, stores are dropped.
    #[default]
    None,
    /// R, G, B, A, 8 bits each, R highest.
    Rgba8,
    /// A 24-bit depth word in three bytes.
    Z24,
    Byte,
    /// Any other field (field.rs), converted through the 0xRRGGBBAA word.
    Coded(Codec),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct View {
    /// The surface's first byte.
    base: u64,
    pitch: u32,
    log2: u8,
    /// The field's first byte inside the pixel.
    at: u8,
    pub kind: Kind,
    pub width: u32,
    pub height: u32,
}

/// The largest drawable on a side (published); a surface's rows past it
/// are no drawing's.
pub(crate) const MAX_DIM: u32 = 4096;

impl View {
    fn of(s: &Surface, at: u8, kind: Kind, height: u32) -> View {
        View {
            base: s.base as u64 * UNIT,
            pitch: s.pitch,
            log2: s.bpp_log2,
            at,
            kind,
            width: s.pitch * TILE,
            height,
        }
    }

    /// Colour field `f` of surface `s`.
    pub fn colour(s: &Surface, f: Field, height: u32) -> View {
        let kind = match f.codec {
            Codec::Texel { bytes: 4, field: FIELD_FOUR } => Kind::Rgba8,
            Codec::Byte => Kind::Byte,
            c => Kind::Coded(c),
        };
        View::of(s, f.at, kind, height)
    }

    /// The depth and stencil fields of layout `l` over `s`.
    pub fn depth_stencil(s: &Surface, l: &Layout, height: u32) -> (View, View) {
        match l.depth {
            Some(at) => (View::of(s, at, Kind::Z24, height), View::of(s, at + 3, Kind::Byte, height)),
            None => (View::default(), View::default()),
        }
    }

    /// The byte address of the field at (x, y), when it is in the view.
    #[inline(always)]
    pub fn offset(&self, x: i32, y: i32) -> Option<u64> {
        if self.kind == Kind::None || x < 0 || y < 0 || x as u32 >= self.width || y as u32 >= self.height {
            return None;
        }
        Some(self.addr(x, y))
    }

    /// The byte address of the field at (x, y), unchecked.
    #[inline(always)]
    pub fn addr(&self, x: i32, y: i32) -> u64 {
        const { assert!(TILE == 16) };
        let (x, y) = (x as u32 as u64, y as u32 as u64);
        let tile = (y >> 4).wrapping_mul(self.pitch as u64).wrapping_add(x >> 4);
        self.base.wrapping_add((tile << 8 | (y & 15) << 4 | (x & 15)) << self.log2).wrapping_add(self.at as u64)
    }

    /// The view's placement: the surface's first byte, its pitch in tiles,
    /// log2 of its pixel size and the field's byte inside the pixel (for
    /// the GPU instantiation's slots).
    #[cfg_attr(not(feature = "wgpu"), allow(dead_code, reason = "used by the GPU raster unit"))]
    pub fn parts(&self) -> (u64, u32, u8, u8) {
        (self.base, self.pitch, self.log2, self.at)
    }

    /// The view `by` bytes further into the pixel.
    pub fn shifted(mut self, by: i64) -> View {
        self.at = (self.at as i64 + by) as u8;
        self
    }

    fn field(&self) -> Field {
        let codec = match self.kind {
            Kind::Coded(c) => c,
            _ => Codec::Byte,
        };
        Field { at: 0, codec }
    }

    /// Bytes the field covers: what a raw copy moves.
    pub fn len(&self) -> u32 {
        match self.kind {
            Kind::None => 0,
            Kind::Rgba8 => 4,
            Kind::Z24 => 3,
            Kind::Byte => 1,
            // The field is 8 bytes; its lanes fill the first 6, and a raw
            // copy moves all 8.
            Kind::Coded(Codec::Lanes) => 8,
            Kind::Coded(Codec::Byte) => 1,
            Kind::Coded(Codec::Texel { bytes, .. }) => bytes as u32,
        }
    }

    /// Bytes the field holds its value in (a coded field's codec).
    fn value_len(&self) -> u32 {
        match self.kind {
            Kind::Coded(_) => self.field().value_len(),
            _ => self.len(),
        }
    }
}

/// Where a drawing lands, from the board's registers alone: the pixel
/// mode's colour fields (A and B, chosen by the buffer select), its depth
/// and stencil, and the window-id and overlay bytes of the window-id buffer
/// the scan-out names (PB&J 0x2800 base, 0x2807 stride), which the id test
/// reads.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DestViews {
    pub colour: [View; 2],
    pub depth: View,
    pub stencil: View,
    pub did: View,
    pub overlay: View,
    /// The status word's SCRATCH_BUFFER: no buffer select, window clip,
    /// id test or window offset.
    pub scratch: bool,
}

impl DestViews {
    pub fn of(pixel_mode: &[u32; 4], wid_base: u32, wid_stride: u32) -> DestViews {
        let f = Format::decode(pixel_mode[2]);
        let s = Surface::new(pixel_mode[0], pixel_mode[1] & PITCH_MASK, f.fb_log2);
        let l = Layout::of(f);
        let (depth, stencil) = View::depth_stencil(&s, &l, MAX_DIM);
        let w = Surface::new(wid_base, wid_stride, 1);
        let wl = Layout::of(Format { fb_log2: 1, win_log2: 1, pixel_fmt: PIXEL_FMT_B, ..Default::default() });
        DestViews {
            colour: [View::colour(&s, l.front, MAX_DIM), View::colour(&s, l.back, MAX_DIM)],
            depth,
            stencil,
            did: if wid_stride != 0 { View::colour(&w, wl.front, MAX_DIM) } else { View::default() },
            overlay: if wid_stride != 0 { View::colour(&w, wl.back, MAX_DIM) } else { View::default() },
            scratch: f.scratch,
        }
    }
}

/// A view with the memory it reaches.
#[derive(Clone, Copy)]
pub(crate) struct Rows {
    mem: VramPtr,
    pub v: View,
}

impl Default for Rows {
    fn default() -> Rows {
        Rows { mem: VramPtr::null(), v: View::default() }
    }
}

impl Rows {
    pub fn new(mem: VramPtr, v: View) -> Rows {
        Rows { mem, v }
    }

    /// The memory, for a store that addresses the field itself (`addr`).
    #[inline(always)]
    pub fn mem(&self) -> VramPtr {
        self.mem
    }

    /// The field's byte address at (x, y), unchecked: the rasteriser and
    /// the clip rectangles keep a drawing in its window. For a view of a
    /// field the surface has (`kind` not None).
    #[inline(always)]
    pub fn addr(&self, x: i32, y: i32) -> u64 {
        self.v.addr(x, y)
    }

    /// The field's word (see the module), when (x, y) is in the view.
    #[inline(always)]
    pub fn load_word(&self, x: i32, y: i32) -> Option<u32> {
        let a = self.v.offset(x, y)?;
        let v = self.mem.load(a, self.v.value_len());
        Some(match self.v.kind {
            Kind::Coded(_) => self.v.field().load(v),
            _ => v as u32,
        })
    }

    #[inline(always)]
    pub fn store_word(&self, x: i32, y: i32, w: u32) {
        let Some(a) = self.v.offset(x, y) else { return };
        let v = match self.v.kind {
            Kind::Coded(_) => self.v.field().store(w),
            Kind::Z24 => w as u64 & 0xff_ffff,
            Kind::Byte => w as u64 & 0xff,
            _ => w as u64,
        };
        self.mem.store(a, self.v.value_len(), v);
    }

    #[inline(always)]
    pub fn load1(&self, x: i32, y: i32) -> Option<u8> {
        let a = self.v.offset(x, y)?;
        Some(self.mem.load(a, 1) as u8)
    }

    #[inline(always)]
    pub fn store1(&self, x: i32, y: i32, v: u8) {
        if let Some(a) = self.v.offset(x, y) {
            self.mem.store(a, 1, v as u64);
        }
    }

    /// The field's raw value, all `len()` bytes of it, unconverted.
    pub fn load_raw(&self, x: i32, y: i32) -> u64 {
        self.v.offset(x, y).map_or(0, |a| self.mem.load(a, self.v.len()))
    }

    pub fn store_raw(&self, x: i32, y: i32, v: u64) {
        if let Some(a) = self.v.offset(x, y) {
            self.mem.store(a, self.v.len(), v);
        }
    }

    /// Store the same word at pixels x0 to x1 of row y, those in the view.
    /// The word is encoded for the field once, as store_word() encodes it.
    pub fn fill(&self, y: i32, x0: i32, x1: i32, w: u32) {
        let (x0, x1) = (x0.max(0), x1.min(self.v.width as i32 - 1));
        if self.v.kind == Kind::None || y < 0 || y as u32 >= self.v.height || x0 > x1 {
            return;
        }
        let v = match self.v.kind {
            Kind::Coded(_) => self.v.field().store(w),
            Kind::Z24 => w as u64 & 0xff_ffff,
            Kind::Byte => w as u64 & 0xff,
            _ => w as u64,
        };
        let len = self.v.value_len();
        for x in x0..=x1 {
            self.mem.store(self.v.addr(x, y), len, v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::vram::Vram;

    #[test]
    fn a_double_buffered_window_keeps_its_fields_apart() {
        let mut m = Vram::new(1 << 20);
        let s = Surface::new(0x100, 2, 4);
        let l = Layout::of(Format::decode(0x20406d50));
        let mem = m.ptr();
        let front = Rows::new(mem, View::colour(&s, l.front, 64));
        let back = Rows::new(mem, View::colour(&s, l.back, 64));
        let (dv, sv) = View::depth_stencil(&s, &l, 64);
        let (depth, stencil) = (Rows::new(mem, dv), Rows::new(mem, sv));
        front.store_word(5, 17, 0x0102_0304);
        back.store_word(5, 17, 0x0506_0708);
        depth.store_word(5, 17, 0xab_cdef);
        stencil.store1(5, 17, 0x5a);
        assert_eq!(front.load_word(5, 17), Some(0x0102_0304));
        assert_eq!(back.load_word(5, 17), Some(0x0506_0708));
        assert_eq!(depth.load_word(5, 17), Some(0xab_cdef));
        assert_eq!(stencil.load1(5, 17), Some(0x5a));
        // What the scan-out's display mode for this window reads.
        let shown = super::super::frame::mode_fields(0x009655a6);
        let px = s.offset(5, 17);
        let read = |f: &super::super::field::Field| f.load(mem.load(px + f.at as u64, f.value_len()));
        assert_eq!(read(&shown[0]), 0x0102_0304);
        assert_eq!(read(&shown[1]), 0x0506_0708);
        assert_eq!(depth.load_word(6, 17), Some(0));
        assert_eq!(front.load_word(32, 0), None);
    }

    #[test]
    fn a_4444_field_converts_on_the_way() {
        let mut m = Vram::new(1 << 20);
        let s = Surface::new(0x100, 2, 3);
        let l = Layout::of(Format::decode(0x49d0));
        assert_eq!((l.front.at, l.back.at, l.depth), (6, 4, Some(0)));
        let front = Rows::new(m.ptr(), View::colour(&s, l.front, 64));
        front.store_word(1, 1, 0xa520_ff00);
        assert_eq!(front.load_word(1, 1), Some(0xaa22_ff00));
    }
}
