//! One field of a surface in board memory, as a drawing reaches it: the
//! colour of the front or back buffer, depth, stencil, the window-id byte or
//! the overlay byte. Built from the registers that name it (the pixel mode,
//! block 0x017; the scan-out bases for the window-id buffer) and nothing
//! else; `Rows` adds the memory.
//!
//! The 4-byte accessors keep the shape the fragment back end was written
//! for: a colour as its R, G, B, A bytes and depth as a big-endian word with
//! a zero top byte. A field stored any other way converts on the way.

use super::field::{Codec, Field, Layout, FIELD_FOUR, PIXEL_FMT_B};
use super::vram::{Format, Surface, VramPtr, PITCH_MASK, TILE, UNIT};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Kind {
    /// No such field: loads find nothing, stores are dropped.
    #[default]
    None,
    /// R, G, B, A bytes.
    Rgba8,
    /// A 24-bit depth word in three bytes, big-endian.
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

    /// The byte offset of the field at (x, y), when it is in the view.
    #[inline(always)]
    pub fn offset(&self, x: i32, y: i32) -> Option<usize> {
        if self.kind == Kind::None || x < 0 || y < 0 || x as u32 >= self.width || y as u32 >= self.height {
            return None;
        }
        let (x, y) = (x as u32, y as u32);
        let tile = (y / TILE) as u64 * self.pitch as u64 + (x / TILE) as u64;
        let pixel = tile * (TILE * TILE) as u64 + ((y % TILE) * TILE + x % TILE) as u64;
        Some((self.base + (pixel << self.log2) + self.at as u64) as usize)
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

    /// Bytes the field covers.
    fn len(&self) -> usize {
        match self.kind {
            Kind::None => 0,
            Kind::Rgba8 => 4,
            Kind::Z24 => 3,
            Kind::Byte => 1,
            // The field is 8 bytes; its lanes fill the first 6, and a raw
            // copy moves all 8.
            Kind::Coded(Codec::Lanes) => 8,
            Kind::Coded(Codec::Byte) => 1,
            Kind::Coded(Codec::Texel { bytes, .. }) => bytes as usize,
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
    ptr: *mut u8,
    len: usize,
    pub v: View,
}

// SAFETY: as VramPtr: the memory outlives the draws, which race as the
// board's units do.
unsafe impl Send for Rows {}
unsafe impl Sync for Rows {}

impl Default for Rows {
    fn default() -> Rows {
        Rows { ptr: std::ptr::null_mut(), len: 0, v: View::default() }
    }
}

impl Rows {
    pub fn new(mem: VramPtr, v: View) -> Rows {
        // SAFETY: only the pointer and length are taken here.
        let b = unsafe { mem.bytes() };
        Rows { ptr: b.as_mut_ptr(), len: b.len(), v }
    }

    /// The field's bytes at (x, y), when they are in the view and memory.
    #[inline(always)]
    fn at(&self, x: i32, y: i32) -> Option<&mut [u8]> {
        let off = self.v.offset(x, y)?;
        let n = self.v.len();
        // SAFETY: inside the memory; see the type.
        (off + n <= self.len).then(|| unsafe { std::slice::from_raw_parts_mut(self.ptr.add(off), n) })
    }

    /// The field in the back end's 4-byte shape (see the module).
    #[inline(always)]
    pub fn load4(&self, x: i32, y: i32) -> Option<[u8; 4]> {
        let b = self.at(x, y)?;
        Some(match self.v.kind {
            Kind::Rgba8 => [b[0], b[1], b[2], b[3]],
            Kind::Z24 => [0, b[0], b[1], b[2]],
            Kind::Byte => [0, 0, 0, b[0]],
            Kind::Coded(_) => self.v.field().load(b).to_be_bytes(),
            Kind::None => return None,
        })
    }

    #[inline(always)]
    pub fn store4(&self, x: i32, y: i32, v: [u8; 4]) {
        let Some(b) = self.at(x, y) else { return };
        match self.v.kind {
            Kind::Rgba8 => b.copy_from_slice(&v),
            Kind::Z24 => b.copy_from_slice(&v[1..]),
            Kind::Byte => b[0] = v[3],
            Kind::Coded(_) => self.v.field().store(b, u32::from_be_bytes(v)),
            Kind::None => {}
        }
    }

    #[inline(always)]
    pub fn load1(&self, x: i32, y: i32) -> Option<u8> {
        self.at(x, y).map(|b| b[0])
    }

    #[inline(always)]
    pub fn store1(&self, x: i32, y: i32, v: u8) {
        if let Some(b) = self.at(x, y) {
            b[0] = v;
        }
    }

    /// The field as the model's word: 0xRRGGBBAA, a colour index, the depth
    /// word or the byte.
    #[inline]
    pub fn load_word(&self, x: i32, y: i32) -> Option<u32> {
        self.load4(x, y).map(u32::from_be_bytes)
    }

    #[inline]
    pub fn store_word(&self, x: i32, y: i32, w: u32) {
        self.store4(x, y, w.to_be_bytes())
    }

    /// The field's stored bytes, at most 8, unconverted.
    pub fn load_raw(&self, x: i32, y: i32, out: &mut [u8; 8]) {
        if let Some(b) = self.at(x, y) {
            out[..b.len()].copy_from_slice(b);
        }
    }

    pub fn store_raw(&self, x: i32, y: i32, v: &[u8; 8]) {
        if let Some(b) = self.at(x, y) {
            let n = b.len();
            b.copy_from_slice(&v[..n]);
        }
    }

    /// Store the same value at pixels x0 to x1 of row y, those in the view.
    pub fn fill4(&self, y: i32, x0: i32, x1: i32, v: [u8; 4]) {
        let (x0, x1) = (x0.max(0), x1.min(self.v.width as i32 - 1));
        for x in x0..=x1 {
            self.store4(x, y, v);
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
        front.store4(5, 17, [1, 2, 3, 4]);
        back.store4(5, 17, [5, 6, 7, 8]);
        depth.store4(5, 17, [0, 0xab, 0xcd, 0xef]);
        stencil.store1(5, 17, 0x5a);
        assert_eq!(front.load4(5, 17), Some([1, 2, 3, 4]));
        assert_eq!(back.load4(5, 17), Some([5, 6, 7, 8]));
        assert_eq!(depth.load4(5, 17), Some([0, 0xab, 0xcd, 0xef]));
        assert_eq!(stencil.load1(5, 17), Some(0x5a));
        // What the scan-out's display mode for this window reads.
        let shown = super::super::frame::mode_fields(0x009655a6);
        let px = m.pixel(&s, 5, 17).unwrap();
        assert_eq!(shown[0].load(px), 0x0102_0304);
        assert_eq!(shown[1].load(px), 0x0506_0708);
        assert_eq!(depth.load4(6, 17), Some([0, 0, 0, 0]));
        assert_eq!(front.load4(32, 0), None);
    }

    #[test]
    fn a_4444_field_converts_on_the_way() {
        let mut m = Vram::new(1 << 20);
        let s = Surface::new(0x100, 2, 3);
        let l = Layout::of(Format::decode(0x49d0));
        assert_eq!((l.front.at, l.back.at, l.depth), (6, 4, Some(0)));
        let front = Rows::new(m.ptr(), View::colour(&s, l.front, 64));
        front.store4(1, 1, [0xa5, 0x20, 0xff, 0x00]);
        assert_eq!(front.load4(1, 1), Some([0xaa, 0x22, 0xff, 0x00]));
    }
}
