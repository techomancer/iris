//! What the bytes of one pixel hold, by the pixel mode's status word: a front
//! field, a back field (the buffer select, block 0x024, picks between them:
//! 0x3 front, 0xc back) and, for most visuals, a depth and stencil field.
//!
//! Where each sits is the GL library's own map (its DMA field-offset
//! routine, traced by hand): by WIN_DEPTH and PIXEL_FMT it gives the front
//! and back colour fields and the depth field as offsets in 2-byte units
//! from the start of the pixel, the units the DMA header's word 4 field
//! codes (bits 23:20 front, 27:24 back) carry. An 8-byte window in a
//! 16-byte frame buffer uses the pixel's first 8 bytes. For example a
//! double-buffered 8:8:8:8 window with depth: front colour bytes 12-15, back
//! 4-7, depth and stencil 8-11; single-buffered: colour 4-7, depth 0-3.
//! (`ignore/odsy/docs/measured.md` section 8.)
//!
//! MODEL CHOICES, which nothing the host can read shows: a 4-byte depth field
//! is the 24-bit depth word big-endian then the stencil byte; an 8-byte colour
//! field (the 12-bit visuals) is four 12-bit lanes, R first, in its first 6
//! bytes; a 2- or 4-byte colour field packs as a texel of that size and
//! FIELD_FMT. The overlay and window-id buffer (B format, 2 bytes) is the
//! window-id byte (front) then the overlay byte (back), the buffer select
//! choosing between them as the X server's id writes do.
//!
//! Words in and out are the model's currency: 0xRRGGBBAA with 8 bits a
//! channel, or a colour index in the low bits. Lanes keep any 32-bit word
//! exactly (each byte becomes the top of its lane); the 12-bit back end
//! stores lanes directly.

use super::vram::{Format, Surface};

/// PIXEL_FMT of the overlay and window-id buffer (the B format).
pub(super) const PIXEL_FMT_B: u8 = 1;
/// FIELD_FMT values: one component, two, four equal, packed 5:5:5:1 or
/// 10:10:10:2.
pub(super) const FIELD_ONE: u8 = 2;
pub(super) const FIELD_TWO: u8 = 3;
pub(super) const FIELD_FOUR: u8 = 5;
pub(super) const FIELD_PACKED: u8 = 6;
/// The model's FIELD_FMT for a colour-index field (the board's is 0): the
/// index in the field's low bits, big-endian.
pub(super) const FIELD_INDEX: u8 = 0;

/// How a field's bytes hold a word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Codec {
    /// Four 12-bit lanes in 6 bytes, R first, big-endian bit order.
    Lanes,
    /// One byte.
    Byte,
    /// A texel of `bytes` bytes in FIELD_FMT `field`.
    Texel { bytes: u8, field: u8 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Field {
    pub at: u8,
    pub codec: Codec,
}

/// One pixel's fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Layout {
    pub front: Field,
    pub back: Field,
    /// Where the 24-bit depth word starts; the stencil byte follows it.
    pub depth: Option<u8>,
}

/// (front, back, colour bytes, depth) for a window of WIN_DEPTH code w and
/// PIXEL_FMT p, offsets in 2-byte units; None for a pair the library does
/// not produce. The stereo formats (2, 5) and (3, 4) give their left eye.
fn window_fields(w: u8, p: u8) -> Option<(u8, u8, u8, Option<u8>)> {
    Some(match (w, p) {
        (1, 0) => (1, 0, 2, None),
        (1, 1) => (0, 0, 4, None),
        (1, 2) | (1, 3) => (1, 1, 2, None),
        (2, 0) => (0, 0, 8, None),
        (2, 1) => (2, 0, 4, None),
        (2, 2) => (2, 2, 4, Some(0)),
        (2, 3) | (2, 4) => (3, 2, 2, Some(0)),
        (2, 5) => (3, 1, 2, None),
        (3, 1) => (4, 0, 8, None),
        (3, 2) => (6, 2, 4, Some(4)),
        (3, 4) => (7, 3, 2, Some(4)),
        _ => return None,
    })
}

fn colour_codec(bytes: u8, field_fmt: u8) -> Codec {
    if bytes >= 8 { Codec::Lanes } else { Codec::Texel { bytes, field: field_fmt } }
}

impl Layout {
    pub fn of(f: Format) -> Layout {
        let w = f.win_log2.saturating_sub(1);
        if f.fb_log2 == 1 && f.pixel_fmt == PIXEL_FMT_B {
            return Layout {
                front: Field { at: 0, codec: Codec::Byte },
                back: Field { at: 1, codec: Codec::Byte },
                depth: None,
            };
        }
        let field_fmt = if f.index { FIELD_INDEX } else { f.field_fmt };
        match window_fields(w, f.pixel_fmt) {
            Some((a, b, bytes, depth)) if 1u32 << f.win_log2 <= 1u32 << f.fb_log2 => {
                let codec = colour_codec(bytes, field_fmt);
                Layout {
                    front: Field { at: 2 * a, codec },
                    back: Field { at: 2 * b, codec },
                    depth: depth.map(|d| 2 * d),
                }
            }
            // One field the size of the pixel: a texel, an image pixel.
            _ => {
                let bytes = 1u8 << f.fb_log2.min(4);
                let t = Field { at: 0, codec: colour_codec(bytes, field_fmt) };
                Layout { front: t, back: t, depth: None }
            }
        }
    }

#[cfg(test)]
    pub fn field(&self, back: bool) -> Field {
        if back { self.back } else { self.front }
    }
}

/// n bits of the 8-bit channel c: its top bits.
#[inline]
fn narrow(c: u32, n: u32) -> u32 {
    c >> (8 - n)
}

/// An n-bit value widened to 8 bits by repeating its bits.
#[inline]
fn widen(v: u32, n: u32) -> u32 {
    let mut out = 0;
    let mut have = 0;
    while have < 8 {
        out = (out << n) | v;
        have += n;
    }
    out >> (have - 8)
}

#[inline]
fn chan(word: u32, k: u32) -> u32 {
    (word >> (24 - 8 * k)) & 0xff
}

#[inline]
fn word4(c: [u32; 4]) -> u32 {
    (c[0] << 24) | (c[1] << 16) | (c[2] << 8) | c[3]
}

/// A word's bytes as 12-bit lanes: each byte at the top of its lane, its
/// top bits repeated below (an 8-bit channel at 12 bits).
#[inline]
pub(super) fn lanes_of(word: u32) -> [u16; 4] {
    std::array::from_fn(|k| {
        let c = chan(word, k as u32);
        ((c << 4) | (c >> 4)) as u16
    })
}

#[inline]
pub(super) fn word_of(lanes: [u16; 4]) -> u32 {
    word4(std::array::from_fn(|k| (lanes[k] as u32 >> 4) & 0xff))
}

#[inline]
pub(super) fn lanes_put(b: &mut [u8], l: [u16; 4]) {
    let v = ((l[0] as u64 & 0xfff) << 36) | ((l[1] as u64 & 0xfff) << 24) | ((l[2] as u64 & 0xfff) << 12) | (l[3] as u64 & 0xfff);
    b[..6].copy_from_slice(&v.to_be_bytes()[2..]);
}

#[inline]
pub(super) fn lanes_get(b: &[u8]) -> [u16; 4] {
    let mut v = [0u8; 8];
    v[2..].copy_from_slice(&b[..6]);
    let v = u64::from_be_bytes(v);
    [(v >> 36) as u16 & 0xfff, (v >> 24) as u16 & 0xfff, (v >> 12) as u16 & 0xfff, v as u16 & 0xfff]
}

/// A texel word (sampler order: luminance, intensity and alpha in the red
/// byte for one component, luminance in red and alpha in alpha for two) into
/// its `bytes` stored bytes, FIELD_FMT `field`.
pub(super) fn texel_pack(word: u32, bytes: u8, field: u8, out: &mut [u8]) {
    let c = |k| chan(word, k);
    match (bytes, field) {
        (2, FIELD_INDEX) => out[..2].copy_from_slice(&(word as u16).to_be_bytes()),
        (4, FIELD_INDEX) => out[..4].copy_from_slice(&word.to_be_bytes()),
        (2, FIELD_ONE) => out[..2].copy_from_slice(&((c(0) << 8) | c(0)).to_be_bytes()[2..]),
        (2, FIELD_TWO) => out[..2].copy_from_slice(&[c(0) as u8, c(3) as u8]),
        (2, FIELD_FOUR) => {
            let v = (0..4).fold(0, |v, k| (v << 4) | narrow(c(k), 4));
            out[..2].copy_from_slice(&(v as u16).to_be_bytes());
        }
        (2, FIELD_PACKED) => {
            let v = (narrow(c(0), 5) << 11) | (narrow(c(1), 5) << 6) | (narrow(c(2), 5) << 1) | narrow(c(3), 1);
            out[..2].copy_from_slice(&(v as u16).to_be_bytes());
        }
        (4, FIELD_TWO) => {
            let w = |v: u32| (v << 8) | v;
            out[..4].copy_from_slice(&((w(c(0)) << 16) | w(c(3))).to_be_bytes());
        }
        (4, FIELD_PACKED) => {
            let w10 = |v: u32| (v << 2) | (v >> 6);
            let v = (w10(c(0)) << 22) | (w10(c(1)) << 12) | (w10(c(2)) << 2) | narrow(c(3), 2);
            out[..4].copy_from_slice(&v.to_be_bytes());
        }
        // RGBA12, as an 8-byte window's colour: four 12-bit lanes.
        (8.., _) => lanes_put(out, lanes_of(word)),
        _ => out[..4.min(bytes as usize)].copy_from_slice(&word.to_be_bytes()[..4.min(bytes as usize)]),
    }
}

/// The inverse of texel_pack(), at the precision the bytes keep.
#[inline(always)]
pub(super) fn texel_unpack(b: &[u8], bytes: u8, field: u8) -> u32 {
    match (bytes, field) {
        (2, FIELD_INDEX) => u16::from_be_bytes([b[0], b[1]]) as u32,
        (4, FIELD_INDEX) => u32::from_be_bytes([b[0], b[1], b[2], b[3]]),
        (2, FIELD_ONE) => {
            let v = b[0] as u32;
            word4([v, v, v, v])
        }
        (2, FIELD_TWO) => word4([b[0] as u32, b[0] as u32, b[0] as u32, b[1] as u32]),
        (2, FIELD_FOUR) => {
            let v = u16::from_be_bytes([b[0], b[1]]) as u32;
            word4(std::array::from_fn(|k| widen((v >> (12 - 4 * k)) & 0xf, 4)))
        }
        (2, FIELD_PACKED) => {
            let v = u16::from_be_bytes([b[0], b[1]]) as u32;
            let f = |s: u32| widen((v >> s) & 0x1f, 5);
            word4([f(11), f(6), f(1), if v & 1 != 0 { 0xff } else { 0 }])
        }
        (4, FIELD_TWO) => word4([b[0] as u32, b[0] as u32, b[0] as u32, b[2] as u32]),
        (4, FIELD_PACKED) => {
            let v = u32::from_be_bytes([b[0], b[1], b[2], b[3]]);
            let f = |s: u32| (v >> (s + 2)) & 0xff;
            word4([f(22), f(12), f(2), widen(v & 3, 2)])
        }
        (8.., _) => word_of(lanes_get(b)),
        _ => {
            let mut w = [0u8; 4];
            let n = 4.min(bytes as usize);
            w[..n].copy_from_slice(&b[..n]);
            u32::from_be_bytes(w)
        }
    }
}

impl Field {
    /// The field's word.
    #[inline]
    pub fn load(&self, pixel: &[u8]) -> u32 {
        let b = &pixel[self.at as usize..];
        match self.codec {
            Codec::Lanes => word_of(lanes_get(b)),
            Codec::Byte => b[0] as u32,
            Codec::Texel { bytes, field } => texel_unpack(b, bytes, field),
        }
    }

    #[inline]
    pub fn store(&self, pixel: &mut [u8], word: u32) {
        let b = &mut pixel[self.at as usize..];
        match self.codec {
            Codec::Lanes => lanes_put(b, lanes_of(word)),
            Codec::Byte => b[0] = word as u8,
            Codec::Texel { bytes, field } => texel_pack(word, bytes, field, b),
        }
    }

    /// 12-bit lanes; a narrower codec keeps the lanes' top bits.
    #[cfg(test)]
    #[inline]
    pub fn load_lanes(&self, pixel: &[u8]) -> [u16; 4] {
        match self.codec {
            Codec::Lanes => lanes_get(&pixel[self.at as usize..]),
            _ => lanes_of(self.load(pixel)),
        }
    }

    #[cfg(test)]
    #[inline]
    pub fn store_lanes(&self, pixel: &mut [u8], l: [u16; 4]) {
        match self.codec {
            Codec::Lanes => lanes_put(&mut pixel[self.at as usize..], l),
            _ => self.store(pixel, word_of(l)),
        }
    }
}

#[cfg(test)]
impl Layout {
    /// The 24-bit depth word, 0 where the pixel has none.
    #[inline]
    pub fn load_depth(&self, pixel: &[u8]) -> u32 {
        self.depth.map_or(0, |at| {
            let b = &pixel[at as usize..];
            u32::from_be_bytes([0, b[0], b[1], b[2]])
        })
    }

    #[inline]
    pub fn store_depth(&self, pixel: &mut [u8], z: u32) {
        if let Some(at) = self.depth {
            pixel[at as usize..at as usize + 3].copy_from_slice(&z.to_be_bytes()[1..]);
        }
    }

    #[inline]
    pub fn load_stencil(&self, pixel: &[u8]) -> u32 {
        self.depth.map_or(0, |at| pixel[at as usize + 3] as u32)
    }

    #[inline]
    pub fn store_stencil(&self, pixel: &mut [u8], s: u32) {
        if let Some(at) = self.depth {
            pixel[at as usize + 3] = s as u8;
        }
    }
}

/// A level descriptor's format index (word 1 bits 25:20) as the bytes and
/// FIELD_FMT its uploads name (traced for every sized internal format,
/// `ignore/odsy/docs/measured.md` section 6). None for an index no traced
/// upload uses.
pub(super) fn texel_storage(index: u32) -> Option<(u8, u8)> {
    Some(match index {
        0 | 2 | 4 | 8 | 9 | 10 => (2, FIELD_ONE),
        11 => (2, FIELD_TWO),
        12 | 48 => (2, FIELD_FOUR),
        13 => (2, FIELD_PACKED),
        16 => (4, FIELD_TWO),
        17 | 49 => (4, FIELD_FOUR),
        18 => (4, FIELD_PACKED),
        32 | 33 => (8, FIELD_FOUR),
        _ => return None,
    })
}

/// The surface and field of a level: its address and row from the
/// descriptor, its storage from the format index.
pub(super) fn level_surface(word0: u32, word1: u32) -> Option<(Surface, Field)> {
    let (bytes, field) = texel_storage((word1 >> 20) & 0x3f)?;
    let s = Surface::new(word0, word1 & 0xfff, bytes.trailing_zeros() as u8);
    Some((s, Field { at: 0, codec: Codec::Texel { bytes, field } }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_sit_where_the_library_puts_them() {
        // Double-buffered 8:8:8:8 with depth, single-buffered, overlay.
        let d = Layout::of(Format::decode(0x20406d50));
        assert_eq!((d.front.at, d.back.at, d.depth), (12, 4, Some(8)));
        let s = Layout::of(Format::decode(0x6950));
        assert_eq!((s.front.at, s.back.at, s.depth), (4, 4, Some(0)));
        assert_eq!(s.front.codec, Codec::Texel { bytes: 4, field: FIELD_FOUR });
        let o = Layout::of(Format::decode(0x20080080));
        assert_eq!((o.front.codec, o.back.at), (Codec::Byte, 1));
        // A texture level: one 4-byte texel.
        let t = Layout::of(Format::decode(0x080024d0));
        assert_eq!((t.front.at, t.front.codec), (0, Codec::Texel { bytes: 4, field: FIELD_FOUR }));
    }

    #[test]
    fn lanes_keep_any_word() {
        let mut p = [0u8; 16];
        // A 12-bit double-buffered window in a 16-byte pixel, plus the depth
        // of a 16-byte window.
        let l = Layout { depth: Some(12), ..Layout::of(Format { fb_log2: 4, win_log2: 4, pixel_fmt: 1, field_fmt: FIELD_FOUR, ..Default::default() }) };
        assert_eq!((l.front.at, l.back.at, l.front.codec), (8, 0, Codec::Lanes));
        for w in [0x0000_0000, 0xffff_ffff, 0x1234_5678, 0x0000_0fff, 0x80ff_7f01] {
            l.back.store(&mut p, w);
            assert_eq!(l.back.load(&p), w);
        }
        l.store_depth(&mut p, 0xabcdef);
        l.store_stencil(&mut p, 0x5a);
        assert_eq!((l.load_depth(&p), l.load_stencil(&p), l.back.load(&p)), (0xabcdef, 0x5a, 0x80ff_7f01));
        l.front.store_lanes(&mut p, [0xfff, 0x123, 0, 0x800]);
        assert_eq!(l.front.load_lanes(&p), [0xfff, 0x123, 0, 0x800]);
        assert_eq!(l.back.load(&p), 0x80ff_7f01);
    }

    // The V10's read-backs (measured.md section 6).
    #[test]
    fn texels_keep_the_boards_precision() {
        let rt = |w, bytes, field| {
            let mut b = [0u8; 8];
            texel_pack(w, bytes, field, &mut b);
            texel_unpack(&b, bytes, field)
        };
        assert_eq!(rt(0x20a5_5a10, 2, FIELD_FOUR), 0x22aa_5511);
        assert_eq!(rt(0x2000_a5ff, 2, FIELD_PACKED), 0x2100_a5ff);
        assert_eq!(rt(0x2000_a5ff, 4, FIELD_FOUR), 0x2000_a5ff);
        assert_eq!(rt(0x2000_a5ff, 4, FIELD_PACKED), 0x2000_a5ff);
        assert_eq!(rt(0x2000_a5ff, 8, FIELD_FOUR), 0x2000_a5ff);
        assert_eq!(rt(0x7f00_0040, 2, FIELD_TWO), 0x7f7f_7f40);
        assert_eq!(rt(0x9a00_0000, 2, FIELD_ONE), 0x9a9a_9a9a);
    }

    #[test]
    fn overlay_buffer_is_id_byte_then_overlay_byte() {
        let l = Layout::of(Format::decode(0x20080080));
        let mut p = [0u8; 2];
        l.front.store(&mut p, 0x1b);
        l.back.store(&mut p, 0x42);
        assert_eq!(p, [0x1b, 0x42]);
    }

    #[test]
    fn level_descriptors_name_their_storage() {
        let (s, f) = level_surface(0x16b800, 0x00c0_0002).unwrap();
        assert_eq!((s.bytes_per_pixel(), s.pitch, f.codec), (2, 2, Codec::Texel { bytes: 2, field: FIELD_FOUR }));
        let (s, _) = level_surface(0x16b800, 0x8210_0002).unwrap();
        assert_eq!(s.bytes_per_pixel(), 8);
    }
}
