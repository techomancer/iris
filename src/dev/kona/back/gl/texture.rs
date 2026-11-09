//! The texture commands: loads into texture memory, binds, and the unpack state a
//! load's sizing reads.
//!
//! A level reaches texture memory as texel draws (`DRAW_TEXELS_INTO_TEXTURE_MEMORY`)
//! and a load finish. The load does not name its level: the header's address word
//! (word 18, or 19 when the bank bit is set) and the bank bit say where it lands.
//! The model keeps its levels by (bank, address) per context (the plan keyed images
//! by cache entry; the entry number reaches a level only through the bind's table,
//! which names the same addresses). A bind and its record list the levels' addresses
//! in order; the levels are found by that table and the first level's bank, flipped
//! for every second level. That is the contract for a numbered bind and for an
//! all-ones bind alike, which is why an image survives its entry's freeing.
//!
//! Grades (the table's): the header words' roles, the texel width by size code, the unit of
//! a level address (256 bits per bank) and the rule that places a small level in its tile are
//! established; the unit as the hardware's own is inferred, and the bank flip is inferred. The
//! texel layout in texture memory is the model's own (`store::texture`).
//!
//! A bind carries no storage code and the pipe stores none per level: a level is found by its bank and address,
//! and its words are sampled under the format the bind names (bits 20:14 of its mode word), whatever storage
//! code loaded them. Sixteen-bit words reloaded under the one-component code (the library's read-back and reload
//! of a level) are therefore kept raw until a bind samples them, and a read-back under that code returns the
//! words themselves.
//!
//! A level smaller than its tile on an axis is sent by the client as 2^(tile exponent - level
//! exponent) copies along that axis, each at its own origin, and the last copy gives the level's
//! origin (established). A level at least as large as the tile is sent once, at the origin, and a
//! later load of part of it is an update. The level's logical size is the bind's level-0 size
//! halved per level (established); the model keeps every load's box and the last load, takes the
//! last copy's origin on an axis the bind says is small, and the origin and logical size on any
//! other.
//!
//! A level takes `pw * ph` texels of the width its size code says, which in units of
//! 256 bits per bank is the extent that address names. Two levels of one bank cannot share a
//! unit, so a load at a new address retires the levels it overlaps (the hardware overwrites
//! them); a bind to a retired level finds nothing. Texture memory is addressed the way the pipe
//! addresses it (`store::texture`): a level at address `a` of a bank is the words from
//! `a * UNIT_WORDS` of that bank's plane, a load at an address that holds a level of the same
//! shape writes it in place, and nothing is placed, so a load is refused only when its units
//! reach past the bank or its texels would take the pool past the product's texture memory
//! (the planes are sparse pages, and the pages held across all of them are capped). A context whose slot is activated afresh gives its planes back
//! (`TexContext::release`). Established: the pool is one per pipe, loads write in place at
//! addresses relative to the owner's block, and a level reloaded at the same place overwrites
//! it; the entry-to-address tables are per context. Approximated: that the overlap is by the
//! unit arithmetic above in each bank separately, that the planes are a context's own (the
//! owner's block offset is not modelled, so contexts neither contend for memory nor alias, so a texture loaded through one member of a share group (contexts the kernel lets see the same physical block) is not visible to a bind from another member),
//! and that a context's texture memory is freed when its slot is activated afresh (the kernel's
//! block release is not read). The clip-map, bicubic, the old-style bind and the load set-up have no
//! sender in the recordings (or are out of scope) and are counted, not modelled.
//!
//! A volume (target 0x806f) is one command per slice, the slice number in header word 15 and the
//! level's physical depth in the r exponent of word 16 (both established). Every slice of a level
//! carries the same pair of address words, so a level is still found by (bank, address) and its
//! slices are stacked in the arena in the model's own layout; where the hardware puts the texels
//! inside the pair is unknown. A volume's bind is the short bind and a record of the 3D forms
//! (tokens 0x17b and 0x17c, framing established). A level's address in the record is looked for
//! in the bank its parity names and then in the other (inferred: the table says the bank bit of
//! a mip-mapped volume's level is its parity and that of a single-level volume is "bank state").

use super::super::raster::texel::{default_weights, spline_samples, Combine, Detail, EnvMode, Filter, Sharpen, TexDesc, TexLevel, Wrap, MAX_LEVELS, CURVE_NODES};
use super::super::store::texture::{TextureStore, PLANES_PER_CONTEXT, UNIT_WORDS};
use crate::dev::kona::stream::Stream;
use crate::dev::kona::defs::*;
use std::collections::HashMap;

/// The unpack state a load's sizing and decoding read (the pixel-store parameters).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Unpack {
    pub row_length: u32,
    pub skip_rows: u32,
    pub skip_pixels: u32,
    pub alignment: u32,
    pub swap_bytes: bool,
}

impl Default for Unpack {
    fn default() -> Self {
        Unpack { row_length: 0, skip_rows: 0, skip_pixels: 0, alignment: 4, swap_bytes: false }
    }
}

impl Unpack {
    /// A pixel-store parameter, GL's names. An alignment that is not 1, 2, 4 or 8 is
    /// refused (the board keeps only a power of two up to 8). The pack side's names
    /// are kept by the board and read by no load, so they are ignored here.
    pub fn set(&mut self, name: u32, value: u32) {
        match name {
            0xcf0 => self.swap_bytes = value != 0,
            0xcf2 => self.row_length = value,
            0xcf3 => self.skip_rows = value,
            0xcf4 => self.skip_pixels = value,
            0xcf5 if matches!(value, 1 | 2 | 4 | 8) => self.alignment = value,
            _ => {}
        }
    }
}

/// A texel draw's header, the twenty words by their roles.
#[derive(Clone, Copy, Debug)]
pub struct Header {
    pub target: u32,
    pub w: u32,
    pub h: u32,
    pub format: u32,
    pub ty: u32,
    pub offset: u32,
    pub dma: bool,
    pub strided: bool,
    pub storage: u32,
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
    pub phys: u32,
    pub misc: u32,
    pub address: u32,
    /// The destination slice (word 15), for a volume.
    pub slice: u32,
}

/// The extent of a level in units of 256 bits: its rectangle's texels at the width the size
/// code says, in 16-bit lanes, over sixteen lanes a unit.
fn units(pw: u32, ph: u32, size_code: u32) -> u64 {
    ((pw as u64 * ph as u64) << size_code).div_ceil(texture::ADDRESS_UNIT_LANES as u64)
}

impl Header {
    pub fn parse(a: &[u32]) -> Option<Header> {
        let a: &[u32; 20] = a.try_into().ok()?;
        let bank = texture::BANK.get(a[17]);
        Some(Header { target: a[0], w: a[1], h: a[2], format: a[3], ty: a[4], offset: a[5] & 3, dma: a[6] != 0, strided: a[7] != 0, storage: a[9], x0: a[11] & 0xffff, y0: a[12] & 0xffff, x1: a[13] & 0xffff, y1: a[14] & 0xffff, phys: a[16], misc: a[17], address: a[18 + bank as usize], slice: a[15] & 0x7ff })
    }

    pub fn bank(&self) -> u8 {
        texture::BANK.get(self.misc) as u8
    }

    /// Whether the load is of a detail image (target 0x8095), which the model keeps apart from the base levels.
    pub fn is_detail(&self) -> bool {
        self.target == texture::DETAIL_TARGET
    }

    /// Whether the load is of a volume (target 0x806f).
    pub fn is_volume(&self) -> bool {
        self.target == texture::VOLUME_TARGET
    }

    /// The level's physical depth in slices: the r exponent of word 16 for a volume (established), one otherwise.
    pub fn depth(&self) -> u32 {
        if self.is_volume() { 1 << ((self.phys >> 8 & 0xf).min(11)) } else { 1 }
    }

    /// The log2 of a texel's width in 16-bit lanes.
    pub fn size_code(&self) -> u32 {
        texture::SIZE_CODE.get(self.misc)
    }

    /// Components of the source format, or None for one the model does not decode
    /// (CMYK, colour index, stencil, depth).
    fn components(&self) -> Option<u32> {
        Some(match self.format {
            0..=3 | 6 => 1,
            4 => 3,
            5 | 8 => 4,
            7 => 2,
            _ => return None,
        })
    }

    /// Bytes of a component, for the plain types (signed and unsigned byte, short, int, float).
    fn type_bytes(&self) -> Option<u32> {
        Some(match self.ty {
            0 | 1 => 1,
            2 | 3 => 2,
            4..=6 => 4,
            _ => return None,
        })
    }

    /// Bytes of a pixel of a packed type: 3-3-2 (code 8), 4-4-4-4 (9), 5-5-5-1 (10), 8-8-8-8 (11) and 10-10-10-2 (12),
    /// the sizes of OpenGL's packed types of those names (established: the library sends the codes, and the pipe
    /// takes their own size as the pixel's). The library requires RGB with 3-3-2 and RGBA with the others, and
    /// refuses any other format (an error in the client), so the model does not decode one.
    fn packed_bytes(&self) -> Option<u32> {
        match (self.ty, self.format) {
            (8, 4) => Some(1),
            (9 | 10, 5) => Some(2),
            (11 | 12, 5) => Some(4),
            _ => None,
        }
    }

    fn pixel_bytes(&self) -> Option<u32> {
        match self.ty {
            8..=12 => self.packed_bytes(),
            _ => Some(self.components()? * self.type_bytes()?),
        }
    }

    /// The row stride and the first byte, from the unpack state.
    fn geometry(&self, u: &Unpack) -> Option<(u64, u64, u64)> {
        let p = self.pixel_bytes()? as u64;
        let l = if u.row_length == 0 { self.w } else { u.row_length } as u64;
        let a = u.alignment.max(1) as u64;
        // The unpack state is the guest's: a product past what fits is no size (counted as an unsupported load).
        let stride = (l * p).div_ceil(a).checked_mul(a)?;
        let start = (u.skip_rows as u64).checked_mul(stride)?.checked_add(self.offset as u64)?.checked_add(u.skip_pixels as u64 * p)?;
        Some((p, stride, start))
    }

    /// Words of image that follow the header in the command stream, from the header
    /// and the unpack state (established); None for a format or type the model does
    /// not decode, whose size is not computed.
    pub fn image_words(&self, u: &Unpack) -> Option<u32> {
        let (p, stride, start) = self.geometry(u)?;
        let (w, h) = (self.w as u64, self.h as u64);
        if w == 0 || h == 0 {
            return Some(0);
        }
        let n = if self.strided {
            // A row's first byte sits at `(start + r * stride) % 4`, which repeats every four rows.
            let row = |r: u64| (start.wrapping_add(r.wrapping_mul(stride)) % 4 + w * p).div_ceil(4);
            let (cycles, rest) = (h / 4, h % 4);
            cycles.checked_mul((0..4).map(row).sum::<u64>())?.checked_add((0..rest).map(row).sum::<u64>())?
        } else {
            // A row length below the width puts the last row's end before its start: no size (counted as an unsupported load).
            let total = stride.checked_mul(h)?.checked_sub(stride.checked_sub(w * p)?)?;
            start.checked_add(total)?.div_ceil(4) - start / 4
        };
        u32::try_from(n).ok()
    }
}

/// A texel read-back's header (token 0x052), eighteen words (the table, section 7.16): the target,
/// the size, the format, type and storage codes, a spare, the first and last texel as x, y, z, and
/// five words copied from the texture's load record: a word of the texture mode, whose bit 28 is the
/// level's bank, the size exponents as a load header's word 16 has them, a format-and-layout byte,
/// and the level's address (word 16, with word 17 zero for an ordinary 2D level). Established; the
/// level is looked up by its bank and address.
#[derive(Clone, Copy, Debug)]
pub struct ReadHeader {
    pub w: u32,
    pub h: u32,
    pub format: u32,
    pub ty: u32,
    pub storage: u32,
    pub first: [u32; 3],
    pub mode: u32,
    pub phys: u32,
    pub address: u32,
}

impl ReadHeader {
    pub fn parse(a: &[u32]) -> Option<ReadHeader> {
        let a: &[u32; 18] = a.try_into().ok()?;
        Some(ReadHeader { w: a[1], h: a[2], format: a[3], ty: a[4], storage: a[5], first: [a[7], a[8], a[9]], mode: a[13], phys: a[14], address: a[16] })
    }

    /// The level's bank: bit 28 of the texture-mode word.
    pub fn bank(&self) -> u8 {
        (self.mode >> 28 & 1) as u8
    }
}

/// The most texels a read-back is answered for. A read asks for what the guest names, and the answer is built
/// as one colour (sixteen bytes) a texel before it is packed, so this bounds that vector. It is a separate cap
/// from the byte one: a one-byte texel makes a 2048 by 1024 read only 2 MiB packed but a 32 MiB colour vector.
pub const MAX_READ_TEXELS: u64 = 1 << 20;
/// The most bytes a read-back may pack, for a read in a bracket (the bracket's buffer holds 8192 by 8192 bytes at
/// most). The packed size includes the pack record's skips, which are the guest's.
pub const MAX_READ_BYTES: u64 = super::pixels::MAX_PACK_BYTES;

/// A texel in the model's layout from component values 0..=255 of a format: missing
/// alpha is one and missing colour zero (alpha only), luminance is the colour.
fn texel(format: u32, c: [u32; 4]) -> u32 {
    let (r, g, b, a) = match format {
        0 => (c[0], 0, 0, 255),
        1 => (0, c[0], 0, 255),
        2 => (0, 0, c[0], 255),
        3 => (0, 0, 0, c[0]),
        4 => (c[0], c[1], c[2], 255),
        5 => (c[0], c[1], c[2], c[3]),
        6 => (c[0], c[0], c[0], 255),
        7 => (c[0], c[0], c[0], c[1]),
        _ => (c[3], c[2], c[1], c[0]),
    };
    a << 24 | r << 16 | g << 8 | b
}

/// The red, green, blue and alpha of a packed pixel of type `ty` (8 to 12, as `Header::packed_bytes`), each as 0..=255, from
/// its bytes: big-endian like the rest of the stream unless the unpack state swaps bytes. The first component is
/// in the most significant bits, as in OpenGL's packed types; a field narrower than 8 bits is scaled to 255 (the
/// pipe's unpacker turns each into an unsigned byte, established, but its scaling is not read: OpenGL's, inferred).
fn packed(ty: u32, bytes: &[u8], swap: bool) -> [u32; 4] {
    let n = bytes.len();
    let raw = bytes.iter().enumerate().fold(0u32, |a, (i, &x)| a | (x as u32) << if swap { 8 * i } else { 8 * (n - 1 - i) });
    let field = |shift: u32, bits: u32| (raw >> shift & ((1 << bits) - 1)) * 255 / ((1 << bits) - 1);
    match ty {
        8 => [field(5, 3), field(2, 3), field(0, 2), 255],
        9 => [field(12, 4), field(8, 4), field(4, 4), field(0, 4)],
        10 => [field(11, 5), field(6, 5), field(1, 5), field(0, 1)],
        11 => [field(24, 8), field(16, 8), field(8, 8), field(0, 8)],
        _ => [field(22, 10), field(12, 10), field(2, 10), field(0, 2)],
    }
}

/// A component as 0..=255 from its bytes: the stream is big-endian (the client's order)
/// unless the unpack state swaps bytes. Signed types keep the non-negative half.
fn component(ty: u32, bytes: &[u8], swap: bool) -> u32 {
    let n = bytes.len();
    let mut raw = 0u32;
    for (i, &x) in bytes.iter().enumerate() {
        raw |= (x as u32) << if swap { 8 * i } else { 8 * (n - 1 - i) };
    }
    match ty {
        0 => (raw as u8 as i8).max(0) as u32 * 255 / 127,
        1 => raw,
        2 => ((raw as u16 as i16).max(0) as u32 * 255 + 16383) / 32767,
        3 => (raw * 255 + 32767) / 65535,
        4 => ((raw as i32).max(0) as u64 * 255 / 0x7fff_ffff) as u32,
        5 => (raw as u64 * 255 / 0xffff_ffff) as u32,
        _ => (f32::from_bits(raw).clamp(0.0, 1.0) * 255.0 + 0.5) as u32,
    }
}

/// Reduce a texel to the width a storage code holds: 0x14 is 5-5-5-1, 0x10 is 4-4-4-4, 0x18 is
/// R5-G6-B5 (no alpha), and anything else keeps its 8 bits. The table gives 0x20 as 12-bit luminance-alpha
/// and 0x40 as 12-bit RGB, which the model's 8 bits a component holds without loss, so they keep 8 (the
/// model's own layout, `store::texture`). The codes the recordings use are 0x30, 0x14, 0x10 and 0x28.
fn reduce(storage: u32, t: u32) -> u32 {
    let (r, g, b, a) = ((t >> 16 & 0xff) as u32, (t >> 8 & 0xff) as u32, (t & 0xff) as u32, (t >> 24) as u32);
    let expand = |v: u32, bits: u32| {
        let max = (1u32 << bits) - 1;
        let q = (v * max + 127) / 255;
        q * 255 / max
    };
    match storage {
        0x14 => expand(a, 1) << 24 | expand(r, 5) << 16 | expand(g, 5) << 8 | expand(b, 5),
        0x10 => expand(a, 4) << 24 | expand(r, 4) << 16 | expand(g, 4) << 8 | expand(b, 4),
        0x18 => 0xff << 24 | expand(r, 5) << 16 | expand(g, 6) << 8 | expand(b, 5),
        _ => t,
    }
}

/// The storage code of a read that takes texture memory as one 16-bit component a texel: luminance (the table's
/// 0x0d), which the library reads back with format 6 and type 3.
const STORAGE_ONE_COMPONENT: u32 = 0x0d;

/// A texel in the model's layout from the 16-bit word a storage code with 16-bit lanes holds, as the pipe samples
/// the word under the format the bind names: 0x14 is 5-5-5-1, 0x10 4-4-4-4 and 0x18 R5-G6-B5, and any other code
/// is one component, taken as luminance. The lane layouts are the table's bus layouts (established); sampling
/// whatever words are there under the bind's format is the table's bind rule.
fn unpack16(storage: u32, w: u32) -> u32 {
    let expand = |v: u32, bits: u32| v * 255 / ((1 << bits) - 1);
    match storage {
        0x14 => expand(w & 1, 1) << 24 | expand(w >> 11 & 31, 5) << 16 | expand(w >> 6 & 31, 5) << 8 | expand(w >> 1 & 31, 5),
        0x10 => expand(w & 15, 4) << 24 | expand(w >> 12 & 15, 4) << 16 | expand(w >> 8 & 15, 4) << 8 | expand(w >> 4 & 15, 4),
        0x18 => 0xff << 24 | expand(w >> 11 & 31, 5) << 16 | expand(w >> 5 & 63, 6) << 8 | expand(w & 31, 5),
        _ => {
            let l = (w * 255 + 32767) / 65535;
            0xff << 24 | l << 16 | l << 8 | l
        }
    }
}

/// A 16-bit word as the float the packer turns back into the same word as an unsigned short.
fn word_as_float(w: u32) -> f32 {
    w as f32 / 65535.0
}

/// The 16-bit word a texel loaded under `storage` holds, the inverse of `unpack16` on the texels `reduce` makes:
/// None for a code whose lanes are not 16-bit packed colour (the table gives 0x10 and 0x14 as read back raw; 0x18 is
/// the same lane layout and is inferred to behave the same).
fn pack16(storage: u32, t: u32) -> Option<u32> {
    let q = |v: u32, bits: u32| (v * ((1 << bits) - 1) + 127) / 255;
    let (a, r, g, b) = (t >> 24, t >> 16 & 0xff, t >> 8 & 0xff, t & 0xff);
    Some(match storage {
        0x14 => q(r, 5) << 11 | q(g, 5) << 6 | q(b, 5) << 1 | q(a, 1),
        0x10 => q(r, 4) << 12 | q(g, 4) << 8 | q(b, 4) << 4 | q(a, 4),
        0x18 => q(r, 5) << 11 | q(g, 6) << 5 | q(b, 5),
        _ => return None,
    })
}

/// Sample the raw 16-bit words of a level's loaded box under `format`, in place.
fn decode_level(store: &TextureStore, l: &Level, format: u32) {
    for z in 0..l.pd {
        for y in l.min.1..=l.max.1.min(l.ph - 1) {
            for x in l.min.0..=l.max.0.min(l.pw - 1) {
                let at = l.off + z * l.pw * l.ph + y * l.pw + x;
                store.set(l.plane, at, unpack16(format, store.get(l.plane, at) & 0xffff));
            }
        }
    }
}

/// A level in texture memory: the physical rectangle (`pw` by `ph`) it was
/// loaded as and the box of texels loaded into it. It starts at word `off` (its address times
/// `UNIT_WORDS`) of plane `plane`.
#[derive(Clone, Copy, Debug)]
struct Level {
    plane: u16,
    off: u32,
    pw: u32,
    ph: u32,
    /// The physical depth in slices: one for a 2D level.
    pd: u32,
    /// The texel size code the level was loaded with.
    size_code: u32,
    /// The box every load so far covered.
    min: (u32, u32),
    max: (u32, u32),
    /// The most recent load, per axis: where it starts and how many texels it spans.
    last: [(u32, u32); 2],
    /// Per axis, whether the first load covered the whole rectangle. A level that fills the tile on an axis
    /// is sent once, whole; a level smaller than the tile is sent as copies, each narrower than the rectangle.
    /// So a level whose first load was whole is never a run of copies on that axis.
    whole: [bool; 2],
    /// The storage code the level's words are in. A bind never matches on it (the pipe stores none per level and
    /// samples the words under the bind's format); it says only how to read the words back as they were loaded.
    storage: u32,
    /// The words are the raw 16-bit lanes a reload under the one-component code sent, not yet sampled under a
    /// bind's format: a bind turns them into texels, and a read-back of the level returns them unchanged.
    raw: bool,
}

impl Level {
    /// The level's origin and size in its rectangle, per axis. A level smaller than the tile on an axis was
    /// sent as copies and the last copy gives its origin; the bind's logical size says which axes those are
    /// (`logical`). Without one (a read-back names no bind) an axis is a small level's when its first load
    /// was not the whole rectangle and the last load is narrower than it, ends at its edge and sits on a
    /// multiple of its own width, which is where the last of a run of copies lands. That cannot be told from
    /// an update of a small level that ends at the edge, since the header does not say; an update of a level
    /// that was first loaded whole is told, and left alone. On any other axis the level starts at the
    /// origin and is the box loaded to (or the logical size, when the bind gives one).
    fn place(&self, logical: [Option<u32>; 2]) -> [(u32, u32); 2] {
        std::array::from_fn(|a| {
            let p = [self.pw, self.ph][a];
            let (at, n) = self.last[a];
            let (lo, hi) = [(self.min.0, self.max.0), (self.min.1, self.max.1)][a];
            match logical[a] {
                Some(l) if l < p => (at, l.min(p - at)),
                Some(l) => (0, l.min(p)),
                None if !self.whole[a] && n > 0 && n < p && at + n == p && at % n == 0 => (at, n),
                None => (lo, hi.saturating_sub(lo) + 1),
            }
        })
    }
}

/// What the texture path counted, for `kona status`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TexCounts {
    pub loads: u64,
    pub dma_loads: u64,
    pub finishes: u64,
    pub binds: u64,
    pub bind_refused: u64,
    pub unsupported_load: u64,
    /// Loads refused because the level's address and extent lie outside a bank, or because its texels would take
    /// texture memory past the product's pool.
    pub full: u64,
    /// Tokens the table says no software sends, 3D and detail binds, and the like.
    pub unmodelled: u64,
    /// Level loads that were detail images (also counted in `loads`).
    pub detail_loads: u64,
    /// Level loads that were slices of a volume (also counted in `loads`), and binds of a volume.
    pub volume_loads: u64,
    pub volume_binds: u64,
    /// Filter-coefficient tables stored (token 0x17e) and detail binds seen (token 0x17d).
    pub filter_tables: u64,
    pub detail_binds: u64,
    /// Mode-1 base binds, by what the detail interaction did: the format row differed from the detail bind's (or
    /// there was none), so the magnification was forced to linear; or it matched, and the detail image the bind's
    /// address pair names was found (resolved) or not (unresolved).
    pub detail_mismatched: u64,
    pub detail_resolved: u64,
    pub detail_unresolved: u64,
    /// Texel read-backs (token 0x052), by route and by how they were answered.
    pub reads: u64,
    pub reads_bracketed: u64,
    /// Answered with zeros: the format or type is not one the packer knows, or the level is not loaded.
    pub read_unsupported: u64,
    pub read_missing: u64,
    /// Too big to answer: over `MAX_READ_TEXELS`, or, bare, more than the return page holds. The
    /// completion is posted with no image, or the bracket is refused.
    pub read_oversize: u64,
    /// With no context or return page to answer on.
    pub read_unposted: u64,
}

/// A map of levels, in key order so the same state is the same bytes.
fn visit_levels<S: Stream>(s: &mut S, levels: &mut HashMap<(u8, u32), Level>) {
    let mut keys: Vec<(u8, u32)> = levels.keys().copied().collect();
    keys.sort_unstable();
    let n = s.len(keys.len());
    let mut out = HashMap::new();
    for i in 0..n {
        if !s.ok() {
            break;
        }
        let (mut bank, mut addr) = keys.get(i).copied().unwrap_or((0, 0));
        let mut l = levels.get(&(bank, addr)).copied().unwrap_or(Level { plane: 0, off: 0, pw: 0, ph: 0, pd: 1, size_code: 0, min: (0, 0), max: (0, 0), last: [(0, 0); 2], whole: [false; 2], storage: 0, raw: false });
        s.u8(&mut bank);
        s.u32(&mut addr);
        let mut plane = l.plane as u32;
        s.u32(&mut plane);
        l.plane = plane as u16;
        let [mut lx, mut ly] = l.last;
        for v in [&mut l.off, &mut l.pw, &mut l.ph, &mut l.pd, &mut l.size_code, &mut l.min.0, &mut l.min.1, &mut l.max.0, &mut l.max.1, &mut lx.0, &mut lx.1, &mut ly.0, &mut ly.1] {
            s.u32(v);
        }
        l.last = [lx, ly];
        s.bool(&mut l.whole[0]);
        s.bool(&mut l.whole[1]);
        s.u32(&mut l.storage);
        s.bool(&mut l.raw);
        out.insert((bank, addr), l);
    }
    if s.loading() {
        *levels = out;
    }
}

/// The most filter-coefficient tables a context keeps (the slots are allocated by the client from zero; a stream
/// that sends more is the guest's, and the extra are dropped).
const MAX_FILTER_TABLES: usize = 512;

/// The detail bind's (token 0x17d) words the pipe keeps: the entry number (or all ones), the s and t size
/// exponents (written to the detail-size register), the detail image's format row, and the address pair. The detail
/// level, add or modulate mode, channel select and coefficient slot are not here: they come from the base bind
/// (established).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DetailBind {
    pub entry: u32,
    pub exps: [u32; 2],
    pub row: u32,
    pub pair: [u32; 2],
}

/// One context's texture state.
#[derive(Default)]
pub struct TexContext {
    /// The context slot (the no-context state is `CONTEXT_SLOTS`): which planes of texture memory are the
    /// context's own.
    slot: usize,
    pub unpack: Unpack,
    levels: HashMap<(u8, u32), Level>,
    /// The detail images (target 0x8095), by the bank and address they were loaded to. A detail image
    /// is one level, sent once to each bank, and is taken to have an address space of its own (inferred: the table
    /// gives its addresses and not that they are apart from the base's): it must not retire or overwrite a base
    /// level at the same address. A detail image's arena space is taken and given back as a base level's is.
    detail: HashMap<(u8, u32), Level>,
    /// The filter-coefficient tables by slot (token 0x17e): the 33 words the firmware stores, kept as sent. The
    /// command carries no type, so a table is decoded only by the bind that names its slot.
    filters: Vec<(u32, [u32; 33])>,
    /// The last detail bind (token 0x17d). None until one is sent: the firmware's format row starts at -1, which
    /// no base bind matches.
    detail_bind: Option<DetailBind>,
    /// A texel draw waiting for its image by DMA.
    pub pending: Option<[u32; 20]>,
    /// The short bind that waits for its record.
    bind: Option<[u32; 5]>,
}

impl TexContext {
    /// The texture state of context slot `slot`.
    pub fn new(slot: usize) -> Self {
        TexContext { slot, ..Default::default() }
    }

    /// The plane of texture memory that holds the base levels, or the detail images, of `bank`.
    fn plane_of(&self, detail: bool, bank: u8) -> u16 {
        (self.slot.min(CONTEXT_SLOTS) * PLANES_PER_CONTEXT + detail as usize * 2 + (bank & 1) as usize) as u16
    }

    /// Whether the context holds any level.
    pub fn holds(&self) -> bool {
        !self.levels.is_empty() || !self.detail.is_empty()
    }

    /// Give the context's texture memory back and forget the levels, as when its slot is activated
    /// afresh. Call only with the tiles drained.
    pub fn release(&mut self, store: &TextureStore) {
        for p in 0..PLANES_PER_CONTEXT {
            store.drop_plane((self.slot.min(CONTEXT_SLOTS) * PLANES_PER_CONTEXT + p) as u16);
        }
        self.levels.clear();
        self.detail.clear();
    }

    /// The unpack state, the levels loaded and the bind waiting for its record, as the stream
    /// module reads and writes them. Levels go in key order so the same state is the same
    /// bytes.
    pub fn visit<S: Stream>(&mut self, s: &mut S) {
        // Every field is named, so a new one fails to compile until it is saved here or left out on purpose.
        let Self { slot: _, unpack: _, levels: _, detail: _, filters: _, detail_bind: _, pending: _, bind: _ } = self;
        for v in [&mut self.unpack.row_length, &mut self.unpack.skip_rows, &mut self.unpack.skip_pixels, &mut self.unpack.alignment] {
            s.u32(v);
        }
        s.bool(&mut self.unpack.swap_bytes);
        visit_levels(s, &mut self.levels);
        visit_levels(s, &mut self.detail);
        let n = s.len(self.filters.len());
        self.filters.resize(n.min(MAX_FILTER_TABLES), (0, [0; 33]));
        for (slot, words) in self.filters.iter_mut() {
            s.u32(slot);
            s.u32s(words);
        }
        let mut have = self.detail_bind.is_some();
        s.bool(&mut have);
        let mut d = self.detail_bind.unwrap_or_default();
        let DetailBind { entry, exps: [sx, tx], row, pair: [p0, p1] } = &mut d;
        for v in [entry, sx, tx, row, p0, p1] {
            s.u32(v);
        }
        if s.loading() {
            self.detail_bind = have.then_some(d);
        }
        let mut p = self.pending.is_some();
        s.bool(&mut p);
        let mut a = self.pending.unwrap_or([0; 20]);
        s.u32s(&mut a);
        if s.loading() {
            self.pending = p.then_some(a);
        }
        let mut b = self.bind.is_some();
        s.bool(&mut b);
        let mut w = self.bind.unwrap_or([0; 5]);
        s.u32s(&mut w);
        if s.loading() {
            self.bind = b.then_some(w);
        }
    }

    /// Decode a texel draw's image (`bytes`, the stream as bytes after the first
    /// `offset` of its first word are skipped) into texture memory. False when the
    /// format or type is not decoded or the level's address lies outside the bank.
    pub fn load(&mut self, store: &TextureStore, a: &[u32], bytes: &[u8], dma: bool, counts: &mut TexCounts) -> bool {
        let Some(h) = Header::parse(a) else { return false };
        let Some((p, stride, start)) = h.geometry(&self.unpack) else {
            counts.unsupported_load += 1;
            return false;
        };
        let (pw, ph, pd) = (1u32 << (h.phys & 0xf).min(12), 1u32 << (h.phys >> 4 & 0xf).min(12), h.depth());
        let slice = if h.is_volume() { h.slice } else { 0 };
        if h.x1 < h.x0 || h.y1 < h.y0 || h.x1 >= pw || h.y1 >= ph || h.w == 0 || h.h == 0 || slice >= pd {
            counts.unsupported_load += 1;
            return false;
        }
        let key = (h.bank(), h.address);
        let plane = self.plane_of(h.is_detail(), h.bank());
        // The load's own texels are the memory it asks for; one that would take texture memory past the product's
        // pool is refused before anything is retired or written (the model's own limit, as the bank's edge is).
        let (rw, rh) = ((h.x1 - h.x0 + 1).min(h.w), (h.y1 - h.y0 + 1).min(h.h));
        let first = h.address as u64 * UNIT_WORDS as u64 + (slice * pw * ph) as u64 + h.y0 as u64 * pw as u64 + h.x0 as u64;
        if !store.fits(plane, first, rw as u64, pw as u64, rh as u64) {
            counts.full += 1;
            return false;
        }
        // A detail image is a level of its own kind: one level, sent to each bank in turn, at addresses that
        // are not the base levels'.
        let levels = if h.is_detail() { &mut self.detail } else { &mut self.levels };
        let mut lv = match levels.get(&key).filter(|l| l.pw == pw && l.ph == ph && l.pd == pd && l.size_code == h.size_code()) {
            Some(l) => *l,
            None => {
                // The level takes the units from its address on: whatever else of the bank sat there is gone,
                // and so is a level of another shape at this very address. A level is refused only when
                // its units reach past the bank, and then nothing is retired.
                let (lo, hi) = (h.address as u64, h.address as u64 + units(pw, ph * pd, h.size_code()));
                if !store.contains(h.address, hi - lo) {
                    counts.full += 1;
                    return false;
                }
                levels.retain(|&(bank, at), l| !(bank == h.bank() && (at == h.address || (at as u64) < hi && at as u64 + units(l.pw, l.ph * l.pd, l.size_code) > lo)));
                // The words are the address's, zeroed: the model's layout puts the level's texels from the
                // start of its first unit, and a level whose units overlap no other's shares no word with it.
                let off = h.address * UNIT_WORDS;
                store.clear(plane, off, pw as usize * ph as usize * pd as usize);
                Level { plane, off, pw, ph, pd, size_code: h.size_code(), min: (u32::MAX, u32::MAX), max: (0, 0), last: [(0, 0); 2], whole: [false; 2], storage: h.storage, raw: false }
            }
        };
        // A reload of 16-bit words under the one-component code keeps them as they are, to be sampled under whatever
        // format the next bind names. A load that covers everything the level holds replaces it, raw or not; a smaller
        // one cannot mix the two kinds, so it is decoded as one-component texels and a raw level is decoded first.
        let is_raw = h.format == pixel::FORMAT_LUMINANCE && h.ty == pixel::TYPE_UNSIGNED_SHORT && h.storage == STORAGE_ONE_COMPONENT;
        let covers = h.x0 <= lv.min.0 && h.y0 <= lv.min.1 && h.x0 + rw - 1 >= lv.max.0 && h.y0 + rh - 1 >= lv.max.1;
        if !covers && lv.raw && !is_raw {
            decode_level(store, &lv, STORAGE_ONE_COMPONENT);
        }
        let raw = if covers { is_raw } else { lv.raw && is_raw };
        lv.raw = raw;
        lv.storage = h.storage;
        let mut words_before = 0u64;
        for row in 0..rh as u64 {
            // Where row `row` starts in the bytes: a DMA image is its rows, `w * p` bytes
            // each; an inline one is one byte stream (row mode 0) or rows that each start on
            // a word and take the words spanning their bytes (row mode 1).
            let address = start + row * stride;
            let at = if dma {
                (row * h.w as u64 * p) as usize
            } else if h.strided {
                let base = (words_before * 4 + address % 4) as usize;
                words_before += (address % 4 + h.w as u64 * p).div_ceil(4);
                base
            } else {
                (start % 4 + row * stride) as usize
            };
            for col in 0..rw as u64 {
                let from = at + (col * p) as usize;
                let Some(px) = bytes.get(from..from + p as usize) else { continue };
                let n = p as usize / h.components().unwrap_or(1) as usize;
                let c: [u32; 4] = if (8..=12).contains(&h.ty) { packed(h.ty, px, self.unpack.swap_bytes) } else { std::array::from_fn(|k| px.get(k * n..(k + 1) * n).map_or(0, |b| component(h.ty, b, self.unpack.swap_bytes))) };
                let t = if raw {
                    let w = [px[0], px[1]];
                    (if self.unpack.swap_bytes { u16::from_le_bytes(w) } else { u16::from_be_bytes(w) }) as u32
                } else {
                    reduce(h.storage, texel(h.format, c))
                };
                store.set(lv.plane, lv.off + slice * pw * ph + (h.y0 + row as u32) * pw + h.x0 + col as u32, t);
            }
        }
        // The box grows to take the load; the load is also kept as the last, since the last copy of a small level
        // gives its origin (`Level::place`) and a partial update must not shrink the level.
        let first = lv.min.0 == u32::MAX;
        let whole = if first { [h.x0 == 0 && rw == pw, h.y0 == 0 && rh == ph] } else { lv.whole };
        let lv = Level { min: (lv.min.0.min(h.x0), lv.min.1.min(h.y0)), max: (lv.max.0.max(h.x0 + rw - 1), lv.max.1.max(h.y0 + rh - 1)), last: [(h.x0, rw), (h.y0, rh)], whole, ..lv };
        levels.insert(key, lv);
        counts.loads += 1;
        counts.detail_loads += h.is_detail() as u64;
        counts.volume_loads += h.is_volume() as u64;
        true
    }

    /// The texels of a level the read-back names, `h.w` by `h.h` from its first texel, as colour: the
    /// level in the header's bank (bit 28 of its mode word) at its address (word 16) whose rectangle matches its
    /// exponents. The empty slot of a level in the other bank is address zero, which can name a real level,
    /// so no other (bank, address) pair is tried. None when no such level is loaded. A texel past the level's own
    /// size reads as zero. The caller bounds `w * h`.
    pub fn read_level(&self, store: &TextureStore, h: &ReadHeader) -> Option<Vec<[f32; 4]>> {
        let (pw, ph) = (1u32 << (h.phys & 0xf).min(12), 1u32 << (h.phys >> 4 & 0xf).min(12));
        let l = self.levels.get(&(h.bank(), h.address)).filter(|l| l.pw == pw && l.ph == ph)?;
        let [(ox, lw), (oy, lh)] = l.place([None, None]);
        // A read of a volume's level names the slice by the first texel's third coordinate; a 2D level has one.
        let z = if l.pd > 1 { h.first[2] } else { 0 };
        if z >= l.pd {
            return Some(vec![[0.0; 4]; h.w as usize * h.h as usize]);
        }
        // A read of luminance in unsigned shorts under the one-component code takes texture memory as 16-bit
        // words, whatever loaded them (the table: the storage code describes the source texels), and returns the
        // words themselves: a level held as raw words as they are, and one loaded with 16-bit lanes (0x10, 0x14)
        // as the packed word it would hold. The packer returns a word carried as `word_as_float` exactly.
        let words = h.format == pixel::FORMAT_LUMINANCE && h.ty == pixel::TYPE_UNSIGNED_SHORT && h.storage == STORAGE_ONE_COMPONENT;
        Some(
            (0..h.h)
                .flat_map(|j| (0..h.w).map(move |i| (i, j)))
                .map(|(i, j)| {
                    let (x, y) = (h.first[0].saturating_add(i), h.first[1].saturating_add(j));
                    if x >= lw || y >= lh {
                        return [0.0; 4];
                    }
                    let t = store.get(l.plane, l.off + z * l.pw * l.ph + (oy + y) * l.pw + ox + x);
                    if words {
                        if let Some(w) = if l.raw { Some(t & 0xffff) } else { pack16(l.storage, t) } {
                            return [word_as_float(w), 0.0, 0.0, 1.0];
                        }
                    }
                    let t = if l.raw { unpack16(STORAGE_ONE_COMPONENT, t & 0xffff) } else { t };
                    [t >> 16 & 0xff, t >> 8 & 0xff, t & 0xff, t >> 24].map(|c| c as f32 / 255.0)
                })
                .collect(),
        )
    }

    /// A filter-coefficient table (token 0x17e): the slot, then 33 words, stored as sent and replaced by a later
    /// table for the slot (established: the firmware stores the 33-word record at the slot and sends nothing to
    /// the hardware until a bind needs it). False for any other length or past the cap.
    pub fn filter_table(&mut self, slot: u32, words: &[u32]) -> bool {
        let Ok(w) = <[u32; 33]>::try_from(words) else { return false };
        if let Some(i) = self.filters.iter().position(|(s, _)| *s == slot) {
            self.filters[i].1 = w;
        } else if self.filters.len() < MAX_FILTER_TABLES {
            self.filters.push((slot, w));
        } else {
            return false;
        }
        true
    }

    /// A detail bind (token 0x17d, six words): the pipe stores the format row and the address pair, and the
    /// exponents reach the detail-size register (established).
    pub fn bind_detail(&mut self, a: &[u32]) -> bool {
        let Ok(a) = <[u32; 6]>::try_from(a) else { return false };
        self.detail_bind = Some(DetailBind { entry: a[0], exps: [a[1], a[2]], row: a[3], pair: [a[4], a[5]] });
        true
    }

    /// The detail image a mode-1 base bind (mode word `mode`) would combine with, if any: the pipe reads the
    /// address pair word the mode word's bit 28 chooses (word 4 when set, word 5 when clear, established). The
    /// client loads the detail image once to each bank, so it is looked for at that address in either; its size must
    /// be the detail bind's exponents.
    fn detail_image(&self, mode: u32) -> Option<&Level> {
        let d = self.detail_bind?;
        let addr = d.pair[if mode >> 28 & 1 != 0 { 0 } else { 1 }];
        [0u8, 1].into_iter().find_map(|bank| self.detail.get(&(bank, addr))).filter(|l| l.min.0 != u32::MAX && l.max.0 + 1 >= 1u32 << d.exps[0].min(12) && l.max.1 + 1 >= 1u32 << d.exps[1].min(12))
    }

    /// The magnification combine a mode-1 base bind asks for, from its mode word's code (3 detail add, 4 detail
    /// modulate, 5 sharpen; the detail codes only with the detail image `found`, and a sharpen with a coarser level
    /// to push away from). The curve is the filter table the control word's slot holds (bits 11:0, inferred),
    /// read as the spline form; the stored curve is the public one doubled for detail and negated for sharpen, which
    /// is undone here (established for the two transforms; that detail uses the doubled form and sharpen the negated
    /// one is inferred from the bind, since the table command carries no type). A slot that holds no table, or one
    /// that is not a spline, stands the client's default curve in (approximated).
    fn combine(&self, ctl: u32, mode: u32, found: Option<Level>, levels: u8) -> Combine {
        let (code, channels) = (mode >> 5 & 7, (mode >> 26 & 3) as u8);
        let curve = |scale: f32| -> [f32; CURVE_NODES] {
            match self.filters.iter().find(|(s, _)| *s == ctl & 0xfff).and_then(|(_, w)| spline_samples(w)) {
                Some(v) => v.map(|x| x * scale),
                None => default_weights(),
            }
        };
        match (code, found, self.detail_bind) {
            (3 | 4, Some(l), Some(b)) => {
                let [(ox, w), (oy, h)] = l.place([Some(1 << b.exps[0].min(12)), Some(1 << b.exps[1].min(12))]);
                let image = TexLevel { plane: l.plane, off: l.off, pitch: l.pw as u16, ox: ox as u16, oy: oy as u16, w: w as u16, h: h as u16, ..Default::default() };
                Combine::Detail(Detail { modulate: code == 4, scale: (mode >> 21 & 7) as u8, channels, image, weights: curve(0.5) })
            }
            (5, ..) if levels > 1 => Combine::Sharpen(Sharpen { channels, weights: curve(-1.0) }),
            _ => Combine::None,
        }
    }

    /// The short bind's five words, kept until its record arrives.
    pub fn short_bind(&mut self, words: &[u32]) {
        self.bind = <[u32; 5]>::try_from(words).ok();
    }

    /// The 2D record (22 words) completes the bind: the levels its table names, found by
    /// the first level's bank (bit 28 of the short bind's mode word) flipped for every
    /// second level. None when no short bind came before it, or level 0 is not found.
    pub fn record(&mut self, store: &TextureStore, r: &[u32], env: (EnvMode, [u32; 4]), counts: &mut TexCounts) -> Option<TexDesc> {
        let b = self.bind.take()?;
        let r: &[u32; 22] = r.try_into().ok()?;
        let (ctl, mut mode, size) = (b[1], b[2], b[3]);
        let first_bank = (mode >> 28 & 1) as u8;
        // The detail interaction of a mode-1 base bind of a 2D texture (target index 2): its format row (bits 23:12 of
        // the control word) is compared with the last detail bind's. A mismatch, or no detail bind, forces the
        // magnification code to 1, plain linear, whatever code the bind carried (nearest, linear, detail or
        // sharpen); a match sends the filter table and the detail address (established). The client sends mode 1
        // only for the detail filters, so in practice only the detail codes are rewritten.
        let mut found = None;
        if ctl >> 24 == 0x21 {
            if self.detail_bind.is_some_and(|d| d.row == ctl >> 12 & 0xfff) {
                if matches!(mode >> 5 & 7, 3 | 4) {
                    found = self.detail_image(mode).copied();
                    counts.detail_resolved += found.is_some() as u64;
                    counts.detail_unresolved += found.is_none() as u64;
                }
            } else {
                mode = mode & !(7 << 5) | 1 << 5;
                counts.detail_mismatched += 1;
            }
        }
        let mut d = TexDesc { env: env.0, env_colour: env.1, class: format_class(ctl >> 12 & 0xfff), ..Default::default() };
        for (k, &addr) in r[6..].iter().take_while(|&&w| w != 0xffff_ffff).take(MAX_LEVELS).enumerate() {
            let Some(l) = self.levels.get_mut(&(first_bank ^ (k as u8 & 1), addr)) else { break };
            // The bind's format samples whatever words the level holds, whatever loaded them.
            if l.raw {
                decode_level(store, l, mode >> 14 & 0x7f);
                l.raw = false;
                l.storage = mode >> 14 & 0x7f;
            }
            let l = *l;
            // The logical size is the bind's level-0 size halved per level; where it is smaller than the
            // rectangle the last copy gives the origin.
            let logical = |a: u32| Some(((1u32 << (size >> (4 * a) & 0xf).min(12)) >> k.min(31)).max(1));
            let [(ox, w), (oy, h)] = l.place([logical(0), logical(1)]);
            d.levels[k] = TexLevel { plane: l.plane, off: l.off, pitch: l.pw as u16, ox: ox as u16, oy: oy as u16, w: w as u16, h: h as u16, ..Default::default() };
            d.count += 1;
        }
        if d.count == 0 {
            return None;
        }
        apply_mode(&mut d, mode, size);
        d.lod_scale = [(r[1] & 0xff) as u8, (r[1] >> 8 & 0xff) as u8, 0];
        if ctl >> 24 == 0x21 {
            d.combine = self.combine(ctl, mode, found, d.count);
        }
        Some(d)
    }
    /// A volume record (token 0x17b with `mip`, 14 words, or 0x17c, 4 words) completes the bind a short bind
    /// began. The 3D mip-mapped form has the four state words of the 2D record (the layout pair, the LOD scales with
    /// r in bits 23:16, the bounds, the physical sizes) and ten descriptor words, level k at word k; the single-level
    /// form has the layout pair, the sizes and the one descriptor pair (framing established). A level's descriptor
    /// word is its address, found in the bank its parity names and then in the other (inferred), under
    /// the first level's bank for the single-level form. The logical depth is the size word's r exponent
    /// (bits 11:8) halved per level, as the other axes are (established). None when no short bind came
    /// before it or level 0 is not found.
    pub fn volume_record(&mut self, store: &TextureStore, r: &[u32], mip: bool, env: (EnvMode, [u32; 4])) -> Option<TexDesc> {
        let b = self.bind.take()?;
        let (ctl, mode, size) = (b[1], b[2], b[3]);
        let table: &[u32] = if mip {
            let r: &[u32; 14] = r.try_into().ok()?;
            &r[4..]
        } else {
            let r: &[u32; 4] = r.try_into().ok()?;
            &r[2..]
        };
        let first_bank = (mode >> 28 & 1) as u8;
        let mut d = TexDesc { env: env.0, env_colour: env.1, class: format_class(ctl >> 12 & 0xfff), volume: true, ..Default::default() };
        for k in 0..(if mip { table.len().min(MAX_LEVELS) } else { 1 }) {
            // A mip-mapped volume's pair k holds levels 2k and 2k + 1, the first in its even word.
            let (primary, addr) = if mip { ((k & 1) as u8, table[k]) } else { (first_bank, 0) };
            let key = [primary, primary ^ 1].into_iter().map(|bank| (bank, if mip { addr } else { table[bank as usize] })).find(|key| self.levels.contains_key(key));
            let Some(key) = key else { break };
            let l = self.levels.get_mut(&key).expect("just found");
            if l.raw {
                decode_level(store, l, mode >> 14 & 0x7f);
                l.raw = false;
                l.storage = mode >> 14 & 0x7f;
            }
            let l = *l;
            let logical = |a: u32| ((1u32 << (size >> (4 * a) & 0xf).min(12)) >> k.min(31)).max(1);
            let [(ox, w), (oy, h)] = l.place([Some(logical(0)), Some(logical(1))]);
            d.levels[k] = TexLevel { plane: l.plane, off: l.off, pitch: l.pw as u16, ox: ox as u16, oy: oy as u16, w: w as u16, h: h as u16, d: logical(2).min(l.pd) as u16, slice: l.pw * l.ph };
            d.count += 1;
        }
        if d.count == 0 {
            return None;
        }
        apply_mode(&mut d, mode, size);
        // The single-level form carries no LOD scales: a scale of one (16) stands in, which only decides where a
        // fragment is magnified or minified (approximated).
        d.lod_scale = if mip { [(r[1] & 0xff) as u8, (r[1] >> 8 & 0xff) as u8, (r[1] >> 16 & 0xff) as u8] } else { [16; 3] };
        Some(d)
    }
}

/// The filters and wraps the short bind's mode and size words give a bound texture (shared by the 2D and
/// volume binds). The wrap bits are the table's bits 12, 13 and 14 of the size word (established; one bit
/// cannot tell the non-repeat modes apart, so every one clamps).
fn apply_mode(d: &mut TexDesc, mode: u32, size: u32) {
    d.min = match mode >> 2 & 7 {
        0 => Filter::Nearest,
        1 | 2 => Filter::Linear,
        n @ 3..=6 => Filter::Mip { within_linear: n & 1 == 0, between_linear: n >= 5 },
        _ => Filter::Linear,
    };
    // Magnification codes 3 and 4 add or modulate a detail image, and 5 sharpens; a mode-1 base bind builds the
    // combine on top of this linear magnification (`TexContext::combine`). With no detail image bound, or one whose
    // format differs from the base's, the pipe magnifies the base linearly and shows it (the pipe has no
    // flat-colour state), which is what every nonzero code draws without a combine.
    d.mag = if mode >> 5 & 7 == 0 { Filter::Nearest } else { Filter::Linear };
    let wrap = |bit: u32| if size >> bit & 1 != 0 { Wrap::Clamp } else { Wrap::Repeat };
    d.wrap = [wrap(12), wrap(13), wrap(14)];
}

/// The format class of an internal-format index (section 7.12): 1 alpha, 2 luminance
/// or RGB, 3 luminance-alpha or RGBA, 4 intensity. Rows the table does not read count as
/// RGBA.
fn format_class(index: u32) -> u8 {
    match index {
        7 | 11 | 17 | 18 | 23 => 1,
        0 | 3 | 6 | 10 | 15 | 16 | 25 | 27 | 28 | 30 | 33 | 37 | 38 | 63 => 2,
        1 | 9 | 13 | 14 | 21 | 2 | 26 | 29 | 31 | 34 | 41 => 3,
        _ => 3,
    }
}

/// Whether `selector` is a texture token the texture path takes.
pub fn is_texture(selector: u32) -> bool {
    matches!(
        selector,
        DRAW_TEXELS_INTO_TEXTURE_MEMORY
            | TEXTURE_LOAD_BY_IDENTIFIER
            | BIND_TEXTURE_CURRENT_FORM
            | BIND_LINEAR_TEXTURE
            | BIND_2D_MIP_MAPPED_TEXTURE
            | BIND_3D_MIP_MAPPED_TEXTURE
            | BIND_3D_TEXTURE
            | BIND_DETAIL_TEXTURE
            | BIND_TEXTURE_BY_IDENTIFIER
            | BIND_ARM_TEXTURE_CACHE_ENTRY
            | FREE_ARM_TEXTURE_CACHE_ENTRY
            | SET_TEXEL_STORE_PARAMETER
            | SET_PIXEL_STORE_PARAMETER_INTEGER_FORM
            | TEXTURE_LOAD_SET_UP
            | BIND_TEXTURE_BY_OLD_STYLE_BINDING
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dev::kona::back::store::texture::PAGE_WORDS;

    /// A texel draw's header: a w x h RGBA image of unsigned bytes, one level at `address`.
    fn header(w: u32, h: u32) -> [u32; 20] {
        let mut a = [0u32; 20];
        a[0] = 0x0de1;
        a[1] = w;
        a[2] = h;
        a[3] = 5;
        a[4] = 1;
        a[9] = 0x30;
        a[13] = w.wrapping_sub(1);
        a[14] = h.wrapping_sub(1);
        a[16] = 0x11;
        a[17] = 0x2200;
        a[18] = 0x1000;
        a
    }

    fn parse(a: &[u32; 20]) -> Header {
        Header::parse(a).unwrap()
    }

    /// A guest's unpack state can make the sizing products overflow 64 bits (a skip of 2^32 rows of a stride of 2^36
    /// bytes): that is no size, not a panic or a wrapped one.
    #[test]
    fn an_unpack_state_whose_sizes_overflow_has_no_image_size() {
        let big = Unpack { row_length: u32::MAX, skip_rows: u32::MAX, skip_pixels: u32::MAX, alignment: 8, swap_bytes: false };
        let mut h = parse(&header(4, u32::MAX));
        assert_eq!(h.image_words(&big), None);
        h.strided = true;
        assert_eq!(h.image_words(&big), None);
        // A tall strided image past what a stream carries stops adding.
        assert_eq!(h.image_words(&Unpack::default()), None);
    }

    /// An unpack row length below the width is no size, not an underflow.
    #[test]
    fn a_row_length_below_the_width_has_no_image_size() {
        let h = parse(&header(100, 4));
        assert_eq!(h.image_words(&Unpack { row_length: 10, ..Unpack::default() }), None);
        assert_eq!(h.image_words(&Unpack { row_length: 100, ..Unpack::default() }), Some(400));
    }

    /// The image's length in words, by the sizing rule: contiguous rows are one byte
    /// stream, strided rows each take the words spanning their bytes.
    #[test]
    fn the_image_length_follows_the_unpack_state() {
        let u = Unpack::default();
        assert_eq!(parse(&header(4, 2)).image_words(&u), Some(8));
        let mut a = header(4, 2);
        a[5] = 2;
        assert_eq!(parse(&a).image_words(&u), Some(9), "a start two bytes into the first word spans one more");
        a[7] = 1;
        assert_eq!(parse(&a).image_words(&u), Some(10), "strided: each row's own span, five words");
        // RGB bytes, three wide, rows padded to the alignment of 4: stride 12, 21 bytes.
        let mut rgb = header(3, 2);
        rgb[3] = 4;
        assert_eq!(parse(&rgb).image_words(&u), Some(6));
        assert_eq!(parse(&rgb).image_words(&Unpack { alignment: 1, ..u }), Some(5), "no padding: 18 bytes");
        assert_eq!(parse(&rgb).image_words(&Unpack { row_length: 8, ..u }), Some(9), "a row length of 8 pixels: stride 24, 33 bytes");
        assert_eq!(parse(&header(0, 0)).image_words(&u), Some(0));
        let mut cmyk = header(2, 2);
        cmyk[3] = 9;
        assert_eq!(parse(&cmyk).image_words(&u), None, "a format the model does not decode");
        let mut packed = header(2, 2);
        packed[4] = 8;
        assert_eq!(parse(&packed).image_words(&u), None, "nor a packed type");
    }

    fn words_to_bytes(w: &[u32]) -> Vec<u8> {
        w.iter().flat_map(|w| w.to_be_bytes()).collect()
    }

    /// Word `i` of the level at (`bank`, `address`) in `c`'s texture memory: a level is where its address puts it.
    fn word(c: &TexContext, store: &TextureStore, bank: u8, address: u32, i: u32) -> u32 {
        store.get(c.plane_of(false, bank), address * UNIT_WORDS + i)
    }

    /// A load puts the texels where the header says, in the model's layout, and a load
    /// into the same level again replaces it.
    #[test]
    fn a_load_stores_texels_in_the_models_layout() {
        let store = TextureStore::default();
        let (mut c, mut n) = (TexContext::default(), TexCounts::default());
        let img = [0xff00_00ffu32, 0x00ff_00ff, 0x0000_ffff, 0xffff_ffff];
        assert!(c.load(&store, &header(2, 2), &words_to_bytes(&img), false, &mut n));
        assert_eq!([0, 1, 2, 3].map(|i| word(&c, &store, 0, 0x1000, i)), [0xffff_0000, 0xff00_ff00, 0xff00_00ff, 0xffff_ffff]);
        let again = [0x0101_01ffu32; 4];
        assert!(c.load(&store, &header(2, 2), &words_to_bytes(&again), false, &mut n));
        assert_eq!((store.resident_pages(), word(&c, &store, 0, 0x1000, 0)), (1, 0xff01_0101), "same address and size: replaced in place");
        assert_eq!(n.loads, 2);
        // RGB: alpha is one. Luminance: the colour, alpha one. Luminance-alpha: both.
        let mut rgb = header(1, 1);
        rgb[3] = 4;
        rgb[18] = 0x2000;
        c.load(&store, &rgb, &words_to_bytes(&[0x1020_3000]), false, &mut n);
        assert_eq!(word(&c, &store, 0, 0x2000, 0), 0xff10_2030, "a level at another address is another place");
        let mut lum = header(1, 1);
        lum[3] = 6;
        lum[18] = 0x3000;
        c.load(&store, &lum, &words_to_bytes(&[0x8000_0000]), false, &mut n);
        assert_eq!(word(&c, &store, 0, 0x3000, 0), 0xff80_8080);
        let mut la = header(1, 1);
        la[3] = 7;
        la[18] = 0x4000;
        c.load(&store, &la, &words_to_bytes(&[0x4020_0000]), false, &mut n);
        assert_eq!(word(&c, &store, 0, 0x4000, 0), 0x2040_4040);
        // Alpha only: no colour.
        let mut al = header(1, 1);
        al[3] = 3;
        al[18] = 0x5000;
        c.load(&store, &al, &words_to_bytes(&[0x7700_0000]), false, &mut n);
        assert_eq!(word(&c, &store, 0, 0x5000, 0), 0x7700_0000);
        // 16-bit unsigned shorts keep their top byte; a float clamps and scales.
        let mut sh = header(1, 1);
        sh[3] = 6;
        sh[4] = 3;
        sh[18] = 0x6000;
        c.load(&store, &sh, &words_to_bytes(&[0x8000_0000]), false, &mut n);
        assert_eq!(word(&c, &store, 0, 0x6000, 0), 0xff80_8080);
        let mut fl = header(1, 1);
        fl[3] = 6;
        fl[4] = 6;
        fl[18] = 0x7000;
        c.load(&store, &fl, &words_to_bytes(&[1.5f32.to_bits()]), false, &mut n);
        assert_eq!(word(&c, &store, 0, 0x7000, 0), 0xffff_ffff);
    }

    /// The packed pixel types (codes 8 to 12) load: the library sends 4-4-4-4 textures that way, and a load the
    /// model refused left every bind of its level refused too. The first component is in the high bits, the stream
    /// is big-endian, and the library pairs 3-3-2 with RGB and the others with RGBA.
    #[test]
    fn a_packed_pixel_type_loads_as_texels() {
        let u = Unpack::default();
        let load = |ty: u32, format: u32, storage: u32, w: u32, bytes: &[u8]| -> Option<Vec<u32>> {
            let store = TextureStore::default();
            let (mut c, mut n) = (TexContext::default(), TexCounts::default());
            let mut a = header(w, 1);
            (a[3], a[4], a[9], a[16]) = (format, ty, storage, 0x33);
            let words = parse(&a).image_words(&u)?;
            assert_eq!(words as usize, bytes.len().div_ceil(4), "the stream is as long as the packed pixels");
            c.load(&store, &a, bytes, false, &mut n).then(|| (0..w).map(|i| word(&c, &store, 0, 0x1000, i)).collect())
        };
        assert_eq!(load(9, 5, 0x10, 2, &[0xf0, 0x5a, 0x12, 0x34]), Some(vec![0xaaff_0055, 0x4411_2233]), "4-4-4-4: red, green, blue, alpha from the top");
        assert_eq!(load(10, 5, 0x14, 2, &[0xf8, 0x01, 0x07, 0xc0]), Some(vec![0xffff_0000, 0x0000_ff00]), "5-5-5-1");
        assert_eq!(load(8, 4, 0x30, 4, &[0xe0, 0x1f, 0x03, 0x00]), Some(vec![0xffff_0000, 0xff00_ffff, 0xff00_00ff, 0xff00_0000]), "3-3-2, RGB");
        assert_eq!(load(11, 5, 0x30, 1, &[0x11, 0x22, 0x33, 0x44]), Some(vec![0x4411_2233]), "8-8-8-8");
        assert_eq!(load(12, 5, 0x30, 1, &[0xff, 0xc0, 0x0f, 0xff]), Some(vec![0xffff_00ff]), "10-10-10-2");
        // A type with the wrong format is the library's error, not a load.
        assert_eq!(load(9, 4, 0x10, 1, &[0, 0]), None);
        assert_eq!(load(8, 5, 0x30, 1, &[0]), None);
        // Swapped bytes read the packed word little-endian.
        let store = TextureStore::default();
        let (mut c, mut n) = (TexContext { unpack: Unpack { swap_bytes: true, ..u }, ..Default::default() }, TexCounts::default());
        let mut a = header(2, 1);
        (a[3], a[4], a[9], a[16]) = (5, 9, 0x10, 0x33);
        assert!(c.load(&store, &a, &[0x5a, 0xf0, 0x34, 0x12], false, &mut n));
        assert_eq!((word(&c, &store, 0, 0x1000, 0), word(&c, &store, 0, 0x1000, 1)), (0xaaff_0055, 0x4411_2233));
    }

    /// A storage code narrower than a texel reduces it as the board's texel would be.
    #[test]
    fn a_narrow_storage_code_reduces_the_texel() {
        let store = TextureStore::default();
        let (mut c, mut n) = (TexContext::default(), TexCounts::default());
        let mut a = header(1, 1);
        a[9] = 0x10;
        c.load(&store, &a, &words_to_bytes(&[0x8040_20ff]), false, &mut n);
        assert_eq!(word(&c, &store, 0, 0x1000, 0), 0xffff_0000 & 0xff00_0000 | 0x0088_4422, "4 bits a component: 0x80 is 8 of 15, 0x88");
        let mut a = header(1, 1);
        a[9] = 0x14;
        a[18] = 0x2000;
        c.load(&store, &a, &words_to_bytes(&[0xff00_0000]), false, &mut n);
        assert_eq!(word(&c, &store, 0, 0x2000, 0), 0x00ff_0000, "5-5-5-1: a clear alpha bit, full red");
        let mut a = header(1, 1);
        a[9] = 0x18;
        a[18] = 0x3000;
        c.load(&store, &a, &words_to_bytes(&[0x8040_20ff]), false, &mut n);
        assert_eq!(word(&c, &store, 0, 0x3000, 0), 0xff83_4020, "R5-G6-B5: red 0x80 is 16 of 31 (0x83), green 0x40 is 16 of 63 (0x40), blue 0x20 is 4 of 31 (0x20), alpha one");
    }

    /// A DMA image is its rows, `w * p` bytes each, with none of the stream's offset or padding.
    #[test]
    fn a_dma_image_is_its_rows() {
        let store = TextureStore::default();
        let (mut c, mut n) = (TexContext::default(), TexCounts::default());
        let mut a = header(2, 1);
        a[6] = 1;
        c.load(&store, &a, &[1, 2, 3, 4, 5, 6, 7, 8], true, &mut n);
        assert_eq!((word(&c, &store, 0, 0x1000, 0), word(&c, &store, 0, 0x1000, 1)), (0x0401_0203, 0x0805_0607));
        // A load whose box leaves its physical rectangle is refused, and memory is not touched.
        let mut bad = header(2, 2);
        bad[13] = 9;
        assert!(!c.load(&store, &bad, &[0; 16], false, &mut n));
        assert_eq!(n.unsupported_load, 1);
    }

    /// A short bind and its record find the levels by (bank, address): the first level's bank
    /// from bit 28 of the mode word, flipped for every second level; filters, wraps and the
    /// LOD scale come from the words; no short bind before it, or no level found, binds nothing.
    #[test]
    fn a_bind_finds_its_levels_by_bank_and_address() {
        let store = TextureStore::default();
        let (mut c, mut n) = (TexContext::default(), TexCounts::default());
        // Level 0, 2 x 2, in bank 1 at 0x1000; level 1, 1 x 1, in bank 0 at 0x2000.
        let mut l0 = header(2, 2);
        l0[17] |= 8;
        l0[19] = 0x1000;
        l0[18] = 0;
        c.load(&store, &l0, &words_to_bytes(&[0xff00_00ff; 4]), false, &mut n);
        let mut l1 = header(1, 1);
        l1[18] = 0x2000;
        l1[16] = 0x00;
        c.load(&store, &l1, &words_to_bytes(&[0x0000_ffff]), false, &mut n);
        let mut rec = [0u32; 22];
        rec[0] = 0x44;
        rec[1] = 0x2020;
        rec[6] = 0x1000;
        rec[7] = 0x2000;
        rec[8] = 0xffff_ffff;
        assert!(c.record(&store, &rec, Default::default(), &mut TexCounts::default()).is_none(), "a record with no short bind");
        // Mode: minification 6 (mip, linear within, linear between), magnification 1, first bank 1.
        let bind = [0xffff_ffffu32, 2 << 12, 6 << 2 | 1 << 5 | 1 << 28, 0x3011, 1];
        c.short_bind(&bind);
        let d = c.record(&store, &rec, Default::default(), &mut TexCounts::default()).expect("both levels found");
        assert_eq!((d.count, d.levels[0].w, d.levels[1].w, d.class), (2, 2, 1, 3));
        assert_eq!(d.min, Filter::Mip { within_linear: true, between_linear: true });
        assert_eq!((d.mag, d.wrap, d.lod_scale), (Filter::Linear, [Wrap::Clamp, Wrap::Clamp, Wrap::Repeat], [0x20, 0x20, 0]));
        // The wrong first bank finds nothing.
        c.short_bind(&[0xffff_ffff, 2 << 12, 6 << 2, 0x11, 1]);
        assert!(c.record(&store, &rec, Default::default(), &mut TexCounts::default()).is_none());
        // Only level 0 loaded: a table of two finds one.
        c.short_bind(&bind);
        rec[7] = 0x9999;
        assert_eq!(c.record(&store, &rec, Default::default(), &mut TexCounts::default()).unwrap().count, 1);
    }

    /// The bind's size word gives the logical size, so an update of part of a level changes nothing, and a small
    /// level's origin is its last copy's.
    #[test]
    fn a_bind_places_levels_by_its_logical_size() {
        let store = TextureStore::default();
        let mut c = TexContext::default();
        load_copy(&mut c, &store, &copy(8, 4, 0, 0, 0x100));
        load_copy(&mut c, &store, &copy(4, 2, 2, 1, 0x100));
        load_copy(&mut c, &store, &copy(1, 1, 7, 3, 0x200));
        let mut rec = [0u32; 22];
        rec[6..8].copy_from_slice(&[0x100, 0xffff_ffff]);
        c.short_bind(&[0xffff_ffff, 2 << 12, 0, 0x23, 1]);
        let d = c.record(&store, &rec, Default::default(), &mut TexCounts::default()).unwrap();
        assert_eq!([d.levels[0].ox, d.levels[0].oy, d.levels[0].w, d.levels[0].h], [0, 0, 8, 4]);
        rec[6] = 0x200;
        c.short_bind(&[0xffff_ffff, 2 << 12, 0, 0x00, 1]);
        let d = c.record(&store, &rec, Default::default(), &mut TexCounts::default()).unwrap();
        assert_eq!([d.levels[0].ox, d.levels[0].oy, d.levels[0].w, d.levels[0].h], [7, 3, 1, 1]);
    }

    /// A slice of a 2 x 2 volume (physical 2 x 2 x 2) at `address` in `bank`, filled with `texel`.
    fn slice(c: &mut TexContext, store: &TextureStore, n: &mut TexCounts, bank: u32, address: u32, z: u32, texel: u32) -> bool {
        let mut a = header(2, 2);
        a[0] = 0x806f;
        a[15] = z;
        a[16] = 0x111;
        a[17] |= bank << 3;
        a[18 + bank as usize] = address;
        c.load(store, &a, &words_to_bytes(&[texel; 4]), false, n)
    }

    /// A volume's slices stack in one level found by the slices' shared address; a slice past the level's physical
    /// depth is refused. The single-level record (token 0x17c) finds that level in the first level's bank and the
    /// sampler reads each slice at its own r.
    #[test]
    fn a_volume_is_its_slices_and_binds_by_its_pair() {
        use super::super::super::raster::texel::sample3;
        let store = TextureStore::default();
        let (mut c, mut n) = (TexContext::default(), TexCounts::default());
        assert!(slice(&mut c, &store, &mut n, 0, 0x40, 0, 0xff00_00ff));
        assert!(slice(&mut c, &store, &mut n, 0, 0x40, 1, 0x0000_ffff));
        assert!(!slice(&mut c, &store, &mut n, 0, 0x40, 2, 0), "the level is two slices deep");
        assert_eq!((n.loads, n.volume_loads, n.unsupported_load, c.levels.len()), (2, 2, 1, 1));
        assert_eq!((store.resident_pages(), word(&c, &store, 0, 0x40, 0), word(&c, &store, 0, 0x40, 4)), (1, 0xffff_0000, 0xff00_00ff), "slices stacked a slice apart");
        // Size word: 2 x 2 x 2, s and t repeat, r clamps. Mode word: nearest, bank 0.
        c.short_bind(&[0xffff_ffff, 3 << 28 | 2 << 12, 0, 0x4011 | 1 << 8, 1]);
        let d = c.volume_record(&store, &[0x44, 0x111, 0x40, 0], false, Default::default()).expect("the pair finds the level");
        assert!(d.volume && d.count == 1);
        assert_eq!((d.levels[0].d, d.levels[0].slice, d.levels[0].w, d.levels[0].h), (2, 4, 2, 2));
        assert_eq!(d.wrap, [Wrap::Repeat, Wrap::Repeat, Wrap::Clamp]);
        assert_eq!(sample3(&store, &d, 0.25, 0.25, 0.25, 0.0), [255, 0, 0, 255]);
        assert_eq!(sample3(&store, &d, 0.25, 0.25, 0.75, 0.0), [0, 0, 255, 255]);
        // No short bind before it, a pair that names nothing, or a record of the wrong length binds nothing.
        assert!(c.volume_record(&store, &[0x44, 0x111, 0x40, 0], false, Default::default()).is_none());
        c.short_bind(&[0xffff_ffff, 3 << 28, 0, 0x111, 1]);
        assert!(c.volume_record(&store, &[0x44, 0x111, 0x99, 0], false, Default::default()).is_none());
        c.short_bind(&[0xffff_ffff, 3 << 28, 0, 0x111, 1]);
        assert!(c.volume_record(&store, &[0x44, 0x111, 0x40], false, Default::default()).is_none());
    }

    /// A mip-mapped volume record (token 0x17b) lists one descriptor word a level and a level's bank is its parity:
    /// level 1 is in bank 1, found by its word of the pair. Each level halves every axis, depth included, and the
    /// LOD scales carry r in bits 23:16.
    #[test]
    fn a_mip_mapped_volume_finds_its_levels_by_parity() {
        let store = TextureStore::default();
        let (mut c, mut n) = (TexContext::default(), TexCounts::default());
        for z in 0..2 {
            assert!(slice(&mut c, &store, &mut n, 0, 0x40, z, 0xff00_00ff));
        }
        // Level 1: 1 x 1 x 1 (physical 1 x 1 x 1: exponent 0), bank 1.
        let mut a = header(1, 1);
        a[0] = 0x806f;
        a[16] = 0;
        a[17] |= 8;
        a[19] = 0x80;
        assert!(c.load(&store, &a, &words_to_bytes(&[0x0000_ffff]), false, &mut n));
        let mut rec = [0u32; 14];
        rec[0] = 0x44;
        rec[1] = 0x10_20_30;
        rec[3] = 0x111;
        rec[4] = 0x40;
        rec[5] = 0x80;
        c.short_bind(&[0xffff_ffff, 3 << 28, 6 << 2 | 1 << 5, 0x111, 1]);
        let d = c.volume_record(&store, &rec, true, Default::default()).expect("both levels found");
        assert_eq!(d.count, 2);
        assert_eq!(d.min, Filter::Mip { within_linear: true, between_linear: true });
        assert_eq!([(d.levels[0].w, d.levels[0].h, d.levels[0].d), (d.levels[1].w, d.levels[1].h, d.levels[1].d)], [(2, 2, 2), (1, 1, 1)]);
        assert_eq!(d.lod_scale, [0x30, 0x20, 0x10]);
        assert_eq!(n.volume_loads, 3);
    }

    /// A filter-coefficient table is stored as sent under its slot and replaced by a later one; a table of any
    /// other length is refused.
    #[test]
    fn a_filter_table_is_kept_by_slot() {
        let mut c = TexContext::default();
        assert!(c.filter_table(3, &[7; 33]));
        assert!(c.filter_table(4, &[8; 33]));
        assert!(c.filter_table(3, &[9; 33]));
        assert!(!c.filter_table(5, &[1; 32]));
        assert_eq!(c.filters, vec![(3, [9; 33]), (4, [8; 33])]);
        for slot in 10..10 + MAX_FILTER_TABLES as u32 {
            c.filter_table(slot, &[0; 33]);
        }
        assert_eq!(c.filters.len(), MAX_FILTER_TABLES, "the extra slots are dropped");
    }

    /// A mode-1 base bind of a 2D texture compares its format row with the last detail bind's. With none, or a
    /// different row, a detail magnification code is forced to linear; with the same row it is
    /// left alone, and the detail image the address pair names (word 4 when the mode word's bit 28 is set, else
    /// word 5) is looked for. The 0x106 form (mode 3) and a bind of another target do neither.
    #[test]
    fn a_mode_1_base_bind_checks_the_last_detail_bind() {
        let store = TextureStore::default();
        let (mut c, mut n) = (TexContext::default(), TexCounts::default());
        load_copy(&mut c, &store, &copy(8, 4, 0, 0, 0));
        let mut other = copy(8, 4, 0, 0, 0);
        other[17] |= 8;
        other[19] = 0;
        other[18] = 0;
        load_copy(&mut c, &store, &other);
        // A 8 x 4 detail image in both banks at address 0x200.
        for bank in [0u32, 1] {
            let mut d = copy(8, 4, 0, 0, 0);
            d[0] = 0x8095;
            d[17] |= bank << 3;
            d[18 + bank as usize] = 0x200;
            assert!(c.load(&store, &d, &vec![0x40; 8 * 4 * 4], false, &mut n));
        }
        let mut rec = [0u32; 22];
        rec[6..8].copy_from_slice(&[0, 0xffff_ffff]);
        let mode1 = |row: u32| 2 << 28 | 1 << 24 | row << 12;
        let bind = |c: &mut TexContext, ctl: u32, code: u32, bank: u32, n: &mut TexCounts| {
            c.short_bind(&[0xffff_ffff, ctl, code << 5 | bank << 28, 0x23, 1]);
            c.record(&store, &rec, Default::default(), n).unwrap()
        };
        let mut n = TexCounts::default();
        // No detail bind: every code is forced to linear, a nearest one and a sharpen one too (established).
        assert_eq!(bind(&mut c, mode1(0x30), 4, 0, &mut n).mag, Filter::Linear);
        assert_eq!(n.detail_mismatched, 1);
        assert_eq!(bind(&mut c, mode1(0x30), 0, 0, &mut n).mag, Filter::Linear);
        assert_eq!(bind(&mut c, mode1(0x30), 5, 0, &mut n).mag, Filter::Linear);
        assert_eq!(n.detail_mismatched, 3);
        // A detail bind of another row: the same.
        assert!(c.bind_detail(&[0xffff_ffff, 3, 2, 0x31, 0x200, 0x200]));
        assert_eq!(bind(&mut c, mode1(0x30), 3, 0, &mut n).mag, Filter::Linear);
        assert_eq!(n.detail_mismatched, 4);
        // The same row: the image is found by the pair word the bit names; a code that is not a detail one
        // keeps itself.
        bind(&mut c, mode1(0x31), 3, 0, &mut n);
        assert_eq!((n.detail_mismatched, n.detail_resolved, n.detail_unresolved), (4, 1, 0));
        assert_eq!(bind(&mut c, mode1(0x31), 0, 0, &mut n).mag, Filter::Nearest);
        assert_eq!(n.detail_mismatched, 4);
        // The pair word the bit chooses names nothing loaded: unresolved. A size that differs is unresolved too.
        assert!(c.bind_detail(&[0xffff_ffff, 3, 2, 0x31, 0x300, 0x200]));
        bind(&mut c, mode1(0x31), 4, 1, &mut n);
        assert_eq!((n.detail_resolved, n.detail_unresolved), (1, 1), "bit 28 set reads word 4, 0x300");
        bind(&mut c, mode1(0x31), 4, 0, &mut n);
        assert_eq!((n.detail_resolved, n.detail_unresolved), (2, 1), "bit 28 clear reads word 5, 0x200");
        assert!(c.bind_detail(&[0xffff_ffff, 4, 2, 0x31, 0x200, 0x200]));
        bind(&mut c, mode1(0x31), 4, 0, &mut n);
        assert_eq!(n.detail_unresolved, 2, "a 16 wide detail bind does not match an 8 wide image");
        // Other forms: no interaction.
        let before = (n.detail_mismatched, n.detail_resolved, n.detail_unresolved);
        assert_eq!(bind(&mut c, 2 << 28 | 3 << 24 | 0x32 << 12, 4, 0, &mut n).mag, Filter::Linear);
        assert_eq!(bind(&mut c, 3 << 28 | 1 << 24 | 0x32 << 12, 4, 0, &mut n).mag, Filter::Linear);
        assert_eq!((n.detail_mismatched, n.detail_resolved, n.detail_unresolved), before);
        assert!(!c.bind_detail(&[0; 5]));
    }

    /// The combine a mode-1 base bind asks for: detail add or modulate with the image found, from the spline table in the
    /// control word's slot with the stored doubling undone; sharpen from the same slot with the negation undone; the
    /// default curve when the slot holds no table; and none when the image is not found or the code is plain.
    #[test]
    fn a_mode_1_base_bind_builds_its_combine_from_the_slot_s_table() {
        let store = TextureStore::default();
        let (mut c, mut n) = (TexContext::default(), TexCounts::default());
        // Two levels, 8 x 4 in bank 0 and 4 x 2 in bank 1 (the bank flips for every second level).
        load_copy(&mut c, &store, &copy(8, 4, 0, 0, 0));
        let mut l1 = copy(4, 2, 0, 0, 0);
        l1[17] |= 8;
        l1[18] = 0;
        l1[19] = 0x100;
        load_copy(&mut c, &store, &l1);
        let mut d = copy(8, 4, 0, 0, 0x200);
        d[0] = 0x8095;
        assert!(c.load(&store, &d, &vec![0x40; 8 * 4 * 4], false, &mut n));
        let mut rec = [0u32; 22];
        rec[6..9].copy_from_slice(&[0, 0x100, 0xffff_ffff]);
        let code = 256u32; // mantissa 64, exponent 0: one half
        let mut table = [code | code << 10 | code << 20; 33];
        for pair in table.chunks_exact_mut(2) {
            pair[1] = code >> 2;
        }
        assert!(c.filter_table(7, &table[..33]));
        assert!(c.bind_detail(&[0xffff_ffff, 3, 2, 0x31, 0x200, 0x200]));
        let ctl = 2 << 28 | 1 << 24 | 0x31 << 12 | 7;
        let bind = |c: &mut TexContext, code: u32, level_code: u32, channels: u32| {
            c.short_bind(&[0xffff_ffff, ctl, code << 5 | level_code << 21 | channels << 26, 0x23, 1]);
            c.record(&store, &rec, Default::default(), &mut TexCounts::default()).unwrap()
        };
        match bind(&mut c, 3, 4, 1).combine {
            Combine::Detail(x) => assert_eq!((x.modulate, x.scale, x.channels, x.image.w, x.image.h, x.weights[10]), (false, 4, 1, 8, 4, 0.25)),
            other => panic!("{other:?}"),
        }
        assert!(matches!(bind(&mut c, 4, 0, 0).combine, Combine::Detail(x) if x.modulate && x.weights[63] == 0.25));
        assert!(matches!(bind(&mut c, 5, 0, 0).combine, Combine::Sharpen(x) if x.weights[0] == -0.5));
        for plain in [0, 1, 2] {
            assert_eq!(bind(&mut c, plain, 0, 0).combine, Combine::None, "code {plain}");
        }
        // No table in the slot: the client's default curve, still combined. No image: no detail combine, only linear.
        let ctl = 2 << 28 | 1 << 24 | 0x31 << 12 | 8;
        c.short_bind(&[0xffff_ffff, ctl, 3 << 5, 0x23, 1]);
        match c.record(&store, &rec, Default::default(), &mut TexCounts::default()).unwrap().combine {
            Combine::Detail(x) => assert_eq!(x.weights, default_weights()),
            other => panic!("{other:?}"),
        }
        assert!(c.bind_detail(&[0xffff_ffff, 3, 2, 0x31, 0x900, 0x900]));
        c.short_bind(&[0xffff_ffff, ctl, 3 << 5, 0x23, 1]);
        let d = c.record(&store, &rec, Default::default(), &mut TexCounts::default()).unwrap();
        assert_eq!((d.combine, d.mag), (Combine::None, Filter::Linear), "the detail image is not loaded: the base shows linearly");
        // A sharpen of a one-level texture has nothing to push away from.
        let mut one = [0u32; 22];
        one[6..8].copy_from_slice(&[0, 0xffff_ffff]);
        c.short_bind(&[0xffff_ffff, ctl, 5 << 5, 0x23, 1]);
        assert_eq!(c.record(&store, &one, Default::default(), &mut TexCounts::default()).unwrap().combine, Combine::None);
    }

    /// A detail image (target 0x8095) is sent once to each bank, at the offsets the base levels use too (0 and 0
    /// are recorded), and must not retire or overwrite a base level there. The base binds as it was loaded, with
    /// linear magnification for the detail codes 3 and 4 and for sharpen, since nothing combines them.
    #[test]
    fn a_detail_load_leaves_the_base_level_alone() {
        let store = TextureStore::default();
        let mut c = TexContext::default();
        let mut n = TexCounts::default();
        let base = copy(8, 4, 0, 0, 0);
        assert!(c.load(&store, &base, &vec![0x11; 8 * 4 * 4], false, &mut n));
        for bank in [0u32, 1] {
            let mut d = copy(8, 4, 0, 0, 0);
            d[0] = 0x8095;
            d[17] |= bank << 3;
            d[18 + bank as usize] = 0;
            assert!(c.load(&store, &d, &vec![0x77; 8 * 4 * 4], false, &mut n));
        }
        assert_eq!((n.loads, n.detail_loads), (3, 2));
        assert_eq!(c.detail.len(), 2, "the same image in each bank");
        assert_eq!(c.levels.len(), 1, "the base level was not retired");
        let l = c.levels[&(0, 0)];
        assert_eq!(store.get(l.plane, l.off), 0x1111_1111, "nor overwritten");
        // The base binds, magnified linearly whatever the detail code says.
        let mut rec = [0u32; 22];
        rec[6..8].copy_from_slice(&[0, 0xffff_ffff]);
        for (code, mag) in [(0, Filter::Nearest), (1, Filter::Linear), (3, Filter::Linear), (4, Filter::Linear), (5, Filter::Linear)] {
            c.short_bind(&[0xffff_ffff, 2 << 12 | 1 << 24, code << 5, 0x23, 1]);
            let d = c.record(&store, &rec, Default::default(), &mut TexCounts::default()).unwrap();
            assert_eq!((d.count, d.mag), (1, mag), "magnification code {code}");
        }
    }

    /// A texel draw of a w x h level at (x0, y0) of an 8 x 4 tile, at `address`.
    fn copy(w: u32, h: u32, x0: u32, y0: u32, address: u32) -> [u32; 20] {
        let mut a = header(w, h);
        a[11] = x0;
        a[12] = y0;
        a[13] = x0 + w - 1;
        a[14] = y0 + h - 1;
        a[16] = 0x23;
        a[18] = address;
        a
    }

    fn load_copy(c: &mut TexContext, store: &TextureStore, a: &[u32; 20]) {
        let (w, h) = (a[1] as usize, a[2] as usize);
        assert!(c.load(store, a, &vec![0x80; w * h * 4], false, &mut TexCounts::default()));
    }

    /// A whole load of a `1 << se` by `1 << te` level at `address`.
    fn whole(se: u32, te: u32, address: u32) -> [u32; 20] {
        let mut a = copy(1 << se, 1 << te, 0, 0, address);
        a[16] = te << 4 | se;
        a
    }

    fn load_whole(c: &mut TexContext, store: &TextureStore, a: &[u32; 20], n: &mut TexCounts) -> bool {
        let (w, h) = (a[1] as usize, a[2] as usize);
        c.load(store, a, &vec![0x80; w * h * 4], false, n)
    }

    /// A long run of reloads at the same addresses keeps texture memory where the first round left it: a
    /// level reloaded at the same place overwrites it, so nothing is made again.
    #[test]
    fn a_long_run_of_reloads_at_the_same_addresses_stays_bounded() {
        let store = TextureStore::with_capacity_bytes(1 << 20);
        let (mut c, mut n) = (TexContext::default(), TexCounts::default());
        for k in 0..16 {
            assert!(load_whole(&mut c, &store, &whole(3, 2, 0x100 + 0x10 * k), &mut n));
        }
        let after_first = (store.resident_pages(), store.used());
        for round in 0..2000 {
            for k in 0..16 {
                assert!(load_whole(&mut c, &store, &whole(3, 2, 0x100 + 0x10 * k), &mut n), "round {round}");
            }
        }
        assert_eq!((store.resident_pages(), store.used()), after_first);
        assert_eq!((n.loads, n.full), (16 * 2001, 0));
    }

    /// A level reloaded at the same address in another size takes the old one's place: the old is retired, so
    /// alternating sizes leave one level and no growth.
    #[test]
    fn a_level_reloaded_in_another_size_replaces_the_old_one() {
        let store = TextureStore::with_capacity_bytes(1 << 20);
        let (mut c, mut n) = (TexContext::default(), TexCounts::default());
        for round in 0..1000 {
            let (se, te) = if round % 2 == 0 { (3, 2) } else { (4, 3) };
            assert!(load_whole(&mut c, &store, &whole(se, te, 0x200), &mut n));
            assert_eq!(c.levels.len(), 1, "round {round}");
        }
        assert_eq!(store.resident_pages(), 1);
    }

    /// A load at a new address that overlaps a level retires it; a level it does not overlap stays.
    #[test]
    fn a_load_that_overlaps_a_level_retires_it() {
        let store = TextureStore::with_capacity_bytes(1 << 20);
        let (mut c, mut n) = (TexContext::default(), TexCounts::default());
        // 16 x 8 texels of 16 bits are eight units, 8 x 4 are two.
        assert!(load_whole(&mut c, &store, &whole(4, 3, 0x100), &mut n));
        assert!(load_whole(&mut c, &store, &whole(3, 2, 0x200), &mut n));
        assert!(load_whole(&mut c, &store, &whole(3, 2, 0x104), &mut n));
        assert_eq!(c.levels.keys().copied().collect::<std::collections::BTreeSet<_>>(), [(0, 0x104), (0, 0x200)].into(), "the level at 0x100 was retired, the one at 0x200 was not");
        assert!(load_whole(&mut c, &store, &whole(3, 2, 0x108), &mut n) && c.levels.len() == 3, "the next unit is free");
    }

    /// A fragmented memory refuses nothing: the reviewer's case, four small levels with a hole each side of
    /// one, then a level that is larger than either hole and smaller than the free total. The hardware has no
    /// allocator to refuse with, so the load goes where its address says and overwrites what is there.
    #[test]
    fn a_load_that_fits_no_hole_is_not_refused() {
        // A bank of 16 units.
        let store = TextureStore::with_capacity_bytes(16 * 64);
        assert_eq!(store.bank_units(), 16);
        let (mut c, mut n) = (TexContext::default(), TexCounts::default());
        // 4 x 4 texels of 16 bits are one unit: levels at units 0, 2, 4 and 6, with a free unit between.
        for u in [0, 2, 4, 6] {
            assert!(load_whole(&mut c, &store, &whole(2, 2, u), &mut n));
        }
        // Three units from unit 1 reach over the levels at 2 and 4 and the holes beside them.
        assert_eq!(units(8, 8, 0), 4);
        assert!(load_whole(&mut c, &store, &whole(3, 3, 1), &mut n));
        assert_eq!(c.levels.keys().copied().collect::<std::collections::BTreeSet<_>>(), [(0, 0), (0, 1), (0, 6)].into(), "the levels at 2 and 4 were overwritten");
        assert_eq!((n.loads, n.full), (5, 0));
        // Only an address that reaches past the bank is refused, and nothing is touched.
        assert!(!load_whole(&mut c, &store, &whole(3, 3, 13), &mut n));
        assert_eq!((n.full, c.levels.len()), (1, 3));
        assert!(load_whole(&mut c, &store, &whole(3, 3, 12), &mut n));
    }

    /// Any run of loads whose addresses lie inside the bank is accepted, however memory fills: a random
    /// stress over many sizes and addresses with no refusal, the levels in the map never overlapping and
    /// the newest always readable where its address puts it.
    #[test]
    fn random_loads_inside_the_bank_are_never_refused() {
        let bank = 256u32;
        let store = TextureStore::with_capacity_bytes(bank as u64 * 64);
        let (mut c, mut n) = (TexContext::default(), TexCounts::default());
        let mut x = 0x2545_f491_4f6c_dd1du64;
        let mut next = move |m: u64| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x % m
        };
        for step in 0..20_000 {
            let (se, te) = (next(5) as u32, next(5) as u32);
            let need = units(1 << se, 1 << te, 0) as u32;
            let address = next((bank - need + 1) as u64) as u32;
            let bank_bit = next(2) as u8;
            let mut a = whole(se, te, address);
            a[17] |= (bank_bit as u32) << 3;
            a[18] = 0;
            a[18 + bank_bit as usize] = address;
            let (w, h) = (a[1] as usize, a[2] as usize);
            assert!(c.load(&store, &a, &vec![(step & 0xff) as u8; w * h * 4], false, &mut n), "step {step}: {se} {te} at {address}");
            let texel = (step as u32 & 0xff) * 0x0101_0101;
            assert_eq!(word(&c, &store, bank_bit, address, 0), texel, "step {step}");
            let mut spans: Vec<(u8, u64, u64)> = c.levels.iter().map(|(&(b, at), l)| (b, at as u64, at as u64 + units(l.pw, l.ph * l.pd, l.size_code))).collect();
            spans.sort();
            assert!(spans.windows(2).all(|p| p[0].0 != p[1].0 || p[0].2 <= p[1].1), "step {step}: levels overlap");
            assert!(spans.iter().all(|s| s.2 <= bank as u64), "step {step}: a level reaches past the bank");
        }
        assert_eq!(n.full, 0);
        assert!(store.used().iter().all(|&(_, pages)| pages <= store.plane_pages()));
    }

    /// A load whose address and extent reach past the bank is refused and counted, as the model's own
    /// limit: nothing already loaded is touched, and a reload of a level that is there always fits. Giving a
    /// context's texture memory back empties it.
    #[test]
    fn loads_past_the_bank_are_refused_and_a_released_context_is_empty() {
        let store = TextureStore::with_capacity_bytes(0x300 * 64);
        let (mut c, mut n) = (TexContext::default(), TexCounts::default());
        assert!(load_whole(&mut c, &store, &whole(3, 2, 0x100), &mut n));
        assert!(load_whole(&mut c, &store, &whole(3, 2, 0x2fe), &mut n));
        assert!(!load_whole(&mut c, &store, &whole(3, 2, 0x2ff), &mut n), "two units from 0x2ff end at 0x301");
        assert_eq!((n.loads, n.full, c.levels.len()), (2, 1, 2));
        // A level of another size at an address in use is replaced only if the new one lies inside the bank.
        assert!(!load_whole(&mut c, &store, &whole(7, 4, 0x2fe), &mut n));
        assert_eq!((n.full, c.levels.len()), (2, 2), "refused: the old level stays");
        assert!(load_whole(&mut c, &store, &whole(3, 2, 0x2fe), &mut n), "the same size at the same place is a reload");
        assert!(c.holds());
        c.release(&store);
        assert_eq!((c.holds(), store.used(), store.resident_pages()), (false, vec![], 0));
        assert!(load_whole(&mut c, &store, &whole(3, 2, 0x2ff - 1), &mut n));
    }

    /// One texel at the top of a bank holds one page of host memory, not the bank: a plane is sparse. A tiny load
    /// into a level of the largest shape does the same.
    #[test]
    fn one_texel_at_the_top_of_a_bank_holds_one_page() {
        let store = TextureStore::default();
        let (mut c, mut n) = (TexContext::new(3), TexCounts::default());
        let top = store.bank_units() - 1;
        assert!(load_whole(&mut c, &store, &whole(0, 0, top), &mut n));
        assert_eq!((store.resident_pages(), word(&c, &store, 0, top, 0), n.full), (1, 0x8080_8080, 0));
        // A level of 4096 x 4096 texels fills the bank and retires the texel at its top; a 1 x 1 load into it
        // holds the page of its corner, not 64 MiB.
        let mut a = copy(1, 1, 0, 0, 0);
        a[16] = 0xcc;
        assert!(c.load(&store, &a, &[0x80; 4], false, &mut n));
        assert_eq!((store.resident_pages(), word(&c, &store, 0, top, 0), word(&c, &store, 0, 0, 0), n.full), (1, 0, 0x8080_8080, 0));
    }

    /// Many contexts that load at high addresses stay within the product's pool: the loads that would pass it are
    /// refused and counted, and what the store holds never passes the ceiling.
    #[test]
    fn many_contexts_loading_high_stay_within_the_pool() {
        // 4 MiB: a bank of 64 pages, 128 pages in all.
        let store = TextureStore::with_capacity_bytes(4 << 20);
        let top = store.bank_units() - 1;
        let (mut n, mut asked) = (TexCounts::default(), 0);
        let mut ctxs: Vec<TexContext> = (0..=crate::dev::kona::defs::CONTEXT_SLOTS).map(TexContext::new).collect();
        for c in ctxs.iter_mut() {
            for bank in 0..2u32 {
                let mut a = whole(0, 0, top);
                a[17] |= bank << 3;
                a[18] = 0;
                a[18 + bank as usize] = top;
                asked += 1;
                c.load(&store, &a, &[0x80; 4], false, &mut n);
                assert!(store.resident_pages() <= store.cap_pages());
            }
        }
        assert_eq!((store.resident_pages(), n.loads + n.full, n.loads), (store.cap_pages(), asked, store.cap_pages() as u64));
        assert!(n.full > 0);
    }

    /// A load past the pool is refused and counted, and the levels already there are left alone, including when the
    /// refused load would have retired one. A reload of what is held still fits.
    #[test]
    fn a_load_past_the_pool_is_refused_and_leaves_the_old_levels() {
        // A bank of 3 pages and a pool of 6.
        let store = TextureStore::with_capacity_bytes(12 * PAGE_WORDS as u64);
        assert_eq!((store.plane_pages(), store.cap_pages()), (3, 6));
        let (mut a, mut b, mut n) = (TexContext::new(1), TexContext::new(2), TexCounts::default());
        let page_units = PAGE_WORDS as u32 / UNIT_WORDS;
        // `b` takes five pages: bank 0's three and bank 1's first two.
        for (bank, pg) in [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)] {
            let mut l = whole(0, 0, pg * page_units);
            l[17] |= bank << 3;
            l[18] = 0;
            l[18 + bank as usize] = pg * page_units;
            assert!(b.load(&store, &l, &[0x80; 4], false, &mut n));
        }
        // `a` takes the sixth, with a 1 x 1 level at the start of bank 0.
        assert!(load_whole(&mut a, &store, &whole(0, 0, 0), &mut n));
        assert_eq!((store.resident_pages(), n.full), (6, 0));
        // A 256 x 128 level at the same address is another shape: it would retire the 1 x 1 and need bank 0's second page.
        assert!(!load_whole(&mut a, &store, &whole(8, 7, 0), &mut n));
        assert_eq!((n.full, a.levels.len(), store.resident_pages(), word(&a, &store, 0, 0, 0)), (1, 1, 6, 0x8080_8080), "refused: the 1 x 1 level and its page stay");
        assert!(a.levels.values().all(|l| l.pw == 1));
        // The same shape at the same place is a reload: its page is held.
        assert!(load_whole(&mut a, &store, &whole(0, 0, 0), &mut n));
        assert_eq!(n.full, 1);
    }

    /// Two contexts that use the same addresses do not share a word: each has planes of its own.
    #[test]
    fn contexts_with_the_same_addresses_do_not_alias() {
        let store = TextureStore::default();
        let (mut a, mut b, mut n) = (TexContext::new(1), TexContext::new(2), TexCounts::default());
        let l = whole(1, 1, 0x40);
        assert!(a.load(&store, &l, &[0x11; 16], false, &mut n));
        assert!(b.load(&store, &l, &[0x22; 16], false, &mut n));
        assert_eq!((word(&a, &store, 0, 0x40, 0), word(&b, &store, 0, 0x40, 0)), (0x1111_1111, 0x2222_2222));
        a.release(&store);
        assert_eq!((word(&a, &store, 0, 0x40, 0), word(&b, &store, 0, 0x40, 0)), (0, 0x2222_2222), "releasing one leaves the other");
    }

    /// The client sends a level smaller than the tile as one copy per tile cell of its size, and the
    /// last copy gives the origin: a 1 x 1 level of an 8 x 4 tile is at (7, 3), a 4 x 4 level at (4, 0),
    /// and an 8 x 1 level (which fills the width) at (0, 3). The box of every copy is not the level.
    /// With the bind's logical size or without it, the read-back's case, the placement is the same.
    #[test]
    fn the_last_copy_of_a_small_level_gives_its_origin() {
        let store = TextureStore::default();
        let mut c = TexContext::default();
        for y in 0..4 {
            for x in 0..8 {
                load_copy(&mut c, &store, &copy(1, 1, x, y, 0x100));
            }
        }
        assert_eq!(c.levels[&(0, 0x100)].place([None, None]), [(7, 1), (3, 1)]);
        assert_eq!(c.levels[&(0, 0x100)].place([Some(1), Some(1)]), [(7, 1), (3, 1)]);
        for x in [0, 4] {
            load_copy(&mut c, &store, &copy(4, 4, x, 0, 0x200));
        }
        assert_eq!(c.levels[&(0, 0x200)].place([None, None]), [(4, 4), (0, 4)]);
        assert_eq!(c.levels[&(0, 0x200)].place([Some(4), Some(4)]), [(4, 4), (0, 4)]);
        for y in 0..4 {
            load_copy(&mut c, &store, &copy(8, 1, 0, y, 0x300));
        }
        assert_eq!(c.levels[&(0, 0x300)].place([None, None]), [(0, 8), (3, 1)]);
        assert_eq!(c.levels[&(0, 0x300)].place([Some(8), Some(1)]), [(0, 8), (3, 1)]);
    }

    /// A level that fills the tile is sent once at the origin, and an update of part of it later does
    /// not make the level that part: an 8 x 4 level, then 4 x 2 at (2, 1), is still (0, 0) to (7, 3).
    #[test]
    fn a_partial_update_does_not_shrink_a_level() {
        let store = TextureStore::default();
        let mut c = TexContext::default();
        load_copy(&mut c, &store, &copy(8, 4, 0, 0, 0x100));
        assert_eq!(c.levels[&(0, 0x100)].place([None, None]), [(0, 8), (0, 4)]);
        load_copy(&mut c, &store, &copy(4, 2, 2, 1, 0x100));
        let l = c.levels[&(0, 0x100)];
        assert_eq!((l.min, l.max), ((0, 0), (7, 3)));
        assert_eq!(l.place([None, None]), [(0, 8), (0, 4)], "without a bind");
        assert_eq!(l.place([Some(8), Some(4)]), [(0, 8), (0, 4)], "with the bind's size");
    }

    /// A read-back names no bind, so it takes a level's placement from its loads. A level whose first load was the
    /// whole rectangle on an axis is never a run of copies there: an update ending at the edge (the right half, the
    /// bottom half) leaves it alone. A small level's last copy still gives its origin.
    #[test]
    fn an_update_that_ends_at_the_edge_does_not_move_a_level_read_back_without_a_bind() {
        let store = TextureStore::default();
        let mut c = TexContext::default();
        load_copy(&mut c, &store, &copy(8, 4, 0, 0, 0x100));
        load_copy(&mut c, &store, &copy(4, 4, 4, 0, 0x100));
        assert_eq!(c.levels[&(0, 0x100)].place([None, None]), [(0, 8), (0, 4)], "the right half");
        load_copy(&mut c, &store, &copy(8, 2, 0, 2, 0x100));
        assert_eq!(c.levels[&(0, 0x100)].place([None, None]), [(0, 8), (0, 4)], "the bottom half");
        // A level first sent as copies is small on that axis: 2 x 4 copies at x = 0, 2, 4, 6 of a 2 x 4 level.
        for x in [0, 2, 4, 6] {
            load_copy(&mut c, &store, &copy(2, 4, x, 0, 0x200));
        }
        assert_eq!(c.levels[&(0, 0x200)].place([None, None]), [(6, 2), (0, 4)]);
    }

    /// The read-back's guess needs both the edge and the alignment of a run of copies. A level first sent in part
    /// whose later load starts off a multiple of its width, or stops short of the edge, is the box that was loaded.
    #[test]
    fn the_read_back_guess_wants_the_edge_and_the_alignment() {
        let store = TextureStore::default();
        let mut c = TexContext::default();
        // 2 texels at the origin, then 3 ending at the edge but starting at 5 (not a multiple of 3).
        load_copy(&mut c, &store, &copy(2, 4, 0, 0, 0x100));
        load_copy(&mut c, &store, &copy(3, 4, 5, 0, 0x100));
        assert_eq!(c.levels[&(0, 0x100)].place([None, None])[0], (0, 8), "off the alignment: the box");
        // 2 texels at 0, then 2 at 2: aligned, but short of the edge.
        load_copy(&mut c, &store, &copy(2, 4, 0, 0, 0x200));
        load_copy(&mut c, &store, &copy(2, 4, 2, 0, 0x200));
        assert_eq!(c.levels[&(0, 0x200)].place([None, None])[0], (0, 4), "short of the edge: the box");
        // And the same two conditions met are the copy.
        load_copy(&mut c, &store, &copy(2, 4, 0, 0, 0x300));
        load_copy(&mut c, &store, &copy(2, 4, 6, 0, 0x300));
        assert_eq!(c.levels[&(0, 0x300)].place([None, None])[0], (6, 2));
    }

    /// The bind's size is halved for each level down: a four-level mip chain of an 8 x 4 texture is 8 x 4, 4 x 2, 2 x 1
    /// and 1 x 1, each sent as copies and each at its last copy's origin. The levels alternate banks.
    #[test]
    fn a_bind_halves_its_logical_size_per_level() {
        let store = TextureStore::default();
        let mut c = TexContext::default();
        let in_bank_1 = |mut a: [u32; 20]| {
            a[17] |= 8;
            a[19] = a[18];
            a[18] = 0;
            a
        };
        load_copy(&mut c, &store, &copy(8, 4, 0, 0, 0x100));
        for y in [0, 2] {
            for x in [0, 4] {
                load_copy(&mut c, &store, &in_bank_1(copy(4, 2, x, y, 0x200)));
            }
        }
        for y in 0..4 {
            for x in [0, 2, 4, 6] {
                load_copy(&mut c, &store, &copy(2, 1, x, y, 0x300));
            }
        }
        for y in 0..4 {
            for x in 0..8 {
                load_copy(&mut c, &store, &in_bank_1(copy(1, 1, x, y, 0x400)));
            }
        }
        let mut rec = [0u32; 22];
        rec[6..11].copy_from_slice(&[0x100, 0x200, 0x300, 0x400, 0xffff_ffff]);
        // Level 0 is 8 x 4 (size nibbles 3 and 2), in bank 0; levels 1 and 3 are in bank 1.
        c.short_bind(&[0xffff_ffff, 2 << 12, 0, 0x23, 1]);
        let d = c.record(&store, &rec, Default::default(), &mut TexCounts::default()).unwrap();
        let got: Vec<_> = d.levels[..d.count as usize].iter().map(|l| (l.ox, l.oy, l.w, l.h)).collect();
        assert_eq!(got, vec![(0, 0, 8, 4), (4, 2, 4, 2), (6, 3, 2, 1), (7, 3, 1, 1)]);
    }

    /// A bind says a level is the size of the tile, so a level whose first load was only part of it is the whole
    /// rectangle from the origin, not the box that happened to be loaded.
    #[test]
    fn a_bind_takes_the_logical_size_over_the_loaded_box() {
        let store = TextureStore::default();
        let mut c = TexContext::default();
        load_copy(&mut c, &store, &copy(4, 2, 0, 0, 0x100));
        assert_eq!(c.levels[&(0, 0x100)].place([None, None]), [(0, 4), (0, 2)], "without a bind, the box");
        let mut rec = [0u32; 22];
        rec[6..8].copy_from_slice(&[0x100, 0xffff_ffff]);
        c.short_bind(&[0xffff_ffff, 2 << 12, 0, 0x23, 1]);
        let d = c.record(&store, &rec, Default::default(), &mut TexCounts::default()).unwrap();
        assert_eq!([d.levels[0].ox, d.levels[0].oy, d.levels[0].w, d.levels[0].h], [0, 0, 8, 4]);
    }

    /// The words of a read: the level's texels as the 16-bit unsigned shorts a luminance read under the one-component
    /// storage code returns.
    fn read_words(c: &TexContext, store: &TextureStore, address: u32) -> Vec<u32> {
        let h = ReadHeader::parse(&[0x0de1, 8, 4, 6, 3, 0xd, 0, 0, 0, 0, 7, 3, 0, 0x0385_0030, 0x23, 0x44, address, 0]).unwrap();
        c.read_level(store, &h).expect("the level is loaded").iter().map(|t| (t[0] * 65535.0 + 0.5) as u32).collect()
    }

    /// 8 x 4 RGBA bytes: red, green, blue and alpha all different from texel to texel.
    fn varied_bytes() -> Vec<u8> {
        (0..32u32).flat_map(|i| [(i * 8) as u8, 255 - (i * 8) as u8, (i * 37 % 256) as u8, if i % 3 == 0 { 255 } else { 0 }]).collect()
    }

    #[test]
    fn a_word_and_its_float_and_a_texel_and_its_word_round_trip() {
        for w in 0..=0xffffu32 {
            assert_eq!((word_as_float(w) * 65535.0 + 0.5) as u32, w);
            for storage in [0x10, 0x14, 0x18] {
                assert_eq!(pack16(storage, unpack16(storage, w)), Some(w), "{storage:#x} {w:#x}");
            }
        }
        assert_eq!(pack16(0x30, 0xffff_ffff), None, "a 32-bit texel has no 16-bit word");
        assert_eq!(unpack16(0x14, 0xf801), 0xffff_0000, "red 31, alpha 1");
        assert_eq!(unpack16(0x10, 0x0f00), 0x00_00_ff_00 | 0 << 24 | 0 << 16, "green 15: alpha 0");
        assert_eq!(unpack16(0xd, 0x8000), 0xff80_8080, "any other code is luminance, as an upload of unsigned shorts is");
    }

    /// The library reads a level back as luminance in unsigned shorts under the one-component code and reloads the
    /// bytes it got, then binds under the format the level was made with. The read returns the packed 16-bit words of
    /// a level loaded as 5-5-5-1 (or 4-4-4-4), the reload keeps them, and the bind samples them as the level was.
    #[test]
    fn a_level_read_back_under_the_one_component_code_and_reloaded_is_sampled_as_it_was() {
        for storage in [0x14u32, 0x10] {
            let store = TextureStore::default();
            let (mut c, mut n) = (TexContext::default(), TexCounts::default());
            let mut a = copy(8, 4, 0, 0, 0x100);
            a[9] = storage;
            assert!(c.load(&store, &a, &varied_bytes(), false, &mut n));
            let mut rec = [0u32; 22];
            rec[6..8].copy_from_slice(&[0x100, 0xffff_ffff]);
            c.short_bind(&[0xffff_ffff, 2 << 12, storage << 14, 0x23, 1]);
            let lv = c.record(&store, &rec, Default::default(), &mut TexCounts::default()).unwrap().levels[0];
            let before: Vec<u32> = (0..32).map(|i| store.get(lv.plane, lv.off + (i / 8) * 8 + i % 8)).collect();
            let words = read_words(&c, &store, 0x100);
            // The words are the quantised components in the lanes: texel 1 is red 8, green 247, blue 37, alpha 0.
            let q = |v: u32, bits: u32| (v * ((1 << bits) - 1) + 127) / 255;
            let expect = if storage == 0x14 { q(8, 5) << 11 | q(247, 5) << 6 | q(37, 5) << 1 } else { q(8, 4) << 12 | q(247, 4) << 8 | q(37, 4) << 4 };
            assert_eq!(words[1], expect, "{storage:#x}");
            // Free and reload the bytes it returned, as 16-bit luminance.
            let bytes: Vec<u8> = words.iter().flat_map(|w| (*w as u16).to_be_bytes()).collect();
            let mut l = copy(8, 4, 0, 0, 0x100);
            l[3] = 6;
            l[4] = 3;
            l[9] = 0xd;
            assert!(c.load(&store, &l, &bytes, false, &mut n));
            assert_eq!(read_words(&c, &store, 0x100), words, "a level reloaded as raw words reads back as it was loaded");
            // A bind under the original format samples the original texels, whatever storage loaded the words.
            c.short_bind(&[0xffff_ffff, 2 << 12, storage << 14, 0x23, 1]);
            let lv = c.record(&store, &rec, Default::default(), &mut TexCounts::default()).unwrap().levels[0];
            let after: Vec<u32> = (0..32).map(|i| store.get(lv.plane, lv.off + (i / 8) * 8 + i % 8)).collect();
            assert_eq!(after, before, "{storage:#x}");
            assert_eq!(read_words(&c, &store, 0x100), words, "and again after the bind has sampled them");
        }
    }

    /// A bind under another format samples the same words as that format: the reloaded 16-bit words bound as
    /// luminance are grey, and bound as 4-4-4-4 are not what they were as 5-5-5-1. No bind drops a level on its code.
    #[test]
    fn a_bind_samples_the_raw_words_under_its_own_format() {
        let store = TextureStore::default();
        let (mut c, mut n) = (TexContext::default(), TexCounts::default());
        let mut l = copy(8, 4, 0, 0, 0x100);
        l[3] = 6;
        l[4] = 3;
        l[9] = 0xd;
        let bytes: Vec<u8> = (0..32u32).flat_map(|i| ((i * 2000) as u16).to_be_bytes()).collect();
        assert!(c.load(&store, &l, &bytes, false, &mut n));
        assert_eq!(read_words(&c, &store, 0x100)[3], 6000);
        let mut rec = [0u32; 22];
        rec[6..8].copy_from_slice(&[0x100, 0xffff_ffff]);
        for (format, texel3) in [(0x0du32, unpack16(0xd, 6000)), (0x10, unpack16(0x10, 6000))] {
            let mut c2 = TexContext::default();
            assert!(c2.load(&store, &l, &bytes, false, &mut n));
            c2.short_bind(&[0xffff_ffff, 2 << 12, format << 14, 0x23, 1]);
            let lv = c2.record(&store, &rec, Default::default(), &mut TexCounts::default()).expect("found by bank and address, not by storage").levels[0];
            assert_eq!(store.get(lv.plane, lv.off + 3), texel3, "{format:#x}");
        }
    }

    /// A load that covers what a level holds replaces it, raw or not. One that does not cannot mix raw words with
    /// decoded texels: it is decoded as one-component texels, and a raw level is decoded first.
    #[test]
    fn a_partial_load_does_not_mix_raw_words_and_decoded_texels() {
        let store = TextureStore::default();
        let (mut c, mut n) = (TexContext::default(), TexCounts::default());
        let mut raw = copy(8, 4, 0, 0, 0x100);
        raw[3] = 6;
        raw[4] = 3;
        raw[9] = 0xd;
        let words: Vec<u8> = (0..32u32).flat_map(|_| 0x8000u16.to_be_bytes()).collect();
        assert!(c.load(&store, &raw, &words, false, &mut n));
        assert!(c.levels[&(0, 0x100)].raw);
        // An RGBA update of one corner: the raw words around it become luminance texels first.
        let mut part = copy(2, 1, 0, 0, 0x100);
        part[9] = 0x30;
        assert!(c.load(&store, &part, &[1, 2, 3, 255, 4, 5, 6, 255], false, &mut n));
        let l = c.levels[&(0, 0x100)];
        assert!(!l.raw);
        assert_eq!((store.get(l.plane, l.off), store.get(l.plane, l.off + 2)), (0xff01_0203, 0xff80_8080));
        // A partial raw load into a decoded level is decoded as luminance too.
        assert!(c.load(&store, &{ let mut p = raw; p[1] = 2; p[2] = 1; p[13] = 1; p[14] = 0; p }, &[0xff, 0xff, 0, 0], false, &mut n));
        assert!(!c.levels[&(0, 0x100)].raw);
        assert_eq!(store.get(l.plane, l.off), 0xffff_ffff);
        // A raw load of everything it holds replaces it as raw again.
        assert!(c.load(&store, &raw, &words, false, &mut n));
        assert!(c.levels[&(0, 0x100)].raw);
    }

    /// The levels, the detail images and the unpack state are what a snapshot keeps: restored, the detail map has
    /// its images and every level places as it did, and saving the restored state gives the same bytes.
    #[test]
    fn a_snapshot_keeps_the_levels_and_the_detail_images() {
        use crate::dev::kona::stream::{Reader, Writer};
        let store = TextureStore::default();
        let (mut c, mut n) = (TexContext::default(), TexCounts::default());
        c.unpack.skip_rows = 3;
        load_copy(&mut c, &store, &copy(8, 4, 0, 0, 0x100));
        load_copy(&mut c, &store, &copy(4, 4, 4, 0, 0x100));
        for x in 0..8 {
            load_copy(&mut c, &store, &copy(1, 4, x, 0, 0x200));
        }
        for bank in [0u32, 1] {
            let mut d = copy(8, 4, 0, 0, 0x300);
            d[0] = 0x8095;
            d[17] |= bank << 3;
            d[18 + bank as usize] = 0x300;
            assert!(c.load(&store, &d, &vec![0x77; 8 * 4 * 4], false, &mut n));
        }
        assert!(slice(&mut c, &store, &mut n, 0, 0x40, 1, 0x1234_5678));
        assert!(c.filter_table(2, &[0xabcd; 33]) && c.bind_detail(&[1, 3, 2, 0x31, 0x300, 0x301]));
        assert_eq!((c.levels.len(), c.detail.len()), (3, 2));
        let mut w = Writer::default();
        c.visit(&mut w);
        let mut back = TexContext::default();
        let mut r = Reader::new(&w.out);
        back.visit(&mut r);
        r.finish().unwrap();
        assert_eq!((back.levels.len(), back.detail.len(), back.unpack.skip_rows), (3, 2, 3));
        assert_eq!(back.levels[&(0, 0x40)].pd, 2, "a volume level keeps its depth");
        assert_eq!((back.filters.as_slice(), back.detail_bind), (&[(2, [0xabcd; 33])][..], c.detail_bind), "the filter tables and the detail bind");
        for key in [(0, 0x100), (0, 0x200)] {
            assert_eq!(back.levels[&key].place([None, None]), c.levels[&key].place([None, None]), "{key:?}");
        }
        assert_eq!(back.levels[&(0, 0x100)].place([None, None]), [(0, 8), (0, 4)], "the whole-first-load bit survived");
        assert_eq!(back.levels[&(0, 0x200)].place([None, None]), [(7, 1), (0, 4)]);
        assert_eq!(back.detail.keys().collect::<std::collections::BTreeSet<_>>(), c.detail.keys().collect());
        let mut again = Writer::default();
        back.visit(&mut again);
        assert_eq!(again.out, w.out);
    }

    /// A level address counts 256 bits a bank, so a rectangle of 32 texels takes 2 units at
    /// 16 bits, 4 at 32 and 8 at the 64-bit slot.
    #[test]
    fn a_levels_extent_is_its_texels_in_units_of_256_bits() {
        assert_eq!([units(8, 4, 0), units(8, 4, 1), units(8, 4, 2)], [2, 4, 8]);
        assert_eq!(units(1, 1, 0), 1, "a part unit still takes the unit");
        assert_eq!(parse(&{ let mut a = header(1, 1); a[17] |= 2; a }).size_code(), 2);
    }

    /// Two levels of one bank cannot share a unit: loading at an address inside another level's
    /// extent retires that level; a level in the other bank, or past the extent, stays.
    #[test]
    fn a_load_retires_the_levels_it_overlaps_in_its_bank() {
        let store = TextureStore::default();
        let mut c = TexContext::default();
        // 8 x 4 at 32 bits is 4 units: [0x100, 0x104).
        let mut a = copy(8, 4, 0, 0, 0x100);
        a[17] |= 1;
        load_copy(&mut c, &store, &a);
        let mut other_bank = copy(8, 4, 0, 0, 0x101);
        other_bank[17] |= 8;
        other_bank[19] = 0x101;
        load_copy(&mut c, &store, &other_bank);
        load_copy(&mut c, &store, &copy(8, 4, 0, 0, 0x104));
        assert_eq!(c.levels.len(), 3, "past the extent, and in the other bank, nothing is retired");
        load_copy(&mut c, &store, &copy(8, 4, 0, 0, 0x103));
        let mut keys: Vec<_> = c.levels.keys().copied().collect();
        keys.sort();
        assert_eq!(keys, vec![(0, 0x103), (1, 0x101)], "the 32-bit level and the one after it were overwritten");
    }

    #[test]
    fn the_unpack_state_keeps_the_names_the_board_keeps() {
        let mut u = Unpack::default();
        u.set(0xcf2, 8);
        u.set(0xcf3, 2);
        u.set(0xcf4, 3);
        u.set(0xcf5, 8);
        u.set(0xcf0, 1);
        assert_eq!(u, Unpack { row_length: 8, skip_rows: 2, skip_pixels: 3, alignment: 8, swap_bytes: true });
        u.set(0xcf5, 3);
        assert_eq!(u.alignment, 8, "an alignment that is not a power of two up to 8 is refused");
        u.set(0xd05, 1);
        assert_eq!(u.alignment, 8, "the pack side is not kept");
    }
}
