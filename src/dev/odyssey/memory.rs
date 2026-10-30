//! SGI graphics board memory and the planes over it. A plane is a typed
//! window (format, size, stride, packing, row origin) over one head's
//! memory; pixel (x, y) is at byte `base + y * stride + x * bpp`. Planes
//! may overlap. Out-of-plane loads read zero and out-of-plane stores are
//! dropped.
//!
//! Ported from include/hw/display/sgi/board-memory.h in jamesbraid/qemu-sgi-origami.
use super::*;

use std::cell::Cell;

pub const SGI_PLANES_MAX: usize = 32;

pub type PlaneFormat = u32;
pub const SGI_PLANE_CI8: PlaneFormat = 0; // 1 byte: colour index
pub const SGI_PLANE_RGB12: PlaneFormat = 1; // 2 bytes: 12-bit colour in a 16-bit word
pub const SGI_PLANE_RGB24: PlaneFormat = 2; // 4 bytes: 24-bit colour in a 32-bit word
pub const SGI_PLANE_RGBA8: PlaneFormat = 3; // 4 bytes
pub const SGI_PLANE_Z24: PlaneFormat = 4; // 4 bytes: 24-bit depth in a 32-bit word
pub const SGI_PLANE_S8: PlaneFormat = 5; // 1 byte: stencil
pub const SGI_PLANE_TAG: PlaneFormat = 6; // 1 byte: per-pixel tag
pub const SGI_PLANE_OVERLAY8: PlaneFormat = 7; // 1 byte
pub const SGI_PLANE_TEXTURE: PlaneFormat = 8; // 4 bytes
pub const SGI_PLANE_RAW: PlaneFormat = 9; // 4 bytes: a word the family interprets

pub type Packing = u32;
pub const SGI_PACK_LE_RED_LOW: Packing = 0;
pub const SGI_PACK_LE_RED_HIGH: Packing = 1;
pub const SGI_PACK_BE_RED_HIGH: Packing = 3;

#[derive(Clone, Copy, Default, Debug)]
pub struct Plane {
    pub id: u32, // index in the board's plane table
    pub base: u64,
    pub len: u64, // bytes within the board memory
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub format: PlaneFormat,
    pub packing: Packing, // what the model writes, labelled
    pub bottom_origin: bool, // plane row 0 is the bottom display line
}

/// A head's board memory: a model-owned byte array, and the planes over it.
#[derive(Clone, Default)]
pub struct BoardMemory {
    pub size: u64,
    pub bytes: Vec<u8>,
    /// A fixed table, so a plane index stays valid for the memory's life.
    pub planes: [Plane; SGI_PLANES_MAX],
    pub nplanes: u32,
}

/// Up to 16 inclusive rectangles, in plane coordinates. With n 0 the list
/// is no clip, unless empty is set: then it keeps no pixel.
pub const SGI_CLIP_MAX: usize = 16;

#[derive(Clone, Copy, Default, Debug)]
pub struct ClipRect {
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
}

#[derive(Clone, Copy, Default, Debug)]
pub struct ClipList {
    pub n: u32,
    pub empty: u32, // 0 or 1; with n 0, 1 keeps nothing
    pub r: [ClipRect; SGI_CLIP_MAX],
}

// The sixteen logic ops, in OpenGL's order (GL_CLEAR + op).
pub const SGI_LOGIC_CLEAR: u32 = 0;
pub const SGI_LOGIC_AND: u32 = 1;
pub const SGI_LOGIC_AND_REVERSE: u32 = 2;
pub const SGI_LOGIC_COPY: u32 = 3;
pub const SGI_LOGIC_AND_INVERTED: u32 = 4;
pub const SGI_LOGIC_NOOP: u32 = 5;
pub const SGI_LOGIC_XOR: u32 = 6;
pub const SGI_LOGIC_OR: u32 = 7;
pub const SGI_LOGIC_NOR: u32 = 8;
pub const SGI_LOGIC_EQUIV: u32 = 9;
pub const SGI_LOGIC_INVERT: u32 = 10;
pub const SGI_LOGIC_OR_REVERSE: u32 = 11;
pub const SGI_LOGIC_COPY_INVERTED: u32 = 12;
pub const SGI_LOGIC_OR_INVERTED: u32 = 13;
pub const SGI_LOGIC_NAND: u32 = 14;

impl BoardMemory {
    /// A zeroed memory of size bytes, at most 4 GiB - 1.
    pub fn init(&mut self, size: u64) {
        // A family constant; the limit is the 32-bit offsets planes use.
        assert!(size != 0 && size <= u32::MAX as u64);
        *self = BoardMemory::default();
        self.size = size;
        self.bytes = vec![0u8; size as usize];
    }
}

pub fn plane_bytes_per_pixel(format: PlaneFormat) -> u32 {
    plane_bpp(format)
}

impl BoardMemory {
    /// Declare a plane. Its id is assigned here and returned; the rest of
    /// *plane is copied. Fails when the plane does not fit the memory, its
    /// rows do not fit its stride and length, or the table already holds
    /// SGI_PLANES_MAX planes.
    pub fn add_plane(&mut self, plane: &Plane) -> Result<u32, String> {
        let bpp = plane_bytes_per_pixel(plane.format);
        let row = plane.width as u64 * bpp as u64;

        if plane.width == 0 || plane.height == 0 {
            return Err("plane has no pixels".to_string());
        }
        if (plane.stride as u64) < row {
            return Err(format!(
                "plane stride {} is shorter than its {}-byte rows",
                plane.stride, row
            ));
        }
        let need = (plane.height as u64 - 1) * plane.stride as u64 + row;
        if need > plane.len {
            return Err(format!("plane needs {} bytes but has {}", need, plane.len));
        }
        if plane.base > self.size || plane.len > self.size - plane.base {
            return Err(format!(
                "plane {:#x}+{:#x} is outside {:#x} bytes of board memory",
                plane.base, plane.len, self.size
            ));
        }

        if self.nplanes as usize == SGI_PLANES_MAX {
            return Err(format!("board memory already has {} planes", SGI_PLANES_MAX));
        }
        let id = self.nplanes;
        self.nplanes += 1;
        let p = &mut self.planes[id as usize];
        *p = *plane;
        p.id = id;
        crate::dlog!(LogModule::Odyssey, "sgi_plane_add: plane {} base {:#x} len {:#x} {}x{} stride {} format {}", p.id, p.base, p.len, p.width, p.height, p.stride, p.format as i32);
        Ok(p.id)
    }
    /// The plane with this id, if there is one.
    pub fn plane(&self, id: u32) -> Option<Plane> {
        (id < self.nplanes).then(|| self.planes[id as usize])
    }
}

impl Plane {
    #[inline]
    pub fn contains(&self, x: i32, y: i32) -> bool {
        x >= 0 && y >= 0 && (x as u32) < self.width && (y as u32) < self.height
    }
}

impl ClipList {
    pub fn contains(&self, x: i32, y: i32) -> bool {
        if self.n == 0 {
            return self.empty == 0;
        }
        for i in 0..self.n.min(SGI_CLIP_MAX as u32) {
            let r = &self.r[i as usize];

            if x >= r.x0 && x <= r.x1 && y >= r.y0 && y <= r.y1 {
                return true;
            }
        }
        false
    }
}
pub fn logic_op(op: u8, src: u32, dst: u32) -> u32 {
    let s = src;
    let d = dst;
    match (op & 0xf) as u32 {
        SGI_LOGIC_CLEAR => 0,
        SGI_LOGIC_AND => s & d,
        SGI_LOGIC_AND_REVERSE => s & !d,
        SGI_LOGIC_COPY => s,
        SGI_LOGIC_AND_INVERTED => !s & d,
        SGI_LOGIC_NOOP => d,
        SGI_LOGIC_XOR => s ^ d,
        SGI_LOGIC_OR => s | d,
        SGI_LOGIC_NOR => !(s | d),
        SGI_LOGIC_EQUIV => !(s ^ d),
        SGI_LOGIC_INVERT => !d,
        SGI_LOGIC_OR_REVERSE => s | !d,
        SGI_LOGIC_COPY_INVERTED => !s,
        SGI_LOGIC_OR_INVERTED => !s | d,
        SGI_LOGIC_NAND => !(s & d),
        _ => !0u32, // SGI_LOGIC_SET
    }
}

impl BoardMemory {
    pub fn plane_load(&self, p: &Plane, x: i32, y: i32) -> u32 {
        if !p.contains(x, y) {
            return 0;
        }
        let bpp = plane_bpp(p.format);
        self.pixel(p.offset(x, y, bpp), bpp).map_or(0, |b| p.get(bpp, b))
    }
    pub fn plane_store(&mut self, p: &Plane, x: i32, y: i32, word: u32) {
        if !p.contains(x, y) {
            return;
        }
        let bpp = plane_bpp(p.format);
        if let Some(b) = self.pixel_mut(p.offset(x, y, bpp), bpp) {
            p.put(bpp, word, b);
        }
    }

    /// The `bpp` bytes at `at`, when they are inside the memory.
    #[inline]
    pub fn pixel(&self, at: u64, bpp: u32) -> Option<&[u8]> {
        self.bytes.get(usize::try_from(at).ok()?..)?.get(..bpp as usize)
    }
    #[inline]
    pub fn pixel_mut(&mut self, at: u64, bpp: u32) -> Option<&mut [u8]> {
        self.bytes.get_mut(usize::try_from(at).ok()?..)?.get_mut(..bpp as usize)
    }
}

/// One plane word of `bpp` bytes, little-endian when `le` is set.
#[inline]
pub fn plane_word_load(bytes: &[u8], bpp: u32, le: bool) -> u32 {
    match bpp {
        1 => bytes[0] as u32,
        2 => {
            if le {
                u16::from_le_bytes(bytes[..2].try_into().unwrap()) as u32
            } else {
                u16::from_be_bytes(bytes[..2].try_into().unwrap()) as u32
            }
        }
        _ => {
            if le {
                u32::from_le_bytes(bytes[..4].try_into().unwrap())
            } else {
                u32::from_be_bytes(bytes[..4].try_into().unwrap())
            }
        }
    }
}

#[inline]
pub fn plane_word_store(bytes: &mut [u8], bpp: u32, le: bool, word: u32) {
    match bpp {
        1 => bytes[0] = word as u8,
        2 => {
            if le {
                bytes[..2].copy_from_slice(&(word as u16).to_le_bytes());
            } else {
                bytes[..2].copy_from_slice(&(word as u16).to_be_bytes());
            }
        }
        _ => {
            if le {
                bytes[..4].copy_from_slice(&word.to_le_bytes());
            } else {
                bytes[..4].copy_from_slice(&word.to_be_bytes());
            }
        }
    }
}

impl Plane {
    /// A plane word to and from its stored bytes, in the plane's packing.
    pub fn decode(&self, bytes: &[u8]) -> u32 {
        self.get(plane_bpp(self.format), bytes)
    }
    pub fn encode(&self, word: u32, bytes: &mut [u8]) {
        self.put(plane_bpp(self.format), word, bytes)
    }
}

// The per-pixel paths take a plane's pixel size once and reuse it.
#[inline]
pub fn plane_bpp(format: PlaneFormat) -> u32 {
    match format {
        SGI_PLANE_CI8 | SGI_PLANE_S8 | SGI_PLANE_TAG | SGI_PLANE_OVERLAY8 => 1,
        SGI_PLANE_RGB12 => 2,
        SGI_PLANE_RGB24 | SGI_PLANE_RGBA8 | SGI_PLANE_Z24 | SGI_PLANE_TEXTURE | SGI_PLANE_RAW => 4,
        _ => unreachable!(),
    }
}

impl Plane {
    pub fn is_le(&self) -> bool {
        self.packing == SGI_PACK_LE_RED_LOW || self.packing == SGI_PACK_LE_RED_HIGH
    }

    #[inline]
    pub fn get(&self, bpp: u32, bytes: &[u8]) -> u32 {
        plane_word_load(bytes, bpp, self.is_le())
    }

    #[inline]
    pub fn put(&self, bpp: u32, word: u32, bytes: &mut [u8]) {
        plane_word_store(bytes, bpp, self.is_le(), word)
    }
}

impl Plane {
    pub fn offset(&self, x: i32, y: i32, bpp: u32) -> u64 {
        self.base
            .wrapping_add((y as u64).wrapping_mul(self.stride as u64))
            .wrapping_add((x as u64).wrapping_mul(bpp as u64))
    }
}

/// A pixel array's rows as a draw borrows them: the whole array, or the
/// rows of one band of a banded draw. A band has every bands-th run of
/// 2^SGI_SOFT_BAND_ROWS_SHIFT rows (SoftDraw::row_in_band()), so bands of
/// one draw never share a row and can write theirs on separate threads.
pub struct RowsMut<'a, T> {
    /// Elements from the start of one row to the next.
    stride: usize,
    chunks: RowChunks<&'a mut [T]>,
}

enum RowChunks<S> {
    Whole(S),
    /// Indexed by row >> SGI_SOFT_BAND_ROWS_SHIFT; None for another band's.
    Band(Vec<Option<S>>),
}

impl<'a, T> RowsMut<'a, T> {
    pub fn new(data: &'a mut [T], stride: usize) -> Self {
        RowsMut { stride, chunks: RowChunks::Whole(data) }
    }

    /// One RowsMut per band, band b holding its rows. Only whole rows split.
    pub fn split(&mut self, bands: u32) -> Vec<RowsMut<'_, T>> {
        let RowChunks::Whole(data) = &mut self.chunks else {
            panic!("a band's rows split again");
        };
        let run = (self.stride << SGI_SOFT_BAND_ROWS_SHIFT).max(1);
        let bands = bands as usize;
        let n = data.len().div_ceil(run);
        let mut out: Vec<Vec<Option<&mut [T]>>> = (0..bands).map(|_| (0..n).map(|_| None).collect()).collect();

        for (i, chunk) in data.chunks_mut(run).enumerate() {
            out[i % bands][i] = Some(chunk);
        }
        out.into_iter().map(|chunks| RowsMut { stride: self.stride, chunks: RowChunks::Band(chunks) }).collect()
    }

    /// The rows as one thread writes them.
    pub fn view(&mut self) -> Rows<'_, T> {
        let chunks = match &mut self.chunks {
            RowChunks::Whole(data) => RowChunks::Whole(Cell::from_mut(&mut **data).as_slice_of_cells()),
            RowChunks::Band(chunks) => RowChunks::Band(
                chunks.iter_mut().map(|c| c.as_mut().map(|c| Cell::from_mut(&mut **c).as_slice_of_cells())).collect(),
            ),
        };
        Rows { stride: self.stride, chunks }
    }
}

/// RowsMut as one thread writes it, through shared references.
pub struct Rows<'a, T> {
    stride: usize,
    chunks: RowChunks<&'a [Cell<T>]>,
}

impl<T> Rows<'_, T> {
    /// The len elements at element `offset` of row y, when that row is
    /// these rows' and long enough.
    #[inline]
    pub fn get(&self, y: usize, offset: usize, len: usize) -> Option<&[Cell<T>]> {
        let (data, at) = match &self.chunks {
            RowChunks::Whole(data) => (*data, y * self.stride + offset),
            RowChunks::Band(chunks) => {
                let chunk = (*chunks.get(y >> SGI_SOFT_BAND_ROWS_SHIFT)?)?;
                (chunk, (y & ((1 << SGI_SOFT_BAND_ROWS_SHIFT) - 1)) * self.stride + offset)
            }
        };
        data.get(at..)?.get(..len)
    }
}

impl Rows<'_, u8> {
    /// Plane p's word at (x, y), or 0 outside the plane or these rows.
    #[inline]
    pub fn plane_load(&self, p: &Plane, x: i32, y: i32) -> u32 {
        if !p.contains(x, y) {
            return 0;
        }
        let bpp = plane_bpp(p.format);
        let Some(cells) = self.get(y as usize, x as usize * bpp as usize, bpp as usize) else {
            return 0;
        };
        let mut bytes = [0u8; 4];
        for (b, c) in bytes.iter_mut().zip(cells) {
            *b = c.get();
        }
        p.get(bpp, &bytes)
    }

    /// Store plane p's word at (x, y), dropped outside the plane or these rows.
    #[inline]
    pub fn plane_store(&self, p: &Plane, x: i32, y: i32, word: u32) {
        if !p.contains(x, y) {
            return;
        }
        let bpp = plane_bpp(p.format);
        let Some(cells) = self.get(y as usize, x as usize * bpp as usize, bpp as usize) else {
            return;
        };
        let mut bytes = [0u8; 4];
        p.put(bpp, word, &mut bytes);
        for (c, b) in cells.iter().zip(bytes) {
            c.set(b);
        }
    }
}

impl BoardMemory {
    /// Plane p's rows, which start at its base.
    pub fn rows(&mut self, p: &Plane) -> RowsMut<'_, u8> {
        let end = (p.base + p.len).min(self.size) as usize;
        RowsMut::new(&mut self.bytes[p.base as usize..end], p.stride as usize)
    }
}
