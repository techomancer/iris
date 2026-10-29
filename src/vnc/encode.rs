//! RFB pixel formats and rectangle encodings: Raw and ZRLE.
//!
//! Frames arrive as `0xAABBGGRR` (R in the low byte), the compositors'
//! format. Our own format is 32 bpp little-endian with red at shift 0, so
//! Raw to a viewer that keeps it is a straight copy.

use flate2::{Compress, Compression, FlushCompress};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PixelFormat {
    pub bpp: u8,
    pub depth: u8,
    pub big_endian: bool,
    pub true_colour: bool,
    pub max: [u16; 3],
    pub shift: [u8; 3],
}

impl PixelFormat {
    /// What the server offers: the compositors' own layout.
    pub const NATIVE: PixelFormat = PixelFormat {
        bpp: 32, depth: 24, big_endian: false, true_colour: true,
        max: [255, 255, 255], shift: [0, 8, 16],
    };

    pub fn to_bytes(&self) -> [u8; 16] {
        let mut b = [0u8; 16];
        b[0] = self.bpp;
        b[1] = self.depth;
        b[2] = self.big_endian as u8;
        b[3] = self.true_colour as u8;
        for i in 0..3 {
            b[4 + 2 * i..6 + 2 * i].copy_from_slice(&self.max[i].to_be_bytes());
            b[10 + i] = self.shift[i];
        }
        b
    }

    pub fn from_bytes(b: &[u8; 16]) -> Result<Self, String> {
        let pf = PixelFormat {
            bpp: b[0],
            depth: b[1],
            big_endian: b[2] != 0,
            true_colour: b[3] != 0,
            max: [u16::from_be_bytes([b[4], b[5]]), u16::from_be_bytes([b[6], b[7]]), u16::from_be_bytes([b[8], b[9]])],
            shift: [b[10], b[11], b[12]],
        };
        if !matches!(pf.bpp, 8 | 16 | 32) {
            return Err(format!("{} bits per pixel", pf.bpp));
        }
        if !pf.true_colour {
            return Err("colour-map pixel formats are not supported".into());
        }
        if pf.max.iter().any(|&m| m == 0 || m & (m + 1) != 0) {
            return Err(format!("colour maxima {:?} are not 2^n-1", pf.max));
        }
        Ok(pf)
    }

    fn bytes(&self) -> usize {
        self.bpp as usize / 8
    }

    /// A compositor pixel in this format.
    #[inline]
    pub fn convert(&self, p: u32) -> u32 {
        if *self == Self::NATIVE {
            return p & 0x00ff_ffff;
        }
        let c = [p & 0xff, (p >> 8) & 0xff, (p >> 16) & 0xff];
        let mut v = 0u32;
        for i in 0..3 {
            let m = self.max[i] as u32;
            v |= ((c[i] * m + 127) / 255) << self.shift[i];
        }
        v
    }

    #[inline]
    fn put(&self, v: u32, out: &mut Vec<u8>) {
        match (self.bytes(), self.big_endian) {
            (4, false) => out.extend_from_slice(&v.to_le_bytes()),
            (4, true) => out.extend_from_slice(&v.to_be_bytes()),
            (2, false) => out.extend_from_slice(&(v as u16).to_le_bytes()),
            (2, true) => out.extend_from_slice(&(v as u16).to_be_bytes()),
            _ => out.push(v as u8),
        }
    }

    /// ZRLE's compact pixel: 32 bpp formats whose colour fits in three bytes
    /// send only those three.
    fn cpixel(&self) -> CPixel {
        if self.bpp != 32 || self.depth > 24 {
            return CPixel::Full;
        }
        let used = (0..3).fold(0u32, |m, i| m | (self.max[i] as u32) << self.shift[i]);
        // Byte offsets in the pixel as sent.
        let (low3, high3) = if self.big_endian { (1, 0) } else { (0, 1) };
        if used & 0xff00_0000 == 0 {
            CPixel::Three(low3)
        } else if used & 0xff == 0 {
            CPixel::Three(high3)
        } else {
            CPixel::Full
        }
    }
}

#[derive(Clone, Copy)]
enum CPixel {
    Full,
    /// Three bytes starting at this offset of the full pixel.
    Three(usize),
}

impl CPixel {
    fn len(self, pf: &PixelFormat) -> usize {
        match self {
            CPixel::Full => pf.bytes(),
            CPixel::Three(_) => 3,
        }
    }
    #[inline]
    fn put(self, pf: &PixelFormat, v: u32, out: &mut Vec<u8>) {
        match self {
            CPixel::Full => pf.put(v, out),
            CPixel::Three(o) => {
                let b = if pf.big_endian { v.to_be_bytes() } else { v.to_le_bytes() };
                out.extend_from_slice(&b[o..o + 3]);
            }
        }
    }
}

/// A rectangle of a tight frame `px` (row length `stride`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
}

pub fn raw(pf: &PixelFormat, px: &[u32], stride: usize, r: Rect, out: &mut Vec<u8>) {
    out.reserve(r.w * r.h * pf.bytes());
    for y in r.y..r.y + r.h {
        for &p in &px[y * stride + r.x..y * stride + r.x + r.w] {
            pf.put(pf.convert(p), out);
        }
    }
}

/// One viewer's ZRLE state: its zlib stream runs for the whole connection.
pub struct Zrle {
    z: Compress,
    tile: Vec<u32>,
    plain: Vec<u8>,
}

impl Zrle {
    pub fn new() -> Self {
        Self { z: Compress::new(Compression::new(3), true), tile: Vec::new(), plain: Vec::new() }
    }

    /// The rectangle's ZRLE data (length word and zlib bytes) onto `out`.
    pub fn encode(&mut self, pf: &PixelFormat, px: &[u32], stride: usize, r: Rect, out: &mut Vec<u8>) -> Result<(), String> {
        let cp = pf.cpixel();
        self.plain.clear();
        for ty in (r.y..r.y + r.h).step_by(64) {
            let th = 64.min(r.y + r.h - ty);
            for tx in (r.x..r.x + r.w).step_by(64) {
                let tw = 64.min(r.x + r.w - tx);
                self.tile.clear();
                for y in ty..ty + th {
                    self.tile.extend(px[y * stride + tx..y * stride + tx + tw].iter().map(|&p| pf.convert(p)));
                }
                encode_tile(pf, cp, &self.tile, tw, &mut self.plain);
            }
        }
        // Deflate with a sync flush so the viewer can decode this rectangle
        // now; worst case a little over the input.
        let len_at = out.len();
        out.extend_from_slice(&[0; 4]);
        let start = out.len();
        let mut input = &self.plain[..];
        loop {
            out.reserve(input.len() / 2 + 1024);
            let before_in = self.z.total_in();
            self.z.compress_vec(input, out, FlushCompress::Sync).map_err(|e| e.to_string())?;
            let used = (self.z.total_in() - before_in) as usize;
            input = &input[used..];
            // Done when all input went in and zlib had room to spare.
            if input.is_empty() && out.len() < out.capacity() {
                break;
            }
        }
        let n = (out.len() - start) as u32;
        out[len_at..len_at + 4].copy_from_slice(&n.to_be_bytes());
        Ok(())
    }
}

fn run_length(mut n: usize, out: &mut Vec<u8>) {
    n -= 1;
    while n >= 255 {
        out.push(255);
        n -= 255;
    }
    out.push(n as u8);
}

fn run_length_size(n: usize) -> usize {
    (n - 1) / 255 + 1
}

/// One ZRLE tile, whichever subencoding comes out smallest.
fn encode_tile(pf: &PixelFormat, cp: CPixel, t: &[u32], tw: usize, out: &mut Vec<u8>) {
    let cpl = cp.len(pf);
    let n = t.len();
    let th = n / tw;

    // Palette (up to 127 colours) and runs.
    let mut pal: Vec<u32> = Vec::with_capacity(16);
    let mut runs = 0usize;
    let mut plain_rle = 0usize;
    let mut pal_rle = 0usize;
    let mut i = 0;
    while i < n {
        let v = t[i];
        let mut j = i + 1;
        while j < n && t[j] == v {
            j += 1;
        }
        let len = j - i;
        runs += 1;
        plain_rle += cpl + run_length_size(len);
        pal_rle += if len == 1 { 1 } else { 1 + run_length_size(len) };
        if pal.len() <= 127 && !pal.contains(&v) {
            pal.push(v);
        }
        i = j;
    }
    let _ = runs;

    if pal.len() == 1 {
        out.push(1);
        cp.put(pf, pal[0], out);
        return;
    }
    let raw_size = n * cpl;
    let packed_bits = match pal.len() { 2 => 1, 3..=4 => 2, 5..=16 => 4, _ => 0 };
    let packed_size = if packed_bits > 0 { pal.len() * cpl + th * (tw * packed_bits).div_ceil(8) } else { usize::MAX };
    let pal_rle_size = if pal.len() <= 127 { pal.len() * cpl + pal_rle } else { usize::MAX };
    let best = raw_size.min(packed_size).min(plain_rle).min(pal_rle_size);
    let index = |v: u32| pal.iter().position(|&p| p == v).unwrap() as u8;

    if best == packed_size {
        out.push(pal.len() as u8);
        for &p in &pal {
            cp.put(pf, p, out);
        }
        for row in t.chunks(tw) {
            let (mut byte, mut nbits) = (0u8, 0);
            for &v in row {
                byte = (byte << packed_bits) | index(v);
                nbits += packed_bits;
                if nbits == 8 {
                    out.push(byte);
                    (byte, nbits) = (0, 0);
                }
            }
            if nbits > 0 {
                out.push(byte << (8 - nbits));
            }
        }
    } else if best == pal_rle_size {
        out.push(128 + pal.len() as u8);
        for &p in &pal {
            cp.put(pf, p, out);
        }
        for_runs(t, |v, len| {
            if len == 1 {
                out.push(index(v));
            } else {
                out.push(index(v) | 128);
                run_length(len, out);
            }
        });
    } else if best == plain_rle {
        out.push(128);
        for_runs(t, |v, len| {
            cp.put(pf, v, out);
            run_length(len, out);
        });
    } else {
        out.push(0);
        for &v in t {
            cp.put(pf, v, out);
        }
    }
}

fn for_runs(t: &[u32], mut f: impl FnMut(u32, usize)) {
    let mut i = 0;
    while i < t.len() {
        let v = t[i];
        let mut j = i + 1;
        while j < t.len() && t[j] == v {
            j += 1;
        }
        f(v, j - i);
        i = j;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::{Decompress, FlushDecompress};

    /// Decode one ZRLE rectangle back to pixels (format values).
    fn unzrle(pf: &PixelFormat, z: &mut Decompress, data: &[u8], r: Rect) -> Vec<u32> {
        let n = u32::from_be_bytes(data[..4].try_into().unwrap()) as usize;
        assert_eq!(n + 4, data.len());
        let mut plain = Vec::with_capacity(1 << 20);
        z.decompress_vec(&data[4..], &mut plain, FlushDecompress::Sync).unwrap();
        let cp = pf.cpixel();
        let cpl = cp.len(pf);
        let rd = |b: &[u8]| -> u32 {
            let mut full = [0u8; 4];
            match cp {
                CPixel::Full => full[..cpl].copy_from_slice(b),
                CPixel::Three(o) => full[o..o + 3].copy_from_slice(b),
            }
            if pf.big_endian { u32::from_be_bytes(full) } else { u32::from_le_bytes(full) }
        };
        let mut out = vec![0u32; r.w * r.h];
        let mut p = 0;
        for ty in (0..r.h).step_by(64) {
            let th = 64.min(r.h - ty);
            for tx in (0..r.w).step_by(64) {
                let tw = 64.min(r.w - tx);
                let sub = plain[p];
                p += 1;
                let mut tile = Vec::new();
                let mut pal = Vec::new();
                let pal_n = match sub { 2..=16 => sub as usize, 130..=255 => sub as usize - 128, _ => 0 };
                for _ in 0..pal_n {
                    pal.push(rd(&plain[p..p + cpl]));
                    p += cpl;
                }
                let rl = |p: &mut usize| { let mut n = 1; loop { let b = plain[*p]; *p += 1; n += b as usize; if b != 255 { return n; } } };
                match sub {
                    0 => for _ in 0..tw * th { tile.push(rd(&plain[p..p + cpl])); p += cpl; },
                    1 => { let v = rd(&plain[p..p + cpl]); p += cpl; tile = vec![v; tw * th]; }
                    2..=16 => {
                        let bits = match sub { 2 => 1, 3 | 4 => 2, _ => 4 };
                        for _ in 0..th {
                            let mut taken = 0;
                            let row_bytes = (tw * bits).div_ceil(8);
                            for x in 0..tw {
                                let bit = x * bits;
                                let b = plain[p + bit / 8];
                                let idx = (b >> (8 - bits - bit % 8)) & ((1 << bits) - 1);
                                tile.push(pal[idx as usize]);
                                taken = row_bytes;
                            }
                            p += taken;
                        }
                    }
                    128 => while tile.len() < tw * th { let v = rd(&plain[p..p + cpl]); p += cpl; let n = rl(&mut p); tile.extend(std::iter::repeat_n(v, n)); },
                    130..=255 => while tile.len() < tw * th {
                        let b = plain[p]; p += 1;
                        let v = pal[(b & 127) as usize];
                        let n = if b & 128 != 0 { rl(&mut p) } else { 1 };
                        tile.extend(std::iter::repeat_n(v, n));
                    },
                    s => panic!("subencoding {s}"),
                }
                assert_eq!(tile.len(), tw * th);
                for y in 0..th {
                    out[(ty + y) * r.w + tx..(ty + y) * r.w + tx + tw].copy_from_slice(&tile[y * tw..(y + 1) * tw]);
                }
            }
        }
        assert_eq!(p, plain.len());
        out
    }

    fn picture(w: usize, h: usize) -> Vec<u32> {
        let mut px = vec![0xff20_4060u32; w * h];
        for y in 0..h {
            for x in 0..w {
                px[y * w + x] = match (x / 40, y / 30) {
                    (0, _) => 0xff00_0000 | ((x * 7 + y * 13) as u32).wrapping_mul(2_654_435_761) & 0xff_ffff, // noise
                    (1, _) => if (x + y) % 2 == 0 { 0xffff_ffff } else { 0xff00_0000 },      // 2 colours
                    (2, _) => [0xff11_2233, 0xff44_5566, 0xff77_8899][(x / 3 + y) % 3],      // 3 colours
                    (3, _) => 0xff00_0000 | ((x / 10) as u32 * 0x30_1020),                   // runs
                    _ => 0xffab_cdef,                                                         // solid
                }
            }
        }
        px
    }

    #[test]
    fn zrle_round_trips() {
        let (w, h) = (230, 150);
        let px = picture(w, h);
        let formats = [
            PixelFormat::NATIVE,
            PixelFormat { big_endian: true, shift: [16, 8, 0], ..PixelFormat::NATIVE },
            PixelFormat { shift: [24, 16, 8], depth: 24, ..PixelFormat::NATIVE },
            PixelFormat { bpp: 16, depth: 16, big_endian: false, true_colour: true, max: [31, 63, 31], shift: [11, 5, 0] },
            PixelFormat { bpp: 8, depth: 8, big_endian: false, true_colour: true, max: [7, 7, 3], shift: [0, 3, 6] },
        ];
        for pf in formats {
            let mut enc = Zrle::new();
            let mut dec = Decompress::new(true);
            // Two rectangles on one stream, as a connection sends them.
            for r in [Rect { x: 0, y: 0, w, h }, Rect { x: 17, y: 9, w: 150, h: 70 }] {
                let mut out = Vec::new();
                enc.encode(&pf, &px, w, r, &mut out).unwrap();
                let got = unzrle(&pf, &mut dec, &out, r);
                for y in 0..r.h {
                    for x in 0..r.w {
                        assert_eq!(got[y * r.w + x], pf.convert(px[(r.y + y) * w + r.x + x]), "{pf:?} at {x},{y}");
                    }
                }
            }
        }
    }

    #[test]
    fn raw_native_is_a_copy() {
        let px = vec![0xff01_0203u32, 0xff04_0506];
        let mut out = Vec::new();
        raw(&PixelFormat::NATIVE, &px, 2, Rect { x: 0, y: 0, w: 2, h: 1 }, &mut out);
        assert_eq!(out, [3, 2, 1, 0, 6, 5, 4, 0]);
    }

    #[test]
    fn convert_16bpp() {
        let pf = PixelFormat { bpp: 16, depth: 16, big_endian: false, true_colour: true, max: [31, 63, 31], shift: [11, 5, 0] };
        assert_eq!(pf.convert(0xff00_00ff), 31 << 11); // red
        assert_eq!(pf.convert(0xffff_0000), 31); // blue
    }
}
