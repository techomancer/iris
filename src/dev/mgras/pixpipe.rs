//! The GE11's pixel pipeline (glDrawPixels, glCopyPixels' draw half, and
//! their transfer modes), high-level.
//!
//! libGLcore (mgras_pxpipe.c, `__glMgrAddModule`) builds the pipeline as a
//! program in GE memory at 0x13B: a list of (microcode entry, state address)
//! pairs, module k's state address sitting in pair k + 1 (the first pair's
//! is 0). Each module is a microcode routine named after its source file in
//! the microcode's line table (`mgras_ge.u`): `inttofloat.s` at 0x1F44 and
//! so on (see the constants). The modules stream one row's components
//! through the GE's working RAM in batches of 24, so every cycling
//! parameter (scale/bias pairs, clamp scales, LUT table offsets) belongs to
//! the component's position in the row's stream, modulo the cycle. The
//! first module reads the host's elements, the last one's output goes to
//! the raster engine in the transfer mode libGLcore sets (XFRMODE, type and
//! RGBA order; `rss::host_bytes`).
//!
//! State blocks (as traced on IRIX 6.5.22, `test/gltest/pixtest.c`): words
//! 0-4 are working-RAM pointers and batch counts (input, ?, output, whole
//! batches, remainder), the parameters follow:
//!   inttofloat  [5] scale (1/255, 2/255 for signed, 1/65535, 2^-31 ...)
//!   shiftadd    [5] shift (signed: left, or right when negative), [6]
//!               add: GL's index shift and offset; UNSIGNED_INT colour
//!               ahead of a signed int-to-float shifts by -1 (2^-31 then
//!               scales it to [0, 1])
//!   scalebias   [5..] 12 (bias, scale) pairs, cycling
//!   expand      [5] mode: 0 alpha, 1 blue, 2 green, 3 red (the missing
//!               components take fills [7..] in order from the one after
//!               the present one), 4 luminance (alpha fill [7]), 5
//!               luminance-alpha, 6 RGB (alpha fill [7]), 7 an index to
//!               all four (ahead of the I_TO_* maps); [6] a pointer
//!               adjustment. The fills already carry any scale/bias that
//!               ran before expand (BYTE's 1/255 bias: fills 1/255, 1/255,
//!               1 + 1/255 for RED).
//!   unpack      [5] element size (0 16-bit, 1 32-bit, 2 8-bit), [6..10]
//!               per-field scales, [10..14] masks, [14..18] right shifts
//!   LUT         [5] cycle length in components, [6] table id, then cycle /
//!               2 table offsets (one per component pair: the high half of a
//!               table word maps the pair's first component, the low half
//!               its second), then 2^23 (float to int)
//!   clamp       [5..] 12 output scales, cycling (255 to bytes, 127 to
//!               signed bytes, 65535, 2^32 - 1, or a LUT's size - 1); the
//!               value is clamped to [0, 1] first
//!   mask        [5] AND mask (stencil indices)
//!   colmatrix   a pass-through when its state has no matrix (5 words)
//!   bitmaptofloat  GL_BITMAP images: a bit a pixel, most significant
//!               first, to 0 or 1 (the index maps follow)
//!   compress    texture loads: [5] the internal format's components out
//!               of RGBA (0 alpha, 1 red, 2 red and alpha, 4 RGB)
//!   pipecontrol the output to the raster engine
//! The host elements are typed by ERAM 0xDBB (0 signed and 1 unsigned
//! bytes, 2 signed and 3 unsigned shorts, 4 32-bit); a pipeline whose first
//! module is none of the converters takes 32-bit elements as floats.
//!
//! Not modelled: colour matrices (SGI_color_matrix), convolution,
//! histogram / minmax, the zoom modules (the draw zooms geometrically).

/// Microcode entry points (6.5.22 `mgras_ge.u`; the same in the revB and
/// HQ4 builds).
pub mod entry {
    pub const CLAMP: u32 = 0x156A;
    pub const EXPAND: u32 = 0x1655;
    pub const SHIFTADD: u32 = 0x1865;
    pub const MASK: u32 = 0x1A3E;
    pub const LUT: u32 = 0x1AC4;
    pub const SCALEBIAS: u32 = 0x1E3E;
    pub const INTTOFLOAT: u32 = 0x1F44;
    pub const PIPECONTROL: u32 = 0x352D;
    pub const ZOOM: [u32; 3] = [0x3663, 0x3715, 0x374A];
    pub const UNPACK: u32 = 0x3874;
    pub const COLMATRIX: u32 = 0x3E65;
    pub const COMPRESS: u32 = 0x399A;
    pub const BITMAPTOFLOAT: u32 = 0x24D7;
    pub const XZOOM: u32 = 0x3EF5;
}

/// GE memory 0x100..0x200 (the pipeline program and some module states).
pub const GE_BASE: u32 = 0x100;
pub const GE_WORDS: usize = 0x100;
/// ERAM 0xC00..0xE00 (the other module states, 0xDA8 / 0xDB7 blocks).
pub const ERAM_BASE: u32 = 0xC00;
pub const ERAM_WORDS: usize = 0x200;
/// Lookup table RAM (INIT_COLOR_TABLE), words of two 16-bit entries.
pub const LUT_WORDS: usize = 0x400;
/// GE address of the pipeline program.
pub const PROGRAM: u32 = 0x13B;
/// ERAM word typing the host elements.
pub const ELEMENT_TYPE: u32 = 0xDBB;

/// A context's pixel pipeline memory. Plain data, valid zeroed (part of
/// `Gl`, saved with the context).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PixMem {
    pub ge: [u32; GE_WORDS],
    pub eram: [u32; ERAM_WORDS],
    pub lut: [u32; LUT_WORDS],
    /// Words in the program at 0x13B, as its last write gave them (a
    /// shorter program leaves the end of the one before it in memory).
    pub prog_len: u32,
}

impl PixMem {
    /// A pixel state word (PIXEL_STATE 0x080 for GE memory, 0x0CD for
    /// ERAM; the address ranges do not overlap).
    pub fn write(&mut self, addr: u32, v: u32) {
        if let Some(w) = self.word_mut(addr) {
            *w = v;
        }
    }

    fn word_mut(&mut self, addr: u32) -> Option<&mut u32> {
        if (GE_BASE..GE_BASE + GE_WORDS as u32).contains(&addr) {
            return self.ge.get_mut((addr - GE_BASE) as usize);
        }
        if (ERAM_BASE..ERAM_BASE + ERAM_WORDS as u32).contains(&addr) {
            return self.eram.get_mut((addr - ERAM_BASE) as usize);
        }
        None
    }

    pub fn read(&self, addr: u32) -> u32 {
        if (GE_BASE..GE_BASE + GE_WORDS as u32).contains(&addr) {
            return self.ge[(addr - GE_BASE) as usize];
        }
        if (ERAM_BASE..ERAM_BASE + ERAM_WORDS as u32).contains(&addr) {
            return self.eram[(addr - ERAM_BASE) as usize];
        }
        0
    }

    fn state(&self, addr: u32, n: usize) -> Vec<u32> {
        (0..n as u32).map(|k| self.read(addr + k)).collect()
    }

    /// INIT_COLOR_TABLE's payload: [0, bytes, table id, 0, offset, count,
    /// words].
    pub fn load_table(&mut self, payload: &[u32]) {
        if payload.len() < 6 {
            return;
        }
        let (off, n) = (payload[4] as usize, payload[5] as usize);
        for (k, &w) in payload[6..].iter().take(n).enumerate() {
            if let Some(slot) = self.lut.get_mut(off + k) {
                *slot = w;
            }
        }
    }

    /// A block write starting at `addr`, `n` words (the program's length
    /// is kept when one starts at 0x13B).
    pub fn note_block(&mut self, addr: u32, n: usize) {
        if addr == PROGRAM {
            self.prog_len = n as u32;
        }
    }

    /// The pipeline: (entry, state address) per module, in order.
    pub fn modules(&self) -> Vec<(u32, u32)> {
        let len = (self.prog_len as usize).min(32) & !1;
        let prog: Vec<u32> = (0..len as u32).map(|k| self.read(PROGRAM + k)).collect();
        let n = prog.len() / 2;
        (0..n).map(|k| (prog[2 * k], if 2 * k + 3 < prog.len() { prog[2 * k + 3] } else { 0 })).collect()
    }

    /// The GE's x replication for a fractional zoom (`xzoom`, its state
    /// word 4: ceil(zx)), 1 without.
    pub fn xzoom_rep(&self) -> usize {
        if !self.has_program() {
            return 1;
        }
        // The state's index table (from word 7) gives each output pixel's
        // source offset: the first source pixel's run is the replication.
        match self.modules().iter().find(|m| m.0 == entry::XZOOM) {
            Some(&(_, st)) => (0..64).take_while(|&k| self.read(st + 7 + k) == 0).count().clamp(1, 64),
            None => 1,
        }
    }

    /// Whether a program is loaded.
    pub fn has_program(&self) -> bool {
        self.prog_len >= 2 && self.read(PROGRAM) != 0
    }

    /// Host components a pixel carries into the pipeline (packed types: 1
    /// element), their element size in bytes, and the element type. With
    /// `fit` (row bytes available, pixels) the size is the largest whose
    /// row fits: a stencil or colour-index image's type word reads 4 for
    /// bytes.
    pub fn input_shape_fit(&self, out_comps: usize, fit: Option<(usize, usize)>) -> (usize, usize, Elem) {
        let (comps, size, elem) = self.input_shape(out_comps);
        let Some((bytes, px)) = fit else { return (comps, size, elem) };
        if elem == Elem::Bit {
            return (comps, size, elem);
        }
        if px == 0 || comps * size * px <= bytes || !matches!(elem, Elem::S32 | Elem::U32) {
            return (comps, size, elem);
        }
        // Integer elements narrower than the type word says.
        let signed = elem == Elem::S32;
        match [2usize, 1].into_iter().find(|&s| comps * s * px <= bytes) {
            Some(2) => (comps, 2, if signed { Elem::S16 } else { Elem::U16 }),
            Some(_) => (comps, 1, if signed && self.modules().first().map(|m| m.0) != Some(entry::MASK) { Elem::S8 } else { Elem::U8 }),
            None => (comps, size, elem),
        }
    }

    pub fn input_shape(&self, out_comps: usize) -> (usize, usize, Elem) {
        let mods = self.modules();
        let first = mods.first().map(|m| m.0).unwrap_or(0);
        let code = self.read(ELEMENT_TYPE);
        let elem = match code {
            0 => Elem::S8,
            1 => Elem::U8,
            2 => Elem::S16,
            3 => Elem::U16,
            _ => match first {
                entry::SHIFTADD => Elem::U32,
                entry::INTTOFLOAT | entry::UNPACK | entry::MASK => Elem::S32,
                _ => Elem::F32,
            },
        };
        if first == entry::BITMAPTOFLOAT {
            return (1, 0, Elem::Bit);
        }
        let comps = if mods.iter().any(|m| m.0 == entry::UNPACK) {
            1
        } else if let Some(&(_, st)) = mods.iter().find(|m| m.0 == entry::EXPAND) {
            match self.read(st + 5) {
                5 => 2,
                6 => 3,
                _ => 1,
            }
        } else {
            out_comps
        };
        (comps, elem.size(), elem)
    }

    /// Run one row of `pixels` pixels of host bytes through the pipeline:
    /// the components the raster engine gets, in order, as numbers of its
    /// transfer type (rounded by the caller's packing).
    pub fn run_row(&self, input: &[u8], pixels: usize, out_comps: usize) -> Vec<f64> {
        let (comps, esize, elem) = self.input_shape_fit(out_comps, Some((input.len(), pixels)));
        let n = pixels * comps;
        let mut v: Vec<f64> = if elem == Elem::Bit {
            // glDrawPixels(GL_COLOR_INDEX / GL_STENCIL_INDEX, GL_BITMAP):
            // one bit a pixel, the first the most significant.
            (0..n).map(|i| (input.get(i / 8).copied().unwrap_or(0) >> (7 - i % 8) & 1) as f64).collect()
        } else {
            (0..n).map(|i| elem.get(input, i * esize)).collect()
        };
        let mut per_px = comps;
        for (e, st) in self.modules() {
            match e {
                entry::SHIFTADD => {
                    // Shift (left for positive, right for negative), then
                    // add: GL's index shift and offset; UNSIGNED_INT colour
                    // ahead of a signed int-to-float shifts by -1.
                    let (sh, add) = (self.read(st + 5) as i32, self.read(st + 6) as i32 as f64);
                    for x in v.iter_mut() {
                        let i = *x as i64;
                        let y = if sh >= 0 { i << sh.min(32) } else { i >> (-sh).min(63) };
                        *x = y as f64 + add;
                    }
                }
                entry::INTTOFLOAT => {
                    let scale = f32::from_bits(self.read(st + 5)) as f64;
                    for x in v.iter_mut() {
                        *x *= scale;
                    }
                }
                entry::UNPACK => {
                    let s = self.state(st, 18);
                    let fields = (0..4).filter(|&k| s[10 + k] != 0).count();
                    let mut out = Vec::with_capacity(v.len() * fields);
                    for &x in &v {
                        let u = x as i64 as u64 as u32;
                        for k in 0..fields {
                            let f = (u & s[10 + k]) >> (s[14 + k] & 31);
                            out.push(f as f64 * f32::from_bits(s[6 + k]) as f64);
                        }
                    }
                    v = out;
                    per_px = fields;
                }
                entry::SCALEBIAS => {
                    let s = self.state(st, 5 + 24);
                    for (i, x) in v.iter_mut().enumerate() {
                        let k = i % 12;
                        let (b, sc) = (f32::from_bits(s[5 + 2 * k]) as f64, f32::from_bits(s[6 + 2 * k]) as f64);
                        *x = *x * sc + b;
                    }
                }
                entry::EXPAND => {
                    let s = self.state(st, 10);
                    let fill = |k: usize| f32::from_bits(s[7 + k]) as f64;
                    let mut out = Vec::with_capacity(pixels * 4);
                    match s[5] {
                        m @ 0..=3 => {
                            // The present component and the missing ones in
                            // order after it (fills 0, 1, 2).
                            let p = [3usize, 2, 1, 0][m as usize];
                            for &x in &v {
                                let mut px = [0.0; 4];
                                px[p] = x;
                                for j in 0..3 {
                                    px[(p + 1 + j) % 4] = fill(j);
                                }
                                out.extend_from_slice(&px);
                            }
                        }
                        4 => {
                            for &x in &v {
                                out.extend_from_slice(&[x, x, x, fill(0)]);
                            }
                        }
                        7 => {
                            // A colour index to all four components (the
                            // I_TO_R/G/B/A maps follow).
                            for &x in &v {
                                out.extend_from_slice(&[x, x, x, x]);
                            }
                        }
                        5 => {
                            for p in v.chunks(2) {
                                let (l, a) = (p[0], p.get(1).copied().unwrap_or(1.0));
                                out.extend_from_slice(&[l, l, l, a]);
                            }
                        }
                        _ => {
                            for p in v.chunks(3) {
                                out.extend_from_slice(&[p[0], *p.get(1).unwrap_or(&0.0), *p.get(2).unwrap_or(&0.0), fill(0)]);
                            }
                        }
                    }
                    v = out;
                    per_px = 4;
                }
                entry::LUT => {
                    let cycle = (self.read(st + 5) as usize).max(2);
                    let offs: Vec<usize> = (0..cycle / 2).map(|k| self.read(st + 7 + k as u32) as usize).collect();
                    for (i, x) in v.iter_mut().enumerate() {
                        let pos = i % cycle;
                        let off = offs[pos / 2];
                        let idx = x.round().max(0.0) as usize;
                        let w = self.lut.get(off + idx).copied().unwrap_or(0);
                        *x = if pos % 2 == 0 { (w >> 16) as f64 } else { (w & 0xFFFF) as f64 };
                    }
                }
                entry::CLAMP => {
                    let s = self.state(st, 5 + 12);
                    for (i, x) in v.iter_mut().enumerate() {
                        let sc = f32::from_bits(s[5 + i % 12]) as f64;
                        *x = x.clamp(0.0, 1.0) * sc;
                    }
                }
                entry::COMPRESS => {
                    // Texture loads: the internal format's components out
                    // of RGBA ([5]: 0 alpha, 1 red (luminance, intensity),
                    // 2 red and alpha, 4 RGB).
                    let keep: &[usize] = match self.read(st + 5) {
                        0 => &[3],
                        1 => &[0],
                        2 => &[0, 3],
                        4 => &[0, 1, 2],
                        _ => &[0, 1, 2, 3],
                    };
                    let n = per_px.max(1);
                    v = v.chunks(n).flat_map(|p| keep.iter().map(move |&k| p.get(k).copied().unwrap_or(0.0))).collect();
                    per_px = keep.len();
                }
                entry::MASK => {
                    let m = self.read(st + 5);
                    for x in v.iter_mut() {
                        *x = ((*x as i64 as u64 as u32) & m) as f64;
                    }
                }
                // colmatrix without a matrix, pipecontrol, zoom: as they
                // are (the draw zooms).
                _ => {}
            }
        }
        let _ = per_px;
        v
    }
}

/// A host element type.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Elem {
    /// One bit a pixel (GL_BITMAP), `size` 0.
    Bit,
    S8,
    U8,
    S16,
    U16,
    S32,
    U32,
    F32,
}

impl Elem {
    pub fn size(self) -> usize {
        match self {
            Elem::Bit => 0,
            Elem::S8 | Elem::U8 => 1,
            Elem::S16 | Elem::U16 => 2,
            _ => 4,
        }
    }

    /// Element at byte `at` of big-endian host data (0 past the end).
    fn get(self, b: &[u8], at: usize) -> f64 {
        let byte = |k: usize| b.get(at + k).copied().unwrap_or(0);
        match self {
            Elem::Bit => byte(0) as f64,
            Elem::S8 => byte(0) as i8 as f64,
            Elem::U8 => byte(0) as f64,
            Elem::S16 => i16::from_be_bytes([byte(0), byte(1)]) as f64,
            Elem::U16 => u16::from_be_bytes([byte(0), byte(1)]) as f64,
            Elem::S32 => i32::from_be_bytes([byte(0), byte(1), byte(2), byte(3)]) as f64,
            Elem::U32 => u32::from_be_bytes([byte(0), byte(1), byte(2), byte(3)]) as f64,
            Elem::F32 => f32::from_be_bytes([byte(0), byte(1), byte(2), byte(3)]) as f64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mem(prog: &[u32]) -> Box<PixMem> {
        // SAFETY: plain numbers.
        let mut m: Box<PixMem> = Box::new(unsafe { std::mem::zeroed() });
        for (k, &w) in prog.iter().enumerate() {
            m.write(PROGRAM + k as u32, w);
        }
        m.note_block(PROGRAM, prog.len());
        m
    }

    fn put(m: &mut PixMem, addr: u32, words: &[u32]) {
        for (k, &w) in words.iter().enumerate() {
            m.write(addr + k as u32, w);
        }
    }

    /// glDrawPixels(GL_RGBA, GL_UNSIGNED_BYTE), traced: bytes to floats,
    /// clamped and scaled back to bytes.
    #[test]
    fn rgba_bytes_round_trip() {
        let mut m = mem(&[entry::INTTOFLOAT, 0, entry::COLMATRIX, 0xD05, entry::CLAMP, 0x1B4, entry::PIPECONTROL, 0xD67]);
        put(&mut m, 0xD05, &[0x34, 0x4064, 0x50C, 2, 4, (1.0f32 / 255.0).to_bits()]);
        put(&mut m, 0x1B4, &[6, 4, 0, 0x50C, 0x50C]);
        put(&mut m, 0xD67, &[0x50C, 0x1A000, 0x50C, 2, 4]);
        put(&mut m, 0xD67 + 5, &[255.0f32.to_bits(); 12]);
        m.write(ELEMENT_TYPE, 1);
        let out = m.run_row(&[0x11, 0x6C, 0xC7, 0x22], 1, 4);
        assert_eq!(out.iter().map(|x| x.round() as u32).collect::<Vec<_>>(), [0x11, 0x6C, 0xC7, 0x22]);
    }

    /// GL_RED with GL_BYTE, traced: signed bytes scaled by 2/255, the
    /// 1/255 bias, then the missing components from the fills.
    #[test]
    fn red_signed_bytes_expand() {
        let mut m = mem(&[
            entry::INTTOFLOAT, 0, entry::SCALEBIAS, 0xD05, entry::EXPAND, 0xC68, entry::COLMATRIX, 0x1D6,
            entry::CLAMP, 0x1B4, entry::PIPECONTROL, 0xD67,
        ]);
        put(&mut m, 0xD05, &[0x14, 0x424A, 0x50C, 0, 0xD, (2.0f32 / 255.0).to_bits()]);
        let mut sb = vec![0x50C, 0, 0x50C, 0, 0xD];
        for _ in 0..12 {
            sb.extend_from_slice(&[(1.0f32 / 255.0).to_bits(), 1.0f32.to_bits()]);
        }
        put(&mut m, 0xC68, &sb);
        let i = 1.0f32 / 255.0;
        put(&mut m, 0x1D6, &[0x50C, 0, 0x50C, 2, 4, 3, 0xFFFF_FC34, i.to_bits(), i.to_bits(), (1.0 + i).to_bits()]);
        put(&mut m, 0x1B4, &[6, 4, 0, 0x50C, 0x50C]);
        put(&mut m, 0xD67, &[0x50C, 0x1A000, 0x50C, 2, 4]);
        put(&mut m, 0xD67 + 5, &[127.0f32.to_bits(); 12]);
        m.write(ELEMENT_TYPE, 0);
        // -111 clamps to 0; 63 is (2 * 63 + 1) / 255 = 0.498 -> 63.
        let out = m.run_row(&[(-111i8) as u8, 63], 2, 4);
        let r: Vec<u32> = out.iter().map(|x| x.round() as u32).collect();
        assert_eq!(r, [0, 0, 0, 127, 63, 0, 0, 127]);
    }

    /// GL_UNSIGNED_SHORT_4_4_4_4, traced: one element, four fields.
    #[test]
    fn packed_4444() {
        let mut m = mem(&[entry::UNPACK, 0, entry::COLMATRIX, 0xD78, entry::CLAMP, 0x1B4, entry::PIPECONTROL, 0xD67]);
        let f = |x: f32| x.to_bits();
        put(&mut m, 0xD78, &[0x1C, 0x4064, 0x50C, 3, 1, 0,
            f(1.0 / 61440.0), f(1.0 / 3840.0), f(1.0 / 240.0), f(1.0 / 15.0),
            0xF000, 0xF00, 0xF0, 0xF, 0, 0, 0, 0]);
        put(&mut m, 0xD67, &[0x50C, 0x1A000, 0x50C, 2, 4]);
        put(&mut m, 0xD67 + 5, &[255.0f32.to_bits(); 12]);
        m.write(ELEMENT_TYPE, 3);
        let out = m.run_row(&[0x1F, 0x8A], 1, 4);
        let r: Vec<u32> = out.iter().map(|x| x.round() as u32).collect();
        assert_eq!(r, [0x11, 0xFF, 0x88, 0xAA]);
    }

    /// GL_MAP_COLOR, traced: indices from clamp x 255, two components a
    /// table word, back to floats by 1/65535.
    #[test]
    fn lut_pairs() {
        let mut m = mem(&[
            entry::INTTOFLOAT, 0, entry::CLAMP, 0xD05, entry::LUT, 0xD23, entry::INTTOFLOAT, 0xCBF,
            entry::CLAMP, 0xD0B, entry::PIPECONTROL, 0xD67,
        ]);
        put(&mut m, 0xD05, &[0x34, 0x4064, 0x50C, 2, 4, (1.0f32 / 255.0).to_bits()]);
        put(&mut m, 0xD23, &[0x50C, 0, 0x50C, 2, 4]);
        put(&mut m, 0xD23 + 5, &[255.0f32.to_bits(); 12]);
        put(&mut m, 0xCBF, &[0x50C, 0, 0x50C, 2, 4, 0xC, 0xC0D, 0, 0x100, 0, 0x100, 0, 0x100, 0x4B00_0000]);
        put(&mut m, 0xD0B, &[0x50C, 0, 0x50C, 2, 4, (1.0f32 / 65535.0).to_bits()]);
        put(&mut m, 0xD67, &[0x50C, 0x1A000, 0x50C, 2, 4]);
        put(&mut m, 0xD67 + 5, &[255.0f32.to_bits(); 12]);
        // R|G at 0 (R map 1 - c, G identity), B|A at 0x100.
        let mut t = vec![0, 0x400, 0xC0D, 0, 0, 0x100];
        t.extend((0..0x100u32).map(|i| (0xFFFF - i * 257) << 16 | i * 257));
        m.load_table(&t);
        let mut t = vec![0, 0x400, 0xC0D, 0, 0x100, 0x100];
        t.extend((0..0x100u32).map(|i| (i * 257) << 16 | 0xFFFF));
        m.load_table(&t);
        m.write(ELEMENT_TYPE, 1);
        let out = m.run_row(&[0x20, 0x40, 0x60, 0x80], 1, 4);
        let r: Vec<u32> = out.iter().map(|x| x.round() as u32).collect();
        assert_eq!(r, [0xDF, 0x40, 0x60, 0xFF]);
    }
}
