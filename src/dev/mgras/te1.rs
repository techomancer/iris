//! TE1, IMPACT's texture engine, and its TRAM.
//!
//! TE1 sits in the raster subsystem between the RE4's rasteriser and its
//! fragment colouring. Its registers are RSS registers (`reg`); what is known
//! of them is in `ignore/gr4/TE1.h`.
//!
//! - **Download.** TRAM has no address/data port. Texels arrive through the
//!   raster engine's transfer engine with `XFRCONTROL` pointing at the
//!   texture side. The texture loader registers (`TL_*`) place them: page,
//!   level size, sub-image origin, components and depth. libGLcore sends the
//!   texels through the GE pixel pipeline (SEND_PIXELS routine 0x511A); our
//!   GE HLE turns that into the transfer (`gl.rs`).
//! - **Sampling.** Per fragment the RE hands the TE S/W, T/W and 1/W. The TE
//!   divides out W, picks a level of detail from the derivatives, then
//!   fetches and filters from TRAM (`Sampler`).
//! - **Texture environment.** The texel is combined with the fragment colour
//!   in the RE (`tex_env`): the environment registers are RE registers.
//!
//! TRAM layout is ours. libGLcore's texture manager hands out 16 KB pages
//! (4 TRAMs; `TXMIPMAP` / `TL_MIPMAP` hold a page and a bank bit per level),
//! and every texel's address must follow from what the registers say. A
//! level starts at its page. Levels of 16x16 texels and smaller share one
//! page, as libGLcore packs them, each at a fixed offset by size. A texel
//! takes `components * depth` nibbles (depth 4, 8 or 12 bits) rounded up to
//! a power of two, as on the board.
//!
//! - **Save and restore.** libGLcore's texture manager moves textures by
//!   saving TRAM pages to host memory and loading them back elsewhere,
//!   through a view of TRAM as a level 64 texels wide in the texture's own
//!   format (`read_raw`, and a load with TL_SPEC 64 x 8192). It saves and
//!   restores in runs of whole pages, so a page here must hold exactly the
//!   texels a page holds on the board.
//!
//! A level's border texels (GL borders, and libGLcore's own) go to its
//! border page (TXBORDER per level, TL_BORDER at load), in the page's
//! upper half. GL_CLAMP clamps coordinates to the texture, so linear
//! filtering at an edge reaches the border: the border texel, or the border
//! colour (TXBCOLOR) when the texture has none (TEXMODE2 bit 16).
//!
//! Readback (glGetTexImage): the GE aims the sampler at the level and
//! arms a read transfer on the texture side; each line is a row of texels,
//! RGBA bytes (`read_line`).
//!
//! Colour arithmetic is 12.16 fixed point (`fixed`).
//!
//! Detail textures (SGIS_detail_texture; observed on the board's driver):
//! TEXMODE2 bit 9 turns them on, bit 10 picks a 128-texel detail image
//! (else 256). TXDETAIL bits 7:0 is its TRAM page; it is stored as a level
//! twice as wide as high (2N x N) whose left N columns hold the image, in
//! the base texture's format. TXSIZE bits 18:16 hold 6 + the detail level
//! L (L < 0), so the detail texel coordinate is the base's level-0
//! coordinate times 2^-L, repeating. The weight F(LOD) comes from the
//! DETAILSCALE table: 32 five-bit weights (31 = 1.0) at k = 32 2^LOD,
//! linear between, zero past the last entry the driver used (TXLOD bits
//! 15:10, twice that index). The board only adds: out = base + F (2 detail
//! - 1), whatever detail mode GL asked for. Detail applies when
//! magnifying (LOD <= 0) only (`Sampler::detail`).
//!
//! Not yet: sharpen textures, 3D textures, texture lookup tables, video
//! texture.

use super::fixed;

/// TE1 / texture-related RSS registers.
pub mod reg {
    pub const TEXMODE1: u32 = 0x111;
    pub const TXENV_RG: u32 = 0x142;
    pub const TXENV_B: u32 = 0x143;
    pub const TEXMODE2: u32 = 0x180;
    pub const TXSIZE: u32 = 0x181;
    pub const TXLOD: u32 = 0x183;
    pub const TXBCOLOR_RG: u32 = 0x184;
    pub const TXBCOLOR_BA: u32 = 0x185;
    /// Index into the per-level tables below, advanced by each table write.
    pub const TXADDR: u32 = 0x190;
    pub const TXMIPMAP: u32 = 0x191;
    /// The detail image's TRAM page (bits 7:0).
    pub const TXDETAIL: u32 = 0x192;
    pub const TXBORDER: u32 = 0x193;
    pub const DETAILSCALE: u32 = 0x194;
    pub const TL_WBUFFER: u32 = 0x1A0;
    pub const TL_MODE: u32 = 0x1A1;
    pub const TL_SPEC: u32 = 0x1A2;
    pub const TL_ADDR: u32 = 0x1A3;
    pub const TL_MIPMAP: u32 = 0x1A4;
    pub const TL_BORDER: u32 = 0x1A5;
    pub const TL_S_SIZE: u32 = 0x1A9;
    pub const TL_T_SIZE: u32 = 0x1AA;
    pub const TL_S_LEFT: u32 = 0x1AB;
    pub const TL_T_BOTTOM: u32 = 0x1AC;
    /// Iterators (double buffered; 64-bit hi/lo pairs): S/W, T/W, 1/W at
    /// the first pixel, their steps along the span (X) and down the major
    /// edge (E), and their Y derivatives.
    pub const SW: u32 = 0x080;
    pub const TW: u32 = 0x082;
    pub const WI: u32 = 0x084;
    pub const DWIE: u32 = 0x086;
    pub const DWIX: u32 = 0x088;
    pub const DWIY: u32 = 0x08A;
    pub const DSWE: u32 = 0x0C0;
    pub const DTWE: u32 = 0x0C2;
    pub const DSWX: u32 = 0x0C4;
    pub const DTWX: u32 = 0x0C6;
    pub const DSWY: u32 = 0x0C8;
    pub const DTWY: u32 = 0x0CA;
}

/// The texture registers a GL context owns (single buffered), which the GE
/// keeps a shadow of and loads into the TE (`gl.rs`). TXADDR and the
/// per-level tables it indexes (TXMIPMAP, TXBORDER, DETAILSCALE) are kept
/// apart.
pub const CONTEXT_REGS: [u32; 31] = [
    0x111, 0x142, 0x143, 0x180, 0x181, 0x182, 0x183, 0x184, 0x185, 0x186, 0x187, 0x188, 0x189, 0x18A,
    0x18B, 0x18C, 0x192, 0x195, 0x1A0, 0x1A1, 0x1A2, 0x1A3, 0x1A4, 0x1A5, 0x1A6, 0x1A7, 0x1A8, 0x1A9,
    0x1AA, 0x1AB, 0x1AC,
];

/// Per-level tables written through TXADDR's index.
pub const TABLE_REGS: [u32; 3] = [reg::TXMIPMAP, reg::TXBORDER, reg::DETAILSCALE];
pub const TABLE_LEN: usize = 32;

/// Fixed point of the iterator registers: value * 2^32 (our choice; the
/// IDE's values suggest 1/W = 1.0 is 2^20 on the real board).
pub const ITER_ONE: f64 = 4_294_967_296.0;

/// TEXMODE1: texture enable, environment mode (bits 2:1: modulate, decal,
/// blend, alpha-only modulate), components - 1 (4:3), texel class (11:9:
/// 4 RGB(A), 2 luminance(-alpha), 1 alpha, 3 intensity).
pub const TEXMODE1_ENABLE: u32 = 1;

/// TEXMODE2: magnification / minification within a level linear, mipmaps
/// (nearest level; with bit 6 blending two), clamp S and T, mipmapping on.
pub(crate) const TM2_MAG_LINEAR: u32 = 1 << 7;
pub(crate) const TM2_MIN_LINEAR: u32 = 1 << 8;
pub(crate) const TM2_MIPMAP: u32 = 1 << 5;
pub(crate) const TM2_MIP_LINEAR: u32 = 1 << 6;
/// Clamp S / T (and R, bit 13) instead of repeating; with bit 14
/// GL_CLAMP's coordinate clamp to [0, 1] (linear filtering at the edge is
/// half border), without it GL_CLAMP_TO_BORDER_SGIS (coordinates may pass
/// the edge, so samples there are all border). libGLcore
/// __glMgrim_TexParameter: GL_CLAMP sets axis bit | 0x4000, clamp to border
/// the axis bit alone.
pub(crate) const TM2_CLAMP_S: u32 = 1 << 11;
pub(crate) const TM2_CLAMP_T: u32 = 1 << 12;
pub(crate) const TM2_GL_CLAMP: u32 = 1 << 14;
/// The texture has no border: clamped fetches outside it read TXBCOLOR.
pub(crate) const TM2_NO_BORDER: u32 = 1 << 16;
/// Detail texture on; with bit 10 its image is 128 texels square, else 256.
pub(crate) const TM2_DETAIL: u32 = 1 << 9;
pub(crate) const TM2_DETAIL_128: u32 = 1 << 10;
pub(crate) const TM2_MM_ENABLE: u32 = 1 << 19;

pub const TRAM_BYTES: usize = 4 << 20;
pub(crate) const TRAM_NIBBLES: usize = TRAM_BYTES * 2;
/// A TRAM page: 16 KB with 4 TRAMs.
pub(crate) const PAGE_NIBBLES: usize = 16384 * 2;

/// Texel offset in a shared page of a level whose larger side is `n` (<=
/// 16): 16x16 first, then 8x8, 4x4, 2x2, 1x1.
pub(crate) fn small_offset(n: usize) -> usize {
    let (mut off, mut k) = (0, 16);
    while k > n {
        off += k * k;
        k /= 2;
    }
    off
}

/// TRAM nibbles a texel cell takes: four components of `d` nibbles,
/// rounded up to a power of two, whatever the texture's own components
/// (RGB8 and L8 alike take 32 bits: libGLcore counts 4096 texels to a
/// 16 KB page for both, 8192 for RGBA4). Textures of one or two
/// components share cells (`slot`). The texture manager's page saves move
/// 64-cell rows, so our pages must hold what its pages hold.
fn cell_nibbles(d: usize) -> usize {
    (4 * d).next_power_of_two()
}

/// Nibble offset in its cell of component `c` of a texture of `nc`
/// components in select slot `sel` (SGIS_texture_select: TL_MODE bits 4:3
/// at load, TEXMODE1 bits 8:7 when sampling; traced, TyrQuake packs four
/// 256x256 luminance lightmaps into one set of pages, slots 0-3).
/// Only textures that leave room share cells: four slots for one
/// component, two for two; three- and four-component textures fill the
/// cell, and the select bits mean nothing for them (TyrQuake draws RGBA
/// textures with TEXMODE1 0x409D9, bits 8:7 = 3).
fn comp_offset(sel: usize, nc: usize, c: usize, d: usize) -> usize {
    let slot = match nc {
        1 => sel & 3,
        2 => (sel & 1) * 2,
        _ => 0,
    };
    (slot + c).min(3) * d
}

/// Nibble address of texel (s, t) in a w x h level at TRAM `page`, texels
/// of `tn` nibbles.
fn texel_addr(page: u32, w: usize, h: usize, s: usize, t: usize, tn: usize) -> usize {
    let side = w.max(h);
    let off = if side <= 16 { small_offset(side) } else { 0 };
    (page as usize * PAGE_NIBBLES + (off + t * w + s) * tn) % TRAM_NIBBLES
}

/// Border texel offset in a shared border page for a level whose larger
/// side is `n` (<= 16): each level's border is 4n + 4 texels.
pub(crate) fn small_border_offset(n: usize) -> usize {
    let (mut off, mut k) = (0, 16);
    while k > n {
        off += 4 * k + 4;
        k /= 2;
    }
    off
}

/// Nibble address of border texel (s, t), s in -1..=w, t in -1..=h, on
/// the border of a w x h level, in the upper half of border page `page`:
/// the bottom row, the top row (w + 2 each, corners included), then the
/// left and right columns (h each).
fn border_addr(page: u32, w: usize, h: usize, s: i64, t: i64, tn: usize) -> usize {
    let (wi, hi) = (w as i64, h as i64);
    let idx = if t < 0 {
        s + 1
    } else if t >= hi {
        wi + 2 + s + 1
    } else if s < 0 {
        2 * (wi + 2) + t
    } else {
        2 * (wi + 2) + hi + t
    } as usize;
    let side = w.max(h);
    let off = if side <= 16 { small_border_offset(side) } else { 0 };
    (page as usize * PAGE_NIBBLES + PAGE_NIBBLES / 2 + (off + idx) * tn) % TRAM_NIBBLES
}

/// A component's depth in nibbles, from TL_MODE bits 6:5 / TEXMODE2 bits
/// 3:2: 0, 1, 2 = 4, 8, 12 bits (traced: libGLcore stores RGBA8 at 0 with 1
/// TRAM, at 1 with 4).
pub(crate) fn depth_nibbles(field: u32) -> usize {
    match field & 3 {
        0 => 1,
        2 => 3,
        _ => 2,
    }
}

/// The texture engine: TRAM and the per-level tables. Plain data, valid
/// zeroed.
#[repr(C)]
pub struct Te1 {
    tram: [u8; TRAM_BYTES],
    tables: [[u32; TABLE_LEN]; 3],
    pub texels_loaded: u64,
    /// Diagnostic: per page, the last level loaded there (page << 16 |
    /// TL_SPEC's sizes), for `stale`.
    owner: [u32; 256],
    /// Pages a load wrote since the GPU unit last took them (one bit a
    /// 16 KB page): what it sends up before it samples again.
    pub dirty: [u64; 4],
}

impl Te1 {
    /// TRAM, for the JIT's texel fetches.
    pub fn tram_ptr(&self) -> *const u8 {
        self.tram.as_ptr()
    }

    /// TRAM, writable (tests).
    #[cfg(test)]
    pub fn tram_mut(&mut self) -> &mut [u8] {
        &mut self.tram
    }

    /// `n` nibbles from nibble address `a`, the first leftmost (monitor).
    pub fn nibbles_hex(&self, a: usize, n: usize) -> String {
        (0..n).map(|i| char::from_digit(self.nibble((a + i) % TRAM_NIBBLES), 16).unwrap_or('?')).collect()
    }

    fn nibble(&self, a: usize) -> u32 {
        let b = self.tram[a / 2] as u32;
        if a & 1 == 0 { b & 0xF } else { b >> 4 }
    }

    fn set_nibble(&mut self, a: usize, v: u32) {
        let b = &mut self.tram[a / 2];
        *b = if a & 1 == 0 { (*b & 0xF0) | (v as u8 & 0xF) } else { (*b & 0x0F) | (v as u8) << 4 };
    }

    /// Component of `d` nibbles at nibble address `a`, widened to 12 bits.
    fn component(&self, a: usize, d: usize) -> u32 {
        let v = (0..d).fold(0, |v, i| v | self.nibble((a + i) % TRAM_NIBBLES) << (4 * i));
        fixed::widen(v, d)
    }

    /// A table write (TXMIPMAP, TXBORDER, DETAILSCALE) at `index`.
    pub fn table_write(&mut self, r: u32, index: u32, v: u32) {
        if let Some(k) = TABLE_REGS.iter().position(|&t| t == r) {
            self.tables[k][index as usize % TABLE_LEN] = v;
        }
    }

    /// One line of a texture load, 4 bytes a texel (components in order,
    /// unused ones ignored), placed by the texture loader registers: line
    /// `line` is texel row TL_T_BOTTOM + line, from column TL_S_LEFT, in
    /// the level of TL_SPEC's size at TL_MIPMAP's page, TL_MODE's
    /// components and depth.
    pub fn load_line(&mut self, regs: &[u32], line: u32, bytes: &[u8]) {
        let mode = regs[reg::TL_MODE as usize];
        let spec = regs[reg::TL_SPEC as usize];
        let nc = ((mode >> 1) & 3) as usize + 1;
        let sel = ((mode >> 3) & 3) as usize;
        let d = depth_nibbles(mode >> 5);
        let (w, h) = (1usize << ((spec >> 4) & 0xF), 1usize << ((spec >> 8) & 0xF));
        let page = regs[reg::TL_MIPMAP as usize] & 0xFF;
        // The origin is signed: a GL border starts at -1. Border texels go
        // to the border page; anything further out is dropped.
        let border_page = regs[reg::TL_BORDER as usize] & 0xFF;
        let s0 = regs[reg::TL_S_LEFT as usize] as i32 as i64;
        let t = regs[reg::TL_T_BOTTOM as usize] as i32 as i64 + line as i64;
        let (wi, hi) = (w as i64, h as i64);
        if t < -1 || t > hi {
            return;
        }
        if line == 0 && w.max(h) > 16 {
            // A restore (the 64-wide page view) brings back what was saved:
            // whose it is, this diagnostic cannot tell.
            let n = (w * h * cell_nibbles(d)).div_ceil(PAGE_NIBBLES).min(64);
            let who = if spec & 0xFF0 == 0xD60 { 0 } else { page << 16 | (spec & 0xFF0) };
            for p in page as usize..(page as usize + n).min(256) {
                self.owner[p] = who;
            }
        }
        let tn = cell_nibbles(d);
        let co: [usize; 4] = std::array::from_fn(|c| comp_offset(sel, nc, c, d));
        // Texel 0 of this row, when the row is inside the level.
        let row = (0..hi).contains(&t).then(|| texel_addr(page, w, h, 0, t as usize, tn));
        for (k, px) in bytes.chunks_exact(4).enumerate() {
            let s = s0 + k as i64;
            if s < -1 || s > wi {
                continue;
            }
            let a = match row {
                Some(row) if (0..wi).contains(&s) => (row + s as usize * tn) % TRAM_NIBBLES,
                _ => border_addr(border_page, w, h, s, t, tn),
            };
            // A cell never straddles pages: cells are a power of two of
            // nibbles, pages hold a whole number of them.
            let page = a / PAGE_NIBBLES;
            self.dirty[page >> 6] |= 1 << (page & 63);
            if d == 2 {
                // Byte components: cells and component offsets are whole
                // bytes, low nibble first, so each component is one byte.
                for (c, &v) in px.iter().take(nc).enumerate() {
                    self.tram[(a + co[c]) % TRAM_NIBBLES / 2] = v;
                }
                self.texels_loaded += 1;
                continue;
            }
            for (c, &v) in px.iter().take(nc).enumerate() {
                let v = v as u32;
                let stored = match d {
                    1 => v >> 4,
                    3 => v << 4 | v >> 4,
                    _ => v,
                };
                for i in 0..d {
                    self.set_nibble((a + co[c] + i) % TRAM_NIBBLES, stored >> (4 * i));
                }
            }
            self.texels_loaded += 1;
        }
    }

    /// Bytes per texel a readback delivers for transfer mode `xfrmode`
    /// (traced, glGetTexImage per internal format): format 9 one
    /// component, 0xA two, 7 RGB, 8 RGBA; type 0 bytes, 1 16-bit.
    pub fn read_texel_bytes(xfrmode: u32) -> u32 {
        let n = match (xfrmode >> 4) & 0xF {
            9 => 1,
            0xA => 2,
            7 => 3,
            _ => 4,
        };
        n * if xfrmode & 0xF == 1 { 2 } else { 1 }
    }

    /// Row `line` of the texture manager's page view (its save of TRAM
    /// pages): from the page of level 0 in the TXMIPMAP table, `w` texels
    /// a row in a level `w` wide, of the components (TEXMODE1) and depth
    /// (TEXMODE2) of the texture being saved. Each texel is its components
    /// in storage order, as many as the transfer mode's format carries
    /// (traced: RGB textures 0x400070, three bytes; RGBA 0x400080, four),
    /// 8 bits each or 16 for type 1. The restore (TL_MODE the same
    /// components and depth, TL_SPEC as wide) stores them back.
    pub fn read_raw(&self, regs: &[u32], line: u32, w: u32) -> Vec<u8> {
        let nc = ((regs[reg::TEXMODE1 as usize] >> 3) & 3) as usize + 1;
        let d = depth_nibbles(regs[reg::TEXMODE2 as usize] >> 2);
        let sel = ((regs[reg::TEXMODE1 as usize] >> 7) & 3) as usize;
        let xfrmode = regs[0x159];
        let wide = xfrmode & 0xF == 1;
        let n = Self::read_texel_bytes(xfrmode) as usize / if wide { 2 } else { 1 };
        let page = self.tables[0][0] & 0xFF;
        let mut out = Vec::with_capacity(w as usize * n * 2);
        for s in 0..w as usize {
            let a = texel_addr(page, w as usize, 1 << 13, s, line as usize, cell_nibbles(d));
            for c in 0..n {
                let o = comp_offset(sel, nc, c, d);
                let v = if c < nc { (0..d).fold(0, |v, i| v | self.nibble((a + o + i) % TRAM_NIBBLES) << (4 * i)) } else { 0 };
                // As 16 bits, then the top 8 for a byte.
                let v16 = match d {
                    1 => v * 0x1111,
                    2 => v * 0x101,
                    _ => v << 4 | v >> 8,
                } as u16;
                if wide {
                    out.extend(v16.to_be_bytes());
                } else {
                    out.push((v16 >> 8) as u8);
                }
            }
        }
        out
    }

    /// Row `line` of the level the sampler is aimed at (level 0 of the
    /// TXMIPMAP table, TXSIZE's size), `w` texels in the transfer mode's
    /// layout (XFRMODE): the texture's own components (one: luminance,
    /// alpha or intensity; two: luminance-alpha; RGB), or RGBA with the
    /// class's components where GL's readback puts them (luminance and
    /// intensity in red, alpha in alpha, 1 where absent); 8 or 16 bits.
    pub fn read_line(&self, regs: &[u32], line: u32, w: u32) -> Vec<u8> {
        let smp = self.sampler(regs);
        let class = (regs[reg::TEXMODE1 as usize] >> 9) & 7;
        let xfrmode = regs[0x159];
        let n = match (xfrmode >> 4) & 0xF {
            9 => 1,
            0xA => 2,
            7 => 3,
            _ => 4,
        };
        let wide = xfrmode & 0xF == 1;
        let mut out = Vec::with_capacity((w * Self::read_texel_bytes(xfrmode)) as usize);
        for k in 0..w {
            // 12-bit components: 16 bits by repeating them, bytes their top.
            let c = smp.texel(self, 0, k as i64, line as i64);
            let rgba = match class {
                1 => [0, 0, 0, c[0]],
                2 => [c[0], 0, 0, if smp.nc == 2 { c[1] } else { 0xFFF }],
                3 => [c[0], 0, 0, 0xFFF],
                _ => [c[0], c[1], c[2], if smp.nc == 4 { c[3] } else { 0xFFF }],
            };
            // Two 8-bit components travel as one 16-bit unit, alpha in the
            // high byte (traced: libGLcore reads LA8 as A, L; LA12 as L, A).
            let comps = match n {
                4 => rgba,
                2 if !wide => [c[1], c[0], 0, 0],
                _ => c,
            };
            for &v in comps.iter().take(n) {
                if wide {
                    out.extend(((v << 4 | v >> 8) as u16).to_be_bytes());
                } else {
                    out.push((v >> 4) as u8);
                }
            }
        }
        out
    }

    /// Diagnostic: level 0 of the bound texture when its page was last
    /// loaded with another level ((page, owner) of that load), else None.
    pub fn stale(&self, regs: &[u32]) -> Option<(u32, u32)> {
        let size = regs[reg::TXSIZE as usize];
        let (ls, lt) = (size & 0xF, (size >> 4) & 0xF);
        if ls.max(lt) <= 4 {
            return None;
        }
        let page = self.tables[0][0] & 0xFF;
        let want = page << 16 | lt << 8 | ls << 4;
        let have = self.owner[page as usize];
        (have != 0 && have != want).then_some((want, have))
    }

    /// The sampler for the registers as they stand (once per primitive).
    pub fn sampler(&self, regs: &[u32]) -> Sampler {
        let m2 = regs[reg::TEXMODE2 as usize];
        let m1 = regs[reg::TEXMODE1 as usize];
        let size = regs[reg::TXSIZE as usize];
        let mut pages = [0u32; 16];
        let mut border_pages = [0u32; 16];
        for (k, p) in pages.iter_mut().enumerate() {
            *p = self.tables[0][k] & 0xFF;
            border_pages[k] = self.tables[1][k] & 0xFF;
        }
        // The border colour as the texel class's components (12 bits).
        let c = |v: u32| v & 0xFFF;
        let (rg, ba) = (regs[reg::TXBCOLOR_RG as usize], regs[reg::TXBCOLOR_BA as usize]);
        let (r, g, b, a) = (c(rg), c(rg >> 12), c(ba), c(ba >> 12));
        let border_color = match (m1 >> 9) & 7 {
            1 => [a, 0, 0, 0],
            2 => [r, a, 0, 0],
            3 => [r, 0, 0, 0],
            _ => [r, g, b, a],
        };
        let mipmaps = m2 & TM2_MM_ENABLE != 0 && m2 & TM2_MIPMAP != 0;
        Sampler {
            mode2: m2,
            ls: size & 0xF,
            lt: (size >> 4) & 0xF,
            max_level: if mipmaps { (regs[reg::TXLOD as usize] & 0xF).min(15) } else { 0 },
            nc: ((m1 >> 3) & 3) + 1,
            sel: (m1 >> 7) & 3,
            d: depth_nibbles(m2 >> 2) as u32,
            pages,
            border_pages,
            border_color,
            detail: self.detail(regs, size),
        }
    }

    /// The detail texture's parameters, or all zero when it is off.
    fn detail(&self, regs: &[u32], size: u32) -> Detail {
        let m2 = regs[reg::TEXMODE2 as usize];
        if m2 & TM2_DETAIL == 0 {
            return Detail::default();
        }
        let log = if m2 & TM2_DETAIL_128 != 0 { 7 } else { 8 };
        // TXSIZE 18:16 = 6 + L: detail texels per level-0 texel 2^(6 - D).
        let up = 6 - ((size >> 16) & 7) as i32;
        let mut table = [0u32; TABLE_LEN];
        let last = ((regs[reg::TXLOD as usize] >> 10) & 0x3F) as usize / 2;
        for (k, w) in table.iter_mut().enumerate().take(last + 1) {
            *w = self.tables[2][k] & 0x1F;
        }
        Detail {
            on: 1,
            page: regs[reg::TXDETAIL as usize] & 0xFF,
            log,
            shift_s: ((size & 0xF) as i32 + up).max(0) as u32,
            shift_t: (((size >> 4) & 0xF) as i32 + up).max(0) as u32,
            pad: [0; 3],
            table,
        }
    }
}

/// `round(256 * 2^(f / 256))` for an 8-bit fraction `f`: the detail
/// weight's table index from the LOD (`Sampler::detail_weight`). The GPU
/// unit uploads the same table.
pub fn exp2_q8() -> &'static [u32; 256] {
    static T: std::sync::LazyLock<[u32; 256]> =
        std::sync::LazyLock::new(|| std::array::from_fn(|f| (256.0 * (f as f64 / 256.0).exp2()).round() as u32));
    &T
}

/// A primitive's detail texture (`Te1::detail`). Plain data.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Detail {
    /// 1 when on.
    pub on: u32,
    /// TRAM page of the 2N x N level holding the image.
    pub page: u32,
    /// log2 N.
    pub log: u32,
    /// log2 of the detail image's size over the base texture's [0, 1]
    /// (log2 of level 0's size - L).
    pub shift_s: u32,
    pub shift_t: u32,
    pub pad: [u32; 3],
    /// Five-bit weights, zero past the last one used.
    pub table: [u32; TABLE_LEN],
}

/// What the TE needs to sample one primitive's fragments. Plain data: the
/// JIT (`rss_jit`) reads it by offset.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Sampler {
    pub mode2: u32,
    /// log2 of level 0's size.
    pub ls: u32,
    pub lt: u32,
    pub max_level: u32,
    pub nc: u32,
    /// Select slot (TEXMODE1 bits 8:7, see `comp_offset`).
    pub sel: u32,
    /// Component depth in nibbles.
    pub d: u32,
    pub pages: [u32; 16],
    pub border_pages: [u32; 16],
    /// 12-bit components.
    pub border_color: [u32; 4],
    /// The detail texture (the JIT reads it by offset too).
    pub detail: Detail,
}

impl Sampler {
    /// Level 0's size in texels.
    pub fn size(&self) -> (f64, f64) {
        ((1u32 << self.ls) as f64, (1u32 << self.lt) as f64)
    }

    /// Texel (i, j) of `level`, its components widened to 12 bits: wrapped
    /// (repeat), or outside the level (clamp) the border texel or the
    /// border colour.
    fn texel(&self, te: &Te1, level: u32, i: i64, j: i64) -> [u32; 4] {
        let w = 1usize << self.ls.saturating_sub(level);
        let h = 1usize << self.lt.saturating_sub(level);
        let axis = |v: i64, n: usize, clamp: bool| if clamp { v.clamp(-1, n as i64) } else { v.rem_euclid(n as i64) };
        let s = axis(i, w, self.mode2 & TM2_CLAMP_S != 0);
        let t = axis(j, h, self.mode2 & TM2_CLAMP_T != 0);
        let (nc, sel, d) = (self.nc as usize, self.sel as usize, self.d as usize);
        let tn = cell_nibbles(d);
        let a = if s < 0 || t < 0 || s >= w as i64 || t >= h as i64 {
            if self.mode2 & TM2_NO_BORDER != 0 {
                return self.border_color;
            }
            border_addr(self.border_pages[level as usize & 15], w, h, s, t, tn)
        } else {
            texel_addr(self.pages[level as usize & 15], w, h, s as usize, t as usize, tn)
        };
        let mut c = [0u32; 4];
        for (k, o) in c.iter_mut().enumerate().take(nc) {
            *o = te.component(a + comp_offset(sel, nc, k, d), d);
        }
        c
    }

    /// A level, sampled at (s, t) (texture coordinates in Q31, 1.0 over
    /// the texture), nearest or bilinear, in 12.16. Per level the
    /// coordinate is texels in Q16: its integer part picks the texel, the
    /// fraction's top 8 bits are the bilinear weight, so the four products
    /// sum to 12.16 exactly.
    fn level(&self, te: &Te1, level: u32, s: i64, t: i64, linear: bool) -> [i32; 4] {
        let (lw, lh) = (self.ls.saturating_sub(level), self.lt.saturating_sub(level));
        let (w, h) = (1i64 << lw, 1i64 << lh);
        // GL_CLAMP: coordinates clamped to [0, 1]; nearest then stays in
        // the texture, linear reaches half a texel past its edge. Clamp to
        // border leaves them be: the texel index clamp reads border there.
        let gl_clamp = self.mode2 & TM2_GL_CLAMP != 0;
        let (cs, ct) = (self.mode2 & TM2_CLAMP_S != 0, self.mode2 & TM2_CLAMP_T != 0);
        let s = if gl_clamp && cs { s.clamp(0, 1 << 31) } else { s };
        let t = if gl_clamp && ct { t.clamp(0, 1 << 31) } else { t };
        let (u, v) = (s >> (15 - lw), t >> (15 - lh));
        if !linear {
            // At s = 1 exactly, clamped nearest stays on the last texel.
            let cap = |x: i64, n: i64, clamp: bool| if clamp && gl_clamp { (x >> 16).min(n - 1) } else { x >> 16 };
            let c = self.texel(te, level, cap(u, w, cs), cap(v, h, ct));
            return c.map(|v| (v << 16) as i32);
        }
        let (u, v) = (u.wrapping_sub(0x8000), v.wrapping_sub(0x8000));
        let (i, j) = (u >> 16, v >> 16);
        let (a, b) = (((u >> 8) & 0xFF) as i32, ((v >> 8) & 0xFF) as i32);
        // Wrapping: a coordinate far outside can sit at i64::MAX.
        let (i1, j1) = (i.wrapping_add(1), j.wrapping_add(1));
        let (t00, t10) = (self.texel(te, level, i, j), self.texel(te, level, i1, j));
        let (t01, t11) = (self.texel(te, level, i, j1), self.texel(te, level, i1, j1));
        let (w00, w10, w01, w11) = ((256 - a) * (256 - b), a * (256 - b), (256 - a) * b, a * b);
        let mut c = [0i32; 4];
        for k in 0..4 {
            c[k] = t00[k] as i32 * w00 + t10[k] as i32 * w10 + t01[k] as i32 * w01 + t11[k] as i32 * w11;
        }
        c
    }

    /// The filtered texel at (s, t) (Q31) for level of detail `lambda`
    /// (Q8 log2 of level-0 texels per pixel): magnification at or below 0,
    /// minification (with mipmaps when on) above.
    pub fn sample(&self, te: &Te1, s: i64, t: i64, lambda: i32) -> [i32; 4] {
        let m = self.mode2;
        if lambda <= 0 {
            let base = self.level(te, 0, s, t, m & TM2_MAG_LINEAR != 0);
            return if self.detail.on != 0 { self.detail(te, base, s, t, lambda) } else { base };
        }
        let linear = m & TM2_MIN_LINEAR != 0;
        if self.max_level == 0 {
            return self.level(te, 0, s, t, linear);
        }
        let top = self.max_level as i32;
        if m & TM2_MIP_LINEAR == 0 {
            // The nearest level: ceil(lambda + 0.5) - 1.
            let l = (((lambda + 128 + 255) >> 8) - 1).clamp(0, top);
            return self.level(te, l as u32, s, t, linear);
        }
        // Between the levels by an 8-bit fraction.
        let l0 = (lambda >> 8).clamp(0, top);
        let l1 = (l0 + 1).min(top);
        let f = (lambda - (l0 << 8)).clamp(0, 256) as i64;
        let (a, b) = (self.level(te, l0 as u32, s, t, linear), self.level(te, l1 as u32, s, t, linear));
        [0, 1, 2, 3].map(|k| a[k] + (((b[k] - a[k]) as i64 * f) >> 8) as i32)
    }

    /// The detail weight F at `lambda` (<= 0, Q8) in 12.16: the table at
    /// k = 32 2^lambda, its top 24 bits an index and an 8-bit fraction.
    pub fn detail_weight(&self, lambda: i32) -> i32 {
        let e = lambda.saturating_add(5 * 256);
        let x = exp2_q8()[(e & 0xFF) as usize];
        let x = if e >= 0 { x << (e >> 8).min(8) } else { x >> (-(e >> 8)).min(31) };
        let (i, f) = if x >= 31 << 8 { (31, 0) } else { ((x >> 8) as usize, x & 0xFF) };
        let tab = &self.detail.table;
        let t1 = if i < 31 { tab[i + 1] } else { 0 };
        let w = tab[i] * (256 - f) + t1 * f;
        // 31 * 256 is 1.0: 2^28 / 7936, rounded.
        (w * 33825) as i32
    }

    /// The detail texture added to the magnified base sample `base` (12.16)
    /// at (s, t) (Q31): the detail image, bilinear and repeating, at the
    /// base's level-0 coordinate scaled by 2^-L, out = base + F (2 detail -
    /// 1) per component, clamped.
    fn detail(&self, te: &Te1, base: [i32; 4], s: i64, t: i64, lambda: i32) -> [i32; 4] {
        use fixed::{mul, ONE};
        let d = &self.detail;
        let coord = |v: i64, sh: u32| if sh <= 15 { v >> (15 - sh) } else { v << (sh - 15) };
        let (u, v) = (coord(s, d.shift_s).wrapping_sub(0x8000), coord(t, d.shift_t).wrapping_sub(0x8000));
        let n = 1i64 << d.log;
        let (i, j) = ((u >> 16).rem_euclid(n), (v >> 16).rem_euclid(n));
        let (a, b) = (((u >> 8) & 0xFF) as i32, ((v >> 8) & 0xFF) as i32);
        let (i1, j1) = ((i + 1) & (n - 1), (j + 1) & (n - 1));
        let (nc, sel, dn) = (self.nc as usize, self.sel as usize, self.d as usize);
        let tn = cell_nibbles(dn);
        let (w, h) = (2 * n as usize, n as usize);
        let texel = |i: i64, j: i64| {
            let at = texel_addr(d.page, w, h, i as usize, j as usize, tn);
            let mut c = [0i32; 4];
            for (k, o) in c.iter_mut().enumerate().take(nc) {
                *o = te.component(at + comp_offset(sel, nc, k, dn), dn) as i32;
            }
            c
        };
        let (t00, t10, t01, t11) = (texel(i, j), texel(i1, j), texel(i, j1), texel(i1, j1));
        let (w00, w10, w01, w11) = ((256 - a) * (256 - b), a * (256 - b), (256 - a) * b, a * b);
        let f = self.detail_weight(lambda);
        let mut out = base;
        for k in 0..4 {
            let td = t00[k] * w00 + t10[k] * w10 + t01[k] * w01 + t11[k] * w11;
            out[k] = fixed::clamp(base[k].wrapping_add(mul(f, 2 * td - ONE)));
        }
        out
    }
}

/// The texture environment (RE4), in 12.16: fragment colour `f` and texel
/// components `tex`, per TEXMODE1 (environment mode, components, texel
/// class) and the environment colour (TXENV_RG: red 11:0, green 23:12;
/// TXENV_B: blue 11:0, alpha 23:12).
pub fn tex_env(mode1: u32, env_rg: u32, env_b: u32, f: [i32; 4], tex: [i32; 4]) -> [i32; 4] {
    use fixed::{mul, ONE};
    let nc = ((mode1 >> 3) & 3) + 1;
    // (texel colour, texel alpha), each if the format has it.
    let (ct, at) = match (mode1 >> 9) & 7 {
        1 => (None, Some(tex[0])),
        2 => (Some([tex[0]; 3]), (nc == 2).then_some(tex[1])),
        3 => (Some([tex[0]; 3]), Some(tex[0])),
        _ if nc >= 3 => (Some([tex[0], tex[1], tex[2]]), (nc == 4).then_some(tex[3])),
        _ => (Some([tex[0]; 3]), (nc == 2).then_some(tex[1])),
    };
    let cc = [fixed::field12(env_rg), fixed::field12(env_rg >> 12), fixed::field12(env_b)];
    let ac = fixed::field12(env_b >> 12);
    let intensity = (mode1 >> 9) & 7 == 3;
    let mut out = f;
    match (mode1 >> 1) & 3 {
        // Decal: RGB replaces; RGBA blends by the texel's alpha; alpha kept.
        1 => {
            if let Some(ct) = ct {
                let a = at.unwrap_or(ONE);
                for k in 0..3 {
                    out[k] = mul(f[k], ONE - a).wrapping_add(mul(ct[k], a));
                }
            }
        }
        // Blend: the texel blends the fragment colour toward the
        // environment colour.
        2 => {
            if let Some(ct) = ct {
                for k in 0..3 {
                    out[k] = mul(f[k], ONE - ct[k]).wrapping_add(mul(cc[k], ct[k]));
                }
            }
            if let Some(at) = at {
                out[3] = if intensity { mul(f[3], ONE - at).wrapping_add(mul(ac, at)) } else { mul(f[3], at) };
            }
        }
        // Alpha-only modulate (alpha textures).
        3 => {
            if let Some(at) = at {
                out[3] = mul(f[3], at);
            }
        }
        // Modulate (and GL_REPLACE, which libGLcore sends as modulate).
        _ => {
            if let Some(ct) = ct {
                for k in 0..3 {
                    out[k] = mul(f[k], ct[k]);
                }
            }
            if let Some(at) = at {
                out[3] = mul(f[3], at);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A detail texture as the driver sets one up (base 8x8 RGBA8, detail
    /// 256 at level -3, a constant weight of 1): the sample is the base
    /// plus 2 detail - 1, the detail image repeating at 2^3 times the
    /// base's texel rate, within three levels of the specification's
    /// doubles.
    #[test]
    fn detail_adds_the_scaled_detail_image() {
        // SAFETY: Te1 is plain data, valid zeroed.
        let mut te = unsafe { Box::<Te1>::new_zeroed().assume_init() };
        let mut regs = vec![0u32; 0x400];
        let mut seed = 0xDE7A_11u32;
        let mut rnd = |lo: u32, n: u32| {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (lo + (seed >> 16) % n) as u8
        };
        let base: Vec<u8> = (0..8 * 8 * 4).map(|_| rnd(48, 160)).collect();
        let det: Vec<u8> = (0..256 * 256 * 4).map(|_| rnd(0, 256)).collect();
        // RGBA, 8-bit components.
        regs[reg::TL_MODE as usize] = 3 << 1 | 1 << 5;
        regs[reg::TL_SPEC as usize] = 3 << 8 | 3 << 4;
        for t in 0..8 {
            te.load_line(&regs, t, &base[t as usize * 32..][..32]);
        }
        // The detail image: the left half of a 512x256 level at page 16.
        regs[reg::TL_SPEC as usize] = 8 << 8 | 9 << 4;
        regs[reg::TL_MIPMAP as usize] = 16;
        for t in 0..256 {
            te.load_line(&regs, t, &det[t as usize * 1024..][..1024]);
        }
        te.table_write(reg::TXMIPMAP, 0, 0);
        for k in 0..32 {
            te.table_write(reg::DETAILSCALE, k, 0x1F);
        }
        regs[reg::TEXMODE1 as usize] = TEXMODE1_ENABLE | 3 << 3 | 4 << 9;
        regs[reg::TEXMODE2 as usize] = 1 << 2 | TM2_MAG_LINEAR | TM2_DETAIL;
        regs[reg::TXSIZE as usize] = 3 << 16 | 3 << 4 | 3;
        regs[reg::TXLOD as usize] = 62 << 10;
        regs[reg::TXDETAIL as usize] = 16;
        let smp = te.sampler(&regs);
        assert_eq!(smp.detail_weight(-3 * 256), fixed::ONE - 256);
        let bilinear = |img: &[u8], n: usize, u: f64, v: f64, c: usize| {
            let (fu, fv) = (u - 0.5, v - 0.5);
            let (a, b) = (fu - fu.floor(), fv - fv.floor());
            let w = |i: f64| (i as i64).rem_euclid(n as i64) as usize;
            let (i0, j0) = (w(fu.floor()), w(fv.floor()));
            let (i1, j1) = ((i0 + 1) % n, (j0 + 1) % n);
            let at = |i: usize, j: usize| img[(j * n + i) * 4 + c] as f64 / 255.0;
            (1.0 - a) * (1.0 - b) * at(i0, j0) + a * (1.0 - b) * at(i1, j0) + (1.0 - a) * b * at(i0, j1) + a * b * at(i1, j1)
        };
        let mut worst = 0.0f64;
        for k in 0..400 {
            let (x, y) = ((k * 37 % 400) as f64 / 400.0 + 0.0013, (k * 91 % 400) as f64 / 400.0 + 0.0007);
            let (s, t) = ((x * 2f64.powi(31)) as i64, (y * 2f64.powi(31)) as i64);
            let got = smp.sample(&te, s, t, -3 * 256);
            for c in 0..4 {
                let tb = bilinear(&base, 8, x * 8.0, y * 8.0, c);
                let td = bilinear(&det, 256, x * 64.0, y * 64.0, c);
                let want = (tb + 2.0 * td - 1.0).clamp(0.0, 1.0) * 255.0;
                worst = worst.max((fixed::to_byte(got[c]) as f64 - want).abs());
            }
        }
        // 8-bit filter weights, and the detail term counts twice.
        assert!(worst <= 3.0, "off by {worst}");
        // Detail off (and minification): the base alone.
        let plain = Sampler { detail: Detail::default(), ..smp };
        assert_eq!(smp.sample(&te, 1 << 29, 1 << 29, 64), plain.sample(&te, 1 << 29, 1 << 29, 64));
    }

    /// A 64x128 RGB8 level takes two pages (32-bit texels), so the page
    /// view of its second page starts at texel row 64: libGLcore saves a
    /// texture page by page, and a run that starts at a page must find
    /// the level's texels there.
    #[test]
    fn rgb8_level_fills_pages_as_on_the_board() {
        // SAFETY: Te1 is plain data, valid zeroed.
        let mut te = unsafe { Box::<Te1>::new_zeroed().assume_init() };
        let mut regs = vec![0u32; 0x400];
        regs[reg::TL_MODE as usize] = 0x34E24;
        regs[reg::TL_SPEC as usize] = 7 << 8 | 6 << 4;
        for t in 0..128u32 {
            regs[reg::TL_T_BOTTOM as usize] = 0;
            let line: Vec<u8> = (0..64u32).flat_map(|s| [s as u8, t as u8, 0xA5, 0]).collect();
            te.load_line(&regs, t, &line);
        }
        regs[reg::TEXMODE1 as usize] = 2 << 3;
        regs[reg::TEXMODE2 as usize] = 1 << 2;
        te.table_write(reg::TXMIPMAP, 0, 1);
        let row = te.read_raw(&regs, 6, 64);
        assert_eq!(&row[..8], &[0, 70, 0xA5, 0, 1, 70, 0xA5, 0]);
    }

    /// `load_line` stores byte components a byte at a time: the same TRAM
    /// as storing every component nibble by nibble, for small and large
    /// levels, borders, and one- and two-component textures in each slot.
    #[test]
    fn byte_component_loads_match_nibble_stores() {
        fn by_nibbles(te: &mut Te1, regs: &[u32], line: u32, bytes: &[u8]) {
            let mode = regs[reg::TL_MODE as usize];
            let spec = regs[reg::TL_SPEC as usize];
            let (nc, sel, d) = (((mode >> 1) & 3) as usize + 1, ((mode >> 3) & 3) as usize, depth_nibbles(mode >> 5));
            let (w, h) = (1usize << ((spec >> 4) & 0xF), 1usize << ((spec >> 8) & 0xF));
            let s0 = regs[reg::TL_S_LEFT as usize] as i32 as i64;
            let t = regs[reg::TL_T_BOTTOM as usize] as i32 as i64 + line as i64;
            let (page, border) = (regs[reg::TL_MIPMAP as usize] & 0xFF, regs[reg::TL_BORDER as usize] & 0xFF);
            for (k, px) in bytes.chunks_exact(4).enumerate() {
                let s = s0 + k as i64;
                if s < -1 || s > w as i64 || t < -1 || t > h as i64 {
                    continue;
                }
                let a = if s < 0 || t < 0 || s >= w as i64 || t >= h as i64 {
                    border_addr(border, w, h, s, t, cell_nibbles(d))
                } else {
                    texel_addr(page, w, h, s as usize, t as usize, cell_nibbles(d))
                };
                for (c, &v) in px.iter().take(nc).enumerate() {
                    for i in 0..d {
                        te.set_nibble((a + comp_offset(sel, nc, c, d) + i) % TRAM_NIBBLES, (v as u32) >> (4 * i));
                    }
                }
            }
        }
        // SAFETY: Te1 is plain data, valid zeroed.
        let mut fast = unsafe { Box::<Te1>::new_zeroed().assume_init() };
        let mut slow = unsafe { Box::<Te1>::new_zeroed().assume_init() };
        let mut regs = vec![0u32; 0x400];
        let mut seed = 0x1234_5678u32;
        for (lw, lh, nc, sel, s0, page) in [(6, 7, 4, 0, 0, 3), (3, 2, 4, 3, -1, 255), (0, 0, 3, 0, -1, 9), (8, 8, 1, 2, 0, 40), (4, 4, 2, 1, -1, 77)] {
            regs[reg::TL_MODE as usize] = (nc - 1) << 1 | sel << 3 | 1 << 5;
            regs[reg::TL_SPEC as usize] = lh << 8 | lw << 4;
            regs[reg::TL_MIPMAP as usize] = page;
            regs[reg::TL_BORDER as usize] = page + 1;
            regs[reg::TL_S_LEFT as usize] = s0 as u32;
            regs[reg::TL_T_BOTTOM as usize] = s0 as u32;
            let (w, h) = (1usize << lw, 1usize << lh);
            for line in 0..h as u32 + 2 {
                let bytes: Vec<u8> = (0..4 * (w + 2))
                    .map(|_| {
                        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                        (seed >> 16) as u8
                    })
                    .collect();
                fast.load_line(&regs, line, &bytes);
                by_nibbles(&mut slow, &regs, line, &bytes);
            }
        }
        assert!(fast.tram == slow.tram);
    }
}
