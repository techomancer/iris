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
//! Not yet: detail and sharpen textures, 3D textures, texture lookup tables,
//! video texture.

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
    /// Diagnostic: stale samples seen, and the last one's (want, have).
    pub stale_events: u32,
    pub stale_last: [u32; 2],
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
        for (k, px) in bytes.chunks_exact(4).enumerate() {
            let s = s0 + k as i64;
            if s < -1 || s > wi {
                continue;
            }
            let a = if s < 0 || t < 0 || s >= wi || t >= hi {
                border_addr(border_page, w, h, s, t, cell_nibbles(d))
            } else {
                texel_addr(page, w, h, s as usize, t as usize, cell_nibbles(d))
            };
            for (c, &v) in px.iter().take(nc).enumerate() {
                let v = v as u32;
                let stored = match d {
                    1 => v >> 4,
                    3 => v << 4 | v >> 4,
                    _ => v,
                };
                for i in 0..d {
                    self.set_nibble((a + comp_offset(sel, nc, c, d) + i) % TRAM_NIBBLES, stored >> (4 * i));
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
        }
    }
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
            return self.level(te, 0, s, t, m & TM2_MAG_LINEAR != 0);
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
}
