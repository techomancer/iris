//! The pipeline key: everything a raster shader is monomorphised on.
//!
//! A key names a primitive's whole pixel pipeline: which primitive walks the
//! pixels, how they are clipped, where they are written and in what format,
//! and (for GL primitives) the per-fragment tests, blending, texturing and
//! fog. Data the pipeline only reads — colours, masks, reference values,
//! page pointers, texture base pages, plane coefficients — is not in it:
//! that goes to the shader in its context (`ctx::RasterCtx`).
//!
//! Keys are normalised: a field that cannot affect the result in the rest
//! of the configuration is zeroed (blend factors with blending off, texture
//! filters with texturing off, codes that behave alike folded together), so
//! two configurations that draw the same share a shader.
//!
//! The packed form is mixed radix, not bit fields. 2D primitives and GL
//! primitives carry disjoint field sets, selected by the primitive digit, and
//! optional groups take one digit (0 = off, else 1 + the group's value).
//! The largest key, a GL one with every group on, needs under 59 bits
//! (`packed_keys_fit_in_64_bits`).

use std::fmt;

/// The primitive whose pixel loop the shader runs.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Prim {
    /// A block filled in one colour.
    #[default]
    Fill = 0,
    /// An X line (Bresenham, optional 32-bit stipple).
    Line = 1,
    /// One chunk of character stipple bits into a block.
    Stipple = 2,
    /// One line of a host pixel transfer into a block.
    Xfer = 3,
    /// A GL triangle (area).
    Tri = 4,
    /// A GL line.
    GlLine = 5,
}

impl Prim {
    pub const ALL: [Prim; 6] = [Prim::Fill, Prim::Line, Prim::Stipple, Prim::Xfer, Prim::Tri, Prim::GlLine];

    pub fn is_gl(self) -> bool {
        matches!(self, Prim::Tri | Prim::GlLine)
    }
}

/// Where pixels go. `Single` is one 36-bit buffer (A or B: which page and
/// which write mask is context data), `Dual` both main buffers (A then B,
/// when they are distinct pages), `Overlay` the 9-bit overlay planes,
/// `Cid` the clip-ID planes.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Draw {
    #[default]
    Single = 0,
    Dual = 1,
    Overlay = 2,
    Cid = 3,
}

impl Draw {
    pub const ALL: [Draw; 4] = [Draw::Single, Draw::Dual, Draw::Overlay, Draw::Cid];
}

/// How a 36-bit buffer write treats the value (PP1 pixel type): kept as is,
/// merged under the RGBA8888 plane masks (`Rgba8`: type 2), under the
/// 12-bit colour-index mask (`Ci12`: type 6), or as a 12-bit pixel pair
/// (`rss::rgb12_pair`: types 0 and 1) whose blending reads buffer A's half
/// (`Rgb12`) or B's (`Rgb12B`, draw field 2), each also dithered (`..D`).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Pix {
    #[default]
    Plain = 0,
    Rgb12 = 1,
    Rgba8 = 2,
    Ci12 = 3,
    Rgb12B = 4,
    Rgb12D = 5,
    Rgb12BD = 6,
}

impl Pix {
    pub const ALL: [Pix; 7] = [Pix::Plain, Pix::Rgb12, Pix::Rgba8, Pix::Ci12, Pix::Rgb12B, Pix::Rgb12D, Pix::Rgb12BD];

    /// A 12-bit pixel pair write.
    pub fn pair(self) -> bool {
        matches!(self, Pix::Rgb12 | Pix::Rgb12B | Pix::Rgb12D | Pix::Rgb12BD)
    }

    /// Blending reads buffer B's half (bits 23:12).
    pub fn pair_b(self) -> bool {
        matches!(self, Pix::Rgb12B | Pix::Rgb12BD)
    }

    pub fn dither(self) -> bool {
        matches!(self, Pix::Rgb12D | Pix::Rgb12BD)
    }

    /// The pair variant for half B or A, dithered or not.
    pub fn of_pair(b: bool, dither: bool) -> Pix {
        match (b, dither) {
            (false, false) => Pix::Rgb12,
            (true, false) => Pix::Rgb12B,
            (false, true) => Pix::Rgb12D,
            (true, true) => Pix::Rgb12BD,
        }
    }
}

/// Host transfer pixel formats (`rss::from_host`), by (PixelFormat,
/// CompType): raw (the bytes as they are), 12-bit index (0, 1), RGBA4
/// (8, 8), RGB5 (8, 10), RGB8 (7, 0), RGB16 (7, 1), RGBA16 (8, 1), depth
/// (2, 3).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum XFmt {
    #[default]
    Raw = 0,
    Ci12 = 1,
    Rgba4 = 2,
    Rgb5 = 3,
    Rgb8 = 4,
    Rgb16 = 5,
    Rgba16 = 6,
    Depth = 7,
}

impl XFmt {
    pub const ALL: [XFmt; 8] = [XFmt::Raw, XFmt::Ci12, XFmt::Rgba4, XFmt::Rgb5, XFmt::Rgb8, XFmt::Rgb16, XFmt::Rgba16, XFmt::Depth];

    /// The format of an XFRMODE (PixelFormat bits 7:4, CompType 3:0).
    pub fn of(xfrmode: u32) -> XFmt {
        match ((xfrmode >> 4) & 0xF, xfrmode & 0xF) {
            (0, 1) => XFmt::Ci12,
            (8, 8) => XFmt::Rgba4,
            (8, 10) => XFmt::Rgb5,
            (7, 0) => XFmt::Rgb8,
            (7, 1) => XFmt::Rgb16,
            (8, 1) => XFmt::Rgba16,
            (2, 3) => XFmt::Depth,
            _ => XFmt::Raw,
        }
    }
}

/// Bytes per transfer pixel the key can name.
pub const XBPP: [u8; 6] = [1, 2, 3, 4, 6, 8];

/// Stencil test and update: compare function (OpenGL order), and the ops
/// on stencil fail, depth fail and depth pass (0 KEEP, 1 ZERO, 2 REPLACE,
/// 3 INCR, 4 DECR, 5 INVERT; the hardware codes 6 and 7 keep too).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Stencil {
    pub func: u8,
    pub fail: u8,
    pub zfail: u8,
    pub zpass: u8,
}

/// Blend factors, 0..=10 in BlendFactor's order (codes above 10 act as
/// ONE and are folded into 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Blend {
    pub src: u8,
    pub dst: u8,
}

/// Texturing: the environment (TEXMODE1 bits 2:1), components (1..=4),
/// select slot for one- and two-component textures, texel class (1 alpha,
/// 2 luminance, 3 intensity, 0 the rest), filters and wrap modes
/// (TEXMODE2), and component depth in nibbles (1..=3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Tex {
    pub env: u8,
    pub nc: u8,
    pub slot: u8,
    pub class: u8,
    pub mag_linear: bool,
    pub min_linear: bool,
    pub mipmap: bool,
    pub mip_linear: bool,
    pub clamp_s: bool,
    pub clamp_t: bool,
    pub gl_clamp: bool,
    pub no_border: bool,
    pub depth: u8,
}

impl Tex {
    /// Fold settings that draw alike.
    pub fn normalized(mut self) -> Tex {
        self.env &= 3;
        self.nc = self.nc.clamp(1, 4);
        self.slot = match self.nc {
            1 => self.slot & 3,
            2 => self.slot & 1,
            _ => 0,
        };
        if !matches!(self.class, 1..=3) {
            self.class = 0;
        }
        if !self.mipmap {
            self.mip_linear = false;
        }
        if !(self.clamp_s || self.clamp_t) {
            self.gl_clamp = false;
            self.no_border = false;
        }
        self.depth = self.depth.clamp(1, 3);
        self
    }

    fn code(&self) -> u64 {
        let ncslot = match self.nc {
            1 => self.slot,
            2 => 4 + self.slot,
            3 => 6,
            _ => 7,
        } as u64;
        let flags = [
            self.mag_linear,
            self.min_linear,
            self.mipmap,
            self.mip_linear,
            self.clamp_s,
            self.clamp_t,
            self.gl_clamp,
            self.no_border,
        ]
        .iter()
        .enumerate()
        .fold(0u64, |a, (i, &f)| a | (f as u64) << i);
        let mut e = Enc::default();
        e.put(self.env as u64, 4);
        e.put(ncslot, 8);
        e.put(self.class as u64, 4);
        e.put(flags, 256);
        e.put(self.depth as u64 - 1, 3);
        e.v
    }

    fn from_code(c: u64) -> Tex {
        let mut d = Dec(c);
        let env = d.get(4) as u8;
        let ncslot = d.get(8) as u8;
        let class = d.get(4) as u8;
        let flags = d.get(256);
        let depth = d.get(3) as u8 + 1;
        let (nc, slot) = match ncslot {
            0..=3 => (1, ncslot),
            4 | 5 => (2, ncslot - 4),
            6 => (3, 0),
            _ => (4, 0),
        };
        let f = |i: u32| flags & (1 << i) != 0;
        Tex {
            env,
            nc,
            slot,
            class,
            mag_linear: f(0),
            min_linear: f(1),
            mipmap: f(2),
            mip_linear: f(3),
            clamp_s: f(4),
            clamp_t: f(5),
            gl_clamp: f(6),
            no_border: f(7),
            depth,
        }
    }

    const RADIX: u64 = 4 * 8 * 4 * 256 * 3;
}

/// A raster pipeline configuration. Build one, `normalized()` it, `pack()`
/// it; `unpack` gives the normalised key back.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct PipeKey {
    pub prim: Prim,
    /// Clip-ID test on (`pp1winmode` CIDmatch nonzero).
    pub cid_test: bool,
    /// Enabled screen masks, packed into the context's first `nclip` slots.
    pub nclip: u8,
    pub draw: Draw,
    pub pix: Pix,
    /// The PP1 logic op, when enabled (0..=15, X11 function numbers).
    pub logic: Option<u8>,
    /// Line stipple (X and GL lines) or polygon stipple (triangles).
    pub stipple: bool,
    // ── 2D primitives ──
    /// Stippled X lines and character stipple: 0 bits draw the background.
    pub opaque: bool,
    /// X lines: CapNotLast.
    pub skip_last: bool,
    pub xfmt: XFmt,
    /// Transfer bytes per pixel (one of `XBPP`).
    pub xbpp: u8,
    // ── GL primitives ──
    /// The pixels are RGB (`Rss::rgb_mode`), not colour indices.
    pub rgb: bool,
    /// Alpha test compare function, when enabled and not ALWAYS.
    pub alpha: Option<u8>,
    pub stencil: Option<Stencil>,
    /// Depth test compare function, when enabled.
    pub z: Option<u8>,
    pub blend: Option<Blend>,
    pub tex: Option<Tex>,
    pub fog: bool,
}

/// Mixed-radix encoder: each `put` adds a digit below `radix`.
#[derive(Default)]
struct Enc {
    v: u64,
    m: u64,
}

impl Enc {
    fn put(&mut self, x: u64, radix: u64) {
        debug_assert!(x < radix, "key digit {x} out of radix {radix}");
        if self.m == 0 {
            self.m = 1;
        }
        self.v += x * self.m;
        self.m *= radix;
    }
}

struct Dec(u64);

impl Dec {
    fn get(&mut self, radix: u64) -> u64 {
        let x = self.0 % radix;
        self.0 /= radix;
        x
    }
}

impl PipeKey {
    /// Zero what cannot matter in this configuration.
    pub fn normalized(mut self) -> PipeKey {
        let gl = self.prim.is_gl();
        self.nclip = self.nclip.min(4);
        if self.prim == Prim::Xfer && self.xfmt == XFmt::Depth {
            // Depth goes to the depth buffer, whatever the draw state.
            self.draw = Draw::Single;
            self.pix = Pix::Plain;
            self.logic = None;
        }
        match self.draw {
            // Clip IDs: only the planes and their write mask.
            Draw::Cid => {
                self.pix = Pix::Plain;
                self.logic = None;
            }
            // The pixel-type write rules apply to 36-bit buffers alone.
            Draw::Overlay => self.pix = Pix::Plain,
            _ => {}
        }
        if let Some(op) = self.logic.as_mut() {
            *op &= 0xF;
        }
        if !matches!(self.prim, Prim::Line | Prim::Tri | Prim::GlLine) {
            self.stipple = false;
        }
        let line_opaque = self.prim == Prim::Line && self.stipple;
        if !(line_opaque || self.prim == Prim::Stipple) {
            self.opaque = false;
        }
        if self.prim != Prim::Line {
            self.skip_last = false;
        }
        if self.prim == Prim::Xfer {
            if !XBPP.contains(&self.xbpp) {
                self.xbpp = 1;
            }
        } else {
            self.xfmt = XFmt::Raw;
            self.xbpp = 1;
        }
        if gl {
            if self.alpha == Some(7) {
                self.alpha = None;
            }
            if let Some(a) = self.alpha.as_mut() {
                *a &= 7;
            }
            if let Some(z) = self.z.as_mut() {
                *z &= 7;
            }
            if let Some(s) = self.stencil.as_mut() {
                let op = |o: u8| if o > 5 { 0 } else { o };
                *s = Stencil { func: s.func & 7, fail: op(s.fail), zfail: op(s.zfail), zpass: op(s.zpass) };
            }
            // A logic op other than COPY replaces blending; colour-index
            // pixels never blend.
            if !self.rgb || self.logic.is_some_and(|op| op != 3) {
                self.blend = None;
            }
            if let Some(b) = self.blend.as_mut() {
                let f = |c: u8| if c > 10 { 1 } else { c };
                *b = Blend { src: f(b.src), dst: f(b.dst) };
            }
            self.tex = self.tex.map(Tex::normalized);
        } else {
            self.rgb = false;
            self.alpha = None;
            self.stencil = None;
            self.z = None;
            self.blend = None;
            self.tex = None;
            self.fog = false;
        }
        self
    }

    /// The packed key (of the normalised configuration).
    pub fn pack(&self) -> u64 {
        let k = self.normalized();
        let mut e = Enc::default();
        e.put(k.prim as u64, 6);
        e.put(k.cid_test as u64, 2);
        e.put(k.nclip as u64, 5);
        e.put(k.draw as u64, 4);
        e.put(k.pix as u64, Pix::ALL.len() as u64);
        e.put(k.logic.map_or(0, |op| op as u64 + 1), 17);
        e.put(k.stipple as u64, 2);
        if k.prim.is_gl() {
            e.put(k.rgb as u64, 2);
            e.put(k.alpha.map_or(0, |f| f as u64 + 1), 9);
            e.put(
                k.stencil.map_or(0, |s| {
                    1 + s.func as u64 + 8 * (s.fail as u64 + 6 * (s.zfail as u64 + 6 * s.zpass as u64))
                }),
                1 + 8 * 216,
            );
            e.put(k.z.map_or(0, |f| f as u64 + 1), 9);
            e.put(k.blend.map_or(0, |b| 1 + b.src as u64 + 11 * b.dst as u64), 1 + 121);
            e.put(k.tex.map_or(0, |t| 1 + t.code()), 1 + Tex::RADIX);
            e.put(k.fog as u64, 2);
        } else {
            e.put(k.opaque as u64, 2);
            e.put(k.skip_last as u64, 2);
            e.put(k.xfmt as u64, 8);
            e.put(XBPP.iter().position(|&b| b == k.xbpp).unwrap_or(0) as u64, 6);
        }
        e.v
    }

    pub fn unpack(v: u64) -> PipeKey {
        let mut d = Dec(v);
        let prim = Prim::ALL[d.get(6) as usize];
        let mut k = PipeKey {
            prim,
            cid_test: d.get(2) != 0,
            nclip: d.get(5) as u8,
            draw: Draw::ALL[d.get(4) as usize],
            pix: Pix::ALL[d.get(Pix::ALL.len() as u64) as usize],
            logic: match d.get(17) {
                0 => None,
                n => Some(n as u8 - 1),
            },
            stipple: d.get(2) != 0,
            xbpp: 1,
            ..PipeKey::default()
        };
        if prim.is_gl() {
            k.rgb = d.get(2) != 0;
            k.alpha = match d.get(9) {
                0 => None,
                n => Some(n as u8 - 1),
            };
            k.stencil = match d.get(1 + 8 * 216) {
                0 => None,
                n => {
                    let n = n - 1;
                    let (func, ops) = (n % 8, n / 8);
                    Some(Stencil { func: func as u8, fail: (ops % 6) as u8, zfail: (ops / 6 % 6) as u8, zpass: (ops / 36) as u8 })
                }
            };
            k.z = match d.get(9) {
                0 => None,
                n => Some(n as u8 - 1),
            };
            k.blend = match d.get(122) {
                0 => None,
                n => Some(Blend { src: ((n - 1) % 11) as u8, dst: ((n - 1) / 11) as u8 }),
            };
            k.tex = match d.get(1 + Tex::RADIX) {
                0 => None,
                n => Some(Tex::from_code(n - 1)),
            };
            k.fog = d.get(2) != 0;
        } else {
            k.opaque = d.get(2) != 0;
            k.skip_last = d.get(2) != 0;
            k.xfmt = XFmt::ALL[d.get(8) as usize];
            k.xbpp = XBPP[d.get(6) as usize];
        }
        k
    }

    /// The radix product of the largest key (a GL key with every optional
    /// group on): the packed form must stay below it.
    pub fn max_packed() -> u128 {
        let common: u128 = 6 * 2 * 5 * 4 * Pix::ALL.len() as u128 * 17 * 2;
        let gl: u128 = 2 * 9 * (1 + 8 * 216) * 9 * 122 * (1 + Tex::RADIX as u128) * 2;
        let d2: u128 = 2 * 2 * 8 * 6;
        common * gl.max(d2)
    }
}

impl fmt::Display for PipeKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let k = self.normalized();
        write!(f, "{:?} draw={:?} pix={:?}", k.prim, k.draw, k.pix)?;
        if k.cid_test {
            write!(f, " cid")?;
        }
        if k.nclip > 0 {
            write!(f, " clip{}", k.nclip)?;
        }
        if let Some(op) = k.logic {
            write!(f, " lop={op:#x}")?;
        }
        if k.stipple {
            write!(f, " stipple")?;
        }
        if k.opaque {
            write!(f, " opaque")?;
        }
        if k.skip_last {
            write!(f, " skip-last")?;
        }
        if k.prim == Prim::Xfer {
            write!(f, " {:?}/{}B", k.xfmt, k.xbpp)?;
        }
        if k.prim.is_gl() {
            write!(f, " {}", if k.rgb { "rgb" } else { "ci" })?;
            if let Some(a) = k.alpha {
                write!(f, " alpha={a}")?;
            }
            if let Some(s) = k.stencil {
                write!(f, " stencil={}:{}/{}/{}", s.func, s.fail, s.zfail, s.zpass)?;
            }
            if let Some(z) = k.z {
                write!(f, " z={z}")?;
            }
            if let Some(b) = k.blend {
                write!(f, " blend={}/{}", b.src, b.dst)?;
            }
            if let Some(t) = k.tex {
                write!(
                    f,
                    " tex(env{} nc{} slot{} class{} d{}{}{}{}{}{}{}{}{})",
                    t.env,
                    t.nc,
                    t.slot,
                    t.class,
                    t.depth,
                    if t.mag_linear { " mag-lin" } else { "" },
                    if t.min_linear { " min-lin" } else { "" },
                    if t.mipmap { " mip" } else { "" },
                    if t.mip_linear { "-lin" } else { "" },
                    if t.clamp_s { " clamp-s" } else { "" },
                    if t.clamp_t { " clamp-t" } else { "" },
                    if t.gl_clamp { " gl-clamp" } else { "" },
                    if t.no_border { " no-border" } else { "" },
                )?;
            }
            if k.fog {
                write!(f, " fog")?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_keys_fit_in_64_bits() {
        let m = PipeKey::max_packed();
        assert!(m <= u64::MAX as u128, "key space {m} needs {} bits", 128 - m.leading_zeros());
    }

    #[test]
    fn keys_round_trip_through_packing() {
        let tex = Tex {
            env: 2,
            nc: 2,
            slot: 1,
            class: 2,
            mag_linear: true,
            mipmap: true,
            mip_linear: true,
            clamp_t: true,
            gl_clamp: true,
            no_border: true,
            depth: 3,
            ..Tex::default()
        };
        let k = PipeKey {
            prim: Prim::Tri,
            cid_test: true,
            nclip: 4,
            draw: Draw::Dual,
            pix: Pix::Ci12,
            logic: Some(3),
            stipple: true,
            rgb: true,
            alpha: Some(6),
            stencil: Some(Stencil { func: 7, fail: 5, zfail: 4, zpass: 3 }),
            z: Some(7),
            blend: Some(Blend { src: 10, dst: 9 }),
            tex: Some(tex),
            fog: true,
            ..PipeKey::default()
        };
        assert_eq!(PipeKey::unpack(k.pack()), k.normalized());
        let x = PipeKey { prim: Prim::Xfer, xfmt: XFmt::Rgb16, xbpp: 6, logic: Some(6), ..PipeKey::default() };
        assert_eq!(PipeKey::unpack(x.pack()), x.normalized());
    }

    #[test]
    fn normalisation_folds_what_draws_alike() {
        let a = PipeKey { prim: Prim::Tri, rgb: false, blend: Some(Blend { src: 4, dst: 5 }), ..PipeKey::default() };
        assert_eq!(a.pack(), PipeKey { prim: Prim::Tri, ..PipeKey::default() }.pack());
        let s = PipeKey { prim: Prim::Fill, tex: Some(Tex::default()), alpha: Some(1), ..PipeKey::default() };
        assert_eq!(s.pack(), PipeKey::default().pack());
    }
}
