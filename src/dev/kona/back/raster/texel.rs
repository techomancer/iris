//! Texture sampling and the texture environment, on plain data.
//!
//! A `TexDesc` is a bound texture as a draw sees it: the levels' places in texture
//! memory, the filters, the wraps and the environment. It holds no pointer, so it
//! sits in `DrawState` and in the scheduler's table. Sampling is nearest or bilinear
//! within a level, nearest or linear between levels; the weights are 8-bit (the
//! model's own). `s = (s/w) / (q/w)` per fragment, so texture is
//! perspective-correct while colour is linear in screen space (both inferred).

use super::ops::{unpack_colour, FULL};
use crate::dev::kona::back::store::texture::{Plane, TextureStore};

/// Levels a texture can hold: a 4096-texel side is 13 halvings.
pub const MAX_LEVELS: usize = 14;

/// One level: where its texels sit in texture memory (word `off` of plane `plane`, rows `pitch` apart,
/// starting `ox`, `oy` into the physical rectangle) and its logical size. A volume's level
/// is `d` slices, `slice` texels apart; a 2D level is one slice (`d` zero).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TexLevel {
    pub plane: u16,
    pub off: u32,
    pub pitch: u16,
    pub ox: u16,
    pub oy: u16,
    pub w: u16,
    pub h: u16,
    pub d: u16,
    pub slice: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Filter {
    #[default]
    Nearest,
    Linear,
    /// Mip-mapped: the filter within a level, then between the two levels.
    Mip { within_linear: bool, between_linear: bool },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Wrap {
    #[default]
    Repeat,
    /// Clamp (to the edge texel: the model's choice for every non-repeat mode).
    Clamp,
}

/// How the texel combines with the fragment.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EnvMode {
    #[default]
    Modulate,
    Decal,
    Blend,
    Replace,
}

impl EnvMode {
    /// The mode a texture-environment mode number names.
    pub fn from_gl(v: u32) -> Option<EnvMode> {
        Some(match v {
            0x2100 => EnvMode::Modulate,
            0x2101 => EnvMode::Decal,
            0x0be2 => EnvMode::Blend,
            0x1e01 | 0x8062 => EnvMode::Replace,
            _ => return None,
        })
    }
}

impl Filter {
    fn bits(self) -> u32 {
        match self {
            Filter::Nearest => 0,
            Filter::Linear => 1,
            Filter::Mip { within_linear, between_linear } => 2 | (within_linear as u32) << 2 | (between_linear as u32) << 3,
        }
    }

    fn from_bits(b: u32) -> Filter {
        match b & 3 {
            0 => Filter::Nearest,
            1 => Filter::Linear,
            _ => Filter::Mip { within_linear: b >> 2 & 1 != 0, between_linear: b >> 3 & 1 != 0 },
        }
    }
}

/// The curve nodes a filter table holds: 64 samples at x = log2(i / 64) (the detail and sharpen
/// curve, established).
pub const CURVE_NODES: usize = 64;

/// The magnification combine a bound 2D texture asks for (mode-word codes 3, 4 and 5).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Combine {
    #[default]
    None,
    Detail(Detail),
    Sharpen(Sharpen),
}

/// A detail combine: the detail image (one level) added to or modulating the magnified base.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Detail {
    /// Modulate (mode-word code 4) and not add (code 3).
    pub modulate: bool,
    /// The detail level code, bits 23:21 of the mode word: the negated public value, 0 to 4 (the library rejects
    /// others; established). The detail image repeats 2^scale times as often as the base across the base
    /// (approximated).
    pub scale: u8,
    /// The magnification channels, bits 27:26 of the mode word: 0 all, 1 colour only, 2 alpha only, 3 level 0.
    pub channels: u8,
    pub image: TexLevel,
    /// The weight at each curve node, the public weight (the stored curve is doubled for detail).
    pub weights: [f32; CURVE_NODES],
}

/// A sharpen combine: the base pushed away from the next coarser level by the weight.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sharpen {
    pub channels: u8,
    /// The weight at each curve node, the public weight (the stored curve is negated for sharpen).
    pub weights: [f32; CURVE_NODES],
}

/// The weights of the client's default curve, points (-4, 1) and (0, 0): linear in the level of detail from 1 at
/// -4 and below to 0 at 0 (established as the default points; the straight line between them is approximated).
pub fn default_weights() -> [f32; CURVE_NODES] {
    std::array::from_fn(|i| if i == 0 { 1.0f32 } else { (-(i as f32 / 64.0f32).log2() / 4.0f32).clamp(0.0f32, 1.0f32) })
}

/// The 64 samples of a spline-form filter table: four ten-bit codes to two words, each a signed mantissa shifted
/// left two in bits 9:2 and an exponent in bits 1:0, value = mantissa x 2^exponent / 128 (established as the encoding a GL client
/// writes; word 33 is padding). None for a word that carries bits the code does not (the last word of a pair is 8 bits).
/// A GL client makes these codes (established): n = trunc(v x 512 + 2) >> 2, clamped to -1024..1023, then halved
/// until it is in -128..127 with the halvings counted, 0 to 3; the code is (n & 0xff) << 2 | count. A value a client
/// writes therefore reaches no further than 127 x 8 / 128.
pub fn spline_samples(words: &[u32; 33]) -> Option<[f32; CURVE_NODES]> {
    let mut out = [0.0f32; CURVE_NODES];
    for i in 0..16 {
        let (first, last) = (words[2 * i], words[2 * i + 1]);
        if last & !0xff != 0 {
            return None;
        }
        let codes = [first & 0x3ff, first >> 10 & 0x3ff, first >> 20 & 0x3ff, first >> 30 | last << 2];
        for (j, c) in codes.into_iter().enumerate() {
            out[4 * i + j] = (c >> 2) as u8 as i8 as f32 * (1u32 << (c & 3)) as f32 / 128.0f32;
        }
    }
    Some(out)
}

impl Combine {
    fn visit<S: crate::dev::kona::stream::Stream>(&mut self, s: &mut S) {
        let (mut kind, mut modulate, mut scale, mut channels) = (0u32, false, 0u8, 0u8);
        let (mut image, mut weights) = (TexLevel::default(), [0.0f32; CURVE_NODES]);
        match self {
            Combine::None => {}
            Combine::Detail(d) => (kind, modulate, scale, channels, image, weights) = (1, d.modulate, d.scale, d.channels, d.image, d.weights),
            Combine::Sharpen(p) => (kind, channels, weights) = (2, p.channels, p.weights),
        }
        s.u32(&mut kind);
        s.bool(&mut modulate);
        s.u8(&mut scale);
        s.u8(&mut channels);
        for v in [&mut image.pitch, &mut image.ox, &mut image.oy, &mut image.w, &mut image.h] {
            s.u16(v);
        }
        s.u32(&mut image.off);
        for w in weights.iter_mut() {
            let mut b = w.to_bits();
            s.u32(&mut b);
            *w = f32::from_bits(b);
        }
        if s.loading() {
            *self = match kind {
                1 => Combine::Detail(Detail { modulate, scale, channels, image, weights }),
                2 => Combine::Sharpen(Sharpen { channels, weights }),
                _ => Combine::None,
            };
        }
    }
}

/// A bound texture and its environment.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TexDesc {
    pub levels: [TexLevel; MAX_LEVELS],
    pub count: u8,
    pub min: Filter,
    pub mag: Filter,
    /// The wraps of s, t and r (r only for a volume).
    pub wrap: [Wrap; 3],
    /// The LOD scales of s, t and r, 16 times the factor (r only for a volume).
    pub lod_scale: [u8; 3],
    /// A volume (3D) texture: its levels have depth and the sampler takes the r coordinate.
    pub volume: bool,
    /// What a magnified fragment is combined with (none for a minified one).
    pub combine: Combine,
    /// The format class: 1 alpha, 2 luminance or RGB, 3 luminance-alpha or RGBA, 4 intensity.
    pub class: u8,
    pub env: EnvMode,
    /// The environment colour in 13-bit units.
    pub env_colour: [u32; 4],
}

impl TexDesc {
    /// A bound texture, as the stream module reads and writes it.
    pub fn visit<S: crate::dev::kona::stream::Stream>(&mut self, s: &mut S) {
        // Every field is named, so a new one fails to compile until it is saved here or left out on purpose.
        let Self { levels: _, count: _, min: _, mag: _, wrap: _, lod_scale: _, volume: _, combine: _, class: _, env: _, env_colour: _ } = self;
        for l in self.levels.iter_mut() {
            s.u32(&mut l.off);
            for v in [&mut l.pitch, &mut l.ox, &mut l.oy, &mut l.w, &mut l.h, &mut l.d] {
                s.u16(v);
            }
            s.u32(&mut l.slice);
        }
        s.u8(&mut self.count);
        for f in [&mut self.min, &mut self.mag] {
            let mut b = f.bits();
            s.u32(&mut b);
            *f = Filter::from_bits(b);
        }
        for w in self.wrap.iter_mut() {
            let mut b = (*w == Wrap::Clamp) as u32;
            s.u32(&mut b);
            *w = if b != 0 { Wrap::Clamp } else { Wrap::Repeat };
        }
        for v in self.lod_scale.iter_mut() {
            s.u8(v);
        }
        s.bool(&mut self.volume);
        self.combine.visit(s);
        s.u8(&mut self.class);
        let mut e = match self.env {
            EnvMode::Modulate => 0,
            EnvMode::Decal => 1,
            EnvMode::Blend => 2,
            EnvMode::Replace => 3,
        };
        s.u32(&mut e);
        self.env = [EnvMode::Modulate, EnvMode::Decal, EnvMode::Blend, EnvMode::Replace][(e & 3) as usize];
        s.u32s(&mut self.env_colour);
    }
}

/// A coordinate wrapped into `0..n`.
#[inline(always)]
fn wrap(i: i32, n: i32, w: Wrap) -> i32 {
    match w {
        Wrap::Repeat => i.rem_euclid(n),
        Wrap::Clamp => i.clamp(0, n - 1),
    }
}

/// Texel `x`, `y` of level `l`, `mem` being the level's plane (`TextureStore::plane`, taken once for a sample).
#[inline(always)]
fn texel(mem: &Plane, l: &TexLevel, x: i32, y: i32) -> u32 {
    mem.get(l.off + (l.oy as u32 + y as u32) * l.pitch as u32 + l.ox as u32 + x as u32)
}

/// A texel's bytes (r, g, b, a) of the word `0xAARRGGBB`.
#[inline(always)]
fn bytes(w: u32) -> [u32; 4] {
    [w >> 16 & 0xff, w >> 8 & 0xff, w & 0xff, w >> 24]
}

/// A level sampled at `s`, `t` in 0..1 repeating: nearest, or bilinear with 8-bit weights.
fn level(store: &TextureStore, l: &TexLevel, s: f32, t: f32, linear: bool, wr: [Wrap; 3]) -> [u32; 4] {
    let mem = store.plane(l.plane);
    let (w, h) = (l.w.max(1) as i32, l.h.max(1) as i32);
    let (u, v) = (s * w as f32, t * h as f32);
    if !linear {
        return bytes(texel(mem, l, wrap(u.floor() as i32, w, wr[0]), wrap(v.floor() as i32, h, wr[1])));
    }
    let (u, v) = (u - 0.5, v - 0.5);
    let (i0, j0) = (u.floor(), v.floor());
    let (fu, fv) = (((u - i0) * 256.0) as u32, ((v - j0) * 256.0) as u32);
    let (i0, j0) = (i0 as i32, j0 as i32);
    let t00 = bytes(texel(mem, l, wrap(i0, w, wr[0]), wrap(j0, h, wr[1])));
    let t10 = bytes(texel(mem, l, wrap(i0 + 1, w, wr[0]), wrap(j0, h, wr[1])));
    let t01 = bytes(texel(mem, l, wrap(i0, w, wr[0]), wrap(j0 + 1, h, wr[1])));
    let t11 = bytes(texel(mem, l, wrap(i0 + 1, w, wr[0]), wrap(j0 + 1, h, wr[1])));
    std::array::from_fn(|k| {
        let top = t00[k] * (256 - fu) + t10[k] * fu;
        let bot = t01[k] * (256 - fu) + t11[k] * fu;
        (top * (256 - fv) + bot * fv + 32768) >> 16
    })
}

/// The texel colour (r, g, b, a in 8 bits) at `s`, `t` with level of detail `lod`.
pub fn sample(store: &TextureStore, d: &TexDesc, s: f32, t: f32, lod: f32) -> [u32; 4] {
    let top = (d.count as usize).saturating_sub(1);
    let minify = lod > 0.0;
    match if minify { d.min } else { d.mag } {
        Filter::Nearest => level(store, &d.levels[0], s, t, false, d.wrap),
        Filter::Linear => level(store, &d.levels[0], s, t, true, d.wrap),
        Filter::Mip { within_linear, between_linear } => {
            let lod = lod.clamp(0.0, top as f32);
            let l0 = lod.floor() as usize;
            if !between_linear || l0 >= top {
                let l = if between_linear { l0 } else { (lod + 0.5) as usize }.min(top);
                return level(store, &d.levels[l], s, t, within_linear, d.wrap);
            }
            let (a, b) = (level(store, &d.levels[l0], s, t, within_linear, d.wrap), level(store, &d.levels[l0 + 1], s, t, within_linear, d.wrap));
            let f = ((lod - l0 as f32) * 256.0) as u32;
            std::array::from_fn(|k| (a[k] * (256 - f) + b[k] * f + 128) >> 8)
        }
    }
}

#[inline(always)]
fn texel3(mem: &Plane, l: &TexLevel, x: i32, y: i32, z: i32) -> u32 {
    mem.get(l.off + z as u32 * l.slice + (l.oy as u32 + y as u32) * l.pitch as u32 + l.ox as u32 + x as u32)
}

/// A volume level sampled at `s`, `t`, `r`: nearest, or trilinear with 8-bit weights (a bilinear
/// blend in each of the two slices the r coordinate falls between, then a blend of the two). The
/// weights are the model's own: the pipe's filter arithmetic is not read (approximated).
fn level3(store: &TextureStore, l: &TexLevel, s: f32, t: f32, r: f32, linear: bool, wr: [Wrap; 3]) -> [u32; 4] {
    let mem = store.plane(l.plane);
    let (w, h, dp) = (l.w.max(1) as i32, l.h.max(1) as i32, l.d.max(1) as i32);
    let (u, v, q) = (s * w as f32, t * h as f32, r * dp as f32);
    if !linear {
        return bytes(texel3(mem, l, wrap(u.floor() as i32, w, wr[0]), wrap(v.floor() as i32, h, wr[1]), wrap(q.floor() as i32, dp, wr[2])));
    }
    let (u, v, q) = (u - 0.5, v - 0.5, q - 0.5);
    let (i0, j0, k0) = (u.floor(), v.floor(), q.floor());
    let (fu, fv, fz) = (((u - i0) * 256.0) as u32, ((v - j0) * 256.0) as u32, ((q - k0) * 256.0) as u32);
    let (i0, j0, k0) = (i0 as i32, j0 as i32, k0 as i32);
    let (xs, ys) = ([wrap(i0, w, wr[0]), wrap(i0 + 1, w, wr[0])], [wrap(j0, h, wr[1]), wrap(j0 + 1, h, wr[1])]);
    // One slice's bilinear blend, 16 fraction bits.
    let slice = |z: i32| -> [u32; 4] {
        let t00 = bytes(texel3(mem, l, xs[0], ys[0], z));
        let t10 = bytes(texel3(mem, l, xs[1], ys[0], z));
        let t01 = bytes(texel3(mem, l, xs[0], ys[1], z));
        let t11 = bytes(texel3(mem, l, xs[1], ys[1], z));
        std::array::from_fn(|k| {
            let top = t00[k] * (256 - fu) + t10[k] * fu;
            let bot = t01[k] * (256 - fu) + t11[k] * fu;
            top * (256 - fv) + bot * fv
        })
    };
    let (a, b) = (slice(wrap(k0, dp, wr[2])), slice(wrap(k0 + 1, dp, wr[2])));
    std::array::from_fn(|k| (a[k] * (256 - fz) + b[k] * fz + (1 << 23)) >> 24)
}

/// The texel colour at `s`, `t`, `r` with level of detail `lod`, for a volume texture: `sample`'s
/// level selection and blending over `level3`.
pub fn sample3(store: &TextureStore, d: &TexDesc, s: f32, t: f32, r: f32, lod: f32) -> [u32; 4] {
    let top = (d.count as usize).saturating_sub(1);
    match if lod > 0.0 { d.min } else { d.mag } {
        Filter::Nearest => level3(store, &d.levels[0], s, t, r, false, d.wrap),
        Filter::Linear => level3(store, &d.levels[0], s, t, r, true, d.wrap),
        Filter::Mip { within_linear, between_linear } => {
            let lod = lod.clamp(0.0, top as f32);
            let l0 = lod.floor() as usize;
            if !between_linear || l0 >= top {
                let l = if between_linear { l0 } else { (lod + 0.5) as usize }.min(top);
                return level3(store, &d.levels[l], s, t, r, within_linear, d.wrap);
            }
            let (a, b) = (level3(store, &d.levels[l0], s, t, r, within_linear, d.wrap), level3(store, &d.levels[l0 + 1], s, t, r, within_linear, d.wrap));
            let f = ((lod - l0 as f32) * 256.0) as u32;
            std::array::from_fn(|k| (a[k] * (256 - f) + b[k] * f + 128) >> 8)
        }
    }
}

/// The curve's weight at a level of detail at or below zero: the node at 64 x 2^lod, blended linearly with the next.
/// Indexing by 64 x 2^lod is inferred; the blend between nodes is approximated (no source says
/// whether the hardware interpolates or takes the nearest node).
#[inline(always)]
fn curve_weight(w: &[f32; CURVE_NODES], lod: f32) -> f32 {
    let at = (lod.exp2() * 64.0f32).min(63.0f32);
    let n = at as usize;
    let f = at - n as f32;
    w[n] * (1.0f32 - f) + w[(n + 1).min(CURVE_NODES - 1)] * f
}

/// Whether a channel (0 to 3: red, green, blue, alpha) is changed under the magnification channel field. Code 3
/// ("level 0") is not read and changes every channel (approximated).
#[inline(always)]
fn channel_on(channels: u8, k: usize) -> bool {
    match channels {
        1 => k < 3,
        2 => k == 3,
        _ => true,
    }
}

/// The texel colour at `s`, `t` for a 2D texture that has a detail or sharpen combine, with level of detail `lod`.
/// A minified fragment, which the combine does not touch, is sampled as `sample` does. A magnified one is the base
/// level sampled bilinearly, then:
///
/// Detail: the detail image sampled bilinearly, repeating, at the base's level-0 texel coordinate times 2^|L|, L the
/// detail level (the code is the negated public value), wrapped to the detail size: the same position as the base's
/// coordinate times the size ratio times 2^code, which is what this takes. Then `base + K (d - 1/2)` for add and
/// `base (1 + K (d - 1/2))` for modulate, clamped, K the stored doubled curve at node 64 x 2^lod, which is the
/// public weight times two (the position and the forms are inferred). The widths, rounding, the half-texel offset
/// and the clamp point are unknown (approximated: single-precision arithmetic).
///
/// Sharpen: `A + K (B - A)` with K the stored curve, which is the public weight negated, A the base sampled bilinearly
/// and B the next coarser level (the form and the sign are inferred; the sign of the stored curve is established). The
/// widths and rounding are unknown (approximated).
///
/// Chosen once per draw (`pixel::run_tri`): a plain texture uses `sample`, which carries none of this.
pub fn sample_enhanced(store: &TextureStore, d: &TexDesc, s: f32, t: f32, lod: f32) -> [u32; 4] {
    if lod > 0.0f32 {
        return sample(store, d, s, t, lod);
    }
    let base = level(store, &d.levels[0], s, t, true, d.wrap);
    let unit = |v: u32| v as f32 / 255.0f32;
    let out = |v: f32| (v.clamp(0.0f32, 1.0f32) * 255.0f32 + 0.5f32) as u32;
    match &d.combine {
        Combine::None => base,
        Combine::Detail(c) => {
            let (b0, img) = (&d.levels[0], &c.image);
            let k = (1u32 << c.scale.min(15)) as f32;
            let (ds, dt) = (s * (b0.w.max(1) as f32 / img.w.max(1) as f32) * k, t * (b0.h.max(1) as f32 / img.h.max(1) as f32) * k);
            let other = level(store, img, ds, dt, true, [Wrap::Repeat; 3]);
            let w = curve_weight(&c.weights, lod);
            std::array::from_fn(|k| {
                if !channel_on(c.channels, k) {
                    return base[k];
                }
                let (b, signed) = (unit(base[k]), w * (2.0f32 * unit(other[k]) - 1.0f32));
                out(if c.modulate { b * (1.0f32 + signed) } else { b + signed })
            })
        }
        Combine::Sharpen(c) => {
            if d.count < 2 {
                return base;
            }
            let coarse = level(store, &d.levels[1], s, t, true, d.wrap);
            let w = curve_weight(&c.weights, lod);
            std::array::from_fn(|k| if channel_on(c.channels, k) { out(unit(base[k]) + w * (unit(base[k]) - unit(coarse[k]))) } else { base[k] })
        }
    }
}

/// `a * b` of two 13-bit values.
#[inline(always)]
fn mul(a: u32, b: u32) -> u32 {
    (a * b + FULL / 2) / FULL
}

/// The fragment colour `f` (13-bit r, g, b, a) after the texture environment, with the
/// texel `t` (8-bit r, g, b, a from `sample`). Class 1 is alpha only (no colour), 2
/// luminance or RGB (no alpha), 3 luminance-alpha or RGBA, 4 intensity (the one
/// component stands for colour and alpha). Decal is defined for classes 2 and 3
/// only; the others modulate. The arithmetic and its rounding are the model's own.
pub fn environment(d: &TexDesc, f: [u32; 4], t: [u32; 4]) -> [u32; 4] {
    let tc = unpack_colour(t[3] << 24 | t[0] << 16 | t[1] << 8 | t[2]);
    let (ct, at) = ([tc[0], tc[1], tc[2]], match d.class {
        1 | 3 => tc[3],
        4 => tc[0],
        _ => FULL,
    });
    let k = d.env_colour;
    let mode = if d.env == EnvMode::Decal && !matches!(d.class, 2 | 3) { EnvMode::Modulate } else { d.env };
    let mut out = match (mode, d.class) {
        (EnvMode::Modulate, 1) => [f[0], f[1], f[2], mul(f[3], at)],
        (EnvMode::Modulate, _) => [mul(f[0], ct[0]), mul(f[1], ct[1]), mul(f[2], ct[2]), mul(f[3], at)],
        (EnvMode::Decal, 2) => [ct[0], ct[1], ct[2], f[3]],
        (EnvMode::Decal, _) => [mul(f[0], FULL - at) + mul(ct[0], at), mul(f[1], FULL - at) + mul(ct[1], at), mul(f[2], FULL - at) + mul(ct[2], at), f[3]],
        (EnvMode::Blend, 1) => [f[0], f[1], f[2], mul(f[3], at)],
        (EnvMode::Blend, c) => {
            let a = if c == 4 { mul(f[3], FULL - at) + mul(k[3], at) } else if c == 2 { f[3] } else { mul(f[3], at) };
            [mul(f[0], FULL - ct[0]) + mul(k[0], ct[0]), mul(f[1], FULL - ct[1]) + mul(k[1], ct[1]), mul(f[2], FULL - ct[2]) + mul(k[2], ct[2]), a]
        }
        (EnvMode::Replace, 1) => [f[0], f[1], f[2], at],
        (EnvMode::Replace, 2) => [ct[0], ct[1], ct[2], f[3]],
        (EnvMode::Replace, _) => [ct[0], ct[1], ct[2], at],
    };
    out.iter_mut().for_each(|c| *c = (*c).min(FULL));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2 x 2 texture at offset 0: red, green / blue, white, opaque.
    fn two() -> (TextureStore, TexDesc) {
        let t = TextureStore::default();
        t.clear(0, 0, 4);
        for (i, w) in [0xffff_0000u32, 0xff00_ff00, 0xff00_00ff, 0xffff_ffff].into_iter().enumerate() {
            t.set(0, i as u32, w);
        }
        let mut d = TexDesc { count: 1, class: 3, ..Default::default() };
        d.levels[0] = TexLevel { off: 0, pitch: 2, ox: 0, oy: 0, w: 2, h: 2, ..Default::default() };
        (t, d)
    }

    #[test]
    fn nearest_picks_the_texel_and_wraps() {
        let (t, d) = two();
        assert_eq!(sample(&t, &d, 0.25, 0.25, 0.0), [255, 0, 0, 255]);
        assert_eq!(sample(&t, &d, 0.75, 0.25, 0.0), [0, 255, 0, 255]);
        assert_eq!(sample(&t, &d, 1.25, 0.75, 0.0), [0, 0, 255, 255], "repeat: 1.25 is 0.25");
        let c = TexDesc { wrap: [Wrap::Clamp; 3], ..d };
        assert_eq!(sample(&t, &c, 1.25, 0.25, 0.0), [0, 255, 0, 255], "clamp to the edge texel");
        assert_eq!(sample(&t, &c, -3.0, 5.0, 0.0), [0, 0, 255, 255]);
    }

    #[test]
    fn bilinear_blends_the_four_texels_at_a_corner() {
        let (t, d) = two();
        let l = TexDesc { mag: Filter::Linear, wrap: [Wrap::Clamp; 3], ..d };
        let mid = sample(&t, &l, 0.5, 0.5, 0.0);
        assert_eq!(mid, [128, 128, 128, 255], "{mid:?}: the mean of red, green, blue and white");
        assert_eq!(sample(&t, &l, 0.25, 0.25, 0.0), [255, 0, 0, 255], "at a texel's centre, that texel");
    }

    #[test]
    fn a_mip_filter_picks_and_blends_levels() {
        let t = TextureStore::default();
        t.clear(0, 0, 5);
        t.set(0, 0, 0xffff_0000);
        t.set(0, 1, 0xffff_0000);
        t.set(0, 2, 0xffff_0000);
        t.set(0, 3, 0xffff_0000);
        t.set(0, 4, 0xff00_00ff);
        let mut d = TexDesc { count: 2, class: 3, ..Default::default() };
        d.levels[0] = TexLevel { off: 0, pitch: 2, w: 2, h: 2, ..Default::default() };
        d.levels[1] = TexLevel { off: 4, pitch: 1, w: 1, h: 1, ..Default::default() };
        d.min = Filter::Mip { within_linear: false, between_linear: false };
        assert_eq!(sample(&t, &d, 0.1, 0.1, 0.4), [255, 0, 0, 255], "nearest level: level 0");
        assert_eq!(sample(&t, &d, 0.1, 0.1, 0.6), [0, 0, 255, 255], "level 1");
        d.min = Filter::Mip { within_linear: false, between_linear: true };
        assert_eq!(sample(&t, &d, 0.1, 0.1, 0.5), [128, 0, 128, 255], "halfway between the two");
        assert_eq!(sample(&t, &d, 0.1, 0.1, 9.0), [0, 0, 255, 255], "past the last level: the last");
        assert_eq!(sample(&t, &d, 0.1, 0.1, 0.0), [255, 0, 0, 255], "magnified: the magnification filter on level 0");
    }

    /// A 2 x 2 x 2 volume at offset 0: slice 0 is red, slice 1 is blue (opaque), each slice 4 texels.
    fn volume() -> (TextureStore, TexDesc) {
        let t = TextureStore::default();
        t.clear(0, 0, 8 + 2);
        for i in 0..4 {
            t.set(0, i, 0xffff_0000);
            t.set(0, 4 + i, 0xff00_00ff);
        }
        let mut d = TexDesc { count: 1, class: 3, volume: true, wrap: [Wrap::Clamp, Wrap::Clamp, Wrap::Repeat], ..Default::default() };
        d.levels[0] = TexLevel { off: 0, pitch: 2, w: 2, h: 2, d: 2, slice: 4, ..Default::default() };
        (t, d)
    }

    #[test]
    fn a_volume_picks_its_slice_by_r_and_wraps_or_clamps_it() {
        let (t, d) = volume();
        assert_eq!(sample3(&t, &d, 0.25, 0.25, 0.25, 0.0), [255, 0, 0, 255], "r in the first half: slice 0");
        assert_eq!(sample3(&t, &d, 0.25, 0.25, 0.75, 0.0), [0, 0, 255, 255], "slice 1");
        assert_eq!(sample3(&t, &d, 0.25, 0.25, 1.25, 0.0), [255, 0, 0, 255], "repeat: 1.25 is 0.25");
        let c = TexDesc { wrap: [Wrap::Clamp; 3], ..d };
        assert_eq!(sample3(&t, &c, 0.25, 0.25, 1.25, 0.0), [0, 0, 255, 255], "clamp: the last slice");
        assert_eq!(sample3(&t, &c, 0.25, 0.25, -2.0, 0.0), [255, 0, 0, 255], "clamp: the first");
    }

    #[test]
    fn trilinear_blends_the_two_slices_around_r() {
        let (t, d) = volume();
        let l = TexDesc { mag: Filter::Linear, ..d };
        assert_eq!(sample3(&t, &l, 0.25, 0.25, 0.5, 0.0), [128, 0, 128, 255], "halfway between the slice centres");
        assert_eq!(sample3(&t, &l, 0.25, 0.25, 0.25, 0.0), [255, 0, 0, 255], "at a slice centre, that slice");
        // Repeating r blends the last slice into the first past the end.
        assert_eq!(sample3(&t, &l, 0.25, 0.25, 1.0, 0.0), [128, 0, 128, 255], "wrapped between slice 1 and slice 0");
        let c = TexDesc { wrap: [Wrap::Clamp; 3], ..l };
        assert_eq!(sample3(&t, &c, 0.25, 0.25, 1.0, 0.0), [0, 0, 255, 255], "clamped past the end: the last slice");
    }

    #[test]
    fn a_volume_mip_chain_halves_every_axis() {
        let (t, mut d) = volume();
        t.set(0, 8, 0xff00_ff00);
        d.count = 2;
        d.levels[1] = TexLevel { off: 8, pitch: 1, w: 1, h: 1, d: 1, slice: 1, ..Default::default() };
        d.min = Filter::Mip { within_linear: false, between_linear: false };
        assert_eq!(sample3(&t, &d, 0.25, 0.25, 0.75, 0.4), [0, 0, 255, 255], "level 0");
        assert_eq!(sample3(&t, &d, 0.25, 0.25, 0.75, 0.6), [0, 255, 0, 255], "level 1: one texel whatever r is");
        d.min = Filter::Mip { within_linear: true, between_linear: true };
        assert_eq!(sample3(&t, &d, 0.25, 0.25, 0.75, 0.5), [0, 128, 128, 255], "halfway between the levels, level 0 at r = 0.75 is the slice 1 centre, blue");
    }

    /// Four codes to two words: a signed mantissa shifted left two and an exponent. Value = mantissa x 2^exponent / 128.
    #[test]
    fn a_spline_table_decodes_to_sixty_four_samples() {
        let mut w = [0u32; 33];
        // Codes: 64 << 2 | 0 (0.5), 0xc0 << 2 | 1 (-64 x 2 / 128 = -1), 1 << 2 | 3 (8/128), 127 << 2 | 0 (127/128).
        let codes = [64u32 << 2, 0xc0 << 2 | 1, 1 << 2 | 3, 127 << 2];
        w[0] = codes[0] | codes[1] << 10 | codes[2] << 20 | (codes[3] & 3) << 30;
        w[1] = codes[3] >> 2;
        let v = spline_samples(&w).unwrap();
        assert_eq!(&v[..5], &[0.5, -1.0, 0.0625, 127.0 / 128.0, 0.0]);
        w[1] |= 0x100;
        assert!(spline_samples(&w).is_none(), "the second word of a pair is 8 bits");
    }

    /// Four stored hardware test vectors decode to the default curve: the doubled 2.0 below
    /// x = -4 is code 0x102, the knee 0x1d5, x = -2 is 0x101, x = -1 is 0x100 and the last sample 0x004.
    #[test]
    fn the_hardware_vectors_decode_to_the_default_curve() {
        let at = |c: u32| {
            let mut w = [0u32; 33];
            w[0] = c;
            spline_samples(&w).unwrap()[0]
        };
        assert_eq!([at(0x102), at(0x101), at(0x100), at(0x004)], [2.0, 1.0, 0.5, 1.0 / 128.0]);
        assert_eq!(at(0x1d5), 117.0 * 2.0 / 128.0);
        assert_eq!(at(0x1ff), 127.0 * 8.0 / 128.0, "the top of the range, 7.9375");
        assert_eq!(at(0x201), -2.0, "-2.0 codes as 0x201");
    }

    /// The default curve is 1 at the finest and falls to 0 at a level of detail of zero.
    #[test]
    fn the_default_curve_runs_from_one_to_zero() {
        let w = default_weights();
        assert_eq!((w[0], w[63] < 0.01), (1.0, true));
        assert!((w[16] - 0.5).abs() < 1e-6, "node 16 is lod -2");
    }

    /// A 2 x 2 base of mid grey and an 8-wide detail of alternating black and white: a magnified fragment moves off
    /// the grey by the weight in the direction of its detail texel (add) or by that fraction of itself (modulate), a
    /// minified one does not, and the channel field restricts it.
    #[test]
    fn a_detail_combine_adds_or_modulates_on_magnification_only() {
        let t = TextureStore::default();
        t.clear(0, 0, 4 + 8);
        for i in 0..4 {
            t.set(0, i, 0xff80_8080);
        }
        for i in 0..8 {
            t.set(0, 4 + i, if i % 2 == 0 { 0xff00_0000 } else { 0xffff_ffff });
        }
        let weights = [0.25f32; CURVE_NODES];
        let mut d = TexDesc { count: 1, class: 3, mag: Filter::Linear, wrap: [Wrap::Clamp; 3], ..Default::default() };
        d.levels[0] = TexLevel { off: 0, pitch: 2, w: 2, h: 2, ..Default::default() };
        let image = TexLevel { off: 4, pitch: 8, w: 8, h: 1, ..Default::default() };
        let det = |modulate, channels| Combine::Detail(Detail { modulate, scale: 0, channels, image, weights });
        // The detail coordinate is the base's times 2/8: s = 0.25 is the centre of detail texel 0 (black), 0.75 of texel 1 (white).
        d.combine = det(false, 0);
        assert_eq!(sample_enhanced(&t, &d, 0.25, 0.5, -1.0)[..3], [64, 64, 64], "0x80 less a weight of a quarter");
        assert_eq!(sample_enhanced(&t, &d, 0.75, 0.5, -1.0)[..3], [192, 192, 192], "white detail: plus a quarter");
        assert_eq!(sample_enhanced(&t, &d, 0.25, 0.5, 1.0), sample(&t, &d, 0.25, 0.5, 1.0), "minified: the base alone");
        d.combine = det(true, 0);
        assert_eq!(sample_enhanced(&t, &d, 0.25, 0.5, -1.0)[..3], [96, 96, 96], "modulate: 0x80 x (1 - 1/4)");
        assert_eq!(sample_enhanced(&t, &d, 0.75, 0.5, -1.0)[..3], [160, 160, 160], "0x80 x (1 + 1/4)");
        d.combine = det(false, 2);
        assert_eq!(sample_enhanced(&t, &d, 0.25, 0.5, -1.0)[..3], [128, 128, 128], "alpha only: the colour is the base's");
        d.combine = det(false, 1);
        let c = sample_enhanced(&t, &d, 0.25, 0.5, -1.0);
        assert_eq!((c[..3].to_vec(), c[3]), (vec![64, 64, 64], 255), "colour only; alpha is the base's");
        // The curve is read at 64 x 2^lod, so lod -1 is node 32, and a zero node is the base.
        let mut w = [0.0f32; CURVE_NODES];
        w[32] = 0.125;
        d.combine = Combine::Detail(Detail { modulate: false, scale: 0, channels: 0, image, weights: w });
        assert_eq!(sample_enhanced(&t, &d, 0.25, 0.5, -1.0)[..3], [96, 96, 96], "node 32 at lod -1: an eighth");
        assert_eq!(sample_enhanced(&t, &d, 0.25, 0.5, -2.0)[..3], [128, 128, 128], "node 16: zero");
        // A scale of one repeats the detail twice as often: the detail coordinate is 2/8 x 2 x s, so s = 0.125 is texel 0's centre.
        d.combine = Combine::Detail(Detail { modulate: false, scale: 1, channels: 0, image, weights });
        assert_eq!(sample_enhanced(&t, &d, 0.125, 0.5, -1.0)[..3], [64, 64, 64]);
    }

    /// A sharpen combine pushes the magnified base away from the next coarser level, and does nothing with one level.
    #[test]
    fn a_sharpen_combine_pushes_away_from_the_coarser_level() {
        let t = TextureStore::default();
        t.clear(0, 0, 5);
        for i in 0..4 {
            t.set(0, i, 0xff80_8080);
        }
        t.set(0, 4, 0xff40_4040);
        let mut d = TexDesc { count: 2, class: 3, mag: Filter::Linear, wrap: [Wrap::Clamp; 3], combine: Combine::Sharpen(Sharpen { channels: 0, weights: [0.5; CURVE_NODES] }), ..Default::default() };
        d.levels[0] = TexLevel { off: 0, pitch: 2, w: 2, h: 2, ..Default::default() };
        d.levels[1] = TexLevel { off: 4, pitch: 1, w: 1, h: 1, ..Default::default() };
        let c = sample_enhanced(&t, &d, 0.25, 0.25, -1.0);
        assert_eq!(c[..3], [160, 160, 160], "0x80 + 1/2 (0x80 - 0x40)");
        d.count = 1;
        assert_eq!(sample_enhanced(&t, &d, 0.25, 0.25, -1.0)[..3], [128, 128, 128], "no coarser level: the base");
    }

    fn env(mode: EnvMode, class: u8, f: [u32; 4], t: [u32; 4]) -> [u32; 4] {
        environment(&TexDesc { env: mode, class, env_colour: [FULL, 0, 0, FULL], ..Default::default() }, f, t)
    }

    #[test]
    fn the_environment_combines_texel_and_fragment_by_mode_and_class() {
        let half = [4096, 4096, 4096, 4096];
        let white = [255, 255, 255, 255];
        assert_eq!(env(EnvMode::Modulate, 3, half, white), half, "modulate by white leaves the fragment");
        assert_eq!(env(EnvMode::Modulate, 3, [FULL; 4], [0, 255, 0, 128])[..3], [0, FULL, 0]);
        assert_eq!(env(EnvMode::Replace, 2, half, [255, 0, 0, 7]), [FULL, 0, 0, 4096], "RGB replace keeps the fragment's alpha");
        assert_eq!(env(EnvMode::Replace, 3, half, [255, 0, 0, 255]), [FULL, 0, 0, FULL]);
        assert_eq!(env(EnvMode::Decal, 3, [0, FULL, 0, FULL], [255, 0, 0, 255]), [FULL, 0, 0, FULL], "opaque RGBA texel decals over the fragment");
        assert_eq!(env(EnvMode::Decal, 3, [0, FULL, 0, FULL], [255, 0, 0, 0]), [0, FULL, 0, FULL], "a transparent texel leaves it");
        assert_eq!(env(EnvMode::Decal, 1, half, white), env(EnvMode::Modulate, 1, half, white), "decal of an alpha texture modulates");
        assert_eq!(env(EnvMode::Blend, 2, [0, 0, 0, FULL], [255, 255, 255, 255])[..3], [FULL, 0, 0], "a white texel takes the environment colour");
        assert_eq!(env(EnvMode::Modulate, 1, [FULL, 0, 0, FULL], [0, 0, 0, 128])[3], mul(FULL, unpack_colour(128 << 24)[3]), "alpha texture scales alpha only");
        let i = unpack_colour(0xff80_8080)[0];
        assert_eq!(env(EnvMode::Modulate, 4, [FULL; 4], [128, 128, 128, 255]), [i; 4], "intensity scales colour and alpha");
    }
}
