//! The per-pixel module: what runs for every pixel of a triangle, a clear or a
//! tile.
//!
//! Integers and single-precision floats only: the edge functions and the depth
//! plane (32.16 fixed point) are exact `i64`, and the colour planes `f32`
//! evaluated as `at_origin + dx * i + dy * j` with separate multiplies and adds
//! (no fused operation), so the result is the same on every host, and a GPU
//! kernel could reproduce it. A test in `golden` holds this file to that. The
//! one write is `WordArray::store`.
//!
//! The order of a fragment's operations is OpenGL's (alpha test, stencil, depth, blend,
//! logic operation, colour mask) with the ownership test first; the board's order is
//! unknown beyond the alpha test coming before the raster engines, so the position of
//! ownership and of both stipples is the model's own. Blend and then logic operation
//! both apply when both are on (the plan's order; OpenGL lets the logic operation
//! replace blending in RGBA). The 13-bit and 8-bit conversions are the model's own.
//!
//! A job is walked in 2 x 2 quads aligned to even screen coordinates, clipped
//! to the job's rectangle and the tile (the walk a texture's level of detail
//! will need). A tile owner calls `run` with its tile as `clip`; the walk is
//! clipped to it and keeps the window's own pixel grid, so a tiled run is
//! bit-exact with an inline one. There is one per-fragment path, which reads the
//! draw state and branches on the operations that are on, and the loop is
//! specialised once for each draw on the features the draw uses (see `run_tri`),
//! so a draw does not pay for those it does not use; other speed work is deferred.

use super::setup::{Tri, TriJob};
use crate::dev::kona::back::tiles::RasterJob;
use super::texel::{environment, sample, sample3, sample_enhanced, TexDesc};
use super::{DrawState, Rect};
use crate::dev::kona::back::store::texture::TextureStore;
use crate::dev::kona::back::store::{PixelStore, WordArray, DID_PLANE};

use super::ops::{blend, ownership_plane, pack_colour, to8, unpack_colour};

/// An attribute plane's value at a sample, clamped to `0..=max` and rounded.
#[inline(always)]
fn at(t: &Tri, k: usize, i: f32, j: f32, max: f32) -> u32 {
    let v = t.at_origin[k] + t.dx[k] * i + t.dy[k] * j;
    (v.clamp(0.0f32, max) + 0.5f32) as u32
}

/// The depth plane's value at a sample: the 32.16 plane's integer part, clamped to
/// the fraction's range.
#[inline(always)]
fn depth_at(t: &Tri, i: i64, j: i64) -> u32 {
    // Wrapping: when the true value fits `i64` the wrapped sum is exact, and a sliver far
    // outside the screen, whose slope can near 2^51, cannot panic a development build.
    let v = t.z.at.wrapping_add(t.z.dx.wrapping_mul(i)).wrapping_add(t.z.dy.wrapping_mul(j));
    (v >> 16).clamp(0, DEPTH_MAX as i64) as u32
}

/// The depth plane's value at sample offset `at` (eighths of a pixel) from pixel (`i`, `j`).
#[inline(always)]
fn depth_at_sample(t: &Tri, i: i64, j: i64, at: [u8; 2]) -> u32 {
    let v = t.z.at.wrapping_add(t.z.dx.wrapping_mul(i)).wrapping_add(t.z.dy.wrapping_mul(j)).wrapping_add((t.z.dx.wrapping_mul(at[0] as i64)).wrapping_add(t.z.dy.wrapping_mul(at[1] as i64)) / 8);
    (v >> 16).clamp(0, DEPTH_MAX as i64) as u32
}

/// The depth fraction's largest value, `0xffffff00`.
const DEPTH_MAX: u32 = 4_294_967_040;

/// Write `new` to a colour word over `old` under a colour mask.
#[inline(always)]
fn masked(old: u32, new: u32, mask: [bool; 4]) -> u32 {
    let keep = (!mask[0] as u32 * 0xff) << 16 | (!mask[1] as u32 * 0xff) << 8 | (!mask[2] as u32 * 0xff) | (!mask[3] as u32 * 0xff) << 24;
    (new & !keep) | (old & keep)
}

/// The planes a fragment is tested against and written to, resolved once per job.
pub struct Planes<'a> {
    depth: Option<WordArray<'a>>,
    stencil: Option<WordArray<'a>>,
    colour: [Option<WordArray<'a>>; 2],
    /// A multisampled window's sample storage; its depth and stencil are here, not above.
    samples: Option<SampleArrays<'a>>,
}

/// The words a multisampled window keeps a sample in, a sample's value at a bit offset in the
/// run of words (the layout's sizes are 24 bits a sample of colour, 24 or 16 of depth and 8 of stencil).
struct SampleArrays<'a> {
    colour: Vec<WordArray<'a>>,
    depth: Vec<WordArray<'a>>,
    stencil: Vec<WordArray<'a>>,
}

/// The bits of sample `k` in a run of words at `index`: `width` bits from bit `k * stride`.
#[inline(always)]
fn sample_get(words: &[WordArray<'_>], index: usize, k: usize, stride: u32, width: u32) -> u32 {
    let bit = k * stride as usize;
    let (w, sh) = (bit / 32, (bit % 32) as u32);
    // SAFETY: the pixel is inside the screen, which every array covers.
    let lo = unsafe { words[w].load(index) } as u64;
    // SAFETY: as above, and a value that crosses a word has one after it.
    let hi = if sh + width > 32 { (unsafe { words[w + 1].load(index) } as u64) << 32 } else { 0 };
    ((lo | hi) >> sh) as u32 & ((1u64 << width) - 1) as u32
}

#[inline(always)]
fn sample_set(words: &[WordArray<'_>], index: usize, k: usize, stride: u32, width: u32, v: u32) {
    let bit = k * stride as usize;
    let (w, sh) = (bit / 32, (bit % 32) as u32);
    let mask = ((1u64 << width) - 1) << sh;
    let put = |a: &WordArray<'_>, mask: u32, v: u32| {
        // SAFETY: the pixel is inside the screen, which every array covers.
        unsafe { a.store(index, (a.load(index) & !mask) | (v & mask)) };
    };
    put(&words[w], mask as u32, ((v as u64) << sh) as u32);
    if sh + width > 32 {
        put(&words[w + 1], (mask >> 32) as u32, ((v as u64) << sh >> 32) as u32);
    }
}

impl<'a> Planes<'a> {
    pub fn new(store: &'a PixelStore, ds: &DrawState) -> Planes<'a> {
        let colours = &ds.colour[..ds.colours as usize];
        let run = |(first, n): (u32, u32)| (first..first + n).map(|w| store.array(w as usize).expect("allocated by set-up")).collect::<Vec<_>>();
        let samples = ds.samples.as_ref().map(|m| SampleArrays { colour: run(m.colour), depth: run(m.depth), stencil: run(m.stencil) });
        let plain = samples.is_none();
        Planes {
            depth: (plain && ds.depth_bits != 0).then(|| store.array(ds.depth_word as usize).expect("allocated by set-up")),
            stencil: (plain && ds.stencil_bits != 0 && ds.stencil.enabled).then(|| store.array(ds.stencil_word as usize).expect("allocated by set-up")),
            colour: [colours.first().map(|&w| store.array(w as usize).expect("allocated")), colours.get(1).map(|&w| store.array(w as usize).expect("allocated"))],
            samples,
        }
    }

    /// One fragment at plane index `index`, through the tests and writes that come after
    /// ownership and the stipples: alpha, stencil, depth, blend, logic operation, colour mask.
    #[inline(always)]
    pub fn shade(&self, ds: &DrawState, index: usize, src: [u32; 4], z: u32) {
        let all = [true; 4];
        let (depth, arrays) = (&self.depth, &self.colour);
        let test = ds.depth_test && depth.is_some();
        let reads_dst = ds.blend.enabled || ds.logic.is_some() || ds.colour_mask != all;
        let st = &ds.stencil;
        if ds.alpha.enabled && !ds.alpha.func.passes(src[3], ds.alpha.reference) {
            return;
        }
        // The stencil test, then the depth test; each outcome has its own operation.
        let mut stored = 0u8;
        if let Some(sa) = &self.stencil {
            // SAFETY: the pixel is inside the screen, which the array covers.
            stored = unsafe { sa.load(index) } as u8;
            let pass = st.func.passes((st.reference & st.read_mask) as u32, (stored & st.read_mask) as u32);
            if !pass {
                let v = st.fail.apply(stored, st.reference);
                // SAFETY: as above.
                unsafe { sa.store(index, ((stored & !st.write_mask) | (v & st.write_mask)) as u32) };
                return;
            }
        }
        // SAFETY: as above.
        let zpass = !test || ds.depth_func.passes(z, unsafe { depth.as_ref().expect("test needs depth").load(index) });
        if let Some(sa) = &self.stencil {
            let v = if zpass { st.zpass } else { st.zfail }.apply(stored, st.reference);
            // SAFETY: as above.
            unsafe { sa.store(index, ((stored & !st.write_mask) | (v & st.write_mask)) as u32) };
        }
        if !zpass {
            return;
        }
        if let (Some(d), true, true) = (depth, ds.depth_test, ds.depth_mask) {
            // SAFETY: as above.
            unsafe { d.store(index, z) };
        }
        for a in arrays.iter().flatten() {
            // SAFETY: as above.
            let old = if reads_dst { unsafe { a.load(index) } } else { 0 };
            // SAFETY: as above.
            unsafe { a.store(index, colour_out(ds, src, old)) };
        }
    }

    /// One fragment of a multisampled window, once for each sample of `covered`: the stencil and depth tests
    /// and the colour operations run against that sample's own values; `z_at` is the fragment's depth at a
    /// sample (a fraction of `u32::MAX`). The pixel's colour word is then the mean of its samples, so
    /// samples two triangles cover between them add up to a whole pixel. A pixel is tested with the alpha
    /// once, before its samples (alpha is a fragment's, not a sample's).
    pub fn shade_samples(&self, ds: &DrawState, index: usize, src: [u32; 4], z_at: impl Fn(usize) -> u32, covered: u32) {
        let (Some(ms), Some(sa)) = (&ds.samples, &self.samples) else { return };
        if ds.alpha.enabled && !ds.alpha.func.passes(src[3], ds.alpha.reference) {
            return;
        }
        let count = ms.count as usize;
        let (cstride, dwidth) = (ms.colour.1 * 32 / ms.count as u32, if ms.depth.1 == 0 { 0 } else { ms.depth.1 * 32 / ms.count as u32 });
        let st = &ds.stencil;
        let stencil_on = ds.stencil.enabled && ds.stencil_bits != 0 && !sa.stencil.is_empty();
        let test = ds.depth_test && !sa.depth.is_empty();
        // Blending reads the destination's alpha from the main word, which samples do not keep.
        let main = self.colour.iter().flatten().next();
        let mut live = 0u32;
        for k in 0..count {
            if covered >> k & 1 == 0 {
                continue;
            }
            let mut stored = 0u8;
            if stencil_on {
                stored = sample_get(&sa.stencil, index, k, 8, 8) as u8;
                if !st.func.passes((st.reference & st.read_mask) as u32, (stored & st.read_mask) as u32) {
                    let v = st.fail.apply(stored, st.reference);
                    sample_set(&sa.stencil, index, k, 8, 8, ((stored & !st.write_mask) | (v & st.write_mask)) as u32);
                    continue;
                }
            }
            let z = z_at(k) >> (32 - dwidth.clamp(1, 32));
            let zpass = !test || ds.depth_func.passes(z, sample_get(&sa.depth, index, k, dwidth, dwidth));
            if stencil_on {
                let v = if zpass { st.zpass } else { st.zfail }.apply(stored, st.reference);
                sample_set(&sa.stencil, index, k, 8, 8, ((stored & !st.write_mask) | (v & st.write_mask)) as u32);
            }
            if !zpass {
                continue;
            }
            if test && ds.depth_mask {
                sample_set(&sa.depth, index, k, dwidth, dwidth, z);
            }
            live |= 1 << k;
        }
        if live == 0 {
            return;
        }
        // SAFETY (loads and stores of the main word): the pixel is inside the screen.
        let alpha = main.map_or(0, |a| unsafe { a.load(index) } >> 24 & 0xff);
        let mut sum = [0u32; 3];
        for k in 0..count {
            let mut rgb = sample_get(&sa.colour, index, k, cstride, 24);
            if live >> k & 1 != 0 {
                let out = colour_out(ds, src, alpha << 24 | rgb);
                rgb = out & 0xff_ffff;
                sample_set(&sa.colour, index, k, cstride, 24, rgb);
            }
            sum[0] += rgb >> 16 & 0xff;
            sum[1] += rgb >> 8 & 0xff;
            sum[2] += rgb & 0xff;
        }
        // Samples keep no alpha: the word's is the last fragment's, so a shared edge leaves no seam in it.
        let new_alpha = to8(src[3]);
        let mean = |v: u32| (v + count as u32 / 2) / count as u32;
        let word = new_alpha << 24 | mean(sum[0]) << 16 | mean(sum[1]) << 8 | mean(sum[2]);
        for a in self.colour.iter().flatten() {
            // SAFETY: as above.
            unsafe { a.store(index, word) };
        }
    }
}

impl Planes<'_> {
    /// Sample 0's depth at `index` as a fraction of `u32::MAX`, in a multisampled window (reads take one sample).
    pub fn sample_depth(&self, ds: &DrawState, index: usize) -> Option<u32> {
        let (ms, sa) = (ds.samples.as_ref()?, self.samples.as_ref()?);
        let w = ms.depth.1 * 32 / ms.count as u32;
        (!sa.depth.is_empty()).then(|| sample_get(&sa.depth, index, 0, w, w) << (32 - w.clamp(1, 32)))
    }

    /// Sample 0's stencil at `index`, in a multisampled window.
    pub fn sample_stencil(&self, index: usize) -> Option<u32> {
        let sa = self.samples.as_ref()?;
        (!sa.stencil.is_empty()).then(|| sample_get(&sa.stencil, index, 0, 8, 8))
    }

    /// A stencil image's value into every sample of the pixel under the write mask, in a multisampled window.
    pub fn set_stencil(&self, ds: &DrawState, index: usize, value: u8, mask: u8) {
        let (Some(ms), Some(sa)) = (&ds.samples, &self.samples) else { return };
        for k in 0..ms.count as usize {
            let old = sample_get(&sa.stencil, index, k, 8, 8);
            sample_set(&sa.stencil, index, k, 8, 8, (old & !mask as u32) | (value & mask) as u32);
        }
    }

    /// A clear of the samples of the pixel at `index`: colour under the colour mask, depth under the
    /// depth mask, stencil under its mask.
    fn clear_samples(&self, ds: &DrawState, index: usize, q: &super::ClearReq) {
        let (Some(ms), Some(sa)) = (&ds.samples, &self.samples) else { return };
        let count = ms.count as usize;
        let (cstride, dwidth) = (ms.colour.1 * 32 / ms.count as u32, if ms.depth.1 == 0 { 0 } else { ms.depth.1 * 32 / ms.count as u32 });
        for k in 0..count {
            if let Some(word) = q.colour {
                let old = sample_get(&sa.colour, index, k, cstride, 24);
                sample_set(&sa.colour, index, k, cstride, 24, masked(old, word, q.colour_mask) & 0xff_ffff);
            }
            if let (Some(z), true, true) = (q.depth, q.depth_mask, !sa.depth.is_empty()) {
                sample_set(&sa.depth, index, k, dwidth, dwidth, z >> (32 - dwidth.clamp(1, 32)));
            }
            if let (Some(v), false) = (q.stencil, sa.stencil.is_empty()) {
                let old = sample_get(&sa.stencil, index, k, 8, 8);
                sample_set(&sa.stencil, index, k, 8, 8, (old & !(q.stencil_mask as u32)) | (v & q.stencil_mask) as u32);
            }
        }
    }
}

/// The colour a fragment writes over `old`: blend, then the logic operation, then the colour mask.
#[inline(always)]
fn colour_out(ds: &DrawState, src: [u32; 4], old: u32) -> u32 {
    let mut c = src;
    if ds.blend.enabled {
        c = blend(src, unpack_colour(old), ds.blend.src, ds.blend.dst, ds.blend.eq, ds.blend.constant);
    }
    let mut word = pack_colour(c);
    if let Some(op) = ds.logic {
        word = op.apply(word, old);
    }
    if ds.colour_mask != [true; 4] {
        word = masked(old, word, ds.colour_mask);
    }
    word
}

/// The samples of pixel (`x`, `y`) inside the triangle, as bits: each at its offset in eighths from the
/// pixel's corner, on the edges' own grid.
#[inline(always)]
fn sample_mask(t: &Tri, ms: &super::multisample::Samples, x: i64, y: i64) -> u32 {
    let mut mask = 0;
    for k in 0..ms.count as usize {
        let [ox, oy] = ms.at[k].map(|v| v as i64);
        let inside = (0..3).all(|e| {
            // The edge at the corner, moved by the sample's offset (a and b are multiples of 8).
            let v = t.a[e] * x + t.b[e] * y + t.c[e] + (t.a[e] * ox + t.b[e] * oy) / 8;
            v > 0 || (v == 0 && t.tie[e])
        });
        mask |= (inside as u32) << k;
    }
    mask
}

/// Run `job`, under draw state `ds`, over the pixels of `clip` it covers.
pub fn run(store: &PixelStore, job: &RasterJob, ds: &DrawState, clip: Rect) {
    let r = job.rect().intersect(&clip);
    if r.is_empty() {
        return;
    }
    let colours = &ds.colour[..ds.colours as usize];
    let pitch = store.width() as usize;
    let all = [true; 4];
    // Whether the window owns the pixel at `index` (the ID plane's bits 7:2 are the main ID).
    let did = ds.ownership.is_some().then(|| store.array(DID_PLANE)).flatten();
    let owns = |index: usize| match &ds.ownership {
        None => true,
        // SAFETY: the pixel is inside the rectangle, inside the screen, which the array covers.
        Some(o) => o.owns(ownership_plane(did.as_ref().map_or(0, |d| unsafe { d.load(index) }))),
    };
    match job {
        RasterJob::Clear(c) => {
            let q = &c.req;
            if let (Some(word), true) = (q.colour, !colours.is_empty()) {
                for &w in colours {
                    let a = store.array(w as usize).expect("allocated by set-up");
                    for y in r.y0..r.y1 {
                        for x in r.x0..r.x1 {
                            let i = y as usize * pitch + x as usize;
                            if !owns(i) {
                                continue;
                            }
                            // SAFETY: the rectangle is inside the screen, which the array covers.
                            unsafe { a.store(i, if q.colour_mask == all { word } else { masked(a.load(i), word, q.colour_mask) }) };
                        }
                    }
                }
            }
            // A multisampled window clears its samples; its depth and stencil words are packed samples, not planes.
            if ds.samples.is_some() {
                let planes = Planes::new(store, ds);
                for y in r.y0..r.y1 {
                    for x in r.x0..r.x1 {
                        let i = y as usize * pitch + x as usize;
                        if owns(i) {
                            planes.clear_samples(ds, i, q);
                        }
                    }
                }
            }
            if let (Some(z), true, true, true) = (q.depth, ds.depth_bits != 0, q.depth_mask, ds.samples.is_none()) {
                let a = store.array(ds.depth_word as usize).expect("allocated by set-up");
                for y in r.y0..r.y1 {
                    for x in r.x0..r.x1 {
                        let i = y as usize * pitch + x as usize;
                        if owns(i) {
                            // SAFETY: as above.
                            unsafe { a.store(i, z) };
                        }
                    }
                }
            }
            if let (Some(v), true, true) = (q.stencil, ds.stencil_bits != 0, ds.samples.is_none()) {
                let a = store.array(ds.stencil_word as usize).expect("allocated by set-up");
                for y in r.y0..r.y1 {
                    for x in r.x0..r.x1 {
                        let i = y as usize * pitch + x as usize;
                        if owns(i) {
                            // SAFETY: as above.
                            unsafe { a.store(i, (a.load(i) & !(q.stencil_mask as u32)) | (v & q.stencil_mask) as u32) };
                        }
                    }
                }
            }
        }
        RasterJob::Tri(job) => {
            // One function per fog curve, per smoothed or plain triangle and per kind of bound
            // texture, chosen here once for the job: a triangle without fog, smoothing, a volume
            // texture or a detail or sharpen combine runs the loop with no code for any of them in it.
            let aa = job.tri.cover != super::smooth::Cover::None;
            let tex = match &ds.texture {
                Some(d) if d.volume => Kind::Volume,
                Some(d) if !matches!(d.combine, super::texel::Combine::None) => Kind::Enhanced,
                _ => Kind::Plain,
            };
            macro_rules! go {
                ($fog:expr) => {
                    match (aa, tex) {
                        (false, Kind::Plain) => run_tri::<{ $fog }, false, Plain>(store, job, ds, r, pitch, &owns),
                        (true, Kind::Plain) => run_tri::<{ $fog }, true, Plain>(store, job, ds, r, pitch, &owns),
                        (false, Kind::Volume) => run_tri::<{ $fog }, false, Volume>(store, job, ds, r, pitch, &owns),
                        (true, Kind::Volume) => run_tri::<{ $fog }, true, Volume>(store, job, ds, r, pitch, &owns),
                        (false, Kind::Enhanced) => run_tri::<{ $fog }, false, Enhanced>(store, job, ds, r, pitch, &owns),
                        (true, Kind::Enhanced) => run_tri::<{ $fog }, true, Enhanced>(store, job, ds, r, pitch, &owns),
                    }
                };
            }
            match super::fog::curve_of(ds) {
                0 => go!(0),
                super::fog::LINEAR => go!(super::fog::LINEAR),
                super::fog::EXP => go!(super::fog::EXP),
                _ => go!(super::fog::EXP2),
            }
        }
    }
}

/// How a draw's bound texture is sampled, chosen once for a job.
#[derive(Clone, Copy)]
enum Kind {
    Plain,
    Volume,
    Enhanced,
}

/// The sampler `run_tri` is built over: a plain 2D texture (or none), a volume, which takes a third
/// coordinate and a depth in the level of detail, or a 2D texture with a detail or sharpen combine.
/// A type for each kind, so the choice is made at compile time and no kind is a catch-all.
trait Sampler {
    /// Whether the draw takes the r coordinate.
    const VOLUME: bool;
    fn sample(store: &TextureStore, d: &TexDesc, s: f32, t: f32, r: f32, lod: f32) -> [u32; 4];
}

struct Plain;
struct Volume;
struct Enhanced;

impl Sampler for Plain {
    const VOLUME: bool = false;
    #[inline(always)]
    fn sample(store: &TextureStore, d: &TexDesc, s: f32, t: f32, _r: f32, lod: f32) -> [u32; 4] {
        sample(store, d, s, t, lod)
    }
}

impl Sampler for Volume {
    const VOLUME: bool = true;
    #[inline(always)]
    fn sample(store: &TextureStore, d: &TexDesc, s: f32, t: f32, r: f32, lod: f32) -> [u32; 4] {
        sample3(store, d, s, t, r, lod)
    }
}

impl Sampler for Enhanced {
    const VOLUME: bool = false;
    #[inline(always)]
    fn sample(store: &TextureStore, d: &TexDesc, s: f32, t: f32, _r: f32, lod: f32) -> [u32; 4] {
        sample_enhanced(store, d, s, t, lod)
    }
}

/// The triangle `job`, in the pixels of `r`, with fog of curve `FOG` (0: none; `fog::LINEAR`,
/// `fog::EXP`, `fog::EXP2`), with coverage (`AA`) when it is a triangle of a smoothed primitive, and
/// sampling a texture with `S`.
fn run_tri<const FOG: u8, const AA: bool, S: Sampler>(store: &PixelStore, job: &TriJob, ds: &DrawState, r: Rect, pitch: usize, owns: &impl Fn(usize) -> bool) {
    let t = &job.tri;
    let planes = Planes::new(store, ds);
    // Quads aligned to even screen coordinates, clipped to the rectangle: the
    // walk a derivative (texture level of detail) needs. A quad with no
    // covered sample is skipped whole; the four samples are otherwise taken
    // in order, so a fragment's result does not depend on the others.
    let (qx0, qy0) = (r.x0 & !1, r.y0 & !1);
    let tex = store.texture();
    // Only read with fog on (`FOG` != 0); the default stands in otherwise and is never used.
    let fog = ds.fog.unwrap_or(super::fog::Fog::OFF);
    for qy in (qy0..r.y1).step_by(2) {
        for qx in (qx0..r.x1).step_by(2) {
            // The texture coordinates at a sample, and the level of detail from the
            // quad's differences: log2 of the larger of the s and t steps in texels of
            // level 0, times the LOD scale (16 times the factor). A volume's third
            // coordinate, r, is taken only for a volume (`S::VOLUME`): a 2D draw has no code for it,
            // not even its divide.
            let volume = S::VOLUME;
            let coord = |x: i32, y: i32| -> (f32, f32, f32) {
                let (i, j) = ((x - t.origin[0]) as f32, (y - t.origin[1]) as f32);
                let w = |k: usize| t.tex[k][0] + t.tex[k][1] * i + t.tex[k][2] * j;
                let q = w(3);
                let r = if volume { w(2) } else { 0.0f32 };
                if q.abs() > 1e-30f32 { (w(0) / q, w(1) / q, if volume { r / q } else { 0.0f32 }) } else { (w(0), w(1), r) }
            };
            let lod = ds.texture.as_ref().map(|d| {
                let (s0, t0, r0) = coord(qx, qy);
                let (sx, tx, rx) = coord(qx + 1, qy);
                let (sy, ty, ry) = coord(qx, qy + 1);
                let l0 = d.levels[0];
                let (fw, fh) = (l0.w.max(1) as f32, l0.h.max(1) as f32);
                let mut rho = ((sx - s0) * fw).abs().max(((sy - s0) * fw).abs()).max(((tx - t0) * fh).abs()).max(((ty - t0) * fh).abs());
                let mut scale = d.lod_scale[0].max(1) as f32;
                if volume {
                    // The depth step in texels of level 0 joins the largest, under its own scale; how the
                    // pipe combines the three steps is not read (approximated).
                    let fd = l0.d.max(1) as f32;
                    let rr = ((rx - r0) * fd).abs().max(((ry - r0) * fd).abs());
                    let rs = d.lod_scale[2].max(1) as f32;
                    if rr * rs > rho * scale {
                        (rho, scale) = (rr, rs);
                    }
                }
                (rho * scale / 16.0f32).max(1e-6f32).log2()
            });
            for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                let (x, y) = (qx + dx, qy + dy);
                if x < r.x0 || x >= r.x1 || y < r.y0 || y >= r.y1 {
                    continue;
                }
                // The samples of the pixel inside the triangle: one, at the corner, without multisampling.
                // A smoothed triangle (never in a multisampled window) takes the share of the pixel it
                // covers instead, and draws the pixel when that is not nothing.
                let mut cov = 1.0f32;
                let covered = if AA {
                    cov = super::smooth::coverage(t, x, y);
                    // A line's and a point's coverage reaches the fragment as an 8-bit alpha.
                    if job.line || matches!(t.cover, super::smooth::Cover::Disc { .. }) {
                        cov = super::smooth::quantise(cov);
                    }
                    // In a multisampled window a smoothed line writes every sample of the pixel.
                    if cov > 0.0f32 { ds.samples.as_ref().map_or(1, |m| (1u32 << m.count as u32) - 1) } else { 0 }
                } else {
                    match &ds.samples {
                        None => (0..3).all(|k| {
                            let e = t.a[k] * x as i64 + t.b[k] * y as i64 + t.c[k];
                            e > 0 || (e == 0 && t.tie[k])
                        }) as u32,
                        Some(ms) => sample_mask(t, ms, x as i64, y as i64),
                    }
                };
                if covered == 0 {
                    continue;
                }
                let (i, j) = ((x - t.origin[0]) as f32, (y - t.origin[1]) as f32);
                let index = y as usize * pitch + x as usize;
                if !owns(index) {
                    continue;
                }
                // Stipple: a line's by its distance along the line, a polygon's by its window position.
                if job.line {
                    if let Some((pattern, repeat)) = ds.line_stipple {
                        let d = t.line_d[0] + t.line_d[1] * i + t.line_d[2] * j;
                        if pattern >> (((d.max(0.0f32) as u32) / repeat.max(1)) & 15) & 1 == 0 {
                            continue;
                        }
                    }
                } else if let Some(rows) = &ds.polygon_stipple {
                    let (col, row) = ((x - ds.origin[0]) & 31, (y - ds.origin[1]) & 31);
                    if rows[row as usize] >> (31 - col) & 1 == 0 {
                        continue;
                    }
                }
                let mut src = [at(t, 0, i, j, 8191.0f32), at(t, 1, i, j, 8191.0f32), at(t, 2, i, j, 8191.0f32), at(t, 3, i, j, 8191.0f32)];
                if let (Some(d), Some(lod)) = (&ds.texture, lod) {
                    let (s, tt, r) = coord(x, y);
                    let texel = S::sample(tex, d, s, tt, r, lod);
                    src = environment(d, src, texel);
                }
                // Fog follows the texture environment and comes before the alpha test: OpenGL's
                // order, the model's choice. The coordinate is the plane of coordinate / w over the plane of 1 / w.
                if FOG != 0 {
                    let (n, q) = (t.fog[0][0] + t.fog[0][1] * i + t.fog[0][2] * j, t.fog[1][0] + t.fog[1][1] * i + t.fog[1][2] * j);
                    let c = if q.abs() > 1e-30f32 { n / q } else { n };
                    src = fog.apply(src, fog.factor::<FOG>(c));
                }
                // The share covered scales alpha, after fog and before the alpha test.
                if AA {
                    src[3] = (src[3] as f32 * cov + 0.5f32) as u32;
                }
                let (i, j) = ((x - t.origin[0]) as i64, (y - t.origin[1]) as i64);
                match &ds.samples {
                    None => planes.shade(ds, index, src, depth_at(t, i, j)),
                    // A multisampled fragment keeps the samples the sample mask and alpha-to-mask leave it,
                    // shaded once and written to each, at the depth of the sample.
                    Some(ms) => {
                        let mut m = covered & ms.allowed(x);
                        if ms.alpha_to_mask {
                            m &= ms.alpha_samples(src[3]);
                        }
                        if ms.alpha_to_one {
                            src[3] = 8191;
                        }
                        if m != 0 {
                            planes.shade_samples(ds, index, src, |k| depth_at_sample(t, i, j, ms.at[k]), m);
                        }
                    }
                }
            }
        }
    }
}
