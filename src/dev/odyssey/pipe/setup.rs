//! Setup: a draw's vertices, as the transform side hands them over, into
//! the rasteriser's primitives (`clanker/plans/odyssey-quad-pipeline.md`).
//!
//! The transform side (the GL context's matrices, viewport, depth range and
//! user clip planes, `Transform`) maps each vertex to clip coordinates, and
//! setup clips, projects and assembles from there; what a primitive does
//! after that comes from the raster, shader and back-end words alone.
//!
//! Window coordinates truncate toward minus infinity onto a grid of 1/256
//! pixel (inferred), through the shared VPro geometry
//! (`dev::gl::vertex::NearW`). Depth is linear in eye distance (a w-buffer):
//! the depth row's product with the eye coordinates over w interpolates
//! linearly in the window, and its ratio to the divisor's is the fragment's
//! depth. Clipping is in single precision (a model choice: the board's
//! clip precision is unknown).

use super::fixed;
use super::interp::{self, TriTex};
use super::raster::{triangle_quads, Quad, RasterVertex, Triangle, SUBPIXEL_BITS};
use super::ring::BackQuad;
use super::texture::{TexQuad, Unit, Wrap};
use super::backend::{colour12, quantise12};
use super::PipeDraw;
use super::super::assembly::{report_w_plane_clip, vtx, CLIP_PLANES, XfState};
use crate::dev::gl::math::xform;
use crate::dev::gl::vertex::{Clip, ClipVertex, NearW};
use crate::devlog::LogModule;

/// One pixel in subpixels.
pub(crate) const SUB: i64 = 1 << SUBPIXEL_BITS;

/// The most vertices clipping a triangle leaves: one more per view-volume
/// plane that clips (`NearW` has five).
pub(crate) const CLIPPED_MAX: usize = 3 + 6;

/// What the transform side hands a draw: the vertices' mapping to clip and
/// window coordinates (its matrices and viewport, or the window matrix),
/// the depth range, the depth row and divisor (`Board::gl_operation_rules`)
/// and the user clip planes, and whether vertices carry back colours.
#[derive(Clone, Copy)]
pub(crate) struct Transform<'a> {
    pub st: &'a XfState,
    pub depth_row: [f64; 4],
    pub divisor: Option<[f64; 4]>,
    pub back_colours: bool,
}

impl Transform<'_> {
    /// A vertex of the draw in eye and clip coordinates, in single
    /// precision through the shared transform: object to eye by the
    /// model-view, eye to clip by the projection or the window matrix.
    pub fn vertex(&self, v: &[f32]) -> VolumeVertex {
        let eye = xform(&self.st.modelview, v[vtx::POSITION..vtx::POSITION + 4].try_into().expect("a position"));

        VolumeVertex { clip: self.st.eye_to_clip(eye), eye, attr: v[..vtx::FLOATS].try_into().expect("a whole vertex") }
    }

    /// The planes geometry is clipped by, under the shared VPro geometry
    /// (`NearW`): x and y to the guard band about the window, w to the
    /// near-w plane, and no near or far plane, since the raster drops a
    /// fragment by its depth (inferred). With `user`, the enabled user clip
    /// planes too, which polygon modes clip their outline by; everything
    /// else tests them per fragment. How the board applies a user plane is
    /// unknown, so both are model choices.
    pub fn clip(&self, user: bool) -> Clip {
        Clip { user: self.st.clip_plane, user_on: if user { self.st.clip_planes } else { 0 } }
    }

    /// The enabled user clip planes.
    pub fn user_planes(&self) -> u32 {
        self.st.clip_planes
    }

    /// Whether a fragment whose user plane values are d lies outside an
    /// enabled plane.
    pub fn user_clipped(&self, d: &[f64]) -> bool {
        d.iter().enumerate().take(CLIP_PLANES as usize).any(|(i, &di)| self.st.clip_planes & (1 << i) != 0 && !(di >= 0.0))
    }

    /// The window pixels a primitive's fragments may fall on within the
    /// window's own bounds `owned`, inclusive: within the viewport. None
    /// when there are none.
    pub fn bounds(&self, owned: (i32, i32, i32, i32)) -> Option<(i32, i32, i32, i32)> {
        let (x0, y0, x1, y1) = owned;
        let [vx, vy, vw, vh] = self.st.viewport;

        if vw <= 0 || vh <= 0 {
            return None;
        }
        Some((x0.max(vx), y0.max(vy), x1.min(vx.wrapping_add(vw).wrapping_sub(1)), y1.min(vy.wrapping_add(vh).wrapping_sub(1))))
    }

    /// Map one clipped vertex: the divide by w and the viewport map in
    /// single precision through the shared mapping, depth over w from the
    /// depth row, the divisor over w, the user plane values and the colour
    /// clamped to [0, 1]. None for a vertex the mapping has no answer for
    /// (w not positive) or one beyond the rasteriser's range.
    pub fn project(&self, cv: &VolumeVertex) -> Option<RasterVertex> {
        let st = self.st;
        let clip = cv.clip.map(f64::from);
        let eye = cv.eye.map(f64::from);
        let Some(win) = st.clip_to_window(&cv.clip, SUB as f32).map(|w| w.map(f64::from)) else {
            crate::dlog!(LogModule::Odyssey, "sgi-gl: clipped vertex at w {} has no window position, primitive dropped", clip[3]);
            return None;
        };
        if !(win[0].abs() < 65536.0 && win[1].abs() < 65536.0) {
            crate::dlog!(LogModule::Odyssey, "sgi-gl: vertex outside the rasterizer's range, primitive dropped");
            return None;
        }
        let dot = |d: &[f64; 4]| d[0] * eye[0] + d[1] * eye[1] + d[2] * eye[2] + d[3] * eye[3];
        let mut out = RasterVertex {
            // Already on the grid: the truncation only makes it an integer.
            x: (win[0] * SUB as f64).floor() as i64,
            y: (win[1] * SUB as f64).floor() as i64,
            z: dot(&self.depth_row) / clip[3],
            inv_w: 1.0 / clip[3],
            ..Default::default()
        };
        out.depth_w = match &self.divisor {
            Some(d) => dot(d) / clip[3],
            None => out.inv_w,
        };
        for k in 0..4 {
            out.tex[k] = cv.attr[vtx::TEXCOORD + k] as f64;
            // max then min: a NaN colour becomes 0.
            #[allow(clippy::manual_clamp, reason = "clamp would keep a NaN")]
            let c = cv.attr[vtx::COLOR + k].max(0.0).min(1.0);
            out.rgba[k] = c as f64;
        }
        for (i, p) in st.clip_plane.iter().enumerate().take(CLIP_PLANES as usize) {
            out.clip[i] = p[0] as f64 * eye[0] + p[1] as f64 * eye[1] + p[2] as f64 * eye[2] + p[3] as f64 * eye[3];
        }
        Some(out)
    }

    /// The depth range [n, f].
    pub fn depth_range(&self) -> [f64; 2] {
        self.st.depth_range.map(f64::from)
    }
}

/// Clip triangle in_ under the shared VPro geometry (`NearW`): the polygon
/// left and its vertex count, 0 when less than a triangle is.
pub(crate) fn clip_triangle(clip: &Clip, in_: &[VolumeVertex; 3]) -> ([VolumeVertex; CLIPPED_MAX], usize) {
    let mut out = [VolumeVertex::default(); CLIPPED_MAX];

    out[..3].copy_from_slice(in_);
    let any = in_.iter().fold(0, |o, v| o | clip.outcode_by::<NearW, _>(v));
    if any == 0 {
        return (out, 3);
    }
    if any & NearW::W_PLANE != 0 {
        report_w_plane_clip();
    }
    let mut tmp = [VolumeVertex::default(); CLIPPED_MAX];
    let (n, _) = clip.clip_polygon_by::<NearW, _>(&mut out, &mut tmp, 3, any);
    (out, if n < 3 { 0 } else { n })
}

/// A vertex in clip coordinates, as the shared GL core's clipper
/// (`dev::gl::vertex::Clip`, under `NearW`) takes and makes it. Clipping is
/// in single precision: a model choice, since the board's clip precision is
/// unknown. The window position and 1/w come from this clip result.
#[derive(Clone, Copy)]
pub(crate) struct VolumeVertex {
    pub clip: [f32; 4],
    pub eye: [f32; 4],
    pub attr: [f32; vtx::FLOATS],
}

impl Default for VolumeVertex {
    fn default() -> Self {
        VolumeVertex { clip: [0.0; 4], eye: [0.0; 4], attr: [0.0; vtx::FLOATS] }
    }
}

impl VolumeVertex {
    /// The point at t along from self to out, in clip and eye coordinates
    /// and every float attribute (2.13.8); the material index and edge flag
    /// are bit patterns and stay self's.
    pub fn lerp(&self, out: &VolumeVertex, t: f32) -> VolumeVertex {
        let mut r = VolumeVertex::default();
        // The shared core's crossing (`dev::gl::vertex::clip_cross`).
        let l = |x: f32, y: f32| x + (y - x) * t;

        for k in 0..4usize {
            r.clip[k] = l(self.clip[k], out.clip[k]);
            r.eye[k] = l(self.eye[k], out.eye[k]);
        }
        for k in 0..vtx::FLOATS {
            r.attr[k] = if k == vtx::MATERIAL || k == vtx::EDGE { self.attr[k] } else { l(self.attr[k], out.attr[k]) };
        }
        r
    }
}

impl ClipVertex for VolumeVertex {
    fn h(&self) -> &[f32; 4] {
        &self.clip
    }
    fn e(&self) -> &[f32; 4] {
        &self.eye
    }
    /// Found from the inside end of the edge, so that two primitives
    /// sharing the edge make the same vertex.
    fn cross(a: &Self, b: &Self, da: f32, db: f32) -> Self {
        if da >= 0.0 { a.lerp(b, da / (da - db)) } else { b.lerp(a, db / (db - da)) }
    }
    const EDGE_FLAGS: bool = true;
    fn edge(&self) -> bool {
        self.attr[vtx::EDGE].to_bits() != 0
    }
    fn set_edge(&mut self, edge: bool) {
        self.attr[vtx::EDGE] = f32::from_bits(edge as u32);
    }
}

/// One draw as setup runs it: the pipeline's state, the transform side's
/// hand-over, the texture, and the window's bounds.
#[derive(Clone, Copy)]
pub(crate) struct Draw<'a> {
    pub pipe: &'a PipeDraw,
    pub xf: Transform<'a>,
    pub tex: Unit<'a>,
    pub textured: bool,
    /// The polygon stipple's rows, where it is on.
    pub stipple: Option<&'a [u32; 32]>,
    /// An opaque stipple's background colour (0x0c21 under 0x043f bit 4).
    pub background: Option<[f64; 4]>,
    /// The window's own bounds in window pixels, and those within the
    /// viewport (Transform::bounds()).
    pub owned: (i32, i32, i32, i32),
    pub bounds: Option<(i32, i32, i32, i32)>,
    /// The draw's shared context, when its triangles may be shaded on the
    /// shader thread.
    pub live: Option<&'a LiveDraw>,
}

/// A fragment of a line, a point or a smoothed polygon: its colour, its
/// perspective-correct texture coordinates (s, t, r, q), and their change to
/// the next pixel right and down, which make its helper pixels for the
/// level of detail.
#[derive(Clone, Copy, Default)]
pub(crate) struct Frag {
    pub rgba: [f64; 4],
    pub tex: [f64; 4],
    pub dx: [f64; 4],
    pub dy: [f64; 4],
}

impl Draw<'_> {
    /// Whether a polygon fragment at window pixel (px, py) is drawn under
    /// the polygon stipple: None where it is not, else whether it takes the
    /// opaque stipple's background.
    #[inline(always)]
    pub fn stipple_pass(&self, px: i32, py: i32) -> Option<bool> {
        if self.stipple.is_none_or(|rows| rows[((py as u32) & 31) as usize] & (1u32 << ((px as u32) & 31)) != 0) {
            return Some(false);
        }
        self.background.is_some().then_some(true)
    }

    /// A fragment at window pixel (px, py) with depth over w z, divisor
    /// over w depth_w and `steps` of polygon offset, of coverage `coverage`:
    /// through the window's ownership and the scissor, shaded, and into the
    /// back end.
    #[allow(clippy::too_many_arguments, reason = "a fragment as its parts")]
    pub fn fragment(&self, px: i32, py: i32, z: f64, depth_w: f64, steps: i64, frag: &Frag, coverage: f64) {
        let p = self.pipe;
        let rst = &p.raster;
        let Some(zw) = rst.window_depth(z, depth_w).filter(|_| coverage > 0.0) else {
            return;
        };
        let (x, y) = (rst.ox.wrapping_add(px), rst.oy.wrapping_add(py));
        let in_scissor = rst.scissor.is_none_or(|(x0, y0, x1, y1)| px >= x0 && px <= x1 && py >= y0 && py <= y1);

        if !in_scissor || !rst.owned(x, y) {
            return;
        }
        let zq = super::super::depth_word::word_with_offset((rst.depth_code)(zw), steps);
        let mut rgba = frag.rgba;
        let mut texel = [[0i32; 4]; 4];
        if self.textured {
            let at = |dx: f64, dy: f64| std::array::from_fn(|k| (frag.tex[k] + dx * frag.dx[k] + dy * frag.dy[k]) as f32);
            let q = TexQuad { c: [at(0.0, 0.0), at(1.0, 0.0), at(0.0, 1.0), at(1.0, 1.0)] };
            self.tex.sample_quad(&q, 1, &mut texel);
        }
        if !p.shader.shade(self.textured, &mut rgba, &texel[0], zw, coverage) {
            return;
        }
        let c = rgba.map(|c| quantise12(c) as u16);
        let mut q = BackQuad { x, y, cov: 1, zq: [zq, 0, 0, 0], rgba: [c, [0; 4], [0; 4], [0; 4]], ..Default::default() };

        p.emit(&mut q);
    }
}

/// A vertex's texture values: S/W, T/W, R/W and Q/W.
pub(crate) fn texture_values(v: &RasterVertex) -> [f64; 4] {
    [v.tex[0] * v.inv_w, v.tex[1] * v.inv_w, v.tex[2] * v.inv_w, v.tex[3] * v.inv_w]
}

/// What shading a triangle's quads needs once its setup is done, one record
/// for the triangle's life: the walk's setup, the vertex colours in 12.12,
/// the perspective unit's texture values and the planes for the rest.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub(crate) struct Tri {
    pub tri: Triangle,
    /// The draw's context, once the draw keeps the record for the shader
    /// thread (LiveDraw::push_tri()).
    pub live: *const LiveDraw,
    pub rgba12: [[i32; 4]; 3],
    pub flat12: [i32; 4],
    pub smooth: bool,
    /// The texture values at the vertices.
    pub tex: TriTex,
    /// 1/w and the user plane values at the vertices.
    pub inv_w: [f64; 3],
    pub clip: [[f64; 6]; 3],
}

impl Tri {
    /// The barycentrics of window pixel (px, py) in double.
    #[inline(always)]
    fn bary(&self, px: i32, py: i32) -> (f64, f64) {
        let e = self.tri.edges_at(px as i64, py as i64);
        let area = self.tri.area as f64;

        (e[1] as f64 / area, e[2] as f64 / area)
    }
}

impl Draw<'_> {
    /// Map and walk one triangle of a clipped polygon, given its three
    /// vertices in the view volume and, for flat shading, the provoking
    /// vertex of the triangle it was clipped from.
    pub fn clipped_triangle(&self, va: &VolumeVertex, vb: &VolumeVertex, vc: &VolumeVertex, provoking: &[f32]) {
        let p = self.pipe;
        let verts = [va, vb, vc];
        let mut v = [RasterVertex::default(); 3];

        for i in 0..3 {
            let Some(r) = self.xf.project(verts[i]) else {
                return;
            };
            v[i] = r;
        }
        let mut area = v[0].edge(&v[1], v[2].x, v[2].y);
        if area == 0 {
            return;
        }
        let (front, culled) = p.raster.facing(area);
        if culled {
            return;
        }
        let mut flat = &provoking[vtx::COLOR..vtx::COLOR + 4];
        if !front && self.xf.back_colours {
            flat = &provoking[vtx::BACK_COLOR..vtx::BACK_COLOR + 4];
            for i in 0..3 {
                let back = &verts[i].attr[vtx::BACK_COLOR..vtx::BACK_COLOR + 4];

                for k in 0..4 {
                    #[allow(clippy::manual_clamp, reason = "clamp would keep a NaN")]
                    let c = back[k].max(0.0).min(1.0);
                    v[i].rgba[k] = c as f64;
                }
            }
        }
        let flat = [0, 1, 2, 3].map(|k| flat[k] as f64);
        if area < 0 {
            // Walk either winding the same way from here.
            v.swap(1, 2);
            area = -area;
        }
        let Some((px0, py0, px1, py1)) = self.bounds else {
            return;
        };
        let repeat = [self.tex.regs.wrap(0) == Wrap::Repeat, self.tex.regs.wrap(1) == Wrap::Repeat];
        let tex = TriTex::setup(&v.map(|r| texture_values(&r)), repeat);
        let smooth = !p.raster.flat;
        let ts = Tri {
            tri: Triangle::setup(v.map(|r| (r.x, r.y)), v.map(|r| r.z), v.map(|r| r.depth_w), area, true),
            live: std::ptr::null(),
            rgba12: [0, 1, 2].map(|i| v[i].rgba.map(fixed::from_f64)),
            flat12: flat.map(fixed::from_f64),
            smooth,
            tex,
            inv_w: v.map(|r| r.inv_w),
            clip: v.map(|r| r.clip),
        };
        let rst = super::raster::RasterState { offset_steps: p.fill_steps, offset_fill: true, ..p.raster };

        if let (Some(live), Some(ring)) = (self.live, p.ring.as_ref().filter(|_| p.shade)) {
            // The shader thread shades the quads in the ring, from the draw's
            // context and the triangle's record.
            live.capture(self);
            let tp = live.push_tri(ts);

            live.used.set(true);
            p.used.set(true);
            // The triangle's tiles walk it, within the draw's bounds, in
            // plane pixels.
            let (tx0, ty0, tx1, ty1) = ts.tri.bounds;
            ring.push_tri(
                tp,
                rst.ox.wrapping_add(tx0.max(px0)),
                rst.oy.wrapping_add(ty0.max(py0)),
                rst.ox.wrapping_add(tx1.min(px1)),
                rst.oy.wrapping_add(ty1.min(py1)),
            );
            return;
        }
        triangle_quads(&rst, &ts.tri, Some((px0, py0, px1, py1)), |q| {
            let mut out = BackQuad { x: rst.ox.wrapping_add(q.x), y: rst.oy.wrapping_add(q.y), ..BackQuad::default() };

            self.shade_quad(&ts, q, &mut out);
            p.emit(&mut out);
        });
    }

    /// The shader's stage for one quad of triangle t: each drawn pixel's
    /// texel (the perspective unit's position, or the planes'), its colour
    /// interpolated, then shaded under the pipeline's words, into out with
    /// its depth word and 12-bit colour.
    pub fn shade_quad(&self, t: &Tri, q: &Quad, out: &mut BackQuad) {
        let p = self.pipe;
        let mut texel = [[0i32; 4]; 4];
        if self.textured {
            self.tex.sample_quad(&t.tex.quad(&q.l1, &q.l2), q.cov, &mut texel);
        }
        let iw = t.inv_w.map(|w| w as f32);
        // Window depth, which fog reads, from the barycentrics as the walk
        // takes it.
        let zw = |i: usize| -> f64 {
            if !p.shader.fog {
                return 0.0;
            }
            p.raster.window_depth32(t.tri.depth_ratio(q.l1[i], q.l2[i])).unwrap_or(0.0) as f64
        };

        for i in 0..4usize {
            if q.cov & (1 << i) == 0 {
                continue;
            }
            let px = q.x + (i & 1) as i32;
            let py = q.y + (i >> 1) as i32;
            let Some(background) = self.stipple_pass(px, py) else {
                continue;
            };
            if self.xf.user_planes() != 0 {
                let (b1, b2) = t.bary(px, py);
                let w = [(1.0 - b1 - b2) * t.inv_w[0], b1 * t.inv_w[1], b2 * t.inv_w[2]];
                let sum = w[0] + w[1] + w[2];
                let d: [f64; CLIP_PLANES as usize] = std::array::from_fn(|k| (w[0] * t.clip[0][k] + w[1] * t.clip[1][k] + w[2] * t.clip[2][k]) / sum);

                if self.xf.user_clipped(&d) {
                    continue;
                }
            }
            let mut c = t.flat12;

            if background {
                c = self.background.expect("an opaque stipple").map(fixed::from_f64);
            } else if t.smooth {
                let Some(w) = interp::weights(q.l1[i], q.l2[i], &iw) else {
                    continue;
                };
                for (k, ck) in c.iter_mut().enumerate() {
                    *ck = fixed::interpolate(&w, [t.rgba12[0][k], t.rgba12[1][k], t.rgba12[2][k]]);
                }
            }
            if !p.shader.shade12(self.textured, &mut c, &texel[i], zw(i)) {
                continue;
            }
            out.cov |= 1 << i;
            out.zq[i] = q.zq[i];
            out.rgba[i] = c.map(|c| colour12(c) as u16);
        }
    }
}

/// What the shader thread reads of a draw: the draw's context as the
/// raster thread set it up and the triangles' records. Written by the
/// raster thread before it pushes the quads that name it, read by the
/// shader thread after.
#[repr(C, align(64))]
pub(crate) struct LiveDraw {
    draw: std::cell::UnsafeCell<Option<Draw<'static>>>,
    tris: std::cell::UnsafeCell<Vec<Vec<Tri>>>,
    /// Some triangle went to the shader thread.
    pub used: std::cell::Cell<bool>,
}

impl Default for LiveDraw {
    fn default() -> Self {
        LiveDraw { draw: std::cell::UnsafeCell::new(None), tris: std::cell::UnsafeCell::new(Vec::new()), used: std::cell::Cell::new(false) }
    }
}

// SAFETY: the raster thread writes a LiveDraw before it publishes the quads
// that name it to the shader thread, and does not write it after; the
// shader thread only reads it.
unsafe impl Sync for LiveDraw {}
unsafe impl Send for LiveDraw {}

/// Triangles a chunk of LiveDraw::tris holds; a chunk never grows past it,
/// so a record's address is stable.
const LIVE_TRI_CHUNK: usize = 256;

impl LiveDraw {
    /// Keep the draw's context for the shader thread, the first time.
    pub fn capture(&self, d: &Draw) {
        // SAFETY: the raster thread alone writes it, before the shader
        // thread reads it (see the Sync impl).
        let slot = unsafe { &mut *self.draw.get() };

        if slot.is_none() {
            // SAFETY: what the context borrows lives in the draw's frame,
            // which outlives every quad the shader thread shades of it.
            *slot = Some(unsafe { std::mem::transmute::<Draw<'_>, Draw<'static>>(*d) });
        }
    }

    /// Keep a triangle's record; the address stays valid as long as the
    /// LiveDraw.
    pub fn push_tri(&self, t: Tri) -> *const Tri {
        // SAFETY: as capture().
        let tris = unsafe { &mut *self.tris.get() };

        if tris.last().is_none_or(|c| c.len() == LIVE_TRI_CHUNK) {
            tris.push(Vec::with_capacity(LIVE_TRI_CHUNK));
        }
        let chunk = tris.last_mut().expect("a chunk");
        chunk.push(Tri { live: self, ..t });
        chunk.last().expect("the record") as *const _
    }

    /// The shader thread's copy of the draw's context.
    pub fn shader_draw(&self) -> Draw<'static> {
        // SAFETY: captured before the first quad was pushed.
        unsafe { (*self.draw.get()).expect("a captured draw") }
    }
}

/// Draw d through setup: its primitives under the pipeline's state p, the
/// transform side's hand-over xf, the polygon stipple's rows where it is on
/// and an opaque stipple's background. A standalone line stipple's phase
/// goes back in d.line_phase.
#[allow(clippy::too_many_arguments, reason = "a draw's parts")]
pub(crate) fn run(p: &PipeDraw, d: &mut super::super::assembly::Prim, xf: Transform, stipple: Option<&[u32; 32]>, background: Option<[f64; 4]>, live: &LiveDraw) {
    use super::prims::LineStipple;

    let ps = &p.prims;
    let mut line = LineStipple {
        counter: match d.line_phase {
            Some(phase) if !d.line_phase_reset => phase,
            _ => 0,
        },
        enabled: ps.stipple,
        repeat: ps.stipple_repeat.clamp(1, 256),
        pattern: ps.stipple_pattern,
        consumed: false,
    };
    let use_stipple = d.line_phase.is_some() || ps.stipple;
    let lines = d.mode >= super::super::assembly::LINES && d.mode <= super::super::assembly::LINE_STRIP;

    line.counter %= 16u32.wrapping_mul(line.repeat);
    let (owned, bounds) = bounds(&p.raster, &xf);
    // Other windows cover all of this one; a standalone stipple still counts
    // the lines it does not draw.
    if bounds.is_none() && owned == NO_PIXELS && !(use_stipple && lines) {
        return;
    }
    let draw = Draw {
        pipe: p,
        xf,
        tex: p.tex,
        textured: p.textured,
        stipple,
        background,
        owned,
        bounds,
        live: Some(live),
    };
    draw.primitives(d, &mut line, use_stipple);
    if d.line_phase.is_some() && line.consumed {
        d.line_phase = Some(line.counter);
    }
}

/// An empty window.
const NO_PIXELS: (i32, i32, i32, i32) = (0, 0, -1, -1);

/// The window's own bounds in window pixels, from its clip rectangles, and
/// those within the viewport (Transform::bounds()); None for a window other
/// windows cover.
pub(crate) fn bounds(rst: &super::raster::RasterState, xf: &Transform) -> ((i32, i32, i32, i32), Option<(i32, i32, i32, i32)>) {
    let n = rst.clip_kept.min(4) as usize;

    if n == 0 {
        return (NO_PIXELS, None);
    }
    let r = &rst.clip_rect[..n];
    let owned = (
        r.iter().map(|r| r.x0).min().unwrap_or(0).wrapping_sub(rst.ox),
        r.iter().map(|r| r.y0).min().unwrap_or(0).wrapping_sub(rst.oy),
        r.iter().map(|r| r.x1).max().unwrap_or(-1).wrapping_sub(rst.ox),
        r.iter().map(|r| r.y1).max().unwrap_or(-1).wrapping_sub(rst.oy),
    );
    (owned, xf.bounds(owned))
}
