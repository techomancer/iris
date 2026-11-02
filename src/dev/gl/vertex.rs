//! The transformed vertex, clipping and the viewport mapping.

/// A vertex after transform: window coordinates (GL: y up), colour 0..1.
/// `T` texture coordinates and `F` fog factors ride along and are
/// interpolated by clipping; a board without them uses 0 (GR2: `Wv<0, 0>`,
/// IMPACT: `Wv<4, 1>`).
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Wv<const T: usize, const F: usize> {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub c: [f32; 4],
    /// Back-face colour (two-sided lighting; = c otherwise).
    pub cb: [f32; 4],
    /// 0 when the vertex is behind the eye (w <= 0): it has no window
    /// position and its primitive must be clipped.
    pub ok: u32,
    /// Clip planes the vertex is outside of (bit i = plane i), computed once
    /// at transform; 0 for vertices made by clipping or in window space.
    pub oc: u32,
    /// Clip-space position (frustum clipping) and eye-space position (user
    /// clip planes). Both are linear in the object position, so clipping
    /// interpolates them along an edge exactly.
    pub h: [f32; 4],
    pub e: [f32; 4],
    /// Texture coordinates (s, t, r, q), after the texture matrix.
    pub t: [f32; T],
    /// Fog factor (1 = no fog), when fog is applied per fragment.
    pub f: [f32; F],
}

// Not derived: `[f32; N]: Default` exists only for literal sizes.
impl<const T: usize, const F: usize> Default for Wv<T, F> {
    fn default() -> Self {
        Wv {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            c: [0.0; 4],
            cb: [0.0; 4],
            ok: 0,
            oc: 0,
            h: [0.0; 4],
            e: [0.0; 4],
            t: [0.0; T],
            f: [0.0; F],
        }
    }
}

/// Clip planes: 0..5 the view volume -w <= x, y, z <= w, 6..11 the user
/// planes (eye space).
pub const CLIP_PLANES: usize = 12;

/// Largest polygon `clip_polygon` handles: the input plus one vertex per
/// plane.
pub const MAX_POLY: usize = 32 + CLIP_PLANES;

/// Point on a-b at the crossing of a plane where the signed distances are
/// da and db: positions and colours interpolated, window position still to
/// be computed (`Viewport::project`).
pub fn clip_cross<const T: usize, const F: usize>(a: &Wv<T, F>, b: &Wv<T, F>, da: f32, db: f32) -> Wv<T, F> {
    let t = da / (da - db);
    let l = |x: f32, y: f32| x + (y - x) * t;
    let mut v = *a;
    v.oc = 0;
    for k in 0..4 {
        v.h[k] = l(a.h[k], b.h[k]);
        v.e[k] = l(a.e[k], b.e[k]);
        v.c[k] = l(a.c[k], b.c[k]);
        v.cb[k] = l(a.cb[k], b.cb[k]);
    }
    for k in 0..T {
        v.t[k] = l(a.t[k], b.t[k]);
    }
    for k in 0..F {
        v.f[k] = l(a.f[k], b.f[k]);
    }
    v
}

/// A vertex the clipper can cut: its clip- and eye-space positions, and the
/// point a fraction of the way along an edge. `Wv` is one; a board whose
/// vertices carry other attributes implements it for its own type.
pub trait ClipVertex: Copy {
    fn h(&self) -> &[f32; 4];
    fn e(&self) -> &[f32; 4];
    /// The point on a-b where a plane with signed distances da and db
    /// crosses it.
    fn cross(a: &Self, b: &Self, da: f32, db: f32) -> Self;
    /// Whether the vertex carries polygon-mode edge flags (the edge from it
    /// to the next vertex is a boundary edge). A clipped polygon then keeps
    /// the flags of the edges it keeps, flags the edges clipping adds, and
    /// adds no vertex where an edge leaves or enters exactly at a vertex.
    const EDGE_FLAGS: bool = false;
    fn edge(&self) -> bool {
        false
    }
    fn set_edge(&mut self, _edge: bool) {}
}

impl<const T: usize, const F: usize> ClipVertex for Wv<T, F> {
    #[inline(always)]
    fn h(&self) -> &[f32; 4] {
        &self.h
    }
    #[inline(always)]
    fn e(&self) -> &[f32; 4] {
        &self.e
    }
    #[inline(always)]
    fn cross(a: &Self, b: &Self, da: f32, db: f32) -> Self {
        clip_cross(a, b, da, db)
    }
}

/// How a geometry engine divides, snaps and clips, fixed per board at
/// compile time: `Frustum` for GR2 and IMPACT, `NearW` for VPro.
pub trait Geometry {
    /// The view-volume planes (bits 0..5 of `Clip::mask`) that clip
    /// geometrically.
    const VOLUME: u32;
    /// Signed distance of clip position h to view-volume plane i (0..5),
    /// >= 0 inside.
    fn volume(h: &[f32; 4], i: usize) -> f32;
    /// Normalized device coordinates of clip position h; None where it has
    /// no window position.
    fn divide(h: &[f32; 4]) -> Option<[f32; 3]>;
    /// Window coordinate v on the grid of 1/`subpixel` of a pixel.
    fn snap(v: f32, subpixel: f32) -> f32;
}

/// OpenGL's pipeline: clip to the whole view volume -w <= x, y, z <= w,
/// divide by w, round to the nearest grid point.
pub struct Frustum;

impl Geometry for Frustum {
    const VOLUME: u32 = 0x3f;
    #[inline(always)]
    fn volume(h: &[f32; 4], i: usize) -> f32 {
        match i {
            0 => h[3] + h[0],
            1 => h[3] - h[0],
            2 => h[3] + h[1],
            3 => h[3] - h[1],
            4 => h[3] + h[2],
            _ => h[3] - h[2],
        }
    }
    #[inline(always)]
    fn divide(h: &[f32; 4]) -> Option<[f32; 3]> {
        let [cx, cy, cz, cw] = *h;
        if cw <= 1e-6 {
            return None;
        }
        Some([cx / cw, cy / cw, cz / cw])
    }
    #[inline(always)]
    fn snap(v: f32, subpixel: f32) -> f32 {
        (v * subpixel).round() / subpixel
    }
}

/// VPro's geometry engine as its graded formats describe it.
///
/// - No near or far plane: the board does not clip the frustum
///   geometrically (inferred), and the raster drops a fragment by its depth
///   instead.
/// - The w plane. The library's preclip is off by default and the library
///   stores positions untested, so what the board does with w <= 0 is
///   unknown. iris clips on w >= `NearW::W_MIN` in place of plane 4, so
///   nothing at or behind the eye reaches the integer edge functions, and
///   the board logs the first time that plane clips anything
///   (`NearW::W_PLANE`).
/// - Planes 0..3 still clip x and y, to a guard band that is a limit of
///   iris's integer edge arithmetic, not of the board. The caller sets
///   the band: `WINDOW_REACH` (8192 pixels about the window) on the
///   combined-matrix path, `GL_WINDOW_GUARD` (16384 pixels from the
///   drawable's origin) on the window-matrix path, each its path's own
///   choice. 8192 matches the library's own class-3 guard (|x| or |y| >
///   8192 |w|) and is twice the largest drawable (under 4096). The hard
///   limit is the +-65536 check in the board's `project()`.
/// - The divide is a single-precision reciprocal rounded to nearest, then a
///   multiply: the order a vendor-simulated frame shows, where a truncated
///   reciprocal fails it.
/// - Window positions truncate toward minus infinity onto the grid: the
///   best-supported inference (the vendor frames bound the grid to 1/16
///   pixel or finer).
pub struct NearW;

impl NearW {
    /// The near-w plane's place: w >= W_MIN. The library's own clipper counts
    /// a vertex inside at distance >= 0 with no epsilon, and plane 4 could
    /// say w >= 0 the same way. iris keeps the smallest normal float so a
    /// vertex it keeps has w > 0 and divides. Single-precision
    /// interpolation can still land a crossing on w = 0, which `divide`
    /// refuses (the primitive is dropped and logged), and the x and y
    /// planes already need w >= max(|x|, |y|) >= 0, so the plane only
    /// matters for primitives edge-on through the eye. A modelling choice,
    /// not a hardware fact.
    pub const W_MIN: f32 = f32::MIN_POSITIVE;
    /// Bit of the w plane in an outcode or a clip mask.
    pub const W_PLANE: u32 = 1 << 4;
}

impl Geometry for NearW {
    const VOLUME: u32 = 0x1f;
    #[inline(always)]
    fn volume(h: &[f32; 4], i: usize) -> f32 {
        match i {
            0 => h[3] + h[0],
            1 => h[3] - h[0],
            2 => h[3] + h[1],
            3 => h[3] - h[1],
            _ => h[3] - NearW::W_MIN,
        }
    }
    #[inline(always)]
    fn divide(h: &[f32; 4]) -> Option<[f32; 3]> {
        let [cx, cy, cz, cw] = *h;
        if !(cw > 0.0) {
            return None;
        }
        let r = 1.0 / cw;
        Some([cx * r, cy * r, cz * r])
    }
    #[inline(always)]
    fn snap(v: f32, subpixel: f32) -> f32 {
        (v * subpixel).floor() / subpixel
    }
}

/// The clip planes in use. Plain data, valid zeroed.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct Clip {
    /// User clip planes (eye space) and their enable bits.
    pub user: [[f32; 4]; 6],
    pub user_on: u32,
}

impl Clip {
    /// Signed distance of `v` to clip plane `i` (>= 0 inside).
    pub fn dist<const T: usize, const F: usize>(&self, v: &Wv<T, F>, i: usize) -> f32 {
        self.dist_by::<Frustum, _>(v, i)
    }

    /// `dist` under geometry `G`.
    #[inline(always)]
    pub fn dist_by<G: Geometry, V: ClipVertex>(&self, v: &V, i: usize) -> f32 {
        if i < 6 {
            return G::volume(v.h(), i);
        }
        let (p, e) = (&self.user[i - 6], v.e());
        p[0] * e[0] + p[1] * e[1] + p[2] * e[2] + p[3] * e[3]
    }

    /// Planes in use: the view volume plus the enabled user planes.
    pub fn mask(&self) -> u32 {
        self.mask_by::<Frustum>()
    }

    /// `mask` under geometry `G`.
    #[inline(always)]
    pub fn mask_by<G: Geometry>(&self) -> u32 {
        G::VOLUME | ((self.user_on & 0x3f) << 6)
    }

    /// Bit i set when `v` is outside plane i.
    pub fn outcode<const T: usize, const F: usize>(&self, v: &Wv<T, F>) -> u32 {
        self.outcode_by::<Frustum, _>(v)
    }

    /// `outcode` under geometry `G`. A NaN distance counts as inside here
    /// (`< 0.0` is false), while the clip loops count it as outside
    /// (`>= 0.0` is false); a NaN vertex then reaches `project()`, which
    /// drops it.
    #[inline(always)]
    pub fn outcode_by<G: Geometry, V: ClipVertex>(&self, v: &V) -> u32 {
        let mask = self.mask_by::<G>();
        (0..CLIP_PLANES).filter(|&i| mask & (1 << i) != 0 && self.dist_by::<G, V>(v, i) < 0.0)
            .fold(0, |o, i| o | (1 << i))
    }

    /// Clip the polygon `v[..n]` against every plane its vertices are
    /// outside of, in place (Sutherland-Hodgman). Returns the new vertex
    /// count (0: nothing left). New vertices still need projecting.
    pub fn clip_polygon<const T: usize, const F: usize>(&self, v: &mut [Wv<T, F>; MAX_POLY], n: usize) -> usize {
        let any = v[..n].iter().fold(0, |o, w| o | w.oc);
        if any == 0 {
            return n;
        }
        let mut tmp = [Wv::default(); MAX_POLY];
        self.clip_polygon_by::<Frustum, _>(v, &mut tmp, n, any).0
    }

    /// `clip_polygon` under geometry `G` for the planes in `any` (the
    /// vertices' outcodes together), with `tmp` as large a workspace as
    /// `v`. Also says whether the polygon outgrew `v`, whose excess
    /// vertices are then dropped.
    #[inline(always)]
    pub fn clip_polygon_by<G: Geometry, V: ClipVertex>(&self, v: &mut [V], tmp: &mut [V], n: usize, any: u32) -> (usize, bool) {
        let cap = v.len().min(tmp.len());
        let mut n = n;
        let mut over = false;
        for plane in 0..CLIP_PLANES {
            if any & (1 << plane) == 0 || n == 0 {
                continue;
            }
            let mut m = 0;
            for i in 0..n {
                let (a, b) = (&v[i], &v[(i + 1) % n]);
                let (da, db) = (self.dist_by::<G, V>(a, plane), self.dist_by::<G, V>(b, plane));
                if da >= 0.0 {
                    if m < cap {
                        tmp[m] = *a;
                        m += 1;
                    } else {
                        over = true;
                    }
                    if !(db >= 0.0) {
                        if V::EDGE_FLAGS && da == 0.0 {
                            // Leaving at a: the edge clipping adds starts there.
                            // m >= 1: da >= 0 just pushed a (m == 0 < cap, cap >= 1) or set over.
                            tmp[m - 1].set_edge(true);
                        } else if m < cap {
                            tmp[m] = V::cross(a, b, da, db);
                            tmp[m].set_edge(true);
                            m += 1;
                        } else {
                            over = true;
                        }
                    }
                } else if db >= 0.0 && !(V::EDGE_FLAGS && db == 0.0) {
                    if m < cap {
                        tmp[m] = V::cross(a, b, da, db);
                        tmp[m].set_edge(a.edge());
                        m += 1;
                    } else {
                        over = true;
                    }
                }
            }
            v[..m].copy_from_slice(&tmp[..m]);
            n = m;
        }
        (n, over)
    }

    /// Clip the segment a-b; None when it is entirely outside.
    pub fn clip_line<const T: usize, const F: usize>(&self, a: Wv<T, F>, b: Wv<T, F>) -> Option<(Wv<T, F>, Wv<T, F>)> {
        let any = a.oc | b.oc;
        self.clip_line_by::<Frustum, _>(a, b, any)
    }

    /// `clip_line` under geometry `G` for the planes in `any`.
    #[inline(always)]
    pub fn clip_line_by<G: Geometry, V: ClipVertex>(&self, mut a: V, mut b: V, any: u32) -> Option<(V, V)> {
        for plane in 0..CLIP_PLANES {
            if any & (1 << plane) == 0 {
                continue;
            }
            let (da, db) = (self.dist_by::<G, V>(&a, plane), self.dist_by::<G, V>(&b, plane));
            match (da >= 0.0, db >= 0.0) {
                (true, true) => {}
                (false, false) => return None,
                (true, false) => b = V::cross(&a, &b, da, db),
                (false, true) => a = V::cross(&a, &b, da, db),
            }
        }
        Some((a, b))
    }
}

/// The viewport (window relative): x0, x1, y0, y1 (inclusive), z scale and
/// centre. Plain data, valid zeroed.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct Viewport {
    pub x0: f32,
    pub x1: f32,
    pub y0: f32,
    pub y1: f32,
    pub zscale: f32,
    pub zcenter: f32,
}

impl Viewport {
    /// glViewport(x, y, w, h) with glDepthRange(n, f) mapped onto
    /// 0..`zmax`.
    pub fn set(&mut self, x: f32, y: f32, w: f32, h: f32, n: f32, f: f32, zmax: f32) {
        (self.x0, self.x1, self.y0, self.y1) = (x, x + w - 1.0, y, y + h - 1.0);
        self.zscale = (f - n) * 0.5 * zmax;
        self.zcenter = (f + n) * 0.5 * zmax;
    }

    /// Window position of `v` from its clip position (`ok` = 0 if w <= 0),
    /// offset by `origin` and snapped to 1/`subpixel` of a pixel (the
    /// raster engine's fixed-point precision; removes float noise such as
    /// 319.99998 on edges).
    pub fn project<const T: usize, const F: usize>(&self, out: Wv<T, F>, origin: [f32; 2], subpixel: f32) -> Wv<T, F> {
        self.project_by::<Frustum, T, F>(out, origin, subpixel)
    }

    /// `project` under geometry `G`.
    #[inline(always)]
    pub fn project_by<G: Geometry, const T: usize, const F: usize>(&self, mut out: Wv<T, F>, origin: [f32; 2], subpixel: f32) -> Wv<T, F> {
        out.ok = 0;
        let Some([nx, ny, nz]) = G::divide(&out.h) else {
            return out;
        };
        let snap = |v: f32| G::snap(v, subpixel);
        out.x = snap(origin[0] + self.x0 + (nx + 1.0) * 0.5 * (self.x1 - self.x0 + 1.0));
        out.y = snap(origin[1] + self.y0 + (ny + 1.0) * 0.5 * (self.y1 - self.y0 + 1.0));
        out.z = self.zcenter + nz * self.zscale;
        out.ok = 1;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Wv = super::Wv<4, 1>;

    fn at(x: f32, y: f32) -> Wv {
        let mut v = Wv { h: [x, y, 0.0, 1.0], ..Default::default() };
        v.oc = Clip::default().outcode(&v);
        v
    }

    #[test]
    fn viewport_maps_ndc_corners_to_pixel_edges() {
        let mut vp = Viewport::default();
        vp.set(0.0, 0.0, 400.0, 300.0, 0.0, 1.0, 1.0);
        let a = vp.project(at(-1.0, -1.0), [0.0, 0.0], 256.0);
        let b = vp.project(at(1.0, 1.0), [0.0, 0.0], 256.0);
        assert_eq!((a.x, a.y, b.x, b.y), (0.0, 0.0, 400.0, 300.0));
    }

    #[test]
    fn polygon_straddling_the_right_plane_is_cut_at_x_equals_w() {
        let clip = Clip::default();
        let mut v = [Wv::default(); MAX_POLY];
        v[0] = at(0.0, 0.0);
        v[1] = at(2.0, 0.0);
        v[2] = at(0.0, 0.5);
        let n = clip.clip_polygon(&mut v, 3);
        assert_eq!(n, 4);
        assert!(v[..n].iter().all(|w| w.h[0] <= 1.0 + 1e-6));
    }

    type Ov = super::Wv<0, 0>;

    fn nearw(x: f32, y: f32, z: f32, w: f32) -> Ov {
        let mut v = Ov { h: [x, y, z, w], ..Default::default() };
        v.oc = Clip::default().outcode_by::<NearW, _>(&v);
        v
    }

    #[test]
    fn nearw_multiplies_by_the_reciprocal_where_frustum_divides() {
        // 7/49 and 5/49 differ in the last place between x * (1/w) and x / w.
        for (x, w) in [(7.0f32, 49.0f32), (5.0, 49.0)] {
            let h = [x, 0.0, 0.0, w];
            let n = NearW::divide(&h).unwrap()[0];
            let f = Frustum::divide(&h).unwrap()[0];
            assert_eq!(n, x * (1.0 / w));
            assert_eq!(f, x / w);
            assert_ne!(n, f, "{x}/{w}");
        }
    }

    #[test]
    fn nearw_floors_to_the_grid_where_frustum_rounds() {
        let v = 3.9999998f32;
        assert_eq!(NearW::snap(v, 256.0), 3.9960938);
        assert_eq!(Frustum::snap(v, 256.0), 4.0);
        // Toward minus infinity, not toward zero.
        assert_eq!(NearW::snap(-0.001, 256.0), -1.0 / 256.0);
        assert_eq!(NearW::snap(-3.0, 256.0), -3.0);
    }

    #[test]
    fn nearw_has_no_z_plane() {
        assert_eq!(NearW::VOLUME, 0x1f);
        let c = Clip::default();
        for z in [-5.0, 5.0] {
            assert_eq!(c.outcode_by::<NearW, _>(&nearw(0.0, 0.0, z, 1.0)), 0);
            assert_ne!(c.outcode::<4, 1>(&at_z(z)), 0);
        }
        let mut v = [nearw(0.0, 0.0, 9.0, 1.0), nearw(0.5, 0.0, -9.0, 1.0), nearw(0.0, 0.5, 9.0, 1.0)];
        let mut tmp = v;
        let (n, over) = c.clip_polygon_by::<NearW, _>(&mut v, &mut tmp, 3, 0);
        assert_eq!((n, over), (3, false));
        assert_eq!(v[1].h[2], -9.0);
    }

    fn at_z(z: f32) -> Wv {
        let mut v = Wv { h: [0.0, 0.0, z, 1.0], ..Default::default() };
        v.oc = Clip::default().outcode(&v);
        v
    }

    #[test]
    fn nearw_clips_at_the_w_plane() {
        let c = Clip::default();
        let behind = nearw(0.0, 0.0, 0.0, -1.0);
        let front = nearw(0.0, 0.0, 0.0, 1.0);
        assert_eq!(behind.oc & NearW::W_PLANE, NearW::W_PLANE);
        assert_eq!(front.oc, 0);
        // The x and y planes already cut the edge at w = 0, and the w plane
        // leaves it there or one rounding step above: nothing kept is behind
        // the eye.
        let (a, b) = c.clip_line_by::<NearW, _>(front, behind, behind.oc | front.oc).unwrap();
        assert_eq!(a.h[3], 1.0);
        assert!(b.h[3] >= 0.0 && b.h[3] < 1e-30, "{}", b.h[3]);
        // Wholly behind the eye: nothing.
        let behind2 = nearw(0.1, 0.0, 0.0, -2.0);
        assert!(c.clip_line_by::<NearW, _>(behind, behind2, behind.oc | behind2.oc).is_none());
        // Nothing divides at or behind the eye.
        assert!(NearW::divide(&[0.0, 0.0, 0.0, 0.0]).is_none());
        assert!(NearW::divide(&behind.h).is_none());
    }

    #[derive(Clone, Copy)]
    struct Ef {
        h: [f32; 4],
        edge: bool,
    }

    impl ClipVertex for Ef {
        fn h(&self) -> &[f32; 4] {
            &self.h
        }
        fn e(&self) -> &[f32; 4] {
            &self.h
        }
        fn cross(a: &Self, b: &Self, da: f32, db: f32) -> Self {
            let t = da / (da - db);
            Ef { h: std::array::from_fn(|k| a.h[k] + (b.h[k] - a.h[k]) * t), edge: false }
        }
        const EDGE_FLAGS: bool = true;
        fn edge(&self) -> bool {
            self.edge
        }
        fn set_edge(&mut self, edge: bool) {
            self.edge = edge;
        }
    }

    #[test]
    fn clipping_keeps_edge_flags_and_flags_the_edges_it_adds() {
        let c = Clip::default();
        let ef = |x: f32, edge: bool| Ef { h: [x, 0.0, 0.0, 1.0], edge };
        // Room for the vertex clipping adds: a triangle clipped by one plane
        // can have four.
        let mut v = [ef(0.0, true), ef(2.0, false), ef(0.0, false), ef(0.0, false)];
        v[2].h[1] = 0.5;
        let any = v[..3].iter().fold(0, |o, w| o | c.outcode_by::<NearW, _>(w));
        let mut tmp = v;
        let (n, over) = c.clip_polygon_by::<NearW, _>(&mut v, &mut tmp, 3, any);
        assert!(!over);
        assert_eq!(n, 4);
        // Kept vertex 0 keeps its flag; the leaving edge's new vertex is a
        // boundary edge; the entering edge's new vertex inherits the entry
        // edge's flag (false); the last kept vertex keeps its own.
        assert_eq!(v.map(|w| w.edge)[..n], [true, true, false, false]);
    }

    #[test]
    fn line_outside_one_plane_is_dropped() {
        assert!(Clip::default().clip_line(at(2.0, 0.0), at(3.0, 0.5)).is_none());
    }
}
