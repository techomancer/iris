//! The transformed vertex, clipping and the viewport mapping.

/// A vertex after transform: window coordinates (GL: y up), colour 0..1.
#[derive(Clone, Copy, Default, Debug)]
#[repr(C)]
pub struct Wv {
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
    pub t: [f32; 4],
    /// Fog factor (1 = no fog), when fog is applied per fragment.
    pub f: f32,
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
pub fn clip_cross(a: &Wv, b: &Wv, da: f32, db: f32) -> Wv {
    let t = da / (da - db);
    let l = |x: f32, y: f32| x + (y - x) * t;
    let mut v = *a;
    v.oc = 0;
    for k in 0..4 {
        v.h[k] = l(a.h[k], b.h[k]);
        v.e[k] = l(a.e[k], b.e[k]);
        v.c[k] = l(a.c[k], b.c[k]);
        v.cb[k] = l(a.cb[k], b.cb[k]);
        v.t[k] = l(a.t[k], b.t[k]);
    }
    v.f = l(a.f, b.f);
    v
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
    pub fn dist(&self, v: &Wv, i: usize) -> f32 {
        let h = &v.h;
        match i {
            0 => h[3] + h[0],
            1 => h[3] - h[0],
            2 => h[3] + h[1],
            3 => h[3] - h[1],
            4 => h[3] + h[2],
            5 => h[3] - h[2],
            _ => {
                let p = &self.user[i - 6];
                p[0] * v.e[0] + p[1] * v.e[1] + p[2] * v.e[2] + p[3] * v.e[3]
            }
        }
    }

    /// Planes in use: the view volume plus the enabled user planes.
    pub fn mask(&self) -> u32 {
        0x3f | ((self.user_on & 0x3f) << 6)
    }

    /// Bit i set when `v` is outside plane i.
    pub fn outcode(&self, v: &Wv) -> u32 {
        let mask = self.mask();
        (0..CLIP_PLANES).filter(|&i| mask & (1 << i) != 0 && self.dist(v, i) < 0.0)
            .fold(0, |o, i| o | (1 << i))
    }

    /// Clip the polygon `v[..n]` against every plane its vertices are
    /// outside of, in place (Sutherland-Hodgman). Returns the new vertex
    /// count (0: nothing left). New vertices still need projecting.
    pub fn clip_polygon(&self, v: &mut [Wv; MAX_POLY], n: usize) -> usize {
        let any = v[..n].iter().fold(0, |o, w| o | w.oc);
        if any == 0 {
            return n;
        }
        let mut n = n;
        let mut tmp = [Wv::default(); MAX_POLY];
        for plane in 0..CLIP_PLANES {
            if any & (1 << plane) == 0 || n == 0 {
                continue;
            }
            let mut m = 0;
            for i in 0..n {
                let (a, b) = (&v[i], &v[(i + 1) % n]);
                let (da, db) = (self.dist(a, plane), self.dist(b, plane));
                if da >= 0.0 && m < MAX_POLY {
                    tmp[m] = *a;
                    m += 1;
                }
                if (da >= 0.0) != (db >= 0.0) && m < MAX_POLY {
                    tmp[m] = clip_cross(a, b, da, db);
                    m += 1;
                }
            }
            v[..m].copy_from_slice(&tmp[..m]);
            n = m;
        }
        n
    }

    /// Clip the segment a-b; None when it is entirely outside.
    pub fn clip_line(&self, mut a: Wv, mut b: Wv) -> Option<(Wv, Wv)> {
        let any = a.oc | b.oc;
        for plane in 0..CLIP_PLANES {
            if any & (1 << plane) == 0 {
                continue;
            }
            let (da, db) = (self.dist(&a, plane), self.dist(&b, plane));
            match (da >= 0.0, db >= 0.0) {
                (true, true) => {}
                (false, false) => return None,
                (true, false) => b = clip_cross(&a, &b, da, db),
                (false, true) => a = clip_cross(&a, &b, da, db),
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
    pub fn project(&self, mut out: Wv, origin: [f32; 2], subpixel: f32) -> Wv {
        let [cx, cy, cz, cw] = out.h;
        out.ok = 0;
        if cw <= 1e-6 {
            return out;
        }
        let (nx, ny, nz) = (cx / cw, cy / cw, cz / cw);
        let snap = |v: f32| (v * subpixel).round() / subpixel;
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

    #[test]
    fn line_outside_one_plane_is_dropped() {
        assert!(Clip::default().clip_line(at(2.0, 0.0), at(3.0, 0.5)).is_none());
    }
}
