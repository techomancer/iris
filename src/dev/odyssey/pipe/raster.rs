//! The rasteriser: a primitive's window-space vertices to the 2x2 quads it
//! covers, under `RasterWords`. Each quad carries which of its four pixels
//! the primitive, the window's clip list, the scissor and the id test keep,
//! their barycentrics and their window depths, and the depth pre-test has
//! dropped the pixels that cannot pass the depth test
//! (`clanker/plans/odyssey-quad-pipeline.md`).
//!
//! The rules are the ones the vendor's simulator frames establish
//! (`ignore/odsy/docs/raster.md`): vertices truncated to 1/256 pixel, a
//! triangle lighting every pixel whose centre is inside it, with a centre
//! on an edge owned by one of two abutting triangles (the tie rule the
//! frames leave untested: a left or top edge owns it), and colour and
//! depth interpolated from exact integer edge functions. The walk is by
//! 8x8 tiles, as BUZZ subdivides, with tiles the edges exclude skipped and
//! tiles they wholly include taken without per-pixel tests.

use super::super::view::{Rows, MAX_DIM};
use super::state::{Compare, Cull, RasterWords};
use super::super::consts::{ODSY_ID_COMPARE_ENABLE, ODSY_ID_COMPARE_MASK_SHIFT};
use super::super::depth_word::{word_for_fraction32_lanes, word_with_offset};
use super::super::consts::ODSY_DEPTH24_MAX;
use super::lanes;
use fearless_simd::{dispatch, Simd};
use super::super::clip::ClipRect;

/// Window coordinates snap to 1/2^SUBPIXEL_BITS pixel (established).
pub(crate) const SUBPIXEL_BITS: u32 = 8;
const SUB: i64 = 1 << SUBPIXEL_BITS;

/// The tile the walk subdivides to, in pixels.
pub(crate) const TILE: i32 = 8;

/// A quad's pixel i is column x + LANE_X[i] of row y + LANE_Y[i].
pub(crate) const LANE_X: [i64; 4] = [0, 1, 0, 1];
pub(crate) const LANE_Y: [i64; 4] = [0, 0, 1, 1];

/// A vertex as the rasteriser takes it: window position in subpixels,
/// depth over w and its divisor, 1/w for perspective, and the attributes
/// the shader interpolates.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub(crate) struct RasterVertex {
    pub x: i64,
    pub y: i64,
    /// Depth over w (the depth row's product with the eye coordinates over
    /// clip w), which interpolates linearly in the window.
    pub z: f64,
    pub inv_w: f64,
    /// The depth divisor over w.
    pub depth_w: f64,
    pub rgba: [f64; 4],
    /// The separate specular colour, RGB (zero unless the program keeps it).
    pub spec: [f64; 3],
    pub tex: [f64; 4],
    pub clip: [f64; 6],
}

impl Default for RasterVertex {
    fn default() -> Self {
        RasterVertex { x: 0, y: 0, z: 0.0, inv_w: 0.0, depth_w: 0.0, rgba: [0.0; 4], spec: [0.0; 3], tex: [0.0; 4], clip: [0.0; 6] }
    }
}

impl RasterVertex {
    /// The edge function from self to b at subpixel point (px, py): positive
    /// to the left of the edge for a counter-clockwise triangle.
    #[inline(always)]
    pub fn edge(&self, b: &RasterVertex, px: i64, py: i64) -> i64 {
        (b.x.wrapping_sub(self.x)).wrapping_mul(py.wrapping_sub(self.y))
            .wrapping_sub((b.y.wrapping_sub(self.y)).wrapping_mul(px.wrapping_sub(self.x)))
    }

}

/// A 2x2 quad of a primitive: pixel i is column `x + (i & 1)` of row
/// `y + (i >> 1)` in window pixels (the draw's coordinates, before the
/// window origin). `cov` bit i is set for a drawn pixel. `b1` and `b2`
/// are the pixels' barycentrics for the triangle's second and third
/// vertices, set for all four pixels (the level of detail reads the
/// undrawn ones). `zw` is the drawn pixels' window depth and `zq` their
/// depth word with the polygon offset applied.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub(crate) struct Quad {
    pub x: i32,
    pub y: i32,
    pub cov: u8,
    /// The pixels' barycentrics for the second and third vertices, as the
    /// pipeline's walk gives them (all four pixels); the old fill loops
    /// use `b1` and `b2` instead.
    pub l1: [f32; 4],
    pub l2: [f32; 4],
    pub b1: [f64; 4],
    pub b2: [f64; 4],
    pub zw: [f64; 4],
    pub zq: [u32; 4],
}

impl Default for Quad {
    fn default() -> Self {
        Quad { x: 0, y: 0, cov: 0, l1: [0.0; 4], l2: [0.0; 4], b1: [0.0; 4], b2: [0.0; 4], zw: [0.0; 4], zq: [0; 4] }
    }
}

/// What the rasteriser decides once per draw.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub(crate) struct RasterState {
    pub cull: Cull,
    pub front_cw: bool,
    /// Flat shading: the provoking vertex's colour.
    pub flat: bool,
    /// The window's origin in the plane.
    pub ox: i32,
    pub oy: i32,
    /// The window's clip rectangles in plane coordinates, and the scissor
    /// in window coordinates, which bound the walk and own the pixels.
    pub clip_rect: [ClipRect; 4],
    pub clip_kept: u32,
    pub scissor: Option<(i32, i32, i32, i32)>,
    /// The destination's width, which bounds the walk.
    pub width: u32,
    /// The id test: the window-id bytes, the value and the mask.
    pub did: Rows,
    pub id_value: u8,
    pub id_mask: u8,
    /// The depth range [n, f] and the slack past [0, 1] a depth may round
    /// (from the transform side for now).
    pub depth_range: [f64; 2],
    pub depth_slack: f64,
    /// The polygon offset for fills, split between window depth and
    /// depth-word steps, and whether it applies.
    pub offset_fill: bool,
    pub offset_steps: i64,
    /// The depth pre-test: the comparison, where the depth test is on and
    /// no stencil operation wants failed fragments.
    pub pretest: Option<Compare>,
    pub depth: Option<Rows>,
    /// The depth word of a window depth (RasterRules::depth_code) and the
    /// unsigned-float depth encoder, as the board's rules make them.
    pub depth_code: fn(f64) -> u32,
    /// The walk takes depth in 32-bit floats from the triangle's planes
    /// and encodes it with word_for_fraction32 (`Triangle::depth4`).
    pub depth32: bool,
}

impl RasterState {
    /// Whether the window keeps plane pixel (x, y): its clip list, and the
    /// id test.
    #[inline(always)]
    pub fn owned(&self, x: i32, y: i32) -> bool {
        if x < 0 || y < 0 || x as u32 >= self.width || y as u32 >= MAX_DIM {
            return false;
        }
        let n = self.clip_kept.min(4) as usize;

        // No rectangle kept: the window owns nothing.
        if !self.clip_rect[..n].iter().any(|r| x >= r.x0 && x <= r.x1 && y >= r.y0 && y <= r.y1) {
            return false;
        }
        if self.id_mask != 0 {
            let id = self.did.load1(x, y).unwrap_or(0);

            if (id ^ self.id_value) & self.id_mask != 0 {
                return false;
            }
        }
        true
    }

    /// window_depth() in 32 bits for the ratio already taken.
    #[inline(always)]
    pub fn window_depth32(&self, z: f32) -> Option<f32> {
        let mut z = z;
        let slack = self.depth_slack as f32;

        if z < 0.0 && z >= -slack {
            z = 0.0;
        } else if z > 1.0 && z <= 1.0 + slack {
            z = 1.0;
        }
        if !(0.0..=1.0).contains(&z) {
            return None;
        }
        let (n, f) = (self.depth_range[0] as f32, self.depth_range[1] as f32);
        let zw = n + (f - n) * z;

        Some(if zw >= 1.0 { 1.0 } else if zw > 0.0 { zw } else { 0.0 })
    }

    /// A fragment's window depth from its interpolated depth over w and
    /// divisor over w: the ratio, mapped through the depth range, with the
    /// polygon offset's window part (none here: the board's offset is in
    /// depth-word steps) and the slack clamps.
    #[inline(always)]
    pub fn window_depth(&self, z_over_w: f64, depth_w: f64) -> Option<f64> {
        let mut z = z_over_w / depth_w;
        let slack = self.depth_slack;

        if z < 0.0 && z >= -slack {
            z = 0.0;
        } else if z > 1.0 && z <= 1.0 + slack {
            z = 1.0;
        }
        if !(0.0..=1.0).contains(&z) {
            return None;
        }
        let (n, f) = (self.depth_range[0], self.depth_range[1]);
        let zw = n + (f - n) * z;

        Some(if zw >= 1.0 { 1.0 } else if zw > 0.0 { zw } else { 0.0 })
    }

    /// The state of a draw under words w, with the window's id bytes and
    /// the depth plane for the pre-test.
    #[allow(clippy::too_many_arguments, reason = "the stage takes its inputs as parts for now")]
    pub fn of(w: &RasterWords, width: u32, did: Rows, depth_range: [f64; 2], depth_slack: f64, depth_code: fn(f64) -> u32, depth: Option<Rows>, pretest_allowed: bool) -> RasterState {
        let id_mask = if w.id_compare & ODSY_ID_COMPARE_ENABLE != 0 { (w.id_compare >> ODSY_ID_COMPARE_MASK_SHIFT) as u8 } else { 0 };
        let (sx0, sy0, sx1, sy1) = w.scissor();

        RasterState {
            cull: w.cull(),
            // Bit 23 is the library's GL_CW; the walk counts rows down the
            // plane, which mirrors the winding, so it is counter-clockwise
            // here (the model's reading since gl::front_face()).
            front_cw: !w.front_cw(),
            flat: w.flat(),
            ox: w.origin[0],
            oy: w.origin[1],
            clip_rect: w.clip_rect,
            clip_kept: w.clip_kept,
            scissor: Some((sx0, sy0, sx1, sy1)),
            width,
            did,
            id_value: w.id_compare as u8,
            id_mask,
            depth_range,
            depth_slack,
            offset_fill: w.offset_fill(),
            offset_steps: w.offset_steps() as i64,
            pretest: (pretest_allowed && depth.is_some()).then(|| w.depth_func()),
            depth,
            depth_code,
            depth32: false,
        }
    }
}

/// A triangle's setup for the walk, derived once from its vertices and
/// kept, with the rest of the triangle, in its one record for its life
/// (`raster_triangle::TriShade`): the three edge functions as integer
/// planes over pixel centres, their tie rule, the area, the pixel bounds,
/// and depth over w and the depth divisor over w at the vertices. The
/// edges' values divided by the area are the pixel's barycentrics, from
/// which every later stage interpolates.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub(crate) struct Triangle {
    /// e0, e1 and e2 (the weights of vertices 0, 1 and 2 times the area)
    /// at the centre of pixel (0, 0), and their steps for one pixel in x
    /// and in y. Wrapping arithmetic: the values a pixel needs are exact.
    pub e00: [i64; 3],
    pub dx: [i64; 3],
    pub dy: [i64; 3],
    /// What each edge value must reach for the centre to be owned: 0 for
    /// an owning edge, 1 for the other.
    pub need: [i64; 3],
    /// The area in edge units (subpixels squared), positive, and its
    /// reciprocal.
    pub area: i64,
    pub inv_area: f32,
    /// The pixel bounding box, inclusive.
    pub bounds: (i32, i32, i32, i32),
    /// Depth over w and the depth divisor over w at the three vertices.
    pub z: [f64; 3],
    pub d: [f64; 3],
    pub front: bool,
}

impl RasterState {
    /// Whether a triangle of signed area `area` (subpixel units, as the
    /// vertices are wound) faces front, and whether it is culled.
    pub fn facing(&self, area: i64) -> (bool, bool) {
        let ccw = area > 0;
        let front = if self.front_cw { !ccw } else { ccw };
        let culled = match self.cull {
            Cull::Both => true,
            Cull::Back => !front,
            Cull::Front => front,
            Cull::None => false,
        };

        (front, culled)
    }
}

impl Triangle {
#[cfg(test)]
    /// The triangle of three vertices, or None where it has no area or is
    /// culled.
    pub fn of(st: &RasterState, a: &RasterVertex, b: &RasterVertex, c: &RasterVertex) -> Option<Triangle> {
        let mut v = [*a, *b, *c];
        let mut area = v[0].edge(&v[1], v[2].x, v[2].y);

        if area == 0 {
            return None;
        }
        let (front, culled) = st.facing(area);

        if culled {
            return None;
        }
        if area < 0 {
            v.swap(1, 2);
            area = -area;
        }
        Some(Triangle::wound(v, area, front))
    }

#[cfg(test)]
    /// The triangle of vertices already wound counter-clockwise, with
    /// their positive area, decided front and not culled.
    pub fn wound(v: [RasterVertex; 3], area: i64, front: bool) -> Triangle {
        Triangle::setup(v.map(|r| (r.x, r.y)), v.map(|r| r.z), v.map(|r| r.depth_w), area, front)
    }

    /// The setup of a triangle wound counter-clockwise whose vertices are
    /// at subpixel positions p, with depth over w z and divisor over w d.
    pub fn setup(p: [(i64, i64); 3], z: [f64; 3], d: [f64; 3], area: i64, front: bool) -> Triangle {
        let edge = |a: (i64, i64), b: (i64, i64), px: i64, py: i64| -> i64 {
            (b.0.wrapping_sub(a.0)).wrapping_mul(py.wrapping_sub(a.1)).wrapping_sub((b.1.wrapping_sub(a.1)).wrapping_mul(px.wrapping_sub(a.0)))
        };
        // The tie rule: a left edge (running down) or a top edge
        // (horizontal, running left) owns the centres on it.
        let owns = |a: (i64, i64), b: (i64, i64)| b.1 < a.1 || (b.1 == a.1 && b.0 < a.0);
        // e0 is the edge v1->v2 (weight of v0), e1 is v2->v0 (weight of
        // v1), e2 is v0->v1 (weight of v2).
        let pairs = [(p[1], p[2]), (p[2], p[0]), (p[0], p[1])];
        let c = SUB / 2;
        let e00 = pairs.map(|(a, b)| edge(a, b, c, c));
        let dx = pairs.map(|(a, b)| -(b.1 - a.1) * SUB);
        let dy = pairs.map(|(a, b)| (b.0 - a.0) * SUB);
        let need = pairs.map(|(a, b)| if owns(a, b) { 0 } else { 1 });
        let sub = SUBPIXEL_BITS;
        let bounds = (
            (p[0].0.min(p[1].0.min(p[2].0)) >> sub) as i32,
            (p[0].1.min(p[1].1.min(p[2].1)) >> sub) as i32,
            (p[0].0.max(p[1].0.max(p[2].0)) >> sub) as i32,
            (p[0].1.max(p[1].1.max(p[2].1)) >> sub) as i32,
        );

        Triangle { e00, dx, dy, need, area, inv_area: (1.0 / area as f64) as f32, bounds, z, d, front }
    }

    /// The three edge values at the centre of pixel (px, py).
    #[inline(always)]
    pub fn edges_at(&self, px: i64, py: i64) -> [i64; 3] {
        [0, 1, 2].map(|k| self.e00[k].wrapping_add(self.dx[k].wrapping_mul(px)).wrapping_add(self.dy[k].wrapping_mul(py)))
    }

    /// A quad's depth at its barycentrics (l1, l2) under st, in 32 bits, as
    /// lanes: depth_ratio() through RasterState::window_depth32(), its word
    /// (word_for_fraction32()) with `steps` of polygon offset, and the
    /// pixels that have a depth.
    #[inline(always)]
    pub fn depth4(&self, st: &RasterState, l1: &[f32; 4], l2: &[f32; 4], steps: i64) -> ([f32; 4], [u32; 4], u8) {
        dispatch!(lanes::level(), s => self.depth4_lanes(s, st, l1, l2, steps))
    }

    #[inline(always)]
    fn depth4_lanes<S: Simd>(&self, s: S, st: &RasterState, l1: &[f32; 4], l2: &[f32; 4], steps: i64) -> ([f32; 4], [u32; 4], u8) {
        let k = |v: f32| s.splat_f32x4(v);
        let (l1, l2) = (s.load_array_f32x4(*l1), s.load_array_f32x4(*l2));
        let plane = |v: &[f64; 3]| s.add_f32x4(s.add_f32x4(k(v[0] as f32), s.mul_f32x4(l1, k((v[1] - v[0]) as f32))), s.mul_f32x4(l2, k((v[2] - v[0]) as f32)));
        let z = s.div_f32x4(plane(&self.z), plane(&self.d));
        // window_depth32(): the slack clamps, the range, the clamp to [0, 1].
        let slack = st.depth_slack as f32;
        let low = s.and_mask32x4(s.simd_lt_f32x4(z, k(0.0)), s.simd_ge_f32x4(z, k(-slack)));
        let high = s.and_mask32x4(s.simd_gt_f32x4(z, k(1.0)), s.simd_le_f32x4(z, k(1.0 + slack)));
        let z = s.select_f32x4(low, k(0.0), s.select_f32x4(high, k(1.0), z));
        let kept = s.and_mask32x4(s.simd_ge_f32x4(z, k(0.0)), s.simd_le_f32x4(z, k(1.0)));
        let (n, f) = (st.depth_range[0] as f32, st.depth_range[1] as f32);
        let zw = s.add_f32x4(k(n), s.mul_f32x4(k(f - n), z));
        let zw = s.select_f32x4(s.simd_ge_f32x4(zw, k(1.0)), k(1.0), s.select_f32x4(s.simd_gt_f32x4(zw, k(0.0)), zw, k(0.0)));
        // word_with_offset(): the word is below 2^24, so steps past 2^25
        // either way clamp as those at 2^25 do.
        let steps = s.splat_i32x4(steps.clamp(-(1 << 25), 1 << 25) as i32);
        let word = s.add_i32x4(word_for_fraction32_lanes(s, zw), steps);
        let word = s.min_i32x4(s.max_i32x4(word, s.splat_i32x4(0)), s.splat_i32x4(ODSY_DEPTH24_MAX as i32));

        (s.as_array_f32x4(zw), s.as_array_i32x4(word).map(|w| w as u32), lanes::bits(s, kept))
    }

    /// Depth over the divisor at barycentrics (l1, l2), in 32 bits.
    #[inline(always)]
    pub fn depth_ratio(&self, l1: f32, l2: f32) -> f32 {
        let (z, d) = (&self.z, &self.d);
        let zz = z[0] as f32 + l1 * (z[1] - z[0]) as f32 + l2 * (z[2] - z[0]) as f32;
        let dd = d[0] as f32 + l1 * (d[1] - d[0]) as f32 + l2 * (d[2] - d[0]) as f32;

        zz / dd
    }
}

/// The window pixels the walk of t under st covers, inclusive: the
/// triangle's bounds within `within` (window pixels), the scissor (window
/// coordinates), the clip rectangles and the plane (plane coordinates);
/// None when nothing is left.
pub(crate) fn walk_bounds(st: &RasterState, t: &Triangle, within: Option<(i32, i32, i32, i32)>) -> Option<(i32, i32, i32, i32)> {
    let (mut x0, mut y0, mut x1, mut y1) = t.bounds;

    if let Some((wx0, wy0, wx1, wy1)) = within {
        x0 = x0.max(wx0);
        y0 = y0.max(wy0);
        x1 = x1.min(wx1);
        y1 = y1.min(wy1);
    }
    if let Some((sx0, sy0, sx1, sy1)) = st.scissor {
        x0 = x0.max(sx0);
        y0 = y0.max(sy0);
        x1 = x1.min(sx1);
        y1 = y1.min(sy1);
    }
    let n = st.clip_kept.min(4) as usize;
    if n == 0 {
        return None;
    }
    {
        let cx0 = st.clip_rect[..n].iter().map(|r| r.x0).min().unwrap();
        let cy0 = st.clip_rect[..n].iter().map(|r| r.y0).min().unwrap();
        let cx1 = st.clip_rect[..n].iter().map(|r| r.x1).max().unwrap();
        let cy1 = st.clip_rect[..n].iter().map(|r| r.y1).max().unwrap();

        x0 = x0.max(cx0.wrapping_sub(st.ox));
        y0 = y0.max(cy0.wrapping_sub(st.oy));
        x1 = x1.min(cx1.wrapping_sub(st.ox));
        y1 = y1.min(cy1.wrapping_sub(st.oy));
    }
    x0 = x0.max(0i32.wrapping_sub(st.ox));
    y0 = y0.max(0i32.wrapping_sub(st.oy));
    x1 = x1.min((st.width as i32 - 1).wrapping_sub(st.ox));
    y1 = y1.min((MAX_DIM as i32 - 1).wrapping_sub(st.oy));
    if x0 > x1 || y0 > y1 {
        return None;
    }
    Some((x0, y0, x1, y1))
}

/// Walk a triangle into quads under st, handing each quad with any drawn
/// pixel to emit. `within`, where given, bounds the walk further, in
/// window pixels (the viewport the transform side keeps).
pub(crate) fn triangle_quads(st: &RasterState, t: &Triangle, within: Option<(i32, i32, i32, i32)>, mut emit: impl FnMut(&Quad)) {
    let Some((x0, y0, x1, y1)) = walk_bounds(st, t, within) else {
        return;
    };

    let area = t.area as f64;
    let steps = if st.offset_fill { st.offset_steps } else { 0 };
    let dz1 = t.z[1] - t.z[0];
    let dz2 = t.z[2] - t.z[0];
    let mut q = Quad::default();
    // Tiles, from the tile holding (x0, y0), on the plane's grid: a tile
    // and a quad (the 2x2 the level of detail reads) start at plane
    // coordinates a multiple of TILE and of two, wherever the window is.
    let (ox, oy) = (st.ox, st.oy);
    let tx0 = (x0.wrapping_add(ox).div_euclid(TILE) * TILE).wrapping_sub(ox);
    let ty0 = (y0.wrapping_add(oy).div_euclid(TILE) * TILE).wrapping_sub(oy);
    let mut ty = ty0;

    while ty <= y1 {
        let mut tx = tx0;

        while tx <= x1 {
            // The edge values at the four corner pixel centres of the tile.
            let e00 = t.edges_at(tx as i64, ty as i64);
            let ex = t.dx.map(|d| d * (TILE as i64 - 1));
            let ey = t.dy.map(|d| d * (TILE as i64 - 1));
            let mut all_in = true;
            let mut any_in = true;

            for k in 0..3 {
                let c = [e00[k], e00[k] + ex[k], e00[k] + ey[k], e00[k] + ex[k] + ey[k]];
                let lo = c.iter().copied().min().unwrap();
                let hi = c.iter().copied().max().unwrap();

                all_in &= lo >= t.need[k];
                any_in &= hi >= t.need[k];
            }
            if !any_in {
                tx += TILE;
                continue;
            }
            let qx0 = tx.max((x0.wrapping_add(ox) & !1).wrapping_sub(ox));
            let qy0 = ty.max((y0.wrapping_add(oy) & !1).wrapping_sub(oy));
            let qx1 = (tx + TILE - 1).min(x1);
            let qy1 = (ty + TILE - 1).min(y1);
            let mut qy = qy0;

            while qy <= qy1 {
                let mut qx = qx0;

                while qx <= qx1 {
                    let e = t.edges_at(qx as i64, qy as i64);
                    // The four pixels as lanes: the edge values, and the
                    // barycentrics for every pixel of the quad (the level
                    // of detail reads the undrawn ones).
                    let ei: [[i64; 4]; 3] = std::array::from_fn(|k| std::array::from_fn(|i| e[k] + t.dx[k] * LANE_X[i] + t.dy[k] * LANE_Y[i]));

                    q.x = qx;
                    q.y = qy;
                    q.l1 = ei[1].map(|v| v as f32 * t.inv_area);
                    q.l2 = ei[2].map(|v| v as f32 * t.inv_area);
                    let mut cov = 0u8;
                    for i in 0..4usize {
                        let (px, py) = (qx + LANE_X[i] as i32, qy + LANE_Y[i] as i32);
                        let inside = px >= x0 && px <= x1 && py >= y0 && py <= y1;
                        let edges = all_in || (ei[0][i] >= t.need[0] && ei[1][i] >= t.need[1] && ei[2][i] >= t.need[2]);
                        cov |= ((inside && edges) as u8) << i;
                    }
                    for i in 0..4usize {
                        if cov & (1 << i) != 0 && !st.owned(st.ox.wrapping_add(qx + LANE_X[i] as i32), st.oy.wrapping_add(qy + LANE_Y[i] as i32)) {
                            cov &= !(1 << i);
                        }
                    }
                    if cov == 0 {
                        qx += 2;
                        continue;
                    }
                    if st.depth32 {
                        // Depth from the barycentrics, in 32 bits.
                        let (zw, zq, kept) = t.depth4(st, &q.l1, &q.l2, steps);
                        cov &= kept;
                        q.zw = zw.map(|z| z as f64);
                        q.zq = zq;
                    } else {
                        for i in 0..4usize {
                            let b1 = ei[1][i] as f64 / area;
                            let b2 = ei[2][i] as f64 / area;
                            let z = t.z[0] + b1 * dz1 + b2 * dz2;
                            let depth_w = (1.0 - b1 - b2) * t.d[0] + b1 * t.d[1] + b2 * t.d[2];
                            match st.window_depth(z, depth_w) {
                                Some(zw) => {
                                    q.zw[i] = zw;
                                    q.zq[i] = word_with_offset((st.depth_code)(zw), steps);
                                }
                                None => cov &= !(1 << i),
                            }
                        }
                    }
                    if let (Some(func), Some(depth)) = (st.pretest, st.depth) {
                        for i in 0..4usize {
                            if cov & (1 << i) == 0 {
                                continue;
                            }
                            if let Some(stored) = depth.load_word(st.ox.wrapping_add(qx + LANE_X[i] as i32), st.oy.wrapping_add(qy + LANE_Y[i] as i32)) {
                                if !func.test(q.zq[i], stored) {
                                    cov &= !(1 << i);
                                }
                            }
                        }
                    }
                    q.cov = cov;
                    if q.cov != 0 {
                        emit(&q);
                    }
                    qx += 2;
                }
                qy += 2;
            }
            tx += TILE;
        }
        ty += TILE;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::super::board::ODSY_WIDTH;
    use super::super::super::consts::gl_cmd;
    use super::super::super::depth_word::word_for_fraction32;
    use super::super::super::raster_bench::Lcg;
    use std::collections::HashSet;

    /// depth4() is, at each pixel, the scalar path: depth_ratio() through
    /// window_depth32(), then word_for_fraction32() and the offset, over
    /// random planes (some with no divisor), barycentrics, depth ranges,
    /// slacks and offsets as far as the field reaches and past it.
    #[test]
    fn the_lane_depth_is_the_scalar_one() {
        let mut rng = Lcg(29);
        let mut r = |lo: f64, hi: f64| rng.range(lo as f32, hi as f32) as f64;
        let words = RasterWords { clip_kept: 1, ..Default::default() };
        let cases: Vec<_> = (0..20_000)
            .map(|n| {
                let range = [r(-0.2, 0.6), r(0.4, 1.2)];
                let slack = if n % 3 == 0 { 0.0 } else { r(0.0, 0.01) };
                let st = RasterState::of(&words, ODSY_WIDTH, Rows::default(), range, slack, super::super::super::depth_word::word_for_fraction, None, false);
                let z = [r(-1.5, 1.5), r(-1.5, 1.5), r(-1.5, 1.5)];
                let d = if n % 17 == 0 { [0.0; 3] } else { [r(0.1, 3.0), r(0.1, 3.0), r(0.1, 3.0)] };
                let t = Triangle::setup([(0, 0), (256, 0), (0, 256)], z, d, 1, true);
                let l1: [f32; 4] = std::array::from_fn(|_| r(-0.2, 1.2) as f32);
                let l2: [f32; 4] = std::array::from_fn(|_| r(-0.2, 1.2) as f32);
                let steps = match n % 5 {
                    0 => 0,
                    1 => r(-16777216.0, 16777216.0) as i64,
                    2 => r(-1.1e12, 1.1e12) as i64,
                    _ => r(-1000.0, 1000.0) as i64,
                };
                (st, t, l1, l2, steps)
            })
            .collect();
        super::super::lanes::each_level!(s, level => {
            for (n, (st, t, l1, l2, steps)) in cases.iter().enumerate() {
                let (zw, zq, kept) = t.depth4_lanes(s, st, l1, l2, *steps);
                for i in 0..4 {
                    match st.window_depth32(t.depth_ratio(l1[i], l2[i])) {
                        Some(want) => {
                            assert!(kept & (1 << i) != 0, "{level}: round {n} pixel {i}");
                            assert_eq!(zw[i].to_bits(), want.to_bits(), "{level}: round {n} pixel {i}");
                            assert_eq!(zq[i], word_with_offset(word_for_fraction32(want), *steps), "{level}: round {n} pixel {i}");
                        }
                        None => assert!(kept & (1 << i) == 0, "{level}: round {n} pixel {i}"),
                    }
                }
            }
        });
    }

    /// The tiled walk keeps exactly the pixels whose centres the edge
    /// functions and their tie rule own, each once, over random triangles
    /// of every size: skipping a tile the edges exclude and taking one they
    /// include whole loses and adds nothing.
    #[test]
    fn the_tiled_walk_keeps_exactly_the_owned_centres() {
        let mut rng = Lcg(11);
        let words = RasterWords {
            clip_rect: [ClipRect { x0: 0, y0: 0, x1: ODSY_WIDTH as i32 - 1, y1: 1023 }; 4],
            clip_kept: 1,
            clip_box: [0xfff, 0xfff],
            ..Default::default()
        };
        let st = RasterState::of(&words, ODSY_WIDTH, Rows::default(), [0.0, 1.0], gl_cmd::DEPTH_SLACK, super::super::super::depth_word::word_for_fraction, None, false);
        let mut quads = 0usize;

        for round in 0..200 {
            let size = if round % 4 == 0 { 300.0 } else { 20.0 };
            let (cx, cy) = (rng.range(0.0, 1000.0), rng.range(0.0, 800.0));
            let v: [RasterVertex; 3] = std::array::from_fn(|_| RasterVertex {
                x: ((cx + rng.range(-size, size)) * 256.0) as i64,
                y: ((cy + rng.range(-size, size)) * 256.0) as i64,
                z: 0.5,
                inv_w: 1.0,
                depth_w: 1.0,
                ..Default::default()
            });
            let Some(t) = Triangle::of(&st, &v[0], &v[1], &v[2]) else {
                continue;
            };
            let mut walked = HashSet::new();
            triangle_quads(&st, &t, None, |q| {
                quads += 1;
                for i in 0..4 {
                    if q.cov & (1 << i) != 0 {
                        assert!(walked.insert((q.x + (i & 1) as i32, q.y + (i >> 1) as i32)), "round {round}: a pixel twice");
                    }
                }
            });
            let (x0, y0, x1, y1) = t.bounds;
            let mut owned = HashSet::new();
            for y in y0.max(0)..=y1.min(1023) {
                for x in x0.max(0)..=x1.min(ODSY_WIDTH as i32 - 1) {
                    let e = t.edges_at(x as i64, y as i64);
                    if (0..3).all(|k| e[k] >= t.need[k]) {
                        owned.insert((x, y));
                    }
                }
            }
            assert_eq!(walked, owned, "round {round}");
        }
        assert!(quads > 1000);
    }
}
