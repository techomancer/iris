//! A draw's primitives through setup: assembly of the transform side's
//! vertices into triangles, lines and points; the line and point walks;
//! polygons drawn as outlines, as vertices or smoothed
//! (`clanker/plans/odyssey-quad-pipeline.md`).
//!
//! Aliased lines take the centre-sampled rule: one pixel per major-axis
//! column (or row) whose centre lies between the ends, the start's side
//! included, the minor-axis pixel holding the segment at that centre, a
//! crossing exactly on a pixel edge taking the lower pixel (the V10 matches
//! it pixel for pixel, `clanker/odyssey/v10-vs-emulator-first-comparison.md`).
//! Wide lines replicate along the minor axis. Smoothed points and lines
//! take the exact box filter's coverage, and smoothed polygons each pixel
//! square's overlap with the polygon: model choices, the board's coverage
//! rules being unknown.

use super::raster::RasterVertex;
use super::setup::{clip_triangle, Draw, Frag, CLIPPED_MAX, SUB};
use super::state::PolygonMode;
use super::setup::VolumeVertex;
use super::super::assembly::{
    report_w_plane_clip, vtx, CLIP_PLANES, LINES, LINE_LOOP, LINE_STRIP, POINTS, POLYGON, QUADS,
    QUAD_STRIP, TRIANGLES, TRIANGLE_FAN, TRIANGLE_STRIP, Prim,
};
use crate::dev::gl::vertex::{ClipVertex, NearW, CLIP_PLANES as VOLUME_PLANES};
use crate::devlog::LogModule;

/// What the raster words give the walks of lines, points and polygons.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PrimState {
    pub point_size: f32,
    pub line_width: f32,
    pub point_smooth: bool,
    pub line_smooth: bool,
    pub polygon_smooth: bool,
    pub stipple: bool,
    pub stipple_repeat: u32,
    pub stipple_pattern: u32,
    pub modes: [PolygonMode; 2],
    /// The polygon offset applies to polygons drawn as points, lines, fills.
    pub offset_point: bool,
    pub offset_line: bool,
    pub offset_fill: bool,
    pub offset_steps: i64,
}

impl Default for PolygonMode {
    fn default() -> Self {
        PolygonMode::Fill
    }
}

impl PrimState {
    pub fn of(w: &super::state::RasterWords) -> PrimState {
        let scale = super::super::consts::line::POINT_SIZE_SCALE;

        PrimState {
            point_size: w.point_size_x128() as f32 / scale,
            line_width: w.line_width_x128() as f32 / scale,
            point_smooth: w.point_smooth(),
            line_smooth: w.line_smooth(),
            polygon_smooth: w.polygon_smooth(),
            stipple: w.line_stipple_on(),
            stipple_repeat: w.line_stipple_repeat(),
            stipple_pattern: w.line_stipple_pattern() as u32,
            modes: [w.polygon_mode_front(), w.polygon_mode_back()],
            offset_point: w.offset_point(),
            offset_line: w.offset_line(),
            offset_fill: w.offset_fill(),
            offset_steps: w.offset_steps() as i64,
        }
    }
}

/// The line stipple's place in its pattern, carried across a strip and,
/// standalone, across draws.
#[derive(Clone, Copy, Default)]
pub(crate) struct LineStipple {
    pub counter: u32,
    pub repeat: u32,
    pub pattern: u32,
    pub enabled: bool,
    /// The counter moved.
    pub consumed: bool,
}

impl LineStipple {
    /// Whether the next pixel is drawn, and the counter past it.
    fn step(&mut self) -> bool {
        let bit = (self.counter / self.repeat) % 16;
        let keep = !self.enabled || (self.pattern & (1u32 << bit)) != 0;

        self.counter = self.counter.wrapping_add(1) % 16u32.wrapping_mul(self.repeat);
        self.consumed = true;
        keep
    }
}

/// An aliased point's or line's size in pixels, half rounding up; None for
/// one the walk cannot take.
pub(crate) fn dimension(requested: f32) -> Option<u32> {
    let n = if requested == 0.0 { 1.0 } else { (requested as f64 + 0.5).floor() };

    if !requested.is_finite() || requested < 0.0 || n > i32::MAX as f64 {
        crate::dlog!(LogModule::Odyssey, "sgi-gl: unsupported aliased dimension {}", requested);
        return None;
    }
    Some(n.max(1.0) as u32)
}

/// Whether pixel (px, py) is on the segment a to b under the centre-sampled
/// rule. Exact on the snapped grid: with the minor coordinate at the centre
/// v = n + t / den, the pixel is ceil(v / 2^sub) - 1. Coordinate differences
/// are at most 2^29, so t fits, and the minor base shifted by the width
/// stays below 2^43.
fn centre_covers(a: &RasterVertex, b: &RasterVertex, px: i64, py: i64) -> bool {
    let one = SUB;
    let half = one >> 1;
    let x_major = b.x.wrapping_sub(a.x).wrapping_abs() >= b.y.wrapping_sub(a.y).wrapping_abs();
    let (la, lb, na, nb) = if x_major { (a.x, b.x, a.y, b.y) } else { (a.y, b.y, a.x, b.x) };
    let c = (if x_major { px } else { py }).wrapping_mul(one).wrapping_add(half);
    let mut den = lb.wrapping_sub(la);

    if if den > 0 { c < la || c >= lb } else { c > la || c <= lb } {
        return false;
    }
    let mut t = nb.wrapping_sub(na).wrapping_mul(c.wrapping_sub(la));
    if den < 0 {
        t = t.wrapping_neg();
        den = den.wrapping_neg();
    }
    // v = na + floor(t / den) plus a fraction that is zero or not.
    let v = na.wrapping_add(t.div_euclid(den));
    let minor = if t % den != 0 { v.div_euclid(one) } else { v.wrapping_neg().div_euclid(one).wrapping_neg().wrapping_sub(1) };
    (if x_major { py } else { px }) == minor
}

#[derive(Clone, Copy, Default)]
struct Point2 {
    x: f64,
    y: f64,
}

/// Clip a convex polygon to one side of a line x = bound (axis 0) or
/// y = bound (axis 1).
fn clip_edge(in_: &[Point2], n: usize, out: &mut [Point2], axis: u32, bound: f64, greater: bool) -> usize {
    let mut count = 0usize;

    for i in 0..n {
        let a = in_[i];
        let b = in_[(i + 1) % n];
        let av = if axis != 0 { a.y } else { a.x };
        let bv = if axis != 0 { b.y } else { b.x };
        let ai = if greater { av >= bound } else { av <= bound };
        let bi = if greater { bv >= bound } else { bv <= bound };

        if ai != bi {
            let t = (bound - av) / (bv - av);
            let mut p = Point2 { x: a.x + t * (b.x - a.x), y: a.y + t * (b.y - a.y) };

            if axis != 0 {
                p.y = bound;
            } else {
                p.x = bound;
            }
            out[count] = p;
            count += 1;
        }
        if bi {
            out[count] = b;
            count += 1;
        }
    }
    count
}

/// The area of pixel square (x, y) a convex polygon covers.
fn square_area(polygon: &[Point2], n: usize, x: i32, y: i32) -> f64 {
    let mut a = [Point2::default(); 12];
    let mut b = [Point2::default(); 12];
    let mut area = 0.0f64;
    let mut n = n;

    a[..n].copy_from_slice(&polygon[..n]);
    n = clip_edge(&a, n, &mut b, 0, x as f64, true);
    n = clip_edge(&b, n, &mut a, 0, (x + 1) as f64, false);
    n = clip_edge(&a, n, &mut b, 1, y as f64, true);
    n = clip_edge(&b, n, &mut a, 1, (y + 1) as f64, false);
    for i in 0..n {
        let p = &a[i];
        let q = &a[(i + 1) % n];

        area += p.x * q.y - p.y * q.x;
    }
    (area.abs() * 0.5).min(1.0)
}

/// The coverage of pixel (x, y) by a smoothed line from a to b of width
/// `width`, in pixels: the rectangle about it.
fn line_coverage(ax: f64, ay: f64, bx: f64, by: f64, width: f64, x: i32, y: i32) -> f64 {
    let dx = bx - ax;
    let dy = by - ay;
    let length = dx.hypot(dy);

    if !(length > 0.0) || !(width > 0.0) {
        return 0.0;
    }
    let nx = -dy * width / (2.0 * length);
    let ny = dx * width / (2.0 * length);
    let rectangle = [
        Point2 { x: ax + nx, y: ay + ny },
        Point2 { x: bx + nx, y: by + ny },
        Point2 { x: bx - nx, y: by - ny },
        Point2 { x: ax - nx, y: ay - ny },
    ];
    square_area(&rectangle, 4, x, y)
}

fn circle_integral(x: f64, radius: f64) -> f64 {
    let y = (radius * radius - x * x).max(0.0).sqrt();

    0.5 * (x * y + radius * radius * (x / radius).asin())
}

/// The coverage of pixel (x, y) by a disc about (cx, cy).
fn circle_coverage(cx: f64, cy: f64, radius: f64, x: i32, y: i32) -> f64 {
    let mut cuts = [0.0f64; 8];
    let mut area = 0.0f64;
    let mut n = 0usize;
    let lo = (x as f64 - cx).max(-radius);
    let hi = ((x + 1) as f64 - cx).min(radius);

    if !(lo < hi) || !(radius > 0.0) {
        return 0.0;
    }
    cuts[n] = lo;
    n += 1;
    cuts[n] = hi;
    n += 1;
    for edge in 0..2 {
        let yedge = (if edge != 0 { y + 1 } else { y }) as f64 - cy;

        if yedge.abs() < radius {
            let root = (radius * radius - yedge * yedge).sqrt();

            if -root > lo && -root < hi {
                cuts[n] = -root;
                n += 1;
            }
            if root > lo && root < hi {
                cuts[n] = root;
                n += 1;
            }
        }
    }
    cuts[..n].sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    for i in 0..n.saturating_sub(1) {
        let a = cuts[i];
        let b = cuts[i + 1];
        let mid = (a + b) * 0.5;
        let circle = (radius * radius - mid * mid).max(0.0).sqrt();
        let ymin = y as f64 - cy;
        let ymax = (y + 1) as f64 - cy;
        let top = ymax.min(circle);
        let bottom = ymin.max(-circle);
        let arc = circle_integral(b, radius) - circle_integral(a, radius);

        if top <= bottom {
            continue;
        }
        area += (if top == circle { arc } else { ymax * (b - a) }) - (if bottom == -circle { -arc } else { ymin * (b - a) });
    }
    area.clamp(0.0, 1.0)
}

/// The point at t along a to b: position, depth and divisor linearly,
/// colour, texture coordinates and user plane values perspective-correct.
fn vertex_at(a: &RasterVertex, b: &RasterVertex, t: f64) -> RasterVertex {
    let wa = (1.0 - t) * a.inv_w;
    let wb = t * b.inv_w;
    let sum = wa + wb;

    RasterVertex {
        x: (a.x as f64 + t * (b.x - a.x) as f64).round() as i64,
        y: (a.y as f64 + t * (b.y - a.y) as f64).round() as i64,
        z: a.z + t * (b.z - a.z),
        inv_w: (1.0 - t) * a.inv_w + t * b.inv_w,
        depth_w: (1.0 - t) * a.depth_w + t * b.depth_w,
        rgba: std::array::from_fn(|k| (wa * a.rgba[k] + wb * b.rgba[k]) / sum),
        spec: std::array::from_fn(|k| (wa * a.spec[k] + wb * b.spec[k]) / sum),
        tex: std::array::from_fn(|k| (wa * a.tex[k] + wb * b.tex[k]) / sum),
        clip: std::array::from_fn(|k| (wa * a.clip[k] + wb * b.clip[k]) / sum),
    }
}

impl Draw<'_> {
    /// The primitives of a draw, in order. The standalone line stipple's
    /// counter carries in `line`.
    pub fn primitives(&self, d: &Prim, line: &mut LineStipple, use_stipple: bool) {
        let n = d.count as usize;
        let v = |i: usize| -> &[f32] { vertex(d, None, i) };
        let ps = &self.pipe.prims;
        let filled = d.mode >= TRIANGLES && d.mode <= POLYGON;

        if filled {
            self.polygon_modes(d);
            return;
        }
        match d.mode {
            POINTS => {
                let Some(size) = dimension(ps.point_size) else {
                    return;
                };
                for i in 0..n {
                    self.point(v(i), size);
                }
            }
            LINES => {
                let Some(width) = dimension(ps.line_width) else {
                    return;
                };
                let mut i = 0usize;
                while i + 1 < n {
                    if d.line_phase.is_none() {
                        line.counter = 0;
                    }
                    self.line(v(i), v(i + 1), v(i + 1), width, use_stipple.then_some(&mut *line));
                    i += 2;
                }
            }
            LINE_STRIP | LINE_LOOP => {
                let Some(width) = dimension(ps.line_width) else {
                    return;
                };
                let mut i = 0usize;
                while i + 1 < n {
                    self.line(v(i), v(i + 1), v(i + 1), width, use_stipple.then_some(&mut *line));
                    i += 1;
                }
                if d.mode == LINE_LOOP && n >= 2 {
                    self.line(v(n - 1), v(0), v(0), width, use_stipple.then_some(&mut *line));
                }
            }
            TRIANGLES | TRIANGLE_STRIP | TRIANGLE_FAN | POLYGON | QUADS | QUAD_STRIP => self.triangles(d),
            _ => crate::dlog!(LogModule::Odyssey, "sgi-gl: draw of mode {} ignored", d.mode),
        }
    }

    /// Clip a triangle of draw vertices.
    fn clipped(&self, va: &[f32], vb: &[f32], vc: &[f32]) -> ([VolumeVertex; CLIPPED_MAX], usize) {
        let in_ = [va, vb, vc].map(|v| self.xf.vertex(v));

        clip_triangle(&self.xf.clip(false), &in_)
    }

    /// What clipping left of a triangle, n vertices of poly, as a fan from
    /// its first vertex, each flat shaded from provoking.
    fn fan(&self, poly: &[VolumeVertex; CLIPPED_MAX], n: usize, provoking: &[f32]) {
        let mut i = 1usize;
        while i + 1 < n {
            self.clipped_triangle(&poly[0], &poly[i], &poly[i + 1], provoking);
            i += 1;
        }
    }

    fn triangle(&self, va: &[f32], vb: &[f32], vc: &[f32], provoking: &[f32]) {
        if self.bounds.is_none() {
            return;
        }
        let (poly, n) = self.clipped(va, vb, vc);
        self.fan(&poly, n, provoking);
    }


    /// The filled primitives of a draw, from TRIANGLES on, in order.
    fn triangles(&self, d: &Prim) {
        let n = d.count as usize;
        let v = |i: usize| -> &[f32] { vertex(d, None, i) };

        match d.mode {
            TRIANGLES => {
                let mut i = 0usize;
                while i + 2 < n {
                    self.triangle(v(i), v(i + 1), v(i + 2), v(i + 2));
                    i += 3;
                }
            }
            TRIANGLE_STRIP => {
                let mut i = 0usize;
                while i + 2 < n {
                    if i & 1 != 0 {
                        self.triangle(v(i + 1), v(i), v(i + 2), v(i + 2));
                    } else {
                        self.triangle(v(i), v(i + 1), v(i + 2), v(i + 2));
                    }
                    i += 1;
                }
            }
            TRIANGLE_FAN | POLYGON => {
                let mut i = 0usize;
                while i + 2 < n {
                    self.triangle(v(0), v(i + 1), v(i + 2), if d.mode == POLYGON { v(0) } else { v(i + 2) });
                    i += 1;
                }
            }
            QUADS => {
                let mut i = 0usize;
                while i + 3 < n {
                    self.triangle(v(i), v(i + 1), v(i + 2), v(i + 3));
                    self.triangle(v(i), v(i + 2), v(i + 3), v(i + 3));
                    i += 4;
                }
            }
            QUAD_STRIP => {
                // Vertices 2i - 1, 2i, 2i + 2 and 2i + 1 in order (2.6.1).
                let mut i = 0usize;
                while i + 3 < n {
                    self.triangle(v(i), v(i + 1), v(i + 3), v(i + 3));
                    self.triangle(v(i), v(i + 3), v(i + 2), v(i + 3));
                    i += 2;
                }
            }
            _ => unreachable!(),
        }
    }

    /// One line, clipped and projected; the user planes are tested per
    /// fragment.
    fn line(&self, va: &[f32], vb: &[f32], provoking: &[f32], width: u32, stipple: Option<&mut LineStipple>) {
        if stipple.is_none() && self.bounds.is_none() {
            return;
        }
        let ca = self.xf.vertex(va);
        let cb = self.xf.vertex(vb);
        let clip = self.xf.clip(false);
        let any = clip.outcode_by::<NearW, _>(&ca) | clip.outcode_by::<NearW, _>(&cb);
        if any & NearW::W_PLANE != 0 {
            report_w_plane_clip();
        }
        let Some((ca, cb)) = clip.clip_line_by::<NearW, _>(ca, cb, any) else {
            return;
        };
        if let (Some(a), Some(b)) = (self.xf.project(&ca), self.xf.project(&cb)) {
            self.line_projected(&a, &b, provoking, 0, width, stipple, true);
        }
    }

    /// One aliased segment, already projected, with the provoking vertex's
    /// colour when flat shaded: width as a minor-axis shift and replication.
    /// A smoothed segment takes the box filter's coverage, stippled as
    /// one-pixel pieces along its length.
    #[allow(clippy::too_many_arguments, reason = "a segment and how it is drawn")]
    fn line_projected(&self, a: &RasterVertex, b: &RasterVertex, provoking: &[f32], steps: i64, width: u32, stipple: Option<&mut LineStipple>, user_clip: bool) {
        let mut stipple = stipple;
        let smooth = !self.pipe.raster.flat;
        let line_smooth = self.pipe.prims.line_smooth;
        let smooth_width = self.pipe.prims.line_width as f64;
        let one = SUB as f64;
        let (a, b) = (*a, *b);
        // A stippled segment outside the viewport still counts, against the
        // window's own rectangle.
        let visible = self.bounds.is_some();
        let (bx0, by0, bx1, by1) = self.bounds.unwrap_or(self.owned);

        if (!visible && stipple.is_none()) || (a.x == b.x && a.y == b.y) {
            return;
        }
        if line_smooth && stipple.as_ref().is_some_and(|s| !s.enabled) {
            let length = ((b.x - a.x) as f64 / one).hypot((b.y - a.y) as f64 / one);
            let s = stipple.as_mut().expect("a stipple");
            let period = 16u32.wrapping_mul(s.repeat);

            // Unstippled, the segment is one rectangle: its fragments are
            // not split between one-pixel pieces, each blended on its own.
            // The count still moves on by the pieces a stipple would take.
            s.counter = s.counter.wrapping_add((length.ceil() % period as f64) as u32) % period;
            s.consumed = true;
            if !visible {
                return;
            }
            stipple = None;
        }
        if let Some(s) = stipple.as_mut().filter(|_| line_smooth) {
            let length = ((b.x - a.x) as f64 / one).hypot((b.y - a.y) as f64 / one);
            let segments = length.ceil();
            let mut i = 0.0f64;

            while i < segments {
                if s.step() {
                    let start = vertex_at(&a, &b, i / length);
                    let end = vertex_at(&a, &b, ((i + 1.0) / length).min(1.0));
                    self.line_projected(&start, &end, provoking, steps, width, None, user_clip);
                }
                i += 1.0;
            }
            return;
        }
        let flat = [0, 1, 2, 3].map(|k| provoking[vtx::COLOR + k] as f64);
        let len2 = (b.x - a.x) as f64 * (b.x - a.x) as f64 + (b.y - a.y) as f64 * (b.y - a.y) as f64;
        let x_major = (b.x - a.x).wrapping_abs() >= (b.y - a.y).wrapping_abs();
        let mut sa = a;
        let mut sb = b;
        let shift = (width.wrapping_sub(1) as i64) * (SUB / 2);
        if x_major {
            sa.y -= shift;
            sb.y -= shift;
        } else {
            sa.x -= shift;
            sb.x -= shift;
        }
        // Walk the major axis a pixel at a time, and in each step try the
        // minor-axis pixels within one of those the segment spans there.
        let major = |v: &RasterVertex| if x_major { v.x } else { v.y };
        let minor = |v: &RasterVertex| if x_major { v.y } else { v.x };
        let mut major0 = (major(&a).min(major(&b)) >> SUBPIXEL_SHIFT) as i32 - 1;
        let mut major1 = (major(&a).max(major(&b)) >> SUBPIXEL_SHIFT) as i32 + 1;
        let pad = smooth_width * 0.5;
        if line_smooth {
            let (l0, l1) = (major(&a) as f64, major(&b) as f64);

            major0 = (l0.min(l1) / one - pad).floor() as i32 - 1;
            major1 = (l0.max(l1) / one + pad).ceil() as i32 + 1;
        }
        let mut step = 1i32;
        if stipple.is_some() {
            // Count coverage before ownership and the scissor, in the
            // submitted direction.
            if major(&b) < major(&a) {
                std::mem::swap(&mut major0, &mut major1);
                step = -1;
            }
        } else {
            major0 = major0.max(if x_major { bx0 } else { by0 });
            major1 = major1.min(if x_major { bx1 } else { by1 });
        }
        let (lo, hi) = if major(&a) <= major(&b) { (&sa, &sb) } else { (&sb, &sa) };
        let (l0, l1) = (major(lo) as f64, major(hi) as f64);
        let (n0, n1) = (minor(lo) as f64, minor(hi) as f64);
        let mut mm = major0;
        loop {
            if !(if step > 0 { mm <= major1 } else { mm >= major1 }) {
                break;
            }
            let m = mm;
            mm += step;
            let mut e0 = (m as f64 * one).max(l0);
            let mut e1 = ((m + 1) as f64 * one).min(l1);
            let (f0, f1) = if l1 > l0 {
                e0 = e0.min(l1);
                e1 = e1.max(l0);
                (n0 + (n1 - n0) * (e0 - l0) / (l1 - l0), n0 + (n1 - n0) * (e1 - l0) / (l1 - l0))
            } else {
                (n0, n1)
            };
            let mut minor0 = (f0.min(f1) / one).floor() as i64 - 1;
            let mut minor1 = (f0.max(f1) / one).floor() as i64 + 1;
            if line_smooth {
                let (q0, q1) = (minor(&a) as f64, minor(&b) as f64);

                minor0 = (q0.min(q1) / one - pad).floor() as i64 - 1;
                minor1 = (q0.max(q1) / one + pad).ceil() as i64 + 1;
            }
            for n in minor0..=minor1 {
                let (base_x, base_y) = if x_major { (m as i64, n) } else { (n, m as i64) };
                let replica0 = n.max((if x_major { by0 } else { bx0 }) as i64);
                let replica1 = if line_smooth { replica0 } else { (n + width as i64 - 1).min((if x_major { by1 } else { bx1 }) as i64) };
                let coverage = if line_smooth {
                    line_coverage(a.x as f64 / one, a.y as f64 / one, b.x as f64 / one, b.y as f64 / one, smooth_width, base_x as i32, base_y as i32)
                } else if centre_covers(&sa, &sb, base_x, base_y) {
                    1.0
                } else {
                    0.0
                };

                if !(coverage > 0.0) {
                    continue;
                }
                let keep = stipple.as_mut().is_none_or(|s| s.step());
                if !keep || !visible || replica0 > replica1 || m < (if x_major { bx0 } else { by0 }) || m > (if x_major { bx1 } else { by1 }) {
                    continue;
                }
                // Visit only the replicas the window can show.
                for rep in replica0..=replica1 {
                    let (px, py) = if x_major { (m, rep as i32) } else { (rep as i32, m) };
                    let cx = (px as f64 + 0.5) * one;
                    let cy = (py as f64 + 0.5) * one;
                    // Approximation: the ends' values, not extrapolated.
                    let t = (((cx - a.x as f64) * (b.x - a.x) as f64 + (cy - a.y as f64) * (b.y - a.y) as f64) / len2).clamp(0.0, 1.0);
                    let wa = (1.0 - t) * a.inv_w;
                    let wb = t * b.inv_w;
                    let mut frag = Frag::default();

                    if user_clip && self.xf.user_planes() != 0 {
                        let d: [f64; CLIP_PLANES as usize] = std::array::from_fn(|i| (wa * a.clip[i] + wb * b.clip[i]) / (wa + wb));

                        if self.xf.user_clipped(&d) {
                            continue;
                        }
                    }
                    for k in 0..4 {
                        frag.tex[k] = (wa * a.tex[k] + wb * b.tex[k]) / (wa + wb);
                        frag.rgba[k] = if smooth { (wa * a.rgba[k] + wb * b.rgba[k]) / (wa + wb) } else { flat[k] };
                    }
                    if self.textured {
                        // The helper one pixel on along the major axis.
                        let t2 = t + one / (major(&b) - major(&a)).abs().max(1) as f64;
                        let (wa2, wb2) = ((1.0 - t2) * a.inv_w, t2 * b.inv_w);
                        let step: [f64; 4] = std::array::from_fn(|k| (wa2 * a.tex[k] + wb2 * b.tex[k]) / (wa2 + wb2) - frag.tex[k]);
                        if x_major { frag.dx = step } else { frag.dy = step }
                    }
                    self.fragment(px, py, a.z + t * (b.z - a.z), (1.0 - t) * a.depth_w + t * b.depth_w, steps, &frag, coverage);
                }
            }
        }
    }

    /// One point, rejected whole by the view volume and the user planes.
    fn point(&self, vp: &[f32], size: u32) {
        let cv = self.xf.vertex(vp);
        let oc = self.xf.clip(false).outcode_by::<NearW, _>(&cv);

        if oc != 0 {
            if oc & NearW::W_PLANE != 0 {
                report_w_plane_clip();
            }
            return;
        }
        if let Some(v) = self.xf.project(&cv).filter(|v| !self.xf.user_clipped(&v.clip)) {
            self.point_projected(&v, None, 0, size);
        }
    }

    /// A point, already projected: an aliased square centred on the
    /// snapped grid, or a smoothed disc's box-filter coverage about its
    /// snapped centre. MODEL CHOICE: the board's coverage rule for a
    /// smoothed point is UNKNOWN; the graded rule (INFERRED) is a table in
    /// the datasheet, from which the disc area differs (about 0.15 at 0.95
    /// pixel against the vendor frames): an approximation, to be replaced.
    fn point_projected(&self, v: &RasterVertex, provoking: Option<&[f32]>, steps: i64, size: u32) {
        let Some((bx0, by0, bx1, by1)) = self.bounds else {
            return;
        };
        let psize = size as i64;
        // Even sizes centre on a pixel corner; odd ones on a centre.
        let even = if size % 2 != 0 { 0 } else { SUB / 2 };
        let px = (v.x + even) >> SUBPIXEL_SHIFT;
        let py = (v.y + even) >> SUBPIXEL_SHIFT;
        let mut frag = Frag { rgba: v.rgba, tex: v.tex, ..Default::default() };

        if let Some(p) = provoking.filter(|_| self.pipe.raster.flat) {
            frag.rgba = [0, 1, 2, 3].map(|k| p[vtx::COLOR + k] as f64);
        }
        if self.pipe.prims.point_smooth {
            let smooth_size = self.pipe.prims.point_size as f64;
            let (cx, cy) = (v.x as f64 / SUB as f64, v.y as f64 / SUB as f64);
            let radius = smooth_size * 0.5;
            let x0 = ((cx - radius).floor() as i64).max(bx0 as i64);
            let y0 = ((cy - radius).floor() as i64).max(by0 as i64);
            let x1 = ((cx + radius).ceil() as i64 - 1).min(bx1 as i64);
            let y1 = ((cy + radius).ceil() as i64 - 1).min(by1 as i64);

            for y in y0..=y1 {
                for x in x0..=x1 {
                    let coverage = circle_coverage(cx, cy, radius, x as i32, y as i32);

                    self.fragment(x as i32, y as i32, v.z, v.depth_w, steps, &frag, coverage);
                }
            }
            return;
        }
        let x0 = (px - psize / 2).max(bx0 as i64);
        let y0 = (py - psize / 2).max(by0 as i64);
        let x1 = (px - psize / 2 + psize - 1).min(bx1 as i64);
        let y1 = (py - psize / 2 + psize - 1).min(by1 as i64);
        for y in y0..=y1 {
            for x in x0..=x1 {
                self.fragment(x as i32, y as i32, v.z, v.depth_w, steps, &frag, 1.0);
            }
        }
    }

    /// Each independently assembled polygon of a draw drawn under the
    /// polygon modes or smoothed, with its provoking vertex.
    fn polygon_modes(&self, d: &Prim) {
        let mut indices = [0u32; 4];
        let n = d.count;

        if d.mode == POLYGON {
            self.polygon(d, None, n as usize, vertex(d, None, 0), false);
            return;
        }
        let stride: u32 = match d.mode {
            TRIANGLES => 3,
            QUADS => 4,
            QUAD_STRIP => 2,
            _ => 1,
        };
        let count: u32 = if d.mode == QUADS || d.mode == QUAD_STRIP { 4 } else { 3 };
        let mut i = 0u32;
        while i + count <= n {
            for k in 0..count {
                indices[k as usize] = i + k;
            }
            if d.mode == TRIANGLE_STRIP && (i & 1) != 0 {
                indices[0] = i + 1;
                indices[1] = i;
            } else if d.mode == TRIANGLE_FAN {
                indices[0] = 0;
            } else if d.mode == QUAD_STRIP {
                indices[2] = i + 3;
                indices[3] = i + 2;
            }
            self.polygon(d, Some(&indices[..count as usize]), count as usize, vertex(d, None, (i + count - 1) as usize), stride < 3);
            i += stride;
        }
    }

    /// One polygon of `count` vertices under the polygon modes: its facing
    /// from its clipped outline, then filled (or smoothed), or its edges
    /// drawn as lines or its vertices as points, where the vertices' edge
    /// flags (all of them with `force_edges`) say.
    fn polygon(&self, d: &Prim, indices: Option<&[u32]>, count: usize, provoking: &[f32], force_edges: bool) {
        let p = self.pipe;
        let ps = &p.prims;

        if count < 3 || count > usize::MAX - VOLUME_PLANES {
            return;
        }
        let capacity = count + VOLUME_PLANES;
        let mut poly: Vec<VolumeVertex> = vec![VolumeVertex::default(); capacity];
        let mut other: Vec<VolumeVertex> = vec![VolumeVertex::default(); capacity];
        for (i, v) in poly.iter_mut().enumerate().take(count) {
            *v = self.xf.vertex(vertex(d, indices, i));
            if !v.clip.iter().chain(&v.eye).all(|c| c.is_finite()) {
                crate::dlog!(LogModule::Odyssey, "sgi-gl: invalid polygon position dropped");
                return;
            }
            if force_edges {
                v.set_edge(true);
            }
        }
        // Edges clipping adds are boundary edges; kept ones keep their flags.
        let clip = self.xf.clip(true);
        let any = poly[..count].iter().fold(0, |o, v| o | clip.outcode_by::<NearW, _>(v));
        if any & NearW::W_PLANE != 0 {
            report_w_plane_clip();
        }
        let (n, over) = clip.clip_polygon_by::<NearW, _>(&mut poly, &mut other, count, any);
        if over {
            crate::dlog!(LogModule::Odyssey, "sgi-gl: polygon exceeds convex clipping bound");
            return;
        }
        if n == 0 {
            return;
        }
        let mut v: Vec<RasterVertex> = Vec::with_capacity(n);
        for cv in &poly[..n] {
            let Some(r) = self.xf.project(cv) else {
                return;
            };
            v.push(r);
        }
        let mut area = 0.0f64;
        for i in 2..n {
            area += v[0].edge(&v[i - 1], v[i].x, v[i].y) as f64;
        }
        if p.raster.front_cw {
            area = -area;
        }
        // Zero stays back-facing through both reversals.
        let front = area > 0.0;
        let culled = match p.raster.cull {
            super::state::Cull::Both => true,
            super::state::Cull::Back => !front,
            super::state::Cull::Front => front,
            super::state::Cull::None => false,
        };
        if culled {
            return;
        }
        let mode = ps.modes[if front { 0 } else { 1 }];
        let back = !front && self.xf.back_colours;
        // The fragment paths below take the colour sum before shading: only
        // filled triangles carry the separate specular to the shader.
        let (colour_at, spec_at) = if back { (vtx::BACK_COLOR, vtx::BACK_SPECULAR) } else { (vtx::COLOR, vtx::SPECULAR) };
        let summed = |attr: &[f32]| -> [f64; 4] {
            [0, 1, 2, 3].map(|k| {
                let s = if k < 3 { attr[spec_at + k] } else { 0.0 };
                // max then min, as project's: a NaN colour becomes 0.
                #[allow(clippy::manual_clamp, reason = "clamp would keep a NaN")]
                let c = (attr[colour_at + k] + s).max(0.0).min(1.0);
                c as f64
            })
        };
        match mode {
            PolygonMode::Fill if ps.polygon_smooth => {
                for (r, cv) in v.iter_mut().zip(&poly[..n]) {
                    r.rgba = summed(&cv.attr);
                }
                let steps = if ps.offset_fill { ps.offset_steps } else { 0 };
                self.polygon_coverage(&v, summed(provoking), steps);
            }
            PolygonMode::Fill => {
                // The fan again, from the draw's own vertices, as filled
                // triangles are.
                let v0 = vertex(d, indices, 0);
                for i in 1..count - 1 {
                    self.triangle(v0, vertex(d, indices, i), vertex(d, indices, i + 1), provoking);
                }
            }
            PolygonMode::Line | PolygonMode::Point => {
                let line = mode == PolygonMode::Line;
                let Some(size) = dimension(if line { ps.line_width } else { ps.point_size }) else {
                    return;
                };
                let mut flat = [0.0f32; vtx::FLOATS];
                flat.copy_from_slice(&provoking[..vtx::FLOATS]);
                let sum = summed(provoking);
                for k in 0..4 {
                    flat[vtx::COLOR + k] = sum[k] as f32;
                }
                for (r, cv) in v.iter_mut().zip(&poly[..n]) {
                    r.rgba = summed(&cv.attr);
                }
                let steps = if (line && ps.offset_line) || (!line && ps.offset_point) { ps.offset_steps } else { 0 };
                let mut stipple = LineStipple { enabled: ps.stipple, repeat: ps.stipple_repeat.clamp(1, 256), pattern: ps.stipple_pattern, ..Default::default() };
                for i in 0..n {
                    if !poly[i].edge() {
                        continue;
                    }
                    if line {
                        self.line_projected(&v[i], &v[(i + 1) % n], &flat, steps, size, Some(&mut stipple), false);
                    } else {
                        self.point_projected(&v[i], Some(&flat), steps, size);
                    }
                }
            }
            PolygonMode::Unknown => crate::dlog!(LogModule::Odyssey, "sgi-gl: polygon mode 3 ignored"),
        }
    }

    /// A clipped convex polygon smoothed: each pixel square's overlap with
    /// it is the fragment's coverage, its values taken at the overlap's
    /// centroid (exact for affine values, an approximation for
    /// perspective-correct ones).
    fn polygon_coverage(&self, v: &[RasterVertex], flat: [f64; 4], steps: i64) {
        let n = v.len();
        let scale = SUB as f64;
        let Some((mut px0, mut py0, mut px1, mut py1)) = self.bounds.filter(|_| n >= 3) else {
            return;
        };
        let mut a = vec![Point2::default(); n + 8];
        let mut b = vec![Point2::default(); n + 8];
        let mut polygon_area = 0.0f64;
        let (mut min_x, mut max_x, mut min_y, mut max_y) = (v[0].x, v[0].x, v[0].y, v[0].y);

        for i in 0..n {
            let next = &v[(i + 1) % n];

            min_x = min_x.min(v[i].x);
            max_x = max_x.max(v[i].x);
            min_y = min_y.min(v[i].y);
            max_y = max_y.max(v[i].y);
            polygon_area += v[i].x.wrapping_mul(next.y).wrapping_sub(next.x.wrapping_mul(v[i].y)) as f64;
        }
        if !(polygon_area.abs() > 0.0) {
            return;
        }
        px0 = px0.max((min_x >> SUBPIXEL_SHIFT) as i32);
        px1 = px1.min((max_x >> SUBPIXEL_SHIFT) as i32);
        py0 = py0.max((min_y >> SUBPIXEL_SHIFT) as i32);
        py1 = py1.min((max_y >> SUBPIXEL_SHIFT) as i32);
        let ccw = polygon_area > 0.0;
        let edge = |p: &RasterVertex, q: &RasterVertex, x: f64, y: f64| {
            (q.x.wrapping_sub(p.x)) as f64 * (y - p.y as f64) - (q.y.wrapping_sub(p.y)) as f64 * (x - p.x as f64)
        };

        for py in py0..=py1 {
            // Pixels beyond the polygon's extent in this row have no overlap.
            let Some((lo, hi)) = row_span(v, scale, py) else {
                continue;
            };
            for px in px0.max(lo)..=px1.min(hi) {
                let Some(background) = self.stipple_pass(px, py) else {
                    continue;
                };
                // Whether the polygon holds all four corners of the square.
                let (x0, y0) = (px as f64 * scale, py as f64 * scale);
                let inside = (0..n).all(|i| {
                    let (p, q) = (&v[i], &v[(i + 1) % n]);
                    [(x0, y0), (x0 + scale, y0), (x0, y0 + scale), (x0 + scale, y0 + scale)]
                        .iter()
                        .all(|&(x, y)| if ccw { edge(p, q, x, y) >= 0.0 } else { edge(p, q, x, y) <= 0.0 })
                });
                let (coverage, x, y) = if inside {
                    (1.0, px as f64 + 0.5, py as f64 + 0.5)
                } else {
                    for i in 0..n {
                        a[i] = Point2 { x: v[i].x as f64 / scale, y: v[i].y as f64 / scale };
                    }
                    let mut count = clip_edge(&a, n, &mut b, 0, px as f64, true);
                    count = clip_edge(&b, count, &mut a, 0, (px + 1) as f64, false);
                    count = clip_edge(&a, count, &mut b, 1, py as f64, true);
                    count = clip_edge(&b, count, &mut a, 1, (py + 1) as f64, false);
                    if count < 3 {
                        continue;
                    }
                    let (mut twice, mut cx, mut cy) = (0.0f64, 0.0f64, 0.0f64);
                    for i in 0..count {
                        let (p, q) = (&a[i], &a[(i + 1) % count]);
                        let (pxp, pyp, qxp, qyp) = (p.x - px as f64, p.y - py as f64, q.x - px as f64, q.y - py as f64);
                        let cross = pxp * qyp - qxp * pyp;

                        twice += cross;
                        cx += (pxp + qxp) * cross;
                        cy += (pyp + qyp) * cross;
                    }
                    if !(twice.abs() > 0.0) {
                        continue;
                    }
                    ((twice.abs() * 0.5).min(1.0), px as f64 + cx / (3.0 * twice), py as f64 + cy / (3.0 * twice))
                };
                // The fan triangle holding the overlap's centroid.
                let found = (1..n - 1).find_map(|i| {
                    let den = edge(&v[0], &v[i], v[i + 1].x as f64, v[i + 1].y as f64);
                    if den == 0.0 {
                        return None;
                    }
                    let b0 = edge(&v[i], &v[i + 1], x * scale, y * scale) / den;
                    let b1 = edge(&v[i + 1], &v[0], x * scale, y * scale) / den;
                    let b2 = 1.0 - b0 - b1;
                    (b0 >= -1e-9 && b1 >= -1e-9 && b2 >= -1e-9).then_some((i, [b0, b1, b2]))
                });
                let Some((i, bary)) = found else {
                    continue;
                };
                let fan = [v[0], v[i], v[i + 1]];
                let w = [bary[0] * fan[0].inv_w, bary[1] * fan[1].inv_w, bary[2] * fan[2].inv_w];
                let sum = w[0] + w[1] + w[2];
                if !(sum > 0.0) || !sum.is_finite() {
                    continue;
                }
                let depth_w = bary[0] * fan[0].depth_w + bary[1] * fan[1].depth_w + bary[2] * fan[2].depth_w;
                let z = bary[0] * fan[0].z + bary[1] * fan[1].z + bary[2] * fan[2].z;
                let mut frag = Frag { rgba: flat, ..Default::default() };
                for k in 0..4 {
                    frag.tex[k] = (w[0] * fan[0].tex[k] + w[1] * fan[1].tex[k] + w[2] * fan[2].tex[k]) / sum;
                    if !self.pipe.raster.flat {
                        frag.rgba[k] = (w[0] * fan[0].rgba[k] + w[1] * fan[1].rgba[k] + w[2] * fan[2].rgba[k]) / sum;
                    }
                }
                if self.xf.user_planes() != 0 {
                    let d: [f64; CLIP_PLANES as usize] = std::array::from_fn(|k| (w[0] * fan[0].clip[k] + w[1] * fan[1].clip[k] + w[2] * fan[2].clip[k]) / sum);

                    if self.xf.user_clipped(&d) {
                        continue;
                    }
                }
                if self.textured {
                    // The helpers a pixel right and down, on the same fan
                    // triangle.
                    let den = edge(&v[0], &v[i], v[i + 1].x as f64, v[i + 1].y as f64);
                    let tex_at = |x: f64, y: f64| -> [f64; 4] {
                        let b0 = edge(&v[i], &v[i + 1], x * scale, y * scale) / den;
                        let b1 = edge(&v[i + 1], &v[0], x * scale, y * scale) / den;
                        let w = [b0 * fan[0].inv_w, b1 * fan[1].inv_w, (1.0 - b0 - b1) * fan[2].inv_w];
                        let sum = w[0] + w[1] + w[2];
                        std::array::from_fn(|k| (w[0] * fan[0].tex[k] + w[1] * fan[1].tex[k] + w[2] * fan[2].tex[k]) / sum)
                    };
                    let (right, down) = (tex_at(x + 1.0, y), tex_at(x, y + 1.0));
                    frag.dx = std::array::from_fn(|k| right[k] - frag.tex[k]);
                    frag.dy = std::array::from_fn(|k| down[k] - frag.tex[k]);
                }
                if background {
                    frag.rgba = self.background.expect("an opaque stipple");
                }
                self.fragment(px, py, z, depth_w, steps, &frag, coverage);
            }
        }
    }
}

const SUBPIXEL_SHIFT: u32 = super::raster::SUBPIXEL_BITS;

/// The pixels of row py that the convex polygon v can overlap, a pixel
/// wider on each side: from the polygon's extent along the row's strip, py
/// to py + 1. None where the strip holds none of the polygon.
fn row_span(v: &[RasterVertex], scale: f64, py: i32) -> Option<(i32, i32)> {
    let n = v.len();
    let (y0, y1) = (py as f64, py as f64 + 1.0);
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;

    for i in 0..n {
        let (ax, ay) = (v[i].x as f64 / scale, v[i].y as f64 / scale);
        let (bx, by) = (v[(i + 1) % n].x as f64 / scale, v[(i + 1) % n].y as f64 / scale);

        if ay >= y0 && ay <= y1 {
            lo = lo.min(ax);
            hi = hi.max(ax);
        }
        // Where the edge crosses either side of the strip.
        for y in [y0, y1] {
            if (ay < y) != (by < y) {
                let x = ax + (y - ay) * (bx - ax) / (by - ay);

                lo = lo.min(x);
                hi = hi.max(x);
            }
        }
    }
    if !(lo <= hi) {
        return None;
    }
    Some(((lo.floor() as i32).saturating_sub(1), (hi.floor() as i32).saturating_add(1)))
}

/// Vertex i of a draw, through `indices` where given.
fn vertex<'a>(d: &Prim<'a>, indices: Option<&[u32]>, i: usize) -> &'a [f32] {
    let index = indices.map_or(i, |ix| ix[i] as usize);

    &d.vertices[index * vtx::FLOATS..][..vtx::FLOATS]
}
