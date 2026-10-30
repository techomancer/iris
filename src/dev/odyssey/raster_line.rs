//! Lines and points: projection, the diamond-exit rule for aliased lines,
//! antialiased line and point coverage, wide lines and line stipple.
//!
//! Ported from hw/display/sgi/rasterizer.c in jamesbraid/qemu-sgi-origami.

use super::*;

impl SoftVertex {
    pub fn texture_line_gradient(&self, b: &SoftVertex, sub: u32, volume: bool, one_dimensional: bool, values: &mut [f64; 4], delta: &mut [f64; 4], length: &mut f64) -> bool {
        let mut end = [0.0f64; 4];

        *length = ((b.x - self.x) as f64).hypot((b.y - self.y) as f64) / (1i32 << sub) as f64;
        if !length.is_finite() {
            return false;
        }
        self.texture_values(volume, one_dimensional, values);
        b.texture_values(volume, one_dimensional, &mut end);
        for k in 0..4 {
            delta[k] = end[k] - values[k];
            if !values[k].is_finite() || !delta[k].is_finite() {
                return false;
            }
        }
        true
    }
}

// MIN and MAX as conditionals (a NaN operand picks the
// second), unlike f64::min and f64::max.
pub fn r2_max<T: PartialOrd>(a: T, b: T) -> T {
    if a > b { a } else { b }
}

pub fn r2_min<T: PartialOrd>(a: T, b: T) -> T {
    if a < b { a } else { b }
}

/// Sign of v + e1 * epsilon + e2 * epsilon squared as epsilon -> 0+.
pub fn soft_limit_sign(v: i64, e1: i64, e2: i64) -> i32 {
    let first = if v != 0 { v } else if e1 != 0 { e1 } else { e2 };

    (first > 0) as i32 - (first < 0) as i32
}

/// Whether |v + e1*epsilon + e2*epsilon squared| is strictly below bound.
pub fn soft_limit_abs(v: i64, e1: i64, e2: i64, bound: i64) -> bool {
    let sign = soft_limit_sign(v, e1, e2) as i64;

    soft_limit_sign(
        sign.wrapping_mul(v).wrapping_sub(bound),
        sign.wrapping_mul(e1),
        sign.wrapping_mul(e2),
    ) < 0
}

/// The endpoint minus (epsilon, epsilon squared) lies in the open diamond.
pub fn soft_diamond_contains(x: i64, y: i64, half: i64) -> bool {
    let sx = soft_limit_sign(x, -1, 0) as i64;
    let sy = soft_limit_sign(y, 0, -1) as i64;

    soft_limit_sign(
        sx.wrapping_mul(x).wrapping_add(sy.wrapping_mul(y)).wrapping_sub(half),
        -sx,
        -sy,
    ) < 0
}

impl SoftVertex {
    /// A segment produces a fragment if it enters the open diamond and its
    /// final endpoint is outside (OpenGL 1.2.1, 3.4.1, with the limiting
    /// perturbation done in lexicographic coefficients on the snapped grid).
    pub fn line_covers(&self, b: &SoftVertex, px: i64, py: i64, sub: i32) -> bool {
        let half: i64 = 1i64 << (sub - 1);
        let cx = px.wrapping_mul(1i64 << sub).wrapping_add(half);
        let cy = py.wrapping_mul(1i64 << sub).wrapping_add(half);
        let dx = b.x.wrapping_sub(self.x);
        let dy = b.y.wrapping_sub(self.y);
        let ex = cx.wrapping_sub(self.x);
        let ey = cy.wrapping_sub(self.y);
        let sdx: i32 = if dx > 0 { 1 } else { -1 };
        let sdy: i32 = if dy > 0 { 1 } else { -1 };

        if soft_diamond_contains(b.x.wrapping_sub(cx), b.y.wrapping_sub(cy), half) {
            return false;
        }
        if soft_diamond_contains(ex.wrapping_neg(), ey.wrapping_neg(), half) {
            return true;
        }
        // The perturbed column crossing has parameter (ex + epsilon) / dx.
        if dx != 0
            && soft_limit_sign(ex, 1, 0) * sdx >= 0
            && soft_limit_sign(dx.wrapping_sub(ex), -1, 0) * sdx >= 0
            && soft_limit_abs(
                dy.wrapping_mul(ex).wrapping_sub(ey.wrapping_mul(dx)),
                dy,
                dx.wrapping_neg(),
                half.wrapping_mul(dx.wrapping_abs()),
            )
        {
            return true;
        }
        dy != 0
            && soft_limit_sign(ey, 0, 1) * sdy >= 0
            && soft_limit_sign(dy.wrapping_sub(ey), 0, -1) * sdy >= 0
            && soft_limit_abs(
                dx.wrapping_mul(ey).wrapping_sub(ex.wrapping_mul(dy)),
                dy.wrapping_neg(),
                dx,
                half.wrapping_mul(dy.wrapping_abs()),
            )
    }
}

#[derive(Clone, Copy, Default)]
pub struct SoftPoint2 {
    pub x: f64,
    pub y: f64,
}

pub fn soft_clip_edge(
    in_: &[SoftPoint2],
    n: u32,
    out: &mut [SoftPoint2],
    axis: u32,
    bound: f64,
    greater: bool,
) -> u32 {
    let mut count = 0usize;

    for i in 0..n as usize {
        let a = in_[i];
        let b = in_[(i + 1) % n as usize];
        let av = if axis != 0 { a.y } else { a.x };
        let bv = if axis != 0 { b.y } else { b.x };
        let ai = if greater { av >= bound } else { av <= bound };
        let bi = if greater { bv >= bound } else { bv <= bound };

        if ai != bi {
            let t = (bound - av) / (bv - av);
            let mut p = SoftPoint2 { x: a.x + t * (b.x - a.x), y: a.y + t * (b.y - a.y) };

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
    count as u32
}

pub fn soft_polygon_square_area(polygon: &[SoftPoint2], n: u32, x: i32, y: i32) -> f64 {
    let mut a = [SoftPoint2::default(); 12];
    let mut b = [SoftPoint2::default(); 12];
    let mut area = 0.0f64;
    let mut n = n;

    a[..n as usize].copy_from_slice(&polygon[..n as usize]);
    n = soft_clip_edge(&a, n, &mut b, 0, x as f64, true);
    n = soft_clip_edge(&b, n, &mut a, 0, (x + 1) as f64, false);
    n = soft_clip_edge(&a, n, &mut b, 1, y as f64, true);
    n = soft_clip_edge(&b, n, &mut a, 1, (y + 1) as f64, false);
    for i in 0..n as usize {
        let p = &a[i];
        let q = &a[(i + 1) % n as usize];

        area += p.x * q.y - p.y * q.x;
    }
    r2_min(1.0, area.abs() * 0.5)
}

pub fn soft_line_coverage(ax: f64, ay: f64, bx: f64, by: f64, width: f64, x: i32, y: i32) -> f64 {
    let dx = bx - ax;
    let dy = by - ay;
    let length = dx.hypot(dy);

    if !(length > 0.0) || !(width > 0.0) {
        return 0.0;
    }
    let nx = -dy * width / (2.0 * length);
    let ny = dx * width / (2.0 * length);
    let rectangle = [
        SoftPoint2 { x: ax + nx, y: ay + ny },
        SoftPoint2 { x: bx + nx, y: by + ny },
        SoftPoint2 { x: bx - nx, y: by - ny },
        SoftPoint2 { x: ax - nx, y: ay - ny },
    ];
    soft_polygon_square_area(&rectangle, 4, x, y)
}

pub fn soft_circle_integral(x: f64, radius: f64) -> f64 {
    let y = r2_max(0.0, radius * radius - x * x).sqrt();

    0.5 * (x * y + radius * radius * (x / radius).asin())
}

pub fn soft_circle_coverage(cx: f64, cy: f64, radius: f64, x: i32, y: i32) -> f64 {
    let mut cuts = [0.0f64; 8];
    let mut area = 0.0f64;
    let mut n = 0usize;
    let lo = r2_max(x as f64 - cx, -radius);
    let hi = r2_min((x + 1) as f64 - cx, radius);

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
    for i in 1..n {
        let value = cuts[i];
        let mut j = i;

        while j != 0 && cuts[j - 1] > value {
            cuts[j] = cuts[j - 1];
            j -= 1;
        }
        cuts[j] = value;
    }
    for i in 0..n.saturating_sub(1) {
        let a = cuts[i];
        let b = cuts[i + 1];
        let mid = (a + b) * 0.5;
        let circle = r2_max(0.0, radius * radius - mid * mid).sqrt();
        let ymin = y as f64 - cy;
        let ymax = (y + 1) as f64 - cy;
        let top = r2_min(ymax, circle);
        let bottom = r2_max(ymin, -circle);
        let arc = soft_circle_integral(b, radius) - soft_circle_integral(a, radius);

        if top <= bottom {
            continue;
        }
        area += (if top == circle { arc } else { ymax * (b - a) })
            - (if bottom == -circle { -arc } else { ymin * (b - a) });
    }
    r2_min(1.0, r2_max(0.0, area))
}

/// SGI_SMOOTH_TWO_PIXEL_FILTER's weight in one axis for a fragment whose
/// centre is d from a footprint of the given size: a box of that size
/// filtered by a one-pixel tent.
pub fn soft_filter_weight(size: f64, d: f64) -> f64 {
    r2_min(1.0, r2_max(0.0, (size + 1.0) * 0.5 - d.abs()))
}

pub fn soft_point_filter(cx: f64, cy: f64, size: f64, x: i32, y: i32) -> f64 {
    soft_filter_weight(size, cx - (x as f64 + 0.5)) * soft_filter_weight(size, cy - (y as f64 + 0.5))
}

pub fn soft_line_filter(ax: f64, ay: f64, bx: f64, by: f64, width: f64, x: i32, y: i32) -> f64 {
    let x_major = (bx - ax).abs() >= (by - ay).abs();
    let mut l0 = if x_major { ax } else { ay };
    let mut l1 = if x_major { bx } else { by };
    let mut n0 = if x_major { ay } else { ax };
    let mut n1 = if x_major { by } else { bx };
    let m = if x_major { x } else { y };
    let n = if x_major { y } else { x };

    if !(l1 != l0) {
        return 0.0;
    }
    if l1 < l0 {
        let mut t = l0;

        l0 = l1;
        l1 = t;
        t = n0;
        n0 = n1;
        n1 = t;
    }
    let lo = r2_max(l0, m as f64);
    let hi = r2_min(l1, m as f64 + 1.0);
    if !(hi > lo) {
        return 0.0;
    }
    let slope = (n1 - n0) / (l1 - l0);
    let minor = n0 + slope * ((lo + hi) * 0.5 - l0);
    let d = minor - (n as f64 + 0.5);
    r2_min(1.0, soft_filter_weight(width, d) * (1.0 + slope * slope).sqrt()) * (hi - lo)
}

impl RasterRules {
    pub fn point_coverage(&self, cx: f64, cy: f64, size: f64, x: i32, y: i32) -> f64 {
        if self.smooth_rule as u32 == SGI_SMOOTH_TWO_PIXEL_FILTER {
            return soft_point_filter(cx, cy, size, x, y);
        }
        soft_circle_coverage(cx, cy, size * 0.5, x, y)
    }

    pub fn line_coverage(&self, ax: f64, ay: f64, bx: f64, by: f64, width: f64, x: i32, y: i32) -> f64 {
        if self.smooth_rule as u32 == SGI_SMOOTH_TWO_PIXEL_FILTER {
            return soft_line_filter(ax, ay, bx, by, width, x, y);
        }
        soft_line_coverage(ax, ay, bx, by, width, x, y)
    }

    /// How far from a smoothed point's centre, or a smoothed line, in pixels,
    /// a fragment can have coverage.
    pub fn smooth_reach(&self, size: f64) -> f64 {
        if self.smooth_rule as u32 == SGI_SMOOTH_TWO_PIXEL_FILTER {
            (size + 1.0) * 0.5
        } else {
            size * 0.5
        }
    }
}

#[derive(Clone, Copy, Default)]
pub struct SoftLineStipple {
    pub counter: u32,
    pub repeat: u32,
    pub pattern: u32,
    pub enabled: bool,
    pub consumed: bool,
}

impl SoftVertex {
    pub fn line_vertex_at(&self, b: &SoftVertex, t: f64, out: &mut SoftVertex) {
        let wa = (1.0 - t) * self.inv_w;
        let wb = t * b.inv_w;
        let sum = wa + wb;

        out.x = (self.x as f64 + t * (b.x - self.x) as f64).round() as i64;
        out.y = (self.y as f64 + t * (b.y - self.y) as f64).round() as i64;
        out.z = self.z + t * (b.z - self.z);
        out.inv_w = (1.0 - t) * self.inv_w + t * b.inv_w;
        out.depth_w = (1.0 - t) * self.depth_w + t * b.depth_w;
        for k in 0..4 {
            out.rgba[k] = (wa * self.rgba[k] + wb * b.rgba[k]) / sum;
            out.texcoord[k] = (wa * self.texcoord[k] + wb * b.texcoord[k]) / sum;
        }
        out.index = (wa * self.index + wb * b.index) / sum;
        out.fog = (wa * self.fog + wb * b.fog) / sum;
        for i in 0..SGI_GL_CLIP_PLANES as usize {
            out.clip[i] = (wa * self.clip[i] + wb * b.clip[i]) / sum;
        }
    }
}

impl SoftDraw<'_> {
    /// Map and rasterize one aliased segment, already projected, with the
    /// provoking vertex's colour when flat shaded. Width follows the
    /// minor-axis shift and replication of 3.4.3. A smoothed segment takes the
    /// coverage of the rules' smooth_rule (3.4.2), stippled as one-pixel pieces
    /// along its length.
    pub fn line_projected(&self, pa: &SoftVertex, pb: &SoftVertex, provoking: &[f32], offset: f64, stipple: Option<&mut SoftLineStipple>, client_clip: bool) {
        let mut stipple = stipple;
        let st = self.st;
        let smooth = st.shade_model != SGI_GL_FLAT;
        let sub = self.rules.subpixel_bits as i32;
        let one_sub = (1i32 << sub) as f64;
        let a = pa.clone();
        let b = pb.clone();
        let (mut bx0, mut by0, mut bx1, mut by1) = (0i32, 0i32, -1i32, -1i32);
        let visible = self.bounds(&mut bx0, &mut by0, &mut bx1, &mut by1);

        if (!visible && stipple.is_none()) || (a.x == b.x && a.y == b.y) {
            return;
        }
        if self.line_smooth && stipple.as_ref().map_or(false, |s| !s.enabled) {
            let length = ((b.x - a.x) as f64 / one_sub).hypot((b.y - a.y) as f64 / one_sub);
            let s = stipple.as_mut().unwrap();
            let period = 16u32.wrapping_mul(s.repeat);

            // Unstippled, the segment is one rectangle (3.4.2): its fragments
            // are not split between one-pixel pieces, each blended on its own.
            // The count still moves on by the pieces a stipple would take.
            s.counter = s.counter.wrapping_add((length.ceil() % period as f64) as u32) % period;
            s.consumed = true;
            if !visible {
                return;
            }
            stipple = None;
        }
        if self.line_smooth && stipple.is_some() {
            let s = stipple.as_mut().unwrap();
            let length = ((b.x - a.x) as f64 / one_sub).hypot((b.y - a.y) as f64 / one_sub);
            let segments = length.ceil();
            let mut i = 0.0f64;

            while i < segments {
                let bit = (s.counter / s.repeat) % 16;
                let keep = !s.enabled || (s.pattern & (1u32 << bit)) != 0;

                s.counter = (s.counter.wrapping_add(1)) % 16u32.wrapping_mul(s.repeat);
                s.consumed = true;
                if keep {
                    let mut start = SoftVertex::default();
                    let mut end = SoftVertex::default();
                    let t0 = i / length;
                    let t1 = r2_min((i + 1.0) / length, 1.0);

                    a.line_vertex_at(&b, t0, &mut start);
                    a.line_vertex_at(&b, t1, &mut end);
                    self.line_projected(&start, &end, provoking, offset, None, client_clip);
                }
                i += 1.0;
            }
            return;
        }
        let mut texture_value = [0.0f64; 4];
        let mut texture_delta = [0.0f64; 4];
        let mut texture_length = 0.0f64;
        if self.texture_base.is_some()
            && !a.texture_line_gradient(
                &b,
                sub as u32,
                self.image().volume,
                self.image().one_dimensional,
                &mut texture_value,
                &mut texture_delta,
                &mut texture_length,
            )
        {
            self.texture_report(16, "invalid primitive coefficients, primitive dropped");
            return;
        }
        let mut flat = [0.0f64; 4];
        for k in 0..4 {
            flat[k] = provoking[SGI_GL_V_COLOR + k] as f64;
        }
        let len2 = (b.x - a.x) as f64 * (b.x - a.x) as f64 + (b.y - a.y) as f64 * (b.y - a.y) as f64;
        let x_major = (b.x - a.x).wrapping_abs() >= (b.y - a.y).wrapping_abs();
        let mut shifted_a = a.clone();
        let mut shifted_b = b.clone();
        // Width <= INT32_MAX and sub <= 12 give a shift below 2^42.
        let shift = (self.line_width.wrapping_sub(1) as i64) * (1i64 << (sub - 1));
        if x_major {
            shifted_a.y -= shift;
            shifted_b.y -= shift;
        } else {
            shifted_a.x -= shift;
            shifted_b.x -= shift;
        }

        // Walk the major axis a pixel at a time, and in each step try the
        // minor-axis pixels within one of those the segment spans there.
        let mut major0 = ((if x_major { a.x.min(b.x) } else { a.y.min(b.y) }) >> sub) as i32;
        let mut major1 = ((if x_major { a.x.max(b.x) } else { a.y.max(b.y) }) >> sub) as i32;
        major0 -= 1;
        major1 += 1;
        if self.line_smooth {
            let pad = self.rules.smooth_reach(self.smooth_line_width);
            let l0 = (if x_major { a.x } else { a.y }) as f64;
            let l1 = (if x_major { b.x } else { b.y }) as f64;

            major0 = (r2_min(l0, l1) / one_sub - pad).floor() as i32 - 1;
            major1 = (r2_max(l0, l1) / one_sub + pad).ceil() as i32 + 1;
        }
        let mut step = 1i32;
        if stipple.is_some() {
            // Count coverage before ownership/scissor, in submitted direction.
            if if x_major { b.x < a.x } else { b.y < a.y } {
                let tmp = major0;

                major0 = major1;
                major1 = tmp;
                step = -1;
            }
        } else {
            major0 = major0.max(if x_major { bx0 } else { by0 });
            major1 = major1.min(if x_major { bx1 } else { by1 });
        }
        let mut mm = major0;
        loop {
            if !(if step > 0 { mm <= major1 } else { mm >= major1 }) {
                break;
            }
            let m = mm;
            mm += step;
            let (lo, hi) = if x_major {
                if a.x <= b.x { (&shifted_a, &shifted_b) } else { (&shifted_b, &shifted_a) }
            } else if a.y <= b.y {
                (&shifted_a, &shifted_b)
            } else {
                (&shifted_b, &shifted_a)
            };
            let l0 = (if x_major { lo.x } else { lo.y }) as f64;
            let l1 = (if x_major { hi.x } else { hi.y }) as f64;
            let n0 = (if x_major { lo.y } else { lo.x }) as f64;
            let n1 = (if x_major { hi.y } else { hi.x }) as f64;
            let mut e0 = r2_max(m as f64 * one_sub, l0);
            let mut e1 = r2_min((m + 1) as f64 * one_sub, l1);
            let f0: f64;
            let f1: f64;

            if l1 > l0 {
                e0 = r2_min(e0, l1);
                e1 = r2_max(e1, l0);
                f0 = n0 + (n1 - n0) * (e0 - l0) / (l1 - l0);
                f1 = n0 + (n1 - n0) * (e1 - l0) / (l1 - l0);
            } else {
                f0 = n0;
                f1 = n1;
            }
            let mut minor0 = (r2_min(f0, f1) / one_sub).floor() as i64 - 1;
            let mut minor1 = (r2_max(f0, f1) / one_sub).floor() as i64 + 1;
            if self.line_smooth {
                let pad = self.rules.smooth_reach(self.smooth_line_width);
                let q0 = (if x_major { a.y } else { a.x }) as f64;
                let q1 = (if x_major { b.y } else { b.x }) as f64;

                minor0 = (r2_min(q0, q1) / one_sub - pad).floor() as i64 - 1;
                minor1 = (r2_max(q0, q1) / one_sub + pad).ceil() as i64 + 1;
            }
            let mut nn = minor0;
            while nn <= minor1 {
                let n = nn;
                nn += 1;
                let base_x: i64 = if x_major { m as i64 } else { n };
                let base_y: i64 = if x_major { n } else { m as i64 };
                let replica0: i64 = r2_max(n, (if x_major { by0 } else { bx0 }) as i64);
                let replica1: i64 = if self.line_smooth {
                    replica0
                } else {
                    r2_min(n + self.line_width as i64 - 1, (if x_major { by1 } else { bx1 }) as i64)
                };
                let mut keep = true;
                let coverage = if self.line_smooth {
                    self.rules.line_coverage(
                        a.x as f64 / one_sub,
                        a.y as f64 / one_sub,
                        b.x as f64 / one_sub,
                        b.y as f64 / one_sub,
                        self.smooth_line_width,
                        base_x as i32,
                        base_y as i32,
                    )
                } else if shifted_a.line_covers(&shifted_b, base_x, base_y, sub) {
                    1.0
                } else {
                    0.0
                };

                if !(coverage > 0.0) {
                    continue;
                }
                if let Some(s) = stipple.as_mut() {
                    let bit = (s.counter / s.repeat) % 16;

                    keep = !s.enabled || (s.pattern & (1u32 << bit)) != 0;
                    s.counter = s.counter.wrapping_add(1) % 16u32.wrapping_mul(s.repeat);
                    s.consumed = true;
                }
                if !keep
                    || !visible
                    || replica0 > replica1
                    || m < (if x_major { bx0 } else { by0 })
                    || m > (if x_major { bx1 } else { by1 })
                {
                    continue;
                }
                // Visit only drawable-visible replicas, never width entries.
                let mut replica = replica0;
                while replica <= replica1 {
                    let rep = replica;
                    replica += 1;
                    let px: i32 = if x_major { m } else { rep as i32 };
                    let py: i32 = if x_major { rep as i32 } else { m };
                    let cx = (px as f64 + 0.5) * one_sub;
                    let cy = (py as f64 + 0.5) * one_sub;
                    let mut frag = SoftFrag::default();

                    frag.fixed = false;
                    let mut t = ((cx - a.x as f64) * (b.x - a.x) as f64 + (cy - a.y as f64) * (b.y - a.y) as f64) / len2;
                    // Approximation: retain endpoint values, not extrapolation.
                    t = r2_min(r2_max(t, 0.0), 1.0);
                    let wa = (1.0 - t) * a.inv_w;
                    let wb = t * b.inv_w;
                    frag.index = if smooth {
                        a.index + wb * (b.index - a.index) / (wa + wb)
                    } else {
                        provoking[SGI_GL_V_INDEX] as f64
                    };
                    for k in 0..4 {
                        frag.texcoord[k] = (wa * a.texcoord[k] + wb * b.texcoord[k]) / (wa + wb);
                    }
                    frag.fog = (wa * a.fog + wb * b.fog) / (wa + wb);
                    if st.clip_planes != 0 && client_clip {
                        let mut d = [0.0f64; SGI_GL_CLIP_PLANES as usize];

                        for i in 0..SGI_GL_CLIP_PLANES as usize {
                            d[i] = (wa * a.clip[i] + wb * b.clip[i]) / (wa + wb);
                        }
                        if st.clipped(&d) {
                            continue;
                        }
                    }
                    for k in 0..4 {
                        frag.rgba[k] = if smooth { (wa * a.rgba[k] + wb * b.rgba[k]) / (wa + wb) } else { flat[k] };
                    }
                    if self.texture_base.is_some() {
                        let mut value = [0.0f64; 4];
                        let mut out = [0.0f64; 5];
                        let zero = [0.0f64; 4];

                        for k in 0..4 {
                            value[k] = texture_value[k] + t * texture_delta[k];
                        }
                        frag.texture_rho = if soft_texture_derivatives(
                            &value,
                            &texture_delta,
                            &zero,
                            self.base_level().width as f64 * soft_texture_lod_scale(self.image().lod_scale_s),
                            if self.image().one_dimensional {
                                0.0
                            } else {
                                self.base_level().height as f64 * soft_texture_lod_scale(self.image().lod_scale_t)
                            },
                            if self.image().volume {
                                self.base_level().depth as f64 * soft_texture_lod_scale(self.image().lod_scale_r)
                            } else {
                                0.0
                            },
                            &mut out,
                        ) {
                            out[4] / texture_length
                        } else {
                            f64::NAN
                        };
                    }
                    self.fragment_coverage(
                        px,
                        py,
                        a.z + t * (b.z - a.z),
                        offset,
                        (1.0 - t) * a.depth_w + t * b.depth_w,
                        &frag,
                        coverage,
                        self.line_smooth,
                    );
                }
            }
        }
    }

    /// Keep standalone client-plane rejection after geometric counting.
    pub fn line(&self, va: &[f32], vb: &[f32], provoking: &[f32], stipple: Option<&mut SoftLineStipple>) {
        let mut ca = VolumeVertex::default();
        let mut cb = VolumeVertex::default();
        let mut a = SoftVertex::default();
        let mut b = SoftVertex::default();
        let (mut bx0, mut by0, mut bx1, mut by1) = (0i32, 0i32, 0i32, 0i32);

        if stipple.is_none() && !self.bounds(&mut bx0, &mut by0, &mut bx1, &mut by1) {
            return;
        }
        self.st.volume_from_draw_vertex(va, &mut ca);
        self.st.volume_from_draw_vertex(vb, &mut cb);
        if ca.clip_segment(&mut cb) && self.project(&ca, &mut a) && self.project(&cb, &mut b) {
            self.line_projected(&a, &b, provoking, 0.0, stipple, true);
        }
    }

    /// Aliased square-point centring follows OpenGL 1.2.1, 3.3, on the shared
    /// snapped subpixel grid. A smoothed point takes the coverage of the
    /// rules' smooth_rule about its snapped centre (3.3.2).
    pub fn point_projected(&self, pv: &SoftVertex, provoking: Option<&[f32]>, offset: f64) {
        let sub = self.rules.subpixel_bits as i32;
        let v = pv.clone();
        let mut frag = SoftFrag::default();
        let (mut bx0, mut by0, mut bx1, mut by1) = (0i32, 0i32, 0i32, 0i32);
        let psize = self.point_size as i64;

        if !self.bounds(&mut bx0, &mut by0, &mut bx1, &mut by1) {
            return;
        }
        // Even dimensions centre on a pixel corner; odd ones on a centre.
        let px: i64 = (v.x + (if self.point_size % 2 != 0 { 0 } else { 1i64 << (sub - 1) })) >> sub;
        let py: i64 = (v.y + (if self.point_size % 2 != 0 { 0 } else { 1i64 << (sub - 1) })) >> sub;
        let mut x0: i64 = r2_max(px - psize / 2, bx0 as i64);
        let mut y0: i64 = r2_max(py - psize / 2, by0 as i64);
        let mut x1: i64 = r2_min(px - psize / 2 + psize - 1, bx1 as i64);
        let mut y1: i64 = r2_min(py - psize / 2 + psize - 1, by1 as i64);
        frag.rgba = v.rgba;
        frag.index = v.index;
        frag.fixed = false;
        if let Some(p) = provoking {
            if self.st.shade_model == SGI_GL_FLAT {
                frag.index = p[SGI_GL_V_INDEX] as f64;
                for k in 0..4 {
                    frag.rgba[k] = p[SGI_GL_V_COLOR + k] as f64;
                }
            }
        }
        frag.fog = v.fog;
        frag.texture_rho = 1.0;
        frag.texcoord = v.texcoord;
        if self.point_smooth {
            let scale = (1i32 << sub) as f64;
            let cx = v.x as f64 / scale;
            let cy = v.y as f64 / scale;
            let radius = self.rules.smooth_reach(self.smooth_point_size);

            x0 = r2_max((cx - radius).floor() as i64, bx0 as i64);
            y0 = r2_max((cy - radius).floor() as i64, by0 as i64);
            x1 = r2_min((cx + radius).ceil() as i64 - 1, bx1 as i64);
            y1 = r2_min((cy + radius).ceil() as i64 - 1, by1 as i64);
            let mut y = y0;
            while y <= y1 {
                let mut x = x0;
                while x <= x1 {
                    let coverage = self.rules.point_coverage(cx, cy, self.smooth_point_size, x as i32, y as i32);

                    self.fragment_coverage(x as i32, y as i32, v.z, offset, v.depth_w, &frag, coverage, true);
                    x += 1;
                }
                y += 1;
            }
            return;
        }
        let mut y = y0;
        while y <= y1 {
            let mut x = x0;
            while x <= x1 {
                self.fragment(x as i32, y as i32, v.z, offset, v.depth_w, &frag);
                x += 1;
            }
            y += 1;
        }
    }

    pub fn point(&self, vp: &[f32]) {
        let mut cv = VolumeVertex::default();
        let mut v = SoftVertex::default();

        self.st.volume_from_draw_vertex(vp, &mut cv);
        if cv.contains() && self.project(&cv, &mut v) && !self.st.clipped(&v.clip) {
            self.point_projected(&v, None, 0.0);
        }
    }
}

/// Internal representation bound, not an SGI width range. Half ties round up.
pub fn soft_dimension(requested: f32, rounded: &mut u32) -> bool {
    let n = if requested == 0.0 { 1.0 } else { (requested as f64 + 0.5).floor() };

    if !requested.is_finite() || requested < 0.0 || n > i32::MAX as f64 {
        crate::dlog!(LogModule::Odyssey, "sgi-gl: unsupported aliased dimension {}", requested);
        return false;
    }
    *rounded = r2_max(n, 1.0) as u32;
    true
}
