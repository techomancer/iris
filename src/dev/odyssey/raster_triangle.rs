//! Triangles and polygons: clipping against the view volume and the user
//! clip planes, the texture-coordinate planes across a triangle, edge
//! walking and spans, antialiased polygon coverage, and polygon modes.
//!
//! Ported from hw/display/sgi/rasterizer.c in jamesbraid/qemu-sgi-origami.

use super::*;

/// The six planes of the view volume (OpenGL 1.2.1, 2.11), -w <= x, y,
/// z <= w, as each one's value at a point, which is >= 0 inside it.
pub const SGI_VOLUME_PLANES: i32 = 6;

impl VolumeVertex {
    pub fn plane(&self, plane: i32) -> f64 {
        let c = self.clip[(plane / 2) as usize];

        if plane & 1 != 0 { self.clip[3] - c } else { self.clip[3] + c }
    }

    pub fn contains(&self) -> bool {
        for i in 0..SGI_VOLUME_PLANES {
            if !(self.plane(i) >= 0.0) {
                return false;
            }
        }
        true
    }

    /// The point at t along from in to out, in clip and eye coordinates and
    /// every float attribute (2.13.8); the material index and edge flag are
    /// bit patterns and stay in's.
    pub fn lerp(&self, out: &VolumeVertex, t: f64, r: &mut VolumeVertex) {
        for k in 0..4usize {
            r.clip[k] = self.clip[k] + t * (out.clip[k] - self.clip[k]);
            r.eye[k] = self.eye[k] + t * (out.eye[k] - self.eye[k]);
        }
        for k in 0..SGI_GL_VERTEX_FLOATS {
            r.attr[k] = if k == SGI_GL_V_MATERIAL || k == SGI_GL_V_EDGE {
                self.attr[k]
            } else {
                (self.attr[k] as f64 + t * (out.attr[k] - self.attr[k]) as f64) as f32
            };
        }
    }

    /// Where the edge from in, inside a plane with value din, to out, outside
    /// it with value dout, meets the plane. It is always found from the inside
    /// end, so that two primitives sharing the edge make the same vertex.
    pub fn cut(&self, din: f64, out: &VolumeVertex, dout: f64, r: &mut VolumeVertex) {
        self.lerp(out, din / (din - dout), r);
    }
}

/// Sutherland and Hodgman's clipping, one plane at a time: each edge
/// keeps its inside end and adds the point where it crosses the plane.
/// A value that is not a number is outside.
pub fn volume_clip_triangle(in_: &[VolumeVertex; 3], out: &mut [VolumeVertex; SGI_VOLUME_TRIANGLE_MAX]) -> u32 {
    let mut src: [VolumeVertex; SGI_VOLUME_TRIANGLE_MAX] = std::array::from_fn(|_| VolumeVertex::default());
    let mut dst: [VolumeVertex; SGI_VOLUME_TRIANGLE_MAX] = std::array::from_fn(|_| VolumeVertex::default());
    let mut n: usize = 3;

    for i in 0..3 {
        out[i] = in_[i].clone();
        src[i] = in_[i].clone();
    }
    if in_[0].contains() && in_[1].contains() && in_[2].contains() {
        return 3;
    }
    for p in 0..SGI_VOLUME_PLANES {
        let mut m: usize = 0;

        for i in 0..n {
            let a = &src[i];
            let b = &src[(i + 1) % n];
            let da = a.plane(p);
            let db = b.plane(p);

            if da >= 0.0 {
                dst[m] = a.clone();
                m += 1;
                if !(db >= 0.0) {
                    let mut c = VolumeVertex::default();
                    a.cut(da, b, db, &mut c);
                    dst[m] = c;
                    m += 1;
                }
            } else if db >= 0.0 {
                let mut c = VolumeVertex::default();
                b.cut(db, a, da, &mut c);
                dst[m] = c;
                m += 1;
            }
        }
        n = m;
        std::mem::swap(&mut src, &mut dst);
        if n < 3 {
            return 0;
        }
    }
    for i in 0..n {
        out[i] = src[i].clone();
    }
    n as u32
}

impl VolumeVertex {
    /// Liang and Barsky's clipping: the segment keeps the part every plane
    /// allows. Each end that a plane bounds moves to where the segment meets
    /// that plane, found from the other end, inside it, as a triangle's cut
    /// is: a fraction sa of the way from b to a, and tb from a to b.
    pub fn clip_segment(&mut self, b: &mut VolumeVertex) -> bool {
        let mut na = self.clone();
        let mut nb = b.clone();
        let mut sa = 1.0f64;
        let mut tb = 1.0f64;

        for p in 0..SGI_VOLUME_PLANES {
            let da = self.plane(p);
            let db = b.plane(p);

            if da >= 0.0 && db >= 0.0 {
                continue;
            }
            if !(da >= 0.0) && !(db >= 0.0) {
                return false;
            }
            if da >= 0.0 {
                tb = fmin(tb, da / (da - db));
            } else {
                sa = fmin(sa, db / (db - da));
            }
        }
        if !(sa + tb >= 1.0) {
            return false;
        }
        if sa < 1.0 {
            b.lerp(self, sa, &mut na);
        }
        if tb < 1.0 {
            self.lerp(b, tb, &mut nb);
        }
        *self = na;
        *b = nb;
        true
    }
}

impl GlDrawState {
    /// One vertex to clip coordinates (OpenGL 1.2.1, 2.10): object to eye by
    /// the modelview and eye to clip by the projection, or by the window
    /// matrix as GlDrawState::eye_to_clip() takes it.
    pub fn volume_from_draw_vertex(&self, v: &[f32], out: &mut VolumeVertex) {
        for row in 0..4usize {
            out.eye[row] = 0.0;
            for k in 0..4usize {
                out.eye[row] += self.modelview[k * 4 + row] as f64 * v[SGI_GL_V_POSITION + k] as f64;
            }
        }
        self.eye_to_clip(&out.eye, &mut out.clip);
        out.attr.copy_from_slice(&v[..SGI_GL_VERTEX_FLOATS]);
    }
}

impl SoftVertex {
    pub fn texture_values(&self, volume: bool, one_dimensional: bool, value: &mut [f64; 4]) {
        value[0] = self.texcoord[0] * self.inv_w;
        value[1] = if one_dimensional { 0.0 } else { self.texcoord[1] * self.inv_w };
        value[2] = self.texcoord[3] * self.inv_w;
        value[3] = if volume { self.texcoord[2] * self.inv_w } else { 0.0 };
    }
}

pub fn soft_texture_plane(v: &[SoftVertex], sub: u32, volume: bool, one_dimensional: bool, g: &mut SoftTexPlane) -> bool {
    let scale = (1i32 << sub) as f64;
    let x1 = (v[1].x - v[0].x) as f64 / scale;
    let y1 = (v[1].y - v[0].y) as f64 / scale;
    let x2 = (v[2].x - v[0].x) as f64 / scale;
    let y2 = (v[2].y - v[0].y) as f64 / scale;
    let area = x1 * y2 - x2 * y1;
    let mut values = [[0.0f64; 4]; 3];

    *g = SoftTexPlane { x: v[0].x as f64 / scale, y: v[0].y as f64 / scale, ..Default::default() };
    if area == 0.0 {
        return true;
    }
    if !area.is_finite() {
        return false;
    }
    for i in 0..3 {
        v[i].texture_values(volume, one_dimensional, &mut values[i]);
    }
    for k in 0..4 {
        let a = values[1][k] - values[0][k];
        let b = values[2][k] - values[0][k];

        g.value[k] = values[0][k];
        g.dx[k] = (a * y2 - b * y1) / area;
        g.dy[k] = (b * x1 - a * x2) / area;
        if !g.value[k].is_finite() || !g.dx[k].is_finite() || !g.dy[k].is_finite() {
            return false;
        }
    }
    true
}

impl SoftDraw {
    pub fn texture_coefficients(&self, v: &[SoftVertex]) -> bool {
        let mut g = SoftTexPlane::default();

        if self.texture_base.is_null()
            || soft_texture_plane(v, self.rules.subpixel_bits as u32, self.texture.volume, self.texture.one_dimensional, &mut g)
        {
            return true;
        }
        self.texture_report(16, "invalid primitive coefficients, primitive dropped");
        false
    }

    /// Whether a triangle's fragments can take SoftDraw::simple_row() instead of
    /// the full per-fragment path.
    pub fn simple_fragments(&self) -> bool {
        let st = &*self.st;
        let t = &*self.t;

        if !self.texture.is_null()
            || st.fog != 0
            || st.clip_planes != 0
            || !self.polygon_stipple.is_null()
            || st.index_bits != 0
            || self.blend
            || self.logic_op
            || st.alpha_test != 0
            || st.stencil_test != 0
        {
            return false;
        }
        // RasterRules::depth_stencil() drops every fragment of such a draw.
        if st.depth_test != 0 && (t.load_depth.is_none() || (st.depth_mask != 0 && t.store_depth.is_none())) {
            return false;
        }
        if t.store_rgba.is_some() {
            return !self.rgba_read;
        }
        self.colour_mask == 0 || self.colour_mask == self.rules.colour_bits()
    }

    /// One row of a triangle under SoftDraw::simple_fragments(): pixels px0 to
    /// px1 of row py, whose edge values for v[2]-v[0] and v[0]-v[1] at px0 are
    /// e1 and e2.
    pub fn simple_row(&self, v: &[SoftVertex], py: i32, mut px0: i32, mut px1: i32, mut e1: i64, mut e2: i64, step1: i64, step2: i64, area: i64, offset: f64, flat: &SoftFrag, smooth: bool, rgba: bool, depth_row: bool) {
        let rules = &*self.rules;
        let st = &*self.st;
        let t = &*self.t;
        let y = self.win.oy + py;
        // A flat colour is the same at every pixel unless dithered.
        let fixed = !smooth && !self.dither;
        let store = if rgba { self.rgba_mask != 0 && self.ntargets != 0 } else { self.colour_mask != 0 };
        let clipped = self.win.clip.n > 1;
        let mut fixed_rgba = RasterColour::default();
        let mut fixed_colour = 0u32;

        if st.scissor_test != 0 && !clipped {
            // SoftDraw::owned_at()'s scissor test, decided for the whole row.
            let sx0 = st.scissor[0] as i64;
            let sx1 = st.scissor[0] as i64 + st.scissor[2] as i64 - 1;

            if !(py >= st.scissor[1] && (py as i64) < st.scissor[1] as i64 + st.scissor[3] as i64) {
                return;
            }
            if sx0 > px0 as i64 {
                if sx0 > px1 as i64 {
                    return;
                }
                e1 += (sx0 - px0 as i64) * step1;
                e2 += (sx0 - px0 as i64) * step2;
                px0 = sx0 as i32;
            }
            px1 = (px1 as i64).min(sx1) as i32;
        }
        if fixed && store {
            if rgba {
                let alpha = rules.alpha(flat.rgba[3]) as f64 / rules.alpha_max() as f64;

                fixed_rgba = rules.rgba_value(rules.rgb(&flat.rgba), alpha);
            } else {
                fixed_colour = rules.rgb(&flat.rgba);
            }
        }
        let mut px = px0;
        while px <= px1 {
            let x = self.win.ox + px;
            let b1 = e1 as f64 / area as f64;
            let b2 = e2 as f64 / area as f64;
            let mut z = v[0].z + b1 * (v[1].z - v[0].z) + b2 * (v[2].z - v[0].z);
            let mut off = offset;
            let mut zw = 0.0f64;
            let mut colour = [0.0f64; 4];

            'pixel: {
                if depth_row {
                    let depth_w = (1.0 - b1 - b2) * v[0].depth_w + b1 * v[1].depth_w + b2 * v[2].depth_w;
                    let slack = rules.depth_slack;

                    z = (z + off) / depth_w;
                    off = 0.0;
                    if z < 0.0 && z >= -slack {
                        z = 0.0;
                    } else if z > 1.0 && z <= 1.0 + slack {
                        z = 1.0;
                    }
                }
                if !st.window_depth(z, off, &mut zw) || !(zw >= 0.0 && zw <= 1.0) {
                    break 'pixel;
                }
                if clipped && !self.owned_at(px, py) {
                    break 'pixel;
                }
                if st.depth_test != 0 || store {
                    let zq = rules.depth_code(zw);

                    if st.depth_test != 0 {
                        if !soft_compare(st.depth_func, zq, (t.load_depth.unwrap())(t.opaque, x, y)) {
                            break 'pixel;
                        }
                        if st.depth_mask != 0 {
                            (t.store_depth.unwrap())(t.opaque, x, y, zq);
                        }
                    }
                }
                if !store {
                    break 'pixel;
                }
                if smooth {
                    let w0 = (1.0 - b1 - b2) * v[0].inv_w;
                    let w1 = b1 * v[1].inv_w;
                    let w2 = b2 * v[2].inv_w;
                    let sum = w0 + w1 + w2;

                    for k in 0..4 {
                        colour[k] = (w0 * v[0].rgba[k] + w1 * v[1].rgba[k] + w2 * v[2].rgba[k]) / sum;
                    }
                } else {
                    colour = flat.rgba;
                }
                if rgba {
                    let mut out = fixed_rgba;

                    if !fixed {
                        let alpha = rules.alpha(colour[3]) as f64 / rules.alpha_max() as f64;

                        out = rules.rgba_value(self.store_rgb(x, y, &colour), alpha);
                    }
                    for i in 0..self.ntargets as usize {
                        (t.store_rgba.unwrap())(t.opaque, self.targets[i], x, y, &out);
                    }
                } else {
                    let out = if fixed { fixed_colour } else { self.store_rgb(x, y, &colour) };

                    for i in 0..self.ntargets as usize {
                        (t.store_colour.unwrap())(t.opaque, self.targets[i], x, y, out);
                    }
                }
            }
            px += 1;
            e1 += step1;
            e2 += step2;
        }
    }

    /// Out of line, once a row.
    pub fn simple_span(&self, v: &[SoftVertex], py: i32, px0: i32, px1: i32, e1: i64, e2: i64, step1: i64, step2: i64, area: i64, offset: f64, flat: &SoftFrag, smooth: bool) {
        let rgba = self.t.store_rgba.is_some();
        let depth_row = self.rules.depth_row_given;

        self.simple_row(v, py, px0, px1, e1, e2, step1, step2, area, offset, flat, smooth, rgba, depth_row);
    }

    /// Map and rasterize one triangle of a clipped polygon, given its three
    /// vertices in the view volume and, for flat shading, the provoking vertex
    /// of the triangle it was clipped from.
    pub fn clipped_triangle(&self, va: &VolumeVertex, vb: &VolumeVertex, vc: &VolumeVertex, provoking: &[f32]) {
        let st = &*self.st;
        let verts: [&VolumeVertex; 3] = [va, vb, vc];
        let mut flat = &provoking[SGI_GL_V_COLOR..SGI_GL_V_COLOR + 4];
        let mut flat_index = provoking[SGI_GL_V_INDEX];
        let mut flat_frag = SoftFrag::default();
        let mut frag;
        let mut v: [SoftVertex; 3] = Default::default();
        let mut gradient = SoftTexPlane::default();
        // The footprint's per-axis scales, and its value where Q/W is flat.
        let mut tex_ns = 0.0f64;
        let mut tex_nt = 0.0f64;
        let mut tex_nr = 0.0f64;
        let mut flat_rho = f64::NAN;
        let mut flat_q = false;
        let mut flat_ok = false;
        // Fragments sample in fixed point, and their lambda where it is flat.
        let fixed: bool;
        let mut fixed_flat: Option<i32> = None;
        // The row loop for textured fragments, and its inputs.
        let mut span: Option<SoftTexSpanFn> = None;
        let mut sp = SoftTexSpan::default();
        let mut area: i64;
        let offset: f64;
        let ccw: bool;
        let front: bool;
        let smooth = st.shade_model != SGI_GL_FLAT;
        let simple = self.simple_fragments();
        let mut tex_row = [0.0f64; 4];
        let sub = self.rules.subpixel_bits as i32;

        for i in 0..3 {
            if !self.project(verts[i], &mut v[i]) {
                return;
            }
        }

        area = v[0].edge(&v[1], v[2].x, v[2].y);
        if area == 0 {
            return;
        }
        if !self.texture_base.is_null()
            && !soft_texture_plane(&v, sub as u32, self.texture.volume, self.texture.one_dimensional, &mut gradient)
        {
            self.texture_report(16, "invalid primitive coefficients, primitive dropped");
            return;
        }
        // An image with no pyramid has no planes to fail: it samples in double.
        fixed = !self.fixed.is_null()
            && (!self.texture_base.is_null() || soft_texture_plane(&v, sub as u32, false, false, &mut gradient));
        if fixed && self.fixed.lod && gradient.dx[2] == 0.0 && gradient.dy[2] == 0.0 {
            // Q/W flat, as for the double footprint below: every fragment's
            // lambda is this one, which SoftFixedTex::fixed_coords() would compute
            // for each of them, since s dQ/dx and t dQ/dx are then zero.
            let q = gradient.value[2];

            if q > 0.0 {
                let iq = 1.0 / q;
                let a = self.fixed.ns * gradient.dx[0] * iq;
                let b = self.fixed.nt * gradient.dx[1] * iq;
                let c = self.fixed.ns * gradient.dy[0] * iq;
                let e = self.fixed.nt * gradient.dy[1] * iq;
                let rho2 = if a * a + b * b > c * c + e * e { a * a + b * b } else { c * c + e * e };

                if rho2 <= f64::MAX {
                    fixed_flat = Some(self.fixed.fixed_decide(rho2, soft_fixed_lambda(rho2)));
                }
            }
        }
        if !self.texture_base.is_null() {
            tex_ns = self.texture_base.width as f64 * soft_texture_lod_scale(self.texture.lod_scale_s);
            tex_nt = if self.texture.one_dimensional {
                0.0
            } else {
                self.texture_base.height as f64 * soft_texture_lod_scale(self.texture.lod_scale_t)
            };
            tex_nr = if self.texture.volume {
                self.texture_base.depth as f64 * soft_texture_lod_scale(self.texture.lod_scale_r)
            } else {
                0.0
            };
            // With Q/W the same at every vertex the footprint is one number.
            flat_q = gradient.dx[2] == 0.0 && gradient.dy[2] == 0.0;
            if flat_q {
                let value = [0.0, 0.0, gradient.value[2], 0.0];
                let mut out = [0.0f64; 5];

                flat_ok = soft_texture_derivatives(&value, &gradient.dx, &gradient.dy, tex_ns, tex_nt, tex_nr, &mut out);
                flat_rho = out[4];
            }
        }
        ccw = if st.window_matrix_given != 0 && st.window_rows_down != 0 { area < 0 } else { area > 0 };
        front = if st.front_face == SGI_GL_CW { !ccw } else { ccw };
        if st.cull_face != 0
            && (st.cull_face_mode == SGI_GL_FRONT_AND_BACK
                || st.cull_face_mode == (if front { SGI_GL_FRONT } else { SGI_GL_BACK }))
        {
            return;
        }
        if !front && self.back_colors {
            flat = &provoking[SGI_GL_V_BACK_COLOR..SGI_GL_V_BACK_COLOR + 4];
            flat_index = provoking[SGI_GL_V_BACK_INDEX];
            for i in 0..3 {
                let back = &verts[i].attr[SGI_GL_V_BACK_COLOR..SGI_GL_V_BACK_COLOR + 4];

                for k in 0..4 {
                    v[i].rgba[k] = back[k].max(0.0).min(1.0) as f64;
                }
                v[i].index = verts[i].attr[SGI_GL_V_BACK_INDEX] as f64;
            }
        }
        for k in 0..4 {
            flat_frag.rgba[k] = flat[k] as f64;
        }
        flat_frag.index = flat_index as f64;
        flat_frag.fixed = false;
        offset = self.offset(&v);
        if area < 0 {
            // Rasterize either winding the same way from here.
            v.swap(1, 2);
            area = -area;
        }

        let min_x = v[0].x.min(v[1].x.min(v[2].x)) >> sub;
        let max_x = v[0].x.max(v[1].x.max(v[2].x)) >> sub;
        let min_y = v[0].y.min(v[1].y.min(v[2].y)) >> sub;
        let max_y = v[0].y.max(v[1].y.max(v[2].y)) >> sub;

        // Window pixels, bounded by the viewport and the owned rectangle.
        let (mut px0, mut py0, mut px1, mut py1) = (0i32, 0i32, 0i32, 0i32);
        if !self.bounds(&mut px0, &mut py0, &mut px1, &mut py1) {
            return;
        }
        px0 = (min_x as i32).max(px0);
        px1 = (max_x as i32).min(px1);
        py0 = (min_y as i32).max(py0);
        py1 = (max_y as i32).min(py1);

        // Each edge function is linear in x, so across a row it steps by an
        // exact integer; solving for the ends of each edge's run of owned
        // centres visits only the pixels a test of every centre would keep.
        let step0: i64 = -(v[2].y - v[1].y) * (1i64 << sub);
        let step1: i64 = -(v[0].y - v[2].y) * (1i64 << sub);
        let step2: i64 = -(v[1].y - v[0].y) * (1i64 << sub);
        if fixed && self.fixed.span.is_some() {
            span = self.fixed.span;
            sp = SoftTexSpan {
                v: P::new(&mut v),
                g: P::new(&mut gradient),
                row: P::new(&mut tex_row),
                flat_lambda: fixed_flat,
                flat: P::new(&mut flat_frag),
                offset,
                area,
                step1,
                step2,
                smooth,
                constant: 0,
            };
            for k in 0..4 {
                let c = v[0].rgba[k];

                if (c == 0.0 || c == 1.0) && v[1].rgba[k] == c && v[2].rgba[k] == c {
                    sp.constant |= 1u8 << k;
                }
            }
        }
        let mut py = py0;
        while py <= py1 {
            let cur_py = py;
            py += 1;
            if !self.row_in_band(cur_py) {
                continue;
            }
            let py = cur_py;
            let cy: i64 = (py as i64 * (1i64 << sub)) + (1i64 << (sub - 1));
            let cx: i64 = (px0 as i64 * (1i64 << sub)) + (1i64 << (sub - 1));
            let e0 = v[1].edge(&v[2], cx, cy);
            let mut e1 = v[2].edge(&v[0], cx, cy);
            let mut e2 = v[0].edge(&v[1], cx, cy);
            let mut lo: i64 = 0;
            let mut hi: i64 = px1 as i64 - px0 as i64;
            v[1].owned_run(&v[2], e0, step0, &mut lo, &mut hi);
            v[2].owned_run(&v[0], e1, step1, &mut lo, &mut hi);
            v[0].owned_run(&v[1], e2, step2, &mut lo, &mut hi);
            if lo > hi {
                continue;
            }
            e1 += lo * step1;
            e2 += lo * step2;
            if simple {
                self.simple_span(
                    &v,
                    py,
                    px0 + lo as i32,
                    px0 + hi as i32,
                    e1,
                    e2,
                    step1,
                    step2,
                    area,
                    offset,
                    &flat_frag,
                    smooth,
                );
                continue;
            }
            if !self.texture_base.is_null() || fixed {
                for k in 0..4 {
                    tex_row[k] = (py as f64 + 0.5 - gradient.y) * gradient.dy[k];
                }
            }
            let mut px = px0 + lo as i32;
            while px <= px0 + hi as i32 {
                let mut background = false;

                if let Some(span) = span {
                    // The row loop draws up to a fragment the fixed sampler
                    // cannot take, which this loop then draws in double.
                    let stop = span(self, &sp, py, px, px0 + hi as i32, e1, e2);

                    e1 = e1.wrapping_add((stop - px) as i64 * step1);
                    e2 = e2.wrapping_add((stop - px) as i64 * step2);
                    px = stop;
                    if px > px0 + hi as i32 {
                        break;
                    }
                }
                let cur_px = px;
                let cur_e1 = e1;
                let cur_e2 = e2;
                px += 1;
                e1 += step1;
                e2 += step2;
                let px = cur_px;
                // Window phase precedes plane translation and fragment work.
                if !self.stipple_pass(px, py, &mut background) {
                    continue;
                }
                frag = flat_frag.clone();
                let b1 = cur_e1 as f64 / area as f64;
                let b2 = cur_e2 as f64 / area as f64;
                let z = v[0].z + b1 * (v[1].z - v[0].z) + b2 * (v[2].z - v[0].z);
                let w0 = (1.0 - b1 - b2) * v[0].inv_w;
                let w1 = b1 * v[1].inv_w;
                let w2 = b2 * v[2].inv_w;
                let sum = w0 + w1 + w2;
                let depth_w = (1.0 - b1 - b2) * v[0].depth_w + b1 * v[1].depth_w + b2 * v[2].depth_w;
                if fixed && span.is_none() {
                    let mut s = 0.0f64;
                    let mut t = 0.0f64;
                    let mut lambda = 0i32;

                    frag.fixed = self.fixed.fixed_coords(&gradient, &tex_row, px, fixed_flat, &mut s, &mut t, &mut lambda)
                        && SoftDraw::fixed_sample::<SGI_SOFT_FIXED_ANY, false>(self, s, t, lambda, &mut frag.texel);
                }
                // Only texturing reads the coordinates, only fog the distance.
                if !self.texture.is_null() && !frag.fixed {
                    for k in 0..4 {
                        frag.texcoord[k] = (w0 * v[0].texcoord[k] + w1 * v[1].texcoord[k] + w2 * v[2].texcoord[k]) / sum;
                    }
                }
                if st.fog != 0 {
                    frag.fog = (w0 * v[0].fog + w1 * v[1].fog + w2 * v[2].fog) / sum;
                }
                if st.clip_planes != 0 {
                    let mut d = [0.0f64; SGI_GL_CLIP_PLANES as usize];

                    for i in 0..SGI_GL_CLIP_PLANES as usize {
                        d[i] = (w0 * v[0].clip[i] + w1 * v[1].clip[i] + w2 * v[2].clip[i]) / sum;
                    }
                    if st.clipped(&d) {
                        continue;
                    }
                }
                if smooth {
                    for k in 0..4 {
                        frag.rgba[k] = (w0 * v[0].rgba[k] + w1 * v[1].rgba[k] + w2 * v[2].rgba[k]) / sum;
                    }
                    // An index is taken as a difference from the first vertex's,
                    // so a constant one stays exact before its integer part is taken.
                    frag.index = v[0].index + (w1 * (v[1].index - v[0].index) + w2 * (v[2].index - v[0].index)) / sum;
                }
                if !self.texture_base.is_null() && !frag.fixed {
                    let mut value = [0.0f64; 4];
                    let mut out = [0.0f64; 5];

                    for k in 0..4 {
                        value[k] = gradient.value[k] + (px as f64 + 0.5 - gradient.x) * gradient.dx[k] + tex_row[k];
                    }
                    if flat_q {
                        let d = value[2];

                        frag.texture_rho = if flat_ok
                            && (value[0] / d).is_finite()
                            && (value[1] / d).is_finite()
                            && (tex_nr == 0.0 || (value[3] / d).is_finite())
                        {
                            flat_rho
                        } else {
                            f64::NAN
                        };
                    } else {
                        frag.texture_rho = if soft_texture_derivatives(&value, &gradient.dx, &gradient.dy, tex_ns, tex_nt, tex_nr, &mut out) {
                            out[4]
                        } else {
                            f64::NAN
                        };
                    }
                }
                if background {
                    self.stipple_background(&mut frag);
                }
                self.fragment(px, py, z, offset, depth_w, &frag);
            }
        }
    }

    /// Transform, clip and rasterize one triangle. Clipped to the view volume
    /// in clip coordinates, what is left is drawn as a fan from its first
    /// vertex, each flat shaded from the triangle's provoking vertex.
    pub fn clip_triangle(&self, va: &[f32], vb: &[f32], vc: &[f32], poly: &mut [VolumeVertex; SGI_VOLUME_TRIANGLE_MAX], n: &mut u32) -> bool {
        let verts: [&[f32]; 3] = [va, vb, vc];
        let mut in_: [VolumeVertex; 3] = Default::default();

        for i in 0..3 {
            self.st.volume_from_draw_vertex(verts[i], &mut in_[i]);
        }
        *n = volume_clip_triangle(&in_, poly);
        for i in 0..*n as usize {
            if !self.texture_vertex_valid(&poly[i]) {
                return false;
            }
        }
        if !self.texture_base.is_null() {
            let mut i = 1usize;
            while i + 1 < *n as usize {
                let vertices: [&VolumeVertex; 3] = [&poly[0], &poly[i], &poly[i + 1]];
                let mut projected: [SoftVertex; 3] = Default::default();

                for j in 0..3 {
                    if !self.project(vertices[j], &mut projected[j]) {
                        return false;
                    }
                }
                if !self.texture_coefficients(&projected) {
                    return false;
                }
                i += 1;
            }
        }
        true
    }

    /// A quad or polygon validates all its clipped pieces before any are drawn.
    pub fn texture_triangle_valid(&self, a: &[f32], b: &[f32], c: &[f32]) -> bool {
        let mut poly: [VolumeVertex; SGI_VOLUME_TRIANGLE_MAX] = Default::default();
        let mut n = 0u32;

        self.clip_triangle(a, b, c, &mut poly, &mut n)
    }

    pub fn triangle(&self, va: &[f32], vb: &[f32], vc: &[f32], provoking: &[f32]) {
        let mut poly: [VolumeVertex; SGI_VOLUME_TRIANGLE_MAX] = Default::default();
        let mut n = 0u32;
        let (mut x0, mut y0, mut x1, mut y1) = (0i32, 0i32, 0i32, 0i32);

        if !self.bounds(&mut x0, &mut y0, &mut x1, &mut y1) || !self.clip_triangle(va, vb, vc, &mut poly, &mut n) {
            return;
        }
        let mut i = 1usize;
        while i + 1 < n as usize {
            self.clipped_triangle(&poly[0], &poly[i], &poly[i + 1], provoking);
            i += 1;
        }
    }
}

impl VolumeVertex {
    pub fn soft_boundary_edge(&self) -> bool {
        self.attr[SGI_GL_V_EDGE].to_bits() != 0
    }

    pub fn soft_boundary_set_edge(&mut self, enabled: bool) {
        self.attr[SGI_GL_V_EDGE] = f32::from_bits(enabled as u32);
    }
}

impl SoftDraw {
    pub fn boundary_plane(&self, v: &VolumeVertex, p: u32) -> f64 {
        if p < SGI_VOLUME_PLANES as u32 {
            v.plane(p as _)
        } else {
            let plane = &self.st.clip_plane[(p - SGI_VOLUME_PLANES as u32) as usize];

            plane[0] as f64 * v.eye[0] + plane[1] as f64 * v.eye[1] + plane[2] as f64 * v.eye[2] + plane[3] as f64 * v.eye[3]
        }
    }

    /// Carry outgoing flags through complete polygon clipping (2.11). Original
    /// retained edges keep their flags; the newly connecting cut edges are
    /// true. The two workspaces swap each pass; the result is left in `poly` by
    /// swapping the vectors at the end.
    pub fn boundary_clip(&self, poly: &mut Vec<VolumeVertex>, other: &mut Vec<VolumeVertex>, count: &mut usize, capacity: usize) -> bool {
        let mut src_is_poly = true;
        let mut n = *count;

        for p in 0..(SGI_VOLUME_PLANES as u32 + SGI_GL_CLIP_PLANES) {
            let mut m = 0usize;

            if p >= SGI_VOLUME_PLANES as u32 && (self.st.clip_planes & (1u32 << (p - SGI_VOLUME_PLANES as u32))) == 0 {
                continue;
            }
            let (src, dst): (&Vec<VolumeVertex>, &mut Vec<VolumeVertex>) =
                if src_is_poly { (&*poly, &mut *other) } else { (&*other, &mut *poly) };
            for i in 0..n {
                let a = &src[i];
                let b = &src[(i + 1) % n];
                let da = self.boundary_plane(a, p);
                let db = self.boundary_plane(b, p);

                if da >= 0.0 {
                    if m == capacity {
                        return false;
                    }
                    dst[m] = a.clone();
                    m += 1;
                    if !(db >= 0.0) {
                        if da == 0.0 {
                            dst[m - 1].soft_boundary_set_edge(true);
                        } else {
                            if m == capacity {
                                return false;
                            }
                            a.cut(da, b, db, &mut dst[m]);
                            dst[m].soft_boundary_set_edge(true);
                            m += 1;
                        }
                    }
                } else if db > 0.0 {
                    if m == capacity {
                        return false;
                    }
                    b.cut(db, a, da, &mut dst[m]);
                    let edge = a.soft_boundary_edge();
                    dst[m].soft_boundary_set_edge(edge);
                    m += 1;
                }
            }
            n = m;
            src_is_poly = !src_is_poly;
            if n == 0 {
                break;
            }
        }
        if !src_is_poly {
            std::mem::swap(poly, other);
        }
        *count = n;
        true
    }
}

impl GlDraw {
    pub fn soft_polygon_vertex<'a>(&self, indices: Option<&[u32]>, i: usize) -> &'a [f32] {
        let index = match indices {
            Some(ix) => ix[i] as usize,
            None => i,
        };

        unsafe { std::slice::from_raw_parts(self.vertices.0.add(index * SGI_GL_VERTEX_FLOATS), SGI_GL_VERTEX_FLOATS) }
    }
}

#[derive(Clone, Copy, Default)]
pub struct SoftAaPoint {
    pub x: f64,
    pub y: f64,
}

pub type SoftAaEdge = u32;

pub const SGI_AA_LEFT: SoftAaEdge = 0;

pub const SGI_AA_RIGHT: SoftAaEdge = 1;

pub const SGI_AA_BOTTOM: SoftAaEdge = 2;

pub const SGI_AA_TOP: SoftAaEdge = 3;

impl SoftAaPoint {
    pub fn aa_inside(&self, edge: SoftAaEdge, value: f64) -> bool {
        match edge {
            SGI_AA_LEFT => self.x >= value,
            SGI_AA_RIGHT => self.x <= value,
            SGI_AA_BOTTOM => self.y >= value,
            SGI_AA_TOP => self.y <= value,
            _ => unreachable!(),
        }
    }

    pub fn aa_intersection(&self, b: &SoftAaPoint, edge: SoftAaEdge, value: f64) -> SoftAaPoint {
        let t = if edge <= SGI_AA_RIGHT { (value - self.x) / (b.x - self.x) } else { (value - self.y) / (b.y - self.y) };

        SoftAaPoint { x: self.x + t * (b.x - self.x), y: self.y + t * (b.y - self.y) }
    }
}

/// Clip a convex window-space polygon to one side of a pixel square.
pub fn soft_aa_clip(
    src: &[SoftAaPoint],
    n: usize,
    dst: &mut [SoftAaPoint],
    capacity: usize,
    edge: SoftAaEdge,
    value: f64,
    out_n: &mut usize,
) -> bool {
    let mut m = 0usize;

    if n == 0 {
        *out_n = 0;
        return true;
    }
    let mut a = &src[n - 1];
    for i in 0..n {
        let b = &src[i];
        let ai = a.aa_inside(edge, value);
        let bi = b.aa_inside(edge, value);

        if ai != bi {
            if m == capacity {
                return false;
            }
            dst[m] = a.aa_intersection(b, edge, value);
            m += 1;
        }
        if bi {
            if m == capacity {
                return false;
            }
            dst[m] = *b;
            m += 1;
        }
        a = b;
    }
    *out_n = m;
    true
}

impl SoftVertex {
    pub fn aa_edge(&self, b: &SoftVertex, x: f64, y: f64) -> f64 {
        (b.x.wrapping_sub(self.x)) as f64 * (y - self.y as f64) - (b.y.wrapping_sub(self.y)) as f64 * (x - self.x as f64)
    }
}

/// A convex polygon contains the square when it contains all four corners.
pub fn soft_aa_contains_pixel(v: &[SoftVertex], n: usize, px: i32, py: i32, scale: f64, ccw: bool) -> bool {
    for i in 0..n {
        let a = &v[i];
        let b = &v[(i + 1) % n];
        let x = px as f64 * scale;
        let y = py as f64 * scale;

        if ccw {
            if a.aa_edge(b, x, y) < 0.0
                || a.aa_edge(b, x + scale, y) < 0.0
                || a.aa_edge(b, x, y + scale) < 0.0
                || a.aa_edge(b, x + scale, y + scale) < 0.0
            {
                return false;
            }
        } else if a.aa_edge(b, x, y) > 0.0
            || a.aa_edge(b, x + scale, y) > 0.0
            || a.aa_edge(b, x, y + scale) > 0.0
            || a.aa_edge(b, x + scale, y + scale) > 0.0
        {
            return false;
        }
    }
    true
}

impl SoftDraw {
    /// Rasterize one clipped convex polygon once. Pixel-square clipping
    /// computes geometric coverage; centroid sampling is exact for affine
    /// datums and an approximation for perspective-correct varyings.
    pub fn polygon_coverage(&self, v: &[SoftVertex], n: usize, flat_color: &[f32], flat_index: f64, offset: f64) {
        let mut flat = SoftFrag::default();
        let (mut px0, mut py0, mut px1, mut py1) = (0i32, 0i32, 0i32, 0i32);
        let sub = self.rules.subpixel_bits as i32;
        let scale = (1i32 << sub) as f64;
        let mut polygon_area = 0.0f64;

        if n < 3 || n > usize::MAX - 8 || !self.bounds(&mut px0, &mut py0, &mut px1, &mut py1) {
            return;
        }
        let capacity = n + 8;
        let mut a = vec![SoftAaPoint::default(); capacity];
        let mut b = vec![SoftAaPoint::default(); capacity];
        let mut min_x = v[0].x;
        let mut max_x = v[0].x;
        let mut min_y = v[0].y;
        let mut max_y = v[0].y;
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
        px0 = px0.max((min_x >> sub) as i32);
        px1 = px1.min((max_x >> sub) as i32);
        py0 = py0.max((min_y >> sub) as i32);
        py1 = py1.min((max_y >> sub) as i32);
        for k in 0..4 {
            flat.rgba[k] = flat_color[k] as f64;
        }
        flat.index = flat_index;

        for py in py0..=py1 {
            for px in px0..=px1 {
                let coverage: f64;
                let x: f64;
                let y: f64;
                let z: f64;
                let mut bary = [0.0f64; 3];
                let mut w = [0.0f64; 3];
                let mut fan_i = 0usize;
                let mut frag = flat.clone();
                let mut found = false;
                let mut background = false;

                if !self.stipple_pass(px, py, &mut background) {
                    continue;
                }
                if soft_aa_contains_pixel(v, n, px, py, scale, polygon_area > 0.0) {
                    coverage = 1.0;
                    x = px as f64 + 0.5;
                    y = py as f64 + 0.5;
                } else {
                    let mut count = n;
                    let mut twice_area = 0.0f64;
                    let mut cx = 0.0f64;
                    let mut cy = 0.0f64;
                    let mut src_is_a = true;

                    // Restore the polygon before clipping this pixel.
                    for i in 0..n {
                        a[i] = SoftAaPoint { x: v[i].x as f64 / scale, y: v[i].y as f64 / scale };
                    }
                    for e in SGI_AA_LEFT..=SGI_AA_TOP {
                        let bound = if e == SGI_AA_LEFT {
                            px as f64
                        } else if e == SGI_AA_RIGHT {
                            (px + 1) as f64
                        } else if e == SGI_AA_BOTTOM {
                            py as f64
                        } else {
                            (py + 1) as f64
                        };
                        let ok = if src_is_a {
                            soft_aa_clip(&a, count, &mut b, capacity, e, bound, &mut count)
                        } else {
                            soft_aa_clip(&b, count, &mut a, capacity, e, bound, &mut count)
                        };

                        if !ok {
                            crate::dlog!(LogModule::Odyssey, "sgi-gl: polygon coverage clip exceeded its bound");
                            return;
                        }
                        src_is_a = !src_is_a;
                        if count < 3 {
                            break;
                        }
                    }
                    if count < 3 {
                        continue;
                    }
                    let src = if src_is_a { &a } else { &b };
                    for i in 0..count {
                        let p = &src[i];
                        let q = &src[(i + 1) % count];
                        let pxp = p.x - px as f64;
                        let pyp = p.y - py as f64;
                        let qxp = q.x - px as f64;
                        let qyp = q.y - py as f64;
                        let cross = pxp * qyp - qxp * pyp;

                        twice_area += cross;
                        cx += (pxp + qxp) * cross;
                        cy += (pyp + qyp) * cross;
                    }
                    if !(twice_area.abs() > 0.0) {
                        continue;
                    }
                    coverage = r2_min(1.0, twice_area.abs() * 0.5);
                    x = px as f64 + cx / (3.0 * twice_area);
                    y = py as f64 + cy / (3.0 * twice_area);
                }

                // Find the fan triangle containing the overlap centroid.
                let mut i = 1usize;
                while i + 1 < n {
                    let den = v[0].aa_edge(&v[i], v[i + 1].x as f64, v[i + 1].y as f64);

                    if den == 0.0 {
                        i += 1;
                        continue;
                    }
                    bary[0] = v[i].aa_edge(&v[i + 1], x * scale, y * scale) / den;
                    bary[1] = v[i + 1].aa_edge(&v[0], x * scale, y * scale) / den;
                    bary[2] = 1.0 - bary[0] - bary[1];
                    if bary[0] >= -0.000000001 && bary[1] >= -0.000000001 && bary[2] >= -0.000000001 {
                        found = true;
                        fan_i = i;
                        break;
                    }
                    i += 1;
                }
                if !found {
                    continue;
                }
                {
                    let i = fan_i;

                    w[0] = bary[0] * v[0].inv_w;
                    w[1] = bary[1] * v[i].inv_w;
                    w[2] = bary[2] * v[i + 1].inv_w;
                    let sum = w[0] + w[1] + w[2];
                    let depth_w = bary[0] * v[0].depth_w + bary[1] * v[i].depth_w + bary[2] * v[i + 1].depth_w;

                    if !(sum > 0.0) || !sum.is_finite() {
                        continue;
                    }
                    let fan = [v[0].clone(), v[i].clone(), v[i + 1].clone()];

                    z = bary[0] * v[0].z + bary[1] * v[i].z + bary[2] * v[i + 1].z;
                    for k in 0..4 {
                        frag.texcoord[k] =
                            (w[0] * v[0].texcoord[k] + w[1] * v[i].texcoord[k] + w[2] * v[i + 1].texcoord[k]) / sum;
                        if self.st.shade_model != SGI_GL_FLAT {
                            frag.rgba[k] = (w[0] * v[0].rgba[k] + w[1] * v[i].rgba[k] + w[2] * v[i + 1].rgba[k]) / sum;
                        }
                    }
                    frag.fog = (w[0] * v[0].fog + w[1] * v[i].fog + w[2] * v[i + 1].fog) / sum;
                    if self.st.shade_model != SGI_GL_FLAT {
                        frag.index = v[0].index
                            + (w[1] * (v[i].index - v[0].index) + w[2] * (v[i + 1].index - v[0].index)) / sum;
                    }
                    if self.st.clip_planes != 0 {
                        let mut d = [0.0f64; SGI_GL_CLIP_PLANES as usize];

                        for k in 0..SGI_GL_CLIP_PLANES as usize {
                            d[k] = (w[0] * fan[0].clip[k] + w[1] * fan[1].clip[k] + w[2] * fan[2].clip[k]) / sum;
                        }
                        if self.st.clipped(&d) {
                            continue;
                        }
                    }
                    if !self.texture_base.is_null() {
                        let mut gradient = SoftTexPlane::default();
                        let mut value = [0.0f64; 4];
                        let mut out = [0.0f64; 5];

                        if !soft_texture_plane(&fan, sub as u32, self.texture.volume, self.texture.one_dimensional, &mut gradient) {
                            continue;
                        }
                        for k in 0..4 {
                            value[k] = gradient.value[k]
                                + (px as f64 + 0.5 - gradient.x) * gradient.dx[k]
                                + (py as f64 + 0.5 - gradient.y) * gradient.dy[k];
                        }
                        frag.texture_rho = if soft_texture_derivatives(
                            &value,
                            &gradient.dx,
                            &gradient.dy,
                            self.texture_base.width as f64 * soft_texture_lod_scale(self.texture.lod_scale_s),
                            if self.texture.one_dimensional {
                                0.0
                            } else {
                                self.texture_base.height as f64 * soft_texture_lod_scale(self.texture.lod_scale_t)
                            },
                            if self.texture.volume {
                                self.texture_base.depth as f64 * soft_texture_lod_scale(self.texture.lod_scale_r)
                            } else {
                                0.0
                            },
                            &mut out,
                        ) {
                            out[4]
                        } else {
                            f64::NAN
                        };
                    }
                    if background {
                        self.stipple_background(&mut frag);
                    }
                    self.fragment_coverage(px, py, z, offset, depth_w, &frag, coverage, true);
                }
            }
        }
    }

    pub fn polygon_smooth(&self, poly: &[VolumeVertex], n: usize, provoking: &[f32], front: bool, slope: f64) {
        let mut v: Vec<SoftVertex> = vec![SoftVertex::default(); n];
        let mut offset = 0.0f64;
        let mut flat_color = &provoking[SGI_GL_V_COLOR..SGI_GL_V_COLOR + 4];
        let mut flat_index = provoking[SGI_GL_V_INDEX];

        if !front && self.back_colors {
            flat_color = &provoking[SGI_GL_V_BACK_COLOR..SGI_GL_V_BACK_COLOR + 4];
            flat_index = provoking[SGI_GL_V_BACK_INDEX];
        }
        for i in 0..n {
            if !self.project(&poly[i], &mut v[i]) {
                return;
            }
            if !front && self.back_colors {
                let back = &poly[i].attr[SGI_GL_V_BACK_COLOR..SGI_GL_V_BACK_COLOR + 4];

                for k in 0..4 {
                    v[i].rgba[k] = r2_min(r2_max(back[k], 0.0f32), 1.0f32) as f64;
                }
                v[i].index = poly[i].attr[SGI_GL_V_BACK_INDEX] as f64;
            }
        }
        if !self.texture_base.is_null() {
            let mut i = 1usize;
            while i + 1 < n {
                let fan = [v[0].clone(), v[i].clone(), v[i + 1].clone()];

                if !self.texture_coefficients(&fan) {
                    return;
                }
                i += 1;
            }
        }
        if self.st.polygon_offset & SGI_GL_OFFSET_FILL != 0 {
            offset = slope * self.st.offset_factor as f64
                + (self.st.offset_units / self.rules.depth_max() as f32) as f64
                + self.st.offset_bias as f64;
        }
        self.polygon_coverage(&v, n, flat_color, flat_index as f64, offset);
    }

    /// Replay the original fan, including its validation and fragment ordering.
    pub fn polygon_fill(&self, d: &GlDraw, indices: Option<&[u32]>, n: usize, provoking: &[f32]) {
        let v0 = d.soft_polygon_vertex(indices, 0);
        let prevalidate = !self.texture.is_null() && !(d.mode == SGI_GL_POLYGON && d.fill_by_triangle);

        if prevalidate {
            let mut i = 1usize;
            while i + 1 < n {
                if !self.texture_triangle_valid(
                    v0,
                    d.soft_polygon_vertex(indices, i),
                    d.soft_polygon_vertex(indices, i + 1),
                ) {
                    return;
                }
                i += 1;
            }
        }
        let mut i = 1usize;
        while i + 1 < n {
            self.triangle(
                v0,
                d.soft_polygon_vertex(indices, i),
                d.soft_polygon_vertex(indices, i + 1),
                provoking,
            );
            i += 1;
        }
    }

    /// Workspace bounds are internal representations, not hardware maxima.
    pub fn polygon(&self, d: &GlDraw, indices: Option<&[u32]>, count: usize, provoking: &[f32], force_edges: bool) -> GlDrawResult {
        let mut positions = self.clone();
        let mut coverage = self.clone();
        let mut fan: [SoftVertex; 3] = Default::default();
        let mut first = SoftVertex::default();
        let mut previous: SoftVertex;
        let mut current = SoftVertex::default();
        let mut stipple = SoftLineStipple {
            enabled: d.line_stipple_enabled,
            repeat: r2_max(1, r2_min(d.line_stipple_repeat, 256)),
            pattern: d.line_stipple_pattern,
            ..Default::default()
        };
        let mut flat = [0.0f32; SGI_GL_VERTEX_FLOATS];
        let mut n = count;
        let mut area = 0.0f64;
        let mut slope = 0.0f64;
        let mut offset = 0.0f64;
        let front: bool;
        let mode: u32;

        if count < 3 {
            return SGI_GL_DRAW_OK;
        }
        if count > usize::MAX - SGI_VOLUME_PLANES as usize - SGI_GL_CLIP_PLANES as usize {
            return SGI_GL_DRAW_OUT_OF_MEMORY;
        }
        let capacity = count + SGI_VOLUME_PLANES as usize + SGI_GL_CLIP_PLANES as usize;
        let mut poly: Vec<VolumeVertex> = vec![VolumeVertex::default(); capacity];
        let mut other: Vec<VolumeVertex> = vec![VolumeVertex::default(); capacity];
        // Facing must not prevalidate texture Q and erase a valid FILL prefix.
        positions.texture = P::null();
        for i in 0..count {
            self.st.volume_from_draw_vertex(d.soft_polygon_vertex(indices, i), &mut poly[i]);
            for k in 0..4 {
                if !poly[i].clip[k].is_finite() || !poly[i].eye[k].is_finite() {
                    crate::dlog!(LogModule::Odyssey, "sgi-gl: invalid polygon position dropped");
                    return SGI_GL_DRAW_OK;
                }
            }
            if force_edges {
                poly[i].soft_boundary_set_edge(true);
            }
        }
        if !self.boundary_clip(&mut poly, &mut other, &mut n, capacity) {
            crate::dlog!(LogModule::Odyssey, "sgi-gl: polygon exceeds convex clipping bound");
            return SGI_GL_DRAW_OK;
        }
        if n == 0 || !positions.project(&poly[0], &mut first) {
            return SGI_GL_DRAW_OK;
        }
        previous = first.clone();
        fan[0] = first.clone();
        for i in 1..n {
            if !positions.project(&poly[i], &mut current) {
                return SGI_GL_DRAW_OK;
            }
            if i >= 2 {
                area += first.edge(&previous, current.x, current.y) as f64;
                fan[1] = previous.clone();
                fan[2] = current.clone();
                slope = r2_max(slope, self.slope(&fan));
            }
            previous = current.clone();
        }
        if self.st.window_matrix_given != 0 && self.st.window_rows_down != 0 {
            area = -area;
        }
        if self.st.front_face == SGI_GL_CW {
            area = -area;
        }
        front = area > 0.0; // Zero remains back-facing through both reversals.
        if self.st.cull_face != 0
            && (self.st.cull_face_mode == SGI_GL_FRONT_AND_BACK
                || self.st.cull_face_mode == (if front { SGI_GL_FRONT } else { SGI_GL_BACK }))
        {
            return SGI_GL_DRAW_OK;
        }
        mode = unsafe { *d.polygon_modes.0.add(if front { 0 } else { 1 }) };
        if mode == SGI_GL_FILL {
            if self.polygon_smooth {
                self.polygon_smooth(&poly, n, provoking, front, slope);
                return SGI_GL_DRAW_OK;
            }
            self.polygon_fill(d, indices, count, provoking);
            return SGI_GL_DRAW_OK;
        }
        if mode != SGI_GL_LINE && mode != SGI_GL_POINT {
            crate::dlog!(LogModule::Odyssey, "sgi-gl: invalid polygon mode {:#x} ignored", mode);
            return SGI_GL_DRAW_OK;
        }
        if (mode == SGI_GL_LINE && !soft_dimension(d.line_width, &mut coverage.line_width))
            || (mode == SGI_GL_POINT && !soft_dimension(d.point_size, &mut coverage.point_size))
        {
            return SGI_GL_DRAW_OK;
        }
        flat.copy_from_slice(&provoking[..SGI_GL_VERTEX_FLOATS]);
        if !front && self.back_colors {
            flat[SGI_GL_V_COLOR..SGI_GL_V_COLOR + 4].copy_from_slice(&provoking[SGI_GL_V_BACK_COLOR..SGI_GL_V_BACK_COLOR + 4]);
            for i in 0..n {
                poly[i].attr.copy_within(SGI_GL_V_BACK_COLOR..SGI_GL_V_BACK_COLOR + 4, SGI_GL_V_COLOR);
            }
        }
        // Validate the entire boundary before any LINE/POINT fragment or sample.
        for i in 0..n {
            if !self.project(&poly[i], &mut current) {
                return SGI_GL_DRAW_OK;
            }
            if !self.texture_base.is_null() && mode == SGI_GL_LINE {
                let mut next = SoftVertex::default();
                let mut values = [0.0f64; 4];
                let mut delta = [0.0f64; 4];
                let mut length = 0.0f64;

                if !self.project(&poly[(i + 1) % n], &mut next)
                    || !current.texture_line_gradient(
                        &next,
                        self.rules.subpixel_bits as u32,
                        self.texture.volume,
                        self.texture.one_dimensional,
                        &mut values,
                        &mut delta,
                        &mut length,
                    )
                {
                    self.texture_report(16, "invalid primitive coefficients, primitive dropped");
                    return SGI_GL_DRAW_OK;
                }
            }
        }
        let offset_enable = if mode == SGI_GL_LINE { SGI_GL_OFFSET_LINE } else { SGI_GL_OFFSET_POINT };
        if self.st.polygon_offset & offset_enable != 0 {
            offset = slope * self.st.offset_factor as f64
                + (self.st.offset_units / self.rules.depth_max() as f32) as f64
                + self.st.offset_bias as f64;
        }
        for i in 0..n {
            let mut next = SoftVertex::default();

            if !poly[i].soft_boundary_edge() {
                continue;
            }
            self.project(&poly[i], &mut current);
            if mode == SGI_GL_POINT {
                coverage.point_projected(&current, Some(&flat), offset);
            } else {
                self.project(&poly[(i + 1) % n], &mut next);
                coverage.line_projected(&current, &next, &flat, offset, Some(&mut stipple), false);
            }
        }
        SGI_GL_DRAW_OK
    }

    /// Dispatch each independently assembled polygon, preserving provoking rules.
    pub fn polygon_modes(&self, d: &GlDraw) -> GlDrawResult {
        let fill_modes: [u32; 2] = [SGI_GL_FILL, SGI_GL_FILL];
        let mut indices = [0u32; 4];
        let n = d.count;

        if d.polygon_modes.is_null() {
            let mut fill = *d;

            fill.polygon_modes = P(fill_modes.as_ptr() as *mut u32);
            return self.polygon_modes(&fill);
        }

        if d.mode == SGI_GL_POLYGON {
            return self.polygon(d, None, n as usize, d.soft_polygon_vertex(None, 0), false);
        }
        let stride: u32 = if d.mode == SGI_GL_TRIANGLES {
            3
        } else if d.mode == SGI_GL_QUADS {
            4
        } else if d.mode == SGI_GL_QUAD_STRIP {
            2
        } else {
            1
        };
        let mut i = 0u32;
        while i < n {
            let count: u32 = if d.mode == SGI_GL_QUADS || d.mode == SGI_GL_QUAD_STRIP { 4 } else { 3 };
            let provoking = i + count - 1;
            let force = stride < 3;

            if n - i < count {
                break;
            }
            for k in 0..count {
                indices[k as usize] = i + k;
            }
            if d.mode == SGI_GL_TRIANGLE_STRIP && (i & 1) != 0 {
                indices[0] = i + 1;
                indices[1] = i;
            } else if d.mode == SGI_GL_TRIANGLE_FAN {
                indices[0] = 0;
            } else if d.mode == SGI_GL_QUAD_STRIP {
                indices[2] = i + 3;
                indices[3] = i + 2;
            }
            let result = self.polygon(
                d,
                Some(&indices[..count as usize]),
                count as usize,
                d.soft_polygon_vertex(None, provoking as usize),
                force,
            );
            if result != SGI_GL_DRAW_OK {
                return result;
            }
            i += stride;
        }
        SGI_GL_DRAW_OK
    }

    /// The filled primitives of a draw, from SGI_GL_TRIANGLES on, in order.
    pub fn draw_triangles(&self, d: &GlDraw) {
        let n = d.count as usize;
        let v = |i: usize| -> &[f32] { d.soft_polygon_vertex(None, i) };

        match d.mode {
            SGI_GL_TRIANGLES => {
                let mut i = 0usize;
                while i + 2 < n {
                    self.triangle(v(i), v(i + 1), v(i + 2), v(i + 2));
                    i += 3;
                }
            }
            SGI_GL_TRIANGLE_STRIP => {
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
            SGI_GL_TRIANGLE_FAN | SGI_GL_POLYGON => {
                if !self.texture.is_null() && d.mode == SGI_GL_POLYGON && !d.fill_by_triangle {
                    let mut i = 0usize;
                    while i + 2 < n {
                        if !self.texture_triangle_valid(v(0), v(i + 1), v(i + 2)) {
                            return;
                        }
                        i += 1;
                    }
                }
                let mut i = 0usize;
                while i + 2 < n {
                    self.triangle(v(0), v(i + 1), v(i + 2), if d.mode == SGI_GL_POLYGON { v(0) } else { v(i + 2) });
                    i += 1;
                }
            }
            SGI_GL_QUADS => {
                let mut i = 0usize;
                while i + 3 < n {
                    if !self.texture.is_null()
                        && (!self.texture_triangle_valid(v(i), v(i + 1), v(i + 2))
                            || !self.texture_triangle_valid(v(i), v(i + 2), v(i + 3)))
                    {
                        i += 4;
                        continue;
                    }
                    self.triangle(v(i), v(i + 1), v(i + 2), v(i + 3));
                    self.triangle(v(i), v(i + 2), v(i + 3), v(i + 3));
                    i += 4;
                }
            }
            SGI_GL_QUAD_STRIP => {
                // Vertices 2i - 1, 2i, 2i + 2 and 2i + 1 in order (2.6.1).
                let mut i = 0usize;
                while i + 3 < n {
                    if !self.texture.is_null()
                        && (!self.texture_triangle_valid(v(i), v(i + 1), v(i + 3))
                            || !self.texture_triangle_valid(v(i), v(i + 3), v(i + 2)))
                    {
                        i += 2;
                        continue;
                    }
                    self.triangle(v(i), v(i + 1), v(i + 3), v(i + 3));
                    self.triangle(v(i), v(i + 3), v(i + 2), v(i + 3));
                    i += 2;
                }
            }
            _ => unreachable!(),
        }
    }

    /// Whether every vertex of a draw has the texture coordinates its image
    /// uses: a draw with one that has not drops that primitive unread, so it
    /// reads its texels as it reaches them rather than all of them first.
    pub fn fixed_coordinates_valid(&self, d: &GlDraw) -> bool {
        for i in 0..d.count as usize {
            let mut v = VolumeVertex::default();

            self.st.volume_from_draw_vertex(d.soft_polygon_vertex(None, i), &mut v);
            if !self.texture.soft_texcoord_valid(self.texture_base.as_opt().map(|b| &*b), &v) {
                return false;
            }
        }
        true
    }
}
