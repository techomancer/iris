//! Pixel work that is not a GL primitive: pixel rectangles, bitmaps,
//! clears and the X server's raster ops, each stored through the same
//! fragment path as a triangle's pixels.
//!
//! Ported from hw/display/sgi/rasterizer.c in jamesbraid/qemu-sgi-origami.

use super::*;

impl SoftDraw {
    /// About how many pixels a draw's filled primitives cover: the area of each
    /// one with its vertices moved into the pixels the window owns, summed
    /// until it reaches limit. A vertex with no window position counts as
    /// reaching it.
    pub fn draw_pixels(&self, d: &GlDraw, limit: u64) -> u64 {
        let n = d.count as usize;
        let mut sum = 0u64;
        let (mut x0, mut y0, mut x1, mut y1) = (0i32, 0i32, 0i32, 0i32);

        if !self.bounds(&mut x0, &mut y0, &mut x1, &mut y1) || n < 3 {
            return 0;
        }
        let mut xy = vec![0.0f64; 2 * n];
        for i in 0..n {
            let mut cv = VolumeVertex::default();
            let mut win = [0.0f64; 3];

            self.st.volume_from_draw_vertex(d.soft_polygon_vertex(None, i), &mut cv);
            if !(cv.clip[3] > 0.0) {
                return limit;
            }
            self.st.clip_to_window(&cv.clip, &mut win);
            xy[2 * i] = win[0];
            xy[2 * i + 1] = win[1];
        }
        // Each primitive's own vertices; a fan's first is left out.
        let step: usize = if d.mode == SGI_GL_TRIANGLES {
            3
        } else if d.mode == SGI_GL_QUADS {
            4
        } else if d.mode == SGI_GL_QUAD_STRIP {
            2
        } else {
            1
        };
        let size: usize = if d.mode == SGI_GL_QUADS || d.mode == SGI_GL_QUAD_STRIP { 4 } else { 3 };
        let mut i = 0usize;
        while i + size <= n && sum < limit {
            let mut px = [0.0f64; 4];
            let mut py = [0.0f64; 4];
            let mut twice = 0.0f64;

            for k in 0..size {
                px[k] = r2_min(r2_max(xy[2 * (i + k)], x0 as f64), (x1 + 1) as f64);
                py[k] = r2_min(r2_max(xy[2 * (i + k) + 1], y0 as f64), (y1 + 1) as f64);
            }
            for k in 0..size {
                let next = (k + 1) % size;

                twice += px[k] * py[next] - px[next] * py[k];
            }
            sum = sum.wrapping_add((twice.abs() / 2.0) as u64);
            i += step;
        }
        sum
    }
}

pub fn r2_clamp(val: f64, low: f64, high: f64) -> f64 {
    if val > high { high } else if val < low { low } else { val }
}

impl RasterRules {
    /// A run of a bitmap's image (OpenGL 1.2.1, 3.7): a fragment for each set
    /// bit, at the image's corner plus the bit's column and row, bounded by the
    /// window the draw state's drawable owns and not by the viewport.
    pub fn bitmap(&self, t: &RasterTarget, op: &RasterOp) {
        let b = &op.bitmap;
        let mut r = SoftDraw { rules: P::from_ref(self), t: P::from_ref(t), st: b.state, ..Default::default() };
        let row_bytes: u32 = (b.width + 7) / 8;
        let mut rgba_reported = false;
        let mut texture_reports = 0u32;
        let mut frag = SoftFrag::default();

        frag.rgba = [b.rgba[0] as f64, b.rgba[1] as f64, b.rgba[2] as f64, b.rgba[3] as f64];
        frag.texcoord = [b.texcoord[0] as f64, b.texcoord[1] as f64, b.texcoord[2] as f64, b.texcoord[3] as f64];
        frag.index = b.index as f64;
        frag.fog = b.distance as f64;

        r.rgba_reported = P::new(&mut rgba_reported);
        r.texture_reports = P::new(&mut texture_reports);
        if !r.texture_setup(true) || !b.state.window(&mut r.win) || !t.soft_check_index(b.state.index_bits) {
            return;
        }
        if !r.texture.is_null() {
            r.texels = r.texture.soft_texel_cache_begin();
        }
        r.ntargets = r2_min(op.ntargets, SGI_GL_MAX_TARGETS);
        r.targets = op.targets;
        if !r.setup_ops() {
            return;
        }
        let mut i = 0u32;
        while i < b.count && row_bytes != 0 {
            let byte = b.first.wrapping_add(i);
            let row: i64 = (byte / row_bytes) as i64;
            let col0: i64 = ((byte % row_bytes).wrapping_mul(8)) as i64;

            if row >= b.height as i64 {
                break;
            }
            let mut bit = 0i32;
            while bit < 8 && col0 + (bit as i64) < b.width as i64 {
                let px: i64 = b.x as i64 + col0 + bit as i64;
                let py: i64 = b.y as i64 + row;
                let byte_value = unsafe { *b.bytes.0.add(i as usize) } as i32;

                bit += 1;
                let bit_now = bit - 1;
                if (byte_value & (0x80 >> bit_now)) == 0
                    || px < (r.win.x0 - r.win.ox) as i64
                    || px > (r.win.x1 - r.win.ox) as i64
                    || py < (r.win.y0 - r.win.oy) as i64
                    || py > (r.win.y1 - r.win.oy) as i64
                {
                    continue;
                }
                r.fragment_at(px as i32, py as i32, b.z as f64, &frag, 1.0, false);
            }
            i += 1;
        }
    }

    pub fn pixels(&self, t: &RasterTarget, op: &RasterOp) {
        let p = &op.pixels;
        let image = !p.rgba.is_null()
            || !p.rgba_float.is_null()
            || !p.rgba_double.is_null()
            || !p.depth.is_null()
            || !p.stencil.is_null()
            || !p.index.is_null();
        let mut r = SoftDraw { rules: P::from_ref(self), t: P::from_ref(t), st: p.state, ..Default::default() };
        let mut frag = SoftFrag::default();
        let mut rgba_reported = false;
        let mut texture_reports = 0u32;
        let mut stencil_max = 0u32;
        let mut stencil_mask = 0u32;
        let ex = p.x + p.zoom[0] as f64 * p.width as f64;
        let ey = p.y + p.zoom[1] as f64 * p.height as f64;

        frag.fog = p.distance as f64;
        frag.texcoord = [p.texcoord[0] as f64, p.texcoord[1] as f64, p.texcoord[2] as f64, p.texcoord[3] as f64];
        r.rgba_reported = P::new(&mut rgba_reported);
        r.texture_reports = P::new(&mut texture_reports);
        if (p.stencil.is_null() && !r.texture_setup(true))
            || (p.stencil.is_null() && !t.soft_check_index(p.state.index_bits))
            || p.width == 0
            || p.height == 0
            || !image
            || !p.x.is_finite()
            || !p.y.is_finite()
            || !p.zoom[0].is_finite()
            || !p.zoom[1].is_finite()
            || p.zoom[0] == 0.0
            || p.zoom[1] == 0.0
            || !p.state.window(&mut r.win)
        {
            return;
        }
        if !r.texture.is_null() {
            r.texels = r.texture.soft_texel_cache_begin();
        }
        r.ntargets = r2_min(op.ntargets, SGI_GL_MAX_TARGETS);
        r.targets = op.targets;
        if !p.stencil.is_null() {
            if !self.stencil_max(t, &mut stencil_max) {
                return;
            }
            stencil_mask = p.state.stencil_write_mask & stencil_max;
            if stencil_mask == 0 {
                return;
            }
        } else if !r.setup_ops() {
            return;
        }
        // Include lower/left centres and exclude upper/right centres (3.6.4).
        let bx0 = r2_max((r2_min(p.x, ex) - 0.5).ceil(), (r.win.x0 as i64 - r.win.ox as i64) as f64);
        let by0 = r2_max((r2_min(p.y, ey) - 0.5).ceil(), (r.win.y0 as i64 - r.win.oy as i64) as f64);
        let bx1 = r2_min((r2_max(p.x, ex) - 0.5).ceil() - 1.0, (r.win.x1 as i64 - r.win.ox as i64) as f64);
        let by1 = r2_min((r2_max(p.y, ey) - 0.5).ceil() - 1.0, (r.win.y1 as i64 - r.win.oy as i64) as f64);
        if bx0 > bx1 || by0 > by1 {
            return;
        }
        // Visible bounds keep conversion and work independent of zoom size.
        let x0 = bx0 as i64;
        let y0 = by0 as i64;
        let x1 = bx1 as i64;
        let y1 = by1 as i64;
        let mut y = y0;
        while y <= y1 {
            let cy = y;
            y += 1;
            let qy = (cy as f64 + 0.5 - p.y) / p.zoom[1] as f64;
            let sy = if p.zoom[1] > 0.0 { qy.floor() } else { qy.ceil() - 1.0 };

            if !(sy >= 0.0 && sy < p.height as f64) {
                continue;
            }
            let mut x = x0;
            while x <= x1 {
                let cx = x;
                x += 1;
                let qx = (cx as f64 + 0.5 - p.x) / p.zoom[0] as f64;
                let sx = if p.zoom[0] > 0.0 { qx.floor() } else { qx.ceil() - 1.0 };

                if !(sx >= 0.0 && sx < p.width as f64) {
                    continue;
                }
                let at: u64 = (sy as u64).wrapping_mul(p.width as u64).wrapping_add(sx as u32 as u64);
                if !p.stencil.is_null() {
                    let px = r.win.ox + cx as i32;
                    let py = r.win.oy + cy as i32;

                    if r.owned_at(cx as i32, cy as i32) {
                        let old = (t.load_stencil.unwrap())(t.opaque, px, py);
                        let src = unsafe { *p.stencil.0.add(at as usize) };

                        (t.store_stencil.unwrap())(t.opaque, px, py, ((old & !stencil_mask) | (src & stencil_mask)) & stencil_max);
                    }
                    continue;
                }
                frag.index = if !p.index.is_null() {
                    (unsafe { *p.index.0.add(at as usize) }) as f64
                } else {
                    p.colour_index as f64
                };
                let rgba: u32 = if !p.rgba.is_null() { unsafe { *p.rgba.0.add(at as usize) } } else { 0 };
                for k in 0..4 {
                    let component: f64 = if !p.depth.is_null() {
                        p.colour[k] as f64
                    } else if !p.rgba_double.is_null() {
                        unsafe { (*p.rgba_double.0.add(at as usize))[k] }
                    } else if !p.rgba_float.is_null() {
                        unsafe { (*p.rgba_float.0.add(at as usize))[k] as f64 }
                    } else {
                        ((rgba >> (24 - 8 * k)) & 0xff) as f64 / 255.0
                    };

                    frag.rgba[k] = r2_clamp(component * p.transfer.scale[k] as f64 + p.transfer.bias[k] as f64, 0.0, 1.0);
                }
                let z: f64 = if !p.depth.is_null() {
                    r2_clamp(
                        unsafe { *p.depth.0.add(at as usize) } * p.depth_scale as f64 + p.depth_bias as f64,
                        0.0,
                        1.0,
                    )
                } else {
                    p.z as f64
                };

                r.fragment_at(cx as i32, cy as i32, z, &frag, 1.0, false);
            }
        }
    }
}

impl ClipList {
    /// Whether one rectangle of a non-empty clip list holds all of row y from
    /// x0 to x1, so that each pixel's own test would pass.
    pub fn soft_clip_covers_row(&self, x0: i32, x1: i32, y: i32) -> bool {
        for i in 0..r2_min(self.n, SGI_CLIP_MAX as u32) {
            let r = &self.r[i as usize];

            if y >= r.y0 && y <= r.y1 && x0 >= r.x0 && x1 <= r.x1 {
                return true;
            }
        }
        false
    }
}

/// A clear's decided values, which every row of it stores.
pub struct SoftClear<'a> {
    pub rules: &'a RasterRules,
    pub t: &'a RasterTarget,
    pub op: &'a RasterOp,
    pub colour: bool,
    pub depth: bool,
    pub stencil: bool,
    pub rgba: bool,
    pub rgba_read: bool,
    pub rgb: u32,
    pub smax: u32,
    pub mask: u32,
    pub rgba_mask: u32,
    pub ntargets: u32,
    pub depth_value: u32,
    pub clear: RasterColour,
    pub rgba_reported: P<bool>,
}

impl SoftClear<'_> {
    /// The rows of a clear that are band's of bands (SoftDraw::row_in_band()).
    pub fn rows(&self, band: u32, bands: u32) {
        let rules = self.rules;
        let t = self.t;
        let op = self.op;
        // Only a read fills it, and only a read's targets are read back.
        let mut dest = [RasterColour::default(); SGI_GL_MAX_TARGETS as usize];

        for y in op.y0..=op.y1 {
            if !(bands <= 1 || ((y as u32) >> SGI_SOFT_BAND_ROWS_SHIFT) % bands == band) {
                continue;
            }
            let whole_row = op.clip.n <= 1 || op.clip.soft_clip_covers_row(op.x0, op.x1, y);
            for x in op.x0..=op.x1 {
                if !whole_row && !op.clip.contains(x, y) {
                    continue;
                }

                if self.rgba && self.rgba_read {
                    dest = [RasterColour::default(); SGI_GL_MAX_TARGETS as usize];
                    if !rules.rgba_load(t, &op.targets, self.ntargets, x, y, &mut dest, self.rgba_reported.get()) {
                        continue;
                    }
                }
                if self.depth {
                    (t.store_depth.unwrap())(t.opaque, x, y, self.depth_value);
                }
                if self.stencil {
                    let smask = op.stencil_mask & self.smax;

                    (t.store_stencil.unwrap())(
                        t.opaque,
                        x,
                        y,
                        ((t.load_stencil.unwrap())(t.opaque, x, y) & !smask & self.smax) | (op.clear_stencil & smask),
                    );
                }
                let mut i = 0u32;
                while self.colour && i < self.ntargets {
                    if op.index_bits != 0 {
                        (t.store_index.unwrap())(t.opaque, op.targets[i as usize], x, y, op.clear_index, op.index_mask);
                    } else if self.rgba {
                        let mut out = self.clear;

                        for k in 0..4 {
                            if self.rgba_mask & (1u32 << k) == 0 {
                                out.component[k] = dest[i as usize].component[k];
                            }
                        }
                        (t.store_rgba.unwrap())(t.opaque, op.targets[i as usize], x, y, &out);
                    } else if self.mask == rules.colour_bits() {
                        (t.store_colour.unwrap())(t.opaque, op.targets[i as usize], x, y, self.rgb);
                    } else {
                        (t.store_colour.unwrap())(
                            t.opaque,
                            op.targets[i as usize],
                            x,
                            y,
                            (self.rgb & self.mask) | ((t.load_colour.unwrap())(t.opaque, op.targets[i as usize], x, y) & !self.mask),
                        );
                    }
                    i += 1;
                }
            }
        }
    }
}

/// SoftClear::rows() as the band pool calls it.
pub fn soft_clear_rows_band(opaque: Opaque, band: u32, bands: u32) {
    let c: &SoftClear = from_opaque(opaque);

    c.rows(band, bands);
}

impl RasterRules {
    /// CLEAR writes the clear colour under the colour mask, or in colour-index
    /// mode the clear index under the index mask, to the op's targets, the
    /// clear depth to the depth plane and the clear stencil value under its
    /// mask to the stencil buffer, where the planes have one, at every pixel of
    /// its rectangle its clip list keeps. BITMAP draws its run of a bitmap, and
    /// PIXELS draws a zoomed RGBA rectangle. The other kinds are not executed
    /// here yet.
    pub fn raster(&self, t: &RasterTarget, op: &RasterOp) {
        let mut colour = op.clear_buffers & SGI_GL_COLOR_BUFFER_BIT != 0;
        let depth = op.clear_buffers & SGI_GL_DEPTH_BUFFER_BIT != 0;
        let mut stencil = op.clear_buffers & SGI_GL_STENCIL_BUFFER_BIT != 0;
        let mut rgb = 0u32;
        let mut smax = 0u32;
        let mut mask = self.colour_bits();
        let rgba_mask = self.rgba_mask(op.clear_colour_mask);
        let rgba = op.index_bits == 0 && t.store_rgba.is_some();
        let rgba_read = colour
            && op.ntargets != 0
            && rgba_mask != 0
            && rgba_mask != self.rgba_mask(SGI_GL_COLOUR_MASK_ALL);
        let ntargets = r2_min(op.ntargets, SGI_GL_MAX_TARGETS);
        let mut rgba_reported = false;
        let mut clear = RasterColour::default();

        if !self.check_rules() {
            return;
        }
        // After the check: the depth maximum shifts by 32 - depth_bits.
        let depth_value = op.clear_depth24 & self.depth_max();
        if op.kind == SGI_RASTER_PIXELS {
            self.pixels(t, op);
            return;
        }
        if op.kind == SGI_RASTER_BITMAP {
            self.bitmap(t, op);
            return;
        }
        if op.kind != SGI_RASTER_CLEAR {
            crate::dlog!(LogModule::Odyssey, "sgi-gl: raster op of kind {} is not executed", op.kind);
            return;
        }
        if op.clip.n == 0 && op.clip.empty != 0 {
            return;
        }
        if stencil && !self.stencil_max(t, &mut smax) {
            stencil = false;
        }
        if colour && op.index_bits != 0 && !t.soft_check_index(op.index_bits) {
            colour = false;
        }
        if op.index_bits == 0 && !self.rgba_capable(t, rgba_read) {
            return;
        }
        if colour && op.index_bits == 0 {
            let c = [op.clear_rgba[0] as f64, op.clear_rgba[1] as f64, op.clear_rgba[2] as f64];

            mask = self.colour_mask(op.clear_colour_mask);
            if !rgba && mask != self.colour_bits() && mask != 0 && t.load_colour.is_none() {
                crate::dlog!(LogModule::Odyssey, "sgi-gl: a partial colour mask needs planes that can be read back; cleared without it");
                mask = self.colour_bits();
            }
            colour = if rgba { rgba_mask != 0 } else { mask != 0 };
            rgb = self.rgb(&c);
            // Clears remain undithered: no dither state is submitted.
            clear = self.rgba_value(rgb, op.clear_rgba[3] as f64);
        }
        let rows = SoftClear {
            rules: self,
            t,
            op,
            colour,
            depth,
            stencil,
            rgba,
            rgba_read,
            rgb,
            smax,
            mask,
            rgba_mask,
            ntargets,
            depth_value,
            clear,
            rgba_reported: P::new(&mut rgba_reported),
        };
        let bands = soft_bands();
        if !t.concurrent_pixels
            || bands < 2
            || op.y1 < op.y0
            || op.x1 < op.x0
            || ((op.x1 - op.x0 + 1) as u64) * ((op.y1 - op.y0 + 1) as u64) < t.soft_band_min_pixels()
            || !soft_bands_run(soft_clear_rows_band, P::from_ref(&rows).opaque(), bands)
        {
            rows.rows(0, 1);
        }
    }
}

/// The X server's logic ops (GX numbers) are GL's logic-op enums less
/// GL_CLEAR; an out-of-range one keeps the source.
pub fn x_logic_op(gx: u32, source: u32, destination: u32) -> u32 {
    if gx <= 15 {
        logic_op(gx as u8, source, destination)
    } else {
        source
    }
}
