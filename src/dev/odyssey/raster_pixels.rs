//! Pixel work that is not a GL primitive: a draw's pixel estimate, clears
//! and the X server's raster ops.
//!
//! Ported from hw/display/sgi/rasterizer.c in jamesbraid/qemu-sgi-origami.

use super::*;

impl SoftDraw<'_> {
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
    pub t: &'a RasterTarget<'a>,
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
    pub rgba_reported: std::sync::atomic::AtomicBool,
}

impl SoftClear<'_> {
    /// The rows of a clear that are band's of bands (SoftDraw::row_in_band()),
    /// stored to px.
    pub fn rows(&self, px: &dyn Planes, band: u32, bands: u32) {
        let rules = self.rules;
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
                    if !rules.rgba_load(px, &op.targets, self.ntargets, x, y, &mut dest, &self.rgba_reported) {
                        continue;
                    }
                }
                if self.depth {
                    px.store_depth(x, y, self.depth_value);
                }
                if self.stencil {
                    let smask = op.stencil_mask & self.smax;

                    px.store_stencil(x, y, (px.load_stencil(x, y) & !smask & self.smax) | (op.clear_stencil & smask));
                }
                let mut i = 0u32;
                while self.colour && i < self.ntargets {
                    let target = op.targets[i as usize];

                    if op.index_bits != 0 {
                        px.store_index(target, x, y, op.clear_index, op.index_mask);
                    } else if self.rgba {
                        let mut out = self.clear;

                        for k in 0..4 {
                            if self.rgba_mask & (1u32 << k) == 0 {
                                out.component[k] = dest[i as usize].component[k];
                            }
                        }
                        px.store_rgba(target, x, y, &out);
                    } else if self.mask == rules.colour_bits() {
                        px.store_colour(target, x, y, self.rgb);
                    } else {
                        px.store_colour(target, x, y, (self.rgb & self.mask) | (px.load_colour(target, x, y) & !self.mask));
                    }
                    i += 1;
                }
            }
        }
    }
}

impl RasterRules {
    /// CLEAR writes the clear colour under the colour mask, or in colour-index
    /// mode the clear index under the index mask, to the op's targets, the
    /// clear depth to the depth plane and the clear stencil value under its
    /// mask to the stencil buffer, where the planes have one, at every pixel of
    /// its rectangle its clip list keeps. The other kinds are not executed
    /// here.
    pub fn raster(&self, t: &RasterTarget, planes: &mut dyn PlaneStore, op: &RasterOp) {
        let mut colour = op.clear_buffers & SGI_GL_COLOR_BUFFER_BIT != 0;
        let depth = op.clear_buffers & SGI_GL_DEPTH_BUFFER_BIT != 0;
        let mut stencil = op.clear_buffers & SGI_GL_STENCIL_BUFFER_BIT != 0;
        let mut rgb = 0u32;
        let mut smax = 0u32;
        let mut mask = self.colour_bits();
        let rgba_mask = self.rgba_mask(op.clear_colour_mask);
        let rgba = op.index_bits == 0 && t.store_rgba;
        let rgba_read = colour
            && op.ntargets != 0
            && rgba_mask != 0
            && rgba_mask != self.rgba_mask(SGI_GL_COLOUR_MASK_ALL);
        let ntargets = r2_min(op.ntargets, SGI_GL_MAX_TARGETS);
        let mut clear = RasterColour::default();

        if !self.check_rules() {
            return;
        }
        // After the check: the depth maximum shifts by 32 - depth_bits.
        let depth_value = op.clear_depth24 & self.depth_max();
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
            if !rgba && mask != self.colour_bits() && mask != 0 && !t.load_colour {
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
            rgba_reported: Default::default(),
        };
        let bands = soft_bands();
        if !t.concurrent_pixels
            || bands < 2
            || op.y1 < op.y0
            || op.x1 < op.x0
            || ((op.x1 - op.x0 + 1) as u64) * ((op.y1 - op.y0 + 1) as u64) < t.soft_band_min_pixels()
            || !soft_bands_run_split(planes, bands, &|band, store| rows.rows(&*store.planes(), band, bands))
        {
            rows.rows(&*planes.planes(), 0, 1);
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
