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

        let Some((x0, y0, x1, y1)) = self.bounds().filter(|_| n >= 3) else {
            return 0;
        };
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
    /// The clear's rows, stored to px. A row a single clip rectangle of the
    /// op holds whole is stored through the row of planes px lends for it
    /// (Planes::row()), where it lends one and the clear stores only what a
    /// lent row takes: RGBA colour, depth and stencil.
    pub fn rows(&self, px: &dyn Planes) {
        let op = self.op;
        let lend = !self.colour || (op.index_bits == 0 && self.rgba);
        // A lent row that reads nothing back stores the same words at every
        // pixel: the loop for the planes the clear names.
        let stores: Option<fn(&SoftClear, &GlPlaneRow, i32)> = match (self.rgba_read, self.depth, self.stencil, self.colour) {
            (true, ..) => None,
            (false, true, false, true) => Some(clear_lent_row::<true, false, true>),
            (false, true, false, false) => Some(clear_lent_row::<true, false, false>),
            (false, false, false, true) => Some(clear_lent_row::<false, false, true>),
            (false, d, true, c) => Some(if d && c {
                clear_lent_row::<true, true, true>
            } else if d {
                clear_lent_row::<true, true, false>
            } else if c {
                clear_lent_row::<false, true, true>
            } else {
                clear_lent_row::<false, true, false>
            }),
            (false, false, false, false) => None,
        };

        for y in op.y0..=op.y1 {
            let whole_row = op.clip.n <= 1 || op.clip.soft_clip_covers_row(op.x0, op.x1, y);

            if lend && whole_row && op.x0 <= op.x1 {
                if let Some(row) = px.row(y, op.x0, op.x1) {
                    match stores {
                        Some(stores) => stores(self, &row, y),
                        None => self.row(&row, y, true),
                    }
                    continue;
                }
            }
            self.row(px, y, whole_row);
        }
    }

    /// Row y of the clear, stored to px; with whole_row unset each pixel
    /// tests the op's clip list.
    #[inline(always)]
    fn row<P: Planes + ?Sized>(&self, px: &P, y: i32, whole_row: bool) {
        let rules = self.rules;
        let op = self.op;
        // Only a read fills it, and only a read's targets are read back.
        let mut dest = [RasterColour::default(); SGI_GL_MAX_TARGETS as usize];

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

/// Row y of clear c, which reads nothing back, through the row of planes px
/// lends: SoftClear::row()'s stores for the planes DEPTH, STENCIL and COLOUR
/// name. The colour is the clear colour whole: only a partial mask reads
/// the destination.
fn clear_lent_row<const DEPTH: bool, const STENCIL: bool, const COLOUR: bool>(c: &SoftClear, px: &GlPlaneRow, y: i32) {
    let op = c.op;
    let smask = op.stencil_mask & c.smax;

    // The planes are independent, so each is stored across the row in turn:
    // depth and colour as runs, which fill a slice where the row's id test
    // passes every pixel, and stencil, which reads, a pixel at a time.
    if DEPTH {
        px.fill_depth(op.x0, op.x1, c.depth_value);
    }
    if STENCIL {
        for x in op.x0..=op.x1 {
            px.store_stencil(x, y, (px.load_stencil(x, y) & !smask & c.smax) | (op.clear_stencil & smask));
        }
    }
    if COLOUR {
        for &target in &op.targets[..c.ntargets as usize] {
            px.fill_rgba(target, op.x0, op.x1, &c.clear);
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
        if stencil {
            match self.stencil_max(t) {
                Some(max) => smax = max,
                None => stencil = false,
            }
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
        rows.rows(&*planes.planes());
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dev::odyssey::raster_bench::*;

    /// The planes of inner, but lending no rows: every pixel through the
    /// planes' own methods.
    struct Unlent<'a>(Box<dyn Planes + 'a>);

    impl Planes for Unlent<'_> {
        fn store_colour(&self, target: ColourTarget, x: i32, y: i32, colour: u32) {
            self.0.store_colour(target, x, y, colour)
        }
        fn load_colour(&self, target: ColourTarget, x: i32, y: i32) -> u32 {
            self.0.load_colour(target, x, y)
        }
        fn store_index(&self, target: ColourTarget, x: i32, y: i32, index: u32, mask: u32) {
            self.0.store_index(target, x, y, index, mask)
        }
        fn store_rgba(&self, target: ColourTarget, x: i32, y: i32, colour: &RasterColour) {
            self.0.store_rgba(target, x, y, colour)
        }
        fn load_rgba(&self, target: ColourTarget, x: i32, y: i32) -> Option<RasterColour> {
            self.0.load_rgba(target, x, y)
        }
        fn load_depth(&self, x: i32, y: i32) -> u32 {
            self.0.load_depth(x, y)
        }
        fn store_depth(&self, x: i32, y: i32, depth: u32) {
            self.0.store_depth(x, y, depth)
        }
        fn load_stencil(&self, x: i32, y: i32) -> u32 {
            self.0.load_stencil(x, y)
        }
        fn store_stencil(&self, x: i32, y: i32, stencil: u32) {
            self.0.store_stencil(x, y, stencil)
        }
    }

    struct UnlentStore<'a>(&'a mut dyn PlaneStore);

    impl PlaneStore for UnlentStore<'_> {
        fn planes(&mut self) -> Box<dyn Planes + '_> {
            Box::new(Unlent(self.0.planes()))
        }
    }

    /// The square of plane pixels the cases clear in.
    const SIDE: i32 = 96;

    /// Both colour buffers, depth and stencil over the square.
    fn snapshot(b: &Bench) -> Vec<u32> {
        let mut out = b.snapshot(SIDE).1;
        out.extend(b.snapshot(SIDE).2);
        for buffer in 0..2 {
            for y in 0..SIDE {
                for x in 0..SIDE {
                    out.push(b.board.buffer_load(buffer, x as i64, y as i64));
                }
            }
        }
        out
    }

    /// A clear in random state on a board in random state, its planes
    /// first filled with noise; through the row the planes lend, or not.
    fn clear(seed: u64, lend: bool) -> Vec<u32> {
        let mut b = Bench::empty();
        let mut rng = Lcg(seed.wrapping_mul(0x9e3779b97f4a7c15) | 1);
        let mut noise = Lcg(seed);

        for buffer in 0..2 {
            for y in 0..SIDE {
                for x in 0..SIDE {
                    b.board.buffer_store(buffer, x as i64, y as i64, noise.next_u32());
                }
            }
        }
        for v in b.board.depth_mem.bytes.iter_mut().take(4 * 1280 * SIDE as usize) {
            *v = noise.next_u32() as u8;
        }
        for v in b.board.stencil_mem.bytes.iter_mut().take(1280 * SIDE as usize) {
            *v = noise.next_u32() as u8;
        }
        if rng.next_u32() % 2 == 0 {
            b.board.id_compare = ODSY_ID_COMPARE_ENABLE | (0x0f << ODSY_ID_COMPARE_MASK_SHIFT) | 0x05;
            for d in b.board.did.iter_mut() {
                *d = if noise.next_u32() % 4 == 0 { 0x15 } else { 0x06 };
            }
        }
        if rng.next_u32() % 2 == 0 {
            // Board clip rectangles that split rows: those rows are not lent.
            b.board.clip_kept = 2;
            b.board.clip_rect[0] = ClipRect { x0: 0, y0: 0, x1: 40, y1: SIDE - 1 };
            b.board.clip_rect[1] = ClipRect { x0: 30, y0: 20, x1: SIDE - 1, y1: 70 };
        }
        let x0 = (rng.next_u32() % 40) as i32;
        let y0 = (rng.next_u32() % 40) as i32;
        let mut op = RasterOp {
            kind: SGI_RASTER_CLEAR,
            ntargets: 1 + rng.next_u32() % 2,
            targets: [0, 1, 0, 0],
            x0,
            y0,
            x1: x0 + 10 + (rng.next_u32() % 50) as i32,
            y1: y0 + 10 + (rng.next_u32() % 50) as i32,
            clear_buffers: [
                SGI_GL_COLOR_BUFFER_BIT | SGI_GL_DEPTH_BUFFER_BIT,
                SGI_GL_COLOR_BUFFER_BIT | SGI_GL_DEPTH_BUFFER_BIT | SGI_GL_STENCIL_BUFFER_BIT,
                SGI_GL_DEPTH_BUFFER_BIT,
                SGI_GL_STENCIL_BUFFER_BIT,
                SGI_GL_COLOR_BUFFER_BIT,
                SGI_GL_DEPTH_BUFFER_BIT | SGI_GL_STENCIL_BUFFER_BIT,
                SGI_GL_COLOR_BUFFER_BIT | SGI_GL_STENCIL_BUFFER_BIT,
            ][rng.next_u32() as usize % 7],
            clear_rgba: [0.2, 0.7, 0.4, 0.9],
            clear_colour_mask: if rng.next_u32() % 3 == 0 { rng.next_u32() % 16 } else { SGI_GL_COLOUR_MASK_ALL },
            clear_depth24: 0x123456,
            clear_stencil: 0x5a,
            stencil_mask: [u32::MAX, 0x0f, 0x81][rng.next_u32() as usize % 3],
            ..Default::default()
        };
        op.clip.n = 1 + rng.next_u32() % 2;
        // An L: rows above y 50 that cross x 30 to 40 lie in neither
        // rectangle whole, and below it pixels right of x 40 are outside.
        op.clip.r[0] = ClipRect { x0: 0, y0: 0, x1: 40, y1: SIDE - 1 };
        op.clip.r[1] = ClipRect { x0: 30, y0: 0, x1: SIDE - 1, y1: 50 };
        if op.clip.n == 1 {
            op.clip.r[0] = ClipRect { x0: 0, y0: 0, x1: SIDE - 1, y1: SIDE - 1 };
        }
        let t = b.board.gl_target();
        let mut parts = b.board.gl_parts(None).0;
        if lend {
            b.rules.raster(&t, &mut parts, &op);
        } else {
            b.rules.raster(&t, &mut UnlentStore(&mut parts), &op);
        }
        drop(parts);
        snapshot(&b)
    }

    // A clear stores through the rows the planes lend what it stores pixel
    // by pixel: colour under any mask, both buffers, depth and stencil
    // under its mask, with the board's id test and clip rectangles.
    #[test]
    fn lent_clear_rows_match_the_pixel_path() {
        for seed in 1..=100u64 {
            let want = clear(seed, false);
            let got = clear(seed, true);

            assert!(want == got, "seed {seed}: {} of {} plane words differ", want.iter().zip(&got).filter(|(a, b)| a != b).count(), want.len());
        }
    }
}
