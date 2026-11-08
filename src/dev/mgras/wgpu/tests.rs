//! The GPU raster unit against the interpreter, word for word.
//!
//! Two raster engines start identical: random pixel memory, clip IDs and
//! TRAM. One draws with the interpreter (rss.rs); the other has a GPU unit
//! attached, so its primitives become records. Both get the same register
//! writes and transfer lines, from the generators the JIT's sweep uses
//! (`test_gen`), and after every step the GPU unit's memory must equal the
//! interpreter's: pixel pages, clip-ID planes, and the CPU-side state
//! (character stipple position, GL line stipple position). Several
//! primitives go into each batch, so order within a batch is tested too.
//! They need a GPU (or lavapipe) and skip without one; run them with few
//! test threads (`clanker/wgpu/cpu-written-buffers-live-in-host-memory.md`).

use std::sync::Arc;

use super::super::pixmem::{PAGE_WORDS, RES, RE_WORDS};
use super::super::rss::{self, reg, Rss, WIDTH};
use super::super::te1::reg as te;
use super::super::test_gen::{self, hilo, pos, Prim, Regs, Rng};
use super::{xline_step, GpuLink, GpuRaster};
use crate::wgpu::{Gpu, GpuOptions};

fn gpu() -> Option<Arc<Gpu>> {
    match Gpu::new(&GpuOptions::default()) {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("skipped: {e}");
            None
        }
    }
}

/// Pages refreshed and compared every step (all of them), per engine, and
/// the clip-ID rows the generators reach.
const LIVE_PAGES: usize = super::super::pixmem::PAGES;
const LIVE_CID_ROWS: usize = 512;

/// A layout in which no two framebuffer pixels share a memory word, as in
/// IRIX's: the generators' narrow `DRBsize` rows (192 to 576 pixels, so
/// x = 192 is the word of x = 0 a tile row down) and overlapping page
/// runs (A from 0x40 into B at 0x80 and the depth buffer from page 0 into
/// A) make a primitive's pixels alias. The interpreter settles aliased
/// writes by scanline order; the GPU unit's tiles run in parallel, so two
/// of them storing one word race. Here rows are 960 to 1344 pixels and
/// the buffers' page runs are apart: depth from 0, A from 0x100, B from
/// 0x200, the overlay at 0x300 (what the generators' geometry reaches,
/// about 900 x 400 pixels, needs at most 0xB0 pages a buffer).
fn unaliased(regs: &mut Regs) {
    for (r, v) in regs.iter_mut() {
        match *r {
            reg::DRBPOINTERS => {
                let page = |p: u32| match p {
                    0 => 0,
                    0xC0 => 0x300,
                    p if p < 0x80 => 0x100 + (p - 0x40),
                    p => 0x200 + (p - 0x80),
                };
                *v = page(*v & 0x3FF) | page((*v >> 10) & 0x3FF) << 10;
            }
            reg::DRBSIZE => *v = (*v & 0xC0) | (5 + *v % 3) << 2 | 2,
            _ => {}
        }
    }
}

struct Pair {
    cpu: Box<Rss>,
    gpu: Box<Rss>,
    unit: Box<GpuRaster>,
    rng: Rng,
    /// Recent register writes, for a failure's report.
    log: Vec<String>,
    /// The live pages as they were at the last check (which side drew a
    /// differing word).
    before: Vec<u64>,
}

impl Pair {
    fn new(g: &Arc<Gpu>, seed: u64) -> Pair {
        let mut cpu = Rss::new_boxed();
        let mut gpu = Rss::new_boxed();
        #[cfg(feature = "gr4-jit")]
        {
            cpu.jit.mode = super::super::rss_jit::MODE_OFF;
            gpu.jit.mode = super::super::rss_jit::MODE_OFF;
        }
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        for b in cpu.te.tram_mut().iter_mut() {
            *b = rng.next() as u8;
        }
        gpu.te.tram_mut().copy_from_slice(cpu.te.tram_mut());
        for w in cpu.mem.words.iter_mut() {
            *w = rng.next();
        }
        gpu.mem.words.copy_from_slice(&cpu.mem.words);
        for c in cpu.cid.iter_mut() {
            *c = rng.below(4) as u8;
        }
        gpu.cid.copy_from_slice(&cpu.cid[..]);
        let mut unit = Box::new(GpuRaster::new(g).unwrap_or_else(|e| panic!("{e}")));
        unit.attach(&gpu.mem, &gpu.cid, &mut gpu.te);
        gpu.gpu = GpuLink::new(&mut unit);
        Pair { cpu, gpu, unit, rng, log: Vec::new(), before: Vec::new() }
    }

    /// New random pixels and clip IDs where the generators draw, on both.
    fn refresh(&mut self) {
        let n = LIVE_PAGES * PAGE_WORDS;
        for re in 0..RES {
            let live = re * RE_WORDS..re * RE_WORDS + n;
            for w in self.cpu.mem.words[live.clone()].iter_mut() {
                *w = self.rng.next();
            }
            self.gpu.mem.words[live.clone()].copy_from_slice(&self.cpu.mem.words[live.clone()]);
            self.unit.write_pix(live.start, &self.cpu.mem.words[live]);
        }
        for y in 0..LIVE_CID_ROWS {
            for x in 0..1024 {
                // Mostly one ID per 16-pixel run, so CID tests pass some.
                let v = if self.rng.chance(0.9) { ((x / 16 + y / 8) % 4) as u8 } else { self.rng.below(4) as u8 };
                self.cpu.cid[y * WIDTH + x] = v;
            }
        }
        let rows = 0..LIVE_CID_ROWS * WIDTH;
        self.gpu.cid[rows.clone()].copy_from_slice(&self.cpu.cid[rows.clone()]);
        self.unit.write_cid(0, &self.cpu.cid[rows]);
        self.snapshot();
    }

    fn write(&mut self, r: u32, v: u32) {
        self.cpu.write(r, v, false);
        self.gpu.write(r, v, false);
        self.log.push(format!("{r:#05x}={v:#x}"));
    }

    fn exec(&mut self, r: u32, v: u32) {
        self.cpu.write(r, v, true);
        self.gpu.write(r, v, true);
        self.log.push(format!("{r:#05x}={v:#x}!"));
    }

    fn apply(&mut self, regs: &Regs) {
        for &(r, v) in regs {
            self.write(r, v);
        }
    }

    fn dma_line(&mut self, line: u32, bytes: &[u8]) {
        self.cpu.dma_write_line(line, bytes);
        self.gpu.dma_write_line(line, bytes);
        self.log.push(format!("dma line {line} ({} bytes)", bytes.len()));
    }

    /// The boards agree: the live pages and clip-ID rows, or everything.
    fn check(&mut self, what: &str, full: bool) {
        let pages = if full { RE_WORDS / PAGE_WORDS } else { LIVE_PAGES };
        for re in 0..RES {
            let range = re * RE_WORDS..re * RE_WORDS + pages * PAGE_WORDS;
            let got = self.unit.read_pix(range.clone()).unwrap_or_else(|e| panic!("{what}: {e}"));
            let want = &self.cpu.mem.words[range.clone()];
            if got[..] != want[..] {
                let diffs: Vec<usize> = (0..got.len()).filter(|&i| got[i] != want[i]).collect();
                let i = diffs[0];
                let w = range.start + i;
                let was = |d: usize| -> &str {
                    let k = re * LIVE_PAGES * PAGE_WORDS + d;
                    match self.before.get(k) {
                        _ if full => "",
                        Some(&b) if b == got[d] => " (the GPU left it)",
                        Some(&b) if b == want[d] => " (the CPU left it)",
                        Some(_) => " (both drew)",
                        None => "",
                    }
                };
                let first: Vec<String> = diffs.iter().take(8).map(|&d| format!("{:#x}: gpu {:#011x} cpu {:#011x}{}", range.start + d, got[d], want[d], was(d))).collect();
                panic!(
                    "{what}: pixel memory differs in {} words; first word {w:#x} (engine {re}, page {:#x}, row {}, col {})\n  {}\nlast writes: {}",
                    diffs.len(),
                    (w % RE_WORDS) / PAGE_WORDS,
                    (w % PAGE_WORDS) / super::super::pixmem::TILE_W,
                    w % super::super::pixmem::TILE_W,
                    first.join("\n  "),
                    self.recent()
                );
            }
        }
        let rows = if full { rss::HEIGHT } else { LIVE_CID_ROWS };
        let got = self.unit.read_cid(0..rows * WIDTH).unwrap_or_else(|e| panic!("{what}: {e}"));
        if got[..] != self.cpu.cid[..rows * WIDTH] {
            let i = (0..got.len()).find(|&i| got[i] != self.cpu.cid[i]).unwrap();
            panic!("{what}: clip IDs differ at ({}, {}): gpu {} cpu {}\nlast writes: {}", i % WIDTH, i / WIDTH, got[i], self.cpu.cid[i], self.recent());
        }
        assert_eq!(self.cpu.gl_line_stipple_pos, self.gpu.gl_line_stipple_pos, "{what}: GL line stipple position");
        assert_eq!(self.cpu.stipple_state(), self.gpu.stipple_state(), "{what}: character stipple position");
        self.log.clear();
        self.snapshot();
    }

    fn snapshot(&mut self) {
        self.before.clear();
        for re in 0..RES {
            self.before.extend_from_slice(&self.cpu.mem.words[re * RE_WORDS..re * RE_WORDS + LIVE_PAGES * PAGE_WORDS]);
        }
    }

    fn recent(&self) -> String {
        let n = self.log.len();
        let mode: Vec<String> = [
            reg::PP1FILLMODE, reg::FILLMODE, reg::DRBPOINTERS, reg::DRBSIZE, reg::COLORMASKLSBSA, reg::COLORMASKLSBSB, reg::COLORMASKMSBS,
            reg::PP1WINMODE, reg::CLIP_MODE, reg::XYWIN, reg::CONFIG, reg::ZMODE, reg::STENCILMODE, reg::STENCILMASK, reg::AFUNCMODE,
            reg::BLENDFACTOR, reg::BLENDCOLOR_RG, reg::BLENDCOLOR_BA, reg::FOG_ON, te::TEXMODE1, te::TEXMODE2, te::TXSIZE, te::TXLOD,
        ]
        .iter()
        .map(|&r| format!("{r:#05x}={:#x}", self.cpu.reg(r)))
        .collect();
        format!("{}\nmode: {}", self.log[n.saturating_sub(400)..].join(" "), mode.join(" "))
    }

    /// One primitive with the registers as they stand (`op` for the IR).
    fn draw(&mut self, prim: Prim, op: u32) {
        match prim {
            Prim::Fill | Prim::Line | Prim::Tri | Prim::GlLine => self.exec(reg::IR, op),
            Prim::Stipple => {
                self.exec(reg::IR, op);
                for _ in 0..self.rng.range(1, 16) {
                    let (h, l) = (self.rng.u32(), self.rng.u32());
                    if self.rng.chance(0.5) {
                        self.exec(reg::CHAR_H, h);
                    } else {
                        self.write(reg::CHAR_H, h);
                        self.exec(reg::CHAR_L, l);
                    }
                }
            }
            Prim::Xfer => {
                self.exec(reg::IR, op);
                let mode = self.cpu.reg(reg::XFRMODE);
                let (bpp, width) = (rss::bytes_per_pixel(mode) as usize, (self.cpu.reg(reg::XFRSIZE) & 0xFFFF) as usize);
                for line in 0..self.rng.range(1, 7) as u32 {
                    let len = match self.rng.below(8) {
                        0 => self.rng.below((width * bpp) as u64 + 1) as usize,
                        1 => width * bpp + self.rng.below(16) as usize,
                        _ => width * bpp,
                    };
                    let bytes: Vec<u8> = (0..len).map(|_| self.rng.next() as u8).collect();
                    self.dma_line(line, &bytes);
                }
            }
        }
    }

    /// A texture load through the transfer engine into random TRAM pages
    /// (the texture loader registers, then DMA lines), between draws.
    fn tram_load(&mut self) {
        let mode = (self.rng.below(4) as u32) << 1 | (self.rng.below(4) as u32) << 3 | (self.rng.below(3) as u32) << 5;
        let (ls, lt) = (self.rng.range(0, 7) as u32, self.rng.range(0, 7) as u32);
        let regs = vec![
            (te::TL_MODE, mode),
            (te::TL_SPEC, ls << 4 | lt << 8),
            (te::TL_MIPMAP, self.rng.below(256) as u32),
            (te::TL_BORDER, self.rng.below(256) as u32),
            (te::TL_S_LEFT, self.rng.range(-1, 4) as u32),
            (te::TL_T_BOTTOM, self.rng.range(-1, 4) as u32),
            (reg::XFRSIZE, 1 << ls),
        ];
        self.apply(&regs);
        self.write(reg::XFRCONTROL, 5);
        for line in 0..self.rng.range(1, (1 << lt) + 2) as u32 {
            let bytes: Vec<u8> = (0..4 * ((1 << ls) + 2)).map(|_| self.rng.next() as u8).collect();
            self.dma_line(line, &bytes);
        }
        self.write(reg::XFRCONTROL, 0);
    }
}

/// `MGRAS_GPU_CHECK_EACH=1`: compare after every primitive, not every step.
fn each() -> bool {
    std::env::var("MGRAS_GPU_CHECK_EACH").is_ok()
}

fn sweep(prim: Prim, salt: u64, steps: usize, tex_bias: f64, loads: bool) {
    let Some(g) = gpu() else { return };
    let mut p = Pair::new(&g, 0xC0FF_EE00 ^ salt);
    let t0 = std::time::Instant::now();
    let mut prims = 0;
    for step in 0..steps {
        if step % 8 == 0 {
            p.refresh();
        }
        let mut mode = test_gen::mode_regs(&mut p.rng, prim, tex_bias);
        unaliased(&mut mode);
        p.apply(&mode);
        for _ in 0..4 {
            if loads && p.rng.chance(0.3) {
                p.tram_load();
                // The mode again: the load moved transfer registers.
                p.apply(&mode);
            }
            let (data, op) = test_gen::data_regs(&mut p.rng, prim);
            p.apply(&data);
            p.draw(prim, op);
            prims += 1;
            if each() {
                p.check(&format!("{prim:?} step {step} primitive {prims}"), false);
            }
        }
        p.check(&format!("{prim:?} step {step}"), false);
    }
    p.check(&format!("{prim:?} at the end"), true);
    let (batches, records) = p.unit.counts();
    eprintln!("{prim:?}: {steps} steps, {prims} primitives, {records} records in {batches} batches, {} dropped, {:.1}s", p.unit.dropped, t0.elapsed().as_secs_f64());
    assert!(records > 0, "{prim:?}: no records reached the unit");
}

#[test]
fn fills_match_the_interpreter() {
    sweep(Prim::Fill, 1, 120, 0.0, false);
}

#[test]
fn x_lines_match_the_interpreter() {
    sweep(Prim::Line, 2, 120, 0.0, false);
}

#[test]
fn character_stipple_matches_the_interpreter() {
    sweep(Prim::Stipple, 3, 120, 0.0, false);
}

#[test]
fn transfers_match_the_interpreter() {
    sweep(Prim::Xfer, 4, 120, 0.0, false);
}

#[test]
fn triangles_match_the_interpreter() {
    sweep(Prim::Tri, 5, 160, 0.3, false);
}

#[test]
fn textured_triangles_match_the_interpreter() {
    sweep(Prim::Tri, 6, 160, 0.95, true);
}

#[test]
fn gl_lines_match_the_interpreter() {
    sweep(Prim::GlLine, 7, 160, 0.5, true);
}

/// Triangles whose edges the f64 walk rounds: steep slopes, positions
/// off the 2^-8 grid (far from the bias), long slivers.
#[test]
fn steep_and_far_triangles_match_the_interpreter() {
    let Some(g) = gpu() else { return };
    let mut p = Pair::new(&g, 0xFA12);
    for step in 0..60 {
        if step % 8 == 0 {
            p.refresh();
        }
        let mut mode = test_gen::mode_regs(&mut p.rng, Prim::Tri, 0.2);
        unaliased(&mut mode);
        // Slopes this steep reach past any row's width: keep the pixels
        // to the 1344-pixel rows with screen mask 0.
        mode.retain(|&(r, _)| r != reg::DRBSIZE);
        mode.push((reg::DRBSIZE, 7 << 2 | 2));
        let clip = mode.iter().find(|&&(r, _)| r == reg::CLIP_MODE).map_or(0, |&(_, v)| v);
        mode.push((reg::CLIP_X, 1343));
        mode.push((reg::CLIP_Y, 1023));
        mode.push((reg::CLIP_MODE, clip | 0x11));
        for _ in 0..4 {
            let mut regs = Regs::new();
            test_gen::colour_regs(&mut p.rng, &mut regs);
            let ymin = p.rng.f(-30.0, 140.0);
            let ymax = ymin + p.rng.f(0.01, 150.0);
            let ymid = p.rng.f(ymin, ymax);
            let x = |rng: &mut Rng| rng.f(-200.0, 600.0);
            let far = p.rng.chance(0.3);
            for (r, v) in [(0x000, x(&mut p.rng)), (0x001, x(&mut p.rng)), (0x002, x(&mut p.rng)), (0x003, ymax), (0x004, ymid), (0x005, ymin)] {
                // Far from the bias, f32 keeps finer fractions.
                let v = if far { ((v - 49151.5 + 30000.0) as f32) as f64 + 49151.5 - 30000.0 } else { v };
                regs.push((r, if far { (v as f32 + 49151.5f32).to_bits() } else { pos(v) }));
            }
            for r in [0x006, 0x008, 0x00A] {
                let s = if p.rng.chance(0.3) { p.rng.f(-3000.0, 3000.0) } else { p.rng.f(-40.0, 40.0) };
                hilo(&mut regs, r, (s * (1u64 << 24) as f64) as i64 + p.rng.range(-3, 3));
            }
            test_gen::plane_regs(&mut p.rng, &mut regs);
            p.apply(&regs);
            let op = if p.rng.chance(0.5) { rss::OP_AREA_LTOR } else { rss::OP_AREA_RTOL };
            p.draw(Prim::Tri, op);
        }
        p.check(&format!("steep step {step}"), false);
    }
}

/// The closed-form X line step equals the interpreter's Bresenham loop on
/// every line between points of a small grid, both cap styles.
#[test]
fn xline_closed_form_matches_bresenham() {
    let r = -7i32..=7;
    for x0 in r.clone() {
        for y0 in r.clone() {
            for x1 in r.clone() {
                for y1 in r.clone() {
                    for skip_last in [false, true] {
                        // `Rss::line`'s walk.
                        let mut want = std::collections::HashMap::new();
                        let (mut x, mut y) = (x0, y0);
                        let (dx, dy) = ((x1 - x).abs(), -(y1 - y).abs());
                        let (sx, sy) = (if x < x1 { 1 } else { -1 }, if y < y1 { 1 } else { -1 });
                        let mut err = dx + dy;
                        let mut k = 0u32;
                        loop {
                            if skip_last && x == x1 && y == y1 {
                                break;
                            }
                            assert!(want.insert((x, y), k).is_none(), "a pixel twice");
                            if x == x1 && y == y1 {
                                break;
                            }
                            let e2 = 2 * err;
                            if e2 >= dy {
                                err += dy;
                                x += sx;
                            }
                            if e2 <= dx {
                                err += dx;
                                y += sy;
                            }
                            k += 1;
                        }
                        for x in -9..=9 {
                            for y in -9..=9 {
                                assert_eq!(
                                    xline_step(x0, y0, x1, y1, skip_last, x, y),
                                    want.get(&(x, y)).copied(),
                                    "line ({x0}, {y0})-({x1}, {y1}) skip_last {skip_last} at ({x}, {y})"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    // Long lines too: the u32 products near their limit.
    for &(x0, y0, x1, y1) in &[(-32768i32, -32768i32, 32767i32, 32766i32), (0, 0, 32767, 1), (5, -30000, -6, 30000)] {
        let (mut x, mut y) = (x0, y0);
        let (dx, dy) = ((x1 - x).abs(), -(y1 - y).abs());
        let (sx, sy) = (if x < x1 { 1 } else { -1 }, if y < y1 { 1 } else { -1 });
        let mut err = dx + dy;
        let mut k = 0u32;
        loop {
            assert_eq!(xline_step(x0, y0, x1, y1, false, x, y), Some(k), "long line ({x0}, {y0})-({x1}, {y1}) at ({x}, {y})");
            if x == x1 && y == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                err += dx;
                y += sy;
            }
            k += 1;
        }
    }
}

/// A texture load lands between two textured triangles that sample the
/// pages it writes: the first must see the texels from before the load
/// (the unit submits its batch first), the second those after.
#[test]
fn a_texture_load_lands_between_the_triangles_around_it() {
    let Some(g) = gpu() else { return };
    let mut p = Pair::new(&g, 0x7E47);
    const PAGE: u32 = 0x21;
    let one = super::super::te1::ITER_ONE;
    let mut mode: Regs = vec![
        (reg::PP1FILLMODE, 2 << 8 | 1 << 14),
        (reg::FILLMODE, 0),
        (reg::DRBPOINTERS, 0x100),
        (reg::DRBSIZE, 7 << 2 | 2),
        (reg::COLORMASKLSBSA, u32::MAX),
        (reg::PP1WINMODE, 0),
        (reg::CLIP_MODE, 0),
        (reg::CONFIG, 0),
        (reg::ZMODE, 0),
        (reg::STENCILMODE, 0),
        (reg::AFUNCMODE, 0),
        (reg::BLENDFACTOR, 0),
        (reg::FOG_ON, 0),
        // RGBA8 8x8, modulate, nearest, repeat; level 0 at PAGE.
        (te::TEXMODE1, 1 | 3 << 3 | 4 << 9),
        (te::TEXMODE2, 1 << 2),
        (te::TXSIZE, 3 | 3 << 4),
        (te::TXLOD, 0),
        (te::TXADDR, 0),
        (te::TXMIPMAP, PAGE),
        (te::TXADDR, 0),
        (te::TXBORDER, PAGE + 1),
    ];
    // Half a 50-pixel square, white, s and t across it once.
    for (r, v) in [(0x000, 10.0), (0x001, 10.0), (0x002, 60.0), (0x003, 60.0), (0x004, 10.0), (0x005, 10.0)] {
        mode.push((r, pos(v)));
    }
    hilo(&mut mode, 0x006, 0);
    hilo(&mut mode, 0x008, 1 << 24);
    hilo(&mut mode, 0x00A, 0);
    for c in 0..4u32 {
        mode.extend([(0x05C + c, 0xFFF000), (0x060 + 2 * c, 0), (0x061 + 2 * c, 0)]);
    }
    hilo(&mut mode, te::WI, one as i64);
    hilo(&mut mode, te::SW, 0);
    hilo(&mut mode, te::TW, 0);
    for (r, v) in [(te::DSWX, 1.0 / 50.0), (te::DTWX, 0.0), (te::DSWE, 0.0), (te::DTWE, 1.0 / 50.0), (te::DWIX, 0.0), (te::DWIE, 0.0)] {
        hilo(&mut mode, r, (v * one) as i64);
    }
    p.apply(&mode);
    p.write(reg::XYWIN, 0);
    p.exec(reg::IR, rss::OP_AREA_LTOR);
    // New texels into level 0's page, then the same triangle further right.
    let load: Regs = vec![
        (te::TL_MODE, 3 << 1 | 1 << 5),
        (te::TL_SPEC, 3 << 4 | 3 << 8),
        (te::TL_MIPMAP, PAGE),
        (te::TL_BORDER, PAGE + 1),
        (te::TL_S_LEFT, 0),
        (te::TL_T_BOTTOM, 0),
    ];
    p.apply(&load);
    p.write(reg::XFRCONTROL, 5);
    for line in 0..8 {
        let bytes: Vec<u8> = (0..32).map(|_| p.rng.next() as u8).collect();
        p.dma_line(line, &bytes);
    }
    p.write(reg::XFRCONTROL, 0);
    p.write(reg::XYWIN, 100);
    p.exec(reg::IR, rss::OP_AREA_LTOR);
    p.check("textured triangles around a load", false);
    // The two triangles differ (the load changed what they sample).
    let b = super::super::pixmem::Buffer::new(0x100, super::super::pixmem::Kind::Wide, 7 << 2 | 2);
    let differ = (12..58).any(|y| (12..30).any(|x| p.cpu.mem.get(&b, x, y) != p.cpu.mem.get(&b, x + 100, y)));
    assert!(differ, "the load changed nothing the triangles sample");
}

/// The GPU scan-out composes what `Frame::snapshot` and `Frame::compose`
/// compose, pixel for pixel, over random pixel memory and random display
/// state: window-ID tables (runs out of order, past the edge, rows with
/// none), every main format (colour index, RGB, both pixel-pair layouts),
/// buffer selects, overlays, colormap blocks, gamma, blanking, cursors.
#[test]
fn scanout_composes_as_the_cpu_frame() {
    use super::super::dcb::Dcb;
    use super::super::frame::{self, Frame};
    use super::super::plain::boxed_zeroed;
    use super::scanout::MgrasScanout;
    let Some(g) = gpu() else { return };
    let mut rng = Rng(0x5CA2_0001);
    let mut r = Rss::new_boxed();
    for w in r.mem.words.iter_mut() {
        *w = rng.next();
    }
    let mut unit = GpuRaster::new(&g).unwrap_or_else(|e| panic!("{e}"));
    unit.attach(&r.mem, &r.cid, &mut r.te);
    let mut sc = MgrasScanout::new(&g, &unit.pix_buffer()).unwrap_or_else(|e| panic!("{e}"));
    let mut dcb = boxed_zeroed::<Dcb>();
    dcb.init([0, 0], [0, 0]);
    let mut snap = boxed_zeroed::<Frame>();
    let mut cpu = vec![0u32; frame::OUT_STRIDE * frame::H];
    let mut out = vec![0u32; frame::OUT_STRIDE * frame::H];
    for trial in 0..16 {
        r.write(reg::DRBSIZE, (rng.chance(0.5) as u32) << 6 | (5 + rng.below(3) as u32) << 2 | 2, false);
        for did in 0..32 {
            let format = match rng.below(6) {
                0 => 1,
                1 => 2,
                2 => 5,
                3 => 7,
                4 => 0x15,
                _ => rng.below(32) as u32,
            };
            let overlay = if rng.chance(0.5) { 0 } else { (rng.below(32) as u32) << 3 | (1 + rng.below(7) as u32) };
            dcb.xmap.set_modes(did, format | (rng.below(32) as u32) << 5, overlay);
        }
        let page = |rng: &mut Rng| rng.below(1024) as u32;
        let (a, b) = (page(&mut rng), page(&mut rng));
        let o = if rng.chance(0.2) { a } else { page(&mut rng) };
        dcb.xmap.set_scanout(rng.u32(), a | b << 10 | o << 20, rng.u32());
        for c in dcb.cmap[0].pal.iter_mut() {
            *c = rng.u32() & 0xFF_FFFF;
        }
        for gm in dcb.dac.gamma.iter_mut() {
            *gm = [0, 1, 2].map(|_| rng.below(1024) as u16);
        }
        dcb.dac.set_pixmask(if rng.chance(0.1) { 0 } else { 0xFF });
        // Window-ID tables: frame tables at 0 (main) and 0x800 (overlay),
        // 32 line tables from 0x2000 the rows share.
        let h = if rng.chance(0.5) { 1024 } else { 768 };
        let vc3 = &mut dcb.vc3;
        vc3.regs[4] = 0;
        vc3.regs[5] = 0x800;
        for k in 0..32usize {
            let base = 0x2000 + 64 * k;
            let n = rng.below(10) as usize;
            for e in 0..n {
                let x = if rng.chance(0.1) { 1300 + rng.below(800) as u16 } else if e == 0 && rng.chance(0.5) { 0 } else { rng.below(1300) as u16 };
                vc3.sram[base + e] = x << 5 | rng.below(32) as u16;
            }
            vc3.sram[base + n] = 0x7FF << 5;
        }
        for y in 0..h {
            vc3.sram[y] = 0x2000 + 64 * rng.below(32) as u16;
            vc3.sram[0x800 + y] = if rng.chance(0.05) { 0xFFFF } else { 0x2000 + 64 * rng.below(32) as u16 };
        }
        vc3.sram[h] = 0xFFFF;
        // A cursor, mostly.
        vc3.regs[0x1D] = if rng.chance(0.8) { 3 | if rng.chance(0.5) { 8 } else { 0 } } else { 0 };
        vc3.regs[1] = 0x6000;
        vc3.regs[2] = rng.below(1400) as u16;
        vc3.regs[3] = rng.below(1100) as u16;
        for w in vc3.sram[0x6000..0x6200].iter_mut() {
            *w = rng.u32() as u16;
        }
        snap.snapshot(&r, &dcb);
        snap.compose(&mut cpu);
        let (w, hh) = sc.read(&r, &dcb, &mut out, frame::OUT_STRIDE).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!((w, hh), (snap.width, snap.height), "trial {trial}: frame size");
        let mut bad = Vec::new();
        for y in 0..hh {
            for x in 0..w {
                let i = y * frame::OUT_STRIDE + x;
                if cpu[i] != out[i] && bad.len() < 8 {
                    bad.push(format!("({x}, {y}) cpu {:08x} gpu {:08x}", cpu[i], out[i]));
                }
            }
        }
        assert!(bad.is_empty(), "trial {trial}: the scan-out differs: {}", bad.join(", "));
    }
}

/// The guest's pixel reads see the GPU unit's drawing: after primitives,
/// a DMA read of a random block (random format, buffer and window
/// state) gives the same bytes on both engines. The GPU engine brings its
/// mirror current only under the block (`Rss::gpu_sync_transfer`), so
/// later reads elsewhere must still find their pages marked.
#[test]
fn transfer_reads_see_the_drawing() {
    let Some(g) = gpu() else { return };
    let mut p = Pair::new(&g, 0x4EAD);
    let mut lines_read = 0;
    for step in 0..120 {
        if step % 8 == 0 {
            p.refresh();
        }
        let prim = [Prim::Fill, Prim::Line, Prim::Tri, Prim::Stipple, Prim::GlLine][step % 5];
        let mut mode = test_gen::mode_regs(&mut p.rng, prim, 0.3);
        unaliased(&mut mode);
        p.apply(&mode);
        for _ in 0..3 {
            let (data, op) = test_gen::data_regs(&mut p.rng, prim);
            p.apply(&data);
            p.draw(prim, op);
        }
        // A read: DMA block type 4, a transfer format the X server and
        // libGLcore use, a block somewhere in the drawing.
        for _ in 0..2 {
            let (f, c) = p.rng.pick(&[(0u32, 0u32), (0, 1), (8, 8), (8, 10), (8, 0), (2, 3), (7, 0), (8, 1), (7, 1), (5, 5)]);
            let order = if (f, c) == (8, 1) { 1 << 22 } else { 0 };
            let mut regs: Regs = vec![(reg::XFRMODE, (p.rng.u32() & 0x7FF00) | f << 4 | c | order), (reg::FILLMODE, 4 << 22), (reg::ZST_STENCIL, 0)];
            test_gen::block_regs(&mut p.rng, &mut regs, (60, 12));
            regs.push((reg::XFRSIZE, p.rng.range(1, 48) as u32));
            p.apply(&regs);
            p.exec(reg::IR, 0x8);
            for line in 0..p.rng.range(1, 8) as u32 {
                let want = p.cpu.dma_read_line(line);
                p.gpu.gpu_sync_transfer();
                let got = p.gpu.dma_read_line(line);
                assert_eq!(got, want, "step {step}, line {line}: the read differs\n{}", p.recent());
                lines_read += 1;
            }
        }
        p.check(&format!("reads step {step}"), false);
    }
    assert!(lines_read > 500, "{lines_read}");
    // The reads brought drawn pages back (they did not pass on a stale
    // mirror that happened to match).
    assert!(p.unit.syncs > 50 && p.unit.read_back > 0, "{} syncs, {} bytes read back", p.unit.syncs, p.unit.read_back);
}
