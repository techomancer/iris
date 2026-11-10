//! Interpreter / JIT equivalence for the raster shaders (`gr4-jit`).
//!
//! Two boards start identical: random pixel memory, clip IDs and TRAM. One
//! draws with the interpreter, the other with JIT shaders compiled on first
//! use (`MODE_SYNC`). Every primitive goes to both through the ordinary
//! register path, and after it the boards must agree word for word: pixel
//! memory, the clip-ID planes, the GL line stipple position and the
//! character stipple's position.
//!
//! The key space is about 2^58, so it is swept, not enumerated. Each step
//! draws a batch of candidate register configurations (colours, masks,
//! modes, geometry, all random) and keeps the one whose key adds the most
//! (field, value) x (field, value) pairs not tested yet; then it draws a few
//! primitives with it. Every test then asserts that every value of every
//! key field its primitive uses was exercised, and reports the pair
//! coverage. `GR4_JIT_SWEEP=<factor>` scales the number of steps (default 1
//! in release, 0.35 in a debug build, where the full sweeps take minutes and
//! the fewest steps that still reach every key value is about 0.23 of the
//! full count), `GR4_JIT_SEED` changes the seed.
//!
//! Run with `cargo test --release --features gr4-jit --lib rss_jit_tests`.

use super::rss::{self, reg, Rss, TriSetup};
use super::rss_jit::{self, key::XBPP, Draw, PipeKey, Pix, Prim, MODE_OFF, MODE_SYNC};
use super::te1::reg as te;
use std::collections::{HashMap, HashSet};
use super::test_gen::{self, block_regs, hilo, pos, Regs, Rng};

/// The generators' name for a primitive kind.
fn gen(p: Prim) -> test_gen::Prim {
    match p {
        Prim::Fill => test_gen::Prim::Fill,
        Prim::Line => test_gen::Prim::Line,
        Prim::Stipple => test_gen::Prim::Stipple,
        Prim::Xfer => test_gen::Prim::Xfer,
        Prim::Tri => test_gen::Prim::Tri,
        Prim::GlLine => test_gen::Prim::GlLine,
    }
}

fn mode_regs(rng: &mut Rng, prim: Prim, tex_bias: f64) -> Regs {
    test_gen::mode_regs(rng, gen(prim), tex_bias)
}

fn data_regs(rng: &mut Rng, prim: Prim) -> (Regs, u32) {
    test_gen::data_regs(rng, gen(prim))
}

fn rss_iter_one() -> f64 {
    super::te1::ITER_ONE
}

// ── randomness ───────────────────────────────────────────────────────────────

fn sweep_factor() -> f64 {
    let default = if cfg!(debug_assertions) { 0.35 } else { 1.0 };
    std::env::var("GR4_JIT_SWEEP").ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn seed(salt: u64) -> u64 {
    std::env::var("GR4_JIT_SEED").ok().and_then(|v| v.parse().ok()).unwrap_or(0x1234_5678) ^ salt
}

// ── two boards ───────────────────────────────────────────────────────────────

/// Pages the tests draw into (A, B, overlay; depth is page 0): the
/// interesting ones are refreshed with random words between steps.
const LIVE_PAGES: usize = 0x100;

struct Pair {
    interp: Box<Rss>,
    jit: Box<Rss>,
    rng: Rng,
    /// Words the primitives changed (interpreter board, live pages).
    changed: u64,
    before: Vec<u64>,
}

impl Pair {
    fn new(seed: u64) -> Pair {
        let mut interp = Rss::new_boxed();
        let mut jit = Rss::new_boxed();
        interp.jit.mode = MODE_OFF;
        jit.jit.mode = MODE_SYNC;
        let mut rng = Rng::new(seed);
        for b in interp.te.tram_mut().iter_mut() {
            *b = rng.next() as u8;
        }
        jit.te.tram_mut().copy_from_slice(interp.te.tram_mut());
        let mut p = Pair { interp, jit, rng, changed: 0, before: Vec::new() };
        p.refresh();
        p
    }

    /// New random pixels and clip IDs where the tests draw (both boards).
    fn refresh(&mut self) {
        let n = LIVE_PAGES * super::pixmem::PAGE_WORDS;
        for re in 0..super::pixmem::RES {
            let live = re * super::pixmem::RE_WORDS..re * super::pixmem::RE_WORDS + n;
            for w in self.interp.mem.words[live.clone()].iter_mut() {
                *w = self.rng.next() & super::pixmem::WORD_MASK;
            }
            self.jit.mem.words[live.clone()].copy_from_slice(&self.interp.mem.words[live]);
        }
        for y in 0..512 {
            for x in 0..1024 {
                // Mostly one ID per 16-pixel run, so CID tests pass some.
                let v = if self.rng.chance(0.9) { ((x / 16 + y / 8) % 4) as u8 } else { self.rng.below(4) as u8 };
                self.interp.cid[y * rss::WIDTH + x] = v;
            }
        }
        self.jit.cid.copy_from_slice(&self.interp.cid[..]);
    }

    fn write(&mut self, r: u32, v: u32) {
        self.interp.write(r, v, false);
        self.jit.write(r, v, false);
    }

    fn exec(&mut self, r: u32, v: u32) {
        self.interp.write(r, v, true);
        self.jit.write(r, v, true);
    }

    fn apply(&mut self, regs: &[(u32, u32)]) {
        for &(r, v) in regs {
            self.write(r, v);
        }
    }

    fn snapshot(&mut self) {
        let n = LIVE_PAGES * super::pixmem::PAGE_WORDS;
        self.before.clear();
        self.before.extend_from_slice(&self.interp.mem.words[..n]);
    }

    fn count_changed(&mut self) {
        let n = self.before.len();
        self.changed += self.interp.mem.words[..n].iter().zip(&self.before).filter(|(a, b)| a != b).count() as u64;
    }

    /// The boards must agree on everything drawing can change.
    fn check(&self, what: &dyn Fn() -> String) {
        let (a, b) = (&self.interp, &self.jit);
        if a.mem.words[..] != b.mem.words[..] {
            let diffs: Vec<usize> = (0..a.mem.words.len()).filter(|&i| a.mem.words[i] != b.mem.words[i]).collect();
            let i = diffs[0];
            let pw = super::pixmem::PAGE_WORDS;
            panic!(
                "pixel memory differs in {} words; first word {i} (page {:#x}, row {}, col {}): interp {:#011x} jit {:#011x}\n{}",
                diffs.len(),
                i / pw,
                (i % pw) / super::pixmem::TILE_W,
                i % super::pixmem::TILE_W,
                a.mem.words[i],
                b.mem.words[i],
                what()
            );
        }
        if a.cid[..] != b.cid[..] {
            let i = (0..a.cid.len()).find(|&i| a.cid[i] != b.cid[i]).unwrap();
            panic!("clip IDs differ at ({}, {}): interp {} jit {}\n{}", i % rss::WIDTH, i / rss::WIDTH, a.cid[i], b.cid[i], what());
        }
        assert_eq!(a.gl_line_stipple_pos, b.gl_line_stipple_pos, "GL line stipple position\n{}", what());
        assert_eq!(a.stipple_state(), b.stipple_state(), "character stipple position\n{}", what());
    }
}

// ── keys and coverage ────────────────────────────────────────────────────────

/// The key the JIT board derives for `prim` with the registers as they
/// stand (registers only: nothing is drawn).
fn key_of(rss: &mut Rss, prim: Prim) -> Option<PipeKey> {
    use rss_jit::Args;
    let fm = rss.reg(reg::FILLMODE);
    let block = rss.current_block();
    let mode = rss.reg(reg::XFRMODE);
    let s = rss::Stipple { block, col: 0, row: 0, opaque: fm & (1 << 4) != 0 };
    let x = rss::Xfer {
        block,
        read: false,
        pio_read: false,
        width: 1,
        bpp: rss::bytes_per_pixel(mode),
        format: rss::xfer_format(mode),
        begin_skip: 0,
        stride_skip: 0,
        pio_dma_order: false,
        line: 0,
        line_begin: 0,
        pending: 0,
        skip: 0,
        rd_line: 0,
        rd_pos: 0,
        rd_begin: 0,
        out_lo: 0,
    };
    let bytes = vec![0u8; x.bpp as usize];
    let t = TriSetup { stipple: (fm & (1 << 2) != 0) as u32, ..TriSetup::default() };
    let l = rss::GlLineSetup { stipple: (fm & rss::FILL_LINE_STIPPLE != 0) as u32, ..rss::GlLineSetup::default() };
    let a = match prim {
        Prim::Fill => Args::Fill(&block),
        Prim::Line => Args::Line,
        Prim::Stipple => Args::Stipple(&s, 0, 0),
        Prim::Xfer => Args::Xfer(&x, 0, &bytes),
        Prim::Tri => Args::Tri(&t),
        Prim::GlLine => Args::GlLine(&l),
    };
    let bits = rss_jit::prepare(rss, &a)?;
    Some(rss_jit::key(&rss.jit, prim, bits))
}

/// A key as (field, value) digits, for coverage.
fn digits(k: &PipeKey) -> Vec<(u8, u32)> {
    let k = k.normalized();
    let opt = |o: Option<u8>| o.map_or(0, |v| v as u32 + 1);
    let mut d = vec![
        (0, k.prim as u32),
        (1, k.cid_test as u32),
        (2, k.nclip as u32),
        (3, k.draw as u32),
        (4, k.pix as u32),
        (5, opt(k.logic)),
        (6, k.stipple as u32),
    ];
    if k.prim.is_gl() {
        let s = k.stencil;
        let b = k.blend;
        d.extend([
            (10, k.rgb as u32),
            (11, opt(k.alpha)),
            (12, opt(s.map(|s| s.func))),
            (13, opt(s.map(|s| s.fail))),
            (14, opt(s.map(|s| s.zfail))),
            (15, opt(s.map(|s| s.zpass))),
            (16, opt(k.z)),
            (17, opt(b.map(|b| b.src))),
            (18, opt(b.map(|b| b.dst))),
            (34, opt(b.map(|b| b.eq))),
            (19, k.fog as u32),
            (20, k.tex.is_some() as u32),
        ]);
        if let Some(t) = k.tex {
            d.extend([
                (21, t.env as u32),
                (22, t.nc as u32),
                (23, t.slot as u32),
                (24, t.class as u32),
                (25, t.mag_linear as u32),
                (26, t.min_linear as u32),
                (27, t.mipmap as u32),
                (28, t.mip_linear as u32),
                (29, t.clamp_s as u32),
                (30, t.clamp_t as u32),
                (31, t.gl_clamp as u32),
                (32, t.no_border as u32),
                (33, t.depth as u32),
                (35, t.detail as u32),
            ]);
        }
    } else {
        d.extend([(40, k.opaque as u32), (41, k.skip_last as u32), (42, k.xfmt as u32), (43, k.xbpp as u32)]);
    }
    d
}

/// Every value each field can take, for the coverage assertion.
fn domain(prim: Prim, field: u8) -> Vec<u32> {
    let r = |n: u32| (0..n).collect::<Vec<u32>>();
    match field {
        0 => vec![prim as u32],
        1 | 6 | 10 | 19 | 20 | 25..=32 | 35 | 40 | 41 => r(2),
        2 => r(5),
        3 => {
            if prim.is_gl() {
                vec![Draw::Single as u32, Draw::Dual as u32, Draw::Overlay as u32]
            } else {
                r(4)
            }
        }
        4 => r(Pix::ALL.len() as u32),
        5 => r(17),
        11 => r(8),
        12 | 16 => r(9),
        13..=15 => r(7),
        17 | 18 => r(16),
        34 => r(6),
        21 => r(4),
        22 => vec![1, 2, 3, 4],
        23 => r(4),
        24 => r(4),
        33 => vec![1, 2, 3],
        42 => r(8),
        43 => XBPP.iter().map(|&b| b as u32).collect(),
        _ => vec![],
    }
}

/// The fields a primitive's keys use.
fn fields(prim: Prim) -> Vec<u8> {
    let mut f = vec![0, 1, 2, 3, 4, 5];
    match prim {
        Prim::Tri | Prim::GlLine => f.extend([6, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35]),
        Prim::Line => f.extend([6, 40, 41]),
        Prim::Stipple => f.push(40),
        Prim::Xfer => f.extend([42, 43]),
        Prim::Fill => {}
    }
    f
}

#[derive(Default)]
struct Coverage {
    singles: HashMap<u8, HashSet<u32>>,
    pairs: HashSet<(u8, u32, u8, u32)>,
    keys: HashSet<u64>,
}

impl Coverage {
    fn gain(&self, d: &[(u8, u32)]) -> usize {
        let mut n = 0;
        for (i, a) in d.iter().enumerate() {
            if !self.singles.get(&a.0).is_some_and(|s| s.contains(&a.1)) {
                n += 64;
            }
            for b in &d[i + 1..] {
                if !self.pairs.contains(&(a.0, a.1, b.0, b.1)) {
                    n += 1;
                }
            }
        }
        n
    }

    fn add(&mut self, k: &PipeKey) {
        let d = digits(k);
        for (i, a) in d.iter().enumerate() {
            self.singles.entry(a.0).or_default().insert(a.1);
            for b in &d[i + 1..] {
                self.pairs.insert((a.0, a.1, b.0, b.1));
            }
        }
        self.keys.insert(k.pack());
    }

    fn assert_complete(&self, prim: Prim) {
        let mut missing = Vec::new();
        for f in fields(prim) {
            let seen = self.singles.get(&f).cloned().unwrap_or_default();
            let want: Vec<u32> = domain(prim, f).into_iter().filter(|v| !seen.contains(v)).collect();
            if !want.is_empty() {
                missing.push(format!("field {f}: {want:?}"));
            }
        }
        assert!(missing.is_empty(), "{prim:?}: key values never tested: {}", missing.join("; "));
    }
}

// ── the sweep ────────────────────────────────────────────────────────────────

/// One step's configuration and how to draw with it.
struct Candidate {
    /// Registers that shape the pipeline (the key).
    mode: Regs,
    /// Geometry and data, then the IR opcode for primitives drawn by an
    /// execute.
    data: Regs,
    op: u32,
    key: PipeKey,
}

fn candidate(p: &mut Pair, prim: Prim, tex_bias: f64) -> Option<Candidate> {
    let mode = mode_regs(&mut p.rng, prim, tex_bias);
    let (data, op) = data_regs(&mut p.rng, prim);
    // Score on the JIT board's registers; both boards get the winner.
    for &(r, v) in mode.iter().chain(&data) {
        p.jit.write(r, v, false);
    }
    let key = key_of(&mut p.jit, prim)?;
    Some(Candidate { mode, data, op, key })
}

/// Draw with a configuration on both boards and compare them.
fn draw(p: &mut Pair, prim: Prim, c: &Candidate, step: usize) {
    p.apply(&c.mode);
    p.apply(&c.data);
    // Sometimes draw as one of Maximum Impact's two raster engines, which
    // takes only its own (even or odd) scanlines.
    let (engine, own_mask) = if p.rng.chance(0.3) { (p.rng.below(2) as u32, 1) } else { (0, 0) };
    p.interp.set_engine(engine, own_mask);
    p.jit.set_engine(engine, own_mask);
    p.snapshot();
    let hits0 = p.jit.jit.hits;
    let what = |extra: &str| {
        let regs: Vec<String> = c.mode.iter().chain(&c.data).map(|(r, v)| format!("{r:#05x}={v:#x}")).collect();
        format!("step {step}, key {} ({:#x}), engine {engine} mask {own_mask}{extra}\nregisters: {}", c.key, c.key.pack(), regs.join(" "))
    };
    match prim {
        Prim::Fill | Prim::Line | Prim::Tri | Prim::GlLine => {
            p.exec(reg::IR, c.op);
            p.check(&|| what(""));
        }
        Prim::Stipple => {
            p.exec(reg::IR, c.op);
            let chunks = p.rng.range(1, 16);
            for _ in 0..chunks {
                let (h, l) = (p.rng.u32(), p.rng.u32());
                if p.rng.chance(0.5) {
                    p.exec(reg::CHAR_H, h);
                } else {
                    p.write(reg::CHAR_H, h);
                    p.exec(reg::CHAR_L, l);
                }
            }
            p.check(&|| what(&format!(", {chunks} chunks")));
        }
        Prim::Xfer => {
            p.exec(reg::IR, c.op);
            let mode = p.jit.reg(reg::XFRMODE);
            let (bpp, width) = (rss::bytes_per_pixel(mode) as usize, (p.jit.reg(reg::XFRSIZE) & 0xFFFF) as usize);
            for line in 0..p.rng.range(1, 7) as u32 {
                let len = match p.rng.below(8) {
                    0 => p.rng.below((width * bpp) as u64 + 1) as usize,
                    1 => width * bpp + p.rng.below(16) as usize,
                    _ => width * bpp,
                };
                let bytes: Vec<u8> = (0..len).map(|_| p.rng.next() as u8).collect();
                p.interp.dma_write_line(line, &bytes);
                p.jit.dma_write_line(line, &bytes);
            }
            p.check(&|| what(", transfer lines"));
        }
    }
    p.count_changed();
    assert_eq!(p.jit.jit.misses, 0, "a shader failed to compile\n{}", what(""));
    if p.jit.jit.hits > hits0 {
        assert_eq!(p.jit.jit.last_key(), c.key.pack(), "the board ran another key than derived\n{}", what(""));
    }
}

fn sweep(prim: Prim, salt: u64, steps: usize, tex_bias: f64) {
    let steps = ((steps as f64) * sweep_factor()).ceil() as usize;
    let mut p = Pair::new(seed(salt));
    let mut cov = Coverage::default();
    let mut prims = 0usize;
    let t0 = std::time::Instant::now();
    for step in 0..steps {
        if step % 4 == 0 {
            p.refresh();
        }
        let mut best: Option<(usize, Candidate)> = None;
        for _ in 0..24 {
            let Some(c) = candidate(&mut p, prim, tex_bias) else { continue };
            let g = cov.gain(&digits(&c.key));
            if best.as_ref().is_none_or(|(bg, _)| g > *bg) {
                best = Some((g, c));
            }
        }
        let Some((_, c)) = best else { continue };
        cov.add(&c.key);
        // A few primitives with this pipeline, new geometry and data each.
        draw(&mut p, prim, &c, step);
        prims += 1;
        for _ in 0..2 {
            let (data, op) = data_regs(&mut p.rng, prim);
            let more = Candidate { mode: c.mode.clone(), data, op, key: c.key };
            draw(&mut p, prim, &more, step);
            prims += 1;
        }
    }
    let hits = p.jit.jit.hits;
    eprintln!(
        "{prim:?}: {steps} steps, {prims} primitives, {} keys, {} field pairs, {hits} shader runs, {} declined, {} words written, {:.1}s",
        cov.keys.len(),
        cov.pairs.len(),
        p.jit.jit.declined,
        p.changed,
        t0.elapsed().as_secs_f64()
    );
    // GL lines are short a third of the time (0 to 3 pixels): about nine
    // words each.
    let floor = if matches!(prim, Prim::GlLine) { 8 } else { 10 };
    assert!(p.changed >= floor * prims as u64, "{prim:?}: the primitives wrote only {} words", p.changed);
    assert!(hits as usize * 2 >= prims, "{prim:?}: shaders ran for only {hits} of {prims} primitives");
    cov.assert_complete(prim);
}

#[test]
fn fills_match_the_interpreter() {
    sweep(Prim::Fill, 1, 200, 0.0);
}

#[test]
fn x_lines_match_the_interpreter() {
    sweep(Prim::Line, 2, 200, 0.0);
}

#[test]
fn character_stipple_matches_the_interpreter() {
    sweep(Prim::Stipple, 3, 160, 0.0);
}

#[test]
fn transfers_match_the_interpreter() {
    sweep(Prim::Xfer, 4, 220, 0.0);
}

#[test]
fn triangles_match_the_interpreter() {
    sweep(Prim::Tri, 5, 350, 0.3);
}

#[test]
fn textured_triangles_match_the_interpreter() {
    sweep(Prim::Tri, 6, 350, 0.95);
}

#[test]
fn gl_lines_match_the_interpreter() {
    sweep(Prim::GlLine, 7, 300, 0.5);
}

/// The dispatcher's keys survive packing.
#[test]
fn derived_keys_round_trip() {
    let mut p = Pair::new(seed(8));
    for prim in Prim::ALL {
        for _ in 0..200 {
            if let Some(c) = candidate(&mut p, prim, 0.5) {
                assert_eq!(PipeKey::unpack(c.key.pack()), c.key.normalized(), "{}", c.key);
            }
        }
    }
}


/// Interpreter vs JIT throughput on representative primitives (run with
/// `--ignored --nocapture`).
#[test]
#[ignore]
fn bench_interpreter_vs_jit() {
    let mut p = Pair::new(seed(77));
    let base: Regs = vec![
        (reg::PP1FILLMODE, 1 << 14 | 2 << 8),
        (reg::DRBPOINTERS, 0x40 | 0x80 << 10),
        (reg::DRBSIZE, 7 << 2 | 2),
        (reg::COLORMASKLSBSA, u32::MAX),
        (reg::COLORMASKMSBS, 0xFF),
        (reg::XYWIN, 1023 << 16),
        (reg::CONFIG, rss::CONFIG_YFLIP),
        (reg::CLIP_MODE, 0),
        (reg::PP1WINMODE, 0),
    ];
    let run = |p: &mut Pair, name: &str, regs: &Regs, pixels: f64, n: u32, go: &dyn Fn(&mut Rss)| {
        p.apply(&base);
        p.apply(regs);
        go(&mut p.jit); // compile
        // Best of five: the host is shared.
        let mut times = [f64::MAX; 2];
        for _ in 0..5 {
            for (k, board) in [&mut p.interp, &mut p.jit].into_iter().enumerate() {
                let t = std::time::Instant::now();
                for _ in 0..n {
                    go(board);
                }
                times[k] = times[k].min(t.elapsed().as_secs_f64() / n as f64);
            }
        }
        eprintln!(
            "{name:34} interp {:8.2} ns/px  jit {:7.2} ns/px  x{:.1}",
            times[0] * 1e9 / pixels,
            times[1] * 1e9 / pixels,
            times[0] / times[1]
        );
    };
    let xy = |x: i32, y: i32| (x as u32 & 0xFFFF) << 16 | (y as u32 & 0xFFFF);
    // X: clear a 1024x768 window (fast fill).
    let fill = vec![(reg::FILLMODE, 1 << 20 | 1 << 22), (reg::BLOCKXYSTARTI, xy(0, 0)), (reg::BLOCKXYENDI, xy(1023, 767))];
    run(&mut p, "fast fill 1024x768", &fill, 1024.0 * 768.0, 3, &|r| {
        r.write(reg::IR, 0x8, true);
    });
    // The same in one page (cache resident: compute, not memory).
    let small = vec![(reg::FILLMODE, 1 << 20 | 1 << 22), (reg::BLOCKXYSTARTI, xy(0, 0)), (reg::BLOCKXYENDI, xy(191, 15))];
    run(&mut p, "fast fill 192x16 (one page)", &small, 192.0 * 16.0, 2000, &|r| {
        r.write(reg::IR, 0x8, true);
    });
    // X: a single-pixel fill (per-primitive overhead).
    let dot = vec![(reg::FILLMODE, 1 << 20 | 1 << 22), (reg::BLOCKXYSTARTI, xy(10, 10)), (reg::BLOCKXYENDI, xy(10, 10))];
    run(&mut p, "1x1 fill (overhead)", &dot, 1.0, 200_000, &|r| {
        r.write(reg::IR, 0x8, true);
    });
    // X: text, 8x16 opaque character stipple, 2 chunks a glyph row.
    let text = vec![(reg::FILLMODE, 1 << 22 | 1 << 3 | 1 << 4), (reg::BLOCKXYSTARTI, xy(100, 100)), (reg::BLOCKXYENDI, xy(107, 115))];
    run(&mut p, "8x16 glyph (16 stipple rows)", &text, 128.0, 20_000, &|r| {
        r.write(reg::IR, 0x8, true);
        for _ in 0..16 {
            r.write(reg::CHAR_H, 0x5A00_0000, true);
        }
    });
    // X: line.
    let line = vec![(reg::FILLMODE, 0), (reg::LINE_START, xy(10, 10)), (reg::LINE_END, xy(600, 300))];
    run(&mut p, "X line 591 px", &line, 591.0, 2000, &|r| {
        r.write(reg::IR, 0x5, true);
    });
    // DMA: a 1024-pixel RGBA8 line.
    let xfer = vec![(reg::FILLMODE, 5 << 22), (reg::XFRMODE, 0x80), (reg::XFRSIZE, 1024), (reg::BLOCKXYSTARTI, xy(0, 0)), (reg::BLOCKXYENDI, xy(1023, 63))];
    let bytes: Vec<u8> = (0..4096).map(|i| i as u8).collect();
    run(&mut p, "DMA RGBA8 line 1024 px", &xfer, 1024.0, 2000, &|r| {
        r.write(reg::IR, 0x8, true);
        r.dma_write_line(0, &bytes);
    });
    // GL: a smooth-shaded, depth-tested 200x200 triangle (half a square).
    let mut tri: Regs = vec![
        (reg::CONFIG, 0),
        (reg::XYWIN, 0),
        (reg::ZMODE, 0x0FFF_FFF0 | rss::ZMODE_TEST | 3),
        (reg::AFUNCMODE, 0),
        (reg::STENCILMODE, 0),
        (reg::BLENDFACTOR, 0),
        (te::TEXMODE1, 0),
        (reg::FOG_ON, 0),
        (reg::FILLMODE, 0),
    ];
    for (r, v) in [(0x000, 50.0), (0x001, 50.0), (0x002, 250.0), (0x003, 250.0), (0x004, 50.0), (0x005, 50.0)] {
        tri.push((r, pos(v)));
    }
    hilo(&mut tri, 0x006, 0);
    hilo(&mut tri, 0x008, 1 << 24);
    hilo(&mut tri, 0x00A, 0);
    for c in 0..4u32 {
        tri.push((0x05C + c, 0x40_0000));
        tri.push((0x060 + 2 * c, 0x1000));
        tri.push((0x061 + 2 * c, 0x800));
    }
    hilo(&mut tri, 0x068, 0x8000 * 4096);
    hilo(&mut tri, 0x06A, 0);
    hilo(&mut tri, 0x06C, 0);
    run(&mut p, "GL shaded z triangle 20k px", &tri, 20_000.0, 300, &|r| {
        r.write(reg::IR, rss::OP_AREA_LTOR, true);
    });
    // GL: the same, textured: RGBA8 256x256, mipmapped, trilinear, modulate.
    let mut ttri = tri.clone();
    let one = rss_iter_one();
    ttri.extend([
        (te::TEXMODE1, 1 | 3 << 3 | 4 << 9),
        (te::TEXMODE2, 1 << 2 | 1 << 5 | 1 << 6 | 1 << 7 | 1 << 8 | 1 << 19),
        (te::TXSIZE, 8 | 8 << 4),
        (te::TXLOD, 8),
        (te::TXADDR, 0),
    ]);
    for l in 0..16 {
        ttri.push((te::TXMIPMAP, l * 4));
    }
    hilo(&mut ttri, te::WI, one as i64);
    hilo(&mut ttri, te::SW, 0);
    hilo(&mut ttri, te::TW, 0);
    for (r, v) in [(te::DSWX, 0.004), (te::DTWX, 0.0), (te::DSWE, 0.0), (te::DTWE, -0.004), (te::DWIX, 0.0), (te::DWIE, 0.0)] {
        hilo(&mut ttri, r, (v * one) as i64);
    }
    run(&mut p, "GL textured trilinear tri 20k px", &ttri, 20_000.0, 100, &|r| {
        r.write(reg::IR, rss::OP_AREA_LTOR, true);
    });
    // Bilinear, no mipmaps: no level of detail to compute.
    let mut btri = ttri.clone();
    btri.push((te::TEXMODE2, 1 << 2 | 1 << 7 | 1 << 8));
    run(&mut p, "GL textured bilinear tri 20k px", &btri, 20_000.0, 100, &|r| {
        r.write(reg::IR, rss::OP_AREA_LTOR, true);
    });
    p.check(&|| "after the benchmark".into());
}

/// Per-primitive overhead only (for profiling).
#[test]
#[ignore]
fn bench_overhead() {
    let mut p = Pair::new(seed(76));
    let xy = |x: i32, y: i32| (x as u32 & 0xFFFF) << 16 | (y as u32 & 0xFFFF);
    p.apply(&[
        (reg::PP1FILLMODE, 1 << 14 | 2 << 8),
        (reg::DRBPOINTERS, 0x40 | 0x80 << 10),
        (reg::DRBSIZE, 7 << 2 | 2),
        (reg::COLORMASKLSBSA, u32::MAX),
        (reg::XYWIN, 1023 << 16),
        (reg::CONFIG, rss::CONFIG_YFLIP),
        (reg::CLIP_MODE, 0),
        (reg::PP1WINMODE, 0),
        (reg::FILLMODE, 1 << 20 | 1 << 22),
        (reg::BLOCKXYSTARTI, xy(10, 10)),
        (reg::BLOCKXYENDI, xy(10, 10)),
    ]);
    let n = std::env::var("N").ok().and_then(|v| v.parse().ok()).unwrap_or(20_000_000u32);
    let t = std::time::Instant::now();
    for _ in 0..n {
        p.jit.write(reg::IR, 0x8, true);
    }
    eprintln!("jit {:.2} ns/prim", t.elapsed().as_secs_f64() * 1e9 / n as f64);
    let t = std::time::Instant::now();
    for _ in 0..n {
        p.interp.write(reg::IR, 0x8, true);
    }
    eprintln!("interp {:.2} ns/prim", t.elapsed().as_secs_f64() * 1e9 / n as f64);
}
