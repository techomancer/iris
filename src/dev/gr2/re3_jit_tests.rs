//! Interpreter / JIT equivalence for the RE3 span shaders (`gr2-jit`).
//!
//! Two RE3s start identical: random VRAM and Z/stencil words. One draws
//! with the interpreter, the other with shaders compiled on first use
//! (`MODE_SYNC`). Every step picks a random pipeline configuration (each
//! key field uniformly, plus masks, scissor, WID, pattern bits, stencil and
//! Z data) and draws a few random SHADED, FLAT and FLAT4 spans through the
//! ordinary register path; after each step the two must agree word for
//! word: VRAM, the Z buffer and the span iterators. The test then asserts
//! that the JIT drew every span and that every value of every key field
//! was exercised. `GR2_JIT_SWEEP=<factor>` scales the steps (default 1 in
//! release, 0.15 in a debug build, where the full sweep takes over a minute
//! and the fewest steps that still draw every key value is about 0.05 of the
//! full count), `GR2_JIT_SEED` changes the seed.
//!
//! Run with `cargo test --release --features gr2-jit --lib re3_jit_tests`.

use super::re3::*;
use super::re3_jit::{self, Fmt, Pat, Prim, SpanKey, Wid, MODE_OFF, MODE_SYNC};
use std::collections::HashSet;

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn u32(&mut self) -> u32 {
        (self.next() >> 32) as u32
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }

    fn chance(&mut self, p: f64) -> bool {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64 <= p
    }

    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + self.below((hi - lo + 1) as u64) as i64
    }
}

fn sweep_factor() -> f64 {
    let default = if cfg!(debug_assertions) { 0.15 } else { 1.0 };
    std::env::var("GR2_JIT_SWEEP").ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn seed(salt: u64) -> u64 {
    std::env::var("GR2_JIT_SEED").ok().and_then(|v| v.parse().ok()).unwrap_or(0x5eed_6172) ^ salt
}

fn new_re3(mode: u32) -> Box<Re3> {
    // SAFETY: Re3 is plain data and valid when zeroed.
    let mut r = unsafe { Box::<Re3>::new_zeroed().assume_init() };
    r.jit.mode = mode;
    r
}

struct Pair {
    interp: Box<Re3>,
    jit: Box<Re3>,
    rng: Rng,
    spans: u64,
}

impl Pair {
    fn new(seed: u64) -> Pair {
        let mut interp = new_re3(MODE_OFF);
        let mut jit = new_re3(MODE_SYNC);
        let mut rng = Rng::new(seed);
        for (a, b) in interp.vram.iter_mut().zip(jit.vram.iter_mut()) {
            *a = rng.u32();
            *b = *a;
        }
        for (a, b) in interp.zbuf.iter_mut().zip(jit.zbuf.iter_mut()) {
            *a = rng.u32();
            *b = *a;
        }
        Pair { interp, jit, rng, spans: 0 }
    }

    fn reg(&mut self, r: usize, v: u32) {
        self.interp.write_reg(r, v);
        self.jit.write_reg(r, v);
    }

    fn both(&mut self, f: impl Fn(&mut Re3)) {
        f(&mut self.interp);
        f(&mut self.jit);
    }

    /// A random pipeline configuration, every key field uniform.
    fn configure(&mut self) {
        let rng = &mut self.rng;
        let mut regs: Vec<(usize, u32)> = vec![
            (REG_FUNC, rng.below(16) as u32),
            (REG_PIXMASK, if rng.chance(0.5) { 0x00ff_ffff } else { rng.u32() }),
            (REG_AUXMASK, rng.u32()),
            (REG_NOPUP, rng.below(2) as u32),
            (REG_UAUXDATA, rng.u32()),
            (REG_WIDDATA, rng.u32()),
            (REG_PUPDATA, rng.u32()),
            (REG_ENABPAT, rng.below(2) as u32),
            (REG_ALIGNPAT, rng.below(2) as u32),
            (REG_ENABDITH, rng.below(2) as u32),
            (REG_ENABWID, rng.below(2) as u32),
            (REG_CURWID, rng.u32()),
            (REG_FBOPTION, rng.u32()),
            (REG_DEPTHFN, rng.u32()),
        ];
        // Scissor: mostly the screen, sometimes a window, sometimes empty.
        let (xmin, xmax, ymin, ymax) = match rng.below(3) {
            0 => (0, FB_W as u32 - 1, 0, FB_H as u32 - 1),
            1 => {
                let x0 = rng.below(1300) as u32;
                let y0 = rng.below(1050) as u32;
                (x0, x0 + rng.below(400) as u32, y0, y0 + rng.below(300) as u32)
            }
            _ => (rng.u32(), rng.u32(), rng.u32(), rng.u32()),
        };
        regs.extend([(REG_XMIN, xmin), (REG_XMAX, xmax), (REG_YMIN, ymin), (REG_YMAX, ymax)]);
        let pixfmt = rng.below(8) as u32;
        // zctl: enable, func, Z write mask (often all, sometimes partial).
        let zm = if rng.chance(0.6) { 0x00ff_ffff } else { rng.u32() & 0x00ff_ffff };
        let zctl = rng.below(2) as u32 | (rng.below(8) as u32) << 1 | zm << 8;
        let mut st = rng.next();
        st &= !(0xffff_f << 28);
        st |= (rng.below(16) << 28) | (rng.below(16) << 32) | (rng.below(16) << 36);
        let blend = rng.below(2) as u32 | (rng.below(8) as u32) << 1 | (rng.below(8) as u32) << 4;
        let (a, da) = (rng.below(0x8_0000), rng.range(-0x2000, 0x2000));
        for (r, v) in regs {
            self.reg(r, v);
        }
        self.both(|r| {
            r.set_pixfmt(pixfmt);
            r.set_zctl(zctl);
            r.set_stencil(st);
            r.set_blend(blend);
            r.ctx.a = a as i64;
            r.ctx.da = da;
        });
    }

    /// One random span of a random instruction; returns its key.
    fn span(&mut self) -> SpanKey {
        let rng = &mut self.rng;
        let ir = [IR_SHADED, IR_FLAT, IR_FLAT4][rng.below(3) as usize];
        let horizontal = rng.chance(0.7);
        let (dx, dy) = if horizontal {
            (1u32 << 14, 0)
        } else {
            (rng.range(-0x6000, 0x6000) as i32 as u32, rng.range(-0x6000, 0x6000) as i32 as u32)
        };
        let n = if rng.chance(0.1) { rng.below(0x800) } else { rng.below(90) } as u32;
        let pos = |rng: &mut Rng, max: u64| {
            if rng.chance(0.8) { rng.below(max) as u32 } else { rng.u32() }
        };
        let x = pos(rng, 1340);
        let y = pos(rng, 1060);
        let regs = [
            (REG_PATL, rng.u32()),
            (REG_PATH, rng.u32()),
            (REG_XYFRAC, rng.u32()),
            (REG_DX, dx),
            (REG_DY, dy),
            (REG_R, rng.u32()),
            (REG_G, rng.u32()),
            (REG_B, rng.u32()),
            (REG_DR, rng.range(-0x4000, 0x4000) as i32 as u32),
            (REG_DG, rng.range(-0x4000, 0x4000) as i32 as u32),
            (REG_DB, if rng.chance(0.2) { rng.u32() } else { rng.range(-0x4000, 0x4000) as i32 as u32 }),
            (REG_Z, rng.u32()),
            (REG_DZI, rng.range(-0x40, 0x40) as i32 as u32),
            (REG_DZF, rng.u32()),
            (REG_X, x),
            (REG_Y, y),
            (REG_NUMPIX, n),
        ];
        for (r, v) in regs {
            self.reg(r, v);
        }
        let prim = Prim::of_ir(ir).unwrap();
        let key = re3_jit::key_of(&self.jit, prim);
        self.reg(REG_IR, ir);
        self.spans += 1;
        key
    }

    fn assert_same(&self, step: usize, key: &SpanKey) {
        let (a, b) = (&self.interp, &self.jit);
        if a.vram[..] != b.vram[..] {
            let i = a.vram.iter().zip(b.vram.iter()).position(|(p, q)| p != q).unwrap();
            panic!("step {step} {key}: VRAM differs at ({}, {}): interp {:#010x} jit {:#010x}",
                i % FB_W, i / FB_W, a.vram[i], b.vram[i]);
        }
        if a.zbuf[..] != b.zbuf[..] {
            let i = a.zbuf.iter().zip(b.zbuf.iter()).position(|(p, q)| p != q).unwrap();
            panic!("step {step} {key}: Z differs at ({}, {}): interp {:#010x} jit {:#010x}",
                i % FB_W, i / FB_W, a.zbuf[i], b.zbuf[i]);
        }
        let it = |c: &Re3Context| (c.x, c.y, c.z, c.r, c.g, c.b, c.a, c.numpix);
        assert_eq!(it(&a.ctx), it(&b.ctx), "step {step} {key}: iterators differ");
    }
}

/// Every key field value the sweep drew, per field.
#[derive(Default)]
struct Coverage(HashSet<(&'static str, u64)>);

impl Coverage {
    fn add(&mut self, k: &SpanKey) {
        let f = &mut self.0;
        f.insert(("prim", k.prim as u64));
        f.insert(("fmt", k.fmt as u64));
        f.insert(("dither", k.dither as u64));
        f.insert(("rop", k.rop as u64));
        f.insert(("pat", k.pat as u64));
        f.insert(("wid", k.wid as u64));
        f.insert(("z", k.z.map_or(0, |z| z as u64 + 1)));
        if let Some(s) = k.stencil {
            f.insert(("sfunc", s.func as u64));
            f.insert(("sfail", s.fail as u64));
            f.insert(("szfail", s.zfail as u64));
            f.insert(("szpass", s.zpass as u64));
        }
        f.insert(("stencil", k.stencil.is_some() as u64));
        if let Some(b) = k.blend {
            f.insert(("bsrc", b.src as u64));
            f.insert(("bdst", b.dst as u64));
        }
        f.insert(("blend", k.blend.is_some() as u64));
    }

    fn assert_full(&self) {
        let want: &[(&str, u64)] = &[
            ("prim", Prim::ALL.len() as u64),
            ("fmt", Fmt::ALL.len() as u64),
            ("dither", 2),
            ("rop", 16),
            ("pat", Pat::ALL.len() as u64),
            ("wid", Wid::ALL.len() as u64),
            ("z", 9),
            ("sfunc", 8),
            ("sfail", 6),
            ("szfail", 6),
            ("szpass", 6),
            ("stencil", 2),
            ("bsrc", 6),
            ("bdst", 6),
            ("blend", 2),
        ];
        for &(field, n) in want {
            for v in 0..n {
                assert!(self.0.contains(&(field, v)), "key field {field} = {v} never drawn");
            }
        }
    }
}

#[test]
fn spans_match_the_interpreter() {
    if re3_jit::forbidden() {
        return;
    }
    let mut p = Pair::new(seed(1));
    let mut cov = Coverage::default();
    let steps = (1500.0 * sweep_factor()) as usize;
    for step in 0..steps {
        p.configure();
        for _ in 0..4 {
            let k = p.span();
            cov.add(&k);
            p.assert_same(step, &k);
        }
    }
    assert_eq!(p.jit.jit.hits, p.spans, "every span drawn by a shader");
    assert_eq!(p.interp.jit.hits, 0);
    cov.assert_full();
}

/// A key-shaping register written between spans takes effect at once (the
/// memoised shader is dropped), and rewriting the same value keeps it.
#[test]
fn state_changes_drop_the_memoised_shader() {
    if re3_jit::forbidden() {
        return;
    }
    let mut p = Pair::new(seed(2));
    p.configure();
    for rop in [3u32, 6, 3, 3, 0xa] {
        p.reg(REG_FUNC, rop);
        let k = p.span();
        assert_eq!(k.rop as u32, rop);
        p.assert_same(0, &k);
        assert_eq!(SpanKey::unpack(p.jit.jit.last_key()).rop as u32, rop);
    }
    for fmt in 0..6 {
        p.both(|r| r.set_pixfmt(fmt));
        let k = p.span();
        p.assert_same(1, &k);
        assert_eq!(SpanKey::unpack(p.jit.jit.last_key()).fmt, Fmt::of(fmt));
    }
}

/// Interpreter and JIT speed on GL-like spans (`--ignored --nocapture`).
#[test]
#[ignore]
fn bench_interpreter_vs_jit() {
    let cases: [(&str, u32, u32, u32, u32); 5] = [
        ("flat fill", IR_FLAT, 0, 0, 0),
        ("smooth", IR_SHADED, 0, 0, 0),
        ("smooth + z", IR_SHADED, 0, 1 | 7 << 1 | 0xff_ffff << 8, 0),
        ("smooth + z + blend", IR_SHADED, 0, 1 | 7 << 1 | 0xff_ffff << 8, 1 | 4 << 1 | 5 << 4),
        ("smooth 12-bit dither", IR_SHADED, PIXFMT_RGB12, 0, 0),
    ];
    for (name, ir, pixfmt, zctl, blend) in cases {
        let mut t = [0f64; 2];
        for (i, mode) in [MODE_OFF, MODE_SYNC].into_iter().enumerate() {
            let mut r = new_re3(mode);
            for (reg, v) in [(REG_FUNC, ROP_COPY), (REG_PIXMASK, 0x00ff_ffff), (REG_XMAX, FB_W as u32 - 1), (REG_YMAX, FB_H as u32 - 1),
                (REG_DX, 1 << 14), (REG_DR, 0x40), (REG_DG, 0x20), (REG_DB, 0x10), (REG_DZI, 1)] {
                r.write_reg(reg, v);
            }
            r.set_pixfmt(pixfmt);
            r.set_zctl(zctl);
            r.set_blend(blend);
            r.write_reg(REG_ENABDITH, (pixfmt == PIXFMT_RGB12) as u32);
            // Warm the shader up.
            r.write_reg(REG_NUMPIX, 8);
            r.write_reg(REG_IR, ir);
            let start = std::time::Instant::now();
            let mut px = 0u64;
            for y in (0..20).flat_map(|_| 0..FB_H as u32) {
                for x0 in [0u32, 400, 800] {
                    r.write_reg(REG_X, x0);
                    r.write_reg(REG_Y, y);
                    r.write_reg(REG_R, 0x1000);
                    r.write_reg(REG_Z, y * 100);
                    r.write_reg(REG_NUMPIX, 400);
                    r.write_reg(REG_IR, ir);
                    px += 400;
                }
            }
            t[i] = px as f64 / start.elapsed().as_secs_f64() / 1e6;
        }
        println!("{name:20} interpreter {:7.1} Mpx/s  jit {:7.1} Mpx/s  x{:.1}", t[0], t[1], t[1] / t[0]);
    }
}
