//! RE3 unit tests: drawing semantics on a bare `Re3` (no threads), plus the
//! CPU -> FIFO -> RE3 thread path through the bus.

use super::re3::*;
use super::*;

fn bare_re3() -> Box<Re3> {
    // SAFETY: Re3 is plain data and valid when zeroed.
    unsafe { Box::<Re3>::new_zeroed().assume_init() }
}

/// Draw state a textport-style client sets up: full scissor, all planes, COPY.
fn setup(r: &mut Re3) {
    for (reg, v) in [
        (REG_FUNC, ROP_COPY),
        (REG_PIXMASK, 0x00ff_ffff),
        (REG_XMIN, (0)),
        (REG_XMAX, (FB_W as u32 - 1)),
        (REG_YMIN, 0),
        (REG_YMAX, FB_H as u32 - 1),
    ] {
        r.write_reg(reg, v);
    }
}

fn flat(r: &mut Re3, x: u32, y: u32, n: u32, ci: u32) {
    r.write_reg(REG_R, ci << 11);
    r.write_reg(REG_X, x);
    r.write_reg(REG_Y, y);
    r.write_reg(REG_NUMPIX, n);
    r.write_reg(REG_IR, IR_FLAT);
}

fn px(r: &Re3, x: usize, y: usize) -> u32 {
    r.vram[y * FB_W + x]
}

#[test]
fn flat_span_draws_numpix_pixels() {
    let mut r = bare_re3();
    setup(&mut r);
    flat(&mut r, 100, 10, 7, 0x42);
    assert_eq!(px(&r, 99, 10), 0);
    for x in 100..107 {
        assert_eq!(px(&r, x, 10), 0x42, "x={x}");
    }
    assert_eq!(px(&r, 107, 10), 0);
}

#[test]
fn scissor_clips_span() {
    let mut r = bare_re3();
    setup(&mut r);
    r.write_reg(REG_XMIN, 10);
    r.write_reg(REG_XMAX, 12);
    flat(&mut r, 5, 0, 20, 1);
    let drawn: Vec<usize> = (0..30).filter(|&x| px(&r, x, 0) != 0).collect();
    assert_eq!(drawn, vec![10, 11, 12]);
}

#[test]
fn pattern_masks_span_msb_first() {
    let mut r = bare_re3();
    setup(&mut r);
    r.write_reg(REG_ENABPAT, 1);
    r.write_reg(REG_PATH, 0b1010_0000_0000_0001);
    r.write_reg(REG_PATL, 0);
    flat(&mut r, 0, 0, 16, 9);
    let drawn: Vec<usize> = (0..16).filter(|&x| px(&r, x, 0) != 0).collect();
    assert_eq!(drawn, vec![0, 2, 15]);
}

#[test]
fn pixmask_limits_written_bits() {
    let mut r = bare_re3();
    setup(&mut r);
    r.vram[0] = 0x00ab_cdef;
    r.write_reg(REG_PIXMASK, 0x0000_00ff);
    flat(&mut r, 0, 0, 1, 0x12);
    assert_eq!(px(&r, 0, 0), 0x00ab_cd12);
}

#[test]
fn aux_planes_untouched_when_auxmask_zero() {
    let mut r = bare_re3();
    setup(&mut r);
    r.vram[0] = 0x5a00_0000;
    flat(&mut r, 0, 0, 1, 0x33);
    assert_eq!(px(&r, 0, 0), 0x5a00_0033);
}

#[test]
fn rop_xor() {
    let mut r = bare_re3();
    setup(&mut r);
    r.vram[0] = 0x0f;
    r.write_reg(REG_FUNC, 6);
    flat(&mut r, 0, 0, 1, 0xff);
    assert_eq!(px(&r, 0, 0), 0xf0);
}

#[test]
fn wid_test_rejects_other_windows() {
    let mut r = bare_re3();
    setup(&mut r);
    r.vram[0] = 0x3000_0000; // cid 3
    r.vram[1] = 0x5000_0000; // cid 5
    r.write_reg(REG_ENABWID, 1);
    r.write_reg(REG_FBOPTION, 1); // 4 WID planes
    r.write_reg(REG_CURWID, 3);
    flat(&mut r, 0, 0, 2, 0x7);
    assert_eq!(px(&r, 0, 0), 0x3000_0007);
    assert_eq!(px(&r, 1, 0), 0x5000_0000);
}

#[test]
fn shaded_span_steps_colour() {
    let mut r = bare_re3();
    setup(&mut r);
    r.write_reg(REG_X, 0);
    r.write_reg(REG_Y, 0);
    r.write_reg(REG_DX, 1 << 14);
    r.write_reg(REG_R, 10 << 11);
    r.write_reg(REG_DR, 1 << 11);
    r.write_reg(REG_NUMPIX, 4);
    r.write_reg(REG_IR, IR_SHADED);
    let got: Vec<u32> = (0..4).map(|x| px(&r, x, 0)).collect();
    assert_eq!(got, vec![10, 11, 12, 13]);
}

#[test]
fn copy_rect_handles_overlap_both_ways() {
    let mut r = bare_re3();
    setup(&mut r);
    for y in 0..8 {
        r.vram[y * FB_W] = y as u32 + 1;
    }
    // Move rows 0..6 up by 2 (overlapping), like a textport scroll.
    r.copy_rect(0, 0, 1, 6, 0, 2);
    let col: Vec<u32> = (0..8).map(|y| px(&r, 0, y)).collect();
    assert_eq!(col, vec![1, 2, 1, 2, 3, 4, 5, 6]);
    // And back down.
    r.copy_rect(0, 2, 1, 6, 0, 0);
    let col: Vec<u32> = (0..8).map(|y| px(&r, 0, y)).collect();
    assert_eq!(col, vec![1, 2, 3, 4, 5, 6, 5, 6]);
}

#[test]
fn writebuf_then_readbuf_stream() {
    let mut r = bare_re3();
    setup(&mut r);
    r.write_reg(REG_X, 4);
    r.write_reg(REG_Y, 3);
    r.write_reg(REG_DX, 1 << 14);
    r.write_reg(REG_NUMPIX, 3);
    r.write_reg(REG_IR, IR_WRITEBUF);
    for v in [0x11, 0x22, 0x33] {
        r.write_reg(REG_RWDATA, v);
    }
    assert_eq!((px(&r, 4, 3), px(&r, 5, 3), px(&r, 6, 3)), (0x11, 0x22, 0x33));
    assert_eq!(r.ctx.stream, STREAM_IDLE);

    r.write_reg(REG_X, 4);
    r.write_reg(REG_NUMPIX, 3);
    r.write_reg(REG_IR, IR_READBUF);
    let mut got = vec![];
    for _ in 0..3 {
        got.push(r.ctx.reg[REG_RWDATA]);
        r.read_buffer();
    }
    assert_eq!(got, vec![0x11, 0x22, 0x33]);
    assert_eq!(r.ctx.stream, STREAM_IDLE);
}

// ── Through the bus: CPU register window -> RE3 FIFO -> RE3 thread ─────────

pub(super) fn live_gr2(variant: Gr2Variant) -> &'static Gr2 {
    let g = Gr2::new(variant, Gr2Stats {
        heartbeat: Arc::new(AtomicU64::new(0)),
        fasttick: Arc::new(AtomicU64::new(0)),
    });
    let g: &'static Gr2 = Box::leak(Box::new(g));
    g.start();
    g
}

/// Store, retrying on bus back-pressure like the CPU does.
pub(super) fn w32(g: &Gr2, off: u32, val: u32) {
    while g.write32(GR2_BASE + off, val) == BUS_BUSY {
        std::hint::spin_loop();
    }
}

pub(super) fn r32(g: &Gr2, off: u32) -> u32 {
    loop {
        let r = g.read32(GR2_BASE + off);
        if r.status != BUS_BUSY {
            return r.data;
        }
    }
}

fn re3_reg(reg: usize) -> u32 {
    if reg < 0x20 { 0x6c200 + reg as u32 * 4 } else { 0x6c280 + (reg as u32 - 0x20) * 4 }
}

#[test]
fn cpu_register_window_draws_and_reads_back() {
    let g = live_gr2(Gr2Variant::Xz);
    w32(g, re3_reg(REG_FUNC), ROP_COPY);
    w32(g, re3_reg(REG_PIXMASK), 0xffffff);
    w32(g, re3_reg(REG_XMAX), 1279);
    w32(g, re3_reg(REG_YMAX), 1023);
    w32(g, re3_reg(REG_R), 0x5 << 11);
    w32(g, re3_reg(REG_YX), (20 << 12) | (30));
    w32(g, re3_reg(REG_NUMPIX), 2);
    w32(g, re3_reg(REG_IR), IR_FLAT);
    g.wait_idle();
    assert_eq!(g.vram()[20 * FB_W + 30], 5);
    assert_eq!(g.vram()[20 * FB_W + 31], 5);

    // READBUF streamed through CPU RWDATA reads.
    w32(g, re3_reg(REG_RWMODE), RWMODE_FB);
    w32(g, re3_reg(REG_DX), 1 << 14);
    w32(g, re3_reg(REG_YX), (20 << 12) | (29));
    w32(g, re3_reg(REG_NUMPIX), 3);
    w32(g, re3_reg(REG_IR), IR_READBUF);
    let got: Vec<u32> = (0..3).map(|_| r32(g, re3_reg(REG_RWDATA))).collect();
    assert_eq!(got, vec![0, 5, 5]);
}
