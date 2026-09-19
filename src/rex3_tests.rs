// REX3 unit tests — ported from SGI NEWPORT IDE diagnostics (rex3.c, vram3.c, minigl3.c)
//
// Strategy: construct Rex3 via Arc, call start() to launch the real GFIFO processor thread.
// Write registers via write32() through the BusDevice path (real GFIFO queue).
// Read registers via read32(); HOSTRW reads block until the draw thread produces a pixel.
// Inspect framebuffers directly via unsafe { &*rex3.fb_rgb.get() }.
//
// Coordinate encoding: XYSTARTI packs (x+COORD_BIAS)<<16 | (y+COORD_BIAS) as u16s.
// Framebuffer index: fb_rgb[y * 2048 + x] for screen coordinate (x, y).

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use crate::traits::{BusRead32, BusRead64};
use super::*;

// ---------------------------------------------------------------------------
// Test harness helpers
// ---------------------------------------------------------------------------

/// Build a running Rex3 with the GFIFO processor thread started.
/// Uses Box::leak to get a 'static reference — memory is reclaimed by the OS after the test
/// process exits. The processor thread also holds a 'static ref via start()'s transmute.
/// Construction runs on a thread with a 64MB stack: Rex3 is large enough to overflow the
/// default Rust test thread stack (2MB), and Rex3Context now embeds the 1 MiB HOSTRW
/// data-port array, which construction moves through several temporaries.
fn make_rex3() -> &'static Rex3 {
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(|| {
            let rex = Box::leak(Box::new(Rex3::new(
                Arc::new(AtomicU64::new(0)),
                Arc::new(AtomicU64::new(0)),
                Arc::new(AtomicU64::new(0)),
                Arc::new(AtomicU64::new(0)),
                Arc::new(AtomicU64::new(0)),
                Arc::new(AtomicU64::new(0)),
            )));
            unsafe {
                (*rex.fb_rgb.get()).fill(0);
                (*rex.fb_aux.get()).fill(0);
            }
            #[cfg(feature = "rex-jit")]
            rex.jit_enabled.store(false, std::sync::atomic::Ordering::Relaxed);
            rex.start();
            rex
        })
        .expect("spawn")
        .join()
        .expect("make_rex3 thread panicked")
}

// Compute the SET address (no GO bit) for a register offset.
fn set_addr(offset: u32) -> u32 { REX3_BASE | offset }
// Compute the GO address (bit 11 set) for a register offset.
fn go_addr(offset: u32) -> u32  { REX3_BASE | 0x0800 | offset }

/// Bus write that retries on `BUS_BUSY`, the way the CPU re-executes a store
/// on `EXEC_RETRY`. `Rex3::write32` returns `BUS_BUSY` when the GFIFO is
/// full; discarding that status silently drops the entry, which turned the
/// throughput benchmarks below into a measurement of how fast a host thread
/// can *attempt* pushes (a flat ~18M spans/s at every span length, i.e.
/// "22 Gpx/s" for 1280-pixel Gouraud spans).
fn w32(rex: &Rex3, addr: u32, value: u32) {
    while rex.write32(addr, value) == crate::traits::BUS_BUSY {
        std::hint::spin_loop();
    }
}

/// 64-bit counterpart of `w32`.
fn w64(rex: &Rex3, addr: u32, value: u64) {
    while rex.write64(addr, value) == crate::traits::BUS_BUSY {
        std::hint::spin_loop();
    }
}

/// Write a register to the SET space (no draw trigger).
fn reg(rex: &Rex3, offset: u32, value: u32) {
    w32(rex, set_addr(offset), value);
}

/// Write a register to the GO space (triggers a draw), then wait for idle.
/// Equivalent to writing to go.reg + REX3WAIT(REX) in SGI diagnostics.
fn reg_go(rex: &Rex3, offset: u32, value: u32) {
    w32(rex, go_addr(offset), value);
    rex.wait_idle();
}

/// Wait for the GFIFO processor to drain (equivalent to REX3WAIT).
fn wait(rex: &Rex3) {
    rex.wait_idle();
}

/// Read a 32-bit context register.  Blocks until the GFIFO is idle first.
fn read_reg(rex: &Rex3, offset: u32) -> u32 {
    // Retry on busy, don't panic. 67 registers are gated behind `busy_or_val!`,
    // which returns busy whenever gfxbusy is set or the GFIFO is non-empty — so
    // `wait_idle()` first is necessary but not sufficient: nothing stops the
    // queue refilling between the wait and the read. Retrying matches what
    // read_hostrw32 and friends already do; panicking here would surface a
    // transient queue state as a bogus "bad status" failure.
    rex.wait_idle();
    loop {
        let r: BusRead32 = rex.read32(set_addr(offset));
        if r.is_ok() { return r.data; }
        if r.status != crate::traits::BUS_BUSY {
            panic!("read_reg: bad status {:#x} for offset {offset:#x}", r.status);
        }
        std::hint::spin_loop();
    }
}

/// Read from HOSTRW0 GO space: returns current word, then triggers next batch.
/// Use for all reads except the final one in a sequence.
fn read_hostrw32(rex: &Rex3) -> u32 {
    loop {
        let r: BusRead32 = rex.read32(go_addr(REX3_HOSTRW0));
        if r.is_ok() { return r.data; }
        std::hint::spin_loop();
    }
}

/// Read from HOSTRW0 SET space: returns current word, does NOT trigger next batch.
/// Use for the final read in a HOSTR sequence.
fn read_hostrw32_last(rex: &Rex3) -> u32 {
    loop {
        let r: BusRead32 = rex.read32(set_addr(REX3_HOSTRW0));
        if r.is_ok() { return r.data; }
        std::hint::spin_loop();
    }
}

/// Read from HOSTRW0 GO space (64-bit): returns current word, triggers next batch.
fn read_hostrw64(rex: &Rex3) -> u64 {
    loop {
        let r: BusRead64 = rex.read64(go_addr(REX3_HOSTRW0));
        if r.is_ok() { return r.data; }
        std::hint::spin_loop();
    }
}

/// Read from HOSTRW0 SET space (64-bit): returns current word, no next-batch trigger.
fn read_hostrw64_last(rex: &Rex3) -> u64 {
    loop {
        let r: BusRead64 = rex.read64(set_addr(REX3_HOSTRW0));
        if r.is_ok() { return r.data; }
        std::hint::spin_loop();
    }
}

/// Write a 32-bit word to HOSTRW0 (CPU→REX draw path).
fn write_hostrw32(rex: &Rex3, val: u32) {
    w32(rex, go_addr(REX3_HOSTRW0), val);
}

/// Write a 64-bit double to HOSTRW0 (CPU→REX draw path, 64-bit GIO bus).
fn write_hostrw64(rex: &Rex3, val: u64) {
    w64(rex, go_addr(REX3_HOSTRW0), val);
}

/// Read fb_rgb pixel at screen (x, y) — direct framebuffer access for verification.
fn read_pixel(rex: &Rex3, x: i32, y: i32) -> u32 {
    unsafe { (*rex.fb_rgb.get())[(y as u32 * 2048 + x as u32) as usize] }
}

/// Encode (x, y) for XYSTARTI/XYENDI: add COORD_BIAS, pack as (x<<16 | y).
fn xy(x: i32, y: i32) -> u32 {
    let xi = (x + REX3_COORD_BIAS) as u16 as u32;
    let yi = (y + REX3_COORD_BIAS) as u16 as u32;
    (xi << 16) | yi
}

// ---------------------------------------------------------------------------
// Fractional (sub-pixel) coordinate helpers for F_LINE/A_LINE.
//
// Use the plain XSTART/YSTART/XEND/YEND registers (16.4(7) format, written
// via from16_4_7 which sign-extends through Rex3RegisterOps::rexset — see
// rex3.rs) rather than the GL-fast-path XSTARTF/YSTARTF/XENDF/YENDF
// aliases. Per the REX3 spec (rex3_pdf.md table 7): "XSTARTF ... GL version
// of XSTART, (zeros 4 msbs)" — XSTARTF is a *narrower* register that zeros
// the top bits and cannot represent REX3_COORD_BIAS-shifted (4096-biased)
// screen coordinates at all (confirmed experimentally: writing a biased
// value through XSTARTF truncates the bias away, then calculate_fb_address
// subtracts COORD_BIAS *again* on top of the now-unbiased value, sending
// every pixel address far out of bounds — nothing gets drawn). Plain
// XSTART/YSTART/XEND/YEND round-trip biased values correctly (confirmed:
// from16_4_7(biased_val) recovers the exact screen coordinate on readback),
// matching how XYSTARTI/XYENDI already bias internally in xy().
//
// `frac4` is a 0-15 nibble in 1/16-pixel units — the only fractional
// precision the interpreter's fline_apply_fract/draw_aline ever consult
// (bits [10:7] of the 21.11 value). `screen_x`/`screen_y` are plain screen
// pixel coordinates (same convention as xy()'s x/y params); REX3_COORD_BIAS
// is added internally so these stay bias-consistent with any other
// XYSTARTI/XYENDI-driven endpoint on the same line.
// ---------------------------------------------------------------------------

fn write_xstartf(rex: &Rex3, screen_x: i32, frac4: i32) {
    let biased = screen_x + REX3_COORD_BIAS;
    reg(rex, REX3_XSTART, ((biased << 11) | (frac4 << 7)) as u32);
}
fn write_ystartf(rex: &Rex3, screen_y: i32, frac4: i32) {
    let biased = screen_y + REX3_COORD_BIAS;
    reg(rex, REX3_YSTART, ((biased << 11) | (frac4 << 7)) as u32);
}
fn write_xendf(rex: &Rex3, screen_x: i32, frac4: i32) {
    let biased = screen_x + REX3_COORD_BIAS;
    reg(rex, REX3_XEND, ((biased << 11) | (frac4 << 7)) as u32);
}
fn write_yendf(rex: &Rex3, screen_y: i32, frac4: i32) {
    let biased = screen_y + REX3_COORD_BIAS;
    reg(rex, REX3_YEND, ((biased << 11) | (frac4 << 7)) as u32);
}

// ============================================================================
// DRAWMODE constants (matching minigl3.c / SGI headers)
// ============================================================================

// DRAWMODE1 combinations. COMPARE is included at its disabled value (0x7) — real
// drawmode1 words always carry it explicitly; omitting it would leave COMPARE=0
// (afunction always-kill) since DrawMode1's bitfield default zero-inits.
const DM1_CI8_SRC: u32   = DRAWMODE1_PLANES_RGB | (1 << 3) | DRAWMODE1_COMPARE_DISABLE_SH | DRAWMODE1_LOGICOP_SRC_SH;
const DM1_RGB24_SRC: u32 = DRAWMODE1_PLANES_RGB | (3 << 3) | (1 << 15) | DRAWMODE1_COMPARE_DISABLE_SH | DRAWMODE1_LOGICOP_SRC_SH;

// DRAWMODE1 host-depth fields (bits [4:3] = hostdepth, bit 16 = rwpacked, bit 17 = rwdouble)
// hostdepth: 0=4bpp, 1=8bpp, 2=12bpp, 3=32bpp
const DM1_HOSTDEPTH8:  u32 = 1 << 3;   // hostdepth=1 → 8bpp packed
const DM1_HOSTDEPTH32: u32 = 3 << 3;   // hostdepth=3 → 32bpp
const DM1_RWPACKED:    u32 = 1 << 16;  // pack multiple pixels per word
const DM1_RWDOUBLE:    u32 = 1 << 17;  // 64-bit GIO bus transfers

// DRAWMODE0 base combinations (stoponx=bit8, stopony=bit9)
const DM0_STOPONX:    u32 = 1 << 8;
const DM0_STOPONY:    u32 = 1 << 9;
const DM0_STOPONXY:   u32 = DM0_STOPONX | DM0_STOPONY;
const DM0_DOSETUP:    u32 = 1 << 5;
const DM0_COLORHOST:  u32 = 1 << 6;  // pixel data comes from / goes to host FIFO (bit 6)

const DM0_DRAW_BLOCK:  u32 = DRAWMODE0_OPCODE_DRAW | DRAWMODE0_ADRMODE_BLOCK_SH | DM0_STOPONXY;
const DM0_DRAW_SPAN:   u32 = DRAWMODE0_OPCODE_DRAW | DRAWMODE0_ADRMODE_SPAN_SH  | DM0_STOPONX;
const DM0_SCR2SCR:     u32 = DRAWMODE0_OPCODE_SCR2SCR | DRAWMODE0_ADRMODE_BLOCK_SH | DM0_DOSETUP | DM0_STOPONXY;
// DRAW with COLORHOST: pixels come from host write FIFO
const DM0_HOSTW_BLOCK: u32 = DRAWMODE0_OPCODE_DRAW | DRAWMODE0_ADRMODE_BLOCK_SH | DM0_STOPONXY | DM0_COLORHOST;
// READ with COLORHOST: reads fb → host read FIFO
const DM0_READ_BLOCK:  u32 = DRAWMODE0_OPCODE_READ | DRAWMODE0_ADRMODE_BLOCK_SH | DM0_STOPONXY | DM0_COLORHOST | DM0_DOSETUP;

/// Initialise REX3 to a known baseline — matches rex3init() from rex3.c.
/// XYWIN is left at 0 (no hardware xbias correction needed in emulation).
/// All CID mask bits set permit every window ID; screen masks are disabled.
fn rex3init(rex: &Rex3) {
    reg(rex, REX3_LSMODE,      0);
    reg(rex, REX3_LSPATTERN,   0);
    reg(rex, REX3_LSPATSAVE,   0);
    reg(rex, REX3_ZPATTERN,    0);
    reg(rex, REX3_COLORBACK,   0xDEADBEEF);
    reg(rex, REX3_COLORVRAM,   0xFFFFFF);
    reg(rex, REX3_SMASK0X,     0);
    reg(rex, REX3_SMASK0Y,     0);
    reg(rex, REX3_XSAVE,       0);
    reg(rex, REX3_XYMOVE,      0);
    reg(rex, REX3_BRESD,       0);
    reg(rex, REX3_BRESS1,      0);
    reg(rex, REX3_BRESOCTINC1, 0);
    reg(rex, REX3_BRESRNDINC2, 0);
    reg(rex, REX3_BRESE1,      0);
    reg(rex, REX3_BRESS2,      0);
    reg(rex, REX3_AWEIGHT0,    0);
    reg(rex, REX3_AWEIGHT1,    0);
    reg(rex, REX3_COLORRED,    0);
    reg(rex, REX3_COLORALPHA,  0);
    reg(rex, REX3_WRMASK,      0xFFFFFF);
    reg(rex, REX3_SMASK1X,     0);
    reg(rex, REX3_SMASK1Y,     0);
    reg(rex, REX3_SMASK2X,     0);
    reg(rex, REX3_SMASK2Y,     0);
    reg(rex, REX3_SMASK3X,     0);
    reg(rex, REX3_SMASK3Y,     0);
    reg(rex, REX3_SMASK4X,     0);
    reg(rex, REX3_SMASK4Y,     0);
    reg(rex, REX3_XYWIN,       0);
    reg(rex, REX3_TOPSCAN,     0x3FF);
    reg(rex, REX3_CLIPMODE,    0xF << CLIPMODE_CIDMATCH_SHIFT);
    wait(rex);
}

// CIDMATCH is a set of permitted two-bit CIDs, not a four-bit equality
// comparison. Popup bits and the other auxiliary lanes must not affect it.
fn check_cid_write_masks(rex: &Rex3, compiled: bool) {
    rex3init(rex);
    #[cfg(feature = "rex-jit")]
    let jit_before = rex.jit_go_count.load(std::sync::atomic::Ordering::Relaxed);
    let src = 10 * 2048 + 10;
    let dst = 20 * 2048 + 10;
    for dm0 in [DM0_DRAW_BLOCK, DRAWMODE0_OPCODE_DRAW | DRAWMODE0_ADRMODE_I_LINE_SH | DM0_DOSETUP | DM0_STOPONXY, DM0_SCR2SCR] {
        for mask in 0..16_u32 {
            let cm = mask << CLIPMODE_CIDMATCH_SHIFT;
            #[cfg(feature = "rex-jit")]
            if compiled {
                let jit = rex.rex_jit.as_ref().unwrap();
                jit.request_compile(dm0, DM1_RGB24_SRC, cm);
                assert!(jit.wait_compiled(dm0, DM1_RGB24_SRC, cm));
            }
            #[cfg(not(feature = "rex-jit"))]
            assert!(!compiled);
            for cid in 0..4_u32 {
                for popup in 0..4_u32 {
                    wait(rex);
                    unsafe {
                        (*rex.fb_rgb.get())[src] = 0x123456;
                        (*rex.fb_rgb.get())[dst] = 0xabcdef;
                        (*rex.fb_aux.get())[dst] = 0x80000000 | (cid << 4) | (popup << 2) | cid;
                    }
                    reg(rex, REX3_DRAWMODE1, DM1_RGB24_SRC);
                    reg(rex, REX3_COLORI, 0x123456);
                    reg(rex, REX3_CLIPMODE, cm);
                    let copying = dm0 == DM0_SCR2SCR;
                    reg(rex, REX3_XYMOVE, if copying { 10 } else { 0 });
                    reg(rex, REX3_XYSTARTI, xy(10, if copying { 10 } else { 20 }));
                    reg(rex, REX3_XYENDI, xy(10, if copying { 10 } else { 20 }));
                    reg_go(rex, REX3_DRAWMODE0, dm0);
                    let expected = if mask & (1 << cid) != 0 { 0x123456 } else { 0xabcdef };
                    assert_eq!(read_pixel(rex, 10, 20), expected,
                        "dm0={dm0:#x} CIDMATCH={mask:04b} cid={cid} popup={popup}");
                }
            }
        }
    }
    #[cfg(feature = "rex-jit")]
    assert_eq!(rex.jit_go_count.load(std::sync::atomic::Ordering::Relaxed) - jit_before,
        if compiled { 768 } else { 0 });
}

#[test]
fn cid_write_masks_interpreter() {
    let rex = make_rex3();
    #[cfg(feature = "rex-jit")]
    rex.jit_enabled.store(false, std::sync::atomic::Ordering::Relaxed);
    check_cid_write_masks(rex, false);
    rex.stop();
}

// ============================================================================
// Tests ported from SGI rex3.c: test_rex3() — register read/write
// ============================================================================

/// Port of SGI test_rex3(): write walking patterns to every context register,
/// read back and verify the expected masked value.
/// Covers: lsmode, lspattern, lspatsave, zpattern, colorback, colorvram, alpharef,
///         smask0x/y, xsave, xymove, bresd, bress1, bresoctinc1, bresrndinc2,
///         brese1, bress2, aweight0/1, colorred, coloralpha, colorgrn, colorblue,
///         wrmask, smask1-4 x/y, topscan, xywin, clipmode, xstarti→xstart readback.
#[test]
fn test_rex3_register_rw() {
    let rex = make_rex3();
    rex3init(&rex);

    // Walking patterns: 0x00000000, 0x55555555, 0xAAAAAAAA, 0xFFFFFFFF
    // (SGI uses i*0x55555555 for i in 0..=3)
    for &pattern in &[0x00000000u32, 0x55555555, 0xAAAAAAAA, 0xFFFFFFFF] {

        // Helper: write, read back, check (data & mask) == got
        let check = |offset: u32, mask: u32| {
            reg(&rex, offset, pattern);
            let got = read_reg(&rex, offset);
            let expect = pattern & mask;
            assert_eq!(got, expect,
                "reg {offset:#06x} pattern={pattern:#010x}: got {got:#010x} expected {expect:#010x}");
        };

        check(REX3_LSMODE,      0x0FFFFFFF); // 28-bit
        check(REX3_LSPATTERN,   0xFFFFFFFF);
        check(REX3_LSPATSAVE,   0xFFFFFFFF);
        check(REX3_ZPATTERN,    0xFFFFFFFF);
        check(REX3_COLORBACK,   0xFFFFFFFF);
        check(REX3_COLORVRAM,   0xFFFFFFFF);
        check(REX3_ALPHAREF,    0xFF);       // 8-bit
        check(REX3_SMASK0X,     0xFFFFFFFF);
        check(REX3_SMASK0Y,     0xFFFFFFFF);
        check(REX3_XYMOVE,      0xFFFFFFFF);
        check(REX3_BRESD,       0x7FFFFFF);  // 27-bit
        check(REX3_BRESS1,      0x1FFFF);    // 17-bit
        check(REX3_BRESOCTINC1, 0x7FFFFFF & !(0xF << 20));
        check(REX3_BRESRNDINC2, 0xFFFFFFFF & !(0x7 << 21));
        check(REX3_BRESE1,      0xFFFF);     // 16-bit
        check(REX3_BRESS2,      0x3FFFFFF);  // 26-bit
        check(REX3_AWEIGHT0,    0xFFFFFFFF);
        check(REX3_AWEIGHT1,    0xFFFFFFFF);
        check(REX3_COLORRED,    0xFFFFFF);   // 24-bit
        check(REX3_COLORALPHA,  0xFFFFF);    // 20-bit
        check(REX3_COLORGRN,    0xFFFFF);    // 20-bit
        check(REX3_COLORBLUE,   0xFFFFF);    // 20-bit
        check(REX3_WRMASK,      0xFFFFFF);   // 24-bit
        check(REX3_SMASK1X,     0xFFFFFFFF);
        check(REX3_SMASK1Y,     0xFFFFFFFF);
        check(REX3_SMASK2X,     0xFFFFFFFF);
        check(REX3_SMASK2Y,     0xFFFFFFFF);
        check(REX3_SMASK3X,     0xFFFFFFFF);
        check(REX3_SMASK3Y,     0xFFFFFFFF);
        check(REX3_SMASK4X,     0xFFFFFFFF);
        check(REX3_SMASK4Y,     0xFFFFFFFF);
        check(REX3_TOPSCAN,     0x3FF);      // 10-bit
        check(REX3_XYWIN,       0xFFFFFFFF);
        check(REX3_CLIPMODE,    0x1FFF);     // 13-bit

        // SGI TIW tests: XSTARTI (integer) writes into _xstart (fixed-point).
        // Writing integer x to XSTARTI stores x<<11 in xstart (I21F11).
        // Mask: lower 16 bits of pattern as i16, shifted <<11.
        {
            let xi = (pattern & 0xFFFF) as i16 as i32;
            reg(&rex, REX3_XSTARTI, pattern);
            let got = read_reg(&rex, REX3_XSTART);
            let expect = (xi << 11) as u32 & (0xFFFFF << 7);
            assert_eq!(got, expect,
                "XSTARTI→XSTART pattern={pattern:#010x}: got {got:#010x} expected {expect:#010x}");
        }

        // XYSTARTI packs x and y; both should appear in xstart and ystart.
        {
            let xi = ((pattern >> 16) & 0xFFFF) as i16 as i32;
            let yi = (pattern & 0xFFFF) as i16 as i32;
            reg(&rex, REX3_XYSTARTI, pattern);
            let xgot = read_reg(&rex, REX3_XSTART);
            let ygot = read_reg(&rex, REX3_YSTART);
            let xexp = (xi << 11) as u32 & (0xFFFFF << 7);
            let yexp = (yi << 11) as u32 & (0xFFFFF << 7);
            assert_eq!(xgot, xexp,
                "XYSTARTI→XSTART pattern={pattern:#010x}: got {xgot:#010x} expected {xexp:#010x}");
            assert_eq!(ygot, yexp,
                "XYSTARTI→YSTART pattern={pattern:#010x}: got {ygot:#010x} expected {yexp:#010x}");
        }

        // XYENDI packs x and y into xend/yend.
        {
            let xi = ((pattern >> 16) & 0xFFFF) as i16 as i32;
            let yi = (pattern & 0xFFFF) as i16 as i32;
            reg(&rex, REX3_XYENDI, pattern);
            let xgot = read_reg(&rex, REX3_XEND);
            let yexp = (yi << 11) as u32 & (0xFFFFF << 7);
            let xexp = (xi << 11) as u32 & (0xFFFFF << 7);
            assert_eq!(xgot, xexp,
                "XYENDI→XEND pattern={pattern:#010x}: got {xgot:#010x} expected {xexp:#010x}");
            let ygot = read_reg(&rex, REX3_YEND);
            assert_eq!(ygot, yexp,
                "XYENDI→YEND pattern={pattern:#010x}: got {ygot:#010x} expected {yexp:#010x}");
        }
    }

    rex3init(&rex); // restore
}

// ============================================================================
// Tests ported from SGI vram3.c: ng1test_vram() — fill and readback via fb_rgb
// ============================================================================

/// Port of SGI ng1test_vram() solid fill + readback.
/// Fills the framebuffer with a pattern color, then reads it back directly
/// from fb_rgb (in-emulator we skip the hostrw FIFO path and read memory directly).
#[test]
fn test_vram_fill_readback_ci8() {
    let rex = make_rex3();
    rex3init(&rex);

    // CI 8-bit mode: test patterns 0x00, 0x55, 0xAA, 0xFF
    for &color in &[0x00u8, 0x55, 0xAA, 0xFF] {
        // Fill entire (small) region
        let (x0, y0, x1, y1) = (0i32, 0i32, 63i32, 15i32);
        reg(&rex, REX3_DRAWMODE1, DM1_CI8_SRC);
        reg(&rex, REX3_WRMASK, 0xFF);
        reg(&rex, REX3_COLORI, color as u32);
        reg(&rex, REX3_XYENDI,   xy(x1, y1));
        reg(&rex, REX3_XYSTARTI, xy(x0, y0));
        reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);

        // Verify every pixel in the region has the correct 8-bit value
        let mut errors = 0;
        for y in y0..=y1 {
            for x in x0..=x1 {
                let px = read_pixel(&rex, x, y);
                // In CI8, write_rgb_8 stores value in bits 7:0 of each pixel group.
                // The actual value stored depends on pixel packing — check low byte.
                if (px & 0xFF) != color as u32 {
                    errors += 1;
                }
            }
        }
        assert_eq!(errors, 0,
            "CI8 fill with {color:#04x}: {errors} pixels mismatched");
    }
}

/// Walking-ones VRAM test (port of vram3.c walking 1's section), CI8 mode.
#[test]
fn test_vram_walking_ones_ci8() {
    let rex = make_rex3();
    rex3init(&rex);

    for bit in 0..8u32 {
        let color = 1u8 << bit;
        let (x0, y0, x1, y1) = (0i32, 0i32, 31i32, 7i32);

        reg(&rex, REX3_DRAWMODE1, DM1_CI8_SRC);
        reg(&rex, REX3_WRMASK, 0xFF);
        reg(&rex, REX3_COLORI, color as u32);
        reg(&rex, REX3_XYENDI,   xy(x1, y1));
        reg(&rex, REX3_XYSTARTI, xy(x0, y0));
        reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);

        let mut errors = 0;
        for y in y0..=y1 {
            for x in x0..=x1 {
                let px = read_pixel(&rex, x, y) & 0xFF;
                if px != color as u32 { errors += 1; }
            }
        }
        assert_eq!(errors, 0, "Walking-1 bit {bit} (color={color:#04x}): {errors} mismatches");
    }
}

/// VRAM test with varying colors per block (small-chunk section of vram3.c).
#[test]
fn test_vram_varying_color_blocks_ci8() {
    let rex = make_rex3();
    rex3init(&rex);

    // Test a 128×4 region with different colors per 64-pixel-wide column
    let ysize = 4i32;
    let (x0, y0) = (0i32, 0i32);
    for col in 0..2i32 {
        let x = x0 + col * 64;
        let color = ((col * 3) & 0xFF) as u8; // simple per-column color
        reg(&rex, REX3_DRAWMODE1, DM1_CI8_SRC);
        reg(&rex, REX3_WRMASK, 0xFF);
        reg(&rex, REX3_COLORI, color as u32);
        reg(&rex, REX3_XYENDI,   xy(x + 63, y0 + ysize - 1));
        reg(&rex, REX3_XYSTARTI, xy(x, y0));
        reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);

        for y in y0..y0+ysize {
            for px_x in x..x+64 {
                let px = read_pixel(&rex, px_x, y) & 0xFF;
                assert_eq!(px, color as u32,
                    "block col={col} at ({px_x},{y}): got {px:#04x} expected {color:#04x}");
            }
        }
    }
}

// ============================================================================
// Tests ported from minigl3.c: ng1_block, ng1_span, ng1_scrtoscr
// ============================================================================

/// ng1_block() in CI8 mode: verify that fill writes exactly the right rectangle.
#[test]
fn test_ng1_block_boundary() {
    let rex = make_rex3();
    rex3init(&rex);

    // Fill a background color across the test area
    reg(&rex, REX3_DRAWMODE1, DM1_CI8_SRC);
    reg(&rex, REX3_WRMASK, 0xFF);
    reg(&rex, REX3_COLORI, 0xAA);
    reg(&rex, REX3_XYENDI,   xy(15, 7));
    reg(&rex, REX3_XYSTARTI, xy(0, 0));
    reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);

    // Draw a smaller rectangle with a different color
    reg(&rex, REX3_COLORI, 0x42);
    reg(&rex, REX3_XYENDI,   xy(9, 5));
    reg(&rex, REX3_XYSTARTI, xy(4, 2));
    reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);

    // Inside the inner rectangle → 0x42
    for y in 2..=5 {
        for x in 4..=9 {
            assert_eq!(read_pixel(&rex, x, y) & 0xFF, 0x42,
                "inner pixel ({x},{y}) should be 0x42");
        }
    }
    // Outside the inner rectangle (but inside outer) → 0xAA
    assert_eq!(read_pixel(&rex, 0, 0) & 0xFF, 0xAA, "outer corner (0,0) should be 0xAA");
    assert_eq!(read_pixel(&rex, 15, 7) & 0xFF, 0xAA, "outer corner (15,7) should be 0xAA");
    assert_eq!(read_pixel(&rex, 3, 2) & 0xFF, 0xAA, "left of inner (3,2) should be 0xAA");
    assert_eq!(read_pixel(&rex, 10, 5) & 0xFF, 0xAA, "right of inner (10,5) should be 0xAA");
}

/// ng1_span(): single horizontal span draws exactly one row.
#[test]
fn test_ng1_span_one_row() {
    let rex = make_rex3();
    rex3init(&rex);

    // Clear background
    reg(&rex, REX3_DRAWMODE1, DM1_CI8_SRC);
    reg(&rex, REX3_WRMASK, 0xFF);
    reg(&rex, REX3_COLORI, 0x00);
    reg(&rex, REX3_XYENDI,   xy(19, 9));
    reg(&rex, REX3_XYSTARTI, xy(0, 0));
    reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);

    // Draw span at y=4 from x=5 to x=14 (inclusive)
    reg(&rex, REX3_COLORI, 0x77);
    reg(&rex, REX3_XYENDI,   xy(14, 4)); // xend only used for STOPONX
    reg(&rex, REX3_XYSTARTI, xy(5, 4));
    reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_SPAN);

    for x in 5..=14 {
        assert_eq!(read_pixel(&rex, x, 4) & 0xFF, 0x77, "span pixel ({x},4) should be 0x77");
    }
    // Row above and below should be untouched
    assert_eq!(read_pixel(&rex, 5, 3) & 0xFF, 0x00, "row above span should be 0");
    assert_eq!(read_pixel(&rex, 5, 5) & 0xFF, 0x00, "row below span should be 0");
    // Pixel before and after span on same row
    assert_eq!(read_pixel(&rex, 4, 4) & 0xFF, 0x00, "pixel before span start should be 0");
    assert_eq!(read_pixel(&rex, 15, 4) & 0xFF, 0x00, "pixel after span end should be 0");
}

/// ng1_scrtoscr(): copy a block from one location to another.
/// Port of the SCR2SCR path in minigl3.c/ng1_scrtoscr().
#[test]
fn test_ng1_scrtoscr() {
    let rex = make_rex3();
    rex3init(&rex);

    // Paint source block at (0,0)..(7,7) with color 0xCC
    reg(&rex, REX3_DRAWMODE1, DM1_CI8_SRC);
    reg(&rex, REX3_WRMASK, 0xFF);
    reg(&rex, REX3_COLORI, 0xCC);
    reg(&rex, REX3_XYENDI,   xy(7, 7));
    reg(&rex, REX3_XYSTARTI, xy(0, 0));
    reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);

    // Clear destination area (16,0)..(23,7) to 0
    reg(&rex, REX3_COLORI, 0x00);
    reg(&rex, REX3_XYENDI,   xy(23, 7));
    reg(&rex, REX3_XYSTARTI, xy(16, 0));
    reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);

    // SCR2SCR: copy (0,0)..(7,7) → (16,0)..(23,7) via XYMOVE=(16,0)
    reg(&rex, REX3_DRAWMODE1, DM1_CI8_SRC);
    reg(&rex, REX3_WRMASK, 0xFF);
    reg(&rex, REX3_XYMOVE, (16u32 << 16) | 0);  // x_move=16, y_move=0
    reg(&rex, REX3_XYENDI,   xy(7, 7));
    reg(&rex, REX3_XYSTARTI, xy(0, 0));
    reg_go(&rex, REX3_DRAWMODE0, DM0_SCR2SCR);

    // Destination should now contain 0xCC
    let mut errors = 0;
    for y in 0..=7 {
        for x in 16..=23 {
            let px = read_pixel(&rex, x, y) & 0xFF;
            if px != 0xCC { errors += 1; }
        }
    }
    assert_eq!(errors, 0, "SCR2SCR: {errors} destination pixels wrong (expected 0xCC)");

    // Source should be unchanged
    for y in 0..=7 {
        for x in 0..=7 {
            assert_eq!(read_pixel(&rex, x, y) & 0xFF, 0xCC,
                "SCR2SCR: source ({x},{y}) should still be 0xCC");
        }
    }
}

/// SCR2SCR with a non-zero Y offset.
#[test]
fn test_ng1_scrtoscr_y_offset() {
    let rex = make_rex3();
    rex3init(&rex);

    // Paint source (0,0)..(7,3) = color 0x33
    reg(&rex, REX3_DRAWMODE1, DM1_CI8_SRC);
    reg(&rex, REX3_WRMASK, 0xFF);
    reg(&rex, REX3_COLORI, 0x33);
    reg(&rex, REX3_XYENDI,   xy(7, 3));
    reg(&rex, REX3_XYSTARTI, xy(0, 0));
    reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);

    // SCR2SCR: copy (0,0)..(7,3) → (0,8)..(7,11) via XYMOVE=(0,8)
    reg(&rex, REX3_XYMOVE, (0u32 << 16) | 8);
    reg(&rex, REX3_XYENDI,   xy(7, 3));
    reg(&rex, REX3_XYSTARTI, xy(0, 0));
    reg_go(&rex, REX3_DRAWMODE0, DM0_SCR2SCR);

    for y in 8..=11 {
        for x in 0..=7 {
            assert_eq!(read_pixel(&rex, x, y) & 0xFF, 0x33,
                "SCR2SCR Y-offset: ({x},{y}) should be 0x33");
        }
    }
}

// ============================================================================
// Additional focused regression tests
// ============================================================================

/// Basic single-pixel CI8 draw — simplest possible drawing test.
#[test]
fn test_block_fill_single_pixel() {
    let rex = make_rex3();
    rex3init(&rex);
    reg(&rex, REX3_DRAWMODE1, DM1_CI8_SRC);
    reg(&rex, REX3_WRMASK, 0xFF);
    reg(&rex, REX3_COLORI, 0xAB);
    reg(&rex, REX3_XYENDI,   xy(10, 20));
    reg(&rex, REX3_XYSTARTI, xy(10, 20));
    reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);
    assert_eq!(read_pixel(&rex, 10, 20) & 0xFF, 0xAB);
}

/// WRMASK=0 must block all writes.
#[test]
fn test_wrmask_zero_blocks_write() {
    let rex = make_rex3();
    rex3init(&rex);
    reg(&rex, REX3_DRAWMODE1, DM1_CI8_SRC);
    reg(&rex, REX3_WRMASK, 0x00);
    reg(&rex, REX3_COLORI, 0xFF);
    reg(&rex, REX3_XYENDI,   xy(3, 3));
    reg(&rex, REX3_XYSTARTI, xy(3, 3));
    reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);
    assert_eq!(read_pixel(&rex, 3, 3), 0, "wrmask=0 should block all writes");
}

/// Partial write mask — only masked bits written.
#[test]
fn test_wrmask_partial() {
    let rex = make_rex3();
    rex3init(&rex);
    reg(&rex, REX3_DRAWMODE1, DM1_CI8_SRC);
    reg(&rex, REX3_WRMASK, 0x0F);  // low nibble only
    reg(&rex, REX3_COLORI, 0xFF);   // would write FF, but only 0F lands
    reg(&rex, REX3_XYENDI,   xy(1, 1));
    reg(&rex, REX3_XYSTARTI, xy(1, 1));
    reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);
    assert_eq!(read_pixel(&rex, 1, 1) & 0xFF, 0x0F);
}

/// LOGICOP_ZERO always produces 0, regardless of source color.
#[test]
fn test_logicop_zero_clears() {
    let rex = make_rex3();
    rex3init(&rex);

    // First paint with SRC
    reg(&rex, REX3_DRAWMODE1, DM1_CI8_SRC);
    reg(&rex, REX3_WRMASK, 0xFF);
    reg(&rex, REX3_COLORI, 0xDE);
    reg(&rex, REX3_XYENDI,   xy(4, 4));
    reg(&rex, REX3_XYSTARTI, xy(4, 4));
    reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);
    assert_ne!(read_pixel(&rex, 4, 4) & 0xFF, 0);

    // Clear with ZERO logicop
    let dm1_zero = DRAWMODE1_PLANES_RGB | (1 << 3) | DRAWMODE1_COMPARE_DISABLE_SH | DRAWMODE1_LOGICOP_ZERO_SH;
    reg(&rex, REX3_DRAWMODE1, dm1_zero);
    reg(&rex, REX3_COLORI, 0xFF);
    reg(&rex, REX3_XYENDI,   xy(4, 4));
    reg(&rex, REX3_XYSTARTI, xy(4, 4));
    reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);
    assert_eq!(read_pixel(&rex, 4, 4) & 0xFF, 0, "LOGICOP_ZERO should write 0");
}

/// XOR twice with same color returns to zero.
#[test]
fn test_logicop_xor_roundtrip() {
    let rex = make_rex3();
    rex3init(&rex);
    let dm1_xor = DRAWMODE1_PLANES_RGB | (1 << 3) | DRAWMODE1_COMPARE_DISABLE_SH | DRAWMODE1_LOGICOP_XOR_SH;

    reg(&rex, REX3_DRAWMODE1, dm1_xor);
    reg(&rex, REX3_WRMASK, 0xFF);
    reg(&rex, REX3_COLORI, 0x55);
    reg(&rex, REX3_XYENDI,   xy(2, 2));
    reg(&rex, REX3_XYSTARTI, xy(2, 2));
    reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);
    assert_ne!(read_pixel(&rex, 2, 2) & 0xFF, 0, "first XOR should be non-zero");

    reg(&rex, REX3_XYENDI,   xy(2, 2));
    reg(&rex, REX3_XYSTARTI, xy(2, 2));
    reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);
    assert_eq!(read_pixel(&rex, 2, 2) & 0xFF, 0, "XOR twice should return to 0");
}

/// Draw at the rightmost valid screen column.
#[test]
fn test_draw_at_right_edge() {
    let rex = make_rex3();
    rex3init(&rex);
    let x = REX3_SCREEN_WIDTH - 1;
    reg(&rex, REX3_DRAWMODE1, DM1_CI8_SRC);
    reg(&rex, REX3_WRMASK, 0xFF);
    reg(&rex, REX3_COLORI, 0x42);
    reg(&rex, REX3_XYENDI,   xy(x, 0));
    reg(&rex, REX3_XYSTARTI, xy(x, 0));
    reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);
    assert_eq!(read_pixel(&rex, x, 0) & 0xFF, 0x42);
}

/// Draw one past the right edge — must not panic, pixel stays unwritten.
#[test]
fn test_draw_past_right_edge_clipped() {
    let rex = make_rex3();
    rex3init(&rex);
    reg(&rex, REX3_DRAWMODE1, DM1_CI8_SRC);
    reg(&rex, REX3_WRMASK, 0xFF);
    reg(&rex, REX3_COLORI, 0x77);
    let x = REX3_SCREEN_WIDTH;
    reg(&rex, REX3_XYENDI,   xy(x, 0));
    reg(&rex, REX3_XYSTARTI, xy(x, 0));
    reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);
    // No assert — just no panic
}

/// NOOP opcode draws nothing even with valid coordinates.
#[test]
fn test_noop_opcode_draws_nothing() {
    let rex = make_rex3();
    rex3init(&rex);
    reg(&rex, REX3_DRAWMODE1, DM1_CI8_SRC);
    reg(&rex, REX3_WRMASK, 0xFF);
    reg(&rex, REX3_COLORI, 0xFF);
    reg(&rex, REX3_XYENDI,   xy(5, 5));
    reg(&rex, REX3_XYSTARTI, xy(0, 0));
    let dm0_noop = DRAWMODE0_OPCODE_NOOP | DRAWMODE0_ADRMODE_BLOCK_SH | DM0_STOPONXY;
    reg_go(&rex, REX3_DRAWMODE0, dm0_noop);
    for y in 0..=5 {
        for x in 0..=5 {
            assert_eq!(read_pixel(&rex, x, y), 0, "NOOP should write no pixels");
        }
    }
}

/// XYSTARTI/XYENDI updates xstart/ystart and xend/yend in context.
#[test]
fn test_register_state_update() {
    let rex = make_rex3();
    rex3init(&rex);
    reg(&rex, REX3_DRAWMODE1, DM1_CI8_SRC);
    reg(&rex, REX3_WRMASK, 0xDEADBEEF & 0xFFFFFF);
    reg(&rex, REX3_XYMOVE, 0x00030004);
    wait(&rex);
    assert_eq!(read_reg(&rex, REX3_DRAWMODE1), DM1_CI8_SRC);
    assert_eq!(read_reg(&rex, REX3_WRMASK), 0xDEADBEEF & 0xFFFFFF);
    assert_eq!(read_reg(&rex, REX3_XYMOVE), 0x00030004);
}

// ============================================================================
// Tests ported from SGI rex3patterns.c: ng1bars, ng1patterns
// ============================================================================

/// Port of ng1bars(): CI8 vertical color bars with per-column color index.
/// Fills N equal-width columns with color i % 256, verifies center pixel of each.
#[test]
fn test_ng1bars_ci8() {
    let rex = make_rex3();
    rex3init(&rex);

    // Use 4 columns of width 8 for speed
    let width = 8i32;
    let num_bars = 4;
    let ysize = 15i32;

    reg(&rex, REX3_DRAWMODE1, DM1_CI8_SRC);
    reg(&rex, REX3_WRMASK, 0xFF);

    for i in 0..num_bars {
        let x = i * width;
        let color = (i % 256) as u8;
        reg(&rex, REX3_COLORI, color as u32);
        reg(&rex, REX3_XYENDI,   xy(x + width - 1, ysize - 1));
        reg(&rex, REX3_XYSTARTI, xy(x, 0));
        reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);
    }

    for i in 0..num_bars {
        let x = i * width;
        let color = (i % 256) as u32;
        // Check center pixel of each bar
        let cx = x + width / 2;
        let cy = ysize / 2;
        let px = read_pixel(&rex, cx, cy) & 0xFF;
        assert_eq!(px, color, "CI8 bar {i}: center ({cx},{cy}) got {px:#04x} expected {color:#04x}");
        // Check all pixels in bar
        for y in 0..ysize {
            for bx in x..x+width {
                let p = read_pixel(&rex, bx, y) & 0xFF;
                assert_eq!(p, color, "CI8 bar {i}: ({bx},{y}) got {p:#04x} expected {color:#04x}");
            }
        }
    }
}

/// Port of ng1patterns() solid fill tests: black, gray (128,128,128), white (255,255,255).
/// Verifies that RGB24 block fills produce the correct pixel value.
#[test]
fn test_patterns_rgb24_solid_fills() {
    let rex = make_rex3();
    rex3init(&rex);

    let (x0, y0, x1, y1) = (0i32, 0i32, 31i32, 15i32);

    // Helper: fill region in RGB24, check a few pixels.
    // Matches ng1_rgbcolor(r,g,b): colorred.word = r<<11, colorgrn.word = g<<11, colorblue.word = b<<11
    // COLORRED/GRN/BLUE are o12.11 format; integer r stored at bits [22:11].
    let test_fill = |r: u32, g: u32, b: u32| {
        reg(&rex, REX3_DRAWMODE1, DM1_RGB24_SRC);
        reg(&rex, REX3_WRMASK, 0xFFFFFF);
        reg(&rex, REX3_COLORRED,  r << 11);
        reg(&rex, REX3_COLORGRN,  g << 11);
        reg(&rex, REX3_COLORBLUE, b << 11);
        reg(&rex, REX3_XYENDI,   xy(x1, y1));
        reg(&rex, REX3_XYSTARTI, xy(x0, y0));
        reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);

        let expected = (b << 16) | (g << 8) | r;
        let mut errors = 0;
        for y in y0..=y1 {
            for x in x0..=x1 {
                let px = read_pixel(&rex, x, y) & 0xFFFFFF;
                if px != expected { errors += 1; }
            }
        }
        assert_eq!(errors, 0,
            "RGB24 fill ({r},{g},{b}): {errors} pixels wrong (expected {expected:#08x})");
    };

    test_fill(0, 0, 0);         // Black
    test_fill(128, 128, 128);   // Gray
    test_fill(255, 255, 255);   // White
}

/// Port of ng1patterns() nested block tests:
/// black background with white center block, then white background with black center.
#[test]
fn test_patterns_rgb24_nested_blocks() {
    let rex = make_rex3();
    rex3init(&rex);

    reg(&rex, REX3_DRAWMODE1, DM1_RGB24_SRC);
    reg(&rex, REX3_WRMASK, 0xFFFFFF);

    let (sw, sh) = (32i32, 16i32);  // small screen size for test

    // --- Black background + white center ---
    // ng1_rgbcolor: colorred.word = r<<11, etc. (o12.11 format, integer in bits [22:11])
    reg(&rex, REX3_COLORRED,  0 << 11);
    reg(&rex, REX3_COLORGRN,  0 << 11);
    reg(&rex, REX3_COLORBLUE, 0 << 11);
    reg(&rex, REX3_XYENDI,   xy(sw-1, sh-1));
    reg(&rex, REX3_XYSTARTI, xy(0, 0));
    reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);

    reg(&rex, REX3_COLORRED,  255 << 11);
    reg(&rex, REX3_COLORGRN,  255 << 11);
    reg(&rex, REX3_COLORBLUE, 255 << 11);
    let (cx0, cy0, cx1, cy1) = (sw/4, sh/4, 3*sw/4, 3*sh/4);
    reg(&rex, REX3_XYENDI,   xy(cx1, cy1));
    reg(&rex, REX3_XYSTARTI, xy(cx0, cy0));
    reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);

    // Center of inner block should be white
    assert_eq!(read_pixel(&rex, sw/2, sh/2) & 0xFFFFFF, 0xFFFFFF,
        "nested black+white: center should be white");
    // Corners of outer block should be black
    assert_eq!(read_pixel(&rex, 0, 0) & 0xFFFFFF, 0x000000,
        "nested black+white: corner (0,0) should be black");
    assert_eq!(read_pixel(&rex, sw-1, sh-1) & 0xFFFFFF, 0x000000,
        "nested black+white: corner (sw-1,sh-1) should be black");
    // Pixel just inside inner rectangle boundary
    assert_eq!(read_pixel(&rex, cx0, cy0) & 0xFFFFFF, 0xFFFFFF,
        "nested black+white: inner top-left corner should be white");

    // --- White background + black center ---
    reg(&rex, REX3_COLORRED,  255 << 11);
    reg(&rex, REX3_COLORGRN,  255 << 11);
    reg(&rex, REX3_COLORBLUE, 255 << 11);
    reg(&rex, REX3_XYENDI,   xy(sw-1, sh-1));
    reg(&rex, REX3_XYSTARTI, xy(0, 0));
    reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);

    reg(&rex, REX3_COLORRED,  0 << 11);
    reg(&rex, REX3_COLORGRN,  0 << 11);
    reg(&rex, REX3_COLORBLUE, 0 << 11);
    reg(&rex, REX3_XYENDI,   xy(cx1, cy1));
    reg(&rex, REX3_XYSTARTI, xy(cx0, cy0));
    reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);

    assert_eq!(read_pixel(&rex, sw/2, sh/2) & 0xFFFFFF, 0x000000,
        "nested white+black: center should be black");
    assert_eq!(read_pixel(&rex, 0, 0) & 0xFFFFFF, 0xFFFFFF,
        "nested white+black: corner (0,0) should be white");
}

/// Port of ng1patterns() 8-color-bar test in RGB24 mode.
/// Colors: black, red, green, yellow, blue, magenta, cyan, white.
/// Each bar spans 1/8 of the screen width (using width=16 per bar for test).
#[test]
fn test_patterns_rgb24_color_bars() {
    let rex = make_rex3();
    rex3init(&rex);

    reg(&rex, REX3_DRAWMODE1, DM1_RGB24_SRC);
    reg(&rex, REX3_WRMASK, 0xFFFFFF);

    // 8 bars, width=16 each, height=8
    let bar_w = 16i32;
    let bar_h = 8i32;

    // SGI ng1patterns color sequence
    let colors: &[(u32, u32, u32)] = &[
        (0,   0,   0  ),  // black
        (255, 0,   0  ),  // red
        (0,   255, 0  ),  // green
        (255, 255, 0  ),  // yellow
        (0,   0,   255),  // blue
        (255, 0,   255),  // magenta
        (0,   255, 255),  // cyan
        (255, 255, 255),  // white
    ];

    for (i, &(r, g, b)) in colors.iter().enumerate() {
        let x0 = i as i32 * bar_w;
        // ng1_rgbcolor: colorred.word = r<<11, colorgrn.word = g<<11, colorblue.word = b<<11
        reg(&rex, REX3_COLORRED,  r << 11);
        reg(&rex, REX3_COLORGRN,  g << 11);
        reg(&rex, REX3_COLORBLUE, b << 11);
        reg(&rex, REX3_XYENDI,   xy(x0 + bar_w - 1, bar_h - 1));
        reg(&rex, REX3_XYSTARTI, xy(x0, 0));
        reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);
    }

    for (i, &(r, g, b)) in colors.iter().enumerate() {
        let x0 = i as i32 * bar_w;
        let expected = (b << 16) | (g << 8) | r;
        // Check center pixel of each bar
        let cx = x0 + bar_w / 2;
        let cy = bar_h / 2;
        let px = read_pixel(&rex, cx, cy) & 0xFFFFFF;
        assert_eq!(px, expected,
            "RGB24 color bar {i} ({r},{g},{b}): center ({cx},{cy}) got {px:#08x} expected {expected:#08x}");
    }
}

/// Port of ng1_polygon() Gouraud shading: draw a shaded span in RGB24 mode.
/// Sets starting color (red=0) and slope (slopered = 1 per pixel), draws a span,
/// then verifies each pixel steps by the expected slope.
///
/// The shade bit (DM0 bit 18) enables per-pixel color += slope DDA.
/// colorred and slopered are plain u32/i32 in o12.11 format.
/// Each pixel: colorred += slopered (wrapping integer add). Integer part = colorred >> 11.
#[test]
fn test_patterns_gouraud_shade_span() {
    let rex = make_rex3();
    rex3init(&rex);

    let span_len = 8i32;    // pixels to draw
    let start_r  = 10u32;   // starting red value (integer)
    let slope_r  = 5u32;    // per-pixel increment in integer units

    reg(&rex, REX3_DRAWMODE1, DM1_RGB24_SRC);
    reg(&rex, REX3_WRMASK, 0xFFFFFF);

    // Set starting color: colorred = start_r, colorgrn=0, colorblue=0
    // ng1_rgbcolor encoding: integer r stored at bits [22:11] of COLORRED (o12.11 format)
    reg(&rex, REX3_COLORRED,  start_r << 11);
    reg(&rex, REX3_COLORGRN,  0);
    reg(&rex, REX3_COLORBLUE, 0);

    // Set slope via register writes.
    // SLOPERED: s(7)12.11 write format — positive integer n = n<<11 with sign bit clear.
    // SLOPEGRN/SLOPEBLUE: s(11)8.11 format — same encoding for positive values.
    reg(&rex, REX3_SLOPERED,  slope_r << 11);
    reg(&rex, REX3_SLOPEGRN,  0);
    reg(&rex, REX3_SLOPEBLUE, 0);

    // DM0 with shade bit (bit 18) + STOPONX span
    let dm0_shade_span = DM0_DRAW_SPAN | (1 << 18);

    reg(&rex, REX3_XYENDI,   xy(span_len - 1, 0));
    reg(&rex, REX3_XYSTARTI, xy(0, 0));
    reg_go(&rex, REX3_DRAWMODE0, dm0_shade_span);

    // Verify each pixel: pixel x should have red = start_r + x * slope_r
    // (shade increments AFTER writing each pixel, so pixel 0 = start_r)
    for x in 0..span_len {
        let expected_r = start_r + x as u32 * slope_r;
        let px = read_pixel(&rex, x, 0);
        let got_r = px & 0xFF;
        assert_eq!(got_r, expected_r,
            "shade span x={x}: red got {got_r} expected {expected_r}");
        // green and blue should be 0
        let got_g = (px >> 8) & 0xFF;
        let got_b = (px >> 16) & 0xFF;
        assert_eq!(got_g, 0, "shade span x={x}: green should be 0, got {got_g}");
        assert_eq!(got_b, 0, "shade span x={x}: blue should be 0, got {got_b}");
    }
}

// ============================================================================
// HOSTRW tests — ported from SGI vram3.c (ng1test_vram, ng1rvram, ng1wvram,
//                ng1test_vram_addr, ng1giobustest, ng1spfastclear).
//
// Two HOSTRW directions:
//   HOSTR (READ):  REX reads fb → host read FIFO → CPU reads via read_hostrw32/64.
//                  DM0 = OPCODE_READ | ADRMODE_BLOCK | STOPONXY | COLORHOST | DOSETUP.
//   HOSTW (WRITE): CPU writes raw pixel words to HOSTRW0 → REX draws into fb.
//                  DM0 = OPCODE_DRAW | ADRMODE_BLOCK | STOPONXY | COLORHOST.
//
// HOSTRW register write (SET space, no GO): walks test data through the loopback
//   path — REX3 sets.hostrw0/1 = data; REX3 go.hostrw0/1 reads it back.
//   This is the ng1giobustest pattern.
//
// 32-bit vs 64-bit:
//   32-bit: write32(go_addr(REX3_HOSTRW0), val32) / read_hostrw32()
//   64-bit: write64(go_addr(REX3_HOSTRW0), val64) / read_hostrw64()
//           Requires DM1_RWDOUBLE in drawmode1.
// ============================================================================

// DM1 values with host-depth and packed/double flags
// CI8: hostdepth=1 (8bpp), rwpacked (bit 7), same draw-plane config as DM1_CI8_SRC
const DM1_CI8_HOSTRW: u32 =
    DRAWMODE1_PLANES_RGB | (1 << 3) | DRAWMODE1_COMPARE_DISABLE_SH | DRAWMODE1_LOGICOP_SRC_SH | (1 << 8) | (1 << 7);
// CI8 64-bit: same as CI8 + rwdouble (bit 10) → 8 CI8 pixels per 64-bit word
const DM1_CI8_HOSTRW64: u32 = DM1_CI8_HOSTRW | (1 << 10);
// RGB24: hostdepth=3 (32bpp), rwpacked (bit 7), same draw-plane as DM1_RGB24_SRC
const DM1_RGB24_HOSTRW: u32 =
    DRAWMODE1_PLANES_RGB | (3 << 3) | (1 << 15) | DRAWMODE1_COMPARE_DISABLE_SH | DRAWMODE1_LOGICOP_SRC_SH | (3 << 8) | (1 << 7);
// RGB24 64-bit: same as above + rwdouble (bit 10)
const DM1_RGB24_HOSTRW64: u32 = DM1_RGB24_HOSTRW | (1 << 10);

// ============================================================================
// GIO bus loopback (ng1giobustest):
// Write to SET.hostrw0/1, read back from SET.hostrw0/1 after drain.
// SET write routes through GFIFO (async), so wait_idle() before reading back.
// ============================================================================

/// Port of ng1giobustest(): walk a 1 through all 32 bits of HOSTRW0 via SET path.
#[test]
fn test_hostrw_gio_bus_walking_ones_32bit() {
    let rex = make_rex3();
    rex3init(&rex);

    for b in 0..32u32 {
        let w = 1u32 << b;
        w32(rex, set_addr(REX3_HOSTRW0), w);
        // SET read: wait for GFIFO to drain, then return hostrw register.
        let got = loop {
            let r: BusRead32 = rex.read32(set_addr(REX3_HOSTRW0));
            if r.is_ok() { break r.data; }
            std::hint::spin_loop();
        };
        assert_eq!(got, w, "HOSTRW0 SET loopback bit {b}: got {got:#010x} expected {w:#010x}");
    }
}

/// Same for HOSTRW1.
#[test]
fn test_hostrw_gio_bus_walking_ones_hostrw1() {
    let rex = make_rex3();
    rex3init(&rex);

    for b in 0..32u32 {
        let w = 1u32 << b;
        w32(rex, set_addr(REX3_HOSTRW1), w);
        let got = loop {
            let r: BusRead32 = rex.read32(set_addr(REX3_HOSTRW1));
            if r.is_ok() { break r.data; }
            std::hint::spin_loop();
        };
        assert_eq!(got, w, "HOSTRW1 SET loopback bit {b}: got {got:#010x} expected {w:#010x}");
    }
}

// ============================================================================
// HOSTR (fb → host) tests — READ opcode.
//
// Protocol (matches SGI vram3.c / newport_accel.c):
//   1. Set up registers (DM1, XYENDI) in SET space.
//   2. Write DM0 = DM0_READ_BLOCK to GO space → triggers first batch → hostrw = word0.
//      (Or write XYSTARTI to GO space with DM0 already set.)
//   3. read_hostrw32(GO) → returns word[i], enqueues pure_go → next batch runs → hostrw = word[i+1].
//   4. For the last word: read_hostrw32_last(SET) → returns word[N-1], no extra batch.
// ============================================================================

/// CI8 HOSTR 32-bit: fill a small region, issue READ block, drain word by word.
/// CI8 + rwpacked: 4 CI8 pixels per 32-bit word.
#[test]
fn test_hostr_ci8_read_block_32bit() {
    let rex = make_rex3();
    rex3init(&rex);

    let color = 0x5Au8;
    let (x0, y0, x1, y1) = (0i32, 0i32, 15i32, 3i32);

    // Fill region via normal draw
    reg(&rex, REX3_DRAWMODE1, DM1_CI8_HOSTRW);
    reg(&rex, REX3_WRMASK, 0xFF);
    reg(&rex, REX3_COLORI, color as u32);
    reg(&rex, REX3_XYENDI,   xy(x1, y1));
    reg(&rex, REX3_XYSTARTI, xy(x0, y0));
    reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);

    // Set up READ block: DM1 + XYENDI in SET space, DM0 to GO triggers first batch.
    reg(&rex, REX3_DRAWMODE1, DM1_CI8_HOSTRW);
    reg(&rex, REX3_XYENDI,   xy(x1, y1));
    reg(&rex, REX3_XYSTARTI, xy(x0, y0));
    reg_go(&rex, REX3_DRAWMODE0, DM0_READ_BLOCK);

    // CI8 rwpacked: 4 pixels per 32-bit word.
    let width = (x1 - x0 + 1) as u32;
    let height = (y1 - y0 + 1) as u32;
    let words = width * height / 4;

    let expected_word = (color as u32)
        | ((color as u32) << 8)
        | ((color as u32) << 16)
        | ((color as u32) << 24);

    for i in 0..words {
        let got = if i < words - 1 { read_hostrw32(&rex) } else { read_hostrw32_last(&rex) };
        assert_eq!(got, expected_word,
            "CI8 HOSTR word {i}: got {got:#010x} expected {expected_word:#010x}");
    }
}

/// RGB24 HOSTR 32-bit: fill a small region, issue READ block, verify each 32-bit word.
/// RGB24 + hostdepth32: 1 pixel per 32-bit word.
#[test]
fn test_hostr_rgb24_read_block_32bit() {
    let rex = make_rex3();
    rex3init(&rex);

    let (r, g, b) = (0xAAu32, 0x55u32, 0xCCu32);
    let (x0, y0, x1, y1) = (0i32, 0i32, 3i32, 1i32);

    // Fill with RGB24
    reg(&rex, REX3_DRAWMODE1, DM1_RGB24_HOSTRW);
    reg(&rex, REX3_WRMASK, 0xFFFFFF);
    reg(&rex, REX3_COLORRED,  r << 11);
    reg(&rex, REX3_COLORGRN,  g << 11);
    reg(&rex, REX3_COLORBLUE, b << 11);
    reg(&rex, REX3_XYENDI,   xy(x1, y1));
    reg(&rex, REX3_XYSTARTI, xy(x0, y0));
    reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);

    // Issue READ block: DM0 GO triggers first batch.
    reg(&rex, REX3_DRAWMODE1, DM1_RGB24_HOSTRW);
    reg(&rex, REX3_XYENDI,   xy(x1, y1));
    reg(&rex, REX3_XYSTARTI, xy(x0, y0));
    reg_go(&rex, REX3_DRAWMODE0, DM0_READ_BLOCK);

    let width  = (x1 - x0 + 1) as u32;
    let height = (y1 - y0 + 1) as u32;
    let words  = width * height;  // 1 pixel per word

    let expected = (b << 16) | (g << 8) | r;
    for i in 0..words {
        let got = if i < words - 1 { read_hostrw32(&rex) } else { read_hostrw32_last(&rex) };
        assert_eq!(got & 0xFFFFFF, expected,
            "RGB24 HOSTR word {i}: got {got:#08x} expected {expected:#08x}");
    }
}

/// Multi-color HOSTR readback: fill each row with a different color, read back per-row.
#[test]
fn test_hostr_rgb24_multicolor_readback() {
    let rex = make_rex3();
    rex3init(&rex);

    let (x0, x1) = (0i32, 3i32);
    let colors: &[(u32, u32, u32)] = &[
        (0xFF, 0x00, 0x00),  // row 0: red
        (0x00, 0xFF, 0x00),  // row 1: green
        (0x00, 0x00, 0xFF),  // row 2: blue
        (0xAA, 0x55, 0xCC),  // row 3: mixed
    ];

    // Fill each row
    for (row, &(r, g, b)) in colors.iter().enumerate() {
        let y = row as i32;
        reg(&rex, REX3_DRAWMODE1, DM1_RGB24_HOSTRW);
        reg(&rex, REX3_WRMASK, 0xFFFFFF);
        reg(&rex, REX3_COLORRED,  r << 11);
        reg(&rex, REX3_COLORGRN,  g << 11);
        reg(&rex, REX3_COLORBLUE, b << 11);
        reg(&rex, REX3_XYENDI,   xy(x1, y));
        reg(&rex, REX3_XYSTARTI, xy(x0, y));
        reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);
    }

    // Read back each row individually.
    let width = (x1 - x0 + 1) as u32;  // 4 pixels → 4 words (1 per word, RGB32)
    for (row, &(r, g, b)) in colors.iter().enumerate() {
        let y = row as i32;
        reg(&rex, REX3_DRAWMODE1, DM1_RGB24_HOSTRW);
        reg(&rex, REX3_XYENDI,   xy(x1, y));
        reg(&rex, REX3_XYSTARTI, xy(x0, y));
        reg_go(&rex, REX3_DRAWMODE0, DM0_READ_BLOCK);

        let expected = (b << 16) | (g << 8) | r;
        for x in 0..width {
            let got = if x < width - 1 { read_hostrw32(&rex) } else { read_hostrw32_last(&rex) };
            assert_eq!(got & 0xFFFFFF, expected,
                "row {row} x={x}: got {got:#08x} expected {expected:#08x} ({r},{g},{b})");
        }
    }
}

// ============================================================================
// HOSTW (host → fb) tests — DRAW opcode with COLORHOST.
//
// Protocol (matches SGI newport_accel.c / ng1wvram()):
//   1. Set up DM1 + WRMASK + XYENDI + XYSTARTI + DM0 in SET space (enqueued in order).
//   2. write_hostrw32(GO, pixel[0]) → stores pixel[0] in hostrw, triggers batch 0
//      (one word's worth of pixels drawn, xstart advanced).
//   3. write_hostrw32(GO, pixel[1]) → batch 1, etc.
//   No separate DM0 GO needed — each HOSTRW GO is self-contained.
// ============================================================================

/// CI8 HOSTW 32-bit: write 4 pixels packed into one 32-bit word.
/// CI8 + rwpacked + hostdepth8: 4 CI8 pixels per word (MSB first).
#[test]
fn test_hostw_ci8_write_block_32bit() {
    let rex = make_rex3();
    rex3init(&rex);

    // Pixels: [x=0]=0x11, [x=1]=0x22, [x=2]=0x33, [x=3]=0x44
    // CI8 unpack_8_32_ci: pixel = (shifter >> 24) & 0xFF, shift left 8 per pixel.
    // 32-bit word loaded as upper 32 bits of u64 shifter for 32-bit mode.
    // Wait — for 32-bit non-rwdouble: host_shifter = val as u64 (zero-extended lower 32).
    // unpack_8_32_ci = ((val as u32) >> 24) & 0xFF = MSB of the 32-bit word.
    // So pack as: pixel0 in MSB, pixel3 in LSB: word = (p0<<24)|(p1<<16)|(p2<<8)|p3.
    let word: u32 = (0x11u32 << 24) | (0x22 << 16) | (0x33 << 8) | 0x44;

    // Setup in SET space, then one HOSTRW GO triggers the single-word batch.
    reg(&rex, REX3_DRAWMODE1, DM1_CI8_HOSTRW);
    reg(&rex, REX3_WRMASK, 0xFF);
    reg(&rex, REX3_XYENDI,   xy(3, 0));
    reg(&rex, REX3_XYSTARTI, xy(0, 0));
    reg(&rex, REX3_DRAWMODE0, DM0_HOSTW_BLOCK);  // SET — loads mode, no draw
    write_hostrw32(&rex, word);   // GO — triggers batch (4 pixels)
    wait(&rex);

    assert_eq!(read_pixel(&rex, 0, 0) & 0xFF, 0x11, "CI8 HOSTW x=0");
    assert_eq!(read_pixel(&rex, 1, 0) & 0xFF, 0x22, "CI8 HOSTW x=1");
    assert_eq!(read_pixel(&rex, 2, 0) & 0xFF, 0x33, "CI8 HOSTW x=2");
    assert_eq!(read_pixel(&rex, 3, 0) & 0xFF, 0x44, "CI8 HOSTW x=3");
}

/// RGB24 HOSTW 32-bit: write one pixel per 32-bit GO write.
/// RGB24 + hostdepth32: 1 pixel per word.
#[test]
fn test_hostw_rgb24_write_block_32bit() {
    let rex = make_rex3();
    rex3init(&rex);

    let pixels: &[u32] = &[0x0000FF, 0x00FF00, 0xFF0000, 0xAABBCC];
    let width = pixels.len() as i32;

    // Setup in SET space: DM1 + WRMASK + coords + DM0, then one GO per pixel.
    reg(&rex, REX3_DRAWMODE1, DM1_RGB24_HOSTRW);
    reg(&rex, REX3_WRMASK, 0xFFFFFF);
    reg(&rex, REX3_XYENDI,   xy(width - 1, 0));
    reg(&rex, REX3_XYSTARTI, xy(0, 0));
    reg(&rex, REX3_DRAWMODE0, DM0_HOSTW_BLOCK);  // SET — no draw yet
    for &p in pixels {
        write_hostrw32(&rex, p);  // each GO triggers one pixel draw
    }
    wait(&rex);

    for (i, &p) in pixels.iter().enumerate() {
        let got = read_pixel(&rex, i as i32, 0) & 0xFFFFFF;
        assert_eq!(got, p & 0xFFFFFF,
            "RGB24 HOSTW pixel[{i}]: got {got:#08x} expected {p:#08x}");
    }
}

/// RGB24 HOSTW 64-bit: write two pixels per 64-bit GO write.
/// RWDOUBLE: high 32 bits = first pixel, low 32 bits = second pixel.
#[test]
fn test_hostw_rgb24_write_block_64bit() {
    let rex = make_rex3();
    rex3init(&rex);

    // p0=blue (0xFF0000 in BGR), p1=green (0x00FF00 in BGR)
    let p0: u32 = 0x00FF0000;
    let p1: u32 = 0x0000FF00;
    let word64: u64 = ((p0 as u64) << 32) | (p1 as u64);

    // Setup in SET space, then one 64-bit GO write draws both pixels.
    reg(&rex, REX3_DRAWMODE1, DM1_RGB24_HOSTRW64);
    reg(&rex, REX3_WRMASK, 0xFFFFFF);
    reg(&rex, REX3_XYENDI,   xy(1, 0));
    reg(&rex, REX3_XYSTARTI, xy(0, 0));
    reg(&rex, REX3_DRAWMODE0, DM0_HOSTW_BLOCK);  // SET
    write_hostrw64(&rex, word64);  // GO — triggers 2-pixel batch
    wait(&rex);

    let got0 = read_pixel(&rex, 0, 0) & 0xFFFFFF;
    let got1 = read_pixel(&rex, 1, 0) & 0xFFFFFF;
    assert_eq!(got0, p0 & 0xFFFFFF,
        "RGB24 HOSTW64 pixel[0]: got {got0:#08x} expected {p0:#08x}");
    assert_eq!(got1, p1 & 0xFFFFFF,
        "RGB24 HOSTW64 pixel[1]: got {got1:#08x} expected {p1:#08x}");
}

/// RGB24 HOSTR 64-bit: fill a region, issue READ block, drain with read_hostrw64().
/// RWDOUBLE packs two pixels per 64-bit word: high 32 bits = first pixel.
#[test]
fn test_hostr_rgb24_read_block_64bit() {
    let rex = make_rex3();
    rex3init(&rex);

    let (r, g, b) = (0x12u32, 0x34u32, 0x56u32);
    let (x0, y0, x1, y1) = (0i32, 0i32, 3i32, 0i32);  // 4 pixels, 1 row → 2 words

    // Fill
    reg(&rex, REX3_DRAWMODE1, DM1_RGB24_HOSTRW64);
    reg(&rex, REX3_WRMASK, 0xFFFFFF);
    reg(&rex, REX3_COLORRED,  r << 11);
    reg(&rex, REX3_COLORGRN,  g << 11);
    reg(&rex, REX3_COLORBLUE, b << 11);
    reg(&rex, REX3_XYENDI,   xy(x1, y1));
    reg(&rex, REX3_XYSTARTI, xy(x0, y0));
    reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);

    // Issue READ block: DM0 GO triggers first batch (2 pixels packed into 1 word).
    reg(&rex, REX3_DRAWMODE1, DM1_RGB24_HOSTRW64);
    reg(&rex, REX3_XYENDI,   xy(x1, y1));
    reg(&rex, REX3_XYSTARTI, xy(x0, y0));
    reg_go(&rex, REX3_DRAWMODE0, DM0_READ_BLOCK);

    let expected_px = (b << 16) | (g << 8) | r;
    let words = 2u32;  // 4 pixels / 2 per word

    for i in 0..words {
        let got = if i < words - 1 { read_hostrw64(&rex) } else { read_hostrw64_last(&rex) };
        let hi = (got >> 32) as u32 & 0xFFFFFF;
        let lo = got as u32 & 0xFFFFFF;
        assert_eq!(hi, expected_px,
            "HOSTR64 word {i} hi: got {hi:#08x} expected {expected_px:#08x}");
        assert_eq!(lo, expected_px,
            "HOSTR64 word {i} lo: got {lo:#08x} expected {expected_px:#08x}");
    }
}

/// HOSTR+HOSTW round-trip: write pixels via HOSTW (one GO per pixel), read back via HOSTR.
#[test]
fn test_hostrw_roundtrip_rgb24() {
    let rex = make_rex3();
    rex3init(&rex);

    let (x0, y0, x1, y1) = (0i32, 0i32, 3i32, 1i32);
    let width  = (x1 - x0 + 1) as usize;
    let height = (y1 - y0 + 1) as usize;

    // Unique per-pixel values
    let mut pixels = vec![0u32; width * height];
    for y in 0..height {
        for x in 0..width {
            pixels[y * width + x] = (y as u32 * 0x10 + x as u32) * 0x010203 & 0xFFFFFF;
        }
    }

    // HOSTW: setup in SET space, then one GO per pixel.
    reg(&rex, REX3_DRAWMODE1, DM1_RGB24_HOSTRW);
    reg(&rex, REX3_WRMASK, 0xFFFFFF);
    reg(&rex, REX3_XYENDI,   xy(x1, y1));
    reg(&rex, REX3_XYSTARTI, xy(x0, y0));
    reg(&rex, REX3_DRAWMODE0, DM0_HOSTW_BLOCK);  // SET — loads draw mode
    for &p in &pixels {
        write_hostrw32(&rex, p);  // GO — draws one pixel
    }
    wait(&rex);

    // HOSTR: DM0 GO triggers first batch, subsequent GO reads advance.
    reg(&rex, REX3_DRAWMODE1, DM1_RGB24_HOSTRW);
    reg(&rex, REX3_XYENDI,   xy(x1, y1));
    reg(&rex, REX3_XYSTARTI, xy(x0, y0));
    reg_go(&rex, REX3_DRAWMODE0, DM0_READ_BLOCK);

    let n = pixels.len() as u32;
    for (i, &expected) in pixels.iter().enumerate() {
        let got = if (i as u32) < n - 1 { read_hostrw32(&rex) } else { read_hostrw32_last(&rex) };
        assert_eq!(got & 0xFFFFFF, expected & 0xFFFFFF,
            "roundtrip pixel[{i}]: got {got:#08x} expected {expected:#08x}");
    }
}

// ============================================================================
// HOSTR/HOSTW partial-word (span-end clamping) tests.
//
// Confirmed via MAME newport.cpp do_pixel_word_read():
//   width = min(x_end - x_start + 1, max_width)
// A span narrower than max_width reads fewer pixels; result is left-aligned
// (MSB-first) with zero-padding in unused LSB slots — no y-wrap occurs.
//
// For HOSTW: if span < host_count, the extra host pixels are simply unused.
// For HOSTR: flush_host_pixel() left-aligns the partial word before storing.
// ============================================================================

/// CI8 HOSTR partial word: 3-pixel-wide span (< 4 pixels/word).
/// Expects one flush with 3 pixels left-aligned and one zero byte at LSB.
#[test]
fn test_hostr_ci8_partial_word() {
    let rex = make_rex3();
    rex3init(&rex);

    // Fill 3 pixels at y=0 with distinct colors
    for x in 0i32..3 {
        reg(&rex, REX3_DRAWMODE1, DM1_CI8_SRC);
        reg(&rex, REX3_WRMASK, 0xFF);
        reg(&rex, REX3_COLORI, (x as u32 + 1) * 0x11);  // 0x11, 0x22, 0x33
        reg(&rex, REX3_XYENDI,   xy(x, 0));
        reg(&rex, REX3_XYSTARTI, xy(x, 0));
        reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);
    }

    // Issue READ block for 3-pixel span: one partial word (3 of 4 slots filled).
    reg(&rex, REX3_DRAWMODE1, DM1_CI8_HOSTRW);
    reg(&rex, REX3_XYENDI,   xy(2, 0));  // x0=0, x1=2 → width=3
    reg(&rex, REX3_XYSTARTI, xy(0, 0));
    reg_go(&rex, REX3_DRAWMODE0, DM0_READ_BLOCK);

    // Expect: pixels packed MSB-first, last byte = 0 (unused).
    // host_pack_8_ci: (acc<<8)|pixel — after 3 pixels: p0<<16 | p1<<8 | p2
    // flush shifts left by 1*8 more: (p0<<24)|(p1<<16)|(p2<<8)|0x00
    let got = read_hostrw32_last(&rex);
    assert_eq!((got >> 24) & 0xFF, 0x11, "CI8 partial HOSTR: pixel0={:#04x}", (got>>24)&0xFF);
    assert_eq!((got >> 16) & 0xFF, 0x22, "CI8 partial HOSTR: pixel1={:#04x}", (got>>16)&0xFF);
    assert_eq!((got >> 8)  & 0xFF, 0x33, "CI8 partial HOSTR: pixel2={:#04x}", (got>>8)&0xFF);
    assert_eq!( got        & 0xFF, 0x00, "CI8 partial HOSTR: unused LSB should be 0");
}

/// CI8 HOSTW partial word: 3-pixel-wide span (< 4 pixels/word).
/// Only the first 3 pixels of the host word are drawn; the 4th is unused.
/// No y-wrap: pixels after x_end are NOT written.
#[test]
fn test_hostw_ci8_partial_word() {
    let rex = make_rex3();
    rex3init(&rex);

    // Pack 4 CI8 pixels in word but only draw 3 (x=0..2).
    // host_unpack_8_32_ci: pixel = (shifter >> 24) & 0xFF, then shift left 8.
    // Pixels in order: p0=0x11 (MSB), p1=0x22, p2=0x33, p3=0x44 (unused).
    let word: u32 = (0x11u32 << 24) | (0x22 << 16) | (0x33 << 8) | 0x44;

    // Write pixel at x=3 with a sentinel (to confirm it's NOT overwritten by p3=0x44).
    reg(&rex, REX3_DRAWMODE1, DM1_CI8_SRC);
    reg(&rex, REX3_WRMASK, 0xFF);
    reg(&rex, REX3_COLORI, 0xAA);
    reg(&rex, REX3_XYENDI,   xy(3, 0));
    reg(&rex, REX3_XYSTARTI, xy(3, 0));
    reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);

    // HOSTW 3-pixel span: only x=0,1,2 drawn; x=3 (p3=0x44) must NOT be written.
    reg(&rex, REX3_DRAWMODE1, DM1_CI8_HOSTRW);
    reg(&rex, REX3_WRMASK, 0xFF);
    reg(&rex, REX3_XYENDI,   xy(2, 0));  // x_end = 2 → span stops after pixel 2
    reg(&rex, REX3_XYSTARTI, xy(0, 0));
    reg(&rex, REX3_DRAWMODE0, DM0_HOSTW_BLOCK);  // SET
    write_hostrw32(&rex, word);  // GO — draws 3 pixels (span end clamps to x=2)
    wait(&rex);

    assert_eq!(read_pixel(&rex, 0, 0) & 0xFF, 0x11, "HOSTW partial: x=0 should be 0x11");
    assert_eq!(read_pixel(&rex, 1, 0) & 0xFF, 0x22, "HOSTW partial: x=1 should be 0x22");
    assert_eq!(read_pixel(&rex, 2, 0) & 0xFF, 0x33, "HOSTW partial: x=2 should be 0x33");
    assert_eq!(read_pixel(&rex, 3, 0) & 0xFF, 0xAA, "HOSTW partial: x=3 sentinel should be 0xAA (unused p3 not drawn)");
}

// ============================================================================
// Multi-word, multi-row HOSTW tests.
//
// These exercise the full DMA-style write path that IRIX uses for blit/image
// transfers: multiple rows, multiple host-write words per row.
//
// CI8 packed (32-bit): 4 CI8 pixels per 32-bit word → 2 words per 8-pixel row.
// RGB24 unpacked (32-bit): 1 RGB pixel per 32-bit word → 5 words per 5-pixel row.
// CI8 packed (64-bit): 8 CI8 pixels per 64-bit word → 1 word per 8-pixel row.
// RGB24 unpacked (64-bit): 2 RGB pixels per 64-bit word → 3 words per 6-pixel row.
// ============================================================================

/// CI8 HOSTW 32-bit packed, multiline, multi-word-per-row.
/// 8 pixels wide × 3 rows = 24 pixels. Packed CI8: 4 pixels per 32-bit word →
/// 2 words per row, 6 words total.  Each row uses a different set of colors.
#[test]
fn test_hostw_ci8_multiline_32bit_packed() {
    let rex = make_rex3();
    rex3init(&rex);

    // Three rows of 8 unique CI8 colors each (MSB-first within each word).
    let rows: [[u8; 8]; 3] = [
        [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88],
        [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x12, 0x34],
        [0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x01, 0x23],
    ];
    let (x0, y0, x1, y1) = (0i32, 0i32, 7i32, 2i32); // 8×3

    reg(&rex, REX3_DRAWMODE1, DM1_CI8_HOSTRW);
    reg(&rex, REX3_WRMASK, 0xFF);
    reg(&rex, REX3_XYENDI,   xy(x1, y1));
    reg(&rex, REX3_XYSTARTI, xy(x0, y0));
    reg(&rex, REX3_DRAWMODE0, DM0_HOSTW_BLOCK); // SET — loads draw mode

    // Send 2 words per row, 3 rows = 6 words total.
    // Packed CI8: word = (p0<<24)|(p1<<16)|(p2<<8)|p3, MSB is drawn first.
    for row in &rows {
        let w0: u32 = ((row[0] as u32) << 24) | ((row[1] as u32) << 16)
                    | ((row[2] as u32) << 8)  |  (row[3] as u32);
        let w1: u32 = ((row[4] as u32) << 24) | ((row[5] as u32) << 16)
                    | ((row[6] as u32) << 8)  |  (row[7] as u32);
        write_hostrw32(&rex, w0); // GO — draws pixels 0-3 of this row
        write_hostrw32(&rex, w1); // GO — draws pixels 4-7, advances y
    }
    wait(&rex);

    for (y, row) in rows.iter().enumerate() {
        for (x, &expected) in row.iter().enumerate() {
            let got = read_pixel(&rex, x as i32, y as i32) & 0xFF;
            assert_eq!(got, expected as u32,
                "CI8 HOSTW32 packed y={y} x={x}: got {got:#04x} expected {expected:#04x}");
        }
    }
}

/// RGB24 HOSTW 32-bit unpacked, multiline, multi-word-per-row.
/// 5 pixels wide × 3 rows = 15 pixels. Unpacked RGB24: 1 pixel per 32-bit word →
/// 5 words per row, 15 words total.
#[test]
fn test_hostw_rgb24_multiline_32bit_unpacked() {
    let rex = make_rex3();
    rex3init(&rex);

    // Three rows of 5 distinct RGB24 colors.
    let rows: [[u32; 5]; 3] = [
        [0xFF0000, 0x00FF00, 0x0000FF, 0xFFFF00, 0xFF00FF],
        [0x00FFFF, 0x804020, 0x102030, 0xABCDEF, 0x010203],
        [0xFEDCBA, 0x123456, 0x789ABC, 0xDEF012, 0x345678],
    ];
    let (x0, y0, x1, y1) = (0i32, 0i32, 4i32, 2i32); // 5×3

    reg(&rex, REX3_DRAWMODE1, DM1_RGB24_HOSTRW);
    reg(&rex, REX3_WRMASK, 0xFFFFFF);
    reg(&rex, REX3_XYENDI,   xy(x1, y1));
    reg(&rex, REX3_XYSTARTI, xy(x0, y0));
    reg(&rex, REX3_DRAWMODE0, DM0_HOSTW_BLOCK); // SET

    // 1 pixel per 32-bit GO write; REX advances x then y automatically.
    for row in &rows {
        for &px in row {
            write_hostrw32(&rex, px);
        }
    }
    wait(&rex);

    for (y, row) in rows.iter().enumerate() {
        for (x, &expected) in row.iter().enumerate() {
            let got = read_pixel(&rex, x as i32, y as i32) & 0xFFFFFF;
            assert_eq!(got, expected,
                "RGB24 HOSTW32 unpacked y={y} x={x}: got {got:#08x} expected {expected:#08x}");
        }
    }
}

/// CI8 HOSTW 64-bit packed, multiline, one word per row.
/// 8 pixels wide × 3 rows = 24 pixels. Packed CI8 + rwdouble: 8 CI8 pixels
/// per 64-bit word → 1 word per row, 3 words total.
#[test]
fn test_hostw_ci8_multiline_64bit_packed() {
    let rex = make_rex3();
    rex3init(&rex);

    let rows: [[u8; 8]; 3] = [
        [0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80],
        [0x91, 0xA2, 0xB3, 0xC4, 0xD5, 0xE6, 0xF7, 0x08],
        [0x19, 0x2A, 0x3B, 0x4C, 0x5D, 0x6E, 0x7F, 0x00],
    ];
    let (x0, y0, x1, y1) = (0i32, 0i32, 7i32, 2i32); // 8×3

    reg(&rex, REX3_DRAWMODE1, DM1_CI8_HOSTRW64);
    reg(&rex, REX3_WRMASK, 0xFF);
    reg(&rex, REX3_XYENDI,   xy(x1, y1));
    reg(&rex, REX3_XYSTARTI, xy(x0, y0));
    reg(&rex, REX3_DRAWMODE0, DM0_HOSTW_BLOCK); // SET

    // 8 CI8 pixels per 64-bit word, MSB-first.
    // Byte layout: bits[63:56]=p0, bits[55:48]=p1, ..., bits[7:0]=p7.
    for row in &rows {
        let w: u64 = (row[0] as u64) << 56 | (row[1] as u64) << 48
                   | (row[2] as u64) << 40 | (row[3] as u64) << 32
                   | (row[4] as u64) << 24 | (row[5] as u64) << 16
                   | (row[6] as u64) <<  8 | (row[7] as u64);
        write_hostrw64(&rex, w);
    }
    wait(&rex);

    for (y, row) in rows.iter().enumerate() {
        for (x, &expected) in row.iter().enumerate() {
            let got = read_pixel(&rex, x as i32, y as i32) & 0xFF;
            assert_eq!(got, expected as u32,
                "CI8 HOSTW64 packed y={y} x={x}: got {got:#04x} expected {expected:#04x}");
        }
    }
}

/// RGB24 HOSTW 64-bit unpacked, multiline, multiple words per row.
/// 6 pixels wide × 3 rows = 18 pixels. RGB24 + rwdouble: 2 pixels per 64-bit
/// word (high 32 bits = first pixel) → 3 words per row, 9 words total.
#[test]
fn test_hostw_rgb24_multiline_64bit_unpacked() {
    let rex = make_rex3();
    rex3init(&rex);

    // Three rows of 6 distinct RGB24 colors.
    let rows: [[u32; 6]; 3] = [
        [0xFF0000, 0x00FF00, 0x0000FF, 0xFFFF00, 0xFF00FF, 0x00FFFF],
        [0x112233, 0x445566, 0x778899, 0xAABBCC, 0xDDEEFF, 0x010203],
        [0xFEDCBA, 0x987654, 0x321098, 0xABCDEF, 0xFEDCBA, 0x654321],
    ];
    let (x0, y0, x1, y1) = (0i32, 0i32, 5i32, 2i32); // 6×3

    reg(&rex, REX3_DRAWMODE1, DM1_RGB24_HOSTRW64);
    reg(&rex, REX3_WRMASK, 0xFFFFFF);
    reg(&rex, REX3_XYENDI,   xy(x1, y1));
    reg(&rex, REX3_XYSTARTI, xy(x0, y0));
    reg(&rex, REX3_DRAWMODE0, DM0_HOSTW_BLOCK); // SET

    // 2 pixels per 64-bit word: high 32 bits = first pixel, low 32 bits = second.
    for row in &rows {
        for pair in row.chunks(2) {
            let w: u64 = ((pair[0] as u64) << 32) | (pair[1] as u64);
            write_hostrw64(&rex, w);
        }
    }
    wait(&rex);

    for (y, row) in rows.iter().enumerate() {
        for (x, &expected) in row.iter().enumerate() {
            let got = read_pixel(&rex, x as i32, y as i32) & 0xFFFFFF;
            assert_eq!(got, expected,
                "RGB24 HOSTW64 unpacked y={y} x={x}: got {got:#08x} expected {expected:#08x}");
        }
    }
}

// ============================================================================
// I_LINE tests
// ============================================================================

// DM0 for a full I_LINE draw (stoponx+stopony so the whole line runs in one GO).
const DM0_DRAW_ILINE: u32 = DRAWMODE0_OPCODE_DRAW | DRAWMODE0_ADRMODE_I_LINE_SH | DM0_DOSETUP | DM0_STOPONXY;
// DM0 for I_LINE single-step mode (no stoponx/stopony — one pixel per GO).
const DM0_DRAW_ILINE_STEP: u32 = DRAWMODE0_OPCODE_DRAW | DRAWMODE0_ADRMODE_I_LINE_SH | DM0_DOSETUP;
// DM0 for a full F_LINE draw — fractional-endpoint Bresenham correction.
const DM0_DRAW_FLINE: u32 = DRAWMODE0_OPCODE_DRAW | DRAWMODE0_ADRMODE_F_LINE_SH | DM0_DOSETUP | DM0_STOPONXY;
// DM0 for a full A_LINE draw — F_LINE plus AWEIGHT-LUT endpoint suppression (needs ENDPTFILTER, bit 22, set separately).
// A_LINE tests are out of scope for this pass (see rules/testing/rex3-fline-fractional-bresenham.md) —
// kept for a future session, not yet exercised by any test.
#[allow(dead_code)]
const DM0_DRAW_ALINE: u32 = DRAWMODE0_OPCODE_DRAW | DRAWMODE0_ADRMODE_A_LINE_SH | DM0_DOSETUP | DM0_STOPONXY;
#[allow(dead_code)]
const DM0_ENDPTFILTER: u32 = 1 << 22;

/// Draw an I_LINE in CI8 and return the set of (x,y) pixels that were written with `color`.
/// Clears a 256x256 region starting at `base` before drawing.
fn draw_iline_pixels(rex: &Rex3, x0: i32, y0: i32, x1: i32, y1: i32, color: u8, dm0: u32) -> Vec<(i32, i32)> {
    // Clear a generous region around the line.
    let bx = x0.min(x1) - 2;
    let by = y0.min(y1) - 2;
    let ex = x0.max(x1) + 2;
    let ey = y0.max(y1) + 2;
    reg(rex, REX3_DRAWMODE1, DM1_CI8_SRC);
    reg(rex, REX3_WRMASK, 0xFF);
    reg(rex, REX3_COLORI, 0);
    reg(rex, REX3_XYENDI,   xy(ex, ey));
    reg(rex, REX3_XYSTARTI, xy(bx, by));
    reg_go(rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);

    // Draw the line.
    reg(rex, REX3_COLORI, color as u32);
    reg(rex, REX3_XYENDI,   xy(x1, y1));
    reg(rex, REX3_XYSTARTI, xy(x0, y0));
    reg_go(rex, REX3_DRAWMODE0, dm0);

    // Collect all written pixels in the bounding box.
    let mut pts = Vec::new();
    for y in by..=ey {
        for x in bx..=ex {
            if read_pixel(rex, x, y) & 0xFF == color as u32 {
                pts.push((x, y));
            }
        }
    }
    pts
}

/// Same but drives one pixel per GO (iterate_one / single-step mode).
fn draw_iline_step(rex: &Rex3, x0: i32, y0: i32, x1: i32, y1: i32, color: u8) -> Vec<(i32, i32)> {
    let dx = (x1 - x0).abs();
    let dy = (y1 - y0).abs();
    let pixel_count = dx.max(dy) + 1;

    let bx = x0.min(x1) - 2;
    let by = y0.min(y1) - 2;
    let ex = x0.max(x1) + 2;
    let ey = y0.max(y1) + 2;

    // Clear region.
    reg(rex, REX3_DRAWMODE1, DM1_CI8_SRC);
    reg(rex, REX3_WRMASK, 0xFF);
    reg(rex, REX3_COLORI, 0);
    reg(rex, REX3_XYENDI,   xy(ex, ey));
    reg(rex, REX3_XYSTARTI, xy(bx, by));
    reg_go(rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);

    // First GO with DOSETUP to establish Bresenham state and draw pixel 0.
    reg(rex, REX3_COLORI, color as u32);
    reg(rex, REX3_XYENDI,   xy(x1, y1));
    reg(rex, REX3_XYSTARTI, xy(x0, y0));
    reg_go(rex, REX3_DRAWMODE0, DM0_DRAW_ILINE_STEP);

    // Subsequent GOs without DOSETUP — each draws one more pixel.
    let dm0_cont = DRAWMODE0_OPCODE_DRAW | DRAWMODE0_ADRMODE_I_LINE_SH;
    for _ in 1..pixel_count {
        reg_go(rex, REX3_DRAWMODE0, dm0_cont);
    }

    let mut pts = Vec::new();
    for y in by..=ey {
        for x in bx..=ex {
            if read_pixel(rex, x, y) & 0xFF == color as u32 {
                pts.push((x, y));
            }
        }
    }
    pts
}

/// Reference F_LINE Bresenham in software — a mechanical transcription of
/// `setup()` + `fline_apply_fract()` + the `bres_step!` macro from
/// `draw_line_bresenham` (rex3.rs), NOT the simplified octant-agnostic form
/// `bres_pixels()` uses. This is deliberately duplicated rather than calling
/// the real implementation: a hand-transcribed oracle must fail if
/// `fline_apply_fract`/`setup()` regress, which a shared-code oracle cannot
/// detect (see `rules/` note on this — a call-through oracle would mask
/// exactly the kind of regression this test exists to catch).
///
/// `x0_frac4`/`y0_frac4` are the *start* endpoint's fractional nibble
/// (0-15, 1/16-pixel units) — the only fractional input `fline_apply_fract`
/// consumes; the end endpoint is integer-only here, matching what the
/// F_LINE hardware path actually uses (draw_aline's AWEIGHT lookup is the
/// only consumer of the *end* endpoint's fraction, handled separately by
/// `aline_endpoint_skip` below).
fn fline_pixels(x0: i32, x0_frac4: i32, y0: i32, y0_frac4: i32, x1: i32, y1: i32) -> Vec<(i32, i32)> {
    // BRES table copied verbatim from draw_line_bresenham (rex3.rs):
    // (incrx1, incrx2, incry1, incry2, y_major), indexed by octant.
    #[rustfmt::skip]
    const BRES: [(i32, i32, i32, i32, bool); 8] = [
        ( 0,  1, -1, -1, true ),  // octant 0
        ( 0,  1,  1,  1, true ),  // octant 1
        ( 0, -1, -1, -1, true ),  // octant 2
        ( 0, -1,  1,  1, true ),  // octant 3
        ( 1,  1,  0, -1, false),  // octant 4
        ( 1,  1,  0,  1, false),  // octant 5
        (-1, -1,  0, -1, false),  // octant 6
        (-1, -1,  0,  1, false),  // octant 7
    ];

    // 21.11 fixed-point endpoint values, matching how the register writes
    // populate ctx.xstart/ystart/xend/yend.
    let xstart = (x0 << 11) | (x0_frac4 << 7);
    let ystart = (y0 << 11) | (y0_frac4 << 7);
    let xend = x1 << 11;
    let yend = y1 << 11;

    // --- setup(): derive octant + initial incr1/incr2/d (rex3.rs:1390-1419) ---
    let dx = xend - xstart;
    let dy = yend - ystart;
    let adx = dx.abs() >> 11;
    let ady = dy.abs() >> 11;

    let mut octant = 0u32;
    if dy < 0 { octant |= 1 << 0; } // OCTANT_YDEC
    if dx < 0 { octant |= 1 << 1; } // OCTANT_XDEC
    if adx > ady { octant |= 1 << 2; } // OCTANT_XMAJOR

    let (major, minor) = if adx > ady { (adx, ady) } else { (ady, adx) };
    let incr1 = 2 * minor;
    let incr2 = 2 * (minor - major);
    let mut d = incr1 - major;

    let (incrx1, incrx2, incry1, incry2, y_major) = BRES[(octant & 7) as usize];

    let mut x = xstart >> 11;
    let mut y = ystart >> 11;
    let x2 = xend >> 11;
    let y2 = yend >> 11;

    // --- fline_apply_fract() (rex3.rs:1754-1816), transcribed verbatim ---
    {
        let x1p = xstart >> 11;
        let y1p = ystart >> 11;
        let x2p = xend >> 11;
        let y2p = yend >> 11;
        let mut fdx = (x1p - x2p).abs();
        let mut fdy = (y1p - y2p).abs();
        let mut xf = (xstart >> 7) & 0xF;
        let mut yf = (ystart >> 7) & 0xF;

        match octant & 7 {
            1 => {
                std::mem::swap(&mut xf, &mut yf);
                std::mem::swap(&mut fdx, &mut fdy);
            }
            3 => {
                xf = 0x10 - xf;
                std::mem::swap(&mut xf, &mut yf);
                std::mem::swap(&mut fdx, &mut fdy);
            }
            7 => { xf = 0x10 - xf; }
            6 => {
                xf = 0x10 - xf;
                yf = 0x10 - yf;
            }
            2 => {
                let t = 0x10 - xf;
                xf = 0x10 - yf;
                yf = t;
                std::mem::swap(&mut fdx, &mut fdy);
            }
            0 => {
                let t = 0x10 - yf;
                yf = xf;
                xf = t;
                std::mem::swap(&mut fdx, &mut fdy);
            }
            4 => { yf = 0x10 - yf; }
            _ => {}
        }

        // Spec-correct base d for F_LINE/A_LINE is 3*minor - 2*major, not
        // I_LINE's 2*minor - major (see rex3.rs fline_apply_fract for the
        // full derivation from rex3_pdf.md 3.6.1.2 and MAME's do_fline).
        // `d` here still holds the I_LINE-formula value; apply the same
        // (minor - major) correction the real fix applies before adding
        // the fractional term.
        d += fdy - fdx;
        d += 2 * (((fdx * yf) >> 4) - ((fdy * xf) >> 4));
        let major_delta = if y_major { fdy } else { fdx };
        let e = d - 2 * major_delta;
        if e > 0 {
            d = e;
            let x_major = !y_major;
            if x_major {
                y -= incry2;
            } else {
                x += incrx2;
            }
        }
    }

    // --- step exactly like bres_step! / draw_line_bresenham's main loop ---
    let adx2 = (x2 - x).abs();
    let ady2 = (y2 - y).abs();
    let pixel_count = adx2.max(ady2) + 1;

    let mut pts = Vec::new();
    for i in 0..pixel_count {
        pts.push((x, y));
        let is_last = i == pixel_count - 1;
        if !is_last {
            if d < 0 {
                x += incrx1; y -= incry1; d += incr1;
            } else {
                x += incrx2; y -= incry2; d += incr2;
            }
        }
    }
    pts
}

/// Reference A_LINE endpoint-suppression decision — mirrors `draw_aline`'s
/// AWEIGHT LUT lookup (rex3.rs:1826-1851). `aweight0`/`aweight1` are the raw
/// 16-entry/4-bit-packed LUT register values. Returns (skip_first, skip_last).
/// A_LINE tests are out of scope for this pass — kept for a future session.
#[allow(dead_code)]
fn aline_endpoint_skip(
    x0_frac4: i32, y0_frac4: i32, x1_frac4: i32, y1_frac4: i32,
    aweight0: u32, aweight1: u32,
) -> (bool, bool) {
    let mut skip_first = false;
    let mut skip_last = false;
    if x0_frac4 != 0 || y0_frac4 != 0 {
        let wi = ((x0_frac4 + y0_frac4) as usize).min(15);
        let w = (aweight0 >> (wi * 4)) & 0xF;
        if w == 0 { skip_first = true; }
    }
    if x1_frac4 != 0 || y1_frac4 != 0 {
        let wi = ((x1_frac4 + y1_frac4) as usize).min(15);
        let w = (aweight1 >> (wi * 4)) & 0xF;
        if w == 0 { skip_last = true; }
    }
    (skip_first, skip_last)
}

/// Reference Bresenham in software — returns the exact pixel sequence for an integer line.
fn bres_pixels(x0: i32, y0: i32, x1: i32, y1: i32) -> Vec<(i32, i32)> {
    let mut pts = Vec::new();
    let dx = (x1 - x0).abs();
    let dy = (y1 - y0).abs();
    let sx = if x1 >= x0 { 1 } else { -1 };
    let sy = if y1 >= y0 { 1 } else { -1 };
    let mut x = x0;
    let mut y = y0;
    if dx >= dy {
        let mut d = 2 * dy - dx;
        for _ in 0..=dx {
            pts.push((x, y));
            if d >= 0 { y += sy; d -= 2 * dx; }
            d += 2 * dy;
            x += sx;
        }
    } else {
        let mut d = 2 * dx - dy;
        for _ in 0..=dy {
            pts.push((x, y));
            if d >= 0 { x += sx; d -= 2 * dy; }
            d += 2 * dx;
            y += sy;
        }
    }
    pts
}

// --- Single pixel line ---

#[test]
fn test_iline_single_pixel() {
    let rex = make_rex3();
    rex3init(&rex);
    let pts = draw_iline_pixels(&rex, 10, 10, 10, 10, 0xAB, DM0_DRAW_ILINE);
    assert_eq!(pts, vec![(10, 10)], "single-pixel line should write exactly one pixel");
}

// --- Horizontal lines ---

#[test]
fn test_iline_horizontal_2px() {
    let rex = make_rex3();
    rex3init(&rex);
    let pts = draw_iline_pixels(&rex, 10, 10, 11, 10, 0xAB, DM0_DRAW_ILINE);
    assert_eq!(pts, vec![(10, 10), (11, 10)], "2-pixel horizontal line");
}

#[test]
fn test_iline_horizontal_8px() {
    let rex = make_rex3();
    rex3init(&rex);
    let pts = draw_iline_pixels(&rex, 10, 20, 17, 20, 0xCD, DM0_DRAW_ILINE);
    let expected: Vec<_> = (10..=17).map(|x| (x, 20)).collect();
    assert_eq!(pts, expected, "8-pixel horizontal line");
}

// --- Vertical lines ---

#[test]
fn test_iline_vertical_2px() {
    let rex = make_rex3();
    rex3init(&rex);
    let pts = draw_iline_pixels(&rex, 20, 10, 20, 11, 0xAB, DM0_DRAW_ILINE);
    assert_eq!(pts, vec![(20, 10), (20, 11)], "2-pixel vertical line");
}

#[test]
fn test_iline_vertical_8px() {
    let rex = make_rex3();
    rex3init(&rex);
    let pts = draw_iline_pixels(&rex, 20, 10, 20, 17, 0xCD, DM0_DRAW_ILINE);
    let expected: Vec<_> = (10..=17).map(|y| (20, y)).collect();
    assert_eq!(pts, expected, "8-pixel vertical line");
}

// --- skip_first / skip_last on horizontal line ---

#[test]
fn test_iline_skipfirst() {
    let rex = make_rex3();
    rex3init(&rex);
    let dm0 = DM0_DRAW_ILINE | (1 << 10); // skipfirst
    let pts = draw_iline_pixels(&rex, 10, 30, 14, 30, 0xEE, dm0);
    // pixels 11..=14 should be drawn, pixel 10 skipped
    let expected: Vec<_> = (11..=14).map(|x| (x, 30)).collect();
    assert_eq!(pts, expected, "skip_first should omit first pixel");
}

#[test]
fn test_iline_skiplast() {
    let rex = make_rex3();
    rex3init(&rex);
    let dm0 = DM0_DRAW_ILINE | (1 << 11); // skiplast
    let pts = draw_iline_pixels(&rex, 10, 30, 14, 30, 0xEE, dm0);
    // pixels 10..=13 should be drawn, pixel 14 skipped
    let expected: Vec<_> = (10..=13).map(|x| (x, 30)).collect();
    assert_eq!(pts, expected, "skip_last should omit last pixel");
}

#[test]
fn test_iline_skipfirst_skiplast() {
    let rex = make_rex3();
    rex3init(&rex);
    let dm0 = DM0_DRAW_ILINE | (1 << 10) | (1 << 11); // skipfirst+skiplast
    let pts = draw_iline_pixels(&rex, 10, 30, 14, 30, 0xEE, dm0);
    // pixels 11..=13 only
    let expected: Vec<_> = (11..=13).map(|x| (x, 30)).collect();
    assert_eq!(pts, expected, "skip_first+skip_last should omit both endpoints");
}

#[test]
fn test_iline_lspattern_stipple() {
    let rex = make_rex3();
    rex3init(&rex);
    // DRAW I_LINE with ENLSPATTERN; pattern has only MSB set.
    let dm0 = DM0_DRAW_ILINE | (1 << 13); // enlspattern
    let bx = 8i32;
    let ex = 14i32;
    let y = 40i32;

    reg(&rex, REX3_DRAWMODE1, DM1_CI8_SRC);
    reg(&rex, REX3_WRMASK, 0xFF);
    reg(&rex, REX3_COLORI, 0);
    reg(&rex, REX3_XYENDI, xy(ex, y));
    reg(&rex, REX3_XYSTARTI, xy(bx, y));
    reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);

    reg(&rex, REX3_LSPATTERN, 0x8000_0000);
    reg(&rex, REX3_LSMODE, 0); // length=17, repeat=1
    reg(&rex, REX3_COLORI, 0xEE);
    reg(&rex, REX3_XYENDI, xy(ex, y));
    reg(&rex, REX3_XYSTARTI, xy(bx, y));
    reg_go(&rex, REX3_DRAWMODE0, dm0);

    let mut drawn = Vec::new();
    for x in bx..=ex {
        if read_pixel(&rex, x, y) & 0xFF == 0xEE {
            drawn.push(x);
        }
    }
    // MSB-only pattern: first pixel on, then off for the rest.
    assert_eq!(drawn, vec![bx], "stippled I_LINE should draw only pattern-on pixels");
}

#[test]
fn test_iline_lsadvlast_advances_on_last_pixel() {
    let rex = make_rex3();
    rex3init(&rex);
    let y = 50i32;
    let dm0 = DM0_DRAW_ILINE | (1 << 13) | (1 << 14); // enlspattern + lsadvlast

    reg(&rex, REX3_DRAWMODE1, DM1_CI8_SRC);
    reg(&rex, REX3_WRMASK, 0xFF);
    reg(&rex, REX3_COLORI, 0);
    reg(&rex, REX3_XYENDI, xy(20, y));
    reg(&rex, REX3_XYSTARTI, xy(8, y));
    reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);

    reg(&rex, REX3_LSPATTERN, 0x8000_0000);
    reg(&rex, REX3_LSMODE, 0);
    // First segment: single pixel at x=8
    reg(&rex, REX3_COLORI, 0xAA);
    reg(&rex, REX3_XYENDI, xy(8, y));
    reg(&rex, REX3_XYSTARTI, xy(8, y));
    reg_go(&rex, REX3_DRAWMODE0, dm0);

    // Second segment without DOSETUP: continue from x=8 (persisted xstart), pat_bit advanced.
    let dm0_cont = dm0 & !(1 << 5); // clear dosetup
    reg(&rex, REX3_COLORI, 0xBB);
    reg(&rex, REX3_XYENDI, xy(10, y));
    reg_go(&rex, REX3_DRAWMODE0, dm0_cont);

    assert_eq!(read_pixel(&rex, 8, y) & 0xFF, 0xAA);
    assert_eq!(read_pixel(&rex, 9, y) & 0xFF, 0, "stipple advanced past MSB — pixel 9 should be off");
    assert_eq!(read_pixel(&rex, 10, y) & 0xFF, 0, "pixel 10 should be off");
}

// --- All octants, r=32 circle, full draw ---

#[test]
fn test_iline_all_octants_full() {
    let rex = make_rex3();
    rex3init(&rex);

    let cx = 100i32;
    let cy = 100i32;
    let r = 32i32;
    let pad = 4;

    // Clear entire working area once.
    reg(&rex, REX3_DRAWMODE1, DM1_CI8_SRC);
    reg(&rex, REX3_WRMASK, 0xFF);
    reg(&rex, REX3_COLORI, 0);
    reg(&rex, REX3_XYENDI,   xy(cx + r + pad, cy + r + pad));
    reg(&rex, REX3_XYSTARTI, xy(cx - r - pad, cy - r - pad));
    reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);

    // Sample every 15 degrees to hit all 8 octants.
    // Each line gets a unique color so stale pixels from other lines don't contaminate.
    for (idx, deg) in (0..360usize).step_by(15).enumerate() {
        let color = (idx + 1) as u8; // 1..24, never 0
        let rad = (deg as f64).to_radians();
        let x1 = cx + (r as f64 * rad.cos()).round() as i32;
        let y1 = cy + (r as f64 * rad.sin()).round() as i32;

        reg(&rex, REX3_COLORI, color as u32);
        reg(&rex, REX3_XYENDI,   xy(x1, y1));
        reg(&rex, REX3_XYSTARTI, xy(cx, cy));
        reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_ILINE);

        // Collect only this line's color in its bounding box.
        let bx = cx.min(x1) - pad; let ex = cx.max(x1) + pad;
        let by = cy.min(y1) - pad; let ey = cy.max(y1) + pad;
        let mut pts: HashSet<(i32,i32)> = HashSet::new();
        for y in by..=ey {
            for x in bx..=ex {
                if read_pixel(&rex, x, y) & 0xFF == color as u32 {
                    pts.insert((x, y));
                }
            }
        }
        let expected: HashSet<(i32,i32)> = bres_pixels(cx, cy, x1, y1).into_iter().collect();

        assert_eq!(pts, expected,
            "octant test deg={deg}: ({cx},{cy})->({x1},{y1})");
    }
}

// --- All octants, single-step (iterate_one) mode ---

#[test]
fn test_iline_all_octants_step() {
    let rex = make_rex3();
    rex3init(&rex);

    let cx = 200i32;
    let cy = 100i32;
    let r = 32i32;
    let pad = 4;

    // Clear entire working area once.
    reg(&rex, REX3_DRAWMODE1, DM1_CI8_SRC);
    reg(&rex, REX3_WRMASK, 0xFF);
    reg(&rex, REX3_COLORI, 0);
    reg(&rex, REX3_XYENDI,   xy(cx + r + pad, cy + r + pad));
    reg(&rex, REX3_XYSTARTI, xy(cx - r - pad, cy - r - pad));
    reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);

    for (idx, deg) in (0..360usize).step_by(15).enumerate() {
        let color = (idx + 1) as u8;
        let rad = (deg as f64).to_radians();
        let x1 = cx + (r as f64 * rad.cos()).round() as i32;
        let y1 = cy + (r as f64 * rad.sin()).round() as i32;

        let dx = (x1 - cx).abs();
        let dy = (y1 - cy).abs();
        let pixel_count = dx.max(dy) + 1;

        // First GO with DOSETUP.
        reg(&rex, REX3_COLORI, color as u32);
        reg(&rex, REX3_XYENDI,   xy(x1, y1));
        reg(&rex, REX3_XYSTARTI, xy(cx, cy));
        reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_ILINE_STEP);

        // Subsequent GOs — one pixel each.
        let dm0_cont = DRAWMODE0_OPCODE_DRAW | DRAWMODE0_ADRMODE_I_LINE_SH;
        for _ in 1..pixel_count {
            reg_go(&rex, REX3_DRAWMODE0, dm0_cont);
        }

        let bx = cx.min(x1) - pad; let ex = cx.max(x1) + pad;
        let by = cy.min(y1) - pad; let ey = cy.max(y1) + pad;
        let mut pts: HashSet<(i32,i32)> = HashSet::new();
        for y in by..=ey {
            for x in bx..=ex {
                if read_pixel(&rex, x, y) & 0xFF == color as u32 {
                    pts.insert((x, y));
                }
            }
        }
        let expected: HashSet<(i32,i32)> = bres_pixels(cx, cy, x1, y1).into_iter().collect();

        assert_eq!(pts, expected,
            "step-mode octant deg={deg}: ({cx},{cy})->({x1},{y1}): got {:?} expected {:?}", pts, expected);
    }
}

// ============================================================================
// F_LINE tests — fractional-endpoint Bresenham correction
// ============================================================================

/// Same 24-direction, 15°-increment sweep as `test_iline_all_octants_full`,
/// but the center (start endpoint) is shifted by exactly half a pixel via
/// REX3_XSTARTF/YSTARTF, and results are checked against `fline_pixels()`
/// (which applies the same fractional correction as `fline_apply_fract`)
/// instead of `bres_pixels()`. This is the "multidirectional line with
/// subpixel precision" test: it would degrade to the plain-integer oracle
/// (and silently pass even with a broken fractional path) if the offset
/// were zero — the half-pixel shift is what actually exercises
/// `fline_apply_fract`'s octant-dependent xf/yf swap-and-mirror logic.
#[test]
fn test_fline_all_octants_half_pixel() {
    let rex = make_rex3();
    rex3init(&rex);

    let cx = 100i32;
    let cy = 100i32;
    let cx_frac = 8; // 0.5px
    let cy_frac = 8; // 0.5px
    let r = 32i32;
    let pad = 4;

    // Clear entire working area once.
    reg(&rex, REX3_DRAWMODE1, DM1_CI8_SRC);
    reg(&rex, REX3_WRMASK, 0xFF);
    reg(&rex, REX3_COLORI, 0);
    reg(&rex, REX3_XYENDI,   xy(cx + r + pad, cy + r + pad));
    reg(&rex, REX3_XYSTARTI, xy(cx - r - pad, cy - r - pad));
    reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);

    for (idx, deg) in (0..360usize).step_by(15).enumerate() {
        let color = (idx + 1) as u8; // 1..24, never 0
        let rad = (deg as f64).to_radians();
        let x1 = cx + (r as f64 * rad.cos()).round() as i32;
        let y1 = cy + (r as f64 * rad.sin()).round() as i32;

        reg(&rex, REX3_COLORI, color as u32);
        // End endpoint must go through the unbiased F-registers too — mixing
        // a biased XYENDI with an unbiased XSTARTF corrupts dx/dy (see the
        // doc comment on write_xstartf).
        write_xendf(&rex, x1, 0);
        write_yendf(&rex, y1, 0);
        write_xstartf(&rex, cx, cx_frac);
        write_ystartf(&rex, cy, cy_frac);
        reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_FLINE);

        let bx = cx.min(x1) - pad; let ex = cx.max(x1) + pad;
        let by = cy.min(y1) - pad; let ey = cy.max(y1) + pad;
        let mut pts: HashSet<(i32,i32)> = HashSet::new();
        for y in by..=ey {
            for x in bx..=ex {
                if read_pixel(&rex, x, y) & 0xFF == color as u32 {
                    pts.insert((x, y));
                }
            }
        }
        let expected: HashSet<(i32,i32)> =
            fline_pixels(cx, cx_frac, cy, cy_frac, x1, y1).into_iter().collect();

        assert_eq!(pts, expected,
            "fline half-pixel octant test deg={deg}: ({cx}.5,{cy}.5)->({x1},{y1}): got {:?} expected {:?}",
            pts, expected);
    }
}

/// Same sweep, but at several distinct fractional offsets (1/4, 1/2, 3/4
/// pixel) to confirm the fractional nibble is consumed proportionally
/// (i.e. actually threaded through the xf/yf octant transform and the d
/// correction term), not just treated as a single boolean "is fractional".
#[test]
fn test_fline_quarter_pixel_offsets() {
    let rex = make_rex3();
    rex3init(&rex);

    let cx = 150i32;
    let cy = 150i32;
    let r = 24i32;
    let pad = 4;

    reg(&rex, REX3_DRAWMODE1, DM1_CI8_SRC);
    reg(&rex, REX3_WRMASK, 0xFF);
    reg(&rex, REX3_COLORI, 0);
    reg(&rex, REX3_XYENDI,   xy(cx + r + pad, cy + r + pad));
    reg(&rex, REX3_XYSTARTI, xy(cx - r - pad, cy - r - pad));
    reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);

    let mut color = 1u8;
    for &frac4 in &[4i32, 8, 12] { // 0.25px, 0.5px, 0.75px
        for &deg in &[0usize, 45, 90, 135, 180, 225, 270, 315] {
            let rad = (deg as f64).to_radians();
            let x1 = cx + (r as f64 * rad.cos()).round() as i32;
            let y1 = cy + (r as f64 * rad.sin()).round() as i32;

            reg(&rex, REX3_COLORI, color as u32);
            write_xendf(&rex, x1, 0);
            write_yendf(&rex, y1, 0);
            write_xstartf(&rex, cx, frac4);
            write_ystartf(&rex, cy, frac4);
            reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_FLINE);

            let bx = cx.min(x1) - pad; let ex = cx.max(x1) + pad;
            let by = cy.min(y1) - pad; let ey = cy.max(y1) + pad;
            let mut pts: HashSet<(i32,i32)> = HashSet::new();
            for y in by..=ey {
                for x in bx..=ex {
                    if read_pixel(&rex, x, y) & 0xFF == color as u32 {
                        pts.insert((x, y));
                    }
                }
            }
            let expected: HashSet<(i32,i32)> =
                fline_pixels(cx, frac4, cy, frac4, x1, y1).into_iter().collect();

            assert_eq!(pts, expected,
                "fline frac4={frac4} deg={deg}: ({cx}+{frac4}/16,{cy}+{frac4}/16)->({x1},{y1}): \
                 got {:?} expected {:?}", pts, expected);

            color = color.wrapping_add(1).max(1);
        }
    }
}

// A_LINE tests: out of scope for this pass — see rules/testing/rex3-fline-fractional-bresenham.md.
// aline_endpoint_skip() (the AWEIGHT-LUT oracle helper) is kept for a future session.

// ============================================================================
// I_LINE line-loop test (XYSTARTI + repeated XYENDI GOs, SKIPLAST, DOSETUP)
// ============================================================================

/// Draw a 10×16 axis-aligned rectangle as a line loop using the IRIX cursor
/// drawing pattern: one XYSTARTI write sets the start, then four XYENDI GOs
/// complete the four sides.  SKIPLAST prevents overdrawing the shared vertex
/// at each corner.  DOSETUP re-derives Bresenham on every GO from xstart.
///
/// Each side is drawn with a distinct CI8 color so we can verify:
///   1. Every expected pixel on each side has the correct color.
///   2. Corner pixels belong to exactly one side (no double-draw from skiplast).
///   3. No "gap" at side starts (xstart not over-advanced from previous segment).
///
/// We test all four starting corners × clockwise + counter-clockwise = 8 rects.
#[test]
fn test_iline_line_loop_rect() {
    let rex = make_rex3();
    rex3init(&rex);

    // DM0: DRAW | I_LINE | DOSETUP | STOPONXY | SKIPLAST
    let dm0_loop: u32 = DRAWMODE0_OPCODE_DRAW | DRAWMODE0_ADRMODE_I_LINE_SH
        | DM0_DOSETUP | DM0_STOPONXY | (1 << 11); // bit11 = skiplast

    reg(&rex, REX3_DRAWMODE1, DM1_CI8_SRC);
    reg(&rex, REX3_WRMASK, 0xFF);

    // Rectangle dimensions (exclusive of endpoint — skiplast omits it).
    // Width=10 (x: +9), Height=16 (y: +15).
    let w = 9;  // dx to far corner
    let h = 15; // dy to far corner

    // Colors: top=1, right=2, bottom=3, left=4
    let colors = [1u8, 2, 3, 4];

    // (corner_x, corner_y, clockwise)
    let cases: &[(i32, i32, bool)] = &[
        (20, 30, true),   // top-left, CW
        (20, 30, false),  // top-left, CCW
        (80, 30, true),   // top-right, CW
        (80, 30, false),  // top-right, CCW
        (20, 80, true),   // bottom-left, CW
        (20, 80, false),  // bottom-left, CCW
        (80, 80, true),   // bottom-right, CW
        (80, 80, false),  // bottom-right, CCW
    ];

    for &(ox, oy, cw) in cases {
        // Four corners of the rectangle.
        let tl = (ox,     oy);
        let tr = (ox + w, oy);
        let br = (ox + w, oy + h);
        let bl = (ox,     oy + h);

        // CW:  TL→TR→BR→BL→TL  (top, right, bottom, left)
        // CCW: TL→BL→BR→TR→TL  (left, bottom, right, top)
        let (p0, p1, p2, p3, p4) = if cw {
            (tl, tr, br, bl, tl)
        } else {
            (tl, bl, br, tr, tl)
        };
        let sides = [
            (p0, p1, colors[0]),
            (p1, p2, colors[1]),
            (p2, p3, colors[2]),
            (p3, p4, colors[3]),
        ];

        // Clear the rect region.
        reg(&rex, REX3_COLORI, 0);
        reg(&rex, REX3_XYENDI,   xy(ox + w + 1, oy + h + 1));
        reg(&rex, REX3_XYSTARTI, xy(ox - 1,     oy - 1));
        reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);

        // XYSTARTI sets start position (no GO).
        reg(&rex, REX3_COLORI, sides[0].2 as u32);
        reg(&rex, REX3_XYENDI,   xy(sides[0].1.0, sides[0].1.1));
        reg(&rex, REX3_XYSTARTI, xy(sides[0].0.0, sides[0].0.1));
        reg_go(&rex, REX3_DRAWMODE0, dm0_loop);

        // Remaining three sides — each XYENDI GO continues from current xstart.
        for &(_, end, color) in &sides[1..] {
            reg(&rex, REX3_COLORI, color as u32);
            reg(&rex, REX3_XYENDI, xy(end.0, end.1));
            reg_go(&rex, REX3_DRAWMODE0, dm0_loop);
        }

        // Verify each side.
        for (si, &((x0, y0), (x1, y1), color)) in sides.iter().enumerate() {
            // Expected pixels: bres_pixels from start to end, minus the endpoint
            // (skiplast omits it — it's the startpoint of the next side).
            let all = bres_pixels(x0, y0, x1, y1);
            let expected_len = all.len() - 1; // skiplast drops endpoint
            let expected: Vec<_> = all[..expected_len].to_vec();

            for &(px, py) in &expected {
                let got = read_pixel(&rex, px, py) & 0xFF;
                assert_eq!(got, color as u32,
                    "case ox={ox} oy={oy} cw={cw} side={si}: pixel ({px},{py}) \
                     expected color {color} got {got}");
            }
        }
    }
}

// ============================================================================
// JIT correctness tests — compare interpreter vs JIT framebuffer output
// ============================================================================
//
// Pattern: run the same draw via interpreter (no JIT), then via JIT (with JIT enabled,
// wait for compile), then assert the framebuffers are identical.

#[cfg(feature = "rex-jit")]
mod jit_tests {
    use super::*;
    use crate::rex3_jit::RexJit;

    #[test]
    fn cid_write_masks_jit() {
        let rex = make_rex3_jit();
        check_cid_write_masks(rex, true);
        rex.stop();
    }

    /// Build a Rex3 with JIT enabled.
    /// A Rex3 with the generated LLVM shaders loaded.
    ///
    /// `Rex3::new` deliberately does not seed them under `cfg(test)` — the
    /// JIT-vs-generic comparison tests need Cranelift to actually run — so a
    /// benchmark that wants to measure the precompiled path has to load them
    /// explicitly.
    fn make_rex3_precompiled() -> &'static Rex3 {
        let rex = make_rex3();
        {
            let mut map = rex.shaders.write();
            for (k, f) in crate::rex3_shaders::SHADERS {
                map.insert(*k, *f);
            }
        }
        // make_rex3 turns dispatch off so ordinary tests exercise the generic
        // path; this fixture exists to measure the precompiled one, so turn it
        // back on.
        #[cfg(feature = "rex-jit")]
        rex.jit_enabled.store(true, std::sync::atomic::Ordering::Relaxed);
        rex
    }

    pub(super) fn make_rex3_jit() -> &'static Rex3 {
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(|| {
                let rex = Box::leak(Box::new(Rex3::new(
                    Arc::new(AtomicU64::new(0)),
                    Arc::new(AtomicU64::new(0)),
                    Arc::new(AtomicU64::new(0)),
                    Arc::new(AtomicU64::new(0)),
                    Arc::new(AtomicU64::new(0)),
                    Arc::new(AtomicU64::new(0)),
                )));
                unsafe {
                    (*rex.fb_rgb.get()).fill(0);
                    (*rex.fb_aux.get()).fill(0);
                }
                // Share the dispatch map so compiled shaders land where execute_go
                // looks for them.
                rex.rex_jit = Some(std::sync::Arc::new(RexJit::new(
                    std::sync::Arc::clone(&rex.shaders),
                )));
                rex.start();
                rex
            })
            .expect("make_rex3_jit thread panicked")
            .join()
            .expect("make_rex3_jit thread panicked")
    }

    /// Dump fb_rgb pixels in region (x0,y0)..(x1,y1) inclusive.
    fn dump_region(rex: &Rex3, x0: i32, y0: i32, x1: i32, y1: i32) -> Vec<u32> {
        let mut out = Vec::new();
        for y in y0..=y1 {
            for x in x0..=x1 {
                out.push(read_pixel(rex, x, y));
            }
        }
        out
    }

    /// Clear fb_rgb in region to 0.
    fn clear_region(rex: &Rex3, x0: i32, y0: i32, x1: i32, y1: i32) {
        unsafe {
            let fb = &mut *rex.fb_rgb.get();
            for y in y0..=y1 {
                for x in x0..=x1 {
                    fb[(y as u32 * 2048 + x as u32) as usize] = 0;
                }
            }
        }
        unsafe {
            let fb = &mut *rex.fb_aux.get();
            for y in y0..=y1 {
                for x in x0..=x1 {
                    fb[(y as u32 * 2048 + x as u32) as usize] = 0;
                }
            }
        }
    }

    /// Core JIT vs interpreter comparison helper.
    /// `setup` writes all registers except the final GO (which calls the draw).
    /// `dm0` is written as the GO trigger. `dm1` is written by setup.
    /// Returns (interp_pixels, jit_pixels) for the given region.
    fn compare_jit_interp(
        x0: i32, y0: i32, x1: i32, y1: i32,
        setup: impl Fn(&Rex3),
        dm0: u32, dm1: u32,
    ) {
        compare_jit_interp_inner(x0, y0, x1, y1, setup, dm0, dm1, false)
    }

    /// `compare_jit_interp` for cases where drawing nothing IS the expected
    /// result (LRONLY aborting the primitive, for example), so the
    /// "interpreter drew nothing" guard must not fire.
    fn compare_jit_interp_expect_blank(
        x0: i32, y0: i32, x1: i32, y1: i32,
        setup: impl Fn(&Rex3),
        dm0: u32, dm1: u32,
    ) {
        compare_jit_interp_inner(x0, y0, x1, y1, setup, dm0, dm1, true)
    }

    fn compare_jit_interp_inner(
        x0: i32, y0: i32, x1: i32, y1: i32,
        setup: impl Fn(&Rex3),
        dm0: u32, dm1: u32,
        expect_blank: bool,
    ) {
        // Interpreter run with JIT dispatch disabled.
        let rex_interp = make_rex3();
        rex3init(rex_interp);
        setup(rex_interp);
        reg_go(rex_interp, REX3_DRAWMODE0, dm0);
        let fb_interp = dump_region(rex_interp, x0, y0, x1, y1);

        // JIT run
        let rex_jit = make_rex3_jit();
        rex3init(rex_jit);
        setup(rex_jit);
        // Drain GFIFO so ctx.clipmode is committed, then read clipmode_key.
        wait(rex_jit);
        let cm = {
            use crate::rex3::CLIPMODE_JIT_KEY_MASK;
            let ctx = unsafe { &*rex_jit.context.get() };
            ctx.clipmode & CLIPMODE_JIT_KEY_MASK
        };
        // First GO: triggers compile + generic fallback.
        reg_go(rex_jit, REX3_DRAWMODE0, dm0);

        // Test builds do not seed the generated LLVM shaders (see Rex3::new), so
        // this shape reaches Cranelift and the comparison below really is
        // JIT-vs-generic rather than generic-vs-itself.
        let compiled = if let Some(ref jit) = rex_jit.rex_jit {
            jit.wait_compiled(dm0, dm1, cm)
        } else { false };
        assert!(compiled, "JIT compile failed for dm0={dm0:#010x} dm1={dm1:#010x} cm={cm:#010x}");

        // Reset fb and re-run via JIT.
        //
        // The "trigger compile" GO above may have executed via the interpreter
        // fallback (compilation happens asynchronously) and, for a dm0 without
        // DOSETUP set, that draw would have advanced ctx.pat_bit/zpat_bit as a
        // side effect (pattern bit position only resets on DOSETUP — see
        // execute_go, rex3.rs). rex3init()/setup() only touch MMIO registers,
        // and pat_bit/zpat_bit have no register mapping (pure internal state),
        // so without this reset the comparison run below would start from
        // whatever pattern position the first GO left behind instead of the
        // fresh state rex_interp's single-GO run used — a test-harness bug
        // that showed up as spurious JIT/interp pixel mismatches for any
        // dm0 lacking DOSETUP (confirmed: jit_lspattern_span_rgb24 and
        // friends all use continuation-style dm0 values with DOSETUP clear).
        clear_region(rex_jit, x0, y0, x1, y1);
        rex3init(rex_jit);
        unsafe {
            let ctx = &mut *rex_jit.context.get();
            ctx.pat_bit = 0;
            ctx.zpat_bit = 0;
        }
        setup(rex_jit);
        reg_go(rex_jit, REX3_DRAWMODE0, dm0);
        let fb_jit = dump_region(rex_jit, x0, y0, x1, y1);

        // A comparison of two blank regions passes whatever the shader does.
        // jit_zpattern_block sat in exactly that state (packed RGB written to
        // an o12.11 colour register drew black on black), so an inverted
        // pattern test in the JIT went undetected. Require that something was
        // actually drawn before trusting the match.
        assert!(expect_blank || fb_interp.iter().any(|&p| p != 0),
            "compare_jit_interp: interpreter drew nothing for dm0={dm0:#010x} \
             dm1={dm1:#010x} — the comparison below would pass vacuously. If a \
             blank result is genuinely expected, use \
             compare_jit_interp_expect_blank.");

        assert_eq!(fb_interp, fb_jit,
            "JIT/interp mismatch: dm0={dm0:#010x} dm1={dm1:#010x}");
    }

    /// RGB24 solid fill block — most common draw mode.
    #[test]
    fn jit_solid_fill_rgb24() {
        let dm1 = DM1_RGB24_SRC;
        let dm0 = DM0_DRAW_BLOCK;
        compare_jit_interp(10, 10, 25, 25,
            |rex| {
                reg(rex, REX3_DRAWMODE1, dm1);
                reg(rex, REX3_WRMASK,    0xFFFFFF);
                reg(rex, REX3_COLORRED,  0x00_A0_50_80u32); // packed RGB24
                reg(rex, REX3_XYENDI,    xy(25, 25));
                reg(rex, REX3_XYSTARTI,  xy(10, 10));
            },
            dm0, dm1,
        );
    }

    /// CI8 solid fill block — 8bpp palette mode.
    #[test]
    fn jit_solid_fill_ci8() {
        let dm1 = DM1_CI8_SRC;
        let dm0 = DM0_DRAW_BLOCK;
        compare_jit_interp(0, 0, 15, 15,
            |rex| {
                reg(rex, REX3_DRAWMODE1, dm1);
                reg(rex, REX3_WRMASK,    0xFF);
                reg(rex, REX3_COLORI,    0x42);
                reg(rex, REX3_XYENDI,    xy(15, 15));
                reg(rex, REX3_XYSTARTI,  xy(0, 0));
            },
            dm0, dm1,
        );
    }

    /// RGB24 XOR logic op block.
    #[test]
    fn jit_logicop_xor_rgb24() {
        // COMPARE must be 0x7 (disabled). Without DRAWMODE1_COMPARE_DISABLE_SH the
        // compare field reads 0, which is an alpha-function test that passes
        // nothing, so every write was inhibited and this test compared two
        // untouched regions -- passing for any shader behaviour. Every working
        // test gets this via DM1_RGB24_SRC; this one open-coded dm1 and lost it.
        let dm1 = DRAWMODE1_PLANES_RGB | (3 << 3) | (1 << 15)
                | DRAWMODE1_COMPARE_DISABLE_SH | DRAWMODE1_LOGICOP_XOR_SH;
        let dm0 = DM0_DRAW_BLOCK;
        compare_jit_interp(0, 0, 15, 15,
            |rex| {
                reg(rex, REX3_DRAWMODE1, dm1);
                reg(rex, REX3_WRMASK,    0xFFFFFF);
                // o12.11 components, not a packed RGB24 word (see
                // jit_zpattern_block): a packed value shifts down to near-zero
                // and XOR against a cleared framebuffer left the region blank,
                // so this compared two empty regions and passed regardless.
                reg(rex, REX3_COLORRED,  200u32 << 11);
                reg(rex, REX3_COLORGRN,  150u32 << 11);
                reg(rex, REX3_COLORBLUE, 100u32 << 11);
                reg(rex, REX3_XYENDI,    xy(15, 15));
                reg(rex, REX3_XYSTARTI,  xy(0, 0));
            },
            dm0, dm1,
        );
    }

    /// RGB24 fastclear block.
    #[test]
    fn jit_fastclear_rgb24() {
        // fastclear = DM1 bit 17; cidmatch must be 0xF for fastclear to activate in interpreter
        let dm1 = DRAWMODE1_PLANES_RGB | (3 << 3) | (1 << 15) | DRAWMODE1_LOGICOP_SRC_SH | (1 << 17);
        let dm0 = DM0_DRAW_BLOCK;
        compare_jit_interp(0, 0, 31, 31,
            |rex| {
                reg(rex, REX3_DRAWMODE1, dm1);
                reg(rex, REX3_COLORVRAM, 0xABCDEF);
                // cidmatch must be 0xF (bits [12:9] of CLIPMODE) for fastclear to fire
                reg(rex, REX3_CLIPMODE,  0xF << 9);
                reg(rex, REX3_XYENDI,    xy(31, 31));
                reg(rex, REX3_XYSTARTI,  xy(0, 0));
            },
            dm0, dm1,
        );
    }

    /// RGB24 solid fill span.
    #[test]
    fn jit_solid_fill_span_rgb24() {
        let dm1 = DM1_RGB24_SRC;
        let dm0 = DM0_DRAW_SPAN;
        compare_jit_interp(5, 5, 20, 5,
            |rex| {
                reg(rex, REX3_DRAWMODE1, dm1);
                reg(rex, REX3_WRMASK,    0xFFFFFF);
                reg(rex, REX3_COLORRED,  0x00_12_34_56u32);
                reg(rex, REX3_XYENDI,    xy(20, 5));
                reg(rex, REX3_XYSTARTI,  xy(5, 5));
            },
            dm0, dm1,
        );
    }

    /// Gouraud shaded span — shade DDA path.
    #[test]
    fn jit_gouraud_shade_span() {
        let dm1 = DM1_RGB24_SRC;
        // DM0 with shade bit 18
        let dm0 = DM0_DRAW_SPAN | (1 << 18);
        compare_jit_interp(0, 0, 15, 0,
            |rex| {
                reg(rex, REX3_DRAWMODE1, dm1);
                reg(rex, REX3_WRMASK,    0xFFFFFF);
                reg(rex, REX3_COLORRED,  10u32 << 11);   // start red=10
                reg(rex, REX3_SLOPERED,  2u32 << 11);    // slope +2/pixel
                reg(rex, REX3_SLOPEGRN,  0);
                reg(rex, REX3_SLOPEBLUE, 0);
                reg(rex, REX3_XYENDI,    xy(15, 0));
                reg(rex, REX3_XYSTARTI,  xy(0, 0));
            },
            dm0, dm1,
        );
    }

    /// Shade span with skipfirst: first pixel is skipped but shade still steps,
    /// so pixel 1 (the first drawn) has color = start + slope.
    #[test]
    fn jit_shade_span_skipfirst() {
        let dm1 = DM1_RGB24_SRC;
        let dm0 = DM0_DRAW_SPAN | (1 << 18) | (1 << 10); // shade + skipfirst
        compare_jit_interp(0, 0, 7, 0,
            |rex| {
                reg(rex, REX3_DRAWMODE1, dm1);
                reg(rex, REX3_WRMASK,    0xFFFFFF);
                reg(rex, REX3_COLORRED,  20u32 << 11);
                reg(rex, REX3_COLORGRN,  0u32);
                reg(rex, REX3_COLORBLUE, 0u32);
                reg(rex, REX3_SLOPERED,  3u32 << 11);
                reg(rex, REX3_SLOPEGRN,  0);
                reg(rex, REX3_SLOPEBLUE, 0);
                reg(rex, REX3_XYENDI,    xy(7, 0));
                reg(rex, REX3_XYSTARTI,  xy(0, 0));
            },
            dm0, dm1,
        );
    }

    /// Shade span with skiplast: last pixel is skipped, all others drawn.
    #[test]
    fn jit_shade_span_skiplast() {
        let dm1 = DM1_RGB24_SRC;
        let dm0 = DM0_DRAW_SPAN | (1 << 18) | (1 << 11); // shade + skiplast
        compare_jit_interp(0, 0, 7, 0,
            |rex| {
                reg(rex, REX3_DRAWMODE1, dm1);
                reg(rex, REX3_WRMASK,    0xFFFFFF);
                reg(rex, REX3_COLORRED,  5u32 << 11);
                reg(rex, REX3_COLORGRN,  0u32);
                reg(rex, REX3_COLORBLUE, 0u32);
                reg(rex, REX3_SLOPERED,  4u32 << 11);
                reg(rex, REX3_SLOPEGRN,  0);
                reg(rex, REX3_SLOPEBLUE, 0);
                reg(rex, REX3_XYENDI,    xy(7, 0));
                reg(rex, REX3_XYSTARTI,  xy(0, 0));
            },
            dm0, dm1,
        );
    }

    /// Shade span that saturates: color ramps up and clamps at 0xFF.
    #[test]
    fn jit_shade_span_saturate() {
        let dm1 = DM1_RGB24_SRC;
        let dm0 = DM0_DRAW_SPAN | (1 << 18); // shade
        compare_jit_interp(0, 0, 15, 0,
            |rex| {
                reg(rex, REX3_DRAWMODE1, dm1);
                reg(rex, REX3_WRMASK,    0xFFFFFF);
                reg(rex, REX3_COLORRED,  240u32 << 11); // start near max
                reg(rex, REX3_COLORGRN,  0u32);
                reg(rex, REX3_COLORBLUE, 0u32);
                reg(rex, REX3_SLOPERED,  10u32 << 11);  // large slope — wraps past 0xFF
                reg(rex, REX3_SLOPEGRN,  0);
                reg(rex, REX3_SLOPEBLUE, 0);
                reg(rex, REX3_XYENDI,    xy(15, 0));
                reg(rex, REX3_XYSTARTI,  xy(0, 0));
            },
            dm0, dm1,
        );
    }

    /// Z-pattern (stipple) block draw.
    #[test]
    fn jit_zpattern_block() {
        let dm1 = DM1_RGB24_SRC;
        // DM0 with enzpattern bit 12
        let dm0 = DM0_DRAW_BLOCK | (1 << 12);
        compare_jit_interp(0, 0, 7, 7,
            |rex| {
                reg(rex, REX3_DRAWMODE1, dm1);
                reg(rex, REX3_WRMASK,    0xFFFFFF);
                // o12.11 fixed-point components, NOT a packed RGB24 word: the
                // shader takes colorred >> 11, so a packed value like
                // 0x00FF8040 shifts down to 0 and the whole region draws black
                // on black. This test compared two all-zero regions and passed
                // for any shader behaviour until that was fixed.
                reg(rex, REX3_COLORRED,  200u32 << 11);
                reg(rex, REX3_COLORGRN,  150u32 << 11);
                reg(rex, REX3_COLORBLUE, 100u32 << 11);
                reg(rex, REX3_ZPATTERN,  0xAAAA_AAAA);  // alternating bits
                reg(rex, REX3_XYENDI,    xy(7, 7));
                reg(rex, REX3_XYSTARTI,  xy(0, 0));
            },
            dm0, dm1,
        );
    }

    /// Gouraud shade block — 2D gradient.
    #[test]
    fn jit_gouraud_shade_block() {
        let dm1 = DM1_RGB24_SRC;
        let dm0 = DM0_DRAW_BLOCK | (1 << 18); // shade bit
        compare_jit_interp(0, 0, 7, 7,
            |rex| {
                reg(rex, REX3_DRAWMODE1, dm1);
                reg(rex, REX3_WRMASK,    0xFFFFFF);
                reg(rex, REX3_COLORRED,  0u32);
                reg(rex, REX3_COLORGRN,  0u32);
                reg(rex, REX3_COLORBLUE, 0u32);
                reg(rex, REX3_SLOPERED,  3u32 << 11);
                reg(rex, REX3_SLOPEGRN,  1u32 << 11);
                reg(rex, REX3_SLOPEBLUE, 0);
                reg(rex, REX3_XYENDI,    xy(7, 7));
                reg(rex, REX3_XYSTARTI,  xy(0, 0));
            },
            dm0, dm1,
        );
    }

    /// Line-stipple (enlspattern) span — the failing verifier case.
    /// dm0=0x00022102: DRAW SPAN STOPONX ENLSPAT LSOPAQUE
    /// dm1=0x3000f319: RGB 24bpp SRC
    #[test]
    fn jit_lspattern_span_rgb24() {
        let dm0 = 0x00022102u32;
        let dm1 = 0x3000f319u32;
        compare_jit_interp(0, 0, 15, 0,
            |rex| {
                reg(rex, REX3_DRAWMODE1, dm1);
                reg(rex, REX3_WRMASK,    0xFFFFFF);
                reg(rex, REX3_COLORRED,  0xFF << 11);
                reg(rex, REX3_LSPATTERN, 0xAAAA_AAAA); // alternating bits
                // lsmode: lsrcount=0, lsrepeat=0, lsrcntsave=0, lslength=0 (length=17)
                reg(rex, REX3_LSMODE,    0);
                reg(rex, REX3_XYENDI,    xy(15, 0));
                reg(rex, REX3_XYSTARTI,  xy(0, 0));
            },
            dm0, dm1,
        );
    }

    /// Shade span with negative slope: color ramps down from high to zero and stays there.
    #[test]
    fn jit_shade_negative_slope() {
        let dm1 = DM1_RGB24_SRC;
        let dm0 = DM0_DRAW_SPAN | (1 << 18); // shade
        // slope = -5 << 11; start at 200 so it hits 0 partway through
        compare_jit_interp(0, 0, 15, 0,
            |rex| {
                reg(rex, REX3_DRAWMODE1, dm1);
                reg(rex, REX3_WRMASK,    0xFFFFFF);
                reg(rex, REX3_COLORRED,  40u32 << 11);
                reg(rex, REX3_COLORGRN,  0u32);
                reg(rex, REX3_COLORBLUE, 0u32);
                // negative slope: bit31=sign, lower bits = magnitude
                reg(rex, REX3_SLOPERED,  0x8000_2800u32); // -5 << 11 = -0x2800
                reg(rex, REX3_SLOPEGRN,  0);
                reg(rex, REX3_SLOPEBLUE, 0);
                reg(rex, REX3_XYENDI,    xy(15, 0));
                reg(rex, REX3_XYSTARTI,  xy(0, 0));
            },
            dm0, dm1,
        );
    }

    /// Shade block with negative slopes on all three channels.
    #[test]
    fn jit_shade_negative_slope_block() {
        let dm1 = DM1_RGB24_SRC;
        let dm0 = DM0_DRAW_BLOCK | (1 << 18); // shade
        compare_jit_interp(0, 0, 7, 7,
            |rex| {
                reg(rex, REX3_DRAWMODE1, dm1);
                reg(rex, REX3_WRMASK,    0xFFFFFF);
                reg(rex, REX3_COLORRED,  0xC0u32 << 11);   // start at 192
                reg(rex, REX3_COLORGRN,  0x80u32 << 11);   // start at 128
                reg(rex, REX3_COLORBLUE, 0x40u32 << 11);   // start at 64
                reg(rex, REX3_SLOPERED,  0x8000_3000u32);  // -6 << 11
                reg(rex, REX3_SLOPEGRN,  0x8000_1800u32);  // -3 << 11
                reg(rex, REX3_SLOPEBLUE, 0x8000_0800u32);  // -1 << 11
                reg(rex, REX3_XYENDI,    xy(7, 7));
                reg(rex, REX3_XYSTARTI,  xy(0, 0));
            },
            dm0, dm1,
        );
    }

    /// RGB 12bpp + SHADE + DITHER block — exact octahedra screensaver draw modes.
    /// dm0=0x002c0126: DRAW BLOCK DOSETUP STOPONX SHADE LRONLY CICLAMP
    /// dm1=0x3009f011: RGB 12bpp host:12bpp logicop:SRC RGB DITHER
    #[test]
    fn jit_shade_rgb12_dither_block() {
        let dm0 = 0x002c0126u32;
        let dm1 = 0x3009f011u32;
        compare_jit_interp(5, 5, 20, 5,
            |rex| {
                reg(rex, REX3_DRAWMODE1, dm1);
                reg(rex, REX3_WRMASK,    0xFFFFFF);
                reg(rex, REX3_COLORRED,  0x80u32 << 11);
                reg(rex, REX3_COLORGRN,  0x40u32 << 11);
                reg(rex, REX3_COLORBLUE, 0x20u32 << 11);
                reg(rex, REX3_SLOPERED,  3u32 << 11);
                reg(rex, REX3_SLOPEGRN,  2u32 << 11);
                reg(rex, REX3_SLOPEBLUE, 1u32 << 11);
                reg(rex, REX3_XYENDI,    xy(20, 5));
                reg(rex, REX3_XYSTARTI,  xy(5, 5));
            },
            dm0, dm1,
        );
    }

    /// RGB 12bpp + SHADE + DITHER block, negative slopes — tests clamp-to-zero.
    #[test]
    fn jit_shade_rgb12_dither_negative() {
        let dm0 = 0x002c0126u32;
        let dm1 = 0x3009f011u32;
        compare_jit_interp(5, 5, 20, 5,
            |rex| {
                reg(rex, REX3_DRAWMODE1, dm1);
                reg(rex, REX3_WRMASK,    0xFFFFFF);
                reg(rex, REX3_COLORRED,  0xA0u32 << 11);
                reg(rex, REX3_COLORGRN,  0x60u32 << 11);
                reg(rex, REX3_COLORBLUE, 0x20u32 << 11);
                reg(rex, REX3_SLOPERED,  0x8000_4000u32); // -8 << 11
                reg(rex, REX3_SLOPEGRN,  0x8000_2000u32); // -4 << 11
                reg(rex, REX3_SLOPEBLUE, 0x8000_1000u32); // -2 << 11
                reg(rex, REX3_XYENDI,    xy(20, 5));
                reg(rex, REX3_XYSTARTI,  xy(5, 5));
            },
            dm0, dm1,
        );
    }

    /// Exact triangle draw mode from simple GL test: ENZPAT + SHADE + LRONLY + CICLAMP + LEN32.
    /// dm0=0x002c9126, dm1=0x3009f009 (RGB 8bpp dither SRC).
    /// Uses zpattern=0xffffff00 (24 on, 8 off).
    #[test]
    fn jit_triangle_exact_mode() {
        let dm0 = 0x002c9126u32;
        let dm1 = 0x3009f009u32;
        compare_jit_interp(0, 0, 32, 0,
            |rex| {
                reg(rex, REX3_DRAWMODE1,  dm1);
                reg(rex, REX3_WRMASK,     0xFF);
                reg(rex, REX3_ZPATTERN,   0xffffff00);
                reg(rex, REX3_COLORRED,   0u32);
                reg(rex, REX3_COLORGRN,   0u32);
                reg(rex, REX3_COLORBLUE,  0x7fffff_u32); // near-max blue
                reg(rex, REX3_SLOPERED,   2u32 << 11);
                reg(rex, REX3_SLOPEGRN,   1u32 << 11);
                reg(rex, REX3_SLOPEBLUE,  0x8000_0800u32); // -1<<11
                reg(rex, REX3_XYSTARTI,   xy(0, 0));
                reg(rex, REX3_XYENDI,     xy(32, 0));
            },
            dm0, dm1,
        );
    }

    /// Multi-row block with STOPONY — tests that shade state persists correctly row-to-row.
    /// dm0=0x002e0126: same as octa shade but with STOPONY added (bit 9).
    #[test]
    fn jit_shade_rgb12_dither_block_multirow() {
        let dm0 = 0x002e0126u32; // DRAW BLOCK DOSETUP STOPONX STOPONY ENZPAT LRONLY CICLAMP SHADE
        let dm1 = 0x3009f011u32;
        compare_jit_interp(5, 5, 20, 10,
            |rex| {
                reg(rex, REX3_DRAWMODE1, dm1);
                reg(rex, REX3_WRMASK,    0xFFFFFF);
                reg(rex, REX3_ZPATTERN,  0xFFFFFFFF); // all-on, no masking
                reg(rex, REX3_COLORRED,  0x80u32 << 11);
                reg(rex, REX3_COLORGRN,  0x20u32 << 11);
                reg(rex, REX3_COLORBLUE, 0x10u32 << 11);
                reg(rex, REX3_SLOPERED,  3u32 << 11);
                reg(rex, REX3_SLOPEGRN,  2u32 << 11);
                reg(rex, REX3_SLOPEBLUE, 1u32 << 11);
                reg(rex, REX3_XYENDI,    xy(20, 10));
                reg(rex, REX3_XYSTARTI,  xy(5, 5));
            },
            dm0, dm1,
        );
    }

    /// LRONLY block: pixels skipped when x_dec=1 (right-to-left), shade always advances.
    /// Set x_dec=1 by making xend < xstart (decreasing x direction).
    #[test]
    fn jit_lronly_block_xdec() {
        let dm0 = DM0_DRAW_BLOCK | (1 << 18) | (1 << 19); // shade + lronly
        let dm1 = DM1_RGB24_SRC;
        // x_dec=1: xstart > xend, octant has XDEC set by dosetup
        // Use dosetup so octant is derived from coordinates
        let dm0 = dm0 | (1 << 5); // dosetup
        // LRONLY aborts the primitive when xstart > xend, so an empty region is
        // the expected result here, not a broken test.
        compare_jit_interp_expect_blank(0, 0, 15, 7,
            |rex| {
                reg(rex, REX3_DRAWMODE1, dm1);
                reg(rex, REX3_WRMASK,    0xFFFFFF);
                reg(rex, REX3_COLORRED,  0x80u32 << 11);
                reg(rex, REX3_COLORGRN,  0u32);
                reg(rex, REX3_COLORBLUE, 0u32);
                reg(rex, REX3_SLOPERED,  5u32 << 11);
                reg(rex, REX3_SLOPEGRN,  0);
                reg(rex, REX3_SLOPEBLUE, 0);
                // xstart > xend → x_dec direction
                reg(rex, REX3_XYSTARTI,  xy(15, 0));
                reg(rex, REX3_XYENDI,    xy(0, 7));
            },
            dm0, dm1,
        );
    }

    /// Exact menu text draw mode: DRAW BLOCK STOPONY ENLSPAT LSOPAQUE, RGB 8bpp.
    /// lsopaque=1: pattern bit=0 → draw colorback; bit=1 → draw foreground color.
    /// This is the most common draw in the popup menu (3416 uses).
    #[test]
    fn jit_lspattern_lsopaque_block_ci8() {
        let dm0 = 0x00022106u32; // DRAW BLOCK STOPONY ENLSPAT LSOPAQUE
        let dm1 = 0x30007109u32; // planes=RGB 8bpp SRC (CI mode)
        compare_jit_interp(0, 0, 15, 7,
            |rex| {
                reg(rex, REX3_DRAWMODE1, dm1);
                reg(rex, REX3_WRMASK,    0xFF);
                reg(rex, REX3_COLORRED,  0x42u32 << 11); // foreground CI index
                reg(rex, REX3_COLORBACK, 0x07u32);       // background CI index
                reg(rex, REX3_LSPATTERN, 0xF0F0_F0F0u32); // alternating nibbles
                reg(rex, REX3_LSMODE,    0);
                reg(rex, REX3_CLIPMODE,  0xF << 9);      // cidmatch=0xF
                reg(rex, REX3_XYENDI,    xy(15, 7));
                reg(rex, REX3_XYSTARTI,  xy(0, 0));
            },
            dm0, dm1,
        );
    }

    /// Timing comparison: full-screen Gouraud-shaded fill, interpreter vs JIT.
    ///
    /// Matches what IRIX actually does: one SPAN GO per scanline (STOPONX, no STOPONY),
    /// color and XYSTARTI/XYENDI set per line, slope constant across the frame.
    /// All scanlines are pushed into the GFIFO without waiting between them; a single
    /// wait_idle() at the end drains the queue.  This measures pure shader throughput
    /// rather than GFIFO round-trip latency.
    ///
    /// Not a correctness test — always passes — output visible with `--nocapture`.
    #[test]
    fn jit_timing_shade_scanlines_fullscreen() {
        const ITERS: u32 = 20;
        const W: i32     = 1279;
        const H: i32     = 1023;

        // SPAN, SHADE, STOPONX (no STOPONY — each GO is exactly one scanline)
        let dm1 = DM1_RGB24_SRC;
        let dm0 = DM0_DRAW_SPAN | (1 << 18); // shade, stoponx already in DM0_DRAW_SPAN

        // Helper: push ITERS full screens as back-to-back scanline GOs, return elapsed nanos.
        // dm1/wrmask/slopes are constant; only colorRGB and xystarti/xyendi change per line.
        let run = |rex: &Rex3| -> u64 {
            reg(rex, REX3_DRAWMODE1,  dm1);
            reg(rex, REX3_WRMASK,     0xFFFFFF);
            reg(rex, REX3_SLOPERED,   2u32 << 11);
            reg(rex, REX3_SLOPEGRN,   1u32 << 11);
            reg(rex, REX3_SLOPEBLUE,  0);

            let start = std::time::Instant::now();
            for i in 0..ITERS {
                let r0 = ((i * 7) % 200) as u32; // vary per frame to prevent dead-code elision
                let g0 = ((i * 3) % 180) as u32;
                let b0 = ((i * 5) % 160) as u32;
                for y in 0..=H {
                    reg(rex, REX3_COLORRED,   r0 << 11);
                    reg(rex, REX3_COLORGRN,   g0 << 11);
                    reg(rex, REX3_COLORBLUE,  b0 << 11);
                    reg(rex, REX3_XYENDI,     xy(W, y));
                    // XYSTARTI+GO in one write — triggers the draw
                    w32(rex, go_addr(REX3_XYSTARTI), xy(0, y));
                }
                // Drain after each full frame so we measure throughput not queue depth
                rex.wait_idle();
            }
            start.elapsed().as_nanos() as u64
        };

        // ── interpreter run ───────────────────────────────────────────────────
        let rex_interp = make_rex3();
        rex3init(rex_interp);
        let interp_ns = run(rex_interp);

        // ── JIT run: trigger compile on first scanline, wait, then run timed loop ──
        let rex_jit = make_rex3_jit();
        rex3init(rex_jit);
        reg(rex_jit, REX3_DRAWMODE1,  dm1);
        reg(rex_jit, REX3_WRMASK,     0xFFFFFF);
        reg(rex_jit, REX3_SLOPERED,   2u32 << 11);
        reg(rex_jit, REX3_SLOPEGRN,   1u32 << 11);
        reg(rex_jit, REX3_SLOPEBLUE,  0);
        reg(rex_jit, REX3_COLORRED,   0u32);
        reg(rex_jit, REX3_COLORGRN,   0u32);
        reg(rex_jit, REX3_COLORBLUE,  0u32);
        reg(rex_jit, REX3_XYENDI,     xy(W, 0));
        reg_go(rex_jit, REX3_XYSTARTI, xy(0, 0)); // triggers compile request
        // Spin until compiled — re-request each iteration in case the channel was full
        // (profile warm-up can fill the 256-entry sync_channel on first boot).
        if let Some(ref jit) = rex_jit.rex_jit {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
            loop {
                if jit.compiled_pairs().contains(&(dm0, dm1, 0xF << CLIPMODE_CIDMATCH_SHIFT)) { break; }
                assert!(std::time::Instant::now() < deadline,
                    "JIT compile timed out for dm0={dm0:#010x} dm1={dm1:#010x}");
                jit.request_compile(dm0, dm1, 0xF << CLIPMODE_CIDMATCH_SHIFT); // retry if channel was full
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        }
        let jit_ns = run(rex_jit);

        let px_per_iter   = (W as u64 + 1) * (H as u64 + 1);
        let interp_mpx    = px_per_iter * ITERS as u64 * 1000 / interp_ns.max(1);
        let jit_mpx       = px_per_iter * ITERS as u64 * 1000 / jit_ns.max(1);
        let speedup_raw   = interp_ns as f64 / jit_ns.max(1) as f64;

        println!(
            "\n=== JIT timing: {} frames of {}×{} Gouraud-shade scanline fill ===\
             \n  Interpreter : {:>8} ms  ({:>6} Mpx/s)\
             \n  JIT         : {:>8} ms  ({:>6} Mpx/s)\
             \n  Speedup     : {:.2}x (JIT is {})\n",
            ITERS, W + 1, H + 1,
            interp_ns / 1_000_000, interp_mpx,
            jit_ns    / 1_000_000, jit_mpx,
            speedup_raw.max(1.0 / speedup_raw.max(f64::EPSILON)),
            if speedup_raw >= 1.0 { "faster" } else { "slower" },
        );
    }

    /// ZPATTERN transparent-miss semantics, checked against explicit expected
    /// pixels rather than an engine-to-engine comparison.
    ///
    /// Written because `jit_zpattern_block` could not catch an inverted pattern
    /// test: it wrote a packed RGB24 word into COLORRED (an o12.11 component
    /// register, so it shifted down to ~0) and compared two all-black regions,
    /// passing for any shader behaviour. This asserts which pixels must change
    /// and which must not, so an inverted or missing test fails loudly.
    ///
    /// ZPATTERN walks MSB-first from bit 31 (see advance_zpat). With pattern
    /// 0xAAAA_AAAA that yields a strict alternation; the phase is taken from the
    /// interpreter (the reference) rather than derived here, since the cursor's
    /// starting position depends on DOSETUP/row-start handling in execute_go.
    /// What this test pins down is the *transparent-miss* contract: missed
    /// pixels must be left completely untouched, drawn pixels must carry the
    /// colour, and the two engines must agree pixel for pixel.
    #[test]
    fn jit_zpattern_transparent_miss_exact() {
        const Y: i32 = 400;
        const N: usize = 32;
        const SENTINEL: u32 = 0x0012_3456;
        let dm0 = DM0_DRAW_SPAN | (1 << 12); // + ENZPATTERN, transparent (no ZPOPAQUE)
        let dm1 = DM1_RGB24_SRC;

        let run = |rex: &Rex3| -> Vec<u32> {
            // zpat_bit has no register mapping and only resets on DOSETUP,
            // which DM0_DRAW_SPAN does not set. Without forcing it here the two
            // engines start the pattern from whatever the previous GO left
            // behind -- the interpreter ran once and began at 0, the JIT's
            // compile-triggering GO left it at 31, and every pixel disagreed.
            // Same harness hazard compare_jit_interp documents.
            unsafe { (*rex.context.get()).zpat_bit = 31; }
            // Fill with a sentinel so an untouched pixel is distinguishable
            // from a pixel drawn black — the flaw that made the old test inert.
            {
                let fb = unsafe { &mut *rex.fb_rgb.get() };
                for x in 0..N { fb[Y as usize * 2048 + x] = SENTINEL; }
            }
            reg(rex, REX3_DRAWMODE0,  dm0);
            reg(rex, REX3_DRAWMODE1,  dm1);
            reg(rex, REX3_WRMASK,     0xFFFFFF);
            reg(rex, REX3_ZPATTERN,   0xAAAA_AAAA);
            reg(rex, REX3_COLORRED,   200u32 << 11);
            reg(rex, REX3_COLORGRN,   150u32 << 11);
            reg(rex, REX3_COLORBLUE,  100u32 << 11);
            reg(rex, REX3_XYENDI,     xy(N as i32 - 1, Y));
            reg_go(rex, REX3_XYSTARTI, xy(0, Y));
            rex.wait_idle();
            let fb = unsafe { &*rex.fb_rgb.get() };
            (0..N).map(|x| fb[Y as usize * 2048 + x]).collect()
        };

        let check = |got: &[u32], who: &str| {
            // Strict alternation, and exactly half the span drawn: a shader that
            // drew everything, drew nothing, or inverted the test all fail here.
            let drawn_count = (0..N).filter(|&x| got[x] != SENTINEL).count();
            assert_eq!(drawn_count, N / 2,
                "{who}: {drawn_count} of {N} pixels drawn, expected exactly half \
                 (0xAAAA_AAAA alternates)");
            // Absolute phase, not derived from the output. Both fixtures enter
            // the draw with zpat_bit = 31 (set explicitly below), and bit 31 of
            // 0xAAAA_AAAA is set, so pixel 0 MUST be drawn. Deriving the phase
            // from got[0] instead made the test blind to an inverted pattern
            // test: inverting it shifts the alternation by one, both engines
            // shift together, and every self-calibrating check still passes.
            for x in 0..N {
                let drawn = got[x] != SENTINEL;
                let expect_drawn = x % 2 == 0;
                assert_eq!(drawn, expect_drawn,
                    "{who}: pixel {x} {} but should {} (value {:#010x}) — \
                     alternation broken",
                    if drawn { "was drawn" } else { "was NOT drawn" },
                    if expect_drawn { "be drawn" } else { "be left alone" },
                    got[x]);
            }
            // Missed pixels must be untouched, not drawn black: that distinction
            // is what the sentinel fill exists for.
            let missed = (0..N).find(|&x| got[x] == SENTINEL).expect("some pixel missed");
            assert_eq!(got[missed], SENTINEL,
                "{who}: missed pixel {missed} was modified");
            // And the drawn pixels must carry the colour, not black.
            let first_drawn = (0..N).find(|&x| got[x] != SENTINEL).unwrap();
            assert_ne!(got[first_drawn] & 0xFFFFFF, 0,
                "{who}: drawn pixels are black — the colour registers were not \
                 interpreted as o12.11 components and this check is vacuous");
        };

        let rex_i = make_rex3();
        rex3init(rex_i);
        unsafe {
            let ctx = &mut *rex_i.context.get();
            ctx.zpat_bit = 31;
            ctx.pat_bit  = 31;
        }
        let interp = run(rex_i);
        check(&interp, "interpreter");

        let rex_j = make_rex3_jit();
        rex3init(rex_j);
        let _ = run(rex_j); // request compile
        {
            let cm = unsafe { (*rex_j.context.get()).clipmode } & CLIPMODE_JIT_KEY_MASK;
            if let Some(ref jit) = rex_j.rex_jit {
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
                while !jit.compiled_pairs().iter().any(|&(a, _, c)| a == dm0 && c == cm) {
                    assert!(std::time::Instant::now() < deadline,
                        "JIT compile timed out for dm0={dm0:#010x} cm={cm:#x}");
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
            }
        }
        // Reset the pattern cursor before the measured run. zpat_bit/pat_bit are
        // internal state with no register mapping, and they only reset on
        // DOSETUP — which DM0_DRAW_SPAN does not set. The JIT fixture draws
        // twice (once to request the compile), so without this it enters the
        // measured draw one bit further along than the interpreter's single
        // draw, and every pixel mismatches. compare_jit_interp carries the same
        // fixup for the same reason.
        unsafe {
            let ctx = &mut *rex_j.context.get();
            ctx.zpat_bit = 31;
            ctx.pat_bit  = 31;
        }
        rex_j.jit_go_count.store(0, Ordering::Relaxed);
        let jit = run(rex_j);
        assert!(rex_j.jit_go_count.load(Ordering::Relaxed) > 0,
            "no GO dispatched through the JIT — this would compare interpreter to itself");
        check(&jit, "jit");
        assert_eq!(interp, jit, "JIT and interpreter disagree on ZPATTERN span");
    }

    /// Scissor (SMASK0) clipping must agree between interpreter and JIT.
    ///
    /// Written because the JIT's smask loads are now gated on `ensmask_key`
    /// (the compile-time clipmode bits), so a wrong gate would silently stop
    /// clipping — and nothing else in this suite draws with clipping enabled:
    /// SMASK0X appeared only in `rex3init` and a register read/write check, so
    /// the whole suite passed either way.
    ///
    /// Draws a span wider than the scissor rect and requires both engines to
    /// clip it identically, then repeats with ensmask off to confirm the same
    /// span is *not* clipped (otherwise a shader that always clips would also
    /// pass).
    #[test]
    fn jit_smask_clip_matches_interpreter() {
        const Y: i32     = 200;
        const X0: i32    = 0;
        const X1: i32    = 120;
        const CLIP_LO: i32 = 30;
        const CLIP_HI: i32 = 80;
        const WIDTH: usize = 140;

        let draw = |rex: &Rex3, clip: bool| -> Vec<u32> {
            {
                let fb = unsafe { &mut *rex.fb_rgb.get() };
                for x in 0..WIDTH { fb[Y as usize * 2048 + x] = 0; }
            }
            // SMASK0X/Y pack (min << 16) | max. The comparison in
            // calculate_fb_address is against x_curr/y_curr, which are the
            // COORD_BIAS-shifted coordinates (xy() biases them), so the bounds
            // must be biased too — unbiased values clip everything away.
            let lo = (CLIP_LO + REX3_COORD_BIAS) as u32 & 0xFFFF;
            let hi = (CLIP_HI + REX3_COORD_BIAS) as u32 & 0xFFFF;
            reg(rex, REX3_SMASK0X, (lo << 16) | hi);
            let ylo = (0 + REX3_COORD_BIAS) as u32 & 0xFFFF;
            let yhi = (1023 + REX3_COORD_BIAS) as u32 & 0xFFFF;
            reg(rex, REX3_SMASK0Y, (ylo << 16) | yhi);
            // Keep cidmatch = 0xF (disabled) as rex3init sets it — writing a
            // bare 0/1 here zeroes cidmatch, and cidmatch=0 rejects every
            // pixel, so nothing draws at all and the test looks like it is
            // clipping when it is really drawing nothing.
            let cidm = 0xFu32 << CLIPMODE_CIDMATCH_SHIFT;
            reg(rex, REX3_CLIPMODE, cidm | if clip { 1 } else { 0 });
            reg(rex, REX3_DRAWMODE0, DM0_DRAW_SPAN);
            reg(rex, REX3_DRAWMODE1, DM1_RGB24_SRC);
            reg(rex, REX3_WRMASK,    0xFFFFFF);
            reg(rex, REX3_COLORRED,  255u32 << 11);
            reg(rex, REX3_COLORGRN,  255u32 << 11);
            reg(rex, REX3_COLORBLUE, 255u32 << 11);
            reg(rex, REX3_XYENDI,    xy(X1, Y));
            reg_go(rex, REX3_XYSTARTI, xy(X0, Y));
            rex.wait_idle();
            let fb = unsafe { &*rex.fb_rgb.get() };
            (0..WIDTH).map(|x| fb[Y as usize * 2048 + x] & 0xFFFFFF).collect()
        };

        for clip in [true, false] {
            let rex_i = make_rex3();
            rex3init(rex_i);
            let interp = draw(rex_i, clip);

            let rex_j = make_rex3_jit();
            rex3init(rex_j);
            let _ = draw(rex_j, clip); // request compile
            {
                let cm = unsafe { (*rex_j.context.get()).clipmode } & CLIPMODE_JIT_KEY_MASK;
                if let Some(ref jit) = rex_j.rex_jit {
                    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
                    while !jit.compiled_pairs().iter()
                        .any(|&(a, _, c)| a == DM0_DRAW_SPAN && c == cm)
                    {
                        assert!(std::time::Instant::now() < deadline,
                            "JIT compile timed out for clip={clip} cm={cm:#x}");
                        std::thread::sleep(std::time::Duration::from_millis(5));
                    }
                }
            }
            rex_j.jit_go_count.store(0, Ordering::Relaxed);
            let jit = draw(rex_j, clip);

            assert!(rex_j.jit_go_count.load(Ordering::Relaxed) > 0,
                "clip={clip}: no GO dispatched through the JIT");

            let lit = |v: &Vec<u32>| (0..WIDTH).filter(|&x| v[x] != 0).count();
            if clip {
                // Both engines must clip to the scissor rect, not draw the full span.
                assert!(lit(&interp) > 0 && lit(&interp) < (X1 - X0) as usize,
                    "interpreter did not clip: {} px lit of {}", lit(&interp), X1 - X0);
                for x in 0..WIDTH {
                    let inside = (x as i32) >= CLIP_LO && (x as i32) <= CLIP_HI;
                    if !inside {
                        assert_eq!(interp[x], 0, "interpreter drew outside scissor at x={x}");
                    }
                }
            } else {
                assert!(lit(&interp) >= (X1 - X0) as usize,
                    "ensmask off should not clip: only {} px lit", lit(&interp));
            }
            assert_eq!(interp, jit,
                "clip={clip}: JIT scissor result differs from interpreter\n  \
                 interp lit {} px, jit lit {} px", lit(&interp), lit(&jit));
        }
    }

    // NOTE: no LSADVLAST interpreter-vs-JIT test here on purpose. The JIT was
    // advancing the line stipple on the last pixel unconditionally where the
    // interpreter gates on LSADVLAST (fixed in rex3_jit/compiler.rs), but two
    // attempts to build a test that observes the difference both passed with
    // the bug reinstated: DOSETUP-per-segment resets pat_bit, and the
    // LSSAVE/LSRESTORE bracketing used for connected stipples round-trips
    // lsrcount. A test that passes either way is worse than none. Deferred
    // until a real XL Indy is available to say which behaviour is correct --
    // see the UNVERIFIED note in compiler.rs's line-shader pattern advance.

    /// GFIFO pressure sweep: how much can we draw per second, as primitives shrink?
    ///
    /// `jit_timing_shade_scanlines_fullscreen` already goes through the GFIFO
    /// (`reg()` calls `rex.write32()`, the real bus entry point), but it draws
    /// 1280-pixel scanlines from 5 register writes -- about 0.004 queue entries
    /// per pixel. Guest GL drawing small triangles is nothing like that: tens of
    /// entries per primitive covering ~32 pixels, call it 1 entry/pixel, some
    /// 250x denser. So the fullscreen number says the queue is fast at *low*
    /// entry density and nothing about high density.
    ///
    /// This runs each configuration for a fixed wall-clock budget and reports
    /// how much it managed, rather than timing a fixed amount of work: at
    /// ~2000 Mpx/s a fixed-work run finishes in milliseconds and measures
    /// mostly noise. Short spans mean more GOs and more register writes for the
    /// same fill, so the Mpx/s curve across span lengths isolates what queue
    /// traffic costs. Flat means the queue is free at any density; collapsing
    /// means per-entry cost dominates once primitives get small -- the regime
    /// real GL content lives in.
    ///
    /// Both plain Gouraud and ZPATTERN-masked spans are measured. ZPATTERN is
    /// Indy's depth path (the GL driver compares in software and hands REX3 a
    /// 32-bit coverage mask per 32-pixel span), so it costs an extra register
    /// write per span *and* a per-pixel mask test -- exactly what depth-tested
    /// content pays, and the guest-side numbers show depth is expensive.
    ///
    /// Not an assertion test: it prints a table. Run with
    /// `cargo test --release --features rex-jit gfifo_pressure_sweep -- --nocapture`
    /// (add `--ignored`; it is ignored by default since it burns real seconds).
    #[test]
    #[ignore = "benchmark: runs for several seconds of wall clock"]
    fn gfifo_pressure_sweep() {
        /// Minimum wall-clock per sample. The loop runs whole batches and stops
        /// once this has elapsed, so a sample is always *at least* this long and
        /// usually a little over — which is why every rate below divides the
        /// pixels actually drawn by the nanoseconds actually measured, never by
        /// an assumed budget.
        const BUDGET: std::time::Duration = std::time::Duration::from_millis(1000);
        /// Samples per cell; the median is reported, with min/max as spread.
        /// Three at >=1s each, rather than one: the short-span rows varied 2.6x
        /// run to run on single samples (span 32 read 163 / 429 / 229 Mpx/s),
        /// and one number gives the reader no way to see that.
        const SAMPLES: usize = 3;
        const SPAN_LENS: [i32; 6] = [1280, 256, 64, 32, 16, 8];
        /// Spans per timing check — checking the clock every span would itself
        /// cost more than the draw at short lengths.
        const BATCH: u64 = 256;

        let dm1 = DM1_RGB24_SRC;
        let dm0_plain = DM0_DRAW_SPAN | (1 << 18);              // shade + stoponx
        // Flat fill: same span, no SHADE. The primitive colour is then constant
        // for the whole draw, which is the case the entry-block colour hoist in
        // rex3_jit/compiler.rs targets -- and the common one in real content
        // (solid rectangles, window fills), which the Gouraud rows never cover.
        let dm0_flat  = DM0_DRAW_SPAN;
        // Depth mode as the GL driver actually drives it: ENZPATTERN (bit 12)
        // for the coverage mask AND LENGTH32 (bit 15), which hard-caps the draw
        // at 32 pixels (see execute_go's `length32 && pixel_count > 32`). That
        // cap is why ZPATTERN is a *fixed* 32-pixel row below rather than part
        // of the span sweep: a longer span would not draw longer, it would just
        // recycle the same 32-bit mask over pixels it never reaches.
        let dm0_zpat  = DM0_DRAW_SPAN | (1 << 18) | (1 << 12) | (1 << 15);

        // Draw spans of `len` pixels for BUDGET, through the GFIFO.
        // Returns (pixels drawn, queue entries pushed, elapsed nanos).
        let run = |rex: &Rex3, len: i32, zpat: bool, shade: bool| -> (u64, u64, u64) {
            let dm0_sel = if zpat { dm0_zpat }
                          else if shade { dm0_plain }
                          else { dm0_flat };
            reg(rex, REX3_DRAWMODE0,  dm0_sel);
            reg(rex, REX3_DRAWMODE1,  dm1);
            reg(rex, REX3_WRMASK,     0xFFFFFF);
            reg(rex, REX3_SLOPERED,   2u32 << 11);
            reg(rex, REX3_SLOPEGRN,   1u32 << 11);
            reg(rex, REX3_SLOPEBLUE,  0);

            // 5 writes + 1 GO per span, plus the ZPATTERN mask when enabled --
            // the extra queue entry per span that depth content actually pays.
            let per_span = if zpat { 7 } else { 6 };
            let mut spans = 0u64;
            let start = std::time::Instant::now();
            loop {
                for _ in 0..BATCH {
                    let i = spans;
                    // Walk across the framebuffer so successive draws touch
                    // different lines rather than rewriting one hot row.
                    let y = (i % 1024) as i32;
                    let x0 = ((i / 1024) as i32 * len) % (1280 - len).max(1);
                    if zpat {
                        // A fresh mask per span, as the GL driver emits after
                        // each 32-pixel software depth compare. Varying it (not
                        // a constant) keeps the per-pixel test honest and stops
                        // the value being hoisted; the alternating-ish patterns
                        // reject roughly half the pixels.
                        reg(rex, REX3_ZPATTERN, 0xAAAA_AAAAu32 ^ (i as u32).wrapping_mul(2654435761));
                    }
                    reg(rex, REX3_COLORRED,  ((i * 7 % 200) as u32) << 11);
                    reg(rex, REX3_COLORGRN,  ((i * 3 % 180) as u32) << 11);
                    reg(rex, REX3_COLORBLUE, ((i * 5 % 160) as u32) << 11);
                    reg(rex, REX3_XYENDI,    xy(x0 + len - 1, y));
                    w32(rex, go_addr(REX3_XYSTARTI), xy(x0, y));
                    spans += 1;
                }
                if start.elapsed() >= BUDGET { break; }
            }
            rex.wait_idle();
            let ns = start.elapsed().as_nanos() as u64;
            // LENGTH32 caps the draw at 32 pixels however long the span is, so
            // count what was actually rasterized, not what was requested.
            let drawn = if zpat { len.min(32) } else { len } as u64;
            (spans * drawn, spans * per_span, ns)
        };

        let rex_interp = make_rex3();
        rex3init(rex_interp);
        // Same device, but with the generated LLVM shaders loaded — the path a
        // shipping build actually takes for a covered shape.
        let rex_pre = make_rex3_precompiled();
        rex3init(rex_pre);
        let rex_jit = make_rex3_jit();
        rex3init(rex_jit);

        // Force both shader variants compiled before timing.
        for dm0 in [dm0_flat, dm0_plain, dm0_zpat] {
            reg(rex_jit, REX3_DRAWMODE0, dm0);
            reg(rex_jit, REX3_DRAWMODE1, dm1);
            reg(rex_jit, REX3_XYENDI,    xy(63, 0));
            reg_go(rex_jit, REX3_XYSTARTI, xy(0, 0));
            if let Some(ref jit) = rex_jit.rex_jit {
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
                while !jit.compiled_pairs().contains(&(dm0, dm1, 0)) {
                    assert!(std::time::Instant::now() < deadline,
                            "JIT compile timed out for dm0={dm0:#010x}");
                    jit.request_compile(dm0, dm1, 0);
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
            }
        }

        // Median of `SAMPLES` runs, plus the observed min/max, in Mpx/s.
        let sample = |rex: &Rex3, len: i32, zpat: bool, shade: bool| -> (u64, u64, u64, u64, u64, u64) {
            let mut rates = Vec::with_capacity(SAMPLES);
            let mut entries = 0u64;
            let mut last_ms = 0u64;
            let mut last_spans = 0u64;
            for _ in 0..SAMPLES {
                let (px, e, ns) = run(rex, len, zpat, shade);
                entries = e;
                last_ms = ns / 1_000_000;
                last_spans = px / if zpat { len.min(32) } else { len } as u64;
                // Measured pixels over measured nanos — never an assumed budget.
                rates.push(px * 1000 / ns.max(1));
            }
            rates.sort_unstable();
            (rates[SAMPLES / 2], rates[0], rates[SAMPLES - 1], entries, last_ms, last_spans)
        };

        // --- self-validation -------------------------------------------------
        // A throughput number from draws that never touched a pixel is worse
        // than no number: it looks like a fast configuration. Likewise a "JIT"
        // column that is really the interpreter. Check both before reporting,
        // on both engines and both modes, rather than trusting the setup.
        // Report the exact keys this benchmark draws, so the generated table
        // can be built for them: the corpus comes from real IRIX drawing and
        // does not otherwise contain these synthetic shapes.
        {
            let cm = unsafe { (*rex_interp.context.get()).clipmode } & CLIPMODE_JIT_KEY_MASK;
            for (name, d0) in [("flat", dm0_flat), ("plain", dm0_plain), ("zpat", dm0_zpat)] {
                let nd1 = crate::rex3_shape::normalize_dm1(dm1, DRAWMODE0_OPCODE_DRAW);
                println!("  bench shape {name:<5}: dm0={d0:#010x} dm1={nd1:#010x} cm={cm:#010x}  \
                          precompiled={}", crate::rex3_shaders::lookup(d0, nd1, cm).is_some());
            }
        }

        for (rex, engine) in [(rex_interp, "generic"), (rex_pre, "precomp"), (rex_jit, "jit")] {
            for (zpat, mode) in [(false, "plain"), (true, "zpat")] {
                // Clear a known region, draw one span into it, confirm it moved.
                let probe_y = 700;
                {
                    let fb = unsafe { &mut *rex.fb_rgb.get() };
                    for x in 0..64usize { fb[probe_y as usize * 2048 + x] = 0; }
                }
                let go_before  = rex.jit_go_count.load(Ordering::Relaxed);
                let int_before = rex.interp_go_count.load(Ordering::Relaxed);

                reg(rex, REX3_DRAWMODE0, if zpat { dm0_zpat } else { dm0_plain });
                reg(rex, REX3_DRAWMODE1, dm1);
                reg(rex, REX3_WRMASK,    0xFFFFFF);
                reg(rex, REX3_SLOPERED,  2u32 << 11);
                reg(rex, REX3_SLOPEGRN,  1u32 << 11);
                reg(rex, REX3_SLOPEBLUE, 0);
                if zpat { reg(rex, REX3_ZPATTERN, 0xFFFF_FFFF); }
                reg(rex, REX3_COLORRED,  200u32 << 11);
                reg(rex, REX3_COLORGRN,  180u32 << 11);
                reg(rex, REX3_COLORBLUE, 160u32 << 11);
                reg(rex, REX3_XYENDI,    xy(31, probe_y));
                reg_go(rex, REX3_XYSTARTI, xy(0, probe_y));

                let changed = {
                    let fb = unsafe { &*rex.fb_rgb.get() };
                    (0..32usize).filter(|&x| fb[probe_y as usize * 2048 + x] != 0).count()
                };
                assert!(changed > 0,
                    "{engine}/{mode}: draw mutated no pixels — the benchmark would be timing nothing");

                let jit_gos = rex.jit_go_count.load(Ordering::Relaxed) - go_before;
                let int_gos = rex.interp_go_count.load(Ordering::Relaxed) - int_before;
                let counters_live = cfg!(feature = "rexdiag");
                println!("  validate {engine:>6}/{mode:<5}: {changed:>2}/32 px written, \
                          GOs jit={jit_gos} interp={int_gos}{}",
                         if counters_live { "" } else { " (counters disabled: rexdiag off)" });
                // The dispatch counters live behind `rexdiag`, which `lightning`
                // turns off (they are lock-prefixed RMWs on the per-GO path).
                // Under lightning both read 0 because they are compiled out, not
                // because dispatch failed, so this check would make the sweep
                // unrunnable on exactly the build it most needs to measure.
                #[cfg(feature = "rexdiag")]
                if engine == "jit" {
                    assert!(jit_gos > 0,
                        "{engine}/{mode}: no GO dispatched through the JIT (jit={jit_gos} \
                         interp={int_gos}) — the 'jit' column would just be the interpreter");
                }
                #[cfg(not(feature = "rexdiag"))]
                let _ = (jit_gos, int_gos);
            }
        }

        for (label, zpat) in [("flat (no SHADE)", false), ("plain Gouraud", false), ("ZPATTERN-masked (LENGTH32: 32px draws)", true)] {
            println!("\n=== GFIFO pressure sweep: {label} ({} ms per cell) ===",
                     BUDGET.as_millis());
            println!("  {:>6}  {:>10}  {:>7}  {:>7}  {:>7}  {:>9}  {:>9}",
                     "span", "entries/px", "generic", "precomp", "jit",
                     "precomp x", "jit x");
            println!("  {:>6}  {:>10}  {:>7}  {:>7}  {:>7}",
                     "", "", "Mpx/s", "Mpx/s", "Mpx/s");
            // ZPATTERN draws are capped at 32 pixels, so sweeping span length
            // past that measures nothing new -- one row is the whole story.
            let lens: &[i32] = if zpat { &[32, 16, 8] } else { &SPAN_LENS };
            for &len in lens {
                let shade = label != "flat (no SHADE)";
                let (i_mpx, i_lo, i_hi, entries, i_ms, i_spans) = sample(rex_interp, len, zpat, shade);
                let (p_mpx, _p_lo, _p_hi, _, _p_ms, _p_spans)      = sample(rex_pre,    len, zpat, shade);
                let (j_mpx, j_lo, j_hi, _, j_ms, j_spans)           = sample(rex_jit,    len, zpat, shade);
                let drawn = if zpat { len.min(32) } else { len } as u64;
                // entries/px uses pixels actually rasterized (LENGTH32 caps
                // ZPATTERN draws at 32), not the span length requested.
                let per_px = entries as f64
                    / (entries / if zpat { 7 } else { 6 }).max(1) as f64
                    / drawn as f64;
                // Ratio must come from WORK DONE, not elapsed time: every
                // sample runs for the same wall-clock budget, so i_ns/j_ns is
                // ~1.00 by construction and says nothing. (It printed a
                // reassuring "1.00x" next to cells where the JIT was doing
                // half the interpreter's work.)
                println!("  {:>6}  {:>10.3}  {:>7}  {:>7}  {:>7}  {:>8.2}x {:>8.2}x",
                         len, per_px,
                         i_mpx, p_mpx, j_mpx,
                         p_mpx as f64 / i_mpx.max(1) as f64,
                         j_mpx as f64 / i_mpx.max(1) as f64);
                let _ = (i_lo, i_hi, j_lo, j_hi, i_ms, j_ms, i_spans, j_spans);
            }
        }
        println!("\n  For comparison: guest-side gltest --bench ~44 Mpx/s,\n                    \x20 --bench --depth ~20 Mpx/s.\n");
    }

    /// Verify Gouraud interpolation pixel-by-pixel: R ramps from 255 down to 0 across 256 pixels.
    ///
    /// slope = (0 - 255) / 255 = -1 per pixel = -1 << 11 in o12.11 fixed-point.
    /// colorred_start = 255 << 11.
    /// Expected fb[x] red channel = 255 - x  for x in 0..=255, then 0 for x > 255.
    ///
    /// Tests both interpreter and JIT paths and prints pixel values on mismatch.
    #[test]
    fn jit_shade_ramp_255_to_0() {
        // dm0: DRAW SPAN STOPONX SHADE
        let dm0 = DM0_DRAW_SPAN | (1 << 18);
        let dm1 = DM1_RGB24_SRC;

        // slope = -1 per pixel = -1 in integer part = -1 << 11 in o12.11
        // from_slope_red: negative slope written as 0x80000000 | magnitude
        // magnitude of -1<<11 = 2048 = 0x800
        let slope_neg1: u32 = 0x8000_0800; // -1 << 11

        let check = |rex: &Rex3, label: &str| {
            reg(rex, REX3_DRAWMODE0,  dm0);
            reg(rex, REX3_DRAWMODE1,  dm1);
            reg(rex, REX3_WRMASK,     0xFFFFFF);
            reg(rex, REX3_COLORRED,   255u32 << 11);
            reg(rex, REX3_COLORGRN,   0u32);
            reg(rex, REX3_COLORBLUE,  0u32);
            reg(rex, REX3_SLOPERED,   slope_neg1);
            reg(rex, REX3_SLOPEGRN,   0);
            reg(rex, REX3_SLOPEBLUE,  0);
            reg(rex, REX3_XYENDI,     xy(255, 0));
            reg_go(rex, REX3_XYSTARTI, xy(0, 0));

            let mut failed = false;
            for x in 0..=255i32 {
                let px   = read_pixel(rex, x, 0);
                let r    = px & 0xFF;
                let expected = (255 - x) as u32;
                if r != expected {
                    if !failed {
                        println!("{label}: ramp mismatch (first 8 errors):");
                        failed = true;
                    }
                    println!("  x={x:3}: got r={r:#04x} ({r}), expected {expected:#04x} ({expected})");
                }
            }
            assert!(!failed, "{label}: ramp 255→0 failed");
        };

        // Interpreter
        let rex_i = make_rex3();
        rex3init(rex_i);
        check(rex_i, "interp");

        // JIT
        let rex_j = make_rex3_jit();
        rex3init(rex_j);
        // First draw triggers compile + interp fallback; wait then re-draw
        reg(rex_j, REX3_DRAWMODE0,  dm0);
        reg(rex_j, REX3_DRAWMODE1,  dm1);
        reg(rex_j, REX3_WRMASK,     0xFFFFFF);
        reg(rex_j, REX3_COLORRED,   255u32 << 11);
        reg(rex_j, REX3_COLORGRN,   0u32);
        reg(rex_j, REX3_COLORBLUE,  0u32);
        reg(rex_j, REX3_SLOPERED,   slope_neg1);
        reg(rex_j, REX3_SLOPEGRN,   0);
        reg(rex_j, REX3_SLOPEBLUE,  0);
        reg(rex_j, REX3_XYENDI,     xy(255, 0));
        reg_go(rex_j, REX3_XYSTARTI, xy(0, 0));
        if let Some(ref jit) = rex_j.rex_jit {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
            loop {
                if jit.compiled_pairs().contains(&(dm0, dm1, 0xF << CLIPMODE_CIDMATCH_SHIFT)) { break; }
                assert!(std::time::Instant::now() < deadline, "JIT compile timeout");
                jit.request_compile(dm0, dm1, 0xF << CLIPMODE_CIDMATCH_SHIFT);
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        }
        // Clear and re-run via JIT
        for x in 0..=255i32 { unsafe { (*rex_j.fb_rgb.get())[(x as u32) as usize] = 0; } }
        check(rex_j, "jit");
    }

    /// Exact octahedra blur mode: SPAN + SHADE + ENZPAT + DITHER + RGB12 + LEN32 + STOPONX + DOSETUP.
    /// zpattern = 0x88888888 (every 4th pixel). Tests that zpat_bit resets to 31 each GO.
    #[test]
    fn jit_shade_enzpat_span_rgb12_dither() {
        let dm0 = 0x00049122u32; // DRAW SPAN DOSETUP STOPONX ENZPAT LEN32 SHADE
        let dm1 = 0x3009f011u32; // RGB 12bpp dither SRC
        compare_jit_interp(0, 0, 40, 0,
            |rex| {
                reg(rex, REX3_DRAWMODE1,  dm1);
                reg(rex, REX3_WRMASK,     0xFFFFFF);
                reg(rex, REX3_ZPATTERN,   0x88888888);
                reg(rex, REX3_COLORRED,   0x80u32 << 11);
                reg(rex, REX3_COLORGRN,   0x20u32 << 11);
                reg(rex, REX3_COLORBLUE,  0x10u32 << 11);
                reg(rex, REX3_SLOPERED,   2u32 << 11);
                reg(rex, REX3_SLOPEGRN,   1u32 << 11);
                reg(rex, REX3_SLOPEBLUE,  0u32);
                reg(rex, REX3_XYSTARTI,   xy(0, 0));
                reg(rex, REX3_XYENDI,     xy(40, 0));
            },
            dm0, dm1,
        );
    }

    /// Same as above but DBLSRC variant (dm1=0x3009f031).
    #[test]
    fn jit_shade_enzpat_span_rgb12_dblsrc() {
        let dm0 = 0x00049122u32;
        let dm1 = 0x3009f031u32; // same + DBLSRC
        compare_jit_interp(0, 0, 40, 0,
            |rex| {
                reg(rex, REX3_DRAWMODE1,  dm1);
                reg(rex, REX3_WRMASK,     0xFFFFFF);
                reg(rex, REX3_ZPATTERN,   0x88888888);
                reg(rex, REX3_COLORRED,   0x80u32 << 11);
                reg(rex, REX3_COLORGRN,   0x20u32 << 11);
                reg(rex, REX3_COLORBLUE,  0x10u32 << 11);
                reg(rex, REX3_SLOPERED,   2u32 << 11);
                reg(rex, REX3_SLOPEGRN,   1u32 << 11);
                reg(rex, REX3_SLOPEBLUE,  0u32);
                reg(rex, REX3_XYSTARTI,   xy(0, 0));
                reg(rex, REX3_XYENDI,     xy(40, 0));
            },
            dm0, dm1,
        );
    }

    /// SCR2SCR block copy (CI8): mirrors test_ng1_scrtoscr.
    /// Source at (0,0)..(7,7) painted 0xCC; copied to (16,0)..(23,7) via XYMOVE=(16,0).
    #[test]
    fn jit_scr2scr_ci8_block() {
        let dm1 = DM1_CI8_SRC;
        let dm0 = DM0_SCR2SCR;
        compare_jit_interp(16, 0, 23, 7,
            |rex| {
                // Paint source region
                reg(rex, REX3_DRAWMODE1, DM1_CI8_SRC);
                reg(rex, REX3_WRMASK,   0xFF);
                reg(rex, REX3_COLORI,   0xCC);
                reg(rex, REX3_XYSTARTI, xy(0, 0));
                reg(rex, REX3_XYENDI,   xy(7, 7));
                reg_go(rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);
                // Set up SCR2SCR registers
                reg(rex, REX3_DRAWMODE1, DM1_CI8_SRC);
                reg(rex, REX3_WRMASK,   0xFF);
                reg(rex, REX3_XYMOVE,   (16u32 << 16) | 0);
                reg(rex, REX3_XYSTARTI, xy(0, 0));
                reg(rex, REX3_XYENDI,   xy(7, 7));
            },
            dm0, dm1,
        );
    }

    /// I_LINE CI8 solid line — covers the basic Bresenham loop.
    #[test]
    fn jit_iline_ci8_solid() {
        let dm1 = DM1_CI8_SRC;
        let dm0 = DM0_DRAW_ILINE;
        // Diagonal line (0,0)..(15,10) — exercises all octant paths collectively.
        compare_jit_interp(0, 0, 17, 12,
            |rex| {
                reg(rex, REX3_DRAWMODE1, dm1);
                reg(rex, REX3_WRMASK,   0xFF);
                reg(rex, REX3_COLORI,   0x77);
                reg(rex, REX3_XYENDI,   xy(15, 10));
                reg(rex, REX3_XYSTARTI, xy(0, 0));
            },
            dm0, dm1,
        );
    }

    /// I_LINE RGB24 solid line — gently sloped, x-major.
    #[test]
    fn jit_iline_rgb24_solid() {
        let dm1 = DM1_RGB24_SRC;
        let dm0 = DM0_DRAW_ILINE;
        compare_jit_interp(0, 0, 22, 8,
            |rex| {
                reg(rex, REX3_DRAWMODE1, dm1);
                reg(rex, REX3_WRMASK,   0xFFFFFF);
                reg(rex, REX3_COLORRED, 0xFF << 11);
                reg(rex, REX3_COLORGRN, 0x80 << 11);
                reg(rex, REX3_COLORBLUE, 0x40 << 11);
                reg(rex, REX3_XYENDI,   xy(20, 7));
                reg(rex, REX3_XYSTARTI, xy(0, 0));
            },
            dm0, dm1,
        );
    }

    /// I_LINE with SKIPFIRST and SKIPLAST — edge skip flags.
    #[test]
    fn jit_iline_skipfirst_skiplast() {
        let dm1 = DM1_CI8_SRC;
        let dm0 = DM0_DRAW_ILINE | (1 << 10) | (1 << 11); // SKIPFIRST | SKIPLAST
        compare_jit_interp(0, 0, 14, 6,
            |rex| {
                reg(rex, REX3_DRAWMODE1, dm1);
                reg(rex, REX3_WRMASK,   0xFF);
                reg(rex, REX3_COLORI,   0xAA);
                reg(rex, REX3_XYENDI,   xy(12, 5));
                reg(rex, REX3_XYSTARTI, xy(0, 0));
            },
            dm0, dm1,
        );
    }

    /// I_LINE step mode (iterate_one): one pixel per GO, driven via compare_jit_interp
    /// using the continuation dm0 (no DOSETUP, no STOPONXY).
    #[test]
    fn jit_iline_step_mode() {
        let dm1 = DM1_CI8_SRC;
        // step mode dm0: same adrmode but no stoponx/stopony/dosetup
        let dm0_cont = DRAWMODE0_OPCODE_DRAW | DRAWMODE0_ADRMODE_I_LINE_SH;
        // We compare a single continuation step (after DOSETUP already set up Bresenham state).
        // Setup: issue the DOSETUP GO first (interpreter-only, not JIT), then compare one step.
        compare_jit_interp(0, 0, 14, 8,
            |rex| {
                reg(rex, REX3_DRAWMODE1, dm1);
                reg(rex, REX3_WRMASK,   0xFF);
                reg(rex, REX3_COLORI,   0x55);
                reg(rex, REX3_XYENDI,   xy(12, 6));
                reg(rex, REX3_XYSTARTI, xy(0, 0));
                // First GO with DOSETUP to establish Bresenham state
                reg_go(rex, REX3_DRAWMODE0, DM0_DRAW_ILINE_STEP);
                // Clear the starting pixel so we only compare the stepped pixels
                // (subsequent GOs with dm0_cont drive the comparison)
            },
            dm0_cont, dm1,
        );
    }

    /// I_LINE with Gouraud shading (shade DDA active on a line).
    #[test]
    fn jit_iline_shade() {
        let dm1 = DM1_RGB24_SRC;
        let dm0 = DM0_DRAW_ILINE | (1 << 18); // SHADE bit
        compare_jit_interp(0, 0, 16, 6,
            |rex| {
                reg(rex, REX3_DRAWMODE1, dm1);
                reg(rex, REX3_WRMASK,   0xFFFFFF);
                // Start color
                reg(rex, REX3_COLORRED,  0xFF << 11);
                reg(rex, REX3_COLORGRN,  0x00 << 11);
                reg(rex, REX3_COLORBLUE, 0x80 << 11);
                // Shade slopes (one step per pixel along the line major axis)
                reg(rex, REX3_SLOPERED,  ((-8i32) as u32) << 11);
                reg(rex, REX3_SLOPEGRN,  (8u32) << 11);
                reg(rex, REX3_SLOPEBLUE, 0);
                reg(rex, REX3_XYENDI,   xy(14, 4));
                reg(rex, REX3_XYSTARTI, xy(0, 0));
            },
            dm0, dm1,
        );
    }

    /// F_LINE with a fractional (half-pixel) start endpoint. Exercises the
    /// same fractional-endpoint Bresenham correction (fline_apply_fract) as
    /// the interpreter-only test_fline_all_octants_half_pixel, but now
    /// through the JIT compile path — this is the parity check that would
    /// have caught the JIT's silent I_LINE-degradation bug for F_LINE before
    /// emit_draw_iline gained real fractional support.
    #[test]
    fn jit_fline_half_pixel_octants() {
        let dm1 = DM1_CI8_SRC;
        let dm0 = DM0_DRAW_FLINE;
        // One representative direction per octant (8 of the 24-direction sweep
        // used by the interpreter-only test) — full 24x would mean 24 separate
        // JIT compiles, expensive for a parity check that just needs octant coverage.
        // Deliberately ASYMMETRIC fractional offsets (xf != yf) at non-45-degree
        // angles for most cases: a symmetric xf==yf offset at an exact 45-degree
        // multiple is a degenerate geometry where the fractional correction term
        // algebraically cancels against the plain I_LINE d — such cases pass even
        // when the correction is missing entirely, so they can't catch a JIT
        // regression on their own (this bit a first draft of this test).
        let cx = 100i32;
        let cy = 100i32;
        let r = 24i32;
        for (deg, xf, yf) in [
            (0, 3, 11), (45, 5, 2), (90, 12, 7), (135, 1, 9),
            (180, 8, 8), (225, 14, 3), (270, 6, 13), (315, 10, 1),
        ] {
            let rad = (deg as f64).to_radians();
            let x1 = cx + (r as f64 * rad.cos()).round() as i32;
            let y1 = cy + (r as f64 * rad.sin()).round() as i32;
            compare_jit_interp(
                cx.min(x1) - 4, cy.min(y1) - 4, cx.max(x1) + 4, cy.max(y1) + 4,
                |rex| {
                    reg(rex, REX3_DRAWMODE1, dm1);
                    reg(rex, REX3_WRMASK,   0xFF);
                    reg(rex, REX3_COLORI,   0x99);
                    write_xendf(rex, x1, 0);
                    write_yendf(rex, y1, 0);
                    write_xstartf(rex, cx, xf);
                    write_ystartf(rex, cy, yf);
                },
                dm0, dm1,
            );
        }
    }

    /// SCR2SCR block copy (RGB24): copy a colored rectangle.
    #[test]
    fn jit_scr2scr_rgb24_block() {
        let dm1 = DM1_RGB24_SRC;
        let dm0 = DM0_SCR2SCR;
        compare_jit_interp(20, 0, 27, 7,
            |rex| {
                // Paint source region in RGB24
                reg(rex, REX3_DRAWMODE1, DM1_RGB24_SRC);
                reg(rex, REX3_WRMASK,   0xFFFFFF);
                reg(rex, REX3_COLORRED, 0xAA << 11);
                reg(rex, REX3_COLORGRN, 0x55 << 11);
                reg(rex, REX3_COLORBLUE, 0x11 << 11);
                reg(rex, REX3_XYSTARTI, xy(0, 0));
                reg(rex, REX3_XYENDI,   xy(7, 7));
                reg_go(rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);
                // Set up SCR2SCR registers (xymove shifts dst by +20,0)
                reg(rex, REX3_DRAWMODE1, DM1_RGB24_SRC);
                reg(rex, REX3_WRMASK,   0xFFFFFF);
                reg(rex, REX3_XYMOVE,   (20u32 << 16) | 0);
                reg(rex, REX3_XYSTARTI, xy(0, 0));
                reg(rex, REX3_XYENDI,   xy(7, 7));
            },
            dm0, dm1,
        );
    }

    // ── HOSTRW JIT tests ──────────────────────────────────────────────────────
    //
    // For HOSTRW, compare_jit_interp can't be used directly since the GO trigger
    // is a write to HOSTRW0 rather than DRAWMODE0.  Each test runs the same HOSTRW
    // sequence on both an interpreter-only instance and a JIT instance, then compares
    // the resulting framebuffer region.
    //
    // Pattern:
    //   1. Set up both instances identically (DM1, WRMASK, coords, DM0 in SET space).
    //   2. Run the HOSTRW GO sequence on the interpreter instance.
    //   3. On the JIT instance: trigger compile with one GO, wait for compile, reset fb,
    //      then replay the full GO sequence via JIT.
    //   4. Compare framebuffer regions.

    /// CI8 HOSTW 32-bit packed: 4 CI8 pixels per 32-bit word.
    /// Mirrors test_hostw_ci8_write_block_32bit.
    #[test]
    fn jit_hostw_ci8_block_32bit() {
        let dm0 = DM0_HOSTW_BLOCK;
        let dm1 = DM1_CI8_HOSTRW;
        let word: u32 = (0x11u32 << 24) | (0x22 << 16) | (0x33 << 8) | 0x44;

        let setup = |rex: &Rex3| {
            reg(rex, REX3_DRAWMODE1, dm1);
            reg(rex, REX3_WRMASK,    0xFF);
            reg(rex, REX3_XYENDI,    xy(3, 0));
            reg(rex, REX3_XYSTARTI,  xy(0, 0));
            reg(rex, REX3_DRAWMODE0, dm0); // SET — no draw yet
        };

        // Interpreter run
        let rex_i = make_rex3();
        rex3init(rex_i);
        setup(rex_i);
        write_hostrw32(rex_i, word);
        wait(rex_i);
        let fb_interp: Vec<u32> = (0..4).map(|x| read_pixel(rex_i, x, 0) & 0xFF).collect();

        // JIT run: trigger compile, wait, replay
        let rex_j = make_rex3_jit();
        rex3init(rex_j);
        setup(rex_j);
        write_hostrw32(rex_j, word); // triggers compile request
        wait(rex_j);
        if let Some(ref jit) = rex_j.rex_jit {
            assert!(jit.wait_compiled(dm0, dm1, 0xF << CLIPMODE_CIDMATCH_SHIFT),
                "JIT compile failed dm0={dm0:#010x} dm1={dm1:#010x}");
        }
        // Reset and replay via JIT
        clear_region(rex_j, 0, 0, 3, 0);
        rex3init(rex_j);
        setup(rex_j);
        write_hostrw32(rex_j, word);
        wait(rex_j);
        let fb_jit: Vec<u32> = (0..4).map(|x| read_pixel(rex_j, x, 0) & 0xFF).collect();

        assert_eq!(fb_interp, fb_jit,
            "CI8 HOSTW JIT/interp mismatch: interp={fb_interp:?} jit={fb_jit:?}");
    }

    /// RGB24 HOSTW 32-bit: 1 pixel per word (non-packed, hostdepth=3).
    /// Mirrors test_hostw_rgb24_write_block_32bit.
    #[test]
    fn jit_hostw_rgb24_block_32bit() {
        let dm0 = DM0_HOSTW_BLOCK;
        let dm1 = DM1_RGB24_HOSTRW;
        let pixels: &[u32] = &[0x0000FF, 0x00FF00, 0xFF0000, 0xAABBCC];

        let setup = |rex: &Rex3| {
            reg(rex, REX3_DRAWMODE1, dm1);
            reg(rex, REX3_WRMASK,    0xFFFFFF);
            reg(rex, REX3_XYENDI,    xy(3, 0));
            reg(rex, REX3_XYSTARTI,  xy(0, 0));
            reg(rex, REX3_DRAWMODE0, dm0);
        };

        // Interpreter run
        let rex_i = make_rex3();
        rex3init(rex_i);
        setup(rex_i);
        for &p in pixels { write_hostrw32(rex_i, p); }
        wait(rex_i);
        let fb_interp: Vec<u32> = (0..4).map(|x| read_pixel(rex_i, x, 0) & 0xFFFFFF).collect();

        // JIT run
        let rex_j = make_rex3_jit();
        rex3init(rex_j);
        setup(rex_j);
        write_hostrw32(rex_j, pixels[0]); // trigger compile
        wait(rex_j);
        if let Some(ref jit) = rex_j.rex_jit {
            assert!(jit.wait_compiled(dm0, dm1, 0xF << CLIPMODE_CIDMATCH_SHIFT),
                "JIT compile failed dm0={dm0:#010x} dm1={dm1:#010x}");
        }
        clear_region(rex_j, 0, 0, 3, 0);
        rex3init(rex_j);
        setup(rex_j);
        for &p in pixels { write_hostrw32(rex_j, p); }
        wait(rex_j);
        let fb_jit: Vec<u32> = (0..4).map(|x| read_pixel(rex_j, x, 0) & 0xFFFFFF).collect();

        assert_eq!(fb_interp, fb_jit,
            "RGB24 HOSTW JIT/interp mismatch: interp={fb_interp:?} jit={fb_jit:?}");
    }

    /// RGB24 HOSTW 64-bit: 2 pixels per 64-bit word.
    /// Mirrors test_hostw_rgb24_write_block_64bit.
    #[test]
    fn jit_hostw_rgb24_block_64bit() {
        let dm0 = DM0_HOSTW_BLOCK;
        let dm1 = DM1_RGB24_HOSTRW64;
        let p0: u32 = 0x00FF0000;
        let p1: u32 = 0x0000FF00;
        let word64: u64 = ((p0 as u64) << 32) | (p1 as u64);

        let setup = |rex: &Rex3| {
            reg(rex, REX3_DRAWMODE1, dm1);
            reg(rex, REX3_WRMASK,    0xFFFFFF);
            reg(rex, REX3_XYENDI,    xy(1, 0));
            reg(rex, REX3_XYSTARTI,  xy(0, 0));
            reg(rex, REX3_DRAWMODE0, dm0);
        };

        let rex_i = make_rex3();
        rex3init(rex_i);
        setup(rex_i);
        write_hostrw64(rex_i, word64);
        wait(rex_i);
        let fb_interp: Vec<u32> = (0..2).map(|x| read_pixel(rex_i, x, 0) & 0xFFFFFF).collect();

        let rex_j = make_rex3_jit();
        rex3init(rex_j);
        setup(rex_j);
        write_hostrw64(rex_j, word64);
        wait(rex_j);
        if let Some(ref jit) = rex_j.rex_jit {
            assert!(jit.wait_compiled(dm0, dm1, 0xF << CLIPMODE_CIDMATCH_SHIFT),
                "JIT compile failed dm0={dm0:#010x} dm1={dm1:#010x}");
        }
        clear_region(rex_j, 0, 0, 1, 0);
        rex3init(rex_j);
        setup(rex_j);
        write_hostrw64(rex_j, word64);
        wait(rex_j);
        let fb_jit: Vec<u32> = (0..2).map(|x| read_pixel(rex_j, x, 0) & 0xFFFFFF).collect();

        assert_eq!(fb_interp, fb_jit,
            "RGB24 HOSTW64 JIT/interp mismatch: interp={fb_interp:?} jit={fb_jit:?}");
    }

    /// CI8 HOSTR 32-bit: read 4 CI8 pixels from framebuffer into one word.
    /// Mirrors test_hostr_ci8_read_block_32bit.
    #[test]
    fn jit_hostr_ci8_block_32bit() {
        // DM0 for READ
        let dm0_read = DRAWMODE0_OPCODE_READ | DRAWMODE0_ADRMODE_BLOCK_SH | DM0_STOPONXY;
        let dm1 = DM1_CI8_HOSTRW;

        // Fill 4 pixels with known CI8 values using interpreter
        let fill_rex = make_rex3();
        rex3init(fill_rex);
        reg(fill_rex, REX3_DRAWMODE1, DM1_CI8_SRC);
        reg(fill_rex, REX3_WRMASK, 0xFF);
        let pixels_in = [0x11u32, 0x22, 0x33, 0x44];
        for (x, &v) in pixels_in.iter().enumerate() {
            reg(fill_rex, REX3_COLORI,    v);
            reg(fill_rex, REX3_XYSTARTI,  xy(x as i32, 0));
            reg_go(fill_rex, REX3_XYENDI, xy(x as i32, 0));
        }

        let setup_read = |rex: &Rex3| {
            // Copy framebuffer from fill_rex
            unsafe {
                let src = &*fill_rex.fb_rgb.get();
                let dst = &mut *rex.fb_rgb.get();
                dst[0..4].copy_from_slice(&src[0..4]);
            }
            reg(rex, REX3_DRAWMODE1, dm1);
            reg(rex, REX3_WRMASK,    0xFF);
            reg(rex, REX3_XYENDI,    xy(3, 0));
            reg(rex, REX3_XYSTARTI,  xy(0, 0));
        };

        // Interpreter HOSTR read
        let rex_i = make_rex3();
        rex3init(rex_i);
        setup_read(rex_i);
        reg_go(rex_i, REX3_DRAWMODE0, dm0_read); // triggers first batch
        let word_interp = read_hostrw32_last(rex_i);

        // JIT HOSTR read
        let rex_j = make_rex3_jit();
        rex3init(rex_j);
        setup_read(rex_j);
        reg_go(rex_j, REX3_DRAWMODE0, dm0_read); // triggers compile + first batch
        if let Some(ref jit) = rex_j.rex_jit {
            assert!(jit.wait_compiled(dm0_read, dm1, 0xF << CLIPMODE_CIDMATCH_SHIFT),
                "JIT compile failed dm0={dm0_read:#010x} dm1={dm1:#010x}");
        }
        // Re-run via JIT
        rex3init(rex_j);
        setup_read(rex_j);
        reg_go(rex_j, REX3_DRAWMODE0, dm0_read);
        let word_jit = read_hostrw32_last(rex_j);

        assert_eq!(word_interp, word_jit,
            "CI8 HOSTR JIT/interp mismatch: interp={word_interp:#010x} jit={word_jit:#010x}");
    }

    /// RGB24 HOSTR 32-bit: read 1 RGB24 pixel per word, multiple words.
    /// Mirrors test_hostr_rgb24_read_block_32bit.
    #[test]
    fn jit_hostr_rgb24_block_32bit() {
        let dm0_read = DRAWMODE0_OPCODE_READ | DRAWMODE0_ADRMODE_BLOCK_SH | DM0_STOPONXY;
        let dm1 = DM1_RGB24_HOSTRW;
        let pixels_in: &[u32] = &[0x112233, 0x445566, 0x778899, 0xAABBCC];
        let width = pixels_in.len() as i32;

        // Fill pixels
        let fill_rex = make_rex3();
        rex3init(fill_rex);
        reg(fill_rex, REX3_DRAWMODE1, DM1_RGB24_SRC);
        reg(fill_rex, REX3_WRMASK, 0xFFFFFF);
        for (x, &v) in pixels_in.iter().enumerate() {
            reg(fill_rex, REX3_COLORRED,  v << 11 & !0x7FF | v >> 11 & 0x7FF); // use raw color
        }
        // Simpler: write fb_rgb directly
        unsafe {
            let fb = &mut *fill_rex.fb_rgb.get();
            for (x, &v) in pixels_in.iter().enumerate() { fb[x] = v; }
        }

        let setup_read = |rex: &Rex3| {
            unsafe {
                let src = &*fill_rex.fb_rgb.get();
                let dst = &mut *rex.fb_rgb.get();
                dst[..width as usize].copy_from_slice(&src[..width as usize]);
            }
            reg(rex, REX3_DRAWMODE1, dm1);
            reg(rex, REX3_WRMASK,    0xFFFFFF);
            reg(rex, REX3_XYENDI,    xy(width - 1, 0));
            reg(rex, REX3_XYSTARTI,  xy(0, 0));
        };

        let read_words = |rex: &Rex3| -> Vec<u32> {
            setup_read(rex);
            reg_go(rex, REX3_DRAWMODE0, dm0_read);
            let mut words = Vec::new();
            for i in 0..width {
                let w = if i < width - 1 { read_hostrw32(rex) } else { read_hostrw32_last(rex) };
                words.push(w);
            }
            words
        };

        let rex_i = make_rex3();
        rex3init(rex_i);
        let words_interp = read_words(rex_i);

        let rex_j = make_rex3_jit();
        rex3init(rex_j);
        // Trigger compile
        setup_read(rex_j);
        reg_go(rex_j, REX3_DRAWMODE0, dm0_read);
        read_hostrw32_last(rex_j); // drain
        if let Some(ref jit) = rex_j.rex_jit {
            assert!(jit.wait_compiled(dm0_read, dm1, 0xF << CLIPMODE_CIDMATCH_SHIFT),
                "JIT compile failed dm0={dm0_read:#010x} dm1={dm1:#010x}");
        }
        let words_jit = read_words(rex_j);

        assert_eq!(words_interp, words_jit,
            "RGB24 HOSTR JIT/interp mismatch:\n  interp={words_interp:08x?}\n  jit   ={words_jit:08x?}");
    }

    /// Multi-row HOSTR block with STOPONY *not* set — mirrors the real cursor
    /// save/restore blit (IRIX login-screen text cursor): each row is its own
    /// one-row primitive (no hardware row auto-advance), width wider than one
    /// packed host word so each row takes 2 GOs, and — critically — the shader
    /// compiles (interpreter fallback) partway through the sequence so later
    /// rows dispatch via the freshly-compiled JIT entry while earlier rows ran
    /// on the interpreter, exactly like the real async-compile timing.
    ///
    /// All existing jit_hostr_*/jit_hostw_* tests use DM0_STOPONXY (single GO,
    /// full block in one shot) and never exercise this multi-GO, multi-row,
    /// no-STOPONY shape — the gap that let this bug ship.
    #[test]
    fn jit_hostr_ci8_block_multirow_no_stopony() {
        let dm0_read = DRAWMODE0_OPCODE_READ | DRAWMODE0_ADRMODE_BLOCK_SH | DM0_COLORHOST;
        let dm1 = DM1_CI8_HOSTRW; // 8bpp packed, 4 pixels/word
        let width = 6i32;  // 2 words/row (ceil(6/4))
        let height = 3i32;
        let words_per_row = 2usize;

        // Fill a width x height region with distinct per-pixel values so any
        // word/row misalignment shows up as a mismatch rather than coincidentally
        // matching (e.g. an all-same-value fill would hide an off-by-one).
        let fill_rex = make_rex3();
        rex3init(fill_rex);
        unsafe {
            let fb = &mut *fill_rex.fb_rgb.get();
            for y in 0..height {
                for x in 0..width {
                    fb[(y as u32 * 2048 + x as u32) as usize] = (0x10 + y * width + x) as u32;
                }
            }
        }

        let setup_read = |rex: &Rex3| {
            unsafe {
                let src = &*fill_rex.fb_rgb.get();
                let dst = &mut *rex.fb_rgb.get();
                for y in 0..height {
                    for x in 0..width {
                        let idx = (y as u32 * 2048 + x as u32) as usize;
                        dst[idx] = src[idx];
                    }
                }
            }
            reg(rex, REX3_DRAWMODE1, dm1);
            reg(rex, REX3_WRMASK,    0xFF);
        };

        // Drive the exact GO sequence a row-at-a-time cursor blit uses: fresh
        // XYSTARTI/XYENDI per row, DRAWMODE0 GO (no DOSETUP, no STOPONY) starts
        // the row, then (words_per_row - 1) HOSTRW0 GO-space reads drain the
        // rest of that row's words. ystart/xsave carry over via ctx state
        // exactly as on real hardware — only the first row's registers are
        // (re)written here since that's all the real driver does too (see
        // rex3.log capture: XYSTARTI/XYENDI written once, then 40 bare GOs).
        let read_all_words = |rex: &Rex3| -> Vec<u32> {
            let mut words = Vec::new();
            reg(rex, REX3_XYSTARTI, xy(0, 0));
            reg(rex, REX3_XYENDI,   xy(width - 1, 0));
            reg_go(rex, REX3_DRAWMODE0, dm0_read);
            words.push(read_hostrw32_last(rex));
            let total_words = words_per_row * height as usize;
            for _ in 1..total_words {
                words.push(read_hostrw32(rex));
            }
            words
        };

        let rex_i = make_rex3();
        rex3init(rex_i);
        setup_read(rex_i);
        let words_interp = read_all_words(rex_i);

        // JIT run: prime compilation with a throwaway pass over the same
        // primitive shape first (separate region so it doesn't consume any
        // of the words compared below), wait for it to land, then run the
        // real comparison pass. Because compilation is requested from
        // whatever GO first sees this (dm0, dm1, cm) key and happens on a
        // background thread, the *real* emulator can and does have a
        // primitive's later rows dispatch via JIT after its earlier rows
        // already ran on the interpreter — this priming pass reproduces that
        // "already compiled by the time the real primitive runs" end state
        // without relying on racy mid-primitive timing.
        let rex_j = make_rex3_jit();
        rex3init(rex_j);
        setup_read(rex_j);
        let _ = read_all_words(rex_j); // throwaway: triggers compile request
        if let Some(ref jit) = rex_j.rex_jit {
            assert!(jit.wait_compiled(dm0_read, dm1, 0xF << CLIPMODE_CIDMATCH_SHIFT),
                "JIT compile failed dm0={dm0_read:#010x} dm1={dm1:#010x}");
        }
        rex3init(rex_j);
        setup_read(rex_j);
        let words_jit = read_all_words(rex_j);

        assert_eq!(words_interp, words_jit,
            "CI8 HOSTR multirow (no STOPONY) JIT/interp mismatch:\n  interp={words_interp:08x?}\n  jit   ={words_jit:08x?}");
    }

    /// Stress test: large multi-row, multi-word-per-row HOSTR block readbacks with
    /// STOPONXY set (DM0_READ_BLOCK's real shape — each row wraps mid-primitive
    /// while more rows remain), comparing interpreter vs pre-warmed JIT output.
    ///
    /// Guards against a real bug (row-wrap-under-STOPONY: the JIT shader's
    /// host_xstop_v word-boundary threshold is computed once at shader entry and
    /// goes stale across an internal row wrap, so a JIT-compiled READ_BLOCK could
    /// silently skip a host-word writeback right after wrapping to a new row).
    /// jit_hostr_ci8_block_multirow_no_stopony above covers the same row-wrap
    /// shape but without STOPONY, so it never touched this path — this test uses
    /// DM0_READ_BLOCK (STOPONXY | COLORHOST | DOSETUP) like the real driver and
    /// test_hostr_ci8_read_block_32bit do, at larger sizes with several rows and
    /// several words per row so more than one row-wrap is exercised.
    fn hostr_stress(dm1: u32, width: i32, height: i32, words_per_pixel_row: u32) {
        // Distinct per-pixel values (not a solid fill) so any word/row
        // misalignment shows up as a mismatch instead of coincidentally matching.
        let fill_rex = make_rex3();
        rex3init(fill_rex);
        unsafe {
            let fb = &mut *fill_rex.fb_rgb.get();
            for y in 0..height {
                for x in 0..width {
                    // Every pixel must stay distinct even after each plane's own
                    // depth mask (CI8 keeps only the low byte) — vary the low byte
                    // directly instead of only the high bits, or a CI8 readback
                    // would see an accidentally-solid fill and hide misalignment.
                    let i = (y * width + x) as u32;
                    fb[(y as u32 * 2048 + x as u32) as usize] =
                        0x1234_5600u32 | (1 + (i % 0xFE));
                }
            }
        }

        let setup_read = |rex: &Rex3| {
            unsafe {
                let src = &*fill_rex.fb_rgb.get();
                let dst = &mut *rex.fb_rgb.get();
                for y in 0..height {
                    for x in 0..width {
                        let idx = (y as u32 * 2048 + x as u32) as usize;
                        dst[idx] = src[idx];
                    }
                }
            }
            reg(rex, REX3_DRAWMODE1, dm1);
            reg(rex, REX3_XYENDI,   xy(width - 1, height - 1));
            reg(rex, REX3_XYSTARTI, xy(0, 0));
        };

        let words = (words_per_pixel_row as i32 * height) as usize;
        let read_all_words = |rex: &Rex3| -> Vec<u32> {
            let mut out = Vec::with_capacity(words);
            reg_go(rex, REX3_DRAWMODE0, DM0_READ_BLOCK);
            for i in 0..words {
                out.push(if i == words - 1 { read_hostrw32_last(rex) } else { read_hostrw32(rex) });
            }
            out
        };

        let rex_i = make_rex3();
        rex3init(rex_i);
        setup_read(rex_i);
        let words_interp = read_all_words(rex_i);

        // Pre-warm: throwaway pass over the same primitive shape forces the JIT
        // shader to be fully compiled *before* the real comparison pass, so
        // every GO below dispatches via JIT from word 0 — reproducing the "already
        // compiled by the time the real primitive runs" end state deterministically
        // instead of relying on racy mid-primitive compile timing.
        let rex_j = make_rex3_jit();
        rex3init(rex_j);
        setup_read(rex_j);
        let _ = read_all_words(rex_j);
        if let Some(ref jit) = rex_j.rex_jit {
            assert!(jit.wait_compiled(DM0_READ_BLOCK, dm1, 0xF << CLIPMODE_CIDMATCH_SHIFT),
                "JIT compile failed dm0={DM0_READ_BLOCK:#010x} dm1={dm1:#010x}");
        }
        rex3init(rex_j);
        setup_read(rex_j);
        let words_jit = read_all_words(rex_j);

        assert_eq!(words_interp, words_jit,
            "HOSTR stress ({width}x{height}, dm1={dm1:#010x}) JIT/interp mismatch:\n  interp={words_interp:08x?}\n  jit   ={words_jit:08x?}");
    }

    #[test]
    fn jit_hostr_ci8_stress_large_block() {
        // 32px wide / 4px-per-word = 8 words/row, 20 rows = 160 words, 20 row-wraps.
        hostr_stress(DM1_CI8_HOSTRW, 32, 20, 8);
    }

    #[test]
    fn jit_hostr_ci8_stress_odd_width_block() {
        // Odd width: rows don't divide evenly into words, so every row-wrap also
        // lands mid-word (ceil(17/4) = 5 words/row, last word only 1 valid pixel).
        hostr_stress(DM1_CI8_HOSTRW, 17, 11, 5);
    }

    /// Companion to jit_hostr_ci8_stress_odd_width_block: hostr_stress only checks
    /// interp == JIT, which would pass even if BOTH engines packed a partial word
    /// wrong the same way. This independently verifies against the fill pattern
    /// that each row's trailing partial word (1 of 4 CI8 slots valid, 17 % 4 == 1)
    /// is packed MSB-first with the unfilled low bytes zeroed — same layout
    /// test_hostr_ci8_partial_word verifies for a single row, checked here across
    /// every row-wrap in a multirow block for both interp and (pre-warmed) JIT.
    #[test]
    fn jit_hostr_ci8_partial_word_alignment_multirow() {
        let (width, height): (i32, i32) = (17, 6);
        let words_per_row = 5usize; // ceil(17/4)
        let dm1 = DM1_CI8_HOSTRW;

        // Same fill formula as hostr_stress: distinct, nonzero low byte per pixel.
        let pixel_at = |x: i32, y: i32| -> u32 {
            let i = (y * width + x) as u32;
            1 + (i % 0xFE)
        };

        let fill_rex = make_rex3();
        rex3init(fill_rex);
        unsafe {
            let fb = &mut *fill_rex.fb_rgb.get();
            for y in 0..height {
                for x in 0..width {
                    fb[(y as u32 * 2048 + x as u32) as usize] = 0x1234_5600 | pixel_at(x, y);
                }
            }
        }

        let setup_read = |rex: &Rex3| {
            unsafe {
                let src = &*fill_rex.fb_rgb.get();
                let dst = &mut *rex.fb_rgb.get();
                for y in 0..height {
                    for x in 0..width {
                        let idx = (y as u32 * 2048 + x as u32) as usize;
                        dst[idx] = src[idx];
                    }
                }
            }
            reg(rex, REX3_DRAWMODE1, dm1);
            reg(rex, REX3_XYENDI,   xy(width - 1, height - 1));
            reg(rex, REX3_XYSTARTI, xy(0, 0));
        };

        let words = words_per_row * height as usize;
        let read_all_words = |rex: &Rex3| -> Vec<u32> {
            let mut out = Vec::with_capacity(words);
            reg_go(rex, REX3_DRAWMODE0, DM0_READ_BLOCK);
            for i in 0..words {
                out.push(if i == words - 1 { read_hostrw32_last(rex) } else { read_hostrw32(rex) });
            }
            out
        };

        let check_alignment = |words: &[u32], engine: &str| {
            for y in 0..height as usize {
                let row = &words[y * words_per_row..(y + 1) * words_per_row];
                // Full words: 4 valid CI8 pixels each, MSB-first.
                for (wi, &w) in row[..words_per_row - 1].iter().enumerate() {
                    for slot in 0..4 {
                        let x = (wi * 4 + slot) as i32;
                        let got = (w >> (24 - 8 * slot)) & 0xFF;
                        assert_eq!(got, pixel_at(x, y as i32),
                            "{engine}: y={y} word={wi} slot={slot}: got {got:#04x}");
                    }
                }
                // Trailing partial word: 17 % 4 == 1 valid pixel at x=16, must sit
                // at the MSB byte with the remaining 3 bytes zero-padded (mirrors
                // flush_host_pixel's left-shift-to-MSB / emit_shader's
                // flushed_shifter padding — packing is LSB-to-MSB as slots fill,
                // so a lone pixel must end up shifted all the way to the top).
                let last = row[words_per_row - 1];
                let expected_pixel = pixel_at(16, y as i32);
                assert_eq!((last >> 24) & 0xFF, expected_pixel,
                    "{engine}: y={y} partial word MSB byte: got {:#04x} expected {expected_pixel:#04x} (word={last:#010x})",
                    (last >> 24) & 0xFF);
                assert_eq!(last & 0x00FF_FFFF, 0,
                    "{engine}: y={y} partial word low 3 bytes should be zero-padded, got {:#08x}", last & 0x00FF_FFFF);
            }
        };

        let rex_i = make_rex3();
        rex3init(rex_i);
        setup_read(rex_i);
        let words_interp = read_all_words(rex_i);
        check_alignment(&words_interp, "interp");

        let rex_j = make_rex3_jit();
        rex3init(rex_j);
        setup_read(rex_j);
        let _ = read_all_words(rex_j); // throwaway: triggers compile request
        if let Some(ref jit) = rex_j.rex_jit {
            assert!(jit.wait_compiled(DM0_READ_BLOCK, dm1, 0xF << CLIPMODE_CIDMATCH_SHIFT),
                "JIT compile failed dm0={DM0_READ_BLOCK:#010x} dm1={dm1:#010x}");
        }
        rex3init(rex_j);
        setup_read(rex_j);
        let words_jit = read_all_words(rex_j);
        check_alignment(&words_jit, "jit");
    }

    #[test]
    fn jit_hostr_rgb24_stress_large_block() {
        // RGB24 hostdepth32: 1 pixel/word, so this is 24 rows x 15 words = 360 row-wraps' worth.
        hostr_stress(DM1_RGB24_HOSTRW, 15, 24, 15);
    }

    /// BLENDALPHA (DRAWMODE1 bit 27) must be honoured identically by the JIT and
    /// the interpreter: it selects what BF_SA resolves to for the SOURCE
    /// multiplier only ('1' = real source alpha, '0' = 1.0), while DFACTOR keeps
    /// its own definition against the real alpha. Runs both polarities over a
    /// spread of alphas against a lit destination, where the two differ most.
    #[test]
    fn jit_blend_blendalpha_matches_interp() {
        for blendalpha in [false, true] {
            let dm1 = DRAWMODE1_PLANES_RGB | (3 << 3) | (1 << 15)
                | DRAWMODE1_COMPARE_DISABLE_SH | (1 << 18)
                | (DRAWMODE1_BF_SA << 19) | (DRAWMODE1_BF_MSA << 22)
                | ((blendalpha as u32) << 27) | DRAWMODE1_LOGICOP_SRC_SH;
            let dm1_src = DRAWMODE1_PLANES_RGB | (3 << 3) | (1 << 15)
                | DRAWMODE1_COMPARE_DISABLE_SH | DRAWMODE1_LOGICOP_SRC_SH;
            let dm0 = DM0_DRAW_BLOCK;
            let alphas: [u32; 5] = [0, 8, 64, 128, 255];

            let setup = |rex: &Rex3| {
                for (i, a) in alphas.iter().enumerate() {
                    let x = i as i32;
                    // Lit destination first, blending off.
                    reg(rex, REX3_DRAWMODE1, dm1_src);
                    reg(rex, REX3_WRMASK, 0xFFFFFF);
                    reg(rex, REX3_COLORALPHA, 255 << 11);
                    reg(rex, REX3_COLORRED,   0x40 << 11);
                    reg(rex, REX3_COLORGRN,   0x50 << 11);
                    reg(rex, REX3_COLORBLUE,  0x60 << 11);
                    reg(rex, REX3_XYENDI,   xy(x, 0));
                    reg(rex, REX3_XYSTARTI, xy(x, 0));
                    reg_go(rex, REX3_DRAWMODE0, dm0);
                    // Then blend over it.
                    reg(rex, REX3_DRAWMODE1, dm1);
                    reg(rex, REX3_ALPHAREF, 0);
                    reg(rex, REX3_COLORALPHA, a << 11);
                    reg(rex, REX3_COLORRED,   0xC0 << 11);
                    reg(rex, REX3_COLORGRN,   0x90 << 11);
                    reg(rex, REX3_COLORBLUE,  0x30 << 11);
                    reg(rex, REX3_XYENDI,   xy(x, 0));
                    reg(rex, REX3_XYSTARTI, xy(x, 0));
                    reg_go(rex, REX3_DRAWMODE0, dm0);
                }
            };

            let rex_i = make_rex3();
            rex3init(rex_i);
            setup(rex_i);
            wait(rex_i);
            let fb_interp: Vec<u32> =
                (0..alphas.len() as i32).map(|x| read_pixel(rex_i, x, 0) & 0xFFFFFF).collect();

            let rex_j = make_rex3_jit();
            rex3init(rex_j);
            setup(rex_j);
            wait(rex_j);
            if let Some(ref jit) = rex_j.rex_jit {
                assert!(jit.wait_compiled(dm0, dm1, 0xF << CLIPMODE_CIDMATCH_SHIFT),
                    "JIT compile failed dm0={dm0:#010x} dm1={dm1:#010x}");
            }
            clear_region(rex_j, 0, 0, alphas.len() as i32 - 1, 0);
            rex3init(rex_j);
            setup(rex_j);
            wait(rex_j);
            let fb_jit: Vec<u32> =
                (0..alphas.len() as i32).map(|x| read_pixel(rex_j, x, 0) & 0xFFFFFF).collect();

            assert_eq!(fb_interp, fb_jit,
                "BLENDALPHA={blendalpha} JIT/interp mismatch: \
                 interp={fb_interp:08x?} jit={fb_jit:08x?}");
        }
    }


    /// FASTCLEAR with CID checking ENABLED must draw as an ordinary draw.
    ///
    /// rex3.pdf says so three times — DRAWMODE1 bit 17 ("when CID checking
    /// disabled (CLIPMODE CIDMATCH = 0xF)"), §3.5.5 ("CID checking is not
    /// allowed for this drawing mode"), and the programming notes ("REX3 will
    /// disable FASTCLEAR mode if CID checking is enabled ... host must setup
    /// fast clear operation by writing COLORVRAM and also setting up DRAWMODE
    /// and COLORI"). The COLORI advice is the tell: COLORI is what gets used
    /// when the window system turns CID checking on behind GL's back.
    ///
    /// jit_fastclear_rgb24 cannot catch this — it pins cidmatch to 0xF, where
    /// the question does not arise. This was invisible for another reason too:
    /// rex3_simd::try_fastclear_block intercepted BLOCK draws before
    /// execute_go's processor selection and never checked CID, so BOTH engines
    /// wrote COLORVRAM and agreed. Removing that path exposed the real
    /// divergence. See rules/rex3/fastclear-cid-divergence.md.
    ///
    /// Colour registers are o12.11 DDA values (get_colori shifts right by 11),
    /// so they are written as `component << 11`, not as packed RGB24 — writing
    /// packed bytes here clamps to black and makes the test vacuous.
    #[test]
    fn jit_fastclear_with_cid_checking_matches_interp() {
        let dm1 = DRAWMODE1_PLANES_RGB | (3 << 3) | (1 << 15)
            | DRAWMODE1_LOGICOP_SRC_SH | (1 << 17) | DRAWMODE1_COMPARE_DISABLE_SH;
        let dm0 = DM0_DRAW_BLOCK;
        compare_jit_interp(0, 0, 7, 7,
            |rex| {
                reg(rex, REX3_DRAWMODE1, dm1);
                // Distinct from the COLORI-derived colour so the two colour
                // sources cannot be confused.
                reg(rex, REX3_COLORVRAM, 0xABCDEF);
                reg(rex, REX3_COLORRED,  0x40u32 << 11);
                reg(rex, REX3_COLORGRN,  0x50u32 << 11);
                reg(rex, REX3_COLORBLUE, 0x60u32 << 11);
                reg(rex, REX3_WRMASK,    0xFFFFFF);
                // CID checking ON, permitting CID 0. make_rex3 zeroes fb_aux, so
                // every pixel has CID 0 and passes — isolating the colour-source
                // question from write suppression.
                reg(rex, REX3_CLIPMODE,  0x1 << 9);
                reg(rex, REX3_XYENDI,    xy(7, 7));
                reg(rex, REX3_XYSTARTI,  xy(0, 0));
            },
            dm0, dm1,
        );
    }

    /// CIDMATCH on a draw whose target plane lives in fb_aux (OLAY/PUP/CID).
    ///
    /// The CID probe re-derives its fb_aux offset from `px_ptr`, which is only
    /// an fb_rgb pointer when the target plane is RGB/RGBA. For an aux-plane
    /// draw `px_ptr` is already fb_aux-based, and subtracting fb_rgb from it
    /// produced `fb_aux + (fb_aux - fb_rgb) + off` — a pointer hundreds of MB
    /// outside either framebuffer. It segfaulted the REX3 thread the moment
    /// IRIX drew into an overlay plane with CID checking live (X11 menus and
    /// the cursor do exactly that). `fb_rgb` and `fb_aux` are two independent
    /// `Box<[u32]>` allocations, so what the wild read hits is a property of
    /// the heap layout, not of the shader: in the emulator (bases 700 MB
    /// apart) SIGSEGV; in this test process (bases adjacent) a stray but
    /// mapped word, which shows up as the CID test answering differently from
    /// the interpreter. Either outcome fails the test.
    ///
    /// Both shader emitters carry the probe, so the line adrmode is covered
    /// alongside the block one. cid_write_masks_jit already covers the
    /// RGB-plane half of the same test.
    #[test]
    fn jit_cidmatch_aux_plane_matches_interp() {
        fn aux_pixel(rex: &Rex3, x: i32, y: i32) -> u32 {
            unsafe { (*rex.fb_aux.get())[(y as u32 * 2048 + x as u32) as usize] }
        }

        let rex_interp = make_rex3();
        let rex_jit    = make_rex3_jit();
        let dst = 20 * 2048 + 10;
        let mut drew_something = false;

        for planes in [DRAWMODE1_PLANES_OLAY, DRAWMODE1_PLANES_PUP, DRAWMODE1_PLANES_CID] {
            let dm1 = planes | DRAWMODE1_COMPARE_DISABLE_SH | DRAWMODE1_LOGICOP_SRC_SH;
            for dm0 in [DM0_DRAW_BLOCK,
                        DRAWMODE0_OPCODE_DRAW | DRAWMODE0_ADRMODE_I_LINE_SH | DM0_DOSETUP | DM0_STOPONXY] {
                for mask in [0b0001u32, 0b0100, 0b1010] {
                    let cm = mask << CLIPMODE_CIDMATCH_SHIFT;
                    let jit = rex_jit.rex_jit.as_ref().unwrap();
                    jit.request_compile(dm0, dm1, cm);
                    assert!(jit.wait_compiled(dm0, dm1, cm),
                        "JIT compile failed for dm0={dm0:#010x} dm1={dm1:#010x} cm={cm:#010x}");

                    for cid in 0..4_u32 {
                        let seed = 0x80000000 | (cid << 4) | cid;
                        let run = |rex: &Rex3| -> u32 {
                            rex3init(rex);
                            wait(rex);
                            unsafe { (*rex.fb_aux.get())[dst] = seed; }
                            reg(rex, REX3_DRAWMODE1, dm1);
                            reg(rex, REX3_COLORI,    0xFF);
                            reg(rex, REX3_CLIPMODE,  cm);
                            reg(rex, REX3_XYSTARTI,  xy(10, 20));
                            reg(rex, REX3_XYENDI,    xy(10, 20));
                            reg_go(rex, REX3_DRAWMODE0, dm0);
                            aux_pixel(rex, 10, 20)
                        };

                        let before  = rex_jit.jit_go_count.load(std::sync::atomic::Ordering::Relaxed);
                        let got_jit = run(rex_jit);
                        assert_eq!(rex_jit.jit_go_count.load(std::sync::atomic::Ordering::Relaxed) - before, 1,
                            "GO did not dispatch to the compiled shader: \
                             dm0={dm0:#010x} dm1={dm1:#010x} cm={cm:#010x}");

                        let got_interp = run(rex_interp);
                        assert_eq!(got_jit, got_interp,
                            "JIT/interp mismatch: planes={planes} dm0={dm0:#010x} \
                             CIDMATCH={mask:04b} cid={cid}");
                        drew_something |= got_interp != seed;
                    }
                }
            }
        }

        // A permitted CID has to actually write, or every comparison above is
        // a comparison of two untouched seed values.
        assert!(drew_something, "no aux-plane write landed — the comparison was vacuous");

        rex_jit.stop();
        rex_interp.stop();
    }
}

// ---------------------------------------------------------------------------
// GFifo ring buffer tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod gfifo_tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    /// Helper: construct a standalone GFifo (not attached to a Rex3).
    /// Uses Arc::new_zeroed to heap-allocate without touching the stack.
    fn make_gfifo() -> Arc<GFifo> {
        unsafe { Arc::new_zeroed().assume_init() }
    }

    /// SPSC: one producer thread pushes N items, one consumer thread pops them.
    /// Verifies ordering, value integrity, and that the queue drains completely.
    #[test]
    fn gfifo_spsc() {
        const N: u64 = 4096;
        let q = make_gfifo();

        let qp = q.clone();
        let producer = thread::spawn(move || {
            for i in 0..N {
                qp.push(i as u32, i.wrapping_mul(0xDEAD_BEEF_0000_0001));
            }
        });

        let qc = q.clone();
        let consumer = thread::spawn(move || {
            let mut received = 0u64;
            while received < N {
                if let Some((addr, val)) = qc.peek().map(|e| { qc.consume(); e }) {
                    assert_eq!(addr as u64, received,
                        "wrong addr at position {received}: got {addr}");
                    assert_eq!(val, received.wrapping_mul(0xDEAD_BEEF_0000_0001),
                        "wrong val at position {received}: got {val:#x}");
                    received += 1;
                } else {
                    std::hint::spin_loop();
                }
            }
            // Queue must be empty after draining all items.
            assert!(qc.is_empty(), "queue not empty after full drain");
        });

        producer.join().unwrap();
        consumer.join().unwrap();
    }

    /// MPSC: multiple producer threads push disjoint ranges; one consumer collects
    /// all values and verifies every expected value arrived exactly once.
    #[test]
    fn gfifo_mpsc() {
        const PRODUCERS: u32 = 4;
        const PER_PRODUCER: u32 = 1024;
        const TOTAL: u32 = PRODUCERS * PER_PRODUCER;

        let q = make_gfifo();

        // Each producer pushes addr = producer_id * PER_PRODUCER + i, val = addr as u64.
        let producers: Vec<_> = (0..PRODUCERS).map(|pid| {
            let qp = q.clone();
            thread::spawn(move || {
                let base = pid * PER_PRODUCER;
                for i in 0..PER_PRODUCER {
                    let v = base + i;
                    qp.push(v, v as u64);
                }
            })
        }).collect();

        let qc = q.clone();
        let consumer = thread::spawn(move || {
            let mut seen = vec![false; TOTAL as usize];
            let mut count = 0u32;
            while count < TOTAL {
                if let Some((addr, val)) = qc.peek().map(|e| { qc.consume(); e }) {
                    assert!((addr as usize) < TOTAL as usize,
                        "addr {addr} out of range");
                    assert_eq!(val, addr as u64,
                        "val mismatch for addr {addr}: got {val}");
                    assert!(!seen[addr as usize],
                        "duplicate entry for addr {addr}");
                    seen[addr as usize] = true;
                    count += 1;
                } else {
                    std::hint::spin_loop();
                }
            }
            assert!(qc.is_empty(), "queue not empty after full drain");
            // Every slot must have been seen exactly once.
            assert!(seen.iter().all(|&s| s), "some entries were never received");
        });

        for p in producers { p.join().unwrap(); }
        consumer.join().unwrap();
    }
}


// ============================================================================
// Blend path: dblsrc slot packing (regression)
//
// At 12bpp two pixels share one 24-bit VRAM word, so a compressed 12-bit result
// must be amplified (val | val<<12) into BOTH packed slots before the write —
// WRMASK then selects which slot actually lands. The logic-op path always did
// this; the blend path did not, so any WRMASK covering the high slot (0xfff000,
// or 0xffffff for both) wrote zeros there, i.e. black. Real IRIX GL hits this
// constantly: an alpha-blended billboard trace showed 77% of blended draws using
// WRMASK 0xfff000/0xffffff.
// ============================================================================

/// DRAWMODE1 for 12bpp RGB, SRC logicop, blend SA+MSA, alpha compare disabled.
/// planes=RGB, drawdepth=2 (12bpp), rgbmode(15), blend(18), sfactor=SA(4)<<19,
/// dfactor=MSA(5)<<22.
const DM1_RGB12_BLEND: u32 = DRAWMODE1_PLANES_RGB
    | (2 << 3)
    | (1 << 15)
    | DRAWMODE1_COMPARE_DISABLE_SH
    | (1 << 18)
    | (4 << 19)
    | (5 << 22)
    | DRAWMODE1_LOGICOP_SRC_SH;

/// Write an opaque blended pixel into the HIGH 12-bit slot and read it back.
/// Before the fix the high slot received 0 while the low slot kept whatever was
/// there, so the drawn pixel came out black.
#[test]
fn test_blend_12bpp_writes_high_slot() {
    let rex = make_rex3();
    rex3init(&rex);
    reg(&rex, REX3_DRAWMODE1, DM1_RGB12_BLEND);
    reg(&rex, REX3_WRMASK, 0xfff000);   // high slot only
    reg(&rex, REX3_ALPHAREF, 0);
    // Fully opaque white source: alpha=255 so SA*src + MSA*dst == src.
    reg(&rex, REX3_COLORALPHA, 255 << 11);
    reg(&rex, REX3_COLORRED,   255 << 11);
    reg(&rex, REX3_COLORGRN,   255 << 11);
    reg(&rex, REX3_COLORBLUE,  255 << 11);
    reg(&rex, REX3_XYENDI,   xy(7, 9));
    reg(&rex, REX3_XYSTARTI, xy(7, 9));
    reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);

    let px = read_pixel(&rex, 7, 9);
    assert_ne!(
        (px >> 12) & 0xfff, 0,
        "blend must amplify into the high 12bpp slot; got {px:#08x} (high slot black)"
    );
}

/// Same, with WRMASK covering both packed slots: both must receive the colour.
#[test]
fn test_blend_12bpp_writes_both_slots() {
    let rex = make_rex3();
    rex3init(&rex);
    reg(&rex, REX3_DRAWMODE1, DM1_RGB12_BLEND);
    reg(&rex, REX3_WRMASK, 0xffffff);   // both slots
    reg(&rex, REX3_ALPHAREF, 0);
    reg(&rex, REX3_COLORALPHA, 255 << 11);
    reg(&rex, REX3_COLORRED,   255 << 11);
    reg(&rex, REX3_COLORGRN,   255 << 11);
    reg(&rex, REX3_COLORBLUE,  255 << 11);
    reg(&rex, REX3_XYENDI,   xy(8, 9));
    reg(&rex, REX3_XYSTARTI, xy(8, 9));
    reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);

    let px = read_pixel(&rex, 8, 9);
    let lo = px & 0xfff;
    let hi = (px >> 12) & 0xfff;
    assert_eq!(lo, hi, "both packed slots must get the same blended pixel: {px:#08x}");
    assert_ne!(lo, 0, "blended pixel should not be black: {px:#08x}");
}

/// The blend result must match the logic-op SRC result for a fully opaque
/// source — same colour, same slot packing, whichever path produced it.
#[test]
fn test_blend_opaque_matches_logicop_src() {
    let rex = make_rex3();
    rex3init(&rex);
    let dm1_src = DRAWMODE1_PLANES_RGB | (2 << 3) | (1 << 15)
        | DRAWMODE1_COMPARE_DISABLE_SH | DRAWMODE1_LOGICOP_SRC_SH;

    for (x, dm1) in [(20, dm1_src), (21, DM1_RGB12_BLEND)] {
        reg(&rex, REX3_DRAWMODE1, dm1);
        reg(&rex, REX3_WRMASK, 0xffffff);
        reg(&rex, REX3_ALPHAREF, 0);
        reg(&rex, REX3_COLORALPHA, 255 << 11);
        reg(&rex, REX3_COLORRED,   0x80 << 11);
        reg(&rex, REX3_COLORGRN,   0x40 << 11);
        reg(&rex, REX3_COLORBLUE,  0xC0 << 11);
        reg(&rex, REX3_XYENDI,   xy(x, 11));
        reg(&rex, REX3_XYSTARTI, xy(x, 11));
        reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);
    }

    let src_px = read_pixel(&rex, 20, 11);
    let blend_px = read_pixel(&rex, 21, 11);
    assert_eq!(
        src_px, blend_px,
        "opaque blend {blend_px:#08x} must equal logicop SRC {src_px:#08x}"
    );
}

/// Alpha==0 with compare != (ALPHAREF=0) must discard the pixel entirely,
/// leaving the framebuffer untouched.
#[test]
fn test_blend_alpha_test_discards_zero_alpha() {
    let rex = make_rex3();
    rex3init(&rex);
    // compare = 0b101 (!=) instead of the disable pattern.
    let dm1 = (DM1_RGB12_BLEND & !DRAWMODE1_COMPARE_DISABLE_SH) | (0b101 << 12);
    reg(&rex, REX3_DRAWMODE1, dm1);
    reg(&rex, REX3_WRMASK, 0xffffff);
    reg(&rex, REX3_ALPHAREF, 0);
    reg(&rex, REX3_COLORALPHA, 0);        // alpha 0 → 0 != 0 is false → discard
    reg(&rex, REX3_COLORRED,   255 << 11);
    reg(&rex, REX3_COLORGRN,   255 << 11);
    reg(&rex, REX3_COLORBLUE,  255 << 11);
    reg(&rex, REX3_XYENDI,   xy(12, 13));
    reg(&rex, REX3_XYSTARTI, xy(12, 13));
    reg_go(&rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);

    assert_eq!(
        read_pixel(&rex, 12, 13), 0,
        "alpha==0 with compare '!=' and ALPHAREF=0 must discard the pixel"
    );
}

// ============================================================================
// BLENDALPHA (DRAWMODE1 bit 27) selects the VALUE of BF_SA
//
// Spec Table 11: "Selects SFACTOR BF_SA source alpha: '1' = source alpha,
// '0' = 1.0", and §3.8 adds the load-bearing qualifier: when BLENDALPHA=0 "the
// source multiplier ... is one instead of source alpha AND DESTINATION
// MULTIPLIER IS DEFINED BY DFACTOR". So the substitution applies to SFACTOR
// only — DFACTOR keeps its own definition and still evaluates against the real
// source alpha. These tests pin that asymmetry.
// ============================================================================

/// Build a 24bpp RGB blend DRAWMODE1 with the given factors and BLENDALPHA.
fn dm1_blend24(sfactor: u32, dfactor: u32, blendalpha: bool) -> u32 {
    DRAWMODE1_PLANES_RGB | (3 << 3) | (1 << 15) | DRAWMODE1_COMPARE_DISABLE_SH
        | (1 << 18) | (sfactor << 19) | (dfactor << 22)
        | ((blendalpha as u32) << 27) | DRAWMODE1_LOGICOP_SRC_SH
}

/// Paint a single pixel with the given mode/colour and return what landed.
fn blend_one(rex: &Rex3, x: i32, y: i32, dm1: u32, alpha: u32, rgb: u32) -> u32 {
    reg(rex, REX3_DRAWMODE1, dm1);
    reg(rex, REX3_WRMASK, 0xFFFFFF);
    reg(rex, REX3_ALPHAREF, 0);
    reg(rex, REX3_COLORALPHA, alpha << 11);
    reg(rex, REX3_COLORRED,   (rgb & 0xFF) << 11);
    reg(rex, REX3_COLORGRN,   ((rgb >> 8) & 0xFF) << 11);
    reg(rex, REX3_COLORBLUE,  ((rgb >> 16) & 0xFF) << 11);
    reg(rex, REX3_XYENDI,   xy(x, y));
    reg(rex, REX3_XYSTARTI, xy(x, y));
    reg_go(rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);
    read_pixel(rex, x, y) & 0xFFFFFF
}

/// Same pairing with BLENDALPHA=1 uses the real source alpha, so a low-alpha
/// source over a black destination is heavily attenuated.
#[test]
fn test_blendalpha1_sa_msa_attenuates() {
    let rex = make_rex3();
    rex3init(&rex);
    let px = blend_one(&rex, 31, 40,
        dm1_blend24(DRAWMODE1_BF_SA, DRAWMODE1_BF_MSA, true), 8, 0x808080);
    assert!(px < 0x0A0A0A,
        "BLENDALPHA=1 + alpha 8 should attenuate src heavily, got {px:#08x}");
}

/// BF_SA/BF_ONE with BLENDALPHA=0 is ADDITIVE (`1*src + 1*dst`), not a no-op.
/// Treating BLENDALPHA=0 as "skip the blend" would wrongly discard dst here.
#[test]
fn test_blendalpha0_sa_one_is_additive() {
    let rex = make_rex3();
    rex3init(&rex);
    // Lay down a destination first, with blending off.
    let dm1_src = DRAWMODE1_PLANES_RGB | (3 << 3) | (1 << 15)
        | DRAWMODE1_COMPARE_DISABLE_SH | DRAWMODE1_LOGICOP_SRC_SH;
    blend_one(&rex, 32, 40, dm1_src, 255, 0x202020);
    // Now blend additively over it.
    let px = blend_one(&rex, 32, 40,
        dm1_blend24(DRAWMODE1_BF_SA, DRAWMODE1_BF_ONE, false), 8, 0x101010);
    assert_eq!(px, 0x303030,
        "BLENDALPHA=0 + BF_SA/BF_ONE must add src and dst, got {px:#08x}");
}

/// BLENDALPHA=0 substitutes only the SOURCE multiplier: SFACTOR BF_SA becomes
/// 1.0, but DFACTOR BF_MSA still evaluates 1 - real source alpha. So a low-alpha
/// source over a lit destination keeps most of the destination, rather than
/// replacing it (which is what substituting in both factors would do).
#[test]
fn test_blendalpha0_substitutes_sfactor_only() {
    let rex = make_rex3();
    rex3init(&rex);
    let dm1_src = DRAWMODE1_PLANES_RGB | (3 << 3) | (1 << 15)
        | DRAWMODE1_COMPARE_DISABLE_SH | DRAWMODE1_LOGICOP_SRC_SH;
    // Destination 0x404040, source 0x101010 at alpha 8.
    blend_one(&rex, 50, 50, dm1_src, 255, 0x404040);
    let px = blend_one(&rex, 50, 50,
        dm1_blend24(DRAWMODE1_BF_SA, DRAWMODE1_BF_MSA, false), 8, 0x101010);
    // out = 1.0*src + (1 - 8/255)*dst = 0x10 + ~0x3E = ~0x4E per channel.
    let ch = px & 0xFF;
    assert!((0x48..=0x52).contains(&ch),
        "expected src + (1-alpha)*dst ≈ 0x4E per channel, got {px:#08x}");
    assert_ne!(px, 0x101010,
        "destination must still contribute — BLENDALPHA=0 must not zero DFACTOR");
}

/// With BLENDALPHA=1 both factors use the real alpha, giving a classic blend.
#[test]
fn test_blendalpha1_uses_alpha_in_both_factors() {
    let rex = make_rex3();
    rex3init(&rex);
    let dm1_src = DRAWMODE1_PLANES_RGB | (3 << 3) | (1 << 15)
        | DRAWMODE1_COMPARE_DISABLE_SH | DRAWMODE1_LOGICOP_SRC_SH;
    blend_one(&rex, 51, 50, dm1_src, 255, 0x404040);
    let px = blend_one(&rex, 51, 50,
        dm1_blend24(DRAWMODE1_BF_SA, DRAWMODE1_BF_MSA, true), 8, 0x101010);
    // out = (8/255)*0x10 + (1 - 8/255)*0x40 ≈ 0x3E — dst dominates.
    let ch = px & 0xFF;
    assert!((0x38..=0x42).contains(&ch),
        "expected classic alpha blend ≈ 0x3E per channel, got {px:#08x}");
}

/// AFUNCTION compares the REAL source alpha (from DDA or host per ALPHAHOST) —
/// spec §3.3 — and is unaffected by BLENDALPHA, which only substitutes the blend's
/// source multiplier. With BLENDALPHA=0 the blender sees BF_SA=1.0, but the alpha
/// test must still see the true alpha: alpha 0 vs ALPHAREF 0 under COMPARE='!='
/// must inhibit the write regardless of BLENDALPHA.
#[test]
fn test_afunction_uses_real_alpha_not_blendalpha() {
    let rex = make_rex3();
    rex3init(&rex);
    let dm1_src = DRAWMODE1_PLANES_RGB | (3 << 3) | (1 << 15)
        | DRAWMODE1_COMPARE_DISABLE_SH | DRAWMODE1_LOGICOP_SRC_SH;

    for (i, blendalpha) in [false, true].iter().enumerate() {
        let x = 60 + i as i32;
        // Lay down a known destination.
        blend_one(&rex, x, 55, dm1_src, 255, 0x123456);
        // COMPARE = 0b101 ("!="), ALPHAREF = 0, source alpha = 0 -> must be killed.
        let dm1 = (dm1_blend24(DRAWMODE1_BF_SA, DRAWMODE1_BF_MSA, *blendalpha)
            & !DRAWMODE1_COMPARE_DISABLE_SH) | (0b101 << 12);
        let px = blend_one(&rex, x, 55, dm1, 0, 0xFFFFFF);
        assert_eq!(
            px, 0x123456,
            "alpha==0 must be discarded by AFUNCTION with BLENDALPHA={blendalpha}, \
             destination should be untouched; got {px:#08x}"
        );
    }
}

// ---------------------------------------------------------------------------
// Plane access: data form must match the function-pointer form
// ---------------------------------------------------------------------------

/// `plane_read_shift_mask` is the single source for how a plane read slices a
/// framebuffer word. Pin every `(planes, drawdepth, dblsrc)` combination against
/// the shift and mask it is required to produce, so a wrong bit offset is a test
/// failure rather than silently wrong pixels in every specialised read.
///
/// The expected values are written out literally on purpose: checking the table
/// against a second implementation of the same table only proves the two copies
/// agree, not that either is right.
#[test]
fn plane_read_shift_mask_is_correct() {
    // (planes, drawdepth, dblsrc) -> (shift, mask)
    let cases: &[(u32, u32, bool, u32, u32)] = &[
        // RGB/RGBA: dblsrc selects the high half of the packed pair.
        (DRAWMODE1_PLANES_RGB, 0, false, 0, 0xF),
        (DRAWMODE1_PLANES_RGB, 0, true, 4, 0xF),
        (DRAWMODE1_PLANES_RGB, 1, false, 0, 0xFF),
        (DRAWMODE1_PLANES_RGB, 1, true, 8, 0xFF),
        (DRAWMODE1_PLANES_RGB, 2, false, 0, 0xFFF),
        (DRAWMODE1_PLANES_RGB, 2, true, 12, 0xFFF),
        // 24bpp fills the word, so there is no second slot to select.
        (DRAWMODE1_PLANES_RGB, 3, false, 0, 0xFFFFFF),
        (DRAWMODE1_PLANES_RGB, 3, true, 0, 0xFFFFFF),
        (DRAWMODE1_PLANES_RGBA, 1, false, 0, 0xFF),
        // Aux planes live at fixed offsets in the aux word, depth-independent.
        (DRAWMODE1_PLANES_OLAY, 0, false, 8, 0xFF),
        (DRAWMODE1_PLANES_OLAY, 0, true, 16, 0xFF),
        (DRAWMODE1_PLANES_PUP, 0, false, 2, 0x3),
        (DRAWMODE1_PLANES_PUP, 0, true, 6, 0x3),
        (DRAWMODE1_PLANES_CID, 0, false, 0, 0x3),
        (DRAWMODE1_PLANES_CID, 0, true, 4, 0x3),
    ];

    for &(planes, depth, dblsrc, want_shift, want_mask) in cases {
        let got = Rex3::plane_read_shift_mask(planes, depth, dblsrc);
        assert_eq!(
            got,
            Some((want_shift, want_mask)),
            "planes={planes} depth={depth} dblsrc={dblsrc}: \
             expected (>>{want_shift} & {want_mask:#x}), got {got:?}"
        );
    }
}

/// The aux/rgb split must agree with which framebuffer a plane actually lives
/// in — getting this backwards is the "JIT scr2scr aux plane" bug class.
#[test]
fn plane_is_aux_matches_framebuffer() {
    for planes in [DRAWMODE1_PLANES_OLAY, DRAWMODE1_PLANES_PUP, DRAWMODE1_PLANES_CID] {
        assert!(Rex3::plane_is_aux(planes), "planes={planes} should be aux");
    }
    for planes in [DRAWMODE1_PLANES_RGB, DRAWMODE1_PLANES_RGBA] {
        assert!(!Rex3::plane_is_aux(planes), "planes={planes} should be rgb");
    }
}

/// `write_masked` is now the single implementation behind all seven writers;
/// confirm the mask semantics it centralises (untouched bits preserved).
#[test]
fn write_masked_preserves_unmasked_bits() {
    let mut fb = vec![0xAAAA_AAAAu32; 4];
    Rex3::write_masked(&mut fb, 2, 0x5555_5555, 0x0000_FFFF);
    assert_eq!(fb[2], 0xAAAA_5555, "masked bits replaced, others preserved");
    assert_eq!(fb[1], 0xAAAA_AAAA, "neighbours untouched");
    Rex3::write_masked(&mut fb, 2, 0xFFFF_FFFF, 0);
    assert_eq!(fb[2], 0xAAAA_5555, "zero mask writes nothing");
}

/// The draw-shape corpus must be recorded in every build, not only when
/// Cranelift is compiled in.
///
/// This was broken until the corpus moved from `RexJit` to `Rex3`: without
/// `rex-jit` there is no `RexJit`, so nothing recorded and nothing saved — the
/// generated shader table could never learn about new shapes from ordinary
/// runs, which is precisely the build most users have.
#[test]
fn corpus_records_shapes_without_jit() {
    let rex = make_rex3();
    rex3init(rex);

    reg(rex, REX3_DRAWMODE1, DM1_RGB24_SRC);
    reg(rex, REX3_WRMASK, 0xFFFFFF);
    reg(rex, REX3_COLORRED, 0x40u32 << 11);
    reg(rex, REX3_XYENDI, xy(4, 4));
    reg(rex, REX3_XYSTARTI, xy(0, 0));
    reg_go(rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);
    wait(rex);

    let seen = rex.seen_shapes.lock();
    assert!(
        !seen.is_empty(),
        "no draw shapes recorded — the corpus cannot grow in this build"
    );

    // The recorded key must be the canonical one the table is keyed on, or a
    // regenerated shader would be filed under something the dispatch never asks
    // for.
    let dm1 = crate::rex3_shape::normalize_dm1(DM1_RGB24_SRC, DRAWMODE0_OPCODE_DRAW);
    // cm carries the clipmode key, which rex3init leaves at CIDMATCH=0xF
    // (checking disabled) — read it back rather than assuming zero.
    let cm = unsafe { (*rex.context.get()).clipmode } & CLIPMODE_JIT_KEY_MASK;
    let expect = (DM0_DRAW_BLOCK, dm1, cm);
    assert!(
        seen.contains(&expect),
        "expected {expect:?} in the corpus, got {:?}",
        seen.iter().collect::<Vec<_>>()
    );
}

/// What does a single GFIFO register write cost?
///
/// The draw backend is now fast enough that the sweep's small-span rows are
/// dominated by queue traffic rather than rasterisation: flat fill runs at
/// 645 Mpx/s across 1280-pixel spans (0.005 entries/px) and 49 Mpx/s across
/// 8-pixel spans (0.75 entries/px) — same pixels, 13x slower, purely entry
/// density. This isolates the push itself so that cost has a number.
///
/// Measures the real bus entry point (`write32`), which is what the CPU store
/// path actually calls, including the register match and the BUS_BUSY retry.
/// The consumer runs concurrently, as it does in practice.
#[test]
#[ignore = "benchmark: runs for several seconds of wall clock"]
fn gfifo_push_cost() {
    use std::time::{Duration, Instant};

    let rex = make_rex3();
    rex3init(rex);

    const BUDGET: Duration = Duration::from_millis(500);

    // A register that is pure queue traffic: no CPU-thread side effect, so the
    // write32 default arm runs and the entry lands in the fifo.
    let push_one = |r: &Rex3| {
        // Retry on BUS_BUSY exactly as the CPU does.
        while r.write32(REX3_COLORRED, 0x40 << 11) == crate::traits::BUS_BUSY {
            std::hint::spin_loop();
        }
    };

    // Warm up: let the consumer thread reach steady state.
    for _ in 0..10_000 { push_one(rex); }
    wait(rex);

    let start = Instant::now();
    let mut pushes: u64 = 0;
    while start.elapsed() < BUDGET {
        for _ in 0..1_000 { push_one(rex); }
        pushes += 1_000;
    }
    let elapsed = start.elapsed();
    wait(rex);

    let ns_per = elapsed.as_nanos() as f64 / pushes as f64;
    println!("\n=== GFIFO push cost ===");
    println!("  {pushes} pushes in {:?}", elapsed);
    println!("  {:.1} ns/entry  ({:.1} M entries/s)", ns_per, 1000.0 / ns_per);
    println!();
    println!("  For scale: a GL triangle of ~32px costs tens of entries, so at");
    println!("  this rate the queue alone caps small-primitive throughput.");
    assert!(pushes > 0);
}

/// A 64-bit store is two register writes. Does pushing them as one atomic pair
/// beat two separate pushes?
///
/// IRIX/GL issues 64-bit stores constantly (coordinate pairs, colour pairs,
/// Bresenham terms), so this is the common case, not a corner.
#[test]
#[ignore = "benchmark: runs for several seconds of wall clock"]
fn gfifo_push64_cost() {
    use std::time::{Duration, Instant};

    let rex = make_rex3();
    rex3init(rex);
    const BUDGET: Duration = Duration::from_millis(500);

    // COLORRED/COLORGRN are adjacent and neither has a CPU-thread side effect,
    // so a 64-bit store to the pair takes the queue path.
    let push64 = |r: &Rex3| {
        while r.write64(REX3_COLORRED, 0x0000_0040_0000_0050) == crate::traits::BUS_BUSY {
            std::hint::spin_loop();
        }
    };
    let push32x2 = |r: &Rex3| {
        while r.write32(REX3_COLORRED, 0x40) == crate::traits::BUS_BUSY {
            std::hint::spin_loop();
        }
        while r.write32(REX3_COLORRED + 4, 0x50) == crate::traits::BUS_BUSY {
            std::hint::spin_loop();
        }
    };

    for _ in 0..10_000 { push64(rex); }
    wait(rex);

    let run = |f: &dyn Fn(&Rex3)| -> f64 {
        let start = Instant::now();
        let mut n: u64 = 0;
        while start.elapsed() < BUDGET {
            for _ in 0..1_000 { f(rex); }
            n += 1_000;
        }
        let e = start.elapsed();
        wait(rex);
        // Nanoseconds per *entry pair*, so the two are directly comparable.
        e.as_nanos() as f64 / n as f64
    };

    let ns_pair = run(&push64);
    let ns_two  = run(&push32x2);

    println!("\n=== GFIFO 64-bit store: paired vs two separate pushes ===");
    println!("  write64 (one try_push2): {:>6.1} ns/pair", ns_pair);
    println!("  two write32 calls:       {:>6.1} ns/pair", ns_two);
    println!("  speedup:                 {:>6.2}x", ns_two / ns_pair.max(0.001));
    println!();
    println!("  Three atomics per pair instead of six, and one trip through the");
    println!("  write32 register match instead of two.");
    assert!(ns_pair > 0.0);
}

/// What does the consumer-side dispatch lookup cost?
///
/// With the backend at ~169 Mpx/s for Gouraud, a small-span GO spends far more
/// time in queue traffic and dispatch than in rasterisation. This isolates the
/// per-GO lookup: SipHash over a 12-byte key plus an RwLock read acquire, both
/// on the hot path, against the memo that is supposed to hide them.
#[test]
#[ignore = "benchmark"]
fn dispatch_lookup_cost() {
    use std::time::{Duration, Instant};
    use std::collections::HashMap;

    const N: usize = 4096;
    let keys: Vec<(u32, u32, u32)> =
        (0..N).map(|i| (0x306 + i as u32, 0x3000_f019, (i as u32 % 16) << 9)).collect();

    let map: HashMap<(u32, u32, u32), u32> =
        keys.iter().enumerate().map(|(i, k)| (*k, i as u32)).collect();
    let locked = parking_lot::RwLock::new(map.clone());

    const BUDGET: Duration = Duration::from_millis(300);
    let bench = |name: &str, mut f: Box<dyn FnMut(usize) -> u32>| {
        let start = Instant::now();
        let mut n = 0usize;
        let mut acc = 0u32;
        while start.elapsed() < BUDGET {
            for _ in 0..1000 { acc = acc.wrapping_add(f(n % N)); n += 1; }
        }
        let ns = start.elapsed().as_nanos() as f64 / n as f64;
        println!("  {name:<38} {ns:>6.1} ns   (acc {acc})");
        ns
    };

    println!("\n=== dispatch lookup cost ===");
    let m = map.clone();
    let ks = keys.clone();
    let hash_only = bench("std HashMap (SipHash), no lock", Box::new(move |i| *m.get(&ks[i]).unwrap()));
    let ks2 = keys.clone();
    let locked_ref = &locked;
    let with_lock = bench("RwLock<HashMap> read + lookup", Box::new(move |i| *locked_ref.read().get(&ks2[i]).unwrap()));
    let fxmap: crate::rex3_shape::ShapeMap<u32> =
        keys.iter().enumerate().map(|(i, k)| (*k, i as u32)).collect();
    let ks_fx = keys.clone();
    let fx = bench("ShapeMap (FxHash), no lock", Box::new(move |i| *fxmap.get(&ks_fx[i]).unwrap()));
    let fxlocked = parking_lot::RwLock::new({
        let m: crate::rex3_shape::ShapeMap<u32> =
            keys.iter().enumerate().map(|(i, k)| (*k, i as u32)).collect();
        m
    });
    let ks_fxl = keys.clone();
    let fxl_ref = &fxlocked;
    let fxl = bench("RwLock<ShapeMap> read + lookup", Box::new(move |i| *fxl_ref.read().get(&ks_fxl[i]).unwrap()));
    let _ = (fx, fxl);
    let ks3 = keys.clone();
    let sorted: Vec<((u32, u32, u32), u32)> = {
        let mut v: Vec<_> = keys.iter().enumerate().map(|(i, k)| (*k, i as u32)).collect();
        v.sort_unstable_by_key(|(k, _)| *k);
        v
    };
    let bsearch = bench("sorted slice binary_search", Box::new(move |i| {
        let k = ks3[i];
        sorted.binary_search_by_key(&k, |(kk, _)| *kk).map(|j| sorted[j].1).unwrap()
    }));
    println!();
    println!("  lock overhead: {:.1} ns", with_lock - hash_only);
    println!("  binary search vs hashmap+lock: {:.2}x", with_lock / bsearch.max(0.001));
    assert!(hash_only > 0.0);
}

/// Where do the 73 ns of a GFIFO push actually go?
///
/// Three atomics on an uncontended cache line should be ~15-20 ns. Measuring
/// 73 means something else dominates — most likely producer/consumer contention
/// on the ring's head/tail lines, which no amount of instruction-shaving fixes.
/// This separates the cases.
#[test]
#[ignore = "benchmark"]
fn gfifo_push_breakdown() {
    use std::time::{Duration, Instant};
    const BUDGET: Duration = Duration::from_millis(400);

    let bench = |name: &str, f: &dyn Fn()| {
        let start = Instant::now();
        let mut n: u64 = 0;
        while start.elapsed() < BUDGET {
            for _ in 0..1000 { f(); }
            n += 1000;
        }
        let ns = start.elapsed().as_nanos() as f64 / n as f64;
        println!("  {name:<44} {ns:>6.1} ns");
        ns
    };

    println!("\n=== GFIFO push breakdown ===");

    // 1. The ring alone, no consumer running at all.
    let quiet = crate::rex3::GFifo::new();
    let a = bench("try_push, NO consumer thread", &|| {
        // Drain by hand so the ring never fills.
        if !quiet.try_push(0x100, 0) {
            while quiet.peek().is_some() { quiet.consume(); }
        }
    });

    // 2. Same ring, but a consumer thread spinning on it — the real topology.
    let live: &'static crate::rex3::GFifo = Box::leak(Box::new(crate::rex3::GFifo::new()));
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop2 = stop.clone();
    let h = std::thread::spawn(move || {
        while !stop2.load(Ordering::Relaxed) {
            while live.peek().is_some() { live.consume(); }
            std::hint::spin_loop();
        }
    });
    let b = bench("try_push, consumer thread draining", &|| {
        while !live.try_push(0x100, 0) { std::hint::spin_loop(); }
    });
    stop.store(true, Ordering::Relaxed);
    let _ = h.join();

    // 3. Through the real bus entry point, for reference.
    let rex = make_rex3();
    rex3init(rex);
    let c = bench("write32 (full bus path + consumer)", &|| {
        while rex.write32(REX3_COLORRED, 0x40 << 11) == crate::traits::BUS_BUSY {
            std::hint::spin_loop();
        }
    });

    println!();
    println!("  ring alone:                {:>6.1} ns", a);
    println!("  + concurrent consumer:     {:>6.1} ns  (+{:.1} contention)", b, b - a);
    println!("  + bus/register dispatch:   {:>6.1} ns  (+{:.1} overhead)", c, c - b);
    assert!(a > 0.0);
}

// ── Batched HOSTRW (VDMA bulk path) ─────────────────────────────────────────

/// A batched write must paint exactly what the same words painted one at a
/// time. This is the core equivalence claim of the HOSTRW batching design:
/// `host_len = N` and N separate PIO writes are the same transfer.
#[test]
fn test_hostw_batch_matches_scalar_writes() {
    use crate::traits::BusDevice;

    let rows: [[u8; 8]; 3] = [
        [0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80],
        [0x91, 0xA2, 0xB3, 0xC4, 0xD5, 0xE6, 0xF7, 0x08],
        [0x19, 0x2A, 0x3B, 0x4C, 0x5D, 0x6E, 0x7F, 0x00],
    ];
    let words: Vec<u64> = rows.iter().map(|row| {
        (row[0] as u64) << 56 | (row[1] as u64) << 48
      | (row[2] as u64) << 40 | (row[3] as u64) << 32
      | (row[4] as u64) << 24 | (row[5] as u64) << 16
      | (row[6] as u64) <<  8 | (row[7] as u64)
    }).collect();

    let setup = |rex: &Rex3| {
        rex3init(rex);
        reg(rex, REX3_DRAWMODE1, DM1_CI8_HOSTRW64);
        reg(rex, REX3_WRMASK, 0xFF);
        reg(rex, REX3_XYENDI,   xy(7, 2));
        reg(rex, REX3_XYSTARTI, xy(0, 0));
        reg(rex, REX3_DRAWMODE0, DM0_HOSTW_BLOCK);
    };

    // Reference: one PIO write per word.
    let rex_scalar = make_rex3();
    setup(rex_scalar);
    for &w in &words { write_hostrw64(rex_scalar, w); }
    wait(rex_scalar);

    // Batched: one bulk call carrying the whole run.
    let rex_batch = make_rex3();
    setup(rex_batch);
    let st = rex_batch.dma_write64_bulk(go_addr(REX3_HOSTRW0), &words);
    assert_eq!(st, crate::traits::BUS_OK, "bulk write should be accepted");
    wait(rex_batch);

    for y in 0..3i32 {
        for x in 0..8i32 {
            let s = read_pixel(rex_scalar, x, y) & 0xFF;
            let b = read_pixel(rex_batch,  x, y) & 0xFF;
            assert_eq!(b, s, "batch vs scalar mismatch at ({x},{y}): {b:#04x} != {s:#04x}");
        }
        }
    // And the pixels are actually the input, not two matching blanks.
    for (y, row) in rows.iter().enumerate() {
        for (x, &want) in row.iter().enumerate() {
            assert_eq!(read_pixel(rex_batch, x as i32, y as i32) & 0xFF, want as u32,
                "batch painted wrong pixel at ({x},{y})");
        }
    }
}

/// A one-word batch must behave exactly like a plain HOSTRW64 write — the
/// property that lets the shader have no batched-vs-single branch.
#[test]
fn test_hostw_batch_of_one_equals_single_write() {
    use crate::traits::BusDevice;

    let word: u64 = 0x1122_3344_5566_7788;
    let setup = |rex: &Rex3| {
        rex3init(rex);
        reg(rex, REX3_DRAWMODE1, DM1_CI8_HOSTRW64);
        reg(rex, REX3_WRMASK, 0xFF);
        reg(rex, REX3_XYENDI,   xy(7, 0));
        reg(rex, REX3_XYSTARTI, xy(0, 0));
        reg(rex, REX3_DRAWMODE0, DM0_HOSTW_BLOCK);
    };

    let rex_scalar = make_rex3();
    setup(rex_scalar);
    write_hostrw64(rex_scalar, word);
    wait(rex_scalar);

    let rex_batch = make_rex3();
    setup(rex_batch);
    assert_eq!(rex_batch.dma_write64_bulk(go_addr(REX3_HOSTRW0), &[word]),
               crate::traits::BUS_OK);
    wait(rex_batch);

    for x in 0..8i32 {
        assert_eq!(read_pixel(rex_batch, x, 0) & 0xFF,
                   read_pixel(rex_scalar, x, 0) & 0xFF,
                   "len-1 batch differs from a single write at x={x}");
    }
}

/// The batch window must not leak into the PIO path that follows it:
/// after a batch retires, a plain HOSTRW write draws from ctx.hostrw again.
#[test]
fn test_pio_write_after_batch_is_unaffected() {
    use crate::traits::BusDevice;

    let rex = make_rex3();
    rex3init(rex);
    reg(rex, REX3_DRAWMODE1, DM1_CI8_HOSTRW64);
    reg(rex, REX3_WRMASK, 0xFF);
    reg(rex, REX3_XYENDI,   xy(7, 0));
    reg(rex, REX3_XYSTARTI, xy(0, 0));
    reg(rex, REX3_DRAWMODE0, DM0_HOSTW_BLOCK);
    let _ = rex.dma_write64_bulk(go_addr(REX3_HOSTRW0), &[0x1111_1111_1111_1111u64; 2]);
    wait(rex);

    // Now a normal PIO write to a different row.
    reg(rex, REX3_XYENDI,   xy(7, 5));
    reg(rex, REX3_XYSTARTI, xy(0, 5));
    reg(rex, REX3_DRAWMODE0, DM0_HOSTW_BLOCK);
    write_hostrw64(rex, 0xAABB_CCDD_EEFF_0102);
    wait(rex);

    let expect = [0xAAu32, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x01, 0x02];
    for (x, &want) in expect.iter().enumerate() {
        assert_eq!(read_pixel(rex, x as i32, 5) & 0xFF, want,
            "PIO write after a batch painted the wrong pixel at x={x}");
    }
}

/// The batched read must return exactly what N scalar `dma_read64` calls
/// return. This is the read-side half of the batching equivalence claim, and
/// the direction where the win is biggest (one pipeline drain per batch instead
/// of one per qword).
#[test]
fn test_hostr_batch_matches_scalar_dma_reads() {
    use crate::traits::BusDevice;

    const NWORDS: usize = 8;
    // CI8 packed into 64-bit words: 8 pixels per word, so NWORDS*8 pixels.
    let (x0, y0, x1, y1) = (0i32, 0i32, 15i32, 3i32);
    let color = 0x5Au8;

    let setup_read = |rex: &Rex3| {
        rex3init(rex);
        // Fill the region first.
        reg(rex, REX3_DRAWMODE1, DM1_CI8_HOSTRW64);
        reg(rex, REX3_WRMASK, 0xFF);
        reg(rex, REX3_COLORI, color as u32);
        reg(rex, REX3_XYENDI,   xy(x1, y1));
        reg(rex, REX3_XYSTARTI, xy(x0, y0));
        reg_go(rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);
        wait(rex);
        // Arm the READ block.
        reg(rex, REX3_DRAWMODE1, DM1_CI8_HOSTRW64);
        reg(rex, REX3_XYENDI,   xy(x1, y1));
        reg(rex, REX3_XYSTARTI, xy(x0, y0));
        reg_go(rex, REX3_DRAWMODE0, DM0_READ_BLOCK);
    };

    let rex_scalar = make_rex3();
    setup_read(rex_scalar);
    let mut scalar = [0u64; NWORDS];
    for slot in scalar.iter_mut() {
        let r = rex_scalar.dma_read64(go_addr(REX3_HOSTRW0));
        assert!(r.is_ok(), "scalar dma_read64 failed");
        *slot = r.data;
    }

    let rex_batch = make_rex3();
    setup_read(rex_batch);
    let mut batched = [0u64; NWORDS];
    let st = rex_batch.dma_read64_bulk(go_addr(REX3_HOSTRW0), &mut batched);
    assert_eq!(st, crate::traits::BUS_OK, "bulk read should be accepted");

    assert_eq!(batched, scalar,
        "batched read differs from the scalar dma_read64 sequence");

    // And it is real pixel data, not two matching zeroes.
    let expect = u64::from_be_bytes([color; 8]);
    assert_eq!(batched[0], expect,
        "batched read returned {:#018x}, expected the filled colour {:#018x}",
        batched[0], expect);
}

/// Ground-truth check for the DMA read path: a READ block armed with a GO
/// already produces word 0, so the first `dma_read64`/`dma_read64_bulk` must
/// return *that* word, not the one after it.
///
/// Comparing batched against scalar cannot catch a shared off-by-one-word
/// phase error — both skip the same word and agree. So this asserts against


/// DMA HOSTR readback, driven the way the real driver does it: set up the READ
/// block **without** an arming GO, then let `dma_read64` supply the GO for each
/// word (see commit 250ed09 — "the go is not sent by the driver ... we need to
/// invert timing for go/hostread in dma scenario").
///
/// Asserts against framebuffer contents, not against the other engine: a batch
/// and a scalar loop that are both off by the same word agree with each other,
/// which is how the SoftWindows phase error hid.
#[test]
fn test_hostr_dma_readback_phase_matches_framebuffer() {
    use crate::traits::BusDevice;

    // 16px wide CI8 @ 8 pixels per 64-bit word = 2 words/row, 2 rows.
    let (w, h) = (16i32, 2i32);
    let seed = |rex: &Rex3| unsafe {
        let fb = &mut *rex.fb_rgb.get();
        for y in 0..h {
            for x in 0..w {
                fb[(y as u32 * 2048 + x as u32) as usize] = (1 + y * w + x) as u32 & 0xFF;
            }
        }
    };
    // READ block armed with NO GO — the DMA path issues it.
    let arm = |rex: &Rex3| {
        reg(rex, REX3_DRAWMODE1, DM1_CI8_HOSTRW64);
        reg(rex, REX3_XYENDI,   xy(w - 1, h - 1));
        reg(rex, REX3_XYSTARTI, xy(0, 0));
        reg(rex, REX3_DRAWMODE0, DM0_READ_BLOCK);
    };

    let want = [
        u64::from_be_bytes([1, 2, 3, 4, 5, 6, 7, 8]),
        u64::from_be_bytes([9, 10, 11, 12, 13, 14, 15, 16]),
        u64::from_be_bytes([17, 18, 19, 20, 21, 22, 23, 24]),
        u64::from_be_bytes([25, 26, 27, 28, 29, 30, 31, 32]),
    ];

    // Scalar: one GO per word, GO before each read.
    let rex_s = make_rex3();
    rex3init(rex_s);
    seed(rex_s);
    arm(rex_s);
    let mut scalar = [0u64; 4];
    for slot in scalar.iter_mut() {
        let r = rex_s.dma_read64(go_addr(REX3_HOSTRW0));
        assert!(r.is_ok());
        *slot = r.data;
    }
    assert_eq!(scalar, want,
        "scalar dma_read64 phase wrong:\n got {scalar:#018x?}\nwant {want:#018x?}");

    // Batched: one token for the whole run.
    let rex_b = make_rex3();
    rex3init(rex_b);
    seed(rex_b);
    arm(rex_b);
    let mut batched = [0u64; 4];
    assert_eq!(rex_b.dma_read64_bulk(go_addr(REX3_HOSTRW0), &mut batched),
               crate::traits::BUS_OK);
    assert_eq!(batched, want,
        "batched dma read phase wrong:\n got {batched:#018x?}\nwant {want:#018x?}");
}

/// A READ block spanning several words must keep its phase across a *chunked*
/// readback: two back-to-back `dma_read64_bulk` calls on the same armed
/// primitive must return consecutive words, not restart or skip.
///
/// This is the shape SoftWindows uses to save the area under a popup, and the
/// one a single-batch test cannot exercise.
#[test]
fn test_hostr_dma_bulk_keeps_phase_across_chunks() {
    use crate::traits::BusDevice;

    // 32px wide CI8 = 4 words/row, 2 rows = 8 words total.
    let (w, h) = (32i32, 2i32);
    let rex = make_rex3();
    rex3init(rex);
    unsafe {
        let fb = &mut *rex.fb_rgb.get();
        for y in 0..h {
            for x in 0..w {
                fb[(y as u32 * 2048 + x as u32) as usize] = (1 + y * w + x) as u32 & 0xFF;
            }
        }
    }
    reg(rex, REX3_DRAWMODE1, DM1_CI8_HOSTRW64);
    reg(rex, REX3_XYENDI,   xy(w - 1, h - 1));
    reg(rex, REX3_XYSTARTI, xy(0, 0));
    reg(rex, REX3_DRAWMODE0, DM0_READ_BLOCK);

    // Read in two chunks of 4 words.
    let mut a = [0u64; 4];
    let mut b = [0u64; 4];
    assert_eq!(rex.dma_read64_bulk(go_addr(REX3_HOSTRW0), &mut a), crate::traits::BUS_OK);
    assert_eq!(rex.dma_read64_bulk(go_addr(REX3_HOSTRW0), &mut b), crate::traits::BUS_OK);

    let mut want = [0u64; 8];
    for (i, slot) in want.iter_mut().enumerate() {
        let base = (i * 8 + 1) as u8;
        *slot = u64::from_be_bytes([base, base+1, base+2, base+3, base+4, base+5, base+6, base+7]);
    }
    let got: Vec<u64> = a.iter().chain(b.iter()).copied().collect();
    assert_eq!(&got[..], &want[..],
        "chunked bulk read lost phase:\n got {got:#018x?}\nwant {want:#018x?}");
}

/// Full save/restore round trip through the DMA paths, the way SoftWindows
/// saves the area under a popup and puts it back.
///
/// Reads a block out with `dma_read64_bulk`, clears it, writes it back with
/// `dma_write64_bulk`, and requires the framebuffer to match what was there
/// before. A phase error in either direction shows up as shifted pixels.
#[test]
fn test_hostrw_dma_save_restore_round_trip() {
    use crate::traits::BusDevice;

    let (w, h) = (32i32, 4i32);
    let rex = make_rex3();
    rex3init(rex);

    let orig: Vec<u32> = (0..(w * h)).map(|i| (1 + i) as u32 & 0xFF).collect();
    unsafe {
        let fb = &mut *rex.fb_rgb.get();
        for y in 0..h {
            for x in 0..w {
                fb[(y as u32 * 2048 + x as u32) as usize] = orig[(y * w + x) as usize];
            }
        }
    }

    // Save: READ block, no arming GO (DMA supplies it).
    reg(rex, REX3_DRAWMODE1, DM1_CI8_HOSTRW64);
    reg(rex, REX3_XYENDI,   xy(w - 1, h - 1));
    reg(rex, REX3_XYSTARTI, xy(0, 0));
    reg(rex, REX3_DRAWMODE0, DM0_READ_BLOCK);

    let nwords = (w * h / 8) as usize;
    let mut saved = vec![0u64; nwords];
    assert_eq!(rex.dma_read64_bulk(go_addr(REX3_HOSTRW0), &mut saved),
               crate::traits::BUS_OK);

    // Scribble over the region so a failed restore is obvious.
    unsafe {
        let fb = &mut *rex.fb_rgb.get();
        for y in 0..h {
            for x in 0..w {
                fb[(y as u32 * 2048 + x as u32) as usize] = 0xEE;
            }
        }
    }

    // Restore: HOSTW block.
    reg(rex, REX3_DRAWMODE1, DM1_CI8_HOSTRW64);
    reg(rex, REX3_WRMASK, 0xFF);
    reg(rex, REX3_XYENDI,   xy(w - 1, h - 1));
    reg(rex, REX3_XYSTARTI, xy(0, 0));
    reg(rex, REX3_DRAWMODE0, DM0_HOSTW_BLOCK);
    assert_eq!(rex.dma_write64_bulk(go_addr(REX3_HOSTRW0), &saved),
               crate::traits::BUS_OK);
    wait(rex);

    for y in 0..h {
        for x in 0..w {
            let got = read_pixel(rex, x, y) & 0xFF;
            let want = orig[(y * w + x) as usize];
            assert_eq!(got, want,
                "save/restore mismatch at ({x},{y}): got {got:#04x} want {want:#04x}");
        }
    }
}

/// After a bulk read of N words, the *next* scalar `dma_read64` must return
/// word N — the batch must consume exactly N words from the primitive, no
/// more. Over-consuming shifts everything that follows, which is what a
/// save-under-popup would see as an 8-pixel slip at CI8.
#[test]
fn test_hostr_bulk_consumes_exactly_its_count() {
    use crate::traits::BusDevice;

    let (w, h) = (32i32, 2i32);
    let rex = make_rex3();
    rex3init(rex);
    unsafe {
        let fb = &mut *rex.fb_rgb.get();
        for y in 0..h {
            for x in 0..w {
                fb[(y as u32 * 2048 + x as u32) as usize] = (1 + y * w + x) as u32 & 0xFF;
            }
        }
    }
    reg(rex, REX3_DRAWMODE1, DM1_CI8_HOSTRW64);
    reg(rex, REX3_XYENDI,   xy(w - 1, h - 1));
    reg(rex, REX3_XYSTARTI, xy(0, 0));
    reg(rex, REX3_DRAWMODE0, DM0_READ_BLOCK);

    let mut first2 = [0u64; 2];
    assert_eq!(rex.dma_read64_bulk(go_addr(REX3_HOSTRW0), &mut first2),
               crate::traits::BUS_OK);

    // Next scalar read must be word 2 = pixels 17..24.
    let next = rex.dma_read64(go_addr(REX3_HOSTRW0));
    assert!(next.is_ok());
    let want = u64::from_be_bytes([17, 18, 19, 20, 21, 22, 23, 24]);
    assert_eq!(next.data, want,
        "bulk of 2 then scalar: expected word 2 ({want:#018x}), got {:#018x} — \
         the batch consumed the wrong number of words", next.data);
}

/// A HOSTW block whose pixel count exceeds one word must consume words in the
/// same order and quantity whether fed one-per-GO (scalar) or as a batch.
///
/// The batch runs the whole primitive under a single GO, so anything the
/// hardware advances *per GO* rather than per word diverges. Tiled/zoomed
/// fills are the case that would expose it.
#[test]
fn test_hostw_batch_matches_scalar_on_a_multiword_row() {
    use crate::traits::BusDevice;

    // 24px wide CI8 @ 8px/word = 3 words in one row.
    let (w, h) = (24i32, 1i32);
    let words: Vec<u64> = (0..3u64)
        .map(|i| {
            let b = (i * 8 + 1) as u8;
            u64::from_be_bytes([b, b+1, b+2, b+3, b+4, b+5, b+6, b+7])
        })
        .collect();

    let setup = |rex: &Rex3| {
        rex3init(rex);
        reg(rex, REX3_DRAWMODE1, DM1_CI8_HOSTRW64);
        reg(rex, REX3_WRMASK, 0xFF);
        reg(rex, REX3_XYENDI,   xy(w - 1, h - 1));
        reg(rex, REX3_XYSTARTI, xy(0, 0));
        reg(rex, REX3_DRAWMODE0, DM0_HOSTW_BLOCK);
    };

    let rex_s = make_rex3();
    setup(rex_s);
    for &v in &words { write_hostrw64(rex_s, v); }
    wait(rex_s);

    let rex_b = make_rex3();
    setup(rex_b);
    assert_eq!(rex_b.dma_write64_bulk(go_addr(REX3_HOSTRW0), &words),
               crate::traits::BUS_OK);
    wait(rex_b);

    for x in 0..w {
        let s = read_pixel(rex_s, x, 0) & 0xFF;
        let b = read_pixel(rex_b, x, 0) & 0xFF;
        assert_eq!(b, s, "batch vs scalar at x={x}: {b:#04x} != {s:#04x}");
        assert_eq!(b, (x + 1) as u32, "wrong pixel value at x={x}");
    }
}

/// A batched upload spanning several rows must paint every row, not just the
/// first.
///
/// Host mode stops the walker at each row boundary and expects the next GO to
/// resume — the CPU's one-word-per-GO feed gives it that for free. A batch has
/// one GO for N words, so `execute_go` has to drive the primitive round until
/// the batch is drained. Without that, the upload paints row 0 and silently
/// drops the rest: pixmaps render as empty frames, tiled fills leave bands of
/// untouched framebuffer.
#[test]
fn test_hostw_batch_paints_every_row_of_a_multirow_block() {
    use crate::traits::BusDevice;

    // 12px wide CI8: each row is one full 64-bit word plus a HALF word, so the
    // walker hits the partial-word row break (`hostcnt > 0`) that a
    // word-aligned width never triggers. 4 rows, 2 words per row.
    let (w, h) = (12i32, 4i32);
    let words: Vec<u64> = (0..(h as u64 * 2))
        .map(|k| {
            let b = (k * 8 + 1) as u8;
            u64::from_be_bytes([b, b+1, b+2, b+3, b+4, b+5, b+6, b+7])
        })
        .collect();

    let setup = |rex: &Rex3| {
        rex3init(rex);
        reg(rex, REX3_DRAWMODE1, DM1_CI8_HOSTRW64);
        reg(rex, REX3_WRMASK, 0xFF);
        reg(rex, REX3_XYENDI,   xy(w - 1, h - 1));
        reg(rex, REX3_XYSTARTI, xy(0, 0));
        reg(rex, REX3_DRAWMODE0, DM0_HOSTW_BLOCK);
    };

    // Reference: the CPU's one-word-per-GO feed, which is what the batch must
    // reproduce. Its output defines correct — including whatever the row
    // boundary does to a partial word.
    let rex_s = make_rex3();
    setup(rex_s);
    for &v in &words { write_hostrw64(rex_s, v); }
    wait(rex_s);

    let rex_b = make_rex3();
    setup(rex_b);
    assert_eq!(rex_b.dma_write64_bulk(go_addr(REX3_HOSTRW0), &words),
               crate::traits::BUS_OK);
    wait(rex_b);

    for y in 0..h {
        for x in 0..w {
            let s = read_pixel(rex_s, x, y) & 0xFF;
            let b = read_pixel(rex_b, x, y) & 0xFF;
            assert_eq!(b, s,
                "row {y} col {x}: batch painted {b:#04x}, scalar painted {s:#04x} \
                 — a batch must reproduce the one-word-per-GO feed exactly");
        }
    }
    // And it is real data, not two matching blanks.
    assert_ne!(read_pixel(rex_b, 0, h - 1) & 0xFF, 0,
        "last row is blank: the batch was dropped before finishing");
}

/// The real icon-upload shape from `vdma.log`: 98 x 98 RGB24, 49 64-bit words
/// per row, 4802 words in one batch.
///
/// Drives the batch against the CPU's one-word-per-GO feed, which is the
/// reference. An odd word count per row means the row boundary lands
/// mid-stream for every row after the first, so any mismatch between the two
/// feeds shows up as a progressive shift — exactly how the login-screen icons
/// came out empty/garbled.
#[test]
fn test_hostw_batch_matches_scalar_on_the_icon_upload_shape() {
    use crate::traits::BusDevice;

    // RGB24 + rwdouble: 2 pixels per 64-bit word.
    let (w, h) = (98i32, 98i32);
    let words_per_row = (w as usize) / 2;              // 49
    let total = words_per_row * h as usize;            // 4802
    let words: Vec<u64> = (0..total as u64)
        .map(|k| {
            // Distinct, non-zero per word so a shift is unmistakable.
            let hi = (0x10_0000u64 + k) & 0xFF_FFFF;
            let lo = (0x80_0000u64 + k) & 0xFF_FFFF;
            (hi << 32) | lo
        })
        .collect();
    assert_eq!(total, 4802, "shape must match the logged icon transfer");

    let setup = |rex: &Rex3| {
        rex3init(rex);
        reg(rex, REX3_DRAWMODE1, DM1_RGB24_HOSTRW64);
        reg(rex, REX3_WRMASK, 0xFFFFFF);
        reg(rex, REX3_XYENDI,   xy(w - 1, h - 1));
        reg(rex, REX3_XYSTARTI, xy(0, 0));
        reg(rex, REX3_DRAWMODE0, DM0_HOSTW_BLOCK);
    };

    let rex_s = make_rex3();
    setup(rex_s);
    for &v in &words { write_hostrw64(rex_s, v); }
    wait(rex_s);

    let rex_b = make_rex3();
    setup(rex_b);
    assert_eq!(rex_b.dma_write64_bulk(go_addr(REX3_HOSTRW0), &words),
               crate::traits::BUS_OK);
    wait(rex_b);

    let mut first_bad = None;
    for y in 0..h {
        for x in 0..w {
            let s = read_pixel(rex_s, x, y) & 0xFFFFFF;
            let b = read_pixel(rex_b, x, y) & 0xFFFFFF;
            if s != b && first_bad.is_none() {
                first_bad = Some((x, y, b, s));
            }
        }
    }
    if let Some((x, y, b, s)) = first_bad {
        panic!("icon-shape batch diverges at ({x},{y}): batch={b:#08x} scalar={s:#08x}");
    }
    // Non-vacuous: the last row must hold real data, not a matching blank.
    assert_ne!(read_pixel(rex_b, 0, h - 1) & 0xFFFFFF, 0,
        "last row blank — the batch stopped early");
}

// ── GFIFO batch push/drain ──────────────────────────────────────────────────

/// `push_batch` + `drain_payload` must move every word, including across the
/// ring's wrap point and for a batch far larger than the 64-entry head-publish
/// interval.
///
/// The icon uploads that came out empty push ~4802 payload entries behind one
/// token, so any truncation in this pair loses most of an image.
#[test]
fn test_gfifo_push_batch_round_trips_every_word() {
    // GFifo is 65536 entries; build it on a thread with room for it,
    // the same reason make_rex3 does.
    std::thread::Builder::new().stack_size(64 * 1024 * 1024).spawn(|| {
        use crate::rex3::{GFifo, HostRwArray, REX3_DMA_BATCH_W, HOSTRW_BUF_QWORDS};

        // Heap-allocate: GFifo is 65536 entries and overflows a test stack.
        let fifo = Box::new(GFifo::new());
        let mut dst = Box::new(HostRwArray::default());

        for n in [1usize, 2, 63, 64, 65, 4802] {
            let words: Vec<u64> = (0..n as u64).map(|k| 0xAAAA_0000_0000_0000 | k).collect();
            fifo.push_batch(REX3_DMA_BATCH_W, n as u64, &words);

            // The consumer sees the token first, exactly as register_processor does.
            let (addr, val) = fifo.peek().expect("token must be queued");
            assert_eq!(addr, REX3_DMA_BATCH_W, "n={n}: first entry must be the token");
            assert_eq!(val, n as u64, "n={n}: token must carry the count");

            let got = fifo.drain_payload(n.min(HOSTRW_BUF_QWORDS), &mut dst);
            assert_eq!(got, n, "n={n}: drained {got} of {n} payload words");
            for (i, w) in words.iter().enumerate() {
                assert_eq!(dst[i], *w, "n={n}: word {i} differs");
            }
            // Retire the last payload entry, as the consumer loop does.
            fifo.consume();
            assert!(fifo.is_empty(), "n={n}: queue must be empty after the batch");
        }
    }).expect("spawn").join().expect("test thread panicked");
}

/// Back-to-back batches must not bleed into each other: the second batch's
/// payload entries carry the same GFIFO_PAYLOAD marker, so `drain_payload`
/// has to stop at its own count rather than at the marker alone.
#[test]
fn test_gfifo_back_to_back_batches_stay_separate() {
    // GFifo is 65536 entries; build it on a thread with room for it,
    // the same reason make_rex3 does.
    std::thread::Builder::new().stack_size(64 * 1024 * 1024).spawn(|| {
        use crate::rex3::{GFifo, HostRwArray, REX3_DMA_BATCH_W, HOSTRW_BUF_QWORDS};

        let fifo = Box::new(GFifo::new());
        let mut dst = Box::new(HostRwArray::default());

        let a: Vec<u64> = (0..100u64).map(|k| 0xA000_0000_0000_0000 | k).collect();
        let b: Vec<u64> = (0..50u64).map(|k| 0xB000_0000_0000_0000 | k).collect();
        fifo.push_batch(REX3_DMA_BATCH_W, a.len() as u64, &a);
        fifo.push_batch(REX3_DMA_BATCH_W, b.len() as u64, &b);

        let (_, va) = fifo.peek().unwrap();
        assert_eq!(va, a.len() as u64);
        assert_eq!(fifo.drain_payload(a.len().min(HOSTRW_BUF_QWORDS), &mut dst), a.len());
        for (i, w) in a.iter().enumerate() {
            assert_eq!(dst[i], *w, "batch A word {i}");
        }
        fifo.consume();

        let (_, vb) = fifo.peek().expect("second token must follow");
        assert_eq!(vb, b.len() as u64, "second batch's token must be next");
        assert_eq!(fifo.drain_payload(b.len().min(HOSTRW_BUF_QWORDS), &mut dst), b.len());
        for (i, w) in b.iter().enumerate() {
            assert_eq!(dst[i], *w, "batch B word {i}");
        }
        fifo.consume();
        assert!(fifo.is_empty());
    }).expect("spawn").join().expect("test thread panicked");
}

/// `drain_payload` must return exactly the promised count, even when the
/// producer is still publishing and `tail` is sampled mid-batch.
///
/// The old version sampled `tail` once and stopped at `head == tail`, so a
/// batch larger than what had been published at that instant came back short —
/// and the caller could not tell a truncated drain from a complete one. A
/// 4802-word icon upload losing its tail is exactly the "pixmaps missing"
/// symptom.
#[test]
fn test_gfifo_drain_payload_returns_the_exact_promised_count() {
    use crate::rex3::{GFifo, HostRwArray, REX3_DMA_BATCH_W, HOSTRW_BUF_QWORDS};
    use std::sync::Arc;

    std::thread::Builder::new().stack_size(64 * 1024 * 1024).spawn(|| {
        let fifo = Arc::new(GFifo::new());
        let mut dst = Box::new(HostRwArray::default());

        const N: usize = 4802; // the logged icon-upload size
        let words: Vec<u64> = (0..N as u64).map(|k| 0xC0DE_0000_0000_0000 | k).collect();

        // `push_batch` publishes the token and all N words with a single
        // Release store on `tail`, so a consumer that merely waits for the
        // token already sees everything — which makes a naive "race" test
        // vacuous. Reproduce the real hazard directly instead: publish the
        // token with a tail that covers only part of the batch, exactly what a
        // single mid-batch `tail` sample looks like to the drain loop.
        fifo.push_batch(REX3_DMA_BATCH_W, N as u64, &words);
        fifo.rewind_tail_for_test(1 + N / 2);

        let (addr, val) = fifo.peek().expect("token");
        assert_eq!(addr, REX3_DMA_BATCH_W);
        assert_eq!(val, N as u64);

        // Republish the rest shortly after the drain starts, as the producer
        // would. The drain must wait for it rather than returning short.
        let f2 = Arc::clone(&fifo);
        let producer = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(20));
            f2.restore_tail_for_test(1 + N);
        });
        let got = fifo.drain_payload(N.min(HOSTRW_BUF_QWORDS), &mut dst);
        assert_eq!(got, N, "drain must return the full promised count, got {got}");
        for i in 0..N {
            assert_eq!(dst[i], words[i], "word {i} differs");
        }
        producer.join().unwrap();
    }).expect("spawn").join().expect("test thread panicked");
}

/// The draw-debug ring must get a record for an ordinary host-mode block —
/// that is what the pixel-under-cursor overlay reads.
///
/// `log_block` bails when `mid_primitive` is set, so anything that leaves that
/// flag latched across GOs silently empties the overlay.
#[test]
#[cfg(feature = "developer")]
fn test_draw_debug_ring_records_a_host_block() {
    let rex = make_rex3();
    rex3init(rex);
    rex.draw_debug.store(true, std::sync::atomic::Ordering::Relaxed);
    rex.draw_ring.lock().count = 0;

    reg(rex, REX3_DRAWMODE1, DM1_CI8_HOSTRW64);
    reg(rex, REX3_WRMASK, 0xFF);
    reg(rex, REX3_XYENDI,   xy(7, 1));
    reg(rex, REX3_XYSTARTI, xy(0, 0));
    reg(rex, REX3_DRAWMODE0, DM0_HOSTW_BLOCK);
    write_hostrw64(rex, 0x1122_3344_5566_7788);
    write_hostrw64(rex, 0x99AA_BBCC_DDEE_FF00);
    wait(rex);

    let n = rex.draw_ring.lock().count;
    assert!(n > 0, "draw ring is empty after a host block — the overlay has nothing to show");
}

/// Same, for a plain DRAW block (no host data) — the other shape the overlay
/// is meant to report.
#[test]
#[cfg(feature = "developer")]
fn test_draw_debug_ring_records_a_plain_block() {
    let rex = make_rex3();
    rex3init(rex);
    rex.draw_debug.store(true, std::sync::atomic::Ordering::Relaxed);
    rex.draw_ring.lock().count = 0;

    reg(rex, REX3_DRAWMODE1, DM1_CI8_HOSTRW);
    reg(rex, REX3_WRMASK, 0xFF);
    reg(rex, REX3_COLORI, 0x5A);
    reg(rex, REX3_XYENDI,   xy(15, 7));
    reg(rex, REX3_XYSTARTI, xy(0, 0));
    reg_go(rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);
    wait(rex);

    let n = rex.draw_ring.lock().count;
    assert!(n > 0, "draw ring is empty after a plain DRAW block");
}

/// A *batched* host block must produce a draw record too.
///
/// The batch runs the whole primitive under one GO, so if `mid_primitive` is
/// still set when `log_block` runs — or the continuation path re-enters
/// `draw_primitive` in a way that suppresses it — the overlay shows nothing for
/// exactly the transfers that matter most (pixmap and icon uploads).
#[test]
#[cfg(feature = "developer")]
fn test_draw_debug_ring_records_a_batched_host_block() {
    use crate::traits::BusDevice;

    let rex = make_rex3();
    rex3init(rex);
    rex.draw_debug.store(true, std::sync::atomic::Ordering::Relaxed);
    rex.draw_ring.lock().count = 0;

    reg(rex, REX3_DRAWMODE1, DM1_CI8_HOSTRW64);
    reg(rex, REX3_WRMASK, 0xFF);
    reg(rex, REX3_XYENDI,   xy(7, 3));
    reg(rex, REX3_XYSTARTI, xy(0, 0));
    reg(rex, REX3_DRAWMODE0, DM0_HOSTW_BLOCK);

    let words: Vec<u64> = (0..4u64).map(|k| 0x1000_0000_0000_0000 | k).collect();
    assert_eq!(rex.dma_write64_bulk(go_addr(REX3_HOSTRW0), &words),
               crate::traits::BUS_OK);
    wait(rex);

    let n = rex.draw_ring.lock().count;
    assert!(n > 0,
        "draw ring is empty after a BATCHED host block — the overlay goes blank \
         for pixmap/icon uploads even though the pixels land");
}

/// The draw-debug overlay's HOSTRW counter must match what a batch actually
/// delivered.
///
/// A DMA batch hands REX3 N words in one call, so counting the token as a
/// single write made the overlay report `0/4802` (or `1/4802`) for icon
/// uploads that carried every word — the counter said the data was missing
/// when it was not, which is worse than no counter at all.
#[test]
#[cfg(feature = "developer")]
fn test_draw_debug_counts_every_word_of_a_batch() {
    use crate::traits::BusDevice;

    // 8px CI8 rows, 4 rows = 4 words, all delivered in one batch.
    let rex = make_rex3();
    rex3init(rex);
    rex.draw_debug.store(true, std::sync::atomic::Ordering::Relaxed);
    rex.draw_ring.lock().count = 0;

    reg(rex, REX3_DRAWMODE1, DM1_CI8_HOSTRW64);
    reg(rex, REX3_WRMASK, 0xFF);
    reg(rex, REX3_XYENDI,   xy(7, 3));
    reg(rex, REX3_XYSTARTI, xy(0, 0));
    reg(rex, REX3_DRAWMODE0, DM0_HOSTW_BLOCK);

    let words: Vec<u64> = (0..4u64).map(|k| 0x1122_3344_5566_0000 | k).collect();
    assert_eq!(rex.dma_write64_bulk(go_addr(REX3_HOSTRW0), &words),
               crate::traits::BUS_OK);
    wait(rex);

    let ring = rex.draw_ring.lock();
    let rec = ring.iter_newest_first().next().copied()
        .expect("a host block must produce a draw record");
    drop(ring);

    assert!(rec.expected_words > 0,
        "colorhost draw must have a non-zero expected word count");
    assert_eq!(rec.hostrw_writes, words.len() as u32,
        "overlay counted {} HOSTRW writes for a {}-word batch (expected_words={}, \
         expected_doubles={}) — the counter must reflect what was delivered",
        rec.hostrw_writes, words.len(), rec.expected_words, rec.expected_doubles);
    assert_eq!(rec.spurious_writes, 0, "no write should be counted as spurious");
}

/// A batch's word count must not leak onto the *next*, non-host primitive.
///
/// `host_len` stays raised until `execute_go` retires it, so a following plain
/// DRAW was being credited with the previous batch's words — and since it has
/// colorhost=0 they landed in `spurious_writes`, so the overlay reported
/// phantom host traffic (`SPURIOUS:1`) on an ordinary block.
#[test]
#[cfg(feature = "developer")]
fn test_draw_debug_batch_count_does_not_leak_to_the_next_draw() {
    use crate::traits::BusDevice;

    let rex = make_rex3();
    rex3init(rex);
    rex.draw_debug.store(true, std::sync::atomic::Ordering::Relaxed);
    rex.draw_ring.lock().count = 0;

    // A host batch.
    reg(rex, REX3_DRAWMODE1, DM1_CI8_HOSTRW64);
    reg(rex, REX3_WRMASK, 0xFF);
    reg(rex, REX3_XYENDI,   xy(7, 3));
    reg(rex, REX3_XYSTARTI, xy(0, 0));
    reg(rex, REX3_DRAWMODE0, DM0_HOSTW_BLOCK);
    let words: Vec<u64> = (0..4u64).map(|k| 0x2233_4455_6677_0000 | k).collect();
    assert_eq!(rex.dma_write64_bulk(go_addr(REX3_HOSTRW0), &words),
               crate::traits::BUS_OK);
    wait(rex);

    // Then an ordinary DRAW that consumes no host data at all.
    reg(rex, REX3_DRAWMODE1, DM1_CI8_HOSTRW);
    reg(rex, REX3_COLORI, 0x5A);
    reg(rex, REX3_XYENDI,   xy(15, 15));
    reg(rex, REX3_XYSTARTI, xy(0, 8));
    reg_go(rex, REX3_DRAWMODE0, DM0_DRAW_BLOCK);
    wait(rex);

    let ring = rex.draw_ring.lock();
    let newest = ring.iter_newest_first().next().copied().expect("record");
    drop(ring);

    assert_eq!(newest.spurious_writes, 0,
        "the plain DRAW was credited with {} spurious HOSTRW writes — the \
         previous batch's count leaked onto it", newest.spurious_writes);
    assert_eq!(newest.hostrw_writes, 0,
        "the plain DRAW consumes no host data but was credited with {} writes",
        newest.hostrw_writes);
}

/// Multi-row HOSTW batch **without STOPONY** — the shape IRIX actually uses
/// for the tiled wallpaper and the login icons.
///
/// Without STOPONY the walker treats each row as its own primitive: it clears
/// `mid_primitive` and breaks at the end of every row, expecting the next GO to
/// start the next row. The CPU's one-word-per-GO feed supplies those GOs. A
/// batch has exactly ONE, so anything that stops the resume loop at the first
/// row boundary silently discards every remaining row — leaving holes in an
/// image whose data arrived intact.
#[test]
fn test_hostw_batch_without_stopony_paints_all_rows() {
    use crate::traits::BusDevice;

    // 8px CI8 rows = exactly one 64-bit word per row, 4 rows.
    const DM0_HOSTW_BLOCK_NO_STOPONY: u32 =
        DRAWMODE0_OPCODE_DRAW | DRAWMODE0_ADRMODE_BLOCK_SH | DM0_STOPONX | DM0_COLORHOST;
    let (w, h) = (8i32, 4i32);
    let words: Vec<u64> = (0..h as u64)
        .map(|r| {
            let b = (r * 8 + 1) as u8;
            u64::from_be_bytes([b, b+1, b+2, b+3, b+4, b+5, b+6, b+7])
        })
        .collect();

    let setup = |rex: &Rex3| {
        rex3init(rex);
        reg(rex, REX3_DRAWMODE1, DM1_CI8_HOSTRW64);
        reg(rex, REX3_WRMASK, 0xFF);
        reg(rex, REX3_XYENDI,   xy(w - 1, h - 1));
        reg(rex, REX3_XYSTARTI, xy(0, 0));
        reg(rex, REX3_DRAWMODE0, DM0_HOSTW_BLOCK_NO_STOPONY);
    };

    // Reference: one GO per word, which is what the hardware feed does.
    let rex_s = make_rex3();
    setup(rex_s);
    for &v in &words { write_hostrw64(rex_s, v); }
    wait(rex_s);

    let rex_b = make_rex3();
    setup(rex_b);
    assert_eq!(rex_b.dma_write64_bulk(go_addr(REX3_HOSTRW0), &words),
               crate::traits::BUS_OK);
    wait(rex_b);

    for y in 0..h {
        for x in 0..w {
            let s = read_pixel(rex_s, x, y) & 0xFF;
            let b = read_pixel(rex_b, x, y) & 0xFF;
            assert_eq!(b, s,
                "row {y} col {x}: batch={b:#04x} scalar={s:#04x} — rows after the \
                 first are dropped when the batch stops at a row boundary");
        }
    }
    assert_ne!(read_pixel(rex_b, 0, h - 1) & 0xFF, 0, "last row is blank");
}


// ── Bulk vs scalar DMA equivalence, with position-bearing pixel data ────────
//
// The tests above each pin one shape. These sweep width x height through the
// *DMA entry points the VDMA engine actually calls*, in both directions, and
// verify every pixel against a generated pattern rather than only against the
// other engine.
//
// The pattern is a 32-bit counter spread over 4 consecutive CI8 pixels, so each
// byte carries a slice of the index of the pixel group it belongs to. That
// makes the two failure modes distinguishable, which batch-vs-scalar comparison
// alone cannot do:
//
//   * a pixel holding the *right* value at the *wrong* place is a skew — the
//     bytes are intact but shifted, so the decoded counter is off by a
//     predictable amount;
//   * a pixel holding a value no counter would ever produce is dropped or
//     merged data.
//
// The widths matter more than the heights. A CI8 row of 8 or 16 px is a whole
// number of 64-bit words, so a row boundary always lands on a word boundary and
// both engines agree even when the row/word interaction is wrong. Widths like
// 12 and 20 leave a *partial* trailing word on every row — that is where a skew
// shows, and it is the shape the taskbar blit that exposed this uses.

const DM0_HOSTW_NO_STOPONY: u32 =
    DRAWMODE0_OPCODE_DRAW | DRAWMODE0_ADRMODE_BLOCK_SH | DM0_STOPONX | DM0_COLORHOST;
const DM0_HOSTR_NO_STOPONY: u32 =
    DRAWMODE0_OPCODE_READ | DRAWMODE0_ADRMODE_BLOCK_SH | DM0_STOPONX | DM0_COLORHOST
        | DM0_DOSETUP;

/// The byte a given pixel index must hold.
///
/// Pixel `i` belongs to counter group `i / 4` and is byte `i % 4` of that
/// group's 32-bit value. The counter is offset by 1 and the low byte forced
/// non-zero so that no legal pixel is ever 0x00 — a cleared framebuffer is
/// therefore never mistaken for correctly-transferred data.
fn counter_byte(i: usize) -> u8 {
    let group = (i / 4) as u32 + 1;
    let counter = group.wrapping_mul(0x0105_0307) | 0x0100_0001;
    counter.to_be_bytes()[i % 4]
}

/// Decode a pixel byte back to the set of indices that could have produced it.
/// Used only to make failure messages actionable.
fn counter_explain(i: usize, got: u8) -> String {
    for cand in 0..4096usize {
        if counter_byte(cand) == got {
            let delta = cand as isize - i as isize;
            return format!("(value belongs to pixel {cand}, skew {delta:+})");
        }
    }
    "(value matches no pixel — dropped or merged data)".to_string()
}

/// Pack a `w` x `h` CI8 image of counter bytes into 64-bit words, one whole
/// word group per row (a partial trailing word is zero-padded, exactly as the
/// hardware's per-row word flush leaves it).
fn counter_words(w: i32, h: i32) -> Vec<u64> {
    let per_row = ((w as usize) + 7) / 8;
    let mut out = Vec::with_capacity(per_row * h as usize);
    for r in 0..h as usize {
        for c in 0..per_row {
            let mut v = 0u64;
            for k in 0..8usize {
                let x = c * 8 + k;
                // Index by absolute pixel position in the image, so the value
                // encodes where the pixel belongs, not merely its order.
                let byte = if x < w as usize { counter_byte(r * w as usize + x) } else { 0 };
                v |= (byte as u64) << (56 - k * 8);
            }
            out.push(v);
        }
    }
    out
}

/// Write the counter pattern straight into the framebuffer, bypassing REX3.
/// The readback tests need a known image on screen without depending on the
/// write path they are meant to be independent of.
fn prefill_counter_image(rex: &Rex3, w: i32, h: i32) {
    unsafe {
        let fb = &mut *rex.fb_rgb.get();
        for y in 0..h as usize {
            for x in 0..w as usize {
                fb[y * 2048 + x] = counter_byte(y * w as usize + x) as u32;
            }
        }
    }
}

const SWEEP_SHAPES: &[(i32, i32)] = &[
    (8, 1), (8, 4), (8, 7),      // exact single word per row
    (16, 3), (16, 5),            // exact two words per row
    (12, 4), (12, 9),            // partial trailing word — the skew case
    (20, 3), (20, 6),            // partial, wider
    (4, 5),  (1, 3),             // sub-word rows
    (33, 2),                     // odd width, > 4 words
];

/// HOSTW: a bulk DMA write must place every pixel where the counter says it
/// belongs, and must match the scalar one-word-per-GO feed doing the same.
#[test]
fn test_hostw_bulk_matches_scalar_and_counter_pattern() {
    use crate::traits::BusDevice;

    for &(w, h) in SWEEP_SHAPES {
        let words = counter_words(w, h);

        let setup = |rex: &Rex3| {
            rex3init(rex);
            reg(rex, REX3_DRAWMODE1, DM1_CI8_HOSTRW64);
            reg(rex, REX3_WRMASK, 0xFF);
            reg(rex, REX3_XYENDI,   xy(w - 1, h - 1));
            reg(rex, REX3_XYSTARTI, xy(0, 0));
            reg(rex, REX3_DRAWMODE0, DM0_HOSTW_NO_STOPONY);
        };

        // Reference engine: one GO per word, as the hardware feed does.
        let rex_s = make_rex3();
        setup(rex_s);
        for &v in &words { write_hostrw64(rex_s, v); }
        wait(rex_s);

        // Under test: one bulk DMA write, exactly as mc_vdma issues it.
        let rex_b = make_rex3();
        setup(rex_b);
        assert_eq!(rex_b.dma_write64_bulk(go_addr(REX3_HOSTRW0), &words),
                   crate::traits::BUS_OK, "{w}x{h}: bulk write refused");
        wait(rex_b);

        for y in 0..h {
            for x in 0..w {
                let i = y as usize * w as usize + x as usize;
                let want = counter_byte(i);
                let b = (read_pixel(rex_b, x, y) & 0xFF) as u8;
                let s = (read_pixel(rex_s, x, y) & 0xFF) as u8;
                assert_eq!(b, want,
                    "{w}x{h} bulk at ({x},{y}) pixel {i}: got {b:#04x} want {want:#04x} {}",
                    counter_explain(i, b));
                assert_eq!(s, want,
                    "{w}x{h} scalar at ({x},{y}) pixel {i}: got {s:#04x} want {want:#04x} {}",
                    counter_explain(i, s));
            }
        }
    }
}

/// HOSTR: a bulk DMA read must return the counter image that is on screen, and
/// must match the scalar GO-wait-read feed reading the same screen.
#[test]
fn test_hostr_bulk_matches_scalar_and_counter_pattern() {
    use crate::traits::BusDevice;

    for &(w, h) in SWEEP_SHAPES {
        let per_row = ((w as usize) + 7) / 8;
        let nwords = per_row * h as usize;
        // What the screen holds, packed the way a correct readback must return
        // it: one whole word group per row, partial trailing word zero-padded.
        let want = counter_words(w, h);

        let arm = |rex: &Rex3| {
            rex3init(rex);
            prefill_counter_image(rex, w, h);
            reg(rex, REX3_DRAWMODE1, DM1_CI8_HOSTRW64);
            reg(rex, REX3_XYENDI,   xy(w - 1, h - 1));
            reg(rex, REX3_XYSTARTI, xy(0, 0));
            reg(rex, REX3_DRAWMODE0, DM0_HOSTR_NO_STOPONY);
        };

        // Scalar: one GO-wait-read per word.
        let rex_s = make_rex3();
        arm(rex_s);
        let mut scalar = Vec::with_capacity(nwords);
        for k in 0..nwords {
            let r = rex_s.dma_read64(go_addr(REX3_HOSTRW0));
            assert!(r.is_ok(), "{w}x{h}: scalar read {k} failed");
            scalar.push(r.data);
        }

        // Bulk: one token, one pipeline drain, copy the array out.
        let rex_b = make_rex3();
        arm(rex_b);
        let mut bulk = vec![0u64; nwords];
        assert_eq!(rex_b.dma_read64_bulk(go_addr(REX3_HOSTRW0), &mut bulk),
                   crate::traits::BUS_OK, "{w}x{h}: bulk read refused");

        for k in 0..nwords {
            let row = k / per_row;
            let col = k % per_row;
            assert_eq!(bulk[k], want[k],
                "{w}x{h} bulk word {k} (row {row}, word {col} of row):\n  \
                 got  {:016x}\n  want {:016x}", bulk[k], want[k]);
            assert_eq!(scalar[k], want[k],
                "{w}x{h} scalar word {k} (row {row}, word {col} of row):\n  \
                 got  {:016x}\n  want {:016x}", scalar[k], want[k]);
        }
    }
}

/// Round trip: upload by bulk DMA, read back by bulk DMA, get the same words.
///
/// Both directions share the walker, so this alone could not catch a fault in
/// the walker — the two tests above pin each direction to the counter pattern
/// for that. This one catches the pairing: an upload and a readback that are
/// each self-consistent but disagree about where row `n` starts.
#[test]
fn test_hostrw_bulk_round_trip_preserves_the_image() {
    use crate::traits::BusDevice;

    for &(w, h) in SWEEP_SHAPES {
        let src = counter_words(w, h);

        let rex = make_rex3();
        rex3init(rex);
        reg(rex, REX3_DRAWMODE1, DM1_CI8_HOSTRW64);
        reg(rex, REX3_WRMASK, 0xFF);
        reg(rex, REX3_XYENDI,   xy(w - 1, h - 1));
        reg(rex, REX3_XYSTARTI, xy(0, 0));
        reg(rex, REX3_DRAWMODE0, DM0_HOSTW_NO_STOPONY);
        assert_eq!(rex.dma_write64_bulk(go_addr(REX3_HOSTRW0), &src),
                   crate::traits::BUS_OK, "{w}x{h}: bulk write refused");
        wait(rex);

        reg(rex, REX3_DRAWMODE1, DM1_CI8_HOSTRW64);
        reg(rex, REX3_XYENDI,   xy(w - 1, h - 1));
        reg(rex, REX3_XYSTARTI, xy(0, 0));
        reg(rex, REX3_DRAWMODE0, DM0_HOSTR_NO_STOPONY);
        let mut back = vec![0u64; src.len()];
        assert_eq!(rex.dma_read64_bulk(go_addr(REX3_HOSTRW0), &mut back),
                   crate::traits::BUS_OK, "{w}x{h}: bulk read refused");

        assert_eq!(back, src,
            "{w}x{h}: round trip changed the image\n  wrote {:016x?}\n  read  {:016x?}",
            src, back);
    }
}

/// The row-boundary word flush, **with STOPONY** — the shape that actually
/// reaches it.
///
/// With STOPONY the block walker advances rows itself instead of ending the
/// primitive at each row, so a row whose width does not fill a whole 64-bit
/// word leaves a partial word open in the shifter. Hardware sends one word per
/// GO in host mode, full or not, so that partial word must be flushed at the
/// row boundary; carrying it into the next row shifts every subsequent row by
/// the remainder.
///
/// The `!stopony` sweep above cannot catch this: it breaks out of the walker at
/// the row boundary before the flush is reached. Widths here are deliberately
/// not multiples of 8.
#[test]
fn test_hostw_stopony_flushes_partial_word_at_row_boundary() {
    use crate::traits::BusDevice;

    const DM0_HOSTW_STOPONY: u32 = DRAWMODE0_OPCODE_DRAW | DRAWMODE0_ADRMODE_BLOCK_SH
        | DM0_STOPONXY | DM0_COLORHOST;

    for &(w, h) in &[(12, 4), (20, 3), (4, 6), (33, 2), (12, 9)] {
        let words = counter_words(w, h);

        let rex = make_rex3();
        rex3init(rex);
        reg(rex, REX3_DRAWMODE1, DM1_CI8_HOSTRW64);
        reg(rex, REX3_WRMASK, 0xFF);
        reg(rex, REX3_XYENDI,   xy(w - 1, h - 1));
        reg(rex, REX3_XYSTARTI, xy(0, 0));
        reg(rex, REX3_DRAWMODE0, DM0_HOSTW_STOPONY);
        assert_eq!(rex.dma_write64_bulk(go_addr(REX3_HOSTRW0), &words),
                   crate::traits::BUS_OK, "{w}x{h}: bulk write refused");
        wait(rex);

        for y in 0..h {
            for x in 0..w {
                let i = y as usize * w as usize + x as usize;
                let want = counter_byte(i);
                let got = (read_pixel(rex, x, y) & 0xFF) as u8;
                assert_eq!(got, want,
                    "{w}x{h} at ({x},{y}) pixel {i}: got {got:#04x} want {want:#04x} {}\n\
                     a partial word left open at the row boundary shifts every \
                     following row", counter_explain(i, got));
            }
        }
    }
}

/// The readback mirror: with STOPONY, a partial trailing word must be published
/// at the row boundary rather than accumulating pixels from the next row.
#[test]
fn test_hostr_stopony_flushes_partial_word_at_row_boundary() {
    use crate::traits::BusDevice;

    const DM0_HOSTR_STOPONY: u32 = DRAWMODE0_OPCODE_READ | DRAWMODE0_ADRMODE_BLOCK_SH
        | DM0_STOPONXY | DM0_COLORHOST | DM0_DOSETUP;

    for &(w, h) in &[(12, 4), (20, 3), (4, 6), (33, 2)] {
        let per_row = ((w as usize) + 7) / 8;
        let nwords = per_row * h as usize;
        let want = counter_words(w, h);

        let rex = make_rex3();
        rex3init(rex);
        prefill_counter_image(rex, w, h);
        reg(rex, REX3_DRAWMODE1, DM1_CI8_HOSTRW64);
        reg(rex, REX3_XYENDI,   xy(w - 1, h - 1));
        reg(rex, REX3_XYSTARTI, xy(0, 0));
        reg(rex, REX3_DRAWMODE0, DM0_HOSTR_STOPONY);

        let mut back = vec![0u64; nwords];
        assert_eq!(rex.dma_read64_bulk(go_addr(REX3_HOSTRW0), &mut back),
                   crate::traits::BUS_OK, "{w}x{h}: bulk read refused");

        for k in 0..nwords {
            assert_eq!(back[k], want[k],
                "{w}x{h} word {k} (row {}, word {} of row):\n  got  {:016x}\n  want {:016x}\n\
                 a partial word carried into the next row skews the readback",
                k / per_row, k % per_row, back[k], want[k]);
        }
    }
}

/// The internal GFIFO sentinels must carry the GO bit and must be nameable.
///
/// Each sentinel's value already has bit 11 set, so `| 0x0800` was a no-op that
/// read as if it were setting it. That is easy to "clean up" into a value with
/// no GO bit, which would stop the token dispatching a primitive at all — the
/// transfer would vanish silently. This pins the bit to the value.
///
/// It also pins the names: these reach the bus log with the GO bit stripped, so
/// a missing arm prints `UNKNOWN`, indistinguishable from a real unmapped
/// register.
#[test]
fn test_gfifo_sentinels_carry_go_and_have_names() {
    for (name, tok, reg) in [
        ("REX3_DMA_PURE_GO", REX3_DMA_PURE_GO, REX3_DMA_PURE_GO_REG),
        ("REX3_DMA_BATCH_W", REX3_DMA_BATCH_W, REX3_DMA_BATCH_W_REG),
        ("REX3_DMA_BATCH_R", REX3_DMA_BATCH_R, REX3_DMA_BATCH_R_REG),
    ] {
        assert_ne!(tok & 0x0800, 0,
            "{name} = {tok:#06x} has no GO bit — its token would never run a primitive");
        assert_eq!(reg, tok & !0x0800, "{name}_REG must be the token minus the GO bit");
        assert_ne!(crate::rex3::rex3_reg_name(reg), "UNKNOWN",
            "{name}_REG ({reg:#06x}) has no name — the bus log cannot tell it from \
             a real unmapped register");
    }
    // The sentinels must not collide with each other or with a real register.
    let toks = [REX3_DMA_PURE_GO, REX3_DMA_BATCH_W, REX3_DMA_BATCH_R];
    for (i, a) in toks.iter().enumerate() {
        for b in &toks[i + 1..] {
            assert_ne!(a, b, "two sentinels share the value {a:#06x}");
        }
    }
}

/// A bulk transfer must leave something usable in the bus log.
///
/// The batch protocol replaces N `HOSTRW64` pushes with ONE token, and the
/// payload words are consumed inside `drain_payload` without ever reaching the
/// logger. So where the scalar path printed a line per word, the bulk path
/// printed a single opaque token line — exactly the path the wallpaper and
/// icons now take, and the one hardest to debug from a log.
///
/// This pins that both directions leave a line carrying the word count and a
/// data sample, so "the data arrived" can be told from "the data was wrong".
#[test]
#[cfg(feature = "developer")]
fn test_bulk_transfers_appear_in_the_bus_log() {
    use crate::traits::BusDevice;
    use std::io::Read;

    let dir = std::env::temp_dir().join(format!("iris-buslog-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("tmpdir");
    let prev = std::env::current_dir().expect("cwd");
    std::env::set_current_dir(&dir).expect("chdir");

    let rex = make_rex3();
    rex3init(rex);

    // `rex buslog on` — same path the monitor command takes.
    {
        let mut log = rex.rex3_log.lock();
        *log = Some(std::fs::File::create("rex3.log").expect("create log"));
    }

    let (w, h) = (8i32, 4i32);
    let words = counter_words(w, h);

    reg(rex, REX3_DRAWMODE1, DM1_CI8_HOSTRW64);
    reg(rex, REX3_WRMASK, 0xFF);
    reg(rex, REX3_XYENDI,   xy(w - 1, h - 1));
    reg(rex, REX3_XYSTARTI, xy(0, 0));
    reg(rex, REX3_DRAWMODE0, DM0_HOSTW_NO_STOPONY);
    assert_eq!(rex.dma_write64_bulk(go_addr(REX3_HOSTRW0), &words), crate::traits::BUS_OK);
    wait(rex);

    reg(rex, REX3_DRAWMODE1, DM1_CI8_HOSTRW64);
    reg(rex, REX3_XYENDI,   xy(w - 1, h - 1));
    reg(rex, REX3_XYSTARTI, xy(0, 0));
    reg(rex, REX3_DRAWMODE0, DM0_HOSTR_NO_STOPONY);
    let mut back = vec![0u64; words.len()];
    assert_eq!(rex.dma_read64_bulk(go_addr(REX3_HOSTRW0), &mut back), crate::traits::BUS_OK);

    { *rex.rex3_log.lock() = None; }

    let mut text = String::new();
    std::fs::File::open("rex3.log").expect("open log")
        .read_to_string(&mut text).expect("read log");
    std::env::set_current_dir(&prev).ok();
    let _ = std::fs::remove_dir_all(&dir);

    assert!(text.contains("DMA_BATCH_W"),
        "bulk write left no named token in the bus log:\n{text}");
    assert!(text.contains(&format!("batch write {} qwords", words.len())),
        "bulk write logged no word count:\n{text}");
    assert!(text.contains(&format!("first={:016x}", words[0])),
        "bulk write logged no data sample:\n{text}");
    assert!(text.contains(&format!("BATCH_R done: {} qwords", back.len())),
        "bulk read logged no result line — a readback's data never appears:\n{text}");
    assert!(text.contains(&format!("last={:016x}", back[back.len() - 1])),
        "bulk read result line carries no data sample:\n{text}");
}

/// JIT/interpreter equivalence for **batched** HOSTRW transfers.
///
/// The general rule for this emulator is that the two engines must be
/// indistinguishable. Batched transfers used to sidestep that rather than
/// satisfy it: `execute_go` forced `entry = None` whenever `host_len > 1`,
/// because the compiled shaders addressed `hostrw[0]` by fixed offset and never
/// stepped the cursor — so a multi-word transfer would consume one word and
/// repeat it for the whole run.
///
/// With the shader addressing `hostrw[hostrw_index()]` and advancing the cursor
/// on store, the bypass is no longer needed and these must agree pixel for
/// pixel and word for word.
#[cfg(feature = "rex-jit")]
mod batch_jit_equivalence {
    use super::*;
    use crate::traits::BusDevice;
    use super::jit_tests::make_rex3_jit;

    /// Drive one batched HOSTW transfer and return the painted region.
    fn run_hostw(rex: &Rex3, w: i32, h: i32, words: &[u64]) -> Vec<u32> {
        rex3init(rex);
        reg(rex, REX3_DRAWMODE1, DM1_CI8_HOSTRW64);
        reg(rex, REX3_WRMASK, 0xFF);
        reg(rex, REX3_XYENDI,   xy(w - 1, h - 1));
        reg(rex, REX3_XYSTARTI, xy(0, 0));
        reg(rex, REX3_DRAWMODE0, DM0_HOSTW_NO_STOPONY);
        assert_eq!(rex.dma_write64_bulk(go_addr(REX3_HOSTRW0), words),
                   crate::traits::BUS_OK);
        wait(rex);
        let mut out = Vec::new();
        for y in 0..h { for x in 0..w { out.push(read_pixel(rex, x, y) & 0xFF); } }
        out
    }

    /// Drive one batched HOSTR readback and return the words.
    fn run_hostr(rex: &Rex3, w: i32, h: i32, n: usize) -> Vec<u64> {
        prefill_counter_image(rex, w, h);
        reg(rex, REX3_DRAWMODE1, DM1_CI8_HOSTRW64);
        reg(rex, REX3_XYENDI,   xy(w - 1, h - 1));
        reg(rex, REX3_XYSTARTI, xy(0, 0));
        reg(rex, REX3_DRAWMODE0, DM0_HOSTR_NO_STOPONY);
        let mut back = vec![0u64; n];
        assert_eq!(rex.dma_read64_bulk(go_addr(REX3_HOSTRW0), &mut back),
                   crate::traits::BUS_OK);
        back
    }

    #[test]
    fn batched_hostw_matches_between_engines() {
        for &(w, h) in &[(8i32, 4i32), (16, 3), (12, 4), (20, 3), (33, 2)] {
            let words = counter_words(w, h);

            let rex_i = make_rex3();            // interpreter (jit_enabled = false)
            let interp = run_hostw(rex_i, w, h, &words);

            let rex_j = make_rex3_jit();
            rex_j.jit_enabled.store(true, std::sync::atomic::Ordering::Relaxed);
            // The first run only *requests* the compile — it happens on another
            // thread, so this run still executes on the interpreter. Wait for
            // the shader to exist before the run that is actually compared, or
            // the comparison is interpreter-vs-interpreter and passes with the
            // JIT arbitrarily broken.
            let _ = run_hostw(rex_j, w, h, &words);
            let cm = 0xF << CLIPMODE_CIDMATCH_SHIFT;
            // The dispatch key is the *normalized* dm1, the same one
            // execute_go and compile_shader use. Passing the raw value looks
            // up a key nothing was ever filed under.
            let dm1_key = crate::rex3_shape::normalize_dm1(
                DM1_CI8_HOSTRW64, DRAWMODE0_OPCODE_DRAW);
            if let Some(ref jit) = rex_j.rex_jit {
                assert!(jit.wait_compiled(DM0_HOSTW_NO_STOPONY, dm1_key, cm),
                    "{w}x{h}: JIT compile failed");
            }
            #[cfg(feature = "rexdiag")]
            let before = rex_j.jit_go_count.load(std::sync::atomic::Ordering::Relaxed);
            let jitted = run_hostw(rex_j, w, h, &words);
            // Without this the comparison is vacuous: if the batch never
            // reaches compiled code, both sides are the interpreter and the
            // test passes with the JIT arbitrarily broken.
            #[cfg(feature = "rexdiag")]
            {
                let after = rex_j.jit_go_count.load(std::sync::atomic::Ordering::Relaxed);
                assert!(after > before,
                    "{w}x{h}: no GO dispatched to compiled code — the batch is still \
                     bypassing the JIT, so this comparison proves nothing");
            }

            assert_eq!(jitted, interp,
                "{w}x{h}: JIT and interpreter disagree on a batched HOSTW");
            // And both must match the source pattern, so an identical-but-wrong
            // pair cannot pass.
            for (i, got) in interp.iter().enumerate() {
                assert_eq!(*got as u8, counter_byte(i),
                    "{w}x{h}: interpreter itself is wrong at pixel {i}");
            }
        }
    }

    #[test]
    fn batched_hostr_matches_between_engines() {
        for &(w, h) in &[(8i32, 4i32), (16, 3), (12, 4), (20, 3)] {
            let n = (((w as usize) + 7) / 8) * h as usize;
            let want = counter_words(w, h);

            let rex_i = make_rex3();
            rex3init(rex_i);
            let interp = run_hostr(rex_i, w, h, n);

            let rex_j = make_rex3_jit();
            rex_j.jit_enabled.store(true, std::sync::atomic::Ordering::Relaxed);
            rex3init(rex_j);
            let _ = run_hostr(rex_j, w, h, n);
            let cm = 0xF << CLIPMODE_CIDMATCH_SHIFT;
            let dm1_key = crate::rex3_shape::normalize_dm1(
                DM1_CI8_HOSTRW64, DRAWMODE0_OPCODE_READ);
            if let Some(ref jit) = rex_j.rex_jit {
                assert!(jit.wait_compiled(DM0_HOSTR_NO_STOPONY, dm1_key, cm),
                    "{w}x{h}: JIT compile failed");
            }
            rex3init(rex_j);
            #[cfg(feature = "rexdiag")]
            let before = rex_j.jit_go_count.load(std::sync::atomic::Ordering::Relaxed);
            let jitted = run_hostr(rex_j, w, h, n);
            #[cfg(feature = "rexdiag")]
            {
                let after = rex_j.jit_go_count.load(std::sync::atomic::Ordering::Relaxed);
                assert!(after >= before + n as u64,
                    "{w}x{h}: only {} GOs reached compiled code for {n} words — the \
                     batch is not being driven through the shader",
                    after - before);
            }

            for k in 0..n {
                assert_eq!(jitted[k], interp[k],
                    "{w}x{h} word {k}: JIT {:016x} != interpreter {:016x}",
                    jitted[k], interp[k]);
            }
            assert_eq!(interp, want, "{w}x{h}: interpreter readback itself is wrong");
        }
    }
}

/// The precompiled (LLVM) shader table must handle batches too.
///
/// In a release build `Rex3::new` seeds the shader map from
/// `rex3_shaders::SHADERS` (462 shapes). Those entries are found *before*
/// Cranelift is ever consulted, so in a real build a batch is served by an LLVM
/// shader — neither by the interpreter nor by the Cranelift JIT the
/// `batch_jit_equivalence` tests cover.
///
/// Tests normally start with an empty map (see `Rex3::new`, `#[cfg(test)]`) so
/// JIT-vs-generic comparisons genuinely exercise Cranelift. That also means
/// nothing else in the suite covers the seeded path.
///
/// The generated wrappers monomorphise `rex3_generic::draw_with_fb` and are
/// rebuilt with the crate, so they inherit `fetch_host_pixel`/`send_host_word`
/// and the cursor accessors automatically. What they do *not* inherit is the
/// batch resume loop in `execute_go` — a shader returns at a word boundary, so
/// the loop that drives it round for the remaining N-1 words has to exist on
/// the shader dispatch path. That is what this pins.
#[test]
fn precompiled_shaders_handle_batched_transfers() {
    use crate::traits::BusDevice;

    // A shape that is genuinely in the corpus, or seeding the map changes
    // nothing and the draw quietly falls through to the interpreter — which is
    // how the first version of this test passed while proving nothing.
    // dm0=0x0046 is DRAW BLOCK COLORHOST with neither STOPONX nor STOPONY;
    // dm1=0x30007589 is DM1_CI8_HOSTRW64; cm=0x1e00 is rex3init's CIDMATCH.
    const DM0_PRECOMPILED_HOSTW: u32 = 0x0046;
    let cm = 0xF << CLIPMODE_CIDMATCH_SHIFT;
    assert_eq!(cm, 0x1e00, "CIDMATCH default changed — re-check the corpus key");
    assert!(crate::rex3_shaders::SHADERS.iter().any(|(k, _)| {
        *k == (DM0_PRECOMPILED_HOSTW, DM1_CI8_HOSTRW64, cm)
    }), "dm0={DM0_PRECOMPILED_HOSTW:#06x} dm1={DM1_CI8_HOSTRW64:#010x} cm={cm:#06x} is not \
         in the generated corpus — this test would exercise the interpreter instead");

    let (w, h) = (8i32, 4i32);
    let words = counter_words(w, h);

    let setup = |rex: &Rex3| {
        rex3init(rex);
        reg(rex, REX3_DRAWMODE1, DM1_CI8_HOSTRW64);
        reg(rex, REX3_WRMASK, 0xFF);
        reg(rex, REX3_XYENDI,   xy(w - 1, h - 1));
        reg(rex, REX3_XYSTARTI, xy(0, 0));
        reg(rex, REX3_DRAWMODE0, DM0_PRECOMPILED_HOSTW);
    };

    // Reference: empty shader map, so this is the interpreter.
    let rex_i = make_rex3();
    setup(rex_i);
    assert_eq!(rex_i.dma_write64_bulk(go_addr(REX3_HOSTRW0), &words),
               crate::traits::BUS_OK);
    wait(rex_i);

    // Under test: seed the generated table, as a non-test build does.
    let rex_s = make_rex3();
    {
        let mut map = rex_s.shaders.write();
        for (k, f) in crate::rex3_shaders::SHADERS { map.insert(*k, *f); }
    }
    // `make_rex3` leaves the dispatch switch off so JIT tests start from a known
    // state; it gates the *whole* shader path, precompiled entries included. A
    // release build has it on.
    #[cfg(feature = "rex-jit")]
    rex_s.jit_enabled.store(true, std::sync::atomic::Ordering::Relaxed);
    setup(rex_s);
    #[cfg(feature = "rexdiag")]
    let before = rex_s.jit_go_count.load(std::sync::atomic::Ordering::Relaxed);
    assert_eq!(rex_s.dma_write64_bulk(go_addr(REX3_HOSTRW0), &words),
               crate::traits::BUS_OK);
    wait(rex_s);
    // Confirm a compiled shader really served it. Without this the seeded map
    // could miss and the comparison would be interpreter-vs-interpreter.
    #[cfg(feature = "rexdiag")]
    {
        let after = rex_s.jit_go_count.load(std::sync::atomic::Ordering::Relaxed);
        assert!(after >= before + words.len() as u64,
            "only {} GOs reached compiled code for {} words — the batch is not \
             being driven through the precompiled shader",
            after - before, words.len());
    }

    for y in 0..h {
        for x in 0..w {
            let i = y as usize * w as usize + x as usize;
            let want = counter_byte(i);
            let a = (read_pixel(rex_i, x, y) & 0xFF) as u8;
            let b = (read_pixel(rex_s, x, y) & 0xFF) as u8;
            assert_eq!(a, want, "interpreter wrong at pixel {i}");
            assert_eq!(b, want,
                "seeded-shader build wrong at ({x},{y}) pixel {i}: got {b:#04x} \
                 want {want:#04x} {} — a precompiled shader served this batch and \
                 did not walk the array", counter_explain(i, b));
        }
    }
}

