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
fn make_rex3() -> &'static Rex3 {
    let rex = Box::leak(Box::new(Rex3::new(
        Arc::new(AtomicU64::new(0)),
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
    rex.start();
    rex
}

// Compute the SET address (no GO bit) for a register offset.
fn set_addr(offset: u32) -> u32 { REX3_BASE | offset }
// Compute the GO address (bit 11 set) for a register offset.
fn go_addr(offset: u32) -> u32  { REX3_BASE | 0x0800 | offset }

/// Write a register to the SET space (no draw trigger).
fn reg(rex: &Rex3, offset: u32, value: u32) {
    rex.write32(set_addr(offset), value);
}

/// Write a register to the GO space (triggers a draw), then wait for idle.
/// Equivalent to writing to go.reg + REX3WAIT(REX) in SGI diagnostics.
fn reg_go(rex: &Rex3, offset: u32, value: u32) {
    rex.write32(go_addr(offset), value);
    rex.wait_idle();
}

/// Wait for the GFIFO processor to drain (equivalent to REX3WAIT).
fn wait(rex: &Rex3) {
    rex.wait_idle();
}

/// Read a 32-bit context register.  Blocks until the GFIFO is idle first.
fn read_reg(rex: &Rex3, offset: u32) -> u32 {
    rex.wait_idle();
    let r: BusRead32 = rex.read32(set_addr(offset));
    if r.is_ok() { r.data } else { panic!("read_reg: bad status for offset {offset:#x}") }
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
    rex.write32(go_addr(REX3_HOSTRW0), val);
}

/// Write a 64-bit double to HOSTRW0 (CPU→REX draw path, 64-bit GIO bus).
fn write_hostrw64(rex: &Rex3, val: u64) {
    rex.write64(go_addr(REX3_HOSTRW0), val);
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

// ============================================================================
// DRAWMODE constants (matching minigl3.c / SGI headers)
// ============================================================================

// DRAWMODE1 combinations
const DM1_CI8_SRC: u32   = DRAWMODE1_PLANES_RGB | (1 << 3) | DRAWMODE1_LOGICOP_SRC;
const DM1_RGB24_SRC: u32 = DRAWMODE1_PLANES_RGB | (3 << 3) | (1 << 15) | DRAWMODE1_LOGICOP_SRC;

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

const DM0_DRAW_BLOCK:  u32 = DRAWMODE0_OPCODE_DRAW | DRAWMODE0_ADRMODE_BLOCK | DM0_STOPONXY;
const DM0_DRAW_SPAN:   u32 = DRAWMODE0_OPCODE_DRAW | DRAWMODE0_ADRMODE_SPAN  | DM0_STOPONX;
const DM0_SCR2SCR:     u32 = DRAWMODE0_OPCODE_SCR2SCR | DRAWMODE0_ADRMODE_BLOCK | DM0_DOSETUP | DM0_STOPONXY;
// DRAW with COLORHOST: pixels come from host write FIFO
const DM0_HOSTW_BLOCK: u32 = DRAWMODE0_OPCODE_DRAW | DRAWMODE0_ADRMODE_BLOCK | DM0_STOPONXY | DM0_COLORHOST;
// READ with COLORHOST: reads fb → host read FIFO
const DM0_READ_BLOCK:  u32 = DRAWMODE0_OPCODE_READ | DRAWMODE0_ADRMODE_BLOCK | DM0_STOPONXY | DM0_COLORHOST | DM0_DOSETUP;

/// Initialise REX3 to a known baseline — matches rex3init() from rex3.c.
/// XYWIN is left at 0 (no hardware xbias correction needed in emulation).
/// clipmode=0 means no smask checking, no CID checking.
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
    reg(rex, REX3_CLIPMODE,    0);  // no CID, no smask checking
    wait(rex);
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
    let dm1_zero = DRAWMODE1_PLANES_RGB | (1 << 3) | DRAWMODE1_LOGICOP_ZERO;
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
    let dm1_xor = DRAWMODE1_PLANES_RGB | (1 << 3) | DRAWMODE1_LOGICOP_XOR;

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
    let dm0_noop = DRAWMODE0_OPCODE_NOOP | DRAWMODE0_ADRMODE_BLOCK | DM0_STOPONXY;
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
    DRAWMODE1_PLANES_RGB | (1 << 3) | DRAWMODE1_LOGICOP_SRC | (1 << 8) | (1 << 7);
// CI8 64-bit: same as CI8 + rwdouble (bit 10) → 8 CI8 pixels per 64-bit word
const DM1_CI8_HOSTRW64: u32 = DM1_CI8_HOSTRW | (1 << 10);
// RGB24: hostdepth=3 (32bpp), rwpacked (bit 7), same draw-plane as DM1_RGB24_SRC
const DM1_RGB24_HOSTRW: u32 =
    DRAWMODE1_PLANES_RGB | (3 << 3) | (1 << 15) | DRAWMODE1_LOGICOP_SRC | (3 << 8) | (1 << 7);
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
        rex.write32(set_addr(REX3_HOSTRW0), w);
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
        rex.write32(set_addr(REX3_HOSTRW1), w);
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
const DM0_DRAW_ILINE: u32 = DRAWMODE0_OPCODE_DRAW | DRAWMODE0_ADRMODE_I_LINE | DM0_DOSETUP | DM0_STOPONXY;
// DM0 for I_LINE single-step mode (no stoponx/stopony — one pixel per GO).
const DM0_DRAW_ILINE_STEP: u32 = DRAWMODE0_OPCODE_DRAW | DRAWMODE0_ADRMODE_I_LINE | DM0_DOSETUP;

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
    let dm0_cont = DRAWMODE0_OPCODE_DRAW | DRAWMODE0_ADRMODE_I_LINE;
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
        let dm0_cont = DRAWMODE0_OPCODE_DRAW | DRAWMODE0_ADRMODE_I_LINE;
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
    let dm0_loop: u32 = DRAWMODE0_OPCODE_DRAW | DRAWMODE0_ADRMODE_I_LINE
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

    /// Build a Rex3 with JIT enabled.
    fn make_rex3_jit() -> &'static Rex3 {
        let rex = Box::leak(Box::new(Rex3::new(
            Arc::new(AtomicU64::new(0)),
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
        rex.rex_jit = Some(std::sync::Arc::new(RexJit::new()));
        rex.start();
        rex
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
        // Interpreter run (no JIT — rex_jit is None)
        let rex_interp = make_rex3();
        rex3init(rex_interp);
        setup(rex_interp);
        reg_go(rex_interp, REX3_DRAWMODE0, dm0);
        let fb_interp = dump_region(rex_interp, x0, y0, x1, y1);

        // JIT run
        let rex_jit = make_rex3_jit();
        rex3init(rex_jit);
        setup(rex_jit);
        // First GO: triggers compile + interpreter fallback
        reg_go(rex_jit, REX3_DRAWMODE0, dm0);
        // Wait for JIT compile
        let compiled = if let Some(ref jit) = rex_jit.rex_jit {
            jit.wait_compiled(dm0, dm1)
        } else { false };
        assert!(compiled, "JIT compile failed for dm0={dm0:#010x} dm1={dm1:#010x}");

        // Reset fb and re-run via JIT
        clear_region(rex_jit, x0, y0, x1, y1);
        rex3init(rex_jit);
        setup(rex_jit);
        reg_go(rex_jit, REX3_DRAWMODE0, dm0);
        let fb_jit = dump_region(rex_jit, x0, y0, x1, y1);

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
        let dm1 = DRAWMODE1_PLANES_RGB | (3 << 3) | (1 << 15) | DRAWMODE1_LOGICOP_XOR;
        let dm0 = DM0_DRAW_BLOCK;
        compare_jit_interp(0, 0, 15, 15,
            |rex| {
                reg(rex, REX3_DRAWMODE1, dm1);
                reg(rex, REX3_WRMASK,    0xFFFFFF);
                reg(rex, REX3_COLORRED,  0x00_FF_00_FFu32);
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
        let dm1 = DRAWMODE1_PLANES_RGB | (3 << 3) | (1 << 15) | DRAWMODE1_LOGICOP_SRC | (1 << 17);
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
                reg(rex, REX3_COLORRED,  0x00_FF_80_40u32);
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
}
