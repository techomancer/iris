//! Whole-board tests through the bus, with the HQ3 and RSS threads running:
//! command FIFO and direct-window drawing and their ordering, PIO reads,
//! host DMA both ways, flags, the context switch, the GE11 diagnostic port,
//! and replay of text recordings (`testdata/*.txt`).

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;

use super::record::tests::{live_board, read, replay_recs, write};
use super::record::load_text;
use super::Mgras;
use crate::traits::{BusDevice, BusRead32, BusRead8, BUS_OK};

const CFIFO: u32 = 0x70080;
const RASTER: u32 = 0x7C000;
const EXEC: u32 = 0x1000;
const FLAGS: u32 = 0x70008;

// Raster registers (rss.rs `reg`).
const IR_ALIAS: u32 = 0x045;
const BLOCKXYSTARTI: u32 = 0x046;
const BLOCKXYENDI: u32 = 0x047;
const XFRCONTROL: u32 = 0x102;
const FILLMODE: u32 = 0x110;
const CONFIG: u32 = 0x112;
const XYWIN: u32 = 0x115;
const DRBPOINTERS: u32 = 0x16D;
const DRBSIZE: u32 = 0x16E;
const XFRSIZE: u32 = 0x153;
const XFRMODE: u32 = 0x159;
const PP1FILLMODE: u32 = 0x161;
const PP1WINMODE: u32 = 0x17B;
const FILL_COLOR_R: u32 = 0x176;
const FILL_FAST: u32 = 1 << 20;

/// A raster register write through the command FIFO.
fn fifo_rss(m: &Mgras, r: u32, val: u32, exec: bool) {
    let cmd = 0x1000 | r | if exec { 0x400 } else { 0 };
    write(m, 32, CFIFO, ((cmd << 8) | 4) as u64);
    write(m, 32, CFIFO, val as u64);
}

/// A raster register write through the direct window.
fn direct_rss(m: &Mgras, r: u32, val: u32, exec: bool) {
    write(m, 32, RASTER + 4 * r + if exec { EXEC } else { 0 }, val as u64);
}

/// A host DMA engine register write through the command FIFO.
fn fifo_dma(m: &Mgras, n: u32, val: u32) {
    write(m, 32, CFIFO, (((0x800 | n) << 8) | 4) as u64);
    write(m, 32, CFIFO, val as u64);
}

/// The X server's raster setup: origin at the top row, Y flipped, colour
/// index drawing.
fn x_server(m: &Mgras) {
    direct_rss(m, CONFIG, 0xCAC, false);
    direct_rss(m, XYWIN, 1023 << 16, false);
    // The PROM's 1280x1024 layout: draw into the main buffer at page 0x240.
    direct_rss(m, DRBSIZE, 0x31E, false);
    direct_rss(m, DRBPOINTERS, 0x240, false);
    // All planes writable, as the X server sets them.
    direct_rss(m, 0x163, u32::MAX, false);
    direct_rss(m, PP1FILLMODE, 0x0C00_4504, false);
}

fn block(m: &Mgras, x0: u32, y0: u32, x1: u32, y1: u32) {
    direct_rss(m, IR_ALIAS, 0x18, false);
    direct_rss(m, BLOCKXYSTARTI, x0 << 16 | y0, false);
    direct_rss(m, BLOCKXYENDI, x1 << 16 | y1, true);
}

/// Flags, polled until `bit` is set (the HQ3 raises them asynchronously).
fn wait_flag(m: &Mgras, bit: u32) -> u32 {
    for _ in 0..1_000_000 {
        let f = read(m, 32, FLAGS) as u32;
        if f & bit != 0 {
            return f;
        }
        std::thread::yield_now();
    }
    panic!("flag {bit:#x} never set");
}

#[test]
fn fifo_fast_fill_lands_top_down() {
    let m = live_board();
    x_server(&m);
    fifo_rss(&m, FILLMODE, FILL_FAST, false);
    fifo_rss(&m, FILL_COLOR_R, 0x13, false);
    fifo_rss(&m, IR_ALIAS, 0x18, false);
    fifo_rss(&m, BLOCKXYSTARTI, 10 << 16 | 20, false);
    fifo_rss(&m, BLOCKXYENDI, 12 << 16 | 21, true);
    assert_eq!(m.fb_pixel(10, 20), 0x13);
    assert_eq!(m.fb_pixel(12, 21), 0x13);
    assert_eq!(m.fb_pixel(13, 21), 0);
    m.stop_engines();
}

#[test]
fn fifo_and_direct_writes_keep_program_order() {
    let m = live_board();
    x_server(&m);
    direct_rss(&m, FILLMODE, FILL_FAST, false);
    // FIFO colour, then a direct colour, then a FIFO-executed block: the
    // direct write came last, so it wins.
    fifo_rss(&m, FILL_COLOR_R, 3, false);
    direct_rss(&m, FILL_COLOR_R, 5, false);
    fifo_rss(&m, IR_ALIAS, 0x18, false);
    fifo_rss(&m, BLOCKXYSTARTI, 1 << 16 | 1, false);
    fifo_rss(&m, BLOCKXYENDI, 1 << 16 | 1, true);
    // And the other way round, executed through the direct window.
    direct_rss(&m, FILL_COLOR_R, 7, false);
    fifo_rss(&m, FILL_COLOR_R, 9, false);
    block(&m, 2, 1, 2, 1);
    assert_eq!(m.fb_pixel(1, 1), 5);
    assert_eq!(m.fb_pixel(2, 1), 9);
    m.stop_engines();
}

#[test]
fn pio_read_through_the_bus() {
    let m = live_board();
    x_server(&m);
    direct_rss(&m, FILLMODE, FILL_FAST, false);
    for (k, v) in [1u32, 2, 3, 4, 5, 6].into_iter().enumerate() {
        let (x, y) = (100 + k as u32 % 3, 50 + k as u32 / 3);
        direct_rss(&m, FILL_COLOR_R, v, false);
        block(&m, x, y, x, y);
    }
    // Three 1-byte pixels a line, two lines, begin skip 2 (rss.rs
    // pio_read_frames_each_line_in_a_fresh_doubleword).
    direct_rss(&m, FILLMODE, 2 << 22, false);
    direct_rss(&m, XFRMODE, 2 << 8, false);
    direct_rss(&m, XFRSIZE, 2 << 16 | 3, false);
    block(&m, 100, 50, 102, 51);
    let mut dw = || (read(&m, 32, 0x7D1C0) << 32) | read(&m, 32, 0x7C1C4);
    assert_eq!(dw(), 0x0000_0102_0300_0000);
    assert_eq!(dw(), 0x0000_0000_0004_0506);
    assert_eq!(dw(), 0);
    m.stop_engines();
}

/// System memory for DMA tests: page-table words and pixel bytes.
#[derive(Default)]
struct TestMem {
    words: Mutex<HashMap<u32, u32>>,
    bytes: Mutex<HashMap<u32, u8>>,
}

impl BusDevice for TestMem {
    fn read8(&self, addr: u32) -> BusRead8 { BusRead8::ok(self.bytes.lock().get(&addr).copied().unwrap_or(0)) }
    fn write8(&self, addr: u32, val: u8) -> u32 { self.bytes.lock().insert(addr, val); BUS_OK }
    fn read32(&self, addr: u32) -> BusRead32 { BusRead32::ok(self.words.lock().get(&addr).copied().unwrap_or(0)) }
}

/// Memory with pool 0's page table at 0x1000 mapping logical page 0 to
/// frame 2 (0x2000), and the board pointed at it: two lines of three bytes,
/// four bytes apart, from logical 0.
fn dma_setup(m: &Mgras) -> Arc<TestMem> {
    let mem = Arc::new(TestMem::default());
    mem.words.lock().insert(0x1000, 2);
    m.set_phys(mem.clone());
    fifo_dma(m, 0x21, 0x1000); // pool 0 table base (low word)
    fifo_dma(m, 0x06, 0); // row start
    fifo_dma(m, 0x05, 0); // row offset
    fifo_dma(m, 0x04, 4); // stride
    fifo_dma(m, 0x07, 2); // lines
    fifo_dma(m, 0x08, 3); // bytes a line
    mem
}

#[test]
fn dma_write_from_host_memory() {
    let m = live_board();
    x_server(&m);
    let mem = dma_setup(&m);
    for (a, b) in [(0x2000, 1u8), (0x2001, 2), (0x2002, 3), (0x2004, 4), (0x2005, 5), (0x2006, 6)] {
        mem.bytes.lock().insert(a, b);
    }
    direct_rss(&m, FILLMODE, 5 << 22, false);
    direct_rss(&m, XFRMODE, 0, false);
    direct_rss(&m, XFRSIZE, 2 << 16 | 3, false);
    block(&m, 10, 10, 12, 11);
    fifo_dma(&m, 0x0B, 1); // start: run, pool 0, host to board
    let got: Vec<u32> = [(10, 10), (11, 10), (12, 10), (10, 11), (11, 11), (12, 11)].iter().map(|&(x, y)| m.fb_pixel(x, y)).collect();
    assert_eq!(got, [1, 2, 3, 4, 5, 6]);
    m.stop_engines();
}

#[test]
fn dma_read_to_host_memory() {
    let m = live_board();
    x_server(&m);
    direct_rss(&m, FILLMODE, FILL_FAST, false);
    for (k, v) in [7u32, 8, 9, 10, 11, 12].into_iter().enumerate() {
        let (x, y) = (20 + k as u32 % 3, 30 + k as u32 / 3);
        direct_rss(&m, FILL_COLOR_R, v, false);
        block(&m, x, y, x, y);
    }
    let mem = dma_setup(&m);
    direct_rss(&m, FILLMODE, 4 << 22, false);
    direct_rss(&m, XFRMODE, 0, false);
    direct_rss(&m, XFRSIZE, 2 << 16 | 3, false);
    block(&m, 20, 30, 22, 31);
    // The host side starts first (board to host), then the raster engine.
    fifo_dma(&m, 0x0B, 0x9);
    fifo_rss(&m, XFRCONTROL, 9, false);
    m.state_hash(); // wait for the board
    let b = mem.bytes.lock();
    let got: Vec<u8> = [0x2000, 0x2001, 0x2002, 0x2004, 0x2005, 0x2006].iter().map(|a| b.get(a).copied().unwrap_or(0)).collect();
    assert_eq!(got, [7, 8, 9, 10, 11, 12]);
    drop(b);
    m.stop_engines();
}

#[test]
fn set_done_command_raises_the_done_flag() {
    let m = live_board();
    write(&m, 32, CFIFO, (0xE04 << 8) as u64);
    assert_ne!(wait_flag(&m, 1 << 16) & 1 << 16, 0);
    // The host clears it.
    write(&m, 32, 0x7000C, 1 << 16);
    assert_eq!(read(&m, 32, FLAGS) as u32 & 1 << 16, 0);
    m.stop_engines();
}

#[test]
fn context_switch_swallows_the_incoming_context() {
    let m = live_board();
    x_server(&m);
    write(&m, 32, 0x50050, 0x100);
    assert_ne!(read(&m, 32, FLAGS) as u32 & 1 << 19, 0, "saved at once");
    // 63 words of context that would parse as commands if not swallowed:
    // each a raster write header announcing four data bytes.
    for _ in 0..63 {
        write(&m, 32, CFIFO, ((0x1000 | FILL_COLOR_R) << 8 | 4) as u64);
    }
    wait_flag(&m, 1 << 6);
    fifo_rss(&m, FILLMODE, FILL_FAST, false);
    fifo_rss(&m, FILL_COLOR_R, 0x2A, false);
    block(&m, 5, 5, 5, 5);
    assert_eq!(m.fb_pixel(5, 5), 0x2A);
    m.stop_engines();
}

#[test]
fn ge11_microcode_reads_back_as_the_driver_verifies_it() {
    let m = live_board();
    let (data, addr) = (0x50040, 0x50044);
    write(&m, 32, addr, 0x20_0000);
    for w in [0x1111_1111u32, 0x2222_2222, 0x0000_0033, 0x4444_4444, 0x5555_5555, 0x0000_0066] {
        write(&m, 32, data, w as u64);
    }
    write(&m, 32, addr, 0x20_0000);
    write(&m, 32, addr, 0x8000_0002); // read two lines
    assert_ne!(read(&m, 32, FLAGS) as u32 & 1 << 18, 0, "readback waiting");
    let words: Vec<u32> = (0..3).map(|_| read(&m, 32, 0x5022C) as u32).collect();
    assert_eq!(words, [0x1111_1111, 0x33, 0x5555_5555]);
    assert_eq!(read(&m, 32, FLAGS) as u32 & 1 << 18, 0, "drained");
    m.stop_engines();
}

/// A text recording replays through the same engine as the goldens.
#[test]
fn text_recording_replays() {
    let recs = load_text(
        "# X-style setup, then a fast fill of (300, 400)-(301, 400) in index 0x44
         W 32 7c448 cac
         W 32 7c454 3ff0000
         W 32 7c584 c004504
         W 32 7c5b8 31e
         W 32 7c5b4 240
         W 32 7c58c ffffffff
         W 32 7c440 100000
         W 32 7c5d8 44
         W 32 7c114 18
         W 32 7c118 12c0190
         W 32 7d11c 12d0190
         T",
    )
    .unwrap();
    let m = live_board();
    let rep = replay_recs(&m, &recs, false).unwrap();
    assert!(!rep.diverged());
    assert_eq!((m.fb_pixel(300, 400), m.fb_pixel(301, 400), m.fb_pixel(302, 400)), (0x44, 0x44, 0));
    m.stop_engines();
}

/// A real capture (testdata/xsetroot_red.txt) replayed on a fresh board:
/// the exposed root takes colour index 1 (red in the IRIS default "4sight"
/// colormap, loaded long before the capture), and the Console window, not
/// exposed, keeps what it had.
#[test]
fn xsetroot_red_trace() {
    let recs = load_text(include_str!("testdata/xsetroot_red.txt")).unwrap();
    let m = live_board();
    let rep = replay_recs(&m, &recs, false).unwrap();
    assert!(!rep.diverged(), "{rep:?}");
    let root = m.fb_pixel(640, 900);
    assert_eq!(root, 1);
    for (x, y) in [(100, 500), (1000, 200), (500, 50), (1270, 1010), (150, 250)] {
        assert_eq!(m.fb_pixel(x, y), root, "root at ({x}, {y})");
    }
    assert_eq!(m.fb_pixel(500, 500), 0, "Console interior not exposed");
    m.stop_engines();
}

/// Window-ID runs: no run at x = 0 starts in ID 0; each run lasts to the
/// next one; the overlay ID picks the overlay mode.
#[test]
fn frame_composes_runs_overlay_and_cursor() {
    use super::frame::{Frame, MainMode, OverlayMode, H, OUT_STRIDE, W};
    let mut f = super::plain::boxed_zeroed::<Frame>();
    (f.width, f.height) = (W, H);
    for (i, g) in f.gamma.iter_mut().enumerate() {
        *g = [i as u8; 3];
    }
    f.cmap[0x100 + 5] = 0x11_2233;
    f.cmap[0x200 + 7] = 0x44_5566;
    f.main_mode[3] = MainMode { rgb: false, cmap_base: 0x100 };
    f.main_mode[4] = MainMode { rgb: true, cmap_base: 0 };
    f.overlay_mode[2] = OverlayMode { on: true, cmap_base: 0x200 };
    f.did_main[10] = 3;
    f.main[10] = 5;
    f.did_main[11] = 4;
    f.main[11] = 0x00_8040_20;
    f.did_overlay[12] = 2;
    f.overlay[12] = 7;
    f.did_main[12] = 4;
    f.main[12] = 0x00FF_FFFF;
    let mut out = vec![0u32; OUT_STRIDE * H];
    f.compose(&mut out);
    assert_eq!(out[10], 0xFF33_2211, "CI through cmap block, red in the low byte");
    assert_eq!(out[11], 0xFF80_4020 & 0xFFFF_FFFF, "RGB as stored");
    assert_eq!(out[12], 0xFF66_5544, "overlay over the main planes");
}

/// Probe: the display size a recording's final state programs into the VC3.
/// MGRAS_SIZE_PROBE=<rec> cargo test --release mgras_size_probe -- --nocapture
#[test]
fn mgras_size_probe() {
    let Ok(path) = std::env::var("MGRAS_SIZE_PROBE") else { return };
    let recs = super::record::load(&path).unwrap();
    let m = live_board();
    replay_recs(&m, &recs, false).unwrap();
    let (t, d) = m.vc3_sizes();
    eprintln!("{path}: timing tables {t:?}, DID frame table {d} lines");
    m.stop_engines();
}

/// A command-processor token and its data words through the command FIFO.
fn fifo_token(m: &Mgras, token: u32, data: &[u32]) {
    write(m, 32, CFIFO, ((token << 8) | (4 * data.len() as u32)) as u64);
    for d in data {
        write(m, 32, CFIFO, *d as u64);
    }
}

fn f(v: f32) -> u32 {
    v.to_bits()
}

/// The glprim triangle (window at the screen's lower left), shade model
/// `shade` (GL_FLAT 0x1D00, GL_SMOOTH 0x1D01).
fn gl_glprim_triangle(m: &Mgras, shade: u32) {
    x_server(m);
    let mut win = vec![0u32; 15];
    win[1] = 0x11;
    (win[9], win[10]) = (399, 299);
    win[11] = 0x240;
    fifo_token(m, 0xE4, &win);
    fifo_token(m, 0x33, &[0, 0, 400, 300]);
    fifo_token(m, 0x2A, &[0x1701]);
    fifo_token(m, 0x2C, &[]);
    fifo_token(m, 0x35, &[f(0.0), f(400.0), f(0.0), f(300.0), f(-1.0), f(1.0)]);
    fifo_token(m, 0x2A, &[0x1700]);
    fifo_token(m, 0x2C, &[]);
    fifo_token(m, 0x47, &[shade]);
    fifo_token(m, 0xBA, &[f(1.0), f(0.5), f(0.75), f(1.0)]);
    fifo_token(m, 0x15, &[]);
    fifo_token(m, 0x1A, &[]);
    for (c, v) in [([1.0, 0.0, 0.0], [80.0, 60.0]), ([0.0, 1.0, 0.0], [320.0, 60.0]), ([0.0, 0.0, 1.0], [200.0, 240.0])] {
        fifo_token(m, 0x02, &[f(c[0]), f(c[1]), f(c[2])]);
        fifo_token(m, 0x00, &[f(v[0]), f(v[1]), f(0.0)]);
    }
    fifo_token(m, 0x24, &[]);
}

/// Smooth shading interpolates the vertex colours across the triangle:
/// near each vertex its colour dominates; the centroid is an even mix.
#[test]
fn gl_smooth_triangle_interpolates() {
    let m = live_board();
    gl_glprim_triangle(&m, 0x1D01);
    let at = |x: usize, y_gl: usize| m.fb_pixel(x, 1023 - y_gl);
    let rgb = |v: u32| [v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF];
    let near_red = rgb(at(82, 60));
    let near_green = rgb(at(317, 60));
    let near_blue = rgb(at(200, 237));
    assert!(near_red[0] > 245 && near_red[1] < 10, "{near_red:?}");
    assert!(near_green[1] > 245 && near_green[0] < 10, "{near_green:?}");
    assert!(near_blue[2] > 240, "{near_blue:?}");
    let mid = rgb(at(200, 120));
    for c in mid {
        assert!((75..=95).contains(&c), "centroid {mid:?}");
    }
    m.stop_engines();
}

/// glprim's default frame as libGLcore sends it (IRIX 6.5.22, traced): a
/// 400x300 window, ortho projection, pink clear, then one flat-shaded
/// triangle (80,60) red, (320,60) green, (200,240) blue. Flat shading takes
/// the last vertex's colour; the window is placed by the kernel's window
/// token (here at the screen's lower left, drawing buffer page 0x240).
#[test]
fn gl_flat_triangle_and_clear() {
    let m = live_board();
    x_server(&m);
    // Window: origin (0, 0) bottom-up, mask 1 = the window, kept inside.
    let mut win = vec![0u32; 15];
    win[0] = 0;
    win[1] = 0x11;
    (win[9], win[10]) = (399, 299);
    win[11] = 0x240;
    fifo_token(&m, 0xE4, &win);
    fifo_token(&m, 0x33, &[0, 0, 400, 300]);
    fifo_token(&m, 0x2A, &[0x1701]);
    fifo_token(&m, 0x2C, &[]);
    fifo_token(&m, 0x35, &[f(0.0), f(400.0), f(0.0), f(300.0), f(-1.0), f(1.0)]);
    fifo_token(&m, 0x2A, &[0x1700]);
    fifo_token(&m, 0x2C, &[]);
    fifo_token(&m, 0x47, &[0x1D00]);
    fifo_token(&m, 0xBA, &[f(1.0), f(0.5), f(0.75), f(1.0)]);
    fifo_token(&m, 0x15, &[]);
    fifo_token(&m, 0x1A, &[]);
    for (c, v) in [([1.0, 0.0, 0.0], [80.0, 60.0]), ([0.0, 1.0, 0.0], [320.0, 60.0]), ([0.0, 0.0, 1.0], [200.0, 240.0])] {
        fifo_token(&m, 0x02, &[f(c[0]), f(c[1]), f(c[2])]);
        fifo_token(&m, 0x00, &[f(v[0]), f(v[1]), f(0.0)]);
    }
    fifo_token(&m, 0x24, &[]);
    // Display rows count down from the top of the 1024-line screen.
    let at = |x: usize, y_gl: usize| m.fb_pixel(x, 1023 - y_gl);
    let pink = 0xBF_80FF;
    assert_eq!(at(5, 5), pink, "clear inside the window");
    assert_eq!(at(399, 299), pink);
    assert_eq!(at(400, 5), 0, "nothing outside the window");
    assert_eq!(at(200, 100), 0xFF_0000, "flat triangle in the last vertex's colour (blue)");
    assert_eq!(at(81, 60), 0xFF_0000, "bottom edge row is in");
    assert_eq!(at(79, 60), pink, "left of the triangle");
    assert_eq!(at(200, 240), pink, "the apex row is out (half-open)");
    m.stop_engines();
}

/// Switch to a context whose image carries `id` (word 0) and a GL window
/// at (`x`, `y`) bottom-up, `w` x `h`, mask 1 = the window, kept inside.
/// GL context ERAM slots (image word 0) as IRIX 6.5.22 hands them out.
const SLOT_A: u32 = 0xA9B;
const SLOT_B: u32 = 0xA9B + 0x17C7;
const SLOT_C: u32 = 0xA9B + 2 * 0x17C7;
const SLOT_D: u32 = 0xA9B + 3 * 0x17C7;

fn switch_to_gl_context(m: &Mgras, id: u32, x: u32, y: u32, w: u32, h: u32) {
    switch_to_gl_context_cid(m, id, x, y, w, h, 0);
}

/// The same, the window's PP1 window mode (image word 4) `pp1winmode`.
fn switch_to_gl_context_cid(m: &Mgras, id: u32, x: u32, y: u32, w: u32, h: u32, pp1winmode: u32) {
    write(m, 32, 0x50050, 0x4FC);
    let mut img = [0u32; 63];
    img[0] = id;
    img[2] = x | y << 16;
    img[3] = 0x11;
    img[4] = pp1winmode;
    img[11] = x << 16 | (x + w - 1);
    img[12] = y << 16 | (y + h - 1);
    img[13] = 0x240;
    // The banks drawn into, B and B, as the kernel stores them in a new
    // context (MgrasValidateBanks).
    (img[16], img[17]) = (1, 1);
    for wd in img {
        write(m, 32, CFIFO, wd as u64);
    }
    wait_flag(m, 1 << 6);
}

fn gl_setup_window(m: &Mgras, w: u32, h: u32, clear: [f32; 3]) {
    fifo_token(m, 0x33, &[0, 0, w, h]);
    fifo_token(m, 0xBA, &[f(clear[0]), f(clear[1]), f(clear[2]), f(1.0)]);
}

/// Two GL contexts, each with its own window and clear colour, switched in
/// and out: each clear lands in its own window in its own colour (window
/// and GL state belong to the context, not the board), and the X server's
/// raster registers, even ones it wrote through the direct window, are
/// back afterwards.
#[test]
fn gl_state_and_window_follow_the_context() {
    let m = live_board();
    x_server(&m);
    switch_to_gl_context(&m, SLOT_A, 0, 0, 100, 100);
    gl_setup_window(&m, 100, 100, [1.0, 0.0, 0.0]);
    switch_to_gl_context(&m, SLOT_B, 200, 0, 100, 100);
    gl_setup_window(&m, 100, 100, [0.0, 0.0, 1.0]);
    switch_to_gl_context(&m, SLOT_A, 0, 0, 100, 100);
    fifo_token(&m, 0x15, &[]);
    switch_to_gl_context(&m, SLOT_B, 200, 0, 100, 100);
    fifo_token(&m, 0x15, &[]);
    let at = |x: usize, y_gl: usize| m.fb_pixel(x, 1023 - y_gl);
    assert_eq!(at(50, 50), 0x00_00FF, "context A clears its window red");
    assert_eq!(at(250, 50), 0xFF_0000, "context B clears its window blue");
    assert_eq!(at(150, 50), 0, "between the windows");
    // The X server's state survives, including CONFIG written directly.
    direct_rss(&m, CONFIG, 0xCAC, false);
    fifo_token(&m, 0x15, &[]);
    fifo_rss(&m, FILLMODE, FILL_FAST, false);
    fifo_rss(&m, FILL_COLOR_R, 0x2A, false);
    block(&m, 5, 5, 5, 5);
    assert_eq!(m.fb_pixel(5, 5), 0x2A, "X draws top-down at its own origin again");
    m.stop_engines();
}

/// A window whose visible region is too complex for the screen masks: the
/// X server paints it with a clip ID (mgrasDrawCID: draw field 0x50, the ID
/// in the fill colour, pp1winmode 0xC00, left set), and the kernel has the
/// window's GL drawing match that ID (pp1winmode 1 << (4 + id)). The clear
/// lands only on the window's pixels with the ID; a window without one, and
/// the X server, draw everywhere.
#[test]
fn gl_draws_only_where_the_clip_id_matches() {
    let m = live_board();
    x_server(&m);
    let x_fillmode = 0x0C00_4504;
    direct_rss(&m, PP1WINMODE, 0, false);
    // Clip ID 0 over the screen, then 1 over the window's left half
    // (screen rows top-down: GL rows 0..99 are 924..1023).
    direct_rss(&m, PP1WINMODE, 0xC00, false);
    direct_rss(&m, PP1FILLMODE, 0x14_2600, false);
    fifo_rss(&m, FILLMODE, FILL_FAST, false);
    fifo_rss(&m, FILL_COLOR_R, 0, false);
    block(&m, 0, 0, 1279, 1023);
    fifo_rss(&m, FILL_COLOR_R, 1, false);
    block(&m, 0, 924, 49, 1023);
    direct_rss(&m, PP1FILLMODE, x_fillmode, false);
    assert_eq!(m.fb_pixel(10, 1000), 0, "clip IDs are not colour");

    switch_to_gl_context_cid(&m, SLOT_A, 0, 0, 100, 100, 0x20);
    gl_setup_window(&m, 100, 100, [1.0, 0.0, 0.0]);
    fifo_token(&m, 0x15, &[]);
    let at = |x: usize, y_gl: usize| m.fb_pixel(x, 1023 - y_gl) & 0xFF_FFFF;
    assert_eq!(at(10, 50), 0x00_00FF, "clip ID 1: the window's own pixels");
    assert_eq!(at(49, 99), 0x00_00FF);
    assert_eq!(at(50, 50), 0, "clip ID 0: another window's pixels");

    switch_to_gl_context_cid(&m, SLOT_B, 0, 0, 100, 100, 0);
    gl_setup_window(&m, 100, 100, [0.0, 0.0, 1.0]);
    fifo_token(&m, 0x15, &[]);
    assert_eq!(at(50, 50), 0xFF_0000, "no clip ID to match: all of it");

    // The X server draws anywhere with its pp1winmode back.
    fifo_rss(&m, FILLMODE, FILL_FAST, false);
    fifo_rss(&m, FILL_COLOR_R, 0x2A, false);
    block(&m, 60, 1000, 60, 1000);
    assert_eq!(m.fb_pixel(60, 1000), 0x2A);
    m.stop_engines();
}

/// A GL context in a 400x300 window at the screen's lower left with an
/// ortho projection (z = -1 near .. 1 far after glOrtho's flip).
fn gl_window_400x300(m: &Mgras) {
    switch_to_gl_context(m, 0xC, 0, 0, 400, 300);
    gl_ortho_400x300(m);
}

/// The current context's 400x300 viewport, ortho projection, flat shading.
fn gl_ortho_400x300(m: &Mgras) {
    fifo_token(m, 0x33, &[0, 0, 400, 300]);
    fifo_token(m, 0x2A, &[0x1701]);
    fifo_token(m, 0x2C, &[]);
    fifo_token(m, 0x35, &[f(0.0), f(400.0), f(0.0), f(300.0), f(-1.0), f(1.0)]);
    fifo_token(m, 0x2A, &[0x1700]);
    fifo_token(m, 0x2C, &[]);
    fifo_token(m, 0x47, &[0x1D00]);
}

fn gl_tri(m: &Mgras, c: [f32; 3], v: [[f32; 3]; 3]) {
    fifo_token(m, 0x1A, &[]);
    fifo_token(m, 0x02, &[f(c[0]), f(c[1]), f(c[2])]);
    for p in v {
        fifo_token(m, 0x00, &[f(p[0]), f(p[1]), f(p[2])]);
    }
    fifo_token(m, 0x24, &[]);
}

/// Depth test (GL_LESS): a near triangle drawn first hides the far one
/// drawn over it; with the depth buffer cleared to 1.0 the far one still
/// shows where it is alone.
#[test]
fn gl_depth_test_hides_the_farther_triangle() {
    let m = live_board();
    x_server(&m);
    gl_window_400x300(&m);
    fifo_token(&m, 0x66, &[1]);
    fifo_token(&m, 0xBA, &[f(0.0), f(0.0), f(0.0), f(1.0)]);
    fifo_token(&m, 0x15, &[]);
    fifo_token(&m, 0x4A, &[]);
    // glOrtho(-1, 1) maps object z to window z = (1 - z) / 2: z = 0.5 near.
    gl_tri(&m, [0.0, 0.0, 1.0], [[50.0, 50.0, 0.5], [250.0, 50.0, 0.5], [150.0, 250.0, 0.5]]);
    gl_tri(&m, [1.0, 0.0, 0.0], [[100.0, 40.0, -0.5], [350.0, 40.0, -0.5], [225.0, 260.0, -0.5]]);
    let at = |x: usize, y_gl: usize| m.fb_pixel(x, 1023 - y_gl);
    assert_eq!(at(150, 100), 0xFF_0000, "near blue wins where they overlap");
    assert_eq!(at(320, 60), 0x00_00FF, "far red where it is alone");
    m.stop_engines();
}

/// Back-face culling: a clockwise triangle (back facing with the default
/// front face, counter-clockwise) is dropped; with culling off it draws.
#[test]
fn gl_back_face_culling() {
    let m = live_board();
    x_server(&m);
    gl_window_400x300(&m);
    let cw = [[80.0, 60.0, 0.0], [200.0, 240.0, 0.0], [320.0, 60.0, 0.0]];
    fifo_token(&m, 0x72, &[]);
    fifo_token(&m, 0x74, &[0x405]);
    gl_tri(&m, [0.0, 1.0, 0.0], cw);
    let at = |x: usize, y_gl: usize| m.fb_pixel(x, 1023 - y_gl);
    assert_eq!(at(200, 100), 0, "culled");
    fifo_token(&m, 0x56, &[0x900]);
    gl_tri(&m, [0.0, 1.0, 0.0], cw);
    assert_eq!(at(200, 100), 0x00_FF00, "front face clockwise: drawn");
    m.stop_engines();
}

/// gltest's setup: glFrustum(-4/3, 4/3, -1, 1, 1, 100), the object
/// translated to z = -5, depth test on, a quad facing the viewer.
#[test]
fn gl_frustum_quad_with_depth() {
    let m = live_board();
    x_server(&m);
    switch_to_gl_context(&m, SLOT_C, 0, 0, 800, 600);
    fifo_token(&m, 0x33, &[0, 0, 800, 600]);
    fifo_token(&m, 0x2A, &[0x1701]);
    fifo_token(&m, 0x2C, &[]);
    fifo_token(&m, 0x34, &[f(-4.0 / 3.0), f(4.0 / 3.0), f(-1.0), f(1.0), f(1.0), f(100.0)]);
    fifo_token(&m, 0x2A, &[0x1700]);
    fifo_token(&m, 0x66, &[1]);
    fifo_token(&m, 0xBA, &[f(0.4), f(0.1), f(0.6), f(1.0)]);
    fifo_token(&m, 0x4A, &[]);
    fifo_token(&m, 0x15, &[]);
    fifo_token(&m, 0x2C, &[]);
    fifo_token(&m, 0x32, &[f(0.0), f(0.0), f(-5.0)]);
    fifo_token(&m, 0x1D, &[]);
    fifo_token(&m, 0x02, &[f(1.0), f(1.0), f(0.0)]);
    for p in [[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]] {
        fifo_token(&m, 0x00, &[f(p[0]), f(p[1]), f(p[2])]);
    }
    fifo_token(&m, 0x27, &[]);
    let at = |x: usize, y_gl: usize| m.fb_pixel(x, 1023 - y_gl);
    assert_eq!(at(400, 300), 0x00_FFFF, "the quad covers the centre");
    assert_eq!(at(10, 10), 0x99_1A66, "clear colour outside it");
    m.stop_engines();
}

// ---- GL raster state, one feature per test --------------------------------

/// A 400x300 GL window (ortho), cleared to `clear`, flat shading.
fn gl_board(clear: [f32; 3]) -> std::sync::Arc<Mgras> {
    let m = live_board();
    x_server(&m);
    gl_window_400x300(&m);
    fifo_token(&m, 0xBA, &[f(clear[0]), f(clear[1]), f(clear[2]), f(1.0)]);
    fifo_token(&m, 0x15, &[]);
    m
}

/// Window pixel (x, y), GL y up, of the 400x300 window at the screen's
/// lower left.
fn gl_px(m: &Mgras, x: usize, y: usize) -> u32 {
    m.fb_pixel(x, 1023 - y) & 0xFF_FFFF
}

fn gl_color4(m: &Mgras, c: [f32; 4]) {
    fifo_token(m, 0x02, &[f(c[0]), f(c[1]), f(c[2]), f(c[3])]);
}

/// A full-window quad in the current colour.
fn gl_full_quad(m: &Mgras) {
    fifo_token(m, 0x1D, &[]);
    for p in [[0.0, 0.0], [400.0, 0.0], [400.0, 300.0], [0.0, 300.0]] {
        fifo_token(m, 0x00, &[f(p[0]), f(p[1]), f(0.0)]);
    }
    fifo_token(m, 0x27, &[]);
}

fn gl_line_tokens(m: &Mgras, a: [f32; 2], b: [f32; 2]) {
    fifo_token(m, 0x17, &[]);
    fifo_token(m, 0x00, &[f(a[0]), f(a[1]), f(0.0)]);
    fifo_token(m, 0x00, &[f(b[0]), f(b[1]), f(0.0)]);
    fifo_token(m, 0x21, &[]);
}

#[test]
fn gl_blend_src_alpha_over_the_clear_colour() {
    let m = gl_board([1.0, 0.0, 0.0]);
    fifo_token(&m, 0x65, &[1]);
    fifo_token(&m, 0x3F, &[0x302, 0x303]);
    gl_color4(&m, [0.0, 0.0, 1.0, 0.5]);
    gl_full_quad(&m);
    let p = gl_px(&m, 200, 150);
    assert_eq!((p & 0xFF, (p >> 8) & 0xFF, p >> 16), (0x80, 0, 0x80), "{p:#x}");
    // ONE, ONE adds.
    fifo_token(&m, 0x3F, &[1, 1]);
    gl_color4(&m, [0.0, 1.0, 0.0, 1.0]);
    gl_full_quad(&m);
    assert_eq!(gl_px(&m, 200, 150), 0x80_FF80);
    m.stop_engines();
}

#[test]
fn gl_alpha_test_greater() {
    let m = gl_board([0.0, 0.0, 0.0]);
    fifo_token(&m, 0x64, &[1]);
    fifo_token(&m, 0x3E, &[4, 0x800]);
    gl_color4(&m, [1.0, 1.0, 1.0, 0.25]);
    gl_full_quad(&m);
    assert_eq!(gl_px(&m, 200, 150), 0, "alpha 0.25 fails GREATER 0.5");
    gl_color4(&m, [1.0, 1.0, 1.0, 0.75]);
    gl_full_quad(&m);
    assert_eq!(gl_px(&m, 200, 150), 0xFF_FFFF, "alpha 0.75 passes");
    m.stop_engines();
}

#[test]
fn gl_logic_op_xor_and_colour_mask() {
    let m = gl_board([1.0, 0.0, 0.0]);
    fifo_token(&m, 0x6B, &[1]);
    fifo_token(&m, 0x40, &[6]);
    gl_color4(&m, [1.0, 1.0, 0.0, 1.0]);
    gl_full_quad(&m);
    assert_eq!(gl_px(&m, 200, 150), 0x00_FF00, "red XOR yellow = green");
    fifo_token(&m, 0x6B, &[0]);
    // Colour mask: red and blue only.
    fifo_token(&m, 0x3B, &[0x5]);
    gl_color4(&m, [1.0, 1.0, 1.0, 1.0]);
    gl_full_quad(&m);
    assert_eq!(gl_px(&m, 200, 150), 0xFF_FFFF & (0xFF_00FF | 0x00_FF00), "green kept, red and blue written");
    m.stop_engines();
}

/// The classic stencil mask: a triangle writes stencil 1 with colour
/// writes off, then a full quad drawn where stencil == 1.
#[test]
fn gl_stencil_masks_a_later_draw() {
    let m = gl_board([0.0, 0.0, 0.0]);
    fifo_token(&m, 0xBE, &[0]);
    fifo_token(&m, 0x4C, &[]);
    fifo_token(&m, 0x6F, &[1]);
    fifo_token(&m, 0x41, &[7, 1, 0xFF]);
    fifo_token(&m, 0x42, &[0, 0, 2]);
    fifo_token(&m, 0x3B, &[0]);
    gl_tri(&m, [1.0, 1.0, 1.0], [[100.0, 50.0, 0.0], [300.0, 50.0, 0.0], [200.0, 250.0, 0.0]]);
    fifo_token(&m, 0x3B, &[0xF]);
    fifo_token(&m, 0x41, &[2, 1, 0xFF]);
    fifo_token(&m, 0x42, &[0, 0, 0]);
    gl_color4(&m, [0.0, 0.0, 1.0, 1.0]);
    gl_full_quad(&m);
    assert_eq!(gl_px(&m, 200, 100), 0xFF_0000, "inside the stencilled triangle");
    assert_eq!(gl_px(&m, 20, 20), 0, "outside it");
    m.stop_engines();
}

/// Depth GREATER with depth writes off: a far quad passes against a
/// cleared-to-0 depth buffer, and leaves it at 0 for the next one.
#[test]
fn gl_depth_func_greater_without_depth_writes() {
    let m = gl_board([0.0, 0.0, 0.0]);
    fifo_token(&m, 0x66, &[1]);
    fifo_token(&m, 0xBC, &[0]);
    fifo_token(&m, 0x4A, &[]);
    fifo_token(&m, 0x43, &[4]);
    fifo_token(&m, 0x3C, &[0]);
    gl_tri(&m, [1.0, 0.0, 0.0], [[0.0, 0.0, -0.5], [400.0, 0.0, -0.5], [200.0, 300.0, -0.5]]);
    assert_eq!(gl_px(&m, 200, 100), 0x00_00FF);
    gl_tri(&m, [0.0, 1.0, 0.0], [[0.0, 0.0, 0.9], [400.0, 0.0, 0.9], [200.0, 300.0, 0.9]]);
    assert_eq!(gl_px(&m, 200, 100), 0x00_FF00, "depth buffer untouched: still 0");
    m.stop_engines();
}

#[test]
fn gl_lines_half_open_width_and_stipple() {
    let m = gl_board([0.0, 0.0, 0.0]);
    gl_color4(&m, [1.0, 1.0, 1.0, 1.0]);
    gl_line_tokens(&m, [10.0, 10.5], [100.0, 10.5]);
    assert_eq!(gl_px(&m, 10, 10), 0xFF_FFFF, "first pixel in");
    assert_eq!(gl_px(&m, 99, 10), 0xFF_FFFF);
    assert_eq!(gl_px(&m, 100, 10), 0, "last pixel out");
    assert_eq!(gl_px(&m, 50, 11), 0, "one pixel wide");
    // Width 3: rows 19..21 around y = 20.5.
    fifo_token(&m, 0x45, &[f(3.0)]);
    gl_line_tokens(&m, [10.0, 20.5], [100.0, 20.5]);
    assert_eq!([gl_px(&m, 50, 19), gl_px(&m, 50, 20), gl_px(&m, 50, 21), gl_px(&m, 50, 22)], [0xFF_FFFF, 0xFF_FFFF, 0xFF_FFFF, 0]);
    fifo_token(&m, 0x45, &[f(1.0)]);
    // Stipple 0x0F0F as libGLcore sends it: bit-reversed (0xF0F0), repeat
    // 1: four pixels on, four off.
    fifo_token(&m, 0x6A, &[1]);
    fifo_token(&m, 0x46, &[0, 0xF0F0]);
    gl_line_tokens(&m, [0.0, 40.5], [16.0, 40.5]);
    let row: Vec<u32> = (0..16).map(|x| (gl_px(&m, x, 40) != 0) as u32).collect();
    assert_eq!(row, [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]);
    m.stop_engines();
}

#[test]
fn gl_polygon_mode_line_and_points() {
    let m = gl_board([0.0, 0.0, 0.0]);
    fifo_token(&m, 0xA8, &[0x408, 0x1B01]);
    gl_tri(&m, [0.0, 1.0, 0.0], [[100.5, 50.5, 0.0], [300.5, 50.5, 0.0], [200.5, 250.5, 0.0]]);
    assert_eq!(gl_px(&m, 200, 120), 0, "interior not filled");
    assert_eq!(gl_px(&m, 200, 50), 0x00_FF00, "bottom edge drawn");
    fifo_token(&m, 0xA8, &[0x408, 0x1B02]);
    fifo_token(&m, 0x16, &[]);
    fifo_token(&m, 0x00, &[f(5.5), f(7.5), f(0.0)]);
    fifo_token(&m, 0x20, &[]);
    assert_eq!(gl_px(&m, 5, 7), 0x00_FF00, "a point lights its pixel");
    assert_eq!(gl_px(&m, 6, 7), 0);
    m.stop_engines();
}

#[test]
fn gl_polygon_stipple_and_scissor() {
    let m = gl_board([0.0, 0.0, 0.0]);
    let rows: Vec<u32> = (0..32).map(|r| if r & 1 == 1 { 0x5555_5555 } else { 0xAAAA_AAAA }).collect();
    fifo_token(&m, 0x4E, &rows);
    fifo_token(&m, 0x6D, &[1]);
    gl_color4(&m, [1.0, 1.0, 1.0, 1.0]);
    gl_full_quad(&m);
    // Row 0 = 0xAAAAAAAA: MSB (x mod 32 = 0) set.
    assert_eq!([gl_px(&m, 0, 0), gl_px(&m, 1, 0), gl_px(&m, 0, 1), gl_px(&m, 1, 1)], [0xFF_FFFF, 0, 0, 0xFF_FFFF]);
    fifo_token(&m, 0x6D, &[0]);
    fifo_token(&m, 0x6E, &[1]);
    // glScissor(30, 40, 100, 50) as libGLcore sends it (traced): x, width
    // - 1, y, height - 1.
    fifo_token(&m, 0x39, &[30, 99, 40, 49]);
    gl_color4(&m, [1.0, 0.0, 0.0, 1.0]);
    gl_full_quad(&m);
    assert_eq!(gl_px(&m, 30, 40), 0x00_00FF);
    assert_eq!(gl_px(&m, 129, 89), 0x00_00FF);
    assert_ne!(gl_px(&m, 130, 89), 0x00_00FF, "right of the scissor box");
    assert_ne!(gl_px(&m, 129, 90), 0x00_00FF, "above it");
    assert_ne!(gl_px(&m, 29, 60), 0x00_00FF);
    m.stop_engines();
}

/// Context slots are recycled: a context loaded for the first time (image
/// word 1 bit 31) starts from OpenGL's defaults, not the state the slot's
/// previous owner left (an XOR logic op here).
#[test]
fn gl_new_context_in_a_recycled_slot_starts_fresh() {
    let m = live_board();
    x_server(&m);
    switch_to_gl_context(&m, SLOT_D, 0, 0, 400, 300);
    fifo_token(&m, 0x6B, &[1]);
    fifo_token(&m, 0x40, &[6]);
    // Same slot (word 0), first load again: a new process.
    write(&m, 32, 0x50050, 0x4FC);
    let mut img = [0u32; 63];
    (img[0], img[1], img[2], img[3]) = (SLOT_D, 1 << 31, 0, 0x11);
    (img[11], img[12], img[13]) = (399, 299, 0x240);
    for wd in img {
        write(&m, 32, CFIFO, wd as u64);
    }
    wait_flag(&m, 1 << 6);
    gl_setup_window(&m, 400, 300, [1.0, 0.0, 0.0]);
    fifo_token(&m, 0x15, &[]);
    gl_color4(&m, [1.0, 1.0, 0.0, 1.0]);
    fifo_token(&m, 0x2A, &[0x1701]);
    fifo_token(&m, 0x35, &[f(0.0), f(400.0), f(0.0), f(300.0), f(-1.0), f(1.0)]);
    fifo_token(&m, 0x2A, &[0x1700]);
    gl_full_quad(&m);
    assert_eq!(gl_px(&m, 200, 150), 0x00_FFFF, "yellow copied, not XORed onto red");
    m.stop_engines();
}

// ---- GL lighting, fog, clip planes -----------------------------------------

/// A quad over the window's middle (100..300 x 75..225) at depth z.
fn gl_mid_quad(m: &Mgras, z: [f32; 4]) {
    fifo_token(m, 0x1D, &[]);
    for (k, p) in [[100.0, 75.0], [300.0, 75.0], [300.0, 225.0], [100.0, 225.0]].iter().enumerate() {
        fifo_token(m, 0x00, &[f(p[0]), f(p[1]), f(z[k])]);
    }
    fifo_token(m, 0x27, &[]);
}

fn grey(v: u32) -> [u32; 3] {
    [v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF]
}

/// OpenGL's default light 0 (white, from +z) and default material on a
/// quad facing it: 0.2 * 0.2 (model ambient) + 0.8 (diffuse) = 0.84.
#[test]
fn gl_default_light_and_material() {
    let m = gl_board([0.0, 0.0, 0.0]);
    fifo_token(&m, 0x0D, &[]);
    fifo_token(&m, 0x71, &[0]);
    fifo_token(&m, 0x07, &[f(0.0), f(0.0), f(1.0)]);
    gl_mid_quad(&m, [0.0; 4]);
    assert_eq!(grey(gl_px(&m, 200, 150)), [214, 214, 214]);
    // A red diffuse material through glMaterial (front, GL_DIFFUSE).
    fifo_token(&m, 0x11, &[0x404, 0x1201, f(1.0), f(0.0), f(0.0), f(1.0)]);
    gl_mid_quad(&m, [0.0; 4]);
    assert_eq!(grey(gl_px(&m, 200, 150)), [10 + 245, 10, 10], "0.04 ambient + 1.0 red diffuse");
    m.stop_engines();
}

/// Colour material (front, ambient and diffuse) follows glColor.
#[test]
fn gl_color_material_tracks_the_colour() {
    let m = gl_board([0.0, 0.0, 0.0]);
    fifo_token(&m, 0x0D, &[]);
    fifo_token(&m, 0x71, &[0]);
    fifo_token(&m, 0xBF, &[0x1602]);
    fifo_token(&m, 0xC2, &[]);
    fifo_token(&m, 0x07, &[f(0.0), f(0.0), f(1.0)]);
    gl_color4(&m, [0.0, 1.0, 0.0, 1.0]);
    gl_mid_quad(&m, [0.0; 4]);
    assert_eq!(grey(gl_px(&m, 200, 150)), [0, 255, 0], "green: 0.2 * 0.2 ambient + 1.0 diffuse, clamped");
    m.stop_engines();
}

/// Two-sided lighting: a clockwise (back-facing) quad shows the back
/// material, lit with the flipped normal.
#[test]
fn gl_two_sided_lighting_uses_the_back_material() {
    let m = gl_board([0.0, 0.0, 0.0]);
    fifo_token(&m, 0x0D, &[]);
    fifo_token(&m, 0x71, &[0]);
    fifo_token(&m, 0x59, &[1]);
    fifo_token(&m, 0x11, &[0x405, 0x1201, f(0.0), f(0.0), f(1.0), f(1.0)]);
    // Normal -z: facing the light once flipped for the back face.
    fifo_token(&m, 0x07, &[f(0.0), f(0.0), f(-1.0)]);
    fifo_token(&m, 0x1D, &[]);
    for p in [[100.0, 75.0], [100.0, 225.0], [300.0, 225.0], [300.0, 75.0]] {
        fifo_token(&m, 0x00, &[f(p[0]), f(p[1]), f(0.0)]);
    }
    fifo_token(&m, 0x27, &[]);
    assert_eq!(grey(gl_px(&m, 200, 150)), [10, 10, 255]);
    m.stop_engines();
}

/// User clip plane 0 keeps x <= 200 (plane (-1, 0, 0, 200)).
#[test]
fn gl_user_clip_plane() {
    let m = gl_board([0.0, 0.0, 0.0]);
    fifo_token(&m, 0xA5, &[0x3000, f(-1.0), f(0.0), f(0.0), f(200.0)]);
    fifo_token(&m, 0xA6, &[0x3000]);
    gl_color4(&m, [1.0, 1.0, 0.0, 1.0]);
    gl_mid_quad(&m, [0.0; 4]);
    assert_eq!(gl_px(&m, 150, 150), 0x00_FFFF);
    assert_eq!(gl_px(&m, 250, 150), 0, "clipped away");
    fifo_token(&m, 0xA7, &[0x3000]);
    gl_mid_quad(&m, [0.0; 4]);
    assert_eq!(gl_px(&m, 250, 150), 0x00_FFFF, "plane disabled");
    m.stop_engines();
}

/// Linear fog from start 0 to end 1 (eye z = -object z under glOrtho):
/// white at the near edge, the fog colour at the far edge.
#[test]
fn gl_linear_fog() {
    let m = gl_board([0.0, 0.0, 0.0]);
    fifo_token(&m, 0x68, &[1]);
    fifo_token(&m, 0xB9, &[0x2601]);
    fifo_token(&m, 0xB5, &[f(0.0)]);
    fifo_token(&m, 0xB6, &[f(1.0)]);
    fifo_token(&m, 0xB4, &[f(0.0), f(1.0), f(0.0), f(1.0)]);
    fifo_token(&m, 0x47, &[0x1D01]);
    gl_color4(&m, [1.0, 1.0, 1.0, 1.0]);
    gl_mid_quad(&m, [0.0, -1.0, -1.0, 0.0]);
    let near = grey(gl_px(&m, 101, 150));
    let far = grey(gl_px(&m, 298, 150));
    assert!(near[0] > 245 && near[1] > 245, "near: {near:?}");
    assert!(far[0] < 10 && far[1] > 245 && far[2] < 10, "far: {far:?}");
    m.stop_engines();
}

/// A token with a control-word data conversion (bits 29:23).
fn fifo_token_conv(m: &Mgras, token: u32, conv: u32, bytes: u32, data: &[u32]) {
    write(m, 32, CFIFO, (conv << 23 | token << 8 | bytes) as u64);
    for d in data {
        write(m, 32, CFIFO, *d as u64);
    }
}

/// The HQ converts integer forms to floats before the GE: glColor4ub
/// (0x29, bytes R G B A in one word, normalised) and glVertex3s (0x72,
/// shorts, not normalised, w padded to 1).
#[test]
fn gl_hq_converts_ubyte_colours_and_short_vertices() {
    let m = gl_board([0.0, 0.0, 0.0]);
    fifo_token_conv(&m, 0x02, 0x29, 4, &[0xFF80_00FF]);
    fifo_token(&m, 0x1D, &[]);
    for (x, y) in [(100u32, 75u32), (300, 75), (300, 225), (100, 225)] {
        fifo_token_conv(&m, 0x00, 0x72, 6, &[x << 16 | y, 0]);
    }
    fifo_token(&m, 0x27, &[]);
    assert_eq!(gl_px(&m, 200, 150), 0x00_80FF, "orange from (255, 128, 0)");
    assert_eq!(gl_px(&m, 50, 150), 0, "outside the quad");
    m.stop_engines();
}

/// Triangle strips keep one winding: with back faces culled, both
/// triangles of a counter-clockwise strip draw. IRIS GL's swaptmesh
/// (0xDC) swaps the two remembered vertices (the new vertex replaces the
/// older one) and keeps the winding: v0 v1 swap v2 swap v3 is the IRIS GL
/// fan idiom, triangles (v0 v1 v2) and (v0 v2 v3).
#[test]
fn gl_tstrip_winding_and_swaptmesh() {
    let m = gl_board([0.0, 0.0, 0.0]);
    fifo_token(&m, 0x72, &[]);
    fifo_token(&m, 0x74, &[0x405]);
    gl_color4(&m, [1.0, 1.0, 1.0, 1.0]);
    let strip = |m: &Mgras, verts: &[[f32; 2]], swap_from: Option<usize>| {
        fifo_token(m, 0x1B, &[]);
        for (k, p) in verts.iter().enumerate() {
            if swap_from.is_some_and(|s| k >= s) {
                fifo_token(m, 0xDC, &[]);
            }
            fifo_token(m, 0x00, &[f(p[0]), f(p[1]), f(0.0)]);
        }
        fifo_token(m, 0x25, &[]);
    };
    // A strip along the bottom: (0,0) (0,100) (100,0) (100,100).
    strip(&m, &[[0.0, 100.0], [0.0, 0.0], [100.0, 100.0], [100.0, 0.0]], None);
    assert_eq!(gl_px(&m, 20, 50), 0xFF_FFFF, "first triangle");
    assert_eq!(gl_px(&m, 80, 50), 0xFF_FFFF, "second triangle, reversed to keep the winding");
    // A fan made with swaptmesh around (250, 50): v0 = centre.
    strip(&m, &[[250.0, 50.0], [300.0, 50.0], [250.0, 100.0], [200.0, 50.0]], Some(2));
    assert_eq!(gl_px(&m, 265, 60), 0xFF_FFFF, "(c, right, top)");
    assert_eq!(gl_px(&m, 235, 60), 0xFF_FFFF, "(c, top, left) after the swap");
    m.stop_engines();
}

/// A context switch can preempt a process halfway through a command: the
/// rest of its words come when it runs again. The parser state belongs to
/// the context, so the next context's commands parse from a clean start
/// (the kernel's SCHEDULE_SWAP was swallowed as the old command's data, a
/// swap timeout that killed the X server when a GL window was resized).
#[test]
fn gl_partial_command_survives_a_context_switch() {
    let m = live_board();
    x_server(&m);
    switch_to_gl_context(&m, SLOT_A, 0, 0, 100, 100);
    gl_setup_window(&m, 100, 100, [0.0, 0.0, 0.0]);
    fifo_token(&m, 0x2A, &[0x1701]);
    fifo_token(&m, 0x35, &[f(0.0), f(100.0), f(0.0), f(100.0), f(-1.0), f(1.0)]);
    fifo_token(&m, 0x2A, &[0x1700]);
    // A: COLOR4F header and its first word, then preempted.
    write(&m, 32, CFIFO, (0x02 << 8 | 16) as u64);
    write(&m, 32, CFIFO, f(1.0) as u64);
    switch_to_gl_context(&m, SLOT_B, 200, 0, 100, 100);
    gl_setup_window(&m, 100, 100, [0.0, 0.0, 1.0]);
    fifo_token(&m, 0x15, &[]);
    assert_eq!(gl_px(&m, 250, 50), 0xFF_0000, "context B's clear ran");
    // A resumes: the colour's last three words, then a quad in it.
    switch_to_gl_context(&m, SLOT_A, 0, 0, 100, 100);
    for w in [f(1.0), f(0.0), f(1.0)] {
        write(&m, 32, CFIFO, w as u64);
    }
    fifo_token(&m, 0x1D, &[]);
    for p in [[10.0, 10.0], [90.0, 10.0], [90.0, 90.0], [10.0, 90.0]] {
        fifo_token(&m, 0x00, &[f(p[0]), f(p[1]), f(0.0)]);
    }
    fifo_token(&m, 0x27, &[]);
    assert_eq!(gl_px(&m, 50, 50), 0x00_FFFF, "A's colour (1, 1, 0) arrived whole");
    m.stop_engines();
}

/// A display-list branch runs the list's commands from host memory, through
/// the display-list page table, then the FIFO carries on (ideas keeps its
/// materials in lists).
#[test]
fn gl_display_list_runs_from_host_memory() {
    let m = gl_board([0.0, 0.0, 0.0]);
    let mem = Arc::new(TestMem::default());
    // The table's low bits carry flags (the kernel writes 0x...d); logical
    // page 0 is frame 2. The list, at offset 0x80: the colour red.
    mem.words.lock().insert(0x1000, 2);
    let list = [0x02 << 8 | 16, f(1.0), f(0.0), f(0.0), f(1.0)];
    for (i, w) in list.iter().enumerate() {
        mem.words.lock().insert(0x2080 + 4 * i as u32, *w);
    }
    m.set_phys(mem.clone());
    fifo_token(&m, 0x800 | 0x28, &[0, 0x1001]);
    fifo_token(&m, 0x800 | 0x0C, &[0x80, list.len() as u32]);
    gl_full_quad(&m);
    assert_eq!(gl_px(&m, 200, 150), 0x0000FF);
    m.stop_engines();
}

/// Pool 0's page table at 0x1000, logical pages 0..n at frames 2.., and a
/// one-line host DMA of `bytes` bytes from logical 0.
fn eram_dma_setup(m: &Mgras, bytes: u32) -> Arc<TestMem> {
    let mem = Arc::new(TestMem::default());
    for p in 0..8 {
        mem.words.lock().insert(0x1000 + 4 * p, 2 + p);
    }
    m.set_phys(mem.clone());
    fifo_dma(m, 0x21, 0x1000);
    fifo_dma(m, 0x06, 0);
    fifo_dma(m, 0x05, 0);
    fifo_dma(m, 0x04, 0);
    fifo_dma(m, 0x07, 1);
    fifo_dma(m, 0x08, bytes);
    mem
}

/// The X server's ERAM write at start-up (traced): a word count, the words
/// as FIFO pixel data, then where they go. Read back the kernel's way: a
/// board to host DMA armed first, then the ERAM read command feeds it.
#[test]
fn eram_write_by_pio_and_read_by_dma() {
    let m = live_board();
    x_server(&m);
    let mem = eram_dma_setup(&m, 8);
    fifo_token(&m, 0xFD, &[2]);
    write(&m, 32, CFIFO, 0x8000_0008);
    write(&m, 32, CFIFO, 0x1234_5678);
    write(&m, 32, CFIFO, 0x9ABC_DEF0);
    fifo_token(&m, 0xFB, &[0x40, 2]);
    fifo_token(&m, 0xFA, &[]);
    fifo_dma(&m, 0x0B, 0x9); // run, pool 0, board to host
    fifo_token(&m, 0xFC, &[0x40, 2]);
    m.state_hash();
    let b = mem.bytes.lock();
    let got: Vec<u8> = (0..8).map(|i| b.get(&(0x2000 + i)).copied().unwrap_or(0)).collect();
    assert_eq!(got, [0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0]);
    drop(b);
    m.stop_engines();
}

/// Slot eviction as IRIX does it: the kernel copies a context's ERAM slot
/// to host memory and later back into whatever slot is free. The context
/// loaded from its new slot still has its GL state (here its clear colour
/// and viewport).
#[test]
fn gl_context_survives_eviction_to_another_slot() {
    let m = live_board();
    x_server(&m);
    switch_to_gl_context(&m, SLOT_A, 0, 0, 100, 100);
    gl_setup_window(&m, 100, 100, [1.0, 0.0, 0.0]);
    switch_to_gl_context(&m, SLOT_B, 200, 0, 100, 100);
    // Evict A: slot to host, then host into slot C.
    let mem = eram_dma_setup(&m, 0x17C7 * 4);
    fifo_dma(&m, 0x0B, 0x9);
    fifo_token(&m, 0xFC, &[SLOT_A, 0x17C8]);
    fifo_token(&m, 0xFD, &[0x17C8]);
    fifo_dma(&m, 0x0B, 0x1); // run, pool 0, host to board
    fifo_token(&m, 0xFB, &[SLOT_C, 0x17C7]);
    fifo_token(&m, 0xFA, &[]);
    m.state_hash();
    assert!(mem.bytes.lock().len() > 1000, "the slot reached host memory");
    switch_to_gl_context(&m, SLOT_C, 400, 0, 100, 100);
    fifo_token(&m, 0x15, &[]);
    assert_eq!(m.fb_pixel(450, 1023 - 50), 0x00_00FF, "A's clear colour, from its new slot");
    m.stop_engines();
}

/// Display lists as libGLcore lays them out (traced): each segment starts
/// with its link, the body following, and the engine runs the body before
/// the link. Here: a segment sets red and calls list 1 (DL_PUSH of the
/// return segment, DL_JUMP to the list), list 1 draws the full window and
/// returns, the return segment sets green and returns to the FIFO; then a
/// quad split over two chained segments (DL_JUMP) in the current colour.
#[test]
fn gl_display_list_calls_and_chains_segments() {
    let m = gl_board([0.0, 0.0, 0.0]);
    let mem = Arc::new(TestMem::default());
    mem.words.lock().insert(0x1000, 2); // logical page 0 = frame 2 (0x2000)
    let put = |at: u32, ws: &[u32]| {
        for (i, w) in ws.iter().enumerate() {
            mem.words.lock().insert(0x2000 + at + 4 * i as u32, *w);
        }
    };
    let (push, jump, ret) = (0x83708, 0x80C08, 0x83904);
    let quad = |x0: f32, y0: f32, x1: f32, y1: f32| -> Vec<u32> {
        let mut v = vec![0x1D00];
        for (x, y) in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)] {
            v.extend([0x0C, f(x), f(y), f(0.0)]);
        }
        v.push(0x2700);
        v
    };
    // List 1 at 0: return; the full window.
    let mut list1 = vec![ret, 0];
    list1.extend(quad(0.0, 0.0, 400.0, 300.0));
    put(0x000, &list1);
    // Return segment at 0x200: return; green.
    let back = [ret, 0, 0x0210, f(0.0), f(1.0), f(0.0), f(1.0)];
    put(0x200, &back);
    // Start segment at 0x100: push 0x200, jump to list 1; red.
    let start = [push, 0x200, back.len() as u32, jump, 0, list1.len() as u32, 0x0210, f(1.0), f(0.0), f(0.0), f(1.0)];
    put(0x100, &start);
    // A small quad over two chained segments at 0x400 and 0x500.
    let q = quad(10.0, 10.0, 50.0, 50.0);
    let mut tail = vec![ret, 0];
    tail.extend(&q[9..]);
    let mut head = vec![jump, 0x500, tail.len() as u32];
    head.extend(&q[..9]);
    put(0x400, &head);
    put(0x500, &tail);
    m.set_phys(mem.clone());
    fifo_token(&m, 0x800 | 0x28, &[0, 0x1001]);
    fifo_token(&m, 0x800 | 0x0C, &[0x100, start.len() as u32]);
    assert_eq!(gl_px(&m, 200, 150), 0x0000FF, "list 1 ran in red");
    fifo_token(&m, 0x800 | 0x0C, &[0x400, head.len() as u32]);
    assert_eq!(gl_px(&m, 30, 30), 0x00FF00, "the chained quad, green from the return segment");
    assert_eq!(gl_px(&m, 200, 150), 0x0000FF, "outside it, still red");
    m.stop_engines();
}

/// glBitmap as libGLcore sends it (traced, glXUseXFont text): the raster
/// position, then BITMAP's header and its rows as FIFO pixel data, bottom
/// row first, most significant bit leftmost. Set bits draw in the raster
/// colour; the raster position moves on for the next glyph.
#[test]
fn gl_bitmap_draws_set_bits_at_the_raster_position() {
    let m = gl_board([0.0, 0.0, 0.0]);
    gl_color4(&m, [1.0, 1.0, 1.0, 1.0]);
    fifo_token(&m, 0x38, &[f(10.0), f(10.0), f(0.0)]);
    let glyph = |m: &Mgras| {
        fifo_token(m, 0x94, &[0x1800F, 16, 2, f(0.0), f(0.0), f(9.0), f(0.0), 1]);
        write(m, 32, CFIFO, 0x8000_0004);
        write(m, 32, CFIFO, 0xC000_8000); // row 0: bits 0, 1; row 1: bit 0
        write(m, 32, CFIFO, 0);
    };
    glyph(&m);
    glyph(&m);
    let px = |x, y| gl_px(&m, x, y);
    assert_eq!([px(10, 10), px(11, 10), px(12, 10)], [0xFF_FFFF, 0xFF_FFFF, 0], "row 0");
    assert_eq!([px(10, 11), px(11, 11)], [0xFF_FFFF, 0], "row 1, above");
    assert_eq!([px(19, 10), px(20, 10), px(19, 11)], [0xFF_FFFF, 0xFF_FFFF, 0xFF_FFFF], "next glyph, 9 to the right");
    assert_eq!(px(15, 10), 0, "between the glyphs");
    m.stop_engines();
}

/// A colour-index GL context (INIT_CI, as gr_osview's IRIS GL window
/// sends): colours are indices and land in the pixels as 12-bit indices,
/// for the colour-index visual's colormap; the clear uses the clear index.
#[test]
fn gl_colour_index_context_draws_indices() {
    let m = live_board();
    x_server(&m);
    gl_window_400x300(&m);
    fifo_token(&m, 0x0A, &[12]);
    fifo_token(&m, 0xBD, &[2]);
    fifo_token(&m, 0x15, &[]);
    fifo_token(&m, 0x02, &[f(5.0)]);
    fifo_token(&m, 0x1D, &[]);
    for p in [[0.0, 0.0], [200.0, 0.0], [200.0, 300.0], [0.0, 300.0]] {
        fifo_token(&m, 0x00, &[f(p[0]), f(p[1]), f(0.0)]);
    }
    fifo_token(&m, 0x27, &[]);
    assert_eq!(m.fb_pixel(100, 1023 - 150) & 0xFFF, 5, "the quad, index 5");
    assert_eq!(m.fb_pixel(300, 1023 - 150) & 0xFFF, 2, "the clear index");
    m.stop_engines();
}


/// Query GE state via __MGR_RETURN_MODE (token 0x0A2) and wait for the answer.
fn gl_return_mode(m: &Mgras, addr: u32) -> u32 {
    write(m, 32, 0x7000C, 1 << 17); // clear FLAG_GE_DATA
    fifo_token(m, 0x0A2, &[addr, 1, 1]);
    wait_flag(m, 1 << 17);
    read(m, 32, 0x70014) as u32
}

#[test]
fn gl_raster_position_queries_valid_bias_and_update() {
    let m = live_board();
    x_server(&m);
    gl_window_400x300(&m);

    // Set a valid raster position at (50.0, 60.0, 0.0)
    fifo_token(&m, 0x02, &[f(0.5), f(0.25), f(0.75), f(1.0)]); // latch color
    fifo_token(&m, 0x38, &[f(50.0), f(60.0), f(0.0)]);

    // 0x2D0 should report valid with bit 2 (0x4) and bit 0 (0x1) set
    let valid = gl_return_mode(&m, 0x2D0);
    assert_ne!(valid & 0x4, 0, "bit 2 must be set for libGLcore");
    assert_ne!(valid & 0x1, 0, "bit 0 must be set");

    // 0x2CD (X) and 0x2CE (Y) should be biased by 49151.5
    let rx = f32::from_bits(gl_return_mode(&m, 0x2CD));
    let ry = f32::from_bits(gl_return_mode(&m, 0x2CE));
    let rz = f32::from_bits(gl_return_mode(&m, 0x2CF));
    let rw = f32::from_bits(gl_return_mode(&m, 0x2C4));
    let dist = f32::from_bits(gl_return_mode(&m, 0x2BF));
    assert_eq!(rx, 50.0 + 49151.5);
    assert_eq!(ry, 60.0 + 49151.5);
    assert_eq!(rz, 0.5); // glOrtho maps z=0 to window z=0.5
    assert_eq!(rw, 1.0);
    assert_eq!(dist, 0.0);

    // Raster color readback at 0x2C5..=0x2C8
    let cr = f32::from_bits(gl_return_mode(&m, 0x2C5));
    let cg = f32::from_bits(gl_return_mode(&m, 0x2C6));
    assert_eq!(cr, 0.5);
    assert_eq!(cg, 0.25);

    // UPDATE_RASTER_POS (0x0D6) advances raster by (dx, dy)
    fifo_token(&m, 0x0D6, &[f(15.0), f(25.0)]);
    let rx2 = f32::from_bits(gl_return_mode(&m, 0x2CD));
    let ry2 = f32::from_bits(gl_return_mode(&m, 0x2CE));
    assert_eq!(rx2, 65.0 + 49151.5);
    assert_eq!(ry2, 85.0 + 49151.5);

    // LOAD_RASTER_POS_INFO (0x0D7) restores current color to raster_color
    fifo_token(&m, 0x02, &[f(0.0), f(0.0), f(1.0), f(1.0)]); // blue
    fifo_token(&m, 0x0D7, &[]); // restore to raster color (red/green)

    // Invalid raster position (outside clip volume)
    fifo_token(&m, 0x38, &[f(500.0), f(60.0), f(0.0)]);
    let valid_out = gl_return_mode(&m, 0x2D0);
    assert_eq!(valid_out, 0, "clipped raster position must be invalid");

    // Scissor readback at 0x200 and 0x202
    fifo_token(&m, 0x39, &[10, 100, 20, 200]);
    let sc_x = gl_return_mode(&m, 0x200);
    let sc_y = gl_return_mode(&m, 0x202);
    assert_eq!(sc_x, (100 << 16) | 10);
    assert_eq!(sc_y, (200 << 16) | 20);

    m.stop_engines();
}

#[test]
fn gl_attribute_queries_restore_scissor_viewport_and_index_mask() {
    let m = gl_board([0.0, 0.0, 0.0]);
    fifo_token(&m, 0x0A, &[12]);
    fifo_token(&m, 0x3D, &[0xFFF]);
    fifo_token(&m, 0x33, &[10, 20, 400, 300]);
    fifo_token(&m, 0x39, &[30, 99, 40, 49]);
    let vp = [0xB91, 0xB92, 0xB93, 0xB94].map(|a| gl_return_mode(&m, a));
    let sc = [0xB98, 0xB99, 0xB9A, 0xB9B].map(|a| gl_return_mode(&m, a));
    let mask = gl_return_mode(&m, 0x29F);
    assert_eq!(vp, [10, 20, 400, 300]);
    assert_eq!(sc, [30, 40, 129, 89]);
    assert_eq!(mask, 0xFFF);

    // Alias uses glPushAttrib/glPopAttrib around its UI drawing. Restore
    // the queried values after a nested draw changed the relevant state.
    fifo_token(&m, 0x33, &[0, 0, 1, 1]);
    fifo_token(&m, 0x39, &[0, 0, 0, 0]);
    fifo_token(&m, 0x3D, &[0]);
    fifo_token(&m, 0x33, &vp);
    fifo_token(&m, 0x39, &[sc[0], sc[2] - sc[0], sc[1], sc[3] - sc[1]]);
    fifo_token(&m, 0x3D, &[mask]);
    fifo_token(&m, 0x6E, &[1]);
    fifo_token(&m, 0xBD, &[f(2044.0)]);
    fifo_token(&m, 0x15, &[]);
    assert_eq!(gl_px(&m, 30, 40), 2044);
    assert_eq!(gl_px(&m, 129, 89), 2044);
    assert_eq!(gl_px(&m, 130, 89), 0);
    assert_eq!(gl_px(&m, 29, 40), 0);
    assert_eq!([0xB91, 0xB92, 0xB93, 0xB94].map(|a| gl_return_mode(&m, a)), vp);

    fifo_token(&m, 0x09, &[]);
    fifo_token(&m, 0x3B, &[0x5]);
    assert_eq!(gl_return_mode(&m, 0x29F), 0x5, "RGB component mask");
    fifo_token(&m, 0x2A, &[0x1700]);
    assert_eq!(gl_return_mode(&m, 0x29), u32::MAX, "modelview mode encoding");
    assert_eq!(gl_return_mode(&m, 0x83), 0x899);
    fifo_token(&m, 0x2F, &[]);
    assert_eq!(gl_return_mode(&m, 0x83), 0x8A9);
    fifo_token(&m, 0x2A, &[0x1701]);
    assert_eq!(gl_return_mode(&m, 0x29), 0);
    assert_eq!(gl_return_mode(&m, 0x896), 0xA99);
    assert_eq!(gl_return_mode(&m, 0x4A), gl_return_mode(&m, 0x3A));
    m.stop_engines();
}

#[test]
fn gl_colour_index_mask_applies_to_clear() {
    let m = live_board();
    x_server(&m);
    gl_window_400x300(&m);
    fifo_token(&m, 0x0A, &[12]);
    fifo_token(&m, 0xBD, &[0x0AA]);
    fifo_token(&m, 0x15, &[]); // clear to 0x0AA
    assert_eq!(m.fb_pixel(100, 1023 - 100) & 0xFFF, 0x0AA);

    // Apply glIndexMask(0x0F0) and clear with 0x055: only the high nibble changes
    fifo_token(&m, 0x3D, &[0x0F0]);
    fifo_token(&m, 0xBD, &[0x055]);
    fifo_token(&m, 0x15, &[]);
    // High nibble was 0xA, becomes 0x5; low nibble remains 0xA: 0x05A
    assert_eq!(m.fb_pixel(100, 1023 - 100) & 0xFFF, 0x05A);
    m.stop_engines();
}

/// glDrawPixels as mandel sends it (traced, colour index, GL_UNSIGNED_SHORT):
/// the transfer mode through RSS_REG_SET (0x07F, register/value pairs as
/// pixel data), the raster position, then SEND_PIXELS (0x08D) with the rows
/// as pixel data. The pixels land at the raster position, left to right.
#[test]
fn gl_draw_pixels_colour_index_shorts() {
    let m = live_board();
    x_server(&m);
    gl_window_400x300(&m);
    fifo_token(&m, 0x0A, &[12]);
    fifo_token(&m, 0xBD, &[0]);
    fifo_token(&m, 0x15, &[]);
    fifo_token(&m, 0x7F, &[1, 2]);
    write(&m, 32, CFIFO, 0x8000_0008);
    write(&m, 32, CFIFO, 0x159);
    write(&m, 32, CFIFO, 0x00C0_0001);
    fifo_token(&m, 0x38, &[f(10.0), f(20.0), f(0.0)]);
    fifo_token(&m, 0x8D, &[2, 0, 0, 1, 0, 1, 0x49D0, 0x18]);
    write(&m, 32, CFIFO, 0x8000_0008);
    write(&m, 32, CFIFO, 0x0005_0006);
    write(&m, 32, CFIFO, 0x0007_0008);
    let px = |x: usize| m.fb_pixel(x, 1023 - 20) & 0xFFF;
    assert_eq!([px(10), px(11), px(12), px(13), px(14)], [5, 6, 7, 8, 0]);
    m.stop_engines();
}

/// Pixel zoom (snoop's magnifier): the GE pixel state at 0x192 holds
/// 1 / zoom (token 0x080, traced: 1/6 at zoom 6); each pixel and row of a
/// SEND_PIXELS image is repeated.
#[test]
fn gl_draw_pixels_zoomed() {
    let m = live_board();
    x_server(&m);
    gl_window_400x300(&m);
    fifo_token(&m, 0x0A, &[12]);
    fifo_token(&m, 0xBD, &[0]);
    fifo_token(&m, 0x15, &[]);
    fifo_token(&m, 0x7F, &[1, 2]);
    write(&m, 32, CFIFO, 0x8000_0008);
    write(&m, 32, CFIFO, 0x159);
    write(&m, 32, CFIFO, 0x00C0_0001);
    fifo_token(&m, 0x80, &[0, 2, 8, 0x191, 0, 2, 2, 0, f(0.5)]);
    fifo_token(&m, 0x38, &[f(10.0), f(20.0), f(0.0)]);
    fifo_token(&m, 0x8D, &[1, 0, 0, 1, 0, 1, 0x49D0, 0x18]);
    write(&m, 32, CFIFO, 0x8000_0008);
    write(&m, 32, CFIFO, 0x0005_0006);
    write(&m, 32, CFIFO, 0);
    let px = |x: usize, y: usize| m.fb_pixel(x, 1023 - y) & 0xFFF;
    assert_eq!([px(10, 20), px(11, 20), px(12, 20), px(13, 20), px(14, 20)], [5, 5, 6, 6, 0]);
    assert_eq!([px(10, 21), px(13, 21), px(10, 22)], [5, 6, 0]);
    m.stop_engines();
}

/// glReadPixels as snoop sends it (traced, one pixel): the read block and
/// transfer size through RSS_REG_SHADOW (0x07C triples), the transfer
/// counters and mode through RSS_REG_SET, GET_PIXELS (0x0DB), then the
/// host arms a board-to-host DMA and starts the raster side through raster
/// interface register 5 (0x4009).
#[test]
fn gl_read_pixels_to_host_memory() {
    let m = gl_board([0.0, 0.0, 0.0]);
    gl_color4(&m, [1.0, 0.5, 0.25, 1.0]);
    gl_full_quad(&m);
    let px = gl_px(&m, 30, 40);
    assert_ne!(px, 0);
    fifo_token(&m, 0xD2, &[0x11E2_3929, 0x100_0000, 0xFF, 0, 2]);
    fifo_token(&m, 0x7C, &[1, 4]);
    write(&m, 32, CFIFO, 0x8000_0010);
    for w in [0x153, 0x0001_0001, 0x226, 0x47] {
        write(&m, 32, CFIFO, w);
    }
    fifo_token(&m, 0x7C, &[2, 6]);
    write(&m, 32, CFIFO, 0x8000_0018);
    for w in [0x46, 30 << 16 | 40, 0x28E, 0x47, 30 << 16 | 40, 0x290] {
        write(&m, 32, CFIFO, w);
    }
    fifo_token(&m, 0x7F, &[2, 4]);
    write(&m, 32, CFIFO, 0x8000_0010);
    for w in [0x158, 0x0001_0001, 0x159, 0x0088_0080] {
        write(&m, 32, CFIFO, w);
    }
    fifo_token(&m, 0xDB, &[0x4009]);
    let mem = eram_dma_setup(&m, 4);
    fifo_dma(&m, 0x0B, 0x9);
    fifo_token(&m, 0xA05, &[0x4009]);
    fifo_token(&m, 0xD3, &[]);
    m.state_hash();
    let b = mem.bytes.lock();
    let got: Vec<u8> = (0..4).map(|i| b.get(&(0x2000 + i)).copied().unwrap_or(0)).collect();
    assert_ne!(got, [0, 0, 0, 0], "the pixel reached host memory (window pixel {px:#x})");
    drop(b);
    m.stop_engines();
}

/// SEND_PIXELS honours the scissor box (the desks overview draws each
/// miniature under its own) and shrinking zooms (nearest source pixel).
#[test]
fn gl_draw_pixels_scissored_and_shrunk() {
    let m = live_board();
    x_server(&m);
    gl_window_400x300(&m);
    fifo_token(&m, 0x0A, &[12]);
    fifo_token(&m, 0xBD, &[0]);
    fifo_token(&m, 0x15, &[]);
    fifo_token(&m, 0x7F, &[1, 2]);
    write(&m, 32, CFIFO, 0x8000_0008);
    write(&m, 32, CFIFO, 0x159);
    write(&m, 32, CFIFO, 0x00C0_0001);
    // Zoom 1/2: four pixels 1, 2, 3, 4 become two, 1 and 3.
    fifo_token(&m, 0x80, &[0, 2, 8, 0x191, 0, 2, 2, 0, f(2.0)]);
    fifo_token(&m, 0x38, &[f(10.0), f(20.0), f(0.0)]);
    fifo_token(&m, 0x8D, &[2, 0, 0, 1, 0, 1, 0x49D0, 0x18]);
    write(&m, 32, CFIFO, 0x8000_0008);
    write(&m, 32, CFIFO, 0x0001_0002);
    write(&m, 32, CFIFO, 0x0003_0004);
    let px = |x: usize, y: usize| m.fb_pixel(x, 1023 - y) & 0xFFF;
    assert_eq!([px(10, 20), px(11, 20), px(12, 20)], [1, 3, 0], "shrunk by 2");
    // Zoom 1, scissor x 31..32: only the middle two pixels of four.
    fifo_token(&m, 0x80, &[0, 2, 8, 0x191, 0, 2, 2, 0, f(1.0)]);
    fifo_token(&m, 0x39, &[31, 1, 0, 299]);
    fifo_token(&m, 0x6E, &[1]);
    fifo_token(&m, 0x38, &[f(30.0), f(20.0), f(0.0)]);
    fifo_token(&m, 0x8D, &[2, 0, 0, 1, 0, 1, 0x49D0, 0x18]);
    write(&m, 32, CFIFO, 0x8000_0008);
    write(&m, 32, CFIFO, 0x0001_0002);
    write(&m, 32, CFIFO, 0x0003_0004);
    assert_eq!([px(30, 20), px(31, 20), px(32, 20), px(33, 20)], [0, 2, 3, 0], "scissored");
    m.stop_engines();
}

/// A GL client's pixel operation sets raster registers through RSS_REG_SET
/// (CONFIG among them, Y-flip off); the X server's must come back after
/// RESTORE_RSS. They leaked: 4Dwm drew snoop's frame with Y-flip off.
#[test]
fn gl_pixel_op_registers_do_not_leak_to_x() {
    let m = gl_board([0.0, 0.0, 0.0]);
    fifo_token(&m, 0xD2, &[0x1E0_3929, 0x140_0000, 0, 0, 3]);
    fifo_token(&m, 0x7F, &[1, 2]);
    write(&m, 32, CFIFO, 0x8000_0008);
    write(&m, 32, CFIFO, CONFIG as u64);
    write(&m, 32, CFIFO, 0x4);
    fifo_token(&m, 0xD3, &[]);
    m.state_hash();
    direct_rss(&m, FILLMODE, FILL_FAST, false);
    direct_rss(&m, FILL_COLOR_R, 0x123, false);
    block(&m, 5, 5, 5, 5);
    m.state_hash();
    assert_eq!(m.fb_pixel(5, 5) & 0xFFF, 0x123, "the X server's fill lands top-down, Y-flip back");
    m.stop_engines();
}

/// The X server sets the IR only when its operation changes: back from a
/// GL batch it executes with the IR as it left it (traced: 4Dwm's lines
/// after a context switch from gltest). GL's triangle opcode leaked into
/// it, and each X line redrew GL's last triangle in X's window.
#[test]
fn gl_batch_leaves_x_instruction_alone() {
    let m = gl_board([0.0, 0.0, 0.0]);
    direct_rss(&m, FILLMODE, FILL_FAST, false);
    direct_rss(&m, FILL_COLOR_R, 0x123, false);
    direct_rss(&m, IR_ALIAS, 0x18, false);
    gl_color4(&m, [0.0, 1.0, 0.0, 1.0]);
    gl_full_quad(&m);
    fifo_token(&m, 0x36, &[]);
    m.state_hash();
    direct_rss(&m, BLOCKXYSTARTI, 5 << 16 | 5, false);
    direct_rss(&m, BLOCKXYENDI, 5 << 16 | 5, true);
    m.state_hash();
    assert_eq!(m.fb_pixel(5, 5) & 0xFFF, 0x123, "X's block, with X's instruction");
    assert_eq!(m.fb_pixel(300, 5) & 0xFF_FFFF, 0, "no GL quad in X's window");
    m.stop_engines();
}

/// Twilight is a full-screen GL root painter. X paints CID 1 only in its
/// visible region; its context image's PP1winmode is 0x20. Iconifying a
/// console changes that region, then Twilight redraws the whole screen.
/// The geometric screen mask alone cannot protect the other X windows.
#[test]
fn gl_root_painter_respects_clipping_ids_after_iconify() {
    use super::rss::reg;
    let m = live_board();
    x_server(&m);
    direct_rss(&m, reg::PP1WINMODE, 0xC00, false);
    direct_rss(&m, FILLMODE, FILL_FAST, false);
    direct_rss(&m, FILL_COLOR_R, 0x77, false);
    block(&m, 0, 724, 399, 1023);

    // The visible root, with two occluding windows left at CID 0.
    let paint_cid = |x0, y0, x1, y1, cid| {
        direct_rss(&m, PP1FILLMODE, 0x142600, false);
        direct_rss(&m, reg::COLORMASKMSBS, 0xFF, false);
        direct_rss(&m, FILL_COLOR_R, cid, false);
        block(&m, x0, y0, x1, y1);
    };
    paint_cid(0, 724, 399, 1023, 1);
    paint_cid(40, 800, 79, 839, 0);
    paint_cid(140, 800, 179, 839, 0);

    switch_to_gl_context(&m, SLOT_A, 0, 0, 400, 300);
    // Reload the same context with the PP1 window word from the trace.
    write(&m, 32, 0x50050, 0x4FC);
    let mut img = [0u32; 63];
    img[0] = SLOT_A;
    img[3] = 0x11;
    img[4] = 0x20;
    img[11] = 399;
    img[12] = 299;
    img[13] = 0x240;
    (img[16], img[17]) = (1, 1);
    for wd in img { write(&m, 32, CFIFO, wd as u64); }
    wait_flag(&m, 1 << 6);
    gl_setup_window(&m, 400, 300, [1.0, 0.0, 0.0]);
    fifo_token(&m, 0x15, &[]);
    assert_eq!(m.fb_pixel(20, 820) & 0xFF_FFFF, 0xFF);
    assert_eq!(m.fb_pixel(50, 820), 0x77, "console protected from GL clear");
    assert_eq!(m.fb_pixel(150, 820), 0x77, "other X window protected");

    // Expose the console rectangle, then validate the current context with
    // CP_WINDOW (window mode in word 1, the same PP1 word in word 2: the
    // kernel stores them as one doubleword, window mode high) and draw a
    // quad.
    paint_cid(40, 800, 79, 839, 1);
    let mut win = [0u32; 15];
    win[1] = 0x11;
    win[2] = 0x20;
    win[9] = 399;
    win[10] = 299;
    win[11] = 0x240;
    fifo_token(&m, 0xE4, &win);
    fifo_token(&m, 0x2A, &[0x1701]);
    fifo_token(&m, 0x2C, &[]);
    fifo_token(&m, 0x35, &[f(0.0), f(400.0), f(0.0), f(300.0), f(-1.0), f(1.0)]);
    fifo_token(&m, 0x2A, &[0x1700]);
    fifo_token(&m, 0x2C, &[]);
    gl_color4(&m, [0.0, 1.0, 0.0, 1.0]);
    gl_full_quad(&m);
    assert_eq!(m.fb_pixel(50, 820) & 0xFF_FFFF, 0xFF00, "exposed root repainted");
    assert_eq!(m.fb_pixel(150, 820), 0x77, "other window survives GL triangles");

    // X's bypass state must also survive the GL bracket: its next fill
    // changes neither PP1winmode nor the CID planes.
    direct_rss(&m, PP1FILLMODE, 0x0C00_4504, false);
    direct_rss(&m, FILL_COLOR_R, 0x55, false);
    block(&m, 150, 820, 150, 820);
    assert_eq!(m.fb_pixel(150, 820), 0x55);
    m.stop_engines();
}

// ---- texture (TE1) ---------------------------------------------------------

/// FIFO pixel data: one pixel command carrying `words`, padded to whole
/// doublewords as the HQ takes them.
fn fifo_pixel_data(m: &Mgras, words: &[u32]) {
    write(m, 32, CFIFO, 0x8000_0000 | 4 * words.len() as u64);
    for w in words {
        write(m, 32, CFIFO, *w as u64);
    }
    if words.len() % 2 == 1 {
        write(m, 32, CFIFO, 0);
    }
}

/// An RSS_REG_SHADOW list (0x07C): (register, value) with a GE shadow
/// address each, padded to whole doublewords, as libGLcore sends them.
fn rss_shadow_list(m: &Mgras, regs: &[(u32, u32)]) {
    let mut words: Vec<u32> = regs.iter().flat_map(|&(r, v)| [r, v, 0x200 + r]).collect();
    if words.len() % 2 == 1 {
        words.push(0x189);
    }
    fifo_token(m, 0x7C, &[regs.len() as u32, words.len() as u32]);
    fifo_pixel_data(m, &words);
}

/// glTexImage2D of one level of a square RGBA8 texture as libGLcore sends
/// it (traced, IRIX 6.5.22, 4 TRAMs): loader registers, SAVE_RSS, the
/// sub-image (0xDA8) and destination (0xDB1: width, ?, level, page | bank
/// << 8) pixel state, SEND_PIXELS routine 0x511A and the texels.
fn gl_tex_level(m: &Mgras, level: u32, n: u32, dest: u32, texels: &[u32]) {
    gl_tex_level_as(m, level, n, dest, texels, 1, 0.0, 0x34E26);
}

/// `gl_tex_level` for host elements `elem` (pixel state 0xDBB: 1 bytes, 3
/// 16-bit, 4 32-bit) with component scale `scale` (0xD0A; 0 leaves it).
fn gl_tex_level_as(m: &Mgras, level: u32, n: u32, dest: u32, texels: &[u32], elem: u32, scale: f32, tl_mode: u32) {
    let log = n.trailing_zeros();
    fifo_token(m, 0x0C, &[]);
    rss_shadow_list(m, &[(0x1A0, 0), (0x18B, 0), (0x111, 0x40858)]);
    rss_shadow_list(m, &[(0x1A1, tl_mode)]);
    rss_shadow_list(m, &[(0x1A2, log << 8 | log << 4 | level)]);
    fifo_token(m, 0xD2, &[0xFFFF_FFFF, 0x0140_0000, 0, 0, 3]);
    fifo_token(m, 0xCD, &[1, 1, 0x14, 0xDA8, 0, 2, 5, 0, 0, n, n, n << 16 | n, 0]);
    fifo_token(m, 0xCD, &[1, 2, 0x18, 0xDB1, 0, 2, 6, n, 0, level, dest, 0, 0, 0]);
    fifo_token(m, 0xCD, &[1, 2, 0x18, 0xDB7, 0, 2, 6, 0, 0, 0, 0, elem, 0, 0]);
    if scale != 0.0 {
        fifo_token(m, 0xCD, &[1, 1, 4, 0xD0A, 0, 2, 1, f(scale), 0]);
    }
    fifo_token(m, 0x8D, &[texels.len() as u32, 0, 0, 1, 0, 1, 0x511A, 0x98]);
    fifo_pixel_data(m, texels);
    fifo_token(m, 0xD3, &[]);
}

/// A 4x4 RGBA8 texture at page 0: texel (s, t) = (s * 0x40, t * 0x40,
/// 0xA5, 0xFF).
fn gl_tex_image_4x4(m: &Mgras) {
    let texels: Vec<u32> = (0..16).map(|i| (i % 4) * 0x40 << 24 | (i / 4) * 0x40 << 16 | 0xA5FF).collect();
    gl_tex_level(m, 0, 4, 0, &texels);
}

/// Texturing on for what follows, as for a draw (traced): TEX_ON, the
/// sampler registers, level 0 at page 0 through RSS_REG_SHADOW_IDX.
fn gl_tex_bind_4x4(m: &Mgras, texmode2: u32, texmode1: u32) {
    fifo_token(m, 0x0B, &[]);
    rss_shadow_list(m, &[(0x180, texmode2)]);
    rss_shadow_list(m, &[(0x111, texmode1)]);
    rss_shadow_list(m, &[(0x184, 0), (0x185, 0), (0x190, 0)]);
    fifo_token(m, 0xEC, &[0xBC6, 0xBC7, 0x191, 0]);
    rss_shadow_list(m, &[(0x181, 0x2222), (0x183, 1), (0x18C, 0x8004)]);
}

/// A quad over window (100..300, 100..200) with texture coordinates 0..1.
fn gl_tex_quad(m: &Mgras) {
    fifo_token(m, 0x1D, &[]);
    for (t, p) in [([0.0, 0.0], [100.0, 100.0]), ([1.0, 0.0], [300.0, 100.0]), ([1.0, 1.0], [300.0, 200.0]), ([0.0, 1.0], [100.0, 200.0])] {
        fifo_token(m, 0x08, &[f(t[0]), f(t[1])]);
        fifo_token(m, 0x00, &[f(p[0]), f(p[1]), f(0.0)]);
    }
    fifo_token(m, 0x27, &[]);
}

/// A texture loaded through the texture loader and drawn nearest with
/// modulate (libGLcore's GL_REPLACE) on a white quad: each texel covers a
/// 50 x 25 pixel block, in GL's orientation (s right, t up).
#[test]
fn gl_texture_nearest_quad() {
    let m = gl_board([1.0, 0.5, 0.75]);
    gl_tex_image_4x4(&m);
    gl_tex_bind_4x4(&m, 0x10004, 0x40859);
    gl_color4(&m, [1.0, 1.0, 1.0, 1.0]);
    gl_tex_quad(&m);
    for (s, t) in [(0, 0), (3, 0), (0, 3), (2, 1)] {
        let (x, y) = (100 + 50 * s + 25, 100 + 25 * t + 12);
        assert_eq!(gl_px(&m, x, y), 0xA5_0000 | (t as u32 * 0x40) << 8 | s as u32 * 0x40, "texel ({s}, {t}) at ({x}, {y})");
    }
    assert_eq!(gl_px(&m, 50, 50), 0xBF_80FF, "clear colour outside");
    m.stop_engines();
}

/// A tilted textured quad, larger than the view so the clipper cuts it
/// into a fan, at `scale` times the distance and size: the same picture
/// at any scale. Returns the window's pixels.
fn far_quad_pixels(scale: f32) -> Vec<u32> {
    let m = gl_board([0.0, 0.0, 0.0]);
    gl_tex_image_4x4(&m);
    fifo_token(&m, 0x2A, &[0x1701]);
    fifo_token(&m, 0x2C, &[]);
    fifo_token(&m, 0x34, &[f(-1.0), f(1.0), f(-0.75), f(0.75), f(1.0), f(1.0e8)]);
    fifo_token(&m, 0x2A, &[0x1700]);
    fifo_token(&m, 0x2C, &[]);
    // Far enough that only the sides clip it (the near plane does not scale).
    fifo_token(&m, 0x32, &[f(0.0), f(0.0), f(-4.0 * scale)]);
    fifo_token(&m, 0x30, &[f(-60.0), f(1.0), f(0.0), f(0.0)]);
    gl_tex_bind_4x4(&m, 0x10004, 0x40859);
    gl_color4(&m, [1.0, 1.0, 1.0, 1.0]);
    fifo_token(&m, 0x1F, &[]);
    for (t, p) in [([0.0, 0.0], [-3.0, -3.0]), ([3.0, 0.0], [3.0, -3.0]), ([3.0, 3.0], [3.0, 3.0]), ([0.0, 3.0], [-3.0, 3.0])] {
        fifo_token(&m, 0x08, &[f(t[0]), f(t[1])]);
        fifo_token(&m, 0x00, &[f(p[0] * scale), f(p[1] * scale), f(0.0)]);
    }
    fifo_token(&m, 0x29, &[]);
    m.state_hash();
    let px = (0..300).flat_map(|y| (0..400).map(move |x| (x, y))).map(|(x, y)| gl_px(&m, x, y)).collect();
    m.stop_engines();
    px
}

/// Far geometry (blast's nebula, W ~ 5e5) keeps its texture mapping: the
/// fixed-point S/W, T/W, Q/W planes are normalised per primitive. Without
/// that, 1/W's steps were one or two LSBs and each triangle of the clipped
/// quad mapped the texture differently.
#[test]
fn gl_texture_far_quad_maps_as_near() {
    let (near, far) = (far_quad_pixels(1.0), far_quad_pixels(1.0e6));
    let textured = near.iter().filter(|&&p| p != 0).count();
    // Texel edges may move by a pixel: a far pixel is wrong when no near
    // pixel within one matches it.
    let at = |v: &[u32], x: i32, y: i32| v.get((y.clamp(0, 299) * 400 + x.clamp(0, 399)) as usize).copied();
    let differ = (0..300 * 400)
        .filter(|&i| {
            let (x, y) = (i % 400, i / 400);
            !(-1..=1).any(|dy| (-1..=1).any(|dx| at(&near, x + dx, y + dy) == Some(far[i as usize])))
        })
        .count();
    assert!(textured > 60_000, "the quad covers most of the window ({textured})");
    assert!(differ < textured / 200, "{differ} of {textured} pixels differ");
}

/// SGIS_texture_select (traced, TyrQuake's lightmaps): one-component
/// textures share cells, each in the slot TL_MODE bits 4:3 give at load
/// and TEXMODE1 bits 8:7 when drawn. Two luminance textures on the same
/// page keep their own texels.
#[test]
fn gl_texture_select_slots_share_a_page() {
    let m = gl_board([0.0, 0.0, 0.0]);
    gl_tex_level_as(&m, 0, 4, 0, &[0x4040_4040; 4], 1, 0.0, 0x34020);
    gl_tex_level_as(&m, 0, 4, 0, &[0xC0C0_C0C0; 4], 1, 0.0, 0x34028);
    for (slot, want) in [(0u32, 0x40_4040), (1, 0xC0_C0C0)] {
        // Luminance class (2), one component, the slot in bits 8:7.
        gl_tex_bind_4x4(&m, 0x10004, 0x40441 | slot << 7);
        gl_color4(&m, [1.0, 1.0, 1.0, 1.0]);
        gl_tex_quad(&m);
        assert_eq!(gl_px(&m, 125, 112), want, "slot {slot}");
    }
    m.stop_engines();
}

/// The select bits mean nothing for textures that fill the cell: TyrQuake
/// draws RGBA textures with TEXMODE1 0x409D9 (bits 8:7 = 3).
#[test]
fn gl_texture_select_ignored_for_rgba() {
    let m = gl_board([1.0, 0.5, 0.75]);
    gl_tex_image_4x4(&m);
    gl_tex_bind_4x4(&m, 0x10004, 0x40859 | 3 << 7);
    gl_color4(&m, [1.0, 1.0, 1.0, 1.0]);
    gl_tex_quad(&m);
    assert_eq!(gl_px(&m, 100 + 50 * 2 + 25, 100 + 25 + 12), 0xA5_0000 | 0x40 << 8 | 2 * 0x40);
    m.stop_engines();
}

/// GL_UNSIGNED_SHORT_4_4_4_4_EXT texels (SGI quake's lightmaps): 16-bit
/// elements at an 8-bit scale, each nibble a component, R on top.
#[test]
fn gl_texture_packed_4444() {
    let m = gl_board([1.0, 0.5, 0.75]);
    let t16 = |i: u32| (i % 4) * 4 << 12 | (i / 4) * 4 << 8 | 0xAF;
    let words: Vec<u32> = (0..8).map(|k| t16(2 * k) << 16 | t16(2 * k + 1)).collect();
    gl_tex_level_as(&m, 0, 4, 0, &words, 3, 1.0 / 255.0, 0x34E26);
    gl_tex_bind_4x4(&m, 0x10004, 0x40859);
    gl_color4(&m, [1.0, 1.0, 1.0, 1.0]);
    gl_tex_quad(&m);
    for (s, t) in [(0, 0), (3, 0), (0, 3), (2, 1)] {
        let (x, y) = (100 + 50 * s + 25, 100 + 25 * t + 12);
        assert_eq!(gl_px(&m, x, y), 0xAA_0000 | (t as u32 * 0x44) << 8 | s as u32 * 0x44, "texel ({s}, {t})");
    }
    m.stop_engines();
}

/// Modulate with a coloured fragment, and texturing off again: TEX_OFF
/// draws plain colour even with the texture still bound.
#[test]
fn gl_texture_modulate_and_off() {
    let m = gl_board([0.0, 0.0, 0.0]);
    gl_tex_image_4x4(&m);
    gl_tex_bind_4x4(&m, 0x10004, 0x40859);
    gl_color4(&m, [1.0, 0.0, 1.0, 1.0]);
    gl_tex_quad(&m);
    assert_eq!(gl_px(&m, 100 + 150 + 25, 100 + 75 + 12), 0xA5_00C0, "texel (3, 3) times magenta");
    fifo_token(&m, 0x0C, &[]);
    gl_tex_quad(&m);
    assert_eq!(gl_px(&m, 100 + 150 + 25, 100 + 75 + 12), 0xFF_00FF, "untextured");
    m.stop_engines();
}

/// Bilinear magnification (TEXMODE2 bit 7): halfway between two texel
/// centres is their average; at a centre, the texel.
#[test]
fn gl_texture_bilinear_magnification() {
    let m = gl_board([0.0, 0.0, 0.0]);
    gl_tex_image_4x4(&m);
    gl_tex_bind_4x4(&m, 0x10084, 0x40859);
    gl_color4(&m, [1.0, 1.0, 1.0, 1.0]);
    gl_tex_quad(&m);
    // Texel centres at x = 125, 175 ... and y = 112.5, 137.5 ...; at x =
    // 200, between texels (1, 1) and (2, 1), the red ramp is halfway: 0x60.
    let mid = gl_px(&m, 200, 137);
    assert!(((mid & 0xFF) as i32 - 0x60).abs() <= 3, "{mid:#x}");
    let centre = gl_px(&m, 175, 137);
    assert!(((centre & 0xFF) as i32 - 0x40).abs() <= 3, "{centre:#x}");
    m.stop_engines();
}

/// A second context with no texture state draws untextured, and the first
/// gets its texture back when it returns: the TE is loaded from each
/// context's own shadow, TRAM is shared.
#[test]
fn gl_texture_state_is_per_context() {
    let m = live_board();
    x_server(&m);
    switch_to_gl_context(&m, SLOT_A, 0, 0, 400, 300);
    gl_ortho_400x300(&m);
    gl_tex_image_4x4(&m);
    gl_tex_bind_4x4(&m, 0x10004, 0x40859);
    switch_to_gl_context(&m, SLOT_B, 0, 0, 400, 300);
    gl_ortho_400x300(&m);
    fifo_token(&m, 0x0B, &[]);
    gl_color4(&m, [0.0, 1.0, 0.0, 1.0]);
    gl_tex_quad(&m);
    assert_eq!(gl_px(&m, 125, 112), 0x00_FF00, "plain green in context B");
    switch_to_gl_context(&m, SLOT_A, 0, 0, 400, 300);
    gl_color4(&m, [1.0, 1.0, 1.0, 1.0]);
    gl_tex_quad(&m);
    assert_eq!(gl_px(&m, 125, 112), 0xA5_0000, "texel (0, 0) in context A");
    assert_eq!(gl_px(&m, 275, 187), 0xA5_C0C0, "texel (3, 3) in context A");
    m.stop_engines();
}

/// Mipmaps: level 0 (4x4) at page 0, level 1 (2x2, flat grey) at page 1
/// bank 1, each placed by its own destination. Magnified the quad shows
/// level 0; drawn at 2x2 pixels (two texels a pixel, lambda 1) level 1.
#[test]
fn gl_texture_mipmap_levels_from_their_pages() {
    let m = gl_board([0.0, 0.0, 0.0]);
    gl_tex_image_4x4(&m);
    gl_tex_level(&m, 1, 2, 0x101, &[0x8080_80FF; 4]);
    fifo_token(&m, 0x0B, &[]);
    rss_shadow_list(&m, &[(0x180, 0x90024), (0x111, 0x40859), (0x190, 0)]);
    fifo_token(&m, 0xEC, &[0xBC6, 0xBC7, 0x191, 0]);
    fifo_token(&m, 0xEC, &[0xBC6, 0xBC7, 0x191, 0x101]);
    rss_shadow_list(&m, &[(0x181, 0x2222), (0x183, 1)]);
    gl_color4(&m, [1.0, 1.0, 1.0, 1.0]);
    gl_tex_quad(&m);
    assert_eq!(gl_px(&m, 275, 187), 0xA5_C0C0, "level 0, texel (3, 3)");
    fifo_token(&m, 0x1D, &[]);
    for (t, p) in [([0.0, 0.0], [10.0, 10.0]), ([1.0, 0.0], [12.0, 10.0]), ([1.0, 1.0], [12.0, 12.0]), ([0.0, 1.0], [10.0, 12.0])] {
        fifo_token(&m, 0x08, &[f(t[0]), f(t[1])]);
        fifo_token(&m, 0x00, &[f(p[0]), f(p[1]), f(0.0)]);
    }
    fifo_token(&m, 0x27, &[]);
    assert_eq!(gl_px(&m, 11, 11), 0x80_8080, "level 1");
    m.stop_engines();
}

/// distort's texture load (traced): 16-bit components (pixel format scale
/// 1/65535 at ERAM 0xD0A), a one-texel border, the sub-image starting at
/// (-1, -1), rows SEND_PIXELS' words-per-row apart. The interior lands in
/// the level, the border is dropped, the top byte of each component kept.
#[test]
fn gl_texture_rgba16_with_border() {
    let m = gl_board([0.0, 0.0, 0.0]);
    fifo_token(&m, 0x0C, &[]);
    rss_shadow_list(&m, &[(0x1A0, 0), (0x18B, 0), (0x111, 0x40858)]);
    rss_shadow_list(&m, &[(0x1A1, 0x35A26)]);
    rss_shadow_list(&m, &[(0x1A2, 0x220)]);
    fifo_token(&m, 0xD2, &[0xFFFF_FFFF, 0x0140_0000, 0, 0, 3]);
    fifo_token(&m, 0xCD, &[1, 2, 0x18, 0xDB1, 0, 2, 6, 4, 0, 0, 0, 0, 0, 0]);
    fifo_token(&m, 0xCD, &[1, 2, 0x18, 0xD05, 0, 2, 6, 0x20C, 0x4064, 0x50C, 0xA, 0x14, 0x3780_0080, 0]);
    fifo_token(&m, 0xCD, &[1, 1, 0x14, 0xDA8, 0, 2, 5, 0xFFFF_FFFF, 0xFFFF_FFFF, 6, 6, 0x60006, 0]);
    // 6 rows of 6 texels, 8 bytes each: 12 words a row.
    let mut words = Vec::new();
    for t in -1i32..5 {
        for s in -1i32..5 {
            let border = !(0..4).contains(&s) || !(0..4).contains(&t);
            let (r, g, b) = if border { (0xFFFF, 0xFFFF, 0xFFFF) } else { (s as u32 * 0x4000 + 0xFF, t as u32 * 0x4000, 0xA5A5) };
            words.push(r << 16 | g);
            words.push(b << 16 | 0xFFFF);
        }
    }
    fifo_token(&m, 0x8D, &[12, 0, 0, 0, 2, 1, 0x511A, 0x98]);
    for row in words.chunks(12) {
        fifo_pixel_data(&m, row);
    }
    fifo_token(&m, 0xD3, &[]);
    gl_tex_bind_4x4(&m, 0x10004, 0x40859);
    gl_color4(&m, [1.0, 1.0, 1.0, 1.0]);
    gl_tex_quad(&m);
    assert_eq!(gl_px(&m, 125, 112), 0xA5_0000, "texel (0, 0)");
    assert_eq!(gl_px(&m, 275, 187), 0xA5_C0C0, "texel (3, 3)");
    assert_eq!(gl_px(&m, 175, 137), 0xA5_4040, "texel (1, 1)");
    m.stop_engines();
}

/// The HQ formatter's swizzle (traced: GL_ABGR_EXT texture loads set the
/// mode to 0xE8D for bytes, 0xE8E for shorts, around the DMA): component
/// order reversed within each pixel, each component's bytes kept; the
/// canonical mode 0xE00 leaves data alone.
#[test]
fn formatter_swizzle_reverses_components() {
    let mut b = [0xFF, 0xA5, 0x00, 0x20, 1, 2, 3, 4];
    super::hq3::format_pixels(0xE8D, &mut b);
    assert_eq!(b, [0x20, 0x00, 0xA5, 0xFF, 4, 3, 2, 1]);
    let mut s = [0x00, 0x00, 0x0F, 0x0F, 0x1B, 0x1B, 0x22, 0x21];
    super::hq3::format_pixels(0xE8E, &mut s);
    assert_eq!(s, [0x22, 0x21, 0x1B, 0x1B, 0x0F, 0x0F, 0x00, 0x00]);
    let mut c = [1, 2, 3, 4];
    super::hq3::format_pixels(0xE00, &mut c);
    assert_eq!(c, [1, 2, 3, 4]);
}

/// A quad over window (100..300, 100..200) with texture coordinates s0..s1
/// and t0..t1.
fn gl_tex_quad_st(m: &Mgras, s: [f32; 2], t: [f32; 2]) {
    fifo_token(m, 0x1D, &[]);
    for (tc, p) in [([s[0], t[0]], [100.0, 100.0]), ([s[1], t[0]], [300.0, 100.0]), ([s[1], t[1]], [300.0, 200.0]), ([s[0], t[1]], [100.0, 200.0])] {
        fifo_token(m, 0x08, &[f(tc[0]), f(tc[1])]);
        fifo_token(m, 0x00, &[f(p[0]), f(p[1]), f(0.0)]);
    }
    fifo_token(m, 0x27, &[]);
}

fn near(v: u32, want: [i32; 3]) -> bool {
    [v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF].iter().zip(want).all(|(&c, w)| (c as i32 - w).abs() <= 3)
}

/// GL_CLAMP with linear filtering and no border (TEXMODE2 0x15984,
/// traced): left of s = 0 the filter blends texel 0 half and half with the
/// border colour (TXBCOLOR, green here).
#[test]
fn gl_texture_clamp_blends_the_border_colour() {
    let m = gl_board([0.0, 0.0, 0.0]);
    gl_tex_image_4x4(&m);
    gl_tex_bind_4x4(&m, 0x15984, 0x40859);
    rss_shadow_list(&m, &[(0x184, 0xFFF000), (0x185, 0xFFF000)]);
    gl_color4(&m, [1.0, 1.0, 1.0, 1.0]);
    gl_tex_quad_st(&m, [-1.0, 2.0], [0.0, 1.0]);
    // y = 137: t = 0.375, exactly texel row 1 (green 0x40).
    let v = gl_px(&m, 105, 137);
    assert!(near(v, [0, (255 + 0x40) / 2, 0xA5 / 2]), "{v:#x}");
    m.stop_engines();
}

/// A texture with a one-texel border (magenta), loaded from (-1, -1), its
/// border texels on their own page (0xDB1's border page): with GL_CLAMP
/// and linear filtering the edge blends with the border texels.
#[test]
fn gl_texture_border_texels() {
    let m = gl_board([0.0, 0.0, 0.0]);
    fifo_token(&m, 0x0C, &[]);
    rss_shadow_list(&m, &[(0x1A0, 0), (0x18B, 0), (0x111, 0x40858)]);
    rss_shadow_list(&m, &[(0x1A1, 0x35A26)]);
    rss_shadow_list(&m, &[(0x1A2, 0x220)]);
    fifo_token(&m, 0xD2, &[0xFFFF_FFFF, 0x0140_0000, 0, 0, 3]);
    fifo_token(&m, 0xCD, &[1, 2, 0x18, 0xDB1, 0, 2, 6, 4, 0, 0, 0, 1, 0, 0]);
    fifo_token(&m, 0xCD, &[1, 1, 0x14, 0xDA8, 0, 2, 5, 0xFFFF_FFFF, 0xFFFF_FFFF, 6, 6, 0x60006, 0]);
    let mut texels = Vec::new();
    for t in -1i32..5 {
        for s in -1i32..5 {
            let border = !(0..4).contains(&s) || !(0..4).contains(&t);
            texels.push(if border { 0xFF00_FFFF } else { (s as u32 * 0x40) << 24 | (t as u32 * 0x40) << 16 | 0xA5FF });
        }
    }
    fifo_token(&m, 0x8D, &[18, 0, 0, 2, 0, 1, 0x511A, 0x98]);
    for part in texels.chunks(18) {
        fifo_pixel_data(&m, part);
    }
    fifo_token(&m, 0xD3, &[]);
    gl_tex_bind_4x4(&m, 0x5984, 0x40859);
    rss_shadow_list(&m, &[(0x190, 0)]);
    fifo_token(&m, 0xEC, &[0xBC6, 0xBC7, 0x193, 1]);
    gl_color4(&m, [1.0, 1.0, 1.0, 1.0]);
    gl_tex_quad_st(&m, [-1.0, 2.0], [0.0, 1.0]);
    let v = gl_px(&m, 105, 137);
    assert!(near(v, [0xFF / 2, 0x40 / 2, (0xFF + 0xA5) / 2]), "{v:#x}");
    // s = 0.375 at x = 191.5: texel (1, 1)'s centre.
    let v = gl_px(&m, 191, 137);
    assert!(near(v, [0x40, 0x40, 0xA5]), "interior texel (1, 1) intact: {v:#x}");
    m.stop_engines();
}

/// Object-linear texgen (traced: TEX_GENIV, plane header and data, enable
/// token 0x075 with GL_TEXTURE_GEN_S / T): planes map the quad to 0..1, the
/// vertices' own coordinates (0) are ignored.
#[test]
fn gl_texgen_object_linear() {
    let m = gl_board([1.0, 0.5, 0.75]);
    gl_tex_image_4x4(&m);
    gl_tex_bind_4x4(&m, 0x10004, 0x40859);
    for (c, plane) in [(0x2000, [1.0 / 200.0, 0.0, 0.0, -0.5]), (0x2001, [0.0, 1.0 / 100.0, 0.0, -1.0])] {
        fifo_token(&m, 0x5A, &[c, 0x2500, 0x2401]);
        fifo_token(&m, 0x5B, &[c, 0x2501]);
        fifo_token(&m, 0x5C, &plane.map(f));
        fifo_token(&m, 0x75, &[c - 0x2000 + 0xC60]);
    }
    gl_color4(&m, [1.0, 1.0, 1.0, 1.0]);
    gl_tex_quad_st(&m, [0.0, 0.0], [0.0, 0.0]);
    for (s, t) in [(0, 0), (3, 0), (0, 3), (2, 1)] {
        let (x, y) = (100 + 50 * s + 25, 100 + 25 * t + 12);
        assert_eq!(gl_px(&m, x, y), 0xA5_0000 | (t as u32 * 0x40) << 8 | s as u32 * 0x40, "texel ({s}, {t})");
    }
    m.stop_engines();
}

/// glPointSize 4 (token 0x04D): a 4x4 square around the point.
#[test]
fn gl_point_size_draws_a_square() {
    let m = gl_board([0.0, 0.0, 0.0]);
    fifo_token(&m, 0x4D, &[f(4.0)]);
    gl_color4(&m, [1.0, 1.0, 1.0, 1.0]);
    fifo_token(&m, 0x16, &[]);
    fifo_token(&m, 0x00, &[f(50.5), f(50.5), f(0.0)]);
    fifo_token(&m, 0x20, &[]);
    for (x, y, on) in [(49, 49, true), (52, 52, true), (49, 52, true), (48, 50, false), (53, 50, false), (50, 48, false), (50, 53, false)] {
        assert_eq!(gl_px(&m, x, y) != 0, on, "({x}, {y})");
    }
    // Size 3 at (100.5, 100.5): pixels 99..101 each way.
    fifo_token(&m, 0x4D, &[f(3.0)]);
    fifo_token(&m, 0x16, &[]);
    fifo_token(&m, 0x00, &[f(100.5), f(100.5), f(0.0)]);
    fifo_token(&m, 0x20, &[]);
    for (x, y, on) in [(99, 99, true), (101, 101, true), (98, 100, false), (102, 100, false), (100, 98, false), (100, 102, false)] {
        assert_eq!(gl_px(&m, x, y) != 0, on, "size 3: ({x}, {y})");
    }
    m.stop_engines();
}

/// A textured line: texture coordinates along it, texel by texel.
#[test]
fn gl_textured_line() {
    let m = gl_board([0.0, 0.0, 0.0]);
    gl_tex_image_4x4(&m);
    gl_tex_bind_4x4(&m, 0x10004, 0x40859);
    gl_color4(&m, [1.0, 1.0, 1.0, 1.0]);
    fifo_token(&m, 0x17, &[]);
    for (s, x) in [(0.0, 100.0), (1.0, 300.0)] {
        fifo_token(&m, 0x08, &[f(s), f(0.375)]);
        fifo_token(&m, 0x00, &[f(x), f(50.5), f(0.0)]);
    }
    fifo_token(&m, 0x21, &[]);
    for k in 0..4u32 {
        assert_eq!(gl_px(&m, 125 + 50 * k as usize, 50), 0xA5_4000 | k * 0x40, "texel ({k}, 1)");
    }
    m.stop_engines();
}

/// Fog with texturing is applied after the texture environment: linear
/// fog 0..1, grey, the quad at eye depth 0.5 (factor 0.5) blends texel
/// (0, 0) half and half with the fog colour.
#[test]
fn gl_fog_after_texture() {
    let m = gl_board([0.0, 0.0, 0.0]);
    gl_tex_image_4x4(&m);
    gl_tex_bind_4x4(&m, 0x10004, 0x40859);
    fifo_token(&m, 0xB9, &[0x2601]);
    fifo_token(&m, 0xB5, &[f(0.0)]);
    fifo_token(&m, 0xB6, &[f(1.0)]);
    fifo_token(&m, 0xB4, &[f(0.5), f(0.5), f(0.5), f(1.0)]);
    fifo_token(&m, 0x68, &[1]);
    gl_color4(&m, [1.0, 1.0, 1.0, 1.0]);
    fifo_token(&m, 0x1D, &[]);
    for (tc, p) in [([0.0, 0.0], [100.0, 100.0]), ([1.0, 0.0], [300.0, 100.0]), ([1.0, 1.0], [300.0, 200.0]), ([0.0, 1.0], [100.0, 200.0])] {
        fifo_token(&m, 0x08, &[f(tc[0]), f(tc[1])]);
        fifo_token(&m, 0x00, &[f(p[0]), f(p[1]), f(-0.5)]);
    }
    fifo_token(&m, 0x27, &[]);
    let v = gl_px(&m, 125, 112);
    assert!(near(v, [0x40, 0x40, (0xA5 + 0x80) / 2]), "{v:#x}");
    m.stop_engines();
}

/// Texgen beyond the texture with GL_REPEAT (glprim --texgen obj's
/// triangle, s 1.29..2.29 across it): the texture repeats along s; nearest
/// sampling does not stop at the last texel.
#[test]
fn gl_texgen_repeats_past_one() {
    let m = gl_board([1.0, 0.5, 0.75]);
    gl_tex_image_4x4(&m);
    gl_tex_bind_4x4(&m, 0x10004, 0x40859);
    for (c, plane) in [(0x2000u32, [1.0f32 / 140.0, 0.0, 0.0, -0.2857]), (0x2001, [0.0, 1.0 / 180.0, 0.0, -0.3333])] {
        fifo_token(&m, 0x5A, &[c, 0x2500, 0x2401]);
        fifo_token(&m, 0x5B, &[c, 0x2501]);
        fifo_token(&m, 0x5C, &plane.map(f));
        fifo_token(&m, 0x75, &[c - 0x2000 + 0xC60]);
    }
    gl_color4(&m, [1.0, 1.0, 1.0, 1.0]);
    fifo_token(&m, 0x1A, &[]);
    for (tc, p) in [([-0.5f32, -0.5f32], [220.0f32, 60.0f32]), ([1.5, -0.5], [360.0, 60.0]), ([0.5, 1.5], [290.0, 240.0])] {
        fifo_token(&m, 0x08, &[f(tc[0]), f(tc[1])]);
        fifo_token(&m, 0x00, &[f(p[0]), f(p[1]), f(0.0)]);
    }
    fifo_token(&m, 0x24, &[]);
    // y = 70: t row 0. x = 240.5: s = 1.43, texel 1; x = 300.5: s = 1.86,
    // texel 3; x = 330.5: s = 2.08, texel 0 again.
    for (x, red) in [(240, 0x40), (300, 0xC0), (330, 0x00)] {
        assert_eq!(gl_px(&m, x, 70) & 0xFF, red, "x {x}");
    }
    m.stop_engines();
}

/// glGetTexImage as libGLcore sends it (traced): SAVE_RSS, the sampler at
/// the level (TXSIZE, TXMIPMAP), READ_TEXTURE (0x0F0), the host DMA armed,
/// then raster interface register 5 = 0x400D starts the read. The texels
/// reach host memory as RGBA bytes, row by row.
#[test]
fn gl_texture_readback_to_host_memory() {
    texture_readback(4, 16);
    // libGLcore reads a small texture as one DMA line (8x8: 256 bytes).
    texture_readback(1, 64);
}

/// The texture manager moving a texture (traced, SGI quake): it saves
/// TRAM pages raw (0xE5 with the sampler on the page, 64 texels of 32 bits
/// a row) and restores them elsewhere through the loader (0x9D, TL_MODE
/// 0x4026, TL_SPEC 64 x 8192, no destination level) from a list-mode
/// host DMA. The texture then samples the same from its new page.
#[test]
fn gl_texture_save_and_restore_pages() {
    texture_save_and_restore(4);
}

/// The same for RGB8: three components of 8 bits a texel in TRAM, a 32-bit
/// cell each in the page view.
#[test]
fn gl_texture_save_and_restore_rgb8() {
    texture_save_and_restore(3);
}

fn texture_save_and_restore(nc: u32) {
    let m = gl_board([0.0, 0.0, 0.0]);
    let tl_mode = (nc - 1) << 1 | 1 << 5;
    let texels: Vec<u32> = (0..16).map(|i| (i % 4) * 0x40 << 24 | (i / 4) * 0x40 << 16 | 0xA5FF).collect();
    gl_tex_level_as(&m, 0, 4, 0, &texels, 1, 0.0, 0x34E00 | tl_mode);
    // Save page 0 (the texture's components and depth): 64 rows of 256
    // bytes into host pages 2..5.
    rss_shadow_list(&m, &[(0x181, 0xD6D6), (0x183, 0), (0x111, (nc - 1) << 3), (0x180, 0x11804), (0x190, 0)]);
    fifo_token(&m, 0xEC, &[0xBC6, 0xBC7, 0x191, 0]);
    fifo_token(&m, 0xD2, &[0xFFFF_FFFF, 0x0100_1000, 0x1F_C0FF, 0, 0]);
    // The transfer mode's texels: RGB three bytes, RGBA four (traced).
    let xfrmode = if nc == 3 { 0x40_0070 } else { 0x40_0080 };
    let bytes = 64 * 64 * nc;
    fifo_token(&m, 0xE5, &[64, xfrmode, 0x400D]);
    let mem = eram_dma_setup(&m, bytes);
    fifo_dma(&m, 0x0B, 0x9);
    fifo_token(&m, 0xA05, &[0x400D]);
    fifo_token(&m, 0xD3, &[]);
    // Page 0 gets another texture.
    gl_tex_level(&m, 0, 4, 0, &[0; 16]);
    // Restore to page 3 from the four host pages, listed.
    rss_shadow_list(&m, &[(0x1A1, 0x4000 | tl_mode), (0x1A2, 0xD60), (0x1A3, 0), (0x1A4, 3)]);
    fifo_token(&m, 0xD2, &[0xFFFF_FFFF, 0x0140_0000, 0, 0, 3]);
    fifo_token(&m, 0x7F, &[1, 2]);
    fifo_pixel_data(&m, &[0x159, xfrmode]);
    fifo_token(&m, 0xCD, &[1, 1, 0x14, 0xDA8, 0, 2, 5, 0, 0, 64, 64, 64 << 16 | 64, 0]);
    fifo_token(&m, 0xCD, &[0, 2, 8, 0xDB1, 0, 2, 2, 0, 0]);
    fifo_token(&m, 0x9D, &[0, 8, 0, 0x511A]);
    write(&m, 32, CFIFO, ((0x800 << 8) | 16) as u64);
    let list = if nc == 3 { [0x0000_0001u32, 0x8002_0000] } else { [0x0000_0001, 0x0002_8003] };
    for w in [list[0], list[1], 0, 0] {
        write(&m, 32, CFIFO, w as u64);
    }
    fifo_dma(&m, 0x05, 0);
    fifo_dma(&m, 0x08, 0x1000);
    fifo_dma(&m, 0x09, 0x1000);
    fifo_dma(&m, 0x07, 1);
    fifo_dma(&m, 0x0B, 0x1);
    fifo_token(&m, 0xD3, &[]);
    // Draw from page 3.
    gl_tex_bind_4x4(&m, 0x10004, 0x40841 | (nc - 1) << 3);
    rss_shadow_list(&m, &[(0x190, 0)]);
    fifo_token(&m, 0xEC, &[0xBC6, 0xBC7, 0x191, 3]);
    gl_color4(&m, [1.0, 1.0, 1.0, 1.0]);
    gl_tex_quad(&m);
    for (s, t) in [(0, 0), (3, 0), (0, 3), (2, 1)] {
        let (x, y) = (100 + 50 * s + 25, 100 + 25 * t + 12);
        assert_eq!(gl_px(&m, x, y), 0xA5_0000 | (t as u32 * 0x40) << 8 | s as u32 * 0x40, "texel ({s}, {t}), {nc} components");
    }
    drop(mem);
    m.stop_engines();
}

fn texture_readback(lines: u32, bytes: u32) {
    let m = gl_board([0.0, 0.0, 0.0]);
    gl_tex_image_4x4(&m);
    rss_shadow_list(&m, &[(0x183, 0), (0x18C, 0x40008), (0x1A1, 0), (0x181, 0x2222), (0x111, 0x58), (0x180, 0x10004)]);
    fifo_token(&m, 0xD2, &[0xFFFF_FFFF, 0x0100_1000, 0x1F_C0FF, 0, 0]);
    rss_shadow_list(&m, &[(0x190, 0)]);
    fifo_token(&m, 0xEC, &[0xBC6, 0xBC7, 0x191, 0]);
    fifo_token(&m, 0xF0, &[4, 4, f(0.5), f(0.5), 0x40_0080, 0x400D]);
    let mem = eram_dma_setup(&m, bytes);
    fifo_dma(&m, 0x07, lines);
    fifo_dma(&m, 0x04, bytes);
    fifo_dma(&m, 0x0B, 0x9);
    fifo_token(&m, 0xA05, &[0x400D]);
    fifo_token(&m, 0xD3, &[]);
    m.state_hash();
    let b = mem.bytes.lock();
    for (s, t) in [(0u32, 0u32), (3, 0), (1, 2), (3, 3)] {
        let at = 0x2000 + t * 16 + s * 4;
        let got: Vec<u8> = (0..4).map(|i| b.get(&(at + i)).copied().unwrap_or(0)).collect();
        assert_eq!(got, [(s * 0x40) as u8, (t * 0x40) as u8, 0xA5, 0xFF], "texel ({s}, {t}), {lines} DMA lines");
    }
    drop(b);
    m.stop_engines();
}

/// glCopyPixels' write-back (traced): the host holds RGB with 16-bit
/// components (6 bytes a pixel), the transfer mode is RGBA16 (0xC00081):
/// the GE widens each pixel (alpha 1) before the raster engine takes it.
#[test]
fn gl_draw_pixels_rgb16_into_rgba16() {
    let m = gl_board([0.0, 0.0, 0.0]);
    fifo_token(&m, 0x7F, &[1, 2]);
    fifo_pixel_data(&m, &[0x159, 0x00C0_0081]);
    fifo_token(&m, 0x38, &[f(10.0), f(20.0), f(0.0)]);
    fifo_token(&m, 0xCD, &[1, 2, 0x18, 0xDA8, 0, 2, 6, 0, 0, 1, 0, 0x1_0002, 1]);
    fifo_token(&m, 0x8D, &[3, 0, 0, 1, 0, 1, 0x49D0, 0x98]);
    fifo_pixel_data(&m, &[0xFFFF_8080, 0xBFBF_0000, 0xFFFF_0000]);
    m.state_hash();
    assert_eq!(gl_px(&m, 10, 20), 0xBF_80FF, "pink");
    assert_eq!(gl_px(&m, 11, 20), 0x00_FF00, "green");
    m.stop_engines();
}

/// A chunk of a larger image (glCopyPixels draws back 80 pixels at a time,
/// traced): pixel state 0xDA8 places it within the image, so it lands that
/// far from the raster position.
#[test]
fn gl_draw_pixels_chunk_offset() {
    let m = gl_board([0.0, 0.0, 0.0]);
    fifo_token(&m, 0x7F, &[1, 2]);
    fifo_pixel_data(&m, &[0x159, 0x00C0_0081]);
    fifo_token(&m, 0x38, &[f(10.0), f(20.0), f(0.0)]);
    fifo_token(&m, 0xCD, &[1, 2, 0x18, 0xDA8, 0, 2, 6, 0x50, 0, 0x51, 0, 0x1_0002, 1]);
    fifo_token(&m, 0x8D, &[3, 0, 0, 1, 0, 1, 0x49D0, 0x98]);
    fifo_pixel_data(&m, &[0xFFFF_8080, 0xBFBF_0000, 0xFFFF_0000]);
    m.state_hash();
    assert_eq!(gl_px(&m, 90, 20), 0xBF_80FF, "at the raster position + 80");
    assert_eq!(gl_px(&m, 10, 20), 0, "not at the raster position");
    m.stop_engines();
}

/// GL_CLAMP_TO_BORDER_SGIS (TEXMODE2 clamp bits without bit 14, libGLcore's
/// mapping): past the edge the samples are all border colour, where
/// GL_CLAMP gives half border, half edge texel.
#[test]
fn gl_texture_clamp_to_border() {
    let m = gl_board([0.0, 0.0, 0.0]);
    gl_tex_image_4x4(&m);
    gl_tex_bind_4x4(&m, 0x11984, 0x40859);
    rss_shadow_list(&m, &[(0x184, 0xFFF000), (0x185, 0xFFF000)]);
    gl_color4(&m, [1.0, 1.0, 1.0, 1.0]);
    gl_tex_quad_st(&m, [-1.0, 2.0], [0.0, 1.0]);
    assert_eq!(gl_px(&m, 105, 137), 0x00_FF00, "border colour only");
    m.stop_engines();
}

/// SEND_PIXELS receives RGBA components after HQ formatting, whereas X
/// transfers use packed ABGR. Opaque black must not turn into red.
#[test]
fn gl_draw_pixels_rgba8_component_order() {
    let m = gl_board([0.0, 0.0, 0.0]);
    fifo_token(&m, 0x7F, &[1, 2]);
    fifo_pixel_data(&m, &[0x159, 0x00C1_0080]);
    fifo_token(&m, 0x38, &[f(10.0), f(20.0), f(0.0)]);
    fifo_token(&m, 0x8D, &[3, 0, 0, 1, 0, 1, 0x49D0, 0x99]);
    fifo_pixel_data(&m, &[0x0000_00FF, 0x1234_5678, 0xFF00_00FF]);
    assert_eq!(gl_px(&m, 10, 20), 0, "opaque black");
    assert_eq!(gl_px(&m, 11, 20), 0x56_3412, "RGB order");
    assert_eq!(gl_px(&m, 12, 20), 0x00_00FF, "red");
    let _sub = m.submit.lock();
    m.wait_idle();
    let rss = unsafe { &*m.rss.get() };
    let b = super::pixmem::Buffer::new(0x240, super::pixmem::Kind::Wide, 0x31E);
    assert_eq!(rss.mem.get(&b, 11, 20) as u32, 0x7856_3412, "alpha survives upload");
    drop(_sub);
    m.stop_engines();
}

/// glCopyPixels' write half by host DMA (Maya copies the front buffer to
/// the back after a full redraw, then redraws only what changes): the
/// image size from pixel state 0xDA8, the transfer mode from the RSS
/// register list, _WRITE_DMAGESETUP with glDrawPixels' routine (0x49D0),
/// then one DMA line with every row. Without it the back buffer keeps
/// whatever it held, and every other frame shows that.
#[test]
fn gl_draw_pixels_by_host_dma() {
    let m = gl_board([0.0, 0.0, 0.0]);
    fifo_token(&m, 0x0A, &[12]);
    fifo_token(&m, 0x38, &[f(10.0), f(20.0), f(0.0)]);
    fifo_token(&m, 0x7F, &[1, 2]);
    fifo_pixel_data(&m, &[0x159, 0xC0_0001]);
    fifo_token(&m, 0xCD, &[1, 2, 0x18, 0xDA8, 0, 2, 6, 0, 0, 3, 1, 2 << 16 | 4, 1, 0]);
    fifo_token(&m, 0x9D, &[0, 0x175, 0x7C, 0x49D0]);
    let mem = eram_dma_setup(&m, 16);
    for (i, v) in [0x801u16, 0x802, 0x803, 0x804, 0x805, 0x806, 0x807, 0x808].iter().enumerate() {
        let [hi, lo] = v.to_be_bytes();
        mem.bytes.lock().insert(0x2000 + 2 * i as u32, hi);
        mem.bytes.lock().insert(0x2001 + 2 * i as u32, lo);
    }
    fifo_dma(&m, 0x0B, 0x1);
    fifo_token(&m, 0xD3, &[]);
    let got: Vec<u32> = [(10, 20), (13, 20), (10, 21), (13, 21)].iter().map(|&(x, y)| gl_px(&m, x, y) & 0xFFF).collect();
    assert_eq!(got, [0x801, 0x804, 0x805, 0x808], "bottom row first");
    drop(mem);
    m.stop_engines();
}

/// Softimage's CI8 overlay has a separate pointer, upper-plane index mask,
/// and absolute draw selector. Clears and geometry must leave main intact.
#[test]
fn gl_native_overlay_clear_geometry_and_bitmap() {
    let m = gl_board([0.25, 0.5, 0.75]);
    let main_before = gl_px(&m, 100, 100);
    {
        let _sub = m.submit.lock();
        m.wait_idle();
        let rss = unsafe { &mut *m.rss.get() };
        let b = super::pixmem::Buffer::new(0x1C0, super::pixmem::Kind::Overlay, 0x31E);
        rss.mem.put(&b, 100, 100, 1); // another overlay plane must survive
    }
    fifo_token(&m, 0xE4, &[0, 0x11, 0, 0, 0, 0, 0, 0, 0, 399, 299, 0x101C0, 0, 0, 0]);
    fifo_token(&m, 0x0A, &[8]);
    fifo_token(&m, 0x9A, &[0x2500, 0x2500]);
    fifo_token(&m, 0x49, &[0x48, 0x48, 1]);
    fifo_token(&m, 0x3D, &[0xF0]);
    fifo_token(&m, 0xBD, &[0x30]);
    fifo_token(&m, 0x15, &[]);
    fifo_token(&m, 0x02, &[f(16.0)]);
    fifo_token(&m, 0x1D, &[]);
    for p in [[0.0, 0.0], [200.0, 0.0], [200.0, 300.0], [0.0, 300.0]] {
        fifo_token(&m, 0x00, &[f(p[0]), f(p[1]), f(0.0)]);
    }
    fifo_token(&m, 0x27, &[]);
    fifo_token(&m, 0x02, &[f(32.0)]);
    fifo_token(&m, 0x38, &[f(250.0), f(100.0), f(0.0)]);
    fifo_token(&m, 0x94, &[0x18000, 16, 1, f(0.0), f(0.0), f(0.0), f(0.0), 1]);
    fifo_pixel_data(&m, &[0x8000_0000]);
    assert_eq!(gl_px(&m, 100, 100), main_before);
    let _sub = m.submit.lock();
    m.wait_idle();
    let rss = unsafe { &*m.rss.get() };
    let b = super::pixmem::Buffer::new(0x1C0, super::pixmem::Kind::Overlay, 0x31E);
    assert_eq!(rss.mem.get(&b, 100, 100), 0x11, "overlay polygon preserves other planes");
    assert_eq!(rss.mem.get(&b, 300, 100), 0x30, "overlay clear");
    assert_eq!(rss.mem.get(&b, 250, 100), 0x20, "overlay glyph");
    drop(_sub);
    m.stop_engines();
}

#[test]
fn gl_native_draw_buffer_masks_follow_the_swap() {
    let m = gl_board([0.0, 0.0, 0.0]);
    fifo_token(&m, 0xE4, &[0, 0x11, 0, 0, 0, 0, 0, 0, 0, 399, 299, 0x240 | 0x140 << 10, 0, 0, 0]);
    fifo_token(&m, 0x49, &[3, 3, 1]);
    gl_color4(&m, [1.0, 0.0, 0.0, 1.0]);
    gl_full_quad(&m);
    fifo_token(&m, 0x49, &[2, 1, 0]); // GL_BACK, 12-bit: B until a swap
    gl_color4(&m, [0.0, 1.0, 0.0, 1.0]);
    gl_tri(&m, [0.0, 1.0, 0.0], [[0.0, 0.0, 0.0], [200.0, 0.0, 0.0], [0.0, 200.0, 0.0]]);
    fifo_token(&m, 0x49, &[0, 0, 0]);
    fifo_token(&m, 0xBA, &[f(0.0), f(0.0), f(1.0), f(1.0)]);
    fifo_token(&m, 0x15, &[]);
    gl_color4(&m, [0.0, 0.0, 1.0, 1.0]);
    gl_full_quad(&m); // DRAW_NONE preserves both pages
    let _sub = m.submit.lock();
    m.wait_idle();
    let rss = unsafe { &*m.rss.get() };
    let a = super::pixmem::Buffer::new(0x240, super::pixmem::Kind::Wide, 0x31E);
    let b = super::pixmem::Buffer::new(0x140, super::pixmem::Kind::Wide, 0x31E);
    assert_eq!(rss.mem.get(&a, 300, 100) as u32 & 0xFF_FFFF, 0xFF);
    assert_eq!(rss.mem.get(&b, 300, 100) as u32 & 0xFF_FFFF, 0xFF, "both pages drawn");
    assert_eq!(rss.mem.get(&b, 50, 50) as u32 & 0xFF_FFFF, 0xFF00, "back before a swap: B");
    assert_eq!(rss.mem.get(&a, 50, 50) as u32 & 0xFF_FFFF, 0xFF, "front A preserved");
    drop(_sub);
    m.stop_engines();
}

/// Every double-buffered demo traced (atlantis, powerflip, solidview)
/// draws with DRAW_BUFFER [4, 1, 0], GL_BACK in a 24-bit visual: buffer B
/// (4) until a swap, A (1) after it, B again after the next.
#[test]
fn gl_back_buffer_alternates_with_swaps() {
    let m = gl_board([0.0, 0.0, 0.0]);
    fifo_token(&m, 0xE4, &[0, 0x11, 0, 0, 0, 0, 0, 0, 0, 399, 299, 0x240 | 0x140 << 10, 0, 0, 0]);
    fifo_token(&m, 0x49, &[4, 1, 0]);
    let page = |m: &Mgras, p: u32| {
        let _sub = m.submit.lock();
        m.wait_idle();
        let rss = unsafe { &*m.rss.get() };
        rss.mem.get(&super::pixmem::Buffer::new(p, super::pixmem::Kind::Wide, 0x31E), 50, 50) as u32 & 0xFF_FFFF
    };
    for (k, c) in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]].iter().enumerate() {
        gl_color4(&m, [c[0], c[1], c[2], 1.0]);
        gl_full_quad(&m);
        let want = (c[0] as u32) * 0xFF | (c[1] as u32) * 0xFF00 | (c[2] as u32) * 0xFF_0000;
        let (drawn, other) = if k % 2 == 0 { (0x140, 0x240) } else { (0x240, 0x140) };
        assert_eq!(page(&m, drawn), want, "frame {k} draws the back buffer");
        assert_ne!(page(&m, other), want, "frame {k} leaves the front alone");
        // The kernel's swap: the next frame's bank, then SCHEDULE_SWAP.
        let next = (k as u32 + 1) % 2 ^ 1;
        fifo_token(&m, 0x98, &[next, next]);
        write(&m, 32, CFIFO, ((0x37 << 8) | 0) as u64);
    }
    m.stop_engines();
}

/// The bank comes from the kernel (VALIDATE_BANKS), not from counting
/// swaps: a swap the kernel did not pair with a bank (or one the GE never
/// saw) leaves the drawing where the kernel last said.
#[test]
fn gl_draw_bank_follows_the_kernel_not_the_swap_count() {
    let m = gl_board([0.0, 0.0, 0.0]);
    fifo_token(&m, 0xE4, &[0, 0x11, 0, 0, 0, 0, 0, 0, 0, 399, 299, 0x240 | 0x140 << 10, 0, 0, 0]);
    fifo_token(&m, 0x49, &[4, 1, 0]);
    let page = |m: &Mgras, p: u32| {
        let _sub = m.submit.lock();
        m.wait_idle();
        let rss = unsafe { &*m.rss.get() };
        rss.mem.get(&super::pixmem::Buffer::new(p, super::pixmem::Kind::Wide, 0x31E), 50, 50) as u32 & 0xFF_FFFF
    };
    fifo_token(&m, 0x98, &[0, 0]);
    write(&m, 32, CFIFO, ((0x37 << 8) | 0) as u64);
    write(&m, 32, CFIFO, ((0x37 << 8) | 0) as u64);
    gl_color4(&m, [1.0, 0.0, 0.0, 1.0]);
    gl_full_quad(&m);
    assert_eq!(page(&m, 0x240), 0xFF, "bank 0: A, whatever the swaps");
    assert_ne!(page(&m, 0x140), 0xFF);
    fifo_token(&m, 0x98, &[1, 1]);
    gl_color4(&m, [0.0, 1.0, 0.0, 1.0]);
    gl_full_quad(&m);
    assert_eq!(page(&m, 0x140), 0xFF00, "bank 1: B");
    m.stop_engines();
}

/// Native IRIS GL clear() reads the current index with SPIN_AND_RETURN,
/// then uses the returned float for CLEAR_INDEX. Echoing address 4 reads
/// as a denormal/zero and erases Softimage's background and grid planes.
#[test]
fn gl_spin_and_return_reads_current_color_for_index_clear() {
    let m = gl_board([0.0, 0.0, 0.0]);
    let spin = |addr: u32| {
        write(&m, 32, 0x7000C, 1 << 17);
        fifo_token(&m, 0xA1, &[addr]);
        wait_flag(&m, 1 << 17);
        read(&m, 32, 0x70014) as u32
    };
    gl_color4(&m, [0.25, 0.5, 0.75, 1.0]);
    for (addr, value) in [(4, 0.25), (5, 0.5), (6, 0.75), (7, 1.0)] {
        assert_eq!(spin(addr), f(value));
    }
    fifo_token(&m, 0x0A, &[12]);
    for index in [21.0, 31.0, 19.0] {
        fifo_token(&m, 0x02, &[f(index)]);
        let got = spin(4);
        assert_eq!(got, f(index), "getcolor must return the index, not its address");
        fifo_token(&m, 0x3D, &[0x3F]);
        fifo_token(&m, 0xBD, &[got]);
        fifo_token(&m, 0x15, &[]);
        assert_eq!(gl_px(&m, 100, 100), index as u32, "background clear");
        fifo_token(&m, 0x3D, &[0x1C0]);
        fifo_token(&m, 0xBD, &[got]);
        fifo_token(&m, 0x15, &[]);
        assert_eq!(gl_px(&m, 100, 100), index as u32, "other planes preserve background");
    }
    m.stop_engines();
}
