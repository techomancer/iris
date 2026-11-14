//! The scan-out composition (frame.rs): checks on scenes in board memory,
//! and a benchmark of how long a whole frame takes.
//!
//! Each scene is a board as the X server leaves it (Board::test_screen()),
//! its frame buffer and window-id buffer filled from a fixed seed, the lookup
//! RAM, display modes of the kinds X sets up, overlay and cursor:
//!
//!     cargo test --release --lib odyssey::display_bench -- --ignored --nocapture --test-threads=1
//!
//! BENCH_SECS is the time each case takes (default one second).

use super::{raster_bench, Odyssey};
use super::board::Board;
use super::consts::{cursor, dispmode, lut, overlay, ODSY_DID_MAIN_SHIFT, ODSY_GAMMA_BASE};
use super::display::Display;
use super::frame::{pbj, Frame};
use super::mmio::BoardLocks;
use super::vram::Surface;
use std::time::{Duration, Instant};

use raster_bench::Lcg;

/// What a scene puts on the screen.
#[derive(Clone, Copy)]
enum Scene {
    /// Every window an 8:8:8:8 single-buffered one; no overlay, no cursor.
    Rgb,
    /// Windows of every kind X sets up, both buffers shown, an overlay over
    /// some pixels, and the glyph cursor.
    Mixed,
    /// As Mixed, with the crosshair cursor.
    Crosshair,
    /// Mixed, scanned out at 1024 x 768 from the same frame buffer.
    Small,
}

const SCENES: [(Scene, &str); 4] =
    [(Scene::Rgb, "rgb"), (Scene::Mixed, "mixed"), (Scene::Crosshair, "crosshair"), (Scene::Small, "small")];

/// The X server's display modes on a 16-byte frame buffer (measured.md
/// section 9): 8:8:8:8 single and double-buffered, 4:4:4:4, 5:5:5:1,
/// 12-bit, colour index 8 and 12.
const MODES: [u32; 7] = [0x0052_15a6, 0x0096_55a6, 0x004b_59a6, 0x004b_57a6, 0x0060_0da6, 0x004b_5f10, 0x0052_0e00];

fn scene_board(scene: Scene) -> Box<Board> {
    let mut s = Board::new("v12", 11).expect("board");
    let mut rng = Lcg(0x5eed_0001);

    s.test_screen();
    // The frame buffer and the window-id buffer, from the seed.
    let fb_end = (0x14_2000 + 0x2_8000) * 16;
    let mem = s.vram.ptr();
    for a in (0x2000 * 16..fb_end).step_by(4) {
        mem.store(a as u64, 4, rng.next_u32() as u64);
    }
    for i in 0..lut::SIZE as usize {
        s.regs().set_lut(i, rng.next_u32() & lut::ENTRY_MASK);
    }
    for i in 0..0x600u32 {
        let v = (i * 0x3ff / 0x5ff) & 0x3ff;
        s.regs().set_lut((ODSY_GAMMA_BASE + i) as usize, v << 20 | v << 10 | v);
    }
    let wid = Surface::new(0x14_2000, 80, 1);
    let rgb = matches!(scene, Scene::Rgb);
    for id in 0..dispmode::COUNT as usize {
        s.regs().set_dispmode(id, if rgb { MODES[0] } else { MODES[id % MODES.len()] });
    }
    for id in 0..overlay::IDS {
        let mode = if !rgb && matches!(id, 0 | 3 | 5 | 6) {
            overlay::MODE_ENABLE | (id + 20) << overlay::MODE_GROUP_SHIFT
        } else {
            id << overlay::MODE_GROUP_SHIFT
        };
        s.regs().set_overlay_mode(id as usize, mode);
    }
    // Windows: 64-pixel blocks of one window id, each with an overlay id;
    // the overlay byte set on about one pixel in five.
    for y in 0..1024u32 {
        for x in 0..1280u32 {
            let block = x / 64 + 3 * (y / 48);
            let at = wid.offset(x, y);
            let mem = s.vram.ptr();
            mem.store(at, 1, if rgb { 0 } else { ((block % 32) << ODSY_DID_MAIN_SHIFT | (block / 3) % 8) as u64 });
            mem.store(at + 1, 1, if !rgb && rng.next_u32() % 5 == 0 { rng.next_u32() as u8 as u64 } else { 0 });
        }
    }
    if rgb {
        return s;
    }
    s.front.lock().displayed_buffer = 0x6d5a_b3c9;
    s.regs().set_cursor(cursor::POSITION as usize, 100 << 16 | 40);
    for i in 0..cursor::COLOUR_COUNT {
        s.regs().set_cursor((cursor::COLOURS + 1 + i) as usize, rng.next_u32() & lut::ENTRY_MASK);
    }
    for w in 0..64u32 {
        s.regs().set_cursor((cursor::PLANE + w) as usize, rng.next_u32());
    }
    for w in 0..128u32 {
        s.regs().set_cursor((cursor::GLYPH + w) as usize, rng.next_u32());
    }
    let control = if let Scene::Crosshair = scene { cursor::CTL_ENABLE | cursor::CTL_CROSSHAIR } else { cursor::CTL_ENABLE };
    s.regs().set_cursor(cursor::CONTROL as usize, control);
    if let Scene::Small = scene {
        s.regs().set_pbj(pbj::WIDTH, 1024);
        s.regs().set_pbj(pbj::HEIGHT, 768);
    }
    s
}

/// The scene composed whole.
fn composed(s: &Board) -> Vec<u32> {
    let regs = s.regs().read();
    let frame = Frame::snapshot(&regs, s.front.lock().displayed_buffer).expect("the scan-out is set up");
    let mut out = vec![0u32; frame.width as usize * frame.height as usize];
    frame.compose(s.vram.ptr_shared(), &regs, &mut out, frame.width as usize, frame.height as usize);
    out
}

#[test]
fn scenes_compose_the_same_frame_every_time() {
    for (scene, name) in SCENES {
        let s = scene_board(scene);
        let a = composed(&s);
        assert_eq!(a, composed(&s), "{name}");
        assert!(a.iter().any(|&p| p != a[0]), "{name}: a flat frame");
    }
}

/// The scene composed pixel by pixel through `Frame::pixel`, the reference.
fn reference(s: &Board) -> Vec<u32> {
    let regs = s.regs().read();
    let frame = Frame::snapshot(&regs, s.front.lock().displayed_buffer).expect("the scan-out is set up");
    let (w, h) = (frame.width as usize, frame.height as usize);
    let mem = s.vram.ptr_shared();
    let mut out = vec![0u32; w * h];
    for y in 0..h {
        let row = &mut out[y * w..(y + 1) * w];
        for (x, o) in row.iter_mut().enumerate() {
            *o = frame.pixel(mem, x as u32, y as u32);
        }
        regs.cursor_row(y as u32, row, w as u32);
    }
    out
}

#[test]
fn compose_matches_the_pixel_reference() {
    for (scene, name) in SCENES {
        let s = scene_board(scene);
        let want = reference(&s);
        assert!(composed(&s) == want, "{name}");
        // The display thread's order: the same pixels, as the screen holds them.
        let regs = s.regs().read();
        let frame = Frame::snapshot_for_screen(&regs, s.front.lock().displayed_buffer).unwrap();
        let mut out = vec![0u32; want.len()];
        frame.compose(s.vram.ptr_shared(), &regs, &mut out, frame.width as usize, frame.height as usize);
        assert!(out.iter().zip(&want).all(|(&o, &w)| o == super::frame::to_screen(w)), "{name}: screen order");
    }
    // A width that ends inside a tile, and spans of one window id with an
    // overlay byte whose overlay id is disabled (scene_board enables 0, 3, 5
    // and 6 only).
    let s = scene_board(Scene::Mixed);
    s.regs().set_pbj(pbj::WIDTH, 1000);
    s.regs().set_pbj(pbj::HEIGHT, 700);
    let wid = Surface::new(0x14_2000, 80, 1);
    let mem = s.vram.ptr_shared();
    for x in 0..64u32 {
        mem.store(wid.offset(320 + x, 5), 1, (7 << ODSY_DID_MAIN_SHIFT | 1) as u64);
        mem.store(wid.offset(320 + x, 5) + 1, 1, 0x42);
    }
    assert!(composed(&s) == reference(&s), "partial width");
}

#[test]
fn a_smaller_scan_out_composes_the_corner_of_the_frame_buffer() {
    let big = composed(&scene_board(Scene::Mixed));
    let small = composed(&scene_board(Scene::Small));
    assert_eq!(small.len(), 1024 * 768);
    for y in [0usize, 100, 767] {
        // Away from the cursor, which both draw at the same place.
        assert_eq!(small[y * 1024 + 500..y * 1024 + 900], big[y * 1280 + 500..y * 1280 + 900], "row {y}");
    }
}

fn bench<F: FnMut()>(mut f: F) -> f64 {
    let secs: f64 = std::env::var("BENCH_SECS").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
    let budget = Duration::from_secs_f64(secs);
    let start = Instant::now();
    let mut best = f64::MAX;

    f();
    while start.elapsed() < budget {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    best
}

#[test]
#[ignore = "benchmark"]
fn compose_and_present_timings() {
    for (scene, name) in SCENES {
        let s = scene_board(scene);
        // As the display thread runs it: a snapshot of the registers, then
        // the frame into a buffer it keeps.
        let regs = s.regs().read();
        let displayed = s.front.lock().displayed_buffer;
        let mem = s.vram.ptr_shared();
        let (w, h) = { let f = Frame::snapshot(&regs, displayed).unwrap(); (f.width as usize, f.height as usize) };
        let mut out = vec![0u32; w * h];
        let ms = bench(|| {
            let frame = Frame::snapshot_for_screen(&regs, displayed).unwrap();
            frame.compose(mem, &regs, &mut out, w, h);
            std::hint::black_box(&out);
        });
        // The floor: the memory the frame reads, read in order and summed.
        let (fb, wid) = Frame::snapshot(&regs, displayed).unwrap().surfaces();
        let span = |f: Surface| (f.offset(0, 0), ((h as u64).div_ceil(16) * f.pitch as u64 * 256) << f.bpp_log2);
        let runs = [span(fb), span(wid)];
        let read = bench(|| {
            let mut sum = 0u64;
            for &(a, n) in &runs {
                sum = sum.wrapping_add(mem.words(a, n as usize / 8).unwrap().iter().fold(0u64, |s, &w| s.wrapping_add(w)));
            }
            std::hint::black_box(sum);
        });
        let mb = runs.iter().map(|r| r.1).sum::<u64>() as f64 / 1e6;
        println!("memory    {name:10}: {read:.3} ms to read its {mb:.1} MB in order ({:.1} GB/s)", mb / read);
        let snap = bench(|| {
            std::hint::black_box(Frame::snapshot(&regs, displayed));
        });
        println!("snapshot {name:10}: {snap:.3} ms");
        let old = bench(|| {
            std::hint::black_box(reference(&s));
        });
        println!("compose {name:10}: {ms:.3} ms per whole frame (snapshot included, best run), pixel by pixel {old:.3} ms");
    }
    let locks = BoardLocks::new(scene_board(Scene::Mixed));
    let display = Display::new(crate::disp::StatusSource::detached());
    // A client that reads every frame, or the frames are not composed.
    crate::gfx_display::GfxDisplay::set_capture(&display, true);
    let ms = bench(|| Odyssey::frame(&locks, &display));
    println!("frame: composed and presented in {ms:.3} ms");
}

/// Captures of real screens (`vpro scanout save`), composed new and pixel
/// by pixel: the frames must match, and the timings are printed.
///
///     VPRO_SCANOUT=a.scan,b.scan cargo test --release --lib odyssey::display_bench::compose_captures -- --ignored --nocapture
#[test]
#[ignore = "needs VPRO_SCANOUT"]
fn compose_captures() {
    use super::frame::Capture;
    use super::vram::Vram;
    let Ok(list) = std::env::var("VPRO_SCANOUT") else {
        println!("compose_captures: set VPRO_SCANOUT=file,file,... to compose captures");
        return;
    };
    for path in list.split(',') {
        let c = Capture::load(&std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"))).unwrap_or_else(|e| panic!("{path}: {e}"));
        let mut vram = Vram::new(c.size);
        let mem = vram.ptr();
        for (start, words) in &c.runs {
            for (i, &w) in words.iter().enumerate() {
                mem.store(start + 8 * i as u64, 8, w);
            }
        }
        let frame = Frame::snapshot(&c.regs, c.displayed).expect("the scan-out is set up");
        let (w, h) = (frame.width as usize, frame.height as usize);
        let mut out = vec![0u32; w * h];
        let ms = bench(|| {
            let frame = Frame::snapshot(&c.regs, c.displayed).unwrap();
            frame.compose(mem, &c.regs, &mut out, w, h);
        });
        let mut want = vec![0u32; w * h];
        let old = bench(|| {
            for y in 0..h {
                let row = &mut want[y * w..(y + 1) * w];
                for (x, o) in row.iter_mut().enumerate() {
                    *o = frame.pixel(mem, x as u32, y as u32);
                }
                c.regs.cursor_row(y as u32, row, w as u32);
            }
        });
        println!("{path}: {w}x{h}: compose {ms:.3} ms (snapshot included), pixel by pixel {old:.3} ms");
        assert!(out == want, "{path}: the frames differ");
        if let Ok(dir) = std::env::var("VPRO_SCANOUT_PNG") {
            let px: Vec<u32> = out.iter().map(|&p| (p & 0xff) << 16 | (p & 0xff00) | p >> 16).collect();
            let name = std::path::Path::new(path).file_stem().unwrap().to_string_lossy().into_owned();
            crate::hostinput::save_png(&format!("{dir}/{name}.png"), &px, w, h, None).unwrap();
        }
        drop(vram);
    }
}
