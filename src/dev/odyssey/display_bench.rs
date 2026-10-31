//! The scanout resolve: frames it must produce exactly, and a benchmark of
//! how long it takes.
//!
//! Each scene is a board loaded with deterministic planes, lookup RAM,
//! display modes, overlay and cursor, then resolved whole. The tests hold
//! the resolved frame to a stored hash, so a change to the resolve that moves
//! one pixel fails. The benchmark reports the resolve of a full 1280x1024
//! screen, a whole display frame, a swap of a window that differs from its other buffer in part of
//! the screen, and the conversion of that frame into the window's buffers:
//!
//!     cargo test --release --lib odyssey::display_bench -- --ignored --nocapture --test-threads=1
//!
//! BENCH_SECS is the time each case takes (default one second).

use super::*;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::time::{Duration, Instant};

use raster_bench::Lcg;

/// What a scene puts on the screen.
#[derive(Clone, Copy)]
enum Scene {
    /// Every drawable an 8-bit RGB visual; no overlay, no cursor.
    Rgb,
    /// Drawables in every kind of mode on both colour buffers, an overlay
    /// that covers some pixels, and the glyph cursor.
    Mixed,
    /// As Mixed, with the crosshair cursor in place of the glyph.
    Crosshair,
    /// Mixed, in a smaller mode than the planes.
    Small,
}

const SCENES: [(Scene, &str); 4] =
    [(Scene::Rgb, "rgb"), (Scene::Mixed, "mixed"), (Scene::Crosshair, "crosshair"), (Scene::Small, "small")];

fn fill_random(bytes: &mut [u8], rng: &mut Lcg) {
    for c in bytes.chunks_exact_mut(4) {
        c.copy_from_slice(&rng.next_u32().to_be_bytes());
    }
}

fn scene_board(scene: Scene) -> Box<Board> {
    let mut s = Board::new("v12", 11).expect("board");
    let mut rng = Lcg(0x5eed_0001);
    let pixels = (SGI_ODYSSEY_WIDTH * SGI_ODYSSEY_HEIGHT) as usize;

    fill_random(&mut s.mem.bytes, &mut rng);
    fill_random(&mut s.buffer1_mem.bytes, &mut rng);
    for v in s.lut.iter_mut() {
        *v = rng.next_u32() & ODSY_LUT_ENTRY_MASK;
    }
    if let Scene::Rgb = scene {
        // The identity-like ramp a guest loads, so the picture is not noise.
        for i in 0..0x600u32 {
            let v = (i * 0x3ff / 0x5ff) & 0x3ff;
            s.lut[(ODSY_GAMMA_BASE + i) as usize] = v << 20 | v << 10 | v;
        }
        for id in 0..ODSY_DISPMODE_COUNT {
            s.dispmode[id as usize] = 26 << ODSY_DISPMODE_GROUP_SHIFT;
        }
        return s;
    }
    // Drawable modes: 8-bit RGB, 4-bit, 5-bit, index groups 17 and 0 and 25,
    // another RGB group, repeating over the 32 ids.
    for id in 0..ODSY_DISPMODE_COUNT {
        s.dispmode[id as usize] = match id % 7 {
            0 => 26 << ODSY_DISPMODE_GROUP_SHIFT,
            1 => 27 << ODSY_DISPMODE_GROUP_SHIFT | ODSY_DISPMODE_WIDTH_4 << ODSY_DISPMODE_WIDTH_SHIFT,
            2 => 28 << ODSY_DISPMODE_GROUP_SHIFT | ODSY_DISPMODE_WIDTH_5 << ODSY_DISPMODE_WIDTH_SHIFT,
            3 => 17 << ODSY_DISPMODE_GROUP_SHIFT,
            4 => 0,
            5 => ODSY_CMAP_UNVERIFIED_GROUP << ODSY_DISPMODE_GROUP_SHIFT,
            _ => 31 << ODSY_DISPMODE_GROUP_SHIFT | 1 << ODSY_DISPMODE_WIDTH_SHIFT,
        };
    }
    for id in 0..ODSY_OVERLAY_IDS {
        // Ids 0, 3, 5 and 6 show the overlay, from different groups.
        s.overlay_mode[id as usize] = if matches!(id, 0 | 3 | 5 | 6) {
            ODSY_OVERLAY_MODE_ENABLE | (id + 20) << ODSY_OVERLAY_MODE_GROUP_SHIFT
        } else {
            id << ODSY_OVERLAY_MODE_GROUP_SHIFT
        };
    }
    // Windows: 64-pixel blocks of one drawable id, each with an overlay id;
    // the overlay is set on about one pixel in five.
    for y in 0..SGI_ODYSSEY_HEIGHT as usize {
        for x in 0..SGI_ODYSSEY_WIDTH as usize {
            let at = y * SGI_ODYSSEY_WIDTH as usize + x;
            let block = (x / 64 + 3 * (y / 48)) as u32;

            s.did[at] = ((block % 32) << ODSY_DID_MAIN_SHIFT | (block / 3) % 8) as u8;
            s.overlay[at] = if rng.next_u32() % 5 == 0 { rng.next_u32() as u8 } else { 0 };
        }
    }
    assert_eq!(s.did.len(), pixels);
    s.front.lock().displayed_buffer = 0x6d5a_b3c9;
    s.cursor[ODSY_CURSOR_POSITION as usize] = 100 << 16 | 40;
    for i in 0..ODSY_CURSOR_COLOUR_COUNT {
        s.cursor[(ODSY_CURSOR_COLOURS + 1 + i) as usize] = rng.next_u32() & ODSY_LUT_ENTRY_MASK;
    }
    for w in 0..64u32 {
        s.cursor[(ODSY_CURSOR_PLANE + w) as usize] = rng.next_u32();
    }
    for w in 0..128u32 {
        s.cursor[(ODSY_CURSOR_GLYPH + w) as usize] = rng.next_u32();
    }
    s.cursor[ODSY_CURSOR_CONTROL as usize] =
        if let Scene::Crosshair = scene { ODSY_CURSOR_CTL_ENABLE | ODSY_CURSOR_CTL_CROSSHAIR } else { ODSY_CURSOR_CTL_ENABLE };
    if let Scene::Small = scene {
        s.set_scanout_size(1000, 700);
    }
    s
}

/// An RGB screen whose two colour buffers differ in the `rows` rows from the
/// middle out: what a swap of a window that size changes.
fn swap_board(rows: usize) -> Box<Board> {
    let mut s = scene_board(Scene::Rgb);
    let stride = SGI_ODYSSEY_WIDTH as usize * 4;
    let mut rng = Lcg(0x5eed_0002);
    let first = (SGI_ODYSSEY_HEIGHT as usize - rows) / 2;

    let same = s.mem.bytes.clone();
    s.buffer1_mem.bytes.copy_from_slice(&same);
    fill_random(&mut s.buffer1_mem.bytes[first * stride..(first + rows) * stride], &mut rng);
    s
}

fn fnv(frame: &[u32]) -> u64 {
    frame.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &p| (h ^ p as u64).wrapping_mul(0x100_0000_01b3))
}

/// The frame of a scene, resolved whole.
fn resolved(scene: Scene) -> Vec<u32> {
    let mut s = scene_board(scene);

    s.scanout.invalidate_all();
    assert!(s.scanout_update().is_some());
    s.scanout.frame.clone()
}

/// Stored from the per-pixel resolve this change replaced.
const EXPECTED: [u64; 4] = [0x811e_c4e1_b153_56b7, 0x48e6_3c81_0278_8e71, 0x875a_cf9e_98b9_71c3, 0x3de4_d246_d889_9652];

#[test]
fn scenes_resolve_to_the_stored_frames() {
    let got: Vec<u64> = SCENES.iter().map(|(scene, _)| fnv(&resolved(*scene))).collect();

    assert_eq!(got, EXPECTED, "hashes {got:#018x?}");
}

/// The rows resolve_row() makes are the per-pixel resolve's, on every scene
/// and at a few lines of each, with the board left as the scene made it.
#[test]
fn the_row_resolve_matches_the_per_pixel_resolve() {
    for (scene, name) in SCENES {
        let mut s = scene_board(scene);
        let w = s.scanout.mode.width as usize;
        let mut fast = vec![0u32; w];
        let mut slow = vec![0u32; w];

        for y in 0..s.scanout.mode.height {
            s.resolve_row(y, &mut fast);
            s.resolve_row_generic(y, &mut slow);
            assert_eq!(fast, slow, "scene {name}, line {y}");
        }
    }
}

/// A line the planes do not cover whole is resolved per pixel, not skipped.
#[test]
fn a_mode_wider_than_the_planes_still_resolves() {
    let mut s = scene_board(Scene::Mixed);
    let mut fast = vec![0u32; 1280];
    let mut slow = vec![0u32; 1280];

    s.mem.bytes.truncate(1280 * 4 * 10);
    s.resolve_row(9, &mut fast);
    s.resolve_row_generic(9, &mut slow);
    assert_eq!(fast, slow);
    s.resolve_row(500, &mut fast);
    s.resolve_row_generic(500, &mut slow);
    assert_eq!(fast, slow, "a line past the end of the board memory");
}

/// The window's buffers hold each pixel as opaque 0xAABBGGRR, in both copies,
/// for the lines presented and nothing else.
#[test]
fn present_swaps_each_pixel_into_both_buffers() {
    let display = Display::new(Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)));
    let (w, h) = (SGI_ODYSSEY_WIDTH as usize, SGI_ODYSSEY_HEIGHT as usize);
    let frame = resolved(Scene::Mixed);
    let rows = &frame[100 * w..300 * w];

    display.present(rows, w, h, 100);
    let screen = display.screen.lock();
    assert_eq!((screen.width, screen.height), (w, h));
    for y in 0..h {
        for x in 0..w {
            let want = if (100..300).contains(&y) {
                let rgb = frame[y * w + x];
                0xff00_0000 | (rgb >> 16) & 0xff | rgb & 0xff00 | (rgb & 0xff) << 16
            } else {
                0
            };
            let at = y * crate::disp::FB_STRIDE + x;
            assert_eq!((screen.fb_rgb[at], screen.rgba[at]), (want, want), "({x}, {y})");
        }
    }
}

/// The fastest of the runs of `f` in the time budget, in ms: a shared host's
/// load only ever slows a run.
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
fn resolve_and_present_timings() {
    for (scene, name) in SCENES {
        let mut s = scene_board(scene);
        let ms = bench(|| {
            s.scanout.invalidate_all();
            s.scanout_update();
        });
        println!("resolve {name:10} {}x{}: {ms:.3} ms per full resolve (best run)", s.scanout.mode.width, s.scanout.mode.height);
    }
    for (scene, name) in SCENES {
        let mut s = scene_board(scene);
        let ctx = s.resolve_ctx();
        let (w, h) = (s.scanout.mode.width as usize, s.scanout.mode.height);
        let width = s.scanout.mode.width.min(SGI_ODYSSEY_WIDTH) as usize;
        let mut dst = vec![0u32; w];
        let ms = bench(|| {
            for y in 0..h {
                if let Some(line) = s.scanout_line(y, width) {
                    std::hint::black_box(resolve_line(&ctx, line, &mut dst));
                }
            }
        });
        println!("lines  {name:10} {w}x{h}: {ms:.3} ms to resolve the planes' lines alone");
    }
    for (rows, name) in [(480, "a 480-row window"), (1024, "every row")] {
        let mut s = swap_board(rows);
        s.scanout_update();
        let ms = bench(|| {
            s.front.lock().swap_arm = u32::MAX;
            s.retrace();
            s.scanout_update();
        });
        println!("swap {name:18} {}x{}: {ms:.3} ms per swap and update", s.scanout.mode.width, s.scanout.mode.height);
    }
    for (rows, name) in [(480, "480 lines"), (1024, "every line")] {
        let locks = BoardLocks::new(scene_board(Scene::Rgb));
        let display = Display::new(Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)));
        let first = (SGI_ODYSSEY_HEIGHT as i32 - rows) / 2;

        Odyssey::frame(&locks, &display);
        let ms = bench(|| {
            locks.board.lock().scanout.invalidate(first, rows);
            Odyssey::frame(&locks, &display);
        });
        println!("frame {name:18} dirty, resolved and presented: {ms:.3} ms");
    }
    let display = Display::new(Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)));
    let frame = resolved(Scene::Mixed);
    let (w, h) = (SGI_ODYSSEY_WIDTH as usize, SGI_ODYSSEY_HEIGHT as usize);
    let ms = bench(|| display.present(&frame, w, h, 0));
    println!("present {w}x{h}: {ms:.3} ms per full frame");
}
