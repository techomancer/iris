//! A fill-rate benchmark for the integer drawing path.
//!
//! Each scene draws into a store of the managed area's size through the same
//! calls the back end makes for a textport or X command, with no guest and no
//! framing in the way, so the numbers are the fills'. Run it from a release
//! build, one test thread:
//!
//!     cargo test --release --lib kona::back::bench -- --ignored --nocapture --test-threads=1
//!
//! `BENCH_SECS` sets the time each scene takes (default half a second). The
//! figure is the best of five rounds, in megapixels of the requested region
//! per second. The pixel transfers (an 8-bit image write, a copy, a read) are
//! scenes of their own, in megapixels of the image.

use super::raster::Surface;
use super::store::PixelStore;
use super::display::tables::Tables;
use super::display::{compose, compose_default, ColourMap};
use super::store::DID_PLANE;
use super::x::state::XState;
use super::x::pixel::{Next, X_PIXEL_COPY, X_PIXEL_READ, X_PIXEL_SETUP, X_PIXEL_WRITE};
use std::time::{Duration, Instant};

const W: u32 = 1280;
const H: u32 = 1024;
const GC_COPY: [u32; 13] = [0x5a000, 0x1ffe, 0xa5000, 0x1ffe, 0xf00fff, 0, 0x180c, 0x40001, 1, 6, 5, 1, 1];

fn pack(a: i32, b: i32) -> u32 {
    (a as u32) << 16 | (b as u32 & 0xffff)
}

fn secs() -> Duration {
    Duration::from_secs_f64(std::env::var("BENCH_SECS").ok().and_then(|v| v.parse().ok()).unwrap_or(0.5))
}

/// Run `f` (one call draws `pixels` pixels) for `secs()`, five rounds; the best
/// rate in megapixels a second.
fn rate(pixels: u64, mut f: impl FnMut()) -> f64 {
    let mut best: f64 = 0.0;
    for _ in 0..5 {
        let (start, mut calls) = (Instant::now(), 0u64);
        while start.elapsed() < secs() / 5 {
            f();
            calls += 1;
        }
        best = best.max((calls * pixels) as f64 / start.elapsed().as_secs_f64() / 1e6);
    }
    best
}

/// Whether any pixel of the store was drawn: a scene that draws nothing
/// would report a fast rate for no work.
fn drawn(store: &PixelStore) -> bool {
    (0..H).any(|y| (0..W).any(|x| store.read(x, y, 1) != 0))
}

fn x_scene(name: &str, setup: impl FnOnce(&mut XState), box_words: [u32; 2]) {
    let store = PixelStore::new(W, H);
    let surface = Surface::new(&store, 1);
    let mut x = XState::default();
    x.command(0x129, &GC_COPY);
    setup(&mut x);
    let (w, h) = ((box_words[1] >> 16) as i64 - (box_words[0] >> 16) as i64, (box_words[1] as u16 as i64) - (box_words[0] as u16 as i64));
    // One draw first: an operation like xor undoes itself every second time, so
    // the store is only checked after the first.
    x.draw_command(&surface, 0x154, &box_words);
    assert!(drawn(&store), "{name} drew nothing");
    let mpix = rate((w * h) as u64, || {
        x.draw_command(&surface, 0x154, &box_words);
    });
    println!("{name:<44} {mpix:>9.1} Mpix/s");
}

/// An 8-bit image `w` x `h` written at (0, 0) through the setup, the write and
/// its rows, as the back end takes them.
fn write_scene(name: &str, w: u32, h: u32) {
    let store = PixelStore::new(W, H);
    let surface = Surface::new(&store, 1);
    let mut x = XState::default();
    x.command(0x129, &GC_COPY);
    let rows: Vec<u32> = (0..w * h / 4).map(|i| i.wrapping_mul(0x0101_0101) ^ 0x0102_0304).collect();
    let mut once = || {
        x.pixel_command(&surface, X_PIXEL_SETUP, &[0, 0, 1, 8, w, h, 0], true);
        assert_eq!(x.pixel_command(&surface, X_PIXEL_WRITE, &[0, 0], true), Next::Words(w * h / 4));
        x.pixel_rows(&surface, &rows);
    };
    once();
    assert!(drawn(&store), "{name} drew nothing");
    let mpix = rate((w * h) as u64, once);
    println!("{name:<44} {mpix:>9.1} Mpix/s");
}

/// A copy of the screen's interior four pixels down and right.
fn copy_scene(name: &str, gx: u32) {
    let store = PixelStore::new(W, H);
    let surface = Surface::new(&store, 1);
    let mut x = XState::default();
    let mut gc = GC_COPY;
    gc[9] = gx * 2;
    x.command(0x129, &gc);
    for y in 0..H {
        for c in 0..W {
            store.write(c, y, 1, (c ^ y) << 8 & 0xff00);
        }
    }
    let (w, h) = (W - 8, H - 8);
    x.pixel_command(&surface, X_PIXEL_SETUP, &[1, 1, 1, 0, w, h, 0], true);
    x.pixel_command(&surface, X_PIXEL_COPY, &[0, 0, 4, 4], true);
    assert_eq!(x.pixel_counts().copies, 1, "{name} copied nothing");
    let mpix = rate((w * h) as u64, || {
        x.pixel_command(&surface, X_PIXEL_COPY, &[0, 0, 4, 4], true);
    });
    println!("{name:<44} {mpix:>9.1} Mpix/s");
}

/// A 64 x 64 read of 8-bit pixels (the high byte of each index).
fn read_scene(name: &str) {
    let store = PixelStore::new(W, H);
    let surface = Surface::new(&store, 1);
    let mut x = XState::default();
    let mut gc = [0u32; 13];
    gc[6..].copy_from_slice(&[0x180c, 0x40001, 1, 6, 5, 1, 1]);
    x.command(0x129, &gc);
    store.write(100, 100, 1, 0x1200);
    x.pixel_command(&surface, X_PIXEL_SETUP, &[2, 1, 0, 8, 64, 64, 0], true);
    let mut once = || match x.pixel_command(&surface, X_PIXEL_READ, &[90, 90], true) {
        Next::Return(d) => assert_eq!(d.len(), 64 * 64),
        other => panic!("{other:?}"),
    };
    once();
    let mpix = rate(64 * 64, once);
    println!("{name:<44} {mpix:>9.1} Mpix/s");
}

/// The frame composed from the store, in megapixels of the frame: the default
/// (textport) composition, and with the X server's tables loaded and the
/// screen in runs of 16 pixels of four windows (8-bit, 12-bit, direct, cleared).
fn compose_scene(name: &str, tables: Option<&Tables>, run: u32) {
    let store = PixelStore::new(W, H);
    let mut map = ColourMap::default();
    for i in 0..256u32 {
        map.load(&[i, i * 4, 1020 - i * 4, i * 2]);
    }
    // Windows in runs: `run` pixels each, or, for 0, runs of 150 to 700 pixels
    // as a desktop's windows are.
    let mut seed = 7u64;
    let mut next = |lo: u32, hi: u32| {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        lo + (seed >> 33) as u32 % (hi - lo)
    };
    for y in 0..H {
        let (mut left, mut id) = (0u32, 1u32);
        for c in 0..W {
            if left == 0 {
                left = if run == 0 { next(150, 700) } else { run };
                id = id % 4 + 1;
            }
            left -= 1;
            store.write(c, y, 1, (c * 7 + y * 13) & 0xffff);
            if tables.is_some() {
                store.write(c, y, DID_PLANE, id << 2);
            }
        }
    }
    let mut out = vec![0u32; (W * H) as usize];
    let snap = tables.map(|t| t.snapshot());
    let mpix = rate((W * H) as u64, || match &snap {
        Some(t) => compose(&store, 1, &map, t, &mut out, W as usize),
        None => compose_default(&store, 1, &map, &mut out, W as usize),
    });
    assert!(out.iter().any(|&p| p != out[0]), "{name} composed one colour");
    println!("{name:<44} {mpix:>9.1} Mpix/s");
}

/// What the display thread takes of the tables each frame, per second, for a
/// board with nothing loaded and for one with the tables loaded.
fn frame_tables_scene(name: &str, t: &Tables) {
    let mut sink = 0usize;
    let rate = rate(1, || sink += std::hint::black_box(t).snapshot().colour(0x2c01) as usize);
    println!("{name:<44} {:>9.1} kframes/s", rate * 1000.0);
    assert!(sink > 0 || t.snapshot().is_default());
}

fn tables_for_compose() -> Tables {
    let mut t = Tables::default();
    t.load_colours(0x2c00, &(0..256u32).map(|i| i << 20 | (255 - i) << 10 | i).collect::<Vec<_>>());
    t.load_colours(0x3800, &(0..4096u32).map(|i| (i & 0x3ff) << 20 | (i >> 2) << 10).collect::<Vec<_>>());
    t.load_did(1, 0x2c00 << 7 | 7 << 3);
    t.load_did(2, 0x3800 << 7 | 11 << 3);
    t.load_mode(3, 4);
    t.load_did(4, 0);
    t.load_gamma(0, &(0..4096).collect::<Vec<_>>());
    t
}

#[test]
#[ignore]
fn fill_rate() {
    let full = [pack(0, 0), pack(W as i32, H as i32)];
    x_scene("x box, solid, whole screen", |_| {}, full);
    x_scene("x box, solid, 64 x 64", |_| {}, [pack(100, 100), pack(164, 164)]);
    x_scene(
        "x box, two clip rectangles",
        |x| {
            x.command(0x145, &[2, 0, 0, W as u32 / 2 - 8]);
            x.command(0x12b, &[H, W as u32 / 2 + 8, 0, W, H]);
        },
        full,
    );
    x_scene(
        "x box, 8 x 8 stipple, whole screen",
        |x| {
            x.command(0x13d, &[1, 8, 8]);
            x.stipple.rows(&[0xaa00_0000, 0x5500_0000, 0xaa00_0000, 0x5500_0000, 0xf000_0000, 0x0f00_0000, 0xf000_0000, 0x0f00_0000]);
            x.command(0x13c, &[1]);
        },
        full,
    );
    x_scene(
        "x box, xor (reads the destination)",
        |x| {
            let mut gc = GC_COPY;
            gc[9] = 12; // gx 6, xor
            x.command(0x129, &gc);
        },
        full,
    );

    let store = PixelStore::new(W, H);
    let surface = Surface::new(&store, 1);
    let mpix = rate((W * H) as u64, || surface.fill_box(0, 0, W as i64, H as i64, 7));
    assert!(drawn(&store), "the textport box drew nothing");
    println!("{:<44} {mpix:>9.1} Mpix/s", "textport box, whole screen");
    // 16 x 16 glyphs at scattered positions, half the bits set.
    let bits: Vec<u32> = (0..8).map(|i| 0xa5a5_5a5a ^ (i * 0x0101_0101)).collect();
    let mut seed = 1u64;
    let mut place = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as i64 % (W as i64 - 16), (seed >> 11) as i64 % (H as i64 - 16))
    };
    let mpix = rate(16 * 16 * 64, || {
        for _ in 0..64 {
            let (x, y) = place();
            surface.bitmap(x, y, 16, 16, &bits, 9);
        }
    });
    assert!(drawn(&store), "the glyphs drew nothing");
    println!("{:<44} {mpix:>9.1} Mpix/s", "textport bitmap, 16 x 16 glyphs");

    write_scene("x image write, 8 bit, 64 x 64", 64, 64);
    write_scene("x image write, 8 bit, whole screen", W, H);
    copy_scene("x copy, whole screen less 8, copy", 3);
    copy_scene("x copy, whole screen less 8, xor", 6);
    read_scene("x read, 8 bit, 64 x 64");
    compose_scene("compose, default", None, 16);
    compose_scene("compose, X windows, runs of 16", Some(&tables_for_compose()), 16);
    compose_scene("compose, X windows, runs of 150 to 700", Some(&tables_for_compose()), 0);
    frame_tables_scene("frame's tables, nothing loaded", &Tables::default());
    frame_tables_scene("frame's tables, loaded", &tables_for_compose());
}
