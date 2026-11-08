//! The model's side of test/gltest/linetest.c: the same lines drawn
//! through the board's own GL draw, printed in the guest program's format,
//! to compare with a real board's output (ignore/linetest/).
//!
//!     LINETEST_OUT=<dir> cargo test --release --lib odyssey::golden::linetest -- --ignored
//!
//! writes `model.txt` (the angle cases) and `model-fine.txt` (`fine`).
//! A frame's rows run up the GL window, as glReadPixels returns them.

use super::scenes::{v, Ctx, WHITE};
use super::{Frame, SIDE};
use crate::dev::odyssey::pipe::encode::{GL_LINES, GL_SRC_ALPHA};
use std::fmt::Write as _;

const GL_ONE: u32 = 1;
const ANGLES: [f64; 8] = [0.0, 5.0, 15.0, 26.565051, 35.0, 45.0, 63.434949, 90.0];

fn ctx(width: f32, smooth: bool) -> Ctx {
    let mut c = Ctx::new();
    c.clear([0.0, 0.0, 0.0, 0.0], 0xff_ffff, 0);
    c.line_width = width;
    if smooth {
        c.line_smooth = true;
        c.blend(GL_SRC_ALPHA, GL_ONE);
    }
    c
}

/// Window pixel (x, y) of the frame, y up the window, as 0xRRGGBBAA:
/// plane row y is GL window row y.
fn at(f: &Frame, x: i32, y: i32) -> u32 {
    f.colour[(y * SIDE + x) as usize]
}

#[test]
#[ignore]
fn the_model_draws_linetest() {
    let Ok(dir) = std::env::var("LINETEST_OUT") else {
        eprintln!("SKIPPED: set LINETEST_OUT");
        return;
    };
    let mut out = String::new();
    for (case, width, smooth) in [(0, 1.0, false), (1, 2.0, false), (2, 3.0, false), (3, 1.0, true), (4, 2.0, true)] {
        for (cell, angle) in ANGLES.iter().enumerate() {
            let mut c = ctx(width, smooth);
            let a = angle.to_radians();
            let (x0, y0) = (20.3f64, 20.6f64);
            let (x1, y1) = (x0 + 60.0 * a.cos(), y0 + 60.0 * a.sin());
            c.draw(GL_LINES, &[v(x0 as f32, y0 as f32, 0.0, WHITE), v(x1 as f32, y1 as f32, 0.0, WHITE)]);
            let f = Frame::of(&c.board);
            let (ox, oy) = ((cell % 4) as i32 * 128, (cell / 4) as i32 * 128);
            for y in 0..SIDE {
                for x in 0..SIDE {
                    let p = at(&f, x, y);
                    let (r, g, b) = (p >> 24, (p >> 16) & 0xff, (p >> 8) & 0xff);
                    if r | g | b != 0 {
                        let _ = writeln!(out, "{case} {cell} {} {} {r} {g} {b}", x + ox, y + oy);
                    }
                }
            }
        }
    }
    std::fs::write(format!("{dir}/model.txt"), out).expect("model.txt");

    let mut fine = String::new();
    for width in [1.0f32, 2.0] {
        for j in 0..=64 {
            for k in 0..16 {
                let mut c = ctx(width, true);
                let y0 = 20.5 + k as f64 / 16.0 - 23.5 * j as f64 / 64.0;
                c.draw(GL_LINES, &[v(8.0, y0 as f32, 0.0, WHITE), v(56.0, (y0 + 48.0 * j as f64 / 64.0) as f32, 0.0, WHITE)]);
                let f = Frame::of(&c.board);
                for y in 0..64 {
                    let r = at(&f, 31, y) >> 24;
                    if r != 0 {
                        let _ = writeln!(fine, "{} {j} {k} {y} {r}", width as u32);
                    }
                }
            }
        }
    }
    std::fs::write(format!("{dir}/model-fine.txt"), fine).expect("model-fine.txt");
}
