//! The raster unit's queue: GL work runs on its own thread, and the
//! orderings the guest can see hold across both queues.

use crate::dev::odyssey::Odyssey;
use crate::dev::odyssey::board::{Board, ODSY_HEIGHT, ODSY_WIDTH};
use crate::dev::odyssey::consts::{
    colour, cursor, dispmode, dma, gl_cmd, light, lut, write, ODSY_BUZZ_1, ODSY_DID_MAIN_SHIFT,
    ODSY_DRAW_MASK_BUFFER0, ODSY_DRAW_MASK_BUFFER1, ODSY_GAMMA_BASE, ODSY_GAMMA_GROUP, ODSY_SW_TOKEN,
};
use crate::dev::odyssey::display::Display;
use crate::dev::odyssey::pipe::encode::{
    vtx, GL_FLAT, GL_LINES,
    GL_TRIANGLES, LibDraw, LibState,
};
use crate::dev::odyssey::mmio::BoardLocks;
use crate::dev::odyssey::raster_unit::{
    RASTER_JOB_OVERHEAD, RASTER_WORK_BUDGET,
    RasterCtx,
};
use crate::dev::odyssey::tests::{board, cmd, pixel, Ram};
use parking_lot::Mutex;
use std::sync::Arc;
use super::*;
use crate::dev::odyssey::raster_bench::Bench;
use std::sync::atomic::Ordering::SeqCst;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::time::{Duration, Instant};

const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];

/// A v12 board whose every pixel belongs to the one window, and the GL
/// draw state the fill-rate benchmark draws with: flat, no depth test.
fn gl_board() -> (Box<Board>, LibState) {
    let mut b = Bench::empty();
    b.board.gl_write_mask |= write::COLOUR_MASK << write::COLOUR_SHIFT;
    b.st.shade_model = GL_FLAT;
    (b.board, b.st)
}

/// Vertices at window pixels (x, y), in the colour given.
fn vertices(points: &[(f32, f32)], colour: [f32; 4]) -> Vec<f32> {
    let mut v = vec![0.0f32; points.len() * vtx::FLOATS];
    for (i, &(x, y)) in points.iter().enumerate() {
        let o = &mut v[i * vtx::FLOATS..][..vtx::FLOATS];
        // The normalised coordinate that the viewport maps to the window
        // position p exactly (a library sends positions it has already put on
        // the board's grid): p / size does not round-trip in single precision,
        // so the nearest coordinate is stepped until the mapping reaches p.
        let ndc = |p: f32, size: u32| {
            let map = |n: f32| (n + 1.0) * 0.5 * size as f32;
            let mut n = p / size as f32 * 2.0 - 1.0;
            for _ in 0..64 {
                if map(n) >= p { break; }
                n = f32::from_bits(if n >= 0.0 { n.to_bits() + 1 } else { n.to_bits() - 1 });
            }
            for _ in 0..64 {
                let down = f32::from_bits(if n > 0.0 { n.to_bits() - 1 } else { n.to_bits() + 1 });
                if map(down) < p { break; }
                n = down;
            }
            n
        };
        o[vtx::POSITION..][..4].copy_from_slice(&[ndc(x, ODSY_WIDTH), ndc(y, ODSY_HEIGHT), 0.0, 1.0]);
        o[vtx::COLOR..][..4].copy_from_slice(&colour);
    }
    v
}

/// One red triangle over the left half of the screen, through the board's
/// own GL draw: set up by the front end, drawn by the raster unit.
fn red_triangle(s: &mut Board, st: &LibState) {
    let v = vertices(&[(0.0, 0.0), (ODSY_WIDTH as f32, 0.0), (0.0, ODSY_HEIGHT as f32)], RED);
    let stipple = [0u32; 32];
    let mut d = LibDraw::new(GL_TRIANGLES, 3, &v, st, &stipple);
    s.lib_draw(&mut d);
}

/// A board with its raster unit on a thread of its own, whose next slice
/// holds the raster unit for `hold_ms` before it draws, as a long draw
/// would; and the draw state.
fn slow_raster(hold_ms: u32) -> (BoardLocks, LibState) {
    let (s, st) = gl_board();
    let h = BoardLocks::new(s);
    h.raster.start();
    h.raster.test_slice_sleep_ms.store(hold_ms, SeqCst);
    (h, st)
}

/// Hold the raster unit with a job that runs until the returned flag is set
/// (or 20 s pass, so a failing test ends instead of hanging), as a long draw
/// would. The test, not a clock, decides how long the unit is busy.
fn hold_raster(h: &BoardLocks) -> Arc<AtomicBool> {
    let release = Arc::new(AtomicBool::new(false));
    let flag = release.clone();
    h.raster.submit(move |_| {
        let deadline = Instant::now() + Duration::from_secs(20);
        while !flag.load(SeqCst) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
    });
    release
}

/// Wait for `what`, up to 20 s.
fn eventually(what: &str, mut done: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while !done() {
        assert!(Instant::now() < deadline, "{what}");
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// A GL draw returns to the front end before it is drawn; a register read
/// that waits for the board's work, issued while the draw is still queued
/// for the raster unit, answers only once it is in the planes. A read that
/// does not wait answers at once.
#[test]
fn a_waiting_read_behind_a_queued_draw_sees_it_drawn() {
    let (h, st) = slow_raster(300);

    red_triangle(&mut h.test_board(), &st);
    assert_eq!(h.raster.pending(), 1, "the draw waits for the raster unit");
    h.write(ODSY_SW_TOKEN, 7, 4);
    assert_eq!(h.read(ODSY_SW_TOKEN, 4), 7);
    assert_eq!(h.raster.pending(), 1, "the token does not wait for the drawing");

    h.read(ODSY_BUZZ_1, 4);
    assert_eq!(h.raster.pending(), 0);
    assert_ne!(pixel(h.test_board(), 0, 10, 10), 0, "drawn before the read answered");
    h.raster.stop();
}

/// The memory word at `0x106c` holds straps and FIFO levels, none of them
/// a drawing result, so a read of it answers while a draw is still queued
/// for the raster unit. It still runs the command stores before it
/// (`a_read_of_a_register_the_work_may_change_finishes_the_work_first`).
#[test]
fn the_memory_word_answers_while_a_draw_is_queued() {
    let (h, st) = slow_raster(300);

    red_triangle(&mut h.test_board(), &st);
    assert_eq!(h.raster.pending(), 1, "the draw waits for the raster unit");
    assert_eq!(h.read(MEMORY_CONFIG, 4) & 0xff00_0000, 0xa100_0000);
    assert_eq!(h.raster.pending(), 1, "the memory word does not wait for the drawing");
    eventually("the draw did not finish", || h.raster.pending() == 0);
    h.raster.stop();
}

/// A fence the front end runs behind a draw is written by the raster unit
/// after the draw: its token never shows while the draw is still queued.
///
/// The raster unit is held by a job that waits for the test, not for a
/// time, and the work budget is lifted. The half-screen triangle costs more
/// than the budget, so with the default one the front end's fence waits
/// for room until the raster unit has drawn it, and the token is then
/// written in the moment the test looks.
#[test]
fn a_fence_behind_a_queued_draw_lands_after_the_draw() {
    let (h, st) = slow_raster(0);
    h.raster.set_budget(u64::MAX);
    let hold = Arc::new(AtomicBool::new(true));
    let held = hold.clone();
    h.raster.submit(move |_| {
        while held.load(SeqCst) {
            std::thread::sleep(Duration::from_millis(1));
        }
    });
    let ram = Ram::new(1 << 20);
    {
        let mut s = h.test_board();
        s.guest = Some(ram.clone());
        s.widget.host_write_addr_lower = 0x1000;
        red_triangle(&mut s, &st);
        s.fence(0x5a5a << dma::FENCE_TOKEN_SHIFT | 3);
    }
    assert_eq!(h.raster.pending(), 3, "the hold, the draw and the fence are all queued");
    assert_eq!(ram.half(0x1006), 0, "the fence waits for the draw before it");
    hold.store(false, SeqCst);
    eventually("the fence did not land", || ram.half(0x1006) == 0x5a5a);
    assert_ne!(pixel(h.test_board(), 0, 10, 10), 0, "the draw was in the planes before the fence");
    h.raster.stop();
}

/// Whether the small triangle at row 100 is in board memory, looked at
/// without waiting for the raster unit.
fn triangle_drawn(h: &BoardLocks) -> bool {
    pixel(&h.test_board(), 0, 10, 110) != 0
}

/// A job that holds the raster unit for `ms`, as a long draw would.
fn busy(h: &BoardLocks, ms: u64) {
    h.raster.submit(move |_| std::thread::sleep(Duration::from_millis(ms)));
}

/// A draw is in memory before the fence behind it writes its token: a
/// guest that sees the token and arms a swap shows the whole drawing. A job
/// queued behind the fence keeps the queue from draining and the slice from
/// ending meanwhile.
#[test]
fn a_fence_follows_the_drawing_before_it() {
    // The first slice waits while everything is queued, and then runs it all.
    let (h, st) = slow_raster(50);
    h.raster.test_slice_limit_ms.store(10_000, SeqCst);
    let ram = Ram::new(1 << 20);
    {
        let mut s = h.test_board();
        s.guest = Some(ram.clone());
        s.widget.host_write_addr_lower = 0x1000;
        small_triangle(&mut s, &st, 100.0, RED);
        s.fence(0x5a5a << dma::FENCE_TOKEN_SHIFT | 3);
        busy(&h, 300);
    }
    eventually("the fence did not land", || ram.half(0x1006) == 0x5a5a);
    assert!(triangle_drawn(&h), "the token was visible before the draw was in memory");
    h.raster.stop();
}

/// A query's answer is a completion too.
#[test]
fn a_query_follows_the_drawing_before_it() {
    let (h, st) = slow_raster(50);
    h.raster.test_slice_limit_ms.store(10_000, SeqCst);
    let ram = Ram::new(1 << 20);
    {
        let mut s = h.test_board();
        s.guest = Some(ram.clone());
        s.widget.ctx_shadow_addr_lower = 0x2000;
        s.widget.host_write_addr_lower = 0x1000;
        small_triangle(&mut s, &st, 100.0, RED);
        for word in [0x4c0, 7, 0x10a5a, 3, 0x5a5a] {
            cmd(&mut s, word);
        }
        busy(&h, 300);
    }
    eventually("the query did not sync", || ram.half(0x1006) == 0x5a5a);
    assert!(triangle_drawn(&h), "the sync word was visible before the draw was in memory");
    h.raster.stop();
}

/// Stopping the machine finishes the raster unit's queued work; a reset
/// waits for it before it clears the planes, so nothing queued before the
/// reset lands after it.
#[test]
fn stop_and_reset_finish_the_raster_queue_first() {
    let (h, st) = slow_raster(200);
    red_triangle(&mut h.test_board(), &st);
    h.work_stop();
    assert!(!h.raster.threaded());
    assert_eq!(h.raster.pending(), 0);
    assert_ne!(pixel(&h.test_board(), 0, 10, 10), 0);

    h.raster.start();
    h.raster.test_slice_sleep_ms.store(200, SeqCst);
    red_triangle(h.test_board(), &st);
    h.test_board().reset();
    assert_eq!(h.raster.pending(), 0);
    assert_eq!(pixel(h.test_board(), 0, 10, 10), 0, "the reset cleared the draw queued before it");
    h.raster.stop();
}

/// A stippled line drawn in two pieces continues its stipple across them
/// whether the raster unit runs on its own thread or inside the draw: the
/// raster unit keeps the phase, and the context carries a marker for it.
#[test]
fn the_stipple_phase_carries_across_draws_on_the_raster_thread() {
    fn two_pieces(h: &BoardLocks, st: &LibState) -> Vec<u32> {
        let stipple = [0u32; 32];
        let mut phase = Some(0);
        for (a, b) in [((10.5f32, 20.5f32), (17.5f32, 20.5f32)), ((17.5, 20.5), (60.5, 20.5))] {
            let v = vertices(&[a, b], RED);
            let mut d = LibDraw::new(GL_LINES, 2, &v, st, &stipple);
            d.line_stipple_enabled = true;
            d.line_stipple_repeat = 1;
            d.line_stipple_pattern = 0x0f0f;
            d.line_phase = phase;
            h.test_board().lib_draw(&mut d);
            phase = d.line_phase;
        }
        let s = h.test_board();
        s.raster_wait();
        (0..64).map(|x| pixel(&s, 0, x, 20)).collect()
    }

    let (s, st) = gl_board();
    let inline = two_pieces(&BoardLocks::new(s), &st);
    let (h, st) = slow_raster(0);
    let threaded = two_pieces(&h, &st);
    h.raster.stop();
    assert!(inline.iter().any(|&p| p != 0) && inline.iter().any(|&p| p == 0), "a stippled line");
    assert_eq!(threaded, inline);
}

/// The raster unit gives a drawn GL job back and the next draw fills it
/// again; the refilled job draws only the new primitive, with its own
/// vertex count, whether it ran on the raster thread or inside its submit.
#[test]
fn a_reused_draw_job_draws_only_its_new_primitive() {
    const BLUE: [f32; 4] = [0.0, 0.0, 1.0, 1.0];
    fn draws(h: &BoardLocks, st: &LibState) -> Vec<u32> {
        let stipple = [0u32; 32];
        let (w, ht) = (ODSY_WIDTH as f32, ODSY_HEIGHT as f32);
        // A six-vertex pair of triangles, then once that is drawn a
        // three-vertex one that covers less.
        let quad = vertices(&[(0.0, 0.0), (64.0, 0.0), (0.0, 64.0), (64.0, 0.0), (64.0, 64.0), (0.0, 64.0)], RED);
        let mut d = LibDraw::new(GL_TRIANGLES, 6, &quad, st, &stipple);
        h.test_board().lib_draw(&mut d);
        h.test_board().raster_wait();
        let tri = vertices(&[(w - 64.0, ht - 64.0), (w, ht - 64.0), (w - 64.0, ht)], BLUE);
        let mut d = LibDraw::new(GL_TRIANGLES, 3, &tri, st, &stipple);
        h.test_board().lib_draw(&mut d);
        let s = h.test_board();
        s.raster_wait();
        let mut px = Vec::new();
        for y in (0..ODSY_HEIGHT as i32).step_by(8) {
            for x in (0..ODSY_WIDTH as i32).step_by(8) {
                px.push(pixel(&s, 0, x, y));
            }
        }
        px
    }

    let (s, st) = gl_board();
    let inline = draws(&BoardLocks::new(s), &st);
    let (h, st) = slow_raster(0);
    let threaded = draws(&h, &st);
    eventually("the second job did not come back", || h.raster.spare_draws() == 1);
    h.raster.stop();
    let colours: std::collections::BTreeSet<u32> = inline.iter().copied().filter(|&p| p != 0).collect();
    assert_eq!(colours.len(), 2, "both primitives drew, each in its own colour");
    assert_eq!(threaded, inline);
}

/// A current-state query behind a queued draw (a small one: a draw of a
/// screenful makes the front end wait for room): the front end goes on at
/// once, and the query's sync word, like a fence, lands only after the
/// draw is in the planes, its snapshot already in place.
#[test]
fn a_query_behind_a_queued_draw_syncs_after_the_draw_without_holding_the_front_end() {
    let (h, st) = slow_raster(0);
    let release = hold_raster(&h);
    let ram = Ram::new(1 << 20);
    {
        let mut s = h.test_board();
        s.guest = Some(ram.clone());
        s.widget.ctx_shadow_addr_lower = 0x2000;
        s.widget.host_write_addr_lower = 0x1000;
        tiny_triangle(&mut s, &st);
        for word in [0x4c0, 7, 0x10a5a, 3, 0x5a5a] {
            cmd(&mut s, word);
        }
    }
    // The raster unit has not run a job since the hold began, so the front
    // end went on without it.
    assert!(h.raster.pending() >= 3, "the hold, the draw and the query wait for the raster unit");
    assert_eq!(ram.half(0x1006), 0, "the sync word waits for the draw before it");
    release.store(true, SeqCst);
    eventually("the query did not sync", || ram.half(0x1006) == 0x5a5a);
    assert_eq!(ram.get(0x2000 + 0x97c, 4), 7u32.to_be_bytes(), "the counter was published before the sync word");
    assert_ne!(pixel(h.test_board(), 0, 10, 10), 0, "the draw was in the planes before the sync word");
    h.raster.stop();
}

/// On the bus, a waiting read behind a queued draw answers busy until the
/// raster unit has drawn it, without the processor waiting inside the
/// read; the memory word answers at once.
#[test]
fn a_waiting_bus_read_is_busy_until_the_raster_unit_has_drawn() {
    let (h, st) = slow_raster(300);
    h.work_start();

    red_triangle(&mut h.test_board(), &st);
    assert_eq!(h.bus_read(ODSY_BUZZ_1, 4), None, "busy while the draw is queued");
    assert_eq!(h.raster.pending(), 1, "and the read did not wait for it");
    assert_eq!(h.bus_read(MEMORY_CONFIG, 4).map(|v| v & 0xff00_0000), Some(0xa100_0000));
    eventually("the read stayed busy", || h.bus_read(ODSY_BUZZ_1, 4).is_some());
    assert_ne!(pixel(&h.test_board(), 0, 10, 10), 0, "drawn before the read answered");
    h.work_stop();
}

/// A small triangle at window row `y`, in the first 64 columns.
fn small_triangle(s: &mut Board, st: &LibState, y: f32, colour: [f32; 4]) {
    let v = vertices(&[(2.0, y), (60.0, y + 3.0), (4.0, y + 40.0)], colour);
    let stipple = [0u32; 32];
    let mut d = LibDraw::new(GL_TRIANGLES, 3, &v, st, &stipple);
    s.lib_draw(&mut d);
}

/// The pixels a one-pixel line lights, in the rectangle of the board's
/// first 80 columns and 64 rows, as (x, y) pairs.
fn line_pixels(a: (f32, f32), b: (f32, f32)) -> Vec<(i32, i32)> {
    let mut bench = Bench::empty();
    bench.board.gl_write_mask |= write::COLOUR_MASK << write::COLOUR_SHIFT;
    bench.st.shade_model = GL_FLAT;
    bench.mode = GL_LINES;
    bench.prims = vec![vertices(&[a, b], RED)];
    bench.run();
    (0..64).flat_map(|y| (0..80).map(move |x| (x, y))).filter(|&(x, y)| pixel(&bench.board, 0, x, y) != 0).collect()
}

/// A one-pixel line takes the centre-sampled rule: a pixel per major-axis
/// column whose centre lies between the ends, the end not included. A line
/// that ends past a column centre but inside that pixel's diamond lights the
/// column; a crossing exactly on a pixel edge goes to the lower pixel
/// whichever way the line slopes. A vertical line exactly on a column edge
/// takes the column to its left.
#[test]
fn a_line_takes_the_centre_sampled_rule() {
    let cases: [((f32, f32), (f32, f32), Vec<(i32, i32)>); 4] = [
        ((1.0, 3.5), (5.75, 3.5), vec![(1, 3), (2, 3), (3, 3), (4, 3), (5, 3)]),
        ((5.75, 3.5), (1.0, 3.5), vec![(1, 3), (2, 3), (3, 3), (4, 3), (5, 3)]),
        ((1.5, 2.0), (5.5, 4.0), vec![(1, 1), (2, 2), (3, 2), (4, 3)]),
        ((4.0, 1.0), (4.0, 5.0), vec![(3, 1), (3, 2), (3, 3), (3, 4)]),
    ];

    for (a, b, want) in cases {
        assert_eq!(line_pixels(a, b), want, "centre-sampled {a:?} to {b:?}");
    }
}

/// Centre-sampled lines at the board's line-test geometry: a horizontal
/// line from x 60.0 to 70.75 lights columns 60 to 70, eleven pixels, the one
/// past the last column centre included; a line rising at slope one half
/// crosses a row edge exactly at every other column centre, and each of those
/// columns takes the lower row, sharing it with the column after.
#[test]
fn centre_sampled_lines_light_the_pixels_the_rule_names() {
    let horizontal = line_pixels((60.0, 10.5), (70.75, 10.5));
    assert_eq!(horizontal, (60..=70).map(|x| (x, 10)).collect::<Vec<_>>());

    let rising = line_pixels((60.0, 9.75), (68.0, 13.75));
    assert_eq!(rising, [(60, 9), (61, 10), (62, 10), (63, 11), (64, 11), (65, 12), (66, 12), (67, 13)]);
}

/// Lighting a vertex reuses one lighting state and puts back only the lights
/// the last vertex enabled, and the colour is what a fresh state gives:
/// across lights switching on, changing and off, with a spot, a positional
/// and a directional light, and with none.
#[test]
fn a_reused_lighting_state_lights_a_vertex_as_a_fresh_one_does() {
    use crate::dev::gl::light::Lighting;

    fn fresh(s: &Board, obj: &[f32; 4]) -> [f32; 4] {
        let mut lt = Lighting::default();
        lt.init();
        let v = s.gl_light_fill(&mut lt, obj);
        let (front, _) = lt.light_vertex(v.eye, v.normal, v.colour);
        [front[0], front[1], front[2], v.alpha]
    }
    fn set_light(s: &mut Board, i: usize, colour: [f32; 3], directional: bool, spot: bool) {
        for k in 0..3 {
            s.geom.light_colour[i * 12 + k] = (0.1f32 * (i + 1) as f32).to_bits();
            s.geom.light_colour[i * 12 + 4 + k] = colour[k].to_bits();
            s.geom.light_colour[i * 12 + 8 + k] = (colour[k] * 0.5).to_bits();
        }
        let blk = &mut s.geom.light_vec[i * gl_cmd::LIGHT_VEC_FLOATS as usize..];
        for k in 0..3 {
            blk[light::POS + k] = [2.0f32, 3.0, 4.0 + i as f32][k].to_bits();
            blk[light::HALF + k] = (if directional { [0.0f32, 0.6, 0.8][k] } else { 0.0 }).to_bits();
        }
        blk[light::ATTEN] = 1.0f32.to_bits();
        blk[light::ATTEN + 1] = 0.1f32.to_bits();
        blk[light::ATTEN + 2] = 0.01f32.to_bits();
        blk[light::SPOT_COS] = (if spot { -0.5f32 } else { 1.0 }).to_bits();
        blk[light::SPOT_EXP] = (-2.0f32).to_bits();
        for k in 0..3 {
            blk[light::SPOT_DIR + k] = [0.0f32, 0.0, 1.0][k].to_bits();
        }
    }
    fn clear_light(s: &mut Board, i: usize) {
        s.geom.light_colour[i * 12..i * 12 + 12].fill(0);
    }

    let (mut s, _) = board("v12");
    let identity: [f32; 16] = [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.];
    s.geom.modelview = identity.map(f32::to_bits);
    s.geom.normal_matrix = identity.map(f32::to_bits);
    s.geom.normal = [0.0f32, 0.6, 0.8].map(f32::to_bits);
    s.geom.colour = [0.7f32, 0.5, 0.3, 0.9].map(f32::to_bits);
    for k in 0..4 {
        s.geom.material[k] = 0.2f32.to_bits();
        s.geom.material[4 + k] = [0.8f32, 0.4, 0.6, 0.5][k].to_bits();
        s.geom.material[8 + k] = 0.5f32.to_bits();
    }
    s.geom.shininess_sqrt = 4.0f32.to_bits();
    let obj = [1.0, 2.0, 3.0, 1.0];
    let mut seen = Vec::new();
    let check = |s: &mut Board, what: &str| {
        let (out, _) = s.gl_light(&obj);
        let want = fresh(s, &obj);
        assert_eq!(out.map(f32::to_bits), want.map(f32::to_bits), "{what}");
        want
    };

    seen.push(check(&mut s, "no lights"));
    set_light(&mut s, 0, [1.0, 0.2, 0.1], true, false);
    set_light(&mut s, 3, [0.3, 0.9, 0.2], false, true);
    seen.push(check(&mut s, "a directional light and a spot"));
    seen.push(check(&mut s, "the same again"));
    clear_light(&mut s, 0);
    set_light(&mut s, 5, [0.2, 0.2, 0.9], false, false);
    seen.push(check(&mut s, "light 0 off, light 5 on"));
    clear_light(&mut s, 3);
    clear_light(&mut s, 5);
    seen.push(check(&mut s, "all off again"));
    // Light 3 was a spot and returns as a plain positional light: nothing of
    // its spot may be left over.
    set_light(&mut s, 3, [0.3, 0.9, 0.2], false, false);
    seen.push(check(&mut s, "light 3 back without its spot"));
    clear_light(&mut s, 3);
    s.geom.colour_material = colour::MATERIAL_DIFFUSE;
    set_light(&mut s, 7, [0.6, 0.6, 0.6], false, false);
    seen.push(check(&mut s, "colour material, light 7"));

    // The cases differ, so the comparisons above could have failed.
    assert_ne!(seen[0], seen[1]);
    assert_ne!(seen[1], seen[3]);
}

/// A two-sided program lights the back face with the back material and the
/// reversed normal; one-sided, the back colour is the front's.
#[test]
fn a_two_sided_program_lights_the_back_with_the_back_material() {
    let (mut s, _) = board("v12");
    let identity: [f32; 16] = [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.];
    s.geom.modelview = identity.map(f32::to_bits);
    s.geom.normal_matrix = identity.map(f32::to_bits);
    s.geom.normal = [0.0f32, 0.0, 1.0].map(f32::to_bits);
    // A white light along +z from the viewer's side: the front faces it.
    for k in 0..3 {
        s.geom.light_colour[4 + k] = 1.0f32.to_bits();
    }
    let blk = &mut s.geom.light_vec[..gl_cmd::LIGHT_VEC_FLOATS as usize];
    blk[light::POS + 2] = 1.0f32.to_bits();
    blk[light::HALF + 2] = 1.0f32.to_bits();
    blk[light::SPOT_COS] = 1.0f32.to_bits();
    for k in 0..4 {
        s.geom.material[4 + k] = [0.0f32, 1.0, 0.0, 1.0][k].to_bits();
        s.geom.back_material[4 + k] = [0.0f32, 0.0, 1.0, 1.0][k].to_bits();
    }
    let obj = [0.0, 0.0, 0.0, 1.0];

    // Lit green from the front; the scene ambient (0.2) on the default
    // material ambient (0.2) adds 0.04 to every component.
    let (front, back) = s.gl_light(&obj);
    assert_eq!(back, front, "one-sided");
    assert!(front[1] > 0.9 && front[2] < 0.05, "lit green from the front: {front:?}");

    s.geom.two_sided = true;
    let (front2, back) = s.gl_light(&obj);
    assert_eq!(front2, front, "front unchanged");
    // The back's normal faces away from the light: ambient only.
    assert!(back[1] < 0.05 && back[2] < 0.05, "back not lit: {back:?}");
    assert_eq!(back[3], 1.0, "the back diffuse alpha");
}

/// A triangle a few pixels across.
fn tiny_triangle(s: &mut Board, st: &LibState) {
    let v = vertices(&[(10.0, 10.0), (14.0, 10.0), (10.0, 14.0)], RED);
    let stipple = [0u32; 32];
    let mut d = LibDraw::new(GL_TRIANGLES, 3, &v, st, &stipple);
    s.lib_draw(&mut d);
}

/// Many small primitives queue behind a busy raster unit without the front
/// end waiting, however many there are in a frame; the queue's work is the
/// sum of theirs.
#[test]
fn many_small_jobs_queue_without_the_front_end_waiting() {
    let (h, st) = slow_raster(0);
    let release = hold_raster(&h);
    let mut s = h.test_board();

    for _ in 0..1000 {
        tiny_triangle(&mut s, &st);
    }
    // The held raster unit has run none of them, and the front end did not
    // wait for it.
    assert_eq!(h.raster.pending(), 1001, "no small job waited for room");
    assert!(h.raster.queued() < RASTER_WORK_BUDGET, "{} queued", h.raster.queued());
    assert!(h.raster.queued() >= 1000 * RASTER_JOB_OVERHEAD as u64);
    release.store(true, SeqCst);
    eventually("the queue did not drain", || h.raster.pending() == 0);
    assert_eq!(h.raster.queued(), 0);
    h.raster.stop();
}

/// One primitive that is a whole screen of work makes the next job wait
/// until the raster unit has drawn it.
#[test]
fn one_big_job_makes_the_front_end_wait() {
    let (h, st) = slow_raster(0);
    let release = hold_raster(&h);
    // The held raster unit runs nothing, so the first job returning shows it
    // did not wait.
    red_triangle(h.test_board(), &st);
    assert!(h.raster.queued() >= RASTER_WORK_BUDGET, "{} queued", h.raster.queued());
    std::thread::scope(|sc| {
        let second = sc.spawn(|| tiny_triangle(h.test_board(), &st));
        eventually("the second job did not wait for room", || h.raster.waiting_for_room());
        assert!(!second.is_finished(), "the second job waits for the raster unit to draw");
        release.store(true, SeqCst);
        second.join().unwrap();
    });
    assert!(h.raster.queued() <= RASTER_WORK_BUDGET / 2, "{} queued", h.raster.queued());
    eventually("the queue did not drain", || h.raster.pending() == 0);
    assert_ne!(pixel(&h.test_board(), 0, 10, 10), 0);
    h.raster.stop();
}

/// The wait ends once the raster unit has worked the queue down to half the
/// budget, not at the first job it finishes and not at the last; every job
/// runs, in order.
#[test]
fn the_bound_releases_as_the_raster_unit_works() {
    let (h, _st) = slow_raster(200);
    let ran = std::sync::Arc::new(Mutex::new(Vec::new()));
    let job = |i: u32| {
        let ran = ran.clone();
        move |_: &mut RasterCtx| ran.lock().push(i)
    };

    h.raster.set_budget(1000);
    h.raster.submit_work(400, job(0));
    h.raster.submit_work(400, job(1));
    h.raster.submit_work(400, job(2));
    assert_eq!(h.raster.queued(), 1200, "under the budget, a job does not wait; at it, the next does");
    let start = Instant::now();
    h.raster.submit_work(400, job(3));
    assert!(start.elapsed() >= Duration::from_millis(150), "the fourth waited for the slice");
    assert!(h.raster.queued() <= 500 + 400, "{} queued", h.raster.queued());
    assert!(ran.lock().len() >= 2, "released after the queue fell to half the budget");
    eventually("the queue did not drain", || h.raster.pending() == 0);
    assert_eq!(*ran.lock(), vec![0, 1, 2, 3]);
    h.raster.stop();
}

/// A stop sets a waiting front end going at once, without waiting for the
/// raster unit's slice; the stop still finishes every job, in order.
#[test]
fn stop_releases_a_waiting_front_end() {
    let (h, _st) = slow_raster(1500);
    let ran = std::sync::Arc::new(Mutex::new(Vec::new()));
    let job = |i: u32| {
        let ran = ran.clone();
        move |_: &mut RasterCtx| ran.lock().push(i)
    };

    h.raster.set_budget(1000);
    h.raster.submit_work(2000, job(0));
    let start = Instant::now();
    std::thread::scope(|sc| {
        let waiter = sc.spawn(|| {
            h.raster.submit_work(1, job(1));
            start.elapsed()
        });
        std::thread::sleep(Duration::from_millis(200));
        assert!(!waiter.is_finished(), "the front end waits for room");
        let stopper = sc.spawn(|| h.raster.stop());
        let waited = waiter.join().unwrap();
        assert!(waited < Duration::from_millis(1000), "released by the stop after {waited:?}, not by the slice");
        stopper.join().unwrap();
    });
    assert_eq!(*ran.lock(), vec![0, 1]);
    assert_eq!(h.raster.queued(), 0);
}

// The display with no lock held, against live writers.

/// A board whose every drawable is an RGB visual through a gamma ramp, four
/// windows in vertical bands; the raster unit running.
fn display_rig() -> (BoardLocks, LibState, Arc<Display>) {
    let (mut s, st) = gl_board();
    for i in 0..0x600u32 {
        let v = (i * 0x3ff / 0x5ff) & 0x3ff;
        s.regs().set_lut((ODSY_GAMMA_BASE + i) as usize, v << 20 | v << 10 | v);
    }
    // Every window the test screen's 8:8:8:8 double-buffered layout, RGB.
    for id in 0..dispmode::COUNT {
        s.regs().set_dispmode(id as usize, 0x0096_4000 | 0xa << 9 | ODSY_GAMMA_GROUP << dispmode::GROUP_SHIFT);
    }
    let did = s.dest_mem().did;
    for y in 0..ODSY_HEIGHT as i32 {
        for x in 0..ODSY_WIDTH as i32 {
            did.store1(x, y, ((x / 320) as u8) << ODSY_DID_MAIN_SHIFT);
        }
    }
    s.set_scanout_size(ODSY_WIDTH, ODSY_HEIGHT);
    let h = BoardLocks::new(s);
    h.raster.start();
    let display = Arc::new(Display::new(crate::disp::StatusSource::detached()));
    (h, st, display)
}

/// Run the display until `stop`, as its thread does with no pause.
fn run_display(h: &BoardLocks, display: &Arc<Display>, stop: &Arc<std::sync::atomic::AtomicBool>) -> std::thread::JoinHandle<()> {
    let (h, display, stop) = (h.clone(), display.clone(), stop.clone());
    std::thread::spawn(move || {
        while !stop.load(SeqCst) {
            Odyssey::frame(&h, &display);
        }
    })
}

/// Frames at the display's pace after the writers stopped, so the retrace
/// comes due and the last marks are consumed.
fn settle(h: &BoardLocks, display: &Display, frames: u32) {
    for _ in 0..frames {
        std::thread::sleep(Duration::from_millis(20));
        Odyssey::frame(h, display);
    }
}

/// The presented frame against a whole composition of board memory now.
fn assert_frame_is_whole_resolve(h: &BoardLocks, display: &Display, what: &str) {
    h.raster.wait_idle();
    let s = h.test_board();
    let reader = display.reader.lock();
    let (w, ht) = (reader.frame_width as usize, reader.frame_height as usize);
    let regs = s.regs().read();
    let frame = crate::dev::odyssey::frame::Frame::snapshot(&regs, s.front.lock().displayed_buffer).expect("the scan-out is set up");
    let mut want = vec![0u32; w * ht];
    frame.compose(s.vram.ptr_shared(), &regs, &mut want, w, ht);
    for y in 0..ht {
        assert_eq!(&reader.frame[y * w..(y + 1) * w], &want[y * w..(y + 1) * w], "{what}: row {y}");
    }
}

/// One front-end thread, the raster unit, lookup and cursor stores and
/// swap arms run against `Odyssey::frame` with no pause; when the writers
/// stop and the raster queue drains the presented frame is a whole resolve
/// of the planes, and a hundred idle frames leave it so without resolving a
/// line.
fn stress(seed: u64, last_is_lut: bool) {
    use crate::dev::odyssey::raster_bench::Lcg;
    let (h, mut st, display) = display_rig();
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let shown = run_display(&h, &display, &stop);
    let mut rng = Lcg(seed);
    let end = Instant::now() + Duration::from_millis(600);

    while Instant::now() < end {
        let mut s = h.test_board();
        for _ in 0..8 {
            match rng.next_u32() % 9 {
                0 | 1 => {
                    let target = rng.next_u32() & 1;
                    st.draw_buffers = 1 << target;
                    let y = (rng.next_u32() % 960) as f32;
                    small_triangle(&mut s, &st, y, [(rng.next_u32() % 2) as f32, (rng.next_u32() % 2) as f32, 1.0, 1.0]);
                }
                2 | 3 => {
                    let (x, y) = ((rng.next_u32() % 1280) as i64, (rng.next_u32() % 1024) as i64);
                    s.draw_mask = if rng.next_u32() & 1 == 0 { ODSY_DRAW_MASK_BUFFER0 } else { ODSY_DRAW_MASK_BUFFER1 };
                    s.paint_pixel(x, y, rng.next_u32());
                }
                4 => {
                    let (x, y) = ((rng.next_u32() % 1200) as i64, (rng.next_u32() % 1000) as i64);
                    s.buffer_store(rng.next_u32() & 1, x, y, rng.next_u32());
                }
                5 => {
                    let i = rng.next_u32() % 0x600;
                    s.backend_data(ODSY_GAMMA_BASE + i, rng.next_u32() & lut::ENTRY_MASK);
                }
                6 => s.backend_data(cursor::BASE + cursor::POSITION, (rng.next_u32() % 1200) << 16 | (rng.next_u32() % 1000)),
                7 => {
                    let (x, y) = (rng.next_u32() % 1200, rng.next_u32() % 1000);
                    s.draw_mask = ODSY_DRAW_MASK_BUFFER0;
                    s.clear_planes(x, y, 40, 3, (rng.next_u32() & 0xf8) as u8, 0xff);
                }
                _ => h.front.lock().swap_arm |= rng.next_u32() & 0xf,
            }
        }
        std::thread::yield_now();
    }
    // A slow tail of single stores of one kind, so the last of them lands at
    // a random point of a frame the display is making. A lookup store marks
    // every line, so a tail of pixels and a tail of lookup stores are run
    // apart: either would repair what the other left stale.
    // No swap is left to repaint what the tail leaves stale.
    h.front.lock().swap_arm = 0;
    std::thread::sleep(Duration::from_millis(40));
    for _ in 0..100u32 {
        {
            let s = h.test_board();
            if !last_is_lut {
                s.draw_mask = ODSY_DRAW_MASK_BUFFER0 | ODSY_DRAW_MASK_BUFFER1;
                s.paint_pixel((rng.next_u32() % 1280) as i64, (rng.next_u32() % 1024) as i64, rng.next_u32());
            } else {
                // The entry every black pixel is looked up in.
                s.backend_data(ODSY_GAMMA_BASE, rng.next_u32() & lut::ENTRY_MASK);
            }
        }
        std::thread::sleep(Duration::from_micros(1500));
    }
    stop.store(true, SeqCst);
    shown.join().unwrap();
    h.raster.wait_idle();
    settle(&h, &display, 4);
    assert_frame_is_whole_resolve(&h, &display, "after the writers stopped");

    let before = display.reader.lock().frame.clone();
    for _ in 0..100 {
        Odyssey::frame(&h, &display);
    }
    let reader = display.reader.lock();
    assert!(reader.frame == before, "an idle frame changed the picture");
    drop(reader);
    assert_frame_is_whole_resolve(&h, &display, "after idle frames");
    h.raster.stop();
}

#[test]
fn the_frame_ends_as_a_whole_resolve_under_live_writers() {
    for (seed, lut) in [(0x5a17, false), (0x77, true), (0xbeef, false), (0x31, true), (0x42, false), (0x99, true)] {
        stress(seed, lut);
    }
}

/// A guest that waits for its fence before it arms a swap never sees a
/// frame that is not the buffer the latch put on the screen: each frame the
/// display makes is one colour, the one drawn into the buffer it shows.
fn swap_atomicity() {
    let (h, mut st, display) = display_rig();
    // One window, so the one swap bit is the whole picture.
    {
        let s = h.test_board();
        let m = s.dest_mem();
        for y in 0..ODSY_HEIGHT as i32 {
            for x in 0..ODSY_WIDTH as i32 {
                m.did.store1(x, y, 0);
                m.colour[0].store_word(x, y, 0);
                m.colour[1].store_word(x, y, 0);
            }
        }
    }
    h.raster.test_slice_limit_ms.store(10_000, SeqCst);
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let colours = [Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0))];
    let bad = Arc::new(Mutex::new(Vec::<String>::new()));
    let shown = {
        let (h, display, stop, colours, bad) = (h.clone(), display.clone(), stop.clone(), colours.clone(), bad.clone());
        std::thread::spawn(move || {
            while !stop.load(SeqCst) {
                Odyssey::frame(&h, &display);
                let reader = display.reader.lock();
                let w = reader.frame_width as usize;
                let bit = (h.front.lock().displayed_buffer & 1) as usize;
                let want = colours[bit].load(SeqCst) as u32;
                // Every 8th pixel of every 8th line, and the corners.
                let mut seen = std::collections::BTreeSet::new();
                for y in (0..reader.frame_height as usize).step_by(8).chain([1023]) {
                    for x in (0..w).step_by(8).chain([w - 1]) {
                        seen.insert(reader.frame[y * w + x] & 0xff_ffff);
                    }
                }
                                if seen.len() != 1 || *seen.iter().next().unwrap() != want {
                    bad.lock().push(format!("displayed buffer {bit}: wanted {want:#08x}, saw {seen:06x?}"));
                }
            }
        })
    };
    let ram = Ram::new(1 << 20);
    let (wf, hf) = (ODSY_WIDTH as f32, ODSY_HEIGHT as f32);

    std::thread::sleep(Duration::from_millis(100));
    // Two swaps with nothing drawn: the first measures the buffers the
    // mode change left unnamed, so from here the display knows the rows
    // the buffers differ in only from the rows the raster unit publishes.
    for _ in 0..2 {
        h.front.lock().swap_arm |= 1;
        eventually("the swap did not latch", || h.front.lock().swap_arm == 0);
        std::thread::sleep(Duration::from_millis(40));
    }
    for i in 1..=10u32 {
        let back = 1 - (h.front.lock().displayed_buffer & 1);
        let red = (i * 9) as f32 / 255.0;
        {
            let s = h.test_board();
            st.draw_buffers = 1 << back;
            let v = vertices(&[(-wf, -hf), (3.0 * wf, -hf), (-wf, 3.0 * hf)], [red, 0.0, 0.0, 1.0]);
            let stipple = [0u32; 32];
            let mut d = LibDraw::new(GL_TRIANGLES, 3, &v, &st, &stipple);
            s.lib_draw(&mut d);
            // The completion the guest waits for, slow to finish after it
            // has written its token.
            let r = ram.clone();
            s.raster.submit_completion(move |_| {
                r.put(0x1000, &[1]);
                std::thread::sleep(Duration::from_millis(150));
            });
        }
        eventually("the completion did not land", || ram.get(0x1000, 1)[0] == 1);
        ram.put(0x1000, &[0]);
        // What the display shows of this buffer once it is latched: the
        // red the client drew through the gamma ramp.
        let want = {
            let s = h.test_board();
            let regs = s.regs().read();
            let r = i * 9;
            regs.gamma(r, lut::RED_SHIFT) << 16 | regs.gamma(0, lut::GREEN_SHIFT) << 8 | regs.gamma(0, lut::BLUE_SHIFT)
        };
        colours[back as usize].store(want as u64, SeqCst);
        h.front.lock().swap_arm |= 1;
        eventually("the swap did not latch", || h.front.lock().swap_arm == 0);
    }
    std::thread::sleep(Duration::from_millis(100));
    stop.store(true, SeqCst);
    shown.join().unwrap();
    h.raster.stop();
    let bad = bad.lock();
    assert!(bad.is_empty(), "{} mixed or stale frames, first: {:?}", bad.len(), bad.first());
}

#[test]
fn a_swap_after_a_fence_shows_the_whole_new_frame_with_no_lock() {
    swap_atomicity();
}

/// The frame waits for no slice and holds no lock the front end needs: a
/// slice that holds the raster unit delays neither the board nor the frame.
/// The slice is held by a job that waits for the test, so the frame ends
/// while the slice is still running only if it waited for nothing; a frame
/// that waited for the slice would never end.
#[test]
fn a_frame_waits_for_no_slice_and_takes_no_lock() {
    let (h, st) = slow_raster(0);
    let display = Arc::new(Display::new(crate::disp::StatusSource::detached()));
    let hold = Arc::new(AtomicBool::new(true));
    let held = hold.clone();
    h.raster.submit(move |_| {
        while held.load(SeqCst) {
            std::thread::sleep(Duration::from_millis(1));
        }
    });

    red_triangle(&mut h.test_board(), &st);
    eventually("the raster unit began its slice", || h.raster.in_slice());
    // The processors' lock is held across the frame, as an access in
    // progress holds it: the display needs nothing a processor holds.
    let cpu = h.test_submit();
    let framed = AtomicBool::new(false);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            Odyssey::frame(&h, &display);
            framed.store(true, SeqCst);
        });
        // Let the slice go if the frame never ends, so the failure is
        // reported rather than hanging the scope.
        let deadline = Instant::now() + Duration::from_secs(20);
        while !framed.load(SeqCst) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
        let ended = framed.load(SeqCst);
        let in_slice = h.raster.in_slice();
        hold.store(false, SeqCst);
        assert!(ended, "the frame waited for the slice");
        assert!(in_slice, "the slice was still running");
    });
    drop(cpu);
    h.raster.stop();
}

/// GL_SPHERE_MAP: a vertex straight ahead with the normal facing the viewer
/// reflects straight back and maps to the centre; one to the right at the
/// same depth reflects to (0.707, 0, 0.707), m = 2 sqrt(0.5 + 1.707^2).
#[test]
fn the_sphere_map_reflects_the_eye_vector_about_the_normal() {
    let (mut s, _) = board("v12");
    let identity: [f32; 16] = [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.];
    s.geom.modelview = identity.map(f32::to_bits);
    s.geom.normal_matrix = identity.map(f32::to_bits);
    s.geom.normal = [0.0f32, 0.0, 1.0].map(f32::to_bits);

    let [ss, tt] = s.sphere_map(&[0.0, 0.0, -1.0, 1.0]);
    assert!((ss - 0.5).abs() < 1e-6 && (tt - 0.5).abs() < 1e-6, "{ss} {tt}");
    let [ss, tt] = s.sphere_map(&[1.0, 0.0, -1.0, 1.0]);
    let r = std::f64::consts::FRAC_1_SQRT_2;
    let m = 2.0 * (r * r + (r + 1.0) * (r + 1.0)).sqrt();
    assert!((ss as f64 - (r / m + 0.5)).abs() < 1e-6, "{ss}");
    assert!((tt - 0.5).abs() < 1e-6, "{tt}");
}
