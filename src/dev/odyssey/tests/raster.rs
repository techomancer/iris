//! The raster unit's queue: GL work runs on its own thread, and the
//! orderings the guest can see hold across both queues.

use super::*;
use crate::dev::odyssey::raster_bench::Bench;
use std::sync::atomic::Ordering::SeqCst;
use std::time::{Duration, Instant};

const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];

/// A v12 board whose every pixel belongs to the one window, and the GL
/// draw state the fill-rate benchmark draws with: flat, no depth test.
fn gl_board() -> (Box<Board>, GlDrawState) {
    let mut b = Bench::empty();
    b.board.gl_write_mask |= ODSY_WRITE_COLOUR_MASK << ODSY_WRITE_COLOUR_SHIFT;
    b.st.shade_model = SGI_GL_FLAT;
    b.st.dither = 0;
    (b.board, b.st)
}

/// Vertices at window pixels (x, y), in the colour given.
fn vertices(points: &[(f32, f32)], colour: [f32; 4]) -> Vec<f32> {
    let mut v = vec![0.0f32; points.len() * SGI_GL_VERTEX_FLOATS];
    for (i, &(x, y)) in points.iter().enumerate() {
        let o = &mut v[i * SGI_GL_VERTEX_FLOATS..][..SGI_GL_VERTEX_FLOATS];
        let ndc = |p: f32, size: u32| p / size as f32 * 2.0 - 1.0;
        o[SGI_GL_V_POSITION..][..4].copy_from_slice(&[ndc(x, SGI_ODYSSEY_WIDTH), ndc(y, SGI_ODYSSEY_HEIGHT), 0.0, 1.0]);
        o[SGI_GL_V_COLOR..][..4].copy_from_slice(&colour);
    }
    v
}

/// One red triangle over the left half of the screen, through the board's
/// own GL draw: set up by the front end, drawn by the raster unit.
fn red_triangle(s: &mut Board, st: &GlDrawState) {
    let v = vertices(&[(0.0, 0.0), (SGI_ODYSSEY_WIDTH as f32, 0.0), (0.0, SGI_ODYSSEY_HEIGHT as f32)], RED);
    let stipple = [0u32; 32];
    let mut d = GlDraw::new(SGI_GL_TRIANGLES, 3, &v, st, &stipple);
    assert_eq!(s.gl_draw(&mut d), SGI_GL_DRAW_OK);
}

/// A board with its raster unit on a thread of its own, whose next slice
/// holds the raster unit for `hold_ms` before it draws, as a long draw
/// would; and the draw state.
fn slow_raster(hold_ms: u32) -> (BoardLocks, GlDrawState) {
    let (s, st) = gl_board();
    let h = BoardLocks::new(s);
    h.raster.start();
    h.raster.test_slice_sleep_ms.store(hold_ms, SeqCst);
    (h, st)
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

    red_triangle(&mut h.board.lock(), &st);
    assert_eq!(h.raster.pending(), 1, "the draw waits for the raster unit");
    h.write(ODSY_SW_TOKEN, 7, 4);
    assert_eq!(h.read(ODSY_SW_TOKEN, 4), 7);
    assert_eq!(h.raster.pending(), 1, "the token does not wait for the drawing");

    h.read(ODSY_BUZZ_1, 4);
    assert_eq!(h.raster.pending(), 0);
    let s = h.board.lock();
    assert_ne!(pixel(&s, 0, 10, 10), 0, "drawn before the read answered");
    drop(s);
    h.raster.stop();
}

/// The memory word at `0x106c` holds straps and FIFO levels, none of them
/// a drawing result, so a read of it answers while a draw is still queued
/// for the raster unit. It still runs the command stores before it
/// (`a_read_of_a_register_the_work_may_change_finishes_the_work_first`).
#[test]
fn the_memory_word_answers_while_a_draw_is_queued() {
    let (h, st) = slow_raster(300);

    red_triangle(&mut h.board.lock(), &st);
    assert_eq!(h.raster.pending(), 1, "the draw waits for the raster unit");
    assert_eq!(h.read(MEMORY_CONFIG, 4) & 0xff00_0000, 0xa100_0000);
    assert_eq!(h.raster.pending(), 1, "the memory word does not wait for the drawing");
    eventually("the draw did not finish", || h.raster.pending() == 0);
    h.raster.stop();
}

/// A fence the front end runs behind a draw is written by the raster unit
/// after the draw: its token never shows while the draw is still queued.
#[test]
fn a_fence_behind_a_queued_draw_lands_after_the_draw() {
    let (h, st) = slow_raster(300);
    let ram = Ram::new(1 << 20);
    {
        let mut s = h.board.lock();
        s.guest = Some(ram.clone());
        s.host_write_addr_lower = 0x1000;
        red_triangle(&mut s, &st);
        s.fence(0x5a5a << ODSY_DMA_FENCE_TOKEN_SHIFT | 3);
    }
    assert_eq!(ram.half(0x1006), 0, "the fence waits for the draw before it");
    eventually("the fence did not land", || ram.half(0x1006) == 0x5a5a);
    let s = h.board.lock();
    let _planes = h.raster.planes();
    assert_ne!(pixel(&s, 0, 10, 10), 0, "the draw was in the planes before the fence");
    drop(_planes);
    drop(s);
    h.raster.stop();
}

/// The display takes the planes only between the raster unit's slices: a
/// frame that starts while a slice is drawing waits for it to end.
#[test]
fn the_display_waits_out_a_raster_slice_in_flight() {
    let (h, st) = slow_raster(200);

    red_triangle(&mut h.board.lock(), &st);
    eventually("the raster slice did not start", || h.raster.in_slice());
    let s = h.board.lock();
    let mut planes = h.raster.planes();
    assert_ne!(pixel(&s, 0, 10, 10), 0, "the frame saw the whole draw");
    let mut rows = Vec::new();
    planes.take_rows(0, |y| rows.push(y));
    assert!(rows.contains(&10), "and the display hears which rows the raster unit drew: {rows:?}");
    drop(planes);
    drop(s);
    h.raster.stop();
}

/// Stopping the machine finishes the raster unit's queued work; a reset
/// waits for it before it clears the planes, so nothing queued before the
/// reset lands after it.
#[test]
fn stop_and_reset_finish_the_raster_queue_first() {
    let (h, st) = slow_raster(200);
    red_triangle(&mut h.board.lock(), &st);
    h.work_stop();
    assert!(!h.raster.threaded());
    assert_eq!(h.raster.pending(), 0);
    assert_ne!(pixel(&h.board.lock(), 0, 10, 10), 0);

    h.raster.start();
    h.raster.test_slice_sleep_ms.store(200, SeqCst);
    let mut s = h.board.lock();
    red_triangle(&mut s, &st);
    s.reset();
    assert_eq!(h.raster.pending(), 0);
    assert_eq!(pixel(&s, 0, 10, 10), 0, "the reset cleared the draw queued before it");
    drop(s);
    h.raster.stop();
}

/// A stippled line drawn in two pieces continues its stipple across them
/// whether the raster unit runs on its own thread or inside the draw: the
/// raster unit keeps the phase, and the context carries a marker for it.
#[test]
fn the_stipple_phase_carries_across_draws_on_the_raster_thread() {
    fn two_pieces(h: &BoardLocks, st: &GlDrawState) -> Vec<u32> {
        let stipple = [0u32; 32];
        let mut phase = Some(0);
        for (a, b) in [((10.5f32, 20.5f32), (17.5f32, 20.5f32)), ((17.5, 20.5), (60.5, 20.5))] {
            let v = vertices(&[a, b], RED);
            let mut d = GlDraw::new(SGI_GL_LINES, 2, &v, st, &stipple);
            d.line_stipple_enabled = true;
            d.line_stipple_repeat = 1;
            d.line_stipple_pattern = 0x0f0f;
            d.line_phase = phase;
            assert_eq!(h.board.lock().gl_draw(&mut d), SGI_GL_DRAW_OK);
            phase = d.line_phase;
        }
        let s = h.board.lock();
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
    fn draws(h: &BoardLocks, st: &GlDrawState) -> Vec<u32> {
        let stipple = [0u32; 32];
        let (w, ht) = (SGI_ODYSSEY_WIDTH as f32, SGI_ODYSSEY_HEIGHT as f32);
        // A six-vertex pair of triangles, then once that is drawn a
        // three-vertex one that covers less.
        let quad = vertices(&[(0.0, 0.0), (64.0, 0.0), (0.0, 64.0), (64.0, 0.0), (64.0, 64.0), (0.0, 64.0)], RED);
        let mut d = GlDraw::new(SGI_GL_TRIANGLES, 6, &quad, st, &stipple);
        assert_eq!(h.board.lock().gl_draw(&mut d), SGI_GL_DRAW_OK);
        h.board.lock().raster_wait();
        let tri = vertices(&[(w - 64.0, ht - 64.0), (w, ht - 64.0), (w - 64.0, ht)], BLUE);
        let mut d = GlDraw::new(SGI_GL_TRIANGLES, 3, &tri, st, &stipple);
        assert_eq!(h.board.lock().gl_draw(&mut d), SGI_GL_DRAW_OK);
        let s = h.board.lock();
        s.raster_wait();
        let mut px = Vec::new();
        for y in (0..SGI_ODYSSEY_HEIGHT as i32).step_by(8) {
            for x in (0..SGI_ODYSSEY_WIDTH as i32).step_by(8) {
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

/// A current-state query behind a queued draw: the front end goes on at
/// once, and the query's sync word, like a fence, lands only after the
/// draw is in the planes, its snapshot already in place.
#[test]
fn a_query_behind_a_queued_draw_syncs_after_the_draw_without_holding_the_front_end() {
    let (h, st) = slow_raster(300);
    let ram = Ram::new(1 << 20);
    let start = Instant::now();
    {
        let mut s = h.board.lock();
        s.guest = Some(ram.clone());
        s.ctx_shadow_addr_lower = 0x2000;
        s.host_write_addr_lower = 0x1000;
        red_triangle(&mut s, &st);
        for word in [0x4c0, 7, 0x10a5a, 3, 0x5a5a] {
            cmd(&mut s, word);
        }
    }
    assert!(start.elapsed() < Duration::from_millis(250), "the front end waited for the raster unit");
    assert_eq!(ram.half(0x1006), 0, "the sync word waits for the draw before it");
    eventually("the query did not sync", || ram.half(0x1006) == 0x5a5a);
    assert_eq!(ram.get(0x2000 + 0x97c, 4), 7u32.to_be_bytes(), "the counter was published before the sync word");
    let s = h.board.lock();
    let _planes = h.raster.planes();
    assert_ne!(pixel(&s, 0, 10, 10), 0, "the draw was in the planes before the sync word");
    drop(_planes);
    drop(s);
    h.raster.stop();
}

/// On the bus, a waiting read behind a queued draw answers busy until the
/// raster unit has drawn it, without the processor waiting inside the
/// read; the memory word answers at once.
#[test]
fn a_waiting_bus_read_is_busy_until_the_raster_unit_has_drawn() {
    let (h, st) = slow_raster(300);
    h.work_start();

    red_triangle(&mut h.board.lock(), &st);
    assert_eq!(h.bus_read(ODSY_BUZZ_1, 4), None, "busy while the draw is queued");
    assert_eq!(h.raster.pending(), 1, "and the read did not wait for it");
    assert_eq!(h.bus_read(MEMORY_CONFIG, 4).map(|v| v & 0xff00_0000), Some(0xa100_0000));
    eventually("the read stayed busy", || h.bus_read(ODSY_BUZZ_1, 4).is_some());
    assert_ne!(pixel(&h.board.lock(), 0, 10, 10), 0, "drawn before the read answered");
    h.work_stop();
}

/// The rows the display would resolve after the draws so far, for a board
/// whose windows show the buffers `displayed` names.
fn drawn_rows(h: &BoardLocks, displayed: u32) -> Vec<u32> {
    let mut rows = Vec::new();
    h.raster.planes().take_rows(displayed, |y| rows.push(y));
    rows
}

/// Which rows of colour buffer b differ between two snapshots.
fn changed_rows(s: &Board, b: u32, before: &[u32]) -> Vec<u32> {
    let w = 64;
    (0..SGI_ODYSSEY_HEIGHT).filter(|&y| (0..w).any(|x| pixel(s, b, x, y as i32) != before[(y * w as u32 + x as u32) as usize])).collect()
}

/// The pixels of the first 64 columns of colour buffer b.
fn snapshot(s: &Board, b: u32) -> Vec<u32> {
    (0..SGI_ODYSSEY_HEIGHT as i32).flat_map(|y| (0..64).map(move |x| (y, x))).map(|(y, x)| pixel(s, b, x, y)).collect()
}

/// A small triangle at window row `y`, in the first 64 columns.
fn small_triangle(s: &mut Board, st: &GlDrawState, y: f32, colour: [f32; 4]) {
    let v = vertices(&[(2.0, y), (60.0, y + 3.0), (4.0, y + 40.0)], colour);
    let stipple = [0u32; 32];
    let mut d = GlDraw::new(SGI_GL_TRIANGLES, 3, &v, st, &stipple);
    assert_eq!(s.gl_draw(&mut d), SGI_GL_DRAW_OK);
}

/// What the display resolves again covers every row of the colour buffer
/// the windows show that a draw changed, and a draw into a buffer no window
/// shows asks for none: its window shows it only after a swap, which
/// repaints the frame.
#[test]
fn a_draw_asks_the_display_for_the_rows_it_changed_in_a_buffer_a_window_shows() {
    for (target, displayed, shown) in [(0, 0, true), (1, 0, false), (1, 0xffff_ffff, true), (0, 0xffff_ffff, false), (1, 1, true)] {
        let (s, mut st) = gl_board();
        let h = BoardLocks::new(s);
        st.drawable.front = target;
        st.drawable.back = target;
        let mut s = h.board.lock();
        let before = snapshot(&s, target);

        small_triangle(&mut s, &st, 100.0, RED);
        small_triangle(&mut s, &st, 400.0, RED);
        let changed = changed_rows(&s, target, &before);
        let rows = drawn_rows(&h, displayed);

        assert!(changed.len() > 40, "the triangles drew");
        if shown {
            assert!(changed.iter().all(|y| rows.contains(y)), "target {target} displayed {displayed:#x}: rows {changed:?} changed, {rows:?} asked for");
            assert!(rows.len() < 200, "and not the whole frame: {}", rows.len());
        } else {
            assert!(rows.is_empty(), "target {target} displayed {displayed:#x}: no window shows it, yet {rows:?}");
        }
        assert!(drawn_rows(&h, displayed).is_empty(), "the rows are handed over once");
    }
}

/// A draw under a window-id test reaches only the windows that test names:
/// into the buffer the window it names does not show, it asks for nothing,
/// even while other windows show that buffer.
#[test]
fn a_draw_under_an_id_test_asks_only_for_the_window_it_reaches() {
    let (mut s, mut st) = gl_board();
    s.did.fill(3 << ODSY_DID_MAIN_SHIFT);
    s.id_compare = ODSY_ID_COMPARE_ENABLE | 0xf8 << ODSY_ID_COMPARE_MASK_SHIFT | 3 << ODSY_DID_MAIN_SHIFT;
    st.drawable.front = 1;
    st.drawable.back = 1;
    let h = BoardLocks::new(s);
    let mut s = h.board.lock();

    small_triangle(&mut s, &st, 100.0, RED);
    assert_ne!(pixel(&s, 1, 10, 110), 0, "the draw passed the id test");
    assert!(drawn_rows(&h, !(1 << 3)).is_empty(), "window 3 shows buffer 0 and the others show buffer 1");
    small_triangle(&mut s, &st, 100.0, RED);
    assert!(drawn_rows(&h, 1 << 3).contains(&110), "window 3 shows buffer 1");
}

/// A clear's rows reach the display the same way.
#[test]
fn a_clear_asks_the_display_for_its_rows() {
    let (s, _) = gl_board();
    let h = BoardLocks::new(s);
    let mut s = h.board.lock();
    let mut op = RasterOp {
        kind: SGI_RASTER_CLEAR,
        ntargets: 1,
        x0: 0,
        y0: 200,
        x1: 63,
        y1: 299,
        clear_buffers: SGI_GL_COLOR_BUFFER_BIT,
        clear_rgba: [0.2, 0.3, 0.4, 1.0],
        clear_colour_mask: SGI_GL_COLOUR_MASK_ALL,
        ..Default::default()
    };
    op.clip.n = 1;
    op.clip.r[0] = ClipRect { x0: 0, y0: 0, x1: 1279, y1: 1023 };

    s.gl_raster(&op);
    h.raster.wait_idle();
    let rows = drawn_rows(&h, 0);
    assert_eq!(rows, (200..300).collect::<Vec<_>>());
}
