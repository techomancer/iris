//! The raster unit's queue: GL work runs on its own thread, and the
//! orderings the guest can see hold across both queues.

use super::*;
use crate::dev::odyssey::raster_bench::Bench;
use std::sync::atomic::Ordering::SeqCst;
use std::sync::atomic::AtomicU64;
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

/// The rows the raster unit has published, as the display would take them,
/// found without waiting for a slice in flight.
fn published_rows(h: &BoardLocks) -> Vec<u32> {
    let mut scanout = fresh_scanout();
    h.raster.take_published_rows(0, &mut scanout.differ, &scanout.marks);
    scanout.marks.dirty_rows().into_iter().map(|y| y as u32).collect()
}

/// A job that holds the raster unit for `ms`, as a long draw would.
fn busy(h: &BoardLocks, ms: u64) {
    h.raster.submit(move |_| std::thread::sleep(Duration::from_millis(ms)));
}

/// The rows a draw wrote are published before the fence behind it writes
/// its token: a guest that sees the token and arms a swap has all of its
/// rows known to the display. A job queued behind the fence keeps the
/// queue from draining and the slice from ending meanwhile.
#[test]
fn a_fence_publishes_the_rows_drawn_before_it() {
    // The first slice waits while everything is queued, and then runs it all.
    let (h, st) = slow_raster(50);
    h.raster.test_slice_limit_ms.store(10_000, SeqCst);
    let ram = Ram::new(1 << 20);
    {
        let mut s = h.board.lock();
        s.guest = Some(ram.clone());
        s.host_write_addr_lower = 0x1000;
        small_triangle(&mut s, &st, 100.0, RED);
        s.fence(0x5a5a << ODSY_DMA_FENCE_TOKEN_SHIFT | 3);
        busy(&h, 300);
    }
    eventually("the fence did not land", || ram.half(0x1006) == 0x5a5a);
    assert!(published_rows(&h).contains(&110), "the token was visible before the draw's rows were");
    h.raster.stop();
}

/// A query's answer is a completion too.
#[test]
fn a_query_publishes_the_rows_drawn_before_it() {
    let (h, st) = slow_raster(50);
    h.raster.test_slice_limit_ms.store(10_000, SeqCst);
    let ram = Ram::new(1 << 20);
    {
        let mut s = h.board.lock();
        s.guest = Some(ram.clone());
        s.ctx_shadow_addr_lower = 0x2000;
        s.host_write_addr_lower = 0x1000;
        small_triangle(&mut s, &st, 100.0, RED);
        for word in [0x4c0, 7, 0x10a5a, 3, 0x5a5a] {
            cmd(&mut s, word);
        }
        busy(&h, 300);
    }
    eventually("the query did not sync", || ram.half(0x1006) == 0x5a5a);
    assert!(published_rows(&h).contains(&110), "the sync word was visible before the draw's rows were published");
    h.raster.stop();
}

/// The queue publishes when it drains, before a caller that waited for it
/// looks at the planes.
#[test]
fn a_drained_queue_has_published_its_rows() {
    let (h, st) = slow_raster(50);
    h.raster.test_slice_limit_ms.store(10_000, SeqCst);
    h.raster.test_slice_end_ms.store(400, SeqCst);
    small_triangle(&mut h.board.lock(), &st, 100.0, RED);
    h.raster.wait_idle();
    assert!(published_rows(&h).contains(&110), "the queue was idle but its rows were not out");
    h.raster.test_slice_end_ms.store(0, SeqCst);
    h.raster.stop();
}

/// A slice that ends with work still queued publishes what it drew, so the
/// display never goes a long job without the rows of the short ones.
#[test]
fn a_slice_end_publishes_the_rows_drawn_in_it() {
    let (h, st) = slow_raster(50);
    {
        let mut s = h.board.lock();
        small_triangle(&mut s, &st, 100.0, RED);
        busy(&h, 5);
        busy(&h, 400);
    }
    eventually("the first jobs did not finish", || h.raster.pending() == 1);
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(h.raster.pending(), 1, "the long job is still running");
    assert!(published_rows(&h).contains(&110), "the slice ended without its rows");
    h.raster.stop();
}

/// A scanout with the whole frame resolved.
struct FreshRows {
    differ: Box<Differ>,
    marks: RowMarks,
}

fn fresh_scanout() -> FreshRows {
    let marks = RowMarks::default();
    marks.set_height(SGI_ODYSSEY_HEIGHT);
    FreshRows { differ: Box::new([[0; ROW_WORDS]; ODSY_DISPMODE_COUNT as usize]), marks }
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
    let mut scanout = fresh_scanout();
    planes.take_rows(0, &mut scanout.differ, &scanout.marks);
    let rows: Vec<u32> = scanout.marks.dirty_rows().into_iter().map(|y| y as u32).collect();
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

/// A current-state query behind a queued draw (a small one: a draw of a
/// screenful makes the front end wait for room): the front end goes on at
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
        tiny_triangle(&mut s, &st);
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
    let mut scanout = fresh_scanout();
    h.raster.planes().take_rows(displayed, &mut scanout.differ, &scanout.marks);
    scanout.marks.dirty_rows().into_iter().map(|y| y as u32).collect()
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

/// The pixels a one-pixel line lights under line rule `rule`, in the
/// rectangle of the board's first 80 columns and 64 rows, as (x, y) pairs.
fn line_pixels(rule: LineRule, a: (f32, f32), b: (f32, f32)) -> Vec<(i32, i32)> {
    let mut bench = Bench::empty();
    bench.board.gl_write_mask |= ODSY_WRITE_COLOUR_MASK << ODSY_WRITE_COLOUR_SHIFT;
    bench.st.shade_model = SGI_GL_FLAT;
    bench.st.dither = 0;
    bench.rules.line_rule = rule as u8;
    bench.mode = SGI_GL_LINES;
    bench.prims = vec![vertices(&[a, b], RED)];
    bench.run();
    (0..64).flat_map(|y| (0..80).map(move |x| (x, y))).filter(|&(x, y)| pixel(&bench.board, 0, x, y) != 0).collect()
}

/// A one-pixel line takes the centre-sampled rule: a pixel per major-axis
/// column whose centre lies between the ends, the end not included. A line
/// that ends past a column centre but inside that pixel's diamond lights the
/// column, where the diamond-exit rule leaves it; a crossing exactly on a
/// pixel edge goes to the lower pixel whichever way the line slopes, where
/// the diamond rule's perturbation follows the slope. Reversed, the end
/// inside a diamond is the start, and both rules agree. A vertical line
/// exactly on a column edge takes the column to its left under both. Cases
/// and expectations from the QEMU sgi-origami branch's line tests
/// (ab9b3f98f25f, c5417243927d).
#[test]
fn a_line_takes_the_centre_sampled_rule() {
    let cases: [((f32, f32), (f32, f32), bool, Vec<(i32, i32)>); 4] = [
        ((1.0, 3.5), (5.75, 3.5), true, vec![(1, 3), (2, 3), (3, 3), (4, 3), (5, 3)]),
        ((5.75, 3.5), (1.0, 3.5), false, vec![(1, 3), (2, 3), (3, 3), (4, 3), (5, 3)]),
        ((1.5, 2.0), (5.5, 4.0), true, vec![(1, 1), (2, 2), (3, 2), (4, 3)]),
        ((4.0, 1.0), (4.0, 5.0), false, vec![(3, 1), (3, 2), (3, 3), (3, 4)]),
    ];

    for (a, b, diamond_differs, want) in cases {
        let centre = line_pixels(SGI_LINE_CENTRE_SAMPLED, a, b);
        let diamond = line_pixels(SGI_LINE_COVERAGE_DOUBLE, a, b);

        assert_eq!(centre, want, "centre-sampled {a:?} to {b:?}");
        assert_eq!(diamond != want, diamond_differs, "diamond-exit {a:?} to {b:?}: {diamond:?}");
    }
}

/// Centre-sampled lines at the board's line-test geometry: a horizontal
/// line from x 60.0 to 70.75 lights columns 60 to 70, eleven pixels, the one
/// past the last column centre included; a line rising at slope one half
/// crosses a row edge exactly at every other column centre, and each of those
/// columns takes the lower row, sharing it with the column after.
#[test]
fn centre_sampled_lines_light_the_pixels_the_rule_names() {
    let horizontal = line_pixels(SGI_LINE_CENTRE_SAMPLED, (60.0, 10.5), (70.75, 10.5));
    assert_eq!(horizontal, (60..=70).map(|x| (x, 10)).collect::<Vec<_>>());

    let rising = line_pixels(SGI_LINE_CENTRE_SAMPLED, (60.0, 9.75), (68.0, 13.75));
    assert_eq!(rising, [(60, 9), (61, 10), (62, 10), (63, 11), (64, 11), (65, 12), (66, 12), (67, 13)]);
}

/// Arm a swap of every window and let the retrace latch it.
fn swap_every_window(s: &mut Board) {
    s.front.lock().swap_arm = u32::MAX;
    s.retrace();
}

/// The rows of the frame a display update resolved, found by the marker
/// the test left in every row first.
fn resolved_rows(s: &mut Board, h: &BoardLocks) -> Vec<u32> {
    const MARK: u32 = 0xdead_0000;
    let (w, ht) = (s.reader.frame_width as usize, s.reader.frame_height as usize);

    s.reader.frame.fill(MARK);
    let displayed = s.front.lock().displayed_buffer;
    s.test_take_rows(h, displayed);
    s.scanout_update();
    (0..ht as u32).filter(|&y| s.reader.frame[y as usize * w] != MARK).collect()
}

/// A swap resolves the rows where the window's two buffers differ and no
/// others, however the buffers came to differ: a draw into the buffer
/// the window does not show asks for nothing until the swap, then the
/// swap resolves those rows, and the swap back resolves them again.
#[test]
fn a_swap_resolves_the_rows_drawn_into_the_other_buffer_and_no_others() {
    let (mut s, mut st) = gl_board();
    s.set_scanout_size(SGI_ODYSSEY_WIDTH, SGI_ODYSSEY_HEIGHT);
    let h = BoardLocks::new(s);
    let mut s = h.board.lock();
    st.drawable.front = 1;
    st.drawable.back = 1;
    s.scanout_update();
    // A swap and its swap back measure the buffers; from then on the draws'
    // own rows are all the display knows.
    swap_every_window(&mut s);
    s.scanout_update();
    swap_every_window(&mut s);
    s.scanout_update();
    assert!(!s.scanout.marks.unnamed_pending());

    let before = snapshot(&s, 1);
    small_triangle(&mut s, &st, 100.0, RED);
    small_triangle(&mut s, &st, 400.0, RED);
    let changed = changed_rows(&s, 1, &before);
    assert!(changed.len() > 40, "the triangles drew");

    assert!(resolved_rows(&mut s, &h).is_empty(), "the window shows buffer 0, which they did not change");
    swap_every_window(&mut s);
    assert_eq!(resolved_rows(&mut s, &h), changed, "the swap shows buffer 1");
    assert!(resolved_rows(&mut s, &h).is_empty(), "and nothing is left to resolve");
    swap_every_window(&mut s);
    assert_eq!(resolved_rows(&mut s, &h), changed, "the swap back shows buffer 0 in those rows");
}

/// A writer that does not name its rows makes the next swap measure the
/// buffers: it resolves the rows they differ in, not the whole frame.
#[test]
fn a_swap_after_an_unnamed_write_resolves_only_the_rows_the_buffers_differ_in() {
    let (mut s, _) = gl_board();
    s.set_scanout_size(SGI_ODYSSEY_WIDTH, SGI_ODYSSEY_HEIGHT);
    let h = BoardLocks::new(s);
    let mut s = h.board.lock();
    s.scanout_update();

    for y in 50..60 {
        s.buffer_store(1, 10, y, 0x1234);
    }
    s.scanout.invalidate(50, 10);
    assert_eq!(resolved_rows(&mut s, &h), (50..60).collect::<Vec<_>>());
    swap_every_window(&mut s);
    assert_eq!(resolved_rows(&mut s, &h), (50..60).collect::<Vec<_>>());
}

/// A front-end pixel write names only its display line: the next swap
/// measures that line and takes the rest from the draws' own rows.
#[test]
fn a_swap_measures_only_the_lines_an_unnamed_write_marked() {
    let (mut s, mut st) = gl_board();
    s.set_scanout_size(SGI_ODYSSEY_WIDTH, SGI_ODYSSEY_HEIGHT);
    let h = BoardLocks::new(s);
    let mut s = h.board.lock();
    st.drawable.front = 1;
    st.drawable.back = 1;
    swap_every_window(&mut s);
    s.scanout_update();
    swap_every_window(&mut s);
    s.scanout_update();

    let before = snapshot(&s, 1);
    small_triangle(&mut s, &st, 100.0, RED);
    let mut want = changed_rows(&s, 1, &before);
    s.buffer_store(1, 300, 700, 0x1234);
    s.scanout.invalidate(700, 1);
    want.push(700);
    // A line that matches in both buffers is marked and found equal.
    s.buffer_store(1, 300, 800, 0);
    s.scanout.invalidate(800, 1);

    assert_eq!(resolved_rows(&mut s, &h), vec![700, 800], "the written lines repaint");
    swap_every_window(&mut s);
    assert_eq!(resolved_rows(&mut s, &h), want, "the swap shows the drawn rows and the line that differs, not the one that does not");
    assert!(!s.scanout.marks.unnamed_pending(), "the swap measured the marked lines");
}

/// Whatever the draws, unnamed writes, swaps and lookup changes, the frame
/// the incremental updates keep is the one a whole resolve makes.
#[test]
fn incremental_updates_keep_the_frame_a_whole_resolve_makes() {
    use crate::dev::odyssey::raster_bench::Lcg;
    let (mut s, mut st) = gl_board();
    s.set_scanout_size(SGI_ODYSSEY_WIDTH, SGI_ODYSSEY_HEIGHT);
    let mut rng = Lcg(0x5a17);
    for v in s.regs().edit().lut.iter_mut().take(4096) {
        *v = rng.next_u32() & ODSY_LUT_ENTRY_MASK;
    }
    // Four windows in vertical bands.
    for (i, d) in s.did.iter_mut().enumerate() {
        *d = (((i % SGI_ODYSSEY_WIDTH as usize) / 320) as u8) << ODSY_DID_MAIN_SHIFT;
    }
    let h = BoardLocks::new(s);
    let mut s = h.board.lock();
    s.scanout_update();

    for step in 0..120 {
        match rng.next_u32() % 6 {
            0 | 1 => {
                let target = rng.next_u32() & 1;
                st.drawable.front = target;
                st.drawable.back = target;
                let y = (rng.next_u32() % 900) as f32;
                small_triangle(&mut s, &st, y, [(rng.next_u32() % 2) as f32, (rng.next_u32() % 2) as f32, 1.0, 1.0]);
            }
            2 => {
                let (x, y) = ((rng.next_u32() % 1200) as i64, (rng.next_u32() % 1000) as i64);
                s.buffer_store(rng.next_u32() & 1, x, y, rng.next_u32());
                s.scanout.invalidate(y as i32, 1);
            }
            3 => {
                let i = (rng.next_u32() % 4096) as usize;
                s.regs().edit().lut[i] = rng.next_u32() & ODSY_LUT_ENTRY_MASK;
                s.scanout.repaint_all();
            }
            _ => {
                s.front.lock().swap_arm = rng.next_u32() & 0xf;
                s.retrace();
            }
        }
        if rng.next_u32() % 2 == 0 {
            h.raster.wait_idle();
            let displayed = s.front.lock().displayed_buffer;
            s.test_take_rows(&h, displayed);
            s.scanout_update();
            let w = s.reader.frame_width;
            let mut want = vec![0u32; w as usize];
            for y in 0..s.reader.frame_height {
                s.resolve_row_generic(y, &mut want);
                assert_eq!(&s.reader.frame[(y * w) as usize..((y + 1) * w) as usize], &want[..], "step {step}, row {y}");
            }
        }
    }
}

/// The screen the display keeps holds the frame, cursor and all, however
/// few lines each frame presents: the first frame takes the whole picture
/// and the ones after it only the lines they resolved.
#[test]
fn the_screen_follows_the_frame_through_frames_that_present_a_few_lines() {
    use crate::dev::odyssey::raster_bench::Lcg;
    use std::sync::atomic::AtomicU64;
    let (mut s, mut st) = gl_board();
    s.set_scanout_size(SGI_ODYSSEY_WIDTH, SGI_ODYSSEY_HEIGHT);
    let mut rng = Lcg(0xfeed);
    for v in s.regs().edit().lut.iter_mut().take(4096) {
        *v = rng.next_u32() & ODSY_LUT_ENTRY_MASK;
    }
    s.regs().edit().cursor[ODSY_CURSOR_POSITION as usize] = 100 << 16 | 40;
    for i in 0..ODSY_CURSOR_COLOUR_COUNT {
        s.regs().edit().cursor[(ODSY_CURSOR_COLOURS + 1 + i) as usize] = rng.next_u32() & ODSY_LUT_ENTRY_MASK;
    }
    for w in 0..128u32 {
        s.regs().edit().cursor[(ODSY_CURSOR_GLYPH + w) as usize] = rng.next_u32();
    }
    s.regs().edit().cursor[ODSY_CURSOR_CONTROL as usize] = ODSY_CURSOR_CTL_ENABLE;
    let h = BoardLocks::new(s);
    let display = Display::new(Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)));
    let (w, ht) = (SGI_ODYSSEY_WIDTH as usize, SGI_ODYSSEY_HEIGHT as usize);

    for step in 0..8 {
        {
            let mut s = h.board.lock();
            let target = step & 1;
            st.drawable.front = target;
            st.drawable.back = target;
            small_triangle(&mut s, &st, 20.0 + 120.0 * step as f32, [1.0, (step & 1) as f32, 1.0, 1.0]);
            if step == 5 {
                s.regs().edit().cursor[ODSY_CURSOR_POSITION as usize] = 200 << 16 | 500;
                let span = s.cursor_span();
                s.cursor_invalidate(span);
            }
            if step % 3 == 2 {
                s.front.lock().swap_arm = u32::MAX;
            }
        }
        h.raster.wait_idle();
        Odyssey::frame(&h, &display);
        let reader = display.reader.lock();
        let screen = display.screen.lock();
        for y in 0..ht {
            for x in 0..w {
                let rgb = reader.frame[y * w + x];
                let want = 0xff00_0000 | (rgb & 0xff) << 16 | rgb & 0xff00 | (rgb >> 16) & 0xff;
                let at = y * crate::disp::FB_STRIDE + x;
                assert_eq!((screen.fb_rgb[at], screen.rgba[at]), (want, want), "step {step} ({x}, {y})");
            }
        }
    }
}

/// The drawable comparison that runs before every command looks at the clip
/// rectangles the list counts and no others: a change inside the count is
/// seen, and one past it, where nothing is drawn against, is not.
#[test]
fn the_drawable_comparison_covers_the_rectangles_the_list_counts() {
    let mut a = GlDrawable::default();
    a.clip.n = 2;
    a.clip.r[0] = ClipRect { x0: 0, y0: 0, x1: 9, y1: 9 };
    a.clip.r[1] = ClipRect { x0: 20, y0: 0, x1: 29, y1: 9 };
    let mut b = a;
    assert!(a.drawable_eq(&b));

    b.clip.r[5] = ClipRect { x0: 1, y0: 2, x1: 3, y1: 4 };
    assert!(a.drawable_eq(&b), "past the count");

    for i in 0..2 {
        let mut c = a;
        c.clip.r[i].y1 += 1;
        assert!(!a.drawable_eq(&c), "rectangle {i}");
    }
    let mut c = a;
    c.clip.n = 3;
    assert!(!a.drawable_eq(&c), "the count");
}

/// A texture chain a setup filled is the default one again after `clear`,
/// though `clear` writes only the levels the setup touched: a deeper
/// texture's levels do not leak into the next draw's sampling, which reads
/// every volume's slice count and any level by number.
#[test]
fn a_cleared_texture_chain_is_the_default_one_again() {
    let mut chain = TextureChain::default();

    for i in [0u32, 1, 6] {
        chain.touch(i);
        chain.volumes[i as usize] = TextureVolume { slices: 3, width: 8 >> i.min(3), ..Default::default() };
        chain.storage[i as usize] = Texture { offset: 5, width: 4, border_rows: true, ..Default::default() };
        chain.levels[i as usize] = GlTexLevel { level: i, width: 4, read_texel: true, ..Default::default() };
    }
    chain.volume = chain.volumes[0];
    chain.pyramid = Some(PyramidShape { count: 7, ..Default::default() });
    chain.detail_storage = Texture { stride: 32, ..Default::default() };
    chain.detail_image = GlTexLevel { detail: true, ..Default::default() };

    chain.clear();
    let fresh = TextureChain::default();

    assert!(chain.volumes == fresh.volumes && chain.storage == fresh.storage && chain.levels == fresh.levels);
    assert!(chain.volume == fresh.volume && chain.pyramid.is_none());
    assert!(chain.detail_storage == fresh.detail_storage && chain.detail_image == fresh.detail_image);
    // A second clear has nothing left to write.
    chain.clear();
    assert!(chain.levels == fresh.levels);
}

/// Lighting a vertex reuses one lighting state and puts back only the lights
/// the last vertex enabled, and the colour is what a fresh state gives:
/// across lights switching on, changing and off, with a spot, a positional
/// and a directional light, and with none.
#[test]
fn a_reused_lighting_state_lights_a_vertex_as_a_fresh_one_does() {
    use crate::dev::gl::light::Lighting;

    fn fresh(s: &Board, obj: &[f64; 4]) -> [f32; 4] {
        let mut lt = Lighting::default();
        lt.init();
        let v = s.gl_light_fill(&mut lt, obj);
        let (front, _) = lt.light_vertex(v.eye, v.normal, v.colour);
        [front[0], front[1], front[2], v.alpha]
    }
    fn set_light(s: &mut Board, i: usize, colour: [f32; 3], directional: bool, spot: bool) {
        for k in 0..3 {
            s.gl_light_colour[i * 12 + k] = (0.1f32 * (i + 1) as f32).to_bits();
            s.gl_light_colour[i * 12 + 4 + k] = colour[k].to_bits();
            s.gl_light_colour[i * 12 + 8 + k] = (colour[k] * 0.5).to_bits();
        }
        let blk = &mut s.gl_light_vec[i * ODSY_GL_LIGHT_VEC_FLOATS as usize..];
        for k in 0..3 {
            blk[ODSY_LIGHT_POS + k] = [2.0f32, 3.0, 4.0 + i as f32][k].to_bits();
            blk[ODSY_LIGHT_HALF + k] = (if directional { [0.0f32, 0.6, 0.8][k] } else { 0.0 }).to_bits();
        }
        blk[ODSY_LIGHT_ATTEN] = 1.0f32.to_bits();
        blk[ODSY_LIGHT_ATTEN + 1] = 0.1f32.to_bits();
        blk[ODSY_LIGHT_ATTEN + 2] = 0.01f32.to_bits();
        blk[ODSY_LIGHT_SPOT_COS] = (if spot { -0.5f32 } else { 1.0 }).to_bits();
        blk[ODSY_LIGHT_SPOT_EXP] = (-2.0f32).to_bits();
        for k in 0..3 {
            blk[ODSY_LIGHT_SPOT_DIR + k] = [0.0f32, 0.0, 1.0][k].to_bits();
        }
    }
    fn clear_light(s: &mut Board, i: usize) {
        s.gl_light_colour[i * 12..i * 12 + 12].fill(0);
    }

    let (mut s, _) = board("v12");
    let identity: [f32; 16] = [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.];
    s.gl_modelview = identity.map(f32::to_bits);
    s.gl_normal_matrix = identity.map(f32::to_bits);
    s.gl_normal = [0.0f32, 0.6, 0.8].map(f32::to_bits);
    s.gl_colour = [0.7f32, 0.5, 0.3, 0.9].map(f32::to_bits);
    for k in 0..4 {
        s.gl_material[k] = 0.2f32.to_bits();
        s.gl_material[4 + k] = [0.8f32, 0.4, 0.6, 0.5][k].to_bits();
        s.gl_material[8 + k] = 0.5f32.to_bits();
    }
    s.gl_shininess_sqrt = 4.0f32.to_bits();
    let obj = [1.0, 2.0, 3.0, 1.0];
    let mut seen = Vec::new();
    let mut out = [0f32; 4];

    let check = |s: &mut Board, what: &str, out: &mut [f32; 4]| {
        s.gl_light(&obj, out);
        let want = fresh(s, &obj);
        assert_eq!(out.map(f32::to_bits), want.map(f32::to_bits), "{what}");
        want
    };

    seen.push(check(&mut s, "no lights", &mut out));
    set_light(&mut s, 0, [1.0, 0.2, 0.1], true, false);
    set_light(&mut s, 3, [0.3, 0.9, 0.2], false, true);
    seen.push(check(&mut s, "a directional light and a spot", &mut out));
    seen.push(check(&mut s, "the same again", &mut out));
    clear_light(&mut s, 0);
    set_light(&mut s, 5, [0.2, 0.2, 0.9], false, false);
    seen.push(check(&mut s, "light 0 off, light 5 on", &mut out));
    clear_light(&mut s, 3);
    clear_light(&mut s, 5);
    seen.push(check(&mut s, "all off again", &mut out));
    // Light 3 was a spot and returns as a plain positional light: nothing of
    // its spot may be left over.
    set_light(&mut s, 3, [0.3, 0.9, 0.2], false, false);
    seen.push(check(&mut s, "light 3 back without its spot", &mut out));
    clear_light(&mut s, 3);
    s.gl_colour_material = ODSY_COLOUR_MATERIAL_DIFFUSE;
    set_light(&mut s, 7, [0.6, 0.6, 0.6], false, false);
    seen.push(check(&mut s, "colour material, light 7", &mut out));

    // The cases differ, so the comparisons above could have failed.
    assert_ne!(seen[0], seen[1]);
    assert_ne!(seen[1], seen[3]);
}

/// A triangle a few pixels across.
fn tiny_triangle(s: &mut Board, st: &GlDrawState) {
    let v = vertices(&[(10.0, 10.0), (14.0, 10.0), (10.0, 14.0)], RED);
    let stipple = [0u32; 32];
    let mut d = GlDraw::new(SGI_GL_TRIANGLES, 3, &v, st, &stipple);
    assert_eq!(s.gl_draw(&mut d), SGI_GL_DRAW_OK);
}

/// A draw's cost follows its area; a clear's pixels count for less.
#[test]
fn a_job_costs_its_pixels() {
    let p = |mode, pts: &[(f64, f64)]| primitive_cost(mode, pts.iter().map(|&p| Some(p)), 1.0, 1.0);

    assert_eq!(p(SGI_GL_TRIANGLES, &[(0.0, 0.0), (100.0, 0.0), (0.0, 100.0)]), RASTER_JOB_OVERHEAD + 5000);
    assert_eq!(p(SGI_GL_QUADS, &[(0.0, 0.0), (100.0, 0.0), (100.0, 10.0), (0.0, 10.0)]), RASTER_JOB_OVERHEAD + 1000);
    assert_eq!(p(SGI_GL_LINES, &[(0.0, 0.0), (99.0, 3.0)]), RASTER_JOB_OVERHEAD + 100);
    // Off the screen counts for nothing; a vertex behind the eye, for all of it.
    assert_eq!(p(SGI_GL_QUADS, &[(-500.0, -500.0), (-400.0, -500.0), (-400.0, -400.0), (-500.0, -400.0)]), RASTER_JOB_OVERHEAD);
    assert_eq!(primitive_cost(SGI_GL_TRIANGLES, [Some((0.0, 0.0)), None, Some((1.0, 1.0))].into_iter(), 1.0, 1.0), RASTER_JOB_OVERHEAD + SGI_ODYSSEY_WIDTH * SGI_ODYSSEY_HEIGHT);
    assert_eq!(rect_cost(0, 0, 80, 80, true), RASTER_JOB_OVERHEAD + 80 * 80 / RASTER_CLEAR_DIVISOR);
    assert_eq!(rect_cost(0, 0, 80, 80, false), RASTER_JOB_OVERHEAD + 80 * 80);
}

/// Many small primitives queue behind a busy raster unit without the front
/// end waiting, however many there are in a frame; the queue's work is the
/// sum of theirs.
#[test]
fn many_small_jobs_queue_without_the_front_end_waiting() {
    let (h, st) = slow_raster(300);
    let start = Instant::now();
    let mut s = h.board.lock();

    for _ in 0..1000 {
        tiny_triangle(&mut s, &st);
    }
    assert!(start.elapsed() < Duration::from_millis(250), "the slice still holds the raster unit");
    assert_eq!(h.raster.pending(), 1000, "no small job waited for room");
    assert!(h.raster.queued() < RASTER_WORK_BUDGET, "{} queued", h.raster.queued());
    assert!(h.raster.queued() >= 1000 * RASTER_JOB_OVERHEAD as u64);
    drop(s);
    eventually("the queue did not drain", || h.raster.pending() == 0);
    assert_eq!(h.raster.queued(), 0);
    h.raster.stop();
}

/// One primitive that is a whole screen of work makes the next job wait
/// until the raster unit has drawn it.
#[test]
fn one_big_job_makes_the_front_end_wait() {
    let (h, st) = slow_raster(300);
    let start = Instant::now();
    {
        let mut s = h.board.lock();
        red_triangle(&mut s, &st);
        assert!(h.raster.queued() >= RASTER_WORK_BUDGET, "{} queued", h.raster.queued());
        assert!(start.elapsed() < Duration::from_millis(250), "the first job does not wait");
        tiny_triangle(&mut s, &st);
        assert!(start.elapsed() >= Duration::from_millis(250), "the second waited for the slice to draw");
        assert!(h.raster.queued() <= RASTER_WORK_BUDGET / 2, "{} queued", h.raster.queued());
    }
    eventually("the queue did not drain", || h.raster.pending() == 0);
    assert_ne!(pixel(&h.board.lock(), 0, 10, 10), 0);
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

/// A display frame that finds a raster slice running does not hold the
/// board lock while it waits for the slice to end: the front end and the
/// processor's accesses take the board meanwhile. Once the slice ends the
/// frame takes the planes and resolves the dirty lines.
#[test]
fn a_frame_waiting_for_a_slice_leaves_the_board_free() {
    let (h, st) = slow_raster(400);
    let display = Arc::new(Display::new(Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0))));

    red_triangle(&mut h.board.lock(), &st);
    eventually("the raster unit began its slice", || h.raster.in_slice());
    let frame = {
        let (h, display) = (h.clone(), display.clone());
        std::thread::spawn(move || Odyssey::frame(&h, &display))
    };
    std::thread::sleep(Duration::from_millis(60));

    let t = Instant::now();
    drop(h.board.lock());
    assert!(t.elapsed() < Duration::from_millis(150), "the frame held the board for the slice: {:?}", t.elapsed());
    assert!(h.raster.in_slice(), "the slice was still running");

    frame.join().unwrap();
    let reader = display.reader.lock();
    assert!(reader.stats.retries >= 1, "the frame let the board go to wait");
    assert_eq!(reader.stats.fallbacks, 0);
    assert_eq!(reader.stats.frames, 1, "and then staged its lines");
    drop(reader);
    h.raster.stop();
}

// The display with no lock held (IRIS_VPRO_DISPLAY=lockless), and the locked
// frame that runs the same code, held to the same pictures.

/// A board whose every drawable is an RGB visual through a gamma ramp, four
/// windows in vertical bands, and its display in `lockless` mode; the raster
/// unit running.
fn display_rig(lockless: bool) -> (BoardLocks, GlDrawState, Arc<Display>) {
    let (mut s, st) = gl_board();
    for i in 0..0x600u32 {
        let v = (i * 0x3ff / 0x5ff) & 0x3ff;
        s.regs().edit().lut[(ODSY_GAMMA_BASE + i) as usize] = v << 20 | v << 10 | v;
    }
    for id in 0..ODSY_DISPMODE_COUNT {
        s.regs().edit().dispmode[id as usize] = ODSY_GAMMA_GROUP << ODSY_DISPMODE_GROUP_SHIFT;
    }
    for (i, d) in s.did.iter_mut().enumerate() {
        *d = (((i % SGI_ODYSSEY_WIDTH as usize) / 320) as u8) << ODSY_DID_MAIN_SHIFT;
    }
    s.set_scanout_size(SGI_ODYSSEY_WIDTH, SGI_ODYSSEY_HEIGHT);
    let h = BoardLocks::new(s);
    h.raster.start();
    let display = Arc::new(Display::new(Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0))));
    display.set_lockless(lockless);
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

/// The presented frame against a whole per-pixel resolve of the planes.
fn assert_frame_is_whole_resolve(h: &BoardLocks, display: &Display, what: &str) {
    h.raster.wait_idle();
    let mut s = h.board.lock();
    let reader = display.reader.lock();
    let w = reader.frame_width as usize;
    let mut want = vec![0u32; w];
    for y in 0..reader.frame_height {
        s.resolve_row_generic(y, &mut want);
        assert_eq!(&reader.frame[y as usize * w..(y as usize + 1) * w], &want[..], "{what}: row {y}");
    }
}

/// One front-end thread, the raster unit, lookup and cursor stores and
/// swap arms run against `Odyssey::frame` with no pause; when the writers
/// stop and the raster queue drains the presented frame is a whole resolve
/// of the planes, and a hundred idle frames leave it so without resolving a
/// line.
fn stress(lockless: bool, seed: u64, last_is_lut: bool) {
    use crate::dev::odyssey::raster_bench::Lcg;
    let (h, mut st, display) = display_rig(lockless);
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let shown = run_display(&h, &display, &stop);
    let mut rng = Lcg(seed);
    let end = Instant::now() + Duration::from_millis(600);

    while Instant::now() < end {
        let mut s = h.board.lock();
        for _ in 0..8 {
            match rng.next_u32() % 9 {
                0 | 1 => {
                    let target = rng.next_u32() & 1;
                    st.drawable.front = target;
                    st.drawable.back = target;
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
                    s.scanout.invalidate(y as i32, 1);
                }
                5 => {
                    let i = rng.next_u32() % 0x600;
                    s.backend_data(ODSY_GAMMA_BASE + i, rng.next_u32() & ODSY_LUT_ENTRY_MASK);
                }
                6 => s.backend_data(ODSY_CURSOR_BASE + ODSY_CURSOR_POSITION, (rng.next_u32() % 1200) << 16 | (rng.next_u32() % 1000)),
                7 => {
                    let (x, y) = (rng.next_u32() % 1200, rng.next_u32() % 1000);
                    s.draw_mask = ODSY_DRAW_MASK_BUFFER0;
                    s.clear_planes(x, y, 40, 3, (rng.next_u32() & 0xf8) as u8, 0xff);
                }
                _ => h.front.lock().swap_arm |= rng.next_u32() & 0xf,
            }
        }
        drop(s);
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
            let mut s = h.board.lock();
            if !last_is_lut {
                s.draw_mask = ODSY_DRAW_MASK_BUFFER0 | ODSY_DRAW_MASK_BUFFER1;
                s.paint_pixel((rng.next_u32() % 1280) as i64, (rng.next_u32() % 1024) as i64, rng.next_u32());
            } else {
                // The entry every black pixel is looked up in.
                s.backend_data(ODSY_GAMMA_BASE, rng.next_u32() & ODSY_LUT_ENTRY_MASK);
            }
        }
        std::thread::sleep(Duration::from_micros(1500));
    }
    stop.store(true, SeqCst);
    shown.join().unwrap();
    h.raster.wait_idle();
    settle(&h, &display, 4);
    assert_frame_is_whole_resolve(&h, &display, "after the writers stopped");

    let (before, rows) = {
        let reader = display.reader.lock();
        (reader.frame.clone(), reader.stats.rows)
    };
    for _ in 0..100 {
        Odyssey::frame(&h, &display);
    }
    let reader = display.reader.lock();
    assert!(reader.frame == before, "an idle frame changed the picture");
    assert_eq!(reader.stats.rows, rows, "idle frames resolved lines nobody marked");
    drop(reader);
    assert_frame_is_whole_resolve(&h, &display, "after idle frames");
    h.raster.stop();
}

#[test]
fn the_locked_frame_ends_as_a_whole_resolve_under_live_writers() {
    for (seed, lut) in [(0x5a17, false), (0x77, true), (0xbeef, false), (0x31, true), (0x42, false), (0x99, true)] {
        stress(false, seed, lut);
    }
}

#[test]
fn the_lockless_frame_ends_as_a_whole_resolve_under_live_writers() {
    for (seed, lut) in [(0x5a17, false), (0x77, true), (0xbeef, false), (0x31, true), (0x42, false), (0x99, true)] {
        stress(true, seed, lut);
    }
}

/// A guest that waits for its fence before it arms a swap never sees a
/// frame that is not the buffer the latch put on the screen: each frame the
/// display makes is one colour, the one drawn into the buffer it shows.
fn swap_atomicity(lockless: bool) {
    let (h, mut st, display) = display_rig(lockless);
    // One window, so the one swap bit is the whole picture.
    {
        let mut s = h.board.lock();
        s.did.fill(0);
        s.mem.bytes.fill(0);
        s.buffer1_mem.bytes.fill(0);
        s.scanout.invalidate_all();
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
    let (wf, hf) = (SGI_ODYSSEY_WIDTH as f32, SGI_ODYSSEY_HEIGHT as f32);

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
            let mut s = h.board.lock();
            st.drawable.front = back;
            st.drawable.back = back;
            let v = vertices(&[(-wf, -hf), (3.0 * wf, -hf), (-wf, 3.0 * hf)], [red, 0.0, 0.0, 1.0]);
            let stipple = [0u32; 32];
            let mut d = GlDraw::new(SGI_GL_TRIANGLES, 3, &v, &st, &stipple);
            assert_eq!(s.gl_draw(&mut d), SGI_GL_DRAW_OK);
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
            let s = h.board.lock();
            let regs = s.regs().read();
            let r = i * 9;
            regs.gamma(r, ODSY_LUT_RED_SHIFT) << 16 | regs.gamma(0, ODSY_LUT_GREEN_SHIFT) << 8 | regs.gamma(0, ODSY_LUT_BLUE_SHIFT)
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
fn a_swap_after_a_fence_shows_the_whole_new_frame_with_the_locked_frame() {
    swap_atomicity(false);
}

#[test]
fn a_swap_after_a_fence_shows_the_whole_new_frame_with_no_lock() {
    swap_atomicity(true);
}

/// The frame waits for no slice and holds no lock the front end needs when
/// it runs with no lock: a slice that holds the raster unit for 400 ms
/// delays neither the board nor the frame.
#[test]
fn a_lockless_frame_waits_for_no_slice_and_takes_no_lock() {
    let (h, st) = slow_raster(400);
    let display = Arc::new(Display::new(Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0))));
    display.set_lockless(true);

    red_triangle(&mut h.board.lock(), &st);
    eventually("the raster unit began its slice", || h.raster.in_slice());
    let board = h.board.lock();
    let t = Instant::now();
    Odyssey::frame(&h, &display);
    assert!(t.elapsed() < Duration::from_millis(200), "the frame waited: {:?}", t.elapsed());
    drop(board);
    assert!(h.raster.in_slice(), "the slice was still running");
    let reader = display.reader.lock();
    assert_eq!(reader.stats.frames, 1, "and the frame staged its lines");
    assert_eq!((reader.stats.retries, reader.stats.fallbacks), (0, 0));
    drop(reader);
    h.raster.stop();
}

/// A lookup store that lands after the display took its marks and before it
/// copied the registers is in the frame it is making, and its repaint mark
/// is kept for the next: the frame ends up showing the new table. With the
/// registers copied first the store would be marked, taken and resolved
/// with the old table, and the picture would stay wrong.
#[test]
fn a_lookup_store_between_taking_the_marks_and_copying_the_registers_is_not_lost() {
    let (s, _) = gl_board();
    let h = BoardLocks::new(s);
    let display = Display::new(Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)));
    display.set_lockless(true);
    Odyssey::frame(&h, &display);
    {
        let shared = h.shared.clone();
        display.reader.lock().test_between = Some(Box::new(move || {
            shared.regs.edit().lut[0] = 0x0123_4567 & ODSY_LUT_ENTRY_MASK;
            shared.marks.repaint(DirtySite::Lut, 0, SGI_ODYSSEY_HEIGHT as i32);
        }));
    }
    h.shared.marks.repaint(DirtySite::Other, 0, SGI_ODYSSEY_HEIGHT as i32);
    Odyssey::frame(&h, &display);
    display.reader.lock().test_between = None;
    Odyssey::frame(&h, &display);
    Odyssey::frame(&h, &display);
    let want = h.shared.regs.read().lut_rgb(0);
    assert_ne!(want, 0);
    assert_eq!(display.reader.lock().frame[10 * SGI_ODYSSEY_WIDTH as usize + 10], want, "the picture shows the old table");
}
