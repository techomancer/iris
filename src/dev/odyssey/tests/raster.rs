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

/// A scanout with the whole frame resolved.
fn fresh_scanout() -> Scanout {
    let mut scanout = Scanout::default();
    scanout.set_mode(&ScanoutMode { width: SGI_ODYSSEY_WIDTH, height: SGI_ODYSSEY_HEIGHT, frame_clocks: 0, clock_hz: 0 });
    scanout.dirty.fill(false);
    scanout
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
    planes.take_rows(0, &mut scanout);
    let rows: Vec<u32> = (0..SGI_ODYSSEY_HEIGHT).filter(|&y| scanout.dirty[y as usize]).collect();
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
    let mut scanout = fresh_scanout();
    h.raster.planes().take_rows(displayed, &mut scanout);
    (0..SGI_ODYSSEY_HEIGHT).filter(|&y| scanout.dirty[y as usize]).collect()
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
    let (w, ht) = (s.scanout.frame_width as usize, s.scanout.frame_height as usize);

    s.scanout.frame.fill(MARK);
    let displayed = s.front.lock().displayed_buffer;
    h.raster.planes().take_rows(displayed, &mut s.scanout);
    s.scanout_update();
    (0..ht as u32).filter(|&y| s.scanout.frame[y as usize * w] != MARK).collect()
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
    assert!(s.scanout.unnamed.iter().all(|&w| w == 0));

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
    assert!(s.scanout.unnamed.iter().all(|&w| w == 0), "the swap measured the marked lines");
}

/// Whatever the draws, unnamed writes, swaps and lookup changes, the frame
/// the incremental updates keep is the one a whole resolve makes.
#[test]
fn incremental_updates_keep_the_frame_a_whole_resolve_makes() {
    use crate::dev::odyssey::raster_bench::Lcg;
    let (mut s, mut st) = gl_board();
    s.set_scanout_size(SGI_ODYSSEY_WIDTH, SGI_ODYSSEY_HEIGHT);
    let mut rng = Lcg(0x5a17);
    for v in s.lut.iter_mut().take(4096) {
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
                s.lut[i] = rng.next_u32() & ODSY_LUT_ENTRY_MASK;
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
            h.raster.planes().take_rows(displayed, &mut s.scanout);
            s.scanout_update();
            let w = s.scanout.frame_width;
            let mut want = vec![0u32; w as usize];
            for y in 0..s.scanout.frame_height {
                s.resolve_row_generic(y, &mut want);
                assert_eq!(&s.scanout.frame[(y * w) as usize..((y + 1) * w) as usize], &want[..], "step {step}, row {y}");
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
    for v in s.lut.iter_mut().take(4096) {
        *v = rng.next_u32() & ODSY_LUT_ENTRY_MASK;
    }
    s.cursor[ODSY_CURSOR_POSITION as usize] = 100 << 16 | 40;
    for i in 0..ODSY_CURSOR_COLOUR_COUNT {
        s.cursor[(ODSY_CURSOR_COLOURS + 1 + i) as usize] = rng.next_u32() & ODSY_LUT_ENTRY_MASK;
    }
    for w in 0..128u32 {
        s.cursor[(ODSY_CURSOR_GLYPH + w) as usize] = rng.next_u32();
    }
    s.cursor[ODSY_CURSOR_CONTROL as usize] = ODSY_CURSOR_CTL_ENABLE;
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
                s.cursor[ODSY_CURSOR_POSITION as usize] = 200 << 16 | 500;
                let span = s.cursor_span();
                s.cursor_invalidate(span);
            }
            if step % 3 == 2 {
                s.front.lock().swap_arm = u32::MAX;
            }
        }
        h.raster.wait_idle();
        Odyssey::frame(&h, &display);
        let s = h.board.lock();
        let screen = display.screen.lock();
        for y in 0..ht {
            for x in 0..w {
                let rgb = s.scanout.frame[y * w + x];
                let want = 0xff00_0000 | (rgb & 0xff) << 16 | rgb & 0xff00 | (rgb >> 16) & 0xff;
                let at = y * crate::disp::FB_STRIDE + x;
                assert_eq!((screen.fb_rgb[at], screen.rgba[at]), (want, want), "step {step} ({x}, {y})");
            }
        }
    }
}
