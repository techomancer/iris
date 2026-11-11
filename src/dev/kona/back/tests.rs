//! The back end's tests: jobs through `KonaBackend::job`, its monitor and the
//! drawing, scheduling and display they reach.

use super::super::tests::{test_cfg, TestHost};
use std::sync::atomic::Ordering::Relaxed;
use super::super::Kona;
use super::*;
use std::sync::Arc;

/// A monitor writer a test can read afterwards.
#[derive(Clone, Default)]
struct Capture(Arc<parking_lot::Mutex<Vec<u8>>>);

impl Write for Capture {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.lock().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Capture {
    fn text(&self) -> String {
        String::from_utf8(self.0.lock().clone()).unwrap()
    }
}

fn board() -> (Arc<TestHost>, Arc<Kona>) {
    let cfg = test_cfg();
    let host = TestHost::new();
    let kona = Kona::new(cfg, host.clone(), host.clone(), crate::dev::kona::tests::test_status(), None).unwrap();
    (host, kona)
}

fn monitor(kona: &Kona, args: &[&str]) -> (bool, String) {
    let mut out = Vec::new();
    let known = kona.with_backend(|b| b.monitor(args, &mut out)).unwrap();
    (known, String::from_utf8(out).unwrap())
}

/// A flush with nothing queued leaves the counters and the queue alone.
#[test]
fn a_flush_with_nothing_queued_changes_nothing() {
    let mut be = KonaBackend::new(test_cfg());
    be.prims.tris.reserve(64);
    let capacity = be.prims.tris.capacity();
    be.flush();
    assert_eq!((be.raster.draws, be.raster.triangles, be.raster.clears, be.raster.off_target), (0, 0, 0, 0));
    assert!(be.prims.state.is_none() && be.prims.tris.is_empty());
    assert_eq!(be.prims.tris.capacity(), capacity);
}

#[test]
fn the_config_follows_the_pipe() {
    assert_eq!(KonaConfig::ir2(1).unwrap().record_words(), 16);
    assert_eq!(KonaConfig::ir2(2).unwrap().record_words(), 32);
    let c = KonaConfig::default().with_pool(0, 8);
    assert_eq!((c.managed, c.threads, c.tile, c.texture_bytes_per_rm()), ((1280, 1024), 0, 8, 64 << 20));
}

/// The texture store holds what the product's texture board holds.
#[test]
fn the_texture_store_is_the_products_size() {
    use crate::dev::kona::ident::Product;
    for (product, mb) in [(Product::Ir2, 64), (Product::Ir2e, 64), (Product::Ir3, 256)] {
        let be = KonaBackend::new(KonaConfig::new(product, 1).unwrap());
        assert_eq!(be.store.texture().bank_units(), mb << 14, "{product:?}: units of 256 bits in one of two banks");
    }
}

#[test]
fn status_and_counters_answer_and_other_words_do_not() {
    let (_host, kona) = board();
    let (known, text) = monitor(&kona, &["status"]);
    assert!(known);
    assert!(text.contains("raster managers   2") && text.contains("texture per RM    64 MiB"), "{text}");
    assert!(text.contains("record bound      32 words per pixel"), "{text}");
    let (known, text) = monitor(&kona, &["counters"]);
    assert!(known && text.contains("GL commands       0"), "{text}");
    assert!(!monitor(&kona, &["frob", "x.png"]).0);
}

/// The monitor reaches the back end through the board's own command.
#[test]
fn the_board_routes_the_kona_command() {
    use crate::traits::Device;
    let (_host, kona) = board();
    assert_eq!(kona.register_commands()[0].0, "kona");
    let out = Capture::default();
    kona.execute_command("kona", &["status"], Box::new(out.clone())).unwrap();
    assert!(out.text().contains("managed area      1280 x 1024"));
    assert!(kona.execute_command("kona", &["frob"], Box::new(out)).is_err());
}

/// Jobs are counted by kind, a retrace completes at once and the managed
/// area is reported when the firmware starts.
#[test]
fn jobs_are_counted_and_retraces_complete() {
    let (_host, kona) = board();
    let w = |off: u64, v: u32| assert_eq!(kona.test_write(off, 4, v as u64), crate::traits::BUS_OK);
    w(0xc10000, 1);
    w(0xc10000, 2);
    w(0xc10000, 7);
    w(0xc10000, 0x1f1);
    w(0xc10000, 0x1000_0000);
    w(0xc10000, 0);
    w(0xc10000, 0x4000);
    w(0xc10000, 0x1e7);
    w(0xc10000, 0x42);
    let (_, text) = monitor(&kona, &["counters"]);
    assert!(text.contains("textport commands 1"), "{text}");
    assert!(text.contains("kernel commands   2"), "{text}");
    assert!(text.contains("retrace requests  1"), "{text}");
    assert_eq!(kona.bus_read(0x800248, 4), (crate::traits::BUS_OK, 0x42));
}

/// A word that opens no command is counted, not answered.
#[test]
fn dropped_words_are_counted() {
    let (_host, kona) = board();
    for w in [1, 2, 7, 0x500, 0x500, 0x7f7f] {
        assert_eq!(kona.test_write(0xc10000, 4, w), crate::traits::BUS_OK);
    }
    let (_, text) = monitor(&kona, &["counters"]);
    assert!(text.contains("dropped words     3"), "{text}");
}

/// Wait for `cond`, up to a deadline, not a count of tries.
fn wait_for(mut cond: impl FnMut() -> bool) -> bool {
    let end = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while std::time::Instant::now() < end {
        if cond() {
            return true;
        }
        std::thread::yield_now();
    }
    cond()
}

fn backend() -> KonaBackend {
    KonaBackend::new(KonaConfig::default())
}

fn run(b: &mut KonaBackend, args: &[&str]) -> (bool, String) {
    let mut out = Vec::new();
    let known = b.monitor(args, &mut out).unwrap();
    (known, String::from_utf8(out).unwrap())
}

/// A GL buffer configuration is laid out within the budget; one the hardware
/// documents do not cover is refused and counted; so is one of the wrong
/// length; the monitor shows them.
#[test]
fn a_buffer_configuration_gets_a_layout_or_a_counted_refusal() {
    let mut b = backend();
    assert!(run(&mut b, &["status"]).1.contains("layouts           0 made"));
    b.buffer_config(&[0, 10, 0, 1, 0, 4, 24, 8, 25, 0]);
    b.buffer_config(&[0, 8, 0, 0, 0, 2, 0, 0, 0, 0]);
    b.buffer_config(&[0, 8, 0]);
    b.buffer_config(&[0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    let text = run(&mut b, &["status"]).1;
    assert!(text.contains("layouts           1 made"), "{text}");
    assert!(text.contains("refused         2 x not 10 words"), "{text}");
    assert!(text.contains("refused         1 x Samples"), "{text}");
    assert!(text.contains("(13 words)"), "{text}");
}

/// Run one job through `Backend::job` with a front end behind it.
fn send(b: &mut KonaBackend, job: Job<'_>) -> Reply {
    let (host, mut core) = crate::dev::kona::tests::bare(&TestHost::new(), test_cfg());
    b.job(&mut BackPort { core: &mut core, host: &host }, job)
}

fn command(b: &mut KonaBackend, grammar: Grammar, selector: u32, args: &[u32]) -> Reply {
    send(b, Job::Command { grammar, selector, args, packed: 0, handled: false })
}

/// A framed GL command reaches the layout through `Backend::job`; the
/// same selector in the textport grammar does not.
#[test]
fn a_framed_gl_buffer_configuration_is_laid_out() {
    let mut b = backend();
    let args = [0, 10, 0, 1, 0, 4, 24, 8, 25, 0];
    command(&mut b, Grammar::Textport, 1, &args);
    command(&mut b, Grammar::Gl, 1, &args);
    command(&mut b, Grammar::Gl, 1, &args[..3]);
    let text = run(&mut b, &["status"]).1;
    assert!(text.contains("layouts           1 made") && text.contains("1 x not 10 words"), "{text}");
    assert!(text.contains("(13 words)"), "{text}");
}

/// A job raises the store's flag for the display once it has run, and a
/// retrace, which changes no pixel, does not.
#[test]
fn a_job_touches_the_store_unless_it_is_a_retrace() {
    let mut b = backend();
    assert!(b.store.take_touched(), "a new store is shown once");
    send(&mut b, Job::Retrace { tag: 1 });
    assert!(!b.store.take_touched());
    command(&mut b, Grammar::Textport, 2, &[7]);
    assert!(b.store.take_touched());
    b.reset();
    assert!(b.store.take_touched(), "a reset is shown");
}

/// Textport commands draw into the colour word of the store, a bitmap
/// draws at the raster position, and the hash follows the contents.
#[test]
fn textport_commands_draw_into_the_store() {
    let mut b = backend();
    let empty = run(&mut b, &["hash"]).1;
    command(&mut b, Grammar::Textport, 2, &[7]);
    command(&mut b, Grammar::Textport, 5, &[0, 0, 3, 3]);
    command(&mut b, Grammar::Textport, 4, &[10, 10]);
    send(&mut b, Job::Bitmap { header: &[1, 1, 0, 0, 2, 0], payload: &[1 << 31] });
    assert_eq!(b.store.read(10, 10, 1), 7);
    assert_eq!(b.store.read(3, 3, 1), 7);
    assert_eq!(b.store.read(1, 1, 1), 7);
    assert_eq!(b.store.read(0, 0, 1), 0, "the near corner is out");
    let text = run(&mut b, &["status"]).1;
    assert!(text.contains("colour 7 raster (12, 10), 2 drawn"), "{text}");
    let drawn = run(&mut b, &["hash"]).1;
    assert_ne!(empty, drawn);
    b.reset();
    assert_ne!(run(&mut b, &["hash"]).1, drawn);
    assert_eq!(b.store.read(3, 3, 1), 0);
}

/// X state commands load the context, position, clip and scissor; the
/// monitor shows what was understood.
#[test]
fn x_state_commands_are_taken_and_shown() {
    let mut b = backend();
    let gc = [0x5a000, 0x1ffe, 0xa5000, 0x1ffe, 0xf00fff, 0, 0x180c, 0x40001, 1, 6, 5, 1, 1];
    command(&mut b, Grammar::Kernel, WINDOW_SYSTEM_GRAPHICS_CONTEXT_LOAD, &gc);
    command(&mut b, Grammar::Kernel, WINDOW_SYSTEM_DRAWABLE_ORIGIN_AND_SIZE_LOAD, &[240, 295, 100, 50]);
    command(&mut b, Grammar::Kernel, WINDOW_SYSTEM_CLIP_LOAD, &[1, 10, 20, 110, 220]);
    command(&mut b, Grammar::Kernel, WINDOW_SYSTEM_SCISSOR_LOAD, &[10, 20, 100, 200]);
    let text = run(&mut b, &["status"]).1;
    assert!(text.contains("MainIndex16 fg 0x5a00 bg 0xa500 mask 0xffff gx 3"), "{text}");
    assert!(text.contains("origin (240, 295), 1 clip rects (0 refused), scissor Some([10, 20, 100, 200])"), "{text}");
    // The same selector numbers in another grammar are not X state.
    command(&mut b, Grammar::Gl, WINDOW_SYSTEM_GRAPHICS_CONTEXT_LOAD, &[0; 13]);
    command(&mut b, Grammar::Kernel, PIPE_SET_UP, &[0, 0, 0]);
    assert!(run(&mut b, &["status"]).1.contains("MainIndex16"));
    b.reset();
    assert!(run(&mut b, &["status"]).1.contains("not decoded"));
}

/// An X box through `Backend::job` draws into the colour word of the store
/// under the loaded context, and a command that is no X token leaves a
/// waiting run alone only if it belongs to the run.
#[test]
fn an_x_box_draws_into_the_store() {
    let mut b = backend();
    let gc = [0x5a000, 0x1ffe, 0xa5000, 0x1ffe, 0xf00fff, 0, 0x180c, 0x40001, 1, 6, 5, 1, 1];
    command(&mut b, Grammar::Kernel, WINDOW_SYSTEM_GRAPHICS_CONTEXT_LOAD, &gc);
    command(&mut b, Grammar::Kernel, FILL_BOXES_TWO_CORNERS, &[2 << 16 | 3, 5 << 16 | 7]);
    assert_eq!(b.store.read(2, 1023 - 3, 1), 0x5a00);
    assert_eq!(b.store.read(5, 1023 - 3, 1), 0);
    // A data token, another command, then a box: the run did not survive.
    command(&mut b, Grammar::Kernel, WINDOW_SYSTEM_2_WORD_DATA_PRELOAD, &[10 << 16 | 10, 12 << 16 | 12]);
    command(&mut b, Grammar::Kernel, PIPE_SET_UP, &[0, 0, 0]);
    command(&mut b, Grammar::Kernel, FILL_BOXES_TWO_CORNERS, &[20 << 16 | 20, 21 << 16 | 21]);
    assert_eq!(b.store.read(10, 1023 - 10, 1), 0);
    assert_eq!(b.store.read(20, 1023 - 20, 1), 0x5a00);
    let text = run(&mut b, &["status"]).1;
    assert!(text.contains("X fills           2 drawn, refused: 0 context, 0 target, 0 destination"), "{text}");
}

/// A stipple store asks for its rows as a data run; with the pattern
/// stored and loaded, a box fill is masked by it.
#[test]
fn a_stipple_is_stored_loaded_and_masks_a_fill() {
    let mut b = backend();
    let gc = [0x5a000, 0x1ffe, 0xa5000, 0x1ffe, 0xf00fff, 0, 0x180c, 0x40001, 1, 6, 5, 1, 1];
    command(&mut b, Grammar::Kernel, WINDOW_SYSTEM_GRAPHICS_CONTEXT_LOAD, &gc);
    // A 2 x 2 checkerboard in entry 4: the reply asks for its two rows.
    assert_eq!(command(&mut b, Grammar::Kernel, STORE_WINDOW_SYSTEM_STIPPLE, &[4, 2, 2]), Reply::Data(2));
    send(&mut b, Job::Data { grammar: Grammar::Kernel, selector: STORE_WINDOW_SYSTEM_STIPPLE, args: &[4, 2, 2], words: &[0x8000_0000, 0x4000_0000] });
    assert_eq!(command(&mut b, Grammar::Kernel, LOAD_WINDOW_SYSTEM_STIPPLE, &[4]), Reply::Done);
    command(&mut b, Grammar::Kernel, FILL_BOXES_TWO_CORNERS, &[0, 4 << 16 | 4]);
    assert_eq!(b.store.read(0, 1023 - 0, 1), 0x5a00);
    assert_eq!(b.store.read(1, 1023 - 0, 1), 0);
    assert_eq!(b.store.read(1, 1023 - 1, 1), 0x5a00);
    assert!(run(&mut b, &["status"]).1.contains("X stipple         loaded"));
    command(&mut b, Grammar::Kernel, LOAD_WINDOW_SYSTEM_STIPPLE, &[u32::MAX]);
    command(&mut b, Grammar::Kernel, FILL_BOXES_TWO_CORNERS, &[10 << 16 | 10, 12 << 16 | 12]);
    assert_eq!(b.store.read(11, 1023 - 10, 1), 0x5a00, "no stipple: solid");
    // A store of no rows asks for none.
    assert_eq!(command(&mut b, Grammar::Kernel, STORE_WINDOW_SYSTEM_STIPPLE, &[4, 2, 0]), Reply::Done);
}

/// With the board running, a retrace request waits for the display thread's
/// next frame boundary; the tag register then reads its tag and `status` shows
/// the thread and its frames.
#[test]
fn a_retrace_completes_at_a_frame_boundary_on_a_running_board() {
    use crate::traits::Device;
    let (_host, kona) = board();
    kona.start();
    for w in [0x1e7u32, 0x77] {
        assert_eq!(kona.bus_write(0xc10000, 4, w as u64), crate::traits::BUS_OK);
    }
    wait_for(|| kona.bus_read(0x800248, 4).1 == 0x77);
    assert_eq!(kona.bus_read(0x800248, 4).1, 0x77);
    // The counter follows the completion within the same frame.
    wait_for(|| monitor(&kona, &["status"]).1.contains("1 retraces completed"));
    let text = monitor(&kona, &["status"]).1;
    assert!(text.contains("thread running"), "{text}");
    kona.stop();
    assert!(monitor(&kona, &["status"]).1.contains("thread stopped"));
}

/// What the pixel tests send jobs through: a front end with a context
/// current and a return and a reason page registered, over guest RAM the
/// test can look at.
struct Rig {
    ram: Arc<TestHost>,
    host: super::super::front::Host,
    core: super::super::pipe::Core,
}

const RETURN_PAGE: u64 = 0x30_0000;
const REASON_PAGE: u64 = 0x31_0000;

impl Rig {
    fn new() -> Rig {
        let ram = TestHost::new();
        let (host, mut core) = crate::dev::kona::tests::bare(&ram, test_cfg());
        core.current = Some(0);
        core.return_page = Some(RETURN_PAGE);
        core.reason_page = Some(REASON_PAGE);
        Rig { ram, host, core }
    }

    fn send(&mut self, b: &mut KonaBackend, job: Job<'_>) -> Reply {
        b.job(&mut BackPort { core: &mut self.core, host: &self.host }, job)
    }

    fn x(&mut self, b: &mut KonaBackend, selector: u32, args: &[u32]) -> Reply {
        self.send(b, Job::Command { grammar: Grammar::Kernel, selector, args, packed: 0, handled: false })
    }
}

const INDEX_GC: [u32; 13] = [0x5a000, 0x1ffe, 0xa5000, 0x1ffe, 0xf00fff, 0, 0x180c, 0x40001, 1, 6, 5, 1, 1];

/// A tile worker that panics (here: a job that names a plane set-up never
/// allocated) poisons the scheduler, and the board reports itself dead.
#[test]
fn a_poisoned_tile_scheduler_makes_the_board_dead() {
    let kona = Kona::new(test_cfg().with_pool(2, 32), TestHost::new(), TestHost::new(), crate::dev::kona::tests::test_status(), None).unwrap();
    assert_eq!(kona.failure(), None);
    kona.with_backend(|b| {
        let full = Rect { x0: 0, y0: 0, x1: 64, y1: 64 };
        let state = b.tiles().state(raster::DrawState { clip: full, colour: [77, 0], colours: 1, ..Default::default() });
        let req = raster::ClearReq { rect: full, colour: Some(1), ..Default::default() };
        let _ = b.tiles().submit(RasterJob::Clear(ClearJob { req, state }));
        b.drain();
    });
    let why = kona.failure().expect("poisoned");
    assert!(why.contains("tile worker"), "{why}");
}

/// The placement's word 9 and the enable in word 10 become the draw state's
/// ownership test: off with the enable clear; encoded as the board does when set;
/// a word in none of the ranges is encoded as the ordinary pair and counted.
#[test]
fn a_placement_sets_the_ownership_test_from_words_nine_and_ten() {
    let (mut b, mut r) = (backend(), Rig::new());
    let screen = Rect { x0: 0, y0: 0, x1: 1280, y1: 1024 };
    let own = |b: &mut KonaBackend| b.contexts.select(Some(0)).gl.draw_state(screen).ownership;
    r.x(&mut b, WINDOW_CLIP_SETUP, &[0, 0, 64, 64, 0, 0, 64, 64, 0, 0xe, 0]);
    assert_eq!(own(&mut b), None, "the enable is clear");
    r.x(&mut b, WINDOW_CLIP_SETUP, &[0, 0, 64, 64, 0, 0, 64, 64, 0, 0xe, 1]);
    assert_eq!(own(&mut b).map(|o| o.register()), Some(0x7e01d));
    r.x(&mut b, WINDOW_CLIP_SETUP, &[0, 0, 64, 64, 0, 0, 64, 64, 0, 0xffff_ffff, 1]);
    assert_eq!(own(&mut b), Some(raster::ops::Ownership { reference: 0x3f, mask: 0x3f }), "all ones is the ordinary pair 0x3f, 0x3f");
    assert_eq!(b.gl.counts.odd_window_ids, 0);
    r.x(&mut b, WINDOW_CLIP_SETUP, &[0, 0, 64, 64, 0, 0, 64, 64, 0, 0x77, 1]);
    assert_eq!(b.gl.counts.odd_window_ids, 1);
}

/// The window placement keeps the window's origin and its clip rectangle
/// apart: the recording's tenth placement has a clip of the whole screen
/// and a window origin of 314, and a later move carries both.
#[test]
fn a_placement_keeps_the_window_origin_and_the_clip_rectangle_apart() {
    let (mut b, mut r) = (backend(), Rig::new());
    r.x(&mut b, WINDOW_CLIP_SETUP, &[314, 0, 299, 1, 0, 0, 1280, 1024, 298, 0, 0]);
    let ds = b.contexts.select(Some(0)).gl.draw_state(Rect { x0: 0, y0: 0, x1: 1280, y1: 1024 });
    assert_eq!((ds.origin, ds.clip), ([314, 0], Rect { x0: 0, y0: 0, x1: 1280, y1: 1024 }));
    // The placement before it in the recording has clip equal to window.
    r.x(&mut b, WINDOW_CLIP_SETUP, &[100, 50, 200, 100, 100, 50, 200, 100, 0, 0, 0]);
    let ds = b.contexts.select(Some(0)).gl.draw_state(Rect { x0: 0, y0: 0, x1: 1280, y1: 1024 });
    assert_eq!((ds.origin, ds.clip), ([100, 50], Rect { x0: 100, y0: 50, x1: 300, y1: 150 }));
    // Words 0 and 1 are halfwords: 0xfff0 is -16, a window partly off the left of the screen.
    r.x(&mut b, WINDOW_CLIP_SETUP, &[0xfff0, 0, 64, 64, 0xfff0, 0, 64, 64, 0, 0, 0]);
    let ds = b.contexts.select(Some(0)).gl.draw_state(Rect { x0: 0, y0: 0, x1: 1280, y1: 1024 });
    assert_eq!((ds.origin, ds.clip), ([-16, 0], Rect { x0: 0, y0: 0, x1: 48, y1: 64 }));
    r.x(&mut b, WINDOW_READ_CLIP_SET_UP, &[16, 10, 0]);
    let ds = b.contexts.select(Some(0)).gl.draw_state(Rect { x0: 0, y0: 0, x1: 1280, y1: 1024 });
    assert_eq!((ds.origin, ds.clip), ([16, 10], Rect { x0: 16, y0: 10, x1: 80, y1: 74 }), "the clip moves with the origin");
}

/// An image whose rows follow the command as a data run draws once the
/// run is in, the store's top row first.
#[test]
fn an_inline_pixel_write_asks_for_its_rows_and_draws_them() {
    let (mut b, mut r) = (backend(), Rig::new());
    r.x(&mut b, WINDOW_SYSTEM_GRAPHICS_CONTEXT_LOAD, &INDEX_GC);
    r.x(&mut b, WINDOW_SYSTEM_PIXEL_PATH_SET_UP, &[0, 0, 1, 8, 3, 2, 0]);
    assert_eq!(r.x(&mut b, WINDOW_SYSTEM_PIXEL_WRITE, &[4, 2]), Reply::Data(2));
    r.send(&mut b, Job::Data { grammar: Grammar::Kernel, selector: WINDOW_SYSTEM_PIXEL_WRITE, args: &[4, 2], words: &[0x0102_0300, 0x0405_0600] });
    assert_eq!([b.store.read(4, 1021, 1), b.store.read(6, 1021, 1), b.store.read(5, 1020, 1)], [0x100, 0x300, 0x500]);
    assert_eq!(r.x(&mut b, WINDOW_SYSTEM_PIXEL_PATH_FINISH, &[0]), Reply::Done);
    assert!(run(&mut b, &["status"]).1.contains("1 writes (0 by DMA), 0 copies, 0 reads, 1 finishes"));
}

/// With the DMA bit set the command asks for nothing and the DMA packet
/// that arrives next carries the rows, read from the guest a line at a
/// time; one that arrives with no write waiting is unclaimed, and a read
/// that fails faults the packet and draws nothing.
#[test]
fn a_dma_pixel_write_takes_the_packet_that_follows() {
    let (mut b, mut r) = (backend(), Rig::new());
    r.x(&mut b, WINDOW_SYSTEM_GRAPHICS_CONTEXT_LOAD, &INDEX_GC);
    r.x(&mut b, WINDOW_SYSTEM_PIXEL_PATH_SET_UP, &[0x10, 0, 1, 8, 4, 2, 0]);
    assert_eq!(r.x(&mut b, WINDOW_SYSTEM_PIXEL_WRITE, &[0, 0]), Reply::Done);
    // Two rows of one word each, the lines 8 bytes apart.
    r.ram.put_words(0x40_0000, &[0x0102_0304, 0xdead_beef, 0x0506_0708]);
    let dma = |io, lines, per_line, pitch| Job::DmaIn { io, physical: true, lines, per_line, pitch };
    assert_eq!(r.send(&mut b, dma(0x40_0000, 2, 4, 8)), Reply::Done);
    assert_eq!([b.store.read(0, 1023, 1), b.store.read(3, 1023, 1), b.store.read(3, 1022, 1)], [0x100, 0x400, 0x800]);
    assert_eq!(r.send(&mut b, dma(0x40_0000, 1, 4, 4)), Reply::Unclaimed, "nothing waits for it");
    assert!(!r.core.dma_fault);
    r.x(&mut b, WINDOW_SYSTEM_PIXEL_PATH_SET_UP, &[0x10, 0, 1, 8, 4, 1, 0]);
    r.x(&mut b, WINDOW_SYSTEM_PIXEL_WRITE, &[0, 4]);
    assert_eq!(r.send(&mut b, dma(0x7f_ff_ff_00, 1, 4, 4)), Reply::Done);
    assert!(r.core.dma_fault, "the bus read failed");
    assert_eq!(b.store.read(0, 1019, 1), 0);
    let text = run(&mut b, &["status"]).1;
    assert!(text.contains("1 writes (1 by DMA)") && text.contains("1 with no write waiting"), "{text}");
}

/// A read answers on the return page after its first word, which is set
/// last; with no context current it is not answered.
#[test]
fn a_pixel_read_answers_on_the_return_page() {
    let (mut b, mut r) = (backend(), Rig::new());
    r.x(&mut b, WINDOW_SYSTEM_GRAPHICS_CONTEXT_LOAD, &INDEX_GC);
    r.x(&mut b, FILL_BOXES_TWO_CORNERS, &[2 << 16, 4 << 16 | 1]);
    assert_eq!(b.store.read(3, 1023, 1), 0x5a00);
    r.x(&mut b, WINDOW_SYSTEM_PIXEL_PATH_SET_UP, &[2, 1, 0, 8, 3, 1, 0]);
    r.x(&mut b, WINDOW_SYSTEM_PIXEL_READ, &[2, 0]);
    assert_eq!(r.ram.word(RETURN_PAGE), 1);
    assert_eq!(r.ram.bytes(RETURN_PAGE + 8, 4), [0x5a, 0x5a, 0, 0], "pixels 2 and 3 are filled, 4 is not");
    r.ram.put(RETURN_PAGE, &[0; 16]);
    r.core.current = None;
    r.x(&mut b, WINDOW_SYSTEM_PIXEL_READ, &[2, 0]);
    assert_eq!(r.ram.word(RETURN_PAGE), 0, "no context, no answer");
}

/// A copy under the context moves pixels the same way a write puts them.
#[test]
fn a_pixel_copy_moves_pixels_inside_the_store() {
    let (mut b, mut r) = (backend(), Rig::new());
    r.x(&mut b, WINDOW_SYSTEM_GRAPHICS_CONTEXT_LOAD, &INDEX_GC);
    r.x(&mut b, FILL_BOXES_TWO_CORNERS, &[0, 2 << 16 | 1]);
    r.x(&mut b, WINDOW_SYSTEM_PIXEL_PATH_SET_UP, &[1, 1, 1, 0, 2, 1, 0]);
    r.x(&mut b, WINDOW_SYSTEM_PIXEL_COPY, &[0, 0, 10, 5]);
    assert_eq!([b.store.read(10, 1018, 1), b.store.read(11, 1018, 1), b.store.read(12, 1018, 1)], [0x5a00, 0x5a00, 0]);
}

/// Each context has its own X state, and a fresh activation starts over.
#[test]
fn the_x_state_follows_the_context() {
    let (mut b, mut r) = (backend(), Rig::new());
    r.x(&mut b, WINDOW_SYSTEM_GRAPHICS_CONTEXT_LOAD, &INDEX_GC);
    r.core.current = Some(1);
    r.send(&mut b, Job::Context(super::super::ContextEvent::Activated { slot: 1, fresh: true }));
    let text = run(&mut b, &["status"]).1;
    assert!(text.contains("X contexts        1 loaded"), "{text}");
    assert!(text.contains("X gc              not decoded"), "{text}");
    r.x(&mut b, FILL_BOXES_TWO_CORNERS, &[0, 2 << 16 | 2]);
    assert_eq!(b.store.read(0, 1023, 1), 0, "context 1 loaded no colour");
    r.core.current = Some(0);
    r.x(&mut b, FILL_BOXES_TWO_CORNERS, &[0, 2 << 16 | 2]);
    assert_eq!(b.store.read(1, 1022, 1), 0x5a00, "context 0 kept its own");
}

/// A 64 x 64 level loaded at address 0x400 of bank 0: texel (x, y) is the grey `(4 x + y) mod 256`.
fn load_grey_level(b: &mut KonaBackend, r: &mut Rig) {
    let mut header = [0u32; 20];
    header[..5].copy_from_slice(&[0x0de1, 64, 64, 5, 1]);
    header[9] = 0x30;
    header[13] = 63;
    header[14] = 63;
    header[16] = 0x66;
    header[17] = 0x2200;
    header[18] = 0x400;
    let words: Vec<u32> = (0..64 * 64).map(|i| { let g = ((4 * (i % 64) + i / 64) % 256) as u32; g << 24 | g << 16 | g << 8 | 0xff }).collect();
    assert_eq!(r.send(b, Job::Command { grammar: Grammar::Gl, selector: DRAW_TEXELS_INTO_TEXTURE_MEMORY, args: &header, packed: 0, handled: false }), Reply::Data(4096));
    r.send(b, Job::Data { grammar: Grammar::Gl, selector: DRAW_TEXELS_INTO_TEXTURE_MEMORY, args: &header, words: &words });
}

/// The 18 words of a read-back of that level as luminance in unsigned shorts, which Quake sends: the target,
/// the size, format 6, type 3, storage 0xd, a spare, the first and last texel, then the load record's words.
const GREY_READ: [u32; 18] = [0x0de1, 64, 64, 6, 3, 0xd, 0, 0, 0, 0, 63, 63, 0, 0x0385_0030, 0x66, 0x44, 0x400, 0];

/// A bare texel read-back (token 0x052) answers on the return page: the level's texels in the requested
/// format and type at offset 8, then a non-zero word 0. A level that is not loaded, or a format the packer does
/// not know, answers zeros all the same, counted; with no context there is nothing to answer on.
#[test]
fn a_bare_texel_read_back_answers_on_the_return_page() {
    let (mut b, mut r) = (backend(), Rig::new());
    load_grey_level(&mut b, &mut r);
    r.send(&mut b, Job::Command { grammar: Grammar::Gl, selector: QUERY_TEXTURE_IMAGE, args: &GREY_READ, packed: 0, handled: false });
    assert_eq!(r.ram.word(RETURN_PAGE), 1, "the completion word is set");
    let image = r.ram.bytes(RETURN_PAGE + 8, 64 * 64 * 2);
    let texel = |x: usize, y: usize| u16::from_be_bytes([image[2 * (64 * y + x)], image[2 * (64 * y + x) + 1]]);
    assert_eq!([texel(0, 0), texel(1, 0), texel(0, 1), texel(63, 63)], [0, 4 * 257, 257, (((4 * 63 + 63) % 256) * 257) as u16]);
    // The same read of a sub-rectangle: first (2, 1), 2 x 2.
    let mut part = GREY_READ;
    part[1..3].copy_from_slice(&[2, 2]);
    part[7..9].copy_from_slice(&[2, 1]);
    part[10..12].copy_from_slice(&[3, 2]);
    r.ram.put(RETURN_PAGE, &[0; 16]);
    r.send(&mut b, Job::Command { grammar: Grammar::Gl, selector: QUERY_TEXTURE_IMAGE, args: &part, packed: 0, handled: false });
    assert_eq!(r.ram.word(RETURN_PAGE), 1);
    assert_eq!(r.ram.bytes(RETURN_PAGE + 8, 8), [(9 * 257u16).to_be_bytes(), (13 * 257u16).to_be_bytes(), (10 * 257u16).to_be_bytes(), (14 * 257u16).to_be_bytes()].concat());
    // No level there: zeros, counted. An unknown format: nothing to pack, still answered.
    let mut missing = GREY_READ;
    missing[16] = 0x800;
    r.ram.put(RETURN_PAGE, &[0xff; 16]);
    r.send(&mut b, Job::Command { grammar: Grammar::Gl, selector: QUERY_TEXTURE_IMAGE, args: &missing, packed: 0, handled: false });
    assert_eq!((r.ram.word(RETURN_PAGE), r.ram.bytes(RETURN_PAGE + 8, 4)), (1, vec![0; 4]));
    let mut unknown = GREY_READ;
    unknown[3] = 0x77;
    r.ram.put(RETURN_PAGE, &[0xff; 0x4000]);
    r.send(&mut b, Job::Command { grammar: Grammar::Gl, selector: QUERY_TEXTURE_IMAGE, args: &unknown, packed: 0, handled: false });
    assert_eq!(r.ram.word(RETURN_PAGE), 1, "never left unposted");
    assert_eq!(r.ram.bytes(RETURN_PAGE + 8, 0x3ff0), vec![0; 0x3ff0], "the reply area is zeros, not what the page held");
    r.core.current = None;
    r.ram.put(RETURN_PAGE, &[0; 16]);
    r.send(&mut b, Job::Command { grammar: Grammar::Gl, selector: QUERY_TEXTURE_IMAGE, args: &GREY_READ, packed: 0, handled: false });
    assert_eq!(r.ram.word(RETURN_PAGE), 0, "no context, no answer");
    let text = run(&mut b, &["gl"]).1;
    assert!(text.contains("texel read-backs  5 (0 in a bracket); answered with zeros: 1 for a format or type the packer does not know, 2 for a level not loaded; 0 too big; 1 with nowhere to answer"), "{text}");
}

/// A texel level of 8 x 4 at `address` of bank `bank`, each texel the word `texel(x, y)` (red, green, blue, alpha).
fn load_level_words(b: &mut KonaBackend, r: &mut Rig, address: u32, bank: u32, texel: impl Fn(u32, u32) -> u32) {
    let mut header = [0u32; 20];
    header[..5].copy_from_slice(&[0x0de1, 8, 4, 5, 1]);
    header[9] = 0x30;
    header[13] = 7;
    header[14] = 3;
    header[16] = 0x23;
    header[17] = 0x2200 | bank << 3;
    header[18 + bank as usize] = address;
    let words: Vec<u32> = (0..32).map(|i| texel(i % 8, i / 8)).collect();
    r.send(b, Job::Command { grammar: Grammar::Gl, selector: DRAW_TEXELS_INTO_TEXTURE_MEMORY, args: &header, packed: 0, handled: false });
    r.send(b, Job::Data { grammar: Grammar::Gl, selector: DRAW_TEXELS_INTO_TEXTURE_MEMORY, args: &header, words: &words });
}

/// The 18 words of a read-back of an 8 x 4 level of `bank` at `address`: `format`, `ty`, storage `storage`.
fn level_read(format: u32, ty: u32, storage: u32, bank: u32, address: u32) -> [u32; 18] {
    let mut a = [0x0de1, 8, 4, format, ty, storage, 0, 0, 0, 0, 7, 3, 0, 0x0385_0030 | bank << 28, 0x23, 0x44, address, 0];
    a[13] = 0x0385_0030 | bank << 28;
    a
}

fn read_texels(b: &mut KonaBackend, r: &mut Rig, args: &[u32]) {
    r.send(b, Job::Command { grammar: Grammar::Gl, selector: QUERY_TEXTURE_IMAGE, args, packed: 0, handled: false });
}

/// A level is found by the bank in the read's mode word and the address in word 16. The empty slot of a
/// level in the other bank is address zero, which names a real level in this one, and must not be taken.
#[test]
fn a_texel_read_finds_its_level_by_bank_and_address() {
    let (mut b, mut r) = (backend(), Rig::new());
    load_level_words(&mut b, &mut r, 0, 1, |_, _| 0x1000_00ff);
    load_level_words(&mut b, &mut r, 0x400, 0, |_, _| 0xf000_00ff);
    // The read is of bank 0 at 0x400 (its other slot, word 17, is the empty zero).
    r.ram.put(RETURN_PAGE, &[0; 16]);
    read_texels(&mut b, &mut r, &level_read(6, 1, 0x0d, 0, 0x400));
    assert_eq!(r.ram.bytes(RETURN_PAGE + 8, 2), [0xf0, 0xf0], "bank 0 at 0x400 is the second level");
    // Bank 0 at address 0 is empty, though bank 1 holds a level at address 0.
    r.ram.put(RETURN_PAGE, &[0xff; 16]);
    read_texels(&mut b, &mut r, &level_read(6, 1, 0x0d, 0, 0));
    assert_eq!(r.ram.bytes(RETURN_PAGE + 8, 2), [0, 0], "no level in bank 0 at address 0");
    // And bank 1 at address 0 is the first.
    read_texels(&mut b, &mut r, &level_read(6, 1, 0x0d, 1, 0));
    assert_eq!(r.ram.bytes(RETURN_PAGE + 8, 2), [0x10, 0x10]);
    let text = run(&mut b, &["gl"]).1;
    assert!(text.contains("1 for a level not loaded"), "{text}");
}

/// A read of a colour texture as luminance is the sum of red, green and blue clamped at 1.0, with the
/// alpha kept; a texture that holds one channel is read as it is.
#[test]
fn a_texel_read_as_luminance_sums_the_colour_of_a_colour_texture() {
    let (mut b, mut r) = (backend(), Rig::new());
    // Texel (0, 0): red 0x10, green 0x20, blue 0x30, alpha 0x40. Texel (1, 0): 0x80 in all three, alpha 0xff.
    load_level_words(&mut b, &mut r, 0x100, 0, |x, y| match (x, y) {
        (0, 0) => 0x1020_3040,
        (1, 0) => 0x8080_80ff,
        _ => 0,
    });
    let mut luminance = level_read(6, 1, 0x30, 0, 0x100);
    luminance[1..3].copy_from_slice(&[2, 1]);
    read_texels(&mut b, &mut r, &luminance);
    assert_eq!(r.ram.bytes(RETURN_PAGE + 8, 2), [0x10 + 0x20 + 0x30, 0xff], "0x60 and a clamped sum");
    let mut both = level_read(7, 1, 0x30, 0, 0x100);
    both[1..3].copy_from_slice(&[2, 1]);
    read_texels(&mut b, &mut r, &both);
    assert_eq!(r.ram.bytes(RETURN_PAGE + 8, 4), [0x60, 0x40, 0xff, 0xff], "luminance then the alpha the texel holds");
    // The same level read as the one-channel texture it would be is its red alone.
    let mut one = level_read(6, 1, 0x0d, 0, 0x100);
    one[1..3].copy_from_slice(&[2, 1]);
    read_texels(&mut b, &mut r, &one);
    assert_eq!(r.ram.bytes(RETURN_PAGE + 8, 2), [0x10, 0x80]);
}

/// A push-attributes token is answered whatever its mask: the pipe's handler opens and closes the
/// read-back on every path, so the completion word is set last and the count word is zero. A
/// push onto a full stack is answered as well (and sets the overflow error); a pop and a bare
/// light-state step (0x10b) post nothing.
#[test]
fn a_push_of_attributes_posts_its_completion_on_every_path() {
    let (mut b, mut r) = (backend(), Rig::new());
    let gl_token = |b: &mut KonaBackend, r: &mut Rig, selector: u32, args: &[u32]| {
        r.send(b, Job::Command { grammar: Grammar::Gl, selector, args, packed: 0, handled: false });
    };
    let words = |r: &Rig| (r.ram.word(RETURN_PAGE), r.ram.bytes(RETURN_PAGE + 4, 4));
    // The list bit alone: nothing is read back, and the client still waits.
    gl_token(&mut b, &mut r, PUSH_ATTRIBUTE_STACK, &[0x20000]);
    assert_eq!(words(&r), (1, vec![0; 4]));
    // A mask with a bit the pipe reads back is answered too, with no data, and counted.
    r.ram.put(RETURN_PAGE, &[0; 16]);
    gl_token(&mut b, &mut r, PUSH_ATTRIBUTE_STACK, &[0x8]);
    assert_eq!(words(&r), (1, vec![0; 4]));
    assert_eq!((b.query_counts.pushes, b.query_counts.pushes_unread), (2, 1));
    // Fill the stack to 16 entries; the push onto the full stack still completes.
    for _ in 0..14 {
        gl_token(&mut b, &mut r, PUSH_ATTRIBUTE_STACK, &[0x20000]);
    }
    assert_eq!(b.contexts.select(Some(0)).gl.attrib_depth, 16);
    assert_eq!(b.contexts.select(Some(0)).gl.error, 0);
    r.ram.put(RETURN_PAGE, &[0; 16]);
    gl_token(&mut b, &mut r, PUSH_ATTRIBUTE_STACK, &[0x20000]);
    assert_eq!(r.ram.word(RETURN_PAGE), 1, "a push onto a full stack completes");
    assert_eq!((b.contexts.select(Some(0)).gl.attrib_depth, b.contexts.select(Some(0)).gl.error, b.query_counts.pushes_full), (16, 0x503, 1));
    // A pop posts nothing and frees an entry.
    r.ram.put(RETURN_PAGE, &[0; 16]);
    gl_token(&mut b, &mut r, POP_ATTRIBUTE_STACK, &[0]);
    assert_eq!(r.ram.word(RETURN_PAGE), 0);
    assert_eq!(b.contexts.select(Some(0)).gl.attrib_depth, 15);
    // The light-state step posts no completion of its own.
    gl_token(&mut b, &mut r, SAVE_LIGHT_ATTRIBUTES_ON_ENGINE_ATTRIBUTE_STACK, &[0]);
    assert_eq!(r.ram.word(RETURN_PAGE), 0);
    // With no return page the completion is dropped and counted.
    r.core.return_page = None;
    gl_token(&mut b, &mut r, PUSH_ATTRIBUTE_STACK, &[0x20000]);
    assert_eq!(b.query_counts.unposted, 1);
}

/// A finish with its wait flag set is a bare read-back: the client polls the completion word until
/// the pipe has drained, so it is posted with a zero count. A finish with the flag clear is not waited
/// on and posts nothing.
#[test]
fn a_finish_with_its_wait_flag_posts_a_bare_completion() {
    let (mut b, mut r) = (backend(), Rig::new());
    let finish = |b: &mut KonaBackend, r: &mut Rig, flag: u32| {
        r.send(b, Job::Command { grammar: Grammar::Gl, selector: FINISH_DRAIN_PIPE_AND_SIGNAL_COMPLETION, args: &[flag], packed: 0, handled: false });
    };
    finish(&mut b, &mut r, 0);
    assert_eq!(r.ram.word(RETURN_PAGE), 0, "a finish nobody waits on posts nothing");
    finish(&mut b, &mut r, 1);
    assert_eq!((r.ram.word(RETURN_PAGE), r.ram.bytes(RETURN_PAGE + 4, 4)), (1, vec![0; 4]));
    // With no return page registered the completion is dropped and counted, as a query's is.
    r.core.return_page = None;
    finish(&mut b, &mut r, 1);
    assert_eq!(b.query_counts.unposted, 1);
}

/// A bare read whose image would not fit the page is not built: only the completion word is posted, the
/// page's reply area is not written past, and the read is counted. Inside a bracket the same read is
/// refused when it asks for more texels than the model will build.
#[test]
fn a_texel_read_too_big_to_answer_is_refused_and_counted() {
    let (mut b, mut r) = (backend(), Rig::new());
    load_grey_level(&mut b, &mut r);
    let mut big = GREY_READ;
    big[1..3].copy_from_slice(&[256, 256]);
    r.ram.put(RETURN_PAGE, &[0xaa; 0x4010]);
    read_texels(&mut b, &mut r, &big);
    assert_eq!(r.ram.word(RETURN_PAGE), 1, "the completion is posted so the client does not wait");
    assert_eq!(r.ram.bytes(RETURN_PAGE + 8, 0x4008), vec![0xaa; 0x4008], "no image is written");
    let mut huge = GREY_READ;
    huge[1..3].copy_from_slice(&[u32::MAX, u32::MAX]);
    read_texels(&mut b, &mut r, &huge);
    r.x(&mut b, START_SELECT_FEEDBACK_RETURN_DMA, &[0x1c0, 0x9000, 8192, 8192, 0, 0]);
    read_texels(&mut b, &mut r, &huge);
    assert!(b.contexts.select(Some(0)).x.feed_finish(&[0x100, 7]).is_none(), "the bracket is refused");
    let text = run(&mut b, &["gl"]).1;
    assert!(text.contains("3 (1 in a bracket)") && text.contains("3 too big"), "{text}");
}

/// The pack record's skips are the guest's, and they lengthen what the packer builds: a small read under a
/// huge skip is oversize, not an allocation that aborts the host, and is answered like any other oversize read.
#[test]
fn a_texel_read_under_a_huge_pack_skip_is_refused_not_allocated() {
    let (mut b, mut r) = (backend(), Rig::new());
    load_grey_level(&mut b, &mut r);
    let store = |b: &mut KonaBackend, r: &mut Rig, name: u32, value: u32| {
        r.send(b, Job::Command { grammar: Grammar::Gl, selector: SET_PIXEL_STORE_PARAMETER_INTEGER_FORM, args: &[name, value], packed: 0, handled: false });
    };
    let mut wide = GREY_READ;
    wide[1..3].copy_from_slice(&[8000, 1]);
    for (name, value) in [(0xd03, u32::MAX), (0xd04, u32::MAX)] {
        store(&mut b, &mut r, name, value);
        r.ram.put(RETURN_PAGE, &[0xaa; 0x4010]);
        read_texels(&mut b, &mut r, &wide);
        assert_eq!(r.ram.word(RETURN_PAGE), 1, "the completion is posted so the client does not wait");
        assert_eq!(r.ram.bytes(RETURN_PAGE + 8, 0x4008), vec![0xaa; 0x4008], "no image is written");
        store(&mut b, &mut r, name, 0);
    }
    // In a bracket the same read is refused, not built.
    store(&mut b, &mut r, 0xd03, u32::MAX);
    r.x(&mut b, START_SELECT_FEEDBACK_RETURN_DMA, &[0x1c0, 0x9000, 16000, 16000, 0, 0]);
    read_texels(&mut b, &mut r, &wide);
    assert!(b.contexts.select(Some(0)).x.feed_finish(&[0x100, 7]).is_none(), "the bracket is refused");
    let text = run(&mut b, &["gl"]).1;
    assert!(text.contains("3 too big"), "{text}");
    // Without the skip the same read is answered.
    store(&mut b, &mut r, 0xd03, 0);
    read_texels(&mut b, &mut r, &wide);
    let text = run(&mut b, &["gl"]).1;
    assert!(text.contains("texel read-backs  4 (1 in a bracket)") && text.contains("3 too big"), "{text}");
    assert_eq!(r.ram.bytes(RETURN_PAGE + 8, 4), [0, 0, 4, 4], "16000 bytes fit the page, and the first texels are the level's");
}

/// The texel cap is its own cap: a read of one-byte texels, 2048 by 1024, packs to 2 MiB, well under the byte
/// cap, but would build 2^21 colours. It is refused in a bracket (where the byte cap does not stop it).
#[test]
fn a_texel_read_past_the_texel_cap_is_refused_though_its_bytes_are_few() {
    let (mut b, mut r) = (backend(), Rig::new());
    load_grey_level(&mut b, &mut r);
    let mut read = GREY_READ;
    read[1..3].copy_from_slice(&[2048, 1024]);
    read[3..5].copy_from_slice(&[0, 1]);
    read[10..12].copy_from_slice(&[2047, 1023]);
    r.x(&mut b, START_SELECT_FEEDBACK_RETURN_DMA, &[0x1c0, 0x9000, 1 << 21, 1 << 21, 0, 0]);
    read_texels(&mut b, &mut r, &read);
    assert!(b.contexts.select(Some(0)).x.feed_finish(&[0x100, 7]).is_none(), "the bracket is refused");
    let text = run(&mut b, &["gl"]).1;
    assert!(text.contains("1 too big") && text.contains("0 for a level not loaded"), "{text}");
    // Half the texels (the cap itself) is answered: the level is not that big, so its texels are zeros.
    read[2] = 512;
    read[11] = 511;
    r.x(&mut b, START_SELECT_FEEDBACK_RETURN_DMA, &[0x1c0, 0x9000, 1 << 20, 1 << 20, 0, 0]);
    read_texels(&mut b, &mut r, &read);
    let done = b.contexts.select(Some(0)).x.feed_finish(&[0x100, 7]).expect("a read at the cap is answered");
    assert_eq!(done.data.len(), 1 << 20);
    let text = run(&mut b, &["gl"]).1;
    assert!(text.contains("1 too big"), "{text}");
}

/// In a bracket, a read of a format or type the packer does not know refuses the bracket, as it has
/// nothing to describe.
#[test]
fn an_unsupported_texel_read_in_a_bracket_refuses_it() {
    let (mut b, mut r) = (backend(), Rig::new());
    load_grey_level(&mut b, &mut r);
    r.x(&mut b, START_SELECT_FEEDBACK_RETURN_DMA, &[0x1c0, 0x9000, 8192, 8192, 0, 0]);
    let mut unknown = GREY_READ;
    unknown[3] = 0x77;
    read_texels(&mut b, &mut r, &unknown);
    assert!(b.contexts.select(Some(0)).x.feed_finish(&[0x100, 7]).is_none());
    let text = run(&mut b, &["gl"]).1;
    assert!(text.contains("1 for a format or type the packer does not know"), "{text}");
}

/// Inside a select-feed bracket the read goes to the bracket's host buffer and not the return page.
#[test]
fn a_bracketed_texel_read_back_leaves_the_return_page_alone() {
    let (mut b, mut r) = (backend(), Rig::new());
    load_grey_level(&mut b, &mut r);
    // A bracket for the whole read: 64 rows of 128 bytes in one line.
    r.x(&mut b, START_SELECT_FEEDBACK_RETURN_DMA, &[0x1c0, 0x9000, 8192, 8192, 0, 0]);
    r.send(&mut b, Job::Command { grammar: Grammar::Gl, selector: QUERY_TEXTURE_IMAGE, args: &GREY_READ, packed: 0, handled: false });
    assert_eq!(r.ram.word(RETURN_PAGE), 0, "the bracket's finish returns it, by output DMA");
    let text = run(&mut b, &["gl"]).1;
    assert!(text.contains("texel read-backs  1 (1 in a bracket)"), "{text}");
    let c = b.contexts.select(Some(0));
    let done = c.x.feed_finish(&[0x100, 7]).expect("the bracket holds the read");
    assert_eq!((done.data.len(), done.line, done.echo), (8192, 8192, 7));
    assert_eq!(&done.data[..4], &[0, 0, 0x04, 0x04], "texel 0 is 0 and texel 1 is 4 x 257");
}

/// A bracketed texel read over 0x3ff0 bytes (the 512 x 256 level Quake reads back) is delivered to the range the
/// start token describes: its device address from the map page and the offset (plus 0x9000), the padded image
/// as one line of the full length with no stride. Nothing goes to the return page, and the finish posts event
/// 0x100 with the slot, no bytes remaining and no overflow.
#[test]
fn a_bracketed_texel_read_is_delivered_by_output_dma_to_the_described_range() {
    use crate::dev::kona::defs::{MAPRAM_KIND_16K, MAPRAM_PAGE_SHIFT};
    let (mut b, mut r) = (backend(), Rig::new());
    let host = 0x40_0000u64;
    let dev = (8u64 << MAPRAM_PAGE_SHIFT) + 0x100;
    for page in 0..18u64 {
        r.core.map_ram[8 + page as usize] = MAPRAM_KIND_16K | ((host + (page << MAPRAM_PAGE_SHIFT)) >> 12) as u32;
    }
    let mut header = [0u32; 20];
    header[..5].copy_from_slice(&[0x0de1, 512, 256, 5, 1]);
    header[9] = 0x30;
    header[13] = 511;
    header[14] = 255;
    header[16] = 0x89;
    header[17] = 0x2200;
    header[18] = 0x400;
    let grey = |x: u32, y: u32| (x + 2 * y) % 256;
    let words: Vec<u32> = (0..512 * 256).map(|i| { let g = grey(i % 512, i / 512); g << 24 | g << 16 | g << 8 | 0xff }).collect();
    assert_eq!(r.send(&mut b, Job::Command { grammar: Grammar::Gl, selector: DRAW_TEXELS_INTO_TEXTURE_MEMORY, args: &header, packed: 0, handled: false }), Reply::Data(512 * 256));
    r.send(&mut b, Job::Data { grammar: Grammar::Gl, selector: DRAW_TEXELS_INTO_TEXTURE_MEMORY, args: &header, words: &words });
    // The read: 512 x 256 luminance in unsigned shorts is 262144 bytes, far over the page.
    let len = 512 * 256 * 2u32;
    r.x(&mut b, START_SELECT_FEEDBACK_RETURN_DMA, &[(dev >> 12) as u32, 0x9000 | (dev & 0xfff) as u32, len, len, 0, 0]);
    let read = [0x0de1, 512, 256, 6, 3, 0xd, 0, 0, 0, 0, 511, 255, 0, 0x0385_0030, 0x89, 0x44, 0x400, 0];
    r.ram.put(RETURN_PAGE, &[0; 16]);
    read_texels(&mut b, &mut r, &read);
    assert_eq!(r.ram.bytes(RETURN_PAGE, 16), [0; 16], "no header and no completion word on the return page");
    assert_eq!(r.ram.bytes(host + 0x100, 4), [0; 4], "and nothing is written until the finish");
    r.x(&mut b, FINISH_SELECT_FEEDBACK_WITH_EVENT_CODE, &[0x100, 7]);
    let image = r.ram.bytes(host + 0x100, len as usize);
    let texel = |x: usize, y: usize| u16::from_be_bytes([image[2 * (512 * y + x)], image[2 * (512 * y + x) + 1]]);
    for (x, y) in [(0, 0), (1, 0), (0, 1), (511, 255), (300, 200), (127, 255)] {
        assert_eq!(texel(x, y) as u32, grey(x as u32, y as u32) * 257, "texel ({x}, {y})");
    }
    assert_eq!(r.ram.bytes(host + 0xff, 1), [0], "the byte before the range is untouched");
    assert_eq!(r.ram.bytes(host + 0x100 + len as u64, 4), [0; 4], "and the one after");
    assert_eq!([r.ram.word(REASON_PAGE), r.ram.word(REASON_PAGE + 4), r.ram.word(REASON_PAGE + 8), r.ram.word(REASON_PAGE + 12)], [0x100, 7, 0, 0], "event 0x100: the slot, nothing remaining, no overflow");
    let text = run(&mut b, &["gl"]).1;
    assert!(text.contains("texel read-backs  1 (1 in a bracket)"), "{text}");
    // A descriptor whose length is not the image's is refused, and nothing is posted.
    r.ram.put(REASON_PAGE, &[0; 16]);
    r.x(&mut b, START_SELECT_FEEDBACK_RETURN_DMA, &[(dev >> 12) as u32, 0x9000 | (dev & 0xfff) as u32, len - 4, len - 4, 0, 0]);
    read_texels(&mut b, &mut r, &read);
    r.x(&mut b, FINISH_SELECT_FEEDBACK_WITH_EVENT_CODE, &[0x100, 8]);
    assert_eq!(r.ram.word(REASON_PAGE), 0);
    assert!(run(&mut b, &["status"]).1.contains("read brackets   1 refused"));
}

/// A GL window on the rig's context: 64 by 64 at the origin, an orthographic projection of the same
/// size, the back buffer for drawing and reading.
fn rig_pixel_window(b: &mut KonaBackend, r: &mut Rig) {
    use crate::dev::kona::defs::gl::{BACK, PROJECTION};
    let mut gl = |b: &mut KonaBackend, selector: u32, args: &[u32]| {
        r.send(b, Job::Command { grammar: Grammar::Gl, selector, args, packed: 0, handled: false });
    };
    gl(b, CONFIGURE_FRAME_BUFFER_LAYOUT, &[0, 8, 0, 1, 0, 0, 24, 0, 0, 0]);
    gl(b, SET_VIEWPORT, &[0, 0, 64, 64]);
    gl(b, SELECT_CURRENT_MATRIX_STACK, &[PROJECTION]);
    gl(b, LOAD_MATRIX_16_FLOATS, &crate::dev::gl::math::ortho(0.0, 64.0, 0.0, 64.0, -1.0, 1.0).map(f32::to_bits));
    gl(b, SELECT_DRAW_BUFFER, &[BACK]);
    gl(b, SELECT_READ_BUFFER, &[BACK]);
    gl(b, SET_RASTER_POSITION_2_COMPONENTS_32_BIT_INTEGER, &[0, 0]);
}

/// A `w` by `h` RGBA image of bytes drawn at the raster position, pixel (x, y) being `(x, y, x ^ y,
/// 0xff)`.
fn rig_draw_pattern(b: &mut KonaBackend, r: &mut Rig, w: u32, h: u32) {
    use crate::dev::kona::defs::pixel::*;
    let hdr = [w, h, FORMAT_RGBA, TYPE_UNSIGNED_BYTE, 0, 0, 0, 0];
    let words: Vec<u32> = (0..w * h).map(|i| (i % w) << 24 | (i / w) << 16 | ((i % w) ^ (i / w)) << 8 | 0xff).collect();
    assert_eq!(r.send(b, Job::Command { grammar: Grammar::Gl, selector: DRAW_PIXEL_RECTANGLE_FROM_HOST_DATA, args: &hdr, packed: 0, handled: false }), Reply::Data(w * h));
    r.send(b, Job::Data { grammar: Grammar::Gl, selector: DRAW_PIXEL_RECTANGLE_FROM_HOST_DATA, args: &hdr, words: &words });
}

/// Map `pages` pages of host memory at `host` into the device address space from page 8 of the map
/// RAM, and return the device address of byte `at` into them.
fn rig_map_host(r: &mut Rig, host: u64, pages: u64, at: u64) -> u64 {
    use crate::dev::kona::defs::{MAPRAM_KIND_16K, MAPRAM_PAGE_SHIFT};
    for page in 0..pages {
        r.core.map_ram[8 + page as usize] = MAPRAM_KIND_16K | ((host + (page << MAPRAM_PAGE_SHIFT)) >> 12) as u32;
    }
    (8u64 << MAPRAM_PAGE_SHIFT) + at
}

/// A read of 16 KiB or more comes inside a select-feed bracket, as a texel read-back over the page
/// does, and goes to the bracket's buffer by output DMA: here 64 by 64 RGBA, which is 16384 bytes,
/// 16 over what the return page takes, the read that hung the client and wedged the pipe when
/// nothing delivered it. The bytes reach host memory at the descriptor's address on the finish,
/// which posts event 0x100 with the slot echoed, nothing remaining and no overflow; the return page
/// is untouched, and a byte either side of the range is too.
#[test]
fn a_pixel_read_inside_a_bracket_is_delivered_by_output_dma() {
    use crate::dev::kona::defs::pixel::*;
    let (mut b, mut r) = (backend(), Rig::new());
    r.x(&mut b, WINDOW_CLIP_SETUP, &[0, 0, 64, 64, 0, 0, 64, 64, 0, 13, 0]);
    rig_pixel_window(&mut b, &mut r);
    rig_draw_pattern(&mut b, &mut r, 64, 64);
    let host = 0x40_0000u64;
    let len = 64 * 64 * 4u32;
    let dev = rig_map_host(&mut r, host, 2, 0x100);
    r.x(&mut b, START_SELECT_FEEDBACK_RETURN_DMA, &[(dev >> 12) as u32, 0x9000 | (dev & 0xfff) as u32, len, len, 0, 0]);
    r.ram.put(RETURN_PAGE, &[0; 16]);
    r.send(&mut b, Job::Command { grammar: Grammar::Gl, selector: READ_PIXEL_RECTANGLE_BACK_TO_HOST, args: &[0, 0, 64, 64, FORMAT_RGBA, TYPE_UNSIGNED_BYTE, 0], packed: 0, handled: false });
    assert_eq!(r.ram.bytes(RETURN_PAGE, 16), [0; 16], "nothing on the return page");
    assert_eq!(r.ram.bytes(host + 0x100, 4), [0; 4], "and nothing is written until the finish");
    r.x(&mut b, FINISH_SELECT_FEEDBACK_WITH_EVENT_CODE, &[0x100, 7]);
    let image = r.ram.bytes(host + 0x100, len as usize);
    for (x, y) in [(0usize, 0usize), (1, 0), (0, 1), (63, 63), (17, 40), (40, 17)] {
        assert_eq!(image[4 * (64 * y + x)..][..4], [x as u8, y as u8, (x ^ y) as u8, 0xff], "pixel ({x}, {y})");
    }
    assert_eq!(r.ram.bytes(host + 0xff, 1), [0], "the byte before the range is untouched");
    assert_eq!(r.ram.bytes(host + 0x100 + len as u64, 4), [0; 4], "and the one after");
    assert_eq!([r.ram.word(REASON_PAGE), r.ram.word(REASON_PAGE + 4), r.ram.word(REASON_PAGE + 8), r.ram.word(REASON_PAGE + 12)], [0x100, 7, 0, 0]);
    let text = run(&mut b, &["gl"]).1;
    assert!(text.contains("1 reads by DMA"), "{text}");
    // The same size with no bracket has no buffer to go to, and is counted, not answered.
    r.send(&mut b, Job::Command { grammar: Grammar::Gl, selector: READ_PIXEL_RECTANGLE_BACK_TO_HOST, args: &[0, 0, 64, 64, FORMAT_RGBA, TYPE_UNSIGNED_BYTE, 0], packed: 0, handled: false });
    assert_eq!(r.ram.bytes(RETURN_PAGE, 16), [0; 16]);
    assert!(run(&mut b, &["gl"]).1.contains("1 reads too big for the page with no buffer"));
}

/// On the DMA route the board writes the caller's finished layout (the whole pack record,
/// inferred): a 13 by 5 RED read of bytes under alignment 4 is five rows of 13 bytes at a stride of
/// 16, 80 bytes, as the descriptor names it; a descriptor of any other length refuses the bracket
/// and posts nothing.
#[test]
fn a_pixel_read_by_dma_is_laid_out_under_the_pack_record() {
    use crate::dev::kona::defs::pixel::*;
    let (mut b, mut r) = (backend(), Rig::new());
    r.x(&mut b, WINDOW_CLIP_SETUP, &[0, 0, 64, 64, 0, 0, 64, 64, 0, 13, 0]);
    rig_pixel_window(&mut b, &mut r);
    rig_draw_pattern(&mut b, &mut r, 13, 5);
    let host = 0x40_0000u64;
    let dev = rig_map_host(&mut r, host, 1, 0);
    let read = [0, 0, 13, 5, FORMAT_RED, TYPE_UNSIGNED_BYTE, 0];
    r.x(&mut b, START_SELECT_FEEDBACK_RETURN_DMA, &[(dev >> 12) as u32, 0x9000 | (dev & 0xfff) as u32, 80, 80, 0, 0]);
    r.send(&mut b, Job::Command { grammar: Grammar::Gl, selector: READ_PIXEL_RECTANGLE_BACK_TO_HOST, args: &read, packed: 0, handled: false });
    r.x(&mut b, FINISH_SELECT_FEEDBACK_WITH_EVENT_CODE, &[0x100, 1]);
    let image = r.ram.bytes(host, 80);
    for y in 0..5usize {
        let row: Vec<u8> = (0..16).map(|x| if x < 13 { x as u8 } else { 0 }).collect();
        assert_eq!(image[16 * y..][..16], row[..], "row {y}");
    }
    assert_eq!(r.ram.word(REASON_PAGE + 4), 1);
    r.ram.put(REASON_PAGE, &[0; 16]);
    r.x(&mut b, START_SELECT_FEEDBACK_RETURN_DMA, &[(dev >> 12) as u32, 0x9000 | (dev & 0xfff) as u32, 65, 65, 0, 0]);
    r.send(&mut b, Job::Command { grammar: Grammar::Gl, selector: READ_PIXEL_RECTANGLE_BACK_TO_HOST, args: &read, packed: 0, handled: false });
    r.x(&mut b, FINISH_SELECT_FEEDBACK_WITH_EVENT_CODE, &[0x100, 2]);
    assert_eq!(r.ram.word(REASON_PAGE), 0, "a descriptor that is not the read's size posts nothing");
}

/// A read in a colour-index window returns the indices the draw wrote, from the colour words they
/// sit in (12-bit index, shifted 4), through the index shift and offset; in a window of colours a
/// colour-index read has nothing to read and is refused, counted.
#[test]
fn a_colour_index_window_is_read_in_indices() {
    use crate::dev::kona::defs::gl::{COLOR_BUFFER_BIT, FRONT};
    use crate::dev::kona::defs::pixel::*;
    let mut b = KonaBackend::new(KonaConfig::default());
    command(&mut b, Grammar::Kernel, LOAD_ONE_ENTRY_OF_DISPLAY_ID_TABLE, &[14, 0x24005c]);
    command(&mut b, Grammar::Kernel, DISPLAY_MODE_BYTE, &[0, 14, 8]);
    command(&mut b, Grammar::Kernel, WINDOW_CLIP_SETUP, &[0, 0, 64, 64, 0, 0, 64, 64, 0, 14, 0]);
    command(&mut b, Grammar::Gl, CONFIGURE_FRAME_BUFFER_LAYOUT, &[0, 8, 0, 1, 0, 0, 24, 0, 0, 0]);
    command(&mut b, Grammar::Gl, SET_VIEWPORT, &[0, 0, 64, 64]);
    command(&mut b, Grammar::Gl, SELECT_DRAW_BUFFER, &[FRONT]);
    command(&mut b, Grammar::Gl, SELECT_READ_BUFFER, &[FRONT]);
    command(&mut b, Grammar::Gl, SET_COLOUR_INDEX_CLEAR_VALUE, &[300.0f32.to_bits()]);
    command(&mut b, Grammar::Gl, CLEAR_SELECTED_BUFFERS, &[COLOR_BUFFER_BIT]);
    command(&mut b, Grammar::Gl, SET_CURRENT_COLOUR_INDEX_SIGNED_SHORT, &[6 << 16]);
    command(&mut b, Grammar::Gl, BEGIN_PRIMITIVE_RECTANGLE, &[0xe]);
    for (x, y) in [(10.5, 10.5), (30.5, 10.5), (30.5, 30.5), (10.5, 30.5)] {
        command(&mut b, Grammar::Gl, SUBMIT_VERTEX_2_COMPONENTS_SINGLE_FLOAT, &[ndc(x), ndc(y)]);
    }
    command(&mut b, Grammar::Gl, END_OPEN_PRIMITIVE, &[0]);
    command(&mut b, Grammar::Kernel, WINDOW_SYSTEM_PIXEL_PATH_FINISH, &[0]);
    let read = |b: &mut KonaBackend, args: &[u32]| {
        let screen = b.gl.screen();
        let ds = b.contexts.select(None).gl.draw_state(screen);
        b.arrays(&ds);
        let c = b.contexts.select(None);
        let env = gl::pixels::Env { store: &b.store, ds };
        gl::pixels::plane_command(&mut c.gl, &mut b.pix_counts, &env, READ_PIXEL_RECTANGLE_BACK_TO_HOST, args)
    };
    let out = read(&mut b, &[9, 20, 1, 1, FORMAT_COLOUR_INDEX, TYPE_UNSIGNED_SHORT, 0]);
    assert_eq!(out, gl::pixels::Out::Reply(vec![0x01, 0x2c, 0, 0]), "the clear index, 300, wants all twelve bits");
    let out = read(&mut b, &[20, 20, 2, 1, FORMAT_COLOUR_INDEX, TYPE_UNSIGNED_SHORT, 0]);
    assert_eq!(out, gl::pixels::Out::Reply(vec![0, 6, 0, 6]), "and the rectangle's index is 6");
    let out = read(&mut b, &[20, 20, 2, 1, FORMAT_COLOUR_INDEX, TYPE_UNSIGNED_BYTE, 0]);
    assert_eq!(out, gl::pixels::Out::Reply(vec![6, 6, 0, 0]));
    // Pixels outside the window read zero, and the shift and offset apply to the indices as GL's
    // do.
    command(&mut b, Grammar::Gl, SET_PIXEL_TRANSFER_PARAMETER_FLOAT_FORM, &[INDEX_OFFSET, 10]);
    let out = read(&mut b, &[20, 20, 1, 1, FORMAT_COLOUR_INDEX, TYPE_UNSIGNED_BYTE, 0]);
    assert_eq!(out, gl::pixels::Out::Reply(vec![16, 0, 0, 0]));
    // A window of colours has no indices.
    let mut b = pixel_window();
    let screen = b.gl.screen();
    let ds = b.contexts.select(None).gl.draw_state(screen);
    b.arrays(&ds);
    let c = b.contexts.select(None);
    let env = gl::pixels::Env { store: &b.store, ds };
    let out = gl::pixels::plane_command(&mut c.gl, &mut b.pix_counts, &env, READ_PIXEL_RECTANGLE_BACK_TO_HOST, &[0, 0, 1, 1, FORMAT_COLOUR_INDEX, TYPE_UNSIGNED_BYTE, 0]);
    assert_eq!(out, gl::pixels::Out::Nothing);
    assert_eq!(b.pix_counts.refused, 1);
}

/// The reply to a read that fits the page has the rows of the width's own bytes: the client lays
/// them out under the pack alignment, row length and skips as it copies them to the caller's
/// buffer, so the board that padded them too put the padding in twice (a 13 by 5 RED read of bytes
/// arrived with 3 zeros after each row, which the client then skipped again). Byte order stays the
/// board's.
#[test]
fn a_read_reply_is_not_padded_by_the_pack_record() {
    use crate::dev::kona::defs::pixel::*;
    let mut b = pixel_window();
    at(&mut b, 0, 0);
    let words: Vec<u32> = (0..13 * 5).map(|i| (i % 13) << 24 | 0xff).collect();
    draw_pixels(&mut b, 13, 5, FORMAT_RGBA, TYPE_UNSIGNED_BYTE, &words);
    command(&mut b, Grammar::Gl, SELECT_READ_BUFFER, &[crate::dev::kona::defs::gl::BACK]);
    let tight: Vec<u8> = (0..5).flat_map(|_| 0..13u8).collect();
    let mut want = tight.clone();
    want.resize(68, 0);
    let mut read = |b: &mut KonaBackend, store: &[(u32, u32)], args: &[u32]| {
        for &(name, value) in store {
            command(b, Grammar::Gl, SET_PIXEL_STORE_PARAMETER_INTEGER_FORM, &[name, value]);
        }
        let screen = b.gl.screen();
        let ds = b.contexts.select(None).gl.draw_state(screen);
        b.arrays(&ds);
        let c = b.contexts.select(None);
        let env = gl::pixels::Env { store: &b.store, ds };
        gl::pixels::plane_command(&mut c.gl, &mut b.pix_counts, &env, READ_PIXEL_RECTANGLE_BACK_TO_HOST, args)
    };
    let args = [0, 0, 13, 5, FORMAT_RED, TYPE_UNSIGNED_BYTE, 0];
    // Alignment 4 (GL's default), then 1: the same bytes. Row length 21 and skips change nothing
    // either.
    assert_eq!(read(&mut b, &[], &args), gl::pixels::Out::Reply(want.clone()));
    assert_eq!(read(&mut b, &[(PACK_BASE + 5, 1)], &args), gl::pixels::Out::Reply(want.clone()));
    assert_eq!(read(&mut b, &[(PACK_BASE + 5, 4), (PACK_BASE + 2, 21), (PACK_BASE + 3, 2), (PACK_BASE + 4, 3)], &args), gl::pixels::Out::Reply(want));
}

/// A pixel draw in a colour-index window writes indices into the colour words (12-bit index,
/// shifted 4), not a grey or mapped colour: the index shift, offset and index-to-index map apply,
/// the index wraps to the window's width, and a copy within the window moves indices through the
/// same stages.
#[test]
fn a_pixel_draw_in_a_colour_index_window_writes_indices() {
    use crate::dev::kona::defs::gl::FRONT;
    use crate::dev::kona::defs::pixel::*;
    let mut b = KonaBackend::new(KonaConfig::default());
    command(&mut b, Grammar::Kernel, LOAD_ONE_ENTRY_OF_DISPLAY_ID_TABLE, &[14, 0x24005c]);
    command(&mut b, Grammar::Kernel, DISPLAY_MODE_BYTE, &[0, 14, 8]);
    command(&mut b, Grammar::Kernel, WINDOW_CLIP_SETUP, &[0, 0, 64, 64, 0, 0, 64, 64, 0, 14, 0]);
    command(&mut b, Grammar::Gl, CONFIGURE_FRAME_BUFFER_LAYOUT, &[0, 8, 0, 1, 0, 0, 24, 0, 0, 0]);
    command(&mut b, Grammar::Gl, SET_VIEWPORT, &[0, 0, 64, 64]);
    command(&mut b, Grammar::Gl, SELECT_DRAW_BUFFER, &[FRONT]);
    command(&mut b, Grammar::Gl, SELECT_READ_BUFFER, &[FRONT]);
    // The front buffer is the layout's second colour word.
    const FRONT_WORD: usize = 1;
    let at = |b: &mut KonaBackend, x: f32, y: f32| command(b, Grammar::Gl, SET_RASTER_POSITION_2_COMPONENTS_FLOAT, &[ndc(x), ndc(y)]);
    let words = |b: &mut KonaBackend| {
        command(b, Grammar::Kernel, WINDOW_SYSTEM_PIXEL_PATH_FINISH, &[0]);
        (0..4).map(|i| b.store.read(20 + i, 20, FRONT_WORD)).collect::<Vec<_>>()
    };
    // Two unsigned shorts, 17 and 300, make one row word.
    at(&mut b, 20.5, 20.5);
    let hdr = [2, 1, FORMAT_COLOUR_INDEX, TYPE_UNSIGNED_SHORT, 0, 0, 0, 0];
    let draw = |b: &mut KonaBackend| {
        assert_eq!(command(b, Grammar::Gl, DRAW_PIXEL_RECTANGLE_FROM_HOST_DATA, &hdr), Reply::Data(1));
        send(b, Job::Data { grammar: Grammar::Gl, selector: DRAW_PIXEL_RECTANGLE_FROM_HOST_DATA, args: &hdr, words: &[17 << 16 | 300] });
    };
    draw(&mut b);
    assert_eq!(words(&mut b), [17 << 4, 300 << 4, 0, 0], "the index sits at the window's shift, not a grey or a replicated byte");
    // An offset applies, and the result wraps to the twelve bits.
    command(&mut b, Grammar::Gl, SET_PIXEL_TRANSFER_PARAMETER_FLOAT_FORM, &[INDEX_OFFSET, 4000]);
    draw(&mut b);
    assert_eq!(words(&mut b), [4017 << 4, 4300 % 4096 << 4, 0, 0]);
    command(&mut b, Grammar::Gl, SET_PIXEL_TRANSFER_PARAMETER_FLOAT_FORM, &[INDEX_OFFSET, 0]);
    // A copy of those pixels moves indices, and applies the offset to them.
    command(&mut b, Grammar::Gl, SET_PIXEL_TRANSFER_PARAMETER_FLOAT_FORM, &[INDEX_OFFSET, 6]);
    at(&mut b, 40.5, 20.5);
    let screen = b.gl.screen();
    let ds = b.contexts.select(None).gl.draw_state(screen);
    b.arrays(&ds);
    let c = b.contexts.select(None);
    let env = gl::pixels::Env { store: &b.store, ds };
    gl::pixels::plane_command(&mut c.gl, &mut b.pix_counts, &env, COPY_PIXEL_RECTANGLE_WITHIN_FRAME_BUFFER, &[20, 20, 2, 1, 0, COPY_COLOUR, 0]);
    command(&mut b, Grammar::Kernel, WINDOW_SYSTEM_PIXEL_PATH_FINISH, &[0]);
    let copied: Vec<u32> = (0..2).map(|i| b.store.read(40 + i, 20, FRONT_WORD)).collect();
    assert_eq!(copied, [(4017 + 6) % 4096 << 4, (4300 % 4096 + 6) << 4]);
}

/// A DAC cursor load is decoded and held by channel, and draws nothing until channel 0's control register enables it; a load of another length, or
/// for a channel the table does not give, is refused and counted.
#[test]
fn a_dac_cursor_load_is_held_by_channel() {
    let mut b = backend();
    let mut load = [0u32; 263];
    load[0] = 3;
    load[1] = 0x3ff;
    send(&mut b, Job::Video { command: VIDEO_DAC_CURSOR_LOAD, data: &load });
    send(&mut b, Job::Video { command: VIDEO_DAC_CURSOR_LOAD, data: &[0; 5] });
    let mut stray = load;
    stray[0] = 0x1234;
    send(&mut b, Job::Video { command: VIDEO_DAC_CURSOR_LOAD, data: &stray });
    // Channel 8 is the first value past the table's 0 to 7.
    stray[0] = 8;
    send(&mut b, Job::Video { command: VIDEO_DAC_CURSOR_LOAD, data: &stray });
    let status = run(&mut b, &["status"]).1;
    assert!(status.contains("DAC cursor        4 loads, 3 refused; held for channels [3]; channel 0's is not drawn"), "{status}");
    assert!(status.contains("control value for the owning channel: 0x5 = X11 cursor, enabled, not interlaced"), "{status}");
    assert_eq!(b.tables.lock().dac().cursors().next().unwrap().foreground, [0x3ff, 0, 0]);
    assert_eq!(crate::dev::kona::tables::VIDEO.get(VIDEO_DAC_CURSOR_LOAD as usize).copied().flatten(), Some(263));
}

/// A video setter reaches the display generator's channels, a Get changes nothing, and a reset
/// puts the power-up state back.
#[test]
fn video_setters_reach_the_channels_and_a_reset_restores_them() {
    let mut b = backend();
    send(&mut b, Job::Video { command: VIDEO_CURSOR_PRIORITY, data: &[2, 9] });
    send(&mut b, Job::Video { command: VIDEO_GET_BOARD_CONFIG, data: &[0] });
    assert_eq!(b.channels.channel[2].cursor_priority, 9);
    assert_eq!(b.channels.channel[0], display::channels::Channels::new(2).channel[0]);
    // `kona status` says which channels hold blanking.
    assert!(run(&mut b, &["status"]).1.contains("video channels   enabled [0], blanked [] (channel 0's, and all while temporarily blanked, are drawn black), frame 1280 x 1024\n"));
    send(&mut b, Job::Video { command: VIDEO_CHANNEL_BLANKING, data: &[3, 1] });
    send(&mut b, Job::Video { command: VIDEO_TEMP_BLANK, data: &[1] });
    assert!(run(&mut b, &["status"]).1.contains("blanked [3] (channel 0's, and all while temporarily blanked, are drawn black), frame 1280 x 1024, all blanked\n"));
    b.reset();
    assert_eq!(b.channels.channel[2].cursor_priority, 0);
}

/// The channels are part of the saved small state: what the video setters left comes back from a snapshot
/// into a fresh backend, and a snapshot of the restored backend is the same bytes.
#[test]
fn the_video_channels_are_saved_and_restored_with_the_backend() {
    use crate::dev::kona::stream::{Reader, Writer};
    let mut b = backend();
    send(&mut b, Job::Video { command: VIDEO_CURSOR_PRIORITY, data: &[2, 9] });
    send(&mut b, Job::Video { command: VIDEO_CHANNEL_BLANKING, data: &[4, 1] });
    send(&mut b, Job::Video { command: VIDEO_TEMP_BLANK, data: &[1] });
    send(&mut b, Job::Video { command: VIDEO_RESIZE, data: &[1, 10, 500, 640, 480, 1, 1, 0] });
    let mut w = Writer::default();
    b.visit_small(&mut w).unwrap();
    let mut back = backend();
    assert_ne!(back.channels, b.channels);
    let mut r = Reader::new(&w.out);
    back.visit_small(&mut r).unwrap();
    r.finish().unwrap();
    assert_eq!(back.channels, b.channels);
    assert_eq!((back.channels.channel[2].cursor_priority, back.channels.channel[4].blanked, back.channels.blanked, back.channels.channel[1].input), (9, true, true, [10, 500, 640, 480]));
    let mut again = Writer::default();
    back.visit_small(&mut again).unwrap();
    assert_eq!(again.out, w.out);
}

/// The loader releasing the firmware runs the stored combination, or the built-in one when there is none: the
/// job reaches the channels without any recording. After it channel 0 runs 1280x1024_60.vfo, channel 1 holds
/// its format blanked, and the Gets say so.
#[test]
fn starting_the_firmware_runs_the_built_in_combination_on_the_channels() {
    let (mut b, mut r) = (backend(), Rig::new());
    let ram = r.ram.clone();
    let words = |at: u64, n: usize| ram.bytes(at, 4 * n).chunks(4).map(|c| u32::from_be_bytes(c.try_into().unwrap())).collect::<Vec<u32>>();
    r.send(&mut b, Job::Video { command: VIDEO_GET_CHANNEL_CONFIG, data: &[0] });
    assert_eq!(words(RETURN_PAGE + 8 + 4, 5)[1], 0, "before the start channel 0 holds no named format");
    r.send(&mut b, Job::FirmwareStart { combination: &[] });
    r.send(&mut b, Job::Video { command: VIDEO_GET_CHANNEL_CONFIG, data: &[0] });
    let w = words(RETURN_PAGE + 8, 21);
    assert_eq!(w[0], 7, "the reply id");
    assert_eq!(&w[2..6], &[u32::from_be_bytes(*b"1280"), u32::from_be_bytes(*b"x102"), u32::from_be_bytes(*b"4_60"), u32::from_be_bytes(*b".vfo")]);
    r.send(&mut b, Job::Video { command: VIDEO_GET_CHANNEL_BLANKING, data: &[1] });
    assert_eq!(words(RETURN_PAGE + 8, 3), [9, 1, 10], "channel 1 is blanked");
    r.send(&mut b, Job::Video { command: VIDEO_GET_CHANNEL_BLANKING, data: &[0] });
    assert_eq!(words(RETURN_PAGE + 8, 3), [9, 0, 9], "and channel 0 is not");
}

/// A Get posts its reply on the return page from byte 8 and the ready word with it (`post_return`
/// writes the data first and the ready word last); a setter posts nothing.
#[test]
fn a_video_get_posts_its_reply_and_the_ready_word() {
    let (mut b, mut r) = (backend(), Rig::new());
    r.send(&mut b, Job::Video { command: VIDEO_CURSOR_PRIORITY, data: &[0, 1] });
    assert_eq!(r.ram.bytes(RETURN_PAGE, 36), [0; 36], "a setter has no reply");
    r.send(&mut b, Job::Video { command: VIDEO_GET_BOARD_CONFIG, data: &[0] });
    let words = |v: &[u32]| v.iter().flat_map(|w| w.to_be_bytes()).collect::<Vec<u8>>();
    assert_eq!(r.ram.bytes(RETURN_PAGE, 4), words(&[1]));
    assert_eq!(r.ram.bytes(RETURN_PAGE + 8, 28), words(&[6, 1, 0xff, 0, 0, 0, 0x106]));
    // With no return page the reply is dropped, logged, and the board carries on.
    r.ram.put(RETURN_PAGE, &[0; 36]);
    r.core.return_page = None;
    r.send(&mut b, Job::Video { command: VIDEO_GET_BOARD_CONFIG, data: &[0] });
    assert_eq!(r.ram.bytes(RETURN_PAGE, 36), [0; 36]);
}

/// A 64 x 64 window at the screen's corner, double buffered, with a viewport that
/// maps clip coordinates -1 to 1 onto it; draws go to colour word 2. The lit pixels
/// of that word are read back with `lit_pixels`.
fn gl_window() -> KonaBackend {
    let mut b = KonaBackend::new(KonaConfig::default());
    command(&mut b, Grammar::Kernel, WINDOW_CLIP_SETUP, &[0, 0, 64, 64, 0, 0, 64, 64, 0, 13, 0]);
    command(&mut b, Grammar::Gl, CONFIGURE_FRAME_BUFFER_LAYOUT, &[0, 8, 0, 1, 0, 0, 24, 0, 0, 0]);
    command(&mut b, Grammar::Gl, SET_VIEWPORT, &[0, 0, 64, 64]);
    command(&mut b, Grammar::Gl, SET_CURRENT_COLOUR_4_COMPONENTS_SINGLE_FLOAT, &[1.0f32.to_bits(), 1.0f32.to_bits(), 1.0f32.to_bits(), 1.0f32.to_bits()]);
    b
}

/// Window coordinate to clip coordinate for `gl_window`'s 64-pixel viewport.
fn ndc(w: f32) -> u32 {
    (w / 32.0 - 1.0).to_bits()
}

fn vertices(b: &mut KonaBackend, mode: u32, points: &[(f32, f32)]) {
    command(b, Grammar::Gl, mode, &[0]);
    for &(x, y) in points {
        command(b, Grammar::Gl, SUBMIT_VERTEX_3_COMPONENTS_SINGLE_FLOAT, &[ndc(x), ndc(y), 0.0f32.to_bits()]);
    }
    command(b, Grammar::Gl, END_OPEN_PRIMITIVE, &[0]);
}

fn lit_pixels(b: &mut KonaBackend) -> Vec<(u32, u32)> {
    command(b, Grammar::Kernel, WINDOW_SYSTEM_PIXEL_PATH_FINISH, &[0]);
    (0..64).flat_map(|y| (0..64u32).map(move |x| (x, y))).filter(|&(x, y)| b.store.read(x, y, 2) != 0).collect()
}

/// A texture through the board: a 2 x 2 image loaded inline, bound by its short bind and
/// record, and drawn with the replace environment, shows its texels at nearest and
/// blends them at linear; with the texture off the draw is the vertex colour.
#[test]
fn a_loaded_and_bound_texture_shows_on_a_triangle() {
    use crate::dev::kona::defs::gl::TEXTURE_2D;
    let mut b = gl_window();
    let mut hdr = [0u32; 20];
    hdr[..4].copy_from_slice(&[0x0de1, 2, 2, 5]);
    hdr[4] = 1;
    hdr[9] = 0x30;
    hdr[13] = 1;
    hdr[14] = 1;
    hdr[16] = 0x11;
    hdr[17] = 0x2200;
    hdr[18] = 0x1000;
    let image = [0xff00_00ffu32, 0x00ff_00ff, 0x0000_ffff, 0xffff_ffff];
    assert_eq!(command(&mut b, Grammar::Gl, DRAW_TEXELS_INTO_TEXTURE_MEMORY, &hdr), Reply::Data(4));
    send(&mut b, Job::Data { grammar: Grammar::Gl, selector: DRAW_TEXELS_INTO_TEXTURE_MEMORY, args: &hdr, words: &image });
    command(&mut b, Grammar::Gl, TEXTURE_LOAD_BY_IDENTIFIER, &[0]);
    let mut rec = [0u32; 22];
    rec[0] = 0x44;
    rec[1] = 0x1010;
    rec[6] = 0x1000;
    rec[7] = 0xffff_ffff;
    let bind = |b: &mut KonaBackend, mag: u32| {
        command(b, Grammar::Gl, BIND_TEXTURE_CURRENT_FORM, &[0xffff_ffff, 2 << 12, mag << 5, 0x0011, 1]);
        command(b, Grammar::Gl, BIND_2D_MIP_MAPPED_TEXTURE, &rec);
    };
    bind(&mut b, 0);
    command(&mut b, Grammar::Gl, SET_TEXTURE_ENVIRONMENT_PARAMETER_FLOAT_FORM, &[0x2200, (0x1e01 as f32).to_bits(), 0, 0, 0]);
    command(&mut b, Grammar::Gl, ENABLE_CAPABILITY, &[TEXTURE_2D]);
    let quad = |b: &mut KonaBackend| {
        command(b, Grammar::Gl, BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES, &[0]);
        for (x, y, s, t) in [(0.0, 0.0, 0.0, 0.0), (64.0, 0.0, 1.0, 0.0), (0.0, 64.0, 0.0, 1.0), (64.0, 0.0, 1.0, 0.0), (64.0, 64.0, 1.0, 1.0), (0.0, 64.0, 0.0, 1.0)] {
            command(b, Grammar::Gl, SET_CURRENT_TEXTURE_COORDINATE_2_COMPONENTS_SINGLE_FLOAT, &[(s as f32).to_bits(), (t as f32).to_bits()]);
            command(b, Grammar::Gl, SUBMIT_VERTEX_3_COMPONENTS_SINGLE_FLOAT, &[ndc(x), ndc(y), 0.0f32.to_bits()]);
        }
        command(b, Grammar::Gl, END_OPEN_PRIMITIVE, &[0]);
    };
    quad(&mut b);
    command(&mut b, Grammar::Kernel, WINDOW_SYSTEM_PIXEL_PATH_FINISH, &[0]);
    let at = |b: &KonaBackend, x, y| b.store.read(x, y, 2);
    assert_eq!([at(&b, 10, 10), at(&b, 50, 10), at(&b, 10, 50), at(&b, 50, 50)], [0xffff_0000, 0xff00_ff00, 0xff00_00ff, 0xffff_ffff], "nearest: each quadrant its texel");
    // Linear magnification blends the four at the middle.
    bind(&mut b, 1);
    quad(&mut b);
    command(&mut b, Grammar::Kernel, WINDOW_SYSTEM_PIXEL_PATH_FINISH, &[0]);
    let mid = at(&b, 32, 32);
    assert!([mid >> 16 & 0xff, mid >> 8 & 0xff, mid & 0xff].iter().all(|&c| (100..160).contains(&c)), "{mid:#x}: red, green, blue and white average to grey");
    // Off, the draw is the vertex colour (white).
    command(&mut b, Grammar::Gl, DISABLE_CAPABILITY, &[TEXTURE_2D]);
    quad(&mut b);
    command(&mut b, Grammar::Kernel, WINDOW_SYSTEM_PIXEL_PATH_FINISH, &[0]);
    assert_eq!(at(&b, 10, 10), 0xffff_ffff);
    let text = run(&mut b, &["status"]).1;
    assert!(text.contains("1 level loads") && text.contains("2 binds (0 refused)"), "{text}");
}

/// A volume through the board: two 2 x 2 slices loaded one command each, bound by the short bind and the
/// single-level volume record, and drawn with the replace environment under the 3D enable. The texture
/// coordinate's r picks the slice; with the 3D enable off, the draw is the vertex colour.
#[test]
fn a_loaded_and_bound_volume_shows_the_slice_r_names() {
    use crate::dev::kona::defs::gl::TEXTURE_3D;
    let mut b = gl_window();
    for (z, texel) in [(0u32, 0xff00_00ffu32), (1, 0x0000_ffff)] {
        let mut hdr = [0u32; 20];
        hdr[..4].copy_from_slice(&[0x806f, 2, 2, 5]);
        hdr[4] = 1;
        hdr[9] = 0x30;
        hdr[13] = 1;
        hdr[14] = 1;
        hdr[15] = z;
        hdr[16] = 0x111;
        hdr[17] = 0x2200;
        hdr[18] = 0x1000;
        assert_eq!(command(&mut b, Grammar::Gl, DRAW_TEXELS_INTO_TEXTURE_MEMORY, &hdr), Reply::Data(4));
        send(&mut b, Job::Data { grammar: Grammar::Gl, selector: DRAW_TEXELS_INTO_TEXTURE_MEMORY, args: &hdr, words: &[texel; 4] });
    }
    command(&mut b, Grammar::Gl, TEXTURE_LOAD_BY_IDENTIFIER, &[0]);
    command(&mut b, Grammar::Gl, BIND_TEXTURE_CURRENT_FORM, &[0xffff_ffff, 3 << 28 | 2 << 12, 0, 0x0111, 1]);
    command(&mut b, Grammar::Gl, BIND_3D_TEXTURE, &[0x44, 0x111, 0x1000, 0]);
    command(&mut b, Grammar::Gl, SET_TEXTURE_ENVIRONMENT_PARAMETER_FLOAT_FORM, &[0x2200, (0x1e01 as f32).to_bits(), 0, 0, 0]);
    command(&mut b, Grammar::Gl, ENABLE_CAPABILITY, &[TEXTURE_3D]);
    let quad = |b: &mut KonaBackend, r: f32, x0: f32| {
        command(b, Grammar::Gl, BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES, &[0]);
        for (x, y) in [(0.0, 0.0), (32.0, 0.0), (0.0, 32.0), (32.0, 0.0), (32.0, 32.0), (0.0, 32.0)] {
            command(b, Grammar::Gl, SET_CURRENT_TEXTURE_COORDINATE_3_COMPONENTS_SINGLE_FLOAT, &[(x / 32.0f32).to_bits(), (y / 32.0f32).to_bits(), r.to_bits()]);
            command(b, Grammar::Gl, SUBMIT_VERTEX_3_COMPONENTS_SINGLE_FLOAT, &[ndc(x + x0), ndc(y), 0.0f32.to_bits()]);
        }
        command(b, Grammar::Gl, END_OPEN_PRIMITIVE, &[0]);
    };
    quad(&mut b, 0.25, 0.0);
    quad(&mut b, 0.75, 32.0);
    command(&mut b, Grammar::Kernel, WINDOW_SYSTEM_PIXEL_PATH_FINISH, &[0]);
    let at = |b: &KonaBackend, x, y| b.store.read(x, y, 2);
    assert_eq!([at(&b, 8, 8), at(&b, 24, 24), at(&b, 40, 8), at(&b, 56, 24)], [0xffff_0000, 0xffff_0000, 0xff00_00ff, 0xff00_00ff], "slice 0 where r is a quarter, slice 1 at three quarters");
    command(&mut b, Grammar::Gl, DISABLE_CAPABILITY, &[TEXTURE_3D]);
    quad(&mut b, 0.25, 0.0);
    command(&mut b, Grammar::Kernel, WINDOW_SYSTEM_PIXEL_PATH_FINISH, &[0]);
    assert_eq!(at(&b, 8, 8), 0xffff_ffff);
    let text = run(&mut b, &["status"]).1;
    assert!(text.contains("1 binds (0 refused)") && text.contains("2 volume slice loads, 1 volume binds"), "{text}");
}

/// A detail bind (token 0x17d) and a filter-coefficient table (token 0x17e and its 33 words) are kept and counted,
/// and neither refuses a bind nor draws a flat colour: the base texture stays what is bound.
#[test]
fn a_detail_bind_and_a_filter_table_are_kept_and_change_nothing_drawn() {
    let mut b = gl_window();
    command(&mut b, Grammar::Gl, BIND_DETAIL_TEXTURE, &[0xffff_ffff, 3, 2, 0x2, 0x400, 0x400]);
    command(&mut b, Grammar::Gl, LOAD_ONE_ENTRY_OF_ARM_TEXTURE_TABLE, &[1]);
    send(&mut b, Job::Data { grammar: Grammar::Gl, selector: LOAD_ONE_ENTRY_OF_ARM_TEXTURE_TABLE, args: &[1], words: &[0; 33] });
    assert_eq!((b.tex_counts.detail_binds, b.tex_counts.filter_tables), (1, 1));
    assert_eq!((b.tex_counts.unmodelled, b.tex_counts.bind_refused), (0, 0));
    assert!(run(&mut b, &["status"]).1.contains("detail            1 binds, 1 filter tables"));
}

/// A perspective-divided textured quad samples with the divided coordinate. The right edge is twice as far (w = 2) as
/// the left, so the screen's middle column is a third of the way across the texture, not half: it shows the first texel
/// of a 2 x 2 image at nearest, where the affine coordinate would show the second.
#[test]
fn a_perspective_textured_quad_samples_with_the_divided_coordinate() {
    use crate::dev::kona::defs::gl::TEXTURE_2D;
    let mut b = gl_window();
    let mut hdr = [0u32; 20];
    hdr[..4].copy_from_slice(&[0x0de1, 2, 2, 5]);
    hdr[4] = 1;
    hdr[9] = 0x30;
    hdr[13] = 1;
    hdr[14] = 1;
    hdr[16] = 0x11;
    hdr[17] = 0x2200;
    hdr[18] = 0x1000;
    let image = [0xff00_00ffu32, 0x00ff_00ff, 0x0000_ffff, 0xffff_ffff];
    command(&mut b, Grammar::Gl, DRAW_TEXELS_INTO_TEXTURE_MEMORY, &hdr);
    send(&mut b, Job::Data { grammar: Grammar::Gl, selector: DRAW_TEXELS_INTO_TEXTURE_MEMORY, args: &hdr, words: &image });
    command(&mut b, Grammar::Gl, TEXTURE_LOAD_BY_IDENTIFIER, &[0]);
    let mut rec = [0u32; 22];
    rec[0] = 0x44;
    rec[1] = 0x1010;
    rec[6] = 0x1000;
    rec[7] = 0xffff_ffff;
    command(&mut b, Grammar::Gl, BIND_TEXTURE_CURRENT_FORM, &[0xffff_ffff, 2 << 12, 0, 0x0011, 1]);
    command(&mut b, Grammar::Gl, BIND_2D_MIP_MAPPED_TEXTURE, &rec);
    command(&mut b, Grammar::Gl, SET_TEXTURE_ENVIRONMENT_PARAMETER_FLOAT_FORM, &[0x2200, (0x1e01 as f32).to_bits(), 0, 0, 0]);
    command(&mut b, Grammar::Gl, ENABLE_CAPABILITY, &[TEXTURE_2D]);
    // Clip coordinates: x = +-1 times w, so the screen's x is 0 and 64 whatever w is.
    let corner = |b: &mut KonaBackend, xn: f32, yn: f32, w: f32, s: f32| {
        command(b, Grammar::Gl, SET_CURRENT_TEXTURE_COORDINATE_2_COMPONENTS_SINGLE_FLOAT, &[s.to_bits(), 0.25f32.to_bits()]);
        command(b, Grammar::Gl, SUBMIT_VERTEX_4_COMPONENTS_SINGLE_FLOAT, &[(xn * w).to_bits(), (yn * w).to_bits(), 0.0f32.to_bits(), w.to_bits()]);
    };
    command(&mut b, Grammar::Gl, BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES, &[0]);
    for (xn, yn, w, s) in [(-1.0, -0.5, 1.0, 0.0), (1.0, -0.5, 2.0, 1.0), (1.0, 0.5, 2.0, 1.0), (-1.0, -0.5, 1.0, 0.0), (1.0, 0.5, 2.0, 1.0), (-1.0, 0.5, 1.0, 0.0)] {
        corner(&mut b, xn, yn, w, s);
    }
    command(&mut b, Grammar::Gl, END_OPEN_PRIMITIVE, &[0]);
    command(&mut b, Grammar::Kernel, WINDOW_SYSTEM_PIXEL_PATH_FINISH, &[0]);
    let at = |b: &KonaBackend, x| b.store.read(x, 32, 2);
    // The middle is a third across (red); the affine half would be green. Near the right edge, s nears 1 (green).
    assert_eq!(at(&b, 32), 0xffff_0000, "the divided coordinate: a third of the way across");
    assert_eq!(at(&b, 8), 0xffff_0000);
    assert_eq!(at(&b, 60), 0xff00_ff00, "near the far edge");
    // The coordinate is half at two thirds of the screen: s = 0.5 t / (1 - 0.5 t), not at the middle.
    let first_green = (0..64).find(|&x| at(&b, x) == 0xff00_ff00).expect("some green");
    assert!((40..46).contains(&first_green), "the texel edge is at {first_green}, two thirds of the screen, not the middle");
}

/// A point of one pixel lights exactly the pixel its vertex is in, even on its
/// edge; one larger than `POINT_PIXEL_MAX` is a disc of about its area.
#[test]
fn a_point_lights_one_pixel_or_a_disc() {
    for (x, y, want) in [(10.5, 10.5, (10, 10)), (20.0, 30.0, (20, 30)), (7.9, 3.0, (7, 3))] {
        let mut b = gl_window();
        vertices(&mut b, BEGIN_PRIMITIVE_POINTS, &[(x, y)]);
        assert_eq!(lit_pixels(&mut b), [want], "a point at ({x}, {y})");
    }
    let mut b = gl_window();
    command(&mut b, Grammar::Gl, SET_POINT_SIZE, &[3.0f32.to_bits(), 0]);
    vertices(&mut b, BEGIN_PRIMITIVE_POINTS, &[(10.5, 10.5)]);
    assert_eq!(lit_pixels(&mut b), [(10, 10)], "size 3 is still one pixel");
    let mut b = gl_window();
    command(&mut b, Grammar::Gl, SET_POINT_SIZE, &[8.0f32.to_bits(), 0]);
    vertices(&mut b, BEGIN_PRIMITIVE_POINTS, &[(32.0, 32.0)]);
    let lit = lit_pixels(&mut b);
    assert!((44..=56).contains(&lit.len()), "a disc of diameter 8 holds about 50 pixels: {}", lit.len());
    assert!(lit.iter().all(|&(x, y)| (27..=36).contains(&x) && (27..=36).contains(&y)));
}

/// A line of width 1 lights one row, from just past its first end to its last pixel
/// (the fill rule leaves the left edge out); a wide line lights its width.
#[test]
fn a_line_lights_its_width() {
    let mut b = gl_window();
    vertices(&mut b, BEGIN_PRIMITIVE_INDEPENDENT_LINES, &[(10.5, 20.5), (40.5, 20.5)]);
    let lit = lit_pixels(&mut b);
    assert!(lit.iter().all(|&(_, y)| y == 20), "one row");
    assert_eq!((lit.first().map(|p| p.0), lit.last().map(|p| p.0), lit.len()), (Some(11), Some(40), 30));
    let mut b = gl_window();
    command(&mut b, Grammar::Gl, SET_LINE_WIDTH, &[3.0f32.to_bits(), 0]);
    vertices(&mut b, BEGIN_PRIMITIVE_INDEPENDENT_LINES, &[(10.5, 20.5), (40.5, 20.5)]);
    let rows: std::collections::BTreeSet<u32> = lit_pixels(&mut b).iter().map(|p| p.1).collect();
    assert_eq!(rows, [19, 20, 21].into(), "width 3");
}

/// The alpha a pixel was written with: smoothing scales it, and with no blending the colour is
/// the vertex's, so alpha reads the coverage back.
fn alpha_at(b: &KonaBackend, x: u32, y: u32) -> u32 {
    b.store.read(x, y, 2) >> 24
}

/// A smoothed line of width 1 centred on a row is that row at full coverage, with each end a half
/// pixel in (its end passes through the end pixels' centres); moved half a pixel up it shares two
/// rows evenly.
#[test]
fn a_smoothed_line_ramps_its_ends_and_shares_a_row_boundary() {
    use crate::dev::kona::defs::gl::LINE_SMOOTH;
    let mut b = gl_window();
    command(&mut b, Grammar::Gl, ENABLE_CAPABILITY, &[LINE_SMOOTH]);
    vertices(&mut b, BEGIN_PRIMITIVE_INDEPENDENT_LINES, &[(10.5, 20.5), (40.5, 20.5)]);
    let lit = lit_pixels(&mut b);
    assert!(lit.iter().all(|&(_, y)| y == 20), "one row: {lit:?}");
    assert_eq!((lit.first().map(|p| p.0), lit.last().map(|p| p.0), lit.len()), (Some(10), Some(40), 31), "the end pixels, half covered, are drawn too");
    assert_eq!([alpha_at(&b, 10, 20), alpha_at(&b, 11, 20), alpha_at(&b, 25, 20), alpha_at(&b, 39, 20), alpha_at(&b, 40, 20)], [128, 255, 255, 255, 128]);
    let mut b = gl_window();
    command(&mut b, Grammar::Gl, ENABLE_CAPABILITY, &[LINE_SMOOTH]);
    vertices(&mut b, BEGIN_PRIMITIVE_INDEPENDENT_LINES, &[(10.5, 21.0), (40.5, 21.0)]);
    lit_pixels(&mut b);
    assert_eq!([alpha_at(&b, 25, 19), alpha_at(&b, 25, 20), alpha_at(&b, 25, 21), alpha_at(&b, 25, 22)], [0, 128, 128, 0]);
    // Off, the same line is a row of whole pixels from the second.
    let mut b = gl_window();
    vertices(&mut b, BEGIN_PRIMITIVE_INDEPENDENT_LINES, &[(10.5, 21.0), (40.5, 21.0)]);
    let lit = lit_pixels(&mut b);
    assert!(lit.iter().all(|&(_, y)| y == 21) && lit.iter().all(|&(x, y)| alpha_at(&b, x, y) == 255));
}

/// A diagonal smoothed line through pixel centres covers its own pixel by a little under a whole and
/// its two neighbours across by about a quarter each: a strip one pixel wide at 45 degrees takes
/// 0.914 of its pixel and 0.25 of each neighbour, 1.414 in all, the line's width times the length
/// of a diagonal step. Set-up snaps the rectangle's corners to eighths of a pixel, which widens it
/// a little, so the figures here are a little above those.
#[test]
fn a_smoothed_diagonal_line_shares_its_width_across_the_pixels_it_touches() {
    use crate::dev::kona::defs::gl::LINE_SMOOTH;
    let mut b = gl_window();
    command(&mut b, Grammar::Gl, ENABLE_CAPABILITY, &[LINE_SMOOTH]);
    vertices(&mut b, BEGIN_PRIMITIVE_INDEPENDENT_LINES, &[(10.5, 10.5), (40.5, 40.5)]);
    lit_pixels(&mut b);
    let a = |dx: u32, dy: u32| alpha_at(&b, dx, dy);
    assert!((235..=250).contains(&a(25, 25)), "{}", a(25, 25));
    assert_eq!((a(26, 25), a(25, 26)), (a(24, 25), a(25, 24)), "the neighbours either side are alike");
    assert!((60..=80).contains(&a(26, 25)), "{}", a(26, 25));
    let sum = (a(25, 25) + a(26, 25) + a(24, 25)) as f32 / 255.0;
    assert!((1.35..=1.6).contains(&sum), "{sum}");
    assert_eq!(a(27, 25), 0, "and none past them");
}

/// A smoothed polygon ramps its outer edges and not the seam of the fan it is drawn as: a square
/// from pixel centre to pixel centre covers its edge pixels by half and its corners by a quarter,
/// and the pixels along its diagonal are whole.
#[test]
fn a_smoothed_polygon_ramps_its_outline_and_not_its_seam() {
    use crate::dev::kona::defs::gl::POLYGON_SMOOTH;
    let square = [(10.5, 10.5), (40.5, 10.5), (40.5, 40.5), (10.5, 40.5)];
    let mut b = gl_window();
    command(&mut b, Grammar::Gl, ENABLE_CAPABILITY, &[POLYGON_SMOOTH]);
    vertices(&mut b, BEGIN_PRIMITIVE_POLYGON, &square);
    lit_pixels(&mut b);
    let a = |x, y| alpha_at(&b, x, y);
    assert_eq!([a(10, 25), a(25, 10), a(40, 25), a(25, 40)], [128; 4], "edge pixels, half covered");
    assert_eq!([a(10, 10), a(40, 10), a(40, 40), a(10, 40)], [64; 4], "corners, a quarter");
    assert_eq!([a(11, 11), a(25, 25), a(39, 39), a(20, 30)], [255; 4], "inside, seam included, whole");
    assert_eq!([a(9, 25), a(41, 25), a(25, 9), a(25, 41)], [0; 4], "and nothing past the ramp");
    // Off, the same polygon is whole pixels only.
    let mut b = gl_window();
    vertices(&mut b, BEGIN_PRIMITIVE_POLYGON, &square);
    let lit = lit_pixels(&mut b);
    assert!(lit.iter().all(|&(x, y)| alpha_at(&b, x, y) == 255));
    // A triangle's slanted edge covers its pixels in part, and the pixels a plain triangle lights
    // keep at least half a pixel's cover on their outer edge.
    let mut b = gl_window();
    command(&mut b, Grammar::Gl, ENABLE_CAPABILITY, &[POLYGON_SMOOTH]);
    vertices(&mut b, BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES, &[(10.0, 10.0), (50.0, 10.0), (10.0, 50.0)]);
    lit_pixels(&mut b);
    let mid = alpha_at(&b, 30, 29);
    assert!((1..255).contains(&mid), "a pixel on the hypotenuse is partly covered: {mid}");
}

/// A smoothed point covers the share of each pixel its disc does: the pixel about a size-1 point's
/// centre about pi / 4, a size-3 point's pixels symmetric about it and summing to the disc's area.
#[test]
fn a_smoothed_point_covers_by_its_disc() {
    use crate::dev::kona::defs::gl::POINT_SMOOTH;
    let mut b = gl_window();
    command(&mut b, Grammar::Gl, ENABLE_CAPABILITY, &[POINT_SMOOTH]);
    vertices(&mut b, BEGIN_PRIMITIVE_POINTS, &[(20.5, 20.5)]);
    let lit = lit_pixels(&mut b);
    assert_eq!(lit, [(20, 20)]);
    // pi / 4 of full, from an 8 x 8 grid of sample positions: within a few percent.
    assert!((195..=212).contains(&alpha_at(&b, 20, 20)), "{}", alpha_at(&b, 20, 20));
    let mut b = gl_window();
    command(&mut b, Grammar::Gl, ENABLE_CAPABILITY, &[POINT_SMOOTH]);
    command(&mut b, Grammar::Gl, SET_POINT_SIZE, &[3.0f32.to_bits(), 0]);
    vertices(&mut b, BEGIN_PRIMITIVE_POINTS, &[(20.5, 20.5)]);
    lit_pixels(&mut b);
    let a = |dx: i32, dy: i32| alpha_at(&b, (20 + dx) as u32, (20 + dy) as u32);
    assert_eq!(a(0, 0), 255);
    assert!(a(-1, 0) == a(1, 0) && a(0, -1) == a(0, 1) && a(1, 0) == a(0, 1), "the four edge neighbours alike");
    assert!(a(1, 1) == a(-1, -1) && a(1, 1) < a(1, 0), "the corners, alike, are less than the edges");
    let sum: u32 = (-2..=2).flat_map(|dy| (-2..=2).map(move |dx| (dx, dy))).map(|(dx, dy)| a(dx, dy)).sum();
    let area = std::f32::consts::PI * 1.5 * 1.5;
    assert!((sum as f32 / 255.0 - area).abs() < 0.2, "coverage sums to {} against the disc's {area}", sum as f32 / 255.0);
}

/// The line stipple indexes its 16-bit pattern by the distance along the line over
/// the repeat count; the polygon stipple, 32 x 32 bits, by window position.
#[test]
fn stipples_keep_the_pixels_their_patterns_name() {
    use crate::dev::kona::defs::gl::{LINE_STIPPLE, POLYGON_STIPPLE};
    let mut b = gl_window();
    command(&mut b, Grammar::Gl, ENABLE_CAPABILITY, &[LINE_STIPPLE]);
    command(&mut b, Grammar::Gl, SET_LINE_STIPPLE, &[1, 0x00ff]);
    vertices(&mut b, BEGIN_PRIMITIVE_INDEPENDENT_LINES, &[(10.5, 20.5), (40.5, 20.5)]);
    let xs: Vec<u32> = lit_pixels(&mut b).iter().map(|p| p.0).collect();
    let want: Vec<u32> = (11..=40).filter(|x| (x - 10) % 16 < 8).collect();
    assert_eq!(xs, want, "the low eight bits of the pattern: eight pixels on, eight off");
    // A repeat of 2 stretches each bit over two pixels.
    let mut b = gl_window();
    command(&mut b, Grammar::Gl, ENABLE_CAPABILITY, &[LINE_STIPPLE]);
    command(&mut b, Grammar::Gl, SET_LINE_STIPPLE, &[2, 0x00ff]);
    vertices(&mut b, BEGIN_PRIMITIVE_INDEPENDENT_LINES, &[(10.5, 20.5), (50.5, 20.5)]);
    let xs: Vec<u32> = lit_pixels(&mut b).iter().map(|p| p.0).collect();
    let want: Vec<u32> = (11..=50).filter(|x| (x - 10) / 2 % 16 < 8).collect();
    assert_eq!(xs, want);
    // Polygon stipple: a checkerboard, row 0 starting set at the leftmost column.
    let mut b = gl_window();
    let mut rows = [0xaaaa_aaaau32; 32];
    rows.iter_mut().skip(1).step_by(2).for_each(|r| *r = 0x5555_5555);
    assert_eq!(command(&mut b, Grammar::Gl, LOAD_POLYGON_STIPPLE_PATTERN, &[0, 0, 0]), Reply::Data(32));
    send(&mut b, Job::Data { grammar: Grammar::Gl, selector: LOAD_POLYGON_STIPPLE_PATTERN, args: &[0, 0, 0], words: &rows });
    command(&mut b, Grammar::Gl, ENABLE_CAPABILITY, &[POLYGON_STIPPLE]);
    vertices(&mut b, BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES, &[(0.5, 0.5), (60.5, 0.5), (0.5, 60.5)]);
    let lit = lit_pixels(&mut b);
    assert!(lit.len() > 500 && lit.iter().all(|&(x, y)| (x + y) % 2 == 0), "a checkerboard of {} pixels", lit.len());
    // With the stipple off the same triangle is solid.
    let mut b = gl_window();
    send(&mut b, Job::Data { grammar: Grammar::Gl, selector: LOAD_POLYGON_STIPPLE_PATTERN, args: &[0, 0, 0], words: &rows });
    vertices(&mut b, BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES, &[(0.5, 0.5), (60.5, 0.5), (0.5, 60.5)]);
    assert!(lit_pixels(&mut b).iter().any(|&(x, y)| (x + y) % 2 == 1));
}

/// A window whose display ID is a 12-bit colour index (the entry and mode a client such as an
/// operating-system monitor makes) takes the GL path's colour indices, not its colours: a clear
/// to index 8 and a rectangle in index 6 leave the words the scan-out reads as 12-bit indices
/// (bits 15:4), and the colour map the X server loaded turns them into colours. With the
/// index commands dropped, as they were, the clear and the rectangle drew the default white,
/// index 4095, an entry nobody loaded, and the window showed black.
#[test]
fn a_colour_index_window_is_drawn_in_indices() {
    use crate::dev::kona::back::display::compose;
    use crate::dev::kona::back::store::DID_PLANE;
    use crate::dev::kona::defs::gl::{COLOR_BUFFER_BIT, FRONT};
    let mut b = KonaBackend::new(KonaConfig::default());
    // The entry names 12 index planes and the colour map from 0x4800; the mode byte says colour index.
    command(&mut b, Grammar::Kernel, LOAD_ONE_ENTRY_OF_DISPLAY_ID_TABLE, &[14, 0x24005c]);
    command(&mut b, Grammar::Kernel, DISPLAY_MODE_BYTE, &[0, 14, 8]);
    command(&mut b, Grammar::Kernel, WINDOW_CLIP_SETUP, &[0, 0, 64, 64, 0, 0, 64, 64, 0, 14, 0]);
    command(&mut b, Grammar::Gl, CONFIGURE_FRAME_BUFFER_LAYOUT, &[0, 8, 0, 1, 0, 0, 24, 0, 0, 0]);
    command(&mut b, Grammar::Gl, SET_VIEWPORT, &[0, 0, 64, 64]);
    command(&mut b, Grammar::Gl, SELECT_DRAW_BUFFER, &[FRONT]);
    command(&mut b, Grammar::Gl, SET_COLOUR_INDEX_CLEAR_VALUE, &[8.0f32.to_bits()]);
    command(&mut b, Grammar::Gl, CLEAR_SELECTED_BUFFERS, &[COLOR_BUFFER_BIT]);
    // The short form carries its value in the top half of the argument word.
    command(&mut b, Grammar::Gl, SET_CURRENT_COLOUR_INDEX_SIGNED_SHORT, &[6 << 16]);
    // A rectangle is four vertices, as a quad is.
    command(&mut b, Grammar::Gl, BEGIN_PRIMITIVE_RECTANGLE, &[0xe]);
    for (x, y) in [(10.5, 10.5), (30.5, 10.5), (30.5, 30.5), (10.5, 30.5)] {
        command(&mut b, Grammar::Gl, SUBMIT_VERTEX_2_COMPONENTS_SINGLE_FLOAT, &[ndc(x), ndc(y)]);
    }
    command(&mut b, Grammar::Gl, END_OPEN_PRIMITIVE, &[0]);
    // A line loop in index 7, and a float-form index.
    command(&mut b, Grammar::Gl, SET_CURRENT_COLOUR_INDEX_SINGLE_FLOAT, &[7.0f32.to_bits()]);
    vertices(&mut b, BEGIN_PRIMITIVE_LINE_LOOP, &[(40.5, 40.5), (60.5, 40.5), (60.5, 60.5), (40.5, 60.5)]);
    command(&mut b, Grammar::Kernel, WINDOW_SYSTEM_PIXEL_PATH_FINISH, &[0]);
    assert_eq!(b.store.read(5, 5, 1), 8 << 4, "cleared to index 8");
    assert_eq!(b.store.read(20, 20, 1), 6 << 4, "the rectangle is index 6");
    assert_eq!(b.store.read(50, 40, 1), 7 << 4, "the line is index 7");
    assert_eq!(b.store.read(5, 5, 2), 0, "the back buffer is untouched");
    // The map the X server loaded for this window: index 6 is red, 8 is blue.
    b.tables.lock().load_colours(0x4800 + 6, &[0x3ff << 20]);
    b.tables.lock().load_colours(0x4800 + 8, &[0x3ff]);
    for &(x, y) in &[(5u32, 5u32), (20, 20)] {
        b.store.write(x, y, DID_PLANE, 14 << 2);
    }
    let mut out = vec![0u32; 1280 * 1024];
    compose(&b.store, 1, &b.map.lock(), &b.tables.lock().snapshot(), &mut out, 1280);
    let shown = |x: usize, y: usize| out[(1023 - y) * 1280 + x];
    let map = b.tables.lock().snapshot();
    assert_eq!(shown(20, 20), map.index12(0x4800)[6]);
    assert_eq!(shown(5, 5), map.index12(0x4800)[8]);
    assert_ne!(shown(20, 20), shown(5, 5));
    assert_ne!(shown(20, 20), 0xff00_0000, "not the black an unloaded entry gives");
    // A bitmap takes the raster position's index too.
    command(&mut b, Grammar::Gl, SET_CURRENT_COLOUR_INDEX_SIGNED_SHORT, &[3 << 16]);
    command(&mut b, Grammar::Gl, SET_RASTER_POSITION_2_COMPONENTS_FLOAT, &[ndc(8.5), ndc(50.5)]);
    let args = [8, 1, 0, 0, 8.0f32.to_bits(), 0, 2, 1];
    assert_eq!(command(&mut b, Grammar::Gl, DRAW_BITMAP_FROM_DISPLAY_LIST, &args), Reply::Data(1));
    send(&mut b, Job::Data { grammar: Grammar::Gl, selector: DRAW_BITMAP_FROM_DISPLAY_LIST, args: &args, words: &[0x8000_0000] });
    command(&mut b, Grammar::Kernel, WINDOW_SYSTEM_PIXEL_PATH_FINISH, &[0]);
    assert_eq!(b.store.read(8, 50, 1), 3 << 4, "the bitmap's fragment is index 3");
}

/// A GL triangle, through the board: the kernel places the window and the
/// buffer configuration lays the planes out; the triangle goes to the back
/// buffer's word, the display shows nothing of it until a swap names the
/// window's id with the buffer bit, and then shows it as direct colour.
#[test]
fn a_gl_triangle_draws_into_the_back_buffer_and_shows_after_the_swap() {
    use crate::dev::kona::back::display::compose;
    use crate::dev::kona::back::store::DID_PLANE;
    let mut b = KonaBackend::new(KonaConfig::default());
    let f = |v: f32| v.to_bits();
    // Window at (10, 20), 50 x 40, display id 13, not partly covered.
    command(&mut b, Grammar::Kernel, WINDOW_CLIP_SETUP, &[10, 20, 50, 40, 10, 20, 50, 40, 0, 13, 0]);
    command(&mut b, Grammar::Gl, 1, &[0, 8, 0, 1, 0, 0, 24, 0, 0, 0]);
    command(&mut b, Grammar::Gl, SET_VIEWPORT, &[0, 0, 50, 40]);
    command(&mut b, Grammar::Gl, SET_CURRENT_COLOUR_4_COMPONENTS_SINGLE_FLOAT, &[f(1.0), f(0.0), f(0.0), f(1.0)]);
    command(&mut b, Grammar::Gl, BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES, &[0]);
    for (x, y) in [(-1.0, -1.0), (1.0, -1.0), (0.0, 1.0)] {
        command(&mut b, Grammar::Gl, SUBMIT_VERTEX_3_COMPONENTS_SINGLE_FLOAT, &[f(x), f(y), f(0.0)]);
    }
    command(&mut b, Grammar::Gl, END_OPEN_PRIMITIVE, &[0]);
    // A job that touches pixels waits for the drawing first.
    command(&mut b, Grammar::Kernel, WINDOW_SYSTEM_PIXEL_PATH_FINISH, &[0]);
    let red = |w: usize| (0..1024).flat_map(|y| (0..1280u32).map(move |x| (x, y))).filter(|&(x, y)| b.store.read(x, y, w) == 0xffff_0000).count();
    assert!(red(1) == 0, "the front buffer is untouched");
    let back = red(2);
    assert!(back > 900 && back <= 1000, "about half of 50 x 40 pixels: {back}");
    assert!(b.store.read(30, 30, 2) == 0xffff_0000 && b.store.read(5, 30, 2) == 0, "inside the window and outside it");
    // The display: id 13 is direct colour, showing the back buffer once swapped.
    b.store.write(30, 30, DID_PLANE, 13 << 2);
    command(&mut b, Grammar::Kernel, DISPLAY_MODE_BYTE, &[0, 13, 4]);
    let shown = |b: &KonaBackend| {
        let mut out = vec![0u32; 1280 * 1024];
        compose(&b.store, 1, &b.map.lock(), &b.tables.lock().snapshot(), &mut out, 1280);
        out[(1023 - 30) * 1280 + 30]
    };
    assert_eq!(shown(&b), 0xff00_0000, "the front buffer is empty");
    command(&mut b, Grammar::Kernel, DISPLAY_MODE_BYTE, &[0x3c0001, 13, 1]);
    assert_eq!(shown(&b), 0xff00_00ff, "red, as 0xAABBGGRR, after the swap");
    let text = run(&mut b, &["status"]).1;
    assert!(text.contains("raster stage      1 triangles"), "{text}");
}

/// Each row of the drain table: what reads or writes the store drains the
/// GL stage, and what does not leaves it alone. Counted, so the test does not
/// depend on a worker being slow enough to lose a race.
#[test]
fn the_drain_table_row_by_row() {
    let mut b = KonaBackend::new(test_cfg().with_pool(2, 32));
    b.tiles();
    let drains = |b: &KonaBackend| b.tiles.as_ref().map_or(0, |t| t.stats().drains);
    let mut expect = 0;
    // Rows that do not drain.
    let quiet = |b: &mut KonaBackend, what: &str, job: Job<'_>| {
        send(b, job);
        assert_eq!(drains(b), expect, "{what} left the workers alone");
    };
    quiet(&mut b, "GL state", Job::Command { grammar: Grammar::Gl, selector: SET_VIEWPORT, args: &[0, 0, 50, 40], packed: 0, handled: false });
    quiet(&mut b, "GL colour", Job::Command { grammar: Grammar::Gl, selector: SET_CURRENT_COLOUR_4_COMPONENTS_SINGLE_FLOAT, args: &[1.0f32.to_bits(); 4], packed: 0, handled: false });
    quiet(&mut b, "GL begin", Job::Command { grammar: Grammar::Gl, selector: BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES, args: &[0], packed: 0, handled: false });
    quiet(&mut b, "GL end", Job::Command { grammar: Grammar::Gl, selector: END_OPEN_PRIMITIVE, args: &[0], packed: 0, handled: false });
    quiet(&mut b, "a GL clear", Job::Command { grammar: Grammar::Gl, selector: CLEAR_SELECTED_BUFFERS, args: &[0], packed: 0, handled: false });
    quiet(&mut b, "an X state load", Job::Command { grammar: Grammar::Kernel, selector: x::state::X_POSITION, args: &[0, 0, 8, 8], packed: 0, handled: false });
    quiet(&mut b, "an X data token", Job::Command { grammar: Grammar::Kernel, selector: x::fill::X_DATA2, args: &[0, 0], packed: 0, handled: false });
    quiet(&mut b, "a display-table load", Job::Command { grammar: Grammar::Kernel, selector: DISPLAY_MODE_BYTE, args: &[0, 13, 4], packed: 0, handled: false });
    quiet(&mut b, "an indirect store", Job::Indirect { selector: XMAP_CURSOR_MODE, value: 0 });
    quiet(&mut b, "a context event", Job::Context(super::super::job::ContextEvent::Activated { slot: 1, fresh: true }));
    quiet(&mut b, "dropped words", Job::Dropped { word: 7 });
    // Rows that drain, each once.
    let mut loud = |b: &mut KonaBackend, what: &str, job: Job<'_>| {
        send(b, job);
        expect += 1;
        assert_eq!(drains(b), expect, "{what} drains");
    };
    loud(&mut b, "a textport command", Job::Command { grammar: Grammar::Textport, selector: TEXTPORT_FILL_BOX, args: &[0, 0, 3, 3], packed: 0, handled: false });
    loud(&mut b, "an X box", Job::Command { grammar: Grammar::Kernel, selector: x::fill::X_BOX, args: &[0, 2 << 16 | 2], packed: 0, handled: false });
    loud(&mut b, "an X pixel finish", Job::Command { grammar: Grammar::Kernel, selector: x::pixel::X_PIXEL_FINISH, args: &[0], packed: 0, handled: false });
    loud(&mut b, "a swap", Job::Command { grammar: Grammar::Kernel, selector: DISPLAY_MODE_BYTE, args: &[0x3c0001, 13, 1], packed: 0, handled: false });
    loud(&mut b, "a region swap-out", Job::Command { grammar: Grammar::Kernel, selector: PIPE_MEMORY_SWAP_OUT, args: &[0; 8], packed: 0, handled: false });
    loud(&mut b, "a retrace request", Job::Retrace { tag: 1 });
    loud(&mut b, "a GL finish", Job::Command { grammar: Grammar::Gl, selector: FINISH_DRAIN_PIPE_AND_SIGNAL_COMPLETION, args: &[0], packed: 0, handled: false });
    b.quiesce();
    assert_eq!(drains(&b), expect + 1, "a quiesce drains");
    assert_eq!(b.tiles.as_ref().map_or(0, |t| t.stats().busy_drains), 0, "no worker had anything queued");
}

/// A raw 16-bit luminance level of 8 x 4 texels at `address` in bank 0, the words zero. A bind of it
/// under a colour format decodes it in place, which writes texture memory: a zero word becomes opaque black,
/// so a page that held nothing is made. `target` is 0x0de1 for a 2D level, 0x806f for a volume slice.
fn load_raw_level(b: &mut KonaBackend, target: u32, address: u32) {
    let mut hdr = [0u32; 20];
    hdr[..5].copy_from_slice(&[target, 8, 4, 6, 3]);
    hdr[9] = 0xd;
    hdr[13] = 7;
    hdr[14] = 3;
    hdr[16] = if target == 0x0de1 { 0x23 } else { 0x123 };
    hdr[17] = 0x2200;
    hdr[18] = address;
    assert_eq!(command(b, Grammar::Gl, DRAW_TEXELS_INTO_TEXTURE_MEMORY, &hdr), Reply::Data(16));
    send(b, Job::Data { grammar: Grammar::Gl, selector: DRAW_TEXELS_INTO_TEXTURE_MEMORY, args: &hdr, words: &[0; 16] });
}

/// Bind the level at `address` under the 5-6-5 colour format (0x18): the 2D form, the single-level volume form
/// or the mip-mapped volume form, by `selector`.
fn bind_level(b: &mut KonaBackend, selector: u32, address: u32) {
    let (modes, record): (_, Vec<u32>) = match selector {
        BIND_2D_MIP_MAPPED_TEXTURE => ([0xffff_ffff, 2 << 12, 0x18 << 14, 0x23, 1], [&[0x44, 0, 0, 0, 0, 0, address, 0xffff_ffff][..], &[0; 14]].concat()),
        BIND_3D_TEXTURE => ([0xffff_ffff, 3 << 28 | 2 << 12, 0x18 << 14, 0x0123, 1], vec![0x44, 0x123, address, 0]),
        _ => ([0xffff_ffff, 3 << 28 | 2 << 12, 0x18 << 14, 0x0123, 1], [&[0x44, 0x123, 0, 0, address][..], &[0xffff_ffff; 9]].concat()),
    };
    command(b, Grammar::Gl, BIND_TEXTURE_CURRENT_FORM, &modes);
    command(b, Grammar::Gl, selector, &record[..match selector { BIND_2D_MIP_MAPPED_TEXTURE => 22, BIND_3D_TEXTURE => 4, _ => 14 }]);
}

/// A bind that decodes a raw level writes texture memory, so the tile workers finish first; one that has
/// nothing to decode leaves them alone. Counted, so the test does not depend on a worker being slow enough
/// to lose a race. Each bind form is checked: the 2D mip-mapped, the single-level volume and the mip-mapped volume.
#[test]
fn a_bind_that_decodes_a_raw_level_drains_the_tiles_first() {
    let mut b = KonaBackend::new(test_cfg().with_pool(2, 32));
    b.tiles();
    let drains = |b: &KonaBackend| b.tiles.as_ref().map_or(0, |t| t.stats().drains);
    for (selector, target, address) in [(BIND_2D_MIP_MAPPED_TEXTURE, 0x0de1, 0x1000), (BIND_3D_TEXTURE, 0x806f, 0x2000), (BIND_3D_MIP_MAPPED_TEXTURE, 0x806f, 0x3000)] {
        load_raw_level(&mut b, target, address);
        assert!(b.contexts.select(None).tex.holds_raw());
        let before = drains(&b);
        bind_level(&mut b, selector, address);
        assert!(drains(&b) > before, "{selector:#x}: the bind wrote texture memory without waiting for the workers");
        assert!(!b.contexts.select(None).tex.holds_raw(), "{selector:#x}: decoded in place");
        let settled = drains(&b);
        bind_level(&mut b, selector, address);
        assert_eq!(drains(&b), settled, "{selector:#x}: a bind of a level already decoded leaves the workers alone");
    }
    let text = run(&mut b, &["status"]).1;
    assert!(text.contains("0 refused"), "{text}");
}

/// Workers sample one texture while binds decode raw levels into the same plane, each past the last so the
/// plane's page table grows. The binds wait for the workers, so the sampled texture is never disturbed.
#[test]
fn workers_sampling_a_texture_survive_binds_that_grow_the_store() {
    use crate::dev::kona::defs::gl::TEXTURE_2D;
    let mut b = KonaBackend::new(test_cfg().with_pool(2, 8));
    command(&mut b, Grammar::Kernel, WINDOW_CLIP_SETUP, &[0, 0, 64, 64, 0, 0, 64, 64, 0, 13, 0]);
    command(&mut b, Grammar::Gl, CONFIGURE_FRAME_BUFFER_LAYOUT, &[0, 8, 0, 1, 0, 0, 24, 0, 0, 0]);
    command(&mut b, Grammar::Gl, SET_VIEWPORT, &[0, 0, 64, 64]);
    command(&mut b, Grammar::Gl, SET_CURRENT_COLOUR_4_COMPONENTS_SINGLE_FLOAT, &[1.0f32.to_bits(); 4]);
    let mut hdr = [0u32; 20];
    hdr[..5].copy_from_slice(&[0x0de1, 2, 2, 5, 1]);
    hdr[9] = 0x30;
    hdr[13] = 1;
    hdr[14] = 1;
    hdr[16] = 0x11;
    hdr[17] = 0x2200;
    hdr[18] = 0x1000;
    assert_eq!(command(&mut b, Grammar::Gl, DRAW_TEXELS_INTO_TEXTURE_MEMORY, &hdr), Reply::Data(4));
    send(&mut b, Job::Data { grammar: Grammar::Gl, selector: DRAW_TEXELS_INTO_TEXTURE_MEMORY, args: &hdr, words: &[0xff00_00ff; 4] });
    command(&mut b, Grammar::Gl, SET_TEXTURE_ENVIRONMENT_PARAMETER_FLOAT_FORM, &[0x2200, (0x1e01 as f32).to_bits(), 0, 0, 0]);
    command(&mut b, Grammar::Gl, ENABLE_CAPABILITY, &[TEXTURE_2D]);
    let mut rec = [0u32; 22];
    rec[0] = 0x44;
    rec[1] = 0x1010;
    rec[6] = 0x1000;
    rec[7] = 0xffff_ffff;
    let quad = |b: &mut KonaBackend| {
        command(b, Grammar::Gl, BIND_TEXTURE_CURRENT_FORM, &[0xffff_ffff, 2 << 12, 0, 0x0011, 1]);
        command(b, Grammar::Gl, BIND_2D_MIP_MAPPED_TEXTURE, &rec);
        command(b, Grammar::Gl, BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES, &[0]);
        for (x, y, s, t) in [(0.0, 0.0, 0.0, 0.0), (64.0, 0.0, 1.0, 0.0), (0.0, 64.0, 0.0, 1.0), (64.0, 0.0, 1.0, 0.0), (64.0, 64.0, 1.0, 1.0), (0.0, 64.0, 0.0, 1.0)] {
            command(b, Grammar::Gl, SET_CURRENT_TEXTURE_COORDINATE_2_COMPONENTS_SINGLE_FLOAT, &[(s as f32).to_bits(), (t as f32).to_bits()]);
            command(b, Grammar::Gl, SUBMIT_VERTEX_3_COMPONENTS_SINGLE_FLOAT, &[ndc(x), ndc(y), 0.0f32.to_bits()]);
        }
        command(b, Grammar::Gl, END_OPEN_PRIMITIVE, &[0]);
    };
    let drains = |b: &KonaBackend| b.tiles.as_ref().map_or(0, |t| t.stats().drains);
    for i in 0..64u32 {
        // Each level is a page or more beyond the last, so the page table has to grow.
        let address = 0x4000 + i * 0x2000;
        for _ in 0..4 {
            quad(&mut b);
        }
        load_raw_level(&mut b, 0x0de1, address);
        let before = drains(&b);
        bind_level(&mut b, BIND_2D_MIP_MAPPED_TEXTURE, address);
        assert!(drains(&b) > before, "bind {i} wrote the store without waiting for the workers");
    }
    quad(&mut b);
    assert!(lit_pixels(&mut b).len() == 64 * 64, "every pixel drawn");
    assert!((0..64).all(|y| (0..64).all(|x| b.store.read(x, y, 2) == 0xffff_0000)), "every sample of the texture is its red texel");
}

/// The region commands are counted with the bytes the swaps name, and the
/// front end still moves the bytes.
#[test]
fn region_commands_are_counted() {
    let (mut b, mut r) = (backend(), Rig::new());
    r.x(&mut b, PIPE_MEMORY_RESERVE, &[2, 9, 0, 1]);
    r.x(&mut b, PIPE_MEMORY_COPY_DOWN, &[2, 0, 0, 0]);
    r.x(&mut b, PIPE_MEMORY_SWAP_OUT, &[3, 0, 4, 0x400, 1, 0, 0x9000, 0x102]);
    r.x(&mut b, PIPE_MEMORY_SWAP_IN, &[3, 0, 2, 0x400, 1, 0]);
    let text = run(&mut b, &["status"]).1;
    assert!(text.contains("1 placed, 1 copied down, 1 swapped out (4096 bytes), 1 swapped in (2048 bytes)"), "{text}");
}

/// A bracketed read whose host buffer is not mapped writes nothing, posts no
/// completion and is counted, as the recorder's model does.
#[test]
fn a_bracketed_read_into_an_unmapped_buffer_posts_nothing() {
    use crate::traits::{Device, BUS_OK};
    let (host, kona) = board();
    let w = |v: u32| assert_eq!(kona.test_write(0xc10000, 4, v as u64), BUS_OK);
    let words = |ws: &[u32]| ws.iter().for_each(|&v| w(v));
    let phys = |a: u64| 0x1000_0000 | (a >> 12) as u32;
    words(&[0x1f1, phys(REASON_PAGE), 0, 0x4000]);
    words(&[0x1fd, phys(0x50_0000), 0, 0x10000]);
    words(&[0x1f2, phys(RETURN_PAGE), 0, 0x4000]);
    words(&[0x129]);
    words(&INDEX_GC);
    words(&[0x154, 0, 2 << 16 | 1]);
    words(&[0x136, 2, 1, 0, 8, 2, 1, 0]);
    // Device page 0x1c1 has no map RAM entry.
    let dev = 0x1c1u64 << 14;
    words(&[0x1f5, (dev >> 12) as u32, 0x9000 | (dev & 0xfff) as u32, 2, 2, 0, 0]);
    words(&[0x138, 0, 0]);
    words(&[0x1f6, 0x100, 0x77]);
    assert_eq!(host.word(REASON_PAGE), 0, "no completion");
    let text = monitor(&kona, &["status"]).1;
    assert!(text.contains("read brackets   0 refused, 1 with no completion posted"), "{text}");
    kona.stop();
}

/// A read inside a select-feed bracket goes to the host buffer at the
/// bracket's finish, which then posts its completion with the second word
/// the finish carried. Through the board's own command port.
#[test]
fn a_bracketed_read_reaches_the_host_buffer() {
    use crate::traits::{Device, BUS_OK};
    let (host, kona) = board();
    let w = |v: u32| assert_eq!(kona.test_write(0xc10000, 4, v as u64), BUS_OK);
    let words = |ws: &[u32]| ws.iter().for_each(|&v| w(v));
    let phys = |a: u64| 0x1000_0000 | (a >> 12) as u32;
    words(&[0x1f1, phys(REASON_PAGE), 0, 0x4000]);
    words(&[0x1fd, phys(0x50_0000), 0, 0x10000]);
    words(&[0x1f2, phys(RETURN_PAGE), 0, 0x4000]);
    // Device page 0x1c0 maps the host buffer.
    let buf = 0x70_0000u64;
    assert_eq!(kona.test_write(0x100000 + 4 * 0x1c0, 4, 0x2000_0000 | (buf >> 12) as u64), BUS_OK);
    let dev = 0x1c0u64 << 14;
    words(&[0x129]);
    words(&INDEX_GC);
    words(&[0x154, 0, 2 << 16 | 1]);
    words(&[0x136, 2, 1, 0, 8, 2, 1, 0]);
    words(&[0x1f5, (dev >> 12) as u32, 0x9000 | (dev & 0xfff) as u32, 2, 2, 0, 0]);
    words(&[0x138, 0, 0]);
    assert_eq!(host.word(RETURN_PAGE), 0, "inside a bracket the bytes wait");
    words(&[0x1f6, 0x100, 0x77]);
    assert_eq!(host.bytes(buf, 2), [0x5a, 0x5a]);
    assert_eq!([host.word(REASON_PAGE), host.word(REASON_PAGE + 4), host.word(REASON_PAGE + 8)], [0x100, 0x77, 0]);
    let text = monitor(&kona, &["status"]).1;
    assert!(text.contains("1 reads"), "{text}");
    kona.stop();
}

/// While someone holds the pipe's state, the pixel commands still answer
/// and a wait for the state gives up instead of hanging.
#[test]
fn pixel_reads_do_not_wait_for_the_state() {
    use crate::traits::Device;
    let (_host, kona) = board();
    kona.with_backend(|_| {
        let out = Capture::default();
        kona.execute_command("kona", &["hash"], Box::new(out.clone())).unwrap();
        assert_eq!(out.text().trim().len(), 16);
        kona.execute_command("kona", &["pix", "1", "1"], Box::new(out.clone())).unwrap();
        assert!(out.text().contains("word arrays allocated"));
    });
}

/// A command that reads the back end's state gives up waiting for it and
/// says so: the answer is "board busy", and the state is its again after.
#[test]
fn a_state_command_answers_busy_when_the_state_stays_out() {
    use crate::traits::Device;
    let cfg = test_cfg();
    let host = TestHost::new();
    let kona = Kona::new(cfg, host.clone(), host.clone(), crate::dev::kona::tests::test_status(), None).unwrap();
    kona.shared.monitor_wait.store(0, Relaxed);
    kona.with_backend(|_| {
        let out = Capture::default();
        assert_eq!(kona.execute_command("kona", &["status"], Box::new(out.clone())), Err("kona: board busy".into()));
        assert!(kona.shared.pipe.try_lock().is_none(), "the state is out");
    });
    assert!(kona.shared.pipe.try_lock().is_some());
    let out = Capture::default();
    kona.execute_command("kona", &["status"], Box::new(out.clone())).unwrap();
    assert!(out.text().contains("raster managers"));
}

/// A stop from the main thread leaves the renderer to it; any other stop
/// has the display thread release it as its last act.
#[test]
fn a_stop_from_the_main_thread_leaves_the_renderer_alone() {
    use crate::dev::ng1::rex3::Renderer;
    use crate::traits::Device;
    use std::sync::atomic::AtomicU32;
    struct Spy(Arc<AtomicU32>);
    impl Renderer for Spy {
        fn present(
            &mut self,
            _: &mut crate::disp::Rex3Screen,
            _: &mut crate::debug_overlay::DebugOverlay,
            _: &mut crate::disp::StatusBar,
            _: &mut crate::disp::StatusBarTexture,
            _: &crate::disp::BarStats,
            _: bool,
            _: Option<&[u32]>,
            _: Option<&[u32]>,
        ) {
        }

        fn stop(&mut self) {
            self.0.fetch_add(1, Relaxed);
        }
    }
    let run = |name: &str| {
        let stops = Arc::new(AtomicU32::new(0));
        let spy = stops.clone();
        std::thread::Builder::new()
            .name(name.into())
            .spawn(move || {
                let cfg = test_cfg();
                let host = TestHost::new();
                let kona = Kona::new(cfg, host.clone(), host.clone(), crate::dev::kona::tests::test_status(), None).unwrap();
                *kona.with_backend(|b| b.display()).renderer_slot().lock() = Some(Box::new(Spy(spy)));
                kona.start();
                kona.stop();
            })
            .unwrap()
            .join()
            .unwrap();
        stops.load(Relaxed)
    };
    assert_eq!(run("main"), 0, "the main thread releases it");
    assert_eq!(run("other"), 1, "the display thread releases it");
}

/// The machine may take the window before or after wiring in its status
/// source: it is the same window.
#[test]
fn the_window_is_the_same_before_and_after_the_status_is_wired() {
    let cfg = test_cfg();
    let backend = KonaBackend::new(cfg);
    let before = backend.display();
    backend.set_status(crate::disp::StatusSource::new(Arc::default(), Arc::default()));
    assert!(Arc::ptr_eq(&before, &backend.display()));
}

#[test]
fn pix_shows_the_allocated_words_of_a_pixel() {
    let mut b = backend();
    let (known, text) = run(&mut b, &["pix", "3", "4"]);
    assert!(known && text.contains("0 word arrays allocated"), "{text}");
    b.store.write(3, 4, 2, 0xabcd);
    let text = run(&mut b, &["pix", "3", "4"]).1;
    assert!(text.contains("word  2  0x0000abcd") && text.contains("1 word arrays"), "{text}");
    assert!(run(&mut b, &["pix", "5000", "0"]).1.contains("outside"));
    assert!(!run(&mut b, &["pix", "x", "0"]).0);
    // A reset zeroes the store in place.
    b.reset();
    assert!(run(&mut b, &["pix", "3", "4"]).1.contains("word  2  0x00000000"));
}

#[test]
fn attach_and_stop_hold_and_release_the_handle() {
    use crate::traits::Device;
    let (_host, kona) = board();
    assert!(monitor(&kona, &["status"]).1.contains("detached"));
    kona.start();
    assert!(monitor(&kona, &["status"]).1.contains("display handle    attached"));
    kona.stop();
    assert!(monitor(&kona, &["status"]).1.contains("detached"));
}

// ---- GL pixel operations ----

/// `gl_window` with a window-sized orthographic projection, so a raster position is a window pixel.
fn pixel_window() -> KonaBackend {
    let mut b = gl_window();
    command(&mut b, Grammar::Gl, SELECT_CURRENT_MATRIX_STACK, &[crate::dev::kona::defs::gl::PROJECTION]);
    let m = crate::dev::gl::math::ortho(0.0, 64.0, 0.0, 64.0, -1.0, 1.0);
    command(&mut b, Grammar::Gl, LOAD_MATRIX_16_FLOATS, &m.map(f32::to_bits));
    b
}

fn at(b: &mut KonaBackend, x: i32, y: i32) {
    command(b, Grammar::Gl, SET_RASTER_POSITION_2_COMPONENTS_32_BIT_INTEGER, &[x as u32, y as u32]);
}

/// A pixel draw: the header, then the image words the back end asked for.
fn draw_pixels(b: &mut KonaBackend, w: u32, h: u32, format: u32, ty: u32, image: &[u32]) {
    let hdr = [w, h, format, ty, 0, 0, 0, 0];
    assert_eq!(command(b, Grammar::Gl, DRAW_PIXEL_RECTANGLE_FROM_HOST_DATA, &hdr), if image.is_empty() { Reply::Done } else { Reply::Data(image.len() as u32) });
    if !image.is_empty() {
        send(b, Job::Data { grammar: Grammar::Gl, selector: DRAW_PIXEL_RECTANGLE_FROM_HOST_DATA, args: &hdr, words: image });
    }
}

fn colour_at(b: &mut KonaBackend, x: u32, y: u32) -> u32 {
    command(b, Grammar::Kernel, WINDOW_SYSTEM_PIXEL_PATH_FINISH, &[0]);
    b.store.read(x, y, 2)
}

/// A 2 x 2 RGBA image of bytes lands with its lower-left pixel at the raster position; zoom
/// replicates each pixel; a raster position outside the clip volume draws nothing.
#[test]
fn draw_pixels_lands_at_the_raster_position_and_zooms() {
    use crate::dev::kona::defs::pixel::*;
    let mut b = pixel_window();
    at(&mut b, 10, 20);
    draw_pixels(&mut b, 2, 2, FORMAT_RGBA, TYPE_UNSIGNED_BYTE, &[0xff00_00ff, 0x00ff_00ff, 0x0000_ffff, 0xffff_ffff]);
    assert_eq!([colour_at(&mut b, 10, 20), colour_at(&mut b, 11, 20), colour_at(&mut b, 10, 21), colour_at(&mut b, 11, 21)], [0xffff_0000, 0xff00_ff00, 0xff00_00ff, 0xffff_ffff]);
    assert_eq!(colour_at(&mut b, 12, 20), 0, "nothing past the image");
    command(&mut b, Grammar::Gl, SET_PIXEL_ZOOM_FACTORS, &[2.0f32.to_bits(), 3.0f32.to_bits()]);
    at(&mut b, 30, 30);
    draw_pixels(&mut b, 1, 1, FORMAT_RGBA, TYPE_UNSIGNED_BYTE, &[0x1020_30ff]);
    let lit = (28..40u32).flat_map(|y| (28..40u32).map(move |x| (x, y))).filter(|&(x, y)| colour_at(&mut b, x, y) == 0xff10_2030).collect::<Vec<_>>();
    assert_eq!(lit, vec![(30, 30), (31, 30), (30, 31), (31, 31), (30, 32), (31, 32)], "two wide and three high");
    at(&mut b, 100, 100);
    draw_pixels(&mut b, 1, 1, FORMAT_RGBA, TYPE_UNSIGNED_BYTE, &[0xffff_ffff]);
    assert_eq!(colour_at(&mut b, 63, 63), 0);
    assert!(run(&mut b, &["gl"]).1.contains("1 with no valid raster position"));
}

/// The pixel-store tokens set the unpack record that sizes the next draw; the transfer scale
/// and bias, and a pixel map under its flag, change what is drawn.
#[test]
fn unpack_state_transfer_and_maps_shape_a_draw() {
    use crate::dev::kona::defs::pixel::*;
    let mut b = pixel_window();
    at(&mut b, 5, 5);
    // 3 x 1 RGB bytes: nine bytes are three words, with alignment 4 or 1.
    let hdr = [3, 1, FORMAT_RGB, TYPE_UNSIGNED_BYTE, 0, 0, 0, 0];
    assert_eq!(command(&mut b, Grammar::Gl, DRAW_PIXEL_RECTANGLE_FROM_HOST_DATA, &hdr), Reply::Data(3));
    send(&mut b, Job::Data { grammar: Grammar::Gl, selector: DRAW_PIXEL_RECTANGLE_FROM_HOST_DATA, args: &hdr, words: &[0, 0, 0] });
    // Rows of 3 x 3 bytes at alignment 4 pad to 12 bytes, so two rows are six words.
    let hdr2 = [3, 2, FORMAT_RGB, TYPE_UNSIGNED_BYTE, 0, 0, 0, 0];
    assert_eq!(command(&mut b, Grammar::Gl, DRAW_PIXEL_RECTANGLE_FROM_HOST_DATA, &hdr2), Reply::Data(6));
    send(&mut b, Job::Data { grammar: Grammar::Gl, selector: DRAW_PIXEL_RECTANGLE_FROM_HOST_DATA, args: &hdr2, words: &[0; 6] });
    command(&mut b, Grammar::Gl, SET_PIXEL_STORE_PARAMETER_INTEGER_FORM, &[UNPACK_BASE + 5, 1]);
    assert_eq!(command(&mut b, Grammar::Gl, DRAW_PIXEL_RECTANGLE_FROM_HOST_DATA, &hdr2), Reply::Data(5), "alignment 1 packs the rows");
    send(&mut b, Job::Data { grammar: Grammar::Gl, selector: DRAW_PIXEL_RECTANGLE_FROM_HOST_DATA, args: &hdr2, words: &[0; 5] });
    // A red scale of 0.5.
    command(&mut b, Grammar::Gl, SET_PIXEL_TRANSFER_PARAMETER_FLOAT_FORM, &[RED_SCALE, 0.5f32.to_bits()]);
    at(&mut b, 20, 20);
    draw_pixels(&mut b, 1, 1, FORMAT_RGBA, TYPE_UNSIGNED_BYTE, &[0xff00_00ff]);
    let c = colour_at(&mut b, 20, 20);
    assert!((0x7f..=0x80).contains(&(c >> 16 & 0xff)), "{c:#x}");
    // A red-to-red map of two entries, loaded as a header and its values, flips red once its flag is set.
    command(&mut b, Grammar::Gl, SET_PIXEL_TRANSFER_PARAMETER_FLOAT_FORM, &[RED_SCALE, 1.0f32.to_bits()]);
    let hdr = [MAP_RED_TO_RED, 2, 1.0f32.to_bits(), 0];
    assert_eq!(command(&mut b, Grammar::Gl, LOAD_PIXEL_MAP_FLOAT_VALUES, &hdr), Reply::Data(2));
    send(&mut b, Job::Data { grammar: Grammar::Gl, selector: LOAD_PIXEL_MAP_FLOAT_VALUES, args: &hdr, words: &[1.0f32.to_bits(), 0.0f32.to_bits()] });
    command(&mut b, Grammar::Gl, SET_PIXEL_TRANSFER_PARAMETER_FLOAT_FORM, &[MAP_COLOUR, 1]);
    at(&mut b, 22, 20);
    draw_pixels(&mut b, 1, 1, FORMAT_RGBA, TYPE_UNSIGNED_BYTE, &[0xff00_00ff]);
    assert_eq!(colour_at(&mut b, 22, 20) >> 16 & 0xff, 0);
}

/// A bitmap lights its set bits in the raster colour, most significant bit leftmost, and moves the
/// position by its advance. The small form's word count includes the byte count, as the recording's does.
#[test]
fn a_small_bitmap_draws_and_advances() {
    let mut b = pixel_window();
    command(&mut b, Grammar::Gl, SET_CURRENT_COLOUR_4_COMPONENTS_SINGLE_FLOAT, &[1.0f32.to_bits(), 0, 0, 1.0f32.to_bits()]);
    at(&mut b, 8, 8);
    // Two rows of eight: 1010 0000 and 0000 0001. Origin (-1, 0): the glyph sits one pixel right; move 8 (origin plus move is 7).
    let args = [8, 2, (-1.0f32).to_bits(), 0, 7.0f32.to_bits(), 0, 2, 2];
    assert_eq!(command(&mut b, Grammar::Gl, DRAW_BITMAP_FROM_DISPLAY_LIST, &args), Reply::Data(1));
    send(&mut b, Job::Data { grammar: Grammar::Gl, selector: DRAW_BITMAP_FROM_DISPLAY_LIST, args: &args, words: &[0xa001_0000] });
    let lit = lit_pixels(&mut b);
    // Bytes 0xa0 (bottom row) and 0x01 (row above): the bottom row is first, at y = 8.
    assert_eq!(lit, vec![(9, 8), (11, 8), (16, 9)]);
    // The advance moved the position to x = 16: the next bitmap starts there.
    let one = [8, 1, 0, 0, 8.0f32.to_bits(), 0, 2, 1];
    command(&mut b, Grammar::Gl, DRAW_BITMAP_FROM_DISPLAY_LIST, &one);
    send(&mut b, Job::Data { grammar: Grammar::Gl, selector: DRAW_BITMAP_FROM_DISPLAY_LIST, args: &one, words: &[0x8000_0000] });
    assert!(lit_pixels(&mut b).contains(&(16, 8)));
    // An empty bitmap (a large one with no rows) moves the position and draws nothing.
    let empty = [0, 0, 0, 0, 0, 0.0f32.to_bits(), 0.0f32.to_bits(), 5.0f32.to_bits(), 0.0f32.to_bits()];
    assert_eq!(command(&mut b, Grammar::Gl, DRAW_BITMAP, &empty), Reply::Done);
    assert!(run(&mut b, &["gl"]).1.contains("3 bitmaps"), "{}", run(&mut b, &["gl"]).1);
}

/// A copy takes the read buffer's rectangle to the raster position; a read packs it for the host;
/// depth and stencil copies and reads carry their planes.
#[test]
fn copy_and_read_pixels() {
    use crate::dev::kona::defs::pixel::*;
    let mut b = pixel_window();
    at(&mut b, 4, 4);
    draw_pixels(&mut b, 2, 1, FORMAT_RGBA, TYPE_UNSIGNED_BYTE, &[0x1122_33ff, 0x4455_66ff]);
    // The read buffer is the back one the draw went to.
    command(&mut b, Grammar::Gl, SELECT_READ_BUFFER, &[crate::dev::kona::defs::gl::BACK]);
    command(&mut b, Grammar::Gl, SELECT_DRAW_BUFFER, &[crate::dev::kona::defs::gl::BACK]);
    at(&mut b, 30, 30);
    command(&mut b, Grammar::Gl, COPY_PIXEL_RECTANGLE_WITHIN_FRAME_BUFFER, &[4, 4, 2, 1, 0, COPY_COLOUR, 0]);
    assert_eq!([colour_at(&mut b, 30, 30), colour_at(&mut b, 31, 30)], [0xff11_2233, 0xff44_5566]);
    // The read's reply is the packed bytes.
    let screen = b.gl.screen();
    let ds = b.contexts.select(None).gl.draw_state(screen);
    b.arrays(&ds);
    let c = b.contexts.select(None);
    let env = gl::pixels::Env { store: &b.store, ds };
    let out = gl::pixels::plane_command(&mut c.gl, &mut b.pix_counts, &env, READ_PIXEL_RECTANGLE_BACK_TO_HOST, &[30, 30, 2, 1, FORMAT_RGBA, TYPE_UNSIGNED_BYTE, 0]);
    assert_eq!(out, gl::pixels::Out::Reply(vec![0x11, 0x22, 0x33, 0xff, 0x44, 0x55, 0x66, 0xff]));
    let out = gl::pixels::plane_command(&mut c.gl, &mut b.pix_counts, &env, READ_PIXEL_RECTANGLE_BACK_TO_HOST, &[30, 30, 1, 1, FORMAT_RED, TYPE_UNSIGNED_SHORT, 2]);
    assert_eq!(out, gl::pixels::Out::Reply(vec![0, 0, 0x11, 0x11]), "the destination's byte offset leads");
    // The overlap query: the copy's source and its destination meet when the rectangles do.
    let out = gl::pixels::plane_command(&mut c.gl, &mut b.pix_counts, &env, ASK_WHETHER_COPY_SOURCE_OVERLAPS_ITS_DESTINATION, &[29, 29, 4, 4]);
    assert_eq!(out, gl::pixels::Out::Reply(vec![0, 0, 0, 4]));
    let out = gl::pixels::plane_command(&mut c.gl, &mut b.pix_counts, &env, ASK_WHETHER_COPY_SOURCE_OVERLAPS_ITS_DESTINATION, &[0, 0, 4, 4]);
    assert_eq!(out, gl::pixels::Out::Reply(vec![0, 0, 0, 0]));
}

/// Stencil and depth images draw into their planes: a stencil image directly under the write
/// mask, a depth image as fragments at its depths.
#[test]
fn stencil_and_depth_draws() {
    use crate::dev::kona::defs::pixel::*;
    let mut b = pixel_window();
    command(&mut b, Grammar::Gl, CONFIGURE_FRAME_BUFFER_LAYOUT, &[0, 8, 0, 1, 0, 0, 24, 8, 0, 0]);
    at(&mut b, 6, 6);
    draw_pixels(&mut b, 2, 1, FORMAT_STENCIL, TYPE_UNSIGNED_BYTE, &[0x0507_0000]);
    let (stencil, depth) = {
        let l = b.contexts.select(None).gl.layout.clone().expect("a layout");
        (l.stencil.start as usize, l.depth.start as usize)
    };
    command(&mut b, Grammar::Kernel, WINDOW_SYSTEM_PIXEL_PATH_FINISH, &[0]);
    let read = |b: &KonaBackend, w: usize, x: u32| b.store.read(x, 6, w);
    assert_eq!([read(&b, stencil, 6), read(&b, stencil, 7)], [5, 7]);
    // A depth image (half and full) under the always-pass depth test writes the depth plane.
    command(&mut b, Grammar::Gl, SET_DEPTH_TEST, &[crate::dev::kona::defs::gl::ALWAYS]);
    command(&mut b, Grammar::Gl, ENABLE_CAPABILITY, &[crate::dev::kona::defs::gl::DEPTH_TEST]);
    at(&mut b, 6, 8);
    draw_pixels(&mut b, 2, 1, FORMAT_DEPTH, TYPE_UNSIGNED_SHORT, &[0x8000_ffff]);
    command(&mut b, Grammar::Kernel, WINDOW_SYSTEM_PIXEL_PATH_FINISH, &[0]);
    let z = |x: u32| b.store.read(x, 8, depth);
    assert!(z(6).abs_diff(0x8000_0000) < 0x0001_0000, "{:#x}", z(6));
    assert!(z(7) > 0xfffe_0000, "{:#x}", z(7));
}

/// Accumulation: load the colour buffer scaled, add, multiply, and return to the draw buffer.
#[test]
fn accumulation_loads_scales_and_returns() {
    use crate::dev::kona::defs::pixel::*;
    let mut b = pixel_window();
    command(&mut b, Grammar::Gl, CONFIGURE_FRAME_BUFFER_LAYOUT, &[0, 8, 0, 1, 0, 0, 24, 0, 16, 0]);
    at(&mut b, 3, 3);
    draw_pixels(&mut b, 1, 1, FORMAT_RGBA, TYPE_UNSIGNED_BYTE, &[0x8040_00ff]);
    command(&mut b, Grammar::Gl, SELECT_READ_BUFFER, &[crate::dev::kona::defs::gl::BACK]);
    let acc = |b: &mut KonaBackend, op: u32, v: f32| {
        command(b, Grammar::Gl, ACCUMULATION_BUFFER_OPERATION, &[op, v.to_bits()]);
    };
    acc(&mut b, ACCUM_LOAD, 0.5);
    acc(&mut b, ACCUM, 0.5);
    acc(&mut b, ACCUM_MULT, 0.5);
    acc(&mut b, ACCUM_ADD, 0.25);
    // Clear the colour, then return the accumulated colour: 0.5 * c + 0.25.
    command(&mut b, Grammar::Gl, SET_COLOUR_CLEAR_VALUE, &[0, 0, 0, 0]);
    command(&mut b, Grammar::Gl, CLEAR_SELECTED_BUFFERS, &[crate::dev::kona::defs::gl::COLOR_BUFFER_BIT]);
    acc(&mut b, ACCUM_RETURN, 1.0);
    let c = colour_at(&mut b, 3, 3);
    // red 0x80 -> 128/255 * 0.5 + 0.25 = 0.502 -> 0x80; green 0x40 -> 0.376 -> 0x60; blue 0 -> 0.25 -> 0x40
    assert_eq!([c >> 16 & 0xff, c >> 8 & 0xff, c & 0xff], [0x80, 0x60, 0x40], "{c:#x}");
    // Clearing the accumulation buffer takes the clear value.
    command(&mut b, Grammar::Gl, SET_ACCUMULATION_BUFFER_CLEAR_VALUE, &[0.5f32.to_bits(), 0.5f32.to_bits(), 0.5f32.to_bits(), 1.0f32.to_bits()]);
    command(&mut b, Grammar::Gl, CLEAR_SELECTED_BUFFERS, &[crate::dev::kona::defs::gl::ACCUM_BUFFER_BIT]);
    acc(&mut b, ACCUM_RETURN, 1.0);
    let c = colour_at(&mut b, 3, 3);
    assert!(c >> 16 & 0xff >= 0x7f && c >> 16 & 0xff <= 0x80, "{c:#x}");
    assert!(run(&mut b, &["gl"]).1.contains("6 accumulations"), "{}", run(&mut b, &["gl"]).1);
}


/// A value of exactly 1.0 runs as two half passes: loading, accumulating and adding it lands where
/// the single pass would, and multiplying by it changes nothing.
#[test]
fn accumulation_by_exactly_one_is_two_half_passes() {
    use crate::dev::kona::defs::pixel::*;
    let mut b = pixel_window();
    command(&mut b, Grammar::Gl, CONFIGURE_FRAME_BUFFER_LAYOUT, &[0, 8, 0, 1, 0, 0, 24, 0, 16, 0]);
    at(&mut b, 3, 3);
    draw_pixels(&mut b, 1, 1, FORMAT_RGBA, TYPE_UNSIGNED_BYTE, &[0x8040_00ff]);
    command(&mut b, Grammar::Gl, SELECT_READ_BUFFER, &[crate::dev::kona::defs::gl::BACK]);
    let acc = |b: &mut KonaBackend, op: u32, v: f32| {
        command(b, Grammar::Gl, ACCUMULATION_BUFFER_OPERATION, &[op, v.to_bits()]);
    };
    let back = |b: &mut KonaBackend| {
        command(b, Grammar::Gl, SET_COLOUR_CLEAR_VALUE, &[0, 0, 0, 0]);
        command(b, Grammar::Gl, CLEAR_SELECTED_BUFFERS, &[crate::dev::kona::defs::gl::COLOR_BUFFER_BIT]);
        acc(b, ACCUM_RETURN, 1.0);
        let c = colour_at(b, 3, 3);
        [c >> 16 & 0xff, c >> 8 & 0xff, c & 0xff]
    };
    acc(&mut b, ACCUM_LOAD, 1.0);
    assert_eq!(back(&mut b), [0x80, 0x40, 0], "a load of 1.0 is the colour");
    acc(&mut b, ACCUM_MULT, 1.0);
    assert_eq!(back(&mut b), [0x80, 0x40, 0], "multiplying by 1.0 changes nothing");
    acc(&mut b, ACCUM, 1.0);
    acc(&mut b, ACCUM_MULT, 0.5);
    assert_eq!(back(&mut b), [0x80, 0x40, 0], "accumulating 1.0 twice the colour, halved");
    acc(&mut b, ACCUM_ADD, 1.0);
    assert_eq!(back(&mut b), [0xff, 0xff, 0xff], "an add of 1.0 saturates a return");
}

// ---- Queries, select and feedback ----

/// A feedback pass takes what survives clipping and culling as GL's records and draws nothing; a
/// pass-through marker is a record of its own; a culled polygon is not fed back.
#[test]
fn a_feedback_pass_collects_records_and_draws_nothing() {
    use crate::dev::kona::back::gl::query::*;
    let mut b = gl_window();
    b.contexts.select(None).gl.feed.start(256);
    command(&mut b, Grammar::Gl, SELECT_FEEDBACK_RECORD_TYPE, &[FEEDBACK_3D]);
    command(&mut b, Grammar::Gl, SWITCH_BETWEEN_RENDER_SELECT_AND_FEEDBACK_MODES, &[FEEDBACK]);
    command(&mut b, Grammar::Gl, INSERT_FEEDBACK_PASS_THROUGH_MARKER, &[7.0f32.to_bits()]);
    vertices(&mut b, BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES, &[(8.0, 8.0), (40.0, 8.0), (8.0, 40.0)]);
    vertices(&mut b, BEGIN_PRIMITIVE_POINTS, &[(20.0, 30.0)]);
    // The back-face cull: this winding is clockwise.
    command(&mut b, Grammar::Gl, SET_FACE_CULLING, &[crate::dev::kona::defs::gl::BACK]);
    command(&mut b, Grammar::Gl, ENABLE_CAPABILITY, &[crate::dev::kona::defs::gl::CULL_FACE]);
    vertices(&mut b, BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES, &[(8.0, 8.0), (8.0, 40.0), (40.0, 8.0)]);
    command(&mut b, Grammar::Gl, SWITCH_BETWEEN_RENDER_SELECT_AND_FEEDBACK_MODES, &[RENDER]);
    let (words, overflow) = b.contexts.select(None).gl.feed.take();
    assert!(!overflow);
    let f = |v: f32| v.to_bits();
    // Pass-through; a polygon of three vertices, x y z each (z is half the depth range for z = 0, as the
    // formatter's scales leave it); a point. The culled triangle is absent.
    let half = 0.5 * 1_048_575.0 / 1_048_576.0;
    let want = [PASS_THROUGH, f(7.0), POLYGON, 3, f(8.0), f(8.0), f(half), f(40.0), f(8.0), f(half), f(8.0), f(40.0), f(half), POINT, f(20.0), f(30.0), f(half)];
    assert_eq!(words, want);
    assert!(lit_pixels(&mut b).is_empty(), "feedback draws nothing");
    assert!(run(&mut b, &["gl"]).1.contains("1 passes"));
}

/// A select pass writes a hit per name change: the marker, the name count, the nearest and farthest depth, the names.
#[test]
fn a_select_pass_records_hits_by_name() {
    use crate::dev::kona::back::gl::query::*;
    let mut b = gl_window();
    b.contexts.select(None).gl.feed.start(256);
    command(&mut b, Grammar::Gl, SWITCH_BETWEEN_RENDER_SELECT_AND_FEEDBACK_MODES, &[SELECT]);
    command(&mut b, Grammar::Gl, CLEAR_SELECTION_NAME_STACK, &[0]);
    command(&mut b, Grammar::Gl, PUSH_SELECTION_NAME, &[5]);
    vertices(&mut b, BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES, &[(8.0, 8.0), (40.0, 8.0), (8.0, 40.0)]);
    command(&mut b, Grammar::Gl, REPLACE_TOP_SELECTION_NAME, &[6]);
    // Beyond the far plane (z = 2): no hit.
    command(&mut b, Grammar::Gl, BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES, &[0]);
    for (x, y) in [(8.0, 8.0), (40.0, 8.0), (8.0, 40.0)] {
        command(&mut b, Grammar::Gl, SUBMIT_VERTEX_3_COMPONENTS_SINGLE_FLOAT, &[ndc(x), ndc(y), 2.0f32.to_bits()]);
    }
    command(&mut b, Grammar::Gl, END_OPEN_PRIMITIVE, &[0]);
    command(&mut b, Grammar::Gl, REPLACE_TOP_SELECTION_NAME, &[7]);
    vertices(&mut b, BEGIN_PRIMITIVE_POINTS, &[(10.0, 10.0)]);
    command(&mut b, Grammar::Gl, SWITCH_BETWEEN_RENDER_SELECT_AND_FEEDBACK_MODES, &[RENDER]);
    let (words, _) = b.contexts.select(None).gl.feed.take();
    let z = (0.5f32 * 4_294_967_040.0) as u32;
    assert_eq!(words, vec![1, 1, z, z, 5, 1, 1, z, z, 7]);
    assert!(lit_pixels(&mut b).is_empty());
}

/// The query tokens reach the context's state: the model-view matrix and the error come back, an
/// unknown generic name is counted.
#[test]
fn queries_are_answered_from_the_contexts_state() {
    let mut b = gl_window();
    command(&mut b, Grammar::Gl, SELECT_CURRENT_MATRIX_STACK, &[crate::dev::kona::defs::gl::MODELVIEW]);
    command(&mut b, Grammar::Gl, TRANSLATE_FLOATS, &[2.0f32.to_bits(), 3.0f32.to_bits(), 4.0f32.to_bits()]);
    let c = b.contexts.select(None);
    let r = gl::query::answer(&mut c.gl, &mut b.query_counts, QUERY_STATE_GENERIC, &[0xba6]);
    assert_eq!(r.len(), 16);
    assert_eq!([f32::from_bits(r[12]), f32::from_bits(r[13]), f32::from_bits(r[14]), f32::from_bits(r[15])], [2.0, 3.0, 4.0, 1.0]);
    // A query token through the job path counts (no return page here, so nothing is posted).
    command(&mut b, Grammar::Gl, QUERY_STATE_GENERIC, &[0x9999]);
    command(&mut b, Grammar::Gl, QUERY_HISTOGRAM, &[0]);
    let text = run(&mut b, &["gl"]).1;
    assert!(text.contains("3 answered") && text.contains("0x9999 x 1") && text.contains("0x104 x 1"), "{text}");
}


// ---- Multisample ----

/// `gl_window` reconfigured with `samples` samples and a 24-bit depth buffer.
fn ms_window(samples: u32) -> KonaBackend {
    let mut b = gl_window();
    command(&mut b, Grammar::Gl, CONFIGURE_FRAME_BUFFER_LAYOUT, &[0, 8, 0, 1, 0, samples, 24, 0, 0, 0]);
    b
}

/// A rectangle of two triangles from (2, 2) to (`right`, 20).
fn rectangle(b: &mut KonaBackend, right: f32) {
    vertices(b, BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES, &[(2.0, 2.0), (right, 2.0), (right, 20.0), (2.0, 2.0), (right, 20.0), (2.0, 20.0)]);
}

fn column(b: &mut KonaBackend, x: u32, y: u32) -> u32 {
    command(b, Grammar::Kernel, WINDOW_SYSTEM_PIXEL_PATH_FINISH, &[0]);
    b.store.read(x, y, 2)
}

/// In a multisampled window a smoothed line takes coverage (smoothing wins for lines) while a smoothed
/// polygon multisamples (multisampling wins for polygons).
#[test]
fn a_multisampled_window_smooths_lines_and_multisamples_polygons() {
    use crate::dev::kona::defs::gl::{LINE_SMOOTH, POLYGON_SMOOTH};
    let mut b = ms_window(4);
    command(&mut b, Grammar::Gl, ENABLE_CAPABILITY, &[LINE_SMOOTH]);
    vertices(&mut b, BEGIN_PRIMITIVE_INDEPENDENT_LINES, &[(10.5, 21.0), (40.5, 21.0)]);
    command(&mut b, Grammar::Kernel, WINDOW_SYSTEM_PIXEL_PATH_FINISH, &[0]);
    assert_eq!([alpha_at(&b, 25, 19), alpha_at(&b, 25, 20), alpha_at(&b, 25, 21), alpha_at(&b, 25, 22)], [0, 128, 128, 0], "half coverage a row either side");
    let mut b = ms_window(4);
    command(&mut b, Grammar::Gl, ENABLE_CAPABILITY, &[POLYGON_SMOOTH]);
    rectangle(&mut b, 10.75);
    assert_eq!([column(&mut b, 9, 10), column(&mut b, 10, 10), column(&mut b, 11, 10)], [0xffff_ffff, 0xff80_8080, 0], "the multisampled edge, not a ramp");
}

/// The right edge of a rectangle at 10.75 (10.25 once the half pixel comes off) falls between samples of its
/// column: the 4-sample pattern's samples at 0 and 2 eighths across are in (one on the edge, which a right edge
/// keeps), those at 4 and 6 out, so half the pixel is white; with 8 samples (0, 6, 5, 1, 2, 7, 3, 4 eighths across)
/// three are in. A column left of it is whole and one right of it empty; the shared diagonal of the two triangles
/// leaves no seam.
#[test]
fn a_four_and_an_eight_sample_edge_share_the_pixel_by_samples() {
    let mut b = ms_window(4);
    rectangle(&mut b, 10.75);
    assert_eq!([column(&mut b, 9, 10), column(&mut b, 10, 10), column(&mut b, 11, 10)], [0xffff_ffff, 0xff80_8080, 0]);
    // Every pixel the diagonal crosses is whole: no seam between the triangles.
    let seams: Vec<_> = (3..10).flat_map(|x| (3..19).map(move |y| (x, y))).filter(|&(x, y)| column(&mut b, x, y) != 0xffff_ffff).collect();
    assert!(seams.is_empty(), "{seams:?}");
    let mut b = ms_window(8);
    rectangle(&mut b, 10.75);
    // Three of eight samples: (3 * 255 + 4) / 8 = 96.
    assert_eq!([column(&mut b, 9, 10), column(&mut b, 10, 10), column(&mut b, 11, 10)], [0xffff_ffff, 0xff60_6060, 0]);
    let seams: Vec<_> = (3..10).flat_map(|x| (3..19).map(move |y| (x, y))).filter(|&(x, y)| column(&mut b, x, y) != 0xffff_ffff).collect();
    assert!(seams.is_empty(), "{seams:?}");
}

/// With the enable off, or no multisample buffer, the same edge is the single-sample one: the pixel is whole or empty.
#[test]
fn without_multisampling_an_edge_is_whole_or_empty() {
    let mut b = ms_window(4);
    command(&mut b, Grammar::Gl, DISABLE_CAPABILITY, &[crate::dev::kona::defs::gl::MULTISAMPLE]);
    rectangle(&mut b, 10.875);
    // The sample at the pixel's corner: 10.0 < 10.375, inside.
    assert_eq!([column(&mut b, 10, 10), column(&mut b, 11, 10)], [0xffff_ffff, 0]);
    let mut b = ms_window(0);
    rectangle(&mut b, 10.875);
    assert_eq!([column(&mut b, 10, 10), column(&mut b, 11, 10)], [0xffff_ffff, 0]);
}

/// Depth is kept a sample: a nearer rectangle and a farther one that share a pixel's edge each win the samples
/// they cover, and a farther fragment does not overwrite a nearer sample. A clear sets every sample.
#[test]
fn depth_and_clears_are_per_sample() {
    use crate::dev::kona::defs::gl::*;
    let mut b = ms_window(4);
    command(&mut b, Grammar::Gl, ENABLE_CAPABILITY, &[DEPTH_TEST]);
    command(&mut b, Grammar::Gl, SET_DEPTH_TEST, &[LESS]);
    command(&mut b, Grammar::Gl, SET_COLOUR_CLEAR_VALUE, &[0, 0, 0.5f32.to_bits(), 1.0f32.to_bits()]);
    command(&mut b, Grammar::Gl, CLEAR_SELECTED_BUFFERS, &[COLOR_BUFFER_BIT | DEPTH_BUFFER_BIT]);
    assert_eq!(column(&mut b, 5, 5), 0xff00_0080, "a clear sets the samples and the word");
    // A white rectangle at z = 0 (nearer), then a red one behind it at z = 0.5 that reaches further right.
    let rect = |b: &mut KonaBackend, right: f32, z: f32| {
        command(b, Grammar::Gl, BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES, &[0]);
        for (x, y) in [(2.0, 2.0), (right, 2.0), (right, 20.0), (2.0, 2.0), (right, 20.0), (2.0, 20.0)] {
            command(b, Grammar::Gl, SUBMIT_VERTEX_3_COMPONENTS_SINGLE_FLOAT, &[ndc(x), ndc(y), z.to_bits()]);
        }
        command(b, Grammar::Gl, END_OPEN_PRIMITIVE, &[0]);
    };
    rect(&mut b, 10.875, 0.0);
    command(&mut b, Grammar::Gl, SET_CURRENT_COLOUR_4_COMPONENTS_SINGLE_FLOAT, &[1.0f32.to_bits(), 0, 0, 1.0f32.to_bits()]);
    rect(&mut b, 14.875, 0.5);
    // Column 5 stays white (the red is behind); column 10 has two white samples and two red ones that the
    // far rectangle won where the near one is not (samples 2 and 3): mean red (2 * 255 + 2 * 255 + 2) / 4 = 255,
    // green and blue (2 * 255 + 2) / 4 = 128; column 12 is red.
    assert_eq!(column(&mut b, 5, 10), 0xffff_ffff);
    assert_eq!(column(&mut b, 10, 10) & 0xffffff, 0xff8080);
    assert_eq!(column(&mut b, 12, 10) & 0xffffff, 0xff0000);
}

/// The sample mask keeps the samples its coverage word names, by pixel column: the half-coverage mask on 4
/// samples is 0x33333333, nibble 3 (samples 0 and 1) a column: half the pixel.
#[test]
fn the_sample_mask_and_alpha_to_mask_thin_the_coverage() {
    use crate::dev::kona::defs::gl::*;
    let mut b = ms_window(4);
    command(&mut b, Grammar::Gl, SET_MULTISAMPLE_COVERAGE_MASK, &[0.5f32.to_bits(), 0]);
    command(&mut b, Grammar::Gl, ENABLE_CAPABILITY, &[SAMPLE_MASK]);
    rectangle(&mut b, 10.875);
    // Mask 0.5 selects the 15-16 entry: 0x33333333, samples 0 and 1 of every pixel: half of a whole pixel.
    assert_eq!(column(&mut b, 5, 10), 0xff80_8080);
    // Inverted, the other half.
    let mut b = ms_window(4);
    command(&mut b, Grammar::Gl, SET_MULTISAMPLE_COVERAGE_MASK, &[0.5f32.to_bits(), 1]);
    command(&mut b, Grammar::Gl, ENABLE_CAPABILITY, &[SAMPLE_MASK]);
    rectangle(&mut b, 10.875);
    assert_eq!(column(&mut b, 5, 10), 0xff80_8080);
    // Alpha to mask with alpha 0.5 lets half the samples through; alpha to one then makes the fragment opaque.
    let mut b = ms_window(4);
    command(&mut b, Grammar::Gl, SET_CURRENT_COLOUR_4_COMPONENTS_SINGLE_FLOAT, &[1.0f32.to_bits(), 1.0f32.to_bits(), 1.0f32.to_bits(), 0.5f32.to_bits()]);
    command(&mut b, Grammar::Gl, ENABLE_CAPABILITY, &[SAMPLE_ALPHA_TO_MASK]);
    command(&mut b, Grammar::Gl, ENABLE_CAPABILITY, &[SAMPLE_ALPHA_TO_ONE]);
    rectangle(&mut b, 10.875);
    assert_eq!(column(&mut b, 5, 10), 0xff80_8080);
    // A bad pattern is an invalid-enumerant error; a good one is taken.
    command(&mut b, Grammar::Gl, SET_MULTISAMPLE_PATTERN, &[0x1234]);
    assert_eq!(b.contexts.select(None).gl.error, 0x500);
    command(&mut b, Grammar::Gl, SET_MULTISAMPLE_PATTERN, &[0x80a3]);
}
