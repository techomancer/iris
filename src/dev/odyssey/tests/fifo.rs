//! The command FIFO: queueing, the board's thread, status, credits.

use crate::dev::odyssey::board::Board;
use crate::dev::odyssey::command::{CFIFO_PREAMBLE, CfifoPreamble};
use crate::dev::odyssey::consts::{
    cfifo, ctx, flctl, host, intr, reg, self, status0, ODSY_BUZZ_1, ODSY_STATUS0, ODSY_SW_TOKEN,
};
use crate::dev::odyssey::fifo::{CommandFifo, RING_STORES, RUN_MAX, Run};
use crate::dev::odyssey::interface::HostInterface;
use crate::dev::odyssey::mmio::BoardLocks;
use crate::dev::odyssey::pi_gfx::{
    pi_gfx_mark_store, pi_gfx_take_mark, pi_gfx_unmark_store, PI_GFX_BIAS, PI_GFX_CREDIT,
    PI_GFX_PAGE, PI_GFX_PAGE_ENABLE, PiGfx, PiGfxLink,
};
use crate::dev::odyssey::tests::{
    board, board_with_ram, cmd, cmd64, reg_read, reg_write, route_to_heart, select, FLAG_VECTOR,
    Ram, VPRO_WIDGET,
};
use crate::dev::odyssey::worker::Budget;
use parking_lot::Mutex;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use super::*;
use std::sync::atomic::Ordering::SeqCst;

/// A board on its own thread, routed to the recorder and with the
/// low-water source enabled and clear.
fn threaded_board() -> (BoardLocks, Arc<Recorder>) {
    let (s, rec) = board("v12");
    let h = BoardLocks::new(s);

    h.work_start();
    h.write(ROUTE_DEST_UPPER, HEART_WIDGET << 25, 4);
    h.write(ROUTE_DEST_LOWER, HEART_PACKET, 4);
    h.write(ROUTE_VECTORS, TEST_VECTOR as u64, 4);
    h.write(INTR_CLEAR, INTR_FIFO_LOW_WATER, 4);
    h.write(INTR_ENABLE_SET, INTR_FIFO_LOW_WATER, 4);
    assert!(!rec.level(TEST_VECTOR));
    (h, rec)
}

/// The self-test preamble twice: it completes only in order.
fn preamble_twice() -> Vec<&'static CfifoPreamble> {
    let words: Vec<_> = CFIFO_PREAMBLE.iter().chain(CFIFO_PREAMBLE.iter()).collect();
    assert!(words.len() as u32 > cfifo::LOW_WATER);
    words
}

/// Queue every FIFO store and have the first run last
/// `hold_ms` as a long draw would; store the first word and return
/// once that run is under way.
fn start_long_slice(h: &BoardLocks, words: &[&CfifoPreamble], hold_ms: u32) {
    h.fifo.test_words.store(1, SeqCst);
    h.test_board().test_slice_sleep_ms = hold_ms;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    h.write(words[0].port, words[0].value, words[0].size);
    while !(h.fifo.slice.load(SeqCst)) {
        assert!(std::time::Instant::now() < deadline, "the first slice did not start");
        std::thread::yield_now();
    }
}

/// A burst of command-FIFO stores queued behind a long slice runs on the
/// board's thread in the order stored, and the low-water flag rises, and
/// its interrupt with it, once the queue drains below the low-water
/// mark. While the slice holds the board, the stores queue and the
/// status register answers without waiting for it.
#[test]
fn the_board_thread_runs_queued_stores_in_order_and_latches_low_water() {
    const STATUS0: u64 = 0x1064;
    let (h, rec) = threaded_board();
    let words = preamble_twice();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);

    start_long_slice(&h, &words, 2000);
    // The rest go in while the first run is under way: only
    // the front lock is needed.
    for w in &words[1..] {
        h.write(w.port, w.value, w.size);
    }
    {
        let level = h.fifo.level();
        assert!(level >= cfifo::LOW_WATER, "level {level}: the first slice should still hold the queue");
        assert_eq!(h.front.lock().intr_flags & intr::FIFO_LOW_WATER, 0);
        // The status register does not wait: not empty, above low water.
        let status = h.read(STATUS0, 4) as u32;
        assert_eq!(status & (status0::FIFO_EMPTY | status0::FIFO_LOW_WATER), 0);
        assert_eq!(h.fifo.level(), level);
        // All of that while the slice still held the board.
        assert!(h.fifo.slice.load(SeqCst));
    }

    // Watch the queue drain on the board's thread, without draining it
    // from here: the flag is up exactly when the level is below the mark.
    loop {
        {
            let f = h.front.lock();
            let level = h.fifo.level();
            let flag = f.intr_flags & intr::FIFO_LOW_WATER != 0;
            assert_eq!(flag, level < cfifo::LOW_WATER, "level {level}");
            if level == 0 && !h.fifo.slice.load(SeqCst) {
                break;
            }
        }
        assert!(std::time::Instant::now() < deadline, "work thread did not drain the queue");
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    {
        let s = h.test_board();
        assert!(s.widget.cfifo_selftest_ready);
        assert_eq!(s.widget.selftest_progress, 0);
    }
    assert!(rec.level(TEST_VECTOR));
    let status = h.read(STATUS0, 4) as u32;
    assert_eq!(
        status & (status0::FIFO_EMPTY | status0::FIFO_LOW_WATER),
        status0::FIFO_EMPTY | status0::FIFO_LOW_WATER
    );
    h.work_stop();
    assert!(h.fifo.thread.lock().is_none());
}

/// A read that waits for the work, issued while the board's thread is in a
/// long run, is answered busy until every store queued before it has run:
/// the self-test register reads OK, though the board's thread runs one store
/// per run and most were still queued when the read began.
#[test]
fn a_read_that_waits_sees_every_earlier_store() {
    let (h, _rec) = threaded_board();
    let words = preamble_twice();

    start_long_slice(&h, &words, 300);
    for w in &words[1..] {
        h.write(w.port, w.value, w.size);
    }
    assert!(h.fifo.level() > 1);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    let selftest = loop {
        if let Some(v) = h.bus_read(cfifo::SELFTEST, 4) {
            break v as u32;
        }
        assert!(std::time::Instant::now() < deadline, "the read stayed busy");
        std::thread::yield_now();
    };
    assert_eq!(selftest, cfifo::SELFTEST_OK);
    assert_eq!(h.fifo.level(), 0);
    h.work_stop();
}

/// A processor store that finds the host ring full while the board's thread
/// runs is refused with the bus busy, and nothing runs on the processor's
/// thread: the level stays at the brim and the slice in flight keeps the
/// board. The retried store goes in, once, when the board's thread has made
/// room, and the level never passes the model's FIFO size.
#[test]
fn a_full_ring_holds_the_store_back_until_the_board_makes_room() {
    let (h, _rec) = threaded_board();
    let words = preamble_twice();

    start_long_slice(&h, &words, 2000);
    while h.fifo.stores() < RING_STORES {
        assert!(h.bus_write(consts::cmd::FIFO, 0, 4));
    }
    let stores = h.fifo.stores();
    assert!(!h.bus_write(consts::cmd::FIFO, 0, 4), "a full ring refuses the store");
    assert!(!h.bus_write(consts::cmd::FIFO, 0, 8));
    assert_eq!((h.fifo.level(), h.fifo.stores()), (cfifo::WORDS, stores), "nothing ran on the processor's thread");
    assert!(h.fifo.slice.load(SeqCst), "the board's thread still holds the board");

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    while !h.bus_write(consts::cmd::FIFO, 0, 8) {
        assert!(h.fifo.level() <= cfifo::WORDS);
        assert!(std::time::Instant::now() < deadline, "the board's thread made no room");
        std::thread::yield_now();
    }
    assert!(h.fifo.level() <= cfifo::WORDS);
    h.fifo.test_words.store(0, SeqCst);
    h.work_stop();
    assert_eq!(h.fifo.level(), 0);
}

/// A read that waits for the work, issued on the bus while stores are
/// queued behind a long slice, answers busy and runs nothing on the
/// processor's thread; retried, it answers once the board's thread has run
/// every store before it, and sees them all. A store that waits answers
/// busy the same way.
#[test]
fn a_waiting_bus_read_is_busy_until_the_board_thread_has_run_the_stores() {
    let (h, _rec) = threaded_board();
    let words = preamble_twice();

    start_long_slice(&h, &words, 300);
    for w in &words[1..] {
        h.write(w.port, w.value, w.size);
    }
    let level = h.fifo.level();
    assert!(level > 1);
    assert_eq!(h.bus_read(cfifo::SELFTEST, 4), None, "busy while the stores wait");
    assert!(!h.bus_write(ODSY_BUZZ_1, 0, 4), "a waiting store is busy too");
    assert_eq!(h.fifo.level(), level, "nothing ran on the processor's thread");

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    let selftest = loop {
        if let Some(v) = h.bus_read(cfifo::SELFTEST, 4) {
            break v as u32;
        }
        assert!(std::time::Instant::now() < deadline, "the board's thread did not run the stores");
        std::thread::yield_now();
    };
    assert_eq!(selftest, cfifo::SELFTEST_OK);
    assert_eq!(h.fifo.level(), 0);
    assert!(h.bus_write(ODSY_BUZZ_1, 0, 4));
    h.fifo.test_words.store(0, SeqCst);
    h.work_stop();
}

/// Records every call a board makes on its processor-interface link.
#[derive(Default)]
struct PiStub(Mutex<Vec<(&'static str, u32, i64)>>);

impl PiGfxLink for PiStub {
    fn return_credits(&self, widget: u32, n: u32) {
        self.0.lock().push(("return", widget, n as i64));
    }
    fn owe_credits(&self, widget: u32, n: i32) {
        self.0.lock().push(("owe", widget, n as i64));
    }
    fn repay_credits(&self, widget: u32, n: u32) {
        self.0.lock().push(("repay", widget, n as i64));
    }
}

impl PiStub {
    fn count(&self, kind: &str) -> i64 {
        self.0.lock().iter().filter(|c| c.0 == kind).map(|c| c.2).sum()
    }
    /// Credits given back so far, returned or repaid.
    fn settled(&self) -> i64 {
        self.count("return") + self.count("repay")
    }
}

/// A board whose stores are all counted and queued, with a stub link.
fn credit_board() -> (Box<Board>, Arc<PiStub>) {
    let (s, _) = board("v12");
    let pi = Arc::new(PiStub::default());
    {
        let mut f = s.front.lock();
        f.pi = Some(pi.clone());
        f.test_counted_stores = 1;
    }
    // No work thread runs: hold the queue and step by hand.
    s.fifo.test_words.store(1, SeqCst);
    (s, pi)
}

/// Command work credits at FIFO entry: a counted store's credit
/// comes back when the board takes it into its FIFO, the oldest
/// `cfifo::DEPTH` waiting stores at once; each store behind them is owed, then
/// repaid as the work ahead runs and it moves up. A store that does
/// not wait is credited inside the store.
///
/// QEMU: command-work-credits-at-fifo-entry.
#[test]
fn a_counted_store_is_credited_as_it_enters_the_fifo() {
    let (mut s, pi) = credit_board();
    let stores = cfifo::DEPTH + 8;

    for _ in 0..stores {
        s.mmio_write(consts::cmd::FIFO, 0, 8);
    }
    // Credits count stores; the level counts words, two a doubleword, and
    // reads no more than the model's FIFO size.
    assert_eq!(s.fifo.stores(), stores);
    assert_eq!(s.fifo.level(), (2 * stores).min(cfifo::WORDS));
    assert_eq!(s.fifo.in_fifo(), cfifo::DEPTH);
    assert_eq!(pi.count("return"), cfifo::DEPTH as i64);
    assert_eq!(pi.count("owe"), (stores - cfifo::DEPTH) as i64);
    assert_eq!(pi.count("repay"), 0);
    assert!(pi.0.lock().iter().all(|c| c.1 == VPRO_WIDGET));

    // One store runs per slice: the next one moves up and is repaid.
    s.work_step(&mut Budget::stores(1));
    assert_eq!(pi.count("repay"), 1);
    assert_eq!(pi.settled(), cfifo::DEPTH as i64 + 1);

    s.work_step(&mut Budget::unlimited());
    assert!(!s.fifo.pending());
    assert_eq!(pi.settled(), stores as i64);

    // A store that does not wait is credited inside the store.
    s.fifo.test_words.store(0, SeqCst);
    s.mmio_write(consts::cmd::FIFO, 0, 8);
    assert!(!s.work_pending());
    assert_eq!(pi.settled(), stores as i64 + 1);
    assert_eq!(pi.count("owe"), (stores - cfifo::DEPTH) as i64);
}

/// A run of stores the board's thread has looked at and then runs leaves
/// the FIFO together. The stores run in order; the level, the store count
/// and the credits end where taking the stores out one by one leaves them,
/// including when a counted store joins the queue after the run was taken.
#[test]
fn a_run_retires_with_the_levels_and_credits_of_one_by_one() {
    fn queue(s: &mut Board) {
        s.front.lock().test_counted_stores = 0;
        for _ in 0..cfifo::DEPTH + 6 {
            s.mmio_write(consts::cmd::FIFO, 0, 8);
        }
    }
    fn counted_join(s: &mut Board) {
        s.front.lock().test_counted_stores = 1;
        for _ in 0..10 {
            s.mmio_write(consts::cmd::FIFO, 0, 8);
        }
    }
    // One at a time: the store count and the credits are the reference.
    let (mut one, pi_one) = credit_board();
    queue(&mut one);
    counted_join(&mut one);
    for _ in 0..RUN_MAX {
        step(&mut one);
    }

    let (mut run, pi_run) = credit_board();
    queue(&mut run);
    let mut taken = Run::new();
    run.fifo.take_run(&mut taken);
    assert_eq!(taken.stores().len(), RUN_MAX, "no counted store waits yet: a full run");
    counted_join(&mut run);
    assert_eq!(run.fifo.stores(), cfifo::DEPTH + 16);
    run.fifo.retire(&run.front, &taken, RUN_MAX);

    assert_eq!(run.fifo.stores(), one.fifo.stores());
    assert_eq!(run.fifo.level(), one.fifo.level());
    assert_eq!(pi_run.count("repay"), pi_one.count("repay"));
    assert_eq!(pi_run.count("return"), pi_one.count("return"));
    assert_eq!(pi_run.count("owe"), pi_one.count("owe"));
    assert!(pi_run.count("repay") > 0, "the run moved owed stores up");
    // The counted stores left all sit inside the credited depth, so the
    // next run is full.
    assert!(run.fifo.stores() <= cfifo::DEPTH);
    run.fifo.take_run(&mut taken);
    assert_eq!(taken.stores().len(), RUN_MAX);
}

/// The board's thread takes stores one at a time only while some wait past
/// the credited depth, where one may owe its credit; a backlog of counted
/// stores inside the depth, all credited, goes in full runs.
#[test]
fn a_run_is_one_store_only_past_the_credited_depth() {
    let (mut s, _) = credit_board();
    for _ in 0..cfifo::DEPTH {
        s.mmio_write(consts::cmd::FIFO, 0, 8);
    }
    let mut run = Run::new();
    s.fifo.take_run(&mut run);
    assert_eq!(run.stores().len(), RUN_MAX, "every waiting store is credited");
    s.mmio_write(consts::cmd::FIFO, 0, 8);
    s.fifo.take_run(&mut run);
    assert_eq!(run.stores().len(), 1, "one store owes its credit");
}

/// A slice cut short by its budget retires only the stores it ran: the rest
/// stay queued, in order, for the next slice.
#[test]
fn a_slice_cut_short_leaves_the_rest_of_its_run_queued() {
    let (mut s, _, ram) = queued_board();
    push_fences(&mut s);
    assert_eq!(s.fifo.stores(), 6);

    s.work_step(&mut Budget::stores(4));
    assert_eq!(s.fifo.stores(), 2);
    assert_eq!(s.fifo.level(), 2);
    assert_eq!(ram.half(HOST_WRITE_BASE + 4), 0x5a00, "the first fence ran");
    assert_eq!(ram.half(HOST_WRITE_BASE + 6), 0, "the second fence has stores left to run");

    finish(&mut s);
    assert_eq!(ram.half(HOST_WRITE_BASE + 6), 0xa5a5);
}

/// A reset abandons the waiting stores and cancels each owed credit
/// (one owe of -1 per owed store); stores not owed cancel nothing.
#[test]
fn a_reset_cancels_the_credits_owed_for_abandoned_stores() {
    let (mut s, pi) = credit_board();

    for _ in 0..cfifo::DEPTH + 8 {
        s.mmio_write(consts::cmd::FIFO, 0, 8);
    }
    assert_eq!(pi.count("owe"), 8);
    s.work_reset();
    assert!(!s.fifo.pending());
    assert_eq!(s.fifo.in_fifo(), 0);
    assert_eq!(pi.count("owe"), 0);
    assert_eq!(pi.0.lock().iter().filter(|c| c.0 == "owe" && c.2 == -1).count(), 8);
    assert_eq!(pi.count("repay"), 0);
}

/// The processor interface's mark makes a store counted, once, and a
/// counted store with no link logs and returns nothing.
#[test]
fn the_processor_mark_counts_exactly_one_store() {
    let (mut s, pi) = credit_board();

    {
        let mut f = s.front.lock();
        f.test_counted_stores = 0;
    }
    s.fifo.test_words.store(0, SeqCst);
    s.mmio_write(consts::cmd::FIFO, 0, 8);
    assert_eq!(pi.settled(), 0);

    let previous = pi_gfx_mark_store(true);
    s.mmio_write(consts::cmd::FIFO, 0, 8);
    pi_gfx_unmark_store(previous);
    assert_eq!(pi.0.lock().as_slice(), &[("return", VPRO_WIDGET, 1)]);

    // One-shot: the mark was taken by the first store.
    s.mmio_write(consts::cmd::FIFO, 0, 8);
    assert_eq!(pi.settled(), 1);

    // The Octane: no link, so the credit is dropped with a log line.
    s.front.lock().pi = None;
    pi_gfx_mark_store(true);
    s.mmio_write(consts::cmd::FIFO, 0, 8);
    assert_eq!(pi.settled(), 1);
    assert!(!pi_gfx_take_mark());
}

/// The register block's pure state (PiGfx).
#[test]
fn the_credit_block_starves_a_widget_that_runs_out_of_credits() {
    let mut g = PiGfx::default();

    g.write(PI_GFX_PAGE, PI_GFX_PAGE_ENABLE | 0x2_0000);
    g.write(PI_GFX_CREDIT, 3);
    g.write(PI_GFX_BIAS, 2);
    assert!(g.covers(0, 0x2_1000));
    assert!(!g.covers(0, 0x4_0000));
    assert!(g.count_store(0, 0x2_0000, "t"));
    assert!(!g.take_starved(0, "t"));
    assert!(g.count_store(0, 0x2_0000, "t"));
    assert!(g.take_starved(0, "t"));
    assert!(!g.take_starved(0, "t"));
    g.add_credits(0, 5, "t");
    assert_eq!(g.read(PI_GFX_CREDIT), 6);
    assert!(!g.starved[0]);
}


/// The model's low-water threshold, in waiting words (a model choice).
const LOW_WATER_MARK: u32 = cfifo::LOW_WATER;
const HOST_WRITE_BASE: u64 = 0xb70000;

/// A board whose stores wait in the queue until a test steps them one
/// store a slice, with RAM for the fences they write.
fn queued_board() -> (Box<Board>, Arc<Recorder>, Arc<Ram>) {
    let (mut s, rec, ram) = board_with_ram();
    reg_write(&mut s, host::WRITE_ADDR_UPPER, 0);
    reg_write(&mut s, host::WRITE_ADDR_LOWER, HOST_WRITE_BASE as u32);
    s.fifo.test_words.store(1, SeqCst);
    (s, rec, ram)
}

/// Two host-write fences, slot 2 then slot 3, each three words.
fn push_fences(s: &mut Board) {
    for (slot, token) in [(2, 0x5a00), (3, 0xa5a5)] {
        cmd(s, select(reg::HOST_WRITE));
        cmd(s, slot);
        cmd(s, token);
    }
}

/// Run one store, as a one-store slice on the board's thread would.
fn step(s: &mut Board) {
    let before = s.fifo.stores();
    assert!(before > 0, "nothing waits to be stepped");
    s.work_step(&mut Budget::stores(1));
    assert_eq!(s.fifo.stores(), before - 1);
}

/// Run everything that waits.
fn finish(s: &mut Board) {
    s.work_step(&mut Budget::unlimited());
    assert!(!s.fifo.pending());
}

/// The stores return with their work waiting; it runs one store a slice,
/// in order, and its fences land when it does.
///
/// QEMU: command-work-runs-in-slices.
#[test]
fn stores_wait_and_run_one_slice_at_a_time() {
    let (mut s, _, ram) = queued_board();
    push_fences(&mut s);

    assert!(s.work_pending());
    assert_eq!(ram.half(HOST_WRITE_BASE + 4), 0);
    assert_eq!(ram.half(HOST_WRITE_BASE + 6), 0);
    let mut steps = 0;
    while s.work_pending() {
        steps += 1;
        assert!(steps < 100);
        step(&mut s);
    }
    assert!(steps >= 3);
    assert_eq!(ram.half(HOST_WRITE_BASE + 4), 0x5a00);
    assert_eq!(ram.half(HOST_WRITE_BASE + 6), 0xa5a5);
}

/// A context switch waits behind the commands before it: the kernel polls
/// the done word, which the board writes only once those have run.
///
/// QEMU: command-work-orders-context-switch.
#[test]
fn a_context_switch_completes_only_after_the_commands_before_it() {
    let (mut s, _, ram) = queued_board();
    let base = 0xb80000u64;
    push_fences(&mut s);
    reg_write(&mut s, ctx::SHADOW_ADDR_UPPER, 0);
    reg_write(&mut s, ctx::SHADOW_ADDR_LOWER, base as u32);
    s.mmio_write(ctx::SWITCH_PORT, 0x0000_0700_0001_0001, 8);

    let done = |ram: &Ram| u32::from_be_bytes(ram.get(base + ctx::SWITCH_DONE_OFFSET, 4).try_into().unwrap());
    assert_eq!(done(&ram), 0);
    while s.work_pending() {
        step(&mut s);
        if done(&ram) != 0 {
            assert_eq!(ram.half(HOST_WRITE_BASE + 6), 0xa5a5, "the fences before the switch had landed");
        }
    }
    assert_ne!(done(&ram), 0);
}

/// A read of a register the work may change finishes the work first; the
/// software token, which no command touches, is answered at once.
///
/// QEMU: command-work-read-waits.
#[test]
fn a_read_of_a_register_the_work_may_change_finishes_the_work_first() {
    let (mut s, _, ram) = queued_board();
    push_fences(&mut s);

    reg_write(&mut s, ODSY_SW_TOKEN, 1);
    assert_eq!(reg_read(&mut s, ODSY_SW_TOKEN), 1);
    assert!(s.work_pending(), "the token is answered without waiting");

    reg_read(&mut s, ODSY_BUZZ_1);
    assert!(!s.work_pending());
    assert_eq!(ram.half(HOST_WRITE_BASE + 6), 0xa5a5);
}

/// The level word is a plain load: with words queued it answers at once
/// with their count in bits 15:7, in doublewords, and runs nothing.
///
/// QEMU: command-work-level-word-reads-live.
#[test]
fn the_level_word_reports_the_queue_without_draining_it() {
    let (mut s, _, ram) = queued_board();
    push_fences(&mut s);
    let (words, stores) = (s.fifo.level(), s.fifo.stores());
    assert!(words > 0);

    let level = (reg_read(&mut s, MEMORY_CONFIG) >> 7) & 0x1ff;
    assert_eq!(level, words.div_ceil(2));
    assert_ne!(level, 0);
    assert_eq!((s.fifo.level(), s.fifo.stores()), (words, stores), "the read left the queue alone");
    assert_eq!(ram.half(HOST_WRITE_BASE + 6), 0, "nothing ran");
    assert_eq!(reg_read(&mut s, MEMORY_CONFIG) & 0xff00_0000, 0xa100_0000, "the straps are still there");
}

/// The drain waits poll bits 15:7 until they read 0, so the level falls to
/// 0 as the board empties the queue.
///
/// QEMU: command-work-level-word-falls-to-zero.
#[test]
fn the_level_word_reads_zero_once_the_board_has_run_the_queue() {
    let (mut s, _, _) = queued_board();
    push_fences(&mut s);
    let mut last = (reg_read(&mut s, MEMORY_CONFIG) >> 7) & 0x1ff;
    assert_ne!(last, 0);
    while s.work_pending() {
        step(&mut s);
        let now = (reg_read(&mut s, MEMORY_CONFIG) >> 7) & 0x1ff;
        assert!(now <= last, "the level never rises while the board drains");
        last = now;
    }
    assert_eq!(last, 0);
    assert_eq!(reg_read(&mut s, MEMORY_CONFIG) & 0xff80, 0);
}

/// MODEL CONTRACT ONLY, no evidence for the hardware: while stores wait the
/// model's status is neither empty nor below its low-water threshold; it
/// reads empty and low again once they have run.
///
/// QEMU: command-work-status-shows-waiting-stores.
#[test]
fn the_status_shows_waiting_stores_until_they_have_run() {
    let (mut s, _, _) = queued_board();
    assert_eq!(reg_read(&mut s, ODSY_STATUS0), 0x1003_0000);

    push_fences(&mut s);
    assert_eq!(reg_read(&mut s, ODSY_STATUS0), 0x1002_0000, "not empty, still below the threshold");
    push_fences(&mut s);
    push_fences(&mut s);
    assert_eq!(s.fifo.level(), 18);
    assert_eq!(reg_read(&mut s, ODSY_STATUS0), 0x1000_0000);

    while s.fifo.level() > LOW_WATER_MARK {
        step(&mut s);
        assert_eq!(reg_read(&mut s, ODSY_STATUS0), 0x1000_0000);
    }
    step(&mut s);
    assert_eq!(reg_read(&mut s, ODSY_STATUS0), 0x1002_0000, "below the threshold once fewer than 16 wait");

    finish(&mut s);
    assert_eq!(reg_read(&mut s, ODSY_STATUS0), 0x1003_0000);
}

/// MODEL CONTRACT ONLY, no evidence for the hardware: the roles of `0x6b2`
/// are unknown. The level counts the FIFO's 32-bit words (a model choice): a
/// doubleword store fills two. The model reads status bit 15 set at or above
/// the written `0x6b2`, and reaching it raises flags bit 25; with no `0x6b2`
/// written it never reads set. The level stops at the model's 32,768-word
/// FIFO however many stores wait.
#[test]
fn the_level_counts_words_against_the_model_fifo_and_the_6b2_threshold() {
    let (mut s, _, _) = queued_board();
    let (high, flag) = (status0::FIFO_HIGH_WATER, intr::FIFO_HIGH_WATER);

    s.mmio_write(consts::cmd::FIFO, 0, 8);
    s.mmio_write(consts::cmd::FIFO, 0, 4);
    assert_eq!((s.fifo.stores(), s.fifo.level()), (2, 3));
    finish(&mut s);

    // No threshold yet: a FIFO filled to the brim never reads high.
    for _ in 0..cfifo::WORDS / 2 {
        s.mmio_write(consts::cmd::FIFO, 0, 8);
    }
    assert_eq!(s.fifo.level(), cfifo::WORDS);
    assert_eq!(reg_read(&mut s, ODSY_STATUS0) & high, 0);
    // Full: the next store still queues, and the level stays at the brim.
    s.mmio_write(consts::cmd::FIFO, 0, 4);
    assert_eq!(s.fifo.level(), cfifo::WORDS);
    finish(&mut s);

    // The 0x6b2 value the kernel writes, 576 words below the model's size.
    cmd64(&mut s, 0x6b2, 32192);
    finish(&mut s);
    reg_write(&mut s, intr::CLEAR, intr::FIFO_HIGH_WATER);
    for _ in 0..32190 / 2 {
        s.mmio_write(consts::cmd::FIFO, 0, 8);
    }
    assert_eq!(reg_read(&mut s, ODSY_STATUS0) & high, 0);
    assert_eq!(reg_read(&mut s, intr::FLAGS) & flag, 0);
    s.mmio_write(consts::cmd::FIFO, 0, 8);
    assert_eq!(s.fifo.level(), 32192);
    assert_eq!(reg_read(&mut s, ODSY_STATUS0) & high, high);
    assert_eq!(reg_read(&mut s, intr::FLAGS) & flag, flag);
    step(&mut s);
    assert_eq!(reg_read(&mut s, ODSY_STATUS0) & high, 0, "below the threshold again");
    finish(&mut s);
}

/// MODEL CONTRACT ONLY, no evidence for the hardware: the roles of `0x6b2` and
/// `0x6b3` are unknown. The model applies `0x6b2` to status bit 15 and only
/// stores `0x6b3`. With `0x6b2` = 12 and `0x6b3` = 8, eight waiting words
/// leave bit 15 clear; swapped, they set it. Flags bit 25 is covered by
/// `the_level_counts_words_against_the_model_fifo_and_the_6b2_threshold`.
///
/// QEMU: command-fifo-high-mark-follows-6b2.
#[test]
fn the_threshold_follows_6b2() {
    let (mut s, _, _) = queued_board();
    let high = status0::FIFO_HIGH_WATER;

    for (mark_6b2, word_6b3, expect) in [(12, 8, 0), (8, 12, high)] {
        cmd64(&mut s, 0x6b1, 256);
        cmd64(&mut s, 0x6b3, word_6b3);
        cmd64(&mut s, 0x6b2, mark_6b2);
        finish(&mut s);
        for _ in 0..4 {
            cmd64(&mut s, 0x6b0, 0);
        }
        assert_eq!(s.fifo.level(), 8);
        assert_eq!(reg_read(&mut s, ODSY_STATUS0) & high, expect, "0x6b2 = {mark_6b2}, 0x6b3 = {word_6b3}");
        finish(&mut s);
    }
}

/// MODEL CONTRACT ONLY: the FIFO's size is unknown and the model's size is
/// its own choice. Whatever `0x6b1` the set-up stores, the level clamps at
/// cfifo::WORDS however many stores wait.
///
/// QEMU: command-work-credit-depth-ignores-setup covers the credit depth.
#[test]
fn the_fifo_size_ignores_the_set_up_words() {
    for word_6b1 in [1u32, 256, 0x7fff_ffff] {
        let (mut s, _, _) = queued_board();

        cmd64(&mut s, 0x6b1, word_6b1);
        finish(&mut s);
        for _ in 0..cfifo::WORDS / 2 {
            cmd64(&mut s, 0x6b0, 0);
        }
        assert_eq!(s.fifo.level(), cfifo::WORDS, "0x6b1 = {word_6b1:#x}");
        // Full: the next store still queues, and the level stays at the brim.
        s.mmio_write(consts::cmd::FIFO, 0, 4);
        assert_eq!(s.fifo.level(), cfifo::WORDS, "0x6b1 = {word_6b1:#x}");
        finish(&mut s);
    }
}

/// MODEL CONTRACT ONLY, no evidence for the hardware: the threshold and the
/// edge are the model's choice. Low water is not raised while 16 or more
/// stores wait, and is raised, with its interrupt, as the level falls below
/// the threshold while stores still wait.
///
/// QEMU: command-work-low-water-after-drain.
#[test]
fn low_water_is_raised_as_the_queue_falls_below_the_mark() {
    let (mut s, rec, _) = queued_board();
    route_to_heart(&mut s);
    for _ in 0..3 {
        push_fences(&mut s);
    }
    reg_write(&mut s, intr::CLEAR, intr::FIFO_LOW_WATER);
    reg_write(&mut s, intr::ENABLE_SET, intr::FIFO_LOW_WATER);

    // 18 wait; two slices leave 16, still at the mark.
    step(&mut s);
    step(&mut s);
    assert!(rec.events(FLAG_VECTOR).is_empty());
    assert_eq!(reg_read(&mut s, intr::FLAGS) & intr::FIFO_LOW_WATER, 0);

    // The next leaves 15, below the mark.
    step(&mut s);
    assert_eq!(s.fifo.level(), LOW_WATER_MARK - 1);
    assert_eq!(reg_read(&mut s, intr::FLAGS) & intr::FIFO_LOW_WATER, intr::FIFO_LOW_WATER);
    assert_eq!(rec.events(FLAG_VECTOR), [true]);
}

/// MODEL CONTRACT ONLY: the FIFO's depth is unknown and the model's depth is
/// its own choice. The stored `0x6b1` and `0x6b2` values (256 and 32192, as
/// the kernels write them) do not widen the credits: only the oldest
/// `cfifo::DEPTH` waiting stores, the model's depth, have their credits back, so the
/// processor interface still holds back a writer that runs ahead of the slow
/// board.
///
/// QEMU: command-work-credit-depth-ignores-setup.
#[test]
fn the_fifo_set_up_does_not_widen_the_credit_depth() {
    let (mut s, pi) = credit_board();
    s.fifo.test_words.store(0, SeqCst);
    cmd64(&mut s, 0x6b1, 256);
    cmd64(&mut s, 0x6b3, 24144);
    cmd64(&mut s, 0x6b2, 32192);
    finish(&mut s);
    pi.0.lock().clear();

    s.fifo.test_words.store(1, SeqCst);
    let stores = cfifo::DEPTH + 8;
    for _ in 0..stores {
        s.mmio_write(consts::cmd::FIFO, 0, 8);
    }
    assert_eq!(s.fifo.stores(), stores);
    assert_eq!(pi.count("return"), cfifo::DEPTH as i64);
    finish(&mut s);
    assert_eq!(pi.settled(), stores as i64);
}

/// MODEL CHOICE: the FIFO holds far more than the host's credit window
/// (the VPro porting guide's millisecond of commands), so a writer that
/// fills the whole window, 256 doublewords on the IP30 and at most 256 on
/// the IP35, gets every credit back as its stores enter, with the board's
/// thread taking none of them, and owes nothing. A credit that waited on
/// the board's thread could miss the high-water handler's ten-microsecond
/// poll.
#[test]
fn the_credit_depth_is_deeper_than_the_host_window() {
    const LARGEST_HOST_WINDOW: u32 = 256;
    let (mut s, pi) = credit_board();
    for _ in 0..=LARGEST_HOST_WINDOW {
        s.mmio_write(consts::cmd::FIFO, 0, 8);
    }
    assert_eq!(s.fifo.stores(), LARGEST_HOST_WINDOW + 1, "the board took none");
    assert_eq!(pi.count("return"), LARGEST_HOST_WINDOW as i64 + 1);
    assert_eq!(pi.count("owe"), 0);
    finish(&mut s);
}

/// MODEL CHOICE: the credited depth is 16,384 stores, the smallest power of
/// two that holds the kernel's start-up program of about 12,291 words at
/// one word per store. With the board taking nothing, every one of the
/// first 16,384 counted stores is credited as it enters, and
/// the next is owed until the board takes a store.
#[test]
fn every_store_inside_the_credited_depth_is_credited_and_the_next_is_owed() {
    const DEPTH: u32 = 16_384;
    let (mut s, pi) = credit_board();
    for _ in 0..DEPTH {
        s.mmio_write(consts::cmd::FIFO, 0, 8);
    }
    assert_eq!(s.fifo.stores(), DEPTH, "the board took none");
    assert_eq!(pi.count("return"), DEPTH as i64);
    assert_eq!(pi.count("owe"), 0);
    s.mmio_write(consts::cmd::FIFO, 0, 8);
    assert_eq!(pi.count("return"), DEPTH as i64);
    assert_eq!(pi.count("owe"), 1, "the store past the depth owes its credit");
    finish(&mut s);
    assert_eq!(pi.settled(), DEPTH as i64 + 1);
}

/// A counted store whose credit is owed has the board's thread run without
/// its pause, so the credit comes back as soon as the stores ahead retire.
/// A guest that watches its credit count reads a register the bus does not
/// refuse, so nothing else hurries the thread; with the pause the kernel saw
/// its count fall below the bias in a burst of a few hundred stores and gave
/// the FIFO up as stuck. The stores inside the model's depth are credited
/// at once and do not hurry.
#[test]
fn a_counted_store_owed_its_credit_hurries_the_board_thread() {
    let (mut s, pi) = credit_board();
    for _ in 0..cfifo::DEPTH {
        s.mmio_write(consts::cmd::FIFO, 0, 8);
    }
    assert!(!s.fifo.take_hurry(), "a credited store leaves the pause alone");
    s.mmio_write(consts::cmd::FIFO, 0, 8);
    assert_eq!(pi.count("owe"), 1);
    assert!(s.fifo.take_hurry(), "an owed credit hurries the board's thread");
    finish(&mut s);

    // On the board's thread, with a pause far longer than the wait below:
    // only the hurry runs the burst in time.
    let stores = cfifo::DEPTH + 136;
    let h = BoardLocks::new(s);
    h.fifo.test_gap_ms.store(60_000, SeqCst);
    // credit_board holds the queue for stepping by hand.
    h.fifo.test_words.store(0, SeqCst);
    pi.0.lock().clear();
    h.work_start();
    // One credited store starts the thread, and the burst waits until the
    // thread is in its pause: a burst that lands before it would find the
    // request already made and prove nothing about waking a sleeper.
    h.write(consts::cmd::FIFO, 0, 8);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while h.fifo.test_parked.load(SeqCst) == 0 {
        assert!(std::time::Instant::now() < deadline, "the board's thread never reached its pause");
        std::thread::yield_now();
    }
    for _ in 1..stores {
        h.write(consts::cmd::FIFO, 0, 8);
    }
    // The last few stores may find the queue run dry and wait out a pause
    // of their own, but inside the depth they are credited at the store.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while pi.settled() != stores as i64 {
        assert!(std::time::Instant::now() < deadline, "the board's thread kept its pause: {} stores wait, {} of {stores} credits back", h.fifo.stores(), pi.settled());
        std::thread::yield_now();
    }
    assert!(pi.count("owe") > 0, "the burst ran past the model's depth");
    assert_eq!(pi.count("repay"), pi.count("owe"));
    // Cut a pause still under way short, so the stop does not wait it out.
    h.fifo.hurry();
    h.work_stop();
    assert_eq!(h.fifo.stores(), 0);
}

/// A stopping machine finishes the work waiting; a reset abandons it.
///
/// QEMU: command-work-stop-and-reset.
#[test]
fn stopping_finishes_the_waiting_work_and_a_reset_abandons_it() {
    let (mut s, _, ram) = queued_board();
    push_fences(&mut s);
    let h = BoardLocks::new(s);

    h.work_stop();
    assert!(!h.fifo.pending());
    assert_eq!(ram.half(HOST_WRITE_BASE + 6), 0xa5a5);

    let mut s = h.test_board();
    for w in [select(reg::HOST_WRITE), 4, 0x1234] {
        cmd(&mut s, w);
    }
    assert!(s.work_pending());
    s.reset();
    assert!(!s.work_pending());
    assert_eq!(ram.half(HOST_WRITE_BASE + 8), 0);
}

/// Two processors storing to the command FIFO at once (`[machine] cpus = 2`)
/// each find their stores come out in the order they made them, and none is
/// lost or run twice. The producers share no lock but the ring's own, so
/// one can find the ring's lock taken and have its store refused; it tries
/// again, as the bus's retry does.
#[test]
fn two_processors_storing_to_the_fifo_lose_nothing_and_keep_each_ones_order() {
    const PER: u64 = 30_000;
    let (s, _) = board("v12");
    let h = Arc::new(BoardLocks::new(s));
    let done = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let producers: Vec<_> = (0..2u64)
        .map(|id| {
            let (h, done) = (h.clone(), done.clone());
            std::thread::spawn(move || {
                // A panic in a push must end the test, not hang the consumer.
                let pushed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    for seq in 0..PER {
                        let size = if seq % 3 == 0 { 8 } else { 4 };
                        while !h.fifo.push(&h.front, consts::cmd::FIFO, id << 32 | seq, size, false) {
                            std::thread::yield_now();
                        }
                    }
                }));
                done.fetch_add(1, SeqCst);
                pushed.is_ok()
            })
        })
        .collect();

    let mut next = [0u64; 2];
    let mut run = Run::new();
    loop {
        h.fifo.take_run(&mut run);
        let n = run.stores().len();
        if n == 0 {
            if done.load(SeqCst) == 2 && h.fifo.stores() == 0 {
                break;
            }
            std::thread::yield_now();
            continue;
        }
        for q in run.stores() {
            let id = (q.value >> 32) as usize;
            assert_eq!(q.value & 0xffff_ffff, next[id], "processor {id}'s stores arrive in order");
            next[id] += 1;
        }
        h.fifo.retire(&h.front, &run, n);
    }
    for p in producers {
        assert!(p.join().unwrap(), "a push panicked");
    }
    assert_eq!(next, [PER, PER]);
    assert_eq!((h.fifo.stores(), h.fifo.level()), (0, 0));
}

/// The board's thread can take a store out of the ring the moment it lands,
/// before the processor's push has returned. The level already counts it
/// then: the retire takes off no more words than the level holds, so the
/// level reads empty rather than wrapping to a full FIFO, and the fall
/// from 20 words to none latches the low-water flag.
#[test]
fn a_store_run_before_its_push_returns_is_already_in_the_level() {
    let (s, _) = board("v12");
    let h = BoardLocks::new(s);
    h.front.lock().intr_flags &= !intr::FIFO_LOW_WATER;
    for seq in 0..20 {
        assert!(h.fifo.push(&h.front, consts::cmd::FIFO, seq, 4, false));
    }
    fn run_everything(fifo: &CommandFifo, front: &Mutex<HostInterface>) {
        let mut run = Run::new();
        fifo.take_run(&mut run);
        let n = run.stores().len();
        assert_eq!(n, 21, "the store being pushed is in the ring");
        fifo.retire(front, &run, n);
        assert_eq!((fifo.stores(), fifo.level()), (0, 0), "the run took out what the level counted");
    }
    *h.fifo.test_after_ring_push.lock() = Some(run_everything);
    assert!(h.fifo.push(&h.front, consts::cmd::FIFO, 20, 8, false));
    assert!(h.fifo.test_after_ring_push.lock().is_none(), "the consumer ran inside the push");
    assert_eq!((h.fifo.stores(), h.fifo.level()), (0, 0));
    assert_ne!(h.front.lock().intr_flags & intr::FIFO_LOW_WATER, 0, "the level fell through the mark");
}

/// The deep ring takes stores past the model's 32,768 words, and the guest
/// still reads the model's own FIFO: the level clamped at the brim, so
/// STATUS0 and the `0x106c` doubleword count read a full FIFO and the
/// high-water flag is up, never a count past the model's size. Level 0
/// returns only once the backlog has run.
#[test]
fn the_deep_ring_accepts_past_the_brim_and_the_guest_reads_a_full_fifo() {
    let (mut s, _, _) = queued_board();
    cmd64(&mut s, 0x6b2, 32192);
    finish(&mut s);
    reg_write(&mut s, intr::CLEAR, intr::FIFO_HIGH_WATER);
    assert_eq!(s.fifo.level(), 0);
    assert_ne!(reg_read(&mut s, ODSY_STATUS0) & status0::FIFO_EMPTY, 0, "idle reads empty");

    for _ in 0..cfifo::WORDS / 2 + 5000 {
        s.mmio_write(consts::cmd::FIFO, 0, 8);
    }
    assert!(s.fifo.queued_words() > cfifo::WORDS, "the ring took more than the model's FIFO");
    assert_eq!(s.fifo.level(), cfifo::WORDS);
    let status = reg_read(&mut s, ODSY_STATUS0);
    assert_ne!(status & status0::FIFO_HIGH_WATER, 0);
    assert_eq!(status & (status0::FIFO_EMPTY | status0::FIFO_LOW_WATER), 0);
    assert_ne!(reg_read(&mut s, intr::FLAGS) & intr::FIFO_HIGH_WATER, 0);
    assert_eq!((reg_read(&mut s, MEMORY_CONFIG) >> 7) & 0x1ff, 0x1ff, "the doubleword count is at its field's top");

    // Draining the backlog keeps the guest's view pinned at the brim until
    // the ring is under it, and level 0 only when everything has run.
    s.work_step(&mut Budget::stores(1000));
    assert_eq!(s.fifo.level(), cfifo::WORDS);
    finish(&mut s);
    assert_eq!(s.fifo.level(), 0);
    let status = reg_read(&mut s, ODSY_STATUS0);
    assert_eq!(status & status0::FIFO_HIGH_WATER, 0);
    assert_ne!(status & status0::FIFO_EMPTY, 0);
    assert_eq!((reg_read(&mut s, MEMORY_CONFIG) >> 7) & 0x1ff, 0);
}

/// A store is refused with the bus busy only when the host ring is full, not
/// at the model's 32,768 words, and the guest never reads past the brim.
#[test]
fn the_bus_is_busy_only_when_the_ring_is_full() {
    let (h, _rec) = threaded_board();
    let words = preamble_twice();
    start_long_slice(&h, &words, 3000);
    // The slice's own first word may have left the ring: count what waits.
    while h.fifo.stores() < RING_STORES {
        assert!(h.bus_write(consts::cmd::FIFO, 0, 4), "refused at {} stores, {} words", h.fifo.stores(), h.fifo.queued_words());
        assert!(h.fifo.level() <= cfifo::WORDS);
    }
    assert_eq!(h.fifo.stores(), RING_STORES);
    assert!(!h.bus_write(consts::cmd::FIFO, 0, 4), "full refuses");
    assert_eq!(h.fifo.stores(), RING_STORES);
    assert_eq!(h.fifo.level(), cfifo::WORDS);
    h.fifo.test_words.store(0, SeqCst);
    h.work_stop();
    assert_eq!(h.fifo.level(), 0, "idle reads zero");
}

/// The flow-control target word is stored, not routed, so a store to it
/// goes in at once while the FIFO holds queued work: the bus answers
/// ready, no refusal is counted, and the word reads back from the front.
#[test]
fn a_flow_control_target_store_does_not_wait_for_the_fifo() {
    let (h, _rec) = threaded_board();
    let words = preamble_twice();

    start_long_slice(&h, &words, 2000);
    assert!(h.bus_write(consts::cmd::FIFO, 0, 4));
    assert!(h.fifo.pending(), "work is queued behind the slice in flight");

    let word = 1 << 17 | 0x1b << 12 | 0xc8;
    assert!(h.bus_write(flctl::TARGET, word, 4), "the store is accepted at once");
    assert_eq!(h.front.lock().flctl_target as u64, word);
    assert_eq!(h.bus_read(flctl::TARGET, 4), Some(word));
    assert!(h.fifo.pending(), "it did not drain the FIFO");

    let mut report = Vec::new();
    h.refusals.report(&mut report).unwrap();
    let report = String::from_utf8(report).unwrap();
    assert!(report.contains("reason store-waits-fifo 0"), "{report}");
    assert!(!report.contains("offset 0x1074"), "{report}");
    h.work_stop();
}

/// Both processors can find room for one more store in a ring that has
/// room for one. The check and the push are not one step, so the second
/// push meets a full ring: exactly one store goes in, the other is
/// refused rather than lost or overrunning the ring, and the levels count
/// what is in it.
#[test]
fn two_processors_racing_for_the_last_place_in_the_ring_leave_one_store_in() {
    const ROUNDS: u32 = 24;
    let (s, _) = board("v12");
    let h = Arc::new(BoardLocks::new(s));
    // The ring has room for one more store when it holds RING_STORES - 1.
    let fill = |h: &BoardLocks| {
        for _ in 0..RING_STORES - 1 {
            assert!(h.fifo.push(&h.front, consts::cmd::FIFO, 0, 4, false));
        }
    };
    fill(&h);
    let start = Arc::new(std::sync::Barrier::new(3));
    let end = Arc::new(std::sync::Barrier::new(3));
    let accepted = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let racers: Vec<_> = (0..2)
        .map(|_| {
            let (h, start, end, accepted) = (h.clone(), start.clone(), end.clone(), accepted.clone());
            std::thread::spawn(move || {
                let mut panics = 0;
                for _ in 0..ROUNDS {
                    start.wait();
                    // A panic in a push must fail the round, not leave the others waiting.
                    let pushed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| h.fifo.push(&h.front, consts::cmd::FIFO, 0, 4, false)));
                    match pushed {
                        Ok(true) => {
                            accepted.fetch_add(1, SeqCst);
                        }
                        Ok(false) => {}
                        Err(_) => panics += 1,
                    }
                    end.wait();
                }
                panics
            })
        })
        .collect();

    let mut run = Run::new();
    for round in 0..ROUNDS {
        accepted.store(0, SeqCst);
        start.wait();
        end.wait();
        assert_eq!(accepted.load(SeqCst), 1, "round {round}: one place, one store");
        assert_eq!((h.fifo.stores(), h.fifo.queued_words()), (RING_STORES, RING_STORES));
        // Empty the ring, which also publishes the consumer's head to the
        // producers, and fill it again.
        while h.fifo.stores() != 0 {
            h.fifo.take_run(&mut run);
            let n = run.stores().len();
            h.fifo.retire(&h.front, &run, n);
        }
        fill(&h);
    }
    for r in racers {
        assert_eq!(r.join().unwrap(), 0, "a push panicked");
    }
}

/// Two processors are serialised against each other and not against the
/// board's thread: while one holds the processors' lock in the middle of an
/// access, a second processor's waiting read, command store and waiting
/// store all wait for it, and the board's thread, which never takes it, runs
/// the queue meanwhile.
#[test]
fn processors_wait_for_each_other_and_the_board_thread_does_not_wait_for_them() {
    let (h, _rec) = threaded_board();
    let words = preamble_twice();
    let held = h.test_submit();
    let started = Arc::new(AtomicU64::new(0));
    let done = Arc::new(AtomicU64::new(0));
    let mut others = Vec::new();
    for which in 0..3u32 {
        let (h, started, done) = (h.clone(), started.clone(), done.clone());
        others.push(std::thread::spawn(move || {
            started.fetch_add(1, SeqCst);
            match which {
                0 => while h.bus_read(cfifo::SELFTEST, 4).is_none() {},
                1 => assert!(h.bus_write(consts::cmd::FIFO, 0, 4)),
                _ => while !h.bus_write(ODSY_BUZZ_1, 1, 4) {},
            }
            done.fetch_add(1, SeqCst);
        }));
    }
    while started.load(SeqCst) != 3 {
        std::thread::yield_now();
    }
    std::thread::sleep(std::time::Duration::from_millis(150));
    assert_eq!(done.load(SeqCst), 0, "a processor ran an access under another's");
    // The board's thread takes no processor lock: it runs what was queued
    // before the lock was held. Nothing is queued here, so queue through the
    // unlocked ring and see it drain.
    for w in &words[..4] {
        assert!(h.fifo.push(&h.front, w.port, w.value, w.size as u32, false));
    }
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    while h.fifo.pending() {
        assert!(std::time::Instant::now() < deadline, "the board's thread did not run under the processors' lock");
        std::thread::yield_now();
    }
    assert_eq!(done.load(SeqCst), 0);
    drop(held);
    for t in others {
        t.join().unwrap();
    }
    assert_eq!(done.load(SeqCst), 3);
    h.work_stop();
}

/// A drain by an access, with the board's thread running, runs nothing: the
/// stores are the thread's to run, in order, and a second runner would be
/// two threads in the board's state.
#[test]
fn a_drain_with_the_board_thread_running_runs_no_store() {
    let (h, _rec) = threaded_board();
    let words = preamble_twice();

    start_long_slice(&h, &words, 1000);
    for w in &words[1..] {
        h.write(w.port, w.value, w.size);
    }
    let (stores, level) = (h.fifo.stores(), h.fifo.level());
    assert!(stores > 1);
    // The test stands in for a caller that drains while the thread runs;
    // the guard is what keeps it from running anything.
    h.test_board().work_drain(cfifo::SELFTEST);
    assert_eq!((h.fifo.stores(), h.fifo.level()), (stores, level), "the drain ran stores");
    h.fifo.test_words.store(0, SeqCst);
    h.work_stop();
    assert_eq!(h.fifo.level(), 0);
}

/// A command store the ring refuses while the board's thread runs is not
/// dropped and does not run around the thread: the caller retries the push
/// until the thread has made room, then it goes in once.
#[test]
fn a_refused_command_push_retries_behind_the_board_thread() {
    let (h, _rec) = threaded_board();
    let words = preamble_twice();

    start_long_slice(&h, &words, 1000);
    while h.fifo.stores() < RING_STORES {
        assert!(h.bus_write(consts::cmd::FIFO, 0, 4));
    }
    let stores = h.fifo.stores();
    let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let store = {
        let (h, done) = (h.clone(), done.clone());
        std::thread::spawn(move || {
            h.write(consts::cmd::FIFO, 0, 4);
            done.store(true, SeqCst);
        })
    };
    std::thread::sleep(std::time::Duration::from_millis(150));
    assert!(!done.load(SeqCst), "the store was dropped or ran around the board's thread");
    assert_eq!(h.fifo.stores(), stores);
    store.join().unwrap();
    assert!(h.fifo.level() <= cfifo::WORDS);
    h.fifo.test_words.store(0, SeqCst);
    h.work_stop();
    assert_eq!(h.fifo.level(), 0);
}
