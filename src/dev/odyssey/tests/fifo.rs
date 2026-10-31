//! The command FIFO: queueing, the board's thread, status, credits.

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
    assert!(words.len() as u32 > ODSY_CFIFO_LOW_WATER);
    words
}

/// Queue every FIFO store and have the first slice hold the board for
/// `hold_ms` as a long draw would; store the first word and return
/// once that slice holds the board lock.
fn start_long_slice(h: &BoardLocks, words: &[&CfifoPreamble], hold_ms: u32) {
    h.fifo.test_words.store(1, SeqCst);
    h.board.lock().test_slice_sleep_ms = hold_ms;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    h.write(words[0].port, words[0].value, words[0].size);
    while !(h.fifo.slice.load(SeqCst) && h.board.is_locked()) {
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
    // The rest go in while the first slice holds the board lock: only
    // the front lock is needed.
    for w in &words[1..] {
        h.write(w.port, w.value, w.size);
    }
    {
        let level = h.fifo.level();
        assert!(level >= ODSY_CFIFO_LOW_WATER, "level {level}: the first slice should still hold the queue");
        assert_eq!(h.front.lock().intr_flags & ODSY_INTR_FIFO_LOW_WATER, 0);
        // The status register does not wait: not empty, above low water.
        let status = h.read(STATUS0, 4) as u32;
        assert_eq!(status & (ODSY_STATUS0_FIFO_EMPTY | ODSY_STATUS0_FIFO_LOW_WATER), 0);
        assert_eq!(h.fifo.level(), level);
        // All of that while the slice still held the board.
        assert!(h.board.is_locked() && h.fifo.slice.load(SeqCst));
    }

    // Watch the queue drain on the board's thread, without draining it
    // from here: the flag is up exactly when the level is below the mark.
    loop {
        {
            let f = h.front.lock();
            let level = h.fifo.level();
            let flag = f.intr_flags & ODSY_INTR_FIFO_LOW_WATER != 0;
            assert_eq!(flag, level < ODSY_CFIFO_LOW_WATER, "level {level}");
            if level == 0 && !h.fifo.slice.load(SeqCst) {
                break;
            }
        }
        assert!(std::time::Instant::now() < deadline, "work thread did not drain the queue");
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    {
        let s = h.board.lock();
        assert!(s.cfifo_selftest_ready);
        assert_eq!(s.selftest_progress, 0);
    }
    assert!(rec.level(TEST_VECTOR));
    let status = h.read(STATUS0, 4) as u32;
    assert_eq!(
        status & (ODSY_STATUS0_FIFO_EMPTY | ODSY_STATUS0_FIFO_LOW_WATER),
        ODSY_STATUS0_FIFO_EMPTY | ODSY_STATUS0_FIFO_LOW_WATER
    );
    h.work_stop();
    assert!(h.fifo.thread.lock().is_none());
}

/// A read that waits for the work, issued while a long slice holds the
/// board, answers only after every store queued before it has run: the
/// self-test register reads OK, though the board's thread runs one
/// store per slice and most were still queued when the read began.
#[test]
fn a_read_that_waits_sees_every_earlier_store() {
    let (h, _rec) = threaded_board();
    let words = preamble_twice();

    start_long_slice(&h, &words, 300);
    for w in &words[1..] {
        h.write(w.port, w.value, w.size);
    }
    assert!(h.fifo.level() > 1);
    let selftest = h.read(ODSY_CFIFO_SELFTEST, 4) as u32;
    assert_eq!(selftest, ODSY_CFIFO_SELFTEST_OK);
    assert_eq!(h.fifo.level(), 0);
    h.work_stop();
}

/// A processor store that finds the board's 32,768-word FIFO full while
/// the board's thread runs is refused with the bus busy, and nothing runs
/// on the processor's thread: the level stays at the brim and the slice
/// in flight keeps the board. The retried store goes in, once, when the
/// board's thread has made room, and the level never passes the brim.
#[test]
fn a_full_fifo_holds_the_store_back_until_the_board_makes_room() {
    let (h, _rec) = threaded_board();
    let words = preamble_twice();

    start_long_slice(&h, &words, 2000);
    while h.fifo.level() + 2 <= ODSY_CFIFO_WORDS {
        assert!(h.bus_write(ODSY_CMD_FIFO, 0, 8));
    }
    while h.fifo.level() < ODSY_CFIFO_WORDS {
        assert!(h.bus_write(ODSY_CMD_FIFO, 0, 4));
    }
    let stores = h.fifo.stores();
    assert!(!h.bus_write(ODSY_CMD_FIFO, 0, 4), "a full FIFO refuses the store");
    assert!(!h.bus_write(ODSY_CMD_FIFO, 0, 8));
    assert_eq!((h.fifo.level(), h.fifo.stores()), (ODSY_CFIFO_WORDS, stores), "nothing ran on the processor's thread");
    assert!(h.fifo.slice.load(SeqCst) && h.board.is_locked(), "the board's thread still holds the board");

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    while !h.bus_write(ODSY_CMD_FIFO, 0, 8) {
        assert!(h.fifo.level() <= ODSY_CFIFO_WORDS);
        assert!(std::time::Instant::now() < deadline, "the board's thread made no room");
        std::thread::yield_now();
    }
    assert!(h.fifo.level() <= ODSY_CFIFO_WORDS);
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
    assert_eq!(h.bus_read(ODSY_CFIFO_SELFTEST, 4), None, "busy while the stores wait");
    assert!(!h.bus_write(ODSY_BUZZ_1, 0, 4), "a waiting store is busy too");
    assert_eq!(h.fifo.level(), level, "nothing ran on the processor's thread");

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    let selftest = loop {
        if let Some(v) = h.bus_read(ODSY_CFIFO_SELFTEST, 4) {
            break v as u32;
        }
        assert!(std::time::Instant::now() < deadline, "the board's thread did not run the stores");
        std::thread::yield_now();
    };
    assert_eq!(selftest, ODSY_CFIFO_SELFTEST_OK);
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
/// comes back when the board takes it into its FIFO, the oldest 64
/// waiting stores at once; each store behind them is owed, then
/// repaid as the work ahead runs and it moves up. A store that does
/// not wait is credited inside the store.
#[test]
fn a_counted_store_is_credited_as_it_enters_the_fifo() {
    let (mut s, pi) = credit_board();
    let stores = 72u32;

    for _ in 0..stores {
        s.mmio_write(ODSY_CMD_FIFO, 0, 8);
    }
    // Credits count stores; the level counts words, two a doubleword.
    assert_eq!(s.fifo.stores(), stores);
    assert_eq!(s.fifo.level(), 2 * stores);
    assert_eq!(s.fifo.in_fifo(), ODSY_CFIFO_DEPTH);
    assert_eq!(pi.count("return"), ODSY_CFIFO_DEPTH as i64);
    assert_eq!(pi.count("owe"), (stores - ODSY_CFIFO_DEPTH) as i64);
    assert_eq!(pi.count("repay"), 0);
    assert!(pi.0.lock().iter().all(|c| c.1 == VPRO_WIDGET));

    // One store runs per slice: the next one moves up and is repaid.
    s.work_step(&mut Budget::stores(1));
    assert_eq!(pi.count("repay"), 1);
    assert_eq!(pi.settled(), ODSY_CFIFO_DEPTH as i64 + 1);

    s.work_step(&mut Budget::unlimited());
    assert!(!s.fifo.pending());
    assert_eq!(pi.settled(), stores as i64);

    // A store that does not wait is credited inside the store.
    s.fifo.test_words.store(0, SeqCst);
    s.mmio_write(ODSY_CMD_FIFO, 0, 8);
    assert!(!s.work_pending());
    assert_eq!(pi.settled(), stores as i64 + 1);
    assert_eq!(pi.count("owe"), (stores - ODSY_CFIFO_DEPTH) as i64);
}

/// A run of stores the board's thread has looked at and then runs leaves
/// the FIFO together. The stores run in order; the level, the store count
/// and the credits end where taking the stores out one by one leaves them,
/// including when a counted store joins the queue after the run was taken.
#[test]
fn a_run_retires_with_the_levels_and_credits_of_one_by_one() {
    fn queue(s: &mut Board) {
        s.front.lock().test_counted_stores = 0;
        for _ in 0..70 {
            s.mmio_write(ODSY_CMD_FIFO, 0, 8);
        }
    }
    fn counted_join(s: &mut Board) {
        s.front.lock().test_counted_stores = 1;
        for _ in 0..10 {
            s.mmio_write(ODSY_CMD_FIFO, 0, 8);
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
    assert_eq!(run.fifo.stores(), 80);
    run.fifo.retire(&run.front, &taken, RUN_MAX);

    assert_eq!(run.fifo.stores(), one.fifo.stores());
    assert_eq!(run.fifo.level(), one.fifo.level());
    assert_eq!(pi_run.count("repay"), pi_one.count("repay"));
    assert_eq!(pi_run.count("return"), pi_one.count("return"));
    assert_eq!(pi_run.count("owe"), pi_one.count("owe"));
    assert!(pi_run.count("repay") > 0, "the run moved owed stores up");
    // With a counted store waiting the next run is one store, as before.
    run.fifo.take_run(&mut taken);
    assert_eq!(taken.stores().len(), 1);
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

    for _ in 0..ODSY_CFIFO_DEPTH + 8 {
        s.mmio_write(ODSY_CMD_FIFO, 0, 8);
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
    s.mmio_write(ODSY_CMD_FIFO, 0, 8);
    assert_eq!(pi.settled(), 0);

    let previous = pi_gfx_mark_store(true);
    s.mmio_write(ODSY_CMD_FIFO, 0, 8);
    pi_gfx_unmark_store(previous);
    assert_eq!(pi.0.lock().as_slice(), &[("return", VPRO_WIDGET, 1)]);

    // One-shot: the mark was taken by the first store.
    s.mmio_write(ODSY_CMD_FIFO, 0, 8);
    assert_eq!(pi.settled(), 1);

    // The Octane: no link, so the credit is dropped with a log line.
    s.front.lock().pi = None;
    pi_gfx_mark_store(true);
    s.mmio_write(ODSY_CMD_FIFO, 0, 8);
    assert_eq!(pi.settled(), 1);
    assert!(!pi_gfx_take_mark());
}

/// The register block's pure state (PiGfx).
#[test]
fn the_credit_block_starves_a_widget_that_runs_out_of_credits() {
    let mut g = PiGfx::default();

    g.write(SGI_PI_GFX_PAGE, SGI_PI_GFX_PAGE_ENABLE | 0x2_0000);
    g.write(SGI_PI_GFX_CREDIT, 3);
    g.write(SGI_PI_GFX_BIAS, 2);
    assert!(g.covers(0, 0x2_1000));
    assert!(!g.covers(0, 0x4_0000));
    assert!(g.count_store(0, 0x2_0000, "t"));
    assert!(!g.take_starved(0, "t"));
    assert!(g.count_store(0, 0x2_0000, "t"));
    assert!(g.take_starved(0, "t"));
    assert!(!g.take_starved(0, "t"));
    g.add_credits(0, 5, "t");
    assert_eq!(g.read(SGI_PI_GFX_CREDIT), 6);
    assert!(!g.starved[0]);
}


/// The low-water mark, in waiting words.
const LOW_WATER_MARK: u32 = ODSY_CFIFO_LOW_WATER;
const HOST_WRITE_BASE: u64 = 0xb70000;

/// A board whose stores wait in the queue until a test steps them one
/// store a slice, with RAM for the fences they write.
fn queued_board() -> (Box<Board>, Arc<Recorder>, Arc<Ram>) {
    let (mut s, rec, ram) = board_with_ram();
    reg_write(&mut s, ODSY_HOST_WRITE_ADDR_UPPER, 0);
    reg_write(&mut s, ODSY_HOST_WRITE_ADDR_LOWER, HOST_WRITE_BASE as u32);
    s.fifo.test_words.store(1, SeqCst);
    (s, rec, ram)
}

/// Two host-write fences, slot 2 then slot 3, each three words.
fn push_fences(s: &mut Board) {
    for (slot, token) in [(2, 0x5a00), (3, 0xa5a5)] {
        cmd(s, select(ODSY_REG_HOST_WRITE));
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
    reg_write(&mut s, ODSY_CTX_SHADOW_ADDR_UPPER, 0);
    reg_write(&mut s, ODSY_CTX_SHADOW_ADDR_LOWER, base as u32);
    s.mmio_write(ODSY_CTX_SWITCH_PORT, 0x0000_0700_0001_0001, 8);

    let done = |ram: &Ram| u32::from_be_bytes(ram.get(base + ODSY_CTX_SWITCH_DONE_OFFSET, 4).try_into().unwrap());
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

    reg_read(&mut s, MEMORY_CONFIG);
    assert!(!s.work_pending());
    assert_eq!(ram.half(HOST_WRITE_BASE + 6), 0xa5a5);
}

/// While stores wait the FIFO is neither empty nor below low water in the
/// status the kernel's credit flow control reads before it sleeps on the
/// low-water interrupt; it reads ready again once they have run.
///
/// QEMU: command-work-status-shows-waiting-stores.
#[test]
fn the_status_shows_waiting_stores_until_they_have_run() {
    let (mut s, _, _) = queued_board();
    assert_eq!(reg_read(&mut s, ODSY_STATUS0), 0x1003_0000);

    push_fences(&mut s);
    assert_eq!(reg_read(&mut s, ODSY_STATUS0), 0x1002_0000, "not empty, still below the mark");
    push_fences(&mut s);
    push_fences(&mut s);
    assert_eq!(s.fifo.level(), 18);
    assert_eq!(reg_read(&mut s, ODSY_STATUS0), 0x1000_0000);

    while s.fifo.level() > LOW_WATER_MARK {
        step(&mut s);
        assert_eq!(reg_read(&mut s, ODSY_STATUS0), 0x1000_0000);
    }
    step(&mut s);
    assert_eq!(reg_read(&mut s, ODSY_STATUS0), 0x1002_0000, "below the mark once fewer than 16 wait");

    finish(&mut s);
    assert_eq!(reg_read(&mut s, ODSY_STATUS0), 0x1003_0000);
}

/// The level counts the FIFO's 32-bit words: a doubleword store fills two.
/// Status bit 15 reads set at or above the kernel's `0x6b2` mark, and
/// reaching the mark raises flags bit 25; with no mark set up it never
/// reads set. A store that would overfill the board's 32,768-word FIFO is
/// not queued: the storing caller runs the queue and then the store.
#[test]
fn the_level_counts_words_against_the_board_fifo_and_its_high_mark() {
    let (mut s, _, _) = queued_board();
    let (high, flag) = (ODSY_STATUS0_FIFO_HIGH_WATER, ODSY_INTR_FIFO_HIGH_WATER);

    s.mmio_write(ODSY_CMD_FIFO, 0, 8);
    s.mmio_write(ODSY_CMD_FIFO, 0, 4);
    assert_eq!((s.fifo.stores(), s.fifo.level()), (2, 3));
    finish(&mut s);

    // No mark yet: a FIFO filled to the brim never reads high.
    for _ in 0..ODSY_CFIFO_WORDS / 2 {
        s.mmio_write(ODSY_CMD_FIFO, 0, 8);
    }
    assert_eq!(s.fifo.level(), ODSY_CFIFO_WORDS);
    assert_eq!(reg_read(&mut s, ODSY_STATUS0) & high, 0);
    // Full: the next store runs everything waiting, then itself.
    s.mmio_write(ODSY_CMD_FIFO, 0, 4);
    assert!(!s.work_pending());

    // The kernel's set-up mark, 576 words below full.
    cmd64(&mut s, 0x6b2, 32192);
    finish(&mut s);
    reg_write(&mut s, ODSY_INTR_CLEAR, ODSY_INTR_FIFO_HIGH_WATER);
    for _ in 0..32190 / 2 {
        s.mmio_write(ODSY_CMD_FIFO, 0, 8);
    }
    assert_eq!(reg_read(&mut s, ODSY_STATUS0) & high, 0);
    assert_eq!(reg_read(&mut s, ODSY_INTR_FLAGS) & flag, 0);
    s.mmio_write(ODSY_CMD_FIFO, 0, 8);
    assert_eq!(s.fifo.level(), 32192);
    assert_eq!(reg_read(&mut s, ODSY_STATUS0) & high, high);
    assert_eq!(reg_read(&mut s, ODSY_INTR_FLAGS) & flag, flag);
    step(&mut s);
    assert_eq!(reg_read(&mut s, ODSY_STATUS0) & high, 0, "below the mark again");
    finish(&mut s);
}

/// The kernel's credit flow control clears low water, finds the FIFO above
/// its mark, enables the source and sleeps until the interrupt. Low water
/// is not raised while 16 or more stores wait, and is raised, with its
/// interrupt, as the level falls below the mark while stores still wait.
///
/// QEMU: command-work-low-water-after-drain.
#[test]
fn low_water_is_raised_as_the_queue_falls_below_the_mark() {
    let (mut s, rec, _) = queued_board();
    route_to_heart(&mut s);
    for _ in 0..3 {
        push_fences(&mut s);
    }
    reg_write(&mut s, ODSY_INTR_CLEAR, ODSY_INTR_FIFO_LOW_WATER);
    reg_write(&mut s, ODSY_INTR_ENABLE_SET, ODSY_INTR_FIFO_LOW_WATER);

    // 18 wait; two slices leave 16, still at the mark.
    step(&mut s);
    step(&mut s);
    assert!(rec.events(FLAG_VECTOR).is_empty());
    assert_eq!(reg_read(&mut s, ODSY_INTR_FLAGS) & ODSY_INTR_FIFO_LOW_WATER, 0);

    // The next leaves 15, below the mark.
    step(&mut s);
    assert_eq!(s.fifo.level(), LOW_WATER_MARK - 1);
    assert_eq!(reg_read(&mut s, ODSY_INTR_FLAGS) & ODSY_INTR_FIFO_LOW_WATER, ODSY_INTR_FIFO_LOW_WATER);
    assert_eq!(rec.events(FLAG_VECTOR), [true]);
}

/// The FIFO's size from the kernel's set-up does not widen the credits:
/// after the set-up (256 tiles, 32,768 words) only the oldest 64 waiting
/// stores have their credits back, so the processor interface still holds
/// back a writer that runs ahead of the slow board.
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
    for _ in 0..72 {
        s.mmio_write(ODSY_CMD_FIFO, 0, 8);
    }
    assert_eq!(s.fifo.stores(), 72);
    assert_eq!(pi.count("return"), ODSY_CFIFO_DEPTH as i64);
    finish(&mut s);
    assert_eq!(pi.settled(), 72);
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

    let mut s = h.board.lock();
    for w in [select(ODSY_REG_HOST_WRITE), 4, 0x1234] {
        cmd(&mut s, w);
    }
    assert!(s.work_pending());
    s.reset();
    assert!(!s.work_pending());
    assert_eq!(ram.half(HOST_WRITE_BASE + 8), 0);
}
