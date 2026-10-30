//! The board's register contract.

use super::*;
use crate::irq::InterruptController;
use parking_lot::Mutex;
use std::sync::Arc;

// The register contract of the board in the Octane's graphics port.

const VPRO_WIDGET: u32 = 11;
const HEART_WIDGET: u64 = 8;
const HEART_PACKET: u64 = 0x80;
const TEST_VECTOR: u32 = 18;
// Aperture offsets: the identity and control words are the low halves.
const IDENTITY: u64 = 0x0004;
const MEMORY_CONFIG: u64 = 0x106c;
const CONTROL: u64 = 0x0024;
const ROUTE_DEST_UPPER: u64 = 0x1044;
const ROUTE_DEST_LOWER: u64 = 0x104c;
const ROUTE_VECTORS: u64 = 0x1054;
const INTR_CLEAR: u64 = 0x101c;
const INTR_ENABLE_SET: u64 = 0x1034;
const INTR_FIFO_LOW_WATER: u64 = 0x200;
const IDENTITY_DEFAULT: u32 = 0xac013047;

/// Records every (line, level) the board drives.
#[derive(Default)]
struct Recorder(Mutex<Vec<(u32, bool)>>);

impl InterruptController for Recorder {
    fn set_line(&self, line: u32, level: bool) {
        self.0.lock().push((line, level));
    }
}

impl Recorder {
    /// The level of `line` after the last event on it.
    fn level(&self, line: u32) -> bool {
        self.0.lock().iter().rev().find(|e| e.0 == line).map(|e| e.1).unwrap_or(false)
    }
}

fn board(product: &str) -> (Box<Board>, Arc<Recorder>) {
    let rec = Arc::new(Recorder::default());
    let s = Board::new(product, VPRO_WIDGET).expect("board");
    s.front.lock().irq = Some(rec.clone());
    (s, rec)
}

#[test]
fn widget_identity() {
    let (mut s, _) = board("v12");
    assert_eq!(s.mmio_read(IDENTITY, 4) as u32, IDENTITY_DEFAULT);
    assert_eq!(s.mmio_read(CONTROL, 4) as u32 & ODSY_CONTROL_WIDGET_MASK, VPRO_WIDGET);
}

#[test]
fn profile_identity_and_routing() {
    // (product, revision, memory size code)
    let cases = [("v6", 2u32, 0u32), ("v8", 2, 2), ("v10", 0xa, 0), ("v12", 0xa, 2)];

    for (name, revision, size_code) in cases {
        let (mut s, rec) = board(name);
        let identity = s.mmio_read(IDENTITY, 4) as u32;
        let memory = s.mmio_read(MEMORY_CONFIG, 4) as u32;

        assert_eq!(identity >> 28, revision, "{name}");
        assert_eq!(identity & 0x0fff_ffff, IDENTITY_DEFAULT & 0x0fff_ffff, "{name}");
        assert_eq!((memory >> 30) & 3, size_code, "{name}");
        assert_eq!(s.mmio_read(CONTROL, 4) as u32 & ODSY_CONTROL_WIDGET_MASK, VPRO_WIDGET);

        s.mmio_write(ROUTE_DEST_UPPER, HEART_WIDGET << 25, 4);
        s.mmio_write(ROUTE_DEST_LOWER, HEART_PACKET, 4);
        s.mmio_write(ROUTE_VECTORS, TEST_VECTOR as u64, 4);
        s.mmio_write(INTR_ENABLE_SET, INTR_FIFO_LOW_WATER, 4);
        assert!(rec.level(TEST_VECTOR), "{name}: vector raised");
        s.mmio_write(INTR_CLEAR, INTR_FIFO_LOW_WATER, 4);
        assert!(!rec.level(TEST_VECTOR), "{name}: vector lowered");
    }
}

/// A raised and a lowered board interrupt each reach HEART, setting and
/// clearing the vector's pending bit.
#[test]
fn interrupt_reaches_heart() {
    let (mut s, rec) = board("v12");

    s.mmio_write(ROUTE_DEST_UPPER, HEART_WIDGET << 25, 4);
    s.mmio_write(ROUTE_DEST_LOWER, HEART_PACKET, 4);
    s.mmio_write(ROUTE_VECTORS, TEST_VECTOR as u64, 4);
    assert!(!rec.level(TEST_VECTOR));

    // The FIFO is empty, so its low-water flag is already pending.
    s.mmio_write(INTR_ENABLE_SET, INTR_FIFO_LOW_WATER, 4);
    assert!(rec.level(TEST_VECTOR));

    s.mmio_write(INTR_CLEAR, INTR_FIFO_LOW_WATER, 4);
    assert!(!rec.level(TEST_VECTOR));
}

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
    h.front.lock().test_words = 1;
    h.board.lock().test_slice_sleep_ms = hold_ms;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    h.write(words[0].port, words[0].value, words[0].size);
    while !(h.front.lock().slice && h.board.is_locked()) {
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
fn work_thread_runs_queued_stores_in_order() {
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
        let level = h.front.lock().fifo_level();
        assert!(level >= ODSY_CFIFO_LOW_WATER, "level {level}: the first slice should still hold the queue");
        assert_eq!(h.front.lock().intr_flags & ODSY_INTR_FIFO_LOW_WATER, 0);
        // The status register does not wait: not empty, above low water.
        let status = h.read(STATUS0, 4) as u32;
        assert_eq!(status & (ODSY_STATUS0_FIFO_EMPTY | ODSY_STATUS0_FIFO_LOW_WATER), 0);
        assert_eq!(h.front.lock().fifo_level(), level);
        // All of that while the slice still held the board.
        assert!(h.board.is_locked() && h.front.lock().slice);
    }

    // Watch the queue drain on the board's thread, without draining it
    // from here: the flag is up exactly when the level is below the mark.
    loop {
        {
            let f = h.front.lock();
            let level = f.fifo_level();
            let flag = f.intr_flags & ODSY_INTR_FIFO_LOW_WATER != 0;
            assert_eq!(flag, level < ODSY_CFIFO_LOW_WATER, "level {level}");
            if level == 0 && !f.slice {
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
    assert!(h.front.lock().worker.is_none());
}

/// A read that waits for the work, issued while a long slice holds the
/// board, answers only after every store queued before it has run: the
/// self-test register reads OK, though the board's thread runs one
/// store per slice and most were still queued when the read began.
#[test]
fn waiting_read_sees_all_earlier_stores() {
    let (h, _rec) = threaded_board();
    let words = preamble_twice();

    start_long_slice(&h, &words, 300);
    for w in &words[1..] {
        h.write(w.port, w.value, w.size);
    }
    assert!(h.front.lock().fifo_level() > 1);
    let selftest = h.read(ODSY_CFIFO_SELFTEST, 4) as u32;
    assert_eq!(selftest, ODSY_CFIFO_SELFTEST_OK);
    assert_eq!(h.front.lock().fifo_level(), 0);
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
        // No work thread runs: hold the queue and step by hand.
        f.test_words = 1;
    }
    (s, pi)
}

/// Command work credits at FIFO entry: a counted store's credit
/// comes back when the board takes it into its FIFO, the oldest 64
/// waiting stores at once; each store behind them is owed, then
/// repaid as the work ahead runs and it moves up. A store that does
/// not wait is credited inside the store.
#[test]
fn counted_store_credits_at_fifo_entry() {
    let (mut s, pi) = credit_board();
    let stores = 72u32;

    for _ in 0..stores {
        s.mmio_write(ODSY_CMD_FIFO, 0, 8);
    }
    assert_eq!(s.front.lock().work_queue.len() as u32, stores);
    assert_eq!(s.front.lock().work_in_fifo, ODSY_CFIFO_DEPTH);
    assert_eq!(pi.count("return"), ODSY_CFIFO_DEPTH as i64);
    assert_eq!(pi.count("owe"), (stores - ODSY_CFIFO_DEPTH) as i64);
    assert_eq!(pi.count("repay"), 0);
    assert!(pi.0.lock().iter().all(|c| c.1 == VPRO_WIDGET));

    // One store runs per slice: the next one moves up and is repaid.
    s.work_step(&mut Budget::words(1));
    assert_eq!(pi.count("repay"), 1);
    assert_eq!(pi.settled(), ODSY_CFIFO_DEPTH as i64 + 1);

    s.work_step(&mut Budget::unlimited());
    assert!(s.front.lock().work_queue.is_empty());
    assert_eq!(pi.settled(), stores as i64);

    // A store that does not wait is credited inside the store.
    s.front.lock().test_words = 0;
    s.mmio_write(ODSY_CMD_FIFO, 0, 8);
    assert!(!s.work_pending());
    assert_eq!(pi.settled(), stores as i64 + 1);
    assert_eq!(pi.count("owe"), (stores - ODSY_CFIFO_DEPTH) as i64);
}

/// A reset abandons the waiting stores and cancels each owed credit
/// (one owe of -1 per owed store); stores not owed cancel nothing.
#[test]
fn reset_cancels_owed_credits() {
    let (mut s, pi) = credit_board();

    for _ in 0..ODSY_CFIFO_DEPTH + 8 {
        s.mmio_write(ODSY_CMD_FIFO, 0, 8);
    }
    assert_eq!(pi.count("owe"), 8);
    s.work_reset();
    assert!(s.front.lock().work_queue.is_empty());
    assert_eq!(s.front.lock().work_in_fifo, 0);
    assert_eq!(pi.count("owe"), 0);
    assert_eq!(pi.0.lock().iter().filter(|c| c.0 == "owe" && c.2 == -1).count(), 8);
    assert_eq!(pi.count("repay"), 0);
}

/// The processor interface's mark makes a store counted, once, and a
/// counted store with no link logs and returns nothing.
#[test]
fn processor_mark_counts_one_store() {
    let (mut s, pi) = credit_board();

    {
        let mut f = s.front.lock();
        f.test_counted_stores = 0;
        f.test_words = 0;
    }
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
fn pi_gfx_block_counts_and_starves() {
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

mod widget {
    //! The Octane VPro integration checks:
    //! the board answers through both widget-11 windows, the Crossbow
    //! carries its link only when it is fitted, and a routed interrupt
    //! sets and clears its vector in HEART's pending register.
    use super::*;
    use crate::dev::heart::{host_clock, Heart, HeartConfig, PIU_BASE};
    use crate::dev::heart_mem::HeartRam;
    use crate::dev::xbow::Xbow;

    const WIDGET: u8 = 11;
    const VECTOR: u32 = 18;

    fn board() -> (Arc<Heart>, Arc<Odyssey>) {
        let heart = Arc::new(Heart::new(HeartConfig::octane([128 << 20, 0, 0, 0]), host_clock()));
        let ram = Arc::new(HeartRam::try_new([128, 0, 0, 0]).expect("ram"));
        let v = Odyssey::new(WIDGET, "v12", heart.clone(), ram.dma_view()).expect("board");
        (heart, v)
    }

    fn pending(heart: &Heart) -> u64 {
        heart.read64(PIU_BASE + 0x10030).data
    }

    #[test]
    fn the_board_answers_through_both_widget_windows() {
        let (_heart, v) = board();
        for base in [widget_small_base(WIDGET), widget_large_base(WIDGET)] {
            assert_eq!(v.read32(base + 0x4).data, 0xac01_3047);
            assert_eq!(v.read32(base + 0x24).data & 0xf, WIDGET as u32);
        }
    }

    #[test]
    fn the_crossbow_carries_the_graphics_link_only_when_fitted() {
        let fitted = Xbow::new(4, (1 << 8) | (1 << 15) | (1 << WIDGET));
        let empty = Xbow::new(4, (1 << 8) | (1 << 15));
        assert!(fitted.carries(WIDGET));
        assert!(!empty.carries(WIDGET));
    }

    #[test]
    fn a_routed_interrupt_sets_and_clears_its_heart_vector() {
        let (heart, v) = board();
        let b = widget_small_base(WIDGET);
        v.write32(b + 0x1044, 8 << 25);
        v.write32(b + 0x104c, 0x80);
        v.write32(b + 0x1054, VECTOR);
        assert_eq!(pending(&heart) & 1 << VECTOR, 0);
        // The FIFO is empty, so its low-water flag is already pending.
        v.write32(b + 0x1034, 0x200);
        assert_ne!(pending(&heart) & 1 << VECTOR, 0);
        v.write32(b + 0x101c, 0x200);
        assert_eq!(pending(&heart) & 1 << VECTOR, 0);
    }
}
