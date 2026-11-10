//! The command FIFO between the processor and the Host Interface Processor's
//! thread: stores queue in the shared ring, a full ring answers the bus busy,
//! registers that depend on the stream wait for it, and the registers a driver
//! polls while the pipe works do not.
//!
//! The cases that fill the ring hold the pipe's lock, so the thread cannot take
//! a run out of it; nothing here waits on a clock.

use super::*;
use crate::traits::Device;

const PORT_DATA: u64 = 0x8003f0;
const CREDIT: u64 = 0x800208;
const ACTIVITY: u64 = 0x8003a8;
const INT_ENABLE: u64 = 0x800230;
const INT_CAUSE: u64 = 0x800238;
const MAILBOX: u64 = 0x800240;
const MAPRAM: u64 = 0x100000;
const EEPROM_OPERATION: u32 = 0x1ec;
const OP_WRITE: u32 = 1;
const OP_READ: u32 = 2;
const OP_REPLY_BUFFER: u32 = 3;
const OP_CHECK: u32 = 4;
const OP_IDENTIFY: u32 = 5;
const TYPE_IMAGE: u32 = 0;
const TYPE_HEADER: u32 = 2;
const FLASH_HEADER: u32 = 0x7ff00;
const HEADER_STATE: usize = 7;
const INTERNAL_FLASH: u32 = 0x800000;
const REPLY_PAGE: u64 = 0x32_0000;

/// Textport colour selects: each is one job, in the order stored.
fn colours(n: u32) -> Vec<u32> {
    (0..n).flat_map(|i| [1, 2, i]).collect()
}

fn indices(jobs: &[String]) -> Vec<u32> {
    jobs.iter()
        .map(|j| u32::from_str_radix(j.split('[').nth(1).unwrap().split(']').next().unwrap(), 16).unwrap())
        .collect()
}

/// Stores go into the ring and the thread runs them in the order the
/// processor stored them; a read the stream feeds waits until it has.
#[test]
fn stores_run_in_order_on_the_thread() {
    let (f, board) = threaded();
    let n = 20_000;
    for w in colours(n) {
        f.w(COMMAND_PORT, w);
    }
    // A register the stream feeds: busy until the ring is empty and the thread
    // idle.
    f.r(PORT_DATA);
    let jobs = f.jobs();
    assert_eq!(jobs.len(), n as usize);
    assert_eq!(indices(&jobs), (0..n).collect::<Vec<_>>());
    drop(board);
}

/// A full ring answers a store busy: it never spins on the processor's thread.
/// The processor retries it, and once the thread has made room every store is
/// run, none twice and none lost.
#[test]
fn a_full_ring_answers_busy_never_spins() {
    let (f, _board) = threaded();
    let held = f.kona.shared.pipe.lock();
    let mut pushed = 0u32;
    let mut busy = false;
    for i in 0..70_000u32 {
        // Stray words: each is dropped and none opens a command.
        match f.kona.bus_write(COMMAND_PORT, 4, 0x1000 + i as u64) {
            BUS_OK => pushed += 1,
            BUS_BUSY => {
                busy = true;
                break;
            }
            other => panic!("status {other}"),
        }
    }
    assert!(busy, "{pushed} stores went in and the ring never filled");
    assert!(pushed > 60_000);
    // The same store is refused again while the thread has not made room.
    assert_eq!(f.kona.bus_write(COMMAND_PORT, 4, 0x1000), BUS_BUSY);
    // A register the stream feeds is busy too, and a polled one is not.
    assert_eq!(f.kona.bus_read(PORT_DATA, 4).0, BUS_BUSY);
    assert_eq!(f.kona.bus_read(CREDIT, 4), (BUS_OK, 0x7fe), "a full FIFO reads full");
    drop(held);
    f.w(COMMAND_PORT, 1);
    f.w(COMMAND_PORT, 9);
    f.w(COMMAND_PORT, 0);
    assert_eq!(f.r(PORT_DATA), 0);
    assert_eq!(f.r(MAILBOX), 0x1234abcd, "the probe after the flood was framed");
    assert_eq!(f.r(CREDIT), 0);
}

/// The registers a driver reads and acknowledges while the pipe works are
/// served at once, with stores behind the thread; every other read waits.
#[test]
fn polled_registers_do_not_wait_for_the_stream() {
    let (f, _board) = threaded();
    let held = f.kona.shared.pipe.lock();
    for w in [1, 9, 0] {
        retry(|| f.kona.bus_write(COMMAND_PORT, 4, w as u64) == BUS_OK);
    }
    // Served at once: the cause, the mailbox, the enable, the activity latch
    // (the pipe is busy), the credit register (words waiting) and the widget id.
    for reg in [INT_CAUSE, MAILBOX, INT_ENABLE, ACTIVITY, CREDIT, 0x04] {
        assert_eq!(f.kona.bus_read(reg, 4).0, BUS_OK, "register {reg:#x}");
    }
    assert_eq!(f.kona.bus_read(ACTIVITY, 4).1, 4);
    assert_eq!(f.kona.bus_read(CREDIT, 4).1, 4, "three words wait, rounded up to two");
    // Everything else reads what the stream writes or changes what it reads.
    for reg in [PORT_DATA, MAPRAM, 0x24, 0x98, 0x800220] {
        assert_eq!(f.kona.bus_read(reg, 4).0, BUS_BUSY, "register {reg:#x}");
    }
    // A write that waits for the stream joins it, in order, behind the words before it.
    assert_eq!(f.kona.bus_write(0x24, 4, 1), BUS_OK);
    assert_eq!(f.kona.bus_write(MAPRAM, 4, 1), BUS_OK, "map RAM stores join the stream");
    drop(held);
    f.r(PORT_DATA);
    assert_eq!(f.r(MAILBOX), 0x1234abcd);
    assert_eq!(f.r(MAPRAM), 1);
}

/// A map RAM store with nothing ahead of it is applied at once, with the
/// thread not running: the driver's word-by-word test of the map RAM then
/// costs no hand-off. One behind a queued store still waits its turn.
#[test]
fn a_map_ram_store_with_an_empty_stream_is_served_at_once() {
    let f = Fixture::idle(KonaConfig::ir2(1).unwrap(), Recorder::default().hook());
    assert!(!f.kona.is_running());
    assert_eq!(f.kona.bus_write(MAPRAM, 4, 0x1000_0001), BUS_OK);
    assert_eq!(f.kona.bus_read(MAPRAM, 4), (BUS_OK, 0x1000_0001));
    assert_eq!(f.kona.bus_write(MAPRAM + 4, 4, 0x1000_0002), BUS_OK);
    assert_eq!(f.kona.bus_read(MAPRAM + 4, 4), (BUS_OK, 0x1000_0002));
    // A command word is in the ring, so the next map RAM store queues behind
    // it, and a read waits.
    assert_eq!(f.kona.bus_write(COMMAND_PORT, 4, 1), BUS_OK);
    assert_eq!(f.kona.bus_write(MAPRAM + 8, 4, 7), BUS_OK);
    assert_eq!(f.kona.bus_read(MAPRAM + 8, 4).0, BUS_BUSY);
}

/// The low-watermark cause rises once the thread has run its work, when the
/// driver enabled it.
#[test]
fn the_low_watermark_cause_follows_the_work() {
    let (f, _board) = threaded();
    f.w(0x34, 0x98000);
    f.w(0x3c, 0x180_0090);
    f.w(0x5c, 0x3e2210);
    f.w(INT_ENABLE, 2);
    for w in colours(100) {
        f.w(COMMAND_PORT, w);
    }
    retry(|| f.r(INT_CAUSE) & 2 == 2);
    f.r(PORT_DATA);
    let sent = f.host.sent();
    assert!(sent.iter().any(|p| p.vector == 0x10), "{sent:?}");
}

/// Without the thread the same stores run in the store, so a test sees the
/// effect at once.
#[test]
fn without_the_thread_a_store_runs_in_the_store() {
    let f = Fixture::hooked();
    f.cmd(&colours(3));
    assert_eq!(f.jobs().len(), 3);
}

/// An 8-byte store to the command port is two words, high first, each as a
/// 4-byte store at the same offset; any other 8-byte store is two word stores
/// at the offset and the next. An 8-byte load is two word loads.
#[test]
fn a_doubleword_is_two_words() {
    for threaded in [false, true] {
        let (f, _board) = if threaded {
            let (f, b) = self::threaded();
            (f, Some(b))
        } else {
            (Fixture::hooked(), None)
        };
        f.w64(COMMAND_PORT, 1 << 32 | 9);
        f.w(COMMAND_PORT, 0);
        f.r(PORT_DATA);
        assert_eq!(f.r(MAILBOX), 0x1234abcd);
        f.w64(MAPRAM, 0x11 << 32 | 0x22);
        assert_eq!(f.r(MAPRAM), 0x11);
        assert_eq!(f.r(MAPRAM + 4), 0x22);
        let mut v = 0;
        retry(|| {
            let (s, d) = f.kona.bus_read(MAPRAM, 8);
            v = d;
            s == BUS_OK
        });
        assert_eq!(v, 0x11 << 32 | 0x22);
        // A doubleword that carries a command's last word and the next one's
        // selector, as the GL client's stores do.
        f.r(PORT_DATA);
        f.take_jobs();
        f.w64(COMMAND_PORT, 1 << 32 | 2);
        f.w64(COMMAND_PORT, 7 << 32 | 1);
        f.w64(COMMAND_PORT, 6 << 32 | 3);
        f.w64(COMMAND_PORT, 4 << 32);
        f.r(PORT_DATA);
        // The last doubleword's low word is padding after the command: a word
        // that opens nothing, which the back end is told of.
        assert_eq!(
            f.take_jobs(),
            ["Textport 0x2 [7] packed=0x0 handled=false", "Textport 0x6 [3, 4] packed=0x0 handled=false", "dropped 0x0"]
        );
    }
}

/// Stopping the thread runs the stores it had not taken, so a stopped board's
/// state is whole.
#[test]
fn stop_runs_what_is_left() {
    let (f, board) = threaded();
    let n = 3000;
    for w in colours(n) {
        f.w(COMMAND_PORT, w);
    }
    board.stop();
    assert!(!board.is_running());
    assert_eq!(f.jobs().len(), n as usize);
    // Stores after the stop wait in the ring until the thread starts again.
    f.cmd(&[1, 2, 99]);
    assert_eq!(f.jobs().len(), n as usize);
    board.start();
    f.settle();
    assert_eq!(f.jobs().len(), n as usize + 1);
}

/// A reset while the thread runs leaves it running, with the stream cleared.
#[test]
fn a_reset_restarts_the_thread() {
    let (f, board) = threaded();
    f.cmd(&[0x1f1, 0x1000_0000]);
    f.kona.power_on();
    assert!(board.is_running());
    f.cmd(&[1, 9, 0]);
    f.r(PORT_DATA);
    assert_eq!(f.r(MAILBOX), 0x1234abcd);
    assert_eq!(f.r(INT_CAUSE) & 0x10, 0x10);
}

/// A back end whose job holds the thread, as a drain on busy tiles does, until
/// the test lets it go.
fn hold(inside: Arc<std::sync::atomic::AtomicBool>, release: Arc<std::sync::atomic::AtomicBool>) -> TestHook {
    use std::sync::atomic::Ordering::SeqCst;
    TestHook {
        job: Box::new(move |_, _| {
            inside.store(true, SeqCst);
            while !release.load(SeqCst) {
                std::thread::yield_now();
            }
            Some(Reply::Done)
        }),
        events: Arc::default(),
        on_attach: None,
    }
}

/// A reset that arrives while the thread is inside a long job waits for it, as
/// it would for a drain, and then clears the stream and restarts the thread;
/// nothing deadlocks and nothing from before the reset survives it.
#[test]
fn a_reset_while_a_job_holds_the_thread_waits_and_then_proceeds() {
    use std::sync::atomic::{AtomicBool, Ordering::SeqCst};
    let (inside, release) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
    let host = TestHost::new();
    let kona = Kona::new(test_cfg(), host.clone(), host.clone(), crate::dev::kona::tests::test_status(), None).unwrap();
    kona.shared.pipe.lock().backend.hook = Some(hold(inside.clone(), release.clone()));
    let f = Fixture::of(host, kona);
    retry(|| f.kona.bus_write(COMMAND_PORT, 4, 0x1000) == BUS_OK);
    retry(|| inside.load(SeqCst));
    let (done, k) = (Arc::new(AtomicBool::new(false)), f.kona.clone());
    let flag = done.clone();
    let resetter = std::thread::spawn(move || {
        crate::traits::Resettable::power_on(&*k);
        flag.store(true, SeqCst);
    });
    std::thread::sleep(std::time::Duration::from_millis(100));
    assert!(!done.load(SeqCst), "the reset waits for the job");
    release.store(true, SeqCst);
    resetter.join().unwrap();
    assert!(f.kona.is_running(), "the thread runs again");
    f.settle();
}

/// An 8-byte store is all or nothing: refused busy it has done none of its
/// words, so the processor's retry does not repeat the first.
#[test]
fn a_busy_doubleword_has_no_effect() {
    let (f, _board) = threaded();
    let held = f.kona.shared.pipe.lock();
    // Fill the ring, so no store has room.
    while f.kona.bus_write(COMMAND_PORT, 4, 0x1000) == BUS_OK {}
    assert_eq!(f.kona.bus_write(0x24, 8, 7 << 32 | 9), BUS_BUSY);
    assert_eq!(f.kona.bus_write(MAPRAM + 0x100, 8, 0xffff_ffff_0000_0001), BUS_BUSY);
    drop(held);
    f.settle();
    assert_eq!(f.r(0x24), 9 << 22, "the busy store left widget control alone");
    f.w64(0x24, 7 << 32 | 9);
    assert_eq!(f.r(0x24), 7);
    f.w64(MAPRAM + 0x100, 0xffff_ffff_0000_0001);
    assert_eq!(f.r(MAPRAM + 0x100), 0xffff_ffff);
}

/// What the graphics install tool needs of the board, for the install tool's
/// the install tool: the credit register reads drained before every
/// command word; the check, header read and identify answer with a complete
/// reply (word 0 stored last); a write group is acknowledged with a non-zero
/// status; the flash gives back byte for byte what was written, in 0x2000-byte
/// reads, and the header words survive; the identify word is the same every
/// time. Run on the thread, as the tool runs against a machine: every word
/// waits for the credit register first and every reply is polled for.
#[test]
fn the_install_tool_can_program_and_verify_the_flash() {
    let (f, _board) = threaded();
    // Before every command word: the masked level below 0x5dd.
    let word = |w: u32| {
        retry(|| f.r(CREDIT) & 0xffe < 0x5dd);
        f.w(COMMAND_PORT, w);
    };
    let words = |ws: &[u32]| ws.iter().for_each(|w| word(*w));
    // Word 0 cleared by the tool, then polled until the board stores it last.
    let request = |ws: &[u32]| -> u64 {
        f.host.put(REPLY_PAGE, &[0; 4]);
        words(ws);
        retry(|| f.host.word(REPLY_PAGE) != 0);
        REPLY_PAGE
    };
    let data = |at: u64, n: u64| -> Vec<u32> { (0..n).map(|i| f.reply_data(at, i)).collect() };

    // Unmanaged path: the reply buffer from the two address words the kernel's
    // mapping returned and the length.
    words(&[EEPROM_OPERATION, OP_REPLY_BUFFER, 0x1000_0000 | (REPLY_PAGE >> 12) as u32, (REPLY_PAGE & 0xfff) as u32, 0x4000]);

    // The check: word 1 not negative, word 2 non-zero.
    let at = request(&[EEPROM_OPERATION, OP_CHECK, 0]);
    assert!((f.host.word(at + 4) as i32) >= 0);
    assert_ne!(f.reply_data(at, 0), 0);

    // The header and the identification word, which must be the same twice.
    let at = request(&[EEPROM_OPERATION, OP_READ, FLASH_HEADER, 0x30]);
    assert_eq!(f.host.word(at + 4), 12);
    let old_header = data(at, 12);
    let at = request(&[EEPROM_OPERATION, OP_IDENTIFY, 0]);
    let identity = f.reply_data(at, 0);
    let at = request(&[EEPROM_OPERATION, OP_IDENTIFY, 0]);
    assert_eq!(f.reply_data(at, 0), identity);
    assert_eq!((identity >> 24, (identity >> 16) & 0xff), (0x1f, 0xc4), "the documented manufacturer and device codes");

    // Code and data in groups of 0x1000 words, a rising 0x4000 bytes each,
    // every group acknowledged before the next.
    let image: Vec<Vec<u32>> =
        (0..3u32).map(|g| (0..0x1000u32).map(|i| (g << 28) ^ i.wrapping_mul(0x9e37_79b1)).collect()).collect();
    for (g, group) in image.iter().enumerate() {
        words(&[EEPROM_OPERATION, OP_WRITE, TYPE_IMAGE, g as u32 * 0x4000, 0x1000]);
        f.host.put(REPLY_PAGE, &[0; 4]);
        words(group);
        retry(|| f.host.word(REPLY_PAGE) != 0);
        assert_ne!(f.reply_data(REPLY_PAGE, 0), 0, "group {g} is acknowledged with a non-zero status");
    }

    // The header, state 2, with the codes it read.
    let mut header: Vec<u32> = (0..12).map(|i| 0x0101_0101 * (i + 1)).collect();
    header[HEADER_STATE] = 2;
    header[5] = identity >> 24;
    header[6] = (identity >> 16) & 0xff;
    words(&[EEPROM_OPERATION, OP_WRITE, TYPE_HEADER, 12]);
    f.host.put(REPLY_PAGE, &[0; 4]);
    words(&header);
    retry(|| f.host.word(REPLY_PAGE) != 0);
    assert_ne!(f.reply_data(REPLY_PAGE, 0), 0);
    assert_ne!(header, old_header);

    // Read back in 0x2000-byte reads and compare.
    for (g, group) in image.iter().enumerate() {
        for half in 0..2u32 {
            let at = request(&[EEPROM_OPERATION, OP_READ, g as u32 * 0x4000 + half * 0x2000, 0x2000]);
            assert_eq!(f.host.word(at + 4), 0x800);
            assert_eq!(data(at, 0x800), group[half as usize * 0x800..][..0x800], "group {g} half {half}");
        }
    }
    let at = request(&[EEPROM_OPERATION, OP_READ, FLASH_HEADER, 0x30]);
    let back = data(at, 12);
    assert_eq!(back, header);
    // Through the indirect port too, as the PROM's check sees it.
    assert_eq!(f.peek(INTERNAL_FLASH + FLASH_HEADER + 4 * HEADER_STATE as u32), 2);
    f.assert_in_step();
}

/// A back end that panics.

/// If the back end panics on the thread, the board is dead and says so: the bus
/// answers an error instead of busy forever, and `failure` names why.
#[test]
fn a_panicking_back_end_kills_the_board_visibly() {
    let host = TestHost::new();
    let kona = Kona::new(test_cfg(), host.clone(), host.clone(), crate::dev::kona::tests::test_status(), None).unwrap();
    kona.shared.pipe.lock().backend.hook = Some(TestHook { job: Box::new(|_, _| panic!("the back end panics on a job")), events: Arc::default(), on_attach: None });
    let board = crate::test_board::TestBoard::new(kona.clone());
    board.start();
    for w in [1, 2, 5] {
        kona.bus_write(COMMAND_PORT, 4, w);
    }
    retry(|| kona.bus_read(PORT_DATA, 4).0 == BUS_ERR);
    assert_eq!(kona.bus_write(COMMAND_PORT, 4, 1), BUS_ERR);
    assert!(kona.failure().unwrap().contains("panicked"));
    assert!(!board.is_running());
    // Stopping a dead board does not run the stores again.
    board.stop();
}

/// A monitor waiting for the processor's state sleeps until it is put back.
#[test]
fn a_monitor_waits_for_the_state_and_gets_it() {
    let (f, _board) = threaded();
    let held = f.kona.shared.pipe.lock();
    let k = f.kona.clone();
    let waiter = std::thread::spawn(move || k.failure());
    drop(held);
    assert_eq!(waiter.join().unwrap(), None);
}
