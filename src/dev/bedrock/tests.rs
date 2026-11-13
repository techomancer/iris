//! Tests of the chip as a whole: what one unit's access does to another,
//! the interrupt publish, and the paths that take no lock.

use super::reg::{io, lb, md, pi, LB_BASE, MD_BASE, PI1_BASE};
use super::*;
use crate::traits::{BUS_ERR, BUS_OK};

/// The L1 UART's line is bank 0 bit 4 of both PIs.
#[test]
fn the_uart_line_reaches_both_interfaces() {
    let rig = Rig::new();
    rig.w(pi::OFFSET_00A8, 1 << 4);
    let line = rig.chip.uart_irq();
    line.set(true);
    assert_eq!(rig.r(pi::PENDING_STATUS_A), 1 << 4);
    assert_eq!(rig.r(PI1_BASE + pi::PENDING_STATUS_A), 1 << 4);
    assert_eq!(rig.cause(), 1);
    line.set(false);
    assert_eq!(rig.cause(), 0);
}

/// Pending-modify stores, mask stores and a reflected level race on four
/// threads. Once every writer has stopped, the processor's word matches
/// what the registers say, whatever the interleaving.
#[test]
fn racing_stores_leave_the_word_matching_the_registers() {
    const ROUNDS: u64 = 20_000;
    let rig = Arc::new(Rig::new());
    let writers: Vec<_> = (0..4u64)
        .map(|n| {
            let rig = rig.clone();
            let uart = rig.chip.uart_irq();
            std::thread::spawn(move || {
                for i in 0..ROUNDS {
                    match n {
                        // Two writers set and clear bank 0 bits 1 and 2.
                        0 | 1 => rig.w(pi::PENDING_MODIFY, if (i + n) % 2 == 0 { 0x100 } else { 0 } | (n + 1)),
                        // One flips bank 0's mask between them.
                        2 => rig.w(pi::OFFSET_00A8, if i % 3 == 0 { 0b010 } else { 0b110 }),
                        // One drives the UART's level, bank 0 bit 4.
                        _ => uart.set(i % 2 == 0),
                    }
                }
            })
        })
        .collect();
    for writer in writers {
        writer.join().unwrap();
    }
    let pending = rig.r(pi::PENDING_STATUS_A);
    let mask = rig.r(pi::OFFSET_00A8);
    assert_eq!(rig.cause(), u64::from(pending & mask != 0), "pending {pending:#x} mask {mask:#x}");
    rig.w(pi::OFFSET_00A8, 1 << 4);
    rig.chip.uart_irq().set(true);
    assert_eq!(rig.cause(), 1);
}

/// One rise and one fall of a pending level, published by three threads in
/// the worst order: A sets the level and is held after sampling it, C
/// publishes the same rise, then B clears the level and publishes the fall,
/// all before A writes what it sampled. The processor must see the one rise
/// only: A's late sample must not bring the bit back after B's clear.
#[test]
fn a_late_publisher_brings_no_level_back() {
    use std::sync::mpsc::channel;
    use std::time::Duration;
    let rig = Arc::new(Rig::new());
    rig.w(pi::OFFSET_00A8, 1 << 9);
    // IP2 as each publisher saw it, in order.
    let seen = Arc::new(parking_lot::Mutex::new(vec![rig.cause() & 1]));
    let watch = |rig: &Arc<Rig>, seen: &Arc<parking_lot::Mutex<Vec<u64>>>| {
        let (rig, seen) = (rig.clone(), seen.clone());
        move || seen.lock().push(rig.cause() & 1)
    };
    let (held, is_held) = channel();
    let (release, released) = channel::<()>();
    let a = {
        let (rig, log) = (rig.clone(), watch(&rig, &seen));
        std::thread::spawn(move || {
            let mut first = true;
            super::pi::tests::on_sample(move |want| {
                log();
                if first && want & 1 != 0 {
                    first = false;
                    held.send(()).unwrap();
                    // A publisher that holds the PI's lock here keeps B
                    // out until this gives up.
                    let _ = released.recv_timeout(Duration::from_millis(300));
                }
            });
            rig.w(pi::PENDING_MODIFY, 0x109);
        })
    };
    is_held.recv().unwrap();
    let c = {
        let (rig, log) = (rig.clone(), watch(&rig, &seen));
        std::thread::spawn(move || {
            super::pi::tests::on_sample(move |_| log());
            rig.w(pi::OFFSET_00A8, 1 << 9);
        })
    };
    c.join().unwrap();
    let b = {
        let (rig, log) = (rig.clone(), watch(&rig, &seen));
        std::thread::spawn(move || {
            let after = log.clone();
            super::pi::tests::on_sample(move |_| log());
            rig.w(pi::PENDING_MODIFY, 0x009);
            after();
            let _ = release.send(());
        })
    };
    b.join().unwrap();
    a.join().unwrap();
    let mut seen = seen.lock().clone();
    seen.push(rig.cause() & 1);
    let rises = seen.windows(2).filter(|w| w == &[0, 1]).count();
    assert_eq!((rises, rig.cause()), (1, 0), "IP2 as published: {seen:?}");
}

/// Run `work` on another thread while this one holds every lock of the
/// chip; it must finish anyway.
fn without_locks(rig: &Arc<Rig>, work: impl FnOnce(&Rig) + Send + 'static) {
    let chip = &rig.chip;
    let held = (
        chip.pi[0].hold(),
        chip.pi[1].hold(),
        chip.md.hold(),
        chip.io.hold(),
        chip.lb.hold(),
        chip.ni.hold(),
        chip.xb.hold(),
    );
    let (done, finished) = std::sync::mpsc::channel();
    let other = rig.clone();
    let worker = std::thread::spawn(move || {
        work(&other);
        done.send(()).unwrap();
    });
    let result = finished.recv_timeout(std::time::Duration::from_secs(10));
    drop(held);
    worker.join().unwrap();
    assert!(result.is_ok(), "the path waited for a lock");
}

/// The PI reads a running guest makes most take no lock: the pending
/// status, a mask and the counter. Stores that change an interrupt input
/// publish under the PI's lock.
#[test]
fn the_pi_hot_reads_take_no_lock() {
    let rig = Arc::new(Rig::new());
    rig.w(pi::OFFSET_00A8, 1 << 9);
    rig.w(pi::PENDING_MODIFY, 0x109);
    without_locks(&rig, |rig| {
        assert_eq!(rig.r(pi::PENDING_STATUS_A), 1 << 9);
        assert_eq!(rig.r(pi::OFFSET_00A8), 1 << 9);
        assert_eq!(rig.cause(), 1);
        rig.set_time(8000);
        assert_eq!(rig.r(pi::COUNTER), 10);
        assert_eq!(rig.regs.read32(pi::COUNTER + 4).data, 10);
    });
}

/// With MD's cell register bit 0 clear, the register window refuses every
/// access, MD's own registers included. The padding after a unit's
/// registers, and anything past the window, answers a bus error.
#[test]
fn the_cell_bit_gates_the_window_and_padding_answers_an_error() {
    let rig = Rig::new();
    assert!(rig.chip.cell_permits());
    assert_eq!(rig.regs.read64(super::pi::SIZE).status, BUS_ERR, "past PI0's registers");
    assert_eq!(rig.regs.read64(MD_BASE - 8).status, BUS_ERR);
    assert_eq!(rig.regs.read64(super::reg::WINDOW).status, BUS_ERR);
    rig.w(MD_BASE + md::CELL, 0);
    assert!(!rig.chip.cell_permits());
    assert_eq!(rig.regs.read64(pi::INTERFACE_ID).status, BUS_ERR);
    assert_eq!(rig.regs.read64(MD_BASE + md::CELL).status, BUS_ERR);
}

/// An inbound interrupt packet's address for PI `pi` of node `node`.
fn packet_address(node: u64, pi: u64) -> u64 {
    (1 << 47) | (node << 39) | 0x180_0000 | (pi * PI1_BASE) | 0x90
}

/// The PI region mask gates interrupt packets: a refused packet answers a
/// bus error, so the sender knows the level was never set, and sets nothing.
#[test]
fn a_refused_interrupt_packet_answers_an_error_from_inbound_write() {
    let rig = Rig::new();
    let chip = &rig.chip;
    rig.w(pi::OFFSET_00A8, 1 << 9);
    rig.w(pi::OFFSET_0010, 0);
    assert_eq!(chip.inbound_write(15, Some(0), packet_address(0, 0), 8, 0x109), BUS_ERR);
    assert_eq!(rig.r(pi::PENDING_STATUS_A), 0);
    assert_eq!(rig.cause(), 0);
    rig.w(pi::OFFSET_0010, 1);
    assert_eq!(chip.inbound_write(15, Some(0), packet_address(0, 0), 8, 0x109), BUS_OK);
    assert_eq!(rig.r(pi::PENDING_STATUS_A), 1 << 9);
    assert_eq!(rig.cause(), 1, "IP2");
    assert_eq!(chip.inbound_write(15, Some(0), packet_address(0, 0), 8, 0x009), BUS_OK);
    assert_eq!(rig.cause(), 0);
}

/// LB's region register names the node; inbound packets and the BTEs'
/// interrupts answer only to that node from then on.
#[test]
fn the_local_blocks_node_number_moves_the_hub_window() {
    let rig = Rig::new();
    let chip = &rig.chip;
    rig.w(LB_BASE + lb::NODE_REGION, 3 << 32);
    assert_eq!(chip.inbound_write(15, Some(0), packet_address(0, 0), 8, 0x101), BUS_ERR);
    assert_eq!(chip.inbound_write(15, Some(0), packet_address(3, 0), 8, 0x101), BUS_OK);
    assert_eq!(rig.r(pi::PENDING_STATUS_A), 1 << 1);
    // A BTE completion interrupt names its node in bits 8:1.
    rig.w(MD_BASE + md::CONFIG, 0x11);
    rig.w(super::reg::IO_BASE + io::BTE1_BASE + super::reg::bte::INTERRUPT, (5 << 16) | (3 << 1));
    rig.w(super::reg::IO_BASE + io::BTE1_BASE, 1 | super::io::BUSY);
    rig.w(super::reg::IO_BASE + io::BTE1_BASE + super::reg::bte::CONTROL, 0);
    assert_eq!(rig.r(pi::PENDING_STATUS_A), (1 << 1) | (1 << 5));
}

/// A node reset leaves as one event on the board's channel, for either
/// cause, and the chip calls nothing else in the board.
#[test]
fn a_node_reset_is_one_event_on_the_boards_channel() {
    let rig = Rig::new();
    rig.w(LB_BASE + lb::SOFT_RESET, 1);
    rig.w(super::reg::NI_BASE + super::reg::ni::LOCAL_RESET, 4);
    let events: Vec<_> = rig.events.lock().try_iter().collect();
    assert!(matches!(events[..], [MachineEvent::HardReset, MachineEvent::HardReset]), "{events:?}");
}

/// Inbound set and clear packets race mask stores and LB's reflected error
/// level on other threads; once every writer has stopped, the processor's
/// word matches what the registers say.
#[test]
fn racing_packets_and_reflected_levels_leave_the_word_matching_the_registers() {
    const ROUNDS: u64 = 20_000;
    let rig = Arc::new(Rig::new());
    rig.w(pi::OFFSET_00B0, 1 << 55);
    let writers: Vec<_> = (0..4u64)
        .map(|n| {
            let rig = rig.clone();
            std::thread::spawn(move || {
                for i in 0..ROUNDS {
                    match n {
                        // Two packet sources set and clear bits 1 and 2.
                        0 | 1 => {
                            let set = if (i + n) % 2 == 0 { 0x100 } else { 0 };
                            let status = rig.chip.inbound_write(15, Some(0), packet_address(0, 0), 8, set | (n + 1));
                            assert_eq!(status, BUS_OK);
                        }
                        // One store flips bank 0's mask between the two bits.
                        2 => rig.w(pi::OFFSET_00A8, if i % 3 == 0 { 0b010 } else { 0b110 }),
                        // One drives LB's error level, bank 1 bit 55.
                        _ => rig.w(LB_BASE + if i % 2 == 0 { lb::ERROR_STATUS } else { lb::ERROR_CLEAR }, 1),
                    }
                }
            })
        })
        .collect();
    for writer in writers {
        writer.join().unwrap();
    }
    let pending = rig.r(pi::PENDING_STATUS_A);
    let mask = rig.r(pi::OFFSET_00A8);
    let reflected = rig.r(pi::PENDING_STATUS_B) & rig.r(pi::OFFSET_00B0);
    let want = u64::from(pending & mask != 0) | u64::from(reflected != 0) << 1;
    assert_eq!(rig.cause(), want, "pending {pending:#x} mask {mask:#x} reflected {reflected:#x}");
}

/// Two stores to LB's error registers race; each sets the reflected bit
/// under LB's lock, so the level both PIs see is the one the register
/// holds when the stores are done.
#[test]
fn a_reflected_level_follows_its_register_under_racing_stores() {
    let rig = Arc::new(Rig::new());
    let writers: Vec<_> = [lb::ERROR_STATUS, lb::ERROR_CLEAR]
        .into_iter()
        .map(|at| {
            let rig = rig.clone();
            std::thread::spawn(move || {
                for _ in 0..20_000 {
                    rig.w(LB_BASE + at, 1);
                }
            })
        })
        .collect();
    for writer in writers {
        writer.join().unwrap();
    }
    let level = rig.r(LB_BASE + lb::ERROR_STATUS) != 0;
    for base in [0, PI1_BASE] {
        assert_eq!(rig.r(base + pi::PENDING_STATUS_B) & (1 << 55) != 0, level);
    }
}

/// The crossbar's data paths take no lock either: an 8-byte inbound memory
/// write and read, and PIO to a widget. An inbound interrupt packet
/// publishes under the PI's lock.
#[test]
fn the_inbound_dma_and_pio_paths_take_no_lock() {
    struct Stub;
    impl BusDevice for Stub {
        fn read64(&self, _: u64) -> BusRead64 {
            BusRead64::ok(0x5a)
        }
    }
    let rig = Arc::new(Rig::new());
    rig.w(MD_BASE + md::CONFIG, 0x11);
    rig.chip.connect_widget(8, Arc::new(Stub));
    rig.w(pi::OFFSET_00A8, 1 << 9);
    assert_eq!(rig.chip.inbound_write(15, Some(0), packet_address(0, 0), 8, 0x109), BUS_OK);
    without_locks(&rig, |rig| {
        assert_eq!(rig.cause(), 1);
        assert_eq!(rig.chip.inbound_write(14, None, 0x1000, 8, 0x1234), BUS_OK);
        assert_eq!(rig.chip.inbound_read(14, None, 0x1000, 8).data, 0x1234);
        assert_eq!(rig.chip.pio().read64(0x800_0000).data, 0x5a);
    });
}

/// Both ends of a cable launch vector requests at each other, and send each
/// other warm resets, in the same instant, round after round. Neither end may
/// hold a unit lock across its call into the other, or each would wait on the
/// other's. Every reply carries the peer's word for that round.
#[test]
fn both_ends_launching_at_once_complete() {
    const ROUNDS: u64 = 2_000;
    const SCRATCH2: u64 = LB_BASE + lb::PIO_WINDOW + 16;
    let (a, b) = cabled_pair();
    for rig in [&a, &b] {
        rig.w(reg::NI_BASE + reg::ni::CONTROL_0, 1);
    }
    let rigs = [Arc::new(a), Arc::new(b)];
    let start = Arc::new(std::sync::Barrier::new(2));
    let (done, finished) = std::sync::mpsc::channel();
    for end in 0..2usize {
        let (me, peer_rig) = (rigs[end].clone(), rigs[1 - end].clone());
        let (start, done) = (start.clone(), done.clone());
        std::thread::spawn(move || {
            let tag = |end: usize, round: u64| ((end as u64 + 1) << 32) | round;
            let params = (0x60_0000 + lb::PIO_WINDOW + 16) | (0x1c8 << 40);
            for round in 0..ROUNDS {
                me.w(SCRATCH2, tag(end, round));
                start.wait();
                me.r(LB_BASE + lb::VECTOR_STATUS_ALIAS);
                me.w(LB_BASE + lb::REGISTER_C008, 0);
                me.w(LB_BASE + lb::VECTOR_REQUEST, params);
                assert_eq!(me.r(LB_BASE + lb::VECTOR_STATUS) >> 62, 2, "round {round}: a valid reply, no overrun");
                assert_eq!(me.r(LB_BASE + lb::REGISTER_C030), tag(1 - end, round), "round {round}");
                if round % 4 == 0 {
                    me.w(reg::NI_BASE + reg::ni::LOCAL_RESET, 2);
                }
                start.wait();
            }
            drop(peer_rig);
            done.send(end).unwrap();
        });
    }
    for _ in 0..2 {
        finished.recv_timeout(std::time::Duration::from_secs(120)).expect("a launch is stuck waiting on the other end");
    }
    // Each end took the other's resets: every fourth round, once.
    for rig in &rigs {
        assert_eq!(rig.events.lock().try_iter().count() as u64, ROUNDS / 4);
    }
}

/// An inbound block is the same traffic as its words one at a time: it
/// stops at the end of the bank, delivers a packet that starts in the
/// register-packet space as single words, and never runs from memory into
/// that space.
#[test]
fn an_inbound_block_follows_the_rules_of_its_words() {
    let rig = Rig::new();
    let chip = &rig.chip;
    rig.w(MD_BASE + md::CONFIG, 0x11);
    let bank = 64u64 << 20;
    let words = [0x1111_2222_3333_4444, 0x5555_6666_7777_8888, 0x99, 0xaa];
    assert_eq!(chip.inbound_write_block(14, Some(2), 0x1000, &words), BUS_OK);
    for (i, &w) in words.iter().enumerate() {
        assert_eq!(chip.inbound_read(14, Some(2), 0x1000 + 8 * i as u64, 8).data, w);
    }
    let mut back = [0; 4];
    assert_eq!(chip.inbound_read_block(14, Some(2), 0x1000, &mut back), BUS_OK);
    assert_eq!(back, words);
    // The end of the bank: the words below it land, then the error.
    assert_eq!(chip.inbound_write_block(14, Some(1), bank - 16, &words), BUS_ERR);
    assert_eq!(chip.dma_view().read64(bank - 8).data, words[1]);
    assert_eq!(chip.inbound_read_block(14, Some(1), bank - 16, &mut back), BUS_ERR);
    // The packet space, the 48-bit limit and an empty block. A DMA engine
    // that moves one doubleword to a PI's pending-modify register sends an
    // interrupt packet, whichever way it hands the word over.
    rig.w(pi::OFFSET_0010, 0);
    assert_eq!(chip.inbound_write_block(15, Some(0), packet_address(0, 0), &[0x109]), BUS_ERR);
    assert_eq!(rig.r(pi::PENDING_STATUS_A), 0);
    rig.w(pi::OFFSET_0010, 1);
    assert_eq!(chip.inbound_write_block(15, Some(0), packet_address(0, 0), &[0x109]), BUS_OK);
    assert_eq!(rig.r(pi::PENDING_STATUS_A), 1 << 9);
    assert_eq!(chip.inbound_read_block(15, Some(0), packet_address(0, 0), &mut back[..1]), BUS_ERR);
    assert_eq!(chip.inbound_write_block(14, Some(1), (1 << 47) - 8, &words[..2]), BUS_ERR);
    assert_eq!(chip.inbound_write_block(14, Some(1), (1 << 48) - 8, &words[..2]), BUS_ERR);
    assert_eq!(chip.inbound_read_block(14, Some(1), 0x1000, &mut []), BUS_OK);
    // The widget's DMA face carries blocks the same way.
    let rig = Rig::new();
    rig.w(MD_BASE + md::CONFIG, 0x11);
    let face = rig.chip.widget_dma(14);
    assert_eq!(face.write_block(0x3000, &words), BUS_OK);
    assert_eq!(face.read64(0x3008).data, words[1]);
    assert_eq!(face.read_block(0x3000, &mut back), BUS_OK);
    assert_eq!(back, words);
    assert_eq!(face.read_block(bank, &mut back), BUS_ERR);
}

/// A test chip's configuration: the Fuel's one processor and one 64 MiB
/// bank, on `clock`.
pub(super) fn test_config(clock: Clock) -> BedrockConfig<'static> {
    BedrockConfig { name: "test", clock, processors: Population::SINGLE, banks_mib: &[64], nodes: 1, events: None }
}

/// Two chips joined by a cable, each on a clock of its own.
pub(super) fn cabled_pair() -> (Rig, Rig) {
    let (a, b) = (Rig::new(), Rig::new());
    Bedrock::cable(&a.chip, &b.chip);
    (a, b)
}

/// A chip on a hand-driven clock, for the unit tests: its register window
/// and CPU 0's interrupt word.
pub(super) struct Rig {
    pub(super) chip: Arc<Bedrock>,
    now: Arc<AtomicU64>,
    pub(super) word: Arc<AtomicU64>,
    pub(super) regs: Arc<dyn BusDevice>,
    /// The node's events.
    pub(super) events: parking_lot::Mutex<std::sync::mpsc::Receiver<MachineEvent>>,
}
impl Rig {
    /// The Fuel's one processor.
    pub(super) fn new() -> Self {
        Self::with(Population::SINGLE)
    }
    /// A node with four processors, for tests that drive the chip as each in
    /// turn: a thread is a processor of the node only if the node fits it.
    pub(super) fn quad() -> Self {
        Self::with(Population::fitted_cpus(4).unwrap())
    }
    pub(super) fn with(processors: Population) -> Self {
        let now = Arc::new(AtomicU64::new(0));
        let clock = now.clone();
        let config = test_config(Arc::new(move || clock.load(Ordering::SeqCst)));
        let (events, receiver) = std::sync::mpsc::channel();
        let chip = Bedrock::new(BedrockConfig { processors, events: Some(events), ..config }).unwrap();
        let word = Arc::new(AtomicU64::new(0));
        chip.set_cpu_interrupts(0, &*word);
        chip.start();
        let regs = chip.registers();
        Self { chip, now, word, regs, events: parking_lot::Mutex::new(receiver) }
    }
    pub(super) fn r(&self, at: u64) -> u64 {
        let r = self.regs.read64(at);
        assert_eq!(r.status, BUS_OK, "read at {at:#x}");
        r.data
    }
    pub(super) fn w(&self, at: u64, value: u64) {
        assert_eq!(self.regs.write64(at, value), BUS_OK, "write at {at:#x}");
    }
    pub(super) fn set_time(&self, ns: u64) {
        self.now.store(ns, Ordering::SeqCst);
    }
    /// CPU 0's IP2..IP6, bit 0 being IP2.
    pub(super) fn cause(&self) -> u64 {
        (self.word.load(Ordering::SeqCst) >> 10) & 0x1f
    }
}
