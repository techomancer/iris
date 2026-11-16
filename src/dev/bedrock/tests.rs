//! Tests of the chip as a whole: what one unit's access does to another,
//! the interrupt publish, and the paths that take no lock.

use super::io::tests::launch;
use super::regs::{io, lb, md, pi, PI1_BASE};
use super::*;
use crate::traits::{BUS_ERR, BUS_OK};

/// The L1 UART's line is bank 0 bit 4 of both PIs.
#[test]
fn the_uart_line_reaches_both_interfaces() {
    let rig = Rig::new();
    rig.w(pi::INT_MASK0_A::OFFSET, 1 << 4);
    let line = rig.chip.uart_irq();
    line.set(true);
    assert_eq!(rig.r(pi::INT_PEND0::OFFSET), 1 << 4);
    assert_eq!(rig.r(PI1_BASE + pi::INT_PEND0::OFFSET), 1 << 4);
    assert_eq!(rig.cause(), 1);
    line.set(false);
    assert_eq!(rig.cause(), 0);
}

/// Pending-modify stores, mask stores and a reflected level race on four threads; the word then
/// matches the registers.
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
                        0 | 1 => rig.w(pi::INT_PEND_MOD::OFFSET, if (i + n) % 2 == 0 { 0x100 } else { 0 } | (n + 1)),
                        // One flips bank 0's mask between them.
                        2 => rig.w(pi::INT_MASK0_A::OFFSET, if i % 3 == 0 { 0b010 } else { 0b110 }),
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
    let pending = rig.r(pi::INT_PEND0::OFFSET);
    let mask = rig.r(pi::INT_MASK0_A::OFFSET);
    assert_eq!(rig.cause(), u64::from(pending & mask != 0), "pending {pending:#x} mask {mask:#x}");
    rig.w(pi::INT_MASK0_A::OFFSET, 1 << 4);
    rig.chip.uart_irq().set(true);
    assert_eq!(rig.cause(), 1);
}

/// A rise and fall published by three threads in the worst order: A's late sample must not revive
/// the bit.
#[test]
fn a_late_publisher_brings_no_level_back() {
    use std::sync::mpsc::channel;
    use std::time::Duration;
    let rig = Arc::new(Rig::new());
    rig.w(pi::INT_MASK0_A::OFFSET, 1 << 9);
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
            rig.w(pi::INT_PEND_MOD::OFFSET, 0x109);
        })
    };
    is_held.recv().unwrap();
    let c = {
        let (rig, log) = (rig.clone(), watch(&rig, &seen));
        std::thread::spawn(move || {
            super::pi::tests::on_sample(move |_| log());
            rig.w(pi::INT_MASK0_A::OFFSET, 1 << 9);
        })
    };
    c.join().unwrap();
    let b = {
        let (rig, log) = (rig.clone(), watch(&rig, &seen));
        std::thread::spawn(move || {
            let after = log.clone();
            super::pi::tests::on_sample(move |_| log());
            rig.w(pi::INT_PEND_MOD::OFFSET, 0x009);
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

/// The PI reads a running guest makes most take no lock; stores that change an interrupt input
/// publish under it.
#[test]
fn the_pi_hot_reads_take_no_lock() {
    let rig = Arc::new(Rig::new());
    rig.w(pi::INT_MASK0_A::OFFSET, 1 << 9);
    rig.w(pi::INT_PEND_MOD::OFFSET, 0x109);
    without_locks(&rig, |rig| {
        assert_eq!(rig.r(pi::INT_PEND0::OFFSET), 1 << 9);
        assert_eq!(rig.r(pi::INT_MASK0_A::OFFSET), 1 << 9);
        assert_eq!(rig.cause(), 1);
        rig.set_time(8000);
        assert_eq!(rig.r(pi::RT_COUNTER::OFFSET), 10);
        assert_eq!(rig.regs.read32(pi::RT_COUNTER::OFFSET + 4).data, 10);
    });
}

/// With MD's cell register bit 0 clear the window refuses every access; padding and past-window
/// accesses are bus errors.
#[test]
fn the_cell_bit_gates_the_window_and_padding_answers_an_error() {
    let rig = Rig::new();
    assert!(rig.chip.md.cell_permits());
    assert_eq!(rig.regs.read64(super::pi::SIZE).status, BUS_ERR, "past PI0's registers");
    assert_eq!(rig.regs.read64(md::BASE - 8).status, BUS_ERR);
    assert_eq!(rig.regs.read64(super::regs::WINDOW_SIZE).status, BUS_ERR);
    rig.w(md::BASE + md::CURRENT_CELL::OFFSET, 0);
    assert!(!rig.chip.md.cell_permits());
    assert_eq!(rig.regs.read64(pi::CPU_NUM::OFFSET).status, BUS_ERR);
    assert_eq!(rig.regs.read64(md::BASE + md::CURRENT_CELL::OFFSET).status, BUS_ERR);
}

/// An inbound interrupt packet's address for PI `pi` of node `node`.
fn packet_address(node: u64, pi: u64) -> u64 {
    (1 << 47) | (node << 39) | 0x180_0000 | (pi * PI1_BASE) | 0x90
}

/// The PI region mask gates interrupt packets: a refused one answers a bus error and sets nothing.
#[test]
fn a_refused_interrupt_packet_answers_an_error_from_inbound_write() {
    let rig = Rig::new();
    let chip = &rig.chip;
    rig.w(pi::INT_MASK0_A::OFFSET, 1 << 9);
    rig.w(pi::IO_PROTECT::OFFSET, 0);
    assert_eq!(chip.inbound_write(15, Some(0), packet_address(0, 0), 8, 0x109), BUS_ERR);
    assert_eq!(rig.r(pi::INT_PEND0::OFFSET), 0);
    assert_eq!(rig.cause(), 0);
    rig.w(pi::IO_PROTECT::OFFSET, 1);
    assert_eq!(chip.inbound_write(15, Some(0), packet_address(0, 0), 8, 0x109), BUS_OK);
    assert_eq!(rig.r(pi::INT_PEND0::OFFSET), 1 << 9);
    assert_eq!(rig.cause(), 1, "IP2");
    assert_eq!(chip.inbound_write(15, Some(0), packet_address(0, 0), 8, 0x009), BUS_OK);
    assert_eq!(rig.cause(), 0);
}

/// A node reset leaves as one event on the board's channel for either cause.
#[test]
fn a_node_reset_is_one_event_on_the_boards_channel() {
    let rig = Rig::new();
    rig.w(lb::BASE + lb::SOFT_RESET::OFFSET, 1);
    rig.w(super::regs::ni::BASE + super::regs::ni::PORT_RESET::OFFSET, 4);
    let events: Vec<_> = rig.events.lock().try_iter().collect();
    assert!(matches!(events[..], [MachineEvent::HardReset, MachineEvent::HardReset]), "{events:?}");
}

/// Inbound set and clear packets race mask stores and LB's reflected error level; the word then
/// matches the registers.
#[test]
fn racing_packets_and_reflected_levels_leave_the_word_matching_the_registers() {
    const ROUNDS: u64 = 20_000;
    let rig = Arc::new(Rig::new());
    rig.w(pi::INT_MASK1_A::OFFSET, 1 << 55);
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
                        2 => rig.w(pi::INT_MASK0_A::OFFSET, if i % 3 == 0 { 0b010 } else { 0b110 }),
                        // One drives LB's error level, bank 1 bit 55.
                        _ => rig.w(lb::BASE + if i % 2 == 0 { lb::ERROR_BITS::OFFSET } else { lb::ERROR_MASK_CLR::OFFSET }, 1),
                    }
                }
            })
        })
        .collect();
    for writer in writers {
        writer.join().unwrap();
    }
    let pending = rig.r(pi::INT_PEND0::OFFSET);
    let mask = rig.r(pi::INT_MASK0_A::OFFSET);
    let reflected = rig.r(pi::INT_PEND1::OFFSET) & rig.r(pi::INT_MASK1_A::OFFSET);
    let want = u64::from(pending & mask != 0) | u64::from(reflected != 0) << 1;
    assert_eq!(rig.cause(), want, "pending {pending:#x} mask {mask:#x} reflected {reflected:#x}");
}

/// Two stores to LB's error registers race; the level both PIs see is the register's when they are
/// done.
#[test]
fn a_reflected_level_follows_its_register_under_racing_stores() {
    let rig = Arc::new(Rig::new());
    let writers: Vec<_> = [lb::ERROR_BITS::OFFSET, lb::ERROR_MASK_CLR::OFFSET]
        .into_iter()
        .map(|at| {
            let rig = rig.clone();
            std::thread::spawn(move || {
                for _ in 0..20_000 {
                    rig.w(lb::BASE + at, 1);
                }
            })
        })
        .collect();
    for writer in writers {
        writer.join().unwrap();
    }
    let level = rig.r(lb::BASE + lb::ERROR_BITS::OFFSET) != 0;
    for base in [0, PI1_BASE] {
        assert_eq!(rig.r(base + pi::INT_PEND1::OFFSET) & (1 << 55) != 0, level);
    }
}

/// The crossbar's data paths take no lock: 8-byte inbound memory access and PIO to a widget.
#[test]
fn the_inbound_dma_and_pio_paths_take_no_lock() {
    struct Stub;
    impl BusDevice for Stub {
        fn read64(&self, _: u64) -> BusRead64 {
            BusRead64::ok(0x5a)
        }
    }
    let rig = Arc::new(Rig::new());
    rig.w(md::BASE + md::MEMORY_CONFIG::OFFSET, 0x11);
    rig.chip.io.connect_widget(8, Arc::new(Stub));
    rig.w(pi::INT_MASK0_A::OFFSET, 1 << 9);
    assert_eq!(rig.chip.inbound_write(15, Some(0), packet_address(0, 0), 8, 0x109), BUS_OK);
    without_locks(&rig, |rig| {
        assert_eq!(rig.cause(), 1);
        assert_eq!(rig.chip.inbound_write(14, None, 0x1000, 8, 0x1234), BUS_OK);
        assert_eq!(rig.chip.inbound_read(14, None, 0x1000, 8).data, 0x1234);
        assert_eq!(rig.chip.pio().read64(0x800_0000).data, 0x5a);
    });
}

/// Both ends launch vector requests and warm resets at each other each round; neither holds a unit
/// lock across the call.
#[test]
fn both_ends_launching_at_once_complete() {
    const ROUNDS: u64 = 2_000;
    const SCRATCH2: u64 = lb::BASE + lb::SCRATCH0::OFFSET + 16;
    let (a, b) = connected_pair();
    for rig in [&a, &b] {
        rig.w(regs::ni::BASE + regs::ni::RESET_ENABLE::OFFSET, 1);
    }
    let rigs = [Arc::new(a), Arc::new(b)];
    let start = Arc::new(std::sync::Barrier::new(2));
    let (done, finished) = std::sync::mpsc::channel();
    for end in 0..2usize {
        let (me, peer_rig) = (rigs[end].clone(), rigs[1 - end].clone());
        let (start, done) = (start.clone(), done.clone());
        std::thread::spawn(move || {
            let tag = |end: usize, round: u64| ((end as u64 + 1) << 32) | round;
            let params = (0x60_0000 + lb::SCRATCH0::OFFSET + 16) | (0x1c8 << 40);
            for round in 0..ROUNDS {
                me.w(SCRATCH2, tag(end, round));
                start.wait();
                me.r(lb::BASE + lb::VECTOR_STATUS_CLEAR::OFFSET);
                me.w(lb::BASE + lb::VECTOR_ROUTE::OFFSET, 0);
                me.w(lb::BASE + lb::VECTOR_PARMS::OFFSET, params);
                assert_eq!(me.r(lb::BASE + lb::VECTOR_STATUS::OFFSET) >> 62, 2, "round {round}: a valid reply, no overrun");
                assert_eq!(me.r(lb::BASE + lb::VECTOR_READ_DATA::OFFSET), tag(1 - end, round), "round {round}");
                if round % 4 == 0 {
                    me.w(regs::ni::BASE + regs::ni::PORT_RESET::OFFSET, 2);
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

/// An inbound block is its words one at a time, stops at the bank's end and never runs from memory
/// into packet space.
#[test]
fn an_inbound_block_follows_the_rules_of_its_words() {
    let rig = Rig::new();
    let chip = &rig.chip;
    rig.w(md::BASE + md::MEMORY_CONFIG::OFFSET, 0x11);
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
    assert_eq!(chip.decode.physical().read64(bank - 8).data, words[1]);
    assert_eq!(chip.inbound_read_block(14, Some(1), bank - 16, &mut back), BUS_ERR);
    // The packet space, the 48-bit limit and an empty block; a DMA engine moving one doubleword to
    // pending-modify sends an interrupt packet.
    rig.w(pi::IO_PROTECT::OFFSET, 0);
    assert_eq!(chip.inbound_write_block(15, Some(0), packet_address(0, 0), &[0x109]), BUS_ERR);
    assert_eq!(rig.r(pi::INT_PEND0::OFFSET), 0);
    rig.w(pi::IO_PROTECT::OFFSET, 1);
    assert_eq!(chip.inbound_write_block(15, Some(0), packet_address(0, 0), &[0x109]), BUS_OK);
    assert_eq!(rig.r(pi::INT_PEND0::OFFSET), 1 << 9);
    assert_eq!(chip.inbound_read_block(15, Some(0), packet_address(0, 0), &mut back[..1]), BUS_ERR);
    assert_eq!(chip.inbound_write_block(14, Some(1), (1 << 47) - 8, &words[..2]), BUS_ERR);
    assert_eq!(chip.inbound_write_block(14, Some(1), (1 << 48) - 8, &words[..2]), BUS_ERR);
    assert_eq!(chip.inbound_read_block(14, Some(1), 0x1000, &mut []), BUS_OK);
    // The widget's DMA port carries blocks the same way.
    let rig = Rig::new();
    rig.w(md::BASE + md::MEMORY_CONFIG::OFFSET, 0x11);
    let port = rig.chip.inbound(14);
    assert_eq!(port.write_block(0x3000, &words), BUS_OK);
    assert_eq!(port.read64(0x3008).data, words[1]);
    assert_eq!(port.read_block(0x3000, &mut back), BUS_OK);
    assert_eq!(back, words);
    assert_eq!(port.read_block(bank, &mut back), BUS_ERR);
}

/// Two connected chips, each with 64 MiB configured, the second numbered 1.
const PROBE: u64 = 0x1000;
fn numbered_pair() -> (Rig, Rig) {
    let (a, b) = connected_pair();
    for rig in [&a, &b] {
        rig.w(md::BASE + md::MEMORY_CONFIG::OFFSET, 0x11);
    }
    b.w(lb::BASE + lb::REV_ID::OFFSET, 1 << 32);
    (a, b)
}
fn node_at(n: u64) -> u64 {
    n << NODE_SHIFT
}

/// An interrupt packet names its node, and reaches that node's PI.
#[test]
fn an_interrupt_packet_reaches_the_other_nodes_hub() {
    let (a, b) = numbered_pair();
    assert_eq!(a.chip.inbound_write(15, Some(0), packet_address(1, 0), 8, 0x108), BUS_OK);
    assert_eq!(b.r(pi::INT_PEND0::OFFSET), 1 << 8, "node 1's processor interface took it");
    assert_eq!(a.r(pi::INT_PEND0::OFFSET), 0);
    assert_eq!(a.chip.inbound_write(15, Some(0), packet_address(1, 1), 8, 0x109), BUS_OK, "its second interface too");
    assert_eq!(b.r(PI1_BASE + pi::INT_PEND0::OFFSET), 1 << 9);
    assert_eq!(a.chip.inbound_write(15, Some(0), packet_address(0, 0), 8, 0x10a), BUS_OK, "its own number is still its own");
    assert_eq!(a.r(pi::INT_PEND0::OFFSET), 1 << 10);
    assert_eq!(a.chip.inbound_write(15, Some(0), packet_address(9, 0), 8, 0x108), BUS_ERR, "nobody carries 9");
}

/// Inbound DMA names memory by node number, node field zero included, whatever the cached alias.
#[test]
fn inbound_dma_reaches_memory_by_node_number() {
    let (a, b) = connected_pair();
    for rig in [&a, &b] {
        rig.w(md::BASE + md::MEMORY_CONFIG::OFFSET, 0x11);
    }
    assert_eq!(a.chip.decode.write64(PROBE, 0x0a00), BUS_OK);
    assert_eq!(b.chip.decode.write64(PROBE, 0x0b11), BUS_OK);
    let dma = |rig: &Rig, address: u64| {
        let r = rig.chip.inbound_read(15, Some(0), address, 8);
        (r.status == BUS_OK).then_some(r.data)
    };
    assert_eq!(dma(&b, PROBE), Some(0x0b11), "both carry 0: each reaches itself");
    b.w(lb::BASE + lb::REV_ID::OFFSET, 1 << 32);
    b.w(pi::CALIAS_SIZE::OFFSET, 4);
    assert_eq!(dma(&a, PROBE), Some(0x0a00));
    assert_eq!(dma(&a, node_at(1) + PROBE), Some(0x0b11), "node 1 by its number");
    assert_eq!(dma(&b, node_at(1) + PROBE), Some(0x0b11), "a node by its own number");
    assert_eq!(dma(&b, PROBE), Some(0x0a00), "node field zero is node 0's, alias or not");
    assert_eq!(a.chip.inbound_write(15, Some(0), node_at(1) + 0x9000, 8, 6), BUS_OK);
    assert_eq!(b.chip.decode.read64(node_at(1) + 0x9000).data, 6, "a DMA store lands on node 1");
    assert_eq!(dma(&a, node_at(9) + PROBE), None, "nobody carries 9");
}

/// A block of inbound DMA goes where its node number sends it, across the link into the peer's
/// memory.
#[test]
fn an_inbound_block_reaches_the_peer_by_node_number() {
    let (a, b) = numbered_pair();
    let words = [0x1111, 0x2222, 0x3333, 0x4444];
    assert_eq!(a.chip.inbound_write_block(15, Some(0), node_at(1) + 0x9000, &words), BUS_OK);
    for (i, &w) in words.iter().enumerate() {
        assert_eq!(b.chip.decode.read64(node_at(1) + 0x9000 + 8 * i as u64).data, w, "word {i} lands on node 1");
        assert_eq!(a.chip.decode.read64(0x9000 + 8 * i as u64).data, 0, "not on the writer's own node");
    }
    let mut back = [0; 4];
    assert_eq!(a.chip.inbound_read_block(15, Some(0), node_at(1) + 0x9000, &mut back), BUS_OK);
    assert_eq!(back, words);
    // A store-conditional to another node's memory is the claimant's compare-and-swap.
    let cache = a.chip.decode.physical().clone();
    let at = node_at(1) + 0x3000;
    assert_eq!(cache.write64(at, 0x1111_2222_3333_4444), BUS_OK);
    assert_eq!(cache.cas64(at, 0x1111_2222_3333_4444, 7), Some(true));
    assert_eq!(cache.cas64(at, 0x1111_2222_3333_4444, 9), Some(false));
    assert_eq!(b.chip.decode.read64(at).data, 7, "the claimant's memory holds the swap");
    assert_eq!(cache.write32(at + 8, 5), BUS_OK);
    assert_eq!(cache.cas32(at + 8, 5, 6), Some(true));
    assert_eq!(cache.cas32(at + 8, 5, 8), Some(false));
    assert_eq!(cache.cas32(node_at(9) + 0x3000, 0, 1), None, "an unclaimed node offers no swap");
}

/// Run a block-transfer engine: `lines` lines from `source` to `destination`, completing by an
/// interrupt to node `node`'s interface `pi` at `level`.
fn transfer(rig: &Rig, lines: u64, source: u64, destination: u64, node: u64, pi: u64, level: u64) {
    rig.w(io::BASE + io::BTE_INTERRUPT_ADDRESS::OFFSET, (level << 16) | (node << 1) | pi);
    launch(rig, io::BTE_LENGTH_STATUS::OFFSET, lines, source, destination, 0, 0);
}

/// The engines copy by node number and their completion interrupt names a node: the other node's,
/// not the issuer's.
#[test]
fn a_block_transfer_completion_reaches_the_node_it_names() {
    let (a, b) = numbered_pair();
    for i in 0..16 {
        assert_eq!(b.chip.decode.write64(node_at(1) + 0x4000 + i * 8, 0xfeed_0000 | i), BUS_OK);
    }
    // Node 0's engine copies node 1's line to its own memory and tells node 1.
    transfer(&a, 1, node_at(1) + 0x4000, 0x6000, 1, 0, 53);
    for i in 0..16 {
        assert_eq!(a.chip.decode.read64(0x6000 + i * 8).data, 0xfeed_0000 | i);
    }
    assert_eq!(a.r(io::BASE + io::BTE_LENGTH_STATUS::OFFSET), 0);
    assert_eq!(b.r(pi::INT_PEND0::OFFSET), 1 << 53, "the completion reached node 1");
    assert_eq!(a.r(pi::INT_PEND0::OFFSET), 0, "and not the issuer");
    // Node 1's engine writes node 0's memory by node field zero and tells node 0's second
    // interface.
    transfer(&b, 1, node_at(1) + 0x4000, 0x7000, 0, 1, 40);
    assert_eq!(a.chip.decode.read64(0x7008).data, 0xfeed_0001, "node field zero is node 0 for an engine");
    assert_eq!(a.r(PI1_BASE + pi::INT_PEND0::OFFSET), 1 << 40);
    assert_eq!(b.r(PI1_BASE + pi::INT_PEND0::OFFSET), 0);
    // A completion naming nobody reaches nobody.
    transfer(&a, 1, 0x4000, 0x6000, 9, 0, 33);
    assert_eq!((a.r(pi::INT_PEND0::OFFSET), b.r(pi::INT_PEND0::OFFSET)), (0, 1 << 53));
    // The engines' notification write is by number too.
    assert_eq!(b.chip.decode.write64(node_at(1) + 0x5000, 0xdead), BUS_OK);
    assert_eq!(a.chip.decode.write64(0x4000, 1), BUS_OK);
    launch(&a, io::BTE_LENGTH_STATUS::OFFSET, 1, 0x4000, 0x4000, node_at(1) + 0x5000, 0x10);
    assert_eq!(b.chip.decode.read64(node_at(1) + 0x5000).data, 0, "the final status of a good transfer");
}

/// A test chip's configuration: the Fuel's one processor and one 64 MiB
/// bank, on `clock`.
pub(super) fn test_config(clock: Clock) -> BedrockConfig<'static> {
    BedrockConfig { name: "test", clock, processors: Population::SINGLE, banks_mib: &[64], nodes: 1, events: None, prom: None }
}

/// Two chips with their NI ports connected to each other, each on a clock of its own.
pub(super) fn connected_pair() -> (Rig, Rig) {
    let (a, b) = (Rig::new(), Rig::new());
    a.chip.connect_peer(&b.chip);
    b.chip.connect_peer(&a.chip);
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
    /// turn: a thread is a processor of the node only if the node populates it.
    pub(super) fn quad() -> Self {
        Self::with(Population::populated_cpus(4).unwrap())
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
    /// Whether bank 1 bit `bit` is reflected, the same on both interfaces.
    pub(super) fn reflected(&self, bit: u32) -> bool {
        let [a, b] = [0, PI1_BASE].map(|base| self.r(base + pi::INT_PEND1::OFFSET) & (1 << bit) != 0);
        assert_eq!(a, b, "both interfaces reflect bit {bit}");
        a
    }
    /// CPU 0's IP2..IP6, bit 0 being IP2.
    pub(super) fn cause(&self) -> u64 {
        (self.word.load(Ordering::SeqCst) >> 10) & 0x1f
    }
}

/// A PI keeps its cached-alias register across the chip's reset, so the remap keeps mapping it.
#[test]
fn a_reset_keeps_the_cached_alias_the_pi_keeps() {
    let (a, b) = connected_pair();
    let alias = calias_bytes(3);
    for (rig, tag) in [(&a, 0xa), (&b, 0xb)] {
        rig.w(md::BASE + md::MEMORY_CONFIG::OFFSET, 0x11);
        rig.chip.decode.physical().banks()[0].write64(alias - 8, tag);
        rig.chip.decode.physical().banks()[0].write64(alias, tag);
    }
    let renumber = |n: u64| b.w(lb::BASE + lb::REV_ID::OFFSET, n << 32);
    b.w(pi::CALIAS_SIZE::OFFSET, 3);
    renumber(1);
    let read = |at| b.chip.decode.physical().read64(at).data;
    assert_eq!((read(alias - 8), read(alias)), (0xb, 0xa));
    Resettable::power_on(&*b.chip);
    assert_eq!(b.r(pi::CALIAS_SIZE::OFFSET), 3);
    // MD's reset turns the banks off; the PROM configures them again.
    b.w(md::BASE + md::MEMORY_CONFIG::OFFSET, 0x11);
    renumber(1);
    assert_eq!((read(alias - 8), read(alias)), (0xb, 0xa), "the alias the register still holds");
}

/// LB's region store, a PI's cached-alias store and MD's configuration store park the other
/// processor. A refresh store does not, and a store to the other node's MD parks that node's
/// processor. Remapping unparked panicked a Tezro.
#[test]
fn the_region_and_cached_alias_stores_park_the_other_processor() {
    use crate::sn::tests::ParkProbe;
    let probe = ParkProbe::new();
    let rig = Rig::new();
    rig.chip.decode.set_world(probe.world());
    assert!(probe.parks_the_other(|| rig.w(lb::BASE + lb::REV_ID::OFFSET, 1 << 32)), "the node number");
    assert!(probe.parks_the_other(|| rig.w(pi::CALIAS_SIZE::OFFSET, 3)), "the cached alias");
    assert!(probe.parks_the_other(|| rig.w(md::BASE + md::MEMORY_CONFIG::OFFSET, 0x33)), "the bank configuration");
    assert!(!probe.parks_the_other(|| rig.w(md::BASE + md::REFRESH_CONTROL::OFFSET, 0)), "a refresh write moves no memory");
    let nodes: Vec<Rig> = (0..2).map(|n| Rig::with(Population::SINGLE.with_first_cpu(n))).collect();
    nodes[0].chip.connect_peer(&nodes[1].chip);
    nodes[1].chip.connect_peer(&nodes[0].chip);
    for node in &nodes {
        node.chip.decode.set_world(probe.world());
    }
    assert!(probe.parks_the_other(|| nodes[1].w(md::BASE + md::MEMORY_CONFIG::OFFSET, 0x33)), "the other node's banks");
}

