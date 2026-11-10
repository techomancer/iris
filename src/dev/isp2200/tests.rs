//! Register, mailbox and ring contract tests for the ISP2200. Host memory is
//! a flat byte array behind the fake bus master the ISP1040 tests use.
//!
//! Ported from tests/qtest/qlogic-isp2200-test.c in
//! jamesbraid/qemu-sgi-origami.
//!
//! The qtests not ported, and why:
//!
//! * `dbrick_loop`, `dbrick_full_loop`: iris has no D-brick enclosure to fit
//!   disks into bays.
//! * `loop_plug`, `unplug_disconnect`: a target attached or removed after
//!   start is not announced to the loop (see the module comment).
//! * `cdrom_block_length`: it pins the SGI CD-ROM block-length quirk of the
//!   reference model's `scsi-cd` device, which iris's target does not carry.
//! * The migration test: iris saves and loads through `Saveable`, covered by
//!   `save_and_load_round_trip`, and does not migrate between builds.
//! * `load_ram_high`, `init_firmware_high`: they need host memory above
//!   4 GiB behind the bridge. `high_address_words_count` pins the refusal
//!   when mailbox 6 or 7 names a high address; the transfer that succeeds
//!   there is not tested.

use super::*;
use crate::dev::isp1040::tests::FakeMem;
use crate::traits::Device;
use std::cell::Cell;

const CSR_RESET: u16 = CSR_SOFT_RESET;
const ICR_ENABLE: u16 = 0x8008;
const SEM_LOCK: u16 = SEMAPHORE_LOCK;

const HCCR_RESET: u16 = 0x1000;
const HCCR_PAUSE: u16 = 0x2000;
const HCCR_RELEASE: u16 = 0x3000;
const HCCR_PARITY4: u16 = 0x4000;
const HCCR_PARITY_PAUSE: u16 = 0x4001;
const HCCR_SET_HOST: u16 = 0x5000;
const HCCR_CLEAR_HOST: u16 = 0x6000;
const HCCR_CLEAR_RISC: u16 = 0x7000;

const CODE_ORG: u16 = 0x1000;

const MBOX_BUSY: u16 = 0x0004;

pub(super) struct Rig {
    pub isp: Isp2200,
    pub mem: Arc<FakeMem>,
    /// The rings the host built: lengths, producer and consumer.
    rqlen: Cell<u16>,
    rslen: Cell<u16>,
    rq_next: Cell<u16>,
    rs_next: Cell<u16>,
    /// The adapter's clock, which only the test moves.
    now: Cell<i64>,
    /// The ring thread is blocked on a device the test holds: writes do not
    /// wait for it to settle.
    busy: Cell<bool>,
}

fn mb(n: usize) -> u32 {
    reg::mbox(n)
}

impl Rig {
    /// An adapter on empty loop with bus mastering enabled.
    pub fn bare() -> Rig {
        Rig::on(Isp2200::new())
    }

    pub fn on(isp: Isp2200) -> Rig {
        let mem = FakeMem::new();
        Device::start(&isp);
        isp.state().connect_dma(mem.clone());
        // I/O and memory decode, and bus mastering.
        isp.state().config().write(1, 0x7);
        isp.set_test_clock(0);
        Rig {
            isp,
            mem,
            rqlen: Cell::new(0),
            rslen: Cell::new(0),
            rq_next: Cell::new(0),
            rs_next: Cell::new(0),
            now: Cell::new(0),
            busy: Cell::new(false),
        }
    }

    pub fn rd(&self, off: u32) -> u16 {
        self.isp.read(off, 2) as u16
    }

    /// A register write. Requests run on the adapter's own thread, so a
    /// write waits for the ring to settle.
    pub fn wr(&self, off: u32, val: u16) {
        self.isp.write(off, u32::from(val), 2);
        self.settle();
    }

    pub fn settle(&self) {
        if !self.busy.get() {
            self.isp.wait_idle();
        }
    }

    /// Mailboxes in ascending order, then the doorbell.
    pub fn mbox(&self, input: &[u16]) -> u16 {
        for (i, &v) in input.iter().enumerate() {
            self.wr(mb(i), v);
        }
        self.wr(reg::HCCR, HCCR_SET_HOST);
        self.rd(mb(0))
    }

    pub fn call(&self, cmd: u16, m1: u16, m2: u16, m3: u16) -> u16 {
        self.mbox(&[cmd, m1, m2, m3])
    }

    /// The order both Linux and OpenBSD use: clear the latch, then release.
    pub fn ack(&self) {
        self.wr(reg::HCCR, HCCR_CLEAR_RISC);
        self.wr(reg::SEMAPHORE, 0);
    }

    pub fn call_ack(&self, cmd: u16, m1: u16, m2: u16, m3: u16) -> u16 {
        let status = self.call(cmd, m1, m2, m3);
        assert_eq!(self.rd(reg::SEMAPHORE) & SEM_LOCK, SEM_LOCK);
        self.ack();
        status
    }

    /// A stand-in image built to sum to zero over the length in its word 3.
    /// The model never runs it.
    fn image_word(i: usize) -> u16 {
        const IMAGE: [u16; 8] = [0x1234, 0x5678, 0x9abc, 8, 0x1111, 0x2222, 0x3333, 0x0000];
        if i < IMAGE.len() - 1 {
            return IMAGE[i];
        }
        IMAGE[..IMAGE.len() - 1].iter().fold(0u16, |sum, w| sum.wrapping_add(*w)).wrapping_neg()
    }

    fn load_image(&self) {
        for i in 0..8 {
            assert_eq!(
                self.call_ack(mbox::WRITE_RAM_WORD, CODE_ORG + i as u16, Self::image_word(i), 0),
                mbox::COMPLETE
            );
        }
    }

    /// What every driver does on attach: download, verify, execute.
    pub fn start(&self) {
        self.load_image();
        assert_eq!(self.call_ack(mbox::VERIFY_CHECKSUM, CODE_ORG, 0, 0), mbox::COMPLETE);
        assert_eq!(self.call_ack(mbox::EXEC_FIRMWARE, CODE_ORG, 0, 0), mbox::COMPLETE);
    }

    /// The soft reset as the drivers poll it.
    pub fn soft_reset(&self) {
        self.wr(reg::CSR, CSR_RESET);
        assert_eq!(self.rd(reg::CSR), CSR_RESET);
        assert_eq!(self.rd(reg::CSR) & CSR_RESET, 0);
    }

    /// The IRIX 6.5.30 driver's soft reset: pause the RISC, write 0x0004 to
    /// three DMA-engine control registers, set CSR bit 0, then read HCCR
    /// (never CSR) until bit 6 clears, with no HCCR write first.
    pub fn soft_reset_hccr_poll(&self) {
        self.wr(reg::HCCR, HCCR_PAUSE);
        assert_eq!(self.rd(reg::HCCR) & HCCR_RISC_PAUSED, HCCR_RISC_PAUSED);
        for off in [0x40, 0x60, 0x20] {
            self.wr(off, 0x0004);
        }
        self.wr(reg::CSR, CSR_RESET);
        assert_eq!(self.rd(reg::HCCR), HCCR_RISC_RESET);
        assert_eq!(self.rd(reg::HCCR), 0);
    }

    pub fn assert_boot_image(&self) {
        assert_eq!(self.rd(mb(0)), 0x0000);
        assert_eq!(self.rd(mb(1)), 0x4953);
        assert_eq!(self.rd(mb(2)), 0x5020);
        assert_eq!(self.rd(mb(3)), 0x2020);
        assert_eq!(self.rd(mb(7)), 0x0004);
        for i in (4..MAILBOX_COUNT).filter(|&i| i != 7) {
            assert_eq!(self.rd(mb(i)), 0, "mailbox {i}");
        }
        assert_eq!(self.rd(reg::HCCR) & (HCCR_RISC_RESET | HCCR_RISC_PAUSED), 0);
    }

    pub fn assert_held_in_reset(&self) {
        assert_eq!(self.rd(mb(0)), MBOX_BUSY);
        for i in 1..MAILBOX_COUNT {
            assert_eq!(self.rd(mb(i)), 0, "mailbox {i}");
        }
        assert_eq!(self.rd(reg::HCCR) & HCCR_RISC_RESET, HCCR_RISC_RESET);
    }

    fn read_ram(&self, addr: u16) -> u16 {
        assert_eq!(self.call(mbox::READ_RAM_WORD, addr, 0, 0), mbox::COMPLETE);
        let val = self.rd(mb(2));
        self.ack();
        val
    }
}

#[test]
fn identity() {
    let r = Rig::bare();
    let id = r.isp.state().config().read(0);
    assert_eq!((id & 0xffff, id >> 16), (0x1077, 0x2200));
    // Approximation: the Fibre Channel class; no driver reads it.
    assert_eq!(r.isp.state().config().read(2) >> 16, 0x0c04);
    // The retail board's entry in the PCI ID list.
    assert_eq!(r.isp.state().config().read(11), 0x0002_1077);
    assert_eq!((r.isp.state().config().read(15) >> 8) & 0xff, 1);
}

/// BAR 0 is I/O, BAR 1 memory; both reach one 0x100-byte register file.
#[test]
fn window() {
    let r = Rig::bare();
    assert_eq!(r.isp.state().config().read(4) & 1, 1);
    assert_eq!(r.isp.state().config().read(5) & 1, 0);
    r.isp.bar_write(0, mb(6), 0x1357, 2);
    assert_eq!(r.isp.bar_read(1, mb(6), 2), 0x1357);
    r.isp.bar_write(1, mb(23), 0x2468, 2);
    assert_eq!(r.isp.bar_read(0, mb(23), 2), 0x2468);
    assert_eq!(r.isp.bar_read(1, mb(1), 2), 0x4953);
}

/// Approximation: the registers no consumer gives a meaning -- the range
/// Linux leaves undescribed, the banked window, flash -- read 0 and take
/// writes without effect, noted. Includes NetBSD's DMA-engine writes after a
/// soft reset and OpenBSD's frame-buffer bank writes.
#[test]
fn undescribed_registers() {
    let r = Rig::bare();
    for off in [0x20, 0x40, 0x60] {
        r.wr(off, 0);
    }
    r.wr(reg::CSR, 0x0010);
    r.wr(0xb0, 0x1212);
    r.wr(0x96, 0x0001);
    r.wr(reg::FLASH_ADDR, 0x5555);
    r.wr(reg::FLASH_DATA, 0xaaaa);
    for off in (0x20..0xc0).step_by(2) {
        assert_eq!(r.rd(off), 0, "{off:#x}");
    }
    assert_eq!(r.rd(reg::FLASH_ADDR), 0);
    assert_eq!(r.rd(reg::FLASH_DATA), 0);
    assert_eq!(r.rd(0x04), 0);
    assert_eq!(r.rd(reg::CSR), 0x0010);
    r.wr(reg::CSR, 0);
    // Nothing else moved: the boot image is still there.
    r.assert_boot_image();
    // The GPIO pair is storage only.
    r.wr(reg::GPIO_DATA, 0x00c0);
    assert_eq!(r.rd(reg::GPIO_DATA), 0x00c0);
}

/// Approximation: the reset states. A device reset leaves the ROM code
/// running and its image readable. A soft reset ends at the first register
/// read after it, or at an HCCR write, with the ROM code running and the
/// image readable; only that first read, of CSR or of HCCR, still shows the
/// reset in progress. An HCCR reset holds the RISC until an explicit release.
#[test]
fn reset_states() {
    let r = Rig::bare();
    r.assert_boot_image();
    // A release sent to a running RISC changes nothing.
    r.wr(reg::HCCR, HCCR_RELEASE);
    r.assert_boot_image();

    // IRIX's order: bit 6 reads set once, then the image, not paused.
    r.soft_reset_hccr_poll();
    r.assert_boot_image();
    assert_eq!(r.rd(reg::CSR) & CSR_RESET, 0);

    // The public drivers' order: CSR polled, then their own hold.
    r.soft_reset();
    assert_eq!(r.rd(reg::HCCR) & HCCR_RISC_RESET, 0);
    r.wr(reg::HCCR, HCCR_RESET);
    r.assert_held_in_reset();
    r.wr(reg::HCCR, HCCR_RELEASE);
    r.assert_boot_image();

    // A first read of a mailbox ends the reset too, with the image.
    r.wr(reg::CSR, CSR_RESET);
    assert_eq!(r.rd(mb(0)), 0);
    assert_eq!(r.rd(reg::HCCR), 0);
    assert_eq!(r.rd(reg::CSR), 0);
    r.assert_boot_image();

    // qlogicfc's order: HCCR reset and release with no read between.
    r.wr(reg::CSR, CSR_RESET);
    r.wr(reg::HCCR, HCCR_RESET);
    r.wr(reg::HCCR, HCCR_RELEASE);
    assert_eq!(r.rd(reg::HCCR), 0);
    assert_eq!(r.rd(reg::CSR), 0);
    r.assert_boot_image();
}

/// Approximation: either soft-reset order, followed by HCCR reset, semaphore
/// 0 and release, a download and Execute, reaches running firmware. A model
/// that holds bit 6 after a soft reset fails the HCCR poll; one that skips
/// the hold after HCCR 0x1000 fails the CSR order.
#[test]
fn soft_reset_orders() {
    let pattern = [mbox::REGISTER_TEST, 0xaaaa, 0x5555, 0xaa55, 0x55aa, 0xa5a5, 0x5a5a, 0x2525];
    let r = Rig::bare();
    for order in 0..2 {
        if order == 0 {
            r.soft_reset_hccr_poll();
            assert_eq!(r.rd(mb(0)), 0);
        } else {
            r.soft_reset();
        }
        r.wr(reg::HCCR, HCCR_RESET);
        assert_eq!(r.rd(reg::HCCR), HCCR_RISC_RESET);
        assert_eq!(r.rd(mb(0)), MBOX_BUSY);
        r.wr(reg::SEMAPHORE, 0);
        r.wr(reg::HCCR, HCCR_RELEASE);
        if order == 0 {
            r.wr(reg::HCCR, HCCR_PARITY_PAUSE);
        }
        r.assert_boot_image();

        r.start();
        assert_eq!(r.call_ack(mbox::NOP, 0, 0, 0), mbox::COMPLETE);
        assert_eq!(r.mbox(&pattern), mbox::COMPLETE);
        for (i, &v) in pattern.iter().enumerate().skip(1) {
            assert_eq!(r.rd(mb(i)), v);
        }
        r.ack();
        // The firmware runs: a ROM-subset reset did not hide it.
        assert_eq!(r.call_ack(mbox::GET_FW_STATE, 0, 0, 0), mbox::COMPLETE);
    }
}

/// HCCR reset holds, release boots, pause defers a mailbox command.
#[test]
fn risc_control() {
    let r = Rig::bare();
    r.wr(reg::HCCR, HCCR_RESET);
    r.assert_held_in_reset();
    r.wr(reg::HCCR, HCCR_RELEASE);
    r.assert_boot_image();

    r.wr(reg::HCCR, HCCR_PAUSE);
    assert_eq!(r.rd(reg::HCCR), HCCR_RISC_PAUSED);
    assert_eq!(r.rd(mb(0)), MBOX_BUSY);
    assert_eq!(r.rd(mb(1)), 0x4953);

    // Issued while paused, the command waits.
    assert_eq!(r.call(mbox::NOP, 0, 0, 0), mbox::NOP);
    assert_eq!(r.rd(reg::SEMAPHORE), 0);
    assert_eq!(r.rd(reg::ISR), 0);

    r.wr(reg::HCCR, HCCR_RELEASE);
    assert_eq!(r.rd(reg::HCCR) & HCCR_RISC_PAUSED, 0);
    assert_eq!(r.rd(mb(0)), mbox::COMPLETE);
    assert_eq!(r.rd(reg::SEMAPHORE), SEM_LOCK);
    r.ack();
}

/// The firmware clears the doorbell when it takes the command, so a driver
/// that waits for bit 7 to drop (the stock PROM's) never has to write the
/// clear command. A command that cannot run yet keeps the bit set, and the
/// clear command still clears it.
#[test]
fn host_interrupt() {
    let r = Rig::bare();
    r.call(mbox::NOP, 0, 0, 0);
    assert_eq!(r.rd(reg::HCCR) & HCCR_HOST_INT, 0);
    r.ack();

    r.wr(reg::HCCR, HCCR_PAUSE);
    r.call(mbox::NOP, 0, 0, 0);
    assert_eq!(r.rd(reg::HCCR) & HCCR_HOST_INT, HCCR_HOST_INT);
    r.wr(reg::HCCR, HCCR_RELEASE);
    assert_eq!(r.rd(reg::HCCR) & HCCR_HOST_INT, 0);
    r.ack();

    r.wr(reg::HCCR, HCCR_PAUSE);
    r.call(mbox::NOP, 0, 0, 0);
    r.wr(reg::HCCR, HCCR_CLEAR_HOST);
    assert_eq!(r.rd(reg::HCCR) & HCCR_HOST_INT, 0);
}

/// Approximation: host commands without a model, and 0x4000/0x4001 which the
/// drivers write between reset and release, change neither the HCCR bits nor
/// a mailbox, running or held.
#[test]
fn unmodelled_host_commands() {
    let r = Rig::bare();
    let before: Vec<u16> = (0..MAILBOX_COUNT).map(|i| r.rd(mb(i))).collect();
    let hccr = r.rd(reg::HCCR);
    for cmd in [0xa000, 0xe000, 0x8000, 0x9000, 0xf000, HCCR_PARITY4, HCCR_PARITY_PAUSE] {
        r.wr(reg::HCCR, cmd);
        assert_eq!(r.rd(reg::HCCR), hccr);
        let after: Vec<u16> = (0..MAILBOX_COUNT).map(|i| r.rd(mb(i))).collect();
        assert_eq!(after, before, "{cmd:#06x}");
    }

    // OpenBSD's order: soft reset, reset, semaphore, 0x4000, release.
    r.soft_reset();
    r.wr(reg::HCCR, HCCR_RESET);
    r.wr(reg::SEMAPHORE, 0);
    r.wr(reg::HCCR, HCCR_PARITY4);
    r.assert_held_in_reset();
    assert_eq!(r.rd(reg::HCCR) & HCCR_RISC_PAUSED, 0);
    r.wr(reg::HCCR, HCCR_RELEASE);
    r.assert_boot_image();

    // NetBSD's: reset, 0x4001, release.
    r.wr(reg::HCCR, HCCR_RESET);
    r.wr(reg::HCCR, HCCR_PARITY_PAUSE);
    r.assert_held_in_reset();
    r.wr(reg::HCCR, HCCR_RELEASE);
    r.assert_boot_image();
}

/// Echoes m1..m7 for Linux's pattern and another, before any queue.
#[test]
fn register_test() {
    let patterns: [[u16; 8]; 2] = [
        [mbox::REGISTER_TEST, 0xaaaa, 0x5555, 0xaa55, 0x55aa, 0xa5a5, 0x5a5a, 0x2525],
        [mbox::REGISTER_TEST, 0x0102, 0x0304, 0x0506, 0x0708, 0x090a, 0x0b0c, 0x0d0e],
    ];
    let r = Rig::bare();
    for pattern in patterns {
        assert_eq!(r.mbox(&pattern), mbox::COMPLETE);
        for (i, &v) in pattern.iter().enumerate().skip(1) {
            assert_eq!(r.rd(mb(i)), v);
        }
        r.ack();
    }
}

/// A completion: status, semaphore, ISR. The line follows the latch with
/// both enables set (ISR bit 15 mirrors the line) and stays low with both
/// clear; the single-enable cases are not pinned.
#[test]
fn completion_interrupt() {
    let r = Rig::bare();
    r.wr(reg::ICR, ICR_ENABLE);
    assert_eq!(r.rd(reg::ICR), ICR_ENABLE);
    assert_eq!(r.call(mbox::NOP, 0, 0, 0), mbox::COMPLETE);
    assert_eq!(r.rd(reg::SEMAPHORE), SEM_LOCK);
    assert_eq!(r.rd(reg::ISR), ISR_PENDING | ISR_RISC);
    assert!(r.isp.interrupting());

    r.wr(reg::HCCR, HCCR_CLEAR_RISC);
    assert_eq!(r.rd(reg::ISR), 0);
    assert_eq!(r.rd(reg::SEMAPHORE), SEM_LOCK);
    r.wr(reg::SEMAPHORE, 0);
    assert_eq!(r.rd(reg::SEMAPHORE), 0);
    assert!(!r.isp.interrupting());

    // Both enables clear: the latch sets, the line does not.
    r.wr(reg::ICR, 0);
    assert_eq!(r.call(mbox::NOP, 0, 0, 0), mbox::COMPLETE);
    assert_eq!(r.rd(reg::ISR), ISR_RISC);
    r.ack();
}

/// The line is written when the bus attaches the adapter, and again each
/// time, because the controller's input may not be what the last write left;
/// accesses that leave the level alone write nothing.
#[test]
fn the_line_is_rewritten_after_the_bus_attaches() {
    use crate::irq::{IrqLine, LevelProbe};
    let r = Rig::bare();
    let (probe, line) = LevelProbe::line();
    r.isp.state().connect_irq(line);
    r.isp.attached();
    assert_eq!(probe.writes(), [false]);
    r.wr(reg::ICR, ICR_ENABLE);
    assert_eq!(r.call(mbox::NOP, 0, 0, 0), mbox::COMPLETE);
    assert_eq!(probe.writes(), [false, true]);
    r.wr(reg::SEMAPHORE, SEM_LOCK);
    assert_eq!(probe.writes(), [false, true], "an access that leaves the level alone writes nothing");

    IrqLine::new(probe.clone(), 0).set(false);
    r.isp.attached();
    assert!(probe.level(), "the line is put back at the adapter's level");
    assert_eq!(probe.writes(), [false, true, false, true]);
}

/// A second thread clears the interrupt while a first one's rise is held in
/// the parent. The adapter drives the line in the hold that changed its
/// state, so the fall waits for the rise to be delivered and the parent sees
/// them in order. A driver that read the level in one hold and wrote the line
/// outside it would deliver the fall first.
#[test]
fn a_fall_does_not_overtake_a_rise_held_in_the_parent() {
    use crate::irq::StallingProbe;
    let r = Rig::bare();
    let (stall, line) = StallingProbe::line();
    r.isp.state().connect_irq(line);
    r.isp.attached();
    r.wr(reg::ICR, ICR_ENABLE);
    r.wr(mb(0), mbox::NOP);
    let (started, go) = stall.arm();
    std::thread::scope(|scope| {
        scope.spawn(|| r.isp.write(reg::HCCR, u32::from(HCCR_SET_HOST), 2));
        started.recv_timeout(std::time::Duration::from_secs(5)).expect("the rise is in the parent");
        let c = scope.spawn(|| r.isp.write(reg::HCCR, u32::from(HCCR_CLEAR_RISC), 2));
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert!(!c.is_finished(), "the fall waits for the rise to be delivered");
        go.send(()).unwrap();
    });
    r.isp.wait_idle();
    assert_eq!(stall.probe.writes(), [false, true, false]);
    assert!(!stall.probe.level());
    assert!(!r.isp.interrupting());
}

/// Four threads raise and clear the interrupt as fast as they can. Every
/// write the parent sees is a change of level, and the line ends where the
/// adapter's state says.
#[test]
fn racing_register_writers_deliver_only_real_edges() {
    use crate::irq::LevelProbe;
    let r = Rig::bare();
    let (probe, line) = LevelProbe::line();
    r.isp.state().connect_irq(line);
    r.isp.attached();
    r.wr(reg::ICR, ICR_ENABLE);
    r.wr(mb(0), mbox::NOP);
    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| {
                for _ in 0..500 {
                    r.isp.write(reg::HCCR, u32::from(HCCR_SET_HOST), 2);
                    r.isp.write(reg::HCCR, u32::from(HCCR_CLEAR_RISC), 2);
                }
            });
        }
    });
    r.isp.wait_idle();
    let writes = probe.writes();
    assert!(writes.windows(2).all(|w| w[0] != w[1]), "a repeated level reached the parent: {writes:?}");
    assert_eq!(probe.level(), r.isp.interrupting());
}

/// Approximation: a second result waits until both the semaphore and the
/// latch are released, then appears intact. The first stays readable under
/// the second command's writes until then.
#[test]
fn second_result_waits() {
    let second = [mbox::REGISTER_TEST, 0x1111, 0x2222, 0x3333, 0x4444, 0x5555, 0x6666, 0x7777];
    let r = Rig::bare();
    assert_eq!(r.call(mbox::ABOUT_FIRMWARE, 0, 0, 0), mbox::COMPLETE);
    assert_eq!(r.mbox(&second), mbox::COMPLETE);
    assert_eq!(r.rd(mb(1)), 1);

    r.wr(reg::HCCR, HCCR_CLEAR_RISC);
    assert_eq!(r.rd(reg::ISR), 0);
    assert_eq!(r.rd(mb(0)), mbox::COMPLETE);
    r.wr(reg::SEMAPHORE, 0);

    assert_eq!(r.rd(reg::SEMAPHORE), SEM_LOCK);
    assert_eq!(r.rd(reg::ISR), ISR_RISC);
    assert_eq!(r.rd(mb(0)), mbox::COMPLETE);
    for (i, &v) in second.iter().enumerate().skip(1) {
        assert_eq!(r.rd(mb(i)), v);
    }
    r.ack();
}

/// Approximation: results are never dropped. One completion waits behind a
/// posted result; a doorbell that finds that slot taken stays latched in
/// HCCR bit 7 and runs once the slot empties, on the mailbox inputs as they
/// are then. Ringing again before that runs one command, not two.
#[test]
fn doorbell_waits_for_slot() {
    let r = Rig::bare();
    assert_eq!(r.call(mbox::NOP, 0, 0, 0), mbox::COMPLETE);

    // Rounds 0..5: round 0 takes the slot, the rest wait on the latch.
    for round in 0..6u16 {
        let mut input = [0u16; 8];
        input[0] = mbox::REGISTER_TEST;
        for (i, v) in input.iter_mut().enumerate().skip(1) {
            *v = (round << 8) | i as u16;
        }
        r.mbox(&input);
        // Round 0 is taken at once; the others find the slot full.
        assert_eq!(r.rd(reg::HCCR) & HCCR_HOST_INT, if round != 0 { HCCR_HOST_INT } else { 0 });
    }

    // The posted No-op's release posts round 0 intact.
    r.ack();
    assert_eq!(r.rd(reg::SEMAPHORE), SEM_LOCK);
    assert_eq!(r.rd(mb(0)), mbox::COMPLETE);
    for i in 1..8 {
        assert_eq!(r.rd(mb(i)), i as u16);
    }

    // Then the latched doorbell, which ran on round 5's inputs.
    r.ack();
    assert_eq!(r.rd(reg::SEMAPHORE), SEM_LOCK);
    assert_eq!(r.rd(mb(0)), mbox::COMPLETE);
    for i in 1..8 {
        assert_eq!(r.rd(mb(i)), (5 << 8) | i as u16);
    }

    r.ack();
    assert_eq!(r.rd(reg::SEMAPHORE), 0);
    assert_eq!(r.rd(reg::ISR), 0);
    assert_eq!(r.call_ack(mbox::NOP, 0, 0, 0), mbox::COMPLETE);
}

/// Approximation: a semaphore write of 1 changes nothing; bit 1 reads 0.
#[test]
fn semaphore_write_one() {
    let r = Rig::bare();
    r.wr(reg::SEMAPHORE, 1);
    assert_eq!(r.rd(reg::SEMAPHORE), 0);
    r.wr(reg::SEMAPHORE, 3);
    assert_eq!(r.rd(reg::SEMAPHORE), 0);

    r.call(mbox::NOP, 0, 0, 0);
    r.wr(reg::SEMAPHORE, 1);
    assert_eq!(r.rd(reg::SEMAPHORE), SEM_LOCK);
    r.ack();
}

/// Approximation: unknown and deferred commands answer 0x4001 and leave the
/// running firmware alone, as do the commands the loop service does not
/// model: abort device, bus reset, target reset, login, enhanced port
/// database.
#[test]
fn unknown_commands() {
    let r = Rig::bare();
    r.start();
    for cmd in [0x00ee, 0x0016, 0x0018, 0x0066, 0x0042, 0x006f, 0x0047] {
        assert_eq!(r.call_ack(cmd, 1, 0, 0), mbox::INVALID, "{cmd:#06x}");
    }
    assert_eq!(r.call_ack(mbox::GET_FW_STATE, 0, 0, 0), mbox::COMPLETE);
}

/// Approximation: before execute only the ROM subset answers; after it Get
/// Firmware State reports configuration wait, never ready.
#[test]
fn rom_subset() {
    let r = Rig::bare();
    for cmd in [mbox::GET_FW_STATE, mbox::FIRMWARE_STATUS, mbox::INIT_FIRMWARE, mbox::SET_FW_OPTIONS] {
        assert_eq!(r.call_ack(cmd, 0, 0, 0), mbox::INVALID, "{cmd:#06x}");
    }
    assert_eq!(r.call_ack(mbox::NOP, 0, 0, 0), mbox::COMPLETE);
    assert_eq!(r.call_ack(mbox::ABOUT_FIRMWARE, 0, 0, 0), mbox::COMPLETE);

    r.start();
    assert_eq!(r.call(mbox::GET_FW_STATE, 0, 0, 0), mbox::COMPLETE);
    assert_eq!(r.rd(mb(1)), 0);
    r.ack();
}

impl Rig {
    fn load_ram(&self, cmd: u16, base: u16, host: u64, count: u16, m8: u16) -> u16 {
        let mut input = [0u16; 9];
        input[0] = cmd;
        input[1] = base;
        input[2] = (host >> 16) as u16;
        input[3] = host as u16;
        input[4] = count;
        input[6] = (host >> 48) as u16;
        input[7] = (host >> 32) as u16;
        input[8] = m8;
        self.mbox(&input)
    }
}

/// The short Load RAM, which the stock PROM uses before any firmware runs,
/// takes its host address from m2/m3 alone and works from the boot ROM.
#[test]
fn load_ram_32() {
    let r = Rig::bare();
    let src = 0x4000u32;
    r.mem.put(src, &[0x34, 0x12, 0x78, 0x56]);
    // Left over from earlier commands; the short form must not use them.
    let input = [mbox::LOAD_RAM_32, 0x1000, (src >> 16) as u16, src as u16, 2, 0, 0xabcd, 0x1234];
    assert_eq!(r.mbox(&input), mbox::COMPLETE);
    r.ack();
    assert_eq!(r.read_ram(0x1000), 0x1234);
    assert_eq!(r.read_ram(0x1001), 0x5678);
}

/// Dump RAM, which the stock PROM's DMA self-test runs against the boot ROM
/// before any firmware: the words Load RAM stored come back as the same
/// little-endian bytes, and a range past RAM is refused.
#[test]
fn dump_ram() {
    let bytes = [0x34, 0x12, 0x78, 0x56, 0xbc, 0x9a, 0xf0, 0xde];
    let r = Rig::bare();
    r.mem.put(0x4000, &bytes);
    assert_eq!(r.load_ram(mbox::LOAD_RAM, 0x1000, 0x4000, 4, 0), mbox::COMPLETE);
    r.ack();
    assert_eq!(r.load_ram(mbox::DUMP_RAM, 0x1000, 0x5000, 4, 0), mbox::COMPLETE);
    r.ack();
    assert_eq!(r.mem.get(0x5000, 8), bytes);
    assert_eq!(r.load_ram(mbox::DUMP_RAM, 0xfffe, 0x5000, 4, 0), mbox::PARAM_ERROR);
    r.ack();
}

/// Approximation: Load RAM moves little-endian words; a refused DMA changes
/// nothing; a range past RAM is refused.
#[test]
fn load_ram() {
    let bytes = [0x34, 0x12, 0x78, 0x56, 0xbc, 0x9a, 0xf0, 0xde];
    let r = Rig::bare();
    r.mem.put(0x4000, &bytes);
    assert_eq!(r.load_ram(mbox::LOAD_RAM, 0x2000, 0x4000, 4, 0), mbox::COMPLETE);
    r.ack();
    assert_eq!(r.read_ram(0x2000), 0x1234);
    assert_eq!(r.read_ram(0x2001), 0x5678);
    assert_eq!(r.read_ram(0x2003), 0xdef0);

    // Past the end of host memory: refused, RAM unchanged.
    assert_eq!(r.load_ram(mbox::LOAD_RAM, 0x2000, 1 << 30, 4, 0), mbox::HOST_INTERFACE_ERROR);
    r.ack();
    assert_eq!(r.read_ram(0x2000), 0x1234);
    assert_eq!(r.read_ram(0x2003), 0xdef0);

    assert_eq!(r.load_ram(mbox::LOAD_RAM, 0xfffe, 0x4000, 4, 0), mbox::PARAM_ERROR);
    r.ack();
    assert_eq!(r.read_ram(0xfffe), 0);

    // Approximation: the extended load does not reach past 64K words.
    assert_eq!(r.load_ram(mbox::LOAD_RAM_EXTENDED, 0x3000, 0x4000, 4, 1), mbox::PARAM_ERROR);
    r.ack();
    assert_eq!(r.read_ram(0x3000), 0);
    assert_eq!(r.load_ram(mbox::LOAD_RAM_EXTENDED, 0x3000, 0x4000, 4, 0), mbox::COMPLETE);
    r.ack();
    assert_eq!(r.read_ram(0x3000), 0x1234);
}

/// Write RAM word loads one word. With 64K words and a 16-bit address in
/// mailbox 1, no Write RAM word address lies past RAM, so that refusal has no
/// input to test.
#[test]
fn write_ram_word() {
    let r = Rig::bare();
    assert_eq!(r.call_ack(mbox::WRITE_RAM_WORD, 0xffff, 0xbeef, 0), mbox::COMPLETE);
    assert_eq!(r.read_ram(0xffff), 0xbeef);
    assert_eq!(r.read_ram(0x0000), 0);
}

/// Approximation: the sum over the word-3 length must be zero; a header or
/// span past RAM is refused, leaving mailbox 1 as written.
#[test]
fn verify_checksum() {
    let r = Rig::bare();
    r.load_image();
    assert_eq!(r.call(mbox::VERIFY_CHECKSUM, CODE_ORG, 0, 0), mbox::COMPLETE);
    assert_eq!(r.rd(mb(1)), 0);
    assert_eq!(r.rd(mb(2)), 0);
    r.ack();

    assert_eq!(r.call_ack(mbox::WRITE_RAM_WORD, CODE_ORG + 5, Rig::image_word(5) ^ 0x0100, 0), mbox::COMPLETE);
    assert_eq!(r.call_ack(mbox::VERIFY_CHECKSUM, CODE_ORG, 0, 0), mbox::TEST_FAILED);

    // A length that runs past the end of RAM.
    assert_eq!(r.call_ack(mbox::WRITE_RAM_WORD, 0xfff3, 0x0020, 0), mbox::COMPLETE);
    assert_eq!(r.call(mbox::VERIFY_CHECKSUM, 0xfff0, 0, 0), mbox::PARAM_ERROR);
    assert_eq!(r.rd(mb(1)), 0xfff0);
    r.ack();
    // A header that does not fit.
    assert_eq!(r.call_ack(mbox::VERIFY_CHECKSUM, 0xfffd, 0, 0), mbox::PARAM_ERROR);
}

/// Approximation: the fixed identity -- a model version, attributes 0x0002
/// (expanded LUN; no fabric 0x0004, no 2K logins 0x0100) -- and a budget of
/// 512 commands.
#[test]
fn firmware_identity() {
    let r = Rig::bare();
    r.start();
    assert_eq!(r.call(mbox::ABOUT_FIRMWARE, 0, 0, 0), mbox::COMPLETE);
    assert_eq!([r.rd(mb(1)), r.rd(mb(2)), r.rd(mb(3))], [1, 0, 0]);
    assert_eq!(r.rd(mb(6)), 0x0002);
    assert_eq!(r.rd(mb(6)) & 0x0104, 0);
    r.ack();
    assert_eq!(r.call(mbox::FIRMWARE_STATUS, 0, 0, 0), mbox::COMPLETE);
    assert_eq!(r.rd(mb(2)), 512);
    r.ack();
}

/// Approximation: firmware options are stored and change nothing.
#[test]
fn firmware_options() {
    let r = Rig::bare();
    r.start();
    assert_eq!(r.call(mbox::GET_FW_OPTIONS, 0, 0, 0), mbox::COMPLETE);
    assert_eq!(r.rd(mb(1)), 0);
    r.ack();
    assert_eq!(r.call_ack(mbox::SET_FW_OPTIONS, 0x0002, 0x1234, 0x5678), mbox::COMPLETE);
    assert_eq!(r.call(mbox::GET_FW_OPTIONS, 0, 0, 0), mbox::COMPLETE);
    assert_eq!([r.rd(mb(1)), r.rd(mb(2)), r.rd(mb(3))], [0x0002, 0x1234, 0x5678]);
    r.ack();

    r.soft_reset();
    r.wr(reg::HCCR, HCCR_RELEASE);
    r.start();
    assert_eq!(r.call(mbox::GET_FW_OPTIONS, 0, 0, 0), mbox::COMPLETE);
    assert_eq!([r.rd(mb(1)), r.rd(mb(2)), r.rd(mb(3))], [0, 0, 0]);
    r.ack();
}

const ICB_AT: u32 = 0x3000;
const RQST_AT: u32 = 0x4000;
const RESP_AT: u32 = 0x5000;

fn icb_build(rqst_len: u16, resp_len: u16, rqst: u64, resp: u64, options: u16) -> [u8; ICB_SIZE] {
    let mut icb = [0u8; ICB_SIZE];
    icb[0] = 1;
    icb[2..4].copy_from_slice(&options.to_le_bytes());
    icb[4..6].copy_from_slice(&1024u16.to_le_bytes());
    icb[36..38].copy_from_slice(&rqst_len.to_le_bytes());
    icb[38..40].copy_from_slice(&resp_len.to_le_bytes());
    icb[40..48].copy_from_slice(&rqst.to_le_bytes());
    icb[48..56].copy_from_slice(&resp.to_le_bytes());
    icb
}

impl Rig {
    /// INIT_FIRMWARE in Linux's input form: mapping D in m2, m3, m6, m7.
    fn init_firmware(&self, icb: u64) -> u16 {
        self.mbox(&[mbox::INIT_FIRMWARE, 0, (icb >> 16) as u16, icb as u16, 0, 0, (icb >> 48) as u16, (icb >> 32) as u16])
    }

    fn try_icb(&self, buf: &[u8; ICB_SIZE]) -> u16 {
        self.mem.put(ICB_AT, buf);
        let status = self.init_firmware(u64::from(ICB_AT));
        self.ack();
        status
    }

    fn assert_queues_uninitialised(&self) {
        self.wr(reg::REQUEST_QUEUE, 0x1234);
        assert_eq!(self.rd(reg::REQUEST_QUEUE), 0x1234);
    }
}

/// Approximation: each rejection leaves the queues uninitialised. Precedence,
/// which the sources leave open, is the model's: an unreadable block first,
/// then the field checks.
#[test]
fn init_firmware_rejects() {
    let r = Rig::bare();
    r.start();
    let (rq, rs) = (u64::from(RQST_AT), u64::from(RESP_AT));

    let mut buf = icb_build(8, 8, rq, rs, 0);
    buf[0] = 2;
    assert_eq!(r.try_icb(&buf), mbox::PARAM_ERROR);
    r.assert_queues_uninitialised();
    for buf in [
        icb_build(0, 8, rq, rs, 0),
        icb_build(8, 0, rq, rs, 0),
        icb_build(8, 8, 0, rs, 0),
        icb_build(8, 8, rq, rs, 0x0008),
    ] {
        assert_eq!(r.try_icb(&buf), mbox::PARAM_ERROR);
        r.assert_queues_uninitialised();
    }

    assert_eq!(r.init_firmware(1 << 30), mbox::HOST_INTERFACE_ERROR);
    r.ack();
    r.assert_queues_uninitialised();

    assert_eq!(r.try_icb(&icb_build(8, 8, rq, rs, 0)), mbox::COMPLETE);
    assert_eq!(r.rd(reg::REQUEST_QUEUE), 0);
}

/// The four address mailboxes place a transfer at a 64-bit address, and
/// mailbox 6 carries bits 63:48. A bridge that has nothing at the address
/// refuses it, so a block or a load that names a high word is refused and
/// leaves the adapter as it was.
#[test]
fn high_address_words_count() {
    let r = Rig::bare();
    r.start();
    let high = 0xe000u64 << 48;
    r.mem.put(ICB_AT, &icb_build(4, 4, u64::from(RQST_AT), u64::from(RESP_AT), 0));
    assert_eq!(r.init_firmware(high | u64::from(ICB_AT)), mbox::HOST_INTERFACE_ERROR);
    r.ack();
    r.assert_queues_uninitialised();
    r.mem.put(0x4000, &[0xcd, 0xab, 0x01, 0xef]);
    assert_eq!(r.load_ram(mbox::LOAD_RAM, 0x2000, high | 0x4000, 2, 0), mbox::HOST_INTERFACE_ERROR);
    r.ack();
    assert_eq!(r.read_ram(0x2000), 0);
    assert_eq!(r.load_ram(mbox::LOAD_RAM, 0x2000, 0x4000, 2, 0), mbox::COMPLETE);
    r.ack();
    assert_eq!(r.read_ram(0x2000), 0xabcd);
    assert_eq!(r.read_ram(0x2001), 0xef01);
}

/// Mailboxes mailbox 4 and 5 are also the request and response queue
/// pointers once INIT_FIRMWARE has built the rings. Approximation: every
/// mailbox reads back before INIT_FIRMWARE; after it, mailbox 4 and 5 read
/// the adapter's pointers while the semaphore is free and a result's data
/// while it is held. No mailbox write starts a command.
#[test]
fn mailbox_storage() {
    let r = Rig::bare();
    for i in 0..MAILBOX_COUNT {
        r.wr(mb(i), 0x1000 + i as u16);
    }
    for i in 0..MAILBOX_COUNT {
        assert_eq!(r.rd(mb(i)), 0x1000 + i as u16);
    }
    assert_eq!(r.rd(reg::SEMAPHORE), 0);
    assert_eq!(r.rd(reg::ISR), 0);

    r.start();
    assert_eq!(r.try_icb(&icb_build(8, 8, u64::from(RQST_AT), u64::from(RESP_AT), 0)), mbox::COMPLETE);
    for i in (0..MAILBOX_COUNT).filter(|&i| i != 4 && i != 5) {
        r.wr(mb(i), 0x2000 + i as u16);
    }
    for i in (0..MAILBOX_COUNT).filter(|&i| i != 4 && i != 5) {
        assert_eq!(r.rd(mb(i)), 0x2000 + i as u16);
    }
    // The window is open (mailbox 0 was written): these are inputs.
    r.wr(reg::REQUEST_QUEUE, 0x0bad);
    r.wr(reg::RESPONSE_QUEUE, 0x0bad);
    assert_eq!(r.rd(reg::REQUEST_QUEUE), 0);
    assert_eq!(r.rd(reg::RESPONSE_QUEUE), 0);

    // With a result held, the two mailboxes are its data: zero here.
    r.wr(mb(0), mbox::NOP);
    r.wr(reg::HCCR, HCCR_SET_HOST);
    assert_eq!(r.rd(reg::SEMAPHORE), SEM_LOCK);
    assert_eq!(r.rd(reg::REQUEST_QUEUE), 0);
    r.ack();
    assert_eq!(r.rd(reg::REQUEST_QUEUE), 0);
}

mod nvram_tests;
