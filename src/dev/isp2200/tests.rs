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
use crate::irq::IrqLine;
use crate::traits::Device;
use crate::test_temp::TempPath;
use std::cell::Cell;

const CSR_RESET: u16 = CSR_SOFT_RESET;
const ICR_ENABLE: u16 = 0x8008;
const SEM_LOCK: u16 = SEMAPHORE_LOCK;

const HCCR_RESET: u16 = 0x1000;
const HCCR_PAUSE: u16 = 0x2000;
const HCCR_RELEASE: u16 = 0x3000;
const HCCR_PARITY4: u16 = 0x4000;
const HCCR_PARITY_PAUSE: u16 = 0x4001;
pub(crate) const HCCR_SET_HOST: u16 = 0x5000;
const HCCR_CLEAR_HOST: u16 = 0x6000;
pub(crate) const HCCR_CLEAR_RISC: u16 = 0x7000;

pub(crate) const CODE_ORG: u16 = 0x1000;

const MBOX_BUSY: u16 = 0x0004;

pub(crate) struct Rig {
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
        let r = Rig::wired(isp);
        Device::start(&r.isp);
        r
    }

    /// Like [`Rig::on`], with the interrupt line wired before the ring
    /// thread starts. The thread drives the line whenever it waits, so a line
    /// connected after it started can be written once by the thread and again
    /// by `attached`, and a test that reads the exact writes would see a
    /// repeat that belongs to the rig and not to the adapter.
    pub fn on_line(isp: Isp2200, line: IrqLine) -> Rig {
        let r = Rig::wired(isp);
        r.isp.state().connect_irq(line);
        r.isp.attached();
        Device::start(&r.isp);
        r
    }

    /// The adapter on its bus with its thread not yet started.
    fn wired(isp: Isp2200) -> Rig {
        let mem = FakeMem::new();
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

    /// An adapter with a small disk at each of `ids`.
    pub fn with_disks(name: &str, ids: &[u8]) -> Rig {
        let r = Rig::bare();
        for &id in ids {
            let path = TempPath::file(&format!("isp2200-{name}-{id}"), &vec![0u8; 64 * isp1040::BLOCK_SIZE]);
            r.isp.bus().attach_image(id, &path, false).expect("attach");
        }
        r
    }

    /// Download, execute and initialise the rings, and have the loop up.
    pub fn ready(&self) {
        self.loop_up(ICB_OPT_PORT_DB_EVENTS, 0);
    }

    /// The event now posted: its code, and m1..m3 into the array. Then
    /// released.
    pub fn take_event(&self) -> (u16, [u16; 3]) {
        assert_eq!(self.rd(reg::SEMAPHORE) & SEM_LOCK, SEM_LOCK);
        assert_eq!(self.rd(reg::ISR) & ISR_RISC, ISR_RISC);
        let code = self.rd(mb(0));
        let m = [self.rd(mb(1)), self.rd(mb(2)), self.rd(mb(3))];
        self.ack();
        (code, m)
    }

    pub fn expect_no_event(&self) {
        assert_eq!(self.rd(reg::SEMAPHORE), 0);
        assert_eq!(self.rd(reg::ISR), 0);
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
    pub(crate) fn image_word(i: usize) -> u16 {
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

/// Soft reset clears the interface, the rings and the work in flight.
#[test]
fn soft_reset_clears() {
    let r = Rig::bare();
    r.start();
    r.rings(8, 8);
    r.wr(reg::ICR, ICR_ENABLE);

    // One completion posted and unread, a second held behind its latch.
    r.publish(&entry_command(ENTRY_CMD64, 0x1111, 1, 0, 0));
    r.publish(&entry_command(ENTRY_CMD64, 0x2222, 1, 0, 0));
    assert_eq!(r.rd(reg::ISR), ISR_PENDING | ISR_RISC);

    r.soft_reset();
    assert_eq!(r.rd(reg::ICR), 0);
    assert_eq!(r.rd(reg::ISR), 0);
    assert_eq!(r.rd(reg::SEMAPHORE), 0);
    assert_eq!(r.rd(reg::NVRAM), 0);
    r.wr(reg::HCCR, HCCR_RELEASE);

    // The rings are gone: mailbox 4 is plain storage again.
    r.wr(reg::REQUEST_QUEUE, 0x55aa);
    assert_eq!(r.rd(reg::REQUEST_QUEUE), 0x55aa);

    // The firmware is gone too, and comes back with a download.
    assert_eq!(r.call_ack(mbox::GET_FW_STATE, 0, 0, 0), mbox::INVALID);
    r.start();
    r.rings(8, 8);
    // The held completion did not survive into the new ring.
    assert_eq!(r.response(0)[0], 0);
    assert_eq!(r.rd(reg::ISR), 0);
    r.tur(0x3333);
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
    use crate::irq::LevelProbe;
    let (probe, line) = LevelProbe::line();
    let r = Rig::on_line(Isp2200::new(), line);
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
    let (stall, line) = StallingProbe::line();
    let r = Rig::on_line(Isp2200::new(), line);
    r.wr(reg::ICR, ICR_ENABLE);
    r.wr(mb(0), mbox::NOP);
    let (started, go) = stall.arm();
    std::thread::scope(|scope| {
        scope.spawn(|| r.isp.write(reg::HCCR, u32::from(HCCR_SET_HOST), 2));
        started.recv_timeout(std::time::Duration::from_secs(5)).expect("the rise is in the parent");
        let c = scope.spawn(|| r.isp.write(reg::HCCR, u32::from(HCCR_CLEAR_RISC), 2));
        // As the ISP1040's equivalent test: the sleep only backs a negative
        // assertion after the handshake above, so it can pass falsely but
        // never fail spuriously.
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
    let (probe, line) = LevelProbe::line();
    let r = Rig::on_line(Isp2200::new(), line);
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

pub(crate) const ICB_AT: u32 = 0x3000;
pub(crate) const RQST_AT: u32 = 0x4000;
pub(crate) const RESP_AT: u32 = 0x5000;

pub(crate) fn icb_build(rqst_len: u16, resp_len: u16, rqst: u64, resp: u64, options: u16) -> [u8; ICB_SIZE] {
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

const ENTRY_CMD32: u8 = ENTRY_TYPE_COMMAND;
pub(crate) const ENTRY_CMD64: u8 = ENTRY_TYPE_COMMAND_A64;
const ENTRY_CMD_LIST: u8 = ENTRY_TYPE_COMMAND_LIST;
const ENTRY_CONT32: u8 = ENTRY_TYPE_CONTINUATION;
const ENTRY_CONT64: u8 = ENTRY_TYPE_CONTINUATION_A64;
pub(crate) const ENTRY_STATUS: u8 = ENTRY_TYPE_STATUS;
const ENTRY_MARKER: u8 = ENTRY_TYPE_MARKER;

pub(crate) type Entry = [u8; ENTRY_SIZE];

pub(crate) fn p16(e: &mut [u8], off: usize, v: u16) {
    e[off..off + 2].copy_from_slice(&v.to_le_bytes());
}
pub(crate) fn p32(e: &mut [u8], off: usize, v: u32) {
    e[off..off + 4].copy_from_slice(&v.to_le_bytes());
}
fn p64(e: &mut [u8], off: usize, v: u64) {
    e[off..off + 8].copy_from_slice(&v.to_le_bytes());
}
pub(crate) fn g16(e: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([e[off], e[off + 1]])
}
pub(crate) fn g32(e: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(e[off..off + 4].try_into().unwrap())
}
pub(crate) fn put_seg64(e: &mut [u8], off: usize, addr: u64, len: u32) {
    p64(e, off, addr);
    p32(e, off + 8, len);
}
fn put_seg32(e: &mut [u8], off: usize, addr: u32, len: u32) {
    p32(e, off, addr);
    p32(e, off + 4, len);
}

pub(crate) fn entry_command(kind: u8, handle: u32, loop_id: u8, lun: u16, opcode: u8) -> Entry {
    let mut e = [0u8; ENTRY_SIZE];
    e[0] = kind;
    e[1] = 1;
    p32(&mut e, 4, handle);
    e[9] = loop_id;
    p16(&mut e, 10, lun);
    e[20] = opcode;
    e
}

impl Rig {
    /// Both rings at fixed places, built with INIT_FIRMWARE.
    fn rings_at(&self, rqlen: u16, rslen: u16, rq: u64, rs: u64, options: u16, xopts: u8) {
        self.rqlen.set(rqlen);
        self.rslen.set(rslen);
        self.rq_next.set(0);
        self.rs_next.set(0);
        if rs + u64::from(rslen) * 64 <= crate::dev::isp1040::tests::MEM as u64 {
            self.mem.put(rs as u32, &vec![0u8; usize::from(rslen) * ENTRY_SIZE]);
        }
        let mut icb = icb_build(rqlen, rslen, rq, rs, options);
        icb[64] = xopts;
        self.mem.put(ICB_AT, &icb);
        assert_eq!(self.init_firmware(u64::from(ICB_AT)), mbox::COMPLETE);
        self.ack();
    }

    fn rings(&self, rqlen: u16, rslen: u16) {
        self.rings_at(rqlen, rslen, u64::from(RQST_AT), u64::from(RESP_AT), 0, 0);
    }

    fn put_request(&self, index: u16, e: &Entry) {
        self.mem.put(RQST_AT + u32::from(index) * 64, e);
    }

    /// Writes the entry at the producer and publishes it.
    fn publish(&self, e: &Entry) {
        self.put_request(self.rq_next.get(), e);
        self.rq_next.set((self.rq_next.get() + 1) % self.rqlen.get());
        self.wr(reg::REQUEST_QUEUE, self.rq_next.get());
    }

    fn response(&self, index: u16) -> Entry {
        self.mem.get(RESP_AT + u32::from(index) * 64, ENTRY_SIZE).try_into().unwrap()
    }

    /// Linux's ring handler order: consume, write response-out, clear.
    fn consume(&self, n: u16) {
        self.rs_next.set((self.rs_next.get() + n) % self.rslen.get());
        self.wr(reg::RESPONSE_QUEUE, self.rs_next.get());
        self.wr(reg::HCCR, HCCR_CLEAR_RISC);
    }

    /// One TEST UNIT READY through the whole ring round trip.
    fn tur(&self, handle: u32) {
        self.publish(&entry_command(ENTRY_CMD64, handle, 1, 0, 0x00));
        assert_eq!(self.rd(reg::ISR) & ISR_RISC, ISR_RISC);
        assert_eq!(self.rd(reg::SEMAPHORE), 0);
        let e = self.response(self.rs_next.get());
        assert_eq!(e[0], ENTRY_STATUS);
        assert_eq!(g32(&e, 4), handle);
        assert_eq!(g16(&e, 10), COMPLETION_PORT_UNAVAILABLE);
        self.consume(1);
    }

    /// The entry at the response consumer, then consumed.
    fn assert_rejected(&self, kind: u8, handle: u32, flag: u8) {
        let e = self.response(self.rs_next.get());
        assert_eq!((e[0], e[3], g32(&e, 4)), (kind, flag, handle));
        self.consume(1);
    }
}

/// A 64-bit command comes back as a whole 64-byte status entry: every byte of
/// the slot is written, so Linux's signature in bytes 60..63 is overwritten.
/// No disk is on the loop, so the completion is "port unavailable".
#[test]
fn command_round_trip() {
    let r = Rig::bare();
    r.start();
    r.rings(8, 8);
    r.mem.put(RESP_AT, &[0xde; ENTRY_SIZE]);

    // A READ(10) whose every CDB byte differs, and a byte count.
    let mut e = entry_command(ENTRY_CMD64, 0xcafe0001, 5, 2, 0x28);
    p16(&mut e, 12, 0x0028);
    for i in 1..16 {
        e[20 + i] = 0xa0 + i as u8;
    }
    p32(&mut e, 36, 0x12345);
    p16(&mut e, 18, 1);
    put_seg64(&mut e, 40, 0x0210_0000, 0x12345);
    r.publish(&e);

    assert_eq!(r.rd(reg::RESPONSE_QUEUE), 1);
    assert_eq!(r.rd(reg::REQUEST_QUEUE), 1);

    let e = r.response(0);
    assert_eq!((e[0], e[1], e[3]), (ENTRY_STATUS, 1, 0));
    assert_eq!(g32(&e, 4), 0xcafe0001);
    assert_eq!(g16(&e, 8), 0);
    assert_eq!(g16(&e, 10), COMPLETION_PORT_UNAVAILABLE);
    assert!(e[12..].iter().all(|&b| b == 0));
}

/// 32-bit commands are equivalent, and segments past the first entry come
/// from continuations: the entry count steps over them, and none is
/// dispatched as a command of its own.
#[test]
fn continuations() {
    let r = Rig::bare();
    r.start();
    r.rings(8, 8);

    // A 32-bit command alone.
    let mut e = entry_command(ENTRY_CMD32, 0x32, 3, 0, 0x12);
    p16(&mut e, 18, 1);
    put_seg32(&mut e, 40, 0x0210_0000, 36);
    r.publish(&e);
    assert_eq!(g32(&r.response(0), 4), 0x32);
    r.consume(1);

    // 64-bit: two inline segments, two more in a 0x0a continuation.
    let mut e = entry_command(ENTRY_CMD64, 0x64, 3, 0, 0x2a);
    e[1] = 2;
    p16(&mut e, 18, 4);
    put_seg64(&mut e, 40, 0x0210_0000, 0x100);
    put_seg64(&mut e, 52, 0x0220_0000, 0x200);
    let mut c = [0u8; ENTRY_SIZE];
    c[0] = ENTRY_CONT64;
    c[1] = 1;
    put_seg64(&mut c, 4, 0x0230_0000, 0x300);
    put_seg64(&mut c, 16, 0x0240_0000, 0x400);
    r.put_request(1, &e);
    r.put_request(2, &c);
    r.rq_next.set(3);
    r.wr(reg::REQUEST_QUEUE, 3);
    assert_eq!(r.rd(reg::REQUEST_QUEUE), 3);
    assert_eq!(r.rd(reg::RESPONSE_QUEUE), 2);
    assert_eq!(g32(&r.response(1), 4), 0x64);
    r.consume(1);

    // 32-bit: three inline, one in a 0x02 continuation from byte 8.
    let mut e = entry_command(ENTRY_CMD32, 0x320, 3, 0, 0x28);
    e[1] = 2;
    p16(&mut e, 18, 4);
    put_seg32(&mut e, 40, 0x0210_0000, 1);
    put_seg32(&mut e, 48, 0x0220_0000, 2);
    put_seg32(&mut e, 56, 0x0230_0000, 4);
    let mut c = [0u8; ENTRY_SIZE];
    c[0] = ENTRY_CONT32;
    c[1] = 1;
    put_seg32(&mut c, 8, 0x0240_0000, 8);
    r.put_request(3, &e);
    r.put_request(4, &c);
    r.rq_next.set(5);

    // Only the command is published: it waits for its continuation.
    r.wr(reg::REQUEST_QUEUE, 4);
    assert_eq!(r.rd(reg::REQUEST_QUEUE), 3);
    assert_eq!(r.rd(reg::RESPONSE_QUEUE), 2);
    r.wr(reg::REQUEST_QUEUE, 5);
    assert_eq!(r.rd(reg::REQUEST_QUEUE), 5);
    assert_eq!(r.rd(reg::RESPONSE_QUEUE), 3);
}

/// OpenBSD's segment-list command reads its list from host memory.
#[test]
fn segment_list() {
    let r = Rig::bare();
    r.start();
    r.rings(8, 8);

    let mut list = [0u8; 36];
    put_seg64(&mut list, 0, 0x0210_0000, 0x1000);
    put_seg64(&mut list, 12, 0x0220_0000, 0x2000);
    put_seg64(&mut list, 24, 0x0230_0000, 0x0800);
    r.mem.put(0x6000, &list);

    let mut e = entry_command(ENTRY_CMD_LIST, 0x15, 4, 1, 0x28);
    p16(&mut e, 18, 3);
    p16(&mut e, 40, 1);
    p64(&mut e, 46, 0x6000);
    r.publish(&e);
    let e = r.response(0);
    assert_eq!(e[0], ENTRY_STATUS);
    assert_eq!(g32(&e, 4), 0x15);
}

/// Approximation: a marker is consumed and answers nothing.
#[test]
fn marker() {
    let r = Rig::bare();
    r.start();
    r.rings(8, 8);
    let mut e = [0u8; ENTRY_SIZE];
    e[0] = ENTRY_MARKER;
    e[1] = 1;
    e[10] = 2;
    r.publish(&e);
    assert_eq!(r.rd(reg::REQUEST_QUEUE), 1);
    assert_eq!(r.rd(reg::RESPONSE_QUEUE), 0);
    assert_eq!(r.rd(reg::ISR), 0);
}

/// The producer wraps at the ring length; entries either side are taken.
#[test]
fn request_wrap() {
    let r = Rig::bare();
    r.start();
    r.rings(4, 8);
    for i in 0..6 {
        r.tur(0x100 + i);
        assert_eq!(r.rd(reg::REQUEST_QUEUE), r.rq_next.get());
    }
    assert_eq!(r.rq_next.get(), 2);
}

/// A completion posted on the ring thread rises while another thread consumes
/// it and clears the interrupt. The ring thread drives in the hold that
/// posted the completion, so the fall waits for the rise to be delivered.
#[test]
fn a_completion_posted_by_the_ring_thread_is_delivered_before_a_later_fall() {
    use crate::irq::StallingProbe;
    let (stall, line) = StallingProbe::line();
    let r = Rig::on_line(Isp2200::new(), line);
    r.start();
    r.rings(4, 8);
    r.wr(reg::ICR, ICR_ENABLE);
    r.put_request(0, &entry_command(ENTRY_CMD64, 0x100, 1, 0, 0x00));
    let (started, go) = stall.arm();
    std::thread::scope(|scope| {
        scope.spawn(|| r.isp.write(reg::REQUEST_QUEUE, 1, 2));
        started.recv_timeout(std::time::Duration::from_secs(5)).expect("the completion's rise is in the parent");
        let c = scope.spawn(|| {
            r.isp.write(reg::RESPONSE_QUEUE, 1, 2);
            r.isp.write(reg::HCCR, u32::from(HCCR_CLEAR_RISC), 2);
        });
        // As the ISP1040's equivalent test: the sleep only backs a negative
        // assertion after the handshake above, so it can pass falsely but
        // never fail spuriously.
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert!(!c.is_finished(), "the fall waits for the rise to be delivered");
        go.send(()).unwrap();
    });
    r.isp.wait_idle();
    assert_eq!(stall.probe.writes(), [false, true, false]);
    assert!(!r.isp.interrupting());
}

/// Approximation: response-in advances after each whole entry; with the ring
/// full, later completions wait and post in order once the host frees room.
#[test]
fn response_ring_full() {
    let r = Rig::bare();
    r.start();
    r.rings(8, 4);
    for i in 0..5u32 {
        r.publish(&entry_command(ENTRY_CMD64, 0x500 + i, 1, 0, 0));
        if i < 3 {
            assert_eq!(r.rd(reg::RESPONSE_QUEUE), i as u16 + 1);
        }
        // The host clears the latch without consuming anything.
        r.wr(reg::HCCR, HCCR_CLEAR_RISC);
    }
    // Three of four slots hold entries; two completions wait.
    assert_eq!(r.rd(reg::RESPONSE_QUEUE), 3);
    assert_eq!(r.rd(reg::ISR), 0);

    r.wr(reg::RESPONSE_QUEUE, 3);
    r.rs_next.set(3);
    assert_eq!(r.rd(reg::ISR), ISR_RISC);
    assert_eq!(r.rd(reg::RESPONSE_QUEUE), 1);
    assert_eq!(g32(&r.response(3), 4), 0x503);
    assert_eq!(g32(&r.response(0), 4), 0x504);
}

/// Approximation: an unknown type comes back as itself, flagged.
#[test]
fn unknown_entry_type() {
    let r = Rig::bare();
    r.start();
    r.rings(8, 8);
    r.publish(&entry_command(0x33, 0xabcd0033, 1, 0, 0));
    let e = r.response(0);
    assert_eq!((e[0], e[3], g32(&e, 4)), (0x33, ENTRY_FLAG_BAD_TYPE, 0xabcd0033));
    assert_eq!(r.rd(reg::REQUEST_QUEUE), 1);
}

/// Approximation: which entry-status flag a malformed request carries is the
/// model's choice. An entry count of 0 or one that does not fit the ring sets
/// bit 4 (invalid count); more segments than the command's entries hold sets
/// bit 3 (invalid parameter); a continuation with no command, or one of the
/// other width, sets bit 5 (invalid order). None reaches the service, and
/// each takes the slots its count claims.
#[test]
fn malformed_entries() {
    let r = Rig::bare();
    r.start();
    r.rings(8, 8);

    let mut e = entry_command(ENTRY_CMD64, 0xbad00000, 1, 0, 0);
    e[1] = 0;
    r.publish(&e);
    assert_eq!(r.rd(reg::REQUEST_QUEUE), 1);
    r.assert_rejected(ENTRY_CMD64, 0xbad00000, ENTRY_FLAG_BAD_COUNT);

    let mut e = entry_command(ENTRY_CMD32, 0xbad00008, 1, 0, 0);
    e[1] = 8;
    r.publish(&e);
    assert_eq!(r.rd(reg::REQUEST_QUEUE), 2);
    r.assert_rejected(ENTRY_CMD32, 0xbad00008, ENTRY_FLAG_BAD_COUNT);

    // A lone 0x19 holds two segments, not three.
    let mut e = entry_command(ENTRY_CMD64, 0xbad00003, 1, 0, 0);
    p16(&mut e, 18, 3);
    r.publish(&e);
    assert_eq!(r.rd(reg::REQUEST_QUEUE), 3);
    r.assert_rejected(ENTRY_CMD64, 0xbad00003, ENTRY_FLAG_BAD_PARAM);

    // Continuations with no command before them.
    let mut c = [0u8; ENTRY_SIZE];
    c[0] = ENTRY_CONT64;
    c[1] = 1;
    p32(&mut c, 4, 0xc0a);
    r.publish(&c);
    assert_eq!(r.rd(reg::REQUEST_QUEUE), 4);
    r.assert_rejected(ENTRY_CONT64, 0xc0a, ENTRY_FLAG_BAD_ORDER);

    c[0] = ENTRY_CONT32;
    p32(&mut c, 4, 0xc02);
    r.publish(&c);
    assert_eq!(r.rd(reg::REQUEST_QUEUE), 5);
    r.assert_rejected(ENTRY_CONT32, 0xc02, ENTRY_FLAG_BAD_ORDER);

    // A 64-bit command followed by a 32-bit continuation: both slots go.
    let mut e = entry_command(ENTRY_CMD64, 0xbad00064, 1, 0, 0);
    e[1] = 2;
    p16(&mut e, 18, 3);
    put_seg64(&mut e, 40, 0x0210_0000, 0x100);
    put_seg64(&mut e, 52, 0x0220_0000, 0x100);
    let mut c = [0u8; ENTRY_SIZE];
    c[0] = ENTRY_CONT32;
    c[1] = 1;
    put_seg32(&mut c, 8, 0x0230_0000, 0x100);
    r.put_request(5, &e);
    r.put_request(6, &c);
    r.rq_next.set(7);
    r.wr(reg::REQUEST_QUEUE, 7);
    assert_eq!(r.rd(reg::REQUEST_QUEUE), 7);
    r.assert_rejected(ENTRY_CMD64, 0xbad00064, ENTRY_FLAG_BAD_ORDER);
    assert_eq!(r.rd(reg::RESPONSE_QUEUE), 6);
}

/// Approximation: a refused read after the entry's own header -- the segment
/// list of a 0x15 command, or a continuation -- returns the command with
/// entry-status bit 6 (request-queue DMA error).
#[test]
fn descriptor_dma_errors() {
    let r = Rig::bare();
    r.start();
    r.rings(8, 8);
    let mut e = entry_command(ENTRY_CMD_LIST, 0xd15, 4, 0, 0x28);
    p16(&mut e, 18, 3);
    p16(&mut e, 40, 1);
    p64(&mut e, 46, 1 << 40);
    r.publish(&e);
    assert_eq!(r.rd(reg::REQUEST_QUEUE), 1);
    r.assert_rejected(ENTRY_CMD_LIST, 0xd15, ENTRY_FLAG_DMA_ERROR);

    // The ring's last backed slot is slot 0; its continuation is not backed.
    let last = (crate::dev::isp1040::tests::MEM - ENTRY_SIZE) as u64;
    r.rings_at(4, 8, last, u64::from(RESP_AT), 0, 0);
    let mut e = entry_command(ENTRY_CMD64, 0xd0a, 1, 0, 0x28);
    e[1] = 2;
    p16(&mut e, 18, 3);
    r.mem.put(last as u32, &e);
    r.rq_next.set(2);
    r.wr(reg::REQUEST_QUEUE, 2);
    assert_eq!(r.rd(reg::REQUEST_QUEUE), 2);
    r.assert_rejected(ENTRY_CMD64, 0xd0a, ENTRY_FLAG_DMA_ERROR);
}

impl Rig {
    fn assert_event(&self, code: u16) {
        assert_eq!(self.rd(reg::SEMAPHORE), SEM_LOCK);
        assert_eq!(self.rd(reg::ISR) & ISR_RISC, ISR_RISC);
        assert_eq!(self.rd(mb(0)), code);
        for i in 1..4 {
            assert_eq!(self.rd(mb(i)), 0);
        }
        self.ack();
    }
}

/// Approximation: a refused request-ring read raises 0x8003 and a refused
/// response write 0x8004, both through the semaphore path.
#[test]
fn ring_dma_errors() {
    let r = Rig::bare();
    r.start();
    let beyond = 1u64 << 40;
    r.rings_at(8, 8, beyond, u64::from(RESP_AT), 0, 0);
    r.wr(reg::REQUEST_QUEUE, 1);
    r.assert_event(event::REQUEST_DMA_ERROR);
    assert_eq!(r.rd(reg::REQUEST_QUEUE), 0);

    r.rings_at(8, 8, u64::from(RQST_AT), beyond, 0, 0);
    r.publish(&entry_command(ENTRY_CMD64, 0x8004, 1, 0, 0));
    r.assert_event(event::RESPONSE_DMA_ERROR);
    assert_eq!(r.rd(reg::RESPONSE_QUEUE), 0);
}

/// Approximation: a posted result stays readable while the host writes a
/// command's mailboxes behind it. Linux writes all 24 mailboxes and rings
/// without draining a result first; the IRIX 6.5.30 driver wrote only
/// mailbox 0 over a posted event on the model. In both, mailboxes 0..3 read
/// the event until its release, and the command then completes on what the
/// host wrote. A model that lets the write through to the read side fails
/// both halves.
#[test]
fn writes_behind_posted_result() {
    let r = Rig::bare();
    r.start();
    r.rings_at(8, 8, 1 << 40, u64::from(RESP_AT), 0, 0);

    // Linux's order: every mailbox, ascending, then the doorbell.
    r.wr(reg::REQUEST_QUEUE, 1);
    assert_eq!(r.rd(mb(0)), event::REQUEST_DMA_ERROR);
    let mut input = [0u16; MAILBOX_COUNT];
    input[0] = mbox::REGISTER_TEST;
    for (i, v) in input.iter_mut().enumerate().skip(1) {
        *v = 0x5a00 | i as u16;
    }
    r.mbox(&input);
    for i in 0..4 {
        assert_eq!(r.rd(mb(i)), if i == 0 { event::REQUEST_DMA_ERROR } else { 0 });
    }
    assert_eq!(r.rd(mb(8)), 0);
    r.ack();
    assert_eq!(r.rd(reg::SEMAPHORE), SEM_LOCK);
    assert_eq!(r.rd(mb(0)), mbox::COMPLETE);
    for i in 1..8 {
        assert_eq!(r.rd(mb(i)), input[i]);
    }
    r.ack();

    // IRIX's case: mailbox 0 alone, then the doorbell.
    r.wr(reg::REQUEST_QUEUE, 2);
    assert_eq!(r.rd(mb(0)), event::REQUEST_DMA_ERROR);
    r.wr(mb(0), mbox::NOP);
    assert_eq!(r.rd(mb(0)), event::REQUEST_DMA_ERROR);
    r.wr(reg::HCCR, HCCR_SET_HOST);
    assert_eq!(r.rd(mb(0)), event::REQUEST_DMA_ERROR);
    r.wr(reg::HCCR, HCCR_CLEAR_RISC);
    assert_eq!(r.rd(mb(0)), event::REQUEST_DMA_ERROR);
    r.wr(reg::SEMAPHORE, 0);
    assert_eq!(r.rd(reg::SEMAPHORE), SEM_LOCK);
    assert_eq!(r.rd(mb(0)), mbox::COMPLETE);
    r.ack();

    // With the lock free and nothing posted, a write reads back.
    assert_eq!(r.rd(reg::SEMAPHORE), 0);
    r.wr(mb(1), 0x1234);
    assert_eq!(r.rd(mb(1)), 0x1234);
}

/// Approximation: events are never dropped, and a refused request read adds
/// no event while an identical one still waits to be posted: that one already
/// reports it. One 0x8003 can wait behind a posted one.
#[test]
fn request_dma_error_waits() {
    let r = Rig::bare();
    r.start();
    r.rings_at(8, 8, 1 << 40, u64::from(RESP_AT), 0, 0);

    // Posted, then one waiting, then retries that add nothing.
    for n in 1..=4 {
        r.wr(reg::REQUEST_QUEUE, n);
    }
    r.assert_event(event::REQUEST_DMA_ERROR);
    r.assert_event(event::REQUEST_DMA_ERROR);
    assert_eq!(r.rd(reg::SEMAPHORE), 0);
    assert_eq!(r.rd(reg::ISR), 0);
    assert_eq!(r.rd(reg::REQUEST_QUEUE), 0);

    // Released, the next producer write reports the ring again.
    r.wr(reg::REQUEST_QUEUE, 5);
    r.assert_event(event::REQUEST_DMA_ERROR);
}

/// With bus mastering off, nothing that needs DMA moves, and it says so.
#[test]
fn bus_master_disabled() {
    let r = Rig::bare();
    r.start();
    r.rings(8, 8);
    r.isp.state().config().write(1, 0x3);

    assert_eq!(r.call_ack(mbox::NOP, 0, 0, 0), mbox::COMPLETE);
    assert_eq!(r.load_ram(mbox::LOAD_RAM, 0x2000, 0x6000, 4, 0), mbox::HOST_INTERFACE_ERROR);
    r.ack();
    r.wr(reg::REQUEST_QUEUE, 1);
    r.assert_event(event::REQUEST_DMA_ERROR);
    assert_eq!(r.rd(reg::REQUEST_QUEUE), 0);
}

/// Three commands through, so both request pointers read 3.
fn three_commands(r: &Rig) {
    r.start();
    r.rings(8, 8);
    r.tur(1);
    r.tur(2);
    r.tur(3);
    assert_eq!(r.rd(reg::REQUEST_QUEUE), 3);
}

/// Approximation: Linux writes all 24 mailboxes on every command, mailbox 4
/// included; inside the command window that is an input, so the ring stays
/// put and nothing is resubmitted.
#[test]
fn window_linux_order() {
    let r = Rig::bare();
    three_commands(&r);
    let mut input = [0u16; MAILBOX_COUNT];
    assert_eq!(r.mbox(&input), mbox::COMPLETE);
    r.ack();
    assert_eq!(r.rd(reg::REQUEST_QUEUE), 3);

    r.tur(4);

    input[4] = 0x1234;
    input[5] = 0x5678;
    assert_eq!(r.mbox(&input), mbox::COMPLETE);
    r.ack();
    assert_eq!(r.rd(reg::REQUEST_QUEUE), 4);
    assert_eq!(r.rd(reg::RESPONSE_QUEUE), 4);
    assert_eq!(r.response(4)[0], 0);
}

/// Approximation: OpenBSD's INIT_FIRMWARE writes m4 = m5 = 0 as inputs, with
/// the queues already live; the command resets the rings and a new entry is
/// taken once.
#[test]
fn window_openbsd_init() {
    let r = Rig::bare();
    three_commands(&r);
    let icb = u64::from(ICB_AT);
    for (i, v) in [(0, mbox::INIT_FIRMWARE), (2, (icb >> 16) as u16), (3, icb as u16), (4, 0), (5, 0), (6, 0), (7, 0)] {
        r.wr(mb(i), v);
    }
    r.wr(reg::HCCR, HCCR_SET_HOST);
    assert_eq!(r.rd(mb(0)), mbox::COMPLETE);
    r.ack();
    assert_eq!(r.rd(reg::REQUEST_QUEUE), 0);
    assert_eq!(r.rd(reg::RESPONSE_QUEUE), 0);

    r.rq_next.set(0);
    r.rs_next.set(0);
    r.mem.put(RESP_AT, &[0u8; ENTRY_SIZE]);
    r.publish(&entry_command(ENTRY_CMD64, 0x38, 1, 0, 0));
    assert_eq!(r.rd(reg::REQUEST_QUEUE), 1);
    assert_eq!(g32(&r.response(0), 4), 0x38);
}

/// OpenBSD's soft-reset sequence writes all four pointers as 0 after the
/// release; with no queue they are mailbox storage.
#[test]
fn window_openbsd_reset() {
    let r = Rig::bare();
    three_commands(&r);
    r.soft_reset();
    r.wr(reg::HCCR, HCCR_RESET);
    r.wr(reg::SEMAPHORE, 0);
    r.wr(reg::HCCR, HCCR_PARITY4);
    r.wr(reg::HCCR, HCCR_RELEASE);
    for off in [reg::REQUEST_QUEUE, reg::REQUEST_QUEUE, reg::RESPONSE_QUEUE, reg::RESPONSE_QUEUE] {
        r.wr(off, 0);
    }
    assert_eq!(r.rd(reg::ISR), 0);

    r.start();
    r.rings(8, 8);
    r.tur(0x39);
}

/// Approximation, the window's limits: an open window turns a producer write
/// into storage until the doorbell; a reset closes it too, and takes the
/// queues and firmware with it.
#[test]
fn window_boundaries() {
    let r = Rig::bare();
    three_commands(&r);
    r.put_request(3, &entry_command(ENTRY_CMD64, 4, 1, 0, 0));

    // (a) Mailbox 0, no doorbell: the producer write publishes nothing.
    r.wr(mb(0), mbox::NOP);
    r.wr(reg::REQUEST_QUEUE, 4);
    assert_eq!(r.rd(reg::REQUEST_QUEUE), 3);

    // (b) The doorbell closes the window; the same write now publishes.
    r.wr(reg::HCCR, HCCR_SET_HOST);
    assert_eq!(r.rd(mb(0)), mbox::COMPLETE);
    r.ack();
    r.wr(reg::REQUEST_QUEUE, 4);
    assert_eq!(r.rd(reg::REQUEST_QUEUE), 4);
    r.consume(1);

    // (c) A reset closes it as well, and empties the adapter.
    r.wr(mb(0), mbox::NOP);
    r.wr(reg::HCCR, HCCR_RESET);
    r.wr(reg::HCCR, HCCR_RELEASE);
    r.wr(reg::REQUEST_QUEUE, 4);
    assert_eq!(r.rd(reg::REQUEST_QUEUE), 4);
    assert_eq!(r.call_ack(mbox::INIT_FIRMWARE, 0, 0, 0), mbox::INVALID);

    r.start();
    r.rings(8, 8);
    r.put_request(0, &entry_command(ENTRY_CMD64, 5, 1, 0, 0));
    r.rq_next.set(1);
    r.wr(reg::REQUEST_QUEUE, 1);
    assert_eq!(g32(&r.response(0), 4), 5);
}

impl Rig {
    /// Fills a two-slot response ring (one usable slot) and leaves one more
    /// completion held behind it, latch clear.
    fn fill_response_ring(&self) {
        self.start();
        self.rings(8, 2);
        self.publish(&entry_command(ENTRY_CMD64, 0xa1, 1, 0, 0));
        self.wr(reg::HCCR, HCCR_CLEAR_RISC);
        self.publish(&entry_command(ENTRY_CMD64, 0xa2, 1, 0, 0));
        assert_eq!(self.rd(reg::RESPONSE_QUEUE), 1);
        assert_eq!(self.rd(reg::ISR), 0);
        assert_eq!(self.response(1)[0], 0);
    }
}

/// Approximation: a response-out write inside the window is an input and
/// leaves the ring where it was; after the doorbell it moves it.
#[test]
fn window_response_out() {
    let r = Rig::bare();
    r.fill_response_ring();
    r.wr(mb(0), mbox::NOP);
    r.wr(reg::RESPONSE_QUEUE, 1);
    r.wr(reg::HCCR, HCCR_SET_HOST);
    assert_eq!(r.rd(mb(0)), mbox::COMPLETE);
    r.ack();
    assert_eq!(r.response(1)[0], 0);
    assert_eq!(r.rd(reg::RESPONSE_QUEUE), 1);

    r.wr(reg::RESPONSE_QUEUE, 1);
    assert_eq!(g32(&r.response(1), 4), 0xa2);
    assert_eq!(r.rd(reg::RESPONSE_QUEUE), 0);
    assert_eq!(r.rd(reg::ISR), ISR_RISC);
}

/// Approximation: nothing is written while the latch is set. A response-out
/// write alone posts nothing; the clear does, and a batch of held completions
/// posts in order under one latch.
#[test]
fn posting_waits_for_latch() {
    let r = Rig::bare();
    r.start();
    r.rings(8, 8);

    r.publish(&entry_command(ENTRY_CMD64, 0xb1, 1, 0, 0));
    assert_eq!(r.rd(reg::ISR), ISR_RISC);
    r.publish(&entry_command(ENTRY_CMD64, 0xb2, 1, 0, 0));
    assert_eq!(r.rd(reg::RESPONSE_QUEUE), 1);
    assert_eq!(r.response(1)[0], 0);

    r.wr(reg::RESPONSE_QUEUE, 1);
    assert_eq!(r.rd(reg::RESPONSE_QUEUE), 1);
    assert_eq!(r.response(1)[0], 0);

    r.wr(reg::HCCR, HCCR_CLEAR_RISC);
    assert_eq!(r.rd(reg::ISR), ISR_RISC);
    assert_eq!(r.rd(reg::RESPONSE_QUEUE), 2);
    assert_eq!(g32(&r.response(1), 4), 0xb2);

    // Three more held behind the standing latch post as one batch.
    for i in 0..3 {
        r.publish(&entry_command(ENTRY_CMD64, 0xc0 + i, 1, 0, 0));
    }
    assert_eq!(r.rd(reg::RESPONSE_QUEUE), 2);
    r.wr(reg::RESPONSE_QUEUE, 2);
    r.wr(reg::HCCR, HCCR_CLEAR_RISC);
    assert_eq!(r.rd(reg::RESPONSE_QUEUE), 5);
    assert_eq!(r.rd(reg::ISR), ISR_RISC);
    for i in 0..3 {
        assert_eq!(g32(&r.response(2 + i), 4), 0xc0 + u32::from(i));
    }
}

/// Approximation: nothing is written while a mailbox result holds the
/// semaphore; a queued mailbox result posts first and alone; and nothing
/// posts while the RISC is paused.
#[test]
fn posting_waits_for_semaphore() {
    let r = Rig::bare();
    r.start();
    r.rings(8, 8);

    // A result holds the semaphore; its latch is cleared, the lock kept.
    r.call(mbox::NOP, 0, 0, 0);
    r.wr(reg::HCCR, HCCR_CLEAR_RISC);
    r.publish(&entry_command(ENTRY_CMD64, 0xd1, 1, 0, 0));
    assert_eq!(r.response(0)[0], 0);
    assert_eq!(r.rd(reg::ISR), 0);
    r.wr(reg::SEMAPHORE, 0);
    assert_eq!(r.rd(reg::ISR), ISR_RISC);
    assert_eq!(r.rd(reg::RESPONSE_QUEUE), 1);
    assert_eq!(g32(&r.response(0), 4), 0xd1);

    // With the latch still set: a mailbox result and a completion queue.
    r.call(mbox::NOP, 0, 0, 0);
    assert_eq!(r.rd(reg::SEMAPHORE), 0);
    r.publish(&entry_command(ENTRY_CMD64, 0xd2, 1, 0, 0));
    r.consume(1);
    // The mailbox result went first, alone.
    assert_eq!(r.rd(reg::SEMAPHORE), SEM_LOCK);
    assert_eq!(r.rd(mb(0)), mbox::COMPLETE);
    assert_eq!(r.response(1)[0], 0);
    r.ack();
    assert_eq!(r.rd(reg::ISR), ISR_RISC);
    assert_eq!(r.rd(reg::RESPONSE_QUEUE), 2);
    assert_eq!(g32(&r.response(1), 4), 0xd2);

    // Held by the latch, then paused: the clear posts nothing.
    r.publish(&entry_command(ENTRY_CMD64, 0xd3, 1, 0, 0));
    r.wr(reg::HCCR, HCCR_PAUSE);
    r.consume(1);
    assert_eq!(r.rd(reg::ISR), 0);
    assert_eq!(r.response(2)[0], 0);
    r.wr(reg::HCCR, HCCR_RELEASE);
    assert_eq!(r.rd(reg::ISR), ISR_RISC);
    assert_eq!(g32(&r.response(2), 4), 0xd3);
}

/// Approximation, not a hardware fact: OpenBSD can land a pointer write
/// between a command's mailbox 0 write and its doorbell. The model takes it
/// as mailbox storage. A producer write is lost until the next one.
#[test]
fn window_openbsd_interleave() {
    let r = Rig::bare();
    three_commands(&r);
    r.put_request(3, &entry_command(ENTRY_CMD64, 0x47a, 1, 0, 0));
    r.put_request(4, &entry_command(ENTRY_CMD64, 0x47b, 1, 0, 0));

    r.wr(mb(0), mbox::NOP);
    r.wr(reg::REQUEST_QUEUE, 4);
    r.wr(reg::HCCR, HCCR_SET_HOST);
    assert_eq!(r.rd(mb(0)), mbox::COMPLETE);
    r.ack();
    assert_eq!(r.rd(reg::REQUEST_QUEUE), 3, "the pointer write took nothing");
    r.wr(reg::REQUEST_QUEUE, 5);
    r.consume(3);
    assert_eq!(r.rd(reg::RESPONSE_QUEUE), 5);
    assert_eq!((g32(&r.response(3), 4), g32(&r.response(4), 4)), (0x47a, 0x47b), "each taken once");
}

/// A response-out write is lost, and the held completion stays held. The
/// repeat without the mailbox 0 write shows the difference.
#[test]
fn window_openbsd_interleave_response() {
    let r = Rig::bare();
    r.fill_response_ring();
    r.wr(mb(0), mbox::NOP);
    r.wr(reg::RESPONSE_QUEUE, 1);
    r.wr(reg::HCCR, HCCR_SET_HOST);
    r.wr(reg::HCCR, HCCR_CLEAR_RISC);
    r.wr(reg::SEMAPHORE, 0);
    assert_eq!(r.response(1)[0], 0);

    // The same, without the mailbox 0 write.
    r.wr(reg::RESPONSE_QUEUE, 1);
    r.wr(reg::HCCR, HCCR_SET_HOST);
    r.wr(reg::HCCR, HCCR_CLEAR_RISC);
    r.wr(reg::SEMAPHORE, 0);
    assert_eq!(g32(&r.response(1), 4), 0xa2);
}

mod irix_tests;
mod loop_tests;
mod nvram_tests;
mod scsi_tests;
mod state_tests;
