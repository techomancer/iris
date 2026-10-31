//! Register-contract and ring tests, for the ISP1040's register
//! and ring contract. Host memory is a flat byte array behind a fake `PciDma`; the disk
//! is a small raw image.
//!
//! Ported from tests/qtest/qlogic-isp-test.c and tests/qtest/sn0-isp12160-test.c
//! in jamesbraid/qemu-sgi-origami.

use super::*;
use crate::irq::LevelProbe;
use crate::pci::PciDmaStatus;
use crate::traits::{Device, Saveable};
use std::sync::Mutex;

const MEM: usize = 1 << 20;

struct FakeMem(Mutex<Vec<u8>>);

impl FakeMem {
    fn new() -> Arc<Self> {
        Arc::new(FakeMem(Mutex::new(vec![0; MEM])))
    }
    fn put(&self, addr: u32, data: &[u8]) {
        self.0.lock().unwrap()[addr as usize..addr as usize + data.len()].copy_from_slice(data);
    }
    fn get(&self, addr: u32, n: usize) -> Vec<u8> {
        self.0.lock().unwrap()[addr as usize..addr as usize + n].to_vec()
    }
}

impl PciDma for FakeMem {
    fn dma_read(&self, addr: u64, buf: &mut [u8]) -> PciDmaStatus {
        let m = self.0.lock().unwrap();
        let a = addr as usize;
        if a.saturating_add(buf.len()) > m.len() { return PciDmaStatus::MasterAbort; }
        buf.copy_from_slice(&m[a..a + buf.len()]);
        PciDmaStatus::Ok
    }
    fn dma_write(&self, addr: u64, data: &[u8]) -> PciDmaStatus {
        let mut m = self.0.lock().unwrap();
        let a = addr as usize;
        if a.saturating_add(data.len()) > m.len() { return PciDmaStatus::MasterAbort; }
        m[a..a + data.len()].copy_from_slice(data);
        PciDmaStatus::Ok
    }
}

const RQST: u32 = 0x1000;
const RESP: u32 = 0x2000;
const BUF: u32 = 0x10000;

struct Rig {
    isp: Isp1040,
    mem: Arc<FakeMem>,
}

fn image(name: &str, blocks: usize) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("iris-isp1040-{}-{name}.img", std::process::id()));
    std::fs::write(&p, vec![0u8; blocks * BLOCK_SIZE]).expect("test image");
    p
}

impl Rig {
    fn on(isp: Isp1040) -> Rig {
        let mem = FakeMem::new();
        Device::start(&isp);
        isp.connect_dma(mem.clone());
        Rig { isp, mem }
    }

    /// A chip on fake memory, no disks.
    fn bare() -> Rig { Rig::on(Isp1040::new()) }

    /// A chip with a disk of `blocks` blocks at target `id`.
    fn with_disk(name: &str, id: u8, blocks: usize) -> Rig {
        let r = Rig::bare();
        r.isp.bus().attach_image(id, &image(name, blocks), false).expect("attach");
        r
    }

    fn r(&self, off: u32) -> u16 { self.isp.read(off, 2) as u16 }
    /// A register write. Commands run on the adapter's own thread, so a
    /// write that can start them waits for the ring to settle.
    fn w(&self, off: u32, v: u16) {
        self.isp.write(off, v as u32, 2);
        if off == reg::REQUEST_QUEUE || off == reg::RESPONSE_QUEUE {
            self.isp.wait_idle();
        }
    }
    fn mb(&self, n: u32) -> u32 { reg::MAILBOX_BASE + n * 2 }

    /// A mailbox command: load the registers, ring the host interrupt, read
    /// the status back, then acknowledge as a driver does.
    fn cmd(&self, regs: &[u16]) -> u16 {
        for (i, &v) in regs.iter().enumerate() {
            self.w(self.mb(i as u32), v);
        }
        self.w(reg::HOST_CMD, 0x5 << HOST_CMD_SHIFT);
        let status = self.r(self.mb(0));
        self.ack();
        status
    }

    fn ack(&self) {
        self.w(reg::HOST_CMD, 0x7 << HOST_CMD_SHIFT);
        self.w(reg::SEMAPHORE, 0);
    }

    fn init_queue(&self, cmd: u16, len: u16, addr: u32) -> u16 {
        self.cmd(&[cmd, len, (addr >> 16) as u16, addr as u16])
    }

    fn init_rings(&self, req_len: u16, resp_len: u16) {
        assert_eq!(self.init_queue(mbox::INIT_REQ_QUEUE, req_len, RQST), mbox::COMPLETE);
        assert_eq!(self.init_queue(mbox::INIT_RSP_QUEUE, resp_len, RESP), mbox::COMPLETE);
    }

    fn put_entry(&self, slot: u32, e: &[u8; ENTRY_SIZE]) {
        self.mem.put(RQST + slot * ENTRY_SIZE as u32, e);
    }

    fn status(&self, slot: u32) -> [u8; ENTRY_SIZE] {
        self.mem.get(RESP + slot * ENTRY_SIZE as u32, ENTRY_SIZE).try_into().unwrap()
    }

    fn doorbell(&self, n: u16) { self.w(reg::REQUEST_QUEUE, n); }
}

fn p16(e: &mut [u8], off: usize, v: u16) { e[off..off + 2].copy_from_slice(&v.to_le_bytes()); }
fn p32(e: &mut [u8], off: usize, v: u32) { e[off..off + 4].copy_from_slice(&v.to_le_bytes()); }
fn g16(e: &[u8], off: usize) -> u16 { u16::from_le_bytes([e[off], e[off + 1]]) }
fn g32(e: &[u8], off: usize) -> u32 { u32::from_le_bytes([e[off], e[off + 1], e[off + 2], e[off + 3]]) }

/// A type-1 request entry for `target`, with up to four inline segments.
fn request(handle: u32, target: u8, cdb: &[u8], segs: &[(u32, u32)]) -> [u8; ENTRY_SIZE] {
    let mut e = [0u8; ENTRY_SIZE];
    e[0] = ENTRY_TYPE_REQUEST;
    e[1] = 1;
    p32(&mut e, 4, handle);
    p16(&mut e, 8, (target as u16) << 8);
    p16(&mut e, 10, cdb.len() as u16);
    p16(&mut e, 18, segs.len() as u16);
    e[20..20 + cdb.len()].copy_from_slice(cdb);
    for (i, &(a, l)) in segs.iter().take(4).enumerate() {
        p32(&mut e, 32 + i * 8, a);
        p32(&mut e, 36 + i * 8, l);
    }
    e
}

fn cdb10(op: u8, lba: u32, blocks: u16) -> [u8; 10] {
    let mut c = [0u8; 10];
    c[0] = op;
    c[2..6].copy_from_slice(&lba.to_be_bytes());
    c[7..9].copy_from_slice(&blocks.to_be_bytes());
    c
}

// ---- identity and plain registers --------------------------------------

#[test]
fn identity_and_window() {
    let r = Rig::bare();
    let c = r.isp.config();
    assert_eq!(c.read(0), 0x1020_1077, "device and vendor id");
    assert_eq!(c.read(2) & 0xff, 0x05, "revision");
    assert_eq!(c.read(2) >> 16, 0x0100, "mass storage, SCSI");
    assert_eq!(c.interrupt_pin(), PCI_INTA);
    c.write(4, 0xffff_ffff);
    assert_eq!(c.read(4), 0xffff_ff01, "256 bytes of I/O space");
    c.write(5, 0xffff_ffff);
    assert_eq!(c.read(5), 0xffff_ff00, "256 bytes of memory space");
    // Both BARs reach the same sixteen-bit registers.
    for bar in 0..2 {
        assert_eq!(r.isp.bar_read(bar, reg::ID_LOW, 2), 0x1077);
        assert_eq!(r.isp.bar_read(bar, reg::ID_HIGH, 2), 0x1020);
    }
}

#[test]
fn bus_cfg0_reports_revision_plus_one() {
    let r = Rig::bare();
    assert_eq!(r.r(reg::BUS_CFG0), 0x0006);
    assert_ne!(r.r(reg::BUS_CFG0) & 0xf, 0x4, "must not select the broken-FIFO workaround");
}

#[test]
fn reset_signature_and_soft_reset() {
    let r = Rig::bare();
    assert_eq!(r.r(r.mb(1)), SIG1);
    assert_eq!(r.r(r.mb(2)), SIG2);
    assert_eq!(r.r(r.mb(3)), SIG3);
    assert_eq!(r.r(r.mb(4)), SIG4);
    r.w(reg::BUS_CFG1, CFG1_BURST_ENABLE);
    r.w(reg::SEMAPHORE, SEMAPHORE_LOCK);
    r.w(r.mb(1), 0xdead);
    r.w(r.mb(7), 0xbeef);
    r.w(reg::HOST_CMD, 0x2 << HOST_CMD_SHIFT);
    assert_eq!(r.r(r.mb(1)), 0xdead);
    r.w(reg::BUS_CTRL, CTRL_SOFT_RESET);
    assert_eq!(r.r(reg::BUS_CTRL), 0, "the bit clears itself and takes the rest with it");
    assert_eq!(r.r(reg::BUS_CFG1), 0);
    assert_eq!(r.r(reg::SEMAPHORE), 0);
    assert_eq!(r.r(reg::HOST_CMD), 0);
    assert_eq!(r.r(r.mb(1)), SIG1);
    assert_eq!(r.r(r.mb(7)), 0);
}

#[test]
fn bus_cfg1_keeps_only_its_own_bits() {
    let r = Rig::bare();
    for v in [0x10, 0x20, 0x30, 0x40, 0x30 | CFG1_BURST_ENABLE, CFG1_BANK_SELECT] {
        r.w(reg::BUS_CFG1, v);
        assert_eq!(r.r(reg::BUS_CFG1), v);
    }
    r.w(reg::BUS_CFG1, 0xffff);
    assert_eq!(r.r(reg::BUS_CFG1), CFG1_FIFO_MASK | CFG1_BANK_SELECT | CFG1_BURST_ENABLE);
}

#[test]
fn dma_block_config_is_storage_and_control_is_inert() {
    let r = Rig::bare();
    for (cfg, ctrl) in [(reg::DMA_CMD_CFG, reg::DMA_CMD_CTRL), (reg::DMA_DATA_CFG, reg::DMA_DATA_CTRL)] {
        assert_eq!(r.r(cfg), 0);
        r.w(cfg, 0x0002);
        assert_eq!(r.r(cfg), 0x0002);
        r.w(cfg, 0xffff);
        assert_eq!(r.r(cfg), 0x0002, "only the burst enable is kept");
        r.w(ctrl, 0xffff);
        assert_eq!(r.r(cfg), 0x0002, "a control write changes nothing");
    }
}

#[test]
fn bus_ctrl_status_and_the_interrupt_line() {
    let r = Rig::bare();
    let (probe, line) = LevelProbe::line();
    r.isp.connect_interrupt(line);
    r.w(reg::BUS_CTRL, CTRL_INT_ENABLE | CTRL_RISC_INT_ENABLE);
    assert_eq!(r.r(reg::BUS_CTRL), CTRL_INT_ENABLE | CTRL_RISC_INT_ENABLE);
    r.w(reg::BUS_CTRL, 0xffff);
    assert_eq!(r.r(reg::BUS_CTRL), 0, "the reset bit is in that write and clears itself");
    r.w(reg::BUS_STATUS, 0xffff);
    assert_eq!(r.r(reg::BUS_STATUS), 0, "status ignores writes");

    assert!(!probe.level());
    r.w(reg::BUS_CTRL, CTRL_INT_ENABLE | CTRL_RISC_INT_ENABLE);
    r.w(r.mb(0), mbox::NOP);
    r.w(reg::HOST_CMD, 0x5 << HOST_CMD_SHIFT);
    assert!(probe.level(), "a mailbox result raises the line");
    assert_eq!(r.r(reg::BUS_STATUS), STATUS_INT_REQ | STATUS_RISC_INT_REQ);
    r.w(reg::HOST_CMD, 0x7 << HOST_CMD_SHIFT);
    assert!(!probe.level(), "clearing the RISC interrupt drops it");
    // Masked: the status bit is set but the line is not driven.
    r.w(reg::BUS_CTRL, 0);
    r.w(reg::HOST_CMD, 0x5 << HOST_CMD_SHIFT);
    assert_eq!(r.r(reg::BUS_STATUS), STATUS_INT_REQ | STATUS_RISC_INT_REQ);
    assert!(!probe.level());
}

#[test]
fn semaphore_holds_two_bits() {
    let r = Rig::bare();
    r.w(reg::SEMAPHORE, SEMAPHORE_LOCK);
    assert_eq!(r.r(reg::SEMAPHORE), SEMAPHORE_LOCK);
    r.w(reg::SEMAPHORE, 0);
    assert_eq!(r.r(reg::SEMAPHORE), 0);
    r.w(reg::SEMAPHORE, 0xffff);
    assert_eq!(r.r(reg::SEMAPHORE), 3);
}

#[test]
fn mailboxes_are_independent_storage() {
    let r = Rig::bare();
    for i in 0..8 { r.w(r.mb(i), 0x1000 + i as u16); }
    for i in 0..8 { assert_eq!(r.r(r.mb(i)), 0x1000 + i as u16); }
}

#[test]
fn queue_pointers_are_mailboxes_until_a_queue_exists() {
    let r = Rig::bare();
    assert_eq!(r.r(reg::REQUEST_QUEUE), SIG4);
    assert_eq!(r.r(reg::RESPONSE_QUEUE), 0);
    r.w(reg::REQUEST_QUEUE, 0x21);
    r.w(reg::RESPONSE_QUEUE, 0x43);
    assert_eq!(r.r(reg::REQUEST_QUEUE), 0x21);
    assert_eq!(r.r(reg::RESPONSE_QUEUE), 0x43);
    assert_eq!(r.r(r.mb(4)), 0x21);
    assert_eq!(r.r(r.mb(5)), 0x43);
    r.w(reg::BUS_CTRL, CTRL_SOFT_RESET);
    assert_eq!(r.r(r.mb(4)), SIG4);
    assert_eq!(r.r(r.mb(5)), 0);
}

#[test]
fn host_command_flags() {
    let r = Rig::bare();
    let hc = |c: u16| r.w(reg::HOST_CMD, c << HOST_CMD_SHIFT);
    hc(0x2);
    assert_eq!(r.r(reg::HOST_CMD), RISC_FLAG_PAUSED);
    hc(0x1);
    assert_eq!(r.r(reg::HOST_CMD), RISC_FLAG_RESET, "reset replaces every flag");
    hc(0x3);
    assert_eq!(r.r(reg::HOST_CMD), 0);
    hc(0x5);
    assert_eq!(r.r(reg::HOST_CMD), RISC_FLAG_HOST_INT);
    hc(0x6);
    assert_eq!(r.r(reg::HOST_CMD), RISC_FLAG_HOST_INT, "clear-host-interrupt is a no-op");
    hc(0x7);
    assert_eq!(r.r(reg::HOST_CMD), 0);
    r.w(reg::HOST_CMD, 0x2);
    assert_eq!(r.r(reg::HOST_CMD), 0, "the command number in the low bits does nothing");
}

#[test]
fn bank_select_is_bit_three() {
    let r = Rig::bare();
    r.w(reg::BANK_BASE, 0x1111);
    r.w(reg::BUS_CFG1, CFG1_BANK_SELECT);
    assert_eq!(r.r(reg::BANK_BASE), 0, "the other bank is separate storage");
    r.w(reg::BANK_BASE, 0x2222);
    r.w(reg::BUS_CFG1, 0);
    assert_eq!(r.r(reg::BANK_BASE), 0x1111);
    r.w(reg::BUS_CFG1, CFG1_BANK_SELECT);
    assert_eq!(r.r(reg::BANK_BASE), 0x2222);
    // Bit 0 is a decoy constant in driver headers and switches nothing.
    r.w(reg::BUS_CFG1, 0x0001);
    assert_eq!(r.r(reg::BANK_BASE), 0x1111);
}

#[test]
fn nvram_echoes_its_outputs_and_never_answers() {
    let r = Rig::bare();
    r.w(reg::NVRAM, 0x7);
    assert_eq!(r.r(reg::NVRAM), 0x7);
    r.w(reg::NVRAM, 0xffff);
    assert_eq!(r.r(reg::NVRAM) & 0x8, 0, "data-in is never asserted");
}

#[test]
fn only_sixteen_bit_accesses_are_decoded() {
    let r = Rig::bare();
    assert_eq!(r.isp.read(reg::ID_LOW, 1), 0);
    assert_eq!(r.isp.read(reg::ID_LOW, 4), 0);
    assert!(r.isp.notes().iter().any(|n| n.contains("not decoded")));
}

// ---- mailbox commands --------------------------------------------------

#[test]
fn simple_mailbox_commands() {
    let r = Rig::bare();
    assert_eq!(r.cmd(&[mbox::NOP]), mbox::COMPLETE);
    assert_eq!(r.cmd(&[mbox::REGISTER_TEST, 1, 2, 3, 4, 5, 6, 7]), mbox::COMPLETE);
    for i in 1..8 { assert_eq!(r.r(r.mb(i)), i as u16, "register test leaves mailboxes alone"); }
    assert_eq!(r.cmd(&[mbox::EXEC_FIRMWARE, 0x1000]), mbox::COMPLETE);
    assert_eq!(r.cmd(&[mbox::ABOUT_FIRMWARE]), mbox::COMPLETE);
    assert_eq!((r.r(r.mb(1)), r.r(r.mb(2)), r.r(r.mb(3))), (1, 0, 0));
    assert_eq!(r.cmd(&[mbox::GET_FIRMWARE_STATUS]), mbox::COMPLETE);
    assert_eq!(r.r(r.mb(2)), MAX_REQUESTS);
    assert_eq!(r.cmd(&[mbox::GET_TARGET_PARAMETERS, 0x0100, 0xffff, 0xffff]), mbox::COMPLETE);
    assert_eq!((r.r(r.mb(2)), r.r(r.mb(3))), (0, 0), "asynchronous and narrow");
    assert_eq!(r.cmd(&[mbox::BUS_RESET, 5]), mbox::COMPLETE);
    // The setters a driver issues at attach all complete.
    for c in [
        mbox::SET_INITIATOR_ID, mbox::SET_SELECTION_TIMEOUT, mbox::SET_RETRY_COUNT, mbox::SET_TAG_AGE_LIMIT,
        mbox::SET_CLOCK_RATE, mbox::SET_ACTIVE_NEGATION, mbox::SET_ASYNC_DATA_SETUP, mbox::SET_PCI_CONTROL,
        mbox::SET_TARGET_PARAMETERS, mbox::SET_DEVICE_QUEUE, mbox::SET_SYSTEM_PARAMETER,
        mbox::SET_FIRMWARE_FEATURES, mbox::SET_DATA_OVERRUN_RECOVERY, mbox::SET_RESET_DELAY_PARAMS,
    ] {
        assert_eq!(r.cmd(&[c, 1, 2]), mbox::COMPLETE, "command {c:#x}");
    }
}

#[test]
fn a_command_this_part_does_not_carry_is_invalid() {
    let r = Rig::bare();
    for c in [0x0100, mbox::ABORT_COMMAND, mbox::LOAD_RAM_A64] {
        assert_eq!(r.cmd(&[c]), mbox::INVALID, "command {c:#x}");
    }
    // And the result is still collected the usual way.
    assert_eq!(r.r(reg::SEMAPHORE), 0);
}

#[test]
fn a_result_takes_the_semaphore_but_an_io_completion_does_not() {
    let r = Rig::bare();
    r.w(r.mb(0), mbox::NOP);
    r.w(reg::HOST_CMD, 0x5 << HOST_CMD_SHIFT);
    assert_eq!(r.r(reg::SEMAPHORE), SEMAPHORE_LOCK);
}

#[test]
fn ram_words_round_trip() {
    let r = Rig::bare();
    assert_eq!(r.cmd(&[mbox::WRITE_RAM_WORD, 0x1234, 0xabcd]), mbox::COMPLETE);
    r.w(r.mb(3), 0x7777);
    assert_eq!(r.cmd(&[mbox::READ_RAM_WORD, 0x1234]), mbox::COMPLETE);
    assert_eq!(r.r(r.mb(2)), 0xabcd);
    assert_eq!(r.r(r.mb(3)), 0, "mailbox 3 is an output and is cleared");
}

#[test]
fn firmware_download_verifies_and_a_corrupt_image_does_not() {
    let r = Rig::bare();
    // An image whose word 3 is its length and whose sum is zero.
    let mut words = vec![0x0100u16, 0x1111, 0x2222, 8, 0x0004, 0, 0, 0];
    let sum = words.iter().fold(0u16, |a, &w| a.wrapping_add(w));
    words[7] = 0u16.wrapping_sub(sum);
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    r.mem.put(BUF, &bytes);
    let load = |base: u16| r.cmd(&[mbox::LOAD_RAM, base, (BUF >> 16) as u16, BUF as u16, 8]);
    assert_eq!(load(0x1000), mbox::COMPLETE);
    assert_eq!(r.cmd(&[mbox::VERIFY_CHECKSUM, 0x1000]), mbox::COMPLETE);
    assert_eq!(r.r(r.mb(1)), 0);
    assert_eq!(r.r(r.mb(2)), 0);
    r.cmd(&[mbox::WRITE_RAM_WORD, 0x1005, 0x4242]);
    assert_eq!(r.cmd(&[mbox::VERIFY_CHECKSUM, 0x1000]), mbox::TEST_FAILED);
    assert_ne!(r.r(r.mb(2)), 0, "the firmware reads the mailbox 2 copy of the sum");
}

#[test]
fn ram_block_transfer_is_little_endian_and_reads_back() {
    let r = Rig::bare();
    r.mem.put(BUF, &[0x34, 0x12, 0x78, 0x56, 0xbc, 0x9a]);
    assert_eq!(r.cmd(&[mbox::LOAD_RAM, 0x200, (BUF >> 16) as u16, BUF as u16, 3]), mbox::COMPLETE);
    r.cmd(&[mbox::READ_RAM_WORD, 0x200]);
    assert_eq!(r.r(r.mb(2)), 0x1234, "bytes arrive little-endian");
    r.cmd(&[mbox::READ_RAM_WORD, 0x202]);
    assert_eq!(r.r(r.mb(2)), 0x9abc);
    let out = BUF + 0x100;
    assert_eq!(r.cmd(&[mbox::DUMP_RAM, 0x200, (out >> 16) as u16, out as u16, 3]), mbox::COMPLETE);
    assert_eq!(r.mem.get(out, 6), vec![0x34, 0x12, 0x78, 0x56, 0xbc, 0x9a]);
}

#[test]
fn a_refused_ram_transfer_leaves_the_array_alone() {
    let r = Rig::bare();
    r.cmd(&[mbox::WRITE_RAM_WORD, 0x300, 0x5a5a]);
    let bad = (MEM - 2) as u32;
    assert_eq!(r.cmd(&[mbox::LOAD_RAM, 0x300, (bad >> 16) as u16, bad as u16, 4]), mbox::HOST_INTERFACE_ERROR);
    r.cmd(&[mbox::READ_RAM_WORD, 0x300]);
    assert_eq!(r.r(r.mb(2)), 0x5a5a);
    assert_eq!(r.cmd(&[mbox::DUMP_RAM, 0x300, (bad >> 16) as u16, bad as u16, 4]), mbox::HOST_INTERFACE_ERROR);
}

#[test]
fn queue_init_rejects_a_zero_length_or_address() {
    let r = Rig::bare();
    assert_eq!(r.init_queue(mbox::INIT_REQ_QUEUE, 0, RQST), mbox::PARAM_ERROR);
    assert_eq!(r.init_queue(mbox::INIT_REQ_QUEUE, 4, 0), mbox::PARAM_ERROR);
    assert_eq!(r.init_queue(mbox::INIT_RSP_QUEUE, 0, RESP), mbox::PARAM_ERROR);
    assert_eq!(r.init_queue(mbox::INIT_RSP_QUEUE, 4, 0), mbox::PARAM_ERROR);
}

// ---- commands through the rings ----------------------------------------

#[test]
fn inquiry_moves_data_and_reports_complete() {
    let r = Rig::with_disk("inquiry", 1, 64);
    r.init_rings(4, 4);
    r.put_entry(0, &request(0x1234, 1, &[0x12, 0, 0, 0, 36, 0], &[(BUF, 36)]));
    r.doorbell(1);
    let s = r.status(0);
    assert_eq!(s[0], ENTRY_TYPE_STATUS);
    assert_eq!(g32(&s, 4), 0x1234);
    assert_eq!(g16(&s, 8), 0, "GOOD");
    assert_eq!(g16(&s, 10), IOCB_COMPLETE);
    assert_eq!(g16(&s, 12), STATE_ALL_PHASES & !STATE_GOT_SENSE);
    assert_eq!(g32(&s, 20), 0, "no residual");
    assert_eq!(&r.mem.get(BUF + 8, 8), b"SGI     ");
    assert_eq!(r.r(reg::RESPONSE_QUEUE), 1);
    assert_eq!(r.r(reg::REQUEST_QUEUE), 1, "the adapter reports its consumer position");
    assert_eq!(r.r(reg::BUS_STATUS), STATUS_INT_REQ | STATUS_RISC_INT_REQ);
    assert_eq!(r.r(reg::SEMAPHORE), 0, "I/O completions leave the semaphore alone");
}

#[test]
fn a_short_answer_is_an_underrun_with_its_residual() {
    let r = Rig::with_disk("underrun", 1, 64);
    r.init_rings(4, 4);
    r.put_entry(0, &request(1, 1, &[0x12, 0, 0, 0, 255, 0], &[(BUF, 255)]));
    r.doorbell(1);
    let s = r.status(0);
    assert_eq!(g16(&s, 10), IOCB_DATA_UNDERRUN);
    assert_eq!(g32(&s, 20), 255 - 36);
}

#[test]
fn a_status_entry_overwrites_stale_bytes() {
    let r = Rig::with_disk("stale", 1, 64);
    r.init_rings(4, 4);
    r.mem.put(RESP, &[0xa5; ENTRY_SIZE]);
    r.put_entry(0, &request(7, 1, &[0, 0, 0, 0, 0, 0], &[]));
    r.doorbell(1);
    let s = r.status(0);
    assert_eq!(s[0], ENTRY_TYPE_STATUS);
    assert!(s[33..].iter().all(|&b| b == 0), "no stale bytes survive past the fields");
    assert_eq!(g16(&s, 18), 0, "no sense");
}

#[test]
fn a_write_reaches_the_disk_across_continuation_entries_and_reads_back() {
    let r = Rig::with_disk("write4", 1, 256);
    r.init_rings(8, 8);
    // Four pages in four host buffers: three inline plus one in a
    // continuation entry, 4 KiB each, 8 blocks.
    let pages: Vec<u32> = (0..4).map(|i| BUF + i * 0x2000).collect();
    for (i, &p) in pages.iter().enumerate() {
        let data: Vec<u8> = (0..4096).map(|j| ((j + i * 7) % 251) as u8).collect();
        r.mem.put(p, &data);
    }
    let mut e = request(0x77, 1, &cdb10(scsi_cmd::WRITE_10, 16, 32), &[(pages[0], 4096), (pages[1], 4096), (pages[2], 4096)]);
    p16(&mut e, 18, 4);
    p32(&mut e, 32 + 3 * 8, pages[3]);
    p32(&mut e, 36 + 3 * 8, 4096);
    r.put_entry(0, &e);
    r.doorbell(1);
    let s = r.status(0);
    assert_eq!(g16(&s, 8), 0, "GOOD");
    assert_eq!(g16(&s, 10), IOCB_COMPLETE);
    assert_eq!(g32(&s, 20), 0);

    // Read it back: four inline segments of 2 KiB, then two of 4 KiB in the
    // continuation entry.
    let rd = BUF + 0x20000;
    let inline: Vec<(u32, u32)> = (0..4).map(|i| (rd + 2048 * i, 2048)).collect();
    let mut e = request(0x78, 1, &cdb10(scsi_cmd::READ_10, 16, 32), &inline);
    e[1] = 2;
    p16(&mut e, 18, 6);
    r.put_entry(1, &e);
    let mut cont = [0u8; ENTRY_SIZE];
    cont[0] = 2;
    for i in 0..2u32 {
        p32(&mut cont, 8 + i as usize * 8, rd + 8192 + 4096 * i);
        p32(&mut cont, 12 + i as usize * 8, 4096);
    }
    r.put_entry(2, &cont);
    r.doorbell(3);
    let s = r.status(1);
    assert_eq!(g32(&s, 4), 0x78);
    assert_eq!(g16(&s, 10), IOCB_COMPLETE);
    assert_eq!(g32(&s, 20), 0);
    for (i, &p) in pages.iter().enumerate() {
        assert_eq!(r.mem.get(rd + 4096 * i as u32, 4096), r.mem.get(p, 4096), "page {i}");
    }
    assert_eq!(r.r(reg::REQUEST_QUEUE), 3, "the continuation entry was consumed with its command");
}

/// A WRITE whose payload the fabric refuses writes nothing and completes
/// nothing, as the QEMU model leaves it: the block keeps what it held and
/// the next command runs.
#[test]
fn a_refused_write_payload_reaches_neither_the_disk_nor_the_response_ring() {
    let r = Rig::with_disk("refused-write", 1, 64);
    r.init_rings(8, 8);
    let pattern: Vec<u8> = (0..BLOCK_SIZE).map(|j| (j % 253) as u8 + 1).collect();
    r.mem.put(BUF, &pattern);
    r.put_entry(0, &request(0x80, 1, &cdb10(scsi_cmd::WRITE_10, 5, 1), &[(BUF, BLOCK_SIZE as u32)]));
    r.doorbell(1);
    assert_eq!(g16(&r.status(0), 10), IOCB_COMPLETE);

    // The payload sits past the end of memory.
    r.put_entry(1, &request(0x81, 1, &cdb10(scsi_cmd::WRITE_10, 5, 1), &[(MEM as u32, BLOCK_SIZE as u32)]));
    r.doorbell(2);
    assert_eq!(r.r(reg::REQUEST_QUEUE), 2, "the entry is consumed");
    assert_eq!(r.r(reg::RESPONSE_QUEUE), 1, "but nothing is posted for it");

    let rd = BUF + 0x1000;
    r.put_entry(2, &request(0x82, 1, &cdb10(scsi_cmd::READ_10, 5, 1), &[(rd, BLOCK_SIZE as u32)]));
    r.doorbell(3);
    let s = r.status(1);
    assert_eq!(g32(&s, 4), 0x82);
    assert_eq!(g16(&s, 10), IOCB_COMPLETE);
    assert_eq!(r.mem.get(rd, BLOCK_SIZE), pattern, "the block was not overwritten with zeros");
}

/// A READ whose payload store the fabric refuses completes nothing, as
/// the QEMU model leaves it; the next command runs.
#[test]
fn a_refused_read_payload_posts_no_completion() {
    let r = Rig::with_disk("refused-read", 1, 64);
    r.init_rings(8, 8);
    // Two blocks: the first segment is memory, the second lies past it.
    r.put_entry(0, &request(0x90, 1, &cdb10(scsi_cmd::READ_10, 0, 2), &[(BUF, BLOCK_SIZE as u32), (MEM as u32, BLOCK_SIZE as u32)]));
    r.doorbell(1);
    assert_eq!(r.r(reg::REQUEST_QUEUE), 1, "the entry is consumed");
    assert_eq!(r.r(reg::RESPONSE_QUEUE), 0, "but nothing is posted for it");
    r.put_entry(1, &request(0x91, 1, &[0, 0, 0, 0, 0, 0], &[]));
    r.doorbell(2);
    assert_eq!(g32(&r.status(0), 4), 0x91);
    assert_eq!(r.r(reg::RESPONSE_QUEUE), 1);
}

#[test]
fn a_check_condition_carries_its_sense() {
    let r = Rig::with_disk("sense", 1, 64);
    r.init_rings(4, 4);
    r.put_entry(0, &request(9, 1, &cdb10(scsi_cmd::READ_10, 0x00ff_ffff, 1), &[(BUF, 512)]));
    r.doorbell(1);
    let s = r.status(0);
    assert_eq!(g16(&s, 8), 0x02 | SCSI_STATUS_SENSE_VALID);
    assert_eq!(g16(&s, 10), IOCB_COMPLETE, "the target's status carries the failure, not an underrun");
    assert_ne!(g16(&s, 12) & STATE_GOT_SENSE, 0);
    assert!(g16(&s, 18) > 0, "sense length");
    assert_eq!(s[32] & 0x7f, 0x70);
    assert_eq!(s[34] & 0x0f, 0x05, "illegal request");
}

#[test]
fn an_absent_target_is_incomplete_with_a_timeout() {
    let r = Rig::with_disk("absent", 1, 64);
    r.init_rings(4, 4);
    r.put_entry(0, &request(0x55, 3, &[0, 0, 0, 0, 0, 0], &[]));
    r.doorbell(1);
    let s = r.status(0);
    assert_eq!(g16(&s, 10), IOCB_INCOMPLETE);
    assert_eq!(g16(&s, 12), STATE_GOT_BUS);
    assert_eq!(g16(&s, 14), STATUS_FLAG_TIMEOUT);
}

#[test]
fn an_absent_lun_on_a_present_target_answers_as_scsi_says() {
    let r = Rig::with_disk("lun", 1, 64);
    r.init_rings(8, 8);
    let mut e = request(0x56, 1, &[0x12, 0, 0, 0, 36, 0], &[(BUF, 36)]);
    e[8] = 1;
    r.put_entry(0, &e);
    let mut e = request(0x57, 1, &[0, 0, 0, 0, 0, 0], &[]);
    e[8] = 3;
    r.put_entry(1, &e);
    r.doorbell(2);
    let s = r.status(0);
    assert_eq!(g16(&s, 10), IOCB_COMPLETE, "INQUIRY is answered, not timed out");
    assert_eq!(r.mem.get(BUF, 1)[0] >> 5, 3, "peripheral qualifier 3: no device at this unit");
    let s = r.status(1);
    assert_eq!(g16(&s, 8), 0x02 | SCSI_STATUS_SENSE_VALID);
    assert_eq!(s[34] & 0x0f, 0x05, "illegal request");
    assert_eq!((s[44], s[45]), (0x25, 0x00), "logical unit not supported");
}

#[test]
fn an_opcode_without_a_length_rule_moves_nothing() {
    let r = Rig::with_disk("unsized", 1, 64);
    r.init_rings(4, 4);
    // No buffer is offered to the target; the residual is what the
    // descriptors named.
    r.mem.put(BUF, &[0x5a; 96]);
    r.put_entry(0, &request(0x60, 1, &[0xee, 0, 0, 0, 0, 0], &[(BUF, 64), (BUF + 64, 32)]));
    r.doorbell(1);
    assert_eq!(r.mem.get(BUF, 96), vec![0x5a; 96], "the buffer is untouched");
    let s = r.status(0);
    assert_eq!(g16(&s, 8), 0x02 | SCSI_STATUS_SENSE_VALID);
    assert_eq!(g16(&s, 10), IOCB_COMPLETE);
    assert_eq!(g32(&s, 20), 96);
    assert_eq!(s[34] & 0x0f, 0x05);
}

#[test]
fn the_second_bus_does_not_exist() {
    let r = Rig::with_disk("bus1", 1, 64);
    r.init_rings(4, 4);
    let mut e = request(0x66, 1, &[0, 0, 0, 0, 0, 0], &[]);
    p16(&mut e, 8, 0x8100);
    r.put_entry(0, &e);
    r.doorbell(1);
    let s = r.status(0);
    assert_eq!(g16(&s, 10), IOCB_INCOMPLETE);
    assert_eq!(g16(&s, 12), 0);
    assert_eq!(g16(&s, 14), 0);
}

#[test]
fn an_unknown_entry_type_is_answered_incomplete_and_the_ring_moves_on() {
    let r = Rig::with_disk("unknown", 1, 64);
    r.init_rings(4, 4);
    let mut e = request(0x99, 1, &[0, 0, 0, 0, 0, 0], &[]);
    e[0] = 0x05;
    r.put_entry(0, &e);
    r.put_entry(1, &request(0x9a, 1, &[0, 0, 0, 0, 0, 0], &[]));
    r.doorbell(2);
    assert_eq!(g32(&r.status(0), 4), 0x99);
    assert_eq!(g16(&r.status(0), 10), IOCB_INCOMPLETE);
    assert_eq!(g32(&r.status(1), 4), 0x9a);
    assert_eq!(g16(&r.status(1), 10), IOCB_COMPLETE);
}

#[test]
fn the_walk_stops_at_the_producer_even_with_a_hostile_entry_count() {
    let r = Rig::with_disk("walk", 1, 64);
    r.init_rings(4, 8);
    let mut e = request(0x11, 1, &[0, 0, 0, 0, 0, 0], &[]);
    e[1] = 200; // claims far more entries than are published
    r.put_entry(0, &e);
    r.doorbell(1);
    assert_eq!(r.r(reg::REQUEST_QUEUE), 1, "consumer is clamped to the producer");
    assert_eq!(r.r(reg::RESPONSE_QUEUE), 1);
}

#[test]
fn a_marker_entry_is_consumed_without_an_answer() {
    let r = Rig::with_disk("marker", 1, 64);
    r.init_rings(4, 4);
    let mut e = [0u8; ENTRY_SIZE];
    e[0] = ENTRY_TYPE_MARKER;
    e[1] = 1;
    r.put_entry(0, &e);
    r.doorbell(1);
    assert_eq!(r.r(reg::REQUEST_QUEUE), 1);
    assert_eq!(r.r(reg::RESPONSE_QUEUE), 0);
}

fn tur_entry(handle: u32) -> [u8; ENTRY_SIZE] {
    request(handle, 0, &[0, 0, 0, 0, 0, 0], &[])
}

#[test]
fn four_completions_on_a_four_entry_ring_do_not_read_as_empty() {
    let r = Rig::with_disk("four", 0, 64);
    r.init_rings(8, 4);
    for i in 0..4 { r.put_entry(i, &tur_entry(0x1000 + i)); }
    for i in 0..4 { r.doorbell(i as u16 + 1); }
    assert_ne!(r.r(reg::RESPONSE_QUEUE), 0, "the producer must not rest on the consumer");
}

#[test]
fn an_acknowledgement_does_not_erase_a_later_completion() {
    let r = Rig::with_disk("ack", 0, 64);
    r.init_rings(8, 8);
    r.put_entry(0, &tur_entry(0x2001));
    r.doorbell(1);
    let first = r.r(reg::RESPONSE_QUEUE);
    assert_eq!(first, 1);
    r.put_entry(1, &tur_entry(0x2002));
    r.doorbell(2);
    r.w(reg::RESPONSE_QUEUE, first);
    assert_eq!(r.r(reg::RESPONSE_QUEUE), 2, "the adapter's own count, not the number just written");
}

#[test]
fn a_held_completion_posts_and_resignals_when_the_driver_frees_room() {
    let r = Rig::with_disk("held", 0, 64);
    r.init_rings(8, 4);
    for i in 0..3 {
        r.put_entry(i, &tur_entry(0x3000 + i));
        r.doorbell(i as u16 + 1);
    }
    assert_eq!(r.r(reg::RESPONSE_QUEUE), 3);
    r.ack();
    assert_eq!(r.r(reg::BUS_STATUS), 0);
    r.put_entry(3, &tur_entry(0x3003));
    r.doorbell(4);
    assert_eq!(r.r(reg::RESPONSE_QUEUE), 3, "held, not lost");
    assert_eq!(r.r(reg::BUS_STATUS), 0, "and not yet announced");
    r.w(reg::RESPONSE_QUEUE, 1);
    assert_eq!(r.r(reg::BUS_STATUS), STATUS_INT_REQ | STATUS_RISC_INT_REQ);
    assert_eq!(r.r(reg::RESPONSE_QUEUE), 0);
    let s = r.status(3);
    assert_eq!(s[0], ENTRY_TYPE_STATUS);
    assert_eq!(g32(&s, 4), 0x3003);
    assert_eq!(g16(&s, 10), IOCB_COMPLETE);
}

#[test]
fn a_held_completion_stalls_the_request_ring_until_room_appears() {
    let r = Rig::with_disk("stall", 0, 64);
    r.init_rings(8, 3);
    for i in 0..4 { r.put_entry(i, &tur_entry(0x4000 + i)); }
    r.doorbell(4);
    // Room for two completions; the third is held and stops the walk.
    assert_eq!(r.r(reg::RESPONSE_QUEUE), 2);
    assert_eq!(r.r(reg::REQUEST_QUEUE), 3);
    r.w(reg::RESPONSE_QUEUE, 2);
    assert_eq!(r.r(reg::REQUEST_QUEUE), 4, "the acknowledgement lets the ring run again");
}

// ---- A64 entries -------------------------------------------------------

fn a64_request(handle: u32, target: u8, cdb: &[u8], segs: &[(u64, u32)]) -> [u8; ENTRY_SIZE] {
    let mut e = [0u8; ENTRY_SIZE];
    e[0] = ENTRY_TYPE_COMMAND_A64;
    e[1] = 1;
    p32(&mut e, 4, handle);
    p16(&mut e, 8, (target as u16) << 8);
    p16(&mut e, 10, cdb.len() as u16);
    p16(&mut e, 18, segs.len() as u16);
    e[20..20 + cdb.len()].copy_from_slice(cdb);
    for (i, &(a, l)) in segs.iter().take(2).enumerate() {
        p32(&mut e, 40 + i * 12, a as u32);
        p32(&mut e, 44 + i * 12, (a >> 32) as u32);
        p32(&mut e, 48 + i * 12, l);
    }
    e
}

#[test]
fn a64_inquiry_and_continuation_segments() {
    let r = Rig::with_disk("a64", 1, 64);
    assert_eq!(r.cmd(&[mbox::INIT_REQ_QUEUE_A64, 4, 0, RQST as u16, 0, 0, 0, 0]), mbox::COMPLETE);
    assert_eq!(r.cmd(&[mbox::INIT_RSP_QUEUE_A64, 4, 0, RESP as u16, 0, 0, 0, 0]), mbox::COMPLETE);
    r.put_entry(0, &a64_request(0xa1, 1, &[0x12, 0, 0, 0, 36, 0], &[(BUF as u64, 36)]));
    r.doorbell(1);
    let s = r.status(0);
    assert_eq!(g32(&s, 4), 0xa1);
    assert_eq!(g16(&s, 10), IOCB_COMPLETE);
    assert_eq!(&r.mem.get(BUF + 8, 8), b"SGI     ");

    // A read of 4 blocks over five segments: two inline, three in the
    // continuation (twelve-byte segments, no handle).
    let rd = BUF + 0x1000;
    let mut e = a64_request(0xa2, 1, &cdb10(scsi_cmd::READ_10, 0, 4), &[(rd as u64, 512), (rd as u64 + 512, 512)]);
    e[1] = 2;
    p16(&mut e, 18, 4);
    r.put_entry(1, &e);
    let mut cont = [0u8; ENTRY_SIZE];
    cont[0] = 0x0a;
    for i in 0..2u32 {
        p32(&mut cont, 4 + i as usize * 12, rd + 1024 + 512 * i);
        p32(&mut cont, 12 + i as usize * 12, 512);
    }
    r.put_entry(2, &cont);
    r.doorbell(3);
    let s = r.status(1);
    assert_eq!(g32(&s, 4), 0xa2);
    assert_eq!(g16(&s, 10), IOCB_COMPLETE);
    assert_eq!(g32(&s, 20), 0);
}

#[test]
fn a_queue_above_four_gigabytes_cannot_be_reached_on_a_32_bit_bus() {
    let r = Rig::with_disk("high", 1, 64);
    assert_eq!(r.cmd(&[mbox::INIT_REQ_QUEUE_A64, 4, 0, RQST as u16, 0, 0, 0, 0]), mbox::COMPLETE);
    assert_eq!(r.cmd(&[mbox::INIT_RSP_QUEUE_A64, 4, 0, 0x2000, 0, 0, 1, 0]), mbox::COMPLETE);
    r.put_entry(0, &tur_entry(1));
    r.doorbell(1);
    assert_eq!(r.r(reg::RESPONSE_QUEUE), 0, "the completion cannot be stored, so it is dropped");
    assert!(r.isp.notes().iter().any(|n| n.contains("completion dropped")));
}

// ---- state -------------------------------------------------------------

// ---- guest-chosen sizes and bases at their limits --------------------

/// A request ring longer than 0x8000 entries, with the consumer about to
/// wrap: the index arithmetic wraps modulo the ring, it does not overflow.
#[test]
fn a_ring_of_65535_entries_wraps_its_consumer() {
    let r = Rig::with_disk("long-ring", 0, 64);
    r.init_rings(0xffff, 8);
    let mut v = r.isp.save_state();
    if let toml::Value::Table(t) = &mut v {
        t.insert("rqst_cons".into(), hex_u16(0xfffe));
    }
    r.isp.load_state(&v).expect("load_state");
    // Entry 0xfffe lies past memory and reads as zeros; entry 0 is real.
    r.put_entry(0, &tur_entry(0x7001));
    r.doorbell(1);
    assert_eq!(r.r(reg::REQUEST_QUEUE), 1, "0xfffe, then 0, then stop at the producer");
    let posted = r.r(reg::RESPONSE_QUEUE);
    let handles: Vec<u32> = (0..posted as u32).map(|i| g32(&r.status(i), 4)).collect();
    assert!(handles.contains(&0x7001), "the real entry ran: {handles:x?}");
}

/// A loaded consumer index past the end of its ring folds into it.
#[test]
fn a_loaded_index_past_its_ring_folds_into_it() {
    let r = Rig::with_disk("fold", 0, 64);
    r.init_rings(4, 4);
    let mut v = r.isp.save_state();
    if let toml::Value::Table(t) = &mut v {
        t.insert("rqst_cons".into(), hex_u16(0xffff));
        t.insert("resp_prod".into(), hex_u16(0xffff));
        // The driver's producer sits on the folded consumer: nothing to run.
        let mut mb = [0u16; MAILBOX_COUNT];
        load_u16_slice(t.get("mailbox").unwrap(), &mut mb);
        mb[MBOX_REQ] = 0xffff % 4;
        t.insert("mailbox".into(), u16_slice_to_toml(&mb));
    }
    r.isp.load_state(&v).expect("load_state");
    r.isp.wait_idle();
    assert_eq!(r.r(reg::REQUEST_QUEUE), 0xffff % 4);
    assert_eq!(r.r(reg::RESPONSE_QUEUE), 0xffff % 4);
}

/// Ring bases at the top of the 64-bit space: an entry whose address
/// would wrap past it is a refused transfer, and the walk stops there.
#[test]
fn ring_bases_at_the_top_of_the_bus_are_refused_not_wrapped() {
    let r = Rig::with_disk("top", 1, 64);
    let top = [0xffff, 0xfff0, 0, 0, 0xffff, 0xffff];
    assert_eq!(r.cmd(&[mbox::INIT_REQ_QUEUE_A64, 4, top[0], top[1], 0, 0, top[4], top[5]]), mbox::COMPLETE);
    assert_eq!(r.cmd(&[mbox::INIT_RSP_QUEUE_A64, 4, 0xffff, 0xffc0, 0, 0, 0xffff, 0xffff]), mbox::COMPLETE);
    r.doorbell(2);
    // Entry 0 (whose fields straddle the top) reads as zeros; entry 1 would
    // wrap, so the walk stops on it.
    assert_eq!(r.r(reg::REQUEST_QUEUE), 1);
    assert_eq!(r.r(reg::RESPONSE_QUEUE), 0, "the answer could not be stored");
    // The device still answers.
    assert_eq!(r.cmd(&[mbox::NOP]), mbox::COMPLETE);
}

#[test]
fn save_load_round_trip() {
    let r = Rig::with_disk("save", 0, 64);
    r.init_rings(8, 3);
    r.cmd(&[mbox::WRITE_RAM_WORD, 0x40, 0x1234]);
    for i in 0..4 { r.put_entry(i, &tur_entry(0x5000 + i)); }
    r.doorbell(4); // one completion ends up held
    let v1 = r.isp.save_state();
    let dst = Rig::with_disk("save2", 0, 64);
    dst.isp.load_state(&v1).expect("load_state");
    assert_eq!(v1, dst.isp.save_state());
    dst.cmd(&[mbox::READ_RAM_WORD, 0x40]);
    assert_eq!(dst.r(dst.mb(2)), 0x1234);
    dst.w(reg::RESPONSE_QUEUE, 2);
    assert_eq!(dst.r(reg::REQUEST_QUEUE), 4, "the held completion and the ring carry over");
}

#[test]
fn a_disk_attaches_where_asked_and_the_monitor_can_see_it() {
    use crate::scsi_monitor::ScsiTargets;
    let r = Rig::bare();
    let blocks = r.isp.bus().attach_image(1, &image("attach", 32), false).unwrap();
    assert_eq!(blocks, 32);
    assert!(r.isp.with_targets(|t| t[1].is_some() && t[0].is_none()));
    assert!(r.isp.bus().attach_image(16, &image("attach", 32), false).is_err());
}

/// Records the bus address of each transfer.
struct AddrProbe(Mutex<Vec<u64>>);

impl PciDma for AddrProbe {
    fn dma_read(&self, addr: u64, buf: &mut [u8]) -> PciDmaStatus {
        self.0.lock().unwrap().push(addr);
        buf.fill(0);
        PciDmaStatus::Ok
    }
    fn dma_write(&self, addr: u64, _data: &[u8]) -> PciDmaStatus {
        self.0.lock().unwrap().push(addr);
        PciDmaStatus::Ok
    }
}

#[test]
fn dma_above_4_gib_reaches_the_bus_in_full() {
    let isp = Isp1040::new();
    Device::start(&isp);
    let probe = Arc::new(AddrProbe(Mutex::new(Vec::new())));
    isp.connect_dma(probe.clone());
    let mut b = [0u8; 4];
    assert!(isp.dma_read(0x1_2345_6000, &mut b));
    assert!(isp.dma_write(0xffff_ffff_ffff_f000, &b));
    // A transfer that would wrap past the top of the space is refused.
    assert!(!isp.dma_write(0xffff_ffff_ffff_fffe, &b));
    assert_eq!(*probe.0.lock().unwrap(), [0x1_2345_6000, 0xffff_ffff_ffff_f000]);
}

// ---- commands off the CPU thread ---------------------------------------

/// Fake memory whose accesses to one address wait until the test opens the
/// gate: a disk answer that is slow to arrive, or a slow payload fetch.
struct GatedMem {
    mem: Arc<FakeMem>,
    addr: u64,
    /// (a write is waiting at the gate, the gate is open)
    gate: Mutex<(bool, bool)>,
    cv: std::sync::Condvar,
}

impl GatedMem {
    fn new(mem: Arc<FakeMem>, addr: u32) -> Arc<Self> {
        Arc::new(GatedMem { mem, addr: addr as u64, gate: Mutex::new((false, false)), cv: std::sync::Condvar::new() })
    }
    /// Block until an access reached the gate.
    fn wait_arrival(&self) {
        let mut g = self.gate.lock().unwrap();
        while !g.0 { g = self.cv.wait(g).unwrap(); }
    }
    fn open(&self) {
        self.gate.lock().unwrap().1 = true;
        self.cv.notify_all();
    }
    fn pass(&self, addr: u64) {
        if addr == self.addr {
            let mut g = self.gate.lock().unwrap();
            g.0 = true;
            self.cv.notify_all();
            while !g.1 { g = self.cv.wait(g).unwrap(); }
        }
    }
}

impl PciDma for GatedMem {
    fn dma_read(&self, addr: u64, buf: &mut [u8]) -> PciDmaStatus {
        self.pass(addr);
        self.mem.dma_read(addr, buf)
    }
    fn dma_write(&self, addr: u64, data: &[u8]) -> PciDmaStatus {
        self.pass(addr);
        self.mem.dma_write(addr, data)
    }
}

/// A chip with a disk at target 1 whose INQUIRY answer into `BUF` waits at
/// the gate, rings initialised and the interrupt enabled into a probe.
fn gated_rig(name: &str) -> (Rig, Arc<GatedMem>, Arc<LevelProbe>) {
    let mem = FakeMem::new();
    let gate = GatedMem::new(mem.clone(), BUF);
    let isp = Isp1040::new();
    Device::start(&isp);
    isp.connect_dma(gate.clone());
    isp.bus().attach_image(1, &image(name, 64), false).expect("attach");
    let (probe, line) = LevelProbe::line();
    isp.connect_interrupt(line);
    let r = Rig { isp, mem };
    r.init_rings(4, 4);
    r.w(reg::BUS_CTRL, CTRL_INT_ENABLE | CTRL_RISC_INT_ENABLE);
    r.put_entry(0, &request(0x77, 1, &[0x12, 0, 0, 0, 36, 0], &[(BUF, 36)]));
    (r, gate, probe)
}

/// The doorbell store returns while the command is still waiting on its
/// target, and the registers answer meanwhile; the completion and the
/// interrupt follow when the target does. A guest's clock interrupt is
/// never held behind a disk.
#[test]
fn the_doorbell_returns_before_the_command_completes() {
    let (r, gate, probe) = gated_rig("async");
    r.isp.write(reg::REQUEST_QUEUE, 1, 2);
    gate.wait_arrival();
    assert_eq!(r.r(reg::RESPONSE_QUEUE), 0, "nothing posted yet");
    assert_eq!(r.r(reg::BUS_STATUS), 0);
    assert!(!probe.level());
    gate.open();
    r.isp.wait_idle();
    assert_eq!(r.r(reg::RESPONSE_QUEUE), 1);
    assert_eq!(g32(&r.status(0), 4), 0x77);
    assert_eq!(&r.mem.get(BUF + 8, 8), b"SGI     ");
    assert!(probe.level(), "the completion raises the line from the adapter's thread");
}

/// A soft reset cancels the command in flight without waiting for its
/// target: the reset store returns while the target is still busy, and
/// the command then reaches neither the disk nor the response ring.
#[test]
fn a_soft_reset_does_not_wait_for_the_target() {
    let mem = FakeMem::new();
    let gate = GatedMem::new(mem.clone(), BUF);
    let isp = Isp1040::new();
    Device::start(&isp);
    isp.connect_dma(gate.clone());
    isp.bus().attach_image(1, &image("async-reset", 64), false).expect("attach");
    let r = Rig { isp, mem };
    let (probe, line) = LevelProbe::line();
    r.isp.connect_interrupt(line);
    r.init_rings(4, 4);
    r.w(reg::BUS_CTRL, CTRL_INT_ENABLE | CTRL_RISC_INT_ENABLE);
    r.mem.put(BUF, &[0x5a; BLOCK_SIZE]);
    r.put_entry(0, &request(0x77, 1, &cdb10(scsi_cmd::WRITE_10, 3, 1), &[(BUF, BLOCK_SIZE as u32)]));
    // The payload fetch stops at the gate; the target is then made busy
    // with something else, so the command waits for it.
    r.isp.write(reg::REQUEST_QUEUE, 1, 2);
    gate.wait_arrival();
    assert!(r.isp.command_running());
    let busy = r.isp.bus().lock();
    gate.open();
    r.isp.write(reg::BUS_CTRL, CTRL_SOFT_RESET as u32, 2);
    assert_eq!(r.r(r.mb(4)), SIG4, "the reset happened with the target still busy");
    drop(busy);
    r.isp.wait_idle();
    assert_eq!(g32(&r.status(0), 4), 0, "no status entry was written");
    assert_eq!(r.r(reg::BUS_STATUS), 0);
    assert!(!probe.level());
    // The block was never written.
    r.init_rings(4, 4);
    r.put_entry(0, &request(0x78, 1, &cdb10(scsi_cmd::READ_10, 3, 1), &[(BUF + 0x1000, BLOCK_SIZE as u32)]));
    r.doorbell(1);
    assert_eq!(g32(&r.status(0), 4), 0x78);
    assert_eq!(r.mem.get(BUF + 0x1000, BLOCK_SIZE), vec![0; BLOCK_SIZE]);
}

/// A soft reset during a store into host memory waits for that one store,
/// then nothing more of the cancelled command reaches memory.
#[test]
fn a_soft_reset_during_a_store_drops_the_completion() {
    let (r, gate, probe) = gated_rig("async-reset-store");
    r.isp.write(reg::REQUEST_QUEUE, 1, 2);
    gate.wait_arrival();
    let before = r.isp.gen.load(Ordering::Acquire);
    std::thread::scope(|s| {
        let reset = s.spawn(|| r.isp.write(reg::BUS_CTRL, CTRL_SOFT_RESET as u32, 2));
        // The reset has moved the generation on and waits for the store.
        while r.isp.gen.load(Ordering::Acquire) == before { std::thread::yield_now(); }
        assert!(!reset.is_finished());
        gate.open();
        reset.join().unwrap();
    });
    r.isp.wait_idle();
    assert_eq!(g32(&r.status(0), 4), 0, "no status entry was written");
    assert_eq!(r.r(reg::BUS_STATUS), 0);
    assert!(!probe.level());
    assert_eq!(r.r(r.mb(4)), SIG4, "the reset state, untouched by the late answer");
}

/// A response ring replaced while a command runs gets that command's
/// completion; the command is not read and run a second time.
#[test]
fn a_new_response_ring_takes_the_completion_of_the_command_in_flight() {
    let (r, gate, _probe) = gated_rig("async-rsp-init");
    const RESP2: u32 = 0x3000;
    r.isp.write(reg::REQUEST_QUEUE, 1, 2);
    gate.wait_arrival();
    // The initialisation does not wait for the command.
    assert_eq!(r.init_queue(mbox::INIT_RSP_QUEUE, 4, RESP2), mbox::COMPLETE);
    gate.open();
    r.isp.wait_idle();
    assert_eq!(r.isp.total_commands(), 1, "the command ran once");
    assert_eq!(r.r(reg::REQUEST_QUEUE), 1, "and its entry was consumed");
    assert_eq!(r.r(reg::RESPONSE_QUEUE), 1);
    let s: [u8; ENTRY_SIZE] = r.mem.get(RESP2, ENTRY_SIZE).try_into().unwrap();
    assert_eq!(g32(&s, 4), 0x77, "the completion is on the new ring");
    assert_eq!(g32(&r.status(0), 4), 0, "not the old one");
}

/// A request ring replaced while a command runs starts at its own index
/// 0; the completion of the command that was running is still posted.
#[test]
fn a_new_request_ring_starts_at_zero_and_the_command_in_flight_completes() {
    let (r, gate, _probe) = gated_rig("async-req-init");
    r.isp.write(reg::REQUEST_QUEUE, 1, 2);
    gate.wait_arrival();
    assert_eq!(r.init_queue(mbox::INIT_REQ_QUEUE, 4, RQST), mbox::COMPLETE);
    gate.open();
    r.isp.wait_idle();
    assert_eq!(r.isp.total_commands(), 1);
    assert_eq!(r.r(reg::REQUEST_QUEUE), 0, "the new ring's consumer is untouched");
    assert_eq!(g32(&r.status(0), 4), 0x77);
    assert_eq!(r.r(reg::RESPONSE_QUEUE), 1);
}

/// While the machine is stopped the adapter takes no entries; on start
/// it runs what the driver published meanwhile.
#[test]
fn a_paused_adapter_takes_nothing_until_it_resumes() {
    let r = Rig::with_disk("pause", 0, 64);
    r.init_rings(8, 8);
    r.isp.pause();
    r.put_entry(0, &tur_entry(0x5001));
    r.doorbell(1);
    assert_eq!(r.r(reg::REQUEST_QUEUE), 0, "nothing taken while paused");
    assert_eq!(r.r(reg::RESPONSE_QUEUE), 0);
    r.isp.resume();
    r.isp.wait_idle();
    assert_eq!(r.r(reg::REQUEST_QUEUE), 1);
    assert_eq!(g32(&r.status(0), 4), 0x5001);
}

/// Pausing waits for the command in flight, which then posts; nothing
/// follows it until the adapter resumes.
#[test]
fn pausing_lets_the_command_in_flight_finish() {
    let (r, gate, probe) = gated_rig("pause-inflight");
    r.put_entry(1, &request(0x78, 1, &[0, 0, 0, 0, 0, 0], &[]));
    r.isp.write(reg::REQUEST_QUEUE, 2, 2);
    gate.wait_arrival();
    std::thread::scope(|s| {
        let pause = s.spawn(|| r.isp.pause());
        while !r.isp.st.lock().ctl.paused { std::thread::yield_now(); }
        gate.open();
        pause.join().unwrap();
    });
    assert_eq!(r.r(reg::RESPONSE_QUEUE), 1, "the command in flight posted");
    assert!(probe.level());
    assert_eq!(r.r(reg::REQUEST_QUEUE), 1, "and the next waits");
    r.isp.resume();
    r.isp.wait_idle();
    assert_eq!(r.r(reg::REQUEST_QUEUE), 2);
    assert_eq!(g32(&r.status(1), 4), 0x78);
}

/// Saving waits for the command in flight, so the state saved has it
/// posted rather than half done.
#[test]
fn saving_waits_for_the_command_in_flight() {
    let (r, gate, _probe) = gated_rig("async-save");
    r.isp.write(reg::REQUEST_QUEUE, 1, 2);
    gate.wait_arrival();
    let saved = std::thread::scope(|s| {
        let save = s.spawn(|| r.isp.save_state());
        while !r.isp.waiting_on_command() { std::thread::yield_now(); }
        gate.open();
        save.join().unwrap()
    });
    assert_eq!(get_field(&saved, "resp_prod").and_then(toml_u16), Some(1));
    assert_eq!(get_field(&saved, "rqst_cons").and_then(toml_u16), Some(1));
}

/// Entries the driver queued but the adapter had not reached when the
/// state was saved run after it is loaded.
#[test]
fn queued_entries_run_after_a_load() {
    let src = Rig::with_disk("async-load-src", 0, 64);
    src.init_rings(8, 8);
    let mut v = src.isp.save_state();
    // The driver had published two entries the adapter had not consumed.
    if let toml::Value::Table(t) = &mut v {
        let mut mb = [0u16; MAILBOX_COUNT];
        load_u16_slice(t.get("mailbox").unwrap(), &mut mb);
        mb[MBOX_REQ] = 2;
        t.insert("mailbox".into(), u16_slice_to_toml(&mb));
    }
    let dst = Rig::with_disk("async-load-dst", 0, 64);
    dst.put_entry(0, &tur_entry(0x6000));
    dst.put_entry(1, &tur_entry(0x6001));
    dst.isp.load_state(&v).expect("load_state");
    dst.isp.wait_idle();
    assert_eq!(dst.r(reg::REQUEST_QUEUE), 2);
    assert_eq!(dst.r(reg::RESPONSE_QUEUE), 2);
    assert_eq!(g32(&dst.status(1), 4), 0x6001);
}

/// The ring thread exists only between `start` and `stop`: nothing runs
/// before the first or after the last, and a restart takes what the driver
/// published meanwhile.
#[test]
fn the_ring_thread_lives_between_start_and_stop() {
    let isp = Isp1040::new();
    assert!(!isp.is_running(), "the constructor starts no thread");
    let mem = FakeMem::new();
    isp.connect_dma(mem.clone());
    isp.bus().attach_image(0, &image("lifecycle", 64), false).expect("attach");
    let r = Rig { isp, mem };
    r.init_rings(8, 8);
    r.isp.start();
    r.isp.start();
    assert!(r.isp.is_running());
    r.isp.stop();
    r.isp.stop();
    assert!(!r.isp.is_running());
    r.put_entry(0, &tur_entry(0x7001));
    r.doorbell(1);
    assert_eq!(r.r(reg::RESPONSE_QUEUE), 0, "nothing taken while stopped");
    r.isp.start();
    r.isp.wait_idle();
    assert_eq!(r.r(reg::REQUEST_QUEUE), 1);
    assert_eq!(g32(&r.status(0), 4), 0x7001);
}

/// The machine fills the bus from `[scsi.N]` and gives it to the controller,
/// which answers for what is on it and still takes a target attached later
/// through the same bus.
#[test]
fn a_controller_built_on_a_filled_bus_answers_for_its_target() {
    let img = std::env::temp_dir().join(format!("iris-isp1040-filled-{}.img", std::process::id()));
    std::fs::write(&img, vec![0u8; 64 * 512]).unwrap();
    let bus = Isp1040::new_bus();
    bus.attach_image(3, &img, false).unwrap();
    let isp = Isp1040::new_at(1, bus);
    assert!(isp.bus().has_target(3));
    assert!(!isp.bus().has_target(4));
    let inquiry = crate::scsi::ScsiRequest {
        cdb: vec![crate::scsi::scsi_cmd::INQUIRY, 0, 0, 0, 36, 0],
        data_len: crate::scsi::ScsiDataLength::Unlimited,
        data_in: None,
    };
    let r = isp.bus().request(3, &inquiry).expect("target answers").unwrap();
    assert_eq!(&r.data[8..16], b"SGI     ");
    assert_eq!(&r.data[16..32], b"IRIS EMUL DISK  ");
    isp.bus().attach_image(4, &img, false).unwrap();
    assert!(isp.bus().has_target(4), "a target attached after boot goes on the same bus");
    let _ = std::fs::remove_file(img);
}


struct MarkerTarget(u8);

impl crate::scsi::ScsiTarget for MarkerTarget {
    fn request(&mut self, _req: &crate::scsi::ScsiRequest) -> std::io::Result<crate::scsi::ScsiResponse> {
        Ok(crate::scsi::ScsiResponse { status: 0, data: vec![self.0] })
    }
}

/// Target 0 on channel 1 must stay distinct from target 0 on channel 0.
#[test]
fn the_isp12160_routes_ring_requests_to_separate_buses() {
    let bus0 = Isp1040::new_bus();
    let bus1 = Isp1040::new_bus();
    bus0.attach(0, Box::new(MarkerTarget(0x10))).unwrap();
    bus1.attach(0, Box::new(MarkerTarget(0x20))).unwrap();
    let r = Rig::on(Isp1040::new_12160(bus0, bus1));
    r.init_rings(4, 4);

    r.put_entry(0, &request(0x1234, 0x80, &[0x12, 0, 0, 0, 36, 0], &[(BUF, 36)]));
    r.doorbell(1);
    assert_eq!(r.mem.get(BUF, 1), vec![0x20], "channel 1 reaches its target 0");

    r.put_entry(1, &request(0x1235, 0, &[0x12, 0, 0, 0, 36, 0], &[(BUF + 64, 36)]));
    r.doorbell(2);
    assert_eq!(r.mem.get(BUF + 64, 1), vec![0x10], "channel 0 keeps its own target 0");
}

struct ResetProbe(Arc<std::sync::atomic::AtomicUsize>);

impl crate::scsi::ScsiTarget for ResetProbe {
    fn request(&mut self, _req: &crate::scsi::ScsiRequest) -> std::io::Result<crate::scsi::ScsiResponse> {
        Ok(crate::scsi::ScsiResponse { status: 0, data: Vec::new() })
    }

    fn bus_reset(&mut self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

/// A snapshot retains both the chip-wide reset and the selected target reset,
/// and the resumed worker delivers both through the original channel buses.
#[test]
fn the_isp12160_preserves_queued_bus_and_target_resets_across_save_load() {
    let src = Isp1040::new_12160(Isp1040::new_bus(), Isp1040::new_bus());
    {
        let mut state = src.st.lock();
        state.reset_targets = true;
        state.pending_target_resets.push_back((1, Some(2)));
    }
    let saved = src.save_state();

    let resets0 = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let resets1 = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let selected_reset = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let bus0 = Isp1040::new_bus();
    let bus1 = Isp1040::new_bus();
    bus0.attach(1, Box::new(ResetProbe(resets0.clone()))).unwrap();
    bus1.attach(1, Box::new(ResetProbe(resets1.clone()))).unwrap();
    bus1.attach(2, Box::new(ResetProbe(selected_reset.clone()))).unwrap();
    let dst = Isp1040::new_12160(bus0, bus1);
    dst.load_state(&saved).unwrap();
    Device::start(&dst);
    dst.wait_idle();
    Device::stop(&dst);

    assert_eq!(resets0.load(Ordering::Relaxed), 1);
    assert_eq!(resets1.load(Ordering::Relaxed), 1);
    assert_eq!(selected_reset.load(Ordering::Relaxed), 2, "the full reset and queued target reset both run");
    assert_eq!(saved.get("reset_targets").and_then(toml::Value::as_bool), Some(true));
    assert_eq!(saved.get("target_resets").and_then(toml::Value::as_array).map(Vec::len), Some(1));
    let after = dst.save_state();
    assert_eq!(after.get("reset_targets").and_then(toml::Value::as_bool), Some(false));
    assert_eq!(after.get("target_resets").and_then(toml::Value::as_array).map(Vec::len), Some(0));
}

// ---- ISP12160 second channel and recovery commands -----------------------

fn rig12160() -> Rig { Rig::on(Isp1040::new_12160(Isp1040::new_bus(), Isp1040::new_bus())) }

/// A 64-block image whose first block is `fill`, so a read says which disk answered.
fn filled_image(name: &str, fill: u8) -> std::path::PathBuf {
    let p = image(name, 64);
    let mut data = vec![0u8; 64 * BLOCK_SIZE];
    data[..BLOCK_SIZE].fill(fill);
    std::fs::write(&p, data).expect("test image");
    p
}

/// A 12160 with a different disk at target 1 on each channel.
fn disks_on_both_channels(name: &str) -> Rig {
    let r = rig12160();
    for channel in 0..2u8 {
        let img = filled_image(&format!("{name}{channel}"), 0xa0 + channel);
        r.isp.bus_for_channel(channel).unwrap().attach_image(1, &img, false).expect("attach");
    }
    r
}

/// Issues entries one at a time on a 16-entry ring and returns each completion.
struct Seq<'a> {
    r: &'a Rig,
    n: u16,
}

impl<'a> Seq<'a> {
    fn new(r: &'a Rig) -> Self {
        r.init_rings(16, 16);
        Seq { r, n: 0 }
    }
    fn run(&mut self, handle: u32, target: u8, cdb: &[u8], segs: &[(u32, u32)]) -> [u8; ENTRY_SIZE] {
        let slot = self.n;
        self.r.put_entry(slot as u32, &request(handle, target, cdb, segs));
        self.n += 1;
        self.r.doorbell(self.n);
        self.r.status(slot as u32)
    }
}

const TUR: [u8; 6] = [0; 6];

/// Counts bus resets and answers any command with a full block pair, so a read
/// whose second buffer segment is unreachable stays outstanding.
struct CountingDisk(Arc<std::sync::atomic::AtomicUsize>);

impl crate::scsi::ScsiTarget for CountingDisk {
    fn request(&mut self, _req: &crate::scsi::ScsiRequest) -> std::io::Result<crate::scsi::ScsiResponse> {
        Ok(crate::scsi::ScsiResponse { status: 0, data: vec![0; 2 * BLOCK_SIZE] })
    }
    fn bus_reset(&mut self) { self.0.fetch_add(1, Ordering::Relaxed); }
}

struct Held {
    r: Rig,
    /// Bus resets seen by the counting disks at target 1 on channel 0, target 1 on channel 1,
    /// and target 2 on channel 1.
    ch0: Arc<std::sync::atomic::AtomicUsize>,
    ch1: Arc<std::sync::atomic::AtomicUsize>,
    ch1_other: Arc<std::sync::atomic::AtomicUsize>,
}

impl Held {
    fn new() -> Held {
        let counters: Vec<_> = (0..3).map(|_| Arc::new(std::sync::atomic::AtomicUsize::new(0))).collect();
        let (bus0, bus1) = (Isp1040::new_bus(), Isp1040::new_bus());
        bus0.attach(1, Box::new(CountingDisk(counters[0].clone()))).unwrap();
        bus1.attach(1, Box::new(CountingDisk(counters[1].clone()))).unwrap();
        bus1.attach(2, Box::new(CountingDisk(counters[2].clone()))).unwrap();
        let r = Rig::on(Isp1040::new_12160(bus0, bus1));
        r.init_rings(16, 16);
        Held { r, ch0: counters[0].clone(), ch1: counters[1].clone(), ch1_other: counters[2].clone() }
    }
    /// A read whose payload store the fabric refuses: the adapter keeps it outstanding, which
    /// is what the abort and reset commands act on.
    fn hold(&self, slot: u32, handle: u32, target: u8) {
        self.r.put_entry(slot, &request(handle, target, &cdb10(scsi_cmd::READ_10, 0, 2),
            &[(BUF, BLOCK_SIZE as u32), (MEM as u32, BLOCK_SIZE as u32)]));
        self.r.doorbell(slot as u16 + 1);
        assert_eq!(g32(&self.r.status(slot), 4), 0, "handle {handle:#x} is held, not completed");
    }
    fn command(&self, regs: &[u16]) -> u16 {
        let status = self.r.cmd(regs);
        self.r.isp.wait_idle();
        status
    }
    /// The completion posted in response slot `slot`: (handle, completion code).
    fn answer(&self, slot: u32) -> (u32, u16) {
        let s = self.r.status(slot);
        (g32(&s, 4), g16(&s, 10))
    }
}

const CH1: u16 = 0x8000;

/// A driver sizing its queues and recognising the part reads these before any ring exists.
#[test]
fn the_isp12160_reports_its_own_identity_firmware_and_unmapped_registers() {
    let r = rig12160();
    assert_eq!(r.r(reg::ID_HIGH), 0x1216);
    assert_eq!(r.r(reg::BUS_CFG0), 0, "no revision field on this part");
    r.w(reg::DMA_CMD_CFG, 0xffff);
    r.w(reg::DMA_DATA_CFG, 0xffff);
    assert_eq!((r.r(reg::DMA_CMD_CFG), r.r(reg::DMA_DATA_CFG)), (0, 0), "the flat ISP1040 DMA blocks are not here");

    // Mailbox 6 is seeded because an untouched output would also read zero.
    r.w(r.mb(6), 0x1234);
    r.w(r.mb(0), mbox::ABOUT_FIRMWARE);
    r.w(reg::HOST_CMD, 0x5 << HOST_CMD_SHIFT);
    assert_eq!(r.r(r.mb(0)), mbox::COMPLETE);
    assert_eq!([r.r(r.mb(1)), r.r(r.mb(2))], [10, 4]);
    assert_eq!(r.r(r.mb(6)), 0, "the attribute word is cleared, not echoed");
    r.ack();
    r.w(r.mb(0), mbox::GET_FIRMWARE_STATUS);
    r.w(reg::HOST_CMD, 0x5 << HOST_CMD_SHIFT);
    assert_eq!(r.r(r.mb(0)), mbox::COMPLETE);
    assert_eq!([r.r(r.mb(1)), r.r(r.mb(2))], [0, MAX_REQUESTS]);
    r.ack();
}

/// The driver switches register banks through bus configuration 1 bits 9:8; each of the
/// four selections must show its own words, and no other bit may stick.
#[test]
fn the_isp12160_selects_four_distinct_banks_and_keeps_only_the_bank_field() {
    let r = rig12160();
    let selections = [0x0000, 0x0100, 0x0200, 0x0300];
    for sel in selections {
        r.w(reg::BUS_CFG1, sel);
        r.w(reg::BANK_BASE, 0x1000 + sel);
        r.w(reg::BANK_BASE + 0x20, 0x2000 + sel);
    }
    for sel in selections {
        r.w(reg::BUS_CFG1, sel);
        assert_eq!(r.r(reg::BANK_BASE), 0x1000 + sel, "bank {sel:#x}");
        assert_eq!(r.r(reg::BANK_BASE + 0x20), 0x2000 + sel, "bank {sel:#x}");
    }
    r.w(reg::BUS_CFG1, 0xb2c5);
    assert_eq!(r.r(reg::BUS_CFG1), 0x0200);

    // A soft reset hands a re-initialising driver empty banks, not the previous run's words.
    r.w(reg::BUS_CTRL, CTRL_SOFT_RESET);
    assert_eq!(r.r(reg::BUS_CFG1), 0);
    for sel in selections {
        r.w(reg::BUS_CFG1, sel);
        assert_eq!(r.r(reg::BANK_BASE), 0, "bank {sel:#x} after reset");
    }
}

/// A second command submitted before the driver acknowledged the first result must not
/// run, or it would overwrite the answer the driver has not read.
#[test]
fn the_isp12160_ignores_a_mailbox_submission_while_the_result_is_unacknowledged() {
    let r = rig12160();
    r.w(r.mb(0), mbox::NOP);
    r.w(reg::HOST_CMD, 0x5 << HOST_CMD_SHIFT);
    assert_eq!(r.r(r.mb(0)), mbox::COMPLETE);
    assert_eq!(r.r(reg::SEMAPHORE) & SEMAPHORE_LOCK, SEMAPHORE_LOCK);

    r.w(r.mb(1), 0x9999);
    r.w(r.mb(0), mbox::ABOUT_FIRMWARE);
    r.w(reg::HOST_CMD, 0x5 << HOST_CMD_SHIFT);
    assert_eq!(r.r(r.mb(0)), mbox::ABOUT_FIRMWARE, "the second submission did not run");
    assert_eq!(r.r(r.mb(1)), 0x9999);

    r.ack();
    r.w(reg::HOST_CMD, 0x5 << HOST_CMD_SHIFT);
    assert_eq!(r.r(r.mb(0)), mbox::COMPLETE);
    assert_eq!(r.r(r.mb(1)), 10, "acknowledged, the same submission runs");
    r.ack();
}

/// An empty slot on either channel times out on its own bus; a driver probing channel 1
/// must see an absent target, not a hang or the channel 0 answer.
#[test]
fn a_missing_target_times_out_on_each_channel() {
    let r = rig12160();
    let mut seq = Seq::new(&r);
    for (handle, target) in [(0x31, 5), (0x32, 0x80 | 5)] {
        let s = seq.run(handle, target, &TUR, &[]);
        assert_eq!(g32(&s, 4), handle);
        assert_eq!(g16(&s, 10), IOCB_INCOMPLETE);
        assert_eq!(g16(&s, 12), STATE_GOT_BUS);
        assert_eq!(g16(&s, 14), STATUS_FLAG_TIMEOUT);
    }
}

/// A command refused on channel 1 must not disturb its neighbour on channel 0, and
/// a write must land on the disk of the channel it named.
#[test]
fn a_refused_command_on_channel_1_leaves_channel_0_and_the_media_alone() {
    let r = disks_on_both_channels("refuse");
    let mut seq = Seq::new(&r);
    // Collect each disk's power-on unit attention so the reads below see real answers.
    seq.run(0xa0, 1, &TUR, &[]);
    seq.run(0xa1, 0x81, &TUR, &[]);

    r.mem.put(BUF, &[0x5c; BLOCK_SIZE]);
    let beyond_disk = seq.run(0xb1, 0x81, &cdb10(scsi_cmd::READ_10, 1000, 1), &[(BUF, BLOCK_SIZE as u32)]);
    assert_eq!(g16(&beyond_disk, 8) & 0xff, CHECK_CONDITION as u16);
    assert_eq!(r.mem.get(BUF, BLOCK_SIZE), vec![0x5c; BLOCK_SIZE], "the refused read stored nothing");

    let good = seq.run(0xb0, 1, &cdb10(scsi_cmd::READ_10, 0, 1), &[(BUF + 0x1000, BLOCK_SIZE as u32)]);
    assert_eq!(g16(&good, 10), IOCB_COMPLETE);
    assert_eq!(r.mem.get(BUF + 0x1000, BLOCK_SIZE), vec![0xa0; BLOCK_SIZE], "channel 0 still reads its own disk");

    r.mem.put(BUF + 0x2000, &[0x77; BLOCK_SIZE]);
    seq.run(0xc0, 0x81, &cdb10(scsi_cmd::WRITE_10, 3, 1), &[(BUF + 0x2000, BLOCK_SIZE as u32)]);
    seq.run(0xc1, 0x81, &cdb10(scsi_cmd::READ_10, 3, 1), &[(BUF + 0x3000, BLOCK_SIZE as u32)]);
    seq.run(0xc2, 1, &cdb10(scsi_cmd::READ_10, 3, 1), &[(BUF + 0x4000, BLOCK_SIZE as u32)]);
    assert_eq!(r.mem.get(BUF + 0x3000, BLOCK_SIZE), vec![0x77; BLOCK_SIZE], "channel 1 holds what was written");
    assert_eq!(r.mem.get(BUF + 0x4000, BLOCK_SIZE), vec![0; BLOCK_SIZE], "channel 0's block 3 is untouched");
}

/// Abort device names a channel in the top bit of its selector; the same target number
/// held on the other bus must stay outstanding.
#[test]
fn abort_device_retires_only_the_named_channels_commands() {
    let h = Held::new();
    h.hold(0, 0x41, 1);
    h.hold(1, 0x42, 0x81);
    assert_eq!(h.command(&[mbox::ABORT_DEVICE, CH1 | 0x0100, 0, 0]), mbox::COMPLETE);
    assert_eq!(h.answer(0), (0x42, IOCB_ABORTED));
    assert_eq!(h.answer(1), (0, 0), "channel 0's command is still held");
    // The survivor is still the adapter's to abort.
    assert_eq!(h.command(&[mbox::ABORT_DEVICE, 0x0100, 0, 0]), mbox::COMPLETE);
    assert_eq!(h.answer(1), (0x41, IOCB_ABORTED));
}

/// Abort command names one handle: two commands to the same channel, target and unit differ
/// only by it, and a selector that matches a held handle on the wrong channel is an error.
#[test]
fn abort_command_matches_the_handle_and_the_whole_selector() {
    let h = Held::new();
    h.hold(0, 0x51, 0x81);
    h.hold(1, 0x52, 0x81);
    assert_eq!(h.command(&[mbox::ABORT_COMMAND, CH1 | 0x0100, 0, 0x52]), mbox::COMPLETE);
    assert_eq!(h.answer(0), (0x52, IOCB_ABORTED));
    assert_eq!(h.answer(1), (0, 0), "its neighbour on the same target is untouched");

    assert_eq!(h.command(&[mbox::ABORT_COMMAND, 0x0100, 0, 0x51]), mbox::COMMAND_ERROR,
        "a held handle named on the wrong channel is not a match");
    assert_eq!(h.answer(1), (0, 0));
    assert_eq!(h.command(&[mbox::ABORT_COMMAND, CH1 | 0x0100, 0, 0x51]), mbox::COMPLETE);
    assert_eq!(h.answer(1), (0x51, IOCB_ABORTED));
}

/// A driver clearing a device it already drained must read success, while naming a handle
/// nothing is using must read as an error rather than claim a retirement.
#[test]
fn aborting_nothing_is_complete_but_aborting_an_unknown_handle_is_refused() {
    let h = Held::new();
    assert_eq!(h.command(&[mbox::ABORT_DEVICE, CH1 | 0x0100, 0, 0]), mbox::COMPLETE);
    assert_eq!(h.command(&[mbox::ABORT_COMMAND, CH1 | 0x0100, 0, 0x77]), mbox::COMMAND_ERROR);
    assert_eq!(h.answer(0), (0, 0), "neither posted a completion");
}

/// Target reset reaches one target on one channel; the same number on the other channel and
/// the neighbour on the same channel must not see it, and the reset target keeps serving.
#[test]
fn target_reset_is_scoped_to_one_target_on_one_channel() {
    let h = Held::new();
    h.hold(0, 0x41, 1);
    h.hold(1, 0x42, 0x81);
    assert_eq!(h.command(&[mbox::ABORT_TARGET, CH1 | 0x0100, 0, 0]), mbox::COMPLETE);
    assert_eq!(h.answer(0), (0x42, IOCB_RESET));
    assert_eq!(h.answer(1), (0, 0));
    assert_eq!([h.ch0.load(Ordering::Relaxed), h.ch1.load(Ordering::Relaxed), h.ch1_other.load(Ordering::Relaxed)],
        [0, 1, 0]);

    h.r.put_entry(2, &request(0x43, 0x81, &TUR, &[]));
    h.r.doorbell(3);
    assert_eq!(h.answer(1), (0x43, IOCB_COMPLETE));
}

/// Bus reset takes its channel from mailbox 2; mailbox 1 is the reset delay and here holds a value
/// that is not even a channel. The whole named bus resets, the other one does not.
#[test]
fn bus_reset_takes_its_channel_from_mailbox_two_and_resets_every_target_on_it() {
    let h = Held::new();
    h.hold(0, 0x41, 1);
    h.hold(1, 0x42, 0x81);
    assert_eq!(h.command(&[mbox::BUS_RESET, 3, 1]), mbox::COMPLETE);
    assert_eq!(h.answer(0), (0x42, IOCB_RESET));
    assert_eq!(h.answer(1), (0, 0));
    assert_eq!([h.ch0.load(Ordering::Relaxed), h.ch1.load(Ordering::Relaxed), h.ch1_other.load(Ordering::Relaxed)],
        [0, 1, 1]);
    assert_eq!(h.command(&[mbox::BUS_RESET, 0, 2]), mbox::PARAM_ERROR, "there is no third bus");
}

/// A bus reset leaves a unit attention on the disks of that channel only; the driver's
/// first command afterwards is expected to collect it.
#[test]
fn a_bus_reset_leaves_a_unit_attention_on_the_reset_channel_only() {
    let r = disks_on_both_channels("uareset");
    let mut seq = Seq::new(&r);
    seq.run(0x10, 1, &TUR, &[]);
    seq.run(0x11, 0x81, &TUR, &[]);
    assert_eq!(g16(&seq.run(0x12, 0x81, &TUR, &[]), 8) & 0xff, 0, "attention was collected");

    assert_eq!(r.cmd(&[mbox::BUS_RESET, 0, 1]), mbox::COMPLETE);
    r.isp.wait_idle();
    let after = seq.run(0x13, 0x81, &TUR, &[]);
    assert_eq!(g16(&after, 8) & 0xff, CHECK_CONDITION as u16);
    assert_eq!(after[34] & 0x0f, 0x06, "unit attention");
    let other = seq.run(0x14, 1, &TUR, &[]);
    assert_eq!(g16(&other, 8) & 0xff, 0, "channel 0 was not reset");
    assert_eq!(g16(&seq.run(0x15, 0x81, &TUR, &[]), 8) & 0xff, 0, "reported once");
}
