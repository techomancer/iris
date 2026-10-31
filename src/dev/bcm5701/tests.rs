//! Register-contract tests for the BCM5701, ported from the QEMU model's
//! qtests. The device is built alone, with a controllable clock and an
//! interrupt probe standing in for the rest of the machine.
//!
//! Ported from tests/qtest/bcm570x-test.c in jamesbraid/qemu-sgi-origami.

use super::eeprom::{Eeprom, Fault};
use super::phy::{reg as phy_reg, IDR1_VALUE, IDR2_BCM5701};
use super::state::{reg, Refused, FW_MAGIC, SRAM_FW_MBOX};
use super::*;
use crate::irq::LevelProbe;
use crate::pci::{PciDma, PciDmaStatus};
use std::time::Duration;

const SRAM_WINDOW: u32 = 0x8000;
const MISC_CFG_RESET: u32 = 1;
const MODE_RESET: u32 = 1;
const RXMODE_ENABLE: u32 = 2;
const HCC_ENABLE: u32 = 2;
const BUFMGR_ENABLE: u32 = 2;
const BUFMGR_ATTN: u32 = 4;
const BUFMGR_START: Duration = Duration::from_millis(8);
const PHY_SETTLE: Duration = Duration::from_millis(100);
const MICOMM_BUSY: u32 = 0x2000_0000;
const MICMD_READ: u32 = 0x0800_0000;
const MICMD_WRITE: u32 = 0x0400_0000;
const MICOMM_READ_FIELDS: u32 = 0x0fff_0000;
const PHY_ADDR: u32 = 1;
const MAC: [u8; 6] = [0x52, 0x54, 0x00, 0xab, 0xcd, 0xef];

const EEPROM_READ: u32 = 1 << 31;
const EEPROM_COMPLETE: u32 = 1 << 30;
const EEPROM_RESET: u32 = 1 << 29;
const EEPROM_START: u32 = 1 << 25;

const MEM_SIZE: usize = 0x90_0000;
const TX_RING: u64 = 0x20_0000;
const TX_BUF: u64 = 0x21_0000;
const STATUS: u64 = 0x22_0000;
const STD_RING: u64 = 0x23_0000;
const RET_RING: u64 = 0x24_0000;
const RX_BUF: u64 = 0x25_0000;
const JUMBO_RING: u64 = 0x26_0000;
const JUMBO_BUF: u64 = 0x27_0000;
const STD_STALE_BUF: u64 = 0x28_0000;
/// Past the memory: nothing backs it.
const UNMAPPED: u64 = 0x400_0000;
const TX_RING_LEN: u32 = 4;
const RX_RING_LEN: u32 = 4;
const STD_BUF_SIZE: u32 = 1536;
const JUMBO_BUF_SIZE: u32 = 9216;
const STATUS_BLK_SIZE: usize = 0x50;
const ARRIVAL: Duration = Duration::from_micros(200);
/// Every DMA swap control set: the chip's image goes on the bus as it is.
const MODE_ALL_SWAPS: u32 = 0x36;

/// Bus-master memory with no bridge in front of it.
struct Mem(std::sync::Mutex<Vec<u8>>);

impl PciDma for Mem {
    fn dma_read(&self, addr: u64, buf: &mut [u8]) -> PciDmaStatus {
        let m = self.0.lock().unwrap();
        let a = addr as usize;
        if a.saturating_add(buf.len()) > m.len() {
            buf.fill(0);
            return PciDmaStatus::MasterAbort;
        }
        buf.copy_from_slice(&m[a..a + buf.len()]);
        PciDmaStatus::Ok
    }
    fn dma_write(&self, addr: u64, data: &[u8]) -> PciDmaStatus {
        let mut m = self.0.lock().unwrap();
        let a = addr as usize;
        if a.saturating_add(data.len()) > m.len() {
            return PciDmaStatus::MasterAbort;
        }
        m[a..a + data.len()].copy_from_slice(data);
        PciDmaStatus::Ok
    }
}

struct Rig {
    dev: Arc<Bcm5701>,
    irq: Arc<LevelProbe>,
    mem: Arc<Mem>,
    /// Frames the guest sent, and a way to hand frames to the guest.
    sent: rtrb::Consumer<Vec<u8>>,
    inject: rtrb::Producer<Vec<u8>>,
    ring_slots: usize,
}

impl Rig {
    fn new() -> Self {
        Self::with(Bcm5701Config::new(MAC))
    }

    fn with(config: Bcm5701Config) -> Self {
        let dev = Bcm5701::new(config).unwrap();
        *dev.frozen.lock() = Some(Instant::now());
        let (irq, line) = LevelProbe::line();
        let mem = Arc::new(Mem(std::sync::Mutex::new(vec![0; MEM_SIZE])));
        dev.state().connect_irq(line);
        dev.state().connect_dma(mem.clone());
        dev.attached();
        let (link, sent, inject) = crate::net::Link::loopback();
        dev.attach(link);
        dev.reset();
        let ring_slots = inject.slots();
        Self { dev, irq, mem, sent, inject, ring_slots }
    }

    /// Advance the clock and run whatever came due.
    fn step(&self, by: Duration) {
        {
            let mut t = self.dev.frozen.lock();
            *t = Some(t.unwrap() + by);
        }
        self.dev.enter(self.dev.now(), |_| ());
    }

    fn mem_write(&self, addr: u64, data: &[u8]) {
        let mut m = self.mem.0.lock().unwrap();
        m[addr as usize..addr as usize + data.len()].copy_from_slice(data);
    }
    fn mem_read(&self, addr: u64, len: usize) -> Vec<u8> {
        self.mem.0.lock().unwrap()[addr as usize..addr as usize + len].to_vec()
    }
    fn mem_fill(&self, addr: u64, len: usize, byte: u8) {
        self.mem.0.lock().unwrap()[addr as usize..addr as usize + len].fill(byte);
    }
    fn status_block(&self) -> Vec<u8> {
        self.mem_read(STATUS, STATUS_BLK_SIZE)
    }
    fn status_tag(&self) -> u32 {
        be32(&self.status_block(), 4)
    }
    fn status_tx_cons(&self) -> u16 {
        get16dma(&self.status_block(), 18)
    }
    fn status_rx_prod(&self) -> u16 {
        get16dma(&self.status_block(), 16)
    }
    /// The producer-ring consumers, the high and low halves of word 0x08.
    fn status_std_cons(&self) -> u16 {
        let sb = self.status_block();
        u16::from_be_bytes([sb[8], sb[9]])
    }
    fn status_jumbo_cons(&self) -> u16 {
        let sb = self.status_block();
        u16::from_be_bytes([sb[10], sb[11]])
    }
    fn line_up(&self) -> bool {
        self.cfg_read(0x70 / 4) & 2 == 0
    }
    fn nic_no_more_rxbds(&self) -> u32 {
        self.readl(SRAM_WINDOW + 0x898)
    }
    /// The frames the guest has sent so far.
    fn sent_frames(&mut self) -> Vec<Vec<u8>> {
        std::iter::from_fn(|| self.sent.pop().ok()).collect()
    }
    /// Frames handed to the guest that it has not yet taken.
    fn waiting(&self) -> usize {
        self.ring_slots - self.inject.slots()
    }
    /// Offer a frame to the receive path, as the backend would.
    fn deliver(&mut self, frame: &[u8]) {
        self.inject.push(frame.to_vec()).unwrap();
        self.dev.poll();
    }

    fn readl(&self, off: u32) -> u32 {
        self.dev.bar_read(0, off, 4)
    }
    fn readb(&self, off: u32) -> u32 {
        self.dev.bar_read(0, off, 1)
    }
    fn writel(&self, off: u32, val: u32) {
        self.dev.bar_write(0, off, val, 4);
    }
    fn cfg_read(&self, reg: usize) -> u32 {
        self.dev.config_dword(reg)
    }
    /// A configuration write as the bus makes it: `width` bytes at byte
    /// `offset`.
    fn cfg_write_sized(&self, offset: usize, width: usize, val: u32) {
        let (reg, shift) = (offset / 4, (offset & 3) * 8);
        if width == 4 && self.dev.config_write(reg, val) {
            return;
        }
        let config = self.dev.state().config();
        let old = config.read(reg);
        config.write_masked(reg, val << shift, (u32::MAX >> (32 - 8 * width)) << shift);
        let new = config.read(reg);
        if new != old {
            self.dev.config_written(reg, old, new);
        }
    }
    fn cfg_write(&self, reg: usize, val: u32) {
        self.cfg_write_sized(reg * 4, 4, val);
    }
    /// A configuration read of `width` bytes at byte `offset`.
    fn cfg_read_sized(&self, offset: usize, width: usize) -> u32 {
        (self.cfg_read(offset / 4) >> ((offset & 3) * 8)) & (u32::MAX >> (32 - 8 * width))
    }

    /// The indirect register window, config 0x78 and 0x80.
    fn ind_reg_write(&self, off: u32, val: u32) {
        self.cfg_write(0x78 / 4, off);
        self.cfg_write(0x80 / 4, val);
    }
    fn ind_reg_read(&self, off: u32) -> u32 {
        self.cfg_write(0x78 / 4, off);
        self.cfg_read(0x80 / 4)
    }
    /// The indirect memory window, config 0x7c and 0x84.
    fn ind_mem_write(&self, off: u32, val: u32) {
        self.cfg_write(0x7c / 4, off);
        self.cfg_write(0x84 / 4, val);
    }
    fn ind_mem_read(&self, off: u32) -> u32 {
        self.cfg_write(0x7c / 4, off);
        self.cfg_read(0x84 / 4)
    }

    fn mii_read(&self, phy: u32, register: u8) -> u16 {
        self.writel(reg::MI_COMM, MICOMM_BUSY | MICMD_READ | phy << 21 | u32::from(register) << 16);
        self.readl(reg::MI_COMM) as u16
    }
    fn mii_write(&self, phy: u32, register: u8, val: u16) {
        self.writel(
            reg::MI_COMM,
            MICOMM_BUSY | MICMD_WRITE | phy << 21 | u32::from(register) << 16 | u32::from(val),
        );
    }
    fn sram_word(&self, off: u32) -> u32 {
        self.dev.st.lock().sram_read(off)
    }
    fn split_addr(&self, hi_reg: u32, lo_reg: u32, addr: u64) {
        self.writel(hi_reg, (addr >> 32) as u32);
        self.writel(lo_reg, addr as u32);
    }
    fn refusals(&self, what: Refused) -> u64 {
        self.dev.refused().iter().find(|(r, _)| *r == what).map_or(0, |(_, n)| *n)
    }
}

fn temp_file(name: &str, contents: &[u8]) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("iris-bcm5701-{}-{name}", std::process::id()));
    std::fs::write(&path, contents).unwrap();
    path
}

/// Dword and sub-dword reads of the identity, so a hidden swap in one half
/// cannot hide behind the other. Ported from test_pci_four_tuple.
#[test]
fn the_function_presents_the_sgi_board_identity() {
    let rig = Rig::new();
    assert_eq!(rig.cfg_read(0), 0x1645_14e4);
    assert_eq!(rig.cfg_read(0x2c / 4), 0x8010_10a9);
    assert_eq!(rig.cfg_read(0x08 / 4) >> 8, 0x02_0000);
    assert_eq!(rig.cfg_read(0x08 / 4) & 0xff, 0x15, "the B5 stepping's PCI revision");
    assert_eq!(rig.dev.state().config().interrupt_pin(), PCI_INTA);
}

/// The capability list is PCI-X at 0x40, power management at 0x48, VPD at
/// 0x50 and MSI at 0x58, in that order, and the status register says there
/// is one. A walk from the pointer visits exactly those.
#[test]
fn the_capability_list_is_pcix_pm_vpd_and_msi() {
    let rig = Rig::new();
    assert_ne!(rig.cfg_read_sized(0x06, 2) & 0x10, 0, "status: capability list");
    let mut walk = Vec::new();
    let mut at = rig.cfg_read_sized(0x34, 1) as usize;
    while at != 0 {
        walk.push((at, rig.cfg_read_sized(at, 1)));
        at = rig.cfg_read_sized(at + 1, 1) as usize;
    }
    assert_eq!(walk, [(0x40, 0x07), (0x48, 0x01), (0x50, 0x03), (0x58, 0x05)]);
    assert_eq!(rig.cfg_read_sized(0x42, 2), 0x0002, "PCI-X command: relaxed ordering");
    assert_eq!(rig.cfg_read(0x44 / 4), 0x0003_fff9, "PCI-X status at reset");
    assert_eq!(rig.cfg_read_sized(0x4a, 2), 0x4002, "PM: PME from D3hot, version 1.1");
    assert_eq!(rig.cfg_read_sized(0x5a, 2), 0x0086, "MSI: 64-bit, eight messages, disabled");
    // The driver saves and restores the PCI-X command halfword around a reset.
    rig.cfg_write_sized(0x42, 2, 0x0023);
    assert_eq!(rig.cfg_read_sized(0x42, 2), 0x0023);
    rig.cfg_write_sized(0x40, 2, 0);
    assert_eq!(rig.cfg_read_sized(0x40, 2), 0x4807, "identity and link are read-only");
    rig.cfg_write_sized(0x5a, 2, 0x0001);
    assert_eq!(rig.cfg_read_sized(0x5a, 2) & 1, 0, "MSI cannot be enabled");
}

/// A VPD read sets the flag at once and puts the four bytes at the address,
/// least significant first, in the data register: from the EEPROM's VPD
/// area at 0x100 when one is fitted, all ones without one. IRIX's driver
/// makes these reads through the BAR's alias of the header with halfword
/// accesses at 0x52, and a flag that never set cost it 6.4 s per attach.
#[test]
fn a_vpd_read_completes_at_once_through_config_or_the_bar_alias() {
    let bare = Rig::new();
    bare.cfg_write_sized(0x52, 2, 0x0010);
    assert_eq!(bare.cfg_read_sized(0x52, 2), 0x8010, "flag set, address kept");
    assert_eq!(bare.cfg_read(0x54 / 4), 0xffff_ffff, "no NVRAM fitted");

    let mut image = vec![0u8; 0x200];
    image[0x100..0x108].copy_from_slice(&[0x82, 0x05, 0x00, b'T', b'e', b's', b't', b'!']);
    let rig = eeprom_rig(&image, 3);
    rig.cfg_write_sized(0x52, 2, 0x0004);
    assert_eq!(rig.cfg_read(0x54 / 4), u32::from_le_bytes(*b"est!"));
    // The same through BAR0's alias, as halfword and word accesses.
    rig.dev.bar_write(0, 0x52, 0x0000, 2);
    assert_eq!(rig.dev.bar_read(0, 0x52, 2), 0x8000);
    assert_eq!(rig.readl(0x54), u32::from_le_bytes([0x82, 0x05, 0x00, b'T']));
    rig.writel(0x50, 0x0004_0000);
    assert_eq!(rig.readl(0x50), 0x8004_5803);
    // A write request is refused and completes by clearing the flag.
    rig.cfg_write_sized(0x52, 2, 0x8008);
    assert_eq!(rig.cfg_read_sized(0x52, 2), 0x0008);
    assert_eq!(rig.refusals(Refused::VpdWrite), 1);
    assert_eq!(rig.dev.eeprom_bytes().unwrap(), image, "storage untouched");
}

/// The chip id in MISC_CTL decodes to the BCM5701, ASIC revision 0, and is
/// read-only while the low half sticks. Ported from test_misc_ctl_asic_revision.
#[test]
fn misc_ctl_presents_the_chip_id() {
    let rig = Rig::new();
    let misc = rig.cfg_read(0x68 / 4);
    assert_eq!(misc >> 16, 0x0105);
    assert_eq!((misc >> 16) >> 12, 0);
    rig.cfg_write(0x68 / 4, 0xffff_0004);
    assert_eq!(rig.cfg_read(0x68 / 4), 0x0105_0004);
}

/// Ported from test_indirect_register_window: the window aliases the file.
#[test]
fn the_indirect_register_window_aliases_the_register_file() {
    let rig = Rig::new();
    rig.ind_reg_write(0x0410, 0x1234_5678);
    assert_eq!(rig.readl(0x0410), 0x1234_5678);
    assert_eq!(rig.ind_reg_read(0x0410), 0x1234_5678);
    assert_eq!(rig.cfg_read(0x78 / 4), 0x0410);
}

/// Only whole dwords reach the windows. A narrower configuration read sees
/// the header's own bytes, which stay zero because nothing writes them, as
/// in the model this is ported from, where the window state sits outside
/// the header and the header bytes are read-only.
#[test]
fn narrow_configuration_accesses_bypass_the_indirect_windows() {
    let rig = Rig::new();
    rig.ind_reg_write(0x0410, 0x1234_5678);
    rig.ind_mem_write(0x0c14, 0xdead_beef);
    let config = rig.dev.state().config();
    for offset in 0x78..0x88 {
        assert_eq!(config.read(offset / 4) >> ((offset & 3) * 8) & 0xff, 0, "byte at {offset:#x}");
    }
    for offset in (0x78..0x88).step_by(2) {
        assert_eq!(config.read(offset / 4) >> ((offset & 3) * 8) & 0xffff, 0, "halfword at {offset:#x}");
    }
    rig.cfg_write_sized(0x80, 1, 0xff);
    rig.cfg_write_sized(0x7e, 2, 0xffff);
    assert_eq!(config.read(0x80 / 4) & 0xff, 0);
    assert_eq!(config.read(0x7c / 4) >> 16, 0);
    assert_eq!(rig.cfg_read_sized(0x78, 4), 0x0410);
    assert_eq!(rig.cfg_read_sized(0x80, 4), 0x1234_5678);
    assert_eq!(rig.cfg_read_sized(0x7c, 4), 0x0c14);
    assert_eq!(rig.cfg_read_sized(0x84, 4), 0xdead_beef);
}

/// Ported from test_indirect_memory_window: SRAM through config and through
/// the BAR window at 0x8000.
#[test]
fn the_indirect_memory_window_aliases_sram() {
    let rig = Rig::new();
    rig.ind_mem_write(0x0c14, 0xdead_beef);
    assert_eq!(rig.ind_mem_read(0x0c14), 0xdead_beef);
    assert_eq!(rig.readl(SRAM_WINDOW + (0x0c14 & 0x7fff)), 0xdead_beef);
    // The base register is masked to a 32 KiB boundary for the BAR window,
    // and the 32 KiB SRAM has one position.
    rig.cfg_write(0x7c / 4, 0x0c14);
    assert_eq!(rig.readl(SRAM_WINDOW), rig.sram_word(0));
}

/// The magic written before the reset (the bge order) reads back as the
/// complement, and the reset self-clears its trigger and re-arms the mailbox.
/// Ported from test_reset_handshake_via_bar.
#[test]
fn the_reset_handshake_completes_when_the_magic_precedes_the_reset() {
    let rig = Rig::new();
    rig.ind_mem_write(SRAM_FW_MBOX, FW_MAGIC);
    assert_eq!(rig.ind_mem_read(SRAM_FW_MBOX), !FW_MAGIC);
    rig.writel(reg::MISC_CFG, MISC_CFG_RESET);
    assert_eq!(rig.readl(reg::MISC_CFG) & MISC_CFG_RESET, 0);
    assert_eq!(rig.ind_mem_read(SRAM_FW_MBOX), !FW_MAGIC);
}

/// The IRIX order: reset first, then the magic. Ported from
/// test_reset_handshake_magic_after_reset.
#[test]
fn the_reset_handshake_completes_when_the_magic_follows_the_reset() {
    let rig = Rig::new();
    rig.writel(reg::MISC_CFG, MISC_CFG_RESET);
    rig.ind_mem_write(SRAM_FW_MBOX, FW_MAGIC);
    assert_eq!(rig.ind_mem_read(SRAM_FW_MBOX), 0xb49a_89ab);
}

/// Ported from test_reset_handshake_via_indirect: the 5700/5701 reset through
/// the configuration window alone.
#[test]
fn the_reset_handshake_completes_through_the_indirect_windows() {
    let rig = Rig::new();
    rig.ind_mem_write(SRAM_FW_MBOX, FW_MAGIC);
    rig.ind_reg_write(reg::MISC_CFG, MISC_CFG_RESET);
    assert_eq!(rig.ind_mem_read(SRAM_FW_MBOX), 0xb49a_89ab);
}

/// Ported from test_firmware_progress_register: the all-ones "no progress
/// yet" sentinel survives a reset.
#[test]
fn the_firmware_progress_register_holds_its_sentinel() {
    let rig = Rig::new();
    assert_eq!(rig.readl(reg::FW_PROGRESS), 0xffff_ffff);
    assert_eq!(rig.ind_reg_read(reg::FW_PROGRESS), 0xffff_ffff);
    rig.writel(reg::MISC_CFG, MISC_CFG_RESET);
    assert_eq!(rig.readl(reg::FW_PROGRESS), 0xffff_ffff);
}

/// A global reset keeps the PCI header, the chip id among it, and the
/// window bases, and restores the bootcode's station address. Ported from
/// test_mac_address_registers.
#[test]
fn a_global_reset_restores_the_bootcode_state_only() {
    let rig = Rig::new();
    rig.cfg_write(0x68 / 4, 0x0000_0004);
    rig.cfg_write(0x7c / 4, 0x1234);
    rig.writel(reg::MAC_ADDR_0_HIGH, 0xa1a2);
    rig.writel(reg::MISC_CFG, MISC_CFG_RESET);
    assert_eq!(rig.cfg_read(0x68 / 4), 0x0105_0004);
    assert_eq!(rig.cfg_read(0x7c / 4), 0x1234);
    assert_eq!(rig.readl(reg::MAC_ADDR_0_HIGH), 0x5254);
}

/// Ported from test_self_clearing_reset_bits.
#[test]
fn mode_register_reset_bits_self_clear_and_the_rest_persists() {
    let rig = Rig::new();
    rig.writel(reg::MAC_MODE, MODE_RESET | 0x4);
    assert_eq!(rig.readl(reg::MAC_MODE) & MODE_RESET, 0);
    assert_eq!(rig.readl(reg::MAC_MODE) & 4, 4);

    rig.writel(reg::HCC_MODE, MODE_RESET);
    assert_eq!(rig.readl(reg::HCC_MODE) & MODE_RESET, 0);

    rig.writel(reg::RX_MODE, MODE_RESET | RXMODE_ENABLE);
    assert_eq!(rig.readl(reg::RX_MODE) & MODE_RESET, 0);
    assert_eq!(rig.readl(reg::RX_MODE) & RXMODE_ENABLE, RXMODE_ENABLE);
    // The IRIX sequence: reset on its own leaves the receiver halted.
    rig.writel(reg::RX_MODE, MODE_RESET);
    assert_eq!(rig.readl(reg::RX_MODE), 0);
}

/// The driver's idle sequence writes ENABLE, then 0, and polls for the whole
/// register to read zero. Ported from test_hcc_mode_idle_poll and
/// test_hcc_mode_clears_to_zero.
#[test]
fn host_coalescing_mode_stores_and_clears() {
    let rig = Rig::new();
    rig.writel(reg::HCC_MODE, HCC_ENABLE);
    rig.writel(reg::RX_MODE, RXMODE_ENABLE);
    assert_eq!(rig.readl(reg::HCC_MODE) & HCC_ENABLE, HCC_ENABLE);
    rig.writel(reg::HCC_MODE, 0);
    assert_eq!(rig.readl(reg::HCC_MODE) & HCC_ENABLE, 0);

    rig.writel(reg::HCC_MODE, 0x13e);
    rig.writel(reg::HCC_MODE, 0);
    assert_eq!(rig.readl(reg::HCC_MODE), 0);
}

/// ENABLE reads back only after the buffer manager's startup, by either
/// access path. Ported from test_bufmgr_registers.
#[test]
fn the_buffer_manager_enable_reads_back_after_its_startup() {
    for indirect in [false, true] {
        let rig = Rig::new();
        let read = |off| if indirect { rig.ind_reg_read(off) } else { rig.readl(off) };
        let write = |off, val| if indirect { rig.ind_reg_write(off, val) } else { rig.writel(off, val) };
        let on = BUFMGR_ENABLE | BUFMGR_ATTN;

        write(reg::BUFMGR_MODE, BUFMGR_ATTN);
        write(reg::BUFMGR_MODE, on);
        assert_eq!(read(reg::BUFMGR_MODE), BUFMGR_ATTN);
        rig.step(Duration::from_millis(7));
        assert_eq!(read(reg::BUFMGR_MODE), BUFMGR_ATTN);
        // A repeated request does not restart the interval.
        write(reg::BUFMGR_MODE, on);
        rig.step(BUFMGR_START - Duration::from_millis(7) - Duration::from_nanos(1));
        assert_eq!(read(reg::BUFMGR_MODE), BUFMGR_ATTN);
        rig.step(Duration::from_nanos(1));
        assert_eq!(read(reg::BUFMGR_MODE), on);

        // Reset cancels startup; the new request starts a fresh interval.
        write(reg::BUFMGR_MODE, MODE_RESET | on);
        assert_eq!(read(reg::BUFMGR_MODE), BUFMGR_ATTN);
        rig.step(BUFMGR_START);
        assert_eq!(read(reg::BUFMGR_MODE), on);

        // Halting is immediate and cancels a pending start.
        write(reg::BUFMGR_MODE, BUFMGR_ATTN);
        assert_eq!(read(reg::BUFMGR_MODE), BUFMGR_ATTN);
        write(reg::BUFMGR_MODE, on);
        write(reg::BUFMGR_MODE, BUFMGR_ATTN);
        rig.step(BUFMGR_START);
        assert_eq!(read(reg::BUFMGR_MODE), BUFMGR_ATTN);

        // The status register is read-only, and the refusal is counted.
        write(reg::BUFMGR_STATUS, on);
        assert_eq!(read(reg::BUFMGR_STATUS), 0);
        assert!(rig.refusals(Refused::BufmgrStatusWrite) >= 1);

        // A global reset cancels a pending startup.
        rig.writel(reg::BUFMGR_MODE, on);
        rig.step(BUFMGR_START / 2);
        rig.writel(reg::MISC_CFG, MISC_CFG_RESET);
        rig.step(BUFMGR_START);
        assert_eq!(rig.readl(reg::BUFMGR_MODE), 0);
    }
}

/// MAC_STATUS bits clear on a write of one, and the write is never readable
/// back as state. Ported from test_mac_status_write_one_clears.
#[test]
fn mac_status_clears_on_write_one() {
    let rig = Rig::new();
    assert_eq!(rig.readl(reg::MAC_STATUS), 0);
    rig.writel(reg::MAC_STATUS, 0x50);
    assert_eq!(rig.readl(reg::MAC_STATUS), 0);
    rig.writel(reg::MAC_STATUS, 0xffff_ffff);
    assert_eq!(rig.readl(reg::MAC_STATUS), 0);
}

/// MI_COMM completes at once; a successful read keeps the command and address
/// fields, replaces the data and clears Start and ReadFailed, whatever
/// reserved bits the request carried; a write without Start only stores; an
/// undefined opcode touches no PHY register and is counted. Ported from
/// test_mi_busy_clears.
#[test]
fn mi_comm_completes_reads_and_writes_by_either_path() {
    for indirect in [false, true] {
        let rig = Rig::new();
        let read = |off| if indirect { rig.ind_reg_read(off) } else { rig.readl(off) };
        let write = |off, val| if indirect { rig.ind_reg_write(off, val) } else { rig.writel(off, val) };
        let anar = |rig: &Rig| {
            let command = MICOMM_BUSY | MICMD_READ | PHY_ADDR << 21 | 4 << 16;
            if indirect {
                rig.ind_reg_write(reg::MI_COMM, command);
                rig.ind_reg_read(reg::MI_COMM)
            } else {
                rig.writel(reg::MI_COMM, command);
                rig.readl(reg::MI_COMM)
            }
        };

        // The discriminating full-word case: a PHY ID read keeps its fields.
        for (command, result) in [(0x2822_ffff, 0x0822_0020), (0xf822_ffff, 0x0822_0020)] {
            write(reg::MI_COMM, command);
            assert_eq!(read(reg::MI_COMM), result);
        }

        let comm = anar(&rig);
        assert_eq!(comm & MICOMM_BUSY, 0);
        let baseline = comm & 0xffff;
        assert_eq!(comm, (MICOMM_BUSY | MICMD_READ | PHY_ADDR << 21 | 4 << 16) & MICOMM_READ_FIELDS | baseline);

        // The 10baseT half-duplex advertisement toggles.
        let value = baseline ^ 0x20;
        let write_command = MICMD_WRITE | PHY_ADDR << 21 | 4 << 16 | value;
        write(reg::MI_COMM, write_command);
        assert_eq!(read(reg::MI_COMM), write_command);
        write(reg::MI_COMM, write_command);
        assert_eq!(read(reg::MI_COMM), write_command);

        // Without Start nothing reaches the PHY.
        assert_eq!(anar(&rig) & 0xffff, baseline);

        let id_command = MICMD_READ | PHY_ADDR << 21 | 2 << 16;
        write(reg::MI_COMM, id_command);
        assert_eq!(read(reg::MI_COMM), id_command);
        write(reg::MI_COMM, id_command | MICOMM_BUSY);
        assert_eq!(read(reg::MI_COMM), (id_command & MICOMM_READ_FIELDS) | u32::from(IDR1_VALUE));

        write(reg::MI_COMM, write_command | MICOMM_BUSY);
        assert_eq!(read(reg::MI_COMM) & MICOMM_BUSY, 0);
        assert_eq!(anar(&rig) & 0xffff, value);

        // Undefined opcodes complete without a PHY access and are counted.
        for opcode in [0, MICMD_READ | MICMD_WRITE] {
            let command = MICOMM_BUSY | opcode | PHY_ADDR << 21 | 4 << 16 | 0x5a5a;
            write(reg::MI_COMM, command);
            assert_eq!(read(reg::MI_COMM), command & !MICOMM_BUSY);
            assert_eq!(anar(&rig) & 0xffff, value);
        }
        assert_eq!(rig.refusals(Refused::MiOpcode(0)), 1);
        assert_eq!(rig.refusals(Refused::MiOpcode(3)), 1);
    }
}

/// Ported from test_phy_readback: identity, a link that is down while
/// negotiating, a settle that must elapse, and MI_STS following the carrier.
#[test]
fn the_phy_negotiates_and_the_link_follows_the_settle() {
    let rig = Rig::new();
    assert_eq!(rig.mii_read(PHY_ADDR, phy_reg::IDR1), IDR1_VALUE);
    assert_eq!(rig.mii_read(PHY_ADDR, phy_reg::IDR2), IDR2_BCM5701);
    assert_eq!(rig.mii_read(PHY_ADDR, phy_reg::BMSR) & 4, 0);
    assert_eq!(rig.readl(reg::MI_STS) & 1, 0);

    // Restarting autonegotiation negotiates again.
    rig.mii_write(PHY_ADDR, phy_reg::BMCR, 0x1000 | 0x0200);
    assert_eq!(rig.mii_read(PHY_ADDR, phy_reg::BMSR) & 4, 0);

    rig.step(PHY_SETTLE + Duration::from_micros(1));
    assert_eq!(rig.mii_read(PHY_ADDR, phy_reg::BMSR) & 4, 4);
    assert_eq!(rig.mii_read(PHY_ADDR, phy_reg::BMSR) & 0x20, 0x20);
    assert_eq!(rig.readl(reg::MI_STS) & 1, 1);

    // A cable pulled drops the carrier without touching negotiation.
    rig.dev.set_backend_link(false);
    assert_eq!(rig.readl(reg::MI_STS) & 1, 0);
    rig.dev.set_backend_link(true);
    assert_eq!(rig.readl(reg::MI_STS) & 1, 1);
}

/// Ported from test_phy_aux_status_link_up: never all-ones, ACOMP and the
/// 1000baseT full-duplex resolution once settled.
#[test]
fn the_phy_auxiliary_status_gates_the_link_up_path() {
    let rig = Rig::new();
    let aux = rig.mii_read(PHY_ADDR, phy_reg::AUXSTS);
    assert_eq!(aux & 0x8000, 0);
    assert_ne!(aux, 0xffff);
    rig.mii_write(PHY_ADDR, phy_reg::BMCR, 0x1000 | 0x0200);
    rig.step(PHY_SETTLE + Duration::from_micros(1));
    let aux = rig.mii_read(PHY_ADDR, phy_reg::AUXSTS);
    assert_eq!(aux & 0x8000, 0x8000);
    assert_eq!(aux & 0x0700, 0x0700);
    assert_ne!(aux, 0xffff);
    // The partner advertises what this end does, and gigabit full duplex.
    assert_eq!(rig.mii_read(PHY_ADDR, phy_reg::GTSR) & 0x0800, 0x0800);
    assert_eq!(rig.mii_read(PHY_ADDR, phy_reg::ANLPAR), rig.mii_read(PHY_ADDR, phy_reg::ANAR) | 0x4000);
}

/// A PHY register the model lacks reads zero, never the all-ones an absent
/// PHY returns, and the access is counted.
#[test]
fn unimplemented_phy_registers_read_zero_and_are_counted() {
    let rig = Rig::new();
    assert_eq!(rig.mii_read(PHY_ADDR, 0x1c), 0);
    rig.mii_write(PHY_ADDR, 0x1c, 0x1234);
    rig.mii_write(PHY_ADDR, phy_reg::IDR1, 0);
    let refused = rig.dev.refused();
    assert!(refused.contains(&(Refused::Phy(phy::Refusal::ReadUnimplemented(0x1c)), 1)));
    assert!(refused.contains(&(Refused::Phy(phy::Refusal::WriteUnimplemented(0x1c)), 1)));
    assert!(refused.contains(&(Refused::Phy(phy::Refusal::WriteReadOnly(phy_reg::IDR1)), 1)));
}

/// The station address lands in the SRAM mailbox the driver reads first.
/// Ported from test_mac_address_mailbox.
#[test]
fn the_station_address_is_in_the_sram_mailbox() {
    let rig = Rig::new();
    assert_eq!(rig.ind_mem_read(0x0c14), 0x484b_5254);
    assert_eq!(rig.ind_mem_read(0x0c18), 0x00ab_cdef);
}

/// The station address is also in filter slot 0, readable a byte at a time
/// from that pair alone. Ported from test_mac_address_registers.
#[test]
fn the_station_address_is_in_filter_slot_zero_with_narrow_reads_there_only() {
    let rig = Rig::new();
    assert_eq!(rig.readl(reg::MAC_ADDR_0_HIGH), 0x0000_5254);
    assert_eq!(rig.readl(reg::MAC_ADDR_0_LOW), 0x00ab_cdef);
    let bytes: Vec<u32> = (0..8).map(|i| rig.readb(reg::MAC_ADDR_0_HIGH + i)).collect();
    assert_eq!(bytes, [0x54, 0x52, 0, 0, 0xef, 0xcd, 0xab, 0]);

    rig.writel(reg::MAC_ADDR_0_HIGH - 4, 0x5566_7788);
    rig.writel(reg::MAC_ADDR_0_LOW + 4, 0x1122_3344);
    // Adjacent byte reads and halfword accesses are rejected and counted.
    assert_eq!(rig.readb(reg::MAC_ADDR_0_HIGH - 1), 0);
    assert_eq!(rig.readb(reg::MAC_ADDR_0_LOW + 4), 0);
    assert_eq!(rig.dev.bar_read(0, reg::MAC_ADDR_0_HIGH, 2), 0);
    // Narrow writes are not part of the byte-read contract.
    rig.dev.bar_write(0, reg::MAC_ADDR_0_HIGH, 0, 1);
    rig.dev.bar_write(0, reg::MAC_ADDR_0_LOW, 0, 2);
    assert_eq!(rig.readl(reg::MAC_ADDR_0_HIGH), 0x5254);
    assert_eq!(rig.readl(reg::MAC_ADDR_0_LOW), 0x00ab_cdef);
    assert_eq!(rig.readl(reg::MAC_ADDR_0_HIGH - 4), 0x5566_7788);
    assert_eq!(rig.readl(reg::MAC_ADDR_0_LOW + 4), 0x1122_3344);
    assert!(rig.refusals(Refused::BarAccess { offset: reg::MAC_ADDR_0_HIGH, width: 1, write: true }) >= 1);
    assert!(rig.refusals(Refused::BarAccess { offset: reg::MAC_ADDR_0_HIGH, width: 2, write: false }) >= 1);
}

/// A write to a register with no modelled behaviour reads back and is counted.
#[test]
fn an_unmodelled_register_stores_and_is_counted() {
    let rig = Rig::new();
    rig.writel(0x1000, 0xcafe_f00d);
    assert_eq!(rig.readl(0x1000), 0xcafe_f00d);
    assert_eq!(rig.refusals(Refused::UnmodelledWrite(0x1000)), 1);
}

/// The first 256 bytes of BAR0 are the configuration registers again; a
/// MISC_CTL acknowledge there lands on the same state.
#[test]
fn bar0_aliases_the_configuration_registers() {
    let rig = Rig::new();
    assert_eq!(rig.readl(0x68), 0x0105_0000);
    rig.writel(0x68, 0x0000_0008);
    assert_eq!(rig.cfg_read(0x68 / 4), 0x0105_0008);
    rig.writel(0x7c, 0x0c14);
    assert_eq!(rig.readl(0x84), rig.ind_mem_read(0x0c14));
}

/// PCISTATE reports conventional PCI and a quiet interrupt line, is chip
/// driven, and CLEAR_INTA self-clears. Ported from the interrupt cases of
/// test_interrupt_mailbox that need no status block.
#[test]
fn pcistate_and_the_interrupt_mailbox_start_quiet() {
    let rig = Rig::new();
    assert_eq!(rig.cfg_read(0x70 / 4), 0x16);
    rig.cfg_write(0x70 / 4, 0);
    assert_eq!(rig.cfg_read(0x70 / 4), 0x16);
    rig.writel(reg::MBX_IRQ0_LO, 0);
    rig.cfg_write(0x68 / 4, 1);
    assert_eq!(rig.cfg_read(0x68 / 4) & 1, 0);
    assert!(!rig.irq.level());
}

/// A slot that drives REQ64# makes the chip report a 64-bit bus, at attach
/// and after each reset.
#[test]
fn pcistate_reports_a_64_bit_bus_in_a_req64_slot() {
    let rig = Rig::with(Bcm5701Config { req64: true, ..Bcm5701Config::new(MAC) });
    assert_eq!(rig.cfg_read(0x70 / 4), 0x06);
    rig.dev.state().config().reset();
    PciFunction::reset(&*rig.dev);
    assert_eq!(rig.cfg_read(0x70 / 4), 0x06);
    let narrow = Rig::new();
    assert_eq!(narrow.cfg_read(0x70 / 4), 0x16);
}

fn eeprom_rig(image: &[u8], device_id: u8) -> Rig {
    Rig::with(Bcm5701Config { eeprom: Some(Eeprom::from_image(device_id, image.to_vec()).unwrap()), ..Bcm5701Config::new(MAC) })
}

/// Ported from test_eeprom_bar_read_write: Start cannot replace an
/// unacknowledged completion, acknowledge-and-start works in one write,
/// transfers stay word aligned, writes reach only their word, and Reset wins
/// over Start.
#[test]
fn the_eeprom_controller_reads_and_writes_aligned_words() {
    let path = temp_file("bar-rw", &[0x66, 0x99, 0x55, 0xaa, 0x12, 0x34, 0x56, 0x78]);
    let rig = Rig::with(Bcm5701Config { eeprom: Some(Eeprom::from_file(3, path.clone()).unwrap()), ..Bcm5701Config::new(MAC) });
    let done = |rig: &Rig| rig.readl(reg::EEPROM_ADDR) & (EEPROM_COMPLETE | EEPROM_START);

    rig.writel(reg::EEPROM_ADDR, EEPROM_READ | EEPROM_START | 3 << 26);
    assert_eq!(rig.readl(reg::EEPROM_DATA), 0xaa55_9966);
    assert_eq!(done(&rig), EEPROM_COMPLETE);

    // Start cannot replace an unacknowledged completion.
    rig.ind_reg_write(reg::EEPROM_ADDR, EEPROM_READ | EEPROM_START | 3 << 26 | 4);
    assert_eq!(rig.readl(reg::EEPROM_DATA), 0xaa55_9966);
    assert_eq!(rig.readl(reg::EEPROM_ADDR) & EEPROM_COMPLETE, EEPROM_COMPLETE);

    // OpenBSD acknowledges the old completion with the next Start.
    rig.ind_reg_write(reg::EEPROM_ADDR, EEPROM_READ | EEPROM_COMPLETE | EEPROM_START | 3 << 26 | 4);
    assert_eq!(rig.ind_reg_read(reg::EEPROM_DATA), 0x7856_3412);
    assert_eq!(done(&rig), EEPROM_COMPLETE);

    // Address bits 1:0 are ignored: this acknowledges and starts nothing.
    rig.writel(reg::EEPROM_DATA, 0x1357_9bdf);
    rig.writel(reg::EEPROM_ADDR, EEPROM_COMPLETE);
    assert_eq!(rig.readl(reg::EEPROM_DATA), 0x1357_9bdf);
    assert_eq!(done(&rig), 0);
    assert_eq!(std::fs::read(&path).unwrap()[4..], [0x12, 0x34, 0x56, 0x78]);

    rig.writel(reg::EEPROM_DATA, 0x89ab_cdef);
    rig.ind_reg_write(reg::EEPROM_ADDR, EEPROM_START | 3 << 26 | 7);
    assert_eq!(rig.readl(reg::EEPROM_ADDR) & EEPROM_COMPLETE, EEPROM_COMPLETE);
    assert_eq!(rig.readl(reg::EEPROM_ADDR) & 3, 0);
    assert_eq!(rig.readl(reg::EEPROM_DELAY), 0);
    assert_eq!(std::fs::read(&path).unwrap(), [0x66, 0x99, 0x55, 0xaa, 0xef, 0xcd, 0xab, 0x89]);

    // Block reset clears the volatile state; the delay register only reads back.
    rig.writel(reg::EEPROM_DELAY, 0x000a_2c2a);
    rig.writel(reg::EEPROM_ADDR, EEPROM_RESET);
    assert_eq!(rig.readl(reg::EEPROM_ADDR), EEPROM_RESET);
    assert_eq!(rig.readl(reg::EEPROM_DATA), 0);
    assert_eq!(rig.readl(reg::EEPROM_DELAY), 0x000a_2c2a);

    // Reset wins over Start, including while the level remains set.
    for data in [0x1122_3344, 0x5566_7788] {
        rig.writel(reg::EEPROM_DATA, data);
        rig.writel(reg::EEPROM_ADDR, EEPROM_RESET | EEPROM_START | 3 << 26 | 4);
        assert_eq!(rig.readl(reg::EEPROM_ADDR), EEPROM_RESET);
        assert_eq!(rig.readl(reg::EEPROM_DATA), 0);
    }

    // Chip and machine resets do not replace the external bytes.
    rig.writel(reg::EEPROM_ADDR, 0);
    rig.writel(reg::MISC_CFG, MISC_CFG_RESET);
    assert_eq!(rig.readl(reg::EEPROM_DELAY), 0);
    rig.writel(reg::EEPROM_ADDR, EEPROM_READ | EEPROM_START | 3 << 26 | 4);
    assert_eq!(rig.readl(reg::EEPROM_DATA), 0x89ab_cdef);
    rig.dev.reset();
    rig.writel(reg::EEPROM_ADDR, EEPROM_READ | EEPROM_START | 3 << 26 | 4);
    assert_eq!(rig.readl(reg::EEPROM_DATA), 0x89ab_cdef);

    // A fresh process sees the guest write in the same file.
    let again = Eeprom::from_file(3, path.clone()).unwrap();
    assert_eq!(again.read_word(3, 1), Ok(0x89ab_cdef));
    std::fs::remove_file(path).unwrap();
}

/// Ported from test_eeprom_missing_backing: no backing means no transfer and
/// the data register is untouched.
#[test]
fn an_eeprom_command_with_no_backing_does_not_complete() {
    let rig = Rig::new();
    rig.writel(reg::EEPROM_DATA, 0x7654_3210);
    rig.writel(reg::EEPROM_ADDR, EEPROM_READ | EEPROM_START);
    assert_eq!(rig.readl(reg::EEPROM_DATA), 0x7654_3210);
    assert_eq!(rig.readl(reg::EEPROM_ADDR) & (EEPROM_COMPLETE | EEPROM_START), 0);
    assert_eq!(rig.refusals(Refused::EepromFault(Fault::NoBacking)), 1);
}

/// Ported from test_eeprom_read_only_backing.
#[test]
fn a_read_only_eeprom_refuses_writes() {
    let mut eeprom = Eeprom::from_image(3, vec![0x66, 0x99, 0x55, 0xaa]).unwrap();
    eeprom.set_read_only(true);
    assert!(eeprom.is_read_only());
    let rig = Rig::with(Bcm5701Config { eeprom: Some(eeprom), ..Bcm5701Config::new(MAC) });
    rig.writel(reg::EEPROM_DATA, 0x1234_5678);
    rig.writel(reg::EEPROM_ADDR, EEPROM_START | 3 << 26);
    assert_eq!(rig.readl(reg::EEPROM_DATA), 0x1234_5678);
    assert_eq!(rig.readl(reg::EEPROM_ADDR) & (EEPROM_COMPLETE | EEPROM_START), 0);
    assert_eq!(rig.dev.eeprom_bytes().unwrap(), [0x66, 0x99, 0x55, 0xaa]);
    assert_eq!(rig.refusals(Refused::EepromFault(Fault::ReadOnly)), 1);
}

/// Ported from test_eeprom_bounds_and_device_id.
#[test]
fn eeprom_words_past_the_backing_and_other_device_ids_fail() {
    let rig = eeprom_rig(&[0x66, 0x99, 0x55, 0xaa], 3);
    rig.writel(reg::EEPROM_DATA, 0x1234_5678);
    for command in [EEPROM_READ | EEPROM_START | 3 << 26 | 4, EEPROM_READ | EEPROM_START | 4 << 26] {
        rig.writel(reg::EEPROM_ADDR, command);
        assert_eq!(rig.readl(reg::EEPROM_DATA), 0x1234_5678);
        assert_eq!(rig.readl(reg::EEPROM_ADDR) & (EEPROM_COMPLETE | EEPROM_START), 0);
    }
    assert_eq!(rig.refusals(Refused::EepromFault(Fault::OutOfRange(1))), 1);
    assert_eq!(rig.refusals(Refused::EepromFault(Fault::WrongDevice(4))), 1);
}

/// Ported from test_eeprom_last_interface_word: the 64 KiB reach of the
/// address field.
#[test]
fn the_last_eeprom_interface_word_is_reachable() {
    let mut image = vec![0u8; 65536];
    image[65532..].copy_from_slice(&[0x66, 0x99, 0x55, 0xaa]);
    let rig = eeprom_rig(&image, 3);
    rig.writel(reg::EEPROM_ADDR, EEPROM_READ | EEPROM_START | 3 << 26 | 0x3fff << 2);
    assert_eq!(rig.readl(reg::EEPROM_DATA), 0xaa55_9966);
    rig.writel(reg::EEPROM_DATA, 0x89ab_cdef);
    rig.writel(reg::EEPROM_ADDR, EEPROM_COMPLETE | EEPROM_START | 3 << 26 | 0x3fff << 2);
    assert_eq!(rig.dev.eeprom_bytes().unwrap()[65532..], [0xef, 0xcd, 0xab, 0x89]);
}

/// Ported from test_eeprom_rejects_non_word_backing and
/// test_eeprom_id_requires_backing: the configuration is validated up front.
#[test]
fn eeprom_backings_are_validated() {
    assert!(Eeprom::from_image(3, vec![1, 2, 3, 4, 5]).unwrap_err().contains("multiple of four"));
    assert!(Eeprom::from_image(3, vec![]).is_err());
    assert!(Eeprom::from_image(3, vec![0; 65540]).is_err());
    assert!(Eeprom::from_image(8, vec![0; 4]).unwrap_err().contains("0..=7"));
    assert!(Bcm5701::new(Bcm5701Config::new([1, 0, 0, 0, 0, 0])).is_err());
}

/// Direct serial-pin control is unsupported and observable, not faked.
#[test]
fn direct_eeprom_pin_control_is_refused() {
    let rig = Rig::new();
    rig.writel(reg::EEPROM_PIN, 0x15);
    assert_eq!(rig.readl(reg::EEPROM_PIN), 0x15);
    assert_eq!(rig.refusals(Refused::EepromPins), 1);
}

// ---- the descriptor engines ----

fn be32(b: &[u8], off: usize) -> u32 {
    u32::from_be_bytes(b[off..off + 4].try_into().unwrap())
}
fn put32(b: &mut [u8], off: usize, v: u32) {
    b[off..off + 4].copy_from_slice(&v.to_be_bytes());
}
/// A 16-bit descriptor field sits in the other half of its word, big-endian.
fn put16dma(b: &mut [u8], off: usize, v: u16) {
    b[off ^ 2..(off ^ 2) + 2].copy_from_slice(&v.to_be_bytes());
}
fn get16dma(b: &[u8], off: usize) -> u16 {
    u16::from_be_bytes([b[off ^ 2], b[(off ^ 2) + 1]])
}

fn tx_bd(buf: u64, len: u16, flags: u16, mss: u16) -> [u8; 16] {
    let mut bd = [0; 16];
    put32(&mut bd, 0, (buf >> 32) as u32);
    put32(&mut bd, 4, buf as u32);
    put16dma(&mut bd, 8, flags);
    put16dma(&mut bd, 10, len);
    put16dma(&mut bd, 14, mss);
    bd
}

const TXF_TCP_UDP_CKSUM: u16 = 1;
const TXF_IP_CKSUM: u16 = 2;
const TXF_END: u16 = 4;
const TXF_INSERT_SRC: u16 = 0x1000;
const TXF_CHOOSE_SRC: fn(u16) -> u16 = |n| n << 13;

fn rx_bd(buf: u64, buf_len: u16) -> [u8; 32] {
    let mut bd = [0; 32];
    put32(&mut bd, 0, (buf >> 32) as u32);
    put32(&mut bd, 4, buf as u32);
    put16dma(&mut bd, 8, buf_len);
    bd
}

impl Rig {
    /// Send ring in memory, status block address, coalescing and the MAC on,
    /// the way a driver brings the transmit side up.
    fn tx_open(&self, ring_len: u32, coal_bds: u32, coal_ticks: u32) {
        self.ind_reg_write(reg::MODE_CTL, MODE_ALL_SWAPS);
        self.ind_mem_write(0x100, 0);
        self.ind_mem_write(0x104, TX_RING as u32);
        self.ind_mem_write(0x108, ring_len << 16);
        self.split_addr(reg::HCC_STATUSBLK_ADDR_HI, reg::HCC_STATUSBLK_ADDR_LO, STATUS);
        self.writel(reg::HCC_TX_MAX_COAL_BDS, coal_bds);
        self.writel(reg::HCC_TX_COAL_TICKS, coal_ticks);
        self.writel(reg::HCC_MODE, HCC_ENABLE);
        self.writel(reg::RX_MODE, RXMODE_ENABLE);
    }

    fn tx_post(&self, slot: u64, bd: [u8; 16]) {
        self.mem_write(TX_RING + slot * 16, &bd);
    }

    /// One whole frame in one descriptor at `slot`, then its doorbell.
    fn tx_send_one(&self, slot: u64, frame: &[u8], flags: u16) {
        let buf = TX_BUF + slot * 0x800;
        self.mem_write(buf, frame);
        self.tx_post(slot, tx_bd(buf, frame.len() as u16, flags | TXF_END, 0));
        self.writel(reg::MBX_TX_PROD_LO, slot as u32 + 1);
    }

    /// Receive side up: standard ring of `STD_BUF_SIZE` buffers, a return
    /// ring of `ret_len`, one completion per update.
    fn rx_open(&self, ret_len: u32, coal_bds: u32, coal_ticks: u32) {
        self.ind_reg_write(reg::MODE_CTL, MODE_ALL_SWAPS);
        self.split_addr(reg::RX_STD_RCB_HADDR_HI, reg::RX_STD_RCB_HADDR_LO, STD_RING);
        self.writel(reg::RX_STD_RCB_MAXLEN_FLAGS, STD_BUF_SIZE << 16);
        self.ind_mem_write(0x200, 0);
        self.ind_mem_write(0x204, RET_RING as u32);
        self.ind_mem_write(0x208, ret_len << 16);
        self.split_addr(reg::HCC_STATUSBLK_ADDR_HI, reg::HCC_STATUSBLK_ADDR_LO, STATUS);
        self.writel(reg::HCC_RX_MAX_COAL_BDS, coal_bds);
        self.writel(reg::HCC_RX_COAL_TICKS, coal_ticks);
        self.writel(reg::HCC_MODE, HCC_ENABLE);
        self.writel(reg::RX_MODE, RXMODE_ENABLE);
        self.writel(reg::MBX_IRQ0_LO, 0);
    }

    /// Post `count` standard buffers, all at `RX_BUF`.
    fn rx_post_std(&self, count: u32) {
        for i in 0..u64::from(count) {
            self.mem_write(STD_RING + i * 32, &rx_bd(RX_BUF, 0));
        }
        self.writel(reg::MBX_RX_STD_PROD_LO, count);
    }

    fn rx_open_jumbo(&self) {
        self.split_addr(reg::RX_JUMBO_RCB_HADDR_HI, reg::RX_JUMBO_RCB_HADDR_LO, JUMBO_RING);
        self.writel(reg::RX_JUMBO_RCB_MAXLEN_FLAGS, JUMBO_BUF_SIZE << 16);
        self.mem_write(JUMBO_RING, &rx_bd(JUMBO_BUF, JUMBO_BUF_SIZE as u16));
        self.writel(reg::MBX_RX_JUMBO_PROD_LO, 1);
    }

    fn return_descriptor(&self, slot: u64) -> Vec<u8> {
        self.mem_read(RET_RING + slot * 32, 32)
    }
}

fn broadcast(len: usize, seed: u8) -> Vec<u8> {
    let mut f: Vec<u8> = (0..len).map(|i| (i as u8).wrapping_mul(seed).wrapping_add(1)).collect();
    f[..6].fill(0xff);
    f
}

/// A doorbell posts the status block with UPDATED set, big-endian, once the
/// coalescing block is enabled; disabled it posts nothing and the line stays
/// down. Ported from test_status_block_byte_order and
/// test_status_block_gated_by_hcc.
#[test]
fn the_status_block_is_posted_only_while_host_coalescing_is_enabled() {
    let rig = Rig::new();
    rig.ind_mem_write(0x108, TX_RING_LEN << 16);
    rig.split_addr(reg::HCC_STATUSBLK_ADDR_HI, reg::HCC_STATUSBLK_ADDR_LO, STATUS);
    rig.ind_reg_write(reg::MODE_CTL, MODE_ALL_SWAPS);
    rig.writel(reg::MBX_IRQ0_LO, 0);

    rig.writel(reg::MBX_TX_PROD_LO, 0);
    assert_eq!(be32(&rig.status_block(), 0), 0);
    assert!(!rig.line_up());

    rig.writel(reg::HCC_MODE, HCC_ENABLE);
    rig.writel(reg::RX_MODE, RXMODE_ENABLE);
    rig.writel(reg::MBX_TX_PROD_LO, 0);
    assert_eq!(rig.status_block()[..4], [0, 0, 0, 1]);
    assert!(rig.line_up());
    assert!(rig.irq.level());
}

/// With ENDIAN_WORDSWAP set, a 32-bit access at +0 of an eight-byte field
/// reaches its more significant word, so the big-endian driver writes each
/// host address high word first and the pair composes with no exchange. An
/// exchange on top sends the status block to the low word shifted into the
/// high half, and IRIX's tg0 then never sees a transmit completion or a
/// received frame. Ported from test_status_block_hostaddr_wordswap.
#[test]
fn host_addresses_compose_high_word_first_under_the_word_swap() {
    let intended = 0x80_0000u64;
    let rig = Rig::new();
    rig.ind_mem_write(0x108, TX_RING_LEN << 16);
    rig.ind_reg_write(reg::MODE_CTL, MODE_ALL_SWAPS);
    rig.cfg_write(0x68 / 4, rig.cfg_read(0x68 / 4) | misc_ctl::ENDIAN_WORDSWAP);
    rig.writel(reg::HCC_STATUSBLK_ADDR_HI, (intended >> 32) as u32);
    rig.writel(reg::HCC_STATUSBLK_ADDR_LO, intended as u32);
    rig.writel(reg::HCC_MODE, HCC_ENABLE);
    rig.writel(reg::RX_MODE, RXMODE_ENABLE);
    rig.writel(reg::MBX_TX_PROD_LO, 0);
    assert_eq!(be32(&rig.mem_read(intended, STATUS_BLK_SIZE), 0), 1);
}

/// Ported from test_transmit_round_trip: the descriptor and its buffer are
/// read, the frame reaches the backend byte for byte, and the status block
/// reports the consumer advanced, in the chip's image order.
#[test]
fn a_transmitted_frame_reaches_the_backend_and_the_status_block_follows() {
    let mut rig = Rig::new();
    let frame: Vec<u8> = (0..60u8).map(|i| i.wrapping_mul(7).wrapping_add(1)).collect();
    rig.tx_open(TX_RING_LEN, 1, 0);
    rig.tx_send_one(0, &frame, 0);
    assert_eq!(rig.sent_frames(), [frame]);
    let sb = rig.status_block();
    assert_eq!(be32(&sb, 0), 1);
    assert_eq!(get16dma(&sb, 18), 1);
    // A consumer of 1 at the field's offset xor 2, big-endian.
    assert_eq!(sb[16..18], [0, 1]);
}

/// With no swap controls the chip's image reaches the bus permuted within each
/// doubleword; the frame still arrives intact. Derived from the lane rule the
/// model documents (PG105 Tables 102-109), not a port.
#[test]
fn the_swap_controls_permute_the_bytes_within_each_doubleword() {
    for (mode, xor) in [(0u32, 7usize), (0x10, 4), (0x20, 3), (0x30, 0)] {
        let mut rig = Rig::new();
        rig.tx_open(TX_RING_LEN, 1, 0);
        rig.ind_reg_write(reg::MODE_CTL, mode);
        let frame: Vec<u8> = (0..64u8).map(|i| i.wrapping_mul(5) ^ 0x21).collect();
        let lanes = |bytes: &[u8], xor: usize| {
            let mut out = vec![0; bytes.len()];
            for (i, b) in bytes.iter().enumerate() {
                out[(i & !7) + ((i & 7) ^ xor)] = *b;
            }
            out
        };
        // Descriptors take the non-frame controls, which are off here.
        rig.mem_write(TX_RING, &lanes(&tx_bd(TX_BUF, 64, TXF_END, 0), 7));
        rig.mem_write(TX_BUF, &lanes(&frame, xor));
        rig.writel(reg::MBX_TX_PROD_LO, 1);
        assert_eq!(rig.sent_frames(), [frame], "frame-data control {mode:#x}");
        // The status block is written through the same non-frame lanes.
        let sb = lanes(&rig.status_block(), 7);
        assert_eq!(be32(&sb, 0), 1);
    }
}

/// A transmit descriptor that cannot be read leaves the ring where it was and
/// the doorbell is retried later. Ported from test_transmit_descriptor_dma_error.
#[test]
fn an_unreadable_send_ring_leaves_the_frame_for_a_retry() {
    let mut rig = Rig::new();
    let frame: Vec<u8> = (0..60u8).map(|i| i.wrapping_mul(7).wrapping_add(1)).collect();
    rig.tx_open(TX_RING_LEN, 0, 0);
    rig.ind_mem_write(0x100, (UNMAPPED >> 32) as u32);
    rig.ind_mem_write(0x104, UNMAPPED as u32);
    rig.writel(reg::MBX_TX_PROD_LO, 1);
    assert!(rig.sent_frames().is_empty());
    assert_eq!(rig.status_tag(), 0);
    assert_eq!(rig.refusals(Refused::DmaFault(DmaWhat::TxDescriptor)), 1);

    rig.ind_mem_write(0x104, TX_RING as u32);
    rig.mem_write(TX_BUF, &frame);
    rig.tx_post(0, tx_bd(TX_BUF, 60, TXF_END, 0));
    rig.writel(reg::MBX_TX_PROD_LO, 1);
    assert_eq!(rig.sent_frames(), [frame]);
}

/// A later fragment's unreadable payload rolls the whole frame back, and an
/// earlier completed frame stays completed. Ported from
/// test_transmit_payload_dma_error and test_transmit_complete_then_fragment_dma_error_retry.
#[test]
fn an_unreadable_fragment_rolls_back_to_the_start_of_its_frame() {
    let mut rig = Rig::new();
    let first: Vec<u8> = (0..60u8).map(|i| i + 1).collect();
    let second: Vec<u8> = (0..60u8).map(|i| i + 100).collect();
    rig.tx_open(8, 1, 0);
    rig.mem_write(TX_BUF, &first);
    rig.tx_post(0, tx_bd(TX_BUF, 60, TXF_END, 0));
    rig.mem_write(TX_BUF + 0x100, &second[..28]);
    rig.tx_post(1, tx_bd(TX_BUF + 0x100, 28, 0, 0));
    rig.tx_post(2, tx_bd(UNMAPPED, 32, TXF_END, 0));
    rig.writel(reg::MBX_TX_PROD_LO, 3);
    // The first frame went and is reported; the second waits.
    assert_eq!(rig.sent_frames(), [first]);
    assert_eq!(rig.status_tx_cons(), 1);
    assert_eq!(rig.refusals(Refused::DmaFault(DmaWhat::TxPayload)), 1);

    rig.mem_write(TX_BUF + 0x200, &second[28..]);
    rig.tx_post(2, tx_bd(TX_BUF + 0x200, 32, TXF_END, 0));
    rig.writel(reg::MBX_TX_PROD_LO, 3);
    assert_eq!(rig.sent_frames(), [second]);
    assert_eq!(rig.status_tx_cons(), 3);
}

/// Ported from test_tx_producer_out_of_range: an index past the ring is
/// refused, observably, and nothing is walked.
#[test]
fn a_transmit_producer_past_the_ring_is_refused() {
    let rig = Rig::new();
    rig.ind_mem_write(0x108, TX_RING_LEN << 16);
    rig.writel(reg::MBX_TX_PROD_LO, TX_RING_LEN);
    rig.writel(reg::MBX_TX_PROD_LO, 0xffff);
    assert_eq!(rig.refusals(Refused::ProducerOutOfRange(Mailbox::TxProducer)), 2);
    assert_eq!(rig.mii_read(PHY_ADDR, phy_reg::IDR1), IDR1_VALUE);
}

/// Ported from test_transmit_zero_length_descriptor: nothing goes out, the
/// descriptor still completes.
#[test]
fn a_zero_length_descriptor_completes_without_a_frame() {
    let mut rig = Rig::new();
    rig.tx_open(TX_RING_LEN, 1, 0);
    rig.tx_post(0, tx_bd(TX_BUF, 0, TXF_END, 0));
    rig.writel(reg::MBX_TX_PROD_LO, 1);
    assert_eq!(be32(&rig.status_block(), 0), 1);
    assert_eq!(rig.status_tx_cons(), 1);
    assert!(rig.sent_frames().is_empty());
}

/// Ported from test_transmit_source_address: the descriptor picks one of four
/// filter slots to replace the frame's source, only when it asks to.
#[test]
fn a_descriptor_can_replace_the_source_address_from_a_filter_slot() {
    let mut rig = Rig::new();
    rig.tx_open(8, 0, 0);
    let slots = [(0x210u32, 0x2030_4050u32), (0x211, 0x2131_4151), (0x212, 0x2232_4252), (0x213, 0x2333_4353)];
    for (i, (hi, lo)) in slots.iter().enumerate() {
        rig.writel(reg::MAC_ADDR_0_HIGH + i as u32 * reg::MAC_ADDR_STRIDE, *hi);
        rig.writel(reg::MAC_ADDR_0_LOW + i as u32 * reg::MAC_ADDR_STRIDE, *lo);
    }
    let frame: Vec<u8> = (0..60u8).map(|i| 0x80 + i).collect();
    let with_source = |slot: usize| {
        let mut f = frame.clone();
        f[6..8].copy_from_slice(&(slots[slot].0 as u16).to_be_bytes());
        f[8..12].copy_from_slice(&slots[slot].1.to_be_bytes());
        f
    };
    for (i, selector) in [2u16, 0, 3, 1].into_iter().enumerate() {
        rig.tx_send_one(i as u64, &frame, TXF_INSERT_SRC | TXF_CHOOSE_SRC(selector));
        assert_eq!(rig.sent_frames(), [with_source(usize::from(selector))]);
    }
    // A selector without INSERT leaves the frame untouched.
    rig.tx_send_one(4, &frame, TXF_CHOOSE_SRC(3));
    assert_eq!(rig.sent_frames(), std::slice::from_ref(&frame));
    // Split after the first source bytes: both fragments carry the flags.
    rig.mem_write(TX_BUF + 5 * 0x800, &frame[..8]);
    rig.mem_write(TX_BUF + 6 * 0x800, &frame[8..]);
    let flags = TXF_INSERT_SRC | TXF_CHOOSE_SRC(2);
    rig.tx_post(5, tx_bd(TX_BUF + 5 * 0x800, 8, flags, 0));
    rig.tx_post(6, tx_bd(TX_BUF + 6 * 0x800, 52, flags | TXF_END, 0));
    rig.writel(reg::MBX_TX_PROD_LO, 7);
    assert_eq!(rig.sent_frames(), [with_source(2)]);
}

fn lso_frame(payload: usize) -> Vec<u8> {
    let mut f = vec![0u8; 54 + payload];
    f[..14].copy_from_slice(&[0x52, 0x54, 0, 0x11, 0x22, 0x33, 0x52, 0x54, 0, 0xbc, 0x57, 1, 8, 0]);
    f[14] = 0x45;
    f[16..18].copy_from_slice(&((40 + payload) as u16).to_be_bytes());
    f[18..20].copy_from_slice(&0x1234u16.to_be_bytes());
    f[22] = 64;
    f[23] = 6;
    f[26..30].copy_from_slice(&[10, 0, 2, 15]);
    f[30..34].copy_from_slice(&[10, 0, 2, 2]);
    f[34..36].copy_from_slice(&0x3039u16.to_be_bytes());
    f[36..38].copy_from_slice(&0x0017u16.to_be_bytes());
    f[38..42].copy_from_slice(&0x1000u32.to_be_bytes());
    f[46] = 0x50;
    f[47] = 0x19;
    f[48..50].copy_from_slice(&0x2000u16.to_be_bytes());
    for (i, b) in f[54..].iter_mut().enumerate() {
        *b = i as u8;
    }
    f
}

fn inet_sum(buf: &[u8], seed: u32) -> u16 {
    let mut sum = seed;
    for pair in buf.chunks(2) {
        sum += u32::from(pair[0]) << 8 | u32::from(*pair.get(1).unwrap_or(&0));
    }
    while sum >> 16 != 0 {
        sum = (sum & 0xffff) + (sum >> 16);
    }
    !(sum as u16)
}

fn tcp_pseudo(frame: &[u8], tcp_len: usize) -> u32 {
    frame[26..34].chunks(2).map(|p| u32::from(p[0]) << 8 | u32::from(p[1])).sum::<u32>() + 6 + tcp_len as u32
}

/// Ported from test_transmit_segmentation: one buffer of three segments goes
/// out as three frames with their own headers, sequence numbers, identifiers
/// and checksums, and the finishing flags only on the last.
#[test]
fn an_oversized_tcp_buffer_is_cut_into_segments() {
    const MSS: usize = 512;
    let mut rig = Rig::new();
    let frame = lso_frame(MSS * 3);
    rig.tx_open(TX_RING_LEN, 0, 0);
    rig.mem_write(TX_BUF, &frame);
    rig.tx_post(0, tx_bd(TX_BUF, frame.len() as u16, 0x7000 | TXF_END, MSS as u16));
    rig.writel(reg::MBX_TX_PROD_LO, 1);
    let sent = rig.sent_frames();
    assert_eq!(sent.len(), 3);
    for (i, got) in sent.iter().enumerate() {
        let mut expected = frame[..54].to_vec();
        expected.extend_from_slice(&frame[54 + i * MSS..54 + (i + 1) * MSS]);
        expected[16..18].copy_from_slice(&((40 + MSS) as u16).to_be_bytes());
        expected[18..20].copy_from_slice(&(0x1234 + i as u16).to_be_bytes());
        expected[38..42].copy_from_slice(&(0x1000 + (i * MSS) as u32).to_be_bytes());
        if i < 2 {
            expected[47] &= !0x09;
        }
        expected[24..26].fill(0);
        expected[50..52].fill(0);
        let ip = inet_sum(&expected[14..34], 0);
        expected[24..26].copy_from_slice(&ip.to_be_bytes());
        let tcp = inet_sum(&expected[34..], tcp_pseudo(&expected, 20 + MSS));
        expected[50..52].copy_from_slice(&tcp.to_be_bytes());
        assert_eq!(got, &expected, "segment {i}");
    }
}

/// A frame no larger than one segment, or one that is not plain IPv4 TCP, is
/// sent whole. Ported from test_transmit_no_segmentation.
#[test]
fn a_frame_that_needs_no_segmentation_is_sent_whole() {
    let mut rig = Rig::new();
    rig.tx_open(TX_RING_LEN, 0, 0);
    let small = lso_frame(100);
    rig.mem_write(TX_BUF, &small);
    rig.tx_post(0, tx_bd(TX_BUF, small.len() as u16, TXF_END, 512));
    rig.writel(reg::MBX_TX_PROD_LO, 1);
    assert_eq!(rig.sent_frames(), [small]);
}

fn cksum_frame(ihl_byte: u8) -> Vec<u8> {
    let mut f = vec![0u8; 62];
    f[..14].copy_from_slice(&[0x52, 0x54, 0, 0x11, 0x22, 0x33, 0x52, 0x54, 0, 0xbc, 0x57, 1, 8, 0]);
    f[14] = ihl_byte;
    f[17] = 48;
    f[22] = 64;
    f[23] = 6;
    f[26..30].copy_from_slice(&[10, 0, 2, 15]);
    f[30..34].copy_from_slice(&[10, 0, 2, 2]);
    f[34..36].copy_from_slice(&[0x30, 0x39]);
    f[37] = 0x17;
    f[46] = 0x50;
    f[47] = 0x02;
    f[48] = 0x20;
    for (i, b) in f.iter_mut().enumerate().skip(54) {
        *b = i as u8;
    }
    f
}

/// Ported from test_transmit_checksum_offload: a driver that leaves the
/// checksum fields for the card gets them filled.
#[test]
fn the_card_fills_in_the_checksums_the_driver_asked_for() {
    let mut rig = Rig::new();
    let frame = cksum_frame(0x45);
    rig.tx_open(TX_RING_LEN, 0, 0);
    rig.tx_send_one(0, &frame, TXF_IP_CKSUM | TXF_TCP_UDP_CKSUM);
    let got = rig.sent_frames().remove(0);
    let ip = inet_sum(&frame[14..34], 0);
    let tcp = inet_sum(&frame[34..], tcp_pseudo(&frame, 28));
    assert_eq!(u16::from_be_bytes([got[24], got[25]]), ip);
    assert_eq!(u16::from_be_bytes([got[50], got[51]]), tcp);
    assert_ne!(ip, 0);
    assert_ne!(tcp, 0);
    assert_eq!(got[..24], frame[..24]);
}

/// A header the offload cannot trust is declined, not walked. Ported from
/// test_transmit_checksum_bad_header.
#[test]
fn a_checksum_offload_on_a_malformed_header_is_declined() {
    let mut rig = Rig::new();
    let frame = cksum_frame(0x44);
    rig.tx_open(TX_RING_LEN, 0, 0);
    rig.tx_send_one(0, &frame, TXF_IP_CKSUM | TXF_TCP_UDP_CKSUM);
    assert_eq!(rig.sent_frames(), [frame]);
}

/// One update per group of descriptors, the tick budget reporting a short
/// group, a zero budget never doing so. Ported from test_transmit_coalescing,
/// test_transmit_coalescing_timer and test_transmit_coalescing_zero_ticks.
#[test]
fn transmit_updates_follow_the_count_and_the_tick_budget() {
    let mut rig = Rig::new();
    let frame = broadcast(64, 3);
    rig.tx_open(64, 4, 0x1000_0000);
    for i in 0..8 {
        rig.tx_send_one(i, &frame, 0);
    }
    assert_eq!(rig.sent_frames().len(), 8);
    // Eight frames in groups of four are two updates, not eight.
    assert_eq!(rig.status_tag(), 2);

    // A short group waits for the budget.
    let rig = Rig::new();
    rig.tx_open(64, 8, 50);
    for i in 0..3 {
        rig.tx_send_one(i, &frame, 0);
    }
    assert_eq!(rig.status_tag(), 0);
    rig.step(Duration::from_micros(49));
    assert_eq!(rig.status_tag(), 0);
    rig.step(Duration::from_micros(1));
    assert_eq!(rig.status_tag(), 1);
    assert_eq!(rig.status_tx_cons(), 3);

    // Zero disables the timer: only the count reports.
    let mut rig = Rig::new();
    rig.tx_open(64, 2, 0);
    rig.tx_send_one(0, &frame, 0);
    rig.step(Duration::from_secs(1));
    assert_eq!(rig.status_tag(), 0);
    rig.tx_send_one(1, &frame, 0);
    assert_eq!(rig.status_tag(), 1);
    assert_eq!(rig.status_tx_cons(), 2);
    assert_eq!(rig.sent_frames().len(), 2);
}

/// Empty and dropped frames still complete every descriptor they consume.
/// Ported from test_transmit_coalescing_empty_and_dropped.
#[test]
fn empty_and_oversize_frames_still_complete_their_descriptors() {
    let mut rig = Rig::new();
    let frame = broadcast(64, 5);
    rig.tx_open(64, 4, 0x1000_0000);
    rig.mem_write(TX_BUF, &frame);
    rig.mem_fill(TX_BUF + 0x1000, 9600, 0xa5);
    rig.tx_post(0, tx_bd(TX_BUF, 64, TXF_END, 0));
    rig.tx_post(1, tx_bd(0, 0, TXF_END, 0));
    rig.tx_post(2, tx_bd(TX_BUF + 0x1000, 9600, 0, 0));
    rig.tx_post(3, tx_bd(TX_BUF + 0x4000, 1, TXF_END, 0));
    rig.writel(reg::MBX_IRQ0_LO, 0);
    rig.writel(reg::MBX_TX_PROD_LO, 4);
    // Only the first is a frame: the oversize pair is dropped whole.
    assert_eq!(rig.sent_frames(), [frame]);
    assert_eq!(rig.status_tx_cons(), 4);
    assert_eq!(rig.status_tag(), 1);
    assert!(rig.line_up());
    assert_eq!(rig.refusals(Refused::TxFrameTooLarge), 1);
}

/// A doorbell that ends inside a frame leaves every fragment to retry; a
/// completed frame earlier is not rolled back. Ported from
/// test_transmit_split_doorbell and test_transmit_complete_then_split_doorbell.
#[test]
fn a_doorbell_that_ends_inside_a_frame_waits_for_the_rest() {
    let mut rig = Rig::new();
    let first: Vec<u8> = (0..60u8).map(|i| i.wrapping_mul(7).wrapping_add(1)).collect();
    let second: Vec<u8> = (0..60u8).map(|i| i.wrapping_mul(3).wrapping_add(9)).collect();
    rig.tx_open(8, 1, 0x1000_0000);
    rig.writel(reg::MBX_IRQ0_LO, 0);
    rig.mem_write(TX_BUF, &first);
    rig.tx_post(0, tx_bd(TX_BUF, 60, TXF_END, 0));
    rig.mem_write(TX_BUF + 0x100, &second[..28]);
    rig.mem_write(TX_BUF + 0x200, &second[28..]);
    rig.tx_post(1, tx_bd(TX_BUF + 0x100, 28, 0, 0));
    rig.tx_post(2, tx_bd(TX_BUF + 0x200, 32, TXF_END, 0));
    // The doorbell stops after the first fragment of the second frame.
    rig.writel(reg::MBX_TX_PROD_LO, 2);
    assert_eq!(rig.sent_frames(), [first]);
    assert_eq!(rig.status_tx_cons(), 1);
    let tag = rig.status_tag();
    rig.writel(reg::MBX_TX_PROD_LO, 3);
    assert_eq!(rig.sent_frames(), [second]);
    assert_eq!(rig.status_tx_cons(), 3);
    assert_eq!(rig.status_tag(), tag + 1);
}

/// A status DMA the guest's address refused is retried when the address is
/// repaired. Ported from test_transmit_coalescing_status_dma_retry.
#[test]
fn a_refused_status_update_is_retried_by_the_budget_timer() {
    let mut rig = Rig::new();
    let frame = broadcast(64, 9);
    rig.tx_open(64, 1, 50);
    rig.split_addr(reg::HCC_STATUSBLK_ADDR_HI, reg::HCC_STATUSBLK_ADDR_LO, UNMAPPED);
    rig.writel(reg::MBX_IRQ0_LO, 0);
    rig.tx_send_one(0, &frame, 0);
    assert_eq!(rig.sent_frames(), [frame]);
    assert_eq!(rig.status_tx_cons(), 0);
    assert!(!rig.line_up());
    assert_eq!(rig.refusals(Refused::DmaFault(DmaWhat::StatusBlock)), 1);

    rig.split_addr(reg::HCC_STATUSBLK_ADDR_HI, reg::HCC_STATUSBLK_ADDR_LO, STATUS);
    rig.step(Duration::from_micros(51));
    assert_eq!(rig.status_tx_cons(), 1);
    assert_eq!(rig.status_tag(), 1);
    assert!(rig.line_up());
}

/// Ported from test_interrupt_mailbox and test_mailbox_nonzero_masks: the
/// mailbox masks and acknowledges, MISC_CTL's global mask holds the line down
/// without losing the update, and any nonzero value masks.
#[test]
fn the_interrupt_line_follows_the_mailbox_and_the_global_mask() {
    let rig = Rig::new();
    rig.ind_mem_write(0x108, TX_RING_LEN << 16);
    rig.split_addr(reg::HCC_STATUSBLK_ADDR_HI, reg::HCC_STATUSBLK_ADDR_LO, STATUS);
    rig.ind_reg_write(reg::MODE_CTL, MODE_ALL_SWAPS);
    rig.writel(reg::HCC_MODE, HCC_ENABLE);
    rig.writel(reg::RX_MODE, RXMODE_ENABLE);

    // Masked out of reset: an update posts, but the line stays down.
    rig.writel(reg::MBX_TX_PROD_LO, 0);
    assert!(!rig.line_up());
    // Enable interrupts; the next update asserts the line.
    rig.writel(reg::MBX_IRQ0_LO, 0);
    rig.writel(reg::MBX_TX_PROD_LO, 0);
    assert!(rig.line_up() && rig.irq.level());
    // Write 1: masks and acknowledges.
    rig.writel(reg::MBX_IRQ0_LO, 1);
    assert!(!rig.line_up() && !rig.irq.level());
    // An update posted while masked is discarded by the zero write.
    rig.writel(reg::MBX_TX_PROD_LO, 0);
    rig.writel(reg::MBX_IRQ0_LO, 0);
    assert!(!rig.line_up());
    rig.writel(reg::MBX_TX_PROD_LO, 0);
    assert!(rig.line_up());
    rig.writel(reg::MBX_IRQ0_LO, 0);
    assert!(!rig.line_up());
    rig.writel(reg::MBX_TX_PROD_LO, 0);
    assert!(rig.line_up());

    // The global mask suppresses the line; the update was never taken.
    let misc = rig.cfg_read(0x68 / 4);
    rig.cfg_write(0x68 / 4, misc | 2);
    assert!(!rig.line_up() && !rig.irq.level());
    rig.cfg_write(0x68 / 4, misc);
    assert!(rig.line_up() && rig.irq.level());
    // CLEAR_INTA acknowledges and self-clears.
    rig.cfg_write(0x68 / 4, misc | 1);
    assert!(!rig.line_up());
    assert_eq!(rig.cfg_read(0x68 / 4), misc);

    // Any nonzero In_ISR value masks while the handler runs.
    rig.writel(reg::MBX_IRQ0_LO, 2);
    rig.writel(reg::MBX_TX_PROD_LO, 0);
    assert!(!rig.line_up());
    rig.writel(reg::MBX_IRQ0_LO, 0);
    assert!(!rig.line_up());
    rig.writel(reg::MBX_TX_PROD_LO, 0);
    assert!(rig.line_up());
}

/// A failed status write advances neither its tag nor the interrupt state.
/// Ported from test_status_dma_error.
#[test]
fn a_failed_status_write_advances_nothing() {
    let rig = Rig::new();
    rig.ind_mem_write(0x108, TX_RING_LEN << 16);
    rig.ind_reg_write(reg::MODE_CTL, MODE_ALL_SWAPS);
    rig.split_addr(reg::HCC_STATUSBLK_ADDR_HI, reg::HCC_STATUSBLK_ADDR_LO, UNMAPPED);
    rig.writel(reg::HCC_MODE, HCC_ENABLE);
    rig.writel(reg::RX_MODE, RXMODE_ENABLE);
    rig.writel(reg::MBX_IRQ0_LO, 0);
    rig.writel(reg::MBX_TX_PROD_LO, 0);
    assert!(!rig.line_up());
    rig.split_addr(reg::HCC_STATUSBLK_ADDR_HI, reg::HCC_STATUSBLK_ADDR_LO, STATUS);
    rig.writel(reg::MBX_TX_PROD_LO, 0);
    assert_eq!(rig.status_tag(), 1);
    assert!(rig.line_up());
}

/// Ported from test_receive_round_trip: the frame lands in the posted buffer,
/// the return descriptor and status block describe it in the chip's image
/// order, and a second buffer exposes the next consumer.
#[test]
fn a_received_frame_lands_in_the_posted_buffer_with_its_completion() {
    let mut rig = Rig::new();
    let frame = broadcast(64, 5);
    rig.rx_open(RX_RING_LEN, 1, 0);
    rig.rx_post_std(1);
    rig.deliver(&frame);
    rig.step(ARRIVAL);
    assert_eq!(rig.mem_read(RX_BUF, 64), frame);
    let ret = rig.return_descriptor(0);
    assert_eq!(get16dma(&ret, 0x0a), 0);
    assert_eq!(get16dma(&ret, 0x08), 64 + 4);
    assert_eq!(get16dma(&ret, 0x0c) & 4, 4);
    assert_eq!(ret[0x0a..0x0c], [0, 0x44]);
    let sb = rig.status_block();
    assert_eq!(get16dma(&sb, 16), 1);
    assert_eq!(sb[8..12], [0, 1, 0, 0]);
    assert_eq!(sb[18..20], [0, 1]);

    // The return-ring acknowledgement is accepted; reposting exposes the
    // next consumer.
    rig.writel(reg::MBX_RX_CONS0_LO, 1);
    rig.mem_write(STD_RING + 32, &rx_bd(RX_BUF + 0x1000, 0));
    rig.writel(reg::MBX_RX_STD_PROD_LO, 2);
    rig.deliver(&frame);
    rig.step(ARRIVAL);
    assert_eq!(rig.status_rx_prod(), 2);
    assert_eq!(rig.status_block()[8..10], [0, 2]);
}

/// The return descriptor is written when the frame lands; the producer the
/// status block reports, and the interrupt, wait for the arrival deadline.
/// Ported from test_receive_waits_for_its_arrival_deadline.
#[test]
fn a_completion_waits_for_its_arrival_deadline() {
    let mut rig = Rig::new();
    let frame = broadcast(64, 7);
    rig.rx_open(RX_RING_LEN, 1, 0);
    rig.rx_post_std(1);
    rig.deliver(&frame);
    assert_eq!(rig.mem_read(RX_BUF, 64), frame);
    assert_eq!(get16dma(&rig.return_descriptor(0), 0x0c) & 4, 4);
    assert_eq!(rig.status_rx_prod(), 0);
    assert!(!rig.line_up());
    rig.step(ARRIVAL - Duration::from_nanos(1));
    assert_eq!(rig.status_rx_prod(), 0);
    assert!(!rig.line_up());
    rig.step(Duration::from_nanos(1));
    assert_eq!(rig.status_rx_prod(), 1);
    assert!(rig.line_up());
}

/// A frame that arrives after a doorbell is not reported until the
/// processor that rang it has run the cycle floor past it, however much host
/// time has gone by: host time passes while the guest is descheduled or slow,
/// and a reply reported while its sender is still in the send path is lost to
/// it. A frame from before any doorbell waits only for its deadline.
#[test]
fn a_reply_waits_for_the_sending_processor_to_run_on() {
    const FLOOR: u64 = 1_000;
    // The processor's cycle counter, which outlives the device.
    let word: *mut u64 = Box::leak(Box::new(5_000u64));
    let counter = crate::cpu::mips_core::CyclesPtr::new(word);
    let mut rig = Rig::with(Bcm5701Config { rx_arrival_cycles: FLOOR, ..Bcm5701Config::new(MAC) });
    rig.dev.set_cpu_cycles(vec![counter]);
    rig.tx_open(TX_RING_LEN, 1, 0);
    rig.rx_open(RX_RING_LEN, 1, 0);
    rig.rx_post_std(2);

    rig.deliver(&broadcast(64, 3));
    rig.step(ARRIVAL);
    assert_eq!(rig.status_rx_prod(), 1, "no doorbell yet: host time alone");

    crate::cpu::smp::set_current_cpu(0);
    rig.tx_send_one(0, &broadcast(64, 5), 0);
    crate::cpu::smp::set_current_cpu(u8::MAX);
    rig.writel(reg::MBX_RX_CONS0_LO, 1);
    rig.deliver(&broadcast(64, 7));
    rig.step(ARRIVAL * 10);
    assert_eq!(rig.status_rx_prod(), 1, "the sender has not run on");
    // SAFETY: the leaked word is live for the process and only this thread
    // touches it.
    unsafe { word.write_volatile(5_000 + FLOOR - 1) };
    rig.step(ARRIVAL);
    assert_eq!(rig.status_rx_prod(), 1, "one cycle short");
    unsafe { word.write_volatile(5_000 + FLOOR) };
    rig.step(ARRIVAL);
    assert_eq!(rig.status_rx_prod(), 2);
}

/// Ported from test_receive_jumbo_frame: a frame beyond the standard buffer
/// goes to the jumbo ring and is marked so.
#[test]
fn a_frame_too_big_for_the_standard_buffer_takes_the_jumbo_ring() {
    let mut rig = Rig::new();
    let frame = broadcast(9000, 7);
    rig.rx_open(RX_RING_LEN, 1, 0);
    rig.rx_post_std(1);
    rig.rx_open_jumbo();
    rig.deliver(&frame);
    rig.step(ARRIVAL);
    assert_eq!(rig.status_rx_prod(), 1);
    let ret = rig.return_descriptor(0);
    assert_eq!(get16dma(&ret, 0x08), 9004);
    assert_eq!(get16dma(&ret, 0x0c) & 0x20, 0x20);
    assert_eq!(rig.mem_read(JUMBO_BUF, 9000), frame);
    // Both consumers report what the chip fetched: the unused standard
    // buffer as well as the used jumbo one.
    let sb = rig.status_block();
    assert_eq!(sb[8..12], [0, 1, 0, 1]);
}

/// Post more standard buffers than the chip's cache holds, the way a driver
/// fills its ring at start.
fn rx_post_std_many(rig: &Rig, count: u32) {
    for i in 0..u64::from(count) {
        rig.mem_write(STD_RING + i * 32, &rx_bd(RX_BUF + (i % 8) * 0x800, 0));
    }
    rig.writel(reg::MBX_RX_STD_PROD_LO, count);
}

/// The status block's standard consumer is the next descriptor the chip
/// fetches, not the next a frame uses: it runs ahead of the frames by as
/// much as the chip's 128-descriptor cache, and never past the producer.
#[test]
fn the_standard_consumer_reports_the_fetch_not_the_frames() {
    let mut rig = Rig::new();
    let frame = broadcast(64, 9);
    rig.rx_open(16, 1, 0);
    rx_post_std_many(&rig, 300);
    rig.deliver(&frame);
    rig.step(ARRIVAL);
    assert_eq!(rig.status_rx_prod(), 1);
    // One used, its cache entry refilled: 1 + 128.
    assert_eq!(rig.status_std_cons(), 129);
    rig.writel(reg::MBX_RX_CONS0_LO, 1);
    rig.deliver(&frame);
    rig.step(ARRIVAL);
    assert_eq!(rig.status_std_cons(), 130);

    // Fewer posted than the cache holds: all of them, before any frame.
    let mut rig = Rig::new();
    rig.rx_open(16, 1, 0);
    rx_post_std_many(&rig, 10);
    rig.deliver(&frame);
    rig.step(ARRIVAL);
    assert_eq!(rig.status_std_cons(), 10);
    assert_eq!(rig.mem_read(RX_BUF, 64), frame);
}

/// The jumbo consumer is the fetch index as well, against a 64-descriptor
/// cache.
#[test]
fn the_jumbo_consumer_reports_the_fetch_not_the_frames() {
    let mut rig = Rig::new();
    let frame = broadcast(9000, 7);
    rig.rx_open(16, 1, 0);
    rig.split_addr(reg::RX_JUMBO_RCB_HADDR_HI, reg::RX_JUMBO_RCB_HADDR_LO, JUMBO_RING);
    rig.writel(reg::RX_JUMBO_RCB_MAXLEN_FLAGS, JUMBO_BUF_SIZE << 16);
    for i in 0..100 {
        rig.mem_write(JUMBO_RING + i * 32, &rx_bd(JUMBO_BUF, JUMBO_BUF_SIZE as u16));
    }
    rig.writel(reg::MBX_RX_JUMBO_PROD_LO, 100);
    rig.deliver(&frame);
    rig.step(ARRIVAL);
    assert_eq!(rig.status_rx_prod(), 1);
    assert_eq!(rig.status_jumbo_cons(), 65);
    assert_eq!(rig.mem_read(JUMBO_BUF, 9000), frame);
}

/// The chip starts a fetch only once the host has made the replenish
/// threshold's worth of descriptors available; until then a frame finds no
/// buffer and waits, though some are posted.
#[test]
fn a_fetch_waits_for_the_replenish_threshold() {
    let mut rig = Rig::new();
    let frame = broadcast(64, 21);
    rig.rx_open(16, 1, 0);
    rig.writel(reg::RX_STD_REPLENISH_THRESH, 25);
    assert_eq!(rig.refusals(Refused::UnmodelledWrite(reg::RX_STD_REPLENISH_THRESH)), 0);
    rx_post_std_many(&rig, 20);
    rig.deliver(&frame);
    rig.step(ARRIVAL);
    assert_eq!(rig.status_rx_prod(), 0);
    assert_eq!(rig.waiting(), 1);
    rx_post_std_many(&rig, 25);
    rig.pump();
    assert_eq!(rig.status_rx_prod(), 1);
    assert_eq!(rig.mem_read(RX_BUF, 64), frame);
    assert_eq!(rig.status_std_cons(), 25);

    // Lowering the threshold lets the posted buffers in without another
    // producer write.
    let mut rig = Rig::new();
    rig.rx_open(16, 1, 0);
    rig.writel(reg::RX_STD_REPLENISH_THRESH, 25);
    rx_post_std_many(&rig, 20);
    rig.deliver(&frame);
    assert_eq!(rig.waiting(), 1);
    rig.writel(reg::RX_STD_REPLENISH_THRESH, 1);
    rig.pump();
    assert_eq!(rig.status_rx_prod(), 1);
    assert_eq!(rig.status_std_cons(), 20);
}

/// With the standard ring empty, a standard-size frame is discarded and
/// counted rather than refused, while a jumbo buffer is posted; the next jumbo
/// frame still gets through. Ported from
/// test_receive_drops_standard_with_empty_standard_ring.
#[test]
fn a_short_frame_with_no_standard_buffer_is_discarded_and_counted() {
    let mut rig = Rig::new();
    let first = broadcast(64, 3);
    let second = broadcast(64, 11);
    let big = broadcast(9000, 13);
    rig.rx_open(RX_RING_LEN, 1, 0);
    rig.rx_post_std(1);
    rig.rx_open_jumbo();
    rig.mem_write(STD_RING + 32, &rx_bd(STD_STALE_BUF, 0));
    assert_eq!(rig.nic_no_more_rxbds(), 0);
    rig.deliver(&first);
    rig.step(ARRIVAL);
    rig.writel(reg::MBX_RX_CONS0_LO, 1);
    rig.mem_fill(STD_STALE_BUF, 64, 0xa5);
    rig.mem_fill(JUMBO_BUF, 64, 0xa5);
    rig.deliver(&second);
    rig.step(ARRIVAL);
    assert_eq!(rig.status_rx_prod(), 1);
    assert_eq!(rig.mem_read(STD_STALE_BUF, 64), vec![0xa5; 64]);
    assert_eq!(rig.mem_read(JUMBO_BUF, 64), vec![0xa5; 64]);
    assert_eq!(rig.nic_no_more_rxbds(), 1);
    assert_eq!(rig.refusals(Refused::RxShortage(Ring::Standard)), 1);
    // The jumbo ring still takes a jumbo frame.
    rig.deliver(&big);
    rig.step(ARRIVAL);
    assert_eq!(rig.status_rx_prod(), 2);
    assert_eq!(rig.mem_read(JUMBO_BUF, 9000), big);
    // Posting the standard buffer afterwards does not resurrect the frame.
    rig.writel(reg::MBX_RX_STD_PROD_LO, 2);
    rig.step(ARRIVAL);
    assert_eq!(rig.status_rx_prod(), 2);
    assert_eq!(rig.nic_no_more_rxbds(), 1);
}

/// The mirror: a jumbo frame with the jumbo ring empty. Ported from
/// test_receive_drops_oversize_with_empty_jumbo_ring.
#[test]
fn a_jumbo_frame_with_no_jumbo_buffer_is_discarded_and_counted() {
    let mut rig = Rig::new();
    let first = broadcast(9000, 3);
    let second = broadcast(9000, 11);
    let small = broadcast(64, 15);
    rig.rx_open(RX_RING_LEN, 1, 0);
    rig.rx_post_std(2);
    rig.rx_open_jumbo();
    rig.mem_write(JUMBO_RING + 32, &rx_bd(0x28_0000, JUMBO_BUF_SIZE as u16));
    rig.deliver(&first);
    rig.step(ARRIVAL);
    rig.writel(reg::MBX_RX_CONS0_LO, 1);
    rig.mem_fill(0x28_0000, 9000, 0xa5);
    rig.deliver(&second);
    rig.step(ARRIVAL);
    assert_eq!(rig.status_rx_prod(), 1);
    assert_eq!(rig.mem_read(0x28_0000, 9000), vec![0xa5; 9000]);
    assert_eq!(rig.nic_no_more_rxbds(), 1);
    assert_eq!(rig.refusals(Refused::RxShortage(Ring::Jumbo)), 1);
    rig.deliver(&small);
    rig.step(ARRIVAL);
    assert_eq!(rig.status_rx_prod(), 2);
    assert_eq!(rig.mem_read(RX_BUF, 64), small);
    assert_eq!(get16dma(&rig.return_descriptor(1), 0x0c) & 0x20, 0);
}

/// A frame too big for a standard buffer with the jumbo ring disabled is
/// dropped, observably. Ported from test_receive_oversize_dropped.
#[test]
fn an_oversize_frame_with_no_jumbo_ring_is_dropped_and_counted() {
    let mut rig = Rig::new();
    rig.rx_open(RX_RING_LEN, 1, 0);
    rig.rx_post_std(1);
    rig.mem_fill(RX_BUF, 2048, 0xa5);
    rig.deliver(&broadcast(2000, 3));
    rig.step(ARRIVAL);
    assert_eq!(rig.status_rx_prod(), 0);
    assert_eq!(rig.mem_read(RX_BUF, 2048), vec![0xa5; 2048]);
    assert_eq!(rig.refusals(Refused::RxNoRoom(Ring::Standard)), 1);
    // The frame was consumed, not left to wedge the queue.
    assert_eq!(rig.waiting(), 0);
    // And one over the model's limit is dropped too.
    rig.deliver(&broadcast(9700, 3));
    assert_eq!(rig.refusals(Refused::RxFrameTooLarge), 1);
}

/// Ported from test_receive_mac_enable: a halted receiver drops what it is
/// offered; with no buffer the frame waits for the guest to make room.
#[test]
fn a_halted_receiver_drops_and_a_full_one_waits() {
    let mut rig = Rig::new();
    let frame = broadcast(64, 3);
    rig.rx_open(RX_RING_LEN, 1, 0);
    rig.rx_post_std(1);
    rig.writel(reg::RX_MODE, 0);
    rig.deliver(&frame);
    assert_eq!(rig.waiting(), 0);
    assert_eq!(rig.status_rx_prod(), 0);
    rig.writel(reg::RX_MODE, RXMODE_ENABLE);
    rig.deliver(&frame);
    // The one posted buffer is used; the next frame waits at the head.
    rig.deliver(&frame);
    rig.step(ARRIVAL);
    assert_eq!(rig.status_rx_prod(), 1);
    assert_eq!(rig.waiting(), 1);
    rig.mem_write(STD_RING + 32, &rx_bd(RX_BUF + 0x1000, 0));
    rig.writel(reg::MBX_RX_STD_PROD_LO, 2);
    rig.writel(reg::MBX_RX_CONS0_LO, 1);
    rig.dev.poll();
    rig.step(ARRIVAL);
    assert_eq!(rig.status_rx_prod(), 2);
    assert_eq!(rig.mem_read(RX_BUF + 0x1000, 64), frame);
}

/// The address filter is slot 0, which the driver programs: other unicast
/// frames are consumed without a completion. Ported from the filter cases of
/// test_receive_checksum_case's setup and test_receive_round_trip.
#[test]
fn the_receive_filter_follows_the_programmed_station_address() {
    let mut rig = Rig::new();
    rig.rx_open(RX_RING_LEN, 1, 0);
    rig.rx_post_std(1);
    let mut other = broadcast(64, 3);
    other[..6].copy_from_slice(&[2, 0, 0, 0, 0, 9]);
    rig.deliver(&other);
    rig.step(ARRIVAL);
    assert_eq!(rig.status_rx_prod(), 0);
    assert_eq!(rig.waiting(), 0);
    let mut mine = other.clone();
    mine[..6].copy_from_slice(&MAC);
    rig.deliver(&mine);
    rig.step(ARRIVAL);
    assert_eq!(rig.status_rx_prod(), 1);
    // IRIX installs its own address over the card's.
    rig.writel(reg::MAC_ADDR_0_HIGH, 0x0200);
    rig.writel(reg::MAC_ADDR_0_LOW, 9);
    rig.mem_write(STD_RING + 32, &rx_bd(RX_BUF + 0x1000, 0));
    rig.writel(reg::MBX_RX_STD_PROD_LO, 2);
    rig.writel(reg::MBX_RX_CONS0_LO, 1);
    rig.deliver(&other);
    rig.step(ARRIVAL);
    assert_eq!(rig.status_rx_prod(), 2);
}

/// A runt is padded to the Ethernet minimum before it is delivered.
#[test]
fn a_short_frame_is_padded_to_the_ethernet_minimum() {
    let mut rig = Rig::new();
    rig.rx_open(RX_RING_LEN, 1, 0);
    rig.rx_post_std(1);
    let mut runt = vec![0xff; 6];
    runt.extend_from_slice(&[1, 2, 3, 4]);
    rig.deliver(&runt);
    rig.step(ARRIVAL);
    let mut padded = runt.clone();
    padded.resize(60, 0);
    assert_eq!(rig.mem_read(RX_BUF, 60), padded);
    assert_eq!(get16dma(&rig.return_descriptor(0), 0x08), 64);
}

fn rx_frame(proto: u8, payload: &[u8], fix: impl Fn(&mut Vec<u8>)) -> Vec<u8> {
    let mut f = vec![0xff; 6];
    f.extend_from_slice(&[0x52, 0x54, 0, 0xbc, 0x57, 1, 8, 0]);
    let total = 20 + payload.len();
    f.extend_from_slice(&[0x45, 0]);
    f.extend_from_slice(&(total as u16).to_be_bytes());
    f.extend_from_slice(&[0, 1, 0, 0, 64, proto, 0, 0, 10, 0, 2, 2, 10, 0, 2, 15]);
    f.extend_from_slice(payload);
    f.resize(f.len().max(60), 0);
    fix(&mut f);
    let ip = inet_sum(&f[14..34], 0);
    f[24..26].copy_from_slice(&ip.to_be_bytes());
    f
}

/// The receive checksum report: an intact TCP segment claims both sums
/// (0xffff) and the OK flags; a corrupt one claims the IP sum only; a frame
/// that is not IPv4 TCP or UDP claims nothing. Ported from
/// test_receive_checksum_offload, test_receive_checksum_bad_transport and
/// test_receive_checksum_not_ip.
#[test]
fn the_receive_report_claims_only_the_sums_it_computed() {
    let mut tcp = vec![0u8; 20];
    tcp[0..4].copy_from_slice(&[0x30, 0x39, 0, 0x17]);
    tcp[12] = 0x50;
    let seal = |f: &mut Vec<u8>| {
        f[14 + 20 + 16..14 + 20 + 18].fill(0);
        let sum = inet_sum(&f[34..54], tcp_pseudo(f, 20));
        f[50..52].copy_from_slice(&sum.to_be_bytes());
    };
    for (label, frame, ip_ok, l4_ok, ip_report, l4_report) in [
        ("intact", rx_frame(6, &tcp, seal), true, true, 0xffffu32, 0xffffu32),
        ("corrupt", rx_frame(6, &tcp, |_| ()), true, false, 0xffff, 0),
    ] {
        let mut rig = Rig::new();
        rig.rx_open(RX_RING_LEN, 1, 0);
        rig.rx_post_std(1);
        rig.deliver(&frame);
        rig.step(ARRIVAL);
        let ret = rig.return_descriptor(0);
        let flags = get16dma(&ret, 0x0c);
        assert_eq!(flags & 0x1000 != 0, ip_ok, "{label} ip flag");
        assert_eq!(flags & 0x2000 != 0, l4_ok, "{label} transport flag");
        assert_eq!(flags & 0x4000, 0x4000, "{label} tcp flag");
        assert_eq!(u32::from(get16dma(&ret, 0x12)), ip_report, "{label} ip sum");
        let l4 = u32::from(get16dma(&ret, 0x10));
        if l4_ok {
            assert_eq!(l4, l4_report);
        } else {
            assert_ne!(l4, 0xffff);
        }
    }
    // Not IPv4: nothing is claimed.
    let mut rig = Rig::new();
    rig.rx_open(RX_RING_LEN, 1, 0);
    rig.rx_post_std(1);
    let mut arp = broadcast(60, 3);
    arp[12..14].copy_from_slice(&[0x08, 0x06]);
    rig.deliver(&arp);
    rig.step(ARRIVAL);
    let ret = rig.return_descriptor(0);
    assert_eq!(get16dma(&ret, 0x0c) & 0x7000, 0);
    assert_eq!(be32(&ret, 0x10), 0);
}

/// Receive coalescing: with a count of two, two frames make one update; the
/// tick budget reports a short group. Ported from
/// test_receive_interrupt_coalescing and test_receive_coalescing_timer.
#[test]
fn receive_updates_follow_the_count_and_the_tick_budget() {
    let mut rig = Rig::new();
    let frame = broadcast(64, 3);
    rig.rx_open(16, 2, 0x1000_0000);
    rig.rx_post_std(4);
    for _ in 0..4 {
        rig.deliver(&frame);
    }
    rig.step(ARRIVAL);
    assert_eq!(rig.status_tag(), 2);
    assert_eq!(rig.status_rx_prod(), 4);

    let mut rig = Rig::new();
    rig.rx_open(16, 8, 1000);
    rig.rx_post_std(4);
    rig.deliver(&frame);
    rig.deliver(&frame);
    rig.step(ARRIVAL);
    assert_eq!(rig.status_tag(), 0);
    rig.step(Duration::from_micros(800));
    assert_eq!(rig.status_tag(), 1);
    assert_eq!(rig.status_rx_prod(), 2);

    // Neither criterion armed: the guest has asked to be told nothing.
    let mut rig = Rig::new();
    rig.rx_open(16, 0, 0);
    rig.rx_post_std(2);
    rig.deliver(&frame);
    rig.step(Duration::from_secs(1));
    assert_eq!(rig.status_tag(), 0);
}

/// Ported from test_rx_producer_out_of_range and the ring-length guards:
/// out-of-range indexes are refused, observably, and not stored.
#[test]
fn receive_mailbox_indexes_past_their_rings_are_refused() {
    let rig = Rig::new();
    rig.rx_open(RX_RING_LEN, 1, 0);
    rig.writel(reg::MBX_RX_STD_PROD_LO, 512);
    rig.writel(reg::MBX_RX_JUMBO_PROD_LO, 256);
    rig.writel(reg::MBX_RX_CONS0_LO, RX_RING_LEN);
    assert_eq!(rig.readl(reg::MBX_RX_STD_PROD_LO), 0);
    assert_eq!(rig.refusals(Refused::ProducerOutOfRange(Mailbox::RxStandardProducer)), 1);
    assert_eq!(rig.refusals(Refused::ProducerOutOfRange(Mailbox::RxJumboProducer)), 1);
    assert_eq!(rig.refusals(Refused::ProducerOutOfRange(Mailbox::RxReturnConsumer)), 1);
}

/// A return-ring descriptor that cannot be read or written leaves the frame
/// queued; repairing the address delivers it. Ported from
/// test_receive_descriptor_dma_error and test_receive_completion_dma_error.
#[test]
fn an_unreachable_ring_defers_the_frame_until_it_is_repaired() {
    let mut rig = Rig::new();
    let frame = broadcast(64, 3);
    rig.rx_open(RX_RING_LEN, 1, 0);
    rig.split_addr(reg::RX_STD_RCB_HADDR_HI, reg::RX_STD_RCB_HADDR_LO, UNMAPPED);
    rig.mem_write(STD_RING, &rx_bd(RX_BUF, 0));
    rig.writel(reg::MBX_RX_STD_PROD_LO, 1);
    rig.deliver(&frame);
    assert_eq!(rig.status_rx_prod(), 0);
    assert_eq!(rig.waiting(), 1);
    assert_eq!(rig.refusals(Refused::DmaFault(DmaWhat::RxDescriptor)), 1);
    rig.split_addr(reg::RX_STD_RCB_HADDR_HI, reg::RX_STD_RCB_HADDR_LO, STD_RING);
    rig.writel(reg::MBX_RX_STD_PROD_LO, 1);
    rig.dev.poll();
    rig.step(ARRIVAL);
    assert_eq!(rig.status_rx_prod(), 1);
    assert_eq!(rig.mem_read(RX_BUF, 64), frame);

    // The same for the return ring.
    let mut rig = Rig::new();
    rig.rx_open(RX_RING_LEN, 1, 0);
    rig.ind_mem_write(0x204, UNMAPPED as u32);
    rig.ind_mem_write(0x200, (UNMAPPED >> 32) as u32);
    rig.rx_post_std(1);
    rig.deliver(&frame);
    assert_eq!(rig.refusals(Refused::DmaFault(DmaWhat::RxCompletion)), 1);
    rig.ind_mem_write(0x200, 0);
    rig.ind_mem_write(0x204, RET_RING as u32);
    rig.writel(reg::MBX_RX_STD_PROD_LO, 1);
    rig.dev.poll();
    rig.step(ARRIVAL);
    assert_eq!(rig.status_rx_prod(), 1);
}

/// A global reset cancels pending coalescing work and the held completions.
/// Ported from test_global_reset_cancels_receive_coalescing and
/// test_system_reset_cancels_transmit_coalescing.
#[test]
fn a_reset_cancels_coalescing_timers_and_held_completions() {
    let rig = Rig::new();
    let frame = broadcast(64, 3);
    rig.tx_open(64, 8, 50);
    rig.tx_send_one(0, &frame, 0);
    rig.writel(reg::MISC_CFG, MISC_CFG_RESET);
    rig.step(Duration::from_millis(1));
    assert_eq!(rig.status_tag(), 0);

    let mut rig = Rig::new();
    rig.rx_open(16, 1, 0);
    rig.rx_post_std(1);
    rig.deliver(&frame);
    rig.dev.reset();
    rig.step(Duration::from_millis(1));
    assert_eq!(rig.status_rx_prod(), 0);
    assert!(!rig.irq.level());
}

/// The receive thread takes a frame when the guest posts a buffer and
/// publishes its completion on the clock, with nothing polling by hand; stop
/// joins it. Exercises `Device::start` and `stop`, which the cases above
/// replace with explicit steps.
#[test]
fn the_receive_thread_delivers_and_publishes_on_its_own() {
    let mut rig = Rig::new();
    *rig.dev.frozen.lock() = None;
    let frame = broadcast(64, 3);
    rig.rx_open(RX_RING_LEN, 1, 0);
    Device::start(&*rig.dev);
    assert!(Device::is_running(&*rig.dev));
    // A frame offered before any buffer is posted waits at the queue head.
    rig.inject.push(frame.clone()).unwrap();
    rig.dev.wake_rx();
    rig.rx_post_std(1);
    let deadline = Instant::now() + Duration::from_secs(5);
    while rig.status_rx_prod() != 1 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(2));
    }
    Device::stop(&*rig.dev);
    assert!(!Device::is_running(&*rig.dev));
    assert_eq!(rig.status_rx_prod(), 1);
    assert_eq!(rig.mem_read(RX_BUF, 64), frame);
    assert!(rig.line_up());
}

// ---- the rest of the plain-memory qtests ----

impl Rig {
    /// What the receive thread does: take what it can, let the arrival
    /// deadline pass.
    fn pump(&mut self) {
        self.dev.poll();
        self.step(ARRIVAL);
    }
    fn ret_flags_end(&self, slot: u64) -> u16 {
        get16dma(&self.return_descriptor(slot), 0x0c) & 4
    }
}

/// A completed frame ahead of a descriptor read failure stays completed; the
/// failed frame retries from its first descriptor once the address is
/// repaired. Ported from test_transmit_complete_then_fragment_dma_error_retry
/// (both its empty-first and 60-byte-first variants).
#[test]
fn a_frame_with_an_unreadable_second_fragment_retries_after_the_first_completes() {
    for empty_first in [false, true] {
        let mut rig = Rig::new();
        let first: Vec<u8> = (0..60u8).map(|i| i.wrapping_mul(7).wrapping_add(1)).collect();
        let second: Vec<u8> = (0..60u8).map(|i| i.wrapping_mul(5).wrapping_add(3)).collect();
        rig.tx_open(TX_RING_LEN, 1, 0);
        rig.writel(reg::MBX_IRQ0_LO, 0);
        rig.mem_write(TX_BUF, &first);
        rig.mem_write(TX_BUF + 0x100, &second[..28]);
        rig.tx_post(0, tx_bd(TX_BUF, if empty_first { 0 } else { 60 }, TXF_END, 0));
        rig.tx_post(1, tx_bd(TX_BUF + 0x100, 28, 0, 0));
        rig.tx_post(2, tx_bd(UNMAPPED, 32, TXF_END, 0));
        let tag = rig.status_tag();
        rig.writel(reg::MBX_TX_PROD_LO, 3);
        let expect_first = if empty_first { vec![] } else { vec![first.clone()] };
        assert_eq!(rig.sent_frames(), expect_first);
        assert_eq!(rig.status_tx_cons(), 1);
        assert_eq!(rig.status_tag(), tag + 1);
        assert!(rig.line_up());
        assert_eq!(rig.refusals(Refused::DmaFault(DmaWhat::TxPayload)), 1);

        rig.writel(reg::MBX_IRQ0_LO, 0);
        rig.mem_write(TX_BUF + 0x100 + 28, &second[28..]);
        rig.tx_post(2, tx_bd(TX_BUF + 0x100 + 28, 32, TXF_END, 0));
        rig.writel(reg::MBX_TX_PROD_LO, 3);
        assert_eq!(rig.sent_frames(), [second]);
        assert_eq!(rig.status_tx_cons(), 3);
        assert_eq!(rig.status_tag(), tag + 2);
        assert!(rig.line_up());
    }
}

/// The same when it is the next descriptor that cannot be read: the ring's
/// last 16 bytes sit at the end of memory. Ported from
/// test_transmit_complete_then_descriptor_dma_error_retry.
#[test]
fn a_frame_with_an_unreadable_next_descriptor_retries_after_the_first_completes() {
    let mut rig = Rig::new();
    let first: Vec<u8> = (0..60u8).map(|i| i.wrapping_mul(7).wrapping_add(1)).collect();
    let second: Vec<u8> = (0..60u8).map(|i| i.wrapping_mul(5).wrapping_add(3)).collect();
    let initial_ring = MEM_SIZE as u64 - 16;
    let repaired_ring = TX_RING - 16;
    rig.tx_open(TX_RING_LEN, 1, 0);
    rig.ind_mem_write(0x104, initial_ring as u32);
    rig.writel(reg::MBX_IRQ0_LO, 0);
    rig.mem_write(initial_ring, &tx_bd(TX_BUF, 60, TXF_END, 0));
    rig.mem_write(TX_BUF, &first);
    let tag = rig.status_tag();
    rig.writel(reg::MBX_TX_PROD_LO, 2);
    assert_eq!(rig.sent_frames(), [first]);
    assert_eq!(rig.status_tx_cons(), 1);
    assert_eq!(rig.status_tag(), tag + 1);
    assert!(rig.line_up());
    assert_eq!(rig.refusals(Refused::DmaFault(DmaWhat::TxDescriptor)), 1);

    rig.writel(reg::MBX_IRQ0_LO, 0);
    rig.mem_write(repaired_ring, &tx_bd(TX_BUF, 60, TXF_END, 0));
    rig.mem_write(TX_RING, &tx_bd(TX_BUF + 0x100, 60, TXF_END, 0));
    rig.mem_write(TX_BUF + 0x100, &second);
    rig.ind_mem_write(0x104, repaired_ring as u32);
    rig.writel(reg::MBX_TX_PROD_LO, 2);
    assert_eq!(rig.sent_frames(), [second]);
    assert_eq!(rig.status_tx_cons(), 2);
    assert_eq!(rig.status_tag(), tag + 2);
}

/// A count of two with a completed frame, a failed one and a repaired retry:
/// the retried frame's two descriptors report at once, and the budget timer
/// has nothing left to say. Ported from
/// test_transmit_partial_error_descriptor_coalescing.
#[test]
fn a_partly_failed_doorbell_coalesces_by_the_descriptors_that_completed() {
    let mut rig = Rig::new();
    let first: Vec<u8> = (0..60u8).map(|i| i.wrapping_mul(7).wrapping_add(1)).collect();
    let second: Vec<u8> = (0..60u8).map(|i| i.wrapping_mul(5).wrapping_add(3)).collect();
    rig.tx_open(TX_RING_LEN, 2, 50);
    rig.writel(reg::MBX_IRQ0_LO, 0);
    rig.mem_write(TX_BUF, &first);
    rig.tx_post(0, tx_bd(TX_BUF, 60, TXF_END, 0));
    rig.tx_post(1, tx_bd(0, 0, TXF_END, 0));
    rig.tx_post(2, tx_bd(UNMAPPED, 60, TXF_END, 0));
    let tag = rig.status_tag();
    rig.writel(reg::MBX_TX_PROD_LO, 3);
    assert_eq!(rig.sent_frames(), [first]);
    assert_eq!(rig.status_tx_cons(), 2);
    assert_eq!(rig.status_tag(), tag + 1);
    assert!(rig.line_up());
    // The qtest's wait for the log line steps the clock a millisecond at a
    // time, so the budget has run out by the retry.
    rig.step(Duration::from_millis(1));
    rig.writel(reg::MBX_IRQ0_LO, 0);
    rig.mem_write(TX_BUF + 0x100, &second);
    rig.tx_post(2, tx_bd(TX_BUF + 0x100, 60, TXF_END, 0));
    rig.writel(reg::MBX_TX_PROD_LO, 3);
    assert_eq!(rig.sent_frames(), [second]);
    assert_eq!(rig.status_tx_cons(), 3);
    assert_eq!(rig.status_tag(), tag + 2);
    assert!(rig.line_up());
    // The budget gives up on a group that already reported.
    rig.step(Duration::from_micros(1050));
    assert_eq!(rig.status_tx_cons(), 3);
    assert_eq!(rig.status_tag(), tag + 2);
}

/// A reply handed back straight after the transmit sees the transmit
/// completion already published. The qtest runs an ARP exchange through
/// QEMU's user-mode network; here the backend queue plays the gateway. Adapted
/// from test_user_backend_reply_waits_for_transmit.
#[test]
fn a_reply_to_a_transmit_finds_the_transmit_already_reported() {
    let mut rig = Rig::new();
    let mut arp_request = vec![0xff; 6];
    arp_request.extend_from_slice(&[0x52, 0x54, 0, 0x12, 0x34, 0x56, 8, 6, 0, 1, 8, 0, 6, 4, 0, 1]);
    arp_request.extend_from_slice(&[0x52, 0x54, 0, 0x12, 0x34, 0x56, 10, 0, 2, 15, 0, 0, 0, 0, 0, 0, 10, 0, 2, 2]);
    assert_eq!(arp_request.len(), 42);
    rig.writel(reg::MAC_ADDR_0_HIGH, 0x5254);
    rig.writel(reg::MAC_ADDR_0_LOW, 0x0012_3456);
    rig.tx_open(TX_RING_LEN, 2, 1_000_000);
    rig.rx_open(RX_RING_LEN, 1, 0);
    rig.rx_post_std(1);
    rig.tx_send_one(0, &arp_request, 0);
    assert_eq!(rig.sent_frames(), [arp_request.clone()]);
    // The gateway's reply, to the address just programmed.
    let mut reply = arp_request[6..12].to_vec();
    reply.extend_from_slice(&[0x52, 0x55, 10, 0, 2, 2, 8, 6, 0, 1, 8, 0, 6, 4, 0, 2]);
    reply.extend_from_slice(&[0x52, 0x55, 10, 0, 2, 2, 10, 0, 2, 2]);
    reply.extend_from_slice(&arp_request[22..32]);
    reply.resize(60, 0);
    rig.deliver(&reply);
    rig.step(ARRIVAL);
    assert_eq!(rig.status_tx_cons(), 1);
    assert_eq!(rig.status_rx_prod(), 1);
    assert!(rig.line_up());
    assert_eq!(rig.mem_read(RX_BUF, 60), reply);
}

/// A frame bigger than the standard buffer, with no jumbo ring, must not
/// write past the buffer. Ported from test_receive_overruns_nothing.
#[test]
fn an_oversize_frame_never_writes_past_the_posted_buffer() {
    let mut rig = Rig::new();
    let frame = broadcast(4096, 7);
    rig.rx_open(RX_RING_LEN, 1, 0);
    rig.rx_post_std(1);
    rig.mem_fill(RX_BUF, 4096, 0xa5);
    rig.deliver(&frame);
    rig.step(ARRIVAL);
    assert_eq!(rig.mem_read(RX_BUF + u64::from(STD_BUF_SIZE), 4096 - STD_BUF_SIZE as usize), vec![0xa5; 4096 - 1536]);
}

fn lap_open(rig: &Rig, coal_bds: u32) {
    rig.rx_open(8, coal_bds, 0);
    rig.writel(reg::RX_STD_RCB_MAXLEN_FLAGS, 1518 << 16);
    rig.rx_post_std(64);
}

/// Five trips round an eight-entry return ring, each completion visible in
/// its slot before the driver acknowledges it. Ported from
/// test_receive_ring_laps.
#[test]
fn the_return_ring_laps_cleanly() {
    let mut rig = Rig::new();
    let frame = broadcast(64, 2);
    lap_open(&rig, 0);
    for i in 0..40u32 {
        let idx = u64::from(i % 8);
        rig.mem_fill(RET_RING + idx * 32, 32, 0);
        rig.deliver(&frame);
        rig.step(ARRIVAL);
        assert_eq!(rig.ret_flags_end(idx), 4, "frame {i}");
        assert_eq!(get16dma(&rig.return_descriptor(idx), 0x08), 68);
        rig.writel(reg::MBX_RX_CONS0_LO, (i + 1) % 8);
        rig.writel(reg::MBX_RX_STD_PROD_LO, (64 + i + 1) % 512);
    }
}

/// A full return ring holds frames back rather than overwrite completions the
/// driver owes; freeing space with a consumer write releases them, without
/// another interrupt acknowledgement. Ported from
/// test_receive_ring_full_recovers.
#[test]
fn a_full_return_ring_holds_frames_until_the_driver_frees_it() {
    let mut rig = Rig::new();
    let frame = broadcast(64, 3);
    lap_open(&rig, 1);
    for _ in 0..20 {
        rig.deliver(&frame);
    }
    rig.step(ARRIVAL * 4);
    // Seven fit; the last slot stays clear and the rest wait in the backend.
    assert_eq!(rig.status_rx_prod(), 7);
    assert_eq!(rig.ret_flags_end(7), 0);
    assert_eq!(rig.waiting(), 13);
    rig.writel(reg::MBX_IRQ0_LO, 0);
    assert!(!rig.line_up());
    rig.step(ARRIVAL);
    assert_eq!(rig.ret_flags_end(7), 0);
    for i in 0..8 {
        rig.mem_fill(RET_RING + i * 32, 32, 0);
    }
    rig.writel(reg::MBX_RX_CONS0_LO, 7);
    rig.pump();
    // The retry fills the ring a second time, wrapping to slot 6.
    assert_eq!(rig.status_rx_prod(), 6);
    assert_eq!(rig.ret_flags_end(6), 0);
    for i in 0..8 {
        rig.mem_fill(RET_RING + i * 32, 32, 0);
    }
    rig.writel(reg::MBX_RX_CONS0_LO, 6);
    rig.pump();
    assert_eq!(rig.status_rx_prod(), 4);
}

/// A halted receiver takes nothing and keeps nothing: frames from before it
/// is enabled are gone. Ported from test_receive_mac_enable.
#[test]
fn frames_offered_to_a_halted_receiver_are_not_kept() {
    let mut rig = Rig::new();
    let frame = broadcast(64, 11);
    rig.rx_open(8, 1, 0);
    rig.writel(reg::RX_MODE, 0);
    rig.rx_post_std(8);
    assert_eq!(rig.readl(reg::RX_MODE) & RXMODE_ENABLE, 0);
    for _ in 0..4 {
        rig.deliver(&frame);
    }
    rig.step(ARRIVAL);
    assert_eq!(rig.status_rx_prod(), 0);
    assert_eq!(rig.waiting(), 0);
    rig.writel(reg::RX_MODE, RXMODE_ENABLE);
    rig.pump();
    assert_eq!(rig.status_rx_prod(), 0);
    assert_eq!(rig.ret_flags_end(0), 0);
    let fresh = broadcast(64, 17);
    rig.deliver(&fresh);
    rig.step(ARRIVAL);
    assert_eq!(rig.status_rx_prod(), 1);
    assert!(rig.line_up());
    assert_eq!(get16dma(&rig.return_descriptor(0), 0x08), 68);
    // Halting again is not a bring-up special case.
    rig.writel(reg::RX_MODE, 0);
    rig.deliver(&broadcast(64, 19));
    rig.step(ARRIVAL);
    assert_eq!(rig.status_rx_prod(), 1);
    rig.writel(reg::RX_MODE, RXMODE_ENABLE);
    rig.pump();
    assert_eq!(rig.status_rx_prod(), 1);
    rig.deliver(&broadcast(64, 23));
    rig.step(ARRIVAL);
    assert_eq!(rig.status_rx_prod(), 2);
    assert_eq!(rig.ret_flags_end(1), 4);
}

/// A frame offered before any buffer is posted waits; posting one releases
/// it. Ported from test_queued_receive_released_by_producer.
#[test]
fn a_waiting_frame_is_released_by_the_producer_mailbox() {
    let mut rig = Rig::new();
    let frame = broadcast(64, 29);
    rig.rx_open(RX_RING_LEN, 1, 0);
    rig.mem_write(STD_RING, &rx_bd(RX_BUF, 0));
    rig.deliver(&frame);
    rig.step(ARRIVAL);
    assert_eq!(rig.status_rx_prod(), 0);
    assert_eq!(rig.ret_flags_end(0), 0);
    assert_eq!(rig.waiting(), 1);
    rig.writel(reg::MBX_RX_STD_PROD_LO, 1);
    rig.pump();
    assert_eq!(rig.status_rx_prod(), 1);
    assert_eq!(rig.mem_read(RX_BUF, 64), frame);
    assert_eq!(get16dma(&rig.return_descriptor(0), 0x0a), 0);
}

/// A frame queued before a reset must not reach the rings brought up after
/// it, by chip reset or machine reset. Ported from test_receive_queue_reset.
#[test]
fn a_reset_discards_frames_the_backend_was_holding() {
    for machine_reset in [false, true] {
        let mut rig = Rig::new();
        let stale = broadcast(64, 0x51);
        let fresh = broadcast(64, 0xa7);
        // Nothing is posted yet, so the stale frame waits at the head.
        rig.rx_open(RX_RING_LEN, 1, 0);
        rig.deliver(&stale);
        assert_eq!(rig.waiting(), 1);
        if machine_reset {
            rig.dev.reset();
        } else {
            rig.writel(reg::MISC_CFG, MISC_CFG_RESET);
        }
        assert_eq!(rig.waiting(), 0);
        rig.rx_open(RX_RING_LEN, 1, 0);
        rig.rx_post_std(1);
        rig.deliver(&fresh);
        rig.step(ARRIVAL);
        assert_eq!(rig.status_rx_prod(), 1);
        assert_eq!(rig.mem_read(RX_BUF, 64), fresh);
    }
}

/// An unreadable posted buffer defers the frame, leaves the descriptor
/// unconsumed, and the repaired buffer delivers it. Derived from the receive
/// payload read in the model; the qtest of the same subject is
/// test_receive_payload_dma_error, whose log-marker checks are not ported.
#[test]
fn an_unreachable_receive_buffer_defers_the_frame() {
    let mut rig = Rig::new();
    let frame = broadcast(64, 3);
    rig.rx_open(RX_RING_LEN, 1, 0);
    rig.mem_write(STD_RING, &rx_bd(UNMAPPED, 0));
    rig.writel(reg::MBX_RX_STD_PROD_LO, 1);
    rig.deliver(&frame);
    assert_eq!(rig.waiting(), 1);
    assert_eq!(rig.status_rx_prod(), 0);
    assert_eq!(rig.refusals(Refused::DmaFault(DmaWhat::RxPayload)), 1);
    rig.mem_write(STD_RING, &rx_bd(RX_BUF, 0));
    rig.writel(reg::MBX_RX_STD_PROD_LO, 1);
    rig.pump();
    assert_eq!(rig.status_rx_prod(), 1);
    assert_eq!(rig.mem_read(RX_BUF, 64), frame);
    assert_eq!(get16dma(&rig.return_descriptor(0), 0x0a), 0);
}

/// With neither criterion armed the receive side stays silent however long
/// the ring is idle, and so does the transmit side. Ported from
/// test_receive_without_a_criterion_is_silent and
/// test_transmit_without_a_criterion_is_silent.
#[test]
fn without_a_criterion_no_update_is_ever_posted() {
    let mut rig = Rig::new();
    rig.rx_open(64, 0, 0);
    rig.rx_post_std(1);
    rig.deliver(&broadcast(64, 3));
    rig.step(Duration::from_secs(1));
    assert_eq!(rig.ret_flags_end(0), 4);
    assert_eq!(be32(&rig.status_block(), 0), 0);
    assert_eq!(rig.status_rx_prod(), 0);
    assert!(!rig.line_up());

    let mut rig = Rig::new();
    let frame = broadcast(64, 5);
    rig.tx_open(64, 0, 0);
    rig.writel(reg::MBX_IRQ0_LO, 0);
    rig.tx_send_one(0, &frame, 0);
    assert_eq!(rig.sent_frames().len(), 1);
    rig.step(Duration::from_secs(1));
    assert_eq!(be32(&rig.status_block(), 0), 0);
    assert_eq!(rig.status_tx_cons(), 0);
    assert!(!rig.line_up());
}

/// A budget that has run out reports the next event as it happens, in either
/// direction. Ported from test_receive_expired_timer_reports_at_once and
/// test_transmit_expired_timer_reports_at_once.
#[test]
fn an_expired_budget_reports_the_next_event_at_once() {
    let mut rig = Rig::new();
    rig.rx_open(64, 8, 50);
    rig.rx_post_std(2);
    rig.step(Duration::from_micros(200));
    assert_eq!(rig.status_tag(), 0);
    rig.deliver(&broadcast(64, 3));
    rig.step(ARRIVAL);
    assert_eq!(rig.status_rx_prod(), 1);
    assert_eq!(rig.status_tag(), 1);

    let mut rig = Rig::new();
    rig.tx_open(64, 8, 50);
    rig.step(Duration::from_micros(200));
    assert_eq!(rig.status_tag(), 0);
    rig.tx_send_one(0, &broadcast(64, 3), 0);
    assert_eq!(rig.sent_frames().len(), 1);
    assert_eq!(rig.status_tag(), 1);
}

/// The producer the status block names follows its completion: the
/// descriptor is written, the next slot is still empty. Ported from
/// test_receive_producer_follows_its_completion.
#[test]
fn the_receive_producer_names_only_completed_descriptors() {
    let mut rig = Rig::new();
    rig.rx_open(64, 1, 0);
    rig.rx_post_std(3);
    for i in 0..3u64 {
        rig.deliver(&broadcast(64, 0x55));
        rig.step(ARRIVAL);
        assert_eq!(u64::from(rig.status_rx_prod()), i + 1);
        let bd = rig.return_descriptor(i);
        assert_eq!(get16dma(&bd, 0x0c) & 4, 4);
        assert_eq!(get16dma(&bd, 0x08), 68);
        let next = rig.return_descriptor(i + 1);
        assert_eq!(get16dma(&next, 0x08), 0);
        assert_eq!(get16dma(&next, 0x0c), 0);
    }
}

/// A receive update the bus refused is retried by the budget timer once the
/// address is repaired. Ported from test_receive_coalescing_status_dma_retry.
#[test]
fn a_refused_receive_update_is_retried_by_the_budget_timer() {
    let mut rig = Rig::new();
    rig.rx_open(RX_RING_LEN, 1, 50);
    rig.split_addr(reg::HCC_STATUSBLK_ADDR_HI, reg::HCC_STATUSBLK_ADDR_LO, UNMAPPED);
    rig.rx_post_std(1);
    rig.deliver(&broadcast(64, 0x6d));
    rig.step(ARRIVAL);
    assert_eq!(rig.ret_flags_end(0), 4);
    assert_eq!(rig.status_rx_prod(), 0);
    assert!(!rig.line_up());
    rig.split_addr(reg::HCC_STATUSBLK_ADDR_HI, reg::HCC_STATUSBLK_ADDR_LO, STATUS);
    rig.step(Duration::from_micros(1051));
    assert_eq!(rig.status_rx_prod(), 1);
    assert_eq!(rig.status_tag(), 1);
    assert!(rig.line_up());
}

fn cksum_rx_frame() -> Vec<u8> {
    let mut f = cksum_frame(0x45);
    f[..6].fill(0xff);
    seal_tcp(&mut f);
    f
}
fn seal_tcp(f: &mut [u8]) {
    f[24..26].fill(0);
    let ip = inet_sum(&f[14..34], 0);
    f[24..26].copy_from_slice(&ip.to_be_bytes());
    f[50..52].fill(0);
    let tcp = inet_sum(&f[34..62], tcp_pseudo(f, 28));
    f[50..52].copy_from_slice(&tcp.to_be_bytes());
}
fn seal_udp(f: &mut [u8]) {
    f[23] = 17;
    f[16] = 0;
    f[17] = 48;
    f[38] = 0;
    f[39] = 28;
    f[40..42].fill(0);
    f[24..26].fill(0);
    let ip = inet_sum(&f[14..34], 0);
    f[24..26].copy_from_slice(&ip.to_be_bytes());
    let pseudo: u32 = f[26..34].chunks(2).map(|p| u32::from(p[0]) << 8 | u32::from(p[1])).sum::<u32>() + 17 + 28;
    let udp = inet_sum(&f[34..62], pseudo);
    f[40..42].copy_from_slice(&udp.to_be_bytes());
}
fn inet_fix_ip(f: &mut [u8]) {
    f[24..26].fill(0);
    let ip = inet_sum(&f[14..34], 0);
    f[24..26].copy_from_slice(&ip.to_be_bytes());
}

/// One row of test_receive_checksum_case: the flags, the IP sum and the
/// transport sum a frame earns under a MODE_CTL setting.
fn rx_checksum_case(frame: &[u8], mode: u32, flags: u16, ip: u16, l4: u16, label: &str) {
    let mut rig = Rig::new();
    rig.rx_open(RX_RING_LEN, 1, 0);
    rig.rx_post_std(1);
    rig.ind_reg_write(reg::MODE_CTL, MODE_ALL_SWAPS | mode);
    rig.deliver(frame);
    rig.step(ARRIVAL);
    assert_eq!(rig.status_rx_prod(), 1, "{label}");
    let ret = rig.return_descriptor(0);
    let got = get16dma(&ret, 0x0c);
    assert_eq!(got, flags | 4, "{label} flags");
    assert_eq!(got & 3, 0, "{label}");
    assert_eq!(get16dma(&ret, 0x12), ip, "{label} ip sum");
    assert_eq!(get16dma(&ret, 0x10), l4, "{label} transport sum");
}

/// Ported from test_receive_checksum_offload, _udp, _udp_zero,
/// _malformed_udp_length, _bad_ip, _bad_transport, _no_pseudo, _padding,
/// _fragment, _later_fragment, _malformed_tcp_header, _malformed_length and
/// _raw_mode.
#[test]
fn the_receive_checksum_report_covers_every_malformed_case() {
    let tcp = cksum_rx_frame();
    rx_checksum_case(&tcp, 0, 0x7000, 0xffff, 0xffff, "tcp");

    let mut udp = cksum_frame(0x45);
    udp[..6].fill(0xff);
    seal_udp(&mut udp);
    rx_checksum_case(&udp, 0, 0x3000, 0xffff, 0xffff, "udp");

    // UDP with the checksum omitted: IP only, and the sum is 0xffff.
    let mut udp_zero = udp.clone();
    udp_zero[42..44].copy_from_slice(&[0x62, 0x6a]);
    udp_zero[40..42].fill(0);
    rx_checksum_case(&udp_zero, 0, 0x1000, 0xffff, 0xffff, "udp, no checksum");

    let mut bad_udp_len = udp.clone();
    bad_udp_len[39] = 27;
    rx_checksum_case(&bad_udp_len, 0, 0x1000, 0xffff, 0, "udp length");

    let mut bad_ip = tcp.clone();
    bad_ip[22] -= 1;
    rx_checksum_case(&bad_ip, 0, 0x6000, 0xfeff, 0xffff, "bad ip");

    let mut bad_tcp = tcp.clone();
    bad_tcp[51] += 1;
    rx_checksum_case(&bad_tcp, 0, 0x5000, 0xffff, 0x0001, "bad transport");

    rx_checksum_case(&tcp, 1 << 23, 0x5000, 0xffff, !inet_sum(&tcp[34..62], 0), "no pseudo header");

    let mut padded = tcp.clone();
    padded.extend_from_slice(&[0xa5; 8]);
    rx_checksum_case(&padded, 0, 0x7000, 0xffff, 0xffff, "padding");

    let mut fragment = tcp.clone();
    fragment[20] = 0x20;
    inet_fix_ip(&mut fragment);
    rx_checksum_case(&fragment, 0, 0x5000, 0xffff, 0, "more fragments");

    let mut later = tcp.clone();
    later[21] = 1;
    inet_fix_ip(&mut later);
    rx_checksum_case(&later, 0, 0x5000, 0xffff, 0, "later fragment");

    let mut short_header = tcp.clone();
    short_header[46] = 0x40;
    rx_checksum_case(&short_header, 0, 0x5000, 0xffff, 0, "tcp data offset");

    let mut bad_length = tcp.clone();
    bad_length[16..18].copy_from_slice(&[1, 0]);
    rx_checksum_case(&bad_length, 0, 0, 0, 0, "ip length");

    rx_checksum_case(&tcp, 1 << 9, 0, 0, 0, "raw mode");
}
