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

struct Rig {
    dev: Arc<Bcm5701>,
    irq: Arc<LevelProbe>,
}

impl Rig {
    fn new() -> Self {
        Self::with(Bcm5701Config { mac: MAC, eeprom: None })
    }

    fn with(config: Bcm5701Config) -> Self {
        let dev = Bcm5701::new(config).unwrap();
        *dev.frozen.lock() = Some(Instant::now());
        let (irq, line) = LevelProbe::line();
        dev.connect_interrupt(line);
        dev.reset();
        Self { dev, irq }
    }

    fn step(&self, by: Duration) {
        let mut t = self.dev.frozen.lock();
        *t = Some(t.unwrap() + by);
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
        self.dev.config().read(reg)
    }
    fn cfg_write(&self, reg: usize, val: u32) {
        self.dev.config().write(reg, val);
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
    assert_eq!(rig.cfg_read(0x08 / 4) & 0xff, 0);
    assert_eq!(rig.dev.config().interrupt_pin(), PCI_INTA);
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
    rig.dev.req64_sampled(true);
    assert_eq!(rig.cfg_read(0x70 / 4), 0x06);
}

fn eeprom_rig(image: &[u8], device_id: u8) -> Rig {
    Rig::with(Bcm5701Config { mac: MAC, eeprom: Some(Eeprom::from_image(device_id, image.to_vec()).unwrap()) })
}

/// Ported from test_eeprom_bar_read_write: Start cannot replace an
/// unacknowledged completion, acknowledge-and-start works in one write,
/// transfers stay word aligned, writes reach only their word, and Reset wins
/// over Start.
#[test]
fn the_eeprom_controller_reads_and_writes_aligned_words() {
    let path = temp_file("bar-rw", &[0x66, 0x99, 0x55, 0xaa, 0x12, 0x34, 0x56, 0x78]);
    let rig = Rig::with(Bcm5701Config { mac: MAC, eeprom: Some(Eeprom::from_file(3, path.clone()).unwrap()) });
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
    assert_eq!(rig.refusals(Refused::EepromFault(Fault::WrongDevice(0))), 1);
}

/// Ported from test_eeprom_read_only_backing.
#[test]
fn a_read_only_eeprom_refuses_writes() {
    let mut eeprom = Eeprom::from_image(3, vec![0x66, 0x99, 0x55, 0xaa]).unwrap();
    eeprom.set_read_only(true);
    assert!(eeprom.is_read_only());
    let rig = Rig::with(Bcm5701Config { mac: MAC, eeprom: Some(eeprom) });
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
    assert!(Bcm5701::new(Bcm5701Config { mac: [1, 0, 0, 0, 0, 0], eeprom: None }).is_err());
}

/// Direct serial-pin control is unsupported and observable, not faked.
#[test]
fn direct_eeprom_pin_control_is_refused() {
    let rig = Rig::new();
    rig.writel(reg::EEPROM_PIN, 0x15);
    assert_eq!(rig.readl(reg::EEPROM_PIN), 0x15);
    assert_eq!(rig.refusals(Refused::EepromPins), 1);
}
