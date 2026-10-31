//! The chip's register file, on-chip SRAM, interrupt line and the engines
//! that run inside register writes. The PCI side and the threads are in the
//! parent module.

use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use super::eeprom::{self, Eeprom};
use super::phy::{self, Phy};
use super::dma::{Doorbells, HeldCompletion, Ring, Timer};
use crate::cpu::smp::MAX_CPUS;
use crate::cpu::mips_core::CyclesPtr;
use crate::pci::PciDma;
use super::Bcm5701;

pub const REG_FILE_SIZE: u32 = 0x8000;
pub const SRAM_SIZE: u32 = 0x8000;
const REG_WORDS: usize = (REG_FILE_SIZE / 4) as usize;
const SRAM_WORDS: usize = (SRAM_SIZE / 4) as usize;

/// Dword indices of the PCI configuration registers the chip owns.
pub mod cfg {
    pub const MISC_CTL: usize = 0x68 / 4;
    pub const PCISTATE: usize = 0x70 / 4;
    pub const REG_BASEADDR: usize = 0x78 / 4;
    pub const MEMWIN_BASEADDR: usize = 0x7c / 4;
    pub const REG_DATA: usize = 0x80 / 4;
    pub const MEMWIN_DATA: usize = 0x84 / 4;
}

pub mod misc_ctl {
    pub const CLEAR_INTA: u32 = 0x1;
    pub const MASK_PCI_INTR: u32 = 0x2;
    pub const ENDIAN_BYTESWAP: u32 = 0x4;
    pub const ENDIAN_WORDSWAP: u32 = 0x8;
    /// The chip id, bits 31:16, is read-only; the low half is writable.
    pub const WRITABLE: u32 = 0xffff;
}

pub mod pcistate {
    pub const INTR_NOT_ACTIVE: u32 = 0x2;
    pub const BUSMODE_PCI: u32 = 0x4;
    /// The chip did not see REQ64# at the end of reset.
    pub const BUS_32BIT: u32 = 0x10;
}

/// The chip id MISC_CTL[31:16] presents; the driver reads the part from it.
pub const CHIPID_BCM5701: u32 = 0x0105;

/// Register-file offsets with modelled behaviour.
pub mod reg {
    pub const MAC_MODE: u32 = 0x0400;
    pub const MAC_STATUS: u32 = 0x0404;
    pub const MAC_ADDR_0_HIGH: u32 = 0x0410;
    pub const MAC_ADDR_0_LOW: u32 = 0x0414;
    pub const MAC_ADDR_STRIDE: u32 = 8;
    pub const MI_COMM: u32 = 0x044c;
    pub const MI_STS: u32 = 0x0450;
    pub const RX_MODE: u32 = 0x0468;
    pub const HCC_MODE: u32 = 0x3c00;
    pub const BUFMGR_MODE: u32 = 0x4400;
    pub const BUFMGR_STATUS: u32 = 0x4404;
    pub const FW_PROGRESS: u32 = 0x5034;
    pub const MODE_CTL: u32 = 0x6800;
    pub const MISC_CFG: u32 = 0x6804;
    pub const EEPROM_ADDR: u32 = 0x6838;
    pub const EEPROM_DATA: u32 = 0x683c;
    pub const EEPROM_PIN: u32 = 0x6840;
    pub const EEPROM_DELAY: u32 = 0x6848;
    pub const MBX_IRQ0_LO: u32 = 0x0204;
    pub const MBX_RX_STD_PROD_LO: u32 = 0x026c;
    pub const MBX_RX_JUMBO_PROD_LO: u32 = 0x0274;
    pub const MBX_RX_CONS0_LO: u32 = 0x0284;
    pub const MBX_TX_PROD_LO: u32 = 0x0304;
    pub const HCC_RX_COAL_TICKS: u32 = 0x3c08;
    pub const HCC_TX_COAL_TICKS: u32 = 0x3c0c;
    pub const HCC_RX_MAX_COAL_BDS: u32 = 0x3c10;
    pub const HCC_TX_MAX_COAL_BDS: u32 = 0x3c14;
    pub const HCC_STATUSBLK_ADDR_HI: u32 = 0x3c38;
    pub const HCC_STATUSBLK_ADDR_LO: u32 = 0x3c3c;
    pub const RX_JUMBO_RCB_HADDR_HI: u32 = 0x2440;
    pub const RX_JUMBO_RCB_HADDR_LO: u32 = 0x2444;
    pub const RX_JUMBO_RCB_MAXLEN_FLAGS: u32 = 0x2448;
    pub const RX_STD_RCB_HADDR_HI: u32 = 0x2450;
    pub const RX_STD_RCB_HADDR_LO: u32 = 0x2454;
    pub const RX_STD_RCB_MAXLEN_FLAGS: u32 = 0x2458;
    pub const RX_STD_REPLENISH_THRESH: u32 = 0x2c18;
    pub const RX_JUMBO_REPLENISH_THRESH: u32 = 0x2c1c;
}

/// Bit 0 of a mode register is a self-clearing reset; bit 0 of MISC_CFG
/// resets the whole chip.
const MODE_RESET: u32 = 1;
const MISCCFG_RESET_CORE_CLOCKS: u32 = 1;
const RXMODE_ENABLE: u32 = 2;
const RCB_FLAG_EXT_RX_ENABLE: u32 = 1;

const BUFMGR_MODE_ENABLE: u32 = 2;
/// The buffer manager's enable reads back only after its startup; a
/// functional approximation inside the driver's 10 ms polling budget.
const BUFMGR_START: Duration = Duration::from_millis(8);

mod eeprom_cmd {
    pub const READ: u32 = 1 << 31;
    pub const COMPLETE: u32 = 1 << 30;
    pub const RESET: u32 = 1 << 29;
    pub const DEVICE_ID: u32 = 7 << 26;
    pub const START: u32 = 1 << 25;
    pub const WORD_ADDR: u32 = 0x3fff << 2;
}

/// The firmware-ready handshake. The driver writes the magic to SRAM 0x0b50
/// and polls for its complement, the on-chip bootcode's ready signal. The
/// mailbox returns the complement of the last value written, so the magic
/// reads back as ready whether the driver writes it before the reset or after.
pub const SRAM_FW_MBOX: u32 = 0x0b50;
pub const FW_MAGIC: u32 = 0x4b65_7654;
/// The firmware-progress register holds the all-ones "no progress yet"
/// sentinel, so a driver polling it sees "unchanged" and keeps polling the
/// mailbox.
const FW_PROGRESS_INIT: u32 = 0xffff_ffff;

/// The station address the on-chip bootcode leaves where a driver looks for
/// it: SRAM 0x0c14 tagged 0x484b in its high half, and MAC address filter
/// slot 0.
const SRAM_MAC_HI: u32 = 0x0c14;
const SRAM_MAC_LO: u32 = 0x0c18;
const MAC_TAG: u32 = 0x484b;

mod mi {
    pub const DATA: u32 = 0x0000_ffff;
    pub const REG: u32 = 0x001f_0000;
    pub const REG_SHIFT: u32 = 16;
    /// Command and address fields, kept across a read.
    pub const READ_FIELDS: u32 = 0x0fff_0000;
    pub const WRITE: u32 = 0x0400_0000;
    pub const READ: u32 = 0x0800_0000;
    pub const BUSY: u32 = 0x2000_0000;
    pub const LINK: u32 = 1;
}

/// What the model could not carry out, counted and logged on first sight so
/// an unexpected access is never silent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Refused {
    /// A write to a register with no modelled behaviour; the value is stored
    /// and reads back.
    UnmodelledWrite(u32),
    /// A write to the read-only buffer-manager status.
    BufmgrStatusWrite,
    /// A vital product data write; the storage is not written.
    VpdWrite,
    /// Direct EEPROM serial-pin control.
    EepromPins,
    EepromFault(eeprom::Fault),
    /// An MI_COMM opcode other than read and write.
    MiOpcode(u32),
    /// A BAR access of a width or alignment the chip does not accept.
    BarAccess { offset: u32, width: u8, write: bool },
    Phy(phy::Refusal),
    /// A transfer to or from host memory that did not complete.
    DmaFault(super::dma::DmaWhat),
    /// A ring index written to a mailbox that lies outside the ring.
    ProducerOutOfRange(super::dma::Mailbox),
    /// A transmit frame larger than the model carries; it is dropped.
    TxFrameTooLarge,
    /// A received frame larger than the model carries; it is dropped.
    RxFrameTooLarge,
    /// A frame discarded for want of a posted descriptor, once per drought.
    RxShortage(Ring),
    /// A frame too big for the ring it would use; it is dropped.
    RxNoRoom(Ring),
    /// The jumbo ring asks for 64-byte descriptors, which are not modelled;
    /// the ring is treated as unavailable.
    JumboExtendedDescriptors,
}

/// The chip's view of its own configuration header, which the bus owns: the
/// device holds it in its function state, and the state holds the device
/// weakly.
pub(super) struct Header(Weak<Bcm5701>);

impl Header {
    pub(super) fn new(device: Weak<Bcm5701>) -> Self {
        Self(device)
    }

    /// MISC_CTL as the guest and the bus have left it.
    fn misc_ctl(&self) -> u32 {
        self.0.upgrade().map_or(0, |dev| dev.config_header().read(cfg::MISC_CTL))
    }

    /// Clear `bits` of MISC_CTL, which the chip does itself.
    fn clear_misc_ctl(&self, bits: u32) {
        if let Some(dev) = self.0.upgrade() {
            dev.config_header().write_masked(cfg::MISC_CTL, 0, bits);
        }
    }
}

pub(super) struct State {
    pub cfg: Header,
    /// PCISTATE, which the chip drives: the bus mode, whether it saw
    /// REQ64# at the end of reset, and whether the interrupt is down.
    pub pcistate: u32,
    /// Whether the slot drives REQ64# as reset ends.
    req64: bool,
    pub mac: [u8; 6],
    pub(super) regfile: Box<[u32; REG_WORDS]>,
    sram: Box<[u32; SRAM_WORDS]>,
    /// The two indirect-window base registers (config 0x78 and 0x7c).
    pub reg_win_base: u32,
    pub mem_win_base: u32,
    eeprom: Option<Eeprom>,
    bufmgr_start: Option<Instant>,
    /// The mask set through the 0x0204 mailbox, and the latch a status-block
    /// update sets and the driver clears. The line is the latch with
    /// neither mask (this or MISC_CTL's) set.
    pub(super) irq_masked: bool,
    pub(super) irq_pending: bool,
    irq_level: bool,
    pub irq_events: Vec<bool>,
    pub phy: Phy,
    pub now: Instant,
    refused: BTreeMap<Refused, u64>,
    pub(super) dma: Arc<dyn PciDma>,
    /// The send-ring consumer, and what the status block reports.
    pub(super) tx_cons: u16,
    pub(super) status_tx_cons: u16,
    pub(super) status_rx_prod: u16,
    /// Where the next return descriptor goes; it runs ahead of
    /// `status_rx_prod` while the arrival line holds a completion.
    pub(super) rx_prod_staged: u16,
    /// The next descriptor a frame takes from each producer ring, and the
    /// next one the chip fetches into its descriptor cache; the status block
    /// reports the second.
    pub(super) rx_std_cons: u16,
    pub(super) rx_jumbo_cons: u16,
    pub(super) rx_std_fetched: u16,
    pub(super) rx_jumbo_fetched: u16,
    /// Frames discarded since a ring's shortage began, so a drought is
    /// reported at both ends rather than once per frame.
    pub(super) rx_shortage_drops: u64,
    pub(super) rx_std_shortage_drops: u64,
    pub(super) status_tag: u8,
    /// Descriptors consumed on each ring since the last status-block update,
    /// and whether each coalescing budget has run out.
    pub(super) rx_coal_pending: u16,
    pub(super) tx_coal_pending: u16,
    pub(super) rx_coal_expired: bool,
    pub(super) tx_coal_expired: bool,
    pub(super) rx_coal_timer: Option<Instant>,
    pub(super) tx_coal_timer: Option<Instant>,
    /// Completions held until their frame's arrival deadline.
    pub(super) rx_latency: VecDeque<HeldCompletion>,
    pub(super) rx_floor: Duration,
    /// The same minimum, counted in cycles of the processor that rang the
    /// send doorbell.
    pub(super) rx_floor_cycles: u64,
    /// Each processor's cycle counter, by CPU number, once the machine has
    /// wired them. Without them the floor is host time alone.
    pub(super) cpu_cycles: Vec<CyclesPtr>,
    /// Each processor's cycle count when it last rang the send doorbell.
    pub(super) tx_doorbell: Doorbells,
    /// Frames the transmit engine produced, for the owner to send.
    pub wire: Vec<Vec<u8>>,
    /// Set when something the guest did can unblock a waiting frame.
    pub wake_rx: bool,
    /// A reset discards the frames the backend still holds for the guest.
    pub purge_rx: bool,
}

impl State {
    pub fn new(
        cfg: Header,
        req64: bool,
        mac: [u8; 6],
        eeprom: Option<Eeprom>,
        rx_floor: Duration,
        rx_floor_cycles: u64,
        now: Instant,
    ) -> Self {
        let mut st = Self {
            cfg,
            pcistate: Self::pcistate_at_reset(req64),
            req64,
            mac,
            regfile: Box::new([0; REG_WORDS]),
            sram: Box::new([0; SRAM_WORDS]),
            reg_win_base: 0,
            mem_win_base: 0,
            eeprom,
            bufmgr_start: None,
            irq_masked: true,
            irq_pending: false,
            irq_level: false,
            irq_events: Vec::new(),
            phy: Phy::new(now),
            now,
            refused: BTreeMap::new(),
            dma: Arc::new(super::dma::NullDma),
            tx_cons: 0,
            status_tx_cons: 0,
            status_rx_prod: 0,
            rx_prod_staged: 0,
            rx_std_cons: 0,
            rx_jumbo_cons: 0,
            rx_std_fetched: 0,
            rx_jumbo_fetched: 0,
            rx_shortage_drops: 0,
            rx_std_shortage_drops: 0,
            status_tag: 0,
            rx_coal_pending: 0,
            tx_coal_pending: 0,
            rx_coal_expired: false,
            tx_coal_expired: false,
            rx_coal_timer: None,
            tx_coal_timer: None,
            rx_latency: VecDeque::new(),
            rx_floor,
            rx_floor_cycles,
            cpu_cycles: Vec::new(),
            tx_doorbell: [None; MAX_CPUS],
            wire: Vec::new(),
            wake_rx: false,
            purge_rx: false,
        };
        st.reset();
        st
    }

    /// Enter at `now`: run whatever came due since the last entry.
    pub fn enter(&mut self, now: Instant) {
        if now > self.now {
            self.now = now;
        }
        self.run_timers();
    }

    /// Run whatever came due, earliest first, each at its own deadline: the
    /// virtual clock the model is ported from reaches a timer's expiry before
    /// it fires, so a timer that arms another counts from there, not from
    /// whenever this entry happened.
    fn run_timers(&mut self) {
        let target = self.now;
        while let Some((deadline, timer)) = self.timers().into_iter().filter(|(d, _)| *d <= target).min() {
            self.now = deadline;
            match timer {
                Timer::PhySettle => self.phy.sync(deadline),
                Timer::BufferManager => {
                    self.bufmgr_start = None;
                    let mode = self.rf(reg::BUFMGR_MODE);
                    self.set_rf(reg::BUFMGR_MODE, mode | BUFMGR_MODE_ENABLE);
                }
                Timer::RxCoalescing => self.rx_coal_timer_fired(),
                Timer::TxCoalescing => self.tx_coal_timer_fired(),
                Timer::RxArrival => self.rx_arrival_due(target),
            }
        }
        self.now = target;
    }

    fn timers(&self) -> Vec<(Instant, Timer)> {
        [
            self.phy.deadline().map(|d| (d, Timer::PhySettle)),
            self.bufmgr_start.map(|d| (d, Timer::BufferManager)),
            self.rx_coal_timer.map(|d| (d, Timer::RxCoalescing)),
            self.tx_coal_timer.map(|d| (d, Timer::TxCoalescing)),
            self.rx_latency.front().map(|held| (held.due, Timer::RxArrival)),
        ]
        .into_iter()
        .flatten()
        .collect()
    }

    /// The earliest moment something here needs attention.
    pub fn deadline(&self) -> Option<Instant> {
        self.timers().into_iter().map(|(d, _)| d).min()
    }

    pub fn refused(&self) -> Vec<(Refused, u64)> {
        self.refused.iter().map(|(what, count)| (*what, *count)).collect()
    }

    pub fn eeprom_bytes(&self) -> Option<Vec<u8>> {
        self.eeprom.as_ref().map(|e| e.bytes().to_vec())
    }

    pub fn refuse(&mut self, what: Refused) {
        let count = self.refused.entry(what).or_insert(0);
        *count += 1;
        if *count == 1 {
            crate::dlog!(crate::devlog::LogModule::Net, "bcm5701: unsupported or unexpected: {what:?}");
        }
    }

    fn drain_phy_refusals(&mut self) {
        for what in std::mem::take(&mut self.phy.refused) {
            self.refuse(Refused::Phy(what));
        }
    }

    // ---- storage ----

    pub(super) fn rf(&self, off: u32) -> u32 {
        self.regfile[((off & (REG_FILE_SIZE - 1)) >> 2) as usize]
    }

    pub(super) fn set_rf(&mut self, off: u32, val: u32) {
        self.regfile[((off & (REG_FILE_SIZE - 1)) >> 2) as usize] = val;
    }

    pub fn sram_read(&self, off: u32) -> u32 {
        self.sram[((off & (SRAM_SIZE - 1)) >> 2) as usize]
    }

    pub fn sram_write(&mut self, off: u32, val: u32) {
        let off = off & (SRAM_SIZE - 1);
        let val = if off == SRAM_FW_MBOX { !val } else { val };
        self.sram[(off >> 2) as usize] = val;
    }

    // ---- resets ----

    /// The station address, as the on-chip bootcode owns it: every reset
    /// puts it back.
    fn load_station_address(&mut self) {
        let m = self.mac;
        let hi = u32::from(m[0]) << 8 | u32::from(m[1]);
        let lo = u32::from_be_bytes([m[2], m[3], m[4], m[5]]);
        self.sram_write(SRAM_MAC_HI, MAC_TAG << 16 | hi);
        self.sram_write(SRAM_MAC_LO, lo);
        self.set_rf(reg::MAC_ADDR_0_HIGH, hi);
        self.set_rf(reg::MAC_ADDR_0_LOW, lo);
    }

    /// The driver's write of MISC_CFG.RESET_CORE_CLOCKS, by either access
    /// path: the register file and SRAM return to their reset values, the
    /// handshake mailbox re-arms to the firmware-ready complement and the
    /// bootcode's station address comes back. PCI configuration, the chip id
    /// in MISC_CTL among it, and the indirect-window bases are untouched.
    fn global_reset(&mut self) {
        self.bufmgr_start = None;
        self.regfile.fill(0);
        self.sram.fill(0);
        self.reset_engines();
        self.irq_masked = true;
        self.irq_pending = false;
        self.sram_write(SRAM_FW_MBOX, FW_MAGIC);
        self.load_station_address();
        self.update_irq(false);
    }

    /// A cold reset: as a global reset, plus the window bases, the PHY, and
    /// no firmware-ready complement (only the driver-triggered reset arranges
    /// that, because it is the one the driver writes the magic for).
    pub fn reset(&mut self) {
        self.bufmgr_start = None;
        self.regfile.fill(0);
        self.sram.fill(0);
        self.reg_win_base = 0;
        self.mem_win_base = 0;
        self.pcistate = Self::pcistate_at_reset(self.req64);
        self.reset_engines();
        self.irq_masked = true;
        self.irq_pending = false;
        self.phy.reset(self.now);
        self.load_station_address();
        self.update_irq(true);
    }

    // ---- interrupt ----

    pub(super) fn misc_ctl(&self) -> u32 {
        self.cfg.misc_ctl()
    }

    /// PCISTATE after reset: conventional PCI, the interrupt line down, and
    /// a 32-bit bus unless the slot drove REQ64#.
    pub fn pcistate_at_reset(req64: bool) -> u32 {
        pcistate::BUSMODE_PCI | pcistate::INTR_NOT_ACTIVE | if req64 { 0 } else { pcistate::BUS_32BIT }
    }

    /// The interrupt is a level: INTA asserts while a status-block update
    /// awaits the driver and neither the mailbox nor MISC_CTL.MASK_PCI_INTR
    /// masks it. PCISTATE.INTR_NOT_ACTIVE mirrors the inverse, so the
    /// driver's shared-interrupt check reads true exactly when the line is
    /// down. `force` delivers the level even if unchanged.
    pub fn update_irq(&mut self, force: bool) {
        let active = self.irq_pending && !self.irq_masked && self.misc_ctl() & misc_ctl::MASK_PCI_INTR == 0;
        if active {
            self.pcistate &= !pcistate::INTR_NOT_ACTIVE;
        } else {
            self.pcistate |= pcistate::INTR_NOT_ACTIVE;
        }
        if force || active != self.irq_level {
            self.irq_level = active;
            self.irq_events.push(active);
        }
    }

    pub fn irq_level(&self) -> bool {
        self.irq_level
    }

    /// A write to MISC_CTL acknowledges if CLEAR_INTA was set, then re-derives
    /// the line. This is how the driver's "set CLEAR_INTA, clear
    /// MASK_PCI_INTR" enables the interrupt.
    pub fn misc_ctl_written(&mut self) {
        if self.misc_ctl() & misc_ctl::CLEAR_INTA != 0 {
            self.irq_pending = false;
            self.cfg.clear_misc_ctl(misc_ctl::CLEAR_INTA);
        }
        self.update_irq(false);
    }

    // ---- EEPROM ----

    fn eeprom_access(&mut self, command: u32) -> bool {
        let device = ((command & eeprom_cmd::DEVICE_ID) >> 26) as u8;
        let word = (command & eeprom_cmd::WORD_ADDR) >> 2;
        let result = match self.eeprom.as_mut() {
            None => {
                self.refuse(Refused::EepromFault(eeprom::Fault::NoBacking));
                return false;
            }
            Some(e) if command & eeprom_cmd::READ != 0 => e.read_word(device, word).map(Some),
            Some(e) => {
                let value = self.regfile[(reg::EEPROM_DATA >> 2) as usize];
                e.write_word(device, word, value).map(|()| None)
            }
        };
        match result {
            Ok(Some(value)) => {
                self.set_rf(reg::EEPROM_DATA, value);
                true
            }
            Ok(None) => true,
            Err(fault) => {
                self.refuse(Refused::EepromFault(fault));
                false
            }
        }
    }

    fn eeprom_command_write(&mut self, val: u32) {
        let old = self.rf(reg::EEPROM_ADDR);
        let mut command = val & !(eeprom_cmd::START | eeprom_cmd::COMPLETE | eeprom_cmd::RESET | 3);
        let acknowledge = val & eeprom_cmd::COMPLETE != 0;
        if val & eeprom_cmd::RESET != 0 {
            self.set_rf(reg::EEPROM_ADDR, eeprom_cmd::RESET);
            self.set_rf(reg::EEPROM_DATA, 0);
            return;
        }
        if !acknowledge {
            command |= old & eeprom_cmd::COMPLETE;
        }
        self.set_rf(reg::EEPROM_ADDR, command);
        // A pending Complete blocks the next transfer until it is
        // acknowledged; acknowledging and starting in one write is allowed.
        if val & eeprom_cmd::START == 0 || (old & eeprom_cmd::COMPLETE != 0 && !acknowledge) {
            return;
        }
        if self.eeprom_access(command) {
            self.set_rf(reg::EEPROM_ADDR, command | eeprom_cmd::COMPLETE);
        }
    }

    // ---- register file ----

    /// Register-file writes, reached identically from the BAR and from the
    /// indirect register window, so a side effect fires whichever path the
    /// driver uses. The 5700/5701 issue the global reset through the window.
    pub fn reg_write(&mut self, off: u32, val: u32) {
        let off = off & (REG_FILE_SIZE - 1);
        let mut val = val;
        match off {
            reg::MISC_CFG => {
                if val & MISCCFG_RESET_CORE_CLOCKS != 0 {
                    self.global_reset();
                    return;
                }
            }
            reg::EEPROM_ADDR => {
                self.eeprom_command_write(val);
                return;
            }
            reg::EEPROM_DATA | reg::EEPROM_DELAY => {}
            reg::EEPROM_PIN => {
                self.refuse(Refused::EepromPins);
                self.set_rf(off, val);
                return;
            }
            // The reset bit self-clears; the rest of the write persists.
            reg::MAC_MODE => val &= !MODE_RESET,
            reg::RX_MODE => {
                // A halt takes effect at once. The chip finishes the frame in
                // flight first and reads back ENABLE until it has, which a
                // model that takes one frame at a time never needs.
                val &= !MODE_RESET;
                self.set_rf(off, val);
                if val & RXMODE_ENABLE != 0 {
                    self.retry_rx_flush();
                }
                return;
            }
            // Write one to clear: the bits written are the ones acknowledged.
            reg::MAC_STATUS => {
                let old = self.rf(off);
                self.set_rf(off, old & !val);
                return;
            }
            reg::HCC_MODE => {
                val &= !MODE_RESET;
                self.set_rf(off, val);
                self.hcc_mode_written();
                return;
            }
            reg::HCC_RX_COAL_TICKS | reg::HCC_TX_COAL_TICKS => {
                self.set_rf(off, val);
                self.coal_ticks_written(off == reg::HCC_RX_COAL_TICKS);
                return;
            }
            reg::MBX_TX_PROD_LO => {
                // Store the producer for read-back, then send up to it.
                self.set_rf(off, val);
                self.tx_kick(val);
                return;
            }
            reg::RX_JUMBO_RCB_MAXLEN_FLAGS => {
                // Max_Len is unused for the jumbo ring; only Disable Ring and
                // Extended RX Enable do anything, and both are read from the
                // stored word. Flag the unmodelled extended format once, here,
                // rather than on every frame that finds the ring unavailable.
                if val & RCB_FLAG_EXT_RX_ENABLE != 0 {
                    self.refuse(Refused::JumboExtendedDescriptors);
                }
                self.set_rf(off, val);
                self.rx_ring_setup_written();
                return;
            }
            // What the descriptor fetch depends on: a ring that becomes
            // usable, or a lower threshold, can start one.
            reg::RX_STD_RCB_HADDR_HI
            | reg::RX_STD_RCB_HADDR_LO
            | reg::RX_STD_RCB_MAXLEN_FLAGS
            | reg::RX_JUMBO_RCB_HADDR_HI
            | reg::RX_JUMBO_RCB_HADDR_LO
            | reg::RX_STD_REPLENISH_THRESH
            | reg::RX_JUMBO_REPLENISH_THRESH => {
                self.set_rf(off, val);
                self.rx_ring_setup_written();
                return;
            }
            reg::MBX_RX_JUMBO_PROD_LO | reg::MBX_RX_STD_PROD_LO | reg::MBX_RX_CONS0_LO => {
                self.rx_mailbox_written(off, val);
                return;
            }
            reg::BUFMGR_MODE => {
                if val & MODE_RESET != 0 {
                    self.bufmgr_start = None;
                    self.set_rf(off, 0);
                }
                val &= !MODE_RESET;
                if val & BUFMGR_MODE_ENABLE == 0 {
                    self.bufmgr_start = None;
                } else if self.rf(off) & BUFMGR_MODE_ENABLE == 0 {
                    if self.bufmgr_start.is_none() {
                        self.bufmgr_start = Some(self.now + BUFMGR_START);
                    }
                    val &= !BUFMGR_MODE_ENABLE;
                }
            }
            reg::BUFMGR_STATUS => {
                self.refuse(Refused::BufmgrStatusWrite);
                return;
            }
            reg::MBX_IRQ0_LO => {
                // Every write clears the pending update; a nonzero value
                // masks the line and zero re-enables it.
                self.irq_pending = false;
                self.irq_masked = val != 0;
                self.set_rf(off, val);
                if val == 0 {
                    self.retry_rx_flush();
                }
                self.update_irq(false);
                return;
            }
            reg::MI_COMM => {
                self.mi_comm_write(val);
                return;
            }
            _ => {
                self.refuse(Refused::UnmodelledWrite(off));
                self.set_rf(off, val);
                return;
            }
        }
        self.set_rf(off, val);
    }

    fn mi_comm_write(&mut self, command: u32) {
        let register = ((command & mi::REG) >> mi::REG_SHIFT) as u8;
        if command & mi::BUSY == 0 {
            self.set_rf(reg::MI_COMM, command);
            return;
        }
        let now = self.now;
        let opcode = command & (mi::READ | mi::WRITE);
        let val = if opcode == mi::READ {
            (command & mi::READ_FIELDS) | u32::from(self.phy.read(register, now))
        } else if opcode == mi::WRITE {
            self.phy.write(register, (command & mi::DATA) as u16, now);
            command & !mi::BUSY
        } else {
            self.refuse(Refused::MiOpcode(opcode >> 26));
            command & !mi::BUSY
        };
        self.set_rf(reg::MI_COMM, val);
        self.drain_phy_refusals();
    }

    /// Register-file reads, reached identically from the BAR and the window.
    /// Most read back their stored value; MI_STS is the PHY's live link so
    /// the driver's auto-poll path sees the carrier.
    pub fn reg_read(&mut self, off: u32) -> u32 {
        match off & (REG_FILE_SIZE - 1) {
            reg::MI_STS => {
                self.phy.sync(self.now);
                if self.phy.link_up() {
                    mi::LINK
                } else {
                    0
                }
            }
            reg::FW_PROGRESS => FW_PROGRESS_INIT,
            _ => self.rf(off),
        }
    }
}
