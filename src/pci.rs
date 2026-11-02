//! PCI: the configuration header every function has, and the trait a
//! function implements to sit behind a host bridge.
//!
//! The bridge (the root complex) owns everything machine-specific: how
//! configuration cycles are generated, where the CPU sees PCI memory, byte
//! lanes, and which interrupt input each slot's pins are wired to. A
//! function only sees configuration registers, accesses to its BARs, and an
//! [`IrqLine`] for its pin.

use std::sync::{Arc, Mutex};

use crate::irq::IrqLine;
use crate::snapshot::{get_field, load_u32_slice, u32_slice_to_toml};
use crate::traits::{BusDevice, BusRead16, BusRead32, BusRead64, BusRead8, BUS_OK};

/// Command register bits.
pub const PCI_COMMAND_IO: u32 = 0x1;
pub const PCI_COMMAND_MEMORY: u32 = 0x2;

/// Interrupt pins, as register 0x3d holds them.
pub const PCI_INTA: u8 = 1;

/// What one base address register decodes, and how much.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Bar {
    Unused,
    /// I/O space, this many bytes (a power of two).
    Io(u32),
    /// 32-bit memory space, this many bytes (a power of two).
    Mem(u32),
}

impl Bar {
    fn size(self) -> u32 {
        match self {
            Bar::Unused => 0,
            Bar::Io(n) | Bar::Mem(n) => n,
        }
    }
}

/// Which address space an access is in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Space {
    Io,
    Mem,
}

/// Initial value and writable bits for a function-specific configuration dword.
#[derive(Clone, Copy, Debug)]
pub struct PciRegisterPolicy {
    /// Dword index. Header, BAR, subsystem identity, and interrupt dwords are managed separately.
    pub reg_dword: usize,
    /// Value restored by [`PciConfig::reset`].
    pub initial: u32,
    /// Bits that configuration writes may change.
    pub writable_mask: u32,
}

/// Function-specific write behavior for a type 0 PCI configuration header.
///
/// Register policies are copied into the header during construction; the
/// borrowed slice is not retained.
#[derive(Clone, Copy, Debug)]
pub struct PciConfigPolicy<'a> {
    /// Fixed Status bits that read as one.
    pub status: u16,
    /// Device-latched Status bits that writes acknowledge with one.
    pub status_w1c_mask: u16,
    /// Writable Command bits.
    pub command_mask: u16,
    /// Writable bits in the latency timer byte.
    pub latency_mask: u8,
    /// Writable bits in the cache-line-size byte.
    pub cache_line_mask: u8,
    /// Initial values and write masks for function-specific dwords.
    pub registers: &'a [PciRegisterPolicy],
}

impl<'a> Default for PciConfigPolicy<'a> {
    fn default() -> Self {
        Self {
            status: 0x0200,
            status_w1c_mask: 0,
            command_mask: 0xffff,
            latency_mask: 0xff,
            cache_line_mask: 0xff,
            registers: &[],
        }
    }
}

/// One function's type 0 configuration header.
///
/// Enough of it to be discovered and configured: identity, class, command
/// and status, the interrupt pin, and base address registers that answer a
/// size probe. Writing all ones to a BAR and reading it back is how firmware
/// learns how much space a device wants, so a BAR that stores whatever it is
/// given reports a size of four gigabytes and the bus layout falls apart.
pub struct PciConfig {
    regs: Mutex<[u32; 64]>,
    bars: [Bar; 6],
    /// The registers as the function powers up, for [`PciConfig::reset`].
    power_on: [u32; 64],
    status: u16,
    status_w1c_mask: u16,
    command_mask: u16,
    latency_mask: u8,
    cache_line_mask: u8,
    writable_masks: [u32; 64],
}

impl PciConfig {
    /// `class` is the 24-bit class/subclass/interface, as it sits in the top
    /// three bytes of register 0x08. `pin` is 0 for none, or [`PCI_INTA`]
    /// and on; the line register is left for firmware to assign.
    pub fn new(vendor: u16, device: u16, class: u32, revision: u8, bars: [Bar; 6], pin: u8) -> Self {
        Self::new_with_policy(
            vendor,
            device,
            class,
            revision,
            bars,
            pin,
            PciConfigPolicy::default(),
        )
    }

    /// Construct a type 0 header with function-specific register behavior.
    pub fn new_with_policy(
        vendor: u16,
        device: u16,
        class: u32,
        revision: u8,
        bars: [Bar; 6],
        pin: u8,
        policy: PciConfigPolicy<'_>,
    ) -> Self {
        let mut r = [0u32; 64];
        r[0] = ((device as u32) << 16) | vendor as u32;
        r[2] = (class << 8) | revision as u32;
        r[15] = (pin as u32) << 8;
        for (i, b) in bars.iter().enumerate() {
            if let Bar::Io(_) = b {
                r[4 + i] = 1;
            }
        }
        let mut writable_masks = [u32::MAX; 64];
        for register in policy.registers {
            assert!(
                matches!(register.reg_dword, 10 | 12..=14 | 16..=63),
                "pci: register policy requires a function-specific dword in 0..64");
            r[register.reg_dword] = register.initial;
            writable_masks[register.reg_dword] = register.writable_mask;
        }
        Self {
            regs: Mutex::new(r),
            bars,
            power_on: r,
            status: policy.status,
            status_w1c_mask: policy.status_w1c_mask,
            command_mask: policy.command_mask,
            latency_mask: policy.latency_mask,
            cache_line_mask: policy.cache_line_mask,
            writable_masks,
        }
    }

    /// RST#: back to the power-up header. The command register clears, so
    /// the function stops decoding, and the BARs and line forget what
    /// firmware assigned.
    pub fn reset(&self) {
        let mut r = self.regs.lock().unwrap();
        // The subsystem identity is the board's, set once after
        // construction (IOC3's selects what a guest instantiates).
        let subsystem = r[11];
        *r = self.power_on;
        r[11] = subsystem;
    }

    /// Register `reg` (a dword index, 0..64).
    pub fn read(&self, reg: usize) -> u32 {
        let r = self.regs.lock().unwrap();
        match reg {
            1 => (r[1] & (0xffff | ((self.status_w1c_mask as u32) << 16)))
                | ((self.status as u32) << 16),
            _ => r[reg],
        }
    }

    pub fn write(&self, reg: usize, val: u32) {
        self.write_masked(reg, val, u32::MAX);
    }

    /// Write selected bits of dword `reg`. Both `val` and `mask` use the
    /// register's bit positions. Unselected Status events are not acknowledged.
    pub fn write_masked(&self, reg: usize, val: u32, mask: u32) {
        if mask == 0 {
            return;
        }
        let mut r = self.regs.lock().unwrap();
        match reg {
            // Identity, class and header type are read-only.
            0 | 2 => {}
            3 => {
                let mask = mask & (((self.latency_mask as u32) << 8) | self.cache_line_mask as u32);
                r[3] = (r[3] & !mask) | (val & mask);
            }
            // Command writes replace selected bits; Status events clear on ones.
            1 => {
                let clear = val & mask & ((self.status_w1c_mask as u32) << 16);
                r[1] = (r[1] & !(mask & 0xffff) & !clear)
                    | (val & mask & self.command_mask as u32);
            }
            4..=9 => {
                let val = (r[reg] & !mask) | (val & mask);
                let bar = self.bars[reg - 4];
                r[reg] = match bar {
                    Bar::Unused => 0,
                    // Address bits above the size take the write; those
                    // below read back as zero, which tells firmware the
                    // size. The type bits stay what the BAR is.
                    Bar::Io(n) => (val & !(n - 1)) | 1,
                    Bar::Mem(n) => val & !(n - 1),
                };
            }
            // Interrupt pin is read-only; the line is firmware's scratch.
            15 => {
                let mask = mask & !0xff00;
                r[15] = (r[15] & !mask) | (val & mask);
            }
            _ => {
                let mask = mask & self.writable_masks[reg];
                r[reg] = (r[reg] & !mask) | (val & mask);
            }
        }
    }

    /// Latch selected device Status events without replacing earlier events.
    pub fn set_status_bits(&self, bits: u16) {
        self.regs.lock().unwrap()[1] |= ((bits & self.status_w1c_mask) as u32) << 16;
    }

    /// Set the board-assigned Subsystem ID, which survives function reset.
    pub fn set_subsystem_identity(&self, value: u32) {
        self.regs.lock().unwrap()[11] = value;
    }

    /// The base address firmware programmed into BAR `i`, type bits masked.
    pub fn bar_base(&self, i: usize) -> u32 {
        let v = self.regs.lock().unwrap()[4 + i];
        match self.bars[i] {
            Bar::Io(_) => v & !3,
            _ => v & !15,
        }
    }

    pub fn command(&self) -> u32 {
        self.regs.lock().unwrap()[1] & 0xffff
    }

    pub fn interrupt_pin(&self) -> u8 {
        (self.regs.lock().unwrap()[15] >> 8) as u8
    }

    /// The windows this function decodes: (space, BAR, base, size) for
    /// each BAR that has been assigned an address in a space the command
    /// register enables, in BAR order. A bridge caches these and rebuilds
    /// them after a configuration write.
    pub fn windows(&self) -> Vec<(Space, usize, u32, u32)> {
        let cmd = self.command();
        (0..6)
            .filter_map(|i| {
                let (space, size) = match self.bars[i] {
                    Bar::Io(n) if cmd & PCI_COMMAND_IO != 0 => (Space::Io, n),
                    Bar::Mem(n) if cmd & PCI_COMMAND_MEMORY != 0 => (Space::Mem, n),
                    _ => return None,
                };
                let base = self.bar_base(i);
                (base != 0).then_some((space, i, base, size))
            })
            .collect()
    }

    /// The BAR decoding `addr` in `space`, and the offset within it: only
    /// once the command register enables that space, and only for a BAR
    /// that has been assigned an address.
    pub fn decode(&self, space: Space, addr: u32) -> Option<(usize, u32)> {
        self.decode_with(space, addr, false)
    }

    /// [`Self::decode`], but a BAR left at zero decodes at zero: what a
    /// function does on a bus whose firmware enables it without moving it
    /// (the Octane's PROM reaches the RAD that way).
    pub fn decode_unmoved(&self, space: Space, addr: u32) -> Option<(usize, u32)> {
        self.decode_with(space, addr, true)
    }

    fn decode_with(&self, space: Space, addr: u32, zero_ok: bool) -> Option<(usize, u32)> {
        let enable = match space {
            Space::Io => PCI_COMMAND_IO,
            Space::Mem => PCI_COMMAND_MEMORY,
        };
        if self.command() & enable == 0 {
            return None;
        }
        (0..6).find_map(|i| {
            let bar = self.bars[i];
            let in_space = matches!((bar, space), (Bar::Io(_), Space::Io) | (Bar::Mem(_), Space::Mem));
            let base = self.bar_base(i);
            let size = bar.size();
            (in_space && (base != 0 || zero_ok) && addr.wrapping_sub(base) < size).then(|| (i, addr - base))
        })
    }
}

/// The header's registers. Identity, class and the BARs' sizes come from
/// the function, so a header only loads into the same kind of function.
impl crate::traits::Saveable for PciConfig {
    fn save_state(&self) -> toml::Value {
        let mut t = toml::map::Map::new();
        t.insert("regs".into(), u32_slice_to_toml(&*self.regs.lock().unwrap()));
        toml::Value::Table(t)
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        let Some(r) = get_field(v, "regs") else { return Ok(()) };
        let mut saved = [0u32; 64];
        load_u32_slice(r, &mut saved);
        let mut regs = self.regs.lock().unwrap();
        if saved[0] != regs[0] {
            return Err(format!("pci: saved header is for {:#010x}, this function is {:#010x}", saved[0], regs[0]));
        }
        *regs = saved;
        Ok(())
    }
}

/// A PCI function, as a host bridge sees it.
///
/// BAR accesses arrive with the offset and value the device sees: whatever
/// the bridge does to byte lanes it has already done.
pub trait PciBusDevice: Send + Sync {
    /// The function's configuration header.
    fn config(&self) -> &PciConfig;

    /// What a configuration read of dword `index` returns to the guest. The
    /// header as stored, unless the chip decodes fewer address lines than
    /// the bus gives it (IOC3 repeats its first 32 bytes).
    fn config_dword(&self, index: usize) -> u32 {
        self.config().read(index)
    }

    /// A read of `width` bytes (1, 2 or 4) at `off` within BAR `bar`.
    fn bar_read(&self, bar: usize, off: u32, width: usize) -> u32;

    fn bar_write(&self, bar: usize, off: u32, val: u32, width: usize);

    /// The bridge wires the function's interrupt pin to `line`.
    fn connect_interrupt(&self, _line: IrqLine) {}

    /// The bridge gives a function that masters the bus its way to memory.
    fn connect_dma(&self, _dma: std::sync::Arc<dyn PciDma>) {}
}

/// How a bus master's access ended.
#[must_use]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PciDmaStatus {
    Ok,
    /// Some of the bytes had no target: reads return zero for them and
    /// writes drop them. A bridge may also log it and raise its error
    /// interrupt.
    MasterAbort,
}

impl PciDmaStatus {
    pub fn is_ok(self) -> bool {
        self == PciDmaStatus::Ok
    }
}

/// Memory as a bus master reaches it. The bridge decides what a PCI address
/// is and in which order its bytes arrive; the function sees bytes at bus
/// addresses, as it would on the bus.
///
/// A function calls only the object `connect_dma` gave it and never learns
/// whether it is shared. The O2's MACE hands every function one object (its
/// translation is fixed); a bridge whose translation depends on the
/// mastering device, as the Octane's does, hands each its own.
pub trait PciDma: Send + Sync {
    /// `buf.len()` bytes from bus address `addr`, 64 bits wide for masters
    /// that use dual address cycles.
    fn dma_read(&self, addr: u64, buf: &mut [u8]) -> PciDmaStatus;

    /// `data` to bus address `addr`.
    fn dma_write(&self, addr: u64, data: &[u8]) -> PciDmaStatus;
}

// ---- BARs on the physical bus ----

/// What a host bridge does to a CPU access on its way to a function: which
/// byte lanes carry it, and how an access wider than the bus is split.
/// The bridge implements it once per way it is configured to move bytes.
pub trait PciLanes: Send + Sync {
    /// A CPU read of `size` bytes (1, 2, 4 or 8) whose first byte is at
    /// offset `off` of BAR `bar` of `f`, before the bridge moves lanes.
    /// The value is in CPU order, most significant byte first in memory.
    fn read(&self, f: &dyn PciBusDevice, bar: usize, off: u32, size: usize) -> u64;

    fn write(&self, f: &dyn PciBusDevice, bar: usize, off: u32, size: usize, val: u64);
}

/// A run of a host bridge's CPU window that one BAR decodes whole, as a
/// decode-table entry: the access goes straight to the function through the
/// bridge's lanes, with no search of the bus and no bridge lock. The bridge
/// builds these with [`bar_runs`] whenever what its windows reach changes,
/// and the machine maps them on its physical bus over the bridge's own
/// window; every other page of the window stays with the bridge.
pub struct PciBar {
    lanes: Arc<dyn PciLanes>,
    dev: Arc<dyn PciBusDevice>,
    bar: usize,
    /// The CPU address bits the bridge decodes, and where in them the BAR's
    /// byte 0 sits (it may lie below the window: a BAR larger than it).
    mask: u64,
    base: u64,
}

impl PciBar {
    pub fn new(lanes: Arc<dyn PciLanes>, dev: Arc<dyn PciBusDevice>, bar: usize, mask: u64, base: u64) -> Self {
        Self { lanes, dev, bar, mask, base }
    }

    #[inline]
    fn off(&self, addr: u64) -> u32 {
        (addr & self.mask).wrapping_sub(self.base) as u32
    }

    fn read(&self, addr: u64, size: usize) -> u64 {
        self.lanes.read(&*self.dev, self.bar, self.off(addr), size)
    }

    fn write(&self, addr: u64, size: usize, val: u64) -> u32 {
        self.lanes.write(&*self.dev, self.bar, self.off(addr), size, val);
        BUS_OK
    }
}

impl BusDevice for PciBar {
    fn read8(&self, a: u64) -> BusRead8 { BusRead8::ok(self.read(a, 1) as u8) }
    fn read16(&self, a: u64) -> BusRead16 { BusRead16::ok(self.read(a, 2) as u16) }
    fn read32(&self, a: u64) -> BusRead32 { BusRead32::ok(self.read(a, 4) as u32) }
    fn read64(&self, a: u64) -> BusRead64 { BusRead64::ok(self.read(a, 8)) }
    fn write8(&self, a: u64, v: u8) -> u32 { self.write(a, 1, v as u64) }
    fn write16(&self, a: u64, v: u16) -> u32 { self.write(a, 2, v as u64) }
    fn write32(&self, a: u64, v: u32) -> u32 { self.write(a, 4, v as u64) }
    fn write64(&self, a: u64, v: u64) -> u32 { self.write(a, 8, v) }
}

/// A run of a bridge's CPU window, `start..end`, and what answers it.
#[derive(Clone)]
pub struct BarRun {
    pub start: u64,
    pub end: u64,
    pub dev: Arc<dyn BusDevice>,
}

/// One BAR some function decodes, as [`bar_runs`] reports it to `entry`:
/// the function's slot, the function, the BAR, and the CPU address (in the
/// window's terms) of its byte 0.
pub struct BarAt<'a> {
    pub slot: usize,
    pub dev: &'a Arc<dyn PciBusDevice>,
    pub bar: usize,
    pub cpu_base: u64,
}

/// The runs of a CPU window `cpu..cpu + len` opening onto PCI `space` at
/// PCI address `pci` that one BAR decodes whole, in whole decode-table
/// pages. `entry` makes (or finds) the table entry for each BAR. Pages a
/// BAR covers only in part, and pages two BARs claim (a misconfiguration
/// the bridge resolves its own way), are left out: the bridge decodes those
/// itself.
pub fn bar_runs(
    funcs: &[(usize, Arc<dyn PciBusDevice>)],
    space: Space,
    cpu: u64,
    pci: u32,
    len: u64,
    mut entry: impl FnMut(BarAt) -> Arc<dyn BusDevice>,
) -> Vec<BarRun> {
    const PAGE: u64 = 1 << crate::ppmem::devmap::SLOT_SHIFT;
    let (wlo, whi) = (pci as u64, pci as u64 + len);
    let mut found: Vec<(u64, u64, usize, usize, u64)> = Vec::new();
    for (k, (_, f)) in funcs.iter().enumerate() {
        for (sp, bar, base, size) in f.config().windows() {
            if sp != space {
                continue;
            }
            let (lo, hi) = ((base as u64).max(wlo), (base as u64 + size as u64).min(whi));
            let (lo, hi) = (lo.next_multiple_of(PAGE), hi & !(PAGE - 1));
            if lo < hi {
                found.push((lo, hi, k, bar, base as u64));
            }
        }
    }
    let overlaps = |i: usize| {
        let (lo, hi, ..) = found[i];
        found.iter().enumerate().any(|(j, &(l, h, ..))| j != i && l < hi && lo < h)
    };
    (0..found.len())
        .filter(|&i| !overlaps(i))
        .map(|i| {
            let (lo, hi, k, bar, base) = found[i];
            let (slot, dev) = (&funcs[k].0, &funcs[k].1);
            let at = |p: u64| cpu.wrapping_add(p).wrapping_sub(wlo);
            BarRun { start: at(lo), end: at(hi), dev: entry(BarAt { slot: *slot, dev, bar, cpu_base: at(base) }) }
        })
        .collect()
}

/// Where a bridge's runs go: the machine's bus maps each over the bridge's
/// window and gives the pages the last call named back to the bridge.
pub type BarRunSink = Box<dyn Fn(Vec<BarRun>) + Send + Sync>;

#[cfg(test)]
mod tests {
    use super::*;

    fn scsi_like() -> PciConfig {
        PciConfig::new(0x9004, 0x8078, 0x0100_00, 1,
            [Bar::Io(0x100), Bar::Mem(0x1000), Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused], PCI_INTA)
    }

    fn scsi_like_with_policy(policy: PciConfigPolicy<'_>) -> PciConfig {
        PciConfig::new_with_policy(
            0x9004, 0x8078, 0x0100_00, 1,
            [Bar::Io(0x100), Bar::Mem(0x1000), Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused], PCI_INTA,
            policy,
        )
    }

    /// Firmware learns how much space a device wants by writing all ones to
    /// a BAR and reading back which address bits stuck.
    #[test]
    fn save_load_round_trip() {
        use crate::traits::Saveable;
        let src = scsi_like();
        src.write(1, PCI_COMMAND_MEMORY | PCI_COMMAND_IO);
        src.write(4, 0x0000_1000);
        src.write(5, 0x8000_1000);
        src.write(15, 0x0000_0108);
        let v1 = src.save_state();
        let dst = scsi_like();
        dst.load_state(&v1).expect("load_state");
        assert_eq!(v1, dst.save_state(), "PciConfig save_state mismatch after load_state round-trip");
        assert_eq!(dst.bar_base(1), 0x8000_1000);
        let other = PciConfig::new(0x1234, 0x5678, 0, 0, [Bar::Unused; 6], 0);
        assert!(other.load_state(&v1).is_err(), "a header loads only into the same kind of function");
    }

    #[test]
    fn a_bar_reports_its_size_to_a_probe() {
        let c = scsi_like();
        assert_eq!(c.read(0), 0x8078_9004, "device and vendor id");
        c.write(4, 0xffff_ffff);
        assert_eq!(c.read(4), 0xffff_ff01, "256 bytes of I/O space");
        c.write(5, 0xffff_ffff);
        assert_eq!(c.read(5), 0xffff_f000, "4 KiB of memory space");
        // An unimplemented BAR reads back as zero however hard it is probed.
        c.write(6, 0xffff_ffff);
        assert_eq!(c.read(6), 0);
        c.write(5, 0x8000_1000);
        assert_eq!(c.bar_base(1), 0x8000_1000, "and then holds the address assigned to it");
    }

    /// A function decodes a space only once it has been told it may, and
    /// only within the BAR it was assigned.
    #[test]
    fn a_function_claims_only_its_own_window_and_only_when_enabled() {
        let c = scsi_like();
        c.write(5, 0x8000_1000);
        assert_eq!(c.decode(Space::Mem, 0x8000_1000), None, "memory decoding is off until enabled");
        c.write(1, PCI_COMMAND_MEMORY);
        assert_eq!(c.decode(Space::Mem, 0x8000_1084), Some((1, 0x84)));
        assert_eq!(c.decode(Space::Mem, 0x8000_2000), None, "past the end of the BAR");
        assert_eq!(c.decode(Space::Io, 0x8000_1084), None, "and not in I/O space");
    }

    #[test]
    fn reset_returns_the_header_to_power_up() {
        let c = scsi_like();
        let fresh: Vec<u32> = (0..64).map(|i| c.read(i)).collect();
        c.write(1, PCI_COMMAND_MEMORY | PCI_COMMAND_IO);
        c.write(4, 0x0000_1000);
        c.write(5, 0x8000_1000);
        c.write(15, 0x0000_0009);
        c.reset();
        assert_eq!((0..64).map(|i| c.read(i)).collect::<Vec<_>>(), fresh);
        assert_eq!(c.decode(Space::Mem, 0x8000_1000), None);
        assert_eq!(c.interrupt_pin(), PCI_INTA, "the pin is wiring, not state");
    }

    /// A subsystem identity set after construction, as IOC3 sets its own,
    /// is the board's and survives a reset.
    #[test]
    fn reset_keeps_the_subsystem_identity() {
        let c = scsi_like();
        c.set_subsystem_identity(0xc304_10a9);
        c.write(4, 0x0000_1000);
        c.reset();
        assert_eq!(c.read(11), 0xc304_10a9);
        assert_eq!(c.read(4), 1);
    }

    /// Firmware programs the latency timer and cache line size, and reads
    /// them back; the header type byte above them stays what it is.
    #[test]
    fn the_latency_timer_and_cache_line_size_are_stored() {
        let c = scsi_like();
        c.write(3, 0x00ff_2810);
        assert_eq!(c.read(3), 0x0000_2810);
        c.reset();
        assert_eq!(c.read(3), 0);
    }

    #[test]
    fn the_interrupt_pin_is_read_only() {
        let c = scsi_like();
        c.write(15, 0x0000_0009);
        assert_eq!(c.read(15) & 0xffff, 0x0109, "line 9 assigned, pin A kept");
        c.write(15, 0);
        assert_eq!(c.interrupt_pin(), PCI_INTA);
    }

    #[test]
    fn header_policies_mask_writes_and_restore_reset_values() {
        for (status, command, latency, cache) in [(0x0200, 0xffff, 0, 0), (0x02b0, 0x0156, 0xf0, 0x0f)] {
            let c = scsi_like_with_policy(PciConfigPolicy {
                status,
                status_w1c_mask: 0,
                command_mask: command,
                latency_mask: latency,
                cache_line_mask: cache,
                registers: &[PciRegisterPolicy {
                    reg_dword: 18,
                    initial: 0xaabb_0020,
                    writable_mask: 0xff00,
                }],
            });
            c.write(1, u32::MAX);
            c.write(3, u32::MAX);
            c.write(18, u32::MAX);
            assert_eq!(c.read(1), (u32::from(status) << 16) | u32::from(command));
            assert_eq!(c.read(3), (u32::from(latency) << 8) | u32::from(cache));
            assert_eq!(c.read(18), 0xaabb_ff20);
            c.reset();
            assert_eq!(c.read(1), u32::from(status) << 16);
            assert_eq!(c.read(3), 0);
            assert_eq!(c.read(18), 0xaabb_0020);
        }
    }

    #[test]
    fn masked_writes_preserve_unselected_fields_and_status_events() {
        let c = scsi_like_with_policy(PciConfigPolicy {
            status: 0x0200,
            status_w1c_mask: 0x8040,
            command_mask: 0x0157,
            latency_mask: 0xff,
            cache_line_mask: 0xff,
            registers: &[PciRegisterPolicy {
                reg_dword: 18,
                initial: 0xaabb_0000,
                writable_mask: 0xffff,
            }],
        });
        c.set_status_bits(0x8040);
        c.write(1, 0x0157);
        c.write(3, 0x1234);
        c.write(5, 0x1234_5000);
        c.write(4, 0x1201);
        for (reg, value, mask, expected) in [
            (1, 0xffff_0002, 0xff, 0x8240_0102),
            (1, 0xffff_0000, 0x8000_0000, 0x0240_0102),
            (1, 0x0040_0000, 0x00ff_0000, 0x0200_0102),
            (1, u32::MAX, 0, 0x0200_0102),
            (3, 0xffff_00ab, 0xff, 0x12ab),
            (5, 0xabcd_ffff, 0xffff_0000, 0xabcd_5000),
            (4, 0x3400, 0xff00, 0x3401),
            (15, u32::MAX, 0xff00, 0x0100),
            (18, u32::MAX, 0xff, 0xaabb_00ff),
        ] {
            c.write_masked(reg, value, mask);
            assert_eq!(c.read(reg), expected, "register {reg}");
        }
        c.set_status_bits(u16::MAX);
        c.reset();
        assert_eq!(c.read(1), 0x0200_0000);
        let default = scsi_like();
        default.set_status_bits(u16::MAX);
        assert_eq!(default.read(1), 0x0200_0000);
    }

    #[test]
    fn snapshots_restore_registers_with_destination_write_policies() {
        use crate::traits::Saveable;
        let src = scsi_like_with_policy(PciConfigPolicy {
            status: 0x02b0,
            status_w1c_mask: 0xc000,
            command_mask: 0x0157,
            latency_mask: 0xff,
            cache_line_mask: 0xff,
            registers: &[PciRegisterPolicy {
                reg_dword: 18,
                initial: 0,
                writable_mask: 0x100,
            }],
        });
        src.set_status_bits(0x8000);
        src.write(3, 0x1234);
        src.write(18, 0x100);
        let saved = src.save_state();
        for (events, command, cache) in [(0, 7, 0x0f), (0xc000, 0x0157, 0xff)] {
            let dst = scsi_like_with_policy(PciConfigPolicy {
                status: 0x0200,
                status_w1c_mask: events,
                command_mask: command,
                latency_mask: 0,
                cache_line_mask: cache,
                registers: &[PciRegisterPolicy {
                    reg_dword: 18,
                    initial: 0x20,
                    writable_mask: 0x200,
                }],
            });
            dst.load_state(&saved).unwrap();
            assert_eq!(dst.save_state(), saved);
            assert_eq!(dst.read(1), 0x0200_0000 | ((0x8000 & u32::from(events)) << 16));
            dst.write_masked(1, u32::MAX, u32::MAX);
            dst.write(3, u32::MAX);
            dst.write(18, u32::MAX);
            assert_eq!(dst.read(1), 0x0200_0000 | u32::from(command));
            assert_eq!(dst.read(3), (0x1234 & !u32::from(cache)) | u32::from(cache));
            assert_eq!(dst.read(18), 0x300);
            dst.reset();
            assert_eq!(dst.read(18), 0x20);
        }
    }

    struct Nothing;
    impl BusDevice for Nothing {}

    struct Func(PciConfig);
    impl PciBusDevice for Func {
        fn config(&self) -> &PciConfig { &self.0 }
        fn bar_read(&self, _: usize, off: u32, _: usize) -> u32 { off }
        fn bar_write(&self, _: usize, _: u32, _: u32, _: usize) {}
    }

    /// Whole pages of a window a BAR decodes become runs at their CPU
    /// addresses; an I/O BAR smaller than a page, a page shared by two
    /// BARs, and a BAR outside the window do not.
    #[test]
    fn bar_runs_are_the_pages_one_bar_decodes_whole() {
        let a: Arc<dyn PciBusDevice> = Arc::new(Func(scsi_like()));
        let b: Arc<dyn PciBusDevice> = Arc::new(Func(scsi_like()));
        for (f, io, mem) in [(&a, 0x100, 0x8000_1000u32), (&b, 0x200, 0x8000_3000)] {
            f.config().write(4, io);
            f.config().write(5, mem);
            f.config().write(1, PCI_COMMAND_IO | PCI_COMMAND_MEMORY);
        }
        let funcs = vec![(1, a.clone()), (2, b.clone())];
        let mut seen = Vec::new();
        let runs = bar_runs(&funcs, Space::Mem, 0x1a00_0000, 0x8000_0000, 0x200_0000, |at| {
            seen.push((at.slot, at.bar, at.cpu_base));
            Arc::new(Nothing) as Arc<dyn BusDevice>
        });
        let spans: Vec<_> = runs.iter().map(|r| (r.start, r.end)).collect();
        assert_eq!(spans, vec![(0x1a00_1000, 0x1a00_2000), (0x1a00_3000, 0x1a00_4000)]);
        assert_eq!(seen, vec![(1, 1, 0x1a00_1000), (2, 1, 0x1a00_3000)]);
        assert!(bar_runs(&funcs, Space::Io, 0x1800_0000, 0, 0x200_0000, |_| unreachable!()).is_empty(), "I/O BARs are smaller than a page");
        b.config().write(5, 0x8000_1000);
        assert!(bar_runs(&funcs, Space::Mem, 0x1a00_0000, 0x8000_0000, 0x200_0000, |_| unreachable!()).is_empty(), "two BARs on one page");
        assert!(bar_runs(&funcs, Space::Mem, 0x1a00_0000, 0x9000_0000, 0x200_0000, |_| unreachable!()).is_empty(), "outside the window");
    }

}
