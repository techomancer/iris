//! PCI: the configuration header every function has, and the trait a
//! function implements to sit behind a host bridge.
//!
//! The bridge (the root complex) owns everything machine-specific: how
//! configuration cycles are generated, where the CPU sees PCI memory, byte
//! lanes, and which interrupt input each slot's pins are wired to. A
//! function only sees configuration registers, accesses to its BARs, and an
//! [`IrqLine`] for its pin.

use std::sync::Mutex;

use crate::irq::IrqLine;
use crate::snapshot::{get_field, load_u32_slice, u32_slice_to_toml};

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
}

impl PciConfig {
    /// `class` is the 24-bit class/subclass/interface, as it sits in the top
    /// three bytes of register 0x08. `pin` is 0 for none, or [`PCI_INTA`]
    /// and on; the line register is left for firmware to assign.
    pub fn new(vendor: u16, device: u16, class: u32, revision: u8, bars: [Bar; 6], pin: u8) -> Self {
        let mut r = [0u32; 64];
        r[0] = ((device as u32) << 16) | vendor as u32;
        r[2] = (class << 8) | revision as u32;
        r[15] = (pin as u32) << 8;
        for (i, b) in bars.iter().enumerate() {
            if let Bar::Io(_) = b {
                r[4 + i] = 1;
            }
        }
        Self { regs: Mutex::new(r), bars }
    }

    /// Register `reg` (a dword index, 0..64).
    pub fn read(&self, reg: usize) -> u32 {
        let r = self.regs.lock().unwrap();
        match reg {
            // Status: a medium-speed device with no errors to report.
            1 => (r[1] & 0xffff) | 0x0200_0000,
            _ => r[reg],
        }
    }

    pub fn write(&self, reg: usize, val: u32) {
        let mut r = self.regs.lock().unwrap();
        match reg {
            // Identity, class and header type are read-only.
            0 | 2 | 3 => {}
            // Command; status bits are write-one-to-clear and none are set.
            1 => r[1] = val & 0xffff,
            4..=9 => {
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
            15 => r[15] = (r[15] & 0xff00) | (val & !0xff00),
            _ => r[reg] = val,
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn scsi_like() -> PciConfig {
        PciConfig::new(0x9004, 0x8078, 0x0100_00, 1,
            [Bar::Io(0x100), Bar::Mem(0x1000), Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused], PCI_INTA)
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
    fn the_interrupt_pin_is_read_only() {
        let c = scsi_like();
        c.write(15, 0x0000_0009);
        assert_eq!(c.read(15) & 0xffff, 0x0109, "line 9 assigned, pin A kept");
        c.write(15, 0);
        assert_eq!(c.interrupt_pin(), PCI_INTA);
    }
}
