//! The Bridge, widget 15 of the Octane: the XIO-to-PCI bridge on the IP30
//! system board, with the on-board PCI devices behind it and the PROM's
//! flash in its external flash select.
//!
//! Offsets are within the widget's 16 MB window, laid out as
//! `docs/Octane/ip30-address-map.md` §7 describes: local registers,
//! internal ATE RAM, type 0 and type 1 configuration space, the interrupt
//! acknowledge cycle, the external SSRAM (not fitted), the eight device
//! windows and the flash. A device window's accesses become PCI memory or
//! I/O cycles at the address its device register programs, and whichever
//! function's BAR decodes that address answers; nothing answering is a
//! bus error, as a PCI master abort would be.
//!
//! Interrupts: a function's pin is a Bridge interrupt pin (device n on
//! pin n), and an enabled pin rising sends its vector to HEART.
//!
//! Local registers without a model read back what was written to them.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::dev::sflash::{Flash, FLASH_SIZE};
use crate::dev::sgi_nic::{Ds2502, OneWire};
use crate::dev::xbow::{widget_id, WIDGET_ID};
use crate::pci::{Bar, PciBusDevice, PciConfig, Space, PCI_INTA};
use crate::traits::{BusDevice, BusRead16, BusRead32, BusRead64, BusRead8, BUS_OK};

const WIDGET_STATUS: u32 = 0x0c;
const STAT_PCI_GIO_N: u32 = 1 << 5;
const BRIDGE_WID_CONTROL: u32 = 0x24;
const BRIDGE_WID_INT_UPPER: u32 = 0x34;
const BRIDGE_WID_INT_LOWER: u32 = 0x3c;
const BRIDGE_NIC: u32 = 0xb4;
const BRIDGE_INT_STATUS: u32 = 0x104;
const BRIDGE_INT_ENABLE: u32 = 0x10c;
const BRIDGE_INT_RST_STAT: u32 = 0x114;
const BRIDGE_INT_MODE: u32 = 0x11c;
const BRIDGE_INT_ADDR0: u32 = 0x134;
const BRIDGE_DEVICE0: u32 = 0x204;
const BRIDGE_ATE_RAM: u32 = 0x1_0000;
const BRIDGE_CONFIG_BASE: u32 = 0x2_0000;
const BRIDGE_CONFIG1_BASE: u32 = 0x2_8000;
const BRIDGE_IACK: u32 = 0x3_0000;
const BRIDGE_EXT_SSRAM: u32 = 0x8_0000;
const BRIDGE_DEVIO0: u32 = 0x20_0000;
const BRIDGE_FLASH: u32 = 0xc0_0000;
const BRIDGE_FLASH_ALT: u32 = 0xe0_0000;

/// Device register bits.
const DEV_SWAP: u32 = 1 << 13;
const DEV_IO_MEM: u32 = 1 << 12;
const DEV_BASE_MASK: u32 = 0xfff;

/// Which device window an offset is in, and where in it.
fn devio_slot(off: u32) -> Option<(usize, u32)> {
    match off {
        0x20_0000..=0x3f_ffff => Some((0, off - 0x20_0000)),
        0x40_0000..=0x5f_ffff => Some((1, off - 0x40_0000)),
        0x60_0000..=0xbf_ffff => {
            let n = (off - 0x60_0000) >> 20;
            Some((2 + n as usize, (off - 0x60_0000) & 0xf_ffff))
        }
        _ => None,
    }
}

/// A PCI function that is only a register file: what stands in for the
/// QL1040s and the RAD until they are modelled. Each BAR remembers what
/// it is told.
pub struct PciStub {
    name: &'static str,
    cfg: PciConfig,
    cells: Mutex<HashMap<(usize, u32), u32>>,
}

impl PciStub {
    pub fn new(name: &'static str, vendor: u16, device: u16, class: u32, bars: [Bar; 6]) -> Self {
        PciStub { name, cfg: PciConfig::new(vendor, device, class, 1, bars, PCI_INTA), cells: Mutex::new(HashMap::new()) }
    }

    pub fn name(&self) -> &'static str {
        self.name
    }
}

impl PciBusDevice for PciStub {
    fn config(&self) -> &PciConfig {
        &self.cfg
    }
    fn bar_read(&self, bar: usize, off: u32, width: usize) -> u32 {
        let w = *self.cells.lock().get(&(bar, off & !3)).unwrap_or(&0);
        match width {
            1 => (w >> (8 * (3 - (off & 3)))) & 0xff,
            2 => (w >> (8 * (2 - (off & 2)))) & 0xffff,
            _ => w,
        }
    }
    fn bar_write(&self, bar: usize, off: u32, val: u32, width: usize) {
        let mut c = self.cells.lock();
        let w = c.entry((bar, off & !3)).or_insert(0);
        match width {
            1 => {
                let sh = 8 * (3 - (off & 3));
                *w = (*w & !(0xff << sh)) | ((val & 0xff) << sh);
            }
            2 => {
                let sh = 8 * (2 - (off & 2));
                *w = (*w & !(0xffff << sh)) | ((val & 0xffff) << sh);
            }
            _ => *w = val,
        }
    }
}

struct BridgeState {
    regs: HashMap<u32, u32>,
    nic: OneWire,
    /// The PCI interrupt pins as the devices drive them.
    pins: u32,
}

pub struct Bridge {
    s: Mutex<BridgeState>,
    slots: [Option<Arc<dyn PciBusDevice>>; 8],
    pub flash: Flash,
    /// Where interrupt packets go: HEART's ISR, by vector.
    irq_out: Mutex<Option<Arc<dyn Fn(u32, bool) + Send + Sync>>>,
}

impl Bridge {
    /// `slots` is the PCI bus; `flash` the part in the external flash select.
    pub fn new(slots: [Option<Arc<dyn PciBusDevice>>; 8], flash: Flash) -> Self {
        // The system board's own NIC, and the CPU board's on the same wire
        // (the PROM never drives HEART's MicroLAN master): part numbers
        // and barcodes as on snorlax.
        let nic = OneWire::with_all(vec![
            Ds2502::board_nic(0x0030_0887, "IP30", "030-0887-003", "G", "FGT687"),
            Ds2502::board_nic(0x0030_0888, "PM10", "030-0888-004", "D", "DYV140"),
        ]);
        Bridge {
            s: Mutex::new(BridgeState { regs: HashMap::new(), nic, pins: 0 }),
            slots,
            flash,
            irq_out: Mutex::new(None),
        }
    }

    /// Where the Bridge's interrupt packets go.
    pub fn connect_interrupts(&self, out: Arc<dyn Fn(u32, bool) + Send + Sync>) {
        *self.irq_out.lock() = Some(out);
    }

    pub fn power_on(&self) {
        let mut s = self.s.lock();
        s.regs.clear();
        s.pins = 0;
    }

    fn reg(&self, off: u32) -> u32 {
        match self.s.lock().regs.get(&off) {
            Some(&v) => v,
            // Device windows come up as PCI memory space at address 0: the
            // PROM reaches the RAD through window 3 without programming
            // either its device register or the RAD's BAR.
            None if (BRIDGE_DEVICE0..BRIDGE_DEVICE0 + 64).contains(&off) && (off - BRIDGE_DEVICE0) % 8 == 0 => DEV_IO_MEM,
            None => 0,
        }
    }

    /// Send pin `pin`'s vector to HEART if the pin is enabled.
    fn forward(&self, pin: u32, level: bool) {
        let (enabled, mode, vec) = {
            let s = self.s.lock();
            (
                s.regs.get(&BRIDGE_INT_ENABLE).copied().unwrap_or(0),
                s.regs.get(&BRIDGE_INT_MODE).copied().unwrap_or(0),
                s.regs.get(&(BRIDGE_INT_ADDR0 + 8 * pin)).copied().unwrap_or(0) & 0xff,
            )
        };
        if enabled & (1 << pin) == 0 {
            return;
        }
        // A rising pin sends its vector; a falling one sends a clear only
        // when interrupt mode asks for it.
        if level || mode & (1 << pin) != 0 {
            if let Some(out) = self.irq_out.lock().clone() {
                out(vec, level);
            }
        }
    }

    fn config_slot(off: u32) -> (usize, usize, usize) {
        let o = off - BRIDGE_CONFIG_BASE;
        ((o >> 12) as usize & 7, (o >> 8) as usize & 7, (o & 0xff) as usize)
    }

    /// A configuration dword. An absent function is a PCI master abort,
    /// which the Bridge turns into a bus error on the PIO: the PROM probes
    /// the slots expecting exactly that, and lists a slot that answers all
    /// ones as an unknown device.
    fn config_read(&self, off: u32) -> Option<u32> {
        let (slot, func, reg) = Self::config_slot(off);
        match &self.slots[slot] {
            Some(d) if func == 0 => Some(d.config().read(reg / 4)),
            _ => None,
        }
    }

    fn config_write(&self, off: u32, val: u32) {
        let (slot, func, reg) = Self::config_slot(off);
        if let (Some(d), 0) = (&self.slots[slot], func) {
            d.config().write(reg / 4, val);
        }
    }

    fn reg_read32(&self, off: u32) -> u32 {
        match off {
            WIDGET_ID => widget_id(4, 0xc002, 0x036), // Bridge rev D
            // Status: a PCI bus (bit 5), not GIO; flash select 0.
            WIDGET_STATUS => self.reg(off) | STAT_PCI_GIO_N,
            BRIDGE_INT_STATUS => {
                let s = self.s.lock();
                (s.regs.get(&off).copied().unwrap_or(0) & !0xff) | (s.pins & 0xff)
            }
            BRIDGE_NIC => self.s.lock().nic.read(),
            o => self.reg(o),
        }
    }

    fn reg_write32(&self, off: u32, val: u32) {
        match off {
            BRIDGE_NIC => self.s.lock().nic.write(val),
            // Write 1s to clear groups of error bits: nothing is ever set.
            BRIDGE_INT_RST_STAT => {}
            BRIDGE_INT_STATUS => {}
            o => {
                self.s.lock().regs.insert(o, val);
                if o == BRIDGE_INT_ENABLE {
                    // Pins already high become visible when enabled.
                    let pins = self.s.lock().pins;
                    for pin in 0..8 {
                        if pins & (1 << pin) != 0 && val & (1 << pin) != 0 {
                            self.forward(pin, true);
                        }
                    }
                }
            }
        }
    }

    /// The PCI address and space a device window's offset goes out as.
    fn devio_target(&self, slot: usize, o: u32) -> (Space, u32, bool) {
        let d = self.reg(BRIDGE_DEVICE0 + 8 * slot as u32);
        let space = if d & DEV_IO_MEM != 0 { Space::Mem } else { Space::Io };
        (space, ((d & DEV_BASE_MASK) << 20).wrapping_add(o), d & DEV_SWAP != 0)
    }

    /// The function whose BAR decodes `addr` in `space`. A BAR the
    /// firmware never moved decodes at zero once its space is enabled.
    fn claim(&self, space: Space, addr: u32) -> Option<(&Arc<dyn PciBusDevice>, usize, u32)> {
        self.slots.iter().flatten().find_map(|d| d.config().decode_unmoved(space, addr).map(|(bar, off)| (d, bar, off)))
    }

    fn devio_read(&self, slot: usize, o: u32, size: u32) -> Option<u64> {
        let (space, pci, swap) = self.devio_target(slot, o);
        let one = |pci: u32, width: u32| -> Option<u32> {
            let pci = if swap { lane_swap(pci, width) } else { pci };
            let (d, bar, off) = self.claim(space, pci)?;
            let v = d.bar_read(bar, off, width as usize);
            Some(if swap { swap_data(v, width) } else { v })
        };
        Some(match size {
            8 => (one(pci, 4)? as u64) << 32 | one(pci + 4, 4)? as u64,
            n => one(pci, n)? as u64,
        })
    }

    fn devio_write(&self, slot: usize, o: u32, size: u32, val: u64) -> bool {
        let (space, pci, swap) = self.devio_target(slot, o);
        let one = |pci: u32, width: u32, v: u32| -> bool {
            let pci = if swap { lane_swap(pci, width) } else { pci };
            let Some((d, bar, off)) = self.claim(space, pci) else { return false };
            d.bar_write(bar, off, if swap { swap_data(v, width) } else { v }, width as usize);
            true
        };
        match size {
            8 => one(pci, 4, (val >> 32) as u32) && one(pci + 4, 4, val as u32),
            n => one(pci, n, val as u32),
        }
    }

    /// An access of `size` bytes at `off` in the widget's window.
    pub fn read(&self, off: u32, size: u32) -> Option<u64> {
        if off < BRIDGE_ATE_RAM {
            return Some(match size {
                8 => self.reg_read32(off | 4) as u64,
                _ => narrow32(self.reg_read32(off & !3), off, size),
            });
        }
        if (BRIDGE_CONFIG_BASE..BRIDGE_CONFIG1_BASE).contains(&off) {
            // Configuration space is seen big-endian: an aligned dword is
            // the PCI register with its bytes reversed, so PCI byte n is at
            // offset n ^ 3 (Linux reads `where ^ (4 - size)`).
            if size == 8 {
                let hi = self.config_read(off & !7)? as u64;
                return Some(hi << 32 | self.config_read((off & !7) + 4)? as u64);
            }
            return Some(narrow32(self.config_read(off & !3)?, off, size));
        }
        if let Some((slot, o)) = devio_slot(off) {
            return self.devio_read(slot, o, size);
        }
        if (BRIDGE_FLASH..BRIDGE_FLASH + FLASH_SIZE).contains(&off) {
            let a = (off - BRIDGE_FLASH) as u64;
            return Some(match size {
                1 => self.flash.read8(a).data as u64,
                2 => self.flash.read16(a).data as u64,
                4 => self.flash.read32(a).data as u64,
                _ => self.flash.read64(a).data,
            });
        }
        let ones = u64::MAX >> (64 - 8 * size);
        match off {
            // The ATE RAM reads back what was written to its halves.
            BRIDGE_ATE_RAM..=0x1_ffff => Some(narrow32(self.reg(off & !3), off, size)),
            // Type 1 configuration: there is no bus behind a bridge here.
            BRIDGE_CONFIG1_BASE..=0x2_ffff => Some(ones),
            // The interrupt acknowledge cycle and the external SSRAM, which
            // this board does not have.
            BRIDGE_IACK..=0x3_ffff | BRIDGE_EXT_SSRAM..=0xf_ffff => Some(0),
            // Past the part in the first flash select, and the second
            // select, which has no part fitted.
            0xc0_0000..=0xff_ffff => Some(ones),
            _ => None,
        }
    }

    pub fn write(&self, off: u32, size: u32, val: u64) -> bool {
        if off < BRIDGE_ATE_RAM {
            if size == 8 {
                self.reg_write32(off | 4, val as u32);
            } else {
                let old = self.reg_read32(off & !3);
                self.reg_write32(off & !3, widen32(old, off, size, val as u32));
            }
            return true;
        }
        if (BRIDGE_CONFIG_BASE..BRIDGE_CONFIG1_BASE).contains(&off) {
            if size == 8 {
                self.config_write(off & !7, (val >> 32) as u32);
                self.config_write((off & !7) + 4, val as u32);
            } else {
                let Some(old) = self.config_read(off & !3) else { return false };
                self.config_write(off & !3, widen32(old, off, size, val as u32));
            }
            return true;
        }
        if let Some((slot, o)) = devio_slot(off) {
            return self.devio_write(slot, o, size, val);
        }
        if (BRIDGE_FLASH..BRIDGE_FLASH + FLASH_SIZE).contains(&off) {
            let a = (off - BRIDGE_FLASH) as u64;
            // The flash is 16 bits wide; the Bridge's write enable gates
            // programming but the PROM's own flash code sets it first.
            let _ = self.reg(BRIDGE_WID_CONTROL);
            match size {
                1 => self.flash.write8(a, val as u8),
                2 => self.flash.write16(a, val as u16),
                4 => self.flash.write32(a, val as u32),
                _ => self.flash.write64(a, val),
            };
            return true;
        }
        match off {
            BRIDGE_ATE_RAM..=0x1_ffff => {
                self.s.lock().regs.insert(off & !3, val as u32);
                true
            }
            BRIDGE_CONFIG1_BASE..=0x3_ffff | BRIDGE_EXT_SSRAM..=0xf_ffff | 0xc0_0000..=0xff_ffff => true,
            _ => false,
        }
    }

    /// The interrupt destination the guest programmed: (target widget,
    /// address), for the trace.
    pub fn interrupt_destination(&self) -> (u32, u64) {
        let s = self.s.lock();
        let up = s.regs.get(&BRIDGE_WID_INT_UPPER).copied().unwrap_or(0);
        let lo = s.regs.get(&BRIDGE_WID_INT_LOWER).copied().unwrap_or(0);
        ((up >> 16) & 0xf, ((up & 0xffff) as u64) << 32 | lo as u64)
    }
}

/// PCI pin `line` of a device, driven by that device.
impl crate::irq::InterruptController for Bridge {
    fn set_line(&self, line: u32, level: bool) {
        let pin = line & 7;
        let changed = {
            let mut s = self.s.lock();
            let old = s.pins;
            s.pins = if level { old | 1 << pin } else { old & !(1 << pin) };
            s.pins != old
        };
        if changed {
            self.forward(pin, level);
        }
    }
}

/// The `size` bytes at `addr` of the 32-bit register `v`, big-endian lanes.
pub fn narrow32(v: u32, addr: u32, size: u32) -> u64 {
    if size >= 4 {
        return v as u64;
    }
    let sh = 8 * (4 - size - (addr & 3 & !(size - 1)));
    ((v >> sh) as u64) & ((1u64 << (8 * size)) - 1)
}

/// `old` with the `size` bytes at `addr` replaced by `val`.
pub fn widen32(old: u32, addr: u32, size: u32, val: u32) -> u32 {
    if size >= 4 {
        return val;
    }
    let sh = 8 * (4 - size - (addr & 3 & !(size - 1)));
    let m = ((1u32 << (8 * size)) - 1) << sh;
    (old & !m) | ((val << sh) & m)
}

/// The byte-lane swap a device window with its swap bit set applies: a
/// little-endian function's register n is reached from a big-endian bus
/// at the mirrored lane.
fn lane_swap(addr: u32, width: u32) -> u32 {
    match width {
        1 => addr ^ 3,
        2 => addr ^ 2,
        _ => addr,
    }
}

fn swap_data(v: u32, width: u32) -> u32 {
    match width {
        2 => (v as u16).swap_bytes() as u32,
        4 => v.swap_bytes(),
        _ => v,
    }
}

/// What the Bridge's PCI bus looks like on an Octane: two QL1040Bs, the
/// IOC3 and the RAD. `ioc3` is the real one; the others are register files.
pub fn octane_slots(ioc3: Arc<dyn PciBusDevice>) -> [Option<Arc<dyn PciBusDevice>>; 8] {
    let ql = |name| -> Arc<dyn PciBusDevice> {
        Arc::new(PciStub::new(name, 0x1077, 0x1020, 0x01_0000,
            [Bar::Io(0x100), Bar::Mem(0x1000), Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused]))
    };
    let rad: Arc<dyn PciBusDevice> = Arc::new(PciStub::new("rad", 0x10a9, 0x0005, 0x04_0100,
        [Bar::Mem(0x10_0000), Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused]));
    [Some(ql("ql0")), Some(ql("ql1")), Some(ioc3), Some(rad), None, None, None, None]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dev::ioc3::Ioc3;

    fn bridge() -> Bridge {
        let ioc3: Arc<dyn PciBusDevice> = Arc::new(Ioc3::new([8, 0, 0x69, 1, 2, 3], None));
        Bridge::new(octane_slots(ioc3), Flash::new(&[0x12, 0x34, 0x56, 0x78], None))
    }

    #[test]
    fn pci_config_finds_the_ioc3_and_sizes_its_bar() {
        let b = bridge();
        let cfg = |slot: u32, reg: u32| BRIDGE_CONFIG_BASE + slot * 0x1000 + reg;
        assert_eq!(b.read(cfg(2, 0), 4), Some(0x0003_10a9));
        assert_eq!(b.read(cfg(5, 0), 4), None, "an empty slot is a bus error");
        assert!(b.write(cfg(2, 0x10), 4, 0xffff_ffff));
        assert_eq!(b.read(cfg(2, 0x10), 4), Some(0xfff0_0000));
        assert_eq!(b.read(cfg(2, 0), 2), Some(0x0003)); // device ID: big-endian lanes
        assert_eq!(b.read(cfg(2, 3), 1), Some(0xa9)); // PCI byte 0
        assert_eq!(b.read(cfg(0, 0), 4), Some(0x1020_1077));
    }

    #[test]
    fn a_device_window_reaches_the_bar_it_is_pointed_at() {
        let b = bridge();
        let cfg = |slot: u32, reg: u32| BRIDGE_CONFIG_BASE + slot * 0x1000 + reg;
        // Nothing decodes before the PROM sets the window up.
        assert_eq!(b.read(0x60_0000 + 0x20178 + 7, 1), None);
        // As PROM 4.17 does: memory space enabled, IOC3's BAR0 at PCI
        // 0x400000, window 2 there.
        b.write(cfg(2, 0x04), 4, 0x2);
        b.write(cfg(2, 0x10), 4, 0x0040_0000);
        b.write(BRIDGE_DEVICE0 + 16, 4, 0x1004);
        b.write(0x60_0000 + 0x20178 + 7, 1, 0x5a); // UART A scratch
        assert_eq!(b.read(0x60_0000 + 0x20178 + 7, 1), Some(0x5a));
        assert_eq!(b.read(0x60_0000 + 0x20178 + 5, 1).map(|v| v & 0x20), Some(0x20));
    }

    #[test]
    fn a_window_reaches_an_unmoved_device_once_enabled() {
        let b = bridge();
        let cfg = |slot: u32, reg: u32| BRIDGE_CONFIG_BASE + slot * 0x1000 + reg;
        // As PROM 4.17 does with the RAD: memory enable, then straight
        // through window 3, no BAR and no device register written.
        assert_eq!(b.read(0x70_0000 + 0x7c, 4), None, "not enabled");
        b.write(cfg(3, 0x04), 4, 0x6);
        assert!(b.write(0x70_0000 + 0x7c, 4, 0xc0c0_c0c0));
        assert_eq!(b.read(0x70_0000 + 0x7c, 4), Some(0xc0c0_c0c0));
        assert_eq!(b.read(BRIDGE_DEVICE0 + 8 * 3, 4), Some(0x1000), "memory space, base 0 at reset");
    }

    #[test]
    fn the_flash_is_in_the_external_select() {
        let b = bridge();
        assert_eq!(b.read(BRIDGE_FLASH, 4), Some(0x1234_5678));
        assert_eq!(b.read(BRIDGE_FLASH_ALT, 4), Some(0xffff_ffff));
        assert_eq!((b.read(WIDGET_ID, 4).unwrap() >> 12) & 0xffff, 0xc002);
    }

    #[test]
    fn an_enabled_pin_sends_its_vector() {
        use crate::irq::InterruptController;
        let b = bridge();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let s = seen.clone();
        b.connect_interrupts(Arc::new(move |v, l| s.lock().push((v, l))));
        b.write(BRIDGE_INT_ADDR0 + 8 * 4, 4, 0x10); // pin 4: vector 16
        b.set_line(4, true);
        assert!(seen.lock().is_empty(), "not enabled yet");
        b.write(BRIDGE_INT_ENABLE, 4, 1 << 4);
        assert_eq!(*seen.lock(), vec![(16, true)]);
        b.set_line(4, false);
        assert_eq!(seen.lock().len(), 1, "no clear packet without interrupt mode");
        assert_eq!(b.read(BRIDGE_INT_STATUS, 4), Some(0));
    }
}
