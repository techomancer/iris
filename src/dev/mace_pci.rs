//! MACE's PCI host bridge on the O2: the root complex. It generates
//! configuration cycles, gives the CPU its windows onto PCI memory (swapping
//! byte lanes on the way, since PCI is little-endian and the CPU is not), and
//! wires each slot's interrupt pin to its CRIME input.

use std::sync::{Arc, Mutex};

use crate::irq::{InterruptController, IrqLine};
use crate::dev::crime_mem::O2Memory;
use crate::pci::{PciBusDevice, PciDma, PciDmaStatus, Space};
use crate::snapshot::{get_field, hex_u32, load_u32_slice, toml_u32, u32_slice_to_toml};
use crate::traits::{BusDevice, BusRead32, BusRead64, Saveable, BUS_OK};
use crate::ip32::*;

pub mod macepci_reg {
    pub const ERROR_ADDR: u32 = 0x0000;
    pub const ERROR_FLAGS: u32 = 0x0004;
    pub const CONTROL: u32 = 0x0008;
    pub const REVISION: u32 = 0x000c;
    pub const CONFIG_ADDR: u32 = 0x0cf8;
    pub const CONFIG_DATA: u32 = 0x0cfc;
}

/// What a MACE PCI bridge reports for itself. Zero here would make the PROM
/// conclude the bridge is missing.
pub const MACEPCI_REVISION_VALUE: u32 = 1;

/// The two SCSI controllers an O2 has on its motherboard. CRIME carries a
/// separate interrupt line for each, which is the clearest evidence that
/// both are expected to be there.
pub const IP32_SCSI_SLOT: u8 = 1;
pub const IP32_SCSI_SLOT_B: u8 = 2;

/// Which of CRIME's interrupt inputs the pin of the function in that slot
/// drives. NetBSD works this out for itself and prints it: "interrupting at
/// crime interrupt 8".
pub const IP32_SCSI_CRIME_INT: u32 = 8;
pub const IP32_SCSI_CRIME_INT_B: u32 = 9;

/// The O2's routing: which CRIME input a slot's pin (1 = INTA) drives.
fn crime_input(slot: u8, pin: u8) -> Option<u32> {
    match (slot, pin) {
        (IP32_SCSI_SLOT, 1) => Some(IP32_SCSI_CRIME_INT),
        (IP32_SCSI_SLOT_B, 1) => Some(IP32_SCSI_CRIME_INT_B),
        _ => None,
    }
}

/// Bus 0 has 32 device numbers, each with four interrupt pins.
const PCI_PINS: usize = 32 * 4;

/// The bridge's interrupt router. Each slot's INTA..INTD is a line
/// (`slot * 4 + pin - 1`); the router keeps every pin's level and drives
/// each system input it routes to with the OR of the pins routed there, so
/// on a shared input one function cannot clear another's interrupt.
struct PciIrqRouter {
    /// Where the system inputs are: CRIME.
    intc: Arc<dyn InterruptController>,
    /// Slot and pin (1 = INTA) to system input.
    route: fn(u8, u8) -> Option<u32>,
    levels: Mutex<[bool; PCI_PINS]>,
}

impl PciIrqRouter {
    fn line(slot: u8, pin: u8) -> u32 {
        slot as u32 * 4 + (pin as u32 - 1)
    }

    fn slot_pin(line: u32) -> (u8, u8) {
        ((line / 4) as u8, (line % 4) as u8 + 1)
    }
}

impl InterruptController for PciIrqRouter {
    fn set_line(&self, line: u32, level: bool) {
        if line as usize >= PCI_PINS {
            return;
        }
        let (slot, pin) = Self::slot_pin(line);
        let Some(input) = (self.route)(slot, pin) else { return };
        // The lock is held while CRIME is driven, so two pins sharing an
        // input cannot reach it out of order.
        let mut levels = self.levels.lock().unwrap();
        levels[line as usize] = level;
        let any = (0..PCI_PINS as u32).any(|l| {
            let (s, p) = Self::slot_pin(l);
            levels[l as usize] && (self.route)(s, p) == Some(input)
        });
        self.intc.set_line(input, any);
    }
}

/// The CPU-side window onto PCI memory, and the PCI address it starts at.
///
/// These are not the same number, which is the whole point. The PROM programs
/// a BAR with a *PCI* address — 0x80001000 — and then reaches it through a
/// window low in the CPU's physical map. The bus-error register dump named
/// `0xba001084`, physical 0x1a001084, which is this window plus 0x1084, which
/// is that BAR plus 0x84. Mapping the window one-to-one instead would put the
/// device nowhere the firmware ever looks.
pub const PCI_MEM_BASE: u32 = 0x1a00_0000;
pub const PCI_MEM_SIZE: u32 = 0x0200_0000;
pub const PCI_MEM_PCI_BASE: u32 = 0x8000_0000;

/// The CPU-side window onto PCI I/O space: port `n` is at `PCI_IO_BASE + n`.
/// Linux probes legacy ISA ports through it (the i8042 at 0x60/0x64, the
/// 8250s at 0x3f8...), which nothing on an O2 answers: the bridge ends such a
/// cycle with a master abort and the read floats high, rather than taking a
/// bus error.
pub const PCI_IO_BASE: u32 = 0x1800_0000;
pub const PCI_IO_SIZE: u32 = 0x0200_0000;

/// PCI memory's second, one-to-one window, from physical 0x2_8000_0000. A
/// 64-bit kernel (OpenBSD's) reaches devices there; the PROM and NetBSD use
/// the low window above. Physical addresses reach this bus as 32 bits, so it
/// arrives as 0x8000_0000 and up. CRIME's no-ECC memory alias has the same
/// low 32 bits (0x00_8000_0000); the CPU folds that one onto memory before it
/// gets here ([`crate::ip32::NO_ECC_BASE`]).
pub const PCI_HI_MEM_BASE: u32 = 0x8000_0000;

/// Host memory on the PCI bus, as bus masters see it (MACE spec 9.3.1): the
/// low gigabyte is the "byte-swapped view", the next the "native view",
/// both onto the same gigabyte of memory, and above that is not host memory.
pub const PCI_SWAPPED_VIEW: u32 = 0x0000_0000;
pub const PCI_NATIVE_VIEW: u32 = 0x4000_0000;
pub const PCI_HOST_END: u32 = 0x8000_0000;

/// What a bus master reaches through the bridge: host memory in its two
/// views, which reach the same bytes as the CPU's own decode.
///
/// Both views hand bytes over in address order. The spec has the low view
/// byte-swapped, but every driver run here needs address order in both:
/// IRIX's HIM keeps its descriptors in the native view and its CDBs in the
/// low one, the BSDs give the chip physical addresses in either range for
/// the same little-endian descriptors and data, and swapping the native
/// view's 32-bit words broke OpenBSD's writes (programs crashed after a
/// reboot). Word order is the device's business, settled per driver.
pub struct HostMemory {
    ram: Arc<O2Memory>,
}

impl HostMemory {
    pub fn new(ram: Arc<O2Memory>) -> Self {
        Self { ram }
    }

    /// The memory byte bus address `a` reaches, if any.
    fn locate(m: &O2Memory, a: u64) -> Option<usize> {
        if a >= PCI_HOST_END as u64 {
            return None;
        }
        m.dma_offset(a as u32)
    }
}

impl HostMemory {
    /// Run `f` over the transfer `[addr, addr + len)` a page at a time,
    /// with the offset each page starts at in memory, or None where the bus
    /// address is not memory. A page never straddles a bank slot, so its
    /// bytes are contiguous: one decode per page, not per byte.
    fn each_page(&self, addr: u64, len: usize, mut f: impl FnMut(std::ops::Range<usize>, Option<usize>)) {
        const PAGE: usize = 4096;
        let mut done = 0;
        while done < len {
            let a = addr.wrapping_add(done as u64);
            let n = (PAGE - (a as usize & (PAGE - 1))).min(len - done);
            f(done..done + n, Self::locate(&self.ram, a));
            done += n;
        }
    }
}

/// No lock: the controller's thread and the CPU share memory the way the
/// hardware's masters do.
impl PciDma for HostMemory {
    fn dma_read(&self, addr: u64, buf: &mut [u8]) -> PciDmaStatus {
        let mut st = PciDmaStatus::Ok;
        self.each_page(addr, buf.len(), |r, at| match at {
            Some(i) => self.ram.read_into(i, &mut buf[r]),
            None => {
                buf[r].fill(0);
                st = PciDmaStatus::MasterAbort;
            }
        });
        st
    }

    fn dma_write(&self, addr: u64, data: &[u8]) -> PciDmaStatus {
        let mut st = PciDmaStatus::Ok;
        self.each_page(addr, data.len(), |r, at| match at {
            Some(i) => self.ram.write(i, &data[r]),
            None => st = PciDmaStatus::MasterAbort,
        });
        st
    }
}

/// A function in a slot, and the configuration registers the guest has
/// touched (first touch, with direction), for the bring-up report.
struct Slot {
    name: &'static str,
    dev: Arc<dyn PciBusDevice>,
    touched: Mutex<Vec<(u8, bool)>>,
}

pub struct MacePci {
    config_addr: Mutex<u32>,
    regs: Mutex<[u32; (MACEPCI_SIZE / 4) as usize]>,
    /// Distinct config addresses the guest selected, in order.
    probed: Mutex<Vec<u32>>,
    /// Functions by device number on bus 0.
    slots: Mutex<std::collections::BTreeMap<u8, Arc<Slot>>>,
    /// The slots' interrupt pins, routed to CRIME.
    irq: Arc<PciIrqRouter>,
    /// Host memory, for the functions that master the bus.
    dma: Arc<HostMemory>,
}

impl MacePci {
    pub fn new(intc: Arc<dyn InterruptController>, ram: Arc<O2Memory>) -> Self {
        Self::with_routing(intc, crime_input, ram)
    }

    /// A bridge with its own routing of slot pins (1 = INTA) to `intc`'s
    /// inputs.
    fn with_routing(intc: Arc<dyn InterruptController>, route: fn(u8, u8) -> Option<u32>, ram: Arc<O2Memory>) -> Self {
        Self {
            config_addr: Mutex::new(0),
            regs: Mutex::new([0u32; (MACEPCI_SIZE / 4) as usize]),
            probed: Mutex::new(Vec::new()),
            slots: Mutex::new(std::collections::BTreeMap::new()),
            irq: Arc::new(PciIrqRouter { intc, route, levels: Mutex::new([false; PCI_PINS]) }),
            dma: Arc::new(HostMemory::new(ram)),
        }
    }

    /// Put a function in a slot on bus 0, wire its interrupt pin to the
    /// bridge, which routes it on, and give it host memory to master.
    pub fn attach(&self, slot: u8, name: &'static str, dev: Arc<dyn PciBusDevice>) {
        dev.connect_dma(self.dma.clone());
        let pin = dev.config().interrupt_pin();
        if (1..=4).contains(&pin) {
            dev.connect_interrupt(IrqLine::new(self.irq.clone(), PciIrqRouter::line(slot, pin)));
        }
        self.slots.lock().unwrap().insert(slot, Arc::new(Slot { name, dev, touched: Mutex::new(Vec::new()) }));
    }

    /// The function in a slot: its name and the registers the guest touched.
    pub fn slot(&self, slot: u8) -> Option<(&'static str, Arc<dyn PciBusDevice>, Vec<(u8, bool)>)> {
        let s = self.slots.lock().unwrap().get(&slot).cloned()?;
        let touched = s.touched.lock().unwrap().clone();
        Some((s.name, s.dev.clone(), touched))
    }

    /// Whether `addr` is in one of the CPU's windows onto PCI memory.
    pub fn in_window(addr: u32) -> bool {
        (PCI_MEM_BASE..PCI_MEM_BASE + PCI_MEM_SIZE).contains(&addr)
            || Self::in_io_window(addr)
            || addr >= PCI_HI_MEM_BASE
    }

    fn in_io_window(addr: u32) -> bool {
        (PCI_IO_BASE..PCI_IO_BASE + PCI_IO_SIZE).contains(&addr)
    }

    /// The PCI address a CPU address in a window reaches.
    fn pci_addr_of(addr: u32) -> u32 {
        if addr >= PCI_HI_MEM_BASE {
            return addr;
        }
        PCI_MEM_PCI_BASE.wrapping_add(addr.wrapping_sub(PCI_MEM_BASE))
    }

    /// The bridge swaps byte lanes: an 8-bit access at CPU offset `A` reaches
    /// device byte `A ^ 3`, and a 16-bit one `A ^ 2`.
    ///
    /// This is not cosmetic. Without it the AIC-7880's initialisation reads
    /// as writes to registers 0x84, 0x91, 0x93 and 0xbc, which are nothing in
    /// particular. With it they are HCNTRL (chip reset, then pause), CLRINT
    /// (clear every interrupt), SCBPTR walking 0,1,2,... and the last byte of
    /// each SCB being cleared — which is exactly how that chip is brought up.
    pub fn lane_swizzle(off: u32, width: usize) -> u32 {
        match width {
            1 => off ^ 3,
            2 => off ^ 2,
            _ => off,
        }
    }

    /// The function decoding a PCI memory address: it, the BAR and the
    /// offset within it.
    fn mem_target(&self, space: Space, pci_addr: u32) -> Option<(Arc<dyn PciBusDevice>, usize, u32)> {
        self.slots.lock().unwrap().values().find_map(|s| {
            let (bar, off) = s.dev.config().decode(space, pci_addr)?;
            Some((s.dev.clone(), bar, off))
        })
    }

    /// The function and BAR a CPU address in a window reaches, in whichever
    /// space that window opens onto.
    fn window_target(&self, addr: u32) -> Option<(Arc<dyn PciBusDevice>, usize, u32)> {
        if Self::in_io_window(addr) {
            return self.mem_target(Space::Io, addr - PCI_IO_BASE);
        }
        self.mem_target(Space::Mem, Self::pci_addr_of(addr))
    }

    fn mem_read(&self, addr: u32, width: usize) -> u32 {
        match self.window_target(addr) {
            Some((f, bar, off)) => f.bar_read(bar, Self::lane_swizzle(off, width), width),
            // Nobody claims it: the bus floats high.
            None => 0xffff_ffff,
        }
    }

    fn mem_write(&self, addr: u32, val: u32, width: usize) {
        if let Some((f, bar, off)) = self.window_target(addr) {
            f.bar_write(bar, Self::lane_swizzle(off, width), val, width);
        }
    }

    /// A CPU read of `width` bytes in a PCI memory window. PCI is 32 bits
    /// wide, so a doubleword is two cycles, the high word first.
    pub fn window_read(&self, addr: u32, width: usize) -> u64 {
        if width == 8 {
            let hi = self.mem_read(addr, 4) as u64;
            return (hi << 32) | self.mem_read(addr.wrapping_add(4), 4) as u64;
        }
        self.mem_read(addr, width) as u64
    }

    pub fn window_write(&self, addr: u32, val: u64, width: usize) {
        if width == 8 {
            self.mem_write(addr, (val >> 32) as u32, 4);
            self.mem_write(addr.wrapping_add(4), val as u32, 4);
            return;
        }
        self.mem_write(addr, val as u32, width);
    }

    /// Split a CONFIG_ADDR value into the function it selects and the
    /// register within it. Only bus 0 exists here.
    fn selected(&self) -> Option<(Arc<Slot>, usize)> {
        // Bit 31 is the enable bit of the standard PCI mechanism, and MACE's
        // bridge does not insist on it: the PROM sets it (0x80000800) and
        // NetBSD's driver does not (0x00000800). Requiring it makes the whole
        // bus invisible to one of them, which is exactly how the kernel came
        // to probe and find nothing.
        let a = *self.config_addr.lock().unwrap() & 0x7fff_ffff;
        if (a >> 16) & 0xff != 0 {
            return None;
        }
        let dev = ((a >> 11) & 0x1f) as u8;
        let func = (a >> 8) & 7;
        if func != 0 {
            return None;
        }
        let s = self.slots.lock().unwrap().get(&dev).cloned()?;
        Some((s, ((a >> 2) & 0x3f) as usize))
    }

    /// Config addresses selected so far, in first-touch order.
    pub fn probed(&self) -> Vec<u32> {
        self.probed.lock().unwrap().clone()
    }
}

impl Slot {
    fn note(&self, reg: usize, write: bool) {
        let mut t = self.touched.lock().unwrap();
        let r = (reg * 4) as u8;
        if !t.contains(&(r, write)) {
            t.push((r, write));
        }
    }
}

/// The bridge's own registers, configuration mechanism #1 among them.
impl BusDevice for MacePci {
    fn read32(&self, addr: u64) -> BusRead32 {
        let off = (addr as u32) & (MACEPCI_SIZE - 1) & !3;
        match off {
            macepci_reg::CONFIG_DATA => BusRead32::ok(match self.selected() {
                Some((s, reg)) => {
                    s.note(reg, false);
                    s.dev.config().read(reg)
                }
                // An absent device: the bus floats high. Zero would read back
                // as vendor ID 0x0000, which firmware takes for a device that
                // is present but broken.
                None => 0xffff_ffff,
            }),
            macepci_reg::CONFIG_ADDR => BusRead32::ok(*self.config_addr.lock().unwrap()),
            macepci_reg::REVISION => BusRead32::ok(MACEPCI_REVISION_VALUE),
            _ => BusRead32::ok(self.regs.lock().unwrap()[(off / 4) as usize]),
        }
    }

    fn write32(&self, addr: u64, val: u32) -> u32 {
        let off = (addr as u32) & (MACEPCI_SIZE - 1) & !3;
        match off {
            macepci_reg::CONFIG_ADDR => {
                *self.config_addr.lock().unwrap() = val;
                let mut p = self.probed.lock().unwrap();
                if val != 0 && !p.contains(&val) {
                    p.push(val);
                }
            }
            macepci_reg::CONFIG_DATA => {
                if let Some((s, reg)) = self.selected() {
                    s.note(reg, true);
                    s.dev.config().write(reg, val);
                }
            }
            _ => self.regs.lock().unwrap()[(off / 4) as usize] = val,
        }
        BUS_OK
    }

    fn read64(&self, addr: u64) -> BusRead64 {
        let hi = self.read32(addr).data as u64;
        let lo = self.read32(addr.wrapping_add(4)).data as u64;
        BusRead64::ok((hi << 32) | lo)
    }

    fn write64(&self, addr: u64, val: u64) -> u32 {
        self.write32(addr, (val >> 32) as u32);
        self.write32(addr.wrapping_add(4), val as u32)
    }
}

/// The bridge's registers, the selected configuration address, the pins'
/// levels (driven into CRIME again on load) and each slot's configuration
/// header, by device number. The functions behind the slots save their own
/// state.
impl Saveable for MacePci {
    fn save_state(&self) -> toml::Value {
        let mut t = toml::map::Map::new();
        t.insert("config_addr".into(), hex_u32(*self.config_addr.lock().unwrap()));
        t.insert("regs".into(), u32_slice_to_toml(&*self.regs.lock().unwrap()));
        let levels = self.irq.levels.lock().unwrap();
        t.insert("pins".into(), toml::Value::Array(levels.iter().map(|&b| toml::Value::Boolean(b)).collect()));
        drop(levels);
        let mut slots = toml::map::Map::new();
        for (n, s) in self.slots.lock().unwrap().iter() {
            slots.insert(n.to_string(), s.dev.config().save_state());
        }
        t.insert("slots".into(), toml::Value::Table(slots));
        toml::Value::Table(t)
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        *self.config_addr.lock().unwrap() = get_field(v, "config_addr").and_then(toml_u32).unwrap_or(0);
        if let Some(r) = get_field(v, "regs") {
            load_u32_slice(r, &mut *self.regs.lock().unwrap());
        }
        if let Some(toml::Value::Table(slots)) = get_field(v, "slots") {
            let ours = self.slots.lock().unwrap();
            for (n, cfg) in slots {
                let s = n.parse::<u8>().ok().and_then(|n| ours.get(&n))
                    .ok_or_else(|| format!("pci: saved state has slot {n}, which is empty here"))?;
                s.dev.config().load_state(cfg)?;
            }
        }
        if let Some(toml::Value::Array(pins)) = get_field(v, "pins") {
            for (line, level) in pins.iter().enumerate().take(PCI_PINS) {
                self.irq.set_line(line as u32, level.as_bool().unwrap_or(false));
            }
        }
        Ok(())
    }
}

impl crate::traits::Device for MacePci {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) {}
    fn start(&self) {}
    fn is_running(&self) -> bool { true }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![("pci".to_string(), "MACE PCI bridge: pci status | pci config <slot>".to_string())]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut w: Box<dyn std::io::Write + Send>) -> Result<(), String> {
        use std::io::Write;
        if cmd != "pci" {
            return Err("Command not found".to_string());
        }
        match args.first().copied().unwrap_or("status") {
            "status" => {
                let r = self.regs.lock().unwrap();
                let reg = |off: u32| r[(off / 4) as usize];
                writeln!(w, "MACE PCI: control {:#010x} revision {:#x} error flags {:#x} at {:#010x}, config address {:#010x}",
                    reg(macepci_reg::CONTROL), reg(macepci_reg::REVISION), reg(macepci_reg::ERROR_FLAGS),
                    reg(macepci_reg::ERROR_ADDR), *self.config_addr.lock().unwrap()).ok();
                for (slot, s) in self.slots.lock().unwrap().iter() {
                    let c = s.dev.config();
                    let id = c.read(0);
                    let bars: Vec<String> = (0..6).map(|i| c.bar_base(i)).filter(|&b| b != 0).map(|b| format!("{b:#x}")).collect();
                    let irq = match c.interrupt_pin() {
                        0 => "no interrupt".to_string(),
                        p => match (self.irq.route)(*slot, p) {
                            Some(l) => format!("INT{} to CRIME {l}", (b'A' + p - 1) as char),
                            None => format!("INT{} unrouted", (b'A' + p - 1) as char),
                        },
                    };
                    writeln!(w, "  slot {slot}: {} {:04x}:{:04x}, command {:#06x}, BARs [{}], {irq}",
                        s.name, id & 0xffff, id >> 16, c.command(), bars.join(", ")).ok();
                }
            }
            "config" => {
                let slot: u8 = args.get(1).and_then(|v| v.parse().ok()).ok_or("Usage: pci config <slot>")?;
                let (_, f, _) = self.slot(slot).ok_or(format!("pci: nothing in slot {slot}"))?;
                for row in 0..16 {
                    let v: Vec<String> = (0..4).map(|k| format!("{:08x}", f.config().read(row * 4 + k))).collect();
                    writeln!(w, "  {:02x}: {}", row * 16, v.join(" ")).ok();
                }
            }
            _ => return Err("Usage: pci status | config <slot>".to_string()),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pci::{Bar, PciConfig, PCI_COMMAND_MEMORY, PCI_INTA};

    /// A function that remembers its last BAR access and drives its pin.
    struct Fake {
        cfg: PciConfig,
        last: Mutex<Option<(usize, u32, usize)>>,
        irq: std::sync::OnceLock<IrqLine>,
    }

    impl PciBusDevice for Fake {
        fn config(&self) -> &PciConfig { &self.cfg }
        fn bar_read(&self, bar: usize, off: u32, width: usize) -> u32 {
            *self.last.lock().unwrap() = Some((bar, off, width));
            0x1234_5678
        }
        fn bar_write(&self, bar: usize, off: u32, _val: u32, width: usize) {
            *self.last.lock().unwrap() = Some((bar, off, width));
        }
        fn connect_interrupt(&self, line: IrqLine) { let _ = self.irq.set(line); }
    }

    struct Crime(Mutex<Vec<(u32, bool)>>);
    impl InterruptController for Crime {
        fn set_line(&self, line: u32, level: bool) { self.0.lock().unwrap().push((line, level)); }
    }

    fn cfg(addr: u32, reg: u32) -> u32 { addr | (1 << 31) | (reg << 2) }
    /// A bridge register, as the bus address it sits at.
    fn bridge(reg: u32) -> u64 { (MACEPCI_BASE + reg) as u64 }

    fn test_ram() -> Arc<O2Memory> {
        Arc::new(O2Memory::single(1 << 20))
    }

    /// A bus master reaches the same memory through both views, bytes in
    /// address order, and nothing above 2 GB.
    #[test]
    fn bus_masters_reach_memory_through_both_views() {
        let ram = test_ram();
        ram.write(0x100, &[0x11, 0x22, 0x33, 0x44]);
        let host = HostMemory::new(ram.clone());
        let mut b = [0u8; 4];
        assert!(host.dma_read((PCI_SWAPPED_VIEW + 0x100) as u64, &mut b).is_ok());
        assert_eq!(b, [0x11, 0x22, 0x33, 0x44]);
        assert!(host.dma_read((PCI_NATIVE_VIEW + 0x100) as u64, &mut b).is_ok());
        assert_eq!(b, [0x11, 0x22, 0x33, 0x44]);
        assert!(host.dma_write((PCI_NATIVE_VIEW + 0x102) as u64, &[0xaa, 0xbb]).is_ok());
        assert_eq!(ram.bytes(0x100, 4), [0x11, 0x22, 0xaa, 0xbb]);
        assert_eq!(host.dma_read((PCI_HOST_END + 0x100) as u64, &mut b), PciDmaStatus::MasterAbort,
            "not host memory");
        assert_eq!(b, [0; 4]);
        // A dual-address-cycle address is not folded into the low 4 GB.
        assert_eq!(host.dma_read(0x1_0000_0100, &mut b), PciDmaStatus::MasterAbort);
        assert_eq!(b, [0; 4]);
    }

    #[test]
    fn the_bridge_configures_decodes_and_routes_a_function() {
        let crime = Arc::new(Crime(Mutex::new(Vec::new())));
        let pci = MacePci::new(crime.clone(), test_ram());
        let f = Arc::new(Fake {
            cfg: PciConfig::new(0x9004, 0x8078, 0x0100_00, 1,
                [Bar::Io(0x100), Bar::Mem(0x1000), Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused], PCI_INTA),
            last: Mutex::new(None),
            irq: std::sync::OnceLock::new(),
        });
        pci.attach(IP32_SCSI_SLOT, "fake", f.clone());

        // Config mechanism #1: slot 1 answers, slot 3 floats high.
        let slot1 = (IP32_SCSI_SLOT as u32) << 11;
        pci.write32(bridge(macepci_reg::CONFIG_ADDR), cfg(slot1, 0));
        assert_eq!(pci.read32(bridge(macepci_reg::CONFIG_DATA)).data, 0x8078_9004);
        pci.write32(bridge(macepci_reg::CONFIG_ADDR), cfg(3 << 11, 0));
        assert_eq!(pci.read32(bridge(macepci_reg::CONFIG_DATA)).data, 0xffff_ffff);

        // Assign BAR1 and enable memory: the low window reaches it, with
        // byte lanes swapped.
        pci.write32(bridge(macepci_reg::CONFIG_ADDR), cfg(slot1, 5));
        pci.write32(bridge(macepci_reg::CONFIG_DATA), 0x8000_1000);
        assert_eq!(pci.window_read(PCI_MEM_BASE + 0x1084, 4), 0xffff_ffff, "not before memory is enabled");
        pci.write32(bridge(macepci_reg::CONFIG_ADDR), cfg(slot1, 1));
        pci.write32(bridge(macepci_reg::CONFIG_DATA), PCI_COMMAND_MEMORY);
        assert_eq!(pci.window_read(PCI_MEM_BASE + 0x1084, 4), 0x1234_5678);
        assert_eq!(*f.last.lock().unwrap(), Some((1, 0x84, 4)));
        pci.window_write(PCI_MEM_BASE + 0x1084, 0, 1);
        assert_eq!(*f.last.lock().unwrap(), Some((1, 0x87, 1)), "HCNTRL, by its little-endian lane");
        assert_eq!(pci.window_read(0x8000_1084, 2) as u16, 0x5678, "and the one-to-one high window");
        assert_eq!(*f.last.lock().unwrap(), Some((1, 0x86, 2)));
        assert_eq!(pci.slot(IP32_SCSI_SLOT).unwrap().2, vec![(0x00, false), (0x14, true), (0x04, true)],
                   "config registers touched, for the bring-up report");

        // Pin A of slot 1 is CRIME input 8.
        f.irq.get().expect("wired at attach").set(true);
        assert_eq!(*crime.0.lock().unwrap(), vec![(IP32_SCSI_CRIME_INT, true)]);
    }
    #[test]
    fn save_load_round_trip() {
        let make = |crime: Arc<Crime>| {
            let pci = MacePci::new(crime, test_ram());
            let f = Arc::new(Fake {
                cfg: PciConfig::new(0x9004, 0x8078, 0x0100_00, 1,
                    [Bar::Io(0x100), Bar::Mem(0x1000), Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused], PCI_INTA),
                last: Mutex::new(None),
                irq: std::sync::OnceLock::new(),
            });
            pci.attach(IP32_SCSI_SLOT, "fake", f.clone());
            (pci, f)
        };
        let (src, f) = make(Arc::new(Crime(Mutex::new(Vec::new()))));
        let slot1 = (IP32_SCSI_SLOT as u32) << 11;
        src.write32(bridge(macepci_reg::CONFIG_ADDR), cfg(slot1, 5));
        src.write32(bridge(macepci_reg::CONFIG_DATA), 0x8000_1000);
        src.write32(bridge(macepci_reg::CONFIG_ADDR), cfg(slot1, 1));
        src.write32(bridge(macepci_reg::CONFIG_DATA), PCI_COMMAND_MEMORY);
        f.irq.get().unwrap().set(true);
        let v1 = src.save_state();

        let crime = Arc::new(Crime(Mutex::new(Vec::new())));
        let (dst, _) = make(crime.clone());
        dst.load_state(&v1).expect("load_state");
        assert_eq!(v1, dst.save_state(), "MacePci save_state mismatch after load_state round-trip");
        assert_eq!(dst.window_read(PCI_MEM_BASE + 0x1084, 4), 0x1234_5678, "BAR and memory enable restored");
        assert!(crime.0.lock().unwrap().contains(&(IP32_SCSI_CRIME_INT, true)), "the pin is driven again");
    }

    /// Two functions whose pins share one system input (slot 3's INTA and
    /// slot 4's INTB, on a made-up routing): the input stays up until both
    /// have dropped, so neither can clear the other's interrupt.
    #[test]
    fn a_shared_input_stays_up_until_every_pin_drops() {
        fn route(slot: u8, pin: u8) -> Option<u32> {
            match (slot, pin) {
                (3, 1) | (4, 2) => Some(12),
                _ => None,
            }
        }
        let crime = Arc::new(Crime(Mutex::new(Vec::new())));
        let pci = MacePci::with_routing(crime.clone(), route, test_ram());
        let fake = |pin| Arc::new(Fake {
            cfg: PciConfig::new(0x1234, 0x5678, 0, 0, [Bar::Unused; 6], pin),
            last: Mutex::new(None),
            irq: std::sync::OnceLock::new(),
        });
        let (a, b) = (fake(PCI_INTA), fake(PCI_INTA + 1));
        pci.attach(3, "a", a.clone());
        pci.attach(4, "b", b.clone());
        let (la, lb) = (a.irq.get().unwrap(), b.irq.get().unwrap());
        let last = || *crime.0.lock().unwrap().last().unwrap();

        la.set(true);
        assert_eq!(last(), (12, true));
        lb.set(true);
        la.set(false);
        assert_eq!(last(), (12, true), "slot 4's INTB still holds it");
        lb.set(false);
        assert_eq!(last(), (12, false));
    }
}
