//! MACE's PCI host bridge on the O2: the root complex. It generates
//! configuration cycles, gives the CPU its windows onto PCI memory (swapping
//! byte lanes on the way, since PCI is little-endian and the CPU is not), and
//! wires each slot's interrupt pin to its CRIME input.

use std::sync::{Arc, Mutex};

use crate::irq::InterruptController;
use crate::dev::crime_mem::O2Memory;
use crate::pci::bus::PciBus;
use crate::pci::host::{CpuWindow, PciHost};
use crate::pci::slots::{BusDesc, PciAddr};
use crate::pci::{PciBusDevice, PciDma, PciDmaStatus, PciLanes, Space};
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
    /// with where each page starts on the host bus, or None where the PCI
    /// address is not memory. A page never straddles a bank slot: one decode
    /// per page, not per byte.
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
            Some(i) => {
                crate::physical::read_bytes(&**self.ram.phys(), i as u64, &mut buf[r]);
            }
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
            Some(i) => {
                crate::physical::write_bytes(&**self.ram.phys(), i as u64, &data[r]);
            }
            None => st = PciDmaStatus::MasterAbort,
        });
        st
    }
}

/// What MACE gives the shared PCI bus: host memory for the masters, CRIME's
/// inputs for the interrupt outputs, and the CPU's three windows.
pub struct MaceHost {
    /// Where the outputs are: CRIME.
    crime: Arc<dyn InterruptController>,
    dma: Arc<HostMemory>,
    lanes: Arc<Lanes>,
}

impl PciHost for MaceHost {
    /// MACE's translation is fixed, so every function gets one object.
    fn dma_for(&self, _device: u8) -> Arc<dyn PciDma> {
        self.dma.clone()
    }

    fn set_output(&self, output: u8, level: bool) {
        self.crime.set_line(output as u32, level);
    }

    fn windows(&self) -> Vec<CpuWindow> {
        let window = |space, cpu: u32, pci, len: u64| CpuWindow {
            space,
            cpu: cpu as u64,
            pci,
            len,
            mask: 0xffff_ffff,
            lanes: self.lanes.clone(),
            lanes_key: 0,
        };
        vec![
            window(Space::Io, PCI_IO_BASE, 0, PCI_IO_SIZE as u64),
            window(Space::Mem, PCI_MEM_BASE, PCI_MEM_PCI_BASE, PCI_MEM_SIZE as u64),
            window(Space::Mem, PCI_HI_MEM_BASE, PCI_HI_MEM_BASE, (1u64 << 32) - PCI_HI_MEM_BASE as u64),
        ]
    }
}

pub struct MacePci {
    config_addr: Mutex<u32>,
    regs: Mutex<[u32; (MACEPCI_SIZE / 4) as usize]>,
    /// The functions on bus 0, their interrupt routing and header state.
    bus: PciBus,
}

impl MacePci {
    /// A bridge for the bus `desc` describes, with its interrupt outputs
    /// wired to `intc`'s inputs and `ram` for its masters.
    pub fn new(desc: &'static BusDesc, intc: Arc<dyn InterruptController>, ram: Arc<O2Memory>) -> Self {
        let host = Arc::new(MaceHost { crime: intc, dma: Arc::new(HostMemory::new(ram)), lanes: Arc::new(Lanes) });
        Self {
            config_addr: Mutex::new(0),
            regs: Mutex::new([0u32; (MACEPCI_SIZE / 4) as usize]),
            bus: PciBus::new(desc, host),
        }
    }

    /// The bus: place functions on it, ask it what is there.
    pub fn bus(&self) -> &PciBus {
        &self.bus
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
    fn lane_swizzle(off: u32, width: usize) -> u32 {
        match width {
            1 => off ^ 3,
            2 => off ^ 2,
            _ => off,
        }
    }

    /// The function and BAR a CPU address in a window reaches, in whichever
    /// space that window opens onto.
    fn window_target(&self, addr: u32) -> Option<(&Arc<dyn PciBusDevice>, usize, u32)> {
        if Self::in_io_window(addr) {
            return self.bus.target(Space::Io, addr - PCI_IO_BASE);
        }
        self.bus.target(Space::Mem, Self::pci_addr_of(addr))
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
    fn selected(&self) -> Option<(PciAddr, usize)> {
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
        Some((PciAddr::dev(dev), ((a >> 2) & 0x3f) as usize))
    }
}

/// MACE's lanes, for [`PciBar`] entries: the same swizzle and the same
/// split of a doubleword into two cycles, high word first, that
/// `window_read` and `window_write` make once they have found the target.
struct Lanes;

impl PciLanes for Lanes {
    fn read(&self, f: &dyn PciBusDevice, bar: usize, off: u32, size: usize) -> u64 {
        if size == 8 {
            let hi = f.bar_read(bar, off, 4) as u64;
            return hi << 32 | f.bar_read(bar, off.wrapping_add(4), 4) as u64;
        }
        f.bar_read(bar, MacePci::lane_swizzle(off, size), size) as u64
    }

    fn write(&self, f: &dyn PciBusDevice, bar: usize, off: u32, size: usize, val: u64) {
        if size == 8 {
            f.bar_write(bar, off, (val >> 32) as u32, 4);
            f.bar_write(bar, off.wrapping_add(4), val as u32, 4);
            return;
        }
        f.bar_write(bar, MacePci::lane_swizzle(off, size), val as u32, size);
    }
}

/// The bridge's own registers, configuration mechanism #1 among them.
impl BusDevice for MacePci {
    fn read32(&self, addr: u64) -> BusRead32 {
        let off = (addr as u32) & (MACEPCI_SIZE - 1) & !3;
        match off {
            macepci_reg::CONFIG_DATA => BusRead32::ok(
                self.selected().and_then(|(at, reg)| self.bus.config_read(at, reg * 4, 4))
                    // An absent device: the bus floats high. Zero would read back
                    // as vendor ID 0x0000, which firmware takes for a device that
                    // is present but broken.
                    .unwrap_or(0xffff_ffff),
            ),
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
            }
            macepci_reg::CONFIG_DATA => {
                if let Some((at, reg)) = self.selected() {
                    self.bus.config_write(at, reg * 4, 4, val);
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

/// The bridge's registers, the selected configuration address, and the
/// bus: each function's configuration header and the interrupt pins'
/// levels (driven into CRIME again on load). The functions behind the
/// slots save their own state.
impl Saveable for MacePci {
    fn save_state(&self) -> toml::Value {
        let mut t = toml::map::Map::new();
        t.insert("config_addr".into(), hex_u32(*self.config_addr.lock().unwrap()));
        t.insert("regs".into(), u32_slice_to_toml(&*self.regs.lock().unwrap()));
        t.insert("pci".into(), self.bus.save_state());
        toml::Value::Table(t)
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        *self.config_addr.lock().unwrap() = get_field(v, "config_addr").and_then(toml_u32).unwrap_or(0);
        if let Some(r) = get_field(v, "regs") {
            load_u32_slice(r, &mut *self.regs.lock().unwrap());
        }
        self.bus.load_state(get_field(v, "pci").ok_or("pci: saved state has no bus")?)?;
        self.bus.redrive();
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
                for line in crate::pci::monitor::report(&self.bus) {
                    writeln!(w, "  {line}").ok();
                }
            }
            "config" => {
                let slot: u8 = args.get(1).and_then(|v| v.parse().ok()).ok_or("Usage: pci config <slot>")?;
                let rows = crate::pci::monitor::config_dump(&self.bus, PciAddr::dev(slot)).ok_or(format!("pci: nothing in slot {slot}"))?;
                for row in rows {
                    writeln!(w, "{row}").ok();
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
    use crate::irq::IrqLine;
    use crate::pci::{Bar, BarRun, PciConfig, PCI_COMMAND_IO, PCI_COMMAND_MEMORY, PCI_INTA};

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

    fn fake(pin: u8) -> Arc<Fake> {
        Arc::new(Fake {
            cfg: PciConfig::new(0x9004, 0x8078, 0x01_0000, 1,
                [Bar::Io(0x100), Bar::Mem(0x1000), Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused], pin),
            last: Mutex::new(None),
            irq: std::sync::OnceLock::new(),
        })
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

    /// A bridge on the O2's bus description, and the CRIME it drives.
    fn o2() -> (MacePci, Arc<Crime>) {
        let crime = Arc::new(Crime(Mutex::new(Vec::new())));
        (MacePci::new(&PCI_BUS, crime.clone(), test_ram()), crime)
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
        let (pci, crime) = o2();
        let f = fake(PCI_INTA);
        pci.bus().attach(PciAddr::dev(IP32_SCSI_SLOT), f.clone());

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

        // Pin A of slot 1 is CRIME input 8.
        f.irq.get().expect("wired at attach").set(true);
        assert_eq!(*crime.0.lock().unwrap(), vec![(IP32_SCSI_CRIME_INT, true)]);
    }

    /// The pins of the option slot (device 3) reach CRIME as the MACE
    /// specification routes them: INTA on 10, INTB, INTC and INTD on 13,
    /// 14 and 15.
    /// Only function 0 of a device answers here, and only on bus 0.
    #[test]
    fn a_function_number_other_than_zero_floats_high() {
        let (pci, _) = o2();
        pci.bus().attach(PciAddr::dev(IP32_SCSI_SLOT), fake(PCI_INTA));
        let slot1 = (IP32_SCSI_SLOT as u32) << 11;
        pci.write32(bridge(macepci_reg::CONFIG_ADDR), cfg(slot1, 0));
        assert_eq!(pci.read32(bridge(macepci_reg::CONFIG_DATA)).data, 0x8078_9004);
        pci.write32(bridge(macepci_reg::CONFIG_ADDR), cfg(slot1 | (1 << 8), 0));
        assert_eq!(pci.read32(bridge(macepci_reg::CONFIG_DATA)).data, 0xffff_ffff, "function 1");
        pci.write32(bridge(macepci_reg::CONFIG_ADDR), cfg(slot1 | (1 << 16), 0));
        assert_eq!(pci.read32(bridge(macepci_reg::CONFIG_DATA)).data, 0xffff_ffff, "bus 1");
    }

    #[test]
    fn device_three_pins_reach_the_spec_crime_inputs() {
        for (pin, input) in [(PCI_INTA, 10), (PCI_INTA + 1, 13), (PCI_INTA + 2, 14), (PCI_INTA + 3, 15)] {
            let (pci, crime) = o2();
            let f = fake(pin);
            pci.bus().attach(PciAddr::dev(3), f.clone());
            f.irq.get().expect("wired at attach").set(true);
            assert_eq!(*crime.0.lock().unwrap(), vec![(input, true)], "pin {pin}");
        }
    }

    #[test]
    fn device_four_inta_is_crime_eleven_and_its_other_pins_lead_nowhere() {
        let (pci, crime) = o2();
        let f = fake(PCI_INTA);
        pci.bus().attach(PciAddr::dev(4), f.clone());
        f.irq.get().unwrap().set(true);
        assert_eq!(*crime.0.lock().unwrap(), vec![(11, true)]);

        let (pci, crime) = o2();
        let g = fake(PCI_INTA + 1);
        pci.bus().attach(PciAddr::dev(4), g.clone());
        g.irq.get().expect("still wired").set(true);
        assert!(crime.0.lock().unwrap().is_empty(), "INTB of device 4 is not routed");
    }

    #[test]
    fn onboard_scsi_keeps_inputs_eight_and_nine() {
        let (pci, crime) = o2();
        let (a, b) = (fake(PCI_INTA), fake(PCI_INTA));
        pci.bus().attach(PciAddr::dev(IP32_SCSI_SLOT), a.clone());
        pci.bus().attach(PciAddr::dev(IP32_SCSI_SLOT_B), b.clone());
        a.irq.get().unwrap().set(true);
        b.irq.get().unwrap().set(true);
        assert_eq!(*crime.0.lock().unwrap(), vec![(IP32_SCSI_CRIME_INT, true), (IP32_SCSI_CRIME_INT_B, true)]);
    }

    /// Once a memory BAR is placed and enabled, the pages it decodes in both
    /// CPU windows are published as runs, and an access through a run is
    /// the window's access, lanes included.
    #[test]
    fn a_placed_bar_is_published_as_runs() {
        let (pci, _) = o2();
        let f = fake(PCI_INTA);
        pci.bus().attach(PciAddr::dev(IP32_SCSI_SLOT), f.clone());
        let runs = Arc::new(Mutex::new(Vec::<BarRun>::new()));
        let sink = runs.clone();
        pci.bus().on_bar_runs(Box::new(move |r| *sink.lock().unwrap() = r));
        assert!(runs.lock().unwrap().is_empty());
        let slot1 = (IP32_SCSI_SLOT as u32) << 11;
        pci.write32(bridge(macepci_reg::CONFIG_ADDR), cfg(slot1, 5));
        pci.write32(bridge(macepci_reg::CONFIG_DATA), 0x8000_1000);
        assert!(runs.lock().unwrap().is_empty(), "not before memory is enabled");
        pci.write32(bridge(macepci_reg::CONFIG_ADDR), cfg(slot1, 1));
        pci.write32(bridge(macepci_reg::CONFIG_DATA), PCI_COMMAND_IO | PCI_COMMAND_MEMORY);
        let got = runs.lock().unwrap().clone();
        let spans: Vec<_> = got.iter().map(|r| (r.start, r.end)).collect();
        let low = (PCI_MEM_BASE + 0x1000) as u64;
        assert_eq!(spans, vec![(low, low + 0x1000), (0x8000_1000, 0x8000_2000)], "the I/O BAR is under a page");
        assert_eq!(got[0].dev.read32(low + 0x84).data, 0x1234_5678);
        assert_eq!(*f.last.lock().unwrap(), Some((1, 0x84, 4)));
        assert_eq!(got[0].dev.write8(low + 0x84, 0), BUS_OK);
        assert_eq!(*f.last.lock().unwrap(), Some((1, 0x87, 1)), "HCNTRL, by its little-endian lane");
        assert_eq!(got[1].dev.read16(0x8000_1084).data, 0x5678);
        assert_eq!(*f.last.lock().unwrap(), Some((1, 0x86, 2)));
    }

    #[test]
    fn save_load_round_trip() {
        let make = || {
            let (pci, crime) = o2();
            let f = fake(PCI_INTA);
            pci.bus().attach(PciAddr::dev(IP32_SCSI_SLOT), f.clone());
            (pci, f, crime)
        };
        let (src, f, _) = make();
        let slot1 = (IP32_SCSI_SLOT as u32) << 11;
        src.write32(bridge(macepci_reg::CONFIG_ADDR), cfg(slot1, 5));
        src.write32(bridge(macepci_reg::CONFIG_DATA), 0x8000_1000);
        src.write32(bridge(macepci_reg::CONFIG_ADDR), cfg(slot1, 1));
        src.write32(bridge(macepci_reg::CONFIG_DATA), PCI_COMMAND_MEMORY);
        f.irq.get().unwrap().set(true);
        let v1 = src.save_state();

        let (dst, _, crime) = make();
        dst.load_state(&v1).expect("load_state");
        assert_eq!(v1, dst.save_state(), "MacePci save_state mismatch after load_state round-trip");
        assert_eq!(dst.window_read(PCI_MEM_BASE + 0x1084, 4), 0x1234_5678, "BAR and memory enable restored");
        assert!(crime.0.lock().unwrap().contains(&(IP32_SCSI_CRIME_INT, true)), "the pin is driven again");
    }

    #[test]
    fn a_saved_state_with_a_device_this_bridge_lacks_is_refused() {
        let (src, ..) = o2();
        src.bus().attach(PciAddr::dev(3), fake(PCI_INTA));
        let v = src.save_state();
        let (dst, _) = o2();
        let e = dst.load_state(&v).unwrap_err();
        assert!(e.contains("device 3, which is empty here"), "{e}");
    }
}
