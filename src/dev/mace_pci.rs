//! MACE's PCI host bridge on the O2: the root complex. It generates
//! configuration cycles, gives the CPU its windows onto PCI memory (swapping
//! byte order on the way, since PCI is little-endian and the CPU is not), and
//! passes each slot's interrupt pin to its CRIME input. The functions, their
//! BAR windows and the interrupt combining are the [`PciBus`] it owns.

use std::sync::{Arc, Mutex, Weak};

use crate::irq::InterruptController;
use crate::dev::crime_mem::O2Memory;
use crate::pci::{CpuWindow, DevFn, IntxRouting, PciBus, PciDma, PciDmaStatus, PciHost, Space};
use crate::pci::ByteOrder;
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

/// The "PCI option" slot: device 3, whose four pins the MACE specification
/// routes to CRIME inputs 10 and 13 to 15.
pub const IP32_OPTION_SLOT: u8 = 3;

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

pub struct MacePci {
    /// The functions on bus 0, their windows and interrupt pins.
    bus: PciBus,
    /// Where the bus's outputs go.
    crime: Arc<dyn InterruptController>,
    /// Host memory, for the functions that master the bus. One object:
    /// MACE's translation is fixed.
    dma: Arc<HostMemory>,
    config_addr: Mutex<u32>,
    regs: Mutex<[u32; (MACEPCI_SIZE / 4) as usize]>,
}

impl PciHost for MacePci {
    fn dma_for(&self, _requester: u8) -> Arc<dyn PciDma> {
        self.dma.clone()
    }

    fn set_irq(&self, output: u8, level: bool) {
        self.crime.set_line(output as u32, level);
    }
}

impl MacePci {
    /// A bridge whose bus is wired as `wiring` (its outputs are `crime`'s
    /// inputs).
    pub fn new(wiring: &'static IntxRouting, crime: Arc<dyn InterruptController>, ram: Arc<O2Memory>) -> Arc<Self> {
        let this = Arc::new_cyclic(|me: &Weak<Self>| Self {
            bus: PciBus::new(0, wiring, me.clone()),
            crime,
            dma: Arc::new(HostMemory::new(ram)),
            config_addr: Mutex::new(0),
            regs: Mutex::new([0u32; (MACEPCI_SIZE / 4) as usize]),
        });
        // The windows are fixed: MACE has no register that moves them.
        this.bus.set_host_windows(&Self::windows);
        this
    }

    pub fn bus(&self) -> &PciBus {
        &self.bus
    }

    /// The CPU's three windows onto PCI: I/O, memory low and memory high.
    fn windows() -> Vec<CpuWindow> {
        let w = |space, cpu: u32, pci: u32, len: u64| CpuWindow { space, cpu: cpu as u64, pci, len, mask: 0xffff_ffff };
        vec![
            w(Space::Io, PCI_IO_BASE, 0, PCI_IO_SIZE as u64),
            w(Space::Mem, PCI_MEM_BASE, PCI_MEM_PCI_BASE, PCI_MEM_SIZE as u64),
            w(Space::Mem, PCI_HI_MEM_BASE, PCI_HI_MEM_BASE, (1u64 << 32) - PCI_HI_MEM_BASE as u64),
        ]
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

    /// The bridge swaps byte order: an 8-bit access at CPU offset `A` reaches
    /// device byte `A ^ 3`, and a 16-bit one `A ^ 2`. Without it the
    /// AIC-7880's initialisation reads as writes to registers that are
    /// nothing in particular; with it they are HCNTRL (chip reset, then
    /// pause), CLRINT, SCBPTR walking 0,1,2,... and the last byte of each
    /// SCB being cleared, which is how that chip is brought up.
    fn order() -> ByteOrder {
        ByteOrder::values(4)
    }

    /// A CPU read of `width` bytes in a PCI window. A doubleword is two
    /// word cycles, the high word first. Nobody claims it: the bus floats
    /// high.
    pub fn window_read(&self, addr: u32, width: usize) -> u64 {
        let target = if Self::in_io_window(addr) {
            self.bus.decode(Space::Io, addr - PCI_IO_BASE)
        } else {
            self.bus.decode(Space::Mem, Self::pci_addr_of(addr))
        };
        match target {
            Some((f, bar, off)) => Self::order().read(f, bar, off, width),
            None if width == 8 => u64::MAX,
            None => 0xffff_ffff,
        }
    }

    pub fn window_write(&self, addr: u32, val: u64, width: usize) {
        let target = if Self::in_io_window(addr) {
            self.bus.decode(Space::Io, addr - PCI_IO_BASE)
        } else {
            self.bus.decode(Space::Mem, Self::pci_addr_of(addr))
        };
        if let Some((f, bar, off)) = target {
            Self::order().write(f, bar, off, width, val);
        }
    }

    /// The bus, function and byte offset CONFIG_ADDR selects. Bit 31 is the
    /// enable bit of the standard PCI mechanism, and MACE's bridge does not
    /// insist on it: the PROM sets it (0x80000800) and NetBSD's driver does
    /// not (0x00000800). Requiring it makes the whole bus invisible to one
    /// of them. Bus 0 is MACE's own; another number is a bus behind a
    /// PCI-to-PCI bridge, if firmware has numbered one that far.
    fn selected(&self) -> Option<(PciBus, DevFn, usize)> {
        let a = *self.config_addr.lock().unwrap() & 0x7fff_ffff;
        let bus = self.bus.find_bus(((a >> 16) & 0xff) as u8)?;
        Some((bus, DevFn { device: ((a >> 11) & 0x1f) as u8, function: ((a >> 8) & 7) as u8 }, (a & 0xfc) as usize))
    }
}

/// The bridge's own registers, configuration mechanism #1 among them.
impl BusDevice for MacePci {
    fn read32(&self, addr: u64) -> BusRead32 {
        let off = (addr as u32) & (MACEPCI_SIZE - 1) & !3;
        match off {
            // An absent device: the bus floats high. Zero would read back
            // as vendor ID 0x0000, which firmware takes for a device that
            // is present but broken.
            macepci_reg::CONFIG_DATA => BusRead32::ok(
                self.selected().and_then(|(bus, at, off)| bus.config_read(at, off, 4)).unwrap_or(0xffff_ffff),
            ),
            macepci_reg::CONFIG_ADDR => BusRead32::ok(*self.config_addr.lock().unwrap()),
            macepci_reg::REVISION => BusRead32::ok(MACEPCI_REVISION_VALUE),
            _ => BusRead32::ok(self.regs.lock().unwrap()[(off / 4) as usize]),
        }
    }

    fn write32(&self, addr: u64, val: u32) -> u32 {
        let off = (addr as u32) & (MACEPCI_SIZE - 1) & !3;
        match off {
            macepci_reg::CONFIG_ADDR => *self.config_addr.lock().unwrap() = val,
            macepci_reg::CONFIG_DATA => {
                if let Some((bus, at, off)) = self.selected() {
                    bus.config_write(at, off, 4, val);
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

/// The bridge's registers, the selected configuration address and the bus
/// (each function's header, the pin and wire levels). The functions behind
/// the headers save their own state; the machine drives the pins into CRIME
/// again with [`PciBus::redrive`] once CRIME has loaded.
impl Saveable for MacePci {
    fn save_state(&self) -> toml::Value {
        let mut t = toml::map::Map::new();
        t.insert("config_addr".into(), hex_u32(*self.config_addr.lock().unwrap()));
        t.insert("regs".into(), u32_slice_to_toml(&*self.regs.lock().unwrap()));
        t.insert("bus".into(), self.bus.save_state());
        toml::Value::Table(t)
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        *self.config_addr.lock().unwrap() = get_field(v, "config_addr").and_then(toml_u32).unwrap_or(0);
        // A field the snapshot lacks is the power-on value, not what the
        // machine held before the load.
        let mut regs = self.regs.lock().unwrap();
        *regs = [0; (MACEPCI_SIZE / 4) as usize];
        if let Some(r) = get_field(v, "regs") {
            load_u32_slice(r, &mut *regs);
        }
        drop(regs);
        self.bus.load_state(get_field(v, "bus").ok_or("pci: saved MACE state has no bus")?)
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
                self.bus.describe(&mut w);
            }
            _ => return self.bus.command(args, &mut w),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pci::BarMapping;
    use crate::pci::testing::TestCard;
    use crate::pci::{Bar, PciFunction, PCI_COMMAND_IO, PCI_COMMAND_MEMORY, PCI_INTA};

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

    fn bars() -> [Bar; 6] {
        [Bar::Io(0x100), Bar::Mem(0x1000), Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused]
    }

    fn bridge_with(crime: &Arc<Crime>) -> Arc<MacePci> {
        MacePci::new(&PCI0, crime.clone(), test_ram())
    }

    fn at(device: u8) -> DevFn {
        DevFn { device, function: 0 }
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
        let pci = bridge_with(&crime);
        let f = TestCard::new(bars(), PCI_INTA);
        pci.bus().attach(at(IP32_SCSI_SLOT), f.clone());

        // Config mechanism #1: slot 1 answers, slot 3 floats high.
        let slot1 = (IP32_SCSI_SLOT as u32) << 11;
        pci.write32(bridge(macepci_reg::CONFIG_ADDR), cfg(slot1, 0));
        assert_eq!(pci.read32(bridge(macepci_reg::CONFIG_DATA)).data, 0x5678_1234);
        pci.write32(bridge(macepci_reg::CONFIG_ADDR), cfg(3 << 11, 0));
        assert_eq!(pci.read32(bridge(macepci_reg::CONFIG_DATA)).data, 0xffff_ffff);

        // Assign BAR1 and enable memory: the low window reaches it, with
        // byte order swapped.
        pci.write32(bridge(macepci_reg::CONFIG_ADDR), cfg(slot1, 5));
        pci.write32(bridge(macepci_reg::CONFIG_DATA), 0x8000_1000);
        assert_eq!(pci.window_read(PCI_MEM_BASE + 0x1084, 4), 0xffff_ffff, "not before memory is enabled");
        pci.write32(bridge(macepci_reg::CONFIG_ADDR), cfg(slot1, 1));
        pci.write32(bridge(macepci_reg::CONFIG_DATA), PCI_COMMAND_MEMORY);
        pci.window_read(PCI_MEM_BASE + 0x1084, 4);
        assert_eq!(f.calls(), vec![(0x84, 4)]);
        pci.window_write(PCI_MEM_BASE + 0x1084, 0, 1);
        assert_eq!(f.calls()[1], (0x87, 1), "HCNTRL, by its little-endian lane");
        pci.window_read(0x8000_1084, 2);
        assert_eq!(f.calls()[2], (0x86, 2), "and the one-to-one high window");

        // Pin A of slot 1 is CRIME input 8.
        f.state().irq().expect("wired at attach").set(true);
        assert_eq!(*crime.0.lock().unwrap(), vec![(8, true)]);
    }

    /// Once a memory BAR is placed and enabled, the pages it decodes in both
    /// CPU windows are published as runs, and an access through a run is
    /// the window's access, order included.
    #[test]
    fn a_placed_bar_is_published_as_runs() {
        use crate::pci::BarDevice;
        let crime = Arc::new(Crime(Mutex::new(Vec::new())));
        let pci = bridge_with(&crime);
        let f = TestCard::new(bars(), PCI_INTA);
        pci.bus().attach(at(IP32_SCSI_SLOT), f.clone());
        let runs = Arc::new(Mutex::new(Vec::<BarMapping>::new()));
        let sink = runs.clone();
        pci.bus().on_bar_mappings(Box::new(move |r| *sink.lock().unwrap() = r));
        assert!(runs.lock().unwrap().is_empty());
        let slot1 = (IP32_SCSI_SLOT as u32) << 11;
        pci.write32(bridge(macepci_reg::CONFIG_ADDR), cfg(slot1, 5));
        pci.write32(bridge(macepci_reg::CONFIG_DATA), 0x8000_1000);
        assert!(runs.lock().unwrap().is_empty(), "not before memory is enabled");
        pci.write32(bridge(macepci_reg::CONFIG_ADDR), cfg(slot1, 1));
        pci.write32(bridge(macepci_reg::CONFIG_DATA), PCI_COMMAND_IO | PCI_COMMAND_MEMORY);
        let got = runs.lock().unwrap().clone();
        let spans: Vec<_> = got.iter().map(|r| (r.cpu, r.cpu + r.len)).collect();
        let low = (PCI_MEM_BASE + 0x1000) as u64;
        assert_eq!(spans, vec![(low, low + 0x1000), (0x8000_1000, 0x8000_2000)], "the I/O BAR is under a page");
        // The machine builds a BarDevice from each run.
        let bar = |r: &BarMapping| BarDevice::new(r.function.clone(), r.bar, r.cpu.wrapping_sub(r.off as u64), 0xffff_ffff, ByteOrder::values(4));
        assert_eq!(bar(&got[0]).read32(low + 0x84).data, 0);
        assert_eq!(f.calls(), vec![(0x84, 4)]);
        assert_eq!(bar(&got[0]).write8(low + 0x84, 0), BUS_OK);
        assert_eq!(f.calls()[1], (0x87, 1), "HCNTRL, by its little-endian lane");
        bar(&got[1]).read16(0x8000_1084);
        assert_eq!(f.calls()[2], (0x86, 2));
    }

    #[test]
    fn save_load_round_trip() {
        let make = |crime: &Arc<Crime>| {
            let pci = bridge_with(crime);
            let f = TestCard::new(bars(), PCI_INTA);
            pci.bus().attach(at(IP32_SCSI_SLOT), f.clone());
            (pci, f)
        };
        let (src, f) = make(&Arc::new(Crime(Mutex::new(Vec::new()))));
        let slot1 = (IP32_SCSI_SLOT as u32) << 11;
        src.write32(bridge(macepci_reg::CONFIG_ADDR), cfg(slot1, 5));
        src.write32(bridge(macepci_reg::CONFIG_DATA), 0x8000_1000);
        src.write32(bridge(macepci_reg::CONFIG_ADDR), cfg(slot1, 1));
        src.write32(bridge(macepci_reg::CONFIG_DATA), PCI_COMMAND_MEMORY);
        src.write32(bridge(macepci_reg::CONTROL), 0x1234);
        f.state().irq().unwrap().set(true);
        let v1 = src.save_state();

        let crime = Arc::new(Crime(Mutex::new(Vec::new())));
        let (dst, g) = make(&crime);
        dst.load_state(&v1).expect("load_state");
        assert_eq!(v1, dst.save_state(), "MacePci save_state mismatch after load_state round-trip");
        dst.window_read(PCI_MEM_BASE + 0x1084, 4);
        assert_eq!(g.calls(), vec![(0x84, 4)], "BAR and memory enable restored");
        assert!(crime.0.lock().unwrap().is_empty(), "load drives nothing; the machine redrives");
        dst.bus().redrive();
        assert!(crime.0.lock().unwrap().contains(&(8, true)), "the pin is driven again");
    }

    /// A snapshot lacking a field leaves the power-on value, not whatever
    /// the machine held before the load.
    #[test]
    fn mace_snapshot_defaults_an_absent_regs_field() {
        let crime = Arc::new(Crime(Mutex::new(Vec::new())));
        let pci = bridge_with(&crime);
        let mut v = pci.save_state();
        pci.write32(bridge(macepci_reg::CONTROL), 0xdead_beef);
        v.as_table_mut().unwrap().remove("regs");
        pci.load_state(&v).unwrap();
        assert_eq!(pci.read32(bridge(macepci_reg::CONTROL)).data, 0);
    }

    /// MACE spec Tables 85, 87 and 90: devices 1 and 2 INTA to CRIME 8 and 9,
    /// device 3 INTA to 10 and INTB-D to 13 to 15. Device 4 INTA to 11 is
    /// iris's spare for virtio-serial. Nothing else is wired.
    #[test]
    fn o2_routes_devices_1_2_3_4_to_crime() {
        let routes = [(1u8, 1u8, 8u32), (2, 1, 9), (3, 1, 10), (3, 2, 13), (3, 3, 14), (3, 4, 15), (4, 1, 11)];
        for (device, pin, input) in routes {
            let crime = Arc::new(Crime(Mutex::new(Vec::new())));
            let pci = bridge_with(&crime);
            let c = TestCard::new(bars(), pin);
            pci.bus().attach(at(device), c.clone());
            c.state().irq().unwrap().set(true);
            assert_eq!(*crime.0.lock().unwrap(), vec![(input, true)], "device {device} INT{}", (b'A' + pin - 1) as char);
        }
    }

    #[test]
    #[should_panic(expected = "does not route")]
    fn a_pin_on_device_5_panics_at_attach() {
        let crime = Arc::new(Crime(Mutex::new(Vec::new())));
        bridge_with(&crime).bus().attach(at(5), TestCard::new(bars(), PCI_INTA));
    }

    #[test]
    fn an_absent_function_floats_high_on_config_read() {
        let crime = Arc::new(Crime(Mutex::new(Vec::new())));
        let pci = bridge_with(&crime);
        pci.bus().attach(at(1), TestCard::new(bars(), PCI_INTA));
        for sel in [cfg(2 << 11, 0), cfg(1 << 11 | 1 << 8, 0), cfg(1 << 11 | 1 << 16, 0)] {
            pci.write32(bridge(macepci_reg::CONFIG_ADDR), sel);
            assert_eq!(pci.read32(bridge(macepci_reg::CONFIG_DATA)).data, 0xffff_ffff, "{sel:#x}");
        }
    }

    /// Bytes `^3`, halves `^2`, words as given, and a doubleword as two word
    /// cycles, high word first (MACE's 64-bit PIO is unknown, see ByteOrder).
    #[test]
    fn mace_window_read_uses_values_4() {
        let crime = Arc::new(Crime(Mutex::new(Vec::new())));
        let pci = bridge_with(&crime);
        let f = TestCard::new(bars(), PCI_INTA);
        pci.bus().attach(at(1), f.clone());
        pci.write32(bridge(macepci_reg::CONFIG_ADDR), cfg(1 << 11, 5));
        pci.write32(bridge(macepci_reg::CONFIG_DATA), 0x8000_1000);
        pci.write32(bridge(macepci_reg::CONFIG_ADDR), cfg(1 << 11, 1));
        pci.write32(bridge(macepci_reg::CONFIG_DATA), PCI_COMMAND_MEMORY);
        let base = PCI_MEM_BASE + 0x1000;
        pci.window_read(base + 4, 1);
        pci.window_read(base + 4, 2);
        pci.window_read(base + 4, 4);
        pci.window_read(base + 8, 8);
        assert_eq!(f.calls(), vec![(7, 1), (6, 2), (4, 4), (8, 4), (12, 4)]);
    }

    /// A configuration cycle naming bus 1 reaches the bus behind a
    /// PCI-to-PCI bridge once firmware has numbered it, and floats high
    /// before.
    #[test]
    fn a_host_config_cycle_for_bus_1_reaches_the_secondary() {
        use crate::pci::PciToPciBridge;
        let crime = Arc::new(Crime(Mutex::new(Vec::new())));
        let pci = bridge_with(&crime);
        let p2p = PciToPciBridge::new(pci.bus(), at(4), 0x1234, 0x5678);
        pci.bus().attach(at(4), p2p.clone());
        p2p.secondary().attach(at(2), TestCard::new(bars(), 0));
        let behind = |reg| cfg(1 << 16 | 2 << 11, reg);
        pci.write32(bridge(macepci_reg::CONFIG_ADDR), behind(0));
        assert_eq!(pci.read32(bridge(macepci_reg::CONFIG_DATA)).data, 0xffff_ffff, "no bus 1 yet");
        // Secondary 1, subordinate 1, as the PROM numbers it.
        pci.write32(bridge(macepci_reg::CONFIG_ADDR), cfg(4 << 11, 6));
        pci.write32(bridge(macepci_reg::CONFIG_DATA), 0x0001_0100);
        pci.write32(bridge(macepci_reg::CONFIG_ADDR), behind(0));
        assert_eq!(pci.read32(bridge(macepci_reg::CONFIG_DATA)).data, 0x5678_1234);
        pci.write32(bridge(macepci_reg::CONFIG_ADDR), cfg(2 << 16 | 2 << 11, 0));
        assert_eq!(pci.read32(bridge(macepci_reg::CONFIG_DATA)).data, 0xffff_ffff, "bus 2 is not behind it");
    }

    /// An NVMe card in the option slot, driven as the CPU and a bus master
    /// would drive it: registers through the CPU's memory window (lanes and
    /// all), queues and data in RAM through the native view, and the card's
    /// pin on CRIME input 10.
    struct NvmeHost {
        pci: Arc<MacePci>,
        ram: Arc<O2Memory>,
    }

    impl crate::dev::nvme::tests::HostIo for NvmeHost {
        fn w(&self, off: u32, v: u32) {
            self.pci.window_write(PCI_MEM_BASE + off, v as u64, 4);
        }
        fn r(&self, off: u32) -> u32 {
            self.pci.window_read(PCI_MEM_BASE + off, 4) as u32
        }
        fn put(&self, at: u64, data: &[u8]) {
            self.ram.write(at as usize, data);
        }
        fn get(&self, at: u64, len: usize) -> Vec<u8> {
            self.ram.bytes(at as usize, len)
        }
        fn bus(&self, at: u64) -> u64 {
            PCI_NATIVE_VIEW as u64 + at
        }
    }

    fn nvme_in_the_option_slot(bridge: bool) {
        let crime = Arc::new(Crime(Mutex::new(Vec::new())));
        let ram = test_ram();
        let pci = MacePci::new(&PCI0, crime.clone(), ram.clone());
        let img = crate::dev::nvme::tests::temp_image(if bridge { "mace-ppb" } else { "mace" });
        let card = crate::dev::nvme::Card::open(&crate::config::NvmeConfig {
            path: img.to_string_lossy().into_owned(),
            overlay: false,
            bridge,
            ..Default::default()
        })
        .unwrap();
        card.fit(pci.bus(), IP32_OPTION_SLOT);
        let cycle = |bus: u32, dev: u32, reg: u32, val: u32| {
            pci.write32(bridge_reg(macepci_reg::CONFIG_ADDR), cfg(bus << 16 | dev << 11, reg));
            pci.write32(bridge_reg(macepci_reg::CONFIG_DATA), val);
        };
        let (bus, dev) = if bridge {
            // As the PROM does: number the bus, open memory 0x80000000 up.
            cycle(0, 3, 6, 0x0001_0100);
            cycle(0, 3, 8, 0x800f_8000);
            cycle(0, 3, 1, PCI_COMMAND_MEMORY);
            (1, 0)
        } else {
            (0, 3)
        };
        cycle(bus, dev, 4, 0x8000_0000);
        cycle(bus, dev, 5, 0);
        cycle(bus, dev, 1, PCI_COMMAND_MEMORY);
        let host = NvmeHost { pci: pci.clone(), ram };
        crate::dev::nvme::tests::smoke(&host);
        // The completions raised INTA of the slot, CRIME input 10, and the
        // host's reading them dropped it.
        assert!(crime.0.lock().unwrap().contains(&(10, true)));
        assert_eq!(crime.0.lock().unwrap().last(), Some(&(10, false)));
    }

    fn bridge_reg(reg: u32) -> u64 {
        bridge(reg)
    }

    #[test]
    fn an_nvme_card_in_the_option_slot_masters_ram_and_raises_input_10() {
        nvme_in_the_option_slot(false);
    }

    #[test]
    fn an_nvme_card_behind_a_bridge_in_the_option_slot_does_the_same() {
        nvme_in_the_option_slot(true);
    }
}
