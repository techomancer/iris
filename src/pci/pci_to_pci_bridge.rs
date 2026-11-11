//! A PCI-to-PCI bridge: a function with a type 1 header and a bus behind it.
//!
//! Firmware numbers the bus (secondary and subordinate bus numbers) and
//! programs the I/O, memory and prefetchable windows; the bridge forwards the
//! BARs of the functions behind it that lie inside an enabled window, so the
//! bus in front decodes them like its own. A pin of the function at device
//! `d` behind the bridge reaches pin `(pin + d) mod 4` of the bridge's own
//! device, and its DMA is the bridge's.

use std::sync::Arc;

use super::{DevFn, PciBus, PciConfig, PciFunction, PciFunctionState};

/// Bridge control bit 6: secondary bus reset.
const SECONDARY_RESET: u32 = 1 << 22;

pub struct PciToPciBridge {
    pci: PciFunctionState,
    sec: PciBus,
}

impl PciToPciBridge {
    /// A bridge with PCI identity `vendor` and `device`, to be fitted at `at`
    /// on `parent`, and the bus behind it. The caller attaches it with
    /// `parent.attach(at, bridge)`. Only the type 1 header is modelled, not
    /// the registers of any chip, so the identity is the machine's to name.
    pub fn new(parent: &PciBus, at: DevFn, vendor: u16, device: u16) -> Arc<Self> {
        Self::with_revision(parent, at, vendor, device, 0)
    }

    /// As [`Self::new`], with the revision byte the chip reports at 0x08.
    pub fn with_revision(parent: &PciBus, at: DevFn, vendor: u16, device: u16, revision: u8) -> Arc<Self> {
        Arc::new(Self {
            pci: PciFunctionState::new(PciConfig::type1(vendor, device, revision)),
            sec: PciBus::secondary(parent.clone(), at),
        })
    }
}

impl PciToPciBridge {
    /// The bus behind the bridge.
    pub fn secondary(&self) -> &PciBus {
        &self.sec
    }
}

impl PciFunction for PciToPciBridge {
    fn state(&self) -> &PciFunctionState {
        &self.pci
    }

    fn bar_read(&self, _bar: usize, _off: u32, _width: usize) -> u32 {
        0
    }

    fn bar_write(&self, _bar: usize, _off: u32, _val: u32, _width: usize) {}


    fn secondary(&self) -> Option<&PciBus> {
        Some(&self.sec)
    }

    /// A write that raises secondary bus reset resets the bus behind the
    /// bridge and nothing else.
    fn config_written(&self, reg: usize, old: u32, new: u32) {
        if reg == 15 && old & SECONDARY_RESET == 0 && new & SECONDARY_RESET != 0 {
            self.sec.reset();
        }
    }

    /// RST# on the primary bus reaches everything behind the bridge.
    fn reset(&self) {
        self.sec.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pci::testing::{TestCard, TestHost};
    use crate::pci::{Bar, CpuWindow, IntxRouting, PciHost, Space, PCI_COMMAND_MEMORY};
    use std::sync::{Mutex, Weak};

    // The bridge sits at device 3, whose INTA..INTD reach outputs 0 to 3.
    static W: IntxRouting = IntxRouting {
        devices: 8,
        irqs: 4,
        intx: &[(3, [Some(0), Some(1), Some(2), Some(3)]), (4, [Some(0), None, None, None])],
        board_irqs: &[],
    };

    fn at(device: u8, function: u8) -> DevFn {
        DevFn { device, function }
    }

    fn bars() -> [Bar; 6] {
        [Bar::Mem(0x1000), Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused]
    }

    struct Rig {
        host: Arc<TestHost>,
        root: PciBus,
        bridge: Arc<PciToPciBridge>,
    }

    /// A bridge at device 3 numbering a bus 1 behind it, with window
    /// registers programmed for memory 0x8000_0000..0x800f_ffff.
    fn rig() -> Rig {
        let host = Arc::new(TestHost::default());
        let up: Weak<dyn PciHost> = Arc::downgrade(&(host.clone() as Arc<dyn PciHost>));
        let root = PciBus::new(0, &W, up);
        let bridge = PciToPciBridge::new(&root, at(3, 0), 0x1234, 0x5678);
        root.attach(at(3, 0), bridge.clone());
        root.config_write(at(3, 0), 0x18, 4, 0x0001_0100); // primary 0, secondary 1, subordinate 1
        root.config_write(at(3, 0), 0x20, 4, 0x8000_8000);
        root.config_write(at(3, 0), 0x04, 2, PCI_COMMAND_MEMORY);
        Rig { host, root, bridge }
    }

    /// A card on the secondary bus whose BAR 0 is at `base`, decoding on.
    fn child(r: &Rig, device: u8, pin: u8, base: u32) -> Arc<TestCard> {
        let c = TestCard::new(bars(), pin);
        let sec = r.bridge.secondary();
        sec.attach(at(device, 0), c.clone());
        sec.config_write(at(device, 0), 0x10, 4, base);
        sec.config_write(at(device, 0), 0x04, 2, PCI_COMMAND_MEMORY);
        c
    }

    fn windows() -> Vec<CpuWindow> {
        let w = |space, cpu, pci, len| CpuWindow { space, cpu, pci, len, mask: 0xffff_ffff };
        vec![w(Space::Io, 0x1800_0000, 0, 0x200_0000), w(Space::Mem, 0x1a00_0000, 0x8000_0000, 0x200_0000)]
    }

    fn cpus(r: &Rig) -> Arc<Mutex<Vec<(u64, u16)>>> {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = seen.clone();
        r.root.on_bar_mappings(Box::new(move |runs| *sink.lock().unwrap() = runs.iter().map(|b| (b.cpu, b.window)).collect()));
        r.root.set_host_windows(&windows);
        seen
    }

    #[test]
    fn type_1_header_probe_and_window_registers() {
        let host = Arc::new(TestHost::default());
        let up: Weak<dyn PciHost> = Arc::downgrade(&(host.clone() as Arc<dyn PciHost>));
        let root = PciBus::new(0, &W, up);
        root.attach(at(3, 0), PciToPciBridge::new(&root, at(3, 0), 0x1234, 0x5678));
        let read = |off, w| root.config_read(at(3, 0), off, w).unwrap();
        assert_eq!(read(0x0e, 1), 1, "header type 1");
        assert_eq!(read(0x09, 3), 0x06_0400, "a PCI-to-PCI bridge");
        assert_eq!(read(0x08, 1), 0, "revision 0 unless the chip names one");
        for (off, probe, stuck) in [(0x18, 0xffff_ffffu32, 0xffff_ffffu32), (0x1c, !0, 0xf0f0), (0x20, !0, 0xfff0_fff0), (0x24, !0, 0xfff0_fff0), (0x28, !0, 0), (0x2c, !0, 0), (0x30, !0, 0), (0x38, !0, 0)] {
            root.config_write(at(3, 0), off, 4, probe);
            assert_eq!(read(off, 4), stuck, "register {off:#x}");
        }
        root.config_write(at(3, 0), 0x10, 4, !0);
        assert_eq!(read(0x10, 4), 0, "no BARs");
        root.config_write(at(3, 0), 0x19, 1, 0x05);
        assert_eq!(read(0x19, 1), 0x05, "the secondary bus number is a byte");
    }

    #[test]
    fn a_bridge_reports_the_revision_it_is_given() {
        let host = Arc::new(TestHost::default());
        let up: Weak<dyn PciHost> = Arc::downgrade(&(host.clone() as Arc<dyn PciHost>));
        let root = PciBus::new(0, &W, up);
        root.attach(at(3, 0), PciToPciBridge::with_revision(&root, at(3, 0), 0x1234, 0x5678, 0x21));
        assert_eq!(root.config_read(at(3, 0), 0x08, 4), Some(0x0604_0021));
        root.reset();
        assert_eq!(root.config_read(at(3, 0), 0x08, 1), Some(0x21), "and keeps it across RST#");
    }

    #[test]
    fn a_child_bar_inside_the_bridge_window_gets_a_run_with_the_parents_window_index() {
        let r = rig();
        let seen = cpus(&r);
        child(&r, 2, 0, 0x8000_2000);
        assert_eq!(*seen.lock().unwrap(), vec![(0x1a00_2000, 1)]);
        let (_, bar, off) = r.root.decode(Space::Mem, 0x8000_2010).expect("the slow path finds it too");
        assert_eq!((bar, off), (0, 0x10));
    }

    /// Firmware opens the bridge's memory window again, wider, over a child
    /// BAR that lay outside the old window (a second enumeration with no
    /// reset in between): the child gains its run and the slow path finds
    /// it, because a window write updates the bus in front.
    #[test]
    fn widening_the_window_over_an_unmoved_bar_publishes_the_child() {
        let r = rig();
        let seen = cpus(&r);
        child(&r, 0, 0, 0x8080_0000);
        assert_eq!(*seen.lock().unwrap(), vec![], "outside the 1 MB window");
        assert!(r.root.decode(Space::Mem, 0x8080_0008).is_none());
        r.root.config_write(at(3, 0), 0x22, 2, 0x80f0);
        assert_eq!(*seen.lock().unwrap(), vec![(0x1a80_0000, 1)]);
        assert_eq!(r.root.decode(Space::Mem, 0x8080_0008).map(|(_, b, o)| (b, o)), Some((0, 8)));
    }

    #[test]
    fn a_child_bar_outside_the_window_has_no_run_and_no_bar_access() {
        let r = rig();
        let seen = cpus(&r);
        let c = child(&r, 2, 0, 0x9000_0000);
        assert!(seen.lock().unwrap().is_empty());
        assert!(r.root.decode(Space::Mem, 0x9000_0010).is_none());
        assert!(r.bridge.secondary().decode(Space::Mem, 0x9000_0010).is_some(), "the bus behind still decodes it");
        let _ = c;
    }

    #[test]
    fn disabling_the_bridge_memory_enable_removes_child_runs() {
        let r = rig();
        let seen = cpus(&r);
        child(&r, 2, 0, 0x8000_2000);
        assert_eq!(seen.lock().unwrap().len(), 1);
        r.root.config_write(at(3, 0), 0x04, 2, 0);
        assert!(seen.lock().unwrap().is_empty());
        assert!(r.root.decode(Space::Mem, 0x8000_2010).is_none());
        r.root.config_write(at(3, 0), 0x04, 2, PCI_COMMAND_MEMORY);
        assert_eq!(seen.lock().unwrap().len(), 1, "and enabling it brings the run back");
    }

    /// A pin of the function at device `d` behind the bridge reaches the
    /// bridge's pin `(pin + d) mod 4`: device 2 INTA is the bridge's INTC.
    #[test]
    fn a_child_interrupt_reaches_parent_pin_p_plus_d_mod_4() {
        for (device, pin, output) in [(2u8, 1u8, 2u32), (2, 2, 3), (2, 3, 0), (1, 1, 1), (0, 4, 3)] {
            let r = rig();
            let c = child(&r, device, pin, 0x8000_1000);
            c.state().irq().unwrap().set(true);
            assert_eq!(r.host.irqs(), vec![(output as u8, true)], "device {device} pin {pin}");
        }
    }

    #[test]
    fn a_child_dma_uses_the_bridges_root_slot() {
        let r = rig();
        let c = child(&r, 2, 0, 0x8000_2000);
        let mut b = [0u8; 1];
        assert!(c.state().dma().unwrap().dma_read(0, &mut b).is_ok());
        assert_eq!(b[0], 3, "the bridge's device number, not the child's");
    }

    #[test]
    fn route_finds_the_bus_by_secondary_and_subordinate() {
        let r = rig();
        let inner = PciToPciBridge::new(r.bridge.secondary(), at(1, 0), 0x1234, 0x5678);
        r.bridge.secondary().attach(at(1, 0), inner.clone());
        // Bus 1 holds bridge 1.0, which numbers bus 2; the outer one takes 1 to 2.
        r.root.config_write(at(3, 0), 0x18, 4, 0x0002_0100);
        r.bridge.secondary().config_write(at(1, 0), 0x18, 4, 0x0002_0201);
        assert_eq!(r.root.find_bus(0).unwrap().number(), r.root.number());
        assert_eq!(r.root.find_bus(1).unwrap().number(), r.bridge.secondary().number());
        assert_eq!(r.root.find_bus(2).unwrap().number(), inner.secondary().number());
        assert!(r.root.find_bus(3).is_none());
    }

    /// Raising secondary bus reset resets what is behind the bridge and
    /// nothing else.
    #[test]
    fn bridge_control_bit_6_resets_the_secondary_only() {
        let r = rig();
        let log = Arc::new(Mutex::new(Vec::new()));
        let behind = TestCard::logged("behind", bars(), 0, &log);
        r.bridge.secondary().attach(at(2, 0), behind);
        r.root.attach(at(5, 0), TestCard::logged("beside", bars(), 0, &log));
        r.root.config_write(at(3, 0), 0x3e, 2, 0x40);
        assert_eq!(*log.lock().unwrap(), ["behind cmd=0x0"]);
        r.root.config_write(at(3, 0), 0x3e, 2, 0x40);
        assert_eq!(log.lock().unwrap().len(), 1, "holding it is not another reset");
    }

    #[test]
    fn a_bridge_control_write_without_bit_6_leaves_the_secondary_alone() {
        let r = rig();
        let log = Arc::new(Mutex::new(Vec::new()));
        r.bridge.secondary().attach(at(2, 0), TestCard::logged("behind", bars(), 0, &log));
        r.root.config_write(at(3, 0), 0x3e, 2, 0x01);
        assert!(log.lock().unwrap().is_empty());
        // RST# on the primary bus does reach behind the bridge.
        r.root.reset();
        assert_eq!(*log.lock().unwrap(), ["behind cmd=0x0"]);
    }

    #[test]
    fn a_secondary_bus_saves_and_loads_with_its_parent() {
        use crate::traits::Saveable;
        let r = rig();
        child(&r, 2, 0, 0x8000_2000);
        let v = r.root.save_state();
        let r2 = rig();
        child(&r2, 2, 0, 0x8000_6000);
        r2.root.load_state(&v).unwrap();
        assert_eq!(r2.root.save_state(), v);
        assert!(r2.root.decode(Space::Mem, 0x8000_2010).is_some() && r2.root.decode(Space::Mem, 0x8000_6010).is_none());
    }
}
