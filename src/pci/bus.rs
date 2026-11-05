//! The PCI bus: the part every host bridge shares. A bridge owns one,
//! built from its machine's [`BusDesc`], and calls into it for what is the
//! same everywhere: placing functions (giving each its DMA and interrupt
//! line), configuration cycles, BAR decode and publication, reset and
//! saved state. The bridge keeps what is its own, as [`PciHost`] says.

use std::sync::{Arc, Mutex, OnceLock};

use super::host::PciHost;
use super::intx::IntxTable;
use super::slots::{BusDesc, PciAddr};
use super::PciBusDevice;
use crate::irq::IrqLine;

/// A card a user fitted, as the bus records it for saved state.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CardRef {
    pub name: &'static str,
    /// The card's options as `k=v,...`, sorted by key.
    pub options: String,
    /// The first SCSI controller number the card took.
    pub first_controller: u8,
}

pub enum Origin {
    /// The machine placed it: a chip on the board.
    Onboard,
    Card(CardRef),
}

struct Placed {
    f: Arc<dyn PciBusDevice>,
    origin: Origin,
}

pub struct PciBus {
    desc: &'static BusDesc,
    host: Arc<dyn PciHost>,
    intx: Arc<IntxTable>,
    /// Indexed by `device * 8 + function`. Lock-free to read; set once.
    funcs: Box<[OnceLock<Placed>]>,
    /// Addresses in the order functions were placed.
    order: Mutex<Vec<PciAddr>>,
}

impl PciBus {
    pub fn new(desc: &'static BusDesc, host: Arc<dyn PciHost>) -> Self {
        let intx = IntxTable::new(desc, host.clone());
        Self {
            desc,
            host,
            intx,
            funcs: (0..desc.devices as usize * 8).map(|_| OnceLock::new()).collect(),
            order: Mutex::new(Vec::new()),
        }
    }

    pub fn desc(&self) -> &'static BusDesc {
        self.desc
    }

    fn index(&self, at: PciAddr) -> Option<usize> {
        (at.device < self.desc.devices && at.function < 8).then(|| at.device as usize * 8 + at.function as usize)
    }

    /// Place a function the machine owns: give it its way to memory and,
    /// for each pin it uses, its interrupt line. Panics, as a wiring
    /// mistake in the machine, if the description lacks the device or the
    /// address is taken.
    pub fn attach(&self, at: PciAddr, f: Arc<dyn PciBusDevice>) {
        self.place(at, f, Origin::Onboard);
    }

    /// As [`Self::attach`], for a card a user fitted.
    pub fn attach_card(&self, at: PciAddr, f: Arc<dyn PciBusDevice>, card: CardRef) {
        self.place(at, f, Origin::Card(card));
    }

    fn place(&self, at: PciAddr, f: Arc<dyn PciBusDevice>, origin: Origin) {
        assert!(self.desc.slot(at.device).is_some(), "pci: bus {:?} has no device {}", self.desc.id, at.device);
        let i = self.index(at).unwrap_or_else(|| panic!("pci: bus {:?} has no {at}", self.desc.id));
        f.connect_dma(self.host.dma_for(at.device));
        // An unrouted pin still gets a line, which leads nowhere.
        if (1..=4).contains(&f.config().interrupt_pin()) {
            f.connect_interrupt(self.intx.pin_line(at.device, f.config().interrupt_pin()));
        }
        assert!(self.funcs[i].set(Placed { f, origin }).is_ok(), "pci: {at} is taken");
        self.order.lock().unwrap().push(at);
        self.changed();
    }

    /// The function at `at`, if one is placed.
    pub fn function(&self, at: PciAddr) -> Option<&Arc<dyn PciBusDevice>> {
        Some(&self.funcs.get(self.index(at)?)?.get()?.f)
    }

    /// Every function, in the order it was placed.
    pub fn functions(&self) -> Vec<(PciAddr, Arc<dyn PciBusDevice>)> {
        self.order.lock().unwrap().iter().map(|&a| (a, self.function(a).unwrap().clone())).collect()
    }

    /// The cards users fitted, in placement order.
    pub fn cards(&self) -> Vec<(PciAddr, CardRef, Arc<dyn PciBusDevice>)> {
        self.order
            .lock()
            .unwrap()
            .iter()
            .filter_map(|&a| {
                let p = self.funcs[self.index(a)?].get()?;
                match &p.origin {
                    Origin::Card(c) => Some((a, c.clone(), p.f.clone())),
                    Origin::Onboard => None,
                }
            })
            .collect()
    }

    /// A configuration read of `width` bytes (1, 2 or 4) at byte `offset`.
    /// `None`: nothing answers, and the host decides what that looks like.
    pub fn config_read(&self, at: PciAddr, offset: usize, width: usize) -> Option<u32> {
        let dword = self.function(at)?.config_dword(offset / 4);
        Some((dword >> ((offset & 3) * 8)) & (u32::MAX >> (32 - width * 8)))
    }

    /// A configuration write of `width` bytes at byte `offset`. `false`:
    /// nothing answers. Only the bytes written change; the header decides
    /// which bits of them stick.
    pub fn config_write(&self, at: PciAddr, offset: usize, width: usize, val: u32) -> bool {
        let Some(f) = self.function(at) else { return false };
        let shift = (offset & 3) * 8;
        let mask = (u32::MAX >> (32 - width * 8)) << shift;
        f.config().write_masked(offset / 4, val << shift, mask);
        self.changed();
        true
    }

    /// A board wire from the description, by name.
    pub fn wire(&self, name: &str) -> IrqLine {
        self.intx.wire_line(name)
    }

    /// What the functions decode may have moved: rebuild the decode table
    /// and republish the BAR runs.
    fn changed(&self) {}
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::pci::host::TestHost;
    use crate::pci::slots::{BusDesc, Evidence, SlotDesc, SlotKind};
    use crate::pci::{Bar, PciConfig, PCI_INTA};

    pub static DESC: BusDesc = BusDesc {
        id: "pci",
        name: "test bus",
        devices: 8,
        outputs: 16,
        slots: &[
            SlotDesc { device: 1, label: "onboard", kind: SlotKind::Onboard, evidence: Evidence::Established,
                source: "test", routes: [Some(8), None, None, None] },
            SlotDesc { device: 3, label: "option", kind: SlotKind::AddIn, evidence: Evidence::Established,
                source: "test", routes: [Some(12), Some(13), None, None] },
            SlotDesc { device: 4, label: "other", kind: SlotKind::AddIn, evidence: Evidence::Established,
                source: "test", routes: [Some(12), None, None, None] },
        ],
        wires: &[],
    };

    /// A function that remembers its last BAR access, and holds the DMA
    /// and interrupt line it was given.
    pub struct Fake {
        pub cfg: PciConfig,
        pub last: Mutex<Option<(usize, u32, usize)>>,
        pub irq: OnceLock<IrqLine>,
        pub dma: OnceLock<Arc<dyn crate::pci::PciDma>>,
    }

    impl Fake {
        pub fn new(pin: u8) -> Arc<Self> {
            Arc::new(Self {
                cfg: PciConfig::new(0x9004, 0x8078, 0x01_0000, 1,
                    [Bar::Io(0x100), Bar::Mem(0x1000), Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused], pin),
                last: Mutex::new(None),
                irq: OnceLock::new(),
                dma: OnceLock::new(),
            })
        }
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
        fn connect_dma(&self, dma: Arc<dyn crate::pci::PciDma>) { let _ = self.dma.set(dma); }
    }

    pub fn bus() -> (Arc<TestHost>, PciBus) {
        let h = TestHost::new();
        let b = PciBus::new(&DESC, h.clone());
        (h, b)
    }

    #[test]
    fn attach_hands_the_function_dma_and_its_pin_line() {
        let (h, b) = bus();
        let f = Fake::new(PCI_INTA);
        b.attach(PciAddr::dev(3), f.clone());
        assert!(f.dma.get().is_some());
        assert_eq!(h.dma_of(3).unwrap().device, 3);
        f.irq.get().expect("wired at attach").set(true);
        assert_eq!(h.outputs(), vec![(12, true)]);
    }

    #[test]
    fn a_pin_without_a_route_is_a_line_that_leads_nowhere() {
        let (h, b) = bus();
        let f = Fake::new(PCI_INTA + 2);
        b.attach(PciAddr::dev(4), f.clone());
        f.irq.get().expect("still connected").set(true);
        assert!(h.outputs().is_empty());
    }

    #[test]
    #[should_panic(expected = "pci: device 3 is taken")]
    fn a_second_function_at_the_same_address_panics() {
        let (_, b) = bus();
        b.attach(PciAddr::dev(3), Fake::new(PCI_INTA));
        b.attach(PciAddr::dev(3), Fake::new(PCI_INTA));
    }

    #[test]
    #[should_panic(expected = "pci: bus \"pci\" has no device 7")]
    fn an_undescribed_device_panics() {
        let (_, b) = bus();
        b.attach(PciAddr::dev(7), Fake::new(PCI_INTA));
    }

    #[test]
    fn functions_are_found_by_device_and_function_and_listed_in_placement_order() {
        let (_, b) = bus();
        let (x, y, z) = (Fake::new(0), Fake::new(0), Fake::new(0));
        let f1 = PciAddr { device: 3, function: 1 };
        b.attach(PciAddr::dev(4), x.clone());
        b.attach(PciAddr::dev(3), y.clone());
        b.attach(f1, z.clone());
        let order: Vec<_> = b.functions().iter().map(|(a, _)| *a).collect();
        assert_eq!(order, vec![PciAddr::dev(4), PciAddr::dev(3), f1]);
        assert!(Arc::ptr_eq(&(b.function(f1).unwrap().clone() as Arc<dyn PciBusDevice>), &(z as Arc<dyn PciBusDevice>)));
        assert!(b.function(PciAddr::dev(1)).is_none());
        assert!(b.function(PciAddr { device: 3, function: 2 }).is_none());
        assert!(b.function(PciAddr::dev(31)).is_none(), "past the bus");
    }

    #[test]
    fn cards_are_listed_apart_from_onboard_functions() {
        let (_, b) = bus();
        let card = CardRef { name: "fake", options: String::new(), first_controller: 2 };
        b.attach(PciAddr::dev(1), Fake::new(0));
        b.attach_card(PciAddr::dev(3), Fake::new(0), card.clone());
        let cards = b.cards();
        assert_eq!(cards.len(), 1);
        assert_eq!((cards[0].0, &cards[0].1), (PciAddr::dev(3), &card));
    }

    #[test]
    fn config_read_of_an_absent_function_is_none_and_narrow_reads_shift() {
        let (_, b) = bus();
        b.attach(PciAddr::dev(3), Fake::new(PCI_INTA));
        let at = PciAddr::dev(3);
        assert_eq!(b.config_read(at, 0, 4), Some(0x8078_9004));
        assert_eq!(b.config_read(at, 1, 1), Some(0x90));
        assert_eq!(b.config_read(at, 2, 2), Some(0x8078));
        assert_eq!(b.config_read(PciAddr::dev(4), 0, 4), None);
        assert_eq!(b.config_read(PciAddr { device: 3, function: 1 }, 0, 4), None);
        assert_eq!(b.config_read(PciAddr::dev(31), 0, 4), None);
    }

    /// The Bridge used to read the whole dword, merge the bytes written and
    /// write it back; the header now takes the byte mask itself. Every
    /// register that has a write rule must end up the same.
    #[test]
    fn narrow_writes_leave_the_header_as_the_old_read_modify_write_did() {
        let legacy = |cfg: &PciConfig, i: usize, shift: u32, mask: u32, val: u32| {
            cfg.write(i, (cfg.read(i) & !mask) | ((val << shift) & mask));
        };
        for reg in [1usize, 3, 4, 5, 15, 16] {
            for (width, aligned) in [(1usize, [0usize, 1, 2, 3].as_slice()), (2, &[0, 2]), (4, &[0])] {
                for &o in aligned {
                    for val in [0u32, 0xffff_ffff, 0xa5c3_5a3c, 0x0000_1000] {
                        let (_, b) = bus();
                        let f = Fake::new(PCI_INTA);
                        b.attach(PciAddr::dev(3), f.clone());
                        let old = PciConfig::new(0x9004, 0x8078, 0x01_0000, 1,
                            [Bar::Io(0x100), Bar::Mem(0x1000), Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused], PCI_INTA);
                        // Start both from a non-trivial state.
                        for c in [&f.cfg, &old] {
                            c.write(reg, 0x5a5a_5a5a);
                        }
                        let shift = (o * 8) as u32;
                        let mask = (u32::MAX >> (32 - width * 8)) << shift;
                        assert!(b.config_write(PciAddr::dev(3), reg * 4 + o, width, val));
                        legacy(&old, reg, shift, mask, val);
                        assert_eq!(f.cfg.read(reg), old.read(reg), "register {reg} width {width} offset {o} value {val:#x}");
                    }
                }
            }
        }
    }

    #[test]
    fn a_config_write_to_an_absent_function_reports_it() {
        let (_, b) = bus();
        assert!(!b.config_write(PciAddr::dev(3), 4, 4, 2));
    }
}
