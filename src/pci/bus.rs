//! The PCI bus: the part every host bridge shares. A bridge owns one,
//! built from its machine's [`BusDesc`], and calls into it for what is the
//! same everywhere: placing functions (giving each its DMA and interrupt
//! line), configuration cycles, BAR decode and publication, reset and
//! saved state. The bridge keeps what is its own, as [`PciHost`] says.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use super::host::PciHost;
use super::intx::IntxTable;
use super::slots::{BusDesc, PciAddr};
use super::{bar_runs, BarAt, BarRun, BarRunSink, PciBar, PciBusDevice, Space};
use crate::irq::IrqLine;
use crate::snapshot::get_field;
use crate::traits::{BusDevice, Saveable};

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

/// Which function, BAR, CPU address, lanes and mask a [`PciBar`] entry serves.
type EntryKey = (usize, usize, u64, u32, u64);

/// The most BAR windows open at once on a bus. A window is one word of the
/// decode table; more than this and the extra are ignored with a warning.
const MAX_WINDOWS: usize = 64;

/// A BAR window as the decode table holds it, packed into one word: base
/// (bits 0-31), log2 of the size (32-37), I/O space (38), function index
/// `device * 8 + function` (40-47), BAR (48-50).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Window {
    space: Space,
    func: u8,
    bar: usize,
    base: u32,
    size: u32,
}

impl Window {
    fn pack(self) -> u64 {
        self.base as u64
            | (self.size.trailing_zeros() as u64) << 32
            | ((self.space == Space::Io) as u64) << 38
            | (self.func as u64) << 40
            | (self.bar as u64) << 48
    }

    fn unpack(w: u64) -> Self {
        Self {
            space: if w & (1 << 38) != 0 { Space::Io } else { Space::Mem },
            func: (w >> 40) as u8,
            bar: ((w >> 48) & 7) as usize,
            base: w as u32,
            size: 1u32 << ((w >> 32) & 0x3f),
        }
    }
}

pub struct PciBus {
    desc: &'static BusDesc,
    host: Arc<dyn PciHost>,
    intx: Arc<IntxTable>,
    /// Indexed by `device * 8 + function`. Lock-free to read; set once.
    funcs: Box<[OnceLock<Placed>]>,
    /// Addresses in the order functions were placed.
    order: Mutex<Vec<PciAddr>>,
    /// The BAR windows the functions decode, rebuilt from their headers
    /// after every change to them. A function's BAR comparators are its own
    /// registers on the bus, not anything the bus locks, so a lookup reads
    /// the table without a lock; configuration writes and accesses both
    /// come from the CPU. Writers take `rebuild`.
    windows: Box<[AtomicU64]>,
    nwindows: AtomicUsize,
    rebuild: Mutex<()>,
    /// Where the runs of the windows a BAR decodes whole go (the machine
    /// maps them over the host's windows), and the entries made for them,
    /// one per (function, BAR, CPU address, lanes, mask).
    sink: RwLock<Option<BarRunSink>>,
    entries: Mutex<HashMap<EntryKey, Arc<dyn BusDevice>>>,
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
            windows: (0..MAX_WINDOWS).map(|_| AtomicU64::new(0)).collect(),
            nwindows: AtomicUsize::new(0),
            rebuild: Mutex::new(()),
            sink: RwLock::new(None),
            entries: Mutex::new(HashMap::new()),
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

    /// RST#: every header returns to power-up, so no function decodes
    /// until firmware assigns its BARs again. The functions' own state is
    /// theirs to reset, and interrupt levels are not touched: the lines
    /// are held by functions this does not reset.
    pub fn reset(&self) {
        for p in self.funcs.iter().filter_map(|p| p.get()) {
            p.f.config().reset();
        }
        self.changed();
    }

    /// Every header, by `<device>.<function>`, and the interrupt levels.
    pub fn save_state(&self) -> toml::Value {
        let mut devices = toml::map::Map::new();
        for (at, f) in self.functions() {
            devices.insert(format!("{}.{}", at.device, at.function), f.config().save_state());
        }
        let mut t = toml::map::Map::new();
        t.insert("devices".into(), toml::Value::Table(devices));
        t.insert("intx".into(), self.intx.save_state());
        toml::Value::Table(t)
    }

    /// Restore the headers and interrupt levels. The saved state must name
    /// exactly the functions this bus has. Nothing is driven into the
    /// host: it loads its own state, and [`Self::redrive`] follows if it
    /// needs the outputs put right.
    pub fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        let Some(toml::Value::Table(saved)) = get_field(v, "devices") else {
            return Err("pci: saved state has no devices".into());
        };
        let mut headers = Vec::new();
        for (key, hdr) in saved {
            let at = key.split_once('.')
                .and_then(|(d, f)| Some(PciAddr { device: d.parse().ok()?, function: f.parse().ok()? }))
                .ok_or_else(|| format!("pci: saved device {key:?} is not <device>.<function>"))?;
            let f = self.function(at).ok_or_else(|| format!("pci: saved state has {at}, which is empty here"))?;
            headers.push((at, f, hdr));
        }
        for (at, _) in self.functions() {
            if !headers.iter().any(|h| h.0 == at) {
                return Err(format!("pci: this bus has {at}, the saved state has none"));
            }
        }
        for (at, f, hdr) in headers {
            f.config().load_state(hdr).map_err(|e| format!("{at}: {e}"))?;
        }
        self.intx.load_state(get_field(v, "intx").ok_or("pci: saved state has no interrupt levels")?)?;
        self.changed();
        Ok(())
    }

    /// Drive every routed host output to the level its sources add up to.
    pub fn redrive(&self) {
        self.intx.redrive();
    }

    /// A board wire from the description, by name.
    pub fn wire(&self, name: &str) -> IrqLine {
        self.intx.wire_line(name)
    }

    /// A function changed its own header, outside a configuration cycle (a
    /// power-on reset of its own): tell the bus, which rebuilds the decode
    /// table and republishes. Headers the bus writes or resets itself need no
    /// call. Until it is made the table still decodes the old BARs.
    pub fn refresh(&self) {
        self.changed();
    }

    /// What the functions decode may have moved: rebuild the decode table
    /// and republish the BAR runs.
    fn changed(&self) {
        self.rebuild();
        self.publish();
    }

    /// Send the runs of the host's CPU windows that one BAR decodes whole
    /// to `sink`, now and after every change to what the functions decode.
    /// Each run is a [`PciBar`] at its CPU address: the machine maps it on
    /// its physical bus over the host's window, so the processor reaches
    /// the function with no search of the windows. Everything else in a
    /// window (BARs smaller than a page, unclaimed addresses) still goes
    /// to the host.
    pub fn on_bar_runs(&self, sink: BarRunSink) {
        *self.sink.write().unwrap() = Some(sink);
        self.publish();
    }

    /// Publish the runs again. A host calls this when one of its own
    /// registers moves a window; the bus does it on every header change.
    pub fn publish(&self) {
        let sink = self.sink.read().unwrap();
        let Some(sink) = sink.as_ref() else { return };
        let funcs: Vec<(usize, Arc<dyn PciBusDevice>)> =
            self.funcs.iter().enumerate().filter_map(|(i, p)| Some((i, p.get()?.f.clone()))).collect();
        let mut runs: Vec<BarRun> = Vec::new();
        for w in self.host.windows() {
            let entry = |b: BarAt| {
                self.entries
                    .lock()
                    .unwrap()
                    .entry((b.slot, b.bar, b.cpu_base, w.lanes_key, w.mask))
                    .or_insert_with(|| Arc::new(PciBar::new(w.lanes.clone(), b.dev.clone(), b.bar, w.mask, b.cpu_base)))
                    .clone()
            };
            runs.extend(bar_runs(&funcs, w.space, w.cpu, w.pci, w.len, entry));
        }
        sink(runs);
    }

    /// Rebuild the decode table from the functions' headers, in device,
    /// function and BAR order: the order in which an overlap resolves, the
    /// lowest first.
    fn rebuild(&self) {
        let _w = self.rebuild.lock().unwrap();
        let mut n = 0;
        for (func, p) in self.funcs.iter().enumerate() {
            let Some(p) = p.get() else { continue };
            for (space, bar, base, size) in p.f.config().windows() {
                if n == MAX_WINDOWS {
                    log::warn!("pci: more than {MAX_WINDOWS} BAR windows open; {} BAR {bar} ignored",
                        PciAddr { device: (func / 8) as u8, function: (func % 8) as u8 });
                    continue;
                }
                self.windows[n].store(Window { space, func: func as u8, bar, base, size }.pack(), Ordering::Relaxed);
                n += 1;
            }
        }
        self.nwindows.store(n, Ordering::Release);
    }

    /// The function decoding `addr` in `space`, the BAR, and the offset
    /// within it. The first window in table order that holds the address
    /// wins.
    pub fn target(&self, space: Space, addr: u32) -> Option<(&Arc<dyn PciBusDevice>, usize, u32)> {
        let n = self.nwindows.load(Ordering::Acquire);
        self.windows[..n].iter().find_map(|w| {
            let w = Window::unpack(w.load(Ordering::Relaxed));
            let off = addr.wrapping_sub(w.base);
            if w.space != space || off >= w.size {
                return None;
            }
            Some((&self.funcs[w.func as usize].get()?.f, w.bar, off))
        })
    }
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
            Self::with_bars([Bar::Io(0x100), Bar::Mem(0x1000), Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused], pin)
        }

        pub fn with_bars(bars: [Bar; 6], pin: u8) -> Arc<Self> {
            Arc::new(Self {
                cfg: PciConfig::new(0x9004, 0x8078, 0x01_0000, 1, bars, pin),
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

    /// Program register `reg` of the function at `at` through a config cycle.
    fn set(b: &PciBus, at: PciAddr, reg: usize, val: u32) {
        assert!(b.config_write(at, reg * 4, 4, val));
    }

    #[test]
    fn a_bar_decodes_only_when_its_space_is_enabled_and_assigned() {
        let (_, b) = bus();
        let f = Fake::new(PCI_INTA);
        let at = PciAddr::dev(3);
        b.attach(at, f.clone());
        set(&b, at, 5, 0x8000_1000);
        assert!(b.target(Space::Mem, 0x8000_1084).is_none(), "not before memory is enabled");
        set(&b, at, 1, 2);
        let (got, bar, off) = b.target(Space::Mem, 0x8000_1084).expect("decoded");
        assert!(Arc::ptr_eq(&got.clone(), &(f.clone() as Arc<dyn PciBusDevice>)));
        assert_eq!((bar, off), (1, 0x84));
        assert!(b.target(Space::Mem, 0x8000_2000).is_none(), "past the end of the BAR");
        assert!(b.target(Space::Mem, 0).is_none(), "a BAR left at zero decodes nothing");
    }

    #[test]
    fn overlapping_windows_resolve_to_the_lowest_device_then_the_lowest_bar() {
        let (_, b) = bus();
        let two_mem = || Fake::with_bars([Bar::Mem(0x1000), Bar::Unused, Bar::Mem(0x1000), Bar::Unused, Bar::Unused, Bar::Unused], PCI_INTA);
        let (d3, d4) = (two_mem(), two_mem());
        // Placed highest device first: the order of placement must not matter.
        b.attach(PciAddr::dev(4), d4.clone());
        b.attach(PciAddr::dev(3), d3.clone());
        for at in [PciAddr::dev(3), PciAddr::dev(4)] {
            set(&b, at, 4, 0x4000_0000);
            set(&b, at, 6, 0x4000_0000);
            set(&b, at, 1, 2);
        }
        let (f, bar, _) = b.target(Space::Mem, 0x4000_0010).unwrap();
        assert!(Arc::ptr_eq(&f.clone(), &(d3.clone() as Arc<dyn PciBusDevice>)), "device 3 before device 4");
        assert_eq!(bar, 0, "BAR 0 before BAR 2");
    }

    #[test]
    fn io_and_memory_are_decoded_separately() {
        let (_, b) = bus();
        let at = PciAddr::dev(3);
        b.attach(at, Fake::new(PCI_INTA));
        set(&b, at, 4, 0x1000);
        set(&b, at, 5, 0x1000);
        set(&b, at, 1, 1);
        assert_eq!(b.target(Space::Io, 0x1004).map(|t| (t.1, t.2)), Some((0, 4)));
        assert!(b.target(Space::Mem, 0x1004).is_none(), "memory is not enabled");
        set(&b, at, 1, 2);
        assert!(b.target(Space::Io, 0x1004).is_none(), "I/O is not enabled");
        assert_eq!(b.target(Space::Mem, 0x1004).map(|t| (t.1, t.2)), Some((1, 4)));
    }

    #[test]
    fn windows_pack_and_unpack() {
        for w in [
            Window { space: Space::Io, func: 255, bar: 5, base: 0xffff_f000, size: 0x1000 },
            Window { space: Space::Mem, func: 0, bar: 0, base: 0x8000_0000, size: 1 << 31 },
        ] {
            assert_eq!(Window::unpack(w.pack()), w);
        }
    }

    /// A host with the O2's three windows: low I/O, low memory (CPU
    /// 0x1a00_0000 onto PCI 0x8000_0000) and a one-to-one high memory.
    fn windowed() -> (Arc<TestHost>, PciBus, Arc<Mutex<Vec<BarRun>>>) {
        use crate::pci::host::CpuWindow;
        let (h, b) = bus();
        let w = |space, cpu, pci, len| CpuWindow { space, cpu, pci, len, mask: 0xffff_ffff, lanes: h.lanes(), lanes_key: 0 };
        h.set_windows(vec![
            w(Space::Io, 0x1800_0000, 0, 0x200_0000),
            w(Space::Mem, 0x1a00_0000, 0x8000_0000, 0x200_0000),
            w(Space::Mem, 0x8000_0000, 0x8000_0000, 0x8000_0000),
        ]);
        let runs = Arc::new(Mutex::new(Vec::<BarRun>::new()));
        let sink = runs.clone();
        b.on_bar_runs(Box::new(move |r| *sink.lock().unwrap() = r));
        (h, b, runs)
    }

    #[test]
    fn a_placed_bar_is_published_as_whole_page_runs() {
        let (_, b, runs) = windowed();
        let at = PciAddr::dev(3);
        b.attach(at, Fake::new(PCI_INTA));
        assert!(runs.lock().unwrap().is_empty());
        set(&b, at, 5, 0x8000_1000);
        assert!(runs.lock().unwrap().is_empty(), "not before memory is enabled");
        set(&b, at, 1, 3);
        let spans: Vec<_> = runs.lock().unwrap().iter().map(|r| (r.start, r.end)).collect();
        let low = 0x1a00_1000u64;
        assert_eq!(spans, vec![(low, low + 0x1000), (0x8000_1000, 0x8000_2000)], "the I/O BAR is under a page");
    }

    #[test]
    fn an_access_through_a_run_uses_the_hosts_lanes() {
        let (h, b, runs) = windowed();
        let at = PciAddr::dev(3);
        let f = Fake::new(PCI_INTA);
        b.attach(at, f.clone());
        set(&b, at, 5, 0x8000_1000);
        set(&b, at, 1, 2);
        let run = runs.lock().unwrap()[0].clone();
        assert_eq!(run.dev.read32(run.start + 0x84).data, 0x1234_5678);
        assert_eq!(h.lanes_log(), vec![(1, 0x84, 4)]);
        assert_eq!(*f.last.lock().unwrap(), Some((1, 0x84, 4)));
    }

    #[test]
    fn publishing_twice_reuses_the_table_entry() {
        let (_, b, runs) = windowed();
        let at = PciAddr::dev(3);
        b.attach(at, Fake::new(PCI_INTA));
        set(&b, at, 5, 0x8000_1000);
        set(&b, at, 1, 2);
        let first = runs.lock().unwrap()[0].dev.clone();
        b.publish();
        let again = runs.lock().unwrap()[0].dev.clone();
        assert!(Arc::ptr_eq(&first, &again));
    }

    #[test]
    fn a_config_write_republishes() {
        let (_, b, runs) = windowed();
        let at = PciAddr::dev(3);
        b.attach(at, Fake::new(PCI_INTA));
        set(&b, at, 5, 0x8000_1000);
        set(&b, at, 1, 2);
        assert_eq!(runs.lock().unwrap().len(), 2);
        set(&b, at, 1, 0);
        assert!(runs.lock().unwrap().is_empty(), "memory decoding off takes the runs away");
        set(&b, at, 1, 2);
        set(&b, at, 5, 0x8000_3000);
        assert_eq!(runs.lock().unwrap()[0].start, 0x1a00_3000, "moved with the BAR");
    }

    #[test]
    fn reset_returns_every_header_to_power_up_and_unpublishes_runs() {
        let (h, b, runs) = windowed();
        let (x, y) = (Fake::new(PCI_INTA), Fake::new(PCI_INTA));
        b.attach(PciAddr::dev(3), x.clone());
        b.attach(PciAddr::dev(4), y.clone());
        let fresh: Vec<u32> = (0..64).map(|i| x.cfg.read(i)).collect();
        for (at, base) in [(PciAddr::dev(3), 0x8000_1000), (PciAddr::dev(4), 0x8000_3000)] {
            set(&b, at, 5, base);
            set(&b, at, 1, 2);
        }
        assert_eq!(runs.lock().unwrap().len(), 4);
        x.irq.get().unwrap().set(true);
        b.reset();
        assert_eq!((0..64).map(|i| x.cfg.read(i)).collect::<Vec<_>>(), fresh);
        assert!((0..64).all(|i| y.cfg.read(i) == fresh[i]));
        assert!(runs.lock().unwrap().is_empty());
        assert!(b.target(Space::Mem, 0x8000_1084).is_none());
        assert_eq!(h.outputs(), vec![(12, true)], "reset drives no interrupt output");
        b.redrive();
        assert_eq!(h.outputs().iter().rev().find(|o| o.0 == 12), Some(&(12, true)), "and the pin is still held");
    }

    /// A bus with functions at devices 3 and 4, for the saved-state tests.
    fn pair() -> (Arc<TestHost>, PciBus, Arc<Fake>, Arc<Fake>) {
        let (h, b) = bus();
        let (x, y) = (Fake::new(PCI_INTA), Fake::new(PCI_INTA));
        b.attach(PciAddr::dev(3), x.clone());
        b.attach(PciAddr::dev(4), y.clone());
        (h, b, x, y)
    }

    /// A function that resets its own header leaves the bus's table stale
    /// until the bus is told.
    #[test]
    fn a_function_resetting_its_own_header_is_followed_after_a_refresh() {
        let (_, b, runs) = windowed();
        let at = PciAddr::dev(3);
        let f = Fake::new(PCI_INTA);
        b.attach(at, f.clone());
        set(&b, at, 5, 0x8000_1000);
        set(&b, at, 1, 2);
        assert!(b.target(Space::Mem, 0x8000_1084).is_some());
        f.cfg.reset();
        assert!(b.target(Space::Mem, 0x8000_1084).is_some(), "stale until told");
        b.refresh();
        assert!(b.target(Space::Mem, 0x8000_1084).is_none());
        assert!(runs.lock().unwrap().is_empty(), "and the runs are gone");
    }

    #[test]
    fn state_round_trips_headers_and_pin_levels() {
        let (_, src, x, _) = pair();
        set(&src, PciAddr::dev(3), 5, 0x8000_1000);
        set(&src, PciAddr::dev(3), 1, 2);
        x.irq.get().unwrap().set(true);
        let saved = src.save_state();

        let (_, dst, ..) = pair();
        dst.load_state(&saved).expect("load_state");
        assert_eq!(dst.save_state(), saved);
        let (_, bar, off) = dst.target(Space::Mem, 0x8000_1084).expect("BAR and memory enable restored");
        assert_eq!((bar, off), (1, 0x84));
    }

    #[test]
    fn load_refuses_a_device_the_bus_does_not_have() {
        let (_, src, ..) = pair();
        let saved = src.save_state();
        let (_, dst) = bus();
        dst.attach(PciAddr::dev(4), Fake::new(PCI_INTA));
        assert_eq!(dst.load_state(&saved).unwrap_err(), "pci: saved state has device 3, which is empty here");
    }

    #[test]
    fn load_refuses_a_device_the_state_lacks() {
        let (_, src) = bus();
        src.attach(PciAddr::dev(4), Fake::new(PCI_INTA));
        let saved = src.save_state();
        let (_, dst, ..) = pair();
        assert_eq!(dst.load_state(&saved).unwrap_err(), "pci: this bus has device 3, the saved state has none");
    }

    #[test]
    fn load_refuses_a_header_for_another_kind_of_function() {
        let (_, src) = bus();
        src.attach(PciAddr::dev(3), Fake::with_bars([Bar::Unused; 6], PCI_INTA));
        let mut saved = src.save_state();
        // Pretend the saved function had another identity.
        let regs = saved["devices"]["3.0"]["regs"].as_array_mut().unwrap();
        regs[0] = toml::Value::String("0x12345678".into());
        let (_, dst) = bus();
        dst.attach(PciAddr::dev(3), Fake::new(PCI_INTA));
        let e = dst.load_state(&saved).unwrap_err();
        assert!(e.starts_with("device 3: pci: saved header is for 0x12345678"), "{e}");
    }

    #[test]
    fn load_drives_nothing_and_redrive_drives_every_set_pin() {
        let (_, src, x, _) = pair();
        x.irq.get().unwrap().set(true);
        let saved = src.save_state();
        let (h, dst, ..) = pair();
        dst.load_state(&saved).unwrap();
        assert!(h.outputs().is_empty());
        dst.redrive();
        assert_eq!(h.outputs(), vec![(8, false), (12, true), (13, false)], "every routed output, in order");
    }
}
