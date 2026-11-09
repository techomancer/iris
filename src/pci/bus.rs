//! One PCI bus: the functions on it, the BAR regions they decode, and the
//! interrupt pins they raise.
//!
//! A host bridge (MACE, the Octane's Bridge) owns a [`PciBus`] and supplies, as
//! a [`PciHost`], only what is its own: the DMA a requester gets and where each
//! host IRQ goes. It pushes its CPU windows with [`PciBus::set_host_windows`]
//! and decides what byte order to apply to an access. The bus has no byte
//! order, no SCSI and no card names; a machine fits its functions with
//! [`PciBus::attach`] and the bus gives each its interrupt line and its way to
//! memory.
//!
//! Headers change only through the bus (`config_write`, `reset`, a snapshot
//! load), and each change updates the mappings, so a stale BAR mapping
//! cannot outlive the header that made it. A function never resets its own
//! header.

use std::io::Write;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Weak};

use super::decode::{BarMappingSink, BarRegion, CpuWindow, Decode};
use super::intx::{BoardIrq, Combiner, IntxRouting};
use super::{DevFn, PciDma, PciFunction, PciHost, Space};
use crate::devlog::LogModule;
use crate::irq::IrqLine;
use crate::snapshot::{get_field, hex_u32, hex_u64, toml_u32, toml_u64};
use crate::traits::Saveable;

/// Header type bit 7 (dword 3 bit 23): more functions follow at this device.
const HEADER_MULTIFUNCTION: u32 = 1 << 23;

/// A bus handle. Cheap to clone; the host holds the long-lived one.
#[derive(Clone)]
pub struct PciBus(Arc<Inner>);

/// What is in front of a bus: the host, for a root bus, or the bridge
/// function it sits behind.
enum Uplink {
    Root {
        up: Weak<dyn PciHost>,
        wiring: &'static IntxRouting,
        combiner: Arc<Combiner>,
        /// The next iris bus number to hand a secondary bus.
        next: AtomicU8,
    },
    /// Held weakly: the parent holds the bridge function, which holds this
    /// bus.
    Behind { parent: Weak<Inner>, bridge: DevFn },
}

struct Inner {
    number: u8,
    uplink: Uplink,
    decode: Decode,
}


impl PciBus {
    /// A bus numbered `number` (iris's own, from 0 per machine) wired as
    /// `wiring`, hosted by `up`. Build it inside `Arc::new_cyclic` so `up`
    /// can point at the host.
    pub fn new(number: u8, wiring: &'static IntxRouting, up: Weak<dyn PciHost>) -> Self {
        Self(Arc::new(Inner {
            number,
            decode: Decode::new(wiring.devices),
            uplink: Uplink::Root { combiner: Combiner::new(wiring, up.clone()), up, wiring, next: AtomicU8::new(number + 1) },
        }))
    }

    /// The bus behind the PCI-to-PCI bridge function that will sit at `bridge`
    /// on `parent`. It has no host and no outputs: its functions' DMA is the
    /// bridge's (so a function behind a bridge inherits the root slot the
    /// bridge sits in), and a pin of device `d` reaches the parent's pin
    /// `(pin + d) mod 4` of the bridge's device. Its regions join the
    /// parent's table while the bridge's header forwards them.
    pub fn secondary(parent: PciBus, bridge: DevFn) -> Self {
        Self(Arc::new(Inner {
            number: parent.next_number(),
            decode: Decode::new(32),
            uplink: Uplink::Behind { parent: Arc::downgrade(&parent.0), bridge },
        }))
    }

    /// This bus's regions and those behind its bridges, for the bus in front.
    pub(super) fn all_regions(&self) -> Vec<BarRegion> {
        self.0.decode.all_regions()
    }

    pub fn number(&self) -> u8 {
        self.0.number
    }

    /// The next iris bus number: the root bus counts them for its tree.
    fn next_number(&self) -> u8 {
        match &self.0.uplink {
            Uplink::Root { next, .. } => next.fetch_add(1, Ordering::Relaxed),
            Uplink::Behind { parent, .. } => parent.upgrade().map_or(0, |p| PciBus(p).next_number()),
        }
    }

    /// The DMA of a function on this bus: the host's for the root slot the
    /// function sits in, or behind, so a bridge's children are masters as
    /// the bridge's slot.
    fn dma_of(&self, requester: u8) -> Arc<dyn PciDma> {
        match &self.0.uplink {
            Uplink::Root { up, .. } => up.upgrade().expect("pci: the host is gone").dma_for(requester),
            Uplink::Behind { parent, bridge } => PciBus(parent.upgrade().expect("pci: the parent bus is gone")).dma_of(bridge.device),
        }
    }

    /// The line device `dev`'s pin (1 = INTA) drives, if it is routed: at
    /// the root through the wiring, behind a bridge by the bridge's own
    /// swizzle onto the pins of the bridge's device.
    fn pin_line(&self, dev: u8, pin: u8) -> Option<IrqLine> {
        match &self.0.uplink {
            Uplink::Root { combiner, .. } => combiner.pin_line(dev, pin),
            Uplink::Behind { parent, bridge } => {
                let parent = PciBus(parent.upgrade()?);
                parent.pin_line(bridge.device, (pin - 1 + dev) % 4 + 1)
            }
        }
    }

    /// Update the decode table and the BAR mappings here and on every bus
    /// in front of this one, which merges this bus's regions.
    fn update_mappings(&self, make: Option<&dyn Fn() -> Vec<CpuWindow>>) {
        self.0.decode.update_mappings(make);
        if let Uplink::Behind { parent, .. } = &self.0.uplink {
            if let Some(parent) = parent.upgrade() {
                PciBus(parent).update_mappings(None);
            }
        }
    }

    /// Fit a function: give it its way to memory and, if it has an interrupt
    /// pin, the line that pin drives, then tell it it is attached. Panics, as a
    /// wiring mistake in the machine, on a device number the wiring lacks, an
    /// address already taken, or a pin the wiring does not route.
    pub fn attach(&self, at: DevFn, f: Arc<dyn PciFunction>) {
        let i = &*self.0;
        assert!(at.device < i.decode.devices(), "pci: bus {} has no device {}", i.number, at.device);
        let pin = f.state().config().interrupt_pin();
        let line = (pin != 0).then(|| {
            self.pin_line(at.device, pin).unwrap_or_else(|| {
                panic!("pci: bus {} does not route INT{} of device {}", i.number, (b'A' + pin - 1) as char, at.device)
            })
        });
        f.state().connect_dma(self.dma_of(at.device));
        if let Some(line) = line {
            f.state().connect_irq(line);
        }
        f.attached();
        assert!(i.decode.insert(at, f), "pci: bus {} device {} function {} is taken", i.number, at.device, at.function);
        self.update_mappings(None);
    }

    pub fn function(&self, at: DevFn) -> Option<&Arc<dyn PciFunction>> {
        self.0.decode.function(at)
    }

    /// Whether any function sits at device number `device`.
    pub fn occupied(&self, device: u8) -> bool {
        (0..8).any(|function| self.function(DevFn { device, function }).is_some())
    }

    /// A configuration read of `width` bytes (1, 2 or 4) at byte `off`.
    /// None: nothing answers on a root bus, and the host decides what that
    /// looks like. On a bus behind a PCI-to-PCI bridge the cycle ends in
    /// master abort at the bridge: it reads all ones and the bridge's
    /// secondary status records the abort.
    ///
    /// Function 0 of a device with a function above it reports the
    /// multifunction bit of the header type, so a scan visits the rest.
    pub fn config_read(&self, at: DevFn, off: usize, width: usize) -> Option<u32> {
        let mask = u32::MAX >> (32 - 8 * width);
        let val = match self.function(at) {
            Some(f) => Some((self.header_dword(at, f, off / 4) >> ((off & 3) * 8)) & mask),
            None => self.master_abort().then_some(mask),
        };
        crate::dlog!(LogModule::Pci, "pci: bus {} {}.{} read {off:#04x}/{width} -> {}", self.0.number, at.device, at.function,
            val.map_or("no answer".to_string(), |v| format!("{v:#x}")));
        val
    }

    /// Dword `index` of `f`'s header as a guest reads it: header type bit 7
    /// is the bus's to say, since only the bus knows whether another
    /// function shares the device.
    fn header_dword(&self, at: DevFn, f: &Arc<dyn PciFunction>, index: usize) -> u32 {
        let multifunction = index == 3 && at.function == 0 && (1..8).any(|function| self.function(DevFn { function, ..at }).is_some());
        f.config_dword(index) | if multifunction { HEADER_MULTIFUNCTION } else { 0 }
    }

    /// A cycle that no function on this bus claims. Behind a bridge it
    /// ends in master abort: the bridge returns all ones for a read and
    /// drops a write, and sets received master abort in its secondary
    /// status. True when this bus handled it so. A root bus leaves the
    /// outcome to its host.
    fn master_abort(&self) -> bool {
        let Uplink::Behind { parent, bridge } = &self.0.uplink else { return false };
        if let Some(parent) = parent.upgrade() {
            if let Some(f) = PciBus(parent).function(*bridge) {
                f.state().config().record_master_abort();
            }
        }
        true
    }

    /// A configuration write of `width` bytes at byte `off`: only those bytes
    /// change, and the header decides which of their bits stick. A function
    /// may take a whole-dword write itself ([`PciFunction::config_write`]). False if
    /// nothing answers on a root bus; behind a bridge the write is discarded by
    /// master abort, as a read of nothing is (see [`PciBus::config_read`]), and
    /// counts as answered. The mappings are updated when the command register,
    /// a BAR, or a bridge's bus numbers or windows changed, the only registers
    /// they are built from.
    pub fn config_write(&self, at: DevFn, off: usize, width: usize, val: u32) -> bool {
        let f = self.function(at);
        crate::dlog!(LogModule::Pci, "pci: bus {} {}.{} write {off:#04x}/{width} {val:#x}{}", self.0.number, at.device, at.function,
            if f.is_some() { "" } else { ", no answer" });
        let Some(f) = f else { return self.master_abort() };
        if width == 4 && f.config_write(off / 4, val) {
            return true;
        }
        let (cfg, reg, shift) = (f.state().config(), off / 4, (off & 3) * 8);
        let mask = (u32::MAX >> (32 - 8 * width)) << shift;
        let old = cfg.read(reg);
        cfg.write_masked(reg, val << shift, mask);
        let new = cfg.read(reg);
        if new != old {
            f.config_written(reg, old, new);
            // The command register, the BARs and, in a bridge's header, its
            // bus numbers and windows: the registers the table is built from.
            if matches!(reg, 1 | 4..=9) {
                self.update_mappings(None);
            }
        }
        true
    }

    /// The function, BAR and BAR offset that PCI address `addr` reaches in
    /// `space`: the host's slow path. A borrow, no clone.
    pub fn decode(&self, space: Space, addr: u32) -> Option<(&dyn PciFunction, usize, u32)> {
        self.0.decode.decode(space, addr)
    }

    /// The host's CPU windows changed (MACE's window registers, the Bridge's
    /// Device(x)): `make` returns them, and the runs are published again.
    /// `make` runs inside the publish mutex, so two CPUs writing registers
    /// cannot publish out of order. The host must not hold a lock `make`
    /// takes when it calls this; the order is publish mutex, then the
    /// host's lock.
    pub fn set_host_windows(&self, make: &dyn Fn() -> Vec<CpuWindow>) {
        self.0.decode.update_mappings(Some(make));
    }

    /// Where the runs of the CPU windows that one BAR decodes whole go. The
    /// machine maps them on the physical bus over the host's windows.
    pub fn on_bar_mappings(&self, sink: BarMappingSink) {
        self.0.decode.set_mapping_sink(sink);
    }

    /// The line board IRQ `w` drives into the host.
    pub fn board_irq(&self, w: BoardIrq) -> IrqLine {
        match &self.0.uplink {
            Uplink::Root { combiner, .. } => combiner.board_irq_line(w),
            Uplink::Behind { .. } => panic!("pci: a bus behind a bridge has no board IRQs"),
        }
    }

    /// RST#: each function in (device, function) order has its header
    /// reset and then [`PciFunction::reset`] called, and the mappings are
    /// updated. Pin levels are left alone: a function that was not reset
    /// keeps its line, as its hardware would, and a function that was drops
    /// its own.
    pub fn reset(&self) {
        self.reset_where(|_| true);
    }

    /// RST# to the functions at device number `device` alone, as `reset`
    /// does for the whole bus. For a host with a reset line per device.
    pub fn reset_device(&self, device: u8) {
        self.reset_where(|at| at.device == device);
    }

    fn reset_where(&self, hit: impl Fn(DevFn) -> bool) {
        for (_, f) in self.0.decode.functions().filter(|&(at, _)| hit(at)) {
            f.state().config().reset();
            f.reset();
        }
        self.update_mappings(None);
    }

    /// Tell the host every output's level again, whatever it already has.
    /// A restore ends with this, after the host and the interrupt
    /// controller behind it have loaded.
    pub fn redrive(&self) {
        if let Uplink::Root { combiner, .. } = &self.0.uplink {
            combiner.redrive();
        }
    }

    /// Each output's (held, fell, force), for a test to compare.
    #[cfg(test)]
    pub(crate) fn delivery_flags(&self) -> Vec<(bool, bool, bool)> {
        match &self.0.uplink {
            Uplink::Root { combiner, .. } => combiner.flags(),
            Uplink::Behind { .. } => Vec::new(),
        }
    }

    /// Latch every output's `held` and `fell`, as an interrupted delivery
    /// would leave them.
    #[cfg(test)]
    pub(crate) fn latch_delivery_flags(&self) {
        if let Uplink::Root { combiner, .. } = &self.0.uplink {
            combiner.out_force_flags();
        }
    }

    /// One line per function: address, identity, command, assigned BARs and
    /// where its interrupt pin goes.
    pub fn describe(&self, w: &mut dyn Write) {
        for (at, f) in self.0.decode.functions() {
            let c = f.state().config();
            let id = c.read(0);
            let bars: Vec<String> = (0..6).map(|i| c.bar_base(i)).filter(|&b| b != 0).map(|b| format!("{b:#x}")).collect();
            let pin = c.interrupt_pin();
            let irq = match pin {
                0 => "no interrupt".to_string(),
                p => {
                    let name = (b'A' + p - 1) as char;
                    match &self.0.uplink {
                        Uplink::Root { combiner, .. } => match combiner.output(at.device, p) {
                            Some(o) => format!("INT{name} to output {o}"),
                            None => format!("INT{name} unrouted"),
                        },
                        Uplink::Behind { .. } if self.pin_line(at.device, p).is_some() => format!("INT{name} through the bridge"),
                        Uplink::Behind { .. } => format!("INT{name} unrouted"),
                    }
                }
            };
            let _ = writeln!(w, "  {}.{}: {:04x}:{:04x}, command {:#06x}, BARs [{}], {irq}",
                at.device, at.function, id & 0xffff, id >> 16, c.command(), bars.join(", "));
            if let Some(sec) = f.secondary() {
                let _ = writeln!(w, "    bus {} behind {}.{}:", sec.number(), at.device, at.function);
                sec.describe(w);
            }
        }
    }

    /// The bus a guest's bus number reaches: bus 0 is this (root) bus, and
    /// any other number is found by the bridges' secondary and subordinate
    /// bus numbers as firmware has programmed them, however deep. None if
    /// no bridge's range holds it. A scan, for configuration cycles.
    pub fn find_bus(&self, guest_bus: u8) -> Option<PciBus> {
        if guest_bus == 0 {
            return Some(self.clone());
        }
        self.find_behind(guest_bus)
    }

    /// The bus a type 1 configuration cycle for `guest_bus` on this bus
    /// reaches. Type 0 functions ignore a type 1 cycle; a PCI-to-PCI bridge
    /// claims it when the number is in its secondary to subordinate range.
    /// None when no bridge does. Unlike [`PciBus::find_bus`], bus 0 is not this
    /// bus: it too reaches only a bridge whose range holds it.
    pub fn type1_target(&self, guest_bus: u8) -> Option<PciBus> {
        self.find_behind(guest_bus)
    }

    /// The bus numbered `guest_bus` among the buses behind this one's
    /// bridges (number 0 is never behind a bridge).
    fn find_behind(&self, guest_bus: u8) -> Option<PciBus> {
        self.0.decode.functions().find_map(|(_, f)| {
            let sec = f.secondary()?;
            let numbers = f.state().config().read(6);
            let (first, last) = ((numbers >> 8 & 0xff) as u8, (numbers >> 16 & 0xff) as u8);
            match guest_bus {
                n if n == first => Some(sec.clone()),
                n if first < n && n <= last => sec.find_behind(n),
                _ => None,
            }
        })
    }
}

impl PciBus {
    /// The `pci` monitor command both hosts share: `status` describes the
    /// functions, `config <device>` dumps one's header.
    pub fn command(&self, args: &[&str], w: &mut dyn Write) -> Result<(), String> {
        match args.first().copied().unwrap_or("status") {
            "status" => self.describe(w),
            "config" => {
                let device: u8 = args.get(1).and_then(|v| v.parse().ok()).ok_or("Usage: pci config <slot>")?;
                let f = self.function(DevFn { device, function: 0 }).ok_or(format!("pci: nothing in slot {device}"))?;
                for row in 0..16 {
                    let v: Vec<String> = (0..4).map(|k| format!("{:08x}", f.state().config().read(row * 4 + k))).collect();
                    let _ = writeln!(w, "  {:02x}: {}", row * 16, v.join(" "));
                }
            }
            _ => return Err("Usage: pci status | config <slot>".to_string()),
        }
        Ok(())
    }
}

/// Every function's header, and the pin and board IRQ levels. A restore
/// replaces all of it and forgets what the host was last told; the machine then
/// calls [`PciBus::redrive`]. The functions behind the headers save their own
/// state.
impl Saveable for PciBus {
    fn save_state(&self) -> toml::Value {
        let i = &*self.0;
        let mut t = toml::map::Map::new();
        let mut funcs = toml::map::Map::new();
        for (at, f) in i.decode.functions() {
            funcs.insert(format!("{}.{}", at.device, at.function), f.state().config().save_state());
        }
        t.insert("functions".into(), toml::Value::Table(funcs));
        // The buses behind bridges save their headers here too.
        let mut behind = toml::map::Map::new();
        for (at, f) in i.decode.functions() {
            if let Some(sec) = f.secondary() {
                behind.insert(format!("{}.{}", at.device, at.function), sec.save_state());
            }
        }
        if !behind.is_empty() {
            t.insert("behind".into(), toml::Value::Table(behind));
        }
        if let Uplink::Root { combiner, .. } = &i.uplink {
            let (lo, hi, wires) = combiner.save();
            t.insert("pins_lo".into(), hex_u64(lo));
            t.insert("pins_hi".into(), hex_u64(hi));
            t.insert("wires".into(), hex_u32(wires));
        }
        toml::Value::Table(t)
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        let i = &*self.0;
        let saved = get_field(v, "functions").and_then(|f| f.as_table()).ok_or("pci: saved state has no functions")?;
        let ours: Vec<String> = i.decode.functions().map(|(at, _)| format!("{}.{}", at.device, at.function)).collect();
        let same = ours.len() == saved.len() && ours.iter().all(|k| saved.contains_key(k));
        if !same {
            return Err(format!("pci: saved state has functions [{}], this bus [{}]",
                saved.keys().cloned().collect::<Vec<_>>().join(", "), ours.join(", ")));
        }
        for (at, f) in i.decode.functions() {
            let key = format!("{}.{}", at.device, at.function);
            f.state().config().load_state(&saved[&key])?;
            if let Some(sec) = f.secondary() {
                let state = get_field(v, "behind").and_then(|b| get_field(b, &key)).ok_or_else(|| format!("pci: saved state has no bus behind {key}"))?;
                sec.load_state(state)?;
            }
        }
        if let Uplink::Root { combiner, .. } = &i.uplink {
            let word = |k: &str| get_field(v, k).and_then(toml_u64).unwrap_or(0);
            combiner.load((word("pins_lo"), word("pins_hi"), get_field(v, "wires").and_then(toml_u32).unwrap_or(0)));
        }
        self.update_mappings(None);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pci::testing::{TestCard, TestHost};
    use crate::pci::{Bar, PciConfig, PCI_COMMAND_MEMORY, PCI_INTA};
    use std::sync::Mutex;

    // Device 1: INTA and INTB on outputs 8 and 9; device 2 INTA shares 8;
    // device 3 INTA..INTD on 10 to 13. One board IRQ, on output 14.
    static W: IntxRouting = IntxRouting {
        devices: 8,
        irqs: 16,
        intx: &[
            (1, [Some(8), Some(9), None, None]),
            (2, [Some(8), None, None, None]),
            (3, [Some(10), Some(11), Some(12), Some(13)]),
        ],
        board_irqs: &[14],
    };

    fn at(device: u8, function: u8) -> DevFn {
        DevFn { device, function }
    }

    fn bus() -> (Arc<TestHost>, PciBus) {
        let h = Arc::new(TestHost::default());
        let up: Weak<dyn PciHost> = Arc::downgrade(&(h.clone() as Arc<dyn PciHost>));
        (h, PciBus::new(0, &W, up))
    }

    fn bars() -> [Bar; 6] {
        [Bar::Io(0x100), Bar::Mem(0x1000), Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused]
    }

    #[test]
    fn attach_connects_the_right_line_and_dma() {
        let (h, b) = bus();
        let c = TestCard::new(bars(), PCI_INTA);
        b.attach(at(3, 0), c.clone());
        assert_eq!(c.state().irq().unwrap().line(), 3 * 4);
        c.state().irq().unwrap().set(true);
        assert_eq!(h.irqs(), vec![(10, true)]);
        let mut buf = [0u8; 2];
        assert!(c.state().dma().unwrap().dma_read(0, &mut buf).is_ok());
        assert_eq!(buf, [3, 3], "the DMA of requester device 3");
        let none = TestCard::new(bars(), 0);
        b.attach(at(4, 0), none.clone());
        assert!(none.state().irq().is_none(), "no pin, no line");
        assert!(none.state().dma().is_some());
    }

    #[test]
    #[should_panic(expected = "is taken")]
    fn duplicate_address_panics() {
        let (_h, b) = bus();
        b.attach(at(1, 0), TestCard::new(bars(), PCI_INTA));
        b.attach(at(1, 0), TestCard::new(bars(), PCI_INTA));
    }

    #[test]
    #[should_panic(expected = "has no device 8")]
    fn device_out_of_range_panics() {
        let (_h, b) = bus();
        b.attach(at(8, 0), TestCard::new(bars(), 0));
    }

    #[test]
    #[should_panic(expected = "does not route INTC of device 1")]
    fn pin_without_a_route_panics() {
        bus().1.attach(at(1, 0), TestCard::new(bars(), PCI_INTA + 2));
    }

    #[test]
    fn multi_function_shares_the_device_pins() {
        let (h, b) = bus();
        let (f0, f1) = (TestCard::new(bars(), PCI_INTA), TestCard::new(bars(), PCI_INTA));
        b.attach(at(3, 0), f0.clone());
        b.attach(at(3, 1), f1.clone());
        assert_eq!(f0.state().irq().unwrap().line(), f1.state().irq().unwrap().line());
        f0.state().irq().unwrap().set(true);
        f1.state().irq().unwrap().set(true);
        f0.state().irq().unwrap().set(false);
        assert_eq!(h.irqs(), vec![(10, true), (10, false)], "one pin, one output");
        assert!(b.occupied(3) && !b.occupied(4));
    }

    /// A function whose dword 0x1e is a window into the chip: a whole-dword
    /// write goes to the chip every time, even when it repeats a value, and
    /// the header stores nothing; a narrower one takes the header's path.
    #[test]
    fn a_function_takes_a_whole_dword_configuration_write_itself() {
        struct Window(crate::pci::PciFunctionState, Mutex<Vec<u32>>);
        impl PciFunction for Window {
            fn state(&self) -> &crate::pci::PciFunctionState {
                &self.0
            }
            fn bar_read(&self, _: usize, _: u32, _: usize) -> u32 {
                0
            }
            fn bar_write(&self, _: usize, _: u32, _: u32, _: usize) {}
            fn config_write(&self, reg: usize, val: u32) -> bool {
                (reg == 0x1e).then(|| self.1.lock().unwrap().push(val)).is_some()
            }
        }
        let (_h, b) = bus();
        let w = Arc::new(Window(crate::pci::PciFunctionState::new(PciConfig::new(1, 2, 0, 1, bars(), PCI_INTA)), Mutex::new(Vec::new())));
        b.attach(at(1, 0), w.clone());
        assert!(b.config_write(at(1, 0), 0x78, 4, 0x1234_5678));
        assert!(b.config_write(at(1, 0), 0x78, 4, 0x1234_5678));
        assert_eq!(*w.1.lock().unwrap(), [0x1234_5678, 0x1234_5678], "a repeated value reaches the chip again");
        assert_eq!(b.config_read(at(1, 0), 0x78, 4), Some(0), "the header stored nothing");
        assert!(b.config_write(at(1, 0), 0x78, 2, 0xffff));
        assert_eq!(w.1.lock().unwrap().len(), 2, "a halfword write is the header's");
        // Another dword is the header's too.
        assert!(b.config_write(at(1, 0), 0x3c, 4, 0x0000_00ab));
        assert_eq!(b.config_read(at(1, 0), 0x3c, 1), Some(0xab));
        assert_eq!(w.1.lock().unwrap().len(), 2);
    }

    #[test]
    fn config_size_probe_and_masks() {
        let (_h, b) = bus();
        b.attach(at(1, 0), TestCard::new(bars(), PCI_INTA));
        assert_eq!(b.config_read(at(1, 0), 0, 4), Some(0x5678_1234));
        assert_eq!(b.config_read(at(1, 0), 0, 2), Some(0x1234), "vendor");
        assert_eq!(b.config_read(at(1, 0), 2, 2), Some(0x5678), "device");
        assert_eq!(b.config_read(at(1, 0), 3, 1), Some(0x56));
        assert!(b.config_write(at(1, 0), 0x10, 4, 0xffff_ffff));
        assert_eq!(b.config_read(at(1, 0), 0x10, 4), Some(0xffff_ff01), "256 bytes of I/O");
        assert!(b.config_write(at(1, 0), 0x14, 4, 0xffff_ffff));
        assert_eq!(b.config_read(at(1, 0), 0x14, 4), Some(0xffff_f000), "4 KiB of memory");
    }

    /// The read-modify-write every host did before the bus did it.
    #[test]
    fn sub_dword_writes_equal_the_old_read_modify_write() {
        for width in [1usize, 2, 4] {
            for off in (0..256).step_by(width) {
                for val in [0xffff_ffffu32, 0xa5c3_1e78, 0] {
                    let (_h, b) = bus();
                    let (new, old) = (TestCard::new(bars(), PCI_INTA), PciConfig::new(0x1234, 0x5678, 0, 1, bars(), PCI_INTA));
                    b.attach(at(1, 0), new.clone());
                    // Start both from the same non-trivial header.
                    for r in 1..16 {
                        new.state().config().write(r, 0x1111_1111 * r as u32);
                        old.write(r, 0x1111_1111 * r as u32);
                    }
                    assert!(b.config_write(at(1, 0), off, width, val));
                    let (i, shift) = (off / 4, 8 * (off & 3));
                    let mask = (u64::MAX >> (64 - 8 * width)) << shift;
                    old.write(i, ((old.read(i) as u64 & !mask) | ((val as u64) << shift & mask)) as u32);
                    for r in 0..64 {
                        assert_eq!(new.state().config().read(r), old.read(r), "width {width} off {off:#x} val {val:#x} reg {r}");
                    }
                }
            }
        }
    }

    #[test]
    fn an_absent_function_reads_none() {
        let (_h, b) = bus();
        b.attach(at(1, 0), TestCard::new(bars(), 0));
        assert_eq!(b.config_read(at(2, 0), 0, 4), None);
        assert_eq!(b.config_read(at(1, 1), 0, 4), None, "function 1 of a single-function device");
        assert!(!b.config_write(at(2, 0), 0, 4, 0));
    }

    #[test]
    fn a_config_write_that_moves_a_bar_updates_the_mappings() {
        let (_h, b) = bus();
        b.attach(at(1, 0), TestCard::new(bars(), 0));
        let runs = Arc::new(Mutex::new(Vec::new()));
        let sink = runs.clone();
        b.on_bar_mappings(Box::new(move |r| *sink.lock().unwrap() = r));
        b.set_host_windows(&|| vec![CpuWindow { space: Space::Mem, cpu: 0x1a00_0000, pci: 0x8000_0000, len: 0x100_0000, mask: 0xffff_ffff }]);
        let cpus = || runs.lock().unwrap().iter().map(|r: &crate::pci::BarMapping| r.cpu).collect::<Vec<_>>();
        assert!(cpus().is_empty());
        b.config_write(at(1, 0), 0x14, 4, 0x8000_2000);
        b.config_write(at(1, 0), 0x04, 2, PCI_COMMAND_MEMORY);
        assert_eq!(cpus(), vec![0x1a00_2000]);
        b.config_write(at(1, 0), 0x14, 4, 0x8000_5000);
        assert_eq!(cpus(), vec![0x1a00_5000], "the old run is gone");
        assert!(b.decode(Space::Mem, 0x8000_2000).is_none());
        assert_eq!(b.decode(Space::Mem, 0x8000_5010).map(|(_, bar, off)| (bar, off)), Some((1, 0x10)));
    }

    /// Only the command register and the BARs shape the decode table, so a
    /// write to any other register, or one that changes nothing, publishes
    /// nothing.
    #[test]
    fn only_a_command_or_bar_change_updates_the_mappings() {
        let (_h, b) = bus();
        b.attach(at(1, 0), TestCard::new(bars(), 0));
        let count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let n = count.clone();
        b.on_bar_mappings(Box::new(move |_| {
            n.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }));
        let published = || count.load(std::sync::atomic::Ordering::SeqCst);
        let before = published();
        for line in 0..=255u32 {
            b.config_write(at(1, 0), 0x3c, 1, line); // interrupt line
            b.config_write(at(1, 0), 0x0c, 1, line); // cache line size
        }
        assert_eq!(published(), before);
        b.config_write(at(1, 0), 0x04, 2, PCI_COMMAND_MEMORY);
        assert_eq!(published(), before + 1, "the command register");
        b.config_write(at(1, 0), 0x04, 2, PCI_COMMAND_MEMORY);
        assert_eq!(published(), before + 1, "the same value again");
        b.config_write(at(1, 0), 0x14, 4, 0x8000_2000);
        assert_eq!(published(), before + 2, "a BAR");
    }

    /// RST# resets the header and then the function, device by device.
    #[test]
    fn reset_orders_header_then_function_by_device_and_function() {
        let (_h, b) = bus();
        let log = Arc::new(Mutex::new(Vec::new()));
        for (name, d, f) in [("c", 3, 0), ("a", 1, 1), ("b", 1, 0)] {
            let c = TestCard::logged(name, bars(), 0, &log);
            b.attach(at(d, f), c);
            b.config_write(at(d, f), 4, 2, PCI_COMMAND_MEMORY);
        }
        b.reset();
        assert_eq!(*log.lock().unwrap(), vec!["b cmd=0x0", "a cmd=0x0", "c cmd=0x0"], "header already reset when reset() runs");
    }

    #[test]
    fn reset_device_resets_the_functions_at_one_device_number_only() {
        let (_h, b) = bus();
        let log = Arc::new(Mutex::new(Vec::new()));
        for (name, d, f) in [("a0", 1, 0), ("a1", 1, 1), ("b", 2, 0), ("c", 3, 0)] {
            b.attach(at(d, f), TestCard::logged(name, bars(), 0, &log));
            b.config_write(at(d, f), 4, 2, PCI_COMMAND_MEMORY);
        }
        b.reset_device(1);
        assert_eq!(*log.lock().unwrap(), vec!["a0 cmd=0x0", "a1 cmd=0x0"]);
        assert_eq!(b.config_read(at(2, 0), 4, 2), Some(PCI_COMMAND_MEMORY), "device 2 keeps its header");
        assert_eq!(b.config_read(at(1, 1), 4, 2), Some(0));
    }

    #[test]
    fn reset_leaves_pin_levels_until_the_card_lowers_its_line() {
        let (h, b) = bus();
        let c = TestCard::new(bars(), PCI_INTA);
        b.attach(at(3, 0), c.clone());
        c.state().irq().unwrap().set(true);
        b.reset();
        assert_eq!(h.irqs(), vec![(10, true)], "the bus did not lower it");
        c.state().irq().unwrap().set(false);
        assert_eq!(h.irqs(), vec![(10, true), (10, false)]);
    }

    #[test]
    fn redrive_after_load_state_resends_a_high_output() {
        let (h, b) = bus();
        let c = TestCard::new(bars(), PCI_INTA);
        b.attach(at(3, 0), c.clone());
        c.state().irq().unwrap().set(true);
        let saved = b.save_state();
        b.load_state(&saved).unwrap();
        assert_eq!(h.irqs().len(), 1, "load tells the host nothing");
        b.redrive();
        let heard: Vec<_> = h.irqs().into_iter().filter(|o| o.0 == 10).collect();
        assert_eq!(heard, vec![(10, true), (10, true)], "the host hears the saved level again");
    }

    #[test]
    fn bus_state_round_trips() {
        let make = || {
            let (h, b) = bus();
            let c = TestCard::new(bars(), PCI_INTA);
            b.attach(at(3, 0), c.clone());
            (h, b, c)
        };
        let (_ha, a, ca) = make();
        a.config_write(at(3, 0), 0x14, 4, 0x8000_1000);
        ca.state().irq().unwrap().set(true);
        a.board_irq(BoardIrq(0)).set(true);
        let v = a.save_state();
        let (_hb, b, cb) = make();
        b.load_state(&v).unwrap();
        assert_eq!(b.save_state(), v);
        assert_eq!(cb.state().config().bar_base(1), 0x8000_1000);
    }

    #[test]
    fn load_state_replaces_headers_pins_and_combiner_flags_outright() {
        let make = || {
            let (h, b) = bus();
            let c = TestCard::new(bars(), PCI_INTA);
            b.attach(at(3, 0), c.clone());
            (h, b, c)
        };
        let (_hs, src, cs) = make();
        src.config_write(at(3, 0), 0x14, 4, 0x8000_1000);
        cs.state().irq().unwrap().set(true);
        let v = src.save_state();
        let (_hf, fresh, _) = make();
        fresh.load_state(&v).unwrap();
        // A bus that has been used: other header, other pins, a latched fall.
        let (_hd, dirty, cd) = make();
        dirty.config_write(at(3, 0), 0x14, 4, 0x9000_0000);
        dirty.config_write(at(3, 0), 0x04, 2, PCI_COMMAND_MEMORY);
        cd.state().irq().unwrap().set(true);
        cd.state().irq().unwrap().set(false);
        dirty.board_irq(BoardIrq(0)).set(true);
        dirty.latch_delivery_flags();
        dirty.load_state(&v).unwrap();
        assert_eq!(dirty.save_state(), fresh.save_state());
        assert_eq!(dirty.delivery_flags(), fresh.delivery_flags());
        assert_eq!(dirty.config_read(at(3, 0), 4, 2), fresh.config_read(at(3, 0), 4, 2));
    }

    #[test]
    fn a_mismatched_function_set_is_refused_by_load_state() {
        let (_h, a) = bus();
        a.attach(at(1, 0), TestCard::new(bars(), 0));
        a.attach(at(2, 0), TestCard::new(bars(), 0));
        let (_h, b) = bus();
        b.attach(at(1, 0), TestCard::new(bars(), 0));
        let err = b.load_state(&a.save_state()).unwrap_err();
        assert!(err.contains("1.0, 2.0") && err.contains("[1.0]"), "{err}");
    }

    #[test]
    fn describe_prints_one_line_per_function() {
        let (_h, b) = bus();
        b.attach(at(1, 0), TestCard::new(bars(), PCI_INTA));
        b.attach(at(3, 0), TestCard::new(bars(), 0));
        b.config_write(at(1, 0), 0x14, 4, 0x8000_1000);
        let mut out = Vec::new();
        b.describe(&mut out);
        let out = String::from_utf8(out).unwrap();
        assert_eq!(out.lines().count(), 2);
        assert!(out.lines().next().unwrap().contains("1.0: 1234:5678") && out.contains("BARs [0x80001000]") && out.contains("INTA to output 8"), "{out}");
        assert!(out.lines().nth(1).unwrap().contains("no interrupt"));
    }

    #[test]
    fn function_0_reports_multifunction_only_while_a_higher_function_is_fitted() {
        let (_host, bus) = bus();
        let header = |function| bus.config_read(at(2, function), 0x0e, 1).unwrap();
        let card = || TestCard::new(bars(), 0);
        bus.attach(at(2, 0), card());
        assert_eq!(header(0) & 0x80, 0, "a single-function device");
        bus.attach(at(2, 1), card());
        assert_eq!(header(0) & 0x80, 0x80, "function 0 tells a scan to visit function 1");
        assert_eq!(header(1) & 0x80, 0, "the bit is function 0's alone");
        assert_eq!(bus.config_read(at(2, 0), 0x0c, 4).unwrap() >> 16 & 0xff, 0x80, "in the dword too");
        bus.attach(at(3, 0), card());
        assert_eq!(bus.config_read(at(3, 0), 0x0e, 1), Some(0), "another device is unaffected");
    }

    #[test]
    fn behind_a_bridge_an_unclaimed_config_cycle_reads_ones_and_a_write_is_dropped() {
        use crate::pci::PciToPciBridge;
        let (_host, bus) = bus();
        let bridge = PciToPciBridge::new(&bus, at(3, 0), 0x1234, 0x5678);
        bus.attach(at(3, 0), bridge.clone());
        let sec = bridge.secondary();
        let status = || bus.config_read(at(3, 0), 0x1e, 2).unwrap();
        assert_eq!(status() & 0x2000, 0);
        assert_eq!(sec.config_read(at(5, 0), 0, 4), Some(0xffff_ffff));
        assert_eq!(sec.config_read(at(5, 0), 2, 2), Some(0xffff));
        assert_eq!(sec.config_read(at(5, 0), 1, 1), Some(0xff));
        assert_eq!(status() & 0x2000, 0x2000, "received master abort");
        assert!(sec.config_write(at(5, 0), 4, 2, 0x2), "dropped, not unanswered");
        // Write one to clear.
        bus.config_write(at(3, 0), 0x1c, 4, 0x2000_0000);
        assert_eq!(status() & 0x2000, 0);
        // The root bus leaves an absent function to its host.
        assert_eq!(bus.config_read(at(5, 0), 0, 4), None);
        assert!(!bus.config_write(at(5, 0), 4, 2, 0x2));
        assert_eq!(status() & 0x2000, 0, "the root bus's miss is not the bridge's");
    }
}
