//! Interrupt pins and board wires, routed to a host's outputs.
//!
//! Every pin of every device number is a line (`device * 4 + pin - 1`), and
//! the bus description's wires follow them. The table keeps each line's level
//! and drives an output with the OR of everything routed to it, so on a
//! shared output one source cannot clear another's interrupt. The routes and
//! the per-output masks are worked out once, from the description, when the
//! table is built; a level change is a bit flip and two mask tests.

use std::sync::{Arc, Mutex};

use super::host::PciHost;
use super::slots::BusDesc;
use crate::irq::{InterruptController, IrqLine};
use crate::snapshot::{get_field, load_u32_slice, u32_slice_to_toml};

/// Lines 0..PIN_LINES are pins; wire `n` is line `PIN_LINES + n`.
const PIN_LINES: usize = 32 * 4;
const MAX_WIRES: usize = 32;
const NO_OUTPUT: u8 = u8::MAX;

#[derive(Default)]
struct Levels {
    pins: u128,
    wires: u32,
}

pub struct IntxTable {
    desc: &'static BusDesc,
    host: Arc<dyn PciHost>,
    /// Host output each line reaches, `NO_OUTPUT` for none.
    route: [u8; PIN_LINES + MAX_WIRES],
    /// Per output, the pin lines and wires routed to it.
    masks: Box<[(u128, u32)]>,
    /// Held while the host is driven, so two sources of one output cannot
    /// reach it out of order.
    levels: Mutex<Levels>,
}

impl IntxTable {
    pub fn new(desc: &'static BusDesc, host: Arc<dyn PciHost>) -> Arc<Self> {
        assert!(desc.devices <= 32, "pci: bus {:?} has more than 32 devices", desc.id);
        assert!(desc.wires.len() <= MAX_WIRES, "pci: bus {:?} has more than {MAX_WIRES} wires", desc.id);
        let mut route = [NO_OUTPUT; PIN_LINES + MAX_WIRES];
        let mut masks = vec![(0u128, 0u32); desc.outputs as usize];
        let mut connect = |line: usize, output: u8| {
            assert!(output < desc.outputs, "pci: bus {:?} routes line {line} to output {output} of {}", desc.id, desc.outputs);
            route[line] = output;
            if line < PIN_LINES {
                masks[output as usize].0 |= 1u128 << line;
            } else {
                masks[output as usize].1 |= 1u32 << (line - PIN_LINES);
            }
        };
        for s in desc.slots {
            assert!(s.device < desc.devices, "pci: bus {:?} describes device {} of {}", desc.id, s.device, desc.devices);
            for (p, r) in s.routes.iter().enumerate() {
                if let Some(o) = *r {
                    connect(s.device as usize * 4 + p, o);
                }
            }
        }
        for (n, w) in desc.wires.iter().enumerate() {
            connect(PIN_LINES + n, w.output);
        }
        Arc::new(Self { desc, host, route, masks: masks.into(), levels: Mutex::new(Levels::default()) })
    }

    /// The line a function's pin (1 = INTA .. 4 = INTD) drives. A pin the
    /// description does not route gets a line that leads nowhere, so the
    /// function can drive it without knowing.
    pub fn pin_line(self: &Arc<Self>, device: u8, pin: u8) -> IrqLine {
        assert!((1..=4).contains(&pin), "pci: no interrupt pin {pin}");
        let line = device as usize * 4 + pin as usize - 1;
        match self.route.get(line) {
            Some(&o) if o != NO_OUTPUT => IrqLine::new(self.clone(), line as u32),
            _ => IrqLine::unconnected(),
        }
    }

    /// A board wire from the description, by name.
    pub fn wire_line(self: &Arc<Self>, name: &str) -> IrqLine {
        let n = self.desc.wires.iter().position(|w| w.name == name)
            .unwrap_or_else(|| panic!("pci: bus {:?} has no wire {name:?}", self.desc.id));
        IrqLine::new(self.clone(), (PIN_LINES + n) as u32)
    }

    fn any(&self, l: &Levels, output: u8) -> bool {
        let m = self.masks[output as usize];
        (l.pins & m.0 != 0) | (l.wires & m.1 != 0)
    }

    /// Drive every routed output to the level its sources now add up to.
    pub fn redrive(&self) {
        let l = self.levels.lock().unwrap();
        for (o, m) in self.masks.iter().enumerate() {
            if *m != (0, 0) {
                self.host.set_output(o as u8, self.any(&l, o as u8));
            }
        }
    }

    pub fn save_state(&self) -> toml::Value {
        let l = self.levels.lock().unwrap();
        let w = [l.pins as u32, (l.pins >> 32) as u32, (l.pins >> 64) as u32, (l.pins >> 96) as u32, l.wires];
        let mut t = toml::map::Map::new();
        t.insert("levels".into(), u32_slice_to_toml(&w));
        toml::Value::Table(t)
    }

    /// Restore the levels. Nothing is driven: the host's own state is
    /// loaded separately, and [`Self::redrive`] puts the outputs right.
    pub fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        let Some(toml::Value::Array(a)) = get_field(v, "levels") else {
            return Err("pci: saved interrupt levels are missing".into());
        };
        if a.len() != 5 {
            return Err(format!("pci: saved interrupt levels have {} words, expected 5", a.len()));
        }
        let mut w = [0u32; 5];
        load_u32_slice(get_field(v, "levels").unwrap(), &mut w);
        let mut l = self.levels.lock().unwrap();
        l.pins = w[0] as u128 | (w[1] as u128) << 32 | (w[2] as u128) << 64 | (w[3] as u128) << 96;
        l.wires = w[4];
        Ok(())
    }
}

impl InterruptController for IntxTable {
    fn set_line(&self, line: u32, level: bool) {
        let line = line as usize;
        let Some(&output) = self.route.get(line) else { return };
        if output == NO_OUTPUT {
            return;
        }
        let mut l = self.levels.lock().unwrap();
        if line < PIN_LINES {
            l.pins = (l.pins & !(1u128 << line)) | (level as u128) << line;
        } else {
            let b = line - PIN_LINES;
            l.wires = (l.wires & !(1u32 << b)) | (level as u32) << b;
        }
        self.host.set_output(output, self.any(&l, output));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pci::host::TestHost;
    use crate::pci::slots::{Evidence, SlotDesc, SlotKind, WireDesc};

    static DESC: BusDesc = BusDesc {
        id: "test",
        name: "test bus",
        devices: 8,
        outputs: 16,
        slots: &[
            SlotDesc { device: 3, label: "option", kind: SlotKind::AddIn, evidence: Evidence::Established,
                source: "test", routes: [Some(12), Some(13), None, None] },
            SlotDesc { device: 4, label: "other", kind: SlotKind::AddIn, evidence: Evidence::Established,
                source: "test", routes: [Some(12), None, None, None] },
        ],
        wires: &[WireDesc { name: "super I/O", output: 13, source: "test" }],
    };

    fn table() -> (Arc<TestHost>, Arc<IntxTable>) {
        let h = TestHost::new();
        let t = IntxTable::new(&DESC, h.clone());
        (h, t)
    }

    fn last(h: &TestHost, output: u8) -> Option<bool> {
        h.outputs().iter().rev().find(|(o, _)| *o == output).map(|(_, l)| *l)
    }

    #[test]
    fn a_shared_output_stays_up_until_every_pin_drops() {
        let (h, t) = table();
        let (a, b) = (t.pin_line(3, 1), t.pin_line(4, 1));
        a.set(true);
        b.set(true);
        a.set(false);
        assert_eq!(h.outputs().last(), Some(&(12, true)), "device 4 still holds it");
        b.set(false);
        assert_eq!(h.outputs().last(), Some(&(12, false)));
    }

    #[test]
    fn a_wire_ors_with_pins_on_its_output() {
        let (h, t) = table();
        let (pin, wire) = (t.pin_line(3, 2), t.wire_line("super I/O"));
        pin.set(true);
        wire.set(true);
        pin.set(false);
        assert_eq!(h.outputs().last(), Some(&(13, true)), "the wire still holds it");
        wire.set(false);
        assert_eq!(h.outputs().last(), Some(&(13, false)));
    }

    #[test]
    fn an_unrouted_pin_is_a_line_that_leads_nowhere() {
        let (h, t) = table();
        t.pin_line(3, 3).set(true);
        t.pin_line(4, 2).set(true);
        t.pin_line(0, 1).set(true);
        assert!(h.outputs().is_empty());
    }

    #[test]
    fn levels_survive_a_save_and_load_but_drive_nothing_until_redriven() {
        let (_, t) = table();
        t.pin_line(3, 1).set(true);
        t.wire_line("super I/O").set(true);
        let saved = t.save_state();
        let (h2, t2) = table();
        t2.load_state(&saved).unwrap();
        assert!(h2.outputs().is_empty());
        assert_eq!(t2.save_state(), saved);
        t2.redrive();
        assert_eq!(last(&h2, 12), Some(true));
        assert_eq!(last(&h2, 13), Some(true));
        assert!(h2.outputs().iter().all(|(o, _)| *o == 12 || *o == 13), "only routed outputs are driven");
    }

    #[test]
    fn a_malformed_saved_state_is_refused() {
        let (_, t) = table();
        assert!(t.load_state(&toml::Value::Table(Default::default())).is_err());
        let mut m = toml::map::Map::new();
        m.insert("levels".into(), u32_slice_to_toml(&[0, 0]));
        assert!(t.load_state(&toml::Value::Table(m)).is_err());
    }

    /// One pin is held high for the whole run while another toggles on the
    /// same output. With the level worked out under the lock the output is
    /// never driven low; worked out outside it, a stale low could land
    /// after the other pin's high.
    #[test]
    fn concurrent_pins_on_one_output_never_drive_it_low_while_one_is_high() {
        let (h, t) = table();
        let (held, toggled) = (t.pin_line(3, 1), t.pin_line(4, 1));
        held.set(true);
        let threads: Vec<_> = (0..2)
            .map(|_| {
                let l = toggled.clone();
                std::thread::spawn(move || {
                    for _ in 0..2000 {
                        l.set(true);
                        l.set(false);
                    }
                })
            })
            .collect();
        for th in threads {
            th.join().unwrap();
        }
        assert!(h.outputs().iter().all(|&(o, level)| o != 12 || level), "output 12 was driven low while a pin held it");
        held.set(false);
        assert_eq!(last(&h, 12), Some(false));
    }
}
