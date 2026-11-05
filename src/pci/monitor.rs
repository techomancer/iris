//! What the monitor shows of a PCI bus: the described devices and what is
//! fitted in them, and a function's header. Formatting lives here, apart from
//! the bus, which knows nothing of how it is displayed.

use std::sync::Arc;

use super::bus::PciBus;
use super::slots::PciAddr;
use super::PciBusDevice;

/// What is on the bus, one line per described device for the monitor:
/// the device number, its label, and either `empty` or what is fitted
/// there (ids, kind, command, BAR bases and where its interrupt pin
/// goes). A further function of a device follows on its own line.
pub fn report(bus: &PciBus) -> Vec<String> {
    let mut lines = Vec::new();
    let describe = |f: &Arc<dyn PciBusDevice>, desc: &super::slots::SlotDesc| {
        let c = f.config();
        let id = c.read(0);
        let bars: Vec<String> = (0..6).map(|i| c.bar_base(i)).filter(|&b| b != 0).map(|b| format!("{b:#x}")).collect();
        let pin = c.interrupt_pin();
        let irq = match pin {
            0 => "no interrupt".to_string(),
            1..=4 => match desc.routes[pin as usize - 1] {
                Some(o) => format!("INT{} to output {o}", (b'A' + pin - 1) as char),
                None => format!("INT{} unrouted", (b'A' + pin - 1) as char),
            },
            _ => format!("pin {pin}"),
        };
        format!("{:04x}:{:04x}, {}, command {:#06x}, BARs [{}], {irq}",
            id & 0xffff, id >> 16, desc.kind.word(), c.command(), bars.join(", "))
    };
    for desc in bus.desc().slots {
        let name = if desc.label.is_empty() { format!("device {}", desc.device) } else { format!("device {} \"{}\"", desc.device, desc.label) };
        let first = PciAddr::dev(desc.device);
        match bus.function(first) {
            Some(f) => lines.push(format!("{name}: {}", describe(f, desc))),
            None => lines.push(format!("{name}: empty")),
        }
        for function in 1..8 {
            let at = PciAddr { device: desc.device, function };
            if let Some(f) = bus.function(at) {
                lines.push(format!("  function {function}: {}", describe(f, desc)));
            }
        }
    }
    lines
}

/// The header of the function at `at` as the guest reads it: 16 rows of
/// 4 dwords.
pub fn config_dump(bus: &PciBus, at: PciAddr) -> Option<Vec<String>> {
    let f = bus.function(at)?;
    Some((0..16).map(|row| {
        let v: Vec<String> = (0..4).map(|k| format!("{:08x}", f.config_dword(row * 4 + k))).collect();
        format!("  {:02x}: {}", row * 16, v.join(" "))
    }).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pci::bus::tests::{bus, Fake};
    use crate::pci::PCI_INTA;

    #[test]
    fn report_lists_every_described_device_with_what_is_fitted() {
        let (_, b) = bus();
        b.attach(PciAddr::dev(4), Fake::new(PCI_INTA));
        b.attach(PciAddr { device: 4, function: 1 }, Fake::new(0));
        let r = report(&b);
        assert_eq!(r.len(), 4, "three devices and one more function: {r:?}");
        assert_eq!(r[0], "device 1 \"onboard\": empty");
        assert_eq!(r[1], "device 3 \"option\": empty");
        assert!(r[2].starts_with("device 4 \"other\": 9004:8078, add-in, command 0x0000"), "{}", r[2]);
        assert!(r[2].ends_with("INTA to output 12"), "{}", r[2]);
        assert!(r[3].starts_with("  function 1: 9004:8078") && r[3].ends_with("no interrupt"), "{}", r[3]);
    }

    #[test]
    fn config_dump_of_an_empty_device_is_none() {
        let (_, b) = bus();
        b.attach(PciAddr::dev(3), Fake::new(PCI_INTA));
        assert!(config_dump(&b, PciAddr::dev(4)).is_none());
        let d = config_dump(&b, PciAddr::dev(3)).unwrap();
        assert_eq!(d.len(), 16);
        assert_eq!(d[0], "  00: 80789004 02000000 01000001 00000000");
        assert!(d[3].starts_with("  30: 00000000 00000000 00000000 00000100"), "{}", d[3]);
    }
}
