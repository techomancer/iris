//! What a machine's PCI bus is, as data: which device numbers exist, what
//! each is (a board's own chip, a socket for a card, a number kept for a
//! paravirtual device), which host output each interrupt pin reaches, and
//! where each of those facts comes from. A machine writes one `static
//! BusDesc` next to its own code; the bus reads it and nothing else knows
//! the chassis.

use std::fmt;

/// A function's address on a bus: device number and function number.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct PciAddr {
    pub device: u8,
    pub function: u8,
}

impl PciAddr {
    /// Function 0 of `device`, which is every function of a single-function card.
    pub const fn dev(device: u8) -> Self {
        Self { device, function: 0 }
    }
}

impl fmt::Display for PciAddr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.function == 0 {
            write!(f, "device {}", self.device)
        } else {
            write!(f, "device {} function {}", self.device, self.function)
        }
    }
}

/// How well a fact about the hardware is known.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Evidence {
    /// From the chip or board documentation.
    Established,
    /// Seen on real hardware.
    Observed,
    /// Worked out from related hardware.
    Inferred,
    /// Chosen so software runs; the hardware has no such thing.
    Approximation,
}

impl Evidence {
    /// Whether a user may be offered a slot on this evidence.
    pub const fn offered(self) -> bool {
        matches!(self, Evidence::Established | Evidence::Observed)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SlotKind {
    /// A chip on the board. The machine places it; a user never does.
    Onboard,
    /// A socket for a card, as the machine labels it.
    AddIn,
    /// A device number kept for a paravirtual device, with no physical slot.
    ParavirtualSpare,
}

impl SlotKind {
    pub fn word(self) -> &'static str {
        match self {
            SlotKind::Onboard => "onboard",
            SlotKind::AddIn => "add-in",
            SlotKind::ParavirtualSpare => "paravirtual",
        }
    }
}

/// One device number on a bus.
pub struct SlotDesc {
    pub device: u8,
    /// What the machine calls it: a rear-panel label, or the chip.
    pub label: &'static str,
    pub kind: SlotKind,
    pub evidence: Evidence,
    pub source: &'static str,
    /// The host output INTA..INTD reach; `None` is a pin that is not routed.
    pub routes: [Option<u8>; 4],
}

/// An interrupt that reaches a host output without passing through a slot,
/// such as a board wire from a chip that happens to share a pin.
pub struct WireDesc {
    pub name: &'static str,
    pub output: u8,
    pub source: &'static str,
}

/// A bus: its size, its slots and its extra wires.
pub struct BusDesc {
    /// Short name for diagnostics and saved state.
    pub id: &'static str,
    pub name: &'static str,
    /// Device numbers `0..devices` exist, at most 32.
    pub devices: u8,
    /// Host outputs `0..outputs` exist.
    pub outputs: u8,
    pub slots: &'static [SlotDesc],
    pub wires: &'static [WireDesc],
}

impl BusDesc {
    pub fn slot(&self, device: u8) -> Option<&SlotDesc> {
        self.slots.iter().find(|s| s.device == device)
    }

    /// The slot a user's label names: ASCII case and surrounding space do not matter.
    pub fn by_label(&self, label: &str) -> Option<&SlotDesc> {
        let label = label.trim();
        self.slots.iter().find(|s| s.label.eq_ignore_ascii_case(label))
    }

    pub fn wire(&self, name: &str) -> Option<&WireDesc> {
        self.wires.iter().find(|w| w.name == name)
    }

    /// Every device, for an error that tells the user what this machine has.
    pub fn listing(&self) -> String {
        self.slots
            .iter()
            .map(|s| {
                if s.label.is_empty() {
                    format!("device {} ({})", s.device, s.kind.word())
                } else {
                    format!("device {} \"{}\" ({})", s.device, s.label, s.kind.word())
                }
            })
            .collect::<Vec<_>>()
            .join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    static TEST: BusDesc = BusDesc {
        id: "pci",
        name: "test bus",
        devices: 8,
        outputs: 4,
        slots: &[
            SlotDesc { device: 1, label: "internal SCSI", kind: SlotKind::Onboard, evidence: Evidence::Established,
                source: "test", routes: [Some(0), None, None, None] },
            SlotDesc { device: 3, label: "PCI option", kind: SlotKind::AddIn, evidence: Evidence::Established,
                source: "test", routes: [Some(1), Some(2), None, None] },
            SlotDesc { device: 4, label: "", kind: SlotKind::ParavirtualSpare, evidence: Evidence::Approximation,
                source: "test", routes: [Some(3), None, None, None] },
        ],
        wires: &[WireDesc { name: "super I/O", output: 2, source: "test" }],
    };

    #[test]
    fn by_label_ignores_case_and_surrounding_space() {
        assert_eq!(TEST.by_label(" pci OPTION ").unwrap().device, 3);
        assert!(TEST.by_label("pci").is_none());
    }

    #[test]
    fn listing_names_every_device_and_its_kind() {
        let l = TEST.listing();
        assert!(l.contains("device 3 \"PCI option\" (add-in)"), "{l}");
        assert!(l.contains("device 1 \"internal SCSI\" (onboard)"), "{l}");
        assert!(l.contains("device 4 (paravirtual)"), "{l}");
        assert!(!l.contains("device 4 \""), "an unlabelled slot prints no quotes: {l}");
    }

    #[test]
    fn only_established_and_observed_evidence_is_offered() {
        assert!(Evidence::Established.offered());
        assert!(Evidence::Observed.offered());
        assert!(!Evidence::Inferred.offered());
        assert!(!Evidence::Approximation.offered());
    }

    #[test]
    fn addr_display() {
        assert_eq!(PciAddr::dev(3).to_string(), "device 3");
        assert_eq!(PciAddr { device: 3, function: 1 }.to_string(), "device 3 function 1");
    }

    #[test]
    fn slots_and_wires_are_found_by_number_and_name() {
        assert_eq!(TEST.slot(4).unwrap().routes[0], Some(3));
        assert!(TEST.slot(5).is_none());
        assert_eq!(TEST.wire("super I/O").unwrap().output, 2);
        assert!(TEST.wire("other").is_none());
    }
}
