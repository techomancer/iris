//! Which PCI slots a machine has, and which of them take an add-in card.
//!
//! A machine's PCI buses are described by `const` tables: a bus id, and for
//! each device number on it either an onboard function (hard-wired by the
//! machine, never configurable) or an add-in slot. A row says how well its
//! device number is known and where that comes from. Only a slot whose
//! number is established or observed can be filled; the others are listed
//! so an error can say what is missing instead of "no such slot".
//!
//! Slots are named the way a machine's PCI bus numbers them, by device
//! number. A slot can also carry the label printed on the chassis; labels
//! are only a way to write the same device number in the configuration.

use crate::config::MachineProfile;

/// How well a fact is known, from strongest to weakest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Evidence {
    /// A manual, or software that must agree with the hardware.
    Established,
    /// Seen on a real machine.
    Observed,
    /// Follows from a neighbouring fact, not seen.
    Inferred,
    /// A modelling choice that behaves like the hardware where tested.
    Approximation,
    /// No source says.
    Unknown,
}

impl Evidence {
    /// A slot can be configured only when its device number is known.
    pub fn selectable(self) -> bool {
        matches!(self, Self::Established | Self::Observed)
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Established => "established",
            Self::Observed => "observed",
            Self::Inferred => "inferred",
            Self::Approximation => "approximation",
            Self::Unknown => "unknown",
        }
    }
}

/// A slot that takes an add-in card.
#[derive(Debug)]
pub struct SlotDesc {
    /// The PCI device number (the IDSEL line) on the bus.
    pub device: u8,
    /// What the chassis calls the slot, or "" when it carries no label.
    pub label: &'static str,
    /// How well `device` is known.
    pub evidence: Evidence,
    /// The public manual or driver the number comes from.
    pub source: &'static str,
    /// What would make an unselectable slot selectable.
    pub settled_by: &'static str,
}

/// One device number on a bus.
#[derive(Debug)]
pub enum BusSlot {
    /// A function the machine always has, wired by the machine.
    Onboard { device: u8, name: &'static str },
    AddIn(SlotDesc),
}

impl BusSlot {
    pub fn device(&self) -> u8 {
        match self {
            Self::Onboard { device, .. } => *device,
            Self::AddIn(s) => s.device,
        }
    }
}

/// A PCI bus and the device numbers that are known on it.
#[derive(Debug)]
pub struct BusDesc {
    /// The id a configuration names the bus by.
    pub id: &'static str,
    /// The bus as an error message names it, "the O2's PCI bus".
    pub title: &'static str,
    /// Ascending by device number.
    pub slots: &'static [BusSlot],
    /// Why the bus has no add-in slot, when it has none.
    pub no_add_in: &'static str,
}

impl BusDesc {
    pub fn add_ins(&self) -> impl Iterator<Item = &SlotDesc> {
        self.slots.iter().filter_map(|s| match s {
            BusSlot::AddIn(d) => Some(d),
            BusSlot::Onboard { .. } => None,
        })
    }

    /// The add-in slots as an error lists them: `3`, or `5 (label)`.
    pub fn add_in_list(&self) -> String {
        let all: Vec<String> = self
            .add_ins()
            .map(|s| if s.label.is_empty() { s.device.to_string() } else { format!("{} ({})", s.device, s.label) })
            .collect();
        if all.is_empty() { "none".into() } else { all.join(", ") }
    }
}

/// The PCI buses of a machine, empty for one with no PCI.
pub fn buses(profile: MachineProfile) -> &'static [BusDesc] {
    match profile {
        MachineProfile::O2Ip32 => crate::ip32::BUSES,
        MachineProfile::OctaneIp30 => crate::ip30::BUSES,
        MachineProfile::IndyIp24 | MachineProfile::Indigo2Ip22 | MachineProfile::Indigo2Ip28 => &[],
    }
}

/// How a message names the machine.
pub fn machine_noun(profile: MachineProfile) -> &'static str {
    match profile {
        MachineProfile::IndyIp24 => "the Indy",
        MachineProfile::Indigo2Ip22 | MachineProfile::Indigo2Ip28 => "the Indigo2",
        MachineProfile::O2Ip32 => "the O2",
        MachineProfile::OctaneIp30 => "the Octane",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_table_is_ascending_and_every_slot_is_sourced() {
        for profile in MachineProfile::EVERY {
            for bus in buses(profile) {
                let devices: Vec<u8> = bus.slots.iter().map(BusSlot::device).collect();
                let mut sorted = devices.clone();
                sorted.sort_unstable();
                sorted.dedup();
                assert_eq!(devices, sorted, "{}: devices ascending and unique", bus.id);
                assert!(devices.iter().all(|&d| d < 32), "{}: a bus has 32 devices", bus.id);
                for s in bus.add_ins() {
                    assert!(!s.source.is_empty(), "{} slot {}: no source", bus.id, s.device);
                    if !s.evidence.selectable() {
                        assert!(!s.settled_by.is_empty(), "{} slot {}: say what settles it", bus.id, s.device);
                    }
                }
                assert!(bus.add_ins().count() > 0 || !bus.no_add_in.is_empty(), "{}: a bus with no slot says why", bus.id);
            }
        }
    }

    #[test]
    fn only_the_o2_has_an_add_in_slot_today() {
        for profile in MachineProfile::EVERY {
            let slots: usize = buses(profile).iter().map(|b| b.add_ins().count()).sum();
            assert_eq!(slots, usize::from(profile == MachineProfile::O2Ip32), "{}", profile.name());
        }
    }
}
