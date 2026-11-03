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

use serde::{Deserialize, Serialize};

use crate::config::MachineProfile;
use crate::pci_cards::{card_def, card_names, CardDef, Options, PciCard};
use crate::scsi::ScsiBus;

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

/// A slot as a configuration names it: the PCI device number, or the label
/// on the chassis. Labels are only a spelling of a device number; they are
/// resolved here and appear nowhere else.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SlotRef {
    Device(u8),
    Label(String),
}

/// One `[[pci]]` entry: a card in a slot, with the card's own options inline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PciEntry {
    pub slot: SlotRef,
    /// The bus id, left out on a machine with one PCI bus.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bus: Option<String>,
    pub card: String,
    /// Every other key, for the card to read.
    #[serde(flatten)]
    pub options: Options,
}

impl PciEntry {
    /// Parse the command-line form `[BUS:]SLOT=CARD[,KEY=VALUE]...`. SLOT is
    /// a PCI device number: chassis labels are for the configuration file.
    /// Option values are booleans
    /// and integers where they read as one, text otherwise.
    pub fn parse_cli(text: &str) -> Result<Self, String> {
        let usage = "expected [BUS:]SLOT=CARD[,KEY=VALUE]...";
        let (place, rest) = text.split_once('=').ok_or_else(|| format!("pci \"{text}\": {usage}"))?;
        let (bus, slot) = match place.split_once(':') {
            Some((b, s)) if !b.is_empty() => (Some(b.to_string()), s),
            Some((_, s)) => (None, s),
            None => (None, place),
        };
        if slot.is_empty() {
            return Err(format!("pci \"{text}\": the slot is empty ({usage})"));
        }
        let slot = slot.parse::<u8>().map(SlotRef::Device).map_err(|_| {
            format!("pci \"{text}\": slot \"{slot}\" is not a PCI device number (0 to 255); chassis labels work only in the configuration file")
        })?;
        let mut parts = rest.split(',');
        let card = parts.next().filter(|c| !c.is_empty()).ok_or_else(|| format!("pci \"{text}\": the card is empty ({usage})"))?;
        let mut options = Options::new();
        for kv in parts {
            let (k, v) = kv.split_once('=').filter(|(k, _)| !k.is_empty()).ok_or_else(|| format!("pci \"{text}\": option \"{kv}\" is not KEY=VALUE"))?;
            let value = match v {
                "true" => toml::Value::Boolean(true),
                "false" => toml::Value::Boolean(false),
                _ => v.parse::<i64>().map_or_else(|_| toml::Value::String(v.to_string()), toml::Value::Integer),
            };
            if options.insert(k.to_string(), value).is_some() {
                return Err(format!("pci \"{text}\": option \"{k}\" is given twice"));
            }
        }
        Ok(Self { slot, bus, card: card.to_string(), options })
    }
}

/// An entry that passed [`validate_pci`], with everything it resolved to.
#[derive(Debug)]
pub struct Placement {
    pub bus: &'static BusDesc,
    pub slot: &'static SlotDesc,
    pub card: &'static CardDef,
    pub options: Options,
    /// The controller number of the card's first SCSI bus: after the
    /// machine's own, and after the cards before it.
    pub scsi_base: u8,
}

/// A card built for its slot.
pub struct Fitted {
    pub bus: &'static str,
    pub device: u8,
    /// The controller number of the card's first SCSI bus.
    pub scsi_base: u8,
    /// The entry's options as `key=value,...`, sorted, for a snapshot.
    pub options: String,
    pub card: Box<dyn PciCard>,
}

/// Build the card of every placement.
pub fn build_cards(placements: Vec<Placement>) -> Result<Vec<Fitted>, String> {
    placements
        .into_iter()
        .map(|p| {
            let card = (p.card.build)(&p.options, p.scsi_base).map_err(|e| format!("card \"{}\" in {} slot {}: {e}", p.card.name, p.bus.id, p.slot.device))?;
            let options = p.options.iter().map(|(k, v)| format!("{k}={}", v.as_str().map_or_else(|| v.to_string(), String::from))).collect::<Vec<_>>().join(",");
            Ok(Fitted { bus: p.bus.id, device: p.slot.device, scsi_base: p.scsi_base, options, card })
        })
        .collect()
}

impl Fitted {
    /// Where the card is and what it is, as a snapshot records it.
    pub fn pci_ref(&self) -> crate::snapshot::PciRef {
        crate::snapshot::PciRef { bus: self.bus.into(), slot: self.device, card: self.card.name().into(), options: self.options.clone() }
    }

    /// The card's buses with their controller numbers.
    pub fn buses(&self) -> impl Iterator<Item = (u8, &ScsiBus)> {
        self.card.scsi_buses().into_iter().enumerate().map(|(i, b)| (self.scsi_base + i as u8, b))
    }
}

/// The bus an entry names, or the machine's only one.
fn find_bus<'a>(all: &'a [BusDesc], noun: &str, bus: Option<&str>) -> Result<&'a BusDesc, String> {
    let ids = || all.iter().map(|b| b.id).collect::<Vec<_>>().join(", ");
    match (bus, all) {
        (_, []) => Err(format!("{noun} has no PCI bus.")),
        (Some(id), _) => all.iter().find(|b| b.id == id).ok_or_else(|| format!("{noun} has no PCI bus \"{id}\" (buses: {}).", ids())),
        (None, [only]) => Ok(only),
        (None, _) => Err(format!("{noun} has several PCI buses ({}); set bus.", ids())),
    }
}

/// The row a slot names on `bus`: a device number, or an add-in slot's label.
fn find_row<'a>(bus: &'a BusDesc, slot: &SlotRef) -> Option<&'a BusSlot> {
    match slot {
        SlotRef::Device(n) => bus.slots.iter().find(|s| s.device() == *n),
        SlotRef::Label(l) => bus.slots.iter().find(|s| matches!(s, BusSlot::AddIn(d) if !d.label.is_empty() && d.label == l)),
    }
}

/// Check every `[[pci]]` entry against the machine's slot tables and the
/// card registry, in the order a reader would fix them: bus, slot, whether
/// the slot is free, card, options. The first problem ends the check.
pub fn validate_pci(profile: MachineProfile, entries: &[PciEntry]) -> Result<Vec<Placement>, String> {
    place(buses(profile), machine_noun(profile), profile.scsi_controllers(), entries)
}

/// [`validate_pci`] for any table of buses, `scsi_base` being the first
/// controller number free for a card's SCSI buses.
fn place(all: &'static [BusDesc], noun: &str, mut scsi_base: u8, entries: &[PciEntry]) -> Result<Vec<Placement>, String> {
    let mut out: Vec<Placement> = Vec::new();
    let mut used: Vec<(&'static str, u8, usize)> = Vec::new();
    for (i, e) in entries.iter().enumerate() {
        let at = format!("pci[{i}]");
        let bus = find_bus(all, noun, e.bus.as_deref()).map_err(|m| format!("{at}: {m}"))?;
        if bus.add_ins().count() == 0 {
            return Err(format!("{at}: {} has no add-in slot: {}.", bus.title, bus.no_add_in));
        }
        let slot = match find_row(bus, &e.slot) {
            Some(BusSlot::AddIn(s)) => s,
            Some(BusSlot::Onboard { device, name }) => {
                return Err(format!(
                    "{at}: device {device} on {} is the onboard {name}, not an add-in slot. Add-in slots: {}.",
                    bus.title,
                    bus.add_in_list()
                ));
            }
            None => {
                let named = match &e.slot {
                    SlotRef::Device(n) => format!("device {n} on {} is not a slot", bus.title),
                    SlotRef::Label(l) if l.parse::<u8>().is_ok() => {
                        return Err(format!(
                            "{at}: slot \"{l}\" is text, so it names a chassis label, and {} has no such label. For device {l} write slot = {l} without quotes.",
                            bus.title
                        ));
                    }
                    SlotRef::Label(l) => format!("{} has no slot labelled \"{l}\"", bus.title),
                };
                return Err(format!("{at}: {named}. Add-in slots: {}. `pci slots` in the monitor lists them.", bus.add_in_list()));
            }
        };
        if !slot.evidence.selectable() {
            return Err(format!(
                "{at}: slot {} on {} cannot be configured: its device number is {}. {}",
                slot.device,
                bus.title,
                slot.evidence.name(),
                slot.settled_by
            ));
        }
        if let Some(&(_, _, j)) = used.iter().find(|&&(b, d, _)| b == bus.id && d == slot.device) {
            return Err(format!("{at}: {} slot {} is already used by pci[{j}] (card \"{}\").", bus.id, slot.device, entries[j].card));
        }
        let card = card_def(&e.card).ok_or_else(|| format!("{at}: unknown card \"{}\" (cards: {}).", e.card, card_names()))?;
        if let Some(k) = e.options.keys().find(|k| !card.options.contains(&k.as_str())) {
            let known = if card.options.is_empty() { "this card takes no options".to_string() } else { format!("known: {}", card.options.join(", ")) };
            return Err(format!("{at} card \"{}\": unknown option \"{k}\" ({known}).", card.name));
        }
        used.push((bus.id, slot.device, i));
        out.push(Placement { bus, slot, card, options: e.options.clone(), scsi_base });
        scsi_base = scsi_base.saturating_add(card.scsi_controllers);
    }
    Ok(out)
}

/// Add the command line's entries to the configuration's: an entry for a
/// slot already filled there replaces it in place, any other is appended.
/// Entries that do not resolve are appended and left for [`validate_pci`]
/// to report.
pub fn merge_cli(profile: MachineProfile, base: &mut Vec<PciEntry>, cli: &[PciEntry]) {
    let identity = |e: &PciEntry| -> Option<(&'static str, u8)> {
        let bus = find_bus(buses(profile), machine_noun(profile), e.bus.as_deref()).ok()?;
        Some((bus.id, find_row(bus, &e.slot)?.device()))
    };
    for e in cli {
        let id = identity(e);
        match base.iter_mut().find(|b| id.is_some() && identity(b) == id) {
            Some(slot) => *slot = e.clone(),
            None => base.push(e.clone()),
        }
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
    fn entry(text: &str) -> PciEntry {
        PciEntry::parse_cli(text).unwrap()
    }

    /// An entry that names its slot by label, as only a config file can.
    fn labelled(bus: &str, label: &str, card: &str) -> PciEntry {
        PciEntry { slot: SlotRef::Label(label.into()), bus: Some(bus.into()), card: card.into(), options: Options::new() }
    }

    fn o2(texts: &[&str]) -> Result<Vec<Placement>, String> {
        validate_pci(MachineProfile::O2Ip32, &texts.iter().map(|t| entry(t)).collect::<Vec<_>>())
    }

    /// A bus with labelled slots in each state, for what the real tables
    /// cannot show yet: more than one slot, labels, an unsettled number.
    static TEST_BUSES: [BusDesc; 2] = [
        BusDesc {
            id: "a",
            title: "bus a",
            slots: &[
                BusSlot::Onboard { device: 1, name: "disk controller" },
                BusSlot::AddIn(SlotDesc { device: 5, label: "slot 1", evidence: Evidence::Established, source: "test", settled_by: "" }),
                BusSlot::AddIn(SlotDesc { device: 6, label: "slot 2", evidence: Evidence::Observed, source: "test", settled_by: "" }),
                BusSlot::AddIn(SlotDesc { device: 7, label: "slot 3", evidence: Evidence::Unknown, source: "test", settled_by: "a capture with a card in it" }),
            ],
            no_add_in: "",
        },
        // Device 5 again, on another bus: a slot is a bus and a device number.
        BusDesc { id: "b", title: "bus b", slots: &[BusSlot::AddIn(SlotDesc { device: 5, label: "", evidence: Evidence::Established, source: "test", settled_by: "" })], no_add_in: "" },
    ];

    fn test_bus(texts: &[&str]) -> Result<Vec<Placement>, String> {
        place(&TEST_BUSES, "the test machine", 2, &texts.iter().map(|t| entry(t)).collect::<Vec<_>>())
    }

    #[test]
    fn the_o2s_slot_takes_a_card_and_numbers_its_scsi_buses_after_the_onboard_two() {
        let p = o2(&["3=isp12160"]).unwrap();
        assert_eq!((p[0].bus.id, p[0].slot.device, p[0].card.name, p[0].scsi_base), ("pci", 3, "isp12160", 2));
    }

    #[test]
    fn cli_form_parses_bus_slot_card_and_options() {
        let e = entry("e:2=isp12160,depth=4,fast=true,mode=wide");
        assert_eq!((e.bus.as_deref(), &e.slot, e.card.as_str()), (Some("e"), &SlotRef::Device(2), "isp12160"));
        assert_eq!(e.options["depth"], toml::Value::Integer(4));
        assert_eq!(e.options["fast"], toml::Value::Boolean(true));
        assert_eq!(e.options["mode"], toml::Value::String("wide".into()));
        for bad in ["isp12160", "3=", "=isp12160", "3=isp12160,x", "3=isp12160,a=1,a=2", ":=x"] {
            assert!(PciEntry::parse_cli(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn the_command_line_takes_device_numbers_only() {
        for bad in ["left=isp12160", "300=isp12160", "a:slot 1=isp12160"] {
            let err = PciEntry::parse_cli(bad).unwrap_err();
            assert!(err.contains("is not a PCI device number") && err.contains("only in the configuration file"), "{bad}: {err}");
        }
    }

    #[test]
    fn a_label_names_the_same_slot_as_its_device_number() {
        let by_label = place(&TEST_BUSES, "the test machine", 2, &[labelled("a", "slot 2", "isp12160")]).unwrap();
        let by_device = test_bus(&["a:6=isp12160"]).unwrap();
        assert_eq!(by_label[0].slot.device, 6);
        assert_eq!(by_label[0].slot.device, by_device[0].slot.device);
        let err = place(&TEST_BUSES, "the test machine", 2, &[labelled("a", "slot 9", "isp12160")]).unwrap_err();
        assert_eq!(err, "pci[0]: bus a has no slot labelled \"slot 9\". Add-in slots: 5 (slot 1), 6 (slot 2), 7 (slot 3). `pci slots` in the monitor lists them.");
    }

    #[test]
    fn two_cards_of_one_kind_get_consecutive_scsi_controllers() {
        let p = test_bus(&["a:5=isp12160", "a:6=isp12160", "b:5=isp12160"]).unwrap();
        assert_eq!(p.iter().map(|p| p.scsi_base).collect::<Vec<_>>(), vec![2, 4, 6]);
    }

    #[test]
    fn a_slot_cannot_be_used_twice_but_another_bus_is_free() {
        let by_label = [entry("a:5=isp12160"), labelled("a", "slot 1", "isp12160")];
        let err = place(&TEST_BUSES, "the test machine", 2, &by_label).unwrap_err();
        assert_eq!(err, "pci[1]: a slot 5 is already used by pci[0] (card \"isp12160\").");
        let err = test_bus(&["a:5=isp12160", "a:5=aic7880"]).unwrap_err();
        assert!(err.contains("already used by pci[0]"), "{err}");
        test_bus(&["a:5=isp12160", "b:5=isp12160"]).expect("device 5 on the other bus is another slot");
    }

    #[test]
    fn errors_say_what_is_wrong_and_what_to_do() {
        assert_eq!(
            o2(&["7=isp12160"]).err().unwrap(),
            "pci[0]: device 7 on the O2's PCI bus is not a slot. Add-in slots: 3. `pci slots` in the monitor lists them."
        );
        assert_eq!(
            o2(&["1=isp12160"]).err().unwrap(),
            "pci[0]: device 1 on the O2's PCI bus is the onboard internal SCSI controller (AIC-7880), not an add-in slot. Add-in slots: 3."
        );
        assert_eq!(o2(&["3=rad9"]).err().unwrap(), "pci[0]: unknown card \"rad9\" (cards: isp12160, aic7880).");
        assert_eq!(
            o2(&["3=isp12160,volume=3"]).err().unwrap(),
            "pci[0] card \"isp12160\": unknown option \"volume\" (this card takes no options)."
        );
        assert_eq!(
            test_bus(&["a:7=isp12160"]).err().unwrap(),
            "pci[0]: slot 7 on bus a cannot be configured: its device number is unknown. a capture with a card in it"
        );
        assert_eq!(test_bus(&["5=isp12160"]).err().unwrap(), "pci[0]: the test machine has several PCI buses (a, b); set bus.");
        assert_eq!(test_bus(&["c:5=isp12160"]).err().unwrap(), "pci[0]: the test machine has no PCI bus \"c\" (buses: a, b).");
        assert_eq!(o2(&["pci:3=isp12160"]).unwrap().len(), 1);
        let quoted = place(&TEST_BUSES, "the test machine", 2, &[labelled("a", "6", "isp12160")]).unwrap_err();
        assert_eq!(
            quoted,
            "pci[0]: slot \"6\" is text, so it names a chassis label, and bus a has no such label. For device 6 write slot = 6 without quotes."
        );
    }

    #[test]
    fn a_machine_without_an_add_in_slot_says_why() {
        let indy = validate_pci(MachineProfile::IndyIp24, &[entry("3=isp12160")]).err().unwrap();
        assert_eq!(indy, "pci[0]: the Indy has no PCI bus.");
        let octane = validate_pci(MachineProfile::OctaneIp30, &[entry("4=isp12160")]).err().unwrap();
        assert!(octane.starts_with("pci[0]: the Octane's Bridge has no add-in slot: "), "{octane}");
    }

    #[test]
    fn a_command_line_entry_replaces_the_config_entry_for_its_slot() {
        let mut base = vec![entry("3=isp12160")];
        merge_cli(MachineProfile::O2Ip32, &mut base, &[entry("pci:3=isp12160"), entry("9=isp12160")]);
        assert_eq!(base.len(), 2);
        assert_eq!(base[0].bus.as_deref(), Some("pci"), "replaced in place");
        assert_eq!(base[1].slot, SlotRef::Device(9), "an entry that names no slot is kept for validate to report");
    }

}
