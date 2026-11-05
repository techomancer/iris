//! Putting the cards a user asked for into a machine's buses.
//!
//! `[[pci]]` entries (or `--pci`) name a slot as the machine labels it, or
//! by device number, and a card. [`resolve`] checks every entry against the
//! machine's bus descriptions before anything is built, so a mistake is one
//! line that names what the machine does have; [`build`] then makes the
//! functions for the machine to attach.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::bus::CardRef;
use super::card::{describe_options, CardBuild, PciCard};
use super::slots::{BusDesc, PciAddr, SlotDesc, SlotKind};
use super::PciBusDevice;

/// Where a card goes: the label the machine gives the slot, or its device
/// number (the advanced spelling, as QEMU's `addr=`).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum SlotRef {
    Device(u8),
    Label(String),
    /// Something that is neither (`slot = true`, `slot = 300`), kept so
    /// that [`resolve`] can name the entry and the slots the machine has.
    Invalid(String),
}

impl Serialize for SlotRef {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            SlotRef::Device(d) => s.serialize_u8(*d),
            SlotRef::Label(l) | SlotRef::Invalid(l) => s.serialize_str(l),
        }
    }
}

impl<'de> Deserialize<'de> for SlotRef {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = SlotRef;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a slot label or a device number")
            }
            fn visit_i64<E>(self, v: i64) -> Result<SlotRef, E> {
                Ok(u8::try_from(v).map_or_else(|_| SlotRef::Invalid(v.to_string()), SlotRef::Device))
            }
            fn visit_u64<E>(self, v: u64) -> Result<SlotRef, E> {
                Ok(u8::try_from(v).map_or_else(|_| SlotRef::Invalid(v.to_string()), SlotRef::Device))
            }
            fn visit_bool<E>(self, v: bool) -> Result<SlotRef, E> {
                Ok(SlotRef::Invalid(v.to_string()))
            }
            fn visit_str<E>(self, v: &str) -> Result<SlotRef, E> {
                Ok(SlotRef::Label(v.to_string()))
            }
            fn visit_f64<E>(self, v: f64) -> Result<SlotRef, E> {
                Ok(SlotRef::Invalid(v.to_string()))
            }
        }
        d.deserialize_any(V)
    }
}

/// One card to fit, as a `[[pci]]` table. Options the card takes sit beside
/// `slot` and `card`.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct PciEntry {
    pub slot: SlotRef,
    pub card: PciCard,
    /// Which bus, for a machine with more than one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bus: Option<String>,
    #[serde(flatten)]
    pub options: toml::Table,
}

impl PciEntry {
    /// `[BUS:]SLOT=CARD[,KEY=VALUE]...`, the command-line spelling.
    pub fn parse_cli(s: &str) -> Result<PciEntry, String> {
        use clap::ValueEnum;
        let (place, what) = s.split_once('=').ok_or_else(|| format!("{s:?}: expected [BUS:]SLOT=CARD[,KEY=VALUE]..."))?;
        let (bus, slot) = match place.split_once(':') {
            Some((b, sl)) => (Some(b.trim().to_string()), sl),
            None => (None, place),
        };
        let slot = slot.trim();
        if slot.is_empty() {
            return Err(format!("{s:?}: no slot before ="));
        }
        let mut parts = what.split(',');
        let name = parts.next().unwrap_or("").trim();
        if name.is_empty() {
            return Err(format!("{s:?}: no card after ="));
        }
        let card = PciCard::from_str(name, false).map_err(|_| unknown_card(name))?;
        let mut options = toml::Table::new();
        for kv in parts {
            let (k, v) = kv.split_once('=').ok_or_else(|| format!("{s:?}: option {kv:?} is not KEY=VALUE"))?;
            // A value is a TOML literal where it parses as one (4, true), and text otherwise.
            let value = toml::from_str::<toml::Table>(&format!("v = {}", v.trim()))
                .ok()
                .and_then(|mut t| t.remove("v"))
                .unwrap_or_else(|| toml::Value::String(v.trim().to_string()));
            options.insert(k.trim().to_string(), value);
        }
        let slot = slot.parse::<u8>().map(SlotRef::Device).unwrap_or_else(|_| SlotRef::Label(slot.to_string()));
        Ok(PciEntry { slot, card, bus, options })
    }
}

fn unknown_card(name: &str) -> String {
    use clap::ValueEnum;
    let cards: Vec<String> = PciCard::value_variants().iter().map(|c| c.name()).collect();
    format!("no card {name:?}; cards are: {}.", cards.join(", "))
}

/// A checked entry: the slot it takes and the controllers it brings.
pub struct Placement {
    pub bus: &'static BusDesc,
    pub addr: PciAddr,
    pub slot: &'static SlotDesc,
    pub card: PciCard,
    pub options: toml::Table,
    /// The number of the first SCSI controller the card takes.
    pub first_controller: u8,
}

impl Placement {
    pub fn card_ref(&self) -> CardRef {
        CardRef { name: self.card.name(), options: describe_options(&self.options), first_controller: self.first_controller }
    }
}

/// The bus an entry names, or the machine's only bus.
fn bus_of(buses: &[&'static BusDesc], e: &PciEntry, i: usize) -> Result<&'static BusDesc, String> {
    match (&e.bus, buses) {
        (_, []) => Err(format!("pci[{i}]: this machine has no PCI bus.")),
        (None, [only]) => Ok(only),
        (None, many) => {
            let ids: Vec<_> = many.iter().map(|b| b.id).collect();
            Err(format!("pci[{i}]: this machine has several PCI buses ({}); name one with bus = \"...\".", ids.join(", ")))
        }
        (Some(id), many) => many.iter().copied().find(|b| b.id == id).ok_or_else(|| {
            let ids: Vec<_> = many.iter().map(|b| b.id).collect();
            format!("pci[{i}]: no PCI bus {id:?}; this machine has: {}.", ids.join(", "))
        }),
    }
}

/// The slot an entry names on `bus`, if it offers one.
fn slot_of(bus: &'static BusDesc, e: &PciEntry, i: usize) -> Result<&'static SlotDesc, String> {
    let have = || format!("this bus has: {}.", bus.listing());
    let invalid = |v: &str| format!("pci[{i}]: slot must be a label or a device number from 0 to 255, not {v}; {}", have());
    if let SlotRef::Invalid(v) = &e.slot {
        return Err(invalid(v));
    }
    if !bus.slots.iter().any(|s| s.kind == SlotKind::AddIn) {
        return Err(format!("pci[{i}]: bus {:?} has no add-in slot; {}", bus.id, have()));
    }
    let slot = match &e.slot {
        SlotRef::Device(d) => bus.slot(*d).ok_or_else(|| format!("pci[{i}]: bus {:?} has no device {d}; {}", bus.id, have()))?,
        SlotRef::Invalid(v) => return Err(invalid(v)),
        SlotRef::Label(l) => bus.by_label(l).ok_or_else(|| format!("pci[{i}]: bus {:?} has no slot labelled {l:?}; {}", bus.id, have()))?,
    };
    let name = if slot.label.is_empty() { format!("device {}", slot.device) } else { format!("device {} ({:?})", slot.device, slot.label) };
    match slot.kind {
        SlotKind::Onboard => Err(format!("pci[{i}]: {name} is an onboard device, not a slot; {}", have())),
        SlotKind::ParavirtualSpare => {
            Err(format!("pci[{i}]: {name} is reserved for a paravirtual device, not a slot; {}", have()))
        }
        SlotKind::AddIn if !slot.evidence.offered() => {
            Err(format!("pci[{i}]: {name} is not offered: its evidence is only {:?}; {}", slot.evidence, have()))
        }
        SlotKind::AddIn => Ok(slot),
    }
}

/// Check every entry against the machine's buses and number the SCSI
/// controllers the cards bring from `first_controller`, in entry order.
pub fn resolve(buses: &[&'static BusDesc], entries: &[PciEntry], first_controller: u8) -> Result<Vec<Placement>, String> {
    let mut placed: Vec<Placement> = Vec::new();
    let mut next = first_controller;
    for (i, e) in entries.iter().enumerate() {
        let bus = bus_of(buses, e, i)?;
        let slot = slot_of(bus, e, i)?;
        if let Some(j) = placed.iter().position(|p| p.bus.id == bus.id && p.slot.device == slot.device) {
            return Err(format!(
                "pci[{i}]: device {} on bus {:?} is already taken by pci[{j}] ({}).",
                slot.device, bus.id, placed[j].card.name()
            ));
        }
        let desc = e.card.desc();
        (desc.check)(&e.options).map_err(|m| format!("pci[{i}]: card {:?}: {m}.", e.card.name()))?;
        placed.push(Placement {
            bus,
            addr: PciAddr::dev(slot.device),
            slot,
            card: e.card,
            options: e.options.clone(),
            first_controller: next,
        });
        next = next.checked_add(desc.scsi.controllers).ok_or_else(|| format!("pci[{i}]: too many SCSI controllers."))?;
    }
    Ok(placed)
}

/// Add the command line's entries to the config's: one for a slot the
/// config already fills, however either spells it, replaces it.
pub fn merge_cli(buses: &[&'static BusDesc], entries: &mut Vec<PciEntry>, cli: &[PciEntry]) {
    let key = |e: &PciEntry| -> (Option<String>, SlotRef) {
        let bus = bus_of(buses, e, 0).ok();
        let slot = bus.and_then(|b| slot_of(b, e, 0).ok()).map_or_else(|| e.slot.clone(), |s| SlotRef::Device(s.device));
        (bus.map(|b| b.id.to_string()).or_else(|| e.bus.clone()), slot)
    };
    for c in cli {
        let k = key(c);
        match entries.iter().position(|e| key(e) == k) {
            Some(at) => entries[at] = c.clone(),
            None => entries.push(c.clone()),
        }
    }
}

/// A card made and ready to attach.
pub struct Fitted {
    pub placement: Placement,
    pub function: Arc<dyn PciBusDevice>,
}

pub fn build(placements: Vec<Placement>) -> Result<Vec<Fitted>, String> {
    placements
        .into_iter()
        .map(|placement| {
            let b = CardBuild { options: &placement.options, first_controller: placement.first_controller };
            let function = (placement.card.desc().build)(&b)
                .map_err(|e| format!("card {:?} for {}: {e}", placement.card.name(), placement.addr))?;
            Ok(Fitted { placement, function })
        })
        .collect()
}

/// The highest target id on SCSI controller `controller`, if a card brings it.
pub fn card_scsi_max_id(placements: &[Placement], controller: u8) -> Option<u8> {
    placements.iter().find_map(|p| {
        let s = p.card.desc().scsi;
        (p.first_controller..p.first_controller.saturating_add(s.controllers))
            .contains(&controller)
            .then_some(s.max_id)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pci::slots::{Evidence, SlotDesc};

    fn entry(slot: SlotRef) -> PciEntry {
        PciEntry { slot, card: PciCard::Aic7880, bus: None, options: toml::Table::new() }
    }

    fn o2() -> [&'static BusDesc; 1] {
        [&crate::ip32::PCI_BUS]
    }

    fn err(buses: &[&'static BusDesc], entries: &[PciEntry]) -> String {
        resolve(buses, entries, 2).err().expect("an error")
    }

    #[test]
    fn the_option_slot_is_found_by_label_or_number() {
        for slot in [SlotRef::Label("PCI option".into()), SlotRef::Device(3), SlotRef::Label("pci OPTION".into())] {
            let p = resolve(&o2(), &[entry(slot)], 2).unwrap();
            assert_eq!((p[0].addr, p[0].slot.label), (PciAddr::dev(3), "PCI option"));
        }
    }

    #[test]
    fn a_card_refuses_an_option_it_does_not_have() {
        let mut e = entry(SlotRef::Device(3));
        e.options.insert("speed".into(), toml::Value::Integer(4));
        let m = err(&o2(), &[e]);
        assert!(m.starts_with("pci[0]: card \"aic7880\": ") && m.contains("speed"), "{m}");
    }

    #[test]
    fn a_slot_that_is_neither_label_nor_number_names_the_entry_and_the_real_slots() {
        for (text, shown) in [("true", "true"), ("300", "300"), ("-1", "-1"), ("1.5", "1.5")] {
            let f: toml::Table = toml::from_str(&format!("slot = {text}\ncard = \"aic7880\"")).unwrap();
            let e: PciEntry = f.try_into().unwrap();
            assert_eq!(e.slot, SlotRef::Invalid(shown.into()), "{text}");
            let m = err(&o2(), &[entry(SlotRef::Device(3)), e]);
            assert!(m.starts_with(&format!("pci[1]: slot must be a label or a device number from 0 to 255, not {shown}; this bus has: device 1 ")), "{m}");
            assert!(m.contains("device 3 \"PCI option\" (add-in)") && !m.contains('\n'), "{m}");
        }
    }

    #[test]
    fn no_buses_is_one_line() {
        assert_eq!(err(&[], &[entry(SlotRef::Device(3))]), "pci[0]: this machine has no PCI bus.");
    }

    #[test]
    fn an_onboard_device_is_not_a_slot() {
        let m = err(&o2(), &[entry(SlotRef::Device(1))]);
        assert!(m.starts_with("pci[0]: device 1 (\"internal SCSI\") is an onboard device, not a slot; this bus has: device 1 "), "{m}");
        assert!(m.contains("device 3 \"PCI option\" (add-in)") && m.ends_with('.'), "{m}");
        assert!(err(&o2(), &[entry(SlotRef::Label("external scsi".into()))]).contains("device 2 (\"external SCSI\") is an onboard device"));
    }

    #[test]
    fn a_paravirtual_spare_is_not_a_slot() {
        let m = err(&o2(), &[entry(SlotRef::Device(4))]);
        assert!(m.starts_with("pci[0]: device 4 (\"virtio-serial\") is reserved for a paravirtual device, not a slot; this bus has: "), "{m}");
    }

    #[test]
    fn an_absent_device_or_label_names_what_exists() {
        let m = err(&o2(), &[entry(SlotRef::Device(7))]);
        assert!(m.starts_with("pci[0]: bus \"pci\" has no device 7; this bus has: "), "{m}");
        let m = err(&o2(), &[entry(SlotRef::Label("left".into()))]);
        assert!(m.starts_with("pci[0]: bus \"pci\" has no slot labelled \"left\"; this bus has: "), "{m}");
    }

    #[test]
    fn a_bus_without_an_add_in_slot_says_so() {
        let m = err(&[&crate::ip30::PCI_BUS], &[entry(SlotRef::Device(2))]);
        assert!(m.starts_with("pci[0]: bus \"pci\" has no add-in slot; this bus has: device 0 \"SCSI 0\" (onboard)"), "{m}");
    }

    #[test]
    fn a_slot_can_be_taken_once() {
        let m = err(&o2(), &[entry(SlotRef::Device(3)), entry(SlotRef::Label("PCI option".into()))]);
        assert_eq!(m, "pci[1]: device 3 on bus \"pci\" is already taken by pci[0] (aic7880).");
    }

    #[test]
    fn a_slot_whose_evidence_is_thin_is_not_offered() {
        static THIN: BusDesc = BusDesc {
            id: "thin", name: "thin", devices: 4, outputs: 1,
            slots: &[SlotDesc { device: 1, label: "guess", kind: SlotKind::AddIn, evidence: Evidence::Inferred, source: "test", routes: [None; 4] }],
            wires: &[],
        };
        let m = err(&[&THIN], &[entry(SlotRef::Device(1))]);
        assert!(m.starts_with("pci[0]: device 1 (\"guess\") is not offered: its evidence is only Inferred; "), "{m}");
    }

    #[test]
    fn several_buses_need_naming() {
        let both: [&'static BusDesc; 2] = [&crate::ip32::PCI_BUS, &crate::ip30::PCI_BUS];
        assert!(err(&both, &[entry(SlotRef::Device(3))]).contains("several PCI buses (pci, pci)"));
        let mut e = entry(SlotRef::Device(3));
        e.bus = Some("nope".into());
        assert!(err(&both, &[e.clone()]).starts_with("pci[0]: no PCI bus \"nope\"; this machine has: "));
        e.bus = Some("pci".into());
        assert!(resolve(&both, &[e], 2).is_ok());
    }

    #[test]
    fn controllers_are_numbered_after_the_machines_own_in_entry_order() {
        static TWO: BusDesc = BusDesc {
            id: "two", name: "two", devices: 8, outputs: 1,
            slots: &[
                SlotDesc { device: 3, label: "a", kind: SlotKind::AddIn, evidence: Evidence::Established, source: "test", routes: [None; 4] },
                SlotDesc { device: 4, label: "b", kind: SlotKind::AddIn, evidence: Evidence::Established, source: "test", routes: [None; 4] },
            ],
            wires: &[],
        };
        let p = resolve(&[&TWO], &[entry(SlotRef::Device(4)), entry(SlotRef::Device(3))], 2).unwrap();
        assert_eq!((p[0].addr.device, p[0].first_controller), (4, 2));
        assert_eq!((p[1].addr.device, p[1].first_controller), (3, 3));
        assert_eq!(card_scsi_max_id(&p, 2), Some(15));
        assert_eq!(card_scsi_max_id(&p, 3), Some(15));
        assert_eq!(card_scsi_max_id(&p, 4), None);
        assert_eq!(card_scsi_max_id(&p, 1), None);
    }

    #[test]
    fn parse_cli_reads_bus_slot_card_and_options() {
        let e = PciEntry::parse_cli("pci:3=aic7880").unwrap();
        assert_eq!((e.bus.as_deref(), &e.slot, e.card), (Some("pci"), &SlotRef::Device(3), PciCard::Aic7880));
        let e = PciEntry::parse_cli("PCI option=aic7880").unwrap();
        assert_eq!((e.bus, e.slot), (None, SlotRef::Label("PCI option".into())));
        let e = PciEntry::parse_cli("3=AIC7880,depth=4,mode=fast,on=true").unwrap();
        assert_eq!(e.options.get("depth"), Some(&toml::Value::Integer(4)));
        assert_eq!(e.options.get("mode"), Some(&toml::Value::String("fast".into())));
        assert_eq!(e.options.get("on"), Some(&toml::Value::Boolean(true)));
        assert!(PciEntry::parse_cli("nonsense").is_err());
        assert!(PciEntry::parse_cli("3=").is_err());
        assert!(PciEntry::parse_cli("=aic7880").is_err());
        assert_eq!(PciEntry::parse_cli("3=nosuch").unwrap_err(), "no card \"nosuch\"; cards are: aic7880.");
        assert!(PciEntry::parse_cli("3=aic7880,depth").is_err());
    }

    #[test]
    fn merge_cli_replaces_the_same_slot_whatever_its_spelling() {
        let mut entries = vec![entry(SlotRef::Label("PCI option".into()))];
        let mut cli = entry(SlotRef::Device(3));
        cli.options.insert("from".into(), toml::Value::String("cli".into()));
        merge_cli(&o2(), &mut entries, &[cli.clone()]);
        assert_eq!(entries, vec![cli]);
        merge_cli(&o2(), &mut entries, &[entry(SlotRef::Device(7))]);
        assert_eq!(entries.len(), 2, "an unrelated slot is added");
    }

    #[test]
    fn toml_entries_round_trip_with_options_beside_slot_and_card() {
        #[derive(Serialize, Deserialize, PartialEq, Debug)]
        struct File {
            pci: Vec<PciEntry>,
        }
        let text = "[[pci]]\nslot = \"PCI option\"\ncard = \"aic7880\"\ndepth = 4\n\n[[pci]]\nslot = 3\ncard = \"aic7880\"\nbus = \"pci\"\n";
        let f: File = toml::from_str(text).unwrap();
        assert_eq!(f.pci[0].slot, SlotRef::Label("PCI option".into()));
        assert_eq!(f.pci[0].options.get("depth"), Some(&toml::Value::Integer(4)));
        assert_eq!((&f.pci[1].slot, f.pci[1].bus.as_deref()), (&SlotRef::Device(3), Some("pci")));
        assert_eq!(toml::from_str::<File>(&toml::to_string(&f).unwrap()).unwrap(), f);
    }

    #[test]
    fn build_makes_each_function_with_its_first_controller() {
        let p = resolve(&o2(), &[entry(SlotRef::Device(3))], 2).unwrap();
        let card_ref = p[0].card_ref();
        assert_eq!((card_ref.name.as_str(), card_ref.first_controller, card_ref.options.as_str()), ("aic7880", 2, ""));
        let fitted = build(p).unwrap();
        assert_eq!(fitted[0].placement.addr, PciAddr::dev(3));
        assert_eq!(fitted[0].function.config().read(0), 0x8078_9004);
        assert_eq!(fitted[0].function.scsi_buses().len(), 1);
    }
}
