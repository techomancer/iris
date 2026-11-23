//! The machine slots: each machine's slots, declared as data by [`MachineProfile::slots`], which builds no hardware.
//! The config check, `--list-slots` and the GUI read that list; [`DevicePlacement`] checks a slot exists, takes the device and is free.

use std::collections::HashMap;

use crate::config::{MachineConfig, MachineProfile};
use crate::devices::ConfiguredDevice;
use crate::slot_path::MachineSlotPath;
use crate::pci::PciBus;

/// The kind of a device, which decides the slots that take it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceFamily {
    /// A CD-ROM or DaynaPort on a SCSI bus.
    Scsi,
    Pci,
}

impl DeviceFamily {
    fn label(self) -> &'static str {
        match self {
            DeviceFamily::Scsi => "scsi devices",
            DeviceFamily::Pci => "pci cards",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum MachineSlotKind {
    /// A built-in part that takes nothing itself.
    Fixed(&'static str),
    /// A SCSI controller (or channel), alias `scsiN` in list order, with
    /// targets 1 to `ids` as slots below it.
    ScsiBus { part: &'static str, ids: u8 },
    /// A board's PCI bus, its devices as slots below it.
    PciBus(&'static str),
    /// A slot that takes devices of these families; none for a slot whose board nothing models.
    Open(&'static [DeviceFamily]),
}

impl MachineSlotKind {
    /// A slot for a PCI card.
    pub const PCI_SLOT: MachineSlotKind = MachineSlotKind::Open(&[DeviceFamily::Pci]);

    /// Whether a device of `family` fits.
    pub fn takes(&self, family: DeviceFamily) -> bool {
        matches!(self, MachineSlotKind::Open(families) if families.contains(&family))
    }

    fn label(&self) -> String {
        match self {
            MachineSlotKind::Open([]) => "nothing modelled".into(),
            MachineSlotKind::Open(families) => {
                let mut labels: Vec<_> = families.iter().map(|k| k.label()).collect();
                labels.dedup();
                labels.join(", ")
            }
            _ => "-".into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct MachineSlot {
    pub path: MachineSlotPath,
    pub kind: MachineSlotKind,
    pub aliases: Vec<String>,
}

fn slot_path(at: &str) -> MachineSlotPath {
    at.parse().expect("a slot path")
}

/// A slot goes after the last slot under its parent, so a device's slots
/// list beside it.
fn insert_slot(slots: &mut Vec<MachineSlot>, path: MachineSlotPath, kind: MachineSlotKind, aliases: Vec<String>) {
    assert!(!slots.iter().any(|s| s.path == path), "{path} declared twice");
    let parent = path.parent().unwrap_or_default();
    let i = slots.iter().rposition(|s| s.path.starts_with(&parent)).map_or(slots.len(), |i| i + 1);
    slots.insert(i, MachineSlot { path, kind, aliases });
}

pub fn fixed_slot(slots: &mut Vec<MachineSlot>, at: &str, part: &'static str, aliases: &[&str]) {
    insert_slot(slots, slot_path(at), MachineSlotKind::Fixed(part), aliases.iter().map(|a| a.to_string()).collect());
}

/// The next SCSI controller at `at`, with targets 1 to `ids`.
pub fn scsi_bus_slots(slots: &mut Vec<MachineSlot>, at: &str, part: &'static str, ids: u8) {
    let at = slot_path(at);
    let n = slots.iter().filter(|s| matches!(s.kind, MachineSlotKind::ScsiBus { .. })).count();
    insert_slot(slots, at.clone(), MachineSlotKind::ScsiBus { part, ids }, vec![format!("scsi{n}")]);
    for id in 1..=ids {
        insert_slot(slots, at.target(id), MachineSlotKind::Open(&[DeviceFamily::Scsi]), vec![]);
    }
}

/// A board's PCI bus at `at`, with a slot for each device its table lists and
/// SGI's name for it, where SGI gives one.
pub fn pci_bus_slots(slots: &mut Vec<MachineSlot>, at: &str, name: &'static str, table: &[(u8, MachineSlotKind, Option<&'static str>)]) {
    let at = slot_path(at);
    insert_slot(slots, at.clone(), MachineSlotKind::PciBus(name), vec![]);
    for (d, kind, alias) in table {
        let dev = at.dev(*d);
        match kind {
            MachineSlotKind::ScsiBus { part, ids } => scsi_bus_slots(slots, &dev.to_string(), part, *ids),
            kind => insert_slot(slots, dev, kind.clone(), alias.map(String::from).into_iter().collect()),
        }
    }
}

/// The SCSI controllers, by number, with their target ids.
pub fn scsi_buses(slots: &[MachineSlot]) -> Vec<(u8, MachineSlotPath, Vec<u8>)> {
    let buses = slots.iter().filter_map(|s| match s.kind {
        MachineSlotKind::ScsiBus { ids, .. } => Some((s.path.clone(), (1..=ids).collect())),
        _ => None,
    });
    buses.enumerate().map(|(n, (at, ids))| (n as u8, at, ids)).collect()
}

/// (controller, id) of a SCSI target slot.
pub fn scsi_target(slots: &[MachineSlot], at: &MachineSlotPath) -> Option<(u8, u8)> {
    let id = at.target_id()?;
    let bus = at.parent()?;
    scsi_buses(slots).into_iter().find(|b| b.1 == bus).map(|b| (b.0, id))
}

/// The target slot for `id` on controller `n`.
pub fn scsi_target_slot(slots: &[MachineSlot], n: u8, id: u8) -> Option<MachineSlotPath> {
    let at = scsi_buses(slots).into_iter().find(|b| b.0 == n)?.1.target(id);
    slots.iter().any(|s| s.path == at).then_some(at)
}

pub(crate) fn slot_alias<'a>(slots: &'a [MachineSlot], name: &str) -> Option<&'a MachineSlotPath> {
    slots.iter().find(|s| s.aliases.iter().any(|a| a == name)).map(|s| &s.path)
}

fn machine_name(profile: MachineProfile) -> &'static str {
    profile.product_name().unwrap_or(profile.label())
}

/// A slot path written with a machine alias or a local name first, as a path.
fn expand(
    machine: &str,
    slots: &[MachineSlot],
    at: &str,
    names: &mut dyn FnMut(&str) -> Result<Option<MachineSlotPath>, String>,
) -> Result<MachineSlotPath, String> {
    let (first, rest) = at.split_once('/').unwrap_or((at, ""));
    let base = match slot_alias(slots, first) {
        Some(l) => l.clone(),
        None => match names(first)? {
            Some(l) => l,
            None if !MachineSlotPath::is_step(first) => return Err(format!("the {machine} has no slot {at}")),
            None => return at.parse(),
        },
    };
    Ok(if rest.is_empty() { base } else { base.extend(&rest.parse()?) })
}

impl MachineProfile {
    /// The slots of the machine, in the order automatic placement tries them.
    pub fn slots(self) -> Vec<MachineSlot> {
        use MachineProfile as P;
        let mut s = Vec::new();
        match self {
            P::IndyIp24 | P::Indigo2Ip22 | P::Indigo2Ip28 => {
                crate::dev::hpc3::Hpc3::slots(&mut s, self.guinness());
            }
            P::O2Ip32 => crate::ip32_machine::slots(&mut s),
            P::OctaneIp30 => crate::ip30_machine::slots(&mut s),
            P::Origin200Ip29 | P::Origin2000Ip27 | P::Onyx2Ip27 => crate::ip27::slots(&mut s, self),
            P::FuelIp34 => crate::ip34_machine::slots(&mut s),
            P::TezroIp53 => crate::ip53_machine::slots(&mut s),
            // Their disks are on the I-brick's Fibre Channel loop.
            P::Origin3200Ip35 | P::Onyx3000Ip35 => {}
        }
        s
    }
}

/// A config placed on its machine's slots: where each device went, the slots grown by what devices brought, and the PCI buses
/// the constructor built. Constructors make one and every attach reads it, so placement runs once.
pub struct DevicePlacement {
    pub slots: Vec<MachineSlot>,
    machine: &'static str,
    devices: Vec<ConfiguredDevice>,
    /// Each device's slot, or why it has none; a default device's failure is not an error.
    at: Vec<Result<MachineSlotPath, String>>,
    pci: HashMap<MachineSlotPath, PciBus>,
}

impl DevicePlacement {
    /// The config's own devices, each in a slot of its machine: the first error of a device that is not a default.
    pub fn new(cfg: &MachineConfig) -> Result<DevicePlacement, String> {
        let p = DevicePlacement::lenient(cfg, &cfg.devices);
        match p.devices.iter().zip(&p.at).find_map(|(d, at)| at.as_ref().err().filter(|_| !d.default)) {
            Some(e) => Err(e.clone()),
            None => Ok(p),
        }
    }

    /// `devices` on `cfg`'s machine, each in its slot where it has one, whatever else is wrong: what an editor shows mid-edit.
    /// The graphics board no entry names claims its board's default slot, as a named device would.
    pub fn lenient(cfg: &MachineConfig, devices: &[ConfiguredDevice]) -> DevicePlacement {
        let machine = machine_name(cfg.machine.profile);
        let (slots, at) = layout(machine, cfg.machine.profile.slots(), devices);
        DevicePlacement { slots, machine, devices: devices.to_vec(), at, pci: HashMap::new() }
    }

    pub fn machine(&self) -> &'static str {
        self.machine
    }

    /// Where device `i` went, if it has a slot.
    pub fn at(&self, i: usize) -> Option<&MachineSlotPath> {
        self.at[i].as_ref().ok()
    }

    /// The devices that have a slot, with it, in config order.
    pub fn placed(&self) -> impl Iterator<Item = (&ConfiguredDevice, &MachineSlotPath)> {
        self.devices.iter().zip(&self.at).filter_map(|(d, at)| Some((d, at.as_ref().ok()?)))
    }

    /// The slot with this path.
    pub fn slot(&self, at: &MachineSlotPath) -> Option<&MachineSlot> {
        self.slots.iter().find(|s| &s.path == at)
    }

    /// Record the PCI bus the machine built at the PCI bus slot `at`.
    pub fn bind_pci(&mut self, at: &str, bus: PciBus) -> Result<(), String> {
        let at: MachineSlotPath = at.parse()?;
        if !matches!(self.slot(&at).map(|s| &s.kind), Some(MachineSlotKind::PciBus(_))) {
            return Err(format!("the {} built a PCI bus at {at}, which its slot list does not declare", self.machine));
        }
        match self.pci.insert(at.clone(), bus) {
            Some(_) => Err(format!("the {} built two PCI buses at {at}", self.machine)),
            None => Ok(()),
        }
    }

    pub(crate) fn pci(&self, at: &MachineSlotPath) -> Option<&PciBus> {
        self.pci.get(at)
    }

    /// Every PCI bus slot has its bus.
    pub(crate) fn check_pci(&self) -> Result<(), String> {
        match self.slots.iter().find(|s| matches!(s.kind, MachineSlotKind::PciBus(_)) && !self.pci.contains_key(&s.path)) {
            Some(s) => Err(format!("the {}'s slot list declares a PCI bus at {}, which the machine did not build", self.machine, s.path)),
            None => Ok(()),
        }
    }

    /// `--list-slots`: every slot, what is in it and what it takes. Runs of
    /// free targets on one bus share a line.
    pub fn list_slots(&self) -> String {
        let has = |at: &MachineSlotPath| (0..self.devices.len()).find(|&i| self.at(i) == Some(at)).map(|i| {
            let d = &self.devices[i];
            let name = d.name.as_ref().map(|n| format!("{n}: ")).unwrap_or_default();
            let note = if d.default { " (default)" } else { "" };
            format!("{name}{}{note}", d.device.summary())
        });
        let alias = |s: &MachineSlot| -> String {
            if !s.aliases.is_empty() {
                return s.aliases.join(", ");
            }
            let parent = s.path.parent().and_then(|p| self.slot(&p).and_then(|p| p.aliases.first().cloned()));
            parent.map(|p| format!("{p}/{}", s.path.to_string().rsplit('/').next().unwrap_or(""))).unwrap_or_else(|| "-".into())
        };
        let mut lines = vec![["SLOT".to_string(), "ALIAS".into(), "HAS".into(), "TAKES".into()]];
        let mut free: Option<(usize, u8)> = None; // the line of a run of free targets and its first id
        for s in &self.slots {
            let line = match &s.kind {
                MachineSlotKind::Fixed(part) | MachineSlotKind::ScsiBus { part, .. } => [s.path.to_string(), alias(s), format!("{part} (onboard)"), "-".into()],
                MachineSlotKind::PciBus(name) => [s.path.to_string(), alias(s), name.to_string(), "-".into()],
                MachineSlotKind::Open(_) => match has(&s.path) {
                    Some(h) => [s.path.to_string(), alias(s), h, "-".into()],
                    // A slot with its own name, or at the root, has a line to itself.
                    None if !s.aliases.is_empty() || s.path.depth() < 2 => [s.path.to_string(), alias(s), "-".into(), s.kind.label()],
                    None => {
                        let id = s.path.last_number().unwrap_or(0);
                        let (at, al) = (s.path.to_string(), alias(s));
                        let target = s.path.target_id().is_some();
                        if let Some((n, first)) = free.filter(|&(n, _)| target && lines[n][0].rsplit_once('/').map(|p| p.0) == at.rsplit_once('/').map(|p| p.0)) {
                            lines[n][0] = format!("{}-{id}", at_first(&at, first));
                            lines[n][1] = if al == "-" { al } else { format!("{}-{id}", at_first(&al, first)) };
                            free = Some((n, first));
                            continue;
                        }
                        free = target.then_some((lines.len(), id));
                        lines.push([at, al, "-".into(), s.kind.label()]);
                        continue;
                    }
                },
            };
            free = None;
            lines.push(line);
        }
        let w: Vec<usize> = (0..3).map(|c| lines.iter().map(|l| l[c].len()).max().unwrap_or(0)).collect();
        lines.iter().map(|l| format!("{:w0$}  {:w1$}  {:w2$}  {}\n", l[0], l[1], l[2], l[3], w0 = w[0], w1 = w[1], w2 = w[2])).collect()
    }
}

/// How a device claims a slot: by a full path, by naming a bus, or anywhere.
#[derive(Clone, Copy, PartialEq, PartialOrd)]
enum SlotRequest {
    Full,
    Bus,
    Any,
}

/// Where each device goes, and the slots grown by what the placed devices bring: full paths claim first, then devices naming a
/// bus, then the rest, each in device order. A claim restarts the scan; a device with no room waits for slots a later one brings.
fn layout(machine: &str, slots: Vec<MachineSlot>, devices: &[ConfiguredDevice]) -> (Vec<MachineSlot>, Vec<Result<MachineSlotPath, String>>) {
    let mut names: HashMap<&str, usize> = HashMap::new();
    let mut out: Vec<Option<Result<MachineSlotPath, String>>> = vec![None; devices.len()];
    for (i, d) in devices.iter().enumerate() {
        let Some(n) = &d.name else { continue };
        let bad = match check_name(n, machine, &slots) {
            Err(e) => Some(format!("[[devices]] {e}")),
            Ok(()) if names.contains_key(n.as_str()) => Some(format!("[[devices]] two devices are named \"{n}\"")),
            Ok(()) => {
                names.insert(n, i);
                None
            }
        };
        out[i] = bad.map(Err);
    }
    let mut taken: HashMap<MachineSlotPath, usize> = HashMap::new();
    for defaults in [false, true] {
        'claim: loop {
            for class in [SlotRequest::Full, SlotRequest::Bus, SlotRequest::Any] {
                for i in (0..devices.len()).filter(|&i| devices[i].default == defaults) {
                    let d = &devices[i];
                    if out[i].is_some() {
                        continue;
                    }
                    let first = d.at.as_deref().and_then(|at| at.split('/').next()).unwrap_or("");
                    if slot_alias(&slots, first).is_none() && names.get(first).is_some_and(|&j| !matches!(out[j], Some(Ok(_)))) {
                        continue;
                    }
                    let name_at = |f: &str| Ok(names.get(f).and_then(|&j| out[j].as_ref()?.as_ref().ok().cloned()));
                    let base = d.at.as_ref().map(|at| expand(machine, &slots, at, &mut |f| name_at(f)));
                    let slot = base.as_ref().and_then(|b| b.as_ref().ok()).and_then(|l| slots.iter().find(|s| &s.path == l));
                    // A slot not in the list yet may come with a device placed later.
                    if slot.is_none() && base.is_some() {
                        continue;
                    }
                    let own = match (&base, slot) {
                        (None, _) => SlotRequest::Any,
                        (_, Some(s)) if !matches!(s.kind, MachineSlotKind::Open(_)) => SlotRequest::Bus,
                        _ => SlotRequest::Full,
                    };
                    if own > class {
                        continue;
                    }
                    let ty = d.device.spec();
                    let got = match (own, base) {
                        (SlotRequest::Full, Some(Ok(loc))) => match slot {
                            Some(s) if !s.kind.takes(ty.family) => Err(format!("{loc} does not take a {}", ty.name)),
                            _ => match taken.get(&loc) {
                                Some(&j) => Err(format!("{loc} is taken by the {} there", devices[j].device.spec().name)),
                                None => Ok(loc),
                            },
                        },
                        (_, under) => {
                            let under = under.map(|b| b.expect("checked above"));
                            let free = slots.iter().find(|s| {
                                s.kind.takes(ty.family) && !taken.contains_key(&s.path) && under.as_ref().is_none_or(|b| s.path.starts_with(b))
                            });
                            match free {
                                Some(s) => Ok(s.path.clone()),
                                // The slots a later device brings may have room.
                                None => continue,
                            }
                        }
                    };
                    match got {
                        Ok(loc) => {
                            taken.insert(loc.clone(), i);
                            out[i] = Some(Ok(loc));
                            continue 'claim;
                        }
                        Err(e) => out[i] = Some(Err(device_error(d, &e))),
                    }
                }
            }
            break;
        }
    }
    // What is left names a slot no device brought, waits on itself, or found no room.
    let left: Vec<usize> = (0..devices.len()).filter(|&i| out[i].is_none()).collect();
    for i in left {
        let d = &devices[i];
        let at = d.at.as_deref().unwrap_or("");
        let first = at.split('/').next().unwrap_or("");
        let waits = slot_alias(&slots, first).is_none() && names.get(first).is_some_and(|&j| !matches!(out[j], Some(Ok(_))));
        let name_at = |f: &str| Ok(names.get(f).and_then(|&j| out[j].as_ref()?.as_ref().ok().cloned()));
        let e = match d.at.as_ref().map(|at| expand(machine, &slots, at, &mut |f| name_at(f))) {
            _ if waits => "its slot names itself through local names".to_string(),
            Some(Err(e)) => e,
            Some(Ok(base)) if !slots.iter().any(|s| s.path == base) => format!("the {machine} has no slot {at}"),
            base => {
                let on = base.map_or(format!("the {machine}"), |b| b.expect("checked above").to_string());
                format!("no free slot for {} on {on}", d.device.spec().family.label())
            }
        };
        out[i] = Some(Err(device_error(d, &e)));
    }
    (slots, out.into_iter().map(|o| o.expect("every device has a result")).collect())
}

fn device_error(d: &ConfiguredDevice, e: &str) -> String {
    match &d.at {
        Some(at) => format!("[[devices]] {} at \"{at}\": {e}", d.device.spec().name),
        None => format!("[[devices]] {}: {e}", d.device.spec().name),
    }
}

/// A local name uses lowercase letters, digits, `_` and `-`, does not look
/// like a path step, and does not shadow a machine alias. Uniqueness is the caller's.
fn check_name(name: &str, machine: &str, slots: &[MachineSlot]) -> Result<(), String> {
    if name.is_empty() || !name.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-') {
        return Err(format!("name \"{name}\": use lowercase letters, digits, _ and -"));
    }
    if MachineSlotPath::is_step(name) {
        return Err(format!("name \"{name}\" looks like a slot path step"));
    }
    if slot_alias(slots, name).is_some() {
        return Err(format!("name \"{name}\" shadows the {machine}'s slot {name}"));
    }
    Ok(())
}

/// `a/b/tN` with its last step's number set to `first`.
fn at_first(at: &str, first: u8) -> String {
    let (head, last) = at.rsplit_once('/').unwrap_or(("", at));
    let tag: String = last.chars().take_while(|c| !c.is_ascii_digit()).collect();
    if head.is_empty() { format!("{tag}{first}") } else { format!("{head}/{tag}{first}") }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MachineConfig;
    use crate::devices::DeviceModel;
    use crate::pci::PciBus;
    use std::sync::Arc;

    /// A machine under test.
    struct Map(MachineConfig);

    fn fill(p: MachineProfile) -> Map {
        let mut cfg = MachineConfig::default();
        cfg.machine.profile = p;
        Map(cfg)
    }

    impl Map {
        fn slots(&self) -> Vec<MachineSlot> { self.0.machine.profile.slots() }
        fn alias(&self, name: &str) -> Option<MachineSlotPath> { slot_alias(&self.slots(), name).cloned() }
        fn row(&self, at: &MachineSlotPath) -> Option<MachineSlot> { self.slots().into_iter().find(|s| &s.path == at) }
        fn scsi_buses(&self) -> Vec<(u8, MachineSlotPath, Vec<u8>)> { scsi_buses(&self.slots()) }
        fn scsi_target(&self, at: &MachineSlotPath) -> Option<(u8, u8)> { scsi_target(&self.slots(), at) }
        fn scsi_target_slot(&self, n: u8, id: u8) -> Option<MachineSlotPath> { scsi_target_slot(&self.slots(), n, id) }
    }

    fn place(devs: &[ConfiguredDevice], map: &Map) -> Result<Vec<Option<MachineSlotPath>>, String> {
        let mut cfg = map.0.clone();
        cfg.devices = devs.to_vec();
        let p = DevicePlacement::new(&cfg)?;
        Ok((0..devs.len()).map(|i| p.at(i).cloned()).collect())
    }

    fn locate(devs: &[ConfiguredDevice], map: &Map) -> Vec<Option<MachineSlotPath>> {
        let p = DevicePlacement::lenient(&map.0, devs);
        (0..devs.len()).map(|i| p.at(i).cloned()).collect()
    }

    fn list_slots(devs: &[ConfiguredDevice], map: &Map) -> Result<String, String> {
        let mut cfg = map.0.clone();
        cfg.devices = devs.to_vec();
        DevicePlacement::new(&cfg).map(|p| p.list_slots())
    }

    fn hdd(at: &str, path: &str) -> ConfiguredDevice {
        ConfiguredDevice::new(at, DeviceModel::Scsi(crate::config::ScsiDeviceConfig { path: path.into(), ..Default::default() }))
    }

    fn loc(s: &str) -> MachineSlotPath { s.parse().unwrap() }

    #[test]
    fn every_map_has_unique_aliases_that_are_not_hops() {
        for p in MachineProfile::EVERY {
            let map = fill(p);
            let mut seen = std::collections::HashSet::new();
            for a in map.slots().iter().flat_map(|r| r.aliases.clone()) {
                assert!(!MachineSlotPath::is_step(&a), "{p:?}: alias {a} looks like a hop");
                assert!(seen.insert(a.clone()), "{p:?}: alias {a} twice");
            }
        }
    }

    #[test]
    fn locate_follows_an_edit_to_the_devices() {
        let map = fill(MachineProfile::OctaneIp30);
        let mut devs = vec![hdd("scsi0/t1", "a")];
        assert_eq!(locate(&devs, &map), [Some(loc("xio15/pci0/0/t1"))]);
        devs[0].at = Some("scsi1/t4".into());
        assert_eq!(locate(&devs, &map), [Some(loc("xio15/pci0/1/t4"))]);
        assert_eq!(locate(&devs, &fill(MachineProfile::IndyIp24)), [None]);
    }

    #[test]
    fn scsi_aliases_name_each_machines_controllers() {
        use MachineProfile as P;
        for (p, want) in [
            (P::IndyIp24, &["hpc3/c0"][..]),
            (P::Indigo2Ip22, &["hpc3/c0", "hpc3/c1"]),
            (P::O2Ip32, &["pci0/1", "pci0/2"]),
            (P::OctaneIp30, &["xio15/pci0/0", "xio15/pci0/1"]),
            (P::Origin200Ip29, &["xio8/pci0/0", "xio8/pci0/1"]),
            (P::Origin2000Ip27, &["xio8/pci0/0", "xio8/pci0/1"]),
            (P::Onyx2Ip27, &["xio15/pci0/0", "xio15/pci0/1"]),
            (P::FuelIp34, &["xio15/pci0/1/c0", "xio15/pci0/1/c1"]),
            (P::TezroIp53, &["xio15/pci0/3/c0", "xio15/pci0/3/c1"]),
            (P::Origin3200Ip35, &[]),
        ] {
            let got: Vec<String> = fill(p).scsi_buses().iter().map(|b| b.1.to_string()).collect();
            assert_eq!(got, want, "{p:?}");
            for (n, at) in want.iter().enumerate() {
                assert_eq!(fill(p).alias(&format!("scsi{n}")).map(|l| l.to_string()).as_deref(), Some(*at));
            }
        }
        assert_eq!(fill(P::OctaneIp30).scsi_buses()[1].2, (1..=15).collect::<Vec<u8>>());
        assert_eq!(fill(P::IndyIp24).scsi_buses()[0].2, (1..=7).collect::<Vec<u8>>());
    }

    #[test]
    fn a_board_table_gives_the_rows_of_its_bus() {
        let o2 = fill(MachineProfile::O2Ip32);
        assert_eq!(o2.alias("pci_slot"), Some(loc("pci0/3")));
        assert_eq!(o2.row(&loc("pci0/3")).map(|r| r.kind), Some(MachineSlotKind::PCI_SLOT));
        let octane = fill(MachineProfile::OctaneIp30);
        assert_eq!(octane.row(&loc("xio15/pci0/2")).map(|r| r.kind), Some(MachineSlotKind::Fixed("ioc3")));
        assert!(matches!(octane.row(&loc("xio15/pci0")).map(|r| r.kind), Some(MachineSlotKind::PciBus(_))));
    }

    #[test]
    fn a_slot_resolves_through_an_alias_a_name_or_a_path() {
        let map = fill(MachineProfile::OctaneIp30);
        let mut named = hdd("xio15/pci0/1/t4", "");
        named.name = Some("cd".into());
        let devs = [hdd("scsi0/t1", "a.raw"), named];
        let got = place(&devs, &map).unwrap();
        assert_eq!(got, [Some(loc("xio15/pci0/0/t1")), Some(loc("xio15/pci0/1/t4"))]);
        assert_eq!(map.scsi_target(&loc("xio15/pci0/1/t4")), Some((1, 4)));
        assert_eq!(map.scsi_target_slot(1, 4), Some(loc("xio15/pci0/1/t4")));
        // A name stands for the slot of the device it names; a disk has no slots below it.
        let e = place(&[devs[0].clone(), devs[1].clone(), hdd("cd/t1", "")], &map).unwrap_err();
        assert!(e.contains("has no slot cd/t1"), "{e}");
    }

    #[test]
    fn a_clash_a_missing_slot_or_a_wrong_family_names_the_slot() {
        let map = fill(MachineProfile::IndyIp24);
        let e = place(&[hdd("scsi0/t1", "a"), hdd("hpc3/c0/t1", "b")], &map).unwrap_err();
        assert_eq!(e, "[[devices]] hdd at \"hpc3/c0/t1\": hpc3/c0/t1 is taken by the hdd there");
        let e = place(&[hdd("scsi1/t1", "a")], &map).unwrap_err();
        assert!(e.contains("has no slot scsi1/t1"), "{e}");
        let e = place(&[hdd("scsi0/t8", "a")], &map).unwrap_err();
        assert!(e.contains("no slot scsi0/t8"), "narrow bus: {e}");
        let e = place(&[hdd("scsi0/t1", "a")], &fill(MachineProfile::Origin3200Ip35)).unwrap_err();
        assert!(e.contains("no slot scsi0/t1"), "{e}");
        assert!(place(&[hdd("nowhere", "a")], &map).unwrap_err().contains("has no slot nowhere"));
        assert!(place(&[hdd("hpc3/c0/x1", "a")], &map).unwrap_err().contains("not a slot path step"));
    }

    #[test]
    fn names_are_unique_and_a_cycle_is_an_error() {
        let map = fill(MachineProfile::OctaneIp30);
        let mut a = hdd("b/t1", "");
        a.name = Some("a".into());
        let mut b = hdd("a/t1", "");
        b.name = Some("b".into());
        assert!(place(&[a.clone(), b.clone()], &map).unwrap_err().contains("names itself"));
        let (a, b) = (ConfiguredDevice { name: Some("a".into()), ..hdd("scsi0/t1", "") }, ConfiguredDevice { name: Some("a".into()), ..hdd("scsi0/t2", "") });
        assert!(place(&[a, b], &map).unwrap_err().contains("two devices are named \"a\""));
        let mut s = hdd("scsi0/t1", "");
        s.name = Some("scsi0".into());
        assert!(place(&[s], &map).unwrap_err().contains("shadows"), "a local name may not shadow a machine alias");
    }

    fn auto(at: Option<&str>) -> ConfiguredDevice {
        ConfiguredDevice { at: at.map(Into::into), ..hdd("", "") }
    }

    fn placed(devs: &[ConfiguredDevice], p: MachineProfile) -> Vec<String> {
        place(devs, &fill(p)).unwrap().into_iter().map(|l| l.unwrap().to_string()).collect()
    }

    #[test]
    fn full_slots_claim_first_then_buses_then_the_rest_and_the_answer_is_stable() {
        // Entry order is the reverse of claim order.
        let devs = [auto(None), auto(Some("scsi1")), auto(Some("scsi0/t1")), auto(None)];
        let want = ["xio15/pci0/0/t2", "xio15/pci0/1/t1", "xio15/pci0/0/t1", "xio15/pci0/0/t3"];
        assert_eq!(placed(&devs, MachineProfile::OctaneIp30), want);
        assert_eq!(placed(&devs, MachineProfile::OctaneIp30), want);
    }

    #[test]
    fn unplaced_devices_take_the_first_free_target_in_array_order() {
        let mut devs = vec![auto(None), auto(None)];
        devs.insert(1, hdd("scsi0/t2", "taken"));
        assert_eq!(placed(&devs, MachineProfile::IndyIp24), ["hpc3/c0/t1", "hpc3/c0/t2", "hpc3/c0/t3"]);
    }

    #[test]
    fn a_full_bus_or_machine_names_the_family() {
        let seven: Vec<ConfiguredDevice> = (1..=7).map(|n| auto(Some(&format!("scsi0/t{n}")))).collect();
        assert_eq!(placed(&seven, MachineProfile::IndyIp24).len(), 7);
        let map = fill(MachineProfile::IndyIp24);
        let e = place(&[seven.clone(), vec![auto(None)]].concat(), &map).unwrap_err();
        assert_eq!(e, "[[devices]] hdd: no free slot for scsi devices on the SGI Indy (IP24)");
        let e = place(&[seven, vec![auto(Some("scsi0"))]].concat(), &map).unwrap_err();
        assert_eq!(e, "[[devices]] hdd at \"scsi0\": no free slot for scsi devices on hpc3/c0");
    }

    #[test]
    fn a_default_disk_yields_to_a_machine_without_its_slot() {
        let mut d = hdd("scsi0/t1", "scsi1.raw");
        d.default = true;
        assert_eq!(place(&[d.clone()], &fill(MachineProfile::Origin3200Ip35)).unwrap(), [None]);
        assert_eq!(place(&[d.clone(), hdd("scsi0/t1", "mine")], &fill(MachineProfile::IndyIp24)).unwrap(), [None, Some(loc("hpc3/c0/t1"))]);
    }

    #[test]
    fn a_built_pci_bus_is_bound_to_a_declared_row_once_and_every_row_needs_one() {
        use crate::pci::testing::TestHost;
        let host = Arc::new(TestHost::default());
        let bus = || PciBus::new(0, &crate::dev::bridge::PIN_WIRING, Arc::downgrade(&host) as std::sync::Weak<dyn crate::pci::PciHost>);
        let mut cfg = MachineConfig::default();
        cfg.machine.profile = MachineProfile::OctaneIp30;
        let mut r = DevicePlacement::new(&cfg).unwrap();
        assert!(r.check_pci().unwrap_err().contains("declares a PCI bus at xio15/pci0"));
        r.bind_pci("xio15/pci0", bus()).unwrap();
        assert!(r.bind_pci("xio15/pci0", bus()).unwrap_err().contains("two PCI buses"));
        assert!(r.bind_pci("xio13/pci0", bus()).unwrap_err().contains("does not declare"));
        r.check_pci().unwrap();
        assert!(r.pci(&loc("xio15/pci0")).is_some());
    }

    #[test]
    fn list_slots_shows_aliases_contents_and_free_runs() {
        let mut cd = hdd("scsi0/t4", "");
        cd.device = DeviceModel::Scsi(crate::config::ScsiDeviceConfig { cdrom: true, kind_field: crate::config::ScsiKind::Cdrom, ..Default::default() });
        cd.default = true;
        let text = list_slots(&[hdd("scsi0/t1", "irix.img"), cd], &fill(MachineProfile::IndyIp24)).unwrap();
        let rows: Vec<Vec<&str>> = text.lines().map(|l| l.split("  ").map(str::trim).filter(|s| !s.is_empty()).collect()).collect();
        assert_eq!(rows, [
            vec!["SLOT", "ALIAS", "HAS", "TAKES"],
            vec!["hpc3", "-", "hpc3 (onboard)", "-"],
            vec!["hpc3/c0", "scsi0", "wd33c93a (onboard)", "-"],
            vec!["hpc3/c0/t1", "scsi0/t1", "hdd irix.img", "-"],
            vec!["hpc3/c0/t2-3", "scsi0/t2-3", "-", "scsi devices"],
            vec!["hpc3/c0/t4", "scsi0/t4", "cdrom (default)", "-"],
            vec!["hpc3/c0/t5-7", "scsi0/t5-7", "-", "scsi devices"],
        ]);
    }

    #[test]
    fn names_follow_the_rule() {
        let slots = MachineProfile::OctaneIp30.slots();
        for good in ["boot", "disk-2", "my_cd", "scsi", "cage"] {
            check_name(good, "Octane", &slots).unwrap_or_else(|e| panic!("{good}: {e}"));
        }
        for hop in ["pci2", "xio15", "t1", "c0", "usb0", "node1", "hpc3", "gfx", "l3", "7", "exp0"] {
            assert!(check_name(hop, "Octane", &slots).unwrap_err().contains("looks like a slot path step"), "{hop}");
        }
        for bad in ["Boot", "a b", "", "d.1"] {
            assert!(check_name(bad, "Octane", &slots).unwrap_err().contains("lowercase"), "{bad}");
        }
        assert!(check_name("scsi1", "Octane", &slots).unwrap_err().contains("shadows"));
    }
}
