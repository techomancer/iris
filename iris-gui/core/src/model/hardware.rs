//! Pure facts about machine profiles and their configs that every shell needs:
//! which family a profile belongs to, how SCSI targets are named, how a config
//! is refitted when the profile changes. No UI.

use iris::config::*;
use iris::devices::DeviceModel;
use iris::topology::{scsi_target, scsi_target_slot, DeviceFamily, DevicePlacement, MachineSlotKind};

/// The Indy and Indigo2 family: the machines built around the MC memory
/// controller, with Newport resolution presets, a DS1386 NVRAM and (on the
/// Indigo2) a motherboard EEPROM.
pub fn mc_machine(profile: MachineProfile) -> bool {
    matches!(profile, MachineProfile::IndyIp24 | MachineProfile::Indigo2Ip22 | MachineProfile::Indigo2Ip28)
}

/// Whether `profile` has the Indigo2's motherboard EEPROM (`nveeprom`). The
/// Indy keeps its PROM environment in the NVRAM, the O2 in its flash.
pub fn has_nveeprom(profile: MachineProfile) -> bool {
    matches!(profile, MachineProfile::Indigo2Ip22 | MachineProfile::Indigo2Ip28)
}

/// Whether `cfg.prom` runs the PROM built into iris: `(embedded)`, an empty
/// value the GUI writes as `(embedded)` before a start, or the default name
/// with neither it nor the machine's release file name in the machine
/// folder (`prom_image::resolve`).
pub fn builtin_prom(cfg: &MachineConfig) -> bool {
    let prom = cfg.prom.as_str();
    let exists = |p: &str| std::path::Path::new(p).exists();
    prom.is_empty()
        || iris::dev::prom_image::is_embedded_keyword(prom)
        || (prom == DEFAULT_PROM
            && !exists(prom)
            && !cfg.machine.profile.release_prom_file().is_some_and(exists))
}

/// The slot key of the disk at (`controller`, `id`), if one is there.
pub fn scsi_key(cfg: &MachineConfig, controller: u8, id: u8) -> Option<u8> {
    let key = iris::config::scsi_slot(controller, id);
    cfg.disk(key).map(|_| key)
}

/// The machine's SCSI targets by slot key, in (controller, id) order.
pub fn scsi_disks(cfg: &MachineConfig) -> Vec<(u8, &ScsiDeviceConfig)> {
    cfg.disks().into_iter().filter_map(|(key, i)| Some((key, cfg.devices[i].device.scsi()?))).collect()
}

/// Take every SCSI target off the machine (not the disks on a Fibre Channel
/// loop, which are not SCSI targets of the machine's own buses).
pub fn clear_disks(cfg: &mut MachineConfig) {
    let mut at: Vec<usize> = cfg.disks().into_iter().map(|(_, i)| i).collect();
    at.sort_unstable();
    for i in at.into_iter().rev() {
        cfg.devices.remove(i);
    }
}

/// The ids a target can take on controller `controller` of this machine.
pub fn scsi_ids(profile: MachineProfile, controller: u8) -> Vec<u8> {
    iris::topology::scsi_buses(&profile.slots()).into_iter().find(|b| b.0 == controller).map(|b| b.2).unwrap_or_default()
}

/// (controller, id) of slot key `key`.
pub fn placement(key: u8) -> (u8, u8) {
    (key / 16, key % 16)
}

/// How the target at slot `key` (`scsi_slot(controller, id)`) is named to
/// the user: its id on controller 0, `controller:id` on another (so key 19
/// reads "1:3", not "19").
pub fn scsi_name(key: u8) -> String {
    if key < 16 { key.to_string() } else { format!("{}:{}", key / 16, key % 16) }
}

/// Choices reflect the boards actually fitted by machine constructors. The O2
/// has a fixed GBE display; its shared board field is only a placeholder.
pub fn board_supported(board: GraphicsBoard, profile: MachineProfile) -> bool {
    board.supports(profile)
        && (profile != MachineProfile::O2Ip32 || board == GraphicsBoard::Newport)
        && (profile != MachineProfile::Onyx2Ip27 || board != GraphicsBoard::Ir3)
}

/// Refit hardware when the user explicitly changes profiles from `from`. Do
/// not run this each frame: it would erase intentional edits and imported
/// configurations.
pub fn switch_profile(cfg: &mut MachineConfig, from: MachineProfile) {
    let p = cfg.machine.profile;
    cfg.machine.cpu = None;
    cfg.machine.cpus = None;
    cfg.clock.fixed_mhz = None;
    cfg.banks = distribute_ram(crate::model::ram::ram_presets(p)[0], p);
    cfg.banks_set = true;
    cfg.graphics.board = Some(if p.ip28() {
        GraphicsBoard::SolidImpact
    } else {
        GraphicsBoard::default_for(p)
    });
    // `compile = "sync"` needs one CPU; the new machine may start with more.
    if cfg.machine.cpus() > 1 {
        cfg.jitv2.compile = JitCompile::Async;
    }
    cfg.graphics.resolution = Default::default();
    cfg.octane.graphics = None;
    // SN0 rejects another product's private table, even when it is inactive.
    cfg.origin200 = Default::default();
    cfg.origin2000 = Default::default();
    cfg.onyx2 = Default::default();
    rehome_disks(cfg, from);
    // Cards need a GIO slot; a second head also needs a Newport board.
    if !mc_machine(p) || cfg.graphics_board() != GraphicsBoard::Newport {
        cfg.set_device("newport", false);
    }
    // What `MachineConfig::validate` refuses on the new machine, and the
    // Advanced tab no longer shows for it.
    if !matches!(p, MachineProfile::FuelIp34 | MachineProfile::TezroIp53) {
        cfg.machine.emu_id = 0;
    }
    if mc_machine(p) && cfg.network.mode == NetMode::Off {
        cfg.network.mode = NetMode::Nat;
    }
    // The console's own channel is logged by `serial_log`.
    if matches!(p, MachineProfile::OctaneIp30 | MachineProfile::O2Ip32) {
        cfg.serial_log_a = None;
    }
    if mc_machine(p) {
        cfg.serial_log_b = None;
    }
}

/// Whether a Fibre Channel adapter goes in one of the machine's PCI slots: not
/// on the O2, and not on the I-brick, whose adapter is built in.
pub fn takes_fibre_card(p: MachineProfile) -> bool {
    iris::topology::slots_take(&p.slots(), DeviceFamily::Pci) && p != MachineProfile::O2Ip32 && !p.is_ip35_bricks()
}

/// Keep each SCSI device through a profile switch at the same controller and
/// target, dropping it where the new machine has no such slot. Other devices
/// keep their slot where they can, else take any, else go.
fn rehome_disks(cfg: &mut MachineConfig, from: MachineProfile) {
    let p = cfg.machine.profile;
    let (old, new) = (from.slots(), p.slots());
    let brick = new.iter().find(|s| s.kind == MachineSlotKind::Fixed("qla2200")).map(|s| s.path.clone());
    let mut old_cfg = cfg.clone();
    old_cfg.machine.profile = from;
    let was = DevicePlacement::lenient(&old_cfg, &cfg.devices);
    let places: Vec<_> = (0..cfg.devices.len()).map(|i| was.at(i).cloned()).collect();
    let devices = std::mem::take(&mut cfg.devices);
    let mut others = Vec::new();
    for (mut d, at) in devices.into_iter().zip(places) {
        if matches!(d.device, DeviceModel::Qla2200) && !takes_fibre_card(p) {
            continue;
        }
        if matches!(d.device, DeviceModel::Nvme(_)) && p.refuse_nvme().is_err() {
            continue;
        }
        if d.device.scsi().is_none() {
            others.push(d);
            continue;
        }
        if let Some(id) = at.as_ref().and_then(|a| a.loop_number()) {
            if let Some(b) = &brick {
                d.at = Some(b.loop_id(id).to_string());
                cfg.devices.push(d);
            }
            continue;
        }
        let target = at.and_then(|at| scsi_target(&old, &at)).and_then(|(c, id)| scsi_target_slot(&new, c, id));
        let Some(target) = target.filter(|_| p.takes_daynaport() || !d.device.scsi().is_some_and(|s| s.is_daynaport())) else { continue };
        // An alias that names the same slot on the new machine is kept as written.
        if DevicePlacement::lenient(cfg, std::slice::from_ref(&d)).at(0) != Some(&target) {
            d.at = Some(target.to_string());
        }
        cfg.devices.push(d);
    }
    for d in others {
        cfg.devices.push(d);
        if DevicePlacement::new(cfg).is_err() {
            cfg.devices.last_mut().expect("just pushed").at = None;
            if DevicePlacement::new(cfg).is_err() {
                cfg.devices.pop();
            }
        }
    }
}

/// A path for one more NVMe card's image that no card has yet.
pub fn new_nvme_path(cfg: &MachineConfig) -> String {
    let taken: Vec<&str> = cfg.nvmes().map(|n| n.path.as_str()).collect();
    ["disks/nvme.raw".to_string()].into_iter().chain((2..).map(|n| format!("disks/nvme{n}.raw"))).find(|p| !taken.contains(&p.as_str())).expect("unbounded")
}

/// The name a new machine of `profile` gets until the user types one:
/// the profile's name up to its board number (`o2`, `octane`, `indigo2`).
pub fn default_name(profile: MachineProfile) -> &'static str {
    let name = profile.name();
    name.split_once('_').map_or(name, |(model, _)| model)
}

/// Fill the profile's banks largest first, without gaps.
pub fn distribute_ram(total: u32, profile: MachineProfile) -> [u32; iris::config::MAX_BANKS] {
    let mut remaining = total;
    let mut banks = [0u32; iris::config::MAX_BANKS];
    for slot in banks.iter_mut().take(profile.memory_banks()) {
        // Pick the largest size in VALID_BANK_SIZES that is <= remaining.
        let pick = VALID_BANK_SIZES.iter()
            .filter(|&&s| s > 0 && s <= remaining && profile.bank_sizes().contains(&s))
            .max().copied().unwrap_or(0);
        *slot = pick;
        remaining -= pick;
        if remaining == 0 { break; }
    }
    banks
}

/// Parse and validate the text of a complete machine config.
pub fn parse_complete(text: &str) -> Result<MachineConfig, String> {
    let cfg = crate::model::machines::parse(text)?;
    cfg.validate()?;
    Ok(cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use iris::dev::xio_pci::Kind;
    use iris::devices::ConfiguredDevice;
    use iris::topology::scsi_buses;

    fn switch(cfg: &mut MachineConfig, to: MachineProfile) {
        let from = cfg.machine.profile;
        cfg.machine.profile = to;
        switch_profile(cfg, from);
    }

    #[test]
    fn every_profile_can_be_created_saved_and_switched_to() {
        assert_eq!(MachineProfile::ALL, MachineProfile::EVERY);
        for p in MachineProfile::ALL {
            let mut cfg = MachineConfig::default();
            switch(&mut cfg, p);
            cfg.validate().unwrap_or_else(|e| panic!("{p:?}: {e}"));
            let loaded = parse_complete(&toml::to_string(&cfg).unwrap()).unwrap();
            assert_eq!(loaded.machine.profile, p);
            assert_eq!(loaded.banks, cfg.banks);
            // Changing away from an SN0 table must not leave a foreign table.
            cfg.origin200.flash = Some("other-flash.bin".into());
            switch(&mut cfg, p);
            cfg.validate()
                .unwrap_or_else(|e| panic!("switch to {p:?}: {e}"));
        }
    }

    #[test]
    fn each_added_nvme_card_gets_its_own_image() {
        let mut cfg = MachineConfig::default();
        for want in ["disks/nvme.raw", "disks/nvme2.raw", "disks/nvme3.raw"] {
            let path = new_nvme_path(&cfg);
            assert_eq!(path, want);
            cfg.devices.push(ConfiguredDevice { at: None, ..ConfiguredDevice::new("", DeviceModel::Nvme(NvmeConfig { path, ..Default::default() })) });
        }
    }

    /// Switching away from a machine with expansion, extra disks and an
    /// identity leaves nothing the new machine refuses.
    #[test]
    fn switching_from_a_populated_machine_leaves_a_valid_config() {
        let nvme = || ConfiguredDevice { at: None, ..ConfiguredDevice::new("", DeviceModel::Nvme(NvmeConfig { path: "nvme.raw".into(), ..Default::default() })) };
        for from in [MachineProfile::OctaneIp30, MachineProfile::TezroIp53, MachineProfile::O2Ip32] {
            for to in MachineProfile::ALL {
                let mut cfg = MachineConfig::default();
                switch(&mut cfg, from);
                match from {
                    MachineProfile::OctaneIp30 => {
                        cfg.devices.push(ConfiguredDevice { at: None, ..ConfiguredDevice::new("", DeviceModel::XioBoard(Kind::Shoehorn)) });
                        cfg.devices.push(ConfiguredDevice { at: None, ..ConfiguredDevice::new("", DeviceModel::XioBoard(Kind::Cage)) });
                        cfg.devices.push(nvme());
                        cfg.devices.push(ConfiguredDevice { name: Some("fibre_channel".into()), at: None, ..ConfiguredDevice::new("", DeviceModel::Qla2200) });
                        cfg.serial_log_b = Some("b.log".into());
                        cfg.set_disk(iris::config::scsi_slot(1, 12), ScsiDeviceConfig { path: "w.raw".into(), ..Default::default() });
                    }
                    MachineProfile::TezroIp53 => {
                        cfg.machine.emu_id = 7;
                        cfg.machine.cpus = Some(1);
                        cfg.jitv2.compile = JitCompile::Sync;
                        cfg.network.mode = NetMode::Off;
                    }
                    _ => {
                        cfg.devices.push(nvme());
                        cfg.serial_log_b = Some("b.log".into());
                    }
                }
                cfg.validate().unwrap_or_else(|e| panic!("populated {from:?}: {e}"));
                switch(&mut cfg, to);
                cfg.validate().unwrap_or_else(|e| panic!("{from:?} -> {to:?}: {e}"));
            }
        }
    }

    /// Every disk keeps its controller and target through a switch, or is
    /// dropped where the new machine has no such slot.
    #[test]
    fn disks_follow_their_controller_and_target_through_a_switch() {
        let disk = |at: &str| ConfiguredDevice::new(at, DeviceModel::Scsi(ScsiDeviceConfig { path: format!("{at}.raw"), ..Default::default() }));
        for from in MachineProfile::ALL {
            for to in MachineProfile::ALL {
                let mut cfg = MachineConfig::default();
                switch(&mut cfg, from);
                cfg.devices = scsi_buses(&from.slots()).iter().flat_map(|(_, at, ids)| [1, *ids.last().unwrap()].map(|id| disk(&format!("{at}/t{id}")))).collect();
                let before: Vec<u8> = cfg.disks().iter().map(|d| d.0).collect();
                switch(&mut cfg, to);
                cfg.validate().unwrap_or_else(|e| panic!("{from:?} -> {to:?}: {e}"));
                let after: Vec<u8> = cfg.disks().iter().map(|d| d.0).collect();
                let new = to.slots();
                let fits: Vec<u8> = before.iter().copied().filter(|&s| scsi_target_slot(&new, s / 16, s % 16).is_some()).collect();
                assert_eq!(after, fits, "{from:?} -> {to:?}");
                assert_eq!(cfg.devices.iter().filter(|d| d.device.scsi().is_some()).count(), after.len(), "{from:?} -> {to:?}: nothing without a slot is kept");
            }
        }
        // An alias stays as written; a full slot is rewritten for the new machine.
        let mut cfg = MachineConfig::default();
        cfg.devices = vec![disk("scsi0/t2"), disk("hpc3/c0/t3")];
        switch(&mut cfg, MachineProfile::OctaneIp30);
        let at: Vec<_> = cfg.devices.iter().filter(|d| d.device.scsi().is_some()).map(|d| d.at.clone().unwrap()).collect();
        assert_eq!(at, ["scsi0/t2", "xio15/pci0/0/t3"]);
    }

    /// A loop disk moves to the I-brick's loop, and nowhere else keeps it or
    /// the card.
    #[test]
    fn a_loop_disk_moves_to_the_ibricks_loop() {
        let mut cfg = MachineConfig::default();
        switch(&mut cfg, MachineProfile::OctaneIp30);
        cfg.devices.push(ConfiguredDevice { name: Some("cage".into()), ..ConfiguredDevice::new("shoebox", DeviceModel::XioBoard(Kind::Cage)) });
        cfg.devices.push(ConfiguredDevice { name: Some("fibre_channel".into()), ..ConfiguredDevice::new("cage/pci0", DeviceModel::Qla2200) });
        cfg.devices.push(ConfiguredDevice::new("fibre_channel/loop7", DeviceModel::Scsi(ScsiDeviceConfig { path: "fc.raw".into(), ..Default::default() })));
        assert_eq!(cfg.loop_disks().iter().map(|d| d.0).collect::<Vec<_>>(), [7]);
        for to in [MachineProfile::Origin3200Ip35, MachineProfile::Onyx3000Ip35] {
            let mut moved = cfg.clone();
            switch(&mut moved, to);
            moved.validate().unwrap();
            let kept = moved.devices.iter().filter(|d| d.device.scsi().is_some());
            assert_eq!(kept.map(|d| d.at.clone()).collect::<Vec<_>>(), [Some("xio15/pci0/1/loop7".to_string())]);
            assert!(!moved.has_device("qla2200"));
        }
        switch(&mut cfg, MachineProfile::IndyIp24);
        assert!(cfg.loop_disks().is_empty() && !cfg.has_device("qla2200"));
    }

    /// Slot keys name (controller, id): key 19 is 1:3.
    #[test]
    fn scsi_targets_are_found_by_slot() {
        let mut cfg = MachineConfig::default();
        cfg.machine.profile = MachineProfile::Indigo2Ip22;
        cfg.devices.clear();
        cfg.set_disk(iris::config::scsi_slot(1, 3), ScsiDeviceConfig::default());
        assert_eq!(scsi_key(&cfg, 0, 3), None);
        assert_eq!(scsi_key(&cfg, 1, 3), Some(19));
        assert_eq!((scsi_name(3), scsi_name(19)), ("3".to_string(), "1:3".to_string()));
        assert_eq!(placement(19), (1, 3));
    }

    #[test]
    fn all_offered_ram_presets_validate() {
        for p in MachineProfile::ALL {
            for &mb in crate::model::ram::ram_presets(p) {
                let mut cfg = MachineConfig::default();
                switch(&mut cfg, p);
                cfg.banks = distribute_ram(mb, p);
                assert_eq!(cfg.banks.iter().sum::<u32>(), mb, "{p:?}: {mb}");
                cfg.validate()
                    .unwrap_or_else(|e| panic!("{p:?} {mb} MB: {e}"));
            }
        }
    }

    #[test]
    fn complete_editor_rejects_invalid_and_unknown_settings() {
        assert!(parse_complete("scale = 0").is_err());
        assert!(parse_complete("made_up = true").is_err());
        assert!(parse_complete("[graphics]\ntile = 19").is_err());
    }
}
