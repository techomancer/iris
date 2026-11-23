use crate::handle::Status;
use iris::config::MachineConfig;

/// Reasons it is risky to force-stop right now.
#[derive(Debug, Clone, Default)]
pub struct UnsafeReasons {
    /// Attached disks whose guest writes land directly in their base image
    /// (a plain read-write HDD), as named to the user: `SCSI 1` (`SCSI 1:3`
    /// on controller 1), `Fibre Channel 1` or `NVMe`. Force-stopping
    /// mid-write can leave the filesystem on these inconsistent.
    pub writable_disks: Vec<String>,
}

impl UnsafeReasons {
    pub fn is_empty(&self) -> bool {
        self.writable_disks.is_empty()
    }
}

/// Whether a disk's guest writes go straight into `path`.
fn persists_to_base(path: &str, overlay: bool) -> bool {
    !path.trim().is_empty() && !overlay && !path.ends_with(".chd")
}

/// Classify whether the GUI should ask before a force-stop.
///
/// This is a config/status heuristic, not a guest shutdown acknowledgement:
/// `cpu_halted` includes zero measured throughput. The machine's own disks
/// are inspected: `[scsi.N]` (not on the IP35 brick machines, which have
/// none), `[fibre_channel.disk.N]` and `[nvme]`. CD-ROMs, scratch, overlays,
/// DaynaPort and `.chd` names are excluded from the warning. The filename
/// check does not distinguish an uncompressed CHD that writes in place from a
/// compressed parent with a diff. See STORAGE.md for the actual persistence
/// guarantees of each backend.
pub fn evaluate(status: &Status, cfg: &MachineConfig) -> UnsafeReasons {
    // Preserve the UI's halted/idle short-circuit before classifying disks.
    if status.cpu_halted {
        return UnsafeReasons::default();
    }

    let mut r = UnsafeReasons::default();
    for (k, i) in cfg.disks() {
        let dev = cfg.devices[i].device.scsi().expect("listed by disks");
        // A DaynaPort has no image behind it.
        if !dev.is_cdrom() && !dev.scratch && !dev.is_daynaport() && persists_to_base(&dev.path, dev.overlay) {
            r.writable_disks.push(t!("safestop-scsi-disk", id = crate::extra_ui::scsi_name(k)));
        }
    }
    if let Some(fc) = &cfg.fibre_channel {
        for (id, disk) in &fc.disks {
            if persists_to_base(&disk.path, disk.overlay) {
                r.writable_disks.push(t!("safestop-fc-disk", id = *id));
            }
        }
    }
    if persists_to_base(&cfg.nvme.path, cfg.nvme.overlay) {
        r.writable_disks.push(t!("safestop-nvme-disk"));
    }
    r
}

/// Human-readable lines for the confirmation dialog.
pub fn reason_lines(r: &UnsafeReasons) -> Vec<String> {
    r.writable_disks
        .iter()
        .map(|disk| t!("safestop-writable-disk", disk = disk))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // MachineConfig::default() attaches scsi1 as a plain read-write disk image,
    // so it is the "unsafe while running" case.
    fn running(halted: bool) -> Status {
        Status { running: true, cpu_halted: halted, ..Status::default() }
    }

    #[test]
    fn writable_disk_is_unsafe_while_cpu_runs() {
        let r = evaluate(&running(false), &MachineConfig::default());
        assert!(!r.is_empty(), "a live rw disk should warn before force-stop");
        assert_eq!(r.writable_disks.len(), 1);
        assert!(r.writable_disks[0].contains('1'));
    }

    /// The Origin 3200's default `[scsi]` map is not attached; its Fibre
    /// Channel disks and an NVMe image are what can be written.
    #[test]
    fn ip35_warns_for_its_fibre_channel_disks_not_the_scsi_map() {
        let mut cfg = MachineConfig::default();
        cfg.machine.profile = iris::config::MachineProfile::Origin3200Ip35;
        let mut fc = iris::config::FibreChannelConfig::default();
        fc.disks.insert(1, iris::config::ScsiDeviceConfig { path: "root.raw".into(), ..Default::default() });
        cfg.fibre_channel = Some(fc);
        let r = evaluate(&running(false), &cfg);
        assert_eq!(r.writable_disks.len(), 1);
        assert!(r.writable_disks[0].contains('1'));
    }

    /// A target on the second controller is named controller:id.
    #[test]
    fn second_controller_disk_is_named_controller_and_id() {
        let mut cfg = MachineConfig::default();
        cfg.machine.profile = iris::config::MachineProfile::Indigo2Ip22;
        cfg.devices.clear();
        cfg.set_disk(19, iris::config::ScsiDeviceConfig { path: "d.raw".into(), ..Default::default() });
        let r = evaluate(&running(false), &cfg);
        assert_eq!(r.writable_disks.len(), 1);
        assert!(r.writable_disks[0].contains("1:3"), "{:?}", r.writable_disks);
    }

    #[test]
    fn halted_cpu_is_always_safe() {
        // A halted status bypasses the disk warning under this UI policy.
        let r = evaluate(&running(true), &MachineConfig::default());
        assert!(r.is_empty(), "a halted CPU must be safe to stop");
    }
}
