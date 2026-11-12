use crate::handle::Status;
use iris::config::MachineConfig;

/// Reasons it is risky to force-stop right now.
#[derive(Debug, Clone, Default)]
pub struct UnsafeReasons {
    /// SCSI IDs of attached disks whose guest writes land directly in their
    /// base image (a plain read-write HDD). Force-stopping mid-write can leave
    /// the filesystem on these inconsistent.
    pub writable_disks: Vec<u8>,
}

impl UnsafeReasons {
    pub fn is_empty(&self) -> bool {
        self.writable_disks.is_empty()
    }
}

/// Classify whether the GUI should ask before a force-stop.
///
/// This is a config/status heuristic, not a guest shutdown acknowledgement:
/// `cpu_halted` includes zero measured throughput, and only `[scsi]` devices
/// are inspected. CD-ROMs, scratch, overlays, DaynaPort and `.chd` names are
/// excluded from the warning. The filename check does not distinguish an
/// uncompressed CHD that writes in place from a compressed parent with a diff.
/// See STORAGE.md for the actual persistence guarantees of each backend.
pub fn evaluate(status: &Status, cfg: &MachineConfig) -> UnsafeReasons {
    // Preserve the UI's halted/idle short-circuit before classifying disks.
    if status.cpu_halted {
        return UnsafeReasons::default();
    }

    let mut r = UnsafeReasons::default();
    for (id, dev) in &cfg.scsi {
        let persists_to_base = !dev.cdrom
            && !dev.overlay
            && !dev.scratch
            && !dev.is_daynaport() // no image behind a network adapter
            && !dev.path.ends_with(".chd");
        if persists_to_base {
            r.writable_disks.push(*id);
        }
    }
    r.writable_disks.sort();
    r
}

/// Human-readable lines for the confirmation dialog.
pub fn reason_lines(r: &UnsafeReasons) -> Vec<String> {
    r.writable_disks
        .iter()
        .map(|id| t!("safestop-writable-disk", id = *id))
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
        assert!(r.writable_disks.contains(&1));
    }

    #[test]
    fn halted_cpu_is_always_safe() {
        // A halted status bypasses the disk warning under this UI policy.
        let r = evaluate(&running(true), &MachineConfig::default());
        assert!(r.is_empty(), "a halted CPU must be safe to stop");
    }
}
