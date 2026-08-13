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

/// Evaluate whether stopping the emulator right now is safe.
///
/// If the CPU has halted (clean shutdown / soft power-off, or idle at the PROM),
/// nothing is writing and stopping is always safe — see the `cpu_halted`
/// short-circuit below.
///
/// Otherwise the core does not expose live dirty-sector state, so we decide
/// purely from config: an abrupt power-off only risks the on-disk image when some attached
/// device persists guest writes straight into its **base image** — i.e. a
/// plain read-write hard disk. Everything else leaves the base image untouched
/// and is safe to power off without warning:
///
/// - **CD-ROM** — read-only.
/// - **COW overlay** (`overlay = true`) — writes go to a `{path}.overlay`
///   sidecar; the base image is never modified (delete the overlay to reset).
/// - **Scratch volume** (`scratch = true`) — a transient host-side file, not a
///   guest filesystem we need to protect.
/// - **CHD** (`*.chd`) — writes go to a `.diff.chd` sidecar; the base CHD is
///   never modified.
///
/// So when no attached device writes through to its base image, powering off
/// will NOT damage the hard disk and we skip the confirmation dialog entirely.
pub fn evaluate(status: &Status, cfg: &MachineConfig) -> UnsafeReasons {
    // If the CPU has halted — a clean IRIX shutdown / soft power-off, or sitting
    // idle at the PROM (0 MIPS) — nothing is writing to any disk, so stopping
    // now cannot corrupt a filesystem. Skip the warning regardless of config.
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
        .map(|id| {
            format!(
                "scsi{id} is a read-write disk image — force-stopping while IRIX \
                 is running can corrupt its filesystem."
            )
        })
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
        // After IRIX shuts down (0 MIPS / power-off) stopping can't corrupt a
        // disk, so the same config must now evaluate as safe.
        let r = evaluate(&running(true), &MachineConfig::default());
        assert!(r.is_empty(), "a halted CPU must be safe to stop");
    }
}
