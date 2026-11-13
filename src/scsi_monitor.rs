//! The monitor's `scsi` and `cow` commands, and the run-time disc changes
//! behind them, over any SCSI controller's target table.
//!
//! A controller implements [`ScsiTargets`] by handing over its [`ScsiBus`];
//! the WD33C93A (Indy, Indigo2), the Adaptec 7880 (O2) and the ISP1040
//! (Octane) all do.
//! Everything that only needs the targets lives here once: `scsi status`,
//! `scsi dayna`, the changer subcommands (`eject`, `add`, `list`, `del`,
//! `next`), the whole of `cow`, and the [`crate::workstation::DiscChanger`]
//! operations the window, the CI socket and iris-gui use. What is one chip's
//! alone (`scsi regs`, the WD's `wdt`, `defer` and `debug`) stays with the
//! chip, which tries [`scsi_subcommand`] first and handles the rest itself.

use std::io::Write;

use crate::block_node::{ChangerNode, Removable};
use crate::scsi::{ScsiBus, ScsiTarget, CDROM_SECTOR};

/// A SCSI controller, for the targets on its bus.
pub trait ScsiTargets {
    /// The bus whose targets the commands act on.
    fn scsi_bus(&self) -> &ScsiBus;

    /// Run `f` over the target table, `None` where nothing is attached.
    fn with_targets<R>(&self, f: impl FnOnce(&mut [Option<Box<dyn ScsiTarget>>]) -> R) -> R {
        self.scsi_bus().with_targets(f)
    }
}

/// The shared subcommands, for a chip's `help` text.
pub const SCSI_USAGE: &str = "scsi status | scsi dayna | scsi eject <id> | scsi add <id> <path> | scsi list <id> | scsi del <id> <ord> | scsi next <id> <ord>";

/// Run `f` on the changer of the CD-ROM at `id`, or say why there is none.
fn cdrom<R>(ctl: &impl ScsiTargets, id: usize, f: impl FnOnce(&mut ChangerNode) -> Result<R, String>) -> Result<R, String> {
    ctl.with_targets(|targets| match targets.get_mut(id).and_then(|d| d.as_mut()) {
        None => Err(format!("No device at SCSI ID {}", id)),
        Some(dev) => match dev.changer() {
            None => Err(format!("SCSI ID {} is not a CD-ROM", id)),
            Some(changer) => f(changer),
        },
    })
}

/// Every CD-ROM: (id, current disc, the changer list, physical and logical
/// block size).
pub fn disc_status(ctl: &impl ScsiTargets) -> Vec<(usize, String, Vec<String>, u64, u64)> {
    ctl.with_targets(|targets| {
        targets.iter_mut().enumerate()
            .filter_map(|(id, d)| {
                let dev = d.as_mut()?;
                let logical = dev.block_size()?;
                let changer = dev.changer()?;
                Some((id, changer.current().unwrap_or("").to_string(), changer.discs().to_vec(), CDROM_SECTOR, logical))
            })
            .collect()
    })
}

/// Put the image at `path` in CD-ROM `id` and make it the active disc, as a
/// hand swapping the disc would. Returns the path loaded.
pub fn load_disc(ctl: &impl ScsiTargets, id: usize, path: String) -> Result<String, String> {
    cdrom(ctl, id, |changer| {
        changer.load(&path).map_err(|e| format!("could not open {}: {}", path, e))?;
        Ok(path)
    })
}

/// Eject CD-ROM `id`'s disc and load the next in its changer list; returns
/// the new active path.
pub fn eject_disc(ctl: &impl ScsiTargets, id: usize) -> Result<String, String> {
    cdrom(ctl, id, |changer| match changer.eject() {
        Ok(Some(path)) => Ok(path),
        Ok(None) => Err(format!("SCSI ID {} has only one disc (nothing to eject to)", id)),
        Err(e) => Err(format!("SCSI ID {}: could not load the next disc: {}", id, e)),
    })
}

/// Eject CD-ROM `id`'s disc and leave the drive empty.
pub fn eject_to_empty(ctl: &impl ScsiTargets, id: usize) -> Result<(), String> {
    cdrom(ctl, id, |changer| { changer.unload(); Ok(()) })
}

/// Queue `path` as the next disc after an eject on CD-ROM `id`.
pub fn add_disc(ctl: &impl ScsiTargets, id: usize, path: String) -> Result<(), String> {
    cdrom(ctl, id, |changer| changer.add(path))
}

/// Remove the disc at `ordinal` from CD-ROM `id`'s queue (not the active one).
pub fn remove_disc(ctl: &impl ScsiTargets, id: usize, ordinal: usize) -> Result<String, String> {
    cdrom(ctl, id, |changer| changer.remove(ordinal))
}

/// Move the disc at `ordinal` to be the next after an eject on CD-ROM `id`.
pub fn move_disc_next(ctl: &impl ScsiTargets, id: usize, ordinal: usize) -> Result<(), String> {
    cdrom(ctl, id, |changer| changer.move_next(ordinal))
}

fn write_queue(w: &mut dyn Write, list: &[String]) {
    for (i, d) in list.iter().enumerate() {
        let tag = if i == 0 { " <active>" } else if i == 1 { " <next>" } else { "" };
        let _ = writeln!(w, "  [{}] {}{}", i, d, tag);
    }
}

/// The `scsi` subcommands every controller has. `None` when `args` names
/// none of them, so the chip can try its own.
pub fn scsi_subcommand(ctl: &impl ScsiTargets, args: &[&str], w: &mut dyn Write) -> Option<Result<(), String>> {
    let id_arg = |n: usize, usage: &str| -> Result<usize, String> {
        args.get(n).and_then(|s| s.parse().ok()).ok_or_else(|| usage.to_string())
    };
    let r = match args.first().copied()? {
        "status" => {
            let lines = ctl.with_targets(|targets| {
                let mut lines = Vec::new();
                for (id, dev) in targets.iter().enumerate() {
                    if let Some(dev) = dev {
                        lines.extend(dev.status(id));
                    }
                }
                lines
            });
            if lines.is_empty() {
                let _ = writeln!(w, "No devices attached");
            }
            for l in lines {
                let _ = writeln!(w, "{}", l);
            }
            Ok(())
        }
        "dayna" => {
            let lines = ctl.with_targets(|targets| {
                targets.iter().flatten()
                    .flat_map(|dev| dev.adapter_status())
                    .collect::<Vec<_>>()
            });
            if lines.is_empty() {
                let _ = writeln!(w, "No DaynaPort targets attached");
            }
            for l in lines {
                let _ = writeln!(w, "{}", l);
            }
            Ok(())
        }
        "eject" => id_arg(1, "Usage: scsi eject <id>").map(|id| {
            match eject_disc(ctl, id) {
                Ok(path) => { let _ = writeln!(w, "SCSI ID {}: switched to {}", id, path); }
                Err(e) => { let _ = writeln!(w, "Error: {}", e); }
            }
        }),
        "add" => id_arg(1, "Usage: scsi add <id> <path>").and_then(|id| {
            let path = args.get(2).ok_or_else(|| "Usage: scsi add <id> <path>".to_string())?.to_string();
            match add_disc(ctl, id, path.clone()) {
                Ok(()) => { let _ = writeln!(w, "SCSI ID {}: added {} as next disc", id, path); }
                Err(e) => { let _ = writeln!(w, "Error: {}", e); }
            }
            Ok(())
        }),
        "list" => id_arg(1, "Usage: scsi list <id>").map(|id| {
            match cdrom(ctl, id, |changer| Ok(changer.discs().to_vec())) {
                Ok(list) => {
                    let _ = writeln!(w, "SCSI ID {} queue ({} disc(s)):", id, list.len());
                    write_queue(w, &list);
                }
                Err(e) => { let _ = writeln!(w, "{}", e); }
            }
        }),
        "del" => id_arg(1, "Usage: scsi del <id> <ordinal>").and_then(|id| {
            let ordinal = id_arg(2, "Usage: scsi del <id> <ordinal>")?;
            match remove_disc(ctl, id, ordinal) {
                Ok(path) => { let _ = writeln!(w, "SCSI ID {}: removed [{}] {}", id, ordinal, path); }
                Err(e) => { let _ = writeln!(w, "Error: {}", e); }
            }
            Ok(())
        }),
        "next" => id_arg(1, "Usage: scsi next <id> <ordinal>").and_then(|id| {
            let ordinal = id_arg(2, "Usage: scsi next <id> <ordinal>")?;
            match move_disc_next(ctl, id, ordinal) {
                Ok(()) => { let _ = writeln!(w, "SCSI ID {}: [{}] moved to next position", id, ordinal); }
                Err(e) => { let _ = writeln!(w, "Error: {}", e); }
            }
            Ok(())
        }),
        _ => return None,
    };
    Some(r)
}

/// `cow status | cow commit [id] | cow reset [id]` over the controller's
/// copy-on-write targets.
pub fn cow_command(ctl: &impl ScsiTargets, args: &[&str], w: &mut dyn Write) -> Result<(), String> {
    let explicit: Option<usize> = match args.get(1) {
        Some(s) => Some(s.parse().map_err(|_| "invalid SCSI ID".to_string())?),
        None => None,
    };
    let lines: Vec<String> = match args.first().copied() {
        Some("status") => ctl.with_targets(|targets| {
            targets.iter().enumerate()
                .filter_map(|(id, d)| d.as_ref().map(|d| (id, d)))
                .map(|(id, d)| if let Some(summary) = d.cow_summary() {
                    format!("SCSI {}: COW overlay, {}", id, summary)
                } else {
                    format!("SCSI {}: direct (no overlay)", id)
                })
                .collect()
        }),
        Some(op @ ("commit" | "reset")) => ctl.with_targets(|targets| {
            let ids: Vec<usize> = match explicit {
                Some(id) => vec![id],
                None => (0..targets.len()).filter(|&i| targets[i].as_ref().is_some_and(|d| d.cow_dirty().is_some())).collect(),
            };
            let mut lines = Vec::new();
            for id in ids {
                let Some(dev) = targets.get_mut(id).and_then(|d| d.as_mut()) else { continue };
                lines.push(if op == "commit" {
                    match dev.cow_commit() {
                        Ok(n) if n > 0 => format!("SCSI {}: committed {} sectors to base image", id, n),
                        Ok(_) => format!("SCSI {}: nothing to commit", id),
                        Err(e) => format!("SCSI {}: commit failed: {}", id, e),
                    }
                } else {
                    match dev.cow_reset() {
                        Ok(()) => format!("SCSI {}: overlay reset (all writes discarded)", id),
                        Err(e) => format!("SCSI {}: reset failed: {}", id, e),
                    }
                });
            }
            lines
        }),
        _ => return Err("Usage: cow status | cow commit [id] | cow reset [id]".to_string()),
    };
    for l in lines {
        let _ = writeln!(w, "{}", l);
    }
    Ok(())
}
