//! The adapter with the disks placed on its loop: opened with them, then
//! attached to a PCI slot.
//!
//! The same shape as the NVMe card's ([`crate::dev::nvme::Card`]). A disk is
//! attached through [`crate::scsi::attach_table`], the call every `[scsi.N]`
//! disk goes through, so the loop disks take whatever a disk table does
//! (overlay, write cache). The disks report the machine-wide disk identity
//! of [`Isp2200::new_bus`].

use std::sync::Arc;

use std::collections::BTreeMap;

use super::Isp2200;
use crate::config::ScsiDeviceConfig;
use crate::hptimer::TimerManager;
use crate::pci::{DevFn, PciBus};
use crate::snapshot::DiskRef;

/// The adapter's number among the machine's SCSI adapters: the Octane's two
/// ISP1040s are 0 and 1, and the monitor's `scsi2` and `cow2` are this one.
pub const ADAPTER_INDEX: u8 = 2;

/// The key a `--ci` run's overlay for the disk at `loop_id` is named by:
/// above every SCSI target's.
const CI_KEY_BASE: u8 = 0x80;

/// The disks on a loop, by loop ID.
pub type Loop<'d> = BTreeMap<u8, &'d ScsiDeviceConfig>;

pub struct Card {
    pub fc: Arc<Isp2200>,
}

impl Card {
    /// The adapter with its loop's disks attached. `ci` is a `--ci` run,
    /// which keeps each overlay in the temporary directory.
    pub fn open(disks: &Loop, ci: bool, tm: &Arc<TimerManager>) -> Result<Card, String> {
        let fc = Arc::new(Isp2200::new_at(ADAPTER_INDEX, Isp2200::new_bus(), tm));
        for (&id, &disk) in disks {
            let at = format!("Fibre Channel loop ID {id}");
            if disk.scratch {
                return Err(format!("{at}: only a plain disk goes on the loop"));
            }
            let ci_overlay = (ci && disk.overlay).then(|| ScsiDeviceConfig::ci_overlay_path(CI_KEY_BASE + id));
            crate::scsi::attach_table(fc.bus(), id, &at, disk, ci_overlay.as_deref(), Some("iris"))?;
        }
        Ok(Card { fc })
    }

    /// Put the card in `device` of `bus`.
    pub fn attach(&self, bus: &PciBus, device: u8) {
        bus.attach(DevFn { device, function: 0 }, self.fc.clone());
    }
}

/// The snapshot manifest's record of each loop disk, so a restore finds the
/// image it was saved with.
pub fn disk_refs(card: &crate::slot_path::MachineSlotPath, disks: &Loop) -> Vec<DiskRef> {
    disks
        .iter()
        .map(|(&id, d)| DiskRef {
            at: card.loop_id(id),
            path: d.path.clone(),
            size_bytes: std::fs::metadata(&d.path).map(|m| m.len()).unwrap_or(0),
        })
        .collect()
}
