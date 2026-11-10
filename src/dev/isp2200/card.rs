//! The adapter as `[fibre_channel]` describes it: opened with its loop disks,
//! then fitted to a slot of an option board.
//!
//! The same shape as the NVMe card's ([`crate::dev::nvme::Card`]). A disk is
//! attached through [`crate::scsi::attach_table`], the call every `[scsi.N]`
//! disk goes through, so the loop disks take whatever a disk table does
//! (overlay, write cache). The disks report the machine-wide disk identity
//! of [`Isp2200::new_bus`].

use std::sync::Arc;

use super::Isp2200;
use crate::config::{FibreChannelConfig, ScsiDeviceConfig};
use crate::dev::nvme::{BRIDGE_DEVICE, BRIDGE_VENDOR};
use crate::pci::{DevFn, PciBus, PciToPciBridge};
use crate::snapshot::DiskRef;

/// The adapter's number among the machine's SCSI adapters: the Octane's two
/// ISP1040s are 0 and 1, and the monitor's `scsi2` and `cow2` are this one.
pub const ADAPTER_INDEX: u8 = 2;

/// The snapshot manifest's id for the disk at loop ID 0; loop ID n is this
/// plus n. Clear of the SCSI slots (0 to 31) and the NVMe disk (0xf0).
pub const SNAPSHOT_BASE: u8 = 0x40;

/// The key a `--ci` run's overlay for the disk at `loop_id` is named by:
/// above every `[scsi.N]` key.
const CI_KEY_BASE: u8 = 0x80;

/// The card a `[fibre_channel]` section describes, before it is in a slot.
pub struct Card {
    pub fc: Arc<Isp2200>,
    bridge: bool,
}

impl Card {
    /// The adapter with its loop's disks attached. `ci` is a `--ci` run,
    /// which keeps each overlay in the temporary directory.
    pub fn open(cfg: &FibreChannelConfig, ci: bool) -> Result<Card, String> {
        let fc = Arc::new(Isp2200::new_at(ADAPTER_INDEX, Isp2200::new_bus()));
        for (&id, disk) in &cfg.disks {
            let at = format!("Fibre Channel loop ID {id}");
            let ci_overlay = (ci && disk.overlay).then(|| ScsiDeviceConfig::ci_overlay_path(CI_KEY_BASE + id));
            crate::scsi::attach_table(fc.bus(), id, &at, disk, ci_overlay.as_deref(), Some("iris"))?;
        }
        Ok(Card { fc, bridge: cfg.bridge })
    }

    /// Put the card in `device` of `bus`: in the slot itself, or behind a
    /// PCI-to-PCI bridge there, as device 0 of the bridge's bus.
    pub fn fit(&self, bus: &PciBus, device: u8) {
        let at = |device| DevFn { device, function: 0 };
        if self.bridge {
            let ppb = PciToPciBridge::new(bus, at(device), BRIDGE_VENDOR, BRIDGE_DEVICE);
            bus.attach(at(device), ppb.clone());
            ppb.secondary().attach(at(0), self.fc.clone());
        } else {
            bus.attach(at(device), self.fc.clone());
        }
    }
}

/// The snapshot manifest's record of each loop disk, so a restore finds the
/// image it was saved with.
pub fn disk_refs(cfg: &FibreChannelConfig) -> Vec<DiskRef> {
    cfg.disks
        .iter()
        .map(|(&id, d)| DiskRef {
            id: SNAPSHOT_BASE + id,
            path: d.path.clone(),
            size_bytes: std::fs::metadata(&d.path).map(|m| m.len()).unwrap_or(0),
        })
        .collect()
}
