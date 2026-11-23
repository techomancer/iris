//! What the IP27 machine tests share: a scratch directory, a PROM stand-in, the
//! base configuration and a PCI function with nothing behind it.

use crate::config::{MachineConfig, MachineProfile};
use crate::cpu::parts;
use std::path::Path;

pub(crate) use crate::display_stop_tests::big_stack;

/// A headless `profile` with `nodes` node boards, 64 MB each, no network and
/// one disk. The serial ports are 0, so a machine that serves them takes a
/// free port of its own rather than one a test bound and let go; the monitor
/// port stays unset because no test starts a machine.
pub(crate) fn config(profile: MachineProfile, dir: &Path, nodes: u8) -> MachineConfig {
    let mut cfg = crate::sn::tests::headless(dir, profile);
    cfg.set_cpu(&parts::R10000_180);
    match profile {
        MachineProfile::Origin200Ip29 => assert_eq!(nodes, 1, "an Origin 200 has one node"),
        MachineProfile::Origin2000Ip27 => cfg.origin2000.nodes = nodes,
        MachineProfile::Onyx2Ip27 => cfg.onyx2.nodes = nodes,
        _ => panic!("{profile:?} is not an IP27 machine"),
    }
    cfg.banks = [64, 0, 0, 0, 0, 0, 0, 0];
    let disk = dir.join("disk.raw");
    std::fs::write(&disk, vec![0u8; 4 << 20]).unwrap();
    cfg.devices.clear();
    cfg.set_disk(1, toml::from_str(&format!("path = {:?}\noverlay = true", disk.to_string_lossy())).unwrap());
    cfg
}

/// A PCI function with nothing behind it but the DMA and interrupt line the
/// Bridge hands it.
pub(crate) struct Card {
    pub(crate) state: crate::pci::PciFunctionState,
}

impl Card {
    pub(crate) fn new() -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self {
            state: crate::pci::PciFunctionState::new(crate::pci::PciConfig::new(
                0x1234,
                0x5678,
                0x01_0000,
                1,
                [crate::pci::Bar::Unused; 6],
                crate::pci::PCI_INTA,
            )),
        })
    }
}

impl crate::pci::PciFunction for Card {
    fn state(&self) -> &crate::pci::PciFunctionState {
        &self.state
    }
    fn bar_read(&self, _bar: usize, _off: u32, _width: usize) -> u32 {
        0
    }
    fn bar_write(&self, _bar: usize, _off: u32, _val: u32, _width: usize) {}
}

/// The machine `cfg`'s profile names, built as `workstation::build` builds it.
pub(crate) fn machine(cfg: &MachineConfig) -> Result<crate::sn::Machine<crate::dev::hub::Hub, crate::dev::node_flash::NodeFlash>, String> {
    use crate::{ip27_machine, ip29_machine};
    match cfg.machine.profile {
        MachineProfile::Origin200Ip29 => crate::ip27::machine(cfg, ip29_machine::node_specs, ip29_machine::build_io),
        _ => crate::ip27::machine(cfg, ip27_machine::node_specs, ip27_machine::build_io),
    }
}
