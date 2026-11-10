//! What the SN0 machine tests share: a scratch directory, a PROM stand-in, the
//! base configuration and a PCI function with nothing behind it.

use crate::config::{MachineConfig, MachineProfile, NetMode};
use crate::cpu::parts;
use std::path::{Path, PathBuf};

pub(crate) use crate::display_stop_tests::big_stack;

/// A directory that goes away with the value. `name` is unique across the
/// SN0 tests, since they run as threads of one process.
pub(crate) struct Scratch(pub PathBuf);
impl Scratch {
    pub(crate) fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("iris-sn0-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A PROM stand-in: a recognisable first word and a configuration block in
/// the shape the flash initialiser patches.
pub(crate) fn prom(dir: &Path) -> String {
    let mut p = vec![0u8; 0x1000];
    p[..4].copy_from_slice(&0x3c08_a600u32.to_be_bytes());
    p[0x68..0x70].copy_from_slice(&0x6970_3237_636f_6e66u64.to_be_bytes());
    let path = dir.join("prom.bin");
    std::fs::write(&path, p).unwrap();
    path.to_string_lossy().into_owned()
}

/// A headless `profile` with `nodes` node boards, 64 MB each, no network and
/// one disk. The serial ports are 0, so a machine that serves them takes a
/// free port of its own rather than one a test bound and let go; the monitor
/// port stays unset because no test starts a machine.
pub(crate) fn config(profile: MachineProfile, dir: &Path, nodes: u8) -> MachineConfig {
    let mut cfg = MachineConfig::default();
    cfg.machine.profile = profile;
    cfg.set_cpu(&parts::R10000_180);
    match profile {
        MachineProfile::Origin200Ip29 => assert_eq!(nodes, 1, "an Origin 200 has one node"),
        MachineProfile::Origin2000Ip27 => cfg.origin2000.nodes = nodes,
        MachineProfile::Onyx2Ip27 => cfg.onyx2.nodes = nodes,
        _ => panic!("{profile:?} is not an SN0 machine"),
    }
    cfg.banks = [64, 0, 0, 0, 0, 0, 0, 0];
    cfg.prom = prom(dir);
    cfg.nvram = dir.join("nvram.bin").to_string_lossy().into_owned();
    cfg.headless = true;
    cfg.network.mode = NetMode::Off;
    cfg.serial_port_a = Some(0);
    cfg.serial_port_b = Some(0);
    let disk = dir.join("disk.raw");
    std::fs::write(&disk, vec![0u8; 4 << 20]).unwrap();
    cfg.scsi.clear();
    cfg.scsi.insert(1, toml::from_str(&format!("path = {:?}\noverlay = true", disk.to_string_lossy())).unwrap());
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
