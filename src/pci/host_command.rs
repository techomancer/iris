//! The `pci` monitor command, the same on every machine with a PCI host.

use std::io::Write;
use std::sync::Arc;

use super::PciBus;
use crate::traits::Device;

/// One PCI host's buses, answering `pci status` and `pci config <slot>`.
struct PciHostCommands {
    label: String,
    buses: Vec<PciBus>,
    status: Option<StatusHook>,
}

/// Extra lines a host prints under its header in `pci status`.
pub type StatusHook = Box<dyn Fn(&mut dyn Write) + Send + Sync>;

/// The `pci` command for the host `label` with `buses`, for a machine that
/// collects its monitor devices before it has a monitor. `status` adds a
/// host's own lines to `pci status`.
pub fn pci_host_commands(label: &str, buses: &[&PciBus], status: Option<StatusHook>) -> Arc<dyn Device> {
    Arc::new(PciHostCommands { label: label.to_string(), buses: buses.iter().map(|&b| b.clone()).collect(), status })
}

impl Device for PciHostCommands {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) {}
    fn start(&self) {}
    fn is_running(&self) -> bool { true }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![("pci".to_string(), "PCI hosts: pci status | pci config <slot>".to_string())]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut w: Box<dyn Write + Send>) -> Result<(), String> {
        if cmd != "pci" {
            return Err("Command not found".to_string());
        }
        let mut done = false;
        let mut err = None;
        for (i, bus) in self.buses.iter().enumerate() {
            let _ = match self.buses.len() {
                1 => writeln!(w, "{}:", self.label),
                _ => writeln!(w, "{} bus {i}:", self.label),
            };
            if let (0, Some(hook), "status") = (i, &self.status, args.first().copied().unwrap_or("status")) {
                hook(&mut w);
            }
            match bus.command(args, &mut w) {
                Ok(()) => done = true,
                Err(e) => err = Some(e),
            }
        }
        match err {
            Some(e) if !done => Err(format!("{}: {e}", self.label)),
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::config::{GraphicsBoard, MachineConfig, MachineProfile as P};
    use crate::display_stop_tests::{big_stack, scratch};
    use std::path::Path;

    fn config(path: &Path, profile: P) -> MachineConfig {
        match profile {
            P::O2Ip32 | P::OctaneIp30 => crate::display_stop_tests::config(path, profile, None),
            P::Origin200Ip29 | P::Origin2000Ip27 | P::Onyx2Ip27 => crate::ip27::test_support::config(profile, path, 1),
            P::FuelIp34 => crate::ip35::test_support::fuel_config(path),
            P::TezroIp53 => crate::ip35::test_support::tezro_config(path, 2),
            P::Origin3200Ip35 => crate::ip35::test_support::config(path, 1),
            _ => {
                let mut cfg = crate::ip35::test_support::headless(path, profile);
                cfg.onyx3000.l1_port = Some(0);
                cfg
            }
        }
    }

    /// Each PCI machine's monitor has the `pci` command, and its status names
    /// the host and lists a function behind it.
    #[test]
    fn every_pci_machine_answers_pci_status_naming_its_host() {
        let machines = [
            (P::O2Ip32, "MACE:"),
            (P::OctaneIp30, "Bridge:"),
            (P::Origin200Ip29, "Bridge:"),
            (P::Origin2000Ip27, "Bridge:"),
            (P::Onyx2Ip27, "Bridge:"),
            (P::FuelIp34, "XBridge bus 1:"),
            (P::Origin3200Ip35, "XBridge bus 1:"),
            (P::Onyx3000Ip35, "XBridge bus 1:"),
            (P::TezroIp53, "PIC at widget 11 bus 0:"),
        ];
        for (profile, host) in machines {
            let dir = scratch("pci-host");
            let cfg = config(&dir, profile);
            let status = big_stack(move || crate::workstation::build(cfg).unwrap().monitor().run("pci", &["status"]));
            assert!(status.contains(host), "{profile:?}: {status}");
            assert!(status.contains(", command "), "{profile:?} lists no function: {status}");
            assert_eq!(status.contains("DMA writes that found no memory"), profile == P::O2Ip32, "{profile:?}: {status}");
        }
    }
}
