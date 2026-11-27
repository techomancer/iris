//! Devices on a guest serial port: the `[[devices]]` types `hostserial`,
//! `midi_synth` and `midi_raw`, which a config puts at `serialN`, IRIX's
//! `/dev/ttydN`, in place of the port's TCP or Unix socket.
//!
//! Each is a [`SerialBackend`]. `hostserial` is a host serial port
//! ([`host_tty`]) that follows the guest's rate, framing and modem lines, so a
//! real device behaves as it would on the machine's own connector.
//! `midi_synth` plays on a SoundFont synth ([`midi`]); `midi_raw` passes the
//! bytes to a host MIDI port. Devices made from host input events (a tablet,
//! a 6DOF mouse) come later; see `clanker/plans/serial-peripherals.md`.

pub mod host_tty;
pub mod midi;

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::config::MachineConfig;
use crate::dev::serial_host::SerialBackend;
use crate::devices::DeviceModel;
use crate::topology::DevicePlacement;

/// Which serial device a `[[devices]]` entry is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SerialKind {
    /// A host serial port, passed through.
    #[default]
    HostSerial,
    /// The SoundFont synth.
    MidiSynth,
    /// A host MIDI port: a raw MIDI device, or a serial port with a MIDI interface.
    MidiRaw,
}

impl SerialKind {
    pub fn name(self) -> &'static str {
        match self {
            SerialKind::HostSerial => "hostserial",
            SerialKind::MidiSynth => "midi_synth",
            SerialKind::MidiRaw => "midi_raw",
        }
    }

    pub fn from_name(name: &str) -> Option<SerialKind> {
        [SerialKind::HostSerial, SerialKind::MidiSynth, SerialKind::MidiRaw].into_iter().find(|k| k.name() == name)
    }
}

/// A serial device's options. `path` is the host end: the serial port for
/// `hostserial`, the SoundFont for `midi_synth`, the MIDI port for `midi_raw`.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SerialDeviceConfig {
    #[serde(skip)]
    pub kind: SerialKind,
    #[serde(default)]
    pub path: String,
    /// `midi_synth`: how loud it plays, times the synth's own level (1.0),
    /// which leaves room for many voices; the host mix clips above full scale.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gain: Option<f32>,
}

impl SerialDeviceConfig {
    pub fn check(&self) -> Result<(), String> {
        if self.path.is_empty() {
            return Err(match self.kind {
                SerialKind::HostSerial => "needs a path: a host serial port such as /dev/ttyUSB0",
                SerialKind::MidiSynth => "needs a path: a General MIDI SoundFont (.sf2)",
                SerialKind::MidiRaw => "needs a path: a host MIDI port such as /dev/snd/midiC1D0, or a serial port",
            }
            .into());
        }
        if self.gain.is_some_and(|g| !(0.0..=16.0).contains(&g)) {
            return Err("gain is from 0 to 16".into());
        }
        Ok(())
    }
}

/// The serial devices of a config that have a slot: (IRIX ttyd number, the
/// device).
fn placed(cfg: &MachineConfig) -> Vec<(u8, &SerialDeviceConfig)> {
    if !cfg.devices.iter().any(|d| matches!(d.device, DeviceModel::Serial(_))) {
        return Vec::new();
    }
    let p = DevicePlacement::lenient(cfg, &cfg.devices);
    cfg.devices
        .iter()
        .enumerate()
        .filter_map(|(i, d)| match &d.device {
            DeviceModel::Serial(s) => Some((p.at(i)?.serial_port()?, s)),
            _ => None,
        })
        .collect()
}

/// The type of the device on serial port `n`, if one is there.
pub fn section_at(cfg: &MachineConfig, n: u8) -> Option<&'static str> {
    placed(cfg).into_iter().find(|(p, _)| *p == n).map(|(_, d)| d.kind.name())
}

/// Each serial device names its port: left to choose, it would take the
/// console. Which ports exist and that each takes one device, placement checks.
pub fn validate(cfg: &MachineConfig) -> Result<(), String> {
    for d in &cfg.devices {
        let DeviceModel::Serial(s) = &d.device else { continue };
        if d.at.is_none() {
            let names: Vec<String> = ports(cfg.machine.profile).iter().map(|p| format!("\"serial{p}\"")).collect();
            return Err(format!("[[devices]] {} needs an at: this machine's serial ports are {}", s.kind.name(), names.join(", ")));
        }
        s.check().map_err(|e| format!("[[devices]] {}: {e}", s.kind.name()))?;
    }
    Ok(())
}

/// The backend for serial port `n` (IRIX's ttydN) if a device is there. An
/// error means one is there but its host end could not be opened.
pub fn backend_for_port(cfg: &MachineConfig, n: u8) -> Option<Result<Arc<dyn SerialBackend>, String>> {
    let (_, d) = placed(cfg).into_iter().find(|(p, _)| *p == n)?;
    let (name, path) = (d.kind.name(), d.path.as_str());
    Some(match d.kind {
        SerialKind::HostSerial | SerialKind::MidiRaw => {
            let open = if d.kind == SerialKind::MidiRaw { host_tty::HostTty::open_midi(path) } else { host_tty::HostTty::open(path) };
            open.map(|t| {
                log::info!("serial{n}: {name} passes through to {path}");
                Arc::new(t) as Arc<dyn SerialBackend>
            })
            .map_err(|e| format!("{name} at serial{n}: {path}: {e}"))
        }
        SerialKind::MidiSynth => midi::MidiSynth::open(path, d.gain.unwrap_or(1.0), &cfg.audio, cfg.no_audio)
            .map(|m| {
                log::info!("serial{n}: midi_synth plays on the SoundFont synth, {path}");
                Arc::new(m) as Arc<dyn SerialBackend>
            })
            .map_err(|e| format!("midi_synth at serial{n}: {e}")),
    })
}

/// The backend for serial port `n` when a device is there, teed to `log`
/// (the port's serial log) when there is one; `None` when no device is
/// there, and the port gets its socket as before.
pub fn peripheral(cfg: &MachineConfig, n: u8, log: Option<Arc<crate::dev::serial_log::SerialLog>>) -> Result<Option<Arc<dyn SerialBackend>>, String> {
    Ok(backend_for_port(cfg, n).transpose()?.map(|b| match log {
        Some(l) => Arc::new(crate::dev::serial_host::TeeBackend::new(b, l)) as Arc<dyn SerialBackend>,
        None => b,
    }))
}

/// The serial ports a device can take: the IRIX ttydN numbers of the ports a
/// machine wires to the host. On the Tezro, ttyd1 is the L1 console and IRIX
/// names the IOC4's first two ports ttyd3 and ttyd4 (it makes no devices for
/// the other two).
pub fn ports(profile: crate::config::MachineProfile) -> &'static [u8] {
    match profile {
        crate::config::MachineProfile::TezroIp53 => &[3, 4],
        _ => &[1, 2],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MachineProfile;

    fn cfg(profile: MachineProfile, devices: &str) -> Result<MachineConfig, String> {
        let mut c: MachineConfig = toml::from_str(devices).map_err(|e| e.to_string())?;
        c.machine.profile = profile;
        c.resolve_slots()?;
        c.validate()?;
        Ok(c)
    }

    /// A serial device names a port the machine has, one device each, with a path.
    #[test]
    fn a_serial_device_names_a_port_the_machine_has_once() {
        let indy = MachineProfile::IndyIp24;
        let c = cfg(indy, "[[devices]]\ntype = \"hostserial\"\nat = \"serial2\"\npath = \"/dev/ttyUSB0\"\n").unwrap();
        assert_eq!(section_at(&c, 2), Some("hostserial"));
        assert_eq!(section_at(&c, 1), None);
        let two = "[[devices]]\ntype = \"hostserial\"\nat = \"serial2\"\npath = \"/dev/ttyUSB0\"\n[[devices]]\ntype = \"midi_raw\"\nat = \"serial2\"\npath = \"/dev/midi3\"\n";
        assert!(cfg(indy, two).unwrap_err().contains("taken"), "{}", cfg(indy, two).unwrap_err());
        let tezro = "[[devices]]\ntype = \"midi_raw\"\nat = \"serial4\"\npath = \"/dev/midi3\"\n";
        assert_eq!(section_at(&cfg(MachineProfile::TezroIp53, tezro).unwrap(), 4), Some("midi_raw"), "a Tezro's are ttyd3 and ttyd4");
        assert!(cfg(indy, tezro).is_err(), "an Indy has no serial4");
        let first = "[[devices]]\ntype = \"hostserial\"\nat = \"serial1\"\npath = \"/dev/ttyUSB0\"\n";
        assert!(cfg(MachineProfile::TezroIp53, first).is_err(), "ttyd1 is the Tezro's L1 console");
        let anywhere = "[[devices]]\ntype = \"hostserial\"\npath = \"/dev/ttyUSB0\"\n";
        assert!(cfg(indy, anywhere).unwrap_err().contains("needs an at"), "{}", cfg(indy, anywhere).unwrap_err());
        let no_path = "[[devices]]\ntype = \"hostserial\"\nat = \"serial1\"\n";
        assert!(cfg(indy, no_path).unwrap_err().contains("needs a path"));
    }

    /// The MIDI types: a SoundFont for the synth, with a gain in range.
    #[test]
    fn midi_types_take_their_paths() {
        let indy = MachineProfile::IndyIp24;
        let synth = "[[devices]]\ntype = \"midi_synth\"\nat = \"serial2\"\npath = \"x.sf2\"\ngain = 2.0\n";
        let c = cfg(indy, synth).unwrap();
        assert_eq!(section_at(&c, 2), Some("midi_synth"));
        assert!(cfg(indy, &synth.replace("2.0", "20.0")).unwrap_err().contains("gain"));
        let raw_gain = "[[devices]]\ntype = \"midi_raw\"\nat = \"serial2\"\npath = \"/dev/midi3\"\ngain = 2.0\n";
        assert!(cfg(indy, raw_gain).unwrap_err().contains("no key gain"));
    }

    /// A port no device names keeps its socket.
    #[test]
    fn an_unnamed_port_gets_no_peripheral() {
        let c = cfg(MachineProfile::IndyIp24, "[[devices]]\ntype = \"hostserial\"\nat = \"serial2\"\npath = \"/dev/ttyUSB0\"\n").unwrap();
        assert!(peripheral(&c, 1, None).unwrap().is_none());
    }
}
