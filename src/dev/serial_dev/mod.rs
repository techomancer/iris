//! What a config puts on a guest serial port with a `[[devices]]` entry at `serialN`, the number printed on the
//! machine's case: `serial`, the port's TCP, Unix socket and log ends, which a port with no entry gets as its default
//! listener, or `hostserial`, `midi_synth` and `midi_raw` in place of them. Each is a [`SerialBackend`].
//!
//! `hostserial` is a host serial port ([`host_tty`]) that follows the guest's rate, framing and modem lines, so a
//! real device behaves as it would on the machine's own connector.
//! `midi_synth` plays on a SoundFont synth ([`midi`]); `midi_raw` passes the
//! bytes to a host MIDI port. Devices made from host input events (a tablet,
//! a 6DOF mouse) come later; see `clanker/plans/serial-peripherals.md`.

pub mod host_tty;
pub mod midi;

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::config::MachineConfig;
use crate::dev::serial_host::{ends, null_serial_backend, End, open_line, open_log, CiSerialBackend, Console, SerialBackend, TeeBackend};
use crate::dev::serial_log::SerialLog;
use crate::devices::DeviceModel;
use crate::topology::DevicePlacement;

/// Which serial device a `[[devices]]` entry is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SerialKind {
    /// The port's host ends: TCP, Unix sockets and logs.
    Line,
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
            SerialKind::Line => "serial",
            SerialKind::HostSerial => "hostserial",
            SerialKind::MidiSynth => "midi_synth",
            SerialKind::MidiRaw => "midi_raw",
        }
    }

    pub fn from_name(name: &str) -> Option<SerialKind> {
        [SerialKind::Line, SerialKind::HostSerial, SerialKind::MidiSynth, SerialKind::MidiRaw].into_iter().find(|k| k.name() == name)
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
    /// `serial`: the host ends, one string or a list: `tcp:PORT`, `unix:PATH` or `log:PATH`. None leaves the port unplugged.
    #[serde(default, deserialize_with = "crate::backend::list", skip_serializing_if = "Vec::is_empty")]
    pub backend: Vec<String>,
    /// `midi_synth`: how loud it plays, times the synth's own level (1.0),
    /// which leaves room for many voices; the host mix clips above full scale.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gain: Option<f32>,
}

impl SerialDeviceConfig {
    /// The first TCP port a `serial` entry listens on.
    pub fn tcp_port(&self) -> Option<u16> {
        ends(&self.backend).ok()?.into_iter().find_map(|e| if let End::Tcp(p) = e { Some(p) } else { None })
    }

    /// The value of the entry's first `kind:` end (`unix` or `log`).
    pub fn end(&self, kind: &str) -> Option<&str> {
        self.backend.iter().find_map(|b| b.strip_prefix(kind)?.strip_prefix(':'))
    }

    /// Replace the entry's `kind:` end with `value`; none drops it.
    pub fn set_end(&mut self, kind: &str, value: Option<&str>) {
        self.backend.retain(|b| b.strip_prefix(kind).and_then(|r| r.strip_prefix(':')).is_none());
        self.backend.extend(value.map(|v| format!("{kind}:{v}")));
    }

    /// Listen on `port` instead, keeping the entry's other ends; none drops the listener.
    pub fn set_tcp_port(&mut self, port: Option<u16>) {
        self.backend.retain(|b| !b.starts_with("tcp:"));
        self.backend.extend(port.map(|p| format!("tcp:{p}")));
    }

    pub fn check(&self) -> Result<(), String> {
        let needs = match self.kind {
            SerialKind::Line => return ends(&self.backend).map(drop),
            SerialKind::HostSerial => "needs a path: a host serial port such as /dev/ttyUSB0",
            SerialKind::MidiSynth => "needs a path: a General MIDI SoundFont (.sf2)",
            SerialKind::MidiRaw => "needs a path: a host MIDI port such as /dev/snd/midiC1D0, or a serial port",
        };
        if self.path.is_empty() {
            return Err(needs.into());
        }
        if self.gain.is_some_and(|g| !(0.0..=16.0).contains(&g)) {
            return Err("gain is from 0 to 16".into());
        }
        Ok(())
    }
}

/// The serial entries of a config that have a port, with the port's default listener: (case number, the entry,
/// whether it is the port's default).
fn placed(cfg: &MachineConfig) -> Vec<(u8, SerialDeviceConfig, bool)> {
    let p = DevicePlacement::lenient(cfg, &cfg.devices);
    p.placed().filter_map(|(d, at)| match &d.device {
        DeviceModel::Serial(s) => Some((at.serial_port()?, s.clone(), d.default)),
        _ => None,
    }).collect()
}

/// The type of the entry on serial port `n`, if one is there.
pub fn section_at(cfg: &MachineConfig, n: u8) -> Option<&'static str> {
    placed(cfg).into_iter().find(|(p, _, default)| *p == n && !default).map(|(_, d, _)| d.kind.name())
}

/// Each serial device names its port: left to choose, it would take the console. Which ports exist and that each
/// takes one device, placement checks.
pub fn validate(cfg: &MachineConfig) -> Result<(), String> {
    console(cfg)?;
    for d in &cfg.devices {
        let DeviceModel::Serial(s) = &d.device else { continue };
        if d.at.is_none() {
            let names: Vec<String> = ports(cfg.machine.profile).iter().map(|p| format!("\"serial{}\"", p.0)).collect();
            return Err(format!("[[devices]] {} needs an at: this machine's serial ports are {}", s.kind.name(), names.join(", ")));
        }
        s.check().map_err(|e| format!("[[devices]] {}: {e}", s.kind.name()))?;
    }
    Ok(())
}

/// The names `machine.console` takes, the machine's own console first: `l1` where it has one, then its serial ports.
pub fn consoles(profile: crate::config::MachineProfile) -> Vec<String> {
    let l1 = matches!(profile, crate::config::MachineProfile::FuelIp34 | crate::config::MachineProfile::TezroIp53 | crate::config::MachineProfile::Origin3200Ip35 | crate::config::MachineProfile::Onyx3000Ip35);
    l1.then(|| "l1".to_string()).into_iter().chain(ports(profile).iter().map(|p| format!("serial{}", p.0))).collect()
}

/// The serial port that serves CI and `serial_log`: `machine.console` (`serialN`, or `l1` on a machine with an L1
/// console, which is not a serial port), else the machine's own. `None` is the L1.
pub fn console(cfg: &MachineConfig) -> Result<Option<u8>, String> {
    let profile = cfg.machine.profile;
    let names = consoles(profile);
    let Some(name) = cfg.machine.console.as_deref() else { return Ok(names[0].strip_prefix("serial").and_then(|n| n.parse().ok())) };
    if !names.iter().any(|n| n == name) {
        return Err(format!("machine.console \"{name}\" is not a console of the {}: write {}", crate::topology::machine_name(profile), names.join(", ")));
    }
    Ok(name.strip_prefix("serial").and_then(|n| n.parse().ok()))
}

/// The TCP port of the console on 127.0.0.1, for a front end to connect to.
pub fn console_port(cfg: &MachineConfig) -> Option<u16> {
    use crate::config::MachineProfile as P;
    match console(cfg).ok()? {
        None => match cfg.machine.profile {
            P::FuelIp34 => Some(cfg.fuel.l1_port.unwrap_or(8882)),
            P::TezroIp53 => Some(cfg.tezro.l1_port.unwrap_or(8882)),
            _ => Some(cfg.ip35_section().l1_port.unwrap_or(8882)),
        },
        Some(n) => {
            placed(cfg).into_iter().find(|p| p.0 == n)?.1.tcp_port()
        }
    }
}

/// The host end of serial port `n` of machine `tag`, `board` being the chip's name for it: the entry placed there, or
/// the default listener. The console also takes `ci` and `log` (`serial_log`, opened here if the caller has none).
pub fn connect(cfg: &MachineConfig, tag: &str, board: &str, n: u8, ci: Option<&Arc<CiSerialBackend>>, log: Option<&Arc<SerialLog>>) -> Result<Arc<dyn SerialBackend>, String> {
    let label = format!("{tag} serial{n}");
    let console = (console(cfg)? == Some(n)).then(|| Console { ci: ci.filter(|_| cfg.ci), log: log.cloned().or_else(|| open_log(cfg, &label)) });
    let is_console = console.is_some();
    let backend = match placed(cfg).into_iter().find(|p| p.0 == n) {
        Some((_, d, default)) if d.kind == SerialKind::Line => open_line(cfg, &label, &ends(&d.backend)?, default, console),
        Some((_, d, _)) => {
            let b = open(n, &d, cfg)?;
            match console {
                Some(c) => {
                    log::warn!("{label} is the console port; the console now goes to the {}", d.kind.name());
                    match c.log {
                        Some(l) => Arc::new(TeeBackend::new(b, l)),
                        None => b,
                    }
                }
                None => b,
            }
        }
        None => null_serial_backend(),
    };
    Ok(crate::dev::serial_map::mapped(tag, board, Some(n), is_console, cfg, backend))
}

/// Wire a board with an L1 console: the L1 (TCP on 8882 unless `l1` sets one, and its log) into `set[0]`, the I/O chip's
/// two ports into the others. Returns the CI console under `--ci`.
pub fn connect_l1_board(cfg: &MachineConfig, tag: &str, chip: [&str; 2], l1: (Option<u16>, Option<&str>), set: [&dyn Fn(Arc<dyn SerialBackend>); 3]) -> Result<Option<Arc<CiSerialBackend>>, String> {
    let ci = cfg.ci.then(|| Arc::new(CiSerialBackend::new()));
    let label = format!("{tag} L1");
    let console = console(cfg)?.is_none().then(|| Console { ci: ci.as_ref(), log: open_log(cfg, &label) });
    let mut ends = vec![End::Tcp(l1.0.unwrap_or(8882))];
    ends.extend(l1.1.filter(|_| console.is_none()).map(|log| End::Log(log.into())));
    let is_console = console.is_some();
    set[0](crate::dev::serial_map::mapped(tag, "L1", None, is_console, cfg, open_line(cfg, &label, &ends, l1.0.is_none(), console)));
    for ((n, _), (board, set)) in ports(cfg.machine.profile).iter().zip(chip.into_iter().zip(&set[1..])) {
        set(connect(cfg, tag, board, *n, ci.as_ref(), None)?);
    }
    Ok(ci)
}

/// The device `d` on serial port `n`. An error means its host end could not be opened.
fn open(n: u8, d: &SerialDeviceConfig, cfg: &MachineConfig) -> Result<Arc<dyn SerialBackend>, String> {
    let (name, path) = (d.kind.name(), d.path.as_str());
    match d.kind {
        SerialKind::Line => unreachable!("a line is opened by connect"),
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
    }
}

/// The serial ports a machine wires to the host: each one's number as the case prints it, and default TCP port. Where
/// the case prints none (an I-brick's single SERIAL), they are numbered from 1 in chip order.
pub fn ports(profile: crate::config::MachineProfile) -> &'static [(u8, Option<u16>)] {
    use crate::config::MachineProfile as P;
    match profile {
        P::IndyIp24 | P::Indigo2Ip22 | P::Indigo2Ip28 => &[(1, Some(8881)), (2, Some(8880))],
        // The case's DB9s are 1 and 2; they are the IOC4's ports 0 and 1.
        P::TezroIp53 => &[(1, Some(8880)), (2, Some(8881))],
        // The case prints tty_1 to tty_4: the BaseIO's two, then the MediaIO card's.
        P::Onyx2Ip27 => &[(1, Some(8880)), (2, Some(8881)), (3, None), (4, None)],
        _ => &[(1, Some(8880)), (2, Some(8881))],
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
        let tezro = "[[devices]]\ntype = \"midi_raw\"\nat = \"serial2\"\npath = \"/dev/midi3\"\n";
        assert_eq!(section_at(&cfg(MachineProfile::TezroIp53, tezro).unwrap(), 2), Some("midi_raw"), "a Tezro's DB9s are 1 and 2");
        let third = tezro.replace("serial2", "serial3");
        assert!(cfg(indy, &third).is_err() && cfg(MachineProfile::TezroIp53, &third).is_err(), "neither has a serial3");
        let anywhere = "[[devices]]\ntype = \"hostserial\"\npath = \"/dev/ttyUSB0\"\n";
        assert!(cfg(indy, anywhere).unwrap_err().contains("needs an at"), "{}", cfg(indy, anywhere).unwrap_err());
        let no_path = "[[devices]]\ntype = \"hostserial\"\nat = \"serial1\"\n";
        assert!(cfg(indy, no_path).unwrap_err().contains("needs a path"));
    }

    /// A `serial` entry's Unix socket and log ends are read and replaced one at a time, leaving its TCP end alone.
    #[test]
    fn a_line_entry_edits_its_unix_and_log_ends() {
        let mut s = SerialDeviceConfig { kind: SerialKind::Line, backend: vec!["tcp:8880".into()], ..Default::default() };
        s.set_end("unix", Some("/run/a"));
        s.set_end("log", Some("a.log"));
        s.set_end("unix", Some("/run/b"));
        assert_eq!((s.end("unix"), s.end("log"), s.tcp_port()), (Some("/run/b"), Some("a.log"), Some(8880)));
        s.set_end("log", None);
        assert_eq!(s.backend, ["tcp:8880", "unix:/run/b"]);
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

    /// A port with no entry listens on its default TCP port, which an entry for the port replaces. The Indy's console,
    /// serial1, is channel B on 8881.
    #[test]
    fn a_port_with_no_entry_gets_its_default_listener() {
        let lines = |c: &MachineConfig| placed(c).into_iter().map(|(n, d, default)| (n, d.backend.join(" "), default)).collect::<Vec<_>>();
        let indy = cfg(MachineProfile::IndyIp24, "").unwrap();
        assert_eq!(lines(&indy), [(1, "tcp:8881".into(), true), (2, "tcp:8880".into(), true)]);
        let entry = "[[devices]]\ntype = \"serial\"\nat = \"serial2\"\nbackend = [\"unix:/run/a\", \"log:a.log\"]\n";
        assert_eq!(lines(&cfg(MachineProfile::IndyIp24, entry).unwrap()), [(2, "unix:/run/a log:a.log".into(), false), (1, "tcp:8881".into(), true)]);
        assert_eq!(lines(&cfg(MachineProfile::TezroIp53, "").unwrap()), [(1, "tcp:8880".into(), true), (2, "tcp:8881".into(), true)]);
        let (onyx2, origin) = (cfg(MachineProfile::Onyx2Ip27, "").unwrap(), cfg(MachineProfile::Origin2000Ip27, "").unwrap());
        assert_eq!((lines(&onyx2).len(), lines(&origin).len()), (2, 2), "the MediaIO's serial3 and serial4 have no listener");
        let mio = entry.replace("serial2", "serial4");
        assert!(cfg(MachineProfile::Onyx2Ip27, &mio).is_ok() && cfg(MachineProfile::Origin2000Ip27, &mio).is_err());
    }

    /// `machine.console` names a serial port or the L1; the old IOC names are refused with the ports to write.
    #[test]
    fn the_console_is_a_port_name() {
        let with = |profile, name: Option<&str>| {
            let mut c = MachineConfig::default();
            c.machine.profile = profile;
            c.machine.console = name.map(Into::into);
            console(&c)
        };
        assert_eq!(with(MachineProfile::IndyIp24, None), Ok(Some(1)));
        assert_eq!(with(MachineProfile::FuelIp34, None), Ok(None), "the L1");
        assert_eq!(with(MachineProfile::FuelIp34, Some("serial2")), Ok(Some(2)));
        assert_eq!(with(MachineProfile::TezroIp53, Some("serial2")), Ok(Some(2)));
        assert_eq!(with(MachineProfile::OctaneIp30, Some("serial2")), Ok(Some(2)));
        let e = with(MachineProfile::FuelIp34, Some("ioc3_a")).unwrap_err();
        assert!(e.contains("write l1, serial1, serial2"), "{e}");
        assert!(with(MachineProfile::OctaneIp30, Some("l1")).unwrap_err().contains("write serial1, serial2"));
        assert!(with(MachineProfile::TezroIp53, Some("serial3")).is_err(), "the Tezro has two");
    }
}
