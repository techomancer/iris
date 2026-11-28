//! The host ends of an IP35 board's serial connections: the L1 and the I/O
//! chip's ports, each on a TCP port or, for the console under CI, in the
//! process.

use crate::config::MachineConfig;
use crate::dev::serial_host::{open_log, serial_line, CiSerialBackend, SerialBackend, SerialLine};
use std::sync::Arc;

/// One of the board's serial connections: what it is called, the TCP port
/// and the setting that names the port, the CI backend it uses when it is
/// the console, the log it keeps when it is not, a Unix socket with the setting that names it,
/// and where the host end goes.
pub(crate) struct SerialPort {
    pub label: &'static str,
    pub port: u16,
    pub setting: &'static str,
    pub ci: Arc<CiSerialBackend>,
    pub log: Option<String>,
    pub socket: Option<(String, &'static str)>,
    /// The IRIX port it is (N of ttydN), which a `[hostserial]`,
    /// `[spaceball]` or `[wacom]` section names as `serialN`; none for the L1.
    pub ttyd: Option<u8>,
    pub connect: Box<dyn Fn(Arc<dyn SerialBackend>) + Send + Sync>,
}

/// Connect the board's serial ports to the host: the console takes `serial_log` and, under CI, the
/// in-process backend. A taken TCP port leaves its stream unconnected; port 0 takes any free port.
pub(crate) fn connect_serial_ports(cfg: &MachineConfig, name: &str, ports: &[SerialPort], console: usize) -> Result<(), String> {
    for (i, a) in ports.iter().enumerate() {
        if let Some(b) = ports[i + 1..].iter().find(|b| b.port == a.port && a.port != 0) {
            return Err(format!("{name} {} and {} serial ports must be distinct", a.label, b.label));
        }
    }
    let mut log_config = cfg.clone();
    for (index, port) in ports.iter().enumerate() {
        log_config.serial_log = if index == console { cfg.serial_log.clone() } else { port.log.clone() };
        let tag = format!("{name} {}", port.label);
        let console_mark = if index == console { " (console)" } else { "" };
        let log = open_log(&log_config, &tag);
        if let Some(b) = port.ttyd.map(|n| crate::dev::serial_dev::peripheral(cfg, n, log.clone())).transpose()?.flatten() {
            if index == console {
                log::warn!("{tag}: this is the console port; the console now goes to the peripheral");
            }
            (port.connect)(crate::dev::serial_map::mapped(port.label, port.ttyd, index == console, cfg, b));
            continue;
        }
        let line = SerialLine {
            label: &format!("{tag}{console_mark}"),
            tcp_port: Some(port.port),
            tcp_setting: port.setting,
            socket: port.socket.as_ref().map(|(path, _)| path.as_str()),
            socket_setting: port.socket.as_ref().map_or("", |(_, setting)| setting),
        };
        let (backend, _) = serial_line(&line, log, (cfg.ci && index == console).then_some(&port.ci));
        (port.connect)(crate::dev::serial_map::mapped(port.label, port.ttyd, index == console, cfg, backend));
    }
    Ok(())
}
