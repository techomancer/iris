//! The host ends of an IP35 board's serial connections: the L1 and the I/O
//! chip's ports, each on a TCP port or, for the console under CI, in the
//! process.

use crate::config::MachineConfig;
use crate::dev::serial_host::{null_serial_backend, open_log, tcp_serial_backend, CiSerialBackend, SerialBackend, TeeBackend};
use std::sync::Arc;

/// One of the board's serial connections: what it is called, the TCP port
/// and the setting that names the port, the CI backend it uses when it is
/// the console, the log it keeps when it is not, and where the host end goes.
pub(crate) struct SerialPort<'a> {
    pub label: &'static str,
    pub port: u16,
    pub setting: &'static str,
    pub ci: &'a Arc<CiSerialBackend>,
    pub log: Option<String>,
    pub connect: Box<dyn Fn(Arc<dyn SerialBackend>) + 'a>,
}

/// Connect the board's serial ports to the host. The console takes
/// `serial_log` and, under CI, the in-process backend; the others listen on
/// their TCP ports, and a port that is taken leaves its stream unconnected
/// rather than stopping the machine. Port 0 takes any free port, so several
/// lines may name it.
pub(crate) fn connect_serial_ports(cfg: &MachineConfig, name: &str, ports: &[SerialPort<'_>], console: usize) -> Result<(), String> {
    for (i, a) in ports.iter().enumerate() {
        if let Some(b) = ports[i + 1..].iter().find(|b| b.port == a.port && a.port != 0) {
            return Err(format!("{name} {} and {} serial ports must be distinct", a.label, b.label));
        }
    }
    let mut log_config = cfg.clone();
    for (index, port) in ports.iter().enumerate() {
        log_config.serial_log = if index == console { cfg.serial_log.clone() } else { port.log.clone() };
        let tag = format!("{name} {}", port.label);
        let log = open_log(&log_config, &tag);
        let backend: Arc<dyn SerialBackend> = if cfg.ci && index == console {
            if let Some(log) = log {
                port.ci.set_log(log);
            }
            port.ci.clone()
        } else {
            let mut backend = null_serial_backend();
            match tcp_serial_backend(port.port) {
                Some((socket, actual_port)) => {
                    eprintln!("iris: {tag} on 127.0.0.1:{actual_port}");
                    backend = socket;
                }
                None => eprintln!(
                    "iris: {tag}: port {} unavailable; is another iris using it? ({})",
                    port.port, port.setting
                ),
            }
            match log {
                Some(log) => Arc::new(TeeBackend::new(backend, log)),
                None => backend,
            }
        };
        (port.connect)(backend);
    }
    Ok(())
}
