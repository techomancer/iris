//! The in-app serial console: which port the machine's console listens on,
//! the connection, scrollback, typed input and the status-bar serial light.
//!
//! The core has no serial counter, so the light is the GUI's own: lit while
//! this console is connected (plan, "Status bar").

use std::sync::Arc;

use super::icons::Icon;
use super::session::{DeviceLight, LightState, LightTarget};
use crate::model::job::Notify;
use crate::model::serial_console::{console_port, SerialConsole};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SerialAction {
    Open,
    Close,
    /// Send a typed line; a newline is added.
    SendLine(String),
    /// Send raw bytes (a key the shell forwards, Ctrl+C as 0x03).
    SendBytes(Vec<u8>),
    Clear,
}

/// What the console window shows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SerialView {
    pub text: String,
    pub connected: bool,
    /// "Connected to 127.0.0.1:8881", "Not connected", or the error.
    pub status: String,
    pub error: bool,
    /// Bumped on every change, so a shell can skip an unchanged redraw.
    pub seq: u64,
}

#[derive(Default)]
pub struct SerialConsoleController {
    console: Option<SerialConsole>,
}

impl SerialConsoleController {
    pub fn is_open(&self) -> bool {
        self.console.is_some()
    }

    pub fn view(&self) -> Option<SerialView> {
        let c = self.console.as_ref()?;
        let (text, connected, error, seq) = c.snapshot();
        let (status, error) = match error {
            Some(e) => (e, true),
            None if connected => (t!("window-serial-connected", addr = c.addr().to_string()), false),
            None => (t!("window-serial-disconnected"), false),
        };
        Some(SerialView { text, connected, status, error, seq })
    }

    /// The serial light: lit while the console window is connected.
    pub fn light(&self, live: bool) -> DeviceLight {
        let connected = self.view().is_some_and(|v| v.connected);
        DeviceLight {
            id: super::ids::status::SERIAL.into(),
            icon: Icon::SerialConsole,
            state: if !live {
                LightState::Off
            } else if connected {
                LightState::Active
            } else {
                LightState::Idle
            },
            tooltip: if connected { t!("status-serial-connected-tip") } else { t!("status-serial-tip") },
            badge: None,
            target: LightTarget::Serial,
        }
    }

    pub(crate) fn apply(&mut self, action: SerialAction, cfg: &iris::config::MachineConfig, notify: Arc<dyn Notify>) {
        match action {
            SerialAction::Open => self.console = console_port(cfg).map(|port| SerialConsole::connect(port, notify)),
            SerialAction::Close => self.console = None,
            SerialAction::SendLine(line) => {
                if let Some(c) = &self.console {
                    let mut bytes = line.into_bytes();
                    bytes.push(b'\n');
                    c.send(&bytes);
                }
            }
            SerialAction::SendBytes(b) => {
                if let Some(c) = &self.console {
                    c.send(&b);
                }
            }
            SerialAction::Clear => {
                if let Some(c) = &self.console {
                    c.clear();
                }
            }
        }
    }

    /// A new machine started: an open console reconnects to its listener.
    pub(crate) fn reconnect(&mut self, cfg: &iris::config::MachineConfig, notify: Arc<dyn Notify>) {
        if self.console.is_some() {
            self.console = console_port(cfg).map(|port| SerialConsole::connect(port, notify));
        }
    }
}
