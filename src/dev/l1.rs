//! Fuel's L1 system controller and its processor-side UART.
//!
//! The Ns16550 owns the UART state; this device keeps the controller protocol
//! and its eight-byte register stride. It answers the requests the stock PROM
//! and IRIX send and logs the rest as unknown. Power down (0x2005) and power
//! cycle (0x2006) act at once. Remote routing and diagnostics are absent; a
//! router interpreter answers only the single-fragment `brick` request.
//!
//! Ported from hw/misc/sgi-l1.c in jamesbraid/qemu-sgi-origami.

use crate::dev::ns16550::{com_reg, Ns16550};
use crate::dev::serial_host::{null_serial_backend, SerialBackend};
use crate::devlog::LogModule;
use crate::hptimer::{Timer, TimerManager};
use crate::irq::IrqLine;
use crate::net::wake::{host_io, HostPump};
use crate::traits::{BusDevice, BusRead64, Device, Resettable, BUS_ERR, BUS_OK};
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::io;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

mod model;
use model::State;
pub use model::{parse_revision, AttachedBrick, BoardIdentity, ComputeNode, CpuContext, L1Config, L1Event, DEFAULT_BRICK_SERIAL, DEFAULT_REVISION};

fn fcs(mut crc: u16, byte: u8) -> u16 {
    crc ^= byte as u16;
    for _ in 0..8 {
        crc = (crc >> 1) ^ if crc & 1 != 0 { 0x8408 } else { 0 };
    }
    crc
}

fn encode(payload: &[u8]) -> Vec<u8> {
    let crc = !payload.iter().fold(0xffff, |value, byte| fcs(value, *byte));
    let mut out = vec![0x7e];
    for byte in payload.iter().copied().chain([crc as u8, (crc >> 8) as u8]) {
        if matches!(byte, 0x7e | 0x7d | 0x0d) {
            out.extend_from_slice(&[0x7d, byte ^ 0x20]);
        } else {
            out.push(byte);
        }
    }
    out.push(0x7e);
    out
}

const UART_BAUDBASE: u32 = 115200;
const UART_WINDOW_SIZE: u64 = 0x40;

type EventHandler = Arc<dyn Fn(L1Event) + Send + Sync>;

#[derive(Default)]
struct Dispatch {
    active: bool,
    pending: bool,
    level: Option<bool>,
}

/// The UART backend does not connect guest L1 commands to a host terminal.
/// Guest transmit bytes are handed to the protocol parser after the UART lock
/// is released; receive bytes come from the protocol reply queue.
struct ProtocolBackend {
    guest_tx: Arc<Mutex<VecDeque<u8>>>,
    protocol_rx: Arc<Mutex<VecDeque<u8>>>,
}

impl SerialBackend for ProtocolBackend {
    fn send_byte(&self, byte: u8) {
        self.guest_tx.lock().push_back(byte);
    }

    fn recv_byte(&self) -> io::Result<u8> {
        self.protocol_rx
            .lock()
            .pop_front()
            .ok_or_else(|| io::ErrorKind::WouldBlock.into())
    }
}

/// Register offsets are relative to the 0x40-byte UART window; only aligned
/// 64-bit accesses decode, values in the low eight bits.
pub struct L1 {
    shared: Arc<Shared>,
    pump: Mutex<Option<HostPump>>,
}

impl std::ops::Deref for L1 {
    type Target = Shared;
    fn deref(&self) -> &Shared {
        &self.shared
    }
}

/// What the serial pump and the receive-timeout timer share with the device.
pub struct Shared {
    // UART RX takes UART -> protocol queue. Model replies take state -> protocol
    // queue. No path holds the protocol queue while acquiring either lock.
    state: Arc<Mutex<State>>,
    // Take dispatch before state when draining, release it before the UART or callbacks.
    dispatch: Arc<Mutex<Dispatch>>,
    uart: Arc<Mutex<Ns16550>>,
    guest_tx: Arc<Mutex<VecDeque<u8>>>,
    backend: Arc<Mutex<Arc<dyn SerialBackend>>>,
    irq: Arc<Mutex<IrqLine>>,
    handler: Arc<Mutex<Option<EventHandler>>>,
    running: Arc<AtomicBool>,
    /// The UART's receive timeout, on the machine clock.
    rx_timer: Timer,
}
impl L1 {
    pub fn new(tm: &Arc<TimerManager>, config: L1Config) -> Result<Self, String> {
        if config.rack > 0x3ff || config.bay > 0x3f || config.node_position > 3 {
            return Err("L1 rack, bay or node position out of range".into());
        }
        if config.compute_node.is_some() && config.node_board.is_none() {
            return Err("L1 compute node needs a node board".into());
        }
        if config.port_speed_first > 1 {
            return Err("L1 first port-speed value must be zero or one".into());
        }
        if config.mac == [0; 6] || config.mac[0] & 1 != 0 {
            return Err("L1 MAC must be nonzero unicast".into());
        }
        if config
            .brick_serial
            .as_ref()
            .is_some_and(|s| s.len() > 37 || s.as_bytes().contains(&0))
        {
            return Err("L1 brick serial is longer than its 37-byte field".into());
        }
        if config
            .system_serial
            .as_ref()
            .is_some_and(|s| s.len() > 37 || s.as_bytes().contains(&0))
        {
            return Err("L1 system serial is longer than its 37-byte field".into());
        }

        let protocol_rx = Arc::new(Mutex::new(VecDeque::new()));
        let state = Arc::new(Mutex::new(State::new(config, protocol_rx.clone())));
        let guest_tx = Arc::new(Mutex::new(VecDeque::new()));
        let protocol_backend = Arc::new(ProtocolBackend {
            guest_tx: guest_tx.clone(),
            protocol_rx,
        });
        let mut uart = Ns16550::new(UART_BAUDBASE, tm.clock().clone());
        uart.set_backend(protocol_backend);

        let shared = Arc::new_cyclic(|me| Shared {
            state,
            dispatch: Arc::new(Mutex::new(Dispatch::default())),
            uart: Arc::new(Mutex::new(uart)),
            guest_tx,
            backend: Arc::new(Mutex::new(null_serial_backend())),
            irq: Arc::new(Mutex::new(IrqLine::unconnected())),
            handler: Arc::new(Mutex::new(None)),
            running: Arc::new(AtomicBool::new(false)),
            rx_timer: Timer::new(tm, me, |s: &Shared, _| {
                s.service_protocol_uart();
                s.flush();
            }),
        });
        Ok(Self { shared, pump: Mutex::new(None) })
    }

    /// Install the separate host-side L1 console connection.
    pub fn set_serial_backend(&self, backend: Arc<dyn SerialBackend>) {
        *self.backend.lock() = backend;
    }

    pub fn set_irq(&self, line: IrqLine) {
        *self.irq.lock() = line;
        self.dispatch.lock().level = None;
        self.flush();
    }

    /// Power/reset requests reach the machine only after the reply is queued
    /// and after the device state lock is released.
    pub fn set_event_handler(&self, handler: EventHandler) {
        *self.handler.lock() = Some(handler);
    }

    #[cfg(test)]
    pub fn node_powered(&self) -> bool {
        self.state.lock().powered
    }

    pub fn station_mac(&self) -> [u8; 6] {
        self.state.lock().config.mac
    }

    #[cfg(test)]
    pub fn config(&self) -> L1Config {
        self.state.lock().config.clone()
    }

    #[cfg(test)]
    pub fn record(&self, component: usize, area: u32) -> Option<Vec<u8>> {
        let index = match area {
            1 => 0,
            2 => 1,
            4 => 2,
            _ => return None,
        };
        self.state.lock().records.get(component).and_then(|r| {
            if r[index].is_empty() {
                None
            } else {
                Some(r[index].clone())
            }
        })
    }

    pub fn send_event(&self, code: u32, number: u32, text: &str) {
        self.state.lock().event(code, number, text);
        self.service_protocol_uart();
        self.flush();
    }

    /// Deterministic input path shared with the polled host serial backend.
    pub fn receive_console(&self, bytes: &[u8]) {
        for &byte in bytes {
            self.state.lock().console_input(byte);
            self.service_protocol_uart();
            self.flush();
        }
    }

    /// Whether it took any input.
    pub fn poll_serial(&self) -> bool {
        let backend = self.backend.lock().clone();
        backend.flush();
        let mut moved = false;
        for _ in 0..256 {
            if self.state.lock().queue.lock().len() > 4088 {
                break;
            }
            match backend.recv_byte() {
                Ok(byte) => {
                    self.receive_console(&[byte]);
                    moved = true;
                }
                Err(error) => {
                    if error.kind() != io::ErrorKind::WouldBlock {
                        log::debug!("L1 serial input: {error}");
                    }
                    break;
                }
            }
        }
        moved
    }

    /// Parse guest transmit only after the UART lock is released. The UART
    /// callback only appends to this independent queue.
    fn process_guest_transmit(&self) {
        let bytes: Vec<_> = self.guest_tx.lock().drain(..).collect();
        if bytes.is_empty() {
            return;
        }
        {
            let mut state = self.state.lock();
            for byte in bytes {
                state.parse(byte);
            }
        }
        self.service_protocol_uart();
    }
}
impl Shared {
    /// Let the native UART pull reply bytes from the model's shared bounded queue.
    fn service_protocol_uart(&self) {
        let mut uart = self.uart.lock();
        uart.pump();
        uart.check_timeout();
    }

    fn flush(&self) {
        let mut dispatch = self.dispatch.lock();
        dispatch.pending = true;
        if dispatch.active {
            return;
        }
        dispatch.active = true;
        loop {
            dispatch.pending = false;
            let (output, actions) = {
                let mut state = self.state.lock();
                (
                    std::mem::take(&mut state.output),
                    std::mem::take(&mut state.actions),
                )
            };
            drop(dispatch);
            // Under the lock, so an older deadline cannot land after a newer one.
            let level = {
                let uart = self.uart.lock();
                self.rx_timer.set_due(uart.timeout_due());
                uart.irq()
            };

            // Reentrant reads request another drain, but must not repeat an
            // unchanged IRQ and recursively trigger the same machine callback.
            dispatch = self.dispatch.lock();
            let changed = dispatch.level != Some(level);
            dispatch.level = Some(level);
            drop(dispatch);

            if changed {
                let irq = self.irq.lock().clone();
                irq.set(level);
            }
            let backend = self.backend.lock().clone();
            for byte in output {
                backend.send_byte(byte);
            }
            let handler = self.handler.lock().clone();
            for action in actions {
                if let Some(handler) = &handler {
                    handler(action);
                } else {
                    crate::dlog!(LogModule::L1, "unconnected L1 action {action:?}");
                }
            }

            dispatch = self.dispatch.lock();
            if !dispatch.pending {
                dispatch.active = false;
                break;
            }
        }
    }
}
impl BusDevice for L1 {
    fn read64(&self, addr: u64) -> BusRead64 {
        if addr >= UART_WINDOW_SIZE || addr & 7 != 0 {
            return BusRead64::err();
        }
        let value = self.uart.lock().read((addr >> 3) as u32);
        self.flush();
        BusRead64::ok(value as u64)
    }

    fn write64(&self, addr: u64, value: u64) -> u32 {
        if addr >= UART_WINDOW_SIZE || addr & 7 != 0 {
            return BUS_ERR;
        }
        // The eight registers sit at an eight-byte stride.
        let register = (addr >> 3) as u32;
        let value = value as u8;
        if register == com_reg::IIR_FCR && value & com_reg::FCR_RFR != 0 {
            self.state.lock().queue.lock().clear();
        }
        self.uart.lock().write(register, value);
        self.process_guest_transmit();
        self.flush();
        BUS_OK
    }
}
impl Resettable for L1 {
    fn power_on(&self) {
        self.state.lock().reset();
        self.guest_tx.lock().clear();
        self.uart.lock().reset();
        self.flush();
    }
}
impl Device for L1 {
    fn start(&self) {
        let mut pump = self.pump.lock();
        if pump.is_some() {
            return;
        }
        self.running.store(true, Ordering::Release);
        self.rx_timer.start();
        let device = Self { shared: self.shared.clone(), pump: Mutex::new(None) };
        *pump = Some(host_io().add(move || device.poll_serial()));
    }

    fn stop(&self) {
        self.running.store(false, Ordering::Release);
        self.rx_timer.stop();
        self.pump.lock().take();
    }

    fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    fn step(&self, _cycles: u64) {}
}
impl Drop for L1 {
    fn drop(&mut self) {
        self.stop();
    }
}
#[cfg(test)]
mod tests;
