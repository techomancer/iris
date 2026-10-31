//! Fuel's L1 system controller and its processor-side UART.
//!
//! The internal L1 connection is separate from IOC3's external serial ports.
//! The Ns16550 owns the UART registers, FIFO, timeout and interrupt state;
//! this device keeps the controller protocol and its eight-byte register stride.
//! Controller identities and power transitions are functional approximations.
//! Remote routing and controller diagnostics are absent.
//!
//! Ported from hw/misc/sgi-l1.c in jamesbraid/qemu-sgi-origami.

use crate::dev::ns16550::{com_reg, Ns16550};
use crate::dev::serial_host::{null_serial_backend, SerialBackend};
use crate::devlog::LogModule;
use crate::irq::IrqLine;
use crate::traits::{BusDevice, BusRead64, Device, Resettable, BUS_ERR, BUS_OK};
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::io;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};

mod model;
use model::State;
pub use model::{L1Config, L1Event};

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

mod uart_offset {
    use super::com_reg;
    pub const DATA: u64 = 0x00;
    pub const IER: u64 = 0x08;
    pub const IIR_FCR: u64 = 0x10;
    pub const LCR: u64 = 0x18;
    pub const MCR: u64 = 0x20;
    pub const LSR: u64 = 0x28;
    pub const MSR: u64 = 0x30;
    pub const SCR: u64 = 0x38;

    pub fn register(offset: u64) -> u32 {
        match offset {
            DATA => com_reg::THR,
            IER => com_reg::IER,
            IIR_FCR => com_reg::IIR_FCR,
            LCR => com_reg::LCR,
            MCR => com_reg::MCR,
            LSR => com_reg::LSR,
            MSR => com_reg::MSR,
            SCR => com_reg::SCR,
            _ => unreachable!("validated L1 UART offset"),
        }
    }
}

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

/// The internal L1 connection is independent of IOC3's external serial ports.
/// Register offsets are relative to the 0x40-byte UART window. Only aligned
/// 64-bit accesses are decoded; register values occupy the low eight bits.
pub struct L1 {
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
    clock: Arc<AtomicU64>,
    worker: Mutex<Option<std::thread::JoinHandle<()>>>,
    wake: Arc<(Mutex<()>, parking_lot::Condvar)>,
}
impl L1 {
    pub fn new(config: L1Config) -> Result<Self, String> {
        if config.rack > 0x3ff || config.bay > 0x3f || config.node_position > 3 {
            return Err("L1 rack, bay or node position out of range".into());
        }
        if config.mac == [0; 6] || config.mac[0] & 1 != 0 {
            return Err("L1 MAC must be nonzero unicast".into());
        }
        if config
            .brick_serial
            .as_ref()
            .is_some_and(|s| s.len() > 37 || s.as_bytes().contains(&0))
        {
            return Err("L1 brick serial must fit a 37-byte text field".into());
        }

        let protocol_rx = Arc::new(Mutex::new(VecDeque::new()));
        let state = Arc::new(Mutex::new(State::new(config, protocol_rx.clone())));
        let guest_tx = Arc::new(Mutex::new(VecDeque::new()));
        let protocol_backend = Arc::new(ProtocolBackend {
            guest_tx: guest_tx.clone(),
            protocol_rx,
        });
        let mut uart = Ns16550::new(UART_BAUDBASE);
        uart.set_backend(protocol_backend);

        Ok(Self {
            state,
            dispatch: Arc::new(Mutex::new(Dispatch::default())),
            uart: Arc::new(Mutex::new(uart)),
            guest_tx,
            backend: Arc::new(Mutex::new(null_serial_backend())),
            irq: Arc::new(Mutex::new(IrqLine::unconnected())),
            handler: Arc::new(Mutex::new(None)),
            running: Arc::new(AtomicBool::new(false)),
            clock: Arc::new(AtomicU64::new(0)),
            worker: Mutex::new(None),
            wake: Arc::new((Mutex::new(()), parking_lot::Condvar::new())),
        })
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
    pub fn display_line(&self) -> String {
        self.state.lock().display.clone()
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

    /// Install a raw area before execution. It replaces only the named area;
    /// station identity remains the configured MAC, as in the Fuel model.
    #[cfg(test)]
    pub fn set_record(&self, component: usize, area: u32, bytes: Vec<u8>) -> Result<(), String> {
        let index = match area {
            1 => 0,
            2 => 1,
            4 => 2,
            _ => return Err("unsupported L1 area".into()),
        };
        if component >= 16 || bytes.len() > 65535 {
            return Err("L1 record out of range".into());
        }
        self.state.lock().records[component][index] = bytes;
        Ok(())
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

    pub fn poll_serial(&self) {
        let backend = self.backend.lock().clone();
        for _ in 0..256 {
            if self.state.lock().queue.lock().len() > 4088 {
                break;
            }
            match backend.recv_byte() {
                Ok(byte) => self.receive_console(&[byte]),
                Err(error) => {
                    if error.kind() != io::ErrorKind::WouldBlock {
                        log::debug!("L1 serial input: {error}");
                    }
                    break;
                }
            }
        }
    }

    /// Let the native UART pull reply bytes from the model's shared bounded queue.
    fn service_protocol_uart(&self) {
        let mut uart = self.uart.lock();
        uart.pump();
        uart.check_timeout(std::time::Instant::now());
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
            let level = self.uart.lock().irq();

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
        let value = self.uart.lock().read(uart_offset::register(addr));
        self.flush();
        BusRead64::ok(value as u64)
    }

    fn write64(&self, addr: u64, value: u64) -> u32 {
        if addr >= UART_WINDOW_SIZE || addr & 7 != 0 {
            return BUS_ERR;
        }
        let register = uart_offset::register(addr);
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
        self.clock.store(0, Ordering::Relaxed);
        self.flush();
    }
}
impl Device for L1 {
    fn start(&self) {
        let mut worker = self.worker.lock();
        if worker.is_some() {
            return;
        }
        self.running.store(true, Ordering::Release);
        let device = Self {
            state: self.state.clone(),
            dispatch: self.dispatch.clone(),
            uart: self.uart.clone(),
            guest_tx: self.guest_tx.clone(),
            backend: self.backend.clone(),
            irq: self.irq.clone(),
            handler: self.handler.clone(),
            running: self.running.clone(),
            clock: self.clock.clone(),
            worker: Mutex::new(None),
            wake: self.wake.clone(),
        };
        *worker = Some(
            std::thread::Builder::new()
                .name("l1-serial".into())
                .spawn(move || {
                    while device.is_running() {
                        device.poll_serial();
                        let mut guard = device.wake.0.lock();
                        if device.is_running() {
                            device
                                .wake
                                .1
                                .wait_for(&mut guard, std::time::Duration::from_millis(2));
                        }
                    }
                })
                .expect("start L1 serial worker"),
        );
    }

    fn stop(&self) {
        let mut slot = self.worker.lock();
        {
            let _guard = self.wake.0.lock();
            self.running.store(false, Ordering::Release);
            self.wake.1.notify_all();
        }
        let worker = slot.take();
        if let Some(worker) = worker {
            if worker.thread().id() != std::thread::current().id() {
                let _ = worker.join();
            }
        }
    }

    fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    fn get_clock(&self) -> u64 {
        self.clock.load(Ordering::Relaxed)
    }

    fn step(&self, cycles: u64) {
        if self.is_running() {
            self.clock.fetch_add(cycles, Ordering::Relaxed);
        }
    }
}
impl Drop for L1 {
    fn drop(&mut self) {
        self.stop();
    }
}
#[cfg(test)]
mod tests;
