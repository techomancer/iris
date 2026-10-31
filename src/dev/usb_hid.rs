//! USB HID boot-protocol keyboard and mouse: low-speed functions that plug
//! into an OHCI root port and are fed from the host's keyboard and mouse.
//!
//! Written from the public specifications: USB 1.1 (chapter 9 requests and
//! descriptors), Device Class Definition for HID 1.11 (class requests and
//! Appendix B, the boot interface descriptors and report formats) and the HID
//! Usage Tables (keyboard/keypad page). Each device has one configuration,
//! one boot interface and one interrupt-IN endpoint.
//!
//! The control pipe follows USB 1.1 section 8.5.2: a SETUP packet, an optional
//! data stage and a status stage. SET_ADDRESS takes effect once its status
//! stage completes. Interrupt reports queue one per host-side change so a key
//! tapped and released between two polls is still seen twice. Input that
//! arrives before the guest has selected a configuration is dropped, as an
//! unconfigured device would not report it.
//!
//! The data toggle on the wire is not tracked: the controller model keeps
//! each endpoint's toggle and a lost transfer cannot be replayed here.
//! Remote wakeup, the wheel, and the host keyboard layout are not modelled.

use crate::dev::ohci::{Pid, UsbFunction, UsbOutcome, UsbPacket};
use crate::dev::ps2::HostInputSink;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::Arc;
use winit::keyboard::KeyCode;

/// A vendor and product pair from the pid.codes open test allocation, since
/// the devices are not products of any vendor.
const VENDOR_ID: u16 = 0x1209;
const PRODUCT_ID: u16 = 0x0001;

// Standard request codes (USB 1.1 table 9-4) and descriptor types (table 9-5).
const GET_STATUS: u8 = 0;
const CLEAR_FEATURE: u8 = 1;
const SET_FEATURE: u8 = 3;
const SET_ADDRESS: u8 = 5;
const GET_DESCRIPTOR: u8 = 6;
const GET_CONFIGURATION: u8 = 8;
const SET_CONFIGURATION: u8 = 9;
const GET_INTERFACE: u8 = 10;
const SET_INTERFACE: u8 = 11;
const DESC_DEVICE: u8 = 1;
const DESC_CONFIGURATION: u8 = 2;
const DESC_STRING: u8 = 3;
const DESC_HID: u8 = 0x21;
const DESC_REPORT: u8 = 0x22;
// HID class requests (HID 1.11 section 7.2).
const GET_REPORT: u8 = 1;
const GET_IDLE: u8 = 2;
const GET_PROTOCOL: u8 = 3;
const SET_REPORT: u8 = 9;
const SET_IDLE: u8 = 10;
const SET_PROTOCOL: u8 = 11;

/// HID 1.11 appendix B.1: the boot keyboard's report descriptor. Input: a
/// modifier bit per key and six key usages; output: five LED bits.
pub const KEYBOARD_REPORT_DESCRIPTOR: [u8; 63] = [
    0x05, 0x01, 0x09, 0x06, 0xa1, 0x01, 0x05, 0x07, 0x19, 0xe0, 0x29, 0xe7, 0x15, 0x00, 0x25, 0x01,
    0x75, 0x01, 0x95, 0x08, 0x81, 0x02, 0x95, 0x01, 0x75, 0x08, 0x81, 0x01, 0x95, 0x05, 0x75, 0x01,
    0x05, 0x08, 0x19, 0x01, 0x29, 0x05, 0x91, 0x02, 0x95, 0x01, 0x75, 0x03, 0x91, 0x01, 0x95, 0x06,
    0x75, 0x08, 0x15, 0x00, 0x25, 0x65, 0x05, 0x07, 0x19, 0x00, 0x29, 0x65, 0x81, 0x00, 0xc0,
];
/// HID 1.11 appendix B.2: the boot mouse's report descriptor. Three buttons
/// and relative X and Y.
pub const MOUSE_REPORT_DESCRIPTOR: [u8; 50] = [
    0x05, 0x01, 0x09, 0x02, 0xa1, 0x01, 0x09, 0x01, 0xa1, 0x00, 0x05, 0x09, 0x19, 0x01, 0x29, 0x03,
    0x15, 0x00, 0x25, 0x01, 0x95, 0x03, 0x75, 0x01, 0x81, 0x02, 0x95, 0x01, 0x75, 0x05, 0x81, 0x01,
    0x05, 0x01, 0x09, 0x30, 0x09, 0x31, 0x15, 0x81, 0x25, 0x7f, 0x75, 0x08, 0x95, 0x02, 0x81, 0x06,
    0xc0, 0xc0,
];

/// Reports held for the guest before the oldest is dropped.
const QUEUE_LIMIT: usize = 64;

/// What differs between the two devices: the report they send and how host
/// input becomes one.
pub trait Boot: Send {
    const PRODUCT: &'static str;
    /// bInterfaceProtocol: 1 keyboard, 2 mouse.
    const PROTOCOL: u8;
    const REPORT_DESCRIPTOR: &'static [u8];
    const MAX_PACKET: u8;
    /// The idle rate at reset, in 4 ms units (HID 1.11 section 7.2.4).
    const DEFAULT_IDLE: u8;
    fn reset(&mut self);
    /// The report GET_REPORT returns now.
    fn current(&self) -> Vec<u8>;
    /// The next queued report, if any.
    fn next_report(&mut self) -> Option<Vec<u8>>;
    /// An output report; false when the device has none.
    fn set_output(&mut self, _data: &[u8]) -> bool {
        false
    }
    /// Host input waiting for the guest, in reports.
    fn backlog(&self) -> usize;
}

enum Stage {
    Idle,
    /// Data stage toward the host, then an OUT status.
    In { data: Vec<u8>, sent: usize },
    /// Data stage from the host for a SET_REPORT of `length` bytes.
    Out { length: usize, data: Vec<u8> },
    /// Status stage of a request with no data toward the host: the next IN
    /// answers with an empty packet and the request takes effect.
    StatusIn(Effect),
}

#[derive(Clone, Copy)]
enum Effect {
    None,
    Address(u8),
    Configuration(u8),
    Idle(u8),
    Protocol(u8),
    Output,
}

struct Inner<B> {
    dev: B,
    address: u8,
    configured: bool,
    /// 0 boot, 1 report; a device starts in report protocol.
    protocol: u8,
    idle: u8,
    remote_wakeup: bool,
    stage: Stage,
    last_report_ms: u64,
}

pub struct UsbHid<B> {
    inner: Mutex<Inner<B>>,
}

impl<B: Boot> UsbHid<B> {
    fn with(dev: B) -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(Inner {
                dev,
                address: 0,
                configured: false,
                protocol: 1,
                idle: B::DEFAULT_IDLE,
                remote_wakeup: false,
                stage: Stage::Idle,
                last_report_ms: 0,
            }),
        })
    }
}

fn string_descriptor(s: &str) -> Vec<u8> {
    let units: Vec<u8> = s.encode_utf16().flat_map(u16::to_le_bytes).collect();
    let mut d = vec![2 + units.len() as u8, DESC_STRING];
    d.extend(units);
    d
}

impl<B: Boot> Inner<B> {
    fn device_descriptor() -> Vec<u8> {
        let mut d = vec![18, DESC_DEVICE, 0x10, 0x01, 0, 0, 0, 8];
        d.extend(VENDOR_ID.to_le_bytes());
        d.extend(PRODUCT_ID.to_le_bytes());
        // bcdDevice, manufacturer and product strings, no serial, one configuration.
        d.extend([0x00, 0x01, 1, 2, 0, 1]);
        d
    }

    fn hid_descriptor() -> Vec<u8> {
        let n = B::REPORT_DESCRIPTOR.len() as u16;
        let mut d = vec![9, DESC_HID, 0x11, 0x01, 0, 1, DESC_REPORT];
        d.extend(n.to_le_bytes());
        d
    }

    /// The configuration descriptor followed by its interface, HID and
    /// endpoint descriptors, as GET_DESCRIPTOR(configuration) returns them.
    fn configuration_descriptor() -> Vec<u8> {
        let interface = [9, 4, 0, 0, 1, 3, 1, B::PROTOCOL, 0];
        let hid = Self::hid_descriptor();
        // Interrupt IN endpoint 1, polled every 10 ms.
        let endpoint = [7, 5, 0x81, 3, B::MAX_PACKET, 0, 10];
        let total = (9 + interface.len() + hid.len() + endpoint.len()) as u16;
        let mut d = vec![9, DESC_CONFIGURATION];
        d.extend(total.to_le_bytes());
        // One interface, configuration 1, no string, bus powered with remote
        // wakeup, 100 mA.
        d.extend([1, 1, 0, 0xa0, 50]);
        d.extend(interface);
        d.extend(hid);
        d.extend(endpoint);
        d
    }

    fn string(index: u8) -> Option<Vec<u8>> {
        match index {
            0 => Some(vec![4, DESC_STRING, 0x09, 0x04]),
            1 => Some(string_descriptor("iris")),
            2 => Some(string_descriptor(B::PRODUCT)),
            _ => None,
        }
    }

    fn reset(&mut self) {
        self.address = 0;
        self.configured = false;
        self.protocol = 1;
        self.idle = B::DEFAULT_IDLE;
        self.remote_wakeup = false;
        self.stage = Stage::Idle;
        self.dev.reset();
    }

    /// Decode a SETUP packet (USB 1.1 table 9-2). None is a request this
    /// device does not have, answered with a STALL.
    fn setup(&mut self, p: &[u8]) -> Option<()> {
        let [rt, req, vl, vh, il, _ih, ll, lh] = <[u8; 8]>::try_from(p).ok()?;
        let value = u16::from_le_bytes([vl, vh]);
        let length = u16::from_le_bytes([ll, lh]) as usize;
        let reply = |data: Vec<u8>| Stage::In { sent: 0, data: data[..data.len().min(length)].to_vec() };
        let (kind, recipient) = ((rt >> 5) & 3, rt & 0x1f);
        let to_host = rt & 0x80 != 0;
        self.stage = match (kind, recipient, to_host, req) {
            // Standard requests (USB 1.1 section 9.4).
            (0, 0, true, GET_STATUS) => reply(vec![u8::from(self.remote_wakeup) << 1, 0]),
            (0, 1 | 2, true, GET_STATUS) if value == 0 => reply(vec![0, 0]),
            (0, 0, false, CLEAR_FEATURE) if value == 1 => {
                self.remote_wakeup = false;
                Stage::StatusIn(Effect::None)
            }
            (0, 0, false, SET_FEATURE) if value == 1 => {
                self.remote_wakeup = true;
                Stage::StatusIn(Effect::None)
            }
            // Clearing a halt on the interrupt endpoint is accepted; nothing halts it.
            (0, 2, false, CLEAR_FEATURE) if value == 0 && il & 0x8f == 0x81 => Stage::StatusIn(Effect::None),
            (0, 0, false, SET_ADDRESS) if value <= 127 => Stage::StatusIn(Effect::Address(value as u8)),
            (0, 0, true, GET_DESCRIPTOR) => match (vh, vl) {
                (DESC_DEVICE, 0) => reply(Self::device_descriptor()),
                (DESC_CONFIGURATION, 0) => reply(Self::configuration_descriptor()),
                (DESC_STRING, n) => reply(Self::string(n)?),
                _ => return None,
            },
            (0, 0, true, GET_CONFIGURATION) => reply(vec![u8::from(self.configured)]),
            (0, 0, false, SET_CONFIGURATION) if value <= 1 => Stage::StatusIn(Effect::Configuration(value as u8)),
            (0, 1, true, GET_INTERFACE) if il == 0 => reply(vec![0]),
            (0, 1, false, SET_INTERFACE) if il == 0 && value == 0 => Stage::StatusIn(Effect::None),
            // The HID descriptor and report descriptor belong to the interface.
            (0, 1, true, GET_DESCRIPTOR) if il == 0 => match (vh, vl) {
                (DESC_HID, 0) => reply(Self::hid_descriptor()),
                (DESC_REPORT, 0) => reply(B::REPORT_DESCRIPTOR.to_vec()),
                _ => return None,
            },
            // HID class requests (HID 1.11 section 7.2), interface 0.
            (1, 1, true, GET_REPORT) if il == 0 && vh == 1 && vl == 0 => reply(self.dev.current()),
            (1, 1, false, SET_REPORT) if il == 0 && vh == 2 && vl == 0 && length > 0 => {
                Stage::Out { length, data: Vec::new() }
            }
            (1, 1, true, GET_IDLE) if il == 0 => reply(vec![self.idle]),
            (1, 1, false, SET_IDLE) if il == 0 => Stage::StatusIn(Effect::Idle(vh)),
            (1, 1, true, GET_PROTOCOL) if il == 0 => reply(vec![self.protocol]),
            (1, 1, false, SET_PROTOCOL) if il == 0 && value <= 1 => Stage::StatusIn(Effect::Protocol(value as u8)),
            _ => return None,
        };
        Some(())
    }

    fn apply(&mut self, effect: Effect, data: &[u8], now_ms: u64) -> bool {
        match effect {
            Effect::None => {}
            Effect::Address(a) => self.address = a,
            Effect::Configuration(c) => {
                self.configured = c != 0;
                self.dev.reset();
                self.last_report_ms = now_ms;
            }
            Effect::Idle(rate) => self.idle = rate,
            Effect::Protocol(p) => self.protocol = p,
            Effect::Output => return self.dev.set_output(data),
        }
        true
    }

    fn control(&mut self, p: UsbPacket) -> UsbOutcome {
        match p.pid {
            Pid::Setup => match self.setup(&p.data) {
                Some(()) => UsbOutcome::Ack(Vec::new()),
                None => {
                    self.stage = Stage::Idle;
                    UsbOutcome::Stall
                }
            },
            Pid::In => match &mut self.stage {
                Stage::In { data, sent } => {
                    let n = p.capacity.min(data.len() - *sent);
                    let chunk = data[*sent..*sent + n].to_vec();
                    *sent += n;
                    UsbOutcome::Ack(chunk)
                }
                Stage::StatusIn(effect) => {
                    let effect = *effect;
                    self.stage = Stage::Idle;
                    if self.apply(effect, &[], p.time_ms) { UsbOutcome::Ack(Vec::new()) } else { UsbOutcome::Stall }
                }
                _ => UsbOutcome::Stall,
            },
            Pid::Out => match &mut self.stage {
                // The status stage of a request that sent data to the host.
                Stage::In { .. } if p.data.is_empty() => {
                    self.stage = Stage::Idle;
                    UsbOutcome::Ack(Vec::new())
                }
                Stage::Out { length, data } => {
                    data.extend(&p.data);
                    if data.len() >= *length {
                        let data = std::mem::take(data);
                        let ok = self.apply(Effect::Output, &data, p.time_ms);
                        self.stage = if ok { Stage::StatusIn(Effect::None) } else { Stage::Idle };
                        if !ok {
                            return UsbOutcome::Stall;
                        }
                    }
                    UsbOutcome::Ack(Vec::new())
                }
                _ => UsbOutcome::Stall,
            },
        }
    }

    /// Interrupt IN endpoint 1 (USB 1.1 section 5.7): a report when there is
    /// one, otherwise NAK. A nonzero idle rate repeats the last state once the
    /// period passes without a change.
    fn interrupt(&mut self, p: UsbPacket) -> UsbOutcome {
        if !self.configured {
            return UsbOutcome::Stall;
        }
        if let Some(r) = self.dev.next_report() {
            self.last_report_ms = p.time_ms;
            return UsbOutcome::Ack(r);
        }
        if self.idle != 0 && p.time_ms.saturating_sub(self.last_report_ms) >= u64::from(self.idle) * 4 {
            self.last_report_ms = p.time_ms;
            return UsbOutcome::Ack(self.dev.current());
        }
        UsbOutcome::Nak
    }
}

impl<B: Boot> UsbFunction for UsbHid<B> {
    fn reset(&self) {
        self.inner.lock().reset();
    }
    fn address(&self) -> u8 {
        self.inner.lock().address
    }
    fn low_speed(&self) -> bool {
        true
    }
    fn transact(&self, packet: UsbPacket) -> UsbOutcome {
        let mut s = self.inner.lock();
        match packet.endpoint {
            0 => s.control(packet),
            1 if packet.pid == Pid::In => s.interrupt(packet),
            _ => UsbOutcome::Stall,
        }
    }
}

// ---------------------------------------------------------------------------
// Keyboard
// ---------------------------------------------------------------------------

/// The HID usage (keyboard/keypad page 0x07) of a host key, or None for a
/// key the page has no entry for.
pub fn usage(key: KeyCode) -> Option<u8> {
    use KeyCode::*;
    Some(match key {
        KeyA => 0x04, KeyB => 0x05, KeyC => 0x06, KeyD => 0x07, KeyE => 0x08, KeyF => 0x09,
        KeyG => 0x0a, KeyH => 0x0b, KeyI => 0x0c, KeyJ => 0x0d, KeyK => 0x0e, KeyL => 0x0f,
        KeyM => 0x10, KeyN => 0x11, KeyO => 0x12, KeyP => 0x13, KeyQ => 0x14, KeyR => 0x15,
        KeyS => 0x16, KeyT => 0x17, KeyU => 0x18, KeyV => 0x19, KeyW => 0x1a, KeyX => 0x1b,
        KeyY => 0x1c, KeyZ => 0x1d,
        Digit1 => 0x1e, Digit2 => 0x1f, Digit3 => 0x20, Digit4 => 0x21, Digit5 => 0x22,
        Digit6 => 0x23, Digit7 => 0x24, Digit8 => 0x25, Digit9 => 0x26, Digit0 => 0x27,
        Enter => 0x28, Escape => 0x29, Backspace => 0x2a, Tab => 0x2b, Space => 0x2c,
        Minus => 0x2d, Equal => 0x2e, BracketLeft => 0x2f, BracketRight => 0x30,
        Backslash => 0x31, Semicolon => 0x33, Quote => 0x34, Backquote => 0x35,
        Comma => 0x36, Period => 0x37, Slash => 0x38, CapsLock => 0x39,
        F1 => 0x3a, F2 => 0x3b, F3 => 0x3c, F4 => 0x3d, F5 => 0x3e, F6 => 0x3f,
        F7 => 0x40, F8 => 0x41, F9 => 0x42, F10 => 0x43, F11 => 0x44, F12 => 0x45,
        PrintScreen => 0x46, ScrollLock => 0x47, Pause => 0x48, Insert => 0x49,
        Home => 0x4a, PageUp => 0x4b, Delete => 0x4c, End => 0x4d, PageDown => 0x4e,
        ArrowRight => 0x4f, ArrowLeft => 0x50, ArrowDown => 0x51, ArrowUp => 0x52,
        NumLock => 0x53, NumpadDivide => 0x54, NumpadMultiply => 0x55,
        NumpadSubtract => 0x56, NumpadAdd => 0x57, NumpadEnter => 0x58,
        Numpad1 => 0x59, Numpad2 => 0x5a, Numpad3 => 0x5b, Numpad4 => 0x5c,
        Numpad5 => 0x5d, Numpad6 => 0x5e, Numpad7 => 0x5f, Numpad8 => 0x60,
        Numpad9 => 0x61, Numpad0 => 0x62, NumpadDecimal => 0x63,
        IntlBackslash => 0x64, ContextMenu => 0x65,
        _ => return None,
    })
}

/// The bit of a modifier key in the report's first byte (usages 0xe0 to 0xe7).
fn modifier(key: KeyCode) -> Option<u8> {
    use KeyCode::*;
    Some(match key {
        ControlLeft => 0x01, ShiftLeft => 0x02, AltLeft => 0x04, SuperLeft => 0x08,
        ControlRight => 0x10, ShiftRight => 0x20, AltRight => 0x40, SuperRight => 0x80,
        _ => return None,
    })
}

#[derive(Default)]
pub struct Keyboard {
    modifiers: u8,
    /// Pressed keys in the order they went down.
    keys: Vec<u8>,
    queue: VecDeque<Vec<u8>>,
    leds: u8,
}

impl Keyboard {
    /// The 8-byte boot report (HID 1.11 B.1): modifiers, reserved, six keys.
    /// More than six keys at once is the phantom state, ErrorRollOver in every slot.
    fn report(&self) -> Vec<u8> {
        let mut r = vec![self.modifiers, 0, 0, 0, 0, 0, 0, 0];
        if self.keys.len() > 6 {
            r[2..].fill(0x01);
        } else {
            r[2..2 + self.keys.len()].copy_from_slice(&self.keys);
        }
        r
    }

    fn push(&mut self) {
        if self.queue.len() >= QUEUE_LIMIT {
            self.queue.pop_front();
        }
        let report = self.report();
        self.queue.push_back(report);
    }
}

impl Boot for Keyboard {
    const PRODUCT: &'static str = "USB Keyboard";
    const PROTOCOL: u8 = 1;
    const REPORT_DESCRIPTOR: &'static [u8] = &KEYBOARD_REPORT_DESCRIPTOR;
    const MAX_PACKET: u8 = 8;
    // 500 ms, the recommended keyboard default.
    const DEFAULT_IDLE: u8 = 125;
    fn reset(&mut self) {
        *self = Keyboard::default();
    }
    fn current(&self) -> Vec<u8> {
        self.report()
    }
    fn next_report(&mut self) -> Option<Vec<u8>> {
        self.queue.pop_front()
    }
    /// The LED output report (HID 1.11 B.1): num lock, caps lock, scroll
    /// lock, compose, kana in bits 0 to 4.
    fn set_output(&mut self, data: &[u8]) -> bool {
        self.leds = data.first().copied().unwrap_or(0) & 0x1f;
        true
    }
    fn backlog(&self) -> usize {
        self.queue.len()
    }
}

pub type UsbKeyboard = UsbHid<Keyboard>;

impl UsbKeyboard {
    pub fn new() -> Arc<Self> {
        Self::with(Keyboard::default())
    }

    /// The LEDs the guest last asked for.
    pub fn leds(&self) -> u8 {
        self.inner.lock().dev.leds
    }
}

impl HostInputSink for UsbKeyboard {
    fn key(&self, key: KeyCode, pressed: bool) {
        let mut s = self.inner.lock();
        if !s.configured {
            return;
        }
        let k = &mut s.dev;
        if let Some(bit) = modifier(key) {
            let before = k.modifiers;
            if pressed { k.modifiers |= bit } else { k.modifiers &= !bit }
            if k.modifiers != before {
                k.push();
            }
        } else if let Some(u) = usage(key) {
            let at = k.keys.iter().position(|&x| x == u);
            match (pressed, at) {
                (true, None) => k.keys.push(u),
                (false, Some(i)) => {
                    k.keys.remove(i);
                }
                _ => return,
            }
            k.push();
        }
    }
    fn mouse(&self, _: u8, _: i32, _: i32, _: i32) {}
    fn mouse_enabled(&self) -> bool {
        false
    }
    fn mouse_pending(&self) -> usize {
        0
    }
}

// ---------------------------------------------------------------------------
// Mouse
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct Mouse {
    buttons: u8,
    /// Movement not yet in a report.
    dx: i32,
    dy: i32,
    queue: VecDeque<Vec<u8>>,
}

impl Mouse {
    /// Move accumulated motion into reports, at most a signed byte on each
    /// axis apiece (HID 1.11 B.2: buttons, X, Y). `force` sends one even
    /// with no motion, for a button change.
    fn flush(&mut self, mut force: bool) {
        while (force || self.dx != 0 || self.dy != 0) && self.queue.len() < QUEUE_LIMIT {
            let (x, y) = (self.dx.clamp(-127, 127), self.dy.clamp(-127, 127));
            self.dx -= x;
            self.dy -= y;
            self.queue.push_back(vec![self.buttons, x as i8 as u8, y as i8 as u8]);
            force = false;
        }
    }
}

impl Boot for Mouse {
    const PRODUCT: &'static str = "USB Mouse";
    const PROTOCOL: u8 = 2;
    const REPORT_DESCRIPTOR: &'static [u8] = &MOUSE_REPORT_DESCRIPTOR;
    const MAX_PACKET: u8 = 4;
    // A mouse reports only on change.
    const DEFAULT_IDLE: u8 = 0;
    fn reset(&mut self) {
        *self = Mouse::default();
    }
    fn current(&self) -> Vec<u8> {
        vec![self.buttons, 0, 0]
    }
    fn next_report(&mut self) -> Option<Vec<u8>> {
        if self.queue.is_empty() {
            self.flush(false);
        }
        self.queue.pop_front()
    }
    fn backlog(&self) -> usize {
        self.queue.len() + usize::from(self.dx != 0 || self.dy != 0)
    }
}

pub type UsbMouse = UsbHid<Mouse>;

impl UsbMouse {
    pub fn new() -> Arc<Self> {
        Self::with(Mouse::default())
    }
}

impl HostInputSink for UsbMouse {
    fn key(&self, _: KeyCode, _: bool) {}
    fn mouse(&self, buttons: u8, dx: i32, dy: i32, _dz: i32) {
        let mut s = self.inner.lock();
        if !s.configured {
            return;
        }
        let m = &mut s.dev;
        m.dx = m.dx.saturating_add(dx);
        m.dy = m.dy.saturating_add(dy);
        let buttons = buttons & 7;
        // Motion up to now travels with the old button state.
        if buttons != m.buttons {
            m.flush(false);
            m.buttons = buttons;
            m.flush(true);
        } else {
            m.flush(false);
        }
    }
    fn mouse_enabled(&self) -> bool {
        self.inner.lock().configured
    }
    fn mouse_pending(&self) -> usize {
        self.inner.lock().dev.backlog()
    }
}

#[cfg(test)]
mod tests;
