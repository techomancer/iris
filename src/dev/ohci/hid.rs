//! USB HID boot-protocol keyboard and mouse for the OHCI root ports.
//!
//! Each is a low-speed function with one configuration, one interface (HID
//! class 3, boot subclass 1, protocol 1 for the keyboard and 2 for the mouse)
//! and one interrupt-IN endpoint, as USB 1.1 chapter 9 and HID 1.11 (the boot
//! interface in section 4.3 and appendix B, the class requests in section 7)
//! describe them. The mouse's report descriptor is the boot example of HID
//! 1.11 appendix B.2. The keyboard's is appendix B.1's with the key array's
//! logical and usage maximum at 101 (0x65) as appendix E.6's keyboard has it,
//! where B.1 gives 255; 101 is the highest usage this model reports. The boot
//! and report protocols carry the same bytes: an
//! 8-byte keyboard report (modifiers, reserved, six key usages) and a 3-byte
//! mouse report (buttons, X, Y). Linux's usbkbd and usbmouse drivers are the
//! software expectation for what a host asks.
//!
//! Modelled: GET_DESCRIPTOR (device, configuration, string, HID, report),
//! SET_ADDRESS, SET_CONFIGURATION, the interface and endpoint queries and
//! the HID class requests (GET/SET_REPORT, GET/SET_IDLE, GET/SET_PROTOCOL),
//! keyboard LEDs through an output report, and the idle rate. Not modelled:
//! data toggles (a repeated packet is not detected), remote wakeup, suspend,
//! the mouse wheel (the boot report has no byte for it), physical
//! descriptors and the optional second keyboard LED endpoint.
//!
//! The vendor and product ID are 0x1209, 0x0001, which the pid.codes registry
//! lists as reserved for use in private testing: not unique, and not to be
//! used on any redistributed, sold or manufactured device. They name no real
//! device, and the strings say these are iris's.

use super::{UsbFunction, UsbOutcome, UsbPacket, Pid};
use crate::dev::ps2::InputSink;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::Arc;
use winit::keyboard::KeyCode;

const VENDOR: u16 = 0x1209;
const PRODUCT: u16 = 0x0001;
/// Largest packet either endpoint sends; low speed allows 8.
const PACKET: usize = 8;
/// Mouse reports waiting for the host to poll them. A host that does not poll
/// loses the newest, and the next report restates the whole state. Keyboard
/// reports are never dropped: a lost key-up leaves a key stuck and a lost
/// key-down loses a character.
const QUEUE: usize = 64;

/// HID 1.11 appendix B.1, the boot keyboard, with the key array's maximum at
/// 101 as appendix E.6 has it.
const KEYBOARD_REPORT: [u8; 63] = [
    0x05, 0x01, 0x09, 0x06, 0xa1, 0x01, 0x05, 0x07, 0x19, 0xe0, 0x29, 0xe7, 0x15, 0x00, 0x25, 0x01,
    0x75, 0x01, 0x95, 0x08, 0x81, 0x02, 0x95, 0x01, 0x75, 0x08, 0x81, 0x01, 0x95, 0x05, 0x75, 0x01,
    0x05, 0x08, 0x19, 0x01, 0x29, 0x05, 0x91, 0x02, 0x95, 0x01, 0x75, 0x03, 0x91, 0x01, 0x95, 0x06,
    0x75, 0x08, 0x15, 0x00, 0x25, 0x65, 0x05, 0x07, 0x19, 0x00, 0x29, 0x65, 0x81, 0x00, 0xc0,
];
/// HID 1.11 appendix B.2, the boot mouse.
const MOUSE_REPORT: [u8; 50] = [
    0x05, 0x01, 0x09, 0x02, 0xa1, 0x01, 0x09, 0x01, 0xa1, 0x00, 0x05, 0x09, 0x19, 0x01, 0x29, 0x03,
    0x15, 0x00, 0x25, 0x01, 0x95, 0x03, 0x75, 0x01, 0x81, 0x02, 0x95, 0x01, 0x75, 0x05, 0x81, 0x01,
    0x05, 0x01, 0x09, 0x30, 0x09, 0x31, 0x15, 0x81, 0x25, 0x7f, 0x75, 0x08, 0x95, 0x02, 0x81, 0x06,
    0xc0, 0xc0,
];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Keyboard,
    Mouse,
}
impl Kind {
    fn report_descriptor(self) -> &'static [u8] {
        match self {
            Kind::Keyboard => &KEYBOARD_REPORT,
            Kind::Mouse => &MOUSE_REPORT,
        }
    }
    /// HID 1.11 section 7.2.4 recommends 500 ms for a keyboard and
    /// indefinite (0) for a mouse, in 4 ms units.
    fn default_idle(self) -> u8 {
        match self {
            Kind::Keyboard => 125,
            Kind::Mouse => 0,
        }
    }
    fn report_len(self) -> usize {
        match self {
            Kind::Keyboard => 8,
            Kind::Mouse => 3,
        }
    }
    fn protocol(self) -> u8 {
        match self {
            Kind::Keyboard => 1,
            Kind::Mouse => 2,
        }
    }
    fn product(self) -> &'static str {
        match self {
            Kind::Keyboard => "iris emulated USB keyboard",
            Kind::Mouse => "iris emulated USB mouse",
        }
    }
    fn device_descriptor(self) -> Vec<u8> {
        let mut d = vec![18, 1, 0x10, 0x01, 0, 0, 0, PACKET as u8];
        d.extend_from_slice(&VENDOR.to_le_bytes());
        d.extend_from_slice(&PRODUCT.to_le_bytes());
        d.extend_from_slice(&[0x00, 0x01, 1, 2, 0, 1]);
        d
    }
    fn hid_descriptor(self) -> Vec<u8> {
        let n = self.report_descriptor().len() as u16;
        vec![9, 0x21, 0x11, 0x01, 0, 1, 0x22, n as u8, (n >> 8) as u8]
    }
    fn configuration_descriptor(self) -> Vec<u8> {
        let mut c = vec![9, 2, 34, 0, 1, 1, 0, 0x80, 50];
        c.extend_from_slice(&[9, 4, 0, 0, 1, 3, 1, self.protocol(), 0]);
        c.extend_from_slice(&self.hid_descriptor());
        // Interrupt IN endpoint 1, polled every 10 ms.
        c.extend_from_slice(&[7, 5, 0x81, 3, PACKET as u8, 0, 10]);
        c
    }
}

fn string_descriptor(text: &str) -> Vec<u8> {
    let mut d = vec![0, 3];
    for u in text.encode_utf16() {
        d.extend_from_slice(&u.to_le_bytes());
    }
    d[0] = d.len() as u8;
    d
}

#[derive(Clone, Copy, Debug)]
struct Setup {
    request_type: u8,
    request: u8,
    value: u16,
    index: u16,
    length: u16,
}
impl Setup {
    fn parse(b: &[u8]) -> Option<Self> {
        let b: &[u8; 8] = b.try_into().ok()?;
        Some(Self {
            request_type: b[0],
            request: b[1],
            value: u16::from_le_bytes([b[2], b[3]]),
            index: u16::from_le_bytes([b[4], b[5]]),
            length: u16::from_le_bytes([b[6], b[7]]),
        })
    }
    fn device_to_host(&self) -> bool {
        self.request_type & 0x80 != 0
    }
}

enum Control {
    Idle,
    /// Device-to-host data stage; the status stage is an OUT.
    DataIn { data: Vec<u8>, pos: usize },
    /// Host-to-device data stage.
    DataOut { setup: Setup, got: Vec<u8> },
    /// Host-to-device request whose status stage is an IN.
    StatusIn,
    /// The request was refused; every packet stalls until the next SETUP.
    Stalled,
}
/// What a request asks of the data stage.
enum Plan {
    None,
    In(Vec<u8>),
    Out,
}

struct State {
    address: u8,
    /// SET_ADDRESS takes effect when its status stage completes.
    pending_address: Option<u8>,
    configured: bool,
    /// 0 boot, 1 report. A device powers up in the report protocol.
    protocol: u8,
    /// SET_IDLE's duration, in 4 ms units; 0 sends reports only on change.
    idle: u8,
    halted: bool,
    control: Control,
    queue: VecDeque<Vec<u8>>,
    /// The last report built, for GET_REPORT and idle repeats.
    last: Vec<u8>,
    /// When a report last went to the host; `None` until the first poll
    /// after configuration starts the idle timer.
    last_sent_ms: Option<u64>,
    // Keyboard.
    modifiers: u8,
    keys: Vec<u8>,
    leds: u8,
    // Mouse.
    buttons: u8,
}

pub struct HidFunction {
    kind: Kind,
    state: Mutex<State>,
}

impl HidFunction {
    pub fn new(kind: Kind) -> Arc<Self> {
        Arc::new(Self {
            kind,
            state: Mutex::new(State {
                address: 0,
                pending_address: None,
                configured: false,
                protocol: 1,
                idle: kind.default_idle(),
                halted: false,
                control: Control::Idle,
                queue: VecDeque::new(),
                last: vec![0; kind.report_len()],
                last_sent_ms: None,
                modifiers: 0,
                keys: vec![],
                leds: 0,
                buttons: 0,
            }),
        })
    }
    pub fn keyboard() -> Arc<Self> {
        Self::new(Kind::Keyboard)
    }
    pub fn mouse() -> Arc<Self> {
        Self::new(Kind::Mouse)
    }
    /// The keyboard LEDs the host last set (HID 1.11 appendix B.1: bit 0 Num
    /// Lock, 1 Caps Lock, 2 Scroll Lock, 3 Compose, 4 Kana).
    pub fn leds(&self) -> u8 {
        self.state.lock().leds
    }
    /// Whether the host has selected the configuration, so reports flow.
    pub fn configured(&self) -> bool {
        self.state.lock().configured
    }
    /// Reports waiting for the host to poll.
    pub fn pending(&self) -> usize {
        self.state.lock().queue.len()
    }

    fn enqueue(&self, s: &mut State, report: Vec<u8>) {
        s.last = report.clone();
        if !s.configured {
            return;
        }
        if self.kind == Kind::Mouse && s.queue.len() >= QUEUE {
            crate::dlog!(crate::devlog::LogModule::Usb, "usb hid {:?}: host is not polling, dropped a report", self.kind);
            return;
        }
        s.queue.push_back(report);
    }

    fn keyboard_report(s: &State) -> Vec<u8> {
        let mut r = vec![s.modifiers, 0, 0, 0, 0, 0, 0, 0];
        if s.keys.len() > 6 {
            // HID 1.11 appendix B.1: too many keys is the rollover error.
            r[2..].fill(1);
        } else {
            r[2..2 + s.keys.len()].copy_from_slice(&s.keys);
        }
        r
    }

    fn key(&self, key: KeyCode, pressed: bool) {
        let Some(usage) = usage_of(key) else { return };
        let mut s = self.state.lock();
        let before = (s.modifiers, s.keys.clone());
        match usage {
            Usage::Modifier(bit) => {
                if pressed {
                    s.modifiers |= 1 << bit;
                } else {
                    s.modifiers &= !(1 << bit);
                }
            }
            Usage::Key(u) => {
                if pressed && !s.keys.contains(&u) {
                    s.keys.push(u);
                } else if !pressed {
                    s.keys.retain(|&k| k != u);
                }
            }
        }
        if (s.modifiers, &s.keys) != (before.0, &before.1) {
            let report = Self::keyboard_report(&s);
            self.enqueue(&mut s, report);
        }
    }

    fn pointer(&self, buttons: u8, dx: i32, dy: i32) {
        let mut s = self.state.lock();
        let buttons = buttons & 7;
        let changed = buttons != s.buttons;
        s.buttons = buttons;
        let (mut dx, mut dy) = (dx, dy);
        if !changed && dx == 0 && dy == 0 {
            return;
        }
        // An axis is a signed byte; a bigger move leaves as several reports.
        // -128 is left out so a report negates cleanly.
        loop {
            let (sx, sy) = (dx.clamp(-127, 127), dy.clamp(-127, 127));
            let report = vec![buttons, sx as i8 as u8, sy as i8 as u8];
            self.enqueue_pointer(&mut s, report);
            (dx, dy) = (dx - sx, dy - sy);
            if dx == 0 && dy == 0 {
                break;
            }
        }
    }

    /// Queue a mouse report, folding it into the newest one when the host is
    /// behind, the buttons match and the sum still fits an axis byte.
    fn enqueue_pointer(&self, s: &mut State, report: Vec<u8>) {
        if s.configured && s.queue.len() >= 8 {
            if let Some(tail) = s.queue.back_mut() {
                let x = i32::from(tail[1] as i8) + i32::from(report[1] as i8);
                let y = i32::from(tail[2] as i8) + i32::from(report[2] as i8);
                if tail[0] == report[0] && x.abs() <= 127 && y.abs() <= 127 {
                    tail[1] = x as i8 as u8;
                    tail[2] = y as i8 as u8;
                    s.last = tail.clone();
                    return;
                }
            }
        }
        self.enqueue(s, report);
    }

    fn descriptor(&self, s: &Setup, class_recipient: bool) -> Result<Vec<u8>, ()> {
        let (kind, index) = ((s.value >> 8) as u8, (s.value & 0xff) as u8);
        match (class_recipient, kind) {
            (false, 1) => Ok(self.kind.device_descriptor()),
            (false, 2) if index == 0 => Ok(self.kind.configuration_descriptor()),
            (false, 3) => match index {
                0 => Ok(vec![4, 3, 0x09, 0x04]),
                1 => Ok(string_descriptor("iris")),
                2 => Ok(string_descriptor(self.kind.product())),
                _ => Err(()),
            },
            (true, 0x21) if s.index == 0 => Ok(self.kind.hid_descriptor()),
            (true, 0x22) if s.index == 0 => Ok(self.kind.report_descriptor().to_vec()),
            _ => Err(()),
        }
    }

    /// Decide a request. No-data requests take effect here; SET_REPORT waits
    /// for its data.
    fn request(&self, st: &mut State, s: &Setup) -> Result<Plan, ()> {
        let recipient = s.request_type & 0x1f;
        let kind = (s.request_type >> 5) & 3;
        let to_host = s.device_to_host();
        let reply = |v: Vec<u8>| if to_host { Ok(Plan::In(v)) } else { Err(()) };
        let command = |f: &mut dyn FnMut()| {
            if to_host || s.length != 0 {
                return Err(());
            }
            f();
            Ok(Plan::None)
        };
        match (kind, recipient, s.request) {
            // GET_STATUS: self-powered and remote wakeup both clear.
            (0, 0 | 1, 0) => reply(vec![0, 0]),
            (0, 2, 0) if s.index == 0x81 => reply(vec![u8::from(st.halted), 0]),
            (0, 2, 0) if s.index & 0x7f == 0 => reply(vec![0, 0]),
            (0, 2, 1) if s.index == 0x81 && s.value == 0 => command(&mut || st.halted = false),
            (0, 2, 3) if s.index == 0x81 && s.value == 0 => command(&mut || st.halted = true),
            (0, 0, 5) if s.value < 128 && s.index == 0 => command(&mut || st.pending_address = Some(s.value as u8)),
            (0, 0, 6) => reply(self.descriptor(s, false)?),
            (0, 0, 8) => reply(vec![u8::from(st.configured)]),
            (0, 0, 9) if s.value <= 1 && s.index == 0 => command(&mut || {
                st.configured = s.value == 1;
                st.last_sent_ms = None;
                st.queue.clear();
                st.halted = false;
            }),
            (0, 1, 6) => reply(self.descriptor(s, true)?),
            (0, 1, 10) if s.index == 0 => reply(vec![0]),
            (0, 1, 11) if s.index == 0 && s.value == 0 => command(&mut || {}),
            // The HID class requests, addressed to interface 0.
            (1, 1, 1) if s.index == 0 && to_host => match (s.value >> 8, s.value & 0xff) {
                (1, 0) => Ok(Plan::In(st.last.clone())),
                (2, 0) if self.kind == Kind::Keyboard => Ok(Plan::In(vec![st.leds])),
                _ => Err(()),
            },
            (1, 1, 2) if s.index == 0 => reply(vec![st.idle]),
            (1, 1, 3) if s.index == 0 => reply(vec![st.protocol]),
            (1, 1, 9) if s.index == 0 && !to_host => {
                // Output report 0: the keyboard's LEDs, one byte.
                if self.kind == Kind::Keyboard && s.value == 0x0200 && s.length == 1 {
                    Ok(Plan::Out)
                } else {
                    Err(())
                }
            }
            (1, 1, 10) if s.index == 0 && s.value & 0xff == 0 => command(&mut || {
                st.idle = (s.value >> 8) as u8;
                st.last_sent_ms = None;
            }),
            (1, 1, 11) if s.index == 0 && s.value <= 1 => command(&mut || st.protocol = s.value as u8),
            _ => Err(()),
        }
    }

    fn setup(&self, st: &mut State, packet: &UsbPacket) -> UsbOutcome {
        let Some(setup) = Setup::parse(&packet.data) else {
            crate::dlog!(crate::devlog::LogModule::Usb, "usb hid {:?}: SETUP of {} bytes ignored", self.kind, packet.data.len());
            return UsbOutcome::NoResponse;
        };
        st.pending_address = None;
        st.control = match self.request(st, &setup) {
            Ok(Plan::None) => Control::StatusIn,
            Ok(Plan::In(mut data)) => {
                data.truncate(usize::from(setup.length));
                Control::DataIn { data, pos: 0 }
            }
            Ok(Plan::Out) => Control::DataOut { setup, got: vec![] },
            Err(()) => {
                crate::dlog!(
                    crate::devlog::LogModule::Usb,
                    "usb hid {:?}: request {:#04x}/{:#04x} value {:#06x} index {:#06x} refused",
                    self.kind, setup.request_type, setup.request, setup.value, setup.index
                );
                Control::Stalled
            }
        };
        UsbOutcome::Ack(vec![])
    }

    fn control_in(&self, st: &mut State, capacity: usize) -> UsbOutcome {
        match &mut st.control {
            Control::DataIn { data, pos } => {
                // Once the data is out, a further IN gets an empty packet:
                // the host asked for more than a whole number of packets.
                let n = capacity.min(PACKET).min(data.len() - *pos);
                let chunk = data[*pos..*pos + n].to_vec();
                *pos += n;
                UsbOutcome::Ack(chunk)
            }
            Control::StatusIn => {
                if let Some(a) = st.pending_address.take() {
                    st.address = a;
                }
                st.control = Control::Idle;
                UsbOutcome::Ack(vec![])
            }
            _ => UsbOutcome::Stall,
        }
    }

    fn control_out(&self, st: &mut State, packet: &UsbPacket) -> UsbOutcome {
        match &mut st.control {
            // The status stage of a device-to-host request.
            Control::DataIn { .. } if packet.data.is_empty() => {
                st.control = Control::Idle;
                UsbOutcome::Ack(vec![])
            }
            Control::DataOut { setup, got } => {
                got.extend_from_slice(&packet.data);
                let want = usize::from(setup.length);
                if got.len() > want {
                    st.control = Control::Stalled;
                    return UsbOutcome::Stall;
                }
                if got.len() == want {
                    st.leds = got[0];
                    st.control = Control::StatusIn;
                }
                UsbOutcome::Ack(vec![])
            }
            _ => UsbOutcome::Stall,
        }
    }

    fn interrupt_in(&self, st: &mut State, now_ms: u64) -> UsbOutcome {
        if !st.configured {
            return UsbOutcome::NoResponse;
        }
        if st.halted {
            return UsbOutcome::Stall;
        }
        if let Some(report) = st.queue.pop_front() {
            st.last_sent_ms = Some(now_ms);
            return UsbOutcome::Ack(report);
        }
        // HID 1.11 section 7.2.4: a nonzero idle rate repeats the current
        // report that often even when nothing changed. The timer starts at
        // the first poll.
        let since = *st.last_sent_ms.get_or_insert(now_ms);
        if st.idle != 0 && now_ms.saturating_sub(since) >= u64::from(st.idle) * 4 {
            st.last_sent_ms = Some(now_ms);
            let mut report = st.last.clone();
            if self.kind == Kind::Mouse {
                // A repeat restates the buttons; motion is not state.
                report = vec![st.buttons, 0, 0];
            }
            return UsbOutcome::Ack(report);
        }
        UsbOutcome::Nak
    }
}

impl UsbFunction for HidFunction {
    fn reset(&self) {
        let mut s = self.state.lock();
        s.address = 0;
        s.pending_address = None;
        s.configured = false;
        s.protocol = 1;
        s.idle = self.kind.default_idle();
        s.last_sent_ms = None;
        s.halted = false;
        s.control = Control::Idle;
        s.queue.clear();
        s.leds = 0;
        // Which keys and buttons are down is the host's hands, not the
        // bus's, so a bus reset leaves them.
    }
    fn address(&self) -> u8 {
        self.state.lock().address
    }
    fn low_speed(&self) -> bool {
        true
    }
    fn transact(&self, packet: UsbPacket) -> UsbOutcome {
        let mut s = self.state.lock();
        match (packet.endpoint, packet.pid) {
            (0, Pid::Setup) => self.setup(&mut s, &packet),
            (0, Pid::In) => self.control_in(&mut s, packet.capacity),
            (0, Pid::Out) => self.control_out(&mut s, &packet),
            (1, Pid::In) => self.interrupt_in(&mut s, packet.time_ms),
            (endpoint, pid) => {
                crate::dlog!(crate::devlog::LogModule::Usb, "usb hid {:?}: {pid:?} to absent endpoint {endpoint}", self.kind);
                UsbOutcome::NoResponse
            }
        }
    }
}

enum Usage {
    /// Bit of the report's modifier byte.
    Modifier(u8),
    /// Keyboard/Keypad page usage ID.
    Key(u8),
}

/// The Keyboard/Keypad page (HID Usage Tables 1.12, page 0x07) usage of a
/// physical key. Keys the page's boot range (up to 0x65) lacks are dropped.
fn usage_of(key: KeyCode) -> Option<Usage> {
    use KeyCode::*;
    use Usage::*;
    Some(match key {
        ControlLeft => Modifier(0),
        ShiftLeft => Modifier(1),
        AltLeft => Modifier(2),
        SuperLeft => Modifier(3),
        ControlRight => Modifier(4),
        ShiftRight => Modifier(5),
        AltRight => Modifier(6),
        SuperRight => Modifier(7),
        KeyA => Key(0x04), KeyB => Key(0x05), KeyC => Key(0x06), KeyD => Key(0x07),
        KeyE => Key(0x08), KeyF => Key(0x09), KeyG => Key(0x0a), KeyH => Key(0x0b),
        KeyI => Key(0x0c), KeyJ => Key(0x0d), KeyK => Key(0x0e), KeyL => Key(0x0f),
        KeyM => Key(0x10), KeyN => Key(0x11), KeyO => Key(0x12), KeyP => Key(0x13),
        KeyQ => Key(0x14), KeyR => Key(0x15), KeyS => Key(0x16), KeyT => Key(0x17),
        KeyU => Key(0x18), KeyV => Key(0x19), KeyW => Key(0x1a), KeyX => Key(0x1b),
        KeyY => Key(0x1c), KeyZ => Key(0x1d),
        Digit1 => Key(0x1e), Digit2 => Key(0x1f), Digit3 => Key(0x20), Digit4 => Key(0x21),
        Digit5 => Key(0x22), Digit6 => Key(0x23), Digit7 => Key(0x24), Digit8 => Key(0x25),
        Digit9 => Key(0x26), Digit0 => Key(0x27),
        Enter => Key(0x28), Escape => Key(0x29), Backspace => Key(0x2a), Tab => Key(0x2b),
        Space => Key(0x2c), Minus => Key(0x2d), Equal => Key(0x2e),
        BracketLeft => Key(0x2f), BracketRight => Key(0x30), Backslash => Key(0x31),
        Semicolon => Key(0x33), Quote => Key(0x34), Backquote => Key(0x35),
        Comma => Key(0x36), Period => Key(0x37), Slash => Key(0x38), CapsLock => Key(0x39),
        F1 => Key(0x3a), F2 => Key(0x3b), F3 => Key(0x3c), F4 => Key(0x3d),
        F5 => Key(0x3e), F6 => Key(0x3f), F7 => Key(0x40), F8 => Key(0x41),
        F9 => Key(0x42), F10 => Key(0x43), F11 => Key(0x44), F12 => Key(0x45),
        PrintScreen => Key(0x46), ScrollLock => Key(0x47), Pause => Key(0x48),
        Insert => Key(0x49), Home => Key(0x4a), PageUp => Key(0x4b), Delete => Key(0x4c),
        End => Key(0x4d), PageDown => Key(0x4e),
        ArrowRight => Key(0x4f), ArrowLeft => Key(0x50), ArrowDown => Key(0x51), ArrowUp => Key(0x52),
        NumLock => Key(0x53), NumpadDivide => Key(0x54), NumpadMultiply => Key(0x55),
        NumpadSubtract => Key(0x56), NumpadAdd => Key(0x57), NumpadEnter => Key(0x58),
        Numpad1 => Key(0x59), Numpad2 => Key(0x5a), Numpad3 => Key(0x5b), Numpad4 => Key(0x5c),
        Numpad5 => Key(0x5d), Numpad6 => Key(0x5e), Numpad7 => Key(0x5f), Numpad8 => Key(0x60),
        Numpad9 => Key(0x61), Numpad0 => Key(0x62), NumpadDecimal => Key(0x63),
        IntlBackslash => Key(0x64), ContextMenu => Key(0x65),
        _ => return None,
    })
}

/// The host's keyboard and mouse as the two USB functions.
pub struct UsbHidInput {
    keyboard: Arc<HidFunction>,
    mouse: Arc<HidFunction>,
}
impl UsbHidInput {
    pub fn new(keyboard: Arc<HidFunction>, mouse: Arc<HidFunction>) -> Arc<Self> {
        Arc::new(Self { keyboard, mouse })
    }
}
impl InputSink for UsbHidInput {
    fn key(&self, key: KeyCode, pressed: bool) {
        self.keyboard.key(key, pressed);
    }
    fn mouse(&self, buttons: u8, dx: i32, dy: i32, _wheel: i32) {
        self.mouse.pointer(buttons, dx, dy);
    }
    fn mouse_enabled(&self) -> bool {
        self.mouse.configured()
    }
    fn mouse_pending(&self) -> usize {
        self.mouse.pending()
    }
}
