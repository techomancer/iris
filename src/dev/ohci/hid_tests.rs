// The keyboard and mouse are driven the way a host driver drives them: control
// transfers and interrupt polls as transfer descriptors the OHCI schedule
// walks, with no test-only path into the functions.
use super::hid::{HidFunction, Kind, UsbHidInput};
use super::tests::{fake_clock_ohci, start_port_reset, Memory};
use super::*;
use crate::dev::ps2::{InputSink, Ps2Controller};
use winit::keyboard::KeyCode;

struct Bus {
    h: Arc<Ohci>,
    m: Arc<Memory>,
}
impl Bus {
    fn new() -> Self {
        let (h, _) = fake_clock_ohci(0);
        let m = Memory::new();
        h.state().connect_dma(m.clone());
        h.state().config().write(1, 6);
        m.words(0x1000, &[0; 64]);
        h.bar_write(0, 0x18, 0x1000, 4);
        // Operational, control list enabled.
        h.bar_write(0, 0x04, 0x90, 4);
        Bus { h, m }
    }
    /// Plug `f` into `port` and reset it the way a driver does.
    fn plug(&self, port: usize, f: Arc<HidFunction>) {
        self.h.attach(port, f).unwrap();
        start_port_reset(&self.h, port as u32);
        self.h.advance_frames(10);
    }
    /// One transaction on a control-list endpoint. Returns the condition
    /// code the descriptor ended with (15 if the function NAKed) and the
    /// bytes an IN delivered.
    fn txn(&self, addr: u32, ep: u32, pid: Pid, out: &[u8], in_len: usize) -> (u32, Vec<u8>) {
        let len = if pid == Pid::In { in_len } else { out.len() };
        self.m.bytes.lock()[0x4000..0x4040].fill(0xa5);
        self.m.bytes.lock()[0x4000..0x4000 + out.len()].copy_from_slice(out);
        let (cbp, be) = if len == 0 { (0, 0) } else { (0x4000, 0x4000 + len as u32 - 1) };
        // Data toggle and low speed are fixed: the functions do not check toggles.
        self.m.words(0x2000, &[addr | (ep << 7) | (8 << 16) | (1 << 13), 0x3020, 0x3000, 0]);
        // Buffer rounding lets a short packet end an IN cleanly.
        self.m.words(0x3000, &[0xf000_0000 | (1 << 18) | ((pid as u32) << 19) | (3 << 24), cbp, 0x3020, be]);
        self.m.words(0x3020, &[0; 4]);
        self.h.bar_write(0, 0x20, 0x2000, 4);
        self.h.bar_write(0, 0x08, 2, 4);
        self.h.advance_frames(1);
        // Take the done queue so the next transaction can publish.
        self.h.bar_write(0, 0x0c, WDH, 4);
        let td = self.m.word(0x3000);
        let cc = td >> 28;
        let got = match self.m.word(0x3004) {
            _ if pid != Pid::In || cc != 0 => 0,
            0 => len,
            cbp => cbp as usize - 0x4000,
        };
        (cc, self.m.bytes.lock()[0x4000..0x4000 + got].to_vec())
    }
    fn setup(&self, addr: u32, s: [u8; 8]) {
        assert_eq!(self.txn(addr, 0, Pid::Setup, &s, 0).0, 0, "SETUP {s:02x?}");
    }
    /// Device-to-host control transfer: SETUP, IN packets, empty OUT status.
    fn control_in(&self, addr: u32, request_type: u8, request: u8, value: u16, index: u16, length: u16) -> Vec<u8> {
        let mut s = [request_type, request, 0, 0, 0, 0, 0, 0];
        s[2..4].copy_from_slice(&value.to_le_bytes());
        s[4..6].copy_from_slice(&index.to_le_bytes());
        s[6..8].copy_from_slice(&length.to_le_bytes());
        self.setup(addr, s);
        let mut data = vec![];
        while data.len() < usize::from(length) {
            let (cc, chunk) = self.txn(addr, 0, Pid::In, &[], 8);
            assert_eq!(cc, 0);
            let short = chunk.len() < 8;
            data.extend(chunk);
            if short {
                break;
            }
        }
        assert_eq!(self.txn(addr, 0, Pid::Out, &[], 0).0, 0, "status stage");
        data
    }
    /// Host-to-device request with no data stage and an IN status.
    fn control_none(&self, addr: u32, request_type: u8, request: u8, value: u16, index: u16) {
        let mut s = [request_type, request, 0, 0, 0, 0, 0, 0];
        s[2..4].copy_from_slice(&value.to_le_bytes());
        s[4..6].copy_from_slice(&index.to_le_bytes());
        self.setup(addr, s);
        assert_eq!(self.txn(addr, 0, Pid::In, &[], 0).0, 0, "status stage");
    }
    fn poll(&self, addr: u32) -> Option<Vec<u8>> {
        match self.txn(addr, 1, Pid::In, &[], 8) {
            (15, _) => None,
            (0, report) => Some(report),
            (cc, _) => panic!("interrupt poll ended with condition code {cc}"),
        }
    }
    /// Enumerate as a driver does: read the device descriptor, give it an
    /// address, configure it. Returns the configuration descriptor.
    fn enumerate(&self, address: u32) -> Vec<u8> {
        let first = self.control_in(0, 0x80, 6, 0x0100, 0, 8);
        assert_eq!(first.len(), 8);
        self.control_none(0, 0x00, 5, address as u16, 0);
        let config = self.control_in(address, 0x80, 6, 0x0200, 0, 255);
        self.control_none(address, 0x00, 9, 1, 0);
        config
    }
}

/// The device descriptor both functions give: USB 1.1, class per interface,
/// 8-byte control packets, the pid.codes test IDs, one configuration.
const DEVICE: [u8; 18] = [18, 1, 0x10, 0x01, 0, 0, 0, 8, 0x09, 0x12, 0x01, 0x00, 0x00, 0x01, 1, 2, 0, 1];
/// The configuration for a boot protocol: 9 + 9 + 9 + 7 bytes, with the HID
/// descriptor naming the report descriptor's length.
fn configuration(protocol: u8, report_len: usize) -> Vec<u8> {
    let mut c = vec![9, 2, 34, 0, 1, 1, 0, 0x80, 50];
    c.extend([9, 4, 0, 0, 1, 3, 1, protocol, 0]);
    c.extend([9, 0x21, 0x11, 0x01, 0, 1, 0x22, report_len as u8, 0]);
    c.extend([7, 5, 0x81, 3, 8, 0, 10]);
    c
}
/// HID 1.11 appendix B.1 with the key array's maximum at 101 (appendix E.6),
/// spelt out as items.
const KEYBOARD_REPORT: [u8; 63] = [
    0x05, 0x01, // Usage Page (Generic Desktop)
    0x09, 0x06, // Usage (Keyboard)
    0xa1, 0x01, // Collection (Application)
    0x05, 0x07, // Usage Page (Keyboard/Keypad)
    0x19, 0xe0, 0x29, 0xe7, // Usage Minimum 224, Maximum 231: the modifiers
    0x15, 0x00, 0x25, 0x01, // Logical Minimum 0, Maximum 1
    0x75, 0x01, 0x95, 0x08, // Report Size 1, Count 8
    0x81, 0x02, // Input (Data, Variable, Absolute)
    0x95, 0x01, 0x75, 0x08, 0x81, 0x01, // one reserved byte
    0x95, 0x05, 0x75, 0x01, // five LED bits
    0x05, 0x08, 0x19, 0x01, 0x29, 0x05, // Usage Page (LEDs), Usage 1 to 5
    0x91, 0x02, // Output (Data, Variable, Absolute)
    0x95, 0x01, 0x75, 0x03, 0x91, 0x01, // three bits of LED padding
    0x95, 0x06, 0x75, 0x08, // six one-byte keys
    0x15, 0x00, 0x25, 0x65, // Logical Minimum 0, Maximum 101
    0x05, 0x07, 0x19, 0x00, 0x29, 0x65, // Usage Page (Keyboard/Keypad), 0 to 101
    0x81, 0x00, // Input (Data, Array)
    0xc0, // End Collection
];
/// HID 1.11 appendix B.2, as items.
const MOUSE_REPORT: [u8; 50] = [
    0x05, 0x01, 0x09, 0x02, 0xa1, 0x01, // Generic Desktop, Mouse, Application
    0x09, 0x01, 0xa1, 0x00, // Usage (Pointer), Collection (Physical)
    0x05, 0x09, 0x19, 0x01, 0x29, 0x03, // Button page, buttons 1 to 3
    0x15, 0x00, 0x25, 0x01, 0x95, 0x03, 0x75, 0x01, 0x81, 0x02, // three button bits
    0x95, 0x01, 0x75, 0x05, 0x81, 0x01, // five bits of padding
    0x05, 0x01, 0x09, 0x30, 0x09, 0x31, // Generic Desktop, X and Y
    0x15, 0x81, 0x25, 0x7f, 0x75, 0x08, 0x95, 0x02, // -127 to 127, two bytes
    0x81, 0x06, // Input (Data, Variable, Relative)
    0xc0, 0xc0,
];

/// USB 1.1 section 9.4 and HID 1.11 appendix B: the descriptors a host reads
/// to enumerate the keyboard, and the address and configuration it assigns.
#[test]
fn keyboard_enumerates_with_boot_keyboard_descriptors() {
    let bus = Bus::new();
    let kb = HidFunction::keyboard();
    bus.plug(0, kb.clone());
    assert_eq!(kb.address(), 0);

    let device = bus.control_in(0, 0x80, 6, 0x0100, 0, 18);
    assert_eq!(device, DEVICE, "the whole device descriptor");

    bus.control_none(0, 0x00, 5, 5, 0);
    assert_eq!(kb.address(), 5);
    let config = bus.control_in(5, 0x80, 6, 0x0200, 0, 255);
    assert_eq!(config, configuration(1, KEYBOARD_REPORT.len()), "the whole configuration, keyboard");

    // The HID descriptor names the report descriptor's length.
    let n = u16::from_le_bytes([config[25], config[26]]);
    let report = bus.control_in(5, 0x81, 6, 0x2200, 0, 255);
    assert_eq!(report.len(), usize::from(n));
    assert_eq!(report, KEYBOARD_REPORT, "the whole keyboard report descriptor");
    let hid = bus.control_in(5, 0x81, 6, 0x2100, 0, 255);
    assert_eq!(hid, &config[18..27]);

    // Strings: the language list, then the product.
    assert_eq!(bus.control_in(5, 0x80, 6, 0x0300, 0, 255), [4, 3, 9, 4]);
    let product = bus.control_in(5, 0x80, 6, 0x0302, 0, 255);
    assert_eq!(product[1], 3);
    assert_eq!(product[0] as usize, product.len());

    assert!(!kb.configured());
    bus.control_none(5, 0x00, 9, 1, 0);
    assert!(kb.configured());
    assert_eq!(bus.control_in(5, 0x80, 8, 0, 0, 1), [1]);
    // A device that has been reset is back at address zero, unconfigured.
    kb.reset();
    assert_eq!((kb.address(), kb.configured()), (0, false));
}

#[test]
fn mouse_enumerates_with_boot_mouse_descriptors() {
    let bus = Bus::new();
    let mouse = HidFunction::mouse();
    bus.plug(0, mouse.clone());
    let config = bus.enumerate(6);
    assert_eq!(config, configuration(2, MOUSE_REPORT.len()), "the whole configuration, mouse");
    let report = bus.control_in(6, 0x81, 6, 0x2200, 0, 255);
    assert_eq!(report, MOUSE_REPORT, "the whole mouse report descriptor");
    assert!(mouse.configured());
    assert!(mouse.low_speed());
}

/// The two functions share the bus once each has its own address.
#[test]
fn two_functions_enumerate_in_turn_and_answer_at_their_addresses() {
    let bus = Bus::new();
    let (kb, mouse) = (HidFunction::keyboard(), HidFunction::mouse());
    bus.plug(0, kb.clone());
    bus.enumerate(5);
    bus.plug(1, mouse.clone());
    bus.enumerate(6);
    let sink = UsbHidInput::new(kb, mouse);
    sink.key(KeyCode::KeyA, true);
    sink.mouse(0, 1, 0, 0);
    assert_eq!(bus.poll(5).unwrap()[2], 4);
    assert_eq!(bus.poll(6).unwrap(), [0, 1, 0]);
}

/// HID 1.11 appendix B.1: modifiers, reserved byte, then up to six usages.
#[test]
fn key_presses_become_boot_keyboard_reports() {
    let bus = Bus::new();
    let kb = HidFunction::keyboard();
    bus.plug(0, kb.clone());
    let sink = UsbHidInput::new(kb.clone(), HidFunction::new(Kind::Mouse));
    // Keys pressed before the host configures the device are not reported.
    sink.key(KeyCode::KeyZ, true);
    sink.key(KeyCode::KeyZ, false);
    bus.enumerate(5);
    assert_eq!(bus.poll(5), None, "nothing to report yet");

    sink.key(KeyCode::KeyA, true);
    assert_eq!(bus.poll(5).unwrap(), [0, 0, 0x04, 0, 0, 0, 0, 0]);
    assert_eq!(bus.poll(5), None);
    sink.key(KeyCode::ShiftLeft, true);
    sink.key(KeyCode::Digit1, true);
    assert_eq!(bus.poll(5).unwrap(), [0x02, 0, 0x04, 0, 0, 0, 0, 0]);
    assert_eq!(bus.poll(5).unwrap(), [0x02, 0, 0x04, 0x1e, 0, 0, 0, 0]);
    sink.key(KeyCode::KeyA, false);
    sink.key(KeyCode::Digit1, false);
    sink.key(KeyCode::ShiftLeft, false);
    assert_eq!(bus.poll(5).unwrap(), [0x02, 0, 0x1e, 0, 0, 0, 0, 0]);
    assert_eq!(bus.poll(5).unwrap(), [0x02, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(bus.poll(5).unwrap(), [0; 8]);
    // Right-hand modifiers and the Enter, arrow and function keys.
    sink.key(KeyCode::ControlRight, true);
    sink.key(KeyCode::Enter, true);
    sink.key(KeyCode::ArrowUp, true);
    sink.key(KeyCode::F12, true);
    let mut last = vec![];
    while let Some(r) = bus.poll(5) {
        last = r;
    }
    assert_eq!(last, [0x10, 0, 0x28, 0x52, 0x45, 0, 0, 0]);
    // A key the report cannot name changes nothing.
    sink.key(KeyCode::MediaPlayPause, true);
    assert_eq!(bus.poll(5), None);
}

/// A typed burst is a host-side batch: like the PS/2 queue, the keyboard
/// keeps every report until the guest polls it, however many there are.
#[test]
fn a_typed_burst_longer_than_the_queue_arrives_whole_and_in_order() {
    let bus = Bus::new();
    let kb = HidFunction::keyboard();
    bus.plug(0, kb.clone());
    let sink = UsbHidInput::new(kb.clone(), HidFunction::new(Kind::Mouse));
    bus.enumerate(5);
    let keys = [KeyCode::KeyA, KeyCode::KeyB, KeyCode::KeyC, KeyCode::KeyD];
    let usages = [0x04u8, 0x05, 0x06, 0x07];
    // 100 presses and releases is 200 reports, well past 64.
    for i in 0..100 {
        sink.key(keys[i % 4], true);
        sink.key(keys[i % 4], false);
    }
    for i in 0..100 {
        assert_eq!(bus.poll(5).unwrap(), [0, 0, usages[i % 4], 0, 0, 0, 0, 0], "press {i}");
        assert_eq!(bus.poll(5).unwrap(), [0; 8], "release {i}");
    }
    assert_eq!(bus.poll(5), None);
}

/// HID 1.11 appendix B.1: more than six keys down reports the rollover error.
#[test]
fn seventh_key_reports_rollover() {
    let bus = Bus::new();
    let kb = HidFunction::keyboard();
    bus.plug(0, kb.clone());
    bus.enumerate(5);
    let sink = UsbHidInput::new(kb, HidFunction::mouse());
    for k in [KeyCode::KeyA, KeyCode::KeyB, KeyCode::KeyC, KeyCode::KeyD, KeyCode::KeyE, KeyCode::KeyF, KeyCode::KeyG] {
        sink.key(k, true);
    }
    let mut reports = vec![];
    while let Some(r) = bus.poll(5) {
        reports.push(r);
    }
    assert_eq!(reports.len(), 7);
    assert_eq!(reports[5], [0, 0, 4, 5, 6, 7, 8, 9]);
    assert_eq!(reports[6], [0, 0, 1, 1, 1, 1, 1, 1]);
}

/// HID 1.11 appendix B.2: buttons, then signed X and Y, positive Y down.
#[test]
fn mouse_motion_and_buttons_become_boot_mouse_reports() {
    let bus = Bus::new();
    let mouse = HidFunction::mouse();
    bus.plug(0, mouse.clone());
    let sink = UsbHidInput::new(HidFunction::keyboard(), mouse.clone());
    assert!(!sink.mouse_enabled(), "not configured yet");
    sink.mouse(0, 9, 9, 0);
    bus.enumerate(6);
    assert!(sink.mouse_enabled());
    assert_eq!(sink.mouse_pending(), 0, "motion before configuration is dropped");
    assert_eq!(bus.poll(6), None);

    sink.mouse(1, 5, -3, 0);
    assert_eq!(sink.mouse_pending(), 1);
    assert_eq!(bus.poll(6).unwrap(), [1, 5, 0xfd]);
    // A click with no motion is a report; no change at all is not.
    sink.mouse(3, 0, 0, 0);
    sink.mouse(3, 0, 0, 0);
    assert_eq!(bus.poll(6).unwrap(), [3, 0, 0]);
    assert_eq!(bus.poll(6), None);
    // An axis is one signed byte, so a long move leaves as several.
    sink.mouse(0, 200, -130, 0);
    assert_eq!(bus.poll(6).unwrap(), [0, 127, 0x81]);
    assert_eq!(bus.poll(6).unwrap(), [0, 73, 0xfd]);
    assert_eq!(bus.poll(6), None);
}

/// A host that does not poll gets the moves summed, not dropped.
#[test]
fn a_slow_host_gets_mouse_moves_folded_together() {
    let bus = Bus::new();
    let mouse = HidFunction::mouse();
    bus.plug(0, mouse.clone());
    bus.enumerate(6);
    for _ in 0..20 {
        mouse_move(&mouse, 1);
    }
    let mut sum = 0;
    let mut n = 0;
    while let Some(r) = bus.poll(6) {
        sum += i32::from(r[1] as i8);
        n += 1;
    }
    assert_eq!(sum, 20);
    assert!(n < 20, "the backlog was folded");
}
fn mouse_move(m: &Arc<HidFunction>, dx: i32) {
    UsbHidInput::new(HidFunction::keyboard(), m.clone()).mouse(0, dx, 0, 0);
}

/// HID 1.11 section 7.2.2 and appendix B.1: SET_REPORT of the output report
/// sets the keyboard LEDs.
#[test]
fn set_report_sets_the_keyboard_leds() {
    let bus = Bus::new();
    let kb = HidFunction::keyboard();
    bus.plug(0, kb.clone());
    bus.enumerate(5);
    assert_eq!(kb.leds(), 0);
    bus.setup(5, [0x21, 9, 0x00, 0x02, 0, 0, 1, 0]);
    assert_eq!(kb.leds(), 0, "not applied before the data arrives");
    assert_eq!(bus.txn(5, 0, Pid::Out, &[0x03], 0).0, 0);
    assert_eq!(kb.leds(), 3);
    assert_eq!(bus.txn(5, 0, Pid::In, &[], 0).0, 0, "status stage");
    assert_eq!(bus.control_in(5, 0xa1, 1, 0x0200, 0, 1), [3]);
    // GET_REPORT of the input report is the last report built.
    let sink = UsbHidInput::new(kb.clone(), HidFunction::mouse());
    sink.key(KeyCode::KeyQ, true);
    assert_eq!(bus.control_in(5, 0xa1, 1, 0x0100, 0, 8), [0, 0, 0x14, 0, 0, 0, 0, 0]);
    // The wrong size of output report is refused.
    bus.setup(5, [0x21, 9, 0x00, 0x02, 0, 0, 2, 0]);
    assert_eq!(bus.txn(5, 0, Pid::Out, &[0, 0], 0).0, 4, "stall");
    assert_eq!(kb.leds(), 3);
}

/// HID 1.11 section 7.2: protocol and idle round-trip, and a nonzero idle
/// rate repeats the keyboard report.
#[test]
fn idle_and_protocol_requests() {
    let bus = Bus::new();
    let kb = HidFunction::keyboard();
    bus.plug(0, kb.clone());
    bus.enumerate(5);
    assert_eq!(bus.control_in(5, 0xa1, 3, 0, 0, 1), [1], "report protocol at power-up");
    bus.control_none(5, 0x21, 0x0b, 0, 0);
    assert_eq!(bus.control_in(5, 0xa1, 3, 0, 0, 1), [0]);
    assert_eq!(bus.control_in(5, 0xa1, 2, 0, 0, 1), [125], "the recommended 500 ms keyboard default");
    bus.control_none(5, 0x21, 0x0a, 0x0200, 0);
    assert_eq!(bus.control_in(5, 0xa1, 2, 0, 0, 1), [2]);
    // 2 x 4 ms: every frame that passes 8 ms repeats the report.
    UsbHidInput::new(kb.clone(), HidFunction::mouse()).key(KeyCode::KeyB, true);
    assert_eq!(bus.poll(5).unwrap()[2], 5);
    assert_eq!(bus.poll(5), None);
    // Each poll is a frame, so the repeat comes within a few.
    let repeat = (0..12).find_map(|_| bus.poll(5));
    assert_eq!(repeat.unwrap()[2], 5, "repeated at the idle rate");
    bus.control_none(5, 0x21, 0x0a, 0, 0);
    assert!((0..20).all(|_| bus.poll(5).is_none()), "idle 0 reports on change only");
}

/// A keyboard left at its default idle repeats its report every 500 ms; the
/// mouse's default is none. Section 7.2.4 exempts neither from a SET_IDLE, so
/// a mouse repeats its buttons, without the motion.
#[test]
fn default_idle_and_mouse_idle_repeats() {
    let bus = Bus::new();
    let (kb, mouse) = (HidFunction::keyboard(), HidFunction::mouse());
    bus.plug(0, kb.clone());
    bus.enumerate(5);
    bus.plug(1, mouse.clone());
    bus.enumerate(6);
    let sink = UsbHidInput::new(kb, mouse);
    sink.key(KeyCode::KeyC, true);
    assert_eq!(bus.poll(5).unwrap()[2], 6);
    assert_eq!(bus.poll(5), None);
    // Each poll is a frame of 1 ms; a stale descriptor would otherwise take
    // the repeat inside advance_frames.
    let mut frames = 0;
    let repeat = loop {
        frames += 1;
        assert!(frames < 600, "no repeat");
        if let Some(r) = bus.poll(5) {
            break r;
        }
    };
    assert!((495..=505).contains(&frames), "repeated after {frames} ms, not 500");
    assert_eq!(repeat, [0, 0, 6, 0, 0, 0, 0, 0], "the held key, repeated");

    assert_eq!(bus.control_in(6, 0xa1, 2, 0, 0, 1), [0], "no mouse default");
    sink.mouse(1, 5, 7, 0);
    assert_eq!(bus.poll(6).unwrap(), [1, 5, 7]);
    assert!((0..40).all(|_| bus.poll(6).is_none()), "no repeats at idle 0");
    bus.control_none(6, 0x21, 0x0a, 0x0300, 0);
    let repeat = (0..20).find_map(|_| bus.poll(6));
    assert_eq!(repeat.unwrap(), [1, 0, 0], "buttons repeat, motion does not");
}

/// A request the function does not implement stalls, and the next SETUP
/// recovers the pipe. A halted interrupt endpoint stalls until cleared.
#[test]
fn unsupported_requests_stall_and_endpoint_halt_is_clearable() {
    let bus = Bus::new();
    let kb = HidFunction::keyboard();
    bus.plug(0, kb.clone());
    bus.enumerate(5);
    bus.setup(5, [0x80, 6, 0, 6, 0, 0, 10, 0]);
    assert_eq!(bus.txn(5, 0, Pid::In, &[], 8).0, 4, "device qualifier: a 1.1 device has none");
    bus.setup(5, [0x40, 0x99, 0, 0, 0, 0, 0, 0]);
    assert_eq!(bus.txn(5, 0, Pid::In, &[], 0).0, 4, "vendor request");
    assert_eq!(bus.control_in(5, 0x80, 8, 0, 0, 1), [1], "the pipe recovered");

    bus.control_none(5, 0x02, 3, 0, 0x81);
    assert_eq!(bus.control_in(5, 0x82, 0, 0, 0x81, 2), [1, 0], "endpoint halted");
    assert_eq!(bus.txn(5, 1, Pid::In, &[], 8).0, 4);
    bus.control_none(5, 0x02, 1, 0, 0x81);
    assert_eq!(bus.poll(5), None);
}

/// An unplugged-then-reset function is not reachable by its old address.
#[test]
fn port_reset_returns_the_function_to_address_zero() {
    let bus = Bus::new();
    let kb = HidFunction::keyboard();
    bus.plug(0, kb.clone());
    bus.enumerate(5);
    start_port_reset(&bus.h, 0);
    bus.h.advance_frames(10);
    assert_eq!((kb.address(), kb.configured()), (0, false));
    assert_eq!(bus.control_in(0, 0x80, 6, 0x0100, 0, 8).len(), 8);
}

/// The host's keys reach the USB keyboard through the PS/2 controller the UI
/// and monitor already talk to, with one keymap for typed text.
#[test]
fn ps2_controller_redirects_host_input_to_usb() {
    let bus = Bus::new();
    let (kb, mouse) = (HidFunction::keyboard(), HidFunction::mouse());
    bus.plug(0, kb.clone());
    bus.enumerate(5);
    bus.plug(1, mouse.clone());
    bus.enumerate(6);
    let ps2 = Ps2Controller::new(None);
    ps2.redirect_input(UsbHidInput::new(kb, mouse));
    assert!(ps2.mouse_enabled());
    assert!(ps2.type_char('A'));
    let mut reports = vec![];
    while let Some(r) = bus.poll(5) {
        reports.push(r);
    }
    assert_eq!(
        reports,
        [
            vec![0x02, 0, 0, 0, 0, 0, 0, 0],
            vec![0x02, 0, 0x04, 0, 0, 0, 0, 0],
            vec![0x02, 0, 0, 0, 0, 0, 0, 0],
            vec![0; 8],
        ]
    );
    ps2.push_mouse_input(4, -2, 7, 0);
    assert_eq!(ps2.mouse_pending(), 1);
    assert_eq!(bus.poll(6).unwrap(), [4, 0xfe, 7]);
    assert_eq!(ps2.mouse_pending(), 0);
}
