// The cases follow USB 1.1 chapter 9 and HID 1.11: descriptor layout and
// lengths, the control pipe's stages, and the boot report formats. The last
// ones put a device behind the OHCI model to show an interrupt transfer
// returns a report through guest memory.
use super::*;
use crate::dev::ohci::Ohci;
use crate::pci::{PciBusDevice, PciDma, PciDmaStatus};
use crate::traits::Device;

fn packet(endpoint: u8, pid: Pid, capacity: usize, data: &[u8], time_ms: u64) -> UsbPacket {
    UsbPacket { endpoint, pid, toggle: false, capacity, data: data.to_vec(), time_ms }
}

fn ack(o: UsbOutcome) -> Vec<u8> {
    match o {
        UsbOutcome::Ack(b) => b,
        other => panic!("expected an ACK, got {other:?}"),
    }
}

/// A control transfer toward the host: SETUP, IN packets of 8 until short,
/// then the OUT status.
fn control_in(f: &dyn UsbFunction, rt: u8, req: u8, value: u16, index: u16, length: u16) -> Result<Vec<u8>, UsbOutcome> {
    let mut setup = vec![rt, req];
    setup.extend(value.to_le_bytes());
    setup.extend(index.to_le_bytes());
    setup.extend(length.to_le_bytes());
    match f.transact(packet(0, Pid::Setup, 8, &setup, 0)) {
        UsbOutcome::Ack(_) => {}
        other => return Err(other),
    }
    let mut data = Vec::new();
    loop {
        let chunk = ack(f.transact(packet(0, Pid::In, 8, &[], 0)));
        let short = chunk.len() < 8;
        data.extend(chunk);
        if short || data.len() >= length as usize {
            break;
        }
    }
    ack(f.transact(packet(0, Pid::Out, 8, &[], 0)));
    Ok(data)
}

/// A control transfer with no data stage: SETUP then the IN status.
fn control_nodata(f: &dyn UsbFunction, rt: u8, req: u8, value: u16, index: u16) -> UsbOutcome {
    let mut setup = vec![rt, req];
    setup.extend(value.to_le_bytes());
    setup.extend(index.to_le_bytes());
    setup.extend(0u16.to_le_bytes());
    match f.transact(packet(0, Pid::Setup, 8, &setup, 0)) {
        UsbOutcome::Ack(_) => f.transact(packet(0, Pid::In, 8, &[], 0)),
        other => other,
    }
}

fn enumerate(f: &dyn UsbFunction, address: u8) {
    f.reset();
    assert!(ack(control_nodata(f, 0x00, SET_ADDRESS, address as u16, 0)).is_empty());
    assert_eq!(f.address(), address);
    assert!(ack(control_nodata(f, 0x00, SET_CONFIGURATION, 1, 0)).is_empty());
}

fn interrupt(f: &dyn UsbFunction, time_ms: u64) -> UsbOutcome {
    f.transact(packet(1, Pid::In, 8, &[], time_ms))
}

#[test]
fn the_device_descriptor_names_a_low_speed_hid_function() {
    for f in [&*UsbKeyboard::new() as &dyn UsbFunction, &*UsbMouse::new()] {
        assert!(f.low_speed());
        let d = control_in(f, 0x80, GET_DESCRIPTOR, 0x0100, 0, 64).unwrap();
        assert_eq!(d.len(), 18);
        assert_eq!(d[0], 18, "bLength");
        assert_eq!(d[1], 1, "bDescriptorType");
        assert_eq!(u16::from_le_bytes([d[2], d[3]]), 0x0110, "USB 1.1");
        assert_eq!(d[4], 0, "the class is in the interface");
        assert_eq!(d[7], 8, "a low-speed control pipe carries 8 bytes");
        assert_eq!(d[17], 1, "one configuration");
        // The guest may read just the first 8 bytes before addressing.
        assert_eq!(control_in(f, 0x80, GET_DESCRIPTOR, 0x0100, 0, 8).unwrap(), d[..8]);
    }
}

/// A configuration, its interface, the HID descriptor and the endpoint come
/// back in one block whose wTotalLength covers them (USB 1.1 9.6.2).
#[test]
fn the_configuration_block_describes_one_boot_interface_and_one_interrupt_endpoint() {
    for (f, protocol, report_len, packet_size) in [
        (&*UsbKeyboard::new() as &dyn UsbFunction, 1u8, KEYBOARD_REPORT_DESCRIPTOR.len(), 8u8),
        (&*UsbMouse::new(), 2, MOUSE_REPORT_DESCRIPTOR.len(), 4),
    ] {
        let c = control_in(f, 0x80, GET_DESCRIPTOR, 0x0200, 0, 255).unwrap();
        assert_eq!(c.len(), 34);
        assert_eq!(&c[..2], [9, 2]);
        assert_eq!(u16::from_le_bytes([c[2], c[3]]) as usize, c.len(), "wTotalLength");
        assert_eq!((c[4], c[5]), (1, 1), "one interface; configuration value 1");
        assert_eq!(&c[9..14], [9, 4, 0, 0, 1], "interface 0, no alternate, one endpoint");
        assert_eq!(&c[14..17], [3, 1, protocol], "HID class, boot subclass, keyboard or mouse");
        assert_eq!(&c[18..20], [9, 0x21], "HID descriptor");
        assert_eq!(u16::from_le_bytes([c[20], c[21]]), 0x0111, "bcdHID 1.11");
        assert_eq!(&c[24..26], [0x22, report_len as u8], "one report descriptor of its length");
        assert_eq!(&c[27..], [7, 5, 0x81, 3, packet_size, 0, 10], "interrupt IN 1, polled at 10 ms");
        // The HID descriptor is also readable on its own through the interface.
        assert_eq!(control_in(f, 0x81, GET_DESCRIPTOR, 0x2100, 0, 9).unwrap(), c[18..27]);
    }
}

#[test]
fn the_report_descriptors_are_the_boot_descriptors() {
    let k = UsbKeyboard::new();
    let r = control_in(&*k, 0x81, GET_DESCRIPTOR, 0x2200, 0, 255).unwrap();
    assert_eq!(r, KEYBOARD_REPORT_DESCRIPTOR);
    // Usage page generic desktop, usage keyboard, application collection.
    assert_eq!(&r[..6], [0x05, 0x01, 0x09, 0x06, 0xa1, 0x01]);
    assert_eq!(r.last(), Some(&0xc0), "collection ends");
    let m = UsbMouse::new();
    let r = control_in(&*m, 0x81, GET_DESCRIPTOR, 0x2200, 0, 255).unwrap();
    assert_eq!(r, MOUSE_REPORT_DESCRIPTOR);
    assert_eq!(&r[..6], [0x05, 0x01, 0x09, 0x02, 0xa1, 0x01]);
}

#[test]
fn strings_are_utf16_with_a_language_table() {
    let k = UsbKeyboard::new();
    assert_eq!(control_in(&*k, 0x80, GET_DESCRIPTOR, 0x0300, 0, 255).unwrap(), [4, 3, 0x09, 0x04]);
    let s = control_in(&*k, 0x80, GET_DESCRIPTOR, 0x0302, 0x0409, 255).unwrap();
    assert_eq!(s[0] as usize, s.len());
    let text: Vec<u16> = s[2..].chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    assert_eq!(String::from_utf16(&text).unwrap(), "USB Keyboard");
    assert!(control_in(&*k, 0x80, GET_DESCRIPTOR, 0x0309, 0x0409, 255).is_err(), "no such string");
}

/// SET_ADDRESS takes effect after its status stage, not before (USB 1.1 9.4.6).
#[test]
fn set_address_applies_after_the_status_stage() {
    let m = UsbMouse::new();
    let mut setup = vec![0x00, SET_ADDRESS, 9, 0, 0, 0, 0, 0];
    assert!(ack(m.transact(packet(0, Pid::Setup, 8, &setup, 0))).is_empty());
    assert_eq!(m.address(), 0, "still the default address during the status stage");
    assert!(ack(m.transact(packet(0, Pid::In, 8, &[], 0))).is_empty());
    assert_eq!(m.address(), 9);
    // An address above 127 is not one.
    setup[2] = 200;
    assert_eq!(m.transact(packet(0, Pid::Setup, 8, &setup, 0)), UsbOutcome::Stall);
    // A bus reset returns the device to the default address.
    m.reset();
    assert_eq!(m.address(), 0);
}

#[test]
fn configuration_state_is_readable_and_unknown_requests_stall() {
    let k = UsbKeyboard::new();
    assert_eq!(control_in(&*k, 0x80, GET_CONFIGURATION, 0, 0, 1).unwrap(), [0]);
    assert!(ack(control_nodata(&*k, 0x00, SET_CONFIGURATION, 1, 0)).is_empty());
    assert_eq!(control_in(&*k, 0x80, GET_CONFIGURATION, 0, 0, 1).unwrap(), [1]);
    assert_eq!(control_nodata(&*k, 0x00, SET_CONFIGURATION, 2, 0), UsbOutcome::Stall);
    assert_eq!(control_in(&*k, 0x81, GET_INTERFACE, 0, 0, 1).unwrap(), [0]);
    assert_eq!(control_in(&*k, 0x80, GET_STATUS, 0, 0, 2).unwrap(), [0, 0]);
    // No device qualifier on a low-speed-only device; no such request.
    assert!(control_in(&*k, 0x80, GET_DESCRIPTOR, 0x0600, 0, 10).is_err());
    assert_eq!(control_nodata(&*k, 0x00, 0x77, 0, 0), UsbOutcome::Stall);
    // A stalled request leaves the pipe usable for the next SETUP.
    assert_eq!(control_in(&*k, 0x80, GET_CONFIGURATION, 0, 0, 1).unwrap(), [1]);
    // The interrupt endpoint is not there before a configuration is selected.
    let fresh = UsbKeyboard::new();
    assert_eq!(interrupt(&*fresh, 0), UsbOutcome::Stall);
}

#[test]
fn hid_class_requests_set_and_report_idle_and_protocol() {
    let k = UsbKeyboard::new();
    // A device starts in report protocol with the keyboard's 500 ms idle.
    assert_eq!(control_in(&*k, 0xa1, GET_PROTOCOL, 0, 0, 1).unwrap(), [1]);
    assert_eq!(control_in(&*k, 0xa1, GET_IDLE, 0, 0, 1).unwrap(), [125]);
    assert!(ack(control_nodata(&*k, 0x21, SET_PROTOCOL, 0, 0)).is_empty());
    assert_eq!(control_in(&*k, 0xa1, GET_PROTOCOL, 0, 0, 1).unwrap(), [0]);
    // SET_IDLE carries the duration in the high byte of wValue.
    assert!(ack(control_nodata(&*k, 0x21, SET_IDLE, 0, 0)).is_empty());
    assert_eq!(control_in(&*k, 0xa1, GET_IDLE, 0, 0, 1).unwrap(), [0]);
    assert_eq!(control_nodata(&*k, 0x21, SET_PROTOCOL, 2, 0), UsbOutcome::Stall);
    // A mouse starts with no idle repeat.
    let m = UsbMouse::new();
    assert_eq!(control_in(&*m, 0xa1, GET_IDLE, 0, 0, 1).unwrap(), [0]);
}

#[test]
fn set_report_takes_the_keyboards_leds_and_a_mouse_has_none() {
    let k = UsbKeyboard::new();
    let setup = [0x21, SET_REPORT, 0, 2, 0, 0, 1, 0];
    assert!(ack(k.transact(packet(0, Pid::Setup, 8, &setup, 0))).is_empty());
    assert!(ack(k.transact(packet(0, Pid::Out, 8, &[0x1f | 0x80], 0))).is_empty());
    assert!(ack(k.transact(packet(0, Pid::In, 8, &[], 0))).is_empty(), "status stage");
    assert_eq!(k.leds(), 0x1f, "five LED bits");
    let m = UsbMouse::new();
    assert!(ack(m.transact(packet(0, Pid::Setup, 8, &setup, 0))).is_empty());
    assert_eq!(m.transact(packet(0, Pid::Out, 8, &[1], 0)), UsbOutcome::Stall);
}

#[test]
fn keyboard_reports_carry_modifiers_and_keys_in_press_order() {
    let k = UsbKeyboard::new();
    enumerate(&*k, 3);
    assert_eq!(interrupt(&*k, 1), UsbOutcome::Nak, "nothing to report yet");
    k.key(KeyCode::ShiftLeft, true);
    k.key(KeyCode::KeyA, true);
    k.key(KeyCode::KeyB, true);
    k.key(KeyCode::KeyA, false);
    k.key(KeyCode::ShiftLeft, false);
    k.key(KeyCode::KeyB, false);
    let reports: Vec<Vec<u8>> = (0..6).map(|i| ack(interrupt(&*k, i))).collect();
    assert_eq!(reports, [
        vec![0x02, 0, 0, 0, 0, 0, 0, 0],
        vec![0x02, 0, 0x04, 0, 0, 0, 0, 0],
        vec![0x02, 0, 0x04, 0x05, 0, 0, 0, 0],
        vec![0x02, 0, 0x05, 0, 0, 0, 0, 0],
        vec![0x00, 0, 0x05, 0, 0, 0, 0, 0],
        vec![0x00, 0, 0, 0, 0, 0, 0, 0],
    ], "a tap between polls still reports both edges");
    assert_eq!(interrupt(&*k, 7), UsbOutcome::Nak);
}

#[test]
fn keyboard_usages_follow_the_usage_table() {
    assert_eq!(usage(KeyCode::KeyA), Some(0x04));
    assert_eq!(usage(KeyCode::KeyZ), Some(0x1d));
    assert_eq!(usage(KeyCode::Digit1), Some(0x1e));
    assert_eq!(usage(KeyCode::Digit0), Some(0x27));
    assert_eq!(usage(KeyCode::Enter), Some(0x28));
    assert_eq!(usage(KeyCode::Escape), Some(0x29));
    assert_eq!(usage(KeyCode::Space), Some(0x2c));
    assert_eq!(usage(KeyCode::F1), Some(0x3a));
    assert_eq!(usage(KeyCode::F12), Some(0x45));
    assert_eq!(usage(KeyCode::ArrowRight), Some(0x4f));
    assert_eq!(usage(KeyCode::ArrowUp), Some(0x52));
    assert_eq!(usage(KeyCode::Numpad0), Some(0x62));
    assert_eq!(usage(KeyCode::ControlLeft), None, "modifiers have a report bit, not a usage slot");
    assert_eq!(modifier(KeyCode::ControlLeft), Some(0x01));
    assert_eq!(modifier(KeyCode::SuperRight), Some(0x80));
}

#[test]
fn a_seventh_key_reports_rollover_and_release_recovers() {
    let k = UsbKeyboard::new();
    enumerate(&*k, 1);
    let keys = [KeyCode::KeyA, KeyCode::KeyB, KeyCode::KeyC, KeyCode::KeyD, KeyCode::KeyE, KeyCode::KeyF, KeyCode::KeyG];
    for key in keys {
        k.key(key, true);
    }
    let reports: Vec<Vec<u8>> = (0..7).map(|i| ack(interrupt(&*k, i))).collect();
    assert_eq!(reports[5], [0, 0, 4, 5, 6, 7, 8, 9]);
    assert_eq!(reports[6], [0, 0, 1, 1, 1, 1, 1, 1], "ErrorRollOver");
    k.key(KeyCode::KeyG, false);
    assert_eq!(ack(interrupt(&*k, 8)), [0, 0, 4, 5, 6, 7, 8, 9]);
    // A repeated press of a held key is not a new edge.
    k.key(KeyCode::KeyA, true);
    assert_eq!(interrupt(&*k, 9), UsbOutcome::Nak);
}

#[test]
fn input_before_configuration_is_dropped_and_get_report_reads_the_state() {
    let k = UsbKeyboard::new();
    k.key(KeyCode::KeyA, true);
    enumerate(&*k, 2);
    assert_eq!(interrupt(&*k, 1), UsbOutcome::Nak, "the early key never reached the report queue");
    k.key(KeyCode::KeyQ, true);
    assert_eq!(control_in(&*k, 0xa1, GET_REPORT, 0x0100, 0, 8).unwrap(), [0, 0, 0x14, 0, 0, 0, 0, 0]);
}

/// With a nonzero idle rate the keyboard repeats its state when the period
/// passes without a change (HID 1.11 7.2.4), and with zero it stays quiet.
#[test]
fn the_idle_rate_repeats_the_last_report() {
    let k = UsbKeyboard::new();
    enumerate(&*k, 2);
    k.key(KeyCode::KeyA, true);
    assert_eq!(ack(interrupt(&*k, 10)), [0, 0, 4, 0, 0, 0, 0, 0]);
    assert_eq!(interrupt(&*k, 509), UsbOutcome::Nak, "500 ms not yet passed");
    assert_eq!(ack(interrupt(&*k, 510)), [0, 0, 4, 0, 0, 0, 0, 0]);
    assert!(ack(control_nodata(&*k, 0x21, SET_IDLE, 0, 0)).is_empty());
    assert_eq!(interrupt(&*k, 5000), UsbOutcome::Nak);
}

#[test]
fn mouse_reports_carry_buttons_and_signed_motion() {
    let m = UsbMouse::new();
    enumerate(&*m, 4);
    assert!(m.mouse_enabled());
    assert_eq!(interrupt(&*m, 1), UsbOutcome::Nak);
    // Host dy is positive down the screen, as HID Y is.
    m.mouse(0, 10, -20, 0);
    assert_eq!(ack(interrupt(&*m, 2)), [0, 10, (-20i8) as u8]);
    // A press travels with its own report, then the release.
    m.mouse(1, 0, 0, 0);
    m.mouse(0, 0, 0, 0);
    assert_eq!(ack(interrupt(&*m, 3)), [1, 0, 0]);
    assert_eq!(ack(interrupt(&*m, 4)), [0, 0, 0]);
    // Buttons in PS/2 order: right is 2, middle is 4.
    m.mouse(2 | 4, 0, 0, 0);
    assert_eq!(ack(interrupt(&*m, 5)), [6, 0, 0]);
    assert_eq!(control_in(&*m, 0xa1, GET_REPORT, 0x0100, 0, 3).unwrap(), [6, 0, 0]);
    // Boot protocol has no wheel byte: the report stays 3 bytes.
    m.mouse(6, 0, 0, 3);
    assert_eq!(interrupt(&*m, 6), UsbOutcome::Nak);
}

#[test]
fn large_motion_splits_across_reports_without_loss() {
    let m = UsbMouse::new();
    enumerate(&*m, 4);
    m.mouse(0, 300, -200, 0);
    assert_eq!(m.mouse_pending(), 3);
    let mut sum = (0i32, 0i32);
    let mut n = 0;
    while let UsbOutcome::Ack(r) = interrupt(&*m, n) {
        sum.0 += r[1] as i8 as i32;
        sum.1 += r[2] as i8 as i32;
        assert!(r[1] as i8 != -128 && r[2] as i8 != -128);
        n += 1;
    }
    assert_eq!(sum, (300, -200));
    assert_eq!(m.mouse_pending(), 0);
}

#[test]
fn a_mouse_before_configuration_is_not_enabled_and_drops_motion() {
    let m = UsbMouse::new();
    assert!(!m.mouse_enabled());
    m.mouse(1, 5, 5, 0);
    assert_eq!(m.mouse_pending(), 0);
    enumerate(&*m, 1);
    assert_eq!(interrupt(&*m, 1), UsbOutcome::Nak);
}

/// Host input arrives through the PS/2 controller's sinks, so the one
/// interface the GUI, VNC and CI use feeds both kinds of device.
#[test]
fn host_input_reaches_usb_devices_through_the_ps2_controller() {
    use crate::dev::ps2::Ps2Controller;
    let ps2 = Ps2Controller::new(None);
    Device::start(&ps2);
    let (k, m) = (UsbKeyboard::new(), UsbMouse::new());
    ps2.add_sink(k.clone());
    ps2.add_sink(m.clone());
    enumerate(&*k, 1);
    enumerate(&*m, 2);
    assert!(ps2.mouse_enabled(), "the USB mouse counts: the PS/2 mouse is still off");
    ps2.push_kb(KeyCode::KeyA, true);
    ps2.type_char('b');
    ps2.push_mouse_input(1, 3, 4, 0);
    assert_eq!(ps2.mouse_pending(), 2, "a move and a button change wait for the guest");
    assert_eq!(ack(interrupt(&*k, 1)), [0, 0, 4, 0, 0, 0, 0, 0]);
    assert_eq!(ack(interrupt(&*k, 2)), [0, 0, 4, 5, 0, 0, 0, 0]);
    assert_eq!(ack(interrupt(&*m, 1)), [0, 3, 4]);
    assert_eq!(ack(interrupt(&*m, 2)), [1, 0, 0]);
}

// ---- Through the OHCI model ------------------------------------------------

struct Memory(Mutex<Vec<u8>>);
impl Memory {
    fn new() -> Arc<Self> {
        Arc::new(Self(Mutex::new(vec![0xa5; 0x10000])))
    }
    fn words(&self, at: usize, words: &[u32]) {
        for (i, w) in words.iter().enumerate() {
            self.0.lock()[at + i * 4..at + i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
    }
    fn word(&self, at: usize) -> u32 {
        u32::from_le_bytes(self.0.lock()[at..at + 4].try_into().unwrap())
    }
}
impl PciDma for Memory {
    fn dma_read(&self, addr: u64, dst: &mut [u8]) -> PciDmaStatus {
        match self.0.lock().get(addr as usize..addr as usize + dst.len()) {
            Some(s) => {
                dst.copy_from_slice(s);
                PciDmaStatus::Ok
            }
            None => PciDmaStatus::MasterAbort,
        }
    }
    fn dma_write(&self, addr: u64, src: &[u8]) -> PciDmaStatus {
        match self.0.lock().get_mut(addr as usize..addr as usize + src.len()) {
            Some(s) => {
                s.copy_from_slice(src);
                PciDmaStatus::Ok
            }
            None => PciDmaStatus::MasterAbort,
        }
    }
}

/// An OHCI controller with the device on port 0, reset and enabled, with
/// the periodic list holding one interrupt-IN endpoint at `address`.
fn ohci_with(f: Arc<dyn UsbFunction>, address: u8) -> (Arc<Ohci>, Arc<Memory>) {
    let h = Ohci::new();
    let m = Memory::new();
    h.connect_dma(m.clone());
    h.attach(0, f.clone()).unwrap();
    h.start();
    h.bar_write(0, 0x54, 0x100, 4);
    h.bar_write(0, 0x54, 0x10, 4);
    h.advance_frames(10);
    // The port reset returned the device to its default state: enumerate it.
    enumerate(&*f, address);
    m.words(0x1000, &[0; 64]);
    // Interrupt table: every slot points at the endpoint. ED: address, endpoint 1,
    // direction from the TD, low speed, max packet 8; TD queue 0x3000 .. tail 0x3020.
    m.words(0x1000, &[0x2000; 32]);
    m.words(0x2000, &[u32::from(address) | (1 << 7) | (1 << 13) | (8 << 16), 0x3020, 0x3000, 0]);
    // TD: direction IN (2), buffer 0x4000..0x4007, buffer rounding allowed, interrupt delay 0.
    m.words(0x3000, &[0xf000_0000 | (2 << 19) | (1 << 18), 0x4000, 0x3020, 0x4007]);
    h.config().write(1, 6);
    h.bar_write(0, 0x18, 0x1000, 4);
    h.bar_write(0, 0x04, 0x84, 4);
    (h, m)
}

/// A report waiting in the device is carried by an interrupt TD into guest
/// memory, with the TD retired as a success.
#[test]
fn an_interrupt_transfer_returns_a_keyboard_report_through_ohci() {
    let k = UsbKeyboard::new();
    let (h, m) = ohci_with(k.clone(), 5);
    k.key(KeyCode::KeyA, true);
    h.advance_frames(2);
    let report = m.0.lock()[0x4000..0x4009].to_vec();
    // The done queue is published once the delay for a low-priority completion passes.
    h.advance_frames(10);
    assert_eq!(report, vec![0, 0, 4, 0, 0, 0, 0, 0, 0xa5], "the report, and nothing past its end");
    assert_eq!(m.word(0x3000) >> 28, 0, "completed without error");
    assert_eq!(m.word(0x1084) & !15, 0x3000, "the TD is on the done queue");
}

/// With nothing to report the device NAKs and the transfer stays queued.
#[test]
fn an_interrupt_transfer_with_no_report_stays_pending() {
    let m = UsbMouse::new();
    let (h, mem) = ohci_with(m.clone(), 5);
    h.advance_frames(3);
    assert_eq!(mem.word(0x3000) >> 28, 0xf, "not accessed");
    m.mouse(2, 1, 1, 0);
    h.advance_frames(2);
    assert_eq!(&mem.0.lock()[0x4000..0x4003], [0, 1, 1]);
    assert_eq!(mem.word(0x3000) >> 28, 0);
}
