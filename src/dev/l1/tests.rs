// Cases follow tests/qtest/sn1-l1-test.c in jamesbraid/qemu-sgi-origami:
// uart, frames, identity, eeprom, errors and reset.

use super::*;

fn send(l1: &L1, bytes: &[u8]) {
    for &b in bytes {
        assert_eq!(l1.write64(0, b as u64), BUS_OK);
    }
}
fn drain(l1: &L1) -> Vec<u8> {
    let mut out = Vec::new();
    while l1.read64(0x28).data & 1 != 0 {
        out.push(l1.read64(0).data as u8);
    }
    out
}
fn request(l1: &L1, command: u16, args: &[u8], count: u8) -> Vec<u8> {
    let mut p = vec![
        0x10,
        0x0f,
        0xff,
        0xf0,
        6,
        (command >> 8) as u8,
        command as u8,
        count,
    ];
    p.extend_from_slice(args);
    send(l1, &encode(&p));
    drain(l1)
}
fn integer_reply(values: &[u32]) -> Vec<u8> {
    let mut p = vec![0x30, 0, 0, 0, 0, values.len() as u8];
    p.extend(model::ints(values));
    encode(&p)
}

/// The PROM accesses the L1 UART through byte-sized register operations.
#[test]
fn the_l1_uart_frames_bytes_and_prioritizes_interrupts() {
    let l1 = L1::new(Default::default()).unwrap();
    assert_eq!(l1.read64(1).status, BUS_ERR);
    assert_eq!(l1.read8(0).status, BUS_ERR);
    assert_eq!(l1.read32(0).status, BUS_ERR);
    assert_eq!(l1.read64(0x28).data, 0x60);
    send(&l1, &[0x7e, 0x10, 0x0f, 0xff, 0xf0, 6, 0, 0x0e, 0, 0xf9, 0x72, 0x7e]);
    assert_eq!(drain(&l1), [0x7e, 0x30, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0x30, 0x98, 0x7e]);
    let (irq, line) = crate::irq::LevelProbe::line();
    l1.set_irq(line);
    l1.write64(8, 3);
    assert!(irq.level());
    assert_eq!(l1.read64(0x10).data, 2);
    assert!(!irq.level());
    l1.write64(0x18, 0x80);
    l1.write64(0, 42);
    l1.write64(8, 7);
    assert_eq!(l1.read64(0).data, 42);
    assert_eq!(l1.read64(8).data, 7);
    l1.write64(0x18, 3);
    send(&l1, &encode(&[0x10, 0x0f, 0xff, 0xf0, 6, 0, 0x0e, 0]));
    assert_eq!(l1.read64(0x10).data, 4);
    drain(&l1);
    assert_eq!(l1.read64(0x10).data, 2);
    assert!(!irq.level());
}

/// The guest reads controller identity and checks the EEPROM record checksum.
#[test]
fn the_l1_returns_identity_and_validated_eeprom_records() {
    let l1 = L1::new(L1Config { revision: Some([1, 48, 1]), ..Default::default() }).unwrap();
    assert_eq!(request(&l1, 3, &[], 0), integer_reply(&[65, 65]));
    let board = l1.record(1, 2).unwrap();
    assert_eq!(board.iter().fold(0u8, |s, x| s.wrapping_add(*x)), 0);
    for text in [b"IP34".as_slice(), b"EMUN00", b"080069123456"] {
        assert!(board.windows(text.len()).any(|x| x == text));
    }
    let mut expected = vec![0x30, 0, 0, 0, 0, 2];
    expected.extend(model::ints(&[0]));
    expected.push(0x80 | board.len() as u8);
    expected.extend(board);
    assert_eq!(request(&l1, 6, &model::ints(&[1, 2, 0, 116]), 4), encode(&expected));
    for (args, count, error) in [([2, 5, 0, 1], 4, 152), ([1, 5, 0, 1], 4, 151), ([1, 2, 0, 1], 3, 156)] {
        assert_eq!(request(&l1, 6, &model::ints(&args), count), encode(&[0x30, 255, 255, 255, error, 0]));
    }
    l1.power_on();
    assert_eq!(request(&l1, 5, &[], 0), integer_reply(&[1, 48, 1]));
}

/// The L1 parser recovers after malformed frames and preserves escaped console text.
#[test]
fn the_l1_resynchronizes_after_bad_frames_and_preserves_escaped_text() {
    let l1 = L1::new(Default::default()).unwrap();
    send(&l1, &[0x7e, 1, 2, 3, 0x7e]);
    send(&l1, &vec![0x41; 259]);
    send(&l1, &[0x7e]);
    assert!(drain(&l1).is_empty());
    assert_eq!(request(&l1, 0x2000, &[], 0), encode(&[0x30, 255, 255, 255, 155, 0]));
    send(&l1, &encode(&[0x10, 0x4f, 0xff, 0xf0, 6, 0, 3, 0]));
    assert_eq!(drain(&l1), encode(&[0x30, 255, 255, 255, 255, 0, 0, 0, 8]));
    let mut text = vec![1];
    text.extend_from_slice(b"hi~}\r\0");
    assert_eq!(request(&l1, 0x1004, &text, 1), integer_reply(&[]));
    assert_eq!(l1.display_line(), "hi~}?");
    text.truncate(1);
    text.extend_from_slice(b"abcdefghijkl\x01\x7f");
    text.extend(std::iter::repeat_n(b'z', 70));
    text.push(0);
    assert_eq!(request(&l1, 0x1004, &text, 1), integer_reply(&[]));
    assert_eq!(l1.display_line(), format!("abcdefghijkl??{}", "z".repeat(58)));
}

/// The PROM may reset L1 during boot; controller settings and channel configuration must remain available.
#[test]
fn l1_reset_clears_uart_state_but_preserves_settings_and_channels() {
    for channel in [3, 23] {
        let l1 = L1::new(Default::default()).unwrap();
        let (irq, line) = crate::irq::LevelProbe::line();
        l1.set_irq(line);
        for (set, value) in [(0x1001, 0x1234), (0x1006, 42), (0x1008, 2), (0x1007, channel)] {
            request(&l1, set, &model::ints(&[value]), 1);
        }
        request(&l1, 0x2001, &[], 0);
        let reasons = request(&l1, 0xf, &[], 0);
        assert_eq!(reasons, integer_reply(&[3, 0]));
        l1.write64(8, 1);
        l1.send_event(0x8000, 7, "before reset");
        assert!(irq.level());
        let stale = encode(&[0x10, 0x0f, 0xff, 0xf0, 6, 0, 9, 0]);
        send(&l1, &stale[..4]);
        l1.write64(0x18, 0x80);
        l1.write64(0, 42);
        l1.write64(8, 7);
        l1.power_on();
        assert!(!irq.level());
        assert_eq!(l1.read64(0x18).data, 0);
        assert_eq!(l1.read64(8).data, 0);
        assert!(drain(&l1).is_empty());
        l1.write64(0x18, 0x80);
        // Ns16550 resets its divisor latch to 0x000c for 9600 baud.
        assert_eq!(l1.read64(0).data, 0x0c);
        assert_eq!(l1.read64(8).data, 0);
        l1.write64(0x18, 0);
        send(&l1, &stale[4..]);
        assert!(drain(&l1).is_empty());
        for (get, value) in [(1, 0x1234), (9, 42), (0xb, 2)] {
            assert_eq!(request(&l1, get, &[], 0), integer_reply(&[value]));
        }
        assert_eq!(request(&l1, 0xf, &[], 0), reasons);
        l1.write64(8, 1);
        l1.send_event(0x8000, 7, "after reset");
        assert_eq!(irq.level(), channel == 23);
        let emitted = drain(&l1);
        assert_eq!(emitted.is_empty(), channel == 3);
        assert!(!irq.level());
    }
}

/// A local console power-down request requires a second confirmation command.
#[test]
fn interactive_power_down_waits_for_a_second_command() {
    let l1 = Arc::new(L1::new(Default::default()).unwrap());
    let actions = Arc::new(Mutex::new(Vec::new()));
    let observed = actions.clone();
    let weak = Arc::downgrade(&l1);
    l1.set_event_handler(Arc::new(move |event| {
        assert!(!weak.upgrade().unwrap().node_powered());
        observed.lock().push(event);
    }));
    request(&l1, 0x1007, &model::ints(&[23]), 1);
    l1.receive_console(&[0x14]);
    l1.receive_console(b"pwr d\r");
    assert!(l1.node_powered());
    assert!(actions.lock().is_empty());
    let mut warning = vec![0x57, 0, 4, 0x10, 4, 0, 0, 0x80, 0, 2];
    warning.extend(model::ints(&[0]));
    warning.push(1);
    warning.extend_from_slice(b"\r\n001c01 ATTN: Power Down: issue 'pwr d' again to immediately power down.\r\n\0");
    assert_eq!(drain(&l1), encode(&warning));
    l1.power_on();
    l1.receive_console(&[0x14]);
    l1.receive_console(b"pwr d\r");
    assert!(l1.node_powered(), "reset clears the first confirmation");
    drain(&l1);
    l1.receive_console(&[0x14]);
    l1.receive_console(b"power down\r");
    assert!(!l1.node_powered());
    assert_eq!(*actions.lock(), [L1Event::Power(false), L1Event::Shutdown]);
}

/// Protocol power-down actions must finish before the L1 request reply.
#[test]
fn protocol_and_task_three_power_down_commands_are_immediate() {
    let protocol = vec![0x10, 0x0f, 0xff, 0xf0, 6, 0x20, 5, 0];
    let mut task_three = vec![0x10, 0x0f, 0xff, 0xf0, 3, 0, 0, 1, 1];
    task_three.extend_from_slice(b"pwr d\0");
    for (case, packet) in [
        ("protocol command", protocol),
        ("task three console", task_three),
    ] {
        let l1 = L1::new(Default::default()).unwrap();
        let actions = Arc::new(Mutex::new(Vec::new()));
        let observed = actions.clone();
        l1.set_event_handler(Arc::new(move |event| observed.lock().push(event)));
        send(&l1, &encode(&packet));
        assert_eq!(drain(&l1), integer_reply(&[]), "{case}");
        assert!(!l1.node_powered(), "{case}");
        assert_eq!(
            *actions.lock(),
            [L1Event::Power(false), L1Event::Shutdown],
            "{case}"
        );
    }
}

/// A CPU may read the UART in its interrupt callback while the worker updates the line.
#[test]
fn l1_worker_irq_delivery_stays_ordered_across_reentrant_cpu_reads() {
    use std::sync::mpsc;
    struct BlockingIrq {
        entered: mpsc::SyncSender<()>,
        resume: std::sync::Mutex<mpsc::Receiver<()>>,
        blocked: AtomicBool,
        levels: Mutex<Vec<bool>>,
        device: std::sync::Weak<L1>,
    }
    impl crate::irq::InterruptController for BlockingIrq {
        fn set_line(&self, _: u32, level: bool) {
            if level && !self.blocked.swap(true, Ordering::AcqRel) {
                self.entered.send(()).unwrap();
                self.resume.lock().unwrap().recv().unwrap();
                // A machine interrupt handler may inspect the UART immediately.
                assert_eq!(self.device.upgrade().unwrap().read64(0x28).data, 0x60);
            }
            self.levels.lock().push(level);
        }
    }
    let l1 = Arc::new(L1::new(Default::default()).unwrap());
    let backend = Arc::new(crate::dev::serial_host::CiSerialBackend::new());
    let (entered_tx, entered_rx) = mpsc::sync_channel(1);
    let (resume_tx, resume_rx) = mpsc::sync_channel(1);
    let irq = Arc::new(BlockingIrq {
        entered: entered_tx,
        resume: std::sync::Mutex::new(resume_rx),
        blocked: AtomicBool::new(false),
        levels: Mutex::new(Vec::new()),
        device: Arc::downgrade(&l1),
    });
    l1.set_irq(IrqLine::new(irq.clone(), 0));
    l1.set_serial_backend(backend.clone());
    l1.write64(8, 1);
    l1.start();
    backend.push_host(b"x");
    let entered = entered_rx.recv_timeout(std::time::Duration::from_secs(2));
    if entered.is_ok() {
        l1.write64(0x10, 2);
    }
    resume_tx.send(()).unwrap();
    l1.stop();
    entered.unwrap();
    assert_eq!(*irq.levels.lock(), [false, true, false]);
}

/// Concurrent worker output and platform actions must preserve their order.
#[test]
fn concurrent_l1_outputs_and_actions_are_delivered_in_order() {
    use std::sync::mpsc;
    struct BlockingBackend {
        entered: mpsc::SyncSender<()>,
        resume: std::sync::Mutex<mpsc::Receiver<()>>,
        output: Mutex<Vec<u8>>,
    }
    impl SerialBackend for BlockingBackend {
        fn send_byte(&self, byte: u8) {
            if byte == b'A' {
                self.entered.send(()).unwrap();
                self.resume.lock().unwrap().recv().unwrap();
            }
            self.output.lock().push(byte);
        }
        fn recv_byte(&self) -> std::io::Result<u8> {
            Err(std::io::ErrorKind::WouldBlock.into())
        }
    }
    let l1 = Arc::new(L1::new(Default::default()).unwrap());
    let (entered_tx, entered_rx) = mpsc::sync_channel(1);
    let (resume_tx, resume_rx) = mpsc::sync_channel(1);
    let backend = Arc::new(BlockingBackend {
        entered: entered_tx,
        resume: std::sync::Mutex::new(resume_rx),
        output: Mutex::new(Vec::new()),
    });
    l1.set_serial_backend(backend.clone());
    let actions = Arc::new(Mutex::new(Vec::new()));
    let captured = actions.clone();
    let weak = Arc::downgrade(&l1);
    l1.set_event_handler(Arc::new(move |action| {
        // Reentrant register reads must not recursively dispatch notifications.
        assert_eq!(weak.upgrade().unwrap().read64(0x28).data, 0x60);
        captured.lock().push(action);
    }));
    {
        let mut s = l1.state.lock();
        s.output.extend(b"AB");
        s.actions.push(L1Event::Reset);
    }
    let worker_device = l1.clone();
    let worker = std::thread::spawn(move || worker_device.flush());
    let entered = entered_rx.recv_timeout(std::time::Duration::from_secs(2));
    if entered.is_ok() {
        let mut s = l1.state.lock();
        s.output.extend(b"CD");
        s.actions.push(L1Event::Nmi);
        drop(s);
        l1.flush();
    }
    resume_tx.send(()).unwrap();
    worker.join().unwrap();
    entered.unwrap();
    assert_eq!(*backend.output.lock(), b"ABCD");
    assert_eq!(*actions.lock(), [L1Event::Reset, L1Event::Nmi]);
}
