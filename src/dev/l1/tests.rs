// Cases follow tests/qtest/sn1-l1-test.c in jamesbraid/qemu-sgi-origami:
// uart, frames, identity, eeprom, errors and reset, plus the port-speed and
// router brick-reply cases (test_port_speed, test_irouter_*).

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
    let l1 = L1::new(Default::default()).unwrap();
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

/// A configured CPU configuration goes into component 0's board area, after
/// the identity fields, as format 2 with the one-byte selector 3, the Hub mode
/// bits, Config0 to Config3, the Hub and CPU MHz and three versions, the
/// unknown ones as empty fields; the I/O board's area and the default record
/// carry none. The layout is QEMU's (`l1_build_board_area`).
#[test]
fn the_node_board_record_carries_a_configured_cpu_configuration() {
    let plain = L1::new(Default::default()).unwrap().record(0, 2).unwrap();
    let cpu = CpuContext { mode_word: 0x589b_ad85, cpu_mhz: 700, hub_mhz: 200 };
    let l1 = L1::new(L1Config { cpu_context: Some(cpu), ..Default::default() }).unwrap();
    let area = l1.record(0, 2).unwrap();
    assert_eq!(area.iter().fold(0u8, |s, x| s.wrapping_add(*x)), 0);
    assert_eq!((area.len() % 8, usize::from(area[1]) * 8), (0, area.len()));
    assert_eq!((plain[0], area[0]), (0, 2));
    // The identity fields end with the two-character text field "00".
    let identity = plain.windows(3).position(|w| w == [0xc2, b'0', b'0']).unwrap() + 3;
    assert_eq!(&area[2..identity], &plain[2..identity], "the identity fields are unchanged");
    let mut fields = vec![1, 3, 0, 4, 0x58, 0x9b, 0xad, 0x85, 0, 0, 0, 4, 0, 0, 0, 200, 4, 0, 0, 0x02, 0xbc, 0, 0, 0];
    for _ in 0..3 {
        fields.extend_from_slice(&[4, 0, 0, 0, 0]);
    }
    fields.push(0xc1);
    assert_eq!(&area[identity..identity + fields.len()], fields);
    assert_eq!(l1.record(1, 2), L1::new(Default::default()).unwrap().record(1, 2), "the I/O board's area");
}

/// A board that names its node board gets that identity in component 0's
/// board area, empty fields included, and no I/O board record; with none
/// named the Fuel's placeholder answers in both, as before.
#[test]
fn a_named_node_board_answers_in_component_0_alone() {
    let ip53 = BoardIdentity { name: "IP53_4CPU", part: "030-1868-001", revision: "-C" };
    let l1 = L1::new(L1Config { node_board: Some(ip53), ..Default::default() }).unwrap();
    let area = l1.record(0, 2).unwrap();
    assert_eq!(area.iter().fold(0u8, |s, x| s.wrapping_add(*x)), 0);
    for text in [b"IP53_4CPU".as_slice(), b"030-1868-001", b"-C"] {
        assert!(area.windows(text.len()).any(|x| x == text), "{}", String::from_utf8_lossy(text));
    }
    assert!(!area.windows(4).any(|x| x == b"IP34"));
    assert_eq!(l1.record(1, 2), None, "no I/O board record");
    assert_eq!(l1.record(1, 1), None);
    let empty = BoardIdentity { name: "", part: "", revision: "" };
    let area = L1::new(L1Config { node_board: Some(empty), ..Default::default() }).unwrap().record(0, 2).unwrap();
    // SYNTHETIC, then an empty name, the serial, an empty part, the FRU file
    // byte and an empty revision.
    let at = area.windows(10).position(|x| x == b"\xc9SYNTHETIC").unwrap() + 10;
    assert_eq!(&area[at..at + 11], b"\xc0\xc6EMUN00\xc0\x00\xc0");
    let fuel = L1::new(Default::default()).unwrap();
    assert_eq!(fuel.record(0, 2), L1::new(L1Config { node_board: None, ..Default::default() }).unwrap().record(0, 2));
    assert!(fuel.record(1, 2).is_some());
}

/// A successful reply carrying one text field.
fn text_reply(text: &str) -> Vec<u8> {
    let mut p = vec![0x30, 0, 0, 0, 0, 1];
    p.push(1);
    p.extend_from_slice(text.as_bytes());
    p.push(0);
    encode(&p)
}

/// With no configuration the controller reports the labelled model defaults:
/// firmware 1.48.1 and the synthetic brick serial.
#[test]
fn the_l1_reports_the_default_revision_and_brick_serial() {
    let l1 = L1::new(Default::default()).unwrap();
    l1.power_on();
    assert_eq!(request(&l1, 5, &[], 0), integer_reply(&[1, 48, 1]));
    assert_eq!(request(&l1, 4, &[], 0), text_reply("EMUN00"));
}

/// Both identity values can be replaced.
#[test]
fn the_l1_revision_and_brick_serial_can_be_overridden() {
    let l1 = L1::new(L1Config {
        revision: Some(parse_revision("2.13.40").unwrap()),
        brick_serial: Some("EMUN07".into()),
        ..Default::default()
    })
    .unwrap();
    l1.power_on();
    assert_eq!(request(&l1, 5, &[], 0), integer_reply(&[2, 13, 40]));
    assert_eq!(request(&l1, 4, &[], 0), text_reply("EMUN07"));
    assert_eq!(
        L1::new(L1Config { brick_serial: Some("x".repeat(38)), ..Default::default() }).err().as_deref(),
        Some("L1 brick serial must fit a 37-byte text field")
    );
}

/// The revision override is exactly three decimal numbers.
#[test]
fn the_l1_revision_text_is_three_decimal_parts() {
    assert_eq!(parse_revision("1.48.1"), Ok([1, 48, 1]));
    for bad in ["", "1", "1.48", "1.48.1.0", "1..1", "a.b.c", "1.48.-1", "1.48.+1", " 1.48.1", "1.48.99999999999"] {
        assert!(parse_revision(bad).is_err(), "{bad:?}");
    }
}

/// Request 0x0008 is the configured station address as twelve uppercase
/// hexadecimal digits in address order, the text the PROM turns back into the
/// Ethernet address.
#[test]
fn the_l1_system_serial_is_the_station_address_in_hex() {
    let l1 = L1::new(Default::default()).unwrap();
    l1.power_on();
    assert_eq!(request(&l1, 8, &[], 0), text_reply("080069123456"));
    let l1 = L1::new(L1Config { mac: [0x00, 0xA0, 0xc9, 0x1f, 0x02, 0xbe], ..Default::default() }).unwrap();
    l1.power_on();
    assert_eq!(request(&l1, 8, &[], 0), text_reply("00A0C91F02BE"));
}

/// A configured system serial replaces the address text in request 0x0008
/// and nothing else: the brick serial and the board serial in the node
/// board's record, which the PROM ties its NVRAM to, stay the same.
#[test]
fn a_system_serial_changes_only_the_system_serial_reply() {
    let plain = L1::new(Default::default()).unwrap();
    let set = L1::new(L1Config { system_serial: Some("LEMU200420".into()), ..Default::default() }).unwrap();
    for l1 in [&plain, &set] {
        l1.power_on();
    }
    assert_eq!(request(&set, 8, &[], 0), text_reply("LEMU200420"));
    assert_eq!(request(&plain, 8, &[], 0), text_reply("080069123456"));
    assert_eq!(request(&set, 4, &[], 0), request(&plain, 4, &[], 0));
    assert_eq!(set.state.lock().records, plain.state.lock().records);
    assert_eq!(
        L1::new(L1Config { system_serial: Some("x".repeat(38)), ..Default::default() }).err().as_deref(),
        Some("L1 system serial must fit a 37-byte text field")
    );
}

/// The L1 parser recovers after malformed frames.
#[test]
fn the_l1_resynchronizes_after_bad_frames() {
    let l1 = L1::new(Default::default()).unwrap();
    send(&l1, &[0x7e, 1, 2, 3, 0x7e]);
    send(&l1, &vec![0x41; 259]);
    send(&l1, &[0x7e]);
    assert!(drain(&l1).is_empty());
    assert_eq!(request(&l1, 0x2000, &[], 0), encode(&[0x30, 255, 255, 255, 155, 0]));
    send(&l1, &encode(&[0x10, 0x4f, 0xff, 0xf0, 6, 0, 3, 0]));
    assert_eq!(drain(&l1), encode(&[0x30, 255, 255, 255, 255, 0, 0, 0, 8]));
    // A good frame after them is answered.
    assert_eq!(request(&l1, 9, &[], 0), integer_reply(&[0]));
}

/// The PROM may reset the L1 during boot: the UART and parser restart, and
/// an event subchannel the operating system set (23) survives while the
/// PROM's own (3) does not.
#[test]
fn l1_reset_clears_uart_state_but_keeps_the_os_event_channel() {
    for channel in [3, 23] {
        let l1 = L1::new(Default::default()).unwrap();
        let (irq, line) = crate::irq::LevelProbe::line();
        l1.set_irq(line);
        request(&l1, 0x1007, &model::ints(&[channel]), 1);
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
        l1.write64(8, 1);
        l1.send_event(0x8000, 7, "after reset");
        assert_eq!(irq.level(), channel == 23);
        let emitted = drain(&l1);
        assert_eq!(emitted.is_empty(), channel == 3);
        assert!(!irq.level());
    }
}

/// Requests neither the stock PROM nor IRIX sends are refused as unknown
/// (minus 101), with no other effect.
#[test]
fn requests_no_guest_sends_are_refused() {
    let l1 = L1::new(Default::default()).unwrap();
    for cmd in [2, 7, 0xf, 0x1001, 0x1004, 0x4002] {
        assert_eq!(request(&l1, cmd, &[], 0), encode(&[0x30, 255, 255, 255, 155, 0]), "{cmd:#x}");
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
        s.actions.push(L1Event::Shutdown);
        drop(s);
        l1.flush();
    }
    resume_tx.send(()).unwrap();
    worker.join().unwrap();
    entered.unwrap();
    assert_eq!(*backend.output.lock(), b"ABCD");
    assert_eq!(*actions.lock(), [L1Event::Reset, L1Event::Shutdown]);
}

/// Request 0xa returns two integers; the first is a PROM display selector.
/// Ported from test_port_speed in tests/qtest/sn1-l1-test.c.
#[test]
fn the_port_speed_reply_carries_the_configured_first_integer() {
    let l1 = L1::new(Default::default()).unwrap();
    assert_eq!(request(&l1, 0xa, &[], 0), integer_reply(&[0, 0]));
    let l1 = L1::new(L1Config { port_speed_first: 1, ..Default::default() }).unwrap();
    assert_eq!(request(&l1, 0xa, &[], 0), integer_reply(&[1, 0]));
    l1.power_on();
    drain(&l1);
    assert_eq!(request(&l1, 0xa, &[], 0), integer_reply(&[1, 0]));
    assert!(L1::new(L1Config { port_speed_first: 2, ..Default::default() }).is_err());
}

const BRICK: [u8; 53] = [
    0x10, 0, 0, 0, 0, 0x30, 1, 1, 0, 0x2b, 0, 0x2b, 2, 0, 0, 0x81, 0, 0, 0x0f, 0xff, 0xf0, 3, 0x2f,
    0xff, 0xf0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x0a, 1, 0, 7, 0, 0, 0, 0x34, b'b', b'r',
    b'i', b'c', b'k', b' ', 0,
];

fn router_l1(type_code: u32, rack: u32, bay: u32, serial: Option<&str>) -> L1 {
    L1::new(L1Config {
        router: true,
        type_code,
        rack,
        bay,
        brick_serial: serial.map(str::to_string),
        ..Default::default()
    })
    .unwrap()
}

fn chunk(channel: u8, command: u16, count: u8, length: u16, data: &[u8]) -> Vec<u8> {
    let mut p = vec![channel, 0, 0, 0, 0, (command >> 8) as u8, command as u8, count];
    p.extend_from_slice(&length.to_be_bytes());
    p.extend_from_slice(data);
    p
}

fn send_packet(l1: &L1, packet: &[u8]) {
    send(l1, &encode(packet));
}

/// Collect one reply message from its chunks: the channel, and the inner bytes.
fn router_receive(l1: &L1) -> (u8, Vec<u8>) {
    let wire = drain(l1);
    let mut frames = wire.split(|b| *b == 0x7e).filter(|f| !f.is_empty());
    let mut decode = || -> Vec<u8> {
        let raw = frames.next().expect("a reply frame");
        let mut out = Vec::new();
        let mut it = raw.iter();
        while let Some(&b) = it.next() {
            out.push(if b == 0x7d { it.next().unwrap() ^ 0x20 } else { b });
        }
        out.truncate(out.len() - 2);
        out
    };
    let first = decode();
    assert_eq!(first[1..5], [0; 4]);
    assert_eq!(first[5..8], [0x30, 1, 1]);
    let channel = first[0];
    let mut inner = first[10..].to_vec();
    while inner.len() < usize::from(u16::from_be_bytes([inner[0], inner[1]])) {
        let next = decode();
        assert_eq!(next[0], channel);
        assert_eq!(next[5..10], [0x30, 3, 0, 0, 0]);
        inner.extend_from_slice(&next[10..]);
    }
    assert!(frames.next().is_none());
    (channel, inner)
}

fn expect_brick(l1: &L1, channel: u8, destination: u32, source: u32, text: &str) {
    let (got, inner) = router_receive(l1);
    assert_eq!(got, channel);
    assert_eq!(usize::from(u16::from_be_bytes([inner[0], inner[1]])), inner.len());
    assert_eq!(u32::from_be_bytes(inner[8..12].try_into().unwrap()), destination);
    assert_eq!(u32::from_be_bytes(inner[12..16].try_into().unwrap()), source);
    assert_eq!(&inner[36..], [text.as_bytes(), &[0]].concat());
}

const DEFAULT_TEXT: &str = "rack: 001, slot: 04, partition: none, type: Chimera Server \
    [?MB flash], serial:(null), source: OTHER\n\n";

/// Ported from test_irouter_brick_exact in tests/qtest/sn1-l1-test.c.
#[test]
fn the_router_brick_reply_matches_the_captured_layout() {
    let l1 = router_l1(0x37, 1, 4, None);
    send_packet(&l1, &BRICK);
    let mut inner = vec![
        0, 0x8b, 2, 0, 0, 0x81, 1, 0, 0, 0x04, 0x42, 0x15, 0, 0x04, 0x40, 3, 2, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0x0a, 1, 0, 0x67, 0, 0, 0, 0x34,
    ];
    inner.extend_from_slice(DEFAULT_TEXT.as_bytes());
    inner.push(0);
    assert_eq!(inner.len(), 139);
    let mut first = vec![0x30, 0, 0, 0, 0, 0x30, 1, 1, 0, 139];
    first.extend_from_slice(&inner[..113]);
    let mut rest = vec![0x30, 0, 0, 0, 0, 0x30, 3, 0, 0, 0];
    rest.extend_from_slice(&inner[113..]);
    let mut expected = encode(&first);
    expected.extend(encode(&rest));
    assert_eq!(drain(&l1), expected);
}

/// Subchannels are answered in place, and a request split across chunks
/// completes on the last continuation. Ported from
/// test_irouter_fragment_channel_transport (the single-transport part).
#[test]
fn the_router_reassembles_chunks_and_answers_on_the_request_subchannel() {
    let l1 = router_l1(0x37, 1, 4, None);
    let rest = &BRICK[10..];
    send_packet(&l1, &chunk(0x10, 0x3001, 1, 43, &rest[..12]));
    assert!(drain(&l1).is_empty());
    let mut other = BRICK;
    other[0] = 0x11;
    send_packet(&l1, &other);
    expect_brick(&l1, 0x31, 0x0004_4235, 0x0004_4003, DEFAULT_TEXT);
    send_packet(&l1, &chunk(0x10, 0x3003, 0, 0, &rest[12..]));
    expect_brick(&l1, 0x30, 0x0004_4215, 0x0004_4003, DEFAULT_TEXT);
}

/// Ported from test_irouter_reset_and_negative.
#[test]
fn a_reset_drops_a_pending_router_message_and_unsupported_types_stay_silent() {
    let l1 = router_l1(0x37, 1, 4, None);
    let rest = &BRICK[10..];
    send_packet(&l1, &chunk(0x10, 0x3001, 1, 43, &rest[..12]));
    l1.power_on();
    drain(&l1);
    send_packet(&l1, &chunk(0x10, 0x3003, 0, 0, &rest[12..]));
    assert!(drain(&l1).is_empty());
    send_packet(&l1, &BRICK);
    expect_brick(&l1, 0x30, 0x0004_4215, 0x0004_4003, DEFAULT_TEXT);

    let off = L1::new(Default::default()).unwrap();
    send_packet(&off, &BRICK);
    assert_eq!(drain(&off), encode(&[0x30, 255, 255, 255, 255, 0, 0, 0, 8]));

    let l1 = router_l1(0x37, 1, 4, None);
    send_packet(&l1, &BRICK[..7]);
    assert!(drain(&l1).is_empty());
    send_packet(&l1, &chunk(0x10, 0x3001, 1, 43, &rest[..12]));
    send_packet(&l1, &BRICK);
    expect_brick(&l1, 0x30, 0x0004_4215, 0x0004_4003, DEFAULT_TEXT);

    let l1 = router_l1(0x34, 1, 4, None);
    send_packet(&l1, &BRICK);
    send_packet(&l1, &chunk(0x10, 0x3003, 0, 0, rest));
    assert!(drain(&l1).is_empty());

    let l1 = router_l1(0x37, 1, 4, None);
    let mut reset = BRICK;
    reset[46..52].copy_from_slice(b"reset\0");
    send_packet(&l1, &reset);
    assert!(drain(&l1).is_empty());
}

/// A new start replaces a pending message. Ported from
/// test_irouter_replace_pending.
#[test]
fn a_new_router_start_replaces_the_pending_message() {
    let l1 = router_l1(0x37, 1, 4, None);
    let mut old = BRICK;
    old[14] = 0x12;
    old[27] = 0x34;
    send_packet(&l1, &chunk(0x10, 0x3001, 1, 43, &old[10..22]));
    send_packet(&l1, &BRICK);
    expect_brick(&l1, 0x30, 0x0004_4215, 0x0004_4003, DEFAULT_TEXT);
    send_packet(&l1, &chunk(0x10, 0x3003, 0, 0, &old[22..]));
    assert!(drain(&l1).is_empty());
}

/// Messages of more than one fragment have no known descriptor layout.
/// Ported from test_irouter_multifragment_refusal.
#[test]
fn a_multi_fragment_router_message_is_refused_and_later_requests_work() {
    let l1 = router_l1(0x37, 1, 4, None);
    let mut request = BRICK;
    request[15] = 0x82;
    send_packet(&l1, &chunk(0x10, 0x3001, 2, 43, &request[10..]));
    assert!(drain(&l1).is_empty());
    send_packet(&l1, &chunk(0x10, 0x3002, 0, 4, &[0xde, 0xad, 0xbe, 0xef]));
    assert!(drain(&l1).is_empty());
    request[0] = 0x11;
    request[15] = 0x81;
    send_packet(&l1, &request);
    expect_brick(&l1, 0x31, 0x0004_4235, 0x0004_4003, DEFAULT_TEXT);
}

/// Ported from test_irouter_identity_overrides, less its partition cases:
/// this L1 is never partitioned. The type 0x39 case is the Tezro identity.
#[test]
fn the_router_reply_reports_rack_slot_type_and_serial() {
    let l1 = router_l1(0x39, 2, 5, Some("SERIAL-X"));
    send_packet(&l1, &BRICK);
    expect_brick(
        &l1,
        0x30,
        0x0008_5215,
        0x0008_5003,
        "rack: 002, slot: 05, partition: none, type: Chimera WS [?MB flash], \
         serial:SERIAL-X, source: OTHER\n\n",
    );
    let l1 = router_l1(0x51, 3, 2, Some(""));
    send_packet(&l1, &BRICK);
    expect_brick(
        &l1,
        0x30,
        0x000c_2215,
        0x000c_2003,
        "rack: 003, slot: 02, partition: none, type: Rackmount WS [?MB flash], \
         serial:, source: OTHER\n\n",
    );
}
