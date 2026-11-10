//! Tests of an Origin 2000 module through each node's adapter, on a hand-set
//! clock. Replies are checked against literal text.

use super::*;
use crate::dev::elsc::tests::{byte_sum, frame, text, Aux, TestClock};
use crate::dev::pcf8584::{ctl, sts};

const DATA: u64 = 0;
const CTRL: u64 = 8;
const MSC: u8 = 0x08;
const INVALID: u8 = 0x7f;
const SECOND: u64 = 1000;
const GRANT: u8 = 0xe0;

struct Rig {
    clock: TestClock,
    module: Arc<Origin2000Module>,
    aux: Arc<Aux>,
}

impl Rig {
    fn new() -> Self {
        Self::with(ModuleConfig { can_reset: true, can_nmi: true, can_power: true, ..Default::default() })
    }

    fn with(cfg: ModuleConfig) -> Self {
        let clock = TestClock::default();
        let nvram = Arc::new(ElscNvram::origin2000(cfg.module_number, None).unwrap());
        let module = Origin2000Module::new(cfg, nvram, clock.clock());
        let aux = Arc::new(Aux::default());
        module.elsc().set_serial_backend(aux.clone());
        module.power_on();
        let rig = Self { clock, module, aux };
        rig.init_adapters();
        rig
    }

    /// What the PROM does to each node's adapter after a reset.
    fn init_adapters(&self) {
        for node in 0..ADAPTERS {
            self.w(node, CTRL, ctl::PIN | ctl::ES1);
            self.w(node, DATA, 0x1c);
            self.w(node, CTRL, ctl::PIN);
            self.w(node, DATA, 0x09 + node as u8);
            self.w(node, CTRL, ctl::PIN | ctl::ESO);
        }
    }

    fn w(&self, node: usize, off: u64, v: u8) {
        self.module.adapter(node).write_register(off, v);
    }

    fn r(&self, node: usize, off: u64) -> u8 {
        self.module.adapter(node).read_register(off)
    }

    fn step_ms(&self, ms: u64) {
        self.clock.advance_ms(ms);
    }

    fn events(&self) -> Vec<ModuleOutput> {
        self.module.take_events()
    }

    fn info(&self) -> String {
        self.module.elsc().describe()
    }

    // ---- transfers -------------------------------------------------------

    fn start(&self, node: usize, addr: u8, recv: bool) -> bool {
        self.w(node, CTRL, ctl::PIN | ctl::ESO);
        self.w(node, DATA, (addr << 1) | recv as u8);
        self.w(node, CTRL, ctl::PIN | ctl::ESO | ctl::STA | ctl::ACK);
        self.r(node, CTRL) & sts::LRB == 0
    }

    fn restart(&self, node: usize, addr: u8, recv: bool) -> bool {
        self.w(node, CTRL, ctl::ESO | ctl::STA | ctl::ACK);
        self.w(node, DATA, (addr << 1) | recv as u8);
        self.r(node, CTRL) & sts::LRB == 0
    }

    fn stop(&self, node: usize) {
        self.w(node, CTRL, ctl::PIN | ctl::ESO | ctl::STO);
    }

    /// One controller command from `node`: the combined transaction and the
    /// closing start to the invalid address, then a stop.
    fn msc(&self, node: usize, request: &[u8]) -> Vec<u8> {
        assert!(self.start(node, MSC, false));
        for b in request {
            self.w(node, DATA, *b);
        }
        assert!(self.restart(node, MSC, true));
        let _ = self.r(node, DATA);
        let mut resp = vec![self.r(node, DATA)];
        let total = resp[0] as usize + 2;
        for i in 1..total {
            if i == total - 1 {
                self.w(node, CTRL, ctl::ESO);
            }
            resp.push(self.r(node, DATA));
        }
        assert!(!self.restart(node, INVALID, false));
        self.stop(node);
        assert_eq!(byte_sum(&resp), 0);
        resp
    }

    fn cmd(&self, node: usize, line: &str) -> String {
        text(&self.msc(node, &frame(line.as_bytes())))
    }

    #[track_caller]
    fn assert_cmd(&self, node: usize, line: &str, want: &str) {
        assert_eq!(self.cmd(node, line), want, "node {node}: {line:?}");
    }

    fn nvram_read(&self, node: usize, addr: u16) -> u8 {
        let a7 = 0x50 + (addr >> 8) as u8;
        assert!(self.start(node, a7, false));
        self.w(node, DATA, addr as u8);
        assert!(self.restart(node, a7, true));
        let _ = self.r(node, DATA);
        self.w(node, CTRL, ctl::ESO);
        let v = self.r(node, DATA);
        assert!(!self.restart(node, INVALID, false));
        self.stop(node);
        v
    }

    fn nvram_write(&self, node: usize, addr: u16, data: &[u8]) {
        let a7 = 0x50 + (addr >> 8) as u8;
        assert!(self.start(node, a7, false));
        self.w(node, DATA, addr as u8);
        for b in data {
            self.w(node, DATA, *b);
            assert_eq!(self.r(node, CTRL) & sts::LRB, 0);
        }
        self.stop(node);
    }

    fn ust_read(&self, node: usize) -> u8 {
        assert!(self.start(node, UST_CONTROL_ADDR, true));
        let _ = self.r(node, DATA);
        self.w(node, CTRL, ctl::ESO);
        let v = self.r(node, DATA);
        self.stop(node);
        v
    }

    fn ust_write(&self, node: usize, v: u8) {
        assert!(self.start(node, UST_CONTROL_ADDR, false));
        self.w(node, DATA, v);
        self.stop(node);
    }

    /// Step until node 0's data register shows the grant `want`.
    #[track_caller]
    fn wait_for_grant(&self, want: u8) {
        for _ in 0..200 {
            if self.r(0, DATA) == want {
                return;
            }
            self.clock.advance_ns(100_000);
        }
        panic!("grant {want:#04x} never published");
    }

    // ---- one-wire --------------------------------------------------------

    fn slot(&self, node: usize, pulse: u32, sample: u32, delay: u32) -> bool {
        let resp = self.msc(node, &frame(format!("nic {pulse:x} {sample:x} {delay:x}").as_bytes()));
        assert_eq!(&resp[1..4], b"ok ");
        resp[4] == b'1'
    }

    fn ow_reset(&self, node: usize) -> bool {
        !self.slot(node, 520, 65, 500)
    }

    fn ow_write_byte(&self, node: usize, v: u8) {
        for bit in 0..8 {
            if v & (1 << bit) != 0 {
                self.slot(node, 6, 110, 0);
            } else {
                self.slot(node, 80, 30, 0);
            }
        }
    }

    fn ow_read_byte(&self, node: usize) -> u8 {
        (0..8).fold(0, |v, bit| v | (self.slot(node, 6, 13, 100) as u8) << bit)
    }

    fn ow_start_read(&self, node: usize, rom: &[u8; 8], command: u8, address: u16) {
        assert!(self.ow_reset(node));
        self.ow_write_byte(node, 0x55);
        for b in rom {
            self.ow_write_byte(node, *b);
        }
        self.ow_write_byte(node, command);
        self.ow_write_byte(node, address as u8);
        self.ow_write_byte(node, (address >> 8) as u8);
    }
}

fn power_events(events: &[ModuleOutput]) -> Vec<bool> {
    events
        .iter()
        .filter_map(|e| if let ModuleOutput::Power(on) = e { Some(*on) } else { None })
        .collect()
}

// ---- adapters, controller, store --------------------------------------------

#[test]
fn the_windows_are_separate_adapters() {
    let rig = Rig::new();
    for node in 0..ADAPTERS {
        rig.w(node, CTRL, ctl::PIN);
        assert_eq!(rig.r(node, DATA), 0x09 + node as u8);
        rig.w(node, CTRL, ctl::PIN | ctl::ESO);
    }
}

/// Four per-node controllers would answer `ver` just as happily, so the
/// discriminator is the test-and-set cell: taken from one node, every other
/// node finds it held.
#[test]
fn one_controller_answers_every_node() {
    let rig = Rig::new();
    for node in 0..ADAPTERS {
        assert_eq!(rig.cmd(node, "ver"), "ok VER 3.2 ");
    }
    rig.assert_cmd(0, "tas A", "ok 0");
    for node in 1..ADAPTERS {
        rig.assert_cmd(node, "tas B", "ok A");
    }
}

#[test]
fn the_nvram_is_shared_by_every_node() {
    let rig = Rig::new();
    for (i, offset) in [0x000u16, 0x123, 0x400].into_iter().enumerate() {
        let value = 0xa0 + i as u8;
        rig.nvram_write(i, offset, &[value]);
        assert_eq!(rig.nvram_read(ADAPTERS - 1, offset), value);
    }
}

#[test]
fn a_new_store_holds_the_marker_the_password_and_module_one() {
    const BLOCK: [u8; 16] = [0x37, b'n', b'o', b'n', b'e', 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0];
    let rig = Rig::new();
    for (i, want) in BLOCK.iter().enumerate() {
        assert_eq!(rig.nvram_read(i % ADAPTERS, 0x700 + i as u16), *want, "byte {i}");
    }
    assert_eq!(rig.nvram_read(0, 0), 0);
}

#[test]
fn the_module_number_comes_from_the_configuration() {
    let rig = Rig::with(ModuleConfig { module_number: 6, ..Default::default() });
    assert_eq!(rig.nvram_read(2, 0x708), 6);
}

#[test]
fn a_twelve_byte_block_write_lands_in_one_transaction() {
    const BLOCK: [u8; 12] = [0x37, b'w', b'x', b'y', b'z', 0x15, 0x16, 0x01, 0x02, 0x03, 0x04, 0x05];
    let rig = Rig::new();
    rig.nvram_write(0, 0x700, &BLOCK);
    for (i, want) in BLOCK.iter().enumerate() {
        assert_eq!(rig.nvram_read(3, 0x700 + i as u16), *want);
    }
    assert_eq!(rig.nvram_read(3, 0x70c), 0);
}

#[test]
fn the_ust_control_port_reads_all_ones_then_the_last_byte_written() {
    let rig = Rig::new();
    assert_eq!(rig.ust_read(1), 0xff);
    rig.ust_write(1, 0x3f);
    assert_eq!(rig.ust_read(1), 0x3f);
    assert_eq!(rig.ust_read(0), 0x3f);
}

#[test]
fn an_open_transfer_leaves_every_other_node_seeing_a_busy_bus() {
    let rig = Rig::new();
    assert_ne!(rig.r(1, CTRL) & sts::BB, 0);
    assert!(rig.start(0, 0x50, false));
    for node in 1..ADAPTERS {
        assert_eq!(rig.r(node, CTRL) & sts::BB, 0, "node {node}");
    }
    rig.stop(0);
    for node in 1..ADAPTERS {
        assert_ne!(rig.r(node, CTRL) & sts::BB, 0, "node {node}");
    }
}

/// One adapter's open transfer must survive another adapter addressing the
/// same part. The part keeps a single address counter, so a second adapter
/// reaching it would move the counter the first is about to read from.
#[test]
fn a_second_adapter_waits_for_the_first_and_does_not_move_its_counter() {
    let rig = Rig::new();
    // Node 0 selects byte 0x701 of the controller's block...
    assert!(rig.start(0, 0x57, false));
    rig.w(0, DATA, 0x01);
    // ...node 1 tries to read from the same block meanwhile: its START waits
    // (the bus reads busy, PIN stays inactive)...
    rig.w(1, CTRL, ctl::PIN | ctl::ESO);
    rig.w(1, DATA, 0x57 << 1);
    rig.w(1, CTRL, ctl::PIN | ctl::ESO | ctl::STA | ctl::ACK);
    assert_ne!(rig.r(1, CTRL) & sts::PIN, 0);
    // ...and node 0 still reads its own byte.
    assert!(rig.restart(0, 0x57, true));
    let _ = rig.r(0, DATA);
    rig.w(0, CTRL, ctl::ESO);
    assert_eq!(rig.r(0, DATA), b'n');
    assert!(!rig.restart(0, INVALID, false));
    rig.stop(0);
    // Once node 0 stopped, node 1's START runs.
    assert_eq!(rig.r(1, CTRL) & sts::PIN, 0);
    assert_eq!(rig.r(1, CTRL) & sts::LRB, 0);
    rig.stop(1);
}

#[test]
fn a_waiting_start_is_withdrawn_by_a_stop_or_by_clearing_eso() {
    let rig = Rig::new();
    assert!(rig.start(0, 0x57, false));
    for withdraw in [ctl::PIN | ctl::ESO | ctl::STO, ctl::PIN] {
        rig.w(1, CTRL, ctl::PIN | ctl::ESO);
        rig.w(1, DATA, 0x57 << 1);
        rig.w(1, CTRL, ctl::PIN | ctl::ESO | ctl::STA | ctl::ACK);
        rig.w(1, CTRL, withdraw);
    }
    rig.stop(0);
    // Nothing started: node 1 is idle and the bus is free.
    rig.w(1, CTRL, ctl::PIN | ctl::ESO);
    assert_eq!(rig.r(1, CTRL), sts::PIN | sts::BB);
}

// ---- grants ------------------------------------------------------------------

#[test]
fn every_adapter_sees_the_eight_code_rotation() {
    let rig = Rig::new();
    for n in 1..=24u64 {
        rig.step_ms(1);
        let want = GRANT | ((n - 1) % 8) as u8;
        for node in 0..ADAPTERS {
            assert_eq!(rig.r(node, DATA), want, "tick {n} node {node}");
        }
    }
}

#[test]
fn a_command_in_a_processors_turn_leaves_the_rotation_running() {
    let rig = Rig::new();
    for processor in 0..8u8 {
        rig.wait_for_grant(GRANT | processor);
        let node = (processor / 2) as usize;
        assert_eq!(rig.cmd(node, "ver"), "ok VER 3.2 ");
        assert_eq!(rig.r(node, CTRL) & (sts::BER | sts::LAB), 0);
    }
    // A reader that starts nothing does not hold the ring either.
    rig.wait_for_grant(GRANT);
    rig.clock.advance_ns(1_500_000);
    for node in 0..ADAPTERS {
        assert_ne!(rig.r(node, CTRL) & sts::BB, 0);
        assert_eq!(rig.r(node, DATA), GRANT | 1);
    }
}

#[test]
fn a_long_idle_gap_keeps_the_ring_on_its_grid() {
    let rig = Rig::new();
    rig.step_ms(1);
    assert_eq!(rig.r(0, DATA), GRANT);
    // 80 000 ticks later is a whole number of revolutions.
    rig.step_ms(80_000);
    assert_eq!(rig.r(0, DATA), GRANT);
    rig.step_ms(1);
    assert_eq!(rig.r(0, DATA), GRANT | 1);
}

#[test]
fn a_bus_held_for_five_seconds_has_the_controller_reinitialised() {
    let rig = Rig::new();
    // Leave a transfer open with the controller selected and a response
    // pending.
    assert!(rig.start(0, MSC, false));
    for b in frame(b"ver") {
        rig.w(0, DATA, b);
    }
    assert!(rig.restart(0, MSC, true));
    // Each tick finds the bus busy; the controller is addressed, so these
    // are not published. 5 s of them reinitialise its protocol queues.
    rig.step_ms(4_000);
    assert!(rig.info().contains("pending response"));
    assert!(!rig.info().contains("pending response: none"));
    rig.step_ms(1_100);
    rig.r(1, CTRL);
    assert!(rig.info().contains("pending response: none"));
}

// ---- the Origin 2000 build -------------------------------------------------

#[test]
fn the_origin_2000_display_takes_dsc_and_dsp() {
    let rig = Rig::new();
    assert!(rig.info().contains("display: \"VER 3.2 \""));
    rig.assert_cmd(0, "dsc 1 A", "ok ");
    rig.assert_cmd(0, "dsc 8 Z", "ok ");
    assert!(rig.info().contains("display: \"AER 3.2Z\""));
    for refused in ["dsc 0 X", "dsc 9 X", "dsc 4  ", "dsc", "dsp"] {
        rig.assert_cmd(0, refused, "err arg");
    }
    rig.assert_cmd(0, "dsp testing", "ok ");
    assert!(rig.info().contains("display: \"testingZ\""));
    rig.assert_cmd(0, "dsp ninechars", "ok ");
    assert!(rig.info().contains("display: \"ninechar\""));
    // Spaces before each argument are skipped.
    rig.assert_cmd(0, "dsc  2   Q", "ok ");
    assert!(rig.info().contains("display: \"nQnechar\""));
    // The message's own leading spaces are characters, and it stops at the
    // first unprintable one, leaving the rest of the display.
    rig.assert_cmd(0, "dsp  ab\x01zz", "ok ");
    assert!(rig.info().contains("display: \" abechar\""));
    rig.assert_cmd(0, "ver", "ok VER 3.2 ");
}

#[test]
fn the_origin_2000_build_has_key_and_vlm_and_no_aut() {
    let rig = Rig::new();
    rig.assert_cmd(0, "key", "ok on");
    rig.assert_cmd(1, "vlm 5", "ok n");
    rig.assert_cmd(1, "vlm 5h", "ok ");
    rig.assert_cmd(2, "vlm 5", "ok h");
    rig.assert_cmd(0, "vlm 3 l", "ok ");
    rig.assert_cmd(0, "vlm 3", "ok l");
    rig.assert_cmd(0, "vlm v", "ok n");
    for bad in ["vlm", "vlm x", "vlm 5 q"] {
        rig.assert_cmd(0, bad, "err arg");
    }
    for absent in ["aut", "mod", "pas none"] {
        rig.assert_cmd(0, absent, "err cmd");
    }
    rig.assert_cmd(0, "sla o", "ok ");
    rig.assert_cmd(0, "tmp", "ok n");
    rig.assert_cmd(0, "dsp abc", "ok ");
    rig.assert_cmd(0, "clr", "ok ");
    assert!(rig.info().contains("display: \"        \""));
    rig.assert_cmd(3, "vlm 5", "ok h");
}

#[test]
fn a_status_message_replaces_the_hosts_display_text_until_the_host_writes() {
    let rig = Rig::new();
    let info = rig.info();
    assert!(info.contains("display: \"VER 3.2 \""));
    assert!(!info.contains("status message"));
    rig.assert_cmd(0, "nmi", "ok ");
    assert!(rig.info().contains("status message: NMI"));
    rig.assert_cmd(0, "dsp hi", "ok ");
    assert!(!rig.info().contains("status message"));
}

#[test]
fn the_dip_switches_reach_the_controller_with_no_role_bit() {
    let rig = Rig::with(ModuleConfig { dip_switches: 0x10, ..Default::default() });
    rig.assert_cmd(2, "rsw", "ok ef");
    let rig = Rig::with(ModuleConfig { dip_switches: 0x04, ..Default::default() });
    rig.assert_cmd(2, "rsw", "ok fb");
}

// ---- literal frames ------------------------------------------------------------

#[test]
fn the_origin_2000_build_answers_the_prom_exchanges_with_these_frames() {
    const OK: [u8; 5] = [0x03, 0x6f, 0x6b, 0x20, 0x03];
    let rig = Rig::new();
    let exchanges: &[(&[u8], &[u8])] = &[
        // dsp and eight blanks; dsc 1 e: the display takes them
        (&[0x0b, 0x64, 0x73, 0x70, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0xae], &OK),
        (&[0x07, 0x64, 0x73, 0x63, 0x20, 0x31, 0x20, 0x65, 0xe9], &OK),
        // rsw: ok ff
        (&[0x03, 0x72, 0x73, 0x77, 0xa1], &[0x05, 0x6f, 0x6b, 0x20, 0x66, 0x66, 0x35]),
        // pwr q: ok 1
        (&[0x05, 0x70, 0x77, 0x72, 0x20, 0x71, 0x11], &[0x04, 0x6f, 0x6b, 0x20, 0x31, 0xd1]),
        // ver: ok VER 3.2
        (
            &[0x03, 0x76, 0x65, 0x72, 0xb0],
            &[0x0b, 0x6f, 0x6b, 0x20, 0x56, 0x45, 0x52, 0x20, 0x33, 0x2e, 0x32, 0x20, 0x3b],
        ),
        // dsc 0 e and dsc 9 e: position out of range, err arg
        (
            &[0x07, 0x64, 0x73, 0x63, 0x20, 0x30, 0x20, 0x65, 0xea],
            &[0x07, 0x65, 0x72, 0x72, 0x20, 0x61, 0x72, 0x67, 0x56],
        ),
        (
            &[0x07, 0x64, 0x73, 0x63, 0x20, 0x39, 0x20, 0x65, 0xe1],
            &[0x07, 0x65, 0x72, 0x72, 0x20, 0x61, 0x72, 0x67, 0x56],
        ),
    ];
    for (request, reply) in exchanges {
        assert_eq!(&rig.msc(0, request), reply, "request {request:02x?}");
    }
}

// ---- messages ----------------------------------------------------------------

#[test]
fn a_controller_message_is_announced_in_every_code_and_fetched_once() {
    let rig = Rig::new();
    rig.assert_cmd(0, "pwr d 8", "ok ");
    rig.wait_for_grant(GRANT | 0x08 | 3);
    rig.assert_cmd(0, "get", "pd remote");
    rig.assert_cmd(0, "get", "acp . ");
    // Code 5 was not fetched: it still sees the message.
    rig.wait_for_grant(GRANT | 0x08 | 5);
    // Code 0 fetched its message: it no longer does.
    rig.wait_for_grant(GRANT);
    rig.assert_cmd(0, "clr", "ok ");
}

#[test]
fn a_controller_message_lasts_a_second_there_and_in_the_controller() {
    let rig = Rig::new();
    rig.assert_cmd(0, "pwr d 30", "ok ");
    assert!(rig.info().contains("controller message: \"pd remote\""));
    rig.step_ms(900);
    assert!(rig.info().contains("controller message: \"pd remote\""));
    rig.step_ms(100);
    assert!(!rig.info().contains("controller message"));
    rig.wait_for_grant(GRANT | 6);
}

#[test]
fn queued_console_input_is_announced_to_the_selected_processor() {
    let rig = Rig::new();
    rig.assert_cmd(0, "sel 2a", "ok ");
    for b in b"hi" {
        rig.module.elsc().serial_input(*b);
    }
    // 2a is processor code 2.
    rig.wait_for_grant(GRANT | 0x08 | 2);
    rig.wait_for_grant(GRANT | 3);
}

// ---- power, reset, NMI -------------------------------------------------------

#[test]
fn nmi_pulses_the_modules_output_after_a_second() {
    let rig = Rig::new();
    rig.events();
    rig.assert_cmd(2, "nmi", "ok ");
    rig.step_ms(990);
    assert!(rig.events().is_empty());
    rig.step_ms(20);
    assert_eq!(rig.events(), [ModuleOutput::Nmi(true)]);
    rig.step_ms(100);
    assert_eq!(rig.events(), [ModuleOutput::Nmi(false)]);
    assert_eq!(rig.cmd(0, "ver"), "ok VER 3.2 ");
}

/// The module is due when its controller's reset wait ends and again when the
/// 100 ms pulse does, and not otherwise.
#[test]
fn the_module_reports_when_it_is_next_due() {
    let ms = Duration::from_millis;
    let rig = Rig::new();
    rig.events();
    assert_eq!(rig.module.next_due(), None);
    rig.assert_cmd(1, "rst", "ok ");
    assert_eq!(rig.module.next_due(), Some(ms(1000)));
    rig.step_ms(1000);
    assert_eq!(rig.events(), [ModuleOutput::Reset]);
    assert_eq!(rig.module.next_due(), Some(ms(100)), "the pulse ends");
    rig.step_ms(100);
    assert!(rig.events().is_empty());
    assert_eq!(rig.module.next_due(), None);
}

/// An access that finds the module's outputs already queued leaves them due,
/// whatever it does to the schedule, so the machine takes them at once rather
/// than at the next unrelated visit.
#[test]
fn queued_outputs_leave_the_module_due_now() {
    let rig = Rig::new();
    rig.events();
    rig.assert_cmd(2, "nmi", "ok ");
    rig.step_ms(1100);
    assert_eq!(rig.cmd(0, "ver"), "ok VER 3.2 ");
    assert_eq!(rig.module.next_due(), Some(Duration::ZERO), "the queued outputs are due now");
    assert_eq!(rig.events(), [ModuleOutput::Nmi(true)]);
}

#[test]
fn rst_pulses_the_modules_reset_output() {
    let rig = Rig::new();
    rig.events();
    rig.assert_cmd(1, "rst", "ok ");
    rig.step_ms(1000);
    assert_eq!(rig.events(), [ModuleOutput::Reset]);
    // The pulse is 100 ms, and a destructive command is refused meanwhile.
    rig.assert_cmd(0, "pwr d 0", "err cmd");
    rig.step_ms(100);
    // The fall of the line is not an output: a restart is the rising edge.
    assert!(rig.events().is_empty());
    assert_eq!(rig.cmd(0, "ver"), "ok VER 3.2 ");
}

#[test]
fn power_down_reaches_the_modules_power_line_and_stops_the_grants() {
    let rig = Rig::new();
    rig.events();
    rig.assert_cmd(0, "pwr d 0", "ok ");
    assert_eq!(power_events(&rig.events()), [false]);
    assert!(!rig.module.powered());
    rig.assert_cmd(0, "pwr", "ok d");
    let data = rig.r(0, DATA);
    rig.step_ms(20);
    assert_eq!(rig.r(0, DATA), data);
}

#[test]
fn a_power_cycle_holds_the_module_for_the_interval_and_the_4_4_second_hold() {
    let rig = Rig::new();
    rig.events();
    rig.assert_cmd(0, "pwr c 5", "ok ");
    rig.step_ms(1);
    rig.assert_cmd(1, "pwr", "ok d");
    assert_eq!(power_events(&rig.events()), [false]);
    rig.assert_cmd(2, "rst", "err cmd");
    rig.assert_cmd(3, "clr", "ok ");

    // 5 s off, then 4.4 s with reset held: back at 9.4 s.
    rig.step_ms(9_300 - 1);
    rig.assert_cmd(0, "pwr", "ok d");
    rig.step_ms(100);
    assert_eq!(power_events(&rig.events()), [true]);
    rig.init_adapters();
    rig.assert_cmd(0, "pwr", "ok u");
    rig.assert_cmd(0, "pwr q", "ok 1");
}

#[test]
fn the_controller_powers_the_module_up_inside_the_reset() {
    let rig = Rig::new();
    let info = rig.info();
    assert!(info.contains("module power: on"));
    assert!(info.contains("display: \"VER 3.2 \""));
    assert!(!info.contains("status message"));
    assert!(!info.contains("start-up power-up pending"));
    assert!(rig.module.powered());
    assert_eq!(power_events(&rig.events()), [true]);
    rig.assert_cmd(0, "pwr", "ok u");
    rig.assert_cmd(0, "pwr q", "ok 1");
}

#[test]
fn with_a_start_delay_the_module_starts_off_and_the_controller_powers_it_up() {
    let rig = Rig::with(ModuleConfig {
        can_power: true,
        can_reset: true,
        can_nmi: true,
        start_delay_ms: 6000,
        ..Default::default()
    });
    assert!(!rig.module.powered());
    rig.assert_cmd(0, "pwr", "ok d");
    assert!(rig.module.elsc().power_return_pending());
    rig.assert_cmd(0, "clr", "ok ");
    rig.step_ms(7 * SECOND);
    assert!(rig.info().contains("powering up"));
    rig.step_ms(5 * SECOND);
    rig.module.poll();
    rig.init_adapters();
    assert!(rig.module.powered());
    rig.assert_cmd(0, "pwr", "ok u");
    rig.assert_cmd(0, "pwr q", "ok 1");
}

#[test]
fn an_ac_cycle_powers_the_module_up_again() {
    let rig = Rig::new();
    rig.assert_cmd(0, "pwr c 258", "ok ");
    rig.step_ms(1000);
    rig.assert_cmd(1, "pwr", "ok d");
    rig.module.power_on();
    rig.init_adapters();
    assert!(rig.module.powered());
    rig.assert_cmd(0, "pwr", "ok u");
    // No power-up is left to come from the cycle the reset ended.
    rig.step_ms(300 * SECOND);
    rig.assert_cmd(0, "pwr", "ok u");
}

#[test]
fn a_module_with_no_power_path_refuses_destructive_commands_and_stays_powered() {
    let rig = Rig::with(ModuleConfig::default());
    for refused in ["pwr d 0", "pwr c", "rst"] {
        rig.assert_cmd(0, refused, "err cmd");
    }
    rig.assert_cmd(0, "pwr", "ok u");
    assert!(rig.module.powered());
}

#[test]
fn the_operators_serial_line_runs_when_the_scheduler_gives_it_a_turn() {
    let rig = Rig::new();
    rig.aux.take();
    for b in b"\x14ver\r" {
        rig.module.elsc().serial_input(*b);
    }
    // The line waits for the scheduler's turn at the next tick.
    assert_eq!(rig.aux.take(), "\r\nMSC> ver\r\n");
    rig.step_ms(1);
    rig.module.poll();
    assert_eq!(rig.aux.take(), "\x14ok VER 3.2 \r\n");
}

#[test]
fn an_unpowered_module_still_takes_serial_commands() {
    let rig = Rig::new();
    rig.assert_cmd(0, "pwr c 258", "ok ");
    rig.step_ms(1000);
    rig.aux.take();
    let serial = |line: &str| {
        for b in format!("\x14{line}\r").as_bytes() {
            rig.module.elsc().serial_input(*b);
        }
        rig.step_ms(2);
        rig.module.poll();
        rig.aux.take()
    };
    assert_eq!(serial("pas none"), "\r\nMSC> pas none\r\n\x14ok\r\n");
    assert_eq!(serial("pwr"), "\r\nMSC> pwr\r\n\x14ok d\r\n");
    assert_eq!(serial("pwr u"), "\r\nMSC> pwr u\r\n\x14ok\r\n");
    rig.step_ms(5 * SECOND);
    rig.module.poll();
    assert!(rig.module.powered());
}

#[test]
fn the_start_announces_the_origin_2000_version_and_fsc_mirrors_dsp() {
    let rig = Rig::new();
    assert_eq!(rig.aux.take(), "\x14MSC VER 3.2 \r\n");
    rig.assert_cmd(0, "dsp quiet", "ok ");
    rig.assert_cmd(0, "fsc 1", "ok ");
    rig.assert_cmd(1, "dsp hi", "ok ");
    assert_eq!(rig.aux.take(), "\x14dsp hi4\r\n");
}

// ---- one-wire --------------------------------------------------------------------

// The ROM serial bytes, 00 'Y' 'U' 'M' 'E' 'Q', are a synthetic marker
// for the MODULEID part, not a real serial. The pages follow from the module
// serial "K000000001".
const MODULE_ROM: [u8; 8] = [0x09, 0x00, 0x59, 0x55, 0x4d, 0x45, 0x51, 0x33];
const MODULE_PAGES: [u8; 64] = [
    0x01, 0x4b, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x31, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xe6, 0xa2, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x4d, 0x4f, 0x44, 0x55, 0x4c, 0x45,
    0x49, 0x44, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x4b, 0x59,
];

#[test]
fn the_identity_pages_are_the_format_byte_the_serial_and_the_moduleid_name() {
    let pages = identity_pages("K000000001").unwrap();
    assert_eq!(pages.iter().flatten().copied().collect::<Vec<_>>(), MODULE_PAGES);
    for bad in ["SHORT", "K0000000012", "K000000:01", "K000000;01", "K000000\x0101"] {
        assert!(identity_pages(bad).is_none(), "{bad:?}");
    }
}

#[test]
fn the_moduleid_part_answers_through_the_nic_line_of_every_node() {
    let rig = Rig::new();
    rig.module.attach_parts("K000000001", 0x5145_4d55_5900, None).unwrap();
    // Read Memory from the start: the command echo CRC 0x8d, then the pages.
    for node in 0..ADAPTERS {
        rig.ow_start_read(node, &MODULE_ROM, 0xf0, 0);
        assert_eq!(rig.ow_read_byte(node), 0x8d, "node {node}");
        assert_eq!(rig.ow_read_byte(node), MODULE_PAGES[0], "node {node}");
    }
    rig.ow_start_read(1, &MODULE_ROM, 0xf0, 0);
    assert_eq!(rig.ow_read_byte(1), 0x8d);
    let pages: Vec<u8> = (0..64).map(|_| rig.ow_read_byte(1)).collect();
    assert_eq!(pages, MODULE_PAGES);
}

#[test]
fn a_line_with_no_part_reads_no_presence_and_a_bad_serial_attaches_nothing() {
    let rig = Rig::new();
    assert!(!rig.ow_reset(0));
    assert!(rig.module.attach_parts("short", 1, None).is_err());
    assert!(!rig.ow_reset(0));
}

#[test]
fn the_net_a_hub_line_can_share_holds_the_attached_part() {
    let rig = Rig::new();
    rig.module.attach_parts("K000000001", 0x5145_4d55_5900, None).unwrap();
    assert_eq!(rig.module.midplane_onewire().lock().roms(), vec![MODULE_ROM]);
}

/// Every adapter sees a grant passively, whoever holds the bus: a node
/// reading while a peer has a transfer open still finds the grant.
#[test]
fn an_adapter_sees_a_grant_while_a_peer_holds_the_bus() {
    let rig = Rig::new();
    rig.step_ms(1);
    // Node 0 takes grant 0xe0 and opens a transfer.
    assert_eq!(rig.r(0, DATA), GRANT);
    assert!(rig.start(0, 0x50, false));
    // Node 2, which never looked, finds the same grant.
    assert_eq!(rig.r(2, DATA), GRANT);
    rig.stop(0);
    assert_eq!(rig.r(3, DATA), GRANT);
}
