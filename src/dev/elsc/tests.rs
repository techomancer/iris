//! Register-level tests of the controller, driven through the PCF8584 the way
//! the PROM drives it, on a manual clock so none of them waits.

use super::*;
use crate::dev::elsc_nvram::ElscNvram;
use crate::dev::i2c::I2cBus;
use crate::dev::pcf8584::{ctl, sts, Pcf8584};
use std::io;
use std::sync::atomic::AtomicU64;

const DATA: u64 = 0;
const CTRL: u64 = 8;
const ELSC_ADDR: u8 = I2C_ADDR;
const NVRAM_PAGE7: u8 = 0x57;

const SIGNAL1: u8 = 0xe0;
const SIGNAL2: u8 = 0xe8;
const STEP_NS: u64 = 10_000;
const SCAN_NS: u64 = 100_000;
const TICK_NS: u64 = 1_000_000;
const SECOND: u64 = 1000;

/// What a default "pwr c" takes on an Origin 200: five seconds off, then four
/// of power-up with reset held.
const CYCLE_SECS: u64 = 9;

const BANNER: &str = "\x14MSC V 3.1S  \r\n";

/// The AUX connector's far end.
#[derive(Default)]
pub(crate) struct Aux(Mutex<Vec<u8>>);

impl SerialBackend for Aux {
    fn send_byte(&self, byte: u8) {
        self.0.lock().push(byte);
    }

    fn recv_byte(&self) -> io::Result<u8> {
        Err(io::Error::new(io::ErrorKind::WouldBlock, "none"))
    }
}

impl Aux {
    pub(crate) fn take(&self) -> String {
        String::from_utf8_lossy(&std::mem::take(&mut *self.0.lock())).into_owned()
    }
}

/// Virtual time, set by hand.
#[derive(Clone, Default)]
pub(crate) struct TestClock(Arc<AtomicU64>);

impl TestClock {
    pub(crate) fn now_ns(&self) -> u64 {
        self.0.load(Ordering::SeqCst)
    }

    pub(crate) fn advance_ns(&self, ns: u64) {
        self.0.fetch_add(ns, Ordering::SeqCst);
    }

    pub(crate) fn advance_ms(&self, ms: u64) {
        self.advance_ns(ms * 1_000_000);
    }

    pub(crate) fn advance_secs(&self, s: u64) {
        self.advance_ms(s * 1000);
    }

    pub(crate) fn clock(&self) -> Clock {
        let now = self.0.clone();
        Arc::new(move || now.load(Ordering::SeqCst))
    }
}

/// The sum of a frame's bytes modulo 256, written out here rather than shared
/// with the controller: a valid frame sums to zero.
pub(crate) fn byte_sum(buf: &[u8]) -> u8 {
    (buf.iter().map(|b| *b as u32).sum::<u32>() % 256) as u8
}

struct Rig {
    busy: BusBusy,
    clock: TestClock,
    elsc: Arc<Elsc>,
    nvram: Arc<ElscNvram>,
    pcf: Pcf8584,
    aux: Arc<Aux>,
}

pub(crate) fn frame(text: &[u8]) -> Vec<u8> {
    let mut req = vec![text.len() as u8];
    req.extend_from_slice(text);
    req.push(((256 - byte_sum(&req) as u32) % 256) as u8);
    req
}

impl Rig {
    fn new() -> Self {
        Self::with(ElscConfig::default())
    }

    fn with(cfg: ElscConfig) -> Self {
        let clock = TestClock::default();
        let busy = BusBusy::default();
        let nvram = Arc::new(ElscNvram::origin200(None).unwrap());
        let elsc = Arc::new(Elsc::new(cfg, nvram.clone(), clock.clock(), busy.clone()));
        let mut bus = I2cBus::new();
        bus.add(elsc.clone());
        bus.add(nvram.clone());
        let pcf = Pcf8584::new(bus, busy.clone());
        let aux = Arc::new(Aux::default());
        elsc.set_serial_backend(aux.clone());
        pcf.power_on();
        elsc.power_on();
        Self { busy, clock, elsc, nvram, pcf, aux }
    }

    fn w(&self, off: u64, v: u8) {
        self.pcf.write_register(off, v);
    }

    fn r(&self, off: u64) -> u8 {
        self.pcf.read_register(off)
    }

    fn step_ms(&self, ms: u64) {
        self.clock.advance_ms(ms);
    }

    fn advance_to_ms(&self, ms: u64) {
        let now = self.clock.now_ns();
        assert!(ms * 1_000_000 >= now, "already past {ms} ms");
        self.clock.advance_ns(ms * 1_000_000 - now);
    }

    fn step_secs(&self, s: u64) {
        self.clock.advance_secs(s);
    }

    fn events(&self) -> Vec<ModuleOutput> {
        self.elsc.take_events()
    }

    /// The Hub's side of a module reset: the adapter resets and the
    /// controller rides through.
    fn module_reset(&self) {
        self.elsc.ride_through_reset();
        self.pcf.power_on();
    }

    // ---- transactions ---------------------------------------------------

    /// One combined transaction, the way the PROM makes it. The read discards
    /// the chip's stale data register first.
    fn transact(&self, req: &[u8]) -> Vec<u8> {
        self.w(CTRL, ctl::PIN | ctl::ESO);
        self.w(DATA, ELSC_ADDR << 1);
        self.w(CTRL, ctl::PIN | ctl::ESO | ctl::STA | ctl::ACK);
        assert_eq!(self.r(CTRL) & sts::LRB, 0);
        for b in req {
            self.w(DATA, *b);
        }
        // Repeated start, switching direction: control first, then address.
        self.w(CTRL, ctl::ESO | ctl::STA | ctl::ACK);
        self.w(DATA, (ELSC_ADDR << 1) | 1);
        let _ = self.r(DATA);
        self.read_frame(true)
    }

    /// The reply text for a request, after checking the frame.
    fn reply_bytes(&self, text: &[u8]) -> Vec<u8> {
        let resp = self.transact(&frame(text));
        assert_eq!(byte_sum(&resp), 0, "bad checksum in {resp:?}");
        resp[1..1 + resp[0] as usize].to_vec()
    }

    fn reply(&self, text: &str) -> String {
        String::from_utf8_lossy(&self.reply_bytes(text.as_bytes())).into_owned()
    }

    #[track_caller]
    fn assert_reply(&self, text: &str, want: &str) {
        assert_eq!(self.reply(text), want, "reply to {text:?}");
    }

    /// A request completed through the direction switch, then abandoned with
    /// the response entirely unread.
    fn request_without_read(&self, text: &str) {
        let req = frame(text.as_bytes());
        self.w(CTRL, ctl::PIN | ctl::ESO);
        self.w(DATA, ELSC_ADDR << 1);
        self.w(CTRL, ctl::PIN | ctl::ESO | ctl::STA | ctl::ACK);
        assert_eq!(self.r(CTRL) & sts::LRB, 0);
        for b in req {
            self.w(DATA, b);
        }
        self.w(CTRL, ctl::ESO | ctl::STA | ctl::ACK);
        self.w(DATA, (ELSC_ADDR << 1) | 1);
        self.w(CTRL, ctl::PIN | ctl::ESO | ctl::STO);
    }

    /// Start a read-only transaction and discard the stale first byte.
    fn begin_read(&self) {
        self.w(DATA, (ELSC_ADDR << 1) | 1);
        self.w(CTRL, ctl::PIN | ctl::ESO | ctl::STA | ctl::ACK);
        assert_eq!(self.r(CTRL) & sts::LRB, 0);
        let _ = self.r(DATA);
    }

    /// Read a whole frame from a started read, NAKing the last byte.
    fn read_frame(&self, stop: bool) -> Vec<u8> {
        let mut resp = vec![self.r(DATA)];
        let total = resp[0] as usize + 2;
        for i in 1..total {
            if i == total - 1 {
                self.w(CTRL, ctl::ESO);
            }
            resp.push(self.r(DATA));
        }
        if stop {
            self.w(CTRL, ctl::PIN | ctl::ESO | ctl::STO);
        }
        resp
    }

    /// A read-only transaction: fetch a response with no request written.
    fn read_response(&self) -> Vec<u8> {
        self.begin_read();
        self.read_frame(true)
    }

    // ---- arbitration ----------------------------------------------------

    fn observer(&self) {
        self.w(CTRL, ctl::PIN);
        self.w(DATA, 0x7f);
        self.w(CTRL, ctl::PIN | ctl::ES1);
        self.w(DATA, 0x1c);
        self.w(CTRL, ctl::PIN);
        assert_eq!(self.r(DATA), 0x7f);
        self.w(DATA, 0x09);
        self.w(CTRL, ctl::PIN | ctl::ESO);
    }

    fn is_signal(byte: u8) -> bool {
        (0xe0..=0xef).contains(&byte)
    }

    /// Step the clock until the ring publishes a code different from the one
    /// currently latched, and return the new grant.
    fn next_code(&self) -> u8 {
        let start = self.r(DATA);
        for _ in 0..2000 {
            let d = self.r(DATA);
            if Self::is_signal(d) && d != start {
                return d;
            }
            self.clock.advance_ns(SCAN_NS);
        }
        panic!("the ring never moved");
    }

    /// Step until a new grant appears, leaving the clock no more than
    /// `STEP_NS` past the tick that published it.
    fn anchor_on_tick(&self) -> u8 {
        let start = self.r(DATA);
        for _ in 0..32 * TICK_NS / STEP_NS {
            self.clock.advance_ns(STEP_NS);
            let d = self.r(DATA);
            if Self::is_signal(d) && d != start {
                return d;
            }
        }
        panic!("no tick");
    }

    /// Loop 1: wait for another processor's grant, with the bus free.
    fn wait_for_absence(&self) {
        for _ in 0..2000 {
            let d = self.r(DATA);
            let s = self.r(CTRL);
            if s & sts::BB != 0 && Self::is_signal(d) && d & 7 != 0 {
                return;
            }
            self.clock.advance_ns(SCAN_NS);
        }
        panic!("no other processor's grant");
    }

    /// Loop 2: step until the bus is free and the data register holds
    /// processor 0's signal 1 or signal 2, and report which.
    fn wait_for_signal(&self) -> u8 {
        for _ in 0..2000 {
            let d = self.r(DATA);
            let s = self.r(CTRL);
            if s & sts::BB != 0 && (d == SIGNAL1 || d == SIGNAL2) {
                return d;
            }
            self.clock.advance_ns(SCAN_NS);
        }
        panic!("no grant for processor 0");
    }

    /// Arbitrate for processor 0's slot.
    fn arbitrate(&self) {
        self.wait_for_absence();
        self.wait_for_signal();
    }

    /// One NVRAM byte as firmware reads it: arbitrated, offset-write then
    /// data-read, closed with the invalid-address step.
    fn firmware_nvram_read(&self, offset: u8) -> u8 {
        self.arbitrate();
        self.w(DATA, NVRAM_PAGE7 << 1);
        self.w(CTRL, ctl::PIN | ctl::ESO | ctl::STA | ctl::ACK);
        assert_eq!(self.r(CTRL) & sts::LRB, 0);
        self.w(DATA, offset);
        self.w(CTRL, ctl::ESO | ctl::STA | ctl::ACK);
        self.w(DATA, (NVRAM_PAGE7 << 1) | 1);
        let _ = self.r(DATA);
        self.w(CTRL, ctl::ESO);
        let value = self.r(DATA);
        self.close_with_invalid_address();
        value
    }

    /// The closing invalid-address step, still inside the transfer: control
    /// first, PIN deliberately absent, then the address byte.
    fn close_with_invalid_address(&self) {
        self.w(CTRL, ctl::ESO | ctl::STA | ctl::ACK);
        self.w(DATA, 0xfe);
        assert_eq!(self.r(CTRL) & sts::LRB, sts::LRB);
        self.w(CTRL, ctl::PIN | ctl::ESO | ctl::STO);
    }

    /// One full command as firmware drives it: arbitrate, the combined
    /// transaction, then the closing repeated start to the invalid address.
    fn firmware_command(&self, text: &str) -> Vec<u8> {
        self.arbitrate();
        self.w(DATA, ELSC_ADDR << 1);
        self.w(CTRL, ctl::PIN | ctl::ESO | ctl::STA | ctl::ACK);
        assert_eq!(self.r(CTRL) & sts::LRB, 0);
        for b in frame(text.as_bytes()) {
            self.w(DATA, b);
        }
        self.w(CTRL, ctl::ESO | ctl::STA | ctl::ACK);
        self.w(DATA, (ELSC_ADDR << 1) | 1);
        let _ = self.r(DATA);
        let resp = self.read_frame(false);
        self.close_with_invalid_address();
        resp
    }

    // ---- serial ---------------------------------------------------------

    fn type_bytes(&self, bytes: &[u8]) {
        for b in bytes {
            self.elsc.serial_input(*b);
        }
    }

    /// One serial command line: Ctrl+T, the text and the reply it prints.
    #[track_caller]
    fn serial_command(&self, line: &str, reply: &str) {
        self.type_bytes(format!("\x14{line}\r").as_bytes());
        assert_eq!(
            self.aux.take(),
            format!("\r\nMSC> {line}\r\n\x14{reply}\r\n"),
            "serial {line:?}"
        );
    }
}

/// Text of a framed response.
pub(crate) fn text(resp: &[u8]) -> String {
    String::from_utf8_lossy(&resp[1..1 + resp[0] as usize]).into_owned()
}

fn power_events(events: &[ModuleOutput]) -> Vec<bool> {
    events
        .iter()
        .filter_map(|e| if let ModuleOutput::Power(on) = e { Some(*on) } else { None })
        .collect()
}

fn count_nmi(events: &[ModuleOutput]) -> usize {
    events.iter().filter(|e| **e == ModuleOutput::Nmi(true)).count()
}

// ---- framing and the request grammar -------------------------------------

#[test]
fn ver_command_succeeds() {
    let rig = Rig::new();
    let resp = rig.transact(&frame(b"ver"));
    assert_eq!(byte_sum(&resp), 0);
    assert!(resp[0] >= 4);
    let t = text(&resp);
    assert!(t.starts_with("ok "), "{t:?}");
    // The synthetic version field stays within the eight-byte limit.
    let version = t[3..].split(' ').next().unwrap();
    assert!(!version.is_empty() && version.len() < 8);
    assert_eq!(t, "ok V 3.1S  ");
}

#[test]
fn bad_request_checksum_gets_err_csum() {
    let rig = Rig::new();
    let mut req = frame(b"ver");
    *req.last_mut().unwrap() ^= 0xff;
    let resp = rig.transact(&req);
    assert_eq!(byte_sum(&resp), 0);
    assert_eq!(resp[0], 8);
    assert_eq!(text(&resp), "err csum");
}

#[test]
fn a_short_or_long_request_is_err_csum() {
    let rig = Rig::new();
    // The length byte says three, but four text bytes and a checksum follow.
    let mut req = frame(b"verx");
    req[0] = 3;
    assert_eq!(text(&rig.transact(&req)), "err csum");
    // Nothing but a length.
    assert_eq!(text(&rig.transact(&[0])), "err csum");
}

#[test]
fn an_unimplemented_command_completes_with_err_cmd() {
    let rig = Rig::new();
    let resp = rig.transact(&frame(b"xyz"));
    assert_eq!(byte_sum(&resp), 0);
    assert_eq!(resp[0], 7);
    assert_eq!(text(&resp), "err cmd");
}

#[test]
fn rsw_reports_the_switches_active_low_with_the_role_from_topology() {
    let rig = Rig::new();
    assert_eq!(rig.transact(&frame(b"rsw"))[0], 5);
    rig.assert_reply("rsw", "ok ff");
    // A configured value comes back inverted.
    let rig = Rig::with(ElscConfig { dip_switches: 1, ..Default::default() });
    rig.assert_reply("rsw", "ok fe");
    // Bit 0x04 is the role, and a lone module's controller is the master.
    let rig = Rig::with(ElscConfig { dip_switches: 4, ..Default::default() });
    rig.assert_reply("rsw", "ok ff");
    let rig = Rig::with(ElscConfig { dip_switches: 16, ..Default::default() });
    rig.assert_reply("rsw", "ok ef");
    // A slave reads the role bit clear.
    let rig = Rig::with(ElscConfig { dip_switches: 16, master: false, ..Default::default() });
    rig.assert_reply("rsw", "ok eb");
}

#[test]
fn request_grammar() {
    let rig = Rig::with(ElscConfig { dip_switches: 240, ..Default::default() });
    // The mnemonic and its separator.
    assert_eq!(text(&rig.transact(&frame(b""))), "err cmd");
    rig.assert_reply("ve", "err cmd");
    rig.assert_reply("VER", "err cmd");
    rig.assert_reply("verx", "err cmd");
    rig.assert_reply("ver", "ok V 3.1S  ");
    rig.assert_reply("ver junk", "ok V 3.1S  ");
    // Hex replies carry no leading zero.
    rig.assert_reply("rsw", "ok f");
    rig.assert_reply("rsw 12", "ok f");
    // The character reader skips spaces and leaves the rest unread.
    rig.assert_reply("pwr   q", "ok 1");
    rig.assert_reply("pwr q   junk", "ok 0");
    // An Origin 200 module is powered whenever the machine runs.
    rig.assert_reply("pwr", "ok u");
    rig.assert_reply("pwr u", "ok ");
    rig.assert_reply("pwr q", "ok 0");
}

#[test]
fn the_origin_200_build_has_no_display_commands() {
    let rig = Rig::new();
    for c in ["dsc 4 E", "dsc 1 A", "dsp testing", "dsp", "dsc", "key", "vlm 5", "fre", "mod"] {
        let resp = rig.transact(&frame(c.as_bytes()));
        assert_eq!(text(&resp), "err cmd", "{c}");
    }
    let resp = rig.transact(&frame(b"ver"));
    assert_eq!(&resp[1..12], b"ok V 3.1S  ");
    assert!(!rig.elsc.describe().contains("display:"));
}

#[test]
fn no_supervisor_on_i2c() {
    let rig = Rig::new();
    for c in ["pas", "pas none", "pas ab", "pas s abcd"] {
        let resp = rig.transact(&frame(c.as_bytes()));
        assert_eq!(resp[0], 7);
        assert_eq!(text(&resp), "err cmd");
    }
    // A destructive command with no "pas" ever sent.
    rig.assert_reply("rst", "ok ");
    rig.step_secs(1);
    assert!(rig.events().contains(&ModuleOutput::Reset));
}

// ---- arbitration -----------------------------------------------------------

#[test]
fn arbitration_observes_a_grant_and_keeps_the_staged_start() {
    let rig = Rig::new();
    rig.observer();
    rig.wait_for_absence();
    let signal = rig.wait_for_signal();

    // A grant is an unmatched address phase, not a payload byte. Its address
    // replaces readable S0 while START retains the staged controller address.
    rig.w(DATA, ELSC_ADDR << 1);
    let grant = rig.next_code();
    assert_ne!(rig.r(CTRL) & sts::BB, 0);
    assert_eq!(grant & 7, ((signal & 7) + 1) % 5);
    rig.w(CTRL, ctl::PIN | ctl::ESO | ctl::STA | ctl::ACK);
    assert_eq!(rig.r(CTRL) & sts::LRB, 0);
    for b in frame(b"ver") {
        rig.w(DATA, b);
    }
    rig.w(CTRL, ctl::ESO | ctl::STA | ctl::ACK);
    rig.w(DATA, (ELSC_ADDR << 1) | 1);
    let _ = rig.r(DATA);
    let resp = rig.read_frame(false);
    assert_eq!(byte_sum(&resp), 0);
    assert!(text(&resp).starts_with("ok"));
    rig.w(CTRL, ctl::PIN | ctl::ESO | ctl::STO);
    rig.arbitrate();

    // Reset also releases an address staged without a following START.
    rig.w(DATA, ELSC_ADDR << 1);
    rig.pcf.power_on();
    rig.observer();
    rig.arbitrate();

    // A benign control change must not abandon the staged address.
    rig.w(DATA, ELSC_ADDR << 1);
    rig.w(CTRL, ctl::PIN | ctl::ESO | ctl::ACK);
    rig.next_code();
    rig.w(CTRL, ctl::PIN | ctl::ESO | ctl::STA | ctl::ACK);
    assert_eq!(rig.r(CTRL) & sts::LRB, 0);
    rig.w(CTRL, ctl::PIN | ctl::ESO | ctl::STO);
    rig.arbitrate();
}

#[test]
fn each_ring_position_lasts_one_millisecond() {
    let rig = Rig::new();
    rig.observer();
    let mut grant = rig.anchor_on_tick();
    for _ in 0..10 {
        let next = SIGNAL1 + ((grant & 7) + 1) % 5;
        // From just past one tick the grant stands until the next one, a
        // millisecond later, which publishes the next.
        rig.clock.advance_ns(TICK_NS - 2 * STEP_NS);
        assert_eq!(rig.r(DATA), grant);
        rig.clock.advance_ns(2 * STEP_NS);
        assert_eq!(rig.r(DATA), next);
        grant = next;
    }
}

#[test]
fn the_ring_turns_through_five_positions() {
    let rig = Rig::new();
    rig.observer();
    loop {
        if rig.anchor_on_tick() == SIGNAL1 {
            break;
        }
    }
    for code in 1..4 {
        rig.clock.advance_ns(TICK_NS);
        assert_eq!(rig.r(DATA), SIGNAL1 + code);
    }
    // The fifth position stands for one tick like the others, with a byte no
    // processor of this module owns.
    rig.clock.advance_ns(TICK_NS);
    assert_eq!(rig.r(DATA), SIGNAL1 + 4);
    rig.clock.advance_ns(TICK_NS);
    assert_eq!(rig.r(DATA), SIGNAL1);
}

#[test]
fn a_busy_tick_publishes_nothing_and_does_not_advance() {
    let rig = Rig::new();
    rig.observer();
    let signal = rig.anchor_on_tick();

    rig.w(DATA, ELSC_ADDR << 1);
    rig.w(CTRL, ctl::PIN | ctl::ESO | ctl::STA | ctl::ACK);
    assert_eq!(rig.r(CTRL) & sts::LRB, 0);
    assert_eq!(rig.r(CTRL) & sts::BB, 0);

    // Three eligible ticks pass while the bus is busy.
    rig.clock.advance_ns(3 * TICK_NS + TICK_NS / 2);
    assert_eq!(rig.r(CTRL) & sts::BB, 0);

    rig.w(CTRL, ctl::PIN | ctl::ESO | ctl::STO);
    assert_ne!(rig.r(CTRL) & sts::BB, 0);

    // The bus is free, but nothing is published before the next tick.
    rig.clock.advance_ns(TICK_NS / 2 - 2 * STEP_NS);
    assert!(!Rig::is_signal(rig.r(DATA)));

    // That tick publishes the code the busy ticks skipped.
    rig.clock.advance_ns(2 * STEP_NS);
    assert_eq!(rig.r(DATA) & 7, ((signal & 7) + 1) % 5);
    assert_eq!(rig.next_code() & 7, ((signal & 7) + 2) % 5);
}

#[test]
fn a_late_clock_replays_the_ring_on_its_grid() {
    let rig = Rig::new();
    rig.observer();
    let grant = rig.anchor_on_tick();
    // 5 000 ticks is a whole number of revolutions. Just short of the last
    // tick the ring stands one position back; at it, it is where it began.
    rig.clock.advance_ns(5000 * TICK_NS - STEP_NS);
    assert_eq!(rig.r(DATA), SIGNAL1 + ((grant & 7) + 4) % 5);
    rig.clock.advance_ns(STEP_NS);
    assert_eq!(rig.r(DATA), grant);
}

#[test]
fn a_pending_response_announces_signal2_to_its_requester() {
    let rig = Rig::new();
    rig.observer();
    rig.request_without_read("ver");
    rig.wait_for_absence();
    assert_eq!(rig.wait_for_signal(), SIGNAL2);
    let info = rig.elsc.describe();
    assert!(info.contains("pending response"), "{info}");
    assert!(info.contains("for processor code 0"), "{info}");
}

#[test]
fn a_consumed_response_reverts_to_signal1() {
    let rig = Rig::new();
    rig.observer();
    rig.reply("ver");
    rig.wait_for_absence();
    assert_eq!(rig.wait_for_signal(), SIGNAL1);
}

#[test]
fn an_announced_response_is_fetched_after_the_stop() {
    let rig = Rig::new();
    rig.observer();
    rig.request_without_read("ver");
    rig.wait_for_absence();
    assert_eq!(rig.wait_for_signal(), SIGNAL2);

    let resp = rig.read_response();
    assert_eq!(byte_sum(&resp), 0);
    assert!(text(&resp).starts_with("ok "));
    rig.wait_for_absence();
    assert_eq!(rig.wait_for_signal(), SIGNAL1);
}

#[test]
fn a_new_request_supersedes_a_pending_response() {
    let rig = Rig::new();
    rig.observer();
    rig.request_without_read("ver");
    rig.wait_for_absence();
    assert_eq!(rig.wait_for_signal(), SIGNAL2);
    rig.assert_reply("rsw", "ok ff");
    rig.wait_for_absence();
    assert_eq!(rig.wait_for_signal(), SIGNAL1);
}

#[test]
fn the_requesting_cpu_is_announced_in_its_own_slot() {
    let rig = Rig::new();
    rig.observer();
    crate::cpu::smp::set_current_cpu(2);
    rig.request_without_read("ver");
    crate::cpu::smp::set_current_cpu(u8::MAX);
    // Code 0 has nothing waiting; code 2 does.
    rig.wait_for_absence();
    assert_eq!(rig.wait_for_signal(), SIGNAL1);
    let mut saw = false;
    for _ in 0..50 {
        if rig.r(DATA) == SIGNAL2 + 2 {
            saw = true;
            break;
        }
        rig.clock.advance_ns(TICK_NS);
    }
    assert!(saw, "no signal 2 for code 2");
}

#[test]
fn frames_match_literal_bytes() {
    // "ver" is 0x76 0x65 0x72: with the length 3 they sum to 0x50, so the
    // checksum that brings the frame to zero is 0xb0.
    assert_eq!(frame(b"ver"), [0x03, 0x76, 0x65, 0x72, 0xb0]);
    let rig = Rig::new();
    let resp = rig.transact(&[0x03, 0x76, 0x65, 0x72, 0xb0]);
    assert_eq!(
        resp,
        [0x0b, b'o', b'k', b' ', b'V', b' ', b'3', b'.', b'1', b'S', b' ', b' ', 0x60]
    );
    // "err csum" for a request whose last byte is wrong by one.
    let resp = rig.transact(&[0x03, 0x76, 0x65, 0x72, 0xb1]);
    assert_eq!(
        resp,
        [0x08, b'e', b'r', b'r', b' ', b'c', b's', b'u', b'm', 0xd7]
    );
}

// ---- console ---------------------------------------------------------------

#[test]
fn acp_writes_console_output_to_the_aux_port() {
    let rig = Rig::new();
    let resp = rig.transact(&frame(b"acp 0 hi"));
    assert_eq!(byte_sum(&resp), 0);
    // "hi" is two characters; the accepted count in hex is "2".
    assert_eq!(resp[0], 4);
    assert_eq!(text(&resp), "ok 2");
    // The controller's start announcement comes first.
    assert_eq!(rig.aux.take(), format!("{BANNER}hi"));
}

#[test]
fn acp_with_no_backend_still_answers() {
    let rig = Rig::new();
    rig.elsc.set_serial_backend(null_serial_backend());
    rig.assert_reply("acp 0 hi", "ok 2");
    // Too short to carry output is a bad argument, never a bare ok.
    rig.assert_reply("acp 0", "err arg");
    rig.assert_reply("acp 0 ", "ok 0");
}

#[test]
fn acp_takes_at_most_86_characters() {
    let rig = Rig::new();
    let long = format!("acp 0 {}", "x".repeat(88));
    rig.assert_reply(&long, "ok 56");
    assert_eq!(rig.aux.take(), format!("{BANNER}{}", "x".repeat(86)));
}

#[test]
fn get_returns_queued_console_input() {
    let rig = Rig::new();
    // Input is announced only once a processor has sent output.
    rig.assert_reply("acp 0 ", "ok 0");
    rig.type_bytes(b"hello");
    rig.observer();
    rig.wait_for_absence();
    assert_eq!(rig.wait_for_signal(), SIGNAL2);
    let resp = rig.transact(&frame(b"get"));
    assert_eq!(byte_sum(&resp), 0);
    assert!(resp[0] >= 6 + 5);
    assert_eq!(text(&resp), "acp . hello");
}

#[test]
fn a_command_escape_byte_is_not_queued() {
    let rig = Rig::new();
    rig.assert_reply("acp 0 ", "ok 0");
    // What ends an escaped command is the model's own choice: a single \r.
    rig.type_bytes(b"AB\x14zzz\rCD");
    rig.observer();
    rig.wait_for_absence();
    assert_eq!(rig.wait_for_signal(), SIGNAL2);
    let resp = rig.transact(&frame(b"get"));
    assert_eq!(resp[0], 6 + 4);
    assert_eq!(text(&resp), "acp . ABCD");
}

#[test]
fn console_selection_commands() {
    let rig = Rig::new();
    rig.aux.take();
    rig.assert_reply("sel", "ok auto");
    rig.assert_reply("see", "ok all");
    rig.assert_reply("get", "acp . ");

    rig.assert_reply("sel 2b", "ok ");
    rig.assert_reply("sel", "ok 2b");
    rig.assert_reply("get", "acp 3 ");
    rig.assert_reply("sel all", "ok ");
    rig.assert_reply("sel", "ok none");
    rig.assert_reply("get", "acp / ");
    rig.assert_reply("sel  4 A", "ok ");
    rig.assert_reply("sel", "ok 4a");
    rig.assert_reply("sel AUTO", "ok ");
    rig.assert_reply("sel", "ok auto");
    rig.assert_reply("sel no", "ok ");
    rig.assert_reply("sel", "ok none");

    // A bad name changes nothing.
    for bad in ["sel 5a", "sel 1", "sel 1c", "sel x", "see au"] {
        rig.assert_reply(bad, "err arg");
    }
    rig.assert_reply("sel", "ok none");

    // "see" with a processor filters output and selects that input.
    rig.assert_reply("see 1b", "ok ");
    rig.assert_reply("see", "ok 1b");
    rig.assert_reply("sel", "ok 1b");
    // Output from processor 0 is counted, not copied.
    rig.assert_reply("acp 0 hi", "ok 2");
    assert_eq!(rig.aux.take(), "");
    // The filter goes by the frame's selector, "1" being processor 1.
    rig.assert_reply("acp 1 ab", "ok 2");
    assert_eq!(rig.aux.take(), "ab");
    rig.assert_reply("see 1a", "ok ");
    rig.assert_reply("acp 0 yo", "ok 2");
    assert_eq!(rig.aux.take(), "yo");
    rig.assert_reply("see none", "ok ");
    rig.assert_reply("see", "ok all");
    rig.assert_reply("sel", "ok 1a");

    // "clr" leaves the selection; a start restores the default.
    rig.assert_reply("clr", "ok ");
    rig.assert_reply("sel", "ok 1a");
    rig.elsc.power_on();
    rig.assert_reply("sel", "ok auto");
}

#[test]
fn queued_input_is_announced_to_the_selected_processor_only() {
    let rig = Rig::new();
    rig.observer();
    rig.assert_reply("sel 1b", "ok ");
    rig.type_bytes(b"hi");
    rig.wait_for_absence();
    assert_eq!(rig.wait_for_signal(), SIGNAL1);

    // A new selection discards the queue; fresh input is announced.
    rig.assert_reply("sel 1a", "ok ");
    assert!(rig.elsc.describe().contains("rx queue: 0 byte(s)"));
    rig.type_bytes(b"yo");
    rig.wait_for_absence();
    assert_eq!(rig.wait_for_signal(), SIGNAL2);
    rig.assert_reply("get", "acp 0 yo");

    // Automatic selection follows the selector in the last output frame, not
    // the processor that sent it. No output yet: nobody is announced to.
    rig.assert_reply("sel auto", "ok ");
    rig.type_bytes(b"zz");
    rig.wait_for_absence();
    assert_eq!(rig.wait_for_signal(), SIGNAL1);
    rig.assert_reply("acp 0 ", "ok 0");
    rig.wait_for_absence();
    assert_eq!(rig.wait_for_signal(), SIGNAL2);
    // A frame with a corrupted checksum still records its sender.
    let mut req = frame(b"acp 1 ");
    *req.last_mut().unwrap() ^= 0xff;
    assert_eq!(text(&rig.transact(&req)), "err csum");
    rig.wait_for_absence();
    assert_eq!(rig.wait_for_signal(), SIGNAL1);
}

#[test]
fn info_dump_names_the_empty_and_the_waiting_queue() {
    let rig = Rig::new();
    let empty = rig.elsc.describe();
    assert!(empty.contains("rx queue: 0 byte(s)"), "{empty}");
    assert!(empty.contains("pending response: none"), "{empty}");
    assert!(empty.contains("signal1"), "{empty}");

    rig.assert_reply("acp 0 ", "ok 0");
    rig.type_bytes(b"hello");
    let full = rig.elsc.describe();
    assert!(full.contains("rx queue: 5 byte(s)"), "{full}");
    assert!(full.contains("hello"), "{full}");
    assert!(full.contains("signal2"), "{full}");
}

// ---- boot sequence ---------------------------------------------------------

#[test]
fn boot_acceptance_sequence() {
    const INITIAL: [u8; 16] = [0x37, b'n', b'o', b'n', b'e', 0, 0, 0x02, 0, 0, 0, 0x78, 0, 0, 0, 0];
    let rig = Rig::new();
    rig.observer();
    let block: Vec<u8> = (0..16).map(|i| rig.firmware_nvram_read(i)).collect();
    // The controller's block as its first power-up leaves it: the marker, the
    // password "none", automatic power-up, no module number and the
    // full-clock midplane's clock-select byte.
    assert_eq!(block, INITIAL);

    let resp = rig.firmware_command("rsw");
    assert_eq!(byte_sum(&resp), 0);
    assert_eq!(resp[0], 5);
    assert_eq!(text(&resp), "ok ff");

    let resp = rig.firmware_command("ver");
    assert_eq!(byte_sum(&resp), 0);
    assert!(text(&resp).starts_with("ok"));
    if resp[0] != 2 {
        assert_eq!(resp[3], b' ');
    }
}

// ---- tas, pwr q, nic, flags ------------------------------------------------

#[test]
fn tas_lock_cell() {
    let rig = Rig::new();
    // Free cell: the pre-acquisition value is 0, then 'A' installs.
    rig.assert_reply("tas A", "ok 0");
    // Re-acquiring returns the literal owner.
    rig.assert_reply("tas A", "ok A");
    // Held by 'A': a different caller reads the holder.
    rig.assert_reply("tas z", "ok A");
    // Release: the pre-release value is the holder, and the cell frees.
    rig.assert_reply("tas 0", "ok A");
    // 'z' is a valid non-hex channel.
    rig.assert_reply("tas z", "ok 0");
    rig.assert_reply("tas z", "ok z");
    // No argument only reports the holder.
    rig.assert_reply("tas", "ok z");
    rig.assert_reply("tas 0", "ok z");
    rig.assert_reply("tas", "ok 0");
    // The owner is the next non-space character and the rest is discarded.
    rig.assert_reply("tas  AB", "ok 0");
    rig.assert_reply("tas B", "ok A");
    rig.assert_reply("tas", "ok A");
    rig.assert_reply("tas 0", "ok A");
    rig.assert_reply("tasx", "err cmd");
}

#[test]
fn a_nul_tas_owner_holds_the_cell_and_round_trips() {
    let rig = Rig::new();
    let nul = frame(b"tas \0");
    let reply = |req: &[u8]| {
        let resp = rig.transact(req);
        assert_eq!(byte_sum(&resp), 0);
        assert_eq!(resp[0], 4);
        resp
    };
    assert_eq!(text(&reply(&nul)), "ok 0");
    // Another caller finds the cell held, and the NUL owner survives framing.
    let resp = reply(&frame(b"tas z"));
    assert_eq!(&resp[1..5], b"ok \0");
    // And 'z' did not steal it.
    assert_eq!(reply(&nul)[4], 0);
    // Release frees it; the pre-release value is the NUL owner.
    assert_eq!(reply(&frame(b"tas 0"))[4], 0);
    assert_eq!(text(&reply(&frame(b"tas z"))), "ok 0");
}

#[test]
fn pwr_q_reports_the_cold_power_on_latch_once() {
    let rig = Rig::new();
    rig.assert_reply("pwr q", "ok 1");
    rig.assert_reply("pwr q", "ok 0");
    rig.assert_reply("pwr z", "err arg");
    rig.assert_reply("pwr q x", "ok 0");
    // The redundant-supply query belongs to later firmware.
    rig.assert_reply("pwr r", "err arg");
}

#[test]
fn nic_reads_an_idle_one_wire_line() {
    let rig = Rig::new();
    rig.assert_reply("nic 1 2 3", "ok 1");
    rig.assert_reply("nic 1 2", "err arg");
    rig.assert_reply("nic 1 2 zz", "err arg");
    // A value shorter than four digits consumes the character after it.
    rig.assert_reply("nic 1 2  3", "err arg");
    // A four-digit value consumes nothing: the next follows with no space.
    rig.assert_reply("nic 000100020003", "ok 1");
    rig.assert_reply("nic 0001 2 3", "err arg");
    rig.assert_reply("nicx", "err cmd");
}

#[test]
fn stored_flag_commands() {
    let rig = Rig::new();
    rig.assert_reply("fan", "ok n");
    rig.assert_reply("fan h", "ok ");
    rig.assert_reply("fan", "ok h");
    rig.assert_reply("fan H", "err arg");
    rig.assert_reply("fan x", "err arg");
    rig.assert_reply("fan", "ok h");
    rig.assert_reply("fan  n junk", "ok ");
    rig.assert_reply("fan", "ok n");

    // The "fsc" query loses its answer over I2C.
    for c in ["fsc", "fsc 1", "fsc", "fsc 0", "fsc x"] {
        rig.assert_reply(c, "ok ");
    }
    // "tst" needs a character; "1" sets and anything else clears.
    rig.assert_reply("tst", "err arg");
    rig.assert_reply("tst 1", "ok ");
    rig.assert_reply("tst x", "ok ");

    // No thermal model: normal, and the slave's letter once it reports.
    rig.assert_reply("tmp", "ok n");
    rig.assert_reply("tmp junk", "ok n");
    rig.assert_reply("sla h", "ok ");
    rig.assert_reply("tmp", "ok n");
    rig.assert_reply("sla \x07", "ok ");
    rig.assert_reply("tmp", "ok n h");
    rig.assert_reply("sla n", "ok ");
    rig.assert_reply("tmp", "ok n n");

    // "clr" drops the fan request and a pending power-down countdown.
    rig.assert_reply("fan h", "ok ");
    rig.assert_reply("pwr d 2", "ok ");
    rig.assert_reply("clr", "ok ");
    rig.assert_reply("fan", "ok n");
    rig.step_secs(3);
    rig.elsc.poll();
    assert!(rig.elsc.module_powered());
}

#[test]
fn a_slave_stores_the_report_it_would_forward_and_publishes_no_grants() {
    let rig = Rig::with(ElscConfig { master: false, ..Default::default() });
    rig.assert_reply("sla o", "ok ");
    rig.assert_reply("tmp", "ok n");
    rig.observer();
    rig.step_ms(20);
    assert!(!Rig::is_signal(rig.r(DATA)));
}

#[test]
fn aut_reads_and_writes_the_nvram_bit() {
    let rig = Rig::new();
    rig.observer();
    rig.assert_reply("aut", "ok 1");
    rig.assert_reply("aut 0", "ok ");
    rig.assert_reply("aut", "ok 0");
    rig.assert_reply("aut 1", "ok ");
    rig.assert_reply("aut", "ok 1");
    assert_eq!(rig.firmware_nvram_read(7), 0x02);
    // The password's last byte, not the configuration byte.
    assert_eq!(rig.firmware_nvram_read(4), b'e');
    rig.assert_reply("aut x", "err arg");
    rig.assert_reply("aut", "ok 1");

    // The NVRAM, not the controller, holds it: a start leaves it set.
    rig.elsc.power_on();
    rig.assert_reply("aut", "ok 1");
    rig.assert_reply("aut  0 junk", "ok ");
    rig.assert_reply("aut", "ok 0");
    assert_eq!(rig.firmware_nvram_read(7), 0);
    assert_eq!(rig.firmware_nvram_read(0), 0x37);
}

// ---- hbt -------------------------------------------------------------------

#[test]
fn hbt_forms() {
    let rig = Rig::new();
    // The hex reader takes no digits from "xy" and so reads no value: a
    // heartbeat, not a bad argument. "hbt 00045" is the four digits 0004.
    for ok in [
        "hbt", "hbt 0", "hbt 30 20", "hbt 5", "hbt 258", "hbt 1e 0", "hbt xy", "hbt 30 xy", "hbt 1e  14",
        "hbt 025814",
    ] {
        let resp = rig.transact(&frame(ok.as_bytes()));
        assert_eq!(resp[0], 3, "{ok}");
        assert_eq!(text(&resp), "ok ");
    }
    for bad in ["hbt 4", "hbt 259", "hbt 1000 5", "hbt 00045"] {
        let resp = rig.transact(&frame(bad.as_bytes()));
        assert_eq!(resp[0], 7, "{bad}");
        assert_eq!(text(&resp), "err arg");
    }
    rig.assert_reply("hbtx", "err cmd");
}

#[test]
fn hbt_stages_warn_pulse_and_cycle() {
    let rig = Rig::new();
    rig.assert_reply("pwr q", "ok 1");
    rig.assert_reply("hbt 5 a", "ok ");
    rig.events();
    rig.clock.advance_ms(9999);
    rig.elsc.poll();
    assert_eq!(count_nmi(&rig.events()), 0);
    rig.step_ms(1);
    rig.elsc.poll();
    assert_eq!(count_nmi(&rig.events()), 1);
    rig.assert_reply("pwr q", "ok 0");

    // The second value, 0xa, is the delay to the power cycle.
    rig.clock.advance_ms(9999);
    rig.assert_reply("pwr q", "ok 0");
    rig.step_ms(1);
    rig.assert_reply("pwr", "ok d");
    assert_eq!(power_events(&rig.events()), [false]);
    rig.step_secs(CYCLE_SECS);
    rig.assert_reply("pwr", "ok u");
    assert_eq!(power_events(&rig.events()), [true]);
    rig.assert_reply("pwr q", "ok 1");

    // Idle now: neither time nor a heartbeat brings another NMI.
    rig.step_secs(60);
    rig.assert_reply("hbt", "ok ");
    rig.step_secs(60);
    assert_eq!(count_nmi(&rig.events()), 0);
    rig.assert_reply("pwr q", "ok 0");
}

#[test]
fn a_heartbeat_in_the_warning_postpones_the_nmi() {
    let rig = Rig::new();
    rig.assert_reply("hbt 5 0", "ok ");
    rig.events();
    rig.step_secs(6);
    rig.assert_reply("hbt", "ok ");
    rig.clock.advance_ms(4999);
    rig.elsc.poll();
    assert_eq!(count_nmi(&rig.events()), 0);
    rig.step_ms(1);
    rig.elsc.poll();
    assert_eq!(count_nmi(&rig.events()), 1);
    // A delay of zero never power-cycles.
    rig.step_secs(120);
    assert!(rig.elsc.module_powered());
    rig.assert_reply("pwr q", "ok 1");
}

#[test]
fn hbt_is_stopped_only_by_clr_or_zero() {
    let rig = Rig::new();
    rig.assert_reply("hbt 5", "ok ");
    rig.step_secs(3);
    rig.assert_reply("clr", "ok ");
    rig.step_secs(20);
    rig.assert_reply("hbt 5", "ok ");
    rig.assert_reply("hbt 0", "ok ");
    rig.step_secs(20);
    rig.elsc.poll();
    assert_eq!(count_nmi(&rig.events()), 0);

    // A "rst" does not stop it: the controller rides through the reset.
    rig.assert_reply("hbt 5", "ok ");
    rig.step_secs(1);
    rig.assert_reply("rst", "ok ");
    rig.step_secs(1);
    assert!(rig.events().contains(&ModuleOutput::Reset));
    rig.module_reset();
    rig.clock.advance_ms(7999);
    rig.elsc.poll();
    assert_eq!(count_nmi(&rig.events()), 0);
    rig.step_ms(1);
    rig.elsc.poll();
    assert_eq!(count_nmi(&rig.events()), 1);
    rig.assert_reply("hbt 0", "ok ");
}

// ---- rst and the deferred-action rules --------------------------------------

#[test]
fn rst_resets_the_module_not_the_controller() {
    let rig = Rig::new();
    // Leave a mark the reset must clear: take the test-and-set lock.
    rig.assert_reply("tas A", "ok 0");
    rig.assert_reply("rst", "ok ");
    rig.step_secs(1);
    assert!(rig.events().contains(&ModuleOutput::Reset));
    rig.module_reset();
    // It freed the lock the controller was holding.
    rig.assert_reply("tas B", "ok 0");
}

#[test]
fn rst_waits_for_the_ack_to_be_read() {
    let rig = Rig::new();
    // Write "rst" and end the transaction without reading the response.
    rig.request_without_read("rst");
    rig.step_secs(5);
    rig.elsc.poll();
    assert!(!rig.events().contains(&ModuleOutput::Reset));

    // The ack survives to be fetched in a separate read transaction.
    let resp = rig.read_response();
    assert_eq!(byte_sum(&resp), 0);
    assert_eq!(resp[0], 3);
    assert_eq!(text(&resp), "ok ");
    // Only now, once the ack has been read, does the reset wait begin.
    rig.step_secs(1);
    assert!(rig.events().contains(&ModuleOutput::Reset));
}

#[test]
fn rst_is_dropped_when_its_ack_is_superseded_unread() {
    let rig = Rig::new();
    rig.assert_reply("tas A", "ok 0");
    rig.request_without_read("rst");
    // A new request supersedes the ack; its own response reads back clean.
    rig.assert_reply("ver", "ok V 3.1S  ");
    // No reset ran: the lock is still held.
    rig.step_secs(5);
    rig.assert_reply("tas B", "ok A");
    assert!(!rig.events().contains(&ModuleOutput::Reset));
}

#[test]
fn rst_fires_without_a_stop() {
    let rig = Rig::new();
    rig.request_without_read("rst");
    // Read the whole ack in a read transaction, issuing no STOP.
    rig.begin_read();
    let resp = rig.read_frame(false);
    assert_eq!(byte_sum(&resp), 0);
    assert_eq!(text(&resp), "ok ");
    // Reading the last ack byte committed the reset.
    rig.step_secs(1);
    assert!(rig.events().contains(&ModuleOutput::Reset));
}

#[test]
fn rst_survives_a_nack_on_the_last_text_byte() {
    let rig = Rig::new();
    rig.request_without_read("rst");
    // Read through the last text byte, NAKing it; the checksum is unread.
    rig.begin_read();
    assert_eq!(rig.r(DATA), 3);
    assert_eq!(rig.r(DATA), b'o');
    assert_eq!(rig.r(DATA), b'k');
    rig.w(CTRL, ctl::ESO);
    assert_eq!(rig.r(DATA), b' ');

    // No STOP: a same-address repeated START carrying an unrelated "ver"
    // must not cancel the reset the guest already read the "ok" of.
    rig.w(CTRL, ctl::ESO | ctl::STA | ctl::ACK);
    rig.w(DATA, ELSC_ADDR << 1);
    for b in frame(b"ver") {
        rig.w(DATA, b);
    }
    rig.w(CTRL, ctl::ESO | ctl::STA | ctl::ACK);
    rig.w(DATA, (ELSC_ADDR << 1) | 1);
    rig.step_secs(1);
    assert!(rig.events().contains(&ModuleOutput::Reset));
}

#[test]
fn rst_releases_the_i2c_bus() {
    let rig = Rig::new();
    rig.request_without_read("rst");
    // Read the whole ack, NAK the last byte, issue no STOP.
    rig.begin_read();
    let resp = rig.read_frame(false);
    assert_eq!(text(&resp), "ok ");
    rig.step_secs(1);
    assert!(rig.events().contains(&ModuleOutput::Reset));
    rig.module_reset();

    // Re-enable and read identity-NVRAM page 7, offset 0, directly. With the
    // bus released this routes to the NVRAM and byte 0 is the 0x37 marker; with
    // the controller left selected it would answer as the controller.
    rig.w(CTRL, ctl::PIN | ctl::ESO);
    rig.w(DATA, NVRAM_PAGE7 << 1);
    rig.w(CTRL, ctl::PIN | ctl::ESO | ctl::STA | ctl::ACK);
    rig.w(DATA, 0);
    rig.w(CTRL, ctl::ESO | ctl::STA | ctl::ACK);
    rig.w(DATA, (NVRAM_PAGE7 << 1) | 1);
    let _ = rig.r(DATA);
    rig.w(CTRL, ctl::ESO);
    assert_eq!(rig.r(DATA), 0x37);
    rig.close_with_invalid_address();
}

#[test]
fn a_committed_action_ends_the_transfer_that_selected_the_controller() {
    // No adapter reset follows: the action itself must end the transfer, or
    // the grant ring would see a busy bus for ever.
    let rig = Rig::new();
    rig.request_without_read("pwr d 0");
    rig.begin_read();
    assert!(rig.busy.load(Ordering::Relaxed));
    // The "ok" is read in full and the last byte NAKed; no STOP follows.
    let resp = rig.read_frame(false);
    assert_eq!(text(&resp), "ok ");
    assert_eq!(power_events(&rig.events()), [false]);
    rig.r(CTRL);
    assert!(!rig.busy.load(Ordering::Relaxed));
}

#[test]
fn rst_waits_one_second_and_a_second_rst_restarts_the_wait() {
    let rig = Rig::new();
    rig.assert_reply("tas A", "ok 0");
    rig.assert_reply("fan h", "ok ");
    rig.assert_reply("rst", "ok ");
    rig.clock.advance_ms(999);
    rig.assert_reply("tas", "ok A");
    rig.assert_reply("rst", "ok ");
    rig.clock.advance_ms(999);
    rig.assert_reply("tas", "ok A");
    rig.step_ms(1);
    rig.elsc.poll();
    assert!(rig.events().contains(&ModuleOutput::Reset));
    rig.module_reset();
    rig.assert_reply("tas", "ok 0");
    // The controller rides through the reset it caused.
    rig.assert_reply("fan", "ok h");

    // The countdown keeps running through the reset.
    rig.assert_reply("pwr d 3", "ok ");
    rig.assert_reply("rst", "ok ");
    rig.step_secs(1);
    rig.module_reset();
    assert!(rig.elsc.module_powered());
    rig.step_secs(2);
    assert_eq!(power_events(&rig.events()), [false]);

    // A reset from outside is the controller's own start and clears the fan
    // request.
    let rig = Rig::new();
    rig.assert_reply("fan h", "ok ");
    rig.elsc.power_on();
    rig.assert_reply("fan", "ok n");
}

#[test]
fn a_power_cycle_inside_the_rst_wait_does_not_cancel_it() {
    let rig = Rig::new();
    rig.assert_reply("rst", "ok ");
    rig.clock.advance_ms(300);
    rig.assert_reply("pwr c", "ok ");
    rig.assert_reply("tas A", "ok 0");
    rig.clock.advance_ms(699);
    rig.assert_reply("tas", "ok A");
    rig.step_ms(1);
    rig.elsc.poll();
    assert!(rig.events().contains(&ModuleOutput::Reset));
}

#[test]
fn a_committed_action_survives_a_second_destructive_command() {
    let rig = Rig::new();
    rig.request_without_read("pwr d 0");
    // Read the "ok" through its last text byte and keep the frame open.
    rig.begin_read();
    assert_eq!(rig.r(DATA), 3);
    assert_eq!(rig.r(DATA), b'o');
    assert_eq!(rig.r(DATA), b'k');
    assert_eq!(rig.r(DATA), b' ');
    // A "rst" arrives while the power-down is committed: it still answers
    // "ok" but the power-down fires.
    rig.w(CTRL, ctl::ESO | ctl::STA | ctl::ACK);
    rig.w(DATA, ELSC_ADDR << 1);
    for b in frame(b"rst") {
        rig.w(DATA, b);
    }
    rig.w(CTRL, ctl::ESO | ctl::STA | ctl::ACK);
    rig.w(DATA, (ELSC_ADDR << 1) | 1);
    assert_eq!(power_events(&rig.events()), [false]);
    assert!(!rig.elsc.module_powered());
}

// ---- pwr -------------------------------------------------------------------

#[test]
fn pwr_refusals_take_no_action() {
    let rig = Rig::new();
    for bad in ["pwr d 259", "pwr d ffff", "pwr c 4", "pwr c 259", "pwr r"] {
        let resp = rig.transact(&frame(bad.as_bytes()));
        assert_eq!(resp[0], 7, "{bad}");
        assert_eq!(text(&resp), "err arg");
    }
    rig.step_secs(1000);
    rig.elsc.poll();
    assert!(rig.events().is_empty());
}

#[test]
fn an_immediate_pwr_d_powers_down_once_the_ack_is_read() {
    let rig = Rig::new();
    rig.assert_reply("pwr d 0", "ok ");
    assert_eq!(power_events(&rig.events()), [false]);
    assert_eq!(rig.reply("pwr"), "ok d");
}

#[test]
fn pwr_c_powers_off_then_back_after_the_interval_and_hold() {
    let rig = Rig::new();
    rig.assert_reply("pwr c", "ok ");
    assert_eq!(power_events(&rig.events()), [false]);
    rig.clock.advance_ms(CYCLE_SECS * SECOND - 1);
    rig.elsc.poll();
    assert!(power_events(&rig.events()).is_empty());
    rig.step_ms(1);
    rig.elsc.poll();
    assert_eq!(power_events(&rig.events()), [true]);
}

#[test]
fn pwr_d_range_is_checked() {
    let rig = Rig::new();
    for ok in ["pwr d 1", "pwr d 4", "pwr d 258"] {
        let resp = rig.transact(&frame(ok.as_bytes()));
        assert_eq!(resp[0], 3);
        assert_eq!(text(&resp), "ok ");
    }
    rig.assert_reply("pwr d 259", "err arg");
}

#[test]
fn pwr_d_countdown_expires() {
    let rig = Rig::new();
    rig.assert_reply("pwr d 1", "ok ");
    rig.clock.advance_ms(999);
    rig.elsc.poll();
    assert!(power_events(&rig.events()).is_empty());
    rig.step_ms(1);
    rig.elsc.poll();
    assert_eq!(power_events(&rig.events()), [false]);
}

#[test]
fn each_timed_pwr_d_reloads_the_countdown() {
    let rig = Rig::new();
    rig.assert_reply("pwr d 8", "ok ");
    rig.step_secs(7);
    rig.assert_reply("pwr d 8", "ok ");
    // The deadline is now 15 s, not the first request's 8 s.
    rig.clock.advance_ms(7999);
    rig.elsc.poll();
    assert!(power_events(&rig.events()).is_empty());
    rig.step_ms(1);
    rig.elsc.poll();
    assert_eq!(power_events(&rig.events()), [false]);

    let rig = Rig::new();
    rig.assert_reply("pwr d 8", "ok ");
    rig.step_secs(1);
    rig.assert_reply("pwr d 2", "ok ");
    // Pulled in to 3 s from 8 s.
    rig.clock.advance_ms(1999);
    rig.elsc.poll();
    assert!(power_events(&rig.events()).is_empty());
    rig.step_ms(1);
    rig.elsc.poll();
    assert_eq!(power_events(&rig.events()), [false]);
}

#[test]
fn pwr_d_argument_parsing() {
    let rig = Rig::new();
    // 0x1234 is above the limit and arms nothing.
    rig.assert_reply("pwr d 12345", "err arg");
    rig.assert_reply("pwr d 259", "err arg");
    rig.step_secs(700);
    rig.elsc.poll();
    assert!(rig.elsc.module_powered());

    // One separator of any kind, then the digits: "pwr d-1x" is one second.
    rig.assert_reply("pwr d-1x", "ok ");
    rig.clock.advance_ms(999);
    rig.elsc.poll();
    assert!(rig.elsc.module_powered());
    rig.step_ms(1);
    rig.elsc.poll();
    assert!(!rig.elsc.module_powered());

    // "pwr d 00001" reads 0000 and powers down at once.
    let rig = Rig::new();
    rig.assert_reply("pwr d 00001", "ok ");
    assert_eq!(power_events(&rig.events()), [false]);
}

#[test]
fn pwr_d_without_a_delay() {
    // Three-second countdown when none is pending...
    let rig = Rig::new();
    rig.assert_reply("pwr d", "ok ");
    rig.step_secs(2);
    rig.elsc.poll();
    assert!(rig.elsc.module_powered());
    rig.step_secs(1);
    rig.elsc.poll();
    assert!(!rig.elsc.module_powered());

    // ...and immediate when one is; text that is not hex counts as no delay.
    let rig = Rig::new();
    rig.assert_reply("pwr d 3c", "ok ");
    assert!(rig.elsc.module_powered());
    rig.assert_reply("pwr d zz", "ok ");
    assert!(!rig.elsc.module_powered());
}

#[test]
fn pwr_c_re_arms_the_cold_power_latch() {
    let rig = Rig::new();
    rig.assert_reply("pwr q", "ok 1");
    rig.assert_reply("pwr q", "ok 0");
    rig.assert_reply("pwr c", "ok ");
    rig.assert_reply("pwr q", "ok 0");
    rig.step_secs(CYCLE_SECS);
    // The cycle re-armed the latch: the first post-cycle query reads 1.
    let resp = rig.transact(&frame(b"pwr q"));
    assert_eq!(resp[0], 4);
    assert_eq!(text(&resp), "ok 1");
}

#[test]
fn an_unpowered_module_keeps_answering_and_refuses_what_needs_power() {
    let rig = Rig::new();
    rig.assert_reply("pwr c 6", "ok ");
    rig.step_ms(1);
    rig.assert_reply("pwr", "ok d");
    rig.assert_reply("rst", "err cmd");
    rig.assert_reply("nmi", "err cmd");
    // "pwr d" has nothing to do and does no range check.
    rig.assert_reply("pwr d 1", "ok ");
    rig.assert_reply("pwr d 259", "ok ");
    rig.assert_reply("clr", "ok ");
    assert!(rig.elsc.describe().contains("module power: off"));
    // The Origin 200 build keeps publishing while the module is off.
    rig.observer();
    rig.wait_for_absence();
    rig.wait_for_signal();

    // Six seconds off, then four of power-up with reset held.
    rig.advance_to_ms(9999);
    rig.assert_reply("pwr", "ok d");
    rig.advance_to_ms(10_001);
    rig.assert_reply("pwr", "ok u");

    // "pwr u" cuts the off interval short; power is back four seconds on.
    rig.assert_reply("pwr c 258", "ok ");
    rig.step_secs(1);
    rig.assert_reply("pwr u", "ok ");
    rig.clock.advance_ms(3999);
    rig.assert_reply("pwr", "ok d");
    rig.step_ms(1);
    rig.assert_reply("pwr", "ok u");
}

#[test]
fn pwr_d_posts_pd_remote_to_the_hubs() {
    let rig = Rig::new();
    rig.observer();
    rig.assert_reply("pwr d 259", "err arg");
    assert!(!rig.elsc.describe().contains("controller message"));

    rig.assert_reply("pwr d 8", "ok ");
    rig.wait_for_absence();
    assert_eq!(rig.wait_for_signal(), SIGNAL2);
    rig.assert_reply("get", "pd remote");
    rig.wait_for_absence();
    assert_eq!(rig.wait_for_signal(), SIGNAL1);
    rig.assert_reply("get", "acp . ");

    // Unfetched, it lapses after a second.
    rig.assert_reply("pwr d 8", "ok ");
    rig.step_ms(1000);
    rig.wait_for_absence();
    assert_eq!(rig.wait_for_signal(), SIGNAL1);
    rig.assert_reply("get", "acp . ");

    // The soft power-down tells the Hubs too.
    rig.assert_reply("clr", "ok ");
    rig.assert_reply("pwr d", "ok ");
    assert!(rig.elsc.describe().contains("controller message: \"pd remote\""));
    rig.assert_reply("clr", "ok ");
}

/// The controller says when it next has an output to give, and not for the
/// grant ring's ticks: an idle controller is never due, a pending NMI is due
/// when its wait ends and its pulse when that ends, and a power-down
/// countdown is due when it runs out.
#[test]
fn the_controller_reports_when_it_is_next_due() {
    let ms = Duration::from_millis;
    let rig = Rig::new();
    assert_eq!(rig.elsc.next_due(), None, "the ring ticks, but nothing is due");
    rig.assert_reply("nmi", "ok ");
    assert_eq!(rig.elsc.next_due(), Some(ms(1000)));
    rig.step_ms(400);
    assert_eq!(rig.elsc.next_due(), Some(ms(600)));
    rig.step_ms(600);
    assert_eq!(rig.elsc.next_due(), Some(ms(0)), "due now");
    rig.elsc.poll();
    assert_eq!(rig.events(), [ModuleOutput::Nmi(true)]);
    assert_eq!(rig.elsc.next_due(), Some(ms(100)), "the pulse ends");
    rig.step_ms(100);
    rig.elsc.poll();
    assert_eq!(rig.events(), [ModuleOutput::Nmi(false)]);
    assert_eq!(rig.elsc.next_due(), None);

    rig.assert_reply("pwr d 3", "ok ");
    let countdown = rig.elsc.next_due().expect("a power-down is counting down");
    assert!(countdown > ms(2900) && countdown <= ms(3000), "{countdown:?}");
    rig.step_secs(3);
    rig.elsc.poll();
    assert_eq!(power_events(&rig.events()), [false]);
}

/// A bus access that passes a deadline before the machine's visit queues the
/// output itself and clears the timer. The controller stays due until the
/// outputs are taken, or the power-off, reset or NMI would wait for the next
/// unrelated access.
#[test]
fn an_access_that_beats_the_visit_leaves_its_outputs_due() {
    let rig = Rig::new();
    rig.assert_reply("nmi", "ok ");
    rig.step_ms(1100);
    rig.assert_reply("ver", "ok V 3.1S  ");
    assert_eq!(rig.elsc.next_due(), Some(Duration::ZERO), "the queued outputs are due now");
    assert_eq!(rig.events(), [ModuleOutput::Nmi(true), ModuleOutput::Nmi(false)]);
    assert_eq!(rig.elsc.next_due(), None, "taken, so nothing is due");
}

/// An access that moves the schedule wakes the machine, and one that leaves
/// it alone does not.
#[test]
fn an_access_wakes_the_machine_only_when_the_schedule_moved() {
    let rig = Rig::new();
    let woken = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let w = woken.clone();
    rig.elsc.set_wake(Box::new(move || {
        w.fetch_add(1, Ordering::SeqCst);
    }));
    rig.assert_reply("ver", "ok V 3.1S  ");
    let after_first = woken.load(Ordering::SeqCst);
    rig.assert_reply("ver", "ok V 3.1S  ");
    assert_eq!(woken.load(Ordering::SeqCst), after_first, "a query changes nothing");
    rig.assert_reply("nmi", "ok ");
    assert!(woken.load(Ordering::SeqCst) > after_first, "an NMI request arms a timer");
}

#[test]
fn nmi_pulses_after_one_second_for_100_ms_and_frees_the_lock() {
    let rig = Rig::new();
    rig.assert_reply("tas A", "ok 0");
    rig.assert_reply("nmi", "ok ");
    rig.clock.advance_ms(999);
    rig.assert_reply("nmi junk", "ok ");
    rig.clock.advance_ms(999);
    assert_eq!(count_nmi(&rig.events()), 0);
    rig.step_ms(1);
    rig.elsc.poll();
    assert_eq!(rig.events(), [ModuleOutput::Nmi(true)]);
    rig.assert_reply("tas", "ok A");
    rig.clock.advance_ms(99);
    rig.elsc.poll();
    assert!(rig.events().is_empty());
    rig.step_ms(1);
    rig.elsc.poll();
    assert_eq!(rig.events(), [ModuleOutput::Nmi(false)]);
    rig.assert_reply("tas", "ok 0");

    // A power cycle inside the wait does not cancel it, but the pulse then
    // reaches no processor of the unpowered module: the lock is still freed.
    rig.assert_reply("nmi", "ok ");
    rig.clock.advance_ms(500);
    rig.assert_reply("pwr c", "ok ");
    rig.assert_reply("tas C", "ok 0");
    rig.clock.advance_ms(500);
    rig.assert_reply("tas", "ok 0");
    assert_eq!(count_nmi(&rig.events()), 0);
}

// ---- start-up power-up ------------------------------------------------------

#[test]
fn start_delay_waits_then_powers_up_when_the_nvram_says_so() {
    let rig = Rig::with(ElscConfig { start_delay_ms: 500, ..Default::default() });
    rig.assert_reply("pwr", "ok d");
    assert!(rig.elsc.power_return_pending());
    rig.clock.advance_ms(499);
    rig.assert_reply("pwr", "ok d");
    rig.step_ms(1);
    // The 4 s power-up reset hold follows.
    rig.clock.advance_ms(3999);
    rig.assert_reply("pwr", "ok d");
    rig.step_ms(1);
    rig.assert_reply("pwr", "ok u");
    assert_eq!(power_events(&rig.events()), [true]);
    assert!(!rig.elsc.power_return_pending());
}

#[test]
fn start_delay_stays_off_without_automatic_power_on() {
    let rig = Rig::with(ElscConfig { start_delay_ms: 100, ..Default::default() });
    rig.assert_reply("aut 0", "ok ");
    rig.step_secs(10);
    rig.assert_reply("pwr", "ok d");
    // The controller says again that the power is off, so the machine asks
    // whether anything is left to bring a module back.
    assert_eq!(power_events(&rig.events()), [false]);
    assert!(!rig.elsc.power_return_pending());
}

// ---- AUX serial line ---------------------------------------------------------

#[test]
fn the_start_announces_the_version_on_the_serial_line() {
    let rig = Rig::new();
    assert_eq!(rig.aux.take(), BANNER);
}

#[test]
fn serial_command_line() {
    let rig = Rig::new();
    rig.aux.take();
    rig.serial_command("ver", "ok V 3.1S  ");
    // Delete and backspace remove the last character.
    rig.type_bytes(b"\x14verx\x7f\r");
    assert_eq!(rig.aux.take(), "\r\nMSC> verx\x08 \x08\r\n\x14ok V 3.1S  \r\n");
    // Enter alone answers "ok" with echo on.
    rig.serial_command("", "ok");
    // Only a carriage return separates a bare mnemonic.
    rig.type_bytes(b"\x14ver\n");
    assert_eq!(rig.aux.take(), "\r\nMSC> ver\n\x14err cmd\r\n");
    rig.type_bytes(b"\x14pwr q\n");
    assert_eq!(rig.aux.take(), "\r\nMSC> pwr q\n\x14ok 1\r\n");
    rig.serial_command("acp 0 x", "err cmd");
    rig.serial_command("get", "err cmd");
    // The serial "fsc" keeps the digit the I2C reply loses.
    rig.serial_command("fsc", "ok0");
    rig.serial_command("fsc 1", "ok");
    rig.serial_command("fsc x", "ok1");

    // A second Ctrl+T abandons the line, keeps it and passes one on.
    rig.type_bytes(b"\x14ve\x14");
    assert_eq!(rig.aux.take(), "\r\nMSC> ve\x08\x08 \x08\x08 \x08\x08 \x08\x08 \x08");
    rig.type_bytes(b"\x14r\r");
    assert_eq!(rig.aux.take(), "\r\nMSC> r\r\n\x14ok V 3.1S  \r\n");
    // A terminal's line feed after the return goes to the processors.
    rig.type_bytes(b"\x14tmp\r\n");
    assert_eq!(rig.aux.take(), "\r\nMSC> tmp\r\n\x14ok n\r\n");
    assert!(rig.elsc.describe().contains("input: \"\\x14\\x0a\""));

    // Processor output arriving while a line is typed is lost.
    rig.type_bytes(b"\x14v");
    assert_eq!(rig.aux.take(), "\r\nMSC> v");
    rig.assert_reply("acp 0 hi", "ok 2");
    assert_eq!(rig.aux.take(), "");
    rig.type_bytes(b"er\r");
    assert_eq!(rig.aux.take(), "er\r\n\x14ok V 3.1S  \r\n");
    rig.assert_reply("acp 0 yo", "ok 2");
    assert_eq!(rig.aux.take(), "yo");

    // "cfg" prints the NVRAM raw, here its first byte, a zero.
    rig.type_bytes(b"\x14cfg\r");
    assert_eq!(rig.aux.take(), "\r\nMSC> cfg\r\n\0\r\n");

    // Echo off: no prompt, no echo, and Enter alone is "err cmd".
    rig.serial_command("ech 0", "ok");
    rig.type_bytes(b"\x14\r");
    assert_eq!(rig.aux.take(), "\x14err cmd\r\n");
    rig.type_bytes(b"\x14ech\r");
    assert_eq!(rig.aux.take(), "\x14ok\r\n");
    rig.serial_command("ech 2", "err arg");
}

#[test]
fn a_serial_line_waits_for_the_bus() {
    let rig = Rig::new();
    rig.aux.take();
    // An I2C transfer to the controller is open: the line queues behind it.
    rig.w(CTRL, ctl::PIN | ctl::ESO);
    rig.w(DATA, ELSC_ADDR << 1);
    rig.w(CTRL, ctl::PIN | ctl::ESO | ctl::STA | ctl::ACK);
    rig.type_bytes(b"\x14ver\r");
    assert_eq!(rig.aux.take(), "\r\nMSC> ver\r\n");
    rig.w(CTRL, ctl::PIN | ctl::ESO | ctl::STO);
    rig.elsc.poll();
    assert_eq!(rig.aux.take(), "\x14ok V 3.1S  \r\n");
}

#[test]
fn a_slave_answers_ctrl_t_with_the_prompt_and_takes_no_command() {
    let rig = Rig::with(ElscConfig { master: false, ..Default::default() });
    rig.aux.take();
    rig.type_bytes(b"\x14");
    assert_eq!(rig.aux.take(), "\r\nMSC> Slave!\r\n");
    rig.type_bytes(b"ver\r");
    assert_eq!(rig.aux.take(), "");
}

#[test]
fn serial_supervisor_gates_the_destructive_commands() {
    let rig = Rig::new();
    rig.aux.take();
    rig.serial_command("hbt 0", "err perm");
    rig.serial_command("fan h", "err perm");
    rig.serial_command("pwr d 8", "err perm");
    rig.serial_command("pwr q", "ok 1");
    rig.serial_command("mod 7", "err perm");
    rig.serial_command("mod", "ok 0");
    rig.serial_command("pas abc", "err arg");
    rig.serial_command("pas nope", "err arg");
    rig.serial_command("pas s abcd", "err perm");
    rig.serial_command("pas none", "ok");
    rig.serial_command("hbt 0", "ok");
    rig.serial_command("mod 7", "ok");
    rig.serial_command("mod", "ok 7");
    rig.serial_command("nvm 708", "ok 7");
    rig.serial_command("nvm 705 3", "ok");
    rig.serial_command("dbg", "ok 3 0");
    rig.serial_command("dbg 1 2", "ok");
    rig.serial_command("dbg", "ok 1 2");
    rig.serial_command("fre", "err cmd");

    // On the serial line the fan request is kept in the NVRAM too.
    rig.serial_command("fan h", "ok");
    rig.serial_command("nvm 707", "ok 3");

    // The Hub message goes out on the line as it is queued.
    rig.type_bytes(b"\x14pwr d 258\r");
    assert_eq!(
        rig.aux.take(),
        "\r\nMSC> pwr d 258\r\n\x14\r\nMSC: pd remote\x14\r\x14ok\r\n"
    );
    // A line waits while a Hub message is pending; it lapses after a second.
    rig.step_secs(1);

    rig.serial_command("pas s ab", "err arg");
    rig.serial_command("pas s  abcd", "err arg");
    rig.serial_command("pas s wxyz12", "ok");
    rig.serial_command("nvm 701", "ok 77");
    rig.serial_command("pas", "ok");
    rig.serial_command("hbt 0", "err perm");
    rig.serial_command("pas none", "err arg");
    rig.serial_command("pas wxyz", "ok");
    // A host's "clr" over I2C ends the session.
    rig.assert_reply("clr", "ok ");
    rig.aux.take();
    rig.serial_command("hbt 0", "err perm");
    rig.serial_command("rst", "err perm");
    assert_eq!(&rig.nvram.contents()[0x701..0x705], b"wxyz");
}

#[test]
fn a_serial_nvm_to_a_missing_device_prints_a_bare_err() {
    let rig = Rig::new();
    rig.aux.take();
    // Device 0xa0 | (8 << 1) is 0x58, which no part answers.
    rig.serial_command("nvm 800", "err ");
}

#[test]
fn a_serial_reset_runs_once_its_reply_is_printed() {
    let rig = Rig::new();
    rig.aux.take();
    rig.serial_command("pas none", "ok");
    rig.type_bytes(b"\x14rst\r");
    let out = rig.aux.take();
    assert!(out.ends_with("\x14ok\r\n\x14\r\nMSC: rst\x14\r"), "{out:?}");
    rig.step_secs(1);
    assert!(rig.events().contains(&ModuleOutput::Reset));
}

#[test]
fn the_serial_input_queue_overwrites_its_oldest_byte() {
    let rig = Rig::new();
    rig.observer();
    // Input waits for the first processor to write before it is announced.
    rig.type_bytes(b"x");
    rig.wait_for_absence();
    assert_eq!(rig.wait_for_signal(), SIGNAL1);
    rig.assert_reply("acp 0 ", "ok 0");
    rig.wait_for_absence();
    assert_eq!(rig.wait_for_signal(), SIGNAL2);

    let mut line = vec![b'a'; 300];
    // The first byte of the overflow is the one that must be lost.
    line[0] = b'b';
    rig.type_bytes(&line);
    rig.type_bytes(b"z");
    let info = rig.elsc.describe();
    assert!(info.contains("rx queue: 256 byte(s)"), "{info}");
    assert!(info.contains("input: \"aaaa"), "{info}");
}

/// The exchanges the Origin 200 PROM makes before the System Maintenance
/// Menu, in order, with the frames spelled out byte for byte: a request and
/// the reply it accepts. Replies it discards (`dsp`, `dsc`) are `err cmd`,
/// which it tolerates.
#[test]
fn the_prom_exchanges_get_the_frames_it_accepts() {
    const ERR_CMD: [u8; 9] = [0x07, 0x65, 0x72, 0x72, 0x20, 0x63, 0x6d, 0x64, 0x5c];
    let exchanges: &[(&[u8], &[u8])] = &[
        // dsp and eight blanks; dsp P 0 M 0
        (&[0x0b, 0x64, 0x73, 0x70, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0xae], &ERR_CMD),
        (&[0x0c, 0x64, 0x73, 0x70, 0x20, 0x50, 0x20, 0x30, 0x20, 0x4d, 0x20, 0x30, 0x20, 0x10], &ERR_CMD),
        // rsw: ok ff
        (&[0x03, 0x72, 0x73, 0x77, 0xa1], &[0x05, 0x6f, 0x6b, 0x20, 0x66, 0x66, 0x35]),
        // pwr q: ok 1
        (&[0x05, 0x70, 0x77, 0x72, 0x20, 0x71, 0x11], &[0x04, 0x6f, 0x6b, 0x20, 0x31, 0xd1]),
        // rsw again, then ver: ok V 3.1S
        (&[0x03, 0x72, 0x73, 0x77, 0xa1], &[0x05, 0x6f, 0x6b, 0x20, 0x66, 0x66, 0x35]),
        (
            &[0x03, 0x76, 0x65, 0x72, 0xb0],
            &[0x0b, 0x6f, 0x6b, 0x20, 0x56, 0x20, 0x33, 0x2e, 0x31, 0x53, 0x20, 0x20, 0x60],
        ),
        // dsc 1 e; dsc 2 0
        (&[0x07, 0x64, 0x73, 0x63, 0x20, 0x31, 0x20, 0x65, 0xe9], &ERR_CMD),
        (&[0x07, 0x64, 0x73, 0x63, 0x20, 0x32, 0x20, 0x30, 0x1d], &ERR_CMD),
        // acp 0 Starting PROM Boot process..: the count is hex and nonzero,
        // or the PROM sends the same characters again for ever. 0x1c is 28.
        (
            &[
                0x22, 0x61, 0x63, 0x70, 0x20, 0x30, 0x20, 0x53, 0x74, 0x61, 0x72, 0x74, 0x69, 0x6e, 0x67, 0x20,
                0x50, 0x52, 0x4f, 0x4d, 0x20, 0x42, 0x6f, 0x6f, 0x74, 0x20, 0x70, 0x72, 0x6f, 0x63, 0x65, 0x73,
                0x73, 0x2e, 0x2e, 0x61,
            ],
            &[0x05, 0x6f, 0x6b, 0x20, 0x31, 0x63, 0x6d],
        ),
        // acp 0 x: ok 1
        (&[0x07, 0x61, 0x63, 0x70, 0x20, 0x30, 0x20, 0x78, 0xdd], &[0x04, 0x6f, 0x6b, 0x20, 0x31, 0xd1]),
    ];
    let rig = Rig::new();
    for (request, reply) in exchanges {
        assert_eq!(&rig.transact(request), reply, "request {request:02x?}");
    }
}

#[test]
fn an_acp_reply_never_counts_more_than_was_sent_nor_zero_for_text() {
    let rig = Rig::new();
    // The PROM sends up to 86 characters per exchange. A request that carries
    // more (up to the 94-byte frame) is cut to 86 and the count says so, so
    // the sender resends only the rest.
    for sent in [1usize, 2, 15, 16, 85, 86, 88] {
        let reply = rig.reply(&format!("acp 0 {}", "x".repeat(sent)));
        let count = usize::from_str_radix(reply.strip_prefix("ok ").unwrap(), 16).unwrap();
        assert!(count >= 1 && count <= sent && count <= 86, "{sent}: {reply}");
    }
}

#[test]
fn replies_stay_below_the_prom_length_limit() {
    // The PROM rejects a reply of 96 bytes or more, or one whose byte count
    // is not length plus two.
    let rig = Rig::new();
    for text in ["ver", "rsw", "get", "tmp", "tas A", "pwr q", "acp 0 abc", "xyz"] {
        let resp = rig.transact(&frame(text.as_bytes()));
        assert!(resp.len() < 96 + 2 && resp.len() == resp[0] as usize + 2, "{text}");
        assert_eq!(byte_sum(&resp), 0);
    }
}

// ---- the Origin 2000 build, with no module owner ---------------------------

fn rig_2000() -> Rig {
    Rig::with(ElscConfig { build: Build::Origin2000, master: false, ..Default::default() })
}

#[test]
fn the_origin_2000_build_has_its_own_version_commands_and_display() {
    let rig = rig_2000();
    assert_eq!(rig.aux.take(), "\x14MSC VER 3.2 \r\n");
    rig.assert_reply("ver", "ok VER 3.2 ");
    rig.assert_reply("key", "ok on");
    rig.assert_reply("vlm 5", "ok n");
    rig.assert_reply("vlm 5h", "ok ");
    rig.assert_reply("vlm 5", "ok h");
    rig.assert_reply("dsp hello", "ok ");
    rig.assert_reply("dsc 1 J", "ok ");
    assert!(rig.elsc.describe().contains("display: \"Jello.2 \""), "{}", rig.elsc.describe());
    // The Origin 200 build's own commands are not in this table.
    for absent in ["aut", "mod", "pas none"] {
        rig.assert_reply(absent, "err cmd");
    }
    // The role bit belongs to the Origin 200 build.
    rig.assert_reply("rsw", "ok ff");
    rig.assert_reply("sla o", "ok ");
}

#[test]
fn the_origin_2000_build_holds_reset_4_4_seconds_when_it_powers_up() {
    let rig = rig_2000();
    rig.assert_reply("pwr c 5", "ok ");
    assert_eq!(power_events(&rig.events()), [false]);
    rig.clock.advance_ms(9_399);
    rig.assert_reply("pwr", "ok d");
    rig.step_ms(1);
    rig.assert_reply("pwr", "ok u");
}
