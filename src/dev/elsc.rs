//! Module system controller (MSC, the ELSC) of an Origin 200 or an Origin
//! 2000 module: a checksummed ASCII command and reply protocol over I2C, the
//! arbitration grants the processors wait for, and the AUX serial port.
//!
//! The two builds are one program built two ways. The Origin 200 build
//! (`V 3.1S`, 4 s power-up hold) publishes its own five-position grant ring
//! and has `aut`. The Origin 2000 build (`VER 3.2`, 4.4 s hold) has an
//! eight-character display that status messages replace, and `dsp`, `dsc`,
//! `key`, `vlm` and (on the serial line) `fre`; the commands a build lacks
//! answer `err cmd`. An Origin 2000 controller leaves grants, power and
//! reset to the module that owns it through [`ModuleOps`].
//!
//! The controller is an I2C target at seven-bit address 0x08 on the Hub's
//! bus, next to its [`ElscNvram`]. The module's power, reset and NMI are
//! outputs, reported as [`ModuleOutput`]s the machine drains with
//! [`Elsc::take_events`].
//!
//! No specification of the controller's protocol, firmware or microcontroller
//! is public and no controller has been measured. Every timing, the version
//! text, the grant schedule, the queue depths and the reply text below are
//! the model's own choices, approximations unless a comment says otherwise:
//!
//! * the version fields `V 3.1S` and `VER 3.2` and their padding;
//! * the one-millisecond grant tick, the five ring positions (processor codes
//!   0 to 3, then a fifth published as code 4's byte), and that a busy tick
//!   publishes nothing and does not advance;
//! * the 4 s (Origin 200) and 4.4 s (Origin 2000) power-up reset hold, the
//!   1 s waits of `rst` and `nmi`, the 100 ms NMI pulse, the 1 s life of a controller message, the 5 s warning
//!   and the 5 s off interval of the heartbeat watchdog;
//! * the 256-byte console input queue that overwrites its oldest byte;
//! * that cold-power latch `pwr q` clears on read;
//! * that a deferred destructive command acts once its `ok` has been read;
//! * that `nic` reads 1 (no device on the one-wire line) and `sla` reports
//!   from a slave are stored or logged only.
//!
//! Time is read from a `heart::Clock`, as the Bedrock devices read it, and
//! timers run lazily, whenever the bus, the serial line or [`Elsc::poll`]
//! touches the controller, so a test sets the clock by hand and never waits.
//! A request is attributed to the CPU thread that made it
//! ([`crate::cpu::smp::current_cpu`]); other threads count as CPU 0.
//!
//! Migration and snapshot state are not saved yet.
//!
//! Not modelled: the second Origin 200 module's controller and the master's
//! forwarding to it, sensors, the electrical effect of the supply margins, the
//! status messages' display texts (unknown), and the Origin 2000 module's
//! shared bus, which the module-management device owns.
//!
//! Ported from hw/misc/sgi-elsc.c in jamesbraid/qemu-sgi-origami.

use crate::dev::elsc_nvram::{self, ElscNvram};
use crate::dev::heart::Clock;
use crate::dev::i2c::{BusBusy, I2cTarget, Service};
use crate::dev::serial_host::{null_serial_backend, SerialBackend};
use crate::devlog::LogModule;
use crate::dlog;
use crate::traits::Resettable;
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::atomic::Ordering;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

/// Seven-bit I2C address the controller answers at.
pub const I2C_ADDR: u8 = 0x08;

/// Length, text, checksum summing to zero. A length above 94 would overrun the
/// firmware's own 96-byte buffers; never emit one.
pub const MAX_TEXT: usize = 94;
pub const MAX_FRAME: usize = MAX_TEXT + 2;

pub const RX_QUEUE_SIZE: usize = 256;
pub const DELAY_MAX: u32 = 0x258;
pub const PROCESSORS: usize = 8;

const VERSION_ORIGIN200: &str = "V 3.1S  ";
const VERSION_ORIGIN2000: &str = "VER 3.2 ";
const DISPLAY_CHARS: usize = 8;
const SUPPLIES: usize = 3;
const SEE_ALL: u8 = 0xff;
const CTRL_T: u8 = 0x14;
const TAS_FREE: u8 = b'0';
const CYCLE_DEFAULT_SECONDS: u32 = 5;
const HBT_CYCLE_SECONDS: u64 = 5;
const HBT_DEFAULT_DELAY: u32 = 30;
const HBT_WARNING: u64 = 5;
const POWER_UP_HOLD_MS_ORIGIN200: u64 = 4000;
const POWER_UP_HOLD_MS_ORIGIN2000: u64 = 4400;
const MESSAGE_LIFETIME_MS: u64 = 1000;
const RESET_WAIT_MS: u64 = 1000;
const NMI_WAIT_MS: u64 = 1000;
const NMI_PULSE_MS: u64 = 100;
const ARB_TICK_NS: u64 = NS_PER_MS;
const ARB_POSITIONS: u32 = 5;
const SIGNAL_BASE: u8 = 0xe0;
const NS_PER_MS: u64 = 1_000_000;

/// What a module controller does to its module and to the processors. The
/// Origin 200's [`Elsc`] and the Origin 2000 module's controller
/// ([`crate::dev::o2000_module`]) both emit it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModuleOutput {
    /// Module power output: low as the module loses power, high as it
    /// returns. A power-down is repeated when the controller learns that a
    /// pending power-up was cancelled, so the machine asks again whether
    /// anything is left to bring a module back.
    Power(bool),
    /// A pulse on the module's reset line (instantaneous in the model; a
    /// controller that holds the line for a while reports its rising edge).
    Reset,
    /// The NMI output, high for the 100 ms of a pulse.
    Nmi(bool),
}

/// Which build of the controller firmware this is: one program built two
/// ways, differing in the commands it answers, its version text and its
/// display.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Build {
    Origin200,
    Origin2000,
}

/// What an Origin 2000 module owns that is module-scoped rather than
/// node-scoped: what a reset or power-off actually reaches, when the
/// controller may take the module's shared bus, which processor the
/// module's one alternate console belongs to, and the midplane one-wire net.
///
/// The controller calls these with its own lock held, so an implementation
/// must answer from state of its own and never call back into the
/// controller. A request that does need the controller (`execute`) is queued
/// and delivered after the controller's lock is released.
pub trait ModuleOps: Send + Sync {
    /// Whether the module is powered.
    fn powered(&self) -> bool;
    /// Whether the module can carry the action out. A command it refuses is
    /// answered `err cmd`, since a caller takes `ok` as proof.
    fn can_execute(&self, action: Action) -> bool;
    /// Carry the action out. Only called after `can_execute` cleared it.
    fn execute(&self, action: Action);
    /// A reset or NMI pulse is running on the module's outputs.
    fn pulsing(&self) -> bool;
    /// Pulse the module's NMI output; false when it has no NMI path or a
    /// pulse is already running.
    fn nmi(&self) -> bool;
    /// This controller is being addressed as a target, or has finished.
    fn slave_active(&self, active: bool);
    /// The processor queued console input is announced to.
    fn console_owner(&self, processor: usize);
    /// Queued console input is or is not waiting for an announcement.
    fn console_input(&self, pending: bool);
    /// A serial command asks for its turn on the module's bus: true once
    /// granted, with grant publication stopped.
    fn serial_turn(&self) -> bool;
    /// The serial command is done with the bus.
    fn serial_done(&self);
    /// One bit slot on the midplane one-wire line.
    fn onewire_slot(&self, pulse: u32, sample: u32) -> bool;
    /// A controller message for the Hubs: announce it to every processor code.
    fn post_message(&self);
    /// Whether the message is still announced to this processor; if so, it
    /// stops being announced to it.
    fn take_message(&self, processor: usize) -> bool;
}

#[derive(Clone, Copy, Debug)]
pub struct ElscConfig {
    pub build: Build,
    /// Front-panel DIP switches as the `dip-switches` property gives them
    /// (inverted when read back by `rsw`).
    pub dip_switches: u8,
    /// Whether this is the master controller, which publishes the grants.
    /// A slave's role bit in `rsw` is clear and it runs no serial commands.
    pub master: bool,
    /// Milliseconds after start that the controller powers its module up on
    /// its own (when the NVRAM says automatic power-on); zero starts the
    /// module powered.
    pub start_delay_ms: u32,
}

impl Default for ElscConfig {
    fn default() -> Self {
        Self { dip_switches: 0, master: true, build: Build::Origin200, start_delay_ms: 0 }
    }
}

/// A destructive action a command can ask for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Action {
    None,
    Reset,
    Cycle,
    Shutdown,
    PowerUp,
}

impl Action {
    fn name(self) -> &'static str {
        match self {
            Action::Reset => "reset",
            Action::Cycle => "power-cycle",
            Action::Shutdown => "shutdown",
            Action::PowerUp => "power-up",
            Action::None => "deferred action",
        }
    }
}

/// The status messages an Origin 2000 module controller writes to its
/// display. Their eight-character texts are not known, so none is invented.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Status {
    None,
    PdRemote,
    PowerOff,
    PowerUp,
    Reset,
    Nmi,
    Heartbeat,
}

impl Status {
    fn name(self) -> &'static str {
        match self {
            Status::None => "",
            Status::PdRemote => "remote power-down",
            Status::PowerOff => "power off",
            Status::PowerUp => "power up",
            Status::Reset => "reset",
            Status::Nmi => "NMI",
            Status::Heartbeat => "heartbeat timeout",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Message {
    None,
    PdRemote,
    PdCycle,
    Rst,
}

impl Message {
    fn text(self) -> &'static str {
        match self {
            Message::PdRemote => "pd remote",
            Message::PdCycle => "pd cycle",
            Message::Rst => "rst",
            Message::None => "",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Select {
    Auto,
    None,
    Explicit,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Heartbeat {
    Idle,
    Waiting,
    Warned,
    Cycling,
}

// Timer slots.
const T_ARB: usize = 0;
const T_PWR_DOWN: usize = 1;
const T_MESSAGE: usize = 2;
const T_RST: usize = 3;
const T_NMI: usize = 4;
const T_NMI_PULSE: usize = 5;
const T_HBT: usize = 6;
const T_POWER_UP: usize = 7;
const T_START: usize = 8;
const TIMERS: usize = 9;

// Commands whose missing effect is logged once.
const UNIMP_FSC: u8 = 1 << 0;
const UNIMP_TST: u8 = 1 << 1;
const UNIMP_FRE: u8 = 1 << 2;
const UNIMP_NIC: u8 = 1 << 3;

fn checksum(buf: &[u8]) -> u8 {
    buf.iter().fold(0u8, |s, b| s.wrapping_add(*b))
}

/// One request as the controller parses it. The firmware keeps the text and
/// puts a carriage return where the checksum byte was, so every command ends
/// in one: the last byte of `text` is always that carriage return. `pos` is
/// the next unread byte and the readers never move it past the end.
struct Req<'a> {
    text: &'a [u8],
    pos: usize,
}

impl Req<'_> {
    fn peek(&self) -> u8 {
        self.text.get(self.pos).copied().unwrap_or(b'\r')
    }

    fn skip(&mut self) {
        if self.pos < self.text.len() {
            self.pos += 1;
        }
    }

    /// The character reader most commands use: skip spaces and take the next
    /// character. A carriage return or line feed is no argument.
    fn read_char(&mut self) -> Option<u8> {
        while self.peek() == b' ' {
            self.skip();
        }
        let c = self.peek();
        if c == b'\r' || c == b'\n' {
            return None;
        }
        self.skip();
        Some(c)
    }

    /// The hex reader: up to four digits, either case, with no leading spaces
    /// skipped. A number shorter than four digits consumes the character that
    /// ends it, so a following number must come after exactly one space; a
    /// four-digit number consumes nothing after it. No digits at all reads as
    /// no value, and the character that ended the attempt is still consumed.
    fn read_hex(&mut self) -> Option<u32> {
        let mut value = 0u32;
        let mut digits = 0;
        while digits < 4 {
            let Some(d) = (self.peek() as char).to_digit(16) else {
                break;
            };
            value = value << 4 | d;
            self.skip();
            digits += 1;
        }
        if digits < 4 {
            self.skip();
        }
        (digits > 0).then_some(value)
    }

    /// A run of non-space characters, up to the end of the line.
    fn read_word(&mut self) -> &[u8] {
        while self.peek() == b' ' {
            self.skip();
        }
        let start = self.pos;
        while self.pos < self.text.len() && !matches!(self.text[self.pos], b' ' | b'\r' | b'\n' | 0) {
            self.pos += 1;
        }
        &self.text[start..self.pos]
    }

    /// The two characters `sel` and `see` take, folded to uppercase. The
    /// second is 0 when there is none.
    fn read_selection(&mut self) -> Option<(u8, u8)> {
        let first = self.read_char()?.to_ascii_uppercase();
        let second = self.read_char().map_or(0, |c| c.to_ascii_uppercase());
        Some((first, second))
    }
}

/// A slot digit and `A` or `B` name one processor.
fn selection_processor(first: u8, second: u8) -> Option<u8> {
    if !(b'1'..=b'4').contains(&first) || !matches!(second, b'A' | b'B') {
        return None;
    }
    Some((first - b'1') * 2 + (second == b'B') as u8)
}

fn processor_name(p: u8) -> [u8; 2] {
    [b'1' + p / 2, if p & 1 != 0 { b'b' } else { b'a' }]
}

type Handler = fn(&mut State, &mut Req);

/// Builds that have a command in their table.
const B200: u8 = 1;
const B2000: u8 = 2;
const BOTH: u8 = B200 | B2000;

struct Command {
    name: &'static [u8; 3],
    builds: u8,
    handler: Handler,
    on_i2c: bool,
    on_serial: bool,
}

const fn cmd(name: &'static [u8; 3], builds: u8, handler: Handler, on_i2c: bool, on_serial: bool) -> Command {
    Command { name, builds, handler, on_i2c, on_serial }
}

impl Command {
    fn in_build(&self, build: Build) -> bool {
        self.builds & if build == Build::Origin200 { B200 } else { B2000 } != 0
    }
}

/// What each build answers, and on which channel. A mnemonic a build lacks
/// answers `err cmd` with no other effect (`aut` is the Origin 200's alone;
/// `dsc`, `dsp`, `key`, `vlm` and `fre` the Origin 2000's), and so does a
/// command on a channel that does not take it: `acp` and `get` belong to the
/// Hubs, and the serial-only ones to the operator.
static COMMANDS: &[Command] = &[
    cmd(b"acp", BOTH, State::handle_acp, true, false),
    cmd(b"aut", B200, State::handle_aut, true, true),
    cmd(b"clr", BOTH, State::handle_clr, true, true),
    cmd(b"dsc", B2000, State::handle_dsc, true, true),
    cmd(b"dsp", B2000, State::handle_dsp, true, true),
    cmd(b"fan", BOTH, State::handle_fan, true, true),
    cmd(b"fsc", BOTH, State::handle_fsc, true, true),
    cmd(b"get", BOTH, State::handle_get, true, false),
    cmd(b"hbt", BOTH, State::handle_hbt, true, true),
    cmd(b"key", B2000, State::handle_key, true, true),
    cmd(b"nic", BOTH, State::handle_nic, true, true),
    cmd(b"nmi", BOTH, State::handle_nmi, true, true),
    cmd(b"pwr", BOTH, State::handle_pwr, true, true),
    cmd(b"rst", BOTH, State::handle_rst, true, true),
    cmd(b"rsw", BOTH, State::handle_rsw, true, true),
    cmd(b"see", BOTH, State::handle_see, true, true),
    cmd(b"sel", BOTH, State::handle_sel, true, true),
    cmd(b"sla", BOTH, State::handle_sla, true, true),
    cmd(b"tas", BOTH, State::handle_tas, true, true),
    cmd(b"tmp", BOTH, State::handle_tmp, true, true),
    cmd(b"tst", BOTH, State::handle_tst, true, true),
    cmd(b"ver", BOTH, State::handle_ver, true, true),
    cmd(b"vlm", B2000, State::handle_vlm, true, true),
    cmd(b"cfg", BOTH, State::handle_cfg, false, true),
    cmd(b"dbg", BOTH, State::handle_dbg, false, true),
    cmd(b"ech", BOTH, State::handle_ech, false, true),
    cmd(b"fre", B2000, State::handle_fre, false, true),
    cmd(b"mod", BOTH, State::handle_mod, false, true),
    cmd(b"nvm", BOTH, State::handle_nvm, false, true),
    cmd(b"pas", BOTH, State::handle_pas, false, true),
];

struct State {
    cfg: ElscConfig,
    nvram: Arc<ElscNvram>,
    bus_busy: BusBusy,

    /// Virtual time of the event being handled, nanoseconds.
    now: u64,
    deadlines: [Option<u64>; TIMERS],

    /// The request, one byte per I2C send; one over the legal maximum so an
    /// over-long guest frame lands here, not in adjacent state.
    rx: Vec<u8>,
    /// The response being read back, built once the request is complete.
    tx: Vec<u8>,
    txpos: usize,
    /// The pending response's owner: the processor code whose request framed
    /// it, -1 when none. While it is unconsumed the owner's grant uses
    /// signal 2.
    resp_owner: i32,

    // Grant ring.
    arb_code: u32,
    arb_signal: u8,
    pending_grant: Option<u8>,

    // Power.
    module_off: bool,
    power_sequence: bool,
    cycle_seconds: u32,
    cold_power_on: bool,

    // Heartbeat watchdog.
    hbt_interval: u32,
    hbt_delay: u32,
    hbt_stage: Heartbeat,
    nmi_pulse_active: bool,

    // AUX serial port.
    rx_queue: VecDeque<u8>,
    serial_command: bool,
    serial_line: Vec<u8>,
    serial_term: u8,
    serial_pending: bool,
    serial_echo: bool,
    supervisor: bool,
    slave_addressed: bool,

    // A serial command in flight.
    serial_running: bool,
    serial_raw: bool,
    serial_reply: Vec<u8>,
    serial_action: Action,

    // Deferred destructive command.
    deferred_action: Action,
    deferred_committed: bool,
    deferred_scheduled: bool,

    tas_lock: u8,
    fan_high: bool,
    fsc_present: bool,
    short_timeouts: bool,
    unimp_logged: u8,
    slave_link: bool,
    slave_temp: u8,
    slave_report: u8,

    message: Message,
    message_mask: u8,

    sel_mode: Select,
    sel_processor: u8,
    see_processor: u8,
    console_sender: i32,

    /// What an Origin 2000 module owns, if one is attached.
    module: Option<Arc<dyn ModuleOps>>,
    /// Actions for the module to carry out, delivered once the lock is
    /// released because carrying them out reaches back into the controller.
    outbox: Vec<Action>,
    /// The Origin 2000 build's front-panel display and the status message
    /// that replaced what the host wrote there.
    display: [u8; DISPLAY_CHARS],
    status: Status,
    /// Each supply's margin (`vlm`): `n`, `h` or `l`.
    margins: [u8; SUPPLIES],

    /// Bytes for the AUX port, sent once the lock is released.
    out: Vec<u8>,
    events: Vec<ModuleOutput>,
    release_bus: bool,
    /// What [`Self::due`] said when the machine was last told it might have
    /// moved; `None` before the first report.
    reported_due: Option<Option<u64>>,
}

impl State {
    fn new(cfg: ElscConfig, nvram: Arc<ElscNvram>, bus_busy: BusBusy) -> Self {
        Self {
            cfg,
            nvram,
            bus_busy,
            now: 0,
            deadlines: [None; TIMERS],
            rx: Vec::new(),
            tx: Vec::new(),
            txpos: 0,
            resp_owner: -1,
            arb_code: 0,
            arb_signal: 0,
            pending_grant: None,
            module_off: false,
            power_sequence: false,
            cycle_seconds: CYCLE_DEFAULT_SECONDS,
            // Set at the controller's first power-up and again whenever its
            // module powers up; no reset re-arms it.
            cold_power_on: true,
            hbt_interval: 0,
            hbt_delay: 0,
            hbt_stage: Heartbeat::Idle,
            nmi_pulse_active: false,
            rx_queue: VecDeque::new(),
            serial_command: false,
            serial_line: Vec::new(),
            serial_term: 0,
            serial_pending: false,
            serial_echo: true,
            supervisor: false,
            slave_addressed: false,
            serial_running: false,
            serial_raw: false,
            serial_reply: Vec::new(),
            serial_action: Action::None,
            deferred_action: Action::None,
            deferred_committed: false,
            deferred_scheduled: false,
            tas_lock: TAS_FREE,
            fan_high: false,
            fsc_present: false,
            short_timeouts: false,
            unimp_logged: 0,
            slave_link: false,
            slave_temp: b'n',
            slave_report: 0,
            message: Message::None,
            message_mask: 0,
            sel_mode: Select::Auto,
            sel_processor: 0,
            see_processor: SEE_ALL,
            console_sender: -1,
            module: None,
            outbox: Vec::new(),
            display: [b' '; DISPLAY_CHARS],
            status: Status::None,
            margins: [b'n'; SUPPLIES],
            out: Vec::new(),
            events: Vec::new(),
            release_bus: false,
            reported_due: None,
        }
    }

    // ---- time -----------------------------------------------------------

    fn arm_ms(&mut self, timer: usize, ms: u64) {
        self.deadlines[timer] = Some(self.now + ms * NS_PER_MS);
    }

    fn arm_secs(&mut self, timer: usize, secs: u64) {
        self.arm_ms(timer, secs * 1000);
    }

    fn pending(&self, timer: usize) -> bool {
        self.deadlines[timer].is_some()
    }

    /// When the controller next has something to tell its machine, in the
    /// clock's nanoseconds: a destructive action committed and waiting to
    /// run, an output already queued that nobody has taken, or a timer that
    /// ends in one of the module's outputs. The queue counts because a bus
    /// access can pass a deadline before the machine's visit and queue the
    /// output itself, which leaves no timer behind to wake the machine for.
    /// The grant ring's tick and a message's expiry change only what a bus
    /// access sees, and the access that sees it replays them first, so they
    /// are not due.
    fn due(&self) -> Option<u64> {
        if self.deferred_scheduled || !self.events.is_empty() {
            return Some(self.now);
        }
        const OUTPUT_TIMERS: [usize; 7] = [T_PWR_DOWN, T_RST, T_NMI, T_NMI_PULSE, T_HBT, T_POWER_UP, T_START];
        OUTPUT_TIMERS.iter().filter_map(|&t| self.deadlines[t]).min()
    }

    fn cancel(&mut self, timer: usize) {
        self.deadlines[timer] = None;
    }

    /// Run every timer due by `clock_now`, oldest first, each at its own
    /// deadline so a long gap replays the same sequence a steady clock gives.
    fn advance(&mut self, clock_now: u64) {
        if clock_now < self.now {
            return;
        }
        while let Some((deadline, timer)) = self
            .deadlines
            .iter()
            .enumerate()
            .filter_map(|(i, d)| d.filter(|d| *d <= clock_now).map(|d| (d, i)))
            .min()
        {
            self.deadlines[timer] = None;
            self.now = deadline.max(self.now);
            self.fire(timer, deadline, clock_now);
        }
        self.now = clock_now;
    }

    fn fire(&mut self, timer: usize, deadline: u64, clock_now: u64) {
        match timer {
            T_ARB => self.arb_tick(deadline, clock_now),
            T_PWR_DOWN => self.pwr_down_expired(),
            T_MESSAGE => {
                self.message = Message::None;
                self.message_mask = 0;
            }
            T_RST => self.reset_wait_expired(),
            T_NMI => self.pulse_nmi(),
            T_NMI_PULSE => {
                self.nmi_pulse_active = false;
                self.events.push(ModuleOutput::Nmi(false));
                self.tas_lock = TAS_FREE;
                self.reinitialize();
            }
            T_HBT => self.hbt_expired(),
            T_POWER_UP => self.power_up_expired(),
            T_START => self.start_expired(),
            _ => unreachable!(),
        }
    }

    // ---- serial line ----------------------------------------------------

    fn serial_write(&mut self, bytes: &[u8]) {
        self.out.extend_from_slice(bytes);
    }

    fn serial_puts(&mut self, text: &str) {
        self.serial_write(text.as_bytes());
    }

    // ---- framing --------------------------------------------------------

    /// A frame above `MAX_TEXT` overruns the firmware's buffer; a model
    /// invariant, not guest-triggerable.
    fn respond(&mut self, text: &[u8]) {
        assert!(text.len() <= MAX_TEXT);
        // A serial command's reply goes to the serial line instead.
        if self.serial_running {
            self.serial_reply = text.to_vec();
            return;
        }
        self.tx.clear();
        self.tx.push(text.len() as u8);
        self.tx.extend_from_slice(text);
        self.tx.push(checksum(&self.tx).wrapping_neg());
        self.txpos = 0;
    }

    fn respond_str(&mut self, text: &str) {
        self.respond(text.as_bytes());
    }

    /// A success with no value: the firmware's three bytes `ok `, trailing
    /// space included. IRIX accepts `ok` followed by a space or the end.
    fn respond_ok(&mut self) {
        self.respond_str("ok ");
    }

    // ---- power ----------------------------------------------------------

    /// Whether this controller publishes its own grant ring: an Origin 200
    /// master does. An Origin 2000 module's scheduler does it for the module.
    fn rings(&self) -> bool {
        !self.is2000() && self.cfg.master
    }

    fn is2000(&self) -> bool {
        self.cfg.build == Build::Origin2000
    }

    fn version(&self) -> &'static str {
        if self.is2000() {
            VERSION_ORIGIN2000
        } else {
            VERSION_ORIGIN200
        }
    }

    /// Record an Origin 2000 display status message.
    fn set_status(&mut self, status: Status) {
        if self.is2000() {
            self.status = status;
        }
    }

    fn powered(&self) -> bool {
        match &self.module {
            Some(m) => m.powered(),
            None => !self.module_off,
        }
    }

    fn can_execute(&self, action: Action) -> bool {
        self.module.as_ref().is_none_or(|m| m.can_execute(action))
    }

    fn pulsing(&self) -> bool {
        self.module.as_ref().is_some_and(|m| m.pulsing())
    }

    /// Tell a module owner where queued console input is announced now.
    fn console_update(&self) {
        let Some(m) = &self.module else {
            return;
        };
        let target = self.console_target();
        if target >= 0 {
            m.console_owner(target as usize);
        }
        m.console_input(target >= 0 && !self.rx_queue.is_empty());
    }

    /// A power-up that was to come may have just been cancelled. On an
    /// unpowered module say again that the power is off, so the machine asks
    /// whether anything is left to power a module up.
    fn recheck_shutdown(&mut self) {
        if !self.powered() && self.module.is_none() {
            self.events.push(ModuleOutput::Power(false));
        }
    }

    /// Whether module power is due back without anyone asking: a power-up
    /// countdown, or a heartbeat watchdog that will power-cycle the module.
    fn power_return_pending(&self) -> bool {
        if self.pending(T_POWER_UP) || self.pending(T_START) {
            return true;
        }
        if !self.pending(T_HBT) {
            return false;
        }
        match self.hbt_stage {
            Heartbeat::Waiting | Heartbeat::Warned => self.hbt_delay != 0,
            Heartbeat::Cycling => true,
            Heartbeat::Idle => false,
        }
    }

    /// Clear the protocol queues and nothing else. An in-flight request is
    /// abandoned; an action already acknowledged to a guest still happens.
    fn reinitialize(&mut self) {
        self.rx.clear();
        self.tx.clear();
        self.txpos = 0;
        self.resp_owner = -1;
        self.rx_queue.clear();
        self.slave_addressed = false;
        if let Some(m) = &self.module {
            m.slave_active(false);
        }
        self.console_update();
    }

    /// Count down to power-up: the off interval of a power cycle, measured
    /// from now and replacing any countdown already running.
    fn power_countdown(&mut self, seconds: u32) {
        self.power_sequence = false;
        self.arm_secs(T_POWER_UP, seconds as u64);
    }

    /// The supplies come on with reset held, the lock is freed and queued
    /// serial input goes. The module runs again when the hold ends.
    fn power_sequence_start(&mut self) {
        self.power_sequence = true;
        self.tas_lock = TAS_FREE;
        self.reinitialize();
        let hold = if self.is2000() { POWER_UP_HOLD_MS_ORIGIN2000 } else { POWER_UP_HOLD_MS_ORIGIN200 };
        self.arm_ms(T_POWER_UP, hold);
    }

    /// Remove module power. The controller keeps running on standby power.
    fn remove_power(&mut self) {
        self.cancel(T_PWR_DOWN);
        if !self.powered() {
            return;
        }
        self.set_status(Status::PowerOff);
        if self.module.is_some() {
            // The module's power output reaches the machine.
            self.outbox.push(Action::Shutdown);
            return;
        }
        self.module_off = true;
        self.reinitialize();
        self.release_bus = true;
        self.events.push(ModuleOutput::Power(false));
    }

    fn power_up_complete(&mut self, announce: bool) {
        self.power_sequence = false;
        self.cold_power_on = true;
        if announce {
            self.set_status(Status::PowerUp);
        }
        if self.module.is_some() {
            self.outbox.push(Action::PowerUp);
            return;
        }
        self.module_off = false;
        self.events.push(ModuleOutput::Power(true));
    }

    fn power_up_expired(&mut self) {
        if !self.power_sequence {
            self.power_sequence_start();
            return;
        }
        self.power_up_complete(true);
    }

    /// Power goes now and comes back after the interval and the power-up
    /// sequence. On a module already off it only restarts the countdown.
    fn power_cycle(&mut self, seconds: u32) {
        self.cancel(T_PWR_DOWN);
        self.power_countdown(seconds);
        self.remove_power();
    }

    /// The controller's own start-up power-up: an Origin 200 master does it
    /// when its NVRAM says automatic power-on; a slave waits for its master.
    fn start_expired(&mut self) {
        if self.powered() {
            return;
        }
        // An Origin 2000 controller powers its module up unconditionally; an
        // Origin 200 master does when its NVRAM says automatic power-on.
        let auto = self.is2000() || self
            .nvram
            .access(elsc_nvram::FIRST_ADDR + 7, 7, None)
            .is_some_and(|c| c & elsc_nvram::CONFIG_AUTO_POWER != 0);
        if !auto {
            self.recheck_shutdown();
            return;
        }
        self.power_sequence_start();
    }

    fn pwr_down_expired(&mut self) {
        // Its "ok" was read long ago, so the shutdown commits at once. It
        // overrides an action whose ack is still unread; one already
        // committed fires instead and the shutdown is lost.
        if self.deferred_committed {
            dlog!(
                LogModule::Module,
                "elsc: \"pwr d\" countdown expired while a {} is committed; that action fires instead",
                self.deferred_action.name()
            );
            return;
        }
        self.deferred_action = Action::Shutdown;
        self.deferred_committed = true;
        self.fire_committed();
    }

    fn pulse_nmi(&mut self) {
        if !self.powered() {
            // The pulse reaches no processor of an unpowered module.
            self.tas_lock = TAS_FREE;
            self.reinitialize();
            return;
        }
        if let Some(m) = &self.module {
            // The module drives the pulse itself and frees the lock as it
            // starts.
            if !m.nmi() {
                dlog!(LogModule::Module, "elsc: the module could not pulse its NMI output");
                return;
            }
            self.tas_lock = TAS_FREE;
            return;
        }
        if self.nmi_pulse_active {
            dlog!(LogModule::Module, "elsc: NMI requested during an NMI pulse");
            return;
        }
        self.nmi_pulse_active = true;
        self.events.push(ModuleOutput::Nmi(true));
        self.arm_ms(T_NMI_PULSE, NMI_PULSE_MS);
    }

    fn reset_wait_expired(&mut self) {
        self.tas_lock = TAS_FREE;
        if self.module.is_some() {
            if !self.can_execute(Action::Reset) {
                dlog!(LogModule::Module, "elsc: the module refused the reset an acknowledged \"rst\" asked for");
                return;
            }
            self.outbox.push(Action::Reset);
            return;
        }
        self.release_bus = true;
        self.events.push(ModuleOutput::Reset);
    }

    fn hbt_arm(&mut self, seconds: u64) {
        self.arm_secs(T_HBT, seconds);
    }

    fn hbt_stop(&mut self) {
        self.hbt_interval = 0;
        self.hbt_stage = Heartbeat::Idle;
        self.cancel(T_HBT);
    }

    /// The watchdog's timer ran out. Waiting: warn, and give the host five
    /// more seconds. Warned: pulse the NMI at once and start the delay.
    /// Cycling: power the module off and on and go idle with the interval
    /// still set. Nothing here looks at module power, so an armed watchdog
    /// brings back a module `pwr d` turned off.
    fn hbt_expired(&mut self) {
        match self.hbt_stage {
            Heartbeat::Waiting => {
                self.set_status(Status::Heartbeat);
                self.hbt_stage = Heartbeat::Warned;
                self.hbt_arm(HBT_WARNING);
            }
            Heartbeat::Warned => {
                self.pulse_nmi();
                self.hbt_stage = Heartbeat::Cycling;
                if self.hbt_delay != 0 {
                    self.hbt_arm(self.hbt_delay as u64);
                }
            }
            Heartbeat::Cycling => {
                if self.can_execute(Action::Cycle) {
                    self.hbt_stage = Heartbeat::Idle;
                    self.power_cycle(HBT_CYCLE_SECONDS as u32);
                } else if self.pulsing() {
                    // The watchdog's own NMI pulse, or a reset, is still
                    // running; the cycle follows once it has ended.
                    self.arm_ms(T_HBT, NMI_PULSE_MS);
                } else {
                    dlog!(LogModule::Module, "elsc: the heartbeat watchdog's power cycle cannot run on this module");
                    self.hbt_stage = Heartbeat::Idle;
                }
            }
            Heartbeat::Idle => {}
        }
    }

    // ---- controller messages -------------------------------------------

    /// Queue a message for the Hubs. It is announced to every processor code
    /// until that processor fetches it with `get` or the second runs out; a
    /// later message replaces it. The Origin 200 build also prints each one on
    /// its serial line.
    fn post_message(&mut self, message: Message) {
        self.message = message;
        if !self.is2000() {
            self.serial_puts("\x14\r\nMSC: ");
            self.serial_puts(message.text());
            self.serial_puts("\x14\r");
        }
        self.arm_ms(T_MESSAGE, MESSAGE_LIFETIME_MS);
        if let Some(m) = &self.module {
            m.post_message();
            return;
        }
        self.message_mask = 0xff;
    }

    /// Whether the message is announced to a processor, which then has it.
    fn take_message(&mut self, processor: i32) -> bool {
        if self.message == Message::None || !(0..PROCESSORS as i32).contains(&processor) {
            return false;
        }
        if let Some(m) = &self.module {
            return m.take_message(processor as usize);
        }
        let bit = 1u8 << processor;
        if self.message_mask & bit == 0 {
            return false;
        }
        self.message_mask &= !bit;
        true
    }

    // ---- console selection ---------------------------------------------

    /// The processor queued console input is announced to, or -1 with input
    /// selection disabled.
    fn console_target(&self) -> i32 {
        match self.sel_mode {
            Select::Explicit => self.sel_processor as i32,
            Select::None => -1,
            Select::Auto if (0..PROCESSORS as i32).contains(&self.console_sender) => self.console_sender,
            Select::Auto => -1,
        }
    }

    /// A new input selection discards queued serial input not yet fetched.
    fn select_input(&mut self, mode: Select, processor: u8) {
        self.sel_mode = mode;
        if mode == Select::Explicit {
            self.sel_processor = processor;
        }
        self.rx_queue.clear();
        self.console_update();
    }

    fn queue_input(&mut self, c: u8) {
        let was_pending = !self.rx_queue.is_empty();
        if self.rx_queue.len() == RX_QUEUE_SIZE {
            self.rx_queue.pop_front();
        }
        self.rx_queue.push_back(c);
        if !was_pending {
            self.console_update();
        }
    }

    // ---- grants ---------------------------------------------------------

    /// Whether something waits for the given code's processor: a controller
    /// message, console input for the console processor, or a response its
    /// requester has not yet consumed.
    fn signal2(&self, code: u32) -> bool {
        (code < PROCESSORS as u32 && self.message_mask & (1 << code) != 0)
            || (!self.rx_queue.is_empty() && self.console_target() == code as i32)
            || (self.resp_owner >= 0 && code == self.resp_owner as u32 && self.txpos < self.tx.len())
    }

    /// Whether a tick would publish nothing: the bus is busy, or a serial
    /// command is waiting for it (which also stops the grants).
    fn arb_blocked(&self) -> bool {
        self.bus_busy.load(Ordering::Relaxed) || (self.serial_pending && self.message_mask == 0)
    }

    /// Publish the next position's grant, which stays readable until the next
    /// tick replaces it.
    fn arb_publish(&mut self) {
        let code = self.arb_code;
        let signal1 = SIGNAL_BASE + code as u8;
        self.arb_signal = if self.signal2(code) { signal1 + 8 } else { signal1 };
        self.pending_grant = Some(self.arb_signal);
        self.arb_code = (code + 1) % ARB_POSITIONS;
    }

    /// One periodic tick. A tick that finds the bus busy publishes nothing and
    /// does not advance, and the controller waits for its next tick rather
    /// than publishing as soon as the bus frees.
    ///
    /// The ring is replayed on its one-millisecond grid when a gap is long,
    /// but nothing between two events changes what a tick sees, so whole
    /// revolutions (or, while blocked, every tick) up to the next other timer
    /// are skipped in one step.
    fn arb_tick(&mut self, deadline: u64, clock_now: u64) {
        let blocked = self.arb_blocked();
        let horizon = self
            .deadlines
            .iter()
            .flatten()
            .copied()
            .filter(|d| *d <= clock_now)
            .min()
            .map_or(clock_now, |d| d.min(clock_now));
        let whole = horizon.saturating_sub(deadline) / ARB_TICK_NS;
        let skip = if blocked { whole } else { whole / ARB_POSITIONS as u64 * ARB_POSITIONS as u64 };
        let at = deadline + skip * ARB_TICK_NS;
        self.now = self.now.max(at);
        if !blocked {
            self.arb_publish();
        }
        self.deadlines[T_ARB] = Some(at + ARB_TICK_NS);
    }

    // ---- deferred destructive commands ---------------------------------

    /// Arm a destructive action whose `ok` has just been framed. Clearing
    /// `deferred_committed` would demote an acknowledged action to armed and
    /// lose the commit after the last text byte is fetched. The committed
    /// action still fires; log if this request's `ok` therefore describes
    /// another action.
    fn arm_action(&mut self, action: Action, command: &str) {
        // A serial command's action runs once its reply is printed.
        if self.serial_running {
            self.serial_action = action;
            return;
        }
        if !self.deferred_committed {
            self.deferred_action = action;
        } else {
            dlog!(
                LogModule::Module,
                "elsc: \"{command}\" while a {} is already committed; that action fires and this \"ok\" may not match it",
                self.deferred_action.name()
            );
        }
    }

    /// The fetch of the last frame byte: the whole ack is delivered. End the
    /// announcement and commit any deferred action. Committing here, not at a
    /// STOP or NACK, keeps the action when a master NAKs that byte and then
    /// repeats the start with no STOP.
    fn response_fully_read(&mut self) {
        self.resp_owner = -1;
        if self.deferred_action != Action::None {
            self.deferred_committed = true;
        }
    }

    /// Mark a committed action to fire. Called at a transaction boundary (a
    /// NACK, the STOP, or a following start) so it runs after the guest has
    /// taken the whole frame, not from inside the read that completed the ack.
    fn fire_committed(&mut self) {
        if self.deferred_action != Action::None && self.deferred_committed {
            self.deferred_scheduled = true;
        }
    }

    fn run_committed_action(&mut self) {
        if !self.deferred_committed {
            self.deferred_scheduled = false;
            return;
        }
        let action = self.deferred_action;
        self.deferred_action = Action::None;
        self.deferred_committed = false;
        self.deferred_scheduled = false;
        self.run_action(action);
    }

    fn run_action(&mut self, action: Action) {
        // Release the bus for every destructive action, before it fires: all
        // three leave this controller selected (the NACK-without-STOP path
        // never ended the transfer) and none ends it themselves.
        self.release_bus = true;
        match action {
            Action::Cycle => {
                // The Hubs hear "pd cycle" before the power goes.
                self.post_message(Message::PdCycle);
                self.power_cycle(self.cycle_seconds);
            }
            Action::Reset => {
                // The reset comes a second later; another "rst" inside that
                // second starts the wait again.
                self.post_message(Message::Rst);
                self.set_status(Status::Reset);
                self.arm_ms(T_RST, RESET_WAIT_MS);
            }
            Action::Shutdown => self.remove_power(),
            Action::PowerUp => {
                // "pwr u" on an unpowered module starts the power-up sequence
                // now, cutting short any countdown.
                if !self.powered() && !(self.power_sequence && self.pending(T_POWER_UP)) {
                    self.power_sequence_start();
                }
            }
            Action::None => unreachable!("only a real action can be committed"),
        }
    }

    // ---- request handling ----------------------------------------------

    /// The direction switch to read marks the request complete.
    fn request_complete(&mut self) {
        if self.rx.is_empty() && self.txpos < self.tx.len() {
            // No request this transaction: a fetch of the pending response.
            return;
        }
        // A new request supersedes any pending response. An armed action (its
        // ack not fully read) is cancelled; a committed one is past the point
        // of no return and survives.
        if !self.deferred_committed {
            self.deferred_action = Action::None;
        }
        self.resp_owner = crate::cpu::smp::current_cpu() as i32;
        self.record_console_sender();

        let n = self.rx.first().map_or(0, |b| *b as usize);
        if self.rx.len() < 2 || self.rx.len() != n + 2 || checksum(&self.rx) != 0 {
            // The protocol names no framing error distinct from "err csum".
            self.respond_str("err csum");
            return;
        }
        let text = self.rx[1..=n].to_vec();
        self.dispatch_i2c(&text);
    }

    /// Console output names its sender in the character after `acp `, which
    /// the controller turns into a number by subtracting the code of `0`, with
    /// no range check. It is recorded as the request is parsed, before the
    /// checksum is looked at, so a corrupted request still records it.
    fn record_console_sender(&mut self) {
        if self.rx.len() >= 7 && self.rx[0] >= 5 && &self.rx[1..5] == b"acp " {
            self.console_sender = self.rx[5] as i32 - b'0' as i32;
            self.console_update();
        }
    }

    /// The first four bytes are a three-letter lowercase mnemonic and its
    /// separator, a space or the end. Matching is case-sensitive, and a
    /// request that is only the terminator is no command. Whatever a handler
    /// leaves unread is discarded.
    fn dispatch_i2c(&mut self, text: &[u8]) {
        let mut line = text.to_vec();
        line.push(b'\r');
        if text.len() < 3 || (line[3] != b' ' && line[3] != b'\r') {
            self.respond_str("err cmd");
            return;
        }
        let pos = if line[3] == b'\r' { 3 } else { 4 };
        let mut req = Req { text: &line, pos };
        match COMMANDS.iter().find(|c| &line[..3] == c.name && c.on_i2c && c.in_build(self.cfg.build)) {
            Some(c) => (c.handler)(self, &mut req),
            None => self.respond_str("err cmd"),
        }
    }

    // ---- commands -------------------------------------------------------

    fn handle_ver(&mut self, _r: &mut Req) {
        self.respond_str(&format!("ok {}", self.version()));
    }

    /// The physical switches, active-low, as hex with no leading zero. The
    /// Origin 200 build reads its role from bit 0x04 of the same byte: set on
    /// the master, clear on a slave. The topology decides the role, so the bit
    /// follows it whatever the configured switches say.
    fn handle_rsw(&mut self, _r: &mut Req) {
        let mut raw = 0xff ^ self.cfg.dip_switches;
        if !self.is2000() {
            raw = if self.cfg.master { raw | 0x04 } else { raw & !0x04 };
        }
        self.respond_str(&format!("ok {raw:x}"));
    }

    /// The console-input poll. A pending controller message comes first, as
    /// its bare text. Then console input: the firmware checks four of the six
    /// header bytes and copies from byte six, so the shape must be exact. The
    /// fifth byte names the input selection mode, not who is reading.
    fn handle_get(&mut self, _r: &mut Req) {
        if self.resp_owner >= 0 && self.take_message(self.resp_owner) {
            self.respond_str(self.message.text());
            return;
        }
        let n = self.rx_queue.len().min(MAX_TEXT - 6);
        let mut buf = b"acp 0 ".to_vec();
        buf[4] = match self.sel_mode {
            Select::Explicit => b'0' + self.sel_processor,
            Select::None => b'/',
            Select::Auto => b'.',
        };
        let before = self.rx_queue.len();
        buf.extend(self.rx_queue.drain(..n));
        self.respond(&buf);
        if before != 0 && self.rx_queue.is_empty() {
            self.console_update();
        }
    }

    /// Console output. A bare `ok` parses as zero accepted and livelocks the
    /// firmware, so the reply is always `ok <count>`. The selector and payload
    /// are taken from fixed offsets, as the firmware's output parser does,
    /// with no validation of either.
    fn handle_acp(&mut self, r: &mut Req) {
        let len = r.text.len() - 1;
        if len < 6 {
            self.respond_str("err arg");
            return;
        }
        let n = (len - 6).min(86);
        // Output from a processor the "see" filter excludes is accepted and
        // counted but not copied to the serial port. While an operator types
        // a command line, all output is accepted and lost.
        if !self.serial_command
            && (self.see_processor == SEE_ALL || self.see_processor as i32 == self.console_sender)
        {
            let text = r.text[6..6 + n].to_vec();
            self.serial_write(&text);
        }
        self.respond_str(&format!("ok {n:x}"));
    }

    /// `sel [<slot><a|b>|all|none|auto]`: the console input selection. With no
    /// argument it reports `ok auto`, `ok none` or `ok` and a processor name.
    /// `all` and `none` (the first two letters are what count) disable input
    /// selection, and `auto` makes it follow console output.
    fn handle_sel(&mut self, r: &mut Req) {
        let Some((first, second)) = r.read_selection() else {
            let name = match self.sel_mode {
                Select::Auto => "auto".to_string(),
                Select::None => "none".to_string(),
                Select::Explicit => String::from_utf8_lossy(&processor_name(self.sel_processor)).into_owned(),
            };
            self.respond_str(&format!("ok {name}"));
            return;
        };
        if let Some(p) = selection_processor(first, second) {
            self.select_input(Select::Explicit, p);
        } else if matches!((first, second), (b'A', b'L') | (b'N', b'O')) {
            self.select_input(Select::None, 0);
        } else if (first, second) == (b'A', b'U') {
            self.select_input(Select::Auto, 0);
        } else {
            self.respond_str("err arg");
            return;
        }
        self.respond_ok();
    }

    /// `see [<slot><a|b>|all|none]`: the console output filter. A processor
    /// name copies only that processor's output to the serial port and also
    /// selects it for input.
    fn handle_see(&mut self, r: &mut Req) {
        let Some((first, second)) = r.read_selection() else {
            let name = if self.see_processor == SEE_ALL {
                "all".to_string()
            } else {
                String::from_utf8_lossy(&processor_name(self.see_processor)).into_owned()
            };
            self.respond_str(&format!("ok {name}"));
            return;
        };
        if let Some(p) = selection_processor(first, second) {
            self.see_processor = p;
            self.select_input(Select::Explicit, p);
        } else if matches!((first, second), (b'A', b'L') | (b'N', b'O')) {
            self.see_processor = SEE_ALL;
        } else {
            self.respond_str("err arg");
            return;
        }
        self.respond_ok();
    }

    /// `pwr d [<sec>]`: power down after a hex delay. Exactly one character
    /// after the `d` is skipped, then the hex reader runs. 1 to 0x258
    /// (re)loads the countdown from this request, so a repeated request moves
    /// the deadline either way. 0 powers down at once, as does no argument
    /// while a countdown is pending; no argument otherwise starts a
    /// three-second countdown. Above 0x258 is `err arg` and changes nothing.
    /// An immediate power-down replies `ok` first and acts once the ack is
    /// read, as `rst` does. A countdown's expiry needs no ack.
    fn handle_pwr_down(&mut self, r: &mut Req) {
        if !self.powered() {
            // Nothing to power down: a bare "ok", with no range check.
            self.respond_ok();
            return;
        }
        r.skip();
        let delay = r.read_hex();
        if delay.is_some_and(|s| s > DELAY_MAX) {
            self.respond_str("err arg");
            return;
        }
        if !self.can_execute(Action::Shutdown) {
            self.respond_str("err cmd");
            return;
        }
        self.respond_ok();
        let immediate = match delay {
            Some(s) => s == 0,
            None => self.pending(T_PWR_DOWN),
        };
        if !immediate {
            // A timed request tells the Hubs: IRIX starts its
            // controller-initiated shutdown on "pd remote".
            self.arm_secs(T_PWR_DOWN, delay.unwrap_or(3) as u64);
            self.post_message(Message::PdRemote);
            self.set_status(Status::PdRemote);
            return;
        }
        // A pending countdown keeps running: the firmware removes power
        // whether or not this ack is read.
        self.arm_action(Action::Shutdown, "pwr d");
    }

    fn handle_pwr(&mut self, r: &mut Req) {
        let Some(sub) = r.read_char() else {
            self.respond_str(if self.powered() { "ok u" } else { "ok d" });
            return;
        };
        match sub {
            b'u' => {
                // Power-up runs the module's power sequence, which also
                // reinitialises this controller's bus interface, so like the
                // destructive commands it waits until the "ok" has been read.
                // A powered module ignores the request.
                self.respond_ok();
                if !self.powered() && self.can_execute(Action::PowerUp) {
                    self.arm_action(Action::PowerUp, "pwr u");
                }
            }
            b'q' => {
                // Power-on query: the PROM issues it once at power-on to
                // detect a cold start. "ok 1" on the first read after a cold
                // power-on, "ok 0" thereafter; clear-on-read is unconfirmed.
                let v = self.cold_power_on as u8;
                self.cold_power_on = false;
                self.respond_str(&format!("ok {v:x}"));
            }
            b'd' => self.handle_pwr_down(r),
            b'c' => {
                // The optional off interval is read like the "pwr d" delay
                // and runs from 5 to 0x258 seconds, 5 without one. Replies
                // "ok" before the power goes and acts once the ack is read.
                r.skip();
                let interval = r.read_hex();
                if interval.is_some_and(|s| !(5..=DELAY_MAX).contains(&s)) {
                    self.respond_str("err arg");
                    return;
                }
                if !self.can_execute(Action::Cycle) {
                    self.respond_str("err cmd");
                    return;
                }
                self.respond_ok();
                self.cycle_seconds = interval.unwrap_or(CYCLE_DEFAULT_SECONDS);
                self.arm_action(Action::Cycle, "pwr c");
            }
            _ => self.respond_str("err arg"),
        }
    }

    /// `hbt [<interval> [<delay>]]`: the heartbeat watchdog. `hbt 0` stops it.
    /// An interval of 5 to 0x258 seconds arms it: the stage goes back to
    /// waiting, the timer starts at the interval, and the delay from the NMI
    /// to the power cycle is the second value, zero allowed, or 30 seconds
    /// without one. Any other interval is `err arg`. With no readable value
    /// the request is a heartbeat, which only reloads the timer: it never
    /// returns the watchdog to an earlier stage.
    fn handle_hbt(&mut self, r: &mut Req) {
        let Some(interval) = r.read_hex() else {
            if self.hbt_interval != 0 {
                self.hbt_arm(self.hbt_interval as u64);
            }
            self.respond_ok();
            return;
        };
        if interval == 0 {
            self.hbt_stop();
            self.recheck_shutdown();
            self.respond_ok();
            return;
        }
        if !(5..=DELAY_MAX).contains(&interval) {
            self.respond_str("err arg");
            return;
        }
        self.hbt_interval = interval;
        self.hbt_delay = r.read_hex().unwrap_or(HBT_DEFAULT_DELAY);
        self.hbt_stage = Heartbeat::Waiting;
        self.hbt_arm(interval as u64);
        self.respond_ok();
    }

    /// `tas [<ch>]`: the test-and-set cell, free when it holds ASCII `0`. The
    /// argument is the next non-space character, a literal byte, so a space
    /// can never be an owner. `tas 0` frees the cell; any other byte, NUL
    /// included, takes it only if it is free; no argument only reports. Every
    /// reply is `ok ` and the byte the cell held before the request, framed
    /// with an explicit length so a NUL owner round-trips.
    fn handle_tas(&mut self, r: &mut Req) {
        let before = self.tas_lock;
        if let Some(ch) = r.read_char() {
            if ch == TAS_FREE {
                self.tas_lock = TAS_FREE;
            } else if self.tas_lock == TAS_FREE {
                self.tas_lock = ch;
            }
        }
        self.respond(&[b'o', b'k', b' ', before]);
    }

    fn log_unimp_once(&mut self, bit: u8, what: &str) {
        if self.unimp_logged & bit == 0 {
            self.unimp_logged |= bit;
            dlog!(LogModule::Module, "elsc: {what}");
        }
    }

    /// `nic <pulse> <sample> <delay>`: one-wire identity passthrough. Three
    /// values from the hex reader, any of them missing a bad argument. No
    /// device is modelled on the line, which idles high.
    fn handle_nic(&mut self, r: &mut Req) {
        let (Some(pulse), Some(sample), Some(_delay)) = (r.read_hex(), r.read_hex(), r.read_hex()) else {
            self.respond_str("err arg");
            return;
        };
        let pulse_sample = (pulse, sample);
        let level = match (&self.module, pulse_sample) {
            (Some(m), (pulse, sample)) => m.onewire_slot(pulse, sample),
            _ => {
                self.log_unimp_once(
                    UNIMP_NIC,
                    "\"nic\" with no one-wire device on this module; the line idles high, an approximation",
                );
                true
            }
        };
        self.respond_str(if level { "ok 1" } else { "ok 0" });
    }

    /// Module reset: panic reboot and fatal-error recovery only. A caller
    /// treats `ok` as proof the reset happened and does not fall back to its
    /// own hardware reset, so the reset must fire, once the guest has read
    /// `ok`.
    fn handle_rst(&mut self, _r: &mut Req) {
        if !self.powered() || !self.can_execute(Action::Reset) {
            self.respond_str("err cmd");
            return;
        }
        self.respond_ok();
        self.arm_action(Action::Reset, "rst");
    }

    fn nvram_byte(&self, offset: u8, write: Option<u8>) -> Option<u8> {
        self.nvram.access(elsc_nvram::FIRST_ADDR + 7, offset, write)
    }

    /// `aut [1|0]`: query, set or clear the automatic power-on bit in the
    /// NVRAM configuration byte. The bit takes effect only when the controller
    /// next starts, which the model does not distinguish from the machine's
    /// own start, so it is only kept in the NVRAM.
    fn handle_aut(&mut self, r: &mut Req) {
        let arg = r.read_char();
        if arg.is_some_and(|c| c != b'1' && c != b'0') {
            self.respond_str("err arg");
            return;
        }
        let mut config = self.nvram_byte(7, None).unwrap_or(0);
        let Some(c) = arg else {
            self.respond_str(if config & elsc_nvram::CONFIG_AUTO_POWER != 0 { "ok 1" } else { "ok 0" });
            return;
        };
        if c == b'1' {
            config |= elsc_nvram::CONFIG_AUTO_POWER;
        } else {
            config &= !elsc_nvram::CONFIG_AUTO_POWER;
        }
        self.nvram_byte(7, Some(config));
        self.respond_ok();
    }

    /// `clr`: cancel a pending power-down countdown, the heartbeat watchdog
    /// and the fan-high request. Accepted with the module on or off. It leaves
    /// the power-up countdown, the `tas` byte, the console selection, the
    /// `fsc` and `tst` flags, the NVRAM and module power alone. It turns
    /// serial echo on and clears supervisor mode.
    fn handle_clr(&mut self, _r: &mut Req) {
        self.cancel(T_PWR_DOWN);
        // The watchdog stops with its interval zeroed and its stage left as
        // it was, so a later heartbeat loads a zero timer and does nothing.
        self.hbt_interval = 0;
        self.cancel(T_HBT);
        self.fan_high = false;
        self.serial_echo = true;
        self.supervisor = false;
        if self.is2000() {
            self.display = [b' '; DISPLAY_CHARS];
            self.status = Status::None;
        }
        self.recheck_shutdown();
        self.respond_ok();
    }

    /// `fan [h|n]`: the fan-high request. The model has no fans or
    /// temperature inputs, so the request alone decides. Over I2C neither
    /// form writes the NVRAM; on the serial line the request is also kept
    /// there.
    fn handle_fan(&mut self, r: &mut Req) {
        let Some(c) = r.read_char() else {
            self.respond_str(if self.fan_high { "ok h" } else { "ok n" });
            return;
        };
        if c != b'h' && c != b'n' {
            self.respond_str("err arg");
            return;
        }
        self.fan_high = c == b'h';
        if self.serial_running {
            if let Some(config) = self.nvram_byte(7, None) {
                let config = if self.fan_high {
                    config | elsc_nvram::CONFIG_FAN_HIGH
                } else {
                    config & !elsc_nvram::CONFIG_FAN_HIGH
                };
                self.nvram_byte(7, Some(config));
            }
        }
        self.respond_ok();
    }

    /// `fsc [1|0]`: declare a multimodule controller present or absent. The
    /// query form and any other argument answer a plain `ok ` over I2C (the
    /// firmware writes the flag where the reply's separator then goes). The
    /// flag only changes what an Origin 2000 build does, so it is only stored.
    fn handle_fsc(&mut self, r: &mut Req) {
        match r.read_char() {
            Some(c @ (b'1' | b'0')) => {
                self.fsc_present = c == b'1';
                if self.fsc_present {
                    self.log_unimp_once(
                        UNIMP_FSC,
                        "\"fsc 1\" is stored only; the serial display mirror and the key-switch power-up rule are not modelled",
                    );
                }
            }
            _ if self.serial_running => {
                // The serial reply keeps the digit the I2C reply loses.
                self.respond_str(if self.fsc_present { "ok1" } else { "ok0" });
                return;
            }
            _ => {}
        }
        self.respond_ok();
    }

    /// `tst <c>`: the factory's short timeouts, set by `1` and cleared by any
    /// other character. They shorten waits the model does not have, so the
    /// flag is only stored.
    fn handle_tst(&mut self, r: &mut Req) {
        let Some(c) = r.read_char() else {
            self.respond_str("err arg");
            return;
        };
        self.short_timeouts = c == b'1';
        if self.short_timeouts {
            self.log_unimp_once(
                UNIMP_TST,
                "\"tst 1\" is stored only; the shortened fan-check and power-up waits are not modelled",
            );
        }
        self.respond_ok();
    }

    /// `tmp`: `ok n` while neither temperature input is active, which in a
    /// model with no thermal sensors is always. A master that has heard from
    /// its slave appends the slave's last reported letter.
    fn handle_tmp(&mut self, _r: &mut Req) {
        let mut resp = b"ok n".to_vec();
        if self.slave_link {
            resp.extend_from_slice(&[b' ', self.slave_temp]);
        }
        self.respond(&resp);
    }

    /// `sla <c>`: the report a slave controller sends its master. The
    /// controller believes whoever sends it. A master takes `n`, `h` and `o`
    /// as its slave's temperature and 0x07 as the slave announcing itself,
    /// after which `tmp` reports the slave too. The shutdowns and fan reports
    /// a master starts on `o`, 0x04, 0x05 and 0x06 need the
    /// controller-to-controller link, which the model does not have.
    fn handle_sla(&mut self, r: &mut Req) {
        if self.is2000() {
            self.respond_ok();
            return;
        }
        if let Some(c) = r.read_char() {
            if !self.cfg.master {
                self.slave_report = c;
            } else if matches!(c, b'n' | b'h' | b'o') {
                self.slave_temp = c;
            } else if c == 0x07 {
                self.slave_link = true;
            }
            if self.cfg.master && (c == b'o' || (0x04..=0x06).contains(&c)) {
                dlog!(
                    LogModule::Module,
                    "elsc: \"sla\" report {c:#04x} from a slave controller; the shutdown or fan report it starts is not modelled"
                );
            }
        }
        self.respond_ok();
    }

    /// `nmi`: an unpowered module answers `err cmd`. Otherwise `ok`, and a
    /// second later the module's processors get a 100 ms NMI pulse; another
    /// `nmi` inside that second starts the wait again. The wait starts as the
    /// command is answered, as the firmware's does.
    fn handle_nmi(&mut self, _r: &mut Req) {
        if !self.powered() {
            self.respond_str("err cmd");
            return;
        }
        self.arm_ms(T_NMI, NMI_WAIT_MS);
        self.set_status(Status::Nmi);
        self.respond_ok();
    }

    // ---- serial-only commands ------------------------------------------

    /// `dsp <text>`: the Origin 2000 build's eight-character front-panel
    /// message. The text starts right after the space that ends the mnemonic,
    /// so spaces in it count. Up to eight characters overwrite the display
    /// from its first position and leave the rest alone. The text stops at the
    /// first unprintable character, the carriage return included. An empty
    /// message is a bad argument.
    fn handle_dsp(&mut self, r: &mut Req) {
        let n = r.text[r.pos.min(r.text.len())..]
            .iter()
            .take(DISPLAY_CHARS)
            .take_while(|c| (0x20..0x7f).contains(*c))
            .count();
        if n == 0 {
            self.respond_str("err arg");
            return;
        }
        let text = r.text[r.pos..r.pos + n].to_vec();
        self.display[..n].copy_from_slice(&text);
        // With a multimodule controller declared, host display text is copied
        // to it over the serial line, with severity digit 4.
        if self.fsc_present {
            self.serial_puts("\x14dsp ");
            self.serial_write(&text);
            self.serial_puts("4\r\n");
        }
        self.status = Status::None;
        self.respond_ok();
    }

    /// `dsc <pos> <ch>`: one display character. Both arguments come from the
    /// character reader, so the position is the digit 1 to 8, for display
    /// positions 0 to 7, and the character is printable and never a space.
    fn handle_dsc(&mut self, r: &mut Req) {
        let (Some(pos), Some(ch)) = (r.read_char(), r.read_char()) else {
            self.respond_str("err arg");
            return;
        };
        if !(b'1'..=b'8').contains(&pos) || !(0x20..0x7f).contains(&ch) {
            self.respond_str("err arg");
            return;
        }
        self.display[(pos - b'1') as usize] = ch;
        self.status = Status::None;
        self.respond_ok();
    }

    /// `key`: the key-switch position. The model's key is always on.
    fn handle_key(&mut self, _r: &mut Req) {
        self.respond_str("ok on");
    }

    /// `vlm <supply> [h|l|n]`: query or set a supply's margin. The supply is
    /// `3`, `5` or `v`; a second character sets high, low or normal. The
    /// margins are stored and answered, and nothing electrical follows.
    fn handle_vlm(&mut self, r: &mut Req) {
        const SUPPLY_NAMES: &[u8] = b"35v";
        let Some(supply) = r.read_char().and_then(|c| SUPPLY_NAMES.iter().position(|s| *s == c)) else {
            self.respond_str("err arg");
            return;
        };
        let Some(m) = r.read_char() else {
            self.respond_str(&format!("ok {}", self.margins[supply] as char));
            return;
        };
        if !matches!(m, b'h' | b'l' | b'n') {
            self.respond_str("err arg");
            return;
        }
        self.margins[supply] = m;
        self.respond_ok();
    }

    /// `fre [<hex>]`, serial line: the clock-select byte. Writing it also
    /// reprograms the midplane clock chip, which the model does not have.
    fn handle_fre(&mut self, r: &mut Req) {
        match r.read_hex() {
            None => self.reply_nvram_byte(11),
            Some(v) => {
                self.nvram_byte(11, Some(v as u8));
                self.log_unimp_once(
                    UNIMP_FRE,
                    "\"fre\" write is stored only; the midplane clock chip is not modelled",
                );
                self.respond_ok();
            }
        }
    }

    /// `ech`: set echo, `1` on, `0` off; with no argument it toggles.
    fn handle_ech(&mut self, r: &mut Req) {
        match r.read_char() {
            None => self.serial_echo = !self.serial_echo,
            Some(c @ (b'1' | b'0')) => self.serial_echo = c == b'1',
            Some(_) => {
                self.respond_str("err arg");
                return;
            }
        }
        self.respond_ok();
    }

    /// `pas`: with no argument, leave supervisor mode. `pas <word>` enters it
    /// when the word is exactly four characters and matches the password at
    /// offsets 1 to 4 of the NVRAM page. `pas s <word>`, in supervisor mode
    /// already, stores the first four characters of a word at least that
    /// long, which must follow exactly one space.
    fn handle_pas(&mut self, r: &mut Req) {
        const LEN: usize = 4;
        let word = r.read_word().to_vec();
        if word.is_empty() {
            self.supervisor = false;
            self.respond_ok();
            return;
        }
        if word == b"s" && r.peek() == b' ' {
            if !self.supervisor {
                self.respond_str("err perm");
                return;
            }
            // The word follows exactly one space: no more are skipped.
            r.skip();
            let new: Vec<u8> = r.text[r.pos..]
                .iter()
                .copied()
                .take_while(|c| !matches!(c, b' ' | b'\r' | b'\n' | 0))
                .collect();
            if new.len() < LEN {
                self.respond_str("err arg");
                return;
            }
            for (i, b) in new[..LEN].iter().enumerate() {
                if self.nvram_byte(1 + i as u8, Some(*b)).is_none() {
                    break;
                }
            }
            self.supervisor = true;
            self.respond_ok();
            return;
        }
        if word.len() != LEN {
            self.respond_str("err arg");
            return;
        }
        for (i, b) in word.iter().enumerate() {
            if self.nvram_byte(1 + i as u8, None) != Some(*b) {
                self.respond_str("err arg");
                return;
            }
        }
        self.supervisor = true;
        self.respond_ok();
    }

    fn reply_nvram_byte(&mut self, offset: u8) {
        let v = self.nvram_byte(offset, None).unwrap_or(0);
        self.respond_str(&format!("ok {v:x}"));
    }

    /// `mod [<hex>]`: the module number. A non-hex argument is the query.
    fn handle_mod(&mut self, r: &mut Req) {
        match r.read_hex() {
            None => self.reply_nvram_byte(8),
            Some(v) => {
                self.nvram_byte(8, Some(v as u8));
                self.respond_ok();
            }
        }
    }

    /// `dbg [<a> [<b>]]`: the two debug-switch bytes.
    fn handle_dbg(&mut self, r: &mut Req) {
        let Some(a) = r.read_hex() else {
            let v5 = self.nvram_byte(5, None).unwrap_or(0);
            let v6 = self.nvram_byte(6, None).unwrap_or(0);
            self.respond_str(&format!("ok {v5:x} {v6:x}"));
            return;
        };
        self.nvram_byte(5, Some(a as u8));
        if let Some(b) = r.read_hex() {
            self.nvram_byte(6, Some(b as u8));
        }
        self.respond_ok();
    }

    /// `nvm <addr> [<data>]`: one byte of the management NVRAM. The address's
    /// low byte is the offset and its high byte picks the device, the firmware
    /// forming the device address byte as 0xa0 ORed with twice the high byte:
    /// 000 to 7ff reach 0x50 to 0x57, and higher addresses alias back onto
    /// those. A device that does not answer gives a bare `err `.
    fn handle_nvm(&mut self, r: &mut Req) {
        let Some(addr) = r.read_hex() else {
            self.respond_str("err arg");
            return;
        };
        let device = ((0xa0 | ((addr >> 8) << 1)) as u8) >> 1;
        let data = r.read_hex();
        match self.nvram.access(device, addr as u8, data.map(|d| d as u8)) {
            None => self.respond_str("err "),
            Some(_) if data.is_some() => self.respond_ok(),
            Some(v) => self.respond_str(&format!("ok {v:x}")),
        }
    }

    /// `cfg`: the management NVRAM's raw bytes from device 0x50 offset 0, up
    /// to and including the first zero byte or 0x700 bytes, then a line end.
    /// No Ctrl+T and no `ok`.
    fn handle_cfg(&mut self, _r: &mut Req) {
        const LIMIT: usize = 0x700;
        self.serial_raw = true;
        for offset in 0..LIMIT {
            let b = self
                .nvram
                .access(elsc_nvram::FIRST_ADDR + (offset >> 8) as u8, offset as u8, None)
                .unwrap_or(0);
            self.serial_write(&[b]);
            if b == 0 {
                break;
            }
        }
        self.serial_puts("\r\n");
    }

    // ---- serial line ----------------------------------------------------

    /// Whether a serial command needs supervisor mode. `pas s` checks for
    /// itself; `nmi` and `rst` check module power before this.
    fn serial_gated(&self, name: &[u8], r: &Req) -> bool {
        let mut probe = Req { text: r.text, pos: r.pos };
        match name {
            b"clr" | b"hbt" | b"nmi" | b"rst" => true,
            b"pwr" => probe.read_char().is_some_and(|c| matches!(c, b'u' | b'd' | b'c')),
            b"fan" | b"aut" => probe.read_char().is_some(),
            b"mod" | b"dbg" | b"fre" => probe.read_hex().is_some(),
            b"vlm" => probe.read_char().is_some() && probe.read_char().is_some(),
            b"nvm" => probe.read_hex().is_some() && probe.read_hex().is_some(),
            _ => false,
        }
    }

    /// Run one serial line. It is parsed as an I2C request is, with the byte
    /// that ended it in place of the I2C carriage return, so only a carriage
    /// return separates a bare mnemonic; then the supervisor gate.
    fn serial_dispatch(&mut self, line: &[u8]) {
        // Ctrl+T and Enter alone.
        if line == b"\r" {
            self.respond_str(if self.serial_echo { "ok " } else { "err cmd" });
            return;
        }
        if line.len() < 4 || (line[3] != b' ' && line[3] != b'\r') {
            self.respond_str("err cmd");
            return;
        }
        let pos = if line[3] == b'\r' { 3 } else { 4 };
        let mut req = Req { text: line, pos };
        let Some(c) = COMMANDS.iter().find(|c| &line[..3] == c.name && c.on_serial && c.in_build(self.cfg.build)) else {
            self.respond_str("err cmd");
            return;
        };
        if matches!(c.name, b"nmi" | b"rst") && !self.powered() {
            self.respond_str("err cmd");
            return;
        }
        if !self.supervisor && self.serial_gated(c.name, &req) {
            self.respond_str("err perm");
            return;
        }
        (c.handler)(self, &mut req);
    }

    /// Run the waiting serial line and print its reply: Ctrl+T, the reply as
    /// the I2C contract gives it but with no separator space after a bare
    /// `ok`, and a line end. No prompt follows. An action the command asked
    /// for runs once the reply is out.
    fn serial_run(&mut self) {
        let mut line = std::mem::take(&mut self.serial_line);
        line.push(self.serial_term);
        self.serial_pending = false;

        self.serial_running = true;
        self.serial_raw = false;
        self.serial_reply.clear();
        self.serial_action = Action::None;
        self.serial_dispatch(&line);
        self.serial_running = false;

        if !self.serial_raw {
            let mut reply = std::mem::take(&mut self.serial_reply);
            if reply == b"ok " {
                reply.pop();
            }
            self.serial_write(&[CTRL_T]);
            self.serial_write(&reply);
            self.serial_puts("\r\n");
        }
        let action = std::mem::replace(&mut self.serial_action, Action::None);
        if action != Action::None {
            self.run_action(action);
        }
    }

    /// A completed line runs once the bus is free, this controller's own I2C
    /// target is idle and no Hub message is pending; an I2C command arriving
    /// at the same moment goes first.
    ///
    /// An Origin 2000 module's scheduler decides that turn instead, and stops
    /// publishing grants once it has given it.
    fn serial_poll(&mut self) {
        if !self.serial_pending {
            return;
        }
        let turn = match &self.module {
            Some(m) => !self.slave_addressed && m.serial_turn(),
            None => {
                self.message_mask == 0 && !self.slave_addressed && !self.bus_busy.load(Ordering::Relaxed)
            }
        };
        if !turn {
            return;
        }
        self.serial_run();
        if let Some(m) = &self.module {
            m.serial_done();
        }
    }

    /// An Origin 200 slave has a serial line that takes no command.
    fn serial_is_slave(&self) -> bool {
        !self.is2000() && !self.cfg.master
    }

    /// Ctrl+T in pass-through: the prompt, or `Slave!` from a slave.
    fn serial_enter_command(&mut self) {
        if self.serial_echo {
            self.serial_puts("\r\nMSC> ");
            if self.serial_is_slave() {
                self.serial_puts("Slave!\r\n");
                return;
            }
        }
        self.serial_command = true;
    }

    /// One byte from the operator. In pass-through everything but Ctrl+T is
    /// console input. In command mode bytes build the line; a carriage
    /// return, line feed or NUL ends it, and a second Ctrl+T abandons it,
    /// keeping what was typed for the next line and passing one Ctrl+T on as
    /// input.
    fn serial_input(&mut self, c: u8) {
        if !self.serial_command {
            if c == CTRL_T {
                self.serial_enter_command();
            } else {
                self.queue_input(c);
            }
            return;
        }
        match c {
            0x08 | 0x7f => {
                if self.serial_line.pop().is_some() && self.serial_echo {
                    self.serial_puts("\x08 \x08");
                }
            }
            b'\r' | b'\n' | 0 => {
                self.serial_term = c;
                if self.serial_echo {
                    if c == b'\r' {
                        self.serial_puts("\r\n");
                    } else {
                        self.serial_write(&[c]);
                    }
                }
                self.serial_command = false;
                if self.serial_is_slave() {
                    // A slave never runs serial commands.
                    self.serial_line.clear();
                    return;
                }
                self.serial_pending = true;
            }
            CTRL_T => {
                self.serial_command = false;
                self.queue_input(CTRL_T);
                if self.serial_echo {
                    self.serial_puts("\x08\x08 \x08\x08 \x08\x08 \x08\x08 \x08");
                }
            }
            _ => {
                if self.serial_line.len() < MAX_TEXT {
                    self.serial_line.push(c);
                }
                if self.serial_echo {
                    self.serial_write(&[c]);
                }
            }
        }
    }

    // ---- start ----------------------------------------------------------

    /// The controller's own start, which the machine's reset stands for (an
    /// AC cycle).
    fn start(&mut self, clock_now: u64) {
        self.reset_queues();
        self.cancel(T_PWR_DOWN);
        self.serial_command = false;
        self.serial_line.clear();
        self.serial_pending = false;
        self.serial_echo = true;
        self.supervisor = false;
        self.cancel(T_POWER_UP);
        self.cancel(T_START);
        self.module_off = false;
        self.power_sequence = false;
        if self.cfg.start_delay_ms != 0 {
            // The module starts off. A controller that powers it up on its
            // own counts the delay; an Origin 200 slave waits for its master.
            // An Origin 2000 module owns its power state, so only the
            // countdown is the controller's.
            self.module_off = !self.is2000();
            if self.is2000() || self.cfg.master {
                self.arm_ms(T_START, self.cfg.start_delay_ms as u64);
            }
        }
        self.status = Status::None;
        self.margins = [b'n'; SUPPLIES];
        // An Origin 2000 controller starts with its version on the display;
        // the Origin 200 build has none, and a blank is a model choice.
        self.display = if self.is2000() {
            let mut d = [b' '; DISPLAY_CHARS];
            d.copy_from_slice(VERSION_ORIGIN2000.as_bytes());
            d
        } else {
            [b' '; DISPLAY_CHARS]
        };
        self.cycle_seconds = CYCLE_DEFAULT_SECONDS;
        self.cancel(T_RST);
        self.cancel(T_NMI);
        self.cancel(T_NMI_PULSE);
        if self.nmi_pulse_active {
            self.nmi_pulse_active = false;
            self.events.push(ModuleOutput::Nmi(false));
        }
        self.hbt_stop();
        self.hbt_delay = 0;
        self.fan_high = false;
        self.fsc_present = false;
        self.short_timeouts = false;
        self.slave_link = false;
        self.slave_temp = b'n';
        self.slave_report = 0;
        self.sel_mode = Select::Auto;
        self.see_processor = SEE_ALL;
        // No processor has sent console output since the start.
        self.console_sender = -1;

        // The start announces the controller on its serial line.
        self.serial_puts("\x14MSC ");
        self.serial_puts(self.version());
        self.serial_puts("\r\n");
        // The rotation starts with code 0, and a later start leaves a running
        // rotation on its schedule.
        if self.rings() && !self.pending(T_ARB) {
            if !self.arb_blocked() {
                self.arb_publish();
            }
            self.deadlines[T_ARB] = Some(clock_now + ARB_TICK_NS);
        }
    }

    /// The part of a start that a module event reaches through the bus the
    /// controller shares with module parts, which it rides through.
    fn reset_queues(&mut self) {
        self.rx.clear();
        self.tx.clear();
        self.txpos = 0;
        self.resp_owner = -1;
        self.arb_code = 0;
        self.arb_signal = 0;
        self.rx_queue.clear();
        self.slave_addressed = false;
        self.deferred_action = Action::None;
        self.deferred_committed = false;
        self.deferred_scheduled = false;
        self.cancel(T_MESSAGE);
        self.message = Message::None;
        self.message_mask = 0;
        self.tas_lock = TAS_FREE;
        if let Some(m) = &self.module {
            m.slave_active(false);
        }
        self.console_update();
    }

    fn describe(&self) -> String {
        use std::fmt::Write;
        let mut s = String::new();
        let resp_pending = self.resp_owner >= 0 && self.txpos < self.tx.len();
        let code0_signal2 = (!self.rx_queue.is_empty() && self.console_target() == 0)
            || self.message_mask & 1 != 0
            || (resp_pending && self.resp_owner == 0);
        let power = if self.powered() {
            "on"
        } else if self.power_sequence {
            "off, powering up"
        } else if self.pending(T_POWER_UP) {
            "off, power-up counting down"
        } else {
            "off"
        };
        let _ = writeln!(s, "elsc");
        let _ = writeln!(s, "  module power: {power}");
        if self.pending(T_START) {
            let _ = writeln!(s, "  start-up power-up pending");
        }
        let _ = writeln!(s, "  rx queue: {} byte(s) waiting for \"get\"", self.rx_queue.len());
        if !self.rx_queue.is_empty() {
            let _ = write!(s, "    input: \"");
            for c in self.rx_queue.iter().take(64) {
                if (0x20..0x7f).contains(c) {
                    s.push(*c as char);
                } else {
                    let _ = write!(s, "\\x{c:02x}");
                }
            }
            let _ = writeln!(s, "{}", if self.rx_queue.len() > 64 { "\"..." } else { "\"" });
        }
        if self.is2000() {
            let shown: String = self
                .display
                .iter()
                .map(|c| if (0x20..0x7f).contains(c) { (*c as char).to_string() } else { format!("\\x{c:02x}") })
                .collect();
            let _ = writeln!(s, "  display: \"{shown}\"");
            if self.status != Status::None {
                let _ = writeln!(
                    s,
                    "  status message: {}, shown in place of the text above (its own text is not known)",
                    self.status.name()
                );
            }
        }
        if self.message != Message::None {
            let _ = writeln!(s, "  controller message: \"{}\"", self.message.text());
        }
        if resp_pending {
            let _ = writeln!(
                s,
                "  pending response: {} byte(s) for processor code {}",
                self.tx.len() - self.txpos,
                self.resp_owner
            );
        } else {
            let _ = writeln!(s, "  pending response: none");
        }
        match self.sel_mode {
            Select::Explicit => {
                let _ = write!(s, "  console input: processor {}", self.sel_processor);
            }
            Select::None => {
                let _ = write!(s, "  console input: disabled");
            }
            Select::Auto => {
                let _ = write!(s, "  console input: automatic, last output from processor {}", self.console_sender);
            }
        }
        if self.see_processor == SEE_ALL {
            let _ = writeln!(s, "; output: all");
        } else {
            let _ = writeln!(s, "; output: processor {}", self.see_processor);
        }
        if self.rings() {
            let _ = writeln!(s, "  arbitration rotation: next position {} of {}", self.arb_code, ARB_POSITIONS);
        }
        let _ = writeln!(
            s,
            "  fan high: {}, fsc: {}, short timeouts: {}, slave report: {:#04x}",
            self.fan_high, self.fsc_present, self.short_timeouts, self.slave_report
        );
        let _ = writeln!(
            s,
            "  console signal (code 0): {:#04x} = signal{}",
            SIGNAL_BASE + if code0_signal2 { 8 } else { 0 },
            if code0_signal2 { 2 } else { 1 }
        );
        s
    }
}

pub struct Elsc {
    state: Mutex<State>,
    clock: Clock,
    backend: Mutex<Arc<dyn SerialBackend>>,
    /// Told, with no controller lock held, that [`Self::next_due`] may have
    /// moved.
    wake: OnceLock<Box<dyn Fn() + Send + Sync>>,
}

impl Elsc {
    /// A controller on the bus whose busy flag is `busy`, keeping its
    /// configuration in `nvram`. Call [`Resettable::power_on`] to start it.
    pub fn new(cfg: ElscConfig, nvram: Arc<ElscNvram>, clock: Clock, busy: BusBusy) -> Self {
        Self {
            state: Mutex::new(State::new(cfg, nvram, busy)),
            clock,
            backend: Mutex::new(null_serial_backend()),
            wake: OnceLock::new(),
        }
    }

    /// Call `wake` after any access that may have moved [`Self::next_due`]:
    /// one that changed it, or one that handed the module an action to carry
    /// out. Once, before the machine runs.
    pub fn set_wake(&self, wake: Box<dyn Fn() + Send + Sync>) {
        let _ = self.wake.set(wake);
    }

    /// How long until the controller next has an output to give, or a
    /// committed action to run: zero if it is due now, `None` if nothing is
    /// pending. A machine that has no bus access to piggy-back on visits the
    /// controller (`take_events`) when this says to.
    pub fn next_due(&self) -> Option<Duration> {
        let due = self.state.lock().due()?;
        Some(Duration::from_nanos(due.saturating_sub((self.clock)())))
    }

    /// Attach the module that owns this controller's power, reset and bus
    /// scheduling. Wiring: call it before [`Resettable::power_on`].
    pub fn set_module_ops(&self, module: Arc<dyn ModuleOps>) {
        self.state.lock().module = Some(module);
    }

    /// Clear the protocol queues without resetting the controller. A module
    /// owner's recovery, reset and power paths reinitialise this much and no
    /// more: the framing state goes and the rest stays.
    pub fn reinitialize(&self) {
        self.run(|st| st.reinitialize());
    }

    /// Whether the controller powers its module up after a delay of its own
    /// rather than at once when it starts.
    pub fn start_delayed(&self) -> bool {
        self.state.lock().cfg.start_delay_ms != 0
    }

    /// The controller's start-up power-up with none of its wait, which an
    /// Origin 2000 module-management device calls from its reset so the
    /// module is powered before any processor runs.
    pub fn power_up_at_start(&self) {
        self.run(|st| st.power_up_complete(false));
    }

    /// Where the AUX connector's transmit bytes go.
    pub fn set_serial_backend(&self, backend: Arc<dyn SerialBackend>) {
        *self.backend.lock() = backend;
    }

    /// Run `f` with the timers brought up to the present, then send what it
    /// wrote to the AUX port with the lock released.
    fn run<R>(&self, f: impl FnOnce(&mut State) -> R) -> R {
        let (value, out, actions, module, moved) = {
            let mut st = self.state.lock();
            st.advance((self.clock)());
            let value = f(&mut st);
            let due = st.due();
            let moved = st.reported_due != Some(due);
            st.reported_due = Some(due);
            (value, std::mem::take(&mut st.out), std::mem::take(&mut st.outbox), st.module.clone(), moved)
        };
        let acted = !actions.is_empty();
        // What the module does reaches back into the controller, so it runs
        // with the lock released.
        if let Some(m) = module {
            for action in actions {
                m.execute(action);
            }
        }
        if !out.is_empty() {
            let backend = self.backend.lock().clone();
            for byte in out {
                backend.send_byte(byte);
            }
        }
        if moved || acted {
            if let Some(wake) = self.wake.get() {
                wake();
            }
        }
        value
    }

    /// Let time pass and carry out what is due. The bus does this around
    /// every register access; a machine whose guest may not touch the bus for
    /// a long time (a countdown expiring, an operator typing) also calls it
    /// from its own tick.
    pub fn poll(&self) {
        self.run(|st| {
            if st.deferred_scheduled {
                st.run_committed_action();
            }
            st.serial_poll();
        });
    }

    /// One byte from the operator on the AUX connector.
    pub fn serial_input(&self, byte: u8) {
        self.run(|st| {
            st.serial_input(byte);
            st.serial_poll();
        });
    }

    /// The module-power, reset and NMI outputs, in the order they happened,
    /// with time brought up to the present first.
    pub fn take_events(&self) -> Vec<ModuleOutput> {
        self.run(|st| {
            if st.deferred_scheduled {
                st.run_committed_action();
            }
            st.serial_poll();
            std::mem::take(&mut st.events)
        })
    }

    pub fn module_powered(&self) -> bool {
        self.run(|st| st.powered())
    }

    /// Whether module power is due back without anyone asking: a power-up
    /// countdown, or a heartbeat watchdog that will power-cycle the module.
    pub fn power_return_pending(&self) -> bool {
        self.run(|st| st.power_return_pending())
    }

    /// A module event that reaches the controller only through a bus it
    /// shares with module parts: the controller runs on standby power and
    /// keeps its state, reinitialising its bus interface and queues, freeing
    /// the lock and dropping what it had queued for the Hubs.
    pub fn ride_through_reset(&self) {
        self.run(|st| st.reset_queues());
    }

    /// A state dump for the monitor.
    pub fn describe(&self) -> String {
        self.run(|st| st.describe())
    }
}

impl Resettable for Elsc {
    /// The controller's own start, which the machine's reset stands for (an
    /// AC cycle): every timer and flag starts over.
    fn power_on(&self) {
        let now = (self.clock)();
        self.run(|st| st.start(now));
    }
}

impl I2cTarget for Elsc {
    fn claims(&self, addr7: u8) -> bool {
        addr7 == I2C_ADDR
    }

    fn start(&self, _addr7: u8, recv: bool) -> bool {
        self.run(|st| {
            st.slave_addressed = true;
            if let Some(m) = &st.module {
                m.slave_active(true);
            }
            if recv {
                // Direction switch: the request just written is complete.
                st.request_complete();
                st.fire_committed();
            } else {
                // A new request is arriving; its start is a boundary too.
                st.fire_committed();
                st.rx.clear();
            }
        });
        true
    }

    fn send(&self, byte: u8) -> bool {
        self.run(|st| {
            if st.rx.len() > MAX_FRAME {
                dlog!(LogModule::Module, "elsc: request longer than the protocol allows, byte dropped");
            } else {
                st.rx.push(byte);
            }
        });
        true
    }

    fn recv(&self) -> u8 {
        self.run(|st| {
            if st.txpos < st.tx.len() {
                let byte = st.tx[st.txpos];
                st.txpos += 1;
                if st.txpos == st.tx.len() {
                    st.response_fully_read();
                }
                byte
            } else {
                // The driver's own length byte said stop; this is over-reading.
                0xff
            }
        })
    }

    fn nack(&self) {
        self.run(|st| st.fire_committed());
    }

    fn finish(&self) {
        self.run(|st| {
            st.slave_addressed = false;
            if let Some(m) = &st.module {
                m.slave_active(false);
            }
            st.rx.clear();
            // An unconsumed response stays pending for its requester until
            // read to its end or superseded; a committed action fires here
            // for a response the master ends with a STOP.
            st.fire_committed();
            if st.txpos >= st.tx.len() {
                st.tx.clear();
                st.txpos = 0;
                st.resp_owner = -1;
            }
        });
    }

    fn service(&self) -> Service {
        self.run(|st| {
            if st.deferred_scheduled {
                st.run_committed_action();
            }
            st.serial_poll();
            Service {
                grant: st.pending_grant.take(),
                release_bus: std::mem::take(&mut st.release_bus),
            }
        })
    }
}

#[cfg(test)]
pub(crate) mod tests;
