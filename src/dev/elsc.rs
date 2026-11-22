//! Module system controller (MSC, the ELSC) of an Origin 200 or an Origin
//! 2000 module: a checksummed ASCII command and reply protocol over I2C, the
//! arbitration grants the processors wait for, and the AUX serial port.
//!
//! Two builds of one program: the Origin 200's (`V 3.1S`, 4 s power-up hold,
//! own five-position grant ring, `aut`) and the Origin 2000's (`VER 3.2`, 4.4 s
//! hold, eight-character display, `dsp`, `dsc`, `key`, `vlm`, serial `fre`),
//! which leaves grants, power and reset to its module ([`ModuleOps`]). A command
//! a build lacks answers `err cmd`. The I2C target sits at 0x08 next to its
//! [`ElscNvram`]; power, reset and NMI come out as [`ModuleOutput`]s via
//! [`Elsc::take_events`].
//!
//! No specification of the protocol or firmware is public and no controller has
//! been measured. Every timing, the version text, the grant schedule, the queue
//! depths and the reply text are the model's own choices, approximations unless
//! a comment says otherwise:
//!
//! * the version fields `V 3.1S` and `VER 3.2` and their padding;
//! * the one-millisecond grant tick, the five ring positions (processor codes
//!   0 to 3, then a fifth published as code 4's byte), and that a busy tick
//!   publishes nothing and does not advance;
//! * the 4 s (Origin 200) and 4.4 s (Origin 2000) power-up reset hold, the 1 s
//!   waits of `rst` and `nmi`, the 100 ms NMI pulse, the 1 s life of a message,
//!   the 5 s warning and the 5 s off interval of the heartbeat watchdog;
//! * the 256-byte console input queue that overwrites its oldest byte;
//! * that cold-power latch `pwr q` clears on read;
//! * that a deferred destructive command acts once its `ok` has been read;
//! * that `nic` reads 1 (no device on the one-wire line) and `sla` reports
//!   from a slave are stored or logged only.
//!
//! Timers run on the machine's timer manager, one kind each, and expire at their own
//! deadlines. A request is attributed to the CPU thread that made it
//! ([`crate::cpu::smp::current_cpu`]); other threads count as CPU 0. Migration
//! state is not saved yet.
//!
//! Not modelled: the second Origin 200 module's controller and the master's
//! forwarding to it, sensors, the electrical effect of the supply margins, the
//! status messages' display texts (unknown), and the Origin 2000 module's
//! shared bus, which the module-management device owns.
//!
//! Ported from hw/misc/sgi-elsc.c in jamesbraid/qemu-sgi-origami.

use crate::dev::elsc_nvram::{self, ElscNvram};
use crate::clock::Clock;
use crate::dev::i2c::{BusBusy, I2cTarget, Service};
use crate::dev::serial_host::{null_serial_backend, SerialBackend};
use crate::devlog::LogModule;
use crate::dlog;
use crate::hptimer::{Timer, TimerManager};
use crate::traits::{Device, Resettable};
use parking_lot::Mutex;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

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

/// What a module controller does to its module and processors; emitted by the
/// Origin 200's [`Elsc`] and the Origin 2000 module's controller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModuleOutput {
    /// Module power output, low as the module loses power. A power-down repeats
    /// when a pending power-up is cancelled, so the machine asks again.
    Power(bool),
    /// A pulse on the module's reset line (instantaneous in the model; a
    /// controller that holds the line for a while reports its rising edge).
    Reset,
    /// The NMI output, high for the 100 ms of a pulse.
    Nmi(bool),
}

/// Which build of the controller firmware this is: one program built two
/// ways, differing in commands, version text and display.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Build {
    Origin200,
    Origin2000,
}

/// What an Origin 2000 module owns that is module-scoped (reset reach, bus arbitration, console,
/// one-wire). Called under the controller's lock, so never call back; `execute` is queued and run
/// after the lock drops.
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
    /// Milliseconds after start that the controller powers its module up (when
    /// the NVRAM says automatic power-on); zero starts it powered.
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
/// Hand the machine what the controller has for it: queued outputs or a committed action.
const T_VISIT: usize = 9;
/// A visit waits a millisecond, so a wake that leaves its outputs queued cannot spin the timer thread.
const VISIT_NS: u64 = NS_PER_MS;
const TIMERS: usize = 10;

// Commands whose missing effect is logged once.
const UNIMP_FSC: u8 = 1 << 0;
const UNIMP_TST: u8 = 1 << 1;
const UNIMP_FRE: u8 = 1 << 2;
const UNIMP_NIC: u8 = 1 << 3;

fn checksum(buf: &[u8]) -> u8 {
    buf.iter().fold(0u8, |s, b| s.wrapping_add(*b))
}

/// One parsed request. The firmware puts a carriage return where the checksum
/// was, so `text` always ends in one; `pos` never passes the end.
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

    /// The hex reader: up to four digits, either case. A shorter number consumes
    /// the character that ends it (one space before the next); none is no value.
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

/// What each build answers. A mnemonic a build lacks (`aut` is the Origin 200's; `dsc`, `dsp`,
/// `key`, `vlm`, `fre` the Origin 2000's) or a command on the wrong channel answers `err cmd`.
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

    clock: Clock,
    /// The deadline of the timer being handled; arming counts from it, so a long gap replays what
    /// a steady clock gives.
    at: Option<u64>,
    deadlines: [Option<u64>; TIMERS],

    /// The request, one byte per I2C send; one over the legal maximum so an
    /// over-long guest frame lands here, not in adjacent state.
    rx: Vec<u8>,
    /// The response being read back, built once the request is complete.
    tx: Vec<u8>,
    txpos: usize,
    /// The processor code whose request framed the pending response, -1 when none;
    /// its grant uses signal 2 until consumed.
    resp_owner: i32,

    // Grant ring.
    arb_code: u32,
    arb_signal: u8,
    /// When the next grant tick falls, or none before the rotation has started. The tick is armed
    /// only while an adapter polls for grants, and the ticks it missed are replayed on the next poll.
    arb_next: Option<u64>,
    arb_polled: bool,
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
}

impl State {
    fn new(cfg: ElscConfig, nvram: Arc<ElscNvram>, bus_busy: BusBusy, clock: Clock) -> Self {
        Self {
            cfg,
            nvram,
            bus_busy,
            clock,
            at: None,
            deadlines: [None; TIMERS],
            rx: Vec::new(),
            tx: Vec::new(),
            txpos: 0,
            resp_owner: -1,
            arb_code: 0,
            arb_signal: 0,
            arb_next: None,
            arb_polled: false,
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
        }
    }

    // ---- time -----------------------------------------------------------

    fn now(&self) -> u64 {
        self.at.unwrap_or_else(|| self.clock.now())
    }

    fn arm_ms(&mut self, timer: usize, ms: u64) {
        self.deadlines[timer] = Some(self.now() + ms * NS_PER_MS);
    }

    fn arm_secs(&mut self, timer: usize, secs: u64) {
        self.arm_ms(timer, secs * 1000);
    }

    fn pending(&self, timer: usize) -> bool {
        self.deadlines[timer].is_some()
    }

    fn cancel(&mut self, timer: usize) {
        self.deadlines[timer] = None;
    }

    fn fire(&mut self, timer: usize, deadline: u64) {
        match timer {
            T_ARB => self.arb_tick(deadline),
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

    fn err_arg(&mut self) {
        self.respond_str("err arg");
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

    /// A power-up may have just been cancelled: on an unpowered module say again
    /// that power is off, so the machine asks whether anything remains to power it up.
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
        // Its "ok" was read long ago, so the shutdown commits at once; it overrides
        // an unread-ack action but a committed one fires and the shutdown is lost.
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

    /// The watchdog timer ran out. Waiting: warn, five more seconds. Warned: NMI, start the delay.
    /// Cycling: power off and on, never checking module power, so it revives a module `pwr d`
    /// turned off.
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

    /// Queue a message for the Hubs, announced to every processor code until its
    /// `get` or a second passes; a later one replaces it. The Origin 200 also prints it.
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

    /// Whether something waits for the code's processor: a message, console input
    /// for the console processor, or an unconsumed response.
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

    /// One periodic tick. A tick that finds the bus busy publishes nothing and does
    /// not advance. The ring lapses unless an adapter polled during the tick.
    fn arb_tick(&mut self, deadline: u64) {
        if !self.arb_blocked() {
            self.arb_publish();
        }
        let next = deadline + ARB_TICK_NS;
        self.arb_next = Some(next);
        self.deadlines[T_ARB] = std::mem::take(&mut self.arb_polled).then_some(next);
    }

    /// An adapter is waiting for a grant: bring the ring up to the clock, as ticks would have, and
    /// keep it ticking while the polls go on.
    fn arb_poll(&mut self) {
        let Some(mut next) = self.arb_next.filter(|_| self.rings()) else { return };
        self.arb_polled = true;
        if self.pending(T_ARB) {
            return;
        }
        let now = self.now();
        if now >= next {
            let missed = (now - next) / ARB_TICK_NS + 1;
            if !self.arb_blocked() {
                self.arb_code = (self.arb_code + ((missed - 1) % u64::from(ARB_POSITIONS)) as u32) % ARB_POSITIONS;
                self.arb_publish();
            }
            next += missed * ARB_TICK_NS;
            self.arb_next = Some(next);
        }
        self.deadlines[T_ARB] = Some(next);
    }

    // ---- deferred destructive commands ---------------------------------

    /// Arm a destructive action whose `ok` was just framed. Clearing `deferred_committed` would
    /// lose an acknowledged action's commit; a committed action still fires, logged if this `ok`
    /// names another.
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

    /// The fetch of the last frame byte: the whole ack is delivered. End the announcement and
    /// commit any deferred action here, so a NAK'd byte re-started with no STOP keeps it.
    fn response_fully_read(&mut self) {
        self.resp_owner = -1;
        if self.deferred_action != Action::None {
            self.deferred_committed = true;
        }
    }

    /// Mark a committed action to fire at a transaction boundary (NACK, STOP or
    /// a following start), after the guest has taken the whole frame.
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
        // Release the bus for every destructive action before it fires: all three
        // leave this controller selected and none ends the transfer.
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
        // A new request supersedes a pending response; an armed action (ack unread)
        // is cancelled, a committed one survives.
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

    /// Console output names its sender after `acp `, turned into a number by
    /// subtracting `0` with no range check; recorded before the checksum is checked.
    fn record_console_sender(&mut self) {
        if self.rx.len() >= 7 && self.rx[0] >= 5 && &self.rx[1..5] == b"acp " {
            self.console_sender = self.rx[5] as i32 - b'0' as i32;
            self.console_update();
        }
    }

    /// A three-letter lowercase mnemonic and a separator (space or end); matching
    /// is case-sensitive, a bare terminator is no command, unread bytes are discarded.
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

    /// The switches, active-low, as hex with no leading zero. The Origin 200 reads
    /// its role from bit 0x04 (set on the master); topology decides, so it follows that.
    fn handle_rsw(&mut self, _r: &mut Req) {
        let mut raw = 0xff ^ self.cfg.dip_switches;
        if !self.is2000() {
            raw = if self.cfg.master { raw | 0x04 } else { raw & !0x04 };
        }
        self.respond_str(&format!("ok {raw:x}"));
    }

    /// The console-input poll: a pending message first, as bare text, then console
    /// input. The firmware checks four of six header bytes; the fifth names the selection mode.
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

    /// Console output. A bare `ok` parses as zero accepted and livelocks the firmware, so the reply
    /// is `ok <count>`; selector and payload sit at fixed offsets, unvalidated.
    fn handle_acp(&mut self, r: &mut Req) {
        let len = r.text.len() - 1;
        if len < 6 {
            return self.err_arg();
        }
        let n = (len - 6).min(86);
        // Output from a processor the "see" filter excludes is counted but not
        // copied to serial; while an operator types a command line, all output is lost.
        if !self.serial_command
            && (self.see_processor == SEE_ALL || self.see_processor as i32 == self.console_sender)
        {
            let text = r.text[6..6 + n].to_vec();
            self.serial_write(&text);
        }
        self.respond_str(&format!("ok {n:x}"));
    }

    /// `sel [<slot><a|b>|all|none|auto]`: console input selection. No argument
    /// reports it; `all` and `none` (first two letters) disable it, `auto` follows output.
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
            return self.err_arg();
        }
        self.respond_ok();
    }

    /// `see [<slot><a|b>|all|none]`: the console output filter; a processor name
    /// copies only its output to serial and selects it for input.
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
            return self.err_arg();
        }
        self.respond_ok();
    }

    /// `pwr d [<sec>]`: power down after a hex delay. 1 to 0x258 (re)loads the countdown; 0, or
    /// none with one pending, acts at once; else 3 s; above 0x258 is `err arg`. An immediate one
    /// replies `ok` and acts once the ack is read.
    fn handle_pwr_down(&mut self, r: &mut Req) {
        if !self.powered() {
            // Nothing to power down: a bare "ok", with no range check.
            self.respond_ok();
            return;
        }
        r.skip();
        let delay = r.read_hex();
        if delay.is_some_and(|s| s > DELAY_MAX) {
            return self.err_arg();
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
                // Power-up runs the module's power sequence, which reinitialises this controller's
                // bus interface, so it waits for the "ok" to be read; a powered module ignores it.
                self.respond_ok();
                if !self.powered() && self.can_execute(Action::PowerUp) {
                    self.arm_action(Action::PowerUp, "pwr u");
                }
            }
            b'q' => {
                // Power-on query: the PROM asks once to detect a cold start. "ok 1" on the
                // first read after cold power-on, "ok 0" after; clear-on-read is unconfirmed.
                let v = self.cold_power_on as u8;
                self.cold_power_on = false;
                self.respond_str(&format!("ok {v:x}"));
            }
            b'd' => self.handle_pwr_down(r),
            b'c' => {
                // The optional off interval reads like the `pwr d` delay, 5 to 0x258 seconds
                // (5 default); it replies "ok" and acts once the ack is read.
                r.skip();
                let interval = r.read_hex();
                if interval.is_some_and(|s| !(5..=DELAY_MAX).contains(&s)) {
                    return self.err_arg();
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

    /// `hbt [<interval> [<delay>]]`: the watchdog. 0 stops it; 5 to 0x258 s arms it, delay to the
    /// power cycle the second value or 30 s; else `err arg`. No value is a heartbeat that only
    /// reloads the timer.
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
            return self.err_arg();
        }
        self.hbt_interval = interval;
        self.hbt_delay = r.read_hex().unwrap_or(HBT_DEFAULT_DELAY);
        self.hbt_stage = Heartbeat::Waiting;
        self.hbt_arm(interval as u64);
        self.respond_ok();
    }

    /// `tas [<ch>]`: the test-and-set cell, free when `0`. `tas 0` frees it, another byte (NUL too)
    /// takes it if free, none reports. Replies `ok ` and the prior byte, length-framed so a NUL
    /// owner round-trips.
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

    /// `nic <pulse> <sample> <delay>`: one-wire passthrough; a missing hex value is
    /// a bad argument. No device is modelled, the line idles high.
    fn handle_nic(&mut self, r: &mut Req) {
        let (Some(pulse), Some(sample), Some(_delay)) = (r.read_hex(), r.read_hex(), r.read_hex()) else {
            return self.err_arg();
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

    /// Module reset, for panic reboot and fatal-error recovery. A caller takes `ok`
    /// as proof and has no fallback, so it must fire once the guest has read `ok`.
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

    /// `aut [1|0]`: query, set or clear the automatic power-on bit in the NVRAM configuration byte.
    /// It takes effect at the next start, which the model does not distinguish, so it is only kept.
    fn handle_aut(&mut self, r: &mut Req) {
        let arg = r.read_char();
        if arg.is_some_and(|c| c != b'1' && c != b'0') {
            return self.err_arg();
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

    /// `clr`: cancel a pending power-down, the watchdog and fan-high, powered or not. Power-up
    /// countdown, `tas`, selection, `fsc`, `tst`, NVRAM and power stay; serial echo goes on,
    /// supervisor mode clears.
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

    /// `fan [h|n]`: the fan-high request. With no fans or temperatures the request
    /// alone decides; over I2C neither form writes the NVRAM, on serial it is kept there.
    fn handle_fan(&mut self, r: &mut Req) {
        let Some(c) = r.read_char() else {
            self.respond_str(if self.fan_high { "ok h" } else { "ok n" });
            return;
        };
        if c != b'h' && c != b'n' {
            return self.err_arg();
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

    /// `fsc [1|0]`: declare a multimodule controller present or absent. Query and other arguments
    /// answer plain `ok ` over I2C. It only changes an Origin 2000 build, so it is only stored.
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

    /// `tst <c>`: the factory's short timeouts, set by `1`, cleared by any other
    /// character. They shorten waits the model lacks, so only stored.
    fn handle_tst(&mut self, r: &mut Req) {
        let Some(c) = r.read_char() else {
            return self.err_arg();
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

    /// `tmp`: `ok n` while no temperature input is active (always, with no sensors);
    /// a master that heard its slave appends the slave's last letter.
    fn handle_tmp(&mut self, _r: &mut Req) {
        let mut resp = b"ok n".to_vec();
        if self.slave_link {
            resp.extend_from_slice(&[b' ', self.slave_temp]);
        }
        self.respond(&resp);
    }

    /// `sla <c>`: a slave's report to its master, believed unchecked: `n`, `h`, `o` temperature,
    /// 0x07 announcing itself. The shutdowns and fan reports on `o`, 0x04 to 0x06 need a
    /// controller-to-controller link the model lacks.
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

    /// `nmi`: `err cmd` unpowered; else `ok` and a second later a 100 ms NMI pulse,
    /// restarted by another `nmi` inside that second (the wait starts at the answer).
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

    /// `dsp <text>`: the Origin 2000's eight-character panel message, from after the mnemonic's
    /// space (spaces count), overwriting from position one, ending at the first unprintable; empty
    /// is a bad argument.
    fn handle_dsp(&mut self, r: &mut Req) {
        let n = r.text[r.pos.min(r.text.len())..]
            .iter()
            .take(DISPLAY_CHARS)
            .take_while(|c| (0x20..0x7f).contains(*c))
            .count();
        if n == 0 {
            return self.err_arg();
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

    /// `dsc <pos> <ch>`: one display character; the position is the digit 1 to 8
    /// (positions 0 to 7), the character printable and never a space.
    fn handle_dsc(&mut self, r: &mut Req) {
        let (Some(pos), Some(ch)) = (r.read_char(), r.read_char()) else {
            return self.err_arg();
        };
        if !(b'1'..=b'8').contains(&pos) || !(0x20..0x7f).contains(&ch) {
            return self.err_arg();
        }
        self.display[(pos - b'1') as usize] = ch;
        self.status = Status::None;
        self.respond_ok();
    }

    /// `key`: the key-switch position. The model's key is always on.
    fn handle_key(&mut self, _r: &mut Req) {
        self.respond_str("ok on");
    }

    /// `vlm <supply> [h|l|n]`: query or set a supply margin; supply `3`, `5` or `v`.
    /// Margins are stored and answered, nothing electrical follows.
    fn handle_vlm(&mut self, r: &mut Req) {
        const SUPPLY_NAMES: &[u8] = b"35v";
        let Some(supply) = r.read_char().and_then(|c| SUPPLY_NAMES.iter().position(|s| *s == c)) else {
            return self.err_arg();
        };
        let Some(m) = r.read_char() else {
            self.respond_str(&format!("ok {}", self.margins[supply] as char));
            return;
        };
        if !matches!(m, b'h' | b'l' | b'n') {
            return self.err_arg();
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
                return self.err_arg();
            }
        }
        self.respond_ok();
    }

    /// `pas`: none leaves supervisor mode; `pas <word>` enters it for the exact password (NVRAM
    /// offsets 1 to 4); `pas s <word>` there stores a word's first four characters, after exactly
    /// one space.
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
                return self.err_arg();
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
            return self.err_arg();
        }
        for (i, b) in word.iter().enumerate() {
            if self.nvram_byte(1 + i as u8, None) != Some(*b) {
                return self.err_arg();
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

    /// `nvm <addr> [<data>]`: one NVRAM byte; low byte the offset, high byte the device (000 to 7ff
    /// reach 0x50 to 0x57, higher alias back); a silent device gives a bare `err `.
    fn handle_nvm(&mut self, r: &mut Req) {
        let Some(addr) = r.read_hex() else {
            return self.err_arg();
        };
        let device = ((0xa0 | ((addr >> 8) << 1)) as u8) >> 1;
        let data = r.read_hex();
        match self.nvram.access(device, addr as u8, data.map(|d| d as u8)) {
            None => self.respond_str("err "),
            Some(_) if data.is_some() => self.respond_ok(),
            Some(v) => self.respond_str(&format!("ok {v:x}")),
        }
    }

    /// `cfg`: the NVRAM's raw bytes from device 0x50 offset 0 through the first zero or
    /// 0x700 bytes, then a line end; no Ctrl+T, no `ok`.
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

    /// Run one serial line, parsed as an I2C request with the ending byte in place
    /// of the carriage return (so only a CR separates a bare mnemonic), then the supervisor gate.
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

    /// Run the waiting serial line and print Ctrl+T, the reply (no separator space
    /// after a bare `ok`) and a line end; no prompt. A requested action runs after the reply.
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

    /// A completed line runs once the bus is free, our I2C target idle and no Hub message pending
    /// (an I2C command arriving together goes first). An Origin 2000 module's scheduler decides
    /// that turn instead.
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

    /// One operator byte. Pass-through: all but Ctrl+T is console input. Command mode: bytes build
    /// the line, CR/LF/NUL ends it, a second Ctrl+T abandons it (typed bytes kept, one Ctrl+T
    /// passed on).
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
    fn start(&mut self) {
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
            // The module starts off. A controller that powers it up counts the delay; an Origin 200
            // slave waits for its master; an Origin 2000 module owns its power, so only the
            // countdown is ours.
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
        if self.rings() && self.arb_next.is_none() {
            if !self.arb_blocked() {
                self.arb_publish();
            }
            self.arb_next = Some(self.now() + ARB_TICK_NS);
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
        let _ = writeln!(s, "elsc\n  module power: {power}");
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
            s.push_str(if self.rx_queue.len() > 64 { "\"...\n" } else { "\"\n" });
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
            s.push_str("  pending response: none\n");
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
            s.push_str("; output: all\n");
        } else {
            let _ = writeln!(s, "; output: processor {}", self.see_processor);
        }
        if self.rings() {
            let _ = writeln!(s, "  arbitration rotation: next position {} of {}", self.arb_code, ARB_POSITIONS);
        }
        let _ = writeln!(
            s,
            "  fan high: {}, fsc: {}, short timeouts: {}, slave report: {:#04x}\n  console signal (code 0): {:#04x} = signal{}",
            self.fan_high,
            self.fsc_present,
            self.short_timeouts,
            self.slave_report,
            SIGNAL_BASE + if code0_signal2 { 8 } else { 0 },
            if code0_signal2 { 2 } else { 1 }
        );
        s
    }
}

pub struct Elsc {
    state: Mutex<State>,
    backend: Mutex<Arc<dyn SerialBackend>>,
    /// Called, with no controller lock held, when the controller has outputs or a committed
    /// action for the machine.
    wake: OnceLock<Box<dyn Fn() + Send + Sync>>,
    timers: [Timer; TIMERS],
    running: AtomicBool,
}

impl Elsc {
    /// A controller on the bus whose busy flag is `busy`, keeping its
    /// configuration in `nvram`. Call [`Resettable::power_on`] to start it.
    pub fn new(cfg: ElscConfig, nvram: Arc<ElscNvram>, tm: &Arc<TimerManager>, busy: BusBusy) -> Arc<Self> {
        Arc::new_cyclic(|me| Self {
            state: Mutex::new(State::new(cfg, nvram, busy, tm.clock().clone())),
            backend: Mutex::new(null_serial_backend()),
            wake: OnceLock::new(),
            timers: std::array::from_fn(|kind| Timer::new(tm, me, move |e: &Self, due| e.expire(kind, due))),
            running: AtomicBool::new(false),
        })
    }

    /// Call `wake` on a timer thread once the controller has outputs or a committed action to hand
    /// the machine. Once, before the machine runs.
    pub fn set_wake(&self, wake: Box<dyn Fn() + Send + Sync>) {
        let _ = self.wake.set(wake);
    }

    /// Hand the machine what is waiting, as [`Self::set_wake`] asked.
    pub fn wake(&self) {
        if let Some(wake) = self.wake.get() {
            wake();
        }
    }

    /// Attach the module that owns this controller's power, reset and bus
    /// scheduling. Wiring: call it before [`Resettable::power_on`].
    pub fn set_module_ops(&self, module: Arc<dyn ModuleOps>) {
        self.state.lock().module = Some(module);
    }

    /// Clear the protocol queues without resetting the controller: a module owner's
    /// recovery, reset and power paths reinitialise the framing state and no more.
    pub fn reinitialize(&self) {
        self.run(|st| st.reinitialize());
    }

    /// The start-up power-up with none of its wait, called from an Origin 2000
    /// module-management reset so the module is powered before any processor runs.
    pub fn power_up_at_start(&self) {
        self.run(|st| st.power_up_complete(false));
    }

    /// Where the AUX connector's transmit bytes go.
    pub fn set_serial_backend(&self, backend: Arc<dyn SerialBackend>) {
        *self.backend.lock() = backend;
    }

    /// Run `f`, arm the timers it moved, then send what it wrote to the AUX port and hand the
    /// module its actions with the lock released.
    fn run<R>(&self, f: impl FnOnce(&mut State) -> R) -> R {
        let (value, out, actions, module) = {
            let mut st = self.state.lock();
            let value = f(&mut st);
            self.arm(&mut st);
            (value, std::mem::take(&mut st.out), std::mem::take(&mut st.outbox), st.module.clone())
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
        if acted {
            self.visit_soon();
        }
        value
    }

    /// Make each timer match its deadline, and ask for a visit if the machine is owed one.
    fn arm(&self, st: &mut State) {
        if self.wake.get().is_some() && (st.deferred_scheduled || !st.events.is_empty()) && !st.pending(T_VISIT) {
            st.deadlines[T_VISIT] = Some(st.now() + VISIT_NS);
        }
        for (timer, due) in self.timers.iter().zip(st.deadlines) {
            timer.set_due(due);
        }
    }

    /// What the module just did may have left it outputs for the machine.
    fn visit_soon(&self) {
        let mut st = self.state.lock();
        if self.wake.get().is_some() && !st.pending(T_VISIT) {
            st.deadlines[T_VISIT] = Some(st.now() + VISIT_NS);
        }
        self.arm(&mut st);
    }

    /// The timer of `kind` armed for `due` has fired. A visit carries out a committed action and
    /// hands the machine the outputs.
    fn expire(&self, kind: usize, due: u64) {
        let live = self.run(|st| {
            if st.deadlines[kind] != Some(due) {
                return false;
            }
            st.deadlines[kind] = None;
            st.at = Some(due);
            if kind != T_VISIT {
                st.fire(kind, due);
            }
            st.at = None;
            true
        });
        if live && kind == T_VISIT {
            self.poll();
            self.wake();
        }
    }

    /// Let time pass and carry out what is due. The bus does this around every
    /// access; a machine whose guest may not touch it for long also calls it from its tick.
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

    /// A module event reaching the controller only through a shared bus: it runs on standby power
    /// and keeps its state, reinitialising its bus interface and queues and dropping Hub messages.
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
        self.run(|st| st.start());
    }
}

impl Device for Elsc {
    fn step(&self, _: u64) {}
    /// The machine's clock holds while it is stopped, so the timers go and come back with it.
    fn start(&self) {
        self.running.store(true, Ordering::Release);
        self.timers.iter().for_each(Timer::start);
    }
    fn stop(&self) {
        self.running.store(false, Ordering::Release);
        self.timers.iter().for_each(Timer::stop);
    }
    fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }
    fn get_clock(&self) -> u64 {
        0
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
            // An unconsumed response stays pending until read to its end or superseded; a
            // committed action fires here for a response the master ends with a STOP.
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
            st.arb_poll();
            Service {
                grant: st.pending_grant.take(),
                release_bus: std::mem::take(&mut st.release_bus),
            }
        })
    }
}

#[cfg(test)]
pub(crate) mod tests;
