//! Module management for an Origin 2000 or Onyx2 module: up to four node adapters in front of one
//! module controller, one configuration NVRAM and the UST control port, with the grant scheduler,
//! power, reset and NMI outputs and one-wire net.
//!
//! Each node's Hub reaches the management bus through its own [`Pcf8584`]; the controller and NVRAM
//! are targets on every bus. The scheduler publishes grants (one per millisecond, eight codes) to
//! every adapter and takes the module's bus turn.
//!
//! Not modelled: the shared clocked wire. Each adapter completes a byte inside the access that
//! starts it, so nodes cannot contend bit by bit, lose arbitration or stretch clocks. One adapter
//! holds the bus START to STOP and another's START waits (a model choice; a START on a busy bus is
//! unspecified here). A mid-byte STOP, a STOP after an acknowledged read and a peer's START
//! overtaking a grant have no counterpart.
//!
//! Every adapter with ESO set and no transfer open sees each grant once, whoever holds the bus;
//! what the PROM or kernel sees if one misses it is unknown, so the documented model stands.
//!
//! Model choices, not measured: the one-millisecond service tick; the 5000-tick recovery count
//! after which a busy bus has the controller reinitialised; the one-second life of a pending
//! message; the 100 ms reset and NMI pulses; that a destructive command is refused for a pulse;
//! that automatic console selection starts at code 0 and treats a CPU index as a processor code;
//! that the UST control port at I2C `0x20` is a one-byte port (as a PCF8574 is) reading all ones
//! from power-on, then the last byte written, with a general call accepted.
//!
//! Migration and snapshot state are not saved yet.
//!
//! Ported from hw/misc/sgi-sn0-module-management.c in jamesbraid/qemu-sgi-origami.

use crate::dev::elsc::{Action, Build, Elsc, ElscConfig, ModuleOps, ModuleOutput};
use crate::clock::Clock;
use crate::dev::elsc_nvram::ElscNvram;
use crate::dev::i2c::{BusBusy, I2cBus, I2cTarget, Service};
use crate::dev::pcf8584::{BusOwner, Pcf8584};
use crate::dev::w1::{self, crc::crc16, sgi::BoardRecord};
use crate::devlog::LogModule;
use crate::dlog;
use crate::hptimer::{Timer, TimerManager};
use crate::traits::{Device, Resettable};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};

/// One adapter per node board on a four-node module.
pub const ADAPTERS: usize = 4;
pub const MODULE_SERIAL_LEN: usize = 10;
/// The module's UST control port, the address the stock PROM uses.
pub const UST_CONTROL_ADDR: u8 = 0x20;

const PROCESSOR_CODES: usize = 8;
const GRANT_BASE: u8 = 0xe0;
const GRANT_MESSAGE_BIT: u8 = 0x08;
const SERVICE_PERIOD_NS: u64 = 1_000_000;
const MESSAGE_LIFETIME_NS: u64 = 1_000_000_000;
const PULSE_NS: u64 = 100_000_000;
const RECOVERY_TICKS: u32 = 5000;

// Timer kinds. Timers due at the same moment run in no promised order.
const K_MESSAGE: usize = 0;
const K_PULSE: usize = 1;
const K_TICK: usize = 2;

#[derive(Clone, Copy, Debug)]
pub struct ModuleConfig {
    /// The module number the NVRAM's module byte starts with.
    pub module_number: u8,
    /// Which destructive actions this installation can carry out. An undeclared one is refused
    /// (`err cmd`), since a caller takes `ok` as proof it happened.
    pub can_reset: bool,
    pub can_nmi: bool,
    pub can_power: bool,
    /// The front-panel DIP switches, as `rsw` reports them inverted.
    pub dip_switches: u8,
    /// Milliseconds after the cabinet starts that the controller powers the module up; zero powers
    /// it up inside the reset, before any processor runs.
    pub start_delay_ms: u32,
}

impl Default for ModuleConfig {
    fn default() -> Self {
        Self {
            module_number: 1,
            can_reset: false,
            can_nmi: false,
            can_power: false,
            dip_switches: 0,
            start_delay_ms: 0,
        }
    }
}

/// The module's scheduler state.
struct State {
    cfg: ModuleConfig,
    clock: Clock,
    /// The deadline of the timer being handled; arming counts from it.
    at: Option<u64>,
    /// The service tick's next instant, and the other deadlines. The tick is armed only while an
    /// adapter polls or the protocol has work for it; the ticks it missed are replayed on arming.
    service_deadline: u64,
    service_armed: bool,
    service_polled: bool,
    message_deadline: Option<u64>,
    pulse_deadline: Option<u64>,
    recovery_ticks: u32,
    processor_code: usize,
    pending_messages: u8,
    console_processor: usize,
    console_input_pending: bool,
    controller_request: bool,
    controller_granted: bool,
    serial_request: bool,
    serial_granted: bool,
    slave_active: bool,
    publication_enabled: bool,
    power_on: bool,
    pulse_active: bool,
    pulse_nmi: bool,
    /// The grant last published and a count of publications; each adapter
    /// keeps the count it last took.
    grant: u8,
    grant_seq: u64,
    taken: [u64; ADAPTERS],
    /// The controller's queues need reinitialising, once the scheduler's lock
    /// is released: the controller takes its own lock for that.
    reinit: bool,
    events: Vec<ModuleOutput>,
    /// The level each output line was last driven to; a line is only
    /// reported when it changes.
    reset_line: Option<bool>,
    nmi_line: Option<bool>,
    power_line: Option<bool>,
}

impl State {
    fn new(cfg: ModuleConfig, clock: Clock) -> Self {
        let mut s = Self {
            cfg,
            service_deadline: clock.now() + SERVICE_PERIOD_NS,
            service_armed: false,
            service_polled: false,
            clock,
            at: None,
            message_deadline: None,
            pulse_deadline: None,
            recovery_ticks: RECOVERY_TICKS,
            processor_code: PROCESSOR_CODES - 1,
            pending_messages: 0,
            console_processor: 0,
            console_input_pending: false,
            controller_request: false,
            controller_granted: false,
            serial_request: false,
            serial_granted: false,
            slave_active: false,
            publication_enabled: true,
            power_on: true,
            pulse_active: false,
            pulse_nmi: false,
            grant: 0,
            grant_seq: 0,
            taken: [0; ADAPTERS],
            reinit: false,
            events: Vec::new(),
            reset_line: None,
            nmi_line: None,
            power_line: None,
        };
        s.reset_state();
        s
    }

    fn now(&self) -> u64 {
        self.at.unwrap_or_else(|| self.clock.now())
    }

    fn reset_state(&mut self) {
        self.processor_code = PROCESSOR_CODES - 1;
        self.pending_messages = 0;
        self.controller_request = false;
        self.controller_granted = false;
        self.serial_request = false;
        self.serial_granted = false;
        self.slave_active = false;
        self.console_input_pending = false;
        self.console_processor = 0;
        self.recovery_ticks = RECOVERY_TICKS;
        self.message_deadline = None;
    }

    /// The controller's own reinitialisation: the scheduler's protocol state and the controller's
    /// queues go back to reset, the publication enable latch stays. A module reset reaches the
    /// nodes through the machine's reset line.
    fn reinitialize(&mut self) {
        self.reset_state();
        self.reinit = true;
    }

    fn recovery_tick(&mut self) {
        if self.recovery_ticks > 0 {
            self.recovery_ticks -= 1;
            if self.recovery_ticks == 0 {
                self.reinitialize();
            }
        }
    }

    fn drive_power(&mut self) {
        if self.cfg.can_power && self.power_line != Some(self.power_on) {
            self.power_line = Some(self.power_on);
            self.events.push(ModuleOutput::Power(self.power_on));
        }
    }

    fn drive_pulse(&mut self, reset: bool, nmi: bool) {
        if self.cfg.can_reset && self.reset_line != Some(reset) {
            self.reset_line = Some(reset);
            // The line is high for the 100 ms of a pulse. A restart is asked
            // for at the rising edge, and nothing waits for the fall.
            if reset {
                self.events.push(ModuleOutput::Reset);
            }
        }
        if self.cfg.can_nmi && self.nmi_line != Some(nmi) {
            self.nmi_line = Some(nmi);
            self.events.push(ModuleOutput::Nmi(nmi));
        }
    }

    fn message_for(&self, code: usize) -> bool {
        self.pending_messages & (1 << code) != 0 || (self.console_input_pending && self.console_processor == code)
    }

    /// Whether a tick finds the bus held: a controller turn, the controller
    /// addressed as a target, or any adapter with a transfer open.
    fn blocked(&self, busy: bool) -> bool {
        self.controller_granted || self.slave_active || busy
    }

    fn publish(&mut self) {
        self.processor_code = (self.processor_code + 1) % PROCESSOR_CODES;
        self.grant = GRANT_BASE
            | self.processor_code as u8
            | if self.message_for(self.processor_code) { GRANT_MESSAGE_BIT } else { 0 };
        self.grant_seq += 1;
        // A completed attempt reloads the recovery counter.
        self.recovery_ticks = RECOVERY_TICKS;
    }

    /// Whether the protocol has work for the next tick whether or not an adapter polls: a handoff
    /// to make, the bus held, or the recovery counter running down.
    fn wants_tick(&self) -> bool {
        self.controller_request
            || self.serial_request
            || self.controller_granted
            || self.serial_granted
            || self.slave_active
            || self.pending_messages != 0
            || self.recovery_ticks != RECOVERY_TICKS
    }

    /// Run the ticks that fell due while the tick was not armed. Whole revolutions of the
    /// processor codes past the recovery count change nothing more, so they are skipped.
    fn catch_up(&mut self, busy: bool) {
        let now = self.now();
        if now < self.service_deadline {
            return;
        }
        let missed = (now - self.service_deadline) / SERVICE_PERIOD_NS + 1;
        let cap = u64::from(RECOVERY_TICKS) + PROCESSOR_CODES as u64;
        let skipped = missed.saturating_sub(cap) / PROCESSOR_CODES as u64 * PROCESSOR_CODES as u64;
        for _ in 0..missed - skipped {
            self.tick(busy);
        }
        self.service_deadline += missed * SERVICE_PERIOD_NS;
    }

    /// One service tick.
    fn tick(&mut self, busy: bool) {
        // A serial command owning the protocol path leaves the recovery
        // counter alone.
        if self.serial_granted {
            return;
        }
        if !self.publication_enabled {
            // An unpowered module publishes no grants, but the controller
            // keeps running and takes serial commands once its bus is free.
            if self.serial_request && self.pending_messages == 0 && !self.slave_active && !busy {
                self.serial_request = false;
                self.serial_granted = true;
            }
            return;
        }
        if self.blocked(busy) {
            self.recovery_tick();
            return;
        }
        // A pending controller interrupt suppresses both handoffs, and a
        // handoff that happens skips publication without advancing the code.
        if self.pending_messages == 0 && self.controller_request {
            self.controller_request = false;
            self.controller_granted = true;
            return;
        }
        if self.pending_messages == 0 && self.serial_request {
            self.serial_request = false;
            self.serial_granted = true;
            return;
        }
        self.publish();
    }

    fn pulse_complete(&mut self) {
        self.drive_pulse(false, false);
        self.pulse_active = false;
        self.pulse_nmi = false;
        self.pulse_deadline = None;
        self.reinitialize();
        if self.power_on {
            self.publication_enabled = true;
        }
    }

    /// The timer of `kind`, armed for `due`, has fired.
    fn fire(&mut self, kind: usize, due: u64, busy: bool) {
        match kind {
            K_MESSAGE => {
                self.pending_messages = 0;
                self.message_deadline = None;
            }
            K_PULSE => self.pulse_complete(),
            _ => {
                self.tick(busy);
                self.service_deadline = due + SERVICE_PERIOD_NS;
                self.service_armed = std::mem::take(&mut self.service_polled) || self.wants_tick();
            }
        }
    }

    fn set_message(&mut self, processor: usize, pending: bool) {
        if processor >= PROCESSOR_CODES {
            dlog!(LogModule::Module, "o2000 module: message for processor {processor}, outside the eight published codes");
            return;
        }
        if pending {
            self.pending_messages |= 1 << processor;
            self.message_deadline = Some(self.now() + MESSAGE_LIFETIME_NS);
        } else {
            self.pending_messages &= !(1 << processor);
            if self.pending_messages == 0 {
                self.message_deadline = None;
            }
        }
    }

    fn start_pulse(&mut self, nmi: bool) -> bool {
        if !(if nmi { self.cfg.can_nmi } else { self.cfg.can_reset }) {
            dlog!(LogModule::Module, "o2000 module: {} requested on a module that declares no such path", if nmi { "NMI" } else { "reset" });
            return false;
        }
        if self.pulse_active {
            dlog!(LogModule::Module, "o2000 module: pulse requested during a {} pulse", if self.pulse_nmi { "NMI" } else { "reset" });
            return false;
        }
        self.pulse_active = true;
        self.pulse_nmi = nmi;
        self.drive_pulse(!nmi, nmi);
        self.pulse_deadline = Some(self.now() + PULSE_NS);
        true
    }

    fn set_power(&mut self, on: bool) -> bool {
        if !self.cfg.can_power {
            dlog!(LogModule::Module, "o2000 module: power {} requested on a module that declares no power path", if on { "up" } else { "down" });
            return false;
        }
        if !on {
            // Publication stops before module reset and power sequencing.
            self.publication_enabled = false;
            self.power_on = false;
            self.reinitialize();
            self.drive_power();
            return true;
        }
        self.power_on = true;
        self.drive_power();
        // The interface is initialised once reset is released, then
        // publication is enabled.
        self.reinitialize();
        self.publication_enabled = true;
        true
    }

    fn can_execute(&self, action: Action) -> bool {
        let declared = match action {
            Action::Reset => self.cfg.can_reset,
            Action::Cycle | Action::Shutdown | Action::PowerUp => self.cfg.can_power,
            Action::None => return false,
        };
        if !declared {
            dlog!(LogModule::Module, "o2000 module: destructive command refused, this module declares no {} path", if action == Action::Reset { "reset" } else { "power" });
            return false;
        }
        if self.pulse_active {
            // No source settles that overlap, so every destructive action is
            // refused for the duration of a pulse. A model choice.
            dlog!(LogModule::Module, "o2000 module: destructive command during a {} pulse", if self.pulse_nmi { "NMI" } else { "reset" });
            return false;
        }
        true
    }
}

struct Shared {
    state: Mutex<State>,
    /// Which adapter has the shared bus open.
    owner: Arc<BusOwner>,
    elsc: Mutex<Weak<Elsc>>,
    onewire: Arc<Mutex<w1::Bus>>,
    timers: [Timer; 3],
    running: AtomicBool,
}

impl Shared {
    fn any_busy(&self) -> bool {
        self.owner.busy()
    }

    /// The state, with the ticks that lapsed while the tick was unarmed replayed first, so `f` sees
    /// them as having happened before it.
    fn locked(&self) -> parking_lot::MutexGuard<'_, State> {
        let mut st = self.state.lock();
        if !st.service_armed {
            st.catch_up(self.any_busy());
        }
        st
    }

    /// Run `f`, then arm the timers it moved.
    fn with<R>(&self, f: impl FnOnce(&mut State) -> R) -> R {
        let mut st = self.locked();
        let value = f(&mut st);
        self.arm(&mut st);
        value
    }

    /// As [`Self::with`], then reinitialise the controller if the scheduler asked; it takes its own
    /// lock, so never call this from inside the controller.
    fn run<R>(&self, f: impl FnOnce(&mut State) -> R) -> R {
        let (value, reinit) = {
            let mut st = self.locked();
            let value = f(&mut st);
            self.arm(&mut st);
            (value, std::mem::take(&mut st.reinit))
        };
        if reinit {
            if let Some(elsc) = self.elsc.lock().upgrade() {
                elsc.reinitialize();
            }
        }
        value
    }

    /// Make each timer match its deadline.
    fn arm(&self, st: &mut State) {
        if st.wants_tick() {
            st.service_armed = true;
        }
        let tick = st.service_armed.then_some(st.service_deadline);
        for (timer, due) in self.timers.iter().zip([st.message_deadline, st.pulse_deadline, tick]) {
            timer.set_due(due);
        }
    }

    /// The timer of `kind` armed for `due` has fired; the machine is told if it left outputs.
    fn expire(&self, kind: usize, due: u64) {
        let queued = self.run(|s| {
            let live = match kind {
                K_MESSAGE => s.message_deadline,
                K_PULSE => s.pulse_deadline,
                _ => s.service_armed.then_some(s.service_deadline),
            };
            if live == Some(due) {
                s.at = Some(due);
                s.fire(kind, due, self.any_busy());
                s.at = None;
            }
            !s.events.is_empty()
        });
        if queued {
            if let Some(elsc) = self.elsc.lock().upgrade() {
                elsc.wake();
            }
        }
    }
}

/// What the controller calls: answers from the scheduler's own state, never
/// calling back into the controller.
struct Link(Arc<Shared>);

impl ModuleOps for Link {
    fn powered(&self) -> bool {
        self.0.with(|s| s.power_on)
    }

    fn can_execute(&self, action: Action) -> bool {
        self.0.with(|s| s.can_execute(action))
    }

    fn execute(&self, action: Action) {
        // Delivered with the controller's lock released, so the controller
        // may be reinitialised on the spot.
        let done = self.0.run(|s| match action {
            Action::Reset => s.start_pulse(false),
            Action::Shutdown => s.set_power(false),
            Action::PowerUp => s.set_power(true),
            // The controller carries a cycle out as a power-off and, after its own off interval, a
            // power-up; a cycle taking no time here would restore power before the interval began.
            Action::Cycle | Action::None => unreachable!("the controller never hands over {action:?}"),
        });
        if !done {
            dlog!(LogModule::Module, "o2000 module: an acknowledged action did not happen");
        }
    }

    fn pulsing(&self) -> bool {
        self.0.with(|s| s.pulse_active)
    }

    fn nmi(&self) -> bool {
        self.0.with(|s| s.start_pulse(true))
    }

    fn slave_active(&self, active: bool) {
        self.0.with(|s| s.slave_active = active);
    }

    fn console_owner(&self, processor: usize) {
        self.0.with(|s| {
            if processor >= PROCESSOR_CODES {
                dlog!(LogModule::Module, "o2000 module: console output from processor {processor}, outside the eight published codes");
            } else {
                s.console_processor = processor;
            }
        });
    }

    fn console_input(&self, pending: bool) {
        self.0.with(|s| s.console_input_pending = pending);
    }

    fn serial_turn(&self) -> bool {
        self.0.with(|s| {
            if s.serial_granted {
                return true;
            }
            s.serial_request = true;
            false
        })
    }

    fn serial_done(&self) {
        self.0.with(|s| s.serial_granted = false);
    }

    fn onewire_slot(&self, pulse: u32, sample: u32) -> bool {
        w1::slot::slot(&mut self.0.onewire.lock(), pulse, sample)
    }

    fn post_message(&self) {
        self.0.with(|s| {
            for code in 0..PROCESSOR_CODES {
                s.set_message(code, true);
            }
        });
    }

    fn take_message(&self, processor: usize) -> bool {
        self.0.with(|s| {
            if processor >= PROCESSOR_CODES || s.pending_messages & (1 << processor) == 0 {
                return false;
            }
            s.set_message(processor, false);
            true
        })
    }
}

/// One adapter's view of the scheduler: a target on that adapter's bus that
/// claims no address and hands the adapter each grant as it is published.
struct Port {
    shared: Arc<Shared>,
    index: usize,
}

impl I2cTarget for Port {
    fn claims(&self, _addr7: u8) -> bool {
        false
    }

    fn answers_general_call(&self) -> bool {
        false
    }

    fn send(&self, _byte: u8) -> bool {
        false
    }

    fn recv(&self) -> u8 {
        0xff
    }

    fn service(&self) -> Service {
        let grant = self.shared.run(|s| {
            s.service_polled = true;
            s.service_armed = true;
            if s.taken[self.index] == s.grant_seq {
                return None;
            }
            s.taken[self.index] = s.grant_seq;
            Some(s.grant)
        });
        Service { grant, release_bus: false }
    }
}

/// The UST control port: one byte, all ones from power-on, then the last byte
/// written.
struct UstControl(Mutex<u8>);

impl I2cTarget for UstControl {
    fn claims(&self, addr7: u8) -> bool {
        addr7 == UST_CONTROL_ADDR
    }

    fn send(&self, byte: u8) -> bool {
        *self.0.lock() = byte;
        true
    }

    fn recv(&self) -> u8 {
        *self.0.lock()
    }
}

/// A module's two one-wire identity pages from its ten-character serial (page 1 is `MODULEID` at
/// offset sixteen), each closed by an inverted CRC-16; `None` if the serial is too long.
pub fn identity_pages(serial: &str) -> Option<[[u8; 32]; 2]> {
    let bytes = serial.as_bytes();
    if bytes.len() != MODULE_SERIAL_LEN || !bytes.iter().all(|b| (0x20..0x7f).contains(b) && !matches!(b, b':' | b';')) {
        return None;
    }
    let mut pages = [[0xffu8; 32]; 2];
    pages[0][0] = 0x01;
    pages[0][1..1 + MODULE_SERIAL_LEN].copy_from_slice(bytes);
    pages[1][16..24].copy_from_slice(b"MODULEID");
    for page in &mut pages {
        let crc = !crc16(0, &page[..30]);
        page[30..].copy_from_slice(&crc.to_le_bytes());
    }
    Some(pages)
}

pub struct Origin2000Module {
    shared: Arc<Shared>,
    elsc: Arc<Elsc>,
    nvram: Arc<ElscNvram>,
    adapters: [Arc<Pcf8584>; ADAPTERS],
}

impl Origin2000Module {
    /// Build a module: the controller (Origin 2000 build), `nvram` ([`ElscNvram::origin2000`]), the
    /// UST control port and four adapters. The machine wires the adapters into the Hubs and drains
    /// [`Self::take_events`].
    pub fn new(cfg: ModuleConfig, nvram: Arc<ElscNvram>, tm: &Arc<TimerManager>) -> Arc<Self> {
        let owner = BusOwner::new();
        let shared = Arc::new_cyclic(|me| Shared {
            state: Mutex::new(State::new(cfg, tm.clock().clone())),
            owner: owner.clone(),
            elsc: Mutex::new(Weak::new()),
            onewire: Arc::new(Mutex::new(w1::Bus::new())),
            timers: std::array::from_fn(|kind| Timer::new(tm, me, move |s: &Shared, due| s.expire(kind, due))),
            running: AtomicBool::new(false),
        });
        let elsc_cfg = ElscConfig {
            build: Build::Origin2000,
            dip_switches: cfg.dip_switches,
            master: true,
            start_delay_ms: cfg.start_delay_ms,
        };
        let elsc = Elsc::new(elsc_cfg, nvram.clone(), tm, BusBusy::default());
        elsc.set_module_ops(Arc::new(Link(shared.clone())));
        *shared.elsc.lock() = Arc::downgrade(&elsc);
        let ust: Arc<dyn I2cTarget> = Arc::new(UstControl(Mutex::new(0xff)));
        let adapters = std::array::from_fn(|index| {
            let mut bus = I2cBus::new();
            bus.add(elsc.clone());
            bus.add(nvram.clone());
            bus.add(ust.clone());
            bus.add(Arc::new(Port { shared: shared.clone(), index }));
            Arc::new(Pcf8584::shared(bus, BusBusy::default(), owner.clone(), index as u32))
        });
        Arc::new(Self { shared, elsc, nvram, adapters })
    }

    /// Node `index`'s adapter, to map in its Hub.
    pub fn adapter(&self, index: usize) -> Arc<Pcf8584> {
        self.adapters[index].clone()
    }

    pub fn elsc(&self) -> &Arc<Elsc> {
        &self.elsc
    }

    pub fn nvram(&self) -> &Arc<ElscNvram> {
        &self.nvram
    }

    /// The reset, NMI and power lines' changes, in the order they happened.
    pub fn take_events(&self) -> Vec<ModuleOutput> {
        self.elsc.poll();
        self.shared.run(|s| std::mem::take(&mut s.events))
    }

    /// Call `wake` on a timer thread once the module or controller has outputs for the machine
    /// ([`Elsc::set_wake`]).
    pub fn set_wake(&self, wake: Box<dyn Fn() + Send + Sync>) {
        self.elsc.set_wake(wake);
    }

    pub fn powered(&self) -> bool {
        self.shared.run(|s| s.power_on)
    }

    /// Attach the module's MODULEID part (ROM serial `identity_serial`, pages from `module_serial`)
    /// and, with `midplane`, the midplane part holding that board record, on the net the
    /// controller's `nic` line drives.
    pub fn attach_parts(
        &self,
        module_serial: &str,
        identity_serial: u64,
        midplane: Option<(&BoardRecord, u64)>,
    ) -> Result<(), String> {
        let pages = identity_pages(module_serial)
            .ok_or_else(|| format!("module serial '{module_serial}' is not ten printable non-separator characters"))?;
        let image: Vec<u8> = pages.iter().flatten().copied().collect();
        let mut net = self.shared.onewire.lock();
        net.add(Box::new(w1::Ds2502::new(0x09, identity_serial, &image)));
        if let Some((record, serial)) = midplane {
            record.check()?;
            net.add(Box::new(w1::sgi::board_record_part(record, serial)));
        }
        Ok(())
    }

    /// The midplane net, to wire the second node's Hub line to it.
    pub fn midplane_onewire(&self) -> Arc<Mutex<w1::Bus>> {
        self.shared.onewire.clone()
    }
}

impl Device for Origin2000Module {
    fn step(&self, _: u64) {}
    /// The machine's clock holds while it is stopped, so the timers go and come back with it.
    fn start(&self) {
        self.shared.running.store(true, Ordering::Release);
        self.shared.timers.iter().for_each(Timer::start);
        Device::start(&*self.elsc);
    }
    fn stop(&self) {
        self.shared.running.store(false, Ordering::Release);
        self.shared.timers.iter().for_each(Timer::stop);
        Device::stop(&*self.elsc);
    }
    fn is_running(&self) -> bool {
        self.shared.running.load(Ordering::Acquire)
    }
}

impl Resettable for Origin2000Module {
    /// The cabinet's own start, the only reset that reaches module management. The module starts
    /// off; the controller powers it up in the reset (no delay) or after the delay; a module with
    /// no power path starts powered.
    fn power_on(&self) {
        for adapter in &self.adapters {
            adapter.power_on();
        }
        self.nvram.power_on();
        self.elsc.power_on();
        let direct = self.shared.run(|s| {
            s.pulse_active = false;
            s.pulse_nmi = false;
            s.pulse_deadline = None;
            s.drive_pulse(false, false);
            s.reset_state();
            s.power_on = !s.cfg.can_power;
            s.publication_enabled = s.power_on;
            s.service_deadline = s.now() + SERVICE_PERIOD_NS;
            let up_now = s.cfg.can_power && s.cfg.start_delay_ms == 0;
            if !up_now {
                s.drive_power();
            }
            up_now
        });
        if direct {
            self.elsc.power_up_at_start();
        }
    }
}

#[cfg(test)]
mod tests;
