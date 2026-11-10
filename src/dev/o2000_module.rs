//! Module management for an Origin 2000 or Onyx2 module: up to four node
//! adapters in front of one module controller, one configuration NVRAM and
//! the UST control port, with the module's grant scheduler, its power, reset
//! and NMI outputs and its midplane one-wire net.
//!
//! Each node's Hub reaches the module's management bus through its own
//! [`Pcf8584`]. The controller and the NVRAM are targets on every adapter's
//! bus. The scheduler publishes the controller's grants (one per
//! millisecond, eight processor codes) so that every adapter sees them, and
//! takes the module's turn on the bus for the controller's own requests.
//!
//! What is not modelled: the shared clocked wire. Each adapter completes a
//! byte inside the register access that starts it, so two nodes cannot
//! contend bit by bit, lose arbitration or stretch each other's clock. One
//! adapter holds the bus from its START to its STOP and another's START waits
//! for it (a model choice; the specification leaves a START on a busy bus
//! undefined). The scheduler treats the bus as busy while any adapter holds
//! it, which keeps grants and the controller's own turns from landing in the
//! middle of a transfer. A STOP written mid-byte, a node's STOP after an
//! acknowledged read and a peer's START overtaking a grant have no
//! counterpart here.
//!
//! A grant is seen by every adapter that has ESO set and no transfer of its
//! own open, whoever holds the bus and whether or not a START waits, and an
//! adapter is offered each grant once. The controller puts the grant on an
//! idle wire and every adapter's slave half observes it passively at the
//! STOP; nothing shows what the PROM or the kernel sees if an adapter
//! misses one, so the documented model stands.
//!
//! Model choices, not measured behaviour: the one-millisecond service tick;
//! the 5000-tick (five second) recovery count after which a bus that stays
//! busy has the controller reinitialised; the one-second life of a pending
//! controller message; the 100 ms reset and NMI pulses; that a destructive
//! command is refused for the length of a pulse; that automatic console
//! selection starts at code 0 and treats a CPU index as a processor code;
//! that the UST control port at I2C `0x20` is a one-byte port (as a PCF8574
//! expander is) reading all ones from power-on and then the last byte
//! written, with a general call accepted.
//!
//! Migration and snapshot state are not saved yet.
//!
//! Ported from hw/misc/sgi-sn0-module-management.c in
//! jamesbraid/qemu-sgi-origami.

use crate::dev::elsc::{Action, Build, Elsc, ElscConfig, ModuleOps, ModuleOutput};
use crate::dev::heart::Clock;
use crate::dev::elsc_nvram::ElscNvram;
use crate::dev::i2c::{BusBusy, I2cBus, I2cTarget, Service};
use crate::dev::pcf8584::{BusOwner, Pcf8584};
use crate::dev::w1::{self, crc::crc16, sgi::BoardRecord};
use crate::devlog::LogModule;
use crate::dlog;
use crate::traits::Resettable;
use parking_lot::Mutex;
use std::time::Duration;
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

#[derive(Clone, Copy, Debug)]
pub struct ModuleConfig {
    /// The module number the NVRAM's module byte starts with.
    pub module_number: u8,
    /// Which destructive actions this installation can carry out. A command
    /// whose action is not declared is refused (`err cmd`), not answered,
    /// since a caller takes `ok` as proof it happened.
    pub can_reset: bool,
    pub can_nmi: bool,
    pub can_power: bool,
    /// The front-panel DIP switches, as `rsw` reports them inverted.
    pub dip_switches: u8,
    /// Milliseconds after the cabinet starts that the controller powers the
    /// module up; zero powers it up inside the reset, before any processor
    /// runs.
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
    /// The service tick's next instant, and the other deadlines.
    service_deadline: u64,
    message_deadline: Option<u64>,
    pulse_deadline: Option<u64>,
    now: u64,
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
    fn new(cfg: ModuleConfig) -> Self {
        let mut s = Self {
            cfg,
            service_deadline: SERVICE_PERIOD_NS,
            message_deadline: None,
            pulse_deadline: None,
            now: 0,
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

    /// The controller's own reinitialisation: the scheduler's protocol state
    /// goes back to reset and the controller's queues follow. The publication
    /// enable latch is left alone. It reaches no further: a module reset
    /// reaches the nodes through the reset line the machine wires.
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

    /// Bring every deadline up to `now`, in order: a message expiry and a
    /// pulse before the service tick at the same instant. Ticks that cannot
    /// change anything are skipped in one step, up to the next other
    /// deadline.
    fn advance(&mut self, now: u64, busy: bool) {
        if now < self.now {
            return;
        }
        let next = [Some(self.service_deadline), self.message_deadline, self.pulse_deadline].into_iter().flatten().min();
        if next.is_none_or(|d| d > now) {
            // Nothing is due: the common case of a query between ticks.
            self.now = now;
            return;
        }
        loop {
            let other = [self.message_deadline, self.pulse_deadline].into_iter().flatten().min();
            let due_other = other.filter(|d| *d <= now);
            if let Some(d) = due_other.filter(|d| *d <= self.service_deadline) {
                if self.message_deadline == Some(d) {
                    self.pending_messages = 0;
                    self.message_deadline = None;
                } else {
                    self.pulse_complete();
                }
                continue;
            }
            if self.service_deadline > now {
                break;
            }
            let horizon = other.map_or(now, |d| d.min(now));
            let remaining = (horizon.saturating_sub(self.service_deadline)) / SERVICE_PERIOD_NS;
            self.tick(busy);
            let mut skip = 0;
            if remaining > 0 {
                if self.serial_granted || !self.publication_enabled {
                    skip = remaining;
                } else if self.blocked(busy) {
                    skip = remaining.min(self.recovery_ticks.saturating_sub(1) as u64);
                    self.recovery_ticks -= skip as u32;
                } else if self.pending_messages == 0 && !self.controller_request && !self.serial_request {
                    // Publishing: a whole revolution leaves the code where
                    // it was.
                    skip = remaining / PROCESSOR_CODES as u64 * PROCESSOR_CODES as u64;
                }
            }
            self.service_deadline += (1 + skip) * SERVICE_PERIOD_NS;
        }
        self.now = now;
    }

    fn set_message(&mut self, processor: usize, pending: bool) {
        if processor >= PROCESSOR_CODES {
            dlog!(LogModule::Module, "o2000 module: message for processor {processor}, outside the eight published codes");
            return;
        }
        if pending {
            self.pending_messages |= 1 << processor;
            self.message_deadline = Some(self.now + MESSAGE_LIFETIME_NS);
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
        self.pulse_deadline = Some(self.now + PULSE_NS);
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
    clock: Clock,
    /// Which adapter has the shared bus open.
    owner: Arc<BusOwner>,
    elsc: Mutex<Weak<Elsc>>,
    onewire: Arc<Mutex<w1::Bus>>,
}

impl Shared {
    fn any_busy(&self) -> bool {
        self.owner.busy()
    }

    /// Bring the scheduler to the present. Anything it owes the controller
    /// is left for [`Self::run`], which holds no controller lock.
    fn with<R>(&self, f: impl FnOnce(&mut State) -> R) -> R {
        let mut st = self.state.lock();
        let busy = self.any_busy();
        st.advance((self.clock)(), busy);
        f(&mut st)
    }

    /// As [`Self::with`], then reinitialise the controller if the scheduler
    /// asked: it takes its own lock, so this is never called from inside the
    /// controller.
    fn run<R>(&self, f: impl FnOnce(&mut State) -> R) -> R {
        let (value, reinit) = {
            let mut st = self.state.lock();
            let busy = self.any_busy();
            st.advance((self.clock)(), busy);
            let value = f(&mut st);
            (value, std::mem::take(&mut st.reinit))
        };
        if reinit {
            if let Some(elsc) = self.elsc.lock().upgrade() {
                elsc.reinitialize();
            }
        }
        value
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
            // The controller carries a cycle out as a power-off and, after
            // its own off interval, a power-up: a cycle that took no time
            // here would put power back before the interval began.
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

/// A module's two one-wire identity pages from its ten-character serial:
/// page 0 is the format byte `0x01` and the serial, page 1 the fixed name
/// `MODULEID` at offset sixteen, every other byte `0xff`, both closed by a
/// stored inverted CRC-16. `None` for a serial that does not fit the record.
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
    /// Build a module: the controller (Origin 2000 build), `nvram` (see
    /// [`ElscNvram::origin2000`]), the UST control port and four adapters.
    /// The machine wires the adapters into the Hubs and drains
    /// [`Self::take_events`] into the module's reset, NMI and power lines.
    pub fn new(cfg: ModuleConfig, nvram: Arc<ElscNvram>, clock: Clock) -> Arc<Self> {
        let owner = BusOwner::new();
        let shared = Arc::new(Shared {
            state: Mutex::new(State::new(cfg)),
            clock: clock.clone(),
            owner: owner.clone(),
            elsc: Mutex::new(Weak::new()),
            onewire: Arc::new(Mutex::new(w1::Bus::new())),
        });
        let elsc_cfg = ElscConfig {
            build: Build::Origin2000,
            dip_switches: cfg.dip_switches,
            master: true,
            start_delay_ms: cfg.start_delay_ms,
        };
        let elsc = Arc::new(Elsc::new(elsc_cfg, nvram.clone(), clock, BusBusy::default()));
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

    /// The reset, NMI and power lines' changes, in the order they happened,
    /// with time brought up to the present first.
    pub fn take_events(&self) -> Vec<ModuleOutput> {
        self.elsc.poll();
        self.shared.run(|s| std::mem::take(&mut s.events))
    }

    /// Call `wake` after any access that may have moved [`Self::next_due`];
    /// see [`Elsc::set_wake`]. The module's own pulse starts only from an
    /// action its controller hands it, so the controller's report covers it.
    pub fn set_wake(&self, wake: Box<dyn Fn() + Send + Sync>) {
        self.elsc.set_wake(wake);
    }

    /// How long until the module or its controller has an output to give, or
    /// a committed action to run; zero if it is due now, `None` if nothing is
    /// pending.
    pub fn next_due(&self) -> Option<Duration> {
        let (pulse_ends, queued) = {
            let st = self.shared.state.lock();
            (st.pulse_deadline, !st.events.is_empty())
        };
        if queued {
            return Some(Duration::ZERO);
        }
        let pulse = pulse_ends.map(|d| Duration::from_nanos(d.saturating_sub((self.shared.clock)())));
        [self.elsc.next_due(), pulse].into_iter().flatten().min()
    }

    /// Let time pass for the module and its controller when no guest is
    /// touching the bus.
    pub fn poll(&self) {
        self.elsc.poll();
        self.shared.run(|_| ());
    }

    pub fn powered(&self) -> bool {
        self.shared.run(|s| s.power_on)
    }

    /// Fit the module's MODULEID part (ROM serial `identity_serial`, pages
    /// from `module_serial`) and, if `midplane` is given, the midplane part
    /// holding that board record, on the net the controller's `nic` line
    /// drives.
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

impl Resettable for Origin2000Module {
    /// The cabinet's own start, the only reset that reaches the module
    /// management: a module's reset or power cycle leaves it alone. The
    /// module starts off. With no start delay the controller powers it up
    /// here, in the reset, so power is on before any processor runs; with a
    /// delay it counts that delay and runs the whole power-up sequence. A
    /// module with no power path starts powered, as it has no controller to
    /// power it.
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
            s.service_deadline = s.now + SERVICE_PERIOD_NS;
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
