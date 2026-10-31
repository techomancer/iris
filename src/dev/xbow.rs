//! Crossbow, the Octane's XIO crossbar: widget 0's registers, and which of
//! the eight links (widgets 8 to 15) are present and carrying traffic.
//!
//! Cut to what the Octane uses; registers follow the Crossbow ASIC
//! specification, revision 12.0 (`docs/Octane/`). The switch
//! translates no address: the machine routes a processor access by widget
//! number and asks [`Xbow::carries`] whether the link is up. Packets from
//! widget 0 itself are its interrupts, which go to the target its interrupt
//! destination registers name; on the Octane that is HEART.
//!
//! Which links are present and alive are atomics, so the routing decision
//! (`carries`) never takes a lock; they change only under the register
//! lock, with the registers that describe them. Interrupt packets and the
//! callbacks (link change, link reset) go out after that lock is dropped.
//!
//! Two approximations: a link reset on a fitted link brings the link back
//! alive as it was before the reset (link training is not modelled, and the
//! stock PROM resets each link it finds and expects it back), and register
//! accesses are 32-bit at the data half of each eight-byte slot only.
//!
//! Revision 5 and later add the reset fences and the resource lock, and the
//! chip can carry a part number and manufacturer, a source field in its
//! status reads and an interrupt that goes to a packet sink instead of
//! HEART: what [`Pxbow`](super::pxbow::Pxbow) needs behind a PIC.
//!
//! With a clock ([`Xbow::with_clock`]) a link reset is the two-core-cycle
//! pulse the QEMU model has: the link is down for 20 ns of the clock, and
//! the chip ends the pulse the next time anything reads its registers or
//! asks [`Xbow::carries`], so what the guest sees follows the clock and not
//! host scheduling. Without one the pulse is over at once. [`Xbow::warm_reset`]
//! takes a warm reset from a link through the revision-5 fences. No board in
//! this tree sends one yet: the Hub's link control is the only sender the
//! QEMU model names.
//!
//! Ported from hw/misc/sgi-xbow.c in jamesbraid/qemu-sgi-origami.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, OnceLock};

use parking_lot::Mutex;

use super::heart::Clock;
use crate::irq::InterruptController;
use crate::snapshot::{field_bool, field_u32, field_u32s, field_u8, StateTable};
use crate::traits::{BusDevice, BusRead16, BusRead32, BusRead64, BusRead8, BUS_ERR, BUS_OK, Saveable};

/// A widget ID register: JTAG IDCODE layout, bit 0 set.
pub const fn widget_id(rev: u32, part: u32, mfgr: u32) -> u32 {
    (rev << 28) | (part << 12) | (mfgr << 1) | 1
}

/// The widget HEART answers as, where widget 0's interrupt packets go.
pub(crate) const WIDGET_HEART: u8 = 8;

pub(crate) const FIRST_WIDGET: u8 = 8;
pub(crate) const LINKS: usize = 8;

/// The register block's span in widget 0's window.
pub const REG_SPAN: u32 = 0x0100_0000;

pub(crate) const IDENTIFICATION: u32 = 0x00;
pub(crate) const STATUS: u32 = 0x08;
pub(crate) const ERR_UPPER: u32 = 0x10;
pub(crate) const ERR_LOWER: u32 = 0x18;
pub(crate) const CONTROL: u32 = 0x20;
pub(crate) const PACKET_TIMEOUT: u32 = 0x28;
pub(crate) const INT_UPPER: u32 = 0x30;
pub(crate) const INT_LOWER: u32 = 0x38;
pub(crate) const ERR_CMD: u32 = 0x40;
pub(crate) const LLP_CONTROL: u32 = 0x48;
pub(crate) const STATUS_CLEAR: u32 = 0x50;
pub(crate) const ARB_RELOAD: u32 = 0x58;
pub(crate) const PERF_A: u32 = 0x60;
pub(crate) const PERF_B: u32 = 0x68;
pub(crate) const NIC: u32 = 0x70;
const CORE_CYCLE_NS: u64 = 10;
/// How long a link reset holds the link down: two core cycles.
const LINK_RESET_NS: u64 = 2 * CORE_CYCLE_NS;
/// A warm reset arriving this many cycles after another is ignored.
const WARM_RESET_QUIET_CYCLES: u64 = 1024;
/// Revision 5 ("Crossbow 2.0") and later: a fence mask for widget 0's core,
/// one per link, and the resource lock.
const FENCED_REVISION: u32 = 5;
pub(crate) const FENCE: u32 = 0x78;
pub(crate) const LINK_FENCE_BASE: u32 = 0x80;
pub(crate) const LINK_FENCE_END: u32 = LINK_FENCE_BASE + 8 * LINKS as u32;
/// The fence block as the other crossbars' models name it: widget 0's
/// fence through link 15's.
pub(crate) const FENCE_BASE: u32 = FENCE;
pub(crate) const FENCE_END: u32 = LINK_FENCE_END;
const FENCE_RESET: u8 = 0xff;
pub(crate) const LOCK: u32 = 0xc0;
pub(crate) const LOCK_CLEAR: u32 = 0xc8;
/// Where a status read carries the widget that asked, when the chip is told
/// to.
const STATUS_SOURCE_SHIFT: u32 = 6;
const STATUS_SOURCE_MASK: u32 = 0xf << STATUS_SOURCE_SHIFT;
pub(crate) const LINK_BASE: u32 = 0x100;
pub(crate) const LINK_STRIDE: u32 = 0x40;
pub(crate) const LINK_END: u32 = LINK_BASE + LINK_STRIDE * LINKS as u32;
pub(crate) const LINK_FLUSH: u32 = 0x00;
pub(crate) const LINK_CONTROL: u32 = 0x08;
pub(crate) const LINK_STATUS: u32 = 0x10;
pub(crate) const LINK_ARB_UPPER: u32 = 0x18;
pub(crate) const LINK_ARB_LOWER: u32 = 0x20;
pub(crate) const LINK_STATUS_CLEAR: u32 = 0x28;
pub(crate) const LINK_RESET: u32 = 0x30;
pub(crate) const LINK_AUX_STATUS: u32 = 0x38;

const STATUS_REQUEST: u32 = 1 << 23;
const STATUS_LINK_REQUESTS: u32 = 0xff00_0000;
const STATUS_ACCESS_ERROR: u32 = 1 << 5;
const STATUS_XTALK_ERROR: u32 = 1 << 2;
const STATUS_MULTIPLE_ERROR: u32 = 1 << 0;
const STATUS_WIDGET0_ERRORS: u32 = STATUS_ACCESS_ERROR | STATUS_XTALK_ERROR | STATUS_MULTIPLE_ERROR;
const STATUS_ASSERTING: u32 = STATUS_LINK_REQUESTS | STATUS_REQUEST;
const CONTROL_MASK: u32 = 0x0000_0026;
const CONTROL_CONDITIONS: u32 = STATUS_ACCESS_ERROR | STATUS_XTALK_ERROR;
const PACKET_TIMEOUT_MASK: u32 = 0x000f_ffff;
const INT_UPPER_MASK: u32 = 0xff07_ffff;
const INT_UPPER_TARGET_HIGH: u32 = 1 << 19;
const LLP_CONTROL_MASK: u32 = 0x03ff_ffff;
const LLP_CONTROL_RESET: u32 = 0x03ff_1810;
const ARB_RELOAD_MASK: u32 = 0x3f;
const ARB_RELOAD_RESET: u32 = 0x02;
const PERF_SELECT_MASK: u32 = 0x00f0_0000;
const PERF_RESET: u32 = 0x0080_0000;
const NIC_MASK: u32 = 0x000f_ffff;
const NIC_RESET: u32 = 0x02;

const LINK_ALIVE: u32 = 1 << 31;
const LINK_MAX_RETRY: u32 = 1 << 5;
const LINK_STATUS_ERRORS: u32 = 0x0007_fffb;
const LINK_ALIVE_ENABLE: u32 = 1 << 31;
const LINK_BANDWIDTH_ENABLE: u32 = 1 << 8;
const LINK_BANDWIDTH_SOURCE: u32 = 0x0000_ff00;
const LINK_DIRECT_ENABLES: u32 = 0x0003_00fb;
const LINK_FORCE_BAD: u32 = 1 << 23;
const LINK_EIGHT_BIT_FORCE: u32 = 1 << 24;
const LINK_CONTROL_MASK: u32 = (1 << 31) | (3 << 28) | (7 << 25) | (1 << 24) | (0x1f << 18) | (1 << 17) | (1 << 16) | 0x1ff;
const LINK_CREDIT_RESET: u32 = 2 << 18;
const AUX_FAILURE_MODE: u32 = 1 << 6;
const AUX_PRESENT: u32 = 1 << 5;
const AUX_EIGHT_BIT: u32 = 1 << 4;
const AUX_ERRORS: u32 = 0xffff_ff00;

struct Regs {
    status: u32,
    err_upper: u32,
    err_lower: u32,
    control: u32,
    packet_timeout: u32,
    int_upper: u32,
    int_lower: u32,
    err_cmd: u32,
    llp_control: u32,
    arb_reload: u32,
    perf_a: u32,
    perf_b: u32,
    nic: u32,
    fence: u8,
    link_fence: [u8; LINKS],
    lock: u8,
    link_control: [u32; LINKS],
    link_status: [u32; LINKS],
    link_arb_upper: [u32; LINKS],
    link_arb_lower: [u32; LINKS],
    link_aux_status: [u32; LINKS],
    /// Conditions since the link's status was last acknowledged, in its
    /// status register's bit positions.
    link_event: [u32; LINKS],
    came_up: u8,
    initializing: u8,
    eight_bit: u8,
    interrupt_asserted: bool,
    /// When each link's reset pulse ends, in clock ns; 0 when none runs.
    reset_due: [u64; LINKS],
    /// Links that were alive when their pulse began and return alive.
    reset_restore: u8,
    /// A warm reset before this time is ignored.
    warm_quiet_until: u64,
}

impl Regs {
    /// `startup`: the links present, which come up alive.
    fn reset(startup: u8) -> Self {
        Self {
            status: 0,
            err_upper: 0,
            err_lower: 0,
            control: 0,
            packet_timeout: PACKET_TIMEOUT_MASK,
            int_upper: INT_UPPER_TARGET_HIGH,
            int_lower: 0,
            err_cmd: 0,
            llp_control: LLP_CONTROL_RESET,
            arb_reload: ARB_RELOAD_RESET,
            perf_a: PERF_RESET,
            perf_b: PERF_RESET,
            nic: NIC_RESET,
            fence: FENCE_RESET,
            link_fence: [FENCE_RESET; LINKS],
            lock: 0,
            link_control: [LINK_CREDIT_RESET; LINKS],
            link_status: [0; LINKS],
            link_arb_upper: [0; LINKS],
            link_arb_lower: [0; LINKS],
            link_aux_status: [0; LINKS],
            link_event: [0; LINKS],
            came_up: startup,
            initializing: 0,
            eight_bit: 0,
            interrupt_asserted: false,
            reset_due: [0; LINKS],
            reset_restore: 0,
            warm_quiet_until: 0,
        }
    }
}

/// What an access leaves to do once the register lock is dropped.
#[derive(Default)]
struct Out {
    /// Widget 0's interrupt packets to HEART: (vector, set).
    packets: Vec<(u8, bool)>,
    /// Packets to a widget other than HEART: widget, address, vector, set.
    xio: Vec<(u8, u64, u8, bool)>,
    /// A link reset's far-end widget.
    link_resets: Vec<u8>,
}

/// Where widget 0's interrupt goes when its target is not HEART: the target
/// widget, the 48-bit address, the vector and whether it raises. False is a
/// refusal.
pub type PacketSink = Box<dyn Fn(u8, u64, u8, bool) -> bool + Send + Sync>;

pub struct Xbow {
    regs: Mutex<Regs>,
    revision: u32,
    /// Revision field of the identification word when it differs from the
    /// revision that selects behaviour.
    reported_revision: Option<u32>,
    part: u32,
    manufacturer: u32,
    /// The widget a status read reports as its source, if the chip does.
    status_source: Option<u8>,
    packets: OnceLock<PacketSink>,
    clock: Option<Clock>,
    /// The earliest running pulse's end, or `u64::MAX`. Written under `regs`.
    reset_due: AtomicU64,
    /// Links with something attached, bit n for widget 8 + n.
    present: AtomicU8,
    /// Links passing traffic, as `present`. Written under `regs`.
    alive: AtomicU8,
    /// Held across "decide a packet, send it", so two threads' packets for
    /// widget 0's one interrupt arrive in the order they were decided.
    /// Never held with `regs`, and nothing but HEART is called under it.
    irq_order: Mutex<()>,
    /// Where widget 0's interrupt packets go when they target HEART.
    heart: OnceLock<Arc<dyn InterruptController>>,
    /// Told when a link comes up or goes down, with the widget and its new
    /// state, outside the register lock.
    on_link_change: OnceLock<Box<dyn Fn(u8, bool) + Send + Sync>>,
    /// Told of a link reset, with the widget, once the switch's own state
    /// has settled: the widget at the far end resets too.
    on_link_reset: OnceLock<Box<dyn Fn(u8) + Send + Sync>>,
    unsupported: Mutex<Vec<(u32, u8, bool)>>,
    /// `unsupported` holds 64: a further access costs one relaxed load.
    unsupported_full: AtomicBool,
    /// A write to an error address register completes; see
    /// [`Xbow::with_posted_error_address_writes`].
    posted_error_address_writes: bool,
}

fn link_bit(widget: u8) -> Option<u8> {
    (FIRST_WIDGET..FIRST_WIDGET + LINKS as u8).contains(&widget).then(|| 1 << (widget - FIRST_WIDGET))
}

impl Xbow {
    /// A Crossbow of `revision` with the links in `widgets` (a widget mask,
    /// bit n for widget n) attached and alive at power-on.
    pub fn new(revision: u32, widgets: u16) -> Self {
        let present = (widgets >> FIRST_WIDGET) as u8;
        Self {
            regs: Mutex::new(Regs::reset(present)),
            revision,
            reported_revision: None,
            part: 0,
            manufacturer: 0,
            status_source: None,
            packets: OnceLock::new(),
            clock: None,
            reset_due: AtomicU64::new(u64::MAX),
            present: AtomicU8::new(present),
            alive: AtomicU8::new(present),
            irq_order: Mutex::new(()),
            heart: OnceLock::new(),
            on_link_change: OnceLock::new(),
            on_link_reset: OnceLock::new(),
            unsupported: Mutex::new(Vec::new()),
            unsupported_full: AtomicBool::new(false),
            posted_error_address_writes: false,
        }
    }

    /// Complete a write to either error address register instead of
    /// refusing it. The registers are read-only and the write still changes
    /// nothing, sets the register-access error and is listed in
    /// [`Xbow::unsupported`]: the specification makes a write to a read-only
    /// register an error condition that widget 0's status reports. What the processor sees is the other half: an
    /// uncached store is a write request with no completion response (NEC
    /// U10278EJ4V0UM p.104, table 6-1), so no error can come back to it.
    /// A stock IP35 PROM writes the lower register on purpose in its crossbar
    /// check. QEMU's `complete-posted-error-address-writes` property.
    pub fn with_posted_error_address_writes(mut self) -> Self {
        self.posted_error_address_writes = true;
        self
    }

    /// Answer to `part` and `manufacturer` in the identification word
    /// instead of the Crossbow's zeros.
    pub fn with_identity(mut self, part: u32, manufacturer: u32) -> Self {
        self.part = part;
        self.manufacturer = manufacturer;
        self
    }

    /// Report `revision` in the identification word while behaviour stays
    /// that of the revision given to [`Xbow::new`].
    pub fn with_reported_revision(mut self, revision: u32) -> Self {
        self.reported_revision = Some(revision);
        self
    }

    /// Time link resets and warm resets by `clock`, in nanoseconds.
    pub fn with_clock(mut self, clock: Clock) -> Self {
        self.clock = Some(clock);
        self
    }

    /// Report `widget` in bits 9:6 of every status read.
    pub fn with_status_source(mut self, widget: u8) -> Self {
        self.status_source = Some(widget);
        self
    }

    /// Send widget 0's interrupt to `sink` when it targets a widget other
    /// than HEART. Once, before the machine runs.
    pub fn connect_packets(&self, sink: PacketSink) {
        let _ = self.packets.set(sink);
    }

    /// Where widget 0's interrupt packets go: HEART, whose line is the
    /// vector the packet names. Once, before the machine runs.
    pub fn connect_heart(&self, heart: Arc<dyn InterruptController>) {
        let _ = self.heart.set(heart);
    }

    /// What to do when a link comes up or goes down (`widget`, `alive`):
    /// the machine points the widget's windows at it or at the bus error.
    /// Called with no Crossbow lock held.
    pub fn on_link_change(&self, f: Box<dyn Fn(u8, bool) + Send + Sync>) {
        let _ = self.on_link_change.set(f);
    }

    /// What to do when a link reset reaches the widget on the far end.
    /// Called with no Crossbow lock held.
    pub fn on_link_reset(&self, f: Box<dyn Fn(u8) + Send + Sync>) {
        let _ = self.on_link_reset.set(f);
    }

    pub fn register_locks(self: &Arc<Self>) {
        use crate::locks::register_lock_fn;
        let me = self.clone();
        register_lock_fn("xbow::regs", move || me.regs.is_locked());
        let me = self.clone();
        register_lock_fn("xbow::irq_order", move || me.irq_order.is_locked());
    }

    /// Run `f` on the registers, then do what it left outside the lock:
    /// widget 0's interrupt packets, link changes, a link reset.
    fn locked<R>(&self, f: impl FnOnce(&mut Regs, &mut Out) -> R) -> R {
        let order = self.irq_order.lock();
        let before = self.alive.load(Ordering::Acquire);
        let mut out = Out::default();
        let ret = {
            let mut r = self.regs.lock();
            self.end_pulses(&mut r, &mut out);
            let ret = f(&mut r, &mut out);
            let due = r.reset_due.iter().copied().filter(|&d| d != 0).min().unwrap_or(u64::MAX);
            self.reset_due.store(due, Ordering::Release);
            ret
        };
        if let Some(h) = self.heart.get() {
            for (vector, set) in out.packets {
                h.set_line(vector as u32, set);
            }
        }
        if let Some(sink) = self.packets.get() {
            for (widget, address, vector, set) in out.xio {
                if !sink(widget, address, vector, set) {
                    crate::dlog!(crate::devlog::LogModule::Bedrock, "xbow: interrupt packet to widget {widget} refused");
                }
            }
        }
        drop(order);
        let changed = before ^ self.alive.load(Ordering::Acquire);
        if let Some(cb) = self.on_link_change.get() {
            for link in (0..LINKS as u8).filter(|l| changed >> l & 1 != 0) {
                cb(FIRST_WIDGET + link, self.alive.load(Ordering::Acquire) >> link & 1 != 0);
            }
        }
        if let Some(cb) = self.on_link_reset.get() {
            for w in out.link_resets {
                cb(w);
            }
        }
        ret
    }

    /// Fit a board on `widget` after construction (graphics), alive.
    pub fn attach(&self, widget: u8) {
        if let Some(bit) = link_bit(widget) {
            self.locked(|r, _| {
                self.present.fetch_or(bit, Ordering::AcqRel);
                self.alive.fetch_or(bit, Ordering::AcqRel);
                r.came_up |= bit;
            });
        }
    }

    /// Whether the switch passes traffic to and from `widget`. Lock-free.
    pub fn carries(&self, widget: u8) -> bool {
        if let Some(clock) = &self.clock {
            if clock() >= self.reset_due.load(Ordering::Acquire) {
                self.locked(|_, _| {});
            }
        }
        self.carries_now(widget)
    }

    /// [`Self::carries`] without ending a pulse that is over, for use with
    /// the register lock held.
    fn carries_now(&self, widget: u8) -> bool {
        link_bit(widget).is_some_and(|b| self.alive.load(Ordering::Acquire) & b != 0)
    }

    pub fn unsupported(&self) -> Vec<(u32, u8, bool)> {
        self.unsupported.lock().clone()
    }

    fn identification(&self) -> u32 {
        (self.reported_revision.unwrap_or(self.revision) << 28) | (self.part << 12) | (self.manufacturer << 1) | 1
    }

    /// A status word as a read reports it: with the asking widget in the
    /// source field, when the chip has one.
    fn reported_status(&self, status: u32) -> u32 {
        match self.status_source {
            Some(widget) => (status & !STATUS_SOURCE_MASK) | ((widget as u32) << STATUS_SOURCE_SHIFT),
            None => status,
        }
    }

    fn link_state(&self, r: &Regs, link: usize) -> u32 {
        let bit = 1u8 << link;
        if self.alive.load(Ordering::Acquire) & bit != 0 {
            LINK_ALIVE
        } else if r.initializing & bit != 0 {
            0
        } else {
            LINK_MAX_RETRY
        }
    }

    fn link_status(&self, r: &Regs, link: usize) -> u32 {
        r.link_status[link] | self.link_state(r, link)
    }

    fn link_aux(&self, r: &Regs, link: usize) -> u32 {
        let bit = 1u8 << link;
        let mut v = r.link_aux_status[link] & AUX_ERRORS;
        if self.present.load(Ordering::Acquire) & bit != 0 {
            v |= AUX_PRESENT;
        }
        if self.alive.load(Ordering::Acquire) & bit != 0 {
            if r.eight_bit & bit != 0 {
                v |= AUX_EIGHT_BIT;
            }
        } else if r.initializing & bit == 0 && r.came_up & bit == 0 {
            v |= AUX_FAILURE_MODE;
        }
        v
    }

    fn link_requesting(r: &Regs, link: usize) -> bool {
        let (enables, events) = (r.link_control[link], r.link_event[link]);
        events & enables & (LINK_ALIVE_ENABLE | LINK_DIRECT_ENABLES) != 0
            || (enables & LINK_BANDWIDTH_ENABLE != 0 && events & LINK_BANDWIDTH_SOURCE != 0)
    }

    /// Recompute the request bits and, when the summary changes, queue widget
    /// 0's interrupt packet: one assertion outstanding at a time, the
    /// deassertion once every request is acknowledged. A packet to anything
    /// but HEART goes nowhere on this machine.
    fn update_interrupt(&self, r: &mut Regs, out: &mut Out) {
        for link in 0..LINKS {
            if Self::link_requesting(r, link) {
                r.status |= 1 << (24 + link);
            } else {
                r.status &= !(1 << (24 + link));
            }
        }
        if r.status & r.control & CONTROL_CONDITIONS != 0 {
            r.status |= STATUS_REQUEST;
        }
        let pending = r.status & STATUS_ASSERTING != 0;
        if pending == r.interrupt_asserted {
            return;
        }
        let target = ((r.int_upper >> 16) & 0xf) as u8;
        let vector = (r.int_upper >> 24) as u8;
        if !self.carries_now(target) {
            return;
        }
        if target == WIDGET_HEART {
            if self.heart.get().is_none() {
                return;
            }
            out.packets.push((vector, pending));
        } else {
            if self.packets.get().is_none() {
                return;
            }
            let address = ((r.int_upper as u64 & 0xffff) << 32) | r.int_lower as u64;
            out.xio.push((target, address, vector, pending));
        }
        r.interrupt_asserted = pending;
    }

    fn record_access_error(&self, r: &mut Regs, out: &mut Out) {
        if r.status & STATUS_ACCESS_ERROR != 0 {
            r.status |= STATUS_MULTIPLE_ERROR;
        }
        r.status |= STATUS_ACCESS_ERROR;
        self.update_interrupt(r, out);
    }

    fn note(&self, off: u32, size: usize, write: bool) {
        if self.unsupported_full.load(Ordering::Relaxed) {
            return;
        }
        let mut u = self.unsupported.lock();
        if !u.iter().any(|&(o, s, w)| o == off && s as usize == size && w == write) {
            u.push((off, size as u8, write));
            if u.len() == 64 {
                self.unsupported_full.store(true, Ordering::Relaxed);
            }
        }
    }

    fn accessible(&self, slot: u32, write: bool) -> bool {
        match slot {
            IDENTIFICATION | STATUS | ERR_UPPER | ERR_LOWER | STATUS_CLEAR => !write,
            CONTROL | PACKET_TIMEOUT | INT_UPPER | INT_LOWER | ERR_CMD | LLP_CONTROL | ARB_RELOAD | PERF_A
            | PERF_B | NIC => true,
            FENCE | LOCK | LOCK_CLEAR => self.revision >= FENCED_REVISION,
            s if (LINK_FENCE_BASE..LINK_FENCE_END).contains(&s) => self.revision >= FENCED_REVISION,
            s if (LINK_BASE..LINK_END).contains(&s) => match (s - LINK_BASE) % LINK_STRIDE {
                LINK_FLUSH | LINK_STATUS | LINK_STATUS_CLEAR | LINK_AUX_STATUS => !write,
                LINK_CONTROL | LINK_ARB_UPPER | LINK_ARB_LOWER => true,
                LINK_RESET => write,
                _ => false,
            },
            _ => false,
        }
    }

    /// Decode an access to widget 0's window. Registers are 32 bits in the
    /// low half of an eight-byte slot; an eight-byte access has no
    /// established packet form and is refused without a status flag, any
    /// other malformed access is an access error.
    fn decode(&self, r: &mut Regs, out: &mut Out, off: u32, size: usize, write: bool) -> Option<u32> {
        if size == 8 {
            self.note(off, size, write);
            return None;
        }
        let slot = off.wrapping_sub(4);
        if size != 4 || off % 8 != 4 || !self.accessible(slot, write) {
            self.note(off, size, write);
            self.record_access_error(r, out);
            return None;
        }
        Some(slot)
    }

    /// A processor read of widget 0's window at `off`; `None` is a bus error.
    pub fn read(&self, off: u32, size: usize) -> Option<u64> {
        self.locked(|r, out| {
            let slot = self.decode(r, out, off, size, false)?;
            let v = match slot {
                IDENTIFICATION => self.identification(),
                STATUS => self.reported_status(r.status),
                ERR_UPPER => r.err_upper,
                ERR_LOWER => r.err_lower,
                CONTROL => r.control,
                PACKET_TIMEOUT => r.packet_timeout,
                INT_UPPER => r.int_upper,
                INT_LOWER => r.int_lower,
                ERR_CMD => r.err_cmd,
                LLP_CONTROL => r.llp_control,
                STATUS_CLEAR => {
                    // Acknowledges widget 0's own errors and request; each
                    // link's request is acknowledged at its own alias.
                    let v = self.reported_status(r.status);
                    r.status &= !(STATUS_WIDGET0_ERRORS | STATUS_REQUEST);
                    self.update_interrupt(r, out);
                    v
                }
                ARB_RELOAD => r.arb_reload,
                PERF_A => r.perf_a,
                PERF_B => r.perf_b,
                NIC => r.nic,
                FENCE => r.fence as u32,
                LOCK | LOCK_CLEAR => r.lock as u32,
                s if (LINK_FENCE_BASE..LINK_FENCE_END).contains(&s) => {
                    r.link_fence[((s - LINK_FENCE_BASE) / 8) as usize] as u32
                }
                s => {
                    let link = ((s - LINK_BASE) / LINK_STRIDE) as usize;
                    match (s - LINK_BASE) % LINK_STRIDE {
                        // Nothing is queued, so a flush is over before it is read.
                        LINK_FLUSH => 0,
                        LINK_CONTROL => r.link_control[link],
                        LINK_STATUS => self.link_status(r, link),
                        LINK_ARB_UPPER => r.link_arb_upper[link],
                        LINK_ARB_LOWER => r.link_arb_lower[link],
                        LINK_STATUS_CLEAR => {
                            let v = self.link_status(r, link);
                            r.link_status[link] &= !LINK_STATUS_ERRORS;
                            r.link_aux_status[link] &= !AUX_ERRORS;
                            r.link_event[link] = 0;
                            self.update_interrupt(r, out);
                            v
                        }
                        LINK_AUX_STATUS => self.link_aux(r, link),
                        _ => unreachable!(),
                    }
                }
            };
            Some(v as u64)
        })
    }

    /// A processor write; false is a bus error.
    pub fn write(&self, off: u32, size: usize, val: u64) -> bool {
        self.locked(|r, out| {
            let Some(slot) = self.decode(r, out, off, size, true) else {
                // Refused and recorded either way; a posted error address
                // write completes all the same.
                return self.posted_error_address_writes
                    && size == 4
                    && matches!(off.wrapping_sub(4), ERR_UPPER | ERR_LOWER);
            };
            let val = val as u32;
            match slot {
                CONTROL => {
                    r.control = val & CONTROL_MASK;
                    self.update_interrupt(r, out);
                }
                PACKET_TIMEOUT => r.packet_timeout = val & PACKET_TIMEOUT_MASK,
                INT_UPPER => r.int_upper = (val & INT_UPPER_MASK) | INT_UPPER_TARGET_HIGH,
                INT_LOWER => r.int_lower = val,
                ERR_CMD => {
                    // Rearms error logging; the data is discarded.
                    r.err_cmd = 0;
                    r.err_upper = 0;
                    r.err_lower = 0;
                }
                LLP_CONTROL => r.llp_control = val & LLP_CONTROL_MASK,
                ARB_RELOAD => r.arb_reload = val & ARB_RELOAD_MASK,
                // The selector is kept; nothing counts.
                PERF_A => r.perf_a = val & PERF_SELECT_MASK,
                PERF_B => r.perf_b = val & PERF_SELECT_MASK,
                NIC => r.nic = val & NIC_MASK,
                FENCE => r.fence = val as u8,
                // A test-and-set: the write only takes while the lock is free.
                LOCK => {
                    if r.lock == 0 {
                        r.lock = val as u8;
                    }
                }
                LOCK_CLEAR => r.lock = 0,
                s if (LINK_FENCE_BASE..LINK_FENCE_END).contains(&s) => {
                    r.link_fence[((s - LINK_FENCE_BASE) / 8) as usize] = val as u8
                }
                s => {
                    let link = ((s - LINK_BASE) / LINK_STRIDE) as usize;
                    match (s - LINK_BASE) % LINK_STRIDE {
                        LINK_CONTROL => {
                            r.link_control[link] = val & LINK_CONTROL_MASK & !LINK_FORCE_BAD;
                            self.update_interrupt(r, out);
                        }
                        // Stored; nothing arbitrates on them.
                        LINK_ARB_UPPER => r.link_arb_upper[link] = val,
                        LINK_ARB_LOWER => r.link_arb_lower[link] = val,
                        LINK_RESET => {
                            self.link_reset(r, link, out);
                            // After the lock: the widget's reset may ask
                            // the switch about its links.
                            out.link_resets.push(FIRST_WIDGET + link as u8);
                        }
                        _ => unreachable!(),
                    }
                }
            }
            true
        })
    }

    /// A link reset. The pulse is two core cycles, over before the
    /// processor can look again, so it is not kept. A fitted link that was
    /// alive comes back alive with an alive event, as the Octane's
    /// restore-alive policy has it; one that was down stays down, still
    /// initializing; an empty port fails initialization.
    fn link_reset(&self, r: &mut Regs, link: usize, out: &mut Out) {
        let bit = 1u8 << link;
        if let Some(clock) = &self.clock {
            self.begin_pulse(r, link, clock(), out);
            return;
        }
        let was_alive = self.alive.load(Ordering::Acquire) & bit != 0;
        r.eight_bit &= !bit;
        if self.present.load(Ordering::Acquire) & bit != 0 {
            if was_alive {
                r.initializing &= !bit;
                r.came_up |= bit;
                r.link_event[link] |= LINK_ALIVE;
                if r.link_control[link] & LINK_EIGHT_BIT_FORCE != 0 {
                    r.eight_bit |= bit;
                }
            } else {
                r.initializing |= bit;
            }
        } else {
            self.alive.fetch_and(!bit, Ordering::AcqRel);
            r.initializing &= !bit;
            r.came_up &= !bit;
        }
        self.update_interrupt(r, out);
    }
}

impl Xbow {
    /// A link reset with a clock: the link drops now and the pulse ends
    /// two core cycles on. A fitted link that was alive comes back alive
    /// then, with an alive event; one that was down comes out of
    /// initialization still down; an empty port fails initialization and
    /// stays failed.
    fn begin_pulse(&self, r: &mut Regs, link: usize, now: u64, out: &mut Out) {
        let bit = 1u8 << link;
        let present = self.present.load(Ordering::Acquire) & bit != 0;
        if present && self.alive.load(Ordering::Acquire) & bit != 0 {
            r.reset_restore |= bit;
        } else {
            r.reset_restore &= !bit;
        }
        self.alive.fetch_and(!bit, Ordering::AcqRel);
        r.eight_bit &= !bit;
        if present {
            r.initializing |= bit;
        } else {
            r.initializing &= !bit;
            r.came_up &= !bit;
        }
        r.reset_due[link] = now + LINK_RESET_NS;
        self.update_interrupt(r, out);
    }

    /// End every pulse the clock says is over.
    fn end_pulses(&self, r: &mut Regs, out: &mut Out) {
        let Some(clock) = &self.clock else { return };
        if r.reset_due.iter().all(|&d| d == 0) {
            return;
        }
        let now = clock();
        for link in 0..LINKS {
            let bit = 1u8 << link;
            if r.reset_due[link] == 0 || r.reset_due[link] > now {
                continue;
            }
            r.reset_due[link] = 0;
            if self.present.load(Ordering::Acquire) & bit == 0 {
                continue;
            }
            r.initializing &= !bit;
            if r.reset_restore & bit != 0 {
                self.alive.fetch_or(bit, Ordering::AcqRel);
                r.came_up |= bit;
                r.link_event[link] |= LINK_ALIVE;
                if r.link_control[link] & LINK_EIGHT_BIT_FORCE != 0 {
                    r.eight_bit |= bit;
                }
            }
            r.reset_restore &= !bit;
            self.update_interrupt(r, out);
        }
    }

    /// A warm reset arriving on the link of `source` widget, at `now` ns.
    /// The first in 1024 core cycles is taken and later ones are ignored.
    /// On revision 5 and later each destination has a fence naming the
    /// sources it admits: the core resets only if its fence admits the
    /// source, and each link whose fence does gets a reset pulse out to
    /// its far end ([`Self::on_link_reset`]). Fences keep their programmed
    /// value; the lock clears with the core. Earlier revisions admit every
    /// source.
    pub fn warm_reset(&self, source: u8, now: u64) {
        let Some(port) = link_bit(source).map(|b| b.trailing_zeros() as usize) else { return };
        self.locked(|r, out| {
            if now < r.warm_quiet_until {
                crate::dlog!(crate::devlog::LogModule::Bedrock, "xbow: warm reset from widget {source} inside the quiet window; ignored");
                return;
            }
            let (fence, link_fence) = (r.fence, r.link_fence);
            if fence >> port & 1 != 0 {
                let present = self.present.load(Ordering::Acquire);
                *r = Regs::reset(present);
                self.alive.store(present, Ordering::Release);
                r.fence = fence;
                r.link_fence = link_fence;
                self.update_interrupt(r, out);
            }
            r.warm_quiet_until = now + WARM_RESET_QUIET_CYCLES * CORE_CYCLE_NS;
            for (link, fence) in link_fence.iter().enumerate() {
                if fence >> port & 1 != 0 {
                    out.link_resets.push(FIRST_WIDGET + link as u8);
                }
            }
        });
    }
}

impl crate::traits::Resettable for Xbow {
    fn power_on(&self) {
        self.locked(|r, _| {
            let present = self.present.load(Ordering::Acquire);
            *r = Regs::reset(present);
            self.alive.store(present, Ordering::Release);
        });
    }
}

/// Widget 0's window as the machine's bus sees it: the full physical
/// address, of which the window's 16 MB is decoded.
const WINDOW_MASK: u64 = REG_SPAN as u64 - 1;

impl BusDevice for Xbow {
    fn read8(&self, addr: u64) -> BusRead8 {
        self.read((addr & WINDOW_MASK) as u32, 1).map_or(BusRead8::err(), |v| BusRead8::ok(v as u8))
    }

    fn read16(&self, addr: u64) -> BusRead16 {
        self.read((addr & WINDOW_MASK) as u32, 2).map_or(BusRead16::err(), |v| BusRead16::ok(v as u16))
    }

    fn read32(&self, addr: u64) -> BusRead32 {
        self.read((addr & WINDOW_MASK) as u32, 4).map_or(BusRead32::err(), |v| BusRead32::ok(v as u32))
    }

    fn read64(&self, addr: u64) -> BusRead64 {
        self.read((addr & WINDOW_MASK) as u32, 8).map_or(BusRead64::err(), BusRead64::ok)
    }

    fn write8(&self, addr: u64, val: u8) -> u32 {
        if self.write((addr & WINDOW_MASK) as u32, 1, val as u64) { BUS_OK } else { BUS_ERR }
    }

    fn write16(&self, addr: u64, val: u16) -> u32 {
        if self.write((addr & WINDOW_MASK) as u32, 2, val as u64) { BUS_OK } else { BUS_ERR }
    }

    fn write32(&self, addr: u64, val: u32) -> u32 {
        if self.write((addr & WINDOW_MASK) as u32, 4, val as u64) { BUS_OK } else { BUS_ERR }
    }

    fn write64(&self, addr: u64, val: u64) -> u32 {
        if self.write((addr & WINDOW_MASK) as u32, 8, val) { BUS_OK } else { BUS_ERR }
    }
}

/// Widget 0's fence and the links' fences, in the order the saved state
/// lists them.
fn fences_of(r: &Regs) -> [u32; 1 + LINKS] {
    let mut f = [0u32; 1 + LINKS];
    f[0] = u32::from(r.fence);
    for (dst, link) in f[1..].iter_mut().zip(&r.link_fence) {
        *dst = u32::from(*link);
    }
    f
}

// The crossbow's state for a snapshot: its registers and each link's
// state. Which links are fitted is the board's, and a load onto other
// links is refused. The interrupt the crossbow drives into HEART is
// HEART's own saved state.
impl Saveable for Xbow {
    fn save_state(&self) -> toml::Value {
        let r = self.regs.lock();
        StateTable::new()
            .u32s("words", &[
                r.status, r.err_upper, r.err_lower, r.control, r.packet_timeout, r.int_upper,
                r.int_lower, r.err_cmd, r.llp_control, r.arb_reload, r.perf_a, r.perf_b, r.nic,
            ])
            .u32s("link_control", &r.link_control)
            .u32s("link_status", &r.link_status)
            .u32s("link_arb_upper", &r.link_arb_upper)
            .u32s("link_arb_lower", &r.link_arb_lower)
            .u32s("link_aux_status", &r.link_aux_status)
            .u32s("link_event", &r.link_event)
            .u8("came_up", r.came_up)
            .u8("initializing", r.initializing)
            .u8("eight_bit", r.eight_bit)
            .bool("interrupt_asserted", r.interrupt_asserted)
            .u32s("fences", &fences_of(&r))
            .u32("lock", u32::from(r.lock))
            .u32("revision", self.revision)
            .u8("present", self.present.load(Ordering::Acquire))
            .u8("alive", self.alive.load(Ordering::Acquire))
            .build()
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        if field_u8(v, "present")? != self.present.load(Ordering::Acquire) {
            return Err("the saved crossbow has other links fitted".into());
        }
        if field_u32(v, "revision")? != self.revision {
            return Err("the saved crossbow is another revision".into());
        }
        let [status, err_upper, err_lower, control, packet_timeout, int_upper, int_lower, err_cmd,
            llp_control, arb_reload, perf_a, perf_b, nic] = field_u32s::<13>(v, "words")?;
        let mut n = Regs {
            status, err_upper, err_lower, control, packet_timeout, int_upper, int_lower, err_cmd,
            llp_control, arb_reload, perf_a, perf_b, nic,
            link_control: field_u32s(v, "link_control")?,
            link_status: field_u32s(v, "link_status")?,
            link_arb_upper: field_u32s(v, "link_arb_upper")?,
            link_arb_lower: field_u32s(v, "link_arb_lower")?,
            link_aux_status: field_u32s(v, "link_aux_status")?,
            link_event: field_u32s(v, "link_event")?,
            came_up: field_u8(v, "came_up")?,
            initializing: field_u8(v, "initializing")?,
            eight_bit: field_u8(v, "eight_bit")?,
            interrupt_asserted: field_bool(v, "interrupt_asserted")?,
            fence: 0,
            link_fence: [0; LINKS],
            lock: field_u32(v, "lock")? as u8,
            // A link reset lasts two core cycles and the warm-reset quiet
            // time is counted on the board clock; a snapshot carries
            // neither, so a restored crossbow has none in progress.
            reset_due: [0; LINKS],
            reset_restore: 0,
            warm_quiet_until: 0,
        };
        let fences: [u32; 1 + LINKS] = field_u32s(v, "fences")?;
        n.fence = fences[0] as u8;
        for (link, f) in n.link_fence.iter_mut().zip(&fences[1..]) {
            *link = *f as u8;
        }
        *self.regs.lock() = n;
        self.reset_due.store(u64::MAX, Ordering::Release);
        self.alive.store(field_u8(v, "alive")?, Ordering::Release);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    /// HEART's ISR as the Crossbow sees it: one bit per vector.
    #[derive(Default)]
    struct Isr(AtomicU64);
    impl InterruptController for Isr {
        fn set_line(&self, line: u32, level: bool) {
            let bit = 1u64 << line;
            if level {
                self.0.fetch_or(bit, Ordering::SeqCst);
            } else {
                self.0.fetch_and(!bit, Ordering::SeqCst);
            }
        }
    }

    fn octane() -> Xbow {
        Xbow::new(4, 0x8100)
    }

    #[test]
    fn identification_and_link_status() {
        let x = octane();
        assert_eq!(x.read(0x04, 4), Some(0x4000_0001));
        // HEART (8) and Bridge (15) alive; 11 empty.
        assert_eq!(x.read(LINK_BASE + 4 * 0 + LINK_STATUS + 4, 4).unwrap() as u32 & LINK_ALIVE, LINK_ALIVE);
        assert_eq!(x.read(LINK_BASE + LINK_STRIDE * 7 + LINK_STATUS + 4, 4).unwrap() as u32 & LINK_ALIVE, LINK_ALIVE);
        assert_eq!(x.read(LINK_BASE + LINK_STRIDE * 3 + LINK_STATUS + 4, 4).unwrap() as u32, LINK_MAX_RETRY);
        let aux = x.read(LINK_BASE + LINK_STRIDE * 3 + LINK_AUX_STATUS + 4, 4).unwrap() as u32;
        assert_eq!(aux & (AUX_PRESENT | AUX_FAILURE_MODE), AUX_FAILURE_MODE);
        assert!(x.carries(8) && x.carries(15) && !x.carries(11) && !x.carries(3));
    }

    #[test]
    fn the_unsupported_list_keeps_the_first_64_and_then_stops() {
        let x = octane();
        for i in 0..100u32 {
            x.note(0x1000 + 8 * i, 4, false);
        }
        let u = x.unsupported();
        assert_eq!(u.len(), 64);
        assert_eq!(u[63], (0x1000 + 8 * 63, 4, false));
        x.note(0x9000, 4, true);
        assert_eq!(x.unsupported(), u, "a full list is left alone");
    }

    #[test]
    fn malformed_accesses_are_refused() {
        let x = octane();
        assert_eq!(x.read(0x00, 4), None);
        assert_eq!(x.read(0x00, 8), None);
        assert_eq!(x.read(STATUS + 4, 4).unwrap() as u32 & STATUS_ACCESS_ERROR, STATUS_ACCESS_ERROR);
        assert!(!x.write(IDENTIFICATION + 4, 4, 0));
        assert_eq!(x.read(STATUS + 4, 4).unwrap() as u32 & STATUS_MULTIPLE_ERROR, STATUS_MULTIPLE_ERROR);
        let s = x.read(STATUS_CLEAR + 4, 4).unwrap() as u32;
        assert_ne!(s & STATUS_ACCESS_ERROR, 0);
        assert_eq!(x.read(STATUS + 4, 4).unwrap() as u32 & STATUS_WIDGET0_ERRORS, 0);
    }

    #[test]
    fn a_link_reset_restores_a_live_link() {
        let x = octane();
        let isr = Arc::new(Isr::default());
        x.connect_heart(isr.clone());
        // Interrupt to HEART, vector 0x2a, alive events enabled on link 15.
        assert!(x.write(INT_UPPER + 4, 4, (0x2a << 24) | (8 << 16)));
        let ctl = LINK_BASE + LINK_STRIDE * 7 + LINK_CONTROL + 4;
        assert!(x.write(ctl, 4, LINK_ALIVE_ENABLE as u64));
        assert!(x.write(LINK_BASE + LINK_STRIDE * 7 + LINK_RESET + 4, 4, 0));
        assert!(x.carries(15));
        assert_eq!(isr.0.load(Ordering::SeqCst), 1 << 0x2a);
        assert_ne!(x.read(STATUS + 4, 4).unwrap() as u32 & (1 << 31), 0);
        x.read(LINK_BASE + LINK_STRIDE * 7 + LINK_STATUS_CLEAR + 4, 4).unwrap();
        assert_eq!(isr.0.load(Ordering::SeqCst), 0);
        // An empty port stays down.
        assert!(x.write(LINK_BASE + LINK_STRIDE * 3 + LINK_RESET + 4, 4, 0));
        assert!(!x.carries(11));
    }

    #[test]
    fn reset_restores_startup_links() {
        let x = octane();
        x.attach(11);
        assert!(x.carries(11));
        assert!(x.write(INT_LOWER + 4, 4, 0x1234));
        crate::traits::Resettable::power_on(&x);
        assert!(x.carries(11));
        assert_eq!(x.read(INT_LOWER + 4, 4), Some(0));
        assert_eq!(x.read(INT_UPPER + 4, 4), Some(INT_UPPER_TARGET_HIGH as u64));
        assert_eq!(x.read(LLP_CONTROL + 4, 4), Some(LLP_CONTROL_RESET as u64));
    }

    /// The machine's bus hands the window the full physical address.
    #[test]
    fn the_window_decodes_its_own_16_mb_of_a_full_address() {
        let x = octane();
        assert_eq!(x.read32(0x1000_0004).data, 0x4000_0001);
        assert_eq!((x.read32(0x1000_0004).data >> 12) & 0xffff, 0, "Crossbow's part number is zero");
        assert!(!x.read32(0x1000_0000).is_ok(), "the empty half of the slot");
        assert!(!x.read64(0x1000_0004).is_ok());
        assert_eq!(x.write32(0x1000_0000 + INT_LOWER as u64 + 4, 0x55), BUS_OK);
        assert_eq!(x.read32(0x1000_0000 + INT_LOWER as u64 + 4).data, 0x55);
        assert_eq!(x.write32(0x1000_0000 + IDENTIFICATION as u64 + 4, 0), BUS_ERR);
    }

    /// A write to a read-only error address register is an access error and
    /// changes nothing. With posted error
    /// address writes the store still completes, as QEMU's crossbar does with
    /// `complete-posted-error-address-writes` (hw/misc/sgi-xbow.c, asserted
    /// set on the Origin 2000 by sn0-origin2000-test's xio-owner case); other
    /// read-only registers are refused as before.
    #[test]
    fn a_posted_error_address_write_completes_and_still_records_the_access_error() {
        let refusing = octane();
        assert_eq!(refusing.write32(ERR_LOWER as u64 + 4, 0xdead_beef), BUS_ERR);
        let x = octane().with_posted_error_address_writes();
        assert_eq!(x.write32(ERR_LOWER as u64 + 4, 0xdead_beef), BUS_OK);
        assert_eq!(x.read(ERR_LOWER + 4, 4), Some(0), "the register keeps its value");
        assert_eq!(x.read(STATUS + 4, 4).unwrap() as u32 & STATUS_ACCESS_ERROR, STATUS_ACCESS_ERROR);
        assert_eq!(x.write32(ERR_UPPER as u64 + 4, 0), BUS_OK);
        assert_eq!(x.read(STATUS + 4, 4).unwrap() as u32 & STATUS_MULTIPLE_ERROR, STATUS_MULTIPLE_ERROR);
        assert_eq!(x.unsupported(), vec![(ERR_LOWER + 4, 4, true), (ERR_UPPER + 4, 4, true)]);
        assert_eq!(x.write32(IDENTIFICATION as u64 + 4, 0), BUS_ERR, "only the error address registers");
        assert_eq!(x.write64(ERR_LOWER as u64, 0), BUS_ERR, "nor an eight-byte write");
    }

    /// The fences, the lock and a non-zero part number belong to revision 5
    /// and later; the Octane's revision 4 leaves them reserved.
    #[test]
    fn fences_the_lock_and_the_identity_fields_need_the_revision_that_has_them() {
        let x = octane();
        for slot in [FENCE, LINK_FENCE_BASE, LOCK, LOCK_CLEAR] {
            assert_eq!(x.read(slot + 4, 4), None, "slot {slot:#x} is reserved on revision 4");
            assert!(!x.write(slot + 4, 4, 1));
        }
        assert_eq!(x.read(IDENTIFICATION + 4, 4), Some(0x4000_0001));
        let y = Xbow::new(5, 0x8400).with_identity(0xd100, 0);
        assert_eq!(y.read(IDENTIFICATION + 4, 4), Some(0x5d10_0001));
        assert_eq!(y.read(FENCE + 4, 4), Some(0xff));
        assert!(y.write(FENCE + 4, 4, 0x1ff));
        assert_eq!(y.read(FENCE + 4, 4), Some(0xff), "a fence is eight bits");
    }

    fn timed(revision: u32) -> (Xbow, Arc<AtomicU64>) {
        let now = Arc::new(AtomicU64::new(0));
        let clock = now.clone();
        let x = Xbow::new(revision, 0x8100).with_clock(Arc::new(move || clock.load(Ordering::SeqCst)));
        (x, now)
    }

    /// Link reset is a pulse of two core cycles: the link is down and not
    /// initializing-failed while it lasts, then a fitted live link returns
    /// with an alive event. (sn0-xbow-test link-reset-pulse)
    #[test]
    fn a_link_reset_holds_the_link_down_for_twenty_nanoseconds() {
        let (x, now) = timed(4);
        let status = |link| LINK_BASE + LINK_STRIDE * link + LINK_STATUS + 4;
        now.store(1000, Ordering::SeqCst);
        assert!(x.write(LINK_BASE + LINK_STRIDE * 3 + LINK_RESET + 4, 4, 0), "widget 11 has nothing fitted");
        assert!(x.write(LINK_BASE + LINK_STRIDE * 7 + LINK_RESET + 4, 4, 0));
        assert!(!x.carries(15));
        assert_eq!(x.read(status(7), 4), Some(0), "initializing shows none of the three bits");
        assert_eq!(x.read(LINK_BASE + LINK_STRIDE * 7 + LINK_AUX_STATUS + 4, 4), Some(AUX_PRESENT as u64));
        now.store(1019, Ordering::SeqCst);
        assert!(!x.carries(15));
        now.store(1020, Ordering::SeqCst);
        assert!(x.carries(15), "the pulse is over by the clock without any register access");
        assert_eq!(x.read(status(7), 4).unwrap() as u32 & LINK_ALIVE, LINK_ALIVE);
        assert!(!x.carries(11), "an empty port stays down");
        assert_eq!(x.read(status(3), 4), Some(LINK_MAX_RETRY as u64));
        assert!(x.carries(8), "other links were not touched");
    }

    /// A register read ends a finished pulse before it answers.
    #[test]
    fn a_register_read_after_the_pulse_sees_the_link_back() {
        let (x, now) = timed(4);
        assert!(x.write(LINK_BASE + LINK_STRIDE * 7 + LINK_RESET + 4, 4, 0));
        now.store(20, Ordering::SeqCst);
        let aux = x.read(LINK_BASE + LINK_STRIDE * 7 + LINK_AUX_STATUS + 4, 4).unwrap() as u32;
        assert_eq!(aux & (AUX_PRESENT | AUX_FAILURE_MODE), AUX_PRESENT);
    }

    fn resets(x: &Xbow) -> Arc<Mutex<Vec<u8>>> {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let s = seen.clone();
        x.on_link_reset(Box::new(move |w| s.lock().push(w)));
        seen
    }

    /// Fences gate a warm reset per destination, an accepted one leaves
    /// them as programmed, and a machine reset restores them.
    /// (sn0-xbow-test fence-persists-across-warm-reset,
    /// fence-independence-link-outlives-core-rejection)
    #[test]
    fn a_warm_reset_obeys_each_destinations_fence_and_keeps_the_fences() {
        let (x, now) = timed(5);
        let seen = resets(&x);
        assert!(x.write(CONTROL + 4, 4, 0x26));
        assert!(x.write(FENCE + 4, 4, 0xbf));
        assert!(x.write(LINK_FENCE_BASE + 8 * 7 + 4, 4, 0x3c));
        assert!(x.write(LINK_FENCE_BASE + 4, 4, 0xfd));
        x.warm_reset(15, 0);
        assert_eq!(x.read(CONTROL + 4, 4), Some(0), "the core admitted widget 15 and reset");
        assert_eq!(x.read(FENCE + 4, 4), Some(0xbf));
        assert_eq!(x.read(LINK_FENCE_BASE + 8 * 7 + 4, 4), Some(0x3c));
        // Link 7's fence refuses port 15 (bit 7 clear); the rest admit it.
        assert_eq!(*seen.lock(), (8..15).collect::<Vec<u8>>());
        // A core that refuses the source is left alone while links still admit.
        assert!(x.write(CONTROL + 4, 4, 0x26));
        assert!(x.write(FENCE + 4, 4, 0xbf & !(1 << 7)));
        now.store(20_000, Ordering::SeqCst);
        x.warm_reset(15, 20_000);
        assert_eq!(x.read(CONTROL + 4, 4), Some(0x26));
        crate::traits::Resettable::power_on(&x);
        assert_eq!(x.read(FENCE + 4, 4), Some(0xff));
        assert_eq!(x.read(LINK_FENCE_BASE + 8 * 7 + 4, 4), Some(0xff));
    }

    /// A second warm reset inside 1024 core cycles is the first one's echo.
    /// The lock clears with the core only. (sn0-xbow-test
    /// lock-clears-with-core-reset-only)
    #[test]
    fn a_warm_reset_inside_the_quiet_window_is_ignored_and_the_lock_follows_the_core() {
        let (x, _) = timed(5);
        let seen = resets(&x);
        assert!(x.write(FENCE + 4, 4, 0xff & !(1 << 7)));
        assert!(x.write(LOCK + 4, 4, 0x66));
        x.warm_reset(15, 100);
        assert_eq!(x.read(LOCK + 4, 4), Some(0x66), "fenced out: the core is untouched");
        assert_eq!(seen.lock().len(), 8, "every link admitted the source");
        assert!(x.write(FENCE + 4, 4, 0xff));
        x.warm_reset(15, 100 + 10_239);
        assert_eq!(x.read(LOCK + 4, 4), Some(0x66), "still inside the window");
        assert_eq!(seen.lock().len(), 8);
        x.warm_reset(15, 100 + 10_240);
        assert_eq!(x.read(LOCK + 4, 4), Some(0));
        assert_eq!(seen.lock().len(), 16);
    }

    /// Revision 4 has no fences, so a warm reset always reaches the core and
    /// every link. (sn0-xbow-test warm-reset-revision-4-unaffected-by-fence)
    #[test]
    fn revision_four_admits_every_warm_reset() {
        let (x, _) = timed(4);
        let seen = resets(&x);
        assert!(x.write(CONTROL + 4, 4, 0x26));
        assert!(!x.write(FENCE + 4, 4, 0), "reserved on this revision");
        x.warm_reset(15, 0);
        assert_eq!(x.read(CONTROL + 4, 4), Some(0));
        assert_eq!(seen.lock().len(), 8);
    }

    #[test]
    fn link_changes_reach_the_callback_outside_the_lock() {
        let x = Arc::new(octane());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let (s, probe) = (seen.clone(), x.clone());
        x.on_link_change(Box::new(move |w, alive| {
            // The callback asks the switch about its links and registers.
            assert_eq!(probe.carries(w), alive);
            assert!(probe.read(STATUS + 4, 4).is_some());
            s.lock().push((w, alive));
        }));
        x.attach(11);
        assert_eq!(*seen.lock(), [(11, true)]);
        // A reset of a fitted live link changes nothing; of an empty port neither.
        assert!(x.write(LINK_BASE + LINK_STRIDE * 7 + LINK_RESET + 4, 4, 0));
        assert!(x.write(LINK_BASE + LINK_STRIDE * 5 + LINK_RESET + 4, 4, 0));
        assert_eq!(seen.lock().len(), 1);
        // Power-on drops nothing it fitted.
        crate::traits::Resettable::power_on(&*x);
        assert_eq!(seen.lock().len(), 1);
    }

    #[test]
    fn a_link_reset_tells_the_far_end_after_the_switch_settled() {
        let x = Arc::new(octane());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let (s, probe) = (seen.clone(), x.clone());
        x.on_link_reset(Box::new(move |w| {
            s.lock().push((w, probe.carries(w), probe.read(STATUS + 4, 4).is_some()));
        }));
        assert!(x.write(LINK_BASE + LINK_STRIDE * 7 + LINK_RESET + 4, 4, 0));
        assert_eq!(*seen.lock(), [(15, true, true)]);
    }

    /// An Octane2's Crossbow (revision code 5) has the reset fences and the
    /// resource lock its PROM programs: the fences reset to all ones and keep
    /// bits 7:0, the lock takes a value only while free and its clear frees
    /// it. Revision code 4 has neither, and an access there is an error.
    #[test]
    fn revision_five_has_reset_fences_and_a_lock() {
        let x = Xbow::new(5, 0x8100);
        for fence in 0..1 + LINKS as u32 {
            let off = FENCE + 8 * fence + 4;
            assert_eq!(x.read(off, 4), Some(0xff), "fence {fence} resets to all ones");
            assert!(x.write(off, 4, 0x1f7));
            assert_eq!(x.read(off, 4), Some(0xf7));
        }
        assert_eq!(x.read(LOCK + 4, 4), Some(0));
        assert!(x.write(LOCK + 4, 4, 3));
        assert!(x.write(LOCK + 4, 4, 5));
        assert_eq!(x.read(LOCK_CLEAR + 4, 4), Some(3), "a held lock keeps its value");
        assert!(x.write(LOCK_CLEAR + 4, 4, 0x55));
        assert_eq!(x.read(LOCK + 4, 4), Some(0));
        assert_eq!(x.read(STATUS + 4, 4).unwrap() as u32 & STATUS_ACCESS_ERROR, 0);

        let old = octane();
        assert_eq!(old.read(FENCE + 4, 4), None);
        assert_ne!(old.read(STATUS + 4, 4).unwrap() as u32 & STATUS_ACCESS_ERROR, 0);
    }

    #[test]
    fn save_load_round_trip() {
        let src = Xbow::new(5, 0x8100);
        src.regs.lock().control = 0x55;
        src.regs.lock().link_event[2] = 7;
        assert!(src.write(FENCE_BASE + 8 * 3 + 4, 4, 0x3c));
        assert!(src.write(LOCK + 4, 4, 9));
        let v = src.save_state();
        let dst = Xbow::new(5, 0x8100);
        dst.load_state(&v).expect("load_state");
        assert_eq!(v, dst.save_state());
        assert_eq!(dst.regs.lock().link_event[2], 7);
        assert_eq!(dst.read(FENCE_BASE + 8 * 3 + 4, 4), Some(0x3c));
        assert_eq!(dst.read(LOCK + 4, 4), Some(9));
        assert!(Xbow::new(5, 0x8400).load_state(&v).is_err(), "other links fitted");
        assert!(Xbow::new(4, 0x8100).load_state(&v).is_err(), "another revision");
    }
}
