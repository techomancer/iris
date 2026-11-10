//! Crossbow, the Octane's XIO crossbar: widget 0's registers, and which of
//! the eight links (widgets 8 to 15) are present and carrying traffic.
//!
//! Cut to what the Octane uses; registers follow the Crossbow ASIC
//! specification, revision 12.0 (`docs/datasheets/octane/`). The switch
//! translates no address: the machine routes a processor access by widget
//! number and asks [`Xbow::carries`] whether the link is up. Packets from
//! widget 0 itself are its interrupts, which go to the target its interrupt
//! destination registers name; on the Octane that is HEART, which the machine
//! gives as an [`XtalkHost`].
//!
//! Which links are present and alive are atomics, so the routing decision
//! (`carries`) never takes a lock; they change only under the register
//! lock, with the registers that describe them. Interrupt packets and the
//! callbacks (link change, link reset) go out after that lock is dropped.
//!
//! Revision code 5 (the Octane2's part) adds the reset fences and the
//! resource lock; the fences are kept but gate nothing, since no warm reset
//! travels through the modelled switch.
//!
//! Two approximations: a link reset on a fitted link brings the link back
//! alive as it was before the reset (link training is not modelled, and the
//! stock PROM resets each link it finds and expects it back), and register
//! accesses are 32-bit at the data half of each eight-byte slot only.
//!
//! # The XBridge's crossbar
//!
//! The IP35 boards' XBridge carries widget 0's Crossbow register map for its
//! own crossbar, built as [`Part::XBridge`] with [`Xbow::xbridge`]. It
//! differs in identification (part 0xd000, revision 2, with the fences and
//! the resource lock), Control's reset value and writable bits, the
//! performance counters' reset value and select field, the requesting
//! port in status bits 9:6, and the MicroLAN register's busy bit. It
//! interrupts for access errors alone, not for link events, and records
//! no multiple error; a malformed access answers zero rather than a bus
//! error. These follow hw/misc/sgi-xbridge-xbow.c in
//! jamesbraid/qemu-sgi-origami, not a published specification.
//!
//! # Behind a PIC
//!
//! The chip can also carry a part number and manufacturer, report a
//! revision other than the one whose behaviour it has, and name a source
//! widget in its status reads: what [`Pxbow`](super::pxbow::Pxbow) needs
//! behind a PIC.
//!
//! With a clock ([`Xbow::with_clock`]) a link reset is the two-core-cycle
//! pulse the QEMU model has: the link is down for 20 ns of the clock, and
//! the chip ends the pulse the next time anything reads its registers or
//! asks [`Xbow::carries_timed`], so what the guest sees follows the clock and not
//! host scheduling. Without one the pulse is over at once. [`Xbow::warm_reset`]
//! takes a warm reset from a link through the revision-5 fences. No board in
//! this tree sends one yet: the Hub's link control is the only sender the
//! QEMU model names.
//!
//! Ported from hw/misc/sgi-xbow.c in jamesbraid/qemu-sgi-origami.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, OnceLock, Weak};

use parking_lot::Mutex;

use super::heart::Clock;
use crate::dev::bridge::Bridge;
use crate::irq::InterruptController;
use crate::snapshot::{field_bool, field_u32, field_u32s, field_u8, StateTable};
use crate::traits::{BusDevice, BusRead16, BusRead32, BusRead64, BusRead8, BUS_ERR, BUS_OK, Saveable};
use crate::xtalk::{HostPort, Upstream, XtalkHost};

/// A widget ID register: JTAG IDCODE layout, bit 0 set.
pub const fn widget_id(rev: u32, part: u32, mfgr: u32) -> u32 {
    (rev << 28) | (part << 12) | (mfgr << 1) | 1
}

const FIRST_WIDGET: u8 = 8;
const LINKS: usize = 8;

/// The register block's span in widget 0's window.
pub const REG_SPAN: u32 = 0x0100_0000;

const IDENTIFICATION: u32 = 0x00;
const STATUS: u32 = 0x08;
const ERR_UPPER: u32 = 0x10;
const ERR_LOWER: u32 = 0x18;
const CONTROL: u32 = 0x20;
const PACKET_TIMEOUT: u32 = 0x28;
const INT_UPPER: u32 = 0x30;
const INT_LOWER: u32 = 0x38;
const ERR_CMD: u32 = 0x40;
const LLP_CONTROL: u32 = 0x48;
const STATUS_CLEAR: u32 = 0x50;
const ARB_RELOAD: u32 = 0x58;
const PERF_A: u32 = 0x60;
const PERF_B: u32 = 0x68;
const NIC: u32 = 0x70;
/// Revision code 5 and later: widget 0's reset fence, then links 8 to 15's
/// at a stride of 8, then the resource lock and its clear.
const FENCE_BASE: u32 = 0x78;
const FENCES: usize = 1 + LINKS;
const FENCE_END: u32 = FENCE_BASE + 8 * FENCES as u32;
const LOCK: u32 = 0xc0;
const LOCK_CLEAR: u32 = 0xc8;
const FENCE_MASK: u32 = 0xff;
/// The first revision code with fences and the lock (Crossbow 2.0 in the
/// software tables; an Octane2's code-5 part shows as 1.4 in its inventory).
const REVISION_FENCES: u32 = 5;
const CORE_CYCLE_NS: u64 = 10;
/// How long a link reset holds the link down: two core cycles.
const LINK_RESET_NS: u64 = 2 * CORE_CYCLE_NS;
/// A warm reset arriving this many cycles after another is ignored.
const WARM_RESET_QUIET_CYCLES: u64 = 1024;
/// Where a status read carries the widget that asked, when the chip is told
/// to.
const STATUS_SOURCE_SHIFT: u32 = 6;
const STATUS_SOURCE_MASK: u32 = 0xf << STATUS_SOURCE_SHIFT;
const LINK_BASE: u32 = 0x100;
const LINK_STRIDE: u32 = 0x40;
const LINK_END: u32 = LINK_BASE + LINK_STRIDE * LINKS as u32;
const LINK_FLUSH: u32 = 0x00;
const LINK_CONTROL: u32 = 0x08;
const LINK_STATUS: u32 = 0x10;
const LINK_ARB_UPPER: u32 = 0x18;
const LINK_ARB_LOWER: u32 = 0x20;
const LINK_STATUS_CLEAR: u32 = 0x28;
const LINK_RESET: u32 = 0x30;
const LINK_AUX_STATUS: u32 = 0x38;

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
    /// Revision code 5 and later: which ports' warm resets may reach the
    /// core (fence 0) and each link (fences 1 to 8). Kept; iris sends no
    /// warm resets through the switch.
    fences: [u32; FENCES],
    /// The resource lock's byte.
    lock: u32,
    /// When each link's reset pulse ends, in clock ns; 0 when none runs.
    reset_due: [u64; LINKS],
    /// Links that were alive when their pulse began and return alive.
    reset_restore: u8,
    /// A warm reset before this time is ignored.
    warm_quiet_until: u64,
}

/// Which crossbar a [`Xbow`] is: see the module documentation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    /// The Octane's and the Origin 2000's Crossbow.
    Crossbow,
    /// The crossbar inside the IP35 boards' XBridge.
    XBridge,
}

/// The XBridge crossbar's identification: part 0xd000, revision 2.
const XBRIDGE_ID: u32 = widget_id(2, 0xd000, 0);
const XBRIDGE_CONTROL_RESET: u32 = 0x80;
const XBRIDGE_CONTROL_MASK: u32 = 0xfe;
const XBRIDGE_PERF_SELECT_MASK: u32 = 0x0070_0000;
/// Status bits 9:6 name the port whose request this is.
const XBRIDGE_REQUESTER_SHIFT: u32 = 6;
/// The XBridge clears its widget 0 errors and request together.
const XBRIDGE_STATUS_CLEARED: u32 = STATUS_REQUEST | 0x3f;

impl Regs {
    /// `startup`: the links present, which come up alive.
    fn reset(part: Part, startup: u8) -> Self {
        let xbridge = part == Part::XBridge;
        Self {
            status: 0,
            err_upper: 0,
            err_lower: 0,
            control: if xbridge { XBRIDGE_CONTROL_RESET } else { 0 },
            packet_timeout: PACKET_TIMEOUT_MASK,
            int_upper: INT_UPPER_TARGET_HIGH,
            int_lower: 0,
            err_cmd: 0,
            llp_control: LLP_CONTROL_RESET,
            arb_reload: ARB_RELOAD_RESET,
            perf_a: if xbridge { 0 } else { PERF_RESET },
            perf_b: if xbridge { 0 } else { PERF_RESET },
            nic: NIC_RESET,
            link_control: [LINK_CREDIT_RESET; LINKS],
            link_status: [0; LINKS],
            link_arb_upper: [0; LINKS],
            link_arb_lower: [0; LINKS],
            link_aux_status: [0; LINKS],
            link_event: [0; LINKS],
            fences: [FENCE_MASK; FENCES],
            lock: 0,
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
    /// Widget 0's interrupt packets to the host: (address, vector, set).
    packets: Vec<(u64, u8, bool)>,
    /// Link resets' far-end widgets.
    link_resets: Vec<u8>,
}

pub struct Xbow {
    regs: Mutex<Regs>,
    revision: u32,
    part: Part,
    /// Revision field of the identification word when it differs from the
    /// revision that selects behaviour.
    reported_revision: Option<u32>,
    /// Part number and manufacturer in a Crossbow's identification word.
    identity: (u32, u32),
    /// The widget a status read reports as its source, if the chip does.
    status_source: Option<u8>,
    clock: Option<Clock>,
    /// The earliest running pulse's end, or `u64::MAX`. Written under `regs`.
    reset_due: AtomicU64,
    /// A write to an error address register completes; see
    /// [`Xbow::with_posted_error_address_writes`].
    posted_error_address_writes: bool,
    /// Links with something attached, bit n for widget 8 + n.
    present: AtomicU8,
    /// Links passing traffic, as `present`. Written under `regs`.
    alive: AtomicU8,
    /// Held across "decide a packet, send it", so two threads' packets for
    /// widget 0's one interrupt arrive in the order they were decided.
    /// Never held with `regs`, and nothing but HEART is called under it.
    irq_order: Mutex<()>,
    /// The host chip widget 0's interrupt packets go to when they target
    /// it: how it is addressed, and its interrupt inputs.
    host: OnceLock<(XtalkHost, Arc<dyn InterruptController>)>,
    /// Told when a link comes up or goes down, with the widget and its new
    /// state, outside the register lock.
    on_link_change: OnceLock<Box<dyn Fn(u8, bool) + Send + Sync>>,
    /// Told of a link reset, with the widget, once the switch's own state
    /// has settled: the widget at the far end resets too.
    on_link_reset: OnceLock<Box<dyn Fn(u8) + Send + Sync>>,
    /// The Bridges behind ports ([`Self::serve_bridge`]), reset with their
    /// port's link.
    served: Mutex<Vec<(u8, Weak<Bridge>)>>,
    unsupported: Mutex<Vec<(u32, u8, bool)>>,
    /// `unsupported` holds 64: a further access costs one relaxed load.
    unsupported_full: AtomicBool,
}

fn link_bit(widget: u8) -> Option<u8> {
    (FIRST_WIDGET..FIRST_WIDGET + LINKS as u8).contains(&widget).then(|| 1 << (widget - FIRST_WIDGET))
}

impl Xbow {
    /// A Crossbow of `revision` with the links in `widgets` (a widget mask,
    /// bit n for widget n) attached and alive at power-on.
    pub fn new(revision: u32, widgets: u16) -> Self {
        Self::with_part(Part::Crossbow, revision, widgets)
    }

    /// The XBridge's crossbar with the links in `widgets` attached and
    /// alive at power-on.
    pub fn xbridge(widgets: u16) -> Self {
        Self::with_part(Part::XBridge, 2, widgets)
    }

    fn with_part(part: Part, revision: u32, widgets: u16) -> Self {
        let present = (widgets >> FIRST_WIDGET) as u8;
        Self {
            regs: Mutex::new(Regs::reset(part, present)),
            revision,
            part,
            reported_revision: None,
            identity: (0, 0),
            status_source: None,
            clock: None,
            reset_due: AtomicU64::new(u64::MAX),
            posted_error_address_writes: false,
            present: AtomicU8::new(present),
            alive: AtomicU8::new(present),
            irq_order: Mutex::new(()),
            host: OnceLock::new(),
            on_link_change: OnceLock::new(),
            on_link_reset: OnceLock::new(),
            served: Mutex::new(Vec::new()),
            unsupported: Mutex::new(Vec::new()),
            unsupported_full: AtomicBool::new(false),
        }
    }

    /// Complete a write to either error address register instead of
    /// refusing it. The registers are read-only and the write still changes
    /// nothing, sets the register-access error and is listed in
    /// [`Xbow::unsupported`]: the specification makes a write to a read-only
    /// register an error condition that widget 0's status reports. What the
    /// processor sees is the other half: an uncached store is a write
    /// request with no completion response (NEC U10278EJ4V0UM p.104, table
    /// 6-1), so no error can come back to it. A stock IP35 PROM writes the
    /// lower register on purpose in its crossbar check. QEMU's
    /// `complete-posted-error-address-writes` property.
    pub fn with_posted_error_address_writes(mut self) -> Self {
        self.posted_error_address_writes = true;
        self
    }

    /// Answer to `part` and `manufacturer` in the identification word
    /// instead of the Crossbow's zeros.
    pub fn with_identity(mut self, part: u32, manufacturer: u32) -> Self {
        self.identity = (part, manufacturer);
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

    /// Where widget 0's interrupt packets go: the host chip addressed as
    /// `host`, whose line is the vector the packet names. Once, before the
    /// machine runs.
    pub fn connect_host(&self, host: XtalkHost, irq: Arc<dyn InterruptController>) {
        let _ = self.host.set((host, irq));
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

    /// Serve the Bridge behind port `widget`: it sends only to widgets with a
    /// live link, and a reset of that link resets it. Every machine that puts
    /// a Bridge behind a Crossbow calls this once for each Bridge.
    pub fn serve_bridge(self: &Arc<Self>, widget: u8, bridge: &Arc<Bridge>) {
        let links = self.clone();
        bridge.set_link_alive(Box::new(move |w| links.carries(w)));
        self.served.lock().push((widget, Arc::downgrade(bridge)));
    }

    pub fn register_locks(self: &Arc<Self>) {
        use crate::locks::register_mutex_fn;
        register_mutex_fn("xbow::regs", self, |x| x.regs.is_locked());
        register_mutex_fn("xbow::irq_order", self, |x| x.irq_order.is_locked());
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
        if let Some((_, h)) = self.host.get() {
            for (address, vector, set) in out.packets {
                h.packet(0, address, vector as u32, set);
            }
        }
        drop(order);
        let changed = before ^ self.alive.load(Ordering::Acquire);
        if let Some(cb) = self.on_link_change.get() {
            for link in (0..LINKS as u8).filter(|l| changed >> l & 1 != 0) {
                cb(FIRST_WIDGET + link, self.alive.load(Ordering::Acquire) >> link & 1 != 0);
            }
        }
        for w in out.link_resets {
            let served: Vec<_> = self.served.lock().iter().filter(|(p, _)| *p == w).filter_map(|(_, b)| b.upgrade()).collect();
            for b in served {
                b.link_reset();
            }
            if let Some(cb) = self.on_link_reset.get() {
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

    /// Whether the switch passes traffic to and from `widget`. A link reset
    /// pulse ends here only on a register access; a chip with a clock asks
    /// [`Self::carries_timed`].
    pub fn carries(&self, widget: u8) -> bool {
        link_bit(widget).is_some_and(|b| self.alive.load(Ordering::Acquire) & b != 0)
    }

    /// [`Self::carries`] on a chip with a clock: a link reset pulse that is
    /// over by the clock ends first. Lock-free unless one is.
    pub fn carries_timed(&self, widget: u8) -> bool {
        if let Some(clock) = &self.clock {
            if clock() >= self.reset_due.load(Ordering::Acquire) {
                self.locked(|_, _| {});
            }
        }
        self.carries(widget)
    }

    pub fn unsupported(&self) -> Vec<(u32, u8, bool)> {
        self.unsupported.lock().clone()
    }

    fn identification(&self) -> u32 {
        match self.part {
            // Part number and manufacturer are zero on Crossbow unless the
            // board names others.
            Part::Crossbow => {
                let (part, manufacturer) = self.identity;
                (self.reported_revision.unwrap_or(self.revision) << 28) | (part << 12) | (manufacturer << 1) | 1
            }
            Part::XBridge => XBRIDGE_ID,
        }
    }

    /// The status register as the host reads it: on the XBridge, with the
    /// host's port as the requester; on a chip told its source, with that.
    fn status(&self, r: &Regs) -> u32 {
        match (self.part, self.host.get(), self.status_source) {
            (Part::XBridge, Some((host, _)), _) => r.status | u32::from(host.widget) << XBRIDGE_REQUESTER_SHIFT,
            (Part::Crossbow, _, Some(widget)) => (r.status & !STATUS_SOURCE_MASK) | (u32::from(widget) << STATUS_SOURCE_SHIFT),
            _ => r.status,
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
    /// but the host chip goes nowhere on this machine.
    fn update_interrupt(&self, r: &mut Regs, out: &mut Out) {
        // The XBridge's crossbar interrupts for its own errors alone.
        for link in (0..LINKS).filter(|_| self.part == Part::Crossbow) {
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
        match self.host.get() {
            Some((host, _)) if target == host.widget && self.carries(target) => {}
            _ => return,
        }
        let address = (((r.int_upper & 0xffff) as u64) << 32) | r.int_lower as u64;
        out.packets.push((address, vector, pending));
        r.interrupt_asserted = pending;
    }

    fn record_access_error(&self, r: &mut Regs, out: &mut Out) {
        if r.status & STATUS_ACCESS_ERROR != 0 && self.part == Part::Crossbow {
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
        let fences = self.revision >= REVISION_FENCES || self.part == Part::XBridge;
        if fences && ((FENCE_BASE..FENCE_END).contains(&slot) || slot == LOCK || slot == LOCK_CLEAR) {
            return slot % 8 == 0;
        }
        match slot {
            IDENTIFICATION | STATUS | ERR_UPPER | ERR_LOWER | STATUS_CLEAR => !write,
            CONTROL | PACKET_TIMEOUT | INT_UPPER | INT_LOWER | ERR_CMD | LLP_CONTROL | ARB_RELOAD | PERF_A
            | PERF_B | NIC => true,
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
    /// other malformed access is an access error. The XBridge's crossbar
    /// takes an eight-byte access as an access error too.
    fn decode(&self, r: &mut Regs, out: &mut Out, off: u32, size: usize, write: bool) -> Option<u32> {
        if size == 8 && self.part == Part::Crossbow {
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
    /// The XBridge's crossbar answers a malformed read with zero.
    pub fn read(&self, off: u32, size: usize) -> Option<u64> {
        self.locked(|r, out| {
            let Some(slot) = self.decode(r, out, off, size, false) else {
                return (self.part == Part::XBridge).then_some(0);
            };
            let v = match slot {
                IDENTIFICATION => self.identification(),
                STATUS => self.status(r),
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
                    let v = self.status(r);
                    r.status &= !match self.part {
                        Part::Crossbow => STATUS_WIDGET0_ERRORS | STATUS_REQUEST,
                        Part::XBridge => XBRIDGE_STATUS_CLEARED,
                    };
                    self.update_interrupt(r, out);
                    v
                }
                ARB_RELOAD => r.arb_reload,
                PERF_A => r.perf_a,
                PERF_B => r.perf_b,
                NIC => r.nic,
                s if (FENCE_BASE..FENCE_END).contains(&s) => r.fences[((s - FENCE_BASE) / 8) as usize],
                LOCK | LOCK_CLEAR => r.lock,
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

    /// A processor write; false is a bus error. The XBridge's crossbar
    /// takes a malformed write and drops it.
    pub fn write(&self, off: u32, size: usize, val: u64) -> bool {
        let xbridge = self.part == Part::XBridge;
        self.locked(|r, out| {
            let Some(slot) = self.decode(r, out, off, size, true) else {
                // Refused and recorded either way; a posted error address
                // write completes all the same.
                return xbridge
                    || (self.posted_error_address_writes
                        && size == 4
                        && matches!(off.wrapping_sub(4), ERR_UPPER | ERR_LOWER));
            };
            let val = val as u32;
            match slot {
                CONTROL => {
                    r.control = val & if xbridge { XBRIDGE_CONTROL_MASK } else { CONTROL_MASK };
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
                PERF_A | PERF_B => {
                    let v = val & if xbridge { XBRIDGE_PERF_SELECT_MASK } else { PERF_SELECT_MASK };
                    *if slot == PERF_A { &mut r.perf_a } else { &mut r.perf_b } = v;
                }
                // The XBridge's one-wire master is not modelled: it reports
                // itself done (bit 1) at once.
                NIC if xbridge => r.nic = (val & NIC_MASK & !3) | NIC_RESET,
                NIC => r.nic = val & NIC_MASK,
                s if (FENCE_BASE..FENCE_END).contains(&s) => r.fences[((s - FENCE_BASE) / 8) as usize] = val & FENCE_MASK,
                // Test-and-set: stored only while the lock is free.
                LOCK => {
                    if r.lock == 0 {
                        r.lock = val & 0xff;
                    }
                }
                LOCK_CLEAR => r.lock = 0,
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

    /// A link reset. Without a clock the pulse is over before the
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
        let fenced = self.revision >= REVISION_FENCES || self.part == Part::XBridge;
        self.locked(|r, out| {
            if now < r.warm_quiet_until {
                crate::dlog!(crate::devlog::LogModule::Bedrock, "xbow: warm reset from widget {source} inside the quiet window; ignored");
                return;
            }
            let fences = r.fences;
            let admits = |fence: u32| !fenced || fence >> port & 1 != 0;
            if admits(fences[0]) {
                let present = self.present.load(Ordering::Acquire);
                *r = Regs::reset(self.part, present);
                self.alive.store(present, Ordering::Release);
                r.fences = fences;
                self.update_interrupt(r, out);
            }
            r.warm_quiet_until = now + WARM_RESET_QUIET_CYCLES * CORE_CYCLE_NS;
            for (link, &fence) in fences[1..].iter().enumerate() {
                if admits(fence) {
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
            *r = Regs::reset(self.part, present);
            self.alive.store(present, Ordering::Release);
        });
    }
}

/// Widget 0's window as the machine's bus sees it: the full physical
/// address, of which the window's 16 MB is decoded.
const WINDOW_MASK: u64 = REG_SPAN as u64 - 1;

/// The widget at one of the Crossbow's ports, as a processor reaches it: the
/// widget answers while the link is up and the access fails while it is not.
/// The Crossbow routes a request by widget number and translates no address,
/// so the window is the widget's own.
struct Port {
    xbow: Arc<Xbow>,
    widget: u8,
    dev: Arc<dyn BusDevice>,
}

/// A packet into a peer's I/O view: the widget it names in bits 51:48 gets it,
/// with those bits cleared, while the link to that widget is up.
struct Packets {
    xbow: Arc<Xbow>,
    ports: Vec<(u8, Arc<dyn BusDevice>)>,
}

/// An access that a Crossbow passes on to a device, or fails.
macro_rules! routed_access {
    ($r:ident, $w:ident, $t:ty, $out:ident) => {
        fn $r(&self, a: u64) -> $out {
            self.route(a).map_or($out::err(), |(d, at)| d.$r(at))
        }
        fn $w(&self, a: u64, v: $t) -> u32 {
            self.route(a).map_or(BUS_ERR, |(d, at)| d.$w(at, v))
        }
    };
}

impl Port {
    fn route(&self, a: u64) -> Option<(&dyn BusDevice, u64)> {
        self.xbow.carries(self.widget).then(|| (&*self.dev, a))
    }
}
impl BusDevice for Port {
    routed_access!(read8, write8, u8, BusRead8);
    routed_access!(read16, write16, u16, BusRead16);
    routed_access!(read32, write32, u32, BusRead32);
    routed_access!(read64, write64, u64, BusRead64);
}

impl Packets {
    fn route(&self, a: u64) -> Option<(&dyn BusDevice, u64)> {
        let widget = ((a >> 48) & 0xf) as u8;
        let (_, dev) = self.ports.iter().find(|(w, _)| *w == widget)?;
        self.xbow.carries(widget).then(|| (&**dev, a & ((1 << 48) - 1)))
    }
}
impl BusDevice for Packets {
    routed_access!(read8, write8, u8, BusRead8);
    routed_access!(read16, write16, u16, BusRead16);
    routed_access!(read32, write32, u32, BusRead32);
    routed_access!(read64, write64, u64, BusRead64);
}

impl Xbow {
    /// What a Bridge behind this Crossbow sends to: each of `hosts` sits on
    /// its widget's port, and a request to it is dropped while that link is
    /// down.
    pub fn upstream(self: &Arc<Self>, hosts: Vec<HostPort>) -> Upstream {
        hosts.into_iter().fold(Upstream::new(), |up, h| {
            let (xbow, widget) = (self.clone(), h.host.widget);
            assert!(link_bit(widget).is_some(), "widget {widget} is not a Crossbow port");
            up.with(h.with_link(Arc::new(move || xbow.carries(widget))))
        })
    }

    /// `dev` as the widget behind port `widget` (8 to 15).
    pub fn port(self: &Arc<Self>, widget: u8, dev: Arc<dyn BusDevice>) -> Arc<dyn BusDevice> {
        assert!(link_bit(widget).is_some(), "widget {widget} is not a Crossbow port");
        Arc::new(Port { xbow: self.clone(), widget, dev })
    }

    /// The packet face a Hub's big window reaches a peer's I/O view through:
    /// each of `ports` is a port widget and the device a packet to it lands on.
    pub fn packets(self: &Arc<Self>, ports: Vec<(u8, Arc<dyn BusDevice>)>) -> Arc<dyn BusDevice> {
        Arc::new(Packets { xbow: self.clone(), ports })
    }
}

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
            .u32s("fences", &r.fences)
            .u32("lock", r.lock)
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
        let n = Regs {
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
            fences: field_u32s(v, "fences")?,
            lock: field_u32(v, "lock")?,
            // A link reset lasts two core cycles and the warm-reset quiet
            // time is counted on the board clock; a snapshot carries
            // neither, so a restored crossbow has none in progress.
            reset_due: [0; LINKS],
            reset_restore: 0,
            warm_quiet_until: 0,
        };
        *self.regs.lock() = n;
        self.reset_due.store(u64::MAX, Ordering::Release);
        self.alive.store(field_u8(v, "alive")?, Ordering::Release);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ip30::HEART_HOST;
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
        x.connect_host(HEART_HOST, isr.clone());
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

    /// Widget 0's interrupt goes to the host the machine named, not to a
    /// fixed widget number.
    #[test]
    fn interrupts_follow_the_host_the_machine_names() {
        let x = octane();
        let isr = Arc::new(Isr::default());
        x.connect_host(XtalkHost { widget: 15, ..HEART_HOST }, isr.clone());
        let ctl = LINK_BASE + LINK_STRIDE * 7 + LINK_CONTROL + 4;
        let reset = LINK_BASE + LINK_STRIDE * 7 + LINK_RESET + 4;
        // Addressed to widget 8, which is not the host here.
        assert!(x.write(INT_UPPER + 4, 4, (0x2a << 24) | (8 << 16)));
        assert!(x.write(ctl, 4, LINK_ALIVE_ENABLE as u64));
        assert!(x.write(reset, 4, 0));
        assert_eq!(isr.0.load(Ordering::SeqCst), 0);
        // Addressed to widget 15, which is.
        x.read(LINK_BASE + LINK_STRIDE * 7 + LINK_STATUS_CLEAR + 4, 4).unwrap();
        assert!(x.write(INT_UPPER + 4, 4, (0x2b << 24) | (15 << 16)));
        assert!(x.write(reset, 4, 0));
        assert_eq!(isr.0.load(Ordering::SeqCst), 1 << 0x2b);
    }

    /// A host that decodes the packet itself gets the destination address
    /// the interrupt registers name, from widget 0.
    #[test]
    fn a_host_that_decodes_packets_sees_their_address() {
        #[derive(Default)]
        struct Decoder(Mutex<Vec<(u8, u64, u32, bool)>>);
        impl InterruptController for Decoder {
            fn set_line(&self, _: u32, _: bool) {
                unreachable!("a packet host takes packets");
            }
            fn packet(&self, source: u8, address: u64, line: u32, level: bool) -> bool {
                self.0.lock().push((source, address, line, level));
                true
            }
        }
        let x = octane();
        let decoder = Arc::new(Decoder::default());
        x.connect_host(HEART_HOST, decoder.clone());
        assert!(x.write(INT_UPPER + 4, 4, (0x2a << 24) | (8 << 16) | 0x12));
        assert!(x.write(INT_LOWER + 4, 4, 0x0180_0090));
        assert!(x.write(CONTROL + 4, 4, STATUS_ACCESS_ERROR as u64));
        assert_eq!(x.read(0x00, 4), None);
        x.read(STATUS_CLEAR + 4, 4).unwrap();
        assert_eq!(*decoder.0.lock(), [(0, 0x12_0180_0090, 0x2a, true), (0, 0x12_0180_0090, 0x2a, false)]);
    }

    /// The lock registry outlives a machine's tests; it must not keep the
    /// Crossbow alive.
    #[test]
    fn registering_the_locks_does_not_keep_the_crossbow_alive() {
        let x = Arc::new(octane());
        x.register_locks();
        let weak = Arc::downgrade(&x);
        drop(x);
        assert!(weak.upgrade().is_none());
    }

    // The XBridge's crossbar. Cases follow tests/qtest/sn1-xbridge-xbow-test.c
    // (reset-values, lock, fences) and sn1-xbridge-test.c (access-errors) in
    // jamesbraid/qemu-sgi-origami.

    /// The Fuel's: Bedrock at 10 and the two PCI heads at 14 and 15.
    fn xbridge() -> Xbow {
        Xbow::xbridge(1 << 10 | 1 << 14 | 1 << 15)
    }

    fn get(x: &Xbow, slot: u32) -> u32 {
        x.read(slot + 4, 4).unwrap() as u32
    }

    /// Software probing the crossbar after reset sees these values. Each
    /// link's control word resets to 0x80000; that value belongs in the
    /// control word, not in the link reset register, which no read can see.
    #[test]
    fn the_xbridge_crossbar_resets_to_its_own_values() {
        let x = xbridge();
        for (slot, value) in [
            (IDENTIFICATION, 0x2d00_0001),
            (ERR_UPPER, 0),
            (ERR_LOWER, 0),
            (CONTROL, 0x80),
            (PACKET_TIMEOUT, 0xf_ffff),
            (INT_UPPER, 0x8_0000),
            (INT_LOWER, 0),
            (ERR_CMD, 0),
            (LLP_CONTROL, 0x03ff_1810),
            (ARB_RELOAD, 2),
            (PERF_A, 0),
            (PERF_B, 0),
            (NIC, 2),
            (FENCE_BASE, 0xff),
            (LOCK, 0),
        ] {
            assert_eq!(get(&x, slot), value, "{slot:#x}");
        }
        for link in 0..LINKS as u32 {
            let base = LINK_BASE + link * LINK_STRIDE;
            assert_eq!(get(&x, FENCE_BASE + 8 + link * 8), 0xff);
            assert_eq!(get(&x, base + LINK_CONTROL), 0x8_0000, "link {}", link + 8);
            assert_eq!(get(&x, base + LINK_ARB_UPPER), 0);
            assert_eq!(get(&x, base + LINK_ARB_LOWER), 0);
        }
        assert!(x.carries(10) && x.carries(14) && x.carries(15) && !x.carries(13));
    }

    /// A write takes the resource lock only while it is free, and every
    /// reset fence bit is read and write in the low eight bits.
    #[test]
    fn the_xbridge_crossbar_has_the_lock_and_the_fences() {
        let x = xbridge();
        assert!(x.write(LOCK + 4, 4, 0x5a));
        assert!(x.write(LOCK + 4, 4, 0x3c));
        assert_eq!((get(&x, LOCK), get(&x, LOCK_CLEAR)), (0x5a, 0x5a));
        assert!(x.write(LOCK_CLEAR + 4, 4, 0));
        assert!(x.write(LOCK + 4, 4, 0x3c));
        assert_eq!(get(&x, LOCK), 0x3c);
        for slot in (FENCE_BASE..FENCE_END).step_by(8) {
            assert!(x.write(slot + 4, 4, 0));
            assert_eq!(get(&x, slot), 0);
            assert!(x.write(slot + 4, 4, u64::from(u32::MAX)));
            assert_eq!(get(&x, slot), 0xff);
        }
    }

    /// A malformed access answers zero and latches the access error, which
    /// interrupts once while enabled; status carries the host's port in
    /// bits 9:6, and reading the clear alias drops the interrupt. Link
    /// events do not interrupt.
    #[test]
    fn the_xbridge_crossbar_reports_its_first_error_once() {
        let x = xbridge();
        let isr = Arc::new(Isr::default());
        x.connect_host(XtalkHost { widget: 10, int_packet: 0, int_packet_mask: 0, vectors: 256 }, isr.clone());
        assert!(x.write(INT_UPPER + 4, 4, 0x150a_0000));
        assert!(x.write(LINK_BASE + LINK_STRIDE * 7 + LINK_CONTROL + 4, 4, LINK_ALIVE_ENABLE as u64));
        assert!(x.write(LINK_BASE + LINK_STRIDE * 7 + LINK_RESET + 4, 4, 0));
        assert_eq!(isr.0.load(Ordering::SeqCst), 0, "a link event does not interrupt");
        assert!(x.write(CONTROL + 4, 4, u64::from(STATUS_ACCESS_ERROR)));
        assert_eq!(x.read(IDENTIFICATION, 8), Some(0));
        assert!(x.write(IDENTIFICATION + 4, 4, 1));
        assert_eq!(isr.0.load(Ordering::SeqCst), 1 << 0x15);
        assert_eq!(get(&x, STATUS), STATUS_REQUEST | STATUS_ACCESS_ERROR | 10 << 6);
        assert_eq!(get(&x, STATUS_CLEAR) & (STATUS_REQUEST | STATUS_ACCESS_ERROR), STATUS_REQUEST | STATUS_ACCESS_ERROR);
        assert_eq!(isr.0.load(Ordering::SeqCst), 0);
        assert_eq!(get(&x, STATUS), 10 << 6);
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

    /// A Crossbow built without the posted-write option, as the Octane and
    /// any other standalone one is, refuses both error address registers and
    /// still records the access error; only the PXBOW completes them.
    #[test]
    fn a_standalone_crossbow_refuses_both_error_address_writes() {
        for revision in [4, 5] {
            let x = Xbow::new(revision, 0x8400);
            assert_eq!(x.write32(ERR_LOWER as u64 + 4, 0xdead_beef), BUS_ERR, "revision {revision}");
            assert_eq!(x.write32(ERR_UPPER as u64 + 4, 0), BUS_ERR, "revision {revision}");
            assert_eq!(x.read(STATUS + 4, 4).unwrap() as u32 & STATUS_ACCESS_ERROR, STATUS_ACCESS_ERROR);
        }
    }

    /// The fences, the lock and a non-zero part number belong to revision 5
    /// and later; the Octane's revision 4 leaves them reserved.
    #[test]
    fn fences_the_lock_and_the_identity_fields_need_the_revision_that_has_them() {
        let x = octane();
        for slot in [FENCE_BASE, (FENCE_BASE + 8), LOCK, LOCK_CLEAR] {
            assert_eq!(x.read(slot + 4, 4), None, "slot {slot:#x} is reserved on revision 4");
            assert!(!x.write(slot + 4, 4, 1));
        }
        assert_eq!(x.read(IDENTIFICATION + 4, 4), Some(0x4000_0001));
        let y = Xbow::new(5, 0x8400).with_identity(0xd100, 0);
        assert_eq!(y.read(IDENTIFICATION + 4, 4), Some(0x5d10_0001));
        assert_eq!(y.read(FENCE_BASE + 4, 4), Some(0xff));
        assert!(y.write(FENCE_BASE + 4, 4, 0x1ff));
        assert_eq!(y.read(FENCE_BASE + 4, 4), Some(0xff), "a fence is eight bits");
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
        assert!(!x.carries_timed(15));
        assert_eq!(x.read(status(7), 4), Some(0), "initializing shows none of the three bits");
        assert_eq!(x.read(LINK_BASE + LINK_STRIDE * 7 + LINK_AUX_STATUS + 4, 4), Some(AUX_PRESENT as u64));
        now.store(1019, Ordering::SeqCst);
        assert!(!x.carries_timed(15));
        now.store(1020, Ordering::SeqCst);
        assert!(x.carries_timed(15), "the pulse is over by the clock without any register access");
        assert_eq!(x.read(status(7), 4).unwrap() as u32 & LINK_ALIVE, LINK_ALIVE);
        assert!(!x.carries_timed(11), "an empty port stays down");
        assert_eq!(x.read(status(3), 4), Some(LINK_MAX_RETRY as u64));
        assert!(x.carries_timed(8), "other links were not touched");
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
        assert!(x.write(FENCE_BASE + 4, 4, 0xbf));
        assert!(x.write((FENCE_BASE + 8) + 8 * 7 + 4, 4, 0x3c));
        assert!(x.write((FENCE_BASE + 8) + 4, 4, 0xfd));
        x.warm_reset(15, 0);
        assert_eq!(x.read(CONTROL + 4, 4), Some(0), "the core admitted widget 15 and reset");
        assert_eq!(x.read(FENCE_BASE + 4, 4), Some(0xbf));
        assert_eq!(x.read((FENCE_BASE + 8) + 8 * 7 + 4, 4), Some(0x3c));
        // Link 7's fence refuses port 15 (bit 7 clear); the rest admit it.
        assert_eq!(*seen.lock(), (8..15).collect::<Vec<u8>>());
        // A core that refuses the source is left alone while links still admit.
        assert!(x.write(CONTROL + 4, 4, 0x26));
        assert!(x.write(FENCE_BASE + 4, 4, 0xbf & !(1 << 7)));
        now.store(20_000, Ordering::SeqCst);
        x.warm_reset(15, 20_000);
        assert_eq!(x.read(CONTROL + 4, 4), Some(0x26));
        crate::traits::Resettable::power_on(&x);
        assert_eq!(x.read(FENCE_BASE + 4, 4), Some(0xff));
        assert_eq!(x.read((FENCE_BASE + 8) + 8 * 7 + 4, 4), Some(0xff));
    }

    /// A second warm reset inside 1024 core cycles is the first one's echo.
    /// The lock clears with the core only. (sn0-xbow-test
    /// lock-clears-with-core-reset-only)
    #[test]
    fn a_warm_reset_inside_the_quiet_window_is_ignored_and_the_lock_follows_the_core() {
        let (x, _) = timed(5);
        let seen = resets(&x);
        assert!(x.write(FENCE_BASE + 4, 4, 0xff & !(1 << 7)));
        assert!(x.write(LOCK + 4, 4, 0x66));
        x.warm_reset(15, 100);
        assert_eq!(x.read(LOCK + 4, 4), Some(0x66), "fenced out: the core is untouched");
        assert_eq!(seen.lock().len(), 8, "every link admitted the source");
        assert!(x.write(FENCE_BASE + 4, 4, 0xff));
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
        assert!(!x.write(FENCE_BASE + 4, 4, 0), "reserved on this revision");
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

    /// A link reset of the Bridge's port resets the Bridge, and a reset of
    /// another port leaves it alone.
    #[test]
    fn a_link_reset_resets_the_bridge_behind_that_port_only() {
        use crate::dev::bridge::reg;
        let x = Arc::new(octane());
        let bridge = Bridge::new(Upstream::new(), &crate::ip30::PCI0);
        x.serve_bridge(15, &bridge);
        let enable = |v: u32| assert_eq!(BusDevice::write32(&*bridge, u64::from(reg::INT_ENABLE + 4), v), BUS_OK);
        enable(0x44);
        assert_eq!(bridge.register(reg::INT_ENABLE), 0x44);
        assert!(x.write(LINK_BASE + LINK_STRIDE * 6 + LINK_RESET + 4, 4, 0));
        assert_eq!(bridge.register(reg::INT_ENABLE), 0x44, "port 14 is not the Bridge's");
        assert!(x.write(LINK_BASE + LINK_STRIDE * 7 + LINK_RESET + 4, 4, 0));
        assert_eq!(bridge.register(reg::INT_ENABLE), 0, "port 15 is");
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
        for fence in 0..FENCES as u32 {
            let off = FENCE_BASE + 8 * fence + 4;
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
        assert_eq!(old.read(FENCE_BASE + 4, 4), None);
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

    /// A widget that answers every read with a marker.
    struct Marker;
    impl BusDevice for Marker {
        fn read32(&self, a: u64) -> BusRead32 {
            BusRead32::ok(0x1234_0000 | a as u32)
        }
        fn write32(&self, _: u64, _: u32) -> u32 {
            BUS_OK
        }
    }

    #[test]
    fn a_port_passes_accesses_while_its_link_is_up_and_fails_them_otherwise() {
        let x = Arc::new(octane());
        let fitted = x.port(15, Arc::new(Marker));
        assert_eq!(fitted.read32(0x10).data, 0x1234_0010);
        assert_eq!(fitted.write32(0x10, 0), BUS_OK);
        // Widget 11 has nothing attached, so its link is down.
        let empty = x.port(11, Arc::new(Marker));
        assert_eq!(empty.read32(0x10).status, BUS_ERR);
        assert_eq!(empty.write32(0x10, 0), BUS_ERR);
    }

    #[test]
    #[should_panic(expected = "not a Crossbow port")]
    fn only_the_eight_link_widgets_are_ports() {
        let _ = Arc::new(octane()).port(7, Arc::new(Marker));
    }

    #[test]
    fn a_packet_goes_to_the_widget_it_names_while_that_link_is_up() {
        let x = Arc::new(octane());
        let packets = x.packets(vec![(15, Arc::new(Marker)), (11, Arc::new(Marker))]);
        // The widget is cleared from the address the device sees.
        assert_eq!(packets.read32(15u64 << 48 | 0x10).data, 0x1234_0010);
        // A named port with no link, and a widget with no port, fail.
        assert_eq!(packets.read32(11u64 << 48 | 0x10).status, BUS_ERR);
        assert_eq!(packets.read32(9u64 << 48 | 0x10).status, BUS_ERR);
    }
}
