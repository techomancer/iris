//! Perfetto tracing (`--features perfetto`).
//!
//! Instrumentation points emit fixed-size records into a per-thread
//! single-producer ring; a writer thread drains the rings, encodes Perfetto
//! protobuf (`proto.rs`) and streams it to a file that loads straight into
//! <https://ui.perfetto.dev>. The hot path never encodes, allocates or takes
//! a lock: one relaxed load to test the category mask, then two clock reads
//! and a 32-byte store
//! (~0.5 ns filtered out, ~20 ns recorded; `tests::emit_cost`). Without the feature the macros expand to nothing.
//!
//! Every event is described by a `static EventDef` (name, category, track,
//! optional argument name). The catalogue lives in `events.rs`, so the set of
//! tracks and what lands on them can be read in one place:
//!
//! ```ignore
//! // events.rs
//! pub static VSYNC: EventDef = EventDef::new("vsync", cat::GFX).on_track("Display");
//! // at the call site
//! crate::pf_instant!(crate::perfetto::events::VSYNC);
//! crate::pf_scope!(crate::perfetto::events::JIT_COMPILE, page_pa);
//! ```
//!
//! Each record carries two timestamps: host time (raw TSC ticks, mapped by
//! the writer onto CLOCK_BOOTTIME on Linux so a trace lines up with a
//! system-wide Perfetto capture) and the guest's retired-instruction count. A session picks which one becomes the
//! timeline (`Clock`); the guest timeline shows one instruction as one
//! nanosecond, so "1 ms" on screen means a million guest instructions.
//!
//! Control: `perfetto start|stop|status|cats` on the monitor, or
//! `--perfetto FILE` at startup. See `docs/perfetto.md`.

pub mod proto;
pub mod events;
mod writer;

use std::cell::{Cell, UnsafeCell};
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use crossbeam_utils::CachePadded;
use parking_lot::Mutex;

/// Event categories: bits in the session mask, so tracing can be narrowed to
/// the subsystems being looked at without rebuilding.
pub mod cat {
    pub const CPU: u32 = 1 << 0;
    pub const IRQ: u32 = 1 << 1;
    pub const GFX: u32 = 1 << 2;
    pub const SCSI: u32 = 1 << 3;
    pub const NET: u32 = 1 << 4;
    pub const DMA: u32 = 1 << 5;
    pub const JIT: u32 = 1 << 6;
    pub const TIMER: u32 = 1 << 7;
    pub const AUDIO: u32 = 1 << 8;
    pub const MISC: u32 = 1 << 9;
    /// Per-primitive graphics events (a REX3 GO): up to millions a second,
    /// so separate from `gfx`.
    pub const PRIM: u32 = 1 << 10;

    pub const ALL: u32 = (1 << 11) - 1;

    pub const NAMES: &[(&str, u32)] = &[
        ("cpu", CPU), ("irq", IRQ), ("gfx", GFX), ("scsi", SCSI), ("net", NET),
        ("dma", DMA), ("jit", JIT), ("timer", TIMER), ("audio", AUDIO), ("misc", MISC),
        ("prim", PRIM),
    ];

    /// `"irq,gfx"`, `"all"`, or `"all,-jit"`.
    pub fn parse(s: &str) -> Result<u32, String> {
        let mut mask = 0;
        for tok in s.split(',').map(str::trim).filter(|t| !t.is_empty()) {
            let (neg, name) = match tok.strip_prefix('-') { Some(n) => (true, n), None => (false, tok) };
            let bits = if name == "all" {
                ALL
            } else {
                NAMES.iter().find(|(n, _)| *n == name).map(|(_, b)| *b)
                    .ok_or_else(|| format!("unknown perfetto category '{name}' (have: all, {})",
                        NAMES.iter().map(|(n, _)| *n).collect::<Vec<_>>().join(", ")))?
            };
            if neg { mask &= !bits } else { mask |= bits }
        }
        Ok(mask)
    }

    pub fn format(mask: u32) -> String {
        if mask == ALL { return "all".into(); }
        let v: Vec<&str> = NAMES.iter().filter(|(_, b)| mask & b != 0).map(|(n, _)| *n).collect();
        if v.is_empty() { "none".into() } else { v.join(",") }
    }
}

/// Where an event is drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Track {
    /// The emitting thread's own track.
    Thread,
    /// A named track under the iris process, shared by every thread: for
    /// things that are not "work done by this thread" (an interrupt line, a
    /// DMA channel), and for slices that begin on one thread and end on
    /// another.
    Global(&'static str),
}

/// One instrumentation point. Declare as a `static` (normally in
/// `events.rs`); records refer to it by address, so it is interned once per
/// trace and costs a pointer per event.
pub struct EventDef {
    pub name: &'static str,
    /// One category, or several (recorded when any of them is on).
    pub cat: u32,
    pub track: Track,
    /// A collapsible parent for a `Global` or counter track ("Interrupts",
    /// "DMA", ...). Thread tracks always sit under the process.
    pub group: Option<&'static str>,
    /// Name of the per-event `u64` argument, shown in the slice details.
    /// `None` drops the argument. Counters always use it as their value.
    pub arg: Option<&'static str>,
}

impl EventDef {
    pub const fn new(name: &'static str, cat: u32) -> Self {
        Self { name, cat, track: Track::Thread, group: None, arg: None }
    }
    pub const fn in_group(mut self, group: &'static str) -> Self {
        self.group = Some(group);
        self
    }
    pub const fn on_track(mut self, track: &'static str) -> Self {
        self.track = Track::Global(track);
        self
    }
    pub const fn with_arg(mut self, arg: &'static str) -> Self {
        self.arg = Some(arg);
        self
    }
}

/// Matches `TrackEvent.Type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Kind {
    Begin = 1,
    End = 2,
    Instant = 3,
    Counter = 4,
}

impl Kind {
    fn from_u8(v: u8) -> Kind {
        match v { 1 => Kind::Begin, 2 => Kind::End, 3 => Kind::Instant, _ => Kind::Counter }
    }
}

/// What a ring slot holds. `guest_kind` packs the guest instruction count
/// (high 56 bits) with the `Kind` (low 8): 2^56 instructions is years of
/// guest time at any speed we reach.
#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct Record {
    def: *const EventDef,
    host_ticks: u64,
    guest_kind: u64,
    arg: u64,
}

impl Record {
    fn def(&self) -> &'static EventDef { unsafe { &*self.def } }
    fn kind(&self) -> Kind { Kind::from_u8(self.guest_kind as u8) }
    fn guest(&self) -> u64 { self.guest_kind >> 8 }
}

// ── Enable mask and clocks ───────────────────────────────────────────────────

/// Categories being recorded; 0 when no session is running.
static MASK: AtomicU32 = AtomicU32::new(0);

#[inline(always)]
pub fn on(def: &EventDef) -> bool {
    MASK.load(Ordering::Relaxed) & def.cat != 0
}

pub fn mask() -> u32 { MASK.load(Ordering::Relaxed) }

/// Change the recorded categories of the running session (no-op without
/// one, so a stopped session stays stopped).
pub fn set_categories(mask: u32) -> Result<(), String> {
    let s = SESSION.lock();
    let Some(sess) = s.as_ref() else { return Err("no perfetto session running".into()) };
    sess.cats.store(mask, Ordering::Relaxed);
    MASK.store(mask, Ordering::Relaxed);
    Ok(())
}

/// The CPU's retired-instruction counter (`Hot::cycles`), or null before a
/// machine wires it up.
static GUEST_CYCLES: AtomicPtr<u64> = AtomicPtr::new(std::ptr::null_mut());

/// Point the guest clock at the CPU's cycle counter. Pass `None` before the
/// core goes away.
pub fn set_guest_cycles(ptr: Option<crate::cpu::mips_core::CyclesPtr>) {
    GUEST_CYCLES.store(ptr.map_or(std::ptr::null_mut(), |p| p.raw() as *mut u64), Ordering::Release);
}

#[inline(always)]
pub(crate) fn guest_now() -> u64 {
    let p = GUEST_CYCLES.load(Ordering::Relaxed);
    if p.is_null() { 0 } else { unsafe { std::ptr::read_volatile(p) } }
}

/// Host time in ns: CLOCK_BOOTTIME on Linux (Perfetto's default trace
/// clock, so this file can be merged with a system trace); elsewhere ns since
/// the first call. ~27 ns per call even through the vDSO, so events record
/// `host_ticks` instead and the writer converts.
pub(crate) fn host_ns() -> u64 {
    #[cfg(target_os = "linux")]
    {
        let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        unsafe { libc::clock_gettime(libc::CLOCK_BOOTTIME, &mut ts) };
        ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
    }
    #[cfg(not(target_os = "linux"))]
    {
        static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
        EPOCH.get_or_init(std::time::Instant::now).elapsed().as_nanos() as u64
    }
}

/// The cheapest monotonic counter the host has: the TSC on x86_64 (~11 ns;
/// invariant on anything recent enough to run IRIS at speed), the virtual
/// counter on aarch64, otherwise `host_ns` itself. Only differences
/// matter: the writer maps ticks to `host_ns` by interpolating between
/// (ticks, ns) anchors it takes at every drain (`writer::HostClock`).
#[inline(always)]
pub(crate) fn host_ticks() -> u64 {
    #[cfg(target_arch = "x86_64")]
    { unsafe { core::arch::x86_64::_rdtsc() } }
    #[cfg(target_arch = "aarch64")]
    {
        let v: u64;
        unsafe { core::arch::asm!("mrs {}, cntvct_el0", out(reg) v, options(nomem, nostack)) };
        v
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    { host_ns() }
}

// ── Per-thread rings ─────────────────────────────────────────────────────────

/// Slots per thread ring (32 bytes each, so 2 MiB). The writer drains every
/// couple of milliseconds; overflow drops events and is counted, never
/// blocks the emitting thread.
const RING_SLOTS: usize = 1 << 16;

/// Single-producer (the owning thread) / single-consumer (the writer, or
/// `start` while no writer runs) ring.
pub(crate) struct Ring {
    head: CachePadded<AtomicU64>,
    tail: CachePadded<AtomicU64>,
    /// Producer's last look at `tail`, so a push only reloads the shared
    /// line when the ring looks full.
    cached_tail: UnsafeCell<u64>,
    slots: Box<[UnsafeCell<MaybeUninit<Record>>]>,
    dropped: AtomicU64,
    pub(crate) index: usize,
    pub(crate) tid: i64,
    pub(crate) name: String,
}

// Records hold `*const EventDef`, which point at statics; slots are only
// touched by the producer before publication and the consumer after.
unsafe impl Send for Ring {}
unsafe impl Sync for Ring {}

impl Ring {
    fn new(index: usize, tid: i64, name: String) -> Self {
        let slots = (0..RING_SLOTS).map(|_| UnsafeCell::new(MaybeUninit::uninit())).collect();
        Self {
            head: CachePadded::new(AtomicU64::new(0)),
            tail: CachePadded::new(AtomicU64::new(0)),
            cached_tail: UnsafeCell::new(0),
            slots,
            dropped: AtomicU64::new(0),
            index,
            tid,
            name,
        }
    }

    /// Owning thread only. Instants and counters may only fill the ring to
    /// `UNPAIRED_LIMIT`; the rest is kept for slice begins and ends, so a
    /// flood of instants overflowing the ring drops instants rather than
    /// the end of some slice (which would leave it open, and every later
    /// slice on that track nested inside it).
    #[inline(always)]
    fn push(&self, r: Record) {
        const UNPAIRED_LIMIT: u64 = (RING_SLOTS - RING_SLOTS / 8) as u64;
        let limit = match Kind::from_u8(r.guest_kind as u8) {
            Kind::Begin | Kind::End => RING_SLOTS as u64,
            _ => UNPAIRED_LIMIT,
        };
        let h = self.head.load(Ordering::Relaxed);
        let cached = unsafe { &mut *self.cached_tail.get() };
        if h - *cached >= limit {
            *cached = self.tail.load(Ordering::Acquire);
            if h - *cached >= limit {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                return;
            }
        }
        unsafe { (*self.slots[h as usize & (RING_SLOTS - 1)].get()).write(r) };
        self.head.store(h + 1, Ordering::Release);
    }

    /// Consumer only: append everything published so far to `out`.
    pub(crate) fn drain_into(&self, out: &mut Vec<(usize, Record)>) {
        let h = self.head.load(Ordering::Acquire);
        let t = self.tail.load(Ordering::Relaxed);
        for i in t..h {
            let r = unsafe { (*self.slots[i as usize & (RING_SLOTS - 1)].get()).assume_init() };
            out.push((self.index, r));
        }
        self.tail.store(h, Ordering::Release);
    }

    /// Consumer only: throw away whatever is queued.
    pub(crate) fn discard(&self) {
        self.tail.store(self.head.load(Ordering::Acquire), Ordering::Release);
        self.dropped.store(0, Ordering::Relaxed);
    }

    pub(crate) fn take_dropped(&self) -> u64 {
        self.dropped.swap(0, Ordering::Relaxed)
    }
}

/// Every ring ever created. Rings are never freed (a thread that exits keeps
/// its ring so the writer can drain it, and a later session reuses it), which
/// is what makes the thread-local raw pointer below sound.
pub(crate) fn rings() -> &'static Mutex<Vec<Arc<Ring>>> {
    static RINGS: OnceLock<Mutex<Vec<Arc<Ring>>>> = OnceLock::new();
    RINGS.get_or_init(|| Mutex::new(Vec::new()))
}

fn current_tid() -> i64 {
    #[cfg(target_os = "linux")]
    { unsafe { libc::gettid() as i64 } }
    #[cfg(not(target_os = "linux"))]
    {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        thread_local!(static TID: i64 = NEXT.fetch_add(1, Ordering::Relaxed) as i64);
        TID.with(|t| *t)
    }
}

thread_local! {
    static RING: Cell<*const Ring> = const { Cell::new(std::ptr::null()) };
}

#[cold]
fn register_ring() -> *const Ring {
    let mut all = rings().lock();
    let name = std::thread::current().name().unwrap_or("unnamed").to_string();
    let ring = Arc::new(Ring::new(all.len(), current_tid(), name));
    let p = Arc::as_ptr(&ring);
    all.push(ring);
    RING.with(|c| c.set(p));
    p
}

/// Record an event on the calling thread. Callers check `on(def)` first
/// (the macros do).
#[inline]
pub fn emit(def: &'static EventDef, kind: Kind, arg: u64) {
    let r = Record {
        def,
        host_ticks: host_ticks(),
        guest_kind: (guest_now() << 8) | kind as u64,
        arg,
    };
    let mut p = RING.with(|c| c.get());
    if p.is_null() {
        p = register_ring();
    }
    unsafe { (*p).push(r) };
}

/// Ends a slice when dropped (`pf_scope!`). Only emits the end if the begin
/// was emitted, so toggling a category mid-slice cannot leave a stray end.
pub struct Scope {
    def: Option<&'static EventDef>,
}

impl Scope {
    #[inline]
    pub fn begin(def: &'static EventDef, arg: u64) -> Self {
        if on(def) {
            emit(def, Kind::Begin, arg);
            Scope { def: Some(def) }
        } else {
            Scope { def: None }
        }
    }
}

impl Drop for Scope {
    #[inline]
    fn drop(&mut self) {
        if let Some(def) = self.def {
            if mask() != 0 {
                emit(def, Kind::End, 0);
            }
        }
    }
}

// ── Edge trackers ────────────────────────────────────────────────────────────
//
// Device state that is a level (an interrupt line, "the guest is waiting for
// the engine") is drawn as a slice from rising to falling edge. The trackers
// remember which edges were emitted *in the current session*, so a session
// started while a line is high opens its slice at the next rising edge
// instead of emitting an unmatched end, and one started with a line high
// (`PfBits`) opens the slice at the first update.
//
// The types exist without the feature too (as zero-sized no-ops behind the
// `pf_level!`/`pf_bits!`/`pf_value!` macros), so device structs can hold them
// unconditionally; see `perfetto_macros.rs`.

/// Bumped by every `start`; 0 means "never recorded".
static SESSION_GEN: AtomicU32 = AtomicU32::new(0);

#[inline(always)]
fn session_gen() -> u32 {
    SESSION_GEN.load(Ordering::Relaxed)
}

impl crate::PfLevel {
    /// Emit `def`'s begin on a rising edge and its end on the falling one.
    #[inline]
    pub fn set(&self, def: &'static EventDef, level: bool, arg: u64) {
        let open = self.open.load(Ordering::Relaxed);
        if level {
            if !on(def) { return; }
            let gen = session_gen();
            if open != gen && self.open.swap(gen, Ordering::Relaxed) != gen {
                emit(def, Kind::Begin, arg);
            }
        } else if open != 0 && self.open.swap(0, Ordering::Relaxed) == session_gen() && mask() != 0 {
            emit(def, Kind::End, 0);
        }
    }
}

impl crate::PfBits {
    /// Compare `now` with the previous value; for each bit that rose emit
    /// `defs[bit]`'s begin, for each that fell its end.
    #[inline]
    pub fn update(&self, defs: &'static [EventDef; 32], now: u32) {
        if mask() == 0 { return; }
        let gen = session_gen();
        // Restarted session: nothing is open in it yet.
        let prev = if self.gen.swap(gen, Ordering::Relaxed) == gen { self.bits.load(Ordering::Relaxed) } else { 0 };
        let changed = prev ^ now;
        if changed == 0 { return; }
        self.bits.store(now, Ordering::Relaxed);
        let mut c = changed;
        while c != 0 {
            let bit = c.trailing_zeros() as usize;
            c &= c - 1;
            let def = &defs[bit];
            if on(def) {
                emit(def, if now >> bit & 1 != 0 { Kind::Begin } else { Kind::End }, 0);
            }
        }
    }
}

impl crate::PfValue {
    /// Emit `def`'s counter when `v` differs from the last one emitted.
    #[inline]
    pub fn set(&self, def: &'static EventDef, v: i64) {
        if !on(def) { return; }
        let gen = session_gen();
        if self.last.load(Ordering::Relaxed) == v && self.gen.load(Ordering::Relaxed) == gen { return; }
        self.last.store(v, Ordering::Relaxed);
        self.gen.store(gen, Ordering::Relaxed);
        emit(def, Kind::Counter, v as u64);
    }
}

// ── Sessions ─────────────────────────────────────────────────────────────────

/// Which timestamp becomes the trace's timeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Clock {
    /// Host time in ns.
    Host,
    /// Guest retired instructions, drawn as 1 instruction = 1 ns.
    Guest,
}

impl Clock {
    pub fn parse(s: &str) -> Result<Clock, String> {
        match s {
            "host" => Ok(Clock::Host),
            "guest" => Ok(Clock::Guest),
            _ => Err(format!("unknown perfetto clock '{s}' (host|guest)")),
        }
    }
    pub fn name(self) -> &'static str {
        match self { Clock::Host => "host", Clock::Guest => "guest" }
    }
}

static SESSION: Mutex<Option<writer::Session>> = Mutex::new(None);

/// Start writing a trace to `path`.
pub fn start(path: &str, cats: u32, clock: Clock) -> Result<(), String> {
    let mut s = SESSION.lock();
    if let Some(sess) = s.as_ref() {
        return Err(format!("perfetto session already running ({})", sess.path));
    }
    let file = std::fs::File::create(path).map_err(|e| format!("perfetto: {path}: {e}"))?;
    // No writer runs yet, so this thread may act as the consumer: drop
    // anything left over from an earlier session before recording resumes.
    for r in rings().lock().iter() {
        r.discard();
    }
    *s = Some(writer::Session::spawn(file, path.to_string(), cats, clock));
    SESSION_GEN.fetch_add(1, Ordering::Relaxed);
    MASK.store(cats, Ordering::Relaxed);
    Ok(())
}

/// Stop the running session, flush it, and describe what was written.
pub fn stop() -> Result<String, String> {
    let sess = SESSION.lock().take().ok_or("no perfetto session running")?;
    MASK.store(0, Ordering::Relaxed);
    let path = sess.path.clone();
    let st = sess.finish()?;
    Ok(format!(
        "perfetto: wrote {path}: {} events, {} dropped, {} bytes",
        st.events, st.dropped, st.bytes
    ))
}

pub fn status() -> String {
    match SESSION.lock().as_ref() {
        None => "perfetto: idle".into(),
        Some(s) => {
            let st = s.stats();
            format!(
                "perfetto: recording to {} for {:.1}s  clock={}  cats={}  {} events, {} dropped, {} bytes, {} threads",
                s.path,
                s.started.elapsed().as_secs_f64(),
                s.clock.name(),
                cat::format(s.cats.load(Ordering::Relaxed)),
                st.events, st.dropped, st.bytes,
                rings().lock().len(),
            )
        }
    }
}

/// Stop the session if one is running (process shutdown).
pub fn shutdown() {
    if SESSION.lock().is_some() {
        match stop() {
            Ok(msg) => eprintln!("{msg}"),
            Err(e) => eprintln!("perfetto: {e}"),
        }
    }
}

/// Parse `start` arguments: `FILE [cats=LIST] [clock=host|guest]`.
pub fn parse_start_args(args: &[&str]) -> Result<(String, u32, Clock), String> {
    let mut it = args.iter();
    let path = it.next().ok_or("usage: perfetto start FILE [cats=LIST] [clock=host|guest]")?;
    let mut cats = cat::ALL;
    let mut clock = Clock::Host;
    for a in it {
        if let Some(v) = a.strip_prefix("cats=") {
            cats = cat::parse(v)?;
        } else if let Some(v) = a.strip_prefix("clock=") {
            clock = Clock::parse(v)?;
        } else {
            return Err(format!("perfetto: unexpected argument '{a}'"));
        }
    }
    Ok((path.to_string(), cats, clock))
}

/// The `perfetto` monitor command.
pub struct PerfettoMonitor;

impl crate::traits::Device for PerfettoMonitor {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) {}
    fn start(&self) {}
    fn is_running(&self) -> bool { false }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![(
            "perfetto".to_string(),
            "Perfetto trace: perfetto start FILE [cats=LIST] [clock=host|guest] | stop | status | cats LIST".to_string(),
        )]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut writer: Box<dyn std::io::Write + Send>) -> Result<(), String> {
        if cmd != "perfetto" {
            return Err(format!("Unknown command: {cmd}"));
        }
        let msg = match args.first().copied().unwrap_or("status") {
            "start" => {
                let (path, cats, clock) = parse_start_args(&args[1..])?;
                start(&path, cats, clock)?;
                format!("perfetto: recording to {path} (clock={}, cats={})", clock.name(), cat::format(cats))
            }
            "stop" => stop()?,
            "status" => status(),
            "cats" => {
                let list = args.get(1).ok_or("usage: perfetto cats LIST")?;
                let m = cat::parse(list)?;
                set_categories(m)?;
                format!("perfetto: cats={}", cat::format(m))
            }
            other => return Err(format!("perfetto: unknown subcommand '{other}'")),
        };
        writeln!(writer, "{msg}").map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests;
