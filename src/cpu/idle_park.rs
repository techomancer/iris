//! Kernel idle-loop detection and in-place CPU thread parking.
//!
//! Used by the CPU run loop in `mips_exec.rs`. See
//! `clanker/perf/idle-pause-work.md`.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::time::{Duration, Instant};

use crate::cpu::mips_core::{CAUSE_IP_MASK, STATUS_IM_MASK};
use crate::cpu::mips_core::MipsCore;
use crate::cpu::smp::{HOST_PARK_BIT, MAX_CPUS};

/// PCs kept for the idle-window test. The batch run loop calls [`sample`]
/// after each ten-instruction group, so with [`SAMPLE_EVERY`] = 32 the ring
/// spans about 10,000 guest instructions — several iterations of even a long
/// idle loop. The single-step path spans 32.
const PC_RING: usize = 32;
/// Number of [`sample`] calls between recorded PCs. The batch loop calls it
/// after every ten instructions (~320 guest instructions per record); the
/// single-step path calls it after each instruction.
const SAMPLE_EVERY: u32 = 32;
/// A "cluster" is a run of sorted sample PCs no more than this far apart.
/// IRIX's idle path is one hot loop plus a few helper functions (spl, the
/// ll/sc lock) reached by calls, so it is a cluster, not one contiguous
/// window.
const CLUSTER_GAP_BYTES: u64 = 256;
/// The dominant cluster must hold at least this many of the PC_RING samples.
/// A short poll or DELAY loop also fills the ring, but with too few distinct
/// PCs (checked separately). Mixed work spreads its samples over several
/// clusters and cannot reach this.
const CLUSTER_MIN_SAMPLES: usize = 12;
/// ... and at least this many distinct PCs, which rejects 2-3 instruction
/// countdown/device-poll loops and the 8-instruction ll/sc lock spin outright.
/// The IRIX 5.3 idle loop has 14 and 6.5's scheduler path has ~89.
const CLUSTER_MIN_DISTINCT: usize = 12;
/// The dominant-cluster test must hold for this many consecutive batches
/// (1 000 instructions per batch) before parking. One batch is enough to
/// reject a short-lived hot function without delaying idle parking, which
/// would otherwise re-run thousands of idle-loop iterations after every
/// timer interrupt before sleeping again.
const IDLE_STREAK: u32 = 1;
/// The newest samples that must lie in the dominant cluster for the window to
/// count as idle: one batch's worth (1000 instructions, a sample every ~320),
/// less one for a helper the idle loop calls from outside its own range.
const RECENT_SAMPLES: usize = 3;
/// Compiled batches between two probe runs of a stationary compiled loop
/// that the last run did not find idle: a busy kernel loop with one exit
/// loses one run ([`PROBE_RUN`] interpreted batches) in this many.
const PROBE_BACKOFF: u32 = 256;
/// Interpreted batches in one probe run: enough to fill the ring (ten) with
/// a margin for a clock handler or the clears an interrupt costs, bounded so
/// a loop the detector keeps resetting (interrupts masked) is not
/// interpreted for good.
const PROBE_RUN: u32 = 16;
/// The slice of a parked CPU: how long a `running` change or, on a
/// multiprocessor, work queued by another CPU's interrupt handler may wait.
const SLICE_NS: u64 = 1_000_000;
/// Region-exit signatures (PC, return address) a CPU remembers as its idle
/// loop seen through compiled code (see [`IdleParkState::known_cluster`]).
const KNOWN_PAIRS: usize = 32;
/// The slice of a multiprocessor's parked CPU while another CPU still runs:
/// how long work that CPU queues for it may wait (see `park`).
const PEER_SLICE_NS: u64 = PEER_SLICE_DEFAULT_NS;
const PEER_SLICE_DEFAULT_NS: u64 = 100_000;
/// Each peer slice that ends with no work found doubles the next, up to
/// `PEER_SLICE_NS << PEER_BACKOFF_MAX`: hand-offs stay quick while the guest
/// is busy and a settled idle guest wakes about once a millisecond.
const PEER_BACKOFF_MAX: u32 = 3;
/// Host-pacing rate for `hot.cycles` while the CPU is parked: one guest cycle
/// per 10 ns of real time, the rate `NS_PER_GUEST_CYCLE` names under
/// `ci_clock`. In every build this keeps cross-thread cycle readers
/// (WD33C93A's deferred-interrupt spin, CP0 Random) moving during a park.
const STALL_NS_PER_CYCLE: u64 = 10;
/// Move `hot.cycles` on by a park's host time, and record that those cycles
/// were not instructions. Guest-visible timing (Count under `ci_clock`,
/// CP0 Random, the cycle-reading device spins) follows `hot.cycles` exactly
/// as before; only the status bars' MIPS reads `CyclesPtr::retired`, which
/// leaves them out.
fn advance_parked(core: &mut MipsCore, elapsed_ns: u64) {
    let cycles = elapsed_ns / STALL_NS_PER_CYCLE;
    core.hot.cycles = core.hot.cycles.wrapping_add(cycles);
    core.hot.parked_cycles = core.hot.parked_cycles.wrapping_add(cycles);
}

/// After a park wakes on a pending line the guest cannot take (`(ip & im) ==
/// 0`), do not park again for this many `hot.cycles`. A genuine idle loop
/// wakes for an unmasked interrupt; waking for a masked one means the
/// sampled cluster was a device-poll or DELAY loop that happened to look
/// idle. Sleeping that loop caps it at one batch per slice, and with the
/// wall-clock model the park no longer moves Count to compensate, so boot
/// crawls. About 20 ms of host pacing at the 10 ns/cycle rate.
const PARK_BACKOFF_CYCLES: u64 = 2_000_000;
/// Do not park during the first minute of CPU time. Parking is a steady-state
/// optimisation: an IRIX boot runs long stretches inside single kernel
/// functions and instruction-counted wait loops, and slicing those at 1 ms
/// stretches a 45-second boot to minutes. The guest is doing real work during
/// boot; keep it at full speed, then start parking once it has settled.
const PARK_WARMUP: Duration = Duration::from_secs(60);

/// The CPU thread while it is parked in [`IdleParkState::park`], so an
/// interrupt source can wake it at once instead of leaving it to notice on its
/// next slice. One slot per CPU.
static PARKER: parking_lot::Mutex<[Option<std::thread::Thread>; MAX_CPUS]> = parking_lot::const_mutex([const { None }; MAX_CPUS]);
/// A bit per CPU, set while its thread is between its last look at the pending
/// word and its sleep. Paired with the writers' order (set the bit, then read
/// this) so a wakeup cannot be lost.
static PARKED: AtomicU8 = AtomicU8::new(0);
/// A bit per CPU whose thread has run the idle detector, so a parked CPU of a
/// multiprocessor knows which others could still hand it work.
static SEEN: AtomicU8 = AtomicU8::new(0);
/// A bit per CPU the detector last found in its idle loop, cleared when it
/// leaves it. A CPU between two parks keeps its bit, so its waking does not
/// count as another CPU running.
static IDLE: AtomicU8 = AtomicU8::new(0);
/// The PC range (lowest, highest) of each CPU's dominant cluster while its
/// window looks idle, zero otherwise. On a multiprocessor a CPU parks only
/// where another CPU's idle-looking cluster lies within [`SHARED_LOOP_BYTES`]
/// of its own: the idle loop is one piece of kernel code every idle CPU runs,
/// where a busy kernel loop that happens to look idle (IRIX's `melf_findaddr`
/// scanning a module's symbols with `strcmp`, in kernel mode with
/// interrupts enabled, 12+ distinct PCs) runs on one CPU alone. Parked, that
/// scan ran 1000 instructions per slice and a 4-CPU boot's rc phase took
/// 88 s against 35 s.
static IDLE_CLUSTER: [[AtomicU64; 2]; MAX_CPUS] = [const { [AtomicU64::new(0), AtomicU64::new(0)] }; MAX_CPUS];
/// How far apart two CPUs' idle-looking clusters may lie and still count as
/// the same loop: the dominant cluster is a sub-range of the idle path
/// (IRIX 6.5's `idler`, `local_idle` and `idlerunq` span about 1.6 KiB).
const SHARED_LOOP_BYTES: u64 = 2048;
/// When this process first built an `IdleParkState`. One `Machine` per
/// process, and the warm-up must not restart every time the monitor stops and
/// starts the CPU thread (which builds a fresh state), so it is process-wide.
static PROCESS_START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();

/// Park statistics, shown by the monitor's `idle status`. They answer "is
/// this guest idle or just being throttled?" without a debugger.
pub static PARK_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static PARK_SLEEP_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static PARK_WAKE_IRQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static PARK_WAKE_OTHER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Parks a multiprocessor ended after one slice because another CPU was
/// still running.
pub static PARK_WAKE_PEER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Parks a multiprocessor ended after a slice with every CPU idle, so the
/// idle loop could poll its run queue.
pub static PARK_WAKE_SLICE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Idle-looking windows a multiprocessor's CPU did not park on because no
/// other CPU was in the same loop.
pub static PARK_ALONE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Parks decided on compiled-code samples alone, from a learned signature.
pub static KNOWN_PARKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Probe runs whose region-exit signature was learned as the idle loop.
pub static LEARNED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Batches a jitv2 build ran interpreted to sample the PCs inside a compiled
/// loop (see [`IdleParkState::wants_probe`]).
pub static PROBES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static UPDATES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static RESET_IRQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static RESET_IP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static RESET_IM0: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static RESET_IE0: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static RESET_USER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static WINDOW_FULL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static WINDOW_SAMPLES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static WINDOW_DISTINCT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Wake the parked CPU threads. Call after setting a bit in `hot.interrupts`
/// from any thread; one relaxed-cost atomic load when no CPU is parked. Every
/// parked CPU wakes and looks at its own word, so the caller need not say whose
/// bit it set; one that finds nothing sleeps again.
#[inline]
pub fn wake() {
    if PARKED.load(Ordering::SeqCst) != 0 {
        for t in PARKER.lock().iter().flatten() {
            t.unpark();
        }
    }
}

fn announce_parked(core: &MipsCore) {
    PARKED.fetch_or(1 << core.cpu_id, Ordering::SeqCst);
}

fn clear_parked(core: &MipsCore) {
    PARKED.fetch_and(!(1 << core.cpu_id), Ordering::SeqCst);
}

/// Detects kernel idle loops from a per-instruction PC sample ring.
///
/// The previous detector hashed PC + GPRs at each 1000-instruction batch and
/// required an exact repeat. It fails IRIX 6.5's idle path (a ~470-byte
/// scheduler loop with calls and stores whose registers evolve every
/// iteration) and, worse, it *does* match short device-poll loops (e.g.
/// `lbu/andi/bne -3` waiting on a SCSI bit), which must keep executing.
///
/// This detector samples the PC every [`SAMPLE_EVERY`] calls (about 320
/// instructions in the batch loop), groups
/// the samples into PC-space clusters, and parks only when one long kernel
/// cluster dominates the ring. User-mode samples are tolerated in a minority
/// (a syscall wake runs a few user instructions between kernel stretches) but
/// a user-dominated ring is refused, so a user process spinning is never
/// parked.
pub struct IdleParkState {
    pcs: [u64; PC_RING],
    pc_len: usize,
    pc_pos: usize,
    /// Instructions until the next PC sample.
    sample_countdown: u32,
    /// Per-slot user-mode flag, parallel to `pcs`.
    user: [bool; PC_RING],
    /// Consecutive batches whose dominant cluster was the same loop.
    streak: u32,
    /// Do not park again before `hot.cycles` reaches this value; set after a
    /// park that woke on a masked line (see [`PARK_BACKOFF_CYCLES`]).
    park_skip_until_cycle: u64,
    /// CPU-thread start, for [`PARK_WARMUP`].
    started: Instant,
    /// Warm-up length; `IRIS_IDLE_WARMUP_SECS` overrides it (0 disables).
    warmup: Duration,
    /// This CPU's bit in `IDLE` (multiprocessors only).
    idle: bool,
    /// The last update saw a full window that is not the idle loop.
    busy: bool,
    /// Doublings of the peer slice since this CPU last had work.
    peer_backoff: u32,
    /// Compiled batches until the next probe run of a stationary loop.
    probe_backoff: u32,
    /// Interpreted batches left in the current probe run.
    probe_left: u32,
    /// Return address (r31) of each sample, parallel to `pcs`. Under jitv2 a
    /// sample is a region exit, often the entry of a helper on another page,
    /// and the return address says which loop called it.
    ras: [u64; PC_RING],
    /// Region-exit signatures learned as this CPU's idle loop: the
    /// (PC, return address) pairs of a stationary compiled ring that a probe
    /// run then found idle at once.
    known: [(u64, u64); KNOWN_PAIRS],
    known_len: usize,
    /// Next slot `learn` overwrites once `known` is full.
    known_next: usize,
    /// The interpreted idle cluster the signatures belong to, published for
    /// the shared-loop rule when a park is decided on compiled samples.
    known_loop: (u64, u64),
    /// The dominant cluster of the last park decided on interpreted samples.
    last_loop: (u64, u64),
    /// The signature of the ring that started the current probe run.
    probe_sig: [(u64, u64); CLUSTER_MIN_DISTINCT],
    probe_sig_len: usize,
    /// The last park was decided on a learned signature.
    parked_known: bool,
    /// `hot.cycles` at the last update: a reset starts it again from zero,
    /// and the signatures of the old kernel go with it.
    last_cycles: u64,
}

/// What to do with the current sample window.
pub enum IdleDecision {
    /// Keep executing: no idle cluster detected, or an interrupt is ready to
    /// deliver so the guest should run, not sleep.
    None,
    /// One long kernel cluster dominates the window with interrupts enabled
    /// and a mask bit set: park the CPU thread; an unmasked interrupt ends the
    /// park.
    Park,
    /// A masked interrupt is already pending (`ip != 0`), so parking would
    /// wait for a line the guest only takes after it drops the spl — which it
    /// can only do running. Pace instead: sleep in bounded slices, advancing
    /// `hot.cycles` at the 10 ns/cycle host-pacing rate so cross-thread
    /// readers keep moving. Each slice re-checks `running` and the interrupt
    /// word, so a monitor command, a soft reset, or a newly-unmasked
    /// interrupt ends it at once.
    Pace,
}

impl Default for IdleParkState {
    fn default() -> Self {
        Self::new()
    }
}

impl IdleParkState {
    pub fn new() -> Self {
        let warmup = match std::env::var("IRIS_IDLE_WARMUP_SECS") {
            Ok(s) => Duration::from_secs(s.parse().unwrap_or(PARK_WARMUP.as_secs())),
            Err(_) => PARK_WARMUP,
        };
        Self {
            pcs: [0; PC_RING],
            pc_len: 0,
            pc_pos: 0,
            sample_countdown: SAMPLE_EVERY,
            user: [false; PC_RING],
            streak: 0,
            idle: false,
            busy: false,
            peer_backoff: 0,
            probe_backoff: 0,
            probe_left: 0,
            ras: [0; PC_RING],
            known: [(0, 0); KNOWN_PAIRS],
            known_len: 0,
            known_next: 0,
            known_loop: (0, 0),
            last_loop: (0, 0),
            probe_sig: [(0, 0); CLUSTER_MIN_DISTINCT],
            probe_sig_len: 0,
            parked_known: false,
            last_cycles: 0,
            park_skip_until_cycle: 0,
            started: *PROCESS_START.get_or_init(Instant::now),
            warmup,
        }
    }

    /// Test constructor with the boot warm-up already elapsed.
    #[cfg(test)]
    pub(crate) fn ready() -> Self {
        let mut s = Self::new();
        s.warmup = Duration::ZERO;
        s.started = Instant::now();
        s
    }

    fn clear_window(&mut self) {
        self.pc_len = 0;
        self.pc_pos = 0;
        self.sample_countdown = SAMPLE_EVERY;
        self.streak = 0;
    }

    /// Sample the PC every [`SAMPLE_EVERY`] calls. The batch run loop calls
    /// this after each ten-instruction group, the single-step path after each
    /// instruction; one predictable decrement and a rarely-taken branch on the
    /// hot path.
    #[inline]
    pub fn sample(&mut self, core: &MipsCore) {
        self.sample_countdown -= 1;
        if self.sample_countdown != 0 {
            return;
        }
        self.sample_countdown = SAMPLE_EVERY;
        let user = !matches!(core.get_privilege_mode(), crate::cpu::mips_core::PrivilegeMode::Kernel);
        self.pcs[self.pc_pos] = core.pc;
        self.ras[self.pc_pos] = core.gpr[31];
        self.user[self.pc_pos] = user;
        self.pc_pos = (self.pc_pos + 1) % PC_RING;
        if self.pc_len < PC_RING {
            self.pc_len += 1;
        }
    }

    /// Largest kernel cluster in the sample ring: `(samples, distinct_pcs,
    /// users, range)`, where `range` is the cluster's lowest and highest PC
    /// and `users` the number of user-mode samples. User samples are ignored
    /// for the cluster (a syscall wake runs a few user instructions between
    /// long kernel stretches) but a ring dominated by user code is refused.
    fn largest_cluster(&self) -> (usize, usize, usize, (u64, u64)) {
        let mut kernel: [u64; PC_RING] = [0; PC_RING];
        let mut n = 0usize;
        let mut users = 0usize;
        for i in 0..PC_RING {
            if self.user[i] {
                users += 1;
            } else {
                kernel[n] = self.pcs[i];
                n += 1;
            }
        }
        let sorted = &mut kernel[..n];
        sorted.sort_unstable();
        let mut best = (0usize, 0usize, users, (0u64, 0u64));
        let mut start = 0usize;
        for end in 1..=n {
            let boundary = end == n
                || sorted[end].saturating_sub(sorted[end - 1]) > CLUSTER_GAP_BYTES;
            if !boundary {
                continue;
            }
            let slice = &sorted[start..end];
            let mut distinct = 0usize;
            for i in 0..slice.len() {
                if !slice[..i].contains(&slice[i]) {
                    distinct += 1;
                }
            }
            if slice.len() > best.0 {
                best = (slice.len(), distinct, users, (slice[0], slice[slice.len() - 1]));
            }
            start = end;
        }
        best
    }

    /// The dominant cluster's PC range when one long, stationary kernel
    /// cluster dominates the ring, user code is only a minority of it, and
    /// the CPU is still in that cluster:
    /// the newest batch's samples ([`RECENT_SAMPLES`], all but one) lie in
    /// its PC range. The ring spans ten batches, so after a park ends the
    /// idle loop's samples still dominate it while the CPU runs the
    /// interrupt handler that woke it, or the work it found on its run
    /// queue; without this test it was parked again after one batch, and
    /// ran that handler 1000 instructions per clock tick (minutes at "The
    /// system is coming up." on a 4-CPU Tezro).
    ///
    /// Returns the range and whether the newest batch is inside it. A
    /// cluster the CPU has just left (the handler of the interrupt that
    /// woke it, or work it found) is not idle, but not a busy window either:
    /// the CPU keeps its idle flag and its published cluster until a full
    /// window of other work displaces the loop. Clearing the flag on the
    /// handler batch made every clock tick flip it, and the other CPUs,
    /// seeing a peer running, cut their parks to the short peer slice.
    fn idle_cluster(&self) -> Option<((u64, u64), bool)> {
        if self.pc_len < PC_RING {
            return None;
        }
        let (samples, distinct, users, (lo, hi)) = self.largest_cluster();
        if users > PC_RING / 4 || samples < CLUSTER_MIN_SAMPLES || distinct < CLUSTER_MIN_DISTINCT {
            return None;
        }
        let recent = (1..=RECENT_SAMPLES)
            .filter(|k| {
                let i = (self.pc_pos + PC_RING - k) % PC_RING;
                !self.user[i] && (lo..=hi).contains(&self.pcs[i])
            })
            .count();
        Some(((lo, hi), recent + 1 >= RECENT_SAMPLES))
    }

    /// Whether to start a probe run: the next batches run interpreted to see
    /// where a compiled loop really is. Compiled code keeps a loop's
    /// back-edges inside the region, so between `step_jit` calls the run
    /// loop samples only region exits: an idle guest shows one or two PCs,
    /// under `CLUSTER_MIN_DISTINCT`, and never parked (a jitv2 Tezro idled at
    /// 400% host CPU). When the ring holds a stationary kernel cluster with
    /// too few distinct PCs, the ring is cleared and [`PROBE_RUN`]
    /// interpreted batches refill it with the loop's own PCs for the
    /// ordinary test ([`IdleParkState::probe_done`] says when to stop). A run
    /// that found no idle loop backs off [`PROBE_BACKOFF`] compiled batches.
    pub fn wants_probe(&mut self) -> bool {
        if self.probe_backoff > 0 {
            self.probe_backoff -= 1;
            return false;
        }
        if self.pc_len < PC_RING {
            return false;
        }
        // A learned idle signature: `decide` reads this ring itself.
        if self.known_cluster().is_some() {
            return false;
        }
        if let Some(range) = self.stationary_range() {
            let (pairs, n) = self.cluster_pairs(range);
            let n = n.min(CLUSTER_MIN_DISTINCT);
            self.probe_sig[..n].copy_from_slice(&pairs[..n]);
            self.probe_sig_len = n;
            self.clear_window();
            self.probe_left = PROBE_RUN;
            return true;
        }
        false
    }

    /// Whether this CPU has learned a compiled signature of its idle loop,
    /// so the batches after a park can run compiled.
    pub fn knows_compiled_idle(&self) -> bool {
        self.known_len > 0
    }

    fn knows(&self, pair: (u64, u64)) -> bool {
        self.known[..self.known_len].contains(&pair)
    }

    /// The distinct (PC, return address) pairs of the ring's kernel samples
    /// inside `range`.
    fn cluster_pairs(&self, (lo, hi): (u64, u64)) -> ([(u64, u64); PC_RING], usize) {
        let mut pairs = [(0u64, 0u64); PC_RING];
        let mut n = 0usize;
        for i in 0..self.pc_len {
            let pair = (self.pcs[i], self.ras[i]);
            if !self.user[i] && (lo..=hi).contains(&pair.0) && !pairs[..n].contains(&pair) {
                pairs[n] = pair;
                n += 1;
            }
        }
        (pairs, n)
    }

    /// The idle loop recognised from compiled samples: a stationary ring
    /// whose every (PC, return address) pair in the dominant cluster is a
    /// learned signature. Returns the learned loop's range and, as
    /// [`IdleParkState::idle_cluster`] does, whether the newest samples
    /// (all but one) are signature pairs. A busy loop that exits through
    /// the same helper as the idle loop is told apart by the return address.
    fn known_cluster(&self) -> Option<((u64, u64), bool)> {
        if self.known_len == 0 {
            return None;
        }
        let range = self.stationary_range()?;
        let (pairs, n) = self.cluster_pairs(range);
        if !pairs[..n].iter().all(|&p| self.knows(p)) {
            return None;
        }
        let recent = (1..=RECENT_SAMPLES)
            .filter(|k| {
                let i = (self.pc_pos + PC_RING - k) % PC_RING;
                !self.user[i] && self.knows((self.pcs[i], self.ras[i]))
            })
            .count();
        Some((self.known_loop, recent + 1 >= RECENT_SAMPLES))
    }

    /// Remember the signature that started the current probe run as the
    /// idle loop seen through compiled code.
    fn learn(&mut self) {
        for k in 0..self.probe_sig_len {
            let pair = self.probe_sig[k];
            if self.knows(pair) {
                continue;
            }
            if self.known_len < KNOWN_PAIRS {
                self.known[self.known_len] = pair;
                self.known_len += 1;
            } else {
                self.known[self.known_next] = pair;
                self.known_next = (self.known_next + 1) % KNOWN_PAIRS;
            }
        }
        self.known_loop = self.last_loop;
        LEARNED.fetch_add(1, Ordering::Relaxed);
    }

    /// A full ring dominated by a stationary kernel cluster of too few
    /// distinct PCs: a compiled loop seen only at its region exits, or a
    /// short poll loop. What the CPU does is unknown, so neither counts as
    /// work: the idle flag and the published cluster are left as they were,
    /// and a peer's probe run can still match the cluster this CPU published
    /// from its own last run, where withdrawing it meant two CPUs had to
    /// probe at the same moment to ever park (none did: 442 parks in a
    /// 3-minute jitv2 boot).
    fn stationary_unknown(&self) -> bool {
        self.stationary_range().is_some()
    }

    /// The dominant cluster's range when the ring is stationary-unknown.
    fn stationary_range(&self) -> Option<(u64, u64)> {
        if self.pc_len < PC_RING {
            return None;
        }
        let (samples, distinct, users, range) = self.largest_cluster();
        (users <= PC_RING / 4 && samples >= CLUSTER_MIN_SAMPLES && distinct < CLUSTER_MIN_DISTINCT).then_some(range)
    }

    /// A probe batch ran and the detector decided `parked`: whether the next
    /// batch probes too. A park gives the run a fresh budget, so the batches
    /// after a wake stay interpreted while the clock handler the wake ran
    /// returns to the idle loop (that batch is not idle, and a back-off there
    /// put the CPU into compiled idle batches, 100 ms of them, every tick).
    /// A run that spends its budget without a park is over, and backs off.
    ///
    /// A run that parks within its first two batches (the ring refills in
    /// two) was started by the idle loop's own exits: their signature is
    /// learned, and from then on compiled batches are read directly
    /// ([`IdleParkState::known_cluster`]) and the run ends at its park.
    pub fn probe_done(&mut self, parked: bool) -> bool {
        PROBES.fetch_add(1, Ordering::Relaxed);
        if parked {
            if self.probe_sig_len > 0 && PROBE_RUN - self.probe_left <= 2 {
                self.learn();
            }
            self.probe_sig_len = 0;
            self.probe_backoff = 0;
            if self.knows_compiled_idle() {
                self.probe_left = 0;
                return false;
            }
            self.probe_left = PROBE_RUN;
            return true;
        }
        self.probe_left = self.probe_left.saturating_sub(1);
        if self.probe_left > 0 {
            return true;
        }
        self.probe_backoff = PROBE_BACKOFF;
        false
    }

    /// Publish this CPU's idle-looking cluster (or none) for the others.
    fn publish_cluster(core: &MipsCore, range: Option<(u64, u64)>) {
        let (lo, hi) = range.unwrap_or((0, 0));
        let slot = &IDLE_CLUSTER[core.cpu_id as usize];
        slot[0].store(lo, Ordering::Relaxed);
        slot[1].store(hi, Ordering::Relaxed);
    }

    /// Whether another CPU's published cluster lies within
    /// [`SHARED_LOOP_BYTES`] of `range`.
    fn shared_by_a_peer(core: &MipsCore, (lo, hi): (u64, u64)) -> bool {
        IDLE_CLUSTER.iter().enumerate().any(|(cpu, slot)| {
            if cpu == core.cpu_id as usize {
                return false;
            }
            let (plo, phi) = (slot[0].load(Ordering::Relaxed), slot[1].load(Ordering::Relaxed));
            phi != 0 && plo <= hi.saturating_add(SHARED_LOOP_BYTES) && lo <= phi.saturating_add(SHARED_LOOP_BYTES)
        })
    }

    /// Update the PC window and decide what to do with the current state.
    ///
    /// Returns `None` unless the CPU has stayed inside one long, narrow
    /// kernel code window for [`PC_RING`] samples with no interrupt ready to
    /// deliver — that is a genuine idle loop. With interrupts enabled and a
    /// mask bit set the spin is safe to `park`; a pending-but-masked interrupt
    /// is paced instead (see [`IdleParkState::pace`]).
    pub fn update(&mut self, core: &MipsCore) -> IdleDecision {
        // A reset restarts the cycle counter, and may boot another kernel.
        if core.hot.cycles < self.last_cycles {
            self.known_len = 0;
            self.known_next = 0;
        }
        self.last_cycles = core.hot.cycles;
        let decision = self.decide(core);
        if core.smp {
            let bit = 1 << core.cpu_id;
            if SEEN.load(Ordering::Relaxed) & bit == 0 {
                SEEN.fetch_or(bit, Ordering::Relaxed);
            }
            // A CPU stays idle through the interrupts that reset its window
            // (the clock tick); only a full window of other work clears it.
            let idle = match decision {
                IdleDecision::Park => true,
                _ => self.idle && !std::mem::take(&mut self.busy),
            };
            if idle != self.idle {
                self.idle = idle;
                if !idle {
                    self.peer_backoff = 0;
                }
                if idle {
                    IDLE.fetch_or(bit, Ordering::Relaxed);
                } else {
                    IDLE.fetch_and(!bit, Ordering::Relaxed);
                }
            }
        }
        decision
    }

    fn decide(&mut self, core: &MipsCore) -> IdleDecision {
        // A recent park woke on a line the guest could not take: keep
        // executing for the back-off window instead of re-parking a loop
        // that is waiting on device progress.
        if self.started.elapsed() < self.warmup {
            self.clear_window();
            return IdleDecision::None;
        }
        if core.hot.cycles < self.park_skip_until_cycle {
            return IdleDecision::None;
        }
        UPDATES.fetch_add(1, Ordering::Relaxed);
        let pending = core.hot.interrupts.load(Ordering::Relaxed) as u32;
        let ip = (core.cp0_cause | pending) & CAUSE_IP_MASK;
        let im = core.cp0_status & STATUS_IM_MASK;
        let interrupt_ready = (ip & im) != 0;

        // An interrupt is ready to deliver: run the guest, don't sleep.
        if interrupt_ready {
            RESET_IRQ.fetch_add(1, Ordering::Relaxed);
            self.clear_window();
            return IdleDecision::None;
        }

        // With every mask bit clear nothing can satisfy `park`'s
        // `(ip & im) != 0`, so parking here never wakes. Do not pace either:
        // an IE=0/IM=0 repeated loop is not an idle guest but a busy wait
        // (the PROM's device-poll and DELAY loops look exactly like this),
        // and sleeping it throttles boot to about one batch per slice.
        if im == 0 {
            RESET_IM0.fetch_add(1, Ordering::Relaxed);
            self.clear_window();
            return IdleDecision::None;
        }

        // IE=0: no interrupt can be delivered at all, so a park could never
        // end. Same busy-wait trap as `im == 0` above; keep executing. This
        // must come before the pending-line pace below: with IE clear the
        // guest is typically polling a device (the kernel's console printf
        // spins on the SCC's transmit-ready bit), and sleeping it waits on a
        // status change only its own next batch can observe.
        if !core.interrupts_enabled() {
            RESET_IE0.fetch_add(1, Ordering::Relaxed);
            self.clear_window();
            return IdleDecision::None;
        }

        // A pending interrupt that is masked now is still about to be
        // taken: a loop that raises and drops the spl (IRIX's splhi leaves
        // only IP6 unmasked on IP22) is mostly seen in its masked stretch,
        // and takes the line at splx a few instructions on. Parking there
        // would wait for something that only running can bring, so pace
        // instead: bounded sleeps keep the host core free without waiting on
        // a wake condition the guest is about to create.
        if ip != 0 {
            RESET_IP.fetch_add(1, Ordering::Relaxed);
            self.clear_window();
            return IdleDecision::Pace;
        }

        // Only kernel code can be parked. A user process spinning in a loop
        // (and there are such loops during IRIX boot) is busy by definition;
        // the kernel idle path is always kernel mode. This is a privilege
        // filter, not an address match, so it is image-independent.
        if self.pc_len == PC_RING {
            WINDOW_FULL.fetch_add(1, Ordering::Relaxed);
            let (samples, distinct, users, _) = self.largest_cluster();
            WINDOW_SAMPLES.store(samples as u64, Ordering::Relaxed);
            WINDOW_DISTINCT.store(distinct as u64, Ordering::Relaxed);
            if users > PC_RING / 4 {
                RESET_USER.fetch_add(1, Ordering::Relaxed);
            }
        }
        let (cluster, known) = match self.idle_cluster() {
            Some(c) => (Some(c), false),
            None => (self.known_cluster(), true),
        };
        match cluster {
            Some((range, true)) => {
                if core.smp {
                    Self::publish_cluster(core, Some(range));
                    if !Self::shared_by_a_peer(core, range) {
                        PARK_ALONE.fetch_add(1, Ordering::Relaxed);
                        self.streak = 0;
                        return IdleDecision::None;
                    }
                }
                self.streak += 1;
                if self.streak >= IDLE_STREAK {
                    self.streak = IDLE_STREAK;
                    self.parked_known = known;
                    if known {
                        KNOWN_PARKS.fetch_add(1, Ordering::Relaxed);
                    } else {
                        self.last_loop = range;
                    }
                    return IdleDecision::Park;
                }
            }
            // The idle loop still dominates the ring, but the newest batch
            // left it: a handler, or work. Neither idle nor busy yet.
            Some((_, false)) => self.streak = 0,
            None => {
                let unknown = self.stationary_unknown();
                if core.smp && !unknown {
                    Self::publish_cluster(core, None);
                }
                self.streak = 0;
                self.busy = self.pc_len == PC_RING && !unknown;
            }
        }
        IdleDecision::None
    }

    /// Park in ≤1 ms slices until an interrupt is pending (masked or not)
    /// or the CPU stops.
    ///
    /// The Count==Compare interrupt needs no special handling here: the
    /// compare timer fires on the hptimer thread and ORs IP7 into
    /// `hot.interrupts` exactly like a device line, and CP0 Count itself is
    /// virtual (materialized from the wall clock on read), so nothing has to
    /// advance it during the sleep. Only `hot.cycles` is advanced — at the
    /// 10 ns/guest-cycle host-pacing rate — so cross-thread cycle readers
    /// (Wd33c93a's deferred-interrupt spin-wait, CP0 Random) keep seeing
    /// progress. Under `ci_clock` the compare deadline is a cycles threshold
    /// instead, so the park also stops once it is crossed.
    pub fn park(&mut self, core: &mut MipsCore, running: &AtomicBool) {
        // No check that the guest armed CP0 Compare. IRIX on IP22 does, but the
        // IP28 kernel keeps time with the IOC 8254 timers and leaves Compare at
        // zero for good, which used to turn every park into a no-op (the idle
        // loop was found and never slept). Nothing here can wait forever
        // without it: each sleep is a bounded slice, an interrupt from any
        // device thread (the 8254 timers included) ends it at once, and the
        // 60 s warm-up keeps the PROM and boot at full speed.
        PARK_COUNT.fetch_add(1, Ordering::Relaxed);
        PARKER.lock()[core.cpu_id as usize] = Some(std::thread::current());
        loop {
            if !running.load(Ordering::Relaxed) {
                PARK_WAKE_OTHER.fetch_add(1, Ordering::Relaxed);
                break;
            }
            // Announce the park before the last look at the pending word: a
            // writer sets its bit and then reads PARKED, so either we see the
            // bit here or the writer sees PARKED and unparks us.
            announce_parked(core);
            let word = core.hot.interrupts.load(Ordering::SeqCst);
            if word & HOST_PARK_BIT != 0 {
                break;
            }
            let pending = word as u32;
            let ip = (core.cp0_cause | pending) & CAUSE_IP_MASK;
            // Any line, masked or not: the mask is the guest's of the moment
            // `update` sampled - the middle of a raised spl, often - and a
            // line it masks there it takes once it drops the spl, which it
            // can only do running. (Waiting for `ip & im` slept through the
            // timer for good in such a loop: an IRIX 6.5 hang.)
            if ip != 0 {
                PARK_WAKE_IRQ.fetch_add(1, Ordering::Relaxed);
                break;
            }
            // ci_clock has no hptimer — the fire point is a cycles threshold
            // checked in step()'s preamble, so stop parking once we cross it.
            #[cfg(feature = "ci_clock")]
            if core.hot.cycles >= core.count_fire_cycle {
                break;
            }

            // An idle IRIX processor polls its run queue, and another
            // processor that queues work for it sends no interrupt. While any
            // other CPU still runs, poll again after a short slice; sleeping
            // until the next clock tick stalls every hand-off (an IP35 boot
            // crawls). Once every CPU is parked, work still arrives without
            // an interrupt for this CPU: the one that takes the clock tick
            // runs the timeouts and queues the threads they wake on the
            // other CPUs' run queues, and the IP35 kernel stops an idle
            // CPU's own clock. So on a multiprocessor every slice ends the
            // park and the idle loop polls; alone, sleep on until a line.
            let peer_running = core.smp && {
                let others = SEEN.load(Ordering::Relaxed) & !(1 << core.cpu_id);
                (PARKED.load(Ordering::SeqCst) | IDLE.load(Ordering::Relaxed)) & others != others
            };
            let slice = if peer_running { peer_slice_ns() << self.peer_backoff } else { SLICE_NS };
            let t0 = Instant::now();
            // Still a bounded slice — `running` and the ci_clock threshold are
            // only polled — but an interrupt now ends it at once.
            std::thread::park_timeout(Duration::from_nanos(slice));
            let elapsed_ns = t0.elapsed().as_nanos() as u64;
            PARK_SLEEP_NS.fetch_add(elapsed_ns, Ordering::Relaxed);
            advance_parked(core, elapsed_ns);
            if peer_running {
                PARK_WAKE_PEER.fetch_add(1, Ordering::Relaxed);
                self.peer_backoff = (self.peer_backoff + 1).min(PEER_BACKOFF_MAX);
                break;
            }
            if core.smp {
                PARK_WAKE_SLICE.fetch_add(1, Ordering::Relaxed);
                break;
            }
        }
        // If the wake was for a line the guest cannot take (masked), this
        // was not an idle loop; back off so a device-poll or DELAY loop runs
        // at full speed instead of one batch per slice.
        let pending = core.hot.interrupts.load(Ordering::SeqCst) as u32;
        let ip = (core.cp0_cause | pending) & CAUSE_IP_MASK;
        let im = core.cp0_status & STATUS_IM_MASK;
        if ip != 0 && (ip & im) == 0 {
            self.park_skip_until_cycle = core.hot.cycles.saturating_add(PARK_BACKOFF_CYCLES);
            // A signature that led here was not the idle loop.
            if self.parked_known {
                self.known_len = 0;
                self.known_next = 0;
            }
        }
        // Every exit, including the ci_clock one, leaves the flag clear: a
        // stale `true` would put `wake` on the mutex for a running CPU.
        clear_parked(core);
    }

    /// Pace a repeated state whose interrupts are masked.
    ///
    /// Unlike [`IdleParkState::park`], this never waits indefinitely: it
    /// sleeps at most one slice per call, so there is no deadlock risk when
    /// nothing can wake the CPU. `hot.cycles` advances during the sleep at the host-pacing
    /// rate (`STALL_NS_PER_CYCLE`), so cross-thread cycle readers keep seeing
    /// progress and, under `ci_clock`, the Compare threshold still arrives.
    /// The sleep ends at once when `running` is cleared (monitor command /
    /// soft reset), when an interrupt becomes deliverable, when the guest
    /// re-enables interrupts (the normal park path takes over next), or —
    /// under `ci_clock` — when the compare deadline is reached.
    pub fn pace(&self, core: &mut MipsCore, running: &AtomicBool) {
        PARKER.lock()[core.cpu_id as usize] = Some(std::thread::current());
        loop {
            if !running.load(Ordering::Relaxed) {
                break;
            }
            // Announce the park before the last look at the pending word
            // (Dekker): a writer sets its bit then reads `PARKED`, so either
            // we see the bit here or the writer sees `PARKED` and unparks us
            // via [`wake`].
            announce_parked(core);
            let ie = core.interrupts_enabled();
            let word = core.hot.interrupts.load(Ordering::SeqCst);
            if word & HOST_PARK_BIT != 0 {
                break;
            }
            let pending = word as u32;
            let ip = (core.cp0_cause | pending) & CAUSE_IP_MASK;
            let im = core.cp0_status & STATUS_IM_MASK;
            if (ip & im) != 0 {
                break;
            }
            // The guest re-enabled interrupts: hand back to the run loop so
            // the next `update` can park normally.
            if ie && im != 0 {
                break;
            }
            // ci_clock: stop once the compare deadline is reached, so the run
            // loop can deliver IP7.
            #[cfg(feature = "ci_clock")]
            if core.hot.cycles >= core.count_fire_cycle {
                break;
            }
            let t0 = Instant::now();
            std::thread::park_timeout(Duration::from_nanos(SLICE_NS));
            let elapsed_ns = t0.elapsed().as_nanos() as u64;
            advance_parked(core, elapsed_ns);
            // One slice, then back to the run loop. Status cannot change
            // while this thread sleeps, so looping here would wait on a
            // condition only executing the guest can bring about.
            break;
        }
        // Every exit leaves the flag clear: a stale `true` would put `wake`
        // on the mutex for a running CPU.
        clear_parked(core);
    }
}

/// Is the CPU thread parked now? (Tests wait for it, not a guessed sleep.)
#[cfg(test)]
pub(crate) fn parked() -> bool {
    PARKED.load(Ordering::SeqCst) != 0
}

/// Tests that park take this first: their cores are all CPU 0, one
/// parked-thread slot, so two parking at once would miss their own wakes.
#[cfg(test)]
pub(crate) static PARK_TEST_LOCK: parking_lot::Mutex<()> = parking_lot::const_mutex(());

/// Runtime parking switch. `IRIS_NO_IDLE` in the environment disables it at
/// startup; the monitor's `idle on|off` overrides that for the run.
static IDLE_ENABLED: AtomicBool = AtomicBool::new(true);
static IDLE_INIT: std::sync::Once = std::sync::Once::new();

/// The multiprocessor masks `idle status` shows: CPUs seen, idle, parked.
pub fn smp_masks() -> (u8, u8, u8) {
    (SEEN.load(Ordering::Relaxed), IDLE.load(Ordering::Relaxed), PARKED.load(Ordering::Relaxed))
}

/// The peer slice, overridable with `IRIS_IDLE_PEER_US` for tuning.
fn peer_slice_ns() -> u64 {
    static NS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *NS.get_or_init(|| {
        std::env::var("IRIS_IDLE_PEER_US").ok().and_then(|v| v.parse::<u64>().ok()).map_or(PEER_SLICE_NS, |us| us * 1000)
    })
}

fn idle_init() {
    IDLE_INIT.call_once(|| {
        if std::env::var_os("IRIS_NO_IDLE").is_some() {
            IDLE_ENABLED.store(false, Ordering::Relaxed);
        }
    });
}

pub fn idle_park_enabled() -> bool {
    idle_init();
    IDLE_ENABLED.load(Ordering::Relaxed)
}

/// Enable or disable parking at runtime (monitor `idle on|off`). Enabling
/// unparks a thread already sleeping in `park`, so the change takes effect
/// at once.
pub fn set_idle_park_enabled(on: bool) {
    idle_init();
    IDLE_ENABLED.store(on, Ordering::Relaxed);
    if on {
        wake();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::mips_core::{MipsCore, PrivilegeMode, STATUS_IE, STATUS_IM_SHIFT};

    /// A kernel-mode core with interrupts enabled and the given mask, no
    /// pending lines and no back-off.
    fn idle_core(status: u32) -> MipsCore {
        let mut core = MipsCore::default();
        core.cp0_status = status;
        core.cp0_cause = 0;
        core.pc = 0xffff_ffff_880d_aedc;
        core
    }

    fn kernel_idle() -> MipsCore {
        idle_core(STATUS_IE | (1 << (STATUS_IM_SHIFT + 7)))
    }

    /// Fill the sample window with `pcs` (one sample each) for enough
    /// consecutive batches to satisfy the streak, returning the last decision.
    fn drive(st: &mut IdleParkState, core: &mut MipsCore, pcs: &[u64]) -> IdleDecision {
        let mut decision = IdleDecision::None;
        for _ in 0..IDLE_STREAK {
            for &pc in pcs {
                core.pc = pc;
                for _ in 0..SAMPLE_EVERY {
                    st.sample(core);
                }
            }
            decision = st.update(core);
        }
        decision
    }

    /// The 6.5 scheduler idle loop as the batch sampler sees it: 24 distinct
    /// PCs spread over ~470 bytes at 0x880daedc..0x880db0b4.
    fn six_five_idle_pcs() -> Vec<u64> {
        (0..PC_RING)
            .map(|i| 0xffff_ffff_880d_aedc + (i as u64) * 19)
            .collect()
    }

    #[test]
    fn update_parks_a_long_stationary_kernel_window() {
        let mut core = kernel_idle();
        let mut st = IdleParkState::ready();
        let pcs = six_five_idle_pcs();
        assert!(
            matches!(drive(&mut st, &mut core, &pcs), IdleDecision::Park),
            "the 6.5 scheduler idle path must be recognised"
        );
    }

    /// After a park the ring still holds the idle loop, but the CPU may be
    /// in the interrupt handler that ended it, or in work it found: the
    /// newest samples decide, so a handler batch is not parked and the idle
    /// loop is parked again after one batch of it.
    #[test]
    fn update_follows_the_newest_batch_after_a_park() {
        let mut core = kernel_idle();
        let mut st = IdleParkState::ready();
        let pcs = six_five_idle_pcs();
        assert!(matches!(drive(&mut st, &mut core, &pcs), IdleDecision::Park));
        // One batch in a clock handler far from the idle loop.
        for k in 0..RECENT_SAMPLES as u64 {
            core.pc = 0xffff_ffff_8810_0000 + k * 8;
            for _ in 0..SAMPLE_EVERY {
                st.sample(&core);
            }
        }
        assert!(
            matches!(st.update(&core), IdleDecision::None),
            "parked on the idle samples from before the park while the handler ran"
        );
        // One batch back in the idle loop, one helper call outside it.
        core.pc = 0xffff_ffff_8801_ba90;
        for _ in 0..SAMPLE_EVERY {
            st.sample(&core);
        }
        for &pc in &pcs[..RECENT_SAMPLES - 1] {
            core.pc = pc;
            for _ in 0..SAMPLE_EVERY {
                st.sample(&core);
            }
        }
        assert!(matches!(st.update(&core), IdleDecision::Park), "one idle batch must park again");
    }

    /// On a multiprocessor an idle-looking window parks only once another CPU
    /// is in the same loop: a busy kernel scan that looks idle runs alone.
    #[test]
    fn smp_parks_only_a_loop_another_cpu_shares() {
        let _one = PARK_TEST_LOCK.lock();
        let mut c0 = kernel_idle();
        c0.smp = true;
        c0.cpu_id = 0;
        let mut c1 = kernel_idle();
        c1.smp = true;
        c1.cpu_id = 1;
        let mut s0 = IdleParkState::ready();
        let mut s1 = IdleParkState::ready();
        let idle = six_five_idle_pcs();
        // CPU 1 alone in a kernel loop far from the idle path: not parked.
        let scan: Vec<u64> = (0..PC_RING).map(|i| 0xffff_ffff_8820_0000 + (i as u64) * 12).collect();
        assert!(matches!(drive(&mut s1, &mut c1, &scan), IdleDecision::None), "a lone kernel loop must not park");
        // CPU 0 alone in the idle loop: not yet.
        assert!(matches!(drive(&mut s0, &mut c0, &idle), IdleDecision::None), "the first idle CPU waits for a second");
        // CPU 1 reaches the idle loop: both park.
        assert!(matches!(drive(&mut s1, &mut c1, &idle), IdleDecision::Park));
        assert!(matches!(drive(&mut s0, &mut c0, &idle), IdleDecision::Park));
        // CPU 1 goes to work and withdraws its cluster: CPU 0 is alone in the
        // idle loop again and runs it.
        assert!(matches!(drive(&mut s1, &mut c1, &scan), IdleDecision::None));
        assert!(matches!(drive(&mut s0, &mut c0, &idle), IdleDecision::None));
        IdleParkState::publish_cluster(&c0, None);
        IdleParkState::publish_cluster(&c1, None);
    }

    /// A ring of one repeated kernel PC (region exits under jitv2) starts a
    /// probe run; the run lasts until the ring is full, a run that parked
    /// continues one batch at a time, and one that found nothing backs off.
    #[test]
    fn a_stationary_ring_of_few_pcs_starts_a_probe_run() {
        let mut core = kernel_idle();
        let mut st = IdleParkState::ready();
        let exits = vec![0xffff_ffff_8800_1000u64; PC_RING];
        assert!(matches!(drive(&mut st, &mut core, &exits), IdleDecision::None));
        assert!(st.wants_probe(), "one repeated exit PC is a compiled loop to look into");
        assert_eq!(st.pc_len, 0, "the run starts with an empty ring");
        // Interpreted batches of a busy loop with few PCs: the run lasts its
        // budget, then gives up and backs off.
        let mut batches = 0;
        loop {
            for &pc in &exits[..3] {
                core.pc = pc;
                for _ in 0..SAMPLE_EVERY {
                    st.sample(&core);
                }
            }
            assert!(!matches!(st.update(&core), IdleDecision::Park));
            batches += 1;
            if !st.probe_done(false) {
                break;
            }
        }
        assert_eq!(batches, PROBE_RUN, "a run is its budget long");
        // The back-off: this many compiled batches are not probed.
        for _ in 0..PROBE_BACKOFF {
            assert!(!st.wants_probe(), "a run that found no idle loop backs off");
        }
        assert!(st.wants_probe(), "and starts again after the back-off");
        // A run whose interpreted samples are the idle loop parks at once:
        // the exits that started it are learned, and the run ends there.
        let idle = six_five_idle_pcs();
        assert!(matches!(drive(&mut st, &mut core, &idle), IdleDecision::Park));
        assert!(!st.probe_done(true), "a learned signature ends the run");
        assert!(st.knows_compiled_idle());
    }

    /// A run that parks only late, or one not started by a stationary ring,
    /// learns nothing and keeps probing after its park, as before.
    #[test]
    fn a_late_park_learns_no_signature() {
        let mut core = kernel_idle();
        let mut st = IdleParkState::ready();
        let exits = vec![0xffff_ffff_8800_1000u64; PC_RING];
        assert!(matches!(drive(&mut st, &mut core, &exits), IdleDecision::None));
        assert!(st.wants_probe());
        for _ in 0..3 {
            assert!(st.probe_done(false));
        }
        assert!(matches!(drive(&mut st, &mut core, &six_five_idle_pcs()), IdleDecision::Park));
        assert!(st.probe_done(true), "no signature: the batches after the park probe on");
        assert!(!st.knows_compiled_idle());
    }

    /// Fill the ring with compiled-loop exits: each PC with its return
    /// address.
    fn drive_exits(st: &mut IdleParkState, core: &mut MipsCore, exits: &[(u64, u64)]) -> IdleDecision {
        for i in 0..PC_RING {
            let (pc, ra) = exits[i % exits.len()];
            core.pc = pc;
            core.gpr[31] = ra;
            for _ in 0..SAMPLE_EVERY {
                st.sample(core);
            }
        }
        st.update(core)
    }

    /// Once a probe run has found the idle loop behind a ring of exits, the
    /// same exits park directly, with no probe; the same exit PC reached
    /// from another caller (return address) is not the idle loop.
    #[test]
    fn a_learned_exit_signature_parks_without_a_probe() {
        let mut core = kernel_idle();
        let mut st = IdleParkState::ready();
        let helper = 0xffff_ffff_8801_ba90u64;
        let idle_ra = 0xffff_ffff_880d_af00u64;
        let exits = [(helper, idle_ra), (helper + 0x40, idle_ra)];
        assert!(matches!(drive_exits(&mut st, &mut core, &exits), IdleDecision::None));
        assert!(st.wants_probe());
        assert!(matches!(drive(&mut st, &mut core, &six_five_idle_pcs()), IdleDecision::Park));
        assert!(!st.probe_done(true));
        let before = KNOWN_PARKS.load(Ordering::Relaxed);
        assert!(
            matches!(drive_exits(&mut st, &mut core, &exits), IdleDecision::Park),
            "the learned exits are the idle loop"
        );
        assert!(KNOWN_PARKS.load(Ordering::Relaxed) > before);
        assert!(!st.wants_probe(), "a known ring needs no probe");
        // A busy loop calling the same helper from elsewhere.
        let busy = [(helper, 0xffff_ffff_8820_0040u64), (helper + 0x40, 0xffff_ffff_8820_0040u64)];
        assert!(matches!(drive_exits(&mut st, &mut core, &busy), IdleDecision::None));
        assert!(st.wants_probe(), "an unknown caller is probed");
    }

    /// A reset (the cycle counter starting again) forgets the signatures.
    #[test]
    fn a_reset_forgets_the_learned_signatures() {
        let mut core = kernel_idle();
        core.hot.cycles = 1_000_000;
        let mut st = IdleParkState::ready();
        let exits = [(0xffff_ffff_8801_ba90u64, 0xffff_ffff_880d_af00u64)];
        drive_exits(&mut st, &mut core, &exits);
        assert!(st.wants_probe());
        drive(&mut st, &mut core, &six_five_idle_pcs());
        st.probe_done(true);
        assert!(st.knows_compiled_idle());
        core.hot.cycles = 10;
        st.update(&core);
        assert!(!st.knows_compiled_idle());
    }

    /// The real 6.5 idle path calls helpers far from the scheduler loop
    /// (`spl` at 0x8801_03xx, the ll/sc lock at 0x8801_ba90). Those samples
    /// form their own small clusters and must not disqualify the loop.
    #[test]
    fn update_parks_a_dominant_cluster_with_distant_helper_calls() {
        let mut core = kernel_idle();
        let mut st = IdleParkState::ready();
        let mut pcs = six_five_idle_pcs();
        pcs[3] = 0xffff_ffff_8801_ba90;
        pcs[7] = 0xffff_ffff_8801_03_00;
        pcs[11] = 0xffff_ffff_8801_ba98;
        pcs[15] = 0xffff_ffff_8801_0318;
        assert!(
            matches!(drive(&mut st, &mut core, &pcs), IdleDecision::Park),
            "distant helper calls must not hide the idle loop"
        );
    }

    #[test]
    fn update_declines_a_short_device_poll_loop() {
        // `lbu v1,0(a3); andi; bne -3` waiting on a device bit: 12 bytes,
        // three PCs, and it needs executed instructions, not interrupts.
        let mut core = kernel_idle();
        let mut st = IdleParkState::ready();
        let poll = [0xffff_ffff_8805_604c, 0xffff_ffff_8805_6050, 0xffff_ffff_8805_6054];
        let pcs: Vec<u64> = (0..PC_RING).map(|i| poll[i % poll.len()]).collect();
        assert!(
            matches!(drive(&mut st, &mut core, &pcs), IdleDecision::None),
            "a short device-poll loop must keep executing"
        );
    }

    #[test]
    fn update_declines_a_two_instruction_delay_loop() {
        // `bgtz a0,-1; subu a0,a0,a1` (IRIX DELAY): 8 bytes, two PCs.
        let mut core = kernel_idle();
        let mut st = IdleParkState::ready();
        let pcs: Vec<u64> = (0..PC_RING)
            .map(|i| if i % 2 == 0 { 0xffff_ffff_8801_9fb0 } else { 0xffff_ffff_8801_9fb4 })
            .collect();
        assert!(
            matches!(drive(&mut st, &mut core, &pcs), IdleDecision::None),
            "an instruction-counted DELAY loop must not be parked"
        );
    }

    #[test]
    fn update_declines_a_user_mode_loop() {
        // Same shape as the idle window, but user code: a user process
        // spinning is busy by definition.
        let mut core = kernel_idle();
        core.cp0_status |= 2 << crate::cpu::mips_core::STATUS_KSU_SHIFT; // user
        assert_eq!(core.get_privilege_mode(), PrivilegeMode::User);
        let mut st = IdleParkState::ready();
        assert!(
            matches!(drive(&mut st, &mut core, &six_five_idle_pcs()), IdleDecision::None),
            "user-mode loops are never parked"
        );
    }

    #[test]
    fn update_declines_a_window_that_keeps_moving() {
        // Ordinary kernel execution: PCs spread over more than one page, so
        // the ring never settles into a stationary window.
        let mut core = kernel_idle();
        let mut st = IdleParkState::ready();
        let pcs: Vec<u64> = (0..PC_RING)
            .map(|i| 0xffff_ffff_880d_aedc + (i as u64) * 0x400)
            .collect();
        assert!(
            matches!(drive(&mut st, &mut core, &pcs), IdleDecision::None),
            "a moving PC window is not an idle loop"
        );
    }

    #[test]
    fn update_declines_to_park_while_a_masked_interrupt_is_pending() {
        // IRIX's splhi on IP22 leaves only IP6 (the bus error) unmasked, and
        // a loop that raises and drops the spl is sampled mostly in its
        // masked stretch. With the timer's IP4 already pending there, the
        // guest takes it a few instructions later at splx: nothing to wait
        // for; pace rather than park.
        use crate::cpu::mips_core::CAUSE_IP4;
        let mut core = idle_core(STATUS_IE | (1 << (STATUS_IM_SHIFT + 6)));
        core.hot.interrupts.store(CAUSE_IP4 as u64, Ordering::SeqCst);
        let mut st = IdleParkState::ready();
        assert!(
            matches!(st.update(&core), IdleDecision::Pace),
            "a pending interrupt is about to be taken"
        );
        assert!(
            matches!(st.update(&core), IdleDecision::Pace),
            "... however often the state repeats"
        );
    }

    #[test]
    fn update_declines_to_park_or_pace_when_every_interrupt_is_masked() {
        // IE set but IM zero: the guest would take an interrupt, but none can
        // be delivered, so the wait `park` performs can never end. This is
        // also what a PROM device-poll or DELAY loop looks like; pacing it
        // throttles boot to one batch per slice, so keep executing.
        let mut core = idle_core(STATUS_IE);
        let mut st = IdleParkState::ready();
        assert!(
            matches!(drive(&mut st, &mut core, &six_five_idle_pcs()), IdleDecision::None),
            "IM == 0 makes park's wake condition unsatisfiable and pacing unsafe"
        );
    }

    #[test]
    fn update_declines_to_pace_a_repeated_state_with_interrupts_disabled() {
        // IE clear: no interrupt can be taken at all. A repeated IE=0 state
        // is a busy wait, not an idle guest.
        let mut core = idle_core(0);
        let mut st = IdleParkState::ready();
        assert!(
            matches!(drive(&mut st, &mut core, &six_five_idle_pcs()), IdleDecision::None),
            "IE == 0 must not sleep"
        );
    }

    #[test]
    fn update_returns_none_when_an_interrupt_is_ready_to_deliver() {
        // Even a stationary window must not sleep while an unmasked interrupt
        // is pending — the guest should take it.
        let mut core = kernel_idle();
        core.cp0_cause = crate::cpu::mips_core::CAUSE_IP7;
        let mut st = IdleParkState::ready();
        assert!(
            matches!(drive(&mut st, &mut core, &six_five_idle_pcs()), IdleDecision::None),
            "a ready unmasked interrupt means do not sleep"
        );
    }

    /// Pacing returns at once when `running` is cleared (monitor command /
    /// soft reset) — no sleep, no cycle advance.
    #[test]
    fn pace_returns_when_not_running() {
        // `pace` publishes this thread in the one global parker slot even when
        // it returns at once; take the lock so it cannot steal a concurrent
        // park test's slot.
        let _one = PARK_TEST_LOCK.lock();
        let mut core = MipsCore::default();
        let running = AtomicBool::new(false);
        let before = core.hot.cycles;
        let st = IdleParkState::default();
        st.pace(&mut core, &running);
        assert_eq!(core.hot.cycles, before, "pace must not sleep when not running");
    }

    /// Pacing returns at once when an interrupt becomes deliverable.
    #[test]
    fn pace_returns_when_an_interrupt_is_deliverable() {
        let _one = PARK_TEST_LOCK.lock();
        let mut core = MipsCore::default();
        core.cp0_status = STATUS_IE | (1 << (STATUS_IM_SHIFT + 7));
        core.cp0_cause = crate::cpu::mips_core::CAUSE_IP7; // unmasked IP7 pending
        let running = AtomicBool::new(true);
        let before = core.hot.cycles;
        let st = IdleParkState::default();
        st.pace(&mut core, &running);
        assert_eq!(
            core.hot.cycles,
            before,
            "pace must not sleep while an interrupt is deliverable"
        );
    }

    /// A pending line the guest has masked, with IE clear, is the guest
    /// polling a device with interrupts off (the kernel's console printf
    /// spins on the SCC's transmit-ready bit). Sleeping it freezes the
    /// guest: the bit it waits on only changes under the polling code.
    #[test]
    fn update_keeps_a_polling_guest_running_when_a_masked_line_is_pending() {
        use crate::cpu::mips_core::CAUSE_IP2;
        let mut core = idle_core(1 << (STATUS_IM_SHIFT + 6)); // IM6 only, IE clear
        core.hot.interrupts.store(CAUSE_IP2 as u64, Ordering::SeqCst);
        let mut st = IdleParkState::ready();
        assert!(matches!(st.update(&core), IdleDecision::None));
    }

    /// `pace` returns after one slice even when nothing wakes it, so the run
    /// loop gets to execute the guest again.
    #[test]
    fn pace_sleeps_one_slice_and_returns() {
        let _one = PARK_TEST_LOCK.lock();
        let mut core = MipsCore::default();
        core.cp0_status = 0;
        let running = AtomicBool::new(true);
        let t0 = Instant::now();
        IdleParkState::ready().pace(&mut core, &running);
        assert!(t0.elapsed() < Duration::from_millis(100), "pace must not wait for a wake");
        assert!(core.hot.cycles > 0);
    }

    /// Pacing advances `hot.cycles` while it sleeps, so cross-thread cycle
    /// readers keep seeing progress.
    #[test]
    fn pace_advances_guest_cycles_while_masked() {
        let _one = PARK_TEST_LOCK.lock();
        let mut core = MipsCore::default();
        core.cp0_status = 0; // interrupts masked: the pace path
        let running = std::sync::Arc::new(AtomicBool::new(true));
        let stop = running.clone();
        let before = core.hot.cycles;
        let stopper = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(5));
            stop.store(false, Ordering::SeqCst);
            wake();
        });
        IdleParkState::ready().pace(&mut core, &running);
        stopper.join().unwrap();
        assert!(
            core.hot.cycles > before,
            "pace must advance guest cycles while it sleeps"
        );
    }
}

#[cfg(test)]
mod wake_tests {
    use super::*;
    use crate::cpu::mips_core::{MipsCore, CAUSE_IP7, STATUS_IE, STATUS_IM_SHIFT};
    use std::sync::atomic::AtomicU64;
    use std::sync::Arc;

    /// `park` holds `&mut MipsCore`; the interrupt writers reach the same word
    /// through `MipsCpu::interrupts_ptr`, which is what this models.
    struct InterruptsPtr(*const AtomicU64);
    unsafe impl Send for InterruptsPtr {}

    /// Time from an interrupt bit being set on another thread to `park`
    /// returning on the CPU thread. `raise_after` picks where in a slice the
    /// interrupt lands.
    fn delivery_latency(raise_after: Duration) -> Duration {
        let _one = PARK_TEST_LOCK.lock();
        let mut core = MipsCore::default();
        core.cp0_status = STATUS_IE | (1 << (STATUS_IM_SHIFT + 7));
        core.cp0_compare = 1;
        let ptr = InterruptsPtr(&core.hot.interrupts as *const AtomicU64);
        let running = Arc::new(AtomicBool::new(true));
        let raised = Arc::new(parking_lot::Mutex::new(None::<Instant>));
        let raised_tx = raised.clone();

        let raiser = std::thread::spawn(move || {
            let p = ptr;
            std::thread::sleep(raise_after);
            *raised_tx.lock() = Some(Instant::now());
            unsafe { &*p.0 }.fetch_or(CAUSE_IP7 as u64, Ordering::SeqCst);
            wake();
        });

        let mut st = IdleParkState::default();
        st.park(&mut core, &running);
        let returned = Instant::now();
        raiser.join().unwrap();
        let at = raised.lock().expect("raiser set the bit");
        returned - at
    }

    /// One pass over the slice. Sweeping matters: a single sample can be fast
    /// by luck, on an interrupt that landed just before a slice boundary.
    fn worst_over_a_slice() -> Duration {
        (0..5)
            .map(|i| delivery_latency(Duration::from_micros(2_100 + i * 200)))
            .max()
            .unwrap()
    }

    /// IRIX on IP28 never arms CP0 Compare (it uses the 8254 timers), so a
    /// park must work, and end on a device interrupt, with Compare at zero.
    #[test]
    fn parks_and_wakes_with_compare_unarmed() {
        let _one = PARK_TEST_LOCK.lock();
        let mut core = MipsCore::default();
        core.cp0_status = STATUS_IE | (1 << (STATUS_IM_SHIFT + 3));
        assert_eq!(core.cp0_compare, 0, "Compare is zero out of reset and stays so");
        let ptr = InterruptsPtr(&core.hot.interrupts as *const AtomicU64);
        let running = Arc::new(AtomicBool::new(true));
        let stop = running.clone();
        let before = PARK_COUNT.load(Ordering::Relaxed);
        let (tx, rx) = std::sync::mpsc::channel();
        let raiser = std::thread::spawn(move || {
            let p = ptr;
            let t0 = Instant::now();
            while !parked() && t0.elapsed() < Duration::from_secs(2) {
                std::thread::yield_now();
            }
            unsafe { &*p.0 }.fetch_or(crate::cpu::mips_core::CAUSE_IP4 as u64, Ordering::SeqCst);
            wake();
            if rx.recv_timeout(Duration::from_secs(2)).is_err() {
                stop.store(false, Ordering::SeqCst);
                wake();
                return false;
            }
            true
        });
        IdleParkState::default().park(&mut core, &running);
        let _ = tx.send(());
        assert!(raiser.join().unwrap(), "the park never saw the interrupt");
        assert!(PARK_COUNT.load(Ordering::Relaxed) > before, "park returned without parking");
    }

    #[test]
    fn a_masked_interrupt_ends_the_park() {
        // Parked with only IP6 unmasked (the spl raised), the timer's IP4
        // arrives: masked now, but taken as soon as the guest drops the spl,
        // which it can only do running. Before the fix `park` waited for
        // `ip & im` and slept through it for good; the deadline turns that
        // hang into a failure.
        use crate::cpu::mips_core::CAUSE_IP4;
        let _one = PARK_TEST_LOCK.lock();
        let mut core = MipsCore::default();
        core.cp0_status = STATUS_IE | (1 << (STATUS_IM_SHIFT + 6));
        core.cp0_compare = 1;
        let ptr = InterruptsPtr(&core.hot.interrupts as *const AtomicU64);
        let running = Arc::new(AtomicBool::new(true));
        let stop = running.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        let raiser = std::thread::spawn(move || {
            let p = ptr;
            let t0 = Instant::now();
            while !parked() && t0.elapsed() < Duration::from_secs(2) {
                std::thread::yield_now();
            }
            unsafe { &*p.0 }.fetch_or(CAUSE_IP4 as u64, Ordering::SeqCst);
            wake();
            // the deadline: release a park that never returns
            if rx.recv_timeout(Duration::from_secs(2)).is_err() {
                stop.store(false, Ordering::SeqCst);
                wake();
                return false;
            }
            true
        });
        IdleParkState::default().park(&mut core, &running);
        let _ = tx.send(());
        assert!(raiser.join().unwrap(),
            "park slept through a pending interrupt masked only by the spl");
    }

    /// A park that wakes on a masked line was a false positive: back off so
    /// a device-poll or DELAY loop keeps running instead of sleeping a slice
    /// per batch. A park woken by an unmasked line leaves parking enabled.
    #[test]
    fn a_park_woken_by_a_masked_line_backs_off() {
        use crate::cpu::mips_core::CAUSE_IP4;
        let _one = PARK_TEST_LOCK.lock();

        // Masked: only IP6 unmasked, IP4 pending.
        let mut core = MipsCore::default();
        core.cp0_status = STATUS_IE | (1 << (STATUS_IM_SHIFT + 6));
        core.cp0_compare = 1;
        core.hot.interrupts.store(CAUSE_IP4 as u64, Ordering::SeqCst);
        let running = AtomicBool::new(false); // return at the first check
        let mut st = IdleParkState::ready();
        st.park(&mut core, &running);
        assert!(
            matches!(st.update(&core), IdleDecision::None),
            "a masked wake must disable parking for the back-off window"
        );

        // Unmasked: IP7 unmasked and pending.
        let mut core = MipsCore::default();
        core.cp0_status = STATUS_IE | (1 << (STATUS_IM_SHIFT + 7));
        core.cp0_compare = 1;
        core.hot.interrupts.store(CAUSE_IP7 as u64, Ordering::SeqCst);
        let mut st = IdleParkState::ready();
        st.park(&mut core, &running);
        assert!(
            matches!(st.update(&core), IdleDecision::None),
            "a deliverable pending interrupt means run, not park (ring cleared)"
        );
    }

    /// The CP0 Compare deadline is an `hptimer` one-shot in a wall-clock
    /// build. A parked CPU must wake when it fires, without any other thread
    /// raising a line: the callback sets IP7 in `hot.interrupts`, and `park`
    /// exits on any pending line. The Compare deadline is the hptimer
    /// one-shot armed by `schedule_compare_timer`.
    #[cfg(not(feature = "ci_clock"))]
    #[test]
    fn an_armed_compare_timer_ends_the_park() {
        let _one = PARK_TEST_LOCK.lock();
        let mut core = MipsCore::default();
        core.cp0_status = STATUS_IE | (1 << (STATUS_IM_SHIFT + 7));
        core.set_timer_manager(Arc::new(crate::hptimer::TimerManager::new()));
        // Arm Compare a few Count ticks ahead: about 1 us at 33 MHz. park's
        // own guard requires a non-zero Compare.
        let now = core.count_now();
        core.write_cp0(11, now.wrapping_add(33) as u64);
        let running = AtomicBool::new(true);

        let t0 = Instant::now();
        IdleParkState::default().park(&mut core, &running);
        let elapsed = t0.elapsed();
        assert_ne!(
            core.hot.interrupts.load(Ordering::SeqCst) as u32 & CAUSE_IP7,
            0,
            "the Compare one-shot must have raised IP7"
        );
        assert!(
            elapsed < Duration::from_millis(100),
            "park waited {elapsed:?} for the armed Compare timer"
        );
    }

    /// Under `ci_clock` there is no host timer: the Compare deadline is a
    /// `hot.cycles` threshold, and the park must stop once the advancing cycle
    /// counter crosses it so the run loop can deliver IP7.
    #[cfg(feature = "ci_clock")]
    #[test]
    fn ci_clock_cycle_compare_threshold_ends_the_park() {
        let _one = PARK_TEST_LOCK.lock();
        let mut core = MipsCore::default();
        core.cp0_status = STATUS_IE | (1 << (STATUS_IM_SHIFT + 7));
        core.cp0_compare = 1;
        core.count_fire_cycle = core.hot.cycles + 300_000; // 3 ms at 10 ns/cycle
        let running = AtomicBool::new(true);

        let t0 = Instant::now();
        IdleParkState::default().park(&mut core, &running);
        let elapsed = t0.elapsed();
        assert!(
            core.hot.cycles >= 300_000,
            "park returned before the cycle threshold: cycles={}",
            core.hot.cycles
        );
        assert!(
            elapsed < Duration::from_millis(200),
            "park waited {elapsed:?} for the cycle threshold"
        );
    }

    #[test]
    fn an_interrupt_ends_the_park_without_waiting_out_the_slice() {
        // Without `wake` every sweep contains a phase that waits out most of a
        // slice, so no number of retries makes this pass; with it, each phase
        // costs a thread wakeup (~6 us here). The retries are only so a
        // scheduling stall on a loaded CI runner does not fail the build.
        const LIMIT: Duration = Duration::from_micros(300);
        let mut seen = Vec::new();
        for _ in 0..3 {
            let worst = worst_over_a_slice();
            if worst < LIMIT {
                return;
            }
            seen.push(worst);
        }
        panic!(
            "worst-phase latency {seen:?} over 3 sweeps, all above {LIMIT:?}; \
             a slice is {:?}",
            Duration::from_nanos(SLICE_NS)
        );
    }
}

#[cfg(test)]
mod retired_tests {
    use super::*;
    use crate::cpu::mips_core::{CyclesPtr, MipsCore};

    fn ptr(core: &MipsCore) -> CyclesPtr {
        CyclesPtr::with_parked(&core.hot.cycles, &core.hot.parked_cycles)
    }

    /// Parked time moves the guest-visible cycle counter but is not retired
    /// instructions, so the MIPS figure must not see it.
    #[test]
    fn parked_time_is_not_retired() {
        let mut core = MipsCore::default();
        core.hot.cycles = 1_000;
        let p = ptr(&core);
        assert_eq!(p.retired(), 1_000);
        advance_parked(&mut core, 1_000_000);
        assert_eq!(core.hot.cycles, 1_000 + 1_000_000 / STALL_NS_PER_CYCLE);
        assert_eq!(p.get(), core.hot.cycles, "timing readers still see the park");
        assert_eq!(p.retired(), 1_000, "a park retires nothing");
        core.hot.cycles += 50;
        assert_eq!(p.retired(), 1_050);
    }

    /// The real pacing path, run until a stopper ends it.
    #[test]
    fn pace_leaves_retired_alone() {
        let _one = PARK_TEST_LOCK.lock();
        let mut core = MipsCore::default();
        core.cp0_status = 0; // interrupts masked: the pace path
        core.hot.cycles = 7;
        let p = ptr(&core);
        let running = std::sync::Arc::new(AtomicBool::new(true));
        let stop = running.clone();
        let stopper = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(5));
            stop.store(false, Ordering::SeqCst);
            wake();
        });
        IdleParkState::ready().pace(&mut core, &running);
        stopper.join().unwrap();
        assert!(p.get() > 7, "the pace moved the cycle counter");
        assert_eq!(p.retired(), 7);
    }

    /// A hard reset starts the cycle counter again from 0; the parked part
    /// must go with it, or the bars read 0 MIPS after a reboot for as long
    /// as the previous run spent parked.
    #[test]
    fn a_reset_starts_the_retired_count_again() {
        let mut core = MipsCore::default();
        core.hot.cycles = 1_000;
        advance_parked(&mut core, 1_000_000);
        let p = ptr(&core);
        core.reset(false);
        assert_eq!(p.get(), 0);
        core.hot.cycles += 50;
        assert_eq!(p.retired(), 50);
    }

    #[test]
    fn plain_pointer_has_no_parked_time() {
        let mut core = MipsCore::default();
        core.hot.cycles = 9;
        let p = CyclesPtr::new(&core.hot.cycles);
        advance_parked(&mut core, 10_000);
        assert_eq!(p.retired(), p.get());
        assert_eq!(CyclesPtr::dangling().retired(), 0);
    }
}
