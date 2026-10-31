//! Per-CPU debugging state the execution paths fill in: the undo buffer, the
//! traceback ring, the idle profiler and the breakpoint types.

use super::*;

/// Default undo buffer capacity: 1M (2^20) instructions. `UndoBuffer` can be
/// resized larger at runtime (`undo resize <n>`) — this is only the size a
/// fresh `UndoBuffer::new()` starts at.
#[cfg(feature = "developer")]
pub(super) const UNDO_BUFFER_SIZE: usize = 1 << 20;

/// Memory write operation for undo tracking
#[cfg(feature = "developer")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct MemoryWrite {
    pub(super) virt_addr: u64,
    pub(super) phys_addr: u64,
    pub(super) old_value: u64,
    pub(super) size: usize,
}

/// CPU state snapshot for undo - includes core state and metadata
#[cfg(feature = "developer")]
#[derive(Debug, Clone, PartialEq, Default)]
pub(super) struct CpuSnapshot {
    // General Purpose Registers
    pub(super) gpr: [u64; 32],

    // Special Registers
    pub(super) pc: u64,
    pub(super) hi: u64,
    pub(super) lo: u64,

    // LL/SC state
    pub(super) llbit: bool,
    pub(super) lladdr: u64,

    // CP0 registers
    pub(super) cp0_index: u32,
    pub(super) cp0_random: u32,
    pub(super) cp0_entrylo0: u64,
    pub(super) cp0_entrylo1: u64,
    pub(super) cp0_context: u64,
    pub(super) cp0_pagemask: u64,
    pub(super) cp0_wired: u32,
    pub(super) cp0_badvaddr: u64,
    pub(super) cp0_count: u64,
    pub(super) cp0_entryhi: u64,
    pub(super) cp0_compare: u64,
    pub(super) cp0_status: u32,
    pub(super) cp0_cause: u32,
    pub(super) cp0_epc: u64,
    pub(super) cp0_prid: u32,
    pub(super) cp0_config: u32,
    pub(super) cp0_watchlo: u32,
    pub(super) cp0_watchhi: u32,
    pub(super) cp0_xcontext: u64,
    pub(super) cp0_ecc: u32,
    pub(super) cp0_cacheerr: u32,
    pub(super) cp0_taglo: u64,
    pub(super) cp0_taghi: u32,
    pub(super) cp0_errorepc: u64,

    // CP1 (FPU) registers
    pub(super) fpr: [u64; 32],
    pub(super) fpu_fir: u32,
    pub(super) fpu_fccr: u32,
    pub(super) fpu_fexr: u32,
    pub(super) fpu_fenr: u32,
    pub(super) fpu_fcsr: u32,

    // Execution state
    pub(super) running: bool,
    pub(super) halted: bool,

    // Delay slot tracking from executor
    pub(super) in_delay_slot: bool,
    pub(super) delay_slot_target: u64,

    // Memory writes that occurred during this instruction
    pub(super) memory_writes: Vec<MemoryWrite>,
}

/// Circular undo buffer for CPU debugging
#[cfg(feature = "developer")]
pub(super) struct UndoBuffer {
    pub(super) enabled: bool,
    pub(super) snapshots: Vec<Option<CpuSnapshot>>,
    pub(super) head: usize,  // Next write position
    pub(super) count: usize, // Number of valid snapshots
}

#[cfg(feature = "developer")]
impl UndoBuffer {
    pub(super) fn new() -> Self {
        Self {
            enabled: false,
            snapshots: vec![None; UNDO_BUFFER_SIZE],
            head: 0,
            count: 0,
        }
    }

    pub(super) fn enable(&mut self) {
        self.enabled = true;
    }

    pub(super) fn disable(&mut self) {
        self.enabled = false;
    }

    pub(super) fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub(super) fn push(&mut self, snapshot: CpuSnapshot) {
        self.push_get_idx(snapshot);
    }

    pub(super) fn push_get_idx(&mut self, snapshot: CpuSnapshot) -> usize {
        let capacity = self.snapshots.len();
        let idx = self.head;
        self.snapshots[idx] = Some(snapshot);
        self.head = (self.head + 1) % capacity;
        if self.count < capacity {
            self.count += 1;
        }
        idx
    }

    pub(super) fn can_undo(&self, steps: usize) -> bool {
        self.enabled && steps <= self.count
    }

    pub(super) fn get(&self, steps_back: usize) -> Option<&CpuSnapshot> {
        if steps_back == 0 || steps_back > self.count {
            return None;
        }

        let capacity = self.snapshots.len();
        let index = if self.head >= steps_back {
            self.head - steps_back
        } else {
            capacity - (steps_back - self.head)
        };

        self.snapshots[index].as_ref()
    }

    pub(super) fn pop(&mut self) {
        if self.count == 0 { return; }
        let capacity = self.snapshots.len();
        self.head = if self.head == 0 { capacity - 1 } else { self.head - 1 };
        self.snapshots[self.head] = None;
        self.count -= 1;
    }

    pub(super) fn clear(&mut self) {
        self.head = 0;
        self.count = 0;
        for snapshot in &mut self.snapshots {
            *snapshot = None;
        }
    }

    /// Current capacity (max instructions the buffer can hold before it
    /// starts overwriting the oldest entries).
    pub(super) fn capacity(&self) -> usize {
        self.snapshots.len()
    }

    /// Grow the buffer's capacity to at least `new_capacity`, preserving
    /// existing snapshots at their current logical undo depth. No-op if
    /// already at least that large (never shrinks — shrinking would need to
    /// discard the oldest entries, which callers can already get via
    /// `clear()` if they actually want a reset). Implemented by draining
    /// snapshots out oldest-to-newest into a fresh, larger buffer rather
    /// than resizing the ring in place, since the ring's wraparound
    /// arithmetic (`head`/`capacity`) has no simple in-place growth that
    /// preserves logical ordering.
    pub(super) fn resize(&mut self, new_capacity: usize) {
        if new_capacity <= self.snapshots.len() {
            return;
        }
        let old_count = self.count;
        // Oldest-first: steps_back == old_count is the oldest entry still held.
        let mut ordered: Vec<Option<CpuSnapshot>> = (1..=old_count)
            .rev()
            .map(|steps_back| self.get(steps_back).cloned())
            .collect();
        ordered.resize_with(new_capacity, || None);
        self.snapshots = ordered;
        self.head = old_count % new_capacity;
        self.count = old_count;
    }
}

#[cfg(all(test, feature = "developer"))]
mod undo_buffer_tests {
    use super::*;

    pub(super) fn snap(pc: u64) -> CpuSnapshot {
        CpuSnapshot { pc, ..Default::default() }
    }

    #[test]
    pub(super) fn resize_preserves_order_below_capacity() {
        let mut buf = UndoBuffer { enabled: true, snapshots: vec![None; 4], head: 0, count: 0 };
        for pc in [1, 2, 3] {
            buf.push(snap(pc));
        }
        buf.resize(8);
        assert_eq!(buf.capacity(), 8);
        assert_eq!(buf.count, 3);
        // steps_back=1 is the most recently pushed (pc=3), steps_back=3 the oldest (pc=1).
        assert_eq!(buf.get(1).unwrap().pc, 3);
        assert_eq!(buf.get(2).unwrap().pc, 2);
        assert_eq!(buf.get(3).unwrap().pc, 1);
    }

    /// The interesting case: resize while the ring has already wrapped
    /// around (head has cycled past 0), so a naive `Vec::resize` on the raw
    /// backing storage (instead of `resize`'s oldest-to-newest drain) would
    /// scramble logical ordering — entries physically after `head` in the
    /// old backing array are actually the *oldest* ones, not contiguous
    /// with what looks adjacent by raw index.
    #[test]
    pub(super) fn resize_preserves_order_after_wraparound() {
        let mut buf = UndoBuffer { enabled: true, snapshots: vec![None; 3], head: 0, count: 0 };
        // Capacity 3, push 5 -> wraps around twice; only the last 3 (3,4,5) survive.
        for pc in [1, 2, 3, 4, 5] {
            buf.push(snap(pc));
        }
        assert_eq!(buf.count, 3);
        assert_eq!(buf.get(1).unwrap().pc, 5);
        assert_eq!(buf.get(2).unwrap().pc, 4);
        assert_eq!(buf.get(3).unwrap().pc, 3);

        buf.resize(6);
        assert_eq!(buf.capacity(), 6);
        assert_eq!(buf.count, 3);
        assert_eq!(buf.get(1).unwrap().pc, 5, "most recent must still be most recent after growing past a wraparound");
        assert_eq!(buf.get(2).unwrap().pc, 4);
        assert_eq!(buf.get(3).unwrap().pc, 3);

        // And the buffer must still behave correctly for further pushes at the new capacity.
        buf.push(snap(6));
        assert_eq!(buf.count, 4);
        assert_eq!(buf.get(1).unwrap().pc, 6);
        assert_eq!(buf.get(4).unwrap().pc, 3);
    }

    #[test]
    pub(super) fn resize_to_smaller_or_equal_capacity_is_a_no_op() {
        let mut buf = UndoBuffer { enabled: true, snapshots: vec![None; 8], head: 0, count: 0 };
        buf.push(snap(1));
        buf.resize(4); // smaller than current capacity 8
        assert_eq!(buf.capacity(), 8, "resize must never shrink");
        buf.resize(8); // equal
        assert_eq!(buf.capacity(), 8);
        assert_eq!(buf.get(1).unwrap().pc, 1, "existing entries must survive a no-op resize");
    }
}

pub(super) const TRACEBACK_SIZE: usize = 1048576; // 1M entries

/// How a traceback entry's instruction was dispatched — finer-grained than a
/// plain jit/interp bool so `dt` can show exactly which arrival path ran,
/// not just "JIT or not": a divergence often depends on *which* of these an
/// instruction is (an entry word reached as a back-edge has different live
/// state than one reached externally; a fallback successor is architecturally
/// a delay slot even though its own raw bits don't look like one).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum InstrOrigin {
    /// Dispatched by the interpreter's own `step()` — includes every entry
    /// word's *external* arrival (step() records it before ever handing off
    /// to JIT-compiled code, so the JIT never double-records it).
    #[default]
    Interp,
    /// A plain (non-entry, non-slot) instruction inside a compiled region —
    /// the common case, recorded by `emit_dev_trace_bp` in the pass-2 head
    /// loop.
    Jit,
    /// A compiled region's entry word, reached via the compiled function's
    /// own dispatch-head `Switch` — a real external arrival (jumping
    /// straight into `jit_fn` for the first time this exact PC was reached
    /// this way), recorded by the compiled function itself
    /// (`dev_trace_origin::JIT_ENTRY`, emitted at each dispatch-head target —
    /// see `emit_dev_trace_bp`'s doc comment) rather than by the Rust-side
    /// caller that looked it up. Distinct from a plain `Jit` interior
    /// instruction: this IS the entry, with its own foreign-slot/exception
    /// handling, architecturally the JIT counterpart to what `Interp` meant
    /// before this variant existed (a real external dispatch landing on this
    /// exact word) — just resolved into a compiled function instead of the
    /// interpreter.
    JitEntry,
    /// The region's entry word, but reached via an *internal* in-region edge
    /// (a backward branch looping to the top of the same compiled function)
    /// rather than step()'s external dispatch — still JIT-executed, still
    /// worth distinguishing from a plain interior instruction since entry
    /// words get special foreign-slot/exception handling.
    JitEntryBackEdge,
    /// A delay slot's own instruction, inlined into its branch/jump/regjump's
    /// compiled unit (`emit_slot_semantics`) — architecturally a distinct
    /// instruction from its branch, with its own address and (for lockstep)
    /// its own verification.
    JitDelaySlot,
    /// An interpreter-fallback head (`Classify::Excluded`, kept in-region) —
    /// runs through `interp_dispatch_one`, not native codegen, but still
    /// inside a compiled region's dispatch, distinct from a plain step().
    FallbackWord,
    /// The word immediately after a fallback that is itself a branch (BC1) —
    /// architecturally a delay slot (`is_branch_fallback_successor`), reached
    /// externally the first time (from the fallback's own interpreter run).
    FallbackSuccessor,
    /// A fallback successor reached via an internal in-region back-edge,
    /// same distinction as `JitEntryBackEdge` but for a fallback successor
    /// word instead of the region's own entry word.
    FallbackSuccessorBackEdge,
}

impl InstrOrigin {
    pub(super) fn tag(self) -> &'static str {
        match self {
            InstrOrigin::Interp => "",
            InstrOrigin::Jit => " [jit]",
            InstrOrigin::JitEntry => " [jit:entry]",
            InstrOrigin::JitEntryBackEdge => " [jit:entry-backedge]",
            InstrOrigin::JitDelaySlot => " [jit:slot]",
            InstrOrigin::FallbackWord => " [fallback]",
            InstrOrigin::FallbackSuccessor => " [fallback:successor]",
            InstrOrigin::FallbackSuccessorBackEdge => " [fallback:successor-backedge]",
        }
    }

    /// Decode the `origin` code `emit_dev_trace_bp`'s codegen call sites pass
    /// (a compile-time constant baked in per call site — see
    /// `codegen::core_offset_of_dev_trace_bp_fn`'s callers). `u32`,
    /// deliberately not `u8` — see `MipsCore::dev_trace_bp_fn`'s doc comment.
    /// Any value outside the known set (shouldn't happen — every call site
    /// uses one of `dev_trace_origin`'s constants) falls back to `Jit` rather
    /// than panicking, since this runs inside an `extern "C"` hook where
    /// unwinding aborts the process.
    #[cfg(all(feature = "jitv2", feature = "developer"))]
    pub(super) fn from_u32(v: u32) -> Self {
        match v {
            0 => InstrOrigin::Interp,
            1 => InstrOrigin::Jit,
            2 => InstrOrigin::JitEntryBackEdge,
            3 => InstrOrigin::JitDelaySlot,
            4 => InstrOrigin::FallbackWord,
            5 => InstrOrigin::FallbackSuccessor,
            6 => InstrOrigin::FallbackSuccessorBackEdge,
            7 => InstrOrigin::JitEntry,
            _ => InstrOrigin::Jit,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct TracebackEntry {
    pub(super) pc: u64,
    pub(super) instr: u32,
    pub(super) origin: InstrOrigin,
}

pub(super) struct TracebackBuffer {
    pub(super) entries: Vec<TracebackEntry>,
    pub(super) head: usize,
    pub(super) count: usize,
}

impl TracebackBuffer {
    pub(super) fn new() -> Self {
        Self {
            entries: vec![TracebackEntry::default(); TRACEBACK_SIZE],
            head: 0,
            count: 0,
        }
    }

    pub(super) fn push(&mut self, pc: u64, instr: u32, origin: InstrOrigin) {
        self.entries[self.head] = TracebackEntry { pc, instr, origin };
        self.head = (self.head + 1) % TRACEBACK_SIZE;
        if self.count < TRACEBACK_SIZE {
            self.count += 1;
        }
    }

    pub(super) fn get_last(&self, n: usize) -> Vec<TracebackEntry> {
        let mut result = Vec::new();
        let count = n.min(self.count);

        for i in 0..count {
            let idx = (self.head + TRACEBACK_SIZE - 1 - i) % TRACEBACK_SIZE;
            result.push(self.entries[idx]);
        }
        result.reverse();
        result
    }
}

#[cfg(feature = "idle-pause")]
/// Per-PC sampling counters: total times this PC was the about-to-execute PC,
/// and how many of those had CPU interrupts enabled (IE=1, EXL=ERL=0).
#[derive(Clone, Copy, Default)]
pub(super) struct IdleSample {
    pub(super) count: u64,
    pub(super) ie_count: u64,
}

/// Lightweight PC-sampling histogram used to locate hot spin loops — primarily
/// the IRIX kernel idle loop, which is a tight backward branch that runs with
/// interrupts enabled and only exits via an interrupt. Disabled by default;
/// when `on` is false `step()` pays a single predictable-not-taken branch.
///
/// Workflow (the executor lock is held for the whole run, so toggling and
/// reporting require the CPU to be stopped first):
///   cpu stop; idleprof on; cont    # let the guest sit at an idle prompt
///   cpu stop; idleprof report      # dump the hottest PCs + IE%
///
#[cfg(feature = "idle-pause")]
/// The idle loop shows up as a small cluster of contiguous PCs that together
/// dominate the samples with ie% == 100.
#[derive(Default)]
pub(super) struct IdleProfiler {
    /// Sample one in every `stride` executed instructions. 0/1 = every instr.
    /// Subsampling bounds hot-path cost; the idle loop still dominates because
    /// it is by far the most frequently executed code while the system idles.
    pub(super) stride: u64,
    pub(super) counter: u64,
    pub(super) total: u64,
    pub(super) hist: std::collections::HashMap<u64, IdleSample>,
}

#[cfg(feature = "idle-pause")]
impl IdleProfiler {
    #[inline(always)]
    pub(super) fn sample(&mut self, pc: u64, ie: bool) {
        self.counter = self.counter.wrapping_add(1);
        if self.stride > 1 && self.counter % self.stride != 0 {
            return;
        }
        self.total += 1;
        let e = self.hist.entry(pc).or_default();
        e.count += 1;
        if ie {
            e.ie_count += 1;
        }
    }

    pub(super) fn reset(&mut self) {
        self.counter = 0;
        self.total = 0;
        self.hist.clear();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BpType {
    Pc        = 0,
    VirtRead  = 1,
    VirtWrite = 2,
    VirtFetch = 3,
    PhysRead  = 4,
    PhysWrite = 5,
    PhysFetch = 6,
    /// Break when a specific value is written to ANY memory address
    WriteValue = 7,
    /// Break every instruction when gpr[reg] == val (and optionally PC in range)
    RegValue   = 8,
}

/// One breakpoint as `bp list` and `bp add` show it:
/// `5: ffffffff88031040 <cmn_err> pc enabled if a0 == 3`.
pub(super) fn format_bp_line(id: usize, addr: u64, kind: BpType, enabled: bool, cond: Option<&str>,
                  symbols: &crate::cpu::mips_dis::SymbolTable) -> String {
    let mut s = format!("{}: {:016x}", id, addr);
    if matches!(kind, BpType::Pc | BpType::VirtRead | BpType::VirtWrite | BpType::VirtFetch) {
        if let Some((base, name)) = symbols.lookup(addr) {
            match addr - base {
                0 => s.push_str(&format!(" <{}>", name)),
                off => s.push_str(&format!(" <{}+{:#x}>", name, off)),
            }
        }
    }
    s.push_str(&format!(" {} {}", kind.name(), if enabled { "enabled" } else { "disabled" }));
    if let Some(c) = cond {
        s.push_str(" if ");
        s.push_str(c);
    }
    s
}

impl BpType {
    /// The name `bp list` shows, matching what `bp add` accepts.
    pub fn name(self) -> &'static str {
        match self {
            BpType::Pc => "pc",
            BpType::VirtRead => "read",
            BpType::VirtWrite => "write",
            BpType::VirtFetch => "fetch",
            BpType::PhysRead => "pread",
            BpType::PhysWrite => "pwrite",
            BpType::PhysFetch => "pfetch",
            BpType::WriteValue => "write-value",
            BpType::RegValue => "reg-value",
        }
    }
}

pub struct Breakpoint {
    pub id: usize,
    pub addr: u64,
    pub kind: BpType,
    pub enabled: bool,
    /// Optional condition expression (evaluated when breakpoint is hit)
    pub condition: Option<Expr>,
    /// The condition as typed, for `bp list` (`condition` is constant-folded).
    pub condition_src: Option<String>,
    /// Times this breakpoint matched (condition true), stopped or not.
    pub hits: u64,
    /// Matches still to let pass before stopping (`bp ignore`).
    pub ignore: u64,
}
