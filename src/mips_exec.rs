// MIPS Execution Engine

use crate::mips_core::*;
use crate::mips_isa::*;
use crate::traits::*;
use crate::mips_tlb::*;
use crate::mips_cache_v2::*;
use crate::devlog::{LogModule, devlog_mask, devlog_is_active};
use crate::mips_dis;
use crate::physical::{HIMEM_BASE, HIMEM_END, LOMEM_BASE, LOMEM_END};
use std::fmt::Write as FmtWrite;
use crate::mips_dis::SymbolTable;
use std::sync::Arc;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU32, Ordering};
use std::thread;
use std::io::Write;
use std::time::Duration;
use crate::exp::{self, Expr, RegTarget};
use crate::snapshot::{get_field, u64_slice_to_toml, load_u64_slice, toml_u64, toml_u32, toml_bool, hex_u64, hex_u32};

// LogModule::Mips bitmask categories
pub const MIPS_LOG_INSN: u32 = 0x0001; // per-instruction disassembly trace
pub const MIPS_LOG_TLB:  u32 = 0x0002; // TLB read/write/probe
pub const MIPS_LOG_MEM:  u32 = 0x0004; // uncached memory accesses
pub const MIPS_LOG_FPU:  u32 = 0x0008; // FP compare/condmove/convert operand+result trace

#[cfg(feature = "developer")]
#[inline(always)]
fn mips_log(bit: u32) -> bool {
    devlog_is_active(LogModule::Mips) && (devlog_mask(LogModule::Mips) & bit) != 0
}

// Without `developer`, dlog_dev! is a no-op, so callers gate on this constant `false`
// instead of paying an atomic load per call site to check a flag that can never fire.
#[cfg(not(feature = "developer"))]
#[inline(always)]
fn mips_log(_bit: u32) -> bool {
    false
}

// Exception codes (from MIPS R4000 documentation)
pub const EXC_INT: u32 = 0;       // Interrupt
pub const EXC_MOD: u32 = 1;       // TLB modification exception
pub const EXC_TLBL: u32 = 2;      // TLB exception (load or instruction fetch)
pub const EXC_TLBS: u32 = 3;      // TLB exception (store)
pub const EXC_ADEL: u32 = 4;      // Address error exception (load or instruction fetch)
pub const EXC_ADES: u32 = 5;      // Address error exception (store)
pub const EXC_IBE: u32 = 6;       // Bus error exception (instruction fetch)
pub const EXC_DBE: u32 = 7;       // Bus error exception (data reference: load or store)
pub const EXC_SYS: u32 = 8;       // Syscall exception
pub const EXC_BP: u32 = 9;        // Breakpoint exception
pub const EXC_RI: u32 = 10;       // Reserved instruction exception
pub const EXC_CPU: u32 = 11;      // Coprocessor Unusable exception
pub const EXC_OV: u32 = 12;       // Arithmetic Overflow exception
pub const EXC_TR: u32 = 13;       // Trap exception
pub const EXC_FPE: u32 = 15;      // Floating point exception

// FCSR (FPU Control/Status Register, CP1 reg 31) bit fields
const FCSR_CI: u32 = 0x00001000;  // Cause: inexact
const FCSR_CU: u32 = 0x00002000;  // Cause: underflow
const FCSR_CO: u32 = 0x00004000;  // Cause: overflow
const FCSR_CZ: u32 = 0x00008000;  // Cause: divide-by-zero
const FCSR_CV: u32 = 0x00010000;  // Cause: invalid operation
const FCSR_CE: u32 = 0x00020000;  // Cause: unimplemented operation
const FCSR_CM: u32 = 0x0001f000;  // Cause mask (V,Z,O,U,I — excludes CE)
const FCSR_EM: u32 = 0x00000f80;  // Enable mask (V,Z,O,U,I)
const FCSR_FM: u32 = 0x0000007c;  // Flag mask (V,Z,O,U,I — sticky)
pub const EXC_WATCH: u32 = 23;    // Reference to WatchHi/WatchLo address
pub const EXC_VCEI: u32 = 14;     // Virtual Coherency Exception (Instruction)
pub const EXC_VCED: u32 = 31;     // Virtual Coherency Exception (Data)

pub const CONFIG_CM: u32 = 31;    // Master checker mode
pub const CONFIG_EC: u32 = 28;    // 3 bits, clock ratio  0 - 2, 1 - 3...
pub const CONFIG_EP: u32 = 24;    // 4 bits transmit data pattern for writeback
pub const CONFIG_SB: u32 = 22;    // 2 bits secondary cache block size: 1=8 words (32B) on R5K
pub const CONFIG_SS: u32 = 21;    // R4K: 1 bit split secondary cache mode
pub const CONFIG_TR_SS: u32 = 20; // Triton: 2 bits secondary cache size [21:20]: 00=512KB 01=1MB 10=2MB 11=none
pub const CONFIG_SW: u32 = 20;    // secondary cache port width 0 - 128bit, 1 - 64bit
pub const CONFIG_EW: u32 = 18;    // 2 bits system port width 0 - 64 bit, 1 - 32 bit
pub const CONFIG_SC: u32 = 17;    // secondary cache present 0 - present, 1 - absent
pub const CONFIG_SM: u32 = 16;    // dirty shared coherency state 0 - enabled, 1 - disabled
pub const CONFIG_BE: u32 = 15;    // 1 - big endian, 0 - little endian
pub const CONFIG_EM: u32 = 14;    // 1 - ecc enabled, 0 - parity enabled
pub const CONFIG_EB: u32 = 13;    // 1 block ordering 1 - sequential 0 - sub block
pub const CONFIG_SE: u32 = 12;    // R5K/Triton: secondary cache enable (R/W); 1=enabled
pub const CONFIG_IC: u32 = 9;     // 3 bits ICache size 2^12+IC
pub const CONFIG_DC: u32 = 6;     // 3 bits DCache size 2^12+IC
pub const CONFIG_IB: u32 = 5;     // icache block size 0=16B 1=32B (R4000/R4400=0, R5000=1)
pub const CONFIG_DB: u32 = 4;     // dcache block size 0=16B 1=32B (R4000/R4400=0, R5000=1)
pub const CONFIG_CU: u32 = 3;     // 0 store conditional uses coherency algo from tlb, 1 - scs uses cacheable coherent update on write
pub const CONFIG_K0: u32 = 0;     // kseg0 coherency algorithm

/// Default undo buffer capacity: 1M (2^20) instructions. `UndoBuffer` can be
/// resized larger at runtime (`undo resize <n>`) — this is only the size a
/// fresh `UndoBuffer::new()` starts at.
#[cfg(feature = "developer")]
const UNDO_BUFFER_SIZE: usize = 1 << 20;

/// Memory write operation for undo tracking
#[cfg(feature = "developer")]
#[derive(Debug, Clone, Copy, PartialEq)]
struct MemoryWrite {
    virt_addr: u64,
    phys_addr: u64,
    old_value: u64,
    size: usize,
}

/// CPU state snapshot for undo - includes core state and metadata
#[cfg(feature = "developer")]
#[derive(Debug, Clone, PartialEq, Default)]
struct CpuSnapshot {
    // General Purpose Registers
    gpr: [u64; 32],

    // Special Registers
    pc: u64,
    hi: u64,
    lo: u64,

    // LL/SC state
    llbit: bool,
    lladdr: u32,

    // CP0 registers
    cp0_index: u32,
    cp0_random: u32,
    cp0_entrylo0: u64,
    cp0_entrylo1: u64,
    cp0_context: u64,
    cp0_pagemask: u64,
    cp0_wired: u32,
    cp0_badvaddr: u64,
    cp0_count: u64,
    cp0_entryhi: u64,
    cp0_compare: u64,
    cp0_status: u32,
    cp0_cause: u32,
    cp0_epc: u64,
    cp0_prid: u32,
    cp0_config: u32,
    cp0_watchlo: u32,
    cp0_watchhi: u32,
    cp0_xcontext: u64,
    cp0_ecc: u32,
    cp0_cacheerr: u32,
    cp0_taglo: u32,
    cp0_taghi: u32,
    cp0_errorepc: u64,

    // CP1 (FPU) registers
    fpr: [u64; 32],
    fpu_fir: u32,
    fpu_fccr: u32,
    fpu_fexr: u32,
    fpu_fenr: u32,
    fpu_fcsr: u32,

    // Execution state
    running: bool,
    halted: bool,

    // Delay slot tracking from executor
    in_delay_slot: bool,
    delay_slot_target: u64,

    // Memory writes that occurred during this instruction
    memory_writes: Vec<MemoryWrite>,
}

/// Circular undo buffer for CPU debugging
#[cfg(feature = "developer")]
struct UndoBuffer {
    enabled: bool,
    snapshots: Vec<Option<CpuSnapshot>>,
    head: usize,  // Next write position
    count: usize, // Number of valid snapshots
}

#[cfg(feature = "developer")]
impl UndoBuffer {
    fn new() -> Self {
        Self {
            enabled: false,
            snapshots: vec![None; UNDO_BUFFER_SIZE],
            head: 0,
            count: 0,
        }
    }

    fn enable(&mut self) {
        self.enabled = true;
    }

    fn disable(&mut self) {
        self.enabled = false;
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn push(&mut self, snapshot: CpuSnapshot) {
        self.push_get_idx(snapshot);
    }

    fn push_get_idx(&mut self, snapshot: CpuSnapshot) -> usize {
        let capacity = self.snapshots.len();
        let idx = self.head;
        self.snapshots[idx] = Some(snapshot);
        self.head = (self.head + 1) % capacity;
        if self.count < capacity {
            self.count += 1;
        }
        idx
    }

    fn can_undo(&self, steps: usize) -> bool {
        self.enabled && steps <= self.count
    }

    fn get(&self, steps_back: usize) -> Option<&CpuSnapshot> {
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

    fn pop(&mut self) {
        if self.count == 0 { return; }
        let capacity = self.snapshots.len();
        self.head = if self.head == 0 { capacity - 1 } else { self.head - 1 };
        self.snapshots[self.head] = None;
        self.count -= 1;
    }

    fn clear(&mut self) {
        self.head = 0;
        self.count = 0;
        for snapshot in &mut self.snapshots {
            *snapshot = None;
        }
    }

    /// Current capacity (max instructions the buffer can hold before it
    /// starts overwriting the oldest entries).
    fn capacity(&self) -> usize {
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
    fn resize(&mut self, new_capacity: usize) {
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

    fn snap(pc: u64) -> CpuSnapshot {
        CpuSnapshot { pc, ..Default::default() }
    }

    #[test]
    fn resize_preserves_order_below_capacity() {
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
    fn resize_preserves_order_after_wraparound() {
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
    fn resize_to_smaller_or_equal_capacity_is_a_no_op() {
        let mut buf = UndoBuffer { enabled: true, snapshots: vec![None; 8], head: 0, count: 0 };
        buf.push(snap(1));
        buf.resize(4); // smaller than current capacity 8
        assert_eq!(buf.capacity(), 8, "resize must never shrink");
        buf.resize(8); // equal
        assert_eq!(buf.capacity(), 8);
        assert_eq!(buf.get(1).unwrap().pc, 1, "existing entries must survive a no-op resize");
    }
}

// Externally-raised interrupt mask (IP7..IP2): the IP bits that other
// threads deliver through `hot.interrupts` and the step() preamble mirrors
// into Cause. IP7 (CP0 Count==Compare) is included since the compare timer
// fires on the hptimer thread and raises it exactly like a device line;
// writing Compare clears the pending bit again (mips_core.rs write_cp0).
const EXT_INT_MASK: u32 = crate::mips_core::CAUSE_IP7 |
                          crate::mips_core::CAUSE_IP6 |
                          crate::mips_core::CAUSE_IP5 |
                          crate::mips_core::CAUSE_IP4 |
                          crate::mips_core::CAUSE_IP3 |
                          crate::mips_core::CAUSE_IP2;

// Bit 63 of the interrupts word = soft-reset request
const SOFT_RESET_BIT: u64 = 1u64 << 63;

const TRACEBACK_SIZE: usize = 1048576; // 1M entries

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
    /// A compiled region's entry word, reached via `exec_decoded`'s own
    /// external JIT-hit dispatch (jumping straight into `jit_fn` for the
    /// first time this exact PC was reached this way) — recorded by
    /// `exec_decoded` itself, not the dev hook (which is internal-edge-only,
    /// see `emit_dev_trace_bp`'s doc comment). Distinct from a plain `Jit`
    /// interior instruction: this IS the entry, with its own foreign-slot/
    /// exception handling, architecturally the JIT counterpart to what
    /// `Interp` meant before this variant existed (a real external dispatch
    /// landing on this exact word) — just resolved into a compiled function
    /// instead of the interpreter.
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
    fn tag(self) -> &'static str {
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
    fn from_u32(v: u32) -> Self {
        match v {
            0 => InstrOrigin::Interp,
            1 => InstrOrigin::Jit,
            2 => InstrOrigin::JitEntryBackEdge,
            3 => InstrOrigin::JitDelaySlot,
            4 => InstrOrigin::FallbackWord,
            5 => InstrOrigin::FallbackSuccessor,
            6 => InstrOrigin::FallbackSuccessorBackEdge,
            _ => InstrOrigin::Jit,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct TracebackEntry {
    pc: u64,
    instr: u32,
    origin: InstrOrigin,
}

struct TracebackBuffer {
    entries: Vec<TracebackEntry>,
    head: usize,
    count: usize,
}

impl TracebackBuffer {
    fn new() -> Self {
        Self {
            entries: vec![TracebackEntry::default(); TRACEBACK_SIZE],
            head: 0,
            count: 0,
        }
    }

    fn push(&mut self, pc: u64, instr: u32, origin: InstrOrigin) {
        self.entries[self.head] = TracebackEntry { pc, instr, origin };
        self.head = (self.head + 1) % TRACEBACK_SIZE;
        if self.count < TRACEBACK_SIZE {
            self.count += 1;
        }
    }

    fn get_last(&self, n: usize) -> Vec<TracebackEntry> {
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
struct IdleSample {
    count: u64,
    ie_count: u64,
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
struct IdleProfiler {
    /// Sample one in every `stride` executed instructions. 0/1 = every instr.
    /// Subsampling bounds hot-path cost; the idle loop still dominates because
    /// it is by far the most frequently executed code while the system idles.
    stride: u64,
    counter: u64,
    total: u64,
    hist: std::collections::HashMap<u64, IdleSample>,
}

#[cfg(feature = "idle-pause")]
impl IdleProfiler {
    #[inline(always)]
    fn sample(&mut self, pc: u64, ie: bool) {
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

    fn reset(&mut self) {
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

pub struct Breakpoint {
    pub id: usize,
    pub addr: u64,
    pub kind: BpType,
    pub enabled: bool,
    /// Optional condition expression (evaluated when breakpoint is hit)
    pub condition: Option<Expr>,
}

/// Execution status returned after each instruction (u32 bitfield).
///
/// Bit layout:
///   bits [6:2]  = exception code (CAUSE_EXCCODE_MASK), valid when IS_EXCEPTION set
///   bits [15:8] = non-exception status tag (valid when IS_EXCEPTION clear)
///   bit  [27]   = EXEC_IS_EXCEPTION: exception or TLB miss occurred
///   bit  [28]   = EXEC_IS_TLB_REFILL: TLB refill (vs generic exception); only when IS_EXCEPTION set
///   bit  [29]   = EXEC_IS_XTLB_REFILL: 64-bit XTLB refill; only when IS_TLB_REFILL set
pub type ExecStatus = u32;

/// Type-erased instruction handler function pointer.
/// Actual type: fn(&mut MipsExecutor<T,C>, &DecodedInstr) -> ExecStatus
pub type RawInstrFn = usize;

/// `DecodedInstr.flags` bit 0: 0 = decoded, 1 = needs (re)decode.
/// Named so `flags == 0` reads naturally as "decoded, nothing else going on" —
/// the resting state for the overwhelming majority of instructions.
pub const FLAG_NOT_DECODED: u8 = 1 << 0;
/// `DecodedInstr.flags` bit 1: input-only, set by the cache fetch path (never
/// stored) before calling `decode_into` to mean "imm currently holds the raw
/// opcode word of the delay-slot instruction from the same L1I line, not the
/// usual pre-processed immediate". `decode_into` consumes this once — for a
/// fusable branch/jump opcode it inspects the word to pick a `_nop` handler
/// variant when it's a literal NOP (0x0), then always overwrites `imm` with
/// the real immediate and clears this bit before returning. `imm` holds the
/// normal pre-processed immediate at rest; this bit is never true afterward.
pub const FLAG_IMM_IS_NEXT: u8 = 1 << 1;

/// Pre-decoded MIPS instruction. All fields extracted from raw word at decode time.
/// Non-generic, suitable for storage in L1I cache lines.
pub struct DecodedInstr {
    pub handler: RawInstrFn,        // type-erased fn ptr
    /// Pre-processed immediate/target. Encoding per opcode:
    ///   J/JAL:              (target26 << 2) as u32  — 28-bit pre-shifted jump offset
    ///   LUI:                (imm16 << 16) as i32 as u32  — sign bit in bit 31
    ///   ANDI/ORI/XORI:      imm16 zero-extended as u32
    ///   all other imm ops:  imm16 sign-extended as i16 as i32 as u32
    ///   R-type / no-imm:    0
    /// Getters immi64()/imms64() widen to u64/i64 on the fly.
    /// See FLAG_IMM_IS_NEXT for the transient decode-time exception.
    pub imm:     u32,
    pub raw:     u32,
    /// See FLAG_NOT_DECODED / FLAG_IMM_IS_NEXT. Kept as a single byte (rather
    /// than a bool plus a separate Option<u32>) so DecodedInstr stays 24 bytes —
    /// Option<u32> can't be niche-packed and would grow this struct to 32 bytes,
    /// which measurably regressed whetstone/dhrystone by bloating the L2
    /// decoded-instruction array and halving its cache-line packing.
    pub flags:   u8,
    pub op:      u8,                // bits [31:26]
    pub rs:      u8,                // bits [25:21]  (also: base for loads/stores, fs for FPU)
    pub rt:      u8,                // bits [20:16]  (also: ft for FPU)
    pub rd:      u8,                // bits [15:11]  (also: fs for FPU)
    pub sa:      u8,                // bits [10:6]   (also: fd for FPU)
    pub funct:   u8,                // bits [5:0]
}

impl DecodedInstr {
    /// Immediate zero-widened to u64.  Only correct for ZE-encoded values (ANDI/ORI/XORI, J/JAL).
    #[inline(always)]
    pub fn immi64(&self) -> u64 { self.imm as u64 }
    /// Immediate sign-extended from i32 to i64.  For SE-encoded values used as signed.
    #[inline(always)]
    pub fn imms64(&self) -> i64 { self.imm as i32 as i64 }
    /// Immediate sign-extended from i32 then reinterpreted as u64.
    /// Used for SE-encoded values in unsigned contexts (SLTIU, TGEIU, TLTIU, addr calc).
    #[inline(always)]
    pub fn immu64(&self) -> u64 { self.imm as i32 as i64 as u64 }

    /// Decode: sign-extend imm16 to 32 bits.  Used by arithmetic/load/store/trap immediates.
    #[inline(always)]
    pub fn set_imm_se(&mut self, raw: u32) {
        self.imm = (raw & 0xFFFF) as i16 as i32 as u32;
    }
    /// Decode: sign-extend imm16 then shift left 2.  Used by branch offsets.
    #[inline(always)]
    pub fn set_imm_se4(&mut self, raw: u32) {
        self.imm = ((raw & 0xFFFF) as i16 as i32 * 4) as u32;
    }
    /// Decode: zero-extend imm16.  Used by ANDI/ORI/XORI.
    #[inline(always)]
    pub fn set_imm_ze(&mut self, raw: u32) {
        self.imm = (raw & 0xFFFF) as u32;
    }
    /// Decode: shift imm16 left 16, keeping sign in bit 31.  Used by LUI.
    #[inline(always)]
    pub fn set_imm_lui(&mut self, raw: u32) {
        self.imm = (raw & 0xFFFF) << 16;
    }
    /// Decode: 26-bit jump target shifted left 2.  Used by J/JAL.
    #[inline(always)]
    pub fn set_imm_j(&mut self, raw: u32) {
        self.imm = (raw & 0x3FFFFFF) << 2;
    }
}

impl Default for DecodedInstr {
    fn default() -> Self {
        Self {
            raw:     0,
            flags:   FLAG_NOT_DECODED,
            op:      0,
            rs:      0,
            rt:      0,
            rd:      0,
            sa:      0,
            funct:   0,
            imm:     0,
            handler: 0,
        }
    }
}

// Non-exception status tags in bits [15:8]. EXEC_COMPLETE carries no
// PC-related meaning — every handler sets core.pc itself before returning
// (directly, or via branch_delay/handle_exec_complete/
// handle_branch_likely_skip) — it's just "ran fine, no
// exception/retry/breakpoint". EXEC_COMPLETE_NO_INC / EXEC_BRANCH_DELAY /
// EXEC_BRANCH_LIKELY_SKIP / EXEC_COMPLETE_SKIP8 used to distinguish *why*
// PC ended up where it did (interpreter PC+=4 vs a JIT/ERET direct-set vs a
// taken branch vs PC+=8) back when callers needed that to decide whether to
// advance PC themselves; now that every handler is unconditionally
// responsible for its own PC, the distinction is dead — removed. Real
// callers that need "was a branch just taken" (e.g. gdbstub's step_one)
// check `core.in_delay_slot` instead.
pub const EXEC_COMPLETE:           ExecStatus = 0x0000_0000; // ran fine, no exception/retry/breakpoint
pub const EXEC_RETRY:              ExecStatus = 0x0000_0100; // bus busy, retry same instr
pub const EXEC_FALLBACK:           ExecStatus = 0x0000_0200; // jitv2+lightning's decode-skip fast path missed; caller must decode and dispatch normally
pub const EXEC_BREAKPOINT:         ExecStatus = 0x0000_0800; // breakpoint hit

/// `jitv2_lockstep`: sentinel `bd` value passed to `lockstep_step_fn` meaning
/// "trust the live `core.pc`/`core.in_delay_slot`/`delay_slot_target` — do NOT
/// overwrite them from the compile-time constants." Codegen passes it for an
/// entry word or a delay slot (branch-fallback successor), whose state is
/// inherited from the arrival and must be preserved. `0xFF` is safely outside
/// the real bd range (0/1). `u32`, not `u8` — see `MipsCore::lockstep_step_fn`'s
/// doc comment.
#[cfg(feature = "jitv2_lockstep")]
pub const LOCKSTEP_BD_LIVE: u32 = 0xFF;

/// `jitv2_lockstep`: count of instructions actually lockstep-compared (a real
/// JIT-vs-interpreter comparison ran, not a skipped class). Surfaced by
/// `j2 stats` as positive proof lockstep is live — a stuck-at-0 here means the
/// JIT never ran (e.g. the async queue stole the codegen so nothing compiled),
/// i.e. verification is silently OFF. Process-global atomic to keep it off the
/// per-instruction locking path.
#[cfg(feature = "jitv2_lockstep")]
pub static LOCKSTEP_VERIFICATIONS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

// Exception flags
pub const EXEC_IS_EXCEPTION:       ExecStatus = 1 << 27; // 0x0800_0000
pub const EXEC_IS_TLB_REFILL:      ExecStatus = 1 << 28; // 0x1000_0000
pub const EXEC_IS_XTLB_REFILL:     ExecStatus = 1 << 29; // 0x2000_0000

// Bus error ExecStatus values — also exported as BUS_ERR / BUS_VCE in traits.rs.
// The values MUST stay identical; enforced by compile-time asserts in traits.rs.
pub const EXEC_BUS_ERR: ExecStatus = exec_exception_const(EXC_DBE);  // 0x0800_001C
pub const EXEC_BUS_VCE: ExecStatus = exec_exception_const(EXC_VCED); // 0x0800_007C

/// `const`-evaluable version of exec_exception (for use in const initializers).
#[inline(always)]
pub const fn exec_exception_const(code: u32) -> ExecStatus {
    EXEC_IS_EXCEPTION | (code << crate::mips_core::CAUSE_EXCCODE_SHIFT)
}

/// Build an exception ExecStatus from an EXC_* code.
#[inline(always)]
pub fn exec_exception(code: u32) -> ExecStatus {
    exec_exception_const(code)
}

/// Build a TLB-refill ExecStatus from an EXC_* code (32-bit UTLB vector).
#[inline(always)]
pub fn exec_tlb_miss(code: u32) -> ExecStatus {
    EXEC_IS_EXCEPTION | EXEC_IS_TLB_REFILL | (code << crate::mips_core::CAUSE_EXCCODE_SHIFT)
}

/// Build an XTLB-refill ExecStatus from an EXC_* code (64-bit XTLB vector, offset 0x080).
#[inline(always)]
pub fn exec_xtlb_miss(code: u32) -> ExecStatus {
    EXEC_IS_EXCEPTION | EXEC_IS_TLB_REFILL | EXEC_IS_XTLB_REFILL | (code << crate::mips_core::CAUSE_EXCCODE_SHIFT)
}

/// Alignment mask for a memory access of SIZE bytes.
/// `addr & align_mask_for::<SIZE>() != 0` means misaligned.
#[inline(always)]
const fn align_mask_for<const SIZE: usize>() -> u64 {
    (SIZE as u64) - 1
}

/// Full data mask for SIZE bytes (e.g. SIZE=4 → 0xFFFF_FFFF).
#[inline(always)]
const fn full_mask_for<const SIZE: usize>() -> u64 {
    if SIZE == 8 { !0u64 } else { !0u64 >> (64 - SIZE * 8) }
}

/// Runtime version of full_mask_for for use in command parsers.
#[inline(always)]
fn full_mask_for_usize(size: usize) -> u64 {
    if size == 8 { !0u64 } else { !0u64 >> (64 - size * 8) }
}

/// Cache coherency attributes — values match the MIPS C0 EntryLo C field.
/// Kept for use inside the TLB layer (TlbResult, NanoTlbEntry).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheAttr {
    Uncached          = 2,
    Cacheable         = 3,
    CacheableCoherent = 5,
}

/// Hardware C-field values packed into TranslateResult.status bits [2:0].
/// Matches CacheAttr discriminants exactly, so no conversion is needed.
pub const TR_UNCACHED:        u32 = 2; // C=2: Uncached
pub const TR_CACHEABLE:       u32 = 3; // C=3: Cacheable (write-back)
pub const TR_CACHEABLE_COH:   u32 = 5; // C=5: Cacheable Coherent Exclusive

/// Result of address translation: 8 bytes, no heap.
///
/// Layout when success (EXEC_IS_EXCEPTION clear in `status`):
///   `phys`   — 32-bit physical address
///   `status` — bits [2:0]: C-field cache attr (TR_UNCACHED/TR_CACHEABLE/TR_CACHEABLE_COH)
///              all other bits 0
///
/// Layout when exception (EXEC_IS_EXCEPTION set in `status`):
///   `phys`   — 0 (ignored)
///   `status` — fully-formed ExecStatus for handle_exception
#[derive(Debug, Clone, Copy)]
pub struct TranslateResult {
    pub phys:   u32,
    pub status: u32,
}

impl TranslateResult {
    #[inline(always)]
    pub fn ok(phys: u64, c_field: u32) -> Self {
        Self { phys: phys as u32, status: c_field }
    }
    #[inline(always)]
    pub fn exc(s: ExecStatus) -> Self {
        Self { phys: 0, status: s }
    }
    #[inline(always)]
    pub fn is_exception(self) -> bool {
        self.status & EXEC_IS_EXCEPTION != 0
    }
    /// True for any cached attribute (C=3 or C=5); false for uncached (C=2).
    #[inline(always)]
    pub fn is_cached(self) -> bool {
        self.status & 0x7 != TR_UNCACHED
    }
}

/// Configuration for the MIPS CPU: TLB and cache hierarchy sizes.
#[derive(Debug, Clone, Copy)]
pub struct MipsCpuConfig {
    pub tlb_entries: usize,
}

impl MipsCpuConfig {
    /// Default configuration matching SGI Indy (R4400): 48-entry TLB.
    /// Cache geometry is fixed at compile time via constants in mips_cache_v2.
    pub const fn indy() -> Self {
        Self { tlb_entries: 48 }
    }
}

/// MIPS Execution Engine - combines CPU core with memory interface and TLB
pub struct MipsExecutor<T: Tlb, C: MipsCache> {
    pub core: MipsCore,
    pub sysad: Arc<dyn BusDevice>,
    /// cheritest convention (`--cheritest-dump-hook`): a guest write to CP0 26
    /// dumps machine state via the test device. Off unless asked for — CP0 26
    /// is ECC on a real R4400 and the PROM writes it during cache init.
    cheritest_dump_hook: bool,
    pub tlb: T,
    pub cache: C,
    #[cfg(feature = "developer")]
    undo_buffer: UndoBuffer,
    #[cfg(feature = "developer")]
    pending_memory_writes: Vec<MemoryWrite>,
    traceback: TracebackBuffer,
    /// Per-instruction execution trace recorder (rules/jitv2's lockstep
    /// verification tooling, `src/trace.rs`). `None` when not recording —
    /// checked on every step() (one branch on a `None` tag), so armed cost
    /// is a file write, disarmed cost is a predictable branch. Developer-only:
    /// this captures full architectural state per instruction, meaningfully
    /// intrusive on the hot path, matching undo_buffer/idle_profiler's own
    /// dev-only gating.
    #[cfg(feature = "developer")]
    trace_writer: Option<crate::trace::TraceWriter>,
    #[cfg(feature = "idle-pause")]
    idle_profiler: IdleProfiler,
    #[cfg(feature = "idle-pause")]
    pub idle_profile_on: Arc<AtomicBool>,
    #[cfg(feature = "idle-pause")]
    pub idle_profile_reset: Arc<AtomicBool>,
    #[cfg(feature = "idle-pause")]
    idle_profile_on_ptr: *const AtomicBool,
    pub symbols: Arc<Mutex<SymbolTable>>,
    pub breakpoints: Vec<Breakpoint>,
    pub next_bp_id: usize,
    pub last_bp_hit: Option<usize>,
    pub pc_bp_count: usize,
    pub mem_bp_count: usize,
    /// When true, the next call to step() skips all breakpoint checks.
    /// Cleared automatically after one step. Used to resume past a breakpoint.
    pub skip_breakpoints: bool,
    /// When true, the next call to step() skips the pending-interrupt check.
    /// Cleared automatically after one step. Used by the debugger `s` command
    /// so single-stepping works inside an interrupt handler whose source is
    /// still asserted (without this, the CPU re-takes the exception every step).
    #[cfg(feature = "developer")]
    pub skip_interrupts: bool,
    /// Current decoded instruction — written by fetch_instr, read by exec_decoded.
    pub ins: DecodedInstr,
    /// Count of instructions that were already decoded (cache hit).
    pub decoded_count: Arc<AtomicU64>,
    /// Count of instructions fetched from uncached address space.
    pub uncached_fetch_count: Arc<AtomicU64>,
    /// Hot-path translation function pointer, updated whenever CP0 Status changes.
    /// Always the non-debug variant; selects the correct 32/64-bit × privilege specialisation.
    pub translate_fn: fn(&mut Self, u64, AccessType) -> TranslateResult,
    /// FR-mode-aware FPR accessors. Switched in update_fpr_mode() whenever STATUS_FR changes.
    /// FR=0: doubles/longs use full even slot; odd single/word regs are upper 32 bits of even slot.
    /// FR=1: all 32 slots are independent 64-bit registers.
    pub fpr_read_d:  fn(&MipsCore, u32) -> f64,
    pub fpr_write_d: fn(&mut MipsCore, u32, f64),
    pub fpr_read_l:  fn(&MipsCore, u32) -> u64,
    pub fpr_write_l: fn(&mut MipsCore, u32, u64),
    pub fpr_read_w:  fn(&MipsCore, u32) -> u32,
    pub fpr_write_w: fn(&mut MipsCore, u32, u32),
    /// Cached external interrupt word — reloaded every 16 instructions.
    pub(crate) cached_pending: u64,
    /// Per-instruction execution frequency counters (feature = "instr_stats").
    #[cfg(feature = "instr_stats")]
    pub instr_stats: crate::mips_instr_stats::InstrStats,
    /// JIT v2 engine state: physical-code-page pool + allocator (rules/jitv2/jit-v2-design.md §2.4).
    /// Lives independently of the executor (`Arc<Mutex<...>>`, constructed by
    /// `Machine` and shared in) — NOT owned by the executor's own struct
    /// literal — specifically so the compile thread (which needs to call
    /// `MipsCpu::stop()`/`start()` on itself to pause the CPU around a memory
    /// flush, see `Codegen::function_count()`'s doc comment) never has to
    /// reach through the executor's own lock to get there: `MipsCpu::stop()`
    /// no longer touches the compile queue at all (decoupled — the queue's
    /// lifecycle belongs to `Machine`/the `j2 inline` monitor command now),
    /// so there is no cycle through the executor mutex for the compile
    /// thread to deadlock on. The mutex here is coarse (whole-struct, not
    /// per-field) since `page_for`/`page_ptr`/`mega_flush` are already
    /// CPU-thread-only by design contract (§6.1.3) and
    /// `compile_queue.send/start/stop` are never called concurrently from
    /// both sides by contract — this isn't a hot-path lock, just ordinary
    /// exclusion for infrequent pool/queue management calls.
    #[cfg(feature = "jitv2")]
    pub jitv2: std::sync::Arc<Mutex<crate::jitv2::Jitv2>>,
    /// Pointer to the `PhysicalCodePage` for the page the fetch-side nanotlb last
    /// resolved to. Updated only on a page change (§2.1 — physical page, not VA);
    /// null until the first fetch translation after construction/reset. Owned and
    /// written exclusively by this thread (§6.1.3 executor-thread ownership) —
    /// valid until the next `jitv2.mega_flush()`, which is why it's re-derived
    /// lazily rather than cached across a flush.
    #[cfg(feature = "jitv2")]
    pub(crate) pcp: *mut crate::jitv2::PhysicalCodePage,
    /// Inline per-instruction lockstep scratch (`jit_lockstep_step`/
    /// `jit_lockstep_compare`): `ls_before` is the pre-instruction state saved
    /// so it can be restored after the interpreter's reference run, before the
    /// JIT runs the same instruction; `ls_expected` is the interpreter's
    /// resulting state, held for `jit_lockstep_compare` to check the JIT
    /// against. `ls_delay_target_before`/`ls_delay_target_expected` preserve
    /// `delay_slot_target` across the restore and for the compare respectively
    /// (it's not in `LockstepSnapshot`'s comparable fields but must round-trip
    /// alongside `pc`/`in_delay_slot` — the three are always read/restored
    /// together, never individually: `lockstep_compare`'s divergence path uses
    /// `ls_before`'s `pc`/`in_delay_slot` plus `ls_delay_target_before` as one
    /// unit to put `core` back at the instruction that actually diverged).
    /// All only live between one instruction's step and compare callbacks;
    /// boxed `Option` so the not-yet-stepped state is unambiguous.
    #[cfg(feature = "jitv2_lockstep")]
    pub(crate) ls_before: Option<LockstepSnapshot>,
    #[cfg(feature = "jitv2_lockstep")]
    pub(crate) ls_expected: Option<LockstepSnapshot>,
    #[cfg(feature = "jitv2_lockstep")]
    pub(crate) ls_delay_target_before: u64,
    #[cfg(feature = "jitv2_lockstep")]
    pub(crate) ls_delay_target_expected: u64,
    /// Runtime switch (not a Cargo feature) for how `exec_decoded`'s real
    /// jitv2 dispatch gate gets a fresh artifact on a miss: `false` hands a
    /// `CompileRequest` to the async `compile_queue` worker thread; `true`
    /// (the default) compiles synchronously on this (the CPU) thread via
    /// `comp::handle_request` and runs the result immediately, no
    /// cross-thread scheduling involved at all. Monitor console: `j2 inline
    /// on|off`. See `exec_decoded`'s doc comment on the gate for why this
    /// exists (ruling scheduling in/out as a bug suspect, and giving tests a
    /// way to exercise jitv2 deterministically instead of depending on
    /// whether the async compile thread happened to win a race within a
    /// short loop).
    #[cfg(feature = "jitv2")]
    pub jitv2_inline_compile: bool,
    /// Scratch analyzer for `jitv2_inline_compile`'s synchronous path — same
    /// `comp::handle_request` compile sequence the real async compile-thread
    /// runs, just called directly on the CPU thread. Kept alive across calls
    /// purely to reuse its scratch buffer. The `Codegen` this path compiles
    /// through is NOT a separate instance here — see `Jitv2::codegen`'s doc
    /// comment for why inline dispatch and the async compile thread share
    /// one Cranelift memory arena rather than each growing its own.
    #[cfg(feature = "jitv2")]
    pub(crate) jitv2_inline_analyzer: crate::jitv2::analyzer::Analyzer,
    /// Runtime kill switch for `exec_decoded`'s real JIT dispatch gate
    /// (`#[cfg(all(feature = "jitv2", not(feature = "jitv2_lockstep")))]`
    /// path) — `true` (default) is normal behavior; `false` makes
    /// `exec_decoded` skip the whole gate unconditionally, falling through
    /// to the interpreter for every instruction, as if `jitv2` weren't
    /// compiled in at all. Monitor console: `j2 dispatch on|off`, and the
    /// engine behind the machine-level `jitcheck <n>` command
    /// (`validate::validate_jit_determinism`, `SystemController::execute_command`
    /// in machine.rs) — captures live state, runs N instructions
    /// interpreter-only, then the same N with real JIT dispatch, diffing
    /// per-instruction — a way to find out whether a given instruction
    /// window's divergence from a pure-interpreter run is caused by real
    /// JIT dispatch at all, on a build that can't simply be rebuilt without
    /// the `jitv2` feature (unlike `jitv2_lockstep`, this doesn't require
    /// recompiling to compare against).
    #[cfg(feature = "jitv2")]
    pub jitv2_dispatch_enabled: bool,
    /// `jitcheck`'s hardware-read fixup: `validate_jit_determinism`'s two
    /// passes run at genuinely different real wall-clock speed, so any
    /// device register whose value is driven by host time (e.g. the MC's
    /// RPSS_CTR, `mc.rs`'s `update_timers` — see `HW_READ_FIXUP_ADDRS`)
    /// legitimately returns a different value on every read between passes,
    /// with zero JIT bug involved. A tight PROM polling loop that rereads
    /// such a register every iteration makes `skip`-based reconvergence
    /// impractical (every loop iteration is its own "divergence"). Instead:
    /// during the reference (interpreter-only) pass, `read_data_impl`
    /// records the real value read from any address in
    /// `HW_READ_FIXUP_ADDRS` into `CpuStateDigest::hw_reads`; during the
    /// replay (JIT-dispatch) pass, `read_data_impl` consults
    /// `hw_read_fixup_replay` (set by the caller to the reference digest
    /// that's expected after the *upcoming* step, before each step) and
    /// substitutes the recorded value instead of the real bus read — so the
    /// JIT pass observes bit-identical "hardware" inputs and only genuine
    /// CPU/JIT logic divergences show up in the diff. `None` = fixup
    /// inactive (normal execution, all other callers).
    #[cfg(feature = "developer")]
    pub(crate) hw_read_fixup_replay: Option<Vec<(u64, u8, u64)>>,
    /// Recording side of the same mechanism — populated by `read_data_impl`
    /// during the reference pass whenever `hw_read_fixup_recording` is
    /// `true`, drained into `CpuStateDigest::hw_reads` by `state_digest()`.
    #[cfg(feature = "developer")]
    pub(crate) hw_read_fixup_recorded: Vec<(u64, u8, u64)>,
    /// `true` while the reference pass wants `read_data_impl` to record
    /// (see `hw_read_fixup_recorded`). Mutually exclusive with
    /// `hw_read_fixup_replay` being `Some` in practice (record pass vs.
    /// replay pass), but both are independent toggles rather than one enum
    /// so the reconverge-mid-replay case (`validate_jit_determinism`
    /// skipping past a divergence) never has to juggle a mode transition.
    #[cfg(feature = "developer")]
    pub(crate) hw_read_fixup_recording: bool,
    /// The `ExecStatus` the most recent `step_one_inline_counting_instructions`
    /// call returned — surfaced through `state_digest`'s `step_status` field
    /// so `validate_jit_determinism` can compare, per instruction, what
    /// status *code* each engine's step actually produced (not just the
    /// resulting register/cop0 state), directly answering "did the JIT step
    /// take an exception the interpreter step didn't" instead of inferring
    /// it indirectly from EXL/EPC alone.
    #[cfg(feature = "developer")]
    pub(crate) last_step_status: ExecStatus,
}

/// Fixed, curated list of physical addresses known to be driven by real host
/// wall-clock time (not architectural CPU cycles), and therefore expected to
/// read differently between `jitcheck`'s two separately-timed passes with no
/// JIT bug involved. Deliberately not learned/inferred — an address only
/// belongs here once a live divergence has been manually confirmed benign
/// (see rules/jitv2/ for the investigation trail). Extend this list as more
/// such registers are found; do not add speculative entries.
#[cfg(feature = "developer")]
pub(crate) const HW_READ_FIXUP_ADDRS: &[u64] = &[
    0x1fa01000, // MC REG_RPSS_CTR (mc.rs) — 100ns free-running counter, host-time driven
    0x1fa01004, // MC decodes this as the same RPSS_CTR word (confirmed live) — same fixup applies
    0x1fa02048, // MC REG_DMA_RUN (mc.rs) — latched by an async DMA-completion thread, not CPU-cycle driven
    0x1fa0204c, // MC decodes this as the same DMA_RUN word (confirmed live) — same fixup applies
    // HPC3 PBUS_BBRAM (ds1x86.rs Dallas DS1386 RTC), sparse-packed one byte
    // per 4-byte-strided word (hpc3.rs read32's PBUS_BBRAM branch:
    // byte_index = (addr - 0x1fbe0000) >> 2). First 16 bytes are the live
    // time-of-day registers, computed from real host wall-clock time on
    // every read (ds1x86.rs) — same host-time-driven class as RPSS_CTR.
    0x1fbe0000, 0x1fbe0004, 0x1fbe0008, 0x1fbe000c,
    0x1fbe0010, 0x1fbe0014, 0x1fbe0018, 0x1fbe001c,
    0x1fbe0020, 0x1fbe0024, 0x1fbe0028, 0x1fbe002c,
    0x1fbe0030, 0x1fbe0034, 0x1fbe0038, 0x1fbe003c,
    // IOC PIT block (ioc.rs IOC_TIMER_CNT0..IOC_TIMER_CTL, offsets
    // 0xB0/0xB4/0xB8/0xBC off IOC_BASE=0x1FBD9800) — hardware timer
    // counters PROM busy-waits on for calibration/delay loops
    // (confirmed live: a `bgtz $t5,-1` countdown loop immediately
    // preceding an `lbu` from this block). `read8`'s `& !3` masks any
    // sub-offset within a 4-byte register to the same counter, so all
    // 4 byte addresses of all 4 registers are listed — `phys_addr` here
    // is the exact byte address the CPU issued, before that masking.
    0x1fbd98b0, 0x1fbd98b1, 0x1fbd98b2, 0x1fbd98b3, // IOC_TIMER_CNT0
    0x1fbd98b4, 0x1fbd98b5, 0x1fbd98b6, 0x1fbd98b7, // IOC_TIMER_CNT1
    0x1fbd98b8, 0x1fbd98b9, 0x1fbd98ba, 0x1fbd98bb, // IOC_TIMER_CNT2
    0x1fbd98bc, 0x1fbd98bd, 0x1fbd98be, 0x1fbd98bf, // IOC_TIMER_CTL
];

/// Comparison key for breakpoint addresses (see `check_breakpoint`): word-aligns,
/// and folds the unmapped kseg0/kseg1 windows onto the physical address they alias
/// so either window hits. Every other address keeps all of its bits.
#[inline]
fn bp_match_key(addr: u64) -> u64 {
    let hi = addr >> 32;
    if hi == 0 || hi == 0xFFFF_FFFF {
        let lo = addr & 0xFFFF_FFFF;
        if (0x8000_0000..0xC000_0000).contains(&lo) {
            return (lo & 0x1FFF_FFFF) & !3;
        }
    }
    addr & !3
}

// ---- translate_fn slow-path wrappers (one per privilege × addressing-mode combination) ------
// These are free functions so they can be stored as bare fn pointers in MipsExecutor.
// They are only called on a nanotlb miss — the nanotlb probe happens before the fn-pointer call.
fn translate_32_kernel<T: Tlb, C: MipsCache>(e: &mut MipsExecutor<T,C>, va: u64, at: AccessType) -> TranslateResult {
    e.translate_32bit_impl::<false, {crate::mips_core::PRIV_KERNEL}>(va, at)
}
fn translate_32_supervisor<T: Tlb, C: MipsCache>(e: &mut MipsExecutor<T,C>, va: u64, at: AccessType) -> TranslateResult {
    e.translate_32bit_impl::<false, {crate::mips_core::PRIV_SUPERVISOR}>(va, at)
}
fn translate_32_user<T: Tlb, C: MipsCache>(e: &mut MipsExecutor<T,C>, va: u64, at: AccessType) -> TranslateResult {
    e.translate_32bit_impl::<false, {crate::mips_core::PRIV_USER}>(va, at)
}
fn translate_64_kernel<T: Tlb, C: MipsCache>(e: &mut MipsExecutor<T,C>, va: u64, at: AccessType) -> TranslateResult {
    e.translate_64bit_impl::<false, {crate::mips_core::PRIV_KERNEL}>(va, at)
}
fn translate_64_supervisor<T: Tlb, C: MipsCache>(e: &mut MipsExecutor<T,C>, va: u64, at: AccessType) -> TranslateResult {
    e.translate_64bit_impl::<false, {crate::mips_core::PRIV_SUPERVISOR}>(va, at)
}
fn translate_64_user<T: Tlb, C: MipsCache>(e: &mut MipsExecutor<T,C>, va: u64, at: AccessType) -> TranslateResult {
    e.translate_64bit_impl::<false, {crate::mips_core::PRIV_USER}>(va, at)
}

/// Free-standing trampoline for the CP0 Status callback installed by `install_status_cb`.
/// Rust does not allow `Self` inside a nested fn, so the generic trampoline lives here.
// Safety: the executor's raw pointers point into allocations owned by
// MipsCore which outlive the executor. The executor is only accessed from the CPU thread.
unsafe impl<T: Tlb, C: MipsCache> Send for MipsExecutor<T, C> {}
unsafe impl<T: Tlb, C: MipsCache> Sync for MipsExecutor<T, C> {}

fn mips_executor_status_cb<T: Tlb, C: MipsCache>(ctx: *mut core::ffi::c_void, old: u32, new: u32) {
    // SAFETY: ctx is `&mut MipsExecutor<T,C>` cast to void, alive for the executor's lifetime,
    // and only ever called from the CPU thread that exclusively owns the executor.
    let exec = unsafe { &mut *(ctx as *mut MipsExecutor<T, C>) };
    exec.on_cp0_status_changed(old, new);
}

// ---- JIT v2 memory-access / exception-delivery trampolines ------
// Free functions (one monomorphized instantiation per <T,C>, like the
// translate_fn wrappers above) so their addresses can be stored as bare
// `unsafe extern "C" fn` pointers in MipsCore (`install_jit_hooks`). Every
// read sets `core.jit_mem_exc`; compiled code must check it after the call
// (see MipsCore's read*_fn field doc comments). `ctx` is always the
// executor's own address, established by `install_jit_hooks` — same safety
// argument as `mips_executor_status_cb` above.
#[cfg(feature = "jitv2")]
unsafe extern "C" fn jit_read8<T: Tlb, C: MipsCache>(ctx: *mut core::ffi::c_void, va: u64) -> u64 {
    let exec = unsafe { &mut *(ctx as *mut MipsExecutor<T, C>) };
    #[cfg(feature = "jitv2_lockstep")]
    { return exec.lockstep_jit_read::<1>(va); }
    #[cfg(not(feature = "jitv2_lockstep"))]
    match exec.read_data::<1>(va) {
        Ok(v) => { exec.core.jit_mem_exc = EXEC_COMPLETE; v }
        Err(status) => { exec.core.jit_mem_exc = status; 0 }
    }
}
#[cfg(feature = "jitv2")]
unsafe extern "C" fn jit_read16<T: Tlb, C: MipsCache>(ctx: *mut core::ffi::c_void, va: u64) -> u64 {
    let exec = unsafe { &mut *(ctx as *mut MipsExecutor<T, C>) };
    #[cfg(feature = "jitv2_lockstep")]
    { return exec.lockstep_jit_read::<2>(va); }
    #[cfg(not(feature = "jitv2_lockstep"))]
    match exec.read_data::<2>(va) {
        Ok(v) => { exec.core.jit_mem_exc = EXEC_COMPLETE; v }
        Err(status) => { exec.core.jit_mem_exc = status; 0 }
    }
}
#[cfg(feature = "jitv2")]
unsafe extern "C" fn jit_read32<T: Tlb, C: MipsCache>(ctx: *mut core::ffi::c_void, va: u64) -> u64 {
    let exec = unsafe { &mut *(ctx as *mut MipsExecutor<T, C>) };
    #[cfg(feature = "jitv2_lockstep")]
    { return exec.lockstep_jit_read::<4>(va); }
    #[cfg(not(feature = "jitv2_lockstep"))]
    match exec.read_data::<4>(va) {
        Ok(v) => { exec.core.jit_mem_exc = EXEC_COMPLETE; v }
        Err(status) => { exec.core.jit_mem_exc = status; 0 }
    }
}
#[cfg(feature = "jitv2")]
unsafe extern "C" fn jit_read64<T: Tlb, C: MipsCache>(ctx: *mut core::ffi::c_void, va: u64) -> u64 {
    let exec = unsafe { &mut *(ctx as *mut MipsExecutor<T, C>) };
    #[cfg(feature = "jitv2_lockstep")]
    { return exec.lockstep_jit_read::<8>(va); }
    #[cfg(not(feature = "jitv2_lockstep"))]
    match exec.read_data::<8>(va) {
        Ok(v) => { exec.core.jit_mem_exc = EXEC_COMPLETE; v }
        Err(status) => { exec.core.jit_mem_exc = status; 0 }
    }
}
#[cfg(feature = "jitv2")]
unsafe extern "C" fn jit_write8<T: Tlb, C: MipsCache>(ctx: *mut core::ffi::c_void, va: u64, val: u64) -> u32 {
    let exec = unsafe { &mut *(ctx as *mut MipsExecutor<T, C>) };
    let val = val as u8; // mask to the real width ourselves — see write8_fn's doc comment on why the FFI parameter is u64
    #[cfg(feature = "jitv2_lockstep")]
    { return exec.lockstep_jit_write::<1>(va, val as u64); }
    #[cfg(not(feature = "jitv2_lockstep"))]
    { let status = exec.write_data::<1>(va, val as u64); exec.core.jit_mem_exc = status; status }
}
#[cfg(feature = "jitv2")]
unsafe extern "C" fn jit_write16<T: Tlb, C: MipsCache>(ctx: *mut core::ffi::c_void, va: u64, val: u64) -> u32 {
    let exec = unsafe { &mut *(ctx as *mut MipsExecutor<T, C>) };
    let val = val as u16; // mask to the real width ourselves — see write16_fn's doc comment
    #[cfg(feature = "jitv2_lockstep")]
    { return exec.lockstep_jit_write::<2>(va, val as u64); }
    #[cfg(not(feature = "jitv2_lockstep"))]
    { let status = exec.write_data::<2>(va, val as u64); exec.core.jit_mem_exc = status; status }
}
#[cfg(feature = "jitv2")]
unsafe extern "C" fn jit_write32<T: Tlb, C: MipsCache>(ctx: *mut core::ffi::c_void, va: u64, val: u64) -> u32 {
    let exec = unsafe { &mut *(ctx as *mut MipsExecutor<T, C>) };
    let val = val as u32; // mask to the real width ourselves — see write32_fn's doc comment
    #[cfg(feature = "jitv2_lockstep")]
    { return exec.lockstep_jit_write::<4>(va, val as u64); }
    #[cfg(not(feature = "jitv2_lockstep"))]
    { let status = exec.write_data::<4>(va, val as u64); exec.core.jit_mem_exc = status; status }
}
#[cfg(feature = "jitv2")]
unsafe extern "C" fn jit_write64<T: Tlb, C: MipsCache>(ctx: *mut core::ffi::c_void, va: u64, val: u64) -> u32 {
    let exec = unsafe { &mut *(ctx as *mut MipsExecutor<T, C>) };
    #[cfg(feature = "jitv2_lockstep")]
    { return exec.lockstep_jit_write::<8>(va, val); }
    #[cfg(not(feature = "jitv2_lockstep"))]
    { let status = exec.write_data::<8>(va, val); exec.core.jit_mem_exc = status; status }
}
/// JIT-callable `write_data64_masked` wrapper — the SWL/SWR/SDL/SDR
/// counterpart to `jit_write64`'s plain full-width write. No
/// `jitv2_lockstep` comparison path: `lockstep_jit_write`'s captured-value
/// comparison assumes a plain, full-width write against
/// `MipsCore::lockstep_mem`'s single captured `(addr, phys, value)` triple
/// (see that struct's doc comment) — a masked write's "did the JIT compute
/// the same partial update" question doesn't fit that shape without
/// inventing a masked-comparison variant, which nothing currently needs;
/// `jitv2_lockstep` is a dev-only single-instruction lockstep verifier, not
/// the normal dispatch path, so this always goes straight through
/// `write_data64_masked` regardless of that feature.
#[cfg(feature = "jitv2")]
unsafe extern "C" fn jit_write64_masked<T: Tlb, C: MipsCache>(ctx: *mut core::ffi::c_void, va: u64, val: u64, mask: u64) -> u32 {
    let exec = unsafe { &mut *(ctx as *mut MipsExecutor<T, C>) };
    let status = exec.write_data64_masked(va, val, mask);
    exec.core.jit_mem_exc = status;
    status
}
/// Single-implementation exception delivery (§4.2): calls the interpreter's
/// own `handle_exception` — the only place EPC/Cause/BD/vectoring are ever
/// computed, for both engines.
#[cfg(feature = "jitv2")]
unsafe extern "C" fn jit_handle_exception<T: Tlb, C: MipsCache>(ctx: *mut core::ffi::c_void, status: u32) -> u32 {
    let exec = unsafe { &mut *(ctx as *mut MipsExecutor<T, C>) };
    // handle_exception reads self.core.in_delay_slot — a single MipsCore
    // field shared by both engines (no separate JIT-only copy or sync step
    // needed): the interpreter's branch_delay/handle_exec_complete set it on
    // the plain dispatch path, and codegen's emit_branch_or_jump/emit_regjump
    // (jitv2/codegen.rs) set it directly around a delay slot's inlined body,
    // since the JIT has no separate dispatch step of its own to hang it on
    // (§6.1.4 — the slot's semantics are inlined straight into the
    // compiled unit). Whichever engine is currently executing has already
    // written the correct value by the time this runs.
    exec.handle_exception(status)
}

/// Force one instruction's worth of real forward progress through the
/// interpreter, bypassing the JIT dispatch gate — see
/// `MipsCore::interp_fallback_fn`'s doc comment for why this needs to exist
/// at all (a plain JIT bail can't force a fallback: `exec_decoded`'s caller
/// can't tell "please retry me through the interpreter" apart from a real
/// retirement, since both return `EXEC_COMPLETE`).
#[cfg(feature = "jitv2")]
unsafe extern "C" fn jit_interp_fallback<T: Tlb, C: MipsCache>(ctx: *mut core::ffi::c_void) -> u32 {
    let exec = unsafe { &mut *(ctx as *mut MipsExecutor<T, C>) };
    exec.interp_dispatch_one()
}

/// Un-publish `offset`'s entry on the executor's currently-tracked physical
/// code page (`self.pcp`) — see `MipsCore::kill_entry_fn`'s doc comment for
/// why compiled code needs this (the FR-mismatch case: the whole compiled
/// unit is wrong, not just this one dispatch, so it must never be
/// re-selected by the JIT gate). `self.pcp` is guaranteed to still be the
/// same page this compiled function was itself entered from: nothing
/// between `exec_decoded`'s dispatch and this call touches `core.pc`'s page
/// (the guard runs first, in `entry_block`, before any real instruction
/// semantics) — `jitv2_track_pcp` only re-derives `self.pcp` on a Fetch
/// nanotlb miss, which a same-page guard check can't trigger.
#[cfg(feature = "jitv2")]
unsafe extern "C" fn jit_kill_entry<T: Tlb, C: MipsCache>(ctx: *mut core::ffi::c_void, offset: u32) {
    let exec = unsafe { &mut *(ctx as *mut MipsExecutor<T, C>) };
    assert!(!exec.pcp.is_null(), "jit_kill_entry reached with no tracked PhysicalCodePage");
    let page = unsafe { &*exec.pcp };
    page.kill(offset as usize);
    #[cfg(feature = "developer")]
    exec.jitv2.lock().stats.kill_entry_calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Developer per-instruction hook for compiled code (`core.dev_trace_bp_fn`,
/// installed only under `developer`). Called from every instruction's block
/// with the synthesized `pc`/`raw`/`bd`, right where the interrupt preamble
/// runs (`codegen::emit_dev_trace_bp`) — so a compiled region's instructions
/// show up in `dt` (tagged JIT-executed) and can hit PC breakpoints, exactly
/// like the interpreter's own `step()`. Records the traceback entry and the
/// trace-file record, then checks PC breakpoints against `pc`; returns
/// `EXEC_BREAKPOINT` on a hit (compiled code bails to its exit with that
/// status, `core.pc` left at `pc` for resume) or `EXEC_COMPLETE` otherwise.
/// `bd` is `core.in_delay_slot` for this instruction — accepted for parity
/// with the interpreter's record shape and future use; the traceback itself
/// keys on pc/raw. Not emitted for the entry word's external-dispatch arm:
/// `step()` already recorded that PC before dispatching into the JIT (only the
/// preamble-bearing internal-edge block calls this — see
/// `compile_region`'s placement).
/// Proof-of-call counter for `jit_dev_trace_bp` — incremented unconditionally
/// at hook entry (before the `lightning`-gated push), so `j2 stats` can show
/// whether the hook is even being reached, independent of whether `dt`'s
/// traceback ends up populated. Diagnostic only.
#[cfg(all(feature = "jitv2", feature = "developer"))]
pub static DEV_TRACE_BP_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[cfg(all(feature = "jitv2", feature = "developer"))]
unsafe extern "C" fn jit_dev_trace_bp<T: Tlb, C: MipsCache>(ctx: *mut core::ffi::c_void, pc: u64, raw: u32, origin: u32) -> u32 {
    DEV_TRACE_BP_CALLS.fetch_add(1, Ordering::Relaxed);
    let exec = unsafe { &mut *(ctx as *mut MipsExecutor<T, C>) };
    #[cfg(not(feature = "lightning"))]
    {
        exec.traceback.push(pc, raw, InstrOrigin::from_u32(origin));
        if let Some(w) = exec.trace_writer.as_mut() {
            let record = crate::trace::TraceRecord::capture(pc, raw, &exec.core);
            let _ = w.push(&record);
        }
        // PC breakpoint, honoring the one-shot skip_breakpoints the monitor
        // sets to resume past a hit (same as step()'s own pre-dispatch check).
        if exec.bp_enabled() && exec.check_breakpoint::<{ BpType::Pc as u8 }>(pc) {
            return EXEC_BREAKPOINT;
        }
    }
    EXEC_COMPLETE
}

/// FPU status hooks: not generic over `<T,C>` (no executor context needed —
/// these wrap host-arch free functions in `platform.rs` directly, same ones
/// `MipsExecutor::fpu_update_fcsr` itself calls). `ctx` is accepted for
/// signature uniformity with the other hooks but unused.
#[cfg(feature = "jitv2")]
unsafe extern "C" fn jit_fpu_get_status(_ctx: *mut core::ffi::c_void) -> u32 {
    crate::platform::get_fpu_status()
}
#[cfg(feature = "jitv2")]
unsafe extern "C" fn jit_fpu_clear_status(_ctx: *mut core::ffi::c_void) {
    crate::platform::clear_fpu_status()
}
#[cfg(feature = "jitv2")]
unsafe extern "C" fn jit_fpu_set_mode(_ctx: *mut core::ffi::c_void, rm: u32) {
    crate::platform::set_fpu_mode(rm as u8)
}

/// Inline lockstep step hook (`core.lockstep_step_fn`), called by compiled code
/// at the START of every JIT instruction with its synthesized `pc`/`raw`/`bd`.
/// Establishes the true per-instruction starting state, runs the instruction
/// once through the real interpreter to get the reference result, then restores
/// the starting state so the JIT can run the same instruction for real. See
/// `MipsCore::lockstep_step_fn`. No-op (returns without comparing) for a
/// `LockstepClass` that can't be compared per-dispatch (Branch/Other).
#[cfg(feature = "jitv2_lockstep")]
unsafe extern "C" fn jit_lockstep_step<T: Tlb, C: MipsCache>(ctx: *mut core::ffi::c_void, pc: u64, raw: u32, bd: u32) {
    let exec = unsafe { &mut *(ctx as *mut MipsExecutor<T, C>) };
    exec.lockstep_step(pc, raw, bd);
}

/// Inline lockstep compare hook (`core.lockstep_compare_fn`), called by compiled
/// code at the END of every JIT instruction. Compares the JIT's now-current
/// state against the interpreter reference `lockstep_step` stashed. On a
/// divergence it prints a full report and returns `EXEC_BREAKPOINT` so compiled
/// code bails to the monitor (rather than aborting the process via a panic).
/// Returns `EXEC_COMPLETE` when equal or when there was nothing to compare
/// (disabled class / non-retiring reference).
#[cfg(feature = "jitv2_lockstep")]
unsafe extern "C" fn jit_lockstep_compare<T: Tlb, C: MipsCache>(ctx: *mut core::ffi::c_void) -> u32 {
    let exec = unsafe { &mut *(ctx as *mut MipsExecutor<T, C>) };
    if exec.lockstep_compare() { EXEC_BREAKPOINT } else { EXEC_COMPLETE }
}

/// MIPS FCSR.RM encoding, shared by the portable rounding primitive below
/// and every CVT.W/CVT.L handler that now honors it dynamically (ROUND/
/// TRUNC/CEIL/FLOOR pass a fixed mode instead — see `RM_*` usage at each
/// call site).
pub const RM_NEAREST_EVEN: u8 = 0;
pub const RM_TOWARD_ZERO: u8 = 1;
pub const RM_TOWARD_POS_INF: u8 = 2;
pub const RM_TOWARD_NEG_INF: u8 = 3;

/// Round `x` to the nearest integer *value* (still returned as a float, not
/// cast to an integer type) per `rm` (MIPS FCSR.RM encoding: 0=nearest-even,
/// 1=toward-zero, 2=toward+inf, 3=toward-inf) — pure bit manipulation on the
/// mantissa, no hardware rounding instruction (`ROUNDSD`, `FRINTX`, etc.)
/// anywhere in the implementation, so it can't inherit ambient host FPU
/// control-register state by construction. See `exec_fround_l_s`'s doc
/// comment for why this replaced `f64::round()`/`.trunc()`/`.ceil()`/
/// `.floor()`: those were empirically found to be sensitive to the host
/// MXCSR rounding-control bits on this build, silently producing a
/// different answer than a fresh, isolated build of the identical
/// expression — not safe to build architecturally-defined rounding on top
/// of, regardless of the exact mechanism.
///
/// NaN and infinities pass through unchanged (the caller's saturating cast
/// to integer handles those; rounding them is a no-op/undefined either way).
/// Magnitudes already >= 2^52 (f64's mantissa width) are already
/// integer-valued in IEEE-754 — returned unchanged rather than risking a
/// shift-by-more-than-63.
pub(crate) fn round_f64_to_int_mode(x: f64, rm: u8) -> f64 {
    if !x.is_finite() {
        return x;
    }
    const MANTISSA_BITS: u32 = 52;
    const EXP_BIAS: i32 = 1023;
    let bits = x.to_bits();
    let sign = bits >> 63;
    let biased_exp = ((bits >> MANTISSA_BITS) & 0x7FF) as i32;
    let exp = biased_exp - EXP_BIAS; // x = 1.mantissa * 2^exp (or 0.mantissa * 2^-1022 if biased_exp==0)
    if exp >= MANTISSA_BITS as i32 {
        return x; // already integer-valued (or zero)
    }
    if exp < 0 {
        // |x| < 1: result is 0 or ±1 depending on mode/magnitude.
        let is_zero_mantissa = (bits & ((1u64 << MANTISSA_BITS) - 1)) == 0 && biased_exp == 0;
        if x == 0.0 || (exp < -1) {
            // Magnitude < 0.5 (or exactly zero): rounds to 0 toward-zero/
            // nearest; toward ±inf still rounds to 0 unless sign pushes it
            // to ±1 in the away-from-origin direction for that mode.
            return match rm {
                RM_TOWARD_POS_INF if sign == 0 && !is_zero_mantissa => 1.0,
                RM_TOWARD_NEG_INF if sign == 1 && !is_zero_mantissa => -1.0,
                _ => if sign == 1 { -0.0 } else { 0.0 },
            };
        }
        // exp == -1: |x| in [0.5, 1.0).
        let is_exactly_half = (bits & ((1u64 << MANTISSA_BITS) - 1)) == 0; // mantissa all-zero => |x| == 0.5 exactly
        let round_away = match rm {
            RM_NEAREST_EVEN => !is_exactly_half, // ties round to 0 (the even integer)
            RM_TOWARD_ZERO => false,
            RM_TOWARD_POS_INF => sign == 0,
            RM_TOWARD_NEG_INF => sign == 1,
            _ => unreachable!(),
        };
        return if round_away { if sign == 1 { -1.0 } else { 1.0 } } else if sign == 1 { -0.0 } else { 0.0 };
    }
    // 0 <= exp < 52: split mantissa into an integer part and a fractional
    // remainder at bit position (MANTISSA_BITS - exp).
    let frac_bits = MANTISSA_BITS as i32 - exp; // 1..=52
    let frac_mask = (1u64 << frac_bits) - 1;
    let full_mantissa = (bits & ((1u64 << MANTISSA_BITS) - 1)) | (1u64 << MANTISSA_BITS); // implicit leading 1
    let frac = full_mantissa & frac_mask;
    if frac == 0 {
        return x; // already exact integer
    }
    let truncated_bits = bits & !frac_mask; // x with the fractional bits cleared (magnitude truncated toward zero)
    let half = 1u64 << (frac_bits - 1);
    let round_up_magnitude = match rm {
        RM_TOWARD_ZERO => false,
        RM_TOWARD_POS_INF => sign == 0,
        RM_TOWARD_NEG_INF => sign == 1,
        RM_NEAREST_EVEN => {
            if frac > half {
                true
            } else if frac < half {
                false
            } else {
                // Exact tie: round to even, i.e. up iff the truncated
                // integer's LSB (bit `frac_bits` of the original mantissa)
                // is 1.
                ((full_mantissa >> frac_bits) & 1) != 0
            }
        }
        _ => unreachable!(),
    };
    if round_up_magnitude {
        // Increment the truncated value by one unit at bit position
        // `frac_bits` — done as *integer* addition on the raw bit pattern
        // (not `truncated_value + 2f64.powi(exp)`) so a mantissa overflow
        // correctly carries into the exponent field (e.g. 1.9999999999999998
        // rounding up to 2.0, where the mantissa is all-ones and the carry
        // must ripple all the way to bit 52).
        f64::from_bits(truncated_bits + (1u64 << frac_bits))
    } else {
        f64::from_bits(truncated_bits)
    }
}

/// `f32` counterpart of `round_f64_to_int_mode` — same algorithm, 23-bit
/// mantissa / 8-bit exponent (bias 127) instead of 52/11 (bias 1023).
fn round_f32_to_int_mode(x: f32, rm: u8) -> f32 {
    if !x.is_finite() {
        return x;
    }
    const MANTISSA_BITS: u32 = 23;
    const EXP_BIAS: i32 = 127;
    let bits = x.to_bits();
    let sign = bits >> 31;
    let biased_exp = ((bits >> MANTISSA_BITS) & 0xFF) as i32;
    let exp = biased_exp - EXP_BIAS;
    if exp >= MANTISSA_BITS as i32 {
        return x;
    }
    if exp < 0 {
        let is_zero_mantissa = (bits & ((1u32 << MANTISSA_BITS) - 1)) == 0 && biased_exp == 0;
        if x == 0.0 || (exp < -1) {
            return match rm {
                RM_TOWARD_POS_INF if sign == 0 && !is_zero_mantissa => 1.0,
                RM_TOWARD_NEG_INF if sign == 1 && !is_zero_mantissa => -1.0,
                _ => if sign == 1 { -0.0 } else { 0.0 },
            };
        }
        let is_exactly_half = (bits & ((1u32 << MANTISSA_BITS) - 1)) == 0;
        let round_away = match rm {
            RM_NEAREST_EVEN => !is_exactly_half,
            RM_TOWARD_ZERO => false,
            RM_TOWARD_POS_INF => sign == 0,
            RM_TOWARD_NEG_INF => sign == 1,
            _ => unreachable!(),
        };
        return if round_away { if sign == 1 { -1.0 } else { 1.0 } } else if sign == 1 { -0.0 } else { 0.0 };
    }
    let frac_bits = MANTISSA_BITS as i32 - exp;
    let frac_mask = (1u32 << frac_bits) - 1;
    let full_mantissa = (bits & ((1u32 << MANTISSA_BITS) - 1)) | (1u32 << MANTISSA_BITS);
    let frac = full_mantissa & frac_mask;
    if frac == 0 {
        return x;
    }
    let truncated_bits = bits & !frac_mask;
    let half = 1u32 << (frac_bits - 1);
    let round_up_magnitude = match rm {
        RM_TOWARD_ZERO => false,
        RM_TOWARD_POS_INF => sign == 0,
        RM_TOWARD_NEG_INF => sign == 1,
        RM_NEAREST_EVEN => {
            if frac > half {
                true
            } else if frac < half {
                false
            } else {
                ((full_mantissa >> frac_bits) & 1) != 0
            }
        }
        _ => unreachable!(),
    };
    if round_up_magnitude {
        // See round_f64_to_int_mode's comment: integer add on the raw bits
        // so a mantissa-overflow carry ripples into the exponent correctly.
        f32::from_bits(truncated_bits + (1u32 << frac_bits))
    } else {
        f32::from_bits(truncated_bits)
    }
}

impl<T: Tlb, C: MipsCache> MipsExecutor<T, C> {
    /// Create a new executor from a config and a bus (sysad) and a TLB.
    /// The cache hierarchy is constructed internally as a unified R4000Cache.
    pub fn new(sysad: Arc<dyn BusDevice>, tlb: T, cfg: &MipsCpuConfig) -> Self
    where
        C: From<Arc<dyn BusDevice>>
    {
        let mut core = MipsCore::new();

        // Build unified cache hierarchy. Cache geometry is fixed at compile time;
        // IC_SIZE/IC_LINE/DC_SIZE/DC_LINE/L2_SIZE/L2_LINE are consts from mips_cache_v2.
        // `mut` is only used by the Triton L2-enable sync below.
        #[cfg_attr(not(feature = "r5ksc_triton"), allow(unused_mut))]
        let mut cache = C::from(sysad.clone());

        // Build CP0 Config register from architecture constants.
        let mut config = 0u32;

        // K0 (bits 2:0): kseg0 coherency algorithm. 3 = Cacheable, non-coherent.
        config |= 3 << CONFIG_K0;

        // DB (bit 4): Primary D-cache line size. 0=16B, 1=32B.
        config |= (if DC_LINE >= 32 { 1 } else { 0 }) << CONFIG_DB;

        // IB (bit 5): Primary I-cache line size. 0=16B, 1=32B.
        config |= (if IC_LINE >= 32 { 1 } else { 0 }) << CONFIG_IB;

        // DC (bits 8:6): Primary D-cache size. size = 2^(12+DC)
        config |= DC_SIZE.trailing_zeros().saturating_sub(12) << CONFIG_DC;

        // IC (bits 11:9): Primary I-cache size. size = 2^(12+IC)
        config |= IC_SIZE.trailing_zeros().saturating_sub(12) << CONFIG_IC;

        // BE (bit 15): Big Endian. 1 for Indy.
        config |= 1 << CONFIG_BE;

        // SC (bit 17): Secondary cache present. 0=present, 1=absent.
        // SC=0: PROM detects L2 via CACH_SD|C_ILT probe → rmi_cacheflush (inclusive L2 path).
        // SC=1: size_2nd_cache() returns 0 → PROM reads L2 size from EEPROM, sets
        //       _two_set_pcaches=icache/2 → __cache_wb_inval does index IWBINV on both
        //       L1D ways + index IINV on both L1I ways (correct for R5K non-inclusive L2).
        // r5ksc_triton: SC=0 — Triton reports integrated L2 (PROM detects it via probe).
        // r5k without r5ksc: SC=1 — no L2 present; PROM uses 2-way index flush.
        // r5k + r5ksc (external): SC=1 — external cache sized via EEPROM; 2-way flush.
        // R4K: SC=0 when L2 present, SC=1 when absent.
        #[cfg(feature = "r5ksc_triton")]
        { config |= 0 << CONFIG_SC; } // Triton: SC=0, integrated L2 present
        #[cfg(all(feature = "r5k", not(feature = "r5ksc_triton")))]
        { config |= 1 << CONFIG_SC; } // R5K non-Triton: SC=1, PROM uses 2-way index flush
        #[cfg(not(feature = "r5k"))]
        { config |= (if L2_SIZE > 0 { 0 } else { 1 }) << CONFIG_SC; }

        // SB (bits 23:22): Secondary cache block size.
        // 00=4 words (16B), 01=8 words (32B), 10=16 words (64B), 11=32 words (128B).
        config |= (match L2_LINE {
            16  => 0b00,
            32  => 0b01,
            64  => 0b10,
            128 => 0b11,
            _   => 0b11,
        }) << CONFIG_SB;

        // Triton CONFIG_TR_SS (bits 21:20): secondary cache size.
        // IP22 PROM probes L2 dynamically and ignores these bits.
        // IP32 (O2) reads CONFIG[21:20] directly — must match for correct L2 detection.
        #[cfg(feature = "r5ksc_triton")]
        {
            let ss: u32 = match L2_SIZE {
                524288  => 0b00,  // 512 KB
                1048576 => 0b01,  // 1 MB
                2097152 => 0b10,  // 2 MB
                _       => 0b11,  // none / 4 MB
            };
            config |= ss << CONFIG_TR_SS;
        }

        core.cp0_config = config;
        core.tlb_entries = cfg.tlb_entries as u32;

        // Triton: sync initial L2 enabled state from Config SE bit (starts 0 = disabled).
        #[cfg(feature = "r5ksc_triton")]
        cache.set_l2_enabled((config >> CONFIG_SE) & 1 != 0);

        /*eprintln!("Cache config: L1I {}KB/{}B-line  L1D {}KB/{}B-line  L2 {}KB/{}B-line  CP0.Config={:#010x}",
            ic_size / 1024, ic_line,
            dc_size / 1024, dc_line,
            l2_size / 1024, l2_line,
            config);*/


        let mut executor = Self {
            core,
            sysad,
            cheritest_dump_hook: false,
            tlb,
            cache,
            #[cfg(feature = "developer")]
            undo_buffer: UndoBuffer::new(),
            #[cfg(feature = "developer")]
            pending_memory_writes: Vec::new(),
            traceback: TracebackBuffer::new(),
            #[cfg(feature = "developer")]
            trace_writer: None,
            #[cfg(feature = "idle-pause")]
            idle_profiler: IdleProfiler::default(),
            #[cfg(feature = "idle-pause")]
            idle_profile_on: Arc::new(AtomicBool::new(false)),
            #[cfg(feature = "idle-pause")]
            idle_profile_reset: Arc::new(AtomicBool::new(false)),
            #[cfg(feature = "idle-pause")]
            idle_profile_on_ptr: std::ptr::null(),
            symbols: Arc::new(Mutex::new(SymbolTable::new())),
            breakpoints: vec![Breakpoint {
                id: 0, addr: 0, kind: BpType::Pc, enabled: false, condition: None
            }],
            next_bp_id: 1,
            last_bp_hit: None,
            pc_bp_count: 0,
            mem_bp_count: 0,
            skip_breakpoints: false,
            #[cfg(feature = "developer")]
            skip_interrupts: false,
            ins: DecodedInstr::default(), // scratch slot for uncached fetches
            decoded_count: Arc::new(AtomicU64::new(0)),
            uncached_fetch_count: Arc::new(AtomicU64::new(0)),
            // Placeholder — overwritten immediately by update_translate_fn below.
            translate_fn: translate_32_kernel::<T, C>,
            // Placeholder — overwritten immediately by update_fpr_mode below.
            fpr_read_d:  crate::mips_core::read_fpr_d_fr0,
            fpr_write_d: crate::mips_core::write_fpr_d_fr0,
            fpr_read_l:  crate::mips_core::read_fpr_l_fr0,
            fpr_write_l: crate::mips_core::write_fpr_l_fr0,
            fpr_read_w:  crate::mips_core::read_fpr_w_fr0,
            fpr_write_w: crate::mips_core::write_fpr_w_fr0,
            cached_pending: 0,
            #[cfg(feature = "instr_stats")]
            instr_stats: crate::mips_instr_stats::InstrStats::default(),
            // Standalone default — production (`Machine::new`) immediately
            // replaces this with its own shared `Arc<Mutex<Jitv2>>` (mirrors
            // the `l1i_fetch_count`/`rebind_atomic_ptrs` Arc-injection pattern
            // right below `Arc::new(MipsCpu::new(executor))` — `Jitv2` can't
            // be constructed before `MipsExecutor::new` returns here, since
            // `Machine`'s own field ordering needs `executor` to exist
            // first). Kept as a real, working default rather than an empty
            // placeholder so every non-`Machine` caller (equiv_test.rs's
            // ~30 direct `MipsExecutor::new()` sites, none of which go
            // through `Machine` at all) still gets a fully functional,
            // self-contained jitv2 pool with no special-casing.
            #[cfg(feature = "jitv2")]
            jitv2: std::sync::Arc::new(Mutex::new(crate::jitv2::Jitv2::new(crate::jitv2::JITV2_INITIAL_PAGE_CAPACITY))),
            #[cfg(feature = "jitv2")]
            pcp: std::ptr::null_mut(),
            #[cfg(feature = "jitv2_lockstep")]
            ls_before: None,
            #[cfg(feature = "jitv2_lockstep")]
            ls_expected: None,
            #[cfg(feature = "jitv2_lockstep")]
            ls_delay_target_before: 0,
            #[cfg(feature = "jitv2_lockstep")]
            ls_delay_target_expected: 0,
            // Threaded compile is now the default — see Machine::new's
            // compile_queue.start() call, which must run whenever this is
            // `false` (an un-started queue would silently drop every
            // compile request: `try_schedule`/`compile_queue.send` succeed,
            // nothing is ever there to pop them). `j2 inline on` still
            // switches back to this executor thread doing its own inline
            // compiles, same as before.
            #[cfg(feature = "jitv2")]
            jitv2_inline_compile: false,
            #[cfg(feature = "jitv2")]
            jitv2_inline_analyzer: crate::jitv2::analyzer::Analyzer::new(),
            #[cfg(feature = "jitv2")]
            jitv2_dispatch_enabled: true,
            #[cfg(feature = "developer")]
            hw_read_fixup_replay: None,
            #[cfg(feature = "developer")]
            hw_read_fixup_recorded: Vec::new(),
            #[cfg(feature = "developer")]
            hw_read_fixup_recording: false,
            #[cfg(feature = "developer")]
            last_step_status: EXEC_COMPLETE,
        };

        executor.rebind_atomic_ptrs();
        executor.update_translate_fn();
        executor.update_fpr_mode();

        executor
    }

    /// Re-sync raw atomic pointers after the shared Arcs are injected post-construction.
    /// (`cycles` has no equivalent — it's an inline `MipsCore` field now, always current.)
    pub fn rebind_atomic_ptrs(&mut self) {
        #[cfg(feature = "idle-pause")]
        { self.idle_profile_on_ptr = Arc::as_ptr(&self.idle_profile_on); }
    }

    /// Raw pointer to this executor's `MipsCore.interrupts` word, for devices
    /// on other threads that need to set/clear interrupt bits (e.g.
    /// `Ioc::set_interrupts`). Always re-derived from `self` at call time —
    /// **never cache this pointer past a move of the executor** (e.g. across
    /// `MipsExecutor` being moved into its owning `Arc<Mutex<...>>`). Callers
    /// should call this once the executor is at its final, stable address
    /// (inside that `Arc`) and treat the result as valid only from then on —
    /// which holds for the rest of the process, since the executor never
    /// moves again once there.
    pub fn interrupts_ptr(&self) -> *const AtomicU64 {
        &self.core.hot.interrupts as *const AtomicU64
    }

    /// Pointer to this executor's `MipsCore.hot.cycles` word, for readers on
    /// other threads (status displays, perf monitoring, `Wd33c93a`'s
    /// deferred-interrupt spin-wait) — see `CyclesPtr`/`Hot::cycles`'s doc
    /// comments. Same "call once at final address, valid for the rest of the
    /// process" contract as `interrupts_ptr` above.
    pub fn cycles_ptr(&self) -> crate::mips_core::CyclesPtr {
        crate::mips_core::CyclesPtr::new(&self.core.hot.cycles as *const u64)
    }

    /// Install the CP0 Status change callback pointing at this executor.
    /// Call once after construction. The callback is invoked (from write_cp0) with
    /// (old_status, new_status) whenever CP0 register 12 is written.
    pub fn install_status_cb(&mut self) {
        let ctx = self as *mut Self as *mut core::ffi::c_void;
        self.core.status_changed_cb = Some((mips_executor_status_cb::<T, C>, ctx));
    }

    /// Install JIT v2's memory-access and exception-delivery hooks
    /// (`MipsCore`'s `read*_fn`/`write*_fn`/`handle_exception_fn` fields —
    /// see their doc comments). Same discipline as `interrupts_ptr`: call
    /// this once the executor is at its final, stable address (inside its
    /// owning `Arc<Mutex<...>>`), never before — `ctx` is `self`'s address,
    /// which must not move again afterward for the life of the process.
    #[cfg(feature = "jitv2")]
    pub fn install_jit_hooks(&mut self) {
        let ctx = self as *mut Self as *mut core::ffi::c_void;
        self.core.jit_ctx = ctx;
        self.core.read8_fn = jit_read8::<T, C>;
        self.core.read16_fn = jit_read16::<T, C>;
        self.core.read32_fn = jit_read32::<T, C>;
        self.core.read64_fn = jit_read64::<T, C>;
        self.core.write8_fn = jit_write8::<T, C>;
        self.core.write16_fn = jit_write16::<T, C>;
        self.core.write32_fn = jit_write32::<T, C>;
        self.core.write64_fn = jit_write64::<T, C>;
        self.core.write64_masked_fn = jit_write64_masked::<T, C>;
        self.core.handle_exception_fn = jit_handle_exception::<T, C>;
        self.core.interp_fallback_fn = jit_interp_fallback::<T, C>;
        self.core.kill_entry_fn = jit_kill_entry::<T, C>;
        #[cfg(feature = "developer")]
        { self.core.dev_trace_bp_fn = jit_dev_trace_bp::<T, C>; }
        self.core.fpu_get_status_fn = jit_fpu_get_status;
        self.core.fpu_clear_status_fn = jit_fpu_clear_status;
        self.core.fpu_set_mode_fn = jit_fpu_set_mode;
        #[cfg(feature = "jitv2_lockstep")]
        {
            self.core.lockstep_step_fn = jit_lockstep_step::<T, C>;
            self.core.lockstep_compare_fn = jit_lockstep_compare::<T, C>;
        }
    }

    /// Probe the nanotlb for `va` using slot AT (Fetch=0, Read=1, Write=2).
    /// On hit: returns the cached result with no function-pointer call.
    /// On miss: calls `translate_fn` (the slow path) and fills the slot on success.
    #[inline(always)]
    fn nanotlb_translate<const AT: u8>(&mut self, va: u64) -> TranslateResult {
        let va_page = va & !0xFFF;
        let slot = &self.core.nanotlb[AT as usize];
        if slot.matches(va_page) {
            #[cfg(feature = "tlbstats")]
            {
                let at = unsafe { std::mem::transmute::<u8, AccessType>(AT) };
                self.tlb.stats_nanotlb_hit(at);
            }
            return TranslateResult::ok(slot.phys_addr(va), slot.cache_attr_raw());
        }
        let at = unsafe { std::mem::transmute::<u8, AccessType>(AT) };
        #[cfg(feature = "tlbstats")]
        self.tlb.stats_nanotlb_miss(at);
        let result = (self.translate_fn)(self, va, at);
        if !result.is_exception() {
            self.core.nanotlb[AT as usize].fill_raw(va_page, result.phys as u64, result.status & 0x7);
            #[cfg(feature = "jitv2")]
            if AT == AccessType::Fetch as u8 {
                self.jitv2_track_pcp(result.phys);
            }
        }
        result
    }

    /// Invalidate the nanotlb and drop the cached PCP pointer along with it.
    /// `self.pcp` is only ever re-derived on a Fetch nanotlb *miss* (see
    /// `jitv2_track_pcp`) — after a nanotlb invalidate, the next fetch is
    /// guaranteed to miss and re-derive it, but nulling here means the pointer
    /// never reads as valid in the gap, and is cheap insurance once other paths
    /// (snapshot restore, rollback, SMC/DMA kill) start calling `mega_flush()`
    /// independently of `jitv2_track_pcp`, which would otherwise leave `self.pcp`
    /// dangling into a cleared pool until the next miss.
    /// Call this instead of `self.core.nanotlb_invalidate()` from executor code.
    #[inline(always)]
    fn nanotlb_invalidate(&mut self) {
        self.core.nanotlb_invalidate();
        #[cfg(feature = "jitv2")]
        { self.pcp = std::ptr::null_mut(); }
    }

    /// JIT v2: re-derive `self.pcp` if the fetch just landed on a different physical
    /// page than the one currently tracked (rules/jitv2/jit-v2-design.md §2.1 — PCPs
    /// are keyed by physical frame, never by VA). Cheap on the common case: same-page
    /// sequential/loop execution never touches the pool or its lookup map, only the
    /// nanotlb hit path above and a single PFN comparison here.
    #[cfg(feature = "jitv2")]
    #[inline(always)]
    fn jitv2_track_pcp(&mut self, phys_addr: u32) {
        let pfn = phys_addr / crate::jitv2::PAGE_SIZE;
        let same_page = !self.pcp.is_null() && unsafe { (*self.pcp).pfn == pfn };
        if same_page {
            return;
        }
        // Landing on a genuinely different physical page than the one this
        // executor was just tracking is itself compile-worthy, regardless of
        // which word within that page pc happens to be at — a strictly more
        // accurate and more general replacement for exec_decoded's old
        // `entry_offset == 0` proxy (which only caught this for the specific
        // case of a sequential fallthrough landing exactly on word 0).
        // Notably, this closes a real gap `entry_offset == 0` missed:
        // exception/TLB-refill vector entry — `deliver_exception`
        // (mips_core.rs) writes `core.pc` directly to a fixed vector address
        // and has no reason to know about jit_trigger (shared, non-jitv2-aware
        // exception delivery logic) — the general-exception vector in
        // particular (0x...80000180) lands at word-offset 0x60 within its
        // page, not 0, so it was never probed by the old check unless that
        // word happened to already be published from some earlier arrival.
        // A harmless, bounded over-trigger case: nanotlb_invalidate nulls
        // self.pcp unconditionally, so the next dispatch after any TLB
        // invalidate always takes this branch even if it lands back on an
        // already-tracked physical page — is_entry_valid still short-circuits
        // correctly either way, this just costs one redundant probe.
        self.core.jit_trigger = true;
        let page_base = pfn * crate::jitv2::PAGE_SIZE;
        let mut jit = self.jitv2.lock();
        match jit.page_for(pfn, page_base, self.sysad.as_ref()) {
            Some(slot) => self.pcp = jit.page_ptr(slot),
            None => {
                // Pool exhausted: `flush_from_cpu_thread` resets to initial
                // state and retries — always leaves room for at least one
                // fresh page. Called from the CPU thread itself (this
                // function), which is "as good as stopped" for its own
                // purposes — no need to pause it — so this only pauses the
                // *compile* queue internally, not the CPU.
                drop(jit); // re-locks internally below; nothing needs it held across that call
                unsafe { self.jitv2.lock().flush_from_cpu_thread(self.sysad.clone()); }
                // Only `self.pcp` is actually stale here — it's a raw
                // pointer into the `Vec<PhysicalCodePage>` mega_flush just
                // cleared. The nanotlb's own translation slots (Fetch/Read/
                // Write) are untouched by mega_flush (physical memory didn't
                // move, only the JIT's compiled-code bookkeeping did), so
                // there's no reason to blow those away too — just null the
                // dangling pointer directly and re-derive it below for the
                // exact page this fetch already landed on.
                self.pcp = std::ptr::null_mut();
                let mut jit = self.jitv2.lock();
                let slot = jit.page_for(pfn, page_base, self.sysad.as_ref())
                    .expect("flush must leave room for at least one page");
                self.pcp = jit.page_ptr(slot);
            }
        }
    }

    /// Re-derive `translate_fn` from the current CP0 Status register.
    /// Must be called after any write to Status (done automatically via the status callback)
    /// and at init/reset time.
    #[inline]
    pub fn update_translate_fn(&mut self) {
        use crate::mips_core::PrivilegeMode;
        let is_64bit = self.core.is_64bit_mode();
        let privilege = self.core.get_privilege_mode();
        self.translate_fn = match (is_64bit, privilege) {
            (false, PrivilegeMode::Kernel)     => translate_32_kernel::<T, C>,
            (false, PrivilegeMode::Supervisor) => translate_32_supervisor::<T, C>,
            (false, PrivilegeMode::User)       => translate_32_user::<T, C>,
            (true,  PrivilegeMode::Kernel)     => translate_64_kernel::<T, C>,
            (true,  PrivilegeMode::Supervisor) => translate_64_supervisor::<T, C>,
            (true,  PrivilegeMode::User)       => translate_64_user::<T, C>,
        };
    }

    /// Re-derive `fpr_read_d/write_d/read_l/write_l` from STATUS_FR.
    /// FR=0 (IRIX 5.3): even/odd 32-bit register pairs.
    /// FR=1 (IRIX 6.5): full 64-bit slots.
    #[inline]
    pub fn update_fpr_mode(&mut self) {
        use crate::mips_core::{
            read_fpr_d_fr0, write_fpr_d_fr0, read_fpr_l_fr0, write_fpr_l_fr0,
            read_fpr_w_fr0, write_fpr_w_fr0,
            read_fpr_d_fr1, write_fpr_d_fr1, read_fpr_l_fr1, write_fpr_l_fr1,
            read_fpr_w_fr1, write_fpr_w_fr1,
        };
        if (self.core.cp0_status & STATUS_FR) != 0 {
            self.fpr_read_d  = read_fpr_d_fr1;
            self.fpr_write_d = write_fpr_d_fr1;
            self.fpr_read_l  = read_fpr_l_fr1;
            self.fpr_write_l = write_fpr_l_fr1;
            self.fpr_read_w  = read_fpr_w_fr1;
            self.fpr_write_w = write_fpr_w_fr1;
        } else {
            self.fpr_read_d  = read_fpr_d_fr0;
            self.fpr_write_d = write_fpr_d_fr0;
            self.fpr_read_l  = read_fpr_l_fr0;
            self.fpr_write_l = write_fpr_l_fr0;
            self.fpr_read_w  = read_fpr_w_fr0;
            self.fpr_write_w = write_fpr_w_fr0;
        }
    }

    /// Called whenever CP0 Status is written.
    #[inline]
    fn on_cp0_status_changed(&mut self, _old: u32, _new: u32) {
        self.update_translate_fn();
        self.update_fpr_mode();
        self.nanotlb_invalidate();
    }

    /// Execute a single instruction (decode into scratch, then execute).
    ///
    /// Not on the `step()` hot path — this is the debug/test entry point
    /// (single-instruction injection, no TLB translation). `step()` derives
    /// `pcp` from `fetch_instr`'s nanotlb translation; callers here bypass
    /// that, so re-derive it directly from `core.pc` treated as already
    /// physical (true for every existing caller: tests use PassthroughTlb /
    /// identity-mapped low addresses). Keeps `exec_decoded`'s "pcp must be
    /// current" invariant real for this entry point too, without adding any
    /// cost to the interpreter's per-instruction hot loop.
    pub fn exec(&mut self, instr: u32) -> ExecStatus {
        #[cfg(feature = "jitv2")]
        self.jitv2_track_pcp(self.core.pc as u32);
        self.ins.raw = instr;
        self.ins.flags = FLAG_NOT_DECODED;
        decode_into::<T, C>(&mut self.ins);
        let d: *const DecodedInstr = &self.ins;
        self.exec_decoded(unsafe { &*d })
    }

    /// Returns true if breakpoints should fire. False when skip_breakpoints is set.
    #[inline(always)]
    fn bp_enabled(&self) -> bool {
        !self.skip_breakpoints
    }

    /// Terminal action for a handler taking a branch/jump: arm the delay slot
    /// and advance PC to the delay-slot instruction. If a delay slot is
    /// already active (branch-in-delay-slot, unusual but legal), the existing
    /// one is left alone — only its target-of-record differs, PC still just
    /// advances by 4 to fetch/execute whatever sits there. `core.in_delay_slot`
    /// is the real signal that a branch/jump was just taken (checked by e.g.
    /// gdbstub's step_one to know whether to also step the delay slot) — the
    /// return value carries no information of its own, every handler already
    /// sets `core.pc` itself.
    #[inline(always)]
    fn branch_delay(&mut self, target: u64) -> ExecStatus {
        self.core.delay_slot_target = target;
        if !self.core.in_delay_slot {
            self.core.in_delay_slot = true;
        }
        self.core.pc = self.core.pc.wrapping_add(4);
        EXEC_COMPLETE
    }

    /// Terminal action for a handler that completes normally: advance into a
    /// pending delay slot if one is active, else PC += 4. Every dispatch-target
    /// exec_* handler that completes without branching calls this as its last
    /// action instead of returning the bare EXEC_COMPLETE constant.
    #[inline(always)]
    fn handle_exec_complete(&mut self) -> ExecStatus {
        if self.core.in_delay_slot {
            self.core.pc = self.core.delay_slot_target;
            self.core.in_delay_slot = false;
            // The delay slot just retired: PC now holds the branch's actual
            // target, landing here for the first time this transfer. Mark it
            // as a compile-worthy arrival (jitv2_track_pcp's `AT == Fetch`
            // path can't tell "just-branched-to" from "sequential" on its own).
            #[cfg(feature = "jitv2")]
            { self.core.jit_trigger = true; }
        } else {
            self.core.pc = self.core.pc.wrapping_add(4);
        }
        EXEC_COMPLETE
    }

    /// Terminal action for "branch likely not taken" / a fused straight-line
    /// sequence completing: PC += 8, no delay-slot interaction.
    #[inline(always)]
    fn handle_branch_likely_skip(&mut self) -> ExecStatus {
        self.core.pc = self.core.pc.wrapping_add(8);
        EXEC_COMPLETE
    }

    /// Terminal action for a *non-likely* conditional branch's not-taken arm
    /// (BEQ/BNE/BLEZ/BGTZ/BLTZ/BGEZ/BLTZAL/BGEZAL — never the "likely" family,
    /// which annuls the slot entirely via `handle_branch_likely_skip` instead).
    /// Per the MIPS spec the delay slot always executes, taken or not — this
    /// is structurally identical to `branch_delay`, just with the resolved
    /// target fixed at `pc+8` instead of the branch's computed target. Before
    /// this existed, not-taken went through the same path as an ordinary
    /// completed instruction (`handle_exec_complete`'s plain `pc+=4`), which
    /// executes correctly (the delay slot's *next* dispatch just lands on
    /// pc+4 and runs normally either way) but leaves `core.in_delay_slot`
    /// false while the delay slot instruction executes — so if that
    /// instruction itself faults, `handle_exception` sees `in_delay_slot ==
    /// false` and gets `Cause.BD` and `EPC` wrong (should point at this
    /// branch with BD set, per spec; would instead point straight at the
    /// delay-slot instruction with BD clear).
    #[inline(always)]
    fn handle_branch_not_taken(&mut self) -> ExecStatus {
        self.branch_delay(self.core.pc.wrapping_add(8))
    }

    /// Terminal action for a handler that has already written `self.core.pc`
    /// directly and has no delay slot to arm (fused branch+NOP handlers,
    /// ERET): marks the new PC as a compile-worthy branch-target arrival —
    /// the counterpart to `handle_exec_complete`'s delay-slot-consuming case
    /// for handlers that skip the delay-slot dance entirely. Status codes no
    /// longer carry PC-related meaning (every handler sets core.pc itself),
    /// so this just returns EXEC_COMPLETE like everything else — jit_trigger
    /// is the real payload here.
    #[inline(always)]
    fn exec_complete_pc_set(&mut self) -> ExecStatus {
        #[cfg(feature = "jitv2")]
        { self.core.jit_trigger = true; }
        EXEC_COMPLETE
    }

    /// Finish a handler given a raw status straight out of read_data/write_data/
    /// translate (i.e. not yet run through handle_exception): dispatches to
    /// handle_exception if the EXEC_IS_EXCEPTION bit is set, otherwise passes
    /// EXEC_BREAKPOINT/EXEC_RETRY straight through unchanged (both are already
    /// terminal — no PC mutation needed, matching the old exec_decoded match's
    /// `EXEC_RETRY => {}` / `EXEC_BREAKPOINT => {}` arms).
    #[inline(always)]
    fn finish_status(&mut self, status: ExecStatus) -> ExecStatus {
        if status & EXEC_IS_EXCEPTION != 0 {
            self.handle_exception(status)
        } else if status == EXEC_COMPLETE {
            self.handle_exec_complete()
        } else {
            // EXEC_RETRY / EXEC_BREAKPOINT: already terminal, no PC mutation.
            status
        }
    }

    /// `lightning`'s decode-skip fast path: check whether jitv2 already has
    /// a valid compiled entry for `self.core.pc` and, if so, call straight
    /// into it — WITHOUT decoding the raw instruction word first. Only ever
    /// worth calling separately from `exec_decoded` under `jitv2` +
    /// `lightning` together: `jitv2_dispatch_enabled` is fixed `true` there
    /// (see `exec_decoded`'s own gate — `lightning`/`developer` are
    /// mutually exclusive, so the runtime switch that could turn dispatch
    /// off is unreachable), so this check is never wasted work the way it
    /// would be in a build where dispatch might be toggled off. Every other
    /// build keeps decoding unconditionally before dispatch — `jitv2_lockstep`
    /// needs the decoded instruction to classify it (ALU/branch/load-store/
    /// FPU), and non-`lightning` builds want `d` regardless of a JIT hit
    /// (traceback, trace recording).
    ///
    /// Must be called with `self.pcp` already current for this pc (i.e.
    /// after `fetch_instr`/`nanotlb_translate`, same precondition
    /// `exec_decoded`'s own gate documents) — mirrors that gate's hit path
    /// exactly, just without needing a `&DecodedInstr` to do it. Returns the
    /// compiled function's own status directly on a hit (decode was
    /// successfully avoided) or `EXEC_FALLBACK` on a miss (caller must fall
    /// through to the normal fetch-decode-`exec_decoded` path; nothing has
    /// been mutated on a miss, so falling through is always safe) — a plain
    /// `ExecStatus` sentinel rather than `Option<ExecStatus>` deliberately:
    /// this runs on every single dispatch under `jitv2`+`lightning`, and an
    /// `Option`'s extra discriminant check/branch is exactly the kind of
    /// cost a hot path shouldn't pay when a spare status bit does the same
    /// job for free (matches every other status code in this file — see
    /// HACKING.md's `ExecStatus` section).
    #[cfg(all(feature = "jitv2", feature = "lightning", not(feature = "jitv2_lockstep")))]
    #[inline(always)]
    fn jitv2_try_dispatch_without_decode(&mut self) -> ExecStatus {
        assert!(!self.pcp.is_null(), "jitv2_try_dispatch_without_decode reached with no tracked PhysicalCodePage");
        let page = unsafe { &mut *self.pcp };
        let entry_offset = ((self.core.pc & 0xFFF) >> 2) as usize;
        if (self.core.jit_trigger || page.is_published(entry_offset)) && page.is_entry_valid(entry_offset) {
            let func = page.entries[entry_offset].func;
            debug_assert!(!func.is_null(), "valid bit set with null func");
            let jit_fn: crate::jitv2::JitFn = unsafe { std::mem::transmute(func) };
            unsafe { jit_fn(&mut self.core as *mut MipsCore) }
        } else {
            EXEC_FALLBACK
        }
    }

    pub fn step(&mut self) -> ExecStatus {
        // Increment the real, shared cycle counter directly — no local
        // shadow, no batching. See Hot::cycles's doc comment for why: a
        // dispatch loop that stays entirely inside JIT-compiled code for a
        // long stretch must still make this visible to other threads as it
        // goes, not just whenever it happens to return to this outer loop.
        // Volatile, not a plain field write: guarantees the compiler can't
        // elide/hoist the write out of an unbounded loop, even though this
        // isn't a synchronizing atomic RMW (readers only need eventual
        // visibility of the count, not ordering against other memory).
        unsafe {
            let p = &mut self.core.hot.cycles as *mut u64;
            std::ptr::write_volatile(p, std::ptr::read_volatile(p).wrapping_add(1));
        }

        // ci_clock: the compare "timer" is a deterministic hot.cycles
        // threshold instead of an hptimer thread — check it here, before the
        // pending load below, so the fire is delivered within this same step.
        #[cfg(feature = "ci_clock")]
        if self.core.hot.cycles >= self.core.count_fire_cycle {
            // Next architectural match is a full 32-bit Count wrap away;
            // normally a Compare write re-arms much sooner.
            let wrap_ns = ((1u128 << 32) * 1_000_000_000) / self.core.count_hz as u128;
            self.core.count_fire_cycle = self.core.hot.cycles
                .saturating_add(wrap_ns as u64 / crate::mips_core::NS_PER_GUEST_CYCLE);
            self.core.hot.interrupts.fetch_or(crate::mips_core::CAUSE_IP7 as u64, Ordering::SeqCst);
            self.core.fasttick_count.fetch_add(1, Ordering::Relaxed);
        }

        /*
        // Reload external interrupt state every 16 instructions
        if self.core.hot.cycles & 0xF == 0 {
            self.cached_pending = self.core.hot.interrupts.load(Ordering::Relaxed);
        }
        let pending = self.cached_pending;
        */
        // this seems to be a wash or slightly better without a branch, relaxed atomic loads are essentially MOV
        let pending = self.core.hot.interrupts.load(Ordering::Relaxed);

        let pc = self.core.pc;

        // Spin/idle-loop PC sampler. Armed lock-free via the shared atomic, so
        // the CPU is never paused/resumed to enable it (resuming corrupts a
        // live kernel). Inert (one relaxed load + branch) when disarmed.
        #[cfg(feature = "idle-pause")]
        if unsafe { &*self.idle_profile_on_ptr }.load(Ordering::Relaxed) {
            if self.idle_profile_reset.load(Ordering::Relaxed) {
                self.idle_profiler.reset();
                self.idle_profile_reset.store(false, Ordering::Relaxed);
            }
            let ie = self.core.interrupts_enabled();
            self.idle_profiler.sample(pc, ie);
        }

        #[cfg(not(feature = "lightning"))]
        if self.bp_enabled() && self.check_breakpoint::<{ BpType::Pc as u8 }>(pc) {
            return EXEC_BREAKPOINT;
        }

        // No per-instruction CP0 Count work: Count is virtual (materialized
        // lazily on read from the wall-clock anchor in mips_core.rs), and the
        // Count==Compare interrupt arrives through `pending` like any device
        // line — armed as an hptimer one-shot on each Compare write.
        // Fast path: skip all signal/interrupt handling when nothing is pending
        if (pending | self.core.cp0_cause as u64) != 0 {
            // Soft reset (bit 63)
            if pending & SOFT_RESET_BIT != 0 {
                self.core.reset(true); // clears interrupts word (including bit 63)
                self.core.in_delay_slot = false;
                self.core.delay_slot_target = 0;
                return EXEC_COMPLETE;
            }

            // Merge external IP bits into Cause
            self.core.cp0_cause = (self.core.cp0_cause & !EXT_INT_MASK) | (pending as u32 & EXT_INT_MASK);

            #[cfg(feature = "developer")]
            let skip_int = self.skip_interrupts;
            #[cfg(not(feature = "developer"))]
            let skip_int = false;

            if self.core.interrupts_enabled() && !skip_int {
                let ip = self.core.cp0_cause & crate::mips_core::CAUSE_IP_MASK;
                let im = self.core.cp0_status & crate::mips_core::STATUS_IM_MASK;
                if (ip & im) != 0 {
                    let s = exec_exception(EXC_INT);
                    return self.handle_exception(s);
                }
            }
        }

        let fetch = self.fetch_instr(pc);
        let result = if fetch.status == EXEC_COMPLETE {
            // jitv2+lightning: self.pcp is already current (fetch_instr's
            // nanotlb_translate just ran jitv2_track_pcp) — check for a
            // compiled hit before paying for decode_into below, since a hit
            // calls straight into the compiled function and never looks at
            // the decoded instruction at all. See
            // jitv2_try_dispatch_without_decode's own doc comment for why
            // this is only safe/worthwhile under this exact build combo.
            #[cfg(all(feature = "jitv2", feature = "lightning", not(feature = "jitv2_lockstep")))]
            {
                let status = self.jitv2_try_dispatch_without_decode();
                if status != EXEC_FALLBACK {
                    return status;
                }
            }
            {
                let slot = fetch.instr as *mut DecodedInstr;
                let d = unsafe { &mut *slot };
                if d.flags != 0 {
                    decode_into::<T, C>(d);
                    #[cfg(feature = "instr_stats")]
                    self.instr_stats.record_decode(d.op, d.rs, d.rt, d.funct);
                } else {
                    #[cfg(feature = "developer")]
                    self.decoded_count.fetch_add(1, Ordering::Relaxed);
                }
            }
            let d = unsafe { &*fetch.instr };
            // Traceback/trace-file recording moved into exec_decoded's own
            // dispatch decision (see its doc comment) instead of pushing
            // here unconditionally: this call doesn't yet know whether pc
            // is about to jump straight into a compiled function (any
            // `is_entry_valid` offset, not just a region's own word 0 — a
            // mid-region word independently published via an earlier
            // back-edge visit counts too) or actually run through the
            // interpreter. Pushing Interp here unconditionally, then
            // separately also recording inside the JIT path, produced a
            // real live bug: every JIT-hit dispatch got a spurious Interp
            // entry in `dt` (not just the entry word — `wd93_hpc1init`'s
            // whole straight-line body showed Interp despite genuinely
            // running under JIT, confirmed via `lockstep_jit_write` firing
            // for it — found live). Exactly one push per real dispatch now,
            // decided by whichever path actually ran.
            self.exec_decoded(d)
        } else if fetch.status & EXEC_IS_EXCEPTION != 0 {
            self.handle_exception(fetch.status)
        } else {
            fetch.status
        };

        if cfg!(not(feature = "lightning")) {
            self.skip_breakpoints = false;
        }
        result
    }

    /// Begin recording a per-instruction execution trace to `path`
    /// (`src/trace.rs`, rules/jitv2's lockstep verification tooling).
    /// Overwrites any existing file at `path`. No-op-safe to call while
    /// already recording — starts a fresh file, silently dropping (not
    /// flushing) whatever writer was active before, so callers that want a
    /// clean handoff should `trace_stop` first.
    #[cfg(feature = "developer")]
    pub fn trace_start(&mut self, path: &std::path::Path) -> std::io::Result<()> {
        self.trace_writer = Some(crate::trace::TraceWriter::create(path)?);
        Ok(())
    }

    /// Stop recording, flushing and closing the trace file. Returns the
    /// number of records written. No-op (returns 0) if not recording.
    #[cfg(feature = "developer")]
    pub fn trace_stop(&mut self) -> std::io::Result<u64> {
        match self.trace_writer.take() {
            Some(mut w) => {
                w.flush()?;
                Ok(w.count())
            }
            None => Ok(0),
        }
    }

    /// Whether a trace is currently being recorded.
    #[cfg(feature = "developer")]
    pub fn trace_active(&self) -> bool {
        self.trace_writer.is_some()
    }

    /// Lightweight step for JIT interpreter bursts. Skips the breakpoint /
    /// idle-profiler preamble — the JIT dispatch loop credits
    /// `core.hot.cycles` itself, in bulk, after the burst (still onto the
    /// same real, shared field — see `Hot::cycles`'s doc comment). Keeps
    /// interrupt checking because the kernel depends on per-instruction
    /// interrupt delivery.
    #[inline(always)]
    pub fn step_lite(&mut self) -> ExecStatus {
        let pending = self.core.hot.interrupts.load(Ordering::Relaxed);

        let pc = self.core.pc;

        if (pending | self.core.cp0_cause as u64) != 0 {
            if pending & SOFT_RESET_BIT != 0 {
                self.core.reset(true);
                self.core.in_delay_slot = false;
                self.core.delay_slot_target = 0;
                return EXEC_COMPLETE;
            }
            self.core.cp0_cause = (self.core.cp0_cause & !EXT_INT_MASK) | (pending as u32 & EXT_INT_MASK);
            if self.core.interrupts_enabled() {
                let ip = self.core.cp0_cause & crate::mips_core::CAUSE_IP_MASK;
                let im = self.core.cp0_status & crate::mips_core::STATUS_IM_MASK;
                if (ip & im) != 0 {
                    let s = exec_exception(EXC_INT);
                    return self.handle_exception(s);
                }
            }
        }

        let fetch = self.fetch_instr(pc);
        if fetch.status != EXEC_COMPLETE {
            return if fetch.status & EXEC_IS_EXCEPTION != 0 {
                self.handle_exception(fetch.status)
            } else {
                fetch.status
            };
        }
        {
            let slot = fetch.instr as *mut DecodedInstr;
            let d = unsafe { &mut *slot };
            if d.flags != 0 { decode_into::<T, C>(d); }
        }
        self.exec_decoded(unsafe { &*fetch.instr })
    }

    #[inline(always)]
    fn check_breakpoint<const KIND: u8>(&mut self, addr: u64) -> bool {
        if KIND == BpType::Pc as u8 {
            if self.pc_bp_count == 0 { return false; }
        } else {
            // Memory breakpoint
            if self.mem_bp_count == 0 { return false; }
        }

        let mut hit = false;
        for bp in &self.breakpoints {
            if bp.enabled && bp.kind as u8 == KIND {
                // Normalize kseg0/kseg1 to physical so a bp on physical
                // 0x1fbb0010 hits through kseg0 (0x9fbb0010) or kseg1
                // (0xbfbb0010), then word-align. Never mask below bit 2.
                if bp_match_key(bp.addr) == bp_match_key(addr) {
                    // Check optional register condition
                    if let Some(expr) = &bp.condition {
                        let symbols = self.symbols.lock();
                        match expr.eval(&self.core, Some(&symbols)) {
                            Ok(val) => if val == 0 { continue; },
                            // If evaluation fails, we assume the condition is not met (or maybe we should break to show error?)
                            Err(_) => continue, 
                        }
                    }
                    if KIND != BpType::Pc as u8 {
                        eprintln!("[bp] mem bp {} hit: kind={:?} bp_addr={:#010x} access_addr={:#010x} pc={:#018x}",
                            bp.id, bp.kind, bp.addr, addr, self.core.pc);
                    }
                    self.last_bp_hit = Some(bp.id);
                    hit = true;
                    if bp.id == 0 {
                        // Always prioritize reporting BP 0 if hit
                        return true;
                    }
                }
            }
        }
        hit
    }

    pub fn set_temp_breakpoint(&mut self, addr: u64) {
        // Breakpoint 0 is reserved for temp/run-until
        if let Some(bp) = self.breakpoints.get_mut(0) {
            if !bp.enabled {
                self.pc_bp_count += 1;
            }
            bp.addr = addr;
            bp.kind = BpType::Pc;
            bp.enabled = true;
            bp.condition = None;
        }
    }

    pub fn clear_temp_breakpoint(&mut self) {
        if let Some(bp) = self.breakpoints.get_mut(0) {
            if bp.enabled {
                self.pc_bp_count -= 1;
                bp.enabled = false;
            }
        }
    }

    /// Test-only: the last `n` traceback entries as `(pc, raw, jit)` tuples
    /// (oldest-to-newest), so cross-module tests can assert on the traceback
    /// (`TracebackEntry`/the `traceback` field are private). `jit` is true for
    /// any compiled-code origin (`InstrOrigin::Interp`/`FallbackWord` are the
    /// only two that map to `false` — a fallback still runs the interpreter
    /// for real, so it's grouped with `Interp` here for this coarse view).
    #[cfg(test)]
    pub fn test_traceback_last(&self, n: usize) -> Vec<(u64, u32, bool)> {
        self.traceback.get_last(n).into_iter()
            .map(|e| (e.pc, e.instr, !matches!(e.origin, InstrOrigin::Interp | InstrOrigin::FallbackWord)))
            .collect()
    }

    /// Same as `test_traceback_last` but with the full origin tag, for tests
    /// that need to distinguish entry/slot/fallback-successor/back-edge
    /// arrivals rather than just "was this JIT or not".
    #[cfg(test)]
    pub(crate) fn test_traceback_last_origin(&self, n: usize) -> Vec<(u64, u32, InstrOrigin)> {
        self.traceback.get_last(n).into_iter().map(|e| (e.pc, e.instr, e.origin)).collect()
    }

    pub fn add_breakpoint(&mut self, id: usize, addr: u64, kind: BpType) {
        // Remove existing breakpoint with same ID if any
        self.remove_breakpoint(id);

        self.breakpoints.push(Breakpoint { id, addr, kind, enabled: true, condition: None });
        if kind == BpType::Pc {
            self.pc_bp_count += 1;
        } else {
            self.mem_bp_count += 1;
        }
    }

    pub fn remove_breakpoint(&mut self, id: usize) -> bool {
        if let Some(idx) = self.breakpoints.iter().position(|bp| bp.id == id) {
            let bp = self.breakpoints.remove(idx);
            if bp.enabled {
                if bp.kind == BpType::Pc {
                    self.pc_bp_count -= 1;
                } else {
                    self.mem_bp_count -= 1;
                }
            }
            true
        } else {
            false
        }
    }

    pub fn set_breakpoint_enabled(&mut self, id: usize, enabled: bool) -> bool {
        if let Some(bp) = self.breakpoints.iter_mut().find(|bp| bp.id == id) {
            if bp.enabled != enabled {
                bp.enabled = enabled;
                let count = if bp.kind == BpType::Pc {
                    &mut self.pc_bp_count
                } else {
                    &mut self.mem_bp_count
                };
                if enabled { *count += 1; } else { *count -= 1; }
            }
            return true;
        }
        false
    }

    // ========== Instruction Execution Methods ==========

    /// Handle reserved instruction exception with logging
    fn reserved_instruction(&self, d: &DecodedInstr) -> ExecStatus {
        let symbols = self.symbols.lock();
        let sym_str = format_pc_symbol(self.core.pc, &symbols);
        dlog_dev!(LogModule::Mips, "Reserved instruction at {:016x}{}: {:08x} {}", self.core.pc, sym_str, d.raw, mips_dis::disassemble(d.raw, self.core.pc, Some(&symbols)));
        #[cfg(feature = "instr_stats")]
        eprintln!("[instr_stats] Reserved/illegal instruction at {:016x}{}: {:08x} {}",
            self.core.pc, sym_str, d.raw, mips_dis::disassemble(d.raw, self.core.pc, Some(&symbols)));
        exec_exception(EXC_RI)
    }

    fn exec_reserved(&mut self, d: &DecodedInstr) -> ExecStatus {
        let s = self.reserved_instruction(d);
        self.handle_exception(s)
    }

    /// No-op handler — used as the default for zero-initialised DecodedInstr.
    fn exec_nop(&mut self, _d: &DecodedInstr) -> ExecStatus {
        self.handle_exec_complete()
    }

    /// Handle an exception: update CP0 registers and jump to handler vector.
    /// Takes an ExecStatus with EXEC_IS_EXCEPTION set; extracts code and TLB-refill flag.
    fn handle_exception(&mut self, status: ExecStatus) -> ExecStatus {
        // In developer builds, bus/address error exceptions break into the
        // monitor at the fault site rather than dispatching to the MIPS
        // vector — must be decided before deliver_exception runs, since that
        // overwrites core.pc with the vector address. EPC's predicted value
        // (what deliver_exception is about to write) is computed the same
        // way deliver_exception itself computes it, just far enough ahead to
        // decide whether to bail on vectoring at all.
        #[cfg(feature = "developerx")]
        {
            let was_exl = (self.core.cp0_status & STATUS_EXL) != 0;
            let epc = if was_exl {
                self.core.cp0_epc // preserved from the first exception
            } else if self.core.in_delay_slot {
                self.core.pc.wrapping_sub(4)
            } else {
                self.core.pc
            };
            let exc_code = (status & CAUSE_EXCCODE_MASK) >> 2;
            if exc_code == EXC_IBE || exc_code == EXC_DBE {
                eprintln!("BUS ERROR ({}) at PC={:#010x} EPC={:#010x}",
                    if exc_code == EXC_IBE { "IBE" } else { "DBE" },
                    self.core.pc, epc);
                return EXEC_BREAKPOINT;
            }
            if exc_code == EXC_ADEL || exc_code == EXC_ADES {
                eprintln!("ADDRESS ERROR ({}) at PC={:#010x} EPC={:#010x} BadVAddr={:#010x}",
                    if exc_code == EXC_ADEL { "ADEL" } else { "ADES" },
                    self.core.pc, epc, self.core.cp0_badvaddr);
                return EXEC_BREAKPOINT;
            }
            if (exc_code == EXC_TLBL || exc_code == EXC_TLBS) && (self.core.cp0_badvaddr as u32 == 0xFF800000) {
                eprintln!("ADDRESS ERROR ({}) at PC={:#010x} EPC={:#010x} BadVAddr={:#010x}",
                    if exc_code == EXC_TLBL { "TLBL" } else { "TLBS" },
                    self.core.pc, epc, self.core.cp0_badvaddr);
                return EXEC_BREAKPOINT;
            }
        }

        // Clear LLBit on any exception
        self.cache.set_llbit(false);

        // Architectural effect (Cause/EPC/Status/vector) — the portable part
        // shared with jitv2_verify (§4.2 single-implementation delivery).
        crate::mips_core::deliver_exception(&mut self.core, status);

        self.nanotlb_invalidate();
        // Reset delay slot state as we are jumping to a new context
        self.core.in_delay_slot = false;
        status
    }

    // Helper to get CPU's current addressing mode
    #[inline]
    fn is_64bit(&self) -> bool {
        self.core.is_64bit_mode()
    }

    /// Returns whether a virtual address would use the XTLB (64-bit) vector on a TLB miss.
    /// Mirrors the xtlb flag logic in translate_32/64bit_impl exactly.
    ///   - 32-bit mode: always false (all TLB segments use UTLB vector)
    ///   - 64-bit mode: true for xuseg (top=0), xsseg (top=1), and true 64-bit xkseg (top=3,
    ///     not in 32-bit compat range 0xFFFFFFFF_xxxxxxxx); false for 32-bit compat xkseg
    #[inline]
    fn is_xtlb_address(&self, virt_addr: u64) -> bool {
        if !self.core.is_64bit_mode() {
            return false;
        }
        match virt_addr >> 62 {
            0 | 1 => true,
            3 => (virt_addr >> 32) != 0xFFFFFFFF,
            _ => false, // xkphys (top=2) is unmapped, never TLB; shouldn't be called for non-TLB addrs
        }
    }

    /// Core translation logic.  When `DEBUG` is true the function:
    /// - always treats the access as kernel-privileged, and
    /// - never writes any CP0 side-effect registers (BadvAddr, EntryHi, Context, XContext).
    #[inline]
    fn translate_impl<const DEBUG: bool>(&mut self, virt_addr: u64, access_type: AccessType) -> TranslateResult {
        use crate::mips_core::{PrivilegeMode, PRIV_KERNEL, PRIV_SUPERVISOR, PRIV_USER};

        let is_64bit = self.is_64bit();
        let privilege = if DEBUG {
            PrivilegeMode::Kernel
        } else {
            self.core.get_privilege_mode()
        };

        if is_64bit {
            match privilege {
                PrivilegeMode::Kernel     => self.translate_64bit_impl::<DEBUG, PRIV_KERNEL>(virt_addr, access_type),
                PrivilegeMode::Supervisor => self.translate_64bit_impl::<DEBUG, PRIV_SUPERVISOR>(virt_addr, access_type),
                PrivilegeMode::User       => self.translate_64bit_impl::<DEBUG, PRIV_USER>(virt_addr, access_type),
            }
        } else {
            match privilege {
                PrivilegeMode::Kernel     => self.translate_32bit_impl::<DEBUG, PRIV_KERNEL>(virt_addr, access_type),
                PrivilegeMode::Supervisor => self.translate_32bit_impl::<DEBUG, PRIV_SUPERVISOR>(virt_addr, access_type),
                PrivilegeMode::User       => self.translate_32bit_impl::<DEBUG, PRIV_USER>(virt_addr, access_type),
            }
        }
    }

    /// Translate 32-bit virtual address.
    /// `PRIV` is a const-generic privilege level — one of `PRIV_KERNEL`, `PRIV_SUPERVISOR`,
    /// or `PRIV_USER` from `mips_core`.  Using a const generic lets the compiler eliminate
    /// dead branches statically rather than relying on runtime dispatch.
    #[inline]
    fn translate_32bit_impl<const DEBUG: bool, const PRIV: u8>(&mut self, virt_addr: u64, access_type: AccessType) -> TranslateResult {
        use crate::mips_core::{PRIV_KERNEL, PRIV_SUPERVISOR};

        // Upper 32 bits are ignored in 32-bit mode; only low 32 bits used for segment decode.
        let virt_addr32 = virt_addr as u32;

        // Extract top 3 bits to determine segment
        let segment = (virt_addr32 >> 29) as u64;

        let addr_exc = |wr: bool| exec_exception(if wr { EXC_ADES } else { EXC_ADEL });

        match segment {
            // KUSEG: 0x00000000 - 0x7FFFFFFF (user segment, TLB mapped)
            0..=3 => {
                // When ERL=1, KUSEG becomes unmapped, uncached identity mapping
                if (self.core.cp0_status & crate::mips_core::STATUS_ERL) != 0 {
                    return TranslateResult::ok(virt_addr32 as u64, TR_UNCACHED);
                }
                // 32-bit mode: XTLB=0 → UTLB vector on miss
                self.tlb_translate_impl::<DEBUG, 0>(virt_addr, access_type)
            }

            // KSEG0: 0x80000000 - 0x9FFFFFFF (kernel unmapped, cached)
            4 => {
                if PRIV == PRIV_KERNEL {
                    TranslateResult::ok((virt_addr32 & 0x1FFFFFFF) as u64, TR_CACHEABLE)
                } else {
                    if !DEBUG { self.core.cp0_badvaddr = virt_addr; }
                    TranslateResult::exc(addr_exc(access_type == AccessType::Write))
                }
            }

            // KSEG1: 0xA0000000 - 0xBFFFFFFF (kernel unmapped, uncached)
            5 => {
                if PRIV == PRIV_KERNEL {
                    TranslateResult::ok((virt_addr32 & 0x1FFFFFFF) as u64, TR_UNCACHED)
                } else {
                    if !DEBUG { self.core.cp0_badvaddr = virt_addr; }
                    TranslateResult::exc(addr_exc(access_type == AccessType::Write))
                }
            }

            // KSSEG: 0xC0000000 - 0xDFFFFFFF (supervisor segment, TLB mapped)
            6 => {
                if PRIV == PRIV_KERNEL || PRIV == PRIV_SUPERVISOR {
                    self.tlb_translate_impl::<DEBUG, 0>(virt_addr, access_type)
                } else {
                    if !DEBUG { self.core.cp0_badvaddr = virt_addr; }
                    TranslateResult::exc(addr_exc(access_type == AccessType::Write))
                }
            }

            // KSEG3: 0xE0000000 - 0xFFFFFFFF (kernel segment, TLB mapped)
            7 => {
                if PRIV == PRIV_KERNEL {
                    self.tlb_translate_impl::<DEBUG, 0>(virt_addr, access_type)
                } else {
                    if !DEBUG { self.core.cp0_badvaddr = virt_addr; }
                    TranslateResult::exc(addr_exc(access_type == AccessType::Write))
                }
            }

            _ => unreachable!(),
        }
    }

    /// Translate 64-bit virtual address.
    /// `PRIV` is a const-generic privilege level — one of `PRIV_KERNEL`, `PRIV_SUPERVISOR`,
    /// or `PRIV_USER` from `mips_core`.
    #[inline]
    fn translate_64bit_impl<const DEBUG: bool, const PRIV: u8>(&mut self, virt_addr: u64, access_type: AccessType) -> TranslateResult {
        use crate::mips_core::{PRIV_KERNEL, PRIV_SUPERVISOR};

        // Check address region based on top bits
        let top_bits = virt_addr >> 62;

        match top_bits {
            // xuseg: 0x0000_0000_0000_0000 - 0x0000_00FF_FFFF_FFFF (user mapped)
            // Accessible from all privilege levels
            0 => {
                if (virt_addr >> 40) != 0 {
                    // Bits 63:40 must be zero for valid user address
                    if !DEBUG { self.core.cp0_badvaddr = virt_addr; }
                    return TranslateResult::exc(exec_exception(if access_type == AccessType::Write { EXC_ADES } else { EXC_ADEL }));
                }

                // When ERL=1, xuseg becomes unmapped, uncached identity mapping
                if (self.core.cp0_status & crate::mips_core::STATUS_ERL) != 0 {
                    return TranslateResult::ok(virt_addr, TR_UNCACHED);
                }

                // xuseg: true 64-bit address → xtlb=true
                self.tlb_translate_impl::<DEBUG, 1>(virt_addr, access_type)
            }

            // xsseg: 0x4000_0000_0000_0000 - 0x7FFF_FFFF_FFFF_FFFF (supervisor segment)
            1 => {
                if PRIV == PRIV_KERNEL || PRIV == PRIV_SUPERVISOR {
                    self.tlb_translate_impl::<DEBUG, 1>(virt_addr, access_type)
                } else {
                    if !DEBUG { self.core.cp0_badvaddr = virt_addr; }
                    TranslateResult::exc(exec_exception(if access_type == AccessType::Write { EXC_ADES } else { EXC_ADEL }))
                }
            }

            // xkphys: 0x8000_0000_0000_0000 - 0xBFFF_FFFF_FFFF_FFFF (unmapped physical)
            2 => {
                let bits_61_59 = (virt_addr >> 59) & 0x7;
                if bits_61_59 >= 2 && bits_61_59 <= 7 {
                    if PRIV == PRIV_KERNEL {
                        let phys_addr = virt_addr & 0x07FF_FFFF_FFFF_FFFF;
                        // bits_61_59 is the C field directly: 2=Uncached, 3=Cacheable, 5=CacheableCoherent
                        let c = match bits_61_59 { 3 | 5 => bits_61_59 as u32, _ => TR_UNCACHED };
                        TranslateResult::ok(phys_addr, c)
                    } else {
                        if !DEBUG { self.core.cp0_badvaddr = virt_addr; }
                        TranslateResult::exc(exec_exception(if access_type == AccessType::Write { EXC_ADES } else { EXC_ADEL }))
                    }
                } else {
                    if !DEBUG { self.core.cp0_badvaddr = virt_addr; }
                    TranslateResult::exc(exec_exception(if access_type == AccessType::Write { EXC_ADES } else { EXC_ADEL }))
                }
            }

            // xkseg: 0xC000_0000_0000_0000 - 0xFFFF_FFFF_FFFF_FFFF (kernel segment)
            3 => {
                if PRIV == PRIV_KERNEL {
                    let addr_32 = virt_addr as u32;
                    // Compatibility segments: top 32 bits all 1s → 32-bit compat (xtlb=false)
                    if (virt_addr >> 32) == 0xFFFFFFFF {
                        match (addr_32 >> 29) & 0x7 {
                            4 => return TranslateResult::ok((addr_32 & 0x1FFFFFFF) as u64, TR_CACHEABLE),
                            5 => return TranslateResult::ok((addr_32 & 0x1FFFFFFF) as u64, TR_UNCACHED),
                            // KSSEG/KSEG3 compat: TLB mapped, 32-bit compat → xtlb=false
                            _ => return self.tlb_translate_impl::<DEBUG, 0>(virt_addr, access_type),
                        }
                    }
                    // True 64-bit xkseg: xtlb=true
                    self.tlb_translate_impl::<DEBUG, 1>(virt_addr, access_type)
                } else {
                    if !DEBUG { self.core.cp0_badvaddr = virt_addr; }
                    TranslateResult::exc(exec_exception(if access_type == AccessType::Write { EXC_ADES } else { EXC_ADEL }))
                }
            }

            _ => unreachable!(),
        }
    }

    /// TLB translation.  When `DEBUG` is true, CP0 side-effect registers are
    /// not written on miss/invalid/modified — the exception result is still
    /// returned so the caller knows the translation failed.
    #[inline]
    fn tlb_translate_impl<const DEBUG: bool, const XTLB: u8>(&mut self, virt_addr: u64, access_type: AccessType) -> TranslateResult {
        use crate::mips_tlb::TlbResult;

        // Get current ASID from EntryHi register
        let asid = (self.core.cp0_entryhi & 0xFF) as u8;

        // Query the TLB — XTLB=1 uses 64-bit VPN comparison mask
        let result = self.tlb.translate::<XTLB>(virt_addr, asid, access_type);

        let tlb_miss_code = if access_type == AccessType::Write { EXC_TLBS } else { EXC_TLBL };

        match result {
            TlbResult::Hit { phys_addr, cache_attr, dirty } => {
                if access_type == AccessType::Write && !dirty {
                    if !DEBUG { self.update_tlb_exception_registers::<XTLB>(virt_addr); }
                    TranslateResult::exc(exec_exception(EXC_MOD))
                } else {
                    TranslateResult::ok(phys_addr, cache_attr as u32)
                }
            }
            TlbResult::Miss { .. } => {
                if !DEBUG { self.update_tlb_exception_registers::<XTLB>(virt_addr); }
                // Miss: XTLB vector (0x080) for 64-bit extended addresses, UTLB vector (0x000) otherwise
                TranslateResult::exc(if XTLB != 0 { exec_xtlb_miss(tlb_miss_code) } else { exec_tlb_miss(tlb_miss_code) })
            }
            TlbResult::Invalid { .. } => {
                if !DEBUG { self.update_tlb_exception_registers::<XTLB>(virt_addr); }
                // Invalid: always general vector (0x180)
                TranslateResult::exc(exec_exception(tlb_miss_code))
            }
            TlbResult::Modified { .. } => {
                if !DEBUG { self.update_tlb_exception_registers::<XTLB>(virt_addr); }
                TranslateResult::exc(exec_exception(EXC_MOD))
            }
        }
    }

    /// Debug helper to translate address without side effects
    pub fn debug_translate(&mut self, virt_addr: u64) -> TranslateResult {
        self.translate_impl::<true>(virt_addr, AccessType::Debug)
    }

    /// Update CP0 BadVAddr, EntryHi, Context, XContext for any TLB exception.
    /// `XTLB=1` for extended (64-bit) translations — uses 64-bit VPN mask for EntryHi.
    #[inline]
    fn update_tlb_exception_registers<const XTLB: u8>(&mut self, virt_addr: u64) {
        const EH_VPN2_32: u64 = 0x0000_0000_FFFF_E000;
        const EH_VPN2_64: u64 = 0x0000_00FF_FFFF_E000;
        const EH_REGION:  u64 = 0xC000_0000_0000_0000;

        self.core.cp0_badvaddr = virt_addr;

        // EntryHi: VPN from address masked per translation mode, ASID preserved.
        let asid = self.core.cp0_entryhi & 0xFF;
        let vpn_mask = if XTLB != 0 { EH_REGION | EH_VPN2_64 } else { EH_VPN2_32 };
        self.core.cp0_entryhi = (virt_addr & vpn_mask) | asid;

        // Context: PTEBase[63:23] preserved, BadVPN2 = virt_addr[31:13] in bits [22:4].
        // Always 32-bit VPN — Context is used by the 32-bit UTLB handler.
        let ptebase = self.core.cp0_context & 0xFFFFFFFF_FF800000;
        let badvpn2 = ((virt_addr & EH_VPN2_32) >> 13) << 4;
        self.core.cp0_context = ptebase | badvpn2;

        // XContext: PTEBase[63:33] preserved, Region[63:62] → bits[32:31], BadVPN2[39:13] → bits[30:4].
        let xptebase = self.core.cp0_xcontext & 0xFFFF_FFFE_0000_0000;
        let xbadvpn2 = ((virt_addr & EH_VPN2_64) >> 13) << 4;
        let region = (virt_addr >> 62) & 0x3;
        self.core.cp0_xcontext = xptebase | (region << 31) | xbadvpn2;
    }

    // ========== Memory Access Wrapper Methods ==========

    /// Fetch instruction: translates virtual address and reads from I-cache
    /// Fetch and decode the instruction at virt_addr.
    /// Returns a pointer to the DecodedInstr (in cache or self.ins scratch) on success,
    /// Fetch instruction, returning `FetchInstrResult`.
    /// `status == EXEC_COMPLETE` means hit; `instr` is valid. Any other status is an error.
    fn fetch_instr(&mut self, virt_addr: u64) -> FetchInstrResult {
        self.fetch_instr_impl::<false>(virt_addr)
    }

    /// Debug instruction fetch: kernel-mode override, no breakpoints, no CP0 side-effects.
    /// Returns the raw instruction word only (no decode).
    pub fn debug_fetch_instr(&mut self, virt_addr: u64) -> Result<u32, ExecStatus> {
        let r = self.fetch_instr_impl::<true>(virt_addr);
        if r.status == EXEC_COMPLETE {
            Ok(unsafe { (*r.instr).raw })
        } else {
            Err(r.status)
        }
    }

    /// Core instruction fetch.  When `DEBUG=true`:
    /// - Privilege is treated as Kernel (via translate_impl)
    /// - Breakpoint checks are skipped
    /// - cp0_badvaddr is never written
    /// Returns `FetchInstrResult`; `status == EXEC_COMPLETE` means hit, `instr` is valid.
    #[inline]
    fn fetch_instr_impl<const DEBUG: bool>(&mut self, virt_addr: u64) -> FetchInstrResult {
        #[cfg(not(feature = "lightning"))]
        if !DEBUG && self.bp_enabled() && self.check_breakpoint::<{ BpType::VirtFetch as u8 }>(virt_addr) {
            return FetchInstrResult::exception(EXEC_BREAKPOINT);
        }

        let translate_result = if DEBUG {
            self.translate_impl::<true>(virt_addr, AccessType::Fetch)
        } else {
            self.nanotlb_translate::<{AccessType::Fetch as u8}>(virt_addr)
        };
        if translate_result.is_exception() { return FetchInstrResult::exception(translate_result.status); }
        let phys_addr = translate_result.phys;
        if translate_result.is_cached() {
            #[cfg(not(feature = "lightning"))]
            if !DEBUG && self.bp_enabled() && self.check_breakpoint::<{ BpType::PhysFetch as u8 }>(phys_addr as u64) {
                return FetchInstrResult::exception(EXEC_BREAKPOINT);
            }

            let r = self.cache.fetch(virt_addr, phys_addr as u64);
            if !DEBUG && r.status == exec_exception(EXC_VCEI) {
                self.core.cp0_badvaddr = virt_addr;
            }
            r
        } else {
            #[cfg(not(feature = "lightning"))]
            if !DEBUG && self.bp_enabled() && self.check_breakpoint::<{ BpType::PhysFetch as u8 }>(phys_addr as u64) {
                return FetchInstrResult::exception(EXEC_BREAKPOINT);
            }

            #[cfg(feature = "developer")]
            self.uncached_fetch_count.fetch_add(1, Ordering::Relaxed);
            let r = self.sysad.read32(phys_addr);
            if r.is_ok() {
                self.ins.flags = FLAG_NOT_DECODED;
                self.ins.raw = r.data;
                FetchInstrResult::hit(&self.ins as *const DecodedInstr)
            } else {
                // BUS_BUSY == EXEC_RETRY (compile-time asserted in traits.rs); pass status through.
                if r.status != BUS_BUSY {
                    eprintln!("Bus error on instruction fetch: PC={:016x} PA={:08x} status={:08x}", virt_addr, phys_addr, r.status);
                }
                FetchInstrResult::exception(r.status)
            }
        }
    }

    /// Production data read (with breakpoints, updates CP0 state on exceptions).
    #[inline]
    pub(crate) fn read_data<const SIZE: usize>(&mut self, virt_addr: u64) -> Result<u64, ExecStatus> {
        self.read_data_impl::<false, SIZE>(virt_addr)
    }

    /// Debug data read: kernel-mode override, no breakpoints, no CP0 side-effects.
    pub fn debug_read(&mut self, virt_addr: u64, size: usize) -> Result<u64, ExecStatus> {
        match size {
            1 => self.read_data_impl::<true, 1>(virt_addr),
            2 => self.read_data_impl::<true, 2>(virt_addr),
            4 => self.read_data_impl::<true, 4>(virt_addr),
            8 => self.read_data_impl::<true, 8>(virt_addr),
            _ => Err(exec_exception(EXC_ADEL)),
        }
    }

    /// Core data read.  When `DEBUG=true`:
    /// - Privilege is treated as Kernel (via translate_impl)
    /// - Breakpoint checks are skipped
    /// - cp0_badvaddr is never written
    #[inline]
    fn read_data_impl<const DEBUG: bool, const SIZE: usize>(&mut self, virt_addr: u64) -> Result<u64, ExecStatus> {
        const { assert!(SIZE == 1 || SIZE == 2 || SIZE == 4 || SIZE == 8, "invalid memory access SIZE") };
        #[cfg(not(feature = "lightning"))]
        if !DEBUG && self.bp_enabled() && self.check_breakpoint::<{ BpType::VirtRead as u8 }>(virt_addr) {
            return Err(EXEC_BREAKPOINT);
        }

        // Check alignment
        if (virt_addr & align_mask_for::<SIZE>()) != 0 {
            if !DEBUG { self.core.cp0_badvaddr = virt_addr; }
            return Err(exec_exception(EXC_ADEL));
        }

        let translate_result = if DEBUG {
            self.translate_impl::<true>(virt_addr, AccessType::Debug)
        } else {
            self.nanotlb_translate::<{AccessType::Read as u8}>(virt_addr)
        };
        if translate_result.is_exception() { return Err(translate_result.status); }
        let result = {
            let phys_addr = translate_result.phys as u64;
            let is_cached = translate_result.is_cached();

            if is_cached {
                    // Cached access uses D-Cache
                    #[cfg(not(feature = "lightning"))]
                    if !DEBUG && self.bp_enabled() && self.check_breakpoint::<{ BpType::PhysRead as u8 }>(phys_addr) {
                        return Err(EXEC_BREAKPOINT);
                    }

                    let r = self.cache.read::<SIZE>(virt_addr, phys_addr);
                    if r.is_ok() {
                        Ok(r.data)
                    } else {
                        if !DEBUG { self.core.cp0_badvaddr = virt_addr; }
                        Err(r.status) // BUS_BUSY, BUS_VCE, or BUS_ERR — all valid ExecStatus
                    }
                } else {
                    // Uncached access
                    #[cfg(not(feature = "lightning"))]
                    if !DEBUG && self.bp_enabled() && self.check_breakpoint::<{ BpType::PhysRead as u8 }>(phys_addr) {
                        return Err(EXEC_BREAKPOINT);
                    }

                    // jitcheck's hardware-read fixup replay: substitute the
                    // reference pass's recorded value for a known-volatile
                    // register (see HW_READ_FIXUP_ADDRS/hw_read_fixup_replay's
                    // doc comment) instead of issuing the real, differently-
                    // timed bus read. Only engages for addresses actually in
                    // the fixup list — every other uncached read is
                    // unaffected. `size` must also match: a stale entry from
                    // a different-sized access at the same address is not a
                    // safe substitute.
                    #[cfg(feature = "developer")]
                    if let Some(fixups) = &self.hw_read_fixup_replay {
                        if let Some(&(_, _, value)) = fixups.iter().find(|&&(addr, size, _)| addr == phys_addr && size == SIZE as u8) {
                            return Ok(value);
                        }
                    }

                    let res = {
                        let r = if SIZE == 1 {
                            let r = self.sysad.read8(phys_addr as u32);
                            BusRead64 { status: r.status, data: r.data as u64 }
                        } else if SIZE == 2 {
                            let r = self.sysad.read16(phys_addr as u32);
                            BusRead64 { status: r.status, data: r.data as u64 }
                        } else if SIZE == 4 {
                            let r = self.sysad.read32(phys_addr as u32);
                            BusRead64 { status: r.status, data: r.data as u64 }
                        } else {
                            self.sysad.read64(phys_addr as u32)
                        };
                        if r.is_ok() {
                            Ok(r.data)
                        } else {
                            if r.status != BUS_BUSY {
                                if !DEBUG {
                                    // jitcheck tag: which pass this read happened in, so a
                                    // divergence investigation can tell at a glance whether
                                    // the reference (interpreter-only) pass ever reached
                                    // this same faulting read at all, or only the replay
                                    // (real-JIT-dispatch) pass did — see hw_read_fixup_recording/
                                    // hw_read_fixup_replay's doc comments for what arms each.
                                    #[cfg(feature = "developer")]
                                    let jitcheck_pass = if self.hw_read_fixup_recording {
                                        " [jitcheck:reference]"
                                    } else if self.hw_read_fixup_replay.is_some() {
                                        " [jitcheck:replay]"
                                    } else {
                                        ""
                                    };
                                    #[cfg(not(feature = "developer"))]
                                    let jitcheck_pass = "";
                                    eprintln!("Bus error on uncached read{}: PC={:016x} VA={:016x} PA={:016x} status={:08x}{}", SIZE*8, self.core.pc, virt_addr, phys_addr, r.status, jitcheck_pass);
                                    self.core.cp0_badvaddr = virt_addr;
                                }
                            }
                            Err(r.status) // BUS_BUSY or BUS_ERR — both valid ExecStatus
                        }
                    };

                    if !DEBUG && mips_log(MIPS_LOG_MEM) {
                        match res {
                            Ok(val) => dlog_dev!(LogModule::Mips, "Uncached Read{}: PC={:016x} VA={:016x} PA={:016x} Val={:016x}", SIZE*8, self.core.pc, virt_addr, phys_addr, val),
                            Err(_) => dlog_dev!(LogModule::Mips, "Uncached Read{}: PC={:016x} VA={:016x} PA={:016x} Error", SIZE*8, self.core.pc, virt_addr, phys_addr),
                        }
                    }

                    // jitcheck's hardware-read fixup record: capture the
                    // reference pass's real value for a known-volatile
                    // register so the replay pass (above) can substitute it
                    // back in later. Only for a successful real access to an
                    // address in the fixup list.
                    #[cfg(feature = "developer")]
                    if self.hw_read_fixup_recording {
                        if let Ok(val) = res {
                            if HW_READ_FIXUP_ADDRS.contains(&phys_addr) {
                                self.hw_read_fixup_recorded.push((phys_addr, SIZE as u8, val));
                            }
                        }
                    }

                    res
                }
        };
        // jitv2_lockstep's load/store verification (see MipsCore's
        // lockstep_mem field doc comment): record what this real access
        // actually did — address, translated physical address, and value —
        // so lockstep_check_load_store can compare the JIT's independently-
        // computed address/value against it without ever issuing a second
        // real bus access (unsafe for a load: could be MMIO with side
        // effects). `Some` only on success: a fault/retry leaves nothing
        // valid to compare (see lockstep_mem's doc comment on why this must
        // be `None`, not stale `Some` data, on that path).
        #[cfg(feature = "jitv2_lockstep")]
        {
            self.core.lockstep_mem = result.ok().map(|val| crate::mips_core::LockstepMemCapture {
                addr: virt_addr,
                phys: translate_result.phys as u64,
                value: val,
            });
        }
        result
    }

    /// Production data write (with breakpoints, undo tracking, updates CP0 state on exceptions).
    #[inline]
    pub(crate) fn write_data<const SIZE: usize>(&mut self, virt_addr: u64, val: u64) -> ExecStatus {
        self.write_data_impl::<false, SIZE>(virt_addr, val)
    }

    /// Partial masked doubleword write for SDL/SDR/SWL/SWR.
    #[inline]
    pub(crate) fn write_data64_masked(&mut self, virt_addr: u64, val: u64, mask: u64) -> ExecStatus {
        self.write_data64_masked_impl::<false>(virt_addr, val, mask)
    }

    /// Debug data write: kernel-mode override, no breakpoints, no undo tracking, no CP0 side-effects.
    pub fn debug_write(&mut self, virt_addr: u64, val: u64, size: usize, mask: u64) -> ExecStatus {
        match size {
            1 => self.write_data_impl::<true, 1>(virt_addr, val),
            2 => self.write_data_impl::<true, 2>(virt_addr, val),
            4 => self.write_data_impl::<true, 4>(virt_addr, val),
            8 => self.write_data_impl::<true, 8>(virt_addr, val),
            _ => exec_exception(EXC_ADES),
        }
    }

    /// Core data write.  When `DEBUG=true`:
    /// - Privilege is treated as Kernel (via translate_impl)
    /// - Breakpoint checks are skipped
    /// - cp0_badvaddr is never written
    /// - Undo buffer tracking is skipped
    #[inline]
    fn write_data_impl<const DEBUG: bool, const SIZE: usize>(&mut self, virt_addr: u64, val: u64) -> ExecStatus {
        const { assert!(SIZE == 1 || SIZE == 2 || SIZE == 4 || SIZE == 8, "invalid memory access SIZE") };
        #[cfg(not(feature = "lightning"))]
        if !DEBUG && self.bp_enabled() && self.check_breakpoint::<{ BpType::VirtWrite as u8 }>(virt_addr) {
            return EXEC_BREAKPOINT;
        }

        // Check alignment
        if (virt_addr & align_mask_for::<SIZE>()) != 0 {
            if !DEBUG { self.core.cp0_badvaddr = virt_addr; }
            return exec_exception(EXC_ADES);
        }

        let translate_result = if DEBUG {
            self.translate_impl::<true>(virt_addr, AccessType::Debug)
        } else {
            self.nanotlb_translate::<{AccessType::Write as u8}>(virt_addr)
        };
        if translate_result.is_exception() { return translate_result.status; }
        let phys_addr = translate_result.phys as u64;
        let is_cached = translate_result.is_cached();

        // Track memory write for undo if it's to lomem/himem (production only)
        #[cfg(feature = "developer")]
        if !DEBUG && self.undo_buffer.is_enabled() {
            let phys_addr_32 = phys_addr as u32;
            let is_main_memory = (phys_addr_32 >= LOMEM_BASE && phys_addr_32 < LOMEM_END) ||
                                 (phys_addr_32 >= HIMEM_BASE && phys_addr_32 < HIMEM_END);
            if is_main_memory {
                let old_value = match self.read_data::<SIZE>(virt_addr) {
                    Ok(v) => v,
                    Err(_) => 0,
                };
                self.track_memory_write(virt_addr, phys_addr, old_value, SIZE);
            }
        }

        #[cfg(not(feature = "lightning"))]
        if !DEBUG && self.bp_enabled() && self.check_breakpoint::<{ BpType::PhysWrite as u8 }>(phys_addr) {
            return EXEC_BREAKPOINT;
        }

        let status = if is_cached {
            let status = self.cache.write::<SIZE>(virt_addr, phys_addr, val);
            if status != BUS_OK && status != BUS_BUSY {
                if !DEBUG { self.core.cp0_badvaddr = virt_addr; }
            }
            status
        } else {
            if !DEBUG && mips_log(MIPS_LOG_MEM) {
                dlog_dev!(LogModule::Mips, "Uncached Write{}: PC={:016x} VA={:016x} PA={:016x} Val={:016x}", SIZE*8, self.core.pc, virt_addr, phys_addr, val);
            }
            let ws = if SIZE == 1 {
                self.sysad.write8(phys_addr as u32, val as u8)
            } else if SIZE == 2 {
                self.sysad.write16(phys_addr as u32, val as u16)
            } else if SIZE == 4 {
                self.sysad.write32(phys_addr as u32, val as u32)
            } else {
                self.sysad.write64(phys_addr as u32, val)
            };
            if ws != BUS_OK && ws != BUS_BUSY && !DEBUG {
                eprintln!("Bus error on uncached write{}: PC={:016x} VA={:016x} PA={:016x} val={:016x} status={:08x}", SIZE*8, self.core.pc, virt_addr, phys_addr, val, ws);
            }
            ws
        };
        // jitv2_lockstep's load/store verification (see MipsCore's
        // lockstep_mem field doc comment, and read_data_impl's matching
        // capture) — `Some` only when the write actually completed
        // (BUS_OK/EXEC_COMPLETE); a retry/VCE/bus error leaves nothing valid
        // to compare (see lockstep_mem's own doc comment on why this must be
        // `None`, not stale `Some` data, on that path).
        #[cfg(feature = "jitv2_lockstep")]
        {
            self.core.lockstep_mem = (status == BUS_OK).then_some(crate::mips_core::LockstepMemCapture {
                addr: virt_addr,
                phys: phys_addr,
                value: val,
            });
        }
        status
    }

    /// Partial masked doubleword store: SDL/SDR/SWL/SWR.
    /// `virt_addr` is 8-byte aligned; val/mask are in MIPS big-endian doubleword space.
    fn write_data64_masked_impl<const DEBUG: bool>(&mut self, virt_addr: u64, val: u64, mask: u64) -> ExecStatus {
        #[cfg(not(feature = "lightning"))]
        if !DEBUG && self.bp_enabled() && self.check_breakpoint::<{ BpType::VirtWrite as u8 }>(virt_addr) {
            return EXEC_BREAKPOINT;
        }

        // virt_addr is already doubleword-aligned (callers guarantee this)
        let translate_result = if DEBUG {
            self.translate_impl::<true>(virt_addr, AccessType::Debug)
        } else {
            self.nanotlb_translate::<{AccessType::Write as u8}>(virt_addr)
        };
        if translate_result.is_exception() { return translate_result.status; }
        let phys_addr = translate_result.phys as u64;
        let is_cached = translate_result.is_cached();

        #[cfg(not(feature = "lightning"))]
        if !DEBUG && self.bp_enabled() && self.check_breakpoint::<{ BpType::PhysWrite as u8 }>(phys_addr) {
            return EXEC_BREAKPOINT;
        }

        if is_cached {
            let status = self.cache.write64_masked(virt_addr, phys_addr, val, mask);
            if status != BUS_OK && status != BUS_BUSY {
                if !DEBUG { self.core.cp0_badvaddr = virt_addr; }
            }
            status
        } else {
            if !DEBUG && mips_log(MIPS_LOG_MEM) {
                dlog_dev!(LogModule::Mips, "Uncached Write64Masked: PC={:016x} VA={:016x} PA={:016x} Val={:016x} Mask={:016x}", self.core.pc, virt_addr, phys_addr, val, mask);
            }
            let ws = self.sysad.write64_masked(phys_addr as u32, val, mask);
            if ws != BUS_OK && ws != BUS_BUSY && !DEBUG {
                eprintln!("Bus error on uncached write64_masked: PC={:016x} VA={:016x} PA={:016x} val={:016x} mask={:016x} status={:08x}", self.core.pc, virt_addr, phys_addr, val, mask, ws);
            }
            ws
        }
    }

    // SPECIAL opcode individual methods (generated from exec_special)
    fn exec_sll(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rt_val = self.core.read_gpr(d.rt as u32);
        let rd_reg = d.rd as u32;
        let sa_val = d.sa as u32;
        self.core.write_gpr(rd_reg, (rt_val << sa_val) as u32 as i32 as i64 as u64);
        self.handle_exec_complete()
    }
    fn exec_movci(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_reg = d.rs as u32;
        let rd_reg = d.rd as u32;
        let cc = (d.raw >> 18) & 0x7;
        let tf = ((d.raw >> 16) & 0x1) != 0;
        let cc_value = self.core.get_fpu_cc(cc);
        let taken = cc_value == tf;
        if taken {
            let rs_val = self.core.read_gpr(rs_reg);
            self.core.write_gpr(rd_reg, rs_val);
        }
        if mips_log(MIPS_LOG_FPU) {
            dlog_dev!(LogModule::Mips, "FPU mov{} PC={:016x} cc{}={} taken={}",
                if tf { "t" } else { "f" }, self.core.pc, cc, cc_value, taken);
        }
        self.handle_exec_complete()
    }
    fn exec_srl(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rt_val = self.core.read_gpr(d.rt as u32) as u32;
        let rd_reg = d.rd as u32;
        let sa_val = d.sa as u32;
        self.core.write_gpr(rd_reg, (rt_val >> sa_val) as i32 as i64 as u64);
        self.handle_exec_complete()
    }
    fn exec_sra(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rt_val = self.core.read_gpr(d.rt as u32) as i32;
        let rd_reg = d.rd as u32;
        let sa_val = d.sa as u32;
        self.core.write_gpr(rd_reg, (rt_val >> sa_val) as i64 as u64);
        self.handle_exec_complete()
    }
    fn exec_sllv(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32);
        let rd_reg = d.rd as u32;
        let sa_val = (rs_val & 0x1F) as u32;
        self.core.write_gpr(rd_reg, (rt_val << sa_val) as u32 as i32 as i64 as u64);
        self.handle_exec_complete()
    }
    fn exec_srlv(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32) as u32;
        let rd_reg = d.rd as u32;
        let sa_val = (rs_val & 0x1F) as u32;
        self.core.write_gpr(rd_reg, (rt_val >> sa_val) as i32 as i64 as u64);
        self.handle_exec_complete()
    }
    fn exec_srav(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32) as i32;
        let rd_reg = d.rd as u32;
        let sa_val = (rs_val & 0x1F) as u32;
        self.core.write_gpr(rd_reg, (rt_val >> sa_val) as i64 as u64);
        self.handle_exec_complete()
    }
    fn exec_jr(&mut self, d: &DecodedInstr) -> ExecStatus {
        let target = self.core.read_gpr(d.rs as u32);
        self.branch_delay(target)
    }

    // JR fused with a NOP delay slot — unconditional, so always PC=target directly.
    // If THIS JR is itself executing from inside another branch's delay slot
    // (unusual but legal), it can't skip its own delay slot too — a delay slot
    // is exactly one instruction, so the "fused" NOP would actually be the real
    // next instruction after this JR's own target. Fall back to plain
    // (unfused) behavior so that NOP is fetched/executed normally.
    #[cfg(feature = "opcodefusion")]
    fn exec_jr_nop(&mut self, d: &DecodedInstr) -> ExecStatus {
        let target = self.core.read_gpr(d.rs as u32);
        if self.core.in_delay_slot {
            return self.branch_delay(target);
        }
        self.core.pc = target;
        self.exec_complete_pc_set()
    }
    fn exec_jalr(&mut self, d: &DecodedInstr) -> ExecStatus {
        let target = self.core.read_gpr(d.rs as u32);
        let rd_reg = d.rd as u32;
        self.core.write_gpr(rd_reg, self.core.pc + 8);
        self.branch_delay(target)
    }
    fn exec_syscall(&mut self, _d: &DecodedInstr) -> ExecStatus {
        let s = exec_exception(EXC_SYS);
        self.handle_exception(s)
    }
    fn exec_break(&mut self, _d: &DecodedInstr) -> ExecStatus {
        let s = exec_exception(EXC_BP);
        self.handle_exception(s)
    }
    fn exec_sync(&mut self, _d: &DecodedInstr) -> ExecStatus {
        self.handle_exec_complete()
    }
    fn exec_mfhi(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rd_reg = d.rd as u32;
        self.core.write_gpr(rd_reg, self.core.hi);
        self.handle_exec_complete()
    }
    fn exec_mthi(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        self.core.hi = rs_val;
        self.handle_exec_complete()
    }
    fn exec_mflo(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rd_reg = d.rd as u32;
        self.core.write_gpr(rd_reg, self.core.lo);
        self.handle_exec_complete()
    }
    fn exec_mtlo(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        self.core.lo = rs_val;
        self.handle_exec_complete()
    }
    fn exec_mult(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i32 as i64;
        let rt_val = self.core.read_gpr(d.rt as u32) as i32 as i64;
        let result = rs_val * rt_val;
        self.core.lo = (result as u32) as i32 as i64 as u64;
        self.core.hi = (result >> 32) as u32 as i32 as i64 as u64;
        self.handle_exec_complete()
    }
    fn exec_multu(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as u32 as u64;
        let rt_val = self.core.read_gpr(d.rt as u32) as u32 as u64;
        let result = rs_val * rt_val;
        self.core.lo = (result as u32) as i32 as i64 as u64;
        self.core.hi = (result >> 32) as u32 as i32 as i64 as u64;
        self.handle_exec_complete()
    }
    fn exec_div(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i32;
        let rt_val = self.core.read_gpr(d.rt as u32) as i32;
        if rt_val == 0 {
            self.handle_exec_complete()
        } else {
            let quotient = rs_val.wrapping_div(rt_val);
            let remainder = rs_val.wrapping_rem(rt_val);
            self.core.lo = quotient as i64 as u64;
            self.core.hi = remainder as i64 as u64;
            self.handle_exec_complete()
        }
    }
    fn exec_divu(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as u32;
        let rt_val = self.core.read_gpr(d.rt as u32) as u32;
        if rt_val == 0 {
            self.handle_exec_complete()
        } else {
            let quotient = rs_val / rt_val;
            let remainder = rs_val % rt_val;
            self.core.lo = quotient as i32 as i64 as u64;
            self.core.hi = remainder as i32 as i64 as u64;
            self.handle_exec_complete()
        }
    }
    fn exec_dmult(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64 as i128;
        let rt_val = self.core.read_gpr(d.rt as u32) as i64 as i128;
        let result = rs_val * rt_val;
        self.core.lo = result as u128 as u64;
        self.core.hi = (result >> 64) as u128 as u64;
        self.handle_exec_complete()
    }
    fn exec_dmultu(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as u128;
        let rt_val = self.core.read_gpr(d.rt as u32) as u128;
        let result = rs_val * rt_val;
        self.core.lo = result as u64;
        self.core.hi = (result >> 64) as u64;
        self.handle_exec_complete()
    }
    fn exec_ddiv(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        let rt_val = self.core.read_gpr(d.rt as u32) as i64;
        if rt_val == 0 {
            self.handle_exec_complete()
        } else if rs_val == i64::MIN && rt_val == -1 {
            self.handle_exec_complete()
        } else {
            self.core.lo = rs_val.wrapping_div(rt_val) as u64;
            self.core.hi = rs_val.wrapping_rem(rt_val) as u64;
            self.handle_exec_complete()
        }
    }
    fn exec_ddivu(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32);
        if rt_val == 0 {
            self.handle_exec_complete()
        } else {
            self.core.lo = rs_val / rt_val;
            self.core.hi = rs_val % rt_val;
            self.handle_exec_complete()
        }
    }
    fn exec_add(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i32;
        let rt_val = self.core.read_gpr(d.rt as u32) as i32;
        let rd_reg = d.rd as u32;
        match rs_val.checked_add(rt_val) {
            Some(result) => {
                self.core.write_gpr(rd_reg, result as i64 as u64);
                self.handle_exec_complete()
            }
            None => { let s = exec_exception(EXC_OV); self.handle_exception(s) }
        }
    }
    fn exec_addu(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as u32;
        let rt_val = self.core.read_gpr(d.rt as u32) as u32;
        let rd_reg = d.rd as u32;
        self.core.write_gpr(rd_reg, rs_val.wrapping_add(rt_val) as i32 as i64 as u64);
        self.handle_exec_complete()
    }
    fn exec_sub(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i32;
        let rt_val = self.core.read_gpr(d.rt as u32) as i32;
        let rd_reg = d.rd as u32;
        match rs_val.checked_sub(rt_val) {
            Some(result) => {
                self.core.write_gpr(rd_reg, result as i64 as u64);
                self.handle_exec_complete()
            }
            None => { let s = exec_exception(EXC_OV); self.handle_exception(s) }
        }
    }
    fn exec_subu(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as u32;
        let rt_val = self.core.read_gpr(d.rt as u32) as u32;
        let rd_reg = d.rd as u32;
        self.core.write_gpr(rd_reg, rs_val.wrapping_sub(rt_val) as i32 as i64 as u64);
        self.handle_exec_complete()
    }

    // ADDU fused with an immediately-following load/store that uses ADDU's
    // destination as its base register (see is_fusable_load_store). `d.imm`
    // holds the load/store's raw word untouched (decode_into leaves it there
    // instead of the usual pre-processed immediate). Neither ADDU nor the
    // fusable load/store opcode set can fault on the arithmetic itself, so the
    // only hazard is the load/store's OWN fault path (TLB miss, bus error) and
    // the delay-slot case — both handled by exec_fused_addr_load_store_tail.
    #[cfg(feature = "opcodefusion")]
    fn exec_addu_ls(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as u32;
        let rt_val = self.core.read_gpr(d.rt as u32) as u32;
        let rd_reg = d.rd as u32;
        let addr = rs_val.wrapping_add(rt_val) as i32 as i64 as u64;
        self.core.write_gpr(rd_reg, addr);
        self.exec_fused_addr_load_store_tail(addr, d.imm)
    }

    // SUBU fused with an immediately-following load/store — see exec_addu_ls.
    #[cfg(feature = "opcodefusion")]
    fn exec_subu_ls(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as u32;
        let rt_val = self.core.read_gpr(d.rt as u32) as u32;
        let rd_reg = d.rd as u32;
        let addr = rs_val.wrapping_sub(rt_val) as i32 as i64 as u64;
        self.core.write_gpr(rd_reg, addr);
        self.exec_fused_addr_load_store_tail(addr, d.imm)
    }

    /// Shared tail for exec_addu_ls/exec_subu_ls/exec_addiu_ls: `addr` is the
    /// address-calc result (already written to its register by the caller);
    /// `next_raw` is the fusable load/store's raw word (from `d.imm`).
    ///
    /// If this fused pair is itself executing from inside another branch's
    /// delay slot, it can't also skip straight to the load/store — a delay
    /// slot is exactly one instruction, so the "fused" load/store is really
    /// the actual next instruction after the branch target, not part of this
    /// one. Bail to plain (unfused) completion so the load/store gets
    /// fetched/decoded/executed normally once PC lands on delay_slot_target
    /// (mirroring exec_lui_imm32/exec_jr_nop's same fallback).
    ///
    /// Otherwise: advance PC by 4 ourselves (for the address-calc instruction)
    /// BEFORE running the load/store, so that if it faults, handle_exception's
    /// self.core.pc-based cp0_epc computation blames the load/store's own
    /// address, not the address-calc instruction's — exactly matching what
    /// would happen if these were two separate, unfused instructions.
    #[cfg(feature = "opcodefusion")]
    #[inline(always)]
    fn exec_fused_addr_load_store_tail(&mut self, addr: u64, next_raw: u32) -> ExecStatus {
        if self.core.in_delay_slot {
            return self.handle_exec_complete();
        }
        self.core.pc = self.core.pc.wrapping_add(4);

        let next_op = (next_raw >> 26) & 0x3F;
        let rt_reg = (next_raw >> 16) & 0x1F;
        let offset = (next_raw & 0xFFFF) as i16 as i64 as u64;
        let virt_addr = addr.wrapping_add(offset);

        let status = match next_op {
            OP_LB => match self.read_data::<1>(virt_addr) {
                Ok(v) => { self.core.write_gpr(rt_reg, v as i8 as i64 as u64); EXEC_COMPLETE }
                Err(s) => s,
            },
            OP_LBU => match self.read_data::<1>(virt_addr) {
                Ok(v) => { self.core.write_gpr(rt_reg, v); EXEC_COMPLETE }
                Err(s) => s,
            },
            OP_LH => match self.read_data::<2>(virt_addr) {
                Ok(v) => { self.core.write_gpr(rt_reg, v as i16 as i64 as u64); EXEC_COMPLETE }
                Err(s) => s,
            },
            OP_LHU => match self.read_data::<2>(virt_addr) {
                Ok(v) => { self.core.write_gpr(rt_reg, v); EXEC_COMPLETE }
                Err(s) => s,
            },
            OP_LW => match self.read_data::<4>(virt_addr) {
                Ok(v) => { self.core.write_gpr(rt_reg, v as i32 as i64 as u64); EXEC_COMPLETE }
                Err(s) => s,
            },
            OP_LD => match self.read_data::<8>(virt_addr) {
                Ok(v) => { self.core.write_gpr(rt_reg, v); EXEC_COMPLETE }
                Err(s) => s,
            },
            OP_SB => self.write_data::<1>(virt_addr, self.core.read_gpr(rt_reg)),
            OP_SH => self.write_data::<2>(virt_addr, self.core.read_gpr(rt_reg)),
            OP_SW => self.write_data::<4>(virt_addr, self.core.read_gpr(rt_reg)),
            OP_SD => self.write_data::<8>(virt_addr, self.core.read_gpr(rt_reg)),
            _ => unreachable!("is_fusable_load_store only allows the opcodes matched above"),
        };
        // PC was already advanced for the address-calc instruction above (and
        // we know we're not in a delay slot, or we'd have bailed earlier), so
        // on success handle_exec_complete's normal PC+4 (relative to that)
        // correctly lands past the load/store itself. On fault, `status`
        // carries the exception; handle_exception reads self.core.pc (already
        // at the load/store's address) for cp0_epc.
        if status == EXEC_COMPLETE {
            self.handle_exec_complete()
        } else {
            self.handle_exception(status)
        }
    }
    
    fn exec_and(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32);
        let rd_reg = d.rd as u32;
        self.core.write_gpr(rd_reg, rs_val & rt_val);
        self.handle_exec_complete()
    }
    fn exec_or(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32);
        let rd_reg = d.rd as u32;
        self.core.write_gpr(rd_reg, rs_val | rt_val);
        self.handle_exec_complete()
    }
    fn exec_xor(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32);
        let rd_reg = d.rd as u32;
        self.core.write_gpr(rd_reg, rs_val ^ rt_val);
        self.handle_exec_complete()
    }
    fn exec_nor(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32);
        let rd_reg = d.rd as u32;
        self.core.write_gpr(rd_reg, !(rs_val | rt_val));
        self.handle_exec_complete()
    }
    fn exec_slt(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        let rt_val = self.core.read_gpr(d.rt as u32) as i64;
        let rd_reg = d.rd as u32;
        self.core.write_gpr(rd_reg, if rs_val < rt_val { 1 } else { 0 });
        self.handle_exec_complete()
    }
    fn exec_sltu(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32);
        let rd_reg = d.rd as u32;
        self.core.write_gpr(rd_reg, if rs_val < rt_val { 1 } else { 0 });
        self.handle_exec_complete()
    }
    fn exec_dadd(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        let rt_val = self.core.read_gpr(d.rt as u32) as i64;
        let rd_reg = d.rd as u32;
        match rs_val.checked_add(rt_val) {
            Some(result) => {
                self.core.write_gpr(rd_reg, result as u64);
                self.handle_exec_complete()
            }
            None => { let s = exec_exception(EXC_OV); self.handle_exception(s) }
        }
    }
    fn exec_daddu(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32);
        let rd_reg = d.rd as u32;
        self.core.write_gpr(rd_reg, rs_val.wrapping_add(rt_val));
        self.handle_exec_complete()
    }
    fn exec_dsub(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        let rt_val = self.core.read_gpr(d.rt as u32) as i64;
        let rd_reg = d.rd as u32;
        match rs_val.checked_sub(rt_val) {
            Some(result) => {
                self.core.write_gpr(rd_reg, result as u64);
                self.handle_exec_complete()
            }
            None => { let s = exec_exception(EXC_OV); self.handle_exception(s) }
        }
    }
    fn exec_dsubu(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32);
        let rd_reg = d.rd as u32;
        self.core.write_gpr(rd_reg, rs_val.wrapping_sub(rt_val));
        self.handle_exec_complete()
    }
    fn exec_tge(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        let rt_val = self.core.read_gpr(d.rt as u32) as i64;
        if rs_val >= rt_val { let s = exec_exception(EXC_TR); self.handle_exception(s) } else { self.handle_exec_complete() }
    }
    fn exec_tgeu(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32);
        if rs_val >= rt_val { let s = exec_exception(EXC_TR); self.handle_exception(s) } else { self.handle_exec_complete() }
    }
    fn exec_tlt(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        let rt_val = self.core.read_gpr(d.rt as u32) as i64;
        if rs_val < rt_val { let s = exec_exception(EXC_TR); self.handle_exception(s) } else { self.handle_exec_complete() }
    }
    fn exec_tltu(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32);
        if rs_val < rt_val { let s = exec_exception(EXC_TR); self.handle_exception(s) } else { self.handle_exec_complete() }
    }
    fn exec_teq(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32);
        if rs_val == rt_val { let s = exec_exception(EXC_TR); self.handle_exception(s) } else { self.handle_exec_complete() }
    }
    fn exec_tne(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32);
        if rs_val != rt_val { let s = exec_exception(EXC_TR); self.handle_exception(s) } else { self.handle_exec_complete() }
    }
    fn exec_movz(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32);
        let rd_reg = d.rd as u32;
        if rt_val == 0 { self.core.write_gpr(rd_reg, rs_val); }
        self.handle_exec_complete()
    }
    fn exec_movn(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32);
        let rd_reg = d.rd as u32;
        if rt_val != 0 { self.core.write_gpr(rd_reg, rs_val); }
        self.handle_exec_complete()
    }
    fn exec_dsll(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rt_val = self.core.read_gpr(d.rt as u32);
        let rd_reg = d.rd as u32;
        let sa_val = d.sa as u32;
        self.core.write_gpr(rd_reg, rt_val << sa_val);
        self.handle_exec_complete()
    }
    fn exec_dsrl(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rt_val = self.core.read_gpr(d.rt as u32);
        let rd_reg = d.rd as u32;
        let sa_val = d.sa as u32;
        self.core.write_gpr(rd_reg, rt_val >> sa_val);
        self.handle_exec_complete()
    }
    fn exec_dsra(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rt_val = self.core.read_gpr(d.rt as u32) as i64;
        let rd_reg = d.rd as u32;
        let sa_val = d.sa as u32;
        self.core.write_gpr(rd_reg, (rt_val >> sa_val) as u64);
        self.handle_exec_complete()
    }
    fn exec_dsll32(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rt_val = self.core.read_gpr(d.rt as u32);
        let rd_reg = d.rd as u32;
        let sa_val = d.sa as u32;
        self.core.write_gpr(rd_reg, rt_val << (sa_val + 32));
        self.handle_exec_complete()
    }
    fn exec_dsrl32(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rt_val = self.core.read_gpr(d.rt as u32);
        let rd_reg = d.rd as u32;
        let sa_val = d.sa as u32;
        self.core.write_gpr(rd_reg, rt_val >> (sa_val + 32));
        self.handle_exec_complete()
    }
    fn exec_dsra32(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rt_val = self.core.read_gpr(d.rt as u32) as i64;
        let rd_reg = d.rd as u32;
        let sa_val = d.sa as u32;
        self.core.write_gpr(rd_reg, (rt_val >> (sa_val + 32)) as u64);
        self.handle_exec_complete()
    }
    fn exec_dsllv(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32);
        let rd_reg = d.rd as u32;
        let sa_val = rs_val & 0x3F;
        self.core.write_gpr(rd_reg, rt_val << sa_val);
        self.handle_exec_complete()
    }
    fn exec_dsrlv(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32);
        let rd_reg = d.rd as u32;
        let sa_val = rs_val & 0x3F;
        self.core.write_gpr(rd_reg, rt_val >> sa_val);
        self.handle_exec_complete()
    }
    fn exec_dsrav(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32) as i64;
        let rd_reg = d.rd as u32;
        let sa_val = rs_val & 0x3F;
        self.core.write_gpr(rd_reg, (rt_val >> sa_val) as u64);
        self.handle_exec_complete()
    }
    // Jump and Branch Instructions

    // J - Jump
    // Unconditional jump within 256MB region
    fn exec_j(&mut self, d: &DecodedInstr) -> ExecStatus {
        // d.immi64() = (target26 << 2): replace low 28 bits of PC+4
        let target = ((self.core.pc + 4) & 0xFFFFFFFF_F0000000) | d.immi64();
        self.branch_delay(target)
    }

    // J fused with a NOP delay slot — unconditional, so always PC=target directly.
    // See exec_jr_nop for why in_delay_slot must fall back to unfused behavior.
    #[cfg(feature = "opcodefusion")]
    fn exec_j_nop(&mut self, d: &DecodedInstr) -> ExecStatus {
        let target = ((self.core.pc + 4) & 0xFFFFFFFF_F0000000) | d.immi64();
        if self.core.in_delay_slot {
            return self.branch_delay(target);
        }
        self.core.pc = target;
        self.exec_complete_pc_set()
    }

    // JAL - Jump and Link
    // Unconditional jump, save return address in r31
    fn exec_jal(&mut self, d: &DecodedInstr) -> ExecStatus {
        let target = ((self.core.pc + 4) & 0xFFFFFFFF_F0000000) | d.immi64();
        self.core.write_gpr(31, self.core.pc + 8); // Return address (PC of delay slot + 4)
        self.branch_delay(target)
    }

    // JAL fused with a NOP delay slot — unconditional, so always PC=target directly.
    // See exec_jr_nop for why in_delay_slot must fall back to unfused behavior.
    #[cfg(feature = "opcodefusion")]
    fn exec_jal_nop(&mut self, d: &DecodedInstr) -> ExecStatus {
        let target = ((self.core.pc + 4) & 0xFFFFFFFF_F0000000) | d.immi64();
        self.core.write_gpr(31, self.core.pc + 8); // Return address (PC of delay slot + 4)
        if self.core.in_delay_slot {
            return self.branch_delay(target);
        }
        self.core.pc = target;
        self.exec_complete_pc_set()
    }

    // BEQ - Branch on Equal
    fn exec_beq(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32);
        if rs_val == rt_val {
            let target = self.core.pc.wrapping_add(4).wrapping_add(d.immu64());
            self.branch_delay(target)
        } else {
            self.handle_branch_not_taken()
        }
    }

    // BEQ fused with a NOP delay slot (see FLAG_IMM_IS_NEXT): the delay slot is
    // never fetched/decoded/dispatched. Taken sets PC=target directly; not
    // taken uses handle_branch_likely_skip's PC+=8. If THIS branch is itself
    // in another branch's delay slot (see exec_jr_nop), neither shortcut is
    // safe — fall back to plain behavior so the "fused" NOP is
    // fetched/executed normally as the real delay slot.
    #[cfg(feature = "opcodefusion")]
    fn exec_beq_nop(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32);
        if rs_val == rt_val {
            let target = self.core.pc.wrapping_add(4).wrapping_add(d.immu64());
            if self.core.in_delay_slot {
                return self.branch_delay(target);
            }
            self.core.pc = target;
            self.exec_complete_pc_set()
        } else if self.core.in_delay_slot {
            self.handle_exec_complete()
        } else {
            self.handle_branch_likely_skip()
        }
    }

    // BNE - Branch on Not Equal
    fn exec_bne(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32);
        if rs_val != rt_val {
            let target = self.core.pc.wrapping_add(4).wrapping_add(d.immu64());
            self.branch_delay(target)
        } else {
            self.handle_branch_not_taken()
        }
    }

    // BNE fused with a NOP delay slot — see exec_beq_nop.
    #[cfg(feature = "opcodefusion")]
    fn exec_bne_nop(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32);
        if rs_val != rt_val {
            let target = self.core.pc.wrapping_add(4).wrapping_add(d.immu64());
            if self.core.in_delay_slot {
                return self.branch_delay(target);
            }
            self.core.pc = target;
            self.exec_complete_pc_set()
        } else if self.core.in_delay_slot {
            self.handle_exec_complete()
        } else {
            self.handle_branch_likely_skip()
        }
    }

    // BLEZ - Branch on Less Than or Equal to Zero
    fn exec_blez(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        if rs_val <= 0 {
            let target = self.core.pc.wrapping_add(4).wrapping_add(d.immu64());
            self.branch_delay(target)
        } else {
            self.handle_branch_not_taken()
        }
    }

    // BGTZ - Branch on Greater Than Zero
    fn exec_bgtz(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        if rs_val > 0 {
            let target = self.core.pc.wrapping_add(4).wrapping_add(d.immu64());
            self.branch_delay(target)
        } else {
            self.handle_branch_not_taken()
        }
    }

    // BEQL - Branch on Equal Likely
    fn exec_beql(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32);
        if rs_val == rt_val {
            let target = self.core.pc.wrapping_add(4).wrapping_add(d.immu64());
            self.branch_delay(target)
        } else {
            self.handle_branch_likely_skip()
        }
    }

    // BNEL - Branch on Not Equal Likely
    fn exec_bnel(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32);
        if rs_val != rt_val {
            let target = self.core.pc.wrapping_add(4).wrapping_add(d.immu64());
            self.branch_delay(target)
        } else {
            self.handle_branch_likely_skip()
        }
    }

    // BLEZL - Branch on Less Than or Equal to Zero Likely
    fn exec_blezl(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        if rs_val <= 0 {
            let target = self.core.pc.wrapping_add(4).wrapping_add(d.immu64());
            self.branch_delay(target)
        } else {
            self.handle_branch_likely_skip()
        }
    }

    // BGTZL - Branch on Greater Than Zero Likely
    fn exec_bgtzl(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        if rs_val > 0 {
            let target = self.core.pc.wrapping_add(4).wrapping_add(d.immu64());
            self.branch_delay(target)
        } else {
            self.handle_branch_likely_skip()
        }
    }


    // REGIMM individual methods
    fn exec_bltz(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        let target = self.core.pc.wrapping_add(4).wrapping_add(d.immu64());
        if rs_val < 0 { self.branch_delay(target) } else { self.handle_branch_not_taken() }
    }
    fn exec_bgez(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        let target = self.core.pc.wrapping_add(4).wrapping_add(d.immu64());
        if rs_val >= 0 { self.branch_delay(target) } else { self.handle_branch_not_taken() }
    }
    fn exec_bltzl_ri(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        let target = self.core.pc.wrapping_add(4).wrapping_add(d.immu64());
        if rs_val < 0 { self.branch_delay(target) } else { self.handle_branch_likely_skip() }
    }
    fn exec_bgezl_ri(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        let target = self.core.pc.wrapping_add(4).wrapping_add(d.immu64());
        if rs_val >= 0 { self.branch_delay(target) } else { self.handle_branch_likely_skip() }
    }
    fn exec_tgei(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        if rs_val >= d.imms64() { let s = exec_exception(EXC_TR); self.handle_exception(s) } else { self.handle_exec_complete() }
    }
    fn exec_tgeiu(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val_u = self.core.read_gpr(d.rs as u32);
        if rs_val_u >= d.immu64() { let s = exec_exception(EXC_TR); self.handle_exception(s) } else { self.handle_exec_complete() }
    }
    fn exec_tlti(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        if rs_val < d.imms64() { let s = exec_exception(EXC_TR); self.handle_exception(s) } else { self.handle_exec_complete() }
    }
    fn exec_tltiu(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val_u = self.core.read_gpr(d.rs as u32);
        if rs_val_u < d.immu64() { let s = exec_exception(EXC_TR); self.handle_exception(s) } else { self.handle_exec_complete() }
    }
    fn exec_teqi(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        if rs_val == d.imms64() { let s = exec_exception(EXC_TR); self.handle_exception(s) } else { self.handle_exec_complete() }
    }
    fn exec_tnei(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        if rs_val != d.imms64() { let s = exec_exception(EXC_TR); self.handle_exception(s) } else { self.handle_exec_complete() }
    }
    fn exec_bltzal(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        let target = self.core.pc.wrapping_add(4).wrapping_add(d.immu64());
        self.core.write_gpr(31, self.core.pc + 8);
        if rs_val < 0 { self.branch_delay(target) } else { self.handle_branch_not_taken() }
    }
    fn exec_bgezal(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        let target = self.core.pc.wrapping_add(4).wrapping_add(d.immu64());
        self.core.write_gpr(31, self.core.pc + 8);
        if rs_val >= 0 { self.branch_delay(target) } else { self.handle_branch_not_taken() }
    }
    fn exec_bltzall(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        let target = self.core.pc.wrapping_add(4).wrapping_add(d.immu64());
        self.core.write_gpr(31, self.core.pc + 8);
        if rs_val < 0 { self.branch_delay(target) } else { self.handle_branch_likely_skip() }
    }
    fn exec_bgezall(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        let target = self.core.pc.wrapping_add(4).wrapping_add(d.immu64());
        self.core.write_gpr(31, self.core.pc + 8);
        if rs_val >= 0 { self.branch_delay(target) } else { self.handle_branch_likely_skip() }
    }
    // Immediate arithmetic/logic instructions

    // ADDI - Add Immediate (with overflow exception)
    // 32-bit operation: sign-extends immediate and low 32 bits of rs, adds them,
    // checks overflow, then sign-extends result to 64 bits
    fn exec_addi(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i32;  // Low 32 bits, sign-extended
        let imm_val = d.imms64() as i32;            // already sign-extended at decode
        let rt_reg = d.rt as u32;

        match rs_val.checked_add(imm_val) {
            Some(result) => {
                // Sign-extend 32-bit result to 64 bits
                self.core.write_gpr(rt_reg, result as i64 as u64);
                self.handle_exec_complete()
            }
            None => { let s = exec_exception(EXC_OV); self.handle_exception(s) }
        }
    }

    // ADDIU - Add Immediate Unsigned (no overflow exception)
    // 32-bit operation: adds low 32 bits of rs and sign-extended immediate,
    // then sign-extends result to 64 bits
    fn exec_addiu(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as u32;  // Low 32 bits
        let imm_val = d.immi64() as u32;            // sign-extended at decode, truncate to 32
        let rt_reg = d.rt as u32;
        // Wrapping add, then sign-extend to 64 bits
        let result = rs_val.wrapping_add(imm_val) as i32 as i64 as u64;
        self.core.write_gpr(rt_reg, result);
        self.handle_exec_complete()
    }

    // ADDIU fused with an immediately-following load/store that uses ADDIU's
    // destination as its base register (see is_fusable_load_store). Unlike the
    // R-type ADDU/SUBU fusions, `d.imm` here holds the load/store's raw word,
    // NOT the usual sign-extended immediate (decode_into skips set_imm_se for
    // this case) — so the ADDIU's own immediate is re-derived from d.raw's
    // low 16 bits directly, same trick used by exec_lui_imm32's delay-slot
    // fallback.
    #[cfg(feature = "opcodefusion")]
    fn exec_addiu_ls(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as u32;
        let imm_val = (d.raw & 0xFFFF) as i16 as i32 as u32;
        let rt_reg = d.rt as u32;
        let addr = rs_val.wrapping_add(imm_val) as i32 as i64 as u64;
        self.core.write_gpr(rt_reg, addr);
        self.exec_fused_addr_load_store_tail(addr, d.imm)
    }

    // DADDI - Doubleword Add Immediate (with overflow exception)
    fn exec_daddi(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        let rt_reg = d.rt as u32;
        match rs_val.checked_add(d.imms64()) {
            Some(result) => {
                self.core.write_gpr(rt_reg, result as u64);
                self.handle_exec_complete()
            }
            None => { let s = exec_exception(EXC_OV); self.handle_exception(s) }
        }
    }

    // DADDIU - Doubleword Add Immediate Unsigned (no overflow exception)
    fn exec_daddiu(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_reg = d.rt as u32;
        self.core.write_gpr(rt_reg, rs_val.wrapping_add(d.immu64()));
        self.handle_exec_complete()
    }

    // SLTI - Set on Less Than Immediate (signed)
    fn exec_slti(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        let rt_reg = d.rt as u32;
        self.core.write_gpr(rt_reg, if rs_val < d.imms64() { 1 } else { 0 });
        self.handle_exec_complete()
    }

    // SLTIU - Set on Less Than Immediate Unsigned (sign-extended imm compared as unsigned)
    fn exec_sltiu(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_reg = d.rt as u32;
        self.core.write_gpr(rt_reg, if rs_val < d.immu64() { 1 } else { 0 });
        self.handle_exec_complete()
    }

    // ANDI - AND Immediate (zero-extended)
    fn exec_andi(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_reg = d.rt as u32;
        self.core.write_gpr(rt_reg, rs_val & d.immi64());
        self.handle_exec_complete()
    }

    // ORI - OR Immediate (zero-extended)
    fn exec_ori(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_reg = d.rt as u32;
        self.core.write_gpr(rt_reg, rs_val | d.immi64());
        self.handle_exec_complete()
    }

    // XORI - XOR Immediate (zero-extended)
    fn exec_xori(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_reg = d.rt as u32;
        self.core.write_gpr(rt_reg, rs_val ^ d.immi64());
        self.handle_exec_complete()
    }

    // LUI - Load Upper Immediate (pre-shifted and sign-extended at decode)
    fn exec_lui(&mut self, d: &DecodedInstr) -> ExecStatus {
        self.core.write_gpr(d.rt as u32, d.immu64());
        self.handle_exec_complete()
    }

    // LUI fused with a same-register ORI the
    // immediately-following instruction — the
    // common 32-bit-immediate materialization idiom `lui rX,hi; ori rX,rX,lo`.
    // decode_into has already verified next is ORI with rs==rt==this rt and
    // pre-combined the result into `imm` (zero-extend combine: no carry, so
    // decode-time combination is exact). Skips decoding/dispatching the ORI.
    #[cfg(feature = "opcodefusion")]
    fn exec_lui_imm32(&mut self, d: &DecodedInstr) -> ExecStatus {
        if self.core.in_delay_slot {
            // This LUI is itself a delay-slot instruction: the fused ORI is NOT
            // also in the delay slot (a delay slot is exactly one instruction) —
            // it's really the instruction at/after the branch target. Treat this
            // as plain unfused LUI so the ORI gets fetched and executed normally
            // once PC lands on delay_slot_target; d.raw is still the original LUI
            // word (only d.imm was repurposed for the fused combine).
            self.core.write_gpr(d.rt as u32, ((d.raw & 0xFFFF) << 16) as i32 as i64 as u64);
            return self.handle_exec_complete();
        }
        self.core.write_gpr(d.rt as u32, d.immu64());
        self.handle_branch_likely_skip()
    }

    // LUI fused with a same-register ADDIU — `lui rX,hi; addiu rX,rX,lo`.
    // Unlike ORI, ADDIU's sign-extending add can carry into bit 16 when lo16's
    // sign bit is set, so decode_into pre-computes the actual wrapping-add
    // result (matching exec_addiu's semantics exactly) rather than just OR-ing
    // the halves together.
    #[cfg(feature = "opcodefusion")]
    fn exec_lui_simm32(&mut self, d: &DecodedInstr) -> ExecStatus {
        if self.core.in_delay_slot {
            // See exec_lui_imm32: this LUI is a delay-slot instruction, so the
            // fused ADDIU is really the (unfused) instruction after the branch
            // target — fall back to plain LUI semantics and let it be fetched
            // and executed normally once PC lands on delay_slot_target.
            self.core.write_gpr(d.rt as u32, ((d.raw & 0xFFFF) << 16) as i32 as i64 as u64);
            return self.handle_exec_complete();
        }
        self.core.write_gpr(d.rt as u32, d.immu64());
        self.handle_branch_likely_skip()
    }

    // Load/Store instructions (converted to use new interface)

    // LB - Load Byte (sign-extended)
    fn exec_lb(&mut self, d: &DecodedInstr) -> ExecStatus {
        let base = self.core.read_gpr(d.rs as u32);
        let virt_addr = base.wrapping_add(d.immu64());
        let rt_reg = d.rt as u32;

        match self.read_data::<1>(virt_addr) {
            Ok(value) => {
                // Sign-extend byte to 64 bits
                self.core.write_gpr(rt_reg, value as i8 as i64 as u64);
                self.handle_exec_complete()
            }
            Err(status) => self.finish_status(status)
        }
    }

    // LBU - Load Byte Unsigned (zero-extended)
    fn exec_lbu(&mut self, d: &DecodedInstr) -> ExecStatus {
        let base = self.core.read_gpr(d.rs as u32);
        let virt_addr = base.wrapping_add(d.immu64());
        let rt_reg = d.rt as u32;

        match self.read_data::<1>(virt_addr) {
            Ok(value) => {
                self.core.write_gpr(rt_reg, value);
                self.handle_exec_complete()
            }
            Err(status) => self.finish_status(status)
        }
    }

    // LH - Load Halfword (sign-extended)
    fn exec_lh(&mut self, d: &DecodedInstr) -> ExecStatus {
        let base = self.core.read_gpr(d.rs as u32);
        let virt_addr = base.wrapping_add(d.immu64());
        let rt_reg = d.rt as u32;

        match self.read_data::<2>(virt_addr) {
            Ok(value) => {
                // Sign-extend halfword to 64 bits
                self.core.write_gpr(rt_reg, value as i16 as i64 as u64);
                self.handle_exec_complete()
            }
            Err(status) => self.finish_status(status)
        }
    }

    // LHU - Load Halfword Unsigned (zero-extended)
    fn exec_lhu(&mut self, d: &DecodedInstr) -> ExecStatus {
        let base = self.core.read_gpr(d.rs as u32);
        let virt_addr = base.wrapping_add(d.immu64());
        let rt_reg = d.rt as u32;

        match self.read_data::<2>(virt_addr) {
            Ok(value) => {
                self.core.write_gpr(rt_reg, value);
                self.handle_exec_complete()
            }
            Err(status) => self.finish_status(status)
        }
    }

    // LW - Load Word (sign-extended)
    fn exec_lw(&mut self, d: &DecodedInstr) -> ExecStatus {
        let base = self.core.read_gpr(d.rs as u32);
        let virt_addr = base.wrapping_add(d.immu64());
        let rt_reg = d.rt as u32;

        match self.read_data::<4>(virt_addr) {
            Ok(value) => {
                // Sign-extend word to 64 bits
                self.core.write_gpr(rt_reg, value as i32 as i64 as u64);
                self.handle_exec_complete()
            }
            Err(status) => self.finish_status(status)
        }
    }

    // LWU - Load Word Unsigned (zero-extended, MIPS III 64-bit)
    fn exec_lwu(&mut self, d: &DecodedInstr) -> ExecStatus {
        let base = self.core.read_gpr(d.rs as u32);
        let virt_addr = base.wrapping_add(d.immu64());
        let rt_reg = d.rt as u32;

        match self.read_data::<4>(virt_addr) {
            Ok(value) => {
                // Zero-extend word to 64 bits
                self.core.write_gpr(rt_reg, value as u64);
                self.handle_exec_complete()
            }
            Err(status) => self.finish_status(status)
        }
    }

    // LD - Load Doubleword (MIPS III, 64-bit)
    fn exec_ld(&mut self, d: &DecodedInstr) -> ExecStatus {
        let base = self.core.read_gpr(d.rs as u32);
        let virt_addr = base.wrapping_add(d.immu64());
        let rt_reg = d.rt as u32;

        match self.read_data::<8>(virt_addr) {
            Ok(value) => {
                self.core.write_gpr(rt_reg, value);
                self.handle_exec_complete()
            }
            Err(status) => self.finish_status(status)
        }
    }

    // SB - Store Byte
    fn exec_sb(&mut self, d: &DecodedInstr) -> ExecStatus {
        let base = self.core.read_gpr(d.rs as u32);
        let virt_addr = base.wrapping_add(d.immu64());
        let rt_val = self.core.read_gpr(d.rt as u32);

        let status = self.write_data::<1>(virt_addr, rt_val);
        self.finish_status(status)
    }

    // SH - Store Halfword
    fn exec_sh(&mut self, d: &DecodedInstr) -> ExecStatus {
        let base = self.core.read_gpr(d.rs as u32);
        let virt_addr = base.wrapping_add(d.immu64());
        let rt_val = self.core.read_gpr(d.rt as u32);

        let status = self.write_data::<2>(virt_addr, rt_val);
        self.finish_status(status)
    }

    // SW - Store Word
    fn exec_sw(&mut self, d: &DecodedInstr) -> ExecStatus {
        let base = self.core.read_gpr(d.rs as u32);
        let virt_addr = base.wrapping_add(d.immu64());
        let rt_val = self.core.read_gpr(d.rt as u32);

        let status = self.write_data::<4>(virt_addr, rt_val);
        self.finish_status(status)
    }

    // SD - Store Doubleword (MIPS III, 64-bit)
    fn exec_sd(&mut self, d: &DecodedInstr) -> ExecStatus {
        let base = self.core.read_gpr(d.rs as u32);
        let virt_addr = base.wrapping_add(d.immu64());
        let rt_val = self.core.read_gpr(d.rt as u32);

        let status = self.write_data::<8>(virt_addr, rt_val);
        self.finish_status(status)
    }

    // LWL - Load Word Left
    // Loads the left portion of a word from an unaligned address
    // For big-endian: loads from MSB down to the byte at virt_addr
    fn exec_lwl(&mut self, d: &DecodedInstr) -> ExecStatus {
        let base = self.core.read_gpr(d.rs as u32);
        let virt_addr = base.wrapping_add(d.immu64());
        let rt_reg = d.rt as u32;

        // Align address to word boundary
        let aligned_addr = virt_addr & !3;
        let byte_offset = (virt_addr & 3) as usize;

        // Read the aligned word
        match self.read_data::<4>(aligned_addr) {
            Ok(mem_word) => {
                let mem_word = mem_word as u32;
                let rt_val = self.core.read_gpr(rt_reg) as u32;

                // Big-endian byte offset to shift amount mapping:
                // offset 0: load all 4 bytes (shift 0)
                // offset 1: load 3 bytes (shift 8)
                // offset 2: load 2 bytes (shift 16)
                // offset 3: load 1 byte (shift 24)
                let shift = byte_offset * 8;

                // Mask preserves lower bytes of rt, loads upper bytes from memory
                let mask = 0xFFFFFFFFu32 << shift;
                let result = (mem_word << shift) | (rt_val & !mask);

                // Sign-extend to 64 bits
                self.core.write_gpr(rt_reg, result as i32 as i64 as u64);
                self.handle_exec_complete()
            }
            Err(status) => self.finish_status(status)
        }
    }

    // LWR - Load Word Right
    // Loads the right portion of a word from an unaligned address
    // For big-endian: loads from the byte at virt_addr down to LSB
    fn exec_lwr(&mut self, d: &DecodedInstr) -> ExecStatus {
        let base = self.core.read_gpr(d.rs as u32);
        let virt_addr = base.wrapping_add(d.immu64());
        let rt_reg = d.rt as u32;

        // Align address to word boundary
        let aligned_addr = virt_addr & !3;
        let byte_offset = (virt_addr & 3) as usize;

        // Read the aligned word
        match self.read_data::<4>(aligned_addr) {
            Ok(mem_word) => {
                let mem_word = mem_word as u32;
                let rt_val = self.core.read_gpr(rt_reg) as u32;

                // Big-endian byte offset to shift amount mapping:
                // offset 0: load 1 byte (shift 24)
                // offset 1: load 2 bytes (shift 16)
                // offset 2: load 3 bytes (shift 8)
                // offset 3: load all 4 bytes (shift 0)
                let shift = (3 - byte_offset) * 8;

                // Mask preserves upper bytes of rt, loads lower bytes from memory
                let mask = 0xFFFFFFFFu32 >> shift;
                let result = (mem_word >> shift) | (rt_val & !mask);

                // Sign-extend to 64 bits
                self.core.write_gpr(rt_reg, result as i32 as i64 as u64);
                self.handle_exec_complete()
            }
            Err(status) => self.finish_status(status)
        }
    }

    // SWL - Store Word Left
    // Stores the left portion of a word to an unaligned address
    // For big-endian: stores from MSB down to the byte at virt_addr
    fn exec_swl(&mut self, d: &DecodedInstr) -> ExecStatus {
        let base = self.core.read_gpr(d.rs as u32);
        let virt_addr = base.wrapping_add(d.immu64());
        let rt_val = self.core.read_gpr(d.rt as u32) as u32;

        let byte_offset = (virt_addr & 3) as usize;
        // Big-endian byte offset to shift and mask:
        // offset 0: store all 4 bytes (mask 0xFFFFFFFF)
        // offset 1: store 3 bytes (mask 0x00FFFFFF)
        // offset 2: store 2 bytes (mask 0x0000FFFF)
        // offset 3: store 1 byte (mask 0x000000FF)
        let word_shift = byte_offset * 8;
        let word_mask = 0xFFFFFFFFu32 >> word_shift;
        let word_val  = rt_val >> word_shift;
        // Promote word mask/val into doubleword space at the dword-aligned address
        let aligned8  = virt_addr & !7;
        let half      = (virt_addr & 4) as usize; // 0 = upper dword half, 4 = lower
        let dw_shift  = (4 - half) << 3;          // 32 for upper half, 0 for lower
        let status = self.write_data64_masked(aligned8, (word_val as u64) << dw_shift, (word_mask as u64) << dw_shift);
        self.finish_status(status)
    }

    // SWR - Store Word Right
    // Stores the right portion of a word to an unaligned address
    // For big-endian: stores from the byte at virt_addr down to LSB
    fn exec_swr(&mut self, d: &DecodedInstr) -> ExecStatus {
        let base = self.core.read_gpr(d.rs as u32);
        let virt_addr = base.wrapping_add(d.immu64());
        let rt_val = self.core.read_gpr(d.rt as u32) as u32;

        let byte_offset = (virt_addr & 3) as usize;
        // Big-endian byte offset to shift and mask:
        // offset 0: store 1 byte (mask 0xFF000000)
        // offset 1: store 2 bytes (mask 0xFFFF0000)
        // offset 2: store 3 bytes (mask 0xFFFFFF00)
        // offset 3: store all 4 bytes (mask 0xFFFFFFFF)
        let word_shift = (3 - byte_offset) * 8;
        let word_mask  = 0xFFFFFFFFu32 << word_shift;
        let word_val   = rt_val << word_shift;
        // Promote word mask/val into doubleword space at the dword-aligned address
        let aligned8  = virt_addr & !7;
        let half      = (virt_addr & 4) as usize; // 0 = upper dword half, 4 = lower
        let dw_shift  = (4 - half) << 3;          // 32 for upper half, 0 for lower
        let status = self.write_data64_masked(aligned8, (word_val as u64) << dw_shift, (word_mask as u64) << dw_shift);
        self.finish_status(status)
    }

    // LDL - Load Doubleword Left (MIPS III)
    // Loads the left portion of a doubleword from an unaligned address
    fn exec_ldl(&mut self, d: &DecodedInstr) -> ExecStatus {
        let base = self.core.read_gpr(d.rs as u32);
        let virt_addr = base.wrapping_add(d.immu64());
        let rt_reg = d.rt as u32;

        // Align address to doubleword boundary
        let aligned_addr = virt_addr & !7;
        let byte_offset = (virt_addr & 7) as usize;

        // Read the aligned doubleword
        match self.read_data::<8>(aligned_addr) {
            Ok(mem_dword) => {
                let rt_val = self.core.read_gpr(rt_reg);

                // Big-endian byte offset to shift amount
                let shift = byte_offset * 8;

                // Mask preserves lower bytes of rt, loads upper bytes from memory
                let mask = 0xFFFFFFFFFFFFFFFFu64 << shift;
                let result = (mem_dword << shift) | (rt_val & !mask);

                self.core.write_gpr(rt_reg, result);
                self.handle_exec_complete()
            }
            Err(status) => self.finish_status(status)
        }
    }

    // LDR - Load Doubleword Right (MIPS III)
    // Loads the right portion of a doubleword from an unaligned address
    fn exec_ldr(&mut self, d: &DecodedInstr) -> ExecStatus {
        let base = self.core.read_gpr(d.rs as u32);
        let virt_addr = base.wrapping_add(d.immu64());
        let rt_reg = d.rt as u32;

        // Align address to doubleword boundary
        let aligned_addr = virt_addr & !7;
        let byte_offset = (virt_addr & 7) as usize;

        // Read the aligned doubleword
        match self.read_data::<8>(aligned_addr) {
            Ok(mem_dword) => {
                let rt_val = self.core.read_gpr(rt_reg);

                // Big-endian byte offset to shift amount
                let shift = (7 - byte_offset) * 8;

                // Mask preserves upper bytes of rt, loads lower bytes from memory
                let mask = 0xFFFFFFFFFFFFFFFFu64 >> shift;
                let result = (mem_dword >> shift) | (rt_val & !mask);

                self.core.write_gpr(rt_reg, result);
                self.handle_exec_complete()
            }
            Err(status) => self.finish_status(status)
        }
    }

    // SDL - Store Doubleword Left (MIPS III)
    // Stores the left portion of a doubleword to an unaligned address
    fn exec_sdl(&mut self, d: &DecodedInstr) -> ExecStatus {
        let base = self.core.read_gpr(d.rs as u32);
        let virt_addr = base.wrapping_add(d.immu64());
        let rt_val = self.core.read_gpr(d.rt as u32);

        // Align address to doubleword boundary
        let aligned_addr = virt_addr & !7;
        let byte_offset = (virt_addr & 7) as usize;

        // Big-endian byte offset to shift and mask
        let shift = byte_offset * 8;
        let mask = 0xFFFFFFFFFFFFFFFFu64 >> shift;
        let value = rt_val >> shift;

        let status = self.write_data64_masked(aligned_addr, value, mask);
        self.finish_status(status)
    }

    // SDR - Store Doubleword Right (MIPS III)
    // Stores the right portion of a doubleword to an unaligned address
    fn exec_sdr(&mut self, d: &DecodedInstr) -> ExecStatus {
        let base = self.core.read_gpr(d.rs as u32);
        let virt_addr = base.wrapping_add(d.immu64());
        let rt_val = self.core.read_gpr(d.rt as u32);

        // Align address to doubleword boundary
        let aligned_addr = virt_addr & !7;
        let byte_offset = (virt_addr & 7) as usize;

        // Big-endian byte offset to shift and mask
        let shift = (7 - byte_offset) * 8;
        let mask = 0xFFFFFFFFFFFFFFFFu64 << shift;
        let value = rt_val << shift;

        let status = self.write_data64_masked(aligned_addr, value, mask);
        self.finish_status(status)
    }


    // CACHE Instruction
    fn exec_cache(&mut self, d: &DecodedInstr) -> ExecStatus {
        // Check CP0 usability (must be kernel or supervisor, or CU0 set)
        let privilege = self.core.get_privilege_mode();
        use crate::mips_core::{PrivilegeMode, STATUS_CU0};

        let cp0_usable = match privilege {
            PrivilegeMode::Kernel => true,
            _ => (self.core.cp0_status & STATUS_CU0) != 0,
        };

        if !cp0_usable {
            return self.cpu_unusable(0);
        }

        let base = self.core.read_gpr(d.rs as u32);
        let virt_addr = base.wrapping_add(d.immu64());

        let cache_op = d.rt as u32;  // Encoded operation: cache_sel in bits[1:0], op in bits[4:2]
        let op = cache_op & 0x1C;

        // Determine if this is a Hit operation that needs address translation
        let needs_translation = matches!(op, C_CDX | C_HINV | C_HWBINV | C_HWB | C_HSV);

        let phys_addr = if needs_translation {
            // Hit operations need address translation
            let tr = self.nanotlb_translate::<{AccessType::Read as u8}>(virt_addr);
            if tr.is_exception() { return self.handle_exception(tr.status); }
            tr.phys as u64
        } else {
            // Index operations use virt_addr as index, no translation needed
            virt_addr
        };

        // For Index_Store_Tag, pass TagLo via phys_addr
        let op = cache_op & 0x1C;
        let phys_addr_or_taglo = if op == C_IST {
            self.core.cp0_taglo as u64
        } else {
            phys_addr
        };

        // Call unified cache interface
        let result = self.cache.cache_op(cache_op, virt_addr, phys_addr_or_taglo);

        // For Index_Load_Tag, update CP0 TagLo from result
        if op == C_ILT {
            self.core.cp0_taglo = result;
            self.core.cp0_taghi = 0;
        }

        self.handle_exec_complete()
    }


    // LL - Load Linked (32-bit)
    fn exec_ll(&mut self, d: &DecodedInstr) -> ExecStatus {
        let base = self.core.read_gpr(d.rs as u32);
        let virt_addr = base.wrapping_add(d.immu64());
        let rt_reg = d.rt as u32;

        match self.read_data::<4>(virt_addr) {
            Ok(value) => {
                // Sign-extend word to 64 bits
                self.core.write_gpr(rt_reg, value as i32 as i64 as u64);
                // Store physical address in LLAddr register
                // The LLAddr register stores bits 35..4 of the physical address
                let tr = self.nanotlb_translate::<{AccessType::Read as u8}>(virt_addr);
                if !tr.is_exception() {
                    let lladdr = (tr.phys >> 4) as u32;
                    self.cache.set_lladdr(lladdr);
                    self.core.cp0_lladdr = lladdr;
                    self.cache.set_llbit(true);
                }
                self.handle_exec_complete()
            }
            Err(status) => self.finish_status(status)
        }
    }

    // SC - Store Conditional (32-bit)
    fn exec_sc(&mut self, d: &DecodedInstr) -> ExecStatus {
        let base = self.core.read_gpr(d.rs as u32);
        let virt_addr = base.wrapping_add(d.immu64());
        let rt_reg = d.rt as u32;

        // Check the LLBit - if clear, the store fails immediately
        if !self.cache.get_llbit() {
            // Store failed, set rt to 0
            self.core.write_gpr(rt_reg, 0);
            return self.handle_exec_complete();
        }

        // Check if address matches the LL address
        let tr = self.nanotlb_translate::<{AccessType::Write as u8}>(virt_addr);
        if tr.is_exception() {
            self.cache.set_llbit(false);
            return self.finish_status(tr.status);
        }
        let phys_addr = tr.phys as u64;
        let ll_addr = (self.cache.get_lladdr() as u64) << 4;
        if (phys_addr & !0xF) == ll_addr {
            let value = self.core.read_gpr(rt_reg);
            let status = self.write_data::<4>(virt_addr, value);
            if status == EXEC_COMPLETE {
                self.core.write_gpr(rt_reg, 1);
                self.cache.set_llbit(false);
            }
            self.finish_status(status)
        } else {
            self.core.write_gpr(rt_reg, 0);
            self.cache.set_llbit(false);
            self.handle_exec_complete()
        }
    }

    // LLD - Load Linked Doubleword (64-bit)
    fn exec_lld(&mut self, d: &DecodedInstr) -> ExecStatus {
        let base = self.core.read_gpr(d.rs as u32);
        let virt_addr = base.wrapping_add(d.immu64());
        let rt_reg = d.rt as u32;

        match self.read_data::<8>(virt_addr) {
            Ok(value) => {
                self.core.write_gpr(rt_reg, value);
                // Store physical address in LLAddr register
                let tr = self.nanotlb_translate::<{AccessType::Read as u8}>(virt_addr);
                if !tr.is_exception() {
                    let lladdr = (tr.phys >> 4) as u32;
                    self.cache.set_lladdr(lladdr);
                    self.core.cp0_lladdr = lladdr;
                    self.cache.set_llbit(true);
                }
                self.handle_exec_complete()
            }
            Err(status) => self.finish_status(status)
        }
    }

    // SCD - Store Conditional Doubleword (64-bit)
    fn exec_scd(&mut self, d: &DecodedInstr) -> ExecStatus {
        let base = self.core.read_gpr(d.rs as u32);
        let virt_addr = base.wrapping_add(d.immu64());
        let rt_reg = d.rt as u32;

        // Check the LLBit - if clear, the store fails immediately
        if !self.cache.get_llbit() {
            // Store failed, set rt to 0
            self.core.write_gpr(rt_reg, 0);
            return self.handle_exec_complete();
        }

        // Check if address matches the LLD address
        let tr = self.nanotlb_translate::<{AccessType::Write as u8}>(virt_addr);
        if tr.is_exception() {
            self.cache.set_llbit(false);
            return self.finish_status(tr.status);
        }
        let phys_addr = tr.phys as u64;
        let ll_addr = (self.cache.get_lladdr() as u64) << 4;
        if (phys_addr & !0xF) == ll_addr {
            // Attempt the store
            let value = self.core.read_gpr(rt_reg);
            let status = self.write_data::<8>(virt_addr, value);
            if status == EXEC_COMPLETE {
                self.core.write_gpr(rt_reg, 1);
                self.cache.set_llbit(false);
            }
            self.finish_status(status)
        } else {
            // Store failed (address mismatch), set rt to 0 and clear LLBit
            self.core.write_gpr(rt_reg, 0);
            self.cache.set_llbit(false);
            self.handle_exec_complete()
        }
    }

    // PREF - Prefetch
    fn exec_pref(&mut self, _d: &DecodedInstr) -> ExecStatus {
        // Prefetch is a hint and can be implemented as a NOP
        // In a real implementation, this might trigger cache line fetches
        // For now, we just complete without doing anything
        self.handle_exec_complete()
    }

    // COP0 Instructions
    fn exec_cop0(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = d.rs as u32;

        match rs_val {
            RS_MFC0 => self.exec_mfc0(d),
            RS_DMFC0 => self.exec_dmfc0(d),
            RS_MTC0 => self.exec_mtc0(d),
            RS_DMTC0 => self.exec_dmtc0(d),
            RS_TLB => self.exec_tlb(d),
            RS_CFC0 | RS_CTC0 => {
                // CFC0/CTC0 are deprecated on R4000 - no separate control registers exist
                // All CP0 registers are accessed via MFC0/MTC0
                let s = self.reserved_instruction(d);
                self.handle_exception(s)
            }
            RS_BC0 => {
                // BC0 (Branch on CP0 condition) is not used on R4000
                let s = self.reserved_instruction(d);
                self.handle_exception(s)
            }
            _ => {
                let s = self.reserved_instruction(d);
                self.handle_exception(s)
            }
        }
    }

    // MFC0 - Move From CP0
    fn exec_mfc0(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rt_reg = d.rt as u32;
        let rd_val = d.rd as u32;
        let value = self.core.read_cp0(rd_val);
        // Sign-extend 32-bit value to 64 bits
        self.core.write_gpr(rt_reg, value as u32 as i32 as i64 as u64);
        self.handle_exec_complete()
    }

    // DMFC0 - Doubleword Move From CP0 (MIPS III)
    fn exec_dmfc0(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rt_reg = d.rt as u32;
        let rd_val = d.rd as u32;
        let value = self.core.read_cp0(rd_val);
        self.core.write_gpr(rt_reg, value);
        self.handle_exec_complete()
    }

    // MTC0 - Move To CP0
    fn exec_mtc0(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rt_val = self.core.read_gpr(d.rt as u32);
        let rd_val = d.rd as u32;
        // Sign-extend from 32 bits
        self.core.write_cp0(rd_val, rt_val as u32 as i32 as i64 as u64);
        self.handle_cp0_side_effects(rd_val);
        self.handle_exec_complete()
    }

    // DMTC0 - Doubleword Move To CP0 (MIPS III)
    fn exec_dmtc0(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rt_val = self.core.read_gpr(d.rt as u32);
        let rd_val = d.rd as u32;
        self.core.write_cp0(rd_val, rt_val);
        self.handle_cp0_side_effects(rd_val);
        self.handle_exec_complete()
    }

    fn handle_cp0_side_effects(&mut self, reg: u32) {
        // cheritest: a write to CP0 26 triggers the test device's dump. Routed
        // over the bus, so with no test device mapped it's an ignored GIO access.
        if reg == 26 && self.cheritest_dump_hook {
            self.sysad.write32(crate::testdev::TEST_DEV_BASE + crate::testdev::REG_DUMP,
                               self.core.cp0_ecc);
        }
        #[cfg(feature = "r5ksc_triton")]
        if reg == 16 {
            // CONFIG_SE (bit 12): wire L2 enable/disable to cache.
            let se = (self.core.cp0_config >> CONFIG_SE) & 1 != 0;
            self.cache.set_l2_enabled(se);
        }
    }

    // TLB Instructions
    fn exec_tlb(&mut self, d: &DecodedInstr) -> ExecStatus {
        let funct_val = d.funct as u32;

        match funct_val {
            // exec_tlbr/wi/wr/p always return bare EXEC_COMPLETE (no other
            // status possible), so run them for effect then finish here.
            FUNCT_TLBR => { self.exec_tlbr(); self.handle_exec_complete() }
            FUNCT_TLBWI => { self.exec_tlbwi(); self.handle_exec_complete() }
            FUNCT_TLBWR => { self.exec_tlbwr(); self.handle_exec_complete() }
            FUNCT_TLBP => { self.exec_tlbp(); self.handle_exec_complete() }
            FUNCT_ERET => self.exec_eret(), // PC set directly, terminal as-is
            FUNCT_WAIT => self.handle_exec_complete(), // phi opcode: invalid but not RI on R4000 (NOP)
            _ => {
                let s = self.reserved_instruction(d);
                self.handle_exception(s)
            }
        }
    }

    // TLBR - Read Indexed TLB Entry
    // Reads the TLB entry indexed by CP0.Index into CP0.EntryHi, CP0.EntryLo0, CP0.EntryLo1, and CP0.PageMask
    fn exec_tlbr(&mut self) -> ExecStatus {
        let index = (self.core.cp0_index as usize) % self.tlb.num_entries();
        let entry = self.tlb.read(index);

        // Per MIPS R4000 spec: Extract G bit from EntryHi bit 12 and populate both EntryLo G bits
        let g_bit = (entry.entry_hi >> 12) & 1;

        // Write to CP0 registers, clearing G bit from EntryHi
        self.core.cp0_entryhi = entry.entry_hi & !0x1000; // Clear bit 12 (G bit)
        self.core.cp0_entrylo0 = (entry.entry_lo[0] & !1) | g_bit; // Set G bit from EntryHi
        self.core.cp0_entrylo1 = (entry.entry_lo[1] & !1) | g_bit; // Set G bit from EntryHi
        self.core.cp0_pagemask = entry.page_mask;

        if mips_log(MIPS_LOG_TLB) { dlog_dev!(LogModule::Mips, "TLBR: Read Index {}\n{}", index, self.tlb.format_entry(index)); }

        EXEC_COMPLETE
    }

    // Helper function to construct a TLB entry from CP0 registers
    // Per MIPS R4000 spec: G bit is formed by ANDing G bits from EntryLo0 and EntryLo1
    // and stored in bit 12 of EntryHi in the TLB entry
    fn create_tlb_entry_from_cp0(&self) -> crate::mips_tlb::TlbEntry {
        use crate::mips_tlb::TlbEntry;

        let g0 = (self.core.cp0_entrylo0 & 1) != 0;
        let g1 = (self.core.cp0_entrylo1 & 1) != 0;
        let g_combined = if g0 && g1 { 1u64 << 12 } else { 0 };

        // EH_WM = 0xC000_00FF_FFFF_E0FF — same mask MAME applies on TLBWI (clears reserved bits).
        // Then clear bit 12 (G bit from EntryHi) and set combined G from Lo0&Lo1.
        const EH_WM: u64 = 0xC000_00FF_FFFF_E0FF;
        TlbEntry {
            page_mask: self.core.cp0_pagemask,
            entry_hi: (self.core.cp0_entryhi & EH_WM & !0x1000) | g_combined,
            entry_lo: [self.core.cp0_entrylo0, self.core.cp0_entrylo1],
            selector_bit_shift: 0, // all derived fields overwritten by MipsTlb::write()
            vcmp32: 0, vpn_hi32: 0, vcmp64: 0, vpn_hi64: 0,
            offset_mask: 0, pfn_base: [0; 2],
        }
    }

    // TLBWI - Write Indexed TLB Entry
    // Writes CP0.EntryHi, CP0.EntryLo0, CP0.EntryLo1, and CP0.PageMask to the TLB entry indexed by CP0.Index
    fn exec_tlbwi(&mut self) -> ExecStatus {
        let index = (self.core.cp0_index as usize) % self.tlb.num_entries();
        let entry = self.create_tlb_entry_from_cp0();
        //eprintln!("TLBWI idx={} entryhi={:#018x} lo0={:#018x} lo1={:#018x} pc={:#018x}", index, entry.entry_hi, entry.entry_lo[0], entry.entry_lo[1], self.core.pc);
        self.tlb.write(index, entry);
        self.nanotlb_invalidate();

        if mips_log(MIPS_LOG_TLB) { dlog_dev!(LogModule::Mips, "TLBWI: Write Index {}\n{}", index, self.tlb.format_entry(index)); }

        EXEC_COMPLETE
    }

    // TLBWR - Write Random TLB Entry
    // Writes CP0.EntryHi, CP0.EntryLo0, CP0.EntryLo1, and CP0.PageMask to a random TLB entry
    // The random index is determined by CP0.Random register
    fn exec_tlbwr(&mut self) -> ExecStatus {
        self.core.update_random();
        let index = (self.core.cp0_random as usize) % self.tlb.num_entries();
        let entry = self.create_tlb_entry_from_cp0();
        self.tlb.write(index, entry);
        self.nanotlb_invalidate();

        if mips_log(MIPS_LOG_TLB) { dlog_dev!(LogModule::Mips, "TLBWR: Write Random Index {}\n{}", index, self.tlb.format_entry(index)); }

        EXEC_COMPLETE
    }

    // TLBP - Probe TLB for Matching Entry
    // Searches the TLB for an entry matching CP0.EntryHi and sets CP0.Index to the matching entry's index
    // If no match is found, sets the high bit (P bit) of CP0.Index
    fn exec_tlbp(&mut self) -> ExecStatus {
        let virt_addr = (self.core.cp0_entryhi as u64) & !0xFF; // VPN2 portion (bits 63:13)
        let asid = (self.core.cp0_entryhi & 0xFF) as u8;
        let xtlb = self.is_xtlb_address(virt_addr);

        let result = self.tlb.probe(virt_addr, asid, xtlb);
        self.core.cp0_index = result;

        if (result & 0x80000000) != 0 {
            if mips_log(MIPS_LOG_TLB) { dlog_dev!(LogModule::Mips, "TLBP: Probe VPN2={:07x} ASID={:02x} -> Miss", virt_addr >> 13, asid); }
        } else {
            if mips_log(MIPS_LOG_TLB) { dlog_dev!(LogModule::Mips, "TLBP: Probe VPN2={:07x} ASID={:02x} -> Hit Index {}", virt_addr >> 13, asid, result); }
        }

        EXEC_COMPLETE
    }

    // ERET - Exception Return
    // Returns from exception by restoring PC from EPC or ErrorEPC and clearing exception status
    // Note: ERET does NOT have a delay slot in MIPS III+
    fn exec_eret(&mut self) -> ExecStatus {
        let target = if (self.core.cp0_status & STATUS_ERL) != 0 {
            // Error level - return to ErrorEPC
            self.core.cp0_status &= !STATUS_ERL;
            self.core.cp0_errorepc
        } else {
            // Exception level - return to EPC
            self.core.cp0_status &= !STATUS_EXL;
            self.core.cp0_epc
        };

        // Clear LLbit (Load Linked bit) on ERET
        // This is implementation-specific but commonly done
        self.cache.set_llbit(false);
        self.nanotlb_invalidate();

        // ERET jumps immediately without delay slot
        self.core.pc = target;

        // PC already set — exec_complete_pc_set marks the target as a
        // compile-worthy arrival.
        self.exec_complete_pc_set()
    }

    // ===== COP1 (FPU) Instructions =====

    /// Set Cause.CE to the given coprocessor number and dispatch EXC_CPU.
    /// Always used as an immediate `return self.cpu_unusable(..)` from a
    /// dispatch-target handler, so it's the terminal action itself.
    #[inline]
    fn cpu_unusable(&mut self, ce: u32) -> ExecStatus {
        self.core.cp0_cause = (self.core.cp0_cause & !CAUSE_CE_MASK) | ((ce & 3) << CAUSE_CE_SHIFT);
        let s = exec_exception(EXC_CPU);
        self.handle_exception(s)
    }

    /// After a FPU arithmetic op: read host exception flags, update FCSR cause+flag bits,
    /// and raise EXC_FPE if any enabled exception fired, otherwise EXEC_COMPLETE.
    /// Must be called after the result is written; host FP flags are cleared by this call.
    #[inline]
    /// Always used as the terminal tail call of an FPU arithmetic exec_*
    /// handler (`self.fpu_update_fcsr()`, nothing runs after), so this is the
    /// terminal action itself: completes normally or dispatches EXC_FPE.
    fn fpu_update_fcsr(&mut self) -> ExecStatus {
        self.fpu_update_fcsr_with_inexact_override(None)
    }

    /// `inexact_override`: when `Some(bool)`, replaces whatever the host
    /// FPU's own Inexact sticky bit (bit 2 of the [6:2] V,Z,O,U,I encoding
    /// `platform::get_fpu_status()` returns) says with this value instead of
    /// trusting it — see `exec_fround_l_s`'s doc comment for why
    /// ROUND/TRUNC/CEIL/FLOOR/CVT.W/CVT.L need this: host hardware state
    /// can't answer "was this MIPS conversion inexact" for them at all (the
    /// two-step `.round() as i32`/etc implementation means the host's own
    /// Inexact bit, if set, reflects something that isn't what MIPS
    /// specifies — see that comment for the full reasoning), so those
    /// callers compute the real answer themselves (comparing the converted-
    /// back-to-float result against the original source value) and this
    /// replaces the host's bit with it, rather than merging the two (an OR
    /// would still let a spurious host-set bit through if the computed
    /// answer were `false` but the host happened to report `true` anyway).
    fn fpu_update_fcsr_with_inexact_override(&mut self, inexact_override: Option<bool>) -> ExecStatus {
        let mut flags = crate::platform::get_fpu_status(); // bits [6:2]: FV,FZ,FO,FU,FI
        crate::platform::clear_fpu_status();
        if let Some(inexact) = inexact_override {
            flags = (flags & !(1 << 2)) | ((inexact as u32) << 2);
        }
        if flags == 0 {
            return self.handle_exec_complete();
        }
        // Promote flag bits [6:2] → cause bits [16:12] (shift up by 10)
        let causes = (flags & FCSR_FM) << 10;
        // OR causes and sticky flags into FCSR (software clears explicitly via SetFPSR)
        self.core.fpu_fcsr |= causes;
        self.core.fpu_fcsr |= flags & FCSR_FM;
        // If underflow occurred and underflow trapping is enabled, set CE (unimplemented)
        // to match real R4400 hardware behavior (hardware punts to software on underflow trap)
        if (causes & FCSR_CU) != 0 && (self.core.fpu_fcsr & 0x100) != 0 {
            self.core.fpu_fcsr |= FCSR_CE;
            let s = exec_exception(EXC_FPE);
            return self.handle_exception(s);
        }
        // Raise FPE if any cause bit has its corresponding enable bit set
        // Causes are 5 bits above enables: (causes >> 5) aligns them with enables
        if ((causes >> 5) & (self.core.fpu_fcsr & FCSR_EM)) != 0 {
            let s = exec_exception(EXC_FPE);
            return self.handle_exception(s);
        }
        self.handle_exec_complete()
    }

    // MFC1 - Move Word From FPU
    fn exec_mfc1(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let rt_reg = d.rt as u32;
        let fs_reg = d.rd as u32;
        let value = (self.fpr_read_w)(&self.core, fs_reg) as i32 as i64 as u64;
        self.core.write_gpr(rt_reg, value);
        self.handle_exec_complete()
    }

    // DMFC1 - Move Doubleword From FPU (MIPS III)
    fn exec_dmfc1(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let rt_reg = d.rt as u32;
        let fs_reg = d.rd as u32;
        let value = (self.fpr_read_l)(&self.core, fs_reg);
        self.core.write_gpr(rt_reg, value);
        self.handle_exec_complete()
    }

    // CFC1 - Move Control Word From FPU
    fn exec_cfc1(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let rt_reg = d.rt as u32;
        let fs_reg = d.rd as u32;
        let value = self.core.read_fpu_control(fs_reg);
        self.core.write_gpr(rt_reg, value as i32 as i64 as u64);
        self.handle_exec_complete()
    }

    // MTC1 - Move Word To FPU
    fn exec_mtc1(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let rt_val = self.core.read_gpr(d.rt as u32) as u32;
        let fs_reg = d.rd as u32;
        (self.fpr_write_w)(&mut self.core, fs_reg, rt_val);
        self.handle_exec_complete()
    }

    // DMTC1 - Move Doubleword To FPU (MIPS III)
    fn exec_dmtc1(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let rt_val = self.core.read_gpr(d.rt as u32);
        let fs_reg = d.rd as u32;
        (self.fpr_write_l)(&mut self.core, fs_reg, rt_val);
        self.handle_exec_complete()
    }

    // CTC1 - Move Control Word To FPU
    fn exec_ctc1(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let rt_val = self.core.read_gpr(d.rt as u32) as u32;
        let fs_reg = d.rd as u32;
        self.core.write_fpu_control(fs_reg, rt_val);
        // After writing FCSR, check if pending cause bits match enabled bits → FPE
        if fs_reg == 31 {
            let fcsr = self.core.fpu_fcsr;
            if (fcsr & FCSR_CE) != 0 || (((fcsr & FCSR_CM) >> 5) & (fcsr & FCSR_EM)) != 0 {
                let s = exec_exception(EXC_FPE);
                return self.handle_exception(s);
            }
        }
        self.handle_exec_complete()
    }

    // BC1 - Branch on FPU Condition Code
    fn exec_bc1(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let rt_val = d.rt as u32;
        let target = self.core.pc.wrapping_add(4).wrapping_add(d.immu64());
        let branch_if_true = (rt_val & 1) != 0;
        let likely = (rt_val & 2) != 0;
        let cc_field = (d.raw >> 18) & 0x7;
        let cc = self.core.get_fpu_cc(cc_field);
        let condition = if branch_if_true { cc } else { !cc };
        if condition {
            self.branch_delay(target)
        } else if likely {
            self.handle_branch_likely_skip()
        } else {
            self.handle_branch_not_taken()
        }
    }

    // ===== COP1 S-format (Single-precision) =====

    fn exec_fadd_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let ft_reg = d.rt as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let result = f32::from_bits((self.fpr_read_w)(&self.core, fs_reg)) + f32::from_bits((self.fpr_read_w)(&self.core, ft_reg));
        (self.fpr_write_w)(&mut self.core, fd_reg, (result).to_bits());
        self.fpu_update_fcsr()
    }
    fn exec_fsub_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let ft_reg = d.rt as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let result = f32::from_bits((self.fpr_read_w)(&self.core, fs_reg)) - f32::from_bits((self.fpr_read_w)(&self.core, ft_reg));
        (self.fpr_write_w)(&mut self.core, fd_reg, (result).to_bits());
        self.fpu_update_fcsr()
    }
    fn exec_fmul_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let ft_reg = d.rt as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let result = f32::from_bits((self.fpr_read_w)(&self.core, fs_reg)) * f32::from_bits((self.fpr_read_w)(&self.core, ft_reg));
        (self.fpr_write_w)(&mut self.core, fd_reg, (result).to_bits());
        self.fpu_update_fcsr()
    }
    fn exec_fdiv_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let ft_reg = d.rt as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let fs_val = f32::from_bits((self.fpr_read_w)(&self.core, fs_reg));
        let ft_val = f32::from_bits((self.fpr_read_w)(&self.core, ft_reg));
        let result = fs_val / ft_val;
        (self.fpr_write_w)(&mut self.core, fd_reg, (result).to_bits());
        if mips_log(MIPS_LOG_FPU) {
            dlog_dev!(LogModule::Mips, "FPU div.s PC={:016x} {} / {} = {}", self.core.pc, fs_val, ft_val, result);
        }
        self.fpu_update_fcsr()
    }
    fn exec_fsqrt_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let result = f32::from_bits((self.fpr_read_w)(&self.core, fs_reg)).sqrt();
        (self.fpr_write_w)(&mut self.core, fd_reg, (result).to_bits());
        self.fpu_update_fcsr()
    }
    fn exec_fabs_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        let result = f32::from_bits((self.fpr_read_w)(&self.core, fs_reg)).abs();
        (self.fpr_write_w)(&mut self.core, fd_reg, (result).to_bits());
        self.handle_exec_complete()
    }
    fn exec_fmov_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        let value = (self.fpr_read_l)(&self.core, fs_reg);
        (self.fpr_write_l)(&mut self.core, fd_reg, value);
        self.handle_exec_complete()
    }
    fn exec_fneg_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        let result = -f32::from_bits((self.fpr_read_w)(&self.core, fs_reg));
        (self.fpr_write_w)(&mut self.core, fd_reg, (result).to_bits());
        self.handle_exec_complete()
    }
    /// Every ROUND/TRUNC/CEIL/FLOOR/CVT.W/CVT.L handler in this file (all 20:
    /// S/D source × W/L dest × the four rounding modes, plus plain CVT) needs
    /// its FCSR Inexact bit computed *by value*, not read off host hardware
    /// state: `f32`/`f64`'s rounding intrinsics (`.round()`/`.trunc()`/
    /// `.ceil()`/`.floor()`) compile to real SSE `ROUNDSS`/`ROUNDSD`
    /// instructions on x86-64, which set MXCSR's Precision (Inexact) sticky
    /// flag as a side effect whenever *that intermediate rounding step*
    /// wasn't a no-op — not whenever the *overall* MIPS conversion
    /// (original source float -> final integer) was inexact, which is what
    /// the architecture actually specifies. The two aren't the same thing:
    /// `ROUND.W.S` on an already-integer value like `79.0` triggers the
    /// host's Precision flag for the intermediate `.round()` call in some
    /// cases (this session's live-boot lockstep divergence on exactly this
    /// case is what surfaced it) while the true MIPS answer is "not
    /// inexact"; conversely `TRUNC.W.S` on `3.7` needs Inexact set even
    /// though the *host* truncate instruction it corresponds to may not set
    /// anything by the time our two-step implementation reads it (the
    /// following `as i32` is a pure software cast, not a hardware
    /// instruction MXCSR observes at all). MAME's r4000.cpp reference
    /// (`ignore/docs/r4000.cpp`) sidesteps this class of bug entirely by
    /// using SoftFloat's `f32_to_i32(fs, roundingMode, true)` — one atomic,
    /// exactly-specified conversion whose own `softfloat_flag_inexact` is
    /// defined precisely as "converted-back result differs from the
    /// original source" — which is exactly the comparison these handlers
    /// now do by hand: convert `result` back to a float and compare against
    /// `fs_val`/`fs_reg`'s original value, then pass that as
    /// `fpu_update_fcsr_with_inexact_override`'s override instead of
    /// trusting whatever the host's Precision bit happens to read. Every
    /// *other* flag (Invalid, DivByZero, Overflow, Underflow) still comes
    /// from the real host read, untouched — only Inexact needed replacing.
    ///
    /// A second, independent problem: on this build/target, `f32`/`f64`'s
    /// `.round()`/`.trunc()`/`.ceil()`/`.floor()` were *empirically* found to
    /// be sensitive to the host's MXCSR rounding-control bits (confirmed via
    /// a live-boot lockstep divergence: `CVT.W.D` on `65535.5` returned
    /// `65535` — round-toward-zero — instead of the documented
    /// round-half-away-from-zero `65536`, exactly matching FCSR.RM=1/RZ that
    /// happened to be live at the time), even though every one of these
    /// handlers calls `.round()` unconditionally, never consulting FCSR.RM
    /// itself. This contradicts the SDM's documented immediate-mode encoding
    /// for the `ROUNDSD`/`ROUNDSS` instructions rustc emits for `.round()`
    /// (disassembly shows a fixed, non-MXCSR-select immediate), so the exact
    /// mechanism is unconfirmed — but the observed behavior is 100%
    /// reproducible, so it can't be trusted regardless of root cause. Fixed
    /// by routing every one of these 20 handlers through
    /// `round_f32_to_int_mode`/`round_f64_to_int_mode` (below) instead: a
    /// pure bit-manipulation implementation with no hardware rounding
    /// instruction anywhere in it, so it can't be sensitive to host FPU
    /// control state by construction — also portable to aarch64 (macOS) for
    /// free, where the host FPCR's rounding-control field isn't even synced
    /// from FCSR today (`platform::set_fpu_mode` only exists for x86).
    /// ROUND/TRUNC/CEIL/FLOOR pass their fixed mode; the two plain-CVT
    /// handlers (`exec_fcvt_w_s`/`_d`, `exec_fcvt_l_s`/`_d`) now honor
    /// FCSR.RM dynamically instead of hardcoding round-half-away-from-zero —
    /// this was a separate, pre-existing, already-documented spec gap
    /// ([[project_fpu_rounding_spec_gap]]) closed in the same pass since
    /// it's the same "don't trust ad-hoc host rounding" root cause.
    fn exec_fround_l_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let fs_val = f32::from_bits((self.fpr_read_w)(&self.core, fs_reg));
        let result = round_f32_to_int_mode(fs_val, RM_NEAREST_EVEN) as i64;
        (self.fpr_write_l)(&mut self.core, fd_reg, result as u64);
        self.fpu_update_fcsr_with_inexact_override(Some(result as f32 != fs_val))
    }
    fn exec_ftrunc_l_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let fs_val = f32::from_bits((self.fpr_read_w)(&self.core, fs_reg));
        let result = round_f32_to_int_mode(fs_val, RM_TOWARD_ZERO) as i64;
        (self.fpr_write_l)(&mut self.core, fd_reg, result as u64);
        self.fpu_update_fcsr_with_inexact_override(Some(result as f32 != fs_val))
    }
    fn exec_fceil_l_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let fs_val = f32::from_bits((self.fpr_read_w)(&self.core, fs_reg));
        let result = round_f32_to_int_mode(fs_val, RM_TOWARD_POS_INF) as i64;
        (self.fpr_write_l)(&mut self.core, fd_reg, result as u64);
        self.fpu_update_fcsr_with_inexact_override(Some(result as f32 != fs_val))
    }
    fn exec_ffloor_l_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let fs_val = f32::from_bits((self.fpr_read_w)(&self.core, fs_reg));
        let result = round_f32_to_int_mode(fs_val, RM_TOWARD_NEG_INF) as i64;
        (self.fpr_write_l)(&mut self.core, fd_reg, result as u64);
        self.fpu_update_fcsr_with_inexact_override(Some(result as f32 != fs_val))
    }
    fn exec_fround_w_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let fs_val = f32::from_bits((self.fpr_read_w)(&self.core, fs_reg));
        let result = round_f32_to_int_mode(fs_val, RM_NEAREST_EVEN) as i32;
        (self.fpr_write_w)(&mut self.core, fd_reg, result as u32);
        self.fpu_update_fcsr_with_inexact_override(Some(result as f32 != fs_val))
    }
    fn exec_ftrunc_w_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let fs_val = f32::from_bits((self.fpr_read_w)(&self.core, fs_reg));
        let result = round_f32_to_int_mode(fs_val, RM_TOWARD_ZERO) as i32;
        (self.fpr_write_w)(&mut self.core, fd_reg, result as u32);
        if mips_log(MIPS_LOG_FPU) {
            dlog_dev!(LogModule::Mips, "FPU trunc.w.s PC={:016x} {} -> {}", self.core.pc, fs_val, result);
        }
        self.fpu_update_fcsr_with_inexact_override(Some(result as f32 != fs_val))
    }
    fn exec_fceil_w_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let fs_val = f32::from_bits((self.fpr_read_w)(&self.core, fs_reg));
        let result = round_f32_to_int_mode(fs_val, RM_TOWARD_POS_INF) as i32;
        (self.fpr_write_w)(&mut self.core, fd_reg, result as u32);
        self.fpu_update_fcsr_with_inexact_override(Some(result as f32 != fs_val))
    }
    fn exec_ffloor_w_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let fs_val = f32::from_bits((self.fpr_read_w)(&self.core, fs_reg));
        let result = round_f32_to_int_mode(fs_val, RM_TOWARD_NEG_INF) as i32;
        (self.fpr_write_w)(&mut self.core, fd_reg, result as u32);
        self.fpu_update_fcsr_with_inexact_override(Some(result as f32 != fs_val))
    }
    fn exec_fmovcf_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        let cc = (d.raw >> 18) & 0x7;
        let tf = ((d.raw >> 16) & 0x1) != 0;
        let cc_value = self.core.get_fpu_cc(cc);
        let taken = cc_value == tf;
        if taken {
            let val = f32::from_bits((self.fpr_read_w)(&self.core, fs_reg));
            (self.fpr_write_w)(&mut self.core, fd_reg, (val).to_bits());
        }
        if mips_log(MIPS_LOG_FPU) {
            dlog_dev!(LogModule::Mips, "FPU fmov{}.s PC={:016x} cc{}={} taken={}",
                if tf { "t" } else { "f" }, self.core.pc, cc, cc_value, taken);
        }
        self.handle_exec_complete()
    }
    fn exec_fmovz_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let ft_reg = d.rt as u32; let fd_reg = d.sa as u32;
        if self.core.read_gpr(ft_reg) == 0 {
            let val = f32::from_bits((self.fpr_read_w)(&self.core, fs_reg));
            (self.fpr_write_w)(&mut self.core, fd_reg, (val).to_bits());
        }
        self.handle_exec_complete()
    }
    fn exec_fmovn_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let ft_reg = d.rt as u32; let fd_reg = d.sa as u32;
        if self.core.read_gpr(ft_reg) != 0 {
            let val = f32::from_bits((self.fpr_read_w)(&self.core, fs_reg));
            (self.fpr_write_w)(&mut self.core, fd_reg, (val).to_bits());
        }
        self.handle_exec_complete()
    }
    fn exec_frecip_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let fs_val = f32::from_bits((self.fpr_read_w)(&self.core, fs_reg));
        let result = 1.0 / fs_val;
        (self.fpr_write_w)(&mut self.core, fd_reg, (result).to_bits());
        if mips_log(MIPS_LOG_FPU) {
            dlog_dev!(LogModule::Mips, "FPU recip.s PC={:016x} 1 / {} = {}", self.core.pc, fs_val, result);
        }
        self.fpu_update_fcsr()
    }
    fn exec_frsqrt_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let result = 1.0 / f32::from_bits((self.fpr_read_w)(&self.core, fs_reg)).sqrt();
        (self.fpr_write_w)(&mut self.core, fd_reg, (result).to_bits());
        self.fpu_update_fcsr()
    }
    fn exec_fcvt_d_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let result = f32::from_bits((self.fpr_read_w)(&self.core, fs_reg)) as f64;
        (self.fpr_write_d)(&mut self.core, fd_reg, result);
        self.fpu_update_fcsr()
    }
    fn exec_fcvt_w_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let fs_val = f32::from_bits((self.fpr_read_w)(&self.core, fs_reg));
        let result = round_f32_to_int_mode(fs_val, (self.core.fpu_fcsr & 0x3) as u8) as i32;
        (self.fpr_write_w)(&mut self.core, fd_reg, result as u32);
        if mips_log(MIPS_LOG_FPU) {
            dlog_dev!(LogModule::Mips, "FPU cvt.w.s PC={:016x} {} -> {}", self.core.pc, fs_val, result);
        }
        self.fpu_update_fcsr_with_inexact_override(Some(result as f32 != fs_val))
    }
    fn exec_fcvt_l_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let fs_val = f32::from_bits((self.fpr_read_w)(&self.core, fs_reg));
        let result = round_f32_to_int_mode(fs_val, (self.core.fpu_fcsr & 0x3) as u8) as i64;
        (self.fpr_write_l)(&mut self.core, fd_reg, result as u64);
        self.fpu_update_fcsr_with_inexact_override(Some(result as f32 != fs_val))
    }
    // S-format compare (all 16 conditions share one handler; funct 0x30-0x3F)
    fn exec_fcc_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let ft_reg = d.rt as u32; let fd_reg = d.sa as u32;
        let funct_val = d.funct as u32;
        let fs_val = f32::from_bits((self.fpr_read_w)(&self.core, fs_reg));
        let ft_val = f32::from_bits((self.fpr_read_w)(&self.core, ft_reg));
        // Signaling comparisons (cond 0x8–0xF) raise V (invalid) if operands are unordered (NaN)
        if (funct_val & 0x8) != 0 && (fs_val.is_nan() || ft_val.is_nan()) {
            self.core.fpu_fcsr |= FCSR_CV | 0x40; // set Cause V + Flag V
            if (self.core.fpu_fcsr & 0x800) != 0 { // EV enable bit
                let s = exec_exception(EXC_FPE);
                return self.handle_exception(s);
            }
        }
        let cond = self.fpu_compare_s(fs_val, ft_val, funct_val);
        let cc = fd_reg & 0x7;
        self.core.set_fpu_cc(cc, cond);
        if mips_log(MIPS_LOG_FPU) {
            dlog_dev!(LogModule::Mips, "FPU c.{:x}.s PC={:016x} cc{}={} fs={} ft={}",
                funct_val, self.core.pc, cc, cond, fs_val, ft_val);
        }
        self.handle_exec_complete()
    }

    // ===== COP1 D-format (Double-precision) =====

    fn exec_fadd_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let ft_reg = d.rt as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let read_d = self.fpr_read_d; let write_d = self.fpr_write_d;
        let result = read_d(&self.core, fs_reg) + read_d(&self.core, ft_reg);
        write_d(&mut self.core, fd_reg, result);
        self.fpu_update_fcsr()
    }
    fn exec_fsub_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let ft_reg = d.rt as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let read_d = self.fpr_read_d; let write_d = self.fpr_write_d;
        let result = read_d(&self.core, fs_reg) - read_d(&self.core, ft_reg);
        write_d(&mut self.core, fd_reg, result);
        self.fpu_update_fcsr()
    }
    fn exec_fmul_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let ft_reg = d.rt as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let read_d = self.fpr_read_d; let write_d = self.fpr_write_d;
        let result = read_d(&self.core, fs_reg) * read_d(&self.core, ft_reg);
        write_d(&mut self.core, fd_reg, result);
        self.fpu_update_fcsr()
    }
    fn exec_fdiv_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let ft_reg = d.rt as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let read_d = self.fpr_read_d; let write_d = self.fpr_write_d;
        let fs_val = read_d(&self.core, fs_reg);
        let ft_val = read_d(&self.core, ft_reg);
        let result = fs_val / ft_val;
        write_d(&mut self.core, fd_reg, result);
        if mips_log(MIPS_LOG_FPU) {
            dlog_dev!(LogModule::Mips, "FPU div.d PC={:016x} {} / {} = {}", self.core.pc, fs_val, ft_val, result);
        }
        self.fpu_update_fcsr()
    }
    fn exec_fsqrt_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let read_d = self.fpr_read_d; let write_d = self.fpr_write_d;
        let result = read_d(&self.core, fs_reg).sqrt();
        write_d(&mut self.core, fd_reg, result);
        self.fpu_update_fcsr()
    }
    fn exec_fabs_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        let read_d = self.fpr_read_d; let write_d = self.fpr_write_d;
        let result = read_d(&self.core, fs_reg).abs();
        write_d(&mut self.core, fd_reg, result);
        self.handle_exec_complete()
    }
    fn exec_fmov_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        let value = (self.fpr_read_l)(&self.core, fs_reg);
        (self.fpr_write_l)(&mut self.core, fd_reg, value);
        self.handle_exec_complete()
    }
    fn exec_fneg_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        let read_d = self.fpr_read_d; let write_d = self.fpr_write_d;
        let result = -read_d(&self.core, fs_reg);
        write_d(&mut self.core, fd_reg, result);
        self.handle_exec_complete()
    }
    fn exec_fround_l_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let fs_val = (self.fpr_read_d)(&self.core, fs_reg);
        let result = round_f64_to_int_mode(fs_val, RM_NEAREST_EVEN) as i64;
        (self.fpr_write_l)(&mut self.core, fd_reg, result as u64);
        self.fpu_update_fcsr_with_inexact_override(Some(result as f64 != fs_val))
    }
    fn exec_ftrunc_l_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let fs_val = (self.fpr_read_d)(&self.core, fs_reg);
        let result = round_f64_to_int_mode(fs_val, RM_TOWARD_ZERO) as i64;
        (self.fpr_write_l)(&mut self.core, fd_reg, result as u64);
        self.fpu_update_fcsr_with_inexact_override(Some(result as f64 != fs_val))
    }
    fn exec_fceil_l_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let fs_val = (self.fpr_read_d)(&self.core, fs_reg);
        let result = round_f64_to_int_mode(fs_val, RM_TOWARD_POS_INF) as i64;
        (self.fpr_write_l)(&mut self.core, fd_reg, result as u64);
        self.fpu_update_fcsr_with_inexact_override(Some(result as f64 != fs_val))
    }
    fn exec_ffloor_l_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let fs_val = (self.fpr_read_d)(&self.core, fs_reg);
        let result = round_f64_to_int_mode(fs_val, RM_TOWARD_NEG_INF) as i64;
        (self.fpr_write_l)(&mut self.core, fd_reg, result as u64);
        self.fpu_update_fcsr_with_inexact_override(Some(result as f64 != fs_val))
    }
    fn exec_fround_w_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let fs_val = (self.fpr_read_d)(&self.core, fs_reg);
        let result = round_f64_to_int_mode(fs_val, RM_NEAREST_EVEN) as i32;
        (self.fpr_write_w)(&mut self.core, fd_reg, result as u32);
        self.fpu_update_fcsr_with_inexact_override(Some(result as f64 != fs_val))
    }
    fn exec_ftrunc_w_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let fs_val = (self.fpr_read_d)(&self.core, fs_reg);
        let result = round_f64_to_int_mode(fs_val, RM_TOWARD_ZERO) as i32;
        (self.fpr_write_w)(&mut self.core, fd_reg, result as u32);
        if mips_log(MIPS_LOG_FPU) {
            dlog_dev!(LogModule::Mips, "FPU trunc.w.d PC={:016x} {} -> {}", self.core.pc, fs_val, result);
        }
        self.fpu_update_fcsr_with_inexact_override(Some(result as f64 != fs_val))
    }
    fn exec_fceil_w_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let fs_val = (self.fpr_read_d)(&self.core, fs_reg);
        let result = round_f64_to_int_mode(fs_val, RM_TOWARD_POS_INF) as i32;
        (self.fpr_write_w)(&mut self.core, fd_reg, result as u32);
        self.fpu_update_fcsr_with_inexact_override(Some(result as f64 != fs_val))
    }
    fn exec_ffloor_w_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let fs_val = (self.fpr_read_d)(&self.core, fs_reg);
        let result = round_f64_to_int_mode(fs_val, RM_TOWARD_NEG_INF) as i32;
        (self.fpr_write_w)(&mut self.core, fd_reg, result as u32);
        self.fpu_update_fcsr_with_inexact_override(Some(result as f64 != fs_val))
    }
    fn exec_fmovcf_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        let cc = (d.raw >> 18) & 0x7;
        let tf = ((d.raw >> 16) & 0x1) != 0;
        let cc_value = self.core.get_fpu_cc(cc);
        let taken = cc_value == tf;
        if taken {
            let read_d = self.fpr_read_d; let write_d = self.fpr_write_d;
            let val = read_d(&self.core, fs_reg);
            write_d(&mut self.core, fd_reg, val);
        }
        if mips_log(MIPS_LOG_FPU) {
            dlog_dev!(LogModule::Mips, "FPU fmov{}.d PC={:016x} cc{}={} taken={}",
                if tf { "t" } else { "f" }, self.core.pc, cc, cc_value, taken);
        }
        self.handle_exec_complete()
    }
    fn exec_fmovz_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let ft_reg = d.rt as u32; let fd_reg = d.sa as u32;
        if self.core.read_gpr(ft_reg) == 0 {
            let read_d = self.fpr_read_d; let write_d = self.fpr_write_d;
            let val = read_d(&self.core, fs_reg);
            write_d(&mut self.core, fd_reg, val);
        }
        self.handle_exec_complete()
    }
    fn exec_fmovn_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let ft_reg = d.rt as u32; let fd_reg = d.sa as u32;
        if self.core.read_gpr(ft_reg) != 0 {
            let read_d = self.fpr_read_d; let write_d = self.fpr_write_d;
            let val = read_d(&self.core, fs_reg);
            write_d(&mut self.core, fd_reg, val);
        }
        self.handle_exec_complete()
    }
    fn exec_frecip_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let read_d = self.fpr_read_d; let write_d = self.fpr_write_d;
        let fs_val = read_d(&self.core, fs_reg);
        let result = 1.0 / fs_val;
        write_d(&mut self.core, fd_reg, result);
        if mips_log(MIPS_LOG_FPU) {
            dlog_dev!(LogModule::Mips, "FPU recip.d PC={:016x} 1 / {} = {}", self.core.pc, fs_val, result);
        }
        self.fpu_update_fcsr()
    }
    fn exec_frsqrt_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let read_d = self.fpr_read_d; let write_d = self.fpr_write_d;
        let result = 1.0 / read_d(&self.core, fs_reg).sqrt();
        write_d(&mut self.core, fd_reg, result);
        self.fpu_update_fcsr()
    }
    fn exec_fcvt_s_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let result = (self.fpr_read_d)(&self.core, fs_reg) as f32;
        (self.fpr_write_w)(&mut self.core, fd_reg, (result).to_bits());
        self.fpu_update_fcsr()
    }
    fn exec_fcvt_w_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let fs_val = (self.fpr_read_d)(&self.core, fs_reg);
        let result = round_f64_to_int_mode(fs_val, (self.core.fpu_fcsr & 0x3) as u8) as i32;
        (self.fpr_write_w)(&mut self.core, fd_reg, result as u32);
        if mips_log(MIPS_LOG_FPU) {
            dlog_dev!(LogModule::Mips, "FPU cvt.w.d PC={:016x} {} -> {}", self.core.pc, fs_val, result);
        }
        self.fpu_update_fcsr_with_inexact_override(Some(result as f64 != fs_val))
    }
    fn exec_fcvt_l_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let fs_val = (self.fpr_read_d)(&self.core, fs_reg);
        let result = round_f64_to_int_mode(fs_val, (self.core.fpu_fcsr & 0x3) as u8) as i64;
        (self.fpr_write_l)(&mut self.core, fd_reg, result as u64);
        self.fpu_update_fcsr_with_inexact_override(Some(result as f64 != fs_val))
    }
    // D-format compare (all 16 conditions share one handler; funct 0x30-0x3F)
    fn exec_fcc_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let ft_reg = d.rt as u32; let fd_reg = d.sa as u32;
        let funct_val = d.funct as u32;
        let read_d = self.fpr_read_d;
        let fs_val = read_d(&self.core, fs_reg);
        let ft_val = read_d(&self.core, ft_reg);
        // Signaling comparisons (cond 0x8–0xF) raise V (invalid) if operands are unordered (NaN)
        if (funct_val & 0x8) != 0 && (fs_val.is_nan() || ft_val.is_nan()) {
            self.core.fpu_fcsr |= FCSR_CV | 0x40; // set Cause V + Flag V
            if (self.core.fpu_fcsr & 0x800) != 0 { // EV enable bit
                let s = exec_exception(EXC_FPE);
                return self.handle_exception(s);
            }
        }
        let cond = self.fpu_compare_d(fs_val, ft_val, funct_val);
        let cc = fd_reg & 0x7;
        self.core.set_fpu_cc(cc, cond);
        if mips_log(MIPS_LOG_FPU) {
            dlog_dev!(LogModule::Mips, "FPU c.{:x}.d PC={:016x} cc{}={} fs={} ft={}",
                funct_val, self.core.pc, cc, cond, fs_val, ft_val);
        }
        self.handle_exec_complete()
    }

    // ===== COP1 W-format (Word fixed-point → float) =====

    fn exec_fcvt_s_w(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let result = (self.fpr_read_w)(&self.core, fs_reg) as i32 as f32;
        (self.fpr_write_w)(&mut self.core, fd_reg, (result).to_bits());
        self.fpu_update_fcsr()
    }
    fn exec_fcvt_d_w(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let result = (self.fpr_read_w)(&self.core, fs_reg) as i32 as f64;
        (self.fpr_write_d)(&mut self.core, fd_reg, result);
        self.fpu_update_fcsr()
    }

    // ===== COP1 L-format (Long fixed-point → float, MIPS III) =====

    fn exec_fcvt_s_l(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let result = (self.fpr_read_l)(&self.core, fs_reg) as i64 as f32;
        (self.fpr_write_w)(&mut self.core, fd_reg, (result).to_bits());
        self.fpu_update_fcsr()
    }
    fn exec_fcvt_d_l(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let result = (self.fpr_read_l)(&self.core, fs_reg) as i64 as f64;
        (self.fpr_write_d)(&mut self.core, fd_reg, result);
        self.fpu_update_fcsr()
    }

    // ===== COP1X (MIPS IV FPU extended) =====

    fn exec_lwxc1(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let base = self.core.read_gpr(d.rs as u32);
        let index = self.core.read_gpr(d.rt as u32);
        let addr = base.wrapping_add(index);
        let fd_reg = d.sa as u32;
        match self.read_data::<4>(addr) {
            Ok(val) => { (self.fpr_write_w)(&mut self.core, fd_reg, val as u32); self.handle_exec_complete() }
            Err(status) => self.finish_status(status),
        }
    }
    fn exec_ldxc1(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let base = self.core.read_gpr(d.rs as u32);
        let index = self.core.read_gpr(d.rt as u32);
        let addr = base.wrapping_add(index);
        let fd_reg = d.sa as u32;
        match self.read_data::<8>(addr) {
            Ok(val) => { (self.fpr_write_l)(&mut self.core, fd_reg, val); self.handle_exec_complete() }
            Err(status) => self.finish_status(status),
        }
    }
    fn exec_swxc1(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let base = self.core.read_gpr(d.rs as u32);
        let index = self.core.read_gpr(d.rt as u32);
        let addr = base.wrapping_add(index);
        let fs_reg = d.rd as u32;
        let val = (self.fpr_read_w)(&self.core, fs_reg) as u64;
        let status = self.write_data::<4>(addr, val);
        self.finish_status(status)
    }
    fn exec_sdxc1(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let base = self.core.read_gpr(d.rs as u32);
        let index = self.core.read_gpr(d.rt as u32);
        let addr = base.wrapping_add(index);
        let fs_reg = d.rd as u32;
        let val = (self.fpr_read_l)(&self.core, fs_reg);
        let status = self.write_data::<8>(addr, val);
        self.finish_status(status)
    }
    fn exec_prefx(&mut self, _d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        self.handle_exec_complete()
    }
    fn exec_madd_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fr_val = f32::from_bits((self.fpr_read_w)(&self.core, d.rs as u32));
        let ft_val = f32::from_bits((self.fpr_read_w)(&self.core, d.rt as u32));
        let fs_val = f32::from_bits((self.fpr_read_w)(&self.core, d.rd as u32));
        let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        (self.fpr_write_w)(&mut self.core, fd_reg, (fs_val.mul_add(ft_val, fr_val)).to_bits());
        self.fpu_update_fcsr()
    }
    fn exec_madd_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let read_d = self.fpr_read_d; let write_d = self.fpr_write_d;
        let fr_val = read_d(&self.core, d.rs as u32);
        let ft_val = read_d(&self.core, d.rt as u32);
        let fs_val = read_d(&self.core, d.rd as u32);
        let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        write_d(&mut self.core, fd_reg, fs_val.mul_add(ft_val, fr_val));
        self.fpu_update_fcsr()
    }
    fn exec_msub_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fr_val = f32::from_bits((self.fpr_read_w)(&self.core, d.rs as u32));
        let ft_val = f32::from_bits((self.fpr_read_w)(&self.core, d.rt as u32));
        let fs_val = f32::from_bits((self.fpr_read_w)(&self.core, d.rd as u32));
        let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        (self.fpr_write_w)(&mut self.core, fd_reg, (fs_val.mul_add(ft_val, -fr_val)).to_bits());
        self.fpu_update_fcsr()
    }
    fn exec_msub_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let read_d = self.fpr_read_d; let write_d = self.fpr_write_d;
        let fr_val = read_d(&self.core, d.rs as u32);
        let ft_val = read_d(&self.core, d.rt as u32);
        let fs_val = read_d(&self.core, d.rd as u32);
        let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        write_d(&mut self.core, fd_reg, fs_val.mul_add(ft_val, -fr_val));
        self.fpu_update_fcsr()
    }
    fn exec_nmadd_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fr_val = f32::from_bits((self.fpr_read_w)(&self.core, d.rs as u32));
        let ft_val = f32::from_bits((self.fpr_read_w)(&self.core, d.rt as u32));
        let fs_val = f32::from_bits((self.fpr_read_w)(&self.core, d.rd as u32));
        let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        (self.fpr_write_w)(&mut self.core, fd_reg, (-fs_val.mul_add(ft_val, fr_val)).to_bits());
        self.fpu_update_fcsr()
    }
    fn exec_nmadd_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let read_d = self.fpr_read_d; let write_d = self.fpr_write_d;
        let fr_val = read_d(&self.core, d.rs as u32);
        let ft_val = read_d(&self.core, d.rt as u32);
        let fs_val = read_d(&self.core, d.rd as u32);
        let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        write_d(&mut self.core, fd_reg, -fs_val.mul_add(ft_val, fr_val));
        self.fpu_update_fcsr()
    }
    fn exec_nmsub_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fr_val = f32::from_bits((self.fpr_read_w)(&self.core, d.rs as u32));
        let ft_val = f32::from_bits((self.fpr_read_w)(&self.core, d.rt as u32));
        let fs_val = f32::from_bits((self.fpr_read_w)(&self.core, d.rd as u32));
        let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        (self.fpr_write_w)(&mut self.core, fd_reg, (-fs_val.mul_add(ft_val, -fr_val)).to_bits());
        self.fpu_update_fcsr()
    }
    fn exec_nmsub_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let read_d = self.fpr_read_d; let write_d = self.fpr_write_d;
        let fr_val = read_d(&self.core, d.rs as u32);
        let ft_val = read_d(&self.core, d.rt as u32);
        let fs_val = read_d(&self.core, d.rd as u32);
        let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        write_d(&mut self.core, fd_reg, -fs_val.mul_add(ft_val, -fr_val));
        self.fpu_update_fcsr()
    }

    // LWC1 - Load Word to FPU
    fn exec_lwc1(&mut self, d: &DecodedInstr) -> ExecStatus {
        // Check if FPU is usable
        if (self.core.cp0_status & STATUS_CU1) == 0 {
            return self.cpu_unusable(1);
        }

        let base = self.core.read_gpr(d.rs as u32);
        let addr = base.wrapping_add(d.immu64());
        let ft_reg = d.rt as u32;

        // Load word from memory (alignment check done by read_data)
        match self.read_data::<4>(addr) {
            Ok(value) => {
                (self.fpr_write_w)(&mut self.core, ft_reg, value as u32);
                self.handle_exec_complete()
            }
            Err(exc_status) => self.finish_status(exc_status)
        }
    }

    // LDC1 - Load Doubleword to FPU
    fn exec_ldc1(&mut self, d: &DecodedInstr) -> ExecStatus {
        // Check if FPU is usable
        if (self.core.cp0_status & STATUS_CU1) == 0 {
            return self.cpu_unusable(1);
        }

        let base = self.core.read_gpr(d.rs as u32);
        let addr = base.wrapping_add(d.immu64());
        let ft_reg = d.rt as u32;

        // Load doubleword from memory (alignment check done by read_data)
        match self.read_data::<8>(addr) {
            Ok(value) => {
                (self.fpr_write_l)(&mut self.core, ft_reg, value);
                self.handle_exec_complete()
            }
            Err(exc_status) => self.finish_status(exc_status)
        }
    }

    // SWC1 - Store Word from FPU
    fn exec_swc1(&mut self, d: &DecodedInstr) -> ExecStatus {
        // Check if FPU is usable
        if (self.core.cp0_status & STATUS_CU1) == 0 {
            return self.cpu_unusable(1);
        }

        let base = self.core.read_gpr(d.rs as u32);
        let addr = base.wrapping_add(d.immu64());
        let ft_reg = d.rt as u32;

        let value = (self.fpr_read_w)(&self.core, ft_reg) as u64;

        // Store word to memory (alignment check done by write_data)
        let status = self.write_data::<4>(addr, value);
        self.finish_status(status)
    }

    // SDC1 - Store Doubleword from FPU
    fn exec_sdc1(&mut self, d: &DecodedInstr) -> ExecStatus {
        // Check if FPU is usable
        if (self.core.cp0_status & STATUS_CU1) == 0 {
            return self.cpu_unusable(1);
        }

        let base = self.core.read_gpr(d.rs as u32);
        let addr = base.wrapping_add(d.immu64());
        let ft_reg = d.rt as u32;

        let value = (self.fpr_read_l)(&self.core, ft_reg);

        // Store doubleword to memory (alignment check done by write_data)
        let status = self.write_data::<8>(addr, value);
        self.finish_status(status)
    }

    // FPU single-precision comparison
    fn fpu_compare_s(&self, fs: f32, ft: f32, funct: u32) -> bool {
        let cond = funct & 0xF;
        let less = fs < ft;
        let equal = fs == ft;
        let unordered = fs.is_nan() || ft.is_nan();

        match cond {
            0x0 => false, // F (always false)
            0x1 => unordered, // UN
            0x2 => equal, // EQ
            0x3 => unordered || equal, // UEQ
            0x4 => less, // OLT (ordered less than)
            0x5 => unordered || less, // ULT
            0x6 => less || equal, // OLE
            0x7 => unordered || less || equal, // ULE
            0x8 => false, // SF (signaling false)
            0x9 => unordered, // NGLE
            0xA => equal, // SEQ
            0xB => unordered || equal, // NGL
            0xC => less, // LT
            0xD => unordered || less, // NGE
            0xE => less || equal, // LE
            0xF => unordered || less || equal, // NGT
            _ => false,
        }
    }

    // FPU double-precision comparison
    fn fpu_compare_d(&self, fs: f64, ft: f64, funct: u32) -> bool {
        let cond = funct & 0xF;
        let less = fs < ft;
        let equal = fs == ft;
        let unordered = fs.is_nan() || ft.is_nan();

        match cond {
            0x0 => false, // F (always false)
            0x1 => unordered, // UN
            0x2 => equal, // EQ
            0x3 => unordered || equal, // UEQ
            0x4 => less, // OLT (ordered less than)
            0x5 => unordered || less, // ULT
            0x6 => less || equal, // OLE
            0x7 => unordered || less || equal, // ULE
            0x8 => false, // SF (signaling false)
            0x9 => unordered, // NGLE
            0xA => equal, // SEQ
            0xB => unordered || equal, // NGL
            0xC => less, // LT
            0xD => unordered || less, // NGE
            0xE => less || equal, // LE
            0xF => unordered || less || equal, // NGT
            _ => false,
        }
    }

    /// Create a snapshot of the current CPU state for undo
    #[cfg(feature = "developer")]
    fn create_snapshot(&self) -> CpuSnapshot {
        CpuSnapshot {
            gpr: self.core.gpr,
            pc: self.core.pc,
            hi: self.core.hi,
            lo: self.core.lo,
            llbit: self.cache.get_llbit(),
            lladdr: self.cache.get_lladdr(),
            cp0_index: self.core.cp0_index,
            cp0_random: self.core.cp0_random,
            cp0_entrylo0: self.core.cp0_entrylo0,
            cp0_entrylo1: self.core.cp0_entrylo1,
            cp0_context: self.core.cp0_context,
            cp0_pagemask: self.core.cp0_pagemask,
            cp0_wired: self.core.cp0_wired,
            cp0_badvaddr: self.core.cp0_badvaddr,
            cp0_count: self.core.cp0_count,
            cp0_entryhi: self.core.cp0_entryhi,
            cp0_compare: self.core.cp0_compare,
            cp0_status: self.core.cp0_status,
            cp0_cause: self.core.cp0_cause,
            cp0_epc: self.core.cp0_epc,
            cp0_prid: self.core.cp0_prid,
            cp0_config: self.core.cp0_config,
            cp0_watchlo: self.core.cp0_watchlo,
            cp0_watchhi: self.core.cp0_watchhi,
            cp0_xcontext: self.core.cp0_xcontext,
            cp0_ecc: self.core.cp0_ecc,
            cp0_cacheerr: self.core.cp0_cacheerr,
            cp0_taglo: self.core.cp0_taglo,
            cp0_taghi: self.core.cp0_taghi,
            cp0_errorepc: self.core.cp0_errorepc,
            fpr: self.core.fpr,
            fpu_fir: self.core.fpu_fir,
            fpu_fccr: self.core.fpu_fccr,
            fpu_fexr: self.core.fpu_fexr,
            fpu_fenr: self.core.fpu_fenr,
            fpu_fcsr: self.core.fpu_fcsr,
            running: self.core.running,
            halted: self.core.halted,
            in_delay_slot: self.core.in_delay_slot,
            delay_slot_target: self.core.delay_slot_target,
            memory_writes: Vec::new(), // Will be populated separately
        }
    }

    /// Restore CPU state from a snapshot
    #[cfg(feature = "developer")]
    fn restore_snapshot(&mut self, snapshot: &CpuSnapshot) {
        self.core.gpr = snapshot.gpr;
        self.core.pc = snapshot.pc;
        self.core.hi = snapshot.hi;
        self.core.lo = snapshot.lo;
        self.cache.set_llbit(snapshot.llbit);
        self.cache.set_lladdr(snapshot.lladdr);
        self.core.cp0_lladdr = snapshot.lladdr;
        self.core.cp0_index = snapshot.cp0_index;
        self.core.cp0_random = snapshot.cp0_random;
        self.core.cp0_entrylo0 = snapshot.cp0_entrylo0;
        self.core.cp0_entrylo1 = snapshot.cp0_entrylo1;
        self.core.cp0_context = snapshot.cp0_context;
        self.core.cp0_pagemask = snapshot.cp0_pagemask;
        self.core.cp0_wired = snapshot.cp0_wired;
        self.core.cp0_badvaddr = snapshot.cp0_badvaddr;
        self.core.cp0_count = snapshot.cp0_count;
        self.core.cp0_entryhi = snapshot.cp0_entryhi;
        self.core.cp0_compare = snapshot.cp0_compare;
        self.core.cp0_status = snapshot.cp0_status;
        self.core.cp0_cause = snapshot.cp0_cause;
        self.core.cp0_epc = snapshot.cp0_epc;
        self.core.cp0_prid = snapshot.cp0_prid;
        self.core.cp0_config = snapshot.cp0_config;
        self.core.cp0_watchlo = snapshot.cp0_watchlo;
        self.core.cp0_watchhi = snapshot.cp0_watchhi;
        self.core.cp0_xcontext = snapshot.cp0_xcontext;
        self.core.cp0_ecc = snapshot.cp0_ecc;
        self.core.cp0_cacheerr = snapshot.cp0_cacheerr;
        self.core.cp0_taglo = snapshot.cp0_taglo;
        self.core.cp0_taghi = snapshot.cp0_taghi;
        self.core.cp0_errorepc = snapshot.cp0_errorepc;
        self.core.fpr = snapshot.fpr;
        self.core.fpu_fir = snapshot.fpu_fir;
        self.core.fpu_fccr = snapshot.fpu_fccr;
        self.core.fpu_fexr = snapshot.fpu_fexr;
        self.core.fpu_fenr = snapshot.fpu_fenr;
        self.core.fpu_fcsr = snapshot.fpu_fcsr;
        // Sync host FPU rounding mode to the restored FCSR.RM — this bypasses
        // write_fpu_control (which is what normally keeps the two in sync via
        // CTC1), so without this the host rounding mode stays whatever it was
        // before the restore instead of matching the snapshotted guest state.
        crate::platform::set_fpu_mode((snapshot.fpu_fcsr & 0x3) as u8);
        self.core.running = snapshot.running;
        self.core.halted = snapshot.halted;
        self.core.in_delay_slot = snapshot.in_delay_slot;
        self.core.delay_slot_target = snapshot.delay_slot_target;
        // cp0_count/cp0_compare were restored as raw fields above — re-anchor
        // the virtual count at the restored value and re-arm the compare
        // timer, exactly as a snapshot-file load does.
        self.core.reanchor_count_and_reschedule();
        // self.pcp (jitv2's tracked PhysicalCodePage for the current fetch)
        // and the nanotlb are both keyed off whatever PC/ASID was live
        // before this restore — neither one has any way to notice pc just
        // moved out from under it (jitv2_track_pcp only re-derives self.pcp
        // on a nanotlb *miss*, and restoring core.pc directly doesn't cause
        // one). Without invalidating here, the next exec_decoded call after
        // a big rewind (e.g. undo N across a page boundary, or j2 replay's
        // full-window rewind) runs jitv2's dispatch gate against a stale
        // self.pcp for a completely different physical page than the one
        // the restored PC actually lives on — silently probing/publishing
        // into the wrong page's entry table. `on_cp0_status_changed` covers
        // this (nanotlb_invalidate nulls self.pcp too — see its own doc
        // comment, which explicitly names "snapshot restore" as an
        // anticipated caller) plus the two other derived-state resyncs a
        // direct `self.core.cp0_status = ...` write above bypasses:
        // `update_translate_fn` (kernel/supervisor/user + 32/64-bit mode
        // selects a different translate_fn) and `update_fpr_mode` (STATUS_FR
        // selects different fpr_read_w/fpr_write_w/etc. function pointers).
        // Restoring cp0_status as a raw field write earlier in this function
        // (matching create_snapshot's shape) never goes through the normal
        // write_cp0_status path that would trigger these automatically — a
        // real, live bug found via `j2 replay`: without this, a rewind
        // crossing any of these mode boundaries left the executor decoding
        // through stale function pointers for the restored state, producing
        // spurious "JIT diverged at instruction 1" reports that were actually
        // this rewind path corrupting state before replay's JIT dispatch
        // ever got a chance to run.
        self.on_cp0_status_changed(0, snapshot.cp0_status);
    }

    /// Track a memory write for potential undo
    #[cfg(feature = "developer")]
    fn track_memory_write(&mut self, virt_addr: u64, phys_addr: u64, old_value: u64, size: usize) {
        if !self.undo_buffer.is_enabled() {
            return;
        }

        self.pending_memory_writes.push(MemoryWrite {
            virt_addr,
            phys_addr,
            old_value,
            size,
        });
    }

    /// Commit the current instruction to the undo buffer
    #[cfg(feature = "developer")]
    fn commit_undo_snapshot(&mut self) {
        if !self.undo_buffer.is_enabled() {
            return;
        }

        let mut snapshot = self.create_snapshot();
        snapshot.memory_writes = std::mem::take(&mut self.pending_memory_writes);
        self.undo_buffer.push(snapshot);
    }
    /// Execute the given decoded instruction. Every dispatch-target exec_*
    /// handler is now fully self-contained: it calls handle_exec_complete /
    /// branch_delay / handle_branch_likely_skip / handle_exception itself as
    /// its terminal action, so PC and in_delay_slot are already correct by
    /// the time control returns here — this is a plain tail call.
    #[inline(always)]
    pub fn exec_decoded(&mut self, d: &DecodedInstr) -> ExecStatus {
        #[cfg(feature = "instr_stats")]
        self.instr_stats.record(d.op, d.rs, d.rt, d.funct, d.raw);

        // JIT v2 dispatch gate (rules/jitv2/jit-v2-design.md §6.1.2's `arrival`,
        // simplified — no promoted-handler inline cache yet, just a direct
        // check on every dispatch). `self.pcp` must already be current: every
        // exec_decoded call is preceded by a fetch_instr -> nanotlb_translate
        // this same step() (the only thing that ever sets it) — null here
        // means that invariant broke, not a state to quietly route around.
        //
        // Two conditions make PC worth probing as a compiled entry:
        // `core.jit_trigger` (a branch/jump just committed this PC — set by
        // the interpreter's handle_exec_complete/exec_complete_pc_set, by
        // JIT-compiled code's own jump/branch exit stubs
        // (emit_absolute_pc_exit/emit_runtime_pc_exit in codegen.rs), *and*
        // by jitv2_track_pcp itself whenever this dispatch's physical page
        // differs from the one previously tracked — covering every way pc
        // can land on a fresh page, not just branch/jump commits: sequential
        // page-crossing fallthrough, and — the case the old, narrower
        // "entry_offset == 0" proxy this replaced actually missed —
        // exception/TLB-refill vector entry, since `deliver_exception`
        // (mips_core.rs) writes `core.pc` directly to a fixed vector address
        // with no reason to know about jit_trigger, and the general-exception
        // vector in particular lands at word-offset 0x60 within its page, not
        // 0), or the offset's valid bit already being set (worth re-probing
        // even without a fresh trigger, e.g. loop back-edges within an
        // already-hot region).
        //
        // Word 0 was refused as an entry offset entirely until this point
        // (§6.1.4's "total entry predicate," original rationale: a page's
        // first word might be the delay slot of a branch at the *previous*
        // page's offset 0xFFC, which this page's compile has no way to see
        // statically — cross-page delay-slot inheritance). That's now
        // handled at runtime instead: every compiled entry word already
        // carries the `core.in_delay_slot`/`delay_slot_target` runtime check
        // (`codegen.rs`, `word == entry_word` branch — built and proven for
        // the same-page case, e.g. the PROM reset vector's `j realstart`)
        // for exactly this "this word might be someone else's inherited
        // slot" scenario. The check is page-agnostic — it reads plain
        // `MipsCore` fields `branch_delay` sets identically regardless of
        // which page the branch was on — so it closes the cross-page case
        // the same way it already closes the same-page one, and entry
        // acceptance no longer needs to statically refuse word 0 to stay
        // correct.
        // jitv2_lockstep now uses this SAME real dispatch gate (redesigned from
        // the old standalone-recompile lockstep_check): it forces inline
        // compile (below) so every region compiles+publishes+runs
        // synchronously on this thread, and codegen brackets every instruction
        // with lockstep_step/lockstep_compare callbacks that run the
        // interpreter reference and compare in-region (see
        // MipsExecutor::lockstep_step). So the gate is NOT compiled out under
        // lockstep anymore — publishing/running compiled regions is exactly
        // what drives the per-instruction verification.
        //
        // `self.jitv2_inline_compile` (runtime, not a Cargo feature) selects
        // between the two ways a miss gets a fresh artifact: `false` (the
        // default, matching normal runs) hands a CompileRequest to the async
        // compile_queue worker thread and falls through to the interpreter
        // for this dispatch, same as always; `true` calls
        // comp::handle_request synchronously right here instead — no
        // cross-thread scheduling at all — and, since the function is
        // already sitting right there having just been compiled, runs it
        // immediately rather than waiting for the next dispatch of this PC
        // to pick it up via is_entry_valid. Tests can flip this to exercise
        // jitv2 deterministically (no dependence on whether the async
        // compile thread won a race within a short loop — see
        // rules/jitv2/codegen-gotchas.md) or to compare inline-vs-threaded
        // compile behavior directly.
        // Under `lightning`, this is a literal `true` (not
        // `self.jitv2_dispatch_enabled` read at runtime) so the compiler can
        // see the gate is unconditional at compile time — `lightning`/
        // `developer` are mutually exclusive (see lib.rs), and
        // jitv2_dispatch_enabled has no setter reachable outside
        // `developer` (`set_jitv2_dispatch_enabled`, `j2 dispatch off`), so
        // the field can never actually be false here under `lightning`
        // anyway; this just lets the compiler know that statically instead
        // of leaving a dead runtime check on the hot path.
        // Under lockstep the gate is on by default (it drives the inline
        // per-instruction verification — `jitv2_dispatch_enabled` defaults
        // true), but still honors that runtime switch so a pure-interpreter
        // test harness (`run_interpreter`, which needs an uninstrumented
        // reference run) can turn it off. The old design got interpreter-only
        // for free by compiling the gate out under lockstep; now it's a runtime
        // switch instead.
        #[cfg(feature = "jitv2")]
        if cfg!(feature = "lightning") || self.jitv2_dispatch_enabled {
            assert!(!self.pcp.is_null(), "exec_decoded reached with no tracked PhysicalCodePage");
            let page = unsafe { &mut *self.pcp };
            let entry_offset = ((self.core.pc & 0xFFF) >> 2) as usize;
            let trigger = self.core.jit_trigger;
            if trigger || page.is_published(entry_offset) {
                if page.is_entry_valid(entry_offset) {
                    // This dispatch is about to jump straight into a
                    // compiled function at pc — record it here, tagged
                    // JitEntry (distinct from a plain interior Jit
                    // instruction), instead of step()'s old unconditional
                    // Interp push (see step()'s doc comment on why that was
                    // wrong): this is this compiled unit's own entry,
                    // reached externally, exactly the arrival
                    // emit_dev_trace_bp's internal-edge-only contract
                    // assumes step() already logged.
                    #[cfg(not(feature = "lightning"))]
                    self.traceback.push(self.core.pc, d.raw, InstrOrigin::JitEntry);
                    #[cfg(feature = "developer")]
                    if let Some(w) = self.trace_writer.as_mut() {
                        let record = crate::trace::TraceRecord::capture(self.core.pc, d.raw, &self.core);
                        let _ = w.push(&record);
                    }
                    let func = page.entries[entry_offset].func;
                    debug_assert!(!func.is_null(), "valid bit set with null func");
                    #[cfg(feature = "developer")]
                    { page.entries[entry_offset].call_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
                    let jit_fn: crate::jitv2::JitFn = unsafe { std::mem::transmute(func) };
                    return unsafe { jit_fn(&mut self.core as *mut MipsCore) };
                }
                // Call-count gate (`j2 min-calls`): only applies to the
                // async path — `jitv2_inline_compile`'s whole contract is
                // "compile and run this exact dispatch immediately"
                // (tests flip it on specifically for that determinism,
                // see its own doc comment below), which a >0 threshold
                // would silently break by sometimes returning to the
                // interpreter instead on the first dispatch. Threshold 0
                // (the default) makes this check a no-op either way.
                // Below threshold: skip sending a request this dispatch
                // (not yet hot enough) and fall through to the
                // interpreter, same as every other "nothing to send"
                // outcome below (denylisted, try_schedule lost, ...) —
                // no early return, this dispatch still needs to actually
                // execute the instruction via the interpreter path past
                // this whole gate.
                // Under `lightning`, `jitv2_inline_compile` is unreachable
                // (no setter exists outside `developer`, and `lightning`/
                // `developer` are mutually exclusive — see lib.rs) and is
                // forced to its literal-`false` compile-time value here, same
                // reasoning as the dispatch gate above: the field can never
                // actually be true, so let the compiler see that statically
                // and drop the inline-compile arm entirely instead of
                // carrying a dead runtime check on the hot path.
                // jitv2_lockstep forces inline compile: the whole verification
                // model depends on every region compiling+running synchronously
                // on this thread so the lockstep_step/compare callbacks fire
                // in-region on the very next instructions, deterministically —
                // an async worker-thread compile would let the interpreter run
                // words unverified until the artifact lands.
                let inline_compile = cfg!(feature = "jitv2_lockstep")
                    || (!cfg!(feature = "lightning") && self.jitv2_inline_compile);
                let below_call_threshold = !inline_compile
                    && !page.count_dispatch_and_check_threshold(entry_offset, crate::jitv2::min_calls_before_compile());
                if !below_call_threshold {
                self.core.jit_trigger = false;
                let req = crate::jitv2::CompileRequest {
                    page: self.pcp,
                    offset: entry_offset as u16,
                    compiled_for_fr1: (self.core.cp0_status & crate::mips_core::STATUS_FR) != 0,
                };
                if inline_compile {
                    // Unlike the async compile thread's own worker_loop,
                    // nothing else ever checks this shared Codegen's
                    // growth on the inline path — so do it here, BEFORE
                    // compiling this request, not after: `flush_from_cpu_thread`
                    // clears the page pool (including `self.pcp`/`page`,
                    // via `nanotlb_invalidate`), which would yank the rug
                    // out from under the "run it immediately" logic below
                    // if it ran right after this call's own `handle_request`
                    // published into that same page. Checking before means
                    // worst case this dispatch's own compile pushes one
                    // past the threshold and the *next* dispatch flushes —
                    // same bounded-overshoot behavior worker_loop already
                    // accepts for the threaded path.
                    let over_threshold = self.jitv2.lock().codegen.lock().as_ref()
                        .is_some_and(|c| c.packing_stats().1 > crate::jitv2::CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES);
                    if over_threshold {
                        // `core.pc` is virtual — jitv2_track_pcp needs a
                        // physical address, and `page`/`self.pcp` are
                        // about to go dangling into the just-cleared
                        // pool, so grab the physical page base from
                        // `page.pfn` (still valid) before flushing rather
                        // than re-translating pc from scratch.
                        let phys_page_base = page.pfn * crate::jitv2::PAGE_SIZE;
                        unsafe { self.jitv2.lock().flush_from_cpu_thread(self.sysad.clone()); }
                        // Only `self.pcp`/`page` above are stale (raw
                        // pointers into the just-cleared pool) — the
                        // nanotlb's own translations are untouched by
                        // mega_flush, so leave them alone; just null the
                        // dangling pcp and re-derive it for the exact
                        // page this dispatch already landed on (mirrors
                        // jitv2_track_pcp's own pool-exhaustion recovery).
                        self.pcp = std::ptr::null_mut();
                        self.jitv2_track_pcp(phys_page_base);
                        return self.exec_decoded(d);
                    }
                    // Take the shared Codegen for the duration of this
                    // one compile only — see Jitv2::codegen's doc
                    // comment for why inline dispatch and the async
                    // compile thread share a single instance rather than
                    // each owning a separate Cranelift memory arena.
                    let mut codegen = self.jitv2.lock().codegen.lock().take();
                    // Under lockstep the whole point is that EVERY region
                    // compiles+runs here so the lockstep_step/compare callbacks
                    // fire. A `None` codegen means it's owned by the async
                    // compile queue (which must never run under lockstep — see
                    // Machine::new) — inline would silently skip, degrading the
                    // whole run to unverified interpreter (exactly the false-
                    // confidence bug that let "clean lockstep boots" verify
                    // nothing). Fail loud instead of silent.
                    #[cfg(feature = "jitv2_lockstep")]
                    assert!(codegen.is_some(),
                        "jitv2_lockstep: inline compile found no Codegen (async compile queue stole it?) — \
                         verification would silently be OFF. The queue must not run under lockstep.");
                    let mut ran_out_of_memory = false;
                    if let Some(codegen) = codegen.as_mut() {
                        #[cfg(feature = "developer")]
                        {
                            let stats = self.jitv2.lock().stats.clone();
                            ran_out_of_memory = crate::jitv2::comp::handle_request(&req, &self.sysad, &mut self.jitv2_inline_analyzer, codegen, &stats);
                        }
                        #[cfg(not(feature = "developer"))]
                        { ran_out_of_memory = crate::jitv2::comp::handle_request(&req, &self.sysad, &mut self.jitv2_inline_analyzer, codegen); }
                    }
                    *self.jitv2.lock().codegen.lock() = codegen;
                    // Status-bar feedback (disp.rs's StatusBar): mirrors
                    // worker_loop's own update on the threaded path. Only
                    // runs on an actual compile (this whole block is gated
                    // by below_call_threshold above), not every dispatch,
                    // so it's fine on this otherwise-hot CPU-thread path.
                    // Queue fill stays whatever the async side last left it
                    // (0 if the queue was never started) — inline mode has
                    // no queue of its own to report.
                    {
                        let reserved_bytes = self.jitv2.lock().codegen.lock().as_ref()
                            .map_or(0, |c| c.packing_stats().1);
                        crate::jit_feedback::JIT_FEEDBACK.set_arena_fill(reserved_bytes, crate::jitv2::CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES);
                    }
                    if ran_out_of_memory {
                        // The compile that just ran couldn't get memory
                        // — flush immediately and retry this exact
                        // dispatch from scratch, regardless of
                        // function_count (see
                        // Codegen::last_compile_ran_out_of_memory's doc
                        // comment for why the count-based threshold
                        // alone isn't enough now that regions can be
                        // much larger than the single-instruction case
                        // it was originally sized against). Same
                        // pcp/nanotlb recovery as the pre-emptive
                        // threshold check above.
                        let phys_page_base = page.pfn * crate::jitv2::PAGE_SIZE;
                        unsafe { self.jitv2.lock().flush_from_cpu_thread(self.sysad.clone()); }
                        self.pcp = std::ptr::null_mut();
                        self.jitv2_track_pcp(phys_page_base);
                        return self.exec_decoded(d);
                    }
                    // Freshly published (if handle_request succeeded):
                    // run it immediately instead of falling through to
                    // the interpreter and waiting for the next dispatch
                    // of this PC to pick it up via is_entry_valid above.
                    if page.is_entry_valid(entry_offset) {
                        // Same as the other jit_fn call site above: record
                        // this external entry as JitEntry before jumping in —
                        // this is the very first dispatch of a page that
                        // just got compiled+published inline (is_entry_valid
                        // was false a few lines up, before handle_request
                        // ran), so it never went through that call site.
                        #[cfg(not(feature = "lightning"))]
                        self.traceback.push(self.core.pc, d.raw, InstrOrigin::JitEntry);
                        #[cfg(feature = "developer")]
                        if let Some(w) = self.trace_writer.as_mut() {
                            let record = crate::trace::TraceRecord::capture(self.core.pc, d.raw, &self.core);
                            let _ = w.push(&record);
                        }
                        let func = page.entries[entry_offset].func;
                        debug_assert!(!func.is_null(), "valid bit set with null func");
                        #[cfg(feature = "developer")]
                        { page.entries[entry_offset].call_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
                        let jit_fn: crate::jitv2::JitFn = unsafe { std::mem::transmute(func) };
                        return unsafe { jit_fn(&mut self.core as *mut MipsCore) };
                    }
                    // handle_request declined (denylisted, e.g. a 0xFFC
                    // hazard or codegen gap) — fall through to the
                    // interpreter below, same as the threaded path's
                    // post-send fallthrough.
                } else if page.try_schedule(entry_offset) {
                    // Nothing valid to run, and no request for this
                    // offset already in flight (try_schedule won the
                    // test-and-set): ask the compile thread for a fresh
                    // artifact and fall through to the interpreter
                    // below. If try_schedule lost (another dispatch
                    // already scheduled this exact offset — e.g. a hot
                    // loop back-edge re-triggering every iteration while
                    // the first request is still queued/compiling),
                    // skip sending a redundant duplicate; still falls
                    // through to the interpreter the same as if we had.
                    {
                        let mut jit = self.jitv2.lock();
                        let stats = jit.stats.clone();
                        jit.compile_queue.send(req, &stats);
                    }
                }
                } // !below_call_threshold
            }
        }

        // Real interpreter dispatch: the JIT gate above fell through without
        // an early return (denylisted page, codegen gap, below the
        // min-calls threshold, or jitv2 not built) — record it here, tagged
        // Interp, same shape as the JIT-hit push above but for this path.
        #[cfg(not(feature = "lightning"))]
        self.traceback.push(self.core.pc, d.raw, InstrOrigin::Interp);
        #[cfg(feature = "developer")]
        if let Some(w) = self.trace_writer.as_mut() {
            let record = crate::trace::TraceRecord::capture(self.core.pc, d.raw, &self.core);
            let _ = w.push(&record);
        }
        type Fn<T, C> = fn(&mut MipsExecutor<T, C>, &DecodedInstr) -> ExecStatus;
        let f: Fn<T, C> = unsafe { std::mem::transmute(d.handler) };
        f(self, d)
    }

    /// Fetch, decode, and dispatch exactly one instruction at `core.pc`
    /// straight to the interpreter's own handler (`d.handler`), bypassing
    /// `exec_decoded`'s JIT dispatch gate entirely. `install_jit_hooks`
    /// wires this up as `core.interp_fallback_fn` (see that field's doc
    /// comment) — compiled code's only way to force real forward progress
    /// on something it bailed on (e.g. `emit_fpu_entry_guard`'s CU1/FR
    /// mismatch check) instead of re-entering the exact same compiled
    /// function on the next dispatch and bailing again, forever.
    ///
    /// Deliberately skips `step()`'s own per-instruction preamble (timer/
    /// interrupt bookkeeping, breakpoint checks, traceback): compiled code's
    /// own preamble already ran the equivalent for this exact PC immediately
    /// before calling in (`emit_pending_interrupt_preamble`/
    /// `emit_increment_cycles`, and — in developer builds — the per-instruction
    /// dev hook `emit_dev_trace_bp` which records the traceback entry and checks
    /// breakpoints for *this* fallback word too, so there's nothing to do here;
    /// same reasoning as `compile_region`'s `skip_entry_preamble`).
    #[cfg(feature = "jitv2")]
    fn interp_dispatch_one(&mut self) -> ExecStatus {
        let pc = self.core.pc;
        let fetch = self.fetch_instr(pc);
        if fetch.status != EXEC_COMPLETE {
            return if fetch.status & EXEC_IS_EXCEPTION != 0 {
                self.handle_exception(fetch.status)
            } else {
                fetch.status
            };
        }
        // fetch_instr/cache.fetch() only fills in `.raw` — decode is lazy,
        // triggered here on `FLAG_NOT_DECODED` exactly like step()'s own
        // fetch+dispatch sequence (mips_exec.rs's main step() body). Missing
        // this call is a real bug, not a redundant safety check: `d.handler`
        // is left null/stale from whatever this scratch slot last held until
        // decode_into runs, and calling through a null handler segfaults
        // (found via a debug-build backtrace after this exact omission
        // crashed a CU1-guard equivalence test).
        let slot = fetch.instr as *mut DecodedInstr;
        let d = unsafe { &mut *slot };
        // SCRATCH cache-vs-physical-memory guard: compare what the cache
        // model just handed back against a direct bus read of the same
        // physical address (bypassing cache entirely, same primitive
        // `j2 analyze` uses). Catches a cache line that's HIT + valid but
        // holds bytes that disagree with real backing memory (found live:
        // intrfast's entry word, 0x08022ab0, an L2 line reported
        // DirtyExclusive/HIT with wrong cached content). Remove once the
        // root cause is found.
        {
            let cached_raw = d.raw;
            let tr = self.debug_translate(pc);
            if !tr.is_exception() {
                let phys_read = self.sysad.read32(tr.phys);
                if phys_read.is_ok() && phys_read.data != cached_raw {
                    eprintln!(
                        "\n=== CACHE/MEMORY MISMATCH at pc={:#018x} phys={:#010x}: cache.raw={:#010x} bus.raw={:#010x} ===\n",
                        pc, tr.phys, cached_raw, phys_read.data
                    );
                    self.core.pc = pc;
                    return EXEC_BREAKPOINT;
                }
            }
        }
        if d.flags != 0 {
            decode_into::<T, C>(d);
        }
        // Record this fallback's own dispatch — without this, a fallback
        // word (an Excluded instruction the analyzer keeps in-region, run
        // through the real interpreter via this function) was completely
        // invisible in `dt`: emit_dev_trace_bp only ever fires for native-
        // codegen instructions, never for this path.
        #[cfg(not(feature = "lightning"))]
        self.traceback.push(pc, d.raw, InstrOrigin::FallbackWord);
        type Fn<T, C> = fn(&mut MipsExecutor<T, C>, &DecodedInstr) -> ExecStatus;
        let f: Fn<T, C> = unsafe { std::mem::transmute(d.handler) };
        let status = f(self, d);
        // jitv2_lockstep: a fallback runs the interpreter here, whose
        // read_data/write_data may set core.lockstep_mem. That capture is the
        // fallback's own committed access — it must NOT survive to be compared
        // against a *later* memory instruction's JIT access in the same region
        // (lockstep_jit_read/write would assert a spurious address/value
        // mismatch). The fallback runs the real interpreter, so there's no
        // separate JIT re-run to compare anyway; clear it. (Each bracketed
        // Sequential instruction re-establishes lockstep_mem via its own
        // lockstep_step before its JIT access.)
        #[cfg(feature = "jitv2_lockstep")]
        { self.core.lockstep_mem = None; }
        status
    }

    /// Inline lockstep STEP (`core.lockstep_step_fn`, via `jit_lockstep_step`):
    /// called at the start of every JIT instruction. Establishes the true
    /// per-instruction starting state (`core.pc = pc`, `in_delay_slot = bd` —
    /// the JIT doesn't keep `core.pc` live for straight-line instructions, so
    /// the callback carries it), snapshots that state, runs the instruction
    /// once through the real interpreter to produce the reference result
    /// (`ls_expected`), then restores the starting state (keeping
    /// `core.lockstep_mem`, the record of any memory access the interpreter
    /// just committed, so the JIT's own re-run compares against it via
    /// `lockstep_jit_read`/`lockstep_jit_write` instead of double-accessing).
    /// The matching `lockstep_compare` a moment later checks the JIT's result
    /// against `ls_expected`.
    ///
    /// If this instruction's `LockstepClass` can't be compared per-dispatch
    /// (Branch/Other — see the classify call below),
    /// or the interpreter's reference run didn't cleanly complete (retry/
    /// breakpoint/exception with nothing to compare — mirrors the old
    /// load-store check's skip), leaves `ls_expected = None` so `lockstep_compare`
    /// is a no-op and the JIT just runs normally.
    #[cfg(feature = "jitv2_lockstep")]
    fn lockstep_step(&mut self, pc: u64, raw: u32, bd: u32) {
        self.ls_before = None;
        self.ls_expected = None;

        // Lockstep is always on when built (no runtime toggle): if you compiled
        // with jitv2_lockstep, every verifiable instruction self-checks — it's
        // driven from the JIT block itself. `lockstep_classify` is only
        // consulted to skip the classes that can't be meaningfully compared by
        // this per-instruction bracket: `Branch` (the JIT inlines the delay
        // slot into one unit, so "one JIT instruction" != "one interpreter
        // dispatch" — a whole separate reconciliation, deferred) and `Other`
        // (opcodefusion producers whose fused dispatch does more than a
        // standalone re-decode). Codegen only ever brackets straight-line
        // Sequential/CP1 anyway, so this is belt-and-suspenders.
        match lockstep_classify(raw) {
            LockstepClass::Alu | LockstepClass::LoadStore | LockstepClass::Fpu => {}
            LockstepClass::Branch | LockstepClass::Other => return,
        }

        // Establish the true starting state for this instruction. For a plain
        // straight-line head the JIT doesn't keep core.pc live, so codegen's
        // emit_lockstep_step materialized pc/in_delay_slot from compile-time
        // constants and passes them here — set them so the interpreter
        // reference run is anchored. For an entry word or delay slot
        // (`bd == LOCKSTEP_BD_LIVE`), core.pc/in_delay_slot/delay_slot_target
        // are already correct from the arrival (the interpreter dispatch or the
        // BC1 fallback that reached this word) and MUST NOT be overwritten — a
        // compile-time bd would clobber an inherited in_delay_slot=true and make
        // the interpreter reference mishandle the pending transfer. Trust the
        // live values in that case (`pc` is still the instruction's own address,
        // which already equals live core.pc, so we can set it either way).
        if bd != LOCKSTEP_BD_LIVE {
            self.core.pc = pc;
            self.core.in_delay_slot = bd != 0;
        }
        // Start each instruction with no captured memory access: a
        // non-memory instruction's interpreter reference run leaves lockstep_mem
        // untouched, so without this a stale Some(..) from an earlier memory
        // instruction could survive and be compared against the *next* memory
        // instruction's JIT access (a spurious address/value divergence).
        self.core.lockstep_mem = None;

        let before = LockstepSnapshot::capture(&self.core);
        let delay_target_before = self.core.delay_slot_target;

        // Run this exact instruction once through the real interpreter, using
        // the `raw` the JIT was compiled from — decode it into the scratch slot
        // and call its handler directly (NOT via exec_decoded, which would
        // re-enter the JIT gate, nor via interp_dispatch_one, which re-fetches
        // from the bus — the JIT probe's instruction is `raw`, which may not be
        // sitting at core.pc in a direct jit_fn test's mock memory). The
        // handler's read_data/write_data populate core.lockstep_mem for the JIT
        // re-run to compare against.
        self.ins.raw = raw;
        self.ins.flags = FLAG_NOT_DECODED;
        decode_into::<T, C>(&mut self.ins);
        let d: *const DecodedInstr = &self.ins;
        type Fn2<T, C> = fn(&mut MipsExecutor<T, C>, &DecodedInstr) -> ExecStatus;
        let handler: Fn2<T, C> = unsafe { std::mem::transmute((*d).handler) };
        let interp_status = handler(self, unsafe { &*d });
        let interp = LockstepSnapshot::capture(&self.core);
        let delay_target_after = self.core.delay_slot_target;

        // Nothing meaningful to compare when the interpreter didn't cleanly
        // retire (bus retry, memory breakpoint, or an exception raised inside
        // the cache layer that the JIT's translate-only mem hooks can't see —
        // same skip the old lockstep_check_load_store made). Leave interp state
        // live (it's the authoritative result) and disable the compare.
        if interp_status != EXEC_COMPLETE {
            self.ls_before = None;
            self.ls_expected = None;
            // core stays in interp's post-state (already live); the JIT is
            // about to re-run from `before`, but the caller made this a no-op —
            // actually restore `before` so the JIT runs cleanly and its own
            // exception path (production mem hooks under lockstep still detect
            // faults via jit_mem_exc) produces the real result the region uses.
            before.restore_into(self, delay_target_before);
            self.core.lockstep_mem = None;
            return;
        }

        // Stash the interpreter reference, then restore the starting state so
        // the JIT runs the same instruction against identical inputs. Keep
        // core.lockstep_mem (the interpreter's committed access) for the JIT's
        // compare-not-recommit mem hooks.
        self.ls_expected = Some(interp);
        self.ls_delay_target_expected = delay_target_after;
        self.ls_before = Some(before);
        self.ls_delay_target_before = delay_target_before;
        let saved_mem = self.core.lockstep_mem;
        before.restore_into(self, delay_target_before);
        self.core.lockstep_mem = saved_mem;
    }

    /// Test-only: force a lockstep divergence and return `lockstep_compare`'s
    /// verdict. Snapshots the current `core` as the "interpreter reference"
    /// (`ls_expected`), mutates one GPR so the live `core` (the "JIT result")
    /// differs, then runs the compare — which must report a divergence
    /// (`true`) and return without panicking (the whole point: a real
    /// divergence bails to the monitor, it never aborts the process). Verifies
    /// the redesigned "print + break, don't panic" behavior end to end.
    #[cfg(all(test, feature = "jitv2_lockstep"))]
    pub fn test_force_lockstep_divergence(&mut self) -> bool {
        let snap = LockstepSnapshot::capture(&self.core);
        self.ls_expected = Some(snap);
        self.ls_before = Some(snap);
        self.ls_delay_target_expected = self.core.delay_slot_target;
        // Make the live ("JIT") state differ from the stashed ("interp") one.
        self.core.gpr[5] = self.core.gpr[5].wrapping_add(0xDEAD);
        self.lockstep_compare()
    }

    /// Inline lockstep COMPARE (`core.lockstep_compare_fn`, via
    /// `jit_lockstep_compare`): called at the end of every JIT instruction,
    /// after its own semantics ran. If `lockstep_step` stashed a reference
    /// (`ls_expected`), compare the JIT's now-current state against it;
    /// otherwise a no-op (disabled class, or the interpreter's reference run
    /// didn't cleanly retire). Clears the scratch either way.
    ///
    /// Returns `true` on a divergence — the caller (`jit_lockstep_compare`)
    /// turns that into an `EXEC_BREAKPOINT` return so compiled code bails to the
    /// monitor instead of aborting the process. On divergence this prints a
    /// full, human-readable report (which specific GPRs / CP0 / FPR fields
    /// differ, jit vs interp) to stderr first, so the state is captured even if
    /// the monitor is then used to poke around (the JIT already ran the wrong
    /// instruction — `core` holds the JIT's post-state at the divergent PC).
    #[cfg(feature = "jitv2_lockstep")]
    fn lockstep_compare(&mut self) -> bool {
        let Some(expected) = self.ls_expected.take() else {
            self.ls_before = None;
            self.core.lockstep_mem = None;
            return false;
        };
        let before = self.ls_before.take();
        LOCKSTEP_VERIFICATIONS.fetch_add(1, Ordering::Relaxed); // a real compare is happening
        let jit = LockstepSnapshot::capture(&self.core);
        let jit_delay_target = self.core.delay_slot_target;

        // Full compare including pc/in_delay_slot/delay_slot_target: the
        // lockstep build makes the JIT materialize these per-instruction (see
        // codegen), so they're directly comparable with no branch/slot-model
        // reconciliation. compare_fpr/compare_delay_slot both true — unlike the
        // old standalone checks that could only compare a subset per class,
        // this runs the real instruction in-region so every field is live.
        let mismatch = jit.mismatches(&expected, true, true)
            || jit_delay_target != self.ls_delay_target_expected;
        self.core.lockstep_mem = None;
        if !mismatch {
            return false;
        }

        // Divergence: print a detailed, field-by-field report, then signal the
        // caller to break into the monitor (do NOT panic — an extern "C"
        // callback panic aborts the whole process, losing the monitor and all
        // state you'd want to inspect).
        let at_pc = before.map(|b| b.pc).unwrap_or(expected.pc);
        // Back core.pc/in_delay_slot/delay_slot_target up to the divergent
        // instruction's own starting values — captured in `before`/
        // `ls_delay_target_before`, on the Rust side, before the interpreter
        // reference ever ran, so no need for codegen to pre-materialize or
        // restore any of the three around the compare call for this. The
        // three are always handled as one unit (same as `restore_into`'s own
        // contract) — never just `pc` alone, since a stale `in_delay_slot`/
        // `delay_slot_target` left over from the JIT's wrong run would make
        // the monitor misinterpret a plain instruction as mid-delay-slot (or
        // vice versa) at the address it just landed on. Every OTHER field is
        // left exactly as the JIT produced it — "JIT's wrong post-state is
        // live" is deliberate, only these three need correcting so the break
        // lands in the right place with a coherent slot state.
        self.core.pc = at_pc;
        if let Some(b) = before {
            self.core.in_delay_slot = b.in_delay_slot;
        }
        self.core.delay_slot_target = self.ls_delay_target_before;
        eprintln!("\n=== jitv2_lockstep DIVERGENCE at pc={:#018x} raw was executed by JIT ===", at_pc);
        for r in 0..32 {
            if jit.gpr[r] != expected.gpr[r] {
                eprintln!("  gpr[{:2}] {:<5} jit={:#018x}  interp={:#018x}", r, crate::mips_dis::reg_name(r as u32), jit.gpr[r], expected.gpr[r]);
            }
        }
        if jit.pc != expected.pc { eprintln!("  pc       jit={:#018x}  interp={:#018x}", jit.pc, expected.pc); }
        if jit.in_delay_slot != expected.in_delay_slot { eprintln!("  in_delay_slot jit={}  interp={}", jit.in_delay_slot, expected.in_delay_slot); }
        if jit_delay_target != self.ls_delay_target_expected { eprintln!("  delay_slot_target jit={:#018x}  interp={:#018x}", jit_delay_target, self.ls_delay_target_expected); }
        if jit.hi != expected.hi { eprintln!("  hi       jit={:#018x}  interp={:#018x}", jit.hi, expected.hi); }
        if jit.lo != expected.lo { eprintln!("  lo       jit={:#018x}  interp={:#018x}", jit.lo, expected.lo); }
        if jit.status != expected.status { eprintln!("  cp0_status jit={:#010x}  interp={:#010x}", jit.status, expected.status); }
        if jit.cause != expected.cause { eprintln!("  cp0_cause  jit={:#010x}  interp={:#010x}", jit.cause, expected.cause); }
        if jit.epc != expected.epc { eprintln!("  cp0_epc  jit={:#018x}  interp={:#018x}", jit.epc, expected.epc); }
        for r in 0..32 {
            if jit.fpr[r] != expected.fpr[r] {
                eprintln!("  fpr[{:2}]  jit={:#018x}  interp={:#018x}", r, jit.fpr[r], expected.fpr[r]);
            }
        }
        if jit.fcsr != expected.fcsr { eprintln!("  fcsr     jit={:#010x}  interp={:#010x}", jit.fcsr, expected.fcsr); }
        eprintln!("=== breaking into monitor; use `dt`, `r`, `d pc` to inspect (JIT's wrong post-state is live) ===\n");
        true
    }

    /// Translate `va` for a lockstep memory-hook comparison — no bus access,
    /// matching `nanotlb_translate`'s own contract. `AT` selects Read (4) vs
    /// Write (2), same `AccessType` encoding `jit_read*`/`jit_write*`'s real
    /// (non-lockstep) `read_data`/`write_data` calls resolve internally.
    #[cfg(feature = "jitv2_lockstep")]
    fn lockstep_translate<const AT: u8>(&mut self, va: u64) -> TranslateResult {
        self.nanotlb_translate::<AT>(va)
    }

    /// Lockstep replacement for every `jit_read*_fn` hook (see
    /// `MipsCore::lockstep_mem_*`'s doc comment and `read_data_impl`'s
    /// matching capture): called by the JIT probe in place of a real bus
    /// read. Never touches the bus itself — a second real read could have
    /// side effects (MMIO) the first, real interpreter dispatch already
    /// committed. Instead: translates `va` independently (proving the JIT's
    /// own address computation agrees with what the real access already
    /// did), and returns the interpreter's already-captured value so the
    /// JIT's register-level effect matches by construction — this hook
    /// verifies address/translation correctness, not data-path plumbing
    /// (there's no second copy of memory to read a genuinely independent
    /// value from). A mismatch panics immediately rather than silently
    /// continuing with divergent state, same as every other lockstep
    /// assertion.
    ///
    /// `core.lockstep_mem == None` means either no real interpreter access
    /// has ever happened on this `MipsCore` (this JIT call didn't arrive via
    /// `lockstep_check_load_store`'s interpreter-first dispatch — every
    /// direct JIT-only test in `equiv_test.rs`, for instance) or the most
    /// recent real access faulted/retried (`lockstep_check_load_store`
    /// already detects that and skips the JIT probe entirely before it
    /// could reach here — see that function's own doc comment — so this
    /// case shouldn't actually be reachable in practice, but there's still
    /// nothing to compare against if it somehow were). Either way: fall
    /// through to a real bus read instead of asserting against absent
    /// capture data — same behavior a direct JIT test expects from a plain
    /// (non-lockstep) build.
    #[cfg(feature = "jitv2_lockstep")]
    fn lockstep_jit_read<const SIZE: usize>(&mut self, va: u64) -> u64 {
        let Some(captured) = self.core.lockstep_mem else {
            let result = match self.read_data::<SIZE>(va) {
                Ok(v) => { self.core.jit_mem_exc = EXEC_COMPLETE; v }
                Err(status) => { self.core.jit_mem_exc = status; 0 }
            };
            // read_data's own jitv2_lockstep instrumentation just set
            // core.lockstep_mem = Some(this access) as a side effect meant
            // for the interpreter-first lockstep_check_load_store path —
            // irrelevant here (this branch means no interpreter-first
            // capture existed for THIS instruction, i.e. a direct JIT-only
            // dispatch), and leaving it Some would wrongly become the
            // comparison target for the *next* JIT-issued access in the
            // same compiled region (a multi-load/store unit — e.g. LWL
            // immediately followed by LWR — would otherwise compare the
            // second access's address/value against the first's capture
            // and assert-fail on the mismatch). Clear it back to None so
            // every JIT-issued access starts from the same "nothing to
            // compare against" state this branch is actually meant to mean.
            self.core.lockstep_mem = None;
            return result;
        };
        let translate_result = self.lockstep_translate::<{ AccessType::Read as u8 }>(va);
        if translate_result.is_exception() {
            self.core.jit_mem_exc = translate_result.status;
            return 0;
        }
        let phys = translate_result.phys as u64;
        if va != captured.addr || phys != captured.phys {
            eprintln!("\n=== jitv2_lockstep DIVERGENCE at pc={:#018x}: read address mismatch ===", self.core.pc);
            eprintln!("  va   jit={:#018x}  interp={:#018x}", va, captured.addr);
            eprintln!("  phys jit={:#018x}  interp={:#018x}", phys, captured.phys);
            eprintln!("=== breaking into monitor; use `dt`, `r`, `d pc` to inspect (JIT's wrong post-state is live) ===\n");
            self.core.lockstep_mem = None;
            self.core.jit_mem_exc = EXEC_BREAKPOINT;
            return 0;
        }
        self.core.jit_mem_exc = EXEC_COMPLETE;
        let mask: u64 = if SIZE == 8 { u64::MAX } else { (1u64 << (SIZE * 8)) - 1 };
        captured.value & mask
    }

    /// Lockstep replacement for every `jit_write*_fn` hook — see
    /// `lockstep_jit_read`'s doc comment for the full reasoning (a second
    /// real write would double-apply it, so this compares address/value
    /// against the interpreter's already-committed write instead of issuing
    /// a second one) and for the `None` fallback (direct JIT-only tests
    /// never populate `core.lockstep_mem`, so fall through to a real write).
    #[cfg(feature = "jitv2_lockstep")]
    fn lockstep_jit_write<const SIZE: usize>(&mut self, va: u64, val: u64) -> u32 {
        let Some(captured) = self.core.lockstep_mem else {
            let status = self.write_data::<SIZE>(va, val);
            self.core.jit_mem_exc = status;
            // See lockstep_jit_read's matching None-arm comment: write_data
            // just set core.lockstep_mem as a side effect meant for the
            // interpreter-first path, which doesn't apply to this
            // direct-JIT-dispatch branch — clear it so it can't leak into
            // comparing the next JIT-issued access in the same compiled
            // region against this one's capture.
            self.core.lockstep_mem = None;
            return status;
        };
        let translate_result = self.lockstep_translate::<{ AccessType::Write as u8 }>(va);
        if translate_result.is_exception() {
            self.core.jit_mem_exc = translate_result.status;
            return translate_result.status;
        }
        let phys = translate_result.phys as u64;
        let mask: u64 = if SIZE == 8 { u64::MAX } else { (1u64 << (SIZE * 8)) - 1 };
        let jit_masked = val & mask;
        let interp_masked = captured.value & mask;
        if va != captured.addr || phys != captured.phys || jit_masked != interp_masked {
            eprintln!("\n=== jitv2_lockstep DIVERGENCE at pc={:#018x}: write mismatch ===", self.core.pc);
            eprintln!("  va    jit={:#018x}  interp={:#018x}", va, captured.addr);
            eprintln!("  phys  jit={:#018x}  interp={:#018x}", phys, captured.phys);
            eprintln!("  value jit={:#018x}  interp={:#018x}  (SIZE={} mask={:#x})", jit_masked, interp_masked, SIZE, mask);
            eprintln!("=== breaking into monitor; use `dt`, `r`, `d pc` to inspect (JIT's wrong post-state is live) ===\n");
            self.core.lockstep_mem = None;
            self.core.jit_mem_exc = EXEC_BREAKPOINT;
            return EXEC_BREAKPOINT;
        }
        self.core.jit_mem_exc = EXEC_COMPLETE;
        EXEC_COMPLETE
    }
}

/// Everything an inline-lockstep-compared instruction can read or write,
/// captured before the interpreter reference run (`lockstep_step`'s `before`,
/// restored so the JIT runs from the same inputs), as the interpreter result
/// (`ls_expected`), and after the JIT's own run (`lockstep_compare`'s `jit`).
/// One shared shape covers every instruction class now that lockstep runs the
/// real in-region instruction rather than a standalone recompile: every field
/// is live and comparable (`mismatches(compare_fpr=true, compare_delay_slot=true)`).
#[cfg(feature = "jitv2_lockstep")]
#[derive(Clone, Copy)]
pub(crate) struct LockstepSnapshot {
    gpr: [u64; 32],
    hi: u64,
    lo: u64,
    pc: u64,
    status: u32,
    cause: u32,
    epc: u64,
    fpr: [u64; 32],
    fcsr: u32,
    fccr: u32,
    fexr: u32,
    fenr: u32,
    in_delay_slot: bool,
}

#[cfg(feature = "jitv2_lockstep")]
impl LockstepSnapshot {
    fn capture(core: &MipsCore) -> Self {
        Self {
            gpr: core.gpr,
            hi: core.hi,
            lo: core.lo,
            pc: core.pc,
            status: core.cp0_status,
            cause: core.cp0_cause,
            epc: core.cp0_epc,
            fpr: core.fpr,
            fcsr: core.fpu_fcsr,
            fccr: core.fpu_fccr,
            fexr: core.fpu_fexr,
            fenr: core.fpu_fenr,
            in_delay_slot: core.in_delay_slot,
        }
    }

    /// Write this snapshot back into `core`/`delay_slot_target` — used both
    /// to restore real state after the JIT probe (before the interpreter
    /// re-run) and, implicitly, is never needed for the interpreter's own
    /// post-run state since that's just left live for the final compare.
    fn restore_into<T: Tlb, C: MipsCache>(&self, exec: &mut MipsExecutor<T, C>, delay_slot_target: u64) {
        exec.core.gpr = self.gpr;
        exec.core.hi = self.hi;
        exec.core.lo = self.lo;
        exec.core.pc = self.pc;
        exec.core.cp0_status = self.status;
        exec.core.cp0_cause = self.cause;
        exec.core.cp0_epc = self.epc;
        exec.core.fpr = self.fpr;
        exec.core.fpu_fcsr = self.fcsr;
        exec.core.fpu_fccr = self.fccr;
        exec.core.fpu_fexr = self.fexr;
        exec.core.fpu_fenr = self.fenr;
        exec.core.in_delay_slot = self.in_delay_slot;
        exec.core.delay_slot_target = delay_slot_target;
    }

    /// `compare_fpr`/`compare_delay_slot`: kept as explicit flags rather than
    /// always comparing every field so a mismatch message never lists a
    /// field its own check can't have caused, which would be confusing to
    /// read while bisecting.
    ///
    /// `compare_delay_slot`: the ALU/FPU checks never let their instruction
    /// touch `in_delay_slot` (a standalone ALU or FPU op never changes
    /// delay-slot state — only a branch/jump does), so `lockstep_check_branch`
    /// is the only caller that passes `true`.
    ///
    /// `compare_fpr`: the ALU check's instruction can't touch `fpr` at all
    /// (CP1 is its own `LockstepClass`), so it passes `false`. The branch
    /// check's delay slot is always ALU-only too (see its own slot
    /// classification), so it can't touch `fpr` either — also `false`,
    /// despite passing `true` for `compare_delay_slot`; the two flags are
    /// independent, not a single bundled one, precisely because FPU needs
    /// the opposite combination: `compare_fpr=true` (the whole point of the
    /// check) but `compare_delay_slot=false` (a standalone CP1 op is no
    /// different from ALU there).
    fn mismatches(&self, other: &Self, compare_fpr: bool, compare_delay_slot: bool) -> bool {
        self.gpr != other.gpr
            || self.hi != other.hi
            || self.lo != other.lo
            || self.pc != other.pc
            || self.status != other.status
            || self.cause != other.cause
            || self.epc != other.epc
            || (compare_fpr && (self.fpr != other.fpr || self.fcsr != other.fcsr || self.fccr != other.fccr || self.fexr != other.fexr || self.fenr != other.fenr))
            || (compare_delay_slot && self.in_delay_slot != other.in_delay_slot)
    }
}

/// Instruction category for `jitv2_lockstep`'s inline self-check (not
/// `analyzer::Classify`, which is about reachability/control-flow, not semantics
/// grouping). Only used to decide *whether* a bracketed instruction can be
/// meaningfully compared per-dispatch: Alu/LoadStore/Fpu yes; Branch (JIT
/// inlines the delay slot into one unit — the two-dispatch reconciliation is
/// deferred) and Other (opcodefusion producers) no. Lockstep itself is always
/// on when built — there's no per-class runtime toggle anymore.
#[cfg(feature = "jitv2_lockstep")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LockstepClass {
    Alu,
    Branch,
    LoadStore,
    Fpu,
    Other,
}

/// Classify `raw` for `jitv2_lockstep`. Deliberately coarse — this is a
/// bring-up bisection tool, not the analyzer's reachability walk, so it only
/// needs to recognize the four buckets the user asked to distinguish;
/// anything it doesn't recognize (CP0, TLB, SYSCALL/BREAK, …) falls into
/// `Other` and is never probed.
#[cfg(feature = "jitv2_lockstep")]
fn lockstep_classify(raw: u32) -> LockstepClass {
    let op = (raw >> 26) & 0x3F;
    let funct = raw & 0x3F;

    // Under opcodefusion, ADDU/SUBU (as an address-calc producer fused with
    // a following load/store) and ADDIU/LUI (as either half of an
    // LS-address-calc or LUI+ORI/ADDIU 32-bit-immediate fusion) can each be
    // dispatched via a fused handler whose real semantics differ from a
    // plain re-decode of the same raw word — the fused handler also
    // consumes d.imm (repurposed to hold the *next* instruction's raw word)
    // and, for LS fusion, performs the load/store too. There's no reliable
    // way to tell from `raw` alone (post-decode, FLAG_IMM_IS_NEXT is always
    // cleared) whether a given dispatch actually fused, so these opcodes are
    // excluded from the Alu bucket entirely whenever fusion is compiled in,
    // rather than risk comparing a standalone recompiled ADDIU/LUI against a
    // dispatch that really did more than that.
    #[cfg(feature = "opcodefusion")]
    let fusable_producer = matches!(op, OP_ADDIU | OP_LUI)
        || (op == OP_SPECIAL && matches!(funct, FUNCT_ADDU | FUNCT_SUBU));
    #[cfg(not(feature = "opcodefusion"))]
    let fusable_producer = false;
    if fusable_producer {
        return LockstepClass::Other;
    }

    match op {
        OP_SPECIAL => match funct {
            FUNCT_JR | FUNCT_JALR => LockstepClass::Branch,
            FUNCT_ADD | FUNCT_ADDU | FUNCT_SUB | FUNCT_SUBU
            | FUNCT_AND | FUNCT_OR | FUNCT_XOR | FUNCT_NOR
            | FUNCT_SLT | FUNCT_SLTU
            | FUNCT_SLL | FUNCT_SRL | FUNCT_SRA | FUNCT_SLLV | FUNCT_SRLV | FUNCT_SRAV
            | FUNCT_MFHI | FUNCT_MTHI | FUNCT_MFLO | FUNCT_MTLO
            | FUNCT_MULT | FUNCT_MULTU | FUNCT_DIV | FUNCT_DIVU
            | FUNCT_DADD | FUNCT_DADDU | FUNCT_DSUB | FUNCT_DSUBU
            | FUNCT_DSLL | FUNCT_DSRL | FUNCT_DSRA | FUNCT_DSLL32 | FUNCT_DSRL32 | FUNCT_DSRA32
            | FUNCT_DSLLV | FUNCT_DSRLV | FUNCT_DSRAV => LockstepClass::Alu,
            _ => LockstepClass::Other,
        },
        OP_REGIMM => LockstepClass::Branch,
        OP_J | OP_JAL => LockstepClass::Branch,
        OP_BEQ | OP_BNE | OP_BLEZ | OP_BGTZ
        | OP_BEQL | OP_BNEL | OP_BLEZL | OP_BGTZL => LockstepClass::Branch,
        OP_ADDI | OP_ADDIU | OP_SLTI | OP_SLTIU | OP_ANDI | OP_ORI | OP_XORI | OP_LUI
        | OP_DADDI | OP_DADDIU => LockstepClass::Alu,
        OP_LB | OP_LH | OP_LWL | OP_LW | OP_LBU | OP_LHU | OP_LWR | OP_LWU
        | OP_SB | OP_SH | OP_SWL | OP_SW | OP_SDL | OP_SDR | OP_SWR
        | OP_LDL | OP_LDR | OP_LL | OP_LLD | OP_SC | OP_SCD
        | OP_LD | OP_SD | OP_CACHE | OP_PREF => LockstepClass::LoadStore,
        OP_COP1 | OP_COP1X | OP_LWC1 | OP_SWC1 | OP_LDC1 | OP_SDC1 => LockstepClass::Fpu,
        _ => LockstepClass::Other,
    }
}

/// True if `next_raw` is one of the fusable load/store opcodes (LB/LBU/LH/LHU/
/// LW/LD/SB/SH/SW/SD — see rules/perf/ notes on the addr-calc+load/store
/// fusion) using `addr_reg` as its base register (`rs` field). Any offset is
/// fine — the fused handler reads the load/store's own offset field exactly
/// like the unfused path does; only ADDU/SUBU/ADDIU are fusable producers
/// (ADD/SUB are excluded because they can themselves fault on overflow,
/// which would need the same fault-exclusion treatment as the load/store
/// itself and hasn't been built).
#[cfg(feature = "opcodefusion")]
#[inline(always)]
fn is_fusable_load_store(next_raw: u32, addr_reg: u8) -> bool {
    let next_op = (next_raw >> 26) & 0x3F;
    let next_rs = (next_raw >> 21) & 0x1F;
    if next_rs != addr_reg as u32 {
        return false;
    }
    matches!(next_op,
        OP_LB | OP_LBU | OP_LH | OP_LHU | OP_LW | OP_LD | OP_SB | OP_SH | OP_SW | OP_SD)
}

/// Decode `raw` into `ins`. Caller is responsible for checking `ins.flags` first.
///
/// If the caller sets FLAG_IMM_IS_NEXT beforehand, `ins.imm` is read here (once,
/// before being overwritten below) as the raw opcode word of the delay-slot
/// instruction from the same L1I line — used to fuse a branch/jump with a NOP
/// delay slot (see FLAG_IMM_IS_NEXT doc comment). The bit and the borrowed
/// `imm` slot are always consumed by the time this function returns. EXCEPTION:
/// for a fusable ADDU/SUBU/ADDIU (see is_fusable_load_store), `imm` is instead
/// left holding the load/store's raw word permanently (not transient) — the
/// fused handler needs it at execution time to decode the load/store's own
/// rt/offset fields.
pub fn decode_into<T: Tlb, C: MipsCache>(ins: &mut DecodedInstr) {
    let raw = ins.raw;

    let op    = ((raw >> 26) & 0x3F) as u8;
    let rs    = ((raw >> 21) & 0x1F) as u8;
    let rt    = ((raw >> 16) & 0x1F) as u8;
    let rd    = ((raw >> 11) & 0x1F) as u8;
    let sa    = ((raw >>  6) & 0x1F) as u8;
    let funct = (raw & 0x3F) as u8;

    type Fn<T, C> = fn(&mut MipsExecutor<T, C>, &DecodedInstr) -> ExecStatus;

    let handler: Fn<T, C> = match op as u32 {
        OP_SPECIAL => match funct as u32 {
            FUNCT_SLL     => MipsExecutor::<T,C>::exec_sll,
            FUNCT_MOVCI   => MipsExecutor::<T,C>::exec_movci,
            FUNCT_SRL     => MipsExecutor::<T,C>::exec_srl,
            FUNCT_SRA     => MipsExecutor::<T,C>::exec_sra,
            FUNCT_SLLV    => MipsExecutor::<T,C>::exec_sllv,
            FUNCT_SRLV    => MipsExecutor::<T,C>::exec_srlv,
            FUNCT_SRAV    => MipsExecutor::<T,C>::exec_srav,
            #[cfg(feature = "opcodefusion")]
            FUNCT_JR      => if ins.flags & FLAG_IMM_IS_NEXT != 0 && ins.imm == 0 { MipsExecutor::<T,C>::exec_jr_nop } else { MipsExecutor::<T,C>::exec_jr },
            #[cfg(not(feature = "opcodefusion"))]
            FUNCT_JR      => MipsExecutor::<T,C>::exec_jr,
            FUNCT_JALR    => MipsExecutor::<T,C>::exec_jalr,
            FUNCT_MOVZ    => MipsExecutor::<T,C>::exec_movz,
            FUNCT_MOVN    => MipsExecutor::<T,C>::exec_movn,
            FUNCT_SYSCALL => MipsExecutor::<T,C>::exec_syscall,
            FUNCT_BREAK   => MipsExecutor::<T,C>::exec_break,
            FUNCT_SYNC    => MipsExecutor::<T,C>::exec_sync,
            FUNCT_MFHI    => MipsExecutor::<T,C>::exec_mfhi,
            FUNCT_MTHI    => MipsExecutor::<T,C>::exec_mthi,
            FUNCT_MFLO    => MipsExecutor::<T,C>::exec_mflo,
            FUNCT_MTLO    => MipsExecutor::<T,C>::exec_mtlo,
            FUNCT_DSLLV   => MipsExecutor::<T,C>::exec_dsllv,
            FUNCT_DSRLV   => MipsExecutor::<T,C>::exec_dsrlv,
            FUNCT_DSRAV   => MipsExecutor::<T,C>::exec_dsrav,
            FUNCT_MULT    => MipsExecutor::<T,C>::exec_mult,
            FUNCT_MULTU   => MipsExecutor::<T,C>::exec_multu,
            FUNCT_DIV     => MipsExecutor::<T,C>::exec_div,
            FUNCT_DIVU    => MipsExecutor::<T,C>::exec_divu,
            FUNCT_DMULT   => MipsExecutor::<T,C>::exec_dmult,
            FUNCT_DMULTU  => MipsExecutor::<T,C>::exec_dmultu,
            FUNCT_DDIV    => MipsExecutor::<T,C>::exec_ddiv,
            FUNCT_DDIVU   => MipsExecutor::<T,C>::exec_ddivu,
            FUNCT_ADD     => MipsExecutor::<T,C>::exec_add,
            #[cfg(feature = "opcodefusion")]
            FUNCT_ADDU    => {
                if ins.flags & FLAG_IMM_IS_NEXT != 0 && is_fusable_load_store(ins.imm, rd) {
                    MipsExecutor::<T,C>::exec_addu_ls
                } else {
                    MipsExecutor::<T,C>::exec_addu
                }
            }
            #[cfg(not(feature = "opcodefusion"))]
            FUNCT_ADDU    => MipsExecutor::<T,C>::exec_addu,
            FUNCT_SUB     => MipsExecutor::<T,C>::exec_sub,
            #[cfg(feature = "opcodefusion")]
            FUNCT_SUBU    => {
                if ins.flags & FLAG_IMM_IS_NEXT != 0 && is_fusable_load_store(ins.imm, rd) {
                    MipsExecutor::<T,C>::exec_subu_ls
                } else {
                    MipsExecutor::<T,C>::exec_subu
                }
            }
            #[cfg(not(feature = "opcodefusion"))]
            FUNCT_SUBU    => MipsExecutor::<T,C>::exec_subu,
            FUNCT_AND     => MipsExecutor::<T,C>::exec_and,
            FUNCT_OR      => MipsExecutor::<T,C>::exec_or,
            FUNCT_XOR     => MipsExecutor::<T,C>::exec_xor,
            FUNCT_NOR     => MipsExecutor::<T,C>::exec_nor,
            FUNCT_SLT     => MipsExecutor::<T,C>::exec_slt,
            FUNCT_SLTU    => MipsExecutor::<T,C>::exec_sltu,
            FUNCT_DADD    => MipsExecutor::<T,C>::exec_dadd,
            FUNCT_DADDU   => MipsExecutor::<T,C>::exec_daddu,
            FUNCT_DSUB    => MipsExecutor::<T,C>::exec_dsub,
            FUNCT_DSUBU   => MipsExecutor::<T,C>::exec_dsubu,
            FUNCT_TGE     => MipsExecutor::<T,C>::exec_tge,
            FUNCT_TGEU    => MipsExecutor::<T,C>::exec_tgeu,
            FUNCT_TLT     => MipsExecutor::<T,C>::exec_tlt,
            FUNCT_TLTU    => MipsExecutor::<T,C>::exec_tltu,
            FUNCT_TEQ     => MipsExecutor::<T,C>::exec_teq,
            FUNCT_TNE     => MipsExecutor::<T,C>::exec_tne,
            FUNCT_DSLL    => MipsExecutor::<T,C>::exec_dsll,
            FUNCT_DSRL    => MipsExecutor::<T,C>::exec_dsrl,
            FUNCT_DSRA    => MipsExecutor::<T,C>::exec_dsra,
            FUNCT_DSLL32  => MipsExecutor::<T,C>::exec_dsll32,
            FUNCT_DSRL32  => MipsExecutor::<T,C>::exec_dsrl32,
            FUNCT_DSRA32  => MipsExecutor::<T,C>::exec_dsra32,
            _             => MipsExecutor::<T,C>::exec_reserved,
        },
        OP_REGIMM => match rt as u32 {
            RT_BLTZ    => { ins.set_imm_se4(raw); MipsExecutor::<T,C>::exec_bltz }
            RT_BGEZ    => { ins.set_imm_se4(raw); MipsExecutor::<T,C>::exec_bgez }
            RT_BLTZL   => { ins.set_imm_se4(raw); MipsExecutor::<T,C>::exec_bltzl_ri }
            RT_BGEZL   => { ins.set_imm_se4(raw); MipsExecutor::<T,C>::exec_bgezl_ri }
            RT_TGEI    => { ins.set_imm_se(raw);     MipsExecutor::<T,C>::exec_tgei }
            RT_TGEIU   => { ins.set_imm_se(raw);     MipsExecutor::<T,C>::exec_tgeiu }
            RT_TLTI    => { ins.set_imm_se(raw);     MipsExecutor::<T,C>::exec_tlti }
            RT_TLTIU   => { ins.set_imm_se(raw);     MipsExecutor::<T,C>::exec_tltiu }
            RT_TEQI    => { ins.set_imm_se(raw);     MipsExecutor::<T,C>::exec_teqi }
            RT_TNEI    => { ins.set_imm_se(raw);     MipsExecutor::<T,C>::exec_tnei }
            RT_BLTZAL  => { ins.set_imm_se4(raw); MipsExecutor::<T,C>::exec_bltzal }
            RT_BGEZAL  => { ins.set_imm_se4(raw); MipsExecutor::<T,C>::exec_bgezal }
            RT_BLTZALL => { ins.set_imm_se4(raw); MipsExecutor::<T,C>::exec_bltzall }
            RT_BGEZALL => { ins.set_imm_se4(raw); MipsExecutor::<T,C>::exec_bgezall }
            _          => MipsExecutor::<T,C>::exec_reserved,
        },
        #[cfg(feature = "opcodefusion")]
        OP_J      => { let fuse = ins.flags & FLAG_IMM_IS_NEXT != 0 && ins.imm == 0; ins.set_imm_j(raw); if fuse { MipsExecutor::<T,C>::exec_j_nop } else { MipsExecutor::<T,C>::exec_j } }
        #[cfg(not(feature = "opcodefusion"))]
        OP_J      => { ins.set_imm_j(raw); MipsExecutor::<T,C>::exec_j }
        #[cfg(feature = "opcodefusion")]
        OP_JAL    => { let fuse = ins.flags & FLAG_IMM_IS_NEXT != 0 && ins.imm == 0; ins.set_imm_j(raw); if fuse { MipsExecutor::<T,C>::exec_jal_nop } else { MipsExecutor::<T,C>::exec_jal } }
        #[cfg(not(feature = "opcodefusion"))]
        OP_JAL    => { ins.set_imm_j(raw); MipsExecutor::<T,C>::exec_jal }
        #[cfg(feature = "opcodefusion")]
        OP_BEQ    => { let fuse = ins.flags & FLAG_IMM_IS_NEXT != 0 && ins.imm == 0; ins.set_imm_se4(raw); if fuse { MipsExecutor::<T,C>::exec_beq_nop } else { MipsExecutor::<T,C>::exec_beq } }
        #[cfg(not(feature = "opcodefusion"))]
        OP_BEQ    => { ins.set_imm_se4(raw); MipsExecutor::<T,C>::exec_beq }
        #[cfg(feature = "opcodefusion")]
        OP_BNE    => { let fuse = ins.flags & FLAG_IMM_IS_NEXT != 0 && ins.imm == 0; ins.set_imm_se4(raw); if fuse { MipsExecutor::<T,C>::exec_bne_nop } else { MipsExecutor::<T,C>::exec_bne } }
        #[cfg(not(feature = "opcodefusion"))]
        OP_BNE    => { ins.set_imm_se4(raw); MipsExecutor::<T,C>::exec_bne }
        OP_BLEZ   => { ins.set_imm_se4(raw); MipsExecutor::<T,C>::exec_blez }
        OP_BGTZ   => { ins.set_imm_se4(raw); MipsExecutor::<T,C>::exec_bgtz }
        OP_BEQL   => { ins.set_imm_se4(raw); MipsExecutor::<T,C>::exec_beql }
        OP_BNEL   => { ins.set_imm_se4(raw); MipsExecutor::<T,C>::exec_bnel }
        OP_BLEZL  => { ins.set_imm_se4(raw); MipsExecutor::<T,C>::exec_blezl }
        OP_BGTZL  => { ins.set_imm_se4(raw); MipsExecutor::<T,C>::exec_bgtzl }
        OP_ADDI   => { ins.set_imm_se(raw); MipsExecutor::<T,C>::exec_addi }
        #[cfg(feature = "opcodefusion")]
        OP_ADDIU  => {
            if ins.flags & FLAG_IMM_IS_NEXT != 0 && is_fusable_load_store(ins.imm, rt) {
                // imm stays as the load/store's raw word (see decode_into doc
                // comment) — NOT overwritten with the sign-extended immediate.
                MipsExecutor::<T,C>::exec_addiu_ls
            } else {
                ins.set_imm_se(raw);
                MipsExecutor::<T,C>::exec_addiu
            }
        }
        #[cfg(not(feature = "opcodefusion"))]
        OP_ADDIU  => { ins.set_imm_se(raw); MipsExecutor::<T,C>::exec_addiu }
        OP_DADDI  => { ins.set_imm_se(raw); MipsExecutor::<T,C>::exec_daddi }
        OP_DADDIU => { ins.set_imm_se(raw); MipsExecutor::<T,C>::exec_daddiu }
        OP_SLTI   => { ins.set_imm_se(raw); MipsExecutor::<T,C>::exec_slti }
        OP_SLTIU  => { ins.set_imm_se(raw); MipsExecutor::<T,C>::exec_sltiu }
        OP_ANDI   => { ins.set_imm_ze(raw);               MipsExecutor::<T,C>::exec_andi }
        OP_ORI    => { ins.set_imm_ze(raw);               MipsExecutor::<T,C>::exec_ori }
        OP_XORI   => { ins.set_imm_ze(raw);               MipsExecutor::<T,C>::exec_xori }
        #[cfg(feature = "opcodefusion")]
        OP_LUI    => {
            // Fuse the common 32-bit-immediate idiom `lui rX,hi; {ori,addiu} rX,rX,lo`
            // into one handler call when the following word (from the same L1I
            // line — see FLAG_IMM_IS_NEXT) is ORI/ADDIU with rs==rt==this rt.
            // ORI can't carry (pure OR), so the decode-time combine is exact;
            // ADDIU can carry when lo16's sign bit is set, so its combine must
            // replicate exec_addiu's wrapping-add semantics exactly, not just OR.
            let fused = if ins.flags & FLAG_IMM_IS_NEXT != 0 {
                let next = ins.imm;
                let next_op = (next >> 26) & 0x3F;
                let next_rs = (next >> 21) & 0x1F;
                let next_rt = (next >> 16) & 0x1F;
                let same_reg = next_rs == rt as u32 && next_rt == rt as u32;
                if same_reg && next_op == OP_ORI {
                    let hi = (raw & 0xFFFF) << 16;
                    let lo = next & 0xFFFF;
                    Some((hi | lo, false))
                } else if same_reg && next_op == OP_ADDIU {
                    let hi = ((raw & 0xFFFF) << 16) as i32;
                    let lo = (next & 0xFFFF) as i16 as i32;
                    Some((hi.wrapping_add(lo) as u32, true))
                } else {
                    None
                }
            } else {
                None
            };
            match fused {
                Some((combined, is_addiu)) => {
                    ins.imm = combined;
                    if is_addiu { MipsExecutor::<T,C>::exec_lui_simm32 } else { MipsExecutor::<T,C>::exec_lui_imm32 }
                }
                None => { ins.set_imm_lui(raw); MipsExecutor::<T,C>::exec_lui }
            }
        }
        #[cfg(not(feature = "opcodefusion"))]
        OP_LUI    => { ins.set_imm_lui(raw); MipsExecutor::<T,C>::exec_lui }
        OP_COP0   => MipsExecutor::<T,C>::exec_cop0,
        OP_COP1 => match rs as u32 {
            RS_MFC1  => MipsExecutor::<T,C>::exec_mfc1,
            RS_DMFC1 => MipsExecutor::<T,C>::exec_dmfc1,
            RS_CFC1  => MipsExecutor::<T,C>::exec_cfc1,
            RS_MTC1  => MipsExecutor::<T,C>::exec_mtc1,
            RS_DMTC1 => MipsExecutor::<T,C>::exec_dmtc1,
            RS_CTC1  => MipsExecutor::<T,C>::exec_ctc1,
            RS_BC1   => { ins.set_imm_se4(raw); MipsExecutor::<T,C>::exec_bc1 }
            RS_S => match funct as u32 {
                FUNCT_FADD     => MipsExecutor::<T,C>::exec_fadd_s,
                FUNCT_FSUB     => MipsExecutor::<T,C>::exec_fsub_s,
                FUNCT_FMUL     => MipsExecutor::<T,C>::exec_fmul_s,
                FUNCT_FDIV     => MipsExecutor::<T,C>::exec_fdiv_s,
                FUNCT_FSQRT    => MipsExecutor::<T,C>::exec_fsqrt_s,
                FUNCT_FABS     => MipsExecutor::<T,C>::exec_fabs_s,
                FUNCT_FMOV     => MipsExecutor::<T,C>::exec_fmov_s,
                FUNCT_FNEG     => MipsExecutor::<T,C>::exec_fneg_s,
                FUNCT_FROUND_L => MipsExecutor::<T,C>::exec_fround_l_s,
                FUNCT_FTRUNC_L => MipsExecutor::<T,C>::exec_ftrunc_l_s,
                FUNCT_FCEIL_L  => MipsExecutor::<T,C>::exec_fceil_l_s,
                FUNCT_FFLOOR_L => MipsExecutor::<T,C>::exec_ffloor_l_s,
                FUNCT_FROUND_W => MipsExecutor::<T,C>::exec_fround_w_s,
                FUNCT_FTRUNC_W => MipsExecutor::<T,C>::exec_ftrunc_w_s,
                FUNCT_FCEIL_W  => MipsExecutor::<T,C>::exec_fceil_w_s,
                FUNCT_FFLOOR_W => MipsExecutor::<T,C>::exec_ffloor_w_s,
                FUNCT_FMOVCF   => MipsExecutor::<T,C>::exec_fmovcf_s,
                FUNCT_FMOVZ    => MipsExecutor::<T,C>::exec_fmovz_s,
                FUNCT_FMOVN    => MipsExecutor::<T,C>::exec_fmovn_s,
                FUNCT_FRECIP   => MipsExecutor::<T,C>::exec_frecip_s,
                FUNCT_FRSQRT   => MipsExecutor::<T,C>::exec_frsqrt_s,
                FUNCT_FCVT_D   => MipsExecutor::<T,C>::exec_fcvt_d_s,
                FUNCT_FCVT_W   => MipsExecutor::<T,C>::exec_fcvt_w_s,
                FUNCT_FCVT_L   => MipsExecutor::<T,C>::exec_fcvt_l_s,
                FUNCT_FC_F ..= FUNCT_FC_NGT => MipsExecutor::<T,C>::exec_fcc_s,
                _              => MipsExecutor::<T,C>::exec_reserved,
            },
            RS_D => match funct as u32 {
                FUNCT_FADD     => MipsExecutor::<T,C>::exec_fadd_d,
                FUNCT_FSUB     => MipsExecutor::<T,C>::exec_fsub_d,
                FUNCT_FMUL     => MipsExecutor::<T,C>::exec_fmul_d,
                FUNCT_FDIV     => MipsExecutor::<T,C>::exec_fdiv_d,
                FUNCT_FSQRT    => MipsExecutor::<T,C>::exec_fsqrt_d,
                FUNCT_FABS     => MipsExecutor::<T,C>::exec_fabs_d,
                FUNCT_FMOV     => MipsExecutor::<T,C>::exec_fmov_d,
                FUNCT_FNEG     => MipsExecutor::<T,C>::exec_fneg_d,
                FUNCT_FROUND_L => MipsExecutor::<T,C>::exec_fround_l_d,
                FUNCT_FTRUNC_L => MipsExecutor::<T,C>::exec_ftrunc_l_d,
                FUNCT_FCEIL_L  => MipsExecutor::<T,C>::exec_fceil_l_d,
                FUNCT_FFLOOR_L => MipsExecutor::<T,C>::exec_ffloor_l_d,
                FUNCT_FROUND_W => MipsExecutor::<T,C>::exec_fround_w_d,
                FUNCT_FTRUNC_W => MipsExecutor::<T,C>::exec_ftrunc_w_d,
                FUNCT_FCEIL_W  => MipsExecutor::<T,C>::exec_fceil_w_d,
                FUNCT_FFLOOR_W => MipsExecutor::<T,C>::exec_ffloor_w_d,
                FUNCT_FMOVCF   => MipsExecutor::<T,C>::exec_fmovcf_d,
                FUNCT_FMOVZ    => MipsExecutor::<T,C>::exec_fmovz_d,
                FUNCT_FMOVN    => MipsExecutor::<T,C>::exec_fmovn_d,
                FUNCT_FRECIP   => MipsExecutor::<T,C>::exec_frecip_d,
                FUNCT_FRSQRT   => MipsExecutor::<T,C>::exec_frsqrt_d,
                FUNCT_FCVT_S   => MipsExecutor::<T,C>::exec_fcvt_s_d,
                FUNCT_FCVT_W   => MipsExecutor::<T,C>::exec_fcvt_w_d,
                FUNCT_FCVT_L   => MipsExecutor::<T,C>::exec_fcvt_l_d,
                FUNCT_FC_F ..= FUNCT_FC_NGT => MipsExecutor::<T,C>::exec_fcc_d,
                _              => MipsExecutor::<T,C>::exec_reserved,
            },
            RS_W => match funct as u32 {
                FUNCT_FCVT_S => MipsExecutor::<T,C>::exec_fcvt_s_w,
                FUNCT_FCVT_D => MipsExecutor::<T,C>::exec_fcvt_d_w,
                _            => MipsExecutor::<T,C>::exec_reserved,
            },
            RS_L => match funct as u32 {
                FUNCT_FCVT_S => MipsExecutor::<T,C>::exec_fcvt_s_l,
                FUNCT_FCVT_D => MipsExecutor::<T,C>::exec_fcvt_d_l,
                _            => MipsExecutor::<T,C>::exec_reserved,
            },
            _ => MipsExecutor::<T,C>::exec_reserved,
        },
        OP_COP1X => match funct as u32 {
            FUNCT_LWXC1   => MipsExecutor::<T,C>::exec_lwxc1,
            FUNCT_LDXC1   => MipsExecutor::<T,C>::exec_ldxc1,
            FUNCT_SWXC1   => MipsExecutor::<T,C>::exec_swxc1,
            FUNCT_SDXC1   => MipsExecutor::<T,C>::exec_sdxc1,
            FUNCT_PREFX   => MipsExecutor::<T,C>::exec_prefx,
            FUNCT_MADD_S  => MipsExecutor::<T,C>::exec_madd_s,
            FUNCT_MADD_D  => MipsExecutor::<T,C>::exec_madd_d,
            FUNCT_MSUB_S  => MipsExecutor::<T,C>::exec_msub_s,
            FUNCT_MSUB_D  => MipsExecutor::<T,C>::exec_msub_d,
            FUNCT_NMADD_S => MipsExecutor::<T,C>::exec_nmadd_s,
            FUNCT_NMADD_D => MipsExecutor::<T,C>::exec_nmadd_d,
            FUNCT_NMSUB_S => MipsExecutor::<T,C>::exec_nmsub_s,
            FUNCT_NMSUB_D => MipsExecutor::<T,C>::exec_nmsub_d,
            _             => MipsExecutor::<T,C>::exec_reserved,
        },
        OP_LB     => { ins.set_imm_se(raw); MipsExecutor::<T,C>::exec_lb }
        OP_LH     => { ins.set_imm_se(raw); MipsExecutor::<T,C>::exec_lh }
        OP_LWL    => { ins.set_imm_se(raw); MipsExecutor::<T,C>::exec_lwl }
        OP_LW     => { ins.set_imm_se(raw); MipsExecutor::<T,C>::exec_lw }
        OP_LBU    => { ins.set_imm_se(raw); MipsExecutor::<T,C>::exec_lbu }
        OP_LHU    => { ins.set_imm_se(raw); MipsExecutor::<T,C>::exec_lhu }
        OP_LWR    => { ins.set_imm_se(raw); MipsExecutor::<T,C>::exec_lwr }
        OP_LWU    => { ins.set_imm_se(raw); MipsExecutor::<T,C>::exec_lwu }
        OP_SB     => { ins.set_imm_se(raw); MipsExecutor::<T,C>::exec_sb }
        OP_SH     => { ins.set_imm_se(raw); MipsExecutor::<T,C>::exec_sh }
        OP_SWL    => { ins.set_imm_se(raw); MipsExecutor::<T,C>::exec_swl }
        OP_SW     => { ins.set_imm_se(raw); MipsExecutor::<T,C>::exec_sw }
        OP_SDL    => { ins.set_imm_se(raw); MipsExecutor::<T,C>::exec_sdl }
        OP_SDR    => { ins.set_imm_se(raw); MipsExecutor::<T,C>::exec_sdr }
        OP_SWR    => { ins.set_imm_se(raw); MipsExecutor::<T,C>::exec_swr }
        OP_CACHE  => { ins.set_imm_se(raw); MipsExecutor::<T,C>::exec_cache }
        OP_LL     => { ins.set_imm_se(raw); MipsExecutor::<T,C>::exec_ll }
        OP_LWC1   => { ins.set_imm_se(raw); MipsExecutor::<T,C>::exec_lwc1 }
        OP_LDC1   => { ins.set_imm_se(raw); MipsExecutor::<T,C>::exec_ldc1 }
        OP_LDL    => { ins.set_imm_se(raw); MipsExecutor::<T,C>::exec_ldl }
        OP_LDR    => { ins.set_imm_se(raw); MipsExecutor::<T,C>::exec_ldr }
        OP_LD     => { ins.set_imm_se(raw); MipsExecutor::<T,C>::exec_ld }
        OP_SC     => { ins.set_imm_se(raw); MipsExecutor::<T,C>::exec_sc }
        OP_SWC1   => { ins.set_imm_se(raw); MipsExecutor::<T,C>::exec_swc1 }
        OP_SDC1   => { ins.set_imm_se(raw); MipsExecutor::<T,C>::exec_sdc1 }
        OP_SD     => { ins.set_imm_se(raw); MipsExecutor::<T,C>::exec_sd }
        OP_PREF   => MipsExecutor::<T,C>::exec_pref,
        OP_LLD    => { ins.set_imm_se(raw); MipsExecutor::<T,C>::exec_lld }
        OP_SCD    => { ins.set_imm_se(raw); MipsExecutor::<T,C>::exec_scd }
        _         => MipsExecutor::<T,C>::exec_reserved,
    };

    ins.op      = op;
    ins.rs      = rs;
    ins.rt      = rt;
    ins.rd      = rd;
    ins.sa      = sa;
    ins.funct   = funct;
    ins.handler = handler as usize;
    ins.flags   = 0; // decoded; clears FLAG_IMM_IS_NEXT too (always consumed by now)
}

// Field extraction helpers have been replaced by DecodedInstr fields

// Helper to format PC with symbol
fn format_pc_symbol(pc: u64, symbols: &SymbolTable) -> String {
    let mut lookup = symbols.lookup(pc);
    let mut effective_pc = pc;
    
    // If not found and address is KSEG1 (0xFFFFFFFF_A...), try KSEG0 (0xFFFFFFFF_8...)
    if lookup.is_none() && (pc >> 32) == 0xFFFFFFFF && ((pc >> 29) & 0x7) == 5 {
        let kseg0_pc = (pc & 0x1FFFFFFF) | 0xFFFF_FFFF_8000_0000;
        if let Some(res) = symbols.lookup(kseg0_pc) {
            lookup = Some(res);
            effective_pc = kseg0_pc;
        }
    }

    if let Some((sym_addr, name)) = lookup {
        let offset = effective_pc - sym_addr;
        if offset > 256 {
            return String::new();
        }
        if offset == 0 {
            return format!(" <{}>", name);
        } else {
            return format!(" <{}+0x{:x}>", name, offset);
        }
    }
    String::new()
}

// Helper to parse command arguments (registers or values)
fn parse_reg_name(arg: &str) -> Option<usize> {
    match exp::parse_reg_target(arg) {
        Some(RegTarget::Gpr(n)) => Some(n as usize),
        _ => None,
    }
}

/// Copy `data` to `vaddr` and zero-fill out to `memsz`. Writes go through the
/// CPU's virtual-address path, so KSEG0/KSEG1 mapping, the MC address mask and
/// bank remapping all apply — never poke `Memory` behind the bus.
fn load_range<T: Tlb, C: MipsCache>(
    exec: &mut MipsExecutor<T, C>,
    vaddr: u64,
    data: &[u8],
    memsz: u64,
) -> Result<(), String> {
    if memsz == 0 {
        return Ok(());
    }
    probe_mapped(exec, vaddr)?;
    probe_mapped(exec, vaddr.wrapping_add(memsz - 1))?;

    let mut off = 0u64;
    while off < memsz {
        let addr = vaddr.wrapping_add(off);
        let byte = |k: u64| *data.get((off + k) as usize).unwrap_or(&0) as u64;
        let (size, val) = if memsz - off >= 4 && addr % 4 == 0 {
            (4usize, (byte(0) << 24) | (byte(1) << 16) | (byte(2) << 8) | byte(3))
        } else {
            (1usize, byte(0))
        };
        let status = exec.debug_write(addr, val, size, full_mask_for_usize(size));
        if status != EXEC_COMPLETE {
            return Err(format!(
                "write failed at {:#018x} (status {:#010x}) — is RAM mapped and addressable there?",
                addr, status
            ));
        }
        off += size as u64;
    }
    invalidate_loaded_range(exec, vaddr, memsz);
    Ok(())
}

/// Unmapped physical addresses are backed by `UnmappedRam`, which accepts every
/// write and reads back zero — so a write into the void is completely silent.
/// Probe with a non-zero pattern before committing an image.
///
/// The probe goes to the bus, not through `debug_write`: KSEG0 is cacheable, so
/// a cached write is absorbed by L1D and reads back fine with nothing behind it.
/// The original word is put back, so this leaves no trace.
fn probe_mapped<T: Tlb, C: MipsCache>(exec: &mut MipsExecutor<T, C>, vaddr: u64) -> Result<(), String> {
    let tr = exec.debug_translate(vaddr);
    if tr.is_exception() {
        return Err(format!("{:#018x} does not translate (status {:#010x})", vaddr, tr.status));
    }
    let phys = tr.phys;
    let unmapped = || {
        Err(format!(
            "nothing is mapped at {:#018x} (physical {:#010x}) — writes there go nowhere. \
             The PROM maps RAM during POST: boot to the PROM prompt first, or use --load-elf, \
             which maps the banks itself",
            vaddr, phys
        ))
    };

    const PATTERN: u32 = 0x5A5A_A5A5;
    if phys % 4 == 0 {
        let saved = exec.sysad.read32(phys);
        if !saved.is_ok() || exec.sysad.write32(phys, PATTERN) != BUS_OK {
            return unmapped();
        }
        let got = exec.sysad.read32(phys);
        exec.sysad.write32(phys, saved.data);
        if !got.is_ok() || got.data != PATTERN {
            return unmapped();
        }
    } else {
        let saved = exec.sysad.read8(phys);
        if !saved.is_ok() || exec.sysad.write8(phys, PATTERN as u8) != BUS_OK {
            return unmapped();
        }
        let got = exec.sysad.read8(phys);
        exec.sysad.write8(phys, saved.data);
        if !got.is_ok() || got.data != PATTERN as u8 {
            return unmapped();
        }
    }
    Ok(())
}

/// Drop stale cache copies of a just-loaded range: writeback+invalidate L1D and
/// L2 so the new bytes reach memory, then invalidate L1I so a stale decoded
/// instruction can never execute. The v1 JIT needs no explicit flush (its
/// CodeCache dies with the CPU thread, and loading requires a stopped CPU);
/// jitv2 self-invalidates from the per-page generation counter the writes bump.
fn invalidate_loaded_range<T: Tlb, C: MipsCache>(exec: &mut MipsExecutor<T, C>, vaddr: u64, len: u64) {
    let (_, iline) = exec.cache.get_config(CACH_PI);
    let (_, dline) = exec.cache.get_config(CACH_PD);
    let step = (iline.min(dline).max(4)) as u64;
    let mut addr = vaddr & !(step - 1);
    let end = vaddr.wrapping_add(len);
    while addr < end {
        let tr = exec.debug_translate(addr);
        if !tr.is_exception() {
            let phys = tr.phys as u64;
            exec.cache.cache_op(C_HWBINV | CACH_PD, addr, phys);
            exec.cache.cache_op(C_HWBINV | CACH_SD, addr, phys);
            exec.cache.cache_op(C_HINV | CACH_PI, addr, phys);
        }
        addr = addr.wrapping_add(step);
    }
}

fn parse_cpu_arg(arg: &str, core: &MipsCore, symbols: Option<&SymbolTable>) -> Result<u64, String> {
    exp::parse_and_eval(arg, core, symbols)
}

fn decode_status(val: u32) -> String {
    let mut s = String::new();
    s.push_str("CU:");
    for i in (0..4).rev() {
        if (val & (1 << (28 + i))) != 0 { s.push_str(&format!("{}", i)); } else { s.push('_'); }
    }
    
    if (val & STATUS_RP) != 0 { s.push_str(" RP"); }
    if (val & STATUS_FR) != 0 { s.push_str(" FR"); }
    if (val & STATUS_RE) != 0 { s.push_str(" RE"); }
    if (val & STATUS_BEV) != 0 { s.push_str(" BEV"); }
    if (val & STATUS_TS) != 0 { s.push_str(" TS"); }
    if (val & STATUS_SR) != 0 { s.push_str(" SR"); }
    if (val & STATUS_CH) != 0 { s.push_str(" CH"); }
    if (val & STATUS_CE) != 0 { s.push_str(" CE"); }
    if (val & STATUS_DE) != 0 { s.push_str(" DE"); }

    s.push_str(" IM:");
    for i in (0..8).rev() {
        if (val & (1 << (8 + i))) != 0 { s.push_str(&format!("{}", i)); } else { s.push('_'); }
    }

    if (val & STATUS_KX) != 0 { s.push_str(" KX"); }
    if (val & STATUS_SX) != 0 { s.push_str(" SX"); }
    if (val & STATUS_UX) != 0 { s.push_str(" UX"); }

    let ksu = (val >> STATUS_KSU_SHIFT) & 3;
    match ksu {
        0 => s.push_str(" K:K"),
        1 => s.push_str(" K:S"),
        2 => s.push_str(" K:U"),
        _ => s.push_str(" K:?"),
    }

    if (val & STATUS_ERL) != 0 { s.push_str(" ERL"); }
    if (val & STATUS_EXL) != 0 { s.push_str(" EXL"); }
    if (val & STATUS_IE) != 0 { s.push_str(" IE"); }

    s
}

fn decode_cause(val: u32) -> String {
    let mut s = String::new();
    if (val & CAUSE_BD) != 0 { s.push_str("BD "); }
    
    let ce = (val >> CAUSE_CE_SHIFT) & 3;
    if ce != 0 { s.push_str(&format!("CE:{} ", ce)); }

    s.push_str("IP:");
    for i in (0..8).rev() {
        if (val & (1 << (8 + i))) != 0 { s.push_str(&format!("{}", i)); } else { s.push('_'); }
    }

    let exc = (val >> CAUSE_EXCCODE_SHIFT) & 0x1F;
    let exc_name = match exc {
        EXC_INT => "INT",
        EXC_MOD => "MOD",
        EXC_TLBL => "TLBL",
        EXC_TLBS => "TLBS",
        EXC_ADEL => "ADEL",
        EXC_ADES => "ADES",
        EXC_IBE => "IBE",
        EXC_DBE => "DBE",
        EXC_SYS => "SYS",
        EXC_BP => "BP",
        EXC_RI => "RI",
        EXC_CPU => "CPU",
        EXC_OV => "OV",
        EXC_TR => "TR",
        EXC_FPE => "FPE",
        EXC_WATCH => "WATCH",
        _ => "?",
    };
    s.push_str(&format!(" Exc:{:02x}({})", exc, exc_name));

    s
}

/// MipsCpu wrapper for threaded execution and monitor control
pub struct MipsCpu<T: Tlb, C: MipsCache> {
    executor: Arc<Mutex<MipsExecutor<T, C>>>,
    running: Arc<AtomicBool>,
    thread: Mutex<Option<thread::JoinHandle<()>>>,
    /// Pointer into the executor's `MipsCore.hot.cycles` — see
    /// `CyclesPtr`/`Hot::cycles`'s doc comments. Same process-lifetime
    /// validity argument as `interrupts_ptr` right below.
    cycles_ptr: crate::mips_core::CyclesPtr,
    /// Raw pointer into the executor's `MipsCore.interrupts` (an inline
    /// field, not `Arc<AtomicU64>` — see that field's doc comment). Valid
    /// for the process lifetime: `executor` below is the only owner of the
    /// `MipsCore` this points into, and it's never dropped before shutdown.
    interrupts_ptr: *const AtomicU64,
    pub fasttick_count: Arc<AtomicU64>,
    debug: Arc<AtomicBool>,
    exception_mask: Arc<AtomicU32>,
    trace_file: Arc<Mutex<Option<std::io::BufWriter<std::fs::File>>>>,
    #[cfg(feature = "idle-pause")]
    idle_profile_on: Arc<AtomicBool>,
    #[cfg(feature = "idle-pause")]
    idle_profile_reset: Arc<AtomicBool>,
}

// Safety: interrupts_ptr points into the MipsCore owned by `executor`
// (Arc<Mutex<MipsExecutor>>), which outlives every MipsCpu clone/thread that
// might read this pointer — the whole struct is already Send/Sync via its
// other Arc fields; this one raw pointer needs the same guarantee spelled
// out explicitly since raw pointers don't get it automatically.
unsafe impl<T: Tlb, C: MipsCache> Send for MipsCpu<T, C> {}
unsafe impl<T: Tlb, C: MipsCache> Sync for MipsCpu<T, C> {}

impl<T: Tlb + Send + 'static, C: MipsCache + Send + 'static> MipsCpu<T, C> {
    pub fn new(executor: MipsExecutor<T, C>) -> Self {
        let fasttick_count = executor.core.fasttick_count.clone();
        #[cfg(feature = "idle-pause")]
        let idle_profile_on = executor.idle_profile_on.clone();
        #[cfg(feature = "idle-pause")]
        let idle_profile_reset = executor.idle_profile_reset.clone();

        let executor_arc = Arc::new(Mutex::new(executor));
        executor_arc.lock().install_status_cb();
        // MipsCore is now at its final, stable address (inside the Arc) —
        // safe to take raw pointers into it that outlive this constructor.
        let interrupts_ptr = executor_arc.lock().interrupts_ptr();
        let cycles_ptr = executor_arc.lock().cycles_ptr();
        #[cfg(feature = "jitv2")]
        executor_arc.lock().install_jit_hooks();

        Self {
            executor: executor_arc,
            running: Arc::new(AtomicBool::new(false)),
            thread: Mutex::new(None),
            cycles_ptr,
            interrupts_ptr,
            fasttick_count,
            debug: Arc::new(AtomicBool::new(false)),
            exception_mask: Arc::new(AtomicU32::new(0)),
            trace_file: Arc::new(Mutex::new(None)),
            #[cfg(feature = "idle-pause")]
            idle_profile_on,
            #[cfg(feature = "idle-pause")]
            idle_profile_reset,
        }
    }

    /// Arm the idle-loop PC sampler without taking the executor lock (so the
    /// running CPU is never paused/resumed). Requests a histogram reset which
    /// the CPU thread performs on its next step.
    #[cfg(feature = "idle-pause")]
    pub fn idle_profile_arm(&self) {
        self.idle_profile_reset.store(true, Ordering::SeqCst);
        self.idle_profile_on.store(true, Ordering::SeqCst);
    }

    /// Disarm the sampler (lock-free).
    #[cfg(feature = "idle-pause")]
    pub fn idle_profile_disarm(&self) {
        self.idle_profile_on.store(false, Ordering::SeqCst);
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Pointer to the executor's `MipsCore.hot.cycles` word — see
    /// `CyclesPtr`/`Hot::cycles`'s doc comments (`cycles` lives inline on
    /// `MipsCore` now, not `Arc<AtomicU64>`, so this can no longer hand out a
    /// cloneable `Arc`). Same process-lifetime validity argument as
    /// `interrupts_ptr` (fixed for the life of this `MipsCpu`, set once in
    /// `new()` after the executor reached its final address).
    /// Enable the cheritest CP0-26 dump hook (`--cheritest-dump-hook`).
    pub fn set_cheritest_dump_hook(&self, on: bool) {
        self.executor.lock().cheritest_dump_hook = on;
    }

    /// Pointer to the executor's `MipsCore`, for devices that report CPU state
    /// from the CPU thread itself (the test device's DUMP). Same
    /// process-lifetime validity argument as `cycles_ptr`.
    pub fn core_ptr(&self) -> *const crate::mips_core::MipsCore {
        &self.executor.lock().core as *const crate::mips_core::MipsCore
    }

    pub fn cycles_ptr(&self) -> crate::mips_core::CyclesPtr {
        self.cycles_ptr
    }

    /// Raw pointer to the executor's `MipsCore.interrupts` word, for wiring
    /// into devices on other threads that set/clear interrupt bits (e.g.
    /// `Ioc::set_interrupts`). Fixed for the life of this `MipsCpu` — set
    /// once in `new()` after the executor reached its final address.
    pub fn interrupts_ptr(&self) -> *const AtomicU64 {
        self.interrupts_ptr
    }

    pub fn running_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.running)
    }

    /// Shared jitv2 engine state (page pool + compile queue + shared
    /// `Codegen`) — narrow accessor so `Machine::new` can inject the
    /// `set_cpu`/`set_owner` handles the compile-thread worker needs
    /// (`Jitv2::codegen`/`CompileQueue`'s doc comments) without needing
    /// broader access to the executor itself.
    #[cfg(feature = "jitv2")]
    pub fn jitv2(&self) -> Arc<Mutex<crate::jitv2::Jitv2>> {
        self.executor.lock().jitv2.clone()
    }

    pub fn run_debug_loop(&self, mut count: Option<usize>, wait: bool, mut writer: Box<dyn Write + Send>) {
        self.stop(); // Ensure stopped before running

        self.running.store(true, Ordering::SeqCst);

        let executor = self.executor.clone();
        let running = self.running.clone();
        let debug = self.debug.clone();
        let exception_mask = self.exception_mask.clone();
        let trace_file = self.trace_file.clone();

        let task = move || {
            let mut exec = executor.lock();

            // Same reasoning as MipsCpu::start(): this closure runs on its own
            // fresh OS thread ("MIPS-Debug"), so the host FPU rounding mode
            // needs to be re-synced from the guest's tracked FCSR.RM rather
            // than assuming the platform default matches.
            crate::platform::set_fpu_mode((exec.core.fpu_fcsr & 0x3) as u8);

            // The CPU counts as running for the duration of this loop: resume
            // the virtual CP0 Count and compare timer (latched by the stop()
            // above), and re-latch on the way out. For a long `run` this
            // gives the guest live timer ticks; for a single `step` the
            // running window is microseconds, so Count stays effectively
            // frozen between steps and IP7 never fires mid-inspection (use
            // the `ip7` command to inject one deliberately).
            exec.core.on_cpu_start();

            if !wait {
                let _ = writeln!(writer, "Running...");
            }

            let mut first_step = true;
            let mut steps_since_yield = 0;

            loop {
                if !running.load(Ordering::Relaxed) {
                    writeln!(writer, "Interrupted").unwrap();
                    break;
                }

                // Check step count
                if let Some(c) = count {
                    if c == 0 { break; }
                    count = Some(c - 1);
                }

                // Capture snapshot BEFORE executing — this is the state to restore on undo.
                // memory_writes are collected during step() into pending_memory_writes, then
                // patched into the snapshot afterwards so a single undo entry is self-contained.
                #[cfg(feature = "developer")]
                let undo_snap_idx = if exec.undo_buffer.is_enabled() {
                    exec.pending_memory_writes.clear();
                    let snapshot = exec.create_snapshot();
                    let idx = exec.undo_buffer.push_get_idx(snapshot);
                    Some(idx)
                } else {
                    None
                };

                // Try step with breakpoints enabled.
                // step() now pushes (pc, instr) into traceback on successful fetch.
                let mut status = exec.step();

                if status == EXEC_BREAKPOINT {
                    // If we hit the temporary breakpoint (ID 0), stop immediately
                    if exec.last_bp_hit == Some(0) {
                        // Don't step over, just let the match below handle the break
                    } else if first_step {
                        // If we hit a user breakpoint right at the start of a command,
                        // step over it to resume execution.
                        exec.skip_breakpoints = true;
                        status = exec.step();
                    }
                }

                if first_step {
                    first_step = false;
                }

                // Patch the memory writes collected during step() into the pre-step snapshot.
                #[cfg(feature = "developer")]
                if let Some(idx) = undo_snap_idx {
                    let writes = std::mem::take(&mut exec.pending_memory_writes);
                    if let Some(ref mut snap) = exec.undo_buffer.snapshots[idx] {
                        snap.memory_writes = writes;
                    }
                }

                // Display executed instruction from traceback (already captured by step())
                let insn_trace = debug.load(Ordering::Relaxed) || mips_log(MIPS_LOG_INSN);
                if insn_trace || count.is_some() || trace_file.lock().is_some() {
                    if let Some(entry) = exec.traceback.get_last(1).into_iter().next() {
                        let symbols = exec.symbols.lock();
                        let sym_str = format_pc_symbol(entry.pc, &symbols);
                        let dis = mips_dis::disassemble(entry.instr, entry.pc, Some(&symbols));
                        let info = format!("{:#018x}{}: {:#010x} {}", entry.pc, sym_str, entry.instr, dis);
                        if insn_trace {
                            dlog_dev!(LogModule::Mips, "{}", info);
                            writeln!(writer, "{}", info).unwrap();
                        }
                        if count.is_some() {
                            // Route through devlog writers so Exec: lines serialize with
                            // cache/device dlog output on the same TCP socket.
                            crate::dlog_unconditional!("Exec: {}", info);
                        }
                        if let Some(ref mut f) = *trace_file.lock() {
                            let _ = writeln!(f, "{}", info);
                        }
                    } else if insn_trace {
                        writeln!(writer, "PC: {:016x} (Fetch failed)", exec.core.pc).unwrap();
                    }
                }

                let pc = exec.core.pc;
                match status {
                    EXEC_RETRY => {
                        writeln!(writer, "PC={:016x}: Retry (Bus Busy)", pc).unwrap();
                        break;
                    }
                    s if s & EXEC_IS_EXCEPTION != 0 && s & EXEC_IS_TLB_REFILL == 0 => {
                        let code = (s >> crate::mips_core::CAUSE_EXCCODE_SHIFT) & 0x1F;
                        let mask = exception_mask.load(Ordering::Relaxed);
                        if (mask & (1 << code)) != 0 {
                            writeln!(writer, "PC={:016x}: Exception code={}", pc, code).unwrap();
                            break;
                        }
                    }
                    s if s & EXEC_IS_EXCEPTION != 0 && s & EXEC_IS_TLB_REFILL != 0 => {
                        let code = (s >> crate::mips_core::CAUSE_EXCCODE_SHIFT) & 0x1F;
                        let mask = exception_mask.load(Ordering::Relaxed);
                        if (mask & (1 << code)) != 0 {
                            writeln!(writer, "PC={:016x}: TLB Miss code={}", pc, code).unwrap();
                            break;
                        }
                    }
                    EXEC_BREAKPOINT => {
                        if let Some(bp_id) = exec.last_bp_hit {
                            if bp_id != 0 {
                                writeln!(writer, "PC={:016x}: Breakpoint {} hit", pc, bp_id).unwrap();
                            }
                        } else {
                            writeln!(writer, "PC={:016x}: Breakpoint hit", pc).unwrap();
                        }
                        break;
                    }
                    _ => {}
                }

                steps_since_yield += 1;
                if steps_since_yield >= 500000 {
                    steps_since_yield = 0;
                    drop(exec);
                    thread::sleep(Duration::from_millis(1));
                    exec = executor.lock();

                    if !running.load(Ordering::Relaxed) {
                        writeln!(writer, "Interrupted").unwrap();
                        break;
                    }
                }
            }
            #[cfg(feature = "developer")]
            { exec.skip_interrupts = false; }
            if let Some(ref mut f) = *trace_file.lock() { let _ = f.flush(); }

            // Print next instruction
            let next_pc = exec.core.pc;
            match exec.debug_fetch_instr(next_pc) {
                 Ok(instr) => {
                     let symbols = exec.symbols.lock();
                     let sym_str = format_pc_symbol(next_pc, &symbols);
                     let dis = mips_dis::disassemble(instr, next_pc, Some(&symbols));
                     writeln!(writer, "Next: {:016x}{}: {:08x} {}", next_pc, sym_str, instr, dis).unwrap();
                 }
                 Err(_) => {
                     writeln!(writer, "Next: {:016x} (Fetch failed)", next_pc).unwrap();
                 }
            }

            // Clear temporary breakpoint (used by run/finish)
            exec.clear_temp_breakpoint();

            exec.core.on_cpu_stop();
            drop(exec);
            running.store(false, Ordering::SeqCst);
        };

        let handle = thread::Builder::new().name("MIPS-Debug".to_string()).spawn(task).unwrap();

        if wait {
            let _ = handle.join();
        } else {
            *self.thread.lock() = Some(handle);
        }
    }

    pub fn register_locks(&self) {
        use crate::locks::register_lock_fn;
        let ex = self.executor.clone();
        register_lock_fn("cpu::executor", move || ex.is_locked());
    }

    fn try_lock_executor(&self) -> Result<parking_lot::MutexGuard<'_, MipsExecutor<T, C>>, String> {
        self.executor.try_lock().ok_or_else(|| "CPU thread holds the executor lock; try 'cpu stop' first".to_string())
    }

    /// Load a static ELF32 MSB image: copy each `PT_LOAD` to its `p_vaddr`,
    /// zero-fill to `p_memsz`, set PC to `e_entry`. Returns a segment summary.
    pub fn load_elf(&self, path: &str) -> Result<String, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {}", path, e))?;
        let elf = crate::elf::parse(&bytes).map_err(|e| format!("{}: {}", path, e))?;
        self.check_stopped()?;
        let mut exec = self.try_lock_executor()?;
        let mut out = String::new();
        for s in &elf.segments {
            load_range(&mut exec, s.vaddr, &bytes[s.offset..s.offset + s.filesz], s.memsz)?;
            out.push_str(&format!(
                "  {:#018x}  filesz {:#x}  memsz {:#x}  {}\n",
                s.vaddr, s.filesz, s.memsz, crate::elf::flag_str(s.flags)
            ));
        }
        exec.core.pc = elf.entry;
        out.push_str(&format!("  entry {:#018x}\n", elf.entry));
        Ok(out)
    }

    /// Load raw bytes at a virtual address. PC is not touched.
    pub fn load_bin(&self, path: &str, vaddr: u64) -> Result<String, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {}", path, e))?;
        self.check_stopped()?;
        let mut exec = self.try_lock_executor()?;
        load_range(&mut exec, vaddr, &bytes, bytes.len() as u64)?;
        Ok(format!("  {:#018x}  {} bytes\n", vaddr, bytes.len()))
    }

    /// Loading over live code races the running dispatch loop, and the v1 JIT's
    /// CodeCache (owned by `run_jit_dispatch`, dropped when the CPU thread exits)
    /// would keep translations of the code we just replaced.
    fn check_stopped(&self) -> Result<(), String> {
        if self.is_running() {
            return Err("CPU is running — 'cpu stop' first".to_string());
        }
        Ok(())
    }

    /// Step the executor `n` times in-line on the calling thread. Caller must
    /// have stopped the runtime CPU thread first (otherwise we deadlock on
    /// the executor mutex). Returns the number of steps actually executed —
    /// will be `< n` only if the CPU stops itself (e.g. soft-reset).
    ///
    /// Used by Phase 3.3 snapshot determinism validator. Single-threaded,
    /// no thread scheduling jitter, so two runs from identical state should
    /// reach identical state after the same number of steps.
    pub fn step_n_inline(&self, n: u64) -> Result<u64, String> {
        let mut exec = self.try_lock_executor()?;
        let mut executed = 0u64;
        for _ in 0..n {
            let _status = exec.step();
            executed += 1;
            // Don't break on exceptions — they're part of normal CPU
            // operation and a deterministic run should re-enter and continue.
        }
        Ok(executed)
    }

    /// Step exactly one `step()` call and return how many architectural
    /// instructions it actually retired, per `core.hot.cycles`' delta.
    /// Ordinarily 1 (the interpreter's `step()` always retires exactly one
    /// instruction), but a real JIT-compiled unit can retire 2+ in a single
    /// `step()` call — a branch/jump's compiled unit always inlines its
    /// delay slot (§6.1.4), so one call there covers both. Callers doing
    /// their own instruction-by-instruction accounting against a *different*
    /// engine's per-instruction reference trace (`validate::validate_jit_determinism`,
    /// the engine behind the monitor's `jitcheck <n>` command) need this,
    /// not `step_n_inline`'s step()-call count, to stay correctly aligned
    /// when JIT dispatch is involved.
    #[cfg(feature = "developer")]
    pub fn step_one_inline_counting_instructions(&self) -> Result<usize, String> {
        let mut exec = self.try_lock_executor()?;
        let cycles_before = exec.core.hot.cycles;
        let status = exec.step();
        exec.last_step_status = status;
        let retired = exec.core.hot.cycles.wrapping_sub(cycles_before) as usize;
        Ok(retired.max(1)) // defensive: never 0, which would stall any caller looping on this
    }

    /// Runtime kill switch for `exec_decoded`'s real JIT dispatch gate — see
    /// `MipsExecutor::jitv2_dispatch_enabled`'s doc comment. Returns the
    /// previous value so callers can restore it afterward.
    #[cfg(all(feature = "jitv2", feature = "developer"))]
    pub fn set_jitv2_dispatch_enabled(&self, enabled: bool) -> Result<bool, String> {
        let mut exec = self.try_lock_executor()?;
        let prev = exec.jitv2_dispatch_enabled;
        exec.jitv2_dispatch_enabled = enabled;
        Ok(prev)
    }

    /// Arm/disarm `jitcheck`'s hardware-read fixup recording — see
    /// `MipsExecutor::hw_read_fixup_recording`'s doc comment. Also clears
    /// any stale recorded-but-undrained entries when disarming, so a
    /// half-finished record pass never leaks into a later digest.
    #[cfg(feature = "developer")]
    pub fn set_hw_read_fixup_recording(&self, recording: bool) -> Result<(), String> {
        let mut exec = self.try_lock_executor()?;
        exec.hw_read_fixup_recording = recording;
        if !recording {
            exec.hw_read_fixup_recorded.clear();
        }
        Ok(())
    }

    /// Set/clear the recorded hardware-read values `read_data_impl`'s replay
    /// branch should substitute for the *next* step — see
    /// `MipsExecutor::hw_read_fixup_replay`'s doc comment. Caller sets this
    /// to the reference pass's digest for the step about to run, each step,
    /// for the whole replay pass; `None` disables the fixup entirely.
    #[cfg(feature = "developer")]
    pub fn set_hw_read_fixup_replay(&self, fixups: Option<Vec<(u64, u8, u64)>>) -> Result<(), String> {
        let mut exec = self.try_lock_executor()?;
        exec.hw_read_fixup_replay = fixups;
        Ok(())
    }

    /// Snapshot the deterministic-from-state CPU registers. Excludes host
    /// wallclock anchors like `count_anchor_instant` (they're meaningless
    /// across runs) but includes their calibrated equivalents (count_hz,
    /// compare_delta_*).
    ///
    /// Also drains `hw_read_fixup_recorded` (see that field's doc comment)
    /// into the returned digest's `hw_reads` and clears it — empty/no-op
    /// whenever `jitcheck`'s fixup recording isn't armed, so this stays free
    /// for every other `state_digest` caller. `step_status` mirrors
    /// `last_step_status` (also empty/`EXEC_COMPLETE`-only-by-coincidence
    /// outside a `step_one_inline_counting_instructions` caller like
    /// `jitcheck` — see that field's doc comment).
    pub fn state_digest(&self) -> Result<CpuStateDigest, String> {
        #[cfg(feature = "developer")]
        let mut exec = self.try_lock_executor()?;
        #[cfg(not(feature = "developer"))]
        let exec = self.try_lock_executor()?;
        #[cfg(feature = "developer")]
        let hw_reads = std::mem::take(&mut exec.hw_read_fixup_recorded);
        #[cfg(not(feature = "developer"))]
        let hw_reads = Vec::new();
        #[cfg(feature = "developer")]
        let step_status = exec.last_step_status;
        #[cfg(not(feature = "developer"))]
        let step_status = 0;
        let c = &exec.core;
        Ok(CpuStateDigest {
            gpr: c.gpr,
            pc: c.pc,
            hi: c.hi,
            lo: c.lo,
            cp0_count: c.cp0_count,
            cp0_compare: c.cp0_compare,
            cp0_status: c.cp0_status,
            cp0_cause: c.cp0_cause,
            cp0_epc: c.cp0_epc,
            cp0_badvaddr: c.cp0_badvaddr,
            cp0_entryhi: c.cp0_entryhi,
            count_hz: c.count_hz,
            in_delay_slot: c.in_delay_slot,
            hw_reads,
            step_status,
        })
    }

    /// Restore the CPU-register subset captured by `state_digest` — used by
    /// `validate::validate_jit_determinism`'s reconverge step: after
    /// detecting a divergence, force the JIT-dispatch pass back onto the
    /// interpreter-only reference pass's ground-truth register state so the
    /// rest of the comparison window stays meaningful instead of running
    /// two engines through completely unrelated code. Registers/cop0 only —
    /// does not touch memory (a `CpuStateDigest` doesn't carry any), FPU
    /// registers, or device state; see `validate_jit_determinism`'s doc
    /// comment for why that's an accepted limitation here.
    #[cfg(feature = "developer")]
    pub fn restore_state_digest(&self, digest: &CpuStateDigest) -> Result<(), String> {
        let mut exec = self.try_lock_executor()?;
        exec.core.gpr = digest.gpr;
        exec.core.pc = digest.pc;
        exec.core.hi = digest.hi;
        exec.core.lo = digest.lo;
        exec.core.cp0_count = digest.cp0_count;
        exec.core.cp0_compare = digest.cp0_compare;
        exec.core.cp0_status = digest.cp0_status;
        exec.core.cp0_cause = digest.cp0_cause;
        exec.core.cp0_epc = digest.cp0_epc;
        exec.core.cp0_badvaddr = digest.cp0_badvaddr;
        exec.core.cp0_entryhi = digest.cp0_entryhi;
        exec.core.count_hz = digest.count_hz;
        exec.core.in_delay_slot = digest.in_delay_slot;
        exec.core.reanchor_count_and_reschedule();
        exec.on_cp0_status_changed(0, digest.cp0_status);
        Ok(())
    }

    /// Force `cp0_count`/`count_hz` to a reference digest's values —
    /// narrower than `restore_state_digest`, and unconditional rather than
    /// reserved for a detected divergence: `validate::validate_jit_determinism`
    /// calls this after *every* replay-pass step, not just when these fields
    /// show up in a diff. They are excluded from that diff comparison
    /// entirely (see that function's doc comment: two separately wall-clock
    /// timed passes legitimately disagree here with zero JIT bug involved),
    /// but a diverged `cp0_count` left in place would still be wrong,
    /// observable CPU state — anything the replayed program itself reads
    /// via `mfc0 $9` (Count) sees the JIT pass's own drifted value instead
    /// of the reference's, poisoning every later comparison that depends on
    /// it (e.g. a timing loop storing Count to a GPR that then gets
    /// compared). Forcing these two fields back to ground truth after every
    /// step keeps the rest of the diff meaningful without having to treat a
    /// whole class of downstream GPR diffs as further "known benign" noise.
    #[cfg(feature = "developer")]
    pub fn fixup_cp0_count(&self, digest: &CpuStateDigest) -> Result<(), String> {
        let mut exec = self.try_lock_executor()?;
        exec.core.cp0_count = digest.cp0_count;
        exec.core.count_hz = digest.count_hz;
        exec.core.reanchor_count_and_reschedule();
        Ok(())
    }
}

/// Deterministic-from-state CPU register snapshot. Excludes host wallclock
/// anchors so two runs from the same starting state can be diffed cleanly.
/// `cycles` is intentionally not included — it's a runtime perf counter
/// that's not part of save_state and stays stale across `load_snapshot`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuStateDigest {
    pub gpr: [u64; 32],
    pub pc: u64,
    pub hi: u64,
    pub lo: u64,
    pub cp0_count: u64,
    pub cp0_compare: u64,
    pub cp0_status: u32,
    pub cp0_cause: u32,
    pub cp0_epc: u64,
    pub cp0_badvaddr: u64,
    pub cp0_entryhi: u64,
    pub count_hz: u64,
    pub in_delay_slot: bool,
    /// `jitcheck`'s hardware-read fixup: `(phys_addr, size, value)` for every
    /// real bus read this step made from an address in `HW_READ_FIXUP_ADDRS`
    /// (e.g. the MC's RPSS_CTR — see that const's doc comment). Empty
    /// outside `validate_jit_determinism`'s reference pass. Not itself
    /// compared by `diff` (see `diff`'s own note) — it's an input the replay
    /// pass consumes, not a piece of CPU state to check for equality.
    pub hw_reads: Vec<(u64, u8, u64)>,
    /// The raw `ExecStatus` the step that produced this digest returned
    /// (`MipsExecutor::last_step_status`, set by
    /// `step_one_inline_counting_instructions`). Compared by `diff` like any
    /// other field — the direct way to see "did this engine's step take an
    /// exception (and which one) that the other engine's step didn't",
    /// rather than inferring it indirectly from EXL/EPC/Cause alone. `0`
    /// (not a valid `ExecStatus`) outside a `jitcheck`-style caller.
    pub step_status: u32,
}

impl CpuStateDigest {
    /// Return a list of (field_name, lhs_repr, rhs_repr) for every field that
    /// differs. Empty if states are bit-identical. For arrays, only diverging
    /// indices are reported.
    pub fn diff(&self, other: &CpuStateDigest) -> Vec<(String, String, String)> {
        let mut out = Vec::new();
        for (i, (a, b)) in self.gpr.iter().zip(other.gpr.iter()).enumerate() {
            if a != b {
                out.push((format!("gpr[{}]", i), format!("0x{:016x}", a), format!("0x{:016x}", b)));
            }
        }
        macro_rules! cmp {
            ($name:ident, $fmt:expr) => {
                if self.$name != other.$name {
                    out.push((stringify!($name).to_string(), format!($fmt, self.$name), format!($fmt, other.$name)));
                }
            };
        }
        cmp!(pc,           "0x{:016x}");
        cmp!(hi,           "0x{:016x}");
        cmp!(lo,           "0x{:016x}");
        cmp!(cp0_count,    "0x{:016x}");
        cmp!(cp0_compare,  "0x{:016x}");
        cmp!(cp0_status,   "0x{:08x}");
        cmp!(cp0_cause,    "0x{:08x}");
        cmp!(cp0_epc,      "0x{:016x}");
        cmp!(cp0_badvaddr, "0x{:016x}");
        cmp!(cp0_entryhi,  "0x{:016x}");
        cmp!(count_hz,     "{}");
        cmp!(in_delay_slot, "{}");
        cmp!(step_status,  "0x{:08x}");
        out
    }
}

fn is_call_instruction(instr: u32) -> bool {
    let op = (instr >> 26) & 0x3F;
    match op {
        OP_JAL => true,
        OP_SPECIAL => {
            let funct = instr & 0x3F;
            funct == FUNCT_JALR
        }
        OP_REGIMM => {
            let rt = (instr >> 16) & 0x1F;
            rt == RT_BGEZAL || rt == RT_BLTZAL || rt == RT_BGEZALL || rt == RT_BLTZALL
        }
        _ => false,
    }
}

impl<T: Tlb + Send + 'static, C: MipsCache + Send + 'static> Device for MipsCpu<T, C> {
    fn step(&self, cycles: u64) {
        let mut exec = self.executor.lock();
        for _ in 0..cycles {
            exec.step();
        }
    }

    fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        if let Some(handle) = self.thread.lock().take() {
            let _ = handle.join();
        }
        // Latch the virtual CP0 Count and silence the compare timer while
        // stopped — debugger stepping must not see Count advance or IP7
        // fire underneath it (mips_core.rs on_cpu_stop's doc comment).
        self.executor.lock().core.on_cpu_stop();
        #[cfg(feature = "tlbstats")]
        self.executor.lock().tlb.stats_print();
        #[cfg(feature = "instr_stats")]
        {
            let exec = self.executor.lock();
            exec.instr_stats.print();
            // Diffable dumps: plain used-instruction list and mnemonic-sorted counts,
            // written alongside the stderr report so two runs' coverage/frequency can be
            // compared with a plain `diff` (write_report's stderr output is sorted by
            // frequency, which reorders on every run and isn't diff-friendly).
            match std::fs::File::create("instr_used.txt") {
                Ok(f) => {
                    let mut w = std::io::BufWriter::new(f);
                    let _ = exec.instr_stats.write_used_list(&mut w);
                }
                Err(e) => eprintln!("instr_stats: cannot open instr_used.txt: {}", e),
            }
            match std::fs::File::create("instr_counts.txt") {
                Ok(f) => {
                    let mut w = std::io::BufWriter::new(f);
                    let _ = exec.instr_stats.write_counts(&mut w);
                }
                Err(e) => eprintln!("instr_stats: cannot open instr_counts.txt: {}", e),
            }
        }
        #[cfg(feature = "jitv2")]
        {
            let mut exec = self.executor.lock();
            // `pcp == null` is an invariant of "CPU is stopped" — established
            // here, unconditionally, rather than left to every caller that
            // stops the CPU around a `jitv2.pages` mutation to remember.
            // `Jitv2::flush_from_jit_thread` (the compile thread's own
            // Codegen-growth flush — see `Jitv2::codegen`'s doc comment)
            // relies on exactly this: it calls `cpu.stop()`, mutates the
            // page pool while nothing can race it (the CPU is provably not
            // running), then `cpu.start()` — the next dispatch re-derives
            // `pcp` fresh against whatever the pool looks like now, same as
            // any other nanotlb miss. Without this, a flush that ran
            // between this stop() and the next start() would leave `pcp`
            // dangling into a cleared/reallocated `Vec` with nothing to
            // catch it. (The CPU-thread-triggered case,
            // `jitv2_track_pcp`/`flush_from_cpu_thread`, never calls
            // `cpu.stop()` at all — it IS the CPU thread, already
            // synchronously "as good as stopped" for its own purposes — so
            // it calls `nanotlb_invalidate()` directly itself instead.)
            exec.nanotlb_invalidate();
            // Locks the same Mutex<Jitv2> that CompileQueue::worker_loop
            // locks around its own flush_from_jit_thread call — safe only
            // because worker_loop calls cpu.stop() (this function) BEFORE
            // taking that lock, not while already holding it (see
            // worker_loop's own comment on that call site: taking it first
            // would self-deadlock the compile thread here).
            let jit = exec.jitv2.lock();
            eprintln!("=== JIT v2 page pool ===");
            eprintln!("  {} / {} pages used", jit.pages_used(), jit.capacity());
        }
        #[cfg(feature = "developer_ip7")]
        {
            let map = &self.executor.lock().core.compare_delta_stats;
            if !map.is_empty() {
                let total: u32 = map.values().sum();
                let mut top: Vec<(u32, u32)> = map.iter().map(|(&k, &v)| (k, v)).collect();
                top.sort_by(|a, b| b.1.cmp(&a.1));
                eprintln!("=== CP0 Compare delta stats ({} samples) ===", total);
                eprintln!("  Top clusters (hw-counts rounded to 100):");
                for (bucket, cnt) in top.iter().take(10) {
                    let pct = *cnt as f64 * 100.0 / total as f64;
                    eprintln!("    ~{:>8}  {:>6}x  {:5.1}%", bucket, cnt, pct);
                }
            }
        }
    }

    fn start(&self) {
        if self.is_running() { return; }

        // Resume the virtual CP0 Count from its latched value and re-arm
        // the compare timer, before the CPU thread spawns and can execute.
        self.executor.lock().core.on_cpu_start();

        self.running.store(true, Ordering::SeqCst);
        let executor = self.executor.clone();
        let running = self.running.clone();

        // The compile queue's lifecycle no longer follows the CPU's own
        // stop()/start() — it's independently managed by `Machine` (started
        // once at machine startup, stopped at shutdown) and by the `j2
        // inline` monitor command (which explicitly starts/stops it when
        // switching between inline and threaded compile modes). This is
        // what lets the compile thread call `cpu.stop()`/`cpu.start()` on
        // itself (to pause the CPU around its own Codegen-growth flush,
        // `Jitv2::codegen`'s doc comment) without the two thread-stop paths
        // fighting over which stops which — a plain CPU pause never touches
        // the compile queue at all anymore.

        *self.thread.lock() = Some(thread::Builder::new().name("MIPS-CPU".to_string()).spawn(move || {
            crate::thread_affinity::pin_current(crate::thread_affinity::PerfRole::MipsCpu);
            let mut guard = executor.lock();

            // A freshly spawned OS thread has its own host FPU rounding-mode
            // state, independent of whatever thread ran the CPU previously —
            // sync it to the guest's tracked FCSR.RM now rather than relying
            // on the platform default happening to match (usually RN, but not
            // guaranteed, and silently wrong if the guest had set a non-RN mode
            // before the CPU was last stopped).
            crate::platform::set_fpu_mode((guard.core.fpu_fcsr & 0x3) as u8);

            // --- perf sampling (comment out to disable) ---
            //let mut last_cycles: u64 = guard.core.hot.cycles;
            //let mut last_time = std::time::Instant::now();
            // --- end perf sampling ---

            // Idle detection + park state (see docs/idle-pause-work.md). Compiled
            // in only with the `idle-pause` feature (off by default; opt in with
            // --features idle-pause). When compiled in, set IRIS_NO_IDLE to keep
            // spinning the host CPU at runtime (for benchmarking/debug).
            //
            // We only park when the architectural state (PC + all GPRs) REPEATS
            // across batches. A polling/idle loop (e.g. the kernel idle loop
            // waiting on the run queue) cycles through the same states and exits
            // only on an interrupt — safe to park. A busy-delay loop (e.g. IRIX
            // DELAY(): `bgezl v1,-1; subu v1,v1,v0`) changes a counter every
            // iteration, so its state never repeats — we must NOT park it or
            // boot stalls. The state-repeat test distinguishes the two.
            #[cfg(feature = "idle-pause")]
            let mut idle_state = crate::idle_park::IdleParkState::default();

            while running.load(Ordering::Relaxed) {
                #[cfg(feature = "lightning")]
                for _ in 0..1000 {
                    // No breakpoints possible in lightning mode; 10x manual unroll
                    // avoids the per-step match and helps LLVM see a larger block.
                    guard.step(); guard.step(); guard.step(); guard.step(); guard.step();
                    guard.step(); guard.step(); guard.step(); guard.step(); guard.step();
                }
                #[cfg(not(feature = "lightning"))]
                for _ in 0..1000 {
                    let status = guard.step();
                    match status {
                        EXEC_BREAKPOINT => {
                            running.store(false, Ordering::SeqCst);
                            if let Some(bp_id) = guard.last_bp_hit {
                                dlog_dev!(LogModule::Mips, "\nBreakpoint {} hit at PC: {:016x}", bp_id, guard.core.pc);
                            } else {
                                dlog_dev!(LogModule::Mips, "\nBreakpoint hit at PC: {:016x}", guard.core.pc);
                            }
                            break;
                        }
                        _ => {}
                    }
                }
                #[cfg(feature = "idle-pause")]
                if crate::idle_park::idle_park_enabled() && idle_state.update(&guard.core) {
                    drop(guard);
                    {
                        let mut guard = executor.lock();
                        idle_state.park(&mut guard.core, &running);
                    }
                    guard = executor.lock();
                }
                // --- end idle park ---
                // --- perf sampling (comment out to disable) ---
                //let cycles = guard.core.hot.cycles;
                //if cycles.wrapping_sub(last_cycles) >= 100_000_000 {
                //    let now = std::time::Instant::now();
                //    let elapsed = now.duration_since(last_time).as_secs_f64();
                //    let mips = (cycles - last_cycles) as f64 / elapsed / 1_000_000.0;
                //    println!("CPU: {:.1} MIPS  (cycles={})", mips, cycles);
                //    last_cycles = cycles;
                //    last_time = now;
                //}
                // --- end perf sampling ---
            }
        }).unwrap());
    }

    fn is_running(&self) -> bool { 
        self.running.load(Ordering::SeqCst) 
    }
    
    fn get_clock(&self) -> u64 {
        self.cycles_ptr.get()
    }

    fn signal(&self, signal: Signal) {
        // Safety: interrupts_ptr points into the executor's MipsCore, which
        // outlives this MipsCpu for the process lifetime (see the field's
        // doc comment) — bypasses the executor mutex to reduce latency.
        let interrupts = unsafe { &*self.interrupts_ptr };
        match signal {
            Signal::Reset(_soft) => {
                interrupts.fetch_or(SOFT_RESET_BIT, Ordering::SeqCst);
            }
            Signal::Interrupt(line, active) => {
                let mask = 1u64 << (line + 8);
                if active {
                    interrupts.fetch_or(mask, Ordering::SeqCst);
                } else {
                    interrupts.fetch_and(!mask, Ordering::SeqCst);
                }
            }
        }
    }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![
            ("cpu".to_string(), "CPU commands: start, stop, run, step, regs, cop0, cop1, mem, dis, jump, translate, trace, undo, sym, loadsym".to_string()),
            ("bp".to_string(), "Breakpoint commands: bp add <addr> [type] [if <expr>], bp list, bp del <id>, bp enable/disable <id>".to_string()),
            ("b".to_string(), "Alias for bp add".to_string()),
            ("bl".to_string(), "Alias for bp list".to_string()),
            ("bd".to_string(), "Alias for bp disable".to_string()),
            ("be".to_string(), "Alias for bp enable".to_string()),
            ("bb".to_string(), "Alias for bp delete".to_string()),
            ("tlb".to_string(), "TLB commands: tlb dump | tlb trans <vaddr> [asid] | tlb debug <on|off> [DEV]".to_string()),
            ("start".to_string(), "Start CPU execution thread".to_string()),
            ("stop".to_string(), "Stop CPU execution thread".to_string()),
            ("status".to_string(), "Show CPU running status and current PC".to_string()),
            ("ip7".to_string(), "Set pending IP7 (CP0 Compare timer) interrupt — simulate a timer fire while stopped".to_string()),
            ("exception".to_string(), "Control exception breaks: exception <class|code|all> <on|off>".to_string()),
            ("run".to_string(), "Run instructions until exception or breakpoint: run [addr]".to_string()),
            ("step".to_string(), "Step n instructions or until address: step [count|addr]".to_string()),
            ("next".to_string(), "Step over function calls: next [count]".to_string()),
            ("finish".to_string(), "Run until function return (jr ra)".to_string()),
            ("fin".to_string(), "Alias for finish".to_string()),
            ("s".to_string(), "Alias for step".to_string()),
            #[cfg(feature = "developer")]
            ("si".to_string(), "Step suppressing interrupt delivery (alias for step, no interrupts taken) [DEV]".to_string()),
            ("n".to_string(), "Alias for next".to_string()),
            ("regs".to_string(), "Dump registers".to_string()),
            ("r".to_string(), "Alias for regs".to_string()),
            ("c".to_string(), "Alias for run".to_string()),
            ("cont".to_string(), "Alias for run".to_string()),
            ("cop0".to_string(), "Dump COP0 registers".to_string()),
            ("cop1".to_string(), "Dump COP1 registers".to_string()),
            ("mem".to_string(), "Dump virtual memory: mem <addr> [count]".to_string()),
            ("m".to_string(), "Alias for mem".to_string()),
            ("mw".to_string(), "Write virtual memory: mw <addr> <val> [size: b|h|w|d]".to_string()),
            ("stack".to_string(), "Dump stack memory: stack [addr] [count]".to_string()),
            ("bt".to_string(), "Print backtrace: bt [frames]".to_string()),
            ("ms".to_string(), "Read string from virtual memory: ms <addr> [max_len]".to_string()),
            ("dis".to_string(), "Disassemble virtual memory: dis [addr] [count]".to_string()),
            ("d".to_string(), "Alias for dis".to_string()),
            ("jump".to_string(), "Set PC to address: jump <addr>".to_string()),
            ("setreg".to_string(), "Set register value: setreg <reg> <value>".to_string()),
            ("translate".to_string(), "Translate virtual address: translate <addr>".to_string()),
            ("t".to_string(), "Alias for translate".to_string()),
            ("debug".to_string(), "CPU instruction trace: debug <on|off|file <path>> [DEV]".to_string()),
            ("ex".to_string(), "Alias for exception".to_string()),
            ("undo".to_string(), "Undo N instructions or control undo buffer: undo [count] | undo <on|off|clear|resize <n>> [DEV]".to_string()),
            ("dt".to_string(), "Disassemble traceback: dt [count] | dt file <path> [count]".to_string()),
            ("idleprof".to_string(), "Locate idle/spin loops via PC sampling: idleprof <on|off|report [count]>".to_string()),
            #[cfg(feature = "instr_stats")]
            ("instrstats".to_string(), "Per-instruction decode/execution counters: instrstats [report|clear|dump] [DEV]".to_string()),
            ("u".to_string(), "Alias for undo [DEV]".to_string()),
            ("sym".to_string(), "Lookup symbol: sym <addr>".to_string()),
            ("loadsym".to_string(), "Load symbols from file: loadsym <file>".to_string()),
            ("loadelf".to_string(), "Load a static ELF32 MSB binary and set PC to its entry: loadelf <file>".to_string()),
            ("loadbin".to_string(), "Load raw bytes at a virtual address: loadbin <file> <addr>".to_string()),
            ("proc".to_string(), "IRIX kernel introspection: proc info  (requires `loadsym` first)".to_string()),
            ("l1i".to_string(), "L1 Instruction Cache commands: l1i <check|dump> <addr|index>".to_string()),
            ("l1d".to_string(), "L1 Data Cache commands: l1d <check|dump> <addr|index>".to_string()),
            ("l2".to_string(), "L2 Cache commands: l2 <check|dump> <addr|index>".to_string()),
            ("ll".to_string(), "Show LL/SC state: llbit and lladdr".to_string()),
            #[cfg(feature = "jitv2")]
            ("j2".to_string(), "JIT v2 introspection: j2 pcp | j2 status (alias: stats) | j2 inline [on|off] | j2 dispatch [on|off] | j2 fallback [on|off] | j2 threads (read-only) | j2 <alu|fpu|branch|loadstore|cop0> [on|off] | j2 instrs [category] | j2 flush | j2 lockstep (status only; always on when built) (see also: jitcheck <n> for JIT-vs-interpreter determinism checking)".to_string()),
            #[cfg(feature = "developer")]
            ("trace".to_string(), "Execution trace capture: trace start <path> | trace stop | trace status".to_string()),
        ]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut writer: Box<dyn Write + Send>) -> Result<(), String> {
        // Handle "cpu" prefix by shifting args
        let (actual_cmd, actual_args) = if cmd == "cpu" {
            if args.is_empty() {
                return Err("Usage: cpu <command> [args...]".to_string());
            }
            (args[0], &args[1..])
        } else {
            (cmd, args)
        };

        if actual_cmd == "bp" || actual_cmd == "b" || actual_cmd == "bd" || actual_cmd == "be" || actual_cmd == "bb" || actual_cmd == "bl" {
            let (subcmd, args) = if actual_cmd == "bp" {
                if actual_args.is_empty() {
                    return Err("Usage: bp <add|list|del|enable|disable> ...".to_string());
                }
                (actual_args[0], &actual_args[1..])
            } else {
                let s = match actual_cmd {
                    "b" => "add",
                    "bd" => "disable",
                    "be" => "enable",
                    "bb" => "delete",
                    "bl" => "list",
                    _ => unreachable!(),
                };
                (s, actual_args)
            };

            let mut exec = self.executor.lock();
            match subcmd {
                "add" => {
                    if args.is_empty() { return Err("Usage: bp add <addr> [type]".to_string()); }

                    // Check for "if <expr>" at the end
                    let (args_before_if, cond_expr) = {
                        if let Some(pos) = args.iter().position(|&a| a == "if") {
                            let expr_str = args[pos+1..].join(" ");
                            let symbols = exec.symbols.lock();
                            let expr = exp::parse_and_fold(&expr_str, Some(&symbols))?;
                            (&args[..pos], Some(expr))
                        } else {
                            (args, None)
                        }
                    };

                    if args_before_if.is_empty() { return Err("Usage: bp add <addr> [type] [if <expr>]".to_string()); }

                    let addr = {
                        let symbols = exec.symbols.lock();
                        parse_cpu_arg(args_before_if[0], &exec.core, Some(&symbols))?
                    };
                    
                    let kind = if args_before_if.len() > 1 {
                        match args_before_if[1] {
                            "pc" => BpType::Pc,
                            "r" | "read" => BpType::VirtRead,
                            "w" | "write" => BpType::VirtWrite,
                            "f" | "fetch" => BpType::VirtFetch,
                            "pr" | "pread" => BpType::PhysRead,
                            "pw" | "pwrite" => BpType::PhysWrite,
                            "pf" | "pfetch" => BpType::PhysFetch,
                            _ => return Err("Invalid breakpoint type. Options: pc, r, w, f, pr, pw, pf".to_string()),
                        }
                    } else {
                        BpType::Pc
                    };

                    let id = exec.next_bp_id;
                    exec.next_bp_id += 1;
                    exec.add_breakpoint(id, addr, kind);

                    if let Some(bp) = exec.breakpoints.iter_mut().find(|bp| bp.id == id) {
                        bp.condition = cond_expr;
                    }

                    if let Some(bp) = exec.breakpoints.iter().find(|bp| bp.id == id).filter(|bp| bp.condition.is_some()) {
                        writeln!(writer, "Breakpoint {} added at {:016x} ({:?}) with condition", id, addr, kind).unwrap();
                    } else {
                        writeln!(writer, "Breakpoint {} added at {:016x} ({:?})", id, addr, kind).unwrap();
                    }
                    return Ok(());
                }
                "list" => {
                    writeln!(writer, "Breakpoints:").unwrap();
                    for bp in &exec.breakpoints {
                        if bp.id == 0 { continue; } // Skip internal breakpoint
                        writeln!(writer, "  {}: {:016x} {:?} {}", bp.id, bp.addr, bp.kind, if bp.enabled { "(enabled)" } else { "(disabled)" }).unwrap();
                    }
                    return Ok(());
                }
                "del" | "delete" => {
                    if args.is_empty() { return Err("Usage: bp del <id>".to_string()); }
                    let id = args[0].parse::<usize>().map_err(|_| "Invalid ID")?;
                    if exec.remove_breakpoint(id) {
                        writeln!(writer, "Breakpoint {} deleted", id).unwrap();
                        return Ok(());
                    } else {
                        return Err(format!("Breakpoint {} not found", id));
                    }
                }
                "enable" => {
                    if args.is_empty() { return Err("Usage: bp enable <id>".to_string()); }
                    let id = args[0].parse::<usize>().map_err(|_| "Invalid ID")?;
                    if exec.set_breakpoint_enabled(id, true) {
                        writeln!(writer, "Breakpoint {} enabled", id).unwrap();
                        return Ok(());
                    } else {
                        return Err(format!("Breakpoint {} not found", id));
                    }
                }
                "disable" => {
                    if args.is_empty() { return Err("Usage: bp disable <id>".to_string()); }
                    let id = args[0].parse::<usize>().map_err(|_| "Invalid ID")?;
                    if exec.set_breakpoint_enabled(id, false) {
                        writeln!(writer, "Breakpoint {} disabled", id).unwrap();
                        return Ok(());
                    } else {
                        return Err(format!("Breakpoint {} not found", id));
                    }
                }
                _ => return Err("Unknown bp subcommand".to_string()),
            }
        }

        if actual_cmd == "loadsym" {
            if actual_args.is_empty() {
                return Err("Usage: loadsym <file>".to_string());
            }
            let exec = self.try_lock_executor()?;
            let mut symbols = exec.symbols.lock();
            match symbols.load(actual_args[0]) {
                Ok(count) => {
                    writeln!(writer, "Loaded {} symbols from {}", count, actual_args[0]).unwrap();
                    return Ok(());
                },
                Err(e) => return Err(format!("Failed to load symbols: {}", e)),
            }
        }

        if actual_cmd == "sym" {
            if actual_args.is_empty() {
                return Err("Usage: sym <addr>".to_string());
            }
            let exec = self.try_lock_executor()?;
            let symbols = exec.symbols.lock();
            let addr = parse_cpu_arg(actual_args[0], &exec.core, Some(&symbols))?;
            
            if let Some((sym_addr, name)) = symbols.lookup(addr) {
                let offset = addr - sym_addr;
                if offset == 0 {
                    writeln!(writer, "{:016x} = {}", addr, name).unwrap();
                } else {
                    writeln!(writer, "{:016x} = {} + 0x{:x}", addr, name, offset).unwrap();
                }
            } else {
                writeln!(writer, "{:016x} = ???", addr).unwrap();
            }
            return Ok(());
        }

        if actual_cmd == "proc" {
            let mut exec = self.try_lock_executor()?;
            // Helper closures around debug_read so the body stays readable.
            // All reads here are big-endian (IRIX kernel).
            let read_u32 = |exec: &mut MipsExecutor<T, C>, vaddr: u64| -> Result<u32, String> {
                exec.debug_read(vaddr, 4)
                    .map(|v| v as u32)
                    .map_err(|e| format!("debug_read({:#x}): {:?}", vaddr, e))
            };
            let read_bytes = |exec: &mut MipsExecutor<T, C>, vaddr: u64, n: usize| -> Result<Vec<u8>, String> {
                let mut out = Vec::with_capacity(n);
                // debug_read returns up to 8 bytes per call; chunk it.
                let mut a = vaddr;
                let mut remaining = n;
                while remaining > 0 {
                    let step = remaining.min(4);
                    let v = exec.debug_read(a, step)
                        .map_err(|e| format!("debug_read({:#x}): {:?}", a, e))?;
                    // big-endian: high byte first within `step` bytes
                    for i in (0..step).rev() {
                        out.push(((v >> (i * 8)) & 0xff) as u8);
                    }
                    a += step as u64;
                    remaining -= step;
                }
                Ok(out)
            };
            // Probe `utsname` symbol and read sysname/release.
            let utsname_va = {
                let symbols = exec.symbols.lock();
                match symbols.get_addr("utsname") {
                    Some(a) => a,
                    None => return Err("symbol 'utsname' not found — run `loadsym <unix.nm>` first".to_string()),
                }
            };
            // IRIX utsname has 5 fields of SYS_NMLN bytes each. SYS_NMLN is
            // 257 on IRIX 5.3 (large for POSIX), but we only read the front
            // null-terminated strings. Read 256 bytes per field to be safe.
            const NMLN: usize = 257;
            let buf = read_bytes(&mut exec, utsname_va, NMLN * 5)?;
            let read_str = |off: usize| -> String {
                let end = buf[off..off + NMLN].iter().position(|&b| b == 0).unwrap_or(NMLN);
                String::from_utf8_lossy(&buf[off..off + end]).into_owned()
            };
            let sysname  = read_str(0 * NMLN);
            let nodename = read_str(1 * NMLN);
            let release  = read_str(2 * NMLN);
            let version  = read_str(3 * NMLN);
            let machine  = read_str(4 * NMLN);

            let sub = actual_args.first().copied().unwrap_or("info");
            if sub == "info" {
                writeln!(writer, "utsname @ {:#x}:", utsname_va).unwrap();
                writeln!(writer, "  sysname  = {:?}", sysname).unwrap();
                writeln!(writer, "  nodename = {:?}", nodename).unwrap();
                writeln!(writer, "  release  = {:?}", release).unwrap();
                writeln!(writer, "  version  = {:?}", version).unwrap();
                writeln!(writer, "  machine  = {:?}", machine).unwrap();
                let nproc_va = exec.symbols.lock().get_addr("nproc").ok_or_else(|| "symbol 'nproc' not found".to_string())?;
                let proc_va  = exec.symbols.lock().get_addr("proc").ok_or_else(|| "symbol 'proc' not found".to_string())?;
                let nproc = read_u32(&mut exec, nproc_va)?;
                let proc_ptr = read_u32(&mut exec, proc_va)?;
                writeln!(writer, "nproc    @ {:#x} = {}", nproc_va, nproc).unwrap();
                writeln!(writer, "proc[]   @ {:#x} -> {:#x}", proc_va, proc_ptr).unwrap();
                return Ok(());
            }
            return Err("Usage: proc info".to_string());
        }

        if actual_cmd == "ll" {
            let exec = self.try_lock_executor()?;
            let llbit  = exec.cache.get_llbit();
            let lladdr = exec.cache.get_lladdr();
            let phys   = (lladdr as u64) << 4;
            writeln!(writer, "llbit:  {}", if llbit { "SET" } else { "clear" }).unwrap();
            writeln!(writer, "lladdr: {:08x}  (phys {:010x})", lladdr, phys).unwrap();
            return Ok(());
        }

        if actual_cmd == "l1i" || actual_cmd == "l1d" || actual_cmd == "l2" {
            if actual_args.is_empty() {
                return Err(format!("Usage: {} <check|dump> <addr|index>", actual_cmd));
            }
            let cache_name = actual_cmd;
            let op = actual_args[0];
            let val_str = if actual_args.len() > 1 { actual_args[1] } else { "0" };
            
            let mut exec = self.try_lock_executor()?;
            let symbols = exec.symbols.lock();
            let val = parse_cpu_arg(val_str, &exec.core, Some(&symbols))?;
            drop(symbols);

            // Perform translation first if needed, as it requires mutable access to exec
            let (virt_addr, phys_addr) = if (op == "check" || op == "probe") && val >= 0x8000_0000 {
                let tr = exec.debug_translate(val);
                if !tr.is_exception() {
                    let pa = tr.phys as u64;
                    writeln!(writer, "Virtual {:016x} -> Physical {:016x}", val, pa).unwrap();
                    (val, pa)
                } else {
                    writeln!(writer, "Virtual {:016x} -> Translation Failed", val).unwrap();
                    (val, val)
                }
            } else {
                (val, val)
            };

            // Call debug methods through the MipsCache trait
            match op {
                "check" | "probe" => {
                    writeln!(writer, "{}", exec.cache.debug_probe(cache_name, virt_addr, phys_addr)).unwrap();
                }
                "dump" => {
                    writeln!(writer, "{}", exec.cache.debug_dump_line(cache_name, val as usize)).unwrap();
                }
                _ => return Err("Unknown operation. Use check or dump".to_string()),
            }
            return Ok(());
        }

        // Helper to dump registers
        let dump_regs = |exec: &mut MipsExecutor<T, C>, out: &mut dyn Write| {
            writeln!(out, "PC: {:#018x}", exec.core.pc).unwrap();
            writeln!(out, "HI: {:#018x} LO: {:#018x}", exec.core.hi, exec.core.lo).unwrap();
            for i in 0..32 {
                let val = exec.core.gpr[i];
                write!(out, "{:4}(${:02}): {:#018x}  ", mips_dis::reg_name(i as u32), i, val).unwrap();
                if (i + 1) % 4 == 0 { writeln!(out).unwrap(); }
            }
            writeln!(out, "CP0 Status: {:#010x} ({})", exec.core.cp0_status, decode_status(exec.core.cp0_status)).unwrap();
            writeln!(out, "CP0 Cause:  {:#010x} ({})", exec.core.cp0_cause, decode_cause(exec.core.cp0_cause)).unwrap();
            writeln!(out, "CP0 EPC: {:#018x} BadVAddr: {:#018x}", exec.core.cp0_epc, exec.core.cp0_badvaddr).unwrap();
        };

        match actual_cmd {
            "help" => {
                writeln!(writer, "CPU Commands:").unwrap();
                for (c, h) in self.register_commands() {
                    writeln!(writer, "  {:12} - {}", c, h).unwrap();
                }
                Ok(())
            }
            "start" => {
                self.start();
                writeln!(writer, "CPU started").unwrap();
                Ok(())
            }
            "stop" => {
                self.stop();
                writeln!(writer, "CPU stopped").unwrap();
                Ok(())
            }
            "status" => {
                let running = self.is_running();
                let exec = self.try_lock_executor()?;
                let pc = exec.core.pc;
                let symbols = exec.symbols.lock();
                let sym_str = format_pc_symbol(pc, &symbols);
                writeln!(writer, "{} pc={:016x}{}", if running { "running" } else { "stopped" }, pc, sym_str).unwrap();
                let hz = exec.core.count_hz;
                let slow = exec.core.compare_delta_slow;
                let fast = exec.core.compare_delta_fast;
                writeln!(writer, "  count_hz={} ({:.3} MHz)  count={:#010x} compare={:#010x}",
                    hz, hz as f64 / 1e6, exec.core.count_peek(), exec.core.cp0_compare as u32).unwrap();
                writeln!(writer, "  compare_delta_slow={} hw-counts  compare_delta_fast={} hw-counts",
                    slow, fast).unwrap();
                Ok(())
            }
            "ip7" => {
                // The compare timer is silenced while the CPU is stopped
                // (on_cpu_stop) — this injects the interrupt it would have
                // raised, through the same pending word, for debugging
                // interrupt delivery under manual stepping.
                let exec = self.try_lock_executor()?;
                exec.core.set_interrupt(7);
                writeln!(writer, "IP7 set pending (delivered on next executed instruction, mask permitting)").unwrap();
                Ok(())
            }
            "run" | "c" | "cont" => {
                let block = actual_args.first() == Some(&"block");
                let args_rest = if block { &actual_args[1..] } else { actual_args };
                let until_pc = if !args_rest.is_empty() {
                    let exec = self.executor.lock();
                    let symbols = exec.symbols.lock();
                    let pc = parse_cpu_arg(args_rest[0], &exec.core, Some(&symbols))?;
                    println!("Running until PC = {:016x}", pc);
                    Some(pc)
                } else {
                    None
                };

                if let Some(pc) = until_pc {
                    self.executor.lock().set_temp_breakpoint(pc);
                }
                self.run_debug_loop(None, block, writer);
                Ok(())
            }
            "finish" | "fin" => {
                let actual_args = if actual_args.first() == Some(&"block") { &actual_args[1..] } else { actual_args };
                let mut exec = self.executor.lock();
                let ret_addr = exec.get_return_address();
                if let Some(addr) = ret_addr {
                    exec.set_temp_breakpoint(addr);
                    drop(exec);
                    self.run_debug_loop(None, true, writer);
                    Ok(())
                } else {
                    Err("Could not determine return address".to_string())
                }
            }
            "step" | "s" | "si" => {
                let actual_args = if actual_args.first() == Some(&"block") { &actual_args[1..] } else { actual_args };
                let mut count = Some(1);
                let mut until_pc = None;

                if !actual_args.is_empty() {
                    let arg = actual_args[0];
                    // If it parses as a number and doesn't start with 0x, treat as count.
                    // Otherwise treat as address/register.
                    if let Ok(c) = arg.parse::<usize>() {
                        if !arg.starts_with("0x") {
                            count = Some(c);
                        } else {
                            let exec = self.executor.lock();
                            let symbols = exec.symbols.lock();
                            until_pc = Some(parse_cpu_arg(arg, &exec.core, Some(&symbols))?);
                            count = None;
                        }
                    } else {
                        let exec = self.executor.lock();
                        let symbols = exec.symbols.lock();
                        until_pc = Some(parse_cpu_arg(arg, &exec.core, Some(&symbols))?);
                        count = None;
                    }
                }

                if let Some(pc) = until_pc {
                    self.executor.lock().set_temp_breakpoint(pc);
                }
                #[cfg(feature = "developer")]
                if cmd == "si" { self.executor.lock().skip_interrupts = true; }
                self.run_debug_loop(count, true, writer);
                Ok(())
            }
            "next" | "n" => {
                let actual_args = if actual_args.first() == Some(&"block") { &actual_args[1..] } else { actual_args };
                let count = if !actual_args.is_empty() {
                    actual_args[0].parse().unwrap_or(1)
                } else {
                    1
                };

                if count > 1 {
                    writeln!(writer, "Warning: 'next' with count > 1 is not fully supported, executing once.").unwrap();
                }

                let mut exec = self.executor.lock();
                let pc = exec.core.pc;
                let is_call = if let Ok(instr) = exec.debug_fetch_instr(pc) {
                    is_call_instruction(instr)
                } else {
                    false
                };

                if is_call {
                    if is_call {
                         if let Ok(instr) = exec.debug_fetch_instr(pc) {
                             let symbols = exec.symbols.lock();
                             let sym_str = format_pc_symbol(pc, &symbols);
                             let dis = mips_dis::disassemble(instr, pc, Some(&symbols));
                             writeln!(writer, "Exec: {:016x}{}: {:08x} {}", pc, sym_str, instr, dis).unwrap();
                         }
                    }
                }

                drop(exec); // Release lock before running loop

                if is_call {
                    self.executor.lock().set_temp_breakpoint(pc + 8);
                    self.run_debug_loop(None, true, writer);
                } else {
                    self.run_debug_loop(Some(1), true, writer);
                }
                Ok(())
            }
            "regs" | "r" => {
                let mut exec = self.try_lock_executor()?;
                if actual_args.is_empty() {
                    dump_regs(&mut exec, &mut writer);
                } else {
                    let symbols = exec.symbols.lock();
                    for arg in actual_args {
                        match parse_cpu_arg(arg, &exec.core, Some(&symbols)) {
                            Ok(val) => {
                                let sym_str = format_pc_symbol(val, &symbols);
                                writeln!(writer, "{}: {:016x} ({}){}", arg, val, val, sym_str).unwrap();
                            }
                            Err(e) => writeln!(writer, "{}", e).unwrap(),
                        }
                    }
                }
                Ok(())
            }
            "cop0" => {
                let mut exec = self.try_lock_executor()?;
                writeln!(writer, "COP0 Registers:").unwrap();
                for i in 0..32 {
                    let val = exec.core.read_cp0(i);
                    let name = mips_dis::cp0_reg_name(i);
                    if name != "?" {
                        write!(writer, "  {:2} {:8}: {:#018x}", i, name, val).unwrap();
                        if i == 12 { // Status
                            write!(writer, " {}", decode_status(val as u32)).unwrap();
                        } else if i == 13 { // Cause
                            write!(writer, " {}", decode_cause(val as u32)).unwrap();
                        }
                        writeln!(writer).unwrap();
                    }
                }
                let c = &exec.core;
                // count_hz is derived purely from which pattern bucket the
                // Compare deltas fall into (assumed 100Hz slow / 1kHz fast —
                // see infer_count_hz's doc comment for why real elapsed time
                // can't be used: it would be circular with our own hptimer's
                // fire time).
                writeln!(writer, "  IP7 timer: count_hz={} ({:.3} MHz)  fired={}",
                    c.count_hz, c.count_hz as f64 / 1e6,
                    c.fasttick_count.load(Ordering::Relaxed)).unwrap();
                writeln!(writer, "    slow_delta={} hw-counts  fast_delta={} hw-counts",
                    c.compare_delta_slow, c.compare_delta_fast).unwrap();
                Ok(())
            }
            "cop1" => {
                let exec = self.try_lock_executor()?;
                writeln!(writer, "COP1 Registers (FPU):").unwrap();
                for i in 0..32 {
                    let val = exec.core.fpr[i];
                    let f32_val = f32::from_bits(val as u32);
                    let f64_val = f64::from_bits(val);
                    writeln!(writer, "  f{:02}: {:#018x}  (f32: {:e}, f64: {:e})", i, val, f32_val, f64_val).unwrap();
                }
                writeln!(writer, "Control Registers:").unwrap();
                writeln!(writer, "  FIR:  {:#010x}", exec.core.fpu_fir).unwrap();
                writeln!(writer, "  FCCR: {:#010x}", exec.core.fpu_fccr).unwrap();
                writeln!(writer, "  FEXR: {:#010x}", exec.core.fpu_fexr).unwrap();
                writeln!(writer, "  FENR: {:#010x}", exec.core.fpu_fenr).unwrap();
                writeln!(writer, "  FCSR: {:#010x}", exec.core.fpu_fcsr).unwrap();
                Ok(())
            }
            "mem" | "m" | "memory" => {
                if actual_args.is_empty() { return Err("Usage: mem <addr> [count]".to_string()); }
                let mut exec = self.try_lock_executor()?;
                let symbols_arc = exec.symbols.clone();
                let symbols = symbols_arc.lock();
                let addr = parse_cpu_arg(actual_args[0], &exec.core, Some(&symbols))?;
                let count = if actual_args.len() > 1 { actual_args[1].parse().unwrap_or(1) } else { 1 };
                
                for i in 0..count {
                    let curr_addr = addr.wrapping_add(i * 4);
                    match exec.debug_read(curr_addr, 4) {
                        Ok(val) => writeln!(writer, "{:#018x}: {:#010x}", curr_addr, val).unwrap(),
                        Err(e) => writeln!(writer, "{:#018x}: Error {:?}", curr_addr, e).unwrap(),
                    }
                }
                Ok(())
            }
            "stack" => {
                let mut exec = self.try_lock_executor()?;
                let sp = exec.core.read_gpr(29);
                let symbols_arc = exec.symbols.clone();
                let symbols = symbols_arc.lock();
                
                let addr = if !actual_args.is_empty() {
                    parse_cpu_arg(actual_args[0], &exec.core, Some(&symbols))?
                } else {
                    sp
                };
                
                let count = if actual_args.len() > 1 { actual_args[1].parse().unwrap_or(16) } else { 16 };
                
                for i in 0..count {
                    let curr_addr = addr.wrapping_add(i * 8); // 64-bit stack slots usually
                    match exec.debug_read(curr_addr, 8) {
                        Ok(val) => writeln!(writer, "{:016x}: {:016x}", curr_addr, val).unwrap(),
                        Err(_) => writeln!(writer, "{:016x}: ????????????????", curr_addr).unwrap(),
                    }
                }
                Ok(())
            }
            "bt" | "backtrace" => {
                writeln!(writer, "{}", self.try_lock_executor()?.backtrace(20)).unwrap();
                Ok(())
            }
            "mw" => {
                if actual_args.len() < 2 { return Err("Usage: mw <addr> <val> [size: b|h|w|d]".to_string()); }
                let mut exec = self.try_lock_executor()?;
                let symbols_arc = exec.symbols.clone();
                let symbols = symbols_arc.lock();
                
                let addr = parse_cpu_arg(actual_args[0], &exec.core, Some(&symbols))?;
                let val = u64::from_str_radix(actual_args[1].trim_start_matches("0x"), 16)
                    .or_else(|_| actual_args[1].parse::<u64>())
                    .map_err(|_| "Invalid value".to_string())?;
                
                let size: usize = if actual_args.len() > 2 {
                    match actual_args[2] {
                        "b" | "byte" => 1,
                        "h" | "half" => 2,
                        "w" | "word" => 4,
                        "d" | "double" => 8,
                        _ => return Err("Invalid size. Use b, h, w, or d".to_string()),
                    }
                } else {
                    4
                };

                let mask = full_mask_for_usize(size);

                match exec.debug_write(addr, val, size, mask) {
                    EXEC_COMPLETE => writeln!(writer, "Wrote {:x} to {:016x}", val, addr).unwrap(),
                    e => writeln!(writer, "Error writing to {:016x}: {:?}", addr, e).unwrap(),
                }
                Ok(())
            }
            "ms" => {
                if actual_args.is_empty() { return Err("Usage: ms <addr> [max_len]".to_string()); }
                let mut exec = self.try_lock_executor()?;
                let symbols_arc = exec.symbols.clone();
                let symbols = symbols_arc.lock();
                
                let addr = parse_cpu_arg(actual_args[0], &exec.core, Some(&symbols))?;
                let max_len = if actual_args.len() > 1 {
                    actual_args[1].parse::<usize>().unwrap_or(256)
                } else {
                    256
                };

                let mut bytes = Vec::new();
                let mut curr = addr;
                for _ in 0..max_len {
                    match exec.debug_read(curr, 1) {
                        Ok(val) => {
                            let b = val as u8;
                            if b == 0 { break; }
                            bytes.push(b);
                            curr = curr.wrapping_add(1);
                        }
                        Err(_) => break,
                    }
                }
                
                let s = String::from_utf8_lossy(&bytes);
                writeln!(writer, "{:016x}: \"{}\"", addr, s).unwrap();
                Ok(())
            }
            "dis" | "d" | "disasm" | "disassemble" => {
                let mut exec = self.try_lock_executor()?;
                let symbols_arc = exec.symbols.clone();
                let symbols = symbols_arc.lock();
                let addr = if !actual_args.is_empty() {
                    parse_cpu_arg(actual_args[0], &exec.core, Some(&symbols))?
                } else {
                    exec.core.pc
                };
                let count = if actual_args.len() > 1 { actual_args[1].parse().unwrap_or(1) } else { 1 };
                
                for i in 0..count {
                    let curr_addr = addr.wrapping_add(i * 4);
                    match exec.debug_fetch_instr(curr_addr) {
                        Ok(instr) => {
                            let sym_str = format_pc_symbol(curr_addr, &symbols);
                            writeln!(writer, "{:#018x}{}: {:#010x} {}", curr_addr, sym_str, instr, mips_dis::disassemble(instr, curr_addr, Some(&symbols))).unwrap()
                        },
                        Err(_) => writeln!(writer, "{:#018x}: Could not fetch", curr_addr).unwrap(),
                    }
                }
                Ok(())
            }
            "loadelf" => {
                if actual_args.is_empty() { return Err("Usage: loadelf <file>".to_string()); }
                write!(writer, "{}", self.load_elf(actual_args[0])?).unwrap();
                Ok(())
            }
            "loadbin" => {
                if actual_args.len() < 2 { return Err("Usage: loadbin <file> <addr>".to_string()); }
                let addr = {
                    let exec = self.try_lock_executor()?;
                    let symbols_arc = exec.symbols.clone();
                    let symbols = symbols_arc.lock();
                    parse_cpu_arg(actual_args[1], &exec.core, Some(&symbols))?
                };
                write!(writer, "{}", self.load_bin(actual_args[0], addr)?).unwrap();
                Ok(())
            }
            "jump" => {
                if actual_args.is_empty() { return Err("Usage: jump <addr>".to_string()); }
                let mut exec = self.try_lock_executor()?;
                let symbols_arc = exec.symbols.clone();
                let symbols = symbols_arc.lock();
                let addr = parse_cpu_arg(actual_args[0], &exec.core, Some(&symbols))?;
                exec.core.pc = addr;
                writeln!(writer, "PC set to {:016x}", addr).unwrap();
                Ok(())
            }
            "setreg" => {
                if actual_args.len() < 2 { return Err("Usage: setreg <reg> <value>".to_string()); }
                let mut exec = self.try_lock_executor()?;
                let symbols_arc = exec.symbols.clone();
                let symbols = symbols_arc.lock();
                let target = exp::parse_reg_target(actual_args[0])
                    .ok_or_else(|| format!("Unknown register: {}", actual_args[0]))?;
                let val = parse_cpu_arg(actual_args[1], &exec.core, Some(&symbols))?;
                exp::write_reg_target(&target, &mut exec.core, val);
                writeln!(writer, "{} = {:016x}", actual_args[0], val).unwrap();
                Ok(())
            }
            "translate" | "t" | "trans" => {
                if actual_args.is_empty() { return Err("Usage: translate <addr>".to_string()); }
                let mut exec = self.try_lock_executor()?;
                let symbols_arc = exec.symbols.clone();
                let symbols = symbols_arc.lock();
                let addr = parse_cpu_arg(actual_args[0], &exec.core, Some(&symbols))?;
                let tr = exec.debug_translate(addr);
                if tr.is_exception() {
                    writeln!(writer, "Exception(0x{:08x})", tr.status).unwrap();
                } else {
                    let c = match tr.status & 0x7 {
                        3 => "Cacheable", 5 => "CacheableCoherent", _ => "Uncached",
                    };
                    writeln!(writer, "Translated {{ phys_addr: 0x{:08x}, cache_attr: {} }}", tr.phys, c).unwrap();
                }
                Ok(())
            }
            "debug" => {
                if actual_args.is_empty() {
                    return Err("Usage: debug <on|off|file> [filename]".to_string());
                }
                match actual_args[0] {
                    "on" | "1" => {
                        self.debug.store(true, Ordering::Relaxed);
                        *self.trace_file.lock() = None;
                        writeln!(writer, "CPU debug enabled (console output)").unwrap();
                    }
                    "off" | "0" => {
                        self.debug.store(false, Ordering::Relaxed);
                        *self.trace_file.lock() = None;
                        writeln!(writer, "CPU debug disabled").unwrap();
                    }
                    "file" => {
                        // file <path>: trace to file only, no socket spam
                        let path = actual_args.get(1).ok_or("Usage: debug file <filename>")?;
                        match std::fs::File::create(path) {
                            Ok(f) => {
                                self.debug.store(false, Ordering::Relaxed);
                                *self.trace_file.lock() = Some(std::io::BufWriter::new(f));
                                writeln!(writer, "CPU trace -> {}", path).unwrap();
                            }
                            Err(e) => return Err(format!("Cannot open {}: {}", path, e)),
                        }
                    }
                    _ => return Err("Usage: debug <on|off|file> [filename]".to_string()),
                }
                Ok(())
            }
            "ex" | "exception" | "exc" => {
                if actual_args.len() < 2 {
                    return Err("Usage: exception <class|code|all> <on|off>".to_string());
                }
                let target = actual_args[0];
                let enable = match actual_args[1] {
                    "on" | "1" => true,
                    "off" | "0" => false,
                    _ => return Err("Usage: exception <class|code|all> <on|off>".to_string()),
                };

                let mut mask = self.exception_mask.load(Ordering::Relaxed);
                let set_bit = |m: &mut u32, bit: u32, val: bool| {
                    if val { *m |= 1 << bit; } else { *m &= !(1 << bit); }
                };

                match target {
                    "all" => mask = if enable { 0xFFFFFFFF } else { 0 },
                    "int" => set_bit(&mut mask, EXC_INT, enable),
                    "tlb" => {
                        set_bit(&mut mask, EXC_MOD, enable);
                        set_bit(&mut mask, EXC_TLBL, enable);
                        set_bit(&mut mask, EXC_TLBS, enable);
                    },
                    "addr" => {
                        set_bit(&mut mask, EXC_ADEL, enable);
                        set_bit(&mut mask, EXC_ADES, enable);
                    },
                    "bus" => {
                        set_bit(&mut mask, EXC_IBE, enable);
                        set_bit(&mut mask, EXC_DBE, enable);
                    },
                    "sys" => {
                        set_bit(&mut mask, EXC_SYS, enable);
                        set_bit(&mut mask, EXC_BP, enable);
                    },
                    "ri" => {
                        set_bit(&mut mask, EXC_RI, enable);
                        set_bit(&mut mask, EXC_CPU, enable);
                    },
                    "arith" => {
                        set_bit(&mut mask, EXC_OV, enable);
                        set_bit(&mut mask, EXC_TR, enable);
                        set_bit(&mut mask, EXC_FPE, enable);
                    },
                    "watch" => set_bit(&mut mask, EXC_WATCH, enable),
                    "vce" => {
                        set_bit(&mut mask, EXC_VCEI, enable);
                        set_bit(&mut mask, EXC_VCED, enable);
                    },
                    s => {
                        if let Ok(code) = s.parse::<u32>() {
                            if code < 32 {
                                set_bit(&mut mask, code, enable);
                            } else {
                                return Err("Invalid exception code".to_string());
                            }
                        } else {
                            return Err("Unknown exception class or code".to_string());
                        }
                    }
                }
                self.exception_mask.store(mask, Ordering::Relaxed);
                writeln!(writer, "Exception mask set to {:08x}", mask).unwrap();
                Ok(())
            }
            "undo" | "u" => {
                #[cfg(feature = "developer")]
                {
                // Handle both "cpu undo on|off|clear" and step-back "undo [count]"
                if !actual_args.is_empty() {
                    match actual_args[0] {
                        "on" | "1" if actual_args[0] == "on" || actual_args[0] == "1" => {
                            let mut exec = self.try_lock_executor()?;
                            exec.undo_buffer.enable();
                            writeln!(writer, "CPU undo buffer enabled").unwrap();
                            return Ok(());
                        }
                        "off" | "0" if actual_args[0] == "off" || actual_args[0] == "0" => {
                            let mut exec = self.try_lock_executor()?;
                            exec.undo_buffer.disable();
                            writeln!(writer, "CPU undo buffer disabled").unwrap();
                            return Ok(());
                        }
                        "clear" => {
                            let mut exec = self.try_lock_executor()?;
                            exec.undo_buffer.clear();
                            writeln!(writer, "CPU undo buffer cleared").unwrap();
                            return Ok(());
                        }
                        "resize" => {
                            let new_capacity: usize = actual_args.get(1)
                                .and_then(|s| s.parse().ok())
                                .ok_or("Usage: undo resize <capacity>")?;
                            let mut exec = self.try_lock_executor()?;
                            let old_capacity = exec.undo_buffer.capacity();
                            exec.undo_buffer.resize(new_capacity);
                            writeln!(writer, "CPU undo buffer capacity: {} -> {}", old_capacity, exec.undo_buffer.capacity()).unwrap();
                            return Ok(());
                        }
                        _ => {}
                    }
                }

                // If not a special command, treat as step-back count
                let count = if !actual_args.is_empty() {
                    actual_args[0].parse().unwrap_or(1)
                } else {
                    1
                };

                let mut exec = self.try_lock_executor()?;

                if !exec.undo_buffer.can_undo(count) {
                    return Err(format!("Cannot undo {} steps (only {} available)", count, exec.undo_buffer.count));
                }

                // Clone the snapshot from 'count' steps back to avoid borrow issues
                if let Some(snapshot) = exec.undo_buffer.get(count).cloned() {
                    // Restore CPU state
                    exec.restore_snapshot(&snapshot);

                    // Restore memory writes in reverse order
                    for mem_write in snapshot.memory_writes.iter().rev() {
                        let _ = exec.debug_write(mem_write.virt_addr, mem_write.old_value, mem_write.size, 0);
                    }

                    // Pop the consumed snapshots so subsequent undos go further back.
                    for _ in 0..count {
                        exec.undo_buffer.pop();
                    }

                    writeln!(writer, "Undid {} instruction(s), PC now at {:016x}", count, exec.core.pc).unwrap();
                    Ok(())
                } else {
                    Err(format!("Failed to retrieve undo snapshot"))
                }
                }
                #[cfg(not(feature = "developer"))]
                Err("undo requires a developer build".to_string())
            }
            "tlb" => {
                if actual_args.is_empty() { return Err("Usage: tlb <dump|trans|debug> ...".to_string()); }
                let exec = self.try_lock_executor()?;
                let tlb = &exec.tlb;

                match actual_args[0] {
                    "dump" => {
                        for i in 0..tlb.num_entries() {
                            let entry_str = tlb.format_entry(i);
                            if !entry_str.is_empty() {
                                writeln!(writer, "{}", entry_str).unwrap();
                            }
                        }
                    }
                    "trans" | "translate" => {
                        if actual_args.len() < 2 { return Err("Usage: tlb trans <vaddr> [asid]".to_string()); }
                        let vaddr = u64::from_str_radix(actual_args[1].trim_start_matches("0x"), 16)
                            .map_err(|_| "Invalid address".to_string())?;
                        let asid = if actual_args.len() > 2 {
                            u8::from_str_radix(actual_args[2].trim_start_matches("0x"), 16)
                                .map_err(|_| "Invalid ASID".to_string())?
                        } else {
                            0
                        };
                        writeln!(writer, "{}", tlb.debug_translate(vaddr, asid)).unwrap();
                    }
                    _ => return Err("Unknown TLB subcommand".to_string()),
                }
                Ok(())
            }
            #[cfg(feature = "jitv2")]
            "j2" => {
                if actual_args.is_empty() { return Err("Usage: j2 <analyze <addr>|pcp [addr]|status|inline|dispatch|fallback|instrs|threads|opt|min-instrs|max-instrs|min-calls|lockstep|hugepages|flush>".to_string()); }
                // "flush" needs the CPU genuinely stopped, not just this
                // lock momentarily free — try_lock_executor() succeeding
                // only proves no one holds the lock *right now* (MipsCpu::step
                // releases it between batches), which isn't enough: a
                // manual flush must be certain nothing is mid-dispatch
                // through JIT-compiled code referencing the pool it's about
                // to clear. Rather than stop/start the CPU ourselves here
                // (this runs on the monitor console thread, not the CPU
                // thread — flush_from_cpu_thread's "as good as stopped"
                // reasoning doesn't apply), just require the developer to
                // have already run `stop` first, same as they'd need to for
                // `j2 pcp` to see anything meaningful anyway.
                if actual_args[0] == "flush" {
                    if self.is_running() {
                        return Err("j2 flush: stop the CPU first (`stop`) — flushing while it's running could race a concurrently-executing compiled function".to_string());
                    }
                    let mut exec = self.try_lock_executor()?;
                    let bus = exec.sysad.clone();
                    unsafe { exec.jitv2.lock().flush_from_cpu_thread(bus); }
                    // Mirrors jitv2_track_pcp's own post-flush recovery —
                    // self.pcp would otherwise dangle into the just-cleared
                    // pool.
                    exec.pcp = std::ptr::null_mut();
                    writeln!(writer, "jitv2: manual flush complete").unwrap();
                    return Ok(());
                }
                let mut exec = self.try_lock_executor()?;

                match actual_args[0] {
                    "inline" => {
                        match actual_args.get(1).copied() {
                            None => {
                                writeln!(writer, "j2 inline compile: {}", if exec.jitv2_inline_compile { "on" } else { "off" }).unwrap();
                            }
                            Some("on") => {
                                #[cfg(feature = "lightning")]
                                return Err("j2 inline on: unavailable under `lightning` — compiles always go to the async queue, never inline".to_string());
                                // Switching TO inline: stop the async
                                // compile queue (if running) and build a
                                // fresh Codegen for Jitv2::codegen's idle
                                // slot, reusing the queue's own still-live
                                // shared arena rather than reserving a new
                                // one — no leaked reservation, and inline
                                // dispatch picks up right where the queue
                                // left off, address-space-wise. The queue's
                                // own (now-discarded) Codegen wrapper is
                                // simply dropped; its arena lives on via the
                                // new Codegen built on top of it below.
                                #[cfg(not(feature = "lightning"))]
                                {
                                    // Lock `jitv2` ONCE for both stop() and the
                                    // codegen hand-back. The previous form
                                    // `if let Some(cg) = exec.jitv2.lock().compile_queue.stop() { *exec.jitv2.lock()... }`
                                    // deadlocked: the first `.lock()` temporary
                                    // in the `if let` scrutinee lives until the
                                    // whole `if let` completes, so the inner
                                    // `exec.jitv2.lock()` re-locked the same
                                    // (non-reentrant) mutex — hanging the
                                    // console thread the instant `j2 inline on`
                                    // ran, before the CPU even started.
                                    {
                                        let mut jit = exec.jitv2.lock();
                                        let stopped = jit.compile_queue.stop();
                                        if let Some(old) = stopped.into_iter().next() {
                                            let (shared, state) = old.shared_arena();
                                            drop(old);
                                            *jit.codegen.lock() = Some(crate::jitv2::codegen::Codegen::new_with_shared_arena(shared, state));
                                        }
                                    }
                                    exec.jitv2_inline_compile = true;
                                    writeln!(writer, "j2 inline compile: on").unwrap();
                                }
                            }
                            Some("off") => {
                                // Switching TO threaded: reuse the inline
                                // slot's own still-live shared arena for the
                                // queue's new worker Codegen (built inside
                                // start() — see that method's own doc
                                // comment) rather than reserving a fresh
                                // one. The old Jitv2::codegen wrapper is
                                // simply dropped once its arena handle is
                                // extracted; the arena itself lives on.
                                exec.jitv2_inline_compile = false;
                                let bus = exec.sysad.clone();
                                let old = exec.jitv2.lock().codegen.lock().take();
                                match old {
                                    Some(old) => {
                                        let (shared, state) = old.shared_arena();
                                        drop(old);
                                        let mut jit = exec.jitv2.lock();
                                        let stats = jit.stats.clone();
                                        jit.compile_queue.start_with_shared_arena(bus, stats, shared, state);
                                    }
                                    None => {
                                        return Err("j2 inline off: no Codegen available (already owned by a running compile queue?)".to_string());
                                    }
                                }
                                writeln!(writer, "j2 inline compile: off").unwrap();
                            }
                            Some(_) => return Err("Usage: j2 inline [on|off]".to_string()),
                        }
                    }
                    "dispatch" => {
                        match actual_args.get(1).copied() {
                            None => {
                                writeln!(writer, "j2 dispatch: {}", if exec.jitv2_dispatch_enabled { "on" } else { "off" }).unwrap();
                            }
                            Some("on") => {
                                exec.jitv2_dispatch_enabled = true;
                                writeln!(writer, "j2 dispatch: on").unwrap();
                            }
                            Some("off") => {
                                #[cfg(feature = "lightning")]
                                return Err("j2 dispatch off: unavailable under `lightning` — JIT dispatch is always on".to_string());
                                #[cfg(not(feature = "lightning"))]
                                {
                                    exec.jitv2_dispatch_enabled = false;
                                    writeln!(writer, "j2 dispatch: off (real JIT dispatch gate skipped entirely, interpreter-only)").unwrap();
                                }
                            }
                            Some(_) => return Err("Usage: j2 dispatch [on|off]".to_string()),
                        }
                    }
                    "fallback" => {
                        // Interpreter-fallback region admission (analyzer's
                        // FALLBACK_ENABLED global): when on, an unsupported
                        // (Excluded) instruction is kept in a compiled region
                        // as a fallback head run through the interpreter,
                        // instead of ending the region. A process-global read
                        // by the analyzer at compile time — already-compiled
                        // regions are unaffected until re-compiled, so a `j2
                        // flush` (CPU stopped) is required for the change to
                        // take effect. Defaults off while the path is being
                        // validated against live boots.
                        match actual_args.get(1).copied() {
                            None => {
                                writeln!(writer, "j2 fallback: {}", if crate::jitv2::analyzer::fallback_enabled() { "on" } else { "off" }).unwrap();
                            }
                            Some(on @ ("on" | "off")) => {
                                crate::jitv2::analyzer::set_fallback_enabled(on == "on");
                                writeln!(writer, "j2 fallback: {} — run `j2 flush` (CPU stopped) for it to take effect on already-compiled regions", on).unwrap();
                            }
                            Some(_) => return Err("Usage: j2 fallback [on|off]".to_string()),
                        }
                    }
                    cat @ ("alu" | "fpu" | "branch" | "loadstore" | "cop0") => {
                        // Category gate: bulk on/off over opcode_support's per-InstrKind
                        // runtime enable table (ENABLED) — every InstrKind whose category()
                        // intersects this one gets set together. Same underlying table a
                        // future `j2 instr <name> [on|off]` will flip one bit of; disabling
                        // a whole category and then re-enabling one instruction under test
                        // composes naturally since there's only ever one table. Same
                        // process-global-read-at-compile-time contract as `j2 fallback`:
                        // already-compiled regions are unaffected until a `j2 flush`. Useful
                        // for bisecting a live-boot divergence into one functional unit
                        // before narrowing further.
                        let bit = match cat {
                            "alu" => crate::mips_instr_stats::InstrCategory::ALU,
                            "fpu" => crate::mips_instr_stats::InstrCategory::FPU,
                            "branch" => crate::mips_instr_stats::InstrCategory::BRANCH,
                            "loadstore" => crate::mips_instr_stats::InstrCategory::LOADSTORE,
                            "cop0" => crate::mips_instr_stats::InstrCategory::COP0,
                            _ => unreachable!(),
                        };
                        match actual_args.get(1).copied() {
                            None => {
                                let on = crate::jitv2::opcode_support::category_enabled(bit);
                                writeln!(writer, "j2 {}: {}", cat, if on { "on" } else { "off" }).unwrap();
                            }
                            Some(on @ ("on" | "off")) => {
                                crate::jitv2::opcode_support::set_category_enabled(bit, on == "on");
                                writeln!(writer, "j2 {}: {} — run `j2 flush` (CPU stopped) for it to take effect on already-compiled regions", cat, on).unwrap();
                            }
                            Some(_) => return Err(format!("Usage: j2 {} [on|off]", cat)),
                        }
                    }
                    "instrs" => {
                        // Inspection dump for the per-instruction toggle table above —
                        // without this there's no way to see which instructions are
                        // currently enabled/disabled short of reading the source, which
                        // defeats the point of a runtime bisection tool.
                        let filter = match actual_args.get(1).copied() {
                            None => None,
                            Some("alu") => Some(crate::mips_instr_stats::InstrCategory::ALU),
                            Some("fpu") => Some(crate::mips_instr_stats::InstrCategory::FPU),
                            Some("branch") => Some(crate::mips_instr_stats::InstrCategory::BRANCH),
                            Some("loadstore") => Some(crate::mips_instr_stats::InstrCategory::LOADSTORE),
                            Some("cop0") => Some(crate::mips_instr_stats::InstrCategory::COP0),
                            Some(_) => return Err("Usage: j2 instrs [alu|fpu|branch|loadstore|cop0]".to_string()),
                        };
                        let _ = crate::jitv2::opcode_support::write_status(&mut writer, filter);
                    }
                    "threads" => {
                        // Read-only: thread_count is fixed at process
                        // startup (iris.toml's jitv2.threads or
                        // --jitv2-threads), never mutable at runtime — see
                        // CompileQueue::set_thread_count's own doc comment.
                        if actual_args.len() > 1 {
                            return Err("j2 threads is read-only — set via iris.toml [jitv2] threads or --jitv2-threads at startup".to_string());
                        }
                        let jit = exec.jitv2.lock();
                        writeln!(writer, "j2 threads: {}", jit.compile_queue.thread_count()).unwrap();
                    }
                    "opt" => {
                        // Process-wide, not per-Codegen (opt_level is baked
                        // into the ISA Flags at JITModule-construction time)
                        // — only takes effect on the next Codegen::new()/
                        // reset() (a flush), not retroactively for functions
                        // already compiled. See Codegen::set_opt_level_speed's
                        // own doc comment.
                        match actual_args.get(1).copied() {
                            None => {
                                writeln!(writer, "j2 opt: {}", if crate::jitv2::codegen::Codegen::opt_level_speed() { "speed" } else { "none" }).unwrap();
                            }
                            Some("speed") => {
                                crate::jitv2::codegen::Codegen::set_opt_level_speed(true);
                                writeln!(writer, "j2 opt: speed (takes effect on the next flush/reset)").unwrap();
                            }
                            Some("none") => {
                                crate::jitv2::codegen::Codegen::set_opt_level_speed(false);
                                writeln!(writer, "j2 opt: none (takes effect on the next flush/reset)").unwrap();
                            }
                            Some(_) => return Err("Usage: j2 opt [none|speed]".to_string()),
                        }
                    }
                    "min-instrs" => {
                        // Applies to future compile decisions only — an
                        // offset already denylisted for being too short
                        // under the old threshold stays denylisted until its
                        // page's next gen bump (same as every other sticky
                        // rejection, §6.4); lowering the threshold doesn't
                        // retroactively un-denylist anything.
                        match actual_args.get(1).copied() {
                            None => {
                                writeln!(writer, "j2 min-instrs: {}", crate::jitv2::comp::min_instrs_to_compile()).unwrap();
                            }
                            Some(n) => match n.parse::<usize>() {
                                Ok(n) => {
                                    crate::jitv2::comp::set_min_instrs_to_compile(n);
                                    writeln!(writer, "j2 min-instrs: {}", crate::jitv2::comp::min_instrs_to_compile()).unwrap();
                                }
                                Err(_) => return Err("Usage: j2 min-instrs [N]".to_string()),
                            },
                        }
                    }
                    "max-instrs" => {
                        // Same sticky-denylist caveat as min-instrs: applies
                        // to future compile decisions only, doesn't
                        // retroactively affect anything already compiled or
                        // denylisted under the old budget.
                        match actual_args.get(1).copied() {
                            None => {
                                writeln!(writer, "j2 max-instrs: {}", crate::jitv2::comp::max_instrs_per_compile()).unwrap();
                            }
                            Some(n) => match n.parse::<usize>() {
                                Ok(n) => {
                                    crate::jitv2::comp::set_max_instrs_per_compile(n);
                                    writeln!(writer, "j2 max-instrs: {}", crate::jitv2::comp::max_instrs_per_compile()).unwrap();
                                }
                                Err(_) => return Err("Usage: j2 max-instrs [N]".to_string()),
                            },
                        }
                    }
                    "min-calls" => {
                        // Only applies to exec_decoded's real dispatch gate,
                        // on the async (non-inline) compile path — see
                        // count_dispatch_and_check_threshold's own doc
                        // comment for why jitv2_inline_compile/tests/lockstep
                        // are exempt by construction, not by this value.
                        match actual_args.get(1).copied() {
                            None => {
                                writeln!(writer, "j2 min-calls: {}", crate::jitv2::min_calls_before_compile()).unwrap();
                            }
                            Some(n) => match n.parse::<u64>() {
                                Ok(n) => {
                                    crate::jitv2::set_min_calls_before_compile(n);
                                    writeln!(writer, "j2 min-calls: {}", crate::jitv2::min_calls_before_compile()).unwrap();
                                }
                                Err(_) => return Err("Usage: j2 min-calls [N]".to_string()),
                            },
                        }
                    }
                    #[cfg(feature = "jitv2_lockstep")]
                    "lockstep" => {
                        // Lockstep is not a runtime toggle anymore: if the build
                        // has the jitv2_lockstep feature, every verifiable
                        // (straight-line ALU/load-store/CP1) JIT instruction
                        // self-checks against the interpreter, always — it's
                        // driven from the JIT block itself. Kept as an
                        // informational command.
                        #[cfg(feature = "jitv2_lockstep")]
                        writeln!(writer, "j2 lockstep: always on (built with jitv2_lockstep). Straight-line ALU/load-store/CP1 instructions self-check every dispatch; branch/jump not yet verified.").unwrap();
                        #[cfg(not(feature = "jitv2_lockstep"))]
                        writeln!(writer, "j2 lockstep: not built (rebuild with the jitv2_lockstep feature to enable).").unwrap();
                    }
                    "analyze" => {
                        // Runs the REAL analyzer walk (Analyzer::walk_bounded,
                        // the same one handle_request uses to decide what a
                        // region actually contains) against live memory, and
                        // prints every visited word with its disassembly and
                        // classification — the static-analysis counterpart to
                        // `j2 pcp` (which only shows what's already been
                        // compiled/published). Built for exactly the kind of
                        // question this session kept needing answered ad hoc
                        // via temporary eprintln instrumentation: "starting
                        // from this exact word, what does the walk include,
                        // and why does it stop where it does" — now a
                        // permanent, no-rebuild-required tool.
                        //
                        // Takes a PHYSICAL address, not virtual: translation
                        // isn't always available (CPU stopped mid-exception,
                        // page not currently TLB-mapped, or the address came
                        // from a log/j2 pcp dump that already gives phys) —
                        // reading via `sysad.read32` directly (the same
                        // primitive `handle_request` itself uses) sidesteps
                        // needing a live translation at all.
                        if actual_args.len() < 2 {
                            return Err("Usage: j2 analyze <phys_addr>".to_string());
                        }
                        let symbols_arc = exec.symbols.clone();
                        let symbols = symbols_arc.lock();
                        let phys_addr = parse_cpu_arg(actual_args[1], &exec.core, Some(&symbols))? as u32;
                        let entry_word = ((phys_addr & 0xFFF) >> 2) as u16;
                        let page_base_phys = phys_addr & !0xFFFu32;

                        let mut words = [0u32; crate::jitv2::ENTRIES_PER_PAGE];
                        for (i, w) in words.iter_mut().enumerate() {
                            let a = page_base_phys + (i as u32) * 4;
                            let r = exec.sysad.read32(a);
                            if !r.is_ok() {
                                return Err(format!("failed to read word at phys {:#010x} (offset {:#x} into page)", a, i * 4));
                            }
                            *w = r.data;
                        }

                        let (walked, non_empty) = exec.jitv2_inline_analyzer.walk_bounded(&words, entry_word, page_base_phys, usize::MAX);
                        writeln!(writer, "analyze phys={:#010x}: page_base_phys={:#010x} entry_page_off={:#05x}", phys_addr, page_base_phys, entry_word * 4).unwrap();
                        if !non_empty {
                            writeln!(writer, "  region is EMPTY — entry word itself is Excluded with no emitter and fallback is off, or otherwise undispatchable").unwrap();
                        } else {
                            let visited: Vec<_> = crate::jitv2::analyzer::instrs_linear(walked).collect();
                            writeln!(writer, "  {} instructions visited:", visited.len()).unwrap();
                            for instr in &visited {
                                let paddr = page_base_phys + (instr.word as u32) * 4;
                                let dis = mips_dis::disassemble(instr.raw, paddr as u64, Some(&symbols));
                                let mut tags = Vec::new();
                                if instr.word == entry_word { tags.push("ENTRY"); }
                                if instr.is_fallback { tags.push("fallback"); }
                                if instr.is_branch_fallback_successor { tags.push("branch-fallback-successor"); }
                                if instr.is_slot_only { tags.push("slot-only"); }
                                let tag_str = if tags.is_empty() { String::new() } else { format!(" [{}]", tags.join(",")) };
                                let exits = match (instr.fallthrough_exit, instr.taken_exit) {
                                    (None, None) => String::new(),
                                    (ft, tk) => format!(" (fallthrough_exit={:?} taken_exit={:?})", ft, tk),
                                };
                                writeln!(writer, "    page_off={:#05x} phys={:#010x}: {:08x} {}{}{}", instr.word * 4, paddr, instr.raw, dis, tag_str, exits).unwrap();
                            }
                        }
                    }
                    "pcp" => {
                        // Optional address argument: inspect an arbitrary
                        // page's compile state (e.g. "why does this function
                        // never show up as JIT-tagged in dt") instead of only
                        // whatever page execution is currently stopped on —
                        // previously silently ignored (this whole command
                        // took no arguments at all), which made it impossible
                        // to check a page other than the live one.
                        // actual_args[0] is "pcp" itself here (this arm is
                        // already inside `match actual_args[0]`) — the
                        // address, if given, is actual_args[1]. Passing [0]
                        // here fed the literal string "pcp" to parse_cpu_arg,
                        // which tried (and failed) to resolve it as a symbol
                        // name — "Symbol not found: pcp" — found live.
                        let explicit_addr = if let Some(&addr_arg) = actual_args.get(1) {
                            let symbols_arc = exec.symbols.clone();
                            let symbols = symbols_arc.lock();
                            Some(parse_cpu_arg(addr_arg, &exec.core, Some(&symbols))?)
                        } else {
                            None
                        };
                        // `self.pcp` is null whenever the CPU is stopped
                        // (`MipsCpu::stop()`'s `nanotlb_invalidate()` — see
                        // its own doc comment for why that invariant exists)
                        // — which is exactly when a developer is most likely
                        // to run this command. Rather than just reporting
                        // "nothing tracked," re-derive the page the same way
                        // a real fetch would: translate the target PC with
                        // `debug_translate` (side-effect-free — no TLB/nanotlb
                        // state touched) and look it up via `page_for`, which
                        // finds the existing pool entry if this PFN was ever
                        // seen before (the overwhelmingly common case here —
                        // we're inspecting whatever page execution just
                        // stopped on, or an explicitly requested one) without
                        // disturbing `self.pcp` itself. An explicit address
                        // always re-derives (never trusts self.pcp, which
                        // tracks a different page whenever one was asked for).
                        let page_ptr = if explicit_addr.is_none() && !exec.pcp.is_null() {
                            exec.pcp
                        } else {
                            let pc = explicit_addr.unwrap_or(exec.core.pc);
                            let result = exec.debug_translate(pc);
                            if result.is_exception() {
                                writeln!(writer, "No PhysicalCodePage tracked (self.pcp is null) and PC {:#018x} doesn't translate.", pc).unwrap();
                                return Ok(());
                            }
                            let phys_addr = result.phys;
                            let pfn = phys_addr / crate::jitv2::PAGE_SIZE;
                            let page_base = pfn * crate::jitv2::PAGE_SIZE;
                            let sysad = exec.sysad.clone();
                            let mut jit = exec.jitv2.lock();
                            match jit.page_for(pfn, page_base, sysad.as_ref()) {
                                Some(slot) => jit.page_ptr(slot),
                                None => {
                                    writeln!(writer, "No PhysicalCodePage tracked (self.pcp is null) and the page pool is full (can't even allocate a fresh lookup entry).").unwrap();
                                    return Ok(());
                                }
                            }
                        };
                        // Safety: try_lock_executor holds the same lock every
                        // mutator of jitv2/pcp takes, and pcp is only ever
                        // reassigned or nulled by the exec thread under that
                        // lock (jitv2_track_pcp, nanotlb_invalidate) — safe
                        // to dereference for the duration of this borrow.
                        // `page_ptr` (the re-derived case) points into the
                        // same `Jitv2::pages` pool under the same lock
                        // discipline, so the same reasoning applies to it.
                        let page = unsafe { &*page_ptr };
                        let pc = explicit_addr.unwrap_or(exec.core.pc);
                        let entry_offset = ((pc & 0xFFF) >> 2) as usize;

                        writeln!(writer, "pfn={:#010x}  gen={}", page.pfn, page.current_gen()).unwrap();
                        writeln!(writer, "pc={:#018x}  page_off={:#05x}", pc, entry_offset * 4).unwrap();
                        writeln!(
                            writer,
                            "  entry_offset: published={} entry_valid={} denylisted={}",
                            page.is_published(entry_offset),
                            page.is_entry_valid(entry_offset),
                            page.is_denylisted(entry_offset),
                        ).unwrap();

                        let mut published = 0usize;
                        let mut denylisted = 0usize;
                        for off in 0..crate::jitv2::ENTRIES_PER_PAGE {
                            if page.is_published(off) { published += 1; }
                            if page.is_denylisted(off) { denylisted += 1; }
                        }
                        writeln!(
                            writer,
                            "totals: {} / {} offsets published, {} denylisted",
                            published, crate::jitv2::ENTRIES_PER_PAGE, denylisted,
                        ).unwrap();

                        // Every published entry: word offset, the guest
                        // virtual address it compiles (vbase | offset*4 —
                        // vbase taken from the live PC's own page, same
                        // derivation the exit block uses), and the compiled
                        // function pointer. Denylisted offsets follow the
                        // same way, without a func pointer (they have none).
                        let vbase = pc & !0xFFFu64;
                        if published > 0 {
                            writeln!(writer, "  published entries (vaddr -> func):").unwrap();
                            for off in 0..crate::jitv2::ENTRIES_PER_PAGE {
                                if !page.is_published(off) { continue; }
                                let vaddr = vbase | ((off as u64) * 4);
                                let func = page.entries[off].func;
                                let gen = page.entries[off].gen.load(Ordering::Relaxed);
                                let stale = gen != page.current_gen();
                                #[cfg(feature = "developer")]
                                let dev_cols = format!(" instrs={} code_size={} calls={} in_blocks={}",
                                    page.entries[off].instr_count,
                                    page.entries[off].code_size,
                                    page.entries[off].call_count.load(Ordering::Relaxed),
                                    page.entries[off].block_include_count.load(Ordering::Relaxed));
                                #[cfg(not(feature = "developer"))]
                                let dev_cols = String::new();
                                writeln!(
                                    writer,
                                    "    page_off={:#05x} vaddr={:#018x} func={:#014x} gen={}{}{}",
                                    off * 4, vaddr, func as usize, gen,
                                    dev_cols,
                                    if stale { " STALE" } else { "" },
                                ).unwrap();
                            }
                        }
                        #[cfg(feature = "developer")]
                        {
                            let mut by_calls: Vec<(usize, u64, u16)> = (0..crate::jitv2::ENTRIES_PER_PAGE)
                                .filter(|&off| page.is_published(off))
                                .map(|off| (off, page.entries[off].call_count.load(Ordering::Relaxed), page.entries[off].instr_count))
                                .filter(|&(_, calls, _)| calls > 0)
                                .collect();
                            if !by_calls.is_empty() {
                                by_calls.sort_by(|a, b| b.1.cmp(&a.1));
                                writeln!(writer, "  hottest entries (page_off, calls, instrs):").unwrap();
                                for &(off, calls, instrs) in by_calls.iter().take(16) {
                                    let vaddr = vbase | ((off as u64) * 4);
                                    writeln!(writer, "    page_off={:#05x} vaddr={:#018x} calls={} instrs={}", off * 4, vaddr, calls, instrs).unwrap();
                                }
                            }

                            // Block-overlap: words included in more than one
                            // compiled region (block_include_count > 1),
                            // regardless of whether the word is itself a
                            // published entry — a word pulled into several
                            // overlapping blocks from different entry points
                            // shows the redundant-recompilation cost of the
                            // same straight-line code directly. Saturates at
                            // 255 ("255+").
                            let mut overlaps: Vec<(usize, u8)> = (0..crate::jitv2::ENTRIES_PER_PAGE)
                                .map(|off| (off, page.entries[off].block_include_count.load(Ordering::Relaxed)))
                                .filter(|&(_, n)| n > 1)
                                .collect();
                            if !overlaps.is_empty() {
                                overlaps.sort_by(|a, b| b.1.cmp(&a.1));
                                writeln!(writer, "  block overlap (page_off, times included in a block):").unwrap();
                                for &(off, n) in overlaps.iter().take(32) {
                                    let vaddr = vbase | ((off as u64) * 4);
                                    let pegged = if n == u8::MAX { "+" } else { "" };
                                    writeln!(writer, "    page_off={:#05x} vaddr={:#018x} in_blocks={}{}", off * 4, vaddr, n, pegged).unwrap();
                                }
                            }
                        }
                        if denylisted > 0 {
                            let denylisted_words: Vec<String> = (0..crate::jitv2::ENTRIES_PER_PAGE)
                                .filter(|&off| page.is_denylisted(off))
                                .map(|off| format!("{:#05x}", off))
                                .collect();
                            writeln!(writer, "  denylisted words: {}", denylisted_words.join(", ")).unwrap();
                        }
                    }
                    "stats" | "status" => {
                        let jit = exec.jitv2.lock();
                        writeln!(writer, "pages: {} / {} used", jit.pages_used(), jit.capacity()).unwrap();
                        // Report the EFFECTIVE inline state: jitv2_lockstep forces
                        // it on regardless of the runtime field (see exec_decoded's
                        // `inline_compile`), so printing the raw field would
                        // misleadingly say "off" on a lockstep build that's
                        // actually inline-compiling every region.
                        #[cfg(feature = "jitv2_lockstep")]
                        {
                            writeln!(writer, "inline compile: on (forced by jitv2_lockstep)").unwrap();
                            // Positive proof lockstep is actually verifying: this
                            // counts real JIT-vs-interpreter compares. Stuck at 0
                            // means the JIT never ran (nothing compiled) and
                            // verification is silently OFF.
                            let verified = crate::mips_exec::LOCKSTEP_VERIFICATIONS.load(Ordering::Relaxed);
                            writeln!(writer, "lockstep: {} instructions verified (0 => NOT verifying — JIT not running)", verified).unwrap();
                        }
                        #[cfg(not(feature = "jitv2_lockstep"))]
                        writeln!(writer, "inline compile: {}", if exec.jitv2_inline_compile { "on" } else { "off" }).unwrap();
                        #[cfg(feature = "developer")]
                        {
                            let dt_calls = crate::mips_exec::DEV_TRACE_BP_CALLS.load(Ordering::Relaxed);
                            writeln!(writer, "dev trace hook: {} calls (0 => emit_dev_trace_bp never reached from compiled code)", dt_calls).unwrap();
                        }
                        // Functions compiled into the active Codegen's
                        // Cranelift arena since the last mega_flush — a
                        // diagnostic count only now; the actual flush trigger
                        // is CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES (real bytes
                        // reserved), printed separately below. Branch on
                        // whether the pool is actually spawned, not on
                        // `jit.codegen`'s Option — that slot stays `Some`
                        // the entire time the pool runs (reserved for
                        // `j2 inline on` to take later, see `Machine::new`'s
                        // own comment), so it's never a reliable "who owns
                        // compilation right now" signal on its own.
                        if jit.compile_queue.is_running() {
                            writeln!(writer, "codegen: {} functions compiled (owned by async compile thread pool, {} threads)",
                                jit.compile_queue.function_count(), jit.compile_queue.thread_count()).unwrap();
                        } else if let Some(n) = jit.codegen.lock().as_ref().map(|c| c.function_count()) {
                            writeln!(writer, "codegen: {} functions compiled", n).unwrap();
                        }
                        {
                            // Real per-worker arena byte counts have no
                            // cross-thread mirror today (only function_count
                            // does — see CompileQueue::function_count's own
                            // doc comment) — only readable here while the
                            // pool isn't running (the idle inline slot).
                            let stats = if jit.compile_queue.is_running() {
                                None
                            } else {
                                jit.codegen.lock().as_ref().map(|c| c.packing_stats())
                            };
                            if let Some((used, reserved)) = stats {
                                writeln!(writer, "arena reserved: {} / {} bytes ({:.1}%), used {} bytes ({:.1}% packing) — real flush trigger",
                                    reserved, crate::jitv2::CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES,
                                    reserved as f64 * 100.0 / crate::jitv2::CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES as f64,
                                    used, if reserved == 0 { 0.0 } else { used as f64 * 100.0 / reserved as f64 }).unwrap();
                            } else if jit.compile_queue.is_running() {
                                writeln!(writer, "arena reserved: n/a while pooled (no cross-thread byte mirror yet — see function_count for a live signal instead)").unwrap();
                            }
                            // Mega-flush count: each flush wipes ALL published
                            // entries, forcing everything to recompile. A count
                            // that keeps climbing during steady-state execution
                            // is the arena-fill/flush/recompile death spiral —
                            // the working set exceeds the arena threshold.
                            let flushes = crate::jit_feedback::JIT_FEEDBACK.flush_events.load(Ordering::Relaxed);
                            writeln!(writer, "mega-flushes (arena/pool wipes, each recompiles everything): {}", flushes).unwrap();
                        }
                        #[cfg(feature = "developer")]
                        {
                            let code_bytes = jit.code_bytes_used();
                            writeln!(writer, "arena bytes (host-page-rounded, ~{}KiB/fn floor): {} ({:.1} KiB) across published entries — best-effort proxy for actual Cranelift arena size, not the arena's own byte count (cranelift_jit::Memory exposes none)",
                                crate::jitv2::codegen::Codegen::HOST_PAGE_SIZE / 1024, code_bytes, code_bytes as f64 / 1024.0).unwrap();
                            let compiles = jit.stats.compiles.load(Ordering::Relaxed);
                            let failed = jit.stats.failed_compiles.load(Ordering::Relaxed);
                            let kills = jit.stats.kill_entry_calls.load(Ordering::Relaxed);
                            writeln!(writer, "compiles: {} ok, {} failed (codegen gap or analyzer-excluded), {} kill_entry_fn bailouts (FR-mismatch)",
                                compiles, failed, kills).unwrap();
                            let fb_words = jit.stats.fallback_words.load(Ordering::Relaxed);
                            let fb_regions = jit.stats.fallback_regions.load(Ordering::Relaxed);
                            writeln!(writer, "fallback: {} interp-fallback words compiled across {} regions ({})",
                                fb_words, fb_regions,
                                if crate::jitv2::analyzer::fallback_enabled() { "j2 fallback: on" } else { "j2 fallback: off" }).unwrap();
                            if failed > 0 {
                                writeln!(writer, "  rejections by reason:").unwrap();
                                for reason in [
                                    crate::jitv2::RejectReason::EntryExcluded,
                                    crate::jitv2::RejectReason::AnalyzerCodegenDisagreement,
                                    crate::jitv2::RejectReason::CraneliftVerifierError,
                                    crate::jitv2::RejectReason::TooShort,
                                ] {
                                    let n = jit.stats.reject_reasons[reason.index()].load(Ordering::Relaxed);
                                    if n == 0 { continue; }
                                    writeln!(writer, "    {:>7}  {}", n, reason.label()).unwrap();
                                }
                            }

                            let dispatches = jit.stats.compile_queue_dispatches.load(Ordering::Relaxed);
                            let full = jit.stats.compile_queue_full.load(Ordering::Relaxed);
                            let depth_sum = jit.stats.compile_queue_depth_sum.load(Ordering::Relaxed);
                            if dispatches > 0 {
                                let avg_depth = depth_sum as f64 / dispatches as f64;
                                let full_pct = full as f64 * 100.0 / dispatches as f64;
                                writeln!(writer, "compile queue: {} / {} now, {} dispatches, {} full ({:.1}%), avg depth at dispatch {:.1}",
                                    jit.compile_queue.queue_occupancy(), crate::jitv2::COMPILE_QUEUE_CAPACITY,
                                    dispatches, full, full_pct, avg_depth).unwrap();
                            }

                            let pending_threshold = jit.stats.batch_flushes_pending_threshold.load(Ordering::Relaxed);
                            let queue_drain = jit.stats.batch_flushes_queue_drain.load(Ordering::Relaxed);
                            if pending_threshold > 0 || queue_drain > 0 {
                                let max_pending = jit.stats.batch_max_pending.load(Ordering::Relaxed);
                                writeln!(writer, "force-seal sweeps: {} pending-threshold, {} queue-drain, largest backlog {} entries",
                                    pending_threshold, queue_drain, max_pending).unwrap();
                            }
                            // Packing quality (bytes actually used by compiled
                            // code vs. bytes reserved for it, per the arena's
                            // own host-page segments) — only readable while
                            // this thread holds the real Codegen directly
                            // (inline mode, or the worker not yet started);
                            // when the async worker owns it, there's no lock
                            // a status command should contend for just to
                            // print a diagnostic (same reasoning as
                            // `function_count`'s own mirror-vs-direct split
                            // above, but packing_stats isn't mirrored since
                            // it's purely informational, not something the
                            // hot compile path needs a lock-free read of).
                            if let Some(codegen) = jit.codegen.lock().as_ref() {
                                let (used, reserved) = codegen.packing_stats();
                                if reserved > 0 {
                                    let pct = used as f64 * 100.0 / reserved as f64;
                                    writeln!(writer, "  packing: {} / {} bytes used ({:.1}%) across this Codegen's host-page segments",
                                        used, reserved, pct).unwrap();
                                }
                            }

                            // Real distribution of published regions' instruction
                            // counts, scanned across every pooled page — ground
                            // truth against MAX_INSTRS_PER_COMPILE (comp.rs),
                            // since a region can land shorter than the budget
                            // (branch/exclusion/page boundary) or, for a branch's
                            // own head instruction, effectively longer once its
                            // mandatory delay slot is counted in. Paired with each
                            // bucket's own code-size distribution (avg/min/max) —
                            // shows whether code size actually scales with
                            // instruction count or is dominated by fixed
                            // per-region overhead (preamble, CU1/FR guard).
                            let hist = jit.code_size_by_instr_count();
                            let total: u32 = hist.iter().flatten().map(|b| b.count).sum();
                            let total_bytes: u64 = hist.iter().flatten().map(|b| b.sum_bytes).sum();
                            let total_instrs: u64 = hist.iter().enumerate()
                                .filter_map(|(n, b)| b.map(|b| n as u64 * b.count as u64))
                                .sum();
                            if total > 0 {
                                writeln!(writer, "instruction-count histogram ({} published entries):", total).unwrap();
                                for (n, bucket) in hist.iter().enumerate() {
                                    let Some(b) = bucket else { continue };
                                    let pct = b.count as f64 * 100.0 / total as f64;
                                    let avg = b.sum_bytes as f64 / b.count as f64;
                                    writeln!(writer, "  {:>3} instr: {:>7}  ({:5.1}%)  code bytes avg={:>6.1} min={:>5} max={:>5}",
                                        n, b.count, pct, avg, b.min_bytes, b.max_bytes).unwrap();
                                }
                                if total_instrs > 0 {
                                    writeln!(writer, "  overall: {:.1} code bytes/instruction ({} bytes across {} instructions)",
                                        total_bytes as f64 / total_instrs as f64, total_bytes, total_instrs).unwrap();
                                }
                            }
                        }
                    }
                    #[cfg(all(target_os = "linux", feature = "developer"))]
                    "hugepages" => {
                        let jit = exec.jitv2.lock();
                        let (ptr, len) = jit.codegen.lock().as_ref()
                            .map(|c| c.arena_range())
                            .unwrap_or((0, 0));
                        if len == 0 {
                            writeln!(writer, "jitv2 arena not yet allocated (no compile has run) or owned by the async compile thread right now — try again after `stop`/a compile.").unwrap();
                            return Ok(());
                        }
                        writeln!(writer, "arena: {:#x}..{:#x} ({} bytes, {} MiB)", ptr, ptr + len, len, len / (1024 * 1024)).unwrap();
                        match crate::jitv2::paged_memory::anon_hugepages_in_range(ptr, len) {
                            Some(hugepage_bytes) => {
                                let pct = hugepage_bytes as f64 * 100.0 / len as f64;
                                writeln!(writer, "AnonHugePages within arena: {} bytes ({} MiB, {:.1}% of reservation) — from /proc/self/smaps",
                                    hugepage_bytes, hugepage_bytes / (1024 * 1024), pct).unwrap();
                                if hugepage_bytes == 0 {
                                    writeln!(writer, "  0% is suspicious once real code has compiled and finalized — check `cat /sys/kernel/mm/transparent_hugepage/enabled` (want `madvise` or `always`, not `never`).").unwrap();
                                }
                            }
                            None => {
                                writeln!(writer, "/proc/self/smaps unavailable (sandboxed/restricted environment?) — can't verify hugepage status this way.").unwrap();
                            }
                        }
                    }
                    _ => return Err("Usage: j2 <analyze <addr>|pcp [addr]|status|inline [on|off]|dispatch [on|off]|fallback [on|off|<category>]|instrs|threads|opt [none|speed]|min-instrs [N]|max-instrs [N]|min-calls [N]|lockstep|hugepages|flush>".to_string()),
                }
                Ok(())
            }
            #[cfg(feature = "developer")]
            "trace" => {
                if actual_args.is_empty() { return Err("Usage: trace <start|stop|status>".to_string()); }
                let mut exec = self.try_lock_executor()?;

                match actual_args[0] {
                    "start" => {
                        if actual_args.len() < 2 { return Err("Usage: trace start <path>".to_string()); }
                        let path = std::path::Path::new(actual_args[1]);
                        exec.trace_start(path).map_err(|e| format!("trace start failed: {}", e))?;
                        writeln!(writer, "Recording execution trace to {}", actual_args[1]).unwrap();
                    }
                    "stop" => {
                        let count = exec.trace_stop().map_err(|e| format!("trace stop failed: {}", e))?;
                        writeln!(writer, "Trace stopped: {} records written", count).unwrap();
                    }
                    "status" => {
                        writeln!(writer, "recording: {}", exec.trace_active()).unwrap();
                    }
                    _ => return Err("Usage: trace <start|stop|status>".to_string()),
                }
                Ok(())
            }
            "dt" | "traceback" => {
                // dt [N]              — dump last N instructions to console (default 10)
                // dt file <path> [N] — dump to file (default: entire buffer)
                let (file_path, count) = if actual_args.first().copied() == Some("file") {
                    let path = actual_args.get(1).copied().unwrap_or("/tmp/iris_trace.txt");
                    let n = actual_args.get(2).and_then(|s| s.parse().ok()).unwrap_or(TRACEBACK_SIZE);
                    (Some(path.to_string()), n)
                } else {
                    let n = actual_args.first().and_then(|s| s.parse().ok()).unwrap_or(10);
                    (None, n)
                };
                let count = count.min(TRACEBACK_SIZE);
                let exec = self.executor.lock();
                let symbols = exec.symbols.lock();
                let entries = exec.traceback.get_last(count);
                let n = entries.len();
                let do_write = |w: &mut dyn Write| {
                    writeln!(w, "Execution Traceback (last {} instructions):", n).unwrap();
                    for entry in &entries {
                        let sym_str = format_pc_symbol(entry.pc, &symbols);
                        let dis = mips_dis::disassemble(entry.instr, entry.pc, Some(&symbols));
                        // Origin tag: plain interpreter entries are unmarked (the
                        // common case), every other arrival class gets its own
                        // distinct tag (see InstrOrigin::tag) so exactly how an
                        // instruction was reached — plain JIT body, entry word via
                        // back-edge, inlined delay slot, fallback, fallback
                        // successor (with or without back-edge) — stands out in
                        // the trace instead of a single collapsed [jit] bit.
                        writeln!(w, "{:#018x}{}: {:#010x} {}{}", entry.pc, sym_str, entry.instr, dis, entry.origin.tag()).unwrap();
                    }
                };
                if let Some(ref path) = file_path {
                    match std::fs::File::create(path) {
                        Ok(f) => {
                            let mut bw = std::io::BufWriter::new(f);
                            do_write(&mut bw);
                            let _ = std::io::Write::flush(&mut bw);
                            writeln!(writer, "Wrote {} instructions to {}", n, path).unwrap();
                        }
                        Err(e) => return Err(format!("Cannot open {}: {}", path, e)),
                    }
                } else {
                    do_write(&mut writer);
                }
                Ok(())
            }
            #[cfg(feature = "idle-pause")]
            "idleprof" => {
                let sub = actual_args.first().copied().unwrap_or("report");
                match sub {
                    "on" => {
                        self.idle_profile_arm();
                        writeln!(writer, "idleprof: armed (lock-free, histogram reset). Let the guest idle, then `stop; idleprof report`.").unwrap();
                    }
                    "off" => {
                        self.idle_profile_disarm();
                        writeln!(writer, "idleprof: disarmed.").unwrap();
                    }
                    "report" => {
                        let count = actual_args.get(1).and_then(|s| s.parse().ok()).unwrap_or(32);
                        let mut exec = self.try_lock_executor()?;
                        exec.idle_profile_report(count, &mut writer);
                    }
                    _ => return Err("Usage: idleprof <on | off | report [count]>".to_string()),
                }
                Ok(())
            }
            #[cfg(feature = "instr_stats")]
            "instrstats" => {
                let sub = actual_args.first().copied().unwrap_or("report");
                let mut exec = self.try_lock_executor()?;
                match sub {
                    "report" => {
                        let _ = exec.instr_stats.write_report(&mut writer);
                    }
                    "clear" => {
                        exec.instr_stats.clear();
                        writeln!(writer, "instrstats: counters cleared").unwrap();
                    }
                    "dump" => {
                        match std::fs::File::create("instr_used.txt") {
                            Ok(f) => {
                                let mut w = std::io::BufWriter::new(f);
                                let _ = exec.instr_stats.write_used_list(&mut w);
                            }
                            Err(e) => return Err(format!("Cannot open instr_used.txt: {}", e)),
                        }
                        match std::fs::File::create("instr_counts.txt") {
                            Ok(f) => {
                                let mut w = std::io::BufWriter::new(f);
                                let _ = exec.instr_stats.write_counts(&mut w);
                            }
                            Err(e) => return Err(format!("Cannot open instr_counts.txt: {}", e)),
                        }
                        writeln!(writer, "instrstats: wrote instr_used.txt, instr_counts.txt").unwrap();
                    }
                    _ => return Err("Usage: instrstats [report | clear | dump]".to_string()),
                }
                Ok(())
            }
            _ => Err(format!("Unknown CPU command: {}", actual_cmd)),
        }
    }

}

impl<T: Tlb, C: MipsCache> MipsExecutor<T, C> {
    /// Dump the hottest sampled PCs and flag a likely idle-loop region: the
    /// smallest contiguous PC window (<=256 bytes) of always-interrupts-enabled
    /// samples that together account for the bulk of execution.
    #[cfg(feature = "idle-pause")]
    pub fn idle_profile_report(&mut self, top: usize, writer: &mut dyn Write) {
        let total = self.idle_profiler.total;
        if total == 0 {
            let _ = writeln!(writer, "idleprof: no samples (run `idleprof on`, let the guest idle, then `cpu stop`)");
            return;
        }

        let mut rows: Vec<(u64, IdleSample)> =
            self.idle_profiler.hist.iter().map(|(&pc, &s)| (pc, s)).collect();
        rows.sort_by(|a, b| b.1.count.cmp(&a.1.count));

        // Pre-fetch instruction words for the displayed PCs while we still hold
        // `&mut self`; disassembly below borrows `self.symbols`, which would
        // otherwise conflict with debug_fetch_instr's `&mut self`.
        let instrs: std::collections::HashMap<u64, Option<u32>> = rows
            .iter()
            .take(top)
            .map(|(pc, _)| (*pc, self.debug_fetch_instr(*pc).ok()))
            .collect();

        let symbols = self.symbols.lock();
        let _ = writeln!(
            writer,
            "idleprof: {} samples (stride {}), {} distinct PCs — top {}:",
            total, self.idle_profiler.stride, rows.len(), top.min(rows.len())
        );
        let _ = writeln!(writer, "  {:>16}  {:>7}  {:>5}  {:>4}  symbol / disasm", "pc", "count", "pct", "ie%");
        for (pc, s) in rows.iter().take(top) {
            let pct = s.count as f64 * 100.0 / total as f64;
            let iepct = s.ie_count as f64 * 100.0 / s.count as f64;
            let sym = format_pc_symbol(*pc, &symbols);
            let dis = match instrs.get(pc).copied().flatten() {
                Some(instr) => mips_dis::disassemble(instr, *pc, Some(&symbols)),
                None => "<fetch failed>".to_string(),
            };
            let _ = writeln!(
                writer,
                "  {:016x}  {:>7}  {:4.1}%  {:3.0}%  {}{}",
                pc, s.count, pct, iepct, sym, format!("  {}", dis)
            );
        }

        // Idle-loop heuristic: among the hottest PCs that are interrupt-enabled
        // ~always, find the tightest contiguous window covering >=50% of samples.
        let mut hot: Vec<(u64, IdleSample)> = rows
            .iter()
            .filter(|(_, s)| s.ie_count * 100 >= s.count * 99) // IE >= 99%
            .cloned()
            .collect();
        hot.sort_by_key(|(pc, _)| *pc);
        let mut best: Option<(u64, u64, u64)> = None; // (lo, hi, covered)
        for i in 0..hot.len() {
            let lo = hot[i].0;
            let mut covered = 0u64;
            let mut hi = lo;
            for &(pc, s) in &hot[i..] {
                if pc.wrapping_sub(lo) > 256 {
                    break;
                }
                hi = pc;
                covered += s.count;
            }
            if covered * 2 >= total && best.map_or(true, |(_, _, c)| covered > c) {
                best = Some((lo, hi, covered));
            }
        }
        match best {
            Some((lo, hi, covered)) => {
                let pct = covered as f64 * 100.0 / total as f64;
                let sym = format_pc_symbol(lo, &symbols);
                let _ = writeln!(
                    writer,
                    "\nidle-loop candidate: {:016x}..={:016x} ({} bytes){}  — {:.1}% of samples, interrupts enabled",
                    lo, hi, hi - lo + 4, sym, pct
                );
            }
            None => {
                let _ = writeln!(
                    writer,
                    "\nno clear idle-loop candidate (no interrupts-enabled PC window covers >=50% of samples)"
                );
            }
        }
    }

    /// Analyze function prologue to determine frame size and RA save location
    fn analyze_prologue(&mut self, start_pc: u64, current_pc: u64) -> (u64, Option<(i64, usize)>) {
        let mut frame_size = 0u64;
        let mut ra_info = None;
        
        let mut pc = start_pc;
        // Limit scanning to avoid infinite loops or huge scans
        while pc < current_pc && (pc - start_pc) < 1024 {
            let instr = match self.debug_fetch_instr(pc) {
                Ok(i) => i,
                Err(_) => break,
            };

            let op = (instr >> 26) & 0x3F;
            let rs = (instr >> 21) & 0x1F;
            let rt = (instr >> 16) & 0x1F;
            let imm = (instr & 0xFFFF) as i16;

            // ADDIU sp, sp, imm (0x09) or DADDIU sp, sp, imm (0x19)
            if (op == 0x09 || op == 0x19) && rs == 29 && rt == 29 {
                // Stack adjustment: usually negative to allocate space
                let adj = imm as i64;
                if adj < 0 {
                    frame_size = frame_size.wrapping_add((-adj) as u64);
                }
            }
            // SW ra, offset(sp) (0x2B)
            else if op == 0x2B && rs == 29 && rt == 31 {
                ra_info = Some((imm as i64, 4usize));
            }
            // SD ra, offset(sp) (0x3F)
            else if op == 0x3F && rs == 29 && rt == 31 {
                ra_info = Some((imm as i64, 8usize));
            }

            pc += 4;
        }
        (frame_size, ra_info)
    }

    pub fn get_return_address(&mut self) -> Option<u64> {
        let pc = self.core.pc;
        let sp = self.core.read_gpr(29);
        let ra = self.core.read_gpr(31);

        let sym_info = {
            let symbols = self.symbols.lock();
            symbols.lookup(pc).map(|(addr, _)| addr)
        };

        if let Some(start_addr) = sym_info {
            let (_, ra_info) = self.analyze_prologue(start_addr, pc);
            if let Some((offset, size)) = ra_info {
                let save_addr = sp.wrapping_add(offset as u64);
                match self.debug_read(save_addr, size) {
                    Ok(val) => Some(val),
                    Err(_) => Some(ra),
                }
            } else {
                Some(ra)
            }
        } else {
            Some(ra)
        }
    }

    pub fn backtrace(&mut self, max_frames: usize) -> String {
        let mut output = String::new();
        let mut pc = self.core.pc;
        let mut sp = self.core.read_gpr(29);
        let ra = self.core.read_gpr(31);

        writeln!(output, "Backtrace:").unwrap();

        for i in 0..max_frames {
            let symbols = self.symbols.lock();
            let sym_info = symbols.lookup(pc);
            let sym_str = if let Some((_, name)) = sym_info {
                let offset = pc.wrapping_sub(sym_info.unwrap().0);
                format!("{} + 0x{:x}", name, offset)
            } else {
                format!("0x{:016x}", pc)
            };

            writeln!(output, "#{:02} pc=0x{:016x} sp=0x{:016x} {}", i, pc, sp, sym_str).unwrap();

            if let Some((start_addr, _)) = sym_info {
                drop(symbols); // Release lock before calling analyze_prologue

                let (frame_size, ra_info) = self.analyze_prologue(start_addr, pc);

                if frame_size > 0 {
                    let prev_sp = sp.wrapping_add(frame_size);
                    let prev_pc = if let Some((offset, size)) = ra_info {
                        let save_addr = sp.wrapping_add(offset as u64);
                        match self.debug_read(save_addr, size) {
                            Ok(val) => val,
                            Err(_) => break, // Cannot read return address
                        }
                    } else {
                        ra // Leaf function, use current RA
                    };

                    sp = prev_sp;
                    pc = prev_pc;
                    if pc == 0 { break; }
                } else {
                    // No frame size found, assume leaf or end of chain
                    if ra == 0 || ra == pc { break; }
                    pc = ra;
                }
            } else {
                break; // No symbol info
            }
        }
        output
    }
}

// ============================================================================
// Resettable + Saveable for MipsCpu (CPU core + TLB)
// ============================================================================

impl<T: Tlb + Send + 'static, C: MipsCache + Send + 'static> Resettable for MipsCpu<T, C> {
    fn power_on(&self) {
        let mut exec = self.executor.lock();
        exec.core.reset(false);
        exec.tlb.power_on();
        exec.cache.power_on();
        exec.core.in_delay_slot = false;
        exec.core.delay_slot_target = 0;
        #[cfg(feature = "developer")]
        exec.undo_buffer.clear();
        exec.traceback = TracebackBuffer::new();
        // breakpoints intentionally preserved — debugger state, not hardware state
        #[cfg(feature = "developer")]
        exec.pending_memory_writes.clear();
        exec.update_translate_fn();
        exec.update_fpr_mode();
    }
}

impl<T: Tlb + Send + 'static, C: MipsCache + Send + 'static> Saveable for MipsCpu<T, C> {
    fn save_state(&self) -> toml::Value {
        let exec = self.executor.lock();
        let c = &exec.core;
        let mut tbl = toml::map::Map::new();

        // GPRs
        tbl.insert("gpr".into(), u64_slice_to_toml(&c.gpr));
        tbl.insert("pc".into(),  hex_u64(c.pc));
        tbl.insert("hi".into(),  hex_u64(c.hi));
        tbl.insert("lo".into(),  hex_u64(c.lo));

        // CP0
        let mut cp0 = toml::map::Map::new();
        macro_rules! cp0u32 {
            ($f:ident) => { cp0.insert(stringify!($f).into(), hex_u32(c.$f)); }
        }
        macro_rules! cp0u64 {
            ($f:ident) => { cp0.insert(stringify!($f).into(), hex_u64(c.$f)); }
        }
        cp0u32!(cp0_index); cp0u32!(cp0_random); cp0u32!(cp0_wired);
        cp0u64!(cp0_count); cp0u64!(cp0_compare);
        // Timer calibration state. Without these, restore loses the kernel's
        // inferred count frequency and runs at the default 33 MHz until IRIX
        // touches Compare again — guest scheduler drifts noticeably for the
        // first few seconds after every restore. count_anchor_instant is
        // intentionally not saved: it's a host-wall anchor, not calibrated
        // state, and must be reset on load.
        cp0u64!(count_hz);
        cp0u64!(compare_delta_slow); cp0u64!(compare_delta_fast);
        cp0u32!(cp0_status); cp0u32!(cp0_cause);
        cp0u32!(cp0_prid); cp0u32!(cp0_config); cp0u32!(cp0_lladdr);
        cp0u32!(cp0_watchlo); cp0u32!(cp0_watchhi); cp0u32!(cp0_ecc); cp0u32!(cp0_cacheerr);
        cp0u32!(cp0_taglo); cp0u32!(cp0_taghi);
        cp0u64!(cp0_badvaddr); cp0u64!(cp0_epc); cp0u64!(cp0_errorepc);
        cp0u64!(cp0_entrylo0); cp0u64!(cp0_entrylo1); cp0u64!(cp0_context);
        cp0u64!(cp0_pagemask); cp0u64!(cp0_entryhi); cp0u64!(cp0_xcontext);
        tbl.insert("cp0".into(), toml::Value::Table(cp0));

        // FPU
        let mut fpu = toml::map::Map::new();
        fpu.insert("fpr".into(), u64_slice_to_toml(&c.fpr));
        fpu.insert("fpu_fir".into(),  hex_u32(c.fpu_fir));
        fpu.insert("fpu_fccr".into(), hex_u32(c.fpu_fccr));
        fpu.insert("fpu_fexr".into(), hex_u32(c.fpu_fexr));
        fpu.insert("fpu_fenr".into(), hex_u32(c.fpu_fenr));
        fpu.insert("fpu_fcsr".into(), hex_u32(c.fpu_fcsr));
        tbl.insert("fpu".into(), toml::Value::Table(fpu));

        // Execution state
        tbl.insert("in_delay_slot".into(),     toml::Value::Boolean(c.in_delay_slot));
        tbl.insert("delay_slot_target".into(), hex_u64(c.delay_slot_target));

        // TLB
        tbl.insert("tlb".into(), exec.tlb.save_state());

        // Cache (L1-I, L1-D, L2 tags + data, LL/SC state)
        tbl.insert("cache".into(), exec.cache.save_cache_state());

        toml::Value::Table(tbl)
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        let mut exec = self.executor.lock();
        let c = &mut exec.core;

        if let Some(arr) = get_field(v, "gpr") { load_u64_slice(arr, &mut c.gpr); }
        if let Some(x) = get_field(v, "pc")  { c.pc = toml_u64(x).unwrap_or(c.pc); }
        if let Some(x) = get_field(v, "hi")  { c.hi = toml_u64(x).unwrap_or(c.hi); }
        if let Some(x) = get_field(v, "lo")  { c.lo = toml_u64(x).unwrap_or(c.lo); }

        if let Some(cp0) = get_field(v, "cp0") {
            macro_rules! ld32 { ($f:ident) => {
                if let Some(x) = get_field(cp0, stringify!($f)) {
                    c.$f = toml_u32(x).unwrap_or(c.$f);
                }
            }}
            macro_rules! ld64 { ($f:ident) => {
                if let Some(x) = get_field(cp0, stringify!($f)) {
                    c.$f = toml_u64(x).unwrap_or(c.$f);
                }
            }}
            ld32!(cp0_index); ld32!(cp0_random); ld32!(cp0_wired);
            ld64!(cp0_count); ld64!(cp0_compare);
            ld64!(count_hz);
            ld64!(compare_delta_slow); ld64!(compare_delta_fast);
            // Compat shim: pre-timer-based snapshots stored cp0_count/
            // cp0_compare and the learned deltas in 32.32 fixed-point
            // (hardware count in the high word). Values above u32::MAX can
            // only be that old format — shift down to plain hardware counts.
            // (Such snapshots carry count_step, not count_hz, so count_hz
            // stays at its default until the guest's tick is re-recognized.)
            if c.cp0_count > u32::MAX as u64 { c.cp0_count >>= 32; }
            if c.cp0_compare > u32::MAX as u64 { c.cp0_compare >>= 32; }
            if c.compare_delta_slow > u32::MAX as u64 { c.compare_delta_slow >>= 32; }
            if c.compare_delta_fast > u32::MAX as u64 { c.compare_delta_fast >>= 32; }
            // Mirror count_hz into its atomic shadow (read by the display
            // thread) so the live UI matches the restored core state.
            c.count_hz_atomic.store(c.count_hz, std::sync::atomic::Ordering::Relaxed);
            ld32!(cp0_status); ld32!(cp0_cause); ld32!(cp0_prid);
            ld32!(cp0_config); ld32!(cp0_lladdr); ld32!(cp0_watchlo); ld32!(cp0_watchhi);
            ld32!(cp0_ecc); ld32!(cp0_cacheerr); ld32!(cp0_taglo); ld32!(cp0_taghi);
            ld64!(cp0_entrylo0); ld64!(cp0_entrylo1); ld64!(cp0_context);
            ld64!(cp0_pagemask); ld64!(cp0_badvaddr); ld64!(cp0_entryhi);
            ld64!(cp0_xcontext); ld64!(cp0_epc); ld64!(cp0_errorepc);
            // Restart the virtual count from the restored value and re-arm
            // the compare timer — Instants from the previous run are
            // meaningless here.
            c.reanchor_count_and_reschedule();
        }

        if let Some(fpu) = get_field(v, "fpu") {
            if let Some(arr) = get_field(fpu, "fpr") { load_u64_slice(arr, &mut c.fpr); }
            macro_rules! ldf { ($f:ident) => {
                if let Some(x) = get_field(fpu, stringify!($f)) {
                    c.$f = toml_u32(x).unwrap_or(c.$f);
                }
            }}
            ldf!(fpu_fir); ldf!(fpu_fccr); ldf!(fpu_fexr); ldf!(fpu_fenr); ldf!(fpu_fcsr);
        }

        if let Some(x) = get_field(v, "in_delay_slot")     { exec.core.in_delay_slot     = toml_bool(x).unwrap_or(false); }
        if let Some(x) = get_field(v, "delay_slot_target") { exec.core.delay_slot_target = toml_u64(x).unwrap_or(0); }

        if let Some(tlb_v) = get_field(v, "tlb") {
            exec.tlb.load_state(tlb_v)?;
        }

        if let Some(cache_v) = get_field(v, "cache") {
            exec.cache.load_cache_state(cache_v)?;
        }

        Ok(())
    }
}

// ── CpuDebug adapter ─────────────────────────────────────────────────────────

use std::sync::atomic::AtomicI32;
use crate::gdb_stub::{CpuDebug, StopReason};
use gdbstub_arch::mips::reg::{MipsCoreRegs, MipsCp0Regs, MipsFpuRegs};
use gdbstub_arch::mips::reg::id::MipsRegId;
use gdbstub::target::ext::breakpoints::WatchKind;

/// Tracks the stop reason from the last `run_blocking` call.
/// Stored in an Arc<Mutex<>> so the spawned continue thread can write it.
struct StopState {
    reason: parking_lot::Mutex<StopReason>,
}

impl StopState {
    fn new() -> Arc<Self> {
        Arc::new(Self { reason: parking_lot::Mutex::new(StopReason::Interrupted) })
    }
    fn set(&self, r: StopReason) { *self.reason.lock() = r; }
    fn get(&self) -> StopReason { *self.reason.lock() }
}

/// Wraps `Arc<MipsCpu<T,C>>` to implement `CpuDebug`.
pub struct MipsCpuDebugAdapter<T: Tlb + Send + 'static, C: MipsCache + Send + 'static> {
    cpu: Arc<MipsCpu<T, C>>,
    stop_state: Arc<StopState>,
    // Allocator for GDB-owned breakpoint IDs (starts at 10000).
    next_gdb_bp_id: parking_lot::Mutex<usize>,
}

impl<T: Tlb + Send + 'static, C: MipsCache + Send + 'static> MipsCpuDebugAdapter<T, C> {
    pub fn new(cpu: Arc<MipsCpu<T, C>>) -> Arc<Self> {
        Arc::new(Self {
            cpu,
            stop_state: StopState::new(),
            next_gdb_bp_id: parking_lot::Mutex::new(10000),
        })
    }
}

impl<T: Tlb + Send + 'static, C: MipsCache + Send + 'static> CpuDebug
    for MipsCpuDebugAdapter<T, C>
{
    fn stop(&self) {
        self.cpu.stop();
    }

    fn start(&self) {
        self.cpu.start();
    }

    fn is_running(&self) -> bool {
        self.cpu.is_running()
    }

    fn step_one(&self) -> StopReason {
        use crate::mips_exec::EXEC_BREAKPOINT;
        self.cpu.stop(); // ensure no thread is running
        let mut exec = self.cpu.executor.lock();
        exec.last_bp_hit = None;

        // Execute one instruction. If it's a branch, also execute the delay
        // slot, so a single GDB-visible step lands past it rather than
        // stopping mid-delay-slot. `core.in_delay_slot` (set by
        // branch_delay, cleared by handle_exec_complete) is the real signal
        // that a branch/jump was just taken and its delay slot hasn't run
        // yet — status codes carry no PC-related information of their own
        // (every handler sets core.pc itself before returning).
        //eprintln!("GDB: step_one: PC={:#018x}", exec.core.pc);
        let status = exec.step();
        //eprintln!("GDB: step_one: after step status={:#010x} PC={:#018x}", status, exec.core.pc);
        let reason = if status == EXEC_BREAKPOINT {
            drop(exec);
            return StopReason::SwBreakpoint;
        } else if exec.core.in_delay_slot {
            // Branch taken — execute delay slot instruction too.
            let ds_status = exec.step();
            if ds_status == EXEC_BREAKPOINT {
                StopReason::SwBreakpoint
            } else {
                StopReason::DoneStep
            }
        } else {
            StopReason::DoneStep
        };
        drop(exec);
        self.stop_state.set(reason);
        reason
    }

    fn run_blocking(&self, count: Option<usize>) -> StopReason {
        // run_debug_loop with wait=true blocks until the CPU stops.
        // We discard the text output (sink writer).
        //eprintln!("GDB: run_blocking: calling run_debug_loop");
        let sink = Box::new(std::io::sink());
        self.cpu.run_debug_loop(count, true, sink);
        //eprintln!("GDB: run_blocking: run_debug_loop returned, is_running={}", self.cpu.is_running());

        // Inspect executor state to determine stop reason.
        let reason = if let Some(exec) = self.cpu.executor.try_lock() {
            if let Some(bp_id) = exec.last_bp_hit {
                let bp = exec.breakpoints.iter().find(|b| b.id == bp_id);
                match bp.map(|b| b.kind) {
                    Some(BpType::Pc) | Some(BpType::VirtFetch) | Some(BpType::PhysFetch) => {
                        StopReason::SwBreakpoint
                    }
                    Some(BpType::VirtRead) | Some(BpType::PhysRead) => {
                        StopReason::Watchpoint { addr: exec.core.pc, kind: WatchKind::Read }
                    }
                    Some(BpType::VirtWrite) | Some(BpType::PhysWrite) => {
                        StopReason::Watchpoint { addr: exec.core.pc, kind: WatchKind::Write }
                    }
                    _ => StopReason::SwBreakpoint,
                }
            } else {
                StopReason::DoneStep
            }
        } else {
            StopReason::DoneStep
        };

        self.stop_state.set(reason);
        reason
    }

    fn read_regs(&self) -> MipsCoreRegs<u64> {
        let exec = self.cpu.executor.lock();
        let core = &exec.core;
        let mut r = [0u64; 32];
        for i in 0..32 { r[i] = core.read_gpr(i as u32); }
        MipsCoreRegs {
            r,
            lo: core.lo,
            hi: core.hi,
            pc: core.pc,
            cp0: MipsCp0Regs {
                status: core.cp0_status as u64,
                badvaddr: core.cp0_badvaddr,
                cause: core.cp0_cause as u64,
            },
            fpu: MipsFpuRegs {
                r: core.fpr,
                fcsr: core.fpu_fcsr as u64,
                fir: core.fpu_fir as u64,
            },
        }
    }

    fn write_regs(&self, regs: &MipsCoreRegs<u64>) {
        let mut exec = self.cpu.executor.lock();
        let core = &mut exec.core;
        for i in 1..32 { core.gpr[i] = regs.r[i]; } // r[0] always 0
        core.lo = regs.lo;
        core.hi = regs.hi;
        core.pc = regs.pc;
        core.cp0_status = regs.cp0.status as u32;
        core.cp0_badvaddr = regs.cp0.badvaddr;
        core.cp0_cause = regs.cp0.cause as u32;
        core.fpr = regs.fpu.r;
        core.fpu_fcsr = regs.fpu.fcsr as u32;
        core.fpu_fir  = regs.fpu.fir as u32;
    }

    fn read_reg(&self, id: MipsRegId<u64>) -> Option<u64> {
        let exec = self.cpu.executor.lock();
        let core = &exec.core;
        Some(match id {
            MipsRegId::Gpr(i) if i < 32 => core.read_gpr(i as u32),
            MipsRegId::Lo => core.lo,
            MipsRegId::Hi => core.hi,
            MipsRegId::Pc => core.pc,
            MipsRegId::Status => core.cp0_status as u64,
            MipsRegId::Badvaddr => core.cp0_badvaddr,
            MipsRegId::Cause => core.cp0_cause as u64,
            MipsRegId::Fpr(i) if i < 32 => core.fpr[i as usize],
            MipsRegId::Fcsr => core.fpu_fcsr as u64,
            MipsRegId::Fir => core.fpu_fir as u64,
            _ => return None,
        })
    }

    fn write_reg(&self, id: MipsRegId<u64>, val: u64) {
        let mut exec = self.cpu.executor.lock();
        let core = &mut exec.core;
        match id {
            MipsRegId::Gpr(i) if i > 0 && i < 32 => { core.gpr[i as usize] = val; }
            MipsRegId::Lo => { core.lo = val; }
            MipsRegId::Hi => { core.hi = val; }
            MipsRegId::Pc => { core.pc = val; }
            MipsRegId::Status => { core.cp0_status = val as u32; }
            MipsRegId::Cause => { core.cp0_cause = val as u32; }
            MipsRegId::Fpr(i) if i < 32 => { core.fpr[i as usize] = val; }
            MipsRegId::Fcsr => { core.fpu_fcsr = val as u32; }
            _ => {}
        }
    }

    fn read_mem(&self, addr: u64, buf: &mut [u8]) -> Result<(), ()> {
        if buf.is_empty() { return Ok(()); }
        let mut exec = self.cpu.executor.lock();

        let mut i = 0usize;
        while i < buf.len() {
            // Read a word at the aligned address, then extract the needed bytes.
            let cur_addr = addr.wrapping_add(i as u64);
            let aligned = cur_addr & !3u64;
            let offset = (cur_addr & 3) as usize;
            let remaining = buf.len() - i;
            let avail = 4 - offset;
            let to_copy = remaining.min(avail);

            match exec.debug_read(aligned, 4) {
                Ok(word) => {
                    let word_bytes = (word as u32).to_be_bytes(); // MIPS is big-endian
                    for j in 0..to_copy {
                        buf[i + j] = word_bytes[offset + j];
                    }
                }
                Err(_) => {
                    // Fill with 0 on error rather than aborting the entire read.
                    for j in 0..to_copy { buf[i + j] = 0; }
                }
            }
            i += to_copy;
        }
        Ok(())
    }

    fn write_mem(&self, addr: u64, data: &[u8]) -> Result<(), ()> {
        if data.is_empty() { return Ok(()); }
        let mut exec = self.cpu.executor.lock();

        let mut i = 0usize;
        while i < data.len() {
            let cur_addr = addr.wrapping_add(i as u64);
            let aligned = cur_addr & !3u64;
            let offset = (cur_addr & 3) as usize;
            let remaining = data.len() - i;
            let avail = 4 - offset;
            let to_write = remaining.min(avail);

            if to_write == 4 {
                // Aligned 4-byte write.
                let val = u32::from_be_bytes([data[i], data[i+1], data[i+2], data[i+3]]);
                exec.debug_write(aligned, val as u64, 4, u64::MAX);
            } else {
                // Partial word: read-modify-write.
                let old = match exec.debug_read(aligned, 4) {
                    Ok(v) => v as u32,
                    Err(_) => 0,
                };
                let mut word_bytes = old.to_be_bytes();
                for j in 0..to_write { word_bytes[offset + j] = data[i + j]; }
                let new_val = u32::from_be_bytes(word_bytes);
                exec.debug_write(aligned, new_val as u64, 4, u64::MAX);
            }
            i += to_write;
        }
        Ok(())
    }

    fn add_bp(&self, addr: u64, kind: BpType) -> usize {
        let id = {
            let mut next = self.next_gdb_bp_id.lock();
            let id = *next;
            *next += 1;
            id
        };
        let mut exec = self.cpu.executor.lock();
        exec.add_breakpoint(id, addr, kind);
        //eprintln!("GDB: add_bp id={} addr={:#018x} kind={:?} pc_bp_count={}", id, addr, kind, exec.pc_bp_count);
        id
    }

    fn remove_bp(&self, id: usize) {
        self.cpu.executor.lock().remove_breakpoint(id);
    }

    fn last_stop_reason(&self) -> StopReason {
        self.stop_state.get()
    }
}

#[cfg(test)]
mod round_to_int_mode_tests {
    use super::*;

    #[test]
    fn f64_nearest_even_ties_round_to_even() {
        assert_eq!(round_f64_to_int_mode(0.5, RM_NEAREST_EVEN), 0.0);
        assert_eq!(round_f64_to_int_mode(1.5, RM_NEAREST_EVEN), 2.0);
        assert_eq!(round_f64_to_int_mode(2.5, RM_NEAREST_EVEN), 2.0);
        assert_eq!(round_f64_to_int_mode(-0.5, RM_NEAREST_EVEN), -0.0);
        assert_eq!(round_f64_to_int_mode(-1.5, RM_NEAREST_EVEN), -2.0);
        assert_eq!(round_f64_to_int_mode(65535.5, RM_NEAREST_EVEN), 65536.0, "tie between odd 65535 and even 65536 rounds to even");
    }

    #[test]
    fn f64_nearest_even_non_ties_round_to_nearest() {
        assert_eq!(round_f64_to_int_mode(1.4, RM_NEAREST_EVEN), 1.0);
        assert_eq!(round_f64_to_int_mode(1.6, RM_NEAREST_EVEN), 2.0);
        assert_eq!(round_f64_to_int_mode(-1.6, RM_NEAREST_EVEN), -2.0);
    }

    #[test]
    fn f64_toward_zero_truncates() {
        assert_eq!(round_f64_to_int_mode(65535.5, RM_TOWARD_ZERO), 65535.0, "this is the live-boot regression case: RZ must truncate toward zero, not round");
        assert_eq!(round_f64_to_int_mode(1.9, RM_TOWARD_ZERO), 1.0);
        assert_eq!(round_f64_to_int_mode(-1.9, RM_TOWARD_ZERO), -1.0);
        assert_eq!(round_f64_to_int_mode(0.5, RM_TOWARD_ZERO), 0.0);
        assert_eq!(round_f64_to_int_mode(-0.5, RM_TOWARD_ZERO), -0.0);
    }

    #[test]
    fn f64_toward_pos_inf() {
        assert_eq!(round_f64_to_int_mode(1.1, RM_TOWARD_POS_INF), 2.0);
        assert_eq!(round_f64_to_int_mode(-1.1, RM_TOWARD_POS_INF), -1.0);
        assert_eq!(round_f64_to_int_mode(0.1, RM_TOWARD_POS_INF), 1.0);
        assert_eq!(round_f64_to_int_mode(-0.1, RM_TOWARD_POS_INF), -0.0);
    }

    #[test]
    fn f64_toward_neg_inf() {
        assert_eq!(round_f64_to_int_mode(1.1, RM_TOWARD_NEG_INF), 1.0);
        assert_eq!(round_f64_to_int_mode(-1.1, RM_TOWARD_NEG_INF), -2.0);
        assert_eq!(round_f64_to_int_mode(0.1, RM_TOWARD_NEG_INF), 0.0);
        assert_eq!(round_f64_to_int_mode(-0.1, RM_TOWARD_NEG_INF), -1.0);
    }

    #[test]
    fn f64_already_integer_unchanged() {
        for rm in [RM_NEAREST_EVEN, RM_TOWARD_ZERO, RM_TOWARD_POS_INF, RM_TOWARD_NEG_INF] {
            assert_eq!(round_f64_to_int_mode(79.0, rm), 79.0);
            assert_eq!(round_f64_to_int_mode(-79.0, rm), -79.0);
            assert_eq!(round_f64_to_int_mode(0.0, rm), 0.0);
        }
    }

    #[test]
    fn f64_large_magnitude_already_integer_valued() {
        // 2^60 has no fractional bits representable at all in f64 (exponent
        // >= mantissa width) -- must pass through unchanged for every mode.
        let big = (1u64 << 60) as f64;
        for rm in [RM_NEAREST_EVEN, RM_TOWARD_ZERO, RM_TOWARD_POS_INF, RM_TOWARD_NEG_INF] {
            assert_eq!(round_f64_to_int_mode(big, rm), big);
            assert_eq!(round_f64_to_int_mode(-big, rm), -big);
        }
    }

    #[test]
    fn f64_nan_and_infinity_pass_through() {
        for rm in [RM_NEAREST_EVEN, RM_TOWARD_ZERO, RM_TOWARD_POS_INF, RM_TOWARD_NEG_INF] {
            assert!(round_f64_to_int_mode(f64::NAN, rm).is_nan());
            assert_eq!(round_f64_to_int_mode(f64::INFINITY, rm), f64::INFINITY);
            assert_eq!(round_f64_to_int_mode(f64::NEG_INFINITY, rm), f64::NEG_INFINITY);
        }
    }

    #[test]
    fn f64_matches_libm_round_trunc_ceil_floor_on_ordinary_values() {
        let cases = [3.7, -3.7, 3.2, -3.2, 100.5, -100.5, 0.999999, -0.999999, 1e10 + 0.5, -1e10 - 0.5];
        for x in cases {
            assert_eq!(round_f64_to_int_mode(x, RM_TOWARD_ZERO), x.trunc(), "trunc mismatch for {x}");
            assert_eq!(round_f64_to_int_mode(x, RM_TOWARD_POS_INF), x.ceil(), "ceil mismatch for {x}");
            assert_eq!(round_f64_to_int_mode(x, RM_TOWARD_NEG_INF), x.floor(), "floor mismatch for {x}");
        }
    }

    /// Rounding up must correctly carry a mantissa overflow into the
    /// exponent field (e.g. 1.9999999999999998's mantissa is all-ones —
    /// rounding away from zero must produce exactly 2.0, not a garbage bit
    /// pattern from an unhandled carry). This is exactly the scenario the
    /// integer-add-on-raw-bits implementation (as opposed to `truncated +
    /// 2f64.powi(exp)`, an earlier, buggier attempt) is designed to get
    /// right for free via ordinary binary carry propagation.
    #[test]
    fn f64_rounding_up_carries_mantissa_overflow_into_exponent() {
        let x = f64::from_bits(0x3FFFFFFFFFFFFFFF); // largest f64 < 2.0
        assert_eq!(round_f64_to_int_mode(x, RM_NEAREST_EVEN), 2.0);
        assert_eq!(round_f64_to_int_mode(x, RM_TOWARD_POS_INF), 2.0);
        assert_eq!(round_f64_to_int_mode(-x, RM_TOWARD_NEG_INF), -2.0);
    }

    #[test]
    fn f32_nearest_even_ties_round_to_even() {
        assert_eq!(round_f32_to_int_mode(0.5, RM_NEAREST_EVEN), 0.0);
        assert_eq!(round_f32_to_int_mode(1.5, RM_NEAREST_EVEN), 2.0);
        assert_eq!(round_f32_to_int_mode(2.5, RM_NEAREST_EVEN), 2.0);
        assert_eq!(round_f32_to_int_mode(-2.5, RM_NEAREST_EVEN), -2.0);
    }

    #[test]
    fn f32_toward_zero_truncates() {
        assert_eq!(round_f32_to_int_mode(3.7, RM_TOWARD_ZERO), 3.0);
        assert_eq!(round_f32_to_int_mode(-3.7, RM_TOWARD_ZERO), -3.0);
    }

    #[test]
    fn f32_matches_libm_on_ordinary_values() {
        let cases = [3.7f32, -3.7, 3.2, -3.2, 100.5, -100.5, 65535.5, -65535.5];
        for x in cases {
            assert_eq!(round_f32_to_int_mode(x, RM_TOWARD_ZERO), x.trunc(), "trunc mismatch for {x}");
            assert_eq!(round_f32_to_int_mode(x, RM_TOWARD_POS_INF), x.ceil(), "ceil mismatch for {x}");
            assert_eq!(round_f32_to_int_mode(x, RM_TOWARD_NEG_INF), x.floor(), "floor mismatch for {x}");
        }
    }

    #[test]
    fn f32_nan_and_infinity_pass_through() {
        for rm in [RM_NEAREST_EVEN, RM_TOWARD_ZERO, RM_TOWARD_POS_INF, RM_TOWARD_NEG_INF] {
            assert!(round_f32_to_int_mode(f32::NAN, rm).is_nan());
            assert_eq!(round_f32_to_int_mode(f32::INFINITY, rm), f32::INFINITY);
            assert_eq!(round_f32_to_int_mode(f32::NEG_INFINITY, rm), f32::NEG_INFINITY);
        }
    }
}
