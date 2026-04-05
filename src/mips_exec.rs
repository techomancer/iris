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

#[inline(always)]
fn mips_log(bit: u32) -> bool {
    devlog_is_active(LogModule::Mips) && (devlog_mask(LogModule::Mips) & bit) != 0
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
pub const CONFIG_SB: u32 = 22;    // 2 bits secondary cache size, 0 - 4 words, 1 - 8 ...
pub const CONFIG_SS: u32 = 21;    // split secondary cache mode
pub const CONFIG_SW: u32 = 20;    // secondary cache port width 0 - 128bit, 1 - 64bit
pub const CONFIG_EW: u32 = 18;    // 2 bits system port width 0 - 64 bit, 1 - 32 bit
pub const CONFIG_SC: u32 = 17;    // secondary cache present 0 - present, 1 - absent
pub const CONFIG_SM: u32 = 16;    // dirty shared coherency state 0 - enabled, 1 - disabled
pub const CONFIG_BE: u32 = 15;    // 1 - big endian, 0 - little endian
pub const CONFIG_EM: u32 = 14;    // 1 - ecc enabled, 0 - parity enabled
pub const CONFIG_EB: u32 = 13;    // 1 block ordering 1 - sequential 0 - sub block
pub const CONFIG_IC: u32 = 9;     // 3 bits ICache size 2^12+IC
pub const CONFIG_DC: u32 = 6;     // 3 bits DCache size 2^12+IC
pub const CONFIG_IB: u32 = 5;     // icache block size 0=16B 1=32B (R4000/R4400=0, R5000=1)
pub const CONFIG_DB: u32 = 4;     // dcache block size 0=16B 1=32B (R4000/R4400=0, R5000=1)
pub const CONFIG_CU: u32 = 3;     // 0 store conditional uses coherency algo from tlb, 1 - scs uses cacheable coherent update on write
pub const CONFIG_K0: u32 = 0;     // kseg0 coherency algorithm

/// Undo buffer size: 1M (2^20) instructions
#[cfg(feature = "developer")]
const UNDO_BUFFER_SIZE: usize = 1 << 20;

/// Memory write operation for undo tracking
#[cfg(feature = "developer")]
#[derive(Debug, Clone, Copy)]
struct MemoryWrite {
    virt_addr: u64,
    phys_addr: u64,
    old_value: u64,
    size: usize,
}

/// CPU state snapshot for undo - includes core state and metadata
#[cfg(feature = "developer")]
#[derive(Clone)]
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
        if !self.enabled {
            return;
        }

        self.snapshots[self.head] = Some(snapshot);
        self.head = (self.head + 1) % UNDO_BUFFER_SIZE;
        if self.count < UNDO_BUFFER_SIZE {
            self.count += 1;
        }
    }

    fn can_undo(&self, steps: usize) -> bool {
        self.enabled && steps <= self.count
    }

    fn get(&self, steps_back: usize) -> Option<&CpuSnapshot> {
        if steps_back == 0 || steps_back > self.count {
            return None;
        }

        let index = if self.head >= steps_back {
            self.head - steps_back
        } else {
            UNDO_BUFFER_SIZE - (steps_back - self.head)
        };

        self.snapshots[index].as_ref()
    }

    fn clear(&mut self) {
        self.head = 0;
        self.count = 0;
        for snapshot in &mut self.snapshots {
            *snapshot = None;
        }
    }
}

// External interrupt mask (IP6..IP2)
const EXT_INT_MASK: u32 = crate::mips_core::CAUSE_IP6 |
                          crate::mips_core::CAUSE_IP5 |
                          crate::mips_core::CAUSE_IP4 |
                          crate::mips_core::CAUSE_IP3 |
                          crate::mips_core::CAUSE_IP2;

// Bit 63 of the interrupts word = soft-reset request
const SOFT_RESET_BIT: u64 = 1u64 << 63;

const TRACEBACK_SIZE: usize = 1048576; // 1M entries

#[derive(Clone, Copy, Debug, Default)]
struct TracebackEntry {
    pc: u64,
    instr: u32,
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

    fn push(&mut self, pc: u64, instr: u32) {
        self.entries[self.head] = TracebackEntry { pc, instr };
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
    pub imm:     u32,
    pub raw:     u32,
    pub decoded: bool,              // true when all fields below are valid for this raw
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
            decoded: false,
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

// Non-exception status tags in bits [15:8]
pub const EXEC_COMPLETE:           ExecStatus = 0x0000_0000; // normal completion (advance PC by 4)
pub const EXEC_COMPLETE_NO_INC:    ExecStatus = 0x0000_0080; // completion, PC already set (no increment)
pub const EXEC_RETRY:              ExecStatus = 0x0000_0100; // bus busy, retry same instr
pub const EXEC_BRANCH_DELAY:       ExecStatus = 0x0000_0200; // branch taken; target in delay_slot_target
pub const EXEC_BRANCH_LIKELY_SKIP: ExecStatus = 0x0000_0400; // branch likely not taken, skip delay slot
pub const EXEC_BREAKPOINT:         ExecStatus = 0x0000_0800; // breakpoint hit

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
    pub ic_size: usize,     // L1 instruction cache size in bytes
    pub ic_line: usize,     // L1 instruction cache line size in bytes
    pub dc_size: usize,     // L1 data cache size in bytes
    pub dc_line: usize,     // L1 data cache line size in bytes
    pub l2_size: usize,     // L2 cache size in bytes
    pub l2_line: usize,     // L2 cache line size in bytes
}

impl MipsCpuConfig {
    /// Default configuration matching SGI Indy (R4600): 48-entry TLB,
    /// 16KB/32B L1 I/D caches, 1MB/128B unified L2.
    pub const fn indy() -> Self {
        Self {
            tlb_entries: 48,
            ic_size:     16 * 1024,
            ic_line:     16,
            dc_size:     16 * 1024,
            dc_line:     16,
            l2_size:     1024 * 1024,
            l2_line:     128,
        }
    }
}

/// MIPS Execution Engine - combines CPU core with memory interface and TLB
pub struct MipsExecutor<T: Tlb, C: MipsCache> {
    pub core: MipsCore,
    pub sysad: Arc<dyn BusDevice>,
    pub tlb: T,
    pub cache: C,
    in_delay_slot: bool,
    pub delay_slot_target: u64,
    #[cfg(feature = "developer")]
    undo_buffer: UndoBuffer,
    #[cfg(feature = "developer")]
    pending_memory_writes: Vec<MemoryWrite>,
    traceback: TracebackBuffer,
    pub symbols: Arc<Mutex<SymbolTable>>,
    pub breakpoints: Vec<Breakpoint>,
    pub next_bp_id: usize,
    pub last_bp_hit: Option<usize>,
    pub pc_bp_count: usize,
    pub mem_bp_count: usize,
    /// When true, the next call to step() skips all breakpoint checks.
    /// Cleared automatically after one step. Used to resume past a breakpoint.
    pub skip_breakpoints: bool,
    /// Current decoded instruction — written by fetch_instr, read by exec_decoded.
    pub ins: DecodedInstr,
    /// Count of instructions that were already decoded (cache hit).
    pub decoded_count: Arc<AtomicU64>,
    /// Count of instructions fetched from uncached address space.
    pub uncached_fetch_count: Arc<AtomicU64>,
    /// Raw pointers into core.cycles/interrupts/fasttick_count — avoids Arc::deref on every step.
    /// Safety: these point into Arcs owned by MipsCore which outlive the executor.
    cycles_ptr:       *const AtomicU64,
    interrupts_ptr:   *const AtomicU64,
    fasttick_ptr:     *const AtomicU64,
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
    /// Local cycle counter — flushed to the shared atomic periodically to avoid
    /// a locked bus op on every instruction.
    local_cycles: u64,
    /// Cached external interrupt word — reloaded every 16 instructions.
    cached_pending: u64,
}

// ---- translate_fn wrappers (one per privilege × addressing-mode combination) ---------------
// These are free functions so they can be stored as bare fn pointers in MipsExecutor.
// Each is a thin shim that delegates to the fully monomorphised translate_32/64bit_impl.

// ---- nano-TLB probe/fill helper -----------------------------------------------
// Shared by all 6 translate_fn wrappers. The slot index is `at as usize` (0/1/2).
// On hit returns the cached TranslateResult directly.
// On miss calls `slow` (the full translate), then fills the slot on success.
#[inline(always)]
fn nanotlb_translate<T: Tlb, C: MipsCache>(
    e: &mut MipsExecutor<T, C>,
    va: u64,
    at: AccessType,
    slow: fn(&mut MipsExecutor<T, C>, u64, AccessType) -> TranslateResult,
) -> TranslateResult {
    let idx = at as usize; // Fetch=0, Read=1, Write=2
    let slot = &e.core.nanotlb[idx];
    if slot.matches(va) {
        return TranslateResult::ok(slot.phys_addr(va), slot.cache_attr_raw());
    }
    let result = slow(e, va, at);
    if !result.is_exception() {
        e.core.nanotlb[idx].fill_raw(va, result.phys as u64, result.status & 0x7);
    }
    result
}

fn translate_32_kernel<T: Tlb, C: MipsCache>(e: &mut MipsExecutor<T,C>, va: u64, at: AccessType) -> TranslateResult {
    nanotlb_translate(e, va, at, |e, va, at| e.translate_32bit_impl::<false, {crate::mips_core::PRIV_KERNEL}>(va, at))
}
fn translate_32_supervisor<T: Tlb, C: MipsCache>(e: &mut MipsExecutor<T,C>, va: u64, at: AccessType) -> TranslateResult {
    nanotlb_translate(e, va, at, |e, va, at| e.translate_32bit_impl::<false, {crate::mips_core::PRIV_SUPERVISOR}>(va, at))
}
fn translate_32_user<T: Tlb, C: MipsCache>(e: &mut MipsExecutor<T,C>, va: u64, at: AccessType) -> TranslateResult {
    nanotlb_translate(e, va, at, |e, va, at| e.translate_32bit_impl::<false, {crate::mips_core::PRIV_USER}>(va, at))
}
fn translate_64_kernel<T: Tlb, C: MipsCache>(e: &mut MipsExecutor<T,C>, va: u64, at: AccessType) -> TranslateResult {
    nanotlb_translate(e, va, at, |e, va, at| e.translate_64bit_impl::<false, {crate::mips_core::PRIV_KERNEL}>(va, at))
}
fn translate_64_supervisor<T: Tlb, C: MipsCache>(e: &mut MipsExecutor<T,C>, va: u64, at: AccessType) -> TranslateResult {
    nanotlb_translate(e, va, at, |e, va, at| e.translate_64bit_impl::<false, {crate::mips_core::PRIV_SUPERVISOR}>(va, at))
}
fn translate_64_user<T: Tlb, C: MipsCache>(e: &mut MipsExecutor<T,C>, va: u64, at: AccessType) -> TranslateResult {
    nanotlb_translate(e, va, at, |e, va, at| e.translate_64bit_impl::<false, {crate::mips_core::PRIV_USER}>(va, at))
}

/// Free-standing trampoline for the CP0 Status callback installed by `install_status_cb`.
/// Rust does not allow `Self` inside a nested fn, so the generic trampoline lives here.
// Safety: the raw pointers (cycles_ptr, interrupts_ptr, fasttick_ptr) point into Arc allocations
// owned by MipsCore which outlive the executor. The executor is only accessed from the CPU thread.
unsafe impl<T: Tlb, C: MipsCache> Send for MipsExecutor<T, C> {}
unsafe impl<T: Tlb, C: MipsCache> Sync for MipsExecutor<T, C> {}

fn mips_executor_status_cb<T: Tlb, C: MipsCache>(ctx: *mut core::ffi::c_void, old: u32, new: u32) {
    // SAFETY: ctx is `&mut MipsExecutor<T,C>` cast to void, alive for the executor's lifetime,
    // and only ever called from the CPU thread that exclusively owns the executor.
    let exec = unsafe { &mut *(ctx as *mut MipsExecutor<T, C>) };
    exec.on_cp0_status_changed(old, new);
}

impl<T: Tlb, C: MipsCache> MipsExecutor<T, C> {
    /// Create a new executor from a config and a bus (sysad) and a TLB.
    /// The cache hierarchy is constructed internally as a unified R4000Cache.
    pub fn new(sysad: Arc<dyn BusDevice>, tlb: T, cfg: &MipsCpuConfig) -> Self
    where
        C: From<(Arc<dyn BusDevice>, R4000CacheConfig)>
    {
        let mut core = MipsCore::new();

        let MipsCpuConfig { ic_size, ic_line, dc_size, dc_line, l2_size, l2_line, .. } = *cfg;

        // Build unified cache hierarchy
        let cache_config = R4000CacheConfig {
            ic_size,
            ic_line_size: ic_line,
            dc_size,
            dc_line_size: dc_line,
            l2_size,
            l2_line_size: l2_line,
        };
        let cache = C::from((sysad.clone(), cache_config));
        
        let mut config = 0;

        // K0 (bits 2:0): kseg0 coherency algorithm. 3 = Cacheable, non-coherent.
        config |= 3 << CONFIG_K0;

        // DB (bit 4): Primary D-cache line size. 0=16B, 1=32B.
        let db_val = if dc_line >= 32 { 1 } else { 0 };
        config |= db_val << CONFIG_DB;

        // IB (bit 5): Primary I-cache line size. 0=16B, 1=32B.
        let ib_val = if ic_line >= 32 { 1 } else { 0 };
        config |= ib_val << CONFIG_IB;

        // DC (bits 8:6): Primary D-cache size. size = 2^(12+DC)
        let dc_val = if dc_size > 0 { (dc_size.trailing_zeros()).saturating_sub(12) } else { 0 };
        config |= dc_val << CONFIG_DC;

        // IC (bits 11:9): Primary I-cache size. size = 2^(12+IC)
        let ic_val = if ic_size > 0 { (ic_size.trailing_zeros()).saturating_sub(12) } else { 0 };
        config |= ic_val << CONFIG_IC;

        // BE (bit 15): Big Endian. 1 for Indy.
        config |= 1 << CONFIG_BE;

        // SC (bit 17): Secondary cache present. 0=present, 1=absent.
        let sc_val = if l2_size > 0 { 0 } else { 1 };
        config |= sc_val << CONFIG_SC;

        // SB (bits 23:22): Secondary cache block size.
        // 00=4 words (16B), 01=8 words (32B), 10=16 words (64B), 11=32 words (128B).
        let sb_val = match l2_line {
            16 => 0b00,
            32 => 0b01,
            64 => 0b10,
            128 => 0b11,
            _ => 0b11, // Default to 128B for our 1MB cache
        };
        config |= sb_val << CONFIG_SB;
/*
For R4000SC/MC CPUs:

  The size is determined algorithmically by the size_2nd_cache function.
   1. Presence Check: It first reads CP0_CONFIG and checks the CONFIG_SC bit (bit 17). If this bit is 0, an L2 cache is present, and the test proceeds.
   2. Sizing Algorithm:
       * It writes data to memory addresses that are powers of two (128KB, 256KB, 512KB, etc.) to fill the cache.
       * It then invalidates the cache tag at index 0, creating a unique "marker."
       * It begins checking addresses again, starting from the smallest possible cache size, and uses a cache instruction to read the tag at each power-of-two boundary.
       * When it reads a tag and finds the "marker" it wrote, it knows the cache has just "wrapped around." The address at which this wrap-around occurred indicates the total size of the
         cache.
   3. Set Variable: The final calculated size is then stored in the _sidcache_size global variable.

*/
/*
 L2 size on R5K

 The size is read directly from a bitfield in CP0_CONFIG.
   1. The code reads CP0_CONFIG and masks for the CONFIG_TR_SS bits (bits 21-20).
   2. The 2-bit value (00, 01, 10, or 11) is extracted.
   3. This value is used as a multiplier for a base size (512KB) to calculate the total L2 cache size (512KB, 1MB, 2MB, or 4MB).

*/        
        core.cp0_config = config;
        core.tlb_entries = cfg.tlb_entries as u32;

        /*eprintln!("Cache config: L1I {}KB/{}B-line  L1D {}KB/{}B-line  L2 {}KB/{}B-line  CP0.Config={:#010x}",
            ic_size / 1024, ic_line,
            dc_size / 1024, dc_line,
            l2_size / 1024, l2_line,
            config);*/


        let mut executor = Self {
            core,
            sysad,
            tlb,
            cache,
            in_delay_slot: false,
            delay_slot_target: 0,
            #[cfg(feature = "developer")]
            undo_buffer: UndoBuffer::new(),
            #[cfg(feature = "developer")]
            pending_memory_writes: Vec::new(),
            traceback: TracebackBuffer::new(),
            symbols: Arc::new(Mutex::new(SymbolTable::new())),
            breakpoints: vec![Breakpoint {
                id: 0, addr: 0, kind: BpType::Pc, enabled: false, condition: None
            }],
            next_bp_id: 1,
            last_bp_hit: None,
            pc_bp_count: 0,
            mem_bp_count: 0,
            skip_breakpoints: false,
            ins: DecodedInstr::default(), // scratch slot for uncached fetches
            decoded_count: Arc::new(AtomicU64::new(0)),
            uncached_fetch_count: Arc::new(AtomicU64::new(0)),
            cycles_ptr:     std::ptr::null(),
            interrupts_ptr: std::ptr::null(),
            fasttick_ptr:   std::ptr::null(),
            // Placeholder — overwritten immediately by update_translate_fn below.
            translate_fn: translate_32_kernel::<T, C>,
            // Placeholder — overwritten immediately by update_fpr_mode below.
            fpr_read_d:  crate::mips_core::read_fpr_d_fr0,
            fpr_write_d: crate::mips_core::write_fpr_d_fr0,
            fpr_read_l:  crate::mips_core::read_fpr_l_fr0,
            fpr_write_l: crate::mips_core::write_fpr_l_fr0,
            fpr_read_w:  crate::mips_core::read_fpr_w_fr0,
            fpr_write_w: crate::mips_core::write_fpr_w_fr0,
            local_cycles: 0,
            cached_pending: 0,
        };

        executor.rebind_atomic_ptrs();
        executor.update_translate_fn();
        executor.update_fpr_mode();
        executor
    }

    /// Re-sync raw atomic pointers after the shared Arcs are injected post-construction.
    /// Must be called whenever core.cycles, core.interrupts, or core.fasttick_count are replaced.
    pub fn rebind_atomic_ptrs(&mut self) {
        self.cycles_ptr     = Arc::as_ptr(&self.core.cycles);
        self.interrupts_ptr = Arc::as_ptr(&self.core.interrupts);
        self.fasttick_ptr   = Arc::as_ptr(&self.core.fasttick_count);
    }

    /// Install the CP0 Status change callback pointing at this executor.
    /// Call once after construction. The callback is invoked (from write_cp0) with
    /// (old_status, new_status) whenever CP0 register 12 is written.
    pub fn install_status_cb(&mut self) {
        let ctx = self as *mut Self as *mut core::ffi::c_void;
        self.core.status_changed_cb = Some((mips_executor_status_cb::<T, C>, ctx));
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
        self.core.nanotlb_invalidate();
    }

    /// Execute a single instruction (decode into scratch, then execute).
    pub fn exec(&mut self, instr: u32) -> ExecStatus {
        self.ins.raw = instr;
        self.ins.decoded = false;
        if decode_into::<T, C>(&mut self.ins) {
            #[cfg(feature = "developer")]
            self.decoded_count.fetch_add(1, Ordering::Relaxed);
        }
        let d: *const DecodedInstr = &self.ins;
        self.exec_decoded(unsafe { &*d })
    }

    /// Returns true if breakpoints should fire. False when skip_breakpoints is set.
    #[inline(always)]
    fn bp_enabled(&self) -> bool {
        !self.skip_breakpoints
    }

    /// Store branch target and return EXEC_BRANCH_DELAY.
    #[inline(always)]
    fn branch_delay(&mut self, target: u64) -> ExecStatus {
        self.delay_slot_target = target;
        EXEC_BRANCH_DELAY
    }

    /// Step one instruction (fetch and execute).
    /// If `self.skip_breakpoints` is set, all breakpoint checks for this one
    /// step (PC, fetch, and all data memory accesses) are suppressed.
    /// It is cleared automatically after the instruction completes.
    /// Flush local cycle counter to the shared atomic.
    #[inline(always)]
    pub fn flush_cycles(&mut self) {
        unsafe { &*self.cycles_ptr }.store(self.local_cycles, Ordering::Relaxed);
    }

    pub fn step(&mut self) -> ExecStatus {
        // Increment local cycle counter (flushed to atomic by outer loop)
        self.local_cycles = self.local_cycles.wrapping_add(1);

        /*
        // Reload external interrupt state every 16 instructions
        if self.local_cycles & 0xF == 0 {
            self.cached_pending = unsafe { &*self.interrupts_ptr }.load(Ordering::Relaxed);
        }
        let pending = self.cached_pending;
        */
        // this seems to be a wash or slightly better without a branch, relaxed atomic loads are essentially MOV
        let pending = unsafe { &*self.interrupts_ptr }.load(Ordering::Relaxed);

        let pc = self.core.pc;
        #[cfg(not(feature = "lightning"))]
        if self.bp_enabled() && self.check_breakpoint::<{ BpType::Pc as u8 }>(pc) {
            return EXEC_BREAKPOINT;
        }

        // Advance cp0_count at calibrated wall-clock rate.
        // cp0_count bits[47:16] are the hardware 32-bit count; bits[15:0] are the fraction.
        // count_step is the per-instruction increment in the same 16.16 representation.
        let prev = self.core.cp0_count;
        self.core.cp0_count = prev.wrapping_add(self.core.count_step) & 0x0000_FFFF_FFFF_FFFF;
        if self.core.cp0_compare != 0 && prev < self.core.cp0_compare && self.core.cp0_count >= self.core.cp0_compare {
            self.core.cp0_cause |= crate::mips_core::CAUSE_IP7;
            unsafe { &*self.fasttick_ptr }.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        // Fast path: skip all signal/interrupt handling when nothing is pending
        if (pending | self.core.cp0_cause as u64) != 0 {
            // Soft reset (bit 63)
            if pending & SOFT_RESET_BIT != 0 {
                self.core.reset(true); // clears interrupts word (including bit 63)
                self.in_delay_slot = false;
                self.delay_slot_target = 0;
                return EXEC_COMPLETE;
            }

            // Merge external IP bits into Cause
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

        let result = match self.fetch_instr(pc) {
            Ok(d) => {
                let slot = d as *mut DecodedInstr;
                if decode_into::<T, C>(unsafe { &mut *slot }) {
                    #[cfg(feature = "developer")]
                    self.decoded_count.fetch_add(1, Ordering::Relaxed);
                }
                let d = unsafe { &*slot };
                #[cfg(not(feature = "lightning"))]
                self.traceback.push(pc, d.raw);
                self.exec_decoded(d)
            }
            Err(s) if s & EXEC_IS_EXCEPTION != 0 => {
                self.handle_exception(s)
            }
            Err(s) => s,
        };
        self.skip_breakpoints = false;
        result
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
                // Ignore bottom 2 bits for address comparison
                if (bp.addr & !3) == (addr & !3) {
                    // Check optional register condition
                    if let Some(expr) = &bp.condition {
                        let symbols = self.symbols.lock();
                        match expr.eval(&self.core, Some(&symbols)) {
                            Ok(val) => if val == 0 { continue; },
                            // If evaluation fails, we assume the condition is not met (or maybe we should break to show error?)
                            Err(_) => continue, 
                        }
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
        exec_exception(EXC_RI)
    }

    fn exec_reserved(&mut self, d: &DecodedInstr) -> ExecStatus {
        self.reserved_instruction(d)
    }

    /// No-op handler — used as the default for zero-initialised DecodedInstr.
    fn exec_nop(&mut self, _d: &DecodedInstr) -> ExecStatus {
        EXEC_COMPLETE
    }

    /// Handle an exception: update CP0 registers and jump to handler vector.
    /// Takes an ExecStatus with EXEC_IS_EXCEPTION set; extracts code and TLB-refill flag.
    fn handle_exception(&mut self, status: ExecStatus) -> ExecStatus {
        let is_tlb_refill  = status & EXEC_IS_TLB_REFILL  != 0;
        let is_xtlb_refill = status & EXEC_IS_XTLB_REFILL != 0;

        // Clear LLBit on any exception
        self.cache.set_llbit(false);

        // 1. Check if EXL is already set (before updating anything)
        // EPC is only updated when EXL transitions from 0 to 1
        let was_exl = (self.core.cp0_status & STATUS_EXL) != 0;

        // 2. Update Cause Register — ExcCode bits [6:2] already in correct position in status
        let mut cause = self.core.cp0_cause;
        cause = (cause & !CAUSE_EXCCODE_MASK) | (status & CAUSE_EXCCODE_MASK);

        // 3. Update EPC only if EXL was 0 (transition 0->1)
        // If EXL was already 1, preserve the original EPC from the first exception
        if !was_exl {
            // Handle Branch Delay
            if self.in_delay_slot {
                cause |= CAUSE_BD;
                self.core.cp0_epc = self.core.pc.wrapping_sub(4); // Point to branch instr
            } else {
                cause &= !CAUSE_BD;
                self.core.cp0_epc = self.core.pc; // Point to faulting instr
            }
        }
        // Note: BD flag in Cause still updated even if EXL was 1, per R4000 spec
        self.core.cp0_cause = cause;

        // 4. Set EXL bit in Status Register (Exception Level)
        self.core.cp0_status |= STATUS_EXL;
        self.core.nanotlb_invalidate();

        // 3. Determine Exception Vector
        let bev = (self.core.cp0_status & STATUS_BEV) != 0;

        // Base address: BEV=1 -> 0xBFC00200, BEV=0 -> 0x80000000
        let vector_base = if bev { 0xFFFFFFFF_BFC00200 } else { 0xFFFFFFFF_80000000 };

        // Offset: XTLB refill (EXL=0) -> 0x080, TLB refill (EXL=0) -> 0x000, General -> 0x180
        let offset = if !was_exl && is_tlb_refill {
            if is_xtlb_refill { 0x080 } else { 0x000 }
        } else {
            0x180
        };

        // In developer builds, bus error exceptions (IBE/DBE) break into the
        // monitor rather than dispatching to the MIPS vector.
        #[cfg(feature = "developerx")]
        {
            let exc_code = (status & CAUSE_EXCCODE_MASK) >> 2;
            if exc_code == EXC_IBE || exc_code == EXC_DBE {
                eprintln!("BUS ERROR ({}) at PC={:#010x} EPC={:#010x}",
                    if exc_code == EXC_IBE { "IBE" } else { "DBE" },
                    self.core.pc, self.core.cp0_epc);
                return EXEC_BREAKPOINT;
            }
        }
        #[cfg(feature = "developerx")]
        {
            let exc_code = (status & CAUSE_EXCCODE_MASK) >> 2;
            if exc_code == EXC_ADEL || exc_code == EXC_ADES {
                eprintln!("ADDRESS ERROR ({}) at PC={:#010x} EPC={:#010x} BadVAddr={:#010x}",
                    if exc_code == EXC_ADEL { "ADEL" } else { "ADES" },
                    self.core.pc, self.core.cp0_epc, self.core.cp0_badvaddr);
                return EXEC_BREAKPOINT;
            }
        }
        #[cfg(feature = "developerx")]
        {
            let exc_code = (status & CAUSE_EXCCODE_MASK) >> 2;
            if (exc_code == EXC_TLBL || exc_code == EXC_TLBS) && (self.core.cp0_badvaddr as u32 == 0xFF800000){
                eprintln!("ADDRESS ERROR ({}) at PC={:#010x} EPC={:#010x} BadVAddr={:#010x}",
                    if exc_code == EXC_TLBL { "TLBL" } else { "TLBS" },
                    self.core.pc, self.core.cp0_epc, self.core.cp0_badvaddr);
                return EXEC_BREAKPOINT;
            }
        }

        // Jump to exception vector
        self.core.pc = vector_base + offset;
        // Reset delay slot state as we are jumping to a new context
        self.in_delay_slot = false;
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
                // 32-bit mode: xtlb=false → UTLB vector on miss
                self.tlb_translate_impl::<DEBUG>(virt_addr, access_type, false)
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
                    self.tlb_translate_impl::<DEBUG>(virt_addr, access_type, false)
                } else {
                    if !DEBUG { self.core.cp0_badvaddr = virt_addr; }
                    TranslateResult::exc(addr_exc(access_type == AccessType::Write))
                }
            }

            // KSEG3: 0xE0000000 - 0xFFFFFFFF (kernel segment, TLB mapped)
            7 => {
                if PRIV == PRIV_KERNEL {
                    self.tlb_translate_impl::<DEBUG>(virt_addr, access_type, false)
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
                self.tlb_translate_impl::<DEBUG>(virt_addr, access_type, true)
            }

            // xsseg: 0x4000_0000_0000_0000 - 0x7FFF_FFFF_FFFF_FFFF (supervisor segment)
            1 => {
                if PRIV == PRIV_KERNEL || PRIV == PRIV_SUPERVISOR {
                    self.tlb_translate_impl::<DEBUG>(virt_addr, access_type, true)
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
                            _ => return self.tlb_translate_impl::<DEBUG>(virt_addr, access_type, false),
                        }
                    }
                    // True 64-bit xkseg: xtlb=true
                    self.tlb_translate_impl::<DEBUG>(virt_addr, access_type, true)
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
    fn tlb_translate_impl<const DEBUG: bool>(&mut self, virt_addr: u64, access_type: AccessType, xtlb: bool) -> TranslateResult {
        use crate::mips_tlb::TlbResult;

        // Get current ASID from EntryHi register
        let asid = (self.core.cp0_entryhi & 0xFF) as u8;

        // Query the TLB — xtlb=true uses 64-bit VPN comparison mask
        let result = self.tlb.translate(virt_addr, asid, access_type, xtlb);

        let tlb_miss_code = if access_type == AccessType::Write { EXC_TLBS } else { EXC_TLBL };

        match result {
            TlbResult::Hit { phys_addr, cache_attr, dirty } => {
                if access_type == AccessType::Write && !dirty {
                    if !DEBUG { self.update_tlb_exception_registers(virt_addr, xtlb); }
                    TranslateResult::exc(exec_exception(EXC_MOD))
                } else {
                    TranslateResult::ok(phys_addr, cache_attr as u32)
                }
            }
            TlbResult::Miss { .. } => {
                if !DEBUG { self.update_tlb_exception_registers(virt_addr, xtlb); }
                // Miss: XTLB vector (0x080) for 64-bit extended addresses, UTLB vector (0x000) otherwise
                TranslateResult::exc(if xtlb { exec_xtlb_miss(tlb_miss_code) } else { exec_tlb_miss(tlb_miss_code) })
            }
            TlbResult::Invalid { .. } => {
                if !DEBUG { self.update_tlb_exception_registers(virt_addr, xtlb); }
                // Invalid: always general vector (0x180)
                TranslateResult::exc(exec_exception(tlb_miss_code))
            }
            TlbResult::Modified { .. } => {
                if !DEBUG { self.update_tlb_exception_registers(virt_addr, xtlb); }
                TranslateResult::exc(exec_exception(EXC_MOD))
            }
        }
    }

    /// Debug helper to translate address without side effects
    pub fn debug_translate(&mut self, virt_addr: u64) -> TranslateResult {
        self.translate_impl::<true>(virt_addr, AccessType::Debug)
    }

    /// Update CP0 BadVAddr, EntryHi, Context, XContext for any TLB exception.
    /// `xtlb`: true for extended (64-bit) translations — uses 64-bit VPN mask for EntryHi.
    fn update_tlb_exception_registers(&mut self, virt_addr: u64, xtlb: bool) {
        const EH_VPN2_32: u64 = 0x0000_0000_FFFF_E000;
        const EH_VPN2_64: u64 = 0x0000_00FF_FFFF_E000;
        const EH_REGION:  u64 = 0xC000_0000_0000_0000;

        self.core.cp0_badvaddr = virt_addr;

        // EntryHi: VPN from address masked per translation mode, ASID preserved.
        let asid = self.core.cp0_entryhi & 0xFF;
        let vpn_mask = if xtlb { EH_REGION | EH_VPN2_64 } else { EH_VPN2_32 };
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
    /// or an error ExecStatus on fault/breakpoint.
    fn fetch_instr(&mut self, virt_addr: u64) -> Result<*const DecodedInstr, ExecStatus> {
        self.fetch_instr_impl::<false>(virt_addr)
    }

    /// Debug instruction fetch: kernel-mode override, no breakpoints, no CP0 side-effects.
    /// Returns the raw instruction word only (no decode).
    pub fn debug_fetch_instr(&mut self, virt_addr: u64) -> Result<u32, ExecStatus> {
        self.fetch_instr_impl::<true>(virt_addr)
            .map(|slot| unsafe { (*slot).raw })
    }

    /// Core instruction fetch.  When `DEBUG=true`:
    /// - Privilege is treated as Kernel (via translate_impl)
    /// - Breakpoint checks are skipped
    /// - cp0_badvaddr is never written
    /// Returns a pointer to the DecodedInstr slot (cache or scratch).
    #[inline]
    fn fetch_instr_impl<const DEBUG: bool>(&mut self, virt_addr: u64) -> Result<*const DecodedInstr, ExecStatus> {
        #[cfg(not(feature = "lightning"))]
        if !DEBUG && self.bp_enabled() && self.check_breakpoint::<{ BpType::VirtFetch as u8 }>(virt_addr) {
            return Err(EXEC_BREAKPOINT);
        }

        let translate_result = if DEBUG {
            self.translate_impl::<true>(virt_addr, AccessType::Fetch)
        } else {
            (self.translate_fn)(self, virt_addr, AccessType::Fetch)
        };
        if translate_result.is_exception() { return Err(translate_result.status); }
        let phys_addr = translate_result.phys;
        if translate_result.is_cached() {
            #[cfg(not(feature = "lightning"))]
            if !DEBUG && self.bp_enabled() && self.check_breakpoint::<{ BpType::PhysFetch as u8 }>(phys_addr as u64) {
                return Err(EXEC_BREAKPOINT);
            }

            match self.cache.fetch(virt_addr, phys_addr as u64) {
                FetchResult::Hit(slot) => Ok(slot),
                FetchResult::Busy => Err(EXEC_RETRY),
                FetchResult::VirtualCoherencyException => {
                    if !DEBUG { self.core.cp0_badvaddr = virt_addr; }
                    Err(exec_exception(EXC_VCEI))
                }
                FetchResult::Error => Err(exec_exception(EXC_IBE)),
            }
        } else {
            #[cfg(not(feature = "lightning"))]
            if !DEBUG && self.bp_enabled() && self.check_breakpoint::<{ BpType::PhysFetch as u8 }>(phys_addr as u64) {
                return Err(EXEC_BREAKPOINT);
            }

            #[cfg(feature = "developer")]
            self.uncached_fetch_count.fetch_add(1, Ordering::Relaxed);
            let r = self.sysad.read32(phys_addr);
            if r.is_ok() {
                if self.ins.raw != r.data { self.ins.decoded = false; }
                self.ins.raw = r.data;
                Ok(&self.ins as *const DecodedInstr)
            } else if r.status == BUS_BUSY {
                Err(EXEC_RETRY)
            } else {
                eprintln!("Bus error on instruction fetch: PC={:016x} PA={:08x} status={:08x}", virt_addr, phys_addr, r.status);
                Err(exec_exception(EXC_IBE))
            }
        }
    }

    /// Production data read (with breakpoints, updates CP0 state on exceptions).
    #[inline]
    fn read_data<const SIZE: usize>(&mut self, virt_addr: u64) -> Result<u64, ExecStatus> {
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

        let access_type = if DEBUG { AccessType::Debug } else { AccessType::Read };
        let translate_result = if DEBUG {
            self.translate_impl::<true>(virt_addr, access_type)
        } else {
            (self.translate_fn)(self, virt_addr, access_type)
        };
        if translate_result.is_exception() { return Err(translate_result.status); }
        {
            let phys_addr = translate_result.phys as u64;
            let is_cached = translate_result.is_cached();

            if is_cached {
                    // Cached access uses D-Cache
                    #[cfg(not(feature = "lightning"))]
                    if !DEBUG && self.bp_enabled() && self.check_breakpoint::<{ BpType::PhysRead as u8 }>(phys_addr) {
                        return Err(EXEC_BREAKPOINT);
                    }

                    let r = self.cache.read(virt_addr, phys_addr, SIZE);
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
                                eprintln!("Bus error on uncached read{}: PC={:016x} VA={:016x} PA={:016x} status={:08x}", SIZE*8, self.core.pc, virt_addr, phys_addr, r.status);
                                if !DEBUG { self.core.cp0_badvaddr = virt_addr; }
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
                    res
                }
        }
    }

    /// Production data write (with breakpoints, undo tracking, updates CP0 state on exceptions).
    #[inline]
    fn write_data<const SIZE: usize>(&mut self, virt_addr: u64, val: u64) -> ExecStatus {
        self.write_data_impl::<false, SIZE>(virt_addr, val)
    }

    /// Partial masked doubleword write for SDL/SDR/SWL/SWR.
    /// Only bytes where the corresponding mask byte is non-zero are written.
    /// `virt_addr` must be 8-byte aligned; val/mask are in MIPS big-endian doubleword space.
    #[inline]
    fn write_data64_masked(&mut self, virt_addr: u64, val: u64, mask: u64) -> ExecStatus {
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

        let access_type = if DEBUG { AccessType::Debug } else { AccessType::Write };
        let translate_result = if DEBUG {
            self.translate_impl::<true>(virt_addr, access_type)
        } else {
            (self.translate_fn)(self, virt_addr, access_type)
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

        if is_cached {
            let status = if SIZE == 1 {
                self.cache.write8(virt_addr, phys_addr, val as u8)
            } else if SIZE == 2 {
                self.cache.write16(virt_addr, phys_addr, val as u16)
            } else if SIZE == 4 {
                self.cache.write32(virt_addr, phys_addr, val as u32)
            } else {
                self.cache.write64(virt_addr, phys_addr, val)
            };
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
            if ws != BUS_OK && ws != BUS_BUSY {
                eprintln!("Bus error on uncached write{}: PC={:016x} VA={:016x} PA={:016x} val={:016x} status={:08x}", SIZE*8, self.core.pc, virt_addr, phys_addr, val, ws);
            }
            ws
        }
    }

    /// Partial masked doubleword store: SDL/SDR/SWL/SWR.
    /// `virt_addr` is 8-byte aligned; val/mask are in MIPS big-endian doubleword space.
    fn write_data64_masked_impl<const DEBUG: bool>(&mut self, virt_addr: u64, val: u64, mask: u64) -> ExecStatus {
        #[cfg(not(feature = "lightning"))]
        if !DEBUG && self.bp_enabled() && self.check_breakpoint::<{ BpType::VirtWrite as u8 }>(virt_addr) {
            return EXEC_BREAKPOINT;
        }

        // virt_addr is already doubleword-aligned (callers guarantee this)
        let access_type = if DEBUG { AccessType::Debug } else { AccessType::Write };
        let translate_result = if DEBUG {
            self.translate_impl::<true>(virt_addr, access_type)
        } else {
            (self.translate_fn)(self, virt_addr, access_type)
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
            if ws != BUS_OK && ws != BUS_BUSY {
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
        EXEC_COMPLETE
    }
    fn exec_movci(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_reg = d.rs as u32;
        let rd_reg = d.rd as u32;
        let cc = (d.raw >> 18) & 0x7;
        let tf = ((d.raw >> 16) & 0x1) != 0;
        let cc_value = self.core.get_fpu_cc(cc);
        if cc_value == tf {
            let rs_val = self.core.read_gpr(rs_reg);
            self.core.write_gpr(rd_reg, rs_val);
        }
        EXEC_COMPLETE
    }
    fn exec_srl(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rt_val = self.core.read_gpr(d.rt as u32) as u32;
        let rd_reg = d.rd as u32;
        let sa_val = d.sa as u32;
        self.core.write_gpr(rd_reg, (rt_val >> sa_val) as i32 as i64 as u64);
        EXEC_COMPLETE
    }
    fn exec_sra(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rt_val = self.core.read_gpr(d.rt as u32) as i32;
        let rd_reg = d.rd as u32;
        let sa_val = d.sa as u32;
        self.core.write_gpr(rd_reg, (rt_val >> sa_val) as i64 as u64);
        EXEC_COMPLETE
    }
    fn exec_sllv(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32);
        let rd_reg = d.rd as u32;
        let sa_val = (rs_val & 0x1F) as u32;
        self.core.write_gpr(rd_reg, (rt_val << sa_val) as u32 as i32 as i64 as u64);
        EXEC_COMPLETE
    }
    fn exec_srlv(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32) as u32;
        let rd_reg = d.rd as u32;
        let sa_val = (rs_val & 0x1F) as u32;
        self.core.write_gpr(rd_reg, (rt_val >> sa_val) as i32 as i64 as u64);
        EXEC_COMPLETE
    }
    fn exec_srav(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32) as i32;
        let rd_reg = d.rd as u32;
        let sa_val = (rs_val & 0x1F) as u32;
        self.core.write_gpr(rd_reg, (rt_val >> sa_val) as i64 as u64);
        EXEC_COMPLETE
    }
    fn exec_jr(&mut self, d: &DecodedInstr) -> ExecStatus {
        let target = self.core.read_gpr(d.rs as u32);
        self.branch_delay(target)
    }
    fn exec_jalr(&mut self, d: &DecodedInstr) -> ExecStatus {
        let target = self.core.read_gpr(d.rs as u32);
        let rd_reg = d.rd as u32;
        self.core.write_gpr(rd_reg, self.core.pc + 8);
        self.branch_delay(target)
    }
    fn exec_syscall(&mut self, _d: &DecodedInstr) -> ExecStatus {
        exec_exception(EXC_SYS)
    }
    fn exec_break(&mut self, _d: &DecodedInstr) -> ExecStatus {
        exec_exception(EXC_BP)
    }
    fn exec_sync(&mut self, _d: &DecodedInstr) -> ExecStatus {
        EXEC_COMPLETE
    }
    fn exec_mfhi(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rd_reg = d.rd as u32;
        self.core.write_gpr(rd_reg, self.core.hi);
        EXEC_COMPLETE
    }
    fn exec_mthi(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        self.core.hi = rs_val;
        EXEC_COMPLETE
    }
    fn exec_mflo(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rd_reg = d.rd as u32;
        self.core.write_gpr(rd_reg, self.core.lo);
        EXEC_COMPLETE
    }
    fn exec_mtlo(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        self.core.lo = rs_val;
        EXEC_COMPLETE
    }
    fn exec_mult(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i32 as i64;
        let rt_val = self.core.read_gpr(d.rt as u32) as i32 as i64;
        let result = rs_val * rt_val;
        self.core.lo = (result as u32) as i32 as i64 as u64;
        self.core.hi = (result >> 32) as u32 as i32 as i64 as u64;
        EXEC_COMPLETE
    }
    fn exec_multu(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as u32 as u64;
        let rt_val = self.core.read_gpr(d.rt as u32) as u32 as u64;
        let result = rs_val * rt_val;
        self.core.lo = (result as u32) as i32 as i64 as u64;
        self.core.hi = (result >> 32) as u32 as i32 as i64 as u64;
        EXEC_COMPLETE
    }
    fn exec_div(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i32;
        let rt_val = self.core.read_gpr(d.rt as u32) as i32;
        if rt_val == 0 {
            EXEC_COMPLETE
        } else {
            let quotient = rs_val.wrapping_div(rt_val);
            let remainder = rs_val.wrapping_rem(rt_val);
            self.core.lo = quotient as i64 as u64;
            self.core.hi = remainder as i64 as u64;
            EXEC_COMPLETE
        }
    }
    fn exec_divu(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as u32;
        let rt_val = self.core.read_gpr(d.rt as u32) as u32;
        if rt_val == 0 {
            EXEC_COMPLETE
        } else {
            let quotient = rs_val / rt_val;
            let remainder = rs_val % rt_val;
            self.core.lo = quotient as i32 as i64 as u64;
            self.core.hi = remainder as i32 as i64 as u64;
            EXEC_COMPLETE
        }
    }
    fn exec_dmult(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64 as i128;
        let rt_val = self.core.read_gpr(d.rt as u32) as i64 as i128;
        let result = rs_val * rt_val;
        self.core.lo = result as u128 as u64;
        self.core.hi = (result >> 64) as u128 as u64;
        EXEC_COMPLETE
    }
    fn exec_dmultu(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as u128;
        let rt_val = self.core.read_gpr(d.rt as u32) as u128;
        let result = rs_val * rt_val;
        self.core.lo = result as u64;
        self.core.hi = (result >> 64) as u64;
        EXEC_COMPLETE
    }
    fn exec_ddiv(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        let rt_val = self.core.read_gpr(d.rt as u32) as i64;
        if rt_val == 0 {
            EXEC_COMPLETE
        } else if rs_val == i64::MIN && rt_val == -1 {
            EXEC_COMPLETE
        } else {
            self.core.lo = rs_val.wrapping_div(rt_val) as u64;
            self.core.hi = rs_val.wrapping_rem(rt_val) as u64;
            EXEC_COMPLETE
        }
    }
    fn exec_ddivu(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32);
        if rt_val == 0 {
            EXEC_COMPLETE
        } else {
            self.core.lo = rs_val / rt_val;
            self.core.hi = rs_val % rt_val;
            EXEC_COMPLETE
        }
    }
    fn exec_add(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i32;
        let rt_val = self.core.read_gpr(d.rt as u32) as i32;
        let rd_reg = d.rd as u32;
        match rs_val.checked_add(rt_val) {
            Some(result) => {
                self.core.write_gpr(rd_reg, result as i64 as u64);
                EXEC_COMPLETE
            }
            None => exec_exception(EXC_OV),
        }
    }
    fn exec_addu(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as u32;
        let rt_val = self.core.read_gpr(d.rt as u32) as u32;
        let rd_reg = d.rd as u32;
        self.core.write_gpr(rd_reg, rs_val.wrapping_add(rt_val) as i32 as i64 as u64);
        EXEC_COMPLETE
    }
    fn exec_sub(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i32;
        let rt_val = self.core.read_gpr(d.rt as u32) as i32;
        let rd_reg = d.rd as u32;
        match rs_val.checked_sub(rt_val) {
            Some(result) => {
                self.core.write_gpr(rd_reg, result as i64 as u64);
                EXEC_COMPLETE
            }
            None => exec_exception(EXC_OV),
        }
    }
    fn exec_subu(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as u32;
        let rt_val = self.core.read_gpr(d.rt as u32) as u32;
        let rd_reg = d.rd as u32;
        self.core.write_gpr(rd_reg, rs_val.wrapping_sub(rt_val) as i32 as i64 as u64);
        EXEC_COMPLETE
    }
    fn exec_and(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32);
        let rd_reg = d.rd as u32;
        self.core.write_gpr(rd_reg, rs_val & rt_val);
        EXEC_COMPLETE
    }
    fn exec_or(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32);
        let rd_reg = d.rd as u32;
        self.core.write_gpr(rd_reg, rs_val | rt_val);
        EXEC_COMPLETE
    }
    fn exec_xor(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32);
        let rd_reg = d.rd as u32;
        self.core.write_gpr(rd_reg, rs_val ^ rt_val);
        EXEC_COMPLETE
    }
    fn exec_nor(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32);
        let rd_reg = d.rd as u32;
        self.core.write_gpr(rd_reg, !(rs_val | rt_val));
        EXEC_COMPLETE
    }
    fn exec_slt(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        let rt_val = self.core.read_gpr(d.rt as u32) as i64;
        let rd_reg = d.rd as u32;
        self.core.write_gpr(rd_reg, if rs_val < rt_val { 1 } else { 0 });
        EXEC_COMPLETE
    }
    fn exec_sltu(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32);
        let rd_reg = d.rd as u32;
        self.core.write_gpr(rd_reg, if rs_val < rt_val { 1 } else { 0 });
        EXEC_COMPLETE
    }
    fn exec_dadd(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        let rt_val = self.core.read_gpr(d.rt as u32) as i64;
        let rd_reg = d.rd as u32;
        match rs_val.checked_add(rt_val) {
            Some(result) => {
                self.core.write_gpr(rd_reg, result as u64);
                EXEC_COMPLETE
            }
            None => exec_exception(EXC_OV),
        }
    }
    fn exec_daddu(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32);
        let rd_reg = d.rd as u32;
        self.core.write_gpr(rd_reg, rs_val.wrapping_add(rt_val));
        EXEC_COMPLETE
    }
    fn exec_dsub(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        let rt_val = self.core.read_gpr(d.rt as u32) as i64;
        let rd_reg = d.rd as u32;
        match rs_val.checked_sub(rt_val) {
            Some(result) => {
                self.core.write_gpr(rd_reg, result as u64);
                EXEC_COMPLETE
            }
            None => exec_exception(EXC_OV),
        }
    }
    fn exec_dsubu(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32);
        let rd_reg = d.rd as u32;
        self.core.write_gpr(rd_reg, rs_val.wrapping_sub(rt_val));
        EXEC_COMPLETE
    }
    fn exec_tge(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        let rt_val = self.core.read_gpr(d.rt as u32) as i64;
        if rs_val >= rt_val { exec_exception(EXC_TR) } else { EXEC_COMPLETE }
    }
    fn exec_tgeu(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32);
        if rs_val >= rt_val { exec_exception(EXC_TR) } else { EXEC_COMPLETE }
    }
    fn exec_tlt(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        let rt_val = self.core.read_gpr(d.rt as u32) as i64;
        if rs_val < rt_val { exec_exception(EXC_TR) } else { EXEC_COMPLETE }
    }
    fn exec_tltu(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32);
        if rs_val < rt_val { exec_exception(EXC_TR) } else { EXEC_COMPLETE }
    }
    fn exec_teq(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32);
        if rs_val == rt_val { exec_exception(EXC_TR) } else { EXEC_COMPLETE }
    }
    fn exec_tne(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32);
        if rs_val != rt_val { exec_exception(EXC_TR) } else { EXEC_COMPLETE }
    }
    fn exec_movz(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32);
        let rd_reg = d.rd as u32;
        if rt_val == 0 { self.core.write_gpr(rd_reg, rs_val); }
        EXEC_COMPLETE
    }
    fn exec_movn(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32);
        let rd_reg = d.rd as u32;
        if rt_val != 0 { self.core.write_gpr(rd_reg, rs_val); }
        EXEC_COMPLETE
    }
    fn exec_dsll(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rt_val = self.core.read_gpr(d.rt as u32);
        let rd_reg = d.rd as u32;
        let sa_val = d.sa as u32;
        self.core.write_gpr(rd_reg, rt_val << sa_val);
        EXEC_COMPLETE
    }
    fn exec_dsrl(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rt_val = self.core.read_gpr(d.rt as u32);
        let rd_reg = d.rd as u32;
        let sa_val = d.sa as u32;
        self.core.write_gpr(rd_reg, rt_val >> sa_val);
        EXEC_COMPLETE
    }
    fn exec_dsra(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rt_val = self.core.read_gpr(d.rt as u32) as i64;
        let rd_reg = d.rd as u32;
        let sa_val = d.sa as u32;
        self.core.write_gpr(rd_reg, (rt_val >> sa_val) as u64);
        EXEC_COMPLETE
    }
    fn exec_dsll32(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rt_val = self.core.read_gpr(d.rt as u32);
        let rd_reg = d.rd as u32;
        let sa_val = d.sa as u32;
        self.core.write_gpr(rd_reg, rt_val << (sa_val + 32));
        EXEC_COMPLETE
    }
    fn exec_dsrl32(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rt_val = self.core.read_gpr(d.rt as u32);
        let rd_reg = d.rd as u32;
        let sa_val = d.sa as u32;
        self.core.write_gpr(rd_reg, rt_val >> (sa_val + 32));
        EXEC_COMPLETE
    }
    fn exec_dsra32(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rt_val = self.core.read_gpr(d.rt as u32) as i64;
        let rd_reg = d.rd as u32;
        let sa_val = d.sa as u32;
        self.core.write_gpr(rd_reg, (rt_val >> (sa_val + 32)) as u64);
        EXEC_COMPLETE
    }
    fn exec_dsllv(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32);
        let rd_reg = d.rd as u32;
        let sa_val = rs_val & 0x3F;
        self.core.write_gpr(rd_reg, rt_val << sa_val);
        EXEC_COMPLETE
    }
    fn exec_dsrlv(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32);
        let rd_reg = d.rd as u32;
        let sa_val = rs_val & 0x3F;
        self.core.write_gpr(rd_reg, rt_val >> sa_val);
        EXEC_COMPLETE
    }
    fn exec_dsrav(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32) as i64;
        let rd_reg = d.rd as u32;
        let sa_val = rs_val & 0x3F;
        self.core.write_gpr(rd_reg, (rt_val >> sa_val) as u64);
        EXEC_COMPLETE
    }
    // Jump and Branch Instructions

    // J - Jump
    // Unconditional jump within 256MB region
    fn exec_j(&mut self, d: &DecodedInstr) -> ExecStatus {
        // d.immi64() = (target26 << 2): replace low 28 bits of PC+4
        let target = ((self.core.pc + 4) & 0xFFFFFFFF_F0000000) | d.immi64();
        self.branch_delay(target)
    }

    // JAL - Jump and Link
    // Unconditional jump, save return address in r31
    fn exec_jal(&mut self, d: &DecodedInstr) -> ExecStatus {
        let target = ((self.core.pc + 4) & 0xFFFFFFFF_F0000000) | d.immi64();
        self.core.write_gpr(31, self.core.pc + 8); // Return address (PC of delay slot + 4)
        self.branch_delay(target)
    }

    // BEQ - Branch on Equal
    fn exec_beq(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_val = self.core.read_gpr(d.rt as u32);
        if rs_val == rt_val {
            let target = self.core.pc.wrapping_add(4).wrapping_add(d.immu64());
            self.branch_delay(target)
        } else {
            EXEC_COMPLETE
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
            EXEC_COMPLETE
        }
    }

    // BLEZ - Branch on Less Than or Equal to Zero
    fn exec_blez(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        if rs_val <= 0 {
            let target = self.core.pc.wrapping_add(4).wrapping_add(d.immu64());
            self.branch_delay(target)
        } else {
            EXEC_COMPLETE
        }
    }

    // BGTZ - Branch on Greater Than Zero
    fn exec_bgtz(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        if rs_val > 0 {
            let target = self.core.pc.wrapping_add(4).wrapping_add(d.immu64());
            self.branch_delay(target)
        } else {
            EXEC_COMPLETE
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
            EXEC_BRANCH_LIKELY_SKIP
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
            EXEC_BRANCH_LIKELY_SKIP
        }
    }

    // BLEZL - Branch on Less Than or Equal to Zero Likely
    fn exec_blezl(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        if rs_val <= 0 {
            let target = self.core.pc.wrapping_add(4).wrapping_add(d.immu64());
            self.branch_delay(target)
        } else {
            EXEC_BRANCH_LIKELY_SKIP
        }
    }

    // BGTZL - Branch on Greater Than Zero Likely
    fn exec_bgtzl(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        if rs_val > 0 {
            let target = self.core.pc.wrapping_add(4).wrapping_add(d.immu64());
            self.branch_delay(target)
        } else {
            EXEC_BRANCH_LIKELY_SKIP
        }
    }


    // REGIMM individual methods
    fn exec_bltz(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        let target = self.core.pc.wrapping_add(4).wrapping_add(d.immu64());
        if rs_val < 0 { self.branch_delay(target) } else { EXEC_COMPLETE }
    }
    fn exec_bgez(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        let target = self.core.pc.wrapping_add(4).wrapping_add(d.immu64());
        if rs_val >= 0 { self.branch_delay(target) } else { EXEC_COMPLETE }
    }
    fn exec_bltzl_ri(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        let target = self.core.pc.wrapping_add(4).wrapping_add(d.immu64());
        if rs_val < 0 { self.branch_delay(target) } else { EXEC_BRANCH_LIKELY_SKIP }
    }
    fn exec_bgezl_ri(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        let target = self.core.pc.wrapping_add(4).wrapping_add(d.immu64());
        if rs_val >= 0 { self.branch_delay(target) } else { EXEC_BRANCH_LIKELY_SKIP }
    }
    fn exec_tgei(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        if rs_val >= d.imms64() { exec_exception(EXC_TR) } else { EXEC_COMPLETE }
    }
    fn exec_tgeiu(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val_u = self.core.read_gpr(d.rs as u32);
        if rs_val_u >= d.immu64() { exec_exception(EXC_TR) } else { EXEC_COMPLETE }
    }
    fn exec_tlti(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        if rs_val < d.imms64() { exec_exception(EXC_TR) } else { EXEC_COMPLETE }
    }
    fn exec_tltiu(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val_u = self.core.read_gpr(d.rs as u32);
        if rs_val_u < d.immu64() { exec_exception(EXC_TR) } else { EXEC_COMPLETE }
    }
    fn exec_teqi(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        if rs_val == d.imms64() { exec_exception(EXC_TR) } else { EXEC_COMPLETE }
    }
    fn exec_tnei(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        if rs_val != d.imms64() { exec_exception(EXC_TR) } else { EXEC_COMPLETE }
    }
    fn exec_bltzal(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        let target = self.core.pc.wrapping_add(4).wrapping_add(d.immu64());
        self.core.write_gpr(31, self.core.pc + 8);
        if rs_val < 0 { self.branch_delay(target) } else { EXEC_COMPLETE }
    }
    fn exec_bgezal(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        let target = self.core.pc.wrapping_add(4).wrapping_add(d.immu64());
        self.core.write_gpr(31, self.core.pc + 8);
        if rs_val >= 0 { self.branch_delay(target) } else { EXEC_COMPLETE }
    }
    fn exec_bltzall(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        let target = self.core.pc.wrapping_add(4).wrapping_add(d.immu64());
        self.core.write_gpr(31, self.core.pc + 8);
        if rs_val < 0 { self.branch_delay(target) } else { EXEC_BRANCH_LIKELY_SKIP }
    }
    fn exec_bgezall(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        let target = self.core.pc.wrapping_add(4).wrapping_add(d.immu64());
        self.core.write_gpr(31, self.core.pc + 8);
        if rs_val >= 0 { self.branch_delay(target) } else { EXEC_BRANCH_LIKELY_SKIP }
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
                EXEC_COMPLETE
            }
            None => exec_exception(EXC_OV),
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
        self.core.write_gpr(rt_reg, rs_val.wrapping_add(imm_val) as i32 as i64 as u64);
        EXEC_COMPLETE
    }

    // DADDI - Doubleword Add Immediate (with overflow exception)
    fn exec_daddi(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        let rt_reg = d.rt as u32;
        match rs_val.checked_add(d.imms64()) {
            Some(result) => {
                self.core.write_gpr(rt_reg, result as u64);
                EXEC_COMPLETE
            }
            None => exec_exception(EXC_OV),
        }
    }

    // DADDIU - Doubleword Add Immediate Unsigned (no overflow exception)
    fn exec_daddiu(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_reg = d.rt as u32;
        self.core.write_gpr(rt_reg, rs_val.wrapping_add(d.immu64()));
        EXEC_COMPLETE
    }

    // SLTI - Set on Less Than Immediate (signed)
    fn exec_slti(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32) as i64;
        let rt_reg = d.rt as u32;
        self.core.write_gpr(rt_reg, if rs_val < d.imms64() { 1 } else { 0 });
        EXEC_COMPLETE
    }

    // SLTIU - Set on Less Than Immediate Unsigned (sign-extended imm compared as unsigned)
    fn exec_sltiu(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_reg = d.rt as u32;
        self.core.write_gpr(rt_reg, if rs_val < d.immu64() { 1 } else { 0 });
        EXEC_COMPLETE
    }

    // ANDI - AND Immediate (zero-extended)
    fn exec_andi(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_reg = d.rt as u32;
        self.core.write_gpr(rt_reg, rs_val & d.immi64());
        EXEC_COMPLETE
    }

    // ORI - OR Immediate (zero-extended)
    fn exec_ori(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_reg = d.rt as u32;
        self.core.write_gpr(rt_reg, rs_val | d.immi64());
        EXEC_COMPLETE
    }

    // XORI - XOR Immediate (zero-extended)
    fn exec_xori(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rs_val = self.core.read_gpr(d.rs as u32);
        let rt_reg = d.rt as u32;
        self.core.write_gpr(rt_reg, rs_val ^ d.immi64());
        EXEC_COMPLETE
    }

    // LUI - Load Upper Immediate (pre-shifted and sign-extended at decode)
    fn exec_lui(&mut self, d: &DecodedInstr) -> ExecStatus {
        self.core.write_gpr(d.rt as u32, d.immu64());
        EXEC_COMPLETE
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
                EXEC_COMPLETE
            }
            Err(status) => status
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
                EXEC_COMPLETE
            }
            Err(status) => status
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
                EXEC_COMPLETE
            }
            Err(status) => status
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
                EXEC_COMPLETE
            }
            Err(status) => status
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
                EXEC_COMPLETE
            }
            Err(status) => status
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
                EXEC_COMPLETE
            }
            Err(status) => status
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
                EXEC_COMPLETE
            }
            Err(status) => status
        }
    }

    // SB - Store Byte
    fn exec_sb(&mut self, d: &DecodedInstr) -> ExecStatus {
        let base = self.core.read_gpr(d.rs as u32);
        let virt_addr = base.wrapping_add(d.immu64());
        let rt_val = self.core.read_gpr(d.rt as u32);

        self.write_data::<1>(virt_addr, rt_val)
    }

    // SH - Store Halfword
    fn exec_sh(&mut self, d: &DecodedInstr) -> ExecStatus {
        let base = self.core.read_gpr(d.rs as u32);
        let virt_addr = base.wrapping_add(d.immu64());
        let rt_val = self.core.read_gpr(d.rt as u32);

        self.write_data::<2>(virt_addr, rt_val)
    }

    // SW - Store Word
    fn exec_sw(&mut self, d: &DecodedInstr) -> ExecStatus {
        let base = self.core.read_gpr(d.rs as u32);
        let virt_addr = base.wrapping_add(d.immu64());
        let rt_val = self.core.read_gpr(d.rt as u32);

        self.write_data::<4>(virt_addr, rt_val)
    }

    // SD - Store Doubleword (MIPS III, 64-bit)
    fn exec_sd(&mut self, d: &DecodedInstr) -> ExecStatus {
        let base = self.core.read_gpr(d.rs as u32);
        let virt_addr = base.wrapping_add(d.immu64());
        let rt_val = self.core.read_gpr(d.rt as u32);

        self.write_data::<8>(virt_addr, rt_val)
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
                EXEC_COMPLETE
            }
            Err(status) => status
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
                EXEC_COMPLETE
            }
            Err(status) => status
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
        self.write_data64_masked(aligned8, (word_val as u64) << dw_shift, (word_mask as u64) << dw_shift)
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
        self.write_data64_masked(aligned8, (word_val as u64) << dw_shift, (word_mask as u64) << dw_shift)
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
                EXEC_COMPLETE
            }
            Err(status) => status
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
                EXEC_COMPLETE
            }
            Err(status) => status
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

        self.write_data64_masked(aligned_addr, value, mask)
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

        self.write_data64_masked(aligned_addr, value, mask)
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
            let tr = (self.translate_fn)(self, virt_addr, AccessType::Read);
            if tr.is_exception() { return tr.status; }
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

        EXEC_COMPLETE
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
                let tr = (self.translate_fn)(self, virt_addr, AccessType::Read);
                if !tr.is_exception() {
                    let lladdr = (tr.phys >> 4) as u32;
                    self.cache.set_lladdr(lladdr);
                    self.core.cp0_lladdr = lladdr;
                    self.cache.set_llbit(true);
                }
                EXEC_COMPLETE
            }
            Err(status) => status
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
            return EXEC_COMPLETE;
        }

        // Check if address matches the LL address
        let tr = (self.translate_fn)(self, virt_addr, AccessType::Write);
        if tr.is_exception() {
            self.cache.set_llbit(false);
            return tr.status;
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
            status
        } else {
            self.core.write_gpr(rt_reg, 0);
            self.cache.set_llbit(false);
            EXEC_COMPLETE
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
                let tr = (self.translate_fn)(self, virt_addr, AccessType::Read);
                if !tr.is_exception() {
                    let lladdr = (tr.phys >> 4) as u32;
                    self.cache.set_lladdr(lladdr);
                    self.core.cp0_lladdr = lladdr;
                    self.cache.set_llbit(true);
                }
                EXEC_COMPLETE
            }
            Err(status) => status
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
            return EXEC_COMPLETE;
        }

        // Check if address matches the LLD address
        let tr = (self.translate_fn)(self, virt_addr, AccessType::Write);
        if tr.is_exception() {
            self.cache.set_llbit(false);
            return tr.status;
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
            status
        } else {
            // Store failed (address mismatch), set rt to 0 and clear LLBit
            self.core.write_gpr(rt_reg, 0);
            self.cache.set_llbit(false);
            EXEC_COMPLETE
        }
    }

    // PREF - Prefetch
    fn exec_pref(&mut self, _d: &DecodedInstr) -> ExecStatus {
        // Prefetch is a hint and can be implemented as a NOP
        // In a real implementation, this might trigger cache line fetches
        // For now, we just complete without doing anything
        EXEC_COMPLETE
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
                self.reserved_instruction(d)
            }
            RS_BC0 => {
                // BC0 (Branch on CP0 condition) is not used on R4000
                self.reserved_instruction(d)
            }
            _ => {
                self.reserved_instruction(d)
            }
        }
    }

    // MFC0 - Move From CP0
    fn exec_mfc0(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rt_reg = d.rt as u32;
        let rd_val = d.rd as u32;
        // CP0 Random (reg 1) derives from cycle count — flush local counter first
        if rd_val == 1 { self.flush_cycles(); }
        let value = self.core.read_cp0(rd_val);
        // Sign-extend 32-bit value to 64 bits
        self.core.write_gpr(rt_reg, value as u32 as i32 as i64 as u64);
        EXEC_COMPLETE
    }

    // DMFC0 - Doubleword Move From CP0 (MIPS III)
    fn exec_dmfc0(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rt_reg = d.rt as u32;
        let rd_val = d.rd as u32;
        if rd_val == 1 { self.flush_cycles(); }
        let value = self.core.read_cp0(rd_val);
        self.core.write_gpr(rt_reg, value);
        EXEC_COMPLETE
    }

    // MTC0 - Move To CP0
    fn exec_mtc0(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rt_val = self.core.read_gpr(d.rt as u32);
        let rd_val = d.rd as u32;
        // Sign-extend from 32 bits
        self.core.write_cp0(rd_val, rt_val as u32 as i32 as i64 as u64);
        EXEC_COMPLETE
    }

    // DMTC0 - Doubleword Move To CP0 (MIPS III)
    fn exec_dmtc0(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rt_val = self.core.read_gpr(d.rt as u32);
        let rd_val = d.rd as u32;
        self.core.write_cp0(rd_val, rt_val);
        EXEC_COMPLETE
    }

    // TLB Instructions
    fn exec_tlb(&mut self, d: &DecodedInstr) -> ExecStatus {
        let funct_val = d.funct as u32;

        match funct_val {
            FUNCT_TLBR => self.exec_tlbr(),
            FUNCT_TLBWI => self.exec_tlbwi(),
            FUNCT_TLBWR => self.exec_tlbwr(),
            FUNCT_TLBP => self.exec_tlbp(),
            FUNCT_ERET => self.exec_eret(),
            FUNCT_WAIT => EXEC_COMPLETE, // phi opcode: invalid but not RI on R4000 (NOP)
            _ => {
                self.reserved_instruction(d)
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
        self.core.cp0_entrylo0 = (entry.entry_lo0 & !1) | g_bit; // Set G bit from EntryHi
        self.core.cp0_entrylo1 = (entry.entry_lo1 & !1) | g_bit; // Set G bit from EntryHi
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
            entry_lo0: self.core.cp0_entrylo0,
            entry_lo1: self.core.cp0_entrylo1,
        }
    }

    // TLBWI - Write Indexed TLB Entry
    // Writes CP0.EntryHi, CP0.EntryLo0, CP0.EntryLo1, and CP0.PageMask to the TLB entry indexed by CP0.Index
    fn exec_tlbwi(&mut self) -> ExecStatus {
        let index = (self.core.cp0_index as usize) % self.tlb.num_entries();
        let entry = self.create_tlb_entry_from_cp0();
        //eprintln!("TLBWI idx={} entryhi={:#018x} lo0={:#018x} lo1={:#018x} pc={:#018x}", index, entry.entry_hi, entry.entry_lo0, entry.entry_lo1, self.core.pc);
        self.tlb.write(index, entry);
        self.core.nanotlb_invalidate();

        if mips_log(MIPS_LOG_TLB) { dlog_dev!(LogModule::Mips, "TLBWI: Write Index {}\n{}", index, self.tlb.format_entry(index)); }

        EXEC_COMPLETE
    }

    // TLBWR - Write Random TLB Entry
    // Writes CP0.EntryHi, CP0.EntryLo0, CP0.EntryLo1, and CP0.PageMask to a random TLB entry
    // The random index is determined by CP0.Random register
    fn exec_tlbwr(&mut self) -> ExecStatus {
        // Flush local cycle counter so update_random sees accurate cycle count
        self.flush_cycles();
        self.core.update_random();
        let index = (self.core.cp0_random as usize) % self.tlb.num_entries();
        let entry = self.create_tlb_entry_from_cp0();
        self.tlb.write(index, entry);
        self.core.nanotlb_invalidate();

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
        self.core.nanotlb_invalidate();

        // ERET jumps immediately without delay slot
        self.core.pc = target;

        // Return EXEC_COMPLETE_NO_INC since we've already set PC
        EXEC_COMPLETE_NO_INC
    }

    // ===== COP1 (FPU) Instructions =====

    /// Set Cause.CE to the given coprocessor number and return EXC_CPU.
    #[inline]
    fn cpu_unusable(&mut self, ce: u32) -> ExecStatus {
        self.core.cp0_cause = (self.core.cp0_cause & !CAUSE_CE_MASK) | ((ce & 3) << CAUSE_CE_SHIFT);
        exec_exception(EXC_CPU)
    }

    /// After a FPU arithmetic op: read host exception flags, update FCSR cause+flag bits,
    /// and raise EXC_FPE if any enabled exception fired, otherwise EXEC_COMPLETE.
    /// Must be called after the result is written; host FP flags are cleared by this call.
    #[inline]
    fn fpu_update_fcsr(&mut self) -> ExecStatus {
        let flags = crate::platform::get_fpu_status(); // bits [6:2]: FV,FZ,FO,FU,FI
        crate::platform::clear_fpu_status();
        if flags == 0 {
            return EXEC_COMPLETE;
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
            return exec_exception(EXC_FPE);
        }
        // Raise FPE if any cause bit has its corresponding enable bit set
        // Causes are 5 bits above enables: (causes >> 5) aligns them with enables
        if ((causes >> 5) & (self.core.fpu_fcsr & FCSR_EM)) != 0 {
            return exec_exception(EXC_FPE);
        }
        EXEC_COMPLETE
    }

    // MFC1 - Move Word From FPU
    fn exec_mfc1(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let rt_reg = d.rt as u32;
        let fs_reg = d.rd as u32;
        let value = (self.fpr_read_w)(&self.core, fs_reg) as i32 as i64 as u64;
        self.core.write_gpr(rt_reg, value);
        EXEC_COMPLETE
    }

    // DMFC1 - Move Doubleword From FPU (MIPS III)
    fn exec_dmfc1(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let rt_reg = d.rt as u32;
        let fs_reg = d.rd as u32;
        let value = (self.fpr_read_l)(&self.core, fs_reg);
        self.core.write_gpr(rt_reg, value);
        EXEC_COMPLETE
    }

    // CFC1 - Move Control Word From FPU
    fn exec_cfc1(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let rt_reg = d.rt as u32;
        let fs_reg = d.rd as u32;
        let value = self.core.read_fpu_control(fs_reg);
        self.core.write_gpr(rt_reg, value as i32 as i64 as u64);
        EXEC_COMPLETE
    }

    // MTC1 - Move Word To FPU
    fn exec_mtc1(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let rt_val = self.core.read_gpr(d.rt as u32) as u32;
        let fs_reg = d.rd as u32;
        (self.fpr_write_w)(&mut self.core, fs_reg, rt_val);
        EXEC_COMPLETE
    }

    // DMTC1 - Move Doubleword To FPU (MIPS III)
    fn exec_dmtc1(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let rt_val = self.core.read_gpr(d.rt as u32);
        let fs_reg = d.rd as u32;
        (self.fpr_write_l)(&mut self.core, fs_reg, rt_val);
        EXEC_COMPLETE
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
                return exec_exception(EXC_FPE);
            }
        }
        EXEC_COMPLETE
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
            EXEC_BRANCH_LIKELY_SKIP
        } else {
            EXEC_COMPLETE
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
        let result = f32::from_bits((self.fpr_read_w)(&self.core, fs_reg)) / f32::from_bits((self.fpr_read_w)(&self.core, ft_reg));
        (self.fpr_write_w)(&mut self.core, fd_reg, (result).to_bits());
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
        (self.fpr_write_w)(&mut self.core, fd_reg, (result).to_bits()); EXEC_COMPLETE
    }
    fn exec_fmov_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        let value = (self.fpr_read_l)(&self.core, fs_reg);
        (self.fpr_write_l)(&mut self.core, fd_reg, value); EXEC_COMPLETE
    }
    fn exec_fneg_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        let result = -f32::from_bits((self.fpr_read_w)(&self.core, fs_reg));
        (self.fpr_write_w)(&mut self.core, fd_reg, (result).to_bits()); EXEC_COMPLETE
    }
    fn exec_fround_l_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let result = f32::from_bits((self.fpr_read_w)(&self.core, fs_reg)).round() as i64;
        (self.fpr_write_l)(&mut self.core, fd_reg, result as u64);
        self.fpu_update_fcsr()
    }
    fn exec_ftrunc_l_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let result = f32::from_bits((self.fpr_read_w)(&self.core, fs_reg)).trunc() as i64;
        (self.fpr_write_l)(&mut self.core, fd_reg, result as u64);
        self.fpu_update_fcsr()
    }
    fn exec_fceil_l_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let result = f32::from_bits((self.fpr_read_w)(&self.core, fs_reg)).ceil() as i64;
        (self.fpr_write_l)(&mut self.core, fd_reg, result as u64);
        self.fpu_update_fcsr()
    }
    fn exec_ffloor_l_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let result = f32::from_bits((self.fpr_read_w)(&self.core, fs_reg)).floor() as i64;
        (self.fpr_write_l)(&mut self.core, fd_reg, result as u64);
        self.fpu_update_fcsr()
    }
    fn exec_fround_w_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let result = f32::from_bits((self.fpr_read_w)(&self.core, fs_reg)).round() as i32;
        (self.fpr_write_w)(&mut self.core, fd_reg, result as u32);
        self.fpu_update_fcsr()
    }
    fn exec_ftrunc_w_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let result = f32::from_bits((self.fpr_read_w)(&self.core, fs_reg)).trunc() as i32;
        (self.fpr_write_w)(&mut self.core, fd_reg, result as u32);
        self.fpu_update_fcsr()
    }
    fn exec_fceil_w_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let result = f32::from_bits((self.fpr_read_w)(&self.core, fs_reg)).ceil() as i32;
        (self.fpr_write_w)(&mut self.core, fd_reg, result as u32);
        self.fpu_update_fcsr()
    }
    fn exec_ffloor_w_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let result = f32::from_bits((self.fpr_read_w)(&self.core, fs_reg)).floor() as i32;
        (self.fpr_write_w)(&mut self.core, fd_reg, result as u32);
        self.fpu_update_fcsr()
    }
    fn exec_fmovcf_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        let cc = (d.raw >> 18) & 0x7;
        let tf = ((d.raw >> 16) & 0x1) != 0;
        if self.core.get_fpu_cc(cc) == tf {
            let val = f32::from_bits((self.fpr_read_w)(&self.core, fs_reg));
            (self.fpr_write_w)(&mut self.core, fd_reg, (val).to_bits());
        }
        EXEC_COMPLETE
    }
    fn exec_fmovz_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let ft_reg = d.rt as u32; let fd_reg = d.sa as u32;
        if self.core.read_gpr(ft_reg) == 0 {
            let val = f32::from_bits((self.fpr_read_w)(&self.core, fs_reg));
            (self.fpr_write_w)(&mut self.core, fd_reg, (val).to_bits());
        }
        EXEC_COMPLETE
    }
    fn exec_fmovn_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let ft_reg = d.rt as u32; let fd_reg = d.sa as u32;
        if self.core.read_gpr(ft_reg) != 0 {
            let val = f32::from_bits((self.fpr_read_w)(&self.core, fs_reg));
            (self.fpr_write_w)(&mut self.core, fd_reg, (val).to_bits());
        }
        EXEC_COMPLETE
    }
    fn exec_frecip_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let result = 1.0 / f32::from_bits((self.fpr_read_w)(&self.core, fs_reg));
        (self.fpr_write_w)(&mut self.core, fd_reg, (result).to_bits());
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
        let result = f32::from_bits((self.fpr_read_w)(&self.core, fs_reg)).round() as i32;
        (self.fpr_write_w)(&mut self.core, fd_reg, result as u32);
        self.fpu_update_fcsr()
    }
    fn exec_fcvt_l_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let result = f32::from_bits((self.fpr_read_w)(&self.core, fs_reg)).round() as i64;
        (self.fpr_write_l)(&mut self.core, fd_reg, result as u64);
        self.fpu_update_fcsr()
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
                return exec_exception(EXC_FPE);
            }
        }
        let cond = self.fpu_compare_s(fs_val, ft_val, funct_val);
        let cc = fd_reg & 0x7;
        self.core.set_fpu_cc(cc, cond);
        EXEC_COMPLETE
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
        let result = read_d(&self.core, fs_reg) / read_d(&self.core, ft_reg);
        write_d(&mut self.core, fd_reg, result);
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
        write_d(&mut self.core, fd_reg, result); EXEC_COMPLETE
    }
    fn exec_fmov_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        let value = (self.fpr_read_l)(&self.core, fs_reg);
        (self.fpr_write_l)(&mut self.core, fd_reg, value); EXEC_COMPLETE
    }
    fn exec_fneg_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        let read_d = self.fpr_read_d; let write_d = self.fpr_write_d;
        let result = -read_d(&self.core, fs_reg);
        write_d(&mut self.core, fd_reg, result); EXEC_COMPLETE
    }
    fn exec_fround_l_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let result = (self.fpr_read_d)(&self.core, fs_reg).round() as i64;
        (self.fpr_write_l)(&mut self.core, fd_reg, result as u64);
        self.fpu_update_fcsr()
    }
    fn exec_ftrunc_l_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let result = (self.fpr_read_d)(&self.core, fs_reg).trunc() as i64;
        (self.fpr_write_l)(&mut self.core, fd_reg, result as u64);
        self.fpu_update_fcsr()
    }
    fn exec_fceil_l_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let result = (self.fpr_read_d)(&self.core, fs_reg).ceil() as i64;
        (self.fpr_write_l)(&mut self.core, fd_reg, result as u64);
        self.fpu_update_fcsr()
    }
    fn exec_ffloor_l_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let result = (self.fpr_read_d)(&self.core, fs_reg).floor() as i64;
        (self.fpr_write_l)(&mut self.core, fd_reg, result as u64);
        self.fpu_update_fcsr()
    }
    fn exec_fround_w_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let result = (self.fpr_read_d)(&self.core, fs_reg).round() as i32;
        (self.fpr_write_w)(&mut self.core, fd_reg, result as u32);
        self.fpu_update_fcsr()
    }
    fn exec_ftrunc_w_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let result = (self.fpr_read_d)(&self.core, fs_reg).trunc() as i32;
        (self.fpr_write_w)(&mut self.core, fd_reg, result as u32);
        self.fpu_update_fcsr()
    }
    fn exec_fceil_w_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let result = (self.fpr_read_d)(&self.core, fs_reg).ceil() as i32;
        (self.fpr_write_w)(&mut self.core, fd_reg, result as u32);
        self.fpu_update_fcsr()
    }
    fn exec_ffloor_w_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let result = (self.fpr_read_d)(&self.core, fs_reg).floor() as i32;
        (self.fpr_write_w)(&mut self.core, fd_reg, result as u32);
        self.fpu_update_fcsr()
    }
    fn exec_fmovcf_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        let cc = (d.raw >> 18) & 0x7;
        let tf = ((d.raw >> 16) & 0x1) != 0;
        if self.core.get_fpu_cc(cc) == tf {
            let read_d = self.fpr_read_d; let write_d = self.fpr_write_d;
            let val = read_d(&self.core, fs_reg);
            write_d(&mut self.core, fd_reg, val);
        }
        EXEC_COMPLETE
    }
    fn exec_fmovz_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let ft_reg = d.rt as u32; let fd_reg = d.sa as u32;
        if self.core.read_gpr(ft_reg) == 0 {
            let read_d = self.fpr_read_d; let write_d = self.fpr_write_d;
            let val = read_d(&self.core, fs_reg);
            write_d(&mut self.core, fd_reg, val);
        }
        EXEC_COMPLETE
    }
    fn exec_fmovn_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let ft_reg = d.rt as u32; let fd_reg = d.sa as u32;
        if self.core.read_gpr(ft_reg) != 0 {
            let read_d = self.fpr_read_d; let write_d = self.fpr_write_d;
            let val = read_d(&self.core, fs_reg);
            write_d(&mut self.core, fd_reg, val);
        }
        EXEC_COMPLETE
    }
    fn exec_frecip_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let read_d = self.fpr_read_d; let write_d = self.fpr_write_d;
        let result = 1.0 / read_d(&self.core, fs_reg);
        write_d(&mut self.core, fd_reg, result);
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
        let result = (self.fpr_read_d)(&self.core, fs_reg).round() as i32;
        (self.fpr_write_w)(&mut self.core, fd_reg, result as u32);
        self.fpu_update_fcsr()
    }
    fn exec_fcvt_l_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        crate::platform::clear_fpu_status();
        let result = (self.fpr_read_d)(&self.core, fs_reg).round() as i64;
        (self.fpr_write_l)(&mut self.core, fd_reg, result as u64);
        self.fpu_update_fcsr()
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
                return exec_exception(EXC_FPE);
            }
        }
        let cond = self.fpu_compare_d(fs_val, ft_val, funct_val);
        let cc = fd_reg & 0x7;
        self.core.set_fpu_cc(cc, cond);
        EXEC_COMPLETE
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
            Ok(val) => { (self.fpr_write_w)(&mut self.core, fd_reg, val as u32); EXEC_COMPLETE }
            Err(status) => status,
        }
    }
    fn exec_ldxc1(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let base = self.core.read_gpr(d.rs as u32);
        let index = self.core.read_gpr(d.rt as u32);
        let addr = base.wrapping_add(index);
        let fd_reg = d.sa as u32;
        match self.read_data::<8>(addr) {
            Ok(val) => { (self.fpr_write_l)(&mut self.core, fd_reg, val); EXEC_COMPLETE }
            Err(status) => status,
        }
    }
    fn exec_swxc1(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let base = self.core.read_gpr(d.rs as u32);
        let index = self.core.read_gpr(d.rt as u32);
        let addr = base.wrapping_add(index);
        let fs_reg = d.rd as u32;
        let val = (self.fpr_read_w)(&self.core, fs_reg) as u64;
        self.write_data::<4>(addr, val)
    }
    fn exec_sdxc1(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let base = self.core.read_gpr(d.rs as u32);
        let index = self.core.read_gpr(d.rt as u32);
        let addr = base.wrapping_add(index);
        let fs_reg = d.rd as u32;
        let val = (self.fpr_read_l)(&self.core, fs_reg);
        self.write_data::<8>(addr, val)
    }
    fn exec_prefx(&mut self, _d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        EXEC_COMPLETE
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
                EXEC_COMPLETE
            }
            Err(exc_status) => exc_status
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
                EXEC_COMPLETE
            }
            Err(exc_status) => exc_status
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
        self.write_data::<4>(addr, value)
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
        self.write_data::<8>(addr, value)
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
            in_delay_slot: self.in_delay_slot,
            delay_slot_target: self.delay_slot_target,
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
        self.core.running = snapshot.running;
        self.core.halted = snapshot.halted;
        self.in_delay_slot = snapshot.in_delay_slot;
        self.delay_slot_target = snapshot.delay_slot_target;
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
    /// Execute the given decoded instruction.
    #[inline(always)]
    pub fn exec_decoded(&mut self, d: &DecodedInstr) -> ExecStatus {
        type Fn<T, C> = fn(&mut MipsExecutor<T, C>, &DecodedInstr) -> ExecStatus;
        let f: Fn<T, C> = unsafe { std::mem::transmute(d.handler) };
        let status = f(self, d);

        // Handle delay slot state machine — fast path for the common case
        if status == EXEC_COMPLETE {
            if self.in_delay_slot {
                self.core.pc = self.delay_slot_target;
                self.in_delay_slot = false;
            } else {
                self.core.pc = self.core.pc.wrapping_add(4);
            }
            return EXEC_COMPLETE;
        }
        match status {
            EXEC_BRANCH_DELAY => {
                if self.in_delay_slot {
                    // Branch in delay slot - unusual but legal
                } else {
                    self.in_delay_slot = true;
                }
                self.core.pc = self.core.pc.wrapping_add(4);
            }
            EXEC_BRANCH_LIKELY_SKIP => {
                self.core.pc = self.core.pc.wrapping_add(8);
            }
            EXEC_COMPLETE_NO_INC => {
                return EXEC_COMPLETE;
            }
            EXEC_RETRY => {}
            EXEC_BREAKPOINT => {}
            s if s & EXEC_IS_EXCEPTION != 0 => {
                return self.handle_exception(s);
            }
            _ => {}
        }
        status
    }

}

/// Decode `raw` into `ins` if not already decoded.
/// Returns `true` if the instruction was already decoded, `false` otherwise.
pub fn decode_into<T: Tlb, C: MipsCache>(ins: &mut DecodedInstr) -> bool {
    if ins.decoded {
        return true;
    }
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
            FUNCT_ADDU    => MipsExecutor::<T,C>::exec_addu,
            FUNCT_SUB     => MipsExecutor::<T,C>::exec_sub,
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
        OP_J      => { ins.set_imm_j(raw); MipsExecutor::<T,C>::exec_j }
        OP_JAL    => { ins.set_imm_j(raw); MipsExecutor::<T,C>::exec_jal }
        OP_BEQ    => { ins.set_imm_se4(raw); MipsExecutor::<T,C>::exec_beq }
        OP_BNE    => { ins.set_imm_se4(raw); MipsExecutor::<T,C>::exec_bne }
        OP_BLEZ   => { ins.set_imm_se4(raw); MipsExecutor::<T,C>::exec_blez }
        OP_BGTZ   => { ins.set_imm_se4(raw); MipsExecutor::<T,C>::exec_bgtz }
        OP_BEQL   => { ins.set_imm_se4(raw); MipsExecutor::<T,C>::exec_beql }
        OP_BNEL   => { ins.set_imm_se4(raw); MipsExecutor::<T,C>::exec_bnel }
        OP_BLEZL  => { ins.set_imm_se4(raw); MipsExecutor::<T,C>::exec_blezl }
        OP_BGTZL  => { ins.set_imm_se4(raw); MipsExecutor::<T,C>::exec_bgtzl }
        OP_ADDI   => { ins.set_imm_se(raw); MipsExecutor::<T,C>::exec_addi }
        OP_ADDIU  => { ins.set_imm_se(raw); MipsExecutor::<T,C>::exec_addiu }
        OP_DADDI  => { ins.set_imm_se(raw); MipsExecutor::<T,C>::exec_daddi }
        OP_DADDIU => { ins.set_imm_se(raw); MipsExecutor::<T,C>::exec_daddiu }
        OP_SLTI   => { ins.set_imm_se(raw); MipsExecutor::<T,C>::exec_slti }
        OP_SLTIU  => { ins.set_imm_se(raw); MipsExecutor::<T,C>::exec_sltiu }
        OP_ANDI   => { ins.set_imm_ze(raw);               MipsExecutor::<T,C>::exec_andi }
        OP_ORI    => { ins.set_imm_ze(raw);               MipsExecutor::<T,C>::exec_ori }
        OP_XORI   => { ins.set_imm_ze(raw);               MipsExecutor::<T,C>::exec_xori }
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
    ins.decoded = true;
    false
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
    cycles: Arc<AtomicU64>,
    pub interrupts: Arc<AtomicU64>,
    pub fasttick_count: Arc<AtomicU64>,
    debug: Arc<AtomicBool>,
    exception_mask: Arc<AtomicU32>,
}

impl<T: Tlb + Send + 'static, C: MipsCache + Send + 'static> MipsCpu<T, C> {
    pub fn new(executor: MipsExecutor<T, C>) -> Self {
        let cycles = executor.core.cycles.clone();
        let interrupts = executor.core.interrupts.clone();
        let fasttick_count = executor.core.fasttick_count.clone();

        let executor_arc = Arc::new(Mutex::new(executor));
        executor_arc.lock().install_status_cb();

        Self {
            executor: executor_arc,
            running: Arc::new(AtomicBool::new(false)),
            thread: Mutex::new(None),
            cycles,
            interrupts,
            fasttick_count,
            debug: Arc::new(AtomicBool::new(false)),
            exception_mask: Arc::new(AtomicU32::new(0)),
        }
    }


    fn run_debug_loop(&self, mut count: Option<usize>, wait: bool, mut writer: Box<dyn Write + Send>) {
        self.stop(); // Ensure stopped before running
        
        self.running.store(true, Ordering::SeqCst);

        let executor = self.executor.clone();
        let running = self.running.clone();
        let debug = self.debug.clone();
        let exception_mask = self.exception_mask.clone();

        let task = move || {
            let mut exec = executor.lock();
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

                // Capture snapshot before executing instruction (for undo)
                #[cfg(feature = "developer")]
                if exec.undo_buffer.is_enabled() {
                    let mut snapshot = exec.create_snapshot();
                    snapshot.memory_writes = Vec::new();
                    exec.undo_buffer.push(snapshot);
                }

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

                // Commit memory writes to the last snapshot after successful execution
                #[cfg(feature = "developer")]
                if exec.undo_buffer.is_enabled() {
                    exec.commit_undo_snapshot();
                }

                // Display executed instruction from traceback (already captured by step())
                let insn_trace = debug.load(Ordering::Relaxed) || mips_log(MIPS_LOG_INSN);
                if insn_trace || count.is_some() {
                    if let Some(entry) = exec.traceback.get_last(1).into_iter().next() {
                        let symbols = exec.symbols.lock();
                        let sym_str = format_pc_symbol(entry.pc, &symbols);
                        let dis = mips_dis::disassemble(entry.instr, entry.pc, Some(&symbols));
                        let info = format!("{:016x}{}: {:08x} {}", entry.pc, sym_str, entry.instr, dis);
                        if insn_trace {
                            dlog_dev!(LogModule::Mips, "{}", info);
                            writeln!(writer, "{}", info).unwrap();
                        }
                        if count.is_some() {
                            writeln!(writer, "Exec: {}", info).unwrap();
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
                    exec.flush_cycles();
                    drop(exec);
                    thread::sleep(Duration::from_millis(1));
                    exec = executor.lock();

                    if !running.load(Ordering::Relaxed) {
                        writeln!(writer, "Interrupted").unwrap();
                        break;
                    }
                }
            }
            exec.flush_cycles();

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
        exec.flush_cycles();
    }

    fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        if let Some(handle) = self.thread.lock().take() {
            let _ = handle.join();
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
        
        self.running.store(true, Ordering::SeqCst);
        let executor = self.executor.clone();
        let running = self.running.clone();

        *self.thread.lock() = Some(thread::Builder::new().name("MIPS-CPU".to_string()).spawn(move || {
            let mut guard = executor.lock();
            // --- perf sampling (comment out to disable) ---
            //let mut last_cycles: u64 = guard.core.cycles.load(Ordering::Relaxed);
            //let mut last_time = std::time::Instant::now();
            // --- end perf sampling ---
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
                // Flush local cycle counter to shared atomic once per batch
                guard.flush_cycles();
                // --- perf sampling (comment out to disable) ---
                //let cycles = guard.core.cycles.load(Ordering::Relaxed);
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
        self.cycles.load(Ordering::Relaxed)
    }

    fn signal(&self, signal: Signal) {
        match signal {
            Signal::Reset(_soft) => {
                self.interrupts.fetch_or(SOFT_RESET_BIT, Ordering::SeqCst);
            }
            Signal::Interrupt(line, active) => {
                // Bypass mutex lock for interrupts to reduce latency
                let mask = 1u64 << (line + 8);
                if active {
                    self.interrupts.fetch_or(mask, Ordering::SeqCst);
                } else {
                    self.interrupts.fetch_and(!mask, Ordering::SeqCst);
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
            ("exception".to_string(), "Control exception breaks: exception <class|code|all> <on|off>".to_string()),
            ("run".to_string(), "Run instructions until exception or breakpoint: run [addr]".to_string()),
            ("step".to_string(), "Step n instructions or until address: step [count|addr]".to_string()),
            ("next".to_string(), "Step over function calls: next [count]".to_string()),
            ("finish".to_string(), "Run until function return (jr ra)".to_string()),
            ("fin".to_string(), "Alias for finish".to_string()),
            ("s".to_string(), "Alias for step".to_string()),
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
            ("debug".to_string(), "Enable/disable CPU tracing: debug <on|off> [DEV]".to_string()),
            ("ex".to_string(), "Alias for exception".to_string()),
            ("undo".to_string(), "Undo N instructions or control undo buffer: undo [count] | undo <on|off|clear> [DEV]".to_string()),
            ("dt".to_string(), "Disassemble traceback: dt [count]".to_string()),
            ("u".to_string(), "Alias for undo [DEV]".to_string()),
            ("sym".to_string(), "Lookup symbol: sym <addr>".to_string()),
            ("loadsym".to_string(), "Load symbols from file: loadsym <file>".to_string()),
            ("l1i".to_string(), "L1 Instruction Cache commands: l1i <check|dump> <addr|index>".to_string()),
            ("l1d".to_string(), "L1 Data Cache commands: l1d <check|dump> <addr|index>".to_string()),
            ("l2".to_string(), "L2 Cache commands: l2 <check|dump> <addr|index>".to_string()),
            ("ll".to_string(), "Show LL/SC state: llbit and lladdr".to_string()),
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
            let exec = self.executor.lock();
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
            let exec = self.executor.lock();
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

        if actual_cmd == "ll" {
            let exec = self.executor.lock();
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
            
            let mut exec = self.executor.lock();
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
            writeln!(out, "PC: {:016x}", exec.core.pc).unwrap();
            writeln!(out, "HI: {:016x} LO: {:016x}", exec.core.hi, exec.core.lo).unwrap();
            for i in 0..32 {
                let val = exec.core.gpr[i];
                write!(out, "{:4}(${:02}): {:016x}  ", mips_dis::reg_name(i as u32), i, val).unwrap();
                if (i + 1) % 4 == 0 { writeln!(out).unwrap(); }
            }
            writeln!(out, "CP0 Status: {:08x} ({})", exec.core.cp0_status, decode_status(exec.core.cp0_status)).unwrap();
            writeln!(out, "CP0 Cause:  {:08x} ({})", exec.core.cp0_cause, decode_cause(exec.core.cp0_cause)).unwrap();
            writeln!(out, "CP0 EPC: {:016x} BadVAddr: {:016x}", exec.core.cp0_epc, exec.core.cp0_badvaddr).unwrap();
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
                let exec = self.executor.lock();
                let pc = exec.core.pc;
                let symbols = exec.symbols.lock();
                let sym_str = format_pc_symbol(pc, &symbols);
                writeln!(writer, "{} pc={:016x}{}", if running { "running" } else { "stopped" }, pc, sym_str).unwrap();
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
            "step" | "s" => {
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
                let mut exec = self.executor.lock();
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
                let mut exec = self.executor.lock();
                writeln!(writer, "COP0 Registers:").unwrap();
                for i in 0..32 {
                    let val = exec.core.read_cp0(i);
                    let name = mips_dis::cp0_reg_name(i);
                    if name != "?" {
                        write!(writer, "  {:2} {:8}: {:016x}", i, name, val).unwrap();
                        if i == 12 { // Status
                            write!(writer, " {}", decode_status(val as u32)).unwrap();
                        } else if i == 13 { // Cause
                            write!(writer, " {}", decode_cause(val as u32)).unwrap();
                        }
                        writeln!(writer).unwrap();
                    }
                }
                Ok(())
            }
            "cop1" => {
                let exec = self.executor.lock();
                writeln!(writer, "COP1 Registers (FPU):").unwrap();
                for i in 0..32 {
                    let val = exec.core.fpr[i];
                    let f32_val = f32::from_bits(val as u32);
                    let f64_val = f64::from_bits(val);
                    writeln!(writer, "  f{:02}: {:016x}  (f32: {:e}, f64: {:e})", i, val, f32_val, f64_val).unwrap();
                }
                writeln!(writer, "Control Registers:").unwrap();
                writeln!(writer, "  FIR:  {:08x}", exec.core.fpu_fir).unwrap();
                writeln!(writer, "  FCCR: {:08x}", exec.core.fpu_fccr).unwrap();
                writeln!(writer, "  FEXR: {:08x}", exec.core.fpu_fexr).unwrap();
                writeln!(writer, "  FENR: {:08x}", exec.core.fpu_fenr).unwrap();
                writeln!(writer, "  FCSR: {:08x}", exec.core.fpu_fcsr).unwrap();
                Ok(())
            }
            "mem" | "m" | "memory" => {
                if actual_args.is_empty() { return Err("Usage: mem <addr> [count]".to_string()); }
                let mut exec = self.executor.lock();
                let symbols_arc = exec.symbols.clone();
                let symbols = symbols_arc.lock();
                let addr = parse_cpu_arg(actual_args[0], &exec.core, Some(&symbols))?;
                let count = if actual_args.len() > 1 { actual_args[1].parse().unwrap_or(1) } else { 1 };
                
                for i in 0..count {
                    let curr_addr = addr.wrapping_add(i * 4);
                    match exec.debug_read(curr_addr, 4) {
                        Ok(val) => writeln!(writer, "{:016x}: {:08x}", curr_addr, val).unwrap(),
                        Err(e) => writeln!(writer, "{:016x}: Error {:?}", curr_addr, e).unwrap(),
                    }
                }
                Ok(())
            }
            "stack" => {
                let mut exec = self.executor.lock();
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
                writeln!(writer, "{}", self.executor.lock().backtrace(20)).unwrap();
                Ok(())
            }
            "mw" => {
                if actual_args.len() < 2 { return Err("Usage: mw <addr> <val> [size: b|h|w|d]".to_string()); }
                let mut exec = self.executor.lock();
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
                let mut exec = self.executor.lock();
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
                let mut exec = self.executor.lock();
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
                            writeln!(writer, "{:016x}{}: {:08x} {}", curr_addr, sym_str, instr, mips_dis::disassemble(instr, curr_addr, Some(&symbols))).unwrap()
                        },
                        Err(_) => writeln!(writer, "{:016x}: Could not fetch", curr_addr).unwrap(),
                    }
                }
                Ok(())
            }
            "jump" => {
                if actual_args.is_empty() { return Err("Usage: jump <addr>".to_string()); }
                let mut exec = self.executor.lock();
                let symbols_arc = exec.symbols.clone();
                let symbols = symbols_arc.lock();
                let addr = parse_cpu_arg(actual_args[0], &exec.core, Some(&symbols))?;
                exec.core.pc = addr;
                writeln!(writer, "PC set to {:016x}", addr).unwrap();
                Ok(())
            }
            "setreg" => {
                if actual_args.len() < 2 { return Err("Usage: setreg <reg> <value>".to_string()); }
                let mut exec = self.executor.lock();
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
                let mut exec = self.executor.lock();
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
                    return Err("Usage: debug <on|off>".to_string());
                }
                let val = match actual_args[0] {
                    "on" | "1" => true,
                    "off" | "0" => false,
                    _ => return Err("Usage: debug <on|off>".to_string()),
                };
                self.debug.store(val, Ordering::Relaxed);
                writeln!(writer, "CPU debug {}", if val { "enabled" } else { "disabled" }).unwrap();
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
                            let mut exec = self.executor.lock();
                            exec.undo_buffer.enable();
                            writeln!(writer, "CPU undo buffer enabled").unwrap();
                            return Ok(());
                        }
                        "off" | "0" if actual_args[0] == "off" || actual_args[0] == "0" => {
                            let mut exec = self.executor.lock();
                            exec.undo_buffer.disable();
                            writeln!(writer, "CPU undo buffer disabled").unwrap();
                            return Ok(());
                        }
                        "clear" => {
                            let mut exec = self.executor.lock();
                            exec.undo_buffer.clear();
                            writeln!(writer, "CPU undo buffer cleared").unwrap();
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

                let mut exec = self.executor.lock();

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
                let exec = self.executor.lock();
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
            "dt" | "traceback" => {
                let count = if !actual_args.is_empty() {
                    actual_args[0].parse().unwrap_or(10)
                } else {
                    10
                };
                let exec = self.executor.lock();
                let symbols = exec.symbols.lock();
                let entries = exec.traceback.get_last(count);
                writeln!(writer, "Execution Traceback (last {} instructions):", entries.len()).unwrap();
                for entry in entries {
                    let sym_str = format_pc_symbol(entry.pc, &symbols);
                    writeln!(writer, "{:016x}{}: {:08x} {}", entry.pc, sym_str, entry.instr, mips_dis::disassemble(entry.instr, entry.pc, Some(&symbols))).unwrap();
                }
                Ok(())
            }
            _ => Err(format!("Unknown CPU command: {}", actual_cmd)),
        }
    }

}

impl<T: Tlb, C: MipsCache> MipsExecutor<T, C> {
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
        exec.in_delay_slot = false;
        exec.delay_slot_target = 0;
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
        cp0u64!(cp0_count); cp0u64!(cp0_compare); cp0u32!(cp0_status); cp0u32!(cp0_cause);
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
        tbl.insert("in_delay_slot".into(),     toml::Value::Boolean(exec.in_delay_slot));
        tbl.insert("delay_slot_target".into(), hex_u64(exec.delay_slot_target));

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
            ld32!(cp0_status); ld32!(cp0_cause); ld32!(cp0_prid);
            ld32!(cp0_config); ld32!(cp0_lladdr); ld32!(cp0_watchlo); ld32!(cp0_watchhi);
            ld32!(cp0_ecc); ld32!(cp0_cacheerr); ld32!(cp0_taglo); ld32!(cp0_taghi);
            ld64!(cp0_entrylo0); ld64!(cp0_entrylo1); ld64!(cp0_context);
            ld64!(cp0_pagemask); ld64!(cp0_badvaddr); ld64!(cp0_entryhi);
            ld64!(cp0_xcontext); ld64!(cp0_epc); ld64!(cp0_errorepc);
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

        if let Some(x) = get_field(v, "in_delay_slot")     { exec.in_delay_slot     = toml_bool(x).unwrap_or(false); }
        if let Some(x) = get_field(v, "delay_slot_target") { exec.delay_slot_target = toml_u64(x).unwrap_or(0); }

        if let Some(tlb_v) = get_field(v, "tlb") {
            exec.tlb.load_state(tlb_v)?;
        }

        if let Some(cache_v) = get_field(v, "cache") {
            exec.cache.load_cache_state(cache_v)?;
        }

        Ok(())
    }
}
