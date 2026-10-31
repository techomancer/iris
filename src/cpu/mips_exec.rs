// MIPS Execution Engine

use crate::cpu::mips_core::*;
use crate::cpu::mips_isa::*;
use crate::traits::*;
use crate::cpu::mips_tlb::*;
use crate::cpu::mips_cache_v2::*;
use crate::devlog::{LogModule, devlog_mask, devlog_is_active};
use crate::cpu::mips_dis;
use crate::ip22_bus::{HIMEM_BASE, HIMEM_END, LOMEM_BASE, LOMEM_END};
use std::fmt::Write as FmtWrite;
use crate::cpu::mips_dis::SymbolTable;
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
pub const MIPS_LOG_CP0:  u32 = 0x0010; // CP0 register traffic: EPC/EntryHi/XContext writes,
                                       // ECC and CacheErr accesses, Status.DE transitions

/// Opt-in gate for the `developerx` Coprocessor-Unusable break (see the three
/// `handle_exception*` wrappers). Off unless `IRIS_BREAK_CPU=1`.
///
/// Must be opt-in: a CpU exception is a perfectly ordinary event — lazy FPU
/// enable raises one on purpose every time a process first touches the FPU —
/// so breaking on all of them unconditionally stops the machine constantly and
/// breaks any test that exercises the CU0/CU1 gates on purpose.
#[cfg(feature = "developerx")]
fn cpu_unusable_break_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("IRIS_BREAK_CPU").map(|v| v == "1").unwrap_or(false))
}

#[cfg(feature = "developer")]
#[inline(always)]
fn mips_log(bit: u32) -> bool {
    devlog_is_active(LogModule::Mips) && (devlog_mask(LogModule::Mips) & bit) != 0
}

/// Like `mips_log`, but live in every build.
///
/// The IP28 bring-up traces this gates were plain environment variables that
/// worked in release, and the machines they diagnose are booted in release —
/// so they keep that reach. The cost is two relaxed atomic loads, against the
/// `getenv` per call some of them used to do.
#[inline(always)]
fn mips_log_always(bit: u32) -> bool {
    devlog_is_active(LogModule::Mips) && (devlog_mask(LogModule::Mips) & bit) != 0
}

// Without `developer`, dlog_dev! is a no-op, so callers gate on this constant `false`
// instead of paying an atomic load per call site to check a flag that can never fire.
#[cfg(not(feature = "developer"))]
#[inline(always)]
fn mips_log(_bit: u32) -> bool {
    false
}

/// `jitv2_smc_check`: total data writes seen landing on the page currently
/// being fetched from, counted *before* the `(pc, pfn)` dedup — so this is the
/// real event count, not the number of distinct reports printed. Exposed so a
/// run can confirm the check fired at all: a zero after a workload that should
/// have tripped it means the instrumentation is broken (or every store took
/// the inline fast path, which bypasses `write_data_impl` entirely — see
/// `smc_check_report`'s doc comment), not that the guest is well-behaved.
#[cfg(all(feature = "jitv2", feature = "jitv2_smc_check"))]
pub static SMC_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `j2 pagewb [on|off]` gate (default off): when on, `jitv2_track_pcp` writes
/// back dirty L1-D/L2 lines covering a physical code page the instant a fetch
/// crosses onto it, before that page can be handed to the JIT for
/// compilation — see `jitv2_track_pcp`'s own doc comment for why. Process-
/// global (mirrors `jitv2::analyzer::FALLBACK_ENABLED`'s style) rather than a
/// per-executor field: it's a diagnostic knob toggled from the monitor
/// console, not per-core state.
#[cfg(feature = "jitv2")]
static JITV2_PAGEWB_ENABLED: AtomicBool = AtomicBool::new(false);

#[cfg(feature = "jitv2")]
#[inline]
fn jitv2_pagewb_enabled() -> bool {
    JITV2_PAGEWB_ENABLED.load(Ordering::Relaxed)
}

#[cfg(feature = "jitv2")]
pub fn set_jitv2_pagewb_enabled(on: bool) {
    JITV2_PAGEWB_ENABLED.store(on, Ordering::Relaxed);
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
const FCSR_FV: u32 = 0x00000040;  // Flag: invalid operation (sticky)
const FCSR_FZ: u32 = 0x00000020;  // Flag: divide-by-zero (sticky)
const FCSR_FO: u32 = 0x00000010;  // Flag: overflow (sticky)
const FCSR_FU: u32 = 0x00000008;  // Flag: underflow (sticky)
const FCSR_FI: u32 = 0x00000004;  // Flag: inexact (sticky)
const FCSR_UM: u32 = 0x00000100;  // Enable: underflow
const FCSR_IM: u32 = 0x00000080;  // Enable: inexact
const FCSR_FS: u32 = 0x01000000;  // Flush denormalized results to zero

/// True if `bits` (an IEEE-754 single-precision pattern) is a signalling NaN:
/// exponent all-1s, mantissa nonzero, and the mantissa's MSB (the "is quiet"
/// bit) clear. A quiet NaN has that bit set.
#[inline]
fn is_snan_s(bits: u32) -> bool {
    (bits & 0x7F80_0000) == 0x7F80_0000 && (bits & 0x007F_FFFF) != 0 && (bits & 0x0040_0000) == 0
}

/// Negate by flipping the sign bit, which is what IEEE-754 negation *is*.
///
/// These exist because `-x.mul_add(y, z)` could not be trusted, back when the
/// multiply-add family was computed with a fused `mul_add` (it no longer is:
/// see `exec_madd_s`). LLVM rewrites `-fma(a, b, -c)` into `fma(-a, b, c)` —
/// algebraically identical, and wrong for signed zeros. It made NMSUB.D(+0, +0, +0) return `+0.0` where MIPS IV
/// requires `-((fs*ft) - fr)` = `-(+0.0)` = `-0.0`, and it did so only in
/// NMSUB: MSUB, computing the very same inner FMA without the outer negation,
/// returned `+0.0` correctly. Two "identical" expressions disagreeing is the
/// tell.
///
/// Found 2026-09-22 by the jitv2 equivalence harness, which flagged a
/// divergence against a new `fneg`-based JIT emitter that was doing the right
/// thing. A bit flip is not reassociable, so it stays correct.
///
/// Only the *outer* negation needs this. Negating an operand (`-fr_val`
/// feeding the addend) is a plain unary neg on a value LLVM cannot fold
/// through an FMA, and is sign-correct as written.
/// `black_box` is load-bearing, not decoration. A plain
/// `f64::from_bits(v.to_bits() ^ SIGN)` does **not** survive: LLVM's
/// InstCombine recognises that idiom as `fneg` and then re-applies the very
/// reassociation this exists to prevent. Measured — the bit-twiddling version
/// alone still returned `+0.0` for NMSUB.D(+0, +0, +0). The barrier stops
/// LLVM tracing the value back to the FMA that produced it.
#[inline]
fn fneg_bits_s(v: f32) -> u32 {
    std::hint::black_box(v).to_bits() ^ 0x8000_0000
}

#[inline]
fn fneg_d_exact(v: f64) -> f64 {
    f64::from_bits(std::hint::black_box(v).to_bits() ^ 0x8000_0000_0000_0000)
}

/// Same as `is_snan_s`, for IEEE-754 double-precision bit patterns.
#[inline]
fn is_snan_d(bits: u64) -> bool {
    (bits & 0x7FF0_0000_0000_0000) == 0x7FF0_0000_0000_0000
        && (bits & 0x000F_FFFF_FFFF_FFFF) != 0
        && (bits & 0x0008_0000_0000_0000) == 0
}

/// True if `bits` is a quiet NaN (any NaN that is not a signalling NaN).
#[inline]
fn is_qnan_s(bits: u32) -> bool {
    f32::from_bits(bits).is_nan() && !is_snan_s(bits)
}

/// Same as `is_qnan_s`, for double precision.
#[inline]
fn is_qnan_d(bits: u64) -> bool {
    f64::from_bits(bits).is_nan() && !is_snan_d(bits)
}

/// Every ADD/SUB/MUL/DIV/SQRT/RECIP/RSQRT/MADD-family handler's exception
/// checks are computed from operand/result bit patterns alone — never from
/// the host FPU's own exception flags (see
/// `clanker/jitv2/fpu-flags-are-computed-not-read.md` for why: Cranelift
/// `call_indirect` and even plain LTO/codegen-units=1 optimization can't be
/// trusted to preserve host FPU status-register state across the boundary
/// between "clear it" and "read it back"). `is_snan_s/d` here catches what
/// `fpu_check_denorm_operand_s/d`'s qNaN check does not — a signalling NaN
/// operand, which today falls through to computation with no Invalid
/// detection at all.
#[inline]
fn is_neg_nonzero_s(bits: u32) -> bool {
    (bits & 0x8000_0000) != 0 && (bits & 0x7FFF_FFFF) != 0 && !is_snan_s(bits) && !is_qnan_s(bits)
}
#[inline]
fn is_neg_nonzero_d(bits: u64) -> bool {
    (bits & 0x8000_0000_0000_0000) != 0 && (bits & 0x7FFF_FFFF_FFFF_FFFF) != 0 && !is_snan_d(bits) && !is_qnan_d(bits)
}
#[inline]
fn is_pos_zero_or_neg_zero_s(bits: u32) -> bool { (bits & 0x7FFF_FFFF) == 0 }
#[inline]
fn is_pos_zero_or_neg_zero_d(bits: u64) -> bool { (bits & 0x7FFF_FFFF_FFFF_FFFF) == 0 }

pub const EXC_WATCH: u32 = 23;    // Reference to WatchHi/WatchLo address
pub const EXC_VCEI: u32 = 14;     // Virtual Coherency Exception (Instruction)
pub const EXC_VCED: u32 = 31;     // Virtual Coherency Exception (Data)

/// XContext as hardware writes it on a TLB miss, for a CPU implementing
/// `va_bits` virtual address bits.
///
/// `docs/R4000_um2.pdf`, Figure 5-10 and Table 5-8, gives the layout for a
/// 40-bit CPU:
///
/// ```text
///  63              33 32 31 30            4 3     0
///        PTEBase         R      BadVPN2         0
///          31            2         27           4
/// ```
///
/// with the accompanying text: "The 27-bit BadVPN2 field has bits 39:13 of the
/// virtual address that caused the TLB miss; bit 12 is excluded because a
/// single TLB entry maps to an even-odd page pair." R is virtual address bits
/// 63:62, and PTEBase is written by the operating system.
///
/// Every field position follows from that one sentence. BadVPN2 is
/// `VA[va_bits-1:13]`, so it is `va_bits - 13` bits wide — 27 when va_bits is
/// 40, which is what the figure shows. It starts at bit 4, so it ends at
/// `va_bits - 10` (30 for 40 bits, per the figure). R is the two bits directly
/// above it, `[va_bits-8:va_bits-9]` (32:31 — the figure). PTEBase is
/// everything above that, `[63:va_bits-7]` (63:33 — the figure). The 40-bit
/// case therefore reproduces the manual exactly, and the general case is the
/// same arithmetic with the address width left as a parameter, which is how
/// the manual itself states it.
///
/// The R10000 implements 44 virtual address bits rather than 40, so on that
/// CPU all three fields sit four bits higher. Using the 40-bit layout there
/// puts R and PTEBase four bits too low: a 64-bit kernel reads XContext in its
/// TLB refill handler to find the page-table entry for the faulting address,
/// so the address it computes lands outside the page table it wired, the
/// nested miss is unrecoverable, and IRIX panics with "Entered tlbmiss with
/// invalid vaddr". That is the observable difference on a CPU that
/// implements more than 40: it boots with the right width and panics with
/// the wrong one.
#[inline]
pub fn xcontext_on_miss(va_bits: u32, old_xcontext: u64, virt_addr: u64) -> u64 {
    // Gaps from the field edges down to the address width, read off the
    // figure: PTEBase starts at 33 = 40 - 7, R at 31 = 40 - 9.
    const XC_PTEBASE_GAP: u32 = 7;
    const XC_REGION_GAP: u32 = 9;

    let vpn2_mask: u64 = (((1u64 << (va_bits - 13)) - 1) << 13) & !0xFFFu64;
    let ptebase = old_xcontext & (!0u64 << (va_bits - XC_PTEBASE_GAP));
    let badvpn2 = ((virt_addr & vpn2_mask) >> 13) << 4;
    let region = (virt_addr >> 62) & 0x3;
    ptebase | (region << (va_bits - XC_REGION_GAP)) | badvpn2
}

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

mod instrument;
mod monitor;
mod gdb;
pub use instrument::*;
pub use gdb::*;
use monitor::*;

// Externally-raised interrupt mask (IP7..IP2): the IP bits that other
// threads deliver through `hot.interrupts` and the step() preamble mirrors
// into Cause. IP7 (CP0 Count==Compare) is included since the compare timer
// fires on the hptimer thread and raises it exactly like a device line;
// writing Compare clears the pending bit again (mips_core.rs write_cp0).
const EXT_INT_MASK: u32 = crate::cpu::mips_core::CAUSE_IP7 |
                          crate::cpu::mips_core::CAUSE_IP6 |
                          crate::cpu::mips_core::CAUSE_IP5 |
                          crate::cpu::mips_core::CAUSE_IP4 |
                          crate::cpu::mips_core::CAUSE_IP3 |
                          crate::cpu::mips_core::CAUSE_IP2;

// Bit 63 of the interrupts word = soft-reset request
const SOFT_RESET_BIT: u64 = 1u64 << 63;

/// CP0 `Index` slot field, bits [5:0] — the only bits that select a TLB entry.
///
/// Every use of `cp0_index` as an entry number must mask with this rather than
/// take it modulo the entry count: `%` folds the probe-failure bit (31) and the
/// out-of-range values 48..63 back onto *valid* slots, silently corrupting live
/// (and typically wired) entries. See `exec_tlbwi`.
pub const CP0_INDEX_SLOT_MASK: u32 = 0x3F;

/// CP0 `Index` probe-failure bit, set by `TLBP` when no entry matches.
pub const CP0_INDEX_P: u32 = 0x8000_0000;


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
/// A Cache Error exception: ErrorEPC/ERL and the uncached vector, not the
/// common one. CacheErr comes from the cache (`MipsCache::take_cache_err`).
pub const EXEC_IS_CACHE_ERR:       ExecStatus = 1 << 30; // 0x4000_0000
pub const EXEC_CACHE_ERR: ExecStatus = EXEC_IS_EXCEPTION | EXEC_IS_CACHE_ERR;

// Bus error ExecStatus values — also exported as BUS_ERR / BUS_VCE in traits.rs.
// The values MUST stay identical; enforced by compile-time asserts in traits.rs.
pub const EXEC_BUS_ERR: ExecStatus = exec_exception_const(EXC_DBE);  // 0x0800_001C
pub const EXEC_BUS_VCE: ExecStatus = exec_exception_const(EXC_VCED); // 0x0800_007C

/// `const`-evaluable version of exec_exception (for use in const initializers).
#[inline(always)]
pub const fn exec_exception_const(code: u32) -> ExecStatus {
    EXEC_IS_EXCEPTION | (code << crate::cpu::mips_core::CAUSE_EXCCODE_SHIFT)
}

/// Build an exception ExecStatus from an EXC_* code.
#[inline(always)]
pub fn exec_exception(code: u32) -> ExecStatus {
    exec_exception_const(code)
}

/// Build a TLB-refill ExecStatus from an EXC_* code (32-bit UTLB vector).
#[inline(always)]
pub fn exec_tlb_miss(code: u32) -> ExecStatus {
    EXEC_IS_EXCEPTION | EXEC_IS_TLB_REFILL | (code << crate::cpu::mips_core::CAUSE_EXCCODE_SHIFT)
}

/// Build an XTLB-refill ExecStatus from an EXC_* code (64-bit XTLB vector, offset 0x080).
#[inline(always)]
pub fn exec_xtlb_miss(code: u32) -> ExecStatus {
    EXEC_IS_EXCEPTION | EXEC_IS_TLB_REFILL | EXEC_IS_XTLB_REFILL | (code << crate::cpu::mips_core::CAUSE_EXCCODE_SHIFT)
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

/// **Contract:** these two flags are set only on the `translate_fn` slow
/// path, and are read only by `nutlb_fill`. A nutlb *hit* returns them clear
/// regardless of the underlying page, because a hit never fills and nothing
/// else consumes them. Do not add a reader without fixing that.
///
/// Set on a successful translation that came from a **TLB-mapped** segment.
/// Clear for the unmapped windows (kseg0/kseg1/xkphys and the ERL identity
/// case), which are ASID-independent by construction. Only the nutlb fill
/// path reads it — see `nutlb_fill`.
pub const TR_MAPPED: u32 = 1 << 3;
/// Set on a TLB-mapped translation whose entry had the **G** (global) bit,
/// meaning the mapping is valid for every ASID. Meaningless unless
/// `TR_MAPPED` is also set.
pub const TR_GLOBAL: u32 = 1 << 4;

/// Which privilege modes may reach `va`, as nutlb tag bits (10:8).
///
/// **Mirrors the segment decode in `translate_32bit_impl` /
/// `translate_64bit_impl` and must stay in step with them.** Returns 0 for any
/// segment shape we decline to cache, which makes `nutlb_fill` skip the write.
///
/// The bits are a downward-inclusive *permission set*, so a segment reachable
/// from user mode also sets the supervisor and kernel bits. That is what makes
/// the probe's `(virttag & cur_sec_mask) != 0` test correct for kernel code
/// touching user pages (`copyin`/`copyout`), which is the case a naive
/// Should a translation for `va` be cached in the nutlb at all?
///
/// Privilege is *not* consulted: nutlb entries carry no permission bits, and
/// a mode switch flushes the whole structure, so an entry can never outlive
/// the privilege epoch that filled it. The only thing rejected here is a VA
/// shape that is not a valid address in the first place.
#[inline]
fn nutlb_cacheable_segment(va: u64, is_64bit: bool) -> bool {
    if !is_64bit {
        // Every 32-bit segment (kuseg/kseg0/kseg1/ksseg/kseg3) is a real
        // address whose finished translation is worth caching.
        return true;
    }
    // xuseg: bits 63:40 must be zero to be a valid user address. Every other
    // 64-bit segment (xsseg, xkphys, xkseg and the 0xFFFFFFFF_* compat
    // windows) is fine — translate_64bit_impl raises before returning ok for
    // the illegal xkphys sub-ranges, so nothing invalid reaches a fill.
    (va >> 62) != 0 || (va >> 40) == 0
}

/// Per-access-class tally of how far a guest load/store would get through the
/// Per-access-class tally of how far a guest load/store would get through the
/// checks the JIT's inline memory path emits, before falling back to Rust.
///
/// Each counter is a *prefix* of the next: an access counted in `tag_hit` also
/// passed `nutlb_hit`. That makes the drop between adjacent rows the cost of
/// that specific stage, and `inline_ok / total` the joint rate — the number
/// that says how much of the workload the inline emit actually captures.
#[cfg(feature = "jitstats")]
#[derive(Default, Clone, Copy)]
pub struct JitMemStats {
    /// Every cached data access reaching the fast-path probe.
    pub total: u64,
    /// Translation was served by the nutlb (no `translate_fn` call).
    pub nutlb_hit: u64,
    /// ...and the L1D tag matched the physical line.
    pub tag_hit: u64,
    /// ...and (stores only) that line was already dirty.
    pub dirty_ok: u64,
    /// ...and (tcache only) the region is transparent/mapped RAM.
    /// The full inline predicate: this is what compiled code would handle
    /// without calling out.
    pub inline_ok: u64,
}

#[cfg(feature = "jitstats")]
#[derive(Default, Clone, Copy)]
pub struct JitMemStatsPair {
    pub load:  JitMemStats,
    pub store: JitMemStats,
}

#[cfg(feature = "jitstats")]
impl JitMemStatsPair {
    pub fn print(&self) {
        fn pct(n: u64, d: u64) -> f64 { if d == 0 { 0.0 } else { n as f64 / d as f64 * 100.0 } }
        eprintln!("\n=== JIT inline memory-path coverage ===");
        eprintln!("  (each stage is a prefix of the next; the drop between rows is that stage's cost)");
        for (name, s) in [("load", &self.load), ("store", &self.store)] {
            if s.total == 0 { continue; }
            eprintln!("  [{name}]  accesses={t}", name = name, t = s.total);
            eprintln!("        nutlb hit : {v:>12}  {p:5.1}%", v = s.nutlb_hit, p = pct(s.nutlb_hit, s.total));
            eprintln!("        + tag hit : {v:>12}  {p:5.1}%", v = s.tag_hit,   p = pct(s.tag_hit,   s.total));
            if name == "store" {
                eprintln!("        + dirty   : {v:>12}  {p:5.1}%", v = s.dirty_ok, p = pct(s.dirty_ok, s.total));
            }
            eprintln!("        = INLINE  : {v:>12}  {p:5.1}%   <- emitted path handles this share",
                v = s.inline_ok, p = pct(s.inline_ok, s.total));
            eprintln!("          callout : {v:>12}  {p:5.1}%",
                v = s.total - s.inline_ok, p = pct(s.total - s.inline_ok, s.total));
        }
        eprintln!("=== End JIT inline memory-path coverage ===\n");
    }
}

/// Result of address translation: 8 bytes, no heap.
///
/// Layout when success (EXEC_IS_EXCEPTION clear in `status`):
///   `phys`   — physical address (up to 40 bits on the R10000)
///   `status` — bits [2:0]: C-field cache attr (TR_UNCACHED/TR_CACHEABLE/TR_CACHEABLE_COH)
///              bit  [3]:   TR_MAPPED — translation came from the TLB
///              bit  [4]:   TR_GLOBAL — that TLB entry was global (G=1)
///              all other bits 0
///
/// Layout when exception (EXEC_IS_EXCEPTION set in `status`):
///   `phys`   — 0 (ignored)
///   `status` — fully-formed ExecStatus for handle_exception
#[derive(Debug, Clone, Copy)]
pub struct TranslateResult {
    pub phys:   u64,
    pub status: u32,
}

impl TranslateResult {
    #[inline(always)]
    pub fn ok(phys: u64, c_field: u32) -> Self {
        Self { phys, status: c_field }
    }
    /// Success from a TLB-mapped segment, carrying the entry's G bit so the
    /// nutlb can decide whether to tag the fill with an ASID.
    #[inline(always)]
    pub fn ok_mapped(phys: u64, c_field: u32, global: bool) -> Self {
        Self {
            phys,
            status: c_field | TR_MAPPED | if global { TR_GLOBAL } else { 0 },
        }
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

    /// The JTLB size the CPU model declares.
    ///
    /// `core.tlb_entries` is already taken from the model, so sizing the TLB
    /// itself from anything else leaves the two disagreeing: Random cycles
    /// over a range the array does not have, and a TLBWI to an index past the
    /// end is silently dropped. The R10000 has 64 entries where the R4400 has
    /// 48, and SGI's IP28 diagnostic writes index 48 on its first cache-alias
    /// test — every one of its reported failures was that write going nowhere.
    pub fn for_model<C: crate::cpu::mips_cache_v2::CpuModel>() -> Self {
        Self { tlb_entries: C::TLB_ENTRIES }
    }
}

/// MIPS Execution Engine - combines CPU core with memory interface and TLB
/// Slot count for the `ll stats` table. Direct-mapped on the low bits of
/// `LLAddr`, so this is a plain power of two; 1024 rows is far more than the
/// handful of kernel lock addresses that matter and still only ~40 KB.
#[cfg(feature = "llstats")]
pub const LL_STATS_SLOTS: usize = 1024;

/// Direct-mapped lookup into the `ll stats` table.
///
/// No hashing and no allocation — this runs on every LL and every SC, in
/// builds that include `lightning`. On a collision (two `LLAddr`s landing on
/// one slot) the slot is reset and re-keyed to the newcomer, so a row always
/// describes exactly one address rather than silently summing two. Collisions
/// therefore lose history but never fabricate it, which is the right trade for
/// a diagnostic whose whole job is "is THIS address stuck".
#[cfg(feature = "llstats")]
#[inline]
fn ll_stats_slot(tbl: &mut [LlSiteStats; LL_STATS_SLOTS], key: u64) -> &mut LlSiteStats {
    let slot = (key as usize) & (LL_STATS_SLOTS - 1);
    let e = &mut tbl[slot];
    if e.key != key {
        *e = LlSiteStats { key, ..Default::default() };
    }
    e
}

/// Read-only slot lookup for tests/inspection: `None` when this address has no
/// row (never seen, or evicted by a collision).
#[cfg(feature = "llstats")]
pub fn ll_stats_lookup(tbl: &[LlSiteStats; LL_STATS_SLOTS], key: u64) -> Option<&LlSiteStats> {
    let e = &tbl[(key as usize) & (LL_STATS_SLOTS - 1)];
    if e.key == key { Some(e) } else { None }
}

/// One `ll stats` row — see `MipsExecutor::ll_stats`.
#[cfg(feature = "llstats")]
#[derive(Default, Clone, Copy)]
pub struct LlSiteStats {
    /// `LLAddr` (phys >> 4) this row describes; `u64::MAX` when the slot has
    /// never been used. Stored because the table is direct-mapped and two
    /// addresses can collide onto one slot — on a collision the slot is reset
    /// and re-keyed, so a row always describes exactly one address.
    pub key: u64,
    /// `LL` executed here, taking a reservation.
    pub ll: u64,
    /// `SC` here found the LLbit set and stored.
    pub ok: u64,
    /// `SC` here found the LLbit already clear (an intervening exception,
    /// ERET, or coherent invalidation) and failed.
    pub fail: u64,
    /// Consecutive failures since the last success — the live run length.
    pub cur_run: u64,
    /// Longest run of consecutive failures ever seen for this address. This
    /// is the interesting column: a loop that retries a couple of times is
    /// normal, one stuck in the hundreds is not.
    pub max_run: u64,
}

pub struct MipsExecutor<T: Tlb, C: CpuModel> {
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
    /// Per-instruction execution trace recorder (clanker/jitv2's lockstep
    /// verification tooling, `src/cpu/trace.rs`). `None` when not recording —
    /// checked on every step() (one branch on a `None` tag), so armed cost
    /// is a file write, disarmed cost is a predictable branch. Developer-only:
    /// this captures full architectural state per instruction, meaningfully
    /// intrusive on the hot path, matching undo_buffer/idle_profiler's own
    /// dev-only gating.
    #[cfg(feature = "developer")]
    trace_writer: Option<crate::cpu::trace::TraceWriter>,
    /// `ll stats`: per-LL-address retry histogram.
    ///
    /// Keyed by the physical address an `LL` reserved (its `LLAddr` value,
    /// i.e. phys >> 4). For each one: how many times the reservation has been
    /// established, how many SCs on it succeeded, how many failed, and the
    /// worst run of consecutive failures seen before one finally succeeded.
    ///
    /// Exists to answer a specific question that static reading cannot: is a
    /// kernel LL/SC loop *failing to make progress*? IRIX builds every atomic
    /// on LL/SC loops (`atomic_ops.h`), including the `mutex_bitlock` on
    /// `k_flags` that `kthread.h` says guards "all locking in the sync
    /// routines" — so an SC that can never land is exactly the shape of
    /// "processes sleep forever while everything else runs". A high
    /// `max_consecutive_fail` on one address names the stuck lock; an
    /// unbounded `fail` with zero `ok` names a loop that never completes.
    ///
    /// Developer-only: a HashMap probe per LL/SC is far too heavy for the
    /// production interpreter path.
    #[cfg(feature = "llstats")]
    pub ll_stats: Box<[LlSiteStats; LL_STATS_SLOTS]>,
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
    /// More about the last hit for the stop note: the access address of a
    /// memory breakpoint, or why its condition could not be evaluated.
    pub bp_hit_detail: Option<String>,
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
    pub instr_stats: crate::cpu::mips_instr_stats::InstrStats,
    /// JIT v2 engine state: physical-code-page pool + allocator (clanker/jitv2/jit-v2-design.md §2.4).
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
    /// `compile_queue.start/stop` (pool lifecycle) is never called
    /// concurrently from both sides by contract — fine for those infrequent
    /// management calls. The one genuinely hot per-dispatch operation,
    /// pushing a compile request, does *not* go through this lock — see
    /// `jitv2_compile_queue`/`jitv2_stats` below.
    #[cfg(feature = "jitv2")]
    pub jitv2: std::sync::Arc<Mutex<crate::cpu::jitv2::Jitv2>>,
    /// Set when other CPUs share `jitv2`: a flush parks them first.
    #[cfg(feature = "jitv2")]
    pub(crate) jitv2_world: Option<Arc<crate::cpu::smp::World>>,
    /// Lock-free `pfn -> PageSlot` lookup: a raw pointer to the `PfnMap`
    /// array inside `jitv2`, captured once by `jitv2_bind_fast_lookup`.
    #[cfg(feature = "jitv2")]
    jitv2_pfn_map: *const AtomicU32,
    /// Companion to `jitv2_pfn_map`: the pool's `pages` array base, so a
    /// lookup hit can produce the `*mut PhysicalCodePage` the caller wants
    /// without going back through `Jitv2`.
    #[cfg(feature = "jitv2")]
    jitv2_pages_base: *mut crate::cpu::jitv2::PhysicalCodePage,
    /// Cheap handle to `jitv2.lock().compile_queue`'s underlying push queue,
    /// cloned once at construction (`CompileQueue::queue_handle`) — lets the
    /// per-dispatch compile-request send (`exec_decoded`'s JIT gate) skip
    /// `jitv2.lock()` entirely. Same `Arc<ArrayQueue<CompileRequest>>` the
    /// worker threads pop from, so this is exactly as sound as the clones
    /// `CompileQueue::start_inner` already hands each worker; it's just one
    /// more clone, held here instead of re-derived through the mutex on
    /// every dispatch.
    #[cfg(feature = "jitv2")]
    pub jitv2_compile_queue_handle: std::sync::Arc<crossbeam_queue::ArrayQueue<crate::cpu::jitv2::CompileRequest>>,
    /// Cheap handle to `jitv2.lock().stats`, cloned once at construction —
    /// same reasoning as `jitv2_compile_queue_handle`: `JitStats` is already
    /// `Arc`-shared internally (see `Jitv2::stats`'s own doc comment), so
    /// there is no reason a per-dispatch send should reacquire `jitv2.lock()`
    /// just to read this field.
    #[cfg(feature = "jitv2")]
    pub jitv2_stats: std::sync::Arc<crate::cpu::jitv2::JitStats>,

    /// Coverage of the JIT's inline load/store path (feature = "jitstats").
    /// Populated on every cached data access; printed at exit.
    #[cfg(feature = "jitstats")]
    pub jit_mem_stats: JitMemStatsPair,
    /// Did the translation for the access currently in flight come from the
    /// nutlb/nanotlb fast path? Set by `nutlb_translate`/`nanotlb_translate`,
    /// consumed by the `jit_probe` call in `read_data`/`write_data` — the
    /// cache layer cannot see translation outcomes, so it is threaded here.
    #[cfg(feature = "jitstats")]
    pub jit_last_xlat_hit: bool,
    /// Pointer to the `PhysicalCodePage` for the page the fetch-side nanotlb last
    /// resolved to. Updated only on a page change (§2.1 — physical page, not VA);
    /// null until the first fetch translation after construction/reset. Owned and
    /// written exclusively by this thread (§6.1.3 executor-thread ownership) —
    /// valid until the next `jitv2.mega_flush()`, which is why it's re-derived
    /// lazily rather than cached across a flush.
    #[cfg(feature = "jitv2")]
    pub(crate) pcp: *mut crate::cpu::jitv2::PhysicalCodePage,
    /// `jitv2_smc_check` dedup set, keyed `(pc, target code pfn)` — see
    /// [`Self::smc_check_report`].
    #[cfg(all(feature = "jitv2", feature = "jitv2_smc_check"))]
    smc_seen: std::collections::HashSet<(u64, u32)>,
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
    /// Rolling history of the last `LS_HISTORY_CAP` inline-lockstep-bracketed
    /// instructions: for each, its address/raw word and the machine state
    /// *entering* it (i.e. the final state of the previous step). Populated by
    /// `lockstep_step` before the interpreter reference runs, so it survives a
    /// divergence bail into the monitor — where `j2 lstate` prints it. The
    /// last entry's `before` is the state right before the diverging
    /// instruction (`lockstep_compare` only keeps `ls_before`'s pc/slot
    /// fields, not the full register file, so without this the "what went in"
    /// context is lost the moment the report prints).
    #[cfg(feature = "jitv2_lockstep")]
    pub(crate) ls_history: std::collections::VecDeque<LockstepHistEntry>,
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
    pub(crate) jitv2_inline_analyzer: crate::cpu::jitv2::analyzer::Analyzer,
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
    /// Called by the CPU thread between batches of instructions (about every
    /// thousand), with the executor locked: for a machine that has to act on
    /// where the guest is running (the O2 hands its cache over to the
    /// shadow model once a kernel starts, and back when POST starts over).
    /// None costs one check per batch.
    pub batch_hook: Option<Box<dyn FnMut(&mut MipsExecutor<T, C>) + Send>>,
    /// A window of physical addresses that reaches memory at another address,
    /// `(base, size, target)`, applied to mapped and XKPHYS translations
    /// before [`Self::phys_mask`]. The O2 has a no-ECC alias of its memory at
    /// 0x8000_0000 that would otherwise reach the PCI window the bus decodes
    /// there. None on the other machines.
    pub phys_alias: Option<(u64, u64, u64)>,
    /// The physical address bits the board decodes, applied to mapped,
    /// XKPHYS and ERL-identity translations after [`Self::phys_alias`]. The
    /// CPU drives 36 or 40 bits, but the Indy, Indigo2 and O2 decode 32 and
    /// alias the rest away, and IRIX relies on it: its page-table entries keep
    /// software bits (the PFN lock among them) just above the PFN, so they
    /// reach EntryLo and come out as physical address bits 32 and up. All
    /// ones (no fold) by default; each machine sets its own.
    pub phys_mask: u64,
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
/// (see clanker/jitv2/ for the investigation trail). Extend this list as more
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
fn translate_32_kernel<T: Tlb, C: CpuModel>(e: &mut MipsExecutor<T,C>, va: u64, at: AccessType) -> TranslateResult {
    e.translate_32bit_impl::<false, {crate::cpu::mips_core::PRIV_KERNEL}>(va, at)
}
fn translate_32_supervisor<T: Tlb, C: CpuModel>(e: &mut MipsExecutor<T,C>, va: u64, at: AccessType) -> TranslateResult {
    e.translate_32bit_impl::<false, {crate::cpu::mips_core::PRIV_SUPERVISOR}>(va, at)
}
fn translate_32_user<T: Tlb, C: CpuModel>(e: &mut MipsExecutor<T,C>, va: u64, at: AccessType) -> TranslateResult {
    e.translate_32bit_impl::<false, {crate::cpu::mips_core::PRIV_USER}>(va, at)
}
fn translate_64_kernel<T: Tlb, C: CpuModel>(e: &mut MipsExecutor<T,C>, va: u64, at: AccessType) -> TranslateResult {
    e.translate_64bit_impl::<false, {crate::cpu::mips_core::PRIV_KERNEL}>(va, at)
}
fn translate_64_supervisor<T: Tlb, C: CpuModel>(e: &mut MipsExecutor<T,C>, va: u64, at: AccessType) -> TranslateResult {
    e.translate_64bit_impl::<false, {crate::cpu::mips_core::PRIV_SUPERVISOR}>(va, at)
}
fn translate_64_user<T: Tlb, C: CpuModel>(e: &mut MipsExecutor<T,C>, va: u64, at: AccessType) -> TranslateResult {
    e.translate_64bit_impl::<false, {crate::cpu::mips_core::PRIV_USER}>(va, at)
}

/// Free-standing trampoline for the CP0 Status callback installed by `install_status_cb`.
/// Rust does not allow `Self` inside a nested fn, so the generic trampoline lives here.
// Safety: the executor's raw pointers point into allocations owned by
// MipsCore which outlive the executor. The executor is only accessed from the CPU thread.
unsafe impl<T: Tlb, C: CpuModel> Send for MipsExecutor<T, C> {}
unsafe impl<T: Tlb, C: CpuModel> Sync for MipsExecutor<T, C> {}

fn mips_executor_status_cb<T: Tlb, C: CpuModel>(ctx: *mut core::ffi::c_void, _old: u32, _new: u32) {
    // SAFETY: ctx is `&mut MipsExecutor<T,C>` cast to void, alive for the executor's lifetime,
    // and only ever called from the CPU thread that exclusively owns the executor.
    //
    // The (old, new) pair is `MipsCore::status_changed_cb`'s declared signature; the resync
    // is unconditional, so neither is read. See `resync_privilege_state`.
    let exec = unsafe { &mut *(ctx as *mut MipsExecutor<T, C>) };
    exec.resync_privilege_state();
}

// ---- JIT v2 memory-access / exception-delivery trampolines ------
// Free functions (one monomorphized instantiation per <T,C>, like the
// translate_fn wrappers above) so their addresses can be stored as bare
// `unsafe extern "C" fn` pointers in MipsCore (`install_jit_hooks`). Every
// read returns its status alongside the value (see `JitReadResult`);
// (see MipsCore's read*_fn field doc comments).
//
// `ctx` is the `*mut MipsCore` the `JitFn` was itself entered with — the
// *same* pointer for every hook, so compiled code passes its own arg 0
// straight through with no load of its own (see `exec_from_core` below and
// `codegen::emit_mem_read_callout` et al). Hooks needing the surrounding
// executor recover it with `exec_from_core`, the Rust equivalent of Linux's
// `container_of`.

/// Turn a JIT callout's arg 0 into the real `*mut MipsCore`.
///
/// **Every JIT->Rust hook must funnel its `ctx` through this function (or
/// through [`exec_from_core`], which calls it) before dereferencing it.**
/// Compiled code does not pass a usable pointer: `codegen::callout_core_arg`
/// adds [`CALLOUT_CORE_BIAS`] first, and this subtracts it back. Using `ctx`
/// raw reads `MipsCore` at a `CALLOUT_CORE_BIAS`-byte offset — every field
/// lands on the wrong bytes, silently, with no fault.
///
/// # Why compiled code biases the pointer at all
///
/// Arg 0 carries a `FixedReg(%rdi)` constraint at the call, while the same
/// core pointer is also the region's base register (`0x50(%r14)`, …) and is
/// live across that call. Handing regalloc one SSA value for both roles puts
/// a single long live range under a fixed-register conflict at *every*
/// callout, and regalloc2 splits a value only `MAX_SPLITS_PER_SPILLSET` (2)
/// times before giving up and calling `split_into_minimal_bundles` — a spill
/// plus a reload at every use. Measured on one corpus page: 2152 reloads into
/// `%rdi`, +1500 `movq`, +4.5 KiB of code.
///
/// A biased pointer is a distinct value that is born at the call and dies
/// there, so it never joins the base pointer's spillset and never reaches
/// that cliff. Emitted arg 0 becomes a single `leaq 4(%r14), %rdi` with the
/// base pointer staying resident. A `+0` copy does *not* work — the
/// `opt_level=speed` optimizer folds it back into the same value (verified:
/// byte-identical output).
///
/// See `clanker/jitv2/callout-arg0-is-core-ptr.md` for the full measurements.
///
/// # Safety
/// `ctx` must be arg 0 as produced by `codegen::callout_core_arg` — a real,
/// live `MipsCore` address plus exactly `CALLOUT_CORE_BIAS`.
#[cfg(feature = "jitv2")]
#[inline(always)]
pub unsafe fn core_from_arg(ctx: *mut core::ffi::c_void) -> *mut MipsCore {
    debug_assert!(
        ctx as usize > crate::cpu::jitv2::codegen::CALLOUT_CORE_BIAS as usize,
        "JIT callout arg 0 must be a biased MipsCore pointer, not null/garbage \
         (see mips_exec::core_from_arg)"
    );
    unsafe {
        ctx.cast::<u8>()
            .sub(crate::cpu::jitv2::codegen::CALLOUT_CORE_BIAS as usize)
            .cast::<MipsCore>()
    }
}

/// Recover the containing `MipsExecutor<T, C>` from a JIT callout's arg 0 —
/// [`core_from_arg`] followed by `container_of(core, MipsExecutor<T,C>, core)`.
///
/// Every JIT->Rust hook takes the same arg 0 — the `*mut MipsCore` compiled
/// code was entered with, biased — rather than a separately-stashed executor
/// pointer: one calling convention for all of them, one fewer load per
/// callout in emitted code, and no way left to pass the wrong one of two
/// type-erased pointers (a mistake that was silent — it corrupted memory at
/// `self + 2*offsetof(core)` rather than faulting; see the history in
/// `codegen::emit_fpu_cvt_to_int_call`).
///
/// `offset_of!` asks the compiler for the layout it actually chose, so this
/// stays correct under `repr(Rust)` field reordering — it is not assuming
/// `core` sits at offset 0 (it does today, so that part compiles to nothing).
///
/// # Safety
/// `ctx` must satisfy [`core_from_arg`]'s contract, and the `MipsCore` it
/// names must be the `core` field of a live, exclusively-owned
/// `MipsExecutor<T, C>` — i.e. must have come from a `jit_fn(&mut exec.core)`
/// dispatch whose hooks were installed by that same executor's
/// `install_jit_hooks`. Deriving the container this way (rather than reading
/// an address captured at install time) is what makes a *moved* executor
/// merely correct instead of a use-after-free.
#[cfg(feature = "jitv2")]
#[inline(always)]
unsafe fn exec_from_core<T: Tlb, C: CpuModel>(ctx: *mut core::ffi::c_void) -> *mut MipsExecutor<T, C> {
    unsafe {
        core_from_arg(ctx)
            .cast::<u8>()
            .sub(std::mem::offset_of!(MipsExecutor<T, C>, core))
            .cast::<MipsExecutor<T, C>>()
    }
}

#[cfg(feature = "jitv2")]
unsafe extern "C" fn jit_read8<T: Tlb, C: CpuModel>(ctx: *mut core::ffi::c_void, va: u64, dst: *mut u64) -> u32 {
    let exec = unsafe { &mut *exec_from_core::<T, C>(ctx) };
    #[cfg(feature = "jitv2_lockstep")]
    { let r = exec.lockstep_jit_read::<1>(va); unsafe { *dst = r.val; } return r.status; }
    #[cfg(not(feature = "jitv2_lockstep"))]
    match exec.read_data::<1>(va) {
        Ok(v) => { unsafe { *dst = v; } EXEC_COMPLETE }
        Err(status) => status,
    }
}
#[cfg(feature = "jitv2")]
unsafe extern "C" fn jit_read8_sext<T: Tlb, C: CpuModel>(ctx: *mut core::ffi::c_void, va: u64, dst: *mut u64) -> u32 {
    let exec = unsafe { &mut *exec_from_core::<T, C>(ctx) };
    #[cfg(feature = "jitv2_lockstep")]
    { let r = exec.lockstep_jit_read::<1>(va); unsafe { *dst = r.val as u8 as i8 as i64 as u64; } return r.status; }
    #[cfg(not(feature = "jitv2_lockstep"))]
    match exec.read_data::<1>(va) {
        Ok(v) => { unsafe { *dst = v as u8 as i8 as i64 as u64; } EXEC_COMPLETE }
        Err(status) => status,
    }
}
#[cfg(feature = "jitv2")]
unsafe extern "C" fn jit_read16<T: Tlb, C: CpuModel>(ctx: *mut core::ffi::c_void, va: u64, dst: *mut u64) -> u32 {
    let exec = unsafe { &mut *exec_from_core::<T, C>(ctx) };
    #[cfg(feature = "jitv2_lockstep")]
    { let r = exec.lockstep_jit_read::<2>(va); unsafe { *dst = r.val; } return r.status; }
    #[cfg(not(feature = "jitv2_lockstep"))]
    match exec.read_data::<2>(va) {
        Ok(v) => { unsafe { *dst = v; } EXEC_COMPLETE }
        Err(status) => status,
    }
}
#[cfg(feature = "jitv2")]
unsafe extern "C" fn jit_read16_sext<T: Tlb, C: CpuModel>(ctx: *mut core::ffi::c_void, va: u64, dst: *mut u64) -> u32 {
    let exec = unsafe { &mut *exec_from_core::<T, C>(ctx) };
    #[cfg(feature = "jitv2_lockstep")]
    { let r = exec.lockstep_jit_read::<2>(va); unsafe { *dst = r.val as u16 as i16 as i64 as u64; } return r.status; }
    #[cfg(not(feature = "jitv2_lockstep"))]
    match exec.read_data::<2>(va) {
        Ok(v) => { unsafe { *dst = v as u16 as i16 as i64 as u64; } EXEC_COMPLETE }
        Err(status) => status,
    }
}
#[cfg(feature = "jitv2")]
unsafe extern "C" fn jit_read32<T: Tlb, C: CpuModel>(ctx: *mut core::ffi::c_void, va: u64, dst: *mut u64) -> u32 {
    let exec = unsafe { &mut *exec_from_core::<T, C>(ctx) };
    #[cfg(feature = "jitv2_lockstep")]
    { let r = exec.lockstep_jit_read::<4>(va); unsafe { *dst = r.val; } return r.status; }
    #[cfg(not(feature = "jitv2_lockstep"))]
    match exec.read_data::<4>(va) {
        Ok(v) => { unsafe { *dst = v; } EXEC_COMPLETE }
        Err(status) => status,
    }
}
#[cfg(feature = "jitv2")]
unsafe extern "C" fn jit_read32_sext<T: Tlb, C: CpuModel>(ctx: *mut core::ffi::c_void, va: u64, dst: *mut u64) -> u32 {
    let exec = unsafe { &mut *exec_from_core::<T, C>(ctx) };
    #[cfg(feature = "jitv2_lockstep")]
    { let r = exec.lockstep_jit_read::<4>(va); unsafe { *dst = r.val as u32 as i32 as i64 as u64; } return r.status; }
    #[cfg(not(feature = "jitv2_lockstep"))]
    match exec.read_data::<4>(va) {
        Ok(v) => { unsafe { *dst = v as u32 as i32 as i64 as u64; } EXEC_COMPLETE }
        Err(status) => status,
    }
}
#[cfg(feature = "jitv2")]
unsafe extern "C" fn jit_read64<T: Tlb, C: CpuModel>(ctx: *mut core::ffi::c_void, va: u64, dst: *mut u64) -> u32 {
    let exec = unsafe { &mut *exec_from_core::<T, C>(ctx) };
    #[cfg(feature = "jitv2_lockstep")]
    { let r = exec.lockstep_jit_read::<8>(va); unsafe { *dst = r.val; } return r.status; }
    #[cfg(not(feature = "jitv2_lockstep"))]
    match exec.read_data::<8>(va) {
        Ok(v) => { unsafe { *dst = v; } EXEC_COMPLETE }
        Err(status) => status,
    }
}
#[cfg(feature = "jitv2")]
unsafe extern "C" fn jit_write8<T: Tlb, C: CpuModel>(ctx: *mut core::ffi::c_void, va: u64, val: u64) -> u32 {
    let exec = unsafe { &mut *exec_from_core::<T, C>(ctx) };
    let val = val as u8; // mask to the real width ourselves — see write8_fn's doc comment on why the FFI parameter is u64
    #[cfg(feature = "jitv2_lockstep")]
    { return exec.lockstep_jit_write::<1>(va, val as u64); }
    #[cfg(not(feature = "jitv2_lockstep"))]
    exec.write_data::<1>(va, val as u64)
}
#[cfg(feature = "jitv2")]
unsafe extern "C" fn jit_write16<T: Tlb, C: CpuModel>(ctx: *mut core::ffi::c_void, va: u64, val: u64) -> u32 {
    let exec = unsafe { &mut *exec_from_core::<T, C>(ctx) };
    let val = val as u16; // mask to the real width ourselves — see write16_fn's doc comment
    #[cfg(feature = "jitv2_lockstep")]
    { return exec.lockstep_jit_write::<2>(va, val as u64); }
    #[cfg(not(feature = "jitv2_lockstep"))]
    exec.write_data::<2>(va, val as u64)
}
#[cfg(feature = "jitv2")]
unsafe extern "C" fn jit_write32<T: Tlb, C: CpuModel>(ctx: *mut core::ffi::c_void, va: u64, val: u64) -> u32 {
    let exec = unsafe { &mut *exec_from_core::<T, C>(ctx) };
    let val = val as u32; // mask to the real width ourselves — see write32_fn's doc comment
    #[cfg(feature = "jitv2_lockstep")]
    { return exec.lockstep_jit_write::<4>(va, val as u64); }
    #[cfg(not(feature = "jitv2_lockstep"))]
    exec.write_data::<4>(va, val as u64)
}
#[cfg(feature = "jitv2")]
unsafe extern "C" fn jit_write64<T: Tlb, C: CpuModel>(ctx: *mut core::ffi::c_void, va: u64, val: u64) -> u32 {
    let exec = unsafe { &mut *exec_from_core::<T, C>(ctx) };
    #[cfg(feature = "jitv2_lockstep")]
    { return exec.lockstep_jit_write::<8>(va, val); }
    #[cfg(not(feature = "jitv2_lockstep"))]
    exec.write_data::<8>(va, val)
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
unsafe extern "C" fn jit_write64_masked<T: Tlb, C: CpuModel>(ctx: *mut core::ffi::c_void, va: u64, val: u64, mask: u64) -> u32 {
    let exec = unsafe { &mut *exec_from_core::<T, C>(ctx) };
    exec.write_data64_masked(va, val, mask)
}
/// Single-implementation exception delivery (§4.2): calls the interpreter's
/// own `handle_exception` — the only place EPC/Cause/BD/vectoring are ever
/// computed, for both engines.
#[cfg(feature = "jitv2")]
unsafe extern "C" fn jit_handle_exception<T: Tlb, C: CpuModel>(ctx: *mut core::ffi::c_void, status: u32) -> u32 {
    let exec = unsafe { &mut *exec_from_core::<T, C>(ctx) };
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

/// [`jit_handle_exception`] with the EPC/BD inputs passed explicitly — see
/// `MipsCore::handle_exception_at_fn` and
/// `mips_core::deliver_exception_at` for why. Unlike its sibling, this does
/// NOT read `core.pc`/`core.in_delay_slot`: compiled code supplies both, so
/// neither has to be live in memory at the fault site.
#[cfg(feature = "jitv2")]
unsafe extern "C" fn jit_handle_exception_at<T: Tlb, C: CpuModel>(ctx: *mut core::ffi::c_void, status: u32, fault_pc: u64, bd: u8) -> u32 {
    let exec = unsafe { &mut *exec_from_core::<T, C>(ctx) };
    exec.handle_exception_at(status, fault_pc, bd != 0)
}

/// Force one instruction's worth of real forward progress through the
/// interpreter, bypassing the JIT dispatch gate — see
/// `MipsCore::interp_fallback_fn`'s doc comment for why this needs to exist
/// at all (a plain JIT bail can't force a fallback: `exec_decoded`'s caller
/// can't tell "please retry me through the interpreter" apart from a real
/// retirement, since both return `EXEC_COMPLETE`).
#[cfg(feature = "jitv2")]
unsafe extern "C" fn jit_interp_fallback<T: Tlb, C: CpuModel>(ctx: *mut core::ffi::c_void) -> u32 {
    let exec = unsafe { &mut *exec_from_core::<T, C>(ctx) };
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
unsafe extern "C" fn jit_kill_entry<T: Tlb, C: CpuModel>(ctx: *mut core::ffi::c_void, offset: u32) {
    let exec = unsafe { &mut *exec_from_core::<T, C>(ctx) };
    assert!(!exec.pcp.is_null(), "jit_kill_entry reached with no tracked PhysicalCodePage");
    let page = unsafe { &*exec.pcp };
    page.kill(offset as usize);
    // Record the FR mode this dispatch actually wanted, so the next compile
    // for this page re-pins to it instead of rebuilding the same wrong-mode
    // code and guard-failing again. `emit_fr_mode_guard` is the only emitter
    // that calls kill_entry, and it only fires when CU1 is set and live FR
    // disagrees with the compiled mode — so live FR here IS the demanded
    // mode. Storing the demanded mode rather than a "flip" bit keeps this
    // idempotent across the several bails a page normally takes before its
    // next compile — one per killed entry (see `fr_repin`'s doc comment).
    page.request_fr_repin((exec.core.cp0_status & crate::cpu::mips_core::STATUS_FR) != 0);
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
/// Count of `fetch_verify` checks performed, so `ll`-style "is this actually
/// verifying?" questions have an answer. A stuck 0 means compiled code never
/// ran with the hook installed.
#[cfg(feature = "fetchverify")]
pub static FETCH_VERIFY_CHECKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Count of mismatches found — i.e. compiled code that was about to execute an
/// instruction word memory no longer holds.
#[cfg(feature = "fetchverify")]
pub static FETCH_VERIFY_MISMATCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);


/// `fetchverify`: honour `IRIS_NO_FETCH_VERIFY=1` once, at executor
/// construction, by clearing the per-core flag. Env-only; there is no runtime
/// global (see `MipsCore::fetch_verify_on` for why).
#[cfg(feature = "fetchverify")]
pub fn fetch_verify_env_default() -> bool {
    static ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENV.get_or_init(|| !std::env::var("IRIS_NO_FETCH_VERIFY").map(|v| v == "1").unwrap_or(false))
}

/// `fetchverify`: re-read the instruction word at `va` and compare it against
/// `expected`, the constant the compiler baked in when it compiled this
/// instruction. See `MipsCore::fetch_verify_fn`.
///
/// Reads through `debug_translate` + the cache's debug fetch path so the check
/// itself perturbs nothing — no cache fills, no LLbit clears, no exceptions of
/// its own. A VA that no longer translates is *not* a mismatch (the mapping
/// legitimately changed, and the next real fetch will fault on its own), so it
/// is skipped rather than reported.
#[cfg(all(feature = "jitv2", feature = "fetchverify"))]
unsafe extern "C" fn jit_fetch_verify<T: Tlb, C: CpuModel>(ctx: *mut core::ffi::c_void, va: u64, expected: u32) -> u32 {
    let exec = unsafe { &mut *exec_from_core::<T, C>(ctx) };
    if !exec.core.fetch_verify_on {
        return EXEC_COMPLETE;
    }
    FETCH_VERIFY_CHECKS.fetch_add(1, Ordering::Relaxed);

    let tr = exec.debug_translate(va);
    if tr.is_exception() {
        return EXEC_COMPLETE;
    }
    let r = exec.sysad.read32(tr.phys);
    if !r.is_ok() {
        return EXEC_COMPLETE;
    }
    if r.data != expected {
        FETCH_VERIFY_MISMATCHES.fetch_add(1, Ordering::Relaxed);
        let symbols = exec.symbols.lock();
        let sym = format_pc_symbol(va, &symbols);
        // Report live `core.pc` and the tracked page alongside the VA. The VA
        // is *derived* — `emit_word_addr` computes it as
        // `(core.pc & !0xFFF) + word*4` — so a region dispatched with `core.pc`
        // on a different page than its snapshot came from reports a wrong
        // address while `expected` is right, which looks like a coherent
        // "shift" but is really two unrelated regions. `tracked_pfn` vs
        // `phys_page` is what tells those apart.
        let pcp_pfn: i64 = if exec.pcp.is_null() { -1 } else { unsafe { (*exec.pcp).pfn as i64 } };
        eprintln!(
            "\n=== STALE COMPILED CODE at {:#018x}{} phys={:#010x} ===\n  compiled-in: {:08x}  {}\n  now in memory: {:08x}  {}\n  live pc={:#018x}  va_page={:#x}  phys_page={:#x}  tracked_pfn={}\n",
            va, sym, tr.phys,
            expected, mips_dis::disassemble(expected, va, Some(&symbols)),
            r.data, mips_dis::disassemble(r.data, va, Some(&symbols)),
            exec.core.pc, va >> 12, tr.phys >> 12, pcp_pfn);
        drop(symbols);
        exec.core.pc = va;
        return EXEC_BREAKPOINT;
    }
    EXEC_COMPLETE
}

#[cfg(all(feature = "jitv2", feature = "developer"))]
unsafe extern "C" fn jit_dev_trace_bp<T: Tlb, C: CpuModel>(ctx: *mut core::ffi::c_void, pc: u64, raw: u32, origin: u32) -> u32 {
    DEV_TRACE_BP_CALLS.fetch_add(1, Ordering::Relaxed);
    let exec = unsafe { &mut *exec_from_core::<T, C>(ctx) };
    #[cfg(not(feature = "lightning"))]
    {
        exec.traceback.push(pc, raw, InstrOrigin::from_u32(origin));
        if let Some(w) = exec.trace_writer.as_mut() {
            let record = crate::cpu::trace::TraceRecord::capture(pc, raw, &exec.core);
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

/// FPU rounding-mode hook: not generic over `<T,C>` (no executor context
/// needed — wraps `platform::set_fpu_mode` directly). `ctx` is accepted for
/// signature uniformity with the other hooks but unused.
#[cfg(feature = "jitv2")]
unsafe extern "C" fn jit_fpu_set_mode(_ctx: *mut core::ffi::c_void, rm: u32) {
    crate::platform::set_fpu_mode(rm as u8)
}

/// Single-implementation CVT.W/L.S/D (§4.2 shape, same as `jit_handle_exception`):
/// `ctx` is the `MipsExecutor<T, C>` pointer, so this reads/writes
/// `exec.core` directly — the JIT hands over `fs_reg`/`fd_reg`/format/width/
/// rounding-mode as plain scalars and gets back whether the conversion
/// trapped, exactly mirroring `MipsExecutor::fpu_cvt_to_int`'s own call into
/// `cvt_to_int_and_commit`. See that function's doc comment (and
/// `cvt_f32_to_int_by_value`'s) for why CVT-to-int needs a by-value, no-host-
/// FPU-status-read path at all: reading MXCSR/FPSR after a hardware
/// conversion instruction races the flag write against the read on
/// out-of-order hardware.
#[cfg(feature = "jitv2")]
unsafe extern "C" fn jit_cvt_to_int<T: Tlb, C: CpuModel>(
    ctx: *mut core::ffi::c_void,
    fs_reg: u32,
    fd_reg: u32,
    fr1: u32,
    src_f64: u32,
    dst_i64: u32,
    rm: u32,
) -> u32 {
    let exec = unsafe { &mut *exec_from_core::<T, C>(ctx) };
    cvt_to_int_and_commit(&mut exec.core, fs_reg, fd_reg, fr1 != 0, src_f64 != 0, dst_i64 != 0, rm as u8) as u32
}

/// Single-implementation CVT.S.W/D.W/S.L/D.L — see `cvt_int_to_float_and_commit`'s
/// doc comment. Same `ctx`-is-`MipsExecutor<T,C>` shape as `jit_cvt_to_int`.
#[cfg(feature = "jitv2")]
unsafe extern "C" fn jit_cvt_int_to_float<T: Tlb, C: CpuModel>(
    ctx: *mut core::ffi::c_void,
    fs_reg: u32,
    fd_reg: u32,
    fr1: u32,
    src_i64: u32,
    dst_f64: u32,
) -> u32 {
    let exec = unsafe { &mut *exec_from_core::<T, C>(ctx) };
    cvt_int_to_float_and_commit(&mut exec.core, fs_reg, fd_reg, fr1 != 0, src_i64 != 0, dst_f64 != 0) as u32
}

/// Single-implementation CVT.S.D — see `cvt_d_to_s_and_commit`'s doc
/// comment. Same `ctx`-is-`MipsExecutor<T,C>` shape as `jit_cvt_to_int`.
#[cfg(feature = "jitv2")]
unsafe extern "C" fn jit_cvt_d_to_s<T: Tlb, C: CpuModel>(ctx: *mut core::ffi::c_void, fs_reg: u32, fd_reg: u32, fr1: u32) -> u32 {
    let exec = unsafe { &mut *exec_from_core::<T, C>(ctx) };
    cvt_d_to_s_and_commit(&mut exec.core, fs_reg, fd_reg, fr1 != 0) as u32
}

/// Inline lockstep step hook (`core.lockstep_step_fn`), called by compiled code
/// at the START of every JIT instruction with its synthesized `pc`/`raw`/`bd`.
/// Establishes the true per-instruction starting state, runs the instruction
/// once through the real interpreter to get the reference result, then restores
/// the starting state so the JIT can run the same instruction for real. See
/// `MipsCore::lockstep_step_fn`. No-op (returns without comparing) for a
/// `LockstepClass` that can't be compared per-dispatch (Branch/Other).
#[cfg(feature = "jitv2_lockstep")]
unsafe extern "C" fn jit_lockstep_step<T: Tlb, C: CpuModel>(ctx: *mut core::ffi::c_void, pc: u64, raw: u32, bd: u32) {
    let exec = unsafe { &mut *exec_from_core::<T, C>(ctx) };
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
unsafe extern "C" fn jit_lockstep_compare<T: Tlb, C: CpuModel>(ctx: *mut core::ffi::c_void) -> u32 {
    let exec = unsafe { &mut *exec_from_core::<T, C>(ctx) };
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

/// Portable (no host FPU status reads) CVT.W/L.S/D result + FCSR-cause
/// computation, shared by both engines: the interpreter calls it directly,
/// jitv2 calls it through a `core.fpu_cvt_to_int_*_fn` pointer exactly like
/// `fpr_read_w`/`write_fpr_w` are shared today. Exists because reading host
/// MXCSR/FPSR after a hardware conversion instruction (Rust's `as iN` cast,
/// or Cranelift's `fcvt_to_sint_sat`) races the flag write against the read
/// on out-of-order hardware — traced live via jitv2's equivalence tests
/// (`cvt_to_int_nan_matches_interpreter` et al. in equiv_test.rs) after a
/// codegen restructuring exposed a case where the JIT observed 0 flags
/// immediately after a conversion the interpreter, with different
/// instruction spacing, correctly saw as Invalid/Inexact. See
/// clanker/cpu/fpu-exception-model-vs-r4400.md.
///
/// `x` must already be rounded to an integral value via
/// `round_f32_to_int_mode`/`round_f64_to_int_mode` — this only classifies
/// and saturates. Two competing specs exist for NaN/out-of-range CVT (R4000
/// manual: Unimplemented Operation, destination untouched; MIPS IV ISA:
/// Invalid Operation, result saturates to the largest representable
/// magnitude) — this implements the MIPS IV ISA route (Invalid, not
/// Unimplemented) since it's expressible without a caller-side branch on
/// whether to skip the write, matching `fpu/cvt_out_of_range`'s accepted
/// second branch (`cpu-tests/tests/fpu/fpu_breadth.c`).
///
/// Returns `(result_bits, invalid, inexact)`. `result_bits` is the raw
/// destination bit pattern (already reinterpreted as unsigned, sign-extended
/// by the caller through the usual GPR/FPR write path); `dst_bits` selects
/// 32- vs 64-bit destination width (`CVT.W` vs `CVT.L`).
fn cvt_f32_to_int_by_value(x: f32, dst_bits: u32) -> (u64, bool, bool) {
    let (max_pos, max_neg): (i64, i64) = if dst_bits == 32 {
        (i32::MAX as i64, i32::MIN as i64)
    } else {
        (i64::MAX, i64::MIN)
    };
    if x.is_nan() {
        return (max_pos as u64, true, false);
    }
    // x is already integral (post-rounding); out-of-range iff it can't fit
    // in the destination width at all.
    let in_range = if dst_bits == 32 {
        x >= i32::MIN as f32 && x <= i32::MAX as f32
    } else {
        // i64::MAX isn't exactly representable in f32/f64; compare against
        // 2^63 (exact in both formats) the same way real hardware's range
        // check works, rather than against a rounded i64::MAX as f32/f64.
        x >= -(9223372036854775808.0f32) && x < 9223372036854775808.0f32
    };
    if !in_range {
        let result = if x.is_sign_negative() { max_neg } else { max_pos };
        return (result as u64, true, false);
    }
    let result = if dst_bits == 32 { x as i32 as i64 } else { x as i64 };
    let inexact = (result as f32) != x;
    (result as u64, false, inexact)
}

/// `f64` counterpart of `cvt_f32_to_int_by_value` — see that function's doc
/// comment for the full design rationale.
fn cvt_f64_to_int_by_value(x: f64, dst_bits: u32) -> (u64, bool, bool) {
    let (max_pos, max_neg): (i64, i64) = if dst_bits == 32 {
        (i32::MAX as i64, i32::MIN as i64)
    } else {
        (i64::MAX, i64::MIN)
    };
    if x.is_nan() {
        return (max_pos as u64, true, false);
    }
    let in_range = if dst_bits == 32 {
        x >= i32::MIN as f64 && x <= i32::MAX as f64
    } else {
        x >= -(9223372036854775808.0f64) && x < 9223372036854775808.0f64
    };
    if !in_range {
        let result = if x.is_sign_negative() { max_neg } else { max_pos };
        return (result as u64, true, false);
    }
    let result = if dst_bits == 32 { x as i32 as i64 } else { x as i64 };
    let inexact = (result as f64) != x;
    (result as u64, false, inexact)
}

/// CVT.W/L.S/D, all in one: reads the source FPR, rounds via
/// `round_f32/64_to_int_mode`, classifies/saturates via
/// `cvt_f32/64_to_int_by_value`, and commits the result and FCSR Cause/Flag
/// bits directly against `core` — mirroring `MipsExecutor::fpu_commit_cvt_to_int`
/// exactly, just operating on `&mut MipsCore` instead of `&mut Self` so a
/// free function (not a `MipsExecutor<T, C>` method) can call it. This is
/// the single function both engines share: the interpreter calls it
/// directly from `exec_fcvt_w_s` et al.; jitv2 calls it through
/// `core.fpu_cvt_to_int_fn` — one `call_indirect` that reads/writes `core`
/// itself, so the JIT never has to carry a partially-computed result or
/// Invalid/Inexact `Value` across the call boundary (Cranelift only needs
/// to know the call may have touched `core.fpr`/`core.fpu_fcsr`, which every
/// access already re-`load`s fresh — no stale cached `Value` risk). Returns
/// `true` if the conversion trapped (Cause set, destination/Flag untouched,
/// caller must raise `EXC_FPE`), `false` if it committed normally.
///
/// `fr1`: FR=1 (paired-64) vs FR=0 register addressing — see
/// `read_fpr_w_fr0`/`write_fpr_w_fr0`. `src_f64`/`dst_i64`: source is
/// D-format (else S); destination is L (else W). `rm`: MIPS FCSR.RM
/// encoding (0=nearest-even, 1=toward-zero, 2=toward+inf, 3=toward-inf) —
/// callers pass a fixed mode for ROUND/TRUNC/CEIL/FLOOR or FCSR's live low
/// 2 bits for plain CVT.W/CVT.L.
fn cvt_to_int_and_commit(core: &mut MipsCore, fs_reg: u32, fd_reg: u32, fr1: bool, src_f64: bool, dst_i64: bool, rm: u8) -> bool {
    let dst_bits = if dst_i64 { 64 } else { 32 };
    let (result, invalid, inexact) = if src_f64 {
        let bits = if fr1 { core.read_fpr_l(fs_reg) } else { read_fpr_l_fr0(core, fs_reg) };
        let x = f64::from_bits(bits);
        let rounded = round_f64_to_int_mode(x, rm);
        cvt_f64_to_int_by_value(rounded, dst_bits)
    } else {
        let bits = if fr1 { core.read_fpr_w(fs_reg) } else { read_fpr_w_fr0(core, fs_reg) };
        let x = f32::from_bits(bits);
        let rounded = round_f32_to_int_mode(x, rm);
        cvt_f32_to_int_by_value(rounded, dst_bits)
    };

    core.fpu_fcsr &= !FCSR_CM;
    let write_result = |core: &mut MipsCore| {
        if dst_i64 {
            if fr1 { core.write_fpr_l(fd_reg, result) } else { write_fpr_l_fr0(core, fd_reg, result) }
        } else if fr1 {
            core.write_fpr_w(fd_reg, result as u32)
        } else {
            write_fpr_w_fr0(core, fd_reg, result as u32)
        }
    };
    if !invalid && !inexact {
        write_result(core);
        return false;
    }
    let cause = if invalid { FCSR_CV } else { FCSR_CI };
    core.fpu_fcsr |= cause;
    let enable_bit = if invalid { 0x800 } else { FCSR_IM };
    if (core.fpu_fcsr & enable_bit) != 0 {
        return true;
    }
    write_result(core);
    core.fpu_fcsr |= if invalid { 0x40 } else { FCSR_FI };
    false
}

/// CVT.S.W/D.W/S.L/D.L (int source): a MIPS integer converted to float can
/// only ever be Inexact (the source is always exactly representable as an
/// integer; loss can only occur fitting it into the destination float's
/// mantissa) — never Invalid, never Overflow. Computed by value: convert,
/// then compare the converted-back-to-int value against the original
/// source, the same "did round-tripping change it" test
/// `round_f32/64_to_int_mode`'s callers already use, avoiding the host
/// MXCSR/FPSR race `cvt_to_int_and_commit`'s doc comment describes (found
/// live in the opposite conversion direction; the same hazard applies here
/// since `fcvt_from_sint` is just as real a hardware instruction).
fn cvt_int_to_float_and_commit(core: &mut MipsCore, fs_reg: u32, fd_reg: u32, fr1: bool, src_i64: bool, dst_f64: bool) -> bool {
    let src: i64 = if src_i64 {
        (if fr1 { core.read_fpr_l(fs_reg) } else { read_fpr_l_fr0(core, fs_reg) }) as i64
    } else {
        (if fr1 { core.read_fpr_w(fs_reg) } else { read_fpr_w_fr0(core, fs_reg) }) as i32 as i64
    };
    let inexact = if dst_f64 {
        let result = src as f64;
        let bits = result.to_bits();
        if fr1 { core.write_fpr_l(fd_reg, bits) } else { write_fpr_l_fr0(core, fd_reg, bits) }
        (result as i64) != src
    } else {
        let result = src as f32;
        let bits = result.to_bits();
        if fr1 { core.write_fpr_w(fd_reg, bits) } else { write_fpr_w_fr0(core, fd_reg, bits) }
        (result as i64) != src
    };
    core.fpu_fcsr &= !FCSR_CM;
    if !inexact {
        return false;
    }
    core.fpu_fcsr |= FCSR_CI;
    if (core.fpu_fcsr & FCSR_IM) != 0 {
        return true;
    }
    core.fpu_fcsr |= FCSR_FI;
    false
}

/// CVT.S.D (narrowing, D source only — the widening CVT.D.S can never raise
/// any FCSR condition and skips this machinery entirely): Rust's `as f32`
/// narrowing cast is round-to-nearest-even, matching the MIPS-mandated
/// narrowing behavior, but (same hazard as
/// `cvt_int_to_float_and_commit`/`cvt_to_int_and_commit`) computed by value
/// rather than trusted from a host MXCSR read after the cast. Inexact: the
/// narrowed result doesn't widen back to the original bit-for-bit. Overflow:
/// the magnitude exceeds `f32::MAX`, in which case the destination is
/// infinity with the source's sign — matches Rust's own narrowing behavior
/// for this case, so no manual saturation is needed, just cause reporting.
fn cvt_d_to_s_and_commit(core: &mut MipsCore, fs_reg: u32, fd_reg: u32, fr1: bool) -> bool {
    let src = if fr1 { core.read_fpr_d(fs_reg) } else { read_fpr_d_fr0(core, fs_reg) };
    let result = src as f32;
    let overflow = result.is_infinite() && !src.is_infinite();
    let inexact = !overflow && (result as f64) != src;
    let bits = result.to_bits();
    core.fpu_fcsr &= !FCSR_CM;
    if !overflow && !inexact {
        if fr1 { core.write_fpr_w(fd_reg, bits) } else { write_fpr_w_fr0(core, fd_reg, bits) }
        return false;
    }
    let cause = if overflow { FCSR_CO } else { FCSR_CI };
    core.fpu_fcsr |= cause;
    let enable_bit = if overflow { 0x200 } else { FCSR_IM }; // Enable.O = bit 9
    if (core.fpu_fcsr & enable_bit) != 0 {
        return true;
    }
    if fr1 { core.write_fpr_w(fd_reg, bits) } else { write_fpr_w_fr0(core, fd_reg, bits) }
    core.fpu_fcsr |= if overflow { FCSR_FO } else { FCSR_FI };
    false
}

/// Bump the real, shared cycle counter directly — no local shadow, no
/// batching. See `Hot::cycles`'s doc comment for why: a dispatch loop that
/// stays entirely inside JIT-compiled code for a long stretch must still
/// make this visible to other threads as it goes, not just whenever it
/// happens to return to this outer loop. Volatile, not a plain field write:
/// guarantees the compiler can't elide/hoist the write out of an unbounded
/// loop, even though this isn't a synchronizing atomic RMW (readers only
/// need eventual visibility of the count, not ordering against other
/// memory). Also runs `ci_clock`'s timer-fire check, immediately after, on
/// the freshly-bumped value — `ci_clock`'s "fire when hot.cycles >=
/// count_fire_cycle" is a threshold test, not an edge trigger, so it stays
/// correct (just coarser-grained, still deterministic) even when the caller
/// only calls this once per N instructions instead of every single one —
/// see `step_int_0cycles`'s doc comment.
macro_rules! step_cycles {
    ($self:ident) => {{
        unsafe {
            let p = &mut $self.core.hot.cycles as *mut u64;
            std::ptr::write_volatile(p, std::ptr::read_volatile(p).wrapping_add(1));
        }

        // ci_clock: the compare "timer" is a deterministic hot.cycles
        // threshold instead of an hptimer thread — check it here, before the
        // pending load below, so the fire is delivered within this same step.
        #[cfg(feature = "ci_clock")]
        if $self.core.hot.cycles >= $self.core.count_fire_cycle {
            // Next architectural match is a full 32-bit Count wrap away;
            // normally a Compare write re-arms much sooner.
            let wrap_ns = ((1u128 << 32) * 1_000_000_000) / $self.core.count_hz as u128;
            $self.core.count_fire_cycle = $self.core.hot.cycles
                .saturating_add(wrap_ns as u64 / crate::cpu::mips_core::NS_PER_GUEST_CYCLE);
            $self.core.hot.interrupts.fetch_or(crate::cpu::mips_core::CAUSE_IP7 as u64, Ordering::SeqCst);
            crate::pf_level!(crate::perfetto::events::IP7_LEVEL, crate::perfetto::events::IP7_COMPARE, true);
            $self.core.fasttick_count.fetch_add(1, Ordering::Relaxed);
        }
    }};
}

/// Per-instruction preamble shared by `step_int`/`step_jit`: cycle counter
/// (via `step_cycles!`), idle-profiler sampling, breakpoint check,
/// timer/interrupt delivery. A macro, not a `#[inline(always)]` fn: every
/// early `return` below must return from the calling `step_int`/`step_jit`
/// itself, not from some separate `step_preamble` call — textual expansion
/// at the call site guarantees that directly, with no reliance on the
/// inliner actually honoring the hint. Falls through (no early `return`)
/// when the instruction at `core.pc` is clear to fetch/dispatch this call.
/// Invoked as `step_preamble!(self)` from inside a `MipsExecutor<T, C>`
/// method — `macro_rules!` is hygienic, so `self` doesn't cross the
/// expansion boundary implicitly and must be passed in explicitly as
/// `$self`. `step_preamble!(self, no_cycles)` skips the `step_cycles!` bump
/// — see `step_int_0cycles`'s doc comment for who wants that and why.
macro_rules! step_preamble {
    ($self:ident) => {{
        step_cycles!($self);
        step_preamble!($self, no_cycles)
    }};
    ($self:ident, no_cycles) => {{
        /*
        // Reload external interrupt state every 16 instructions
        if $self.core.hot.cycles & 0xF == 0 {
            $self.cached_pending = $self.core.hot.interrupts.load(Ordering::Relaxed);
        }
        let pending = $self.cached_pending;
        */
        // this seems to be a wash or slightly better without a branch, relaxed atomic loads are essentially MOV
        let pending = $self.core.hot.interrupts.load(Ordering::Relaxed);

        let pc = $self.core.pc;

        // Spin/idle-loop PC sampler. Armed lock-free via the shared atomic, so
        // the CPU is never paused/resumed to enable it (resuming corrupts a
        // live kernel). Inert (one relaxed load + branch) when disarmed.
        #[cfg(feature = "idle-pause")]
        if unsafe { &*$self.idle_profile_on_ptr }.load(Ordering::Relaxed) {
            if $self.idle_profile_reset.load(Ordering::Relaxed) {
                $self.idle_profiler.reset();
                $self.idle_profile_reset.store(false, Ordering::Relaxed);
            }
            let ie = $self.core.interrupts_enabled();
            $self.idle_profiler.sample(pc, ie);
        }

        #[cfg(not(feature = "lightning"))]
        if $self.bp_enabled() && $self.check_breakpoint::<{ BpType::Pc as u8 }>(pc) {
            // This runs before the fetch, so only honor the breakpoint once the
            // instruction is fetchable: an instruction whose fetch takes a TLB
            // miss never executes, and a debugger can't read that page either.
            if !$self.debug_translate(pc).is_exception() {
                return EXEC_BREAKPOINT;
            }
            // Not taken: drop the id check_breakpoint() just recorded, or the
            // next stop is misreported as this breakpoint.
            $self.last_bp_hit = None;
        }

        // No per-instruction CP0 Count work: Count is virtual (materialized
        // lazily on read from the wall-clock anchor in mips_core.rs), and the
        // Count==Compare interrupt arrives through `pending` like any device
        // line — armed as an hptimer one-shot on each Compare write.
        // Fast path: skip all signal/interrupt handling when nothing is pending
        if (pending | $self.core.cp0_cause as u64) != 0 {
            // Soft reset (bit 63): stop this step call here — the reset
            // vector's first instruction runs on the NEXT step call, not
            // this one. EXEC_RETRY, not EXEC_COMPLETE: this call retired
            // nothing (real semantic meaning, same as a bus-busy retry) —
            // EXEC_COMPLETE is reserved for "ran fine", which this didn't.
            if pending & SOFT_RESET_BIT != 0 {
                $self.core.reset(true); // clears interrupts word (including bit 63)
                // reset() rewrote cp0_status (BEV|ERL|SR) by direct field write and set
                // pc to the KSEG1 reset vector. Without re-deriving, a soft reset taken
                // while the guest was in user mode would leave the user translator
                // installed and fault on the reset vector fetch itself.
                $self.resync_privilege_state();
                $self.core.in_delay_slot = false;
                $self.core.delay_slot_target = 0;
                return EXEC_RETRY;
            }

            // Merge external IP bits into Cause
            $self.core.cp0_cause = ($self.core.cp0_cause & !EXT_INT_MASK) | (pending as u32 & EXT_INT_MASK);

            #[cfg(feature = "developer")]
            let skip_int = $self.skip_interrupts;
            #[cfg(not(feature = "developer"))]
            let skip_int = false;

            if $self.core.interrupts_enabled() && !skip_int {
                let ip = $self.core.cp0_cause & crate::cpu::mips_core::CAUSE_IP_MASK;
                let im = $self.core.cp0_status & crate::cpu::mips_core::STATUS_IM_MASK;
                if (ip & im) != 0 {
                    crate::pf_instant!(crate::perfetto::events::INT_TAKEN, (ip & im) >> 8);
                    let s = exec_exception(EXC_INT);
                    return $self.handle_exception(s);
                }
            }
        }
    }};
}

impl<T: Tlb, C: CpuModel> MipsExecutor<T, C> {
    /// The suffix a bus-error line carries when the device behind `phys_addr`
    /// can say why the access failed. Asked only after a failure, so a good
    /// access never pays; it changes nothing the guest sees.
    #[cold]
    fn fault_note(&self, phys_addr: u64, size: usize) -> String {
        self.sysad.describe_fault(phys_addr, size).map_or_else(String::new, |why| format!(" [{why}]"))
    }

    /// Create a new executor from a config and a bus (sysad) and a TLB.
    /// The cache hierarchy is constructed internally as a unified R4000Cache.
    pub fn new(sysad: Arc<dyn BusDevice>, mut tlb: T, cfg: &MipsCpuConfig) -> Self
    where
        C: From<Arc<dyn BusDevice>>
    {
        let mut core = MipsCore::new();
        // Mapped pages decode their C field with this model's table, as KSEG0
        // and xkphys do.
        tlb.set_cache_attrs(Self::cache_attr_table());

        // Cache geometry comes from the model type C, so Config describes this CPU.
        // `mut` is only used by the Triton L2-enable sync below.
        #[cfg_attr(not(feature = "r5ksc_triton"), allow(unused_mut))]
        let mut cache = C::from(sysad.clone());

        // Build CP0 Config register from architecture constants.
        let mut config = 0u32;

        // K0 (bits 2:0): kseg0 coherency algorithm. 3 = Cacheable, non-coherent.
        config |= 3 << CONFIG_K0;

        // DB (bit 4): Primary D-cache line size. 0=16B, 1=32B.
        config |= (if C::DC_LINE >= 32 { 1 } else { 0 }) << CONFIG_DB;

        // IB (bit 5): Primary I-cache line size. 0=16B, 1=32B.
        config |= (if C::IC_LINE >= 32 { 1 } else { 0 }) << CONFIG_IB;

        // DC (bits 8:6): Primary D-cache size. size = 2^(12+DC)
        config |= C::DC_SIZE.trailing_zeros().saturating_sub(12) << CONFIG_DC;

        // IC (bits 11:9): Primary I-cache size. size = 2^(12+IC)
        config |= C::IC_SIZE.trailing_zeros().saturating_sub(12) << CONFIG_IC;

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
        config |= (if C::L2_SIZE > 0 { 0 } else { 1 }) << CONFIG_SC;

        // SB (bits 23:22): Secondary cache block size.
        // 00=4 words (16B), 01=8 words (32B), 10=16 words (64B), 11=32 words (128B).
        config |= (match C::L2_LINE {
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
            let ss: u32 = match C::L2_SIZE {
                524288  => 0b00,  // 512 KB
                1048576 => 0b01,  // 1 MB
                2097152 => 0b10,  // 2 MB
                _       => 0b11,  // none / 4 MB
            };
            config |= ss << CONFIG_TR_SS;
        }

        // R10000 lays Config out completely differently from an R4000, and the
        // PROM sizes its cache diagnostics from it. Fields, per NetBSD's
        // mips/include/cpuregs.h (MIPS4_CONFIG_*):
        //   [31:29] IC  primary I-cache size, as 4096 << field
        //   [28:26] DC  primary D-cache size, likewise
        //   [18:16] SS  secondary cache size
        //   [15]    BE  big endian
        //   [13]    SB  secondary block size, 0 = 64B, 1 = 128B
        //   [2:0]   K0  kseg0 cacheability, which the PROM sets for itself
        // Presenting an R4000 Config here told an R10000 PROM that its
        // secondary cache size field was zero.
        if C::R10K_CACHE_OPS {
            let log2 = |n: usize| (n / 4096).trailing_zeros();
            let mut c = 0u32;
            c |= log2(32768) << 29;          // 32 KB L1I
            c |= log2(32768) << 26;          // 32 KB L1D
            c |= 1 << 15;                    // big endian
            if C::L2_LINE == 128 { c |= 1 << 13; }
            // Secondary cache size. The encoding is not in anything to hand,
            // so it was swept against the PROM rather than guessed.
            //
            // 1 is the value to keep: IRIX reports "Secondary unified
            // instruction/data cache size: 1 Mbyte", which agrees with this
            // model's own `L2_SIZE`, and the PROM's power-on diagnostics pass
            // and IRIX boots with it. 4 also passes POST but has IRIX report
            // 8 MB, contradicting the model.
            c |= 1 << 16;
            c |= 2;                          // K0 = uncached at reset
            config = c;
        }

        core.cp0_config = config;
        core.tlb_entries = C::TLB_ENTRIES as u32;
        // MipsCore::new already ran reset_registers, so set both the reset value
        // and the live register — otherwise the model's identity only appears
        // after the next power_on.
        core.reset_prid = C::PRID;
        core.reset_fir = C::FIR;
        core.cp0_prid = C::PRID;
        core.fpu_fir = C::FIR;

        // Triton: sync initial L2 enabled state from Config SE bit (starts 0 = disabled).
        #[cfg(feature = "r5ksc_triton")]
        cache.set_l2_enabled((config >> CONFIG_SE) & 1 != 0);

        /*eprintln!("Cache config: L1I {}KB/{}B-line  L1D {}KB/{}B-line  L2 {}KB/{}B-line  CP0.Config={:#010x}",
            ic_size / 1024, ic_line,
            dc_size / 1024, dc_line,
            l2_size / 1024, l2_line,
            config);*/

        // Built before `jitv2` moves into its `Arc<Mutex<_>>` below, so the
        // hot dispatch-gate path (`exec_decoded`'s compile_queue.send) can
        // hold these two `Arc` handles directly and never has to take
        // `jitv2.lock()` just to push a compile request or bump a stats
        // counter — see `jitv2_compile_queue_handle`/`jitv2_stats`'s own
        // doc comments on the struct fields.
        #[cfg(feature = "jitv2")]
        let mut jitv2 = crate::cpu::jitv2::Jitv2::new(crate::cpu::jitv2::JITV2_INITIAL_PAGE_CAPACITY);
        // The ISA level travels as data rather than through a global: the
        // model's `MIPS4` const goes to the compile pool here and to this
        // executor's own inline analyzer below, and from there into
        // `classify`. The interpreter reads `C::MIPS4` at every decode; the
        // compile workers have no `C` in scope, so they are handed the value
        // before they exist. An R4400 must raise Reserved Instruction on
        // MIPS IV encodings where an R5000/R10000 may execute them, and one
        // binary serves both.
        #[cfg(feature = "jitv2")]
        jitv2.compile_queue.set_isa(C::MIPS4);
        #[cfg(feature = "jitv2")]
        let jitv2_compile_queue_handle = jitv2.compile_queue.queue_handle();
        #[cfg(feature = "jitv2")]
        let jitv2_stats = jitv2.stats.clone();

        let mut executor = Self {
            core,
            sysad,
            cheritest_dump_hook: false,
            tlb,
            cache,
            #[cfg(feature = "jitstats")]
            jit_mem_stats: JitMemStatsPair::default(),
            #[cfg(feature = "jitstats")]
            jit_last_xlat_hit: false,
            #[cfg(feature = "developer")]
            undo_buffer: UndoBuffer::new(),
            #[cfg(feature = "developer")]
            pending_memory_writes: Vec::new(),
            traceback: TracebackBuffer::new(),
            #[cfg(feature = "developer")]
            trace_writer: None,
            #[cfg(feature = "llstats")]
            ll_stats: Box::new([LlSiteStats { key: u64::MAX, ..Default::default() }; LL_STATS_SLOTS]),
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
                id: 0, addr: 0, kind: BpType::Pc, enabled: false, condition: None, condition_src: None,
                hits: 0, ignore: 0,
            }],
            next_bp_id: 1,
            last_bp_hit: None,
            bp_hit_detail: None,
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
            fpr_read_d:  crate::cpu::mips_core::read_fpr_d_fr0,
            fpr_write_d: crate::cpu::mips_core::write_fpr_d_fr0,
            fpr_read_l:  crate::cpu::mips_core::read_fpr_l_fr0,
            fpr_write_l: crate::cpu::mips_core::write_fpr_l_fr0,
            fpr_read_w:  crate::cpu::mips_core::read_fpr_w_fr0,
            fpr_write_w: crate::cpu::mips_core::write_fpr_w_fr0,
            cached_pending: 0,
            #[cfg(feature = "instr_stats")]
            instr_stats: crate::cpu::mips_instr_stats::InstrStats::default(),
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
            jitv2: std::sync::Arc::new(Mutex::new(jitv2)),
            #[cfg(feature = "jitv2")]
            jitv2_pfn_map: std::ptr::null(),
            #[cfg(feature = "jitv2")]
            jitv2_world: None,
            #[cfg(feature = "jitv2")]
            jitv2_pages_base: std::ptr::null_mut(),
            #[cfg(feature = "jitv2")]
            jitv2_compile_queue_handle,
            #[cfg(feature = "jitv2")]
            jitv2_stats,
            #[cfg(feature = "jitv2")]
            pcp: std::ptr::null_mut(),
            #[cfg(all(feature = "jitv2", feature = "jitv2_smc_check"))]
            smc_seen: std::collections::HashSet::new(),
            #[cfg(feature = "jitv2_lockstep")]
            ls_before: None,
            #[cfg(feature = "jitv2_lockstep")]
            ls_expected: None,
            #[cfg(feature = "jitv2_lockstep")]
            ls_delay_target_before: 0,
            #[cfg(feature = "jitv2_lockstep")]
            ls_delay_target_expected: 0,
            #[cfg(feature = "jitv2_lockstep")]
            ls_history: std::collections::VecDeque::with_capacity(LS_HISTORY_CAP),
            // Threaded compile is now the default — see Machine::new's
            // compile_queue.start() call, which must run whenever this is
            // `false` (an un-started queue would silently drop every
            // compile request: `try_schedule`/`compile_queue.send` succeed,
            // nothing is ever there to pop them). `j2 inline on` still
            // switches back to this executor thread doing its own inline
            // compiles, same as before. `jitv2_lockstep` always starts `true`
            // instead: `Machine::new` deliberately leaves the compile queue
            // stopped under lockstep (see its own comment on why — inline
            // compile takes the shared Codegen the queue would otherwise
            // need), so this field must already read `true` before the CPU
            // thread's very first dispatch, not rely on `step_jit`
            // re-deriving the same fact via `cfg!(feature = "jitv2_lockstep")`
            // at every call site. The `j2 inline off` monitor command refuses
            // under `jitv2_lockstep` (same shape as its existing `lightning`
            // refusal) so this can never be flipped back to `false` live either.
            #[cfg(feature = "jitv2")]
            jitv2_inline_compile: cfg!(feature = "jitv2_lockstep"),
            #[cfg(feature = "jitv2")]
            jitv2_inline_analyzer: crate::cpu::jitv2::analyzer::Analyzer::with_isa(C::MIPS4),
            // IRIS_JIT_DISPATCH=off starts with the JIT dispatch gate shut: the
            // same binary as an interpreter-only reference, for comparing
            // results against compiled code (cpu-tests/jitcov). The monitor's
            // `j2 dispatch on|off` still flips it at runtime.
            #[cfg(feature = "jitv2")]
            jitv2_dispatch_enabled: !matches!(std::env::var("IRIS_JIT_DISPATCH").as_deref(), Ok("off") | Ok("0")),
            batch_hook: None,
            phys_alias: None,
            phys_mask: u64::MAX,
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
        #[cfg(feature = "fetchverify")]
        { executor.core.fetch_verify_on = fetch_verify_env_default(); }
        executor.update_translate_fn();
        executor.update_fpr_mode();

        executor
    }

    /// Re-sync raw atomic pointers after the shared Arcs are injected post-construction.
    /// (`cycles` has no equivalent — it's an inline `MipsCore` field now, always current.)
    pub fn rebind_atomic_ptrs(&mut self) {
        #[cfg(feature = "idle-pause")]
        { self.idle_profile_on_ptr = Arc::as_ptr(&self.idle_profile_on); }
        #[cfg(feature = "jitv2")]
        self.jitv2_bind_fast_lookup();
    }

    /// ppmem: pointer to this executor's inline `ppmem_bitmap` word, for
    /// `PpMemSpace` to write through on every remap.
    ///
    /// Same contract as [`Self::interrupts_ptr`] — call it once the executor is
    /// at its final address (inside its owning `Arc`), never cache it across a
    /// move. Unlike `interrupts_ptr` this is deliberately **not** atomic: the
    /// only writer is the MC's MEMCFG path, which runs on this same CPU thread
    /// (CPU store → MC register write → `remap_banks` → ppmem), and no device
    /// ever reads or writes it.
    pub fn ppmem_bitmap_ptr(&self) -> *mut u64 {
        &self.core.ppmem_bitmap as *const u64 as *mut u64
    }

    /// tcache: hand the cache ppmem's window base and mapped-region bitmap, so
    /// cacheable RAM accesses read/write the window directly instead of copying
    /// line data into the cache's own arrays.
    ///
    /// # Safety
    /// Same contract as `CpuCache::set_tcache_window`: `base` must be ppmem's
    /// window base and `bitmap` a live `u64` outliving this cache.
    pub unsafe fn set_tcache_window(&self, base: *mut u8) {
        unsafe { self.cache.set_tcache_window(base) };
    }

    /// tcache: pointer to the cache's inline bitmap field.
    pub fn tcache_bitmap_ptr(&self) -> *mut u64 {
        self.cache.tcache_bitmap_ptr()
    }

    /// tcache + jitv2: see `CpuCache::set_tcache_gen_window`.
    ///
    /// # Safety
    /// Same contract as that method.
    #[cfg(feature = "jitv2")]
    pub unsafe fn set_tcache_gen_window(&mut self, gen_base: *mut std::sync::atomic::AtomicU64) {
        unsafe { self.cache.set_tcache_gen_window(gen_base) };
        // The jitv2 inline store path reads this pointer from `MipsCore` at a
        // fixed offset, so the copy taken at `install_jit_hooks` time is stale
        // the moment the window is (re)published — refresh it here. Same for
        // the transparent base and bitmap, which ppmem wires up in the same
        // machine.rs block.
        #[cfg(feature = "jitv2")]
        self.install_jit_mem_ptrs();
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
    pub fn cycles_ptr(&self) -> crate::cpu::mips_core::CyclesPtr {
        crate::cpu::mips_core::CyclesPtr::new(&self.core.hot.cycles as *const u64)
    }

    /// Install the CP0 Status change callback pointing at this executor.
    ///
    /// This exists for exactly one reason: `MipsCore::write_cp0` has only a
    /// `&mut MipsCore` and so cannot reach [`Self::resync_privilege_state`] on its own,
    /// so a guest `MTC0/DMTC0 $x, Status` needs this bridge to resync. **Every other**
    /// Status mutation (ERET, exception delivery, reset, snapshot/digest restore, gdb
    /// register writes) calls `resync_privilege_state` directly and does not depend on
    /// this being installed.
    ///
    /// **Must not be called before the executor reaches its final address.** It stores
    /// `self as *mut Self`, and `MipsExecutor` is neither pinned nor `Box`ed: it is
    /// returned by value from `new()` and moved again into `Arc<Mutex<_>>` by
    /// `MipsCpu::new`, which is why the production call site sits there and not in
    /// `new()`. Installing earlier records a stack frame that is dead by the time the
    /// CPU thread runs.
    ///
    /// A directly-constructed executor (tests) has no callback installed, so it must
    /// call this itself if it executes a real `MTC0 Status` instruction — see
    /// `mips_exec_test::test_nutlb_kernel_entry_unreachable_from_user`.
    pub fn install_status_cb(&mut self) {
        let ctx = self as *mut Self as *mut core::ffi::c_void;
        self.core.status_changed_cb = Some((mips_executor_status_cb::<T, C>, ctx));
    }

    /// Publish the cache/ppmem base pointers the inline load/store fast path
    /// indexes from compiled code (docs/jit-inline-memory.md §3.2).
    ///
    /// Must be re-run after anything that can move the backing storage —
    /// snapshot restore, cache reconfigure, ppmem remap — because compiled
    /// code holds these as plain addresses. Called from `install_jit_hooks`,
    /// so any path already re-running that is covered.
    #[cfg(feature = "jitv2")]
    pub fn install_jit_mem_ptrs(&mut self) {
        let mut geom = self.cache.jit_dc_geometry();
        geom.atomic_gen = self.core.smp || std::env::var_os("IRIS_JIT_ATOMIC_GEN").is_some();
        // Publish for the async compile worker, which owns its Codegen by
        // value and so cannot be stamped directly (see worker_loop).
        let core_addr = &mut self.core as *mut MipsCore as usize;
        let geometry_changed;
        {
            let j = self.jitv2.lock();
            geometry_changed = *j.dc_geometry.lock() != geom && j.pages_used() > 0;
            *j.dc_geometry.lock() = geom;
            // Everything codegen may bake as an immediate reaches its final
            // value here — this method is the last thing that moves any of
            // it (see `codegen::JitConsts`). Republished on every call so a
            // tcache window remap keeps the constants honest.
            let consts = crate::cpu::jitv2::codegen::JitConsts {
                // IRIS_JIT_PIC=1: publish no core address, so codegen loads the
                // hook pointers from the core instead of baking them. Compiled
                // code then holds no host address at all (the memory helpers,
                // the other baked address, are off unless IRIS_MEM_HELPERS is
                // set) -- the property a persistent code cache needs.
                // Several CPUs share compiled code, so no one core's address.
                core: if self.core.smp || std::env::var_os("IRIS_JIT_PIC").is_some() { None } else { core::num::NonZeroUsize::new(core_addr) },
            };
            *j.jit_consts.lock() = consts;
            // The inline-compile path (`jitv2_compile_inline`) takes this
            // `Codegen` directly rather than going through `worker_loop`, so
            // it has to be stamped here too — it was built by `Jitv2::new`,
            // before this core existed.
            let mut cg_slot = j.codegen.lock();
            if let Some(cg) = cg_slot.as_mut() {
                cg.dc_geometry = geom;
                cg.jit_consts = consts;
            }
        }
        if geom.supported {
            self.core.jit_dc_tags = self.cache.jit_dc_tags_ptr();
            self.core.jit_dc_lru  = self.cache.jit_dc_lru_ptr();
        } else {
            // Null tags = "do not emit the inline path" (R5000's 2-way L1-D,
            // PassthroughCache, anything else unsupported).
            self.core.jit_dc_tags = std::ptr::null_mut();
            self.core.jit_dc_lru  = std::ptr::null_mut();
        }
        self.core.jit_tc_base = self.cache.tcache_base_ptr();
        self.core.jit_tc_gen = self.cache.tcache_gen_ptr();
        self.core.jit_l2_tags = self.cache.jit_l2_tags_ptr();
        // A cache that changes shape at run time (the O2's hand-over from the
        // full model to the tagless one once a kernel starts): code compiled
        // so far probes the old shape, and the compile workers took their
        // geometry when they started. Flushing discards that code and
        // restarts the workers, which then read the geometry published
        // above. Without it, every inline load on the O2 probed the full
        // model's emptied tags and called out (73M helper calls a second
        // at idle); going back to the full model on a reboot would also
        // leave tagless code reading memory under the full model's dirty
        // lines.
        if geometry_changed {
            self.jitv2_flush_from_here(false);
            // Not clear_pcp alone: the fetch nanotlb would still hit, and
            // nothing would re-derive the page step_jit needs.
            self.nanotlb_invalidate();
        }
    }

    /// Install JIT v2's memory-access and exception-delivery hooks
    /// (`MipsCore`'s `read*_fn`/`write*_fn`/`handle_exception_fn` fields —
    /// see their doc comments).
    ///
    /// Installs only function pointers, no context pointer: every hook takes
    /// the `*mut MipsCore` compiled code was entered with and recovers the
    /// executor from it via `exec_from_core`. So, unlike `interrupts_ptr`,
    /// this does *not* require the executor to be at a final, stable address
    /// — nothing here captures `self`'s address, and an executor that moves
    /// after installation stays correct rather than dangling. (Callers that
    /// box the executor before installing, e.g. `equiv_test`'s `run_jit`,
    /// are now belt-and-braces rather than load-bearing.)
    #[cfg(feature = "jitv2")]
    pub fn install_jit_hooks(&mut self) {
        self.install_jit_mem_ptrs();
        self.core.read8_fn = jit_read8::<T, C>;
        self.core.read8_sext_fn = jit_read8_sext::<T, C>;
        self.core.read16_fn = jit_read16::<T, C>;
        self.core.read16_sext_fn = jit_read16_sext::<T, C>;
        self.core.read32_fn = jit_read32::<T, C>;
        self.core.read32_sext_fn = jit_read32_sext::<T, C>;
        self.core.read64_fn = jit_read64::<T, C>;
        self.core.write8_fn = jit_write8::<T, C>;
        self.core.write16_fn = jit_write16::<T, C>;
        self.core.write32_fn = jit_write32::<T, C>;
        self.core.write64_fn = jit_write64::<T, C>;
        self.core.write64_masked_fn = jit_write64_masked::<T, C>;
        self.core.handle_exception_fn = jit_handle_exception::<T, C>;
        self.core.handle_exception_at_fn = jit_handle_exception_at::<T, C>;
        self.core.interp_fallback_fn = jit_interp_fallback::<T, C>;
        self.core.kill_entry_fn = jit_kill_entry::<T, C>;
        #[cfg(feature = "developer")]
        { self.core.dev_trace_bp_fn = jit_dev_trace_bp::<T, C>; }
        #[cfg(feature = "fetchverify")]
        { self.core.fetch_verify_fn = jit_fetch_verify::<T, C>; }
        self.core.fpu_set_mode_fn = jit_fpu_set_mode;
        self.core.fpu_cvt_to_int_fn = jit_cvt_to_int::<T, C>;
        self.core.fpu_cvt_int_to_float_fn = jit_cvt_int_to_float::<T, C>;
        self.core.fpu_cvt_d_to_s_fn = jit_cvt_d_to_s::<T, C>;
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
        // Read/Write are served by the data-side micro-TLB instead of the
        // single nanotlb slot. Dispatching here rather than at each call site
        // means every caller (read_data, write_data, the LL/SC paths, the
        // jitv2 hooks) reaches it with no edit of its own.
        //
        // Fetch is deliberately NOT routed here — it keeps the untagged,
        // barrier-flushed nanotlb slot (docs/nutlb-design.md §7), so the
        // jitv2 `pcp` tracking below is unaffected.
        if AT == AccessType::Read as u8 {
            return self.nutlb_translate::<false>(va);
        }
        if AT == AccessType::Write as u8 {
            return self.nutlb_translate::<true>(va);
        }
        // Only Fetch reaches here — Read/Write returned above — and the
        // inline load/store path jitstats measures is data-side only, so
        // there is nothing to record for this access type.

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
            self.core.nanotlb[AT as usize].fill_raw(va_page, result.phys, result.status & 0x7);
            #[cfg(feature = "jitv2")]
            if AT == AccessType::Fetch as u8 {
                self.jitv2_track_pcp(result.phys);
            }
        }
        result
    }

    /// Probe the nutlb for `va` on the Read (`W=false`) or Write (`W=true`)
    /// array, falling back to `translate_fn` and filling on miss.
    ///
    /// The hit test is three checks, and they are *not* foldable into one
    /// compare (docs/nutlb-design.md §3a):
    ///
    /// 1. **Identity** — page number + ASID, one 64-bit compare. `asid_mask`
    ///    is 0 for a global page, which drops the ASID from both sides.
    /// 2. **Generation** — the TLB must not have been rewritten since the
    ///    fill.
    /// 3. **Permission** — `(virttag & cur_sec_mask) != 0`, a *subset* test:
    ///    "is the current mode in this page's permitted set". Tag bits 10:8
    ///    are downward-inclusive (a KUSEG page sets all three), so kernel
    ///    reaching user memory — the `copyin`/`copyout` path — hits. Folding
    ///    this into the equality compare is what makes the obvious version of
    ///    this test wrong: it would spuriously miss on exactly that path.
    ///
    /// A permission failure deliberately falls through to the slow path
    /// rather than raising here, so `cp0_badvaddr` and the rest are set by
    /// the existing `translate_*bit_impl` code. The nutlb never synthesizes
    /// an exception.
    #[inline(always)]
    fn nutlb_translate<const W: bool>(&mut self, va: u64) -> TranslateResult {
        let idx = crate::cpu::mips_core::nutlb_index(va);
        let arr = if W { crate::cpu::mips_core::NUTLB_WRITE } else { crate::cpu::mips_core::NUTLB_READ };
        let e = &self.core.nutlb[arr][idx];

        // The whole hit test: one page compare, one bitmask bit. No
        // generation, no ASID, no permission — all three are covered by
        // flushing instead (see `MipsExecutor::nanotlb_invalidate`).
        //
        // Bitwise `&`, not `&&`: deliberately branchless. Short-circuit `&&`
        // emits a second conditional jump for a predicate that is already
        // this cheap; `&` costs one extra ALU op and zero branches, and is
        // flat in the miss rate where `&&` degrades with it.
        //
        // The bitmask word load is independent of the entry load, so the two
        // issue in parallel rather than serializing.
        if (e.virttag == (va & !0xFFF)) & self.core.nutlb_is_valid(arr, idx) {
            #[cfg(feature = "tlbstats")]
            self.tlb.stats_nutlb_hit(if W { AccessType::Write } else { AccessType::Read });
            #[cfg(feature = "jitstats")]
            { self.jit_last_xlat_hit = true; }
            return TranslateResult::ok(e.phys_addr(va), e.cache_attr_raw());
        }

        #[cfg(feature = "tlbstats")]
        {
            let at = if W { AccessType::Write } else { AccessType::Read };
            // Distinguish conflict from capacity/cold: a valid tag with a
            // different page number means this set was occupied by someone
            // else, which is the argument for more sets (or associativity).
            //
            // A zero tag is a legal page number, so validity has to come
            // from the bitmask rather than from the tag being non-zero —
            // otherwise every post-flush refill would be miscounted as a
            // conflict.
            let e = &self.core.nutlb[arr][idx];
            let occupied = self.core.nutlb_is_valid(arr, idx);
            if occupied && (e.virttag & !0xFFF) != (va & !0xFFF) {
                self.tlb.stats_nutlb_conflict(at);
            }
            self.tlb.stats_nutlb_miss(at);
        }

        #[cfg(feature = "jitstats")]
        { self.jit_last_xlat_hit = false; }
        let at = if W { AccessType::Write } else { AccessType::Read };
        let result = (self.translate_fn)(self, va, at);
        if !result.is_exception() {
            self.nutlb_fill::<W>(va, &result);
        }
        result
    }

    /// Tally one cached data access against the JIT inline fast path's
    /// predicate. Must be called *before* the cache read/write, since a miss
    /// mutates the very tag state being probed (a fill would make every
    /// access look like a tag hit after the fact).
    #[cfg(feature = "jitstats")]
    #[inline]
    fn jit_stat_probe<const STORE: bool>(&mut self, virt_addr: u64, phys_addr: u64) {
        let inline_ok = self.cache.jit_probe::<STORE>(virt_addr, phys_addr);
        let xlat = self.jit_last_xlat_hit;
        let s = if STORE { &mut self.jit_mem_stats.store } else { &mut self.jit_mem_stats.load };
        s.total += 1;
        // Prefix counters: each stage implies the ones above it. `inline_ok`
        // already folds tag/dirty/transparent, so the intermediate rows are
        // reconstructed from it plus the translation outcome rather than
        // re-probed stage by stage.
        if xlat {
            s.nutlb_hit += 1;
            if inline_ok {
                s.tag_hit += 1;
                if STORE { s.dirty_ok += 1; }
                s.inline_ok += 1;
            }
        }
    }

    /// Fill the nutlb set for `va` from a successful translation.
    #[inline]
    fn nutlb_fill<const W: bool>(&mut self, va: u64, result: &TranslateResult) {
        use crate::cpu::mips_core::*;

        // ERL is a do-not-cache window: while it is set, KUSEG/xuseg stop
        // being TLB-mapped and become unmapped uncached identity
        // (docs/nutlb-design.md §5). Caching that would be both worthless
        // (the translation is arithmetically free) and wrong to leave behind
        // once ERL clears, since nothing distinguishes the two regimes in the
        // tag. Rare window, cold branch.
        if (self.core.cp0_status & STATUS_ERL) != 0 {
            return;
        }

        // Segment shapes we decline to cache at all — currently only a
        // 64-bit xuseg VA with a nonzero bits 63:40, which is not a valid
        // user address. Everything else is cacheable regardless of which
        // mode filled it, because privilege is enforced by flushing on
        // transition rather than by the tag.
        if !nutlb_cacheable_segment(va, self.core.is_64bit_mode()) {
            return;
        }

        let idx = nutlb_index(va);
        let arr = if W { NUTLB_WRITE } else { NUTLB_READ };
        let e = &mut self.core.nutlb[arr][idx];
        e.virttag = va & !0xFFF;
        e.phys    = (result.phys & !0xFFF) | (result.status & 0x7) as u64;
        // Tag first, then validity. The order matters only for a reader that
        // could observe the entry between the two stores; nothing does today
        // (single-threaded per core), but publishing validity last is the
        // order that stays correct if that ever changes.
        self.core.nutlb_set_valid(arr, idx);
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
        // nutlb entries are not self-describing — the tag is a bare page
        // number — so every barrier that protects the untagged nanotlb has to
        // protect them too. Hanging the flush here rather than on the ~10
        // individual call sites means a new barrier can never be added
        // without covering both structures.
        //
        // This fires on every exception, ERET, mode switch and ASID change,
        // and each one discards up to 256 live translations. That cost is
        // real and was measured against the alternative (self-describing tags
        // that survive the transition); the flushes won. See
        // docs/nutlb-design.md §12.
        self.core.nutlb_clear();
        #[cfg(feature = "jitv2")]
        { self.clear_pcp(); }
    }

    /// `jitv2_smc_check`: report a data write that lands on the physical page
    /// the CPU is currently fetching/compiling from — self-modifying code
    /// hitting live compiled code.
    ///
    /// Called from `write_data_impl` *after* translation (the physical
    /// address is the whole point — a code page is identified by its frame,
    /// never by VA, and the writer's virtual mapping is usually a completely
    /// different alias from the fetcher's) and *before* the write commits, so
    /// the report names the state that produced it.
    ///
    /// Reports only, never invalidates: this exists to answer "does this
    /// actually happen, and where", not to fix it. jitv2's real invalidation
    /// path is elsewhere.
    ///
    /// **Blind spot worth stating:** the JIT's inline store fast path
    /// (`codegen::emit_mem_write_split`) writes through the L1-D data array
    /// directly and never enters `write_data_impl`, so a store that takes it
    /// is invisible here. Run with `j2 inline_mem off` for a complete picture
    /// — otherwise a clean result means
    /// "no SMC among the writes that took the slow path", which is a much
    /// weaker statement than "no SMC".
    ///
    /// Deduplicated on `(pc, target pfn)`: a memset or a loader relocating a
    /// whole page would otherwise emit one line per store. The set is capped
    /// and cleared wholesale when it grows past the cap, which is fine for a
    /// diagnostic — worst case a long run re-reports a pair it already showed.
    #[cfg(all(feature = "jitv2", feature = "jitv2_smc_check"))]
    #[cold]
    #[inline(never)]
    fn smc_check_report(&mut self, virt_addr: u64, phys_addr: u64, size: usize, pfn: u32) {
        const DEDUP_CAP: usize = 4096;
        // Counted before dedup: the total is "how many SMC writes happened",
        // which is what tells you the instrumentation fired at all. A zero
        // here after a run that should have tripped it means broken wiring,
        // not absence of SMC — verify the counter moves before trusting a
        // clean result.
        SMC_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let key = (self.core.pc, pfn);
        if self.smc_seen.len() >= DEDUP_CAP {
            self.smc_seen.clear();
        }
        if !self.smc_seen.insert(key) {
            return;
        }
        let page_base = pfn as u64 * crate::cpu::jitv2::PAGE_SIZE as u64;
        let word = ((phys_addr - page_base) / 4) as usize;
        eprintln!(
            "jitv2 SMC: write to the page being executed — pc={:#018x} wrote {} byte(s) \
va={:#018x} phys={:#010x} (code pfn {:#x}, page {:#010x}, word {}/{})",
            self.core.pc, size, virt_addr, phys_addr, pfn, page_base, word,
            crate::cpu::jitv2::ENTRIES_PER_PAGE,
        );
    }

    /// `jitv2_smc_check` hot-path test: does this write's physical address
    /// fall inside the code page currently being fetched from? Inlined and
    /// branch-free apart from the compare so a build with the feature on is
    /// still usable; all the work lives in the `#[cold]`
    /// [`Self::smc_check_report`].
    ///
    /// `cur_code_pfn == u32::MAX` (no page tracked) can never match a real
    /// frame, so the "nothing tracked" case needs no separate test.
    #[cfg(all(feature = "jitv2", feature = "jitv2_smc_check"))]
    #[inline(always)]
    fn smc_check_write(&mut self, virt_addr: u64, phys_addr: u64, size: usize) {
        let pfn = self.core.cur_code_pfn;
        if (phys_addr / crate::cpu::jitv2::PAGE_SIZE as u64) as u32 == pfn {
            self.smc_check_report(virt_addr, phys_addr, size, pfn);
        }
    }

    /// Set `self.pcp` and keep `core.cur_code_pfn` in lockstep with it.
    ///
    /// The two must never disagree: `cur_code_pfn` is what the data-write
    /// path tests against to detect self-modifying code hitting the page
    /// being executed (`smc_check_write`), and a stale mirror produces either
    /// false positives (naming a page we long since left) or silent misses.
    /// Funnelling every assignment through this pair is what makes that
    /// structural rather than a convention six call sites have to remember.
    #[cfg(feature = "jitv2")]
    #[inline]
    fn set_pcp(&mut self, page: *mut crate::cpu::jitv2::PhysicalCodePage) {
        self.pcp = page;
        self.core.cur_code_pfn = if page.is_null() {
            u32::MAX
        } else {
            unsafe { (*page).pfn }
        };
    }

    /// Drop the currently tracked code page — the `null` counterpart to
    /// [`Self::set_pcp`], kept as its own name so the intent reads at the
    /// call sites (nutlb clear, arena flush) rather than as a bare null store.
    #[cfg(feature = "jitv2")]
    #[inline]
    fn clear_pcp(&mut self) {
        self.set_pcp(std::ptr::null_mut());
    }

    /// Flush the page pool and compiled code from this CPU: every other CPU
    /// sharing them parks first, and none of the locks it may be waiting on is
    /// held meanwhile. `only_if_exhausted`: the CPU ahead of this one in line
    /// may have made the flush unnecessary.
    #[cfg(feature = "jitv2")]
    pub(crate) fn jitv2_flush_from_here(&mut self, only_if_exhausted: bool) {
        let bus = self.sysad.clone();
        let jit = self.jitv2.clone();
        let flush = || {
            let mut j = jit.lock();
            if !only_if_exhausted || j.pool_exhausted() {
                unsafe { j.flush_from_cpu_thread(bus); }
            }
        };
        match self.jitv2_world.clone() {
            Some(world) => world.stop_others(flush),
            None => flush(),
        }
    }

    #[cfg(feature = "jitv2")]
    fn jitv2_share(&mut self, jit: Arc<Mutex<crate::cpu::jitv2::Jitv2>>, world: Arc<crate::cpu::smp::World>) {
        {
            let j = jit.lock();
            self.jitv2_compile_queue_handle = j.compile_queue.queue_handle();
            self.jitv2_stats = j.stats.clone();
        }
        self.jitv2 = jit;
        self.jitv2_world = Some(world);
        self.jitv2_bind_fast_lookup();
    }

    /// Capture the page pool's array base pointers for the lock-free
    /// `page_for` fast path (`jitv2_pfn_map`/`jitv2_pages_base`).
    #[cfg(feature = "jitv2")]
    pub fn jitv2_bind_fast_lookup(&mut self) {
        let (map, pages) = self.jitv2.lock().fast_lookup_ptrs();
        self.jitv2_pfn_map = map;
        self.jitv2_pages_base = pages;
    }

    /// Lock-free `pfn -> *mut PhysicalCodePage`, or `None` on a miss (which
    /// the caller must then resolve under the lock, since claiming mutates
    /// the free list).
    #[cfg(feature = "jitv2")]
    #[inline(always)]
    fn jitv2_lookup_page_fast(&self, pfn: u32) -> Option<*mut crate::cpu::jitv2::PhysicalCodePage> {
        if self.jitv2_pfn_map.is_null() {
            return None;
        }
        let slot = unsafe { (*self.jitv2_pfn_map.add(pfn as usize)).load(Ordering::Acquire) };
        if slot == crate::cpu::jitv2::jitv2::PFN_MAP_EMPTY {
            return None;
        }
        Some(unsafe { self.jitv2_pages_base.add(slot as usize) })
    }

    /// JIT v2: re-derive `self.pcp` if the fetch just landed on a different physical
    /// page than the one currently tracked (clanker/jitv2/jit-v2-design.md §2.1 — PCPs
    /// are keyed by physical frame, never by VA). Cheap on the common case: same-page
    /// sequential/loop execution never touches the pool or its lookup map, only the
    /// nanotlb hit path above and a single PFN comparison here.
    #[cfg(feature = "jitv2")]
    #[inline(always)]
    fn jitv2_track_pcp(&mut self, phys_addr: u64) {
        // Code above the JIT's pfn map gets no page and runs in the
        // interpreter (`step_jit` checks for the null page).
        let Some(pfn) = crate::cpu::jitv2::jitv2::pfn_of(phys_addr) else {
            self.clear_pcp();
            return;
        };
        let same_page = !self.pcp.is_null() && unsafe { (*self.pcp).pfn == pfn };
        if same_page {
            return;
        }
        // Diagnostic (`j2 pagewb [on|off]`, default off): write back any
        // dirty L1-D/L2 lines covering the page we're about to fetch/compile
        // from, phys-only (no known virt_addr here, and none needed — see
        // MipsCache::writeback's doc comment on VIPT alias coverage). Chases
        // the suspicion that relocated/self-modified executable code can
        // still be sitting dirty in cache when a fresh physical page is
        // handed to the JIT for compilation, so the compiler snapshots stale
        // RAM bytes instead of what the program actually just wrote. Placed
        // on the page-crossing path (not every fetch) to keep the common
        // same-page case free of this cost even when the toggle is on.
        if jitv2_pagewb_enabled() {
            let page_base = pfn * crate::cpu::jitv2::PAGE_SIZE;
            self.cache.writeback(None, page_base as u64, crate::cpu::jitv2::PAGE_SIZE as u64);
        }
        // A page crossing landing exactly on word 0 (the exception/TLB-refill
        // vector case included — `deliver_exception` in mips_core.rs writes
        // `core.pc` straight to a fixed vector address, no reason to know
        // about jit_trigger) is compile-worthy: it's the start of a region
        // and worth probing/requesting fresh. Any other offset within the
        // new page is not — treating every page crossing as a trigger
        // (regardless of landing offset) was tried and reverted: it fires on
        // ordinary sequential fallthrough across a page boundary just as
        // readily as on a real region entry, which both flags whole-page
        // in-progress regions for a redundant recompile on every fallthrough
        // arrival and, in the steady state, sets up overlapping compiles of
        // the same page from two different offsets. `entry_offset == 0`
        // (checked below via `phys_addr`, before pfn/page_base rounding
        // throws the low bits away) is the actual proxy exec_decoded's gate
        // wants; `page.is_published`/`is_runnable` still handle every
        // other arrival at an already-compiled offset without needing this
        // flag at all.
        if phys_addr & (crate::cpu::jitv2::PAGE_SIZE as u64 - 1) == 0 {
            self.core.jit_trigger = true;
        }

        // Lock-free hit path: an already-claimed page needs nothing but the
        // `pfn -> slot -> page` indirection, which is two loads from arrays that
        // never move; this skips `jitv2.lock()`.
        if let Some(page) = self.jitv2_lookup_page_fast(pfn) {
            self.pcp = page;
            self.core.cur_code_pfn = pfn;
            debug_assert_eq!(unsafe { (*self.pcp).pfn }, pfn,
                "jitv2_track_pcp fast path: pfn_map[{:#x}] pointed at a slot whose own pfn is {:#x}",
                pfn, unsafe { (*self.pcp).pfn });
            return;
        }

        let page_base = pfn * crate::cpu::jitv2::PAGE_SIZE;
        // §13: live FR mode at first-arrival time, pinned into the page for
        // its whole lifetime (PhysicalCodePage::fr1's own doc comment) —
        // ignored by page_for on a lookup hit (an already-claimed page keeps
        // whatever it was pinned to).
        let fr1 = (self.core.cp0_status & crate::cpu::mips_core::STATUS_FR) != 0;

        let mut jit = self.jitv2.lock();
        let lookup = jit.page_for(pfn, page_base, self.sysad.as_ref(), fr1);
        match lookup {
            Some(slot) => {
                // Assigned directly rather than through `set_pcp`: the
                // `jit` guard is still live here, so a `&mut self` call
                // would conflict. `pfn` is the value `set_pcp` would have
                // read back out of the slot anyway — this keeps the mirror
                // exact without the borrow.
                self.pcp = jit.page_ptr(slot);
                self.core.cur_code_pfn = pfn;
                // `pfn` here is the real physical frame number (`phys_addr`
                // is post-translation, from nanotlb_translate's successful
                // result — never a virtual page) — this must match the slot
                // `page_for` just handed back, or self.pcp is now tracking
                // the wrong physical page for this fetch (the exact failure
                // shape `j2 pcp` can otherwise only reveal after the fact).
                debug_assert_eq!(unsafe { (*self.pcp).pfn }, pfn,
                    "jitv2_track_pcp: page_for({:#x}) returned a slot whose own pfn is {:#x}",
                    pfn, unsafe { (*self.pcp).pfn });
            }
            None => {
                // Pool exhausted: `flush_from_cpu_thread` resets to initial
                // state and retries — always leaves room for at least one
                // fresh page. Called from the CPU thread itself (this
                // function), which is "as good as stopped" for its own
                // purposes — no need to pause it — so this only pauses the
                // *compile* queue internally, not the CPU.
                drop(jit); // re-locks internally below; nothing needs it held across that call
                self.jitv2_flush_from_here(true);
                // Only `self.pcp` is actually stale here — it's a raw
                // pointer into the `Vec<PhysicalCodePage>` mega_flush just
                // cleared. The nanotlb's own translation slots (Fetch/Read/
                // Write) are untouched by mega_flush (physical memory didn't
                // move, only the JIT's compiled-code bookkeeping did), so
                // there's no reason to blow those away too — just null the
                // dangling pointer directly and re-derive it below for the
                // exact page this fetch already landed on.
                self.clear_pcp();
                let mut jit = self.jitv2.lock();
                let slot = jit.page_for(pfn, page_base, self.sysad.as_ref(), fr1)
                    .expect("flush must leave room for at least one page");
                // Direct assignment for the same borrow reason as the
                // hit path above; `pfn` is what `set_pcp` would mirror.
                self.pcp = jit.page_ptr(slot);
                self.core.cur_code_pfn = pfn;
                debug_assert_eq!(unsafe { (*self.pcp).pfn }, pfn,
                    "jitv2_track_pcp (post-flush retry): page_for({:#x}) returned a slot whose own pfn is {:#x}",
                    pfn, unsafe { (*self.pcp).pfn });
            }
        }
    }

    /// Re-derive `translate_fn` from the current CP0 Status register.
    /// Must be called after any write to Status (done automatically via the status callback)
    /// and at init/reset time.
    #[inline]
    pub fn update_translate_fn(&mut self) {
        use crate::cpu::mips_core::PrivilegeMode;
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
        use crate::cpu::mips_core::{
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

    /// Re-derive every piece of executor state cached off CP0 Status: `translate_fn`
    /// (privilege x 32/64-bit), the six `fpr_*` accessors (STATUS_FR), and the untagged
    /// translation caches that carry no permission bits of their own (nanotlb + nutlb +
    /// the jitv2 pcp, all bundled inside `nanotlb_invalidate`).
    ///
    /// **Every write to `core.cp0_status` must be followed by this call.** `write_cp0`
    /// reg 12 gets here via `status_changed_cb` (see [`Self::install_status_cb`]); every
    /// other site — ERET, exception delivery, soft reset, snapshot/digest restore, gdb
    /// register writes — calls it directly, and must, because the callback is not
    /// installed on a directly-constructed executor.
    ///
    /// Flushing alone is *not* sufficient, which is the trap this method exists to close:
    /// the flush guarantees the next access misses and therefore calls `translate_fn` —
    /// so a stale `translate_fn` is not merely tolerated across a privilege change, it is
    /// guaranteed to be consulted.
    ///
    /// Unconditional by design — no `(old ^ new)` guard. ERET and exception delivery
    /// always toggle EXL/ERL, which `get_privilege_mode` reads, so a guard would fire on
    /// 100% of the transitions that matter while adding a branch and a mask constant to
    /// get subtly wrong. The flush it rides behind already costs far more.
    #[inline]
    pub fn resync_privilege_state(&mut self) {
        self.update_translate_fn();
        self.update_fpr_mode();
        self.nanotlb_invalidate();
        #[cfg(feature = "cache-ecc")]
        self.sync_cache_error_control();
    }

    /// Tell the cache which Status bits govern parity: CE forces the ECC
    /// register into stored check bits, and DE or ERL turn detection off.
    /// Every Status change comes through `resync_privilege_state`.
    #[cfg(feature = "cache-ecc")]
    #[inline]
    fn sync_cache_error_control(&self) {
        let s = self.core.cp0_status;
        self.cache.set_error_control(s & STATUS_CE != 0, s & (STATUS_DE | STATUS_ERL) == 0);
        self.cache.set_cache_ecc(self.core.cp0_ecc);
    }

    /// Store a new CP0 Status word and re-derive everything cached off it.
    ///
    /// For callers that have the whole word (snapshot restore, digest restore, gdb).
    /// Callers doing read-modify-write on the field (ERET, reset) set it themselves and
    /// call [`Self::resync_privilege_state`] directly.
    #[inline]
    pub fn set_cp0_status(&mut self, new: u32) {
        self.core.cp0_status = new;
        self.resync_privilege_state();
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
        self.jitv2_track_pcp(self.core.pc & 0xFFFF_FFFF);
        self.ins.raw = instr;
        self.ins.flags = FLAG_NOT_DECODED;
        decode_into::<T, C>(&mut self.ins);
        let d: *const DecodedInstr = &self.ins;
        self.exec_decoded_int(unsafe { &*d })
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
            // A register-sourced target can be misaligned. This is the ONLY
            // place a retiring delay slot installs it, and it is already past
            // the slot's side effects — which is exactly the architectural
            // ordering (see `check_pc_alignment`).
            //
            // Ordered *before* `jit_trigger` so a faulting transfer doesn't
            // leave the trigger set for a target that never became live. The
            // exception's own vector is reached through `deliver_exception`,
            // which is a fresh arrival the normal machinery already handles.
            if (self.core.pc & 0x3) != 0 {
                return self.misaligned_fetch_exception();
            }
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

    /// AdEL for a misaligned PC, delivered as a real exception.
    ///
    /// Split out (and `#[cold]`) so the aligned path of
    /// `handle_exec_complete`/`exec_complete_pc_set` stays a single test and
    /// branch, with no exception machinery inlined into it.
    #[cold]
    #[inline(never)]
    fn misaligned_fetch_exception(&mut self) -> ExecStatus {
        self.core.cp0_badvaddr = self.core.pc;
        self.handle_exception(exec_exception(EXC_ADEL))
    }

    /// A misaligned instruction address raises **AdEL** — R4400 UM §3, and it
    /// is a fetch-side fault, not a fault of the instruction that produced the
    /// address. Without this check a misaligned target fetched and executed the
    /// word *containing* the address, silently running an instruction the guest
    /// never branched to. cpucritique C1.
    ///
    /// **Where the check lives, and why not in `fetch_instr_impl`.** Testing
    /// every fetch would be correct but pays on the hottest path in the
    /// emulator for a condition only a handful of instructions can create. PC
    /// only becomes misaligned when something installs a *register-sourced*
    /// address:
    ///
    /// | how PC is set | can be misaligned? |
    /// |---|---|
    /// | `pc += 4` (sequential, every ordinary instruction) | no — was aligned |
    /// | `pc += 8` (branch-likely nullified) | no — same |
    /// | J/JAL, all B* (PC-relative or immediate `<< 2`) | no — low 2 bits are structurally 0 |
    /// | **JR / JALR** (`read_gpr`) | **yes** |
    /// | **ERET** (`cp0_epc` / `cp0_errorepc`) | **yes** |
    ///
    /// So exactly two sources, arriving through three terminals:
    ///
    /// 1. `handle_exec_complete`'s **delay-slot branch only** — a retiring slot
    ///    installing `delay_slot_target`. This is how plain JR/JALR arrive. The
    ///    sequential `pc += 4` arm needs no check (PC was already aligned), so
    ///    ordinary instructions pay nothing.
    /// 2. `exec_complete_pc_set` — a direct `core.pc = target`. Of its callers
    ///    only the fused `jr_nop` is register-sourced; the rest (`j_nop`,
    ///    `jal_nop`, fused `beq`/`bne`) are PC-relative or immediate and
    ///    structurally aligned.
    /// 3. `exec_eret` — open-codes its own completion rather than calling
    ///    `exec_complete_pc_set`, so it needs the check separately.
    ///
    /// Those three cover every case and cost one `test`+`jcc`, on paths that
    /// were already doing branchy PC arithmetic. Compare the alternative of
    /// testing inside `fetch_instr_impl`: equally correct, but it pays on the
    /// hottest path in the emulator for a condition two instruction forms can
    /// create.
    ///
    /// **The ordering matters and this placement is what gets it right.** The
    /// fault belongs on the fetch of the target, *after* the delay slot has
    /// executed: `exec_jr` deliberately does not validate its target, it just
    /// calls `branch_delay`, so the slot retires normally and only then does
    /// the bad PC become live. Putting the check inside `exec_jr` instead would
    /// skip the slot's architecturally visible side effect. Both terminals sit
    /// after the slot, so both are correct by construction.
    ///
    /// EPC therefore lands on the misaligned target (not on the JR), BD is
    /// clear, and BadVAddr holds the misaligned address.
    ///
    /// Returns `EXEC_COMPLETE` when the PC is aligned — i.e. callers can
    /// `return` it unconditionally, since that is exactly what they would have
    /// returned anyway. No `Option`: the status word already encodes "nothing
    /// happened", so wrapping it only adds a discriminant to branch on.
    #[inline(always)]
    fn check_pc_alignment(&mut self) -> ExecStatus {
        if (self.core.pc & 0x3) != 0 {
            self.misaligned_fetch_exception()
        } else {
            EXEC_COMPLETE
        }
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
    /// is the real payload here. (The one exception is a misaligned
    /// register-sourced PC, which `check_pc_alignment` turns into AdEL.)
    #[inline(always)]
    fn exec_complete_pc_set(&mut self) -> ExecStatus {
        #[cfg(feature = "jitv2")]
        { self.core.jit_trigger = true; }
        // ERET and the fused `jr_nop` install a register-sourced PC through
        // here; see `check_pc_alignment` for why this is one of only two
        // places that needs the test.
        self.check_pc_alignment()
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

    /// `fetchverify`, interpreter half: compare the cached decoded word
    /// against what memory holds at `pc`.
    ///
    /// The interpreter caches decoded instructions per I-cache line and only
    /// re-decodes when `d.flags != 0`, so a cached slot can outlive a change to
    /// the underlying memory — the same staleness the JIT can have, reached a
    /// different way. Deliberately checked here rather than inside `fetch`, so
    /// it sees exactly the word that is about to be dispatched.
    ///
    /// Reads via `debug_translate` + a raw bus read so the check perturbs
    /// nothing. A VA that no longer translates is skipped rather than reported
    /// (the mapping legitimately changed; the next real fetch will fault).
    #[cfg(feature = "fetchverify")]
    pub fn fetch_verify_interp(&mut self, pc: u64, cached: u32) -> ExecStatus {
        if !self.core.fetch_verify_on {
            return EXEC_COMPLETE;
        }
        FETCH_VERIFY_CHECKS.fetch_add(1, Ordering::Relaxed);
        let tr = self.debug_translate(pc);
        if tr.is_exception() {
            return EXEC_COMPLETE;
        }
        let r = self.sysad.read32(tr.phys);
        if !r.is_ok() || r.data == cached {
            return EXEC_COMPLETE;
        }
        FETCH_VERIFY_MISMATCHES.fetch_add(1, Ordering::Relaxed);
        let symbols = self.symbols.lock();
        let sym = format_pc_symbol(pc, &symbols);
        eprintln!(
            "\n=== STALE INTERPRETER DECODE at {:#018x}{} phys={:#010x} ===\n  \
             cached: {:08x}  {}\n  now in memory: {:08x}  {}\n",
            pc, sym, tr.phys,
            cached, mips_dis::disassemble(cached, pc, Some(&symbols)),
            r.data, mips_dis::disassemble(r.data, pc, Some(&symbols)));
        drop(symbols);
        self.core.pc = pc;
        EXEC_BREAKPOINT
    }

    pub fn step_int(&mut self) -> ExecStatus {
        step_preamble!(self);
        let pc = self.core.pc;

        let fetch = self.fetch_instr(pc);
        let result = if fetch.status == EXEC_COMPLETE {
            {
                let slot = fetch.instr as *mut DecodedInstr;
                let d = unsafe { &mut *slot };
                if d.flags != 0 {
                    decode_into::<T, C>(d);
                } else {
                    #[cfg(feature = "developer")]
                    self.decoded_count.fetch_add(1, Ordering::Relaxed);
                }
            }
            let d = unsafe { &*fetch.instr };
            // `fetchverify`: the interpreter half. `d.raw` is the *cached*
            // decoded word — `decode_into` only re-runs when `d.flags != 0`,
            // so a slot can legitimately survive a memory change. Verify it
            // against memory before dispatching.
            #[cfg(feature = "fetchverify")]
            if self.fetch_verify_interp(pc, d.raw) == EXEC_BREAKPOINT {
                return EXEC_BREAKPOINT;
            }
            #[cfg(not(feature = "lightning"))]
            self.traceback.push(pc, d.raw, InstrOrigin::Interp);
            self.exec_decoded_int(d)
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

    /// JIT-aware step: fetch, then run the real jitv2 dispatch gate inline
    /// (compile scheduling, inline compile, JIT entry). Deliberately does
    /// NOT run `step_preamble!` itself: cycles/timer/interrupt/breakpoint
    /// handling for a compiled-code hit comes from the analyzer's own
    /// inferred `cycles_delta` and the compiled code's own inline
    /// `emit_pending_interrupt_preamble` check, not from a Rust-side
    /// preamble paid on every single call regardless of whether anything hit.
    /// `step_preamble!` only runs via a fallback exit calling `step_int()`
    /// directly — exactly when something (a gate miss, a pending interrupt
    /// caught inline) actually needs the interpreter's real preamble.
    /// Full dispatch-gate rationale: clanker/jitv2/jit-v2-design.md §13.
    #[cfg(feature = "jitv2")]
    pub fn step_jit(&mut self) -> ExecStatus {
        let pc = self.core.pc;

        let fetch = self.jit_entry_fetch(pc);
        // Retired AFTER fetch_instr, not before: fetch_instr's own
        // nanotlb_translate is what runs jitv2_track_pcp, the thing that
        // sets `core.jit_trigger = true` on a fresh page's first arrival.
        // Reading/clearing it earlier captured a stale `false` and lost
        // that signal (found live: a from-scratch executor's very first
        // step_jit() call never triggered a compile at all).
        let trigger = self.core.jit_trigger;
        self.core.jit_trigger = false;
        let result = if fetch.status == EXEC_COMPLETE && self.pcp.is_null() {
            // Fetched from above the JIT's pfn map (`jitv2_track_pcp`).
            self.step_int()
        } else if fetch.status == EXEC_COMPLETE {
            // Shared: another CPU may be on this page too.
            let page = unsafe { &*self.pcp };
            let entry_offset = ((pc & 0xFFF) >> 2) as usize;
            'gate: {
            if trigger || page.is_published(entry_offset) {
                if page.is_runnable(entry_offset) {
                    let func = page.func();
                    debug_assert!(!func.is_null(), "valid bit set with null func");
                    #[cfg(feature = "developer")]
                    { page.call_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
                    let jit_fn: crate::cpu::jitv2::JitFn = unsafe { std::mem::transmute(func) };
                    let status = unsafe { jit_fn(&mut self.core as *mut MipsCore) };
                    break 'gate if status != EXEC_FALLBACK { status } else { self.step_int() };
                }
                // A denylisted offset has nothing to compile against these
                // bytes: `prepare` masks denied offsets out of every
                // candidate set. Requesting anyway (a fresh arrival at a
                // kernel entry the JIT excludes happens on every exception)
                // cost a full page snapshot per arrival, and kept a page
                // whose FR pin and installed code disagreed recompiling in a
                // loop. It also moved the page's FR pin for a compile that
                // could not use it. Once the bytes change (`entry_gen` behind
                // `current_gen`), the next publish resets `denied`, so the
                // arrival asks again as before.
                if page.is_denylisted(entry_offset) && page.entry_gen() == page.current_gen() {
                    break 'gate self.step_int();
                }
                let inline_compile = self.jitv2_inline_compile;
                self.core.jit_trigger = false;
                // Must mark requested before building `req` —
                // CompileRequest carries no offset, so whichever side
                // compiles reads `requested` fresh (§13.3).
                page.mark_requested(entry_offset);
                let live_fr1 = (self.core.cp0_status & crate::cpu::mips_core::STATUS_FR) != 0;
                // Pinned FR per physical page (one function per page,
                // FR baked into every FPR emitter), consuming any outstanding
                // correction first.
                let compiled_for_fr1 = page.take_fr_repin(live_fr1);
                let req = crate::cpu::jitv2::CompileRequest {
                    page: self.pcp,
                    compiled_for_fr1,
                };
                if inline_compile {
                    break 'gate self.jitv2_compile_inline(req, trigger);
                } else {
                    #[cfg(feature = "developer")]
                    page.mark_schedule_attempt();
                    if page.try_schedule_page() {
                        #[cfg(feature = "developer")]
                        page.mark_send_attempted();
                        // A dropped push (queue full) must clear the
                        // in-flight flag itself — handle_request is the
                        // only other clearer, and it never runs for a
                        // request that never reached the queue
                        // (confirmed live: starves the page of compiles
                        // forever otherwise).
                        if !crate::cpu::jitv2::jitv2::push_compile_request(&self.jitv2_compile_queue_handle, req, &self.jitv2_stats) {
                            crate::pf_level!(crate::perfetto::events::JIT_QUEUE_FULL_LEVEL, crate::perfetto::events::JIT_QUEUE_FULL, true, pc);
                            page.clear_scheduled();
                            #[cfg(feature = "developer")]
                            page.mark_send_dropped_queue_full();
                        } else {
                            crate::pf_level!(crate::perfetto::events::JIT_QUEUE_FULL_LEVEL, crate::perfetto::events::JIT_QUEUE_FULL, false);
                            crate::pf_instant!(crate::perfetto::events::JIT_REQUEST, pc);
                        }
                    } else if page.note_waiting_arrival() {
                        // Already queued, and being executed hard while it
                        // waits: jump the queue (jitv2::HOT_QUEUE).
                        crate::pf_instant!(crate::perfetto::events::JIT_HOT_REQUEST, pc);
                        crate::cpu::jitv2::jitv2::push_hot_request(req);
                    }
                    break 'gate self.step_int();
                }
            }
            self.step_int()
            }
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

    /// `step_jit`'s fetch at `pc`. Entering compiled code needs the page
    /// (`self.pcp`) and the offset, not the instruction, and every other way
    /// out of the gate (`step_int`, an exception from it) fetches for itself.
    /// So when the cache's fetch changes nothing (`MipsCache::fetch_is_pure`)
    /// and no breakpoint can match (a `lightning` build), only the
    /// translation runs: it is what raises a fetch's TLB exceptions and moves
    /// `self.pcp` onto a new page. A cacheable page then needs no cache
    /// access; an uncached one goes to the bus as before, its translation
    /// already in the slot. Profiled, the cache fetch was most of the cost of
    /// entering compiled code, which calls, syscalls and a GL driver's inner
    /// loop all do constantly.
    #[cfg(feature = "jitv2")]
    #[inline(always)]
    fn jit_entry_fetch(&mut self, pc: u64) -> FetchInstrResult {
        if !(cfg!(feature = "lightning") && self.cache.fetch_is_pure()) {
            return self.fetch_instr(pc);
        }
        let t = self.nanotlb_translate::<{ AccessType::Fetch as u8 }>(pc);
        if t.is_exception() {
            FetchInstrResult::exception(t.status)
        } else if t.is_cached() {
            FetchInstrResult::hit(std::ptr::null())
        } else {
            self.fetch_instr(pc)
        }
    }

    /// Begin recording a per-instruction execution trace to `path`
    /// (`src/cpu/trace.rs`, clanker/jitv2's lockstep verification tooling).
    /// Overwrites any existing file at `path`. No-op-safe to call while
    /// already recording — starts a fresh file, silently dropping (not
    /// flushing) whatever writer was active before, so callers that want a
    /// clean handoff should `trace_stop` first.
    #[cfg(feature = "developer")]
    pub fn trace_start(&mut self, path: &std::path::Path) -> std::io::Result<()> {
        self.trace_writer = Some(crate::cpu::trace::TraceWriter::create(path)?);
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

    #[inline(always)]
    fn check_breakpoint<const KIND: u8>(&mut self, addr: u64) -> bool {
        if KIND == BpType::Pc as u8 {
            if self.pc_bp_count == 0 { return false; }
        } else {
            // Memory breakpoint
            if self.mem_bp_count == 0 { return false; }
        }

        let mut hit = false;
        for bp in &mut self.breakpoints {
            if bp.enabled && bp.kind as u8 == KIND {
                // Normalize kseg0/kseg1 to physical so a bp on physical
                // 0x1fbb0010 hits through kseg0 (0x9fbb0010) or kseg1
                // (0xbfbb0010), then word-align. Never mask below bit 2.
                if bp_match_key(bp.addr) == bp_match_key(addr) {
                    let mut detail = None;
                    // Check optional register condition
                    if let Some(expr) = &bp.condition {
                        let symbols = self.symbols.lock();
                        match expr.eval(&self.core, Some(&symbols)) {
                            Ok(val) => if val == 0 { continue; },
                            // `bp add` refuses conditions that cannot resolve,
                            // so this is a runtime failure (division by zero):
                            // stop, rather than hide the breakpoint forever.
                            Err(e) => detail = Some(format!("condition failed to evaluate: {}", e)),
                        }
                    }
                    bp.hits += 1;
                    if bp.ignore > 0 {
                        bp.ignore -= 1;
                        continue;
                    }
                    if KIND != BpType::Pc as u8 {
                        let access = format!("access to {:#x}", addr);
                        detail = Some(match detail { Some(d) => format!("{}; {}", access, d), None => access });
                    }
                    self.bp_hit_detail = detail;
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

    /// What `last_bp_hit` was, as a sentence for the stop note.
    fn breakpoint_hit_reason(&self) -> String {
        match self.last_bp_hit {
            Some(0) => "reached the run-until address".to_string(),
            Some(id) => match self.breakpoints.iter().find(|b| b.id == id) {
                Some(bp) => {
                    let symbols = self.symbols.lock();
                    let mut s = format!("breakpoint {} (hit {} time{})",
                            format_bp_line(bp.id, bp.addr, bp.kind, bp.enabled, bp.condition_src.as_deref(), &symbols),
                            bp.hits, if bp.hits == 1 { "" } else { "s" });
                    if let Some(e) = &self.bp_hit_detail {
                        s.push_str("; ");
                        s.push_str(e);
                    }
                    s
                }
                None => format!("breakpoint {} (since deleted)", id),
            },
            None => "breakpoint".to_string(),
        }
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

        self.breakpoints.push(Breakpoint { id, addr, kind, enabled: true, condition: None, condition_src: None,
                                           hits: 0, ignore: 0 });
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
    /// IP28 bring-up: dump everything about the one exception whose BadVAddr
    /// is `IRIS_IP28_EXC_VADDR`.
    ///
    /// A first-N trace is useless for a fault that lands after thousands of
    /// ordinary TLB misses, and reading the register file from the monitor
    /// afterwards shows the guest's panic handler, not the fault. Costs one
    /// relaxed load per exception on the R10000 model and folds away on every
    /// other; exceptions are not a hot path.
    #[inline]
    fn ip28_trace_exception(&self, status: ExecStatus) {
        if !C::R10K_CACHE_OPS {
            return;
        }
        static WANT: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
        let want = *WANT.get_or_init(|| {
            std::env::var("IRIS_IP28_EXC_VADDR").ok().and_then(|v| {
                let v = v.trim();
                if v.is_empty() { return None; }
                u64::from_str_radix(v.trim_start_matches("0x"), 16).ok()
            })
        });
        let Some(want) = want else { return };

        // Keep the run-up. The exception that panics the guest is the second
        // one: the first is an ordinary miss, and its handler then faults.
        // Only the run-up says what the handler was handed, and BadVAddr has
        // already been overwritten by the time the match fires.
        static RING: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
        const KEEP: usize = 12;
        let line = format!(
            "code={:<2} pc={:#018x} bd={} badvaddr={:#018x} xcontext={:#018x} \
             context={:#018x} entryhi={:#018x} ra={:#018x}",
            (status & CAUSE_EXCCODE_MASK) >> 2,
            self.core.pc,
            self.core.in_delay_slot as u8,
            self.core.cp0_badvaddr,
            self.core.cp0_xcontext,
            self.core.cp0_context,
            self.core.cp0_entryhi,
            self.core.read_gpr(31),
        );
        if let Ok(mut ring) = RING.lock() {
            if ring.len() == KEEP {
                ring.remove(0);
            }
            ring.push(line);
            if want == self.core.cp0_badvaddr {
                eprintln!("ip28exc: --- last {} exceptions, oldest first ---", ring.len());
                for (i, l) in ring.iter().enumerate() {
                    eprintln!("ip28exc: [{i}] {l}");
                }
            }
        }

        if want != self.core.cp0_badvaddr {
            return;
        }
        const NAMES: [&str; 32] = [
            "zero", "at", "v0", "v1", "a0", "a1", "a2", "a3",
            "t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7",
            "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7",
            "t8", "t9", "k0", "k1", "gp", "sp", "fp", "ra",
        ];
        let code = (status & CAUSE_EXCCODE_MASK) >> 2;
        eprintln!(
            "ip28exc: code={code} pc={:#018x} delay_slot={} badvaddr={:#018x} status={:#010x}",
            self.core.pc, self.core.in_delay_slot, self.core.cp0_badvaddr,
            self.core.cp0_status,
        );
        for i in 0..32u32 {
            if self.core.read_gpr(i) == self.core.cp0_badvaddr {
                eprintln!("ip28exc:   {} (${i}) holds the bad address", NAMES[i as usize]);
            }
        }
        for chunk in (0..32u32).collect::<Vec<_>>().chunks(4) {
            let mut line = String::from("ip28exc: ");
            for &i in chunk {
                line.push_str(&format!(" {:>4}={:#018x}", NAMES[i as usize], self.core.read_gpr(i)));
            }
            eprintln!("{line}");
        }
    }

    fn handle_exception(&mut self, status: ExecStatus) -> ExecStatus {
        self.ip28_trace_exception(status);
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
            // Coprocessor Unusable. Worth breaking at the fault site: IRIX's
            // `VEC_cpfault` panics "kernel used coprocessor" whenever a CpU
            // arrives with SR_PREVMODE == 0, and it checks that *before*
            // looking at Cause.CE — so the guest message names no coprocessor,
            // and by the time it prints, the handler has rewritten Status.
            // Stopping here preserves the state that actually caused it.
            if exc_code == EXC_CPU && cpu_unusable_break_enabled() {
                let ce = (self.core.cp0_cause & CAUSE_CE_MASK) >> CAUSE_CE_SHIFT;
                eprintln!("COPROCESSOR UNUSABLE (CE={}) at PC={:#018x} EPC={:#018x} Status={:#010x} Cause={:#010x} priv={:?}",
                    ce, self.core.pc, epc, self.core.cp0_status, self.core.cp0_cause,
                    self.core.get_privilege_mode());
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

        // Every other exception delivered before a syscall's matching ERET
        // (a fault inside the syscall handler, or an interrupt landing on
        // top of it) clears syscall_pending: that ERET returns into the
        // handler, not out of the syscall, so the eventual real
        // syscall-return ERET must no longer count as one either —
        // deliberately conservative (misses a trigger) rather than firing on
        // the wrong ERET. See MipsCore::syscall_pending's doc comment and
        // handle_exception_syscall (the EXC_SYS-only counterpart of this
        // function, which sets it instead).
        #[cfg(feature = "jitv2")]
        { self.core.syscall_pending = false; }

        if status & EXEC_IS_CACHE_ERR != 0 {
            self.core.cp0_cacheerr = self.cache.take_cache_err();
        }
        // Architectural effect (Cause/EPC/Status/vector) — the portable part
        // shared with jitv2-verify (§4.2 single-implementation delivery).
        crate::cpu::mips_core::deliver_exception(&mut self.core, status);

        // deliver_exception forces kernel privilege by setting EXL directly on
        // cp0_status, without going through write_cp0, so this is the only barrier on
        // the path — and it must re-derive translate_fn, not just flush. The handler
        // vector is in KSEG0/KSEG1, which only the kernel translator will accept.
        self.resync_privilege_state();
        // `deliver_exception` clears in_delay_slot itself — see its doc comment.
        status
    }

    /// [`Self::handle_exception`] with the EPC/BD inputs passed explicitly
    /// rather than read from `core.pc`/`core.in_delay_slot` — see
    /// [`crate::cpu::mips_core::deliver_exception_at`] for why compiled code wants
    /// this (it knows both as compile-time constants and would otherwise
    /// have to store them to memory purely to have them read straight back).
    ///
    /// Identical in every other respect, including the `developerx`
    /// bus/address-error monitor break, the LLBit clear, `syscall_pending`,
    /// the nutlb flush, and the trailing `in_delay_slot = false`.
    #[cfg(feature = "jitv2")]
    fn handle_exception_at(&mut self, status: ExecStatus, fault_pc: u64, bd: bool) -> ExecStatus {
        #[cfg(feature = "developerx")]
        {
            let was_exl = (self.core.cp0_status & STATUS_EXL) != 0;
            let epc = if was_exl {
                self.core.cp0_epc
            } else if bd {
                fault_pc.wrapping_sub(4)
            } else {
                fault_pc
            };
            let exc_code = (status & CAUSE_EXCCODE_MASK) >> 2;
            if exc_code == EXC_IBE || exc_code == EXC_DBE {
                eprintln!("BUS ERROR ({}) at PC={:#010x} EPC={:#010x}",
                    if exc_code == EXC_IBE { "IBE" } else { "DBE" },
                    fault_pc, epc);
                return EXEC_BREAKPOINT;
            }
            if exc_code == EXC_ADEL || exc_code == EXC_ADES {
                eprintln!("ADDRESS ERROR ({}) at PC={:#010x} EPC={:#010x} BadVAddr={:#010x}",
                    if exc_code == EXC_ADEL { "ADEL" } else { "ADES" },
                    fault_pc, epc, self.core.cp0_badvaddr);
                return EXEC_BREAKPOINT;
            }
            // Coprocessor Unusable. Worth breaking at the fault site: IRIX's
            // `VEC_cpfault` panics "kernel used coprocessor" whenever a CpU
            // arrives with SR_PREVMODE == 0, and it checks that *before*
            // looking at Cause.CE — so the guest message names no coprocessor,
            // and by the time it prints, the handler has rewritten Status.
            // Stopping here preserves the state that actually caused it.
            if exc_code == EXC_CPU && cpu_unusable_break_enabled() {
                let ce = (self.core.cp0_cause & CAUSE_CE_MASK) >> CAUSE_CE_SHIFT;
                // `fault_pc`, NOT `core.pc`: on this (jitv2) path compiled code
                // supplies the faulting address explicitly and may not have
                // materialized `core.pc` yet, so reading the field would report
                // a stale or unrelated PC.
                eprintln!("COPROCESSOR UNUSABLE (CE={}) at PC={:#018x} EPC={:#018x} Status={:#010x} Cause={:#010x} priv={:?} [jit]",
                    ce, fault_pc, epc, self.core.cp0_status, self.core.cp0_cause,
                    self.core.get_privilege_mode());
                return EXEC_BREAKPOINT;
            }
            if (exc_code == EXC_TLBL || exc_code == EXC_TLBS) && (self.core.cp0_badvaddr as u32 == 0xFF800000) {
                eprintln!("ADDRESS ERROR ({}) at PC={:#010x} EPC={:#010x} BadVAddr={:#010x}",
                    if exc_code == EXC_TLBL { "TLBL" } else { "TLBS" },
                    fault_pc, epc, self.core.cp0_badvaddr);
                return EXEC_BREAKPOINT;
            }
        }

        self.cache.set_llbit(false);
        self.core.syscall_pending = false;

        // NOTE: deliberately does NOT touch `cp0_badvaddr`. An earlier revision
        // set it from `fault_pc` for AdEL/AdES on the theory that the JIT's
        // data-path address errors never reach here — they do. A misaligned
        // load/store branches to `slow_block`, re-runs in Rust where
        // `read_data_impl`/`write_data_impl` set BadVAddr to the faulting
        // *data* address, and then arrives here with `fault_pc` = the
        // instruction address. Writing it clobbered the right value with the
        // wrong one; the `ad{e,s}l_*_matches_interpreter` equivalence tests
        // caught it as a one-field divergence. Every caller that needs
        // BadVAddr sets it before calling.
        if status & EXEC_IS_CACHE_ERR != 0 {
            self.core.cp0_cacheerr = self.cache.take_cache_err();
        }
        crate::cpu::mips_core::deliver_exception_at(&mut self.core, status, fault_pc, bd);
        self.resync_privilege_state();
        status
    }

    /// `handle_exception`'s exact twin, used only by `exec_syscall`: sets
    /// `syscall_pending` instead of clearing it (see that field's doc
    /// comment). A dedicated copy rather than a parameter on the shared
    /// function because every other call site (~40 of them: TLB faults,
    /// overflow, trap, address error, FPU exceptions, the JIT's own
    /// exception callback...) is unconditionally the non-syscall case and
    /// shouldn't carry a branch or an argument for a distinction that never
    /// varies for them.
    #[cfg(feature = "jitv2")]
    fn handle_exception_syscall(&mut self, status: ExecStatus) -> ExecStatus {
        #[cfg(feature = "developerx")]
        {
            let was_exl = (self.core.cp0_status & STATUS_EXL) != 0;
            let epc = if was_exl {
                self.core.cp0_epc
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
            // Coprocessor Unusable. Worth breaking at the fault site: IRIX's
            // `VEC_cpfault` panics "kernel used coprocessor" whenever a CpU
            // arrives with SR_PREVMODE == 0, and it checks that *before*
            // looking at Cause.CE — so the guest message names no coprocessor,
            // and by the time it prints, the handler has rewritten Status.
            // Stopping here preserves the state that actually caused it.
            if exc_code == EXC_CPU && cpu_unusable_break_enabled() {
                let ce = (self.core.cp0_cause & CAUSE_CE_MASK) >> CAUSE_CE_SHIFT;
                eprintln!("COPROCESSOR UNUSABLE (CE={}) at PC={:#018x} EPC={:#018x} Status={:#010x} Cause={:#010x} priv={:?}",
                    ce, self.core.pc, epc, self.core.cp0_status, self.core.cp0_cause,
                    self.core.get_privilege_mode());
                return EXEC_BREAKPOINT;
            }
            if (exc_code == EXC_TLBL || exc_code == EXC_TLBS) && (self.core.cp0_badvaddr as u32 == 0xFF800000) {
                eprintln!("ADDRESS ERROR ({}) at PC={:#010x} EPC={:#010x} BadVAddr={:#010x}",
                    if exc_code == EXC_TLBL { "TLBL" } else { "TLBS" },
                    self.core.pc, epc, self.core.cp0_badvaddr);
                return EXEC_BREAKPOINT;
            }
        }

        self.cache.set_llbit(false);
        self.core.syscall_pending = true;
        crate::cpu::mips_core::deliver_exception(&mut self.core, status);
        self.resync_privilege_state();
        status
    }

    // Helper to get CPU's current addressing mode
    #[inline]
    fn is_64bit(&self) -> bool {
        self.core.is_64bit_mode()
    }

    /// Returns whether a virtual address would use the XTLB (64-bit) vector on a TLB miss.
    /// Mirrors the xtlb flag logic in translate_32/64bit_impl exactly.
    ///   - 32-bit mode (KX=0): always false — all TLB segments use the UTLB vector.
    ///   - 64-bit mode (KX=1): true for every mapped segment, including the
    ///     32-bit-shaped kseg3/ksseg compat range (`0xFFFF_FFFF_xxxx_xxxx`).
    ///
    /// The vector follows the *mode*, not the address shape: R4400 User Manual
    /// §5 selects XTLB whenever the relevant X-bit (KX/SX/UX) is set for the
    /// mode the miss was taken in. An earlier version returned false for the
    /// compat range, which sent a KX=1 kernel's cksseg miss to the 32-bit
    /// refill vector. cpucritique.md T1.
    ///
    /// xkphys (top=2) is unmapped and never reaches the TLB, so it keeps
    /// returning false; the only caller that can pass it is `exec_tlbp`, which
    /// is asking about an EntryHi value rather than a real access.
    #[inline]
    fn is_xtlb_address(&self, virt_addr: u64) -> bool {
        self.core.is_64bit_mode() && (virt_addr >> 62) != 2
    }

    /// Decode a 3-bit cache-coherency attribute (EntryLo's C field, or
    /// `Config.K0`, which uses the same encoding) into a `TR_*` value.
    ///
    /// **The two parts have genuinely different tables — this is not one table
    /// with reserved holes.** Verified against the manuals in `docs/`:
    ///
    /// | C | R4400 (`R4000_um2.pdf` Table 4-6) | VR5000 (`VR5000 User Manual.pdf` Table 6-6) |
    /// |---|---|---|
    /// | 0 | Reserved | Cacheable noncoherent, **write-through, no write allocate** |
    /// | 1 | Reserved | Cacheable noncoherent, **write-through, write allocate** |
    /// | 2 | **Uncached** | **Uncached** |
    /// | 3 | **Cacheable noncoherent** | **Cacheable noncoherent** |
    /// | 4 | Cacheable coherent exclusive | Reserved |
    /// | 5 | Cacheable coherent exclusive on write | Reserved |
    /// | 6 | Cacheable coherent update on write | Reserved |
    /// | 7 | Reserved | Reserved |
    ///
    /// Note 4/5/6 are **coherent** variants on R4400 — not write-through. The
    /// write-through modes are R5000's 0/1, i.e. the encodings R4400 reserves.
    /// Getting these backwards maps every R5000 write-through page to uncached
    /// and every R4400 coherent page to cacheable-with-the-wrong-attribute.
    ///
    /// **Only 2 and 3 matter in practice**: those are what the PROM flips
    /// between around cache work (measured: five K0 changes over a full IRIX
    /// boot/run/shutdown, all 3<->2, from `0xbfc047ac`/`0xbfc047ec` running
    /// uncached out of KSEG1), and what IRIX leaves pages mapped with. The rest
    /// are decoded for correctness, not because anything exercises them.
    ///
    /// Write-through (R5000 C=0/1) folds onto write-back: `mips_cache_v2` has
    /// no write-through path, so write-back is the closest representable
    /// behaviour, and it is the right approximation — the alternative,
    /// treating a cacheable page as uncached, is a coherency error rather than
    /// a conservative fallback. Coherent variants (R4400 C=4/5/6) map to
    /// `TR_CACHEABLE_COH` where the cache models it and `TR_CACHEABLE`
    /// otherwise; the Indy is uniprocessor, so the distinction has no
    /// observable effect here.
    ///
    /// Reserved encodings stay **uncached**: undefined on the part in
    /// question, so a guest programming one is already out of spec, and
    /// declining to cache is the safe reading.
    ///
    /// `C::MIPS4` is the model discriminator — R5000 is the MIPS IV part,
    /// R4400 the MIPS III one — matching how the rest of this file selects
    /// per-model behaviour. The R10000 is MIPS IV too and has a third table
    /// (`R10000 UM` Table 4-1), selected by `C::R10K_CACHE_OPS`.
    #[inline]
    pub(crate) fn decode_cache_attr(c: u32) -> u32 {
        if C::R10K_CACHE_OPS {
            // R10000 UM Table 4-1: MIPS IV like the R5000, but its own table.
            // 4 and 5 are the coherent algorithms, not reserved: the O2's
            // IRIX runs KSEG0 with K0=5, and reading that as uncached sent
            // every kernel fetch, load and store to the bus. 7 (uncached
            // accelerated) only gathers stores; a load is a plain uncached one.
            match c & 0x7 {
                3     => TR_CACHEABLE,
                4 | 5 => TR_CACHEABLE_COH,
                _     => TR_UNCACHED, // 2 and 7; 0, 1, 6 reserved
            }
        } else if C::MIPS4 {
            // VR5000 Table 6-6.
            match c & 0x7 {
                0 | 1 => TR_CACHEABLE, // write-through -> write-back (see above)
                3     => TR_CACHEABLE,
                _     => TR_UNCACHED,  // 2 = Uncached; 4,5,6,7 reserved
            }
        } else {
            // R4400 Table 4-6.
            match c & 0x7 {
                3         => TR_CACHEABLE,
                4 | 5 | 6 => TR_CACHEABLE_COH, // coherent variants
                _         => TR_UNCACHED,      // 2 = Uncached; 0,1,7 reserved
            }
        }
    }

    /// [`Self::decode_cache_attr`] for every C value, for the TLB.
    pub(crate) fn cache_attr_table() -> [CacheAttr; 8] {
        std::array::from_fn(|c| match Self::decode_cache_attr(c as u32) {
            TR_UNCACHED => CacheAttr::Uncached,
            TR_CACHEABLE => CacheAttr::Cacheable,
            _ => CacheAttr::CacheableCoherent,
        })
    }

    /// KSEG0's cache attribute, from `Config.K0` (bits 2:0) rather than a
    /// hardcoded "cacheable".
    ///
    /// `Config.K0` uses the same encoding as EntryLo's C field (R4400 UM
    /// Table 4-6 / VR5000 UM Table 6-6), so this is just
    /// [`Self::decode_cache_attr`] applied to the register — see there for the
    /// per-model tables and why they differ.
    ///
    /// Why it matters even though IRIX leaves K0=3 for its whole life: the
    /// PROM legitimately sets K0=2 to run KSEG0 uncached around cache work,
    /// including at runtime on video-mode changes. Hardcoding `TR_CACHEABLE`
    /// made those accesses go through the cache anyway, which is exactly the
    /// window where doing so is wrong. cpucritique T3.
    ///
    /// There is no equivalent knob for KSEG1 — architecturally always
    /// uncached, not K0-controlled.
    #[inline]
    fn kseg0_cache_attr(&self) -> u32 {
        Self::decode_cache_attr(self.core.cp0_config)
    }

    /// Core translation logic.  When `DEBUG` is true the function:
    /// - always treats the access as kernel-privileged, and
    /// - never writes any CP0 side-effect registers (BadvAddr, EntryHi, Context, XContext).
    #[inline]
    fn translate_impl<const DEBUG: bool>(&mut self, virt_addr: u64, access_type: AccessType) -> TranslateResult {
        use crate::cpu::mips_core::{PrivilegeMode, PRIV_KERNEL, PRIV_SUPERVISOR, PRIV_USER};

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
        use crate::cpu::mips_core::{PRIV_KERNEL, PRIV_SUPERVISOR};

        // Upper 32 bits are ignored in 32-bit mode; only low 32 bits used for segment decode.
        let virt_addr32 = virt_addr as u32;

        // Extract top 3 bits to determine segment
        let segment = (virt_addr32 >> 29) as u64;

        let addr_exc = |wr: bool| exec_exception(if wr { EXC_ADES } else { EXC_ADEL });

        match segment {
            // KUSEG: 0x00000000 - 0x7FFFFFFF (user segment, TLB mapped)
            0..=3 => {
                // When ERL=1, KUSEG becomes unmapped, uncached identity mapping
                if (self.core.cp0_status & crate::cpu::mips_core::STATUS_ERL) != 0 {
                    return TranslateResult::ok(virt_addr32 as u64, TR_UNCACHED);
                }
                // 32-bit mode: XTLB=0 → UTLB vector on miss
                self.tlb_translate_impl::<DEBUG, 0>(virt_addr, access_type)
            }

            // KSEG0: 0x80000000 - 0x9FFFFFFF (kernel unmapped; cacheability
            // from Config.K0 — see `kseg0_cache_attr`)
            4 => {
                if PRIV == PRIV_KERNEL {
                    TranslateResult::ok((virt_addr32 & 0x1FFFFFFF) as u64, self.kseg0_cache_attr())
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
        use crate::cpu::mips_core::{PRIV_KERNEL, PRIV_SUPERVISOR};

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
                if (self.core.cp0_status & crate::cpu::mips_core::STATUS_ERL) != 0 {
                    return TranslateResult::ok(self.fold_phys(virt_addr), TR_UNCACHED);
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
                        // bits_61_59 is the C field, with this model's meaning:
                        // the Octane's IRIX reaches memory through 0xa000... (C=4).
                        let c = Self::decode_cache_attr(bits_61_59 as u32);
                        TranslateResult::ok(self.fold_phys(phys_addr), c)
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
                            // ckseg0 — same Config.K0 control as the 32-bit
                            // KSEG0 arm; it is the same physical window.
                            4 => return TranslateResult::ok((addr_32 & 0x1FFFFFFF) as u64, self.kseg0_cache_attr()),
                            5 => return TranslateResult::ok((addr_32 & 0x1FFFFFFF) as u64, TR_UNCACHED),
                            // KSSEG/KSEG3 compat: TLB mapped. `xtlb=1` even
                            // though the address is 32-bit-shaped — we are only
                            // here because `is_64bit_mode()` (Status.KX) is
                            // already true, and the R4400 chooses the refill
                            // vector from the *mode*, not the address shape.
                            // cpucritique.md T1.
                            //
                            // The compare width has to follow for the same
                            // reason: `vcmp64` keeps the Region field, which is
                            // `11` for these sign-extended addresses. Dropping
                            // it (`vcmp32`) let a cksseg VA alias an xuseg entry
                            // whose VPN2[31:13] happened to collide. This is
                            // only consistent because EntryHi is now committed
                            // through the unified `EH_VPN_MASK`, so a refill
                            // handler's TLBWI writes R=11 and bits[39:32]=0xFF
                            // — exactly what `vcmp64` will compare against.
                            _ => return self.tlb_translate_impl::<DEBUG, 1>(virt_addr, access_type),
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

    /// A CPU physical address as the board decodes it: through
    /// [`Self::phys_alias`] if it falls in it, then cut to [`Self::phys_mask`].
    #[inline(always)]
    fn fold_phys(&self, pa: u64) -> u64 {
        let pa = match self.phys_alias {
            Some((base, size, target)) if pa.wrapping_sub(base) < size => pa - base + target,
            _ => pa,
        };
        pa & self.phys_mask
    }

    /// TLB translation.  When `DEBUG` is true, CP0 side-effect registers are
    /// not written on miss/invalid/modified — the exception result is still
    /// returned so the caller knows the translation failed.
    #[inline]
    fn tlb_translate_impl<const DEBUG: bool, const XTLB: u8>(&mut self, virt_addr: u64, access_type: AccessType) -> TranslateResult {
        use crate::cpu::mips_tlb::TlbResult;

        // Get current ASID from EntryHi register
        let asid = (self.core.cp0_entryhi & 0xFF) as u8;

        // Query the TLB — XTLB=1 uses 64-bit VPN comparison mask
        let result = self.tlb.translate::<XTLB>(virt_addr, asid, access_type);

        let tlb_miss_code = if access_type == AccessType::Write { EXC_TLBS } else { EXC_TLBL };

        match result {
            TlbResult::Hit { phys_addr, cache_attr, dirty, global } => {
                if access_type == AccessType::Write && !dirty {
                    if !DEBUG { self.update_tlb_exception_registers::<XTLB>(virt_addr); }
                    TranslateResult::exc(exec_exception(EXC_MOD))
                } else {
                    // ok_mapped, not ok: this came from the TLB, so the nutlb
                    // fill needs TR_MAPPED plus the entry's G bit to decide
                    // whether to ASID-qualify the tag it stores.
                    TranslateResult::ok_mapped(self.fold_phys(phys_addr), cache_attr as u32, global)
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

        // EntryHi: Region + VPN2 from the address, ASID preserved. One mask for
        // both translation modes — no `XTLB` branch.
        //
        // The 32-bit case needs no narrower mask because every address reaching
        // the TLB in 32-bit mode is sign-extended: KUSEG has bit 31 clear, so
        // bits [63:32] are all 0 and both Region and [39:32] fall out as zero;
        // KSSEG/KSEG3 (0xFFFF_FFFF_C000_0000 / _E000_0000) have bit 31 set, so
        // bits [63:32] are all 1, giving Region `11` — the architecturally
        // required value — and [39:32] = 0xFF, which R4400 leaves undefined in
        // 32-bit mode. The sign extension already encodes the region, so
        // masking it through is both correct and branch-free.
        //
        // Using the narrow 32-bit mask (the original behaviour) wiped bits
        // [63:32] on a KSSEG/KSEG3 miss, leaving Region = `00` (User); a refill
        // handler that then issued TLBWI committed the entry under the wrong
        // region. cpucritique.md TLB-4.
        //
        // This is MAME r4000's `EH_WM` (0xC000_00FF_FFFF_E0FF) less the ASID
        // bits, which are carried separately below.
        const EH_VPN_MASK: u64 = EH_REGION | EH_VPN2_64; // 0xC000_00FF_FFFF_E000
        let asid = self.core.cp0_entryhi & 0xFF;
        self.core.cp0_entryhi = (virt_addr & EH_VPN_MASK) | asid;

        // Context: PTEBase[63:23] preserved, BadVPN2 = virt_addr[31:13] in bits [22:4].
        // Always 32-bit VPN — Context is used by the 32-bit UTLB handler.
        let ptebase = self.core.cp0_context & 0xFFFFFFFF_FF800000;
        let badvpn2 = ((virt_addr & EH_VPN2_32) >> 13) << 4;
        self.core.cp0_context = ptebase | badvpn2;

        // XContext: PTEBase[63:33] preserved, Region[63:62] → bits[32:31], BadVPN2[39:13] → bits[30:4].
        // XContext's fields are laid out around the width of BadVPN2, which is
        // VA[VA_BITS-1:13] — 27 bits on a 40-bit CPU, 31 on the R10000's
        // 44-bit one. Region sits directly above it and PTEBase above that, so
        // all three move together: PTEBase[63:VA_BITS-7],
        // Region[VA_BITS-8:VA_BITS-9], BadVPN2[VA_BITS-10:4].
        //
        // Using the 40-bit layout on an R10000 puts Region and PTEBase four
        // bits too low. A 64-bit kernel reads XContext in its TLB refill
        // handler to find the page-table entry for the faulting address, so
        // the address it computes lands outside the page table it wired, the
        // nested miss is unrecoverable, and IRIX panics with "Entered tlbmiss
        // with invalid vaddr".
        //
        // EntryHi keeps the 40-bit VPN2 field on every model: a 64-bit kernel
        // zero-fills those upper bits itself when it builds an EntryHi value,
        // and widening the mask here breaks it.
        self.core.cp0_xcontext =
            xcontext_on_miss(C::VA_BITS, self.core.cp0_xcontext, virt_addr);
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
            if !DEBUG && self.bp_enabled() && self.check_breakpoint::<{ BpType::PhysFetch as u8 }>(phys_addr) {
                return FetchInstrResult::exception(EXEC_BREAKPOINT);
            }

            let r = self.cache.fetch(virt_addr, phys_addr);
            if !DEBUG && r.status == exec_exception(EXC_VCEI) {
                self.core.cp0_badvaddr = virt_addr;
            }
            r
        } else {
            #[cfg(not(feature = "lightning"))]
            if !DEBUG && self.bp_enabled() && self.check_breakpoint::<{ BpType::PhysFetch as u8 }>(phys_addr) {
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
                    eprintln!("Bus error on instruction fetch: PC={:016x} PA={:08x} status={:08x}{}", virt_addr, phys_addr, r.status, self.fault_note(phys_addr, 4));
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
            let phys_addr = translate_result.phys;
            let is_cached = translate_result.is_cached();

            if is_cached {
                    // Cached access uses D-Cache
                    #[cfg(not(feature = "lightning"))]
                    if !DEBUG && self.bp_enabled() && self.check_breakpoint::<{ BpType::PhysRead as u8 }>(phys_addr) {
                        return Err(EXEC_BREAKPOINT);
                    }

                    #[cfg(feature = "jitstats")]
                    self.jit_stat_probe::<false>(virt_addr, phys_addr);

                    let r = self.cache.read::<SIZE>(virt_addr, phys_addr);
                    if r.is_ok() {
                        Ok(r.data)
                    } else {
                        // A Cache Error leaves BadVAddr alone; it changes
                        // only CacheErr, ErrorEPC and ERL.
                        if !DEBUG && r.status & EXEC_IS_CACHE_ERR == 0 { self.core.cp0_badvaddr = virt_addr; }
                        Err(r.status) // BUS_BUSY, BUS_VCE, BUS_ERR or EXEC_CACHE_ERR — all valid ExecStatus
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
                            let r = self.sysad.read8(phys_addr);
                            BusRead64 { status: r.status, data: r.data as u64 }
                        } else if SIZE == 2 {
                            let r = self.sysad.read16(phys_addr);
                            BusRead64 { status: r.status, data: r.data as u64 }
                        } else if SIZE == 4 {
                            let r = self.sysad.read32(phys_addr);
                            BusRead64 { status: r.status, data: r.data as u64 }
                        } else {
                            self.sysad.read64(phys_addr)
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
                                    eprintln!("Bus error on uncached read{}: PC={:016x} VA={:016x} PA={:016x} status={:08x}{}{}", SIZE*8, self.core.pc, virt_addr, phys_addr, r.status, jitcheck_pass, self.fault_note(phys_addr, SIZE));
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
            self.core.lockstep_mem = result.ok().map(|val| crate::cpu::mips_core::LockstepMemCapture {
                addr: virt_addr,
                phys: translate_result.phys,
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
        let phys_addr = translate_result.phys;
        let is_cached = translate_result.is_cached();

        // Diagnostic (`jitv2_smc_check`): is this write landing on the code
        // page the CPU is currently fetching/compiling from? Placed here —
        // after translation, before the write commits — because a code page
        // is identified by physical frame, and the writer's virtual alias is
        // usually not the fetcher's. `DEBUG` accesses are monitor/tooling
        // reads-and-writes, not guest stores, so they are excluded.
        #[cfg(all(feature = "jitv2", feature = "jitv2_smc_check"))]
        if !DEBUG {
            self.smc_check_write(virt_addr, phys_addr, SIZE);
        }

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
            #[cfg(feature = "jitstats")]
            self.jit_stat_probe::<true>(virt_addr, phys_addr);

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
                self.sysad.write8(phys_addr, val as u8)
            } else if SIZE == 2 {
                self.sysad.write16(phys_addr, val as u16)
            } else if SIZE == 4 {
                self.sysad.write32(phys_addr, val as u32)
            } else {
                self.sysad.write64(phys_addr, val)
            };
            if ws != BUS_OK && ws != BUS_BUSY && !DEBUG {
                eprintln!("Bus error on uncached write{}: PC={:016x} VA={:016x} PA={:016x} val={:016x} status={:08x}{}", SIZE*8, self.core.pc, virt_addr, phys_addr, val, ws, self.fault_note(phys_addr, SIZE));
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
            self.core.lockstep_mem = (status == BUS_OK).then_some(crate::cpu::mips_core::LockstepMemCapture {
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
        let phys_addr = translate_result.phys;
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
            let ws = self.sysad.write64_masked(phys_addr, val, mask);
            if ws != BUS_OK && ws != BUS_BUSY && !DEBUG {
                eprintln!("Bus error on uncached write64_masked: PC={:016x} VA={:016x} PA={:016x} val={:016x} mask={:016x} status={:08x}{}", self.core.pc, virt_addr, phys_addr, val, mask, ws, self.fault_note(phys_addr, 8));
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
        // MOVF/MOVT read the FP condition codes out of FCSR, so they are CP1
        // accesses and take Coprocessor Unusable when CU1 is clear — exactly
        // like the COP1-*encoded* MOVF.fmt handler (`exec_fmovcf_s`) a few
        // hundred lines down. Easy to miss because this one is dispatched
        // from the integer SPECIAL funct table (funct 0x01), not through any
        // COP1 gate.
        //
        // IRIX unloads the FPU lazily on context switch (`swtch.c` checkfp)
        // and sets SR_CU1 on the resulting CpU fault. Without this check a
        // MIPS IV process resumed after a context switch reads the *previous*
        // process's FCSR condition bits instead of faulting into the restore.
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
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
    /// Answer a host call (system calls 3000-3009, see iris-hostcall) if this
    /// `syscall` is one and a service is registered for it. Only from user
    /// mode and outside a delay slot; anything else goes to IRIX as usual.
    #[cfg(feature = "hostcall")]
    fn host_call(&mut self) -> Option<ExecStatus> {
        if self.core.get_privilege_mode() != PrivilegeMode::User || self.core.in_delay_slot {
            return None;
        }
        let v0 = self.core.read_gpr(2);
        let n = v0 as u32;
        if !(iris_hostcall::FIRST..=iris_hostcall::LAST).contains(&n) && n != iris_hostcall::SYS_SYSCALL {
            return None;
        }
        let mut regs = [0u64; 8];
        for (i, r) in regs.iter_mut().enumerate() {
            *r = self.core.read_gpr(4 + i as u32);
        }
        let (number, args) = iris_hostcall::decode(v0, &regs)?;
        let reply = iris_hostcall::dispatch(number, &mut HostMemory { exec: self }, &args)?;
        let (rv0, rv1, ra3) = iris_hostcall::registers(reply);
        self.core.write_gpr(2, rv0);
        self.core.write_gpr(3, rv1);
        self.core.write_gpr(7, ra3);
        Some(self.handle_exec_complete())
    }

    fn exec_syscall(&mut self, _d: &DecodedInstr) -> ExecStatus {
        // A host call (iris-hostcall) is answered here and never becomes an
        // exception: the program resumes at the next instruction.
        #[cfg(feature = "hostcall")]
        if let Some(status) = self.host_call() {
            return status;
        }
        // Compile-worthy on entry too, symmetric with exec_eret's
        // syscall_pending-gated trigger on return — see MipsCore's own doc
        // comment on that field for why neither can rely on the general
        // exception-vector arrival being probed automatically.
        #[cfg(feature = "jitv2")]
        { self.core.jit_trigger = true; }
        let s = exec_exception(EXC_SYS);
        #[cfg(feature = "jitv2")]
        { self.handle_exception_syscall(s) }
        #[cfg(not(feature = "jitv2"))]
        { self.handle_exception(s) }
    }
    fn exec_break(&mut self, _d: &DecodedInstr) -> ExecStatus {
        let s = exec_exception(EXC_BP);
        self.handle_exception(s)
    }
    fn exec_sync(&mut self, _d: &DecodedInstr) -> ExecStatus {
        // x86 host stores are already ordered as strongly as SYNC asks.
        #[cfg(not(target_arch = "x86_64"))]
        if self.core.smp {
            std::sync::atomic::fence(Ordering::SeqCst);
        }
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
        // Check CP0 usability: Kernel mode, or Status.CU0 set. Supervisor is
        // NOT implicitly allowed on R4000/R4400 — it falls into the `_` arm and
        // needs CU0 like user mode. (Same rule as `exec_cop0`.)
        let privilege = self.core.get_privilege_mode();
        use crate::cpu::mips_core::{PrivilegeMode, STATUS_CU0};

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
        // On an R10000 the encodings for 5/6/7 are index operations, not the
        // R4000 hit operations — translating them would fault on an index that
        // is not a valid virtual address. See C_R10K_ISD in mips_cache_v2.
        let sel = cache_op & 3;
        let r10k_index_op = C::R10K_CACHE_OPS
            && match op {
                C_R10K_CBARRIER => sel == CACH_PI,
                C_R10K_ILD => matches!(sel, CACH_PI | CACH_PD | CACH_SD),
                C_R10K_ISD => matches!(sel, CACH_SI | CACH_SD),
                _ => false,
            };
        let needs_translation =
            !r10k_index_op && matches!(op, C_CDX | C_HINV | C_HWBINV | C_HWB | C_HSV);

        let phys_addr = if needs_translation {
            // Hit operations need address translation
            let tr = self.nanotlb_translate::<{AccessType::Read as u8}>(virt_addr);
            if tr.is_exception() { return self.handle_exception(tr.status); }
            tr.phys
        } else {
            // Index operations use virt_addr as index, no translation needed
            virt_addr
        };

        // For Index_Store_Tag, pass TagLo via phys_addr
        let op = cache_op & 0x1C;
        // A cache tag is TagHi:TagLo, not TagLo alone. The IP28 PROM writes
        // the low 32 bits to $28 and the bits above to $29 — a 36-bit
        // secondary tag arrives as TagLo=0xffffcdfe, TagHi=0xf — so a model
        // that reads only TagLo sees a different tag from the one written.
        // Harmless for R4400/R5000, whose tags fit in 32 bits and who leave
        // TagHi zero: the shift-in contributes nothing there.
        let stores_taglo = op == C_IST || (r10k_index_op && op == C_R10K_ISD);
        let phys_addr_or_taglo = if stores_taglo {
            ((self.core.cp0_taghi as u64) << 32) | (self.core.cp0_taglo as u32 as u64)
        } else {
            phys_addr
        };

        // On an R10000 the check bits travel with cache data through CP0 ECC.
        if C::R10K_CACHE_OPS {
            self.cache.set_cache_ecc(self.core.cp0_ecc);
        }
        // IP28 cache-error investigation (`log mips mask cp0`). Two kinds
        // of op are worth seeing: one carrying non-zero check bits, which is a
        // guest staging a parity error on purpose, and any op from outside the
        // PROM — i.e. from a loaded diagnostic, which runs out of XKPHYS.
        //
        // Both halves of that gate were learned the hard way. Gating on ECC
        // alone hid every op the IDE issued, because the IDE's ECC reads back
        // zero: the exact symptom under investigation was also blinding the
        // instrument to its cause. The cap then has to be generous, because a
        // cap of 600 silently truncated a run at precisely the boundary and
        // made a partial picture look like the whole one. It exists only so a
        // hot loop cannot rewrite the timing it is measuring.
        if crate::cpu::mips_core::cachediag_on() {
            let from_prom = (self.core.pc >> 32) == 0xFFFF_FFFF;
            // CBARRIER is pure ordering: it names no line and carries no check
            // bits, and it outnumbers everything else ~8:1 (83534 of 94322 in
            // one IDE run). Tracing it swamped the log and slowed the guest
            // enough that the run no longer reached the loop being studied —
            // so drop it unless it is carrying staged check bits.
            let noise = op == C_R10K_CBARRIER && self.core.cp0_ecc == 0;
            if (self.core.cp0_ecc != 0 || !from_prom) && !noise {
                const CACHE_TRACE_CAP: u32 = 200_000;
                static SEEN: std::sync::atomic::AtomicU32 =
                    std::sync::atomic::AtomicU32::new(0);
                let n = SEEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if n < CACHE_TRACE_CAP {
                    crate::dlog!(LogModule::Mips,
                        "ip28cd: CACHE op={:#04x} sel={} vaddr={:#018x} ECC={:#010x} \
                         TagHi:Lo={:#010x}:{:#018x} pc={:#018x}",
                        op, sel, virt_addr, self.core.cp0_ecc,
                        self.core.cp0_taghi, self.core.cp0_taglo, self.core.pc);
                } else if n == CACHE_TRACE_CAP {
                    crate::dlog!(LogModule::Mips,
                        "ip28cd: CACHE trace capped at {CACHE_TRACE_CAP} ops \
                         — output past this point is incomplete");
                }
            }
        }
        // Call unified cache interface
        let result = self.cache.cache_op(cache_op, virt_addr, phys_addr_or_taglo);
        // An I-cache write-back of a line with bad parity raises a Cache
        // Error from the CACHE instruction itself.
        #[cfg(feature = "cache-ecc")]
        {
            let op_status = self.cache.take_op_status();
            if op_status != 0 {
                return self.handle_exception(op_status);
            }
        }
        // Index_Load_Tag also loads CP0 ECC with the selected doubleword's
        // check bits (R4000 UM A-42); the R10000 does this on Index_Load_Data.
        if (C::R10K_CACHE_OPS && op == C_R10K_ILD)
            || (cfg!(feature = "cache-ecc") && !C::R10K_CACHE_OPS && op == C_ILT) {
            self.core.cp0_ecc = self.cache.cache_op_ecc();
            self.cache.set_cache_ecc(self.core.cp0_ecc);
        }

        // For Index_Load_Tag, update CP0 TagLo from result
        if op == C_ILT || (r10k_index_op && op == C_R10K_ILD) {
            // Split back the way it arrived. Zeroing TagHi unconditionally
            // threw away the top of every tag wider than 32 bits.
            self.core.cp0_taglo = result & 0xFFFF_FFFF;
            self.core.cp0_taghi = (result >> 32) as u32;
        }

        self.handle_exec_complete()
    }


    // LL - Load Linked (32-bit)
    /// `ll stats` bookkeeping: a reservation was just taken at `lladdr`.
    #[inline]
    fn ll_stat_ll(&mut self, #[allow(unused_variables)] lladdr: u64) {
        #[cfg(feature = "llstats")]
        { ll_stats_slot(&mut self.ll_stats, lladdr).ll += 1; }
    }

    /// `ll stats` bookkeeping: an `SC` resolved. `ok` distinguishes a store
    /// that landed from one that found the LLbit already clear.
    ///
    /// Keyed by the *current* `LLAddr`, which is the address the live
    /// reservation was taken on — not the SC's own address. They are normally
    /// the same; when they are not, attributing to the reservation is what
    /// makes a stuck loop legible.
    #[inline]
    fn ll_stat_sc(&mut self, #[allow(unused_variables)] ok: bool) {
        #[cfg(feature = "llstats")]
        {
            let key = self.cache.get_lladdr();
            let e = ll_stats_slot(&mut self.ll_stats, key);
            if ok {
                e.ok += 1;
                e.cur_run = 0;
            } else {
                e.fail += 1;
                e.cur_run += 1;
                if e.cur_run > e.max_run { e.max_run = e.cur_run; }
            }
        }
    }

    fn exec_ll(&mut self, d: &DecodedInstr) -> ExecStatus {
        let base = self.core.read_gpr(d.rs as u32);
        let virt_addr = base.wrapping_add(d.immu64());
        let rt_reg = d.rt as u32;

        match self.read_data::<4>(virt_addr) {
            Ok(value) => {
                // Sign-extend word to 64 bits
                self.core.write_gpr(rt_reg, value as i32 as i64 as u64);
                // Store physical address in LLAddr register
                // LLAddr holds PA[PA_BITS-1:4]: 35:4 on the R4x00/R5000, 39:4 on the R10000
                let tr = self.nanotlb_translate::<{AccessType::Read as u8}>(virt_addr);
                if !tr.is_exception() {
                    let lladdr = (tr.phys >> 4) & C::LLADDR_MASK;
                    self.cache.set_lladdr(lladdr);
                    self.core.cp0_lladdr = lladdr;
                    self.core.ll_value = value;
                    self.core.ll_phys = tr.phys;
                    self.cache.set_llbit(true);
                    self.ll_stat_ll(lladdr);
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
            self.ll_stat_sc(false);
            self.core.write_gpr(rt_reg, 0);
            return self.handle_exec_complete();
        }

        // Check if address matches the LL address
        let tr = self.nanotlb_translate::<{AccessType::Write as u8}>(virt_addr);
        if tr.is_exception() {
            self.cache.set_llbit(false);
            return self.finish_status(tr.status);
        }

        // Succeeds on the LLbit ALONE — `LLAddr` is deliberately NOT consulted.
        //
        // R4000 manual: LLAddr is a diagnostic register, written by LL and
        // readable by software, but not part of SC's success condition.
        // MAME's r4000.cpp agrees exactly: `case 0x38: // SC` tests
        // `m_ll_active` only, and writes `m_cp0[CP0_LLAddr]` at LL time purely
        // so the guest can read it back.
        //
        // Comparing the address here (as this used to) can only make SC fail
        // where hardware succeeds: LLAddr is a single global register while
        // the LLbit is the real reservation, and every architectural reason
        // for SC to fail — an intervening exception, ERET, or a coherent
        // invalidation of the line — already clears the LLbit.
        let value = self.core.read_gpr(rt_reg);
        // Several CPUs: another's store cannot clear this LLbit, so the store
        // itself has to notice it (`clanker/cpu/mp-llsc-is-a-host-compare-exchange.md`).
        if self.core.smp && virt_addr & 3 == 0 && tr.phys == self.core.ll_phys {
            if let Some(ok) = self.cache.cas_ram32(tr.phys, self.core.ll_value as u32, value as u32) {
                self.ll_stat_sc(ok);
                self.core.write_gpr(rt_reg, ok as u64);
                self.cache.set_llbit(false);
                return self.handle_exec_complete();
            }
        }
        let status = self.write_data::<4>(virt_addr, value);
        if status == EXEC_COMPLETE {
            self.ll_stat_sc(true);
            self.core.write_gpr(rt_reg, 1);
            self.cache.set_llbit(false);
        }
        self.finish_status(status)
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
                    let lladdr = (tr.phys >> 4) & C::LLADDR_MASK;
                    self.cache.set_lladdr(lladdr);
                    self.core.cp0_lladdr = lladdr;
                    self.core.ll_value = value;
                    self.core.ll_phys = tr.phys;
                    self.cache.set_llbit(true);
                    self.ll_stat_ll(lladdr);
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
            self.ll_stat_sc(false);
            self.core.write_gpr(rt_reg, 0);
            return self.handle_exec_complete();
        }

        // Check if address matches the LLD address
        let tr = self.nanotlb_translate::<{AccessType::Write as u8}>(virt_addr);
        if tr.is_exception() {
            self.cache.set_llbit(false);
            return self.finish_status(tr.status);
        }
        // Succeeds on the LLbit alone — see `exec_sc` for why `LLAddr` is not
        // consulted (diagnostic register, not part of SC's success condition;
        // MAME's r4000.cpp tests `m_ll_active` only).
        let value = self.core.read_gpr(rt_reg);
        if self.core.smp && virt_addr & 7 == 0 && tr.phys == self.core.ll_phys {
            if let Some(ok) = self.cache.cas_ram64(tr.phys, self.core.ll_value, value) {
                self.ll_stat_sc(ok);
                self.core.write_gpr(rt_reg, ok as u64);
                self.cache.set_llbit(false);
                return self.handle_exec_complete();
            }
        }
        let status = self.write_data::<8>(virt_addr, value);
        if status == EXEC_COMPLETE {
            self.ll_stat_sc(true);
            self.core.write_gpr(rt_reg, 1);
            self.cache.set_llbit(false);
        }
        self.finish_status(status)
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
        // CP0 is usable iff Status.CU0 is set, or the CPU is in Kernel mode.
        // Supervisor gets no implicit access on R4000/R4400 — it is a segment
        // privilege level, not a coprocessor-0 one.
        //
        // Without this, every CP0 instruction — MTC0 Status included — was
        // executable from user mode, so a user process could simply write
        // itself into kernel mode. This is the single dispatch point for all of
        // OP_COP0: jitv2 classifies the whole major opcode `Excluded`
        // (analyzer.rs), so compiled code retires these through
        // `interp_fallback_fn` into this same function.
        {
            use crate::cpu::mips_core::{PrivilegeMode, STATUS_CU0};
            let cp0_usable = match self.core.get_privilege_mode() {
                PrivilegeMode::Kernel => true,
                _ => (self.core.cp0_status & STATUS_CU0) != 0,
            };
            if !cp0_usable {
                return self.cpu_unusable(0);
            }
        }

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
        self.ip28_cp0_trace("mfc0", rd_val, value);
        // Sign-extend 32-bit value to 64 bits
        self.core.write_gpr(rt_reg, value as u32 as i32 as i64 as u64);
        self.handle_exec_complete()
    }

    // DMFC0 - Doubleword Move From CP0 (MIPS III)
    fn exec_dmfc0(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rt_reg = d.rt as u32;
        let rd_val = d.rd as u32;
        let value = self.core.read_cp0(rd_val);
        self.ip28_cp0_trace("dmfc0", rd_val, value);
        self.core.write_gpr(rt_reg, value);
        self.handle_exec_complete()
    }

    // MTC0 - Move To CP0
    /// The CP0 registers that are 64 bits wide.
    ///
    /// EntryLo0/1, Context, BadVAddr, EntryHi, EPC, XContext and ErrorEPC are
    /// 64-bit in MIPS III and stay so in MIPS IV. TagLo is 32-bit on R4x00 but
    /// 64-bit on the R10000, which is why it is asked of the model rather than
    /// listed flat.
    fn cp0_is_64bit(reg: u32) -> bool {
        matches!(reg, 2 | 3 | 4 | 8 | 10 | 14 | 20 | 30) || (C::R10K_CACHE_OPS && reg == 28)
    }

    fn exec_mtc0(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rt_val = self.core.read_gpr(d.rt as u32);
        self.ip28_cp0_trace("mtc0", d.rd as u32, rt_val);
        let rd_val = d.rd as u32;

        // MTC0 moves the whole register. DMTC0 differs in what the
        // assembler will accept, not in how much data reaches CP0, so a
        // 64-bit CP0 register takes `rt` intact and only a 32-bit one takes
        // the low word (sign-extended, an artifact of holding it in a 64-bit
        // field here).
        //
        // Sign-extending into the 64-bit registers as well was silently fine
        // for a 32-bit kernel, whose registers are sign-extended anyway, and
        // fatal to a 64-bit one. IRIX's 64-bit standalone code returns from an
        // exception with `mtc0 ra, EPC` / `mtc0 ra, ErrorEPC` / `eret`: with
        // `ra` = 0xa8000000208b9998 the ERET went to 0x00000000208b9998 and
        // landed in unmapped user space.
        //
        // Two other readings were tried against both guests and rejected.
        // Preserving bits 63:32 of the old value fixes the 64-bit case and
        // breaks the 32-bit one — EntryHi's upper half carries the TLB Region,
        // so a stale one files entries under the wrong region and IRIX 6.5.22
        // dies in sash on a kseg3 address. Doing that only in 64-bit
        // addressing mode works for both but needs a mode test MTC0 does not
        // architecturally have. Moving the whole register needs neither, and
        // both guests are happy with it.
        let value = if Self::cp0_is_64bit(rd_val) {
            rt_val
        } else {
            rt_val as u32 as i32 as i64 as u64
        };
        self.core.write_cp0(rd_val, value);
        self.handle_cp0_side_effects(rd_val);
        self.handle_exec_complete()
    }

    // DMTC0 - Doubleword Move To CP0 (MIPS III)
    /// IP28 bring-up: watch the cache-tag CP0 registers (26 ECC, 28 TagLo,
    /// 29 TagHi) under `log mips mask cp0`. Only the R10000 model compiles
    /// this in.
    ///
    /// The env-var form this replaced did an uncached `getenv` on every
    /// DMTC0/MTC0, armed or not.
    #[inline(always)]
    fn ip28_cp0_trace(&self, what: &str, reg: u32, val: u64) {
        if C::R10K_CACHE_OPS && matches!(reg, 26 | 28 | 29) && mips_log_always(MIPS_LOG_CP0) {
            crate::dlog!(LogModule::Mips, "ip28cp0: {what} ${reg} = {val:#018x}");
        }
    }

    fn exec_dmtc0(&mut self, d: &DecodedInstr) -> ExecStatus {
        let rt_val = self.core.read_gpr(d.rt as u32);
        self.ip28_cp0_trace("dmtc0", d.rd as u32, rt_val);
        let rd_val = d.rd as u32;
        self.core.write_cp0(rd_val, rt_val);
        self.handle_cp0_side_effects(rd_val);
        self.handle_exec_complete()
    }

    fn handle_cp0_side_effects(&mut self, reg: u32) {
        // EntryHi (reg 10) carries the current ASID in bits [7:0]. The nanotlb
        // tags hold only a VA, so their validity depends on the ASID that was
        // live when they were filled. Every *other* invalidation site is a
        // `resync_privilege_state` call (the status callback, all three
        // `handle_exception` paths, `exec_eret`, soft reset, the restore paths)
        // — i.e. a privilege transition, which is what normally
        // contains an ASID switch, since a kernel changes ASID inside EXL and
        // returns through ERET. A bare `MTC0/DMTC0 EntryHi` has no such
        // transition around it, so nothing else would flush, and a following
        // access to a mapped VA could hit a slot filled under the *previous*
        // ASID and return the wrong address space's physical page.
        //
        // Cold path (a handful per context switch), so the unconditional
        // flush costs nothing measurable. See `exec_tlbr` for the other
        // ASID-mutating site, and docs/nutlb-design.md §1a.
        // LLAddr implements PA[PA_BITS-1:4]; the bits above read as zero.
        if reg == 17 {
            self.core.cp0_lladdr &= C::LLADDR_MASK;
        }
        if reg == 10 {
            // Flushes the nutlb too — its tags carry no ASID, so entries
            // filled under the old one would otherwise stay live and resolve
            // to the wrong address space.
            self.nanotlb_invalidate();
        }
        // Config (reg 16), K0 field (bits 2:0): KSEG0's cache attribute, read
        // by `kseg0_cache_attr` on every KSEG0/ckseg0 translation.
        //
        // This invalidates the **nutlb only** — the cached *translations*. It
        // does not touch L1/L2, and must not: flipping K0 writes back nothing
        // on real hardware either. Getting dirty lines out of the caches before
        // changing a region's cacheability is software's job, and software that
        // skips it loses them. All we owe here is that the next access
        // re-derives its attribute instead of reusing a cached one.
        //
        // Needed because `nutlb_fill` stores the attribute *inside* the entry
        // (`result.status & 0x7`), and a nutlb tag is a bare page number
        // recording nothing about which K0 filled it — so without this,
        // entries filled before a K0 change keep serving the old cacheability.
        // Structurally identical to the EntryHi/ASID flush above.
        //
        // Unconditional rather than gated on "did K0 actually change":
        // measured over a full IRIX boot + run + shutdown, the PROM writes
        // Config five times and **every one of them changes K0** (3<->2, from
        // `0xbfc047ac`/`0xbfc047ec`, running uncached out of KSEG1 around cache
        // work — including at runtime on video-mode changes, not just at boot).
        // A change-detection gate would therefore have skipped exactly zero
        // flushes on a real workload while costing a field to keep in sync
        // across every path that writes cp0_config directly.
        //
        // Five events per boot also means the flush's cost is irrelevant, so
        // there is nothing to optimise here.
        //
        // KSEG1 needs no barrier: architecturally always uncached.
        if reg == 16 {
            self.nanotlb_invalidate();
        }
        // cheritest: a write to CP0 26 triggers the test device's dump. Routed
        // over the bus, so with no test device mapped it's an ignored GIO access.
        // CP0 ECC: what a store writes as check bits while Status.CE is set.
        if reg == 26 {
            self.cache.set_cache_ecc(self.core.cp0_ecc);
        }
        if reg == 26 && self.cheritest_dump_hook {
            self.sysad.write32((crate::dev::testdev::TEST_DEV_BASE + crate::dev::testdev::REG_DUMP) as u64,
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
            // exec_tlbr/p always return bare EXEC_COMPLETE (no other status
            // possible — they don't mutate the TLB), so run them for effect
            // then finish here. Note "doesn't mutate the TLB" is not the same
            // as "needs no invalidation": exec_tlbr rewrites EntryHi and so
            // can change the current ASID, and flushes the nanotlb itself for
            // that reason. exec_tlbwi/wr retire PC themselves and may
            // return EXEC_BREAKPOINT (feature = "tlbcheck" found a bad write),
            // so their return value is passed straight through.
            FUNCT_TLBR => { self.exec_tlbr(); self.handle_exec_complete() }
            FUNCT_TLBWI => self.exec_tlbwi(),
            FUNCT_TLBWR => self.exec_tlbwr(),
            FUNCT_TLBP => { self.exec_tlbp(); self.handle_exec_complete() }
            FUNCT_ERET => self.exec_eret(), // PC set directly, terminal as-is
            // WAIT (R4600/R5000; the R4000/R4400 have no such instruction —
            // there it is an unimplemented COP0 function that reads as a
            // no-op rather than raising RI, which is why this arm is not
            // model-gated).
            FUNCT_WAIT => self.exec_wait(),
            _ => {
                let s = self.reserved_instruction(d);
                self.handle_exception(s)
            }
        }
    }

    // TLBR - Read Indexed TLB Entry
    // Reads the TLB entry indexed by CP0.Index into CP0.EntryHi, CP0.EntryLo0, CP0.EntryLo1, and CP0.PageMask
    fn exec_tlbr(&mut self) -> ExecStatus {
        // Index selects the slot from bits [5:0] only — mask, never modulo.
        // `%` folds the TLBP-failure bit (31) back into a *valid* slot number
        // (0x8000_0000 % 48 == 32), silently reading a live entry; it also maps
        // an out-of-range 48..63 onto 0..15, which are the wired entries. See
        // `exec_tlbwi` for the full argument and the MAME cross-check.
        let index = (self.core.cp0_index & CP0_INDEX_SLOT_MASK) as usize;
        if index >= self.tlb.num_entries() {
            return EXEC_COMPLETE;
        }
        let entry = self.tlb.read(index);

        // Per MIPS R4000 spec: Extract G bit from EntryHi bit 12 and populate both EntryLo G bits
        let g_bit = (entry.entry_hi >> 12) & 1;

        // Write to CP0 registers, clearing G bit from EntryHi
        self.core.cp0_entryhi = entry.entry_hi & !0x1000; // Clear bit 12 (G bit)
        self.core.cp0_entrylo0 = (entry.entry_lo[0] & !1) | g_bit; // Set G bit from EntryHi
        self.core.cp0_entrylo1 = (entry.entry_lo[1] & !1) | g_bit; // Set G bit from EntryHi
        self.core.cp0_pagemask = entry.page_mask;

        // TLBR overwrites EntryHi wholesale from the indexed entry — ASID
        // bits [7:0] included — so the current ASID can change here even
        // though the TLB itself is untouched. Same hazard as `MTC0 EntryHi`
        // (see handle_cp0_side_effects): no privilege transition accompanies
        // a TLBR, so without this flush a stale nanotlb slot filled under the
        // old ASID stays live. Cold path.
        self.nanotlb_invalidate();

        if mips_log(MIPS_LOG_TLB) { dlog_dev!(LogModule::Mips, "TLBR: Read Index {}\n{}", index, self.tlb.format_entry(index)); }

        EXEC_COMPLETE
    }

    // Helper function to construct a TLB entry from CP0 registers
    // Per MIPS R4000 spec: G bit is formed by ANDing G bits from EntryLo0 and EntryLo1
    // and stored in bit 12 of EntryHi in the TLB entry
    fn create_tlb_entry_from_cp0(&self) -> crate::cpu::mips_tlb::TlbEntry {
        use crate::cpu::mips_tlb::TlbEntry;

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
    /// Report every TLB write, under `log mips mask tlb`.
    ///
    /// `IRIS_IP28_TLBW=wired` additionally narrows it to writes below CP0
    /// Wired — the mappings a kernel means never to be evicted. That stays an
    /// environment variable because it selects a *subset*, which the module
    /// mask has no way to express; the on/off is devlog's.
    #[inline]
    fn ip28_trace_tlb_write(&self, op: &str, index: usize, entry: &crate::cpu::mips_tlb::TlbEntry) {
        if !mips_log_always(MIPS_LOG_TLB) {
            return;
        }
        static WIRED_ONLY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let wired_only = *WIRED_ONLY.get_or_init(|| {
            matches!(std::env::var("IRIS_IP28_TLBW").as_deref(), Ok("wired"))
        });
        if wired_only && index >= self.core.cp0_wired as usize {
            return;
        }
        crate::dlog!(LogModule::Mips,
            "ip28tlbw: {op} idx={index:<2} wired={} entryhi={:#018x} lo0={:#018x} lo1={:#018x} \
             mask={:#x} pc={:#018x}",
            self.core.cp0_wired, entry.entry_hi, entry.entry_lo[0], entry.entry_lo[1],
            entry.page_mask, self.core.pc);
    }

    fn exec_tlbwi(&mut self) -> ExecStatus {
        // The slot is Index[5:0]. Masking rather than `%` matters twice:
        //
        // 1. A failed `TLBP` leaves bit 31 set (`0x8000_0000`). Under `%` that
        //    became `0x8000_0000 % 48 == 32` — a perfectly valid slot — so a
        //    guest that issued TLBWI after a missed probe silently overwrote a
        //    live entry 32 instead of touching the slot the architecture names.
        // 2. `%` maps an out-of-range 48..63 onto 0..15, which are exactly the
        //    wired entries a kernel relies on staying put.
        //
        // R4400 leaves TLBWI with Index >= num_entries *undefined*; skipping is
        // our choice, not a fidelity claim. It matches MAME's r4000 and mips3
        // cores, which both do `Index & 0x3f` followed by a bounds check with no
        // else branch.
        //
        // Note the failed-probe case now writes entry 0 (`0x8000_0000 & 0x3f`)
        // rather than entry 32. That is the architectural behaviour, not a
        // safety net: entry 0 is typically wired, so a guest doing TLBWI after a
        // missed TLBP still corrupts something. It just corrupts what real
        // hardware would.
        let index = (self.core.cp0_index & CP0_INDEX_SLOT_MASK) as usize;
        if index >= self.tlb.num_entries() {
            return self.handle_exec_complete();
        }
        let entry = self.create_tlb_entry_from_cp0();
        self.ip28_trace_tlb_write("tlbwi", index, &entry);
        self.tlb.write(index, entry);
        // Flushes the nutlb too — the TLB it caches just changed.
        self.nanotlb_invalidate();

        if mips_log(MIPS_LOG_TLB) { dlog_dev!(LogModule::Mips, "TLBWI: Write Index {}\n{}", index, self.tlb.format_entry(index)); }

        #[cfg(feature = "tlbcheck")]
        {
            let bad = self.tlb.consistency_check(&format!("TLBWI idx={} pc={:#010x}", index, self.core.pc));
            self.handle_exec_complete();
            return if bad { EXEC_BREAKPOINT } else { EXEC_COMPLETE };
        }
        #[cfg(not(feature = "tlbcheck"))]
        self.handle_exec_complete()
    }

    // TLBWR - Write Random TLB Entry
    // Writes CP0.EntryHi, CP0.EntryLo0, CP0.EntryLo1, and CP0.PageMask to a random TLB entry
    // The random index is determined by CP0.Random register
    fn exec_tlbwr(&mut self) -> ExecStatus {
        self.core.update_random();
        let index = (self.core.cp0_random as usize) % self.tlb.num_entries();
        let entry = self.create_tlb_entry_from_cp0();
        self.ip28_trace_tlb_write("tlbwr", index, &entry);
        self.tlb.write(index, entry);
        self.nanotlb_invalidate();

        if mips_log(MIPS_LOG_TLB) { dlog_dev!(LogModule::Mips, "TLBWR: Write Random Index {}\n{}", index, self.tlb.format_entry(index)); }

        #[cfg(feature = "tlbcheck")]
        {
            let bad = self.tlb.consistency_check(&format!("TLBWR idx={} pc={:#010x}", index, self.core.pc));
            self.handle_exec_complete();
            return if bad { EXEC_BREAKPOINT } else { EXEC_COMPLETE };
        }
        #[cfg(not(feature = "tlbcheck"))]
        self.handle_exec_complete()
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
    /// `WAIT` (R4600/R5000): stall until an interrupt is pending.
    ///
    /// **The instruction completes.** Per the MIPS spec the processor stalls
    /// *after* WAIT graduates, so an interrupt taken here reports
    /// `EPC = WAIT + 4` and `ERET` resumes at the *following* instruction —
    /// it does not re-execute the WAIT. That matters for IRIX's idle path
    /// (`R4Kasm.s`'s `wait_for_interrupt`), where the WAIT is followed by
    /// `NOP_0_4; j ra`: the function is meant to *return* once an interrupt
    /// has been taken. Leaving PC on the WAIT would make it spin forever and
    /// never reach the `j ra`.
    ///
    /// So: advance PC first (`handle_exec_complete`), then stall here.
    ///
    /// ## Why a spin and not a host sleep
    ///
    /// `step_cycles!` only runs in `step_preamble!`, i.e. once per dispatch —
    /// so a stall implemented *inside* this handler has to advance
    /// `hot.cycles` itself, which it does. Device timing workarounds depend on
    /// that progress (the SCSI one NetBSD needs in particular) and would stall
    /// outright if the CPU thread parked on an idle guest. `std::hint::spin_loop`
    /// keeps it polite to SMT siblings without stopping the clock.
    ///
    /// ## Exits
    ///
    /// Returns as soon as *any* of:
    /// - an enabled, unmasked interrupt is pending — the architectural wake;
    /// - a soft reset is requested (bit 63) — checked because a reset must
    ///   escape even a guest that executed WAIT with interrupts disabled;
    ///
    /// A WAIT that can never be released (`Status.IE` clear, or `EXL`/`ERL`
    /// set) is handled *before* the PC update instead: see the top of the
    /// body. It returns `EXEC_RETRY` with PC still on the WAIT, so the guest
    /// correctly never progresses while the CPU thread keeps returning to its
    /// run loop.
    fn exec_wait(&mut self) -> ExecStatus {
        // Unreleasable WAIT (IE clear, or EXL/ERL set): no interrupt can ever
        // end this stall. Checked BEFORE the PC update and answered with
        // EXEC_RETRY, so PC stays on the WAIT and the run loop simply
        // re-dispatches it. That reproduces the hardware behaviour (a WAIT
        // with interrupts disabled never completes) while keeping the CPU
        // thread returning to its loop every iteration, so the monitor, the
        // debugger and the soft-reset check all stay responsive — none of
        // which would be true if we spun inside this one instruction.
        //
        // IRIX never gets here: `wait_for_interrupt_fix_loc` requires the
        // `mtc0 a1,C0_SR` that re-enables interrupts to be *adjacent* to the
        // WAIT, which is what its "must be adjacent to avoid a race with an
        // interrupt" comment is about.
        if !self.core.interrupts_enabled() {
            return EXEC_RETRY;
        }

        // WAIT graduates: PC moves to the next instruction before the stall.
        let status = self.handle_exec_complete();

        let stall_start = std::time::Instant::now();
        let mut issued: u64 = 0;
        loop {
            let pending = self.core.hot.interrupts.load(Ordering::Relaxed);
            if pending & (SOFT_RESET_BIT | crate::cpu::smp::HOST_PARK_BIT) != 0 {
                break;
            }
            let ip = (self.core.cp0_cause | (pending as u32)) & crate::cpu::mips_core::CAUSE_IP_MASK;
            let im = self.core.cp0_status & crate::cpu::mips_core::STATUS_IM_MASK;
            if (ip & im) != 0 {
                break;
            }
            // `hot.cycles` MUST keep advancing while stalled, but at a paced
            // rate, not once per host iteration.
            //
            // Must advance: it is the clock other threads wait on. The
            // WD33C93A's deferred-interrupt path (`wd33c93a.rs`, "Required for
            // OpenBSD/NetBSD") spins until `cpu_cycles` has moved 10000 —
            // holding still here deadlocks it against a CPU waiting for the
            // very interrupt that spin is about to deliver.
            //
            // Must be paced: it is also the *virtual time base*
            // (`NS_PER_GUEST_CYCLE` = 10ns/cycle), and under `ci_clock` CP0
            // Count derives straight from it (`count_now`). Bumping once per
            // host iteration would run guest time at hundreds of millions of
            // cycles per real second inside one instruction — Count would leap
            // and timers would fire early.
            //
            // So: one guest cycle per 10ns of real time, which is exactly the
            // 1:1 rate `NS_PER_GUEST_CYCLE` defines, sampled off the host
            // clock rather than off loop iterations. An idle guest's clock
            // then tracks wall time the same way a running one does.
            let now = std::time::Instant::now();
            let elapsed_ns = now.duration_since(stall_start).as_nanos() as u64;
            // 10ns/cycle — the same rate `mips_core::NS_PER_GUEST_CYCLE`
            // defines, restated here because that constant is `ci_clock`-only
            // while this pacing must hold in every build.
            const STALL_NS_PER_CYCLE: u64 = 10;
            let want = elapsed_ns / STALL_NS_PER_CYCLE;
            if want > issued {
                let delta = want - issued;
                issued = want;
                unsafe {
                    let p = &mut self.core.hot.cycles as *mut u64;
                    std::ptr::write_volatile(p, std::ptr::read_volatile(p).wrapping_add(delta));
                }
            }
            std::hint::spin_loop();
        }

        status
    }

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
        // ERET clears ERL/EXL on cp0_status *directly*, not through write_cp0, so the
        // Status callback never fires — this is the only barrier on the privilege change,
        // and it has to be the full resync rather than a bare flush. Clearing EXL/ERL is
        // exactly what makes `get_privilege_mode()` stop reporting Kernel and start
        // reporting KSU, so `translate_fn` must be re-derived here or the guest runs user
        // code through the kernel translator. The flush alone cannot cover it: it
        // *guarantees* the next access misses and therefore calls `translate_fn`.
        self.resync_privilege_state();

        // `log mips mask cp0`. If the target's top half is already gone by
        // the time we get here, the guest wrote a truncated EPC; if it is
        // intact and the next fetch is truncated anyway, the loss is
        // downstream of this.
        if C::R10K_CACHE_OPS && mips_log_always(MIPS_LOG_CP0) {
            crate::dlog!(LogModule::Mips,
                "ip28epc: eret -> {target:#018x} (epc={:#018x} errorepc={:#018x} status={:#010x})",
                self.core.cp0_epc, self.core.cp0_errorepc, self.core.cp0_status);
        }

        // ERET jumps immediately without delay slot
        self.core.pc = target;

        // ERET has no delay slot of its own, but it can *be* one. An ERET in
        // the delay slot of a branch is architecturally UNPREDICTABLE, so the
        // case that matters is the reachable one: a NOT-TAKEN branch routes
        // its slot through `branch_delay(pc+8)` (so a fault in the slot still
        // reports EPC=branch/BD=1), which leaves `in_delay_slot` set with
        // `delay_slot_target` pointing just past the branch. Without this
        // clear, `handle_exec_complete` would then jump to that stale target
        // instead of EPC — hijacking the first instruction at the return
        // address. Cheap unconditional clear rather than a branch: ERET is
        // already a slow path.
        self.core.in_delay_slot = false;

        // Unlike every other exec_complete_pc_set caller, ERET's target is
        // compile-worthy only when it's actually returning from a syscall —
        // an ordinary exception return (TLB refill, interrupt, ...) lands
        // back on whatever PC it interrupted, which the interpreter was
        // already happily running before the exception hit, so retriggering
        // there is pure waste (and for a frequent interrupt source, a
        // standing one: same over-trigger shape jitv2_track_pcp's own doc
        // comment describes for page crossings). See
        // MipsCore::syscall_pending's doc comment.
        #[cfg(feature = "jitv2")]
        {
            self.core.jit_trigger = self.core.syscall_pending;
            self.core.syscall_pending = false;
        }

        // ERET's target is EPC/ErrorEPC — register-sourced, so it can be
        // misaligned. This is the third PC-install terminal (`exec_eret`
        // open-codes its completion rather than calling
        // `exec_complete_pc_set`), and the last one that needs the check; see
        // `check_pc_alignment` for the enumeration.
        //
        // Deliberately *after* the Status/privilege updates above: ERET's
        // architectural effect on EXL/ERL happens regardless, and the AdEL is
        // then delivered from the returned-to privilege level, which is what
        // sets EXL again. Checking before would leave the guest stuck in EXL
        // with no way to observe the fault.
        self.check_pc_alignment()
    }

    // ===== COP1 (FPU) Instructions =====

    /// Set Cause.CE to the given coprocessor number and dispatch EXC_CPU.
    /// Always used as an immediate `return self.cpu_unusable(..)` from a
    /// dispatch-target handler, so it's the terminal action itself.
    #[inline]
    fn cpu_unusable(&mut self, ce: u32) -> ExecStatus {
        self.cpu_unusable_report(ce);
        self.core.cp0_cause = (self.core.cp0_cause & !CAUSE_CE_MASK) | ((ce & 3) << CAUSE_CE_SHIFT);
        let s = exec_exception(EXC_CPU);
        self.handle_exception(s)
    }

    /// Diagnostic for every Coprocessor Unusable exception: which coprocessor,
    /// which instruction, and the CPU state that made it unusable.
    ///
    /// Worth having as its own funnel because the guest-side symptom is
    /// actively misleading. IRIX's `VEC_cpfault` panics with **"kernel used
    /// coprocessor"** whenever a CpU arrives with `SR_PREVMODE == 0`, and it
    /// does that *before* looking at `Cause.CE` (`vec_cpfault.s`) — so the
    /// message says "coprocessor" no matter which one, and invites blaming an
    /// FPU change when the faulting word can just as easily be a `CACHE` or
    /// `MFC0` going through the CP0 gate.
    ///
    /// Reports the live privilege inputs rather than just the answer, because
    /// `get_privilege_mode()` folds three things together (EXL, ERL, KSU) and
    /// which one is responsible is the whole question.
    ///
    /// `developer` only — a no-op `dlog_dev!` otherwise, so production pays
    /// nothing for it.
    #[cold]
    fn cpu_unusable_report(&mut self, #[allow(unused_variables)] ce: u32) {
        #[cfg(feature = "developer")]
        {
            use crate::cpu::mips_core::{STATUS_CU0, STATUS_CU1, STATUS_EXL, STATUS_ERL,
                                   STATUS_IE, STATUS_KSU_SHIFT};
            let st = self.core.cp0_status;
            let pc = self.core.pc;

            // The faulting word: re-read through the debug path so this never
            // perturbs cache state or takes an exception of its own. Done
            // before taking the symbol lock — `debug_translate` needs `&mut
            // self`, which the lock's borrow would otherwise block.
            let raw = {
                let tr = self.debug_translate(pc);
                if tr.is_exception() {
                    None
                } else {
                    let r = self.sysad.read32(tr.phys);
                    if r.is_ok() { Some(r.data) } else { None }
                }
            };

            let symbols = self.symbols.lock();
            let sym_str = format_pc_symbol(pc, &symbols);
            let (raw_str, dis) = match raw {
                Some(w) => (format!("{:08x}", w), mips_dis::disassemble(w, pc, Some(&symbols))),
                None => ("????????".to_string(), "<unreadable>".to_string()),
            };

            let ksu = (st >> STATUS_KSU_SHIFT) & 0x3;
            dlog_dev!(LogModule::Mips,
                "CpU: coprocessor {} unusable at {:016x}{}: {} {}\n                       Status={:08x} [CU1={} CU0={} EXL={} ERL={} IE={} KSU={}] priv={:?} -> {}",
                ce, pc, sym_str, raw_str, dis,
                st,
                (st & STATUS_CU1 != 0) as u8,
                (st & STATUS_CU0 != 0) as u8,
                (st & STATUS_EXL != 0) as u8,
                (st & STATUS_ERL != 0) as u8,
                (st & STATUS_IE  != 0) as u8,
                ksu,
                self.core.get_privilege_mode(),
                // Name the specific reason this instruction was refused.
                if ce == 1 {
                    "CU1 clear (FPU not enabled for this context)"
                } else if self.core.get_privilege_mode() != crate::cpu::mips_core::PrivilegeMode::Kernel {
                    "not kernel mode and CU0 clear"
                } else {
                    "kernel mode with CU0 clear — should not happen for CE=0"
                });
        }
    }

    /// After a FPU arithmetic op has computed its result but *before* it is
    /// committed: fold in caller-computed exception flags (never read from
    /// the host FPU — see `clanker/jitv2/fpu-flags-are-computed-not-read.md`),
    /// update FCSR cause bits, and either commit the result (calling
    /// `write_result`) and raise EXC_FPE if any enabled exception fired, or
    /// — if trapping — leave the destination register and the sticky Flag
    /// field untouched. R4000 manual: "When a floating-point exception is
    /// taken, no results are stored, and the only state affected is the
    /// Cause bit" and "the flag bits are not set by the hardware" on a taken
    /// exception. `write_result` must be idempotent-free of side effects
    /// other than the register write itself (it may not run).
    ///
    /// `flags`: bits [6:2] (FV,FZ,FO,FU,FI) this instruction raised, computed
    /// by the caller from operand/result bit patterns — e.g. `FCSR_FV` for a
    /// signalling-NaN operand, `FCSR_FZ` for divide-by-zero. Always 0 for
    /// plain ADD/SUB/MUL/DIV/SQRT/RECIP/RSQRT/MADD-family unless one of those
    /// specific conditions applies — Inexact and Overflow are deliberately
    /// never set here (see the same rules note: not worth the IR to compute
    /// for ordinary arithmetic, and nothing in IRIX depends on them).
    ///
    /// Always used as the terminal tail call of an FPU arithmetic exec_*
    /// handler (`self.fpu_update_fcsr(flags, |e| ...)`, nothing runs after),
    /// so this is the terminal action itself: completes normally or
    /// dispatches EXC_FPE.
    #[inline]
    fn fpu_update_fcsr(&mut self, flags: u32, write_result: impl FnOnce(&mut Self)) -> ExecStatus {
        self.fpu_update_fcsr_full(flags, None, move |exec, _flush| write_result(exec))
    }

    /// `result_is_denorm`: `Some(result_is_negative)` when the (already-
    /// computed, not-yet-written) result is a denormalized nonzero value —
    /// checked by the caller via `result.is_subnormal()` before calling in,
    /// since only the caller knows the result's type/width. Implements the
    /// R4000/VR5000 manuals' full Table 9-1/7-1 Underflow rule (both manuals
    /// are word-for-word identical here — see
    /// clanker/cpu/fpu-exception-model-vs-r4400.md): a denormalized result
    /// is never delivered by hardware. With FS clear, or with FS set but
    /// Underflow/Inexact trapping enabled, it is an Unimplemented Operation
    /// (E) instead — E has no Enable/Flag bit and always traps, and per the
    /// manual "the result register is not modified". Only with FS set and
    /// neither U nor I enabled does hardware flush the result to a signed
    /// zero (sign of the *unrounded* result) and set Cause.U and Cause.I —
    /// not whatever the host FPU happened to report. `write_result`'s `bool`
    /// argument is `true` exactly on that flush path — the real, non-denormal
    /// result should be written when `false`, a signed zero of the given sign
    /// when `true` (the value itself, not just the sign, is the caller's job
    /// since S/D width and the FPR write function differ per call site).
    fn fpu_update_fcsr_full(
        &mut self,
        flags: u32,
        result_is_denorm: Option<bool>,
        write_result: impl FnOnce(&mut Self, bool),
    ) -> ExecStatus {
        // Cause holds only the last instruction's exceptions (R4000 manual:
        // "the results of only one instruction") and is rewritten every FP
        // instruction, unlike the sticky Flag field. Clear it unconditionally,
        // even when this instruction raised nothing.
        self.core.fpu_fcsr &= !FCSR_CM;

        if result_is_denorm.is_some() {
            let fs_set = (self.core.fpu_fcsr & FCSR_FS) != 0;
            let ui_enabled = (self.core.fpu_fcsr & (FCSR_UM | FCSR_IM)) != 0;
            if !fs_set || ui_enabled {
                return self.fpu_unimplemented();
            }
            // FS set, neither U nor I enabled: flush to the signed zero of
            // the intermediate result and force Cause.U+I — regardless of
            // what the host's own underflow/inexact flags said.
            self.core.fpu_fcsr |= FCSR_CU | FCSR_CI;
            write_result(self, true);
            self.core.fpu_fcsr |= FCSR_FU | FCSR_FI;
            return self.handle_exec_complete();
        }

        if flags == 0 {
            write_result(self, false);
            return self.handle_exec_complete();
        }
        // Promote flag bits [6:2] → cause bits [16:12] (shift up by 10)
        let causes = (flags & FCSR_FM) << 10;
        self.core.fpu_fcsr |= causes;
        // Raise FPE if any cause bit has its corresponding enable bit set.
        // Causes are 5 bits above enables: (causes >> 5) aligns them with enables.
        let traps = ((causes >> 5) & (self.core.fpu_fcsr & FCSR_EM)) != 0;
        if traps {
            // Trapped: neither the destination register nor the sticky Flag
            // field are touched — only Cause (already set above).
            let s = exec_exception(EXC_FPE);
            return self.handle_exception(s);
        }
        // Not trapping: commit the result and accumulate the sticky Flag
        // field (cleared explicitly by software via SetFPSR/CTC1).
        write_result(self, false);
        self.core.fpu_fcsr |= flags & FCSR_FM;
        self.handle_exec_complete()
    }

    /// Unimplemented Operation (Cause.E): R4000/VR5000 manuals, ch 7/9.4.6 —
    /// "no Enable bit and no Flag bit; whenever this exception occurs, an
    /// unimplemented exception trap is taken" and "the result register is
    /// not modified". Shared terminal helper for every E-raising site
    /// (denormal/qNaN operand, denormal/underflowed result without the FS
    /// escape hatch).
    #[inline]
    fn fpu_unimplemented(&mut self) -> ExecStatus {
        self.core.fpu_fcsr = (self.core.fpu_fcsr & !FCSR_CM) | FCSR_CE;
        let s = exec_exception(EXC_FPE);
        self.handle_exception(s)
    }

    /// R4000/VR5000 manuals: "Denormalized operand" and "Quiet Not a Number
    /// operand" are Unimplemented Operation for any convert or computational
    /// instruction (explicitly excepting Compare, and Moves which "do not
    /// trap if their operands are either denormalized or NaNs"). Call before
    /// computing on either operand of ADD/SUB/MUL/DIV/SQRT; returns `Some`
    /// (the terminal ExecStatus, already dispatched) when the operand check
    /// itself must end the instruction, `None` to proceed to computation.
    #[inline]
    fn fpu_check_denorm_operand_s(&mut self, bits: u32) -> Option<ExecStatus> {
        let v = f32::from_bits(bits);
        (v.is_subnormal() || is_qnan_s(bits)).then(|| self.fpu_unimplemented())
    }
    #[inline]
    fn fpu_check_denorm_operand_d(&mut self, bits: u64) -> Option<ExecStatus> {
        let v = f64::from_bits(bits);
        (v.is_subnormal() || is_qnan_d(bits)).then(|| self.fpu_unimplemented())
    }

    /// Flags [6:2] for ADD/SUB/MUL (S or D): Invalid if either operand is a
    /// signalling NaN (the case `fpu_check_denorm_operand_s/d`'s qNaN check
    /// doesn't catch), else 0 — feeds `fpu_update_fcsr`/`_full`. See
    /// `clanker/jitv2/fpu-flags-are-computed-not-read.md`.
    #[inline]
    fn fpu_arith_flags_snan_only_s(fs_bits: u32, ft_bits: u32) -> u32 {
        if is_snan_s(fs_bits) || is_snan_s(ft_bits) { FCSR_FV } else { 0 }
    }
    #[inline]
    fn fpu_arith_flags_snan_only_d(fs_bits: u64, ft_bits: u64) -> u32 {
        if is_snan_d(fs_bits) || is_snan_d(ft_bits) { FCSR_FV } else { 0 }
    }

    /// MADD/MSUB/NMADD/NMSUB (fr op= fs*ft, three source operands): Invalid
    /// if any of the three is a signalling NaN, else 0.
    #[inline]
    fn fpu_arith_flags_snan_only3_s(a_bits: u32, b_bits: u32, c_bits: u32) -> u32 {
        if is_snan_s(a_bits) || is_snan_s(b_bits) || is_snan_s(c_bits) { FCSR_FV } else { 0 }
    }
    #[inline]
    fn fpu_arith_flags_snan_only3_d(a_bits: u64, b_bits: u64, c_bits: u64) -> u32 {
        if is_snan_d(a_bits) || is_snan_d(b_bits) || is_snan_d(c_bits) { FCSR_FV } else { 0 }
    }

    /// Flags [6:2] for DIV/RECIP (S or D): Invalid takes priority over
    /// divide-by-zero (a NaN dividend/divisor is Invalid, never Z), else Z
    /// when the divisor is zero and the dividend is a finite non-NaN value —
    /// matches the R4000 manual's Divide-by-zero definition ("the divisor is
    /// zero and the dividend is a finite nonzero number"; zero/zero and
    /// inf/zero are Invalid on real hardware, but qNaN operand already
    /// traps via `fpu_check_denorm_operand_s/d` before this runs, and this
    /// emulator doesn't distinguish 0/0 from finite/0 — both give Z, which
    /// is the "gross problem" this exists to catch, not hardware-exact
    /// classification of every zero/zero corner case).
    #[inline]
    fn fpu_arith_flags_div_s(fs_bits: u32, ft_bits: u32) -> u32 {
        let snan = Self::fpu_arith_flags_snan_only_s(fs_bits, ft_bits);
        if snan != 0 { return snan; }
        if is_pos_zero_or_neg_zero_s(ft_bits) { FCSR_FZ } else { 0 }
    }
    #[inline]
    fn fpu_arith_flags_div_d(fs_bits: u64, ft_bits: u64) -> u32 {
        let snan = Self::fpu_arith_flags_snan_only_d(fs_bits, ft_bits);
        if snan != 0 { return snan; }
        if is_pos_zero_or_neg_zero_d(ft_bits) { FCSR_FZ } else { 0 }
    }

    /// Flags [6:2] for SQRT/RSQRT (S or D): Invalid for a signalling-NaN
    /// operand or a negative nonzero operand (real sqrt of a negative is
    /// Invalid; -0.0 is not, per IEEE 754 sqrt(-0)=-0), else 0.
    #[inline]
    fn fpu_arith_flags_sqrt_s(fs_bits: u32) -> u32 {
        if is_snan_s(fs_bits) || is_neg_nonzero_s(fs_bits) { FCSR_FV } else { 0 }
    }
    #[inline]
    fn fpu_arith_flags_sqrt_d(fs_bits: u64) -> u32 {
        if is_snan_d(fs_bits) || is_neg_nonzero_d(fs_bits) { FCSR_FV } else { 0 }
    }

    /// ABS.fmt/NEG.fmt: unlike a MOV, these are arithmetic operations (R4000
    /// manual: "absolute value (ABS) and negate (NEG) are considered to be
    /// arithmetic operations and cause this exception if one or both
    /// operands is a signaling NaN") but they never invoke the host FPU, so
    /// there's no `platform::get_fpu_status()` to read — just the one
    /// possible cause, checked against the source operand directly. Same
    /// commit-only-if-not-trapping shape as `fpu_update_fcsr`.
    #[inline]
    fn fpu_check_snan_operand(&mut self, is_snan: bool, write_result: impl FnOnce(&mut Self)) -> ExecStatus {
        self.core.fpu_fcsr = (self.core.fpu_fcsr & !FCSR_CM) | if is_snan { FCSR_CV } else { 0 };
        if is_snan {
            if (self.core.fpu_fcsr & 0x800) != 0 { // EV enable bit
                let s = exec_exception(EXC_FPE);
                return self.handle_exception(s);
            }
            self.core.fpu_fcsr |= FCSR_FV; // untrapped: set sticky Flag V too
        }
        write_result(self);
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
        let fs_bits = (self.fpr_read_w)(&self.core, fs_reg);
        let ft_bits = (self.fpr_read_w)(&self.core, ft_reg);
        if let Some(s) = self.fpu_check_denorm_operand_s(fs_bits).or_else(|| self.fpu_check_denorm_operand_s(ft_bits)) { return s; }
        let flags = Self::fpu_arith_flags_snan_only_s(fs_bits, ft_bits);
        let result = f32::from_bits(fs_bits) + f32::from_bits(ft_bits);
        let write_w = self.fpr_write_w;
        let denorm = result.is_subnormal().then(|| result.is_sign_negative());
        self.fpu_update_fcsr_full(flags, denorm, |exec, flush| write_w(&mut exec.core, fd_reg, if flush { if result.is_sign_negative() { 0x8000_0000 } else { 0 } } else { (result).to_bits() }))
    }
    fn exec_fsub_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let ft_reg = d.rt as u32; let fd_reg = d.sa as u32;
        let fs_bits = (self.fpr_read_w)(&self.core, fs_reg);
        let ft_bits = (self.fpr_read_w)(&self.core, ft_reg);
        if let Some(s) = self.fpu_check_denorm_operand_s(fs_bits).or_else(|| self.fpu_check_denorm_operand_s(ft_bits)) { return s; }
        let flags = Self::fpu_arith_flags_snan_only_s(fs_bits, ft_bits);
        let result = f32::from_bits(fs_bits) - f32::from_bits(ft_bits);
        let write_w = self.fpr_write_w;
        let denorm = result.is_subnormal().then(|| result.is_sign_negative());
        self.fpu_update_fcsr_full(flags, denorm, |exec, flush| write_w(&mut exec.core, fd_reg, if flush { if result.is_sign_negative() { 0x8000_0000 } else { 0 } } else { (result).to_bits() }))
    }
    fn exec_fmul_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let ft_reg = d.rt as u32; let fd_reg = d.sa as u32;
        let fs_bits = (self.fpr_read_w)(&self.core, fs_reg);
        let ft_bits = (self.fpr_read_w)(&self.core, ft_reg);
        if let Some(s) = self.fpu_check_denorm_operand_s(fs_bits).or_else(|| self.fpu_check_denorm_operand_s(ft_bits)) { return s; }
        let flags = Self::fpu_arith_flags_snan_only_s(fs_bits, ft_bits);
        let result = f32::from_bits(fs_bits) * f32::from_bits(ft_bits);
        let write_w = self.fpr_write_w;
        let denorm = result.is_subnormal().then(|| result.is_sign_negative());
        self.fpu_update_fcsr_full(flags, denorm, |exec, flush| write_w(&mut exec.core, fd_reg, if flush { if result.is_sign_negative() { 0x8000_0000 } else { 0 } } else { (result).to_bits() }))
    }
    fn exec_fdiv_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let ft_reg = d.rt as u32; let fd_reg = d.sa as u32;
        let fs_bits = (self.fpr_read_w)(&self.core, fs_reg);
        let ft_bits = (self.fpr_read_w)(&self.core, ft_reg);
        if let Some(s) = self.fpu_check_denorm_operand_s(fs_bits).or_else(|| self.fpu_check_denorm_operand_s(ft_bits)) { return s; }
        let flags = Self::fpu_arith_flags_div_s(fs_bits, ft_bits);
        let fs_val = f32::from_bits(fs_bits);
        let ft_val = f32::from_bits(ft_bits);
        let result = fs_val / ft_val;
        if mips_log(MIPS_LOG_FPU) {
            dlog_dev!(LogModule::Mips, "FPU div.s PC={:016x} {} / {} = {}", self.core.pc, fs_val, ft_val, result);
        }
        let write_w = self.fpr_write_w;
        let denorm = result.is_subnormal().then(|| result.is_sign_negative());
        self.fpu_update_fcsr_full(flags, denorm, |exec, flush| write_w(&mut exec.core, fd_reg, if flush { if result.is_sign_negative() { 0x8000_0000 } else { 0 } } else { (result).to_bits() }))
    }
    fn exec_fsqrt_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        let fs_bits = (self.fpr_read_w)(&self.core, fs_reg);
        if let Some(s) = self.fpu_check_denorm_operand_s(fs_bits) { return s; }
        let flags = Self::fpu_arith_flags_sqrt_s(fs_bits);
        let result = f32::from_bits(fs_bits).sqrt();
        let write_w = self.fpr_write_w;
        let denorm = result.is_subnormal().then(|| result.is_sign_negative());
        self.fpu_update_fcsr_full(flags, denorm, |exec, flush| write_w(&mut exec.core, fd_reg, if flush { if result.is_sign_negative() { 0x8000_0000 } else { 0 } } else { (result).to_bits() }))
    }
    fn exec_fabs_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        let fs_bits = (self.fpr_read_w)(&self.core, fs_reg);
        let result = f32::from_bits(fs_bits).abs();
        let write_w = self.fpr_write_w;
        self.fpu_check_snan_operand(is_snan_s(fs_bits), |exec| write_w(&mut exec.core, fd_reg, (result).to_bits()))
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
        let fs_bits = (self.fpr_read_w)(&self.core, fs_reg);
        let result = -f32::from_bits(fs_bits);
        let write_w = self.fpr_write_w;
        self.fpu_check_snan_operand(is_snan_s(fs_bits), |exec| write_w(&mut exec.core, fd_reg, (result).to_bits()))
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
        self.fpu_cvt_to_int(d.rd as u32, d.sa as u32, false, true, RM_NEAREST_EVEN)
    }
    fn exec_ftrunc_l_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        self.fpu_cvt_to_int(d.rd as u32, d.sa as u32, false, true, RM_TOWARD_ZERO)
    }
    fn exec_fceil_l_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        self.fpu_cvt_to_int(d.rd as u32, d.sa as u32, false, true, RM_TOWARD_POS_INF)
    }
    fn exec_ffloor_l_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        self.fpu_cvt_to_int(d.rd as u32, d.sa as u32, false, true, RM_TOWARD_NEG_INF)
    }
    fn exec_fround_w_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        self.fpu_cvt_to_int(d.rd as u32, d.sa as u32, false, false, RM_NEAREST_EVEN)
    }
    fn exec_ftrunc_w_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        if mips_log(MIPS_LOG_FPU) {
            let fs_val = f32::from_bits((self.fpr_read_w)(&self.core, fs_reg));
            dlog_dev!(LogModule::Mips, "FPU trunc.w.s PC={:016x} {}", self.core.pc, fs_val);
        }
        self.fpu_cvt_to_int(fs_reg, fd_reg, false, false, RM_TOWARD_ZERO)
    }
    fn exec_fceil_w_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        self.fpu_cvt_to_int(d.rd as u32, d.sa as u32, false, false, RM_TOWARD_POS_INF)
    }
    fn exec_ffloor_w_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        self.fpu_cvt_to_int(d.rd as u32, d.sa as u32, false, false, RM_TOWARD_NEG_INF)
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
        let fs_bits = (self.fpr_read_w)(&self.core, fs_reg);
        let flags = Self::fpu_arith_flags_div_s(1.0f32.to_bits(), fs_bits);
        let fs_val = f32::from_bits(fs_bits);
        let result = 1.0 / fs_val;
        if mips_log(MIPS_LOG_FPU) {
            dlog_dev!(LogModule::Mips, "FPU recip.s PC={:016x} 1 / {} = {}", self.core.pc, fs_val, result);
        }
        let write_w = self.fpr_write_w;
        self.fpu_update_fcsr(flags, |exec| write_w(&mut exec.core, fd_reg, (result).to_bits()))
    }
    fn exec_frsqrt_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        let fs_bits = (self.fpr_read_w)(&self.core, fs_reg);
        // RSQRT = 1/sqrt(x): Invalid for a negative operand (real sqrt) or
        // signalling NaN, else Divide-by-zero when x is zero (sqrt(0)=0, so
        // 1/sqrt(0) divides by zero) — union of both component checks.
        let flags = {
            let sqrt_flags = Self::fpu_arith_flags_sqrt_s(fs_bits);
            if sqrt_flags != 0 { sqrt_flags } else if is_pos_zero_or_neg_zero_s(fs_bits) { FCSR_FZ } else { 0 }
        };
        let result = 1.0 / f32::from_bits(fs_bits).sqrt();
        let write_w = self.fpr_write_w;
        self.fpu_update_fcsr(flags, |exec| write_w(&mut exec.core, fd_reg, (result).to_bits()))
    }
    fn exec_fcvt_d_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        let result = f32::from_bits((self.fpr_read_w)(&self.core, fs_reg)) as f64;
        let write_d = self.fpr_write_d;
        self.fpu_update_fcsr(0, |exec| write_d(&mut exec.core, fd_reg, result))
    }
    /// Shared tail for every ROUND/TRUNC/CEIL/FLOOR/CVT.W/CVT.L handler
    /// (S or D source, W or L dest, fixed or FCSR-dynamic rounding mode):
    /// call the portable `cvt_to_int_and_commit` and turn its trap/no-trap
    /// result into the usual `ExecStatus`. `fpu_fcsr` is only read for
    /// Cause, so no destructuring/write-closure dance is needed here — the
    /// shared function commits (or doesn't) directly against `self.core`.
    #[inline]
    fn fpu_cvt_to_int(&mut self, fs_reg: u32, fd_reg: u32, src_f64: bool, dst_i64: bool, rm: u8) -> ExecStatus {
        let fr1 = (self.core.cp0_status & STATUS_FR) != 0;
        if cvt_to_int_and_commit(&mut self.core, fs_reg, fd_reg, fr1, src_f64, dst_i64, rm) {
            let s = exec_exception(EXC_FPE);
            self.handle_exception(s)
        } else {
            self.handle_exec_complete()
        }
    }
    /// CVT.S.W/D.W/S.L/D.L: shared tail calling `cvt_int_to_float_and_commit`.
    #[inline]
    fn fpu_cvt_int_to_float(&mut self, fs_reg: u32, fd_reg: u32, src_i64: bool, dst_f64: bool) -> ExecStatus {
        let fr1 = (self.core.cp0_status & STATUS_FR) != 0;
        if cvt_int_to_float_and_commit(&mut self.core, fs_reg, fd_reg, fr1, src_i64, dst_f64) {
            let s = exec_exception(EXC_FPE);
            self.handle_exception(s)
        } else {
            self.handle_exec_complete()
        }
    }
    /// CVT.S.D: shared tail calling `cvt_d_to_s_and_commit`.
    #[inline]
    fn fpu_cvt_d_to_s(&mut self, fs_reg: u32, fd_reg: u32) -> ExecStatus {
        let fr1 = (self.core.cp0_status & STATUS_FR) != 0;
        if cvt_d_to_s_and_commit(&mut self.core, fs_reg, fd_reg, fr1) {
            let s = exec_exception(EXC_FPE);
            self.handle_exception(s)
        } else {
            self.handle_exec_complete()
        }
    }

    fn exec_fcvt_w_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        if mips_log(MIPS_LOG_FPU) {
            let fs_val = f32::from_bits((self.fpr_read_w)(&self.core, fs_reg));
            dlog_dev!(LogModule::Mips, "FPU cvt.w.s PC={:016x} {}", self.core.pc, fs_val);
        }
        self.fpu_cvt_to_int(fs_reg, fd_reg, false, false, (self.core.fpu_fcsr & 0x3) as u8)
    }
    fn exec_fcvt_l_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        self.fpu_cvt_to_int(fs_reg, fd_reg, false, true, (self.core.fpu_fcsr & 0x3) as u8)
    }
    // S-format compare (all 16 conditions share one handler; funct 0x30-0x3F)
    fn exec_fcc_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let ft_reg = d.rt as u32;
        let funct_val = d.funct as u32;
        let fs_bits = (self.fpr_read_w)(&self.core, fs_reg);
        let ft_bits = (self.fpr_read_w)(&self.core, ft_reg);
        let fs_val = f32::from_bits(fs_bits);
        let ft_val = f32::from_bits(ft_bits);
        // Invalid is raised whenever either operand is a signalling NaN
        // (Table 7-2, "Comparison ... on a signaling NaN"), and additionally
        // for the eight signalling predicates (cond 0x8-0xF, funct bit 3)
        // whenever either operand is any NaN.
        let is_signaling_predicate = (funct_val & 0x8) != 0;
        let has_nan = fs_val.is_nan() || ft_val.is_nan();
        let has_snan = is_snan_s(fs_bits) || is_snan_s(ft_bits);
        let invalid = has_snan || (is_signaling_predicate && has_nan);
        // Cause reflects only the last instruction (R4000 manual), so it is
        // rewritten every compare, not just when this one raises Invalid.
        self.core.fpu_fcsr = (self.core.fpu_fcsr & !FCSR_CM) | if invalid { FCSR_CV } else { 0 };
        if invalid {
            if (self.core.fpu_fcsr & 0x800) != 0 { // EV enable bit
                // Trapped: Cause is set above, but the Flag field is not —
                // R4000 manual: "the flag bits are not set by the hardware"
                // when a floating-point exception is taken.
                let s = exec_exception(EXC_FPE);
                return self.handle_exception(s);
            }
            self.core.fpu_fcsr |= FCSR_FV; // untrapped: set sticky Flag V too
        }
        let cond = self.fpu_compare_s(fs_val, ft_val, funct_val);
        let cc = (d.raw >> 8) & 0x7;
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
        let read_d = self.fpr_read_d; let write_d = self.fpr_write_d;
        let fs_val = read_d(&self.core, fs_reg); let ft_val = read_d(&self.core, ft_reg);
        if let Some(s) = self.fpu_check_denorm_operand_d(fs_val.to_bits()).or_else(|| self.fpu_check_denorm_operand_d(ft_val.to_bits())) { return s; }
        let flags = Self::fpu_arith_flags_snan_only_d(fs_val.to_bits(), ft_val.to_bits());
        let result = fs_val + ft_val;
        let denorm = result.is_subnormal().then(|| result.is_sign_negative());
        self.fpu_update_fcsr_full(flags, denorm, |exec, flush| write_d(&mut exec.core, fd_reg, if flush { if result.is_sign_negative() { -0.0 } else { 0.0 } } else { result }))
    }
    fn exec_fsub_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let ft_reg = d.rt as u32; let fd_reg = d.sa as u32;
        let read_d = self.fpr_read_d; let write_d = self.fpr_write_d;
        let fs_val = read_d(&self.core, fs_reg); let ft_val = read_d(&self.core, ft_reg);
        if let Some(s) = self.fpu_check_denorm_operand_d(fs_val.to_bits()).or_else(|| self.fpu_check_denorm_operand_d(ft_val.to_bits())) { return s; }
        let flags = Self::fpu_arith_flags_snan_only_d(fs_val.to_bits(), ft_val.to_bits());
        let result = fs_val - ft_val;
        let denorm = result.is_subnormal().then(|| result.is_sign_negative());
        self.fpu_update_fcsr_full(flags, denorm, |exec, flush| write_d(&mut exec.core, fd_reg, if flush { if result.is_sign_negative() { -0.0 } else { 0.0 } } else { result }))
    }
    fn exec_fmul_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let ft_reg = d.rt as u32; let fd_reg = d.sa as u32;
        let read_d = self.fpr_read_d; let write_d = self.fpr_write_d;
        let fs_val = read_d(&self.core, fs_reg); let ft_val = read_d(&self.core, ft_reg);
        if let Some(s) = self.fpu_check_denorm_operand_d(fs_val.to_bits()).or_else(|| self.fpu_check_denorm_operand_d(ft_val.to_bits())) { return s; }
        let flags = Self::fpu_arith_flags_snan_only_d(fs_val.to_bits(), ft_val.to_bits());
        let result = fs_val * ft_val;
        let denorm = result.is_subnormal().then(|| result.is_sign_negative());
        self.fpu_update_fcsr_full(flags, denorm, |exec, flush| write_d(&mut exec.core, fd_reg, if flush { if result.is_sign_negative() { -0.0 } else { 0.0 } } else { result }))
    }
    fn exec_fdiv_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let ft_reg = d.rt as u32; let fd_reg = d.sa as u32;
        let read_d = self.fpr_read_d; let write_d = self.fpr_write_d;
        let fs_val = read_d(&self.core, fs_reg);
        let ft_val = read_d(&self.core, ft_reg);
        if let Some(s) = self.fpu_check_denorm_operand_d(fs_val.to_bits()).or_else(|| self.fpu_check_denorm_operand_d(ft_val.to_bits())) { return s; }
        let flags = Self::fpu_arith_flags_div_d(fs_val.to_bits(), ft_val.to_bits());
        let result = fs_val / ft_val;
        if mips_log(MIPS_LOG_FPU) {
            dlog_dev!(LogModule::Mips, "FPU div.d PC={:016x} {} / {} = {}", self.core.pc, fs_val, ft_val, result);
        }
        let denorm = result.is_subnormal().then(|| result.is_sign_negative());
        self.fpu_update_fcsr_full(flags, denorm, |exec, flush| write_d(&mut exec.core, fd_reg, if flush { if result.is_sign_negative() { -0.0 } else { 0.0 } } else { result }))
    }
    fn exec_fsqrt_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        let read_d = self.fpr_read_d; let write_d = self.fpr_write_d;
        let fs_val = read_d(&self.core, fs_reg);
        if let Some(s) = self.fpu_check_denorm_operand_d(fs_val.to_bits()) { return s; }
        let flags = Self::fpu_arith_flags_sqrt_d(fs_val.to_bits());
        let result = fs_val.sqrt();
        let denorm = result.is_subnormal().then(|| result.is_sign_negative());
        self.fpu_update_fcsr_full(flags, denorm, |exec, flush| write_d(&mut exec.core, fd_reg, if flush { if result.is_sign_negative() { -0.0 } else { 0.0 } } else { result }))
    }
    fn exec_fabs_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        let read_d = self.fpr_read_d; let write_d = self.fpr_write_d;
        let fs_val = read_d(&self.core, fs_reg);
        let result = fs_val.abs();
        self.fpu_check_snan_operand(is_snan_d(fs_val.to_bits()), |exec| write_d(&mut exec.core, fd_reg, result))
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
        let fs_val = read_d(&self.core, fs_reg);
        let result = -fs_val;
        self.fpu_check_snan_operand(is_snan_d(fs_val.to_bits()), |exec| write_d(&mut exec.core, fd_reg, result))
    }
    fn exec_fround_l_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        self.fpu_cvt_to_int(d.rd as u32, d.sa as u32, true, true, RM_NEAREST_EVEN)
    }
    fn exec_ftrunc_l_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        self.fpu_cvt_to_int(d.rd as u32, d.sa as u32, true, true, RM_TOWARD_ZERO)
    }
    fn exec_fceil_l_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        self.fpu_cvt_to_int(d.rd as u32, d.sa as u32, true, true, RM_TOWARD_POS_INF)
    }
    fn exec_ffloor_l_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        self.fpu_cvt_to_int(d.rd as u32, d.sa as u32, true, true, RM_TOWARD_NEG_INF)
    }
    fn exec_fround_w_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        self.fpu_cvt_to_int(d.rd as u32, d.sa as u32, true, false, RM_NEAREST_EVEN)
    }
    fn exec_ftrunc_w_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        if mips_log(MIPS_LOG_FPU) {
            let fs_val = (self.fpr_read_d)(&self.core, fs_reg);
            dlog_dev!(LogModule::Mips, "FPU trunc.w.d PC={:016x} {}", self.core.pc, fs_val);
        }
        self.fpu_cvt_to_int(fs_reg, fd_reg, true, false, RM_TOWARD_ZERO)
    }
    fn exec_fceil_w_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        self.fpu_cvt_to_int(d.rd as u32, d.sa as u32, true, false, RM_TOWARD_POS_INF)
    }
    fn exec_ffloor_w_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        self.fpu_cvt_to_int(d.rd as u32, d.sa as u32, true, false, RM_TOWARD_NEG_INF)
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
        let read_d = self.fpr_read_d; let write_d = self.fpr_write_d;
        let fs_val = read_d(&self.core, fs_reg);
        let flags = Self::fpu_arith_flags_div_d(1.0f64.to_bits(), fs_val.to_bits());
        let result = 1.0 / fs_val;
        if mips_log(MIPS_LOG_FPU) {
            dlog_dev!(LogModule::Mips, "FPU recip.d PC={:016x} 1 / {} = {}", self.core.pc, fs_val, result);
        }
        self.fpu_update_fcsr(flags, |exec| write_d(&mut exec.core, fd_reg, result))
    }
    fn exec_frsqrt_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        let read_d = self.fpr_read_d; let write_d = self.fpr_write_d;
        let fs_val = read_d(&self.core, fs_reg);
        let fs_bits = fs_val.to_bits();
        let flags = {
            let sqrt_flags = Self::fpu_arith_flags_sqrt_d(fs_bits);
            if sqrt_flags != 0 { sqrt_flags } else if is_pos_zero_or_neg_zero_d(fs_bits) { FCSR_FZ } else { 0 }
        };
        let result = 1.0 / fs_val.sqrt();
        self.fpu_update_fcsr(flags, |exec| write_d(&mut exec.core, fd_reg, result))
    }
    fn exec_fcvt_s_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        self.fpu_cvt_d_to_s(d.rd as u32, d.sa as u32)
    }
    fn exec_fcvt_w_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        if mips_log(MIPS_LOG_FPU) {
            let fs_val = (self.fpr_read_d)(&self.core, fs_reg);
            dlog_dev!(LogModule::Mips, "FPU cvt.w.d PC={:016x} {}", self.core.pc, fs_val);
        }
        self.fpu_cvt_to_int(fs_reg, fd_reg, true, false, (self.core.fpu_fcsr & 0x3) as u8)
    }
    fn exec_fcvt_l_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let fd_reg = d.sa as u32;
        self.fpu_cvt_to_int(fs_reg, fd_reg, true, true, (self.core.fpu_fcsr & 0x3) as u8)
    }
    // D-format compare (all 16 conditions share one handler; funct 0x30-0x3F)
    fn exec_fcc_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fs_reg = d.rd as u32; let ft_reg = d.rt as u32;
        let funct_val = d.funct as u32;
        let read_d = self.fpr_read_d;
        let fs_val = read_d(&self.core, fs_reg);
        let ft_val = read_d(&self.core, ft_reg);
        // Invalid is raised whenever either operand is a signalling NaN
        // (Table 7-2, "Comparison ... on a signaling NaN"), and additionally
        // for the eight signalling predicates (cond 0x8-0xF, funct bit 3)
        // whenever either operand is any NaN.
        let is_signaling_predicate = (funct_val & 0x8) != 0;
        let has_nan = fs_val.is_nan() || ft_val.is_nan();
        let has_snan = is_snan_d(fs_val.to_bits()) || is_snan_d(ft_val.to_bits());
        let invalid = has_snan || (is_signaling_predicate && has_nan);
        // Cause reflects only the last instruction (R4000 manual), so it is
        // rewritten every compare, not just when this one raises Invalid.
        self.core.fpu_fcsr = (self.core.fpu_fcsr & !FCSR_CM) | if invalid { FCSR_CV } else { 0 };
        if invalid {
            if (self.core.fpu_fcsr & 0x800) != 0 { // EV enable bit
                // Trapped: Cause is set above, but the Flag field is not —
                // R4000 manual: "the flag bits are not set by the hardware"
                // when a floating-point exception is taken.
                let s = exec_exception(EXC_FPE);
                return self.handle_exception(s);
            }
            self.core.fpu_fcsr |= FCSR_FV; // untrapped: set sticky Flag V too
        }
        let cond = self.fpu_compare_d(fs_val, ft_val, funct_val);
        let cc = (d.raw >> 8) & 0x7;
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
        self.fpu_cvt_int_to_float(d.rd as u32, d.sa as u32, false, false)
    }
    fn exec_fcvt_d_w(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        self.fpu_cvt_int_to_float(d.rd as u32, d.sa as u32, false, true)
    }

    // ===== COP1 L-format (Long fixed-point → float, MIPS III) =====

    fn exec_fcvt_s_l(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        self.fpu_cvt_int_to_float(d.rd as u32, d.sa as u32, true, false)
    }
    fn exec_fcvt_d_l(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        self.fpu_cvt_int_to_float(d.rd as u32, d.sa as u32, true, true)
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
    // MADD/MSUB/NMADD/NMSUB.S/D: `fd = ±(fs*ft ± fr)`, *not* fused. MIPS IV
    // rounds the product to the format and then adds, rounding again; the
    // R10000 does it as one pass through its multiplier and a second through
    // its adder (User's Manual, ch. 1). A fused multiply-add (`mul_add`)
    // rounds once and differs in the last bit on many inputs, which compilers
    // that emit MADD for `a*b + c` (MIPSpro, LLVM) do not expect. Rust never
    // fuses a separate `*` and `+`. (MIPS32r6's MADDF is the fused one.)
    fn exec_madd_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fr_bits = (self.fpr_read_w)(&self.core, d.rs as u32);
        let ft_bits = (self.fpr_read_w)(&self.core, d.rt as u32);
        let fs_bits = (self.fpr_read_w)(&self.core, d.rd as u32);
        let (fr_val, ft_val, fs_val) = (f32::from_bits(fr_bits), f32::from_bits(ft_bits), f32::from_bits(fs_bits));
        let fd_reg = d.sa as u32;
        let flags = Self::fpu_arith_flags_snan_only3_s(fr_bits, ft_bits, fs_bits);
        let write_w = self.fpr_write_w;
        self.fpu_update_fcsr(flags, |exec| write_w(&mut exec.core, fd_reg, (fs_val * ft_val + fr_val).to_bits()))
    }
    fn exec_madd_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let read_d = self.fpr_read_d; let write_d = self.fpr_write_d;
        let fr_val = read_d(&self.core, d.rs as u32);
        let ft_val = read_d(&self.core, d.rt as u32);
        let fs_val = read_d(&self.core, d.rd as u32);
        let fd_reg = d.sa as u32;
        let flags = Self::fpu_arith_flags_snan_only3_d(fr_val.to_bits(), ft_val.to_bits(), fs_val.to_bits());
        self.fpu_update_fcsr(flags, |exec| write_d(&mut exec.core, fd_reg, fs_val * ft_val + fr_val))
    }
    fn exec_msub_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fr_bits = (self.fpr_read_w)(&self.core, d.rs as u32);
        let ft_bits = (self.fpr_read_w)(&self.core, d.rt as u32);
        let fs_bits = (self.fpr_read_w)(&self.core, d.rd as u32);
        let (fr_val, ft_val, fs_val) = (f32::from_bits(fr_bits), f32::from_bits(ft_bits), f32::from_bits(fs_bits));
        let fd_reg = d.sa as u32;
        let flags = Self::fpu_arith_flags_snan_only3_s(fr_bits, ft_bits, fs_bits);
        let write_w = self.fpr_write_w;
        self.fpu_update_fcsr(flags, |exec| write_w(&mut exec.core, fd_reg, (fs_val * ft_val - fr_val).to_bits()))
    }
    fn exec_msub_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let read_d = self.fpr_read_d; let write_d = self.fpr_write_d;
        let fr_val = read_d(&self.core, d.rs as u32);
        let ft_val = read_d(&self.core, d.rt as u32);
        let fs_val = read_d(&self.core, d.rd as u32);
        let fd_reg = d.sa as u32;
        let flags = Self::fpu_arith_flags_snan_only3_d(fr_val.to_bits(), ft_val.to_bits(), fs_val.to_bits());
        self.fpu_update_fcsr(flags, |exec| write_d(&mut exec.core, fd_reg, fs_val * ft_val - fr_val))
    }
    fn exec_nmadd_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fr_bits = (self.fpr_read_w)(&self.core, d.rs as u32);
        let ft_bits = (self.fpr_read_w)(&self.core, d.rt as u32);
        let fs_bits = (self.fpr_read_w)(&self.core, d.rd as u32);
        let (fr_val, ft_val, fs_val) = (f32::from_bits(fr_bits), f32::from_bits(ft_bits), f32::from_bits(fs_bits));
        let fd_reg = d.sa as u32;
        let flags = Self::fpu_arith_flags_snan_only3_s(fr_bits, ft_bits, fs_bits);
        let write_w = self.fpr_write_w;
        self.fpu_update_fcsr(flags, |exec| write_w(&mut exec.core, fd_reg, fneg_bits_s(fs_val * ft_val + fr_val)))
    }
    fn exec_nmadd_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let read_d = self.fpr_read_d; let write_d = self.fpr_write_d;
        let fr_val = read_d(&self.core, d.rs as u32);
        let ft_val = read_d(&self.core, d.rt as u32);
        let fs_val = read_d(&self.core, d.rd as u32);
        let fd_reg = d.sa as u32;
        let flags = Self::fpu_arith_flags_snan_only3_d(fr_val.to_bits(), ft_val.to_bits(), fs_val.to_bits());
        self.fpu_update_fcsr(flags, |exec| write_d(&mut exec.core, fd_reg, fneg_d_exact(fs_val * ft_val + fr_val)))
    }
    fn exec_nmsub_s(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let fr_bits = (self.fpr_read_w)(&self.core, d.rs as u32);
        let ft_bits = (self.fpr_read_w)(&self.core, d.rt as u32);
        let fs_bits = (self.fpr_read_w)(&self.core, d.rd as u32);
        let (fr_val, ft_val, fs_val) = (f32::from_bits(fr_bits), f32::from_bits(ft_bits), f32::from_bits(fs_bits));
        let fd_reg = d.sa as u32;
        let flags = Self::fpu_arith_flags_snan_only3_s(fr_bits, ft_bits, fs_bits);
        let write_w = self.fpr_write_w;
        self.fpu_update_fcsr(flags, |exec| write_w(&mut exec.core, fd_reg, fneg_bits_s(fs_val * ft_val - fr_val)))
    }
    fn exec_nmsub_d(&mut self, d: &DecodedInstr) -> ExecStatus {
        if (self.core.cp0_status & STATUS_CU1) == 0 { return self.cpu_unusable(1); }
        let read_d = self.fpr_read_d; let write_d = self.fpr_write_d;
        let fr_val = read_d(&self.core, d.rs as u32);
        let ft_val = read_d(&self.core, d.rt as u32);
        let fs_val = read_d(&self.core, d.rd as u32);
        let flags = Self::fpu_arith_flags_snan_only3_d(fr_val.to_bits(), ft_val.to_bits(), fs_val.to_bits());
        let fd_reg = d.sa as u32;
        self.fpu_update_fcsr(flags, |exec| write_d(&mut exec.core, fd_reg, fneg_d_exact(fs_val * ft_val - fr_val)))
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
        // into the wrong page's entry table. `resync_privilege_state` covers
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
        self.resync_privilege_state();
        // The restored snapshot brings its own TLB contents, so every cached
        // nutlb translation describes the *previous* TLB. `resync_privilege_state`
        // above routes through `nanotlb_invalidate()`, which retires them.
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

    /// `step_jit`'s inline-compile path, factored out so the
    /// gate's own control flow (probe hit / decide to compile / pick inline
    /// vs threaded) isn't buried under this compile's own bookkeeping
    /// (arena-growth check, taking/returning the shared `Codegen`, running
    /// `handle_request`, retrying on flush/OOM, status-bar feedback). Called
    /// only when `inline_compile` is true; always ends in a `return` back to
    /// the caller, one way or another — never falls through.
    #[cfg(feature = "jitv2")]
    /// Synchronous compile-and-run for `jitv2_inline_compile = true`.
    /// Full rationale: clanker/jitv2/jit-v2-design.md §13.4.
    fn jitv2_compile_inline(&mut self, req: crate::cpu::jitv2::CompileRequest, trigger: bool) -> ExecStatus {
        let page = unsafe { &mut *self.pcp };
        // Check arena growth BEFORE compiling — flush_from_cpu_thread
        // clears self.pcp/page, which must not happen after this call's
        // own compile just published into that same page.
        let over_threshold = self.jitv2.lock().codegen.lock().as_ref()
            .is_some_and(|c| c.packing_stats().1 > crate::cpu::jitv2::CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES);
        if over_threshold {
            let phys_page_base = page.pfn * crate::cpu::jitv2::PAGE_SIZE;
            unsafe { self.jitv2.lock().flush_from_cpu_thread(self.sysad.clone()); }
            self.clear_pcp();
            self.jitv2_track_pcp(phys_page_base as u64);
            return EXEC_RETRY;
        }
        let mut codegen = self.jitv2.lock().codegen.lock().take();
        #[cfg(feature = "jitv2_lockstep")]
        assert!(codegen.is_some(),
            "jitv2_lockstep: inline compile found no Codegen (async compile queue stole it?) — \
             verification would silently be OFF. The queue must not run under lockstep.");
        let mut ran_out_of_memory = false;
        if let Some(codegen) = codegen.as_mut() {
            // Stamp the L1-D geometry the inline load/store fast path needs.
            // Done on every take rather than once at startup so a cache
            // reconfigure can't leave compiled code indexing the old shape;
            // it is a plain struct copy on a path that is already compiling.
            // Stamp the L1-D geometry the inline load/store path needs. No
            // helper build here — see the note in Jitv2::worker_loop.
            codegen.dc_geometry = self.cache.jit_dc_geometry();
            #[cfg(feature = "developer")]
            {
                let stats = self.jitv2.lock().stats.clone();
                ran_out_of_memory = crate::cpu::jitv2::comp::handle_request(&req, &self.sysad, &mut self.jitv2_inline_analyzer, codegen, &stats);
            }
            #[cfg(not(feature = "developer"))]
            { ran_out_of_memory = crate::cpu::jitv2::comp::handle_request(&req, &self.sysad, &mut self.jitv2_inline_analyzer, codegen); }
        }
        *self.jitv2.lock().codegen.lock() = codegen;
        {
            let reserved_bytes = self.jitv2.lock().codegen.lock().as_ref()
                .map_or(0, |c| c.packing_stats().1);
            crate::cpu::jit_feedback::JIT_FEEDBACK.set_arena_fill(reserved_bytes, crate::cpu::jitv2::CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES);
        }
        if ran_out_of_memory {
            let page = unsafe { &mut *self.pcp };
            let phys_page_base = page.pfn * crate::cpu::jitv2::PAGE_SIZE;
            unsafe { self.jitv2.lock().flush_from_cpu_thread(self.sysad.clone()); }
            self.clear_pcp();
            self.jitv2_track_pcp(phys_page_base as u64);
            return EXEC_RETRY;
        }
        // Re-check is_runnable: handle_request can decline this exact
        // offset even on a compile that published elsewhere.
        let page = unsafe { &mut *self.pcp };
        let entry_offset = ((self.core.pc & 0xFFF) >> 2) as usize;
        if page.is_runnable(entry_offset) {
            let func = page.func();
            debug_assert!(!func.is_null(), "valid bit set with null func");
            #[cfg(feature = "developer")]
            { page.call_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed); }
            let jit_fn: crate::cpu::jitv2::JitFn = unsafe { std::mem::transmute(func) };
            let status = unsafe { jit_fn(&mut self.core as *mut MipsCore) };
            if status != EXEC_FALLBACK {
                status
            } else {
                self.step_int()
            }
        } else {
            self.step_int()
        }
    }

    /// Pure-interpreter dispatch for an already-decoded instruction — no
    /// jitv2 dispatch gate, no compile-request scheduling, no JIT entry at
    /// all. This is `exec_decoded`'s old pre-jitv2 shape (see commit
    /// `8e2fee7`, the last revision before jitv2 landed): just the
    /// `instr_stats` hook and a straight call through the decoded handler.
    /// `step_int()` calls this so the pure-interpreter path never touches
    /// jitv2 state, even when the `jitv2` feature is compiled in (e.g. a
    /// debugger single-step must not itself trigger/observe a compile).
    #[inline(always)]
    pub fn exec_decoded_int(&mut self, d: &DecodedInstr) -> ExecStatus {
        #[cfg(feature = "instr_stats")]
        self.instr_stats.record(d.op, d.rs, d.rt, d.funct, d.raw);

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
        // reference run is anchored.
        //
        // `pc` is ALWAYS this instruction's own address (`emit_word_addr(word)`),
        // authoritative regardless of how the word was reached — set it
        // unconditionally. An earlier version trusted live `core.pc` for the
        // `bd == LOCKSTEP_BD_LIVE` case on the assumption it "already equals the
        // instruction's own address": true when the word was reached by
        // interpreter dispatch or a BC1 fallback, but NOT when a JIT branch/jump
        // took its *in-region* edge into this word's block — `emit_target_edge`
        // jumps straight to the block without writing `core.pc`, so it is left
        // stale at whatever the last materialization set (e.g. the branch's own
        // address, via `emit_slot_semantics`' `saved_pc` restore). That stale pc
        // then anchored the interpreter reference one word early, a spurious
        // `pc` divergence at every `is_branch_fallback_successor` word that is
        // also a JIT branch target.
        //
        // `in_delay_slot`/`delay_slot_target`, on the other hand, MUST stay
        // trusted-live for `LOCKSTEP_BD_LIVE`: a fallback-successor can arrive
        // mid-delay-slot (BC1 fallback armed a pending transfer), and a
        // compile-time `bd` would clobber that inherited `in_delay_slot=true`.
        self.core.pc = pc;
        if bd != LOCKSTEP_BD_LIVE {
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

        // Record this step's entry state for `j2 lstate`. Done here (before the
        // interpreter reference runs) so it lands whether or not the step
        // cleanly retires or diverges — the last entry is then always the
        // state that fed the instruction the monitor breaks on.
        if self.ls_history.len() == LS_HISTORY_CAP {
            self.ls_history.pop_front();
        }
        self.ls_history.push_back(LockstepHistEntry { pc, raw, bd, before });

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
            // exception path (production mem hooks under lockstep still report
            // faults via their returned status) produces the real result the
            // region uses.
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
                eprintln!("  gpr[{:2}] {:<5} jit={:#018x}  interp={:#018x}", r, crate::cpu::mips_dis::reg_name(r as u32), jit.gpr[r], expected.gpr[r]);
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
    fn lockstep_jit_read<const SIZE: usize>(&mut self, va: u64) -> JitReadResult {
        let Some(captured) = self.core.lockstep_mem else {
            let result = match self.read_data::<SIZE>(va) {
                Ok(v) => JitReadResult { val: v, status: EXEC_COMPLETE },
                Err(status) => JitReadResult { val: 0, status },
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
            return JitReadResult { val: 0, status: translate_result.status };
        }
        let phys = translate_result.phys;
        if va != captured.addr || phys != captured.phys {
            eprintln!("\n=== jitv2_lockstep DIVERGENCE at pc={:#018x}: read address mismatch ===", self.core.pc);
            eprintln!("  va   jit={:#018x}  interp={:#018x}", va, captured.addr);
            eprintln!("  phys jit={:#018x}  interp={:#018x}", phys, captured.phys);
            eprintln!("=== breaking into monitor; use `dt`, `r`, `d pc` to inspect (JIT's wrong post-state is live) ===\n");
            self.core.lockstep_mem = None;
            return JitReadResult { val: 0, status: EXEC_BREAKPOINT };
        }
        let mask: u64 = if SIZE == 8 { u64::MAX } else { (1u64 << (SIZE * 8)) - 1 };
        JitReadResult { val: captured.value & mask, status: EXEC_COMPLETE }
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
            return translate_result.status;
        }
        let phys = translate_result.phys;
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
            return EXEC_BREAKPOINT;
        }
        EXEC_COMPLETE
    }
}

/// How many recent lockstep steps `MipsExecutor::ls_history` retains for
/// `j2 lstate`. Small — this is a "what led up to the divergence" window, not
/// a full trace; each entry is ~640 bytes (two register files).
#[cfg(feature = "jitv2_lockstep")]
pub(crate) const LS_HISTORY_CAP: usize = 32;

/// One entry in `MipsExecutor::ls_history`: a lockstep-bracketed instruction
/// and the machine state that entered it (the previous step's final state).
#[cfg(feature = "jitv2_lockstep")]
#[derive(Clone, Copy)]
pub(crate) struct LockstepHistEntry {
    pub(crate) pc: u64,
    pub(crate) raw: u32,
    /// `bd` as passed to `lockstep_step` (0/1, or `LOCKSTEP_BD_LIVE` for an
    /// entry word / delay slot whose slot state was inherited, not materialized).
    pub(crate) bd: u32,
    pub(crate) before: LockstepSnapshot,
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
    fn restore_into<T: Tlb, C: CpuModel>(&self, exec: &mut MipsExecutor<T, C>, delay_slot_target: u64) {
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
        // Direct `fpu_fcsr` write bypasses `write_fpu_control` / CTC1, the
        // only path that normally reprograms the host FPU rounding-control
        // field to match FCSR.RM (mirrors the snapshot-restore and
        // thread-spawn `set_fpu_mode` calls elsewhere in this file). Without
        // this, an interpreter reference run that reprogrammed the host mode
        // (a CTC1 in the bracketed stream) leaves it ahead of the restored
        // FCSR.RM, and the JIT's re-run — whose `add.d`/`sub.d`/`mul.d` lower
        // to plain host FP ops that read the live rounding mode, no per-op
        // resync — then rounds under the wrong mode. The `(x + 2^52) - 2^52`
        // round-to-integer idiom (software %/÷ by constants) is acutely
        // sensitive to this: one ULP of rounding difference flips the integer.
        // The FR-pointer analogue of this is documented in
        // `clanker/jitv2/codegen-gotchas.md`.
        crate::platform::set_fpu_mode((self.fcsr & 0x3) as u8);
        // That FR-pointer analogue, closed: the direct `cp0_status` write above equally
        // bypasses the path that re-derives `translate_fn` and the fpr accessors, so a
        // restore across a privilege or FR boundary would otherwise resume on pointers
        // describing the pre-restore mode.
        exec.resync_privilege_state();
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
/// LW/LD/SB/SH/SW/SD — see clanker/perf/ notes on the addr-calc+load/store
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
pub fn decode_into<T: Tlb, C: CpuModel>(ins: &mut DecodedInstr) {
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
            // MOVCI is MIPS IV (MOVF/MOVT); an R4400 (MIPS III) must raise RI.
            FUNCT_MOVCI => if C::MIPS4 { MipsExecutor::<T,C>::exec_movci } else { MipsExecutor::<T,C>::exec_reserved },
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
            // MOVZ/MOVN are MIPS IV; an R4400 (MIPS III) must raise RI.
            FUNCT_MOVZ => if C::MIPS4 { MipsExecutor::<T,C>::exec_movz } else { MipsExecutor::<T,C>::exec_reserved },
            FUNCT_MOVN => if C::MIPS4 { MipsExecutor::<T,C>::exec_movn } else { MipsExecutor::<T,C>::exec_reserved },
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
                // MOVF.fmt/MOVT.fmt/MOVZ.fmt/MOVN.fmt/RECIP.fmt/RSQRT.fmt are
                // MIPS IV; an R4400 (MIPS III) must raise RI for all five.
                FUNCT_FMOVCF => if C::MIPS4 { MipsExecutor::<T,C>::exec_fmovcf_s } else { MipsExecutor::<T,C>::exec_reserved },
                FUNCT_FMOVZ => if C::MIPS4 { MipsExecutor::<T,C>::exec_fmovz_s } else { MipsExecutor::<T,C>::exec_reserved },
                FUNCT_FMOVN => if C::MIPS4 { MipsExecutor::<T,C>::exec_fmovn_s } else { MipsExecutor::<T,C>::exec_reserved },
                FUNCT_FRECIP => if C::MIPS4 { MipsExecutor::<T,C>::exec_frecip_s } else { MipsExecutor::<T,C>::exec_reserved },
                FUNCT_FRSQRT => if C::MIPS4 { MipsExecutor::<T,C>::exec_frsqrt_s } else { MipsExecutor::<T,C>::exec_reserved },
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
                // MOVF.fmt/MOVT.fmt/MOVZ.fmt/MOVN.fmt/RECIP.fmt/RSQRT.fmt are
                // MIPS IV; an R4400 (MIPS III) must raise RI for all five.
                FUNCT_FMOVCF => if C::MIPS4 { MipsExecutor::<T,C>::exec_fmovcf_d } else { MipsExecutor::<T,C>::exec_reserved },
                FUNCT_FMOVZ => if C::MIPS4 { MipsExecutor::<T,C>::exec_fmovz_d } else { MipsExecutor::<T,C>::exec_reserved },
                FUNCT_FMOVN => if C::MIPS4 { MipsExecutor::<T,C>::exec_fmovn_d } else { MipsExecutor::<T,C>::exec_reserved },
                FUNCT_FRECIP => if C::MIPS4 { MipsExecutor::<T,C>::exec_frecip_d } else { MipsExecutor::<T,C>::exec_reserved },
                FUNCT_FRSQRT => if C::MIPS4 { MipsExecutor::<T,C>::exec_frsqrt_d } else { MipsExecutor::<T,C>::exec_reserved },
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
        // The entire COP1X opcode (LWXC1/LDXC1/SWXC1/SDXC1/PREFX and the
        // MADD/MSUB/NMADD/NMSUB family) is a MIPS IV addition — no MIPS III
        // encoding uses it, so an R4400 must raise RI for the whole opcode.
        OP_COP1X => if C::MIPS4 {
                match funct as u32 {
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
            }
        } else { MipsExecutor::<T,C>::exec_reserved },
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
        // PREF is MIPS IV; an R4400 (MIPS III) must raise RI.
        OP_PREF => if C::MIPS4 { MipsExecutor::<T,C>::exec_pref } else { MipsExecutor::<T,C>::exec_reserved },
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


/// MipsCpu wrapper for threaded execution and monitor control
/// Why the CPU last stopped on its own (a breakpoint, an exception catch, a
/// run/step finishing), readable without the executor lock so `status` and
/// `wait` work while the CPU runs. `seq` counts stops.
#[derive(Clone, Default)]
pub struct StopNote {
    pub seq: u64,
    pub pc: u64,
    pub reason: String,
}

impl StopNote {
    fn record(&mut self, pc: u64, reason: String) {
        self.seq += 1;
        self.pc = pc;
        self.reason = reason;
    }
}

pub struct MipsCpu<T: Tlb, C: CpuModel> {
    executor: Arc<Mutex<MipsExecutor<T, C>>>,
    running: Arc<AtomicBool>,
    stop_note: Arc<Mutex<StopNote>>,
    thread: Mutex<Option<thread::JoinHandle<()>>>,
    /// Pointer into the executor's `MipsCore.hot.cycles` — see
    /// `CyclesPtr`/`Hot::cycles`'s doc comments. Same process-lifetime
    /// validity argument as `interrupts_ptr` right below.
    cycles_ptr: crate::cpu::mips_core::CyclesPtr,
    /// Raw pointer into the executor's `MipsCore.interrupts` (an inline
    /// field, not `Arc<AtomicU64>` — see that field's doc comment). Valid
    /// for the process lifetime: `executor` below is the only owner of the
    /// `MipsCore` this points into, and it's never dropped before shutdown.
    interrupts_ptr: *const AtomicU64,
    /// ppmem: pointer to the executor's inline `ppmem_bitmap` field, handed to
    /// `PpMemSpace` so remaps publish straight into the CPU. Same
    /// process-lifetime validity argument as `interrupts_ptr` above.
    ppmem_bitmap_ptr: *mut u64,
    /// Where another thread holds this CPU still.
    park: Arc<crate::cpu::smp::ParkSlot>,
    pub fasttick_count: Arc<AtomicU64>,
    debug: Arc<AtomicBool>,
    exception_mask: Arc<AtomicU32>,
    trace_file: Arc<Mutex<Option<std::io::BufWriter<std::fs::File>>>>,
    #[cfg(feature = "idle-pause")]
    idle_profile_on: Arc<AtomicBool>,
    #[cfg(feature = "idle-pause")]
    idle_profile_reset: Arc<AtomicBool>,
}

/// See [`MipsCpu::pause`].
struct Paused<'a, T: Tlb + Send + 'static, C: CpuModel + Send + 'static> {
    cpu: &'a MipsCpu<T, C>,
    resume: bool,
}

impl<T: Tlb + Send + 'static, C: CpuModel + Send + 'static> Drop for Paused<'_, T, C> {
    fn drop(&mut self) {
        if self.resume {
            Device::start(self.cpu);
        }
    }
}

// Safety: interrupts_ptr points into the MipsCore owned by `executor`
// (Arc<Mutex<MipsExecutor>>), which outlives every MipsCpu clone/thread that
// might read this pointer — the whole struct is already Send/Sync via its
// other Arc fields; this one raw pointer needs the same guarantee spelled
// out explicitly since raw pointers don't get it automatically.
unsafe impl<T: Tlb, C: CpuModel> Send for MipsCpu<T, C> {}
unsafe impl<T: Tlb, C: CpuModel> Sync for MipsCpu<T, C> {}

impl<T: Tlb + Send + 'static, C: CpuModel + Send + 'static> MipsCpu<T, C> {
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
        let ppmem_bitmap_ptr = executor_arc.lock().ppmem_bitmap_ptr();
        #[cfg(feature = "jitv2")]
        executor_arc.lock().install_jit_hooks();

        Self {
            executor: executor_arc,
            running: Arc::new(AtomicBool::new(false)),
            stop_note: Arc::new(Mutex::new(StopNote::default())),
            thread: Mutex::new(None),
            cycles_ptr,
            interrupts_ptr,
            ppmem_bitmap_ptr,
            park: Arc::new(crate::cpu::smp::ParkSlot::new(interrupts_ptr)),
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
    pub fn core_ptr(&self) -> *const crate::cpu::mips_core::MipsCore {
        &self.executor.lock().core as *const crate::cpu::mips_core::MipsCore
    }

    pub fn cycles_ptr(&self) -> crate::cpu::mips_core::CyclesPtr {
        self.cycles_ptr
    }

    /// Raw pointer to the executor's `MipsCore.interrupts` word, for wiring
    /// into devices on other threads that set/clear interrupt bits (e.g.
    /// `Ioc::set_interrupts`). Fixed for the life of this `MipsCpu` — set
    /// once in `new()` after the executor reached its final address.
    pub fn interrupts_ptr(&self) -> *const AtomicU64 {
        self.interrupts_ptr
    }

    /// ppmem: pointer to the executor's inline `ppmem_bitmap` field, for
    /// `PpMemSpace::set_bitmap_sink`. Fixed for the life of this `MipsCpu` —
    /// taken once in `new()` after the executor reached its final address.
    pub fn ppmem_bitmap_ptr(&self) -> *mut u64 {
        self.ppmem_bitmap_ptr
    }

    /// tcache: see `MipsExecutor::set_tcache_window`.
    ///
    /// # Safety
    /// Same contract as `CpuCache::set_tcache_window`.
    pub unsafe fn set_tcache_window(&self, base: *mut u8) {
        unsafe { self.executor.lock().set_tcache_window(base) };
    }

    /// tcache: pointer to the cache's inline bitmap field.
    pub fn tcache_bitmap_ptr(&self) -> *mut u64 {
        self.executor.lock().tcache_bitmap_ptr()
    }

    /// tcache + jitv2: see `MipsExecutor::set_tcache_gen_window`.
    ///
    /// # Safety
    /// Same contract as that method.
    #[cfg(feature = "jitv2")]
    pub unsafe fn set_tcache_gen_window(&self, gen_base: *mut AtomicU64) {
        unsafe { self.executor.lock().set_tcache_gen_window(gen_base) };
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
    pub fn jitv2(&self) -> Arc<Mutex<crate::cpu::jitv2::Jitv2>> {
        self.executor.lock().jitv2.clone()
    }


    pub fn register_locks(&self) {
        use crate::locks::register_lock_fn;
        let ex = self.executor.clone();
        register_lock_fn("cpu::executor", move || ex.is_locked());
    }

    fn try_lock_executor(&self) -> Result<parking_lot::MutexGuard<'_, MipsExecutor<T, C>>, String> {
        self.executor.try_lock().ok_or_else(|| "CPU thread holds the executor lock; try 'cpu stop' first".to_string())
    }


    /// Stop the CPU thread and leave the core in the stopped state (Count
    /// latched, jitv2 `pcp` cleared), without the stats `Device::stop` prints.
    fn halt(&self) {
        self.running.store(false, Ordering::SeqCst);
        if let Some(handle) = self.thread.lock().take() {
            let _ = handle.join();
        }
        let mut exec = self.executor.lock();
        // Latch the virtual CP0 Count and silence the compare timer while
        // stopped — debugger stepping must not see Count advance or IP7
        // fire underneath it (mips_core.rs on_cpu_stop's doc comment).
        exec.core.on_cpu_stop();
        // `pcp == null` is an invariant of "CPU is stopped"; see the note in
        // `Device::stop`.
        #[cfg(feature = "jitv2")]
        exec.nanotlb_invalidate();
        #[cfg(not(feature = "jitv2"))]
        let _ = &mut exec;
    }

    /// Hold the core still for a monitor command: a running CPU is stopped
    /// now and restarted when the guard drops; a stopped one is left alone.
    /// Every command that reads or changes core state goes through this, so
    /// none of them can block on, or fail because of, a running CPU.
    fn pause(&self) -> Paused<'_, T, C> {
        let resume = self.is_running();
        if resume {
            self.halt();
        }
        Paused { cpu: self, resume }
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
            let _status = exec.step_int();
            executed += 1;
            // Don't break on exceptions — they're part of normal CPU
            // operation and a deterministic run should re-enter and continue.
        }
        Ok(executed)
    }

    /// Step exactly one instruction and return how many architectural
    /// instructions it actually retired, per `core.hot.cycles`' delta.
    /// Ordinarily 1 (the interpreter's dispatch always retires exactly one
    /// instruction), but a real JIT-compiled unit can retire 2+ in a single
    /// `step_jit()` call — a branch/jump's compiled unit always inlines its
    /// delay slot (§6.1.4), so one call there covers both. Callers doing
    /// their own instruction-by-instruction accounting against a *different*
    /// engine's per-instruction reference trace (`validate::validate_jit_determinism`,
    /// the engine behind the monitor's `jitcheck <n>` command) need this,
    /// not `step_n_inline`'s step-call count, to stay correctly aligned when
    /// JIT dispatch is involved.
    ///
    /// Picks `step_int()` vs `step_jit()` itself, based on
    /// `jitv2_dispatch_enabled`: `validate_jit_determinism`'s reference pass
    /// sets that `false` and wants genuinely jitv2-free execution — calling
    /// `step_jit()` there would rely on `step_jit`'s own
    /// dispatch gate to decline every dispatch via that same field, which
    /// used to work but no longer does now that the gate isn't re-checked
    /// per-call the way `step()` once was (see `step_jit`'s own doc comment
    /// on why the Rust-side preamble/gate only runs on `EXEC_FALLBACK`
    /// now) — so this reads the field directly instead of leaving the
    /// distinction to a callee that no longer makes it.
    #[cfg(feature = "developer")]
    pub fn step_one_inline_counting_instructions(&self) -> Result<usize, String> {
        let mut exec = self.try_lock_executor()?;
        let cycles_before = exec.core.hot.cycles;
        #[cfg(feature = "jitv2")]
        let status = if exec.jitv2_dispatch_enabled { exec.step_jit() } else { exec.step_int() };
        #[cfg(not(feature = "jitv2"))]
        let status = exec.step_int();
        exec.last_step_status = status;
        let retired = exec.core.hot.cycles.wrapping_sub(cycles_before) as usize;
        Ok(retired.max(1)) // defensive: never 0, which would stall any caller looping on this
    }

    /// Runtime kill switch for `exec_decoded`'s real JIT dispatch gate — see
    /// `MipsExecutor::jitv2_dispatch_enabled`'s doc comment. Returns the
    /// previous value so callers can restore it afterward. Not `developer`-
    /// gated (unlike the rest of the `jitcheck`/monitor-command family this
    /// originally shipped alongside): `start()`'s dispatch loop reads
    /// `jitv2_dispatch_enabled` under `lightning` too now (see its own doc
    /// comment on why that's free), and `IRIS_JIT_DISPATCH=0` (checked once
    /// in `MipsCpu::start`, before the CPU thread's loop begins — `lightning`
    /// has no monitor console to toggle this live) needs a way to actually
    /// set it under `lightning`, where `developer` can never be on.
    #[cfg(feature = "jitv2")]
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
    /// the fixed count_hz).
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
        exec.resync_privilege_state();
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


impl<T: Tlb + Send + 'static, C: CpuModel + Send + 'static> Device for MipsCpu<T, C> {

    fn step(&self, cycles: u64) {
        let mut exec = self.executor.lock();
        for _ in 0..cycles {
            exec.step_int();
        }
    }

    fn stop(&self) {
        self.halt();
        #[cfg(feature = "tlbstats")]
        self.executor.lock().tlb.stats_print();
        #[cfg(feature = "jitstats")]
        self.executor.lock().jit_mem_stats.print();
        #[cfg(feature = "jitv2")]
        {
            use std::sync::atomic::Ordering;
            let e = crate::cpu::jitv2::codegen::INLINE_MEM_EMITTED.load(Ordering::Relaxed);
            let d = crate::cpu::jitv2::codegen::INLINE_MEM_DECLINED.load(Ordering::Relaxed);
            if e + d > 0 {
                eprintln!("jitv2 inline memory: emitted={e} declined={d}");
            }
        }
        #[cfg(all(feature = "jitv2", feature = "jitv2_smc_check"))]
        {
            // Always printed, including the zero case: "0" is a real result
            // here only if you also know the check was live, and printing it
            // unconditionally is what distinguishes "no SMC observed" from
            // "the feature wasn't compiled in". Note the inline-store blind
            // spot — pair a zero with `jitv2 inline memory: emitted=0`
            // before concluding anything.
            let hits = SMC_HITS.load(std::sync::atomic::Ordering::Relaxed);
            eprintln!("jitv2 SMC check: {hits} write(s) landed on the page being executed");
        }
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
            let exec = self.executor.lock();
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
            // (Done in `halt()`, which every stop path goes through.)
            //
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
    }

    fn start(&self) {
        if self.is_running() { return; }

        // IRIS_JIT_DISPATCH=0: force jitv2 dispatch off for this run, even
        // under `lightning` where there's no monitor console (`j2 dispatch
        // off`) or `developer` build to reach `set_jitv2_dispatch_enabled`
        // through otherwise. Checked once per `start()` (idempotent — cheap,
        // and `start()`/`stop()` can cycle across a single process run) so a
        // `lightning` build can still be tested with real JIT dispatch
        // disabled, without needing a `developer` rebuild.
        #[cfg(feature = "jitv2")]
        if std::env::var("IRIS_JIT_DISPATCH").as_deref() == Ok("0") {
            let _ = self.set_jitv2_dispatch_enabled(false);
        }

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

        let stop_note = self.stop_note.clone();
        let park = self.park.clone();
        park.set_active(true);
        let cpu_id = self.executor.lock().core.cpu_id;
        let name = if cpu_id == 0 { "MIPS-CPU".to_string() } else { format!("MIPS-CPU{cpu_id}") };
        *self.thread.lock() = Some(thread::Builder::new().name(name).stack_size(16 * 1024 * 1024).spawn(move || {
            crate::cpu::smp::set_current_cpu(cpu_id);
            // `[perf] cpu_core` is one core: the first CPU's.
            if cpu_id == 0 {
                crate::thread_affinity::pin_current(crate::thread_affinity::PerfRole::MipsCpu);
            }
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

            // Idle detection + park state (see clanker/perf/idle-pause-work.md). Compiled
            // in only with the `idle-pause` feature (off by default; opt in with
            // --features idle-pause). When compiled in, set IRIS_NO_IDLE to keep
            // spinning the host CPU at runtime (for benchmarking/debug).
            //
            // The detector samples the PC once per 32 sample calls (about 320
            // instructions in the batch loop) and parks only when the execution
            // stayed inside one long, narrow, kernel code window with
            // interrupts enabled and nothing pending. Short poll/delay loops
            // (2-3 instructions) are rejected by the window's minimum span and
            // distinct-PC count; user-mode code is never parked. See
            // src/cpu/idle_park.rs.
            #[cfg(feature = "idle-pause")]
            let mut idle_state = crate::cpu::idle_park::IdleParkState::default();

            while running.load(Ordering::Relaxed) {
                // Another thread is holding this CPU: stop here, between
                // instructions, outside compiled code, with the executor free.
                if guard.core.hot.interrupts.load(Ordering::Relaxed) & crate::cpu::smp::HOST_PARK_BIT != 0 {
                    #[cfg(feature = "jitv2")]
                    guard.nanotlb_invalidate();
                    drop(guard);
                    park.park(&running);
                    guard = executor.lock();
                    continue;
                }
                // Read the runtime jitv2 on/off switch once per 1000-instruction
                // batch, outside the inner loop, and run the WHOLE batch through
                // one of two loops picking step_int()/step_jit() (not a
                // per-instruction if/else inside a single loop, which would
                // re-branch on every step) — `j2 dispatch off`
                // (`jitv2_dispatch_enabled = false`) is supposed to mean
                // genuinely interpreter-only, not "keep paying for the JIT gate
                // check and just decline it every time". Honored under
                // `lightning` too now (doesn't cost anything: read once per
                // batch, same as every other build) — useful for testing with
                // JIT dispatch off in an otherwise-lightning build. Toggling
                // takes effect within at most 1000 instructions, same
                // granularity this loop already gives breakpoint checks/
                // idle-park/perf sampling.
                #[cfg(feature = "jitv2")]
                let jit_on = guard.jitv2_dispatch_enabled;
                #[cfg(not(feature = "jitv2"))]
                let jit_on = false;

                #[cfg(feature = "lightning")]
                {
                    // No breakpoints possible in lightning mode; 10x manual
                    // unroll avoids the per-step match and helps LLVM see a
                    // larger block.
                    if jit_on {
                        #[cfg(feature = "jitv2")]
                        for _ in 0..1000 {
                            guard.step_jit(); guard.step_jit(); guard.step_jit(); guard.step_jit(); guard.step_jit();
                            guard.step_jit(); guard.step_jit(); guard.step_jit(); guard.step_jit(); guard.step_jit();
                            #[cfg(feature = "idle-pause")]
                            idle_state.sample(&guard.core);
                        }
                    } else {
                        // is this going to be a problem for netbsd?
                        for _ in 0..1000 {
                            guard.step_int(); guard.step_int(); guard.step_int(); guard.step_int(); guard.step_int();
                            guard.step_int(); guard.step_int(); guard.step_int(); guard.step_int(); guard.step_int();
                            #[cfg(feature = "idle-pause")]
                            idle_state.sample(&guard.core);
                        }
                                                 
                    }
                }
                #[cfg(not(feature = "lightning"))]
                {
                    macro_rules! run_batch {
                        ($step:ident) => {
                            for _ in 0..1000 {
                                let status = guard.$step();
                                #[cfg(feature = "idle-pause")]
                                idle_state.sample(&guard.core);
                                match status {
                                    EXEC_BREAKPOINT => {
                                        running.store(false, Ordering::SeqCst);
                                        // Reported through the stop note (`status`, `wait`),
                                        // not printed: the terminal is a monitor console.
                                        let why = guard.breakpoint_hit_reason();
                                        stop_note.lock().record(guard.core.pc, why);
                                        break;
                                    }
                                    _ => {}
                                }
                            }
                        };
                    }
                    #[cfg(feature = "jitv2")]
                    if jit_on {
                        run_batch!(step_jit);
                    } else {
                        run_batch!(step_int);
                    }
                    #[cfg(not(feature = "jitv2"))]
                    run_batch!(step_int);
                }
                if let Some(mut hook) = guard.batch_hook.take() {
                    hook(&mut guard);
                    guard.batch_hook = Some(hook);
                }
                #[cfg(feature = "idle-pause")]
                if crate::cpu::idle_park::idle_park_enabled() {
                    match idle_state.update(&guard.core) {
                        crate::cpu::idle_park::IdleDecision::Park => {
                            drop(guard);
                            {
                                let mut guard = executor.lock();
                                idle_state.park(&mut guard.core, &running);
                            }
                            guard = executor.lock();
                        }
                        crate::cpu::idle_park::IdleDecision::Pace => {
                            drop(guard);
                            {
                                let mut guard = executor.lock();
                                idle_state.pace(&mut guard.core, &running);
                            }
                            guard = executor.lock();
                        }
                        crate::cpu::idle_park::IdleDecision::None => {}
                    }
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
            drop(guard);
            park.set_active(false);
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
                #[cfg(feature = "idle-pause")]
                crate::cpu::idle_park::wake();
            }
            Signal::Interrupt(line, active) => {
                let mask = 1u64 << (line + 8);
                if active {
                    interrupts.fetch_or(mask, Ordering::SeqCst);
                    #[cfg(feature = "idle-pause")]
                    crate::cpu::idle_park::wake();
                } else {
                    interrupts.fetch_and(!mask, Ordering::SeqCst);
                }
            }
        }
    }


}


// ============================================================================
// Resettable + Saveable for MipsCpu (CPU core + TLB)
// ============================================================================

/// Type-erased CPU handle. The CPU thread runs inside the implementor, so every
/// call here is setup, control or debug — never the instruction hot path.
pub trait CpuDevice: Device + Resettable + Saveable + Send + Sync {
    fn as_device(self: Arc<Self>) -> Arc<dyn Device>;
    fn debug_adapter(self: Arc<Self>) -> Arc<dyn crate::gdb_stub::CpuDebug>;
    /// This CPU's monitor commands; `CpuMonitor` is what the monitor talks to.
    fn monitor_commands(&self) -> Vec<(String, String)>;
    /// Where another thread holds this CPU still.
    fn park_slot(&self) -> Arc<crate::cpu::smp::ParkSlot>;
    /// Use this page pool and compiled code, shared with the other CPUs, whose
    /// flushes `world` coordinates. Before the CPU first runs.
    #[cfg(feature = "jitv2")]
    fn share_jitv2(&self, jit: Arc<Mutex<crate::cpu::jitv2::Jitv2>>, world: Arc<crate::cpu::smp::World>);
    fn monitor_execute(&self, cmd: &str, args: &[&str], writer: Box<dyn Write + Send>) -> Result<(), String>;
    fn cycles_ptr(&self) -> crate::cpu::mips_core::CyclesPtr;
    fn interrupts_ptr(&self) -> *const AtomicU64;
    /// ppmem: the executor's inline `ppmem_bitmap` field, for
    /// `PpMemSpace::set_bitmap_sink`.
    fn ppmem_bitmap_ptr(&self) -> *mut u64;
    /// tcache: hand the cache ppmem's window base + mapped-region bitmap.
    ///
    /// # Safety
    /// Same contract as `CpuCache::set_tcache_window`.
    unsafe fn set_tcache_window(&self, base: *mut u8);
    /// tcache: pointer to the cache's inline mapped-region bitmap.
    fn tcache_bitmap_ptr(&self) -> *mut u64;
    /// tcache + jitv2: hand the cache ppmem's generation-window base.
    ///
    /// # Safety
    /// Same contract as `CpuCache::set_tcache_gen_window`.
    #[cfg(feature = "jitv2")]
    unsafe fn set_tcache_gen_window(&self, gen_base: *mut AtomicU64);
    fn core_ptr(&self) -> *const crate::cpu::mips_core::MipsCore;
    fn register_locks(&self);
    fn load_elf(&self, path: &str) -> Result<String, String>;
    fn load_elf_bytes(&self, bytes: &[u8], name: &str) -> Result<String, String>;
    fn step_n_inline(&self, n: u64) -> Result<u64, String>;
    #[cfg(feature = "developer")]
    fn step_one_inline_counting_instructions(&self) -> Result<usize, String>;
    fn state_digest(&self) -> Result<CpuStateDigest, String>;
    #[cfg(feature = "developer")]
    fn restore_state_digest(&self, d: &CpuStateDigest) -> Result<(), String>;
    #[cfg(feature = "developer")]
    fn fixup_cp0_count(&self, d: &CpuStateDigest) -> Result<(), String>;
    #[cfg(all(feature = "jitv2", feature = "developer"))]
    fn set_jitv2_dispatch_enabled(&self, on: bool) -> Result<bool, String>;
    #[cfg(feature = "developer")]
    fn set_hw_read_fixup_recording(&self, on: bool) -> Result<(), String>;
    #[cfg(feature = "developer")]
    fn set_hw_read_fixup_replay(&self, f: Option<Vec<(u64, u8, u64)>>) -> Result<(), String>;
    /// CPU model name, as the guest and the benchmark report see it.
    fn model_name(&self) -> &'static str;
    /// Cache model: "full" or "shadow" (`CpuModel::CACHE_KIND`).
    fn cache_kind(&self) -> &'static str;
    fn set_cheritest_dump_hook(&self, on: bool);
    fn running_flag(&self) -> Arc<AtomicBool>;
    fn fasttick_count(&self) -> Arc<AtomicU64>;
    fn count_hz_atomic(&self) -> Arc<AtomicU64>;
    #[cfg(feature = "jitv2")]
    fn jitv2(&self) -> Arc<Mutex<crate::cpu::jitv2::Jitv2>>;
}

impl<T: Tlb + Send + 'static, C: CpuModel + Send + 'static> CpuDevice for MipsCpu<T, C> {
    fn as_device(self: Arc<Self>) -> Arc<dyn Device> { self }
    fn debug_adapter(self: Arc<Self>) -> Arc<dyn crate::gdb_stub::CpuDebug> {
        MipsCpuDebugAdapter::new(self)
    }
    fn monitor_commands(&self) -> Vec<(String, String)> { MipsCpu::monitor_commands(self) }
    fn park_slot(&self) -> Arc<crate::cpu::smp::ParkSlot> { self.park.clone() }
    #[cfg(feature = "jitv2")]
    fn share_jitv2(&self, jit: Arc<Mutex<crate::cpu::jitv2::Jitv2>>, world: Arc<crate::cpu::smp::World>) {
        let mut exec = self.executor.lock();
        exec.jitv2_share(jit, world);
    }
    fn monitor_execute(&self, cmd: &str, args: &[&str], writer: Box<dyn Write + Send>) -> Result<(), String> {
        MipsCpu::monitor_execute(self, cmd, args, writer)
    }
    fn cycles_ptr(&self) -> crate::cpu::mips_core::CyclesPtr { MipsCpu::cycles_ptr(self) }
    fn interrupts_ptr(&self) -> *const AtomicU64 { MipsCpu::interrupts_ptr(self) }
    fn ppmem_bitmap_ptr(&self) -> *mut u64 { MipsCpu::ppmem_bitmap_ptr(self) }
    unsafe fn set_tcache_window(&self, base: *mut u8) {
        unsafe { MipsCpu::set_tcache_window(self, base) }
    }
    fn tcache_bitmap_ptr(&self) -> *mut u64 { MipsCpu::tcache_bitmap_ptr(self) }
    #[cfg(feature = "jitv2")]
    unsafe fn set_tcache_gen_window(&self, gen_base: *mut AtomicU64) {
        unsafe { MipsCpu::set_tcache_gen_window(self, gen_base) }
    }
    fn core_ptr(&self) -> *const crate::cpu::mips_core::MipsCore { MipsCpu::core_ptr(self) }
    fn register_locks(&self) { MipsCpu::register_locks(self) }
    fn load_elf(&self, p: &str) -> Result<String, String> { MipsCpu::load_elf(self, p) }
    fn load_elf_bytes(&self, b: &[u8], n: &str) -> Result<String, String> { MipsCpu::load_elf_bytes(self, b, n) }
    fn step_n_inline(&self, n: u64) -> Result<u64, String> { MipsCpu::step_n_inline(self, n) }
    #[cfg(feature = "developer")]
    fn step_one_inline_counting_instructions(&self) -> Result<usize, String> {
        MipsCpu::step_one_inline_counting_instructions(self)
    }
    fn state_digest(&self) -> Result<CpuStateDigest, String> { MipsCpu::state_digest(self) }
    #[cfg(feature = "developer")]
    fn restore_state_digest(&self, d: &CpuStateDigest) -> Result<(), String> { MipsCpu::restore_state_digest(self, d) }
    #[cfg(feature = "developer")]
    fn fixup_cp0_count(&self, d: &CpuStateDigest) -> Result<(), String> { MipsCpu::fixup_cp0_count(self, d) }
    #[cfg(all(feature = "jitv2", feature = "developer"))]
    fn set_jitv2_dispatch_enabled(&self, on: bool) -> Result<bool, String> { MipsCpu::set_jitv2_dispatch_enabled(self, on) }
    #[cfg(feature = "developer")]
    fn set_hw_read_fixup_recording(&self, on: bool) -> Result<(), String> { MipsCpu::set_hw_read_fixup_recording(self, on) }
    #[cfg(feature = "developer")]
    fn set_hw_read_fixup_replay(&self, f: Option<Vec<(u64, u8, u64)>>) -> Result<(), String> {
        MipsCpu::set_hw_read_fixup_replay(self, f)
    }
    fn model_name(&self) -> &'static str { C::NAME }
    fn cache_kind(&self) -> &'static str { C::CACHE_KIND }
    fn set_cheritest_dump_hook(&self, on: bool) { MipsCpu::set_cheritest_dump_hook(self, on) }
    fn running_flag(&self) -> Arc<AtomicBool> { MipsCpu::running_flag(self) }
    fn fasttick_count(&self) -> Arc<AtomicU64> { Arc::clone(&self.fasttick_count) }
    fn count_hz_atomic(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.executor.lock().core.count_hz_atomic)
    }
    #[cfg(feature = "jitv2")]
    fn jitv2(&self) -> Arc<Mutex<crate::cpu::jitv2::Jitv2>> { MipsCpu::jitv2(self) }
}

impl<T: Tlb + Send + 'static, C: CpuModel + Send + 'static> Resettable for MipsCpu<T, C> {
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
        // `reset()` above rewrote cp0_status by direct field write. Re-derive
        // translate_fn/fpr accessors from it, and retire every cached translation:
        // the reset TLB + CP0 state means all of them describe the *previous* machine.
        // (This also covers the nanotlb and the jitv2 pcp, which the older
        // hand-rolled `nutlb_clear()` here left live across a power-on.)
        exec.resync_privilege_state();
    }
}

impl<T: Tlb + Send + 'static, C: CpuModel + Send + 'static> Saveable for MipsCpu<T, C> {
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
        cp0u32!(cp0_status); cp0u32!(cp0_cause);
        cp0u32!(cp0_prid); cp0u32!(cp0_config); cp0u64!(cp0_lladdr);
        cp0u32!(cp0_watchlo); cp0u32!(cp0_watchhi); cp0u32!(cp0_ecc); cp0u32!(cp0_cacheerr);
        cp0u64!(cp0_taglo); cp0u32!(cp0_taghi);
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
            // Compat shim: pre-timer-based snapshots stored cp0_count/
            // cp0_compare and the learned deltas in 32.32 fixed-point
            // (hardware count in the high word). Values above u32::MAX can
            // only be that old format — shift down to plain hardware counts.
            // (Such snapshots carry count_step, not count_hz, so count_hz
            // stays at its default until the guest's tick is re-recognized.)
            if c.cp0_count > u32::MAX as u64 { c.cp0_count >>= 32; }
            if c.cp0_compare > u32::MAX as u64 { c.cp0_compare >>= 32; }
            // Mirror count_hz into its atomic shadow (read by the display
            // thread) so the live UI matches the restored core state.
            c.count_hz_atomic.store(c.count_hz, std::sync::atomic::Ordering::Relaxed);
            ld32!(cp0_status); ld32!(cp0_cause); ld32!(cp0_prid);
            ld32!(cp0_config); ld64!(cp0_lladdr); ld32!(cp0_watchlo); ld32!(cp0_watchhi);
            ld32!(cp0_ecc); ld32!(cp0_cacheerr); ld64!(cp0_taglo); ld32!(cp0_taghi);
            ld64!(cp0_entrylo0); ld64!(cp0_entrylo1); ld64!(cp0_context);
            ld64!(cp0_pagemask); ld64!(cp0_badvaddr); ld64!(cp0_entryhi);
            ld64!(cp0_xcontext); ld64!(cp0_epc); ld64!(cp0_errorepc);
            // Restart the virtual count from the restored value and re-arm
            // the compare timer — Instants from the previous run are
            // meaningless here — with a live ticket for the restored deadline.
            c.restore_compare_deadline();
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

        // cp0_status was restored above by direct field write, so translate_fn and the
        // fpr accessors still describe the *previous* machine. Deliberately last: the
        // flush this routes through must retire translations made stale by the restored
        // TLB above, not run before it.
        exec.resync_privilege_state();

        Ok(())
    }
}


/// A host call's view of the calling program's memory, a page at a time: its
/// virtual addresses, translated through the TLB under the current
/// address-space id with the access actually needed, and no CPU state touched.
/// A page that would fault -- no TLB entry, invalid, or not dirty for a write
/// -- is reported for the caller to fault in (see `iris_hostcall`).
#[cfg(feature = "hostcall")]
struct HostMemory<'a, T: Tlb, C: CpuModel> {
    exec: &'a mut MipsExecutor<T, C>,
}

#[cfg(feature = "hostcall")]
impl<T: Tlb, C: CpuModel> HostMemory<'_, T, C> {
    /// Sign-extend like a 32-bit register holds an address.
    fn canon(addr: u64) -> u64 {
        addr as u32 as i32 as i64 as u64
    }

    /// Can the program itself make this access to `page`? User addresses only
    /// -- DEBUG translation runs with kernel privilege, so this range check is
    /// what keeps a program from handing us a kernel address -- and the TLB
    /// entry must be there, valid, and dirty for a write.
    fn usable(&mut self, page: u64, write: bool) -> Result<(), iris_hostcall::Fault> {
        let fault = iris_hostcall::Fault { page, write };
        if page >= 0x8000_0000 {
            return Err(fault);
        }
        let access = if write { AccessType::Write } else { AccessType::Read };
        if self.exec.translate_impl::<true>(Self::canon(page), access).is_exception() {
            return Err(fault);
        }
        Ok(())
    }
}

#[cfg(feature = "hostcall")]
impl<T: Tlb, C: CpuModel> iris_hostcall::PageAccess for HostMemory<'_, T, C> {
    fn space(&self) -> u64 {
        self.exec.core.cp0_entryhi & 0xFF
    }

    fn read_page(&mut self, page: u64, buf: &mut [u8; iris_hostcall::PAGE as usize]) -> Result<(), iris_hostcall::Fault> {
        self.usable(page, false)?;
        for (i, chunk) in buf.chunks_exact_mut(8).enumerate() {
            let v = self.exec.read_data_impl::<true, 8>(Self::canon(page + (i as u64) * 8))
                .map_err(|_| iris_hostcall::Fault { page, write: false })?;
            // The host buffer is guest bytes: this is the edge where a guest
            // word becomes bytes in guest (big-endian) order.
            chunk.copy_from_slice(&v.to_be_bytes());
        }
        Ok(())
    }

    fn write_in_page(&mut self, addr: u64, data: &[u8]) -> Result<(), iris_hostcall::Fault> {
        let page = addr & !(iris_hostcall::PAGE - 1);
        debug_assert!(addr + data.len() as u64 <= page + iris_hostcall::PAGE);
        self.usable(page, true)?;
        for (i, &b) in data.iter().enumerate() {
            if self.exec.write_data_impl::<true, 1>(Self::canon(addr + i as u64), b as u64) != EXEC_COMPLETE {
                return Err(iris_hostcall::Fault { page, write: true });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod xcontext_layout_tests {
    use super::xcontext_on_miss;

    /// Field edges as `docs/R4000_um2.pdf` Figure 5-10 draws them, for the
    /// 40-bit CPU the manual describes. These are the numbers in the figure,
    /// not numbers derived from the code under test.
    const R4000_PTEBASE_LSB: u32 = 33;
    const R4000_REGION_LSB: u32 = 31;
    const R4000_REGION_MSB: u32 = 32;
    const R4000_BADVPN2_LSB: u32 = 4;
    const R4000_BADVPN2_MSB: u32 = 30;
    const R4000_BADVPN2_WIDTH: u32 = 27;

    /// A virtual address with a distinct bit in every VPN2 position, so a
    /// misplaced field shows up as a shifted result rather than a coincidence.
    /// Region `11` (kernel), and VA bits 43:13 all set.
    const VA_ALL_VPN2: u64 = 0xFFFF_FFFF_FFFF_E000;

    #[test]
    fn forty_bit_layout_matches_the_r4000_manual_figure() {
        // BadVPN2 is VA[39:13] — the manual's own sentence — landing at
        // bits [30:4], 27 bits wide.
        let got = xcontext_on_miss(40, 0, VA_ALL_VPN2);
        let badvpn2 = (got >> R4000_BADVPN2_LSB) & ((1 << R4000_BADVPN2_WIDTH) - 1);
        assert_eq!(badvpn2, (1u64 << R4000_BADVPN2_WIDTH) - 1,
                   "VA[39:13] should fill the 27-bit BadVPN2 field");
        assert_eq!(got & 0xF, 0, "bits [3:0] are the figure's zero field");
        // Nothing above BadVPN2's top bit except Region, which is next.
        assert_eq!((got >> R4000_REGION_LSB) & 0x3, 0x3, "Region = VA[63:62] = 11");
        assert_eq!(R4000_BADVPN2_MSB + 1, R4000_REGION_LSB,
                   "Region sits directly above BadVPN2");
        assert_eq!(R4000_REGION_MSB + 1, R4000_PTEBASE_LSB,
                   "PTEBase sits directly above Region");
    }

    #[test]
    fn forty_bit_ptebase_is_preserved_above_bit_32() {
        // The OS writes PTEBase; hardware must leave it alone.
        let ptebase = 0xDEAD_BEEF_0000_0000u64 & (!0u64 << R4000_PTEBASE_LSB);
        let got = xcontext_on_miss(40, ptebase, VA_ALL_VPN2);
        assert_eq!(got & (!0u64 << R4000_PTEBASE_LSB), ptebase);
        // And a stale value below PTEBase must not survive.
        let stale = xcontext_on_miss(40, !0u64, 0);
        assert_eq!(stale & !(!0u64 << R4000_PTEBASE_LSB), 0,
                   "everything below PTEBase is rewritten by hardware");
    }

    #[test]
    fn region_carries_virtual_address_bits_63_62() {
        // Table 5-8: 00 user, 01 supervisor, 11 kernel.
        for (va_top, want) in [(0u64, 0u64), (1, 1), (3, 3)] {
            let va = va_top << 62;
            let got = xcontext_on_miss(40, 0, va);
            assert_eq!((got >> R4000_REGION_LSB) & 0x3, want);
        }
    }

    /// The R10000 implements 44 virtual address bits, so BadVPN2 is
    /// `44 - 13 = 31` bits and every field above it moves up by four.
    #[test]
    fn forty_four_bit_layout_shifts_every_field_up_by_four() {
        const W: u32 = 44 - 13; // 31
        let got = xcontext_on_miss(44, 0, VA_ALL_VPN2);
        assert_eq!((got >> 4) & ((1u64 << W) - 1), (1u64 << W) - 1,
                   "VA[43:13] should fill the 31-bit BadVPN2 field");
        assert_eq!((got >> (R4000_REGION_LSB + 4)) & 0x3, 0x3,
                   "Region moves from [32:31] to [36:35]");
        let ptebase = 0xABCD_0000_0000_0000u64 & (!0u64 << (R4000_PTEBASE_LSB + 4));
        let kept = xcontext_on_miss(44, ptebase, VA_ALL_VPN2);
        assert_eq!(kept & (!0u64 << (R4000_PTEBASE_LSB + 4)), ptebase,
                   "PTEBase moves from [63:33] to [63:37]");
    }

    /// The bug this layout fixed: on a 44-bit CPU the 40-bit layout puts
    /// Region and PTEBase four bits too low, so the two disagree.
    #[test]
    fn forty_and_forty_four_bit_layouts_actually_differ() {
        let va = 0xC000_0FFF_FFFF_E000u64;
        assert_ne!(xcontext_on_miss(40, 0, va), xcontext_on_miss(44, 0, va));
    }

    /// VA bits above the implemented width must not reach BadVPN2 — on a
    /// 40-bit CPU, VA[43:40] is not part of the page number.
    #[test]
    fn address_bits_above_the_implemented_width_are_excluded() {
        let only_high = 0x0000_0F00_0000_0000u64; // VA[43:40]
        assert_eq!(xcontext_on_miss(40, 0, only_high), 0,
                   "VA[43:40] is outside a 40-bit CPU's BadVPN2");
        assert_ne!(xcontext_on_miss(44, 0, only_high), 0,
                   "but it is inside a 44-bit CPU's");
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
