// MIPS R4000/R10000 CPU Core

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

// CP0 Status Register bit definitions
pub const STATUS_IE: u32 = 1 << 0;      // Interrupt Enable
pub const STATUS_EXL: u32 = 1 << 1;     // Exception Level
pub const STATUS_ERL: u32 = 1 << 2;     // Error Level
pub const STATUS_KSU_MASK: u32 = 0x3 << 3; // Kernel/Supervisor/User mode mask
pub const STATUS_KSU_SHIFT: u32 = 3;    // KSU field shift
pub const STATUS_UX: u32 = 1 << 5;      // User mode 64-bit addressing
pub const STATUS_SX: u32 = 1 << 6;      // Supervisor mode 64-bit addressing
pub const STATUS_KX: u32 = 1 << 7;      // Kernel mode 64-bit addressing
pub const STATUS_IM_MASK: u32 = 0xFF << 8; // Interrupt Mask (8 bits)
pub const STATUS_IM_SHIFT: u32 = 8;     // Interrupt Mask shift
pub const STATUS_DE: u32 = 1 << 16;     // Disable Cache Exceptions
pub const STATUS_CE: u32 = 1 << 17;     // Cache Error
pub const STATUS_CH: u32 = 1 << 18;     // Cache Hit
pub const STATUS_SR: u32 = 1 << 20;     // Soft Reset
pub const STATUS_TS: u32 = 1 << 21;     // TLB Shutdown
pub const STATUS_BEV: u32 = 1 << 22;    // Bootstrap Exception Vectors
pub const STATUS_RE: u32 = 1 << 25;     // Reverse Endian
pub const STATUS_FR: u32 = 1 << 26;     // FPU Register mode (32/64-bit)
pub const STATUS_RP: u32 = 1 << 27;     // Reduced Power
pub const STATUS_CU0: u32 = 1 << 28;    // Coprocessor 0 Usable
pub const STATUS_CU1: u32 = 1 << 29;    // Coprocessor 1 (FPU) Usable
pub const STATUS_CU2: u32 = 1 << 30;    // Coprocessor 2 Usable
pub const STATUS_CU3: u32 = 1 << 31;    // Coprocessor 3 Usable

// CP0 Cause Register bit definitions
pub const CAUSE_EXCCODE_MASK: u32 = 0x1F << 2; // Exception Code mask
pub const CAUSE_EXCCODE_SHIFT: u32 = 2;        // Exception Code shift
pub const CAUSE_IP_MASK: u32 = 0xFF << 8;      // Interrupt Pending mask
pub const CAUSE_IP_SHIFT: u32 = 8;             // Interrupt Pending shift
pub const CAUSE_IP0: u32 = 1 << 8;
pub const CAUSE_IP1: u32 = 1 << 9;
pub const CAUSE_IP2: u32 = 1 << 10;
pub const CAUSE_IP3: u32 = 1 << 11;
pub const CAUSE_IP4: u32 = 1 << 12;
pub const CAUSE_IP5: u32 = 1 << 13;
pub const CAUSE_IP6: u32 = 1 << 14;
pub const CAUSE_IP7: u32 = 1 << 15;            // Timer interrupt (IP7)

/// Default virtual CP0 Count frequency until a recognized periodic tick lets
/// `infer_count_hz` measure the real rate the guest believes in.
pub const DEFAULT_COUNT_HZ: u64 = 33_000_000;

/// ci_clock: synthetic nanoseconds per retired instruction (~100 MIPS R4400).
/// Virtual time is derived from `hot.cycles` instead of the wall clock so the
/// snapshot determinism validator stays reproducible at any host speed.
#[cfg(feature = "ci_clock")]
pub const NS_PER_GUEST_CYCLE: u64 = 10;

/// Raw pointer to a `MipsCore`'s `hot.interrupts` word, captured by the armed
/// compare-timer callback so the hptimer thread can raise IP7. Same
/// process-lifetime validity argument as `CyclesPtr` above and
/// `Ioc::set_interrupts`'s stored pointer: it points into the core owned by
/// the executor's top-level `Arc<Mutex<..>>`, and `Drop for MipsCore` removes
/// the timer before the core goes away.
struct TimerIrqPtr(*const AtomicU64);
unsafe impl Send for TimerIrqPtr {}
pub const CAUSE_CE_MASK: u32 = 0x3 << 28;      // Coprocessor Error mask
pub const CAUSE_CE_SHIFT: u32 = 28;            // Coprocessor Error shift
pub const CAUSE_BD: u32 = 1 << 31;             // Branch Delay

// KSU field values
pub const KSU_KERNEL: u32 = 0b00;
pub const KSU_SUPERVISOR: u32 = 0b01;
pub const KSU_USER: u32 = 0b10;

/// The two words in `MipsCore` every other thread in the process might read
/// on any given tick — grouped and cache-line-aligned so they share one
/// line deliberately (both are single-word, cross-thread-read,
/// per-instruction-adjacent counters — no reason to let the compiler's
/// default layout scatter them next to fields with completely different
/// access patterns and risk false sharing).
///
/// Both fields are inline (not `Arc<AtomicU64>`) so they're fixed
/// `offset_of!` fields reachable directly from a bare `*mut MipsCore` — the
/// interpreter's hot loop and JIT-compiled code both need to touch these
/// with zero indirection (no Arc deref, no separate heap allocation to
/// chase). External devices/threads that need to read/set them get a raw
/// pointer into this struct instead of a cloned `Arc` — see
/// `MipsExecutor::interrupts_ptr`/`cycles_ptr` and `MipsCpu`'s own
/// same-named accessors. Safe because the executor (and therefore this
/// `MipsCore`) lives in a top-level `Arc<Mutex<...>>` that outlives every
/// device for the life of the process — callers obtain the pointer once,
/// after the executor has reached its final, stable address.
#[repr(align(64))]
#[derive(Default)]
pub struct Hot {
    /// Interrupt-pending word. Bits 8..15 = IP0..IP7 (mirror CAUSE.IP
    /// layout). Bit 63 = soft-reset request. A real atomic (unlike
    /// `cycles` below): devices set/clear individual bits from their own
    /// thread via `fetch_or`/`fetch_and`, which needs a genuine RMW, not
    /// just eventual visibility of a monotonic count.
    pub interrupts: AtomicU64,
    /// Instruction/cycle counter — incremented directly, every instruction,
    /// by whichever engine (interpreter or JIT-compiled code) retires it.
    /// Never a batched local shadow flushed only when control returns to
    /// `step()`'s outer loop: a dispatch loop that stays entirely inside
    /// JIT-compiled code for a long stretch (a hot guest loop) would
    /// otherwise never publish progress to `cycles` at all until it
    /// happened to exit — and at least one real guest workaround (a BSD
    /// SCSI driver's busy-wait, `Wd33c93a`'s deferred-interrupt worker
    /// thread) depends on this counter visibly incrementing while it spins,
    /// not just eventually catching up.
    ///
    /// Plain `u64`, not `AtomicU64`: readers on other threads only need
    /// eventual visibility of the count itself, not a synchronizing RMW —
    /// incremented via `ptr::write_volatile`/read via `ptr::read_volatile`
    /// (see `MipsExecutor::step`'s increment site) rather than `fetch_add`,
    /// cheaper on the hot path while still guaranteeing the compiler can't
    /// elide or reorder the write away, which a plain non-volatile write
    /// technically could across an unbounded loop.
    pub cycles: u64,
}

/// A raw pointer into `Hot::cycles`, handed out by `MipsExecutor::cycles_ptr`/
/// `MipsCpu::cycles_ptr` to devices on other threads that need to read the
/// live cycle count (status displays, `Wd33c93a`'s deferred-interrupt
/// spin-wait). Wraps the bare `*const u64` in a `Send`/`Sync` newtype
/// instead of blanket-asserting `unsafe impl Send`/`Sync` on every struct
/// that stores one — narrows the safety claim to exactly this one pointer
/// (valid for the process lifetime once obtained: it points into the
/// `MipsCore` owned by the executor's own top-level `Arc<Mutex<...>>`, which
/// outlives every device) rather than silencing the compiler's check for a
/// struct's other fields too, which would then need re-verifying by
/// inspection every time a new field is added.
///
/// `Copy`: cheap to pass/store by value everywhere a bare pointer would be,
/// no lifetime to track. Read with `.get()`, which does the
/// `ptr::read_volatile` itself — see `Hot::cycles`'s doc comment for why a
/// volatile read, not a plain one.
#[derive(Clone, Copy)]
pub struct CyclesPtr(*const u64);

unsafe impl Send for CyclesPtr {}
unsafe impl Sync for CyclesPtr {}

impl CyclesPtr {
    /// Construct from a raw pointer obtained via `MipsExecutor::cycles_ptr`/
    /// `MipsCpu::cycles_ptr` — never call this with anything else.
    pub fn new(ptr: *const u64) -> Self {
        Self(ptr)
    }

    /// A `CyclesPtr` that reads as `0` forever — for fields not yet wired up
    /// (`Rex3`/`Wd33c93a` are constructed before the CPU exists; see
    /// `set_cpu_cycles` on each).
    pub const fn dangling() -> Self {
        Self(std::ptr::null())
    }

    /// Read the current cycle count. Volatile: see `Hot::cycles`'s doc
    /// comment for why a plain read isn't enough. Returns 0 if this
    /// `CyclesPtr` hasn't been wired up yet (`dangling()`/not yet set).
    pub fn get(self) -> u64 {
        if self.0.is_null() { 0 } else { unsafe { std::ptr::read_volatile(self.0) } }
    }
}

/// MIPS CPU Core with full register state.
///
/// Field order is deliberately cache-conscious, not declaration-order or
/// architectural-grouping: the fields every dispatch touches first
/// (`hot`, then the timer trio, then PC/branch-delay state, then the
/// register file) lead the struct; large, per-instruction-irrelevant CP0
/// registers and the compare-calibration bookkeeping (only ever touched on
/// a CP0 Compare write, not every dispatch) trail at the end.
///
/// `#[repr(C)]`: JIT v2 compiled code addresses fields directly via a pinned
/// base pointer + fixed byte offset (`mov [base+disp]`, rules/jitv2/jit-v2-design.md
/// §5 — "memory-resident registers", no copy-in/copy-out shadow struct like v1's
/// `JitContext`). That requires deterministic, ABI-stable field layout, which
/// `repr(Rust)` does not guarantee. Offsets must be computed via
/// `std::mem::offset_of!` at codegen time, never hardcoded — cfg-gated fields are
/// still fine, they just shift the layout deterministically per build.
#[repr(C)]
pub struct MipsCore {
    /// Interrupt-pending word and instruction/cycle counter — the two
    /// fields every other thread in the process might read on any given
    /// tick, grouped into one cache-line-aligned struct (see [`Hot`]'s own
    /// doc comment) rather than left to whatever the compiler's default
    /// field layout happens to produce. `#[repr(align(64))]` on `Hot` itself
    /// means it fills exactly one cache line on its own — placed first so
    /// the timer trio and pc/branch-delay state right after it start a
    /// fresh line of their own instead of splitting across `Hot`'s tail.
    pub hot: Hot,

    // Timer pair — CP0 Count is *virtual*: nothing increments it per
    // instruction. `cp0_count` holds the value last materialized by
    // `count_now()` (plain hardware counts in the low 32 bits), advanced
    // lazily from the wall-clock anchor (`count_anchor_instant` ×
    // `count_hz`) whenever the guest reads Count or writes Compare. The
    // Count==Compare interrupt is delivered by an hptimer one-shot armed on
    // every Compare write, whose callback just sets IP7 in `hot.interrupts`
    // — the step() preamble's existing pending-interrupt merge does the rest.
    pub cp0_count: u64,       // 9: Timer Count at last materialization (plain hw counts, low 32 bits)
    pub cp0_compare: u64,     // 11: Timer Compare (plain hw counts, low 32 bits)

    // PC and branch-delay state — read/written together on every dispatch.
    pub pc: u64,         // Program Counter
    /// Whether the instruction about to execute is a branch/jump's delay
    /// slot — set by `branch_delay` when the branch itself dispatches,
    /// cleared by `handle_exec_complete` once the slot retires. Consulted by
    /// `deliver_exception`/`handle_exception` to compute EPC/Cause.BD
    /// correctly for an exception raised from within a delay slot (EPC must
    /// point at the *branch*, not the slot). One field, one meaning, for
    /// both the interpreter (which dispatches the slot as its own step) and
    /// jitv2-compiled code (`emit_slot_semantics` sets/clears this directly
    /// around a delay slot it inlines, since it has no separate dispatch
    /// step of its own to hang the flag on) — there is no second,
    /// JIT-specific copy of this state.
    pub in_delay_slot: bool,
    /// The real branch/jump target `handle_exec_complete` commits to `pc`
    /// once the pending delay slot (`in_delay_slot`) retires — set by
    /// `MipsExecutor::branch_delay` alongside `in_delay_slot = true`.
    /// Lives on `MipsCore` (not just the executor) for the same reason
    /// `in_delay_slot` does: a delay-slot word can be JIT-compiled and
    /// dispatched as an *ordinary* standalone entry — the interpreter having
    /// armed `in_delay_slot`/`delay_slot_target` on the previous dispatch
    /// has no bearing on how that word got compiled (compile_region has no
    /// idea a given entry word might sometimes be a foreign delay slot in
    /// flight; see `EmitCtx`'s doc comment on `entry_word`) — so a plain
    /// entry-word compile's exit must check `in_delay_slot` at runtime and,
    /// if set, honor `delay_slot_target` instead of its own compile-time
    /// fallthrough word, exactly mirroring `handle_exec_complete`. Without
    /// this being readable from JIT-compiled code, that word silently
    /// discards the pending transfer (observed live: the IRIX PROM reset
    /// vector's own `j realstart` — reset vector's delay-slot nop got
    /// JIT-compiled standalone via the `entry_offset == 1` dispatch-gate
    /// probe, in `mips_exec.rs`'s `exec_decoded` — landed on the next
    /// sequential word instead of `realstart`).
    pub delay_slot_target: u64,

    // General Purpose Registers (GPRs)
    pub gpr: [u64; 32],  // r0-r31, where r0 is always zero
    pub hi: u64,         // Multiply/Divide HI result
    pub lo: u64,         // Multiply/Divide LO result

    // CP1 - Floating Point Unit Registers
    pub fpr: [u64; 32],       // FPU data registers (64-bit, can be used as pairs for 32-bit)
    pub fpu_fir: u32,         // 0: FP Implementation/Revision
    pub fpu_fccr: u32,        // 25: FP Condition Codes
    pub fpu_fexr: u32,        // 26: FP Exceptions
    pub fpu_fenr: u32,        // 28: FP Enables
    pub fpu_fcsr: u32,        // 31: FP Control/Status

    /// Nano-TLB: 3-entry direct-mapped cache, one slot per access type (Fetch/Read/Write).
    /// Indexed by AccessType discriminant (0=Fetch, 1=Read, 2=Write).
    pub nanotlb: [NanoTlbEntry; 3],

    /// Called whenever CP0 Status (reg 12) is written, with (old_value, new_value).
    /// The first element is the callback function, the second is an opaque context pointer
    /// (typically a type-erased `*mut MipsExecutor<T,C>` set by the executor after construction).
    pub status_changed_cb: Option<(fn(*mut core::ffi::c_void, u32, u32), *mut core::ffi::c_void)>,

    /// JIT v2: monomorphized C-ABI memory-access and exception-delivery
    /// hooks, installed once by `MipsExecutor::install_jit_hooks` (mirrors
    /// `status_changed_cb`'s established pattern — a type-erased context
    /// pointer plus free functions generated per `<T,C>` instantiation, e.g.
    /// `translate_32_kernel::<T,C>`). Compiled code reaches all of these at
    /// fixed `offset_of!` offsets — no vtable/trait-object call, and no
    /// per-instantiation Cranelift codegen needed since the *pointer value*
    /// (not the compiled code) carries the monomorphization.
    ///
    /// `jit_ctx` is the shared first argument to every fn below — a
    /// type-erased `*mut MipsExecutor<T,C>`, exactly like
    /// `status_changed_cb`'s context pointer (never derived from `&MipsCore`
    /// itself, since `MipsCore` is not guaranteed to be `MipsExecutor`'s
    /// first field in memory — see `install_jit_hooks`'s doc comment).
    #[cfg(feature = "jitv2")]
    pub jit_ctx: *mut core::ffi::c_void,
    /// Read wrappers: return the loaded value (zero-extended to u64 for
    /// sub-64-bit widths, matching `MipsExecutor::read_data`'s own
    /// convention), and set `jit_mem_exc` to `EXEC_COMPLETE` (0) on success
    /// or the fault's `ExecStatus` (`EXEC_IS_EXCEPTION` bit set) otherwise —
    /// compiled code must check `jit_mem_exc` after every call before
    /// trusting the returned value. A single reusable field rather than an
    /// out-param: exactly one memory op is ever in flight per compiled unit
    /// before the check, so nothing aliases, and this avoids a Cranelift
    /// stack-slot alloca per access.
    #[cfg(feature = "jitv2")]
    pub read8_fn: unsafe extern "C" fn(*mut core::ffi::c_void, u64) -> u64,
    #[cfg(feature = "jitv2")]
    pub read16_fn: unsafe extern "C" fn(*mut core::ffi::c_void, u64) -> u64,
    #[cfg(feature = "jitv2")]
    pub read32_fn: unsafe extern "C" fn(*mut core::ffi::c_void, u64) -> u64,
    #[cfg(feature = "jitv2")]
    pub read64_fn: unsafe extern "C" fn(*mut core::ffi::c_void, u64) -> u64,
    /// Write wrappers: return `EXEC_COMPLETE` (0) on success or the fault's
    /// `ExecStatus` otherwise — the return value here doubles as
    /// `jit_mem_exc`'s value (also mirrored into the field for symmetry with
    /// the read path, so compiled code can use one check helper for both).
    ///
    /// The value parameter is `u64` for every width, not `u8`/`u16`/`u32` —
    /// deliberately: the x86-64 SysV C ABI does not guarantee a sub-word
    /// integer argument's upper register bits are zeroed by the caller (the
    /// *callee* is responsible for masking if it needs a clean value), and a
    /// hand-built `cranelift_jit::JITModule::call_indirect` signature (see
    /// `codegen.rs`'s `emit_mem_write`) has no obligation to do so either —
    /// only whatever's in the low N bits is meaningful. A `u8`/`u16`/`u32`-
    /// typed parameter here previously let garbage in the unused high bits
    /// of the argument register leak into any `val as u64` widening the
    /// callee did (observed live: SB writing garbage-prefixed values like
    /// `0xffffff00` instead of `0x00` — the low byte was always correct, the
    /// upper 24 bits were uninitialized register contents). `write*_fn`'s
    /// caller (`jit_write8`/`jit_write16`/`jit_write32`/`jit_write64` in
    /// `mips_exec.rs`) is responsible for masking to the real width itself
    /// before using the value, exactly like `read*_fn`'s callers already
    /// mask/extend a full u64 return value down to the size they need.
    #[cfg(feature = "jitv2")]
    pub write8_fn: unsafe extern "C" fn(*mut core::ffi::c_void, u64, u64) -> u32,
    #[cfg(feature = "jitv2")]
    pub write16_fn: unsafe extern "C" fn(*mut core::ffi::c_void, u64, u64) -> u32,
    #[cfg(feature = "jitv2")]
    pub write32_fn: unsafe extern "C" fn(*mut core::ffi::c_void, u64, u64) -> u32,
    #[cfg(feature = "jitv2")]
    pub write64_fn: unsafe extern "C" fn(*mut core::ffi::c_void, u64, u64) -> u32,
    /// Masked doubleword write for the unaligned store family (SWL/SWR/
    /// SDL/SDR — `MipsExecutor::write_data64_masked`'s JIT-callable
    /// equivalent): `(jit_ctx, virt_addr, val, mask)`, writes only the byte
    /// lanes set in `mask` at the (already doubleword-aligned) `virt_addr`,
    /// leaving the rest of that doubleword untouched. Unlike
    /// `write8_fn`/`write16_fn`/`write32_fn` (which each address a plain,
    /// fixed-width, natively-aligned unit), a masked write goes through a
    /// genuinely different bus/cache path (`BusDevice::write64_masked`/
    /// `MipsCache::write64_masked`) that can partially update a doubleword
    /// no single plain-width write could express — SWL/SWR/SDL/SDR need
    /// this because they write a runtime-variable byte range starting at an
    /// unaligned address, not a fixed natural width. Returns `EXEC_COMPLETE`
    /// (0) on success or the fault's `ExecStatus` otherwise, same
    /// convention as `write*_fn`.
    #[cfg(feature = "jitv2")]
    pub write64_masked_fn: unsafe extern "C" fn(*mut core::ffi::c_void, u64, u64, u64) -> u32,
    /// Single-implementation exception delivery (§4.2): identical to
    /// `MipsExecutor::handle_exception`, callable from compiled code.
    /// Mutates Cause/EPC/Status/pc (vectors `core.pc` to the handler) and
    /// returns the same `status` it was given. Compiled code's exception
    /// exit stub calls this, then returns `EXEC_COMPLETE` to the
    /// interpreter loop — pc is already at the vector, nothing left to do.
    #[cfg(feature = "jitv2")]
    pub handle_exception_fn: unsafe extern "C" fn(*mut core::ffi::c_void, u32) -> u32,
    /// Fetch, decode, and execute exactly one instruction at the current
    /// `core.pc` through the real interpreter dispatch (`MipsExecutor::step`'s
    /// own fetch+exec_decoded path) and return its `ExecStatus`. Exists so
    /// compiled code can force genuine forward progress on a condition it
    /// can't itself resolve — `emit_fpu_entry_guard`'s FR-mismatch case
    /// (paired with `kill_entry_fn`, see that field) is the current caller:
    /// the JIT's own bail-to-exit_block just re-sets `core.pc` back to the
    /// same instruction and returns `EXEC_COMPLETE`, which `exec_decoded`'s
    /// caller can't distinguish from a real retirement — if this same PC is
    /// still published/hot, the very next dispatch calls the identical
    /// compiled function again, which bails again, forever. Calling this
    /// directly instead guarantees the interpreter's real semantics run for
    /// this one instruction no matter what compiled code decided it
    /// couldn't handle. (CU1-clear no longer routes through here — see
    /// `emit_fpu_entry_guard`'s doc comment: it materializes the real
    /// `EXC_CPU` exception directly via `handle_exception_fn`, the same way
    /// every other JIT-detected fault does, since the exact exception code
    /// is known statically and doesn't need the interpreter's help to
    /// determine.)
    #[cfg(feature = "jitv2")]
    pub interp_fallback_fn: unsafe extern "C" fn(*mut core::ffi::c_void) -> u32,
    /// Un-publish the calling compiled function's own `(page, offset)` entry
    /// (`PhysicalCodePage::kill`) so the JIT dispatch gate stops re-selecting
    /// it. `offset` is the entry's own word offset within its page — the
    /// only piece of `(page, offset)` compiled code doesn't already have
    /// another way to recover (the executor's own `self.pcp` already tracks
    /// the live page). Paired with `interp_fallback_fn` in
    /// `emit_fpu_entry_guard`'s FR-mismatch arm: the whole compiled unit was
    /// built assuming a specific `STATUS_FR` value that's no longer live, so
    /// unlike a CU1 fault (a real, opcode-independent MIPS exception —
    /// materialized directly, no kill needed) the entire artifact is wrong,
    /// not just this one dispatch — killing it means the next visit gets a
    /// genuine fresh compile against whatever FR mode is actually live then,
    /// instead of this exact same stale function being dispatched and
    /// bailing again on every future visit. `u32`, not `u16` (the real value
    /// range) — a sub-32-bit `extern "C"` param isn't reliably zero-extended
    /// by every caller/ABI; see `dev_trace_bp_fn`'s doc comment for the live
    /// bug this class of mistake already caused once in this same file.
    #[cfg(feature = "jitv2")]
    pub kill_entry_fn: unsafe extern "C" fn(*mut core::ffi::c_void, u32),
    /// Developer-only per-instruction hook, called by compiled code at the top
    /// of every instruction's block (right where the interrupt preamble runs —
    /// `codegen::emit_dev_trace_bp`), with the synthesized `pc`/`raw` and an
    /// `origin` code (`mips_exec::InstrOrigin as u32`, a compile-time constant
    /// each codegen call site passes for its own case — plain body, entry
    /// word via back-edge, inlined delay slot, fallback successor, etc.).
    /// Deliberately `u32`, not `u8`: a sub-32-bit `extern "C"` param isn't
    /// guaranteed zero/sign-extended by the caller on every ABI, and this
    /// codebase has already been bitten by that exact class of bug for
    /// narrow write-values crossing this same kind of call — every hook
    /// signature in this file uses `u32`+ for that reason, this one included
    /// (was `u8`, silently read back wrong — the origin tag never showed up
    /// in `dt` on a live boot despite the hook demonstrably firing millions
    /// of times per `j2 stats`'s `dev trace hook` counter).
    /// Records the traceback entry tagged with that origin and the trace-file
    /// record, and checks PC breakpoints — restoring the per-instruction
    /// `dt`/breakpoint visibility the interpreter's `step()` has but a
    /// compiled region otherwise runs straight through. Returns
    /// `EXEC_BREAKPOINT` if a breakpoint fired at `pc` (compiled code then
    /// bails to the exit with that status, stopping *before* the instruction
    /// executes, `core.pc` already at it so resume works), else
    /// `EXEC_COMPLETE`. Only emitted/installed under `developer` (a plain
    /// release/`lightning` build never pays for it); the not-installed sentinel
    /// panics if ever reached without `install_jit_hooks`.
    #[cfg(all(feature = "jitv2", feature = "developer"))]
    pub dev_trace_bp_fn: unsafe extern "C" fn(*mut core::ffi::c_void, u64, u32, u32) -> u32,
    /// Scratch: exception status from the most recent `read*_fn`/`write*_fn`
    /// call (`EXEC_COMPLETE` i.e. 0 = no fault). See the read-wrapper fields'
    /// doc comment above for the full contract.
    #[cfg(feature = "jitv2")]
    pub jit_mem_exc: u32,
    /// FPU host-status hooks, mirroring `MipsExecutor::fpu_update_fcsr`'s
    /// `platform::clear_fpu_status`/`platform::get_fpu_status` calls so
    /// compiled FP arithmetic can update FCSR's Cause/Flag bits and raise
    /// `EXC_FPE` exactly like the interpreter — these are plain host-arch
    /// free functions (no executor/generic context needed, unlike the
    /// memory/exception hooks), so `jit_ctx` is passed but unused; kept for
    /// signature uniformity with the other hooks rather than a special case.
    /// `fpu_get_status_fn` returns host FP exception flags already
    /// translated into MIPS FCSR bit positions [6:2] (V,Z,O,U,I) — see
    /// `platform::get_fpu_status`'s doc comment. `fpu_clear_status_fn`
    /// clears the host's sticky flags for the next op.
    #[cfg(feature = "jitv2")]
    pub fpu_get_status_fn: unsafe extern "C" fn(*mut core::ffi::c_void) -> u32,
    #[cfg(feature = "jitv2")]
    pub fpu_clear_status_fn: unsafe extern "C" fn(*mut core::ffi::c_void),
    /// Reprogram the host FPU rounding mode, mirroring
    /// `MipsCore::write_fpu_control`'s `platform::set_fpu_mode(rm)` call on
    /// an FCSR (reg 31) write. `rm` is the 2-bit MIPS rounding mode (FCSR
    /// bits [1:0]). Same host-arch-free-function shape as the status hooks
    /// above — `jit_ctx` unused.
    #[cfg(feature = "jitv2")]
    pub fpu_set_mode_fn: unsafe extern "C" fn(*mut core::ffi::c_void, u32),
    /// `jitv2_lockstep` load/store verification scratch (see
    /// `MipsExecutor::lockstep_check_load_store`): records the real
    /// interpreter dispatch's virtual address, translated physical address,
    /// and data value for whichever single load/store instruction is being
    /// compared, so the JIT probe's own lockstep-only memory hooks
    /// (`lockstep_jit_read`/`lockstep_jit_write`) can compare against a real
    /// access instead of touching the bus a second time — a load can't be
    /// safely re-issued (MMIO side effects) and a store can't be safely
    /// re-applied (would double it), so this is the only way to compare "did
    /// the JIT compute the same address/value" without actually running the
    /// access twice.
    ///
    /// `Option`-shaped rather than three plain fields plus a separate
    /// "valid" bool on purpose: an earlier version used a `lockstep_mem_valid:
    /// bool` that `read_data_impl`/`write_data_impl` set `true`
    /// unconditionally (success or fault) while only updating the
    /// address/value fields on success — meaning a faulted or retried real
    /// access left `valid=true` pointing at stale data from whatever earlier
    /// access last succeeded, which `lockstep_jit_read`/`lockstep_jit_write`
    /// would then silently compare the JIT probe against, producing a
    /// spurious divergence report for a real EXEC_RETRY/exception that
    /// `lockstep_check_load_store` now instead detects up front and skips
    /// the JIT probe for entirely. Making "nothing to compare" an actual
    /// `None` closes that class of bug structurally: `read_data_impl`/
    /// `write_data_impl` only ever write `Some(..)` on a real success, and
    /// explicitly write `None` on any non-success path — there is no way
    /// for a stale `Some` to survive past a failed access.
    ///
    /// `None` also naturally covers "no real interpreter access has ever
    /// happened on this `MipsCore` at all" (every direct `jit_fn(...)` test
    /// in `equiv_test.rs` that touches memory, for instance, which never
    /// goes through `lockstep_check_load_store`'s interpreter-first
    /// dispatch at all): `lockstep_jit_read`/`lockstep_jit_write` see `None`
    /// and fall through to a real bus access instead of asserting — same
    /// behavior a direct JIT test expects today.
    #[cfg(feature = "jitv2_lockstep")]
    pub lockstep_mem: Option<LockstepMemCapture>,
    /// Inline per-instruction lockstep: called by compiled code at the *start*
    /// of every JIT instruction (`codegen::emit_lockstep_step`), with the
    /// synthesized `pc`/`raw`/`bd` for that instruction. It sets `core.pc`/
    /// `in_delay_slot` to those, snapshots the pre-instruction state, runs the
    /// instruction once through the real interpreter handler (committing any
    /// memory side effect and capturing it in `lockstep_mem`), stashes the
    /// interpreter's resulting state for the later compare, then restores the
    /// pre-instruction state so the JIT can run the same instruction for real
    /// against identical inputs. See `MipsExecutor::jit_lockstep_step`. `bd`
    /// is `u32`, not `u8`, despite only ever holding 0/1/`LOCKSTEP_BD_LIVE`
    /// (0xFF) — a sub-32-bit `extern "C"` param isn't reliably zero-extended
    /// by every caller/ABI (see `dev_trace_bp_fn`'s doc comment for the
    /// exact live bug this class of mistake already caused once in this same
    /// file).
    #[cfg(feature = "jitv2_lockstep")]
    pub lockstep_step_fn: unsafe extern "C" fn(*mut core::ffi::c_void, u64, u32, u32),
    /// Inline per-instruction lockstep: called by compiled code at the *end* of
    /// every JIT instruction (`codegen::emit_lockstep_compare`), after the JIT
    /// ran the instruction's own semantics. Compares the JIT's now-current
    /// state against the interpreter result `lockstep_step_fn` stashed and
    /// panics on any divergence. Because the lockstep build makes the JIT
    /// materialize `core.pc` per instruction (start = pc, end = pc+4 or the
    /// branch's own target write), the pc/delay-slot compare is direct — no
    /// branch/slot-model reconciliation needed. See
    /// `MipsExecutor::jit_lockstep_compare`.
    #[cfg(feature = "jitv2_lockstep")]
    pub lockstep_compare_fn: unsafe extern "C" fn(*mut core::ffi::c_void) -> u32,
    /// Set whenever PC is about to land on an offset that is a legitimate
    /// branch/jump target — the compile-worthiness signal `exec_decoded`'s
    /// JIT dispatch gate checks alongside the offset-4/valid-bit conditions
    /// (see `MipsExecutor::exec_decoded`'s doc comment for the full gate).
    /// Lives on `MipsCore` rather than `MipsExecutor` specifically so
    /// JIT-compiled code's own jump/branch exit stubs (`emit_absolute_pc_exit`,
    /// `emit_runtime_pc_exit` in codegen.rs) can set it directly via a plain
    /// store through `core_ptr` before returning to the interpreter — without
    /// this, a jump taken *from* JIT-compiled code landing on a fresh,
    /// never-before-published word would arrive with no trigger at all
    /// (unless it happened to coincidentally sit at offset 4), silently
    /// stalling that address in the interpreter forever even under a hot
    /// loop reached exclusively via JIT-to-JIT control transfer. The
    /// interpreter's own terminal actions (`handle_exec_complete`'s
    /// delay-slot retirement, `exec_complete_pc_set`) set it the same way,
    /// just from Rust instead of emitted IR. Cleared by `exec_decoded`'s
    /// dispatch-time check, which is the only reader.
    #[cfg(feature = "jitv2")]
    pub jit_trigger: bool,

    // --- everything below is cold: not touched on the common per-instruction path ---

    // CP0 - System Control Coprocessor Registers (rest of the bank)
    pub cp0_index: u32,       // 0: TLB Index
    pub cp0_random: u32,      // 1: TLB Random
    pub cp0_entrylo0: u64,    // 2: TLB Entry Low 0 (64-bit, truncated in 32-bit mode)
    pub cp0_entrylo1: u64,    // 3: TLB Entry Low 1 (64-bit, truncated in 32-bit mode)
    pub cp0_context: u64,     // 4: Context (page table pointer)
    pub cp0_pagemask: u64,    // 5: TLB Page Mask (64-bit, truncated in 32-bit mode)
    pub cp0_wired: u32,       // 6: TLB Wired boundary
    pub cp0_badvaddr: u64,    // 8: Bad Virtual Address
    pub cp0_entryhi: u64,     // 10: TLB Entry High (64-bit, truncated in 32-bit mode)
    pub cp0_status: u32,      // 12: Status Register
    pub cp0_cause: u32,       // 13: Cause Register
    pub cp0_epc: u64,         // 14: Exception Program Counter
    pub cp0_prid: u32,        // 15: Processor Revision ID
    pub cp0_config: u32,      // 16: Configuration Register
    pub cp0_lladdr: u32,      // 17: LLAddr (also mirrored on d_cache for invalidation)
    pub cp0_watchlo: u32,     // 18: Watchpoint Low
    pub cp0_watchhi: u32,     // 19: Watchpoint High
    pub cp0_xcontext: u64,    // 20: Extended Context (64-bit)
    pub cp0_ecc: u32,         // 26: ECC Register
    pub cp0_cacheerr: u32,    // 27: Cache Error
    pub cp0_taglo: u32,       // 28: Cache Tag Low
    pub cp0_taghi: u32,       // 29: Cache Tag High
    pub cp0_errorepc: u64,    // 30: Error Exception PC

    pub tlb_entries: u32,     // Total TLB entries
    pub cp0_random_cycle: u64, // Cycle count of last Random update

    /// Counts every CP0 Count==Compare match (i.e. every fastick interrupt).
    pub fasttick_count: Arc<AtomicU64>,

    // Execution state
    pub running: bool,
    pub halted: bool,

    // Timer bookkeeping — touched only on a CP0 Count/Compare access or a
    // timer (re)schedule, never on the common per-instruction path.
    /// Atomic shadow of `count_hz` — updated whenever the inferred frequency
    /// changes. Shared with the display refresh thread for status bar display
    /// (watch it to see how stable the inference is).
    pub count_hz_atomic: Arc<AtomicU64>,
    /// Wall-clock instant `cp0_count` was last materialized at. Advanced by
    /// exactly the whole-tick duration consumed on each materialization so
    /// the sub-tick remainder is never rounded away. Reset to
    /// `Instant::now()` on snapshot load — Instants from a previous run are
    /// meaningless across a restore.
    pub(crate) count_anchor_instant: std::time::Instant,
    /// `hot.cycles` at the last `count_now()` materialization. Memo key: a
    /// second read within the same instruction returns the identical value
    /// (jitv2_lockstep re-executes an instruction on both engines and must
    /// see the same Count both times).
    pub(crate) count_read_cycle: u64,
    /// Inferred CP0 Count frequency in Hz. Default 33 MHz (user-facing
    /// assumption for an uncalibrated core); replaced as soon as a Compare
    /// delta matches a recognized slow (100 Hz) or fast (1 kHz) tick.
    /// Left alone (never overwritten) once `count_hz_fixed` is set.
    pub count_hz: u64,
    /// When set, `count_hz` stays pinned at this value forever and
    /// `infer_count_hz`'s slow/fast tick pattern-matching is skipped
    /// entirely. Set from `[clock] fixed_mhz` (`iris.toml`) / `--clock-fixed-mhz`
    /// for guests (e.g. Linux) whose periodic tick doesn't fit IRIX's
    /// two-bucket (100 Hz slow / 1 kHz fast) model — the inference would
    /// either misclassify their tick or chase a moving target. None = the
    /// default IRIX-oriented auto-inference behavior.
    pub count_hz_fixed: Option<u64>,
    /// True while the CPU thread is stopped (`on_cpu_stop`): the virtual
    /// count is latched at `cp0_count` (reads don't advance it) and the
    /// compare timer is silenced, so monitor `cpu stop` / debugger stepping
    /// sees a frozen Count and no IP7 firing underneath it. Cleared by
    /// `on_cpu_start`, which re-anchors and re-arms. Ignored under ci_clock
    /// (count follows `hot.cycles` there — debug steps advancing it is the
    /// deterministic behavior CI wants).
    pub(crate) count_paused: bool,
    /// Learned slow-tick Compare delta in hardware counts.
    /// Initialised to 0 (unknown). First delta seen is assumed to be the 100 Hz (slow) tick.
    pub compare_delta_slow: u64,
    /// Learned fast-tick Compare delta in hardware counts.
    /// Initialised to 0 (unknown). Set once we see a delta ~10x smaller than delta_slow.
    pub compare_delta_fast: u64,
    /// Last Compare delta that matched neither learned bucket. If the next
    /// unrecognized delta fuzzy-matches this one, the guest has genuinely
    /// switched its periodic tick — re-seed the slow bucket from it.
    pub(crate) compare_delta_unrecognized: u64,
    /// The Compare-minus-Count delta computed on the *previous* Compare
    /// write, carried forward one write — matches the pre-hptimer
    /// calibration's own `compare_delta_prev` exactly. `infer_count_hz`
    /// classifies this deferred value, not the current write's own fresh
    /// delta: at the time a write happens, its own delta hasn't been
    /// "lived through" yet, whereas `compare_delta_prev` is the interval the
    /// guest just finished actually waiting out. Zero = no previous write
    /// yet (skip classification).
    pub(crate) compare_delta_prev: u64,
    /// hptimer that delivers the Count==Compare interrupt: its callback sets
    /// IP7 in `hot.interrupts` and bumps `fasttick_count`. Re-armed
    /// (remove + add_one_shot) on every Compare or Count write. None until
    /// `set_timer_manager` is wired (unit tests, ci_clock builds).
    pub(crate) timer_mgr: Option<Arc<crate::hptimer::TimerManager>>,
    pub(crate) timer_id: Option<crate::hptimer::TimerId>,
    /// ci_clock: `hot.cycles` value the virtual count is anchored at
    /// (synthetic 10 ns per instruction instead of wall clock, so CI runs
    /// stay deterministic).
    #[cfg(feature = "ci_clock")]
    pub(crate) count_anchor_cycle: u64,
    /// ci_clock: `hot.cycles` value at which IP7 fires (checked in step()'s
    /// preamble instead of an hptimer). u64::MAX = disarmed.
    #[cfg(feature = "ci_clock")]
    pub count_fire_cycle: u64,
    /// Frequency map of CP0 Compare delta values (hardware counts, rounded to nearest 100).
    /// Key = `(delta >> 16) / 100 * 100`, value = number of occurrences. Debug-only
    /// bookkeeping the JIT never touches — kept at the tail, out of the way of the
    /// codegen-visible fields above (see struct doc comment).
    #[cfg(feature = "developer_ip7")]
    pub compare_delta_stats: std::collections::HashMap<u32, u32>,
}

/// A completed real memory access, captured for `jitv2_lockstep`'s
/// load/store verification — see `MipsCore::lockstep_mem`'s doc comment.
#[cfg(feature = "jitv2_lockstep")]
#[derive(Clone, Copy)]
pub struct LockstepMemCapture {
    pub addr: u64,
    pub phys: u64,
    pub value: u64,
}

/// Single nano-TLB entry.
///
/// `va_tag`     — `(va & !0xFFF) | 1`: page-aligned VA with bit 0 as valid sentinel.
///                Zero (default) = invalid. Single compare suffices for both validity and VA match.
/// `pa_encoded` — bits [63:12] = physical page base (PA & !0xFFF),
///                bits  [2:0]  = hardware C-field cache attr (2=Uncached, 3=Cacheable, 5=CacheableCoherent).
///
/// C-field stored directly in bits [2:0] so cache_attr_raw() is a plain mask.
/// Validity is encoded in va_tag bit 0 — no separate valid flag needed.
#[derive(Clone, Copy, Default)]
pub struct NanoTlbEntry {
    pub va_tag:     u64,
    pub pa_encoded: u64,
}

impl NanoTlbEntry {
    #[inline(always)]
    pub fn is_valid(&self) -> bool { self.va_tag != 0 }

    /// Single-comparison hot-path match: checks valid + VA in one 64-bit compare.
    #[inline(always)]
    pub fn matches(&self, va_page: u64) -> bool {
        self.va_tag == va_page | 1
    }

    /// Decode the physical address (page base + page offset).
    #[inline(always)]
    pub fn phys_addr(&self, va: u64) -> u64 {
        (self.pa_encoded & !0xFFF) | (va & 0xFFF)
    }

    /// Decode the CacheAttr (used by TLB layer).
    #[inline(always)]
    pub fn cache_attr(&self) -> crate::mips_exec::CacheAttr {
        use crate::mips_exec::CacheAttr;
        match self.pa_encoded & 0x7 {
            3 => CacheAttr::Cacheable,
            5 => CacheAttr::CacheableCoherent,
            _ => CacheAttr::Uncached,
        }
    }

    /// Return the hardware C-field value (2/3/5) for use in TranslateResult.status bits [2:0].
    /// Bits [2:0] of pa_encoded ARE the C-field — no shift needed.
    #[inline(always)]
    pub fn cache_attr_raw(&self) -> u32 {
        (self.pa_encoded & 0x7) as u32
    }

    /// Fill entry from a successful translation.
    #[inline(always)]
    pub fn fill(&mut self, va: u64, phys_addr: u64, attr: crate::mips_exec::CacheAttr) {
        self.va_tag     = (va & !0xFFF) | 1;
        self.pa_encoded = (phys_addr & !0xFFF) | (attr as u64);
    }

    /// Fill entry from a raw C-field value (2=Uncached, 3=Cacheable, 5=CacheableCoherent).
    /// Used by nanotlb_translate to avoid re-converting through CacheAttr enum.
    #[inline(always)]
    pub fn fill_raw(&mut self, va_page: u64, phys_addr: u64, c_field: u32) {
        self.va_tag     = va_page | 1;
        self.pa_encoded = (phys_addr & !0xFFF) | c_field as u64;
    }

    #[inline(always)]
    pub fn invalidate(&mut self) { self.va_tag = 0; }
}

/// Placeholder for `MipsCore`'s `read*_fn` fields before
/// `MipsExecutor::install_jit_hooks` runs. Panics — compiled code must never
/// be dispatched against a core whose hooks aren't installed yet.
#[cfg(feature = "jitv2")]
unsafe extern "C" fn jit_hooks_not_installed_read(_ctx: *mut core::ffi::c_void, _va: u64) -> u64 {
    panic!("jitv2: read hook called before MipsExecutor::install_jit_hooks");
}
#[cfg(feature = "jitv2")]
unsafe extern "C" fn jit_hooks_not_installed_write8(_ctx: *mut core::ffi::c_void, _va: u64, _v: u64) -> u32 {
    panic!("jitv2: write hook called before MipsExecutor::install_jit_hooks");
}
#[cfg(feature = "jitv2")]
unsafe extern "C" fn jit_hooks_not_installed_write16(_ctx: *mut core::ffi::c_void, _va: u64, _v: u64) -> u32 {
    panic!("jitv2: write hook called before MipsExecutor::install_jit_hooks");
}
#[cfg(feature = "jitv2")]
unsafe extern "C" fn jit_hooks_not_installed_write32(_ctx: *mut core::ffi::c_void, _va: u64, _v: u64) -> u32 {
    panic!("jitv2: write hook called before MipsExecutor::install_jit_hooks");
}
#[cfg(feature = "jitv2")]
unsafe extern "C" fn jit_hooks_not_installed_write64(_ctx: *mut core::ffi::c_void, _va: u64, _v: u64) -> u32 {
    panic!("jitv2: write hook called before MipsExecutor::install_jit_hooks");
}
#[cfg(feature = "jitv2")]
unsafe extern "C" fn jit_hooks_not_installed_write64_masked(_ctx: *mut core::ffi::c_void, _va: u64, _v: u64, _mask: u64) -> u32 {
    panic!("jitv2: write64_masked hook called before MipsExecutor::install_jit_hooks");
}
#[cfg(feature = "jitv2")]
unsafe extern "C" fn jit_hooks_not_installed_exception(_ctx: *mut core::ffi::c_void, _status: u32) -> u32 {
    panic!("jitv2: exception hook called before MipsExecutor::install_jit_hooks");
}
#[cfg(feature = "jitv2")]
unsafe extern "C" fn jit_hooks_not_installed_interp_fallback(_ctx: *mut core::ffi::c_void) -> u32 {
    panic!("jitv2: interp_fallback hook called before MipsExecutor::install_jit_hooks");
}
#[cfg(feature = "jitv2")]
unsafe extern "C" fn jit_hooks_not_installed_kill_entry(_ctx: *mut core::ffi::c_void, _offset: u32) {
    panic!("jitv2: kill_entry hook called before MipsExecutor::install_jit_hooks");
}
/// Unlike the other not-installed sentinels (which panic — reaching them means
/// a real correctness hook was skipped), the dev trace/breakpoint hook is pure
/// diagnostics: a bare codegen unit test that compiles and runs a `jit_fn`
/// without `install_jit_hooks` legitimately has no traceback/breakpoint state
/// to touch, so the sentinel is a harmless no-op (return EXEC_COMPLETE = "no
/// breakpoint") rather than an abort.
#[cfg(all(feature = "jitv2", feature = "developer"))]
unsafe extern "C" fn jit_hooks_not_installed_dev_trace_bp(_ctx: *mut core::ffi::c_void, _pc: u64, _raw: u32, _origin: u32) -> u32 {
    crate::mips_exec::EXEC_COMPLETE
}
/// No-op (not a panic) like the dev-hook sentinel: lockstep is pure
/// verification, and a bare codegen unit test that runs a `jit_fn` without
/// `install_jit_hooks` has no executor/interpreter to compare against — the
/// compiled code just runs normally, unverified, exactly as a plain jitv2
/// build would. (A real lockstep run always installs hooks.)
#[cfg(feature = "jitv2_lockstep")]
unsafe extern "C" fn jit_hooks_not_installed_lockstep_step(_ctx: *mut core::ffi::c_void, _pc: u64, _raw: u32, _bd: u32) {}
#[cfg(feature = "jitv2_lockstep")]
unsafe extern "C" fn jit_hooks_not_installed_lockstep_compare(_ctx: *mut core::ffi::c_void) -> u32 { crate::mips_exec::EXEC_COMPLETE }
#[cfg(feature = "jitv2")]
unsafe extern "C" fn jit_hooks_not_installed_fpu_get_status(_ctx: *mut core::ffi::c_void) -> u32 {
    panic!("jitv2: fpu_get_status hook called before MipsExecutor::install_jit_hooks");
}
#[cfg(feature = "jitv2")]
unsafe extern "C" fn jit_hooks_not_installed_fpu_clear_status(_ctx: *mut core::ffi::c_void) {
    panic!("jitv2: fpu_clear_status hook called before MipsExecutor::install_jit_hooks");
}
#[cfg(feature = "jitv2")]
unsafe extern "C" fn jit_hooks_not_installed_fpu_set_mode(_ctx: *mut core::ffi::c_void, _rm: u32) {
    panic!("jitv2: fpu_set_mode hook called before MipsExecutor::install_jit_hooks");
}

// SAFETY: The raw pointer in status_changed_cb is only accessed from the CPU thread.
unsafe impl Send for MipsCore {}

impl Drop for MipsCore {
    fn drop(&mut self) {
        // The armed compare timer's callback holds a raw pointer into this
        // core (`TimerIrqPtr`) — take it out of the manager before the
        // pointee goes away.
        self.disarm_compare_timer();
    }
}

impl MipsCore {
    /// Create a new MIPS core with reset state
    pub fn new() -> Self {
        let mut core = Self {
            hot: Hot::default(),
            cp0_count: 0,
            cp0_compare: 0,
            pc: 0,
            in_delay_slot: false,
            delay_slot_target: 0,
            gpr: [0; 32],
            hi: 0,
            lo: 0,
            fpr: [0; 32],
            fpu_fir: 0,
            fpu_fccr: 0,
            fpu_fexr: 0,
            fpu_fenr: 0,
            fpu_fcsr: 0,
            nanotlb: [NanoTlbEntry::default(); 3],
            status_changed_cb: None,
            // Placeholders — overwritten immediately by
            // MipsExecutor::install_jit_hooks (same pattern as translate_fn's
            // own placeholder init below in MipsExecutor::new). Panic if
            // ever actually called: that would mean compiled code ran before
            // hooks were installed, which must not happen for any executor
            // that has jitv2 compiled units live.
            #[cfg(feature = "jitv2")]
            jit_ctx: std::ptr::null_mut(),
            #[cfg(feature = "jitv2")]
            read8_fn: jit_hooks_not_installed_read,
            #[cfg(feature = "jitv2")]
            read16_fn: jit_hooks_not_installed_read,
            #[cfg(feature = "jitv2")]
            read32_fn: jit_hooks_not_installed_read,
            #[cfg(feature = "jitv2")]
            read64_fn: jit_hooks_not_installed_read,
            #[cfg(feature = "jitv2")]
            write8_fn: jit_hooks_not_installed_write8,
            #[cfg(feature = "jitv2")]
            write16_fn: jit_hooks_not_installed_write16,
            #[cfg(feature = "jitv2")]
            write32_fn: jit_hooks_not_installed_write32,
            #[cfg(feature = "jitv2")]
            write64_fn: jit_hooks_not_installed_write64,
            #[cfg(feature = "jitv2")]
            write64_masked_fn: jit_hooks_not_installed_write64_masked,
            #[cfg(feature = "jitv2")]
            handle_exception_fn: jit_hooks_not_installed_exception,
            #[cfg(feature = "jitv2")]
            interp_fallback_fn: jit_hooks_not_installed_interp_fallback,
            #[cfg(feature = "jitv2")]
            kill_entry_fn: jit_hooks_not_installed_kill_entry,
            #[cfg(all(feature = "jitv2", feature = "developer"))]
            dev_trace_bp_fn: jit_hooks_not_installed_dev_trace_bp,
            #[cfg(feature = "jitv2")]
            jit_mem_exc: 0,
            #[cfg(feature = "jitv2")]
            fpu_get_status_fn: jit_hooks_not_installed_fpu_get_status,
            #[cfg(feature = "jitv2")]
            fpu_clear_status_fn: jit_hooks_not_installed_fpu_clear_status,
            #[cfg(feature = "jitv2")]
            fpu_set_mode_fn: jit_hooks_not_installed_fpu_set_mode,
            #[cfg(feature = "jitv2_lockstep")]
            lockstep_step_fn: jit_hooks_not_installed_lockstep_step,
            #[cfg(feature = "jitv2_lockstep")]
            lockstep_compare_fn: jit_hooks_not_installed_lockstep_compare,
            #[cfg(feature = "jitv2_lockstep")]
            lockstep_mem: None,
            #[cfg(feature = "jitv2")]
            jit_trigger: false,
            cp0_index: 0,
            cp0_random: 0,
            cp0_entrylo0: 0,
            cp0_entrylo1: 0,
            cp0_context: 0,
            cp0_pagemask: 0,
            cp0_wired: 0,
            cp0_badvaddr: 0,
            cp0_entryhi: 0,
            cp0_status: 0,
            cp0_cause: 0,
            cp0_epc: 0,
            cp0_prid: 0,
            cp0_config: 0x8000, // Default to Big Endian (Bit 15)
            cp0_lladdr: 0,
            cp0_watchlo: 0,
            cp0_watchhi: 0,
            cp0_xcontext: 0,
            cp0_ecc: 0,
            cp0_cacheerr: 0,
            cp0_taglo: 0,
            cp0_taghi: 0,
            cp0_errorepc: 0,
            tlb_entries: 48,
            cp0_random_cycle: 0,
            fasttick_count: Arc::new(AtomicU64::new(0)),
            running: false,
            halted: false,
            count_hz_atomic: Arc::new(AtomicU64::new(DEFAULT_COUNT_HZ)),
            count_anchor_instant: std::time::Instant::now(),
            count_read_cycle: 0,
            count_hz: DEFAULT_COUNT_HZ,
            count_hz_fixed: None,
            count_paused: false,
            compare_delta_slow: 0,
            compare_delta_fast: 0,
            compare_delta_unrecognized: 0,
            compare_delta_prev: 0,
            timer_mgr: None,
            timer_id: None,
            #[cfg(feature = "ci_clock")]
            count_anchor_cycle: 0,
            #[cfg(feature = "ci_clock")]
            count_fire_cycle: u64::MAX,
            #[cfg(feature = "developer_ip7")]
            compare_delta_stats: std::collections::HashMap::new(),
        };
        core.reset_registers(false);
        core
    }

    fn reset_registers(&mut self, soft: bool) {
        if !soft {
            self.gpr.fill(0);
            self.hi = 0;
            self.lo = 0;
            self.fpr.fill(0);
            
            // CP0 registers
            self.cp0_index = 0;
            self.cp0_random = 0;
            self.cp0_entrylo0 = 0;
            self.cp0_entrylo1 = 0;
            self.cp0_context = 0;
            self.cp0_pagemask = 0;
            self.cp0_badvaddr = 0;
            self.cp0_count = 0;
            self.cp0_entryhi = 0;
            self.cp0_compare = 0;
            self.hot.cycles = 0;
            self.count_read_cycle = 0;
            self.count_anchor_instant = std::time::Instant::now();
            #[cfg(feature = "ci_clock")]
            {
                self.count_anchor_cycle = 0;
                self.count_fire_cycle = u64::MAX;
            }
            // compare_delta_slow/fast/unrecognized/prev deliberately NOT
            // cleared here, matching the pre-hptimer calibration's own
            // behavior: they're cheap-to-recheck learned state about the
            // guest's tick rate, not per-boot state, and get re-verified (or
            // replaced) on the next real Compare write regardless.
            self.disarm_compare_timer();
            self.cp0_random = self.tlb_entries - 1;
            self.cp0_random_cycle = 0;
            #[cfg(not(feature = "r5k"))]
            { self.cp0_prid = 0x00000440; } // R4400, imp=0x04, majrev=4, minrev=0
            #[cfg(feature = "r5k")]
            { self.cp0_prid = 0x00002321; } // R5000, imp=0x23, rev=2.1
            self.cp0_watchlo = 0;
            self.cp0_watchhi = 0;
            self.cp0_xcontext = 0;
            self.cp0_ecc = 0;
            self.cp0_cacheerr = 0;
            self.cp0_taglo = 0;
            self.cp0_taghi = 0;

            // CP1 registers
            #[cfg(not(feature = "r5k"))]
            { self.fpu_fir = 0x00000500; } // R4000 FPU: imp=0x05, rev=0
            #[cfg(feature = "r5k")]
            { self.fpu_fir = 0x00002300; } // R5000 FPU: imp=0x23, rev=0
            self.fpu_fccr = 0;
            self.fpu_fexr = 0;
            self.fpu_fenr = 0;
            self.fpu_fcsr = 0;
            // Sync host FPU rounding mode to match (RM=0=RN). Without this the
            // host thread's rounding mode is left at whatever it was before
            // reset — usually RN by platform default, but not guaranteed, and
            // it would otherwise silently desync from the software-tracked
            // fpu_fcsr until the next CTC1 write.
            crate::platform::set_fpu_mode(0);
        }

        self.pc = 0xFFFFFFFF_BFC00000; // Reset vector in KSEG1 (uncached, sign-extended)
        self.cp0_wired = 0;
        self.cp0_status = STATUS_BEV | STATUS_ERL; // BEV=1, ERL=1 (boot exception vectors)
        if soft {
            self.cp0_status |= STATUS_SR;
        }
        self.cp0_cause = 0;
        self.cp0_epc = 0;
        self.cp0_errorepc = 0;

        self.running = false;
        self.halted = false;
    }

    /// Reset the CPU to initial state
    pub fn reset(&mut self, soft: bool) {
        self.reset_registers(soft);
        self.hot.interrupts.store(0, Ordering::SeqCst);
    }

    /// Read a GPR by index. gpr[0] is always kept at zero.
    #[inline(always)]
    pub fn read_gpr(&self, reg: u32) -> u64 {
        unsafe { *self.gpr.get_unchecked(reg as usize) }
    }

    /// Write a GPR by index. Unconditionally re-zeros gpr[0] to avoid a branch.
    #[inline(always)]
    pub fn write_gpr(&mut self, reg: u32, value: u64) {
        unsafe { *self.gpr.get_unchecked_mut(reg as usize) = value; }
        self.gpr[0] = 0;
    }

    /// Update Random register based on current cycle count
    pub fn update_random(&mut self) {
        let current_cycles = self.hot.cycles;
        let wired = self.cp0_wired;
        let max_entry = self.tlb_entries - 1;

        if wired > max_entry {
            self.cp0_random = max_entry;
            self.cp0_random_cycle = current_cycles;
            return;
        }

        let range = self.tlb_entries - wired;
        if range == 0 {
             self.cp0_random = max_entry;
             self.cp0_random_cycle = current_cycles;
             return;
        }

        let delta = current_cycles.wrapping_sub(self.cp0_random_cycle);
        if delta > 0 {
            // Random decrements from max_entry down to wired, then wraps to max_entry
            // Normalize current value to 0..range-1
            let current_val = if self.cp0_random >= wired { self.cp0_random } else { max_entry };
            let current_offset = current_val - wired;

            let step = (delta % (range as u64)) as u32;

            let new_offset = (current_offset + range - step) % range;

            self.cp0_random = new_offset + wired;
            self.cp0_random_cycle = current_cycles;
        }
    }

    /// Read CP0 register by index
    pub fn read_cp0(&mut self, reg: u32) -> u64 {
        match reg {
            0 => self.cp0_index as u64,
            1 => {
                self.update_random();
                self.cp0_random as u64
            }
            2 => self.cp0_entrylo0 & 0x3FFFFFFF, // PFN is 24 bits (29:6), flags in lower bits
            3 => self.cp0_entrylo1 & 0x3FFFFFFF, // PFN is 24 bits (29:6), flags in lower bits
            4 => self.cp0_context,
            5 => self.cp0_pagemask & 0x01FFE000, // PageMask: only bits 24:13 are valid
            6 => self.cp0_wired as u64,
            8 => self.cp0_badvaddr,
            9 => self.count_now() as u64,
            10 => self.cp0_entryhi,
            11 => self.cp0_compare,
            12 => self.cp0_status as u64,
            13 => self.cp0_cause as u64,
            14 => self.cp0_epc,
            15 => self.cp0_prid as u64,
            16 => self.cp0_config as u64,
            17 => self.cp0_lladdr as u64,
            18 => self.cp0_watchlo as u64,
            19 => self.cp0_watchhi as u64,
            20 => self.cp0_xcontext,
            26 => self.cp0_ecc as u64,
            27 => self.cp0_cacheerr as u64,
            28 => self.cp0_taglo as u64,
            29 => self.cp0_taghi as u64,
            30 => self.cp0_errorepc,
            _ => 0, // Unimplemented registers read as 0
        }
    }

    /// Read CP0 register by index (non-mutating, for debugger use — skips Random update)
    pub fn read_cp0_debug(&self, reg: u32) -> u64 {
        match reg {
            0 => self.cp0_index as u64,
            1 => self.cp0_random as u64,
            2 => self.cp0_entrylo0 & 0x3FFFFFFF,
            3 => self.cp0_entrylo1 & 0x3FFFFFFF,
            4 => self.cp0_context,
            5 => self.cp0_pagemask & 0x01FFE000,
            6 => self.cp0_wired as u64,
            8 => self.cp0_badvaddr,
            9 => self.count_peek() as u64,
            10 => self.cp0_entryhi,
            11 => self.cp0_compare,
            12 => self.cp0_status as u64,
            13 => self.cp0_cause as u64,
            14 => self.cp0_epc,
            15 => self.cp0_prid as u64,
            16 => self.cp0_config as u64,
            17 => self.cp0_lladdr as u64,
            18 => self.cp0_watchlo as u64,
            19 => self.cp0_watchhi as u64,
            20 => self.cp0_xcontext,
            26 => self.cp0_ecc as u64,
            27 => self.cp0_cacheerr as u64,
            28 => self.cp0_taglo as u64,
            29 => self.cp0_taghi as u64,
            30 => self.cp0_errorepc,
            _ => 0,
        }
    }

    /// Materialize the virtual CP0 Count: advance `cp0_count` by the whole
    /// hardware ticks elapsed since the anchor at `count_hz`, move the anchor
    /// forward by exactly the duration those ticks consumed (the sub-tick
    /// remainder is preserved, so frequent reads don't drift the count slow),
    /// and return the current 32-bit count.
    ///
    /// Memoized on `hot.cycles`: called twice within the same instruction it
    /// returns the identical value (jitv2_lockstep runs the same instruction
    /// on both engines and compares the resulting GPR state).
    pub fn count_now(&mut self) -> u32 {
        #[cfg(not(feature = "ci_clock"))]
        if self.count_paused {
            return self.cp0_count as u32;
        }
        let cycles = self.hot.cycles;
        if cycles == self.count_read_cycle {
            return self.cp0_count as u32;
        }
        #[cfg(feature = "ci_clock")]
        let elapsed_ns = cycles.wrapping_sub(self.count_anchor_cycle).saturating_mul(NS_PER_GUEST_CYCLE);
        #[cfg(not(feature = "ci_clock"))]
        let elapsed_ns = self.count_anchor_instant.elapsed().as_nanos() as u64;
        let ticks = ((elapsed_ns as u128 * self.count_hz as u128) / 1_000_000_000) as u64;
        if ticks != 0 {
            let consumed_ns = ((ticks as u128 * 1_000_000_000) / self.count_hz as u128) as u64;
            #[cfg(feature = "ci_clock")]
            { self.count_anchor_cycle = self.count_anchor_cycle.wrapping_add(consumed_ns / NS_PER_GUEST_CYCLE); }
            #[cfg(not(feature = "ci_clock"))]
            { self.count_anchor_instant += std::time::Duration::from_nanos(consumed_ns); }
            self.cp0_count = (self.cp0_count as u32).wrapping_add(ticks as u32) as u64;
        }
        self.count_read_cycle = cycles;
        self.cp0_count as u32
    }

    /// Non-mutating variant of `count_now` for debugger/monitor reads: same
    /// computation, but neither the anchor nor the memo state moves.
    pub fn count_peek(&self) -> u32 {
        #[cfg(not(feature = "ci_clock"))]
        if self.count_paused {
            return self.cp0_count as u32;
        }
        #[cfg(feature = "ci_clock")]
        let elapsed_ns = self.hot.cycles.wrapping_sub(self.count_anchor_cycle).saturating_mul(NS_PER_GUEST_CYCLE);
        #[cfg(not(feature = "ci_clock"))]
        let elapsed_ns = self.count_anchor_instant.elapsed().as_nanos() as u64;
        let ticks = ((elapsed_ns as u128 * self.count_hz as u128) / 1_000_000_000) as u64;
        (self.cp0_count as u32).wrapping_add(ticks as u32)
    }

    /// Infer `count_hz` from a Compare delta (hardware counts to the next
    /// interrupt) by pattern only — never from real elapsed time. Real time
    /// can't be used here: `count_hz` determines *when our own hptimer
    /// fires* (`schedule_compare_timer`), which is what raises IP7, which is
    /// what prompts the guest's clock handler to write the next Compare —
    /// so measuring `count_hz` from the real interval between two Compare
    /// writes would be measuring the effect of our own previous `count_hz`
    /// guess, a circular dependency that can compound instead of converge
    /// (a bad first guess skews the fire time, which skews the "measured"
    /// interval, which re-skews count_hz). Same two-bucket model as the
    /// pre-hptimer `bin_compare_delta`: a recognized slow tick is *assumed*
    /// to mean 100 Hz, a recognized fast tick (~slow/10) 1 kHz, both with
    /// ±5% fuzzy matching — `d` hw-counts spanning the bucket's assumed
    /// period directly gives `count_hz = d / period`. An unrecognized delta
    /// leaves `count_hz` alone unless it repeats (fuzzy-equal twice in a
    /// row), which means the guest genuinely retuned its periodic tick, so
    /// the slow bucket is re-seeded from it.
    fn infer_count_hz(&mut self, d: u64) {
        // A fixed clock (`[clock] fixed_mhz`) opts out of this entirely:
        // count_hz was pinned at construction and must never move, so skip
        // the pattern-matching (and its `compare_delta_*` bookkeeping) below.
        if self.count_hz_fixed.is_some() {
            return;
        }
        // Bound `d` against the full plausible range of real MIPS Count
        // clocks (Count = CPU_clock/2), roughly 10 MHz to 300 MHz for any
        // guest this emulator targets: a 1 kHz tick spans at least
        // 10,000,000/1000 = 10,000 counts at the slowest plausible clock,
        // and a 100 Hz tick spans at most 300,000,000/100 = 3,000,000
        // counts at the fastest. PROM runs short one-shot Compare writes
        // before IRIX's own clock handler ever starts (diagnostic delays,
        // early watchdogs, e.g. a live 58-count delta observed at boot),
        // and one-shots can also be implausibly *large* (multi-second
        // watchdog timeouts) — trusting either extreme anywhere this
        // function can write into compare_delta_slow/fast poisons every
        // later classification (everything else gets fuzzy-matched against
        // that bad baseline). This bound used to guard only the very first
        // call (`compare_delta_slow == 0 && compare_delta_fast == 0`), on
        // the theory that once a bucket is seeded, further bad deltas just
        // fall through to compare_delta_unrecognized — but that reasoning
        // missed that a *seeded-but-tiny* bucket keeps being read, not just
        // written: `compare_delta_slow / 10` (the "recognize the fast
        // tick" branch below) can itself land below the floor if
        // compare_delta_slow is small, and `fuzzy_eq` against a tiny value
        // degenerates (its ±5% threshold rounds to 0, making it an
        // exact-match test at low magnitudes) — a live run latched
        // compare_delta_fast=1 this way. Apply the bound unconditionally,
        // every call, not just the first.
        const MIN_PLAUSIBLE_TICK_DELTA: u64 = 10_000;
        const MAX_PLAUSIBLE_TICK_DELTA: u64 = 3_000_000;
        if d < MIN_PLAUSIBLE_TICK_DELTA || d > MAX_PLAUSIBLE_TICK_DELTA {
            return;
        }

        // ±5% fuzzy equality.
        let fuzzy_eq = |a: u64, b: u64| -> bool {
            let threshold = a.max(b) * 5 / 100;
            a.abs_diff(b) <= threshold
        };

        let tick_ns: u64 = if self.compare_delta_slow == 0 && self.compare_delta_fast == 0 {
            // First delta ever — no learned bucket to compare against, so
            // classify by magnitude against the *default* 33 MHz Count rate
            // instead of blindly assuming "first delta = slow (100 Hz)".
            // That assumption is wrong whenever the guest's first Compare
            // write we observe is already on the fast (1 kHz, ~33,000
            // counts at 33 MHz) cadence — e.g. IRIX using only its 1 kHz
            // scheduler tick this early in boot, nothing at 100 Hz yet —
            // which mislabels a fast-tick delta as the slow bucket and
            // computes count_hz an order of magnitude too low (a live
            // ~32,900-count delta was seen classified as "100 Hz" giving
            // ~3.29 MHz, when it's actually the 1 kHz tick giving the
            // correct ~32.9 MHz). Midpoint between the two expected
            // magnitudes (100 Hz≈330,000 / 1 kHz≈33,000 counts at 33 MHz)
            // is ~180,000 — anything below that is presumed fast, at/above
            // presumed slow.
            const DEFAULT_FAST_SLOW_MIDPOINT: u64 = DEFAULT_COUNT_HZ / 1_000 * 5; // ~180,000 (5.5ms)
            if d < DEFAULT_FAST_SLOW_MIDPOINT {
                self.compare_delta_fast = d;
                1_000_000
            } else {
                self.compare_delta_slow = d;
                10_000_000
            }
        } else if self.compare_delta_slow != 0 && fuzzy_eq(d, self.compare_delta_slow) {
            10_000_000
        } else if self.compare_delta_fast != 0 && fuzzy_eq(d, self.compare_delta_fast) {
            1_000_000
        } else if self.compare_delta_slow >= 10 && fuzzy_eq(d, self.compare_delta_slow / 10) {
            // ~10x smaller than slow → this is the fast tick; learn it.
            self.compare_delta_fast = d;
            1_000_000
        } else if self.compare_delta_fast != 0 && fuzzy_eq(d, self.compare_delta_fast.saturating_mul(10)) {
            // ~10x bigger than the already-known fast tick → this is the
            // slow tick; learn it. Symmetric case to the branch above, for
            // when the fast bucket was seeded first (see the first-delta
            // classification comment).
            self.compare_delta_slow = d;
            10_000_000
        } else if self.compare_delta_unrecognized != 0 && fuzzy_eq(d, self.compare_delta_unrecognized) {
            // Second consecutive unrecognized delta of the same size: the
            // guest switched its periodic tick. Re-seed — but classify by
            // magnitude against the *other* still-known bucket (or the
            // 33 MHz default midpoint if neither is known) rather than
            // unconditionally assuming "re-seed = slow". Blindly stamping
            // this into compare_delta_slow regardless of size was the same
            // bug as the old first-delta seeding: a repeated fast-tick-sized
            // one-shot (e.g. two similar deltas around a reset) got
            // mislabeled as a retuned 100 Hz tick, wiping out a correctly
            // learned compare_delta_fast and dragging count_hz an order of
            // magnitude off (a live 32,950-count repeat did exactly this,
            // landing count_hz at 3.29 MHz instead of ~33 MHz).
            let other = if self.compare_delta_slow != 0 { self.compare_delta_slow }
                        else if self.compare_delta_fast != 0 { self.compare_delta_fast * 10 }
                        else { DEFAULT_COUNT_HZ / 1_000 * 5 }; // ~180,000, same midpoint as first-delta seeding
            if d < other {
                self.compare_delta_fast = d;
                self.compare_delta_unrecognized = 0;
                1_000_000
            } else {
                self.compare_delta_slow = d;
                self.compare_delta_fast = 0;
                self.compare_delta_unrecognized = 0;
                10_000_000
            }
        } else {
            // One-shot or first sighting of a new interval — remember it and
            // keep the current frequency.
            self.compare_delta_unrecognized = d;
            return;
        };
        self.compare_delta_unrecognized = 0;
        // Recognized periodic tick: `d` hardware counts span `tick_ns` real ns.
        let hz = d.saturating_mul(1_000_000_000) / tick_ns;
        if hz != 0 {
            self.count_hz = hz;
            self.count_hz_atomic.store(hz, Ordering::Relaxed);
        }
    }

    /// (Re-)arm the Count==Compare interrupt for the current
    /// `cp0_count`/`cp0_compare` values. Caller must have just materialized
    /// the count (`count_now`) so the delta is measured from *now*.
    ///
    /// Real-time builds: removes the previous hptimer one-shot (a stale
    /// generation in the manager's FIFO is skipped, so this is race-free)
    /// and arms a fresh one whose callback ORs IP7 into `hot.interrupts` and
    /// bumps `fasttick_count`. ci_clock builds: computes the deterministic
    /// `hot.cycles` value the interrupt fires at instead — step()'s preamble
    /// checks it.
    fn schedule_compare_timer(&mut self) {
        // Real MIPS Count==Compare semantics: IP7 fires when the free-running
        // 32-bit Count becomes numerically equal to Compare. If Count is
        // already past Compare at arm time (Compare written "in the past",
        // or real elapsed time crept Count forward since the last read),
        // that equality genuinely does not happen again until Count wraps
        // through 0 and climbs back up to Compare — there is no "fire
        // immediately because we're past it" on real hardware, so don't
        // synthesize one here either. Unsigned wrapping_sub naturally gives
        // that full-wrap distance for any "already passed" case, and the
        // explicit 0 case (Compare == Count exactly) is the same "next
        // match is a full wrap away" situation, not "now".
        let delta = match (self.cp0_compare as u32).wrapping_sub(self.cp0_count as u32) as u64 {
            0 => 1u64 << 32,
            d => d,
        };
        let ns = ((delta as u128 * 1_000_000_000) / self.count_hz as u128) as u64;
        #[cfg(feature = "ci_clock")]
        {
            self.count_fire_cycle = self.hot.cycles.saturating_add(ns / NS_PER_GUEST_CYCLE);
        }
        #[cfg(not(feature = "ci_clock"))]
        {
            if self.count_paused {
                // CPU is stopped: stay disarmed; on_cpu_start re-arms from
                // the then-current count/compare.
                return;
            }
            let Some(tm) = self.timer_mgr.as_ref() else { return };
            if let Some(id) = self.timer_id.take() {
                tm.remove(id);
            }
            let irq = TimerIrqPtr(&self.hot.interrupts as *const AtomicU64);
            let fasttick = self.fasttick_count.clone();
            // After firing, hardware would next match Compare again after a
            // full 32-bit Count wrap — self-reschedule that far out. In
            // practice the guest's interrupt handler writes Compare long
            // before then, which re-arms through this function afresh.
            let wrap = std::time::Duration::from_nanos(
                (((1u128 << 32) * 1_000_000_000) / self.count_hz as u128) as u64,
            );
            self.timer_id = Some(tm.add_one_shot(
                std::time::Duration::from_nanos(ns),
                (irq, fasttick, wrap),
                |(irq, fasttick, wrap)| {
                    // SAFETY: points into the MipsCore owned by the executor's
                    // top-level Arc<Mutex<..>>, which outlives the armed timer
                    // (Drop for MipsCore removes it). Same contract as
                    // Ioc::set_interrupts's stored pointer.
                    unsafe { &*irq.0 }.fetch_or(CAUSE_IP7 as u64, Ordering::SeqCst);
                    fasttick.fetch_add(1, Ordering::Relaxed);
                    crate::hptimer::TimerReturn::RescheduleOneShot(*wrap)
                },
            ));
        }
    }

    /// Pin `count_hz` at `hz` forever: `infer_count_hz` becomes a no-op (see
    /// its own doc comment) and this value is used as-is for both
    /// `count_now`/`count_peek` materialization and compare-timer scheduling.
    /// Must be called before the core starts executing (construction time) —
    /// it does not re-anchor or re-arm an already-running timer.
    pub fn set_fixed_clock_hz(&mut self, hz: u64) {
        self.count_hz_fixed = Some(hz);
        self.count_hz = hz;
        self.count_hz_atomic.store(hz, Ordering::Relaxed);
    }

    /// Cancel any armed Count==Compare interrupt source.
    fn disarm_compare_timer(&mut self) {
        #[cfg(feature = "ci_clock")]
        { self.count_fire_cycle = u64::MAX; }
        if let (Some(tm), Some(id)) = (self.timer_mgr.as_ref(), self.timer_id.take()) {
            tm.remove(id);
        }
    }

    /// Wire the machine's hptimer manager in. Must be called after the core
    /// has reached its final address (inside the executor's `Arc<Mutex<..>>`)
    /// — the armed timer callback keeps a raw pointer to `hot.interrupts`.
    pub fn set_timer_manager(&mut self, tm: Arc<crate::hptimer::TimerManager>) {
        self.timer_mgr = Some(tm);
    }

    /// Re-anchor the virtual count at the current `cp0_count` value (after a
    /// Count write or a snapshot restore that set it as a raw field) and
    /// re-arm the compare timer against it. While the CPU is stopped
    /// (`count_paused`) the timer stays silenced — `schedule_compare_timer`
    /// itself refuses to arm — and `on_cpu_start` re-arms.
    pub fn reanchor_count_and_reschedule(&mut self) {
        self.count_read_cycle = self.hot.cycles;
        #[cfg(feature = "ci_clock")]
        { self.count_anchor_cycle = self.hot.cycles; }
        #[cfg(not(feature = "ci_clock"))]
        { self.count_anchor_instant = std::time::Instant::now(); }
        self.schedule_compare_timer();
    }

    /// CPU thread is stopping: latch the virtual count at its current value
    /// and silence the compare timer, so a stopped CPU (monitor `cpu stop`,
    /// debugger stepping) sees a frozen Count and never gets IP7 raised
    /// underneath it (use the monitor's `ip7` command to inject one
    /// manually). Idempotent. No-op semantics under ci_clock (count follows
    /// `hot.cycles` there, which stops advancing on its own).
    pub fn on_cpu_stop(&mut self) {
        if self.count_paused {
            return;
        }
        #[cfg(not(feature = "ci_clock"))]
        {
            // Materialize directly, bypassing count_now's cycles memo —
            // cycles may not have moved since the last read, but real time
            // has.
            let elapsed_ns = self.count_anchor_instant.elapsed().as_nanos() as u64;
            let ticks = ((elapsed_ns as u128 * self.count_hz as u128) / 1_000_000_000) as u64;
            self.cp0_count = (self.cp0_count as u32).wrapping_add(ticks as u32) as u64;
        }
        self.count_read_cycle = self.hot.cycles;
        self.count_paused = true;
        if let (Some(tm), Some(id)) = (self.timer_mgr.as_ref(), self.timer_id) {
            tm.disable(id);
        }
    }

    /// CPU thread is starting: resume counting from the latched value (the
    /// anchor restarts at now, so the stopped interval simply never
    /// happened as far as the guest's Count is concerned) and re-arm the
    /// compare timer against it.
    pub fn on_cpu_start(&mut self) {
        self.count_paused = false;
        self.reanchor_count_and_reschedule();
    }

    /// Write CP0 register by index.
    /// When reg 12 (Status) is written, invokes `status_changed_cb` with (old, new).
    pub fn write_cp0(&mut self, reg: u32, value: u64) {
        match reg {
            0 => self.cp0_index = value as u32,
            1 => { /* Random is read-only */ }
            2 => self.cp0_entrylo0 = value & 0x3FFFFFFF, // PFN is 24 bits (29:6), flags in lower bits
            3 => self.cp0_entrylo1 = value & 0x3FFFFFFF, // PFN is 24 bits (29:6), flags in lower bits
            4 => {
                //eprintln!("MTC0 Context = {:#018x} (PTEBase={:#018x})", value, value & 0xFFFFFFFF_FF800000);
                self.cp0_context = value;
            }
            5 => self.cp0_pagemask = value & 0x01FFE000, // PageMask: only bits 24:13 are writable
            6 => {
                self.cp0_wired = value as u32;
                self.cp0_random = self.tlb_entries - 1;
                self.cp0_random_cycle = self.hot.cycles;
            }
            8 => { /* BadVAddr is read-only */ }
            9 => {
                // Re-anchor the virtual count at the written value and re-arm
                // the compare timer against the moved count.
                self.cp0_count = value as u32 as u64;
                self.reanchor_count_and_reschedule();
            }
            10 => { // always use 64bit mask because the entries need to be valid in 64 bit mode even when they were set from 32 bit mode
                self.cp0_entryhi = value & 0xC000_00FF_FFFF_E0FF;
            },
            11 => {
                // Materialize the count *first* so both new_delta below and
                // schedule_compare_timer arm/classify against the true
                // current Count/Compare relationship, then arm the
                // interrupt.
                self.count_now();
                self.cp0_compare = value as u32 as u64;
                // Writing Compare acknowledges the timer interrupt: clear
                // IP7 both in Cause and in the shared pending word (the
                // step() preamble mirrors the pending word into Cause, so
                // leaving the atomic bit set would immediately re-raise it).
                self.cp0_cause &= !CAUSE_IP7;
                self.hot.interrupts.fetch_and(!(CAUSE_IP7 as u64), Ordering::SeqCst);

                // new_delta is this write's own Compare-minus-Count — what
                // the guest is asking to wait *starting now*, not yet lived
                // through. Classify compare_delta_prev instead: the delta
                // programmed on the *previous* write, which the guest just
                // finished actually waiting out (the interval that just
                // elapsed really did contain that many hw-counts, since IP7
                // firing is what prompted this write) — matches the
                // pre-hptimer calibration's own `compare_delta_prev`/
                // `bin_compare_delta(prev_delta)` shape exactly. Classifying
                // new_delta instead would be judging an interval before it's
                // happened. Pure pattern matching on magnitude in
                // infer_count_hz, never real elapsed time — count_hz decides
                // when our own hptimer fires (schedule_compare_timer below),
                // so measuring real time between Compare writes would be
                // circular with our own previous count_hz guess.
                let new_delta = (self.cp0_compare as u32).wrapping_sub(self.cp0_count as u32);
                #[cfg(feature = "developer_ip7")]
                {
                    let bucket = (new_delta / 100) * 100;
                    *self.compare_delta_stats.entry(bucket).or_insert(0) += 1;
                }
                if self.compare_delta_prev != 0 {
                    self.infer_count_hz(self.compare_delta_prev);
                }
                // Top bit set means this write's own Compare is already
                // behind Count (written "in the past", or Count/Compare are
                // unrelated one-shots the guest isn't using periodically) —
                // don't carry a meaningless negative interval forward as
                // the next write's classification input; treat it like "no
                // previous write yet" instead of poisoning that round too.
                self.compare_delta_prev = if new_delta >> 31 == 0 { new_delta as u64 } else { 0 };

                self.schedule_compare_timer();
            }
            12 => {
                let old = self.cp0_status;
                self.cp0_status = value as u32;
                if let Some((cb, ctx)) = self.status_changed_cb {
                    cb(ctx, old, self.cp0_status);
                }
            }
            13 => {
                // Cause register: Only IP0 and IP1 are writable software interrupts
                let mask = CAUSE_IP0 | CAUSE_IP1;
                self.cp0_cause = (self.cp0_cause & !mask) | ((value as u32) & mask);
            }
            14 => self.cp0_epc = value,
            15 => { /* PRId is read-only */ }
            16 => {
                // Bits 5:0 always writable (K0, CU, DB, IB).
                // Triton: bit 12 (CONFIG_SE) also writable.
                #[cfg(not(feature = "r5ksc_triton"))]
                let mask = 0x3F;
                #[cfg(feature = "r5ksc_triton")]
                let mask = 0x103F; // adds bit 12 (SE)
                let old = self.cp0_config;
                let new = (old & !mask) | ((value as u32) & mask);
                if new != old {
                    let ib = (new >> 5) & 1;
                    let db = (new >> 4) & 1;
                    let ic_line = if ib == 1 { 32 } else { 16 };
                    let dc_line = if db == 1 { 32 } else { 16 };
                    //eprintln!("CP0 Config written: {:#010x} -> {:#010x}  (L1I-line={}B L1D-line={}B K0={})",
                    //    old, new, ic_line, dc_line, new & 7);
                }
                self.cp0_config = new;
            }
            17 => self.cp0_lladdr = value as u32,
            18 => self.cp0_watchlo = value as u32,
            19 => self.cp0_watchhi = value as u32,
            20 => self.cp0_xcontext = value,
            26 => self.cp0_ecc = value as u32,
            27 => self.cp0_cacheerr = value as u32,
            28 => self.cp0_taglo = value as u32,
            29 => self.cp0_taghi = value as u32,
            30 => self.cp0_errorepc = value,
            _ => {} // Writes to unimplemented registers are ignored
        }
    }

    /// Invalidate all nano-TLB entries.
    /// Must be called on any TLB write, CP0 Status change, or ASID change.
    #[inline]
    pub fn nanotlb_invalidate(&mut self) {
        self.nanotlb[0].invalidate();
        self.nanotlb[1].invalidate();
        self.nanotlb[2].invalidate();
    }

    /// Set interrupt bit
    #[inline]
    pub fn set_interrupt(&self, bit: u8) {
        self.hot.interrupts.fetch_or(1u64 << (bit + 8), Ordering::SeqCst);
    }

    /// Clear interrupt bit
    #[inline]
    pub fn clear_interrupt(&self, bit: u8) {
        self.hot.interrupts.fetch_and(!(1u64 << (bit + 8)), Ordering::SeqCst);
    }

    /// Get current privilege mode
    pub fn get_privilege_mode(&self) -> PrivilegeMode {
        // EXL or ERL forces kernel mode
        if (self.cp0_status & (STATUS_EXL | STATUS_ERL)) != 0 {
            return PrivilegeMode::Kernel;
        }

        // Otherwise, check KSU field
        let ksu = (self.cp0_status >> STATUS_KSU_SHIFT) & 0x3;
        match ksu {
            KSU_KERNEL => PrivilegeMode::Kernel,
            KSU_SUPERVISOR => PrivilegeMode::Supervisor,
            KSU_USER => PrivilegeMode::User,
            _ => PrivilegeMode::Kernel, // Reserved, treat as kernel
        }
    }

    /// Check if CPU is in 64-bit mode for the current privilege level
    pub fn is_64bit_mode(&self) -> bool {
        let mode = self.get_privilege_mode();
        match mode {
            PrivilegeMode::Kernel => (self.cp0_status & STATUS_KX) != 0,
            PrivilegeMode::Supervisor => (self.cp0_status & STATUS_SX) != 0,
            PrivilegeMode::User => (self.cp0_status & STATUS_UX) != 0,
        }
    }

    /// Check if CPU is in kernel mode
    #[inline]
    pub fn is_kernel_mode(&self) -> bool {
        matches!(self.get_privilege_mode(), PrivilegeMode::Kernel)
    }

    /// Check if interrupts are enabled
    pub fn interrupts_enabled(&self) -> bool {
        // IE bit must be set, and not in exception mode (EXL=0, ERL=0)
        let ie = (self.cp0_status & STATUS_IE) != 0;
        let exl = (self.cp0_status & STATUS_EXL) != 0;
        let erl = (self.cp0_status & STATUS_ERL) != 0;
        ie && !exl && !erl
    }

    // FPU register access helpers

    /// Read FPR as single-precision float
    #[inline]
    pub fn read_fpr_s(&self, reg: u32) -> f32 {
        f32::from_bits(self.fpr[reg as usize] as u32)
    }

    /// Write FPR as single-precision float (lower 32 bits)
    #[inline]
    pub fn write_fpr_s(&mut self, reg: u32, value: f32) {
        // Keep upper 32 bits unchanged
        self.fpr[reg as usize] = (self.fpr[reg as usize] & 0xFFFFFFFF_00000000) | (value.to_bits() as u64);
    }

    /// Read FPR as double-precision float
    #[inline]
    pub fn read_fpr_d(&self, reg: u32) -> f64 {
        f64::from_bits(self.fpr[reg as usize])
    }

    /// Write FPR as double-precision float
    #[inline]
    pub fn write_fpr_d(&mut self, reg: u32, value: f64) {
        self.fpr[reg as usize] = value.to_bits();
    }

    /// Read FPR as word (lower 32 bits as u32)
    #[inline]
    pub fn read_fpr_w(&self, reg: u32) -> u32 {
        self.fpr[reg as usize] as u32
    }

    /// Write FPR as word (lower 32 bits, upper bits preserved)
    #[inline]
    pub fn write_fpr_w(&mut self, reg: u32, value: u32) {
        self.fpr[reg as usize] = (self.fpr[reg as usize] & 0xFFFFFFFF_00000000) | (value as u64);
    }

    /// Read FPR as doubleword (full 64 bits)
    #[inline]
    pub fn read_fpr_l(&self, reg: u32) -> u64 {
        self.fpr[reg as usize]
    }

    /// Write FPR as doubleword (full 64 bits)
    #[inline]
    pub fn write_fpr_l(&mut self, reg: u32, value: u64) {
        self.fpr[reg as usize] = value;
    }
}

// ---------------------------------------------------------------------------
// FR-mode accessor free functions (used as bare fn pointers in MipsExecutor)
// ---------------------------------------------------------------------------
//
// FR=0 (32-bit FPU mode, IRIX 5.3):
//   Each physical 64-bit fpr[] slot holds an even/odd pair:
//     fpr[n & !1] bits 31:0  = FPR(n & !1)  — even register (single/word)
//     fpr[n & !1] bits 63:32 = FPR(n | 1)   — odd register  (single/word)
//   Double/long use the full 64-bit even slot (reg must be even):
//     fpr[reg & !1] bits 63:0 = FPR(reg) as double/long
//
// FR=1 (64-bit FPU mode, IRIX 6.5): each fpr[n] is an independent 64-bit slot.

// --- FR=0 double/long ---

pub fn read_fpr_d_fr0(core: &MipsCore, reg: u32) -> f64 {
    f64::from_bits(core.fpr[(reg & !1) as usize])
}
pub fn write_fpr_d_fr0(core: &mut MipsCore, reg: u32, value: f64) {
    core.fpr[(reg & !1) as usize] = value.to_bits();
}
pub fn read_fpr_l_fr0(core: &MipsCore, reg: u32) -> u64 {
    core.fpr[(reg & !1) as usize]
}
pub fn write_fpr_l_fr0(core: &mut MipsCore, reg: u32, value: u64) {
    core.fpr[(reg & !1) as usize] = value;
}

// --- FR=0 single/word (odd reg lives in upper half of even slot) ---

#[inline]
pub fn read_fpr_w_fr0(core: &MipsCore, reg: u32) -> u32 {
    let shift = (reg & 1) << 5;  // 0 for even, 32 for odd
    (core.fpr[(reg & !1) as usize] >> shift) as u32
}
#[inline]
pub fn write_fpr_w_fr0(core: &mut MipsCore, reg: u32, value: u32) {
    let shift = (reg & 1) << 5;  // 0 for even, 32 for odd
    let slot = &mut core.fpr[(reg & !1) as usize];
    *slot = (*slot & !(0xFFFF_FFFFu64 << shift)) | ((value as u64) << shift);
}

// --- FR=1 wrappers with matching signatures ---

pub fn read_fpr_d_fr1(core: &MipsCore, reg: u32) -> f64        { core.read_fpr_d(reg) }
pub fn write_fpr_d_fr1(core: &mut MipsCore, reg: u32, v: f64)  { core.write_fpr_d(reg, v) }
pub fn read_fpr_l_fr1(core: &MipsCore, reg: u32) -> u64        { core.read_fpr_l(reg) }
pub fn write_fpr_l_fr1(core: &mut MipsCore, reg: u32, v: u64)  { core.write_fpr_l(reg, v) }
pub fn read_fpr_w_fr1(core: &MipsCore, reg: u32) -> u32        { core.read_fpr_w(reg) }
pub fn write_fpr_w_fr1(core: &mut MipsCore, reg: u32, v: u32)  { core.write_fpr_w(reg, v) }

impl MipsCore {

    /// Pack condition codes cc0..cc7 out of FCSR into an 8-bit FCCR-style value
    /// (bit i = cc_i): FCSR bit 23 = cc0, bits [31:25] = cc1..cc7.
    #[inline]
    fn fccr_from_fcsr(fcsr: u32) -> u32 {
        let cc0 = (fcsr >> 23) & 1;
        let cc1_7 = (fcsr >> 25) & 0x7F; // bits 25..31 -> cc1..cc7
        cc0 | (cc1_7 << 1)
    }

    /// Scatter an 8-bit FCCR-style value (bit i = cc_i) back into FCSR's
    /// condition-code bits (bit 23 = cc0, bits [31:25] = cc1..cc7).
    #[inline]
    fn fcsr_with_fccr(fcsr: u32, fccr: u32) -> u32 {
        let cc0 = fccr & 1;
        let cc1_7 = (fccr >> 1) & 0x7F;
        (fcsr & !((1 << 23) | (0x7F << 25)))
            | (cc0 << 23)
            | (cc1_7 << 25)
    }

    /// Read FPU control register
    #[inline]
    pub fn read_fpu_control(&self, reg: u32) -> u32 {
        match reg {
            0 => self.fpu_fir,
            25 => Self::fccr_from_fcsr(self.fpu_fcsr),
            26 => self.fpu_fexr,
            28 => self.fpu_fenr,
            31 => self.fpu_fcsr,
            _ => 0, // Undefined registers read as 0
        }
    }

    /// Write FPU control register
    #[inline]
    pub fn write_fpu_control(&mut self, reg: u32, value: u32) {
        match reg {
            0 => { /* FIR is read-only */ }
            25 => {
                self.fpu_fccr = value & 0xFF;
                self.fpu_fcsr = Self::fcsr_with_fccr(self.fpu_fcsr, self.fpu_fccr);
            }
            26 => self.fpu_fexr = value,
            28 => self.fpu_fenr = value,
            31 => {
                self.fpu_fcsr = value;
                self.fpu_fccr = Self::fccr_from_fcsr(value);

                // Update host FPU rounding mode to match
                let rm = value & 0x3;
                crate::platform::set_fpu_mode(rm as u8);
            }
            _ => {} // Writes to undefined registers are ignored
        }
    }

    /// Get FPU condition code bit: cc0 is FCSR bit 23, cc1..cc7 are FCSR bits [31:25].
    #[inline]
    pub fn get_fpu_cc(&self, cc: u32) -> bool {
        let bit = if cc == 0 { 23 } else if cc < 8 { 24 + cc } else { return false; };
        (self.fpu_fcsr >> bit) & 1 != 0
    }

    /// Set FPU condition code bit: cc0 is FCSR bit 23, cc1..cc7 are FCSR bits [31:25].
    #[inline]
    pub fn set_fpu_cc(&mut self, cc: u32, value: bool) {
        if cc >= 8 { return; }
        let bit = if cc == 0 { 23 } else { 24 + cc };
        if value {
            self.fpu_fcsr |= 1 << bit;
        } else {
            self.fpu_fcsr &= !(1 << bit);
        }
        self.fpu_fccr = Self::fccr_from_fcsr(self.fpu_fcsr);
    }
}

/// Deliver an exception's architectural effect onto `core`: update
/// Cause/EPC/Status per the R4000 exception-entry sequence and select the
/// vector, exactly as `MipsExecutor::handle_exception` (`mips_exec.rs`) does
/// — this function *is* that logic, extracted so it has exactly one
/// implementation (the design doc's §4.2 "single-implementation delivery"
/// principle already governs every other jitv2 exception path; this closes
/// the one gap where the interpreter's exception vectoring itself wasn't
/// callable independently of a full `MipsExecutor`).
///
/// Reads `core.in_delay_slot` directly — one field, one meaning, shared by
/// both engines (see its doc comment): the interpreter's
/// `branch_delay`/`handle_exec_complete` set it on the plain dispatch path,
/// and jitv2's `emit_slot_semantics` (`jitv2/codegen.rs`) sets it directly
/// around a delay slot's inlined body, since the JIT has no separate
/// dispatch step of its own to hang it on. `jitv2_verify` (no executor, only
/// a bare `MipsCore` reconstructed from a trace record) leaves it at
/// whatever `MipsCore::new()`/the trace's seeded state left it as — a trace
/// record's pre-state has no delay-slot context to recover, the same
/// limitation that already makes load/store instructions unverifiable
/// there.
///
/// Does NOT perform the two executor-level side effects the real
/// `handle_exception` also does (`cache.set_llbit(false)`,
/// `nanotlb_invalidate()`) — neither affects any field `CoreState`
/// (`src/trace.rs`) tracks, and `jitv2_verify` has no cache/TLB to touch in
/// the first place. `handle_exception` remains responsible for those; this
/// function only owns the part that's genuinely portable.
///
/// `status` is the raw exception status word (`ExecStatus`, `mips_exec.rs`)
/// — taken as `u32` here rather than the `ExecStatus` type alias to avoid a
/// `mips_core -> mips_exec` module dependency for what's just `type
/// ExecStatus = u32`. `EXEC_IS_TLB_REFILL`/`EXEC_IS_XTLB_REFILL`'s bit
/// values (`1 << 28`, `1 << 29`) are inlined below; their canonical
/// definitions and doc comments live in `mips_exec.rs`.
pub fn deliver_exception(core: &mut MipsCore, status: u32) {
    const EXEC_IS_TLB_REFILL: u32 = 1 << 28;
    const EXEC_IS_XTLB_REFILL: u32 = 1 << 29;

    let is_tlb_refill = status & EXEC_IS_TLB_REFILL != 0;
    let is_xtlb_refill = status & EXEC_IS_XTLB_REFILL != 0;

    let was_exl = (core.cp0_status & STATUS_EXL) != 0;

    let mut cause = core.cp0_cause;
    cause = (cause & !CAUSE_EXCCODE_MASK) | (status & CAUSE_EXCCODE_MASK);

    if !was_exl {
        if core.in_delay_slot {
            cause |= CAUSE_BD;
            core.cp0_epc = core.pc.wrapping_sub(4);
        } else {
            cause &= !CAUSE_BD;
            core.cp0_epc = core.pc;
        }
    }
    core.cp0_cause = cause;

    core.cp0_status |= STATUS_EXL;

    let bev = (core.cp0_status & STATUS_BEV) != 0;
    let vector_base = if bev { 0xFFFFFFFF_BFC00200u64 } else { 0xFFFFFFFF_80000000u64 };
    let offset = if !was_exl && is_tlb_refill {
        if is_xtlb_refill { 0x080 } else { 0x000 }
    } else {
        0x180
    };

    core.pc = vector_base + offset;
}

/// CPU Privilege Modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrivilegeMode {
    Kernel     = 0,
    Supervisor = 1,
    User       = 2,
}

/// Const-generic privilege level values, matching PrivilegeMode discriminants.
pub const PRIV_KERNEL: u8     = PrivilegeMode::Kernel     as u8;
pub const PRIV_SUPERVISOR: u8 = PrivilegeMode::Supervisor as u8;
pub const PRIV_USER: u8       = PrivilegeMode::User       as u8;

impl PrivilegeMode {
    #[inline]
    pub const fn as_u8(self) -> u8 { self as u8 }
}

impl Default for MipsCore {
    fn default() -> Self {
        Self::new()
    }
}
