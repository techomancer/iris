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
pub const CAUSE_CE_MASK: u32 = 0x3 << 28;      // Coprocessor Error mask
pub const CAUSE_CE_SHIFT: u32 = 28;            // Coprocessor Error shift
pub const CAUSE_BD: u32 = 1 << 31;             // Branch Delay

// KSU field values
pub const KSU_KERNEL: u32 = 0b00;
pub const KSU_SUPERVISOR: u32 = 0b01;
pub const KSU_USER: u32 = 0b10;

/// MIPS CPU Core with full register state
pub struct MipsCore {
    // General Purpose Registers (GPRs)
    pub gpr: [u64; 32],  // r0-r31, where r0 is always zero

    // Special Registers
    pub pc: u64,         // Program Counter
    pub hi: u64,         // Multiply/Divide HI result
    pub lo: u64,         // Multiply/Divide LO result

    // CP0 - System Control Coprocessor Registers
    pub cp0_index: u32,       // 0: TLB Index
    pub cp0_random: u32,      // 1: TLB Random
    pub cp0_entrylo0: u64,    // 2: TLB Entry Low 0 (64-bit, truncated in 32-bit mode)
    pub cp0_entrylo1: u64,    // 3: TLB Entry Low 1 (64-bit, truncated in 32-bit mode)
    pub cp0_context: u64,     // 4: Context (page table pointer)
    pub cp0_pagemask: u64,    // 5: TLB Page Mask (64-bit, truncated in 32-bit mode)
    pub cp0_wired: u32,       // 6: TLB Wired boundary
    pub cp0_badvaddr: u64,    // 8: Bad Virtual Address
    pub cp0_count: u64,       // 9: Timer Count — bits[63:32]=hardware count, bits[31:0]=fraction (32.32 fp)
    pub cp0_entryhi: u64,     // 10: TLB Entry High (64-bit, truncated in 32-bit mode)
    pub cp0_compare: u64,     // 11: Timer Compare — bits[63:32]=hardware compare, bits[31:0]=0 (32.32 fp)
    /// Per-instruction cp0_count increment in 32.32 fixed-point (same format as cp0_count).
    /// Calibrated on every cp0_compare write to hit the programmed interval in real time.
    /// Default 1<<31 = 0.5 hardware counts per instruction (R4400: Count runs at half CPU clock).
    pub count_step: u64,
    /// Atomic shadow of `count_step` — updated whenever count_step changes.
    /// Shared with the display refresh thread for status bar display.
    pub count_step_atomic: Arc<AtomicU64>,
    /// Cycle count when cp0_compare was last written (0 = never written yet).
    compare_last_cycles: u64,
    /// Wall-clock instant when cp0_compare was last written.
    compare_last_instant: std::time::Instant,
    /// Frequency map of CP0 Compare delta values (hardware counts, rounded to nearest 100).
    /// Key = `(delta >> 16) / 100 * 100`, value = number of occurrences.
    #[cfg(feature = "developer_ip7")]
    pub compare_delta_stats: std::collections::HashMap<u32, u32>,
    /// Learned slow-tick CP0 delta in hardware counts (32.32 fixed-point, >> 32 = integer counts).
    /// Initialised to 0 (unknown). First delta seen is assumed to be the 100 Hz (slow) tick.
    pub compare_delta_slow: u64,
    /// Learned fast-tick CP0 delta in hardware counts (32.32 fixed-point).
    /// Initialised to 0 (unknown). Set once we see a delta ~10x smaller than delta_slow.
    pub compare_delta_fast: u64,
    /// The raw 32.32 fixed-point delta programmed in the *previous* Compare write.
    /// Used for calibration: dt_ns/dc measure the old interval, so count_step must be
    /// computed against the old delta, not the new one.  Zero = no previous write yet.
    pub compare_delta_prev: u64,
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

    // CP1 - Floating Point Unit Registers
    pub fpr: [u64; 32],       // FPU data registers (64-bit, can be used as pairs for 32-bit)
    pub fpu_fir: u32,         // 0: FP Implementation/Revision
    pub fpu_fccr: u32,        // 25: FP Condition Codes
    pub fpu_fexr: u32,        // 26: FP Exceptions
    pub fpu_fenr: u32,        // 28: FP Enables
    pub fpu_fcsr: u32,        // 31: FP Control/Status

    /// Local (non-atomic) cycle counter — incremented every instruction by MipsExecutor::step().
    /// Used by write_cp0 calibration to get an accurate dc without waiting for the atomic flush.
    pub local_cycles: u64,

    // Interrupt handling
    // Bits 8..15 = IP0..IP7 (mirror CAUSE.IP layout).  Bit 63 = soft-reset request.
    pub interrupts: Arc<AtomicU64>,
    pub cycles: Arc<AtomicU64>,
    /// Counts every CP0 Count==Compare match (i.e. every fastick interrupt).
    pub fasttick_count: Arc<AtomicU64>,

    // Execution state
    pub running: bool,
    pub halted: bool,

    /// Called whenever CP0 Status (reg 12) is written, with (old_value, new_value).
    /// The first element is the callback function, the second is an opaque context pointer
    /// (typically a type-erased `*mut MipsExecutor<T,C>` set by the executor after construction).
    pub status_changed_cb: Option<(fn(*mut core::ffi::c_void, u32, u32), *mut core::ffi::c_void)>,

    /// Nano-TLB: 3-entry direct-mapped cache, one slot per access type (Fetch/Read/Write).
    /// Indexed by AccessType discriminant (0=Fetch, 1=Read, 2=Write).
    pub nanotlb: [NanoTlbEntry; 3],
}

/// Single nano-TLB entry.
///
/// `va_tag`     — bits [63:12] of the virtual address (page-aligned VA >> 12).
/// `pa_encoded` — bits [63:12] = physical page base (PA & !0xFFF),
///                bit   [3]    = valid flag (1 = entry is live),
///                bits  [2:0]  = hardware C-field cache attr (2=Uncached, 3=Cacheable, 5=CacheableCoherent).
///
/// C-field stored directly in bits [2:0] so cache_attr_raw() is a plain mask,
/// and fill_raw() needs no shift or table lookup for the cache attribute.
#[derive(Clone, Copy, Default)]
pub struct NanoTlbEntry {
    pub va_tag:     u64,
    pub pa_encoded: u64,
}

impl NanoTlbEntry {
    pub const VALID_BIT: u64 = 0x8;

    // Invalidation always zeroes pa_encoded, so nonzero implies VALID_BIT is set.
    #[inline(always)]
    pub fn is_valid(&self) -> bool { self.pa_encoded != 0 }

    #[inline(always)]
    pub fn matches(&self, va: u64) -> bool {
        self.pa_encoded != 0 && self.va_tag == (va >> 12)
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
        self.va_tag     = va >> 12;
        self.pa_encoded = (phys_addr & !0xFFF) | Self::VALID_BIT | (attr as u64);
    }

    /// Fill entry from a raw C-field value (2=Uncached, 3=Cacheable, 5=CacheableCoherent).
    /// Used by nanotlb_translate to avoid re-converting through CacheAttr enum.
    #[inline(always)]
    pub fn fill_raw(&mut self, va: u64, phys_addr: u64, c_field: u32) {
        self.va_tag     = va >> 12;
        self.pa_encoded = (phys_addr & !0xFFF) | Self::VALID_BIT | c_field as u64;
    }

    #[inline(always)]
    pub fn invalidate(&mut self) { self.pa_encoded = 0; }
}

// SAFETY: The raw pointer in status_changed_cb is only accessed from the CPU thread.
unsafe impl Send for MipsCore {}

impl MipsCore {
    /// Create a new MIPS core with reset state
    pub fn new() -> Self {
        let mut core = Self {
            gpr: [0; 32],
            pc: 0,
            hi: 0,
            lo: 0,
            cp0_index: 0,
            cp0_random: 0,
            cp0_entrylo0: 0,
            cp0_entrylo1: 0,
            cp0_context: 0,
            cp0_pagemask: 0,
            cp0_wired: 0,
            cp0_badvaddr: 0,
            cp0_count: 0,
            cp0_entryhi: 0,
            cp0_compare: 0,
            count_step: 1 << 31,
            count_step_atomic: Arc::new(AtomicU64::new(1 << 31)),
            compare_last_cycles: 0,
            compare_last_instant: std::time::Instant::now(),
            #[cfg(feature = "developer_ip7")]
            compare_delta_stats: std::collections::HashMap::new(),
            compare_delta_slow: 0,
            compare_delta_fast: 0,
            compare_delta_prev: 0,
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
            fpr: [0; 32],
            fpu_fir: 0,
            fpu_fccr: 0,
            fpu_fexr: 0,
            fpu_fenr: 0,
            fpu_fcsr: 0,
            local_cycles: 0,
            interrupts: Arc::new(AtomicU64::new(0)),
            cycles: Arc::new(AtomicU64::new(0)),
            fasttick_count: Arc::new(AtomicU64::new(0)),
            running: false,
            halted: false,
            status_changed_cb: None,
            nanotlb: [NanoTlbEntry::default(); 3],
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
            self.count_step = 1 << 31;
            self.count_step_atomic.store(1 << 31, Ordering::Relaxed);
            self.local_cycles = 0;
            self.compare_last_cycles = 0;
            self.compare_last_instant = std::time::Instant::now();
            self.cp0_random = self.tlb_entries - 1;
            self.cp0_random_cycle = 0;
            self.cp0_prid = 0x00000440; // R4400, imp=0x04, majrev=4, minrev=0
            self.cp0_watchlo = 0;
            self.cp0_watchhi = 0;
            self.cp0_xcontext = 0;
            self.cp0_ecc = 0;
            self.cp0_cacheerr = 0;
            self.cp0_taglo = 0;
            self.cp0_taghi = 0;

            // CP1 registers
            self.fpu_fir = 0x00000500; // R4000 FPU: imp=0x05, rev=0
            self.fpu_fccr = 0;
            self.fpu_fexr = 0;
            self.fpu_fenr = 0;
            self.fpu_fcsr = 0;
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
        self.interrupts.store(0, Ordering::SeqCst);
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
        let current_cycles = self.cycles.load(Ordering::Relaxed);
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
            9 => self.cp0_count >> 32,
            10 => self.cp0_entryhi,
            11 => self.cp0_compare >> 32,
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
            9 => self.cp0_count >> 32,
            10 => self.cp0_entryhi,
            11 => self.cp0_compare >> 32,
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

    /// Bin a CP0 Compare delta into a target tick period in nanoseconds
    /// (1_000_000 for 1 kHz, 10_000_000 for 100 Hz).
    ///
    /// `delta` may be in any consistent unit (e.g. raw 32.32 fixed-point) — only the
    /// ratios between slow and fast buckets matter.  Maintains two learned buckets:
    /// `compare_delta_slow` (100 Hz, seeded on first call) and `compare_delta_fast`
    /// (~10x smaller, 1 kHz).  All comparisons use ±5% fuzzy equality.
    /// Returns `Some(target_ns)` or `None` for a zero/degenerate delta.
    fn bin_compare_delta(&mut self, d: u64) -> Option<u64> {
        if d == 0 {
            return None;
        }
        // ±5% fuzzy equality.
        let fuzzy_eq = |a: u64, b: u64| -> bool {
            let threshold = a.max(b) * 5 / 100;
            a.abs_diff(b) <= threshold
        };

        if self.compare_delta_slow == 0 {
            // First delta ever — seed slow bucket, assume 100 Hz.
            self.compare_delta_slow = d;
            return Some(10_000_000);
        }

        if fuzzy_eq(d, self.compare_delta_slow) {
            return Some(10_000_000);
        }

        if self.compare_delta_fast != 0 && fuzzy_eq(d, self.compare_delta_fast) {
            return Some(1_000_000);
        }

        // Check if d ≈ slow/10 (i.e. ~10x smaller → fast tick).
        // Use division to avoid overflow; guard ensures slow has a meaningful integer part.
        if self.compare_delta_slow >= (10 << 32) && fuzzy_eq(d, self.compare_delta_slow / 10) {
            self.compare_delta_fast = d;
            return Some(1_000_000);
        }

        if d > self.compare_delta_slow {
            // Larger than slow — one-shot or low-freq timer; update slow bucket.
            self.compare_delta_slow = d;
            Some(10_000_000)
        } else {
            // Unrecognised intermediate — fall back to slow.
            Some(10_000_000)
        }
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
                self.cp0_random_cycle = self.cycles.load(Ordering::Relaxed);
            }
            8 => { /* BadVAddr is read-only */ }
            9 => self.cp0_count = value << 32,
            10 => { // always use 64bit mask because the entries need to be valid in 64 bit mode even when they were set from 32 bit mode
                self.cp0_entryhi = value & 0xC000_00FF_FFFF_E0FF;
            },
            11 => {
                self.cp0_compare = value << 32;
                // Clear timer interrupt when Compare is written
                self.cp0_cause &= !CAUSE_IP7;

                // Recalibrate count_step so the timer fires at the correct wall-clock rate.
                // CP0 Count increments at half the CPU clock (R4400: Count = CPU_clock / 2),
                // so nominal step = 1<<15 (0.5 hardware counts per emulated instruction).
                // Formula: count_step = delta * dt_ns / (dc * 1_000_000)
                //   delta = count units to next compare (what the kernel programmed)
                //   dc    = instructions executed in last interval
                //   dt_ns = wall-clock ns elapsed in last interval
                // = (count units per instruction) * (wall-clock stretch factor)
                // Only calibrate for ~1ms timer intervals (IRIX 1000 Hz scheduler);
                // leave count_step unchanged for other timer uses (one-shot, low-freq).
                let now = std::time::Instant::now();
                let cycles_now = self.local_cycles;
                // Compute new_delta before the calibration block so we can guard on it.
                // Top bit set means cp0_count > cp0_compare — counter hasn't wrapped correctly
                // or the kernel is writing a compare in the past. Skip calibration entirely.
                let new_delta = self.cp0_compare.wrapping_sub(self.cp0_count);
                if new_delta >> 63 != 0 {
                    self.compare_last_cycles = cycles_now;
                    self.compare_last_instant = now;
                } else if self.compare_last_cycles != 0 {
                    let dc = cycles_now.wrapping_sub(self.compare_last_cycles);
                    let dt_ns = now.duration_since(self.compare_last_instant).as_nanos() as u64;
                    // new_delta: what the *next* interval will fire at, stored as 32.32 fp.
                    #[cfg(feature = "developer_ip7")]
                    {
                        let bucket = ((new_delta >> 32) as u32 / 100) * 100;
                        *self.compare_delta_stats.entry(bucket).or_insert(0) += 1;
                    }
                    // dt_ns/dc measure the interval since the last Compare write, which
                    // ran under compare_delta_prev.  Calibrate against that, not new_delta,
                    // so a tick-rate switch doesn't mix old timing with the new delta.
                    // If there was no previous delta (first write after reset), skip.
                    let prev_delta = self.compare_delta_prev;
                    if prev_delta != 0 {
                        if let Some(snapped_ns) = self.bin_compare_delta(prev_delta) {
                            if dc > 0 {
                                let denom = (dc as u128).saturating_mul(snapped_ns.into());
                                self.count_step = ((prev_delta as u128).saturating_mul(dt_ns.into()) / denom)
                                    .clamp(1 << 30, 10 << 31) as u64;
                                #[cfg(feature = "developer_ip7")]
                                {
                                    let total_samples: u32 = self.compare_delta_stats.values().sum();
                                    if total_samples <= 10 {
                                        eprintln!("compare calib: prev_d={} new_d={} dt_ns={} dc={} \
                                            snapped={}ms slow={} fast={} count_step={}",
                                            prev_delta >> 32, new_delta >> 32, dt_ns, dc,
                                            snapped_ns / 1_000_000,
                                            self.compare_delta_slow >> 32,
                                            self.compare_delta_fast >> 32,
                                            self.count_step);
                                    }
                                }
                            } else {
                                self.count_step = 1 << 31;
                            }
                            self.count_step_atomic.store(self.count_step, Ordering::Relaxed);
                        }
                    }
                    self.compare_delta_prev = new_delta;
                }
                // First write: keep default count_step (1<<15), just record state.
                self.compare_last_cycles = cycles_now;
                self.compare_last_instant = now;
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
                // Only bits 5:0 are writable (K0, CU, DB, IB)
                let mask = 0x3F;
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
        self.interrupts.fetch_or(1u64 << (bit + 8), Ordering::SeqCst);
    }

    /// Clear interrupt bit
    #[inline]
    pub fn clear_interrupt(&self, bit: u8) {
        self.interrupts.fetch_and(!(1u64 << (bit + 8)), Ordering::SeqCst);
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

    /// Read FPU control register
    #[inline]
    pub fn read_fpu_control(&self, reg: u32) -> u32 {
        match reg {
            0 => self.fpu_fir,
            25 => self.fpu_fccr,
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
                // Sync CC0 to FCSR bit 23
                if (self.fpu_fccr & 1) != 0 {
                    self.fpu_fcsr |= 1 << 23;
                } else {
                    self.fpu_fcsr &= !(1 << 23);
                }
            }
            26 => self.fpu_fexr = value,
            28 => self.fpu_fenr = value,
            31 => {
                self.fpu_fcsr = value;
                // Sync FCSR bit 23 to FCCR bit 0
                let cc0 = (value >> 23) & 1;
                self.fpu_fccr = (self.fpu_fccr & !1) | cc0;

                // Update host FPU rounding mode to match
                let rm = value & 0x3;
                crate::platform::set_fpu_mode(rm as u8);
            }
            _ => {} // Writes to undefined registers are ignored
        }
    }

    /// Get FPU condition code bit (default: CC0, bit 23 of FCSR)
    #[inline]
    pub fn get_fpu_cc(&self, cc: u32) -> bool {
        if cc < 8 {
            (self.fpu_fccr >> cc) & 1 != 0
        } else {
            false
        }
    }

    /// Set FPU condition code bit (default: CC0, bit 23 of FCSR)
    #[inline]
    pub fn set_fpu_cc(&mut self, cc: u32, value: bool) {
        if cc < 8 {
            if value {
                self.fpu_fccr |= 1 << cc;
            } else {
                self.fpu_fccr &= !(1 << cc);
            }

            // Sync CC0 to FCSR bit 23
            if cc == 0 {
                if value {
                    self.fpu_fcsr |= 1 << 23;
                } else {
                    self.fpu_fcsr &= !(1 << 23);
                }
            }
        }
    }
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
