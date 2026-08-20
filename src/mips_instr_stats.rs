// Per-instruction identity map: classifies every decoded MIPS instruction
// into a flat `InstrKind` enum. `InstrKind`/`classify_instr`/`IsaClass`
// compile unconditionally — jitv2's `opcode_support.rs` depends on them to
// answer "does codegen have an emitter for this instruction's category"
// without needing the `instr_stats` feature.
//
// `InstrStats` (the actual counters + exit-time report) stays behind
// `#[cfg(feature = "instr_stats")]`: purely diagnostic, intrusive on the hot
// path. Originally built to investigate MIPS IV instruction usage in
// Nekoware MIPS4-ISA builds (Firefox/SeaMonkey) that ran correctly in MAME
// but misbehaved here — the working assumption was that MIPS III emulation
// was solid and the bug was in a MIPS IV instruction (COP1X, MOVCI/MOVZ/MOVN,
// FMOVZ/FMOVN/FMOVCF, PREFX, paired-single, etc).
//
// Classification mirrors `decode_into` in mips_exec.rs field-for-field, but
// produces an enum discriminant instead of a function pointer, so it does not
// touch `DecodedInstr` (which is deliberately kept minimal — it lives in the
// L1I cache line shadow).

use crate::mips_isa::*;

/// One variant per instruction handler in mips_exec.rs (exec_* functions),
/// plus Reserved for undecoded/illegal encodings.
#[allow(non_camel_case_types)] // Fadd_s etc. mirror MIPS mnemonic dotted suffixes (.s/.d/.w/.l)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum InstrKind {
    Reserved = 0,
    Nop,
    Sll, Movci, Srl, Sra, Sllv, Srlv, Srav, Jr, Jalr, Movz, Movn,
    Syscall, Break, Sync, Mfhi, Mthi, Mflo, Mtlo,
    Dsllv, Dsrlv, Dsrav, Mult, Multu, Div, Divu, Dmult, Dmultu, Ddiv, Ddivu,
    Add, Addu, Sub, Subu, And, Or, Xor, Nor, Slt, Sltu,
    Dadd, Daddu, Dsub, Dsubu,
    Tge, Tgeu, Tlt, Tltu, Teq, Tne,
    Dsll, Dsrl, Dsra, Dsll32, Dsrl32, Dsra32,

    Bltz, Bgez, Bltzl, Bgezl, Tgei, Tgeiu, Tlti, Tltiu, Teqi, Tnei,
    Bltzal, Bgezal, Bltzall, Bgezall,

    J, Jal, Beq, Bne, Blez, Bgtz, Beql, Bnel, Blezl, Bgtzl,
    Addi, Addiu, Daddi, Daddiu, Slti, Sltiu, Andi, Ori, Xori, Lui,

    // COP0
    Mfc0, Dmfc0, Mtc0, Dmtc0, Tlbr, Tlbwi, Tlbwr, Tlbp, Eret,

    // COP1 move/branch
    Mfc1, Dmfc1, Cfc1, Mtc1, Dmtc1, Ctc1, Bc1,

    // COP1.S
    Fadd_s, Fsub_s, Fmul_s, Fdiv_s, Fsqrt_s, Fabs_s, Fmov_s, Fneg_s,
    Fround_l_s, Ftrunc_l_s, Fceil_l_s, Ffloor_l_s,
    Fround_w_s, Ftrunc_w_s, Fceil_w_s, Ffloor_w_s,
    Fmovcf_s, Fmovz_s, Fmovn_s, Frecip_s, Frsqrt_s,
    Fcvt_d_s, Fcvt_w_s, Fcvt_l_s, Fcc_s,

    // COP1.D
    Fadd_d, Fsub_d, Fmul_d, Fdiv_d, Fsqrt_d, Fabs_d, Fmov_d, Fneg_d,
    Fround_l_d, Ftrunc_l_d, Fceil_l_d, Ffloor_l_d,
    Fround_w_d, Ftrunc_w_d, Fceil_w_d, Ffloor_w_d,
    Fmovcf_d, Fmovz_d, Fmovn_d, Frecip_d, Frsqrt_d,
    Fcvt_s_d, Fcvt_w_d, Fcvt_l_d, Fcc_d,

    // COP1.W / COP1.L (convert-from)
    Fcvt_s_w, Fcvt_d_w, Fcvt_s_l, Fcvt_d_l,

    // COP1X (MIPS IV)
    Lwxc1, Ldxc1, Swxc1, Sdxc1, Prefx,
    Madd_s, Madd_d, Msub_s, Msub_d, Nmadd_s, Nmadd_d, Nmsub_s, Nmsub_d,

    // Loads/stores
    Lb, Lh, Lwl, Lw, Lbu, Lhu, Lwr, Lwu,
    Sb, Sh, Swl, Sw, Sdl, Sdr, Swr, Cache,
    Ll, Lwc1, Ldc1, Ldl, Ldr, Ld, Sc, Swc1, Sdc1, Sd,
    Pref, Lld, Scd,
}

pub const NUM_INSTR_KINDS: usize = InstrKind::Scd as usize + 1;

/// ISA/functional class an instruction belongs to. Used to bucket the
/// frequency report so MIPS IV usage (the prime suspect for the Nekoware
/// Firefox/SeaMonkey bug — our MIPS III path is assumed correct since
/// everything else works) stands out from baseline MIPS I/II traffic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum IsaClass {
    /// MIPS I/II baseline integer/branch/load-store — present since the R2000/R3000.
    Mips12,
    /// MIPS III (R4000) 64-bit integer ops: DADD*, DSUB*, DMULT*, DDIV*, DSLL*/DSRL*/DSRA*,
    /// 64-bit loads/stores (LD/SD/LWU/LDL/LDR/SDL/SDR/LLD/SCD), ERET.
    Mips3,
    /// MIPS III FPU additions: DMFC1/DMTC1 (64-bit FPR move) and the *.l (long) FP convert/round/trunc/ceil/floor ops.
    Mips3Fpu,
    /// COP0 (system coprocessor) moves and TLB management — privileged, not ISA-level-specific,
    /// but broken out separately since it's a distinct functional unit from the integer/FPU pipes.
    Cop0,
    /// MIPS IV (R5000/R8000/R10000) integer/control additions: MOVCI/MOVZ/MOVN (conditional move),
    /// PREF/PREFX (prefetch), indexed FP loads/stores (LWXC1/LDXC1/SWXC1/SDXC1).
    Mips4,
    /// MIPS IV FPU additions: FMOVZ/FMOVN/FMOVCF (conditional FP move), FRECIP/FRSQRT,
    /// and the fused MADD/MSUB/NMADD/NMSUB family.
    Mips4Fpu,
    /// Undecoded/illegal encoding (exec_reserved).
    Reserved,
}

impl IsaClass {
    pub fn label(self) -> &'static str {
        match self {
            IsaClass::Mips12    => "MIPS I/II (baseline)",
            IsaClass::Mips3     => "MIPS III (64-bit integer)",
            IsaClass::Mips3Fpu  => "MIPS III (FPU: 64-bit move / long convert)",
            IsaClass::Cop0      => "COP0 (system coprocessor)",
            IsaClass::Mips4     => "MIPS IV (integer/control)",
            IsaClass::Mips4Fpu  => "MIPS IV (FPU: cond-move / recip / fused madd)",
            IsaClass::Reserved  => "reserved/illegal",
        }
    }
}

/// Coarse functional-unit bitmask (ALU/FPU/branch/load-store/COP0), orthogonal to
/// [`IsaClass`]'s "which MIPS generation" axis. A plain `u32` bitmask (not the `bitflags`
/// crate — not a dependency here) so a caller can OR bits together or test membership in
/// several categories with one comparison, e.g. jitv2 gating logic reasoning about "is this
/// whole category supported" independent of the fine-grained per-instruction
/// [`InstrKind::has_jitv2_emitter`] check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InstrCategory(u32);

impl InstrCategory {
    pub const NONE: InstrCategory = InstrCategory(0);
    pub const ALU: InstrCategory = InstrCategory(1 << 0);
    pub const FPU: InstrCategory = InstrCategory(1 << 1);
    pub const BRANCH: InstrCategory = InstrCategory(1 << 2);
    pub const LOADSTORE: InstrCategory = InstrCategory(1 << 3);
    pub const COP0: InstrCategory = InstrCategory(1 << 4);
    pub const ALL: InstrCategory = InstrCategory(
        Self::ALU.0 | Self::FPU.0 | Self::BRANCH.0 | Self::LOADSTORE.0 | Self::COP0.0
    );

    pub const fn bits(self) -> u32 {
        self.0
    }

    pub const fn from_bits_truncate(bits: u32) -> InstrCategory {
        InstrCategory(bits & (Self::ALU.0 | Self::FPU.0 | Self::BRANCH.0 | Self::LOADSTORE.0 | Self::COP0.0))
    }

    pub const fn contains(self, other: InstrCategory) -> bool {
        (self.0 & other.0) == other.0
    }

    pub const fn intersects(self, other: InstrCategory) -> bool {
        (self.0 & other.0) != 0
    }
}

impl std::ops::BitOr for InstrCategory {
    type Output = InstrCategory;
    fn bitor(self, rhs: InstrCategory) -> InstrCategory {
        InstrCategory(self.0 | rhs.0)
    }
}

impl InstrKind {
    /// Classify into the ISA/functional bucket used for the frequency report.
    pub fn isa_class(self) -> IsaClass {
        use InstrKind::*;
        match self {
            Reserved => IsaClass::Reserved,

            // COP0 moves + TLB management
            Mfc0 | Dmfc0 | Mtc0 | Dmtc0 | Tlbr | Tlbwi | Tlbwr | Tlbp => IsaClass::Cop0,

            // MIPS III: 64-bit integer ops, 64-bit loads/stores, ERET
            Dsllv | Dsrlv | Dsrav | Dmult | Dmultu | Ddiv | Ddivu |
            Dadd | Daddu | Dsub | Dsubu | Dsll | Dsrl | Dsra | Dsll32 | Dsrl32 | Dsra32 |
            Daddi | Daddiu | Ldl | Ldr | Lwu | Sdl | Sdr | Ld | Sd | Lld | Scd |
            Eret => IsaClass::Mips3,

            // MIPS III FPU: 64-bit FPR move, long (*.l) convert/round/trunc/ceil/floor
            Dmfc1 | Dmtc1 |
            Fround_l_s | Ftrunc_l_s | Fceil_l_s | Ffloor_l_s | Fcvt_l_s |
            Fround_l_d | Ftrunc_l_d | Fceil_l_d | Ffloor_l_d | Fcvt_l_d |
            Fcvt_s_l | Fcvt_d_l => IsaClass::Mips3Fpu,

            // MIPS IV: conditional move, prefetch, indexed FP load/store
            Movci | Movz | Movn | Pref | Prefx | Lwxc1 | Ldxc1 | Swxc1 | Sdxc1 => IsaClass::Mips4,

            // MIPS IV FPU: conditional FP move, reciprocal/rsqrt estimate, fused madd/msub
            Fmovcf_s | Fmovz_s | Fmovn_s | Frecip_s | Frsqrt_s |
            Fmovcf_d | Fmovz_d | Fmovn_d | Frecip_d | Frsqrt_d |
            Madd_s | Madd_d | Msub_s | Msub_d | Nmadd_s | Nmadd_d | Nmsub_s | Nmsub_d => IsaClass::Mips4Fpu,

            // Everything else: MIPS I/II baseline (integer, branch, load/store, basic FPU)
            _ => IsaClass::Mips12,
        }
    }

    /// Coarse functional-unit bucket (ALU/FPU/branch/load-store/COP0/other), as a bitmask
    /// so a caller can test membership in several buckets at once (e.g. jitv2 gating logic
    /// that wants "ALU or load-store" as one check). This is a coarser, orthogonal axis to
    /// [`InstrKind::isa_class`] (which buckets by MIPS ISA generation, not functional unit) —
    /// e.g. `Dadd` is `IsaClass::Mips3` and `InstrCategory::ALU` at the same time.
    pub fn category(self) -> InstrCategory {
        use InstrKind::*;
        match self {
            Reserved => InstrCategory::NONE,

            Nop | Syscall | Break | Sync => InstrCategory::ALU,

            Jr | Jalr => InstrCategory::BRANCH,
            Bltz | Bgez | Bltzl | Bgezl | Bltzal | Bgezal | Bltzall | Bgezall
            | J | Jal | Beq | Bne | Blez | Bgtz | Beql | Bnel | Blezl | Bgtzl => InstrCategory::BRANCH,

            Mfc0 | Dmfc0 | Mtc0 | Dmtc0 | Tlbr | Tlbwi | Tlbwr | Tlbp | Eret => InstrCategory::COP0,

            Mfc1 | Dmfc1 | Cfc1 | Mtc1 | Dmtc1 | Ctc1 | Bc1
            | Fadd_s | Fsub_s | Fmul_s | Fdiv_s | Fsqrt_s | Fabs_s | Fmov_s | Fneg_s
            | Fround_l_s | Ftrunc_l_s | Fceil_l_s | Ffloor_l_s
            | Fround_w_s | Ftrunc_w_s | Fceil_w_s | Ffloor_w_s
            | Fmovcf_s | Fmovz_s | Fmovn_s | Frecip_s | Frsqrt_s
            | Fcvt_d_s | Fcvt_w_s | Fcvt_l_s | Fcc_s
            | Fadd_d | Fsub_d | Fmul_d | Fdiv_d | Fsqrt_d | Fabs_d | Fmov_d | Fneg_d
            | Fround_l_d | Ftrunc_l_d | Fceil_l_d | Ffloor_l_d
            | Fround_w_d | Ftrunc_w_d | Fceil_w_d | Ffloor_w_d
            | Fmovcf_d | Fmovz_d | Fmovn_d | Frecip_d | Frsqrt_d
            | Fcvt_s_d | Fcvt_w_d | Fcvt_l_d | Fcc_d
            | Fcvt_s_w | Fcvt_d_w | Fcvt_s_l | Fcvt_d_l
            | Madd_s | Madd_d | Msub_s | Msub_d | Nmadd_s | Nmadd_d | Nmsub_s | Nmsub_d
                => InstrCategory::FPU,

            Lb | Lh | Lwl | Lw | Lbu | Lhu | Lwr | Lwu
            | Sb | Sh | Swl | Sw | Sdl | Sdr | Swr | Cache
            | Ll | Ldl | Ldr | Ld | Sc | Sd | Pref | Lld | Scd
                => InstrCategory::LOADSTORE,
            // FPU loads/stores: both a memory op and CP1-adjacent (they're routed through
            // codegen's CP1 lookup table — see opcode_support.rs's has_cp1_emitter doc
            // comment) — tag with both bits.
            Lwxc1 | Ldxc1 | Swxc1 | Sdxc1 | Prefx
                => InstrCategory::from_bits_truncate(InstrCategory::LOADSTORE.bits() | InstrCategory::FPU.bits()),
            Lwc1 | Ldc1 | Swc1 | Sdc1
                => InstrCategory::from_bits_truncate(InstrCategory::LOADSTORE.bits() | InstrCategory::FPU.bits()),

            // Everything else: plain integer ALU (SPECIAL funct arithmetic/logic/shift/trap,
            // REGIMM trap-immediate, and the immediate-form ALU opcodes).
            _ => InstrCategory::ALU,
        }
    }

    /// True if `jitv2::codegen` has a real emitter for this instruction — the single source
    /// of truth backing `jitv2::opcode_support::has_emitter`. Every arm here must mirror
    /// `codegen.rs`'s `lookup_semantics`/`lookup_cp1_semantics` coverage exactly; this is
    /// intentionally a *coverage* question only ("is there an emitter"), not the emitter
    /// itself. Branches/jumps/register-indirect-jumps are handled by `analyzer::classify`'s
    /// own dispatch before this would ever be consulted for them (same caveat as the old
    /// `has_emitter`'s doc comment) — this only answers the Sequential-vs-Excluded question
    /// for everything else.
    pub fn has_jitv2_emitter(self) -> bool {
        use InstrKind::*;
        // Movz/Movn/Movci/Pref/Fmovcf_s/Fmovcf_d are MIPS IV — real jitv2
        // emitters exist (`lookup_semantics`), but on a non-`mips4` build
        // they must be excluded here so they fall back to the interpreter,
        // which raises Reserved Instruction (see `mips_exec.rs`'s decode
        // table). Keep this arm list in sync with that gate.
        if !cfg!(feature = "mips4")
            && matches!(self, Movz | Movn | Movci | Pref | Fmovcf_s | Fmovcf_d)
        {
            return false;
        }
        matches!(self,
            // Integer ALU (mirrors has_integer_emitter's OP_SPECIAL/OP_REGIMM/immediate list)
            Add | Addu | Sub | Subu | And | Or | Xor | Nor | Slt | Sltu
            | Sll | Srl | Sra | Sllv | Srlv | Srav
            | Mfhi | Mthi | Mflo | Mtlo
            | Mult | Multu | Div | Divu | Dmult | Dmultu | Ddiv | Ddivu
            | Dadd | Daddu | Dsub | Dsubu
            | Dsll | Dsrl | Dsra | Dsll32 | Dsrl32 | Dsra32
            | Dsllv | Dsrlv | Dsrav
            | Movz | Movn | Movci
            | Tge | Tgeu | Tlt | Tltu | Teq | Tne | Sync
            | Tgei | Tgeiu | Tlti | Tltiu | Teqi | Tnei
            | Addi | Addiu | Daddi | Daddiu | Slti | Sltiu | Andi | Ori | Xori | Lui
            | Lb | Lbu | Lh | Lhu | Lw | Lwu | Ld
            | Lwl | Lwr | Ldl | Ldr
            | Sb | Sh | Sw | Sd
            | Swl | Swr | Sdl | Sdr
            | Pref
            // CP1 (mirrors has_cp1_emitter's OP_LWC1/LDC1/SWC1/SDC1 + OP_COP1 coverage)
            | Lwc1 | Ldc1 | Swc1 | Sdc1
            | Fadd_s | Fsub_s | Fmul_s | Fdiv_s | Fsqrt_s
            | Fabs_s | Fneg_s | Fmov_s | Fmovcf_s
            | Fcvt_d_s | Fcvt_w_s | Fcvt_l_s
            | Fround_w_s | Ftrunc_w_s | Fceil_w_s | Ffloor_w_s
            | Fround_l_s | Ftrunc_l_s | Fceil_l_s | Ffloor_l_s
            | Fcc_s
            | Fadd_d | Fsub_d | Fmul_d | Fdiv_d | Fsqrt_d
            | Fabs_d | Fneg_d | Fmov_d | Fmovcf_d
            | Fcvt_s_d | Fcvt_w_d | Fcvt_l_d
            | Fround_w_d | Ftrunc_w_d | Fceil_w_d | Ffloor_w_d
            | Fround_l_d | Ftrunc_l_d | Fceil_l_d | Ffloor_l_d
            | Fcc_d
            | Fcvt_s_w | Fcvt_d_w | Fcvt_s_l | Fcvt_d_l
            | Mfc1 | Dmfc1 | Cfc1 | Mtc1 | Dmtc1 | Ctc1
        )
    }

    /// True if jitv2 can compile this instruction at all — `has_jitv2_emitter()`
    /// (the `lookup_semantics`/`lookup_cp1_semantics`-backed table) plus the
    /// branch/jump/register-indirect-jump kinds, which have their own emitters
    /// (`codegen::lookup_branch_or_jump`/`lookup_regjump`) but are deliberately
    /// excluded from `has_jitv2_emitter` since `analyzer::classify` resolves them
    /// by construction rather than an emitter-coverage lookup (see that method's
    /// doc comment). This is the set `opcode_support`'s per-instruction runtime
    /// toggle table (`ENABLED`) initializes to "on" — i.e. "supported today,
    /// absent any `j2 <instr>`/`j2 <category>` override."
    pub fn has_jitv2_support(self) -> bool {
        use InstrKind::*;
        matches!(self,
            Jr | Jalr
            | Bltz | Bgez | Bltzl | Bgezl | Bltzal | Bgezal | Bltzall | Bgezall
            | J | Jal | Beq | Bne | Blez | Bgtz | Beql | Bnel | Blezl | Bgtzl
        ) || self.has_jitv2_emitter()
    }

    pub fn name(self) -> &'static str {
        use InstrKind::*;
        match self {
            Reserved => "reserved", Nop => "nop",
            Sll => "sll", Movci => "movci", Srl => "srl", Sra => "sra",
            Sllv => "sllv", Srlv => "srlv", Srav => "srav", Jr => "jr", Jalr => "jalr",
            Movz => "movz", Movn => "movn",
            Syscall => "syscall", Break => "break", Sync => "sync",
            Mfhi => "mfhi", Mthi => "mthi", Mflo => "mflo", Mtlo => "mtlo",
            Dsllv => "dsllv", Dsrlv => "dsrlv", Dsrav => "dsrav",
            Mult => "mult", Multu => "multu", Div => "div", Divu => "divu",
            Dmult => "dmult", Dmultu => "dmultu", Ddiv => "ddiv", Ddivu => "ddivu",
            Add => "add", Addu => "addu", Sub => "sub", Subu => "subu",
            And => "and", Or => "or", Xor => "xor", Nor => "nor",
            Slt => "slt", Sltu => "sltu",
            Dadd => "dadd", Daddu => "daddu", Dsub => "dsub", Dsubu => "dsubu",
            Tge => "tge", Tgeu => "tgeu", Tlt => "tlt", Tltu => "tltu",
            Teq => "teq", Tne => "tne",
            Dsll => "dsll", Dsrl => "dsrl", Dsra => "dsra",
            Dsll32 => "dsll32", Dsrl32 => "dsrl32", Dsra32 => "dsra32",
            Bltz => "bltz", Bgez => "bgez", Bltzl => "bltzl", Bgezl => "bgezl",
            Tgei => "tgei", Tgeiu => "tgeiu", Tlti => "tlti", Tltiu => "tltiu",
            Teqi => "teqi", Tnei => "tnei",
            Bltzal => "bltzal", Bgezal => "bgezal", Bltzall => "bltzall", Bgezall => "bgezall",
            J => "j", Jal => "jal", Beq => "beq", Bne => "bne",
            Blez => "blez", Bgtz => "bgtz", Beql => "beql", Bnel => "bnel",
            Blezl => "blezl", Bgtzl => "bgtzl",
            Addi => "addi", Addiu => "addiu", Daddi => "daddi", Daddiu => "daddiu",
            Slti => "slti", Sltiu => "sltiu", Andi => "andi", Ori => "ori",
            Xori => "xori", Lui => "lui",
            Mfc0 => "mfc0", Dmfc0 => "dmfc0", Mtc0 => "mtc0", Dmtc0 => "dmtc0",
            Tlbr => "tlbr", Tlbwi => "tlbwi", Tlbwr => "tlbwr", Tlbp => "tlbp", Eret => "eret",
            Mfc1 => "mfc1", Dmfc1 => "dmfc1", Cfc1 => "cfc1",
            Mtc1 => "mtc1", Dmtc1 => "dmtc1", Ctc1 => "ctc1", Bc1 => "bc1",
            Fadd_s => "fadd.s", Fsub_s => "fsub.s", Fmul_s => "fmul.s", Fdiv_s => "fdiv.s",
            Fsqrt_s => "fsqrt.s", Fabs_s => "fabs.s", Fmov_s => "fmov.s", Fneg_s => "fneg.s",
            Fround_l_s => "fround.l.s", Ftrunc_l_s => "ftrunc.l.s",
            Fceil_l_s => "fceil.l.s", Ffloor_l_s => "ffloor.l.s",
            Fround_w_s => "fround.w.s", Ftrunc_w_s => "ftrunc.w.s",
            Fceil_w_s => "fceil.w.s", Ffloor_w_s => "ffloor.w.s",
            Fmovcf_s => "fmovcf.s", Fmovz_s => "fmovz.s", Fmovn_s => "fmovn.s",
            Frecip_s => "frecip.s", Frsqrt_s => "frsqrt.s",
            Fcvt_d_s => "fcvt.d.s", Fcvt_w_s => "fcvt.w.s", Fcvt_l_s => "fcvt.l.s", Fcc_s => "fc.cond.s",
            Fadd_d => "fadd.d", Fsub_d => "fsub.d", Fmul_d => "fmul.d", Fdiv_d => "fdiv.d",
            Fsqrt_d => "fsqrt.d", Fabs_d => "fabs.d", Fmov_d => "fmov.d", Fneg_d => "fneg.d",
            Fround_l_d => "fround.l.d", Ftrunc_l_d => "ftrunc.l.d",
            Fceil_l_d => "fceil.l.d", Ffloor_l_d => "ffloor.l.d",
            Fround_w_d => "fround.w.d", Ftrunc_w_d => "ftrunc.w.d",
            Fceil_w_d => "fceil.w.d", Ffloor_w_d => "ffloor.w.d",
            Fmovcf_d => "fmovcf.d", Fmovz_d => "fmovz.d", Fmovn_d => "fmovn.d",
            Frecip_d => "frecip.d", Frsqrt_d => "frsqrt.d",
            Fcvt_s_d => "fcvt.s.d", Fcvt_w_d => "fcvt.w.d", Fcvt_l_d => "fcvt.l.d", Fcc_d => "fc.cond.d",
            Fcvt_s_w => "fcvt.s.w", Fcvt_d_w => "fcvt.d.w",
            Fcvt_s_l => "fcvt.s.l", Fcvt_d_l => "fcvt.d.l",
            Lwxc1 => "lwxc1", Ldxc1 => "ldxc1", Swxc1 => "swxc1", Sdxc1 => "sdxc1", Prefx => "prefx",
            Madd_s => "madd.s", Madd_d => "madd.d", Msub_s => "msub.s", Msub_d => "msub.d",
            Nmadd_s => "nmadd.s", Nmadd_d => "nmadd.d", Nmsub_s => "nmsub.s", Nmsub_d => "nmsub.d",
            Lb => "lb", Lh => "lh", Lwl => "lwl", Lw => "lw",
            Lbu => "lbu", Lhu => "lhu", Lwr => "lwr", Lwu => "lwu",
            Sb => "sb", Sh => "sh", Swl => "swl", Sw => "sw",
            Sdl => "sdl", Sdr => "sdr", Swr => "swr", Cache => "cache",
            Ll => "ll", Lwc1 => "lwc1", Ldc1 => "ldc1", Ldl => "ldl", Ldr => "ldr",
            Ld => "ld", Sc => "sc", Swc1 => "swc1", Sdc1 => "sdc1", Sd => "sd",
            Pref => "pref", Lld => "lld", Scd => "scd",
        }
    }
}

/// Classify a decoded instruction from its raw fields. Mirrors the match
/// structure of `decode_into` in mips_exec.rs exactly — keep both in sync.
#[inline]
pub fn classify_instr(op: u8, rs: u8, rt: u8, funct: u8) -> InstrKind {
    use InstrKind::*;
    match op as u32 {
        OP_SPECIAL => match funct as u32 {
            FUNCT_SLL => Sll, FUNCT_MOVCI => Movci, FUNCT_SRL => Srl, FUNCT_SRA => Sra,
            FUNCT_SLLV => Sllv, FUNCT_SRLV => Srlv, FUNCT_SRAV => Srav,
            FUNCT_JR => Jr, FUNCT_JALR => Jalr, FUNCT_MOVZ => Movz, FUNCT_MOVN => Movn,
            FUNCT_SYSCALL => Syscall, FUNCT_BREAK => Break, FUNCT_SYNC => Sync,
            FUNCT_MFHI => Mfhi, FUNCT_MTHI => Mthi, FUNCT_MFLO => Mflo, FUNCT_MTLO => Mtlo,
            FUNCT_DSLLV => Dsllv, FUNCT_DSRLV => Dsrlv, FUNCT_DSRAV => Dsrav,
            FUNCT_MULT => Mult, FUNCT_MULTU => Multu, FUNCT_DIV => Div, FUNCT_DIVU => Divu,
            FUNCT_DMULT => Dmult, FUNCT_DMULTU => Dmultu, FUNCT_DDIV => Ddiv, FUNCT_DDIVU => Ddivu,
            FUNCT_ADD => Add, FUNCT_ADDU => Addu, FUNCT_SUB => Sub, FUNCT_SUBU => Subu,
            FUNCT_AND => And, FUNCT_OR => Or, FUNCT_XOR => Xor, FUNCT_NOR => Nor,
            FUNCT_SLT => Slt, FUNCT_SLTU => Sltu,
            FUNCT_DADD => Dadd, FUNCT_DADDU => Daddu, FUNCT_DSUB => Dsub, FUNCT_DSUBU => Dsubu,
            FUNCT_TGE => Tge, FUNCT_TGEU => Tgeu, FUNCT_TLT => Tlt, FUNCT_TLTU => Tltu,
            FUNCT_TEQ => Teq, FUNCT_TNE => Tne,
            FUNCT_DSLL => Dsll, FUNCT_DSRL => Dsrl, FUNCT_DSRA => Dsra,
            FUNCT_DSLL32 => Dsll32, FUNCT_DSRL32 => Dsrl32, FUNCT_DSRA32 => Dsra32,
            _ => Reserved,
        },
        OP_REGIMM => match rt as u32 {
            RT_BLTZ => Bltz, RT_BGEZ => Bgez, RT_BLTZL => Bltzl, RT_BGEZL => Bgezl,
            RT_TGEI => Tgei, RT_TGEIU => Tgeiu, RT_TLTI => Tlti, RT_TLTIU => Tltiu,
            RT_TEQI => Teqi, RT_TNEI => Tnei,
            RT_BLTZAL => Bltzal, RT_BGEZAL => Bgezal, RT_BLTZALL => Bltzall, RT_BGEZALL => Bgezall,
            _ => Reserved,
        },
        OP_J => J, OP_JAL => Jal,
        OP_BEQ => Beq, OP_BNE => Bne, OP_BLEZ => Blez, OP_BGTZ => Bgtz,
        OP_BEQL => Beql, OP_BNEL => Bnel, OP_BLEZL => Blezl, OP_BGTZL => Bgtzl,
        OP_ADDI => Addi, OP_ADDIU => Addiu, OP_DADDI => Daddi, OP_DADDIU => Daddiu,
        OP_SLTI => Slti, OP_SLTIU => Sltiu,
        OP_ANDI => Andi, OP_ORI => Ori, OP_XORI => Xori, OP_LUI => Lui,
        OP_COP0 => match rs as u32 {
            RS_MFC0 => Mfc0, RS_DMFC0 => Dmfc0, RS_MTC0 => Mtc0, RS_DMTC0 => Dmtc0,
            RS_TLB => match funct as u32 {
                FUNCT_TLBR => Tlbr, FUNCT_TLBWI => Tlbwi, FUNCT_TLBWR => Tlbwr,
                FUNCT_TLBP => Tlbp, FUNCT_ERET => Eret,
                FUNCT_WAIT => Nop,
                _ => Reserved,
            },
            _ => Reserved,
        },
        OP_COP1 => match rs as u32 {
            RS_MFC1 => Mfc1, RS_DMFC1 => Dmfc1, RS_CFC1 => Cfc1,
            RS_MTC1 => Mtc1, RS_DMTC1 => Dmtc1, RS_CTC1 => Ctc1, RS_BC1 => Bc1,
            RS_S => match funct as u32 {
                FUNCT_FADD => Fadd_s, FUNCT_FSUB => Fsub_s, FUNCT_FMUL => Fmul_s, FUNCT_FDIV => Fdiv_s,
                FUNCT_FSQRT => Fsqrt_s, FUNCT_FABS => Fabs_s, FUNCT_FMOV => Fmov_s, FUNCT_FNEG => Fneg_s,
                FUNCT_FROUND_L => Fround_l_s, FUNCT_FTRUNC_L => Ftrunc_l_s,
                FUNCT_FCEIL_L => Fceil_l_s, FUNCT_FFLOOR_L => Ffloor_l_s,
                FUNCT_FROUND_W => Fround_w_s, FUNCT_FTRUNC_W => Ftrunc_w_s,
                FUNCT_FCEIL_W => Fceil_w_s, FUNCT_FFLOOR_W => Ffloor_w_s,
                FUNCT_FMOVCF => Fmovcf_s, FUNCT_FMOVZ => Fmovz_s, FUNCT_FMOVN => Fmovn_s,
                FUNCT_FRECIP => Frecip_s, FUNCT_FRSQRT => Frsqrt_s,
                FUNCT_FCVT_D => Fcvt_d_s, FUNCT_FCVT_W => Fcvt_w_s, FUNCT_FCVT_L => Fcvt_l_s,
                FUNCT_FC_F..=FUNCT_FC_NGT => Fcc_s,
                _ => Reserved,
            },
            RS_D => match funct as u32 {
                FUNCT_FADD => Fadd_d, FUNCT_FSUB => Fsub_d, FUNCT_FMUL => Fmul_d, FUNCT_FDIV => Fdiv_d,
                FUNCT_FSQRT => Fsqrt_d, FUNCT_FABS => Fabs_d, FUNCT_FMOV => Fmov_d, FUNCT_FNEG => Fneg_d,
                FUNCT_FROUND_L => Fround_l_d, FUNCT_FTRUNC_L => Ftrunc_l_d,
                FUNCT_FCEIL_L => Fceil_l_d, FUNCT_FFLOOR_L => Ffloor_l_d,
                FUNCT_FROUND_W => Fround_w_d, FUNCT_FTRUNC_W => Ftrunc_w_d,
                FUNCT_FCEIL_W => Fceil_w_d, FUNCT_FFLOOR_W => Ffloor_w_d,
                FUNCT_FMOVCF => Fmovcf_d, FUNCT_FMOVZ => Fmovz_d, FUNCT_FMOVN => Fmovn_d,
                FUNCT_FRECIP => Frecip_d, FUNCT_FRSQRT => Frsqrt_d,
                FUNCT_FCVT_S => Fcvt_s_d, FUNCT_FCVT_W => Fcvt_w_d, FUNCT_FCVT_L => Fcvt_l_d,
                FUNCT_FC_F..=FUNCT_FC_NGT => Fcc_d,
                _ => Reserved,
            },
            RS_W => match funct as u32 {
                FUNCT_FCVT_S => Fcvt_s_w, FUNCT_FCVT_D => Fcvt_d_w,
                _ => Reserved,
            },
            RS_L => match funct as u32 {
                FUNCT_FCVT_S => Fcvt_s_l, FUNCT_FCVT_D => Fcvt_d_l,
                _ => Reserved,
            },
            _ => Reserved,
        },
        OP_COP1X => match funct as u32 {
            FUNCT_LWXC1 => Lwxc1, FUNCT_LDXC1 => Ldxc1, FUNCT_SWXC1 => Swxc1, FUNCT_SDXC1 => Sdxc1,
            FUNCT_PREFX => Prefx,
            FUNCT_MADD_S => Madd_s, FUNCT_MADD_D => Madd_d,
            FUNCT_MSUB_S => Msub_s, FUNCT_MSUB_D => Msub_d,
            FUNCT_NMADD_S => Nmadd_s, FUNCT_NMADD_D => Nmadd_d,
            FUNCT_NMSUB_S => Nmsub_s, FUNCT_NMSUB_D => Nmsub_d,
            _ => Reserved,
        },
        OP_LB => Lb, OP_LH => Lh, OP_LWL => Lwl, OP_LW => Lw,
        OP_LBU => Lbu, OP_LHU => Lhu, OP_LWR => Lwr, OP_LWU => Lwu,
        OP_SB => Sb, OP_SH => Sh, OP_SWL => Swl, OP_SW => Sw,
        OP_SDL => Sdl, OP_SDR => Sdr, OP_SWR => Swr, OP_CACHE => Cache,
        OP_LL => Ll, OP_LWC1 => Lwc1, OP_LDC1 => Ldc1, OP_LDL => Ldl, OP_LDR => Ldr, OP_LD => Ld,
        OP_SC => Sc, OP_SWC1 => Swc1, OP_SDC1 => Sdc1, OP_SD => Sd,
        OP_PREF => Pref, OP_LLD => Lld, OP_SCD => Scd,
        _ => Reserved,
    }
}

/// Execution frequency counters, one slot per `InstrKind`. Also tracks
/// decode-time counts separately from execution-time counts: `decode_into`
/// (mips_exec.rs) only runs once per L1I/L2 cache fill, while `exec_decoded`
/// runs once per dispatch, so "how many times was this decoded" and "how
/// many times was this executed" are genuinely different numbers for any
/// instruction inside a loop.
///
/// Note on scope: `exec_counts` only sees interpreter-path dispatches. Once
/// jitv2 has compiled a region, `exec_decoded`'s JIT-hit path calls the
/// native function pointer directly and returns without recording anything
/// here (mips_exec.rs's `exec_decoded`), and the `lightning`-build fast path
/// skips decode/exec entirely on a hit. So under `jitv2`, `exec_counts`
/// undercounts actual guest execution once compilation kicks in — expected,
/// not a bug. `decode_counts` is unaffected (decoding still happens the
/// first time a JIT-compiled region's underlying words are read into the
/// cache, before compilation).
#[cfg(feature = "instr_stats")]
#[derive(Clone)]
pub struct InstrStats {
    pub exec_counts: Box<[u64; NUM_INSTR_KINDS]>,
    pub decode_counts: Box<[u64; NUM_INSTR_KINDS]>,
    /// Raw 32-bit instruction word -> hit count, populated only for `InstrKind::Reserved`
    /// (executed path only). `InstrKind::Reserved` lumps every illegal encoding into one
    /// counter slot; this map recovers which distinct encodings were actually hit for the
    /// aggregate report.
    pub reserved_words: std::collections::HashMap<u32, u64>,
}

#[cfg(feature = "instr_stats")]
impl Default for InstrStats {
    fn default() -> Self {
        Self {
            exec_counts: Box::new([0u64; NUM_INSTR_KINDS]),
            decode_counts: Box::new([0u64; NUM_INSTR_KINDS]),
            reserved_words: std::collections::HashMap::new(),
        }
    }
}

#[cfg(feature = "instr_stats")]
impl InstrStats {
    #[inline(always)]
    pub fn record(&mut self, op: u8, rs: u8, rt: u8, funct: u8, raw: u32) {
        let kind = classify_instr(op, rs, rt, funct);
        self.exec_counts[kind as usize] += 1;
        if kind == InstrKind::Reserved {
            *self.reserved_words.entry(raw).or_insert(0) += 1;
        }
    }

    /// Record a decode (cache fill), separate from `record`'s execution count — see the
    /// struct doc comment. Called from the `decode_into::<T, C>(d)` call site (mips_exec.rs),
    /// not from inside `decode_into` itself, so `decode_into`'s signature stays unchanged.
    #[inline(always)]
    pub fn record_decode(&mut self, op: u8, rs: u8, rt: u8, funct: u8) {
        let kind = classify_instr(op, rs, rt, funct);
        self.decode_counts[kind as usize] += 1;
    }

    pub fn total(&self) -> u64 {
        self.exec_counts.iter().sum()
    }

    /// Reset all counters to zero (monitor `cpu instrstats clear`).
    pub fn clear(&mut self) {
        self.exec_counts.fill(0);
        self.decode_counts.fill(0);
        self.reserved_words.clear();
    }

    /// Print a frequency-sorted report to stderr, grouped by ISA class
    /// (MIPS I/II, MIPS III, MIPS III FPU, COP0, MIPS IV, MIPS IV FPU).
    pub fn print(&self) {
        let mut out = std::io::stderr();
        let _ = self.write_report(&mut out);
    }

    /// Write the same report to an arbitrary sink (used by the monitor `cpu instrstats` command).
    /// Reports `exec_counts` (interpreter-path executions — see the struct doc comment for the
    /// jitv2-bypass caveat).
    pub fn write_report(&self, w: &mut dyn std::io::Write) -> std::io::Result<()> {
        let total = self.total();
        if total == 0 {
            writeln!(w, "instrstats: no instructions recorded yet")?;
            return Ok(());
        }

        let rows: Vec<(InstrKind, u64)> = (0..NUM_INSTR_KINDS)
            .filter_map(|k| {
                let count = self.exec_counts[k];
                if count == 0 { return None; }
                // SAFETY: k is in range 0..NUM_INSTR_KINDS, which matches the enum's contiguous discriminants.
                let kind: InstrKind = unsafe { std::mem::transmute(k as u16) };
                Some((kind, count))
            })
            .collect();

        writeln!(w, "\n=== MIPS Instruction Frequency Statistics ===")?;
        writeln!(w, "  Total instructions executed (interpreter path only — see jitv2 caveat in InstrStats doc comment): {}", total)?;

        const CLASSES: &[IsaClass] = &[
            IsaClass::Mips12, IsaClass::Mips3, IsaClass::Mips3Fpu,
            IsaClass::Cop0, IsaClass::Mips4, IsaClass::Mips4Fpu, IsaClass::Reserved,
        ];

        writeln!(w, "\n  By ISA class:")?;
        for &class in CLASSES {
            let class_total: u64 = rows.iter().filter(|(k, _)| k.isa_class() == class).map(|(_, c)| c).sum();
            if class_total == 0 { continue; }
            writeln!(w, "    {:<42} {:>14}  {:>6.2}%", class.label(), class_total, pct(class_total, total))?;
        }

        for &class in CLASSES {
            if class == IsaClass::Mips12 { continue; } // baseline traffic isn't the diagnostic target
            let mut class_rows: Vec<_> = rows.iter().filter(|(k, _)| k.isa_class() == class).cloned().collect();
            if class_rows.is_empty() { continue; }
            class_rows.sort_by(|a, b| b.1.cmp(&a.1));
            writeln!(w, "\n  {} instructions used:", class.label())?;
            for (kind, count) in &class_rows {
                writeln!(w, "    {:<14} {:>14}  {:>6.2}%", kind.name(), count, pct(*count, total))?;
            }
        }

        if !self.reserved_words.is_empty() {
            let mut words: Vec<(u32, u64)> = self.reserved_words.iter().map(|(&w, &c)| (w, c)).collect();
            words.sort_by(|a, b| b.1.cmp(&a.1));
            writeln!(w, "\n  Reserved/illegal instructions by encoding:")?;
            for (raw, count) in &words {
                let op = (raw >> 26) & 0x3F;
                let rs = (raw >> 21) & 0x1F;
                let rt = (raw >> 16) & 0x1F;
                let funct = raw & 0x3F;
                writeln!(w, "    raw={:08x}  count={:<8} op={:#04x} rs={:#04x} rt={:#04x} funct={:#04x}",
                    raw, count, op, rs, rt, funct)?;
            }
        }

        let mut top_rows = rows.clone();
        top_rows.sort_by(|a, b| b.1.cmp(&a.1));
        writeln!(w, "\n  Top instructions overall:")?;
        for (kind, count) in top_rows.iter().take(40) {
            writeln!(w, "    {:<14} {:>14}  {:>6.2}%  [{}]", kind.name(), count, pct(*count, total), kind.isa_class().label())?;
        }
        Ok(())
    }

    /// Diffable dump: mnemonics with a nonzero decode OR exec count, one per line,
    /// alphabetically sorted so the file diffs cleanly across runs regardless of
    /// execution-order/timing noise. Written at CPU stop as `instr_used.txt`.
    pub fn write_used_list(&self, w: &mut dyn std::io::Write) -> std::io::Result<()> {
        let mut names: Vec<&'static str> = (0..NUM_INSTR_KINDS)
            .filter(|&k| self.exec_counts[k] != 0 || self.decode_counts[k] != 0)
            .map(|k| {
                // SAFETY: k is in range 0..NUM_INSTR_KINDS, matching the enum's contiguous discriminants.
                let kind: InstrKind = unsafe { std::mem::transmute(k as u16) };
                kind.name()
            })
            .collect();
        names.sort_unstable();
        for name in names {
            writeln!(w, "{}", name)?;
        }
        Ok(())
    }

    /// Diffable dump: mnemonic, decode count, exec count — one per line, alphabetically
    /// sorted by mnemonic (not by frequency, unlike `write_report`) so line-for-line diffs
    /// against a previous run's dump are meaningful. Written at CPU stop as `instr_counts.txt`.
    pub fn write_counts(&self, w: &mut dyn std::io::Write) -> std::io::Result<()> {
        let mut rows: Vec<(&'static str, u64, u64)> = (0..NUM_INSTR_KINDS)
            .filter(|&k| self.exec_counts[k] != 0 || self.decode_counts[k] != 0)
            .map(|k| {
                // SAFETY: k is in range 0..NUM_INSTR_KINDS, matching the enum's contiguous discriminants.
                let kind: InstrKind = unsafe { std::mem::transmute(k as u16) };
                (kind.name(), self.decode_counts[k], self.exec_counts[k])
            })
            .collect();
        rows.sort_unstable_by_key(|&(name, _, _)| name);
        writeln!(w, "# mnemonic decoded executed")?;
        for (name, decoded, executed) in rows {
            writeln!(w, "{:<14} {:>14} {:>14}", name, decoded, executed)?;
        }
        Ok(())
    }
}

fn pct(num: u64, den: u64) -> f64 {
    if den == 0 { 0.0 } else { num as f64 / den as f64 * 100.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r_type(op: u32, rs: u32, rt: u32, rd: u32, sa: u32, funct: u32) -> u32 {
        (op << 26) | (rs << 21) | (rt << 16) | (rd << 11) | (sa << 6) | funct
    }
    fn i_type(op: u32, rs: u32, rt: u32, imm: u16) -> u32 {
        (op << 26) | (rs << 21) | (rt << 16) | imm as u32
    }
    fn fields(raw: u32) -> (u8, u8, u8, u8) {
        (
            ((raw >> 26) & 0x3F) as u8,
            ((raw >> 21) & 0x1F) as u8,
            ((raw >> 16) & 0x1F) as u8,
            (raw & 0x3F) as u8,
        )
    }

    /// Differential regression guard: for a representative sample of encodings covering
    /// every top-level opcode/funct group `decode_into` (mips_exec.rs) dispatches through,
    /// `classify_instr` must never fall back to `Reserved` for an encoding that has a real
    /// handler. This doesn't replace keeping the two match trees in sync by hand (the user
    /// explicitly chose to keep them as two trees, not merge them), but it catches the
    /// common failure mode: a new opcode added to `decode_into` and forgotten here.
    #[test]
    fn classify_instr_never_reserved_for_known_encodings() {
        let samples: &[u32] = &[
            r_type(OP_SPECIAL, 1, 2, 3, 0, FUNCT_ADDU),
            r_type(OP_SPECIAL, 1, 2, 3, 0, FUNCT_MOVZ),
            r_type(OP_SPECIAL, 1, 2, 0, 0, FUNCT_JR),
            r_type(OP_SPECIAL, 1, 0, 0, 0, FUNCT_SYSCALL),
            r_type(OP_SPECIAL, 1, 0, 0, 0, FUNCT_DSRA32),
            i_type(OP_REGIMM, 1, RT_BLTZ, 0),
            i_type(OP_REGIMM, 1, RT_TGEI, 5),
            i_type(OP_J, 0, 0, 0) | 1, // J with a nonzero target field
            i_type(OP_BEQ, 1, 2, 0),
            i_type(OP_ADDIU, 1, 2, 5),
            i_type(OP_DADDIU, 1, 2, 5),
            i_type(OP_LW, 1, 2, 0),
            i_type(OP_SD, 1, 2, 0),
            r_type(OP_COP0, RS_MFC0, 2, 3, 0, 0),
            r_type(OP_COP0, RS_TLB, 0, 0, 0, FUNCT_TLBWI),
            r_type(OP_COP0, RS_TLB, 0, 0, 0, FUNCT_ERET),
            r_type(OP_COP1, RS_MFC1, 2, 3, 0, 0),
            r_type(OP_COP1, RS_S, 2, 3, 0, FUNCT_FADD),
            r_type(OP_COP1, RS_D, 2, 3, 0, FUNCT_FMOVZ),
            r_type(OP_COP1, RS_W, 2, 3, 0, FUNCT_FCVT_S),
            r_type(OP_COP1, RS_BC1, 0, 0, 0, 0),
            r_type(OP_COP1X, 1, 2, 3, 0, FUNCT_LWXC1),
            r_type(OP_COP1X, 1, 2, 3, 0, FUNCT_MADD_S),
        ];
        for &raw in samples {
            let (op, rs, rt, funct) = fields(raw);
            let kind = classify_instr(op, rs, rt, funct);
            assert_ne!(kind, InstrKind::Reserved, "raw={:08x} op={:#04x} rs={:#04x} rt={:#04x} funct={:#04x} classified as Reserved", raw, op, rs, rt, funct);
        }
    }

    #[test]
    fn category_loadstore_fpu_overlap_for_indexed_fp_mem_ops() {
        assert!(InstrKind::Lwxc1.category().contains(InstrCategory::LOADSTORE));
        assert!(InstrKind::Lwxc1.category().contains(InstrCategory::FPU));
        assert!(InstrKind::Lwc1.category().contains(InstrCategory::LOADSTORE));
        assert!(InstrKind::Lwc1.category().contains(InstrCategory::FPU));
    }

    #[test]
    fn category_plain_alu_and_branch() {
        assert_eq!(InstrKind::Addu.category(), InstrCategory::ALU);
        assert_eq!(InstrKind::Beq.category(), InstrCategory::BRANCH);
        assert_eq!(InstrKind::Jr.category(), InstrCategory::BRANCH);
        assert_eq!(InstrKind::Mfc0.category(), InstrCategory::COP0);
        assert_eq!(InstrKind::Reserved.category(), InstrCategory::NONE);
    }

    #[cfg(feature = "instr_stats")]
    #[test]
    fn instr_stats_record_and_dump_roundtrip() {
        let mut stats = InstrStats::default();
        let (op, rs, rt, funct) = fields(r_type(OP_SPECIAL, 1, 2, 3, 0, FUNCT_ADDU));
        stats.record(op, rs, rt, funct, r_type(OP_SPECIAL, 1, 2, 3, 0, FUNCT_ADDU));
        stats.record_decode(op, rs, rt, funct);
        stats.record_decode(op, rs, rt, funct);

        assert_eq!(stats.exec_counts[InstrKind::Addu as usize], 1);
        assert_eq!(stats.decode_counts[InstrKind::Addu as usize], 2);

        let mut used = Vec::new();
        stats.write_used_list(&mut used).unwrap();
        assert_eq!(String::from_utf8(used).unwrap(), "addu\n");

        let mut counts = Vec::new();
        stats.write_counts(&mut counts).unwrap();
        let text = String::from_utf8(counts).unwrap();
        assert!(text.contains("addu"));
        assert!(text.lines().next().unwrap().starts_with('#'));
    }
}
