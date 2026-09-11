// MIPS3 + MIPS4 ISA Definitions

// Primary Opcode (OP) - Bits 31..26
pub const OP_SPECIAL: u32 = 0x00;
pub const OP_REGIMM: u32 = 0x01;
pub const OP_J: u32 = 0x02;
pub const OP_JAL: u32 = 0x03;
pub const OP_BEQ: u32 = 0x04;
pub const OP_BNE: u32 = 0x05;
pub const OP_BLEZ: u32 = 0x06;
pub const OP_BGTZ: u32 = 0x07;
pub const OP_ADDI: u32 = 0x08;
pub const OP_ADDIU: u32 = 0x09;
pub const OP_SLTI: u32 = 0x0A;
pub const OP_SLTIU: u32 = 0x0B;
pub const OP_ANDI: u32 = 0x0C;
pub const OP_ORI: u32 = 0x0D;
pub const OP_XORI: u32 = 0x0E;
pub const OP_LUI: u32 = 0x0F;
pub const OP_COP0: u32 = 0x10;
pub const OP_COP1: u32 = 0x11;
pub const OP_COP2: u32 = 0x12;
pub const OP_COP1X: u32 = 0x13; // MIPS IV
pub const OP_BEQL: u32 = 0x14;
pub const OP_BNEL: u32 = 0x15;
pub const OP_BLEZL: u32 = 0x16;
pub const OP_BGTZL: u32 = 0x17;
pub const OP_DADDI: u32 = 0x18; // MIPS III
pub const OP_DADDIU: u32 = 0x19; // MIPS III
pub const OP_LDL: u32 = 0x1A; // MIPS III
pub const OP_LDR: u32 = 0x1B; // MIPS III
pub const OP_LB: u32 = 0x20;
pub const OP_LH: u32 = 0x21;
pub const OP_LWL: u32 = 0x22;
pub const OP_LW: u32 = 0x23;
pub const OP_LBU: u32 = 0x24;
pub const OP_LHU: u32 = 0x25;
pub const OP_LWR: u32 = 0x26;
pub const OP_LWU: u32 = 0x27; // MIPS III
pub const OP_SB: u32 = 0x28;
pub const OP_SH: u32 = 0x29;
pub const OP_SWL: u32 = 0x2A;
pub const OP_SW: u32 = 0x2B;
pub const OP_SDL: u32 = 0x2C; // MIPS III
pub const OP_SDR: u32 = 0x2D; // MIPS III
pub const OP_SWR: u32 = 0x2E;
pub const OP_CACHE: u32 = 0x2F;
pub const OP_LL: u32 = 0x30;
pub const OP_LWC1: u32 = 0x31;
pub const OP_LWC2: u32 = 0x32;
pub const OP_PREF: u32 = 0x33; // MIPS IV
pub const OP_LLD: u32 = 0x34; // MIPS III
pub const OP_LDC1: u32 = 0x35;
pub const OP_LDC2: u32 = 0x36;
pub const OP_LD: u32 = 0x37; // MIPS III
pub const OP_SC: u32 = 0x38;
pub const OP_SWC1: u32 = 0x39;
pub const OP_SWC2: u32 = 0x3A;
pub const OP_SCD: u32 = 0x3C; // MIPS III
pub const OP_SDC1: u32 = 0x3D;
pub const OP_SDC2: u32 = 0x3E;
pub const OP_SD: u32 = 0x3F; // MIPS III

/// JIT region-boundary sentinel — a reserved primary opcode (0x1E, unused by
/// the R4000 and by this emulator's decoder) with a distinctive low pattern,
/// recognized by the jitv2 analyzer (`classify` -> `Classify::RegionBoundary`)
/// as a hard region end: never visited, never compiled, never run through the
/// interpreter fallback. Exists so tests (and corpus tooling) that need a
/// deterministic, zero-side-effect, no-delay-slot region terminator have one —
/// now that ordinary unsupported/`Excluded` instructions are kept in the region
/// as interpreter-fallback heads (`CompiledInstr::is_fallback`) instead of
/// ending it. The interpreter would treat this word as a Reserved Instruction
/// (exec_reserved) if it ever actually executed it, but a correct region never
/// reaches it: the analyzer stops the walk at the predecessor's edge, exactly
/// as it used to for `Excluded` words before fallback existed.
pub const JIT_REGION_BOUNDARY_SENTINEL: u32 = 0x7800_0B0B;

// SPECIAL Function Codes (FUNCT) - Bits 5..0
pub const FUNCT_SLL: u32 = 0x00;
pub const FUNCT_MOVCI: u32 = 0x01; // MIPS IV (MOVCI)
pub const FUNCT_SRL: u32 = 0x02;
pub const FUNCT_SRA: u32 = 0x03;
pub const FUNCT_SLLV: u32 = 0x04;
pub const FUNCT_SRLV: u32 = 0x06;
pub const FUNCT_SRAV: u32 = 0x07;
pub const FUNCT_JR: u32 = 0x08;
pub const FUNCT_JALR: u32 = 0x09;
pub const FUNCT_MOVZ: u32 = 0x0A; // MIPS IV
pub const FUNCT_MOVN: u32 = 0x0B; // MIPS IV
pub const FUNCT_SYSCALL: u32 = 0x0C;
pub const FUNCT_BREAK: u32 = 0x0D;
pub const FUNCT_SYNC: u32 = 0x0F;
pub const FUNCT_MFHI: u32 = 0x10;
pub const FUNCT_MTHI: u32 = 0x11;
pub const FUNCT_MFLO: u32 = 0x12;
pub const FUNCT_MTLO: u32 = 0x13;
pub const FUNCT_DSLLV: u32 = 0x14; // MIPS III
pub const FUNCT_DSRLV: u32 = 0x16; // MIPS III
pub const FUNCT_DSRAV: u32 = 0x17; // MIPS III
pub const FUNCT_MULT: u32 = 0x18;
pub const FUNCT_MULTU: u32 = 0x19;
pub const FUNCT_DIV: u32 = 0x1A;
pub const FUNCT_DIVU: u32 = 0x1B;
pub const FUNCT_DMULT: u32 = 0x1C; // MIPS III
pub const FUNCT_DMULTU: u32 = 0x1D; // MIPS III
pub const FUNCT_DDIV: u32 = 0x1E; // MIPS III
pub const FUNCT_DDIVU: u32 = 0x1F; // MIPS III
pub const FUNCT_ADD: u32 = 0x20;
pub const FUNCT_ADDU: u32 = 0x21;
pub const FUNCT_SUB: u32 = 0x22;
pub const FUNCT_SUBU: u32 = 0x23;
pub const FUNCT_AND: u32 = 0x24;
pub const FUNCT_OR: u32 = 0x25;
pub const FUNCT_XOR: u32 = 0x26;
pub const FUNCT_NOR: u32 = 0x27;
pub const FUNCT_SLT: u32 = 0x2A;
pub const FUNCT_SLTU: u32 = 0x2B;
pub const FUNCT_DADD: u32 = 0x2C; // MIPS III
pub const FUNCT_DADDU: u32 = 0x2D; // MIPS III
pub const FUNCT_DSUB: u32 = 0x2E; // MIPS III
pub const FUNCT_DSUBU: u32 = 0x2F; // MIPS III
pub const FUNCT_TGE: u32 = 0x30;
pub const FUNCT_TGEU: u32 = 0x31;
pub const FUNCT_TLT: u32 = 0x32;
pub const FUNCT_TLTU: u32 = 0x33;
pub const FUNCT_TEQ: u32 = 0x34;
pub const FUNCT_TNE: u32 = 0x36;
pub const FUNCT_DSLL: u32 = 0x38; // MIPS III
pub const FUNCT_DSRL: u32 = 0x3A; // MIPS III
pub const FUNCT_DSRA: u32 = 0x3B; // MIPS III
pub const FUNCT_DSLL32: u32 = 0x3C; // MIPS III
pub const FUNCT_DSRL32: u32 = 0x3E; // MIPS III
pub const FUNCT_DSRA32: u32 = 0x3F; // MIPS III

// REGIMM (RT) - Bits 20..16
pub const RT_BLTZ: u32 = 0x00;
pub const RT_BGEZ: u32 = 0x01;
pub const RT_BLTZL: u32 = 0x02;
pub const RT_BGEZL: u32 = 0x03;
pub const RT_TGEI: u32 = 0x08;
pub const RT_TGEIU: u32 = 0x09;
pub const RT_TLTI: u32 = 0x0A;
pub const RT_TLTIU: u32 = 0x0B;
pub const RT_TEQI: u32 = 0x0C;
pub const RT_TNEI: u32 = 0x0E;
pub const RT_BLTZAL: u32 = 0x10;
pub const RT_BGEZAL: u32 = 0x11;
pub const RT_BLTZALL: u32 = 0x12;
pub const RT_BGEZALL: u32 = 0x13;

// COP0 (RS) - Bits 25..21
pub const RS_MFC0: u32 = 0x00;
pub const RS_DMFC0: u32 = 0x01; // MIPS III
pub const RS_CFC0: u32 = 0x02;
pub const RS_MTC0: u32 = 0x04;
pub const RS_DMTC0: u32 = 0x05; // MIPS III
pub const RS_CTC0: u32 = 0x06;
pub const RS_BC0: u32 = 0x08;
pub const RS_TLB: u32 = 0x10;

// COP0 Function (FUNCT) - Bits 5..0 (when RS=TLB)
pub const FUNCT_TLBR: u32 = 0x01;
pub const FUNCT_TLBWI: u32 = 0x02;
pub const FUNCT_TLBWR: u32 = 0x06;
pub const FUNCT_TLBP: u32 = 0x08;
pub const FUNCT_ERET: u32 = 0x18; // MIPS III
pub const FUNCT_WAIT: u32 = 0x20; // R4600/R5000 WAIT; on R4000/R4400 an unimplemented COP0 function that reads as a no-op rather than raising RI

// COP1 (RS) - Bits 25..21
pub const RS_MFC1: u32 = 0x00;
pub const RS_DMFC1: u32 = 0x01; // MIPS III
pub const RS_CFC1: u32 = 0x02;
pub const RS_MTC1: u32 = 0x04;
pub const RS_DMTC1: u32 = 0x05; // MIPS III
pub const RS_CTC1: u32 = 0x06;
pub const RS_BC1: u32 = 0x08;
pub const RS_S: u32 = 0x10;
pub const RS_D: u32 = 0x11;
pub const RS_W: u32 = 0x14;
pub const RS_L: u32 = 0x15; // MIPS III
pub const RS_PS: u32 = 0x16; // MIPS IV

// COP1 Function (FUNCT) - Bits 5..0
pub const FUNCT_FADD: u32 = 0x00;
pub const FUNCT_FSUB: u32 = 0x01;
pub const FUNCT_FMUL: u32 = 0x02;
pub const FUNCT_FDIV: u32 = 0x03;
pub const FUNCT_FSQRT: u32 = 0x04;
pub const FUNCT_FABS: u32 = 0x05;
pub const FUNCT_FMOV: u32 = 0x06;
pub const FUNCT_FNEG: u32 = 0x07;
pub const FUNCT_FROUND_L: u32 = 0x08; // MIPS III
pub const FUNCT_FTRUNC_L: u32 = 0x09; // MIPS III
pub const FUNCT_FCEIL_L: u32 = 0x0A; // MIPS III
pub const FUNCT_FFLOOR_L: u32 = 0x0B; // MIPS III
pub const FUNCT_FROUND_W: u32 = 0x0C;
pub const FUNCT_FTRUNC_W: u32 = 0x0D;
pub const FUNCT_FCEIL_W: u32 = 0x0E;
pub const FUNCT_FFLOOR_W: u32 = 0x0F;
pub const FUNCT_FMOVCF: u32 = 0x11; // MIPS IV (MOVCF)
pub const FUNCT_FMOVZ: u32 = 0x12; // MIPS IV
pub const FUNCT_FMOVN: u32 = 0x13; // MIPS IV
pub const FUNCT_FRECIP: u32 = 0x15; // MIPS IV
pub const FUNCT_FRSQRT: u32 = 0x16; // MIPS IV
pub const FUNCT_FCVT_S: u32 = 0x20;
pub const FUNCT_FCVT_D: u32 = 0x21;
pub const FUNCT_FCVT_W: u32 = 0x24;
pub const FUNCT_FCVT_L: u32 = 0x25; // MIPS III
pub const FUNCT_FCVT_PS: u32 = 0x26; // MIPS IV
pub const FUNCT_FC_F: u32 = 0x30;
pub const FUNCT_FC_UN: u32 = 0x31;
pub const FUNCT_FC_EQ: u32 = 0x32;
pub const FUNCT_FC_UEQ: u32 = 0x33;
pub const FUNCT_FC_OLT: u32 = 0x34;
pub const FUNCT_FC_ULT: u32 = 0x35;
pub const FUNCT_FC_OLE: u32 = 0x36;
pub const FUNCT_FC_ULE: u32 = 0x37;
pub const FUNCT_FC_SF: u32 = 0x38;
pub const FUNCT_FC_NGLE: u32 = 0x39;
pub const FUNCT_FC_SEQ: u32 = 0x3A;
pub const FUNCT_FC_NGL: u32 = 0x3B;
pub const FUNCT_FC_LT: u32 = 0x3C;
pub const FUNCT_FC_NGE: u32 = 0x3D;
pub const FUNCT_FC_LE: u32 = 0x3E;
pub const FUNCT_FC_NGT: u32 = 0x3F;

// COP1X (Funct) - Bits 5..0 (MIPS IV)
pub const FUNCT_LWXC1: u32 = 0x00;
pub const FUNCT_LDXC1: u32 = 0x01;
pub const FUNCT_SWXC1: u32 = 0x08;
pub const FUNCT_SDXC1: u32 = 0x09;
pub const FUNCT_PREFX: u32 = 0x0F;
pub const FUNCT_MADD_S: u32 = 0x20;
pub const FUNCT_MADD_D: u32 = 0x21;
pub const FUNCT_MADD_PS: u32 = 0x26;
pub const FUNCT_MSUB_S: u32 = 0x28;
pub const FUNCT_MSUB_D: u32 = 0x29;
pub const FUNCT_MSUB_PS: u32 = 0x2E;
pub const FUNCT_NMADD_S: u32 = 0x30;
pub const FUNCT_NMADD_D: u32 = 0x31;
pub const FUNCT_NMADD_PS: u32 = 0x36;
pub const FUNCT_NMSUB_S: u32 = 0x38;
pub const FUNCT_NMSUB_D: u32 = 0x39;
pub const FUNCT_NMSUB_PS: u32 = 0x3E;

// Cache Targets (Bits 17..16)
pub const CACH_PI: u32 = 0x0;     // Primary Instruction
pub const CACH_PD: u32 = 0x1;     // Primary Data
pub const CACH_SI: u32 = 0x2;     // Secondary Instruction
pub const CACH_SD: u32 = 0x3;     // Secondary Data

// Cache Operations (Bits 20..18, shifted left by 2 in rt field)
pub const C_IINV: u32 = 0x00;     // Index Invalidate (I, SI)
pub const C_IWBINV: u32 = 0x00;   // Index Writeback Invalidate (D, SD)
pub const C_ILT: u32 = 0x04;      // Index Load Tag
pub const C_IST: u32 = 0x08;      // Index Store Tag
pub const C_CDX: u32 = 0x0C;      // Create Dirty Exclusive (D, SD)
pub const C_HINV: u32 = 0x10;     // Hit Invalidate
pub const C_HWBINV: u32 = 0x14;   // Hit Writeback Invalidate (D, SD)
pub const C_FILL: u32 = 0x14;     // Fill (I)
pub const C_HWB: u32 = 0x18;      // Hit Writeback
pub const C_HSV: u32 = 0x1C;      // Hit Set Virtual (SI, SD)

// R5K/Triton-specific secondary cache ops (target must be SI or SD)
pub const C_INVALL:  u32 = 0x00;  // Triton: Invalidate all secondary cache lines (same encoding as C_IINV)
pub const C_INVPAGE: u32 = 0x14;  // Triton: Invalidate secondary cache page (same encoding as C_HWBINV)
