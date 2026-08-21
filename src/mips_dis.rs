// MIPS Disassembler

use crate::mips_isa::*;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufRead};

#[derive(Clone, Default)]
pub struct SymbolTable {
    pub symbols: BTreeMap<u64, String>,
}

impl SymbolTable {
    pub fn new() -> Self {
        Self { symbols: BTreeMap::new() }
    }

    pub fn load(&mut self, path: &str) -> io::Result<usize> {
        let file = File::open(path)?;
        let reader = io::BufReader::new(file);
        let mut count = 0;
        for line in reader.lines() {
            let line = line?;
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                // Format: address type symbol
                // We ignore type (parts[1])
                if let Ok(mut addr) = u64::from_str_radix(parts[0].trim_start_matches("0x"), 16) {
                    // Sign-extend 32-bit kernel addresses (0x80000000-0xFFFFFFFF)
                    if addr <= 0xFFFF_FFFF && (addr & 0x8000_0000) != 0 {
                        addr |= 0xFFFF_FFFF_0000_0000;
                    }
                    self.symbols.insert(addr, parts[2].to_string());
                    count += 1;
                }
            }
        }
        Ok(count)
    }

    pub fn lookup(&self, addr: u64) -> Option<(u64, &str)> {
        if let Some((&sym_addr, name)) = self.symbols.range(..=addr).next_back() {
            if addr - sym_addr < 0x10000 {
                return Some((sym_addr, name.as_str()));
            }
        }
        None
    }

    pub fn get_addr(&self, name: &str) -> Option<u64> {
        for (addr, sym_name) in &self.symbols {
            if sym_name == name {
                return Some(*addr);
            }
        }
        None
    }
}

// Helper functions to extract instruction fields
#[inline]
fn opcode(instr: u32) -> u32 {
    (instr >> 26) & 0x3F
}

#[inline]
fn rs(instr: u32) -> u32 {
    (instr >> 21) & 0x1F
}

#[inline]
fn rt(instr: u32) -> u32 {
    (instr >> 16) & 0x1F
}

#[inline]
fn rd(instr: u32) -> u32 {
    (instr >> 11) & 0x1F
}

#[inline]
fn sa(instr: u32) -> u32 {
    (instr >> 6) & 0x1F
}

#[inline]
fn funct(instr: u32) -> u32 {
    instr & 0x3F
}

#[inline]
fn imm(instr: u32) -> u32 {
    instr & 0xFFFF
}

#[inline]
fn imm_signed(instr: u32) -> i16 {
    (instr & 0xFFFF) as i16
}

#[inline]
fn target(instr: u32) -> u32 {
    instr & 0x3FFFFFF
}

// Register name lookup
pub fn reg_name(r: u32) -> &'static str {
    match r {
        0 => "zero",
        1 => "at",
        2 => "v0",
        3 => "v1",
        4 => "a0",
        5 => "a1",
        6 => "a2",
        7 => "a3",
        8 => "t0",
        9 => "t1",
        10 => "t2",
        11 => "t3",
        12 => "t4",
        13 => "t5",
        14 => "t6",
        15 => "t7",
        16 => "s0",
        17 => "s1",
        18 => "s2",
        19 => "s3",
        20 => "s4",
        21 => "s5",
        22 => "s6",
        23 => "s7",
        24 => "t8",
        25 => "t9",
        26 => "k0",
        27 => "k1",
        28 => "gp",
        29 => "sp",
        30 => "fp",
        31 => "ra",
        _ => "??",
    }
}

// FPU register name
fn freg_name(r: u32) -> String {
    format!("f{}", r)
}

// CP0 register name lookup
pub fn cp0_reg_name(r: u32) -> &'static str {
    match r {
        0 => "Index",
        1 => "Random",
        2 => "EntryLo0",
        3 => "EntryLo1",
        4 => "Context",
        5 => "PageMask",
        6 => "Wired",
        8 => "BadVAddr",
        9 => "Count",
        10 => "EntryHi",
        11 => "Compare",
        12 => "Status",
        13 => "Cause",
        14 => "EPC",
        15 => "PRId",
        16 => "Config",
        17 => "LLAddr",
        18 => "WatchLo",
        19 => "WatchHi",
        20 => "XContext",
        23 => "Debug",
        24 => "DEPC",
        26 => "ECC",
        27 => "CacheErr",
        28 => "TagLo",
        29 => "TagHi",
        30 => "ErrorEPC",
        _ => "?",
    }
}

// CP1 control register name lookup
pub fn cp1_control_reg_name(r: u32) -> &'static str {
    match r {
        0 => "FIR",       // FP Implementation/Revision
        25 => "FCCR",     // FP Condition Codes
        26 => "FEXR",     // FP Exceptions
        28 => "FENR",     // FP Enables
        31 => "FCSR",     // FP Control/Status
        _ => "?",
    }
}

// MIPS address segment detection and formatting
fn format_address(addr: u64, symbols: Option<&SymbolTable>) -> String {
    // Check if this is a true 64-bit address or 32-bit compatibility mode
    let is_64bit = (addr >> 32) != 0 && (addr >> 32) != 0xFFFFFFFF;

    let base = if is_64bit {
        // MIPS64 addressing segments
        match addr {
            // XKUSEG: 0x0000_0000_0000_0000 - 0x0000_00FF_FFFF_FFFF (user, mapped)
            0x0000_0000_0000_0000..=0x0000_00FF_FFFF_FFFF => {
                format!("0x{:016x} [xkuseg]", addr)
            }

            // XKSSEG: 0x4000_0000_0000_0000 - 0x4000_00FF_FFFF_FFFF (supervisor, mapped)
            0x4000_0000_0000_0000..=0x4000_00FF_FFFF_FFFF => {
                format!("0x{:016x} [xksseg]", addr)
            }

            // XKPHYS: 0x8000_0000_0000_0000 - 0xBFFF_FFFF_FFFF_FFFF (kernel physical, unmapped)
            0x8000_0000_0000_0000..=0xBFFF_FFFF_FFFF_FFFF => {
                let cca = (addr >> 59) & 0x7; // Cache coherency attribute
                let paddr = addr & 0x07FF_FFFF_FFFF_FFFF;
                format!("0x{:016x} [xkphys:cca={}:0x{:x}]", addr, cca, paddr)
            }

            // XKSEG: 0xC000_0000_0000_0000 - 0xC000_00FF_FFFF_FFFF (kernel, mapped)
            0xC000_0000_0000_0000..=0xC000_00FF_FFFF_FFFF => {
                format!("0x{:016x} [xkseg]", addr)
            }

            // CKSEG0: 0xFFFF_FFFF_8000_0000 - 0xFFFF_FFFF_9FFF_FFFF (compat kseg0)
            0xFFFF_FFFF_8000_0000..=0xFFFF_FFFF_9FFF_FFFF => {
                let offset = addr & 0x1FFF_FFFF;
                format!("0x{:016x} [ckseg0:0x{:08x}]", addr, offset)
            }

            // CKSEG1: 0xFFFF_FFFF_A000_0000 - 0xFFFF_FFFF_BFFF_FFFF (compat kseg1)
            0xFFFF_FFFF_A000_0000..=0xFFFF_FFFF_BFFF_FFFF => {
                let offset = addr & 0x1FFF_FFFF;
                format!("0x{:016x} [ckseg1:0x{:08x}]", addr, offset)
            }

            // CKSSEG: 0xFFFF_FFFF_C000_0000 - 0xFFFF_FFFF_DFFF_FFFF (compat ksseg)
            0xFFFF_FFFF_C000_0000..=0xFFFF_FFFF_DFFF_FFFF => {
                format!("0x{:016x} [cksseg]", addr)
            }

            // CKSEG3: 0xFFFF_FFFF_E000_0000 - 0xFFFF_FFFF_FFFF_FFFF (compat kseg3)
            0xFFFF_FFFF_E000_0000..=0xFFFF_FFFF_FFFF_FFFF => {
                format!("0x{:016x} [ckseg3]", addr)
            }

            _ => format!("0x{:016x}", addr),
        }
    } else {
        // MIPS32 or 32-bit compatibility mode addressing
        let addr32 = addr as u32;

        match addr32 {
            // KUSEG: 0x00000000 - 0x7FFFFFFF (user segment, mapped)
            0x0000_0000..=0x7FFF_FFFF => format!("0x{:08x}", addr32),

            // KSEG0: 0x80000000 - 0x9FFFFFFF (kernel segment 0, unmapped, cached)
            0x8000_0000..=0x9FFF_FFFF => {
                let offset = addr32 & 0x1FFF_FFFF;
                format!("0x{:08x} [kseg0:0x{:08x}]", addr32, offset)
            }

            // KSEG1: 0xA0000000 - 0xBFFFFFFF (kernel segment 1, unmapped, uncached)
            0xA000_0000..=0xBFFF_FFFF => {
                let offset = addr32 & 0x1FFF_FFFF;
                format!("0x{:08x} [kseg1:0x{:08x}]", addr32, offset)
            }

            // KSSEG: 0xC0000000 - 0xDFFFFFFF (kernel supervisor segment, mapped)
            0xC000_0000..=0xDFFF_FFFF => format!("0x{:08x} [ksseg]", addr32),

            // KSEG3: 0xE0000000 - 0xFFFFFFFF (kernel segment 3, mapped)
            0xE000_0000..=0xFFFF_FFFF => format!("0x{:08x} [kseg3]", addr32),
        }
    };

    if let Some(syms) = symbols {
        let mut lookup = syms.lookup(addr);
        let mut effective_addr = addr;

        // If not found and address is KSEG1 (0xFFFFFFFF_A...), try KSEG0 (0xFFFFFFFF_8...)
        if lookup.is_none() && (addr >> 32) == 0xFFFFFFFF && ((addr >> 29) & 0x7) == 5 {
            let kseg0_addr = (addr & 0x1FFFFFFF) | 0xFFFF_FFFF_8000_0000;
            if let Some(res) = syms.lookup(kseg0_addr) {
                lookup = Some(res);
                effective_addr = kseg0_addr;
            }
        }

        if let Some((sym_addr, name)) = lookup {
            let offset = effective_addr - sym_addr;
            if offset > 256 {
                return base;
            }
            if offset == 0 {
                return format!("{} <{}>", base, name);
            } else {
                return format!("{} <{}+0x{:x}>", base, name, offset);
            }
        }
    }
    base
}

// Disassemble SPECIAL opcode instructions
fn disasm_special(instr: u32) -> String {
    let rs_val = rs(instr);
    let rt_val = rt(instr);
    let rd_val = rd(instr);
    let sa_val = sa(instr);
    let funct_val = funct(instr);

    match funct_val {
        FUNCT_SLL => {
            if instr == 0 {
                "nop".to_string()
            } else {
                format!("sll {}, {}, {}", reg_name(rd_val), reg_name(rt_val), sa_val)
            }
        }
        FUNCT_MOVCI => {
            // MOVCI: bits [20:18] = cc, bit [16] = tf
            let cc = (instr >> 18) & 0x7;
            let tf = ((instr >> 16) & 0x1) != 0;
            let mnemonic = if tf { "movt" } else { "movf" };
            if cc == 0 {
                format!("{} {}, {}", mnemonic, reg_name(rd_val), reg_name(rs_val))
            } else {
                format!("{} {}, {}, ${}", mnemonic, reg_name(rd_val), reg_name(rs_val), cc)
            }
        }
        FUNCT_SRL => format!("srl {}, {}, {}", reg_name(rd_val), reg_name(rt_val), sa_val),
        FUNCT_SRA => format!("sra {}, {}, {}", reg_name(rd_val), reg_name(rt_val), sa_val),
        FUNCT_SLLV => format!("sllv {}, {}, {}", reg_name(rd_val), reg_name(rt_val), reg_name(rs_val)),
        FUNCT_SRLV => format!("srlv {}, {}, {}", reg_name(rd_val), reg_name(rt_val), reg_name(rs_val)),
        FUNCT_SRAV => format!("srav {}, {}, {}", reg_name(rd_val), reg_name(rt_val), reg_name(rs_val)),
        FUNCT_JR => format!("jr {}", reg_name(rs_val)),
        FUNCT_JALR => {
            if rd_val == 31 {
                format!("jalr {}", reg_name(rs_val))
            } else {
                format!("jalr {}, {}", reg_name(rd_val), reg_name(rs_val))
            }
        }
        FUNCT_MOVZ => format!("movz {}, {}, {}", reg_name(rd_val), reg_name(rs_val), reg_name(rt_val)),
        FUNCT_MOVN => format!("movn {}, {}, {}", reg_name(rd_val), reg_name(rs_val), reg_name(rt_val)),
        FUNCT_SYSCALL => format!("syscall 0x{:x}", (instr >> 6) & 0xFFFFF),
        FUNCT_BREAK => format!("break 0x{:x}", (instr >> 6) & 0xFFFFF),
        FUNCT_SYNC => format!("sync"),
        FUNCT_MFHI => format!("mfhi {}", reg_name(rd_val)),
        FUNCT_MTHI => format!("mthi {}", reg_name(rs_val)),
        FUNCT_MFLO => format!("mflo {}", reg_name(rd_val)),
        FUNCT_MTLO => format!("mtlo {}", reg_name(rs_val)),
        FUNCT_DSLLV => format!("dsllv {}, {}, {}", reg_name(rd_val), reg_name(rt_val), reg_name(rs_val)),
        FUNCT_DSRLV => format!("dsrlv {}, {}, {}", reg_name(rd_val), reg_name(rt_val), reg_name(rs_val)),
        FUNCT_DSRAV => format!("dsrav {}, {}, {}", reg_name(rd_val), reg_name(rt_val), reg_name(rs_val)),
        FUNCT_MULT => format!("mult {}, {}", reg_name(rs_val), reg_name(rt_val)),
        FUNCT_MULTU => format!("multu {}, {}", reg_name(rs_val), reg_name(rt_val)),
        FUNCT_DIV => format!("div {}, {}", reg_name(rs_val), reg_name(rt_val)),
        FUNCT_DIVU => format!("divu {}, {}", reg_name(rs_val), reg_name(rt_val)),
        FUNCT_DMULT => format!("dmult {}, {}", reg_name(rs_val), reg_name(rt_val)),
        FUNCT_DMULTU => format!("dmultu {}, {}", reg_name(rs_val), reg_name(rt_val)),
        FUNCT_DDIV => format!("ddiv {}, {}", reg_name(rs_val), reg_name(rt_val)),
        FUNCT_DDIVU => format!("ddivu {}, {}", reg_name(rs_val), reg_name(rt_val)),
        FUNCT_ADD => format!("add {}, {}, {}", reg_name(rd_val), reg_name(rs_val), reg_name(rt_val)),
        FUNCT_ADDU => format!("addu {}, {}, {}", reg_name(rd_val), reg_name(rs_val), reg_name(rt_val)),
        FUNCT_SUB => format!("sub {}, {}, {}", reg_name(rd_val), reg_name(rs_val), reg_name(rt_val)),
        FUNCT_SUBU => format!("subu {}, {}, {}", reg_name(rd_val), reg_name(rs_val), reg_name(rt_val)),
        FUNCT_AND => format!("and {}, {}, {}", reg_name(rd_val), reg_name(rs_val), reg_name(rt_val)),
        FUNCT_OR => format!("or {}, {}, {}", reg_name(rd_val), reg_name(rs_val), reg_name(rt_val)),
        FUNCT_XOR => format!("xor {}, {}, {}", reg_name(rd_val), reg_name(rs_val), reg_name(rt_val)),
        FUNCT_NOR => format!("nor {}, {}, {}", reg_name(rd_val), reg_name(rs_val), reg_name(rt_val)),
        FUNCT_SLT => format!("slt {}, {}, {}", reg_name(rd_val), reg_name(rs_val), reg_name(rt_val)),
        FUNCT_SLTU => format!("sltu {}, {}, {}", reg_name(rd_val), reg_name(rs_val), reg_name(rt_val)),
        FUNCT_DADD => format!("dadd {}, {}, {}", reg_name(rd_val), reg_name(rs_val), reg_name(rt_val)),
        FUNCT_DADDU => format!("daddu {}, {}, {}", reg_name(rd_val), reg_name(rs_val), reg_name(rt_val)),
        FUNCT_DSUB => format!("dsub {}, {}, {}", reg_name(rd_val), reg_name(rs_val), reg_name(rt_val)),
        FUNCT_DSUBU => format!("dsubu {}, {}, {}", reg_name(rd_val), reg_name(rs_val), reg_name(rt_val)),
        FUNCT_TGE => format!("tge {}, {}", reg_name(rs_val), reg_name(rt_val)),
        FUNCT_TGEU => format!("tgeu {}, {}", reg_name(rs_val), reg_name(rt_val)),
        FUNCT_TLT => format!("tlt {}, {}", reg_name(rs_val), reg_name(rt_val)),
        FUNCT_TLTU => format!("tltu {}, {}", reg_name(rs_val), reg_name(rt_val)),
        FUNCT_TEQ => format!("teq {}, {}", reg_name(rs_val), reg_name(rt_val)),
        FUNCT_TNE => format!("tne {}, {}", reg_name(rs_val), reg_name(rt_val)),
        FUNCT_DSLL => format!("dsll {}, {}, {}", reg_name(rd_val), reg_name(rt_val), sa_val),
        FUNCT_DSRL => format!("dsrl {}, {}, {}", reg_name(rd_val), reg_name(rt_val), sa_val),
        FUNCT_DSRA => format!("dsra {}, {}, {}", reg_name(rd_val), reg_name(rt_val), sa_val),
        FUNCT_DSLL32 => format!("dsll32 {}, {}, {}", reg_name(rd_val), reg_name(rt_val), sa_val),
        FUNCT_DSRL32 => format!("dsrl32 {}, {}, {}", reg_name(rd_val), reg_name(rt_val), sa_val),
        FUNCT_DSRA32 => format!("dsra32 {}, {}, {}", reg_name(rd_val), reg_name(rt_val), sa_val),
        _ => format!("unknown_special 0x{:08x}", instr),
    }
}

// Disassemble REGIMM opcode instructions
fn disasm_regimm(instr: u32, pc: u64, symbols: Option<&SymbolTable>) -> String {
    let rs_val = rs(instr);
    let rt_val = rt(instr);
    let offset = imm_signed(instr);
    let target_addr = pc.wrapping_add(4).wrapping_add((offset as i64 * 4) as u64);

    match rt_val {
        RT_BLTZ => format!("bltz {}, {} <={}>", reg_name(rs_val), offset, format_address(target_addr, symbols)),
        RT_BGEZ => format!("bgez {}, {} <={}>", reg_name(rs_val), offset, format_address(target_addr, symbols)),
        RT_BLTZL => format!("bltzl {}, {} <={}>", reg_name(rs_val), offset, format_address(target_addr, symbols)),
        RT_BGEZL => format!("bgezl {}, {} <={}>", reg_name(rs_val), offset, format_address(target_addr, symbols)),
        RT_TGEI => format!("tgei {}, {}", reg_name(rs_val), offset),
        RT_TGEIU => format!("tgeiu {}, {}", reg_name(rs_val), offset as u16),
        RT_TLTI => format!("tlti {}, {}", reg_name(rs_val), offset),
        RT_TLTIU => format!("tltiu {}, {}", reg_name(rs_val), offset as u16),
        RT_TEQI => format!("teqi {}, {}", reg_name(rs_val), offset),
        RT_TNEI => format!("tnei {}, {}", reg_name(rs_val), offset),
        RT_BLTZAL => format!("bltzal {}, {} <={}>", reg_name(rs_val), offset, format_address(target_addr, symbols)),
        RT_BGEZAL => format!("bgezal {}, {} <={}>", reg_name(rs_val), offset, format_address(target_addr, symbols)),
        RT_BLTZALL => format!("bltzall {}, {} <={}>", reg_name(rs_val), offset, format_address(target_addr, symbols)),
        RT_BGEZALL => format!("bgezall {}, {} <={}>", reg_name(rs_val), offset, format_address(target_addr, symbols)),
        _ => format!("unknown_regimm 0x{:08x}", instr),
    }
}

// Disassemble COP0 instructions
fn disasm_cop0(instr: u32) -> String {
    let rs_val = rs(instr);
    let rt_val = rt(instr);
    let rd_val = rd(instr);
    let funct_val = funct(instr);

    match rs_val {
        RS_MFC0 => format!("mfc0 {}, ${} ({})", reg_name(rt_val), rd_val, cp0_reg_name(rd_val)),
        RS_DMFC0 => format!("dmfc0 {}, ${} ({})", reg_name(rt_val), rd_val, cp0_reg_name(rd_val)),
        RS_MTC0 => format!("mtc0 {}, ${} ({})", reg_name(rt_val), rd_val, cp0_reg_name(rd_val)),
        RS_DMTC0 => format!("dmtc0 {}, ${} ({})", reg_name(rt_val), rd_val, cp0_reg_name(rd_val)),
        RS_TLB => match funct_val {
            FUNCT_TLBR => "tlbr".to_string(),
            FUNCT_TLBWI => "tlbwi".to_string(),
            FUNCT_TLBWR => "tlbwr".to_string(),
            FUNCT_TLBP => "tlbp".to_string(),
            FUNCT_ERET => "eret".to_string(),
            FUNCT_WAIT => "wait".to_string(),
            _ => format!("unknown_cop0_tlb 0x{:08x}", instr),
        },
        _ => format!("unknown_cop0 0x{:08x}", instr),
    }
}

// Disassemble COP1 instructions
fn disasm_cop1(instr: u32) -> String {
    let rs_val = rs(instr);
    let rt_val = rt(instr);
    let rd_val = rd(instr);
    let fs = rd_val;
    let ft = rt_val;
    let fd = sa(instr);
    let funct_val = funct(instr);

    match rs_val {
        RS_MFC1 => format!("mfc1 {}, {}", reg_name(rt_val), freg_name(fs)),
        RS_DMFC1 => format!("dmfc1 {}, {}", reg_name(rt_val), freg_name(fs)),
        RS_CFC1 => format!("cfc1 {}, ${} ({})", reg_name(rt_val), fs, cp1_control_reg_name(fs)),
        RS_MTC1 => format!("mtc1 {}, {}", reg_name(rt_val), freg_name(fs)),
        RS_DMTC1 => format!("dmtc1 {}, {}", reg_name(rt_val), freg_name(fs)),
        RS_CTC1 => format!("ctc1 {}, ${} ({})", reg_name(rt_val), fs, cp1_control_reg_name(fs)),
        RS_BC1 => {
            let offset = imm_signed(instr);
            let cc = (instr >> 18) & 0x7;
            let base_mnemonic = match rt_val & 0x3 {
                0 => "bc1f",
                1 => "bc1t",
                2 => "bc1fl",
                3 => "bc1tl",
                _ => unreachable!(),
            };
            if cc == 0 {
                format!("{} {}", base_mnemonic, offset)
            } else {
                format!("{} ${}, {}", base_mnemonic, cc, offset)
            }
        }
        RS_S | RS_D | RS_W | RS_L | RS_PS => {
            let fmt = match rs_val {
                RS_S => "s",
                RS_D => "d",
                RS_W => "w",
                RS_L => "l",
                RS_PS => "ps",
                _ => "?",
            };

            match funct_val {
                FUNCT_FADD => format!("add.{} {}, {}, {}", fmt, freg_name(fd), freg_name(fs), freg_name(ft)),
                FUNCT_FSUB => format!("sub.{} {}, {}, {}", fmt, freg_name(fd), freg_name(fs), freg_name(ft)),
                FUNCT_FMUL => format!("mul.{} {}, {}, {}", fmt, freg_name(fd), freg_name(fs), freg_name(ft)),
                FUNCT_FDIV => format!("div.{} {}, {}, {}", fmt, freg_name(fd), freg_name(fs), freg_name(ft)),
                FUNCT_FSQRT => format!("sqrt.{} {}, {}", fmt, freg_name(fd), freg_name(fs)),
                FUNCT_FABS => format!("abs.{} {}, {}", fmt, freg_name(fd), freg_name(fs)),
                FUNCT_FMOV => format!("mov.{} {}, {}", fmt, freg_name(fd), freg_name(fs)),
                FUNCT_FNEG => format!("neg.{} {}, {}", fmt, freg_name(fd), freg_name(fs)),
                FUNCT_FROUND_L => format!("round.l.{} {}, {}", fmt, freg_name(fd), freg_name(fs)),
                FUNCT_FTRUNC_L => format!("trunc.l.{} {}, {}", fmt, freg_name(fd), freg_name(fs)),
                FUNCT_FCEIL_L => format!("ceil.l.{} {}, {}", fmt, freg_name(fd), freg_name(fs)),
                FUNCT_FFLOOR_L => format!("floor.l.{} {}, {}", fmt, freg_name(fd), freg_name(fs)),
                FUNCT_FROUND_W => format!("round.w.{} {}, {}", fmt, freg_name(fd), freg_name(fs)),
                FUNCT_FTRUNC_W => format!("trunc.w.{} {}, {}", fmt, freg_name(fd), freg_name(fs)),
                FUNCT_FCEIL_W => format!("ceil.w.{} {}, {}", fmt, freg_name(fd), freg_name(fs)),
                FUNCT_FFLOOR_W => format!("floor.w.{} {}, {}", fmt, freg_name(fd), freg_name(fs)),
                FUNCT_FMOVZ => format!("movz.{} {}, {}, {}", fmt, freg_name(fd), freg_name(fs), reg_name(rt_val)),
                FUNCT_FMOVN => format!("movn.{} {}, {}, {}", fmt, freg_name(fd), freg_name(fs), reg_name(rt_val)),
                FUNCT_FMOVCF => {
                    let cc = (instr >> 18) & 0x7;
                    let tf = ((instr >> 16) & 0x1) != 0;
                    let mnemonic = if tf { "movt" } else { "movf" };
                    if cc == 0 {
                        format!("{}.{} {}, {}", mnemonic, fmt, freg_name(fd), freg_name(fs))
                    } else {
                        format!("{}.{} {}, {}, ${}", mnemonic, fmt, freg_name(fd), freg_name(fs), cc)
                    }
                }
                FUNCT_FRECIP => format!("recip.{} {}, {}", fmt, freg_name(fd), freg_name(fs)),
                FUNCT_FRSQRT => format!("rsqrt.{} {}, {}", fmt, freg_name(fd), freg_name(fs)),
                FUNCT_FCVT_S => format!("cvt.s.{} {}, {}", fmt, freg_name(fd), freg_name(fs)),
                FUNCT_FCVT_D => format!("cvt.d.{} {}, {}", fmt, freg_name(fd), freg_name(fs)),
                FUNCT_FCVT_W => format!("cvt.w.{} {}, {}", fmt, freg_name(fd), freg_name(fs)),
                FUNCT_FCVT_L => format!("cvt.l.{} {}, {}", fmt, freg_name(fd), freg_name(fs)),
                FUNCT_FCVT_PS => format!("cvt.ps.{} {}, {}", fmt, freg_name(fd), freg_name(fs)),
                FUNCT_FC_F | FUNCT_FC_UN | FUNCT_FC_EQ | FUNCT_FC_UEQ |
                FUNCT_FC_OLT | FUNCT_FC_ULT | FUNCT_FC_OLE | FUNCT_FC_ULE |
                FUNCT_FC_SF | FUNCT_FC_NGLE | FUNCT_FC_SEQ | FUNCT_FC_NGL |
                FUNCT_FC_LT | FUNCT_FC_NGE | FUNCT_FC_LE | FUNCT_FC_NGT => {
                    let cc = fd & 0x7; // CC field is in fd (bits [10:8])
                    let cond_name = match funct_val {
                        FUNCT_FC_F => "f",
                        FUNCT_FC_UN => "un",
                        FUNCT_FC_EQ => "eq",
                        FUNCT_FC_UEQ => "ueq",
                        FUNCT_FC_OLT => "olt",
                        FUNCT_FC_ULT => "ult",
                        FUNCT_FC_OLE => "ole",
                        FUNCT_FC_ULE => "ule",
                        FUNCT_FC_SF => "sf",
                        FUNCT_FC_NGLE => "ngle",
                        FUNCT_FC_SEQ => "seq",
                        FUNCT_FC_NGL => "ngl",
                        FUNCT_FC_LT => "lt",
                        FUNCT_FC_NGE => "nge",
                        FUNCT_FC_LE => "le",
                        FUNCT_FC_NGT => "ngt",
                        _ => "?",
                    };
                    if cc == 0 {
                        format!("c.{}.{} {}, {}", cond_name, fmt, freg_name(fs), freg_name(ft))
                    } else {
                        format!("c.{}.{} ${}, {}, {}", cond_name, fmt, cc, freg_name(fs), freg_name(ft))
                    }
                }
                _ => format!("unknown_cop1_{} 0x{:08x}", fmt, instr),
            }
        }
        _ => format!("unknown_cop1 0x{:08x}", instr),
    }
}

// Disassemble COP1X instructions (MIPS IV)
fn disasm_cop1x(instr: u32) -> String {
    let rs_val = rs(instr);
    let rt_val = rt(instr);
    let rd_val = rd(instr);
    let fs = sa(instr);
    let ft = rt_val;
    let fd = rd_val;
    let fr = (instr >> 6) & 0x1F;
    let funct_val = funct(instr);

    match funct_val {
        FUNCT_LWXC1 => format!("lwxc1 {}, {}({})", freg_name(fd), reg_name(rs_val), reg_name(rt_val)),
        FUNCT_LDXC1 => format!("ldxc1 {}, {}({})", freg_name(fd), reg_name(rs_val), reg_name(rt_val)),
        FUNCT_SWXC1 => format!("swxc1 {}, {}({})", freg_name(fs), reg_name(rs_val), reg_name(rt_val)),
        FUNCT_SDXC1 => format!("sdxc1 {}, {}({})", freg_name(fs), reg_name(rs_val), reg_name(rt_val)),
        FUNCT_PREFX => format!("prefx {}, {}({})", rt_val, reg_name(rs_val), reg_name(rt_val)),
        FUNCT_MADD_S => format!("madd.s {}, {}, {}, {}", freg_name(fd), freg_name(fr), freg_name(fs), freg_name(ft)),
        FUNCT_MADD_D => format!("madd.d {}, {}, {}, {}", freg_name(fd), freg_name(fr), freg_name(fs), freg_name(ft)),
        FUNCT_MADD_PS => format!("madd.ps {}, {}, {}, {}", freg_name(fd), freg_name(fr), freg_name(fs), freg_name(ft)),
        FUNCT_MSUB_S => format!("msub.s {}, {}, {}, {}", freg_name(fd), freg_name(fr), freg_name(fs), freg_name(ft)),
        FUNCT_MSUB_D => format!("msub.d {}, {}, {}, {}", freg_name(fd), freg_name(fr), freg_name(fs), freg_name(ft)),
        FUNCT_MSUB_PS => format!("msub.ps {}, {}, {}, {}", freg_name(fd), freg_name(fr), freg_name(fs), freg_name(ft)),
        FUNCT_NMADD_S => format!("nmadd.s {}, {}, {}, {}", freg_name(fd), freg_name(fr), freg_name(fs), freg_name(ft)),
        FUNCT_NMADD_D => format!("nmadd.d {}, {}, {}, {}", freg_name(fd), freg_name(fr), freg_name(fs), freg_name(ft)),
        FUNCT_NMADD_PS => format!("nmadd.ps {}, {}, {}, {}", freg_name(fd), freg_name(fr), freg_name(fs), freg_name(ft)),
        FUNCT_NMSUB_S => format!("nmsub.s {}, {}, {}, {}", freg_name(fd), freg_name(fr), freg_name(fs), freg_name(ft)),
        FUNCT_NMSUB_D => format!("nmsub.d {}, {}, {}, {}", freg_name(fd), freg_name(fr), freg_name(fs), freg_name(ft)),
        FUNCT_NMSUB_PS => format!("nmsub.ps {}, {}, {}, {}", freg_name(fd), freg_name(fr), freg_name(fs), freg_name(ft)),
        _ => format!("unknown_cop1x 0x{:08x}", instr),
    }
}

fn disasm_cache(instr: u32) -> String {
    let rt_val = rt(instr);
    let offset = imm_signed(instr);
    let base = rs(instr);

    let cache = rt_val & 0x3;
    let op = rt_val & 0x1C;

    let cache_str = match cache {
        CACH_PI => "PI",
        CACH_PD => "PD",
        CACH_SI => "SI",
        CACH_SD => "SD",
        _ => "?",
    };

    let op_str = match op {
        C_IINV => match cache {
            CACH_PI | CACH_SI => "Index_Invalidate",
            _ => "Index_Writeback_Inv",
        },
        C_ILT => "Index_Load_Tag",
        C_IST => "Index_Store_Tag",
        C_CDX => "Create_Dirty_Excl",
        C_HINV => "Hit_Invalidate",
        C_HWBINV => match cache {
            CACH_PI => "Fill",
            _ => "Hit_Writeback_Inv",
        },
        C_HWB => "Hit_Writeback",
        C_HSV => "Hit_Set_Virt",
        _ => "Unknown",
    };

    format!("cache {}({}), {}({})", op_str, cache_str, offset, reg_name(base))
}

// Main disassembler function
pub fn disassemble(instr: u32, pc: u64, symbols: Option<&SymbolTable>) -> String {
    let op = opcode(instr);
    let rs_val = rs(instr);
    let rt_val = rt(instr);
    let imm_val = imm(instr);
    let imm_s = imm_signed(instr);
    let target_val = target(instr);

    match op {
        OP_SPECIAL => disasm_special(instr),
        OP_REGIMM => disasm_regimm(instr, pc, symbols),
        OP_J => {
            let target_addr = ((pc + 4) & 0xFFFFFFFF_F0000000) | ((target_val << 2) as u64);
            format!("j 0x{:x} <={}>", target_val << 2, format_address(target_addr, symbols))
        }
        OP_JAL => {
            let target_addr = ((pc + 4) & 0xFFFFFFFF_F0000000) | ((target_val << 2) as u64);
            format!("jal 0x{:x} <={}>", target_val << 2, format_address(target_addr, symbols))
        }
        OP_BEQ => {
            let target_addr = pc.wrapping_add(4).wrapping_add((imm_s as i64 * 4) as u64);
            format!("beq {}, {}, {} <={}>", reg_name(rs_val), reg_name(rt_val), imm_s, format_address(target_addr, symbols))
        }
        OP_BNE => {
            let target_addr = pc.wrapping_add(4).wrapping_add((imm_s as i64 * 4) as u64);
            format!("bne {}, {}, {} <={}>", reg_name(rs_val), reg_name(rt_val), imm_s, format_address(target_addr, symbols))
        }
        OP_BLEZ => {
            let target_addr = pc.wrapping_add(4).wrapping_add((imm_s as i64 * 4) as u64);
            format!("blez {}, {} <={}>", reg_name(rs_val), imm_s, format_address(target_addr, symbols))
        }
        OP_BGTZ => {
            let target_addr = pc.wrapping_add(4).wrapping_add((imm_s as i64 * 4) as u64);
            format!("bgtz {}, {} <={}>", reg_name(rs_val), imm_s, format_address(target_addr, symbols))
        }
        OP_ADDI => format!("addi {}, {}, {}", reg_name(rt_val), reg_name(rs_val), imm_s),
        OP_ADDIU => format!("addiu {}, {}, {}", reg_name(rt_val), reg_name(rs_val), imm_s),
        OP_SLTI => format!("slti {}, {}, {}", reg_name(rt_val), reg_name(rs_val), imm_s),
        OP_SLTIU => format!("sltiu {}, {}, {}", reg_name(rt_val), reg_name(rs_val), imm_s),
        OP_ANDI => format!("andi {}, {}, 0x{:x}", reg_name(rt_val), reg_name(rs_val), imm_val),
        OP_ORI => format!("ori {}, {}, 0x{:x}", reg_name(rt_val), reg_name(rs_val), imm_val),
        OP_XORI => format!("xori {}, {}, 0x{:x}", reg_name(rt_val), reg_name(rs_val), imm_val),
        OP_LUI => format!("lui {}, 0x{:x}", reg_name(rt_val), imm_val),
        OP_COP0 => disasm_cop0(instr),
        OP_COP1 => disasm_cop1(instr),
        OP_COP2 => format!("cop2 0x{:08x}", instr & 0x1FFFFFF),
        OP_COP1X => disasm_cop1x(instr),
        OP_BEQL => {
            let target_addr = pc.wrapping_add(4).wrapping_add((imm_s as i64 * 4) as u64);
            format!("beql {}, {}, {} <={}>", reg_name(rs_val), reg_name(rt_val), imm_s, format_address(target_addr, symbols))
        }
        OP_BNEL => {
            let target_addr = pc.wrapping_add(4).wrapping_add((imm_s as i64 * 4) as u64);
            format!("bnel {}, {}, {} <={}>", reg_name(rs_val), reg_name(rt_val), imm_s, format_address(target_addr, symbols))
        }
        OP_BLEZL => {
            let target_addr = pc.wrapping_add(4).wrapping_add((imm_s as i64 * 4) as u64);
            format!("blezl {}, {} <={}>", reg_name(rs_val), imm_s, format_address(target_addr, symbols))
        }
        OP_BGTZL => {
            let target_addr = pc.wrapping_add(4).wrapping_add((imm_s as i64 * 4) as u64);
            format!("bgtzl {}, {} <={}>", reg_name(rs_val), imm_s, format_address(target_addr, symbols))
        }
        OP_DADDI => format!("daddi {}, {}, {}", reg_name(rt_val), reg_name(rs_val), imm_s),
        OP_DADDIU => format!("daddiu {}, {}, {}", reg_name(rt_val), reg_name(rs_val), imm_s),
        OP_LDL => format!("ldl {}, {}({})", reg_name(rt_val), imm_s, reg_name(rs_val)),
        OP_LDR => format!("ldr {}, {}({})", reg_name(rt_val), imm_s, reg_name(rs_val)),
        OP_LB => format!("lb {}, {}({})", reg_name(rt_val), imm_s, reg_name(rs_val)),
        OP_LH => format!("lh {}, {}({})", reg_name(rt_val), imm_s, reg_name(rs_val)),
        OP_LWL => format!("lwl {}, {}({})", reg_name(rt_val), imm_s, reg_name(rs_val)),
        OP_LW => format!("lw {}, {}({})", reg_name(rt_val), imm_s, reg_name(rs_val)),
        OP_LBU => format!("lbu {}, {}({})", reg_name(rt_val), imm_s, reg_name(rs_val)),
        OP_LHU => format!("lhu {}, {}({})", reg_name(rt_val), imm_s, reg_name(rs_val)),
        OP_LWR => format!("lwr {}, {}({})", reg_name(rt_val), imm_s, reg_name(rs_val)),
        OP_LWU => format!("lwu {}, {}({})", reg_name(rt_val), imm_s, reg_name(rs_val)),
        OP_SB => format!("sb {}, {}({})", reg_name(rt_val), imm_s, reg_name(rs_val)),
        OP_SH => format!("sh {}, {}({})", reg_name(rt_val), imm_s, reg_name(rs_val)),
        OP_SWL => format!("swl {}, {}({})", reg_name(rt_val), imm_s, reg_name(rs_val)),
        OP_SW => format!("sw {}, {}({})", reg_name(rt_val), imm_s, reg_name(rs_val)),
        OP_SDL => format!("sdl {}, {}({})", reg_name(rt_val), imm_s, reg_name(rs_val)),
        OP_SDR => format!("sdr {}, {}({})", reg_name(rt_val), imm_s, reg_name(rs_val)),
        OP_SWR => format!("swr {}, {}({})", reg_name(rt_val), imm_s, reg_name(rs_val)),
        OP_CACHE => disasm_cache(instr),
        OP_LL => format!("ll {}, {}({})", reg_name(rt_val), imm_s, reg_name(rs_val)),
        OP_LWC1 => format!("lwc1 {}, {}({})", freg_name(rt_val), imm_s, reg_name(rs_val)),
        OP_LWC2 => format!("lwc2 ${}, {}({})", rt_val, imm_s, reg_name(rs_val)),
        OP_PREF => format!("pref 0x{:x}, {}({})", rt_val, imm_s, reg_name(rs_val)),
        OP_LLD => format!("lld {}, {}({})", reg_name(rt_val), imm_s, reg_name(rs_val)),
        OP_LDC1 => format!("ldc1 {}, {}({})", freg_name(rt_val), imm_s, reg_name(rs_val)),
        OP_LDC2 => format!("ldc2 ${}, {}({})", rt_val, imm_s, reg_name(rs_val)),
        OP_LD => format!("ld {}, {}({})", reg_name(rt_val), imm_s, reg_name(rs_val)),
        OP_SC => format!("sc {}, {}({})", reg_name(rt_val), imm_s, reg_name(rs_val)),
        OP_SWC1 => format!("swc1 {}, {}({})", freg_name(rt_val), imm_s, reg_name(rs_val)),
        OP_SWC2 => format!("swc2 ${}, {}({})", rt_val, imm_s, reg_name(rs_val)),
        OP_SCD => format!("scd {}, {}({})", reg_name(rt_val), imm_s, reg_name(rs_val)),
        OP_SDC1 => format!("sdc1 {}, {}({})", freg_name(rt_val), imm_s, reg_name(rs_val)),
        OP_SDC2 => format!("sdc2 ${}, {}({})", rt_val, imm_s, reg_name(rs_val)),
        OP_SD => format!("sd {}, {}({})", reg_name(rt_val), imm_s, reg_name(rs_val)),
        _ => format!("unknown 0x{:08x}", instr),
    }
}
