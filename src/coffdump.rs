use std::env;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};

// COFF File Header Constants
pub const MIPSEBMAGIC: u16 = 0x0160;
pub const MIPSELMAGIC: u16 = 0x0162;
pub const SMIPSEBMAGIC: u16 = 0x6001;
pub const SMIPSELMAGIC: u16 = 0x6201;

// Optional Header Magic
pub const OMAGIC: u16 = 0o407; // 0x107
pub const NMAGIC: u16 = 0o410; // 0x108
pub const ZMAGIC: u16 = 0o413; // 0x10b

// Section Flags
pub const STYP_REG: u32 = 0x00000000;
pub const STYP_DSECT: u32 = 0x00000001;
pub const STYP_NOLOAD: u32 = 0x00000002;
pub const STYP_GROUP: u32 = 0x00000004;
pub const STYP_PAD: u32 = 0x00000008;
pub const STYP_COPY: u32 = 0x00000010;
pub const STYP_TEXT: u32 = 0x00000020;
pub const STYP_DATA: u32 = 0x00000040;
pub const STYP_BSS: u32 = 0x00000080;
pub const STYP_RDATA: u32 = 0x00000100;
pub const STYP_SDATA: u32 = 0x00000200;
pub const STYP_SBSS: u32 = 0x00000400;
pub const STYP_UCODE: u32 = 0x00000800;
pub const STYP_LIT8: u32 = 0x08000000;
pub const STYP_LIT4: u32 = 0x10000000;

// Relocation Types
pub const R_ABS: u16 = 0;
pub const R_REFHALF: u16 = 1;
pub const R_REFWORD: u16 = 2;
pub const R_JMPADDR: u16 = 3;
pub const R_REFHI: u16 = 4;
pub const R_REFLO: u16 = 5;
pub const R_GPREL: u16 = 6;
pub const R_LITERAL: u16 = 7;

#[derive(Debug, Default)]
pub struct FileHeader {
    pub f_magic: u16,
    pub f_nscns: u16,
    pub f_timdat: u32,
    pub f_symptr: u32,
    pub f_nsyms: u32,
    pub f_opthdr: u16,
    pub f_flags: u16,
}

#[derive(Debug, Default)]
pub struct AoutHeader {
    pub magic: u16,
    pub vstamp: u16,
    pub tsize: u32,
    pub dsize: u32,
    pub bsize: u32,
    pub entry: u32,
    pub text_start: u32,
    pub data_start: u32,
    pub bss_start: u32,
    pub gprmask: u32,
    pub cprmask: [u32; 4],
    pub gp_value: u32,
}

#[derive(Debug, Default)]
pub struct SectionHeader {
    pub s_name: String,
    pub s_paddr: u32,
    pub s_vaddr: u32,
    pub s_size: u32,
    pub s_scnptr: u32,
    pub s_relptr: u32,
    pub s_lnnoptr: u32,
    pub s_nreloc: u16,
    pub s_nlnno: u16,
    pub s_flags: u32,
}

// MIPS Symbolic Header (HDRR)
#[derive(Debug, Default)]
struct Hdrr {
    magic: u16,
    vstamp: u16,
    iline_max: u32,
    cb_line: u32,
    cb_line_offset: u32,
    idn_max: u32,
    cb_dn_offset: u32,
    ipd_max: u32,
    cb_pd_offset: u32,
    isym_max: u32,
    cb_sym_offset: u32,
    iopt_max: u32,
    cb_opt_offset: u32,
    iaux_max: u32,
    cb_aux_offset: u32,
    iss_max: u32,
    cb_ss_offset: u32,
    iss_ext_max: u32,
    cb_ss_ext_offset: u32,
    ifd_max: u32,
    cb_fd_offset: u32,
    crfd: u32,
    cb_rfd_offset: u32,
    iext_max: u32,
    cb_ext_offset: u32,
}

// File Descriptor Record (FDR)
// Mirrors the on-disk record: every field is parsed, only some are printed.
#[allow(dead_code)]
#[derive(Debug, Default, Clone)]
struct Fdr {
    adr: u32,
    rss: u32,
    iss_base: u32,
    cb_ss: u32,
    isym_base: u32,
    csym: u32,
    iline_base: u32,
    cline: u32,
    iopt_base: u32,
    copt: u32,
    ipd_first: u16,
    cpd: u16,
    iaux_base: u32,
    caux: u32,
    rfd_base: u32,
    crfd: u32,
    bits: u32, // lang:5, fMerge:1, fReadin:1, fBigendian:1, glevel:2, reserved:22
    cb_line_offset: u32,
    cb_line: u32,
}

// Procedure Descriptor Record (PDR)
// Mirrors the on-disk record: every field is parsed, only some are printed.
#[allow(dead_code)]
#[derive(Debug, Default, Clone)]
struct Pdr {
    adr: u32,
    isym: u32,
    iline: u32,
    regmask: u32,
    regoffset: u32,
    iopt: u32,
    fregmask: u32,
    fregoffset: u32,
    frameoffset: u32,
    framereg: u16,
    pcreg: u16,
    ln_low: u32,
    ln_high: u32,
    cb_line_offset: u32,
}

// Symbol Record (SYMR)
#[derive(Debug, Default, Clone)]
struct Symr {
    iss: u32,
    value: u32,
    st: u8,
    sc: u8,
    index: u32,
}

fn get_sc_name(sc: u8) -> &'static str {
    match sc {
        0 => "Nil",
        1 => "Text",
        2 => "Data",
        3 => "Bss",
        4 => "Register",
        5 => "Abs",
        6 => "Undefined",
        7 => "CdbLocal",
        8 => "Bits",
        9 => "CdbSystem",
        10 => "RegImage",
        11 => "Info",
        12 => "UserStruct",
        13 => "SData",
        14 => "SBss",
        15 => "RData",
        16 => "Var",
        17 => "Common",
        18 => "SCommon",
        19 => "VarRegister",
        20 => "Variant",
        21 => "SUndefined",
        22 => "Init",
        23 => "BasedVar",
        _ => "Unknown",
    }
}

fn get_st_name(st: u8) -> &'static str {
    match st {
        0 => "Nil",
        1 => "Global",
        2 => "Static",
        3 => "Param",
        4 => "Local",
        5 => "Label",
        6 => "Proc",
        7 => "Block",
        8 => "End",
        _ => "Other",
    }
}

fn read_u16(f: &mut File) -> io::Result<u16> {
    let mut buf = [0u8; 2];
    f.read_exact(&mut buf)?;
    Ok(u16::from_be_bytes(buf))
}

fn read_u32(f: &mut File) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

fn get_string(table: &[u8], offset: usize) -> String {
    if offset >= table.len() {
        return format!("*OFF* {:X}", offset);
    }
    let mut end = offset;
    while end < table.len() && table[end] != 0 {
        end += 1;
    }
    String::from_utf8_lossy(&table[offset..end]).to_string()
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <file>", args[0]);
        return Ok(());
    }

    let mut f = File::open(&args[1])?;

    // Read File Header
    let mut file_header = FileHeader::default();
    file_header.f_magic = read_u16(&mut f)?;
    file_header.f_nscns = read_u16(&mut f)?;
    file_header.f_timdat = read_u32(&mut f)?;
    file_header.f_symptr = read_u32(&mut f)?;
    file_header.f_nsyms = read_u32(&mut f)?;
    file_header.f_opthdr = read_u16(&mut f)?;
    file_header.f_flags = read_u16(&mut f)?;

    println!("File Header:");
    println!("  Magic:    0x{:04X}", file_header.f_magic);
    println!("  Sections: {}", file_header.f_nscns);
    println!("  Time:     0x{:08X}", file_header.f_timdat);
    println!("  SymPtr:   0x{:08X}", file_header.f_symptr);
    println!("  NumSyms:  {}", file_header.f_nsyms);
    println!("  OptHdrSz: {}", file_header.f_opthdr);
    println!("  Flags:    0x{:04X}", file_header.f_flags);

    // Read Optional Header
    let mut entry_point = 0;
    if file_header.f_opthdr > 0 {
        let mut aout_header = AoutHeader::default();
        aout_header.magic = read_u16(&mut f)?;
        aout_header.vstamp = read_u16(&mut f)?;
        aout_header.tsize = read_u32(&mut f)?;
        aout_header.dsize = read_u32(&mut f)?;
        aout_header.bsize = read_u32(&mut f)?;
        aout_header.entry = read_u32(&mut f)?;
        aout_header.text_start = read_u32(&mut f)?;
        aout_header.data_start = read_u32(&mut f)?;

        entry_point = aout_header.entry;

        // MIPS specific fields (if header is large enough)
        if file_header.f_opthdr >= 56 {
            aout_header.bss_start = read_u32(&mut f)?;
            aout_header.gprmask = read_u32(&mut f)?;
            for i in 0..4 {
                aout_header.cprmask[i] = read_u32(&mut f)?;
            }
            aout_header.gp_value = read_u32(&mut f)?;
        } else {
            // Skip remaining bytes if any (standard AOUT is 28 bytes)
            let remaining = file_header.f_opthdr as i64 - 28;
            if remaining > 0 {
                f.seek(SeekFrom::Current(remaining))?;
            }
        }

        println!("\nOptional Header:");
        println!("  Magic:      0x{:04X}", aout_header.magic);
        println!("  VStamp:     0x{:04X}", aout_header.vstamp);
        println!("  Text Size:  0x{:08X}", aout_header.tsize);
        println!("  Data Size:  0x{:08X}", aout_header.dsize);
        println!("  BSS Size:   0x{:08X}", aout_header.bsize);
        println!("  Entry:      0x{:08X}", aout_header.entry);
        println!("  Text Start: 0x{:08X}", aout_header.text_start);
        println!("  Data Start: 0x{:08X}", aout_header.data_start);
        if file_header.f_opthdr >= 56 {
            println!("  BSS Start:  0x{:08X}", aout_header.bss_start);
            println!("  GPR Mask:   0x{:08X}", aout_header.gprmask);
            println!("  GP Value:   0x{:08X}", aout_header.gp_value);
        }
    }

    // Read Section Headers
    println!("\nSections:");
    let mut sections = Vec::new();
    for i in 0..file_header.f_nscns {
        let mut name_buf = [0u8; 8];
        f.read_exact(&mut name_buf)?;
        let name = String::from_utf8_lossy(&name_buf).trim_matches('\0').to_string();

        let mut sh = SectionHeader::default();
        sh.s_name = name;
        sh.s_paddr = read_u32(&mut f)?;
        sh.s_vaddr = read_u32(&mut f)?;
        sh.s_size = read_u32(&mut f)?;
        sh.s_scnptr = read_u32(&mut f)?;
        sh.s_relptr = read_u32(&mut f)?;
        sh.s_lnnoptr = read_u32(&mut f)?;
        sh.s_nreloc = read_u16(&mut f)?;
        sh.s_nlnno = read_u16(&mut f)?;
        sh.s_flags = read_u32(&mut f)?;

        println!("  [{:2}] {:<8} PAddr: {:08X} VAddr: {:08X} Size: {:08X} Flags: {:08X}", 
            i, sh.s_name, sh.s_paddr, sh.s_vaddr, sh.s_size, sh.s_flags);
        
        if sh.s_flags & STYP_TEXT != 0 { print!(" TEXT"); }
        if sh.s_flags & STYP_DATA != 0 { print!(" DATA"); }
        if sh.s_flags & STYP_BSS != 0 { print!(" BSS"); }
        if sh.s_flags & STYP_RDATA != 0 { print!(" RDATA"); }
        println!();
        sections.push(sh);
    }

    // Dump Relocations
    println!("\nRelocations:");
    for (i, sh) in sections.iter().enumerate() {
        if sh.s_nreloc > 0 {
            println!("  Section [{:2}] {} ({} entries):", i, sh.s_name, sh.s_nreloc);
            f.seek(SeekFrom::Start(sh.s_relptr as u64))?;
            for _ in 0..sh.s_nreloc {
                let r_vaddr = read_u32(&mut f)?;
                let r_info = read_u32(&mut f)?;
                
                let r_symndx = (r_info >> 8) & 0xFFFFFF;
                let r_type = (r_info >> 1) & 0xF;
                let r_extern = r_info & 1;
                
                let type_str = match r_type as u16 {
                    R_ABS => "R_ABS",
                    R_REFHALF => "R_REFHALF",
                    R_REFWORD => "R_REFWORD",
                    R_JMPADDR => "R_JMPADDR",
                    R_REFHI => "R_REFHI",
                    R_REFLO => "R_REFLO",
                    R_GPREL => "R_GPREL",
                    R_LITERAL => "R_LITERAL",
                    _ => "UNKNOWN",
                };
                
                println!("    VAddr: {:08X}  SymNdx: {:06X}  Type: {:02X} ({}) Extern: {}", 
                    r_vaddr, r_symndx, r_type, type_str, r_extern);
            }
        }
    }

    if entry_point != 0 {
        println!("\nEntry Point Dump (0x{:08X}):", entry_point);
        let mut found = false;
        for sh in &sections {
            if entry_point >= sh.s_vaddr && entry_point < sh.s_vaddr + sh.s_size {
                if sh.s_scnptr == 0 { continue; } // Skip sections with no file data (like BSS)
                let offset = sh.s_scnptr + (entry_point - sh.s_vaddr);
                f.seek(SeekFrom::Start(offset as u64))?;
                
                for i in 0..4 {
                    print!("{:08X}:", entry_point + (i * 16));
                    for _ in 0..4 {
                        match read_u32(&mut f) {
                            Ok(val) => print!(" {:08X}", val),
                            Err(_) => print!(" ????????"),
                        }
                    }
                    println!();
                }
                found = true;
                break;
            }
        }
        if !found {
            println!("Entry point not found in any loaded section.");
        }
    }

    // Dump Symbol Table
    if file_header.f_symptr != 0 {
        f.seek(SeekFrom::Start(file_header.f_symptr as u64))?;
        let magic_check = read_u16(&mut f)?;

        if magic_check == 0x7009 {
            // MIPS ECOFF Symbolic Header
            let mut hdrr = Hdrr::default();
            hdrr.magic = magic_check;
            hdrr.vstamp = read_u16(&mut f)?;
            hdrr.iline_max = read_u32(&mut f)?;
            hdrr.cb_line = read_u32(&mut f)?;
            hdrr.cb_line_offset = read_u32(&mut f)?;
            hdrr.idn_max = read_u32(&mut f)?;
            hdrr.cb_dn_offset = read_u32(&mut f)?;
            hdrr.ipd_max = read_u32(&mut f)?;
            hdrr.cb_pd_offset = read_u32(&mut f)?;
            hdrr.isym_max = read_u32(&mut f)?;
            hdrr.cb_sym_offset = read_u32(&mut f)?;
            hdrr.iopt_max = read_u32(&mut f)?;
            hdrr.cb_opt_offset = read_u32(&mut f)?;
            hdrr.iaux_max = read_u32(&mut f)?;
            hdrr.cb_aux_offset = read_u32(&mut f)?;
            hdrr.iss_max = read_u32(&mut f)?;
            hdrr.cb_ss_offset = read_u32(&mut f)?;
            hdrr.iss_ext_max = read_u32(&mut f)?;
            hdrr.cb_ss_ext_offset = read_u32(&mut f)?;
            hdrr.ifd_max = read_u32(&mut f)?;
            hdrr.cb_fd_offset = read_u32(&mut f)?;
            hdrr.crfd = read_u32(&mut f)?;
            hdrr.cb_rfd_offset = read_u32(&mut f)?;
            hdrr.iext_max = read_u32(&mut f)?;
            hdrr.cb_ext_offset = read_u32(&mut f)?;

            println!("\nMIPS Symbolic Header:");
            println!("  Magic: {:04X} VStamp: {:04X}", hdrr.magic, hdrr.vstamp);
            println!("  Local Symbols: {} at 0x{:X}", hdrr.isym_max, hdrr.cb_sym_offset);
            println!("  Ext Symbols:   {} at 0x{:X}", hdrr.iext_max, hdrr.cb_ext_offset);
            println!("  Local Strings: {} bytes at 0x{:X}", hdrr.iss_max, hdrr.cb_ss_offset);
            println!("  Ext Strings:   {} bytes at 0x{:X}", hdrr.iss_ext_max, hdrr.cb_ss_ext_offset);
            println!("  Files:         {} at 0x{:X}", hdrr.ifd_max, hdrr.cb_fd_offset);
            println!("  Procedures:    {} at 0x{:X}", hdrr.ipd_max, hdrr.cb_pd_offset);

            // Read Local Strings
            let mut local_strings = vec![0u8; hdrr.iss_max as usize];
            f.seek(SeekFrom::Start(hdrr.cb_ss_offset as u64))?;
            f.read_exact(&mut local_strings)?;

            // Read External Strings
            let mut ext_strings = vec![0u8; hdrr.iss_ext_max as usize];
            f.seek(SeekFrom::Start(hdrr.cb_ss_ext_offset as u64))?;
            f.read_exact(&mut ext_strings)?;

            // Read All Local Symbols
            let mut local_syms = Vec::with_capacity(hdrr.isym_max as usize);
            f.seek(SeekFrom::Start(hdrr.cb_sym_offset as u64))?;
            for _ in 0..hdrr.isym_max {
                let iss = read_u32(&mut f)?;
                let value = read_u32(&mut f)?;
                let info = read_u32(&mut f)?;
                local_syms.push(Symr {
                    iss,
                    value,
                    st: ((info >> 26) & 0x3F) as u8,
                    sc: ((info >> 21) & 0x1F) as u8,
                    index: info & 0xFFFFF,
                });
            }

            // Read Procedures
            let mut pdrs = Vec::with_capacity(hdrr.ipd_max as usize);
            f.seek(SeekFrom::Start(hdrr.cb_pd_offset as u64))?;
            for _ in 0..hdrr.ipd_max {
                pdrs.push(Pdr {
                    adr: read_u32(&mut f)?,
                    isym: read_u32(&mut f)?,
                    iline: read_u32(&mut f)?,
                    regmask: read_u32(&mut f)?,
                    regoffset: read_u32(&mut f)?,
                    iopt: read_u32(&mut f)?,
                    fregmask: read_u32(&mut f)?,
                    fregoffset: read_u32(&mut f)?,
                    frameoffset: read_u32(&mut f)?,
                    framereg: read_u16(&mut f)?,
                    pcreg: read_u16(&mut f)?,
                    ln_low: read_u32(&mut f)?,
                    ln_high: read_u32(&mut f)?,
                    cb_line_offset: read_u32(&mut f)?,
                });
            }

            // Dump Files and their Symbols/Procedures
            if hdrr.ifd_max > 0 {
                println!("\nFiles:");
                f.seek(SeekFrom::Start(hdrr.cb_fd_offset as u64))?;
                for i in 0..hdrr.ifd_max {
                    let fdr = Fdr {
                        adr: read_u32(&mut f)?,
                        rss: read_u32(&mut f)?,
                        iss_base: read_u32(&mut f)?,
                        cb_ss: read_u32(&mut f)?,
                        isym_base: read_u32(&mut f)?,
                        csym: read_u32(&mut f)?,
                        iline_base: read_u32(&mut f)?,
                        cline: read_u32(&mut f)?,
                        iopt_base: read_u32(&mut f)?,
                        copt: read_u32(&mut f)?,
                        ipd_first: read_u16(&mut f)?,
                        cpd: read_u16(&mut f)?,
                        iaux_base: read_u32(&mut f)?,
                        caux: read_u32(&mut f)?,
                        rfd_base: read_u32(&mut f)?,
                        crfd: read_u32(&mut f)?,
                        bits: read_u32(&mut f)?,
                        cb_line_offset: read_u32(&mut f)?,
                        cb_line: read_u32(&mut f)?,
                    };

                    let fname = get_string(&local_strings, (fdr.iss_base + fdr.rss) as usize);
                    println!("  File [{}]: {} (Addr: {:08X}, Syms: {}, Procs: {})", 
                        i, fname, fdr.adr, fdr.csym, fdr.cpd);

                    // Dump Procedures for this file
                    for p in 0..fdr.cpd {
                        let pdr_idx = (fdr.ipd_first + p) as usize;
                        if pdr_idx < pdrs.len() {
                            let pdr = &pdrs[pdr_idx];
                            // Try to find procedure name from its isym
                            let mut proc_name = "???".to_string();
                            if (pdr.isym as usize) < local_syms.len() {
                                let sym = &local_syms[pdr.isym as usize];
                                // For local symbols, iss is relative to fdr.iss_base
                                proc_name = get_string(&local_strings, (fdr.iss_base + sym.iss) as usize);
                            }
                            println!("    Proc: {:<20} Adr: {:08X} Frame: {:4} RegMask: {:08X}", 
                                proc_name, pdr.adr, pdr.frameoffset, pdr.regmask);
                        }
                    }

                    // Dump Symbols for this file
                    for s in 0..fdr.csym {
                        let sym_idx = (fdr.isym_base + s) as usize;
                        if sym_idx < local_syms.len() {
                            let sym = &local_syms[sym_idx];
                            let sym_name = get_string(&local_strings, (fdr.iss_base + sym.iss) as usize);
                            println!("    Sym:  {:<20} Val: {:08X} Type: {:<6} Class: {:<10} Index: {:5}", 
                                sym_name, sym.value, get_st_name(sym.st), get_sc_name(sym.sc), sym.index);
                        }
                    }
                }
            }

            // Dump External Symbols
            let mut map_syms: Vec<(u32, char, String)> = Vec::new();
            if hdrr.iext_max > 0 {
                println!("\nExternal Symbols:");
                f.seek(SeekFrom::Start(hdrr.cb_ext_offset as u64))?;
                for i in 0..hdrr.iext_max {
                    let info1 = read_u32(&mut f)?;
                    let ifd = (info1 & 0xFFFF) as u16;

                    let iss = read_u32(&mut f)?;
                    let value = read_u32(&mut f)?;
                    let info2 = read_u32(&mut f)?;

                    let st = (info2 >> 26) & 0x3F;
                    let sc = (info2 >> 21) & 0x1F;
                    let index = info2 & 0xFFFFF;

                    let name = get_string(&ext_strings, iss as usize);
                    println!("  [{:4}] {:<20} Val: {:08X} Type: {:<6} Class: {:<10} Index: {:5} Ifd: {}",
                        i, name, value, get_st_name(st as u8), get_sc_name(sc as u8), index, ifd);

                    // nm type character from storage class
                    let nm_char = match sc as u8 {
                        1  => 'T', // Text
                        2  => 'D', // Data
                        3  => 'B', // Bss
                        5  => 'A', // Abs
                        6  => 'U', // Undefined
                        13 => 'D', // SData
                        14 => 'B', // SBss
                        15 => 'R', // RData
                        _  => '?',
                    };
                    if nm_char != '?' && nm_char != 'U' && !name.is_empty() {
                        map_syms.push((value, nm_char, name));
                    }
                }
            }

            // Write .map file in nm format
            if !map_syms.is_empty() {
                map_syms.sort_by_key(|s| s.0);
                let map_path = format!("{}.map", args[1]);
                let mut map_file = File::create(&map_path)?;
                for (addr, ty, name) in &map_syms {
                    writeln!(map_file, "{:08x} {} {}", addr, ty, name)?;
                }
                println!("\nWrote {} symbols to {}", map_syms.len(), map_path);
            }
        } else if file_header.f_nsyms != 0 {
            // Standard COFF Symbol Table
            println!("\nSymbol Table (Standard COFF):");
            f.seek(SeekFrom::Start(file_header.f_symptr as u64))?;

            let sym_table_size = file_header.f_nsyms as usize * 18;
            let mut sym_table_data = vec![0u8; sym_table_size];
            f.read_exact(&mut sym_table_data)?;

            // Try to read string table (follows symbol table)
            let mut string_table = Vec::new();
            let mut str_len_buf = [0u8; 4];
            if f.read_exact(&mut str_len_buf).is_ok() {
                let str_len = u32::from_be_bytes(str_len_buf);
                if str_len > 4 && str_len < 10 * 1024 * 1024 { // Sanity check
                    let mut str_data = vec![0u8; (str_len - 4) as usize];
                    if f.read_exact(&mut str_data).is_ok() {
                        string_table.extend_from_slice(&str_len_buf);
                        string_table.extend_from_slice(&str_data);
                    }
                }
            }

            // Dump to binary file
            let dump_filename = format!("{}.syms", args[1]);
            let mut dump_file = File::create(&dump_filename)?;
            dump_file.write_all(&sym_table_data)?;
            if !string_table.is_empty() {
                dump_file.write_all(&string_table)?;
            }
            println!("Symbol table dumped to {}", dump_filename);

            // Decode symbols
            let mut i = 0;
            while i < file_header.f_nsyms as usize {
                let offset = i * 18;
                let entry = &sym_table_data[offset..offset+18];
                
                let n_value = u32::from_be_bytes(entry[8..12].try_into().unwrap());
                let n_scnum = i16::from_be_bytes(entry[12..14].try_into().unwrap());
                let n_type = u16::from_be_bytes(entry[14..16].try_into().unwrap());
                let n_sclass = entry[16];
                let n_numaux = entry[17];
                
                let name = if entry[0..4] == [0, 0, 0, 0] {
                    let str_offset = u32::from_be_bytes(entry[4..8].try_into().unwrap());
                    if !string_table.is_empty() && (str_offset as usize) < string_table.len() {
                        let start = str_offset as usize;
                        let mut end = start;
                        while end < string_table.len() && string_table[end] != 0 { end += 1; }
                        String::from_utf8_lossy(&string_table[start..end]).to_string()
                    } else {
                        format!("*OFF* {:08X}", str_offset)
                    }
                } else {
                    String::from_utf8_lossy(&entry[0..8]).trim_matches('\0').to_string()
                };
                
                println!("[{:4}] {:<20} Val:{:08X} Scn:{:3} Type:{:04X} Cls:{:2} Aux:{}", 
                    i, name, n_value, n_scnum, n_type, n_sclass, n_numaux);
                
                i += 1;
                for _ in 0..n_numaux {
                    if i < file_header.f_nsyms as usize {
                        let aux_offset = i * 18;
                        let aux = &sym_table_data[aux_offset..aux_offset+18];
                        println!("       Aux: {:02X?}", aux);
                        i += 1;
                    }
                }
            }
        }
    }

    Ok(())
}