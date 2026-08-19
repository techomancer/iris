//! Static ELF32 MSB reader for the monitor's `loadelf` and `--load-elf`.
//!
//! Only what a bare-metal MIPS test binary needs: the program headers and the
//! entry point. Byte-swapping here is correct — file parsing is "The Edge".

/// A `PT_LOAD` segment: `data` is `p_filesz` bytes to copy to `vaddr`, then
/// zero-fill out to `memsz`.
#[derive(Debug)]
pub struct Segment {
    pub vaddr: u64,
    pub memsz: u64,
    pub flags: u32,
    pub offset: usize,
    pub filesz: usize,
}

#[derive(Debug)]
pub struct Elf32 {
    pub entry: u64,
    pub segments: Vec<Segment>,
}

const PT_LOAD: u32 = 1;
const PT_DYNAMIC: u32 = 2;
const PT_INTERP: u32 = 3;

/// `rwx` for a segment's `p_flags`.
pub fn flag_str(flags: u32) -> String {
    let bit = |m: u32, c: char| if flags & m != 0 { c } else { '-' };
    [bit(4, 'r'), bit(2, 'w'), bit(1, 'x')].iter().collect()
}

/// Parse a static ELF32 MSB `ET_EXEC` for `EM_MIPS`. Anything else is rejected
/// by name rather than half-loaded.
pub fn parse(bytes: &[u8]) -> Result<Elf32, String> {
    if bytes.len() < 52 {
        return Err(format!("too short to be an ELF file ({} bytes)", bytes.len()));
    }
    if &bytes[0..4] != b"\x7fELF" {
        return Err("not an ELF file (bad magic)".to_string());
    }
    match bytes[4] {
        1 => {}
        2 => return Err("ELF64 is not supported — build a 32-bit (-melf32btsmip) binary".to_string()),
        c => return Err(format!("unknown ELF class {}", c)),
    }
    if bytes[5] != 2 {
        return Err("little-endian ELF — MIPS/IRIX binaries must be big-endian (MSB)".to_string());
    }

    let be16 = |o: usize| u16::from_be_bytes([bytes[o], bytes[o + 1]]);
    let be32 = |o: usize| u32::from_be_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);

    match be16(16) {
        2 => {}
        3 => return Err("ET_DYN (shared/PIE) — link statically as ET_EXEC".to_string()),
        t => return Err(format!("not an executable (e_type {})", t)),
    }
    if be16(18) != 8 {
        return Err(format!("not a MIPS binary (e_machine {})", be16(18)));
    }

    // 32-bit MIPS addresses are sign-extended, so KSEG0 works in both 32- and
    // 64-bit addressing modes.
    let sext = |v: u32| v as i32 as i64 as u64;

    let phoff = be32(28) as usize;
    let phentsize = be16(42) as usize;
    let phnum = be16(44) as usize;
    if phnum == 0 {
        return Err("no program headers — nothing to load".to_string());
    }
    if phentsize < 32 || phoff + phnum * phentsize > bytes.len() {
        return Err("program header table is out of bounds".to_string());
    }

    let mut segments = Vec::new();
    for i in 0..phnum {
        let p = phoff + i * phentsize;
        let ptype = be32(p);
        if ptype == PT_DYNAMIC || ptype == PT_INTERP {
            return Err("dynamically linked — link statically (-static)".to_string());
        }
        if ptype != PT_LOAD {
            continue;
        }
        let offset = be32(p + 4) as usize;
        let filesz = be32(p + 16) as usize;
        let memsz = be32(p + 20) as u64;
        if offset + filesz > bytes.len() {
            return Err(format!("segment {} runs past the end of the file", i));
        }
        if (filesz as u64) > memsz {
            return Err(format!("segment {} has p_filesz > p_memsz", i));
        }
        segments.push(Segment { vaddr: sext(be32(p + 8)), memsz, flags: be32(p + 24), offset, filesz });
    }
    if segments.is_empty() {
        return Err("no PT_LOAD segments — nothing to load".to_string());
    }

    Ok(Elf32 { entry: sext(be32(24)), segments })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal ELF32 MSB MIPS executable with one PT_LOAD segment.
    fn build_elf(payload: &[u8], vaddr: u32, memsz: u32) -> Vec<u8> {
        let mut v = vec![0u8; 52 + 32];
        v[0..4].copy_from_slice(b"\x7fELF");
        v[4] = 1; // ELFCLASS32
        v[5] = 2; // ELFDATA2MSB
        v[6] = 1; // EV_CURRENT
        v[16..18].copy_from_slice(&2u16.to_be_bytes()); // ET_EXEC
        v[18..20].copy_from_slice(&8u16.to_be_bytes()); // EM_MIPS
        v[24..28].copy_from_slice(&vaddr.to_be_bytes()); // e_entry
        v[28..32].copy_from_slice(&52u32.to_be_bytes()); // e_phoff
        v[42..44].copy_from_slice(&32u16.to_be_bytes()); // e_phentsize
        v[44..46].copy_from_slice(&1u16.to_be_bytes());  // e_phnum
        let p = 52;
        v[p..p + 4].copy_from_slice(&PT_LOAD.to_be_bytes());
        v[p + 4..p + 8].copy_from_slice(&84u32.to_be_bytes()); // p_offset
        v[p + 8..p + 12].copy_from_slice(&vaddr.to_be_bytes());
        v[p + 16..p + 20].copy_from_slice(&(payload.len() as u32).to_be_bytes());
        v[p + 20..p + 24].copy_from_slice(&memsz.to_be_bytes());
        v[p + 24..p + 28].copy_from_slice(&5u32.to_be_bytes()); // r-x
        v.extend_from_slice(payload);
        v
    }

    #[test]
    fn parses_a_static_be_mips_executable() {
        let elf = parse(&build_elf(&[0x24, 0x02, 0x00, 0x01], 0x8010_0000, 0x100)).unwrap();
        assert_eq!(elf.entry, 0xFFFF_FFFF_8010_0000, "32-bit vaddrs must be sign-extended");
        assert_eq!(elf.segments.len(), 1);
        let s = &elf.segments[0];
        assert_eq!(s.vaddr, 0xFFFF_FFFF_8010_0000);
        assert_eq!(s.filesz, 4);
        assert_eq!(s.memsz, 0x100);
        assert_eq!(flag_str(s.flags), "r-x");
        assert_eq!(s.offset, 84);
    }

    #[test]
    fn rejects_wrong_class_endianness_type_and_machine() {
        let good = build_elf(&[0; 4], 0x8010_0000, 4);

        let mut v = good.clone(); v[4] = 2;
        assert!(parse(&v).unwrap_err().contains("ELF64"));
        let mut v = good.clone(); v[5] = 1;
        assert!(parse(&v).unwrap_err().contains("little-endian"));
        let mut v = good.clone(); v[16..18].copy_from_slice(&3u16.to_be_bytes());
        assert!(parse(&v).unwrap_err().contains("ET_DYN"));
        let mut v = good.clone(); v[18..20].copy_from_slice(&243u16.to_be_bytes());
        assert!(parse(&v).unwrap_err().contains("not a MIPS binary"));
        let mut v = good.clone(); v[52..56].copy_from_slice(&PT_DYNAMIC.to_be_bytes());
        assert!(parse(&v).unwrap_err().contains("dynamically linked"));

        assert!(parse(b"not an elf at all, really truly not, padded out to 52 bytes ok").is_err());
        assert!(parse(&[]).unwrap_err().contains("too short"));
    }

    #[test]
    fn rejects_a_segment_that_runs_past_the_file() {
        let mut v = build_elf(&[0; 4], 0x8010_0000, 4);
        v[52 + 16..52 + 20].copy_from_slice(&0x1000u32.to_be_bytes()); // p_filesz
        v[52 + 20..52 + 24].copy_from_slice(&0x1000u32.to_be_bytes()); // p_memsz
        assert!(parse(&v).unwrap_err().contains("past the end"));
    }
}
