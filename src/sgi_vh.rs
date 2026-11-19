//! Minimal SGI Volume Header writer for the Phase 2.4 scratch volume.
//!
//! IRIX requires a recognisable partition table at sector 0 before the
//! `/dev/rdsk/dks0dNvol` and `/dev/rdsk/dks0dNvh` device nodes return real
//! data. Without one IRIX enumerates the SCSI target on `hinv` but every
//! read returns "I/O error". This module writes a 512-byte SGI Volume Header
//! into sector 0 of a freshly-created scratch image with two partition
//! entries:
//!
//! - **slot 0 ("payload")**: type 3 (`PT_RAW`), spans sectors 8..end. IRIX
//!   surfaces this as `/dev/rdsk/dks0dNs0`. This is the partition the host
//!   injects payload bytes into and that the guest reads — `first_block` is
//!   honoured so reads from offset 0 of `s0` map to byte 4096 of the disk
//!   (right after the VH).
//! - **slot 8 ("vh")**: type 0 (`PT_VOLHDR`), spans sectors 0..7. IRIX
//!   surfaces this as `/dev/rdsk/dks0dNvh`. Present only so IRIX's standard
//!   convention is satisfied; the host-side `scratch-write` never touches it.
//! - **slot 10 ("vol")**: type 6 (`PT_VOLUME`), spans the entire disk. IRIX
//!   surfaces this as `/dev/rdsk/dks0dNvol`. The `vol` partition by SGI
//!   convention always covers sector 0 onwards regardless of `first_block`,
//!   so reading it returns the VH first — use `s0` for payload reads.
//!
//! NB: IRIX raw block-device reads must be sector-aligned (multiples of 512
//! bytes). `dd if=/dev/rdsk/dks0dNs0 bs=512 count=N` works; `bs=64` returns
//! "Read error: I/O error" with no SCSI-level error.
//!
//! Convention: host writes payload at offset `SCRATCH_PAYLOAD_OFFSET` (4096
//! = sector 8). Guest reads payload from offset 0 of the `vol` partition,
//! which the kernel maps to sector 8 of the underlying disk.
//!
//! All values are big-endian per SGI convention.

use std::fs::File;
use std::io::{self, Seek, SeekFrom, Write};
use std::path::Path;

/// First payload byte. Reserved bytes 0..4095 hold the 8-sector VH partition.
pub const SCRATCH_PAYLOAD_OFFSET: u64 = 4096;

const SECTOR_SIZE: u64 = 512;
const VH_SECTORS: u64 = 8;
const SGI_MAGIC: u32 = 0x0BE5_A941;

const PT_VOLHDR: u32 = 0;
const PT_RAW:    u32 = 3;
const PT_VOLUME: u32 = 6;

const PT_TABLE_OFFSET: usize = 0x138;
const PT_ENTRY_SIZE: usize = 12;
const CSUM_OFFSET: usize = 0x1F8;

/// Create a fresh scratch image at `path` of `total_bytes` size, with a
/// minimal SGI Volume Header at sector 0. Overwrites any existing file.
pub fn create_scratch_image(path: &Path, total_bytes: u64) -> io::Result<()> {
    if total_bytes < SCRATCH_PAYLOAD_OFFSET + SECTOR_SIZE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "scratch size {} bytes is too small (minimum {} bytes)",
                total_bytes,
                SCRATCH_PAYLOAD_OFFSET + SECTOR_SIZE
            ),
        ));
    }
    if total_bytes % SECTOR_SIZE != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("scratch size {} is not a multiple of {} bytes", total_bytes, SECTOR_SIZE),
        ));
    }

    let total_sectors = total_bytes / SECTOR_SIZE;
    let vol_sectors = total_sectors - VH_SECTORS;

    let mut vh = build_vh(vol_sectors);
    fix_csum(&mut vh);

    let f = File::create(path)?;
    f.set_len(total_bytes)?;
    let mut f = f;
    f.write_all(&vh)?;
    f.sync_all()?;
    Ok(())
}

fn build_vh(vol_sectors: u64) -> [u8; SECTOR_SIZE as usize] {
    let mut vh = [0u8; SECTOR_SIZE as usize];

    // Magic.
    vh[0..4].copy_from_slice(&SGI_MAGIC.to_be_bytes());

    // root_partnum / swap_partnum / bootfile / device_parameters all stay 0.

    // Partition table at PT_TABLE_OFFSET (0x138).
    // Slot 0 ("payload"): type PT_RAW, sectors 8..end. IRIX maps this to
    // /dev/rdsk/dks0dNs0 with first_block honoured — reads at offset 0 of
    // s0 land at byte 4096 of the disk (right after the VH).
    write_pt_entry(&mut vh, 0, vol_sectors as u32, VH_SECTORS as u32, PT_RAW);
    // Slot 8 ("vh"): type PT_VOLHDR, sectors 0..7. IRIX maps this to
    // /dev/rdsk/dks0dNvh.
    write_pt_entry(&mut vh, 8, VH_SECTORS as u32, 0, PT_VOLHDR);
    // Slot 10 ("vol"): type PT_VOLUME, whole disk. IRIX maps this to
    // /dev/rdsk/dks0dNvol — convenient for raw whole-disk dumps but always
    // starts at sector 0 (the VH), so use s0 for payload reads.
    let total_sectors_u32 = (vol_sectors + VH_SECTORS) as u32;
    write_pt_entry(&mut vh, 10, total_sectors_u32, 0, PT_VOLUME);

    vh
}

fn write_pt_entry(vh: &mut [u8; SECTOR_SIZE as usize], slot: usize, nblks: u32, first: u32, ty: u32) {
    let off = PT_TABLE_OFFSET + slot * PT_ENTRY_SIZE;
    vh[off..off + 4].copy_from_slice(&nblks.to_be_bytes());
    vh[off + 4..off + 8].copy_from_slice(&first.to_be_bytes());
    vh[off + 8..off + 12].copy_from_slice(&ty.to_be_bytes());
}

/// Set csum so the 32-bit two's-complement sum of all 128 big-endian words
/// equals zero. fx, prtvtoc, and the IRIX kernel all check this.
fn fix_csum(vh: &mut [u8; SECTOR_SIZE as usize]) {
    // Zero the existing csum first, then sum, then store -sum.
    vh[CSUM_OFFSET..CSUM_OFFSET + 4].fill(0);
    let mut sum: u32 = 0;
    for chunk in vh.chunks_exact(4) {
        let w = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        sum = sum.wrapping_add(w);
    }
    let csum = (!sum).wrapping_add(1); // -sum
    vh[CSUM_OFFSET..CSUM_OFFSET + 4].copy_from_slice(&csum.to_be_bytes());
}

// ---------------------------------------------------------------------------
// Volume directory: named standalone files the PROM boots by name, e.g.
// `boot -f dksc(0,1,8)cputest`. 15 entries of 16 bytes at 0x048.
// ---------------------------------------------------------------------------

const BOOTFILE_OFFSET: usize = 0x008;
const BOOTFILE_LEN: usize = 16;
const VD_OFFSET: usize = 0x048;
const VD_ENTRY_SIZE: usize = 16;
/// Volume directory slots in `struct volume_header`.
pub const VD_MAX_ENTRIES: usize = 15;
/// `vd_name` is 8 bytes, NUL-padded and *not* NUL-terminated when full.
pub const VD_NAME_LEN: usize = 8;
/// Partition-table slots in `struct volume_header`.
pub const PT_MAX_ENTRIES: usize = 16;

/// One volume-directory entry. `lbn` is a 512-byte block from the start of the
/// volume; `nbytes` is the exact file length.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VolDirEntry {
    pub name: String,
    pub lbn: u32,
    pub nbytes: u32,
}

/// One partition-table entry, with the slot it occupies.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartEntry {
    pub slot: usize,
    pub nblks: u32,
    pub first_block: u32,
    pub ptype: u32,
}

/// A 512-byte SGI volume header, mutable in place. Every mutation refreshes the
/// checksum, so `as_bytes()` is always valid to write to sector 0.
#[derive(Clone)]
pub struct VolumeHeader {
    raw: [u8; SECTOR_SIZE as usize],
}

impl Default for VolumeHeader {
    fn default() -> Self { Self::new() }
}

impl VolumeHeader {
    /// An empty header: magic + valid checksum, no files, no partitions.
    pub fn new() -> Self {
        let mut raw = [0u8; SECTOR_SIZE as usize];
        raw[0..4].copy_from_slice(&SGI_MAGIC.to_be_bytes());
        let mut vh = Self { raw };
        fix_csum(&mut vh.raw);
        vh
    }

    /// Parse sector 0 of an existing image. Rejects anything without SGI magic.
    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        if bytes.len() < SECTOR_SIZE as usize {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "volume header is shorter than 512 bytes"));
        }
        let magic = u32::from_be_bytes(bytes[0..4].try_into().unwrap());
        if magic != SGI_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("bad SGI magic {:#010x} (expected {:#010x})", magic, SGI_MAGIC),
            ));
        }
        let mut raw = [0u8; SECTOR_SIZE as usize];
        raw.copy_from_slice(&bytes[..SECTOR_SIZE as usize]);
        Ok(Self { raw })
    }

    pub fn as_bytes(&self) -> &[u8; SECTOR_SIZE as usize] { &self.raw }

    /// `vh_rootpt` / `vh_swappt`: default root and swap partition numbers.
    pub fn root_swap_parts(&self) -> (u16, u16) {
        (
            u16::from_be_bytes(self.raw[4..6].try_into().unwrap()),
            u16::from_be_bytes(self.raw[6..8].try_into().unwrap()),
        )
    }

    /// Default boot file (`vh_bootfile`), NUL-trimmed.
    pub fn bootfile(&self) -> String {
        trim_nul(&self.raw[BOOTFILE_OFFSET..BOOTFILE_OFFSET + BOOTFILE_LEN])
    }

    pub fn set_bootfile(&mut self, name: &str) -> io::Result<()> {
        if name.len() > BOOTFILE_LEN {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("bootfile name '{}' exceeds {} characters", name, BOOTFILE_LEN),
            ));
        }
        self.raw[BOOTFILE_OFFSET..BOOTFILE_OFFSET + BOOTFILE_LEN].fill(0);
        self.raw[BOOTFILE_OFFSET..BOOTFILE_OFFSET + name.len()].copy_from_slice(name.as_bytes());
        fix_csum(&mut self.raw);
        Ok(())
    }

    /// Record a file in the first free voldir slot.
    pub fn add_file(&mut self, name: &str, lbn: u32, nbytes: u32) -> io::Result<()> {
        if name.is_empty() || name.len() > VD_NAME_LEN {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("voldir name '{}' must be 1..={} characters", name, VD_NAME_LEN),
            ));
        }
        if !name.is_ascii() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("voldir name '{}' must be ASCII", name)));
        }
        let slot = (0..VD_MAX_ENTRIES)
            .find(|&i| self.raw[VD_OFFSET + i * VD_ENTRY_SIZE] == 0)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("volume directory is full ({} entries)", VD_MAX_ENTRIES),
                )
            })?;
        let off = VD_OFFSET + slot * VD_ENTRY_SIZE;
        self.raw[off..off + VD_ENTRY_SIZE].fill(0);
        self.raw[off..off + name.len()].copy_from_slice(name.as_bytes());
        self.raw[off + 8..off + 12].copy_from_slice(&lbn.to_be_bytes());
        self.raw[off + 12..off + 16].copy_from_slice(&nbytes.to_be_bytes());
        fix_csum(&mut self.raw);
        Ok(())
    }

    /// Every non-empty voldir entry, in slot order.
    pub fn files(&self) -> Vec<VolDirEntry> {
        (0..VD_MAX_ENTRIES)
            .filter_map(|i| {
                let off = VD_OFFSET + i * VD_ENTRY_SIZE;
                if self.raw[off] == 0 { return None; }
                Some(VolDirEntry {
                    name: trim_nul(&self.raw[off..off + VD_NAME_LEN]),
                    lbn: u32::from_be_bytes(self.raw[off + 8..off + 12].try_into().unwrap()),
                    nbytes: u32::from_be_bytes(self.raw[off + 12..off + 16].try_into().unwrap()),
                })
            })
            .collect()
    }

    pub fn set_partition(&mut self, slot: usize, nblks: u32, first_block: u32, ptype: u32) -> io::Result<()> {
        if slot >= PT_MAX_ENTRIES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("partition slot {} out of range (0..{})", slot, PT_MAX_ENTRIES),
            ));
        }
        write_pt_entry(&mut self.raw, slot, nblks, first_block, ptype);
        fix_csum(&mut self.raw);
        Ok(())
    }

    /// Every partition slot with a non-zero size, in slot order.
    pub fn partitions(&self) -> Vec<PartEntry> {
        (0..PT_MAX_ENTRIES)
            .filter_map(|slot| {
                let off = PT_TABLE_OFFSET + slot * PT_ENTRY_SIZE;
                let nblks = u32::from_be_bytes(self.raw[off..off + 4].try_into().unwrap());
                if nblks == 0 { return None; }
                Some(PartEntry {
                    slot,
                    nblks,
                    first_block: u32::from_be_bytes(self.raw[off + 4..off + 8].try_into().unwrap()),
                    ptype: u32::from_be_bytes(self.raw[off + 8..off + 12].try_into().unwrap()),
                })
            })
            .collect()
    }

    /// True when the 128 big-endian words sum to zero, as IRIX and the PROM check.
    pub fn csum_valid(&self) -> bool {
        let mut sum: u32 = 0;
        for chunk in self.raw.chunks_exact(4) {
            sum = sum.wrapping_add(u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        sum == 0
    }
}

fn trim_nul(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

/// One partition-table slot in an image description, with optional raw contents.
///
/// `first_block: None` places the partition immediately after the volume-header
/// partition, the way SGI's own media does — on the 6.5.22 Installation Tools CD
/// partition 7 starts at block 48736, exactly partition 8's length.
#[derive(Clone, Debug)]
pub struct PartitionSpec {
    pub slot: usize,
    pub first_block: Option<u32>,
    pub nblks: u32,
    pub ptype: u32,
    pub content: Option<std::path::PathBuf>,
}

/// Description of a bootable raw image: voldir files plus partition entries.
#[derive(Clone, Debug, Default)]
pub struct ImageSpec {
    /// Image size in bytes; 0 sizes the image to fit its contents.
    pub total_bytes: u64,
    /// `vh_bootfile`; defaults to the first voldir file.
    pub bootfile: Option<String>,
    /// Voldir name -> host file to copy in.
    pub files: Vec<(String, std::path::PathBuf)>,
    pub partitions: Vec<PartitionSpec>,
    /// Split each ELF file's `PT_LOAD` segments so none carries more than
    /// this many bytes of file data ([`split_load_segments`]); 0 copies the
    /// files unchanged. [`PROM_SEGMENT_BYTES`] is what the PROMs can read.
    pub max_segment: u64,
}

/// The most file data one `PT_LOAD` segment of a volume-header file should
/// carry for the PROM to load it.
///
/// The PROM reads each segment from the volume header with one SCSI command.
/// The Octane's PROM (4.17) describes the transfer to its ISP1040 as one
/// entry per 16 KB page, in a fixed array in its per-adapter state with room
/// for about 73, and queues it on an 8-entry request ring (one command entry
/// with 4 data segments, continuations with 7). A segment of more than 45
/// pages, about 720 KB, never fits the ring and the command times out; one of
/// more than 73 pages also writes its page list over the next adapter's
/// state, and the PROM panics. 512 KB leaves room for a segment that does not
/// start on a page. IRIX's own volume-header files (`sash`, `IP30prom`) stay
/// under the limit; an 11 MB Linux `vmlinux` does not.
pub const PROM_SEGMENT_BYTES: u64 = 512 * 1024;

/// A voldir file name and what [`split_load_segments`] did to it.
pub type SplitFile = (String, SegmentSplit);

/// What [`split_load_segments`] did to one ELF file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SegmentSplit {
    /// `PT_LOAD` segments before and after.
    pub loads_before: usize,
    pub loads_after: usize,
    /// The largest segment's file data before the split.
    pub largest: u64,
}

/// Split every `PT_LOAD` segment of an MSB ELF executable (`ET_EXEC`, 32- or
/// 64-bit) that carries more than `max` bytes of file data into consecutive
/// segments of at most `max` bytes each, so a loader that reads a segment at
/// a time ([`PROM_SEGMENT_BYTES`]) reads it in pieces. The pieces cover the
/// same file bytes at the same addresses; the last one keeps the zero fill
/// (`p_memsz` beyond `p_filesz`). Other program headers are kept, in order.
///
/// The new program header table goes where the old one was when the bytes
/// it grows into belong to no segment, section or section header table, and
/// after the end of the file otherwise.
///
/// `Ok(None)` when the file is not such an ELF executable or needs no split.
pub fn split_load_segments(data: &[u8], max: u64) -> Result<Option<(Vec<u8>, SegmentSplit)>, String> {
    if max == 0 || data.len() < 52 || &data[0..4] != b"\x7fELF" || data[5] != 2 {
        return Ok(None);
    }
    let wide = match data[4] {
        1 => false,
        2 => true,
        _ => return Ok(None),
    };
    let rd = |off: usize, n: usize| -> Result<u64, String> {
        let b = data.get(off..off + n).ok_or_else(|| format!("ELF header field at {off:#x} is past the end of the file"))?;
        Ok(b.iter().fold(0u64, |v, &x| (v << 8) | u64::from(x)))
    };
    if rd(16, 2)? != 2 {
        return Ok(None); // not ET_EXEC: a relocatable sash, say
    }
    // (offset, size) of e_phoff, e_shoff; then e_phentsize, e_phnum, e_shentsize, e_shnum.
    let (phoff_at, shoff_at, word, half_at) = if wide { (0x20, 0x28, 8, 0x36) } else { (0x1c, 0x20, 4, 0x2a) };
    let phoff = rd(phoff_at, word)? as usize;
    let shoff = rd(shoff_at, word)? as usize;
    let phentsize = rd(half_at, 2)? as usize;
    let phnum = rd(half_at + 2, 2)? as usize;
    let shentsize = rd(half_at + 4, 2)? as usize;
    let shnum = rd(half_at + 6, 2)? as usize;
    let want = if wide { 56 } else { 32 };
    if phnum == 0 {
        return Ok(None);
    }
    if phentsize != want {
        return Err(format!("program header entries are {phentsize} bytes, not {want}"));
    }

    // p_type, p_flags, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz, p_align.
    type Ph = [u64; 8];
    let read_ph = |i: usize| -> Result<Ph, String> {
        let b = phoff + i * phentsize;
        Ok(if wide {
            [rd(b, 4)?, rd(b + 4, 4)?, rd(b + 8, 8)?, rd(b + 16, 8)?, rd(b + 24, 8)?, rd(b + 32, 8)?, rd(b + 40, 8)?, rd(b + 48, 8)?]
        } else {
            [rd(b, 4)?, rd(b + 24, 4)?, rd(b + 4, 4)?, rd(b + 8, 4)?, rd(b + 12, 4)?, rd(b + 16, 4)?, rd(b + 20, 4)?, rd(b + 28, 4)?]
        })
    };
    let old: Vec<Ph> = (0..phnum).map(read_ph).collect::<Result<_, _>>()?;
    const PT_LOAD: u64 = 1;
    let loads_before = old.iter().filter(|p| p[0] == PT_LOAD).count();
    let largest = old.iter().filter(|p| p[0] == PT_LOAD).map(|p| p[5]).max().unwrap_or(0);
    if largest <= max {
        return Ok(None);
    }

    let mut new: Vec<Ph> = Vec::new();
    for p in &old {
        let [ty, flags, off, vaddr, paddr, filesz, memsz, align] = *p;
        if ty != PT_LOAD || filesz <= max {
            new.push(*p);
            continue;
        }
        if off.checked_add(filesz).is_none_or(|end| end > data.len() as u64) || memsz < filesz {
            return Err(format!("PT_LOAD at {vaddr:#x} lies outside the file or is shorter in memory than on disk"));
        }
        let mut done = 0;
        while done < filesz {
            let n = (filesz - done).min(max);
            let last = done + n == filesz;
            let m = if last { memsz - done } else { n };
            new.push([ty, flags, off + done, vaddr + done, paddr + done, n, m, align]);
            done += n;
        }
    }
    if new.len() > 0xfffe {
        return Err(format!("splitting would need {} program headers", new.len()));
    }
    let loads_after = new.iter().filter(|p| p[0] == PT_LOAD).count();

    // Byte ranges the new table must not overwrite if it stays in place.
    let mut used: Vec<(u64, u64)> = new.iter().filter(|p| p[5] > 0).map(|p| (p[2], p[2] + p[5])).collect();
    if shoff != 0 && shnum != 0 {
        used.push((shoff as u64, (shoff + shnum * shentsize) as u64));
        for i in 0..shnum {
            let b = shoff + i * shentsize;
            let (ty, off, size) = if wide { (rd(b + 4, 4)?, rd(b + 24, 8)?, rd(b + 32, 8)?) } else { (rd(b + 4, 4)?, rd(b + 16, 4)?, rd(b + 20, 4)?) };
            const SHT_NOBITS: u64 = 8;
            if ty != SHT_NOBITS && size > 0 {
                used.push((off, off + size));
            }
        }
    }
    let table = (new.len() * phentsize) as u64;
    let fits = |at: u64| at + table <= data.len() as u64 && !used.iter().any(|&(a, b)| a < at + table && at < b);
    let mut out = data.to_vec();
    let at = if fits(phoff as u64) {
        phoff as u64
    } else {
        out.resize(out.len().next_multiple_of(8), 0);
        let at = out.len() as u64;
        out.resize(out.len() + table as usize, 0);
        at
    };

    let put = |out: &mut Vec<u8>, off: usize, n: usize, v: u64| {
        for i in 0..n {
            out[off + i] = (v >> (8 * (n - 1 - i))) as u8;
        }
    };
    for (i, p) in new.iter().enumerate() {
        let b = at as usize + i * phentsize;
        let [ty, flags, off, vaddr, paddr, filesz, memsz, align] = *p;
        if wide {
            for (o, n, v) in [(0, 4, ty), (4, 4, flags), (8, 8, off), (16, 8, vaddr), (24, 8, paddr), (32, 8, filesz), (40, 8, memsz), (48, 8, align)] {
                put(&mut out, b + o, n, v);
            }
        } else {
            for (o, v) in [(0, ty), (4, off), (8, vaddr), (12, paddr), (16, filesz), (20, memsz), (24, flags), (28, align)] {
                put(&mut out, b + o, 4, v);
            }
        }
    }
    put(&mut out, phoff_at, word, at);
    put(&mut out, half_at + 2, 2, new.len() as u64);
    Ok(Some((out, SegmentSplit { loads_before, loads_after, largest })))
}

/// Build a complete raw image from `spec` and return the placed voldir entries.
///
/// Files are laid out sector-aligned starting at block `VH_SECTORS`, and the
/// volume-header partition (slot 8) is sized to cover them unless the caller
/// supplies slot 8 itself.
pub fn build_image(path: &Path, spec: &ImageSpec) -> io::Result<Vec<VolDirEntry>> {
    build_image_with_splits(path, spec).map(|(files, _)| files)
}

/// [`build_image`], also returning each voldir file whose segments were split
/// ([`ImageSpec::max_segment`]) with what was done to it.
pub fn build_image_with_splits(path: &Path, spec: &ImageSpec) -> io::Result<(Vec<VolDirEntry>, Vec<SplitFile>)> {
    let mut vh = VolumeHeader::new();
    let mut placed: Vec<(u32, Vec<u8>)> = Vec::new();
    let mut splits = Vec::new();
    let mut lbn = VH_SECTORS as u32;

    for (name, src) in &spec.files {
        let mut data = std::fs::read(src)
            .map_err(|e| io::Error::new(e.kind(), format!("{}: {}", src.display(), e)))?;
        match split_load_segments(&data, spec.max_segment) {
            Ok(Some((split, what))) => {
                data = split;
                splits.push((name.clone(), what));
            }
            Ok(None) => {}
            Err(e) => return Err(io::Error::new(io::ErrorKind::InvalidData, format!("{}: {}", src.display(), e))),
        }
        let nbytes = u32::try_from(data.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, format!("{} is larger than 4 GB", src.display())))?;
        vh.add_file(name, lbn, nbytes)?;
        lbn += sectors_for(data.len() as u64) as u32;
        placed.push((lbn - sectors_for(data.len() as u64) as u32, data));
    }

    let bootfile = spec.bootfile.clone().or_else(|| spec.files.first().map(|(n, _)| n.clone()));
    if let Some(name) = bootfile { vh.set_bootfile(&name)?; }

    // Slot 8 is the volume header partition by SGI convention, and it must span
    // every file the voldir points at — the 6.5.22 Installation Tools CD gives it
    // 48736 blocks for ~24 MB of sash/miniroot, not the 8-block scratch minimum.
    let vh_blocks = lbn.max(VH_SECTORS as u32);
    let vh_blocks = spec
        .partitions
        .iter()
        .find(|p| p.slot == 8)
        .map_or(vh_blocks, |p| p.nblks.max(vh_blocks));
    vh.set_partition(8, vh_blocks, 0, PT_VOLHDR)?;

    // A partition with no explicit first_block starts right after the header.
    let placed_parts: Vec<(usize, u32, u32, u32, Option<&std::path::PathBuf>)> = spec
        .partitions
        .iter()
        .filter(|p| p.slot != 8)
        .map(|p| (p.slot, p.first_block.unwrap_or(vh_blocks), p.nblks, p.ptype, p.content.as_ref()))
        .collect();
    for (slot, first, nblks, ptype, _) in &placed_parts {
        vh.set_partition(*slot, *nblks, *first, *ptype)?;
    }

    // End of the last byte any partition or file claims, rounded up to a sector.
    let mut end_blocks = vh_blocks as u64;
    for (_, first, nblks, _, _) in &placed_parts {
        end_blocks = end_blocks.max(*first as u64 + *nblks as u64);
    }
    let total_bytes = if spec.total_bytes == 0 {
        end_blocks * SECTOR_SIZE
    } else {
        if spec.total_bytes % SECTOR_SIZE != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("image size {} is not a multiple of {} bytes", spec.total_bytes, SECTOR_SIZE),
            ));
        }
        if spec.total_bytes < end_blocks * SECTOR_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("image size {} is too small for its contents ({} bytes)", spec.total_bytes, end_blocks * SECTOR_SIZE),
            ));
        }
        spec.total_bytes
    };

    // Slot 10 ("vol") covers the whole disk, as create_scratch_image does.
    if !placed_parts.iter().any(|(slot, ..)| *slot == 10) {
        vh.set_partition(10, (total_bytes / SECTOR_SIZE) as u32, 0, PT_VOLUME)?;
    }

    let mut f = File::create(path)?;
    f.set_len(total_bytes)?;
    f.write_all(vh.as_bytes())?;
    for (block, data) in &placed {
        f.seek(SeekFrom::Start(*block as u64 * SECTOR_SIZE))?;
        f.write_all(data)?;
    }
    for (slot, first, nblks, _, content) in &placed_parts {
        let Some(src) = content else { continue };
        let data = std::fs::read(src)
            .map_err(|e| io::Error::new(e.kind(), format!("{}: {}", src.display(), e)))?;
        if data.len() as u64 > *nblks as u64 * SECTOR_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("{} ({} bytes) does not fit partition {} ({} blocks)", src.display(), data.len(), slot, nblks),
            ));
        }
        f.seek(SeekFrom::Start(*first as u64 * SECTOR_SIZE))?;
        f.write_all(&data)?;
    }
    f.sync_all()?;
    Ok((vh.files(), splits))
}

fn sectors_for(bytes: u64) -> u64 {
    bytes.div_ceil(SECTOR_SIZE)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unique_tmp_path(tag: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("iris-vh-{}-{}.raw", tag, nanos))
    }

    #[test]
    fn scratch_image_has_correct_size_and_magic() {
        let p = unique_tmp_path("size");
        let size: u64 = 4 * 1024 * 1024; // 4 MB
        create_scratch_image(&p, size).expect("create");
        let meta = std::fs::metadata(&p).unwrap();
        assert_eq!(meta.len(), size, "image size must match request");
        let bytes = std::fs::read(&p).unwrap();
        assert_eq!(&bytes[0..4], &SGI_MAGIC.to_be_bytes(), "missing SGI magic");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn partition_table_describes_vol_and_vh() {
        let p = unique_tmp_path("pt");
        let size: u64 = 64 * 1024 * 1024;
        create_scratch_image(&p, size).expect("create");
        let bytes = std::fs::read(&p).unwrap();

        // Slot 0 (payload): nblks = total - 8, first = 8, type = PT_RAW.
        let off0 = PT_TABLE_OFFSET;
        let nblks = u32::from_be_bytes(bytes[off0..off0 + 4].try_into().unwrap());
        let first = u32::from_be_bytes(bytes[off0 + 4..off0 + 8].try_into().unwrap());
        let ty    = u32::from_be_bytes(bytes[off0 + 8..off0 + 12].try_into().unwrap());
        assert_eq!(nblks, (size / SECTOR_SIZE - VH_SECTORS) as u32);
        assert_eq!(first, VH_SECTORS as u32);
        assert_eq!(ty, PT_RAW);

        // Slot 8 (vh): nblks = 8, first = 0, type = PT_VOLHDR.
        let off8 = PT_TABLE_OFFSET + 8 * PT_ENTRY_SIZE;
        let nblks = u32::from_be_bytes(bytes[off8..off8 + 4].try_into().unwrap());
        let first = u32::from_be_bytes(bytes[off8 + 4..off8 + 8].try_into().unwrap());
        let ty    = u32::from_be_bytes(bytes[off8 + 8..off8 + 12].try_into().unwrap());
        assert_eq!(nblks, VH_SECTORS as u32);
        assert_eq!(first, 0);
        assert_eq!(ty, PT_VOLHDR);

        // Slot 10 (vol): nblks = total, first = 0, type = PT_VOLUME (whole disk).
        let off10 = PT_TABLE_OFFSET + 10 * PT_ENTRY_SIZE;
        let nblks = u32::from_be_bytes(bytes[off10..off10 + 4].try_into().unwrap());
        let first = u32::from_be_bytes(bytes[off10 + 4..off10 + 8].try_into().unwrap());
        let ty    = u32::from_be_bytes(bytes[off10 + 8..off10 + 12].try_into().unwrap());
        assert_eq!(nblks, (size / SECTOR_SIZE) as u32);
        assert_eq!(first, 0);
        assert_eq!(ty, PT_VOLUME);

        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn checksum_sums_to_zero() {
        let p = unique_tmp_path("csum");
        let size: u64 = 64 * 1024 * 1024;
        create_scratch_image(&p, size).expect("create");
        let bytes = std::fs::read(&p).unwrap();
        let mut sum: u32 = 0;
        for chunk in bytes[..512].chunks_exact(4) {
            let w = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            sum = sum.wrapping_add(w);
        }
        assert_eq!(sum, 0, "VH csum must make 32-bit sum of 128 BE words == 0");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn rejects_too_small_image() {
        let p = unique_tmp_path("small");
        let r = create_scratch_image(&p, 4096); // exactly VH size, no payload
        assert!(r.is_err());
    }

    #[test]
    fn rejects_non_sector_aligned_size() {
        let p = unique_tmp_path("misaligned");
        let r = create_scratch_image(&p, 4096 + 100);
        assert!(r.is_err());
    }

    #[test]
    fn voldir_round_trips() {
        let mut vh = VolumeHeader::new();
        vh.add_file("sash", 4, 300_000).unwrap();
        vh.add_file("cputest", 1024, 8192).unwrap();
        let files = vh.files();
        assert_eq!(files.len(), 2);
        assert_eq!(files[0], VolDirEntry { name: "sash".into(), lbn: 4, nbytes: 300_000 });
        assert_eq!(files[1], VolDirEntry { name: "cputest".into(), lbn: 1024, nbytes: 8192 });

        let reparsed = VolumeHeader::from_bytes(vh.as_bytes()).unwrap();
        assert_eq!(reparsed.files(), files);
        assert!(reparsed.csum_valid());
    }

    #[test]
    fn eight_char_name_is_not_nul_terminated() {
        let mut vh = VolumeHeader::new();
        vh.add_file("sashARCS", 2, 512).unwrap();
        let off = VD_OFFSET;
        assert_eq!(&vh.as_bytes()[off..off + 8], b"sashARCS", "name must fill all 8 bytes");
        assert_eq!(vh.files()[0].name, "sashARCS");
        assert_eq!(vh.files()[0].lbn, 2, "lbn must start at byte 8, not after a NUL");
    }

    #[test]
    fn checksum_stays_valid_after_every_mutation() {
        let mut vh = VolumeHeader::new();
        assert!(vh.csum_valid());
        vh.set_bootfile("cputest").unwrap();
        assert!(vh.csum_valid());
        vh.add_file("cputest", 8, 4096).unwrap();
        assert!(vh.csum_valid());
        vh.set_partition(7, 100, 8, PT_RAW).unwrap();
        assert!(vh.csum_valid());
    }

    #[test]
    fn rejects_bad_names_and_a_full_directory() {
        let mut vh = VolumeHeader::new();
        assert!(vh.add_file("toolongname", 0, 0).is_err(), "9+ char name must be rejected");
        assert!(vh.add_file("", 0, 0).is_err(), "empty name must be rejected");
        for i in 0..VD_MAX_ENTRIES {
            vh.add_file(&format!("f{}", i), i as u32, 1).unwrap();
        }
        assert!(vh.add_file("onemore", 0, 1).is_err(), "16th entry must be rejected");
        assert_eq!(vh.files().len(), VD_MAX_ENTRIES);
    }

    #[test]
    fn from_bytes_rejects_non_sgi_media() {
        assert!(VolumeHeader::from_bytes(&[0u8; 512]).is_err(), "no magic");
        assert!(VolumeHeader::from_bytes(&[0u8; 16]).is_err(), "short read");
    }

    #[test]
    fn build_image_places_files_and_partitions() {
        let p = unique_tmp_path("build");
        let payload = unique_tmp_path("payload");
        std::fs::write(&payload, vec![0xAAu8; 600]).unwrap();

        let spec = ImageSpec {
            total_bytes: 1024 * 1024,
            bootfile: None,
            files: vec![("cputest".into(), payload.clone())],
            partitions: vec![PartitionSpec { slot: 7, first_block: Some(64), nblks: 128, ptype: PT_RAW, content: None }],
            max_segment: 0,
        };
        let placed = build_image(&p, &spec).expect("build");
        assert_eq!(placed.len(), 1);
        assert_eq!(placed[0].nbytes, 600);
        assert_eq!(placed[0].lbn, VH_SECTORS as u32, "first file lands right after the VH region");

        let bytes = std::fs::read(&p).unwrap();
        assert_eq!(bytes.len(), 1024 * 1024);
        let vh = VolumeHeader::from_bytes(&bytes).unwrap();
        assert!(vh.csum_valid());
        assert_eq!(vh.bootfile(), "cputest", "bootfile defaults to the first file");
        assert_eq!(vh.files(), placed);

        // File contents must be readable at lbn * 512, exactly nbytes long.
        let off = placed[0].lbn as usize * SECTOR_SIZE as usize;
        assert_eq!(&bytes[off..off + 600], &vec![0xAAu8; 600][..]);

        let parts = vh.partitions();
        assert!(parts.iter().any(|e| e.slot == 7 && e.first_block == 64 && e.nblks == 128));
        assert!(parts.iter().any(|e| e.slot == 8 && e.ptype == PT_VOLHDR), "slot 8 auto-added");
        assert!(parts.iter().any(|e| e.slot == 10 && e.nblks == 2048), "slot 10 covers the disk");

        let _ = std::fs::remove_file(&p);
        let _ = std::fs::remove_file(&payload);
    }

    #[test]
    fn volume_header_partition_spans_the_voldir_payload() {
        // The 6.5.22 Installation Tools CD gives slot 8 48736 blocks to cover
        // sash/miniroot; 8 blocks is only right for a header with no files.
        let p = unique_tmp_path("vhspan");
        let big = unique_tmp_path("vhspan-file");
        std::fs::write(&big, vec![0x55u8; 100 * SECTOR_SIZE as usize + 3]).unwrap();
        let spec = ImageSpec {
            files: vec![("sash".into(), big.clone())],
            partitions: vec![PartitionSpec { slot: 7, first_block: None, nblks: 64, ptype: 5, content: None }],
            ..Default::default()
        };
        let placed = build_image(&p, &spec).expect("build");
        let vh = VolumeHeader::from_bytes(&std::fs::read(&p).unwrap()).unwrap();

        let part8 = vh.partitions().into_iter().find(|e| e.slot == 8).expect("slot 8");
        let file_end = placed[0].lbn + placed[0].nbytes.div_ceil(SECTOR_SIZE as u32);
        assert!(part8.nblks >= file_end, "slot 8 ({} blocks) must span the voldir payload (ends at {})", part8.nblks, file_end);

        // first_block: None starts the filesystem right after the header, as the CD does.
        let part7 = vh.partitions().into_iter().find(|e| e.slot == 7).expect("slot 7");
        assert_eq!(part7.first_block, part8.nblks, "partition 7 must begin where partition 8 ends");
        assert_eq!(part7.ptype, 5, "SGI ships EFS filesystems as PT_SYSV");

        let _ = std::fs::remove_file(&p);
        let _ = std::fs::remove_file(&big);
    }

    /// An MSB ELF executable: one `PT_NOTE` and one `PT_LOAD` of `filesz`
    /// bytes (each byte its offset mod 251) with `bss` bytes of zero fill,
    /// the load at file offset 0x1000 and `pad` free bytes after the program
    /// headers. A section header table with one `PROGBITS` section over the
    /// segment follows the data.
    fn test_elf(wide: bool, filesz: u64, bss: u64) -> Vec<u8> {
        let (ehsize, phentsize, shentsize) = if wide { (64usize, 56usize, 64usize) } else { (52, 32, 40) };
        let load_off = 0x1000u64;
        let vaddr = if wide { 0xa800_0000_2000_4000u64 } else { 0x8800_4000 };
        let shoff = (load_off + filesz).next_multiple_of(8);
        let mut d = vec![0u8; shoff as usize + 2 * shentsize];
        let put = |d: &mut Vec<u8>, off: usize, n: usize, v: u64| {
            for i in 0..n { d[off + i] = (v >> (8 * (n - 1 - i))) as u8; }
        };
        d[0..4].copy_from_slice(b"\x7fELF");
        d[4] = if wide { 2 } else { 1 };
        d[5] = 2;
        d[6] = 1;
        put(&mut d, 16, 2, 2); // ET_EXEC
        put(&mut d, 18, 2, 8); // EM_MIPS
        let w = if wide { 8 } else { 4 };
        let (phoff_at, shoff_at, half_at) = if wide { (0x20, 0x28, 0x36) } else { (0x1c, 0x20, 0x2a) };
        put(&mut d, phoff_at, w, ehsize as u64);
        put(&mut d, shoff_at, w, shoff);
        put(&mut d, half_at, 2, phentsize as u64);
        put(&mut d, half_at + 2, 2, 2);
        put(&mut d, half_at + 4, 2, shentsize as u64);
        put(&mut d, half_at + 6, 2, 2);
        let ph = |d: &mut Vec<u8>, i: usize, f: [u64; 8]| {
            let b = ehsize + i * phentsize;
            let [ty, flags, off, va, pa, fsz, msz, align] = f;
            if wide {
                for (o, n, v) in [(0, 4, ty), (4, 4, flags), (8, 8, off), (16, 8, va), (24, 8, pa), (32, 8, fsz), (40, 8, msz), (48, 8, align)] { put(d, b + o, n, v); }
            } else {
                for (o, v) in [(0, ty), (4, off), (8, va), (12, pa), (16, fsz), (20, msz), (24, flags), (28, align)] { put(d, b + o, 4, v); }
            }
        };
        ph(&mut d, 0, [4, 4, load_off, vaddr, vaddr, 0x10, 0x10, 4]);
        ph(&mut d, 1, [1, 7, load_off, vaddr, vaddr, filesz, filesz + bss, 0x4000]);
        for i in 0..filesz as usize { d[load_off as usize + i] = (i % 251) as u8; }
        // Section 1: PROGBITS over the segment.
        let s1 = shoff as usize + shentsize;
        put(&mut d, s1 + 4, 4, 1);
        if wide { put(&mut d, s1 + 24, 8, load_off); put(&mut d, s1 + 32, 8, filesz); }
        else { put(&mut d, s1 + 16, 4, load_off); put(&mut d, s1 + 20, 4, filesz); }
        d
    }

    /// The `PT_LOAD` headers of an MSB ELF file as (offset, vaddr, filesz, memsz).
    fn loads(d: &[u8]) -> Vec<(u64, u64, u64, u64)> {
        let wide = d[4] == 2;
        let rd = |off: usize, n: usize| d[off..off + n].iter().fold(0u64, |v, &x| (v << 8) | x as u64);
        let (phoff, es, n) = if wide { (rd(0x20, 8), rd(0x36, 2), rd(0x38, 2)) } else { (rd(0x1c, 4), rd(0x2a, 2), rd(0x2c, 2)) };
        (0..n as usize).filter_map(|i| {
            let b = phoff as usize + i * es as usize;
            if rd(b, 4) != 1 { return None; }
            Some(if wide { (rd(b + 8, 8), rd(b + 16, 8), rd(b + 32, 8), rd(b + 40, 8)) }
                 else { (rd(b + 4, 4), rd(b + 8, 4), rd(b + 16, 4), rd(b + 20, 4)) })
        }).collect()
    }

    #[test]
    fn split_load_segments_cuts_a_large_segment_into_prom_sized_pieces() {
        for wide in [true, false] {
            let filesz = 2 * PROM_SEGMENT_BYTES + 0x1234;
            let elf = test_elf(wide, filesz, 0x3000);
            let (out, what) = split_load_segments(&elf, PROM_SEGMENT_BYTES).unwrap().expect("split");
            assert_eq!(what, SegmentSplit { loads_before: 1, loads_after: 3, largest: filesz });
            let got = loads(&out);
            let (off0, va0) = (got[0].0, got[0].1);
            let mut at = 0;
            for (i, &(off, va, fsz, msz)) in got.iter().enumerate() {
                assert!(fsz <= PROM_SEGMENT_BYTES, "piece {i} carries {fsz:#x} bytes");
                assert_eq!((off, va), (off0 + at, va0 + at), "piece {i} continues the last");
                let last = i + 1 == got.len();
                assert_eq!(msz, if last { fsz + 0x3000 } else { fsz }, "only the last piece keeps the zero fill");
                at += fsz;
            }
            assert_eq!(at, filesz, "the pieces cover the segment");
            // The table had room after the ELF header: the data is untouched.
            assert_eq!(out.len(), elf.len());
            assert_eq!(&out[0x1000..], &elf[0x1000..]);
        }
    }

    #[test]
    fn split_load_segments_moves_the_table_past_the_end_when_it_has_no_room() {
        let mut elf = test_elf(true, 3 * PROM_SEGMENT_BYTES, 0);
        // Put the program headers right before the segment: no room to grow.
        let phoff = 0x1000 - 2 * 56;
        elf.copy_within(64..64 + 2 * 56, phoff);
        elf[0x20..0x28].copy_from_slice(&(phoff as u64).to_be_bytes());
        let (out, _) = split_load_segments(&elf, PROM_SEGMENT_BYTES).unwrap().expect("split");
        let new_phoff = u64::from_be_bytes(out[0x20..0x28].try_into().unwrap());
        assert!(new_phoff >= elf.len() as u64 && new_phoff % 8 == 0, "table at {new_phoff:#x}");
        assert_eq!(&out[..0x20], &elf[..0x20]);
        assert_eq!(&out[0x40..elf.len()], &elf[0x40..], "everything after the header is kept");
        assert_eq!(loads(&out).len(), 3);
    }

    #[test]
    fn split_load_segments_leaves_small_and_foreign_files_alone() {
        assert_eq!(split_load_segments(&test_elf(true, PROM_SEGMENT_BYTES, 0x100), PROM_SEGMENT_BYTES), Ok(None));
        let mut rel = test_elf(true, 4 * PROM_SEGMENT_BYTES, 0);
        rel[17] = 1; // ET_REL, like sash
        assert_eq!(split_load_segments(&rel, PROM_SEGMENT_BYTES), Ok(None));
        assert_eq!(split_load_segments(&vec![0xAAu8; 2 << 20], PROM_SEGMENT_BYTES), Ok(None));
        assert_eq!(split_load_segments(&test_elf(true, 4 * PROM_SEGMENT_BYTES, 0), 0), Ok(None));
    }

    #[test]
    fn build_image_splits_an_oversized_kernel_only_when_asked() {
        let kernel = unique_tmp_path("kernel");
        let elf = test_elf(true, PROM_SEGMENT_BYTES * 5 / 2, 0x100);
        std::fs::write(&kernel, &elf).unwrap();
        for (max, want) in [(PROM_SEGMENT_BYTES, 3), (0, 1)] {
            let p = unique_tmp_path("split");
            let spec = ImageSpec { files: vec![("vmlinux".into(), kernel.clone())], max_segment: max, ..Default::default() };
            let (placed, splits) = build_image_with_splits(&p, &spec).expect("build");
            assert_eq!(splits.len(), usize::from(max != 0));
            let bytes = std::fs::read(&p).unwrap();
            let off = placed[0].lbn as usize * SECTOR_SIZE as usize;
            let file = &bytes[off..off + placed[0].nbytes as usize];
            assert_eq!(loads(file).len(), want);
            let _ = std::fs::remove_file(&p);
        }
        let _ = std::fs::remove_file(&kernel);
    }

    #[test]
    fn build_image_rejects_an_image_too_small_for_its_contents() {
        let p = unique_tmp_path("toosmall");
        let payload = unique_tmp_path("toosmall-payload");
        std::fs::write(&payload, vec![0u8; 4096]).unwrap();
        let spec = ImageSpec {
            total_bytes: 4096,
            files: vec![("cputest".into(), payload.clone())],
            ..Default::default()
        };
        assert!(build_image(&p, &spec).is_err());
        let _ = std::fs::remove_file(&p);
        let _ = std::fs::remove_file(&payload);
    }
}
