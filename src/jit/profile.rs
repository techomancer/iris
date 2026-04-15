//! JIT profile cache: persists hot block metadata across emulator runs.
//!
//! On shutdown, saves metadata (virt_pc, phys_pc, tier, len_mips, content_hash,
//! hit_count) for all blocks above Alu tier, sorted by hit_count descending so
//! the hottest blocks replay first. On startup, the dispatch loop loads the
//! profile into a queue but does NOT compile upfront — bulk pre-compilation
//! evicts L2/D-cache lines the kernel depends on during early boot and causes
//! UTLB panics. Instead, replay drip-feeds one entry per probe once the
//! kernel has reached userspace (boot-settle detection).

use std::fs;
use std::io::{self, Read, Write, BufReader, BufWriter};
use std::path::PathBuf;

use super::cache::BlockTier;

/// One entry in the profile: a block that reached a tier worth persisting.
#[derive(Debug, Clone)]
pub struct ProfileEntry {
    /// Saved for diagnostics only — NOT used for cache insertion. Physical
    /// addresses depend on TLB state which may differ between sessions.
    pub phys_pc: u64,
    /// Virtual PC — stable across sessions, used to re-trace the block.
    pub virt_pc: u64,
    pub tier: BlockTier,
    /// Instruction count at save time. Cheap staleness check.
    pub len_mips: u32,
    /// FNV-1a 32-bit hash of the raw instruction words. Definitive staleness
    /// check: catches same-length different-code cases (DSO reused at same VA).
    pub content_hash: u32,
    /// Hit count from the previous session, used to prioritize replay order.
    pub hit_count: u32,
}

const PROFILE_MAGIC: &[u8; 4] = b"IRJP"; // IRIS JIT Profile
const PROFILE_VERSION: u8 = 2;

// On-disk entry: 8+8+1+4+4+4 = 29 bytes, padded to 32 for alignment.
const ENTRY_BYTES: usize = 32;

/// Default profile path: ~/.iris/jit-profile.bin
fn default_profile_path() -> PathBuf {
    if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home).join(".iris").join("jit-profile.bin")
    } else {
        PathBuf::from("jit-profile.bin")
    }
}

/// Get the profile path, respecting IRIS_JIT_PROFILE env var override.
pub fn profile_path() -> PathBuf {
    match std::env::var_os("IRIS_JIT_PROFILE") {
        Some(p) => PathBuf::from(p),
        None => default_profile_path(),
    }
}

/// Load profile entries from disk. Returns empty vec on any error.
/// Entries are returned in save order (sorted by hit_count descending).
pub fn load_profile() -> Vec<ProfileEntry> {
    let path = profile_path();
    let file = match fs::File::open(&path) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };
    let mut reader = BufReader::new(file);

    let mut magic = [0u8; 4];
    if reader.read_exact(&mut magic).is_err() || &magic != PROFILE_MAGIC {
        eprintln!("JIT profile: invalid magic in {:?}, ignoring", path);
        return Vec::new();
    }

    let mut version = [0u8; 1];
    if reader.read_exact(&mut version).is_err() || version[0] != PROFILE_VERSION {
        eprintln!("JIT profile: version mismatch in {:?} (found {}, need {}), ignoring",
            path, version[0], PROFILE_VERSION);
        return Vec::new();
    }

    let mut count_buf = [0u8; 4];
    if reader.read_exact(&mut count_buf).is_err() {
        return Vec::new();
    }
    let count = u32::from_le_bytes(count_buf) as usize;

    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        let mut buf = [0u8; ENTRY_BYTES];
        if reader.read_exact(&mut buf).is_err() {
            break;
        }
        let phys_pc      = u64::from_le_bytes(buf[0..8].try_into().unwrap());
        let virt_pc      = u64::from_le_bytes(buf[8..16].try_into().unwrap());
        let tier_byte    = buf[16];
        let len_mips     = u32::from_le_bytes(buf[17..21].try_into().unwrap());
        let content_hash = u32::from_le_bytes(buf[21..25].try_into().unwrap());
        let hit_count    = u32::from_le_bytes(buf[25..29].try_into().unwrap());
        // buf[29..32] is padding, ignore
        let tier = match tier_byte {
            0 => BlockTier::Alu,
            1 => BlockTier::Loads,
            2 => BlockTier::Full,
            _ => continue,
        };
        entries.push(ProfileEntry {
            phys_pc, virt_pc, tier, len_mips, content_hash, hit_count,
        });
    }

    eprintln!("JIT profile: loaded {} entries from {:?}", entries.len(), path);
    entries
}

/// Save profile entries to disk. Writes atomically via tmp file + rename to
/// avoid truncated files on interrupted writes. Sorts entries by hit_count
/// descending so the hottest blocks are first in the queue on next load.
pub fn save_profile(entries: &[ProfileEntry]) -> io::Result<()> {
    let path = profile_path();

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut sorted: Vec<&ProfileEntry> = entries.iter().collect();
    sorted.sort_by(|a, b| b.hit_count.cmp(&a.hit_count));

    let tmp_path = path.with_extension("bin.tmp");
    {
        let file = fs::File::create(&tmp_path)?;
        let mut writer = BufWriter::new(file);

        writer.write_all(PROFILE_MAGIC)?;
        writer.write_all(&[PROFILE_VERSION])?;
        writer.write_all(&(sorted.len() as u32).to_le_bytes())?;

        for entry in &sorted {
            let mut buf = [0u8; ENTRY_BYTES];
            buf[0..8].copy_from_slice(&entry.phys_pc.to_le_bytes());
            buf[8..16].copy_from_slice(&entry.virt_pc.to_le_bytes());
            buf[16] = match entry.tier {
                BlockTier::Alu => 0,
                BlockTier::Loads => 1,
                BlockTier::Full => 2,
            };
            buf[17..21].copy_from_slice(&entry.len_mips.to_le_bytes());
            buf[21..25].copy_from_slice(&entry.content_hash.to_le_bytes());
            buf[25..29].copy_from_slice(&entry.hit_count.to_le_bytes());
            writer.write_all(&buf)?;
        }

        writer.flush()?;
    }
    fs::rename(&tmp_path, &path)?;
    eprintln!("JIT profile: saved {} entries to {:?}", sorted.len(), path);
    Ok(())
}
