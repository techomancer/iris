//! REX3 JIT profile: persists hot DrawMode key pairs across emulator runs.
//!
//! On shutdown, saves all (DrawMode0, DrawMode1) pairs that were compiled.
//! On startup, loads the profile and eagerly queues those pairs for compilation.

use std::fs;
use std::io::{self, Read, Write, BufReader, BufWriter};
use std::path::PathBuf;

const PROFILE_MAGIC: &[u8; 4] = b"IRXP"; // IRIS REX3 JIT Profile
const PROFILE_VERSION: u8 = 1;

fn default_profile_path() -> PathBuf {
    if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home).join(".iris").join("rex-jit-profile.bin")
    } else {
        PathBuf::from("rex-jit-profile.bin")
    }
}

pub fn profile_path() -> PathBuf {
    match std::env::var_os("IRIS_REX_JIT_PROFILE") {
        Some(p) => PathBuf::from(p),
        None => default_profile_path(),
    }
}

/// Load (dm0, dm1) pairs from disk. Returns empty vec on any error.
pub fn load_profile() -> Vec<(u32, u32)> {
    let path = profile_path();
    let file = match fs::File::open(&path) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };
    let mut reader = BufReader::new(file);

    let mut magic = [0u8; 4];
    if reader.read_exact(&mut magic).is_err() || &magic != PROFILE_MAGIC {
        eprintln!("REX JIT profile: invalid magic in {:?}, ignoring", path);
        return Vec::new();
    }

    let mut version = [0u8; 1];
    if reader.read_exact(&mut version).is_err() || version[0] != PROFILE_VERSION {
        eprintln!("REX JIT profile: version mismatch in {:?}, ignoring", path);
        return Vec::new();
    }

    let mut count_buf = [0u8; 4];
    if reader.read_exact(&mut count_buf).is_err() {
        return Vec::new();
    }
    let count = u32::from_le_bytes(count_buf) as usize;

    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        let mut buf = [0u8; 8];
        if reader.read_exact(&mut buf).is_err() {
            break;
        }
        let dm0 = u32::from_le_bytes(buf[0..4].try_into().unwrap());
        let dm1 = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        entries.push((dm0, dm1));
    }

    eprintln!("REX JIT profile: loaded {} entries from {:?}", entries.len(), path);
    entries
}

/// Save (dm0, dm1) pairs to disk.
pub fn save_profile(entries: &[(u32, u32)]) -> io::Result<()> {
    let path = profile_path();

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let file = fs::File::create(&path)?;
    let mut writer = BufWriter::new(file);

    writer.write_all(PROFILE_MAGIC)?;
    writer.write_all(&[PROFILE_VERSION])?;
    writer.write_all(&(entries.len() as u32).to_le_bytes())?;

    for (dm0, dm1) in entries {
        writer.write_all(&dm0.to_le_bytes())?;
        writer.write_all(&dm1.to_le_bytes())?;
    }

    writer.flush()?;
    eprintln!("REX JIT profile: saved {} entries to {:?}", entries.len(), path);
    Ok(())
}
