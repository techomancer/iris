//! Copy-on-write disk overlay for SCSI disk images.
//!
//! Protects the base disk image from writes by redirecting them to a sparse
//! overlay file. Reads check the overlay first, falling back to the base image
//! for clean sectors. Deleting the overlay file resets the disk to its original state.

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const SECTOR_SIZE: u64 = 512;

/// Clone `src` to `dst` via filesystem-level CoW (APFS clonefile, Linux
/// FICLONE) when supported; fall back to a regular byte copy otherwise. On a
/// reflink-capable filesystem this is metadata-only — sub-millisecond for any
/// size — which makes per-snapshot overlay capture essentially free.
pub(crate) fn reflink_or_copy(src: &Path, dst: &Path) -> io::Result<()> {
    let _ = std::fs::remove_file(dst);
    if try_reflink(src, dst).is_ok() {
        return Ok(());
    }
    std::fs::copy(src, dst).map(|_| ())
}

#[cfg(target_os = "macos")]
fn try_reflink(src: &Path, dst: &Path) -> io::Result<()> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    let src_c = CString::new(src.as_os_str().as_bytes())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    let dst_c = CString::new(dst.as_os_str().as_bytes())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    let rc = unsafe { libc::clonefile(src_c.as_ptr(), dst_c.as_ptr(), 0) };
    if rc == 0 { Ok(()) } else { Err(io::Error::last_os_error()) }
}

#[cfg(target_os = "linux")]
fn try_reflink(src: &Path, dst: &Path) -> io::Result<()> {
    use std::os::unix::io::AsRawFd;
    // FICLONE = _IOW(0x94, 9, int); see linux/fs.h.
    const FICLONE: libc::c_ulong = 0x40049409;
    let src_f = File::open(src)?;
    let dst_f = OpenOptions::new().write(true).create(true).truncate(true).open(dst)?;
    let rc = unsafe { libc::ioctl(dst_f.as_raw_fd(), FICLONE, src_f.as_raw_fd()) };
    if rc == 0 {
        Ok(())
    } else {
        let err = io::Error::last_os_error();
        let _ = std::fs::remove_file(dst);
        Err(err)
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn try_reflink(_src: &Path, _dst: &Path) -> io::Result<()> {
    Err(io::Error::new(io::ErrorKind::Unsupported, "reflink not supported on this OS"))
}

/// Sidecar file holding the dirty sector list. Written next to the overlay
/// (e.g. `foo.overlay.dirty`). Format: binary, a `u64` little-endian count
/// followed by `u64` sector LBAs, also LE. The count is the number of LBAs
/// the file was last rewritten with; [`crate::block_node::CowNode`] appends
/// LBAs after those as the overlay takes on new sectors, so a reader takes
/// every whole LBA up to the end of the file. An older reader stops at the
/// count and so sees what a flush recorded, as it always did. After a newer
/// iris was killed that is the disk as it was before the appends were made:
/// the torn disk this record exists to prevent. Open such a disk with a newer
/// iris first; the older one's flush would drop the appended sectors from the
/// record for good.
fn dirty_sidecar_path(overlay_path: &str) -> PathBuf {
    PathBuf::from(format!("{}.dirty", overlay_path))
}

/// The `.dirty` sidecar that persists an overlay's dirty sector set. Shared
/// with [`crate::block_node::CowNode`], which is the layered re-expression of
/// this backend.
pub(crate) fn sidecar_path_for(overlay_path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.dirty", overlay_path.display()))
}

/// Read the sector set a sidecar records. No sidecar is an empty set. A
/// sidecar that cannot be read, or that has no whole header, is an error: the
/// caller must not treat it as empty, or the next rewrite would replace a
/// record it could not read with nothing.
pub(crate) fn load_dirty_sidecar(path: &Path) -> io::Result<HashSet<u64>> {
    let mut f = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(HashSet::new()),
        Err(e) => return Err(e),
    };
    let mut count_buf = [0u8; 8];
    f.read_exact(&mut count_buf).map_err(|e| {
        io::Error::new(e.kind(), format!("{}: no whole header: {e}", path.display()))
    })?;
    let count = u64::from_le_bytes(count_buf) as usize;
    let mut set = HashSet::with_capacity(count.min(1 << 24));
    let mut buf = [0u8; 8];
    // The counted LBAs and then any appended since. A short last entry is
    // an append cut off by the process dying, and was never acknowledged;
    // that alone is the end of the record. Any other read error is not.
    loop {
        match f.read_exact(&mut buf) {
            Ok(()) => {
                set.insert(u64::from_le_bytes(buf));
            }
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }
    }
    Ok(set)
}

/// Add `lbas` to the end of a sidecar opened for appending (see
/// [`dirty_sidecar_path`] for the format). One write, so a process killed
/// after it returns has recorded every one of them: what the kernel has
/// accepted outlives the process. Surviving a host crash still takes a flush.
pub(crate) fn append_dirty_sidecar(f: &mut File, lbas: &[u64]) -> io::Result<()> {
    let bytes: Vec<u8> = lbas.iter().flat_map(|l| l.to_le_bytes()).collect();
    f.write_all(&bytes)
}

pub(crate) fn save_dirty_sidecar(path: &Path, dirty: &HashSet<u64>) -> io::Result<()> {
    // Write atomically: write to a temp file then rename.
    let tmp = path.with_extension("dirty.tmp");
    {
        let mut f = File::create(&tmp)?;
        let mut bytes = Vec::with_capacity(8 * (dirty.len() + 1));
        bytes.extend_from_slice(&(dirty.len() as u64).to_le_bytes());
        for &s in dirty {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        f.write_all(&bytes)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    // The rename is not on the host's disk until its directory is.
    crate::chd_disk::fsync_dir(path.parent())
}

pub struct CowDisk {
    base: File,
    overlay: File,
    dirty: HashSet<u64>,
    base_size: u64,
    overlay_path: String,
}

impl CowDisk {
    /// Open a COW disk with the given base image (read-only) and overlay file (read-write).
    /// If the overlay file exists, its dirty sectors are reconstructed from its sparse extent.
    /// If it doesn't exist, a new empty overlay is created.
    pub fn new(base_path: &str, overlay_path: &str) -> io::Result<Self> {
        let base = File::open(base_path)?;
        let base_size = base.metadata()?.len();

        let overlay = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(overlay_path)?;

        // Recover the dirty set from our sidecar file (written on flush /
        // shutdown by previous runs). If the sidecar is missing we start
        // empty — any prior writes in the overlay file are effectively
        // invisible until the sidecar gets written. This is deliberate:
        // "dirty" means "the host finished writing this sector," not "the
        // file has some bytes here" (sparse allocation can contain partial
        // writes from an interrupted run, which can't be trusted).
        let sidecar = dirty_sidecar_path(overlay_path);
        let dirty = load_dirty_sidecar(&sidecar)?;

        eprintln!("iris: COW overlay active (base: {}, overlay: {}, dirty sectors: {})",
                  base_path, overlay_path, dirty.len());
        if dirty.is_empty() && std::fs::metadata(overlay_path).map(|m| m.len()).unwrap_or(0) > 0 {
            eprintln!("iris: note: overlay file has data but no .dirty sidecar — prior writes are not in use");
        }
        eprintln!("iris: to reset disk to clean state, delete {} and {}",
                  overlay_path, sidecar.display());

        Ok(Self {
            base,
            overlay,
            dirty,
            base_size,
            overlay_path: overlay_path.to_string(),
        })
    }

    /// Read `count` sectors starting at `lba`.
    /// Dirty sectors are read from the overlay, clean sectors from the base.
    pub fn read_sectors(&mut self, lba: u64, count: usize) -> io::Result<Vec<u8>> {
        let total = count * SECTOR_SIZE as usize;
        let mut data = vec![0u8; total];

        // Batch consecutive sectors from the same source to minimize seeks.
        let mut pos = 0usize;
        let mut sector = lba;
        while pos < total {
            // Determine run length from the same source.
            let is_dirty = self.dirty.contains(&sector);
            let mut run = 1usize;
            while pos + run * SECTOR_SIZE as usize <= total {
                let next = sector + run as u64;
                if self.dirty.contains(&next) != is_dirty {
                    break;
                }
                run += 1;
            }
            // Don't overshoot.
            let run_sectors = run.min((total - pos) / SECTOR_SIZE as usize);
            let run_bytes = run_sectors * SECTOR_SIZE as usize;

            let file = if is_dirty { &mut self.overlay } else { &mut self.base };
            file.seek(SeekFrom::Start(sector * SECTOR_SIZE))?;
            file.read_exact(&mut data[pos..pos + run_bytes])?;

            pos += run_bytes;
            sector += run_sectors as u64;
        }

        Ok(data)
    }

    /// Write sectors starting at `lba`. Data length must be a multiple of 512.
    /// Writes go to the overlay file only; the base image is never modified.
    pub fn write_sectors(&mut self, lba: u64, data: &[u8]) -> io::Result<()> {
        debug_assert!(data.len() % SECTOR_SIZE as usize == 0);
        let count = data.len() / SECTOR_SIZE as usize;

        self.overlay.seek(SeekFrom::Start(lba * SECTOR_SIZE))?;
        self.overlay.write_all(data)?;

        for i in 0..count as u64 {
            self.dirty.insert(lba + i);
        }

        Ok(())
    }

    /// Base image size in bytes.
    pub fn size(&self) -> u64 {
        self.base_size
    }

    /// Merge all dirty overlay sectors into the base image, then truncate overlay.
    pub fn commit(&mut self) -> io::Result<usize> {
        // Reopen base as read-write for the commit.
        // (We can't just change the mode of self.base, so we open a second handle.)
        let base_path = {
            // Get the path from /proc/self/fd on Linux, or just require it as a param.
            // For simplicity, we'll do the commit through the overlay path convention:
            // base path = overlay path without the ".overlay" suffix.
            if self.overlay_path.ends_with(".overlay") {
                self.overlay_path[..self.overlay_path.len() - 8].to_string()
            } else {
                return Err(io::Error::new(io::ErrorKind::Other,
                    "cannot determine base path from overlay path"));
            }
        };

        let mut base_rw = OpenOptions::new().read(true).write(true).open(&base_path)?;
        let mut buf = vec![0u8; SECTOR_SIZE as usize];
        let mut committed = 0usize;

        for &lba in &self.dirty {
            self.overlay.seek(SeekFrom::Start(lba * SECTOR_SIZE))?;
            self.overlay.read_exact(&mut buf)?;
            base_rw.seek(SeekFrom::Start(lba * SECTOR_SIZE))?;
            base_rw.write_all(&buf)?;
            committed += 1;
        }

        base_rw.sync_all()?;
        self.dirty.clear();
        self.overlay.set_len(0)?;

        // Reopen base read-only to pick up committed data.
        self.base = File::open(&base_path)?;

        eprintln!("iris: COW committed {} sectors to {}", committed, base_path);
        Ok(committed)
    }

    /// Delete the overlay file and create a fresh empty one (for state load).
    pub fn reset_overlay(&mut self) -> io::Result<()> {
        self.dirty.clear();
        self.overlay.set_len(0)?;
        self.overlay.seek(SeekFrom::Start(0))?;
        // Also clear the sidecar so we don't "remember" sectors that no
        // longer exist after the truncation.
        let _ = std::fs::remove_file(dirty_sidecar_path(&self.overlay_path));
        Ok(())
    }

    /// Flush the overlay file's data and persist the dirty sector set to
    /// the sidecar. Call this on clean shutdown or before snapshot save so
    /// a subsequent run can read back what we wrote.
    pub fn flush(&mut self) -> io::Result<()> {
        self.overlay.sync_all()?;
        save_dirty_sidecar(&dirty_sidecar_path(&self.overlay_path), &self.dirty)
    }

    /// Number of dirty sectors in the overlay.
    pub fn dirty_count(&self) -> usize {
        self.dirty.len()
    }

    /// Copy the current overlay file to `dest` and return the dirty sector
    /// list (sorted, ascending). Used by snapshot save so the entire disk
    /// state — base + overlay — is captured consistently with RAM.
    pub fn export_overlay(&mut self, dest: &Path) -> io::Result<Vec<u64>> {
        self.overlay.sync_all()?;
        reflink_or_copy(Path::new(&self.overlay_path), dest)?;
        let mut dirty: Vec<u64> = self.dirty.iter().copied().collect();
        dirty.sort_unstable();
        Ok(dirty)
    }

    /// Replace the overlay contents with `source` and adopt `dirty` as the
    /// dirty sector set. Used by snapshot load. If `source` doesn't exist
    /// the overlay is truncated instead (matches `reset_overlay` behavior —
    /// handles old snapshots without overlay data).
    pub fn import_overlay(&mut self, source: &Path, dirty: Vec<u64>) -> io::Result<()> {
        if source.exists() {
            reflink_or_copy(source, Path::new(&self.overlay_path))?;
        } else {
            // Clear the overlay: nothing saved for this device.
            std::fs::File::create(&self.overlay_path)?;
        }
        // Reopen the file handle — the previous File object points at the
        // old inode (which std::fs::copy replaced on some platforms).
        self.overlay = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&self.overlay_path)?;
        self.dirty = dirty.into_iter().collect();
        Ok(())
    }
}

impl Drop for CowDisk {
    fn drop(&mut self) {
        if let Err(e) = self.flush() {
            eprintln!("iris: COW flush on drop failed for {}: {} (writes may be lost)",
                      self.overlay_path, e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    fn unique_tmp(tag: &str, ext: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("iris-cow-{}-{}.{}", tag, nanos, ext))
    }

    #[test]
    fn reflink_or_copy_preserves_bytes() {
        let src = unique_tmp("reflink-src", "bin");
        let dst = unique_tmp("reflink-dst", "bin");
        let payload: Vec<u8> = (0u8..=255).cycle().take(64 * 1024 + 17).collect();
        {
            let mut f = File::create(&src).unwrap();
            f.write_all(&payload).unwrap();
            f.sync_all().unwrap();
        }
        reflink_or_copy(&src, &dst).expect("reflink_or_copy");
        let read_back = std::fs::read(&dst).unwrap();
        assert_eq!(read_back, payload);
        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&dst);
    }

    #[test]
    fn reflink_or_copy_overwrites_existing_dst() {
        let src = unique_tmp("reflink-src2", "bin");
        let dst = unique_tmp("reflink-dst2", "bin");
        std::fs::write(&src, b"new content").unwrap();
        std::fs::write(&dst, b"old content that is longer than the new one").unwrap();
        reflink_or_copy(&src, &dst).expect("overwrite");
        assert_eq!(std::fs::read(&dst).unwrap(), b"new content");
        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&dst);
    }
}
