//! The files beside a copy-on-write overlay (`CowNode` in `block_node`):
//! the dirty-sector record, and copying an overlay out and back for
//! snapshots.

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const SECTOR_SIZE: u64 = 512;

/// Clone `src` to `dst` via filesystem-level CoW (APFS clonefile, Linux
/// FICLONE) when supported; otherwise copy just the `dirty` sectors (see
/// [`copy_dirty_sectors`]). On a reflink-capable filesystem this is
/// metadata-only — sub-millisecond for any size — which makes per-snapshot
/// overlay capture essentially free.
pub(crate) fn reflink_or_copy(src: &Path, dst: &Path, dirty: &[u64]) -> io::Result<()> {
    let _ = std::fs::remove_file(dst);
    if try_reflink(src, dst).is_ok() {
        return Ok(());
    }
    copy_dirty_sectors(src, dst, dirty)
}

/// Copy the overlay `src` to `dst`: a file of the same length that holds the
/// `dirty` sectors (ascending) and nothing else, leaving the rest a hole.
/// `std::fs::copy` writes a sparse file's holes out as zeros whenever the
/// kernel cannot copy in-filesystem (between filesystems, on NFS), so an 8 GB
/// overlay holding 100 MB became an 8 GB snapshot. This reads and writes in
/// proportion to what the guest wrote, on any filesystem and OS.
fn copy_dirty_sectors(src: &Path, dst: &Path, dirty: &[u64]) -> io::Result<()> {
    let mut from = File::open(src)?;
    let len = from.metadata()?.len();
    let mut to = OpenOptions::new().write(true).create(true).truncate(true).open(dst)?;
    mark_sparse(&to);
    to.set_len(len)?;
    let mut buf = vec![0u8; 1 << 20];
    let mut i = 0;
    while i < dirty.len() {
        // A run of adjacent sectors is one seek and a few large transfers.
        let first = dirty[i];
        let mut last = first;
        i += 1;
        while i < dirty.len() && last.checked_add(1) == Some(dirty[i]) {
            last = dirty[i];
            i += 1;
        }
        // A sector number comes from a saved file, so it may be nonsense;
        // one that overflows or lies past the end of the source holds nothing
        // to copy.
        let Some(start) = first.checked_mul(SECTOR_SIZE) else {
            continue;
        };
        let end = last.saturating_add(1).saturating_mul(SECTOR_SIZE).min(len);
        let mut pos = start;
        if pos >= end {
            continue;
        }
        from.seek(SeekFrom::Start(pos))?;
        to.seek(SeekFrom::Start(pos))?;
        while pos < end {
            let want = ((end - pos) as usize).min(buf.len());
            from.read_exact(&mut buf[..want])?;
            to.write_all(&buf[..want])?;
            pos += want as u64;
        }
    }
    Ok(())
}

/// NTFS zero-fills the gap when a write lands past the end of a file that is
/// not marked sparse. Other filesystems need no mark; a failure here (FAT,
/// some network shares) only means the holes get written as zeros.
#[cfg(windows)]
fn mark_sparse(f: &File) {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::System::{Ioctl::FSCTL_SET_SPARSE, IO::DeviceIoControl};
    let mut returned = 0u32;
    // SAFETY: the handle is open for as long as `f` is borrowed. FSCTL_SET_SPARSE
    // takes no input or output buffer, so null with length 0 is correct, and the
    // handle was opened without FILE_FLAG_OVERLAPPED, so a null OVERLAPPED is
    // valid and the call completes before it returns.
    let ok = unsafe {
        DeviceIoControl(
            f.as_raw_handle() as _,
            FSCTL_SET_SPARSE,
            std::ptr::null(),
            0,
            std::ptr::null_mut(),
            0,
            &mut returned,
            std::ptr::null_mut(),
        )
    };
    if ok == 0 {
        eprintln!(
            "iris: cannot mark the snapshot overlay sparse ({}); it will be written at full size",
            io::Error::last_os_error()
        );
    }
}

#[cfg(not(windows))]
fn mark_sparse(_f: &File) {}

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

/// The `.dirty` sidecar that persists an overlay's dirty sector set. Written next to the overlay
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
/// [`sidecar_path_for`] for the format). One write, so a process killed
/// after it returns has recorded every one of them: what the kernel has
/// accepted outlives the process. Surviving a host crash still takes a flush.
pub(crate) fn append_dirty_sidecar(f: &mut File, lbas: &[u64]) -> io::Result<()> {
    let bytes: Vec<u8> = lbas.iter().flat_map(|l| l.to_le_bytes()).collect();
    f.write_all(&bytes)
}

pub(crate) fn save_dirty_sidecar(path: &Path, dirty: &HashSet<u64>) -> io::Result<()> {
    let ctx = |what: &str, p: &Path| {
        let what = what.to_string();
        let p = p.display().to_string();
        move |e: io::Error| io::Error::new(e.kind(), format!("{what} {p}: {e}"))
    };
    // Write atomically: write to a temp file then rename.
    let tmp = path.with_extension("dirty.tmp");
    {
        let mut f = File::create(&tmp).map_err(ctx("creating", &tmp))?;
        let mut bytes = Vec::with_capacity(8 * (dirty.len() + 1));
        bytes.extend_from_slice(&(dirty.len() as u64).to_le_bytes());
        for &s in dirty {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        f.write_all(&bytes).map_err(ctx("writing", &tmp))?;
        f.sync_all().map_err(ctx("syncing", &tmp))?;
    }
    std::fs::rename(&tmp, path).map_err(ctx("replacing", path))?;
    // The rename is not on the host's disk until its directory is.
    crate::chd_disk::fsync_dir(path.parent()).map_err(ctx("syncing the directory of", path))
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

    /// The fallback copies the dirty sectors and nothing else: a 64 MiB
    /// overlay with three runs of written sectors copies to the same bytes,
    /// and a sector the dirty list leaves out reads as zeros even though the
    /// source holds junk there. That junk check holds on every filesystem; a
    /// whole-file copy carries the junk over. The allocation check below needs
    /// a temporary directory that supports sparse files.
    #[test]
    fn a_copied_overlay_stays_sparse() {
        let (src, dst) = (unique_tmp("sparse-src", "bin"), unique_tmp("sparse-dst", "bin"));
        let runs: [(u64, u64); 3] = [(0, 8), (80_000, 1), (131_071, 1)];
        let mut dirty = Vec::new();
        {
            let mut f = File::create(&src).unwrap();
            f.set_len(64 << 20).unwrap();
            for (n, &(lba, count)) in runs.iter().enumerate() {
                f.seek(SeekFrom::Start(lba * SECTOR_SIZE)).unwrap();
                f.write_all(&vec![0xa0 + n as u8; (count * SECTOR_SIZE) as usize]).unwrap();
                dirty.extend(lba..lba + count);
            }
            // Written by the guest, then discarded: the bytes stay in the file
            // but the sector is not in the dirty list.
            f.seek(SeekFrom::Start(1000 * SECTOR_SIZE)).unwrap();
            f.write_all(&[0xee; SECTOR_SIZE as usize]).unwrap();
        }
        copy_dirty_sectors(&src, &dst, &dirty).unwrap();
        assert_eq!(std::fs::metadata(&dst).unwrap().len(), 64 << 20);
        let mut copy = File::open(&dst).unwrap();
        for (n, &(lba, count)) in runs.iter().enumerate() {
            let mut got = vec![0u8; (count * SECTOR_SIZE) as usize];
            copy.seek(SeekFrom::Start(lba * SECTOR_SIZE)).unwrap();
            copy.read_exact(&mut got).unwrap();
            assert!(got.iter().all(|&b| b == 0xa0 + n as u8), "run {n} differs");
        }
        let mut hole = vec![0xffu8; 4096];
        copy.seek(SeekFrom::Start(20 << 20)).unwrap();
        copy.read_exact(&mut hole).unwrap();
        assert!(hole.iter().all(|&b| b == 0), "a hole reads as zeros");
        let mut skipped = vec![0xffu8; SECTOR_SIZE as usize];
        copy.seek(SeekFrom::Start(1000 * SECTOR_SIZE)).unwrap();
        copy.read_exact(&mut skipped).unwrap();
        assert!(skipped.iter().all(|&b| b == 0), "a sector outside the dirty list is not copied");
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let used = std::fs::metadata(&dst).unwrap().blocks() * 512;
            assert!(used <= 1 << 20, "{used} bytes allocated for 10 written sectors");
        }
        // An overlay exported before any write holds no data at all.
        copy_dirty_sectors(&src, &dst, &[]).unwrap();
        assert_eq!(std::fs::metadata(&dst).unwrap().len(), 64 << 20);
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            assert_eq!(std::fs::metadata(&dst).unwrap().blocks(), 0);
        }
        for p in [&src, &dst] {
            let _ = std::fs::remove_file(p);
        }
    }

    /// Sector numbers past the end of the source (a truncated save, a
    /// `cow.toml` from a differently sized disk) or large enough to overflow
    /// are skipped; the copy keeps the source's length and the in-range
    /// sectors, and the run that straddles the end is cut at the end.
    #[test]
    fn dirty_sectors_beyond_the_source_are_clamped() {
        let (src, dst) = (unique_tmp("clamp-src", "bin"), unique_tmp("clamp-dst", "bin"));
        std::fs::write(&src, vec![0x5au8; 3 * SECTOR_SIZE as usize]).unwrap();
        // Sectors 2 and 3 are one run; sector 3 lies past the end. 1<<60
        // overflows when multiplied by the sector size, u64::MAX when
        // incremented.
        copy_dirty_sectors(&src, &dst, &[2, 3, 1 << 60, u64::MAX]).unwrap();
        let got = std::fs::read(&dst).unwrap();
        assert_eq!(got.len(), 3 * SECTOR_SIZE as usize);
        assert!(got[..2 * SECTOR_SIZE as usize].iter().all(|&b| b == 0));
        assert!(got[2 * SECTOR_SIZE as usize..].iter().all(|&b| b == 0x5a));
        for p in [&src, &dst] {
            let _ = std::fs::remove_file(p);
        }
    }

    /// The sidecar is created, then replaced while an append handle to the old
    /// one is still open, as `CowNode` does on every flush and reopen. On
    /// Windows the directory sync after the rename used to fail with access
    /// denied, which reached the guest as a write error on its root disk.
    #[test]
    fn the_dirty_sidecar_can_be_replaced_repeatedly() {
        let path = unique_tmp("sidecar", "dirty");
        let mut dirty: HashSet<u64> = [1, 2, 3].into_iter().collect();
        save_dirty_sidecar(&path, &dirty).expect("create");
        let mut record = std::fs::OpenOptions::new().append(true).open(&path).unwrap();
        append_dirty_sidecar(&mut record, &[4]).unwrap();
        dirty.insert(4);
        for extra in [10u64, 11] {
            dirty.insert(extra);
            save_dirty_sidecar(&path, &dirty).expect("replace");
            record = std::fs::OpenOptions::new().append(true).open(&path).unwrap();
        }
        append_dirty_sidecar(&mut record, &[12]).unwrap();
        dirty.insert(12);
        drop(record);
        assert_eq!(load_dirty_sidecar(&path).unwrap(), dirty);
        let _ = std::fs::remove_file(&path);
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
        reflink_or_copy(&src, &dst, &(0..=128).collect::<Vec<u64>>()).expect("reflink_or_copy");
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
        reflink_or_copy(&src, &dst, &[0]).expect("overwrite");
        assert_eq!(std::fs::read(&dst).unwrap(), b"new content");
        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&dst);
    }
}
