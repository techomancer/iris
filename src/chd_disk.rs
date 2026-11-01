//! CHD-backed disk implementations for the SCSI subsystem.
//!
//! Two flavors:
//!   * [`ChdHd`] — hard-disk CHD as a writable block device. Uncompressed CHDs
//!     are written in place; compressed CHDs get an uncompressed `.diff.chd`
//!     sidecar so the parent stays untouched (MAME's strategy).
//!   * [`ChdCd`] — single-track MODE1 CD CHD exposed as a 2048-byte/sector
//!     read-only stream via libchdman-rs's `CdCookedReader`.

use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use libchdman_rs::cd::CdCookedReader;
use libchdman_rs::hd::HdImage;
use libchdman_rs::Chd;

fn map_err<E: std::fmt::Debug>(e: E) -> io::Error {
    io::Error::new(io::ErrorKind::Other, format!("{:?}", e))
}

/// Writable hard-disk CHD backend.
pub struct ChdHd {
    img: HdImage,
    sector_size: u32,
    total_bytes: u64,
    /// The base CHD path (the file the user configured).
    base_path: PathBuf,
    /// The `.diff.chd` sidecar, when writes go to a diff (compressed base, or a
    /// COW-style overlay). `None` when writing the base in place (uncompressed).
    diff_path: Option<PathBuf>,
    /// Whether the diff holds changes worth folding back into the base on a
    /// clean shutdown — either we wrote this session, or we reattached a diff
    /// that already existed (carrying changes from a previous session).
    dirty: bool,
    /// Copy-on-write requested for this disk (the per-disk `overlay`/COW flag).
    /// When set we ALWAYS overlay (even an uncompressed base gets a diff, so the
    /// base is never written in-session) and we NEVER auto-fold on exit — the
    /// user commits or rolls back deliberately via `cow commit` / `cow reset`.
    cow: bool,
}

// The underlying MAME chd_file holds a raw pointer (`*mut ChdFile`), making it
// !Send by default. We only ever own these from the SCSI worker thread (the
// backend is moved in once and never shared), so transferring ownership across
// threads is safe — we just don't share refs (no Sync).
unsafe impl Send for ChdHd {}
unsafe impl Send for ChdCd {}

impl ChdHd {
    pub fn open(path: &str, cow: bool) -> io::Result<Self> {
        let p = Path::new(path);
        let diff = diff_path_for(p);

        // Cases:
        //  - a diff already exists → reattach to it; it carries changes from a
        //    previous session, so mark dirty so a (non-COW) clean exit folds it.
        //  - COW on → always overlay, even an uncompressed base, so the base is
        //    never written in-session (protect + rollback).
        //  - no diff, base opens writable in place (uncompressed, COW off) → no diff.
        //  - no diff, base won't open writable (compressed) → create a fresh diff.
        let (img, diff_path, dirty) = if diff.exists() {
            (HdImage::reopen_diff(p, &diff).map_err(map_err)?, Some(diff), true)
        } else if cow {
            (HdImage::open_with_diff(p, &diff).map_err(map_err)?, Some(diff), false)
        } else {
            match HdImage::open(p) {
                Ok(img) => (img, None, false),
                Err(_) => (HdImage::open_with_diff(p, &diff).map_err(map_err)?, Some(diff), false),
            }
        };

        let sector_size = img.sector_size();
        let total_bytes = img.sector_count() * u64::from(sector_size);
        Ok(Self { img, sector_size, total_bytes, base_path: p.to_path_buf(), diff_path, dirty, cow })
    }

    /// `(base, diff)` paths if a clean exit should **auto-fold** this disk's diff
    /// back into the base — i.e. it has diff-borne changes AND COW is off (COW on
    /// means "keep separate"; commit/rollback are then manual). `None` otherwise.
    pub fn pending_sync(&self) -> Option<(PathBuf, PathBuf)> {
        if self.cow {
            return None; // keep changes separate; never auto-fold
        }
        self.overlay_paths().filter(|_| self.dirty)
    }

    /// The base CHD this disk was opened from.
    pub fn base_path(&self) -> &Path {
        &self.base_path
    }

    /// Whether this disk is in copy-on-write mode (the per-disk COW flag).
    pub fn is_cow(&self) -> bool {
        self.cow
    }

    /// `(base, diff)` when writes are landing in a `.diff.chd` overlay (regardless
    /// of the COW flag — a compressed base always overlays). Used by commit/reset.
    pub fn overlay_paths(&self) -> Option<(PathBuf, PathBuf)> {
        self.diff_path.as_ref().map(|d| (self.base_path.clone(), d.clone()))
    }

    /// Whether the overlay holds uncommitted changes.
    pub fn diff_dirty(&self) -> bool {
        self.dirty
    }

    pub fn size(&self) -> u64 {
        self.total_bytes
    }

    /// The CHD's sector size in bytes (512 for a hard-disk image).
    pub fn sector_size(&self) -> u32 {
        self.sector_size
    }

    pub fn read_blocks(&mut self, lba: u64, count: usize, block_size: u64) -> io::Result<Vec<u8>> {
        let ss = u64::from(self.sector_size);
        if block_size != ss {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("CHD HD sector size {} != requested block size {}", ss, block_size),
            ));
        }
        let mut buf = vec![0u8; count * ss as usize];
        for i in 0..count {
            let off = i * ss as usize;
            self.img
                .read_sector(lba + i as u64, &mut buf[off..off + ss as usize])
                .map_err(map_err)?;
        }
        Ok(buf)
    }

    pub fn write_sectors(&mut self, lba: u64, data: &[u8]) -> io::Result<()> {
        let ss = self.sector_size as usize;
        if !data.len().is_multiple_of(ss) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("CHD HD write length {} not a multiple of sector {}", data.len(), ss),
            ));
        }
        let count = data.len() / ss;
        for i in 0..count {
            let off = i * ss;
            self.img
                .write_sector(lba + i as u64, &data[off..off + ss])
                .map_err(map_err)?;
        }
        // Writing to a diff means the sidecar now diverges from the base, so a
        // clean shutdown should fold it back. (No-op for an in-place base.)
        if self.diff_path.is_some() {
            self.dirty = true;
        }
        Ok(())
    }
}

/// A sequential `Read` over the merged (parent + diff) sectors of an [`HdImage`],
/// used to feed [`flatten_diff`]'s rebuild. Reads one sector at a time.
///
/// The image sits in a slot shared with `flatten_diff` rather than being owned
/// here: libchdman-rs's `ChdCompressor::new` hands the reader to C++ with
/// `Box::into_raw` and never frees it, so anything the reader owns stays alive
/// — base and diff open — after `create_from_reader` returns. Windows will not
/// rename over an open file ("Access is denied"), so `flatten_diff` empties the
/// slot itself to close both before the rename.
struct MergedReader {
    img: Arc<Mutex<Option<HdImage>>>,
    sector_size: usize,
    sector_count: u64,
    next_lba: u64,
    buf: Vec<u8>,
    pos: usize, // bytes consumed from `buf`
    len: usize, // valid bytes in `buf`
}

impl MergedReader {
    fn new(img: Arc<Mutex<Option<HdImage>>>, sector_size: usize, sector_count: u64) -> Self {
        Self { img, sector_size, sector_count, next_lba: 0, buf: vec![0u8; sector_size], pos: 0, len: 0 }
    }
}

impl Read for MergedReader {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if self.pos >= self.len {
            if self.next_lba >= self.sector_count {
                return Ok(0); // EOF — every sector streamed
            }
            let mut img = self.img.lock().unwrap();
            let img = img.as_mut().ok_or_else(|| io::Error::other("CHD image closed mid-read"))?;
            img.read_sector(self.next_lba, &mut self.buf).map_err(map_err)?;
            self.next_lba += 1;
            self.pos = 0;
            self.len = self.sector_size;
        }
        let n = (self.len - self.pos).min(out.len());
        out[..n].copy_from_slice(&self.buf[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

/// Fold a `.diff.chd` back into its base CHD: rebuild the base from the merged
/// (parent + diff) view, preserving the base's codecs / geometry / hunk+unit
/// sizes (so a compressed base stays compressed), via a temp file + atomic
/// rename, then delete the diff.
///
/// Safety: the base is only ever replaced by an atomic rename of a fully-written,
/// fsynced temp file, and the diff is deleted only after that rename succeeds. On
/// any error or cancellation the base and diff are left exactly as they were, so
/// the next launch simply reattaches the diff — nothing is lost.
///
/// `progress(fraction)` receives 0.0..=1.0; `cancel()` aborts cleanly. The caller
/// MUST have dropped any open [`ChdHd`] for this base first (so the files are
/// closed) before calling this.
pub fn flatten_diff(
    base: &Path,
    diff: &Path,
    progress: &mut dyn FnMut(f32),
    cancel: &dyn Fn() -> bool,
) -> io::Result<()> {
    use libchdman_rs::hd::{create_from_reader, read_geometry, HdCreateOptions};
    use libchdman_rs::{Chd, CompressionProgress};

    let base_str = base.to_str().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "non-UTF-8 CHD path"))?;

    // Read the base's structure so the rebuilt CHD matches it byte-for-byte in
    // codecs/geometry (compressed stays compressed). Scope the handle so it's
    // closed before we rename over the base.
    let (codecs, hunk_bytes, unit_bytes, logical, geom) = {
        let bchd = Chd::open(base_str, false, None).map_err(map_err)?;
        let info = bchd.info().map_err(map_err)?;
        if !info.is_hd {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "not a hard-disk CHD"));
        }
        (info.codecs, info.hunk_bytes, info.unit_bytes, info.logical_bytes, read_geometry(&bchd).ok())
    };

    // Merged view: parent (base) with the diff applied. Its sectors are the
    // contents we rebuild the base from.
    let merged = HdImage::reopen_diff(base, diff).map_err(map_err)?;
    let sector_size = merged.sector_size() as usize;
    let sector_count = merged.sector_count();

    // Rebuild into a temp file next to the base (same filesystem → the rename is
    // atomic). The reader is never dropped (see MergedReader), so the merged
    // image goes in a shared slot that is emptied below, closing the base+diff
    // handles before the rename.
    let merged = Arc::new(Mutex::new(Some(merged)));
    let tmp = temp_sync_path_for(base);
    let opts = HdCreateOptions {
        logical_size: logical,
        hunk_size: hunk_bytes,
        unit_size: unit_bytes,
        codecs,
        geometry: geom,
        ident: None,
    };
    let reader = MergedReader::new(Arc::clone(&merged), sector_size, sector_count);
    let total = logical.max(1);
    let mut cb = |cp: CompressionProgress| {
        progress((cp.bytes_done as f64 / total as f64).min(1.0) as f32);
    };
    let created = create_from_reader(reader, &tmp, opts, &mut cb, cancel);
    drop(merged.lock().unwrap().take());
    if let Err(e) = created {
        let _ = std::fs::remove_file(&tmp); // base + diff untouched
        return Err(map_err(e));
    }

    // Durably replace the base, then drop the diff. The diff is removed only
    // after the rename, so an interruption anywhere above leaves base+diff intact.
    fsync_path(&tmp)?;
    std::fs::rename(&tmp, base)?;
    let _ = fsync_dir(base.parent());
    let _ = std::fs::remove_file(diff);
    progress(1.0);
    Ok(())
}

/// Temp path for the rebuilt CHD, alongside the base so the rename is atomic.
fn temp_sync_path_for(base: &Path) -> PathBuf {
    let mut s = base.as_os_str().to_owned();
    s.push(".synctmp.chd");
    PathBuf::from(s)
}

fn fsync_path(p: &Path) -> io::Result<()> {
    std::fs::OpenOptions::new().read(true).write(true).open(p)?.sync_all()
}

pub(crate) fn fsync_dir(dir: Option<&Path>) -> io::Result<()> {
    if let Some(d) = dir {
        // A bare file name has the parent "", which means the current directory.
        let d = if d.as_os_str().is_empty() { Path::new(".") } else { d };
        std::fs::File::open(d)?.sync_all()?;
    }
    Ok(())
}

/// Read-only CD CHD backend.
pub struct ChdCd {
    reader: CdCookedReader,
    total_bytes: u64,
}

impl ChdCd {
    pub fn open(path: &str) -> io::Result<Self> {
        let chd = Chd::open(path, false, None).map_err(map_err)?;
        let reader = CdCookedReader::open(chd).map_err(map_err)?;
        let total_bytes = reader.len();
        Ok(Self { reader, total_bytes })
    }

    pub fn size(&self) -> u64 {
        self.total_bytes
    }

    pub fn read_blocks(&mut self, lba: u64, count: usize, block_size: u64) -> io::Result<Vec<u8>> {
        let byte_offset = lba * block_size;
        let byte_count = (count as u64) * block_size;
        let mut buf = vec![0u8; byte_count as usize];
        self.read_at(byte_offset, &mut buf)?;
        Ok(buf)
    }

    /// Read exactly `buf.len()` bytes starting at byte `offset`. The cooked
    /// reader is a plain byte stream, so no sector alignment is required.
    pub fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        self.reader.seek(SeekFrom::Start(offset))?;
        self.reader.read_exact(buf)
    }
}

/// The `.diff.chd` sidecar path for a base CHD (honors `IRIS_CHD_DIFF_DIR`).
/// Public so the GUI can check for / locate a disk's overlay for commit/rollback.
pub fn diff_path_for(parent: &Path) -> PathBuf {
    // A compressed HD CHD can't be written in place, so writes go to an
    // uncompressed `.diff.chd` sidecar. By default it sits next to the parent.
    //
    // That fails under the macOS App Sandbox: the user grants access to the CHD
    // *file*, but creating a new sibling in its directory needs write access to
    // the directory, which the sandbox denies. iris-gui's App Store build sets
    // IRIS_CHD_DIFF_DIR to a writable container path; when present, put the diff
    // there, named by the parent's stem plus a hash of its full path so two
    // like-named CHDs in different folders don't collide.
    if let Some(dir) = std::env::var_os("IRIS_CHD_DIFF_DIR") {
        let dir = PathBuf::from(dir);
        let _ = std::fs::create_dir_all(&dir);
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        parent.hash(&mut h);
        let stem = parent.file_stem().and_then(|s| s.to_str()).unwrap_or("disk");
        return dir.join(format!("{stem}.{:016x}.diff.chd", h.finish()));
    }
    let mut s = parent.as_os_str().to_owned();
    s.push(".diff.chd");
    PathBuf::from(s)
}

pub fn is_chd(path: &str) -> bool {
    Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("chd"))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use libchdman_rs::hd::{create_from_reader, HdCreateOptions};
    use libchdman_rs::{Chd, CHD_CODEC_ZLIB};
    use std::io::Cursor;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn unique_base() -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        std::env::temp_dir().join(format!(
            "iris_flatten_{}_{}.chd",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ))
    }

    /// End-to-end: a compressed base gets a write via its diff, and flatten folds
    /// the write back into the (still-compressed) base, removing the diff.
    #[test]
    fn flatten_folds_diff_into_compressed_base() {
        let base = unique_base();
        let logical = 256 * 1024u64; // 512 sectors of 512 bytes
        // Build a COMPRESSED base full of 0xAB.
        let src = vec![0xABu8; logical as usize];
        create_from_reader(
            Cursor::new(src),
            &base,
            HdCreateOptions {
                logical_size: logical,
                hunk_size: 4096,
                unit_size: 512,
                codecs: [CHD_CODEC_ZLIB, 0, 0, 0],
                geometry: None,
                ident: None,
            },
            &mut |_| {},
            &|| false,
        )
        .unwrap();

        // Open via ChdHd: compressed → can't write in place → diff is created.
        let mut hd = ChdHd::open(base.to_str().unwrap(), false).unwrap();
        assert!(hd.pending_sync().is_none(), "fresh diff, nothing written → not pending");
        hd.write_sectors(2, &[0x5Au8; 512]).unwrap();
        let (b, d) = hd.pending_sync().expect("a write makes it pending");
        assert_eq!(b, base);
        assert!(d.exists(), "diff sidecar exists");
        drop(hd); // close the CHD before flattening

        // Flatten: fold the diff back into the base.
        let mut last = 0.0f32;
        flatten_diff(&b, &d, &mut |f| last = f, &|| false).unwrap();
        assert_eq!(last, 1.0, "progress reaches 100%");
        assert!(!d.exists(), "diff removed after a successful flatten");

        // Base now carries the write, is still readable, and still compressed.
        {
            let chd = Chd::open(base.to_str().unwrap(), false, None).unwrap();
            assert!(chd.info().unwrap().compressed, "base is still a compressed CHD");
            chd.verify().expect("flattened base verifies");
        }
        // Reopen as a disk (compressed → fresh empty diff) and read the sectors.
        let mut hd2 = ChdHd::open(base.to_str().unwrap(), false).unwrap();
        assert_eq!(hd2.read_blocks(2, 1, 512).unwrap(), vec![0x5A; 512], "the written sector folded in");
        assert_eq!(hd2.read_blocks(0, 1, 512).unwrap(), vec![0xAB; 512], "untouched sectors preserved");
        drop(hd2);

        // Cleanup.
        let _ = std::fs::remove_file(&base);
        let _ = std::fs::remove_file(diff_path_for(&base));
    }

    /// COW keeps changes in the overlay (no auto-fold), the same diff DOES auto-
    /// fold when COW is off, and a rollback (discard the diff) restores the base.
    #[test]
    fn cow_keeps_changes_separate_and_rolls_back() {
        let base = unique_base();
        let logical = 128 * 1024u64;
        create_from_reader(
            Cursor::new(vec![0xCCu8; logical as usize]),
            &base,
            HdCreateOptions {
                logical_size: logical,
                hunk_size: 4096,
                unit_size: 512,
                codecs: [CHD_CODEC_ZLIB, 0, 0, 0],
                geometry: None,
                ident: None,
            },
            &mut |_| {},
            &|| false,
        )
        .unwrap();

        // COW on: writes land in the overlay and are NOT pending an auto-fold.
        let mut hd = ChdHd::open(base.to_str().unwrap(), true).unwrap();
        assert!(hd.is_cow());
        assert!(hd.overlay_paths().is_some());
        hd.write_sectors(1, &[0x33u8; 512]).unwrap();
        assert!(hd.diff_dirty());
        assert!(hd.pending_sync().is_none(), "COW keeps changes separate — never auto-folds");
        drop(hd);

        // Reopen COW off: the same dirty diff IS now pending an exit fold, and the
        // write is visible through the overlay.
        let mut hd_off = ChdHd::open(base.to_str().unwrap(), false).unwrap();
        assert!(hd_off.pending_sync().is_some(), "COW off: the diff auto-folds on a clean exit");
        assert_eq!(hd_off.read_blocks(1, 1, 512).unwrap(), vec![0x33; 512], "write visible via the overlay");
        drop(hd_off);

        // Roll back: discard the diff. A fresh overlay over the untouched base
        // reads the original content again.
        std::fs::remove_file(diff_path_for(&base)).unwrap();
        let mut hd_rb = ChdHd::open(base.to_str().unwrap(), true).unwrap();
        assert_eq!(hd_rb.read_blocks(1, 1, 512).unwrap(), vec![0xCC; 512], "rollback restored the base content");
        drop(hd_rb);

        let _ = std::fs::remove_file(&base);
        let _ = std::fs::remove_file(diff_path_for(&base));
    }

    /// A cancelled flatten leaves the base and diff intact.
    #[test]
    fn cancelled_flatten_preserves_base_and_diff() {
        let base = unique_base();
        let logical = 128 * 1024u64;
        create_from_reader(
            Cursor::new(vec![0x11u8; logical as usize]),
            &base,
            HdCreateOptions {
                logical_size: logical,
                hunk_size: 4096,
                unit_size: 512,
                codecs: [CHD_CODEC_ZLIB, 0, 0, 0],
                geometry: None,
                ident: None,
            },
            &mut |_| {},
            &|| false,
        )
        .unwrap();
        let mut hd = ChdHd::open(base.to_str().unwrap(), false).unwrap();
        hd.write_sectors(1, &[0x22u8; 512]).unwrap();
        let (b, d) = hd.pending_sync().unwrap();
        drop(hd);

        let err = flatten_diff(&b, &d, &mut |_| {}, &|| true).unwrap_err();
        let _ = err; // cancellation surfaces as an error
        assert!(b.exists(), "base intact after cancel");
        assert!(d.exists(), "diff intact after cancel");
        assert!(!temp_sync_path_for(&b).exists(), "no temp left behind");

        let _ = std::fs::remove_file(&base);
        let _ = std::fs::remove_file(&d);
    }
}
