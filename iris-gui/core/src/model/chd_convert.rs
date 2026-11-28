//! What a CHD hard disk is compressed with, and converting it to a format
//! that is quicker to stop.
//!
//! A compressed CHD cannot be written in place: the guest's writes go to a
//! `.diff.chd`, and every stop folds it back by rebuilding and recompressing
//! the whole disk ("Synchronizing disks"). How long that takes depends on the
//! codecs: chdman's default for hard disks (lzma first) is the slowest; zlib
//! alone rebuilds faster and every chdman reads it; an uncompressed CHD is
//! written in place and never rebuilds, but its file grows to the disk's used
//! size. Timings: `clanker/perf/chd-codecs-compressed-disks-sync-on-every-stop.md`.
//!
//! A conversion reads the disk as the guest sees it (the base with its diff,
//! if one is pending), writes the new CHD beside it and renames it over the
//! base. With a diff it follows the emulator's own fold protocol
//! (`iris::chd_disk::flatten_diff`): a `.folding` marker holding the new
//! base's SHA-1 is made durable before the rename, so a crash between the
//! rename and removing the diff is finished by the next open.

use std::io::{self, Read};
use std::path::{Path, PathBuf};

use libchdman_rs::hd::{create_from_reader, read_geometry, HdCreateOptions};
use libchdman_rs::{Chd, CompressionProgress, CHD_CODEC_ZLIB};

/// What a conversion writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChdFormat {
    /// zlib alone: small, a faster rebuild than lzma, read by every chdman.
    Zlib,
    /// Written in place: no rebuild at all, a larger file.
    Uncompressed,
}

impl ChdFormat {
    fn codecs(self) -> [u32; 4] {
        match self {
            ChdFormat::Zlib => [CHD_CODEC_ZLIB, 0, 0, 0],
            ChdFormat::Uncompressed => [0; 4],
        }
    }
}

/// A CHD's header, as far as choosing a format goes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChdHeader {
    /// The codec tags in use ("lzma", "zlib", ...), empty when uncompressed.
    pub codecs: Vec<String>,
    /// The disk's size as the guest sees it.
    pub logical_bytes: u64,
    /// The file's size.
    pub file_bytes: u64,
}

impl ChdHeader {
    pub fn compressed(&self) -> bool {
        !self.codecs.is_empty()
    }

    /// Compressed with a codec that makes every stop slow to rebuild (lzma,
    /// chdman's default for hard disks, or zstd at MAME's level).
    pub fn slow(&self) -> bool {
        self.codecs.iter().any(|c| c == "lzma" || c == "zstd")
    }

    /// Already the zlib-only format a conversion writes.
    pub fn zlib_only(&self) -> bool {
        self.codecs == ["zlib"]
    }
}

/// Read the v5 header of the CHD at `path`: `None` for anything else (not a
/// CHD, an older version, unreadable). Reads 124 bytes; never decompresses.
pub fn header(path: &Path) -> Option<ChdHeader> {
    let mut f = std::fs::File::open(path).ok()?;
    let mut h = [0u8; 124];
    f.read_exact(&mut h).ok()?;
    if &h[..8] != b"MComprHD" || u32::from_be_bytes(h[12..16].try_into().ok()?) != 5 {
        return None;
    }
    let codecs = (0..4)
        .map(|i| &h[16 + 4 * i..20 + 4 * i])
        .filter(|t| t.iter().any(|&b| b != 0))
        .map(|t| String::from_utf8_lossy(t).into_owned())
        .collect();
    let logical_bytes = u64::from_be_bytes(h[32..40].try_into().ok()?);
    let file_bytes = f.metadata().ok()?.len();
    Some(ChdHeader { codecs, logical_bytes, file_bytes })
}

fn other(e: impl std::fmt::Debug) -> io::Error {
    io::Error::other(format!("{e:?}"))
}

/// The disk as the guest sees it, front to back.
struct DiskReader {
    // Declared first, dropped first: the diff refers to its parent.
    diff: Option<Chd>,
    base: Chd,
    pos: u64,
    size: u64,
}

impl Read for DiskReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.pos >= self.size {
            return Ok(0);
        }
        let n = buf.len().min((self.size - self.pos) as usize);
        self.diff.as_ref().unwrap_or(&self.base).read_bytes(self.pos, &mut buf[..n]).map_err(other)?;
        self.pos += n as u64;
        Ok(n)
    }
}

fn with_suffix(p: &Path, suffix: &str) -> PathBuf {
    let mut s = p.as_os_str().to_owned();
    s.push(suffix);
    PathBuf::from(s)
}

fn fsync(p: &Path) -> io::Result<()> {
    std::fs::OpenOptions::new().read(true).write(true).open(p)?.sync_all()
}

#[cfg(not(windows))]
fn fsync_dir(p: &Path) -> io::Result<()> {
    match p.parent() {
        Some(d) if !d.as_os_str().is_empty() => std::fs::File::open(d)?.sync_all(),
        _ => Ok(()),
    }
}

#[cfg(windows)]
fn fsync_dir(_p: &Path) -> io::Result<()> {
    Ok(())
}

/// Rewrite the CHD at `path` in `format`, folding in its pending `.diff.chd`
/// at `diff_path` (see [`crate::model::machines::chd_diff`]). The file stays
/// where it is, under the same name, so the machine's settings need no
/// change. On an error or a cancel, everything is left as it was. Returns
/// the new file's size.
pub fn convert(
    path: &Path,
    diff_path: &Path,
    format: ChdFormat,
    progress: &mut dyn FnMut(f32),
    cancel: &dyn Fn() -> bool,
) -> io::Result<u64> {
    let name = path.to_str().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "the path is not UTF-8"))?;
    let base = Chd::open(name, false, None).map_err(other)?;
    let info = base.info().map_err(other)?;
    if !info.is_hd {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "not a hard-disk CHD"));
    }
    let geometry = read_geometry(&base).ok();
    let diff = if diff_path.exists() {
        let d = diff_path.to_str().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "the path is not UTF-8"))?;
        Some(Chd::open(d, false, Some(&base)).map_err(other)?)
    } else {
        None
    };
    let had_diff = diff.is_some();
    let reader = DiskReader { diff, base, pos: 0, size: info.logical_bytes };
    let tmp = with_suffix(path, ".convert.chd");
    let _ = std::fs::remove_file(&tmp);
    let opts = HdCreateOptions {
        logical_size: info.logical_bytes,
        hunk_size: info.hunk_bytes,
        unit_size: info.unit_bytes,
        codecs: format.codecs(),
        geometry,
        ident: None,
    };
    let total = info.logical_bytes.max(1) as f64;
    let mut cb = |p: CompressionProgress| progress((p.bytes_done as f64 / total).min(1.0) as f32);
    if let Err(e) = create_from_reader(reader, &tmp, opts, &mut cb, cancel) {
        let _ = std::fs::remove_file(&tmp);
        return Err(other(e));
    }
    if cancel() {
        let _ = std::fs::remove_file(&tmp);
        return Err(io::Error::new(io::ErrorKind::Interrupted, "cancelled"));
    }
    let marker = with_suffix(&diff_path, ".folding");
    let ready = fsync(&tmp).and_then(|()| {
        if had_diff {
            let sha = Chd::open(tmp.to_str().unwrap_or_default(), false, None).map_err(other)?.sha1();
            std::fs::write(&marker, sha)?;
            fsync(&marker)?;
            fsync_dir(&marker)?;
        }
        Ok(())
    });
    if let Err(e) = ready {
        let _ = std::fs::remove_file(&marker);
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    std::fs::rename(&tmp, path)?;
    let _ = fsync_dir(path);
    if had_diff {
        let _ = std::fs::remove_file(&diff_path);
        let _ = fsync_dir(&diff_path);
        let _ = std::fs::remove_file(&marker);
    }
    progress(1.0);
    Ok(std::fs::metadata(path)?.len())
}

/// Free bytes on the volume holding `path`, when the host says.
#[cfg(unix)]
pub fn free_bytes(path: &Path) -> Option<u64> {
    use std::os::unix::ffi::OsStrExt;
    let dir = path.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let c = std::ffi::CString::new(dir.as_os_str().as_bytes()).ok()?;
    let mut s: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: a NUL-terminated path and a zeroed struct for statvfs to fill.
    (unsafe { libc::statvfs(c.as_ptr(), &mut s) } == 0).then(|| s.f_bavail as u64 * s.f_frsize as u64)
}

#[cfg(not(unix))]
pub fn free_bytes(_path: &Path) -> Option<u64> {
    None
}

/// What the new file may need, at most: the whole disk uncompressed; for
/// zlib, a little more than the file now (it compresses a little less than
/// lzma: 1.27 GB against 1.15 GB in the measurements).
pub fn room_needed(h: &ChdHeader, format: ChdFormat) -> u64 {
    match format {
        ChdFormat::Uncompressed => h.logical_bytes,
        ChdFormat::Zlib => h.file_bytes + h.file_bytes / 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small compressed HD CHD of `bytes`, filled with a pattern.
    fn make(dir: &Path, name: &str, bytes: u64, codecs: [u32; 4]) -> PathBuf {
        let p = dir.join(name);
        let data: Vec<u8> = (0..bytes).map(|i| (i / 512 % 251) as u8).collect();
        let opts = HdCreateOptions { logical_size: bytes, hunk_size: 4096, unit_size: 512, codecs, geometry: None, ident: None };
        create_from_reader(&data[..], &p, opts, &mut |_| {}, &|| false).unwrap();
        p
    }

    #[test]
    fn a_compressed_disk_converts_and_keeps_its_contents() {
        let root = crate::model::machines::testing::TempRoot::new("chd-convert");
        let src = make(&root.0, "d.chd", 1 << 20, [libchdman_rs::CHD_CODEC_LZMA, CHD_CODEC_ZLIB, 0, 0]);
        let before = header(&src).unwrap();
        assert!(before.slow() && before.compressed(), "{before:?}");
        assert_eq!(before.logical_bytes, 1 << 20);
        for (format, want) in [(ChdFormat::Zlib, vec!["zlib".to_string()]), (ChdFormat::Uncompressed, vec![])] {
            let mut last = 0.0;
            convert(&src, &iris::chd_disk::diff_path_for(&src), format, &mut |f| last = f, &|| false).unwrap();
            assert_eq!(last, 1.0);
            let h = header(&src).unwrap();
            assert_eq!(h.codecs, want);
            let chd = Chd::open(src.to_str().unwrap(), false, None).unwrap();
            let mut buf = vec![0u8; 4096];
            chd.read_bytes(600 * 1024, &mut buf).unwrap();
            let want: Vec<u8> = (600 * 1024..600 * 1024 + 4096u64).map(|i| (i / 512 % 251) as u8).collect();
            assert_eq!(buf, want, "{format:?} kept the data");
        }
        assert!(!with_suffix(&src, ".convert.chd").exists());
    }

    #[test]
    fn writes_waiting_in_the_diff_go_into_the_new_disk() {
        let root = crate::model::machines::testing::TempRoot::new("chd-diff");
        let src = make(&root.0, "d.chd", 1 << 20, [libchdman_rs::CHD_CODEC_LZMA, 0, 0, 0]);
        {
            // What the guest does: a compressed base takes its writes in a diff.
            let mut hd = iris::chd_disk::ChdHd::open(src.to_str().unwrap(), false).unwrap();
            hd.write_sectors(100, &[0xAB; 1024]).unwrap();
            hd.file().sync(true).unwrap();
        }
        let diff = iris::chd_disk::diff_path_for(&src);
        assert!(diff.exists());
        convert(&src, &diff, ChdFormat::Zlib, &mut |_| {}, &|| false).unwrap();
        assert!(!diff.exists(), "the diff was folded in and removed");
        assert!(!with_suffix(&diff, ".folding").exists());
        let chd = Chd::open(src.to_str().unwrap(), false, None).unwrap();
        let mut buf = vec![0u8; 2048];
        chd.read_bytes(100 * 512 - 512, &mut buf).unwrap();
        assert_eq!(buf[..512], vec![(99u64 % 251) as u8; 512][..], "the sector before is unchanged");
        assert!(buf[512..1536].iter().all(|&b| b == 0xAB), "the guest's write is there");
        assert_eq!(buf[1536..], vec![(102u64 % 251) as u8; 512][..]);
    }

    #[test]
    fn a_cancelled_conversion_leaves_the_disk_alone() {
        let root = crate::model::machines::testing::TempRoot::new("chd-cancel");
        let src = make(&root.0, "d.chd", 1 << 20, [libchdman_rs::CHD_CODEC_LZMA, 0, 0, 0]);
        let before = std::fs::read(&src).unwrap();
        assert!(convert(&src, &iris::chd_disk::diff_path_for(&src), ChdFormat::Uncompressed, &mut |_| {}, &|| true).is_err());
        assert_eq!(std::fs::read(&src).unwrap(), before);
        assert!(!with_suffix(&src, ".convert.chd").exists());
    }

    #[test]
    fn a_file_that_is_not_a_chd_has_no_header() {
        let root = crate::model::machines::testing::TempRoot::new("chd-none");
        let p = root.0.join("disk.img");
        std::fs::write(&p, vec![0u8; 4096]).unwrap();
        assert_eq!(header(&p), None);
    }
}
