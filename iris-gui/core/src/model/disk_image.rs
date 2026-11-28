//! Blank hard-disk images: a sparse raw file or a CHD, written on a worker.
//!
//! The shape follows Prismatic's `src/disk_image.rs` (salvogendut/prismatic,
//! BSD-3-Clause): raw or CHD, whole megabytes, created in the background with
//! progress and cancel, and **never replacing an existing file** — the caller
//! picks another name instead of being offered an overwrite. The image holds
//! zero-filled 512-byte sectors and has no partition table or filesystem; the
//! guest's own tools (IRIX `fx`) lay it out.

use std::io::{self, Read};
use std::path::{Path, PathBuf};

/// The smallest and largest image offered, in MiB. The ceiling is 1 TiB,
/// inside the guests' 32-bit sector addressing.
pub const MIN_MB: u64 = 1;
pub const MAX_MB: u64 = 1_048_576;
pub const DEFAULT_MB: u64 = 4096;

/// A sector, and the hunk a CHD compresses.
const SECTOR: u32 = 512;
const HUNK: u32 = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// A sparse file where the host filesystem allows one.
    Raw,
    /// Compressed, with the geometry and hashes iris's protected writes want.
    Chd,
}

impl Format {
    pub const ALL: [Format; 2] = [Format::Raw, Format::Chd];

    /// The extensions this format is written with; the first is added when
    /// the name has none.
    pub fn extensions(self) -> &'static [&'static str] {
        match self {
            Format::Raw => &["img", "raw", "dsk", "hdd"],
            Format::Chd => &["chd"],
        }
    }

    /// `path` with this format's extension, adding the default when there is
    /// none. An extension belonging to the other format is replaced.
    pub fn with_extension(self, path: &Path) -> PathBuf {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or_default().to_ascii_lowercase();
        if self.extensions().contains(&ext.as_str()) {
            return path.to_path_buf();
        }
        path.with_extension(self.extensions()[0])
    }

    /// The format a name asks for, by its extension.
    pub fn of(path: &Path) -> Option<Format> {
        let ext = path.extension()?.to_str()?.to_ascii_lowercase();
        Format::ALL.into_iter().find(|f| f.extensions().contains(&ext.as_str()))
    }
}

/// Zero bytes, as many as asked for: the source of a blank image.
struct Zeroes {
    left: u64,
}

impl Read for Zeroes {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = buf.len().min(self.left as usize);
        buf[..n].fill(0);
        self.left -= n as u64;
        Ok(n)
    }
}

/// Whether anything sits at `path`, or at a sidecar a disk there would pick
/// up at its first start: a copy-on-write overlay or a CHD diff left by an
/// earlier image of the same name. A new image there would read as the old
/// one, so the name counts as taken.
pub fn taken(path: &Path) -> bool {
    let sidecar = |s: &str| {
        let mut p = path.as_os_str().to_owned();
        p.push(s);
        PathBuf::from(p)
    };
    [path.to_path_buf(), sidecar(".overlay"), sidecar(".overlay.dirty"), sidecar(".diff.chd"), iris::chd_disk::diff_path_for(path)]
        .iter()
        .any(|p| std::fs::symlink_metadata(p).is_ok())
}

/// The temporary file an image is written as, beside `path`, so a cancel or
/// a crash never leaves a half-written image under the name asked for.
fn temporary(path: &Path) -> PathBuf {
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    path.with_file_name(format!(".{name}.creating-{}", std::process::id()))
}

/// Removes the temporary file unless it was published by a rename.
struct Cleanup(PathBuf, bool);

impl Drop for Cleanup {
    fn drop(&mut self) {
        if !self.1 {
            let _ = std::fs::remove_file(&self.0);
        }
    }
}

/// Write a blank image of `mb` MiB at `path`.
///
/// Fails rather than replacing anything: an existing file, or an old
/// overlay or diff a disk of that name would pick up ([`taken`]), keeps its
/// contents and the caller is told to choose another name. The image is
/// written beside `path` and published with a hard link, which fails if
/// something took the name meanwhile. `progress` is called with a fraction
/// in `0.0..=1.0`; returning true from `cancel` stops it and leaves nothing
/// behind.
pub fn create(
    path: &Path,
    format: Format,
    mb: u64,
    progress: &mut dyn FnMut(f32),
    cancel: &dyn Fn() -> bool,
) -> Result<(), String> {
    let mb = mb.clamp(MIN_MB, MAX_MB);
    let bytes = mb * 1024 * 1024;
    let exists = || t!("create-disk-exists", file = path.display().to_string());
    if taken(path) {
        return Err(exists());
    }
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
    }
    let tmp = temporary(path);
    let mut cleanup = Cleanup(tmp.clone(), false);
    match format {
        Format::Raw => {
            // create_new: nothing is ever truncated.
            let f = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&tmp)
                .map_err(|e| format!("{}: {e}", tmp.display()))?;
            f.set_len(bytes).map_err(|e| e.to_string())?;
            f.sync_all().map_err(|e| e.to_string())?;
        }
        Format::Chd => {
            use libchdman_rs::hd::{create_from_reader, HdCreateOptions};
            let opts = HdCreateOptions {
                logical_size: bytes,
                hunk_size: HUNK,
                unit_size: SECTOR,
                codecs: [libchdman_rs::CHD_CODEC_ZLIB, 0, 0, 0],
                geometry: None,
                ident: None,
            };
            let mut report = |p: libchdman_rs::CompressionProgress| {
                let total = if p.bytes_total > 0 { p.bytes_total } else { bytes.max(1) };
                progress((p.bytes_done.min(total) as f64 / total as f64) as f32);
            };
            create_from_reader(Zeroes { left: bytes }, &tmp, opts, &mut report, cancel)
                .map_err(|e| format!("{e:?}"))?;
        }
    }
    if cancel() {
        return Err("cancelled".into());
    }
    match std::fs::hard_link(&tmp, path) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => return Err(exists()),
        // A filesystem without hard links: rename, having just checked.
        Err(_) if !taken(path) => {
            std::fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))?;
            cleanup.1 = true;
        }
        Err(e) => return Err(format!("{}: {e}", path.display())),
    }
    progress(1.0);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::machines::testing::TempRoot;

    #[test]
    fn a_raw_image_is_made_once_and_never_replaced() {
        let root = TempRoot::new("disk-image");
        let path = root.0.join("disks").join("new.img");
        create(&path, Format::Raw, 4, &mut |_| {}, &|| false).expect("created");
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 4 * 1024 * 1024);
        // A second go leaves the first alone.
        let err = create(&path, Format::Raw, 8, &mut |_| {}, &|| false).expect_err("refused");
        assert!(err.contains("new.img"), "{err}");
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 4 * 1024 * 1024);
    }

    /// An old overlay or diff would make a new image read as the old disk,
    /// so its name counts as taken; nothing but the finished image is left.
    #[test]
    fn a_name_with_an_old_sidecar_is_taken_and_nothing_is_left_behind() {
        let root = TempRoot::new("disk-image-sidecar");
        let path = root.0.join("old.img");
        std::fs::write(root.0.join("old.img.overlay"), b"old overlay").unwrap();
        assert!(taken(&path));
        assert!(create(&path, Format::Raw, 1, &mut |_| {}, &|| false).is_err());
        assert!(!path.exists());
        let chd = root.0.join("new.chd");
        let mut last = 0.0;
        create(&chd, Format::Chd, 4, &mut |f| last = f, &|| false).expect("created");
        assert!(last > 0.99, "progress reached the end: {last}");
        let c = libchdman_rs::Chd::open(chd.to_str().unwrap(), false, None).unwrap();
        assert_eq!(c.info().unwrap().logical_bytes, 4 * 1024 * 1024);
        assert!(create(&root.0.join("gone.chd"), Format::Chd, 64, &mut |_| {}, &|| true).is_err());
        let mut names: Vec<_> = std::fs::read_dir(&root.0).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
        names.sort();
        assert_eq!(names, ["new.chd", "old.img.overlay"], "no temporary or cancelled image left");
    }

    #[test]
    fn a_name_takes_its_formats_extension() {
        assert_eq!(Format::Chd.with_extension(Path::new("/a/b")), Path::new("/a/b.chd"));
        assert_eq!(Format::Chd.with_extension(Path::new("/a/b.img")), Path::new("/a/b.chd"));
        assert_eq!(Format::Raw.with_extension(Path::new("/a/b.raw")), Path::new("/a/b.raw"));
        assert_eq!(Format::of(Path::new("/a/b.chd")), Some(Format::Chd));
        assert_eq!(Format::of(Path::new("/a/b.iso")), None);
    }
}
