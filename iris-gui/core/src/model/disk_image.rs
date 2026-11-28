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

/// Write a blank image of `mb` MiB at `path`.
///
/// Fails rather than replacing anything: an existing file, even an old
/// overlay sidecar, keeps its contents and the caller is told to choose
/// another name. `progress` is called with a fraction in `0.0..=1.0`;
/// returning true from `cancel` stops it and leaves nothing behind.
pub fn create(
    path: &Path,
    format: Format,
    mb: u64,
    progress: &mut dyn FnMut(f32),
    cancel: &dyn Fn() -> bool,
) -> Result<(), String> {
    let mb = mb.clamp(MIN_MB, MAX_MB);
    let bytes = mb * 1024 * 1024;
    if path.exists() {
        return Err(t!("create-disk-exists", file = path.display().to_string()));
    }
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
    }
    match format {
        Format::Raw => {
            // create_new: the file must not be there, so nothing is truncated.
            let f = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .map_err(|e| format!("{}: {e}", path.display()))?;
            f.set_len(bytes).map_err(|e| {
                let _ = std::fs::remove_file(path);
                e.to_string()
            })?;
            progress(1.0);
            Ok(())
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
            create_from_reader(Zeroes { left: bytes }, path, opts, &mut report, cancel).map_err(|e| {
                let _ = std::fs::remove_file(path);
                format!("{e:?}")
            })
        }
    }
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

    #[test]
    fn a_name_takes_its_formats_extension() {
        assert_eq!(Format::Chd.with_extension(Path::new("/a/b")), Path::new("/a/b.chd"));
        assert_eq!(Format::Chd.with_extension(Path::new("/a/b.img")), Path::new("/a/b.chd"));
        assert_eq!(Format::Raw.with_extension(Path::new("/a/b.raw")), Path::new("/a/b.raw"));
        assert_eq!(Format::of(Path::new("/a/b.chd")), Some(Format::Chd));
        assert_eq!(Format::of(Path::new("/a/b.iso")), None);
    }
}
