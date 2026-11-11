//! A temporary file or directory owned by one test, removed when the test
//! ends, whether it passed or panicked.
//!
//! Tests used to build a path under the temp dir and remove it with a trailing
//! `remove_file`, which a failed assert skips; the leftovers piled up in the
//! host's temp dir. Every test that needs a path on disk takes a `TempPath`
//! instead.
//!
//! Names carry the process id and a per-process counter: parallel cargo runs
//! and nextest share one temp dir, and under plain `cargo test` every test
//! shares one pid, so neither alone keeps two tests apart.

use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

pub(crate) struct TempPath(PathBuf);

/// Point a machine's NVRAM and EEPROM files into `dir`. A config that leaves
/// them at their defaults writes `nvram.bin` and `nveeprom.bin` into the
/// working directory when the machine stops, which under cargo is the
/// package root.
pub(crate) fn nvram_in(cfg: &mut crate::config::MachineConfig, dir: &Path) {
    cfg.nvram = dir.join("nvram").to_string_lossy().into_owned();
    cfg.nveeprom = dir.join("nveeprom").to_string_lossy().into_owned();
}

impl TempPath {
    /// A fresh path under the temp dir named after `tag`. Nothing exists there yet.
    pub(crate) fn new(tag: &str) -> Self {
        static N: AtomicU32 = AtomicU32::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!("iris-{tag}-{}-{n}", std::process::id()));
        // A killed run of an earlier process with the same pid may have left one.
        let _ = std::fs::remove_dir_all(&p);
        let _ = std::fs::remove_file(&p);
        Self(p)
    }

    /// A fresh path holding a file with `bytes`.
    pub(crate) fn file(tag: &str, bytes: &[u8]) -> Self {
        let p = Self::new(tag);
        std::fs::write(&p.0, bytes).expect("write temp file");
        p
    }

    /// A fresh path holding an empty directory.
    pub(crate) fn dir(tag: &str) -> Self {
        let p = Self::new(tag);
        std::fs::create_dir_all(&p.0).expect("create temp dir");
        p
    }
}

impl Deref for TempPath {
    type Target = Path;
    fn deref(&self) -> &Path {
        &self.0
    }
}

impl AsRef<Path> for TempPath {
    fn as_ref(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempPath {
    fn drop(&mut self) {
        // A disk opened with copy-on-write keeps its writes in `<path>.overlay`
        // beside the image, with the dirty-sector record in `.overlay.dirty`.
        let overlay = format!("{}.overlay", self.0.display());
        for p in [overlay.clone(), format!("{overlay}.dirty")] {
            let _ = std::fs::remove_file(p);
        }
        // Removing an open file fails on Windows; the device that holds it
        // has closed it by the time the test's locals unwind.
        if self.0.is_dir() {
            let _ = std::fs::remove_dir_all(&self.0);
        } else {
            let _ = std::fs::remove_file(&self.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_file_is_removed_with_its_overlay_even_on_panic() {
        let (img, overlay) = {
            let t = TempPath::file("temp-test", b"abc");
            let overlay = PathBuf::from(format!("{}.overlay", t.display()));
            std::fs::write(&overlay, b"x").unwrap();
            assert_eq!(std::fs::read(&t).unwrap(), b"abc");
            let paths = (t.to_path_buf(), overlay.clone());
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                let _t = t;
                panic!("test failed");
            }));
            assert!(r.is_err());
            paths
        };
        assert!(!img.exists() && !overlay.exists());
    }

    #[test]
    fn a_dir_is_removed_with_its_contents_and_names_are_distinct() {
        let a = TempPath::dir("temp-test");
        let b = TempPath::dir("temp-test");
        assert_ne!(*a, *b);
        std::fs::write(a.join("f"), b"1").unwrap();
        let p = a.to_path_buf();
        drop(a);
        assert!(!p.exists() && b.is_dir());
    }
}
