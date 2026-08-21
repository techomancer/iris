//! Where a file dialog opens.
//!
//! The rule this module exists to enforce: **a dialog is always given a real
//! directory.** That sounds obvious, and every picker in the app used to get it
//! wrong in the same way — seed the panel at the current path's folder *if that
//! folder happens to exist*, otherwise set nothing at all.
//!
//! Setting nothing is not neutral. On macOS `NSOpenPanel`/`NSSavePanel` restore
//! their own last-used location when `setDirectoryURL:` is not given one, so
//! "do nothing" means "open wherever the user last happened to be". That is
//! wrong precisely when it is most annoying: a disk image that has not been
//! created yet has no existing parent folder, so browsing for it opened
//! somewhere unrelated instead of where the image is destined to go. Windows
//! and the GTK/portal backends have their own remembered defaults and behave
//! the same way, so this is a shared fix rather than a macOS workaround.
//!
//! So: resolve the folder the file lives in or is headed for, walk *up* to the
//! nearest ancestor that exists rather than giving up, and fall back to the
//! app's own managed directory — creating that one, since it is ours and it is
//! the location the UI tells people their disks go to.

use std::path::{Path, PathBuf};

use crate::settings::GuiSettings;

/// Which managed directory to fall back on when the current value offers no
/// usable folder of its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Anchor {
    /// Disk images — `<data_dir>/disks`.
    Disks,
    /// Everything else the app owns: PROM, NVRAM, logs, exports, screenshots.
    Data,
}

impl Anchor {
    /// The managed directory, created if missing.
    ///
    /// Creating it as a side effect of opening a picker is deliberate and is
    /// limited to directories the app owns: it is where the UI says disks go,
    /// an empty one is harmless, and the alternative is the panel opening
    /// somewhere unrelated on a fresh install. A path the *user* chose is never
    /// created here — that walks up to an existing ancestor instead.
    fn managed(self) -> Option<PathBuf> {
        let dir = match self {
            Anchor::Disks => GuiSettings::disks_dir()?,
            Anchor::Data => GuiSettings::data_dir()?,
        };
        if !dir.is_dir() {
            let _ = std::fs::create_dir_all(&dir);
        }
        dir.is_dir().then_some(dir)
    }
}

/// The folder a dialog for `current` should open in. Always exists.
pub fn start_dir(current: &str, anchor: Anchor) -> PathBuf {
    start_dir_in(current, anchor.managed())
}

/// `start_dir` with the managed directory supplied, so the logic is testable
/// without touching the real one.
fn start_dir_in(current: &str, managed: Option<PathBuf>) -> PathBuf {
    let current = current.trim();
    if !current.is_empty() {
        // A bare or relative name means the managed directory, not the process
        // working directory — that differs between `cargo run` and a bundled
        // .app, which is the same trap `GuiSettings::default_nvram_path`
        // documents.
        let p = match (Path::new(current).is_absolute(), &managed) {
            (false, Some(base)) => base.join(current),
            _ => PathBuf::from(current),
        };
        if let Some(dir) = p.parent().and_then(nearest_existing) {
            return dir;
        }
    }
    // Validated rather than trusted: `Anchor::managed` already filters, but this
    // function must not depend on its caller having done so — returning a
    // directory that does not exist is the exact failure it exists to prevent.
    managed.filter(|d| d.is_dir()).unwrap_or_else(last_resort)
}

/// The nearest ancestor of `dir` that exists — `dir` itself when it does.
///
/// Walking up is what makes a not-yet-created destination useful: a disk bound
/// for `~/VMs/indy/disks/root.raw` opens at `~/VMs/indy` if that is as far as
/// the tree goes, which is one folder from where the user is aiming rather than
/// wherever they last browsed.
fn nearest_existing(dir: &Path) -> Option<PathBuf> {
    dir.ancestors()
        .find(|a| !a.as_os_str().is_empty() && a.is_dir())
        .map(PathBuf::from)
}

fn last_resort() -> PathBuf {
    dirs::home_dir()
        .filter(|d| d.is_dir())
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("."))
}

/// An `rfd::FileDialog` seeded at [`start_dir`], with the file name pre-filled
/// when `current` names one (which a save panel shows and an open panel
/// ignores).
pub fn dialog(title: &str, current: &str, anchor: Anchor) -> rfd::FileDialog {
    let mut d = rfd::FileDialog::new()
        .set_title(title)
        .set_directory(start_dir(current, anchor));
    if let Some(name) = Path::new(current.trim()).file_name() {
        d = d.set_file_name(name.to_string_lossy());
    }
    d
}

/// A picker opened *at* `dir` rather than at its parent.
///
/// For choosing a folder when you already know which one you mean — granting
/// sandbox access to the folder a disk image sits in, say — so confirming it is
/// one click. `dialog` deliberately does the opposite for files, where the
/// parent is the useful view.
pub fn dialog_at_dir(title: &str, dir: &str, anchor: Anchor) -> rfd::FileDialog {
    let start = nearest_existing(Path::new(dir.trim()))
        .unwrap_or_else(|| start_dir("", anchor));
    rfd::FileDialog::new().set_title(title).set_directory(start)
}

/// `dialog` with a set of `(label, extensions)` filters applied.
pub fn dialog_with(
    title: &str,
    current: &str,
    anchor: Anchor,
    filters: &[(&str, &[&str])],
) -> rfd::FileDialog {
    let mut d = dialog(title, current, anchor);
    for (label, exts) in filters {
        d = d.add_filter(*label, exts);
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A directory this test owns, under the scratch area rather than the
    /// user's real config dir.
    fn tmp(name: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!("iris-filedialog-{name}-{nanos}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn an_existing_folder_is_used_as_is() {
        let managed = tmp("existing");
        let img = managed.join("scsi1.raw");
        std::fs::write(&img, b"x").unwrap();
        assert_eq!(start_dir_in(&img.to_string_lossy(), Some(managed.clone())), managed);
        std::fs::remove_dir_all(&managed).ok();
    }

    /// The case that motivated all of this: the image does not exist yet, so
    /// its folder does not either. The old code set no directory at all and the
    /// panel opened at whatever the OS remembered.
    #[test]
    fn a_destination_that_does_not_exist_yet_walks_up_to_one_that_does() {
        let managed = tmp("walkup");
        let deep = managed.join("indy").join("disks").join("root.raw");
        assert_eq!(start_dir_in(&deep.to_string_lossy(), Some(managed.clone())), managed,
                   "must climb to the nearest real folder, not give up");

        // One level materialises: now that is the answer.
        let mid = managed.join("indy");
        std::fs::create_dir_all(&mid).unwrap();
        assert_eq!(start_dir_in(&deep.to_string_lossy(), Some(managed.clone())), mid);
        std::fs::remove_dir_all(&managed).ok();
    }

    #[test]
    fn a_bare_name_resolves_against_the_managed_folder_not_the_working_directory() {
        let managed = tmp("bare");
        // `scsi1.raw` with no directory means the managed one. Resolving it
        // against the process cwd would differ between `cargo run` and a
        // bundled .app.
        assert_eq!(start_dir_in("scsi1.raw", Some(managed.clone())), managed);
        assert_eq!(start_dir_in("sub/scsi1.raw", Some(managed.clone())), managed);
        std::fs::remove_dir_all(&managed).ok();
    }

    #[test]
    fn an_empty_value_falls_back_to_the_managed_folder() {
        let managed = tmp("empty");
        assert_eq!(start_dir_in("", Some(managed.clone())), managed);
        assert_eq!(start_dir_in("   ", Some(managed.clone())), managed);
        std::fs::remove_dir_all(&managed).ok();
    }

    #[test]
    fn a_folder_picker_opens_at_the_folder_not_above_it() {
        let managed = tmp("atdir");
        let sub = managed.join("images");
        std::fs::create_dir_all(&sub).unwrap();

        // The grant flow hands us a folder and wants it confirmed in one click,
        // so taking its parent (which is right for a *file*) would be wrong.
        let d = nearest_existing(&sub).unwrap();
        assert_eq!(d, sub);
        // A folder that has since been deleted still lands somewhere real.
        std::fs::remove_dir_all(&sub).unwrap();
        assert_eq!(nearest_existing(&sub).unwrap(), managed);
        std::fs::remove_dir_all(&managed).ok();
    }

    /// The invariant the whole module exists for. Whatever it is handed —
    /// including a managed directory that could not be created — it must name a
    /// real directory, because handing the panel nothing is what made it open
    /// in the wrong place.
    #[test]
    fn the_result_is_always_a_directory_that_exists() {
        let cases = ["", "   ", "scsi1.raw", "/nonexistent/deep/path/x.raw", "relative/x.raw"];
        for managed in [None, Some(PathBuf::from("/nonexistent/managed"))] {
            for c in cases {
                let d = start_dir_in(c, managed.clone());
                assert!(d.is_dir(), "start_dir_in({c:?}, {managed:?}) = {d:?}, which is not a directory");
            }
        }
    }
}
