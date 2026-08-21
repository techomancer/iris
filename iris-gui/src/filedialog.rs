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
//!
//! # Never set a file name and a directory together on macOS
//!
//! Doing both silently throws the directory away. rfd's AppKit backend
//! (`backend/macos/file_dialog/panel_ffi.rs`, `set_path`) joins them into a
//! single path and hands *that* to `setDirectoryURL:` with `isDirectory: YES`:
//!
//! ```text
//!   set_directory("~/disks") + set_file_name("scsi1.raw")
//!     -> NSURL fileURLWithPath:"~/disks/scsi1.raw" isDirectory:YES
//!     -> file:///Users/…/disks/scsi1.raw/          <- not a directory
//!     -> AppKit discards it, panel opens at its own default (Documents)
//! ```
//!
//! Which is the exact symptom this module was first written to fix, and did
//! not: seeding the directory correctly does nothing while a file name is set
//! beside it. The two are mutually exclusive on macOS, and the directory is
//! what matters — an `NSOpenPanel` has no name field for the name to appear in
//! anyway, so setting it there is pure downside.
//!
//! The Linux portal backend keeps `current_folder` and `current_name` as
//! separate fields and is unaffected, so it still gets the pre-filled name on
//! a save panel. Hence [`Purpose`], and hence the platform split — which is
//! deliberate and load-bearing, not an oversight to be tidied away.

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

/// What the dialog is for. Only a save panel has a name field worth filling,
/// and only some platforms can fill it without losing the directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Purpose {
    /// Pick something that exists. Never pre-fills a name — an `NSOpenPanel`
    /// has no field to put one in, and on macOS setting one costs the
    /// directory.
    Open,
    /// Name something new. Pre-fills where the platform allows it.
    Save,
}

/// Whether pre-filling the name field would cost us the starting directory.
/// See the module docs — true on macOS, false everywhere else.
const NAME_FIELD_BREAKS_DIRECTORY: bool = cfg!(target_os = "macos");

/// The name to pre-fill, or `None` when it must be left alone.
///
/// Takes the platform behaviour as an argument rather than reading the `cfg`
/// directly, so both branches are testable from either host.
fn name_to_prefill(current: &str, purpose: Purpose, name_breaks_dir: bool) -> Option<String> {
    if purpose == Purpose::Open || name_breaks_dir {
        return None;
    }
    Path::new(current.trim())
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
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

/// An `rfd::FileDialog` seeded at [`start_dir`].
///
/// The starting directory is always set. The name field is filled only for a
/// save panel, and only where that does not cost us the directory — see the
/// module docs.
pub fn dialog(title: &str, current: &str, anchor: Anchor, purpose: Purpose) -> rfd::FileDialog {
    let mut d = rfd::FileDialog::new()
        .set_title(title)
        .set_directory(start_dir(current, anchor));
    if let Some(name) = name_to_prefill(current, purpose, NAME_FIELD_BREAKS_DIRECTORY) {
        d = d.set_file_name(name);
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
    purpose: Purpose,
    filters: &[(&str, &[&str])],
) -> rfd::FileDialog {
    let mut d = dialog(title, current, anchor, purpose);
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

    /// The bug the module docs describe: on macOS a name beside a directory
    /// throws the directory away, so there must be no name.
    #[test]
    fn a_name_is_never_pre_filled_where_it_would_cost_us_the_directory() {
        // macOS: never, for either kind of panel.
        assert_eq!(name_to_prefill("/d/scsi1.raw", Purpose::Save, true), None);
        assert_eq!(name_to_prefill("/d/scsi1.raw", Purpose::Open, true), None);

        // Elsewhere: a save panel gets it, an open panel still does not — it
        // has nowhere to show it and it is one more thing to go wrong.
        assert_eq!(name_to_prefill("/d/scsi1.raw", Purpose::Save, false),
                   Some("scsi1.raw".to_string()));
        assert_eq!(name_to_prefill("/d/scsi1.raw", Purpose::Open, false), None);

        // Nothing to take a name from.
        assert_eq!(name_to_prefill("", Purpose::Save, false), None);
        assert_eq!(name_to_prefill("/d/", Purpose::Save, false), Some("d".to_string()));
    }

    /// Whatever the platform does about the name, the directory is set — that
    /// is the half the user actually sees.
    #[test]
    fn the_directory_is_seeded_regardless_of_the_name_decision() {
        let managed = tmp("both");
        let img = managed.join("scsi1.raw");
        std::fs::write(&img, b"x").unwrap();
        for breaks in [true, false] {
            for purpose in [Purpose::Open, Purpose::Save] {
                assert_eq!(start_dir_in(&img.to_string_lossy(), Some(managed.clone())), managed);
                let _ = name_to_prefill(&img.to_string_lossy(), purpose, breaks);
            }
        }
        std::fs::remove_dir_all(&managed).ok();
    }

    /// The reported case, verbatim: scsi1 pointing at the managed image on a
    /// Mac. Browse must open the folder that holds it.
    ///
    /// The path is the real `~/Library/Application Support` one rather than a
    /// sandbox container path, and it has a space in it — both worth keeping in
    /// the fixture, since either could plausibly have been the culprit and
    /// neither was.
    #[test]
    fn the_reported_mac_case_opens_the_folder_holding_the_image() {
        let home = tmp("maccase");
        let disks = home.join("Library/Application Support/iris/disks");
        std::fs::create_dir_all(&disks).unwrap();
        let img = disks.join("scsi1.raw");
        std::fs::write(&img, b"x").unwrap();

        let value = img.to_string_lossy().into_owned();
        assert_eq!(start_dir_in(&value, Some(disks.clone())), disks,
                   "Browse must open the disks folder, not anywhere else");

        // And on macOS no name may ride along, or AppKit throws that directory
        // away and falls back to Documents — which is the whole bug.
        assert_eq!(name_to_prefill(&value, Purpose::Open, true), None);
        assert_eq!(name_to_prefill(&value, Purpose::Open, false), None);

        // Still right if the image has not been created yet.
        std::fs::remove_file(&img).unwrap();
        assert_eq!(start_dir_in(&value, Some(disks.clone())), disks);

        std::fs::remove_dir_all(&home).ok();
    }

    /// A model of what rfd's AppKit backend does with what we hand it, so the
    /// bug and the fix are demonstrated rather than asserted from a code read.
    ///
    /// Mirrors `set_path` in rfd 0.17's
    /// `backend/macos/file_dialog/panel_ffi.rs`: join the name onto the
    /// directory when both are present, then hand the result to
    /// `setDirectoryURL:` as though it were a directory. This is a copy, so it
    /// cannot notice rfd changing — what it guards is *our* side never feeding
    /// it the combination that breaks.
    fn rfd_macos_directory_url(dir: &Path, file_name: Option<&str>) -> PathBuf {
        match file_name {
            Some(name) if dir.is_dir() => dir.join(name),
            _ => dir.to_path_buf(),
        }
    }

    #[test]
    fn what_we_hand_rfd_survives_its_macos_backend() {
        let disks = tmp("appkit");
        let img = disks.join("scsi1.raw");
        std::fs::write(&img, b"x").unwrap();
        let value = img.to_string_lossy().into_owned();
        let dir = start_dir_in(&value, Some(disks.clone()));

        // What the old code did: directory *and* name. The backend joins them
        // and setDirectoryURL: receives a file, which AppKit discards — the
        // panel then opens at its own default, which is what was reported.
        let broken = rfd_macos_directory_url(&dir, Some("scsi1.raw"));
        assert!(!broken.is_dir(),
                "the joined path is a file, which is exactly why AppKit ignored it: {broken:?}");

        // What we do now on macOS: no name, so nothing is joined and the URL is
        // the folder the image is in.
        let name = name_to_prefill(&value, Purpose::Open, /* macOS */ true);
        let good = rfd_macos_directory_url(&dir, name.as_deref());
        assert_eq!(good, disks);
        assert!(good.is_dir(), "setDirectoryURL: must receive a real directory");

        std::fs::remove_dir_all(&disks).ok();
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
