//! macOS App Sandbox security-scoped bookmarks (Mac App Store build).
//!
//! Under the App Sandbox the app may only touch files the user explicitly hands
//! it through an NSOpenPanel/NSSavePanel. That grant lasts only for the current
//! launch: the file paths in each machine TOML (disk images, PROM, ISOs,
//! an NFS export directory, …) are *not* reachable on the next launch.
//!
//! The fix is the standard one: at save time we mint a **security-scoped
//! bookmark** for every file we can currently reach ([`harvest`]) and stash the
//! bytes in [`GuiSettings`](crate::settings::GuiSettings). On the next launch we
//! resolve each bookmark and call `startAccessingSecurityScopedResource`
//! ([`restore`]) before any machine can start.
//!
//! We deliberately never call `stopAccessingSecurityScopedResource`: the
//! emulator needs the backing files for the whole session, so access is held
//! until the process exits. The number of held resources equals the number of
//! attached files (a handful), so the kernel-resource cost is negligible.
//!
//! Everything here is a no-op off macOS and off the `appstore` feature — the
//! regular notarized builds are not sandboxed and open paths directly.

use iris::config::MachineConfig;
use std::collections::BTreeMap;
use std::path::Path;

/// Existing resources, resolved against their own machine folder. Relative
/// paths can reach outside the container via `..` and still need bookmarks.
pub fn config_paths(cfg: &MachineConfig, dir: &Path) -> Vec<String> {
    let mut out = Vec::new();
    let mut cfg = cfg.clone();
    crate::machines::visit_paths(&mut cfg, |p| {
        if p.is_empty() || p == "(embedded)" { return; }
        let path = crate::machines::resolve(dir, p);
        if path.exists() {
            // Bookmarks use a stable absolute key, independent of the selected
            // machine's cwd. Canonicalize only this key, never the TOML path.
            let path = path.canonicalize().unwrap_or(path);
            out.push(path.to_string_lossy().into_owned());
        }
    });
    out
}

/// (Re)create a security-scoped bookmark for every reachable `path` and merge it
/// into `bookmarks`. Paths we can't reach right now (typed by hand, nonexistent,
/// or never user-selected) are left untouched — an existing bookmark is never
/// dropped, so an inactive machine's images survive a save that can't see them.
pub fn harvest<'a>(paths: impl IntoIterator<Item = &'a str>, bookmarks: &mut BTreeMap<String, Vec<u8>>) {
    for path in paths {
        if let Some(bytes) = imp::make_bookmark(path) {
            bookmarks.insert(path.to_string(), bytes);
        }
    }
}

/// Resolve every stored bookmark and begin accessing it for the process
/// lifetime. Call once at startup, after loading settings and before a machine
/// can start. Stale/failed bookmarks are logged and skipped.
pub fn restore(bookmarks: &BTreeMap<String, Vec<u8>>) {
    for (path, bytes) in bookmarks {
        let _ = imp::start_access(path, bytes);
    }
}

/// Reveal `path` in Finder (select it). Uses `NSWorkspace` rather than spawning
/// `open`, so it works under the App Sandbox (which forbids launching helper
/// processes). Best-effort; a missing path just does nothing. Available on all
/// macOS builds (objc2 is a macOS-wide dependency, not gated on `appstore`).
#[cfg(target_os = "macos")]
pub fn reveal_in_finder(path: &str) {
    use objc2::runtime::AnyObject;
    use objc2::{class, msg_send};
    use objc2_foundation::NSString;

    let ns_path = NSString::from_str(path);
    let root = NSString::from_str(""); // empty root → Finder picks the parent
    // SAFETY: `+[NSWorkspace sharedWorkspace]` returns the shared singleton;
    // `-selectFile:inFileViewerRootedAtPath:` takes two NSString* and returns
    // BOOL. AppKit is loaded (this is a GUI app), and both strings are valid.
    unsafe {
        let ws: *mut AnyObject = msg_send![class!(NSWorkspace), sharedWorkspace];
        let _: bool = msg_send![ws, selectFile: &*ns_path, inFileViewerRootedAtPath: &*root];
    }
}

#[cfg(all(target_os = "macos", feature = "appstore"))]
mod imp {
    use objc2_foundation::{
        NSData, NSString, NSURL, NSURLBookmarkCreationOptions, NSURLBookmarkResolutionOptions,
    };

    /// Mint a security-scoped bookmark for `path`, or `None` if we don't
    /// currently have access to it (so the caller keeps any prior bookmark).
    pub fn make_bookmark(path: &str) -> Option<Vec<u8>> {
        let ns_path = NSString::from_str(path);
        let url = NSURL::fileURLWithPath(&ns_path);
        match url.bookmarkDataWithOptions_includingResourceValuesForKeys_relativeToURL_error(
            NSURLBookmarkCreationOptions::WithSecurityScope,
            None,
            None,
        ) {
            Ok(data) => Some(data.to_vec()),
            Err(_) => None,
        }
    }

    /// Resolve `bytes` back to a URL and start accessing it. Returns whether
    /// access was granted. We never stop accessing — the file is needed for the
    /// whole session (see module docs).
    pub fn start_access(path: &str, bytes: &[u8]) -> bool {
        let data = NSData::with_bytes(bytes);
        // SAFETY: `data` is a valid NSData; null is an allowed `is_stale` out-ptr.
        let url = unsafe {
            NSURL::URLByResolvingBookmarkData_options_relativeToURL_bookmarkDataIsStale_error(
                &data,
                NSURLBookmarkResolutionOptions::WithSecurityScope,
                None,
                std::ptr::null_mut(),
            )
        };
        match url {
            // SAFETY: `url` came from resolving a security-scoped bookmark.
            Ok(url) => {
                let ok = unsafe { url.startAccessingSecurityScopedResource() };
                if ok {
                    // The security-scoped access is bound to this NSURL: when the
                    // URL deallocates the access is released, which is why a
                    // freshly-resolved bookmark would read briefly and then fail
                    // deeper in the loader. We intend to hold access for the whole
                    // process (we never call stopAccessing), so leak the URL to
                    // keep it — and the access — alive until exit.
                    std::mem::forget(url);
                }
                ok
            }
            Err(_) => {
                log::warn!("sandbox: could not resolve security-scoped bookmark for {path}");
                false
            }
        }
    }
}

#[cfg(not(all(target_os = "macos", feature = "appstore")))]
mod imp {
    pub fn make_bookmark(_path: &str) -> Option<Vec<u8>> {
        None
    }
    pub fn start_access(_path: &str, _bytes: &[u8]) -> bool {
        false
    }
}

// Runtime verification of the Objective-C bookmark interop. Security-scoped
// bookmarks are created/resolved fine *outside* the sandbox too (they just act
// as ordinary bookmarks), so this exercises the real NSURL round-trip without
// needing a signed/sandboxed build. The sandbox-only behaviour (access denied
// without a bookmark) still has to be verified on a signed App Store build.
#[cfg(all(test, target_os = "macos", feature = "appstore"))]
mod tests {
    use super::*;

    fn temp_file(suffix: &str) -> String {
        let p = std::env::temp_dir().join(format!(
            "iris_sbtest_{}_{}{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            suffix
        ));
        std::fs::write(&p, b"\0\0\0\0").unwrap();
        p.to_string_lossy().into_owned()
    }

    #[test]
    fn bookmark_roundtrip() {
        let path = temp_file(".bin");
        let bytes = imp::make_bookmark(&path).expect("bookmark an accessible file");
        assert!(!bytes.is_empty(), "bookmark bytes should be non-empty");
        assert!(imp::start_access(&path, &bytes), "resolving a fresh bookmark should grant access");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn make_bookmark_missing_file_is_none() {
        assert!(imp::make_bookmark("/nonexistent/iris/does/not/exist.raw").is_none());
    }

    #[test]
    fn config_paths_harvest_restore() {
        let path = temp_file(".raw");
        let mut cfg = MachineConfig::default();
        cfg.scsi.get_mut(&1).unwrap().path = path.clone();
        cfg.prom = "prom.bin".into(); // missing → skipped

        let paths = config_paths(&cfg, Path::new("/tmp/iris-machine"));
        let key = Path::new(&path).canonicalize().unwrap().to_string_lossy().into_owned();
        assert!(paths.contains(&key), "absolute scsi image should be collected");
        assert!(!paths.iter().any(|p| p == "prom.bin"), "relative paths should be skipped");

        let mut bm = BTreeMap::new();
        harvest(paths.iter().map(String::as_str), &mut bm);
        assert!(bm.contains_key(&key), "accessible image should be bookmarked");

        restore(&bm); // must not panic
        let _ = std::fs::remove_file(&path);
    }
}
