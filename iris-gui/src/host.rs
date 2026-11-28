//! Host services the egui shell uses to carry out effects.

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

/// Open `path` with the host's default application (a TOML in the user's
/// editor). macOS goes through `NSWorkspace`, which the App Sandbox allows.
pub fn open_path(path: &std::path::Path) {
    #[cfg(target_os = "macos")]
    {
        use objc2::runtime::AnyObject;
        use objc2::{class, msg_send};
        use objc2_foundation::{NSString, NSURL};
        let url = NSURL::fileURLWithPath(&NSString::from_str(&path.to_string_lossy()));
        // SAFETY: the shared NSWorkspace and a valid file URL; returns BOOL.
        unsafe {
            let ws: *mut AnyObject = msg_send![class!(NSWorkspace), sharedWorkspace];
            let _: bool = msg_send![ws, openURL: &*url];
        }
    }
    #[cfg(target_os = "windows")]
    {
        let _ = std::process::Command::new("cmd").args(["/c", "start", ""]).arg(path).spawn();
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let _ = std::process::Command::new("xdg-open").arg(path).spawn();
    }
}

/// Open a web address in the user's browser.
pub fn open_url(url: &str) {
    #[cfg(target_os = "macos")]
    {
        use objc2::runtime::AnyObject;
        use objc2::{class, msg_send};
        use objc2_foundation::{NSString, NSURL};
        let Some(url) = NSURL::URLWithString(&NSString::from_str(url)) else { return };
        // SAFETY: the shared NSWorkspace and a valid URL; returns BOOL.
        unsafe {
            let ws: *mut AnyObject = msg_send![class!(NSWorkspace), sharedWorkspace];
            let _: bool = msg_send![ws, openURL: &*url];
        }
    }
    #[cfg(target_os = "windows")]
    {
        let _ = std::process::Command::new("cmd").args(["/c", "start", "", url]).spawn();
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let _ = std::process::Command::new("xdg-open").arg(url).spawn();
    }
}
