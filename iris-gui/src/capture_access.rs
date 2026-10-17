//! Cross-platform "enable packet capture" helper for the PCAP networking backend.
//!
//! Opening a raw libpcap capture needs privilege the GUI lacks by default: root /
//! `CAP_NET_RAW` on Linux, group access to `/dev/bpf*` on macOS, and a
//! WinPcap-compatible driver (+ Administrator) on Windows. This module is the
//! single entry point the UI calls to obtain that permission, dispatching to the
//! platform's native mechanism.
//!
//! The detailed per-OS flows land incrementally — Linux pkexec/setcap (task A2),
//! macOS ChmodBPF admin install (task A4), Windows Npcap (task A7, mostly in the
//! installer). This file defines the stable surface they plug into; until a flow
//! is wired up, [`enable_packet_capture`] returns the manual steps so the button
//! is still useful.

/// Result of an [`enable_packet_capture`] attempt.
#[allow(dead_code)] // `Enabled` / `NeedsRelaunch` are produced once A2/A4 land.
pub enum EnableOutcome {
    /// Capture is (or should now be) permitted; the user can use PCAP mode.
    Enabled,
    /// The privileged step succeeded but the user must **quit & reopen IRIS** for
    /// it to take effect (e.g. macOS group membership only applies to new login
    /// sessions). Carries a message to show.
    NeedsRelaunch(String),
    /// The user cancelled the OS privilege prompt — no change, no error needed.
    Cancelled,
    /// Couldn't enable; carries a human-readable reason / manual next step.
    Failed(String),
}

/// Attempt to grant this build permission to open a raw PCAP capture using the
/// platform's native mechanism. Safe to call from the GUI thread; may block on a
/// native admin prompt while it's open.
pub fn enable_packet_capture() -> EnableOutcome {
    imp::enable()
}

/// One-line, platform-specific hint shown next to the PCAP warning, describing
/// how capture permission is obtained on this OS.
pub fn permission_hint() -> String {
    imp::hint()
}

#[cfg(target_os = "linux")]
mod imp {
    //! Linux capture elevation.
    //!
    //! libpcap opens the capture inside the process, and there's no way to inject
    //! a pre-opened fd, so the whole process must be privileged. Two paths:
    //!   * **setcap** (the package default, applied by the deb/rpm postinst —
    //!     task A6): `cap_net_raw,cap_net_admin+eip` on the binary → capture works
    //!     unprivileged, no prompt, and this `enable()` is never even reached.
    //!   * **pkexec re-exec** (this function — the portable / AppImage fallback):
    //!     relaunch the whole process elevated. Ported from rusty-backup
    //!     `src/os/linux.rs`. Note: pkexec **replaces** this process, so a
    //!     cancelled auth dialog ends the app — that's why setcap is preferred for
    //!     installed packages.

    use super::EnableOutcome;
    use std::os::unix::fs::MetadataExt;
    use std::os::unix::process::CommandExt;
    use std::path::PathBuf;
    use std::process::Command;

    pub fn hint() -> String {
        t!("capture-hint-linux")
    }

    pub fn enable() -> EnableOutcome {
        // On success `exec` replaces this process and never returns; any return is
        // an error (pkexec missing, etc.). A cancelled auth dialog terminates the
        // (already-replaced) process, so there is no "Cancelled" return here.
        EnableOutcome::Failed(relaunch_with_elevation())
    }

    /// Re-exec the whole process under `pkexec`, re-injecting the GUI/session env
    /// pkexec strips. Returns only on failure (the returned String is the reason);
    /// on success `exec` has already replaced this process.
    fn relaunch_with_elevation() -> String {
        // Inside an AppImage, current_exe() is the per-user FUSE mount under
        // /tmp/.mount_* which root can't read; APPIMAGE points at the real file,
        // and elevating that re-bootstraps the squashfs as root.
        let target = match std::env::var_os("APPIMAGE") {
            Some(p) => PathBuf::from(p),
            None => match std::env::current_exe() {
                Ok(p) => p,
                Err(e) => return t!("capture-error-no-exe", error = e.to_string()),
            },
        };
        let args: Vec<String> = std::env::args().skip(1).collect();

        if which_pkexec().is_none() {
            return t!("capture-error-no-pkexec", path = target.display());
        }

        // pkexec strips the environment: re-inject display + identity vars so the
        // elevated process can reach X11/Wayland and resolve the real user's home.
        let mut env_args: Vec<String> = Vec::new();
        for var in [
            "DISPLAY", "WAYLAND_DISPLAY", "WAYLAND_SOCKET", "XAUTHORITY",
            "XDG_RUNTIME_DIR", "HOME", "APPIMAGE", "ARGV0",
        ] {
            if let Ok(val) = std::env::var(var) {
                env_args.push(format!("{var}={val}"));
            }
        }
        if let Ok(user) = std::env::var("USER") {
            env_args.push(format!("SUDO_USER={user}"));
        }
        // /proc/self is owned by the process's real uid/gid — read them without a
        // libc dependency so the elevated process can recover the original user.
        if let Ok(meta) = std::fs::metadata("/proc/self") {
            env_args.push(format!("SUDO_UID={}", meta.uid()));
            env_args.push(format!("SUDO_GID={}", meta.gid()));
        }

        // `exec` replaces the current process image; returns only on failure.
        let err = Command::new("pkexec")
            .arg("env")
            .args(&env_args)
            .arg(&target)
            .args(&args)
            .exec();
        t!("capture-error-pkexec", error = err.to_string())
    }

    fn which_pkexec() -> Option<PathBuf> {
        let path = std::env::var_os("PATH")?;
        std::env::split_paths(&path)
            .map(|d| d.join("pkexec"))
            .find(|p| p.is_file())
    }
}

#[cfg(target_os = "macos")]
mod imp {
    //! macOS ChmodBPF install flow (Wireshark's model).
    //!
    //! A one-time admin step installs a LaunchDaemon that makes `/dev/bpf*`
    //! group-readable by the `access_bpf` group and adds the user to it; the
    //! **unprivileged** IRIS + the libpcap it links then open the capture
    //! normally. The emulator never runs as root.
    //!
    //! The daemon script + plist are embedded at build time (so this works in a
    //! plain dev build, not just a packaged .app), staged to a temp dir, and
    //! moved into place by a single privileged shell script run via one native
    //! `osascript … with administrator privileges` prompt.

    use super::EnableOutcome;
    use std::os::unix::fs::PermissionsExt;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    pub fn hint() -> String {
        t!("capture-hint-macos")
    }

    /// The daemon helper resources, baked into the binary from the repo so the
    /// flow needs no .app bundle layout at runtime.
    const CHMODBPF_SCRIPT: &str = include_str!("../../installer/macos/chmod-bpf/ChmodBPF");
    const CHMODBPF_PLIST: &str =
        include_str!("../../installer/macos/chmod-bpf/io.techomancer.iris.ChmodBPF.plist");

    const PLIST_NAME: &str = "io.techomancer.iris.ChmodBPF.plist";
    const DAEMON_PLIST: &str = "/Library/LaunchDaemons/io.techomancer.iris.ChmodBPF.plist";
    const SUPPORT_DIR: &str = "/Library/Application Support/IRIS/ChmodBPF";

    pub fn enable() -> EnableOutcome {
        // Already permitted (e.g. the daemon ran and we relaunched) → no prompt.
        if bpf_accessible() {
            return EnableOutcome::Enabled;
        }
        let user = match std::env::var("USER") {
            Ok(u) if !u.is_empty() && u != "root" => u,
            _ => return EnableOutcome::Failed(t!("capture-error-no-user")),
        };
        let staged = match stage_resources() {
            Ok(s) => s,
            Err(e) => return EnableOutcome::Failed(t!("capture-error-stage", error = e.to_string())),
        };
        let script_path = staged.dir.join("install.sh");
        if let Err(e) = std::fs::write(&script_path, install_script(&staged, &user)) {
            return EnableOutcome::Failed(t!("capture-error-script", error = e.to_string()));
        }
        match run_admin_shell(&script_path) {
            // The new group membership only applies to processes started afterwards.
            AdminResult::Ok => EnableOutcome::NeedsRelaunch(t!("capture-enabled-relaunch")),
            AdminResult::Cancelled => EnableOutcome::Cancelled,
            AdminResult::Failed(e) => EnableOutcome::Failed(t!("capture-error-install", error = e)),
        }
    }

    /// Whether this process can already open a BPF capture device. Probes
    /// `/dev/bpf0..bpf15`: a successful open or `EBUSY` (in use, but reachable)
    /// means permissions are fine; `EACCES`/`EPERM` means they aren't. `ENOENT`
    /// just means that node doesn't exist yet — keep probing.
    fn bpf_accessible() -> bool {
        use std::io::ErrorKind;
        for n in 0..16 {
            match std::fs::OpenOptions::new().read(true).write(true).open(format!("/dev/bpf{n}")) {
                Ok(_) => return true,
                Err(e) => match e.kind() {
                    ErrorKind::PermissionDenied => return false,
                    // EBUSY (16 on Darwin): node exists and we may open it, it's
                    // just in use → permissions are fine, treat as reachable.
                    _ if e.raw_os_error() == Some(16) => return true,
                    _ => continue, // ENOENT / other → try the next node
                },
            }
        }
        false
    }

    struct Staged {
        dir: PathBuf,
        script: PathBuf,
        plist: PathBuf,
    }

    /// Write the embedded ChmodBPF script + plist to a private temp dir.
    fn stage_resources() -> std::io::Result<Staged> {
        let dir = std::env::temp_dir().join(format!("iris-chmodbpf-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let script = dir.join("ChmodBPF");
        let plist = dir.join(PLIST_NAME);
        std::fs::write(&script, CHMODBPF_SCRIPT)?;
        std::fs::write(&plist, CHMODBPF_PLIST)?;
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755))?;
        Ok(Staged { dir, script, plist })
    }

    /// The privileged `/bin/sh` script: install the daemon, create+join the
    /// `access_bpf` group, apply perms to current nodes, and (re)load the daemon.
    /// `set -e` aborts on a hard failure; tolerant steps are guarded with `|| true`.
    fn install_script(s: &Staged, user: &str) -> String {
        // Paths are from std::env::temp_dir() (no quotes); single-quote them anyway.
        let sh = |p: &Path| format!("'{}'", p.display());
        format!(
            "#!/bin/sh\n\
             set -e\n\
             mkdir -p '{support}'\n\
             cp {staged_script} '{support}/ChmodBPF'\n\
             chown root:wheel '{support}/ChmodBPF'\n\
             chmod 755 '{support}/ChmodBPF'\n\
             cp {staged_plist} '{daemon}'\n\
             chown root:wheel '{daemon}'\n\
             chmod 644 '{daemon}'\n\
             dseditgroup -o create access_bpf 2>/dev/null || true\n\
             dseditgroup -o edit -a '{user}' -t user access_bpf\n\
             chgrp access_bpf /dev/bpf* 2>/dev/null || true\n\
             chmod g+rw /dev/bpf* 2>/dev/null || true\n\
             launchctl bootout system '{daemon}' 2>/dev/null || true\n\
             launchctl bootstrap system '{daemon}'\n",
            support = SUPPORT_DIR,
            daemon = DAEMON_PLIST,
            staged_script = sh(&s.script),
            staged_plist = sh(&s.plist),
            user = user,
        )
    }

    enum AdminResult {
        Ok,
        Cancelled,
        Failed(String),
    }

    /// Run `script_path` as root behind one native admin prompt via
    /// `osascript … "do shell script … with administrator privileges"`.
    fn run_admin_shell(script_path: &Path) -> AdminResult {
        // Only the script path is interpolated into the AppleScript string; it
        // comes from temp_dir() so it has no quotes to escape.
        let apple = format!(
            "do shell script \"/bin/sh '{}'\" with administrator privileges",
            script_path.display()
        );
        match Command::new("osascript").arg("-e").arg(&apple).output() {
            Ok(out) if out.status.success() => AdminResult::Ok,
            Ok(out) => {
                let err = String::from_utf8_lossy(&out.stderr);
                // osascript reports a user cancel as "User canceled. (-128)".
                if err.contains("-128") || err.to_lowercase().contains("cancel") {
                    AdminResult::Cancelled
                } else {
                    AdminResult::Failed(err.trim().to_string())
                }
            }
            Err(e) => AdminResult::Failed(t!("capture-error-osascript", error = e.to_string())),
        }
    }
}

#[cfg(target_os = "windows")]
mod imp {
    use super::EnableOutcome;
    use std::process::Command;

    pub fn hint() -> String {
        t!("capture-hint-windows")
    }

    /// Whether a WinPcap-compatible driver (Npcap, or legacy WinPcap) is present.
    fn npcap_present() -> bool {
        let root = std::env::var("SystemRoot").unwrap_or_else(|_| "C:\\Windows".into());
        let p = std::path::Path::new(&root);
        p.join("System32\\Npcap\\wpcap.dll").exists() || p.join("System32\\wpcap.dll").exists()
    }

    pub fn enable() -> EnableOutcome {
        // Check first: if the driver is already present, the only remaining
        // requirement is Administrator.
        if npcap_present() {
            return EnableOutcome::NeedsRelaunch(t!("capture-npcap-installed"));
        }
        // We never bundle or silently download the driver. Open the official page
        // so the user installs Npcap themselves, then relaunches and tries again.
        let _ = Command::new("explorer")
            .arg("https://npcap.com/#download")
            .spawn();
        EnableOutcome::NeedsRelaunch(t!("capture-npcap-download"))
    }
}

// Fallback for any other target so the crate still builds.
#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
mod imp {
    use super::EnableOutcome;
    pub fn hint() -> String {
        t!("capture-hint-other")
    }
    pub fn enable() -> EnableOutcome {
        EnableOutcome::Failed(t!("capture-unsupported"))
    }
}
