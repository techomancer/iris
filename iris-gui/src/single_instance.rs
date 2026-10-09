//! Single-instance guard.
//!
//! A hard crash (abort/SIGKILL) skips `on_exit`/`Drop`, but the kernel still
//! reaps the dead process and frees its TCP ports — so a *crashed* instance
//! needs no cleanup. The case that genuinely lingers is a previous `iris-gui`
//! that is still **alive** (hung, or a forgotten copy): it keeps the monitor /
//! serial ports (8888 / 8880 / 8881) bound, which is what produced the early
//! "AddrInUse" failures.
//!
//! On startup we therefore terminate any still-alive previous instance to
//! reclaim those ports, then record our own PID. The pidfile is removed on
//! clean exit; a stale one left by a crash is harmless (its PID is dead, so we
//! skip it and overwrite). Unix only — a no-op elsewhere.

use std::path::PathBuf;

fn pidfile_path() -> Option<PathBuf> {
    crate::settings::GuiSettings::data_dir().map(|d| d.join("iris-gui.pid"))
}

/// Reclaim resources from a previous instance (if any), then claim the lock
/// for this process. Call once at startup, before launching the emulator.
pub fn acquire() {
    let Some(path) = pidfile_path() else { return };

    if let Ok(contents) = std::fs::read_to_string(&path) {
        if let Ok(pid) = contents.trim().parse::<i32>() {
            #[cfg(unix)]
            reclaim_previous(pid);
            #[cfg(not(unix))]
            let _ = pid;
        }
    }

    // Record our PID (best effort). Done after reclaim_previous() has confirmed
    // the old process is gone, so it can't remove the file we just wrote.
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(&path, std::process::id().to_string());
}

/// Remove the pidfile on clean exit.
pub fn release() {
    if let Some(path) = pidfile_path() {
        let _ = std::fs::remove_file(path);
    }
}

#[cfg(unix)]
fn reclaim_previous(pid: i32) {
    use std::process::Command;
    use std::time::Duration;

    let me = std::process::id() as i32;
    if pid <= 1 || pid == me {
        return;
    }

    // Only act if the PID is alive AND is actually an iris-gui — the OS may
    // have recycled a dead PID to an unrelated program we must not kill.
    let comm = Command::new("ps")
        .args(["-p", &pid.to_string(), "-o", "comm="])
        .output();
    let is_iris = matches!(&comm, Ok(o) if o.status.success()
        && String::from_utf8_lossy(&o.stdout).contains("iris-gui"));
    if !is_iris {
        return;
    }

    log::warn!("terminating previous iris-gui instance (pid {pid}) to reclaim its ports");
    let _ = Command::new("kill").arg(pid.to_string()).status(); // SIGTERM

    // Wait up to ~2s for it to exit (and release its ports) before escalating.
    let mut alive = true;
    for _ in 0..20 {
        std::thread::sleep(Duration::from_millis(100));
        let still = Command::new("kill").args(["-0", &pid.to_string()]).status();
        if matches!(still, Ok(s) if !s.success()) {
            alive = false;
            break;
        }
    }
    if alive {
        let _ = Command::new("kill").args(["-9", &pid.to_string()]).status(); // SIGKILL
        // Brief grace for the kernel to tear the process down and free ports.
        std::thread::sleep(Duration::from_millis(200));
    }
}
