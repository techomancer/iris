use iris::config::{load_config, NfsConfig};
use iris::machine::Machine;
use iris::ui::Ui;
use winit::event_loop::EventLoop;

fn main() {
    let (cfg, scale) = load_config();
    let event_loop = EventLoop::new().unwrap();

    // Start unfsd before the machine so NFS is ready when IRIX boots.
    let nfs_proc = cfg.nfs.as_ref().map(|nfs| start_unfsd(nfs));

    // Machine::new() allocates >1MB on the stack (Physical device_map), which overflows
    // the default stack on Windows (1MB). We spawn a thread with a larger stack to create it.
    let mut machine = std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(move || Box::new(Machine::new(cfg)))
        .unwrap()
        .join()
        .unwrap();
    machine.register_system_controller();
    machine.start();
    std::thread::spawn(|| {
        Machine::run_console_client();
    });

    let ui = Ui::new(machine.get_ps2(), machine.get_rex3(), machine.get_timer_manager(), &event_loop, scale);
    ui.run(event_loop);

    machine.stop();

    // Kill unfsd on exit.
    if let Some(proc) = nfs_proc {
        proc.kill();
    }
}

struct UnfsdProc {
    /// On Windows the Child holds the real process handle; kill() works directly.
    /// On Unix unfsd daemonizes, so Child is the short-lived launcher. We record
    /// the daemon PID from the pidfile and kill that instead.
    #[cfg(windows)]
    child: std::process::Child,
    #[cfg(not(windows))]
    pid_path: std::path::PathBuf,
}

impl UnfsdProc {
    fn kill(self) {
        #[cfg(windows)]
        {
            let mut child = self.child;
            let _ = child.kill();
            let _ = child.wait();
        }
        #[cfg(not(windows))]
        {
            // Read the PID written by unfsd -i, then SIGTERM it.
            // Give the daemon a moment to write the file if it hasn't yet.
            for _ in 0..20 {
                if self.pid_path.exists() { break; }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            if let Ok(s) = std::fs::read_to_string(&self.pid_path) {
                if let Ok(pid) = s.trim().parse::<i32>() {
                    unsafe { libc::kill(pid, libc::SIGTERM); }
                }
            }
            let _ = std::fs::remove_file(&self.pid_path);
        }
    }
}

fn start_unfsd(nfs: &NfsConfig) -> UnfsdProc {
    use std::io::Write as _;

    // NFS requires an absolute path in the exports file.
    let abs_dir = std::fs::canonicalize(&nfs.shared_dir)
        .unwrap_or_else(|e| panic!("NFS shared_dir '{}': {}", nfs.shared_dir, e));
    let exports_path = std::env::temp_dir().join("iris_nfs.exports");
    {
        let mut f = std::fs::File::create(&exports_path)
            .expect("failed to create NFS exports file");
        // Export only to 127.0.0.1 — all VM traffic arrives via NAT from localhost.
        // insecure: NAT uses unprivileged source ports (>1024).
        writeln!(f, "{} 127.0.0.1(rw,insecure)",
                 abs_dir.display()).expect("failed to write exports file");
    }

    let pid_path = std::env::temp_dir().join("iris_nfs.pid");

    let child = std::process::Command::new(&nfs.unfsd)
        .arg("-u")                                       // don't require root
        .arg("-p")                                       // don't register with host portmap
        .arg("-n").arg(nfs.nfs_host_port.to_string())
        .arg("-m").arg(nfs.mountd_host_port.to_string())
        .arg("-l").arg("127.0.0.1")
        .arg("-e").arg(&exports_path)
        .arg("-i").arg(&pid_path)
        .spawn()
        .unwrap_or_else(|e| panic!("failed to start unfsd '{}': {}", nfs.unfsd, e));

    eprintln!("iris: unfsd started (pid {}) nfs=127.0.0.1:{} mountd=127.0.0.1:{} dir={}",
              child.id(), nfs.nfs_host_port, nfs.mountd_host_port, abs_dir.display());

    // On Unix, wait for the launcher to exit (it forks the daemon and quits).
    #[cfg(not(windows))]
    { let mut c = child; let _ = c.wait(); }

    #[cfg(windows)]
    return UnfsdProc { child };

    #[cfg(not(windows))]
    return UnfsdProc { pid_path };
}
