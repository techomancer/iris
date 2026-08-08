use crate::framebuffer::{new_capture_renderer, FrameSink};
use crossbeam_channel::{unbounded, Receiver, Sender};
use iris::config::{MachineConfig, PortForwardConfig};
use iris::machine::Machine;
use iris::ps2::Ps2Controller;
use parking_lot::Mutex;
use std::net::Ipv4Addr;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::JoinHandle;

#[derive(Debug)]
pub enum Cmd {
    Start(Box<MachineConfig>),
    Stop,
    /// Type `halt\n` at the IRIX serial console in-process (no loopback socket)
    /// for a clean guest shutdown.
    HaltIrix,
    /// Move the running NAT onto a new subnet live (CIDR string, e.g.
    /// `192.168.1.0/24`) — no reboot. Ignored if not running / invalid.
    SetNatSubnet(String),
    /// Rebind the running NAT's inbound port-forward listeners from this rule
    /// set, live — no reboot. Ignored if not running.
    SetPortForwards(Vec<PortForwardConfig>),
    /// Reopen the PCAP capture on a different host interface (`None` = auto-pick)
    /// without rebooting the guest. Ignored if not running / not in PCAP mode.
    SetPcapInterface(Option<String>),
    SaveState(String),
    RestoreState(String),
    Screenshot(PathBuf),
    /// Stop the machine and fold any pending CHD `.diff.chd` sidecars back into
    /// their bases ("Synchronizing disks"), emitting `SyncProgress`/`SyncDone`.
    /// Sent on a clean exit when `Status::chd_sync_pending` is set.
    SyncDisks,
    /// Commit a single disk's COW overlay into its base ("apply changes"). File-
    /// level (no machine) — only valid while stopped. `chd` picks the `.diff.chd`
    /// vs raw `.overlay` path. A CHD commit streams `SyncProgress`/`SyncDone`
    /// (it recompresses); a raw commit ends with `CowDone`.
    CowCommit { base: String, chd: bool },
    /// Discard a single disk's COW overlay ("roll back") — delete the
    /// `.diff.chd` / `.overlay`. File-level; only valid while stopped.
    CowReset { base: String, chd: bool },
    /// Load a disc image into a CD-ROM device (live hot-swap).
    /// Valid only while running. The path is loaded as the active disc.
    LoadDisc { id: u8, path: String, remount: bool },
    /// Empty the CD-ROM tray on a running machine (SCSI layer only).
    EjectCdrom { id: u8 },
    /// Inject an IRIX shell command to (re)mount /CDROM for SCSI `id`.
    RemountCdrom { id: u8 },
    Quit,
}

// PowerOff is emitted when iris exposes `Machine::subscribe_events` (still
// pending). The rest are emitted by the worker on the relevant Cmd success
// path; Status is emitted on a periodic tick while a machine is running.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum Evt {
    Started,
    Stopped,
    PowerOff,
    StateSaved(String),
    StateRestored(String),
    Screenshot(PathBuf),
    Error(String),
    Status(Status),
    /// Per-disk progress of a `SyncDisks` run: `disk` of `total` disks, the
    /// current disk `fraction` (0.0..=1.0) through its rebuild.
    SyncProgress { disk: usize, total: usize, fraction: f32 },
    /// `SyncDisks` finished; the app may now close. Carries the disks synced.
    SyncDone(usize),
    /// A `CowCommit` (raw) or `CowReset` finished. `committed` = changes were
    /// applied (vs rolled back / nothing to do).
    CowDone { committed: bool },
}

#[derive(Debug, Clone, Default)]
pub struct Status {
    pub running: bool,
    /// CPU is currently in PROM (not yet booted IRIX, or post-halt).
    pub in_prom: bool,
    /// IRIX has shut down cleanly (PowerOff event observed).
    pub power_off_seen: bool,
    /// Count of dirty COW overlay sectors across all SCSI devices.
    pub dirty_cow: usize,
    /// Approximate instructions/sec (millions).
    pub mips: f32,
    /// The CPU is not executing: either stopped (soft power-off) or idle at the
    /// PROM after an IRIX `halt` (0 MIPS). When set, the guest has shut down and
    /// stopping the machine can't corrupt a disk — see [`crate::safe_stop`].
    pub cpu_halted: bool,
    /// The CPU thread has actually stopped — a soft power-off called
    /// `Machine::stop`. Unlike `cpu_halted` this is NOT set by mere 0-MIPS idle
    /// (PROM prompt, idle desktop), so it's the precise "the machine powered
    /// off" signal the framebuffer overlay uses.
    pub cpu_stopped: bool,
    /// At least one attached CHD has diff-borne changes pending a fold-back into
    /// its base on a clean shutdown — drives the "Synchronizing disks" step.
    pub chd_sync_pending: bool,
    /// Cumulative count of guest Ethernet frames the NAT engine has processed.
    /// Monotonic within a run; the handle watches it advance to light the
    /// internal-network indicator (see [`EmulatorHandle::net_state`]).
    pub net_frames: u64,
    /// The guest's observed source IP (None until a frame reveals one) and the
    /// address NAT expects it to have. Drive the "Check networking" diagnosis.
    pub net_guest_ip: Option<Ipv4Addr>,
    /// The guest's likely default gateway (passively inferred from its ARPs).
    pub net_guest_gateway: Option<Ipv4Addr>,
    /// IRIS's current NAT gateway (reflects any live adoption).
    pub net_nat_gateway: Option<Ipv4Addr>,
    /// Live PCAP capture-backend status. `Inactive` in NAT mode / non-pcap
    /// builds; `PermissionDenied` when a pcap-mode machine couldn't open the raw
    /// capture, which drives the "Enable packet capture" elevation prompt.
    pub pcap_status: iris::net::PcapStatus,
}

/// State of the internal-network ("NET") indicator shown next to MIPS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetState {
    /// Guest isn't executing — stopped, halted, or idle at the PROM (grey).
    Off,
    /// NAT IP traffic has flowed this run — networking is up (green).
    Active,
    /// Running, but no NAT IP traffic seen yet this run (red).
    Idle,
}

/// Pure decision for the NET indicator, factored out so it's unit-testable
/// without a live machine. Grey whenever the guest isn't executing; otherwise
/// green once NAT IP traffic has been seen this run (`net_seen > 0`) — the
/// signal latches, since a guest that has networked doesn't become misconfigured
/// just by going idle — else red.
fn net_state_for(running: bool, halted: bool, net_seen: u64) -> NetState {
    if !running || halted {
        return NetState::Off;
    }
    if net_seen > 0 { NetState::Active } else { NetState::Idle }
}

pub struct EmulatorHandle {
    cmd_tx: Sender<Cmd>,
    evt_rx: Receiver<Evt>,
    thread: Option<JoinHandle<()>>,
    /// Shared latest-framebuffer slot, written by the CaptureRenderer
    /// inside the worker and read by the GUI each egui frame.
    pub frame_sink: FrameSink,
    /// Second Newport head (`graphics.heads == 2`); inactive when seq stays 0.
    pub frame_sink_head1: FrameSink,
    /// Handle to the live machine's PS/2 controller (when running), so
    /// the GUI thread can push keyboard / mouse events at it directly.
    /// `None` when no machine is up.
    pub ps2: Arc<Mutex<Option<Arc<Ps2Controller>>>>,
    pub status: Status,
    /// NAT IP-frame count observed this run (reset on Start). Non-zero once the
    /// guest's networking has actually carried traffic; latches the NET
    /// indicator green for the rest of the run.
    net_seen_frames: u64,
}

impl EmulatorHandle {
    pub fn spawn() -> Self {
        let (cmd_tx, cmd_rx) = unbounded::<Cmd>();
        let (evt_tx, evt_rx) = unbounded::<Evt>();
        let frame_sink = FrameSink::new();
        let frame_sink_head1 = FrameSink::new();
        let sink_for_worker = frame_sink.clone();
        let sink_head1_for_worker = frame_sink_head1.clone();
        let ps2: Arc<Mutex<Option<Arc<Ps2Controller>>>> = Arc::new(Mutex::new(None));
        let ps2_for_worker = ps2.clone();
        let thread = std::thread::Builder::new()
            .name("iris-gui-emu".into())
            // Machine::new alone puts >1 MB on the stack (Physical::device_map),
            // and unlike the CLI — which builds the machine on a minimal,
            // dedicated thread — we call it from inside worker_loop's deeper
            // frame (catch_unwind + loop). With unoptimized debug-sized frames
            // the 8 MB the CLI uses overflows during Rex3::new, so give the
            // worker generous headroom. This is virtual address space, lazily
            // committed, so the large reservation has no real cost.
            .stack_size(64 * 1024 * 1024)
            .spawn(move || worker_loop(cmd_rx, evt_tx, sink_for_worker, sink_head1_for_worker, ps2_for_worker))
            .expect("spawn iris-gui-emu thread");
        Self {
            cmd_tx,
            evt_rx,
            thread: Some(thread),
            frame_sink,
            frame_sink_head1,
            ps2,
            status: Status::default(),
            net_seen_frames: 0,
        }
    }

    pub fn send(&self, cmd: Cmd) {
        let _ = self.cmd_tx.send(cmd);
    }

    /// Drain pending events; return them for the UI to consume.
    pub fn drain_events(&mut self) -> Vec<Evt> {
        let mut out = Vec::new();
        while let Ok(evt) = self.evt_rx.try_recv() {
            if let Evt::Status(s) = &evt {
                // The worker only knows the perf-derived fields; `running`,
                // `power_off_seen` and `in_prom` are driven by lifecycle
                // events, so merge rather than replace to avoid clobbering them.
                self.status.mips = s.mips;
                self.status.dirty_cow = s.dirty_cow;
                self.status.cpu_halted = s.cpu_halted;
                self.status.cpu_stopped = s.cpu_stopped;
                self.status.chd_sync_pending = s.chd_sync_pending;
                // Latch the NET light: once NAT IP traffic has flowed this run
                // the guest's networking is up, so keep it green through idle
                // lulls (it resets to red on the next Start).
                if s.net_frames > self.net_seen_frames {
                    self.net_seen_frames = s.net_frames;
                }
                self.status.net_frames = s.net_frames;
                self.status.net_guest_ip = s.net_guest_ip;
                self.status.net_guest_gateway = s.net_guest_gateway;
                self.status.net_nat_gateway = s.net_nat_gateway;
                self.status.pcap_status = s.pcap_status;
            }
            match &evt {
                Evt::Started => {
                    self.status.running = true;
                    // Clear a stale stop from the previous run so the new boot
                    // isn't dimmed before the first status tick lands.
                    self.status.cpu_stopped = false;
                    // Fresh machine → fresh NAT counter (starts at 0); reset our
                    // tracking so the indicator starts red and only greens on
                    // this run's first observed NAT traffic.
                    self.net_seen_frames = 0;
                }
                Evt::Stopped => self.status.running = false,
                Evt::PowerOff => self.status.power_off_seen = true,
                _ => {}
            }
            out.push(evt);
        }
        out
    }

    pub fn is_running(&self) -> bool { self.status.running }

    /// Whether a clean exit needs a "Synchronizing disks" step (a CHD has
    /// diff-borne changes to fold back into its base). Latest reported status.
    pub fn has_pending_chd_sync(&self) -> bool { self.status.chd_sync_pending }

    /// State of the internal-network indicator: grey when the guest isn't
    /// executing (stopped/halted/PROM), green once NAT IP traffic has flowed
    /// this run, red while a running guest has produced no NAT traffic yet.
    pub fn net_state(&self) -> NetState {
        net_state_for(self.status.running, self.status.cpu_halted, self.net_seen_frames)
    }

    /// Live PCAP capture-backend status, sampled from the running machine.
    /// `Inactive` when no machine is up, in NAT mode, or on a non-pcap build.
    /// The app watches this for `PermissionDenied` to raise the "Enable packet
    /// capture" elevation prompt.
    pub fn pcap_status(&self) -> iris::net::PcapStatus { self.status.pcap_status }

    /// Stop the machine (if running) and join the worker thread. Idempotent.
    /// Call this from the GUI's `on_exit` so a running machine is cleaned up
    /// even when the user closes the window without pressing Stop — and so the
    /// cleanup completes synchronously rather than racing process teardown.
    /// The worker's Quit handler bounds the stop with a timeout, so this can't
    /// hang on a wedged guest.
    pub fn shutdown(&mut self) {
        let _ = self.cmd_tx.send(Cmd::Quit);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for EmulatorHandle {
    // Backstop in case `shutdown()` wasn't called explicitly (e.g. a panic
    // path). No-op once the worker has already been joined.
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn net_indicator_states() {
        // Unpowered (not running) → grey, regardless of past traffic.
        assert_eq!(net_state_for(false, false, 42), NetState::Off);
        // Running but halted / idle at the PROM → grey.
        assert_eq!(net_state_for(true, true, 42), NetState::Off);
        // Running, no NAT traffic seen yet → red.
        assert_eq!(net_state_for(true, false, 0), NetState::Idle);
        // Running, NAT traffic has flowed → green (and latches, so it stays).
        assert_eq!(net_state_for(true, false, 1), NetState::Active);
    }
}

/// Worker thread loop. Owns the `Machine` while it exists. The eframe app
/// thread sends `Cmd`s and drains `Evt`s, never touching the machine
/// directly. All `Machine` calls are wrapped in `catch_unwind` so a panic
/// becomes an `Evt::Error` toast rather than killing the worker.
fn worker_loop(
    cmd_rx: Receiver<Cmd>,
    evt_tx: Sender<Evt>,
    frame_sink: FrameSink,
    frame_sink_head1: FrameSink,
    ps2_slot: Arc<Mutex<Option<Arc<Ps2Controller>>>>,
) {
    let mut machine: Option<Box<Machine>> = None;
    // Live MIPS estimate: read REX3's free-running cycle counter and divide
    // the delta by wall-clock between ticks. Mirrors the status-bar math in
    // src/disp.rs, but driven here since the GUI never runs REX3's own
    // refresh/status-bar loop. `None` until a machine is up.
    let mut cycles: Option<iris::mips_core::CyclesPtr> = None;
    let mut prev_cycles: u64 = 0;
    let mut prev_tick = std::time::Instant::now();
    // Tick cadence for the status poll while idle on the command channel.
    const STATUS_TICK: std::time::Duration = std::time::Duration::from_millis(500);
    loop {
        match cmd_rx.recv_timeout(STATUS_TICK) {
            // Periodic tick (no command pending): refresh the MIPS estimate.
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                if let Some(c) = cycles {
                    let now = std::time::Instant::now();
                    let dt = now.duration_since(prev_tick).as_secs_f64();
                    if dt >= 0.1 {
                        let cur = c.get();
                        let dc = cur.wrapping_sub(prev_cycles);
                        let mips = (dc as f64 / dt / 1_000_000.0 * 10.0).round() as f32 / 10.0;
                        prev_cycles = cur;
                        prev_tick = now;
                        // The guest has shut down when the CPU thread has stopped
                        // (soft power-off calls Machine::stop) or has retired no
                        // instructions this window (halted/idle at the PROM, 0 MIPS).
                        let cpu_stopped = machine.as_ref().map_or(true, |m| !m.cpu_is_running());
                        let cpu_halted = cpu_stopped || mips == 0.0;
                        let chd_sync_pending = machine.as_ref().map_or(false, |m| m.pending_chd_sync_count() > 0);
                        let net_frames = machine.as_ref().map_or(0, |m| m.net_guest_frames());
                        let net_guest_ip = machine.as_ref().and_then(|m| m.net_observed_guest_ip());
                        let net_guest_gateway = machine.as_ref().and_then(|m| m.net_observed_gateway());
                        let net_nat_gateway = machine.as_ref().map(|m| m.nat_expected().1);
                        let pcap_status = machine.as_ref()
                            .map_or(iris::net::PcapStatus::Inactive, |m| m.net_pcap_status());
                        let _ = evt_tx.send(Evt::Status(Status {
                            mips, cpu_halted, cpu_stopped, chd_sync_pending,
                            net_frames, net_guest_ip, net_guest_gateway, net_nat_gateway,
                            pcap_status,
                            ..Status::default()
                        }));
                    }
                }
                continue;
            }
            Ok(Cmd::Start(cfg)) => {
                if machine.is_some() {
                    let _ = evt_tx.send(Evt::Error("emulator already running".into()));
                    continue;
                }
                // Clear the previous run's last frame so the restarted machine
                // shows the "waiting for first REX3 frame" placeholder instead
                // of the stale screen until its first frame is rendered.
                frame_sink.reset();
                frame_sink_head1.reset();
                // Wrap construction in catch_unwind: Machine::new and
                // friends may panic on missing files, bad images, etc.
                // We surface those as Evt::Error toasts instead of
                // silently killing the worker thread.
                //
                // We do NOT force `headless = true` here — iris-gui needs
                // REX3 alive so it can capture the framebuffer. Iris
                // itself never opens a winit window unless `main.rs`
                // calls `Ui::run`; we don't, so there's no event-loop
                // conflict with eframe.
                let cfg_owned = *cfg;
                let sink_for_machine = frame_sink.clone();
                let sink_for_head1 = frame_sink_head1.clone();
                let heads = cfg_owned.graphics.heads;
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut m = Box::new(Machine::new(cfg_owned));
                    m.register_system_controller();
                    // Install the capture renderer before the CPU starts
                    // so the very first REX3 frame already lands in the
                    // sink the GUI can read.
                    if let Some(rex3) = m.get_rex3() {
                        *rex3.renderer.lock() =
                            Some(new_capture_renderer(sink_for_machine));
                    }
                    if heads >= 2 {
                        if let Some(rex3) = m.get_rex3_head1() {
                            *rex3.renderer.lock() =
                                Some(new_capture_renderer(sink_for_head1));
                        }
                    }
                    m.start();
                    m
                }));
                match result {
                    Ok(m) => {
                        // Tell the NAT the host's own networks so it won't adopt a
                        // guest subnet that overlaps the host's real LAN/VPN/Docker.
                        m.set_host_nets(
                            crate::netplan::gather_host_ifaces()
                                .into_iter().map(|h| (h.network, h.prefix)).collect());
                        *ps2_slot.lock() = Some(m.get_ps2());
                        // Latch REX3's cycle counter for the live MIPS estimate.
                        cycles = m.get_rex3().map(|r| r.cycles.get());
                        prev_cycles = cycles.map(|c| c.get()).unwrap_or(0);
                        prev_tick = std::time::Instant::now();
                        machine = Some(m);
                        let _ = evt_tx.send(Evt::Started);
                    }
                    Err(panic) => {
                        let msg = panic_msg(&panic);
                        let _ = evt_tx.send(Evt::Error(format!("start failed: {msg}")));
                    }
                }
            }
            Ok(Cmd::HaltIrix) => {
                match machine.as_ref() {
                    Some(m) => m.inject_serial_console(b"halt\n"),
                    None => { let _ = evt_tx.send(Evt::Error("halt: not running".into())); }
                }
            }
            Ok(Cmd::SetNatSubnet(cidr)) => {
                match machine.as_ref() {
                    Some(m) => match iris::config::parse_nat_subnet(&cidr) {
                        Ok((gateway, client, netmask)) => m.set_nat_subnet(gateway, client, netmask),
                        Err(e) => { let _ = evt_tx.send(Evt::Error(format!("set NAT subnet '{cidr}': {e}"))); }
                    },
                    None => { let _ = evt_tx.send(Evt::Error("set NAT subnet: not running".into())); }
                }
            }
            Ok(Cmd::SetPortForwards(rules)) => {
                if let Some(m) = machine.as_ref() { m.set_port_forwards(rules); }
            }
            Ok(Cmd::SetPcapInterface(iface)) => {
                if let Some(m) = machine.as_ref() { m.set_pcap_interface(iface); }
            }
            Ok(Cmd::Stop) => {
                if let Some(m) = machine.take() {
                    *ps2_slot.lock() = None;
                    cycles = None;
                    // Always report the machine as stopped so the user regains
                    // control, even if the stop failed or had to be abandoned.
                    if let Err(msg) = stop_machine_timed(m) {
                        let _ = evt_tx.send(Evt::Error(msg));
                    }
                    let _ = evt_tx.send(Evt::Stopped);
                } else {
                    let _ = evt_tx.send(Evt::Error("not running".into()));
                }
            }
            Ok(Cmd::SyncDisks) => {
                // Clean-exit disk sync: stop the machine (quiescing disk I/O),
                // then fold each pending CHD diff back into its base, streaming
                // progress so the GUI can show "Synchronizing disks…". The
                // machine is dropped afterwards, exactly like a Stop.
                let mut synced = 0usize;
                if let Some(mut m) = machine.take() {
                    *ps2_slot.lock() = None;
                    cycles = None;
                    m.stop();
                    synced = match m.sync_chd_disks(
                        &mut |disk, total, fraction| {
                            let _ = evt_tx.send(Evt::SyncProgress { disk, total, fraction });
                        },
                        &|| false,
                    ) {
                        Ok(n) => n,
                        Err(e) => {
                            // Don't swallow this. The fold writes a `.synctmp.chd`
                            // beside the base and atomically renames it over the
                            // base — both need write access to the *folder*, which
                            // under the macOS App Sandbox a file-scoped grant (just
                            // picking the disk image) does not convey. Surfaced so
                            // the diff isn't silently left unmerged and the disk
                            // never shrinks.
                            let _ = evt_tx.send(Evt::Error(format!(
                                "couldn't compact CHD disks on exit: {e} — grant the disk's \
                                 folder (File » \"Grant a disk folder…\") so IRIS can write beside it")));
                            0
                        }
                    };
                    // `m` dropped here → fully torn down.
                }
                let _ = evt_tx.send(Evt::Stopped);
                let _ = evt_tx.send(Evt::SyncDone(synced));
            }
            Ok(Cmd::CowCommit { base, chd }) => {
                // File-level commit (the GUI only offers this while stopped, so
                // the disk files are closed). CHD recompresses with progress; raw
                // applies the overlay in place.
                if chd {
                    let diff = iris::chd_disk::diff_path_for(std::path::Path::new(&base));
                    if diff.exists() {
                        let _ = evt_tx.send(Evt::SyncProgress { disk: 0, total: 1, fraction: 0.0 });
                        match iris::chd_disk::flatten_diff(
                            std::path::Path::new(&base),
                            &diff,
                            &mut |f| { let _ = evt_tx.send(Evt::SyncProgress { disk: 0, total: 1, fraction: f }); },
                            &|| false,
                        ) {
                            Ok(()) => { let _ = evt_tx.send(Evt::SyncDone(1)); }
                            Err(e) => {
                                let _ = evt_tx.send(Evt::Error(format!("commit failed: {e}")));
                                let _ = evt_tx.send(Evt::SyncDone(0));
                            }
                        }
                    } else {
                        let _ = evt_tx.send(Evt::CowDone { committed: false });
                    }
                } else {
                    let overlay = format!("{base}.overlay");
                    if std::path::Path::new(&overlay).exists() {
                        match iris::cow_disk::CowDisk::new(&base, &overlay).and_then(|mut c| c.commit()) {
                            Ok(_) => { let _ = evt_tx.send(Evt::CowDone { committed: true }); }
                            Err(e) => { let _ = evt_tx.send(Evt::Error(format!("commit failed: {e}"))); }
                        }
                    } else {
                        let _ = evt_tx.send(Evt::CowDone { committed: false });
                    }
                }
            }
            Ok(Cmd::CowReset { base, chd }) => {
                // Roll back: discard the overlay. File-level; stopped-only.
                let target = if chd {
                    iris::chd_disk::diff_path_for(std::path::Path::new(&base))
                } else {
                    let _ = std::fs::remove_file(format!("{base}.overlay.dirty"));
                    std::path::PathBuf::from(format!("{base}.overlay"))
                };
                match std::fs::remove_file(&target) {
                    Ok(()) => { let _ = evt_tx.send(Evt::CowDone { committed: false }); }
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                        let _ = evt_tx.send(Evt::CowDone { committed: false });
                    }
                    Err(e) => { let _ = evt_tx.send(Evt::Error(format!("roll back failed: {e}"))); }
                }
            }
            Ok(Cmd::SaveState(name)) => {
                let Some(m) = machine.as_mut() else {
                    let _ = evt_tx.send(Evt::Error("save: not running".into()));
                    continue;
                };
                // save_snapshot stops the CPU as part of its work; once it
                // returns, restart the CPU so the user can keep using the
                // machine without an explicit Start.
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let r = m.save_snapshot(&name);
                    m.start();
                    r
                }));
                match result {
                    Ok(Ok(())) => { let _ = evt_tx.send(Evt::StateSaved(name)); }
                    Ok(Err(e)) => { let _ = evt_tx.send(Evt::Error(format!("save '{name}' failed: {e}"))); }
                    Err(p) => { let _ = evt_tx.send(Evt::Error(format!("save panic: {}", panic_msg(&p)))); }
                }
            }
            Ok(Cmd::RestoreState(name)) => {
                let Some(m) = machine.as_mut() else {
                    let _ = evt_tx.send(Evt::Error("restore: not running".into()));
                    continue;
                };
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    m.ci_restore(&name)
                }));
                match result {
                    Ok(Ok(())) => { let _ = evt_tx.send(Evt::StateRestored(name)); }
                    Ok(Err(e)) => { let _ = evt_tx.send(Evt::Error(format!("restore '{name}' failed: {e}"))); }
                    Err(p) => { let _ = evt_tx.send(Evt::Error(format!("restore panic: {}", panic_msg(&p)))); }
                }
            }
            Ok(Cmd::Screenshot(path)) => {
                // Pull the most recently rendered frame from the sink and
                // encode as PNG. We do this in the worker (rather than the
                // GUI thread) because PNG encoding is non-trivial CPU.
                let frame = frame_sink.snapshot();
                if frame.width == 0 || frame.height == 0 {
                    let _ = evt_tx.send(Evt::Error("screenshot: no frame yet".into()));
                    continue;
                }
                match write_png(&path, frame.width as u32, frame.height as u32, &frame.rgba) {
                    Ok(()) => { let _ = evt_tx.send(Evt::Screenshot(path)); }
                    Err(e) => { let _ = evt_tx.send(Evt::Error(format!("screenshot failed: {e}"))); }
                }
            }
            Ok(Cmd::LoadDisc { id, path, remount }) => {
                match machine.as_ref() {
                    Some(m) => {
                        match m.hpc3().scsi().load_disc(id as usize, path.clone()) {
                            Ok(loaded_path) => {
                                let filename = std::path::Path::new(&loaded_path)
                                    .file_name()
                                    .map(|n| n.to_string_lossy().into_owned())
                                    .unwrap_or_else(|| loaded_path.clone());
                                let msg = if remount {
                                    m.remount_cdrom_guest(id);
                                    format!(
                                        "SCSI #{id}: loaded {filename} — remount sent to console"
                                    )
                                } else {
                                    format!("SCSI #{id}: loaded {filename}")
                                };
                                let _ = evt_tx.send(Evt::Error(msg));
                            }
                            Err(e) => {
                                let _ = evt_tx.send(Evt::Error(format!("SCSI #{id}: {e}")));
                            }
                        }
                    }
                    None => { let _ = evt_tx.send(Evt::Error("load disc: not running".into())); }
                }
            }
            Ok(Cmd::EjectCdrom { id }) => {
                match machine.as_ref() {
                    Some(m) => {
                        match m.hpc3().scsi().eject_to_empty(id as usize) {
                            Ok(()) => {
                                let _ = evt_tx.send(Evt::Error(format!("SCSI #{id}: tray empty")));
                            }
                            Err(e) => {
                                let _ = evt_tx.send(Evt::Error(format!("SCSI #{id}: {e}")));
                            }
                        }
                    }
                    None => { let _ = evt_tx.send(Evt::Error("eject: not running".into())); }
                }
            }
            Ok(Cmd::RemountCdrom { id }) => {
                match machine.as_ref() {
                    Some(m) => {
                        m.remount_cdrom_guest(id);
                        let _ = evt_tx.send(Evt::Error(format!(
                            "SCSI #{id}: /CDROM remount sent to console"
                        )));
                    }
                    None => { let _ = evt_tx.send(Evt::Error("remount: not running".into())); }
                }
            }
            Ok(Cmd::Quit) | Err(_) => {
                *ps2_slot.lock() = None;
                if let Some(m) = machine.take() {
                    // Bounded so a wedged guest can't hang app exit (Drop joins
                    // this thread). If the stop is abandoned, the process is
                    // exiting anyway and the OS reclaims everything.
                    let _ = stop_machine_timed(m);
                }
                return;
            }
        }
    }
}

/// Stop a machine, but never block longer than `STOP_TIMEOUT`. `Machine::stop()`
/// starts with `cpu.stop()`, which waits for the CPU thread to acknowledge the
/// halt; a wedged guest can make that never return. We run it on a detached
/// helper thread and give up after the timeout — the helper thread and that
/// `Machine` then leak, but the caller (and the whole GUI) stays responsive.
fn stop_machine_timed(m: Box<Machine>) -> Result<(), String> {
    const STOP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);
    let (done_tx, done_rx) = std::sync::mpsc::channel::<Result<(), String>>();
    if std::thread::Builder::new()
        .name("iris-gui-stop".into())
        .spawn(move || {
            let mut m = m;
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| m.stop()));
            let _ = done_tx.send(r.map_err(|p| panic_msg(&p)));
        })
        .is_err()
    {
        return Err("stop: failed to spawn worker thread".into());
    }
    match done_rx.recv_timeout(STOP_TIMEOUT) {
        Ok(Ok(())) => Ok(()),
        Ok(Err(msg)) => Err(format!("stop failed: {msg}")),
        Err(_) => Err("stop timed out after 5s — the machine appears wedged; abandoning it".into()),
    }
}

fn write_png(path: &std::path::Path, w: u32, h: u32, rgba: &[u8]) -> Result<(), String> {
    use std::fs::File;
    use std::io::BufWriter;
    let file = File::create(path).map_err(|e| e.to_string())?;
    let mut enc = png::Encoder::new(BufWriter::new(file), w, h);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header().map_err(|e| e.to_string())?;
    writer.write_image_data(rgba).map_err(|e| e.to_string())
}

fn panic_msg(p: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = p.downcast_ref::<&'static str>() { return (*s).to_string(); }
    if let Some(s) = p.downcast_ref::<String>()       { return s.clone(); }
    "<non-string panic>".into()
}
