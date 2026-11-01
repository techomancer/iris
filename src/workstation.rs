//! What the front ends need from an emulated machine, whichever machine it is.
//!
//! `main.rs`, the window (`ui.rs`), the CI socket (`ci.rs`), the monitor's
//! system commands and iris-gui hold a `Box<dyn Workstation>` (or reach one
//! through a [`MachineSlot`]) rather than a concrete machine type. Today the
//! one implementation is [`Machine`], the Indy and Indigo2 family; a machine
//! built around a different chipset implements the same trait.
//!
//! The trait splits in two. The required methods are what every machine has:
//! starting and stopping, the screen, the keyboard and mouse, the serial
//! console, the monitor. The provided methods are optional facilities
//! (snapshots, CHD folding, the scratch volume, swappable discs, the NAT
//! controls) and default to "not on this machine", so a new machine can come
//! up with the essentials and grow the rest.

use std::any::Any;
use std::io::Write;
use std::net::Ipv4Addr;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use parking_lot::Mutex;

use crate::config::MachineConfig;
use crate::gdb_stub::CpuDebug;
use crate::dev::ps2::Ps2Controller;
use crate::dev::serial_host::CiSerialBackend;
use crate::gfx_display::GfxDisplay;
use crate::hptimer::TimerManager;
use crate::machine::Machine;
use crate::machine_state::{GuestState, MachineStatus, TransitionAck};
use crate::monitor::Monitor;
use crate::net::NatControl;
use crate::traits::{Device, MachineEvent};

/// Build the machine `cfg` describes, or say why it cannot be built.
///
/// Construction can put more than a megabyte on the stack, so call this from
/// a thread with a large one, as `main.rs` and iris-gui do.
pub fn build(cfg: MachineConfig) -> Result<Box<dyn Workstation>, String> {
    crate::machine::set_emulator_profile(cfg.machine.profile);
    match cfg.machine.profile {
        // The O2 is its own machine (CRIME, MACE, GBE), not an MC/HPC3 one.
        crate::config::MachineProfile::O2Ip32 => Ok(Box::new(crate::ip32_machine::O2Machine::new(&cfg)?)),
        // So is the Octane (HEART, Xbow, Bridge).
        crate::config::MachineProfile::OctaneIp30 => Ok(Box::new(crate::ip30_machine::OctaneMachine::new(&cfg)?)),
        _ => Ok(Box::new(Machine::new(cfg))),
    }
}

/// An emulated machine as the front ends see it. See the module header.
pub trait Workstation: Send {
    // ---- lifecycle ----

    /// Start the peripherals, the monitor and (outside CI mode) the CPU.
    fn start(&mut self);
    /// Stop the CPU and every peripheral thread.
    fn stop(&mut self);
    /// Put the hardware back in its power-on state.
    fn reset(&mut self);
    /// Restart the peripheral threads after `stop`, leaving the CPU stopped.
    fn restart_peripherals(&mut self);
    /// Start the CPU thread.
    fn cpu_start(&self);
    /// Whether the CPU thread is running. False after a guest power-off.
    fn cpu_is_running(&self) -> bool;

    /// The shared, nullable pointer slot for this machine (see
    /// [`MachineSlot`]), created on first use. Call it only once the machine
    /// is in its final place (boxed): the slot records this address.
    fn machine_slot(&mut self) -> MachineSlot;
    /// Register the monitor's system commands (`reset`, `save`, `load`, ...)
    /// and start the thread that carries out the devices' machine events.
    /// Call once, after the machine is boxed.
    fn register_system_controller(&mut self);
    /// The address the machine's monitor serves on, or None when it could not
    /// bind its port or has not started.
    fn monitor_addr(&self) -> Option<std::net::SocketAddr>;

    /// Where the machine is in its start/stop lifecycle, as it stores it.
    /// Never `Running`: [`Workstation::status`] derives that from the CPU's
    /// own running flag, so an execution-control command cannot leave it
    /// stale. `start` moves it to `Starting` before the monitor binds and
    /// `stop` to `Halted` (unless it was never started). See
    /// [`crate::machine_state`].
    fn lifecycle(&self) -> GuestState;
    fn set_lifecycle(&mut self, state: GuestState);

    /// The guest's current state and whether the monitor is accepting
    /// commands. Every surface that reports state — `machine-status`, the
    /// `machine-start` acknowledgement, and the CI socket's `status` — goes
    /// through here, so they cannot disagree.
    fn status(&self) -> MachineStatus {
        let cpu_running = self.cpu_is_running();
        MachineStatus {
            // Running is read from the CPU, not stored: `start`/`stop` and
            // breakpoints change the CPU's flag without a machine-level
            // transition, and a stale lifecycle copy would be a second,
            // wrong answer.
            state: if cpu_running { GuestState::Running } else { self.lifecycle() },
            cpu_running,
            monitor_addr: self.monitor_addr(),
        }
    }

    /// Start the CPU and peripherals on behalf of `machine-start` and the CI
    /// socket's `start`.
    ///
    /// Idempotent: a running machine is left alone and the acknowledgement says
    /// so. A machine that is `Starting` only needs the CPU released; an `Idle`
    /// or `Halted` one also restarts its peripherals. The returned
    /// [`TransitionAck`] is what the monitor prints.
    fn start_guest(&mut self) -> Result<TransitionAck, String> {
        let from = self.status().state;
        if from == GuestState::Running {
            return Ok(TransitionAck { from, via: None, to: from });
        }

        let via = if from == GuestState::Starting {
            None
        } else {
            let starting = self
                .lifecycle()
                .transition(GuestState::Starting)
                .map_err(|e| e.to_string())?;
            self.set_lifecycle(starting);
            Some(GuestState::Starting)
        };
        // Peripherals are idempotently startable; do it even from `Starting`
        // in case a previous stop left one of them down.
        self.restart_peripherals();
        self.cpu_start();
        let to = self.status().state;
        Ok(TransitionAck { from, via, to })
    }

    /// Stop the CPU and peripherals on behalf of `machine-stop`.
    ///
    /// Idempotent: stopping a halted machine changes nothing. Returns the
    /// transition for the monitor's acknowledgement.
    fn stop_guest(&mut self) -> TransitionAck {
        let from = self.status().state;
        self.stop();
        let to = self.status().state;
        TransitionAck { from, via: None, to }
    }

    // ---- host interfaces ----

    /// The display of the first graphics head, or None when headless.
    fn get_display(&self) -> Option<Arc<dyn GfxDisplay>>;
    /// The second head's display, on machines that can have one.
    fn get_display_head1(&self) -> Option<Arc<dyn GfxDisplay>> {
        None
    }
    /// The keyboard and mouse the host window types into.
    fn get_ps2(&self) -> Arc<Ps2Controller>;
    /// Add a host-side command set (the screen, pointer and key commands)
    /// to this machine's monitor.
    fn register_monitor_device(&self, device: Arc<dyn crate::traits::Device>);
    fn get_timer_manager(&self) -> Arc<TimerManager>;
    /// The CPU's debug interface, for the GDB stub.
    fn get_cpu_debug(&self) -> Arc<dyn CpuDebug>;
    /// The kernel's clock-tick counter: its rate over wall time is the guest's
    /// Hz in the status bars.
    fn fasttick_count(&self) -> Arc<AtomicU64>;
    /// Every CPU's retired-cycle counter and clock-tick counter, CPU 0 first,
    /// for the front-end's MIPS and Hz. A machine with one CPU needs no more
    /// than the default.
    fn cpu_counters(&self) -> Vec<(crate::cpu::mips_core::CyclesPtr, Arc<AtomicU64>)> {
        self.get_display().map(|d| (d.cycles(), self.fasttick_count())).into_iter().collect()
    }

    /// The in-process serial console backend installed in CI mode.
    fn get_ci_serial(&self) -> Option<Arc<CiSerialBackend>>;
    /// Type bytes at the guest's serial console, in-process.
    fn inject_serial_console(&self, bytes: &[u8]);

    /// For code that only works on one machine type (the JIT validators):
    /// downcast with `as_any_mut().downcast_mut::<Machine>()`.
    fn as_any_mut(&mut self) -> &mut dyn Any;

    // ---- optional facilities ----

    /// Load an ELF into RAM before the CPU starts (`--load-elf`).
    fn load_elf(&self, _path: &str) -> Result<String, String> {
        Err(NOT_HERE.into())
    }
    fn save_snapshot(&mut self, _name: &str) -> Result<(), String> {
        Err(NOT_HERE.into())
    }
    fn load_snapshot(&mut self, _name: &str) -> Result<(), String> {
        Err(NOT_HERE.into())
    }
    /// The CI socket's `restore`: load a snapshot and keep what a later
    /// `rollback` needs to return to it quickly.
    fn ci_restore(&mut self, name: &str) -> Result<(), String> {
        self.load_snapshot(name)
    }
    /// The CI socket's `rollback` to the last `ci_restore`.
    fn ci_rollback(&mut self) -> Result<(), String> {
        Err(NOT_HERE.into())
    }

    /// Write the PROM's non-volatile settings to `path`, or to the configured
    /// file when None. Returns the path written.
    fn save_nvram(&self, _path: Option<&str>) -> Result<String, String> {
        Err(NOT_HERE.into())
    }
    /// The configured scratch SCSI volume, if any (see `STORAGE.md`).
    fn scratch_path(&self) -> Option<&std::path::Path> {
        None
    }
    /// The controller whose CD-ROM drives take disc changes.
    fn disc_changer(&self) -> Option<Arc<dyn DiscChanger>> {
        None
    }
    /// Ask the guest to remount `/CDROM` after a disc change on `scsi_id`.
    fn remount_cdrom_guest(&self, _scsi_id: u8) {}

    /// Attached CHD disks with changes waiting to be folded into their bases.
    fn pending_chd_sync_count(&self) -> usize {
        0
    }
    /// Make every disk's accepted writes survive the process, including a
    /// copy-on-write overlay's record of the sectors it holds. Run it before
    /// the process exits without dropping the machine.
    fn flush_disks(&self) -> std::io::Result<()>;
    /// Fold pending CHD diffs into their bases (see `Machine::sync_chd_disks`).
    fn sync_chd_disks(
        &self,
        _only: Option<usize>,
        _progress: &mut dyn FnMut(usize, usize, f32),
        _cancel: &dyn Fn() -> bool,
    ) -> std::io::Result<usize> {
        Ok(0)
    }

    /// The count of guest frames every network backend feeds, for the GUI's
    /// network indicator.
    fn net_frames(&self) -> Option<Arc<std::sync::atomic::AtomicU64>> {
        None
    }
    /// The NAT engine's controls, when the machine's Ethernet runs on it.
    fn nat_control(&self) -> Option<Arc<NatControl>> {
        None
    }
    /// The addresses the NAT hands the guest: (client, gateway, netmask).
    fn nat_expected(&self) -> Option<(Ipv4Addr, Ipv4Addr, Ipv4Addr)> {
        None
    }
}

/// The error an optional facility returns on a machine that lacks it.
const NOT_HERE: &str = "not supported on this machine";

impl dyn Workstation + '_ {
    /// Briefly stop the machine, run `work`, then restart the peripherals, and
    /// the CPU only if it was running before. The CI scratch commands use it
    /// to change the scratch file without racing the disk's in-flight reads;
    /// a file injected before boot stays injected without starting the CPU.
    pub fn with_paused<R>(&mut self, work: impl FnOnce() -> R) -> R {
        let was_running = self.cpu_is_running();
        self.stop();
        let r = work();
        self.restart_peripherals();
        if was_running {
            self.cpu_start();
        }
        r
    }

    // The NAT controls by name, for embedders. Each is a no-op (or reports
    // nothing) on a machine without a NAT.

    /// Guest-originated Ethernet frames the network backend has processed so far.
    pub fn net_guest_frames(&self) -> u64 {
        self.net_frames().map_or(0, |f| f.load(std::sync::atomic::Ordering::Relaxed))
    }
    /// The guest's own source IP as last seen on the wire.
    pub fn net_observed_guest_ip(&self) -> Option<Ipv4Addr> {
        self.nat_control().and_then(|n| n.observed_guest_ip())
    }
    /// The gateway the guest keeps ARP-ing for, inferred passively.
    pub fn net_observed_gateway(&self) -> Option<Ipv4Addr> {
        self.nat_control().and_then(|n| n.observed_gateway())
    }
    /// Live status of the PCAP bridged-capture backend.
    pub fn net_pcap_status(&self) -> crate::net::PcapStatus {
        self.nat_control().map_or(crate::net::PcapStatus::Inactive, |n| n.pcap_status())
    }
    /// Move the running NAT onto a new subnet without a reboot.
    pub fn set_nat_subnet(&self, gateway: Ipv4Addr, client: Ipv4Addr, netmask: Ipv4Addr) {
        if let Some(n) = self.nat_control() {
            n.request_subnet(gateway, client, netmask);
        }
    }
    /// The host's own networks, which the NAT must not shadow.
    pub fn set_host_nets(&self, nets: Vec<(Ipv4Addr, u8)>) {
        if let Some(n) = self.nat_control() {
            n.set_host_nets(nets);
        }
    }
    /// Replace the inbound port-forward rules without a reboot.
    pub fn set_port_forwards(&self, rules: Vec<crate::config::PortForwardConfig>) {
        if let Some(n) = self.nat_control() {
            n.set_port_forwards(rules);
        }
    }
    /// Change the PCAP bridged host interface (`None` = auto-pick).
    pub fn set_pcap_interface(&self, iface: Option<String>) {
        if let Some(n) = self.nat_control() {
            n.request_pcap_interface(iface);
        }
    }
}

/// A SCSI controller whose CD-ROM drives take disc changes at run time: the
/// window's RCtrl+F12, the CI socket's `cdrom-load`/`cdrom-eject`, iris-gui's
/// CD menu.
pub trait DiscChanger: Send + Sync {
    /// Every CD-ROM drive: (id, current disc, the configured disc list,
    /// list position, list length).
    fn disc_status(&self) -> Vec<(usize, String, Vec<String>, u64, u64)>;
    /// Put the disc at `path` in drive `id`; returns the path loaded.
    fn load_disc(&self, id: usize, path: String) -> Result<String, String>;
    /// Eject drive `id`'s disc and load the next in its list; returns its path.
    fn eject_disc(&self, id: usize) -> Result<String, String>;
    /// Eject drive `id`'s disc and leave it empty.
    fn eject_to_empty(&self, id: usize) -> Result<(), String>;
}

/// Every controller that lends its targets (the WD33C93A, the Adaptec 7880)
/// takes disc changes the same way; the operations are `scsi_monitor`'s.
impl<T: crate::scsi_monitor::ScsiTargets + Send + Sync> DiscChanger for T {
    fn disc_status(&self) -> Vec<(usize, String, Vec<String>, u64, u64)> {
        crate::scsi_monitor::disc_status(self)
    }
    fn load_disc(&self, id: usize, path: String) -> Result<String, String> {
        crate::scsi_monitor::load_disc(self, id, path)
    }
    fn eject_disc(&self, id: usize) -> Result<String, String> {
        crate::scsi_monitor::eject_disc(self, id)
    }
    fn eject_to_empty(&self, id: usize) -> Result<(), String> {
        crate::scsi_monitor::eject_to_empty(self, id)
    }
}

// ---- the shared machine slot ----

/// A raw pointer to a live machine, or null once that machine is dropped.
pub struct MachineRef(pub(crate) *mut dyn Workstation);

impl MachineRef {
    /// The pointer a dropped machine leaves in its slot.
    pub(crate) fn null() -> *mut dyn Workstation {
        std::ptr::null_mut::<Machine>() as *mut dyn Workstation
    }
}

// SAFETY: only dereferenced under the slot's Mutex, while non-null; a
// machine's `Drop` nulls it under that same Mutex before any field is torn
// down.
unsafe impl Send for MachineRef {}
unsafe impl Sync for MachineRef {}

/// How the monitor's system controller, the machine-events thread and the CI
/// server reach the machine: one slot per machine, shared by all of them and
/// nulled by the machine's `Drop`, so a holder that outlives the machine (the
/// monitor thread, after iris-gui's Stop) sees "machine is not running"
/// instead of freed memory.
pub type MachineSlot = Arc<Mutex<MachineRef>>;

/// Stop `machine` and flush its disks: what every path that ends the process
/// without dropping the machine does first. A failed flush is reported, not
/// fatal, since the process is going away either way.
pub fn stop_and_flush(machine: &mut dyn Workstation, at: &str) -> Flushed {
    stop_flushing(machine, |m| m.stop(), |m| m.flush_disks(), at);
    Flushed(())
}

/// What [`exit_process`] asks its caller's work to return, so that handing it
/// nothing does not compile. A nudge in the type, not a proof: `flushed` is
/// public and will vouch for any result, so what the work really does is for
/// the machine tests to check.
#[must_use]
pub struct Flushed(());

/// Report a flush's failure and vouch for it, for code that flushes a bus
/// itself (see [`exit_process`]).
pub fn flushed(result: std::io::Result<()>, at: &str) -> Flushed {
    if let Err(e) = result {
        eprintln!("iris: disk flush at {at} failed: {e} (guest writes may be lost)");
    }
    Flushed(())
}

/// [`stop_and_flush`]'s order: flush, stop, flush. The first flush is for a
/// stop that hangs in some device: the process is then killed, and the writes
/// made before the stop are already safe. It costs nothing when there is
/// nothing new.
fn stop_flushing<M: ?Sized>(
    machine: &mut M,
    stop: impl FnOnce(&mut M),
    flush: impl Fn(&M) -> std::io::Result<()>,
    at: &str,
) {
    let report = |r: std::io::Result<()>| {
        if let Err(e) = r {
            eprintln!("iris: disk flush at {at} failed: {e} (guest writes may be lost)");
        }
    };
    report(flush(machine));
    stop(machine);
    report(flush(machine));
}

#[cfg(test)]
#[test]
fn stop_and_flush_flushes_before_the_stop_and_after() {
    let log = std::cell::RefCell::new(Vec::new());
    stop_flushing(
        &mut (),
        |_| log.borrow_mut().push("stop"),
        |_| {
            log.borrow_mut().push("flush");
            Ok(())
        },
        "test",
    );
    assert_eq!(*log.borrow(), ["flush", "stop", "flush"]);
}

/// Flush a machine's disks as it is dropped. The disks hold on to their
/// overlays through handles the devices share with threads and tables that can
/// outlive the machine, so the machine cannot count on its disks' own `Drop`
/// to write the overlay's sector record.
pub(crate) fn flush_on_drop(machine: &dyn Workstation) {
    if let Err(e) = machine.flush_disks() {
        eprintln!("iris: disk flush at drop failed: {e} (guest writes may be lost)");
    }
}

/// [`stop_and_flush`] for the machine behind `slot`, if it is still there.
pub fn stop_and_flush_slot(slot: &MachineSlot, at: &str) -> Flushed {
    if let Err(e) = with_machine_slot(slot, |m| stop_and_flush(m, at)) {
        eprintln!("iris: could not stop the machine to flush its disks at {at}: {e}");
    }
    Flushed(())
}

/// What the terminal console runs when the monitor hangs up (`quit`): stop and
/// flush the machine behind `slot`, then end the process.
pub fn console_quit(slot: MachineSlot) -> impl FnOnce() + Send + 'static {
    move || exit_process("console quit", 0, || stop_and_flush_slot(&slot, "console quit"))
}

/// End the process on SIGINT (Ctrl-C at the terminal), SIGTERM or SIGHUP the
/// way `quit` does: stop and flush the machine behind `slot`, then exit with
/// 128 plus the signal number. Without this the signal's default action ends
/// the process with nothing flushed: a CHD loses what it took since its last
/// flush, and an overlay's sector record is left uncompacted.
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub fn exit_on_signal(slot: MachineSlot) {
    on_exit_signal(move |sig| {
        // Not eprintln!: on a terminal that has hung up it panics, and a panic
        // aborts the process before anything is flushed.
        let _ = writeln!(std::io::stderr(), "iris: signal {sig}: stopping the machine and flushing its disks before exit");
        exit_process("signal", 128 + sig, || stop_and_flush_slot(&slot, "signal"));
    });
}

/// Run `on_signal` once, on a thread of its own, when SIGINT, SIGTERM or
/// SIGHUP arrives, in place of the default action that ends the process at
/// once. It is handed the signal number and is expected to end the process
/// after stopping its machines, as [`exit_on_signal`] does.
///
/// A signal the process inherited as ignored stays ignored: `nohup` ignores
/// SIGHUP, and a shell ignores SIGINT for a background job, so that the
/// process outlives the terminal.
///
/// The handler only writes the signal number to a pipe; the thread reads it
/// and does the work, since a handler can do almost nothing safely. The
/// thread puts the default actions back first, so a second Ctrl-C ends a
/// stop that hangs. A process has one of these: a second call is refused.
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub fn on_exit_signal(on_signal: impl FnOnce(libc::c_int) + Send + 'static) {
    use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
    static PIPE: AtomicI32 = AtomicI32::new(-1);
    static INSTALLED: AtomicBool = AtomicBool::new(false);
    if INSTALLED.swap(true, Ordering::SeqCst) {
        eprintln!("iris: the exit-signal handler is already installed; not installing another");
        return;
    }
    let signals = exit_signals_to_catch();

    extern "C" fn handler(sig: libc::c_int) {
        let byte = sig as u8;
        // SAFETY: write(2) is async-signal-safe; errno is put back for the
        // code the signal interrupted.
        unsafe {
            let errno = *errno_location();
            libc::write(PIPE.load(Ordering::Relaxed), &byte as *const u8 as *const libc::c_void, 1);
            *errno_location() = errno;
        }
    }

    let mut fds = [-1; 2];
    // SAFETY: `fds` is two ints, as pipe(2) needs. Close-on-exec, so no
    // program iris starts holds the pipe.
    if unsafe {
        libc::pipe(fds.as_mut_ptr()) != 0
            || libc::fcntl(fds[0], libc::F_SETFD, libc::FD_CLOEXEC) != 0
            || libc::fcntl(fds[1], libc::F_SETFD, libc::FD_CLOEXEC) != 0
    } {
        eprintln!("iris: no pipe for the signal handler ({}); a signal will not flush the disks",
                  std::io::Error::last_os_error());
        return;
    }
    let (read_fd, write_fd) = (fds[0], fds[1]);
    PIPE.store(write_fd, Ordering::Relaxed);
    let thread_signals = signals.clone();
    let thread = std::thread::Builder::new().name("iris-signals".into()).spawn(move || {
        let signals = thread_signals;
        let mut byte = 0u8;
        loop {
            // SAFETY: reads one byte into `byte`.
            let n = unsafe { libc::read(read_fd, &mut byte as *mut u8 as *mut libc::c_void, 1) };
            if n == 1 {
                break;
            }
            if n < 0 && std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return;
        }
        for &sig in &signals {
            // SAFETY: restores a default action.
            unsafe { libc::signal(sig, libc::SIG_DFL) };
        }
        // A hangup means the terminal is gone and its descriptors fail with
        // EIO, which makes any print from any thread panic, and so abort the
        // process, part way through the flush. Point them at /dev/null.
        if libc::c_int::from(byte) == libc::SIGHUP {
            silence_output(&[1, 2]);
        }
        on_signal(libc::c_int::from(byte));
    });
    if let Err(e) = thread {
        eprintln!("iris: no thread for the signal handler ({e}); a signal will not flush the disks");
        return;
    }
    for &sig in &signals {
        // SAFETY: installs a handler that only calls write(2).
        unsafe {
            let mut sa: libc::sigaction = std::mem::zeroed();
            sa.sa_sigaction = handler as *const () as usize;
            sa.sa_flags = libc::SA_RESTART;
            libc::sigemptyset(&mut sa.sa_mask);
            libc::sigaction(sig, &sa, std::ptr::null_mut());
        }
    }
}

/// Point each of `fds` at /dev/null, so writing to it succeeds whatever became
/// of what it was.
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn silence_output(fds: &[libc::c_int]) {
    // SAFETY: opens /dev/null, duplicates it over each descriptor and closes it.
    unsafe {
        let null = libc::open(c"/dev/null".as_ptr(), libc::O_WRONLY);
        if null < 0 {
            return;
        }
        for &fd in fds {
            libc::dup2(null, fd);
        }
        libc::close(null);
    }
}

/// SIGINT, SIGTERM and SIGHUP, less any this process inherited as ignored.
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn exit_signals_to_catch() -> Vec<libc::c_int> {
    [libc::SIGINT, libc::SIGTERM, libc::SIGHUP]
        .into_iter()
        .filter(|&sig| {
            // SAFETY: a null new action only reads the current one.
            let mut old: libc::sigaction = unsafe { std::mem::zeroed() };
            unsafe { libc::sigaction(sig, std::ptr::null(), &mut old) };
            old.sa_sigaction != libc::SIG_IGN
        })
        .collect()
}

#[cfg(all(test, any(target_os = "linux", target_os = "macos")))]
#[test]
fn an_inherited_ignored_signal_stays_ignored() {
    // SAFETY: sets and reads back dispositions; SIGHUP is put back as found.
    let before = unsafe { libc::signal(libc::SIGHUP, libc::SIG_IGN) };
    let caught = exit_signals_to_catch();
    unsafe { libc::signal(libc::SIGHUP, before) };
    assert!(!caught.contains(&libc::SIGHUP), "nohup's ignored SIGHUP would end the process");
    assert!(caught.contains(&libc::SIGTERM));
}

/// The terminal hangs up: its slave side fails with EIO once the master is
/// closed, as a stderr that eprintln! would panic on. After `silence_output`
/// the same descriptor takes writes.
#[cfg(all(test, target_os = "linux"))]
#[test]
fn a_hung_up_terminals_output_is_silenced() {
    let (mut master, mut slave) = (-1, -1);
    // SAFETY: openpty fills two descriptors; the rest is plain fd handling.
    unsafe {
        assert_eq!(libc::openpty(&mut master, &mut slave, std::ptr::null_mut(), std::ptr::null(), std::ptr::null()), 0);
        libc::close(master);
        let put = |fd: libc::c_int| libc::write(fd, b"x".as_ptr().cast(), 1);
        assert_eq!(put(slave), -1, "a pty with no master should fail the write");
        silence_output(&[slave]);
        assert_eq!(put(slave), 1, "the silenced descriptor should take the write");
        libc::close(slave);
    }
}

#[cfg(target_os = "linux")]
unsafe fn errno_location() -> *mut libc::c_int {
    unsafe { libc::__errno_location() }
}

#[cfg(target_os = "macos")]
unsafe fn errno_location() -> *mut libc::c_int {
    unsafe { libc::__error() }
}

/// How many exits `IRIS_NO_EXIT_ON_POWEROFF` has turned into returns, for the
/// tests that wait for one.
#[cfg(test)]
pub(crate) static EXIT_REQUESTS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// End the process. This is the only place machine and device code may call
/// `std::process::exit`: it skips every Drop, which is where a copy-on-write
/// overlay would otherwise write its sector record, so the caller has to hand
/// in `stop_and_flush`, the work that makes the guest's writes durable
/// (`stop_and_flush`, `stop_and_flush_slot`, or `flushed` around a bus's
/// `flush_disks`), whose [`Flushed`] result the type requires. It runs first,
/// then the process exits unless `IRIS_NO_EXIT_ON_POWEROFF` is set, the escape
/// hatch for hosts that embed iris and for tests. That variable suppresses
/// every exit through here, the terminal console's `quit` too. A guard test
/// fails on any other `process::exit` in the library.
pub fn exit_process(at: &str, code: i32, stop_and_flush: impl FnOnce() -> Flushed) {
    let Flushed(()) = stop_and_flush();
    if std::env::var_os("IRIS_NO_EXIT_ON_POWEROFF").is_some() {
        eprintln!("iris: exit at {at} suppressed (IRIS_NO_EXIT_ON_POWEROFF)");
        #[cfg(test)]
        EXIT_REQUESTS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        return;
    }
    std::process::exit(code);
}

/// Run `f` against the machine behind `slot`, or fail if it has been dropped.
pub fn with_machine_slot<R>(slot: &MachineSlot, f: impl FnOnce(&mut dyn Workstation) -> R) -> Result<R, String> {
    let guard = slot.lock();
    let ptr = guard.0;
    if ptr.is_null() {
        return Err("machine is not running".to_string());
    }
    // SAFETY: non-null only while the machine is alive. Drop nulls it while
    // holding this lock, and `guard` is held until `f` returns, so the machine
    // cannot be freed while `f` runs.
    let result = f(unsafe { &mut *ptr });
    drop(guard);
    Ok(result)
}

/// Register a [`SystemController`] for the machine behind `slot` with
/// `monitor`, and when `events` is given, start the thread that carries out
/// those machine events. Each implementation's `register_system_controller`
/// calls this once, before the monitor server starts.
pub(crate) fn register_system_controller(
    monitor: &Arc<Monitor>,
    slot: MachineSlot,
    events: Option<std::sync::mpsc::Receiver<MachineEvent>>,
) {
    let ctrl = Arc::new(SystemController { machine: slot });
    // SAFETY: called once, before the monitor server thread starts, while the
    // caller has exclusive access to the machine that owns `monitor`.
    let monitor_ptr = Arc::as_ptr(monitor) as *mut Monitor;
    unsafe {
        (*monitor_ptr).register_device(ctrl.clone());
    }

    // The event thread goes through the same controller, which stops the
    // machine's threads before it changes machine state.
    if let Some(rx) = events {
        std::thread::Builder::new().name("machine-events".to_string()).spawn(move || {
            while let Ok(event) = rx.recv() {
                match event {
                    MachineEvent::HardReset => {
                        let _ = ctrl.with_machine(|machine| {
                            println!("Machine: SIN hard reset");
                            machine.reset();
                            machine.cpu_start();
                            Ok(())
                        });
                    }
                    MachineEvent::PowerOff => {
                        println!("Machine: soft power-off");
                        // The process may exit below without dropping the
                        // disks, and an overlay's sector record is only
                        // written out on a flush.
                        let flush = || stop_and_flush_slot(&ctrl.machine, "power-off");
                        // Hosts that embed iris as a library (e.g. iris-gui)
                        // set IRIS_NO_EXIT_ON_POWEROFF=1 so a guest halt
                        // does not kill the host process.
                        #[cfg(not(feature = "developer"))]
                        exit_process("power-off", 0, flush);
                        #[cfg(feature = "developer")]
                        let _ = flush();
                    }
                }
            }
        }).unwrap();
    }
}

/// A thin monitor device that reaches the machine through its slot so the
/// monitor thread can issue system-level commands (reset, save, load).
pub struct SystemController {
    pub(crate) machine: MachineSlot,
}

impl SystemController {
    fn with_machine<R>(&self, f: impl FnOnce(&mut dyn Workstation) -> Result<R, String>) -> Result<R, String> {
        with_machine_slot(&self.machine, f)?
    }
}

impl Device for SystemController {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) {}
    fn start(&self) {}
    fn is_running(&self) -> bool { false }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![
            ("machine-start".to_string(), "Start CPU and all peripherals (idempotent; prints the state transition)".to_string()),
            ("machine-stop".to_string(),  "Stop CPU and all peripherals (idempotent; prints the state transition)".to_string()),
            ("machine-status".to_string(), "machine-status [json] — Guest state (idle/starting/running/halted) and monitor readiness".to_string()),
            ("reset".to_string(),         "Reset all hardware to power-on state".to_string()),
            ("save".to_string(),          "save <name> — Save snapshot to saves/<name>/".to_string()),
            ("load".to_string(),          "load <name> — Load snapshot from saves/<name>/".to_string()),
            #[cfg(all(feature = "jitv2", feature = "developer"))]
            ("jitcheck".to_string(), "jitcheck <n> [skip] — capture live state, run n instructions interpreter-only vs real JIT dispatch, reconverge past the first `skip` divergences and stop at the next one [DEV]".to_string()),
        ]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut writer: Box<dyn Write + Send>) -> Result<(), String> {
        match cmd {
            "machine-stop" => {
                let ack = self.with_machine(|m| Ok(m.stop_guest()))?;
                let _ = writeln!(writer, "machine-stop: {}", ack.text());
                Ok(())
            }
            "machine-start" => {
                let ack = self.with_machine(|m| m.start_guest())?;
                let _ = writeln!(writer, "machine-start: {}", ack.text());
                Ok(())
            }
            "machine-status" => {
                let json = match args {
                    [] => false,
                    ["json"] => true,
                    _ => return Err("Usage: machine-status [json]".to_string()),
                };
                let status = self.with_machine(|m| Ok(m.status()))?;
                if json {
                    let _ = writeln!(writer, "{}", status.to_json());
                } else {
                    let _ = writeln!(writer, "{}", status.text());
                }
                Ok(())
            }
            "reset" => {
                let _ = writeln!(writer, "Resetting machine...");
                self.with_machine(|m| { m.reset(); Ok(()) })
            }
            "save" => {
                let name = args.first().ok_or_else(|| "Usage: save <name>".to_string())?;
                let _ = writeln!(writer, "Saving snapshot '{}'...", name);
                self.with_machine(|m| m.save_snapshot(name))
            }
            "load" => {
                let name = args.first().ok_or_else(|| "Usage: load <name>".to_string())?;
                let _ = writeln!(writer, "Loading snapshot '{}'...", name);
                self.with_machine(|m| m.load_snapshot(name))
            }
            #[cfg(all(feature = "jitv2", feature = "developer"))]
            "jitcheck" => {
                let n: u64 = args.first()
                    .and_then(|s| s.parse().ok())
                    .ok_or("Usage: jitcheck <n> [skip]".to_string())?;
                let skip: u64 = args.get(1)
                    .map(|s| s.parse().map_err(|_| "Usage: jitcheck <n> [skip]".to_string()))
                    .transpose()?
                    .unwrap_or(0);
                let _ = writeln!(writer, "jitcheck: capturing live state, running {} instructions interpreter-only then with real JIT dispatch (skipping first {} divergence(s))...", n, skip);
                self.with_machine(|m| {
                    let m = m.as_any_mut().downcast_mut::<Machine>()
                        .ok_or_else(|| format!("jitcheck: {NOT_HERE}"))?;
                    let report = crate::validate::validate_jit_determinism(m, n, skip)
                        .map_err(|e| format!("jitcheck failed: {}", e))?;
                    for (i, d) in report.skipped.iter().enumerate() {
                        let _ = writeln!(writer, "jitcheck: SKIPPED divergence #{} at instruction {} of {} (pc={:#018x}), reconverged and continued", i + 1, d.instruction, n, d.replay_pc);
                        for (field, a, b) in &d.diffs {
                            let _ = writeln!(writer, "  {}: interp={} jit={}", field, a, b);
                        }
                    }
                    let next_divergence_number = report.skipped.len() + 1;
                    match &report.stopped_at {
                        None => {
                            let _ = writeln!(writer, "jitcheck: no further divergence across {} instructions", n);
                            let _ = writeln!(writer, "jitcheck: CPU left stopped (peripherals still running) with the replay (JIT-dispatch) pass's final state loaded — 'start' to resume the CPU");
                        }
                        Some(d) => {
                            let _ = writeln!(writer, "jitcheck: DIVERGED (divergence #{}) at instruction {} of {} (pc={:#018x})", next_divergence_number, d.instruction, n, d.replay_pc);
                            for (field, a, b) in &d.diffs {
                                let _ = writeln!(writer, "  {}: interp={} jit={}", field, a, b);
                            }
                            let _ = writeln!(writer, "jitcheck: rerun with skip={} to reconverge past this one too", next_divergence_number);
                            let _ = writeln!(writer, "jitcheck: stopped immediately at the divergence — CPU left stopped exactly as the JIT-dispatch pass left it (peripherals still running, console stays live) — 'start' to resume the CPU");
                        }
                    }
                    Ok(())
                })
            }
            _ => Err(format!("Unknown command: {}", cmd)),
        }
    }
}
