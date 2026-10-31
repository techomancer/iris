//! Stopping a machine from the main thread must not wait on its renderer.
//!
//! On macOS a renderer's `stop` needs the main thread, which is the thread
//! stopping the machine (see `gfx_display::DisplayStop`). The renderer here
//! does the same: its `stop` asks for an answer from the test thread. One
//! machine per display family, stopped from a reset and then from main:
//! Newport (REX3), GR2, IMPACT (MGRAS) and the O2.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::SeqCst};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

use crate::config::{CpuModel, GraphicsBoard, MachineConfig, MachineProfile};
use crate::dev::ng1::rex3::Renderer;
use crate::workstation::Workstation;

/// A stop that waits on the main thread never returns.
const BOUND: Duration = Duration::from_secs(8);

/// A renderer that counts presents and runs `on_stop` when torn down.
#[derive(Clone)]
pub(crate) struct Probe {
    presents: Arc<AtomicUsize>,
    pub(crate) stopped: Arc<AtomicBool>,
    on_stop: Arc<dyn Fn() + Send + Sync>,
}

impl Probe {
    pub(crate) fn new(on_stop: impl Fn() + Send + Sync + 'static) -> Self {
        Self { presents: Arc::default(), stopped: Arc::default(), on_stop: Arc::new(on_stop) }
    }

    /// A probe that notes whether `disks_running` was true when torn down.
    pub(crate) fn watching(disks_running: impl Fn() -> bool + Send + Sync + 'static) -> (Self, Arc<AtomicBool>) {
        let seen = Arc::new(AtomicBool::default());
        let flag = seen.clone();
        (Self::new(move || if disks_running() { flag.store(true, SeqCst) }), seen)
    }
}

impl Renderer for Probe {
    fn present(
        &mut self, _: &mut crate::disp::Rex3Screen, _: &mut crate::debug_overlay::DebugOverlay,
        _: &mut crate::disp::StatusBar, _: &mut crate::disp::StatusBarTexture, _: &crate::disp::BarStats,
        _: bool, _: Option<&[u32]>, _: Option<&[u32]>,
    ) {
        self.presents.fetch_add(1, SeqCst);
    }

    fn stop(&mut self) {
        (self.on_stop)();
        self.stopped.store(true, SeqCst);
    }
}

/// A scratch directory for one test's files (the Octane and O2 read a PROM
/// from a file; these tests run no guest code, so zeros will do). Tests run
/// side by side in one process and each removes its own directory, so each
/// takes a name of its own.
pub(crate) fn scratch(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("iris-display-stop-{test}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // A REX3 saves the shapes it drew when it stops, to $HOME/.iris unless
    // told otherwise. The path is the same for every test, as in
    // `exit_persistence_tests::serial`.
    std::env::set_var("IRIS_REX_JIT_PROFILE", std::env::temp_dir().join("iris-display-stop-rex-jit-profile.bin"));
    dir
}

pub(crate) fn config(dir: &Path, profile: MachineProfile, board: GraphicsBoard) -> MachineConfig {
    let (mb, cpu) = match profile {
        MachineProfile::OctaneIp30 => (64, CpuModel::R10000),
        MachineProfile::O2Ip32 => (32, CpuModel::R5000),
        _ => (16, CpuModel::default()),
    };
    let mut cfg = crate::bench_runner::bench_config([mb, 0, 0, 0], cpu);
    cfg.headless = false;
    cfg.machine.profile = profile;
    cfg.monitor_port = Some(0);
    cfg.nvram = dir.join("nvram").to_string_lossy().into_owned();
    cfg.nveeprom = dir.join("nveeprom").to_string_lossy().into_owned();
    if matches!(profile, MachineProfile::OctaneIp30 | MachineProfile::O2Ip32) {
        let prom = dir.join("prom.bin");
        std::fs::write(&prom, vec![0u8; 1 << 20]).unwrap();
        cfg.prom = prom.to_string_lossy().into_owned();
    }
    // The O2's GBE is its own display; it has no `[graphics] board`.
    cfg.graphics.board = (profile != MachineProfile::O2Ip32).then_some(board);
    if board == GraphicsBoard::Newport {
        // A Newport shows nothing until a video mode is programmed. Set it
        // here rather than wait for a guest to get that far.
        cfg.graphics.resolution = crate::dev::ng1::vc2_timings::NewportResolution::Res1280x1024;
    }
    cfg
}

pub(crate) fn big_stack<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    std::thread::Builder::new().stack_size(64 * 1024 * 1024).spawn(f).unwrap().join().unwrap()
}

/// Stop `machine` on a thread called `name`, and drop it there too if `last`
/// (the O2 stops its display in its drop). The machine comes back unless it
/// was dropped, or `None` if that did not finish within [`BOUND`].
fn stop_as(name: &str, mut machine: Box<dyn Workstation>, last: bool) -> Option<Option<Box<dyn Workstation>>> {
    let (done, finished) = mpsc::channel();
    std::thread::Builder::new().name(name.into()).spawn(move || {
        machine.stop();
        if last {
            drop(machine);
            let _ = done.send(None);
        } else {
            let _ = done.send(Some(machine));
        }
    }).unwrap();
    finished.recv_timeout(BOUND).ok()
}

#[test]
fn stopping_from_the_main_thread_never_waits_on_the_renderer() {
    let dir = scratch("main-thread");
    let families = [
        (MachineProfile::IndyIp24, GraphicsBoard::Newport),
        (MachineProfile::Indigo2Ip22, GraphicsBoard::Xz),
        (MachineProfile::OctaneIp30, GraphicsBoard::SolidImpact),
        (MachineProfile::O2Ip32, GraphicsBoard::Newport),
    ];
    let mut failures = Vec::new();
    for (profile, board) in families {
        // Stopped from the main thread, the renderer's stop waits for an
        // answer that never comes. Stopped from another thread (a reset) the
        // main thread answers, and the renderer must still be released.
        let answered = Arc::new(AtomicBool::new(true));
        let gate = answered.clone();
        let probe = Probe::new(move || while !gate.load(SeqCst) { std::thread::sleep(Duration::from_millis(5)) });
        // The reset comes first and the machine starts again for the stop
        // from main. The O2's display thread ends only in its drop, so it
        // needs a machine for each.
        let reuse = profile != MachineProfile::O2Ip32;
        let mut machine: Option<Box<dyn Workstation>> = None;
        for (name, from_main) in [("reset", false), ("main", true)] {
            let label = format!("{profile:?} {board:?} stopped from {name}");
            let before = probe.presents.load(SeqCst);
            let m = match machine.take() {
                Some(mut m) => {
                    m.restart_peripherals();
                    m
                }
                None => {
                    let cfg = config(&dir, profile, board);
                    let mut m = big_stack(move || crate::workstation::build(cfg).expect("machine"));
                    *m.get_display().expect("a display").renderer_slot().lock() = Some(Box::new(probe.clone()));
                    m.start();
                    m
                }
            };
            let up = Instant::now() + Duration::from_secs(10);
            while probe.presents.load(SeqCst) == before && Instant::now() < up {
                std::thread::sleep(Duration::from_millis(10));
            }
            if probe.presents.load(SeqCst) == before {
                failures.push(format!("{label}: the display never presented"));
                break;
            }
            probe.stopped.store(false, SeqCst);
            answered.store(!from_main, SeqCst);
            match stop_as(name, m, from_main || !reuse) {
                None => failures.push(format!("{label}: stop did not return within {BOUND:?}")),
                Some(back) => {
                    machine = back;
                    if !from_main && !probe.stopped.load(SeqCst) {
                        failures.push(format!("{label}: the renderer was not released"));
                    }
                }
            }
            answered.store(true, SeqCst); // let a hung stop finish
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// The graphics board can hang in its stop, so the SCSI controllers and
/// IOC3 must already be stopped when its renderer is torn down.
#[test]
fn the_octane_stops_its_disks_before_its_graphics() {
    use crate::traits::Device;
    let dir = scratch("octane-order");
    let cfg = config(&dir, MachineProfile::OctaneIp30, GraphicsBoard::SolidImpact);
    let mut machine = big_stack(move || crate::ip30_machine::OctaneMachine::new(&cfg).expect("Octane"));
    let (scsi, ioc3) = (machine.scsi.clone(), machine.ioc3.clone());
    let (probe, overlap) = Probe::watching(move || ioc3.is_running() || scsi.iter().any(|s| s.is_running()));
    *machine.gfx.as_ref().expect("graphics").display().renderer_slot().lock() = Some(Box::new(probe.clone()));
    machine.restart_peripherals();
    assert!(!machine.scsi.is_empty() && machine.scsi.iter().all(|s| s.is_running()), "the SCSI controllers are running");
    assert!(machine.ioc3.is_running(), "IOC3 is running");
    machine.stop();
    assert!(probe.stopped.load(SeqCst), "the display thread released its renderer");
    assert!(!overlap.load(SeqCst), "a disk controller was still running when the graphics stopped");
    let _ = std::fs::remove_dir_all(&dir);
}
