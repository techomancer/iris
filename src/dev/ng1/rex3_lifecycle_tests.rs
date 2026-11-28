use super::*;
use crate::dev::ng1::wgpu::resident::{
    Observation, ObservationReason, PlaneMask, ResidentBackend, ResidentError, ResidentStatus,
    PLANE_WORDS,
};
use crate::gfx_display::{copy_frame, GfxDisplay};
use crate::test_board::TestBoard;
use crate::workstation::Workstation;
use std::io;
use std::path::Path;
use std::sync::mpsc;
use std::time::Duration;

struct ProbeState {
    rgb: Vec<u32>,
    aux: Vec<u32>,
    observations: usize,
    resets: usize,
    restores: usize,
    health_error: Option<ResidentError>,
    status_error: Option<ResidentError>,
    observation_error: Option<ResidentError>,
    reset_error: Option<ResidentError>,
    restore_error: Option<ResidentError>,
    observation_barrier: Option<(mpsc::Sender<()>, mpsc::Receiver<()>)>,
    reset_barrier: Option<(mpsc::Sender<()>, mpsc::Receiver<()>)>,
}

struct Probe(Arc<Mutex<ProbeState>>);

impl ResidentBackend for Probe {
    fn enqueue(&mut self, _: &mut Rex3Context) -> Result<(), ResidentError> { Ok(()) }
    fn submit(&mut self) -> Result<(), ResidentError> { Ok(()) }
    fn poll_device_health(&mut self) -> Result<(), ResidentError> {
        self.0.lock().health_error.clone().map_or(Ok(()), Err)
    }
    fn status(&mut self) -> Result<ResidentStatus, ResidentError> {
        self.0.lock().status_error.clone().map_or_else(|| Ok(ResidentStatus::for_test()), Err)
    }
    fn observe(&mut self, reason: ObservationReason, planes: PlaneMask) -> Result<Observation, ResidentError> {
        assert_eq!((reason, planes), (ObservationReason::Snapshot, PlaneMask::Both));
        let mut state = self.0.lock();
        state.observations += 1;
        if let Some(error) = &state.observation_error {
            return Err(error.clone());
        }
        if let Some((entered, release)) = state.observation_barrier.take() {
            entered.send(()).unwrap();
            drop(state);
            release.recv_timeout(Duration::from_secs(5)).unwrap();
            state = self.0.lock();
        }
        Ok(Observation::for_test(
            reason, planes,
            Some(bytemuck::cast_slice(&state.rgb).to_vec()),
            Some(bytemuck::cast_slice(&state.aux).to_vec()),
        ))
    }
    fn reset(&mut self, rgb: &[u32], aux: &[u32]) -> Result<(), ResidentError> {
        let mut state = self.0.lock();
        state.resets += 1;
        if let Some(error) = &state.reset_error { return Err(error.clone()); }
        if let Some((entered, release)) = state.reset_barrier.take() {
            entered.send(()).unwrap();
            drop(state);
            release.recv_timeout(Duration::from_secs(5)).unwrap();
            state = self.0.lock();
        }
        assert_eq!((rgb.len(), aux.len()), (PLANE_WORDS, PLANE_WORDS));
        assert!(rgb.iter().all(|&word| word == 0) && aux.iter().all(|&word| word == 0));
        state.rgb.copy_from_slice(rgb);
        state.aux.copy_from_slice(aux);
        Ok(())
    }
    fn restore(&mut self, rgb: &[u32], aux: &[u32]) -> Result<(), ResidentError> {
        let mut state = self.0.lock();
        state.restores += 1;
        if let Some(error) = state.restore_error.clone() {
            state.rgb[0] ^= 0xffff_ffff;
            return Err(error);
        }
        assert_eq!((rgb.len(), aux.len()), (PLANE_WORDS, PLANE_WORDS));
        state.rgb.copy_from_slice(rgb);
        state.aux.copy_from_slice(aux);
        Ok(())
    }
}

fn on_large_stack(f: impl FnOnce() + Send + 'static) {
    thread::Builder::new().name("resident-lifecycle-test".into()).stack_size(64 * 1024 * 1024)
        .spawn(f).unwrap().join().unwrap_or_else(|panic| std::panic::resume_unwind(panic));
}

fn probe_state() -> Arc<Mutex<ProbeState>> {
    let state = Arc::new(Mutex::new(ProbeState {
        rgb: vec![0x1234_5678; PLANE_WORDS],
        aux: vec![0x89ab_cdef; PLANE_WORDS],
        observations: 0, resets: 0, restores: 0,
        health_error: None, status_error: None, observation_error: None, reset_error: None, restore_error: None,
        observation_barrier: None, reset_barrier: None,
    }));
    {
        let mut data = state.lock();
        data.rgb[PLANE_WORDS - 1] = 0xfedc_ba98;
        data.aux[PLANE_WORDS - 1] = 0x7654_3210;
    }
    state
}

fn fixture() -> (
    TestBoard<Rex3>,
    Arc<Rex3>,
    Arc<Mutex<ProbeState>>,
    plane::PlaneProtection,
    plane::PlaneProtection,
) {
    let mut rex = Box::new(Rex3::new(crate::status::MachineStatus::detached()));
    rex.fb_rgb.get_mut().isolate().unwrap();
    rex.fb_aux.get_mut().isolate().unwrap();
    let state = probe_state();
    rex.attach_resident_raster(Probe(Arc::clone(&state))).unwrap();
    rex.fb_rgb.get_mut().reset_access_counts();
    rex.fb_aux.get_mut().reset_access_counts();
    let rgb_guard = unsafe { (&*rex.fb_rgb.get()).protect() }.unwrap();
    let aux_guard = unsafe { (&*rex.fb_aux.get()).protect() }.unwrap();
    let rex = Arc::<Rex3>::from(rex);
    (TestBoard::new(Arc::clone(&rex)), rex, state, rgb_guard, aux_guard)
}

fn assert_no_cpu_plane_access(rex: &Rex3) {
    assert_eq!(unsafe { (&*rex.fb_rgb.get()).access_counts() }, Default::default());
    assert_eq!(unsafe { (&*rex.fb_aux.get()).access_counts() }, Default::default());
}

fn panics(f: impl FnOnce()) {
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).is_err());
}

#[test]
fn resident_lifecycle_restores_registers_planes_and_continuations_atomically() {
    on_large_stack(|| {
        let (_owner, rex, state, _rgb_guard, _aux_guard) = fixture();
        let ctx = unsafe { &mut *rex.context.get() };
        ctx.colorred = 0x123456;
        ctx.host_shifter = 0x1234_5678_9abc_def0;
        ctx.hostcnt = 5;
        ctx.hostrw[0] = 0xfedc_ba98_7654_3210;
        ctx.pat_bit = 7;
        ctx.zpat_bit = 11;
        ctx.mid_primitive = true;
        rex.config.config.store(0x45678, Ordering::Relaxed);
        rex.dcb.lock().dcbdata1 = 0xabc_def0;
        rex.vc2.lock().ram[123] = 0x5a5a;
        rex.xmap1.lock().mode_table[9] = 0x123456;
        {
            let mut cmap = rex.cmap0.lock();
            cmap.addr_lo = 3;
            cmap.write_crs(crate::dev::ng1::cmap::CMAP_REG_PALETTE, 0x12);
            cmap.write_crs(crate::dev::ng1::cmap::CMAP_REG_PALETTE, 0x34);
        }
        rex.bt445.lock().palette[17] = [3, 5, 7];

        let registers = rex.save_state();
        let (rgb, aux) = rex.snapshot_framebuffers_inmem();
        assert_eq!(state.lock().observations, 1);
        assert_eq!((rgb[0], rgb[PLANE_WORDS - 1]), (0x1234_5678, 0xfedc_ba98));
        assert_eq!((aux[0], aux[PLANE_WORDS - 1]), (0x89ab_cdef, 0x7654_3210));

        rex.power_on();
        assert_eq!(state.lock().resets, 1);
        assert!(state.lock().rgb.iter().all(|&word| word == 0));
        assert_eq!(unsafe { (*rex.context.get()).colorred }, 0);
        rex.load_state(&registers).unwrap();
        assert_eq!(unsafe { (*rex.context.get()).colorred }, 0, "staged registers are not published");
        assert_eq!(rex.start_resident_processor(), Err(ResidentError::InvalidState("resident restore is incomplete")));
        assert!(rex.apply_resident_display_resolution(
            crate::dev::ng1::vc2_timings::NewportResolution::Res1024x768, true,
        ).is_err());
        assert!(rex.execute_command("bt445", &["identity"], Box::new(io::sink())).is_err());
        panics(|| { let _ = rex.save_state(); });
        assert!(rex.snapshot_resident_planes().is_err());

        rex.restore_framebuffers_inmem(&rgb, &aux);
        assert_eq!(state.lock().restores, 1);
        assert_eq!(rex.save_state(), registers);
        let ctx = unsafe { &*rex.context.get() };
        assert_eq!((ctx.host_shifter, ctx.hostcnt, ctx.hostrw[0]), (0x1234_5678_9abc_def0, 5, 0xfedc_ba98_7654_3210));
        assert_eq!((ctx.pat_bit, ctx.zpat_bit, ctx.mid_primitive), (7, 11, true));
        {
            let mut cmap = rex.cmap0.lock();
            cmap.write_crs(crate::dev::ng1::cmap::CMAP_REG_PALETTE, 0x56);
            assert_eq!(cmap.palette[3], 0x563412);
        }
        assert_eq!(&state.lock().rgb, &rgb);
        assert_eq!(&state.lock().aux, &aux);

        let expected = rex.save_state();
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        state.lock().observation_barrier = Some((entered_tx, release_rx));
        let snapshot_board = Arc::clone(&rex);
        let snapshot = thread::Builder::new().stack_size(64 * 1024 * 1024)
            .spawn(move || snapshot_board.snapshot_resident_checkpoint()).unwrap();
        entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(rex.resident_lifecycle.try_write().is_none());
        let (reset_tx, reset_rx) = mpsc::channel();
        let reset_board = Arc::clone(&rex);
        let reset = thread::Builder::new().stack_size(64 * 1024 * 1024)
            .spawn(move || reset_tx.send(reset_board.reset_resident_board()).unwrap()).unwrap();
        assert!(matches!(reset_rx.recv_timeout(Duration::from_millis(100)), Err(mpsc::RecvTimeoutError::Timeout)));
        release_tx.send(()).unwrap();
        let (saved_state, (saved_rgb, saved_aux)) = snapshot.join().unwrap().unwrap();
        assert_eq!(saved_state, expected);
        assert_eq!(saved_rgb, rgb);
        assert_eq!(saved_aux, aux);
        reset_rx.recv_timeout(Duration::from_secs(5)).unwrap().unwrap();
        reset.join().unwrap();
        assert_eq!(unsafe { (*rex.context.get()).colorred }, 0);
        assert_no_cpu_plane_access(&rex);
    });
}

#[test]
fn resident_lifecycle_rejects_nonquiescent_and_incomplete_transfers() {
    on_large_stack(|| {
        let (_owner, rex, state, _rgb_guard, _aux_guard) = fixture();
        let registers = rex.save_state();
        rex.start();
        assert!(rex.save_framebuffers_named(Path::new("target"), "resident-m7-must-not-exist").is_err());
        panics(|| { let _ = rex.snapshot_framebuffers_inmem(); });
        panics(|| { let _ = rex.save_state(); });
        assert!(rex.load_state(&registers).is_err());
        panics(|| rex.power_on());
        assert_eq!(state.lock().observations, 0);
        rex.stop();

        assert_eq!(rex.write32((REX3_BASE | REX3_WRMASK) as u64, 1), BUS_ERR);
        assert!(rex.gfifo.try_push(REX3_WRMASK, 5));
        assert!(rex.snapshot_resident_planes().is_err());
        assert!(rex.start_resident_processor().is_err());
        panics(|| rex.power_on());
        rex.gfifo.reset();
        unsafe { (*rex.context.get()).host_len = 2; }
        assert!(rex.snapshot_resident_planes().is_err());
        assert!(rex.load_state(&registers).is_err());
        assert!(rex.execute_command("rex", &["status"], Box::new(io::sink())).is_err());
        unsafe { (*rex.context.get()).host_len = 1; (*rex.context.get()).host_cursor = 1; }
        assert!(rex.snapshot_resident_planes().is_err());
        unsafe { (*rex.context.get()).host_cursor = 0; }
        assert!(rex.snapshot_resident_planes().is_ok());
        assert_no_cpu_plane_access(&rex);
    });
}

#[test]
fn resident_lifecycle_reset_hides_frontend_before_upload() {
    on_large_stack(|| {
        use crate::dev::ng1::vc2::{VC2_CTRL_CURSOR_EN, VC2_REG_DISPLAY_CONTROL};
        let (_owner, rex, state, _rgb_guard, _aux_guard) = fixture();
        {
            let mut screen = rex.head.screen().lock();
            screen.width = 32;
            screen.height = 32;
        }
        rex.vc2.lock().regs[VC2_REG_DISPLAY_CONTROL as usize] = VC2_CTRL_CURSOR_EN;
        assert!(rex.hw_cursor().is_some());

        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        state.lock().reset_barrier = Some((entered_tx, release_rx));
        let resetting = Arc::clone(&rex);
        let resetting = thread::Builder::new().stack_size(64 * 1024 * 1024)
            .spawn(move || resetting.power_on()).unwrap();
        entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        let hidden_during_upload = rex.hw_cursor().is_none();
        let debugger_busy = rex.execute_command("bt445", &["identity"], Box::new(io::sink())).is_err();
        release_tx.send(()).unwrap();
        resetting.join().unwrap();
        assert!(hidden_during_upload);
        assert!(debugger_busy);
        assert!(rex.hw_cursor().is_none());
        rex.execute_command("bt445", &["identity"], Box::new(io::sink())).unwrap();
        assert_eq!(rex.bt445.lock().palette[1], [1, 1, 1]);
        assert_no_cpu_plane_access(&rex);
    });
}

#[test]
fn resident_lifecycle_stop_waits_for_producers_and_restart_rebinds_wakeup() {
    on_large_stack(|| {
        let (_owner, rex, _state, _rgb_guard, _aux_guard) = fixture();
        rex.start();
        rex.execute_command("rex", &["status"], Box::new(io::sink())).unwrap();
        while rex.resident_unparker.lock().is_none() { thread::yield_now(); }
        let first = rex.resident_unparker.lock().as_ref().unwrap().id();
        let (held_tx, held_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let producer = Arc::clone(&rex);
        let producer = thread::spawn(move || {
            let _guard = producer.resident_bus_guard().unwrap().unwrap();
            held_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        });
        held_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        let (done_tx, done_rx) = mpsc::channel();
        let stopping = Arc::clone(&rex);
        let stopping = thread::spawn(move || {
            stopping.stop();
            done_tx.send(()).unwrap();
        });
        assert!(done_rx.recv_timeout(Duration::from_millis(100)).is_err());
        release_tx.send(()).unwrap();
        producer.join().unwrap();
        done_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        stopping.join().unwrap();
        assert_eq!(rex.read32((REX3_BASE | REX3_WRMASK) as u64).status, BUS_ERR);

        rex.start();
        while rex.resident_unparker.lock().is_none() { thread::yield_now(); }
        assert_ne!(rex.resident_unparker.lock().as_ref().unwrap().id(), first);
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while !rex.processor_parked.load(Ordering::Acquire) {
            assert!(std::time::Instant::now() < deadline);
            thread::yield_now();
        }
        assert_eq!(rex.write32((REX3_BASE | REX3_WRMASK) as u64, 0x12345678), BUS_OK);
        rex.wait_resident_idle().unwrap();
        assert_eq!(unsafe { (*rex.context.get()).wrmask }, 0x12345678);
        rex.stop();
        assert_no_cpu_plane_access(&rex);
    });
}

#[test]
fn resident_lifecycle_machine_head1_checkpoint_round_trip() {
    const CHILD: &str = "IRIS_RESIDENT_M7_HEAD1_CHILD";
    if std::env::var_os(CHILD).is_none() {
        // Not under the working directory: a relocated target dir leaves no `target/` there.
        let path = crate::test_temp::TempPath::dir("resident-m7-head1");
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .current_dir(&*path)
            .env(CHILD, &*path)
            .env("IRIS_REX_JIT_PROFILE", "profile.bin")
            .args([
                "--exact", "dev::ng1::rex3::lifecycle_tests::resident_lifecycle_machine_head1_checkpoint_round_trip",
                "--nocapture", "--test-threads=1",
            ])
            .output().unwrap();
        assert!(output.status.success(), "head 1 child failed:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
        return;
    }

    let path = std::path::PathBuf::from(std::env::var_os(CHILD).unwrap());
    // The temp dir may sit behind a symlink (macOS /var), so compare resolved paths.
    assert_eq!(std::env::current_dir().unwrap().canonicalize().unwrap(), path.canonicalize().unwrap());
    assert!(path.file_name().and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with("iris-resident-m7-head1-")));
    on_large_stack(|| {
        let mut cfg = crate::bench_runner::bench_config([16, 0, 0, 0], &crate::cpu::parts::R4400_150);
        cfg.graphics.board = Some(crate::config::GraphicsBoard::Newport);
        cfg.set_device("newport", true);
        cfg.nvram.clear();
        cfg.nveeprom.clear();
        cfg.monitor_port = Some(0);
        cfg.graphics.resolution = crate::dev::ng1::vc2_timings::NewportResolution::Res1024x768;
        {
            let mut cpu_machine = Box::new(crate::machine::Machine::new(cfg.clone()));
            cpu_machine.save_snapshot("m7-head1").unwrap();
            let snapshot = Path::new("saves").join("m7-head1");
            assert!(snapshot.join("rex3_head1_rgb.bin").is_file());
            assert!(snapshot.join("rex3_head1_aux.bin").is_file());
            cpu_machine.stop();
        }
        let mut machine = Box::new(crate::machine::Machine::new(cfg));
        let primary = machine.get_rex3().expect("primary Newport head");
        let rex = machine.get_rex3_head1().expect("second Newport head");
        primary.start();
        rex.start();
        machine.stop();
        assert!(primary.check_resident_quiescent().is_ok());
        assert!(rex.check_resident_quiescent().is_ok());
        let primary_state = probe_state();
        {
            let mut data = primary_state.lock();
            data.rgb[0] = 0xaabb_ccdd;
            data.aux[0] = 0x1122_3344;
        }
        let state = probe_state();
        let attach = |board: &Rex3, data: &Arc<Mutex<ProbeState>>| {
            unsafe {
                (&mut *board.fb_rgb.get()).isolate().unwrap();
                (&mut *board.fb_aux.get()).isolate().unwrap();
                (&mut *board.fb_rgb.get()).reset_access_counts();
                (&mut *board.fb_aux.get()).reset_access_counts();
            }
            {
                let mut raster = board.raster.lock();
                assert_eq!(raster.kind(), RasterModeKind::Cpu);
                *raster = RasterMode::Resident(crate::dev::ng1::wgpu::resident::ResidentRaster::new(
                    Probe(Arc::clone(data)),
                ));
            }
            board.resident_frame.activate();
            (
                unsafe { (&*board.fb_rgb.get()).protect() }.unwrap(),
                unsafe { (&*board.fb_aux.get()).protect() }.unwrap(),
            )
        };
        let primary_guards = attach(&primary, &primary_state);
        let head1_guards = attach(&rex, &state);
        let visible = |board: &Rex3| {
            let vc2 = board.vc2.lock();
            crate::dev::ng1::vc2_timings::decode_vc2_visible_size(&vc2.regs, &vc2.ram)
        };
        machine.start();
        machine.stop();
        assert_eq!(visible(&primary), (1024, 768));
        assert_eq!(visible(&rex), (1024, 768));
        assert_eq!(primary_state.lock().rgb[0], 0xaabb_ccdd);
        assert_eq!(state.lock().rgb[0], 0x1234_5678);
        unsafe {
            (*primary.context.get()).colorred = 0x456789;
            (*rex.context.get()).colorred = 0x123456;
        }
        primary.vc2.lock().ram[123] = 0x4567;
        rex.vc2.lock().ram[123] = 0x1234;
        primary.start();
        rex.start();

        let checkpoint = machine.capture_live_checkpoint();
        unsafe {
            (*primary.context.get()).colorred = 0x112233;
            (*rex.context.get()).colorred = 0xabcdef;
        }
        {
            let mut data = state.lock();
            data.rgb[0] = 0;
            data.aux[0] = 0;
        }
        {
            let mut data = primary_state.lock();
            data.rgb[0] = 0;
            data.aux[0] = 0;
        }
        machine.restore_live_checkpoint(&checkpoint).unwrap();
        assert!(primary.is_running());
        assert!(rex.is_running());
        assert_eq!(unsafe { (*primary.context.get()).colorred }, 0x456789);
        assert_eq!(unsafe { (*rex.context.get()).colorred }, 0x123456);
        assert_eq!((primary.vc2.lock().ram[123], rex.vc2.lock().ram[123]), (0x4567, 0x1234));
        {
            let data = primary_state.lock();
            assert_eq!((data.rgb[0], data.aux[0]), (0xaabb_ccdd, 0x1122_3344));
            assert_eq!((data.observations, data.restores), (1, 1));
        }
        {
            let data = state.lock();
            assert_eq!((data.rgb[0], data.aux[0]), (0x1234_5678, 0x89ab_cdef));
            assert_eq!((data.observations, data.restores), (1, 1));
        }
        machine.save_snapshot("m7-head1").unwrap();
        let snapshot = Path::new("saves").join("m7-head1");
        assert!(snapshot.join("rex3_head1_rgb.bin").is_file());
        assert!(snapshot.join("rex3_head1_aux.bin").is_file());
        machine.stop();
        unsafe {
            (*primary.context.get()).colorred = 0x998877;
            (*rex.context.get()).colorred = 0xfedcba;
        }
        {
            let mut data = state.lock();
            data.rgb[0] = 0;
            data.aux[0] = 0;
        }
        {
            let mut data = primary_state.lock();
            data.rgb[0] = 0;
            data.aux[0] = 0;
        }
        machine.load_snapshot_paused("m7-head1").unwrap();
        assert_eq!(unsafe { (*primary.context.get()).colorred }, 0x456789);
        assert_eq!(unsafe { (*rex.context.get()).colorred }, 0x123456);
        assert_eq!((primary.vc2.lock().ram[123], rex.vc2.lock().ram[123]), (0x4567, 0x1234));
        {
            let data = primary_state.lock();
            assert_eq!((data.rgb[0], data.aux[0]), (0xaabb_ccdd, 0x1122_3344));
            assert_eq!((data.observations, data.restores), (2, 2));
        }
        machine.start();
        assert_eq!((primary.vc2.lock().ram[123], rex.vc2.lock().ram[123]), (0x4567, 0x1234));
        machine.stop();
        primary.apply_display_resolution(crate::dev::ng1::vc2_timings::NewportResolution::Res1280x960);
        assert_eq!(visible(&primary).1, 960);
        assert_eq!(primary_state.lock().rgb[0], 0xaabb_ccdd);
        {
            let data = state.lock();
            assert_eq!((data.rgb[0], data.aux[0]), (0x1234_5678, 0x89ab_cdef));
            assert_eq!((data.observations, data.restores), (2, 2));
        }
        let rgb_path = snapshot.join("rex3_head1_rgb.bin");
        let rgb_file = std::fs::read(&rgb_path).unwrap();
        machine.ci_restore("m7-head1").unwrap();
        std::fs::remove_file(&rgb_path).unwrap();
        machine.stop();
        unsafe { (*rex.context.get()).colorred = 0x111111; }
        {
            let mut data = state.lock();
            data.rgb[0] = 0;
            data.aux[0] = 0;
        }
        machine.ci_rollback().unwrap();
        machine.stop();
        assert_eq!(unsafe { (*rex.context.get()).colorred }, 0x123456);
        {
            let data = state.lock();
            assert_eq!((data.rgb[0], data.aux[0]), (0x1234_5678, 0x89ab_cdef));
            assert_eq!((data.observations, data.restores), (2, 4));
        }
        assert!(machine.load_snapshot_paused("m7-head1").is_err());
        std::fs::write(&rgb_path, rgb_file).unwrap();
        let state_path = snapshot.join("rex3_head1.bin");
        let state_file = std::fs::read(&state_path).unwrap();
        std::fs::remove_file(&state_path).unwrap();
        assert!(machine.load_snapshot_paused("m7-head1").is_err());
        assert_eq!(state.lock().restores, 4);
        std::fs::write(&state_path, state_file).unwrap();
        let chunks_path = snapshot.join("chunks.bin");
        let chunks_file = std::fs::read(&chunks_path).unwrap();
        let mut chunks: crate::snapshot::ChunksManifest = postcard::from_bytes(&chunks_file).unwrap();
        chunks.framebuffer_chunks = None;
        std::fs::write(&chunks_path, postcard::to_allocvec(&chunks).unwrap()).unwrap();
        assert!(machine.load_snapshot_paused("m7-head1").unwrap_err().contains("no framebuffer"));
        assert_eq!(primary_state.lock().restores, 4);
        machine.reset();
        assert_eq!(visible(&primary), (1024, 768));
        assert_eq!(visible(&rex), (1024, 768));
        assert!(primary_state.lock().rgb.iter().all(|&word| word == 0));
        assert!(state.lock().rgb.iter().all(|&word| word == 0));
        std::fs::write(&chunks_path, &chunks_file).unwrap();
        let mut chunks: crate::snapshot::ChunksManifest =
            postcard::from_bytes(&std::fs::read(&chunks_path).unwrap()).unwrap();
        chunks.framebuffer_chunks = Some((Vec::new(), Vec::new()));
        std::fs::write(&chunks_path, postcard::to_allocvec(&chunks).unwrap()).unwrap();
        let error = machine.load_snapshot_paused("m7-head1").unwrap_err();
        assert!(error.contains("resident RGB seed has 0 words"), "{error}");
        assert!(primary.start_resident_processor().is_err());
        machine.reset();
        std::fs::write(&chunks_path, chunks_file).unwrap();
        std::fs::remove_file(&rgb_path).unwrap();
        std::fs::create_dir(&rgb_path).unwrap();
        assert!(machine.save_snapshot("m7-head1").is_err());
        assert!(snapshot.join("snapshot_incomplete").exists());
        let reset_counts = (primary_state.lock().resets, state.lock().resets);
        assert!(machine.load_snapshot_paused("m7-head1").unwrap_err().contains("incomplete"));
        assert_eq!((primary_state.lock().resets, state.lock().resets), reset_counts);
        std::fs::remove_dir(&rgb_path).unwrap();
        machine.save_snapshot("m7-head1").unwrap();
        assert!(!snapshot.join("snapshot_incomplete").exists());
        machine.load_snapshot_paused("m7-head1").unwrap();

        let lost = ResidentError::DeviceLost("injected snapshot completion failure".into());
        primary_state.lock().status_error = Some(lost.clone());
        assert!(machine.save_snapshot("m7-head1").is_err());
        assert!(!snapshot.join("snapshot_incomplete").exists(), "the capture failed before the directory was touched");
        assert_eq!(primary.resident_fault(), Some(lost.clone()));
        assert_eq!(primary.start_resident_processor(), Err(lost));
        assert_no_cpu_plane_access(&primary);
        assert_no_cpu_plane_access(&rex);
        machine.stop();
        drop(machine);
        drop(head1_guards);
        drop(primary_guards);
    });
}

#[test]
fn resident_direct_rejects_cpu_plane_debug_overlays() {
    on_large_stack(|| {
        let (_owner, rex, _state, _rgb_guard, _aux_guard) = fixture();
        *rex.resident_display_mode.lock() = Some(ResidentDisplayMode::DirectWgpu);

        for (command, args) in [
            ("rex", &["cmap", "on"][..]),
            ("disp", &["debug", "on"][..]),
        ] {
            let error = rex.execute_command(command, args, Box::new(io::sink())).unwrap_err();
            assert!(error.contains("unavailable in resident direct mode"), "{error}");
        }
        assert!(!rex.show_cmap.load(Ordering::Relaxed));
        assert!(!rex.show_disp_debug.load(Ordering::Relaxed));

        #[cfg(feature = "developer")]
        {
            let error = rex.execute_command("draw", &["debug", "on"], Box::new(io::sink())).unwrap_err();
            assert!(error.contains("unavailable in resident direct mode"), "{error}");
            assert!(!rex.draw_debug.load(Ordering::Relaxed));
        }
    });
}

#[test]
fn resident_lifecycle_invalid_inputs_cannot_mix_staged_snapshots() {
    on_large_stack(|| {
        let (_owner, rex, data, _rgb_guard, _aux_guard) = fixture();
        let registers = rex.save_state();
        let rgb = data.lock().rgb.clone();
        let aux = data.lock().aux.clone();
        unsafe { (*rex.context.get()).colorred = 0x7777; }

        let mut bad = registers.clone();
        bad["vc2"]["regs"].as_array_mut().unwrap()[0] = toml::Value::String("0x00010000".into());
        rex.restore_framebuffers_inmem(&rgb, &aux);
        assert!(rex.load_state(&bad).is_err());
        assert!(rex.start_resident_processor().is_err());
        assert_eq!(unsafe { (*rex.context.get()).colorred }, 0x7777);
        assert_eq!(data.lock().restores, 0);

        rex.load_state(&registers).unwrap();
        assert!(rex.stage_resident_planes(&rgb[..rgb.len() - 1], &aux).is_err());
        assert!(rex.start_resident_processor().is_err());
        rex.restore_framebuffers_inmem(&rgb, &aux);
        assert!(rex.start_resident_processor().is_err(), "rejected state cannot be paired with later planes");
        rex.load_state(&registers).unwrap();
        assert_eq!(data.lock().restores, 1);
        assert_eq!(rex.save_state(), registers);

        rex.load_state(&registers).unwrap();
        assert!(rex.load_state(&registers).is_err(), "a duplicate register half is ambiguous");
        rex.restore_framebuffers_inmem(&rgb, &aux);
        assert!(rex.start_resident_processor().is_err());
        rex.load_state(&registers).unwrap();
        assert_eq!(data.lock().restores, 2);

        rex.restore_framebuffers_inmem(&rgb, &aux);
        assert!(rex.stage_resident_planes(&rgb, &aux).is_err(), "a duplicate plane half is ambiguous");
        rex.load_state(&registers).unwrap();
        assert!(rex.start_resident_processor().is_err());
        rex.restore_framebuffers_inmem(&rgb, &aux);
        assert_eq!(data.lock().restores, 3);

        let mut bad = registers.clone();
        bad["xmap0"]["config"] = toml::Value::String("0x00000100".into());
        assert!(rex.load_state(&bad).is_err());
        bad = registers.clone();
        bad["context"]["hostcnt"] = toml::Value::String("0x00000009".into());
        assert!(rex.load_state(&bad).is_err());
        assert_eq!(rex.save_state(), registers);
        assert_no_cpu_plane_access(&rex);
    });
}

#[test]
fn resident_lifecycle_device_failures_are_sticky_and_never_publish_partial_state() {
    on_large_stack(|| {
        for stage in ["health", "observation", "restore", "reset"] {
            let (_owner, rex, data, _rgb_guard, _aux_guard) = fixture();
            let (sender, receiver) = mpsc::channel();
            rex.set_fatal_sender(sender);
            unsafe { (*rex.context.get()).colorred = 0x1111; }
            let registers = rex.save_state();
            unsafe { (*rex.context.get()).colorred = 0x2222; }
            let error = if stage == "health" || stage == "observation" {
                ResidentError::DeviceLost(format!("injected {stage}"))
            } else {
                ResidentError::Failed(format!("injected {stage}"))
            };
            match stage {
                "health" => {
                    data.lock().health_error = Some(error.clone());
                    assert_eq!(rex.snapshot_resident_planes(), Err(error.clone()));
                }
                "observation" => {
                    data.lock().observation_error = Some(error.clone());
                    assert_eq!(rex.snapshot_resident_planes(), Err(error.clone()));
                }
                "restore" => {
                    let rgb = data.lock().rgb.clone();
                    let aux = data.lock().aux.clone();
                    rex.load_state(&registers).unwrap();
                    data.lock().restore_error = Some(error.clone());
                    assert_eq!(rex.stage_resident_planes(&rgb, &aux), Err(error.clone()));
                }
                "reset" => {
                    data.lock().reset_error = Some(error.clone());
                    assert_eq!(rex.reset_resident_board(), Err(error.clone()));
                }
                _ => unreachable!(),
            }
            assert_eq!(rex.resident_fault(), Some(error.clone()));
            assert_eq!(unsafe { (*rex.context.get()).colorred }, 0x2222);
            assert!(rex.load_state(&registers).is_err());
            assert_eq!(rex.snapshot_resident_planes(), Err(error.clone()));
            assert_eq!(rex.reset_resident_board(), Err(error.clone()));
            assert_eq!(rex.start_resident_processor(), Err(error.clone()));
            let vc2_regs = rex.vc2.lock().regs;
            assert_eq!(rex.apply_resident_display_resolution(
                crate::dev::ng1::vc2_timings::NewportResolution::Res1024x768, true,
            ), Err(error.clone()));
            assert_eq!(rex.vc2.lock().regs, vc2_regs);
            assert!(rex.execute_command("rex", &["status"], Box::new(io::sink())).is_err());
            panics(|| { let _ = rex.save_state(); });
            panics(|| rex.restore_framebuffers_inmem(&[], &[]));
            assert!(rex.save_framebuffers_named(Path::new("target"), "resident-m7-failed").is_err());
            assert!(rex.load_framebuffers_named(Path::new("target"), "resident-m7-failed").is_err());
            assert!(copy_frame(&*rex, &mut Vec::new()).is_none());
            let notification = receiver.recv_timeout(Duration::from_secs(2)).unwrap();
            assert_eq!(notification.message, error.to_string());
            assert!(receiver.try_recv().is_err(), "only the first failure is reported");
            assert_no_cpu_plane_access(&rex);
        }
    });
}

#[test]
fn resident_lifecycle_disk_restore_requires_two_complete_planes() {
    on_large_stack(|| {
        struct Remove(std::path::PathBuf);
        impl Drop for Remove {
            fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
        }
        let path = Path::new("target").join(format!(
            "resident-m7-files-{}-{:?}", std::process::id(), thread::current().id(),
        ).replace(['(', ')'], ""));
        std::fs::create_dir_all(&path).unwrap();
        let _cleanup = Remove(path.clone());
        let (_owner, rex, state, _rgb_guard, _aux_guard) = fixture();
        let registers = rex.save_state();
        rex.save_framebuffers_named(&path, "board").unwrap();
        assert_eq!(state.lock().observations, 1);
        let rgb = std::fs::read(path.join("board_rgb.bin")).unwrap();
        let aux = std::fs::read(path.join("board_aux.bin")).unwrap();
        assert_eq!(&rgb[..4], &0x1234_5678u32.to_be_bytes());
        assert_eq!(&aux[..4], &0x89ab_cdefu32.to_be_bytes());
        unsafe { (*rex.context.get()).colorred = 0x7777; }
        rex.load_state(&registers).unwrap();
        std::fs::write(path.join("board_aux.bin"), &aux[..7]).unwrap();
        assert_eq!(rex.load_framebuffers_named(&path, "board").unwrap_err().kind(), io::ErrorKind::InvalidData);
        assert_eq!(unsafe { (*rex.context.get()).colorred }, 0x7777);
        assert!(rex.start_resident_processor().is_err());
        assert_eq!(state.lock().restores, 0);
        std::fs::remove_file(path.join("board_aux.bin")).unwrap();
        assert_eq!(rex.load_framebuffers_named(&path, "board").unwrap_err().kind(), io::ErrorKind::NotFound);
        std::fs::write(path.join("board_aux.bin"), &aux).unwrap();
        rex.load_framebuffers_named(&path, "board").unwrap();
        assert!(rex.start_resident_processor().is_err());
        rex.load_state(&registers).unwrap();
        assert_eq!(rex.save_state(), registers);
        assert_eq!(state.lock().restores, 1);

        {
            let mut data = state.lock();
            data.rgb[0] = 0x0102_0304;
            data.aux[0] = 0x0506_0708;
        }
        let aux_path = path.join("board_aux.bin");
        std::fs::remove_file(&aux_path).unwrap();
        std::fs::create_dir(&aux_path).unwrap();
        assert!(rex.save_framebuffers_named(&path, "board").is_err());
        assert!(path.join("board_resident_incomplete").exists());
        std::fs::remove_dir(&aux_path).unwrap();
        std::fs::write(&aux_path, &aux).unwrap();
        assert_eq!(&std::fs::read(path.join("board_rgb.bin")).unwrap()[..4], &0x0102_0304u32.to_be_bytes());
        rex.load_state(&registers).unwrap();
        assert_eq!(rex.load_framebuffers_named(&path, "board").unwrap_err().kind(), io::ErrorKind::InvalidData);
        assert!(rex.start_resident_processor().is_err());
        let (rgb, aux) = {
            let data = state.lock();
            (data.rgb.clone(), data.aux.clone())
        };
        rex.restore_framebuffers_inmem(&rgb, &aux);
        assert!(rex.start_resident_processor().is_err());
        rex.load_state(&registers).unwrap();
        rex.save_framebuffers_named(&path, "board").unwrap();
        assert!(!path.join("board_resident_incomplete").exists());
        rex.load_state(&registers).unwrap();
        rex.load_framebuffers_named(&path, "board").unwrap();
        assert_eq!(state.lock().restores, 3);
        assert_eq!(state.lock().observations, 3);
        {
            let data = state.lock();
            assert_eq!((data.rgb[0], data.aux[0]), (0x0102_0304, 0x0506_0708));
        }
        assert_no_cpu_plane_access(&rex);
    });
}
