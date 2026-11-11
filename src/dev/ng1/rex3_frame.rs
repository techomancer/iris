//! Test-only resident frame cuts, direct presentation and composed capture.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex, MutexGuard};

use super::{RasterMode, Rex3, Renderer, GFIFO_RESIDENT_FRAME, STATUS_VRINT};
use crate::dev::ng1::wgpu::direct::{CaptureFault, Ng1ResidentScanout, ResidentFrameStamp, ResidentScanoutTraffic};
use crate::dev::ng1::wgpu::resident::{Position, ResidentError};
use crate::dev::ng1::wgpu::Ng1Scanout;
use crate::disp::{BarStats, StatusBar, StatusBarTexture, FB_STRIDE};
use crate::debug_overlay::DebugOverlay;
use crate::wgpu::Gpu;

const FRAME_WAIT: Duration = Duration::from_millis(50);

struct Marker {
    id: u64,
    reached: Option<Result<(Position, usize), ResidentError>>,
}

#[derive(Default)]
struct MarkerState {
    next: u64,
    active: Option<Marker>,
}

/// One marker at a time. The processor acknowledges the marker and then
/// waits without holding the raster lock until the display releases it.
#[derive(Default)]
pub(super) struct ResidentFrameControl {
    active: AtomicBool,
    serial: Mutex<()>,
    marker: Mutex<MarkerState>,
    changed: Condvar,
    requested: AtomicU64,
    completed: AtomicU64,
    screenshots: AtomicU64,
    screenshots_completed: AtomicU64,
    captured: AtomicBool,
    error: Mutex<Option<ResidentError>>,
}

#[derive(Clone, Copy)]
struct Requests {
    frame: u64,
    screenshot: u64,
    needs_frame: bool,
    needs_screenshot: bool,
}

impl ResidentFrameControl {
    pub(super) fn activate(&self) {
        self.requested.store(0, Ordering::Relaxed);
        self.completed.store(0, Ordering::Relaxed);
        self.screenshots.store(0, Ordering::Relaxed);
        self.screenshots_completed.store(0, Ordering::Relaxed);
        self.captured.store(false, Ordering::Relaxed);
        *self.error.lock() = None;
        self.active.store(true, Ordering::Release);
    }

    pub(super) fn is_active(&self) -> bool {
        self.active.load(Ordering::Acquire)
    }

    pub(super) fn request_frame(&self) {
        self.requested.fetch_add(1, Ordering::AcqRel);
    }

    pub(super) fn request_screenshot(&self) {
        self.screenshots.fetch_add(1, Ordering::AcqRel);
    }

    pub(super) fn frame_pending(&self) -> bool {
        self.requested.load(Ordering::Acquire) != self.completed.load(Ordering::Acquire)
    }

    pub(super) fn screenshot_pending(&self) -> bool {
        self.screenshots.load(Ordering::Acquire) != self.screenshots_completed.load(Ordering::Acquire)
    }

    pub(super) fn frame_available(&self) -> bool {
        self.captured.load(Ordering::Acquire) && !self.frame_pending() && !self.screenshot_pending()
    }

    fn requests(&self) -> Requests {
        let frame = self.requested.load(Ordering::Acquire);
        let screenshot = self.screenshots.load(Ordering::Acquire);
        Requests {
            frame,
            screenshot,
            needs_frame: frame != self.completed.load(Ordering::Acquire),
            needs_screenshot: screenshot != self.screenshots_completed.load(Ordering::Acquire),
        }
    }

    fn complete(&self, requests: Requests) {
        self.completed.store(requests.frame, Ordering::Release);
        self.screenshots_completed.store(requests.screenshot, Ordering::Release);
        self.captured.store(true, Ordering::Release);
    }

    fn invalidate(&self) {
        self.captured.store(false, Ordering::Release);
    }

    pub(super) fn wake(&self) {
        self.changed.notify_all();
    }

    pub(crate) fn last_error(&self) -> Option<ResidentError> {
        self.error.lock().clone()
    }
}

struct FrameCut<'a> {
    control: &'a ResidentFrameControl,
    id: u64,
    position: Position,
    topscan: usize,
    requests: Requests,
    _serial: MutexGuard<'a, ()>,
}

impl Drop for FrameCut<'_> {
    fn drop(&mut self) {
        let mut state = self.control.marker.lock();
        if state.active.as_ref().is_some_and(|marker| marker.id == self.id) {
            state.active = None;
            self.control.changed.notify_all();
        }
    }
}

impl Rex3 {
    fn resident_display_fence_complete(&self) -> bool {
        let target = self.xmap_fence.load(Ordering::Acquire);
        self.gfifo_fence.load(Ordering::Acquire).wrapping_sub(target) < 0x8000_0000
    }

    /// Unlike the legacy refresh loop, a timeout does not advance this fence.
    pub(crate) fn wait_resident_display_fence(&self, timeout: Duration) -> Result<(), ResidentError> {
        let target = self.xmap_fence.load(Ordering::Acquire);
        let deadline = Instant::now() + timeout;
        let backoff = crossbeam_utils::Backoff::new();
        loop {
            let current = self.gfifo_fence.load(Ordering::Acquire);
            if current.wrapping_sub(target) < 0x8000_0000 {
                return Ok(());
            }
            if !self.running.load(Ordering::Acquire) {
                return Err(ResidentError::InvalidState("resident GFIFO is stopped"));
            }
            if Instant::now() >= deadline {
                return Err(ResidentError::Timeout("display fence"));
            }
            backoff.snooze();
        }
    }

    fn resident_frame_cut(&self, timeout: Duration) -> Result<FrameCut<'_>, ResidentError> {
        let control = &self.resident_frame;
        let deadline = Instant::now() + timeout;
        let serial = control.serial.try_lock_for(timeout)
            .ok_or(ResidentError::Timeout("frame marker serialization"))?;
        if !self.running.load(Ordering::Acquire) {
            return Err(ResidentError::InvalidState("resident GFIFO is stopped"));
        }
        let requests = control.requests();
        let id = {
            let mut state = control.marker.lock();
            state.next = state.next.checked_add(1)
                .ok_or(ResidentError::InvalidState("resident frame markers exhausted"))?;
            let id = state.next;
            state.active = Some(Marker { id, reached: None });
            id
        };
        while !self.gfifo_try_push(GFIFO_RESIDENT_FRAME, id) {
            if !self.running.load(Ordering::Acquire) || Instant::now() >= deadline {
                let mut state = control.marker.lock();
                if state.active.as_ref().is_some_and(|marker| marker.id == id) {
                    state.active = None;
                }
                control.changed.notify_all();
                return Err(if self.running.load(Ordering::Acquire) {
                    ResidentError::Timeout("frame marker publication")
                } else {
                    ResidentError::InvalidState("resident GFIFO is stopped")
                });
            }
            std::thread::yield_now();
        }
        let mut state = control.marker.lock();
        loop {
            if let Some(result) = state.active.as_ref()
                .filter(|marker| marker.id == id)
                .and_then(|marker| marker.reached.clone())
            {
                match result {
                    Ok((position, topscan)) => {
                        drop(state);
                        return Ok(FrameCut { control, id, position, topscan, requests, _serial: serial });
                    }
                    Err(error) => {
                        state.active = None;
                        control.changed.notify_all();
                        return Err(error);
                    }
                }
            }
            if !self.running.load(Ordering::Acquire) || Instant::now() >= deadline {
                if state.active.as_ref().is_some_and(|marker| marker.id == id) {
                    state.active = None;
                }
                control.changed.notify_all();
                return Err(if self.running.load(Ordering::Acquire) {
                    ResidentError::Timeout("frame marker")
                } else {
                    ResidentError::InvalidState("resident GFIFO is stopped")
                });
            }
            control.changed.wait_for(&mut state, deadline.saturating_duration_since(Instant::now()));
        }
    }

    pub(super) fn process_resident_frame_marker(&self, id: u64) {
        let control = &self.resident_frame;
        if !control.marker.lock().active.as_ref().is_some_and(|marker| marker.id == id) {
            return;
        }
        // Only the GFIFO consumer reads the context here; later entries have
        // not run. No CPU framebuffer reference is formed.
        let result = self.resident_status().map(|status| {
            let topscan = unsafe { (*self.context.get()).topscan as usize };
            (status.accepted, topscan)
        });
        if let Err(error) = &result {
            self.fail_resident(error.clone());
        }
        let mut state = control.marker.lock();
        if let Some(marker) = state.active.as_mut().filter(|marker| marker.id == id) {
            marker.reached = Some(result);
            control.changed.notify_all();
            while state.active.as_ref().is_some_and(|marker| marker.id == id)
                && self.running.load(Ordering::Acquire)
            {
                control.changed.wait_for(&mut state, FRAME_WAIT);
            }
        }
    }

    /// Opt in only from tests, after starting the resident GFIFO processor.
    pub(crate) fn start_resident_display(&self, gpu: Arc<Gpu>) -> Result<(), ResidentError> {
        if !self.running.load(Ordering::Acquire) {
            return Err(ResidentError::InvalidState("resident GFIFO is stopped"));
        }
        let mut thread_slot = self.refresh_thread.lock();
        if thread_slot.is_some() {
            return Err(ResidentError::InvalidState("resident display already started"));
        }
        let mut display = ResidentDisplay::new(self, &gpu)?;
        if !self.running.load(Ordering::Acquire) {
            return Err(ResidentError::InvalidState("resident GFIFO is stopped"));
        }
        let rex = unsafe { std::mem::transmute::<&Rex3, &'static Rex3>(self) };
        let handle = std::thread::Builder::new().name("REX3-Resident-Display-Test".into())
            .spawn(move || {
                if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| display.run(rex))).is_err() {
                    rex.fail_resident(ResidentError::Failed("resident display worker panicked".into()));
                }
            })
            .map_err(|error| ResidentError::Failed(format!("start resident display: {error}")))?;
        *thread_slot = Some(handle);
        Ok(())
    }

    fn resident_vblank(&self) {
        self.config.status.fetch_or(STATUS_VRINT, Ordering::Relaxed);
        {
            let mut vc2 = self.vc2.lock();
            let working = crate::dev::ng1::vc2::VC2_REG_WORKING_CURSOR_Y as usize;
            let y = vc2.regs[crate::dev::ng1::vc2::VC2_REG_CURSOR_Y_LOC as usize];
            if vc2.regs[working] != y {
                vc2.regs[working] = y;
                vc2.dirty = true;
            }
        }
        if let Some(irq) = self.vblank_cb.lock().clone() {
            irq.set(true);
        }
    }
}

pub(super) fn validate_presenter(renderer: &dyn Renderer, gpu: &Arc<Gpu>) -> Result<(), ResidentError> {
    if !renderer.takes_gpu_frames() {
        return Err(ResidentError::Unsupported("resident frames require a wgpu presenter"));
    }
    let device = renderer.resident_gpu_device()
        .ok_or(ResidentError::Unsupported("presenter did not identify its GPU"))?;
    if !Arc::ptr_eq(device, gpu) {
        return Err(ResidentError::DeviceMismatch);
    }
    Ok(())
}

/// Owns the composed output and its staging buffer across refreshes.
pub(crate) struct ResidentDisplay {
    scanout: Ng1ResidentScanout,
    overlay: DebugOverlay,
    status: StatusBar,
    sbtex: StatusBarTexture,
    capture: Vec<u32>,
    save_screenshots: bool,
    resize_pending: bool,
}

impl ResidentDisplay {
    pub(crate) fn new(rex: &Rex3, gpu: &Arc<Gpu>) -> Result<Self, ResidentError> {
        if let Some(renderer) = rex.renderer.lock().as_deref() {
            validate_presenter(renderer, gpu)?;
        }
        let scanout = Ng1Scanout::new_resident(gpu, &rex)?;
        Ok(Self {
            scanout,
            overlay: DebugOverlay::new(),
            status: StatusBar::new(),
            sbtex: StatusBarTexture::new(),
            capture: Vec::new(),
            save_screenshots: true,
            resize_pending: true,
        })
    }

    pub(crate) fn traffic(&self) -> ResidentScanoutTraffic {
        self.scanout.traffic()
    }

    pub(crate) fn fail_next_capture(&mut self, fault: CaptureFault) {
        self.scanout.fail_next_capture(fault);
    }

    pub(crate) fn suppress_screenshot_files(&mut self) {
        self.save_screenshots = false;
    }

    /// The hook runs at the FIFO cut, before the raster lock is taken. It is
    /// used to deterministically queue later draws in the ordering tests.
    pub(crate) fn render(
        &mut self,
        rex: &Rex3,
        timeout: Duration,
        at_marker: impl FnOnce(Position),
    ) -> Result<ResidentFrameStamp, ResidentError> {
        rex.wait_resident_display_fence(timeout)?;
        let cut = rex.resident_frame_cut(timeout)?;
        let requests = cut.requests;
        let capture = requests.needs_frame || requests.needs_screenshot
            || rex.capture_users.load(Ordering::Acquire) != 0;
        at_marker(cut.position);

        let mut screen = rex.screen.lock();
        rex.resident_frame.invalidate();
        screen.topscan = cut.topscan;
        screen.status_bar_only = false;
        let resized = screen.refresh_resident_display_state(
            &rex.vc2,
            &rex.xmap0,
            &rex.cmap0,
            &rex.bt445,
            &rex.diag,
        );
        self.resize_pending |= resized;
        // A mode write after the initial wait may have published a later
        // fence. Its GFIFO entry cannot pass this marker, so defer rather
        // than wait here or compose state not ordered by the cut.
        if !rex.resident_display_fence_complete() {
            return Err(ResidentError::Timeout("display fence after frame marker"));
        }
        self.overlay.show_cmap = rex.show_cmap.load(Ordering::Relaxed);
        self.overlay.show_disp_debug = rex.show_disp_debug.load(Ordering::Relaxed);
        self.overlay.show_draw_debug = rex.draw_debug_active();
        if self.overlay.active() {
            return Err(ResidentError::Unsupported("resident debug overlays require CPU planes"));
        }
        let mut renderer = rex.renderer.lock();
        if let Some(renderer) = renderer.as_deref_mut() {
            validate_presenter(renderer, self.scanout.gpu())?;
            if self.resize_pending {
                renderer.resize(screen.width, screen.height);
                self.resize_pending = false;
            }
        } else {
            self.resize_pending = true;
        }

        let stats: BarStats = rex.status.snapshot(Instant::now(), rex.gfifo.len());

        // The cut keeps the consumer from accepting later draws; the raster
        // lock also excludes every other submitter until both consumers have
        // queued their work using the same output buffer and device.
        let mut mode = rex.raster.lock();
        let RasterMode::Resident(raster) = &mut *mode else {
            return Err(ResidentError::InvalidState("resident raster is not active"));
        };
        if raster.status()?.accepted != cut.position {
            return Err(ResidentError::InvalidState("resident frame marker stream cut changed"));
        }

        raster.submit()?;
        let frame = self.scanout.compose(&screen.display_state(), raster)?;
        if frame.position != cut.position {
            return Err(ResidentError::InvalidState("resident scanout used the wrong stream cut"));
        }
        let stamp = frame.stamp();
        let buffer = frame.buffer.clone();
        let gpu = Arc::clone(frame.gpu);
        if let Some(renderer) = renderer.as_deref_mut() {
            validate_presenter(renderer, &gpu)?;
            renderer.present_resident_gpu_frame(
                &buffer, stamp.width, stamp.height, &mut screen, &mut self.overlay,
                &mut self.status, &mut self.sbtex, &stats,
            ).map_err(ResidentError::Failed)?;
        }
        if capture {
            let (width, height) = (stamp.width as usize, stamp.height as usize);
            self.capture.resize(width * height, 0);
            self.scanout.capture(stamp, raster, &mut self.capture)?;
            for y in 0..height {
                let row = y * FB_STRIDE;
                for (dst, &src) in screen.rgba[row..row + width]
                    .iter_mut().zip(&self.capture[y * width..(y + 1) * width])
                {
                    *dst = 0xff00_0000 | src.swap_bytes() >> 8;
                }
            }
            screen.frame_seq = screen.frame_seq.wrapping_add(1);
            rex.resident_frame.complete(requests);
            if requests.needs_screenshot && self.save_screenshots {
                let pixels = screen.rgba.clone();
                let n = rex.screenshot_counter.fetch_add(1, Ordering::Relaxed);
                let path = format!("screenshot_{n:04}.png");
                std::thread::spawn(move || {
                    match crate::disp::save_screenshot(&path, &pixels, width, height) {
                        Ok(()) => println!("iris: screenshot saved to {path}"),
                        Err(error) => println!("iris: screenshot failed: {error}"),
                    }
                });
            }
        }
        Ok(stamp)
    }

    fn run(&mut self, rex: &Rex3) {
        let period = Duration::from_micros(16667);
        while rex.running.load(Ordering::Acquire) {
            let start = Instant::now();
            let requests = rex.resident_frame.requests();
            let needs_frame = requests.needs_frame || requests.needs_screenshot
                || rex.capture_users.load(Ordering::Acquire) != 0;
            if needs_frame || rex.renderer.lock().is_some() {
                let result = self.render(rex, FRAME_WAIT, |_| {});
                *rex.resident_frame.error.lock() = result.as_ref().err().cloned();
                if let Err(error) = result {
                    let fatal = rex.resident_fault().or_else(|| {
                        self.scanout.gpu().check().err().map(|message| {
                            if message.starts_with("wgpu device lost (") {
                                ResidentError::DeviceLost(message)
                            } else {
                                ResidentError::Failed(message)
                            }
                        })
                    }).or_else(|| matches!(error, ResidentError::DeviceLost(_)).then_some(error));
                    if let Some(fatal) = fatal {
                        rex.fail_resident(fatal);
                        break;
                    }
                }
            }
            rex.resident_vblank();
            let elapsed = start.elapsed();
            if elapsed < period {
                std::thread::sleep(period - elapsed);
            }
        }
        rex.display_stop.finish(&rex.renderer);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dev::ng1::rex3::{Rex3Context, GFIFO_DEPTH, REX3_WRMASK};
    use crate::dev::ng1::wgpu::resident::{ResidentBackend, ResidentStatus};
    use crate::gfx_display::{fresh_frame, GfxDisplay};
    use crate::traits::Device;

    struct Probe;

    impl ResidentBackend for Probe {
        fn enqueue(&mut self, _context: &mut Rex3Context) -> Result<(), ResidentError> { Ok(()) }
        fn submit(&mut self) -> Result<(), ResidentError> { Ok(()) }
        fn status(&mut self) -> Result<ResidentStatus, ResidentError> { Ok(ResidentStatus::for_test()) }
    }

    #[test]
    fn resident_frame_requests_keep_newer_tickets_pending() {
        let control = ResidentFrameControl::default();
        control.activate();
        assert!(!control.frame_available());
        control.request_frame();
        control.request_screenshot();
        let first = control.requests();
        control.request_frame();
        control.request_screenshot();
        control.complete(first);
        assert!(control.frame_pending());
        assert!(control.screenshot_pending());
        assert!(!control.frame_available());
        let second = control.requests();
        control.complete(second);
        assert!(!control.frame_pending());
        assert!(!control.screenshot_pending());
        assert!(control.frame_available());
        control.invalidate();
        assert!(!control.frame_available(), "an unpublished display state is not a captured frame");
    }

    #[test]
    fn resident_frame_timeouts_never_publish_fences_or_stale_pixels() {
        std::thread::Builder::new().stack_size(64 * 1024 * 1024).spawn(|| {
            let mut rex = Box::new(Rex3::new(crate::disp::StatusSource::detached()));
            rex.attach_resident_raster(Probe).unwrap();
            rex.running.store(true, Ordering::Release);
            rex.xmap_fence.store(9, Ordering::Release);
            rex.gfifo_fence.store(8, Ordering::Release);
            rex.request_frame();
            rex.request_screenshot();
            {
                let mut screen = rex.screen.lock();
                screen.width = 1;
                screen.height = 1;
                screen.rgba[0] = 0xff12_3456;
                screen.frame_seq = 7;
            }
            assert_eq!(rex.wait_resident_display_fence(Duration::ZERO), Err(ResidentError::Timeout("display fence")));
            assert_eq!(rex.gfifo_fence.load(Ordering::Acquire), 8);
            assert!(rex.frame_pending() && rex.screenshot_pending());
            let mut pixels = vec![0xfeed_beef];
            assert_eq!(fresh_frame(&*rex, &mut pixels, Duration::ZERO), None);
            assert_eq!(pixels, [0xfeed_beef]);
            assert_eq!(rex.screen.lock().frame_seq, 7);

            rex.gfifo_fence.store(9, Ordering::Release);
            assert!(matches!(rex.resident_frame_cut(Duration::from_millis(10)),
                Err(ResidentError::Timeout("frame marker"))));
            assert!(rex.resident_frame.marker.lock().active.is_none());
            assert!(rex.frame_pending() && rex.screenshot_pending());
            let (addr, cancelled) = rex.gfifo.peek().expect("cancelled marker remains in FIFO");
            assert_eq!(addr, GFIFO_RESIDENT_FRAME);
            rex.gfifo.consume();
            rex.gfifo.flush_head();
            rex.process_resident_frame_marker(cancelled);
            std::thread::scope(|scope| {
                let rex = &*rex;
                let worker = scope.spawn(|| {
                    let deadline = Instant::now() + Duration::from_secs(2);
                    loop {
                        if let Some((addr, id)) = rex.gfifo.peek() {
                            assert_eq!(addr, GFIFO_RESIDENT_FRAME);
                            rex.process_resident_frame_marker(id);
                            rex.gfifo.consume();
                            rex.gfifo.flush_head();
                            break;
                        }
                        assert!(Instant::now() < deadline, "retry marker did not arrive");
                        std::thread::yield_now();
                    }
                });
                let cut = rex.resident_frame_cut(Duration::from_secs(2)).expect("retry marker");
                assert_ne!(cut.id, cancelled);
                drop(cut);
                worker.join().unwrap();
            });
            rex.running.store(false, Ordering::Release);
            rex.gfifo.reset();
        }).unwrap().join().unwrap();
    }

    #[test]
    fn resident_device_loss_with_full_gfifo_stops_and_prevents_restart() {
        const CHILD: &str = "IRIS_RESIDENT_FULL_FIFO_LOSS_CHILD";
        if std::env::var_os(CHILD).is_some() {
            std::thread::Builder::new().stack_size(64 * 1024 * 1024).spawn(|| {
                let mut board = Box::new(Rex3::new(crate::disp::StatusSource::detached()));
                board.fb_rgb.get_mut().isolate().unwrap();
                board.fb_aux.get_mut().isolate().unwrap();
                board.fb_rgb.get_mut().reset_access_counts();
                board.fb_aux.get_mut().reset_access_counts();
                let (sender, receiver) = std::sync::mpsc::channel();
                board.set_fatal_sender(sender);
                board.attach_resident_raster(Probe).unwrap();
                let mut board = Arc::<Rex3>::from(board);
                let rgb_guard = unsafe { (&*board.fb_rgb.get()).protect() }.unwrap();
                let aux_guard = unsafe { (&*board.fb_aux.get()).protect() }.unwrap();
                board.start();
                let cut = board.resident_frame_cut(Duration::from_secs(2)).unwrap();
                let mut queued = 0;
                while board.gfifo.try_push(REX3_WRMASK, 0) { queued += 1; }
                assert!(queued >= GFIFO_DEPTH - 3, "the held marker must leave a full GFIFO");
                let first = ResidentError::DeviceLost("injected full-FIFO loss".into());
                board.fail_resident(first.clone());
                board.fail_resident(ResidentError::Failed("later failure".into()));
                let notification = receiver.recv_timeout(Duration::from_secs(2)).unwrap();
                assert_eq!(notification.message, first.to_string());

                let stopping = Arc::clone(&board);
                let worker = std::thread::spawn(move || stopping.stop());
                let deadline = Instant::now() + Duration::from_secs(10);
                while !worker.is_finished() {
                    assert!(Instant::now() < deadline, "stop hung behind the full GFIFO");
                    std::thread::sleep(Duration::from_millis(1));
                }
                worker.join().unwrap();
                drop(cut);
                assert!(!board.is_running());
                assert!(board.gfifo.is_empty());
                assert_eq!(board.resident_fault(), Some(first.clone()));
                assert_eq!(board.start_resident_processor(), Err(first.clone()));
                assert!(matches!(receiver.try_recv(), Err(std::sync::mpsc::TryRecvError::Empty)));
                assert_eq!(unsafe { (&*board.fb_rgb.get()).access_counts() }, Default::default());
                assert_eq!(unsafe { (&*board.fb_aux.get()).access_counts() }, Default::default());
                assert_eq!(Arc::get_mut(&mut board).unwrap().reset_resident_gpu(), Err(first));
                assert_eq!(unsafe { (&*board.fb_rgb.get()).access_counts() }, Default::default());
                assert_eq!(unsafe { (&*board.fb_aux.get()).access_counts() }, Default::default());
                drop(aux_guard);
                drop(rgb_guard);
            }).unwrap().join().unwrap();
            return;
        }

        let name = std::thread::current().name().unwrap().to_owned();
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", &name, "--nocapture", "--test-threads=1"])
            .env(CHILD, "1")
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn().expect("spawn full-FIFO device-loss test");
        let deadline = Instant::now() + Duration::from_secs(20);
        while child.try_wait().unwrap().is_none() {
            if Instant::now() >= deadline {
                child.kill().unwrap();
                let output = child.wait_with_output().unwrap();
                panic!("full-FIFO device-loss test timed out: {}", String::from_utf8_lossy(&output.stderr));
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        let output = child.wait_with_output().unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(output.status.success() && stdout.contains("1 passed; 0 failed"), "{stdout}\n{stderr}");
    }
}
