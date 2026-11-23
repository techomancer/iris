//! Resident frame cuts, direct presentation and composed capture.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex, MutexGuard};

use super::{RasterMode, Rex3, GFIFO_RESIDENT_FRAME};
use crate::display::{Composed, Want};
use crate::dev::ng1::wgpu::direct::{
    CaptureCount, Ng1ResidentScanout, ResidentFrameStamp, ResidentScanoutTraffic,
};
#[cfg(test)]
use crate::dev::ng1::wgpu::direct::CaptureFault;
use crate::dev::ng1::wgpu::resident::{
    Observation, ObservationReason, PlaneMask, Position, ResidentError,
};
use crate::dev::ng1::wgpu::telemetry::{duration_ns, FlushReason};
use crate::dev::ng1::wgpu::Ng1Scanout;
use crate::disp::{Rex3Screen, FB_STRIDE};
use crate::debug_overlay::DebugOverlay;
use crate::wgpu::Gpu;

const FRAME_WAIT: Duration = Duration::from_millis(50);

/// How long the refresh thread waits for the GFIFO to reach a frame cut
/// before it shows the last frame again and tries at the next retrace.
const CUT_WAIT: Duration = Duration::from_millis(8);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResidentDisplayMode {
    DirectWgpu,
    ReadbackCompatibility,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ResidentDisplayFrame {
    pub(crate) position: Position,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

struct Marker {
    id: u64,
    reached: Option<Result<(Position, usize), ResidentError>>,
}

#[derive(Default)]
struct MarkerState {
    next: u64,
    active: Option<Marker>,
}

#[derive(Default)]
struct CaptureTotals {
    direct: CaptureCount,
    compatibility: CaptureCount,
}

impl CaptureTotals {
    fn for_mode(&mut self, mode: ResidentDisplayMode) -> &mut CaptureCount {
        match mode {
            ResidentDisplayMode::DirectWgpu => &mut self.direct,
            ResidentDisplayMode::ReadbackCompatibility => &mut self.compatibility,
        }
    }
}

/// One marker at a time. The processor acknowledges the marker and then
/// waits without holding the raster lock until the display releases it.
///
/// Frame and screenshot requests are the display head's own (`HeadHandle`):
/// the refresh thread asks for a readback frame through [`Want`].
#[derive(Default)]
pub(super) struct ResidentFrameControl {
    active: AtomicBool,
    serial: Mutex<()>,
    marker: Mutex<MarkerState>,
    changed: Condvar,
    // Like raw observations, composed-capture totals survive resident resets.
    captures: Mutex<CaptureTotals>,
    error: Mutex<Option<ResidentError>>,
}

impl ResidentFrameControl {
    pub(super) fn activate(&self) {
        *self.error.lock() = None;
        self.active.store(true, Ordering::Release);
    }

    pub(super) fn is_active(&self) -> bool {
        self.active.load(Ordering::Acquire)
    }

    pub(super) fn quiescent(&self) -> bool {
        let Some(_serial) = self.serial.try_lock() else { return false; };
        self.marker.lock().active.is_none()
    }

    fn capture_count(&self, mode: ResidentDisplayMode) -> CaptureCount {
        *self.captures.lock().for_mode(mode)
    }

    fn capture_attempt(&self, mode: ResidentDisplayMode, bytes: u64) -> Result<(), ResidentError> {
        self.captures.lock().for_mode(mode).attempt(bytes)
    }

    fn capture_success(&self, mode: ResidentDisplayMode, bytes: u64) -> Result<(), ResidentError> {
        self.captures.lock().for_mode(mode).success(bytes)
    }

    fn capture_delta(&self, mode: ResidentDisplayMode, before: CaptureCount, after: CaptureCount) -> Result<(), ResidentError> {
        let mut totals = self.captures.lock();
        let total = totals.for_mode(mode);
        let add_delta = |current: u64, old: u64, new: u64| {
            new.checked_sub(old).and_then(|delta| current.checked_add(delta))
                .ok_or(ResidentError::InvalidState("composed capture counter decreased or overflowed"))
        };
        *total = CaptureCount {
            attempts: add_delta(total.attempts, before.attempts, after.attempts)?,
            attempted_bytes: add_delta(total.attempted_bytes, before.attempted_bytes, after.attempted_bytes)?,
            successes: add_delta(total.successes, before.successes, after.successes)?,
            successful_bytes: add_delta(total.successful_bytes, before.successful_bytes, after.successful_bytes)?,
        };
        Ok(())
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
    pub(crate) fn resident_composed_captures(&self, mode: ResidentDisplayMode) -> CaptureCount {
        self.resident_frame.capture_count(mode)
    }

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
        let started = Instant::now();
        let control = &self.resident_frame;
        let deadline = Instant::now() + timeout;
        let serial = control.serial.try_lock_for(timeout)
            .ok_or(ResidentError::Timeout("frame marker serialization"))?;
        if !self.running.load(Ordering::Acquire) {
            return Err(ResidentError::InvalidState("resident GFIFO is stopped"));
        }
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
                        let cut = FrameCut { control, id, position, topscan, _serial: serial };
                        let elapsed = duration_ns(started.elapsed()).map_err(ResidentError::InvalidState)?;
                        self.resident_telemetry.lock().cut(elapsed);
                        return Ok(cut);
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

    /// Install the fixed resident display mode, after the GFIFO processor:
    /// the shared refresh thread composes it from now on (`Rex3Head`).
    pub(crate) fn start_resident_display(
        &self,
        gpu: Arc<Gpu>,
        mode: ResidentDisplayMode,
    ) -> Result<(), ResidentError> {
        let _lifecycle = self.resident_lifecycle.write();
        if !self.running.load(Ordering::Acquire) {
            return Err(ResidentError::InvalidState("resident GFIFO is stopped"));
        }
        self.check_resident_health(false)?;
        let mut slot = self.resident_display.lock();
        if slot.is_some() {
            return Err(ResidentError::InvalidState("resident display already started"));
        }
        let display = match ResidentDisplay::new(self, &gpu, mode) {
            Ok(display) => display,
            Err(error) => {
                if matches!(error, ResidentError::DeviceLost(_) | ResidentError::Failed(_)) {
                    self.fail_resident(error.clone());
                }
                return Err(error);
            }
        };
        *slot = Some(display);
        drop(slot);
        // The shared refresh thread drives the head, as for the CPU raster:
        // the guest's timing, and frames composed on the GPU
        // (`Rex3Head::compose`).
        let rex3 = unsafe { std::mem::transmute::<&Rex3, &'static Rex3>(self) };
        let shim = self.head_shim.get_or_init(|| Arc::new(super::Rex3Head(rex3)));
        let head: Arc<dyn crate::display::Head> = shim.clone();
        self.head.attach(Arc::downgrade(&head));
        self.head.start();
        Ok(())
    }

    /// Take the resident display down (`stop`).
    pub(crate) fn stop_resident_display(&self) {
        *self.resident_display.lock() = None;
    }

    /// One frame of the resident display, for the refresh thread.
    pub(super) fn resident_compose(&self, screen: &mut Rex3Screen, want: Want) -> Composed {
        let mut slot = self.resident_display.lock();
        let Some(display) = slot.as_mut() else { return Composed::NotReady };
        match display.compose(self, screen, want, CUT_WAIT, |_| {}) {
            Ok(composed) => composed,
            Err(error) => {
                self.resident_display_failed(&display.gpu, error);
                Composed::NotReady
            }
        }
    }

    /// A frame that could not be composed: a timeout waits for the next
    /// retrace; a lost or failed device stops the board.
    fn resident_display_failed(&self, gpu: &Gpu, error: ResidentError) {
        *self.resident_frame.error.lock() = Some(error.clone());
        let fatal = self.resident_fault().or_else(|| {
            gpu.check().err().map(|message| {
                if message.starts_with("wgpu device lost (") {
                    ResidentError::DeviceLost(message)
                } else {
                    ResidentError::Failed(message)
                }
            })
        }).or_else(|| matches!(error, ResidentError::DeviceLost(_)).then_some(error));
        if let Some(fatal) = fatal {
            self.fail_resident(fatal);
        }
    }

    /// Whether the direct mode shows the frame: its debug overlays need CPU
    /// planes it does not keep.
    pub(super) fn resident_direct(&self) -> bool {
        self.resident_display.lock().as_ref().is_some_and(|d| d.mode() == ResidentDisplayMode::DirectWgpu)
    }
}

fn observed_words(observation: &Observation, rgb: bool) -> Result<Vec<u32>, ResidentError> {
    let bytes = if rgb { observation.rgb() } else { observation.aux() }
        .ok_or(ResidentError::InvalidState("compatibility observation is missing a plane"))?;
    let mut chunks = bytes.chunks_exact(size_of::<u32>());
    let words = chunks
        .by_ref()
        .map(|bytes| u32::from_ne_bytes(bytes.try_into().expect("u32-sized chunk")))
        .collect();
    if !chunks.remainder().is_empty() {
        return Err(ResidentError::InvalidState("compatibility observation has a partial word"));
    }
    Ok(words)
}

/// Explicitly slow validation/compatibility path: read both authoritative
/// planes into the screen, for the renderer's compositor to compose as it
/// does the CPU planes.
struct ResidentCompatibilityDisplay {
    capture: CaptureCount,
    #[cfg(test)]
    capture_fault: Option<CaptureFault>,
}

impl ResidentCompatibilityDisplay {
    fn new() -> Self {
        Self {
            capture: CaptureCount::default(),
            #[cfg(test)]
            capture_fault: None,
        }
    }

    fn compose(
        &mut self,
        rex: &Rex3,
        screen: &mut Rex3Screen,
        want: Want,
        timeout: Duration,
        at_marker: impl FnOnce(Position),
    ) -> Result<Composed, ResidentError> {
        rex.wait_resident_display_fence(timeout)?;
        let cut = rex.resident_frame_cut(timeout)?;
        at_marker(cut.position);
        screen.topscan = cut.topscan;
        screen.prebuilt = false;
        if !rex.resident_display_fence_complete() {
            return Err(ResidentError::Timeout("display fence after frame marker"));
        }
        let mut mode = rex.raster.lock();
        let RasterMode::Resident(raster) = &mut *mode else {
            return Err(ResidentError::InvalidState("resident raster is not active"));
        };
        if raster.status()?.accepted != cut.position {
            return Err(ResidentError::InvalidState("resident frame marker stream cut changed"));
        }
        raster.submit_for(FlushReason::FrameCut)?;
        let observation = raster.observe(ObservationReason::Compatibility, PlaneMask::Both)?;
        if observation.position() != cut.position {
            return Err(ResidentError::InvalidState("compatibility observation used the wrong stream cut"));
        }
        drop(mode);
        let rgb = observed_words(&observation, true)?;
        let aux = observed_words(&observation, false)?;
        screen.fb_rgb.resize(rgb.len(), 0);
        screen.fb_aux.resize(aux.len(), 0);
        #[cfg(feature = "rex-dirty-upload")]
        let damage = {
            use crate::dev::ng1::rex3_dirty::{DamagePlane, DamageSnapshot};
            let mut damage = DamageSnapshot::default();
            damage.mark_full(DamagePlane::Rgb);
            damage.mark_full(DamagePlane::Aux);
            damage
        };
        screen.refresh(
            &rgb,
            &aux,
            #[cfg(feature = "rex-dirty-upload")]
            &damage,
            #[cfg(not(feature = "rex-dirty-upload"))]
            true,
            &rex.vc2,
            &rex.xmap0,
            &rex.cmap0,
            &rex.bt445,
            &rex.diag,
        );
        if !rex.resident_display_fence_complete() {
            return Err(ResidentError::Timeout("display fence after compatibility observation"));
        }
        if screen.width == 0 || screen.width > FB_STRIDE || screen.height == 0
            || screen.height > super::REX3_SCREEN_HEIGHT as usize
        {
            return Err(ResidentError::InvalidState("resident scanout dimensions exceed the Newport planes"));
        }
        if want.readback {
            // The refresh thread composes the planes into `rgba` for the
            // reader; this counts the frames read back for one.
            let bytes = (screen.width * screen.height * size_of::<u32>()) as u64;
            self.capture.attempt(bytes)?;
            rex.resident_frame.capture_attempt(ResidentDisplayMode::ReadbackCompatibility, bytes)?;
            #[cfg(test)]
            if self.capture_fault.take().is_some() {
                return Err(ResidentError::Failed("injected composed capture failure".into()));
            }
            self.capture.success(bytes)?;
            rex.resident_frame.capture_success(ResidentDisplayMode::ReadbackCompatibility, bytes)?;
        }
        Ok(Composed::Frame)
    }
}

/// Owns the composed output and its staging buffer across refreshes.
struct ResidentDirectDisplay {
    scanout: Ng1ResidentScanout,
    capture: Vec<u32>,
}

impl ResidentDirectDisplay {
    fn new(rex: &Rex3, gpu: &Arc<Gpu>) -> Result<Self, ResidentError> {
        let scanout = Ng1Scanout::new_resident(gpu, &rex)?;
        Ok(Self { scanout, capture: Vec::new() })
    }

    /// The hook runs at the FIFO cut, before the raster lock is taken. It is
    /// used to deterministically queue later draws in the ordering tests.
    ///
    /// The frame stays on the GPU for a presenter on the board's device
    /// ([`Want::gpu`]); a reader, or a renderer that composes on the CPU,
    /// gets it read back into the screen as finished pixels.
    fn compose(
        &mut self,
        rex: &Rex3,
        screen: &mut Rex3Screen,
        want: Want,
        timeout: Duration,
        at_marker: impl FnOnce(Position),
    ) -> Result<(Composed, ResidentFrameStamp), ResidentError> {
        rex.wait_resident_display_fence(timeout)?;
        let cut = rex.resident_frame_cut(timeout)?;
        at_marker(cut.position);
        screen.topscan = cut.topscan;
        screen.refresh_resident_display_state(
            &rex.vc2,
            &rex.xmap0,
            &rex.cmap0,
            &rex.bt445,
            &rex.diag,
        );
        // A mode write after the initial wait may have published a later
        // fence. Its GFIFO entry cannot pass this marker, so defer rather
        // than wait here or compose state not ordered by the cut.
        if !rex.resident_display_fence_complete() {
            return Err(ResidentError::Timeout("display fence after frame marker"));
        }

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
        raster.submit_for(FlushReason::FrameCut)?;
        let frame = self.scanout.compose(&screen.display_state(), raster)?;
        if frame.position != cut.position {
            return Err(ResidentError::InvalidState("resident scanout used the wrong stream cut"));
        }
        let stamp = frame.stamp();
        if want.gpu {
            screen.prebuilt = false;
            let buffer = frame.buffer.clone();
            return Ok((Composed::Gpu { frame: buffer, width: stamp.width, height: stamp.height }, stamp));
        }
        let (width, height) = (stamp.width as usize, stamp.height as usize);
        self.capture.resize(width * height, 0);
        let before = self.scanout.traffic().composed_readbacks;
        let result = self.scanout.capture(stamp, raster, &mut self.capture);
        let after = self.scanout.traffic().composed_readbacks;
        rex.resident_frame.capture_delta(ResidentDisplayMode::DirectWgpu, before, after)?;
        result?;
        // A finished frame, as a board that composes on the CPU leaves it.
        screen.width = width;
        screen.height = height;
        screen.prebuilt = true;
        if screen.fb_rgb.len() < FB_STRIDE * height {
            screen.fb_rgb.resize(FB_STRIDE * height, 0);
        }
        for y in 0..height {
            let row = y * FB_STRIDE;
            for (dst, &src) in screen.fb_rgb[row..row + width]
                .iter_mut().zip(&self.capture[y * width..(y + 1) * width])
            {
                *dst = 0xff00_0000 | src.swap_bytes() >> 8;
            }
        }
        Ok((Composed::Frame, stamp))
    }
}

enum ResidentDisplayDriver {
    Direct(ResidentDirectDisplay),
    Compatibility(ResidentCompatibilityDisplay),
}

/// The selected mode is fixed for this display; errors never change it.
pub(crate) struct ResidentDisplay {
    gpu: Arc<Gpu>,
    driver: ResidentDisplayDriver,
}

impl ResidentDisplay {
    pub(crate) fn new(rex: &Rex3, gpu: &Arc<Gpu>, mode: ResidentDisplayMode) -> Result<Self, ResidentError> {
        let resident_gpu = {
            let mut raster = rex.raster.lock();
            let RasterMode::Resident(raster) = &mut *raster else {
                return Err(ResidentError::InvalidState("resident raster is not active"));
            };
            raster.status()?;
            raster.gpu().ok_or(ResidentError::Unsupported("resident display requires a GPU"))?
        };
        if !Arc::ptr_eq(&resident_gpu, gpu) {
            return Err(ResidentError::DeviceMismatch);
        }
        let driver = match mode {
            ResidentDisplayMode::DirectWgpu => ResidentDisplayDriver::Direct(ResidentDirectDisplay::new(rex, gpu)?),
            ResidentDisplayMode::ReadbackCompatibility => {
                ResidentDisplayDriver::Compatibility(ResidentCompatibilityDisplay::new())
            }
        };
        Ok(Self { gpu: Arc::clone(gpu), driver })
    }

    pub(crate) fn mode(&self) -> ResidentDisplayMode {
        match self.driver {
            ResidentDisplayDriver::Direct(_) => ResidentDisplayMode::DirectWgpu,
            ResidentDisplayDriver::Compatibility(_) => ResidentDisplayMode::ReadbackCompatibility,
        }
    }

    pub(crate) fn direct_traffic(&self) -> Option<ResidentScanoutTraffic> {
        match &self.driver {
            ResidentDisplayDriver::Direct(display) => Some(display.scanout.traffic()),
            ResidentDisplayDriver::Compatibility(_) => None,
        }
    }

    pub(crate) fn capture_count(&self) -> CaptureCount {
        match &self.driver {
            ResidentDisplayDriver::Direct(display) => display.scanout.traffic().composed_readbacks,
            ResidentDisplayDriver::Compatibility(display) => display.capture,
        }
    }

    #[cfg(test)]
    pub(crate) fn fail_next_capture(&mut self, fault: CaptureFault) {
        match &mut self.driver {
            ResidentDisplayDriver::Direct(display) => display.scanout.fail_next_capture(fault),
            ResidentDisplayDriver::Compatibility(display) => display.capture_fault = Some(fault),
        }
    }

    /// Compose the frame at a new GFIFO cut into `screen`, as `want` asks:
    /// what `Rex3Head::compose` answers the refresh thread with.
    pub(crate) fn compose(
        &mut self,
        rex: &Rex3,
        screen: &mut Rex3Screen,
        want: Want,
        timeout: Duration,
        at_marker: impl FnOnce(Position),
    ) -> Result<Composed, ResidentError> {
        let started = Instant::now();
        let kept = (screen.width, screen.height, screen.topscan, screen.prebuilt);
        let composed = match &mut self.driver {
            ResidentDisplayDriver::Direct(display) => display.compose(rex, screen, want, timeout, at_marker).map(|c| c.0),
            ResidentDisplayDriver::Compatibility(display) => display.compose(rex, screen, want, timeout, at_marker),
        };
        // A frame that was not composed leaves the screen as the last one
        // left it: the thread shows that again, at its size.
        let composed = composed.inspect_err(|_| {
            (screen.width, screen.height, screen.topscan, screen.prebuilt) = kept;
        })?;
        let elapsed = duration_ns(started.elapsed()).map_err(ResidentError::InvalidState)?;
        rex.resident_telemetry.lock().presentation(elapsed);
        Ok(composed)
    }

    /// One frame as the refresh thread makes it, for the tests: the
    /// thread's own frame for the board's head (requests, screenshots,
    /// presenting, readback), with this display composing it. The hook runs
    /// at the cut, as in [`compose`](Self::compose).
    #[cfg(test)]
    pub(crate) fn render(
        &mut self,
        rex: &Rex3,
        timeout: Duration,
        at_marker: impl FnOnce(Position) + Send,
    ) -> Result<ResidentDisplayFrame, ResidentError> {
        struct Standin<'a> {
            rex: &'a Rex3,
            display: Mutex<&'a mut ResidentDisplay>,
            at_marker: Mutex<Option<Box<dyn FnOnce(Position) + Send + 'a>>>,
            timeout: Duration,
            result: Mutex<Option<Result<(Position, u32, u32), ResidentError>>>,
        }
        impl crate::display::Head for Standin<'_> {
            fn changed(&self) -> bool { true }
            fn compose(&self, screen: &mut Rex3Screen, want: Want) -> Composed {
                let hook = self.at_marker.lock().take();
                let mut cut = None;
                let result = ResidentDisplay::compose(&mut self.display.lock(), self.rex, screen, want, self.timeout, |p| {
                    cut = Some(p);
                    if let Some(hook) = hook { hook(p) }
                });
                let (composed, result) = match result {
                    Ok(composed) => {
                        let (w, h) = match &composed {
                            Composed::Gpu { width, height, .. } => (*width, *height),
                            _ => (screen.width as u32, screen.height as u32),
                        };
                        (composed, Ok((cut.expect("a compose reaches its cut"), w, h)))
                    }
                    Err(error) => (Composed::NotReady, Err(error)),
                };
                *self.result.lock() = Some(result);
                composed
            }
            fn gfifo_pending(&self) -> usize { self.rex.gfifo.len() }
        }
        let standin = Standin {
            rex,
            display: Mutex::new(self),
            at_marker: Mutex::new(Some(Box::new(at_marker))),
            timeout,
            result: Mutex::new(None),
        };
        rex.head.frame_now_as(&standin);
        let result = standin.result.lock().take()
            .ok_or(ResidentError::InvalidState("the refresh frame did not compose"))?;
        let (position, width, height) = result?;
        Ok(ResidentDisplayFrame { position, width, height })
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

    /// Composed-capture totals are counted per mode and survive a resident
    /// reset; a counter that would overflow or run backwards is an error.
    #[test]
    fn resident_capture_totals_survive_resets() {
        let control = ResidentFrameControl::default();
        control.activate();
        control.capture_attempt(ResidentDisplayMode::DirectWgpu, 64).unwrap();
        control.capture_success(ResidentDisplayMode::DirectWgpu, 64).unwrap();
        control.capture_attempt(ResidentDisplayMode::ReadbackCompatibility, 32).unwrap();
        control.activate();
        assert_eq!(control.capture_count(ResidentDisplayMode::DirectWgpu), CaptureCount {
            attempts: 1, successes: 1, attempted_bytes: 64, successful_bytes: 64,
        });
        assert_eq!(control.capture_count(ResidentDisplayMode::ReadbackCompatibility), CaptureCount {
            attempts: 1, successes: 0, attempted_bytes: 32, successful_bytes: 0,
        });
        let exhausted = CaptureCount { attempts: u64::MAX, attempted_bytes: 64, successes: 1, successful_bytes: 64 };
        control.captures.lock().direct = exhausted;
        assert!(control.capture_attempt(ResidentDisplayMode::DirectWgpu, 1).is_err());
        assert_eq!(control.capture_count(ResidentDisplayMode::DirectWgpu), exhausted);
        assert!(control.capture_delta(
            ResidentDisplayMode::ReadbackCompatibility,
            CaptureCount { attempts: 1, ..CaptureCount::default() },
            CaptureCount::default(),
        ).is_err());
    }

    #[test]
    fn resident_frame_timeouts_never_publish_fences_or_frames() {
        std::thread::Builder::new().stack_size(64 * 1024 * 1024).spawn(|| {
            let mut rex = Box::new(Rex3::new(crate::status::MachineStatus::detached()));
            rex.attach_resident_raster(Probe).unwrap();
            rex.running.store(true, Ordering::Release);
            rex.xmap_fence.store(9, Ordering::Release);
            rex.gfifo_fence.store(8, Ordering::Release);
            rex.request_frame();
            rex.request_screenshot();
            {
                let mut screen = rex.head.screen().lock();
                screen.width = 1;
                screen.height = 1;
                screen.rgba[0] = 0xff12_3456;
                screen.frame_seq = 7;
            }
            assert_eq!(rex.wait_resident_display_fence(Duration::ZERO), Err(ResidentError::Timeout("display fence")));
            assert_eq!(rex.gfifo_fence.load(Ordering::Acquire), 8);
            assert!(rex.frame_pending() && rex.screenshot_pending());
            assert_eq!(rex.head.screen().lock().frame_seq, 7, "a timed-out frame publishes nothing");

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
            assert!(rex.snapshot_resident_registers().is_ok());
            let serial = rex.resident_frame.serial.lock();
            assert!(rex.snapshot_resident_registers().is_err());
            drop(serial);
            rex.resident_frame.marker.lock().active = Some(Marker { id: 1, reached: None });
            assert!(rex.snapshot_resident_registers().is_err());
            rex.resident_frame.marker.lock().active = None;
            assert!(rex.snapshot_resident_registers().is_ok());
        }).unwrap().join().unwrap();
    }

    #[test]
    fn resident_device_loss_with_full_gfifo_stops_and_prevents_restart() {
        const CHILD: &str = "IRIS_RESIDENT_FULL_FIFO_LOSS_CHILD";
        if std::env::var_os(CHILD).is_some() {
            std::thread::Builder::new().stack_size(64 * 1024 * 1024).spawn(|| {
                let mut board = Box::new(Rex3::new(crate::status::MachineStatus::detached()));
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
