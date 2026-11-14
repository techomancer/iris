//! Test-only Newport raster with GPU-authoritative linear planes.
//!
//! Activation seeds both complete planes before accepting draws. No CPU
//! framebuffer is passed to the raster after that boundary. Raw planes are
//! read back only at an explicit observation; this is not a live renderer.

use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::time::Duration;

use bytemuck::{Pod, Zeroable};
use parking_lot::Mutex;

use crate::dev::ng1::rex3::{Rex3Context, REX3_SCREEN_HEIGHT, REX3_SCREEN_WIDTH};
use crate::dev::ng1::rex3_flat_fill::{prepare_resident_flat_fill, FillPlane};
use crate::wgpu::tiles::{COARSE, LIST_CAP, MASK_WORDS, TILE_H, TILE_W};
use crate::wgpu::{ArchShaders, Batch, BatchPipelines, Fragment, Full, Gpu, TileGrid};

const STRIDE: u32 = 2048;
const HEIGHT: u32 = 1024;
const PLANE_WORDS: usize = (STRIDE * HEIGHT) as usize;
const PLANE_BYTES: u64 = PLANE_WORDS as u64 * size_of::<u32>() as u64;
const READBACK_BYTES: u64 = 2 * PLANE_BYTES;
const MAX_SLOTS: u32 = 1024;
const MAX_SETS: usize = 3;
const READBACK_CALLBACK_TIMEOUT: Duration = Duration::from_secs(30);
const FLAT: Fragment = Fragment::new("resident_flat.wgsl", include_str!("shaders/resident_flat.wgsl"));

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ResidentError {
    Unsupported(&'static str),
    InvalidState(&'static str),
    InvalidSeed { plane: &'static str, expected: usize, actual: usize },
    InvalidShadow { plane: &'static str, reason: &'static str },
    Limit { name: &'static str, required: u64, available: u64 },
    DeviceMismatch,
    Timeout(&'static str),
    DeviceLost(String),
    Failed(String),
}

impl std::fmt::Display for ResidentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(reason) => write!(f, "unsupported resident operation: {reason}"),
            Self::InvalidState(reason) => write!(f, "invalid resident state: {reason}"),
            Self::InvalidSeed { plane, expected, actual } => {
                write!(f, "resident {plane} seed has {actual} words, expected {expected}")
            }
            Self::InvalidShadow { plane, reason } => {
                write!(f, "resident {plane} CPU shadow: {reason}")
            }
            Self::Limit { name, required, available } => {
                write!(f, "resident GPU limit {name} is {available}, requires {required}")
            }
            Self::DeviceMismatch => write!(f, "resident raster, display and presenter must use the same GPU"),
            Self::Timeout(reason) => write!(f, "resident frame timed out waiting for {reason}"),
            Self::DeviceLost(reason) => write!(f, "resident GPU device lost: {reason}"),
            Self::Failed(reason) => write!(f, "resident raster failed: {reason}"),
        }
    }
}

impl std::error::Error for ResidentError {}

fn gpu_error(error: String) -> ResidentError {
    if error.starts_with("wgpu device lost (") {
        ResidentError::DeviceLost(error)
    } else {
        ResidentError::Failed(error)
    }
}

/// Epochs are process-unique, including across board destruction and
/// reactivation. Sequence zero denotes the submitted initialization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Epoch(u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Sequence(u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Position {
    pub(crate) epoch: Epoch,
    pub(crate) sequence: Sequence,
}

/// Raw-plane observations are explicit and attributed to their consumer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ObservationReason {
    HostRead,
    Snapshot,
    Monitor,
    Recording,
    Validation,
    Compatibility,
}

impl ObservationReason {
    const COUNT: usize = 6;

    fn index(self) -> usize {
        match self {
            Self::HostRead => 0,
            Self::Snapshot => 1,
            Self::Monitor => 2,
            Self::Recording => 3,
            Self::Validation => 4,
            Self::Compatibility => 5,
        }
    }
}

/// A nonempty selection of complete raw planes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PlaneMask {
    Rgb,
    Aux,
    Both,
}

impl PlaneMask {
    fn rgb(self) -> bool {
        matches!(self, Self::Rgb | Self::Both)
    }

    fn aux(self) -> bool {
        matches!(self, Self::Aux | Self::Both)
    }

    fn transfers(self) -> u64 {
        u64::from(self.rgb()) + u64::from(self.aux())
    }

    fn bytes(self) -> u64 {
        self.transfers() * PLANE_BYTES
    }
}

/// The returned bytes are a snapshot, not an installed board CPU shadow.
/// Plane generations are the last possible writes at the captured stream cut,
/// which can be earlier than `position` when the other plane or a no-op ran.
#[derive(Debug)]
pub(crate) struct Observation {
    position: Position,
    reason: ObservationReason,
    planes: PlaneMask,
    rgb: Option<Vec<u8>>,
    aux: Option<Vec<u8>>,
    rgb_generation: Option<Position>,
    aux_generation: Option<Position>,
}

impl Observation {
    pub(crate) fn position(&self) -> Position { self.position }
    pub(crate) fn reason(&self) -> ObservationReason { self.reason }
    pub(crate) fn planes(&self) -> PlaneMask { self.planes }
    pub(crate) fn rgb(&self) -> Option<&[u8]> { self.rgb.as_deref() }
    pub(crate) fn aux(&self) -> Option<&[u8]> { self.aux.as_deref() }
    pub(crate) fn rgb_generation(&self) -> Option<Position> { self.rgb_generation }
    pub(crate) fn aux_generation(&self) -> Option<Position> { self.aux_generation }
}

/// Counts observations and their selected full-plane transfers separately.
/// Attempts are recorded before submit; a success requires the whole copy,
/// wait and map to succeed. Positions refer to the stream cut, not a plane.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ObservationCounts {
    pub(crate) attempts: u64,
    pub(crate) successes: u64,
    pub(crate) attempted_transfers: u64,
    pub(crate) successful_transfers: u64,
    pub(crate) attempted_bytes: u64,
    pub(crate) successful_bytes: u64,
    pub(crate) last_attempted: Option<Position>,
    pub(crate) last_successful: Option<Position>,
}

impl ObservationCounts {
    fn attempt(&mut self, planes: PlaneMask, position: Position) -> Result<(), ResidentError> {
        let attempts = self.attempts.checked_add(1);
        let transfers = self.attempted_transfers.checked_add(planes.transfers());
        let bytes = self.attempted_bytes.checked_add(planes.bytes());
        let (Some(attempts), Some(transfers), Some(bytes)) = (attempts, transfers, bytes) else {
            return Err(ResidentError::InvalidState("observation counters exhausted"));
        };
        self.attempts = attempts;
        self.attempted_transfers = transfers;
        self.attempted_bytes = bytes;
        self.last_attempted = Some(position);
        Ok(())
    }

    fn success(&mut self, planes: PlaneMask, position: Position) -> Result<(), ResidentError> {
        let successes = self.successes.checked_add(1);
        let transfers = self.successful_transfers.checked_add(planes.transfers());
        let bytes = self.successful_bytes.checked_add(planes.bytes());
        let (Some(successes), Some(transfers), Some(bytes)) = (successes, transfers, bytes) else {
            return Err(ResidentError::InvalidState("observation counters exhausted"));
        };
        self.successes = successes;
        self.successful_transfers = transfers;
        self.successful_bytes = bytes;
        self.last_successful = Some(position);
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ObservationCounters {
    reasons: [ObservationCounts; ObservationReason::COUNT],
}

impl ObservationCounters {
    pub(crate) fn for_reason(&self, reason: ObservationReason) -> &ObservationCounts {
        &self.reasons[reason.index()]
    }

    fn for_reason_mut(&mut self, reason: ObservationReason) -> &mut ObservationCounts {
        &mut self.reasons[reason.index()]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PlaneStatus {
    /// The last accepted command that can write this plane, or seed position.
    pub(crate) last_write: Position,
    /// The generation installed in the board's CPU copy. None means unknown
    /// or mixed contents, e.g. after a partial restore.
    pub(crate) cpu_shadow: Option<Position>,
    /// The stream cut of the latest successful returned snapshot, if any.
    pub(crate) last_observed: Option<Position>,
}

impl PlaneStatus {
    pub(crate) fn cpu_shadow_current(self) -> bool {
        self.cpu_shadow == Some(self.last_write)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ResidentStatus {
    pub(crate) accepted: Position,
    pub(crate) submitted: Position,
    /// None until a queue completion callback actually confirms the seed.
    pub(crate) completed: Option<Position>,
    pub(crate) rgb: PlaneStatus,
    pub(crate) aux: PlaneStatus,
    pub(crate) initialization_uploads: u32,
    pub(crate) initialization_bytes: u64,
    /// Successfully queued restore-prefix uploads, not GPU completions.
    pub(crate) restore_uploads: u64,
    pub(crate) restore_bytes: u64,
    /// Cumulative across resets and restores for this backend instance.
    pub(crate) observations: ObservationCounters,
    /// Raw-plane staging maps attempted, including maps that fail.
    pub(crate) raw_plane_map_attempts: u64,
}

impl ResidentStatus {
    #[cfg(test)]
    pub(crate) fn for_test() -> Self {
        Self::seeded(Epoch(0))
    }

    fn seeded(epoch: Epoch) -> Self {
        let seed = Position { epoch, sequence: Sequence(0) };
        let plane = PlaneStatus { last_write: seed, cpu_shadow: Some(seed), last_observed: None };
        Self {
            accepted: seed,
            submitted: seed,
            completed: None,
            rgb: plane,
            aux: plane,
            initialization_uploads: 2,
            initialization_bytes: 2 * PLANE_BYTES,
            restore_uploads: 0,
            restore_bytes: 0,
            observations: ObservationCounters::default(),
            raw_plane_map_attempts: 0,
        }
    }

    fn reseeded(epoch: Epoch, previous: &Self) -> Result<Self, ResidentError> {
        let mut next = Self::seeded(epoch);
        next.initialization_uploads = previous.initialization_uploads.checked_add(2)
            .ok_or(ResidentError::InvalidState("initialization counters exhausted"))?;
        next.initialization_bytes = previous.initialization_bytes.checked_add(2 * PLANE_BYTES)
            .ok_or(ResidentError::InvalidState("initialization counters exhausted"))?;
        next.restore_uploads = previous.restore_uploads;
        next.restore_bytes = previous.restore_bytes;
        next.observations = previous.observations.clone();
        next.raw_plane_map_attempts = previous.raw_plane_map_attempts;
        Ok(next)
    }

    fn restored(&self, epoch: Epoch, rgb_words: usize, aux_words: usize) -> Result<Self, ResidentError> {
        let seed = Position { epoch, sequence: Sequence(0) };
        let plane = |previous: PlaneStatus, changed: bool| PlaneStatus {
            last_write: seed,
            cpu_shadow: (!changed && previous.cpu_shadow_current()).then_some(seed),
            last_observed: None,
        };
        let uploads = u64::from(rgb_words != 0) + u64::from(aux_words != 0);
        let bytes = (rgb_words + aux_words) as u64 * size_of::<u32>() as u64;
        Ok(Self {
            accepted: seed,
            submitted: seed,
            completed: None,
            rgb: plane(self.rgb, rgb_words != 0),
            aux: plane(self.aux, aux_words != 0),
            initialization_uploads: self.initialization_uploads,
            initialization_bytes: self.initialization_bytes,
            restore_uploads: self.restore_uploads.checked_add(uploads)
                .ok_or(ResidentError::InvalidState("restore counters exhausted"))?,
            restore_bytes: self.restore_bytes.checked_add(bytes)
                .ok_or(ResidentError::InvalidState("restore counters exhausted"))?,
            observations: self.observations.clone(),
            raw_plane_map_attempts: self.raw_plane_map_attempts,
        })
    }

    fn next_position(&self) -> Result<Position, ResidentError> {
        let sequence = self.accepted.sequence.0.checked_add(1)
            .ok_or(ResidentError::InvalidState("stream sequence exhausted"))?;
        Ok(Position { epoch: self.accepted.epoch, sequence: Sequence(sequence) })
    }

    fn accept(&mut self, position: Position, plane: Option<FillPlane>) {
        debug_assert_eq!(position.epoch, self.accepted.epoch);
        debug_assert_eq!(position.sequence.0, self.accepted.sequence.0 + 1);
        self.accepted = position;
        match plane {
            Some(FillPlane::Rgb) => self.rgb.last_write = position,
            Some(FillPlane::Aux) => self.aux.last_write = position,
            None => {}
        }
    }

    fn submit(&mut self, position: Position) {
        debug_assert_eq!(position.epoch, self.accepted.epoch);
        debug_assert!(position.sequence <= self.accepted.sequence);
        debug_assert!(position.sequence >= self.submitted.sequence);
        self.submitted = position;
    }

    fn confirm(&mut self, position: Position) {
        if position.epoch != self.accepted.epoch || position.sequence > self.submitted.sequence {
            return;
        }
        if self.completed.is_none_or(|old| position.sequence > old.sequence) {
            self.completed = Some(position);
        }
    }
}

static NEXT_EPOCH: AtomicU64 = AtomicU64::new(1);

fn next_epoch() -> Result<Epoch, ResidentError> {
    NEXT_EPOCH
        .try_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .map(Epoch)
        .map_err(|_| ResidentError::InvalidState("resident epochs exhausted"))
}

fn validate_limits(limits: &wgpu::Limits) -> Result<(), ResidentError> {
    let check = |name, available: u64, required: u64| {
        if available < required {
            Err(ResidentError::Limit { name, required, available })
        } else {
            Ok(())
        }
    };
    let grid = TileGrid::new(REX3_SCREEN_WIDTH as u32, REX3_SCREEN_HEIGHT as u32);
    let workgroup_bytes = 4 * (MASK_WORDS * (1 + COARSE * COARSE) + 64 + 2);
    check("max_buffer_size", limits.max_buffer_size, READBACK_BYTES.max(LIST_CAP as u64 * 4))?;
    check("max_storage_buffer_binding_size", limits.max_storage_buffer_binding_size, PLANE_BYTES.max(LIST_CAP as u64 * 4))?;
    // The bin layout has four storage buffers in group 0, two in group 1,
    // and both planes in group 2.
    check("max_storage_buffers_per_shader_stage", limits.max_storage_buffers_per_shader_stage as u64, 8)?;
    check("max_bind_groups", limits.max_bind_groups as u64, 3)?;
    check("max_bindings_per_bind_group", limits.max_bindings_per_bind_group as u64, 6)?;
    check("max_compute_invocations_per_workgroup", limits.max_compute_invocations_per_workgroup as u64, (TILE_W * TILE_H) as u64)?;
    check("max_compute_workgroup_size_x", limits.max_compute_workgroup_size_x as u64, (TILE_W * TILE_H) as u64)?;
    check("max_compute_workgroup_storage_size", limits.max_compute_workgroup_storage_size as u64, workgroup_bytes as u64)?;
    check("max_compute_workgroups_per_dimension", limits.max_compute_workgroups_per_dimension as u64, grid.tiles() as u64)?;
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FillRecord {
    min_x: u32,
    min_y: u32,
    max_x: u32,
    max_y: u32,
    slot: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FillSlot {
    color: u32,
    mask: u32,
    plane: u32,
    _pad: u32,
}

/// The only owner of the raw plane handles. The bind-group seam below gives
/// scanout read-only bindings, never writable buffer handles.
struct Planes {
    gpu: Arc<Gpu>,
    rgb: wgpu::Buffer,
    aux: wgpu::Buffer,
}

/// A scanout pipeline can only bind the planes through this read-only layout.
/// Display metadata must be bound separately; this seam accepts no foreign
/// buffer handles whose device ownership wgpu cannot safely inspect. A
/// consumer must discard bindings from an old epoch and submit pending raster
/// work before scanout; this seam does neither.
pub(crate) struct ResidentScanoutBindings {
    pub(in crate::dev::ng1) epoch: Epoch,
    pub(in crate::dev::ng1) layout: wgpu::BindGroupLayout,
    pub(in crate::dev::ng1) group: wgpu::BindGroup,
}

impl Planes {
    fn new(gpu: &Arc<Gpu>) -> Self {
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        Self {
            gpu: Arc::clone(gpu),
            rgb: gpu.buffer("resident rex3 rgb", PLANE_BYTES, usage),
            aux: gpu.buffer("resident rex3 aux", PLANE_BYTES, usage),
        }
    }

    fn scanout_bindings(
        &self,
        gpu: &Arc<Gpu>,
        epoch: Epoch,
    ) -> Result<ResidentScanoutBindings, ResidentError> {
        if !Arc::ptr_eq(&self.gpu, gpu) {
            return Err(ResidentError::DeviceMismatch);
        }
        gpu.check().map_err(gpu_error)?;
        let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("resident rex3 scanout planes"),
            entries: &[entry(0), entry(1)],
        });
        let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resident rex3 scanout planes"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.rgb.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.aux.as_entire_binding() },
            ],
        });
        if let Some(error) = pollster::block_on(scope.pop()) {
            return Err(ResidentError::Failed(format!("scanout bindings: {error}")));
        }
        Ok(ResidentScanoutBindings { epoch, layout, group })
    }
}

fn record_completion(gpu: &Arc<Gpu>, progress: &Arc<Mutex<ResidentStatus>>, position: Position) {
    let gpu: Weak<Gpu> = Arc::downgrade(gpu);
    let progress = Arc::clone(progress);
    // The submitted position is published before installing the callback.
    // Another thread may poll the device as soon as it is installed.
    if let Some(gpu_ref) = gpu.upgrade() {
        gpu_ref.on_done(move || {
            if gpu.upgrade().is_some_and(|gpu| gpu.check().is_ok()) {
                progress.lock().confirm(position);
            }
        });
    }
}

/// A resident draw owns its destination planes and never receives the CPU
/// framebuffer. Errors are sticky and must not trigger CPU replay.
pub(crate) trait ResidentBackend: Send {
    fn enqueue(&mut self, context: &mut Rex3Context) -> Result<(), ResidentError>;

    fn gpu(&self) -> Option<&Arc<Gpu>> { None }

    fn poll_device_health(&mut self) -> Result<(), ResidentError> { Ok(()) }

    fn submit(&mut self) -> Result<(), ResidentError> {
        Err(ResidentError::Unsupported("submit"))
    }

    /// Submit the accepted prefix and return the selected full planes at its
    /// stream cut. This does not install or acknowledge a CPU shadow.
    fn observe(
        &mut self,
        _reason: ObservationReason,
        _planes: PlaneMask,
    ) -> Result<Observation, ResidentError> {
        Err(ResidentError::Unsupported("observation"))
    }

    /// Each supplied shadow must be the allocation used at activation/reset
    /// and must already contain the complete corresponding capture.
    fn acknowledge_cpu_shadow(
        &mut self,
        _observation: &Observation,
        _rgb: Option<&[u32]>,
        _aux: Option<&[u32]>,
    ) -> Result<(), ResidentError> {
        Err(ResidentError::Unsupported("CPU shadow acknowledgement"))
    }

    fn status(&mut self) -> Result<ResidentStatus, ResidentError> {
        Err(ResidentError::Unsupported("status"))
    }

    /// The last recorded state without polling a failed device. Used only
    /// for diagnostics; the raster's sticky fault is reported alongside it.
    fn recorded_status(&self) -> Result<ResidentStatus, ResidentError> {
        Err(ResidentError::Unsupported("recorded status"))
    }

    /// With GFIFO and scanout quiescent, discard the old stream and seed new
    /// planes from the board's complete CPU-shadow allocations.
    fn reset(&mut self, _rgb: &[u32], _aux: &[u32]) -> Result<(), ResidentError> {
        Err(ResidentError::Unsupported("reset"))
    }

    /// With GFIFO and scanout quiescent, submit the old prefix and upload
    /// only the supplied plane prefixes. Overlong inputs are clamped to a
    /// plane. Unchanged GPU words survive; changed CPU shadows are unknown
    /// until a full captured plane is installed and acknowledged.
    fn restore(&mut self, _rgb: &[u32], _aux: &[u32]) -> Result<(), ResidentError> {
        Err(ResidentError::Unsupported("restore"))
    }

    fn scanout_bindings(
        &self,
        _gpu: &Arc<Gpu>,
    ) -> Result<ResidentScanoutBindings, ResidentError> {
        Err(ResidentError::Unsupported("scanout bindings"))
    }
}

pub(crate) struct ResidentRaster {
    backend: Box<dyn ResidentBackend>,
    fault: Option<ResidentError>,
}

#[derive(Clone, Debug)]
pub(crate) struct ResidentDiagnostics {
    pub(crate) status: ResidentStatus,
    pub(crate) fault: Option<ResidentError>,
}

#[cfg(test)]
pub(crate) enum ResidentFaultInjection {
    DeviceLost,
    DeviceError,
}

impl ResidentRaster {
    pub(crate) fn new(backend: impl ResidentBackend + 'static) -> Self {
        Self { backend: Box::new(backend), fault: None }
    }

    pub(crate) fn gpu(&self) -> Option<Arc<Gpu>> {
        self.backend.gpu().cloned()
    }

    fn run<T>(&mut self, operation: impl FnOnce(&mut dyn ResidentBackend) -> Result<T, ResidentError>) -> Result<T, ResidentError> {
        if let Some(error) = &self.fault {
            return Err(error.clone());
        }
        operation(self.backend.as_mut()).map_err(|error| {
            self.fail(error.clone());
            error
        })
    }

    pub(crate) fn enqueue(&mut self, context: &mut Rex3Context) -> Result<(), ResidentError> {
        if let Some(error) = &self.fault {
            return Err(error.clone());
        }
        match self.backend.enqueue(context) {
            Ok(()) => Ok(()),
            Err(error) => {
                // A rejected GO must not strand an already accepted prefix.
                // A device failure while submitting that prefix takes priority.
                let error = if matches!(&error, ResidentError::Unsupported(_)) {
                    self.backend.submit().err().unwrap_or(error)
                } else {
                    error
                };
                self.fault = Some(error.clone());
                Err(error)
            }
        }
    }

    pub(crate) fn submit(&mut self) -> Result<(), ResidentError> {
        self.run(|backend| backend.submit())
    }

    pub(crate) fn poll_device_health(&mut self) -> Result<(), ResidentError> {
        self.run(|backend| backend.poll_device_health())
    }

    pub(crate) fn observe(
        &mut self,
        reason: ObservationReason,
        planes: PlaneMask,
    ) -> Result<Observation, ResidentError> {
        self.run(|backend| backend.observe(reason, planes))
    }

    pub(crate) fn acknowledge_cpu_shadow(
        &mut self,
        observation: &Observation,
        rgb: Option<&[u32]>,
        aux: Option<&[u32]>,
    ) -> Result<(), ResidentError> {
        if let Some(error) = &self.fault {
            return Err(error.clone());
        }
        match self.backend.acknowledge_cpu_shadow(observation, rgb, aux) {
            Err(error @ (ResidentError::InvalidShadow { .. } | ResidentError::InvalidState(_))) => Err(error),
            Err(error) => {
                self.fail(error.clone());
                Err(error)
            }
            Ok(()) => Ok(()),
        }
    }

    pub(crate) fn status(&mut self) -> Result<ResidentStatus, ResidentError> {
        self.run(|backend| backend.status())
    }

    pub(crate) fn diagnostics(&self) -> Result<ResidentDiagnostics, ResidentError> {
        Ok(ResidentDiagnostics {
            status: self.backend.recorded_status()?,
            fault: self.fault.clone(),
        })
    }

    pub(crate) fn reset(&mut self, rgb: &[u32], aux: &[u32]) -> Result<(), ResidentError> {
        if let Some(error @ (ResidentError::DeviceLost(_) | ResidentError::Failed(_))) = &self.fault {
            return Err(error.clone());
        }
        match self.backend.reset(rgb, aux) {
            Ok(()) => {
                self.fault = None;
                Ok(())
            }
            Err(error) => {
                self.fault = Some(error.clone());
                Err(error)
            }
        }
    }

    pub(crate) fn restore(&mut self, rgb: &[u32], aux: &[u32]) -> Result<(), ResidentError> {
        self.run(|backend| backend.restore(rgb, aux))
    }

    pub(in crate::dev::ng1) fn scanout_bindings(
        &self,
        gpu: &Arc<Gpu>,
    ) -> Result<ResidentScanoutBindings, ResidentError> {
        if let Some(error) = &self.fault {
            return Err(error.clone());
        }
        self.backend.scanout_bindings(gpu)
    }

    pub(crate) fn fault(&self) -> Option<&ResidentError> {
        self.fault.as_ref()
    }

    pub(crate) fn fail(&mut self, error: ResidentError) {
        self.fault.get_or_insert(error);
    }

    #[cfg(test)]
    pub(crate) fn inject_fault_for_test(&mut self, injection: ResidentFaultInjection) {
        let error = match injection {
            ResidentFaultInjection::DeviceLost => ResidentError::DeviceLost("injected device loss".into()),
            ResidentFaultInjection::DeviceError => ResidentError::Failed("injected device error".into()),
        };
        self.fail(error);
    }
}

/// The test-only GPU implementation. Draws change the GPU planes directly;
/// reset and explicit restore are the only later raw-plane upload boundaries.
pub(crate) struct ResidentGpu {
    gpu: Arc<Gpu>,
    planes: Planes,
    pipelines: BatchPipelines,
    group: wgpu::BindGroup,
    batch: Batch<FillRecord, FillSlot>,
    progress: Arc<Mutex<ResidentStatus>>,
    readback: wgpu::Buffer,
    /// Allocation identities only; the backend never dereferences them.
    rgb_shadow: usize,
    aux_shadow: usize,
    max_slots: u32,
    max_sets: NonZeroUsize,
}

impl ResidentGpu {
    /// The seed slices are the board's persistent CPU-shadow allocations.
    /// They must not be replaced while this backend is active.
    pub(crate) fn activate(gpu: Arc<Gpu>, rgb: &[u32], aux: &[u32]) -> Result<Self, ResidentError> {
        Self::activate_with_limits(gpu, rgb, aux, MAX_SLOTS, NonZeroUsize::new(MAX_SETS).unwrap(), None)
    }

    fn activate_with_limits(
        gpu: Arc<Gpu>,
        rgb: &[u32],
        aux: &[u32],
        max_slots: u32,
        max_sets: NonZeroUsize,
        reuse: Option<(Arc<Mutex<ResidentStatus>>, wgpu::Buffer)>,
    ) -> Result<Self, ResidentError> {
        for (plane, data) in [("RGB", rgb), ("auxiliary", aux)] {
            if data.len() != PLANE_WORDS {
                return Err(ResidentError::InvalidSeed { plane, expected: PLANE_WORDS, actual: data.len() });
            }
        }
        if max_slots == 0 {
            return Err(ResidentError::InvalidState("a batch needs at least one slot"));
        }
        validate_limits(&gpu.device.limits())?;
        gpu.check().map_err(gpu_error)?;
        let epoch = next_epoch()?;
        let (progress, readback, initial) = match reuse {
            Some((progress, readback)) => {
                let initial = ResidentStatus::reseeded(epoch, &progress.lock())?;
                (progress, readback, initial)
            }
            None => {
                let initial = ResidentStatus::seeded(epoch);
                let readback = gpu.buffer(
                    "resident rex3 readback",
                    READBACK_BYTES,
                    wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                );
                (Arc::new(Mutex::new(initial.clone())), readback, initial)
            }
        };

        let entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(PLANE_BYTES),
            },
            count: None,
        };
        let layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("resident rex3 raster planes"),
            entries: &[entry(0), entry(1)],
        });
        let pipelines = BatchPipelines::new(&gpu, &ArchShaders {
            label: "resident rex3",
            raster: &[FLAT],
            bin_accept: None,
            library: &[],
            defines: &[
                ("PLANE_STRIDE", STRIDE),
                ("WRITE_WIDTH", REX3_SCREEN_WIDTH as u32),
                ("PLANE_HEIGHT", HEIGHT),
            ],
            layouts: &[&layout],
        }).map_err(ResidentError::Failed)?;
        let planes = Planes::new(&gpu);
        let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resident rex3 raster planes"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: planes.rgb.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: planes.aux.as_entire_binding() },
            ],
        });
        let mut batch = Batch::new_bounded(
            &gpu,
            &pipelines,
            TileGrid::new(REX3_SCREEN_WIDTH as u32, REX3_SCREEN_HEIGHT as u32),
            max_slots,
            max_sets,
        );
        // The shared optional batch timer maps and waits at every flush.
        // A resident submission must remain a submission-only boundary.
        batch.timing = None;
        gpu.instance.poll_all(false);
        gpu.check().map_err(gpu_error)?;

        // Both uploads precede the seed submission, which in turn precedes
        // every draw submitted by this backend on the common queue.
        gpu.write(&planes.rgb, 0, bytemuck::cast_slice(rgb));
        gpu.write(&planes.aux, 0, bytemuck::cast_slice(aux));
        gpu.submit(gpu.encoder("resident rex3 seed"));
        let seed = initial.submitted;
        *progress.lock() = initial;
        record_completion(&gpu, &progress, seed);
        gpu.check().map_err(gpu_error)?;
        Ok(Self {
            gpu, planes, pipelines, group, batch, progress, readback,
            rgb_shadow: rgb.as_ptr() as usize,
            aux_shadow: aux.as_ptr() as usize,
            max_slots, max_sets,
        })
    }

    fn read_planes(&mut self, planes: PlaneMask) -> Result<(Option<Vec<u8>>, Option<Vec<u8>>), ResidentError> {
        let mut encoder = self.gpu.encoder("resident rex3 observation");
        let mut at = 0;
        if planes.rgb() {
            encoder.copy_buffer_to_buffer(&self.planes.rgb, 0, &self.readback, at, PLANE_BYTES);
            at += PLANE_BYTES;
        }
        if planes.aux() {
            encoder.copy_buffer_to_buffer(&self.planes.aux, 0, &self.readback, at, PLANE_BYTES);
            at += PLANE_BYTES;
        }
        let submission = self.gpu.submit(encoder);
        let slice = self.readback.slice(..at);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        {
            let mut status = self.progress.lock();
            status.raw_plane_map_attempts = status.raw_plane_map_attempts.checked_add(1)
                .ok_or(ResidentError::InvalidState("raw-plane map counter exhausted"))?;
        }
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let result = (|| {
            self.gpu.wait(Some(submission)).map_err(gpu_error)?;
            rx.recv_timeout(READBACK_CALLBACK_TIMEOUT)
                .map_err(|error| ResidentError::Failed(format!("readback callback: {error}")))?
                .map_err(|error| ResidentError::Failed(format!("map readback: {error}")))?;
            let view = slice.get_mapped_range()
                .map_err(|error| ResidentError::Failed(format!("readback range: {error}")))?;
            let mut offset = 0;
            let rgb = planes.rgb().then(|| {
                let bytes = view[offset..offset + PLANE_BYTES as usize].to_vec();
                offset += PLANE_BYTES as usize;
                bytes
            });
            let aux = planes.aux().then(|| view[offset..offset + PLANE_BYTES as usize].to_vec());
            drop(view);
            self.gpu.check().map_err(gpu_error)?;
            Ok((rgb, aux))
        })();
        // Also cancels a pending map on a wait/callback error.
        self.readback.unmap();
        result
    }

    fn observe_using(
        &mut self,
        reason: ObservationReason,
        planes: PlaneMask,
        readback: impl FnOnce(&mut Self, PlaneMask) -> Result<(Option<Vec<u8>>, Option<Vec<u8>>), ResidentError>,
    ) -> Result<Observation, ResidentError> {
        let (position, rgb_generation, aux_generation) = {
            let mut status = self.progress.lock();
            let position = status.accepted;
            let rgb = planes.rgb().then_some(status.rgb.last_write);
            let aux = planes.aux().then_some(status.aux.last_write);
            status.observations.for_reason_mut(reason).attempt(planes, position)?;
            (position, rgb, aux)
        };
        // This is the complete accepted prefix at entry. Later drawing cannot
        // be enqueued through this exclusively borrowed backend until it ends.
        self.submit()?;
        let (rgb, aux) = readback(self, planes)?;
        let valid = |bytes: &Option<Vec<u8>>, selected: bool| match bytes {
            Some(bytes) => selected && bytes.len() == PLANE_BYTES as usize,
            None => !selected,
        };
        if !valid(&rgb, planes.rgb()) || !valid(&aux, planes.aux()) {
            return Err(ResidentError::InvalidState("incomplete raw-plane observation"));
        }
        self.gpu.check().map_err(gpu_error)?;
        {
            let mut status = self.progress.lock();
            if status.accepted != position {
                return Err(ResidentError::InvalidState("observation stream cut changed"));
            }
            status.observations.for_reason_mut(reason).success(planes, position)?;
            if planes.rgb() { status.rgb.last_observed = Some(position); }
            if planes.aux() { status.aux.last_observed = Some(position); }
            // Waiting for the copy also proves all work before it completed.
            status.confirm(position);
        }
        Ok(Observation { position, reason, planes, rgb, aux, rgb_generation, aux_generation })
    }

    fn check_shadow(
        plane: &'static str,
        shadow: &[u32],
        identity: usize,
        captured: Option<&[u8]>,
    ) -> Result<(), ResidentError> {
        let captured = captured.ok_or(ResidentError::InvalidShadow {
            plane, reason: "plane was not selected for observation",
        })?;
        if shadow.len() != PLANE_WORDS {
            return Err(ResidentError::InvalidShadow { plane, reason: "a complete plane is required" });
        }
        if shadow.as_ptr() as usize != identity {
            return Err(ResidentError::InvalidShadow { plane, reason: "not the activated board shadow" });
        }
        if bytemuck::cast_slice::<u32, u8>(shadow) != captured {
            return Err(ResidentError::InvalidShadow { plane, reason: "complete captured plane is not installed" });
        }
        Ok(())
    }
}

impl ResidentBackend for ResidentGpu {
    fn gpu(&self) -> Option<&Arc<Gpu>> { Some(&self.gpu) }

    fn poll_device_health(&mut self) -> Result<(), ResidentError> {
        self.gpu.instance.poll_all(false);
        self.gpu.check().map_err(gpu_error)
    }

    fn enqueue(&mut self, context: &mut Rex3Context) -> Result<(), ResidentError> {
        self.gpu.check().map_err(gpu_error)?;
        let command = prepare_resident_flat_fill(context).map_err(ResidentError::Unsupported)?;
        let position = self.progress.lock().next_position()?;
        let fill = command.fill();
        if let Some(fill) = fill {
            let slot = FillSlot {
                color: fill.color,
                mask: fill.write_mask,
                plane: match fill.plane { FillPlane::Rgb => 0, FillPlane::Aux => 1 },
                _pad: 0,
            };
            loop {
                let result = self.batch.prim_with_slot(
                    slot,
                    fill.min_x as i32,
                    fill.min_y as i32,
                    (fill.min_x + fill.width - 1) as i32,
                    (fill.min_y + fill.height - 1) as i32,
                    |slot| FillRecord {
                        min_x: fill.min_x,
                        min_y: fill.min_y,
                        max_x: fill.min_x + fill.width - 1,
                        max_y: fill.min_y + fill.height - 1,
                        slot,
                    },
                );
                match result {
                    Ok(Some(_)) => break,
                    Ok(None) => return Err(ResidentError::InvalidState("prepared fill missed its raster grid")),
                    Err(Full) if !self.batch.is_empty() => self.submit()?,
                    Err(Full) => return Err(ResidentError::InvalidState("one fill exceeds an empty batch")),
                }
            }
        }
        self.progress.lock().accept(position, fill.map(|fill| fill.plane));
        command.finish(context);
        Ok(())
    }

    fn submit(&mut self) -> Result<(), ResidentError> {
        self.gpu.check().map_err(gpu_error)?;
        let (accepted, submitted) = {
            let status = self.progress.lock();
            (status.accepted, status.submitted)
        };
        if accepted == submitted {
            return Ok(());
        }
        let submission = self.batch.try_flush(&self.pipelines, &[&self.group])
            .map_err(gpu_error)?;
        if submission.is_none() {
            // A run of no-op GOs still needs a queue marker for its stream
            // position, but it does not change either plane's generation.
            self.gpu.submit(self.gpu.encoder("resident rex3 no-op marker"));
        }
        self.progress.lock().submit(accepted);
        record_completion(&self.gpu, &self.progress, accepted);
        self.gpu.check().map_err(gpu_error)
    }

    fn observe(
        &mut self,
        reason: ObservationReason,
        planes: PlaneMask,
    ) -> Result<Observation, ResidentError> {
        self.observe_using(reason, planes, Self::read_planes)
    }

    fn acknowledge_cpu_shadow(
        &mut self,
        observation: &Observation,
        rgb: Option<&[u32]>,
        aux: Option<&[u32]>,
    ) -> Result<(), ResidentError> {
        self.gpu.check().map_err(gpu_error)?;
        if rgb.is_none() && aux.is_none() {
            return Err(ResidentError::InvalidState("no CPU shadows supplied"));
        }
        if self.progress.lock().accepted.epoch != observation.position.epoch {
            return Err(ResidentError::InvalidState("observation belongs to an old epoch"));
        }
        if let Some(rgb) = rgb { Self::check_shadow("RGB", rgb, self.rgb_shadow, observation.rgb())?; }
        if let Some(aux) = aux { Self::check_shadow("auxiliary", aux, self.aux_shadow, observation.aux())?; }
        let mut status = self.progress.lock();
        if status.accepted.epoch != observation.position.epoch {
            return Err(ResidentError::InvalidState("observation belongs to an old epoch"));
        }
        if rgb.is_some() {
            status.rgb.cpu_shadow = observation.rgb_generation;
        }
        if aux.is_some() {
            status.aux.cpu_shadow = observation.aux_generation;
        }
        Ok(())
    }

    fn status(&mut self) -> Result<ResidentStatus, ResidentError> {
        self.gpu.instance.poll_all(false);
        self.gpu.check().map_err(gpu_error)?;
        Ok(self.progress.lock().clone())
    }

    fn recorded_status(&self) -> Result<ResidentStatus, ResidentError> {
        Ok(self.progress.lock().clone())
    }

    fn reset(&mut self, rgb: &[u32], aux: &[u32]) -> Result<(), ResidentError> {
        let replacement = Self::activate_with_limits(
            Arc::clone(&self.gpu),
            rgb,
            aux,
            self.max_slots,
            self.max_sets,
            Some((Arc::clone(&self.progress), self.readback.clone())),
        )?;
        *self = replacement;
        Ok(())
    }

    fn restore(&mut self, rgb: &[u32], aux: &[u32]) -> Result<(), ResidentError> {
        // Submit the old prefix first so untouched suffixes include its writes.
        self.submit()?;
        let epoch = next_epoch()?;
        let rgb_words = rgb.len().min(PLANE_WORDS);
        let aux_words = aux.len().min(PLANE_WORDS);
        let next = self.progress.lock().restored(epoch, rgb_words, aux_words)?;
        if rgb_words != 0 {
            self.gpu.write(&self.planes.rgb, 0, bytemuck::cast_slice(&rgb[..rgb_words]));
        }
        if aux_words != 0 {
            self.gpu.write(&self.planes.aux, 0, bytemuck::cast_slice(&aux[..aux_words]));
        }
        self.gpu.submit(self.gpu.encoder("resident rex3 restore"));
        let seed = next.submitted;
        *self.progress.lock() = next;
        record_completion(&self.gpu, &self.progress, seed);
        self.gpu.check().map_err(gpu_error)
    }

    fn scanout_bindings(
        &self,
        gpu: &Arc<Gpu>,
    ) -> Result<ResidentScanoutBindings, ResidentError> {
        let epoch = self.progress.lock().accepted.epoch;
        self.planes.scanout_bindings(gpu, epoch)
    }
}

#[path = "resident_tests.rs"]
mod tests;
