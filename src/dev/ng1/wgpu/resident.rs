//! Test-only Newport raster with GPU-authoritative linear planes.
//!
//! Activation seeds both complete planes before accepting draws. No CPU
//! framebuffer is passed to the raster after that boundary. Raw planes are
//! read back only at an explicit observation; this is not a live renderer.

use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use crate::dev::ng1::rex3::{
    Rex3Context, DRAWMODE0_ADRMODE_A_LINE, DRAWMODE0_ADRMODE_F_LINE,
    DRAWMODE0_ADRMODE_I_LINE, DRAWMODE0_OPCODE_DRAW, DRAWMODE0_OPCODE_NOOP,
    DRAWMODE0_OPCODE_READ, DRAWMODE0_OPCODE_SCR2SCR, DRAWMODE1_LOGICOP_SRC,
    DRAWMODE1_PLANES_CID, DRAWMODE1_PLANES_OLAY, DRAWMODE1_PLANES_PUP,
    DRAWMODE1_PLANES_RGB, DRAWMODE1_PLANES_RGBA, OCTANT_XDEC, OCTANT_YDEC,
    REX3_SCREEN_HEIGHT, REX3_SCREEN_WIDTH,
};
use crate::dev::ng1::rex3_flat_fill::{
    prepare_resident_flat_fill, FillKind, FillPlane,
};
use crate::dev::ng1::rex3_generic::{draw_with_fb, Framebuffers};
use crate::dev::ng1::rex3_shape;
use crate::wgpu::host::HostBuffer;
use crate::wgpu::tiles::{COARSE, LIST_CAP, MASK_WORDS, TILE_H, TILE_W};
use crate::wgpu::{ArchShaders, Batch, BatchPipelines, Fragment, Full, Gpu, TileGrid};

use super::command::{
    cid_test, COMMAND_A_LINE, COMMAND_BLOCK, COMMAND_CLEAR, COMMAND_F_LINE,
    COMMAND_COPY_A_LINE, COMMAND_COPY_BLOCK, COMMAND_COPY_F_LINE, COMMAND_COPY_I_LINE,
    COMMAND_COPY_SPAN, COMMAND_HOST_A_LINE, COMMAND_HOST_BLOCK, COMMAND_HOST_F_LINE,
    COMMAND_HOST_I_LINE, COMMAND_HOST_SPAN, COMMAND_I_LINE, COMMAND_SPAN,
    FLAG_FAST_CLEAR, FLAG_FILL_BLOCK, FLAG_FILL_FIRST_ROW_COMPLETE, FLAG_LINE_MAJOR_DECREASING,
    FLAG_LINE_Y_MAJOR, FLAG_READ_AUX, FLAG_SHADED, FLAG_WRITE_AUX, FLAG_WRITE_RGB,
    MAX_PAYLOAD_BYTES, MAX_PAYLOAD_WORDS, PayloadBatch, RexCommand, RexStateSlot, STATE_ALPHA_REF,
    STATE_CLIP_MODE, STATE_COLOR_ALPHA, STATE_COLOR_BACK, STATE_COLOR_BLUE, STATE_COLOR_GREEN,
    STATE_COLOR_RED, STATE_DRAW_MODE0, STATE_DRAW_MODE1, STATE_LINE_MODE, STATE_LINE_PATTERN,
    STATE_HOST_CURSOR, STATE_HOST_LENGTH, STATE_HOST_XEND, STATE_HOST_XSAVE,
    STATE_HOST_XSTART, STATE_HOST_YEND, STATE_HOST_YSTART, STATE_PATTERN_BITS,
    STATE_SLOPE_ALPHA, STATE_SLOPE_BLUE, STATE_SLOPE_GREEN, STATE_SLOPE_RED,
    STATE_SMASK0X, STATE_SMASK0Y, STATE_SMASK1X, STATE_SOURCE_COLOR, STATE_WORDS,
    STATE_WRITE_MASK, STATE_XYMOVE, STATE_XYWIN, STATE_Z_PATTERN,
    PayloadRef,
};
use super::copy::prepare_copy_fill;
use super::host::prepare_host_fill;
use super::line::{prepare_copy_line, prepare_host_line, prepare_line};
use super::telemetry::{duration_ns, Coverage, DrawSample, FlushReason, Workload};

const STRIDE: u32 = 2048;
const HEIGHT: u32 = 1024;
pub(crate) const PLANE_WORDS: usize = (STRIDE * HEIGHT) as usize;
const PLANE_BYTES: u64 = PLANE_WORDS as u64 * size_of::<u32>() as u64;
const READBACK_BYTES: u64 = 2 * PLANE_BYTES;
const MAX_SLOTS: u32 = 1024;
const MAX_SETS: usize = 3;
/// Flush at the next GFIFO command boundary after four milliseconds. A single
/// long command or a bounded GPU-set wait cannot be preempted by this policy.
pub(crate) const MAX_BATCH_AGE: Duration = Duration::from_millis(4);
const READBACK_CALLBACK_TIMEOUT: Duration = Duration::from_secs(30);
const FLAT: Fragment = Fragment::new("resident_flat.wgsl", include_str!("shaders/resident_flat.wgsl"));

fn reads_destination(context: &Rex3Context) -> bool {
    if context.drawmode1.blend() {
        !context.drawmode1.backblend()
    } else {
        context.drawmode1.logicop() != DRAWMODE1_LOGICOP_SRC
    }
}

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

impl Epoch {
    pub(crate) fn value(self) -> u64 { self.0 }
}

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

    #[cfg(test)]
    pub(crate) fn for_test(
        reason: ObservationReason,
        planes: PlaneMask,
        rgb: Option<Vec<u8>>,
        aux: Option<Vec<u8>>,
    ) -> Self {
        let position = ResidentStatus::for_test().accepted;
        Self {
            position, reason, planes,
            rgb_generation: rgb.as_ref().map(|_| position),
            aux_generation: aux.as_ref().map(|_| position),
            rgb, aux,
        }
    }
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
    /// Successfully completed restore-prefix uploads.
    pub(crate) restore_uploads: u64,
    pub(crate) restore_bytes: u64,
    /// Cumulative across resets and restores for this backend instance.
    pub(crate) observations: ObservationCounters,
    /// Raw-plane staging maps attempted, including maps that fail.
    pub(crate) raw_plane_map_attempts: u64,
    /// Accepted HostW GOs and immutable qwords staged for them.
    pub(crate) host_write_transfers: u64,
    pub(crate) host_write_qwords: u64,
    pub(crate) workload: Workload,
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
            host_write_transfers: 0,
            host_write_qwords: 0,
            workload: Workload::default(),
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
        next.host_write_transfers = previous.host_write_transfers;
        next.host_write_qwords = previous.host_write_qwords;
        next.workload = previous.workload.clone();
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
            host_write_transfers: self.host_write_transfers,
            host_write_qwords: self.host_write_qwords,
            workload: self.workload.clone(),
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
    check(
        "max_buffer_size",
        limits.max_buffer_size,
        READBACK_BYTES.max(LIST_CAP as u64 * 4).max(MAX_PAYLOAD_BYTES),
    )?;
    check(
        "max_storage_buffer_binding_size",
        limits.max_storage_buffer_binding_size,
        PLANE_BYTES.max(LIST_CAP as u64 * 4).max(MAX_PAYLOAD_BYTES),
    )?;
    // The raster layout has three storage buffers in group 0, two in group
    // 1, and both planes plus the current payload set in group 2. The
    // accept-all bin pipeline does not bind the board resources.
    check("max_storage_buffers_per_shader_stage", limits.max_storage_buffers_per_shader_stage as u64, 8)?;
    check("max_bind_groups", limits.max_bind_groups as u64, 3)?;
    check("max_bindings_per_bind_group", limits.max_bindings_per_bind_group as u64, 6)?;
    check("max_compute_invocations_per_workgroup", limits.max_compute_invocations_per_workgroup as u64, (TILE_W * TILE_H) as u64)?;
    check("max_compute_workgroup_size_x", limits.max_compute_workgroup_size_x as u64, (TILE_W * TILE_H) as u64)?;
    check("max_compute_workgroup_storage_size", limits.max_compute_workgroup_storage_size as u64, workgroup_bytes as u64)?;
    check("max_compute_workgroups_per_dimension", limits.max_compute_workgroups_per_dimension as u64, grid.tiles() as u64)?;
    Ok(())
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

    fn submit_for(&mut self, _reason: FlushReason) -> Result<(), ResidentError> {
        self.submit()
    }

    fn flush_if_due(&mut self, _now: Instant) -> Result<(), ResidentError> { Ok(()) }

    fn host_read_result(&mut self, _qwords: u64, _generation_ns: u64) -> Result<(), ResidentError> { Ok(()) }

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
    /// planes from the supplied complete data, without acknowledging shadows.
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
    host_read: HostReadSnapshot,
}

#[derive(Default)]
struct HostReadSnapshot {
    key: Option<(FillPlane, Position)>,
    words: Vec<u32>,
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
        Self {
            backend: Box::new(backend),
            fault: None,
            host_read: HostReadSnapshot::default(),
        }
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

    fn enqueue_backend(&mut self, context: &mut Rex3Context) -> Result<(), ResidentError> {
        if let Some(error) = &self.fault {
            return Err(error.clone());
        }
        match self.backend.enqueue(context) {
            Ok(()) => Ok(()),
            Err(error) => {
                // A rejected GO must not strand an already accepted prefix.
                // A device failure while submitting that prefix takes priority.
                let error = if matches!(&error, ResidentError::Unsupported(_)) {
                    self.backend.submit_for(FlushReason::Unsupported).err().unwrap_or(error)
                } else {
                    error
                };
                self.fault = Some(error.clone());
                Err(error)
            }
        }
    }

    fn host_read_plane(context: &Rex3Context) -> Option<FillPlane> {
        match context.drawmode1.planes() {
            DRAWMODE1_PLANES_RGB | DRAWMODE1_PLANES_RGBA => Some(FillPlane::Rgb),
            DRAWMODE1_PLANES_OLAY | DRAWMODE1_PLANES_PUP | DRAWMODE1_PLANES_CID => {
                Some(FillPlane::Aux)
            }
            _ => None,
        }
    }

    fn prepare_host_read_snapshot(&mut self, plane: FillPlane) -> Result<(), ResidentError> {
        let status = self.status()?;
        let generation = match plane {
            FillPlane::Rgb => status.rgb.last_write,
            FillPlane::Aux => status.aux.last_write,
        };
        if self.host_read.key != Some((plane, generation)) {
            let mask = match plane {
                FillPlane::Rgb => PlaneMask::Rgb,
                FillPlane::Aux => PlaneMask::Aux,
            };
            let observation = self.observe(ObservationReason::HostRead, mask)?;
            let (bytes, observed_generation) = match plane {
                FillPlane::Rgb => (observation.rgb(), observation.rgb_generation()),
                FillPlane::Aux => (observation.aux(), observation.aux_generation()),
            };
            if observed_generation != Some(generation) {
                return Err(ResidentError::InvalidState(
                    "HostR snapshot generation changed",
                ));
            }
            let bytes = bytes.ok_or(ResidentError::InvalidState(
                "HostR snapshot omitted its selected plane",
            ))?;
            if bytes.len() != PLANE_BYTES as usize {
                return Err(ResidentError::InvalidState(
                    "HostR snapshot has the wrong size",
                ));
            }
            self.host_read.words.resize(PLANE_WORDS, 0);
            for (word, bytes) in self
                .host_read
                .words
                .iter_mut()
                .zip(bytes.chunks_exact(size_of::<u32>()))
            {
                *word = u32::from_ne_bytes(bytes.try_into().unwrap());
            }
            self.host_read.key = Some((plane, generation));
        }
        Ok(())
    }

    fn enqueue_host_read(&mut self, context: &mut Rex3Context) -> Result<(), ResidentError> {
        if let Some(plane) = Self::host_read_plane(context) {
            self.prepare_host_read_snapshot(plane)?;
        } else {
            self.host_read.words.resize(PLANE_WORDS, 0);
        }
        let initial_cursor = context.host_cursor;
        self.enqueue_backend(context)?;
        let generation_start = Instant::now();
        let mode = rex3_shape::unpack(
            context.drawmode0.0,
            context.drawmode1.0,
            context.clipmode,
        );
        {
            let pointer = self.host_read.words.as_mut_ptr();
            let framebuffers = Framebuffers {
                rgb: pointer,
                aux: pointer,
            };
            draw_with_fb(context, &framebuffers, &mode);
            while context.host_len > 1 && context.host_cursor < context.host_len {
                let before = context.host_cursor;
                context.hostcnt = 0;
                draw_with_fb(context, &framebuffers, &mode);
                if context.host_cursor == before {
                    break;
                }
            }
        }
        let produced = context.host_cursor.checked_sub(initial_cursor)
            .ok_or(ResidentError::InvalidState("HostR cursor decreased"))?;
        let generation_ns = duration_ns(generation_start.elapsed()).map_err(ResidentError::InvalidState)?;
        self.run(|backend| backend.host_read_result(u64::from(produced), generation_ns))?;
        context.host_len = 1;
        context.host_cursor = 0;
        Ok(())
    }

    pub(crate) fn enqueue(&mut self, context: &mut Rex3Context) -> Result<(), ResidentError> {
        if context.drawmode0.opcode() == DRAWMODE0_OPCODE_READ {
            self.enqueue_host_read(context)
        } else {
            self.enqueue_backend(context)
        }
    }

    pub(crate) fn submit(&mut self) -> Result<(), ResidentError> {
        self.submit_for(FlushReason::Explicit)
    }

    pub(crate) fn submit_for(&mut self, reason: FlushReason) -> Result<(), ResidentError> {
        self.run(|backend| backend.submit_for(reason))
    }

    pub(crate) fn flush_if_due(&mut self, now: Instant) -> Result<(), ResidentError> {
        self.run(|backend| backend.flush_if_due(now))
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
                self.host_read = HostReadSnapshot::default();
                Ok(())
            }
            Err(error) => {
                self.fault = Some(error.clone());
                Err(error)
            }
        }
    }

    pub(crate) fn restore(&mut self, rgb: &[u32], aux: &[u32]) -> Result<(), ResidentError> {
        self.run(|backend| backend.restore(rgb, aux))?;
        self.host_read = HostReadSnapshot::default();
        Ok(())
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

/// The authoritative GPU implementation. Draws change the GPU planes directly;
/// reset and explicit restore are the only later raw-plane upload boundaries.
pub(crate) struct ResidentGpu {
    gpu: Arc<Gpu>,
    planes: Planes,
    pipelines: BatchPipelines,
    sets: Vec<RasterSet>,
    batch: Batch<RexCommand, RexStateSlot>,
    payload: PayloadBatch,
    serial_host_batch: bool,
    progress: Arc<Mutex<ResidentStatus>>,
    readback: wgpu::Buffer,
    /// Allocation identities only; the backend never dereferences them.
    rgb_shadow: usize,
    aux_shadow: usize,
    max_slots: u32,
    max_sets: NonZeroUsize,
    batch_started: Option<Instant>,
    current_coverage: Option<Coverage>,
}

struct RasterSet {
    payload: HostBuffer,
    group: wgpu::BindGroup,
}

impl ResidentGpu {
    /// The seed slices are the board's persistent CPU-shadow allocations.
    /// They must not be replaced while this backend is active.
    pub(crate) fn activate(gpu: Arc<Gpu>, rgb: &[u32], aux: &[u32]) -> Result<Self, ResidentError> {
        Self::activate_with_limits(gpu, rgb, aux, MAX_SLOTS, NonZeroUsize::new(MAX_SETS).unwrap(), None)
    }

    /// Profiling deliberately maps 64 bytes and waits for every nonempty
    /// batch. It is never inherited from the ordinary batch environment.
    pub(crate) fn enable_profiling(&mut self) -> Result<(), ResidentError> {
        let status = self.progress.lock();
        if status.accepted != status.submitted || status.workload.profiling {
            return Err(ResidentError::InvalidState("profiling requires an idle unprofiled raster"));
        }
        drop(status);
        self.batch.enable_profiling().map_err(ResidentError::Failed)?;
        self.progress.lock().workload.profiling = true;
        Ok(())
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

        let entry = |binding, read_only, bytes| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(bytes),
            },
            count: None,
        };
        let layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("resident rex3 raster planes"),
            entries: &[
                entry(0, false, PLANE_BYTES),
                entry(1, false, PLANE_BYTES),
                entry(2, true, MAX_PAYLOAD_BYTES),
            ],
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
                ("REX_COMMAND_BLOCK", COMMAND_BLOCK),
                ("REX_COMMAND_SPAN", COMMAND_SPAN),
                ("REX_COMMAND_CLEAR", COMMAND_CLEAR),
                ("REX_COMMAND_I_LINE", COMMAND_I_LINE),
                ("REX_COMMAND_F_LINE", COMMAND_F_LINE),
                ("REX_COMMAND_A_LINE", COMMAND_A_LINE),
                ("REX_COMMAND_COPY_BLOCK", COMMAND_COPY_BLOCK),
                ("REX_COMMAND_COPY_SPAN", COMMAND_COPY_SPAN),
                ("REX_COMMAND_COPY_I_LINE", COMMAND_COPY_I_LINE),
                ("REX_COMMAND_COPY_F_LINE", COMMAND_COPY_F_LINE),
                ("REX_COMMAND_COPY_A_LINE", COMMAND_COPY_A_LINE),
                ("REX_COMMAND_HOST_BLOCK", COMMAND_HOST_BLOCK),
                ("REX_COMMAND_HOST_SPAN", COMMAND_HOST_SPAN),
                ("REX_COMMAND_HOST_I_LINE", COMMAND_HOST_I_LINE),
                ("REX_COMMAND_HOST_F_LINE", COMMAND_HOST_F_LINE),
                ("REX_COMMAND_HOST_A_LINE", COMMAND_HOST_A_LINE),
                ("REX_FLAG_WRITE_RGB", FLAG_WRITE_RGB),
                ("REX_FLAG_WRITE_AUX", FLAG_WRITE_AUX),
                ("REX_FLAG_READ_AUX", FLAG_READ_AUX),
                ("REX_FLAG_LINE_Y_MAJOR", FLAG_LINE_Y_MAJOR),
                ("REX_FLAG_LINE_MAJOR_DECREASING", FLAG_LINE_MAJOR_DECREASING),
                ("REX_FLAG_SHADED", FLAG_SHADED),
                ("REX_FLAG_FAST_CLEAR", FLAG_FAST_CLEAR),
                ("REX_FLAG_FILL_FIRST_ROW_COMPLETE", FLAG_FILL_FIRST_ROW_COMPLETE),
                ("REX_FLAG_FILL_BLOCK", FLAG_FILL_BLOCK),
                ("REX_OCTANT_XDEC", OCTANT_XDEC),
                ("REX_OCTANT_YDEC", OCTANT_YDEC),
                ("REX_STATE_WORDS", STATE_WORDS as u32),
                ("REX_STATE_DRAW_MODE0", STATE_DRAW_MODE0 as u32),
                ("REX_STATE_DRAW_MODE1", STATE_DRAW_MODE1 as u32),
                ("REX_STATE_LINE_MODE", STATE_LINE_MODE as u32),
                ("REX_STATE_CLIP_MODE", STATE_CLIP_MODE as u32),
                ("REX_STATE_WRITE_MASK", STATE_WRITE_MASK as u32),
                ("REX_STATE_XYWIN", STATE_XYWIN as u32),
                ("REX_STATE_COLOR_BACK", STATE_COLOR_BACK as u32),
                ("REX_STATE_ALPHA_REF", STATE_ALPHA_REF as u32),
                ("REX_STATE_SMASK0X", STATE_SMASK0X as u32),
                ("REX_STATE_SMASK0Y", STATE_SMASK0Y as u32),
                ("REX_STATE_SMASK1X", STATE_SMASK1X as u32),
                ("REX_STATE_COLOR_RED", STATE_COLOR_RED as u32),
                ("REX_STATE_COLOR_ALPHA", STATE_COLOR_ALPHA as u32),
                ("REX_STATE_COLOR_GREEN", STATE_COLOR_GREEN as u32),
                ("REX_STATE_COLOR_BLUE", STATE_COLOR_BLUE as u32),
                ("REX_STATE_SLOPE_RED", STATE_SLOPE_RED as u32),
                ("REX_STATE_SLOPE_ALPHA", STATE_SLOPE_ALPHA as u32),
                ("REX_STATE_SLOPE_GREEN", STATE_SLOPE_GREEN as u32),
                ("REX_STATE_SLOPE_BLUE", STATE_SLOPE_BLUE as u32),
                ("REX_STATE_LINE_PATTERN", STATE_LINE_PATTERN as u32),
                ("REX_STATE_Z_PATTERN", STATE_Z_PATTERN as u32),
                ("REX_STATE_PATTERN_BITS", STATE_PATTERN_BITS as u32),
                ("REX_STATE_XYMOVE", STATE_XYMOVE as u32),
                ("REX_STATE_HOST_XSTART", STATE_HOST_XSTART as u32),
                ("REX_STATE_HOST_YSTART", STATE_HOST_YSTART as u32),
                ("REX_STATE_HOST_XEND", STATE_HOST_XEND as u32),
                ("REX_STATE_HOST_YEND", STATE_HOST_YEND as u32),
                ("REX_STATE_HOST_XSAVE", STATE_HOST_XSAVE as u32),
                ("REX_STATE_HOST_LENGTH", STATE_HOST_LENGTH as u32),
                ("REX_STATE_HOST_CURSOR", STATE_HOST_CURSOR as u32),
                ("REX_STATE_SOURCE_COLOR", STATE_SOURCE_COLOR as u32),
            ],
            layouts: &[&layout],
        }).map_err(ResidentError::Failed)?;
        let planes = Planes::new(&gpu);
        let sets = (0..max_sets.get())
            .map(|index| {
                let payload = HostBuffer::new(
                    &gpu,
                    &format!("resident rex3 payload {index}"),
                    MAX_PAYLOAD_BYTES,
                    wgpu::BufferUsages::STORAGE,
                );
                let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("resident rex3 raster planes and payload"),
                    layout: &layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: planes.rgb.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: planes.aux.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: payload.buffer.as_entire_binding(),
                        },
                    ],
                });
                RasterSet { payload, group }
            })
            .collect();
        let batch = Batch::new_bounded_unprofiled(
            &gpu,
            &pipelines,
            TileGrid::new(REX3_SCREEN_WIDTH as u32, REX3_SCREEN_HEIGHT as u32),
            max_slots,
            max_sets,
        );
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
            gpu,
            planes,
            pipelines,
            sets,
            batch,
            payload: PayloadBatch::new(MAX_PAYLOAD_WORDS),
            serial_host_batch: false,
            progress,
            readback,
            rgb_shadow: rgb.as_ptr() as usize,
            aux_shadow: aux.as_ptr() as usize,
            max_slots, max_sets,
            batch_started: None,
            current_coverage: None,
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
        self.submit_for(if reason == ObservationReason::HostRead {
            FlushReason::HostRead
        } else {
            FlushReason::Observation
        })?;
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

    fn enqueue_copy(&mut self, context: &mut Rex3Context) -> Result<(), ResidentError> {
        let line_shape = matches!(
            context.drawmode0.adrmode(),
            DRAWMODE0_ADRMODE_I_LINE
                | DRAWMODE0_ADRMODE_F_LINE
                | DRAWMODE0_ADRMODE_A_LINE
        );
        if line_shape {
            let prepared = prepare_copy_line(context).map_err(ResidentError::Unsupported)?;
            let line = prepared.line();
            self.current_coverage = line.and_then(|line|
                Coverage::new(line.min_x, line.min_y, line.max_x, line.max_y));
            if line.is_some() && prepared.payload().len() > MAX_PAYLOAD_WORDS {
                return Err(ResidentError::Unsupported("copy line payload length"));
            }
            self.submit_for(FlushReason::CopyHazard)?;
            let position = self.progress.lock().next_position()?;
            if let Some(line) = line {
                let slot = RexStateSlot::for_copy(context);
                let checkpoint = self.payload.checkpoint();
                let payload = self
                    .payload
                    .append_iter(prepared.payload())
                    .map_err(|Full| ResidentError::InvalidState("one copy line exceeds the payload arena"))?;
                let result = self.batch.prim_with_slot(
                    slot,
                    line.min_x,
                    line.min_y,
                    line.min_x,
                    line.min_y,
                    |slot| RexCommand::copy_line(slot, line, payload, cid_test(context.clipmode)),
                );
                match result {
                    Ok(Some(_)) => {}
                    Ok(None) => {
                        self.payload.rollback(checkpoint);
                        return Err(ResidentError::InvalidState("copy line missed its anchor tile"));
                    }
                    Err(Full) => {
                        self.payload.rollback(checkpoint);
                        return Err(ResidentError::InvalidState("one copy line exceeds an empty batch"));
                    }
                }
            }
            self.progress.lock().accept(position, line.map(|line| line.plane));
            prepared.finish(context);
            return self.submit_for(FlushReason::CopyHazard);
        }

        let prepared = prepare_copy_fill(context).map_err(ResidentError::Unsupported)?;
        let fill = prepared.fill();
        self.current_coverage = fill.and_then(|fill|
            Coverage::new(fill.min_x, fill.min_y, fill.max_x, fill.max_y));
        self.submit_for(FlushReason::CopyHazard)?;
        let position = self.progress.lock().next_position()?;
        if let Some(fill) = fill {
            let slot = RexStateSlot::for_copy(context);
            let result = self.batch.prim_with_slot(
                slot,
                fill.min_x,
                fill.min_y,
                fill.min_x,
                fill.min_y,
                |slot| RexCommand::copy_fill(slot, fill, cid_test(context.clipmode)),
            );
            match result {
                Ok(Some(_)) => {}
                Ok(None) => {
                    return Err(ResidentError::InvalidState("copy fill missed its anchor tile"));
                }
                Err(Full) => {
                    return Err(ResidentError::InvalidState("one copy fill exceeds an empty batch"));
                }
            }
        }
        self.progress.lock().accept(position, fill.map(|fill| fill.plane));
        prepared.finish(context);
        self.submit_for(FlushReason::CopyHazard)
    }

    fn enqueue_line(&mut self, context: &mut Rex3Context) -> Result<(), ResidentError> {
        let command = prepare_line(context).map_err(ResidentError::Unsupported)?;
        let position = self.progress.lock().next_position()?;
        let line = command.line();
        self.current_coverage = line.and_then(|line|
            Coverage::new(line.min_x, line.min_y, line.max_x, line.max_y));
        if let Some(line) = line {
            if command.payload().len() > MAX_PAYLOAD_WORDS {
                return Err(ResidentError::Unsupported("line payload length"));
            }
            let slot = RexStateSlot::for_line(context, line);
            loop {
                let checkpoint = self.payload.checkpoint();
                let payload = match self.payload.append_iter(command.payload()) {
                    Ok(payload) => payload,
                    Err(Full) if !self.batch.is_empty() => {
                        self.submit_for(FlushReason::Capacity)?;
                        continue;
                    }
                    Err(Full) => {
                        return Err(ResidentError::InvalidState("one line exceeds the payload arena"));
                    }
                };
                let result = self.batch.prim_with_slot(
                    slot,
                    line.min_x,
                    line.min_y,
                    line.max_x,
                    line.max_y,
                    |slot| {
                        RexCommand::line(
                            slot,
                            line,
                            payload,
                            cid_test(context.clipmode),
                            reads_destination(context),
                        )
                    },
                );
                match result {
                    Ok(Some(_)) => break,
                    Ok(None) => {
                        self.payload.rollback(checkpoint);
                        return Err(ResidentError::InvalidState("prepared line missed its raster grid"));
                    }
                    Err(Full) if !self.batch.is_empty() => {
                        self.payload.rollback(checkpoint);
                        self.submit_for(FlushReason::Capacity)?;
                    }
                    Err(Full) => {
                        self.payload.rollback(checkpoint);
                        return Err(ResidentError::InvalidState("one line exceeds an empty batch"));
                    }
                }
            }
        }
        self.progress.lock().accept(position, line.map(|line| line.plane));
        command.finish(context);
        Ok(())
    }

    fn enqueue_host_command(
        &mut self,
        context: &Rex3Context,
        slot: RexStateSlot,
        payload_words: usize,
        missed_anchor: &'static str,
        make_command: impl Fn(u32, PayloadRef) -> RexCommand,
    ) -> Result<(), ResidentError> {
        if !self.batch.is_empty() && !self.serial_host_batch {
            self.submit_for(FlushReason::HostTransition)?;
        }
        loop {
            let checkpoint = self.payload.checkpoint();
            let payload = match self.payload.append_iter((0..payload_words).map(|index| {
                let value = context.hostrw[index / 2];
                if index & 1 == 0 {
                    (value >> 32) as u32
                } else {
                    value as u32
                }
            })) {
                Ok(payload) => payload,
                Err(Full) if !self.batch.is_empty() => {
                    self.submit_for(FlushReason::Capacity)?;
                    continue;
                }
                Err(Full) => {
                    return Err(ResidentError::InvalidState(
                        "one host transfer exceeds the payload arena",
                    ));
                }
            };
            let result = self
                .batch
                .prim_with_slot(slot, 0, 0, 0, 0, |slot| make_command(slot, payload));
            match result {
                Ok(Some(_)) => {
                    self.serial_host_batch = true;
                    return Ok(());
                }
                Ok(None) => {
                    self.payload.rollback(checkpoint);
                    return Err(ResidentError::InvalidState(missed_anchor));
                }
                Err(Full) if !self.batch.is_empty() => {
                    self.payload.rollback(checkpoint);
                    self.submit_for(FlushReason::Capacity)?;
                }
                Err(Full) => {
                    self.payload.rollback(checkpoint);
                    return Err(ResidentError::InvalidState(
                        "one host transfer exceeds an empty batch",
                    ));
                }
            }
        }
    }

    fn enqueue_host_line(&mut self, context: &mut Rex3Context) -> Result<(), ResidentError> {
        let prepared = prepare_host_line(context).map_err(ResidentError::Unsupported)?;
        let payload_qwords = prepared.payload_qwords() as u64;
        let position = {
            let status = self.progress.lock();
            status
                .host_write_transfers
                .checked_add(1)
                .ok_or(ResidentError::InvalidState(
                    "HostW transfer counter exhausted",
                ))?;
            status.host_write_qwords.checked_add(payload_qwords).ok_or(
                ResidentError::InvalidState("HostW payload counter exhausted"),
            )?;
            status.next_position()?
        };
        let line = prepared.line();
        self.current_coverage = line.and_then(|_| prepared.coverage());
        if let Some(line) = line {
            let fast_clear = prepared.fast_clear();
            let payload_words = prepared
                .payload_qwords()
                .checked_mul(2)
                .ok_or(ResidentError::Unsupported("host payload length"))?;
            if payload_words > MAX_PAYLOAD_WORDS {
                return Err(ResidentError::Unsupported("host payload length"));
            }
            let slot = RexStateSlot::for_host_fill(context);
            let cid_test = cid_test(context.clipmode);
            let read_destination = !fast_clear && reads_destination(context);
            self.enqueue_host_command(
                context,
                slot,
                payload_words,
                "host line missed its anchor tile",
                |slot, payload| {
                    RexCommand::host_line(
                        slot,
                        line,
                        payload,
                        fast_clear,
                        cid_test,
                        read_destination,
                    )
                },
            )?;
        }
        {
            let mut status = self.progress.lock();
            status.host_write_transfers += 1;
            status.host_write_qwords += payload_qwords;
            status.accept(position, line.map(|line| line.plane));
        }
        prepared.finish(context);
        Ok(())
    }

    fn enqueue_host(&mut self, context: &mut Rex3Context) -> Result<(), ResidentError> {
        if matches!(
            context.drawmode0.adrmode(),
            DRAWMODE0_ADRMODE_I_LINE
                | DRAWMODE0_ADRMODE_F_LINE
                | DRAWMODE0_ADRMODE_A_LINE
        ) {
            return self.enqueue_host_line(context);
        }
        let prepared = prepare_host_fill(context).map_err(ResidentError::Unsupported)?;
        let payload_qwords = prepared.payload_qwords() as u64;
        let position = {
            let status = self.progress.lock();
            status
                .host_write_transfers
                .checked_add(1)
                .ok_or(ResidentError::InvalidState(
                    "HostW transfer counter exhausted",
                ))?;
            status.host_write_qwords.checked_add(payload_qwords).ok_or(
                ResidentError::InvalidState("HostW payload counter exhausted"),
            )?;
            status.next_position()?
        };
        let fill = prepared.fill();
        self.current_coverage = fill.and_then(|_| prepared.coverage());
        if let Some(fill) = fill {
            let fast_clear = prepared.fast_clear();
            let payload_words = prepared
                .payload_qwords()
                .checked_mul(2)
                .ok_or(ResidentError::Unsupported("host payload length"))?;
            if payload_words > MAX_PAYLOAD_WORDS {
                return Err(ResidentError::Unsupported("host payload length"));
            }
            let slot = RexStateSlot::for_host_fill(context);
            let cid_test = cid_test(context.clipmode);
            let read_destination = !fast_clear && reads_destination(context);
            self.enqueue_host_command(
                context,
                slot,
                payload_words,
                "host transfer missed its anchor tile",
                |slot, payload| {
                    RexCommand::host_fill(
                        slot,
                        fill,
                        payload,
                        fast_clear,
                        cid_test,
                        read_destination,
                    )
                },
            )?;
        }
        {
            let mut status = self.progress.lock();
            status.host_write_transfers += 1;
            status.host_write_qwords += payload_qwords;
            status.accept(position, fill.map(|fill| fill.plane));
        }
        prepared.finish(context);
        Ok(())
    }

    fn enqueue_fill(
        &mut self,
        context: &Rex3Context,
        fill: crate::dev::ng1::rex3_flat_fill::FlatFill,
        slot: RexStateSlot,
        words: &[u32],
        shaded: bool,
    ) -> Result<(), ResidentError> {
        loop {
            let checkpoint = self.payload.checkpoint();
            let payload = match self.payload.append(words) {
                Ok(payload) => payload,
                Err(Full) if !self.batch.is_empty() => {
                    self.submit_for(FlushReason::Capacity)?;
                    continue;
                }
                Err(Full) => {
                    return Err(ResidentError::InvalidState("one fill exceeds the payload arena"));
                }
            };
            let result = self.batch.prim_with_slot(
                slot,
                fill.min_x as i32,
                fill.min_y as i32,
                (fill.min_x + fill.width - 1) as i32,
                (fill.min_y + fill.height - 1) as i32,
                |slot| {
                    RexCommand::flat_fill(
                        slot,
                        fill,
                        payload,
                        shaded,
                        cid_test(context.clipmode),
                        fill.kind != FillKind::Clear && reads_destination(context),
                    )
                },
            );
            match result {
                Ok(Some(_)) => return Ok(()),
                Ok(None) => {
                    self.payload.rollback(checkpoint);
                    return Err(ResidentError::InvalidState("prepared fill missed its raster grid"));
                }
                Err(Full) if !self.batch.is_empty() => {
                    self.payload.rollback(checkpoint);
                    self.submit_for(FlushReason::Capacity)?;
                }
                Err(Full) => {
                    self.payload.rollback(checkpoint);
                    return Err(ResidentError::InvalidState("one fill exceeds an empty batch"));
                }
            }
        }
    }
    fn enqueue_inner(&mut self, context: &mut Rex3Context) -> Result<(), ResidentError> {
        self.gpu.check().map_err(gpu_error)?;
        let opcode = context.drawmode0.opcode();
        if opcode == DRAWMODE0_OPCODE_NOOP {
            let position = self.progress.lock().next_position()?;
            self.progress.lock().accept(position, None);
            context.host_len = 1;
            context.host_cursor = 0;
            return Ok(());
        }
        if opcode == DRAWMODE0_OPCODE_READ {
            let position = self.progress.lock().next_position()?;
            self.progress.lock().accept(position, None);
            return Ok(());
        }
        if opcode == DRAWMODE0_OPCODE_SCR2SCR {
            return self.enqueue_copy(context);
        }
        if opcode != DRAWMODE0_OPCODE_DRAW {
            return Err(ResidentError::Unsupported("opcode"));
        }
        if context.drawmode0.colorhost() || context.drawmode0.alphahost() {
            return self.enqueue_host(context);
        }
        if self.serial_host_batch {
            self.submit_for(FlushReason::HostTransition)?;
        }
        if matches!(
            context.drawmode0.adrmode(),
            DRAWMODE0_ADRMODE_I_LINE
                | DRAWMODE0_ADRMODE_F_LINE
                | DRAWMODE0_ADRMODE_A_LINE
        ) {
            return self.enqueue_line(context);
        }
        let command = prepare_resident_flat_fill(context).map_err(ResidentError::Unsupported)?;
        let position = self.progress.lock().next_position()?;
        let fill = command.fill();
        self.current_coverage = fill.and_then(|fill| Coverage::new(
            fill.min_x as i32, fill.min_y as i32,
            (fill.min_x + fill.width - 1) as i32,
            (fill.min_y + fill.height - 1) as i32,
        ));
        if let Some(fill) = fill {
            let slot = RexStateSlot::for_flat_fill(context, fill);
            let shade_payload = command.shade_payload();
            let words = shade_payload.unwrap_or_default();
            if words.len() > MAX_PAYLOAD_WORDS {
                return Err(ResidentError::Unsupported("fill shade payload length"));
            }
            self.enqueue_fill(context, fill, slot, words, shade_payload.is_some())?;
        }
        self.progress.lock().accept(position, fill.map(|fill| fill.plane));
        command.finish(context);
        Ok(())
    }

    fn enqueue_at(&mut self, context: &mut Rex3Context, now: Instant) -> Result<(), ResidentError> {
        self.flush_if_due(now)?;
        let start = Instant::now();
        let sample = DrawSample::from_context(context);
        let requested_qwords = u64::from(context.host_len.max(1));
        let (rgb, aux, host_qwords, submit_ns) = {
            let status = self.progress.lock();
            (status.rgb.last_write, status.aux.last_write, status.host_write_qwords,
                status.workload.submission_ns)
        };
        self.current_coverage = None;
        let result = self.enqueue_inner(context);
        let elapsed = duration_ns(start.elapsed()).map_err(ResidentError::InvalidState)?;
        let pending = {
            let mut status = self.progress.lock();
            let nested_submit = status.workload.submission_ns.checked_sub(submit_ns)
                .ok_or(ResidentError::InvalidState("submission time decreased"))?;
            let generation = elapsed.checked_sub(nested_submit)
                .ok_or(ResidentError::InvalidState("nested submission exceeds command time"))?;
            if matches!(result, Err(ResidentError::Unsupported(_))) {
                status.workload.unsupported = status.workload.unsupported.checked_add(1)
                    .ok_or(ResidentError::InvalidState("unsupported counter exhausted"))?;
            }
            if result.is_ok() {
                let noop = !sample.is_host_read()
                    && status.rgb.last_write == rgb && status.aux.last_write == aux;
                let transfer = if sample.is_host_write() {
                    status.host_write_qwords.checked_sub(host_qwords)
                        .ok_or(ResidentError::InvalidState("HostW qword counter decreased"))?
                } else {
                    requested_qwords
                };
                status.workload.record_draw(sample, self.current_coverage, noop, transfer, generation)
                    .map_err(ResidentError::InvalidState)?;
            }
            status.accepted != status.submitted
        };
        if pending && self.batch_started.is_none() {
            self.batch_started = Some(now);
        }
        result
    }
}

impl ResidentBackend for ResidentGpu {
    fn gpu(&self) -> Option<&Arc<Gpu>> { Some(&self.gpu) }

    fn poll_device_health(&mut self) -> Result<(), ResidentError> {
        self.gpu.instance.poll_all(false);
        self.gpu.check().map_err(gpu_error)
    }

    fn enqueue(&mut self, context: &mut Rex3Context) -> Result<(), ResidentError> {
        self.enqueue_at(context, Instant::now())
    }

    fn submit(&mut self) -> Result<(), ResidentError> {
        self.submit_for(FlushReason::Explicit)
    }

    fn flush_if_due(&mut self, now: Instant) -> Result<(), ResidentError> {
        if self.batch_started.is_some_and(|start|
            now.checked_duration_since(start).is_some_and(|age| age >= MAX_BATCH_AGE))
        {
            self.submit_for(FlushReason::Latency)?;
        }
        Ok(())
    }

    fn host_read_result(&mut self, qwords: u64, generation_ns: u64) -> Result<(), ResidentError> {
        self.progress.lock().workload.record_host_read(qwords, generation_ns)
            .map_err(ResidentError::InvalidState)
    }

    fn submit_for(&mut self, reason: FlushReason) -> Result<(), ResidentError> {
        self.gpu.check().map_err(gpu_error)?;
        let (accepted, submitted) = {
            let status = self.progress.lock();
            (status.accepted, status.submitted)
        };
        if accepted == submitted {
            self.batch_started = None;
            return Ok(());
        }
        if self.batch.is_empty() && !self.payload.is_empty() {
            return Err(ResidentError::InvalidState("payload exists without a command"));
        }
        let start = Instant::now();
        let commands = self.batch.len();
        let tile_pairs = self.batch.tile_pair_bound();
        let waits_before = self.batch.wait_stats();
        let profile_before = self.batch.timing.as_ref().map(|timing| (
            timing.batches, timing.tiles, timing.pairs, timing.bin_ns, timing.raster_ns,
        ));
        let result = if self.batch.is_empty() {
            self.batch.try_flush(&self.pipelines, &[])
        } else {
            let batch = &mut self.batch;
            let index = batch.try_next_set(&self.pipelines).map_err(gpu_error)?;
            let set = self
                .sets
                .get(index)
                .ok_or(ResidentError::InvalidState("batch selected an unknown payload set"))?;
            if !self.payload.is_empty() {
                set.payload.write(0, self.payload.bytes());
            }
            batch.try_flush(&self.pipelines, &[&set.group])
        };
        if result.is_ok() {
            self.payload.clear();
            self.serial_host_batch = false;
        }
        let submission = result.map_err(gpu_error)?;
        if submission.is_none() {
            // A run of no-op GOs still needs a queue marker for its stream
            // position, but it does not change either plane's generation.
            self.gpu.submit(self.gpu.encoder("resident rex3 no-op marker"));
        }
        self.progress.lock().submit(accepted);
        record_completion(&self.gpu, &self.progress, accepted);
        self.gpu.check().map_err(gpu_error)?;
        let elapsed = duration_ns(start.elapsed()).map_err(ResidentError::InvalidState)?;
        let waits_after = self.batch.wait_stats();
        let waits = waits_after.waits.checked_sub(waits_before.waits)
            .ok_or(ResidentError::InvalidState("bounded-set wait count decreased"))?;
        let wait_ns = waits_after.nanoseconds.checked_sub(waits_before.nanoseconds)
            .ok_or(ResidentError::InvalidState("bounded-set wait time decreased"))?;
        let profile_error = self.batch.timing.as_ref().and_then(|timing| timing.failure);
        let mut status = self.progress.lock();
        status.workload.record_flush(reason, commands, tile_pairs, elapsed, waits, wait_ns)
            .map_err(ResidentError::InvalidState)?;
        if commands != 0 {
            if let (Some(before), Some(after)) = (profile_before, self.batch.timing.as_ref()) {
                let workload = &mut status.workload;
                workload.profile_maps = workload.profile_maps.checked_add(1)
                    .ok_or(ResidentError::InvalidState("profiling map counter exhausted"))?;
                workload.profile_bytes = workload.profile_bytes.checked_add(64)
                    .ok_or(ResidentError::InvalidState("profiling byte counter exhausted"))?;
                let samples = after.batches.checked_sub(before.0)
                    .ok_or(ResidentError::InvalidState("profiling sample count decreased"))?;
                let tiles = after.tiles.checked_sub(before.1)
                    .ok_or(ResidentError::InvalidState("profiling busy tiles decreased"))?;
                let pairs = after.pairs.checked_sub(before.2)
                    .ok_or(ResidentError::InvalidState("profiling pairs decreased"))?;
                if samples > 1 || tiles > u64::from(self.batch.grid().tiles()) || pairs > tile_pairs || tiles > pairs {
                    return Err(ResidentError::InvalidState("profiling busy-tile bounds are invalid"));
                }
                let elapsed = |after: f64, before: f64| -> Result<u64, ResidentError> {
                    let delta = after - before;
                    if !delta.is_finite() || delta < 0.0 || delta >= u64::MAX as f64 {
                        return Err(ResidentError::InvalidState("profiling duration invalid"));
                    }
                    Ok(delta as u64)
                };
                workload.profile_samples = workload.profile_samples.checked_add(samples)
                    .ok_or(ResidentError::InvalidState("profiling sample counter exhausted"))?;
                workload.profile_busy_tiles = workload.profile_busy_tiles.checked_add(tiles)
                    .ok_or(ResidentError::InvalidState("profiling tile counter exhausted"))?;
                workload.profile_tile_pairs = workload.profile_tile_pairs.checked_add(pairs)
                    .ok_or(ResidentError::InvalidState("profiling pair counter exhausted"))?;
                workload.profile_bin_ns = workload.profile_bin_ns.checked_add(elapsed(after.bin_ns, before.3)?)
                    .ok_or(ResidentError::InvalidState("profiling bin time exhausted"))?;
                workload.profile_raster_ns = workload.profile_raster_ns.checked_add(elapsed(after.raster_ns, before.4)?)
                    .ok_or(ResidentError::InvalidState("profiling raster time exhausted"))?;
            }
        }
        drop(status);
        self.batch_started = None;
        if let Some(error) = profile_error {
            return Err(ResidentError::Failed(format!("resident profiling: {error}")));
        }
        Ok(())
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
        let rgb_is_shadow = rgb.as_ptr() as usize == self.rgb_shadow;
        let aux_is_shadow = aux.as_ptr() as usize == self.aux_shadow;
        let profiling = self.progress.lock().workload.profiling;
        let progress = Arc::new(Mutex::new(self.progress.lock().clone()));
        let mut replacement = Self::activate_with_limits(
            Arc::clone(&self.gpu),
            rgb,
            aux,
            self.max_slots,
            self.max_sets,
            Some((progress, self.readback.clone())),
        )?;
        if profiling {
            replacement.batch.enable_profiling().map_err(ResidentError::Failed)?;
        }
        replacement.gpu.wait(None).map_err(gpu_error)?;
        replacement.rgb_shadow = self.rgb_shadow;
        replacement.aux_shadow = self.aux_shadow;
        {
            let mut status = replacement.progress.lock();
            if !rgb_is_shadow { status.rgb.cpu_shadow = None; }
            if !aux_is_shadow { status.aux.cpu_shadow = None; }
            let seed = status.submitted;
            status.confirm(seed);
        }
        *self = replacement;
        Ok(())
    }

    fn restore(&mut self, rgb: &[u32], aux: &[u32]) -> Result<(), ResidentError> {
        // Submit the old prefix first so untouched suffixes include its writes.
        self.submit_for(FlushReason::Restore)?;
        let epoch = next_epoch()?;
        let rgb_words = rgb.len().min(PLANE_WORDS);
        let aux_words = aux.len().min(PLANE_WORDS);
        let mut next = self.progress.lock().restored(epoch, rgb_words, aux_words)?;
        if rgb_words != 0 {
            self.gpu.write(&self.planes.rgb, 0, bytemuck::cast_slice(&rgb[..rgb_words]));
        }
        if aux_words != 0 {
            self.gpu.write(&self.planes.aux, 0, bytemuck::cast_slice(&aux[..aux_words]));
        }
        let submission = self.gpu.submit(self.gpu.encoder("resident rex3 restore"));
        self.gpu.wait(Some(submission)).map_err(gpu_error)?;
        let seed = next.submitted;
        next.confirm(seed);
        *self.progress.lock() = next;
        Ok(())
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
