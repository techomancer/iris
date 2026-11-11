//! Opt-in REX3 host-GPU prototype for replay and experimental live execution.

use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytemuck::{Pod, Zeroable};
use parking_lot::Mutex;
use wgpu::util::DeviceExt;

use super::rex3::{
    DrawMode0, DrawMode1, Rex3Context, CLIPMODE_CIDMATCH_MASK, CLIPMODE_CIDMATCH_SHIFT,
    CLIPMODE_ENSMASK_MASK, DRAWMODE0_ADRMODE_BLOCK, DRAWMODE0_ADRMODE_BLOCK_SH,
    DRAWMODE0_ADRMODE_SPAN, DRAWMODE0_ADRMODE_SPAN_SH, DRAWMODE0_OPCODE_DRAW,
    DRAWMODE1_COMPARE_DISABLE, DRAWMODE1_COMPARE_DISABLE_SH, DRAWMODE1_DRAWDEPTH_24,
    DRAWMODE1_LOGICOP_SRC, DRAWMODE1_LOGICOP_SRC_SH, DRAWMODE1_PLANES_RGB, OCTANT_XDEC,
    OCTANT_YDEC, REX3_COORD_BIAS, REX3_SCREEN_HEIGHT, REX3_SCREEN_WIDTH,
};
use super::rex3_dirty::{DamagePlane, DamageRect, DamageSnapshot};

#[path = "rex3_gpu_telemetry.rs"]
mod telemetry;
pub(crate) use telemetry::FlushReason;
#[cfg(test)]
pub(crate) use telemetry::{
    abba_log_rate_ratio, live_aa_stop, live_gate_stats, log_rate_ratio, paired_order_shift,
    paired_rate_stats, paired_rate_stats_with_alpha, require_usable_timestamps, LiveAaOptions,
    LiveAaStop,
};
use telemetry::{record_timestamp, timestamp_duration_ns};
pub use telemetry::{FlushCounts, ReplayReport, StageTelemetry, TimestampSupport, TimestampTotals};

const FB_STRIDE: u32 = 2048;
const FB_PIXELS: usize = FB_STRIDE as usize * REX3_SCREEN_HEIGHT as usize;
pub(crate) const DEFAULT_MAX_BATCH_DRAWS: usize = 16384;
const MAX_CONFIGURED_BATCH_DRAWS: usize = 65536;
// Outstanding accepted draws are journaled so a GPU failure can reapply them
// to the CPU plane. These caps bound that replay; reaching one forces a
// checkpoint before the next draw is accepted.
const MAX_JOURNAL_DRAWS: usize = 65536;
const MAX_JOURNAL_REPLAY_PIXELS: usize = 4 * 1024 * 1024;
const MAP_TIMEOUT: Duration = Duration::from_secs(10);
const MAP_SPIN: Duration = Duration::from_millis(2);
const PROBE_STACK_BYTES: usize = 8 * 1024 * 1024;
const DRAW_TIMING_SAMPLE_INTERVAL: u64 = 256;
const MAX_PENDING_TIMESTAMP_SUBMISSIONS: usize = 64;

// The GPU plane covers only the visible screen, stored tile-major so the
// shader can give each 32x32 tile to one workgroup and a checkpoint can copy
// back only damaged tiles.
const TILE_SIZE: u32 = 32;
const TILES_X: u32 = REX3_SCREEN_WIDTH as u32 / TILE_SIZE;
const TILES_Y: u32 = REX3_SCREEN_HEIGHT as u32 / TILE_SIZE;
const TILE_COUNT: usize = (TILES_X * TILES_Y) as usize;
const _: () = assert!(REX3_SCREEN_WIDTH as u32 % TILE_SIZE == 0);
const _: () = assert!(REX3_SCREEN_HEIGHT as u32 % TILE_SIZE == 0);
// Tile references per batch; a batch is submitted early rather than exceed it.
const MAX_BIN_ENTRIES: usize = 1 << 20;
const BIN_WORDS: usize = TILE_COUNT + 1 + MAX_BIN_ENTRIES;
const BIN_BYTES: u64 = (BIN_WORDS * std::mem::size_of::<u32>()) as u64;
const TILE_PIXELS: usize = (TILE_SIZE * TILE_SIZE) as usize;
const TILE_BYTES: u64 = (TILE_PIXELS * std::mem::size_of::<u32>()) as u64;
const GPU_PLANE_WORDS: usize = TILE_COUNT * TILE_PIXELS;
pub(crate) const GPU_PLANE_BYTES: u64 = (GPU_PLANE_WORDS * std::mem::size_of::<u32>()) as u64;

/// Lay the screen area of a linear CPU plane out tile-major for upload.
fn tile_major_from_linear(linear: &[u32], tiled: &mut Vec<u32>) {
    tiled.clear();
    tiled.reserve(GPU_PLANE_WORDS);
    for tile_row in 0..TILES_Y {
        for tile_column in 0..TILES_X {
            for row in 0..TILE_SIZE {
                let start =
                    ((tile_row * TILE_SIZE + row) * FB_STRIDE + tile_column * TILE_SIZE) as usize;
                tiled.extend_from_slice(&linear[start..start + TILE_SIZE as usize]);
            }
        }
    }
}

/// Copy one tile of a tile-major plane back into a linear CPU plane.
fn copy_tile_to_linear(tile: u32, tiled: &[u32], linear: &mut [u32]) {
    let x = (tile % TILES_X) * TILE_SIZE;
    let y = (tile / TILES_X) * TILE_SIZE;
    let source = tile as usize * TILE_PIXELS;
    for row in 0..TILE_SIZE {
        let from = source + (row * TILE_SIZE) as usize;
        let to = ((y + row) * FB_STRIDE + x) as usize;
        linear[to..to + TILE_SIZE as usize]
            .copy_from_slice(&tiled[from..from + TILE_SIZE as usize]);
    }
}

/// Tile ranges (in tile-major order) covering `damage`, sorted and merged.
fn damaged_tile_runs(
    damage: &DamageSnapshot,
    rectangles: &mut Vec<DamageRect>,
    runs: &mut Vec<std::ops::Range<u32>>,
) {
    damage.rectangles_into(DamagePlane::Rgb, rectangles);
    runs.clear();
    for rect in rectangles.iter() {
        let first_column = (rect.x as u32 / TILE_SIZE).min(TILES_X);
        let end_column = ((rect.x + rect.width) as u32)
            .div_ceil(TILE_SIZE)
            .min(TILES_X);
        let first_row = (rect.y as u32 / TILE_SIZE).min(TILES_Y);
        let end_row = ((rect.y + rect.height) as u32)
            .div_ceil(TILE_SIZE)
            .min(TILES_Y);
        if first_column < end_column {
            for row in first_row..end_row {
                runs.push(row * TILES_X + first_column..row * TILES_X + end_column);
            }
        }
    }
    runs.sort_unstable_by_key(|run| run.start);
    let mut merged = 0;
    for index in 0..runs.len() {
        if merged != 0 && runs[index].start <= runs[merged - 1].end {
            runs[merged - 1].end = runs[merged - 1].end.max(runs[index].end);
        } else {
            runs[merged] = runs[index].clone();
            merged += 1;
        }
    }
    runs.truncate(merged);
}

/// Reject an adapter whose limits cannot hold the engine's buffers or dispatch.
fn check_limits(
    info: &wgpu::AdapterInfo,
    limits: &wgpu::Limits,
    max_batch_draws: usize,
) -> Result<(), String> {
    let parameter_bytes = (max_batch_draws * std::mem::size_of::<FlatFill>()) as u64;
    let largest = GPU_PLANE_BYTES.max(parameter_bytes).max(BIN_BYTES);
    if limits.max_storage_buffer_binding_size < largest || limits.max_buffer_size < largest {
        return Err(format!(
            "{} only exposes {}-byte storage bindings and {}-byte buffers; REX3 needs {largest}",
            info.name, limits.max_storage_buffer_binding_size, limits.max_buffer_size
        ));
    }
    if limits.max_compute_workgroups_per_dimension < TILE_COUNT as u32
        || limits.max_storage_buffers_per_shader_stage < 3
        || limits.max_compute_invocations_per_workgroup < 256
    {
        return Err(format!(
            "{} cannot dispatch {TILE_COUNT} workgroups of 256 invocations over 3 storage buffers",
            info.name
        ));
    }
    Ok(())
}

fn elapsed_ns(start: Instant) -> u64 {
    start.elapsed().as_nanos().min(u64::MAX as u128) as u64
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RequestedBackend {
    Auto,
    Vulkan,
    Metal,
    Dx12,
}

impl RequestedBackend {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "auto" => Ok(Self::Auto),
            "vulkan" => Ok(Self::Vulkan),
            "metal" => Ok(Self::Metal),
            "dx12" => Ok(Self::Dx12),
            other => Err(format!(
                "REX3_GPU_BACKEND={other}: want auto, vulkan, metal, or dx12"
            )),
        }
    }

    fn candidates(self) -> Vec<(&'static str, wgpu::Backends)> {
        match self {
            Self::Vulkan => vec![("vulkan", wgpu::Backends::VULKAN)],
            Self::Metal => vec![("metal", wgpu::Backends::METAL)],
            Self::Dx12 => vec![("dx12", wgpu::Backends::DX12)],
            Self::Auto if cfg!(target_os = "windows") => vec![
                ("vulkan", wgpu::Backends::VULKAN),
                ("dx12", wgpu::Backends::DX12),
            ],
            Self::Auto if cfg!(target_os = "macos") => {
                vec![("metal", wgpu::Backends::METAL)]
            }
            Self::Auto => vec![("vulkan", wgpu::Backends::VULKAN)],
        }
    }
}

#[derive(Clone, Debug)]
pub struct AdapterDiagnostics {
    pub backend: wgpu::Backend,
    pub name: String,
    pub device_type: wgpu::DeviceType,
    pub driver: String,
    pub driver_info: String,
    pub max_storage_buffer_binding_size: u64,
    pub max_compute_workgroups_per_dimension: u32,
    pub max_batch_draws: usize,
    pub timestamps: TimestampSupport,
}

impl fmt::Display for AdapterDiagnostics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "backend={:?} adapter={:?} type={:?} driver={:?} driver_info={:?} \
             max_storage_buffer={} max_workgroups={} batch_draws={} \
             gpu_timestamps={:?} gpu_upload_timing=unavailable(queue.write_buffer)",
            self.backend,
            self.name,
            self.device_type,
            self.driver,
            self.driver_info,
            self.max_storage_buffer_binding_size,
            self.max_compute_workgroups_per_dimension,
            self.max_batch_draws,
            self.timestamps,
        )
    }
}

#[derive(Clone, Debug)]
pub struct ProbeResult {
    pub diagnostics: AdapterDiagnostics,
    pub report: ReplayReport,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnqueueResult {
    Enqueued,
    FlushRequired,
    Unsupported(&'static str),
}

enum FlatFillMatch {
    Supported(Option<FlatFill>),
    Unsupported(&'static str),
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ProfilingOptions {
    pub telemetry: bool,
    pub timestamps: bool,
}

fn profiling_flag(name: &str) -> Result<bool, String> {
    match std::env::var(name) {
        Err(std::env::VarError::NotPresent) => Ok(false),
        Ok(value) if value == "0" => Ok(false),
        Ok(value) if value == "1" => Ok(true),
        Ok(value) => Err(format!("{name}={value}: expected 0 or 1")),
        Err(error) => Err(format!("read {name}: {error}")),
    }
}

impl ProfilingOptions {
    fn from_env() -> Result<Self, String> {
        let timestamps = profiling_flag("REX3_GPU_TIMESTAMPS")?;
        Ok(Self {
            telemetry: profiling_flag("REX3_GPU_TELEMETRY")? || timestamps,
            timestamps,
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct FlatFill {
    min_x: u32,
    min_y: u32,
    width: u32,
    height: u32,
    color: u32,
    write_mask: u32,
    stride: u32,
    plane_len: u32,
}

impl FlatFill {
    fn replay_pixels(self) -> usize {
        self.width as usize * self.height as usize
    }

    fn damage(self, damage: &mut DamageSnapshot) {
        if let Some(rect) = DamageRect::clipped(
            self.min_x as i64,
            self.min_y as i64,
            (self.min_x + self.width) as i64,
            (self.min_y + self.height) as i64,
        ) {
            damage.mark_rect(DamagePlane::Rgb, rect);
        }
    }

    /// Apply this fill to a linear CPU plane, exactly as the shader does.
    fn replay(self, rgb: &mut [u32]) {
        let value = self.color & self.write_mask;
        for y in self.min_y..self.min_y + self.height {
            let start = (y * self.stride + self.min_x) as usize;
            let end = start + self.width as usize;
            for pixel in &mut rgb[start..end] {
                *pixel = (*pixel & !self.write_mask) | value;
            }
        }
    }

    /// Tile columns and rows this (already screen-clipped) fill covers.
    fn tiles(self) -> (std::ops::Range<u32>, std::ops::Range<u32>) {
        (
            self.min_x / TILE_SIZE..(self.min_x + self.width - 1) / TILE_SIZE + 1,
            self.min_y / TILE_SIZE..(self.min_y + self.height - 1) / TILE_SIZE + 1,
        )
    }

    fn tile_count(self) -> usize {
        let (columns, rows) = self.tiles();
        columns.len() * rows.len()
    }
}

/// Per-tile lists of the batch's fills, in submission order.
///
/// Encoded as one `u32` array for the shader: `TILE_COUNT + 1` offsets into
/// the array itself, then each tile's fill indices. A tile's list order is
/// the order its pixels must be written, so binning preserves draw order
/// without any GPU synchronization.
struct TileBins {
    tiles: Vec<Vec<u32>>,
    entries: usize,
    encoded: Vec<u32>,
}

impl TileBins {
    fn new() -> Self {
        Self {
            tiles: vec![Vec::new(); TILE_COUNT],
            entries: 0,
            encoded: Vec::with_capacity(TILE_COUNT + 1),
        }
    }

    fn fits(&self, fill: FlatFill) -> bool {
        self.entries + fill.tile_count() <= MAX_BIN_ENTRIES
    }

    fn push(&mut self, draw: u32, fill: FlatFill) {
        let (columns, rows) = fill.tiles();
        for row in rows {
            let first = (row * TILES_X) as usize;
            for column in columns.clone() {
                self.tiles[first + column as usize].push(draw);
            }
        }
        self.entries += fill.tile_count();
    }

    /// Encode the batch for upload and leave the bins empty.
    fn encode(&mut self) -> &[u32] {
        self.encoded.clear();
        self.encoded.resize(TILE_COUNT + 1, 0);
        for (tile, draws) in self.tiles.iter_mut().enumerate() {
            self.encoded[tile] = self.encoded.len() as u32;
            self.encoded.extend_from_slice(draws);
            draws.clear();
        }
        self.encoded[TILE_COUNT] = self.encoded.len() as u32;
        debug_assert_eq!(self.encoded.len(), TILE_COUNT + 1 + self.entries);
        self.entries = 0;
        &self.encoded
    }

    fn clear(&mut self) {
        if self.entries != 0 {
            self.tiles.iter_mut().for_each(Vec::clear);
            self.entries = 0;
        }
    }
}

fn remember_first_error(slot: &Mutex<Option<String>>, error: String) -> String {
    let mut slot = slot.lock();
    slot.get_or_insert(error).clone()
}

/// The wgpu objects of one engine. Kept together so a failed engine can be
/// leaked as a unit: destroying a lost device can panic or block inside
/// wgpu, and the CPU renderer already holds the authoritative plane by then.
struct GpuBackend {
    pipeline: wgpu::ComputePipeline,
    rgb: wgpu::Buffer,
    readback: wgpu::Buffer,
    parameters: wgpu::Buffer,
    bins: wgpu::Buffer,
    // References every buffer above.
    bind_group: wgpu::BindGroup,
    queue: wgpu::Queue,
    device: wgpu::Device,
    instance: wgpu::Instance,
}

struct TimestampResources {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    count: u32,
    compute_index: Option<u32>,
    transfer_index: Option<u32>,
}

struct PendingTimestamps {
    readback: wgpu::Buffer,
    bytes: u64,
    compute_index: Option<u32>,
    transfer_index: Option<u32>,
}

pub struct Rex3Gpu {
    /// `None` once the engine has failed and its objects were leaked.
    backend: Option<GpuBackend>,
    pending: Vec<FlatFill>,
    bins: TileBins,
    pending_damage: DamageSnapshot,
    submitted_damage: DamageSnapshot,
    readback_rects: Vec<DamageRect>,
    // Tile ranges copied to the readback buffer by the pending checkpoint.
    readback_runs: Vec<std::ops::Range<u32>>,
    upload_staging: Vec<u32>,
    // Pixel effects of every draw accepted since the last checkpoint, in
    // order. Empty draws are kept while earlier work is outstanding.
    journal: Vec<Option<FlatFill>>,
    journal_limit: usize,
    journal_pixels: usize,
    checkpoint_pending: bool,
    gpu_current: bool,
    submitted_work: bool,
    disabled: bool,
    max_batch_draws: usize,
    diagnostics: AdapterDiagnostics,
    report: ReplayReport,
    async_error: Arc<Mutex<Option<String>>>,
    timestamp_period: f32,
    timestamp_pending: Vec<PendingTimestamps>,
}

impl Rex3Gpu {
    pub fn new(backend: &str) -> Result<Self, String> {
        Self::new_with_options(backend, ProfilingOptions::from_env()?)
    }

    pub(crate) fn new_with_options(
        backend: &str,
        options: ProfilingOptions,
    ) -> Result<Self, String> {
        let requested = RequestedBackend::parse(backend)?;
        let max_batch_draws = match std::env::var("REX3_GPU_BATCH_DRAWS") {
            Ok(value) => {
                let parsed = value.parse::<usize>().map_err(|error| {
                    format!("REX3_GPU_BATCH_DRAWS={value}: expected an integer: {error}")
                })?;
                if !(1..=MAX_CONFIGURED_BATCH_DRAWS).contains(&parsed) {
                    return Err(format!(
                        "REX3_GPU_BATCH_DRAWS={value}: want 1..={MAX_CONFIGURED_BATCH_DRAWS}"
                    ));
                }
                parsed
            }
            Err(std::env::VarError::NotPresent) => DEFAULT_MAX_BATCH_DRAWS,
            Err(error) => return Err(format!("read REX3_GPU_BATCH_DRAWS: {error}")),
        };
        let mut failures = Vec::new();
        for (name, backends) in requested.candidates() {
            match Self::new_for_backend(backends, max_batch_draws, options) {
                Ok(gpu) => return Ok(gpu),
                Err(error) => failures.push(format!("{name}: {error}")),
            }
        }
        Err(format!(
            "no usable REX3 GPU backend for {backend}: {}",
            failures.join("; ")
        ))
    }

    fn new_for_backend(
        backends: wgpu::Backends,
        max_batch_draws: usize,
        options: ProfilingOptions,
    ) -> Result<Self, String> {
        let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle();
        instance_desc.backends = backends;
        let instance = wgpu::Instance::new(instance_desc);
        let request_adapter = || {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            }))
        };
        let adapter = request_adapter().map_err(|error| format!("request adapter: {error}"))?;

        let mut info = adapter.get_info();
        let mut limits = adapter.limits();
        check_limits(&info, &limits, max_batch_draws)?;

        let features = adapter.features();
        let mut timestamp_support = if !options.timestamps {
            TimestampSupport::Disabled
        } else if !features.contains(wgpu::Features::TIMESTAMP_QUERY) {
            TimestampSupport::Unsupported("adapter does not expose TIMESTAMP_QUERY")
        } else {
            TimestampSupport::Enabled {
                readback_transfer: features
                    .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS),
            }
        };
        let mut required_features = wgpu::Features::empty();
        if let TimestampSupport::Enabled { readback_transfer } = timestamp_support {
            required_features |= wgpu::Features::TIMESTAMP_QUERY;
            if readback_transfer {
                required_features |= wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
            }
        }
        let mut descriptor = wgpu::DeviceDescriptor {
            label: Some("IRIS REX3 replay device"),
            required_features,
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        };
        let (device, queue) = match pollster::block_on(adapter.request_device(&descriptor)) {
            Ok(device) => device,
            Err(error) if !required_features.is_empty() => {
                timestamp_support = TimestampSupport::Unavailable(format!(
                    "device rejected optional timestamp features: {error}"
                ));
                descriptor.required_features = wgpu::Features::empty();
                let fallback = request_adapter().map_err(|retry| {
                    format!("timestamp device request failed ({error}); request fallback adapter: {retry}")
                })?;
                info = fallback.get_info();
                limits = fallback.limits();
                check_limits(&info, &limits, max_batch_draws)?;
                pollster::block_on(fallback.request_device(&descriptor)).map_err(|retry| {
                    format!("timestamp device request failed ({error}); request device for {} without timestamps: {retry}", info.name)
                })?
            }
            Err(error) => return Err(format!("request device for {}: {error}", info.name)),
        };
        let timestamp_period = queue.get_timestamp_period();
        if matches!(timestamp_support, TimestampSupport::Enabled { .. })
            && (!timestamp_period.is_finite() || timestamp_period <= 0.0)
        {
            timestamp_support = TimestampSupport::Unavailable(format!(
                "invalid GPU timestamp period {timestamp_period}"
            ));
        }

        let async_error = Arc::new(Mutex::new(None));
        let uncaptured = Arc::clone(&async_error);
        device.on_uncaptured_error(Arc::new(move |error| {
            remember_first_error(&uncaptured, format!("uncaptured wgpu error: {error}"));
        }));
        let lost = Arc::clone(&async_error);
        device.set_device_lost_callback(move |reason, message| {
            remember_first_error(&lost, format!("wgpu device lost ({reason:?}): {message}"));
        });

        let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("IRIS REX3 flat-fill shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("rex3_gpu.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("IRIS REX3 flat-fill pipeline"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let storage = |label, size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let rgb = storage(
            "IRIS REX3 RGB plane",
            GPU_PLANE_BYTES,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );
        let readback = storage(
            "IRIS REX3 RGB readback",
            GPU_PLANE_BYTES,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let parameters = storage(
            "IRIS REX3 flat-fill batch parameters",
            (max_batch_draws * std::mem::size_of::<FlatFill>()) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let bins = storage(
            "IRIS REX3 flat-fill tile bins",
            BIN_BYTES,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        // Every binding is a whole fixed buffer: the per-batch bin offsets
        // say which fills to read, so one bind group serves every batch.
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("IRIS REX3 flat-fill bind group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: rgb.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: parameters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bins.as_entire_binding(),
                },
            ],
        });
        device
            .poll(wgpu::PollType::Poll)
            .map_err(|error| format!("poll after pipeline creation: {error}"))?;
        if let Some(error) = pollster::block_on(error_scope.pop()) {
            return Err(format!("create flat-fill pipeline and buffers: {error}"));
        }
        if let Some(error) = async_error.lock().clone() {
            return Err(error);
        }

        Ok(Self {
            backend: Some(GpuBackend {
                pipeline,
                rgb,
                readback,
                parameters,
                bins,
                bind_group,
                queue,
                device,
                instance,
            }),
            pending: Vec::new(),
            bins: TileBins::new(),
            pending_damage: DamageSnapshot::default(),
            submitted_damage: DamageSnapshot::default(),
            readback_rects: Vec::new(),
            readback_runs: Vec::new(),
            upload_staging: Vec::new(),
            journal: Vec::new(),
            journal_limit: MAX_JOURNAL_DRAWS,
            journal_pixels: 0,
            checkpoint_pending: false,
            gpu_current: false,
            submitted_work: false,
            disabled: false,
            max_batch_draws,
            diagnostics: AdapterDiagnostics {
                backend: info.backend,
                name: info.name,
                device_type: info.device_type,
                driver: info.driver,
                driver_info: info.driver_info,
                max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
                max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
                max_batch_draws,
                timestamps: timestamp_support.clone(),
            },
            report: ReplayReport {
                timestamp_support: timestamp_support.clone(),
                telemetry: (options.telemetry || options.timestamps)
                    .then(|| Self::new_telemetry(&timestamp_support)),
                ..ReplayReport::fresh()
            },
            async_error,
            timestamp_period,
            timestamp_pending: Vec::new(),
        })
    }

    fn new_telemetry(timestamps: &TimestampSupport) -> StageTelemetry {
        let mut telemetry = StageTelemetry::default();
        if let TimestampSupport::Enabled { readback_transfer } = timestamps {
            telemetry.gpu_compute = Some(TimestampTotals::default());
            if *readback_transfer {
                telemetry.gpu_readback_transfer = Some(TimestampTotals::default());
            }
        }
        telemetry
    }

    pub fn diagnostics(&self) -> &AdapterDiagnostics {
        &self.diagnostics
    }

    pub fn report(&self) -> ReplayReport {
        let mut report = self.report.clone();
        report.pending_draws = self.pending.len();
        report.timestamp_support = self.diagnostics.timestamps.clone();
        if report.error.is_none() {
            report.error = self.async_error.lock().clone();
        }
        report
    }

    #[cfg(test)]
    pub(crate) fn set_telemetry_enabled(&mut self, enabled: bool) -> Result<(), String> {
        if !self.pending.is_empty() || self.submitted_work {
            return Err("cannot change telemetry while GPU work is pending".into());
        }
        if !enabled
            && matches!(
                self.diagnostics.timestamps,
                TimestampSupport::Enabled { .. }
            )
        {
            return Err("cannot disable telemetry while GPU timestamps are enabled".into());
        }
        if enabled {
            if self.report.telemetry.is_none() {
                self.report.telemetry = Some(Self::new_telemetry(&self.diagnostics.timestamps));
                self.report.advance_counter_epoch();
            }
        } else if self.report.telemetry.is_some() {
            self.report.telemetry = None;
            self.report.advance_counter_epoch();
        }
        Ok(())
    }

    pub fn note_fallback(&mut self, reason: &'static str) {
        self.report.fallback_draws += 1;
        *self
            .report
            .fallback_reasons
            .entry(reason.to_owned())
            .or_default() += 1;
    }

    /// True once a failure has been recovered; the engine accepts no more work.
    pub(crate) fn is_disabled(&self) -> bool {
        self.disabled
    }

    #[cfg(test)]
    pub(crate) fn set_test_limits(&mut self, batch: usize, journal: usize) {
        assert!(batch > 0 && journal > 0);
        assert!(batch <= self.max_batch_draws);
        assert!(self.pending.is_empty() && self.journal.is_empty() && !self.submitted_work);
        self.max_batch_draws = batch;
        self.journal_limit = journal;
    }

    /// Record `error` as if a wgpu callback had reported it.
    #[cfg(test)]
    pub(crate) fn inject_error_for_test(&self, error: &str) {
        remember_first_error(&self.async_error, error.to_owned());
    }

    fn backend(&self) -> &GpuBackend {
        self.backend.as_ref().expect("active GPU backend")
    }

    /// Run device callbacks without blocking and report the first error.
    fn poll(&self) -> Result<(), String> {
        self.check_async_error()?;
        // Instance polling reports HAL failures through the device-lost
        // callback rather than the fatal path of a blocking device poll.
        self.backend().instance.poll_all(false);
        self.check_async_error()
    }

    fn wait_for_mapping(
        &self,
        receiver: &std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
        description: &str,
    ) -> Result<(), String> {
        let start = Instant::now();
        loop {
            self.poll()?;
            match receiver.try_recv() {
                Ok(Ok(())) => return Ok(()),
                Ok(Err(error)) => {
                    return Err(remember_first_error(
                        &self.async_error,
                        format!("map {description}: {error}"),
                    ));
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    return Err(remember_first_error(
                        &self.async_error,
                        format!("{description} callback disconnected"),
                    ));
                }
                Err(std::sync::mpsc::TryRecvError::Empty) if start.elapsed() >= MAP_TIMEOUT => {
                    return Err(remember_first_error(
                        &self.async_error,
                        format!("wait for {description} timed out"),
                    ));
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
            // A checkpoint usually completes within a millisecond, and a
            // timed sleep can overshoot that several times on some hosts.
            if start.elapsed() < MAP_SPIN {
                std::thread::yield_now();
            } else {
                std::thread::sleep(Duration::from_micros(100));
            }
        }
    }

    /// Attempt one draw. On `FlushRequired`, call [`Self::submit`], or
    /// [`Self::flush`] when [`Self::checkpoint_required`], and retry the
    /// unchanged context. Accepted draws advance the context immediately.
    pub fn try_enqueue(
        &mut self,
        context: &mut Rex3Context,
        rgb: &[u32],
    ) -> Result<EnqueueResult, String> {
        self.check_async_error()?;
        let sample_start = self.report.telemetry.as_mut().and_then(|telemetry| {
            telemetry.match_attempts += 1;
            (telemetry.match_attempts % DRAW_TIMING_SAMPLE_INTERVAL == 0).then(Instant::now)
        });
        let fill = flat_fill(context);
        let enqueue_start = sample_start.map(|start| {
            let now = Instant::now();
            if let Some(telemetry) = &mut self.report.telemetry {
                telemetry.sampled_match_ns += now.duration_since(start).as_nanos() as u64;
            }
            now
        });
        let result = self.enqueue_matched(context, rgb, fill);
        if let (Some(start), Some(telemetry)) = (enqueue_start, &mut self.report.telemetry) {
            telemetry.sampled_draws += 1;
            telemetry.sampled_enqueue_ns += elapsed_ns(start);
        }
        result
    }

    /// The journal is full, or a draw was refused until a checkpoint.
    pub(crate) fn checkpoint_required(&self) -> bool {
        self.journal.len() >= self.journal_limit || self.checkpoint_pending
    }

    fn enqueue_matched(
        &mut self,
        context: &mut Rex3Context,
        rgb: &[u32],
        fill: FlatFillMatch,
    ) -> Result<EnqueueResult, String> {
        let fill = match fill {
            FlatFillMatch::Supported(fill) => fill,
            FlatFillMatch::Unsupported(reason) => {
                return Ok(EnqueueResult::Unsupported(reason));
            }
        };

        let replay_pixels = fill.map_or(0, FlatFill::replay_pixels);
        if replay_pixels > MAX_JOURNAL_REPLAY_PIXELS {
            return Ok(EnqueueResult::Unsupported("replay pixel budget"));
        }
        if self.checkpoint_required()
            || replay_pixels > MAX_JOURNAL_REPLAY_PIXELS - self.journal_pixels
        {
            self.checkpoint_pending = true;
            return Ok(EnqueueResult::FlushRequired);
        }
        if let Some(fill) = fill {
            if self.pending.len() == self.max_batch_draws || !self.bins.fits(fill) {
                debug_assert!(!self.pending.is_empty(), "one fill always fits a batch");
                return Ok(EnqueueResult::FlushRequired);
            }
            if !self.gpu_current {
                if rgb.len() != FB_PIXELS {
                    return self.fail(format!(
                        "REX3 RGB plane has {} pixels, expected {FB_PIXELS}",
                        rgb.len()
                    ));
                }
                let upload_start = self.report.telemetry.as_ref().map(|_| Instant::now());
                tile_major_from_linear(rgb, &mut self.upload_staging);
                let backend = self.backend();
                backend.queue.write_buffer(
                    &backend.rgb,
                    0,
                    bytemuck::cast_slice(&self.upload_staging),
                );
                if let (Some(start), Some(telemetry)) = (upload_start, &mut self.report.telemetry) {
                    telemetry.upload_enqueue_ns += elapsed_ns(start);
                }
                self.report.bytes_uploaded += GPU_PLANE_BYTES;
                self.gpu_current = true;
            }
            fill.damage(&mut self.pending_damage);
            self.bins.push(self.pending.len() as u32, fill);
            self.pending.push(fill);
        }
        // An empty draw with no preceding GPU work leaves the CPU current.
        if fill.is_some() || !self.journal.is_empty() {
            self.journal.push(fill);
            self.journal_pixels += replay_pixels;
        }
        advance_flat_fill_context(context);
        self.report.handled_draws += 1;
        Ok(EnqueueResult::Enqueued)
    }

    pub(crate) fn submit(&mut self, reason: FlushReason) -> Result<(), String> {
        match self.submit_inner(false, reason) {
            Ok(_) => Ok(()),
            Err(error) => self.fail(error),
        }
    }

    /// Synchronize accepted draws into the caller's CPU RGB plane.
    ///
    /// The slice must be the plane last passed to `try_enqueue`, with no CPU
    /// mutation since. On error, call [`Self::recover`] with the same plane.
    pub(crate) fn flush(
        &mut self,
        rgb: &mut [u32],
        reason: FlushReason,
    ) -> Result<DamageSnapshot, String> {
        match self.flush_inner(rgb, reason) {
            Ok(damage) => Ok(damage),
            Err(error) => self.fail(error),
        }
    }

    fn flush_inner(
        &mut self,
        rgb: &mut [u32],
        reason: FlushReason,
    ) -> Result<DamageSnapshot, String> {
        self.check_async_error()?;
        if self.pending.is_empty() && !self.submitted_work {
            debug_assert!(self.journal.is_empty());
            self.checkpoint_pending = false;
            return Ok(DamageSnapshot::default());
        }
        if rgb.len() != FB_PIXELS {
            return Err(format!(
                "REX3 RGB plane has {} pixels, expected {FB_PIXELS}",
                rgb.len()
            ));
        }

        if self.submit_inner(true, reason)?.is_none() {
            return Err("REX3 GPU readback requested without submitted work".to_owned());
        }

        let map_start = self.report.telemetry.as_ref().map(|_| Instant::now());
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        self.backend()
            .readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        let timestamp_receivers: Vec<_> = self
            .timestamp_pending
            .iter()
            .map(|pending| {
                let (sender, receiver) = std::sync::mpsc::sync_channel(1);
                pending
                    .readback
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |result| {
                        let _ = sender.send(result);
                    });
                receiver
            })
            .collect();
        if let (Some(start), Some(telemetry)) = (map_start, &mut self.report.telemetry) {
            telemetry.map_request_ns += elapsed_ns(start);
        }
        let wait_start = self.report.telemetry.as_ref().map(|_| Instant::now());
        let wait_result = self.wait_for_mapping(&receiver, "REX3 GPU readback");
        if let (Some(start), Some(telemetry)) = (wait_start, &mut self.report.telemetry) {
            telemetry.device_wait_ns += elapsed_ns(start);
        }
        wait_result?;
        self.report.fences += 1;

        if !timestamp_receivers.is_empty() {
            if let Err(error) = self.collect_timestamps(timestamp_receivers) {
                // Profiling is optional: report it and keep rendering.
                log::error!("REX3 GPU timestamp profiling unavailable: {error}");
                self.report.timestamp_collection_failed(error.clone());
                self.diagnostics.timestamps = TimestampSupport::Unavailable(error);
            }
            // A device failure seen while waiting still invalidates the batch.
            self.check_async_error()?;
        }

        let copy_start = self.report.telemetry.as_ref().map(|_| Instant::now());
        {
            let mapped = self
                .backend()
                .readback
                .slice(..)
                .get_mapped_range()
                .map_err(|error| format!("access REX3 GPU readback: {error}"))?;
            let words: &[u32] = bytemuck::cast_slice(&mapped);
            // Only fills since the last checkpoint changed the GPU plane, and
            // any CPU mutation revokes GPU authority, so every tile outside
            // the damaged runs already matches the caller's plane.
            for run in &self.readback_runs {
                for tile in run.clone() {
                    copy_tile_to_linear(tile, words, rgb);
                }
            }
        }
        self.backend().readback.unmap();
        self.submitted_work = false;
        self.journal.clear();
        self.journal_pixels = 0;
        self.checkpoint_pending = false;
        if let (Some(start), Some(telemetry)) = (copy_start, &mut self.report.telemetry) {
            telemetry.cpu_readback_copy_ns += elapsed_ns(start);
        }
        self.report.bytes_read_back += self
            .readback_runs
            .iter()
            .map(|run| u64::from(run.end - run.start) * TILE_BYTES)
            .sum::<u64>();
        Ok(std::mem::take(&mut self.submitted_damage))
    }

    fn submit_inner(
        &mut self,
        copy_to_readback: bool,
        reason: FlushReason,
    ) -> Result<Option<wgpu::SubmissionIndex>, String> {
        self.check_async_error()?;
        if self.pending.is_empty() && (!copy_to_readback || !self.submitted_work) {
            return Ok(None);
        }

        let resource_start = self.report.telemetry.as_ref().map(|_| Instant::now());
        let dispatch = !self.pending.is_empty();
        if dispatch {
            // Queue writes for a later submission execute after earlier
            // submissions, so the fixed buffers are safe to reuse in order.
            // The bin offsets bound what the shader reads, so stale fills
            // past this batch in the parameter buffer are never seen.
            let backend = self.backend.as_ref().expect("active GPU backend");
            backend
                .queue
                .write_buffer(&backend.parameters, 0, bytemuck::cast_slice(&self.pending));
            let words = self.bins.encode();
            backend
                .queue
                .write_buffer(&backend.bins, 0, bytemuck::cast_slice(words));
        }
        let timestamps = self.timestamp_resources(dispatch, copy_to_readback);

        if let (Some(start), Some(telemetry)) = (resource_start, &mut self.report.telemetry) {
            telemetry.resource_creation_ns += elapsed_ns(start);
        }
        let encode_start = self.report.telemetry.as_ref().map(|_| Instant::now());
        let backend = self.backend.as_ref().expect("active GPU backend");
        let mut encoder = backend
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("IRIS REX3 GPU batch"),
            });
        if dispatch {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("IRIS REX3 flat-fill batch"),
                timestamp_writes: timestamps.as_ref().and_then(|timestamps| {
                    timestamps
                        .compute_index
                        .map(|index| wgpu::ComputePassTimestampWrites {
                            query_set: &timestamps.queries,
                            beginning_of_pass_write_index: Some(index),
                            end_of_pass_write_index: Some(index + 1),
                        })
                }),
            });
            pass.set_pipeline(&backend.pipeline);
            pass.set_bind_group(0, &backend.bind_group, &[]);
            pass.dispatch_workgroups(TILE_COUNT as u32, 1, 1);
        }
        if copy_to_readback {
            if let Some(timestamps) = &timestamps {
                if let Some(index) = timestamps.transfer_index {
                    encoder.write_timestamp(&timestamps.queries, index);
                }
            }
            // Pending fills are merged into submitted damage only after this
            // submission, so include them here.
            let mut damage = self.submitted_damage.clone();
            damage.merge(&self.pending_damage);
            damaged_tile_runs(&damage, &mut self.readback_rects, &mut self.readback_runs);
            for run in &self.readback_runs {
                let offset = u64::from(run.start) * TILE_BYTES;
                encoder.copy_buffer_to_buffer(
                    &backend.rgb,
                    offset,
                    &backend.readback,
                    offset,
                    u64::from(run.end - run.start) * TILE_BYTES,
                );
            }
            if let Some(timestamps) = &timestamps {
                if let Some(index) = timestamps.transfer_index {
                    encoder.write_timestamp(&timestamps.queries, index + 1);
                }
            }
        }
        if let Some(timestamps) = &timestamps {
            let size = u64::from(timestamps.count) * u64::from(wgpu::QUERY_SIZE);
            encoder.resolve_query_set(
                &timestamps.queries,
                0..timestamps.count,
                &timestamps.resolve,
                0,
            );
            encoder.copy_buffer_to_buffer(&timestamps.resolve, 0, &timestamps.readback, 0, size);
        }
        let commands = encoder.finish();
        if let (Some(start), Some(telemetry)) = (encode_start, &mut self.report.telemetry) {
            telemetry.command_encoding_ns += elapsed_ns(start);
        }
        let submit_start = self.report.telemetry.as_ref().map(|_| Instant::now());
        let submission = backend.queue.submit([commands]);
        if let Some(timestamps) = timestamps {
            self.timestamp_pending.push(PendingTimestamps {
                readback: timestamps.readback,
                bytes: u64::from(timestamps.count) * u64::from(wgpu::QUERY_SIZE),
                compute_index: timestamps.compute_index,
                transfer_index: timestamps.transfer_index,
            });
        }
        if let (Some(start), Some(telemetry)) = (submit_start, &mut self.report.telemetry) {
            telemetry.queue_submit_ns += elapsed_ns(start);
            telemetry.record_batch(self.pending.len(), std::mem::size_of::<FlatFill>());
            telemetry.flush_reasons.record(reason);
        }
        self.report.submissions += 1;
        self.submitted_work = true;
        self.pending.clear();
        self.submitted_damage.merge(&self.pending_damage);
        self.pending_damage = DamageSnapshot::default();

        if !copy_to_readback {
            let poll_start = self.report.telemetry.as_ref().map(|_| Instant::now());
            let poll_result = self.poll();
            if let (Some(start), Some(telemetry)) = (poll_start, &mut self.report.telemetry) {
                telemetry.device_poll_ns += elapsed_ns(start);
            }
            poll_result?;
        }

        Ok(Some(submission))
    }

    fn timestamp_resources(
        &mut self,
        compute: bool,
        copy_to_readback: bool,
    ) -> Option<TimestampResources> {
        let TimestampSupport::Enabled { readback_transfer } = self.diagnostics.timestamps else {
            return None;
        };
        let transfer = copy_to_readback && readback_transfer;
        if !compute && !transfer {
            return None;
        }
        // Keep one slot for the eventual readback without adding a wait to full batches.
        let limit = if copy_to_readback {
            MAX_PENDING_TIMESTAMP_SUBMISSIONS
        } else {
            MAX_PENDING_TIMESTAMP_SUBMISSIONS - 1
        };
        if self.timestamp_pending.len() >= limit {
            if let Some(telemetry) = &mut self.report.telemetry {
                telemetry.timestamp_submissions_skipped += 1;
            }
            return None;
        }

        let compute_index = compute.then_some(0);
        let transfer_index = transfer.then_some(if compute { 2 } else { 0 });
        let count = 2 * (u32::from(compute) + u32::from(transfer));
        let size = u64::from(count) * u64::from(wgpu::QUERY_SIZE);
        let queries = self
            .backend()
            .device
            .create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("IRIS REX3 GPU timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count,
            });
        let resolve = self
            .backend()
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("IRIS REX3 timestamp resolve"),
                size,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
        let readback = self
            .backend()
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("IRIS REX3 timestamp readback"),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        Some(TimestampResources {
            queries,
            resolve,
            readback,
            count,
            compute_index,
            transfer_index,
        })
    }

    fn collect_timestamps(
        &mut self,
        receivers: Vec<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    ) -> Result<(), String> {
        let pending = std::mem::take(&mut self.timestamp_pending);
        if pending.len() != receivers.len() {
            return Err("timestamp readback callback count does not match pending queries".into());
        }
        let start = self.report.telemetry.as_ref().map(|_| Instant::now());
        for (pending, receiver) in pending.into_iter().zip(receivers) {
            self.wait_for_mapping(&receiver, "REX3 GPU timestamp readback")?;
            let (compute_ns, transfer_ns) = {
                let mapped = pending
                    .readback
                    .slice(..)
                    .get_mapped_range()
                    .map_err(|error| format!("access timestamp readback: {error}"))?;
                let compute = pending
                    .compute_index
                    .map(|index| timestamp_duration_ns(&mapped, index, self.timestamp_period))
                    .transpose()?;
                let transfer = pending
                    .transfer_index
                    .map(|index| timestamp_duration_ns(&mapped, index, self.timestamp_period))
                    .transpose()?;
                (compute, transfer)
            };
            pending.readback.unmap();
            if let Some(telemetry) = &mut self.report.telemetry {
                telemetry.timestamp_readback_bytes += pending.bytes;
                if let (Some(duration), Some(total)) = (compute_ns, &mut telemetry.gpu_compute) {
                    record_timestamp(total, duration)?;
                }
                if let (Some(duration), Some(total)) =
                    (transfer_ns, &mut telemetry.gpu_readback_transfer)
                {
                    record_timestamp(total, duration)?;
                }
            }
        }
        if let (Some(start), Some(telemetry)) = (start, &mut self.report.telemetry) {
            telemetry.timestamp_readback_ns += elapsed_ns(start);
        }
        Ok(())
    }

    /// Synchronize accepted draws into `rgb` and leave the engine ready for
    /// the caller to modify that plane. On a GPU failure the journal is
    /// replayed into `rgb`, the engine is disabled and the error returned.
    pub fn drain(&mut self, rgb: &mut [u32]) -> Result<(), String> {
        if rgb.len() != FB_PIXELS {
            return Err(format!(
                "REX3 RGB plane has {} pixels, expected {FB_PIXELS}",
                rgb.len()
            ));
        }
        match self.flush(rgb, FlushReason::Explicit) {
            Ok(_) => {
                self.cpu_framebuffer_changed();
                Ok(())
            }
            Err(error) => {
                if !self.disabled {
                    self.recover(rgb, error.clone());
                }
                Err(error)
            }
        }
    }

    pub(crate) fn cpu_framebuffer_changed(&mut self) {
        debug_assert!(self.pending.is_empty());
        debug_assert!(!self.submitted_work);
        debug_assert!(self.journal.is_empty());
        self.gpu_current = false;
    }

    /// Recover from a GPU failure and disable the engine.
    ///
    /// `rgb` still holds the last checkpoint, so reapplying the journal's
    /// pixel effects brings it up to date; the live context already advanced
    /// when each draw was accepted. The wgpu objects are leaked rather than
    /// destroyed (see [`GpuBackend`]). Returns the damage to publish.
    pub(crate) fn recover(&mut self, rgb: &mut [u32], error: String) -> DamageSnapshot {
        assert_eq!(rgb.len(), FB_PIXELS);
        let original = remember_first_error(&self.async_error, error);
        self.report.error.get_or_insert(original);
        self.disabled = true;
        for fill in self.journal.iter().flatten() {
            fill.replay(rgb);
        }
        self.journal.clear();
        self.journal_pixels = 0;
        self.checkpoint_pending = false;
        self.pending.clear();
        self.bins.clear();
        let mut damage = std::mem::take(&mut self.submitted_damage);
        damage.merge(&std::mem::take(&mut self.pending_damage));
        self.submitted_work = false;
        self.gpu_current = false;
        self.leak_backend();
        damage
    }

    fn leak_backend(&mut self) {
        if let Some(backend) = self.backend.take() {
            std::mem::forget(backend);
            std::mem::forget(std::mem::take(&mut self.timestamp_pending));
        }
    }

    fn check_async_error(&self) -> Result<(), String> {
        if self.disabled {
            return Err(self
                .report
                .error
                .clone()
                .unwrap_or_else(|| "REX3 GPU disabled".into()));
        }
        match self.async_error.lock().clone() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn fail<T>(&mut self, error: String) -> Result<T, String> {
        let original = remember_first_error(&self.async_error, error);
        self.report.error.get_or_insert_with(|| original.clone());
        Err(original)
    }
}

impl Drop for Rex3Gpu {
    fn drop(&mut self) {
        // Destroy a healthy, idle engine normally. Anything else may be
        // waiting on a lost device, which can block or panic inside wgpu.
        let idle = self.pending.is_empty() && !self.submitted_work && self.journal.is_empty();
        if !idle || self.async_error.lock().is_some() {
            self.leak_backend();
        }
    }
}

pub fn probe(backend: &str) -> Result<ProbeResult, String> {
    let backend = backend.to_owned();
    std::thread::Builder::new()
        .name("rex3-gpu-probe".into())
        .stack_size(PROBE_STACK_BYTES)
        .spawn(move || probe_inner(&backend))
        .map_err(|error| format!("start REX3 GPU probe: {error}"))?
        .join()
        .map_err(|_| "REX3 GPU probe thread panicked".to_owned())?
}

fn probe_inner(backend: &str) -> Result<ProbeResult, String> {
    let mut gpu = Rex3Gpu::new(backend)?;
    let initial = 0xAA55_CC33;
    let mut rgb = vec![initial; FB_PIXELS];

    let context = |color: [u32; 3], start: (i32, i32), end: (i32, i32)| {
        let mut context = Rex3Context::power_on_default();
        context.drawmode0 =
            DrawMode0(DRAWMODE0_OPCODE_DRAW | DRAWMODE0_ADRMODE_BLOCK_SH | (1 << 8) | (1 << 9));
        context.drawmode1 = DrawMode1(
            DRAWMODE1_PLANES_RGB
                | (DRAWMODE1_DRAWDEPTH_24 << 3)
                | DRAWMODE1_COMPARE_DISABLE_SH
                | (1 << 15)
                | DRAWMODE1_LOGICOP_SRC_SH,
        );
        context.clipmode = 0xF << CLIPMODE_CIDMATCH_SHIFT;
        context.wrmask = 0x00FF_FFFF;
        context.colorred = color[0] << 11;
        context.colorgrn = color[1] << 11;
        context.colorblue = color[2] << 11;
        context.xstart = (start.0 + REX3_COORD_BIAS) << 11;
        context.ystart = (start.1 + REX3_COORD_BIAS) << 11;
        context.xsave = context.xstart;
        context.xend = (end.0 + REX3_COORD_BIAS) << 11;
        context.yend = (end.1 + REX3_COORD_BIAS) << 11;
        context
    };

    let mut first = context([0x11, 0x22, 0x33], (10, 20), (200, 80));
    let mut second = context([0xA0, 0xB0, 0xC0], (100, 50), (250, 100));
    if gpu.try_enqueue(&mut first, &rgb)? != EnqueueResult::Enqueued
        || gpu.try_enqueue(&mut second, &rgb)? != EnqueueResult::Enqueued
    {
        return Err("the probe's flat fills were rejected by the GPU matcher".into());
    }
    if first.xstart != (10 + REX3_COORD_BIAS) << 11
        || first.ystart != (81 + REX3_COORD_BIAS) << 11
        || second.xstart != (100 + REX3_COORD_BIAS) << 11
        || second.ystart != (101 + REX3_COORD_BIAS) << 11
        || first.pat_bit != 31
        || first.zpat_bit != 31
        || first.mid_primitive
        || second.mid_primitive
    {
        return Err("flat-fill context progression did not match the CPU walker".into());
    }
    let _ = gpu.flush(&mut rgb, FlushReason::Explicit)?;

    let pixel = |x: usize, y: usize| rgb[y * FB_STRIDE as usize + x];
    let with_rgb = |color: u32| (initial & 0xFF00_0000) | color;
    let checks = [
        ("outside", pixel(0, 0), initial),
        ("first fill", pixel(10, 20), with_rgb(0x0033_2211)),
        ("overlap order", pixel(100, 50), with_rgb(0x00C0_B0A0)),
        ("second fill", pixel(250, 100), with_rgb(0x00C0_B0A0)),
        ("outside edge", pixel(251, 100), initial),
    ];
    for (name, got, expected) in checks {
        if got != expected {
            return Err(format!("{name}: got {got:08x}, expected {expected:08x}"));
        }
    }
    if gpu.report.handled_draws != 2
        || gpu.report.fallback_draws != 0
        || gpu.report.submissions != 1
        || gpu.report.fences != 1
        || gpu.report.bytes_uploaded != GPU_PLANE_BYTES
        || gpu.report.bytes_read_back == 0
        || gpu.report.bytes_read_back > GPU_PLANE_BYTES
        || !gpu.pending.is_empty()
    {
        return Err(format!("unexpected probe report: {:?}", gpu.report()));
    }
    if let Some(error) = &gpu.report.timestamp_error {
        return Err(format!("GPU timestamp profiling failed: {error}"));
    }

    Ok(ProbeResult {
        diagnostics: gpu.diagnostics.clone(),
        report: gpu.report(),
    })
}

fn flat_fill(context: &Rex3Context) -> FlatFillMatch {
    let dm0 = context.drawmode0;
    let dm1 = context.drawmode1;
    let adrmode = dm0.adrmode();
    let full_primitive = match adrmode {
        DRAWMODE0_ADRMODE_SPAN => dm0.stoponx(),
        DRAWMODE0_ADRMODE_BLOCK => dm0.stoponx() && dm0.stopony(),
        _ => false,
    };
    if dm0.opcode() != DRAWMODE0_OPCODE_DRAW {
        return FlatFillMatch::Unsupported("opcode");
    }
    if !full_primitive {
        return FlatFillMatch::Unsupported("primitive shape");
    }
    if context.mid_primitive {
        return FlatFillMatch::Unsupported("continuation state");
    }
    if adrmode == DRAWMODE0_ADRMODE_BLOCK && context.xsave != context.xstart {
        return FlatFillMatch::Unsupported("block start state");
    }
    // A batch token raises `host_len` for the GO that follows it. The CPU
    // path then resumes the primitive until the batch cursor stops moving,
    // even for a draw that reads no host data, so it is not a single fill.
    if context.host_len > 1 {
        return FlatFillMatch::Unsupported("host transfer");
    }
    if dm0.colorhost()
        || dm0.alphahost()
        || dm0.skipfirst()
        || dm0.skiplast()
        || dm0.enzpattern()
        || dm0.enlspattern()
        || dm0.length32()
        || dm0.zpopaque()
        || dm0.lsopaque()
        || dm0.shade()
        || dm0.lronly()
        || dm0.xyoffset()
        || dm0.ciclamp()
        || dm0.endptfilter()
        || dm0.ystride()
    {
        return FlatFillMatch::Unsupported("draw mode 0");
    }
    if dm1.planes() != DRAWMODE1_PLANES_RGB
        || dm1.drawdepth() != DRAWMODE1_DRAWDEPTH_24
        || !dm1.rgbmode()
    {
        return FlatFillMatch::Unsupported("pixel format");
    }
    if dm1.dblsrc()
        || dm1.yflip()
        || dm1.compare() != DRAWMODE1_COMPARE_DISABLE
        || dm1.dither()
        || dm1.fastclear()
        || dm1.blend()
        || dm1.logicop() != DRAWMODE1_LOGICOP_SRC
    {
        return FlatFillMatch::Unsupported("pixel operation");
    }
    if context.clipmode & CLIPMODE_ENSMASK_MASK != 0
        || context.clipmode & CLIPMODE_CIDMATCH_MASK != 0xF << CLIPMODE_CIDMATCH_SHIFT
    {
        return FlatFillMatch::Unsupported("clip mode");
    }
    if (context.xstart | context.xend | context.ystart | context.yend) & 0x7FF != 0 {
        return FlatFillMatch::Unsupported("fractional coordinates");
    }
    if context.xend < context.xstart
        || (adrmode == DRAWMODE0_ADRMODE_BLOCK && context.yend < context.ystart)
        || context.bresoctinc1.octant() & (OCTANT_XDEC | OCTANT_YDEC) != 0
    {
        return FlatFillMatch::Unsupported("negative direction");
    }

    let x_window = (context.xywin >> 16) as i16 as i32;
    let y_window = context.xywin as i16 as i32;
    let x0 = (context.xstart >> 11) + x_window - REX3_COORD_BIAS;
    let x1 = (context.xend >> 11) + x_window - REX3_COORD_BIAS;
    let y0 = (context.ystart >> 11) + y_window - REX3_COORD_BIAS;
    let y1 = if adrmode == DRAWMODE0_ADRMODE_BLOCK {
        (context.yend >> 11) + y_window - REX3_COORD_BIAS
    } else {
        y0
    };

    let min_x = x0.max(0);
    let max_x = x1.min(REX3_SCREEN_WIDTH - 1);
    let min_y = y0.max(0);
    let max_y = y1.min(REX3_SCREEN_HEIGHT - 1);
    if min_x > max_x || min_y > max_y || context.wrmask == 0 {
        return FlatFillMatch::Supported(None);
    }

    FlatFillMatch::Supported(Some(FlatFill {
        min_x: min_x as u32,
        min_y: min_y as u32,
        width: (max_x - min_x + 1) as u32,
        height: (max_y - min_y + 1) as u32,
        color: (context.get_colori() & 0x00FF_FFFF)
            | (Rex3Context::clamp_color_component(context.coloralpha) << 24),
        write_mask: context.wrmask,
        stride: FB_STRIDE,
        plane_len: FB_PIXELS as u32,
    }))
}

fn advance_flat_fill_context(context: &mut Rex3Context) {
    if context.drawmode0.adrmode() == DRAWMODE0_ADRMODE_BLOCK {
        context.xstart = context.xsave;
        context.ystart = context.yend + (1 << 11);
    } else {
        context.xstart = context.xend + (1 << 11);
    }
    context.pat_bit = 31;
    context.zpat_bit = 31;
    context.mid_primitive = false;
    // Every CPU GO retires the data port to its resting single-word state.
    context.host_len = 1;
    context.host_cursor = 0;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context(adrmode: u32, start: (i32, i32), end: (i32, i32)) -> Box<Rex3Context> {
        let worker = std::thread::Builder::new()
            .name("rex3-gpu-test-context".into())
            .stack_size(PROBE_STACK_BYTES)
            .spawn(move || {
                let stop = if adrmode == DRAWMODE0_ADRMODE_BLOCK_SH {
                    (1 << 8) | (1 << 9)
                } else {
                    1 << 8
                };
                let mut context = Rex3Context::power_on_default();
                context.drawmode0 = DrawMode0(DRAWMODE0_OPCODE_DRAW | adrmode | stop);
                context.drawmode1 = DrawMode1(
                    DRAWMODE1_PLANES_RGB
                        | (DRAWMODE1_DRAWDEPTH_24 << 3)
                        | DRAWMODE1_COMPARE_DISABLE_SH
                        | (1 << 15)
                        | DRAWMODE1_LOGICOP_SRC_SH,
                );
                context.clipmode = 0xF << CLIPMODE_CIDMATCH_SHIFT;
                context.wrmask = 0x00FF_FFFF;
                context.colorred = 0x11 << 11;
                context.colorgrn = 0x22 << 11;
                context.colorblue = 0x33 << 11;
                context.xstart = (start.0 + REX3_COORD_BIAS) << 11;
                context.ystart = (start.1 + REX3_COORD_BIAS) << 11;
                context.xsave = context.xstart;
                context.xend = (end.0 + REX3_COORD_BIAS) << 11;
                context.yend = (end.1 + REX3_COORD_BIAS) << 11;
                Box::new(context)
            })
            .expect("spawn REX3 GPU test context worker");
        match worker.join() {
            Ok(context) => context,
            Err(panic) => std::panic::resume_unwind(panic),
        }
    }

    fn supported_fill(context: &Rex3Context) -> FlatFill {
        match flat_fill(context) {
            FlatFillMatch::Supported(Some(fill)) => fill,
            FlatFillMatch::Supported(None) => panic!("expected a non-empty fill"),
            FlatFillMatch::Unsupported(reason) => {
                panic!("expected a supported fill, got {reason}")
            }
        }
    }

    fn unsupported_reason(context: &Rex3Context) -> &'static str {
        match flat_fill(context) {
            FlatFillMatch::Unsupported(reason) => reason,
            FlatFillMatch::Supported(_) => panic!("expected an unsupported fill"),
        }
    }

    #[test]
    fn backend_override_is_explicit() {
        assert_eq!(RequestedBackend::parse("auto"), Ok(RequestedBackend::Auto));
        assert_eq!(
            RequestedBackend::parse("vulkan"),
            Ok(RequestedBackend::Vulkan)
        );
        assert_eq!(
            RequestedBackend::parse("metal"),
            Ok(RequestedBackend::Metal)
        );
        assert_eq!(RequestedBackend::parse("dx12"), Ok(RequestedBackend::Dx12));
        assert!(RequestedBackend::parse("gl").is_err());
        assert!(RequestedBackend::parse("Vulkan").is_err());
    }

    #[test]
    fn timestamp_telemetry_marks_unavailable_stages_absent() {
        for support in [
            TimestampSupport::Disabled,
            TimestampSupport::Unsupported("test adapter"),
            TimestampSupport::Unavailable("test device".into()),
        ] {
            let telemetry = Rex3Gpu::new_telemetry(&support);
            assert_eq!(telemetry.gpu_compute, None);
            assert_eq!(telemetry.gpu_readback_transfer, None);
        }
        let compute_only = Rex3Gpu::new_telemetry(&TimestampSupport::Enabled {
            readback_transfer: false,
        });
        assert_eq!(compute_only.gpu_compute, Some(TimestampTotals::default()));
        assert_eq!(compute_only.gpu_readback_transfer, None);
        let both = Rex3Gpu::new_telemetry(&TimestampSupport::Enabled {
            readback_transfer: true,
        });
        assert_eq!(both.gpu_compute, Some(TimestampTotals::default()));
        assert_eq!(both.gpu_readback_transfer, Some(TimestampTotals::default()));
    }

    fn empty_enqueue_attempts(gpu: &mut Rex3Gpu, context: &mut Rex3Context, attempts: u64) {
        let ystart = context.yend;
        for _ in 0..attempts {
            context.ystart = ystart;
            let result = gpu.try_enqueue(context, &[]).expect("enqueue empty draw");
            assert_eq!(result, EnqueueResult::Enqueued);
        }
    }

    #[test]
    #[ignore = "requires a conformant host GPU"]
    fn gpu_empty_enqueue_can_repeat_timed_blocks() {
        let backend = std::env::var("REX3_GPU_BACKEND").unwrap_or_else(|_| "auto".into());
        let mut gpu = Rex3Gpu::new_with_options(&backend, ProfilingOptions::default()).unwrap();
        let mut context = context(DRAWMODE0_ADRMODE_BLOCK_SH, (0, 0), (7, 0));
        context.wrmask = 0;
        for enabled in [false, true, true, false] {
            gpu.set_telemetry_enabled(enabled).unwrap();
            empty_enqueue_attempts(&mut gpu, &mut context, 2);
        }
        assert_eq!(gpu.report().handled_draws, 8);
        assert_eq!(gpu.report().submissions, 0);
    }

    #[test]
    #[ignore = "hardware-dependent isolated matcher/enqueue telemetry measurement"]
    fn gpu_empty_enqueue_telemetry_cost() {
        // Long enough to reduce scheduling-quantum noise on typical hosts.
        const ATTEMPTS: u64 = 256 * 32768;
        const BLOCKS: usize = 16;

        fn sample(gpu: &mut Rex3Gpu, context: &mut Rex3Context, enabled: bool) -> (f64, Instant) {
            gpu.set_telemetry_enabled(enabled)
                .expect("toggle idle telemetry");
            let before = gpu.report();
            let start = Instant::now();
            empty_enqueue_attempts(gpu, context, ATTEMPTS);
            let elapsed = start.elapsed();
            let delta = gpu
                .report()
                .delta_since(&before)
                .expect("enqueue telemetry delta");
            assert_eq!(delta.handled_draws, ATTEMPTS);
            assert_eq!(delta.fallback_draws, 0);
            assert_eq!(delta.submissions, 0);
            assert_eq!(delta.pending_draws, 0);
            assert_eq!(delta.telemetry.is_some(), enabled);
            if let Some(telemetry) = delta.telemetry {
                assert_eq!(telemetry.match_attempts, ATTEMPTS);
                assert_eq!(
                    telemetry.sampled_draws,
                    ATTEMPTS / DRAW_TIMING_SAMPLE_INTERVAL
                );
            }
            (ATTEMPTS as f64 / elapsed.as_secs_f64(), start + elapsed / 2)
        }

        let backend = std::env::var("REX3_GPU_BACKEND").unwrap_or_else(|_| "auto".into());
        let mut gpu = Rex3Gpu::new_with_options(&backend, ProfilingOptions::default())
            .expect("initialize GPU for isolated enqueue measurement");
        let mut context = context(DRAWMODE0_ADRMODE_BLOCK_SH, (0, 0), (7, 0));
        context.wrmask = 0;
        for enabled in [false, true, true, false] {
            sample(&mut gpu, &mut context, enabled);
        }
        let mut controls = Vec::with_capacity(BLOCKS);
        let mut effects = Vec::with_capacity(BLOCKS);
        let mut orders = Vec::with_capacity(BLOCKS);
        for block in 0..BLOCKS {
            let aa = [
                sample(&mut gpu, &mut context, false),
                sample(&mut gpu, &mut context, false),
            ]
            .map(|sample| sample.0);
            let (a, b) = if block % 2 == 0 {
                (aa[0], aa[1])
            } else {
                (aa[1], aa[0])
            };
            controls.push(log_rate_ratio(a, b));
            let on_first = block % 2 != 0;
            let samples = [on_first, !on_first, !on_first, on_first]
                .map(|enabled| sample(&mut gpu, &mut context, enabled));
            let start = samples[0].1;
            let rates = samples.map(|sample| sample.0);
            let times = samples.map(|sample| sample.1.duration_since(start).as_secs_f64());
            effects.push(abba_log_rate_ratio(rates, times, on_first));
            orders.push(on_first);
            eprintln!(
                "enqueue block {}: A/A={aa:.0?} on_first={on_first} ABBA={rates:.0?}",
                block + 1
            );
        }
        let control = paired_rate_stats(&controls);
        let effect = paired_rate_stats(&effects);
        eprintln!(
            "isolated empty-fill enqueue: A/A={control:?} telemetry={effect:?} order_shift={:.3}%; \
             signed throughput ratios, no GPU submissions or GFIFO",
            paired_order_shift(&effects, &orders) * 100.0,
        );
    }

    #[test]
    #[ignore = "requires a conformant host GPU with optional timestamps"]
    fn gpu_timestamp_readback_is_bounded() {
        let backend = std::env::var("REX3_GPU_BACKEND").unwrap_or_else(|_| "auto".into());
        let mut gpu = Rex3Gpu::new_with_options(
            &backend,
            ProfilingOptions {
                telemetry: true,
                timestamps: true,
            },
        )
        .expect("initialize GPU timestamp test");
        eprintln!("REX3 timestamp test {}", gpu.diagnostics());
        let transfer = match &gpu.diagnostics().timestamps {
            TimestampSupport::Enabled { readback_transfer } => *readback_transfer,
            TimestampSupport::Unsupported(_) | TimestampSupport::Unavailable(_) => return,
            TimestampSupport::Disabled => panic!("requested timestamps were reported disabled"),
        };
        let before = gpu.report();
        let mut context = context(DRAWMODE0_ADRMODE_BLOCK_SH, (0, 0), (0, 0));
        let color = context.get_colori() & 0x00FF_FFFF;
        let mut rgb = vec![0; FB_PIXELS];
        let draws = MAX_PENDING_TIMESTAMP_SUBMISSIONS + 1;
        for _ in 0..draws {
            context.ystart = context.yend;
            assert_eq!(
                gpu.try_enqueue(&mut context, &rgb).expect("enqueue draw"),
                EnqueueResult::Enqueued
            );
            gpu.submit(FlushReason::BatchFull).expect("submit draw");
        }
        gpu.flush(&mut rgb, FlushReason::Explicit)
            .expect("read back framebuffer and timestamps");
        let delta = gpu.report().delta_since(&before).expect("timestamp delta");
        assert_eq!(delta.error, None);
        assert_eq!(delta.timestamp_error, None);
        assert_eq!(delta.handled_draws, draws as u64);
        assert_eq!(delta.fallback_draws, 0);
        assert_eq!(delta.submissions, draws as u64 + 1);
        assert_eq!(delta.fences, 1);
        assert_eq!(rgb[0], color);
        assert!(rgb[1..].iter().all(|pixel| *pixel == 0));

        let telemetry = delta.telemetry.expect("timestamp telemetry");
        assert_eq!(telemetry.batch_count, draws as u64);
        assert_eq!(telemetry.batch_draws, draws as u64);
        assert_eq!(telemetry.flush_reasons.batch_full, draws as u64);
        assert_eq!(telemetry.flush_reasons.explicit, 1);
        assert_eq!(telemetry.timestamp_submissions_skipped, 2);
        let compute = telemetry.gpu_compute.expect("compute timestamps");
        assert_eq!(
            compute.count + compute.discarded,
            (MAX_PENDING_TIMESTAMP_SUBMISSIONS - 1) as u64
        );
        require_usable_timestamps("compute", &compute).unwrap_or_else(|error| panic!("{error}"));
        if transfer {
            let transfer = telemetry
                .gpu_readback_transfer
                .expect("transfer timestamps");
            assert_eq!(transfer.count + transfer.discarded, 1);
            require_usable_timestamps("readback transfer", &transfer)
                .unwrap_or_else(|error| panic!("{error}"));
        } else {
            assert_eq!(telemetry.gpu_readback_transfer, None);
        }
        assert_eq!(
            telemetry.timestamp_readback_bytes,
            (MAX_PENDING_TIMESTAMP_SUBMISSIONS as u64 - 1) * 16 + if transfer { 16 } else { 0 }
        );
    }

    struct TestRng(u64);

    impl TestRng {
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 16) as u32
        }

        fn below(&mut self, bound: u32) -> u32 {
            self.next() % bound
        }
    }

    const TEST_MASKS: [u32; 6] = [
        0x00ff_ffff,
        0xffff_ffff,
        0xff00_0000,
        0x00ff_00ff,
        0xaa00_ff00,
        0x0000_0001,
    ];

    fn seeded_plane() -> Vec<u32> {
        (0..FB_PIXELS as u32)
            .map(|index| index.wrapping_mul(0x9e37_79b9) ^ 0x5a5a_0000)
            .collect()
    }

    fn first_difference(actual: &[u32], expected: &[u32]) -> Option<String> {
        (actual != expected).then(|| {
            let index = actual
                .iter()
                .zip(expected)
                .position(|(got, want)| got != want)
                .unwrap();
            format!(
                "pixel ({}, {}): {:08x} != {:08x}",
                index % FB_STRIDE as usize,
                index / FB_STRIDE as usize,
                actual[index],
                expected[index]
            )
        })
    }

    /// The WGSL kernel, statement for statement, on a tile-major plane.
    fn run_tiled_kernel(rgb: &mut [u32], fills: &[FlatFill], bins: &[u32]) {
        for tile in 0..TILE_COUNT as u32 {
            let (begin, end) = (bins[tile as usize], bins[tile as usize + 1]);
            if begin == end {
                continue;
            }
            for lane in 0..256 {
                let x = (tile % TILES_X) * TILE_SIZE + lane % TILE_SIZE;
                let y0 = (tile / TILES_X) * TILE_SIZE + lane / TILE_SIZE;
                for row in 0..4 {
                    let y = y0 + row * 8;
                    let address = (tile * TILE_PIXELS as u32 + lane + row * 256) as usize;
                    let mut pixel = rgb[address];
                    for &draw in &bins[begin as usize..end as usize] {
                        let fill = fills[draw as usize];
                        if x.wrapping_sub(fill.min_x) < fill.width
                            && y.wrapping_sub(fill.min_y) < fill.height
                        {
                            pixel = (pixel & !fill.write_mask) | (fill.color & fill.write_mask);
                        }
                    }
                    rgb[address] = pixel;
                }
            }
        }
    }

    fn random_clipped_fill(rng: &mut TestRng) -> FlatFill {
        let width = REX3_SCREEN_WIDTH as u32;
        let height = REX3_SCREEN_HEIGHT as u32;
        let (min_x, min_y) = (rng.below(width), rng.below(height));
        let (span_x, span_y) = match rng.below(4) {
            0 => (8, 8),
            1 => (96, 96),
            2 => (width, 3),
            _ => (width, height),
        };
        FlatFill {
            min_x,
            min_y,
            width: 1 + rng.below(span_x.min(width - min_x)),
            height: 1 + rng.below(span_y.min(height - min_y)),
            color: rng.next(),
            write_mask: TEST_MASKS[rng.below(TEST_MASKS.len() as u32) as usize],
            stride: FB_STRIDE,
            plane_len: FB_PIXELS as u32,
        }
    }

    #[test]
    fn tile_bins_preserve_order_and_cover_exactly_each_fill() {
        let mut rng = TestRng(0x0123_4567_89ab_cdef);
        let mut bins = TileBins::new();
        let mut expected = seeded_plane();
        let mut tiled = Vec::new();
        tile_major_from_linear(&expected, &mut tiled);
        assert_eq!(tiled.len(), GPU_PLANE_WORDS);
        for _ in 0..6 {
            let fills: Vec<_> = (0..48).map(|_| random_clipped_fill(&mut rng)).collect();
            for (draw, fill) in fills.iter().enumerate() {
                assert!(bins.fits(*fill));
                bins.push(draw as u32, *fill);
                fill.replay(&mut expected);
            }
            let encoded = bins.encode().to_vec();
            assert_eq!(encoded[0] as usize, TILE_COUNT + 1);
            assert_eq!(encoded[TILE_COUNT] as usize, encoded.len());
            for tile in 0..TILE_COUNT {
                let list = &encoded[encoded[tile] as usize..encoded[tile + 1] as usize];
                assert!(list.windows(2).all(|pair| pair[0] < pair[1]), "tile {tile}");
            }
            run_tiled_kernel(&mut tiled, &fills, &encoded);
            // The off-screen columns never reach the GPU plane; they come
            // from the original CPU plane, as in the real readback.
            let mut actual = seeded_plane();
            for tile in 0..TILE_COUNT as u32 {
                copy_tile_to_linear(tile, &tiled, &mut actual);
            }
            if let Some(difference) = first_difference(&actual, &expected) {
                panic!("tiled kernel diverged from CPU replay at {difference}");
            }
            // Later batches start from the replayed plane.
            tile_major_from_linear(&expected, &mut tiled);
        }
        assert_eq!(bins.entries, 0);

        // Edge tiles, and the per-batch reference budget.
        let full = FlatFill {
            min_x: 0,
            min_y: 0,
            width: REX3_SCREEN_WIDTH as u32,
            height: REX3_SCREEN_HEIGHT as u32,
            color: 0,
            write_mask: !0,
            stride: FB_STRIDE,
            plane_len: FB_PIXELS as u32,
        };
        assert_eq!(full.tile_count(), TILE_COUNT);
        let corner = FlatFill {
            min_x: REX3_SCREEN_WIDTH as u32 - 1,
            min_y: REX3_SCREEN_HEIGHT as u32 - 1,
            width: 1,
            height: 1,
            ..full
        };
        assert_eq!(corner.tiles(), (TILES_X - 1..TILES_X, TILES_Y - 1..TILES_Y));
        let mut pushed = 0;
        while bins.fits(full) {
            bins.push(pushed, full);
            pushed += 1;
        }
        assert_eq!(pushed as usize, MAX_BIN_ENTRIES / TILE_COUNT);
        bins.clear();
        assert!(bins.encode()[..=TILE_COUNT]
            .iter()
            .all(|&o| o as usize == TILE_COUNT + 1));
    }

    #[test]
    fn damaged_tile_runs_cover_damage_in_tile_major_order() {
        let mut rectangles = Vec::new();
        let mut runs = Vec::new();
        let mut damage = DamageSnapshot::default();
        damaged_tile_runs(&damage, &mut rectangles, &mut runs);
        assert!(runs.is_empty());

        // A full plane, including off-screen columns, is one copy of the
        // screen tiles.
        damage.mark_full(DamagePlane::Rgb);
        damaged_tile_runs(&damage, &mut rectangles, &mut runs);
        assert_eq!(runs, vec![0..TILE_COUNT as u32]);

        // A 512-pixel column band is one run per tile row.
        let mut damage = DamageSnapshot::default();
        damage.mark_rect(
            DamagePlane::Rgb,
            DamageRect {
                x: 64,
                y: 0,
                width: 512,
                height: 1024,
            },
        );
        damaged_tile_runs(&damage, &mut rectangles, &mut runs);
        assert_eq!(runs.len(), TILES_Y as usize);
        for (row, run) in runs.iter().enumerate() {
            let first = row as u32 * TILES_X;
            assert_eq!(*run, first + 2..first + 18);
        }

        // Every damaged pixel lies in some run; adjacent full rows merge.
        let mut rng = TestRng(0x5eed_1234_abcd_0001);
        for _ in 0..200 {
            let mut damage = DamageSnapshot::default();
            let mut pixels = Vec::new();
            for _ in 0..1 + rng.below(6) {
                let fill = random_clipped_fill(&mut rng);
                fill.damage(&mut damage);
                pixels.push((fill.min_x, fill.min_y));
                pixels.push((fill.min_x + fill.width - 1, fill.min_y + fill.height - 1));
            }
            damaged_tile_runs(&damage, &mut rectangles, &mut runs);
            assert!(runs.windows(2).all(|pair| pair[0].end < pair[1].start));
            for (x, y) in pixels {
                let tile = (y / TILE_SIZE) * TILES_X + x / TILE_SIZE;
                assert!(runs.iter().any(|run| run.contains(&tile)), "({x}, {y})");
            }
        }
    }

    /// Processor-thread cost per tiny fill, without the GFIFO: the matcher
    /// alone, then the whole accept/submit/checkpoint path the live engine
    /// runs for each GO. Run in release.
    #[test]
    #[ignore = "hardware-dependent host cost measurement; run in release"]
    fn gpu_tiny_fill_host_cost() {
        const DRAWS: u32 = 4 * 1024 * 1024;
        let backend = std::env::var("REX3_GPU_BACKEND").unwrap_or_else(|_| "auto".into());
        let mut gpu = Rex3Gpu::new_with_options(&backend, ProfilingOptions::default())
            .expect("initialize GPU for host cost measurement");
        let mut context = context(DRAWMODE0_ADRMODE_BLOCK_SH, (0, 0), (7, 0));
        let mut rgb = seeded_plane();
        let place = |context: &mut Rex3Context, draw: u32| {
            let y = (draw % 1024) as i32;
            let x = ((draw / 1024 * 8) % 1272) as i32;
            context.xstart = (x + REX3_COORD_BIAS) << 11;
            context.xsave = context.xstart;
            context.xend = (x + 7 + REX3_COORD_BIAS) << 11;
            context.ystart = (y + REX3_COORD_BIAS) << 11;
            context.yend = context.ystart;
            context.colorred = (draw & 0xff) << 11;
        };

        let start = Instant::now();
        let mut matched = 0u64;
        for draw in 0..DRAWS {
            place(&mut context, draw);
            if let FlatFillMatch::Supported(Some(fill)) = flat_fill(std::hint::black_box(&context))
            {
                matched += u64::from(std::hint::black_box(fill).width);
            }
        }
        let place_and_match = start.elapsed();
        assert_eq!(matched, u64::from(DRAWS) * 8);

        let mut flush_time = Duration::ZERO;
        let mut submit_time = Duration::ZERO;
        let start = Instant::now();
        for draw in 0..DRAWS {
            place(&mut context, draw);
            loop {
                match gpu.try_enqueue(&mut context, &rgb).expect("enqueue") {
                    EnqueueResult::Enqueued => break,
                    EnqueueResult::FlushRequired if gpu.checkpoint_required() => {
                        let flush = Instant::now();
                        gpu.flush(&mut rgb, FlushReason::Explicit).expect("flush");
                        flush_time += flush.elapsed();
                    }
                    EnqueueResult::FlushRequired => {
                        let submit = Instant::now();
                        gpu.submit(FlushReason::BatchFull).expect("submit");
                        submit_time += submit.elapsed();
                    }
                    EnqueueResult::Unsupported(reason) => panic!("{reason}"),
                }
            }
        }
        gpu.drain(&mut rgb).expect("drain");
        let total = start.elapsed();

        let lock = Mutex::new(0u64);
        let start = Instant::now();
        for _ in 0..DRAWS {
            *std::hint::black_box(&lock).lock() += 1;
        }
        let lock_pair = start.elapsed();

        let per = |time: Duration| time.as_nanos() as f64 / f64::from(DRAWS);
        let accept = total - flush_time - submit_time;
        eprintln!(
            "tiny fill host cost per draw: place+match={:.1}ns total={:.1}ns \
             accept(incl. place+match)={:.1}ns submit={:.1}ns checkpoint={:.1}ns \
             uncontended lock pair={:.1}ns; {}",
            per(place_and_match),
            per(total),
            per(accept),
            per(submit_time),
            per(flush_time),
            per(lock_pair),
            gpu.diagnostics(),
        );
    }

    #[test]
    #[ignore = "requires a conformant host GPU"]
    fn gpu_tiled_fills_match_cpu_replay_at_every_checkpoint() {
        let backend = std::env::var("REX3_GPU_BACKEND").unwrap_or_else(|_| "auto".into());
        let mut gpu = Rex3Gpu::new_with_options(&backend, ProfilingOptions::default())
            .expect("initialize GPU for tiled fill validation");
        eprintln!("REX3 tiled fill test {}", gpu.diagnostics());
        // Odd limits so batch, journal and pixel-budget boundaries all move.
        gpu.set_test_limits(301, 4099);
        let mut rgb = seeded_plane();
        let mut expected = rgb.clone();
        let mut block = context(DRAWMODE0_ADRMODE_BLOCK_SH, (0, 0), (0, 0));
        let mut span = context(DRAWMODE0_ADRMODE_SPAN_SH, (0, 0), (0, 0));
        let mut rng = TestRng(0xfeed_beef_0bad_f00d);
        let mut checkpoints = 0;
        for draw in 0..8000 {
            let is_span = rng.below(5) == 0;
            let (width, height) = match rng.below(8) {
                0..=3 => (1 + rng.below(8), 1 + rng.below(8)),
                4 | 5 => (1 + rng.below(96), 1 + rng.below(96)),
                6 => (1 + rng.below(1400), 1 + rng.below(4)),
                _ => (1 + rng.below(700), 1 + rng.below(700)),
            };
            let x0 = rng.below(REX3_SCREEN_WIDTH as u32 + 64) as i32 - 48;
            let y0 = rng.below(REX3_SCREEN_HEIGHT as u32 + 64) as i32 - 48;
            let ctx = if is_span { &mut span } else { &mut block };
            ctx.xstart = (x0 + REX3_COORD_BIAS) << 11;
            ctx.xsave = ctx.xstart;
            ctx.ystart = (y0 + REX3_COORD_BIAS) << 11;
            ctx.xend = (x0 + width as i32 - 1 + REX3_COORD_BIAS) << 11;
            ctx.yend = (y0 + height as i32 - 1 + REX3_COORD_BIAS) << 11;
            ctx.colorred = rng.below(256) << 11;
            ctx.colorgrn = rng.below(256) << 11;
            ctx.colorblue = rng.below(256) << 11;
            ctx.coloralpha = rng.below(0x200) << 11;
            ctx.wrmask = match rng.below(8) {
                0 => 0,
                selector => TEST_MASKS[(selector - 1) as usize % TEST_MASKS.len()],
            };
            let fill = match flat_fill(ctx) {
                FlatFillMatch::Supported(fill) => fill,
                FlatFillMatch::Unsupported(reason) => panic!("draw {draw} unsupported: {reason}"),
            };
            loop {
                match gpu.try_enqueue(ctx, &rgb).expect("enqueue tiled fill") {
                    EnqueueResult::Enqueued => break,
                    EnqueueResult::FlushRequired if gpu.checkpoint_required() => {
                        gpu.flush(&mut rgb, FlushReason::Explicit)
                            .expect("checkpoint tiled fills");
                        checkpoints += 1;
                        if let Some(difference) = first_difference(&rgb, &expected) {
                            panic!("checkpoint {checkpoints} before draw {draw}: {difference}");
                        }
                    }
                    EnqueueResult::FlushRequired => {
                        gpu.submit(FlushReason::BatchFull)
                            .expect("submit tiled batch");
                    }
                    EnqueueResult::Unsupported(reason) => panic!("draw {draw}: {reason}"),
                }
            }
            if let Some(fill) = fill {
                fill.replay(&mut expected);
            }
        }
        gpu.drain(&mut rgb).expect("drain tiled fills");
        if let Some(difference) = first_difference(&rgb, &expected) {
            panic!("final framebuffer: {difference}");
        }
        let report = gpu.report();
        eprintln!("tiled fill report: checkpoints={checkpoints} {report:?}");
        assert_eq!(report.error, None);
        assert_eq!(report.handled_draws, 8000);
        assert!(checkpoints > 10 && report.submissions > report.fences);
    }

    #[test]
    #[ignore = "requires a conformant host GPU"]
    fn gpu_failure_replays_the_journal_onto_the_cpu_plane() {
        let backend = std::env::var("REX3_GPU_BACKEND").unwrap_or_else(|_| "auto".into());
        let mut gpu = Rex3Gpu::new_with_options(&backend, ProfilingOptions::default())
            .expect("initialize GPU for recovery validation");
        // Small batches so the failure finds submitted and pending work.
        gpu.set_test_limits(64, 4099);
        let mut rgb = seeded_plane();
        let mut expected = rgb.clone();
        let mut ctx = context(DRAWMODE0_ADRMODE_BLOCK_SH, (0, 0), (0, 0));
        let mut rng = TestRng(0x0dd_c0ffee_1234);
        let mut checkpoints = 0;
        let mut accepted = 0u64;
        for draw in 0..1500 {
            let x0 = rng.below(REX3_SCREEN_WIDTH as u32) as i32;
            let y0 = rng.below(REX3_SCREEN_HEIGHT as u32) as i32;
            let (width, height) = (1 + rng.below(200), 1 + rng.below(200));
            ctx.xstart = (x0 + REX3_COORD_BIAS) << 11;
            ctx.xsave = ctx.xstart;
            ctx.ystart = (y0 + REX3_COORD_BIAS) << 11;
            ctx.xend = (x0 + width as i32 - 1 + REX3_COORD_BIAS) << 11;
            ctx.yend = (y0 + height as i32 - 1 + REX3_COORD_BIAS) << 11;
            ctx.colorred = rng.below(256) << 11;
            ctx.colorgrn = rng.below(256) << 11;
            ctx.colorblue = rng.below(256) << 11;
            ctx.wrmask = TEST_MASKS[rng.below(TEST_MASKS.len() as u32) as usize];
            let fill = match flat_fill(&ctx) {
                FlatFillMatch::Supported(fill) => fill,
                FlatFillMatch::Unsupported(reason) => panic!("draw {draw} unsupported: {reason}"),
            };
            loop {
                match gpu
                    .try_enqueue(&mut ctx, &rgb)
                    .expect("enqueue before failure")
                {
                    EnqueueResult::Enqueued => break,
                    EnqueueResult::FlushRequired if gpu.checkpoint_required() => {
                        gpu.flush(&mut rgb, FlushReason::Explicit)
                            .expect("checkpoint");
                        checkpoints += 1;
                    }
                    EnqueueResult::FlushRequired => {
                        gpu.submit(FlushReason::BatchFull).expect("submit");
                    }
                    EnqueueResult::Unsupported(reason) => panic!("draw {draw}: {reason}"),
                }
            }
            accepted += 1;
            if let Some(fill) = fill {
                fill.replay(&mut expected);
            }
        }
        assert!(
            checkpoints > 0,
            "the test must cross a checkpoint before failing"
        );
        assert!(gpu.submitted_work && !gpu.pending.is_empty() && !gpu.journal.is_empty());

        gpu.inject_error_for_test("injected device loss");
        let error = gpu
            .flush(&mut rgb, FlushReason::Explicit)
            .expect_err("a failed GPU must not report a checkpoint");
        assert_eq!(error, "injected device loss");
        let damage = gpu.recover(&mut rgb, error);
        assert!(
            !damage.is_empty(),
            "recovery must publish the replayed region"
        );
        if let Some(difference) = first_difference(&rgb, &expected) {
            panic!("recovered framebuffer: {difference}");
        }
        assert!(gpu.is_disabled() && gpu.backend.is_none());
        assert!(gpu.journal.is_empty() && gpu.pending.is_empty());
        let report = gpu.report();
        assert_eq!(report.handled_draws, accepted);
        assert_eq!(report.error.as_deref(), Some("injected device loss"));
        let mut next = context(DRAWMODE0_ADRMODE_BLOCK_SH, (1, 1), (2, 2));
        let before = next.xstart;
        assert!(gpu.try_enqueue(&mut next, &rgb).is_err());
        assert_eq!(
            next.xstart, before,
            "a refused draw must not advance the context"
        );
        drop(gpu);
    }

    #[test]
    fn block_match_clips_and_preserves_masked_rgb() {
        let mut context = context(DRAWMODE0_ADRMODE_BLOCK_SH, (-10, -5), (10, 5));
        context.wrmask = 0xAA00_00FF;
        context.coloralpha = 0x7A << 11;
        let fill = supported_fill(&context);

        assert_eq!(fill.min_x, 0);
        assert_eq!(fill.min_y, 0);
        assert_eq!(fill.width, 11);
        assert_eq!(fill.height, 6);
        assert_eq!(fill.color, 0x7A33_2211);
        assert_eq!(fill.write_mask, 0xAA00_00FF);
        assert_eq!(fill.stride, FB_STRIDE);
        assert_eq!(fill.plane_len, FB_PIXELS as u32);
        context.coloralpha = 0x140 << 11;
        assert_eq!(supported_fill(&context).color, 0xFF33_2211);
        context.coloralpha = 0x180 << 11;
        assert_eq!(supported_fill(&context).color, 0x0033_2211);
    }

    #[test]
    fn span_match_is_one_row() {
        let context = context(DRAWMODE0_ADRMODE_SPAN_SH, (7, 9), (17, 200));
        let fill = supported_fill(&context);

        assert_eq!((fill.min_x, fill.min_y), (7, 9));
        assert_eq!((fill.width, fill.height), (11, 1));
    }

    #[test]
    fn empty_supported_fills_and_high_masks_are_distinct() {
        let mut masked = context(DRAWMODE0_ADRMODE_BLOCK_SH, (1, 2), (3, 4));
        masked.wrmask = 0xFF00_0000;
        assert_eq!(supported_fill(&masked).write_mask, 0xFF00_0000);
        masked.wrmask = 0;
        assert!(matches!(flat_fill(&masked), FlatFillMatch::Supported(None)));

        let offscreen = context(
            DRAWMODE0_ADRMODE_BLOCK_SH,
            (REX3_SCREEN_WIDTH + 1, 10),
            (REX3_SCREEN_WIDTH + 20, 20),
        );
        assert!(matches!(
            flat_fill(&offscreen),
            FlatFillMatch::Supported(None)
        ));
    }

    #[test]
    fn unsupported_modes_report_stable_reasons() {
        let mut ctx = context(DRAWMODE0_ADRMODE_BLOCK_SH, (1, 2), (3, 4));

        ctx.drawmode0 = DrawMode0(ctx.drawmode0.0 & !0x3);
        assert_eq!(unsupported_reason(&ctx), "opcode");

        ctx = context(DRAWMODE0_ADRMODE_BLOCK_SH, (1, 2), (3, 4));
        ctx.drawmode0 = DrawMode0(ctx.drawmode0.0 & !(1 << 9));
        assert_eq!(unsupported_reason(&ctx), "primitive shape");

        ctx = context(DRAWMODE0_ADRMODE_BLOCK_SH, (1, 2), (3, 4));
        ctx.mid_primitive = true;
        assert_eq!(unsupported_reason(&ctx), "continuation state");

        ctx = context(DRAWMODE0_ADRMODE_BLOCK_SH, (1, 2), (3, 4));
        ctx.xsave += 1 << 11;
        assert_eq!(unsupported_reason(&ctx), "block start state");

        ctx = context(DRAWMODE0_ADRMODE_BLOCK_SH, (1, 2), (3, 4));
        ctx.host_len = 2;
        assert_eq!(unsupported_reason(&ctx), "host transfer");
        ctx.host_len = 0;
        supported_fill(&ctx);

        ctx = context(DRAWMODE0_ADRMODE_BLOCK_SH, (1, 2), (3, 4));
        ctx.drawmode0 = DrawMode0(ctx.drawmode0.0 | (1 << 18));
        assert_eq!(unsupported_reason(&ctx), "draw mode 0");

        ctx = context(DRAWMODE0_ADRMODE_BLOCK_SH, (1, 2), (3, 4));
        ctx.drawmode1 = DrawMode1((ctx.drawmode1.0 & !0x7) | 4);
        assert_eq!(unsupported_reason(&ctx), "pixel format");

        ctx = context(DRAWMODE0_ADRMODE_BLOCK_SH, (1, 2), (3, 4));
        ctx.drawmode1 = DrawMode1(ctx.drawmode1.0 | (1 << 18));
        assert_eq!(unsupported_reason(&ctx), "pixel operation");

        ctx = context(DRAWMODE0_ADRMODE_BLOCK_SH, (1, 2), (3, 4));
        ctx.clipmode = 0;
        assert_eq!(unsupported_reason(&ctx), "clip mode");

        ctx = context(DRAWMODE0_ADRMODE_BLOCK_SH, (1, 2), (3, 4));
        ctx.xstart |= 1;
        ctx.xsave = ctx.xstart;
        assert_eq!(unsupported_reason(&ctx), "fractional coordinates");

        ctx = context(DRAWMODE0_ADRMODE_BLOCK_SH, (3, 2), (1, 4));
        assert_eq!(unsupported_reason(&ctx), "negative direction");
    }

    #[test]
    fn supported_draws_advance_like_complete_cpu_primitives() {
        let mut block = context(DRAWMODE0_ADRMODE_BLOCK_SH, (10, 20), (30, 40));
        block.pat_bit = 7;
        block.zpat_bit = 9;
        block.mid_primitive = true;
        block.host_len = 0;
        block.host_cursor = 1;
        advance_flat_fill_context(&mut block);
        assert_eq!((block.host_len, block.host_cursor), (1, 0));
        assert_eq!(block.xstart, (10 + REX3_COORD_BIAS) << 11);
        assert_eq!(block.ystart, (41 + REX3_COORD_BIAS) << 11);
        assert_eq!((block.pat_bit, block.zpat_bit), (31, 31));
        assert!(!block.mid_primitive);

        let mut span = context(DRAWMODE0_ADRMODE_SPAN_SH, (10, 20), (30, 40));
        advance_flat_fill_context(&mut span);
        assert_eq!(span.xstart, (31 + REX3_COORD_BIAS) << 11);
        assert_eq!(span.ystart, (20 + REX3_COORD_BIAS) << 11);
    }
}
