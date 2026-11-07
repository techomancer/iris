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
const FB_BYTES: u64 = (FB_PIXELS * std::mem::size_of::<u32>()) as u64;
const DEFAULT_MAX_BATCH_DRAWS: usize = 1024;
const MAX_CONFIGURED_BATCH_DRAWS: usize = 65536;
const MAP_TIMEOUT: Duration = Duration::from_secs(30);
const PROBE_STACK_BYTES: usize = 8 * 1024 * 1024;
const DRAW_TIMING_SAMPLE_INTERVAL: u64 = 256;
const MAX_PENDING_TIMESTAMP_SUBMISSIONS: usize = 64;

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
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    rgb: wgpu::Buffer,
    readback: wgpu::Buffer,
    pending: Vec<FlatFill>,
    pending_damage: DamageSnapshot,
    submitted_damage: DamageSnapshot,
    gpu_current: bool,
    submitted_work: bool,
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
        if limits.max_storage_buffer_binding_size < FB_BYTES {
            return Err(format!(
                "{} only exposes a {}-byte storage binding; REX3 RGB needs {FB_BYTES} bytes",
                info.name, limits.max_storage_buffer_binding_size
            ));
        }

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
                if limits.max_storage_buffer_binding_size < FB_BYTES {
                    return Err(format!(
                        "{} only exposes a {}-byte storage binding; REX3 RGB needs {FB_BYTES} bytes",
                        info.name, limits.max_storage_buffer_binding_size
                    ));
                }
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
            *uncaptured.lock() = Some(format!("uncaptured wgpu error: {error}"));
        }));
        let lost = Arc::clone(&async_error);
        device.set_device_lost_callback(move |reason, message| {
            *lost.lock() = Some(format!("wgpu device lost ({reason:?}): {message}"));
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
        device
            .poll(wgpu::PollType::Poll)
            .map_err(|error| format!("poll after pipeline creation: {error}"))?;
        if let Some(error) = pollster::block_on(error_scope.pop()) {
            return Err(format!("create flat-fill pipeline: {error}"));
        }
        let bind_group_layout = pipeline.get_bind_group_layout(0);

        let rgb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("IRIS REX3 RGB plane"),
            size: FB_BYTES,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("IRIS REX3 RGB readback"),
            size: FB_BYTES,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            rgb,
            readback,
            pending: Vec::new(),
            pending_damage: DamageSnapshot::default(),
            submitted_damage: DamageSnapshot::default(),
            gpu_current: false,
            submitted_work: false,
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

        if let Some(fill) = fill {
            if self.pending.len() == self.max_batch_draws {
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
                self.queue
                    .write_buffer(&self.rgb, 0, bytemuck::cast_slice(rgb));
                if let (Some(start), Some(telemetry)) = (upload_start, &mut self.report.telemetry) {
                    telemetry.upload_enqueue_ns += elapsed_ns(start);
                }
                self.report.bytes_uploaded += FB_BYTES;
                self.gpu_current = true;
            }
            if let Some(rect) = DamageRect::clipped(
                fill.min_x as i64,
                fill.min_y as i64,
                (fill.min_x + fill.width) as i64,
                (fill.min_y + fill.height) as i64,
            ) {
                self.pending_damage.mark_rect(DamagePlane::Rgb, rect);
            }
            self.pending.push(fill);
        }
        advance_flat_fill_context(context);
        self.report.handled_draws += 1;
        Ok(EnqueueResult::Enqueued)
    }

    pub(crate) fn submit(&mut self, reason: FlushReason) -> Result<(), String> {
        let result = self.submit_inner(false, reason).map(|_| ());
        if let Err(error) = &result {
            self.report.error = Some(error.clone());
        }
        result
    }

    pub(crate) fn flush(
        &mut self,
        rgb: &mut [u32],
        reason: FlushReason,
    ) -> Result<DamageSnapshot, String> {
        let result = self.flush_inner(rgb, reason);
        if let Err(error) = &result {
            self.report.error = Some(error.clone());
        }
        result
    }

    fn flush_inner(
        &mut self,
        rgb: &mut [u32],
        reason: FlushReason,
    ) -> Result<DamageSnapshot, String> {
        self.check_async_error()?;
        if self.pending.is_empty() && !self.submitted_work {
            return Ok(DamageSnapshot::default());
        }
        if rgb.len() != FB_PIXELS {
            return self.fail(format!(
                "REX3 RGB plane has {} pixels, expected {FB_PIXELS}",
                rgb.len()
            ));
        }

        let submission = self
            .submit_inner(true, reason)?
            .ok_or_else(|| "REX3 GPU readback requested without submitted work".to_owned())?;

        let map_start = self.report.telemetry.as_ref().map(|_| Instant::now());
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        self.readback
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
        let wait_result = self.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(MAP_TIMEOUT),
        });
        if let (Some(start), Some(telemetry)) = (wait_start, &mut self.report.telemetry) {
            telemetry.device_wait_ns += elapsed_ns(start);
        }
        wait_result.map_err(|error| format!("wait for REX3 GPU batch: {error}"))?;
        self.report.fences += 1;
        let callback_start = self.report.telemetry.as_ref().map(|_| Instant::now());
        let callback_result = receiver
            .recv_timeout(MAP_TIMEOUT)
            .map_err(|error| format!("wait for REX3 GPU readback callback: {error}"));
        if let (Some(start), Some(telemetry)) = (callback_start, &mut self.report.telemetry) {
            telemetry.map_callback_wait_ns += elapsed_ns(start);
        }
        callback_result?.map_err(|error| format!("map REX3 GPU readback: {error}"))?;
        self.check_async_error()?;

        if !timestamp_receivers.is_empty() {
            if let Err(error) = self.collect_timestamps(timestamp_receivers) {
                log::error!("REX3 GPU timestamp profiling unavailable: {error}");
                self.report.timestamp_collection_failed(error.clone());
                self.diagnostics.timestamps = TimestampSupport::Unavailable(error);
            }
        }

        let copy_start = self.report.telemetry.as_ref().map(|_| Instant::now());
        {
            let mapped = self
                .readback
                .slice(..)
                .get_mapped_range()
                .map_err(|error| format!("access REX3 GPU readback: {error}"))?;
            let words: &[u32] = bytemuck::cast_slice(&mapped);
            rgb.copy_from_slice(words);
        }
        self.readback.unmap();
        if let (Some(start), Some(telemetry)) = (copy_start, &mut self.report.telemetry) {
            telemetry.cpu_readback_copy_ns += elapsed_ns(start);
        }
        self.submitted_work = false;
        self.report.bytes_read_back += FB_BYTES;
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
        let fill_buffer = if self.pending.is_empty() {
            None
        } else {
            Some(
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("IRIS REX3 flat-fill batch parameters"),
                        contents: bytemuck::cast_slice(&self.pending),
                        usage: wgpu::BufferUsages::STORAGE,
                    }),
            )
        };
        let bind_group = fill_buffer.as_ref().map(|fill_buffer| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("IRIS REX3 flat-fill bind group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.rgb.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: fill_buffer.as_entire_binding(),
                    },
                ],
            })
        });
        let timestamps = self.timestamp_resources(!self.pending.is_empty(), copy_to_readback);

        if let (Some(start), Some(telemetry)) = (resource_start, &mut self.report.telemetry) {
            telemetry.resource_creation_ns += elapsed_ns(start);
        }
        let encode_start = self.report.telemetry.as_ref().map(|_| Instant::now());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("IRIS REX3 GPU batch"),
            });
        if let Some(bind_group) = &bind_group {
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
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        if copy_to_readback {
            if let Some(timestamps) = &timestamps {
                if let Some(index) = timestamps.transfer_index {
                    encoder.write_timestamp(&timestamps.queries, index);
                }
            }
            encoder.copy_buffer_to_buffer(&self.rgb, 0, &self.readback, 0, FB_BYTES);
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
        let submission = self.queue.submit([commands]);
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
            let poll_result = self.device.poll(wgpu::PollType::Poll);
            if let (Some(start), Some(telemetry)) = (poll_start, &mut self.report.telemetry) {
                telemetry.device_poll_ns += elapsed_ns(start);
            }
            poll_result.map_err(|error| format!("poll after REX3 GPU submission: {error}"))?;
            self.check_async_error()?;
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
        let queries = self.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("IRIS REX3 GPU timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count,
        });
        let resolve = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("IRIS REX3 timestamp resolve"),
            size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
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
            receiver
                .recv_timeout(MAP_TIMEOUT)
                .map_err(|error| format!("wait for timestamp readback callback: {error}"))?
                .map_err(|error| format!("map timestamp readback: {error}"))?;
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

    pub(crate) fn cpu_framebuffer_changed(&mut self) {
        debug_assert!(self.pending.is_empty());
        debug_assert!(!self.submitted_work);
        self.gpu_current = false;
    }

    fn check_async_error(&self) -> Result<(), String> {
        match self.async_error.lock().clone() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn fail<T>(&mut self, error: String) -> Result<T, String> {
        self.report.error = Some(error.clone());
        Err(error)
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
        || gpu.report.bytes_uploaded != FB_BYTES
        || gpu.report.bytes_read_back != FB_BYTES
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
    if min_x > max_x || min_y > max_y || context.wrmask & 0x00FF_FFFF == 0 {
        return FlatFillMatch::Supported(None);
    }

    FlatFillMatch::Supported(Some(FlatFill {
        min_x: min_x as u32,
        min_y: min_y as u32,
        width: (max_x - min_x + 1) as u32,
        height: (max_y - min_y + 1) as u32,
        color: context.get_colori() & 0x00FF_FFFF,
        write_mask: context.wrmask & 0x00FF_FFFF,
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

    #[test]
    fn block_match_clips_and_preserves_masked_rgb() {
        let mut context = context(DRAWMODE0_ADRMODE_BLOCK_SH, (-10, -5), (10, 5));
        context.wrmask = 0xAA00_00FF;
        let fill = supported_fill(&context);

        assert_eq!(fill.min_x, 0);
        assert_eq!(fill.min_y, 0);
        assert_eq!(fill.width, 11);
        assert_eq!(fill.height, 6);
        assert_eq!(fill.color, 0x0033_2211);
        assert_eq!(fill.write_mask, 0x0000_00FF);
        assert_eq!(fill.stride, FB_STRIDE);
        assert_eq!(fill.plane_len, FB_PIXELS as u32);
    }

    #[test]
    fn span_match_is_one_row() {
        let context = context(DRAWMODE0_ADRMODE_SPAN_SH, (7, 9), (17, 200));
        let fill = supported_fill(&context);

        assert_eq!((fill.min_x, fill.min_y), (7, 9));
        assert_eq!((fill.width, fill.height), (11, 1));
    }

    #[test]
    fn empty_supported_fills_do_not_require_gpu_work() {
        let mut masked = context(DRAWMODE0_ADRMODE_BLOCK_SH, (1, 2), (3, 4));
        masked.wrmask = 0xFF00_0000;
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
        advance_flat_fill_context(&mut block);
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
