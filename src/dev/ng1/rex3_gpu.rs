//! Opt-in, replay-only REX3 host-GPU prototype.

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use bytemuck::{Pod, Zeroable};
use parking_lot::Mutex;
use wgpu::util::DeviceExt;

use super::rex3::{
    CLIPMODE_CIDMATCH_MASK, CLIPMODE_CIDMATCH_SHIFT, CLIPMODE_ENSMASK_MASK,
    DRAWMODE0_ADRMODE_BLOCK, DRAWMODE0_ADRMODE_BLOCK_SH, DRAWMODE0_ADRMODE_SPAN,
    DRAWMODE0_ADRMODE_SPAN_SH, DRAWMODE0_OPCODE_DRAW, DRAWMODE1_COMPARE_DISABLE,
    DRAWMODE1_COMPARE_DISABLE_SH, DRAWMODE1_DRAWDEPTH_24, DRAWMODE1_LOGICOP_SRC,
    DRAWMODE1_LOGICOP_SRC_SH, DRAWMODE1_PLANES_RGB, DrawMode0, DrawMode1, OCTANT_XDEC, OCTANT_YDEC,
    REX3_COORD_BIAS, REX3_SCREEN_HEIGHT, REX3_SCREEN_WIDTH, Rex3Context,
};
use super::rex3_dirty::{DamagePlane, DamageRect, DamageSnapshot};

const FB_STRIDE: u32 = 2048;
const FB_PIXELS: usize = FB_STRIDE as usize * REX3_SCREEN_HEIGHT as usize;
const FB_BYTES: u64 = (FB_PIXELS * std::mem::size_of::<u32>()) as u64;
const WORKGROUP_SIZE: u32 = 256;
const MAX_BATCH_DRAWS: usize = 1024;
const MAP_TIMEOUT: Duration = Duration::from_secs(30);
const PROBE_STACK_BYTES: usize = 8 * 1024 * 1024;

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
}

impl fmt::Display for AdapterDiagnostics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "backend={:?} adapter={:?} type={:?} driver={:?} driver_info={:?} \
             max_storage_buffer={} max_workgroups={}",
            self.backend,
            self.name,
            self.device_type,
            self.driver,
            self.driver_info,
            self.max_storage_buffer_binding_size,
            self.max_compute_workgroups_per_dimension,
        )
    }
}

#[derive(Clone, Debug, Default)]
pub struct ReplayReport {
    pub handled_draws: u64,
    pub fallback_draws: u64,
    pub fallback_reasons: std::collections::BTreeMap<String, u64>,
    pub submissions: u64,
    pub fences: u64,
    pub bytes_uploaded: u64,
    pub bytes_read_back: u64,
    pub pending_draws: usize,
    pub error: Option<String>,
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
    fn pixels(self) -> u32 {
        self.width * self.height
    }
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
    diagnostics: AdapterDiagnostics,
    report: ReplayReport,
    async_error: Arc<Mutex<Option<String>>>,
}

impl Rex3Gpu {
    pub fn new(backend: &str) -> Result<Self, String> {
        let requested = RequestedBackend::parse(backend)?;
        let mut failures = Vec::new();
        for (name, backends) in requested.candidates() {
            match Self::new_for_backend(backends) {
                Ok(gpu) => return Ok(gpu),
                Err(error) => failures.push(format!("{name}: {error}")),
            }
        }
        Err(format!(
            "no usable REX3 GPU backend for {backend}: {}",
            failures.join("; ")
        ))
    }

    fn new_for_backend(backends: wgpu::Backends) -> Result<Self, String> {
        let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle();
        instance_desc.backends = backends;
        let instance = wgpu::Instance::new(instance_desc);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|error| format!("request adapter: {error}"))?;

        let info = adapter.get_info();
        let limits = adapter.limits();
        if limits.max_storage_buffer_binding_size < FB_BYTES {
            return Err(format!(
                "{} only exposes a {}-byte storage binding; REX3 RGB needs {FB_BYTES} bytes",
                info.name, limits.max_storage_buffer_binding_size
            ));
        }

        let descriptor = wgpu::DeviceDescriptor {
            label: Some("IRIS REX3 replay device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&descriptor))
            .map_err(|error| format!("request device for {}: {error}", info.name))?;

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
            diagnostics: AdapterDiagnostics {
                backend: info.backend,
                name: info.name,
                device_type: info.device_type,
                driver: info.driver,
                driver_info: info.driver_info,
                max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
                max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
            },
            report: ReplayReport::default(),
            async_error,
        })
    }

    pub fn diagnostics(&self) -> &AdapterDiagnostics {
        &self.diagnostics
    }

    pub fn report(&self) -> ReplayReport {
        let mut report = self.report.clone();
        report.pending_draws = self.pending.len();
        if report.error.is_none() {
            report.error = self.async_error.lock().clone();
        }
        report
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
        let fill = match flat_fill(context) {
            FlatFillMatch::Supported(fill) => fill,
            FlatFillMatch::Unsupported(reason) => {
                return Ok(EnqueueResult::Unsupported(reason));
            }
        };

        if let Some(fill) = fill {
            if self.pending.len() == MAX_BATCH_DRAWS {
                return Ok(EnqueueResult::FlushRequired);
            }
            if self.pending.is_empty() {
                if rgb.len() != FB_PIXELS {
                    return self.fail(format!(
                        "REX3 RGB plane has {} pixels, expected {FB_PIXELS}",
                        rgb.len()
                    ));
                }
                self.queue
                    .write_buffer(&self.rgb, 0, bytemuck::cast_slice(rgb));
                self.report.bytes_uploaded += FB_BYTES;
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

    pub(crate) fn flush(&mut self, rgb: &mut [u32]) -> Result<DamageSnapshot, String> {
        let result = self.flush_inner(rgb);
        if let Err(error) = &result {
            self.report.error = Some(error.clone());
        }
        result
    }

    fn flush_inner(&mut self, rgb: &mut [u32]) -> Result<DamageSnapshot, String> {
        self.check_async_error()?;
        if self.pending.is_empty() {
            return Ok(DamageSnapshot::default());
        }
        if rgb.len() != FB_PIXELS {
            return self.fail(format!(
                "REX3 RGB plane has {} pixels, expected {FB_PIXELS}",
                rgb.len()
            ));
        }

        let mut uniform_buffers = Vec::with_capacity(self.pending.len());
        let mut bind_groups = Vec::with_capacity(self.pending.len());
        for fill in &self.pending {
            let uniform = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("IRIS REX3 flat-fill parameters"),
                    contents: bytemuck::bytes_of(fill),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("IRIS REX3 flat-fill bind group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.rgb.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: uniform.as_entire_binding(),
                    },
                ],
            });
            uniform_buffers.push(uniform);
            bind_groups.push(bind_group);
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("IRIS REX3 replay batch"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("IRIS REX3 flat-fill batch"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            for (fill, bind_group) in self.pending.iter().zip(&bind_groups) {
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(fill.pixels().div_ceil(WORKGROUP_SIZE), 1, 1);
            }
        }
        encoder.copy_buffer_to_buffer(&self.rgb, 0, &self.readback, 0, FB_BYTES);
        let submission = self.queue.submit([encoder.finish()]);
        self.report.submissions += 1;

        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        self.readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(MAP_TIMEOUT),
            })
            .map_err(|error| format!("wait for REX3 GPU batch: {error}"))?;
        self.report.fences += 1;
        receiver
            .recv_timeout(MAP_TIMEOUT)
            .map_err(|error| format!("wait for REX3 GPU readback callback: {error}"))?
            .map_err(|error| format!("map REX3 GPU readback: {error}"))?;
        self.check_async_error()?;

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
        self.pending.clear();
        let damage = std::mem::take(&mut self.pending_damage);
        self.report.bytes_read_back += FB_BYTES;
        drop(uniform_buffers);
        Ok(damage)
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
    let _ = gpu.flush(&mut rgb)?;

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

    fn context(adrmode: u32, start: (i32, i32), end: (i32, i32)) -> Rex3Context {
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
        context
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
