//! The adapter, device and queue the boards share, with the limits the
//! raster kernels need and the errors wgpu reports asynchronously.

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;

/// How to open the GPU. All fields have defaults; `backend` is the only
/// one users set.
#[derive(Clone, Debug)]
pub struct GpuOptions {
    /// `auto`, `vulkan`, `metal`, `dx12` or `gl`.
    pub backend: String,
    /// Prefer the discrete adapter.
    pub high_performance: bool,
    /// Ask for the optional features the kernels can use when present
    /// (subgroups, 64-bit integers and atomics, timestamps).
    pub optional_features: bool,
    /// Prefer a software adapter (lavapipe, SwiftShader) when one exists:
    /// what CI runs the goldens on.
    pub software: bool,
}

impl Default for GpuOptions {
    fn default() -> Self {
        GpuOptions {
            backend: "auto".into(),
            high_performance: true,
            optional_features: true,
            software: false,
        }
    }
}

/// What the device turned out to have, for logs and for kernels that
/// branch on it.
#[derive(Clone, Debug)]
pub struct Caps {
    pub backend: wgpu::Backend,
    pub adapter: String,
    pub device_type: wgpu::DeviceType,
    pub driver: String,
    pub max_storage_binding: u64,
    pub max_buffer: u64,
    pub max_workgroups: u32,
    pub subgroups: bool,
    pub int64: bool,
    pub int64_atomic_min_max: bool,
    pub timestamps: bool,
}

impl fmt::Display for Caps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?} {} ({:?}, {}) storage {} MiB buffer {} MiB groups {} subgroups {} int64 {} int64-atomics {} timestamps {}",
            self.backend,
            self.adapter,
            self.device_type,
            self.driver,
            self.max_storage_binding >> 20,
            self.max_buffer >> 20,
            self.max_workgroups,
            self.subgroups,
            self.int64,
            self.int64_atomic_min_max,
            self.timestamps,
        )
    }
}

/// One GPU for the process. Boards hold an `Arc<Gpu>`; they share the
/// queue, so work submitted by two boards is ordered by submission.
pub struct Gpu {
    /// The instance and adapter the device came from: a window's surface
    /// for presenting has to come from the same instance, and the adapter
    /// says whether it can present to it (`src/wgpu/present.rs`).
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub caps: Caps,
    /// The first asynchronous error (validation outside an error scope, or
    /// device loss). Checked by `check()`; a kernel bug shows up here.
    error: Arc<Mutex<Option<String>>>,
}

impl Gpu {
    /// Open the GPU `opts` asks for. Fails with the reasons each candidate
    /// backend gave when none works.
    pub fn new(opts: &GpuOptions) -> Result<Arc<Gpu>, String> {
        let mut failures = Vec::new();
        for (name, backends) in candidates(&opts.backend)? {
            match Self::open(backends, opts) {
                Ok(gpu) => {
                    log::info!("wgpu: {}", gpu.caps);
                    return Ok(gpu);
                }
                Err(e) => failures.push(format!("{name}: {e}")),
            }
        }
        Err(format!("no usable GPU for backend={}: {}", opts.backend, failures.join("; ")))
    }

    fn open(backends: wgpu::Backends, opts: &GpuOptions) -> Result<Arc<Gpu>, String> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = backends;
        let instance = wgpu::Instance::new(desc);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: if opts.high_performance {
                wgpu::PowerPreference::HighPerformance
            } else {
                wgpu::PowerPreference::None
            },
            force_fallback_adapter: opts.software,
            compatible_surface: None,
            ..Default::default()
        }))
        .map_err(|e| format!("request adapter: {e}"))?;
        let info = adapter.get_info();
        let have = adapter.features();
        let limits = adapter.limits();

        // Features: only what is there. Kernels branch on `caps`.
        let mut features = wgpu::Features::empty();
        if opts.optional_features {
            for f in [
                wgpu::Features::SUBGROUP,
                wgpu::Features::SHADER_INT64,
                wgpu::Features::SHADER_INT64_ATOMIC_MIN_MAX,
                wgpu::Features::TIMESTAMP_QUERY,
            ] {
                if have.contains(f) {
                    features |= f;
                }
            }
        }
        // Limits: the adapter's own for buffer sizes, so a 128 MiB board
        // memory is one binding; the defaults elsewhere.
        let mut required = wgpu::Limits::default();
        required.max_storage_buffer_binding_size = limits.max_storage_buffer_binding_size;
        required.max_buffer_size = limits.max_buffer_size;
        required.max_storage_buffers_per_shader_stage =
            limits.max_storage_buffers_per_shader_stage.min(16).max(required.max_storage_buffers_per_shader_stage);
        required.max_compute_workgroups_per_dimension = limits.max_compute_workgroups_per_dimension;
        required.max_compute_invocations_per_workgroup = limits.max_compute_invocations_per_workgroup;
        required.max_compute_workgroup_storage_size = limits.max_compute_workgroup_storage_size;

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("IRIS wgpu raster"),
            required_features: features,
            required_limits: required,
            ..Default::default()
        }))
        .map_err(|e| format!("request device on {}: {e}", info.name))?;

        let error = Arc::new(Mutex::new(None));
        let sink = Arc::clone(&error);
        device.on_uncaptured_error(Arc::new(move |e| {
            let msg = format!("wgpu error: {e}");
            log::error!("{msg}");
            sink.lock().get_or_insert(msg);
        }));
        let sink = Arc::clone(&error);
        device.set_device_lost_callback(move |reason, msg| {
            let msg = format!("wgpu device lost ({reason:?}): {msg}");
            log::error!("{msg}");
            sink.lock().get_or_insert(msg);
        });

        let caps = Caps {
            backend: info.backend,
            adapter: info.name.clone(),
            device_type: info.device_type,
            driver: format!("{} {}", info.driver, info.driver_info),
            max_storage_binding: limits.max_storage_buffer_binding_size,
            max_buffer: limits.max_buffer_size,
            max_workgroups: limits.max_compute_workgroups_per_dimension,
            subgroups: features.contains(wgpu::Features::SUBGROUP),
            int64: features.contains(wgpu::Features::SHADER_INT64),
            int64_atomic_min_max: features.contains(wgpu::Features::SHADER_INT64_ATOMIC_MIN_MAX),
            timestamps: features.contains(wgpu::Features::TIMESTAMP_QUERY),
        };
        Ok(Arc::new(Gpu { instance, adapter, device, queue, caps, error }))
    }

    /// The first asynchronous error, if any. Kernels that misbehave are
    /// reported here, not at the call that queued them.
    pub fn check(&self) -> Result<(), String> {
        match self.error.lock().clone() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Submit one command buffer.
    pub fn submit(&self, encoder: wgpu::CommandEncoder) -> wgpu::SubmissionIndex {
        self.queue.submit([encoder.finish()])
    }

    /// Block until `submission` has finished (or everything, with None).
    pub fn wait(&self, submission: Option<wgpu::SubmissionIndex>) -> Result<(), String> {
        self.device
            .poll(wgpu::PollType::Wait { submission_index: submission, timeout: Some(Duration::from_secs(30)) })
            .map(|_| ())
            .map_err(|e| format!("wait for GPU: {e}"))?;
        self.check()
    }

    /// Let callbacks run without blocking.
    pub fn poll(&self) {
        let _ = self.device.poll(wgpu::PollType::Poll);
    }

    /// Run `f` after every submission so far has finished. The callback
    /// runs on a wgpu thread or inside a later `poll`; keep it short.
    pub fn on_done(&self, f: impl FnOnce() + Send + 'static) {
        self.queue.on_submitted_work_done(f);
    }

    /// A command encoder with a label.
    pub fn encoder(&self, label: &str) -> wgpu::CommandEncoder {
        self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) })
    }

    /// A buffer of `size` bytes with `usage`, zeroed.
    pub fn buffer(&self, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false })
    }
}

fn candidates(backend: &str) -> Result<Vec<(&'static str, wgpu::Backends)>, String> {
    Ok(match backend {
        "vulkan" => vec![("vulkan", wgpu::Backends::VULKAN)],
        "metal" => vec![("metal", wgpu::Backends::METAL)],
        "dx12" => vec![("dx12", wgpu::Backends::DX12)],
        "gl" => vec![("gl", wgpu::Backends::GL)],
        "auto" if cfg!(target_os = "macos") => vec![("metal", wgpu::Backends::METAL)],
        "auto" if cfg!(target_os = "windows") => {
            vec![("vulkan", wgpu::Backends::VULKAN), ("dx12", wgpu::Backends::DX12)]
        }
        "auto" => vec![("vulkan", wgpu::Backends::VULKAN), ("gl", wgpu::Backends::GL)],
        other => return Err(format!("backend={other}: want auto, vulkan, metal, dx12 or gl")),
    })
}
