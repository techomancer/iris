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
    /// Writes waiting for the next submission (`write`).
    uploads: Mutex<Uploads>,
}

/// The writes queued by `Gpu::write` and `Gpu::write_texture`: copies out
/// of a fixed ring of staging buffers, recorded into one encoder that the
/// next `Gpu::submit` sends ahead of its own work.
///
/// The ring is allocated with the device, sized so a batch's worst case
/// fits one slot. A slot is written while mapped, unmapped by the
/// submission that copies out of it, and mapped again once that submission
/// completes. A writer that finds every slot in use adds one, which stays:
/// nothing is ever freed and allocated again.
///
/// `Queue::write_buffer` and `write_texture` create a fresh mappable
/// staging buffer for every call. On Vulkan that memory is CPU-visible
/// VRAM, freed when the submission completes and allocated again by the
/// next write, and the kernel's placement of each new buffer in the small
/// visible window dominated the profile of a draw-heavy session (40% of
/// all samples).
struct Uploads {
    slots: Vec<UploadSlot>,
    /// The slot being written.
    cur: usize,
    encoder: Option<wgpu::CommandEncoder>,
}

struct UploadSlot {
    buffer: wgpu::Buffer,
    /// In host memory (`host::raw`): the mapping, which stays, and the
    /// memory. Null and None for a wgpu-mapped slot (other backends), which
    /// is unmapped while in flight and mapped again behind it.
    ptr: *mut u8,
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    raw: Option<super::host::vk::RawMemory>,
    /// Mapped and writable from `cursor` on.
    mapped: bool,
    cursor: u64,
    /// Written since the last submission.
    used: bool,
    /// Unmapped by the submission being sent: to map again behind it.
    sent: bool,
    /// Set once the slot's last submission is done (the map callback, or
    /// the queue's work-done callback in host memory).
    ready: Arc<std::sync::atomic::AtomicBool>,
}

/// The ring: its first slots and their size. A batch's worst case is
/// about 6 MiB (4096 records of 352 bytes, the boxes, 1024 state slots and
/// the 4 MiB blit stage), so one slot holds a batch; four let four
/// submissions be in flight before the ring grows. A larger write (a
/// mirror upload) is split over slots.
const UPLOAD_SLOTS: usize = 4;
const UPLOAD_SLOT_BYTES: u64 = 8 << 20;

// SAFETY: the mapping is plain memory, reached only under the uploads lock.
unsafe impl Send for UploadSlot {}

impl UploadSlot {
    fn new(device: &wgpu::Device, i: usize) -> UploadSlot {
        let label = format!("upload ring {i}");
        let ready = Arc::new(std::sync::atomic::AtomicBool::new(false));
        if let Some((buffer, ptr, raw)) = super::host::raw(device, &label, UPLOAD_SLOT_BYTES, wgpu::BufferUsages::COPY_SRC) {
            #[cfg(not(any(target_os = "linux", target_os = "windows")))]
            let _ = raw;
            return UploadSlot {
                buffer,
                ptr,
                #[cfg(any(target_os = "linux", target_os = "windows"))]
                raw: Some(raw),
                mapped: true,
                cursor: 0,
                used: false,
                sent: false,
                ready,
            };
        }
        UploadSlot {
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&label),
                size: UPLOAD_SLOT_BYTES,
                usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: true,
            }),
            ptr: std::ptr::null_mut(),
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            raw: None,
            mapped: true,
            cursor: 0,
            used: false,
            sent: false,
            ready,
        }
    }

    /// Store `bytes` at `at` of the mapped slot.
    fn store(&self, at: u64, bytes: &[u8]) {
        if self.ptr.is_null() {
            self.buffer.slice(at..at + bytes.len() as u64).get_mapped_range_mut().expect("a mapped upload slot").copy_from_slice(bytes);
        } else {
            // SAFETY: reserve() handed out [at, at + len) inside the slot,
            // which no submission in flight reads.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.ptr.add(at as usize), bytes.len()) };
        }
    }
}

impl Uploads {
    fn new(device: &wgpu::Device) -> Uploads {
        let slots = (0..UPLOAD_SLOTS).map(|i| UploadSlot::new(device, i)).collect();
        Uploads { slots, cur: 0, encoder: None }
    }

    /// Room for `len` bytes (at most a slot): the slot and offset to write
    /// at. Moves to the next slot when the current one is full, and waits
    /// for it when it is still in flight.
    /// `bulk`: this write has already submitted part of itself, so slots
    /// in flight are its own and coming back; wait for one rather than
    /// grow the ring for a one-off large upload.
    fn reserve(&mut self, gpu: &Gpu, len: u64, bulk: &mut bool) -> (usize, u64) {
        assert!(len <= UPLOAD_SLOT_BYTES, "an upload piece larger than a slot");
        loop {
            for s in &mut self.slots {
                if !s.mapped && s.ready.swap(false, std::sync::atomic::Ordering::Acquire) {
                    s.mapped = true;
                    s.cursor = 0;
                }
            }
            let s = &mut self.slots[self.cur];
            if s.mapped && s.cursor + len <= UPLOAD_SLOT_BYTES {
                let at = s.cursor;
                // Copies need 4-byte offsets, mapped ranges 8-byte ones.
                s.cursor = (at + len + 7) & !7;
                s.used = true;
                return (self.cur, at);
            }
            // The current slot is full or in flight: a mapped slot nothing
            // was written to yet; else send the writes queued so far, which
            // frees their slots once done; else, every slot being in flight,
            // wait for one within a large write, or add one for good.
            let n = self.slots.len();
            if let Some(k) = (1..n).map(|d| (self.cur + d) % n).find(|&k| self.slots[k].mapped && !self.slots[k].used) {
                self.cur = k;
            } else if self.slots.iter().any(|s| s.used) {
                gpu.submit_locked(self, None);
                *bulk = true;
            } else if *bulk {
                let _ = gpu.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(Duration::from_secs(30)) });
            } else {
                self.slots.push(UploadSlot::new(&gpu.device, n));
                self.cur = n;
                log::info!("wgpu: upload ring grown to {} slots of {} MiB", n + 1, UPLOAD_SLOT_BYTES >> 20);
            }
        }
    }

    fn encoder(&mut self, device: &wgpu::Device) -> &mut wgpu::CommandEncoder {
        self.encoder.get_or_insert_with(|| device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("uploads") }))
    }
}

impl Drop for Gpu {
    /// The upload ring's host memory is ours to free, once the GPU is done
    /// with it.
    fn drop(&mut self) {
        let _ = self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(Duration::from_secs(30)) });
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        for s in &mut self.uploads.get_mut().slots {
            if let Some(raw) = s.raw.take() {
                s.buffer.destroy();
                raw.free();
            }
        }
    }
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
            // Small blocks for what is suballocated: the default (256 MiB
            // of device memory, 64 MiB of mappable memory a block) more than
            // doubled what a board took. Large buffers (a board's memory)
            // get blocks of their own either way.
            memory_hints: wgpu::MemoryHints::Manual { suballocated_device_memory_block_size: (4 << 20)..(32 << 20) },
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
        let uploads = Mutex::new(Uploads::new(&device));
        Ok(Arc::new(Gpu { instance, adapter, device, queue, caps, error, uploads }))
    }

    /// The first asynchronous error, if any. Kernels that misbehave are
    /// reported here, not at the call that queued them.
    pub fn check(&self) -> Result<(), String> {
        match self.error.lock().clone() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Submit one command buffer, after the writes queued so far.
    pub fn submit(&self, encoder: wgpu::CommandEncoder) -> wgpu::SubmissionIndex {
        let mut u = self.uploads.lock();
        let index = self.submit_locked(&mut u, Some(encoder));
        drop(u);
        // The ring's slots come back as their maps complete, which a poll
        // reports.
        let _ = self.device.poll(wgpu::PollType::Poll);
        index
    }

    /// Submit the queued writes and then `encoder`; unmap the slots the
    /// writes used and map them again behind the submission.
    fn submit_locked(&self, u: &mut Uploads, encoder: Option<wgpu::CommandEncoder>) -> wgpu::SubmissionIndex {
        let up = u.encoder.take().map(|e| e.finish());
        for s in &mut u.slots {
            if s.used {
                if s.ptr.is_null() {
                    s.buffer.unmap();
                }
                s.mapped = false;
                s.used = false;
                s.sent = true;
            }
        }
        let index = self.queue.submit(up.into_iter().chain(encoder.map(|e| e.finish())));
        for s in &mut u.slots {
            if std::mem::take(&mut s.sent) {
                let ready = Arc::clone(&s.ready);
                if !s.ptr.is_null() {
                    self.queue.on_submitted_work_done(move || ready.store(true, std::sync::atomic::Ordering::Release));
                    continue;
                }
                s.buffer.slice(..).map_async(wgpu::MapMode::Write, move |r| {
                    if r.is_ok() {
                        ready.store(true, std::sync::atomic::Ordering::Release);
                    }
                });
            }
        }
        index
    }

    /// Queue a write of `bytes` at `offset` of `target`, as
    /// `Queue::write_buffer` does: it lands before the work of the next
    /// `submit`. Offset and length must be multiples of 4.
    pub fn write(&self, target: &wgpu::Buffer, offset: u64, bytes: &[u8]) {
        let mut u = self.uploads.lock();
        let mut bulk = false;
        for (i, piece) in bytes.chunks(UPLOAD_SLOT_BYTES as usize).enumerate() {
            let (slot, at) = u.reserve(self, piece.len() as u64, &mut bulk);
            let buffer = u.slots[slot].buffer.clone();
            u.slots[slot].store(at, piece);
            u.encoder(&self.device).copy_buffer_to_buffer(&buffer, at, target, offset + i as u64 * UPLOAD_SLOT_BYTES, piece.len() as u64);
        }
    }

    /// Queue a write of `height` rows of `width` texels (4 bytes each,
    /// `stride` texels apart in `rows`) to the top-left of `texture`, as
    /// `Queue::write_texture` does.
    pub fn write_texture(&self, texture: &wgpu::Texture, rows: &[u32], stride: usize, width: u32, height: u32) {
        let pitch = (4 * width as u64).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as u64);
        let per_piece = (UPLOAD_SLOT_BYTES / pitch).max(1) as u32;
        let mut u = self.uploads.lock();
        let mut bulk = false;
        let mut y0 = 0;
        while y0 < height {
            let n = per_piece.min(height - y0);
            let (slot, at) = u.reserve(self, n as u64 * pitch, &mut bulk);
            let buffer = u.slots[slot].buffer.clone();
            for r in 0..n as usize {
                let src = &rows[(y0 as usize + r) * stride..][..width as usize];
                u.slots[slot].store(at + r as u64 * pitch, bytemuck::cast_slice(src));
            }
            u.encoder(&self.device).copy_buffer_to_texture(
                wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: at, bytes_per_row: Some(pitch as u32), rows_per_image: Some(n) } },
                wgpu::TexelCopyTextureInfo { texture, mip_level: 0, origin: wgpu::Origin3d { x: 0, y: y0, z: 0 }, aspect: wgpu::TextureAspect::All },
                wgpu::Extent3d { width, height: n, depth_or_array_layers: 1 },
            );
            y0 += n;
        }
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
