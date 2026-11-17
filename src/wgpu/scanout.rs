//! The display's scanout on the GPU: a board's compose, one lane per screen
//! pixel, from its memory into a frame buffer on the GPU. Same-device
//! presenters use that output directly; explicit capture reads it back.
//!
//! A board provides a fragment defining `scanout_pixel(x, y) -> u32`
//! (0x00RRGGBB), the display slot its kernel reads at group 1 binding 0
//! (the scan-out registers and tables of the frame, as the board packs
//! them), and the bind group layouts of its memory from group 2 up, bound
//! as its raster kernels bind them. `shaders/scanout.wgsl` is the driver.

use std::sync::Arc;

use super::device::Gpu;
use super::memory::Staging;
use super::shader::{Fragment, Source};

const DRIVER: Fragment = Fragment::new("scanout.wgsl", include_str!("shaders/scanout.wgsl"));

/// A board's side of the scanout kernel.
pub struct ScanoutShaders<'a> {
    pub label: &'a str,
    /// Fragments defining `scanout_pixel` and declaring the display slot
    /// at group 1 binding 0.
    pub body: &'a [Fragment],
    pub library: &'a [Fragment],
    pub defines: &'a [(&'a str, u32)],
    /// Bind group layouts for groups 2 and up.
    pub layouts: &'a [&'a wgpu::BindGroupLayout],
}

/// The kernel and the layouts of groups 0 and 1.
pub struct ScanoutPipeline {
    pub frame_layout: wgpu::BindGroupLayout,
    pub slot_layout: wgpu::BindGroupLayout,
    pub pipeline: wgpu::ComputePipeline,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Header {
    width: u32,
    height: u32,
    stride: u32,
    pad: u32,
}

fn entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

impl ScanoutPipeline {
    pub fn new(gpu: &Gpu, arch: &ScanoutShaders<'_>) -> Result<ScanoutPipeline, String> {
        let frame_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scanout frame"),
            entries: &[entry(0, wgpu::BufferBindingType::Uniform), entry(1, wgpu::BufferBindingType::Storage { read_only: false })],
        });
        let slot_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scanout slot"),
            entries: &[entry(0, wgpu::BufferBindingType::Storage { read_only: true })],
        });
        let mut layouts: Vec<&wgpu::BindGroupLayout> = vec![&frame_layout, &slot_layout];
        layouts.extend_from_slice(arch.layouts);
        let mut src = Source::new(&format!("{} scanout", arch.label));
        for (k, v) in arch.defines {
            src = src.define(k, *v);
        }
        for f in arch.library {
            src = src.library(*f);
        }
        src = src.body(DRIVER);
        for f in arch.body {
            src = src.body(*f);
        }
        let pipeline = src.compute_pipeline(&gpu.device, "scanout", &layouts)?;
        Ok(ScanoutPipeline { frame_layout, slot_layout, pipeline })
    }
}

/// A frame on the GPU and the slot the kernel reads, with a readback.
pub struct Scanout {
    gpu: Arc<Gpu>,
    header_buf: wgpu::Buffer,
    out_buf: wgpu::Buffer,
    slot_buf: wgpu::Buffer,
    frame_group: wgpu::BindGroup,
    slot_group: wgpu::BindGroup,
    staging: Staging,
    max_pixels: u32,
    last_dimensions: Option<(u32, u32)>,
    slot_initialized: bool,
    /// Frames composed and bytes read back so far.
    pub frames: u64,
    pub read_back_bytes: u64,
    /// Composed-output readback attempts and successful copies.
    pub read_back_attempts: u64,
    pub read_back_successes: u64,
    pub read_back_attempted_bytes: u64,
}

impl Scanout {
    /// Room for `max_pixels` and a slot of `slot_bytes`.
    pub fn new(gpu: &Arc<Gpu>, pipeline: &ScanoutPipeline, max_pixels: u32, slot_bytes: u64) -> Scanout {
        let header_buf = gpu.buffer("scanout header", 16, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let out_buf = gpu.buffer("scanout frame", max_pixels.max(1) as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
        let slot_buf = gpu.buffer("scanout slot", slot_bytes.max(16), wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        let frame_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scanout frame"),
            layout: &pipeline.frame_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: header_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: out_buf.as_entire_binding() },
            ],
        });
        let slot_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scanout slot"),
            layout: &pipeline.slot_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: slot_buf.as_entire_binding() }],
        });
        Scanout {
            gpu: Arc::clone(gpu), header_buf, out_buf, slot_buf, frame_group, slot_group,
            staging: Staging::default(), max_pixels, last_dimensions: None, slot_initialized: false,
            frames: 0, read_back_bytes: 0, read_back_attempts: 0, read_back_successes: 0,
            read_back_attempted_bytes: 0,
        }
    }

    /// The device the scan-out runs on.
    pub fn gpu(&self) -> &Arc<Gpu> {
        &self.gpu
    }

    /// Compose a `width` x `height` frame under `slot` with `groups` bound
    /// from group 2 up, and read it back into `out` (rows of `width`).
    /// Ordered after every submission so far, so it shows whole batches.
    pub fn run(&mut self, pipeline: &ScanoutPipeline, width: u32, height: u32, slot: &[u8], groups: &[&wgpu::BindGroup], out: &mut [u32]) -> Result<(), String> {
        let pixels = width as u64 * height as u64;
        if pixels == 0 {
            return Ok(());
        }
        if out.len() < pixels as usize {
            return Err(format!("scanout: a {width} x {height} frame does not fit"));
        }
        self.compose(pipeline, width, height, slot, groups)?;
        self.read_output(width, height, out)
    }

    /// Read the last composed output, without composing or observing board
    /// memory. `out` changes only after the copy, map and device check succeed.
    pub fn read_output(&mut self, width: u32, height: u32, out: &mut [u32]) -> Result<(), String> {
        let pixels = width as u64 * height as u64;
        if pixels == 0 || self.last_dimensions != Some((width, height)) || self.frames == 0 {
            return Err("scanout: no composed frame with these dimensions".into());
        }
        if pixels > self.max_pixels as u64 || out.len() < pixels as usize {
            return Err(format!("scanout: a {width} x {height} frame does not fit"));
        }
        let bytes = pixels * 4;
        self.read_back_attempts += 1;
        self.read_back_attempted_bytes += bytes;
        self.gpu.check()?;
        let readback = self.staging.read(&self.gpu, &self.out_buf, &[(0, bytes)])?;
        self.gpu.check()?;
        if readback.len() != bytes as usize {
            return Err("scanout: incomplete composed-frame readback".into());
        }
        out[..pixels as usize].copy_from_slice(bytemuck::cast_slice(&readback));
        self.read_back_successes += 1;
        self.read_back_bytes += bytes;
        Ok(())
    }

    /// The buffer frames are composed into: 0x00RRGGBB words, rows of the
    /// last frame's width.
    pub fn output(&self) -> &wgpu::Buffer {
        &self.out_buf
    }

    /// Compose a frame as `run` does into [`output`](Self::output) and
    /// submit it, with no readback: a renderer on the same device presents
    /// it from there.
    pub fn compose(&mut self, pipeline: &ScanoutPipeline, width: u32, height: u32, slot: &[u8], groups: &[&wgpu::BindGroup]) -> Result<(), String> {
        self.compose_inner(pipeline, width, height, Some(slot), groups, true)
    }

    /// Reuse an already submitted slot, and only update the frame header when
    /// its dimensions change. The first nonempty frame must supply a slot.
    pub(crate) fn compose_cached(
        &mut self,
        pipeline: &ScanoutPipeline,
        width: u32,
        height: u32,
        slot: Option<&[u8]>,
        groups: &[&wgpu::BindGroup],
    ) -> Result<(), String> {
        self.compose_inner(pipeline, width, height, slot, groups, false)
    }

    fn compose_inner(
        &mut self,
        pipeline: &ScanoutPipeline,
        width: u32,
        height: u32,
        slot: Option<&[u8]>,
        groups: &[&wgpu::BindGroup],
        always_write_header: bool,
    ) -> Result<(), String> {
        let pixels = width as u64 * height as u64;
        if pixels == 0 {
            return Ok(());
        }
        if pixels > self.max_pixels as u64 {
            return Err(format!("scanout: a {width} x {height} frame does not fit"));
        }
        if let Some(slot) = slot {
            if slot.len() as u64 > self.slot_buf.size() {
                return Err(format!("scanout: a slot of {} bytes does not fit {}", slot.len(), self.slot_buf.size()));
            }
        } else if !self.slot_initialized {
            return Err("scanout: no previously submitted slot".into());
        }
        if always_write_header || self.last_dimensions != Some((width, height)) {
            self.gpu.write(&self.header_buf, 0, bytemuck::bytes_of(&Header { width, height, stride: width, pad: 0 }));
        }
        if let Some(slot) = slot {
            self.gpu.write(&self.slot_buf, 0, slot);
        }
        let mut enc = self.gpu.encoder("scanout");
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("scanout"), timestamp_writes: None });
            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(0, &self.frame_group, &[]);
            pass.set_bind_group(1, &self.slot_group, &[]);
            for (i, g) in groups.iter().enumerate() {
                pass.set_bind_group(2 + i as u32, *g, &[]);
            }
            pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        }
        self.gpu.submit(enc);
        self.frames += 1;
        self.gpu.check()?;
        self.last_dimensions = Some((width, height));
        self.slot_initialized |= slot.is_some();
        Ok(())
    }
}
