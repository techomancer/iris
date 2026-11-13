//! Presenting frames into a window through wgpu.
//!
//! A [`Presenter`] owns the window's surface on a [`Gpu`] and draws three
//! layers into it each frame: the frame, letterboxed into the window above
//! the status bar; the debug overlay, alpha-blended over the frame; and the
//! status bar. The frame comes either from the CPU ([`Frame::Cpu`], uploaded)
//! or from a buffer already on the device ([`Frame::Gpu`], a board's
//! scan-out output, unpacked by a small kernel with no readback). Sampling
//! is nearest at an integer scale and linear otherwise, as the GL renderer
//! does.
//!
//! The surface must come from the instance the device was opened on; a
//! board that draws on a [`Gpu`] shares it with the window this way
//! (`GfxDisplay::gpu`). Threading follows the GL renderer's rules: make the
//! presenter on the thread that owns the window, present from the board's
//! display thread.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use super::shader::{Fragment, Source};
use super::Gpu;

const PRESENT: Fragment = Fragment::new("present.wgsl", include_str!("shaders/present.wgsl"));
const UNPACK: Fragment = Fragment::new("unpack.wgsl", include_str!("shaders/unpack.wgsl"));

/// Where a quad goes (normalised device coordinates: x0, top, x1, bottom)
/// and the texture coordinates it shows (u0, v0, u1, v1).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default, PartialEq)]
struct Quad {
    rect: [f32; 4],
    uv: [f32; 4],
}

/// A frame to present.
pub enum Frame<'a> {
    /// Words with R in the low byte and A in the high (0xAABBGGRR), rows of
    /// `stride` words: what the boards' `Rex3Screen.rgba` holds.
    Cpu { rgba: &'a [u32], stride: usize, width: u32, height: u32 },
    /// 0x00RRGGBB words on this presenter's device, rows of `width`.
    Gpu { buffer: &'a wgpu::Buffer, width: u32, height: u32 },
}

/// One drawn layer: its texture, the quad's uniform and bind groups for
/// both samplers.
struct Layer {
    texture: wgpu::Texture,
    uniform: wgpu::Buffer,
    nearest: wgpu::BindGroup,
    linear: wgpu::BindGroup,
    width: u32,
    height: u32,
    last: Quad,
}

/// The window's surface and what draws into it.
pub struct Presenter {
    gpu: Arc<Gpu>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    opaque: wgpu::RenderPipeline,
    blended: wgpu::RenderPipeline,
    frame: Layer,
    overlay: Layer,
    status: Layer,
    unpack: wgpu::ComputePipeline,
    unpack_layout: wgpu::BindGroupLayout,
    unpack_size: wgpu::Buffer,
    /// The bind group for the last device frame unpacked, and its buffer.
    unpack_group: Option<(wgpu::Buffer, wgpu::BindGroup)>,
    status_rows: u32,
    pub frames: u64,
}

/// What a frame's placement in the window comes to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placement {
    /// The frame's quad in window pixels: x0, y0 (top), x1, y1.
    pub frame: [f32; 4],
    /// The status bar's height in window pixels, under the frame.
    pub status: f32,
    /// The scale the frame is drawn at.
    pub scale: f32,
}

impl Placement {
    /// The GL renderer's letterbox: the frame and the status bar under it
    /// scaled together to fit the window, centred, the left and top edges
    /// on whole pixels.
    pub fn letterbox(width: f32, height: f32, status_rows: f32, win_w: f32, win_h: f32) -> Placement {
        let total_h = height + status_rows;
        let scale = (win_w / width).min(win_h / total_h);
        let (w, h, sb) = (width * scale, height * scale, status_rows * scale);
        let x0 = ((win_w - w) * 0.5).floor();
        let y0 = ((win_h - h - sb) * 0.5).floor();
        Placement { frame: [x0, y0, x0 + w, y0 + h], status: sb, scale }
    }
}

impl Presenter {
    /// A presenter for `window` on `gpu`, its surface `width` x `height`
    /// pixels. Frames are at most `frame_w` x `frame_h`; the status bar is
    /// `status_w` x `status_rows`.
    pub fn new(
        gpu: &Arc<Gpu>,
        window: impl Into<wgpu::SurfaceTarget<'static>>,
        (width, height): (u32, u32),
        (frame_w, frame_h): (u32, u32),
        (status_w, status_rows): (u32, u32),
    ) -> Result<Presenter, String> {
        let surface = gpu.instance.create_surface(window).map_err(|e| format!("create surface: {e}"))?;
        if !gpu.adapter.is_surface_supported(&surface) {
            return Err(format!("{} cannot present to this window", gpu.caps.adapter));
        }
        let caps = surface.get_capabilities(&gpu.adapter);
        // A plain (non-sRGB) format: the frame's bytes are display values
        // already, as on the GL path; an sRGB target would encode them again.
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .or_else(|| caps.formats.first().copied())
            .ok_or("the surface offers no format")?;
        let present_mode = if caps.present_modes.contains(&wgpu::PresentMode::Fifo) { wgpu::PresentMode::Fifo } else { caps.present_modes[0] };
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: width.max(1),
            height: height.max(1),
            present_mode,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            color_space: Default::default(),
        };
        surface.configure(&gpu.device, &config);

        let device = &gpu.device;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("present layer"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });
        let module = Source::new("present").body(PRESENT).compile(device)?;
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("present"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |blend: Option<wgpu::BlendState>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("present"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState { module: &module, entry_point: Some("quad_vs"), compilation_options: Default::default(), buffers: &[] },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("quad_fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState { format, blend, write_mask: wgpu::ColorWrites::ALL })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let opaque = pipeline(None);
        let blended = pipeline(Some(wgpu::BlendState::ALPHA_BLENDING));

        let sampler = |filter: wgpu::FilterMode| {
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("present"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                mag_filter: filter,
                min_filter: filter,
                ..Default::default()
            })
        };
        let (nearest, linear) = (sampler(wgpu::FilterMode::Nearest), sampler(wgpu::FilterMode::Linear));
        let make_layer = |label: &str, w: u32, h: u32, storage: bool| {
            let mut usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
            if storage {
                usage |= wgpu::TextureUsages::STORAGE_BINDING;
            }
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage,
                view_formats: &[],
            });
            let view = texture.create_view(&Default::default());
            let uniform = gpu.buffer(label, size_of::<Quad>() as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
            let group = |s: &wgpu::Sampler| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(label),
                    layout: &layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(s) },
                        wgpu::BindGroupEntry { binding: 2, resource: uniform.as_entire_binding() },
                    ],
                })
            };
            let (n, l) = (group(&nearest), group(&linear));
            Layer { texture, uniform, nearest: n, linear: l, width: w, height: h, last: Quad::default() }
        };
        let frame = make_layer("present frame", frame_w, frame_h, true);
        let overlay = make_layer("present overlay", frame_w, frame_h, false);
        let status = make_layer("present status bar", status_w, status_rows.max(1), false);

        let unpack_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("present unpack"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });
        let unpack = Source::new("present unpack").body(UNPACK).compute_pipeline(device, "unpack", &[&unpack_layout])?;
        let unpack_size = gpu.buffer("present unpack size", 16, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        gpu.check()?;
        Ok(Presenter {
            gpu: Arc::clone(gpu),
            surface,
            config,
            opaque,
            blended,
            frame,
            overlay,
            status,
            unpack,
            unpack_layout,
            unpack_size,
            unpack_group: None,
            status_rows,
            frames: 0,
        })
    }

    /// The device this presenter draws with.
    pub fn gpu(&self) -> &Arc<Gpu> {
        &self.gpu
    }

    /// The window's new size in pixels.
    pub fn resize(&mut self, width: u32, height: u32) {
        let (w, h) = (width.max(1), height.max(1));
        if (w, h) != (self.config.width, self.config.height) {
            self.config.width = w;
            self.config.height = h;
            self.surface.configure(&self.gpu.device, &self.config);
        }
    }

    /// Words of 0xAABBGGRR, rows of `stride`, into `layer` at the top left.
    fn upload(&self, layer: &Layer, rgba: &[u32], stride: usize, width: u32, height: u32) {
        let (w, h) = (width.min(layer.width), height.min(layer.height));
        if w == 0 || h == 0 || rgba.len() < (h as usize - 1) * stride + w as usize {
            return;
        }
        self.gpu.write_texture(&layer.texture, rgba, stride, w, h);
    }

    /// Draw `frame` letterboxed, `overlay` (0xAABBGGRR, stride `stride`)
    /// over it when given, and `status` (stride `stride`, `status_rows`
    /// rows) under it, and present. A frame the surface cannot take now
    /// (minimised, being resized) is skipped.
    pub fn present(&mut self, frame: Frame<'_>, overlay: Option<&[u32]>, status: &[u32], stride: usize) -> Result<(), String> {
        let (fw, fh) = match frame {
            Frame::Cpu { width, height, .. } | Frame::Gpu { width, height, .. } => (width.min(self.frame.width), height.min(self.frame.height)),
        };
        if fw == 0 || fh == 0 {
            return Ok(());
        }
        // A texture the surface calls suboptimal (the window changed size
        // under it, as while its border is dragged) is given back before the
        // surface is configured again: wgpu refuses to configure a surface
        // with a texture of its outstanding, and that refusal left the
        // surface stale for every frame after. One retry on the fresh
        // configuration; a frame that still cannot be had is skipped.
        let mut target = None;
        for attempt in 0..2 {
            match self.surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(t) => {
                    target = Some(t);
                    break;
                }
                wgpu::CurrentSurfaceTexture::Suboptimal(t) => {
                    if attempt == 1 {
                        // Still suboptimal on the fresh configuration: the
                        // window moved again; draw it, the next frame catches up.
                        target = Some(t);
                        break;
                    }
                    drop(t);
                    self.surface.configure(&self.gpu.device, &self.config);
                }
                wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                    self.surface.configure(&self.gpu.device, &self.config);
                    return Ok(());
                }
                wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => return Ok(()),
                wgpu::CurrentSurfaceTexture::Validation => return self.gpu.check(),
            }
        }
        let Some(target) = target else { return Ok(()) };

        let mut enc = self.gpu.encoder("present");
        match frame {
            Frame::Cpu { rgba, stride, .. } => self.upload(&self.frame, rgba, stride, fw, fh),
            Frame::Gpu { buffer, .. } => {
                if self.unpack_group.as_ref().is_none_or(|(b, _)| b != buffer) {
                    let view = self.frame.texture.create_view(&Default::default());
                    let group = self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("present unpack"),
                        layout: &self.unpack_layout,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&view) },
                            wgpu::BindGroupEntry { binding: 2, resource: self.unpack_size.as_entire_binding() },
                        ],
                    });
                    self.unpack_group = Some((buffer.clone(), group));
                }
                self.gpu.write(&self.unpack_size, 0, bytemuck::cast_slice(&[fw, fh, 0u32, 0u32]));
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("present unpack"), timestamp_writes: None });
                pass.set_pipeline(&self.unpack);
                pass.set_bind_group(0, &self.unpack_group.as_ref().expect("the unpack group").1, &[]);
                pass.dispatch_workgroups(fw.div_ceil(8), fh.div_ceil(8), 1);
            }
        }
        if let Some(o) = overlay {
            self.upload(&self.overlay, o, stride, fw, fh);
        }
        self.upload(&self.status, status, stride, fw, self.status_rows);

        let p = Placement::letterbox(fw as f32, fh as f32, self.status_rows as f32, self.config.width as f32, self.config.height as f32);
        let integer = p.scale == p.scale.round() && p.scale >= 1.0;
        let r = p.frame;
        let win = (self.config.width as f32, self.config.height as f32);
        let q = &*self.gpu;
        place(q, win, &mut self.frame, r, fw, fh);
        place(q, win, &mut self.overlay, r, fw, fh);
        place(q, win, &mut self.status, [r[0], r[3], r[2], r[3] + p.status], fw, self.status_rows);

        let view = target.texture.create_view(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("present"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                ..Default::default()
            });
            fn group(l: &Layer, integer: bool) -> &wgpu::BindGroup {
                if integer { &l.nearest } else { &l.linear }
            }
            pass.set_pipeline(&self.opaque);
            pass.set_bind_group(0, group(&self.frame, integer), &[]);
            pass.draw(0..6, 0..1);
            if overlay.is_some() {
                pass.set_pipeline(&self.blended);
                pass.set_bind_group(0, group(&self.overlay, integer), &[]);
                pass.draw(0..6, 0..1);
                pass.set_pipeline(&self.opaque);
            }
            pass.set_bind_group(0, group(&self.status, integer), &[]);
            pass.draw(0..6, 0..1);
        }
        self.gpu.submit(enc);
        self.gpu.queue.present(target);
        self.frames += 1;
        self.gpu.check()
    }
}

/// Point `layer`'s quad at window pixels `r` (x0, y0, x1, y1) of a window
/// `win` pixels in size, showing its top-left `w` x `h` texels; the uniform
/// is written only when that changes.
fn place(gpu: &Gpu, (ww, wh): (f32, f32), layer: &mut Layer, r: [f32; 4], w: u32, h: u32) {
    let ndc_x = |x: f32| x / ww * 2.0 - 1.0;
    let ndc_y = |y: f32| 1.0 - y / wh * 2.0;
    let q = Quad {
        rect: [ndc_x(r[0]), ndc_y(r[1]), ndc_x(r[2]), ndc_y(r[3])],
        uv: [0.0, 0.0, w as f32 / layer.width as f32, h as f32 / layer.height as f32],
    };
    if q != layer.last {
        gpu.write(&layer.uniform, 0, bytemuck::bytes_of(&q));
        layer.last = q;
    }
}

#[cfg(test)]
mod tests {
    use super::Placement;

    /// The GL renderer's letterbox: a 1280 x 1024 frame with a 16-row bar in
    /// a window of exactly that size is drawn at 1x from the top left, and in
    /// a wider window it is centred at the height's scale.
    #[test]
    fn the_letterbox_matches_the_gl_renderer() {
        let p = Placement::letterbox(1280.0, 1024.0, 16.0, 1280.0, 1040.0);
        assert_eq!((p.frame, p.status, p.scale), ([0.0, 0.0, 1280.0, 1024.0], 16.0, 1.0));
        let p = Placement::letterbox(1280.0, 1024.0, 16.0, 3000.0, 2080.0);
        assert_eq!((p.frame, p.status, p.scale), ([220.0, 0.0, 2780.0, 2048.0], 32.0, 2.0));
    }
}
