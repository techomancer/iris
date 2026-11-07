//! Newport's display composed on the host GPU: the GL compositor's work
//! (`gl_compositor.rs`) and the software one's (`SwCompositor`) as a kernel
//! on the common scan-out driver (`crate::wgpu::scanout`), so the wgpu
//! window renderer composes REX3 frames on the device it presents with.
//!
//! The planes are uploaded each frame from the `CompositorSource` the
//! renderers already get; the slot carries the VC2 cursor registers, the
//! XMAP9 modes, the Bt445 RAMDAC and the CMAP.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use crate::compositor::CompositorSource;
use crate::dev::ng1::vc2::{
    VC2_CTRL_CURSOR_EN, VC2_CTRL_CURSOR_SIZE, VC2_REG_CURRENT_CURSOR_X, VC2_REG_CURSOR_ENTRY_PTR, VC2_REG_DISPLAY_CONTROL,
    VC2_REG_WORKING_CURSOR_Y,
};
use crate::wgpu::shader::Fragment;
use crate::wgpu::{Gpu, Scanout, ScanoutPipeline, ScanoutShaders};

#[cfg(test)]
mod tests;

const SCANOUT: Fragment = Fragment::new("ng1_scanout.wgsl", include_str!("shaders/scanout.wgsl"));

/// REX3's planes the display reads: 2048 x 1024.
const PLANE_WORDS: usize = 2048 * 1024;
/// The window-id map's bytes: one a screen pixel, rows of 2048.
const DID_BYTES: usize = crate::disp::FB_STRIDE * crate::disp::FB_MAX_H;
/// The VC2's RAM in 16-bit words.
const VC2_WORDS: usize = 32768;
/// The largest frame.
const MAX_PIXELS: u32 = (crate::disp::FB_STRIDE * crate::disp::FB_MAX_H) as u32;

/// A frame's registers and tables (`shaders/scanout.wgsl`, `Ng1Slot`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Ng1Slot {
    head: [u32; 16],
    xmap: [u32; 32],
    ramdac: [u32; 256],
    cmap: [u32; 8192],
}

impl Ng1Slot {
    fn of(src: &CompositorSource<'_>) -> Ng1Slot {
        let r = src.vc2_regs;
        let ctrl = r[VC2_REG_DISPLAY_CONTROL as usize];
        let mut head = [0u32; 16];
        head[0] = ((r[VC2_REG_CURRENT_CURSOR_X as usize] as i32) - 31 + src.cursor_x_adjust) as u32;
        head[1] = ((r[VC2_REG_WORKING_CURSOR_Y as usize] as i32) - 31) as u32;
        head[2] = r[VC2_REG_CURSOR_ENTRY_PTR as usize] as u32;
        head[3] = ((ctrl & VC2_CTRL_CURSOR_EN != 0) as u32) | ((ctrl & VC2_CTRL_CURSOR_SIZE != 0) as u32) << 1;
        head[4] = src.xmap_cursor_cmap as u32;
        head[5] = src.xmap_popup_cmap as u32;
        head[6] = (src.topscan + 1) as u32;
        head[7] = src.vc2_ram.len().min(VC2_WORDS) as u32;
        Ng1Slot { head, xmap: *src.xmap_mode, ramdac: *src.ramdac, cmap: *src.cmap }
    }
}

/// The display's compose on the GPU.
pub struct Ng1Scanout {
    pipeline: ScanoutPipeline,
    scanout: Scanout,
    rgb: wgpu::Buffer,
    aux: wgpu::Buffer,
    did: wgpu::Buffer,
    vc2: wgpu::Buffer,
    group: wgpu::BindGroup,
    /// Bytes of planes uploaded so far.
    pub uploaded: u64,
}

impl Ng1Scanout {
    pub fn new(gpu: &Arc<Gpu>) -> Result<Ng1Scanout, String> {
        let entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ng1 scanout planes"),
            entries: &[entry(0), entry(1), entry(2), entry(3)],
        });
        let pipeline = ScanoutPipeline::new(gpu, &ScanoutShaders { label: "ng1", body: &[SCANOUT], library: &[], defines: &[], layouts: &[&layout] })?;
        let scanout = Scanout::new(gpu, &pipeline, MAX_PIXELS, size_of::<Ng1Slot>() as u64);
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let rgb = gpu.buffer("ng1 rgb plane", 4 * PLANE_WORDS as u64, usage);
        let aux = gpu.buffer("ng1 aux plane", 4 * PLANE_WORDS as u64, usage);
        let did = gpu.buffer("ng1 window ids", DID_BYTES as u64, usage);
        let vc2 = gpu.buffer("ng1 vc2 ram", 2 * VC2_WORDS as u64, usage);
        let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ng1 scanout planes"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: rgb.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: aux.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: did.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: vc2.as_entire_binding() },
            ],
        });
        gpu.check()?;
        Ok(Ng1Scanout { pipeline, scanout, rgb, aux, did, vc2, group, uploaded: 0 })
    }

    /// The device it runs on.
    pub fn gpu(&self) -> &Arc<Gpu> {
        self.scanout.gpu()
    }

    /// The planes and the slot of `src` up to the GPU.
    fn upload(&mut self, src: &CompositorSource<'_>) -> Vec<u8> {
        let q = &self.scanout.gpu().queue;
        let words = |s: &[u32]| PLANE_WORDS.min(s.len()) & !1;
        let n = words(src.fb_rgb);
        q.write_buffer(&self.rgb, 0, bytemuck::cast_slice(&src.fb_rgb[..n]));
        let m = words(src.fb_aux);
        q.write_buffer(&self.aux, 0, bytemuck::cast_slice(&src.fb_aux[..m]));
        // The window-id map's rows on screen, whole words.
        let d = (src.height.max(1) * crate::disp::FB_STRIDE).min(src.did.len()).min(DID_BYTES) & !3;
        q.write_buffer(&self.did, 0, &src.did[..d]);
        let v = src.vc2_ram.len().min(VC2_WORDS) & !1;
        q.write_buffer(&self.vc2, 0, bytemuck::cast_slice(&src.vc2_ram[..v]));
        self.uploaded += 4 * (n + m) as u64 + d as u64 + 2 * v as u64;
        bytemuck::bytes_of(&Ng1Slot::of(src)).to_vec()
    }

    /// Compose `src` and submit it with no readback; the buffer the frame
    /// is in (0x00RRGGBB, rows of `src.width`), for a renderer on this
    /// device.
    pub fn compose(&mut self, src: &CompositorSource<'_>) -> Result<wgpu::Buffer, String> {
        let slot = self.upload(src);
        self.scanout.compose(&self.pipeline, src.width as u32, src.height as u32, &slot, &[&self.group])?;
        Ok(self.scanout.output().clone())
    }

    /// The buffer the last frame was composed into.
    pub fn output(&self) -> wgpu::Buffer {
        self.scanout.output().clone()
    }

    /// Compose `src` and read the frame back into `out` (0x00RRGGBB, rows
    /// of `src.width`).
    pub fn run(&mut self, src: &CompositorSource<'_>, out: &mut [u32]) -> Result<(), String> {
        let slot = self.upload(src);
        self.scanout.run(&self.pipeline, src.width as u32, src.height as u32, &slot, &[&self.group], out)
    }
}
