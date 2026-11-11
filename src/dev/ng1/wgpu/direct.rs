//! Test-only scanout of the resident raster's authoritative planes.

use std::sync::Arc;

use super::resident::{Position, ResidentError, ResidentRaster, ResidentScanoutBindings, ResidentStatus};
use super::{Ng1Scanout, Ng1Slot, RESIDENT_BINDINGS, SCANOUT, VC2_WORDS};
use crate::dev::ng1::rex3::{Rex3, REX3_SCREEN_HEIGHT};
use crate::dev::ng1::vc2::{
    VC2_CTRL_DID_EN, VC2_REG_DID_ENTRY_PTR, VC2_REG_DISPLAY_CONTROL, VC2_REG_SCANLINE_LEN,
};
use crate::disp::{Ng1DisplayGenerations, Ng1DisplaySource, FB_STRIDE};
use crate::wgpu::{Fragment, Gpu, Scanout, ScanoutPipeline, ScanoutShaders, Source};

const HEIGHT: usize = REX3_SCREEN_HEIGHT as usize;
const DID_BYTES: usize = FB_STRIDE * HEIGHT;
const MAX_PIXELS: u32 = (FB_STRIDE * HEIGHT) as u32;
const DID_EXPAND: Fragment =
    Fragment::new("ng1_did_expand.wgsl", include_str!("shaders/did_expand.wgsl"));

fn scanout_error(error: String) -> ResidentError {
    if error.starts_with("wgpu device lost (") {
        ResidentError::DeviceLost(error)
    } else {
        ResidentError::Failed(error)
    }
}

/// The narrow resident seam: no raw buffer handles or CPU planes are exposed.
pub(crate) trait ResidentScanoutSource {
    fn scanout_bindings(&self, gpu: &Arc<Gpu>) -> Result<ResidentScanoutBindings, ResidentError>;
    fn scanout_status(&mut self) -> Result<ResidentStatus, ResidentError>;
}

impl ResidentScanoutSource for ResidentRaster {
    fn scanout_bindings(&self, gpu: &Arc<Gpu>) -> Result<ResidentScanoutBindings, ResidentError> {
        ResidentRaster::scanout_bindings(self, gpu)
    }

    fn scanout_status(&mut self) -> Result<ResidentStatus, ResidentError> {
        self.status()
    }
}

impl ResidentScanoutSource for &Rex3 {
    fn scanout_bindings(&self, gpu: &Arc<Gpu>) -> Result<ResidentScanoutBindings, ResidentError> {
        self.resident_scanout_bindings(gpu)
    }

    fn scanout_status(&mut self) -> Result<ResidentStatus, ResidentError> {
        self.resident_status()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct UploadCount {
    pub(crate) attempts: u64,
    pub(crate) attempted_bytes: u64,
    pub(crate) submitted: u64,
    pub(crate) submitted_bytes: u64,
}

impl UploadCount {
    fn attempt(&mut self, count: u64, bytes: u64) {
        self.attempts += count;
        self.attempted_bytes += bytes;
    }

    fn submit(&mut self, count: u64, bytes: u64) {
        self.submitted += count;
        self.submitted_bytes += bytes;
    }
}

/// Traffic caused by this compositor, not by raster initialization or
/// separately requested composed-frame capture. `did` counts decoded-map
/// uploads and must remain zero; `did_expansions` counts GPU expansion passes.
/// Submitted counts mean queued with scanout, not GPU completion.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ResidentScanoutTraffic {
    pub(crate) display: UploadCount,
    pub(crate) did: UploadCount,
    pub(crate) did_expansions: DispatchCount,
    pub(crate) raw_plane_uploads: UploadCount,
    pub(crate) raw_plane_readbacks: UploadCount,
    pub(crate) composed_readbacks: CaptureCount,
    pub(crate) frames_submitted: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CaptureCount {
    pub(crate) attempts: u64,
    pub(crate) attempted_bytes: u64,
    pub(crate) successes: u64,
    pub(crate) successful_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DispatchCount {
    pub(crate) attempts: u64,
    pub(crate) submitted: u64,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DidParams {
    did_ptr: u32,
    scan_len: u32,
    width: u32,
    height: u32,
    enabled: u32,
    pad: [u32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ResidentFrameStamp {
    serial: u64,
    pub(crate) position: Position,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

/// A same-device output. Its contents remain valid until this scanout
/// composes again; queue its consumer before releasing the borrow.
pub(crate) struct Ng1ResidentFrame<'a> {
    pub(crate) gpu: &'a Arc<Gpu>,
    pub(crate) buffer: &'a wgpu::Buffer,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) position: Position,
    stamp: ResidentFrameStamp,
}

impl Ng1ResidentFrame<'_> {
    pub(crate) fn stamp(&self) -> ResidentFrameStamp {
        self.stamp
    }
}

#[derive(Clone, Copy)]
pub(crate) enum CaptureFault {
    BeforeCopy,
    AfterMap,
}

/// The group retains the raster's plane buffers without exposing their
/// handles. Only DID, VC2 RAM and the small display slot are owned here.
pub(crate) struct Ng1ResidentScanout {
    pipeline: ScanoutPipeline,
    scanout: Scanout,
    bindings: ResidentScanoutBindings,
    did: wgpu::Buffer,
    vc2: wgpu::Buffer,
    did_params: wgpu::Buffer,
    did_pipeline: wgpu::ComputePipeline,
    did_group: wgpu::BindGroup,
    display_group: wgpu::BindGroup,
    uploaded: Option<Ng1DisplayGenerations>,
    dimensions: Option<(u32, u32)>,
    traffic: ResidentScanoutTraffic,
    last_frame: Option<ResidentFrameStamp>,
    capture_scratch: Vec<u32>,
    capture_fault: Option<CaptureFault>,
}

impl Ng1Scanout {
    /// Create a direct compositor from the resident raster's same-device,
    /// read-only binding seam. This does not submit raster work.
    pub(crate) fn new_resident(
        gpu: &Arc<Gpu>,
        resident: &impl ResidentScanoutSource,
    ) -> Result<Ng1ResidentScanout, ResidentError> {
        let bindings = resident.scanout_bindings(gpu)?;
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
            label: Some("resident ng1 display state"),
            entries: &[entry(0), entry(1)],
        });
        let pipeline = ScanoutPipeline::new(gpu, &ScanoutShaders {
            label: "resident ng1",
            body: &[SCANOUT],
            library: &[RESIDENT_BINDINGS],
            defines: &[],
            layouts: &[&bindings.layout, &layout],
        }).map_err(scanout_error)?;
        let scanout = Scanout::new(gpu, &pipeline, MAX_PIXELS, size_of::<Ng1Slot>() as u64);
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let did = gpu.buffer("resident ng1 decoded DID", DID_BYTES as u64, usage);
        let vc2 = gpu.buffer("resident ng1 vc2 ram", (2 * VC2_WORDS) as u64, usage);
        let did_params = gpu.buffer(
            "resident ng1 DID parameters",
            size_of::<DidParams>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let did_entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let did_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("resident ng1 DID expansion"),
            entries: &[
                did_entry(0, wgpu::BufferBindingType::Storage { read_only: true }),
                did_entry(1, wgpu::BufferBindingType::Storage { read_only: false }),
                did_entry(2, wgpu::BufferBindingType::Uniform),
            ],
        });
        let did_pipeline = Source::new("resident ng1 DID expansion")
            .body(DID_EXPAND)
            .compute_pipeline(&gpu.device, "expand_did", &[&did_layout])
            .map_err(scanout_error)?;
        let did_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resident ng1 DID expansion"),
            layout: &did_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: vc2.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: did.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: did_params.as_entire_binding() },
            ],
        });
        let display_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resident ng1 display state"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: did.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: vc2.as_entire_binding() },
            ],
        });
        gpu.check().map_err(scanout_error)?;
        Ok(Ng1ResidentScanout {
            pipeline, scanout, bindings, did, vc2, did_params, did_pipeline, did_group, display_group,
            uploaded: None, dimensions: None, traffic: ResidentScanoutTraffic::default(),
            last_frame: None, capture_scratch: Vec::new(), capture_fault: None,
        })
    }
}

impl Ng1ResidentScanout {
    pub(crate) fn gpu(&self) -> &Arc<Gpu> {
        self.scanout.gpu()
    }

    pub(crate) fn traffic(&self) -> ResidentScanoutTraffic {
        self.traffic
    }

    pub(crate) fn fail_next_capture(&mut self, fault: CaptureFault) {
        self.capture_fault = Some(fault);
    }

    fn expand_did(&self, params: &DidParams) -> Result<(), ResidentError> {
        let gpu = self.scanout.gpu();
        gpu.write(&self.did_params, 0, bytemuck::bytes_of(params));
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("resident ng1 DID expansion"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("resident ng1 DID expansion"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.did_pipeline);
            pass.set_bind_group(0, &self.did_group, &[]);
            pass.dispatch_workgroups((HEIGHT as u32).div_ceil(64), 1, 1);
        }
        gpu.submit(encoder);
        gpu.check().map_err(scanout_error)
    }

    /// Capture only the still-current composed output. An older frame token
    /// cannot read a newer composition, even when its dimensions are equal.
    pub(crate) fn capture(
        &mut self,
        stamp: ResidentFrameStamp,
        resident: &mut impl ResidentScanoutSource,
        out: &mut [u32],
    ) -> Result<(), ResidentError> {
        if self.last_frame != Some(stamp) {
            return Err(ResidentError::InvalidState("resident composed frame is stale"));
        }
        let status = resident.scanout_status()?;
        if status.submitted.epoch != stamp.position.epoch || status.submitted.sequence < stamp.position.sequence {
            return Err(ResidentError::InvalidState("resident composed frame belongs to an old epoch"));
        }
        let pixels = stamp.width as usize * stamp.height as usize;
        if out.len() < pixels {
            return Err(ResidentError::InvalidState("resident capture output is too small"));
        }
        let bytes = pixels as u64 * 4;
        self.traffic.composed_readbacks.attempts += 1;
        self.traffic.composed_readbacks.attempted_bytes += bytes;
        let fault = self.capture_fault.take();
        if matches!(fault, Some(CaptureFault::BeforeCopy)) {
            return Err(ResidentError::Failed("injected composed capture failure before copy".into()));
        }
        self.capture_scratch.resize(pixels, 0);
        self.scanout.read_output(stamp.width, stamp.height, &mut self.capture_scratch)
            .map_err(scanout_error)?;
        if matches!(fault, Some(CaptureFault::AfterMap)) {
            return Err(ResidentError::Failed("injected composed capture failure after map".into()));
        }
        self.scanout.gpu().check().map_err(scanout_error)?;
        out[..pixels].copy_from_slice(&self.capture_scratch);
        self.traffic.composed_readbacks.successes += 1;
        self.traffic.composed_readbacks.successful_bytes += bytes;
        Ok(())
    }

    /// Compose the submitted resident prefix; this method never submits
    /// pending raster work. The caller must serialize its submission, this
    /// compose and the frame consumer against later raster submissions.
    /// Epoch changes require constructing a new scanout.
    pub(crate) fn compose(
        &mut self,
        src: &Ng1DisplaySource<'_>,
        resident: &mut impl ResidentScanoutSource,
    ) -> Result<Ng1ResidentFrame<'_>, ResidentError> {
        let status = resident.scanout_status()?;
        if status.submitted.epoch != self.bindings.epoch {
            return Err(ResidentError::InvalidState("resident scanout bindings belong to an old or different epoch"));
        }
        if status.accepted != status.submitted {
            return Err(ResidentError::InvalidState("resident scanout requires all accepted work to be submitted"));
        }
        if src.width == 0 || src.width > FB_STRIDE || src.height == 0 || src.height > HEIGHT {
            return Err(ResidentError::InvalidState("resident scanout dimensions exceed the Newport planes"));
        }
        let cursor_x = src.vc2_regs[crate::dev::ng1::vc2::VC2_REG_CURRENT_CURSOR_X as usize] as i64
            - 31 + src.cursor_x_adjust as i64;
        if src.vc2_ram.len() != VC2_WORDS || src.topscan.checked_add(1).is_none()
            || !(i32::MIN as i64..=i32::MAX as i64).contains(&cursor_x)
        {
            return Err(ResidentError::InvalidState("resident scanout display state is incomplete"));
        }
        self.scanout.gpu().check().map_err(scanout_error)?;

        let generation = src.generations;
        let previous = self.uploaded;
        let changed = |select: fn(Ng1DisplayGenerations) -> u64| {
            previous.is_none_or(|old| old.source != generation.source || select(old) != select(generation))
        };
        let vc2_changed = changed(|g| g.vc2_ram);
        let vc2_regs_changed = changed(|g| g.vc2_regs);
        let geometry_changed = changed(|g| g.geometry);
        let dimensions = (src.width as u32, src.height as u32);
        let header_changed = self.dimensions != Some(dimensions);
        let did_changed = vc2_changed || vc2_regs_changed || header_changed;
        let slot_changed = vc2_regs_changed
            || changed(|g| g.xmap)
            || changed(|g| g.cmap)
            || changed(|g| g.ramdac)
            || geometry_changed;
        let slot = slot_changed.then(|| Ng1Slot::of_display(src));

        let did_params = DidParams {
            did_ptr: src.vc2_regs[VC2_REG_DID_ENTRY_PTR as usize] as u32,
            scan_len: (src.vc2_regs[VC2_REG_SCANLINE_LEN as usize] >> 5) as u32,
            width: dimensions.0,
            height: dimensions.1,
            enabled: u32::from(
                src.vc2_regs[VC2_REG_DISPLAY_CONTROL as usize] & VC2_CTRL_DID_EN != 0,
            ),
            pad: [0; 3],
        };
        let expansion_uploads = vc2_changed as u64 + did_changed as u64;
        let expansion_bytes = if vc2_changed { (2 * VC2_WORDS) as u64 } else { 0 }
            + if did_changed { size_of::<DidParams>() as u64 } else { 0 };
        let scanout_uploads = slot_changed as u64 + header_changed as u64;
        let scanout_bytes = if slot_changed { size_of::<Ng1Slot>() as u64 } else { 0 }
            + if header_changed { 16 } else { 0 };
        self.traffic.display.attempt(
            expansion_uploads + scanout_uploads,
            expansion_bytes + scanout_bytes,
        );
        if vc2_changed {
            self.scanout.gpu().write(&self.vc2, 0, bytemuck::cast_slice(src.vc2_ram));
        }
        if did_changed {
            self.traffic.did_expansions.attempts += 1;
            self.expand_did(&did_params)?;
            self.traffic.display.submit(expansion_uploads, expansion_bytes);
            self.traffic.did_expansions.submitted += 1;
        }
        let before_submission = self.scanout.frames;
        let result = self.scanout.compose_cached(
            &self.pipeline,
            dimensions.0,
            dimensions.1,
            slot.as_ref().map(|slot| bytemuck::bytes_of(slot).as_ref()),
            &[&self.bindings.group, &self.display_group],
        );
        if self.scanout.frames != before_submission {
            self.traffic.display.submit(scanout_uploads, scanout_bytes);
            self.traffic.frames_submitted += 1;
        }
        result.map_err(scanout_error)?;

        self.uploaded = Some(generation);
        self.dimensions = Some(dimensions);
        let stamp = ResidentFrameStamp {
            serial: self.scanout.frames,
            position: status.submitted,
            width: dimensions.0,
            height: dimensions.1,
        };
        self.last_frame = Some(stamp);
        Ok(Ng1ResidentFrame {
            gpu: self.scanout.gpu(),
            buffer: self.scanout.output(),
            width: dimensions.0,
            height: dimensions.1,
            position: status.submitted,
            stamp,
        })
    }
}
