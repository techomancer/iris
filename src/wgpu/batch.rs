//! The batch: what the CPU records between two points where memory is
//! observed, and the two dispatches that draw it.
//!
//! A board appends one setup record per primitive (`prim`) and one state
//! slot per distinct state (`slot`); a record names its slot by index.
//! `flush` uploads the records, slots and tile boxes, runs the bin and
//! raster kernels and empties the recorder. Records, slots and the common
//! buffers live in fixed GPU buffers, so a board's bind groups are created
//! once.
//!
//! Buffer bindings: group 0 is the batch (`shaders/common.wgsl`); group 1
//! is the records at binding 0 and the slots at binding 1; groups from 2
//! up are the board's (its memory, its tables). `BatchPipelines` builds
//! both kernels around the board's hook fragments.

use std::marker::PhantomData;
use std::sync::Arc;

use bytemuck::Pod;

use super::device::Gpu;
use super::shader::{Fragment, Source};
use super::tiles::{BatchHeader, BusyHead, DispatchArgs, TileBox, TileGrid, MASK_WORDS, MAX_GROUPS, MAX_PRIMS, TILE_H, TILE_W};

/// The batch has no room for another primitive or slot: flush it first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Full;

const COMMON: Fragment = Fragment::new("common.wgsl", include_str!("shaders/common.wgsl"));
const BIN: Fragment = Fragment::new("bin.wgsl", include_str!("shaders/bin.wgsl"));
const BIN_ACCEPT_ALL: Fragment = Fragment::new("bin_accept_all.wgsl", include_str!("shaders/bin_accept_all.wgsl"));
const RASTER: Fragment = Fragment::new("raster.wgsl", include_str!("shaders/raster.wgsl"));

/// A board's side of the kernels.
pub struct ArchShaders<'a> {
    pub label: &'a str,
    /// Fragments defining `arch_tile_begin`, `arch_prim` and
    /// `arch_tile_end`, and the group 1 and 2+ bindings they use.
    pub raster: &'a [Fragment],
    /// A fragment defining `bin_accept`; None accepts every tile of the
    /// box. It may bind group 1 and up like the raster fragments.
    pub bin_accept: Option<Fragment>,
    /// Fragments `#include` may name.
    pub library: &'a [Fragment],
    /// `const NAME: u32` definitions both kernels see.
    pub defines: &'a [(&'a str, u32)],
    /// Bind group layouts for groups 2 and up.
    pub layouts: &'a [&'a wgpu::BindGroupLayout],
}

/// The two kernels and the layouts of groups 0 and 1. Group 0 has two
/// layouts: the bin pass also binds the dispatch arguments it raises,
/// which the raster pass consumes as its indirect buffer instead.
pub struct BatchPipelines {
    pub bin_layout: wgpu::BindGroupLayout,
    pub batch_layout: wgpu::BindGroupLayout,
    pub records_layout: wgpu::BindGroupLayout,
    pub bin: wgpu::ComputePipeline,
    pub raster: wgpu::ComputePipeline,
}

impl BatchPipelines {
    pub fn new(gpu: &Gpu, arch: &ArchShaders<'_>) -> Result<BatchPipelines, String> {
        let common = [
            layout_entry(0, wgpu::BufferBindingType::Uniform),
            layout_entry(1, wgpu::BufferBindingType::Storage { read_only: false }),
            layout_entry(2, wgpu::BufferBindingType::Storage { read_only: false }),
            layout_entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
            layout_entry(4, wgpu::BufferBindingType::Storage { read_only: true }),
        ];
        let batch_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("batch"),
            entries: &common,
        });
        let mut with_args = common.to_vec();
        with_args.push(layout_entry(5, wgpu::BufferBindingType::Storage { read_only: false }));
        let bin_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("batch bin"),
            entries: &with_args,
        });
        let records_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("batch records"),
            entries: &[
                layout_entry(0, wgpu::BufferBindingType::Storage { read_only: true }),
                layout_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
            ],
        });
        let mut bin_layouts: Vec<&wgpu::BindGroupLayout> = vec![&bin_layout, &records_layout];
        bin_layouts.extend_from_slice(arch.layouts);
        let mut layouts: Vec<&wgpu::BindGroupLayout> = vec![&batch_layout, &records_layout];
        layouts.extend_from_slice(arch.layouts);

        let base = |label: &str| {
            let mut s = Source::new(label)
                .define("TILE_W", TILE_W)
                .define("TILE_H", TILE_H)
                .define("MASK_WORDS", MASK_WORDS)
                .define("MAX_GROUPS", MAX_GROUPS)
                .library(COMMON);
            for (k, v) in arch.defines {
                s = s.define(k, *v);
            }
            for f in arch.library {
                s = s.library(*f);
            }
            s
        };
        let mut bin_src = base(&format!("{} bin", arch.label)).body(BIN);
        bin_src = bin_src.body(arch.bin_accept.unwrap_or(BIN_ACCEPT_ALL));
        let bin = bin_src.compute_pipeline(&gpu.device, "bin", &bin_layouts)?;

        let mut raster_src = base(&format!("{} raster", arch.label)).body(RASTER);
        for f in arch.raster {
            raster_src = raster_src.body(*f);
        }
        let raster = raster_src.compute_pipeline(&gpu.device, "raster", &layouts)?;
        Ok(BatchPipelines { bin_layout, batch_layout, records_layout, bin, raster })
    }
}

fn layout_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

/// The recorder and its GPU buffers. `R` is the board's setup record, `S`
/// its state slot; both are plain `repr(C)` data laid out as the board's
/// WGSL declares them.
pub struct Batch<R: Pod, S: Pod> {
    gpu: Arc<Gpu>,
    grid: TileGrid,
    tiles_allocated: u32,
    max_slots: u32,
    records: Vec<R>,
    boxes: Vec<[u32; 4]>,
    slots: Vec<S>,
    header_buf: wgpu::Buffer,
    masks_buf: wgpu::Buffer,
    busy_buf: wgpu::Buffer,
    args_buf: wgpu::Buffer,
    flags_buf: wgpu::Buffer,
    boxes_buf: wgpu::Buffer,
    records_buf: wgpu::Buffer,
    slots_buf: wgpu::Buffer,
    bin_group: wgpu::BindGroup,
    batch_group: wgpu::BindGroup,
    records_group: wgpu::BindGroup,
    /// Batches flushed so far.
    pub flushes: u64,
    /// Primitives drawn so far.
    pub prims_total: u64,
    _p: PhantomData<(R, S)>,
}

impl<R: Pod, S: Pod> Batch<R, S> {
    /// A recorder over `grid` with room for `max_slots` slots a batch.
    pub fn new(gpu: &Arc<Gpu>, pipelines: &BatchPipelines, grid: TileGrid, max_slots: u32) -> Batch<R, S> {
        let tiles = grid.tiles().max(1);
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let header_buf = gpu.buffer("batch header", 32, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let masks_buf = gpu.buffer("batch masks", tiles as u64 * MASK_WORDS as u64 * 4, storage);
        let busy_buf = gpu.buffer("batch busy", 16 + tiles as u64 * 4, storage);
        let args_buf = gpu.buffer("batch dispatch args", 16, storage | wgpu::BufferUsages::INDIRECT);
        let flags_buf = gpu.buffer("batch flags", tiles as u64 * 4, storage);
        let boxes_buf = gpu.buffer("batch boxes", MAX_PRIMS as u64 * 16, storage);
        let records_buf = gpu.buffer("batch records", MAX_PRIMS as u64 * size_of::<R>().max(4) as u64, storage);
        let slots_buf = gpu.buffer("batch slots", max_slots.max(1) as u64 * size_of::<S>().max(4) as u64, storage);
        let batch_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("batch"),
            layout: &pipelines.batch_layout,
            entries: &[
                entry(0, &header_buf),
                entry(1, &masks_buf),
                entry(2, &busy_buf),
                entry(3, &flags_buf),
                entry(4, &boxes_buf),
            ],
        });
        let bin_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("batch bin"),
            layout: &pipelines.bin_layout,
            entries: &[
                entry(0, &header_buf),
                entry(1, &masks_buf),
                entry(2, &busy_buf),
                entry(3, &flags_buf),
                entry(4, &boxes_buf),
                entry(5, &args_buf),
            ],
        });
        let records_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("batch records"),
            layout: &pipelines.records_layout,
            entries: &[entry(0, &records_buf), entry(1, &slots_buf)],
        });
        Batch {
            gpu: Arc::clone(gpu),
            grid,
            tiles_allocated: tiles,
            max_slots,
            records: Vec::with_capacity(MAX_PRIMS as usize),
            boxes: Vec::with_capacity(MAX_PRIMS as usize),
            slots: Vec::with_capacity(max_slots as usize),
            header_buf,
            masks_buf,
            busy_buf,
            args_buf,
            flags_buf,
            boxes_buf,
            records_buf,
            slots_buf,
            bin_group,
            batch_group,
            records_group,
            flushes: 0,
            prims_total: 0,
            _p: PhantomData,
        }
    }

    pub fn grid(&self) -> TileGrid {
        self.grid
    }

    /// Change the target. Only between batches; a grid with more tiles than
    /// were allocated is refused (allocate the largest at construction).
    pub fn set_grid(&mut self, grid: TileGrid) -> Result<(), String> {
        if !self.is_empty() {
            return Err("set_grid on a batch with primitives".into());
        }
        if grid.tiles() > self.tiles_allocated {
            return Err(format!("grid of {} tiles exceeds the {} allocated", grid.tiles(), self.tiles_allocated));
        }
        self.grid = grid;
        Ok(())
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn slots_used(&self) -> usize {
        self.slots.len()
    }

    /// Append a state slot; its index for records that follow.
    pub fn slot(&mut self, s: S) -> Result<u32, Full> {
        if self.slots.len() as u32 >= self.max_slots {
            return Err(Full);
        }
        self.slots.push(s);
        Ok(self.slots.len() as u32 - 1)
    }

    /// Append a primitive covering at most the pixel box `x0..=x1`,
    /// `y0..=y1`. Ok(None) when the box misses the target.
    pub fn prim(&mut self, r: R, x0: i32, y0: i32, x1: i32, y1: i32) -> Result<Option<u32>, Full> {
        let Some(b) = self.grid.tile_box(x0, y0, x1, y1) else { return Ok(None) };
        self.prim_tiles(r, b).map(Some)
    }

    /// Append a primitive over an explicit tile box.
    pub fn prim_tiles(&mut self, r: R, b: TileBox) -> Result<u32, Full> {
        if self.records.len() as u32 >= MAX_PRIMS {
            return Err(Full);
        }
        self.records.push(r);
        self.boxes.push(b.packed());
        Ok(self.records.len() as u32 - 1)
    }

    /// The group 0 and group 1 bind groups, for a board that encodes its
    /// own passes around the batch.
    pub fn bind_groups(&self) -> (&wgpu::BindGroup, &wgpu::BindGroup) {
        (&self.batch_group, &self.records_group)
    }

    /// Upload and draw what was recorded, with `groups` bound from group 2
    /// up. Returns the submission, or None when nothing was recorded.
    pub fn flush(&mut self, pipelines: &BatchPipelines, groups: &[&wgpu::BindGroup]) -> Option<wgpu::SubmissionIndex> {
        if self.records.is_empty() {
            self.slots.clear();
            return None;
        }
        let q = &self.gpu.queue;
        let header = BatchHeader {
            prim_count: self.records.len() as u32,
            tile_cols: self.grid.cols(),
            tile_rows: self.grid.rows(),
            mask_words: MASK_WORDS,
            origin_x: self.grid.origin_x,
            origin_y: self.grid.origin_y,
            _pad: [0; 2],
        };
        q.write_buffer(&self.header_buf, 0, bytemuck::bytes_of(&header));
        q.write_buffer(&self.busy_buf, 0, bytemuck::bytes_of(&BusyHead::default()));
        q.write_buffer(&self.args_buf, 0, bytemuck::bytes_of(&DispatchArgs::EMPTY));
        q.write_buffer(&self.boxes_buf, 0, bytemuck::cast_slice(&self.boxes));
        q.write_buffer(&self.records_buf, 0, bytemuck::cast_slice(&self.records));
        if !self.slots.is_empty() {
            q.write_buffer(&self.slots_buf, 0, bytemuck::cast_slice(&self.slots));
        }

        let mut enc = self.gpu.encoder("batch");
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("bin"), timestamp_writes: None });
            pass.set_pipeline(&pipelines.bin);
            pass.set_bind_group(0, &self.bin_group, &[]);
            pass.set_bind_group(1, &self.records_group, &[]);
            for (i, g) in groups.iter().enumerate() {
                pass.set_bind_group(2 + i as u32, *g, &[]);
            }
            pass.dispatch_workgroups(self.records.len() as u32, 1, 1);
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("raster"), timestamp_writes: None });
            pass.set_pipeline(&pipelines.raster);
            pass.set_bind_group(0, &self.batch_group, &[]);
            pass.set_bind_group(1, &self.records_group, &[]);
            for (i, g) in groups.iter().enumerate() {
                pass.set_bind_group(2 + i as u32, *g, &[]);
            }
            pass.dispatch_workgroups_indirect(&self.args_buf, 0);
        }
        let submission = self.gpu.submit(enc);
        self.flushes += 1;
        self.prims_total += self.records.len() as u64;
        self.records.clear();
        self.boxes.clear();
        self.slots.clear();
        Some(submission)
    }

    /// Zero the masks and flags. Only needed after a failed batch; a
    /// completed raster pass leaves them clean.
    pub fn reset(&self) {
        let mut enc = self.gpu.encoder("batch reset");
        enc.clear_buffer(&self.masks_buf, 0, None);
        enc.clear_buffer(&self.flags_buf, 0, None);
        self.gpu.submit(enc);
    }
}

fn entry(binding: u32, b: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource: b.as_entire_binding() }
}
