//! The batch: what the CPU records between two points where memory is
//! observed, and the two dispatches that draw it.
//!
//! A board appends one setup record per primitive (`prim`) and one state
//! slot per distinct state (`slot`); a record names its slot by index.
//! `flush` writes the records, slots and tile boxes into host memory the
//! kernels read directly, runs the bin and raster kernels and empties the
//! recorder. Those buffers come in a few sets made with the batch, used in
//! turn and each reused once its submission completes; the buffers the
//! kernels write are shared. Nothing is allocated per batch.
//!
//! Buffer bindings: group 0 is the batch (`shaders/common.wgsl`); group 1
//! is the records at binding 0 and the slots at binding 1; groups from 2
//! up are the board's (its memory, its tables). `BatchPipelines` builds
//! both kernels around the board's hook fragments.

use std::marker::PhantomData;
use std::sync::Arc;

use bytemuck::Pod;

use super::device::Gpu;
use super::host::HostBuffer;
use std::sync::atomic::{AtomicBool, Ordering};
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

/// What the CPU writes for one batch, in host memory the kernels read
/// directly (`HostBuffer`): the header, the tile boxes, the records and the
/// slots, with the bind groups over them. Several sets rotate; one is
/// written again only once the submission that read it has completed.
struct BatchSet {
    header: HostBuffer,
    boxes: HostBuffer,
    records: HostBuffer,
    slots: HostBuffer,
    bin_group: wgpu::BindGroup,
    batch_group: wgpu::BindGroup,
    records_group: wgpu::BindGroup,
    /// No submission in flight reads the set.
    done: Arc<AtomicBool>,
}

/// GPU time per batch, measured with timestamp queries when
/// `WGPU_BATCH_TIMING` is set (and the device has them). Each flush then
/// waits for its batch, so CPU and GPU no longer overlap: the GPU figures
/// are true, the wall time is not.
pub struct Timing {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    read: wgpu::Buffer,
    period_ns: f64,
    pub batches: u64,
    pub prims: u64,
    pub bin_ns: f64,
    pub raster_ns: f64,
    /// Busy tiles, summed over batches, and the most in one.
    pub tiles: u64,
    pub tiles_max: u32,
    /// Batches by raster time: under 0.1, 0.3, 1, 3, 10 ms and over.
    pub raster_hist: [u64; 6],
}

impl Timing {
    fn new(gpu: &Gpu) -> Option<Timing> {
        if std::env::var("WGPU_BATCH_TIMING").is_err() || !gpu.device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        let queries = gpu.device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("batch timing"), ty: wgpu::QueryType::Timestamp, count: 4 });
        let resolve = gpu.buffer("batch timing resolve", 32, wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC);
        let read = gpu.buffer("batch timing read", 48, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
        let period_ns = gpu.queue.get_timestamp_period() as f64;
        Some(Timing { queries, resolve, read, period_ns, batches: 0, prims: 0, bin_ns: 0.0, raster_ns: 0.0, tiles: 0, tiles_max: 0, raster_hist: [0; 6] })
    }

    fn writes(&self, first: u32) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        Some(wgpu::ComputePassTimestampWrites { query_set: &self.queries, beginning_of_pass_write_index: Some(first), end_of_pass_write_index: Some(first + 1) })
    }

    /// Wait for the batch and add up its times and busy tiles.
    fn take(&mut self, gpu: &Gpu, submission: wgpu::SubmissionIndex, prims: usize) {
        let slice = self.read.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        if gpu.wait(Some(submission)).is_err() {
            return;
        }
        let _ = gpu.wait(None);
        let (t, busy) = {
            let Ok(v) = slice.get_mapped_range() else {
                self.read.unmap();
                return;
            };
            let t: [u64; 4] = bytemuck::pod_read_unaligned(&v[0..32]);
            let busy: u32 = bytemuck::pod_read_unaligned(&v[32..36]);
            (t, busy)
        };
        self.read.unmap();
        let ns = |a: u64, b: u64| b.saturating_sub(a) as f64 * self.period_ns;
        let raster = ns(t[2], t[3]);
        self.batches += 1;
        self.prims += prims as u64;
        self.bin_ns += ns(t[0], t[1]);
        self.raster_ns += raster;
        self.tiles += busy as u64;
        self.tiles_max = self.tiles_max.max(busy);
        let ms = raster / 1e6;
        let i = [0.1, 0.3, 1.0, 3.0, 10.0].iter().position(|&l| ms < l).unwrap_or(5);
        self.raster_hist[i] += 1;
    }

    /// One line: the totals and the averages per batch.
    pub fn report(&self) -> String {
        let n = self.batches.max(1) as f64;
        format!(
            "gpu timing: {} batches, {:.1} prims/batch, bin {:.3} s, raster {:.3} s ({:.3} ms/batch), busy tiles {:.0}/batch (max {}), raster ms <0.1/<0.3/<1/<3/<10/more: {:?}",
            self.batches,
            self.prims as f64 / n,
            self.bin_ns / 1e9,
            self.raster_ns / 1e9,
            self.raster_ns / 1e6 / n,
            self.tiles as f64 / n,
            self.tiles_max,
            self.raster_hist
        )
    }
}

/// Sets made with the batch: enough for the submissions a running board
/// keeps in flight. More are added, and kept, when all are busy.
const BATCH_SETS: usize = 4;

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
    // Written by the kernels, shared by every set: the queue orders the
    // batches.
    masks_buf: wgpu::Buffer,
    busy_buf: wgpu::Buffer,
    args_buf: wgpu::Buffer,
    flags_buf: wgpu::Buffer,
    sets: Vec<BatchSet>,
    /// The set the next flush writes, once chosen (`next_set`).
    claimed: Option<usize>,
    /// The set the last flush wrote.
    last_set: usize,
    /// Batches flushed so far.
    pub flushes: u64,
    /// Primitives drawn so far.
    pub prims_total: u64,
    /// GPU times, with WGPU_BATCH_TIMING.
    pub timing: Option<Timing>,
    _p: PhantomData<(R, S)>,
}

impl<R: Pod, S: Pod> Batch<R, S> {
    /// A recorder over `grid` with room for `max_slots` slots a batch.
    pub fn new(gpu: &Arc<Gpu>, pipelines: &BatchPipelines, grid: TileGrid, max_slots: u32) -> Batch<R, S> {
        let tiles = grid.tiles().max(1);
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let masks_buf = gpu.buffer("batch masks", tiles as u64 * MASK_WORDS as u64 * 4, storage);
        let busy_buf = gpu.buffer("batch busy", 16 + tiles as u64 * 4, storage | wgpu::BufferUsages::COPY_SRC);
        let args_buf = gpu.buffer("batch dispatch args", 16, storage | wgpu::BufferUsages::INDIRECT);
        let flags_buf = gpu.buffer("batch flags", tiles as u64 * 4, storage);
        let mut b = Batch {
            gpu: Arc::clone(gpu),
            grid,
            tiles_allocated: tiles,
            max_slots,
            records: Vec::with_capacity(MAX_PRIMS as usize),
            boxes: Vec::with_capacity(MAX_PRIMS as usize),
            slots: Vec::with_capacity(max_slots as usize),
            masks_buf,
            busy_buf,
            args_buf,
            flags_buf,
            sets: Vec::with_capacity(BATCH_SETS),
            claimed: None,
            last_set: 0,
            flushes: 0,
            prims_total: 0,
            timing: Timing::new(gpu),
            _p: PhantomData,
        };
        for _ in 0..BATCH_SETS {
            b.add_set(pipelines);
        }
        b
    }

    /// One more set of the CPU-written buffers and their bind groups.
    fn add_set(&mut self, pipelines: &BatchPipelines) {
        let gpu = &self.gpu;
        let storage = wgpu::BufferUsages::STORAGE;
        let n = self.sets.len();
        let header = HostBuffer::new(gpu, &format!("batch header {n}"), 32, wgpu::BufferUsages::UNIFORM);
        let boxes = HostBuffer::new(gpu, &format!("batch boxes {n}"), MAX_PRIMS as u64 * 16, storage);
        let records = HostBuffer::new(gpu, &format!("batch records {n}"), MAX_PRIMS as u64 * size_of::<R>().max(4) as u64, storage);
        let slots = HostBuffer::new(gpu, &format!("batch slots {n}"), self.max_slots.max(1) as u64 * size_of::<S>().max(4) as u64, storage);
        let batch_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("batch"),
            layout: &pipelines.batch_layout,
            entries: &[
                entry(0, &header.buffer),
                entry(1, &self.masks_buf),
                entry(2, &self.busy_buf),
                entry(3, &self.flags_buf),
                entry(4, &boxes.buffer),
            ],
        });
        let bin_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("batch bin"),
            layout: &pipelines.bin_layout,
            entries: &[
                entry(0, &header.buffer),
                entry(1, &self.masks_buf),
                entry(2, &self.busy_buf),
                entry(3, &self.flags_buf),
                entry(4, &boxes.buffer),
                entry(5, &self.args_buf),
            ],
        });
        let records_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("batch records"),
            layout: &pipelines.records_layout,
            entries: &[entry(0, &records.buffer), entry(1, &slots.buffer)],
        });
        self.sets.push(BatchSet { header, boxes, records, slots, bin_group, batch_group, records_group, done: Arc::new(AtomicBool::new(true)) });
    }

    /// The set the next flush writes: the next one whose last submission
    /// has completed, or a new one when none has. A board with buffers of
    /// its own per set (`GpuRaster`'s stage) writes that set's before the
    /// flush.
    pub fn next_set(&mut self, pipelines: &BatchPipelines) -> usize {
        if let Some(k) = self.claimed {
            return k;
        }
        let n = self.sets.len();
        let free = |sets: &[BatchSet]| (1..=n).map(|d| (self.last_set + d) % n).find(|&k| sets[k].done.load(Ordering::Acquire));
        let k = match free(&self.sets) {
            Some(k) => k,
            None => {
                self.gpu.poll();
                match free(&self.sets) {
                    Some(k) => k,
                    None => {
                        self.add_set(pipelines);
                        log::info!("wgpu: batch sets grown to {}", n + 1);
                        n
                    }
                }
            }
        };
        self.claimed = Some(k);
        k
    }

    /// Sets made so far.
    pub fn sets(&self) -> usize {
        self.sets.len()
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

    /// Upload and draw what was recorded, with `groups` bound from group 2
    /// up. Returns the submission, or None when nothing was recorded.
    pub fn flush(&mut self, pipelines: &BatchPipelines, groups: &[&wgpu::BindGroup]) -> Option<wgpu::SubmissionIndex> {
        if self.records.is_empty() {
            self.slots.clear();
            return None;
        }
        let k = self.next_set(pipelines);
        self.claimed = None;
        self.last_set = k;
        let set = &self.sets[k];
        let header = BatchHeader {
            prim_count: self.records.len() as u32,
            tile_cols: self.grid.cols(),
            tile_rows: self.grid.rows(),
            mask_words: MASK_WORDS,
            origin_x: self.grid.origin_x,
            origin_y: self.grid.origin_y,
            _pad: [0; 2],
        };
        set.header.write(0, bytemuck::bytes_of(&header));
        set.boxes.write(0, bytemuck::cast_slice(&self.boxes));
        set.records.write(0, bytemuck::cast_slice(&self.records));
        if !self.slots.is_empty() {
            set.slots.write(0, bytemuck::cast_slice(&self.slots));
        }
        self.gpu.write(&self.busy_buf, 0, bytemuck::bytes_of(&BusyHead::default()));
        self.gpu.write(&self.args_buf, 0, bytemuck::bytes_of(&DispatchArgs::EMPTY));

        let mut enc = self.gpu.encoder("batch");
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("bin"), timestamp_writes: self.timing.as_ref().and_then(|t| t.writes(0)) });
            pass.set_pipeline(&pipelines.bin);
            pass.set_bind_group(0, &set.bin_group, &[]);
            pass.set_bind_group(1, &set.records_group, &[]);
            for (i, g) in groups.iter().enumerate() {
                pass.set_bind_group(2 + i as u32, *g, &[]);
            }
            pass.dispatch_workgroups(self.records.len() as u32, 1, 1);
        }
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("raster"), timestamp_writes: self.timing.as_ref().and_then(|t| t.writes(2)) });
            pass.set_pipeline(&pipelines.raster);
            pass.set_bind_group(0, &set.batch_group, &[]);
            pass.set_bind_group(1, &set.records_group, &[]);
            for (i, g) in groups.iter().enumerate() {
                pass.set_bind_group(2 + i as u32, *g, &[]);
            }
            pass.dispatch_workgroups_indirect(&self.args_buf, 0);
        }
        if let Some(t) = &self.timing {
            enc.resolve_query_set(&t.queries, 0..4, &t.resolve, 0);
            enc.copy_buffer_to_buffer(&t.resolve, 0, &t.read, 0, 32);
            enc.copy_buffer_to_buffer(&self.busy_buf, 0, &t.read, 32, 16);
        }
        set.done.store(false, Ordering::Release);
        let submission = self.gpu.submit(enc);
        if let Some(t) = &mut self.timing {
            t.take(&self.gpu, submission.clone(), self.records.len());
        }
        let done = Arc::clone(&set.done);
        self.gpu.on_done(move || done.store(true, Ordering::Release));
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
