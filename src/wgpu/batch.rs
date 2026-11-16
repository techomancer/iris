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
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytemuck::Pod;

use super::device::Gpu;
use super::host::HostBuffer;
use std::sync::atomic::{AtomicBool, Ordering};
use super::shader::{Fragment, Source};
use super::tiles::{BatchHeader, BusyHead, DispatchArgs, TileBox, TileGrid, BUSY_ENTRY, COARSE, LIST_CAP, MASK_WORDS, MAX_GROUPS, MAX_PRIMS, TILE_H, TILE_W};

/// The batch has no room for another primitive or slot: flush it first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Full;

const COMMON: Fragment = Fragment::new("common.wgsl", include_str!("shaders/common.wgsl"));
const BIN: Fragment = Fragment::new("bin.wgsl", include_str!("shaders/bin.wgsl"));
const BIN_ACCEPT_ALL: Fragment = Fragment::new("bin_accept_all.wgsl", include_str!("shaders/bin_accept_all.wgsl"));
const RASTER: Fragment = Fragment::new("raster.wgsl", include_str!("shaders/raster.wgsl"));
const RASTER_ENTRY: Fragment = Fragment::new("raster_entry.wgsl", include_str!("shaders/raster_entry.wgsl"));
const RASTER_ENTRY_SUBGROUP: Fragment = Fragment::new("raster_entry_subgroup.wgsl", include_str!("shaders/raster_entry_subgroup.wgsl"));
const QUAD_SHARED: Fragment = Fragment::new("quad_shared.wgsl", include_str!("shaders/quad_shared.wgsl"));
const QUAD_SUBGROUP: Fragment = Fragment::new("quad_subgroup.wgsl", include_str!("shaders/quad_subgroup.wgsl"));

/// Whether the raster kernel's quads exchange values with subgroup
/// operations: the device has them, and `WGPU_NO_SUBGROUPS` is not set
/// (which tries the workgroup-memory path on a device that has them).
pub fn quad_subgroups(gpu: &Gpu) -> bool {
    gpu.caps.subgroups && std::env::var("WGPU_NO_SUBGROUPS").is_err()
}

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
    /// Bind group layouts for raster groups 2 and up. A custom `bin_accept`
    /// receives the same groups; the built-in accept-all bin omits them.
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
    bin_uses_arch_groups: bool,
}

impl BatchPipelines {
    pub fn new(gpu: &Gpu, arch: &ArchShaders<'_>) -> Result<BatchPipelines, String> {
        // Binding 1 is the tiles' primitive lists: the bin pass writes them,
        // the raster pass reads them.
        let common = |list_read_only: bool| {
            vec![
                layout_entry(0, wgpu::BufferBindingType::Uniform),
                layout_entry(1, wgpu::BufferBindingType::Storage { read_only: list_read_only }),
                layout_entry(2, wgpu::BufferBindingType::Storage { read_only: false }),
                layout_entry(4, wgpu::BufferBindingType::Storage { read_only: true }),
            ]
        };
        let batch_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("batch"),
            entries: &common(true),
        });
        let mut with_args = common(false);
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
        let bin_uses_arch_groups = arch.bin_accept.is_some();
        let mut bin_layouts: Vec<&wgpu::BindGroupLayout> = vec![&bin_layout, &records_layout];
        if bin_uses_arch_groups {
            bin_layouts.extend_from_slice(arch.layouts);
        }
        let mut layouts: Vec<&wgpu::BindGroupLayout> = vec![&batch_layout, &records_layout];
        layouts.extend_from_slice(arch.layouts);

        let base = |label: &str| {
            let mut s = Source::new(label)
                .define("TILE_W", TILE_W)
                .define("TILE_H", TILE_H)
                .define("MASK_WORDS", MASK_WORDS)
                .define("COARSE", COARSE)
                .define("LIST_CAP", LIST_CAP)
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

        let sub = quad_subgroups(gpu);
        let mut raster_src = base(&format!("{} raster", arch.label))
            .body(RASTER)
            .body(if sub { RASTER_ENTRY_SUBGROUP } else { RASTER_ENTRY })
            .body(if sub { QUAD_SUBGROUP } else { QUAD_SHARED });
        for f in arch.raster {
            raster_src = raster_src.body(*f);
        }
        let raster = raster_src.compute_pipeline(&gpu.device, "raster", &layouts)?;
        Ok(BatchPipelines {
            bin_layout,
            batch_layout,
            records_layout,
            bin,
            raster,
            bin_uses_arch_groups,
        })
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
    /// Order of this set's last submission by this batch.
    submitted: u64,
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
    /// Tile-primitive pairs the bin pass listed, summed over batches.
    pub pairs: u64,
    /// Batches by raster time: under 0.1, 0.3, 1, 3, 10 ms and over.
    pub raster_hist: [u64; 6],
    /// `WGPU_BATCH_LOG=file`: one line a batch (prims, busy tiles, pairs,
    /// bin ns, raster ns, graphics and memory clock MHz or 0).
    log: Option<std::io::LineWriter<std::fs::File>>,
    /// The card's clocks, when Linux reports them.
    clocks: Option<Clocks>,
    /// Raster time the clocks were read for, and it weighted by them
    /// (ns x MHz).
    clock_ns: f64,
    gfx_ns_mhz: f64,
    mem_ns_mhz: f64,
}

/// The card's average graphics and memory clocks from amdgpu's
/// `gpu_metrics` (Linux, readable without root): what a timing was taken
/// at, since the driver lowers the clocks while the GPU is mostly idle and
/// a GPU time means little without them. Only the layouts of format 1,
/// content revisions 1 to 3 (discrete RDNA and Vega cards) are read; any
/// other card, driver or system has none, and the report leaves them out.
struct Clocks {
    path: std::path::PathBuf,
}

impl Clocks {
    /// The `gpu_metrics` of the card with the adapter's PCI ids.
    fn of(gpu: &Gpu) -> Option<Clocks> {
        let info = gpu.adapter.get_info();
        let hex = |p: &std::path::Path| std::fs::read_to_string(p).ok().and_then(|s| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok());
        for e in std::fs::read_dir("/sys/class/drm").ok()?.flatten() {
            let dev = e.path().join("device");
            if hex(&dev.join("vendor")) == Some(info.vendor) && hex(&dev.join("device")) == Some(info.device) && dev.join("gpu_metrics").exists() {
                let c = Clocks { path: dev.join("gpu_metrics") };
                return c.read().map(|_| c);
            }
        }
        None
    }

    /// The average graphics and memory clocks, in MHz.
    fn read(&self) -> Option<(f64, f64)> {
        Clocks::parse(&std::fs::read(&self.path).ok()?)
    }

    /// The clocks in a `gpu_metrics` blob, when it has a layout we know.
    fn parse(b: &[u8]) -> Option<(f64, f64)> {
        // Header: size, format revision 1, content revision 1 to 3; the
        // average gfxclk and uclk at bytes 40 and 44.
        if b.len() < 46 || b[2] != 1 || !(1..=3).contains(&b[3]) {
            return None;
        }
        let at = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]);
        let (gfx, mem) = (at(40), at(44));
        // 0xffff: the firmware does not report it.
        if gfx == u16::MAX || mem == u16::MAX || gfx == 0 {
            return None;
        }
        Some((gfx as f64, mem as f64))
    }
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
        Some(Timing { queries, resolve, read, period_ns, batches: 0, prims: 0, bin_ns: 0.0, raster_ns: 0.0, tiles: 0, tiles_max: 0, pairs: 0, raster_hist: [0; 6], log: std::env::var("WGPU_BATCH_LOG").ok().and_then(|f| std::fs::File::create(f).ok()).map(std::io::LineWriter::new), clocks: Clocks::of(gpu), clock_ns: 0.0, gfx_ns_mhz: 0.0, mem_ns_mhz: 0.0 })
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
            let head: BusyHead = bytemuck::pod_read_unaligned(&v[32..48]);
            (t, head)
        };
        self.read.unmap();
        let ns = |a: u64, b: u64| b.saturating_sub(a) as f64 * self.period_ns;
        let raster = ns(t[2], t[3]);
        self.batches += 1;
        self.prims += prims as u64;
        self.bin_ns += ns(t[0], t[1]);
        self.raster_ns += raster;
        self.tiles += busy.count as u64;
        self.tiles_max = self.tiles_max.max(busy.count);
        self.pairs += busy.pairs as u64;
        let ms = raster / 1e6;
        let i = [0.1, 0.3, 1.0, 3.0, 10.0].iter().position(|&l| ms < l).unwrap_or(5);
        self.raster_hist[i] += 1;
        let (gfx, mem) = self.clocks.as_ref().and_then(|c| c.read()).unwrap_or((0.0, 0.0));
        if gfx != 0.0 {
            self.clock_ns += raster;
            self.gfx_ns_mhz += raster * gfx;
            self.mem_ns_mhz += raster * mem;
        }
        if let Some(l) = &mut self.log {
            use std::io::Write;
            let _ = writeln!(l, "{} {} {} {:.0} {:.0} {:.0} {:.0}", prims, busy.count, busy.pairs, ns(t[0], t[1]), raster, gfx, mem);
        }
    }

    /// One line: the totals and the averages per batch.
    pub fn report(&self) -> String {
        let n = self.batches.max(1) as f64;
        let clocks = if self.clock_ns > 0.0 {
            format!(", raster at gfx {:.0} MHz mem {:.0} MHz", self.gfx_ns_mhz / self.clock_ns, self.mem_ns_mhz / self.clock_ns)
        } else {
            String::new()
        };
        let line = format!(
            "gpu timing: {} batches, {:.1} prims/batch, bin {:.3} s, raster {:.3} s ({:.3} ms/batch), busy tiles {:.0}/batch (max {}), {:.2} prims/busy tile, raster ms <0.1/<0.3/<1/<3/<10/more: {:?}",
            self.batches,
            self.prims as f64 / n,
            self.bin_ns / 1e9,
            self.raster_ns / 1e9,
            self.raster_ns / 1e6 / n,
            self.tiles as f64 / n,
            self.tiles_max,
            self.pairs as f64 / self.tiles.max(1) as f64,
            self.raster_hist
        );
        line + &clocks
    }
}

/// Sets made with the batch: enough for the submissions a running board
/// keeps in flight. More are added, and kept, when all are busy.
const BATCH_SETS: usize = 4;
const SET_WAIT_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone, Copy)]
enum SetPolicy {
    Grow,
    Bounded(NonZeroUsize),
}

#[derive(Debug, PartialEq, Eq)]
enum SetChoice {
    Free(usize),
    Grow,
    Wait(usize),
}

/// Choose a free set in rotation, grow if permitted, otherwise wait for
/// the oldest submission. `state` returns (available, submission order).
fn choose_set(
    count: usize,
    last: usize,
    policy: SetPolicy,
    state: impl Fn(usize) -> (bool, u64),
) -> SetChoice {
    for d in 1..=count {
        let k = (last + d) % count;
        if state(k).0 {
            return SetChoice::Free(k);
        }
    }
    if matches!(policy, SetPolicy::Grow)
        || matches!(policy, SetPolicy::Bounded(limit) if count < limit.get())
    {
        return SetChoice::Grow;
    }
    SetChoice::Wait((0..count).min_by_key(|&k| state(k).1).expect("a batch has sets"))
}

/// The CPU-side recording, independent of the buffers that will receive it.
struct Recording<R: Pod, S: Pod> {
    max_slots: u32,
    records: Vec<R>,
    boxes: Vec<[u32; 4]>,
    slots: Vec<S>,
    cover: Option<TileBox>,
    pairs_bound: u32,
}

impl<R: Pod, S: Pod> Recording<R, S> {
    fn new(max_slots: u32) -> Self {
        Self {
            max_slots,
            records: Vec::with_capacity(MAX_PRIMS as usize),
            boxes: Vec::with_capacity(MAX_PRIMS as usize),
            slots: Vec::with_capacity(max_slots as usize),
            cover: None,
            pairs_bound: 0,
        }
    }

    fn slot(&mut self, slot: S) -> Result<u32, Full> {
        if self.slots.len() as u32 >= self.max_slots {
            return Err(Full);
        }
        self.slots.push(slot);
        Ok(self.slots.len() as u32 - 1)
    }

    fn fits(&self, b: TileBox) -> bool {
        (self.records.len() as u32) < MAX_PRIMS
            && self.pairs_bound.saturating_add(b.count()) <= LIST_CAP
    }

    fn push_prim(&mut self, record: R, b: TileBox) -> u32 {
        self.records.push(record);
        self.boxes.push(b.packed());
        self.pairs_bound += b.count();
        self.cover = Some(self.cover.map_or(b, |c| c.union(&b)));
        self.records.len() as u32 - 1
    }

    fn prim_tiles(&mut self, record: R, b: TileBox) -> Result<u32, Full> {
        if !self.fits(b) {
            return Err(Full);
        }
        Ok(self.push_prim(record, b))
    }

    fn prim_with_slot(
        &mut self,
        slot: S,
        b: Option<TileBox>,
        record: impl FnOnce(u32) -> R,
    ) -> Result<Option<(u32, u32)>, Full> {
        let Some(b) = b else { return Ok(None) };
        if self.slots.len() as u32 >= self.max_slots || !self.fits(b) {
            return Err(Full);
        }
        let slot_index = self.slots.len() as u32;
        let record = record(slot_index);
        self.slots.push(slot);
        let prim_index = self.push_prim(record, b);
        Ok(Some((prim_index, slot_index)))
    }

    fn clear(&mut self) {
        self.records.clear();
        self.boxes.clear();
        self.slots.clear();
        self.cover = None;
        self.pairs_bound = 0;
    }
}

/// The recorder and its GPU buffers. `R` is the board's setup record, `S`
/// its state slot; both are plain `repr(C)` data laid out as the board's
/// WGSL declares them.
pub struct Batch<R: Pod, S: Pod> {
    gpu: Arc<Gpu>,
    grid: TileGrid,
    tiles_allocated: u32,
    recording: Recording<R, S>,
    // Written by the kernels, shared by every set: the queue orders the
    // batches.
    list_buf: wgpu::Buffer,
    busy_buf: wgpu::Buffer,
    args_buf: wgpu::Buffer,
    sets: Vec<BatchSet>,
    set_policy: SetPolicy,
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
        Self::with_set_policy(gpu, pipelines, grid, max_slots, SetPolicy::Grow)
    }

    /// Like `new`, but cap the reusable sets. Once all are in flight, the
    /// fallible `try_next_set` / `try_flush` wait for the oldest submission.
    /// Existing `new` users retain the grow-on-demand policy.
    pub fn new_bounded(
        gpu: &Arc<Gpu>,
        pipelines: &BatchPipelines,
        grid: TileGrid,
        max_slots: u32,
        max_sets: NonZeroUsize,
    ) -> Batch<R, S> {
        Self::with_set_policy(gpu, pipelines, grid, max_slots, SetPolicy::Bounded(max_sets))
    }

    fn with_set_policy(
        gpu: &Arc<Gpu>,
        pipelines: &BatchPipelines,
        grid: TileGrid,
        max_slots: u32,
        set_policy: SetPolicy,
    ) -> Batch<R, S> {
        let initial_sets = match set_policy {
            SetPolicy::Grow => BATCH_SETS,
            SetPolicy::Bounded(limit) => BATCH_SETS.min(limit.get()),
        };
        let tiles = grid.tiles().max(1);
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let list_buf = gpu.buffer("batch tile lists", LIST_CAP as u64 * 4, storage);
        let busy_buf = gpu.buffer("batch busy", 16 + tiles as u64 * BUSY_ENTRY, storage | wgpu::BufferUsages::COPY_SRC);
        let args_buf = gpu.buffer("batch dispatch args", 16, storage | wgpu::BufferUsages::INDIRECT);
        let mut b = Batch {
            gpu: Arc::clone(gpu),
            grid,
            tiles_allocated: tiles,
            recording: Recording::new(max_slots),
            list_buf,
            busy_buf,
            args_buf,
            sets: Vec::with_capacity(initial_sets),
            set_policy,
            claimed: None,
            last_set: 0,
            flushes: 0,
            prims_total: 0,
            timing: Timing::new(gpu),
            _p: PhantomData,
        };
        for _ in 0..initial_sets {
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
        let slots = HostBuffer::new(gpu, &format!("batch slots {n}"), self.recording.max_slots.max(1) as u64 * size_of::<S>().max(4) as u64, storage);
        let batch_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("batch"),
            layout: &pipelines.batch_layout,
            entries: &[entry(0, &header.buffer), entry(1, &self.list_buf), entry(2, &self.busy_buf), entry(4, &boxes.buffer)],
        });
        let bin_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("batch bin"),
            layout: &pipelines.bin_layout,
            entries: &[entry(0, &header.buffer), entry(1, &self.list_buf), entry(2, &self.busy_buf), entry(4, &boxes.buffer), entry(5, &self.args_buf)],
        });
        let records_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("batch records"),
            layout: &pipelines.records_layout,
            entries: &[entry(0, &records.buffer), entry(1, &slots.buffer)],
        });
        self.sets.push(BatchSet {
            header, boxes, records, slots, bin_group, batch_group, records_group,
            done: Arc::new(AtomicBool::new(true)),
            submitted: 0,
        });
    }

    /// The set the next flush writes. A board with buffers of its own per
    /// set (`GpuRaster`'s stage) writes that set's before the flush.
    /// Bounded users should use `try_next_set` to handle a failed wait.
    pub fn next_set(&mut self, pipelines: &BatchPipelines) -> usize {
        self.try_next_set(pipelines).expect("wgpu: batch set acquisition failed")
    }

    fn choose_set(&self) -> SetChoice {
        choose_set(self.sets.len(), self.last_set, self.set_policy, |k| {
            (self.sets[k].done.load(Ordering::Acquire), self.sets[k].submitted)
        })
    }

    /// Claim a completed set, grow within the configured limit, or wait for
    /// the oldest in-flight set. An acquisition error makes no new claim.
    pub fn try_next_set(&mut self, pipelines: &BatchPipelines) -> Result<usize, String> {
        if matches!(self.set_policy, SetPolicy::Bounded(_)) {
            self.gpu.check()?;
        }
        if let Some(k) = self.claimed {
            return Ok(k);
        }
        let mut choice = self.choose_set();
        if !matches!(choice, SetChoice::Free(_)) {
            match self.set_policy {
                SetPolicy::Grow => self.gpu.poll(),
                SetPolicy::Bounded(_) => {
                    // Report HAL failures through the device-lost callback.
                    self.gpu.instance.poll_all(false);
                    self.gpu.check()?;
                }
            }
            choice = self.choose_set();
        }
        let k = match choice {
            SetChoice::Free(k) => k,
            SetChoice::Grow => {
                let k = self.sets.len();
                self.add_set(pipelines);
                log::info!("wgpu: batch sets grown to {}", k + 1);
                k
            }
            SetChoice::Wait(k) => {
                let start = Instant::now();
                loop {
                    self.gpu.instance.poll_all(false);
                    self.gpu.check()?;
                    if self.sets[k].done.load(Ordering::Acquire) {
                        break;
                    }
                    if start.elapsed() >= SET_WAIT_TIMEOUT {
                        return Err("timed out waiting for the oldest wgpu batch set".into());
                    }
                    std::thread::sleep(Duration::from_micros(100));
                }
                k
            }
        };
        self.claimed = Some(k);
        Ok(k)
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
        self.recording.records.is_empty()
    }

    pub fn len(&self) -> usize {
        self.recording.records.len()
    }

    pub fn slots_used(&self) -> usize {
        self.recording.slots.len()
    }

    /// Append a state slot; its index for records that follow.
    pub fn slot(&mut self, s: S) -> Result<u32, Full> {
        self.recording.slot(s)
    }

    /// Append a primitive covering at most the pixel box `x0..=x1`,
    /// `y0..=y1`. Ok(None) when the box misses the target.
    pub fn prim(&mut self, r: R, x0: i32, y0: i32, x1: i32, y1: i32) -> Result<Option<u32>, Full> {
        let Some(b) = self.grid.tile_box(x0, y0, x1, y1) else { return Ok(None) };
        self.prim_tiles(r, b).map(Some)
    }

    /// Append a primitive over an explicit tile box. Full when the batch
    /// holds MAX_PRIMS, or when its boxes could list more tile-primitive
    /// pairs than LIST_CAP (one box alone never does: a grid has fewer
    /// tiles).
    pub fn prim_tiles(&mut self, r: R, b: TileBox) -> Result<u32, Full> {
        self.recording.prim_tiles(r, b)
    }

    /// Append a new slot and its primitive as one transaction. The record
    /// constructor receives the slot index and runs only once both fit.
    /// Returns `(primitive_index, slot_index)`, or `None` when the pixel
    /// box misses the target. Neither a full batch nor an offscreen box
    /// consumes a slot or calls the constructor.
    pub fn prim_with_slot(
        &mut self,
        slot: S,
        x0: i32,
        y0: i32,
        x1: i32,
        y1: i32,
        record: impl FnOnce(u32) -> R,
    ) -> Result<Option<(u32, u32)>, Full> {
        self.recording.prim_with_slot(slot, self.grid.tile_box(x0, y0, x1, y1), record)
    }

    /// Upload and draw what was recorded, with `groups` bound from group 2
    /// up. Returns the submission, or None when nothing was recorded.
    pub fn flush(&mut self, pipelines: &BatchPipelines, groups: &[&wgpu::BindGroup]) -> Option<wgpu::SubmissionIndex> {
        self.try_flush(pipelines, groups).expect("wgpu: batch flush failed")
    }

    /// Fallible flush for a bounded set pool. On an acquisition error the
    /// recording remains intact and no new set is claimed.
    pub fn try_flush(&mut self, pipelines: &BatchPipelines, groups: &[&wgpu::BindGroup]) -> Result<Option<wgpu::SubmissionIndex>, String> {
        if self.recording.records.is_empty() {
            self.recording.slots.clear();
            return Ok(None);
        }
        let k = self.try_next_set(pipelines)?;
        self.claimed = None;
        self.last_set = k;
        let set = &mut self.sets[k];
        set.submitted = self.flushes;
        // The bin pass's grid: the coarse tiles the boxes cover.
        let cover = self.recording.cover.expect("a batch with records covers tiles");
        let (cx0, cy0) = (cover.x0 as u32 / COARSE, cover.y0 as u32 / COARSE);
        let (coarse_cols, coarse_rows) = (cover.x1 as u32 / COARSE - cx0 + 1, cover.y1 as u32 / COARSE - cy0 + 1);
        let header = BatchHeader {
            prim_count: self.recording.records.len() as u32,
            tile_cols: self.grid.cols(),
            tile_rows: self.grid.rows(),
            coarse_cols,
            origin_x: self.grid.origin_x,
            origin_y: self.grid.origin_y,
            coarse_x0: cx0,
            coarse_y0: cy0,
        };
        set.header.write(0, bytemuck::bytes_of(&header));
        set.boxes.write(0, bytemuck::cast_slice(&self.recording.boxes));
        set.records.write(0, bytemuck::cast_slice(&self.recording.records));
        if !self.recording.slots.is_empty() {
            set.slots.write(0, bytemuck::cast_slice(&self.recording.slots));
        }
        self.gpu.write(&self.busy_buf, 0, bytemuck::bytes_of(&BusyHead::default()));
        self.gpu.write(&self.args_buf, 0, bytemuck::bytes_of(&DispatchArgs::EMPTY));

        let mut enc = self.gpu.encoder("batch");
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("bin"), timestamp_writes: self.timing.as_ref().and_then(|t| t.writes(0)) });
            pass.set_pipeline(&pipelines.bin);
            pass.set_bind_group(0, &set.bin_group, &[]);
            pass.set_bind_group(1, &set.records_group, &[]);
            if pipelines.bin_uses_arch_groups {
                for (i, g) in groups.iter().enumerate() {
                    pass.set_bind_group(2 + i as u32, *g, &[]);
                }
            }
            pass.dispatch_workgroups(coarse_cols, coarse_rows, 1);
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
            t.take(&self.gpu, submission.clone(), self.recording.records.len());
        }
        let done = Arc::clone(&set.done);
        self.gpu.on_done(move || done.store(true, Ordering::Release));
        self.flushes += 1;
        self.prims_total += self.recording.records.len() as u64;
        self.recording.clear();
        Ok(Some(submission))
    }
}

fn entry(binding: u32, b: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource: b.as_entire_binding() }
}

#[cfg(test)]
mod recording_tests {
    use super::*;
    use std::cell::Cell;

    fn tile() -> TileBox {
        TileBox { x0: 0, y0: 0, x1: 0, y1: 0 }
    }

    fn snapshot(r: &Recording<u32, u32>) -> (Vec<u32>, Vec<[u32; 4]>, Vec<u32>, Option<TileBox>, u32) {
        (r.records.clone(), r.boxes.clone(), r.slots.clone(), r.cover, r.pairs_bound)
    }

    #[test]
    fn slot_and_record_are_transactional_at_each_limit() {
        let builds = Cell::new(0);
        let build = |index| {
            builds.set(builds.get() + 1);
            index
        };

        let mut slots = Recording::<u32, u32>::new(1);
        slots.slot(17).unwrap();
        let before = snapshot(&slots);
        assert_eq!(slots.prim_with_slot(18, Some(tile()), build), Err(Full));
        assert_eq!(snapshot(&slots), before);
        assert_eq!(builds.get(), 0);

        let mut primitives = Recording::<u32, u32>::new(2);
        primitives.slot(17).unwrap();
        for i in 0..MAX_PRIMS {
            primitives.prim_tiles(i, tile()).unwrap();
        }
        let before = snapshot(&primitives);
        assert_eq!(primitives.prim_with_slot(18, Some(tile()), build), Err(Full));
        assert_eq!(snapshot(&primitives), before);
        assert_eq!(builds.get(), 0);

        let mut pairs = Recording::<u32, u32>::new(2);
        pairs.slot(17).unwrap();
        let full_list = TileBox { x0: 0, y0: 0, x1: 2047, y1: 1023 };
        assert_eq!(full_list.count(), LIST_CAP);
        pairs.prim_tiles(42, full_list).unwrap();
        let before = snapshot(&pairs);
        assert_eq!(pairs.prim_with_slot(18, Some(tile()), build), Err(Full));
        assert_eq!(snapshot(&pairs), before);
        assert_eq!(builds.get(), 0);

        primitives.clear();
        assert_eq!(primitives.prim_with_slot(18, Some(tile()), build), Ok(Some((0, 0))));
        assert_eq!(primitives.slots, [18]);
        assert_eq!(primitives.records, [0]);
        assert_eq!(builds.get(), 1);
    }

    #[test]
    fn offscreen_and_panicking_record_do_not_consume_a_slot() {
        let mut recording = Recording::<u32, u32>::new(1);
        let grid = TileGrid::new(16, 16);
        let builds = Cell::new(0);
        let before = snapshot(&recording);
        let offscreen = grid.tile_box(16, 0, 20, 2);
        assert_eq!(recording.prim_with_slot(1, offscreen, |_| {
            builds.set(builds.get() + 1);
            7
        }), Ok(None));
        assert_eq!(snapshot(&recording), before);
        assert_eq!(builds.get(), 0);

        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = recording.prim_with_slot(1, Some(tile()), |_| -> u32 {
                panic!("record construction failed")
            });
        }));
        assert!(panic.is_err());
        assert_eq!(snapshot(&recording), before);
    }

    #[test]
    fn bounded_sets_reuse_free_sets_and_wait_for_the_oldest() {
        let cap = SetPolicy::Bounded(NonZeroUsize::new(4).unwrap());
        let busy = [(false, 12), (false, 9), (false, 11), (false, 10)];
        assert_eq!(choose_set(4, 0, cap, |i| busy[i]), SetChoice::Wait(1));
        assert_eq!(choose_set(4, 0, SetPolicy::Grow, |i| busy[i]), SetChoice::Grow);
        let larger_cap = SetPolicy::Bounded(NonZeroUsize::new(5).unwrap());
        assert_eq!(choose_set(4, 0, larger_cap, |i| busy[i]), SetChoice::Grow);
        let available = [(true, 12), (false, 9), (true, 11), (false, 10)];
        assert_eq!(choose_set(4, 0, cap, |i| available[i]), SetChoice::Free(2));
        assert_eq!(choose_set(4, 2, cap, |i| available[i]), SetChoice::Free(0));
    }

    #[test]
    fn full_retry_keeps_prepared_fill_and_advances_context_once() {
        std::thread::Builder::new()
            .name("batch-flat-fill-retry-test".into())
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                use crate::dev::ng1::rex3::{
                    DrawMode0, DrawMode1, Rex3Context, CLIPMODE_CIDMATCH_SHIFT,
                    DRAWMODE0_ADRMODE_BLOCK_SH, DRAWMODE0_OPCODE_DRAW,
                    DRAWMODE1_COMPARE_DISABLE_SH, DRAWMODE1_DRAWDEPTH_24,
                    DRAWMODE1_LOGICOP_SRC_SH, DRAWMODE1_PLANES_RGB, REX3_COORD_BIAS,
                };
                use crate::dev::ng1::rex3_flat_fill::prepare_flat_fill;

                let mut ctx = Rex3Context::power_on_default();
                ctx.drawmode0 = DrawMode0(DRAWMODE0_OPCODE_DRAW | DRAWMODE0_ADRMODE_BLOCK_SH | (1 << 8) | (1 << 9));
                ctx.drawmode1 = DrawMode1(DRAWMODE1_PLANES_RGB | (DRAWMODE1_DRAWDEPTH_24 << 3)
                    | DRAWMODE1_COMPARE_DISABLE_SH | (1 << 15) | DRAWMODE1_LOGICOP_SRC_SH);
                ctx.clipmode = 0xF << CLIPMODE_CIDMATCH_SHIFT;
                ctx.wrmask = u32::MAX;
                ctx.xstart = (10 + REX3_COORD_BIAS) << 11;
                ctx.xsave = ctx.xstart;
                ctx.xend = (12 + REX3_COORD_BIAS) << 11;
                ctx.ystart = (20 + REX3_COORD_BIAS) << 11;
                ctx.yend = (21 + REX3_COORD_BIAS) << 11;
                ctx.host_len = 0;
                ctx.host_cursor = 1;
                ctx.pat_bit = 4;
                ctx.zpat_bit = 5;

                let mut preparations = 0;
                let command = {
                    preparations += 1;
                    prepare_flat_fill(&ctx).unwrap()
                };
                let fill = command.fill().unwrap();
                let b = TileGrid::new(2048, 1024).tile_box(
                    fill.min_x as i32, fill.min_y as i32,
                    (fill.min_x + fill.width - 1) as i32,
                    (fill.min_y + fill.height - 1) as i32,
                );
                let mut recording = Recording::<u32, u32>::new(2);
                recording.slot(7).unwrap();
                for i in 0..MAX_PRIMS {
                    recording.prim_tiles(i, tile()).unwrap();
                }
                let before = snapshot(&recording);
                let old_context = (ctx.xstart, ctx.ystart, ctx.pat_bit, ctx.zpat_bit, ctx.host_len, ctx.host_cursor);
                let builds = Cell::new(0);
                let build = |index| {
                    builds.set(builds.get() + 1);
                    index
                };
                assert_eq!(recording.prim_with_slot(fill.color, b, build), Err(Full));
                assert_eq!(snapshot(&recording), before);
                assert_eq!((ctx.xstart, ctx.ystart, ctx.pat_bit, ctx.zpat_bit, ctx.host_len, ctx.host_cursor), old_context);
                assert_eq!((preparations, builds.get()), (1, 0));

                recording.clear();
                assert_eq!(recording.prim_with_slot(fill.color, b, build), Ok(Some((0, 0))));
                command.finish(&mut ctx);
                assert_eq!((preparations, builds.get()), (1, 1));
                assert_eq!(ctx.xstart, (10 + REX3_COORD_BIAS) << 11);
                assert_eq!(ctx.ystart, (22 + REX3_COORD_BIAS) << 11);
                assert_eq!((ctx.pat_bit, ctx.zpat_bit, ctx.host_len, ctx.host_cursor), (31, 31, 1, 0));
            })
            .expect("spawn retry test")
            .join()
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
    }
}

#[cfg(test)]
mod clock_tests {
    use super::Clocks;

    /// A `gpu_metrics` blob of format 1, content `rev`, with the average
    /// graphics and memory clocks at their places.
    fn blob(rev: u8, gfx: u16, mem: u16) -> Vec<u8> {
        let mut b = vec![0u8; 120];
        b[0] = 120;
        b[2] = 1;
        b[3] = rev;
        b[40..42].copy_from_slice(&gfx.to_le_bytes());
        b[44..46].copy_from_slice(&mem.to_le_bytes());
        b
    }

    #[test]
    fn known_layouts_give_the_average_clocks() {
        for rev in 1..=3 {
            assert_eq!(Clocks::parse(&blob(rev, 2639, 1000)), Some((2639.0, 1000.0)));
        }
    }

    #[test]
    fn other_layouts_and_unreported_clocks_give_none() {
        assert_eq!(Clocks::parse(&blob(0, 2639, 1000)), None);
        assert_eq!(Clocks::parse(&blob(4, 2639, 1000)), None);
        let mut apu = blob(1, 2639, 1000);
        apu[2] = 2;
        assert_eq!(Clocks::parse(&apu), None);
        assert_eq!(Clocks::parse(&blob(3, u16::MAX, 1000)), None);
        assert_eq!(Clocks::parse(&blob(3, 2639, u16::MAX)), None);
        assert_eq!(Clocks::parse(&blob(3, 0, 1000)), None);
        assert_eq!(Clocks::parse(&[1, 0, 1, 3]), None);
        assert_eq!(Clocks::parse(&[]), None);
    }
}
