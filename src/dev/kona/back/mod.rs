//! The InfiniteReality pipe's drawing and display back end.
//!
//! `KonaBackend` takes the jobs the front end frames, keeps the pipe's
//! configuration (raster managers, texture memory per manager, managed area),
//! counts what it was sent and answers the `kona` monitor command. Behind it
//! are the commands that say what to draw, the tile scheduler, the pixel store
//! and the display.
//!
//! - `store`: the pixel store and the plane layout inside a pixel's words.
//! - `raster`: drawing into the store, the fill rule, and triangle set-up.
//! - `tiles`: the tile scheduler (slot ring, tile queues, workers, budget).
//! - `x`, `textport`, `gl`: the X server's integer path, the PROM's console and
//!   the GL client's commands; `context`: the state each rendering context keeps.
//! - `display`: the tables the scan-out reads and the frame's composition.
//! - `bench` (an ignored test): the fill rate of the integer drawing path.
//! - `census`, `replay` (tests only): the jobs a command stream sends the back
//!   end, and the replay of a recorded stream into a board.

#[cfg(test)]
mod bench;
pub mod display;
pub mod raster;
pub mod tiles;
pub mod snapshot;
pub mod store;
pub mod context;
pub mod gl;
pub mod textport;
pub mod x;
#[cfg(test)]
mod golden;
#[cfg(test)]
pub mod census;
#[cfg(test)]
mod replay;
#[cfg(test)]
mod switch_tests;


use super::defs::*;
use super::ident::KonaConfig;
use super::handle::DisplayHandle;
use super::job::{BackPort, Grammar, Job, Reply};
use display::clock::{Clock, DisplayThread, FRAME_PERIOD};
use crate::gfx_display::GfxDisplay;
use display::screen::Display;
use display::tables::Tables;
use display::{ColourMap, Compositor};
use store::layout::{BufferConfig, Layout, Refusal};
use textport::Textport;
use context::Contexts;
use raster::setup::Shape;
use raster::Rect;
use tiles::{ClearJob, RasterJob, Tiles};
use store::texture::{Exclusive, TextureStore, PAGE_WORDS};
use store::PixelStore;
use super::stream;
use crate::devlog::LogModule;
use std::io::{self, Write};
use std::sync::Arc;

/// The kinds of job the counters tell apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum JobKind {
    Textport,
    Kernel,
    Gl,
    Data,
    Bitmap,
    Video,
    Retrace,
    Context,
    Indirect,
    DmaIn,
    Dropped,
    FirmwareStart,
}

impl JobKind {
    const ALL: [JobKind; 12] = [
        JobKind::Textport,
        JobKind::Kernel,
        JobKind::Gl,
        JobKind::Data,
        JobKind::Bitmap,
        JobKind::Video,
        JobKind::Retrace,
        JobKind::Context,
        JobKind::Indirect,
        JobKind::DmaIn,
        JobKind::Dropped,
        JobKind::FirmwareStart,
    ];

    fn name(self) -> &'static str {
        match self {
            JobKind::Textport => "textport commands",
            JobKind::Kernel => "kernel commands",
            JobKind::Gl => "GL commands",
            JobKind::Data => "data runs",
            JobKind::Bitmap => "bitmaps",
            JobKind::Video => "video commands",
            JobKind::Retrace => "retrace requests",
            JobKind::Context => "context changes",
            JobKind::Indirect => "indirect stores",
            JobKind::DmaIn => "DMA packets",
            JobKind::Dropped => "dropped words",
            JobKind::FirmwareStart => "firmware starts",
        }
    }

    fn of(job: &Job<'_>) -> JobKind {
        match job {
            Job::Command { grammar: Grammar::Textport, .. } => JobKind::Textport,
            Job::Command { grammar: Grammar::Kernel, .. } => JobKind::Kernel,
            Job::Command { grammar: Grammar::Gl, .. } => JobKind::Gl,
            Job::Data { .. } => JobKind::Data,
            Job::Bitmap { .. } => JobKind::Bitmap,
            Job::Video { .. } => JobKind::Video,
            Job::Retrace { .. } => JobKind::Retrace,
            Job::Context(_) => JobKind::Context,
            Job::Indirect { .. } => JobKind::Indirect,
            Job::DmaIn { .. } => JobKind::DmaIn,
            Job::Dropped { .. } => JobKind::Dropped,
            Job::FirmwareStart { .. } => JobKind::FirmwareStart,
        }
    }
}

/// Whether `selector` moves a region's pixels: a copy-down or a swap.
fn moves_pixels(selector: u32) -> bool {
    matches!(selector, PIPE_MEMORY_COPY_DOWN | PIPE_MEMORY_SWAP_IN | PIPE_MEMORY_SWAP_OUT)
}

/// The kernel's pipe-memory region commands the back end has seen, and the bytes
/// the swaps named. The front end frames them, answers them and moves the bytes
/// (a swap-out writes zeros of the region's size, a swap-in reads the buffer and
/// discards it); nothing in a pool lives in the back end yet, so it counts what
/// it was told. The texture and list steps give a swap-out the model's own
/// encoding of the region (the hardware's is unknown, and nothing the guest does
/// interprets the bytes between the two, inferred).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct RegionCounts {
    placed: u64,
    copy_downs: u64,
    swapped_out: u64,
    swapped_in: u64,
    bytes_out: u64,
    bytes_in: u64,
}

impl RegionCounts {
    /// Count a kernel command if it is a region command. A swap's size is its
    /// size in granules times the granule times the copy count.
    fn note(&mut self, selector: u32, args: &[u32]) {
        let bytes = |a: &[u32]| a.get(2..5).map_or(0, |s| s.iter().map(|&w| w as u64).product());
        match selector {
            PIPE_MEMORY_RESERVE => self.placed += 1,
            PIPE_MEMORY_COPY_DOWN => self.copy_downs += 1,
            PIPE_MEMORY_SWAP_OUT => {
                self.swapped_out += 1;
                self.bytes_out += bytes(args);
            }
            PIPE_MEMORY_SWAP_IN => {
                self.swapped_in += 1;
                self.bytes_in += bytes(args);
            }
            _ => {}
        }
    }
}

/// A test double for the back end: it may answer a job (`Some` short-circuits
/// the real one), and it notes the lifecycle calls, which it takes in place of the
/// real back end's (its display thread is not started). Only tests install one.
#[cfg(test)]
pub(super) struct TestHook {
    pub job: Box<dyn FnMut(&mut BackPort<'_>, &Job<'_>) -> Option<Reply> + Send>,
    pub events: Arc<parking_lot::Mutex<Vec<&'static str>>>,
    pub on_attach: Option<Box<dyn FnMut(DisplayHandle) + Send>>,
}

/// A hook that answers every job as nothing, as a board with no drawing does:
/// retrace requests complete at once, DMA rows are unclaimed.
#[cfg(test)]
pub(super) fn null_hook() -> TestHook {
    TestHook {
        job: Box::new(|port, job| {
            Some(match job {
                Job::Retrace { tag } => {
                    port.retrace_done(*tag);
                    Reply::Done
                }
                Job::DmaIn { .. } => Reply::Unclaimed,
                _ => Reply::Done,
            })
        }),
        events: Arc::default(),
        on_attach: None,
    }
}

/// The InfiniteReality back end.
pub struct KonaBackend {
    cfg: KonaConfig,
    counts: [u64; JobKind::ALL.len()],
    resets: u64,
    /// Held between `attach` and `stop`, for the display thread once there is
    /// one.
    handle: Option<DisplayHandle>,
    /// What the display thread completes, and the thread while the board runs.
    clock: Arc<Clock>,
    thread: Option<DisplayThread>,
    /// The window the board shows the guest, which the machine keeps a clone
    /// of (`KonaBackend::display`).
    display: Arc<Display>,
    /// The textport's colour map, copied by the display once a frame.
    map: Arc<parking_lot::Mutex<ColourMap>>,
    /// The X server's colour map and display-ID table, which the display
    /// reads a frame at a time.
    tables: Arc<parking_lot::Mutex<Tables>>,
    store: Arc<PixelStore>,
    textport: Textport,
    /// The X path's state, one copy per rendering context.
    contexts: Contexts,
    regions: RegionCounts,
    /// DAC cursor loads and refusals, as counts. The cursors themselves are in the display tables.
    dac_cursor_loads: u64,
    dac_cursor_refused: u64,
    /// The display generator's channels, which the video commands set.
    channels: display::channels::Channels,
    /// What the texture commands did.
    tex_counts: gl::texture::TexCounts,
    pix_counts: gl::pixels::Counts,
    query_counts: gl::query::Counts,
    /// Test hooks: a back end test double that may answer a job, and the census tap.
    #[cfg(test)]
    pub(super) hook: Option<TestHook>,
    #[cfg(test)]
    pub(super) tap: Option<(Arc<parking_lot::Mutex<census::Census>>, Arc<std::sync::atomic::AtomicUsize>)>,
    /// The GL half of the token layer: assembly scratch and counts, and where its vertices go.
    gl: gl::GlPath,
    /// What the GL path made in one job, until set-up takes it.
    prims: raster::Prims,
    /// The tile scheduler, made when the first raster job comes so a board that
    /// never draws GL has no worker threads, and what set-up counted.
    tiles: Option<Tiles>,
    raster: RasterCounts,
    /// Buffer-configuration commands that got a layout, and those refused by
    /// reason (indexed by `Refusal`).
    layouts: u64,
    refused: [u64; Refusal::ALL.len()],
    /// Buffer-configuration commands that were not ten words, and whether that
    /// has been logged.
    malformed: u64,
    malformed_logged: bool,
    /// The layout of the last buffer configuration that got one.
    last_layout: Option<Layout>,
    /// A layout `buffer_config` made, for the job to hand the current context:
    /// its level and whether the window is double buffered.
    pending_config: Option<(i32, bool, Layout, u32)>,
}

/// Where on the return page a video Get's reply goes.
const VIDEO_REPLY_OFFSET: u32 = 8;

/// Where on the return page a pixel read's bytes go: after the ready word and
/// the word the reader skips.
const PIXEL_REPLY_OFFSET: u32 = 8;

/// Write a finished read to its host buffer a line at a time, after checking
/// that every line has somewhere to go, so a read is written whole or not at
/// all. False if a line is unmapped or fails on the bus.
fn write_lines(port: &mut BackPort<'_>, done: &x::pixel::FeedDone) -> bool {
    let lines = || done.data.chunks(done.line).enumerate().map(|(i, line)| (done.io + i as u64 * done.pitch, line));
    lines().all(|(at, line)| port.device_mapped(at, line.len())) && lines().all(|(at, line)| port.device_write(at, line))
}

impl KonaBackend {
    pub fn new(cfg: KonaConfig) -> Self {
        let map = Arc::new(parking_lot::Mutex::new(ColourMap::default()));
        let store = Arc::new(PixelStore::with_texture(cfg.managed.0, cfg.managed.1, TextureStore::with_capacity_bytes(cfg.texture_bytes_per_rm())));
        let tables = Arc::<parking_lot::Mutex<Tables>>::default();
        let display = Arc::new(Display::detached());
        display.set_tables(tables.clone());
        KonaBackend {
            store: store.clone(),
            textport: Textport::with_map(map.clone()),
            map,
            tables,
            display,
            contexts: Contexts::default(),
            regions: RegionCounts::default(),
            dac_cursor_loads: 0,
            dac_cursor_refused: 0,
            channels: display::channels::Channels::new(cfg.rm_count),
            tex_counts: Default::default(),
            pix_counts: Default::default(),
            query_counts: Default::default(),
            #[cfg(test)]
            hook: None,
            #[cfg(test)]
            tap: None,
            gl: gl::GlPath::new(cfg.managed),
            prims: raster::Prims::default(),
            tiles: None,
            raster: RasterCounts::default(),
            cfg,
            counts: [0; JobKind::ALL.len()],
            resets: 0,
            handle: None,
            clock: Arc::default(),
            thread: None,
            layouts: 0,
            refused: [0; Refusal::ALL.len()],
            malformed: 0,
            malformed_logged: false,
            last_layout: None,
            pending_config: None,
        }
    }

    /// A buffer configuration: lay its planes out in a pixel's words, within
    /// the host's budget, or count why not. The token layer will keep the
    /// layout per context; until then the last one is shown by `kona status`.
    fn buffer_config(&mut self, args: &[u32]) {
        let Ok(words) = <&[u32; BufferConfig::WORDS]>::try_from(args) else {
            self.malformed += 1;
            if !std::mem::replace(&mut self.malformed_logged, true) {
                crate::dlog!(crate::devlog::LogModule::Kona, "kona: buffer configuration of {} words, not {}; refused", args.len(), BufferConfig::WORDS);
            }
            return;
        };
        let config = BufferConfig::from_words(words);
        match Layout::new(&config, 0, self.cfg.record_words()) {
            Ok(l) => {
                self.layouts += 1;
                self.pending_config = Some((config.level, config.double_buffer, l.clone(), config.samples));
                self.last_layout = Some(l);
            }
            Err(why) => self.refused[why as usize] += 1,
        }
    }

    /// The window this board shows the guest, to hand to the machine so
    /// `get_display` can return it.
    pub fn display(&self) -> Arc<Display> {
        self.display.clone()
    }

    /// Parse the small state over throwaway objects, to see that it parses, changing nothing.
    pub fn check_small<S: stream::Stream>(s: &mut S) {
        Tables::default().visit(s);
        Textport::default().visit(s);
        Contexts::default().visit(s);
        display::channels::Channels::new(0).visit(s);
    }

    /// Tell the display whether channel 0, the frame it shows, is blanked: by its own blanking or by
    /// a temporary blank of every channel.
    fn sync_blank(&self) {
        self.tables.lock().set_blank(self.channels.blanked || self.channels.channel[0].blanked);
    }

    /// The small state the back end holds, as the stream module reads and writes it: the
    /// display tables, the textport and every context. Refused on save when a context is
    /// part way through a primitive or a pixel transfer. The counters, the tile scheduler and
    /// what a frame is composed from are not saved.
    pub fn visit_small<S: stream::Stream>(&mut self, s: &mut S) -> Result<(), String> {
        if !s.loading() {
            self.drain();
            if self.contexts.busy() {
                return Err("a context is in the middle of a primitive or a pixel transfer".into());
            }
        }
        self.tables.lock().visit(s);
        self.textport.visit(s);
        self.contexts.visit(s);
        self.channels.visit(s);
        self.sync_blank();
        if s.loading() {
            self.drain();
            self.last_layout = None;
            self.pending_config = None;
        }
        Ok(())
    }

    /// The bulk state as chunks (see `snapshot`), after every queued raster job has run.
    pub fn save_chunks(&mut self, cs: &crate::chunk_store::ChunkStore) -> std::io::Result<Vec<snapshot::KonaChunk>> {
        self.drain();
        snapshot::save(&self.store, &self.tables.lock(), cs)
    }

    /// Read and check the bulk state from chunks, changing nothing.
    pub fn prepare_chunks(&mut self, cs: &crate::chunk_store::ChunkStore, list: &[snapshot::KonaChunk]) -> Result<snapshot::Prepared, String> {
        self.drain();
        snapshot::prepare(&self.store, &self.tables.lock(), cs, list)
    }

    /// Put what `prepare_chunks` read into the store.
    pub fn apply_chunks(&mut self, p: snapshot::Prepared) -> Result<(), String> {
        self.drain();
        let applied = snapshot::apply(&self.store, &mut self.tables.lock(), p);
        self.store.touch();
        applied
    }

    /// Restore the bulk state from chunks.
    pub fn load_chunks(&mut self, cs: &crate::chunk_store::ChunkStore, list: &[snapshot::KonaChunk]) -> Result<(), String> {
        let p = self.prepare_chunks(cs, list)?;
        self.apply_chunks(p)
    }

    /// Wire the machine's status-bar inputs into the display.
    pub fn set_status(&self, status: Arc<crate::disp::StatusSource>) {
        // The same window, whether the machine took `display()` before or after.
        self.display.set_status(status);
    }

    /// A reader of the store that needs no access to the back end's state.
    pub(super) fn pixels(&self) -> display::Pixels {
        display::Pixels::new(self.store.clone(), self.map.clone(), self.tables.clone())
    }

    pub fn config(&self) -> &KonaConfig {
        &self.cfg
    }

    fn status(&self, w: &mut dyn Write) -> io::Result<()> {
        let c = &self.cfg;
        writeln!(w, "product           {}", c.spec().label)?;
        writeln!(w, "raster managers   {}", c.rm_count)?;
        writeln!(w, "texture per RM    {} MiB", c.texture_bytes_per_rm() >> 20)?;
        writeln!(w, "managed area      {} x {}", c.managed.0, c.managed.1)?;
        writeln!(w, "record bound      {} words per pixel", c.record_words())?;
        writeln!(w, "display handle    {}", if self.handle.is_some() { "attached" } else { "detached" })?;
        writeln!(w, "display           {} frames, {} retraces completed, thread {}", self.clock.frames(), self.clock.completed(), if self.thread.is_some() { "running" } else { "stopped" })?;
        writeln!(w, "pixel store       {} word arrays of {} x {}", self.store.allocated().len(), self.store.width(), self.store.height())?;
        writeln!(w, "textport          colour {} raster {:?}, {} drawn, {} unmodelled", self.textport.colour(), self.textport.raster(), self.textport.drawn, self.textport.unmodelled)?;
        let x = &self.contexts.current().x;
        let gc = match x.gc() {
            Ok(g) => format!("{:?} fg {:#x} bg {:#x} mask {:#x} gx {}", g.target, g.foreground, g.background, g.mask, g.gx),
            Err(why) => format!("not decoded ({why:?})"),
        };
        writeln!(w, "X contexts        {} loaded; the last job's:", self.contexts.held())?;
        writeln!(w, "X gc              {gc}")?;
        writeln!(
            w,
            "X state           origin {:?}, {} clip rects ({} refused), scissor {:?}, {} runs dropped",
            x.origin(),
            x.clip_rects().len(),
            x.clips_refused,
            x.scissor(),
            x.runs_dropped
        )?;
        writeln!(w, "X stipple         {}", if x.stipple.is_loaded() { "loaded" } else { "none loaded" })?;
        let f = &x.fills;
        writeln!(w, "X fills           {} drawn, refused: {} context, {} target, {} destination", f.drawn, f.bad_context, f.other_target, f.bad_destination)?;
        let p = x.pixel_counts();
        writeln!(
            w,
            "X pixels          {} setups, {} writes ({} by DMA), {} copies, {} reads, {} finishes",
            p.setups, p.writes, p.dma_writes, p.copies, p.reads, p.finishes
        )?;
        writeln!(
            w,
            "  refused         {} context, {} target, {} destination, {} window copies, {} reads answered with zeros ({} frame-buffer word)",
            p.bad_context, p.other_target, p.bad_destination, p.window_copies, p.reads_refused, p.fb_word_reads
        )?;
        writeln!(w, "  rows            {} short or left owing, {} with no write waiting", p.short, p.unexpected)?;
        writeln!(w, "  read brackets   {} refused, {} with no completion posted (host buffer unwritable or no reason page)", p.bracket_refused, p.bracket_unposted)?;
        let r = &self.regions;
        writeln!(
            w,
            "memory regions    {} placed, {} copied down, {} swapped out ({} bytes), {} swapped in ({} bytes)",
            r.placed, r.copy_downs, r.swapped_out, r.bytes_out, r.swapped_in, r.bytes_in
        )?;
        writeln!(w, "display tables    {}", self.tables.lock().describe())?;
        let on: Vec<usize> = (0..display::channels::CHANNELS).filter(|&c| self.channels.channel[c].enabled).collect();
        let blanked: Vec<usize> = (0..display::channels::CHANNELS).filter(|&c| self.channels.channel[c].blanked).collect();
        writeln!(w, "video channels   enabled {:?}, blanked {:?} (channel 0's, and all while temporarily blanked, are drawn black), frame {} x {}{}", on, blanked, self.channels.frame.0, self.channels.frame.1, if self.channels.blanked { ", all blanked" } else { "" })?;
        let tables = self.tables.lock();
        let dac = tables.dac();
        writeln!(w, "DAC cursor        {} loads, {} refused; held for channels {:?}; channel 0's is {}", self.dac_cursor_loads, self.dac_cursor_refused, dac.cursors().map(|c| c.channel).collect::<Vec<_>>(), if tables.snapshot().has_dac_cursor() { "drawn" } else { "not drawn" })?;
        writeln!(w, "  the kernel's control value for the owning channel: {}", crate::dev::kona::defs::dac_cursor_control::describe(crate::dev::kona::defs::dac_cursor_control::KERNEL_OWNER))?;
        for c in dac.cursors() {
            writeln!(w, "  {}", c.describe())?;
        }
        drop(tables);
        let c = &self.tex_counts;
        let t = self.store.texture();
        writeln!(
            w,
            "textures          {} level loads ({} by DMA, {} refused, {} past memory), {} finishes, {} binds ({} refused), {} tokens not modelled; {} volume slice loads, {} volume binds; texture memory {} KiB held (peak {} KiB, ceiling {} KiB)",
            c.loads, c.dma_loads, c.unsupported_load, c.full, c.finishes, c.binds, c.bind_refused, c.unmodelled, c.volume_loads, c.volume_binds, t.resident_pages() * (PAGE_WORDS / 256), t.peak_pages() * (PAGE_WORDS / 256), t.cap_pages() * (PAGE_WORDS / 256)
        )?;
        writeln!(w, "detail            {} binds, {} filter tables; mode-1 base binds: {} forced linear (no matching detail bind), {} with a detail image found, {} without", c.detail_binds, c.filter_tables, c.detail_mismatched, c.detail_resolved, c.detail_unresolved)?;
        let g = &self.gl.counts;
        let t = self.tiles.as_ref().map(|t| t.stats()).unwrap_or_default();
        writeln!(
            w,
            "GL path           {} commands, {} vertices in {} primitives, {} draws handed over; {} clipped ({} away), {} culled, {} with no window position, {} errors",
            g.commands, g.vertices, g.primitives, self.raster.draws, g.clipped, g.clipped_away, g.culled, g.no_window_position, g.errors
        )?;
        let r = &self.raster;
        writeln!(
            w,
            "raster stage      {} triangles ({} with no area, {} with no sample, {} off the target), {} clears; {} points and {} lines handed over as triangles ({} of no length); {} workers, {}-pixel tiles; {} drains, {} with work queued{}",
            r.triangles, r.no_area, r.no_sample, r.off_target, r.clears, g.points, g.lines, g.empty_lines, self.cfg.threads, self.cfg.tile, t.drains, t.busy_drains, if self.is_poisoned() { ", POISONED" } else { "" }
        )?;
        if let Some(t) = &self.tiles {
            let s = t.stats();
            writeln!(w, "  scheduler       {} jobs, {} tile runs, at most {} pixels in flight, producer waited {} times", s.submitted, s.tile_runs, s.peak_cost, s.waits)?;
        }
        if !g.unmodelled.is_empty() || !g.caps.is_empty() {
            let list: Vec<String> = g.unmodelled.iter().map(|(s, n)| format!("{s:#x} x {n}")).chain(g.caps.iter().map(|(c, n)| format!("enable/disable of {c:#x} x {n}"))).collect();
            writeln!(w, "  not taken       {}", list.join(", "))?;
        }
        writeln!(w, "layouts           {} made", self.layouts)?;
        if self.malformed != 0 {
            writeln!(w, "  refused         {} x not {} words", self.malformed, BufferConfig::WORDS)?;
        }
        for r in Refusal::ALL {
            let n = self.refused[r as usize];
            if n != 0 {
                writeln!(w, "  refused         {n} x {r:?}")?;
            }
        }
        if let Some(l) = &self.last_layout {
            writeln!(w, "  last            colour {:?} depth {:?} stencil {:?} multisample {:?} accumulation {:?} ({} words)", l.colour, l.depth, l.stencil, l.multisample, l.accumulation, l.words())?;
        }
        writeln!(w, "resets            {}", self.resets)
    }

    /// What the GL path counted: the census of vertices, assembly, the
    /// texture-coordinate forms and what it did not take.
    fn gl_census(&self, w: &mut dyn Write) -> io::Result<()> {
        let g = &self.gl.counts;
        writeln!(w, "commands          {}", g.commands)?;
        writeln!(w, "begins by mode    {:?} (points, lines, loop, strip, triangles, tri strip, fan, quads, quad strip, polygon)", g.begins)?;
        writeln!(w, "vertices          {} in {} primitives; {} outside a begin", g.vertices, g.primitives, g.stray)?;
        #[cfg(test)]
        writeln!(w, "vertex census     {} inside the window, {} outside it, {} outside the volume, {} behind the eye", g.inside, g.outside_window, g.outside_volume, g.behind)?;
        writeln!(w, "texcoord forms    1f {}, 2f {}, 2s {}, 3f {}, 4f {}", g.texcoord[0], g.texcoord[1], g.texcoord[2], g.texcoord[3], g.texcoord[4])?;
        writeln!(w, "clipping          {} clipped ({} away), {} culled, {} with no window position", g.clipped, g.clipped_away, g.culled, g.no_window_position)?;
        writeln!(w, "handed over       {} draws, {} triangles", self.raster.draws, self.raster.triangles)?;
        let r = &self.raster;
        writeln!(w, "raster stage      {} triangles ({} no area, {} no sample, {} off the target), {} clears; {} points and {} lines", r.triangles, r.no_area, r.no_sample, r.off_target, r.clears, g.points, g.lines)?;
        writeln!(w, "multisample       {} draws ({} with alpha to mask, {} with a sample mask), by the model's own rules", r.multisampled, r.alpha_to_mask, r.sample_mask)?;
        let c = &self.tex_counts;
        writeln!(w, "textures          {} level loads ({} by DMA, {} detail image loads, {} volume slice loads, {} refused, {} past memory), {} finishes, {} binds ({} of volumes, {} refused), {} tokens not modelled", c.loads, c.dma_loads, c.detail_loads, c.volume_loads, c.unsupported_load, c.full, c.finishes, c.binds, c.volume_binds, c.bind_refused, c.unmodelled)?;
        let t = self.store.texture();
        writeln!(w, "texture memory   {} KiB held in {} pages, peak {} KiB, ceiling {} KiB", t.resident_pages() * (PAGE_WORDS / 256), t.resident_pages(), t.peak_pages() * (PAGE_WORDS / 256), t.cap_pages() * (PAGE_WORDS / 256))?;
        writeln!(w, "texel read-backs  {} ({} in a bracket); answered with zeros: {} for a format or type the packer does not know, {} for a level not loaded; {} too big; {} with nowhere to answer", c.reads, c.reads_bracketed, c.read_unsupported, c.read_missing, c.read_oversize, c.read_unposted)?;
        if let Some(l) = &self.last_layout {
            writeln!(w, "last layout       colour {:?} depth {:?} stencil {:?} multisample {:?} accumulation {:?}", l.colour, l.depth, l.stencil, l.multisample, l.accumulation)?;
        }
        writeln!(w, "pixel operations  {}", self.pix_counts.render())?;
        writeln!(w, "queries           {}", self.query_counts.render())?;
        for (s, n) in &g.unmodelled {
            writeln!(w, "not taken         {s:#x} x {n}")?;
        }
        for (c, n) in &g.caps {
            writeln!(w, "capability        {c:#x} (enable or disable) not read x {n}")?;
        }
        Ok(())
    }

    fn counters(&self, w: &mut dyn Write) -> io::Result<()> {
        for k in JobKind::ALL {
            writeln!(w, "{:18}{}", k.name(), self.counts[k as usize])?;
        }
        Ok(())
    }
}

/// What the front end calls on the back end: a job, and the board's lifecycle.
impl KonaBackend {
    pub fn job(&mut self, port: &mut BackPort<'_>, job: Job<'_>) -> Reply {
        #[cfg(test)]
        {
            if let Some((census, at)) = &self.tap {
                census.lock().tally(&job, at.load(std::sync::atomic::Ordering::Relaxed));
            }
            if let Some(hook) = &mut self.hook {
                if let Some(reply) = (hook.job)(port, &job) {
                    return reply;
                }
            }
        }
        // A retrace changes no pixel; every other job may. The flag goes up after
        // the job's own writes (the tiles raise it for theirs), so a frame the
        // display started meanwhile is followed by another.
        let retrace = matches!(job, Job::Retrace { .. });
        let reply = self.handle_job(port, job);
        if !retrace {
            self.store.touch();
        }
        reply
    }

    /// The board is starting: spawn the display thread, which completes retrace
    /// requests and raises the video interrupt through `handle`.
    pub fn attach(&mut self, handle: DisplayHandle) {
        #[cfg(test)]
        if let Some(hook) = &mut self.hook {
            hook.events.lock().push("attach");
            if let Some(f) = &mut hook.on_attach {
                f(handle);
            }
            return;
        }
        let sink = Compositor::new(self.store.clone(), textport::COLOUR_WORD, self.map.clone(), self.tables.clone(), self.display.clone());
        self.thread = Some(DisplayThread::spawn(self.clock.clone(), handle.clone(), FRAME_PERIOD, sink));
        self.handle = Some(handle);
    }

    pub fn reset(&mut self) {
        #[cfg(test)]
        if let Some(hook) = &self.hook {
            hook.events.lock().push("reset");
            return;
        }
        self.resets += 1;
        self.clock.cancel();
        // Jobs queued before the reset finish first: they write the store it clears.
        self.drain();
        // SAFETY: the tiles are drained, and nothing draws until the reset returns.
        unsafe { self.store.reset() };
        self.textport.reset();
        self.tables.lock().reset();
        self.contexts.reset();
        self.regions = RegionCounts::default();
        self.dac_cursor_loads = 0;
        self.dac_cursor_refused = 0;
        self.channels.reset();
        self.sync_blank();
        self.tex_counts = Default::default();
        self.pix_counts = Default::default();
        self.query_counts = Default::default();
        #[cfg(test)]
        let census = self.gl.census;
        self.gl = gl::GlPath::new(self.cfg.managed);
        #[cfg(test)]
        {
            self.gl.census = census;
        }
        self.raster = Default::default();
        self.last_layout = None;
        self.pending_config = None;
    }

    pub fn stop(&mut self) {
        #[cfg(test)]
        if let Some(hook) = &self.hook {
            hook.events.lock().push("stop");
            return;
        }
        // The tile workers run what is queued, then join; the display thread joins after.
        self.tiles = None;
        // As VPro's stop does: a stop from the main thread must leave the
        // renderer to the main thread (the macOS hang `DisplayStop` exists for),
        // so the request is made before the display thread, which releases the
        // renderer as its last act, is joined.
        self.display.display_stop().request();
        // The display thread joins first: it may be inside a handle call.
        self.thread = None;
        self.handle = None;
        self.clock.cancel();
    }

    pub fn quiesce(&mut self) {
        #[cfg(test)]
        if let Some(hook) = &self.hook {
            hook.events.lock().push("quiesce");
            return;
        }
        self.drain();
        self.clock.quiesce(self.thread.is_some());
    }

    pub fn resume(&mut self) {
        #[cfg(test)]
        if let Some(hook) = &self.hook {
            hook.events.lock().push("resume");
            return;
        }
        self.clock.resume();
    }

    pub fn monitor(&mut self, args: &[&str], w: &mut dyn Write) -> io::Result<bool> {
        match args {
            ["status"] => self.status(w)?,
            ["counters"] => self.counters(w)?,
            _ if self.pixels().read(args, w)? => {}
            ["gl"] => self.gl_census(w)?,
            #[cfg(test)]
            ["gl", "census"] => {
                self.gl.census = true;
                writeln!(w, "kona: GL census on: every vertex is also measured against the window")?;
            }
            ["gl", "clear"] => {
                self.gl.counts = Default::default();
                self.raster = Default::default();
            }
            ["help"] | [] => writeln!(w, "kona: status|counters|hash|pix X Y|shot FILE.png|gl [clear]")?,
            _ => return Ok(false),
        }
        Ok(true)
    }
}

/// Whether `job` reads or writes the pixel store without taking a surface, and
/// so has to wait for the queued raster jobs itself: a swap, a retrace, a GL
/// finish and the region moves. Anything that draws or reads pixels gets its
/// `Surface` from `raster::surface`, which waits.
fn drains_first(job: &Job<'_>) -> bool {
    match job {
        Job::Command { grammar: Grammar::Gl, selector, .. } => *selector == FINISH_DRAIN_PIPE_AND_SIGNAL_COMPLETION,
        Job::Retrace { .. } => true,
        Job::Command { grammar: Grammar::Kernel, selector, args, .. } => {
            moves_pixels(*selector)
                // A swap, the display-mode token with a window.
                || (*selector == DISPLAY_MODE_BYTE && args.first().is_some_and(|&window| window != 0))
        }
        _ => false,
    }
}

/// Exclusive use of the texture store, taken after the tile workers have finished. The workers read
/// the planes without a lock, so a write may happen only while none runs; every writer in the
/// back end gets its `Exclusive` here, which waits for them first, so a write that skips the wait
/// cannot be written down. Nothing draws until the caller returns: the producer is this thread.
fn texture_exclusive<'a>(tiles: &mut Option<Tiles>, store: &'a PixelStore) -> Exclusive<'a> {
    if let Some(t) = tiles {
        let _ = t.drain();
    }
    // SAFETY: the tiles are drained, and the producer submits nothing until the caller is done.
    unsafe { store.texture().exclusive() }
}

/// What set-up counted, over the draws handed to it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RasterCounts {
    pub draws: u64,
    pub triangles: u64,
    pub no_area: u64,
    /// Triangles with area that no pixel's sample falls inside.
    pub no_sample: u64,
    pub clears: u64,
    /// Triangles and clears that had no pixel to write: no colour or depth
    /// target, or nothing inside the clip rectangle and viewport.
    pub off_target: u64,
    /// Draws and clears in a multisampled window, and of those the draws with alpha-to-mask and with
    /// the sample mask on: each is a model choice the table does not settle (see `raster::multisample`).
    pub multisampled: u64,
    pub alpha_to_mask: u64,
    pub sample_mask: u64,
}

/// Slots in the job ring, and the budget in pixels of work in flight.
const RING: usize = 1 << 10;
const BUDGET: u64 = 1 << 17;
/// The tile size used when `[graphics] tile` is not a power of two.
const DEFAULT_TILE: u32 = 32;

impl KonaBackend {
    /// The tile scheduler, made now if it is not yet.
    fn tiles(&mut self) -> &mut Tiles {
        let (store, cfg, handle) = (&self.store, &self.cfg, &self.handle);
        self.tiles.get_or_insert_with(|| {
            // A tile size that is not a power of two is a configuration mistake: say so and use the default.
            Tiles::new(store.clone(), cfg.tile, cfg.threads, RING, BUDGET, handle.clone()).unwrap_or_else(|why| {
                crate::dlog!(crate::devlog::LogModule::Kona, "kona: tile size {} refused ({why:?}); using {DEFAULT_TILE}", cfg.tile);
                Tiles::new(store.clone(), DEFAULT_TILE, cfg.threads, RING, BUDGET, handle.clone()).expect("the default tile size is a power of two")
            })
        })
    }

    /// Wait until every raster job has run.
    fn drain(&mut self) {
        if let Some(t) = &mut self.tiles {
            let _ = t.drain();
        }
    }

    /// A worker panicked: the board is dead.
    pub(super) fn is_poisoned(&self) -> bool {
        self.tiles.as_ref().is_some_and(|t| t.is_poisoned())
    }

    /// Allocate the arrays a job writes, before a worker can see it.
    fn arrays(&self, ds: &raster::DrawState) {
        ds.colour[..ds.colours as usize].iter().for_each(|&w| self.store.ensure(w as usize));
        if ds.depth_bits != 0 {
            self.store.ensure(ds.depth_word as usize);
        }
        if ds.stencil_bits != 0 {
            self.store.ensure(ds.stencil_word as usize);
        }
        if ds.ownership.is_some() {
            self.store.ensure(store::DID_PLANE);
        }
        if let Some(ms) = &ds.samples {
            [ms.colour, ms.depth, ms.stencil].iter().for_each(|&(first, n)| (first..first + n).for_each(|w| self.store.ensure(w as usize)));
        }
    }

    /// Set up what the GL path made in `self.prims` and submit it, then empty it.
    fn flush(&mut self) {
        // The GL path sets the state with every primitive it queues and `Prims::clear` unsets it,
        // so no state means nothing is queued. This runs after every geometry command, and the
        // moves below carry the whole `Prims` out and back.
        if self.prims.state.is_none() {
            return;
        }
        let mut prims = std::mem::take(&mut self.prims);
        if let Some(ds) = prims.state.filter(|_| !prims.tris.is_empty() || !prims.lines.is_empty() || !prims.aa.is_empty() || !prims.discs.is_empty() || !prims.clears.is_empty()) {
            self.raster.draws += !(prims.tris.is_empty() && prims.lines.is_empty() && prims.aa.is_empty() && prims.discs.is_empty()) as u64;
            self.raster.triangles += (prims.tris.len() + prims.lines.len() + prims.aa.iter().map(|p| p.n as usize - 2).sum::<usize>() + 2 * prims.discs.len()) as u64;
            self.raster.clears += prims.clears.len() as u64;
            if let Some(ms) = &ds.samples {
                self.raster.multisampled += 1;
                self.raster.alpha_to_mask += ms.alpha_to_mask as u64;
                self.raster.sample_mask += (ms.cover != !0) as u64;
            }
            let drawable = !ds.clip.is_empty() && (ds.colours != 0 || (ds.depth_test && ds.depth_bits != 0) || (ds.stencil.enabled && ds.stencil_bits != 0));
            let state = self.tiles().state(ds);
            if drawable {
                self.arrays(&ds);
            }
            // A poisoned scheduler refuses a job; it stays poisoned and the board is dead,
            // so a refusal carries nothing more.
            let run = |b: &mut Self, t: &[raster::WVertex; 3], line: bool, shape: Shape<'_>| {
                if !drawable {
                    b.raster.off_target += 1;
                    return;
                }
                match raster::setup::shaped(t, &ds, state, shape) {
                    Ok(mut job) => {
                        job.line = line;
                        let _ = b.tiles().submit(RasterJob::Tri(job));
                    }
                    Err(raster::setup::Skip::NoArea) => b.raster.no_area += 1,
                    Err(raster::setup::Skip::NoSample) => b.raster.no_sample += 1,
                    Err(raster::setup::Skip::Outside) => b.raster.off_target += 1,
                }
            };
            for t in &prims.tris {
                run(self, t, false, Shape::Plain);
            }
            for t in &prims.lines {
                run(self, t, true, Shape::Plain);
            }
            // A smoothed polygon or line: its fan, each triangle weighing its pixels by the whole outline.
            for poly in &prims.aa {
                let v = &prims.aa_verts[poly.start as usize..(poly.start + poly.n) as usize];
                let Some(outline) = raster::setup::Outline::new(v, &ds) else {
                    self.raster.no_area += 1;
                    continue;
                };
                for i in 1..v.len() - 1 {
                    // The edges from the fan's apex and back to it are inner, but at its two ends.
                    let inner = [i != 1, false, i != v.len() - 2];
                    run(self, &[v[0], v[i], v[i + 1]], poly.line, Shape::Smooth { inner, outline: &outline });
                }
            }
            for d in &prims.discs {
                let shape = Shape::Disc { x: d.x, y: d.y, r: d.r };
                let q = d.quad;
                run(self, &[q[0], q[1], q[2]], false, shape);
                run(self, &[q[0], q[2], q[3]], false, shape);
            }
            for c in &prims.clears {
                let mut req = *c;
                req.colour = req.colour.filter(|_| ds.colours != 0);
                req.depth = req.depth.filter(|_| ds.depth_bits != 0);
                req.stencil = req.stencil.filter(|_| ds.stencil_bits != 0);
                if req.rect.is_empty() || (req.colour.is_none() && req.depth.is_none() && req.stencil.is_none()) {
                    self.raster.off_target += 1;
                    continue;
                }
                self.arrays(&ds);
                let _ = self.tiles().submit(RasterJob::Clear(ClearJob { req, state }));
            }
        }
        prims.clear();
        self.prims = prims;
    }

    /// A texel draw's header: the image words that follow in the stream, sized from the
    /// header and the unpack state, or the DMA packet that will carry them.
    fn texel_draw(&mut self, port: &BackPort<'_>, args: &[u32]) -> Reply {
        let tex = &mut self.contexts.select(port.context()).tex;
        let Some(h) = gl::texture::Header::parse(args) else { return Reply::Done };
        let Some(n) = h.image_words(&tex.unpack) else {
            self.tex_counts.unsupported_load += 1;
            return Reply::Done;
        };
        if h.dma {
            tex.pending = <[u32; 20]>::try_from(args).ok();
            return Reply::Done;
        }
        if n == 0 {
            let mut store = texture_exclusive(&mut self.tiles, &self.store);
            self.contexts.select(port.context()).tex.load(&mut store, args, &[], false, &mut self.tex_counts);
            return Reply::Done;
        }
        Reply::Data(n)
    }

    /// A texel draw's image words, in the stream, as bytes in the client's order.
    fn texel_words(&mut self, port: &BackPort<'_>, args: &[u32], words: &[u32]) {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_be_bytes()).collect();
        let mut store = texture_exclusive(&mut self.tiles, &self.store);
        self.contexts.select(port.context()).tex.load(&mut store, args, &bytes, false, &mut self.tex_counts);
    }

    /// A texel read-back (token 0x052): the named level's texels from the texture arena, packed in the
    /// requested host format and type by the pixel packer under the pack record, with no header. There are two routes,
    /// chosen by the client from the size of the read (established): a read of more than 0x3ff0 bytes comes inside a
    /// select-feed bracket and returns by the bracket's output DMA; a smaller one is bare and returns on the return
    /// page from offset 8, with the completion word written last. The client has no timeout on the bare form, so
    /// every read is answered. A level that is not loaded answers zeros, and so does a format or type the packer does
    /// not know (the whole reply area is zeroed, then the completion word), counted. A read too big to build, or a bare
    /// one whose image would not fit the page, is refused and counted: only the completion word is posted, or the
    /// bracket is refused. A read with no context or return page to answer on is counted.
    ///
    /// Luminance and luminance-alpha come from the sum of red, green and blue clamped at 1.0, as the pipe's colour path
    /// makes them, unless the texture holds one channel, which is read as it is (the table: stage selection established,
    /// the clamped sum inferred). Alpha is kept.
    fn texel_read(&mut self, port: &mut BackPort<'_>, args: &[u32]) {
        use crate::dev::kona::defs::{pixel::*, texture::storage_is_one_channel};
        self.tex_counts.reads += 1;
        let c = self.contexts.select(port.context());
        let header = gl::texture::ReadHeader::parse(args);
        let pack = c.gl.pix.pack;
        let bracket = c.x.bracket_open();
        // What the read answers with: the packed bytes and the row they step by, or nothing to pack.
        enum Answer {
            Image(Vec<u8>, u32),
            Zeros,
            TooBig,
        }
        let sized = header.and_then(|h| {
            let (_, stride) = gl::pixels::row_bytes(h.w as usize, h.format, h.ty, &pack)?;
            // The bytes the packer will build, which include the pack record's skips: all of them the guest's.
            let bytes = gl::pixels::packed_len(h.w as usize, h.h as usize, h.format, h.ty, 0, &pack).unwrap_or(u64::MAX);
            Some((h, stride, bytes))
        });
        let answer = match sized {
            None => {
                self.tex_counts.read_unsupported += 1;
                Answer::Zeros
            }
            Some((h, stride, bytes)) => {
                let texels = h.w as u64 * h.h as u64;
                if texels > gl::texture::MAX_READ_TEXELS || bytes > gl::texture::MAX_READ_BYTES || (!bracket && bytes > READ_REPLY_LIMIT as u64) {
                    self.tex_counts.read_oversize += 1;
                    Answer::TooBig
                } else {
                    let texels = match c.tex.read_level(self.store.texture(), &h) {
                        Some(t) => t,
                        None => {
                            self.tex_counts.read_missing += 1;
                            vec![[0.0; 4]; texels as usize]
                        }
                    };
                    let texels = match h.format {
                        FORMAT_LUMINANCE | FORMAT_LUMINANCE_ALPHA if !storage_is_one_channel(h.storage) => texels.into_iter().map(|[r, g, b, a]| [(r + g + b).min(1.0), 0.0, 0.0, a]).collect(),
                        FORMAT_LUMINANCE | FORMAT_LUMINANCE_ALPHA => texels.into_iter().map(|[r, _, _, a]| [r, 0.0, 0.0, a]).collect(),
                        _ => texels,
                    };
                    let image = gl::pixels::Image { w: h.w as usize, h: h.h as usize, plane: gl::pixels::Plane::Colour(texels) };
                    Answer::Image(gl::pixels::pack(&image, h.format, h.ty, 0, &pack).unwrap_or_default(), stride as u32)
                }
            }
        };
        if bracket {
            self.tex_counts.reads_bracketed += 1;
            match (answer, header) {
                (Answer::Image(bytes, row), Some(h)) => c.x.feed_texels(row, h.h, bytes),
                // Nothing to describe to the bracket: a row of none refuses it.
                _ => c.x.feed_texels(0, 0, Vec::new()),
            };
        } else if port.context().is_some() && port.return_page_valid() {
            match answer {
                Answer::Image(bytes, _) if bytes.len() <= READ_REPLY_LIMIT => port.post_return(8, &bytes),
                Answer::Zeros => port.post_return(8, &[0; READ_REPLY_LIMIT]),
                _ => port.post_return(8, &[]),
            };
        } else {
            self.tex_counts.read_unposted += 1;
        }
    }

    /// The texture tokens that are not an image: the load finish, binds and their records,
    /// the pixel-store parameters, and the ones nothing sends.
    fn texture_command(&mut self, port: &BackPort<'_>, selector: u32, args: &[u32]) {
        let c = self.contexts.select(port.context());
        match selector {
            TEXTURE_LOAD_BY_IDENTIFIER => self.tex_counts.finishes += 1,
            SET_TEXEL_STORE_PARAMETER | SET_PIXEL_STORE_PARAMETER_INTEGER_FORM => {
                if let [name, value] = *args {
                    c.tex.unpack.set(name, value);
                }
            }
            // The linear form (0x106) is the same five words; the record follows it as it does the other.
            BIND_TEXTURE_CURRENT_FORM | BIND_LINEAR_TEXTURE => c.tex.short_bind(args),
            BIND_2D_MIP_MAPPED_TEXTURE => {
                self.tex_counts.binds += 1;
                // A bind decodes a level loaded raw in place, which writes texture memory the tile workers
                // read; `record` asks for the store only then, so a bind of decoded levels waits for nothing.
                let (tiles, store) = (&mut self.tiles, &self.store);
                match c.tex.record(|| texture_exclusive(tiles, store), args, Default::default(), &mut self.tex_counts) {
                    Some(d) => c.gl.set_texture(Some(d)),
                    None => {
                        self.tex_counts.bind_refused += 1;
                        c.gl.set_texture(None);
                    }
                }
            }
            BIND_DETAIL_TEXTURE => {
                self.tex_counts.detail_binds += 1;
                if !c.tex.bind_detail(args) {
                    self.tex_counts.unmodelled += 1;
                }
            }
            BIND_3D_MIP_MAPPED_TEXTURE | BIND_3D_TEXTURE => {
                self.tex_counts.binds += 1;
                self.tex_counts.volume_binds += 1;
                // As for the 2D bind, the store is asked for only when a level is raw.
                let (tiles, store) = (&mut self.tiles, &self.store);
                match c.tex.volume_record(|| texture_exclusive(tiles, store), args, selector == BIND_3D_MIP_MAPPED_TEXTURE, Default::default()) {
                    Some(d) => c.gl.set_texture(Some(d)),
                    None => {
                        self.tex_counts.bind_refused += 1;
                        c.gl.set_texture(None);
                    }
                }
            }
            // Naming the bound texture by number, and the cache entry's bind and free, change
            // nothing the model keeps: the record's table found the image.
            BIND_TEXTURE_BY_IDENTIFIER | BIND_ARM_TEXTURE_CACHE_ENTRY | FREE_ARM_TEXTURE_CACHE_ENTRY => {}
            // Detail, the load set-up and the old-style bind: no
            // recording draws from them (or no software sends them); counted.
            _ => self.tex_counts.unmodelled += 1,
        }
    }

    /// Make the arrays a pixel operation reads or writes, and the draw state it runs under.
    fn pixel_state(&mut self, port: &BackPort<'_>) -> raster::DrawState {
        let screen = self.gl.screen();
        let c = self.contexts.select(port.context());
        let ds = c.gl.draw_state(screen);
        if let Some(l) = c.gl.layout.clone() {
            l.colour.chain(l.accumulation).for_each(|w| self.store.ensure(w as usize));
        }
        self.arrays(&ds);
        ds
    }

    /// glClear with the accumulation bit: the accumulation words to the clear value.
    fn clear_accumulation(&mut self, port: &BackPort<'_>) {
        self.drain();
        let ds = self.pixel_state(port);
        let c = self.contexts.select(port.context());
        gl::pixels::clear_accumulation(&c.gl, &mut self.pix_counts, &gl::pixels::Env { store: &self.store, ds });
    }

    /// A pixel-state token, which stores into the context, or a pixel operation, which waits for
    /// the drawing queued before it and runs inline: its image words or reply follow.
    fn pixel_command(&mut self, port: &mut BackPort<'_>, selector: u32, args: &[u32]) -> Reply {
        // The texel loads read the same pixel-store token (the recordings send no texel-store one).
        if selector == SET_PIXEL_STORE_PARAMETER_INTEGER_FORM {
            self.texture_command(port, selector, args);
        }
        if gl::pixels::is_state(selector) {
            let c = self.contexts.select(port.context());
            gl::pixels::command(&mut c.gl, &mut self.pix_counts, selector, args);
            return match gl::pixels::run_words(&c.gl, selector, args) {
                Some(n) if n > 0 => Reply::Data(n),
                _ => Reply::Done,
            };
        }
        self.drain();
        if let Some(n) = gl::pixels::run_words(&self.contexts.select(port.context()).gl, selector, args) {
            let c = self.contexts.select(port.context());
            if selector == DRAW_PIXEL_RECTANGLE_FROM_HOST_DATA && gl::pixels::Header::parse(args).is_some_and(|h| h.dma) {
                c.gl.pix.pending = <[u32; 8]>::try_from(args).ok();
                return Reply::Done;
            }
            if n > 0 {
                return Reply::Data(n);
            }
            self.pixel_data(port, selector, args, &[]);
            return Reply::Done;
        }
        let ds = self.pixel_state(port);
        let c = self.contexts.select(port.context());
        match gl::pixels::plane_command(&mut c.gl, &mut self.pix_counts, &gl::pixels::Env { store: &self.store, ds }, selector, args) {
            gl::pixels::Out::Reply(bytes) if port.context().is_some() && port.return_page_valid() => {
                port.post_return(PIXEL_REPLY_OFFSET, &bytes);
            }
            _ => {}
        }
        Reply::Done
    }

    /// The words behind a pixel-state or draw token: pixel map values, a pixel draw's image, a bitmap's.
    fn pixel_data(&mut self, port: &BackPort<'_>, selector: u32, args: &[u32], words: &[u32]) {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_be_bytes()).collect();
        self.pixel_bytes(port, selector, args, &bytes, words);
    }

    fn pixel_bytes(&mut self, port: &BackPort<'_>, selector: u32, args: &[u32], bytes: &[u8], words: &[u32]) {
        use gl::pixels::{bitmap, draw, large_bitmap, load_map, Env, Header, Record};
        if matches!(selector, LOAD_PIXEL_MAP_FLOAT_VALUES | LOAD_PIXEL_MAP_UNSIGNED_32_BIT_VALUES | LOAD_PIXEL_MAP_UNSIGNED_16_BIT_VALUES) {
            let c = self.contexts.select(port.context());
            load_map(&mut c.gl, &mut self.pix_counts, selector, args, words);
            return;
        }
        self.drain();
        let ds = self.pixel_state(port);
        let c = self.contexts.select(port.context());
        let env = Env { store: &self.store, ds };
        let f = |w: u32| f32::from_bits(w);
        match selector {
            DRAW_PIXEL_RECTANGLE_FROM_HOST_DATA => {
                if let Ok(h) = <[u32; 8]>::try_from(args) {
                    draw(&mut c.gl, &mut self.pix_counts, &env, &h, bytes);
                }
            }
            DRAW_BITMAP_FROM_DISPLAY_LIST if args.len() == 8 => {
                let h = Header { w: args[0], h: args[1], format: crate::dev::kona::defs::pixel::FORMAT_COLOUR_INDEX, ty: crate::dev::kona::defs::pixel::TYPE_BITMAP, offset: 0, dma: false, row_mode: 0, column: 0 };
                let (xo, yo) = (f(args[2]), f(args[3]));
                // The fifth and sixth words are the origin plus the move.
                let image = &bytes[..bytes.len().min(args[7] as usize)];
                bitmap(&mut c.gl, &mut self.pix_counts, &env, &h, &Record { alignment: 1, ..Record::default() }, [xo, yo], [f(args[4]) - xo, f(args[5]) - yo], image);
            }
            DRAW_BITMAP => {
                if let Some(h) = large_bitmap(args) {
                    let rec = c.gl.pix.unpack_in_force();
                    bitmap(&mut c.gl, &mut self.pix_counts, &env, &h, &rec, [f(args[5]), f(args[6])], [f(args[7]), f(args[8])], bytes);
                }
            }
            _ => {}
        }
    }

    /// A DMA packet: the rows of a pixel write that announced them, read a
    /// line at a time into words (big-endian, as the programmed-I/O path
    /// sends them). A packet with no write waiting is unclaimed. A read that
    /// fails on the bus faults the packet and draws nothing.
    fn dma_rows(&mut self, port: &mut BackPort<'_>, io: u64, physical: bool, lines: u32, per_line: u32, pitch: u64) -> Reply {
        if let Some(args) = self.contexts.select(port.context()).tex.pending.take() {
            // A texel draw's image: its rows, `per_line` bytes each, exactly.
            let mut bytes = vec![0u8; (lines as u64 * per_line as u64) as usize];
            let ok = (0..lines as u64).all(|l| port.dma_read(io.wrapping_add(l.wrapping_mul(pitch)), physical, &mut bytes[(l * per_line as u64) as usize..][..per_line as usize]));
            if !ok {
                port.dma_fault();
                return Reply::Done;
            }
            self.tex_counts.dma_loads += 1;
            let mut store = texture_exclusive(&mut self.tiles, &self.store);
            self.contexts.select(port.context()).tex.load(&mut store, &args, &bytes, true, &mut self.tex_counts);
            return Reply::Done;
        }
        if let Some(args) = self.contexts.select(port.context()).gl.pix.pending.take() {
            // A pixel draw's image: its rows, `per_line` bytes each, as the stream would have carried them.
            let mut bytes = vec![0u8; (lines as u64 * per_line as u64) as usize];
            let ok = (0..lines as u64).all(|l| port.dma_read(io.wrapping_add(l.wrapping_mul(pitch)), physical, &mut bytes[(l * per_line as u64) as usize..][..per_line as usize]));
            if !ok {
                port.dma_fault();
                return Reply::Done;
            }
            self.pix_counts.dma_draws += 1;
            self.pixel_bytes(port, DRAW_PIXEL_RECTANGLE_FROM_HOST_DATA, &args, &bytes, &[]);
            return Reply::Done;
        }
        let x = &mut self.contexts.select(port.context()).x;
        let Some(owed) = x.dma_write_words() else {
            x.pixel_unexpected();
            return Reply::Unclaimed;
        };
        let surface = raster::surface(&mut self.tiles, &self.store, textport::COLOUR_WORD);
        let total = lines as u64 * per_line as u64;
        // The packet's bytes, held to what the write owes: more is dropped.
        let take = (total.min(owed as u64 * 4) as usize) / 4 * 4;
        let mut bytes = vec![0u8; take];
        let mut done = 0;
        for l in 0..lines as u64 {
            if done == take {
                break;
            }
            let n = (per_line as usize).min(take - done);
            if !port.dma_read(io.wrapping_add(l.wrapping_mul(pitch)), physical, &mut bytes[done..done + n]) {
                port.dma_fault();
                x.pixel_rows(&surface, &[]);
                return Reply::Done;
            }
            done += n;
        }
        let words: Vec<u32> = bytes.chunks_exact(4).map(|c| u32::from_be_bytes([c[0], c[1], c[2], c[3]])).collect();
        x.pixel_rows(&surface, &words);
        Reply::Done
    }

    fn handle_job(&mut self, port: &mut BackPort<'_>, job: Job<'_>) -> Reply {
        self.counts[JobKind::of(&job) as usize] += 1;
        // What reads or writes the store without a surface waits for the jobs queued before it.
        if drains_first(&job) {
            self.drain();
        }
        match job {
            // With the display thread running, a retrace completes at its next
            // frame boundary; with none (before the board starts, in a test)
            // there is no boundary, so it completes at once.
            Job::Retrace { tag } => match &self.handle {
                Some(h) if self.thread.is_some() => self.clock.request(h.epoch(), tag),
                _ => port.retrace_done(tag),
            },
            Job::DmaIn { io, physical, lines, per_line, pitch } => return self.dma_rows(port, io, physical, lines, per_line, pitch),
            Job::FirmwareStart { combination } => {
                // The store keeps the configured size; a combination that sets another is logged and the
                // configured size is what the configuration request reports.
                let frame = self.channels.start(combination);
                self.sync_blank();
                if frame != self.cfg.managed {
                    crate::dlog!(LogModule::Kona, "kona: the combination set a managed area of {}x{}; the board keeps {}x{}", frame.0, frame.1, self.cfg.managed.0, self.cfg.managed.1);
                }
                port.set_managed_size(self.cfg.managed.0, self.cfg.managed.1)
            }
            Job::Command { grammar: Grammar::Gl, selector: CONFIGURE_FRAME_BUFFER_LAYOUT, args, .. } => {
                self.buffer_config(args);
                if let Some((level, double, layout, samples)) = self.pending_config.take() {
                    self.contexts.select(port.context()).gl.configure(level, double, layout, samples);
                }
            }
            Job::Command { grammar: Grammar::Gl, selector: DRAW_TEXELS_INTO_TEXTURE_MEMORY, args, .. } => return self.texel_draw(port, args),
            Job::Data { grammar: Grammar::Gl, selector: DRAW_TEXELS_INTO_TEXTURE_MEMORY, args, words } => self.texel_words(port, args, words),
            Job::Command { grammar: Grammar::Gl, selector: QUERY_TEXTURE_IMAGE, args, .. } => self.texel_read(port, args),
            Job::Command { grammar: Grammar::Gl, selector, args, .. } if gl::query::is_query(selector) => {
                let c = self.contexts.select(port.context());
                let words = gl::query::answer(&mut c.gl, &mut self.query_counts, selector, args);
                // The reply: the count at byte 4, the words from byte 8, then the completion word.
                if port.context().is_some() && port.return_page_valid() {
                    port.post_return(4, &gl::query::reply_bytes(&words));
                } else {
                    self.query_counts.unposted += 1;
                }
            }
            // A push-attributes token is a bracketed read-back whatever its mask: the client waits for the
            // completion word after every one.
            Job::Command { grammar: Grammar::Gl, selector: PUSH_ATTRIBUTE_STACK, args, .. } => {
                let c = self.contexts.select(port.context());
                let words = gl::query::push_attributes(&mut c.gl, &mut self.query_counts, args.first().copied().unwrap_or(0));
                if port.context().is_some() && port.return_page_valid() {
                    port.post_return(4, &gl::query::reply_bytes(&words));
                } else {
                    self.query_counts.unposted += 1;
                }
            }
            // A finish drains the pipe (above), and with its flag set it runs the read-back's start and
            // finish back to back, which is the completion the client's wait polls for
            // (inferred; the flag's meaning is inferred from the sender, which always sends 1).
            Job::Command { grammar: Grammar::Gl, selector: FINISH_DRAIN_PIPE_AND_SIGNAL_COMPLETION, args, .. } if args.first().is_some_and(|&f| f != 0) => {
                if port.context().is_some() && port.return_page_valid() {
                    port.post_return(4, &gl::query::reply_bytes(&[]));
                } else {
                    self.query_counts.unposted += 1;
                }
            }
            Job::Command { grammar: Grammar::Gl, selector: POP_ATTRIBUTE_STACK, .. } => {
                let c = self.contexts.select(port.context());
                gl::query::pop_attributes(&mut c.gl, &mut self.query_counts);
            }
            // The restart (token 0x1bc) reprograms the output DMA from the saved start words while a feed is active.
            // The client sends it when it registers the instruments buffer again, which the model does not
            // keep; the select and feedback passes here have no DMA channel to reprogram, so it changes nothing.
            Job::Command { grammar: Grammar::Gl, selector: RESTART_SELECT_OR_FEEDBACK, .. } => {}
            Job::Command { grammar: Grammar::Gl, selector, args, .. } if gl::query::is_mode(selector) => {
                let c = self.contexts.select(port.context());
                gl::query::mode_command(&mut c.gl, &mut self.query_counts, selector, args);
            }
            Job::Command { grammar: Grammar::Gl, selector, args, .. } if gl::pixels::is_state(selector) || gl::pixels::is_plane(selector) => return self.pixel_command(port, selector, args),
            Job::Data { grammar: Grammar::Gl, selector, args, words } if gl::pixels::is_data(selector) => self.pixel_data(port, selector, args, words),
            Job::Data { grammar: Grammar::Gl, selector: LOAD_ONE_ENTRY_OF_ARM_TEXTURE_TABLE, args: &[slot], words } => {
                self.tex_counts.filter_tables += self.contexts.select(port.context()).tex.filter_table(slot, words) as u64;
            }
            Job::Command { grammar: Grammar::Gl, selector, args, .. } if gl::texture::is_texture(selector) => self.texture_command(port, selector, args),
            // The polygon stipple's 32 rows follow its command as a run the back end sizes.
            Job::Command { grammar: Grammar::Gl, selector: LOAD_POLYGON_STIPPLE_PATTERN, .. } => return Reply::Data(32),
            Job::Data { grammar: Grammar::Gl, selector: LOAD_POLYGON_STIPPLE_PATTERN, words, .. } => {
                if let Ok(rows) = <[u32; 32]>::try_from(words) {
                    self.contexts.select(port.context()).gl.set_polygon_stipple(rows);
                }
            }
            Job::Command { grammar: Grammar::Gl, selector, args, packed, .. } if gl::is_geometry(selector) => {
                if selector == CLEAR_SELECTED_BUFFERS && args.first().is_some_and(|m| m & crate::dev::kona::defs::gl::ACCUM_BUFFER_BIT != 0) {
                    self.clear_accumulation(port);
                }
                let c = self.contexts.select(port.context());
                self.gl.command(&mut c.gl, selector, args, packed, &mut self.prims);
                self.flush();
            }
            Job::Command { grammar: Grammar::Textport, selector, args, .. } => {
                let (tiles, store) = (&mut self.tiles, &self.store);
                self.textport.command(|| raster::surface(tiles, store, textport::COLOUR_WORD), selector, args);
            }
            Job::Command { grammar: Grammar::Kernel, selector, args, .. } => {
                self.regions.note(selector, args);
                match (selector, args) {
                    // The kernel placing the current context's window: words 0 and 1 are the
                    // origin, 4 to 7 the clip rectangle (origin and size), each a halfword;
                    // words 2 and 3 hold the window's own size, which no draw reads.
                    (WINDOW_CLIP_SETUP, a) if a.len() == 11 => {
                        let shown = self.tables.lock().bank(a[9]);
                        let (signed, size) = (|w: u32| w as u16 as i16 as i32, |w: u32| w as u16 as i32);
                        let clip = Rect { x0: signed(a[4]), y0: signed(a[5]), x1: signed(a[4]) + size(a[6]), y1: signed(a[5]) + size(a[7]) };
                        {
                            if !raster::ops::Ownership::encode(a[9] as i32).1 {
                                self.gl.counts.odd_window_ids += 1;
                            }
                            self.contexts.select(port.context()).gl.place([signed(a[0]), signed(a[1])], clip, a[9], shown, a[10] != 0)
                        }
                    }
                    (WINDOW_READ_CLIP_SET_UP, a) if a.len() == 3 => self.contexts.select(port.context()).gl.move_to(a[0] as u16 as i16 as i32, a[1] as u16 as i16 as i32),
                    // A swap: the window's id, the display id and the mode. The low bit of
                    // the display's mode byte is the back buffer being shown (the X server's
                    // case split, established); that this command's third word is that same
                    // byte is inferred from the recorded swaps, which set it to 0 and 1
                    // alternately.
                    (DISPLAY_MODE_BYTE, &[window, id, mode]) if window != 0 => {
                        self.tables.lock().load_bank(id, (mode & 1) as u8);
                        self.contexts.swapped(id, (mode & 1) as u8);
                    }
                    (LOAD_ONE_ENTRY_OF_DISPLAY_ID_TABLE, &[id, word]) => self.tables.lock().load_did(id, word),
                    (DISPLAY_MODE_BYTE, &[0, id, mode]) => self.tables.lock().load_mode(id, mode),
                    _ => {}
                }
                let c = self.contexts.select(port.context());
                let x = &mut c.x;
                if !x.command(selector, args) {
                    if x::fill::paints(selector) {
                        let surface = raster::surface(&mut self.tiles, &self.store, textport::COLOUR_WORD);
                        x.draw_command(&surface, selector, args);
                    } else if x::fill::is_draw(selector) {
                        // A data token only collects words for the paint that follows.
                        x.collect_command(selector, args);
                    } else if x::pixel::is_pixel(selector) {
                        let surface = raster::surface(&mut self.tiles, &self.store, textport::COLOUR_WORD);
                        let answered = port.context().is_some() && port.return_page_valid();
                        match x.pixel_command(&surface, selector, args, answered) {
                            x::pixel::Next::Nothing => {}
                            x::pixel::Next::Words(n) => return Reply::Data(n),
                            x::pixel::Next::Dma => {}
                            x::pixel::Next::Return(data) => {
                                if answered {
                                    port.post_return(PIXEL_REPLY_OFFSET, &data);
                                }
                            }
                        }
                    }
                }
                match selector {
                    START_SELECT_FEEDBACK_RETURN_DMA => {
                        x.feed_start(args, port.context().is_some());
                        if let (Some(&bytes), true) = (args.get(2), port.context().is_some()) {
                            c.gl.feed.start(bytes);
                        }
                    }
                    // A bracket opened in a select or feedback mode takes the pass's records; the client
                    // switches the mode before it finishes the one it leaves.
                    FINISH_SELECT_FEEDBACK_WITH_EVENT_CODE if c.gl.feed.bracket => {
                        let echo = args.get(1).copied().unwrap_or(0);
                        let (words, overflow) = c.gl.feed.take();
                        let buffer = x.feed_buffer();
                        self.query_counts.overflows += overflow as u64;
                        self.query_counts.feedback_values += words.len() as u64;
                        let mut remaining = buffer.map_or(0, |b| b.3);
                        if let Some((io, pitch, line, total)) = buffer {
                            let mut data: Vec<u8> = words.iter().flat_map(|w| w.to_be_bytes()).collect();
                            data.truncate(total);
                            let done = x::pixel::FeedDone { io, pitch, line, data, echo };
                            if write_lines(port, &done) {
                                remaining = total - done.data.len();
                            } else {
                                crate::dlog!(crate::devlog::LogModule::Kona, "kona: select or feedback buffer not written");
                            }
                        }
                        if port.reason_page_valid() {
                            port.post_reason_words(FINISH_SELECT_FEEDBACK_WITH_EVENT_CODE, &[0x100, echo, remaining as u32, if overflow { gl::query::OVERFLOW_BIT } else { 0 }]);
                        }
                    }
                    FINISH_SELECT_FEEDBACK_WITH_EVENT_CODE => {
                        if let Some(done) = x.feed_finish(args) {
                            // As the recorder's model does: a read that cannot be
                            // written, or a finish with no reason page to post to,
                            // posts no completion. It is counted and logged.
                            if port.reason_page_valid() && write_lines(port, &done) {
                                port.post_reason_words(FINISH_SELECT_FEEDBACK_WITH_EVENT_CODE, &[0x100, done.echo, 0, 0]);
                            } else {
                                x.bracket_unposted();
                                crate::dlog!(crate::devlog::LogModule::Kona, "kona: read bracket's {} bytes not written or no reason page; no completion posted", done.data.len());
                            }
                        }
                    }
                    _ => {}
                }
                // A stipple store's rows follow it as a run the back end sizes.
                if selector == x::stipple::X_STORE_STIPPLE && x.stipple.pending_rows() != 0 {
                    return Reply::Data(x.stipple.pending_rows());
                }
            }
            Job::Data { grammar: Grammar::Kernel, selector: LOAD_GAMMA_TABLE_ENTRIES, args: &[start, _], words } => self.tables.lock().load_gamma(start, words),
            Job::Data { grammar: Grammar::Kernel, selector: LOAD_COLOUR_MAP_ENTRIES, args: &[start, _], words } => self.tables.lock().load_colours(start, words),
            Job::Data { grammar: Grammar::Kernel, selector: x::stipple::X_STORE_STIPPLE, words, .. } => self.contexts.select(port.context()).x.stipple.rows(words),
            Job::Data { grammar: Grammar::Kernel, selector: x::pixel::X_PIXEL_WRITE, words, .. } => {
                let surface = raster::surface(&mut self.tiles, &self.store, textport::COLOUR_WORD);
                self.contexts.select(port.context()).x.pixel_rows(&surface, words);
            }
            Job::Data { grammar: Grammar::Kernel, selector: LOAD_CURSOR_IMAGE_BLOCK, args: &[at], words } => self.tables.lock().load_cursor(at, words),
            Job::Indirect { selector, value } => self.tables.lock().indirect(selector, value),
            Job::Video { command: VIDEO_DAC_CURSOR_LOAD, data } => {
                self.dac_cursor_loads += 1;
                match display::dac_cursor::DacCursor::from_request(data) {
                    Some(c) => self.tables.lock().load_dac_cursor(c),
                    None => self.dac_cursor_refused += 1,
                }
            }
            Job::Video { command, data } => {
                self.channels.set(command, data);
                self.sync_blank();
                // A Get's reply: the id, the data and their sum from byte 8, then the ready word.
                if let Some(reply) = self.channels.reply(command, data) {
                    port.post_return(VIDEO_REPLY_OFFSET, &reply);
                }
            }
            Job::Context(e) => {
                // A slot activated afresh starts from power-on: its levels give their texture memory back first,
                // after the workers that may be reading it are done.
                if let super::job::ContextEvent::Activated { slot, fresh: true } = e {
                    if slot < CONTEXT_SLOTS && self.contexts.select(Some(slot)).tex.holds() {
                        let mut store = texture_exclusive(&mut self.tiles, &self.store);
                        self.contexts.select(Some(slot)).tex.release(&mut store);
                    }
                }
                self.contexts.event(e)
            }
            Job::Bitmap { header, payload } => {
                let surface = raster::surface(&mut self.tiles, &self.store, textport::COLOUR_WORD);
                self.textport.bitmap(&surface, header, payload)
            }
            _ => {}
        }
        Reply::Done
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod region_tests {
    use super::*;

    #[test]
    fn region_commands_are_counted_with_their_sizes() {
        let mut r = RegionCounts::default();
        r.note(PIPE_MEMORY_RESERVE, &[2, 9, 0, 1]);
        r.note(PIPE_MEMORY_COPY_DOWN, &[2, 0, 0, 0]);
        r.note(PIPE_MEMORY_SWAP_OUT, &[3, 0, 4, 0x400, 1, 0x20, 0x9000, 0x102]);
        r.note(PIPE_MEMORY_SWAP_IN, &[3, 0x10, 2, 0x400, 1, 0x20]);
        let seen = r.clone();
        r.note(PIPE_MEMORY_RESERVE + 0x400, &[]);
        r.note(0x154, &[]);
        assert_eq!(r, seen, "a command that is no region command counts nothing");
        assert_eq!(r, RegionCounts { placed: 1, copy_downs: 1, swapped_out: 1, swapped_in: 1, bytes_out: 4096, bytes_in: 2048 });
    }
}
