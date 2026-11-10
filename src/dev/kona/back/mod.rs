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

pub mod display;
pub mod raster;
pub mod tiles;
pub mod store;
pub mod context;
pub mod gl;
pub mod textport;
pub mod x;


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
use store::texture::{TextureStore, PAGE_WORDS};
use store::PixelStore;
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
    /// Test hooks: a back end test double that may answer a job.
    #[cfg(test)]
    pub(super) hook: Option<TestHook>,
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
            #[cfg(test)]
            hook: None,
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

    /// Tell the display whether channel 0, the frame it shows, is blanked: by its own blanking or by
    /// a temporary blank of every channel.
    fn sync_blank(&self) {
        self.tables.lock().set_blank(self.channels.blanked || self.channels.channel[0].blanked);
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
        writeln!(w, "vertex census     {} inside the window, {} outside it, {} outside the volume, {} behind the eye (counted only in tests)", g.inside, g.outside_window, g.outside_volume, g.behind)?;
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
            if let Some(hook) = &mut self.hook {
                if let Some(reply) = (hook.job)(port, &job) {
                    return reply;
                }
            }
        }
        self.handle_job(port, job)
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
        self.store.reset();
        self.textport.reset();
        self.tables.lock().reset();
        self.contexts.reset();
        self.regions = RegionCounts::default();
        self.dac_cursor_loads = 0;
        self.dac_cursor_refused = 0;
        self.channels.reset();
        self.sync_blank();
        self.tex_counts = Default::default();
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
            self.drain();
            let tex = &mut self.contexts.select(port.context()).tex;
            tex.load(self.store.texture(), args, &[], false, &mut self.tex_counts);
            return Reply::Done;
        }
        Reply::Data(n)
    }

    /// A texel draw's image words, in the stream, as bytes in the client's order.
    fn texel_words(&mut self, port: &BackPort<'_>, args: &[u32], words: &[u32]) {
        self.drain();
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_be_bytes()).collect();
        self.contexts.select(port.context()).tex.load(self.store.texture(), args, &bytes, false, &mut self.tex_counts);
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
                match c.tex.record(self.store.texture(), args, Default::default(), &mut self.tex_counts) {
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
                match c.tex.volume_record(self.store.texture(), args, selector == BIND_3D_MIP_MAPPED_TEXTURE, Default::default()) {
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
            self.drain();
            self.tex_counts.dma_loads += 1;
            self.contexts.select(port.context()).tex.load(self.store.texture(), &args, &bytes, true, &mut self.tex_counts);
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
            // The texture loads read the pixel-store parameters the client sends.
            Job::Command { grammar: Grammar::Gl, selector: SET_PIXEL_STORE_PARAMETER_INTEGER_FORM, args, .. } => self.texture_command(port, SET_PIXEL_STORE_PARAMETER_INTEGER_FORM, args),
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
                        self.drain();
                        self.contexts.select(Some(slot)).tex.release(self.store.texture());
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
mod tests {
    use super::super::tests::{test_cfg, TestHost};
    use std::sync::atomic::Ordering::Relaxed;
    use super::super::Kona;
    use super::*;
    use std::sync::Arc;

    /// A monitor writer a test can read afterwards.
    #[derive(Clone, Default)]
    struct Capture(Arc<parking_lot::Mutex<Vec<u8>>>);

    impl Write for Capture {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.lock().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl Capture {
        fn text(&self) -> String {
            String::from_utf8(self.0.lock().clone()).unwrap()
        }
    }

    fn board() -> (Arc<TestHost>, Arc<Kona>) {
        let cfg = test_cfg();
        let host = TestHost::new();
        let kona = Kona::new(cfg, host.clone(), host.clone(), crate::dev::kona::tests::test_status(), None).unwrap();
        (host, kona)
    }

    fn monitor(kona: &Kona, args: &[&str]) -> (bool, String) {
        let mut out = Vec::new();
        let known = kona.with_backend(|b| b.monitor(args, &mut out)).unwrap();
        (known, String::from_utf8(out).unwrap())
    }

    #[test]
    fn the_config_follows_the_pipe() {
        assert_eq!(KonaConfig::ir2(1).unwrap().record_words(), 16);
        assert_eq!(KonaConfig::ir2(2).unwrap().record_words(), 32);
        let c = KonaConfig::default().with_pool(0, 8);
        assert_eq!((c.managed, c.threads, c.tile, c.texture_bytes_per_rm()), ((1280, 1024), 0, 8, 64 << 20));
    }

    /// The texture store holds what the product's texture board holds.
    #[test]
    fn the_texture_store_is_the_products_size() {
        use crate::dev::kona::ident::Product;
        for (product, mb) in [(Product::Ir2, 64), (Product::Ir2e, 64), (Product::Ir3, 256)] {
            let be = KonaBackend::new(KonaConfig::new(product, 1).unwrap());
            assert_eq!(be.store.texture().bank_units(), mb << 14, "{product:?}: units of 256 bits in one of two banks");
        }
    }

    #[test]
    fn status_and_counters_answer_and_other_words_do_not() {
        let (_host, kona) = board();
        let (known, text) = monitor(&kona, &["status"]);
        assert!(known);
        assert!(text.contains("raster managers   2") && text.contains("texture per RM    64 MiB"), "{text}");
        assert!(text.contains("record bound      32 words per pixel"), "{text}");
        let (known, text) = monitor(&kona, &["counters"]);
        assert!(known && text.contains("GL commands       0"), "{text}");
        assert!(!monitor(&kona, &["frob", "x.png"]).0);
    }

    /// The monitor reaches the back end through the board's own command.
    #[test]
    fn the_board_routes_the_kona_command() {
        use crate::traits::Device;
        let (_host, kona) = board();
        assert_eq!(kona.register_commands()[0].0, "kona");
        let out = Capture::default();
        kona.execute_command("kona", &["status"], Box::new(out.clone())).unwrap();
        assert!(out.text().contains("managed area      1280 x 1024"));
        assert!(kona.execute_command("kona", &["frob"], Box::new(out)).is_err());
    }

    /// Jobs are counted by kind, a retrace completes at once and the managed
    /// area is reported when the firmware starts.
    #[test]
    fn jobs_are_counted_and_retraces_complete() {
        let (_host, kona) = board();
        let w = |off: u64, v: u32| assert_eq!(kona.test_write(off, 4, v as u64), crate::traits::BUS_OK);
        w(0xc10000, 1);
        w(0xc10000, 2);
        w(0xc10000, 7);
        w(0xc10000, 0x1f1);
        w(0xc10000, 0x1000_0000);
        w(0xc10000, 0);
        w(0xc10000, 0x4000);
        w(0xc10000, 0x1e7);
        w(0xc10000, 0x42);
        let (_, text) = monitor(&kona, &["counters"]);
        assert!(text.contains("textport commands 1"), "{text}");
        assert!(text.contains("kernel commands   2"), "{text}");
        assert!(text.contains("retrace requests  1"), "{text}");
        assert_eq!(kona.bus_read(0x800248, 4), (crate::traits::BUS_OK, 0x42));
    }

    /// A word that opens no command is counted, not answered.
    #[test]
    fn dropped_words_are_counted() {
        let (_host, kona) = board();
        for w in [1, 2, 7, 0x500, 0x500, 0x7f7f] {
            assert_eq!(kona.test_write(0xc10000, 4, w), crate::traits::BUS_OK);
        }
        let (_, text) = monitor(&kona, &["counters"]);
        assert!(text.contains("dropped words     3"), "{text}");
    }

    /// Wait for `cond`, up to a deadline, not a count of tries.
    fn wait_for(mut cond: impl FnMut() -> bool) -> bool {
        let end = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while std::time::Instant::now() < end {
            if cond() {
                return true;
            }
            std::thread::yield_now();
        }
        cond()
    }

    fn backend() -> KonaBackend {
        KonaBackend::new(KonaConfig::default())
    }

    fn run(b: &mut KonaBackend, args: &[&str]) -> (bool, String) {
        let mut out = Vec::new();
        let known = b.monitor(args, &mut out).unwrap();
        (known, String::from_utf8(out).unwrap())
    }

    /// A GL buffer configuration is laid out within the budget; one the hardware
    /// documents do not cover is refused and counted; so is one of the wrong
    /// length; the monitor shows them.
    #[test]
    fn a_buffer_configuration_gets_a_layout_or_a_counted_refusal() {
        let mut b = backend();
        assert!(run(&mut b, &["status"]).1.contains("layouts           0 made"));
        b.buffer_config(&[0, 10, 0, 1, 0, 4, 24, 8, 25, 0]);
        b.buffer_config(&[0, 8, 0, 0, 0, 2, 0, 0, 0, 0]);
        b.buffer_config(&[0, 8, 0]);
        b.buffer_config(&[0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let text = run(&mut b, &["status"]).1;
        assert!(text.contains("layouts           1 made"), "{text}");
        assert!(text.contains("refused         2 x not 10 words"), "{text}");
        assert!(text.contains("refused         1 x Samples"), "{text}");
        assert!(text.contains("(13 words)"), "{text}");
    }

    /// Run one job through `Backend::job` with a front end behind it.
    fn send(b: &mut KonaBackend, job: Job<'_>) -> Reply {
        let (host, mut core) = crate::dev::kona::tests::bare(&TestHost::new(), test_cfg());
        b.job(&mut BackPort { core: &mut core, host: &host }, job)
    }

    fn command(b: &mut KonaBackend, grammar: Grammar, selector: u32, args: &[u32]) -> Reply {
        send(b, Job::Command { grammar, selector, args, packed: 0, handled: false })
    }

    /// A framed GL command reaches the layout through `Backend::job`; the
    /// same selector in the textport grammar does not.
    #[test]
    fn a_framed_gl_buffer_configuration_is_laid_out() {
        let mut b = backend();
        let args = [0, 10, 0, 1, 0, 4, 24, 8, 25, 0];
        command(&mut b, Grammar::Textport, 1, &args);
        command(&mut b, Grammar::Gl, 1, &args);
        command(&mut b, Grammar::Gl, 1, &args[..3]);
        let text = run(&mut b, &["status"]).1;
        assert!(text.contains("layouts           1 made") && text.contains("1 x not 10 words"), "{text}");
        assert!(text.contains("(13 words)"), "{text}");
    }

    /// Textport commands draw into the colour word of the store, a bitmap
    /// draws at the raster position, and the hash follows the contents.
    #[test]
    fn textport_commands_draw_into_the_store() {
        let mut b = backend();
        let empty = run(&mut b, &["hash"]).1;
        command(&mut b, Grammar::Textport, 2, &[7]);
        command(&mut b, Grammar::Textport, 5, &[0, 0, 3, 3]);
        command(&mut b, Grammar::Textport, 4, &[10, 10]);
        send(&mut b, Job::Bitmap { header: &[1, 1, 0, 0, 2, 0], payload: &[1 << 31] });
        assert_eq!(b.store.read(10, 10, 1), 7);
        assert_eq!(b.store.read(3, 3, 1), 7);
        assert_eq!(b.store.read(1, 1, 1), 7);
        assert_eq!(b.store.read(0, 0, 1), 0, "the near corner is out");
        let text = run(&mut b, &["status"]).1;
        assert!(text.contains("colour 7 raster (12, 10), 2 drawn"), "{text}");
        let drawn = run(&mut b, &["hash"]).1;
        assert_ne!(empty, drawn);
        b.reset();
        assert_ne!(run(&mut b, &["hash"]).1, drawn);
        assert_eq!(b.store.read(3, 3, 1), 0);
    }

    /// X state commands load the context, position, clip and scissor; the
    /// monitor shows what was understood.
    #[test]
    fn x_state_commands_are_taken_and_shown() {
        let mut b = backend();
        let gc = [0x5a000, 0x1ffe, 0xa5000, 0x1ffe, 0xf00fff, 0, 0x180c, 0x40001, 1, 6, 5, 1, 1];
        command(&mut b, Grammar::Kernel, WINDOW_SYSTEM_GRAPHICS_CONTEXT_LOAD, &gc);
        command(&mut b, Grammar::Kernel, WINDOW_SYSTEM_DRAWABLE_ORIGIN_AND_SIZE_LOAD, &[240, 295, 100, 50]);
        command(&mut b, Grammar::Kernel, WINDOW_SYSTEM_CLIP_LOAD, &[1, 10, 20, 110, 220]);
        command(&mut b, Grammar::Kernel, WINDOW_SYSTEM_SCISSOR_LOAD, &[10, 20, 100, 200]);
        let text = run(&mut b, &["status"]).1;
        assert!(text.contains("MainIndex16 fg 0x5a00 bg 0xa500 mask 0xffff gx 3"), "{text}");
        assert!(text.contains("origin (240, 295), 1 clip rects (0 refused), scissor Some([10, 20, 100, 200])"), "{text}");
        // The same selector numbers in another grammar are not X state.
        command(&mut b, Grammar::Gl, WINDOW_SYSTEM_GRAPHICS_CONTEXT_LOAD, &[0; 13]);
        command(&mut b, Grammar::Kernel, PIPE_SET_UP, &[0, 0, 0]);
        assert!(run(&mut b, &["status"]).1.contains("MainIndex16"));
        b.reset();
        assert!(run(&mut b, &["status"]).1.contains("not decoded"));
    }

    /// An X box through `Backend::job` draws into the colour word of the store
    /// under the loaded context, and a command that is no X token leaves a
    /// waiting run alone only if it belongs to the run.
    #[test]
    fn an_x_box_draws_into_the_store() {
        let mut b = backend();
        let gc = [0x5a000, 0x1ffe, 0xa5000, 0x1ffe, 0xf00fff, 0, 0x180c, 0x40001, 1, 6, 5, 1, 1];
        command(&mut b, Grammar::Kernel, WINDOW_SYSTEM_GRAPHICS_CONTEXT_LOAD, &gc);
        command(&mut b, Grammar::Kernel, FILL_BOXES_TWO_CORNERS, &[2 << 16 | 3, 5 << 16 | 7]);
        assert_eq!(b.store.read(2, 1023 - 3, 1), 0x5a00);
        assert_eq!(b.store.read(5, 1023 - 3, 1), 0);
        // A data token, another command, then a box: the run did not survive.
        command(&mut b, Grammar::Kernel, WINDOW_SYSTEM_2_WORD_DATA_PRELOAD, &[10 << 16 | 10, 12 << 16 | 12]);
        command(&mut b, Grammar::Kernel, PIPE_SET_UP, &[0, 0, 0]);
        command(&mut b, Grammar::Kernel, FILL_BOXES_TWO_CORNERS, &[20 << 16 | 20, 21 << 16 | 21]);
        assert_eq!(b.store.read(10, 1023 - 10, 1), 0);
        assert_eq!(b.store.read(20, 1023 - 20, 1), 0x5a00);
        let text = run(&mut b, &["status"]).1;
        assert!(text.contains("X fills           2 drawn, refused: 0 context, 0 target, 0 destination"), "{text}");
    }

    /// A stipple store asks for its rows as a data run; with the pattern
    /// stored and loaded, a box fill is masked by it.
    #[test]
    fn a_stipple_is_stored_loaded_and_masks_a_fill() {
        let mut b = backend();
        let gc = [0x5a000, 0x1ffe, 0xa5000, 0x1ffe, 0xf00fff, 0, 0x180c, 0x40001, 1, 6, 5, 1, 1];
        command(&mut b, Grammar::Kernel, WINDOW_SYSTEM_GRAPHICS_CONTEXT_LOAD, &gc);
        // A 2 x 2 checkerboard in entry 4: the reply asks for its two rows.
        assert_eq!(command(&mut b, Grammar::Kernel, STORE_WINDOW_SYSTEM_STIPPLE, &[4, 2, 2]), Reply::Data(2));
        send(&mut b, Job::Data { grammar: Grammar::Kernel, selector: STORE_WINDOW_SYSTEM_STIPPLE, args: &[4, 2, 2], words: &[0x8000_0000, 0x4000_0000] });
        assert_eq!(command(&mut b, Grammar::Kernel, LOAD_WINDOW_SYSTEM_STIPPLE, &[4]), Reply::Done);
        command(&mut b, Grammar::Kernel, FILL_BOXES_TWO_CORNERS, &[0, 4 << 16 | 4]);
        assert_eq!(b.store.read(0, 1023 - 0, 1), 0x5a00);
        assert_eq!(b.store.read(1, 1023 - 0, 1), 0);
        assert_eq!(b.store.read(1, 1023 - 1, 1), 0x5a00);
        assert!(run(&mut b, &["status"]).1.contains("X stipple         loaded"));
        command(&mut b, Grammar::Kernel, LOAD_WINDOW_SYSTEM_STIPPLE, &[u32::MAX]);
        command(&mut b, Grammar::Kernel, FILL_BOXES_TWO_CORNERS, &[10 << 16 | 10, 12 << 16 | 12]);
        assert_eq!(b.store.read(11, 1023 - 10, 1), 0x5a00, "no stipple: solid");
        // A store of no rows asks for none.
        assert_eq!(command(&mut b, Grammar::Kernel, STORE_WINDOW_SYSTEM_STIPPLE, &[4, 2, 0]), Reply::Done);
    }

    /// With the board running, a retrace request waits for the display thread's
    /// next frame boundary; the tag register then reads its tag and `status` shows
    /// the thread and its frames.
    #[test]
    fn a_retrace_completes_at_a_frame_boundary_on_a_running_board() {
        use crate::traits::Device;
        let (_host, kona) = board();
        kona.start();
        for w in [0x1e7u32, 0x77] {
            assert_eq!(kona.bus_write(0xc10000, 4, w as u64), crate::traits::BUS_OK);
        }
        wait_for(|| kona.bus_read(0x800248, 4).1 == 0x77);
        assert_eq!(kona.bus_read(0x800248, 4).1, 0x77);
        // The counter follows the completion within the same frame.
        wait_for(|| monitor(&kona, &["status"]).1.contains("1 retraces completed"));
        let text = monitor(&kona, &["status"]).1;
        assert!(text.contains("thread running"), "{text}");
        kona.stop();
        assert!(monitor(&kona, &["status"]).1.contains("thread stopped"));
    }

    /// What the pixel tests send jobs through: a front end with a context
    /// current and a return and a reason page registered, over guest RAM the
    /// test can look at.
    struct Rig {
        ram: Arc<TestHost>,
        host: super::super::front::Host,
        core: super::super::pipe::Core,
    }

    const RETURN_PAGE: u64 = 0x30_0000;
    const REASON_PAGE: u64 = 0x31_0000;

    impl Rig {
        fn new() -> Rig {
            let ram = TestHost::new();
            let (host, mut core) = crate::dev::kona::tests::bare(&ram, test_cfg());
            core.current = Some(0);
            core.return_page = Some(RETURN_PAGE);
            core.reason_page = Some(REASON_PAGE);
            Rig { ram, host, core }
        }

        fn send(&mut self, b: &mut KonaBackend, job: Job<'_>) -> Reply {
            b.job(&mut BackPort { core: &mut self.core, host: &self.host }, job)
        }

        fn x(&mut self, b: &mut KonaBackend, selector: u32, args: &[u32]) -> Reply {
            self.send(b, Job::Command { grammar: Grammar::Kernel, selector, args, packed: 0, handled: false })
        }
    }

    const INDEX_GC: [u32; 13] = [0x5a000, 0x1ffe, 0xa5000, 0x1ffe, 0xf00fff, 0, 0x180c, 0x40001, 1, 6, 5, 1, 1];

    /// A tile worker that panics (here: a job that names a plane set-up never
    /// allocated) poisons the scheduler, and the board reports itself dead.
    #[test]
    fn a_poisoned_tile_scheduler_makes_the_board_dead() {
        let kona = Kona::new(test_cfg().with_pool(2, 32), TestHost::new(), TestHost::new(), crate::dev::kona::tests::test_status(), None).unwrap();
        assert_eq!(kona.failure(), None);
        kona.with_backend(|b| {
            let full = Rect { x0: 0, y0: 0, x1: 64, y1: 64 };
            let state = b.tiles().state(raster::DrawState { clip: full, colour: [77, 0], colours: 1, ..Default::default() });
            let req = raster::ClearReq { rect: full, colour: Some(1), ..Default::default() };
            let _ = b.tiles().submit(RasterJob::Clear(ClearJob { req, state }));
            b.drain();
        });
        let why = kona.failure().expect("poisoned");
        assert!(why.contains("tile worker"), "{why}");
    }

    /// The placement's word 9 and the enable in word 10 become the draw state's
    /// ownership test: off with the enable clear; encoded as the board does when set;
    /// a word in none of the ranges is encoded as the ordinary pair and counted.
    #[test]
    fn a_placement_sets_the_ownership_test_from_words_nine_and_ten() {
        let (mut b, mut r) = (backend(), Rig::new());
        let screen = Rect { x0: 0, y0: 0, x1: 1280, y1: 1024 };
        let own = |b: &mut KonaBackend| b.contexts.select(Some(0)).gl.draw_state(screen).ownership;
        r.x(&mut b, WINDOW_CLIP_SETUP, &[0, 0, 64, 64, 0, 0, 64, 64, 0, 0xe, 0]);
        assert_eq!(own(&mut b), None, "the enable is clear");
        r.x(&mut b, WINDOW_CLIP_SETUP, &[0, 0, 64, 64, 0, 0, 64, 64, 0, 0xe, 1]);
        assert_eq!(own(&mut b).map(|o| o.register()), Some(0x7e01d));
        r.x(&mut b, WINDOW_CLIP_SETUP, &[0, 0, 64, 64, 0, 0, 64, 64, 0, 0xffff_ffff, 1]);
        assert_eq!(own(&mut b), Some(raster::ops::Ownership { reference: 0x3f, mask: 0x3f }), "all ones is the ordinary pair 0x3f, 0x3f");
        assert_eq!(b.gl.counts.odd_window_ids, 0);
        r.x(&mut b, WINDOW_CLIP_SETUP, &[0, 0, 64, 64, 0, 0, 64, 64, 0, 0x77, 1]);
        assert_eq!(b.gl.counts.odd_window_ids, 1);
    }

    /// The window placement keeps the window's origin and its clip rectangle
    /// apart: the recording's tenth placement has a clip of the whole screen
    /// and a window origin of 314, and a later move carries both.
    #[test]
    fn a_placement_keeps_the_window_origin_and_the_clip_rectangle_apart() {
        let (mut b, mut r) = (backend(), Rig::new());
        r.x(&mut b, WINDOW_CLIP_SETUP, &[314, 0, 299, 1, 0, 0, 1280, 1024, 298, 0, 0]);
        let ds = b.contexts.select(Some(0)).gl.draw_state(Rect { x0: 0, y0: 0, x1: 1280, y1: 1024 });
        assert_eq!((ds.origin, ds.clip), ([314, 0], Rect { x0: 0, y0: 0, x1: 1280, y1: 1024 }));
        // The placement before it in the recording has clip equal to window.
        r.x(&mut b, WINDOW_CLIP_SETUP, &[100, 50, 200, 100, 100, 50, 200, 100, 0, 0, 0]);
        let ds = b.contexts.select(Some(0)).gl.draw_state(Rect { x0: 0, y0: 0, x1: 1280, y1: 1024 });
        assert_eq!((ds.origin, ds.clip), ([100, 50], Rect { x0: 100, y0: 50, x1: 300, y1: 150 }));
        // Words 0 and 1 are halfwords: 0xfff0 is -16, a window partly off the left of the screen.
        r.x(&mut b, WINDOW_CLIP_SETUP, &[0xfff0, 0, 64, 64, 0xfff0, 0, 64, 64, 0, 0, 0]);
        let ds = b.contexts.select(Some(0)).gl.draw_state(Rect { x0: 0, y0: 0, x1: 1280, y1: 1024 });
        assert_eq!((ds.origin, ds.clip), ([-16, 0], Rect { x0: 0, y0: 0, x1: 48, y1: 64 }));
        r.x(&mut b, WINDOW_READ_CLIP_SET_UP, &[16, 10, 0]);
        let ds = b.contexts.select(Some(0)).gl.draw_state(Rect { x0: 0, y0: 0, x1: 1280, y1: 1024 });
        assert_eq!((ds.origin, ds.clip), ([16, 10], Rect { x0: 16, y0: 10, x1: 80, y1: 74 }), "the clip moves with the origin");
    }

    /// An image whose rows follow the command as a data run draws once the
    /// run is in, the store's top row first.
    #[test]
    fn an_inline_pixel_write_asks_for_its_rows_and_draws_them() {
        let (mut b, mut r) = (backend(), Rig::new());
        r.x(&mut b, WINDOW_SYSTEM_GRAPHICS_CONTEXT_LOAD, &INDEX_GC);
        r.x(&mut b, WINDOW_SYSTEM_PIXEL_PATH_SET_UP, &[0, 0, 1, 8, 3, 2, 0]);
        assert_eq!(r.x(&mut b, WINDOW_SYSTEM_PIXEL_WRITE, &[4, 2]), Reply::Data(2));
        r.send(&mut b, Job::Data { grammar: Grammar::Kernel, selector: WINDOW_SYSTEM_PIXEL_WRITE, args: &[4, 2], words: &[0x0102_0300, 0x0405_0600] });
        assert_eq!([b.store.read(4, 1021, 1), b.store.read(6, 1021, 1), b.store.read(5, 1020, 1)], [0x100, 0x300, 0x500]);
        assert_eq!(r.x(&mut b, WINDOW_SYSTEM_PIXEL_PATH_FINISH, &[0]), Reply::Done);
        assert!(run(&mut b, &["status"]).1.contains("1 writes (0 by DMA), 0 copies, 0 reads, 1 finishes"));
    }

    /// With the DMA bit set the command asks for nothing and the DMA packet
    /// that arrives next carries the rows, read from the guest a line at a
    /// time; one that arrives with no write waiting is unclaimed, and a read
    /// that fails faults the packet and draws nothing.
    #[test]
    fn a_dma_pixel_write_takes_the_packet_that_follows() {
        let (mut b, mut r) = (backend(), Rig::new());
        r.x(&mut b, WINDOW_SYSTEM_GRAPHICS_CONTEXT_LOAD, &INDEX_GC);
        r.x(&mut b, WINDOW_SYSTEM_PIXEL_PATH_SET_UP, &[0x10, 0, 1, 8, 4, 2, 0]);
        assert_eq!(r.x(&mut b, WINDOW_SYSTEM_PIXEL_WRITE, &[0, 0]), Reply::Done);
        // Two rows of one word each, the lines 8 bytes apart.
        r.ram.put_words(0x40_0000, &[0x0102_0304, 0xdead_beef, 0x0506_0708]);
        let dma = |io, lines, per_line, pitch| Job::DmaIn { io, physical: true, lines, per_line, pitch };
        assert_eq!(r.send(&mut b, dma(0x40_0000, 2, 4, 8)), Reply::Done);
        assert_eq!([b.store.read(0, 1023, 1), b.store.read(3, 1023, 1), b.store.read(3, 1022, 1)], [0x100, 0x400, 0x800]);
        assert_eq!(r.send(&mut b, dma(0x40_0000, 1, 4, 4)), Reply::Unclaimed, "nothing waits for it");
        assert!(!r.core.dma_fault);
        r.x(&mut b, WINDOW_SYSTEM_PIXEL_PATH_SET_UP, &[0x10, 0, 1, 8, 4, 1, 0]);
        r.x(&mut b, WINDOW_SYSTEM_PIXEL_WRITE, &[0, 4]);
        assert_eq!(r.send(&mut b, dma(0x7f_ff_ff_00, 1, 4, 4)), Reply::Done);
        assert!(r.core.dma_fault, "the bus read failed");
        assert_eq!(b.store.read(0, 1019, 1), 0);
        let text = run(&mut b, &["status"]).1;
        assert!(text.contains("1 writes (1 by DMA)") && text.contains("1 with no write waiting"), "{text}");
    }

    /// A read answers on the return page after its first word, which is set
    /// last; with no context current it is not answered.
    #[test]
    fn a_pixel_read_answers_on_the_return_page() {
        let (mut b, mut r) = (backend(), Rig::new());
        r.x(&mut b, WINDOW_SYSTEM_GRAPHICS_CONTEXT_LOAD, &INDEX_GC);
        r.x(&mut b, FILL_BOXES_TWO_CORNERS, &[2 << 16, 4 << 16 | 1]);
        assert_eq!(b.store.read(3, 1023, 1), 0x5a00);
        r.x(&mut b, WINDOW_SYSTEM_PIXEL_PATH_SET_UP, &[2, 1, 0, 8, 3, 1, 0]);
        r.x(&mut b, WINDOW_SYSTEM_PIXEL_READ, &[2, 0]);
        assert_eq!(r.ram.word(RETURN_PAGE), 1);
        assert_eq!(r.ram.bytes(RETURN_PAGE + 8, 4), [0x5a, 0x5a, 0, 0], "pixels 2 and 3 are filled, 4 is not");
        r.ram.put(RETURN_PAGE, &[0; 16]);
        r.core.current = None;
        r.x(&mut b, WINDOW_SYSTEM_PIXEL_READ, &[2, 0]);
        assert_eq!(r.ram.word(RETURN_PAGE), 0, "no context, no answer");
    }

    /// A copy under the context moves pixels the same way a write puts them.
    #[test]
    fn a_pixel_copy_moves_pixels_inside_the_store() {
        let (mut b, mut r) = (backend(), Rig::new());
        r.x(&mut b, WINDOW_SYSTEM_GRAPHICS_CONTEXT_LOAD, &INDEX_GC);
        r.x(&mut b, FILL_BOXES_TWO_CORNERS, &[0, 2 << 16 | 1]);
        r.x(&mut b, WINDOW_SYSTEM_PIXEL_PATH_SET_UP, &[1, 1, 1, 0, 2, 1, 0]);
        r.x(&mut b, WINDOW_SYSTEM_PIXEL_COPY, &[0, 0, 10, 5]);
        assert_eq!([b.store.read(10, 1018, 1), b.store.read(11, 1018, 1), b.store.read(12, 1018, 1)], [0x5a00, 0x5a00, 0]);
    }

    /// Each context has its own X state, and a fresh activation starts over.
    #[test]
    fn the_x_state_follows_the_context() {
        let (mut b, mut r) = (backend(), Rig::new());
        r.x(&mut b, WINDOW_SYSTEM_GRAPHICS_CONTEXT_LOAD, &INDEX_GC);
        r.core.current = Some(1);
        r.send(&mut b, Job::Context(super::super::ContextEvent::Activated { slot: 1, fresh: true }));
        let text = run(&mut b, &["status"]).1;
        assert!(text.contains("X contexts        1 loaded"), "{text}");
        assert!(text.contains("X gc              not decoded"), "{text}");
        r.x(&mut b, FILL_BOXES_TWO_CORNERS, &[0, 2 << 16 | 2]);
        assert_eq!(b.store.read(0, 1023, 1), 0, "context 1 loaded no colour");
        r.core.current = Some(0);
        r.x(&mut b, FILL_BOXES_TWO_CORNERS, &[0, 2 << 16 | 2]);
        assert_eq!(b.store.read(1, 1022, 1), 0x5a00, "context 0 kept its own");
    }

    /// A DAC cursor load is decoded and held by channel, and draws nothing until channel 0's control register enables it; a load of another length, or
    /// for a channel the table does not give, is refused and counted.
    #[test]
    fn a_dac_cursor_load_is_held_by_channel() {
        let mut b = backend();
        let mut load = [0u32; 263];
        load[0] = 3;
        load[1] = 0x3ff;
        send(&mut b, Job::Video { command: VIDEO_DAC_CURSOR_LOAD, data: &load });
        send(&mut b, Job::Video { command: VIDEO_DAC_CURSOR_LOAD, data: &[0; 5] });
        let mut stray = load;
        stray[0] = 0x1234;
        send(&mut b, Job::Video { command: VIDEO_DAC_CURSOR_LOAD, data: &stray });
        // Channel 8 is the first value past the table's 0 to 7.
        stray[0] = 8;
        send(&mut b, Job::Video { command: VIDEO_DAC_CURSOR_LOAD, data: &stray });
        let status = run(&mut b, &["status"]).1;
        assert!(status.contains("DAC cursor        4 loads, 3 refused; held for channels [3]; channel 0's is not drawn"), "{status}");
        assert!(status.contains("control value for the owning channel: 0x5 = X11 cursor, enabled, not interlaced"), "{status}");
        assert_eq!(b.tables.lock().dac().cursors().next().unwrap().foreground, [0x3ff, 0, 0]);
        assert_eq!(crate::dev::kona::tables::VIDEO.get(VIDEO_DAC_CURSOR_LOAD as usize).copied().flatten(), Some(263));
    }

    /// A video setter reaches the display generator's channels, a Get changes nothing, and a reset
    /// puts the power-up state back.
    #[test]
    fn video_setters_reach_the_channels_and_a_reset_restores_them() {
        let mut b = backend();
        send(&mut b, Job::Video { command: VIDEO_CURSOR_PRIORITY, data: &[2, 9] });
        send(&mut b, Job::Video { command: VIDEO_GET_BOARD_CONFIG, data: &[0] });
        assert_eq!(b.channels.channel[2].cursor_priority, 9);
        assert_eq!(b.channels.channel[0], display::channels::Channels::new(2).channel[0]);
        // `kona status` says which channels hold blanking.
        assert!(run(&mut b, &["status"]).1.contains("video channels   enabled [0], blanked [] (channel 0's, and all while temporarily blanked, are drawn black), frame 1280 x 1024\n"));
        send(&mut b, Job::Video { command: VIDEO_CHANNEL_BLANKING, data: &[3, 1] });
        send(&mut b, Job::Video { command: VIDEO_TEMP_BLANK, data: &[1] });
        assert!(run(&mut b, &["status"]).1.contains("blanked [3] (channel 0's, and all while temporarily blanked, are drawn black), frame 1280 x 1024, all blanked\n"));
        b.reset();
        assert_eq!(b.channels.channel[2].cursor_priority, 0);
    }

    /// The loader releasing the firmware runs the stored combination, or the built-in one when there is none: the
    /// job reaches the channels without any recording. After it channel 0 runs 1280x1024_60.vfo, channel 1 holds
    /// its format blanked, and the Gets say so.
    #[test]
    fn starting_the_firmware_runs_the_built_in_combination_on_the_channels() {
        let (mut b, mut r) = (backend(), Rig::new());
        let ram = r.ram.clone();
        let words = |at: u64, n: usize| ram.bytes(at, 4 * n).chunks(4).map(|c| u32::from_be_bytes(c.try_into().unwrap())).collect::<Vec<u32>>();
        r.send(&mut b, Job::Video { command: VIDEO_GET_CHANNEL_CONFIG, data: &[0] });
        assert_eq!(words(RETURN_PAGE + 8 + 4, 5)[1], 0, "before the start channel 0 holds no named format");
        r.send(&mut b, Job::FirmwareStart { combination: &[] });
        r.send(&mut b, Job::Video { command: VIDEO_GET_CHANNEL_CONFIG, data: &[0] });
        let w = words(RETURN_PAGE + 8, 21);
        assert_eq!(w[0], 7, "the reply id");
        assert_eq!(&w[2..6], &[u32::from_be_bytes(*b"1280"), u32::from_be_bytes(*b"x102"), u32::from_be_bytes(*b"4_60"), u32::from_be_bytes(*b".vfo")]);
        r.send(&mut b, Job::Video { command: VIDEO_GET_CHANNEL_BLANKING, data: &[1] });
        assert_eq!(words(RETURN_PAGE + 8, 3), [9, 1, 10], "channel 1 is blanked");
        r.send(&mut b, Job::Video { command: VIDEO_GET_CHANNEL_BLANKING, data: &[0] });
        assert_eq!(words(RETURN_PAGE + 8, 3), [9, 0, 9], "and channel 0 is not");
    }

    /// A Get posts its reply on the return page from byte 8 and the ready word with it (`post_return`
    /// writes the data first and the ready word last); a setter posts nothing.
    #[test]
    fn a_video_get_posts_its_reply_and_the_ready_word() {
        let (mut b, mut r) = (backend(), Rig::new());
        r.send(&mut b, Job::Video { command: VIDEO_CURSOR_PRIORITY, data: &[0, 1] });
        assert_eq!(r.ram.bytes(RETURN_PAGE, 36), [0; 36], "a setter has no reply");
        r.send(&mut b, Job::Video { command: VIDEO_GET_BOARD_CONFIG, data: &[0] });
        let words = |v: &[u32]| v.iter().flat_map(|w| w.to_be_bytes()).collect::<Vec<u8>>();
        assert_eq!(r.ram.bytes(RETURN_PAGE, 4), words(&[1]));
        assert_eq!(r.ram.bytes(RETURN_PAGE + 8, 28), words(&[6, 1, 0xff, 0, 0, 0, 0x106]));
        // With no return page the reply is dropped, logged, and the board carries on.
        r.ram.put(RETURN_PAGE, &[0; 36]);
        r.core.return_page = None;
        r.send(&mut b, Job::Video { command: VIDEO_GET_BOARD_CONFIG, data: &[0] });
        assert_eq!(r.ram.bytes(RETURN_PAGE, 36), [0; 36]);
    }

    /// A 64 x 64 window at the screen's corner, double buffered, with a viewport that
    /// maps clip coordinates -1 to 1 onto it; draws go to colour word 2. The lit pixels
    /// of that word are read back with `lit_pixels`.
    fn gl_window() -> KonaBackend {
        let mut b = KonaBackend::new(KonaConfig::default());
        command(&mut b, Grammar::Kernel, WINDOW_CLIP_SETUP, &[0, 0, 64, 64, 0, 0, 64, 64, 0, 13, 0]);
        command(&mut b, Grammar::Gl, CONFIGURE_FRAME_BUFFER_LAYOUT, &[0, 8, 0, 1, 0, 0, 24, 0, 0, 0]);
        command(&mut b, Grammar::Gl, SET_VIEWPORT, &[0, 0, 64, 64]);
        command(&mut b, Grammar::Gl, SET_CURRENT_COLOUR_4_COMPONENTS_SINGLE_FLOAT, &[1.0f32.to_bits(), 1.0f32.to_bits(), 1.0f32.to_bits(), 1.0f32.to_bits()]);
        b
    }

    /// Window coordinate to clip coordinate for `gl_window`'s 64-pixel viewport.
    fn ndc(w: f32) -> u32 {
        (w / 32.0 - 1.0).to_bits()
    }

    fn vertices(b: &mut KonaBackend, mode: u32, points: &[(f32, f32)]) {
        command(b, Grammar::Gl, mode, &[0]);
        for &(x, y) in points {
            command(b, Grammar::Gl, SUBMIT_VERTEX_3_COMPONENTS_SINGLE_FLOAT, &[ndc(x), ndc(y), 0.0f32.to_bits()]);
        }
        command(b, Grammar::Gl, END_OPEN_PRIMITIVE, &[0]);
    }

    fn lit_pixels(b: &mut KonaBackend) -> Vec<(u32, u32)> {
        command(b, Grammar::Kernel, WINDOW_SYSTEM_PIXEL_PATH_FINISH, &[0]);
        (0..64).flat_map(|y| (0..64u32).map(move |x| (x, y))).filter(|&(x, y)| b.store.read(x, y, 2) != 0).collect()
    }

    /// A texture through the board: a 2 x 2 image loaded inline, bound by its short bind and
    /// record, and drawn with the replace environment, shows its texels at nearest and
    /// blends them at linear; with the texture off the draw is the vertex colour.
    #[test]
    fn a_loaded_and_bound_texture_shows_on_a_triangle() {
        use crate::dev::kona::defs::gl::TEXTURE_2D;
        let mut b = gl_window();
        let mut hdr = [0u32; 20];
        hdr[..4].copy_from_slice(&[0x0de1, 2, 2, 5]);
        hdr[4] = 1;
        hdr[9] = 0x30;
        hdr[13] = 1;
        hdr[14] = 1;
        hdr[16] = 0x11;
        hdr[17] = 0x2200;
        hdr[18] = 0x1000;
        let image = [0xff00_00ffu32, 0x00ff_00ff, 0x0000_ffff, 0xffff_ffff];
        assert_eq!(command(&mut b, Grammar::Gl, DRAW_TEXELS_INTO_TEXTURE_MEMORY, &hdr), Reply::Data(4));
        send(&mut b, Job::Data { grammar: Grammar::Gl, selector: DRAW_TEXELS_INTO_TEXTURE_MEMORY, args: &hdr, words: &image });
        command(&mut b, Grammar::Gl, TEXTURE_LOAD_BY_IDENTIFIER, &[0]);
        let mut rec = [0u32; 22];
        rec[0] = 0x44;
        rec[1] = 0x1010;
        rec[6] = 0x1000;
        rec[7] = 0xffff_ffff;
        let bind = |b: &mut KonaBackend, mag: u32| {
            command(b, Grammar::Gl, BIND_TEXTURE_CURRENT_FORM, &[0xffff_ffff, 2 << 12, mag << 5, 0x0011, 1]);
            command(b, Grammar::Gl, BIND_2D_MIP_MAPPED_TEXTURE, &rec);
        };
        bind(&mut b, 0);
        command(&mut b, Grammar::Gl, SET_TEXTURE_ENVIRONMENT_PARAMETER_FLOAT_FORM, &[0x2200, (0x1e01 as f32).to_bits(), 0, 0, 0]);
        command(&mut b, Grammar::Gl, ENABLE_CAPABILITY, &[TEXTURE_2D]);
        let quad = |b: &mut KonaBackend| {
            command(b, Grammar::Gl, BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES, &[0]);
            for (x, y, s, t) in [(0.0, 0.0, 0.0, 0.0), (64.0, 0.0, 1.0, 0.0), (0.0, 64.0, 0.0, 1.0), (64.0, 0.0, 1.0, 0.0), (64.0, 64.0, 1.0, 1.0), (0.0, 64.0, 0.0, 1.0)] {
                command(b, Grammar::Gl, SET_CURRENT_TEXTURE_COORDINATE_2_COMPONENTS_SINGLE_FLOAT, &[(s as f32).to_bits(), (t as f32).to_bits()]);
                command(b, Grammar::Gl, SUBMIT_VERTEX_3_COMPONENTS_SINGLE_FLOAT, &[ndc(x), ndc(y), 0.0f32.to_bits()]);
            }
            command(b, Grammar::Gl, END_OPEN_PRIMITIVE, &[0]);
        };
        quad(&mut b);
        command(&mut b, Grammar::Kernel, WINDOW_SYSTEM_PIXEL_PATH_FINISH, &[0]);
        let at = |b: &KonaBackend, x, y| b.store.read(x, y, 2);
        assert_eq!([at(&b, 10, 10), at(&b, 50, 10), at(&b, 10, 50), at(&b, 50, 50)], [0xffff_0000, 0xff00_ff00, 0xff00_00ff, 0xffff_ffff], "nearest: each quadrant its texel");
        // Linear magnification blends the four at the middle.
        bind(&mut b, 1);
        quad(&mut b);
        command(&mut b, Grammar::Kernel, WINDOW_SYSTEM_PIXEL_PATH_FINISH, &[0]);
        let mid = at(&b, 32, 32);
        assert!([mid >> 16 & 0xff, mid >> 8 & 0xff, mid & 0xff].iter().all(|&c| (100..160).contains(&c)), "{mid:#x}: red, green, blue and white average to grey");
        // Off, the draw is the vertex colour (white).
        command(&mut b, Grammar::Gl, DISABLE_CAPABILITY, &[TEXTURE_2D]);
        quad(&mut b);
        command(&mut b, Grammar::Kernel, WINDOW_SYSTEM_PIXEL_PATH_FINISH, &[0]);
        assert_eq!(at(&b, 10, 10), 0xffff_ffff);
        let text = run(&mut b, &["status"]).1;
        assert!(text.contains("1 level loads") && text.contains("2 binds (0 refused)"), "{text}");
    }

    /// A volume through the board: two 2 x 2 slices loaded one command each, bound by the short bind and the
    /// single-level volume record, and drawn with the replace environment under the 3D enable. The texture
    /// coordinate's r picks the slice; with the 3D enable off, the draw is the vertex colour.
    #[test]
    fn a_loaded_and_bound_volume_shows_the_slice_r_names() {
        use crate::dev::kona::defs::gl::TEXTURE_3D;
        let mut b = gl_window();
        for (z, texel) in [(0u32, 0xff00_00ffu32), (1, 0x0000_ffff)] {
            let mut hdr = [0u32; 20];
            hdr[..4].copy_from_slice(&[0x806f, 2, 2, 5]);
            hdr[4] = 1;
            hdr[9] = 0x30;
            hdr[13] = 1;
            hdr[14] = 1;
            hdr[15] = z;
            hdr[16] = 0x111;
            hdr[17] = 0x2200;
            hdr[18] = 0x1000;
            assert_eq!(command(&mut b, Grammar::Gl, DRAW_TEXELS_INTO_TEXTURE_MEMORY, &hdr), Reply::Data(4));
            send(&mut b, Job::Data { grammar: Grammar::Gl, selector: DRAW_TEXELS_INTO_TEXTURE_MEMORY, args: &hdr, words: &[texel; 4] });
        }
        command(&mut b, Grammar::Gl, TEXTURE_LOAD_BY_IDENTIFIER, &[0]);
        command(&mut b, Grammar::Gl, BIND_TEXTURE_CURRENT_FORM, &[0xffff_ffff, 3 << 28 | 2 << 12, 0, 0x0111, 1]);
        command(&mut b, Grammar::Gl, BIND_3D_TEXTURE, &[0x44, 0x111, 0x1000, 0]);
        command(&mut b, Grammar::Gl, SET_TEXTURE_ENVIRONMENT_PARAMETER_FLOAT_FORM, &[0x2200, (0x1e01 as f32).to_bits(), 0, 0, 0]);
        command(&mut b, Grammar::Gl, ENABLE_CAPABILITY, &[TEXTURE_3D]);
        let quad = |b: &mut KonaBackend, r: f32, x0: f32| {
            command(b, Grammar::Gl, BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES, &[0]);
            for (x, y) in [(0.0, 0.0), (32.0, 0.0), (0.0, 32.0), (32.0, 0.0), (32.0, 32.0), (0.0, 32.0)] {
                command(b, Grammar::Gl, SET_CURRENT_TEXTURE_COORDINATE_3_COMPONENTS_SINGLE_FLOAT, &[(x / 32.0f32).to_bits(), (y / 32.0f32).to_bits(), r.to_bits()]);
                command(b, Grammar::Gl, SUBMIT_VERTEX_3_COMPONENTS_SINGLE_FLOAT, &[ndc(x + x0), ndc(y), 0.0f32.to_bits()]);
            }
            command(b, Grammar::Gl, END_OPEN_PRIMITIVE, &[0]);
        };
        quad(&mut b, 0.25, 0.0);
        quad(&mut b, 0.75, 32.0);
        command(&mut b, Grammar::Kernel, WINDOW_SYSTEM_PIXEL_PATH_FINISH, &[0]);
        let at = |b: &KonaBackend, x, y| b.store.read(x, y, 2);
        assert_eq!([at(&b, 8, 8), at(&b, 24, 24), at(&b, 40, 8), at(&b, 56, 24)], [0xffff_0000, 0xffff_0000, 0xff00_00ff, 0xff00_00ff], "slice 0 where r is a quarter, slice 1 at three quarters");
        command(&mut b, Grammar::Gl, DISABLE_CAPABILITY, &[TEXTURE_3D]);
        quad(&mut b, 0.25, 0.0);
        command(&mut b, Grammar::Kernel, WINDOW_SYSTEM_PIXEL_PATH_FINISH, &[0]);
        assert_eq!(at(&b, 8, 8), 0xffff_ffff);
        let text = run(&mut b, &["status"]).1;
        assert!(text.contains("1 binds (0 refused)") && text.contains("2 volume slice loads, 1 volume binds"), "{text}");
    }

    /// A detail bind (token 0x17d) and a filter-coefficient table (token 0x17e and its 33 words) are kept and counted,
    /// and neither refuses a bind nor draws a flat colour: the base texture stays what is bound.
    #[test]
    fn a_detail_bind_and_a_filter_table_are_kept_and_change_nothing_drawn() {
        let mut b = gl_window();
        command(&mut b, Grammar::Gl, BIND_DETAIL_TEXTURE, &[0xffff_ffff, 3, 2, 0x2, 0x400, 0x400]);
        command(&mut b, Grammar::Gl, LOAD_ONE_ENTRY_OF_ARM_TEXTURE_TABLE, &[1]);
        send(&mut b, Job::Data { grammar: Grammar::Gl, selector: LOAD_ONE_ENTRY_OF_ARM_TEXTURE_TABLE, args: &[1], words: &[0; 33] });
        assert_eq!((b.tex_counts.detail_binds, b.tex_counts.filter_tables), (1, 1));
        assert_eq!((b.tex_counts.unmodelled, b.tex_counts.bind_refused), (0, 0));
        assert!(run(&mut b, &["status"]).1.contains("detail            1 binds, 1 filter tables"));
    }

    /// A perspective-divided textured quad samples with the divided coordinate. The right edge is twice as far (w = 2) as
    /// the left, so the screen's middle column is a third of the way across the texture, not half: it shows the first texel
    /// of a 2 x 2 image at nearest, where the affine coordinate would show the second.
    #[test]
    fn a_perspective_textured_quad_samples_with_the_divided_coordinate() {
        use crate::dev::kona::defs::gl::TEXTURE_2D;
        let mut b = gl_window();
        let mut hdr = [0u32; 20];
        hdr[..4].copy_from_slice(&[0x0de1, 2, 2, 5]);
        hdr[4] = 1;
        hdr[9] = 0x30;
        hdr[13] = 1;
        hdr[14] = 1;
        hdr[16] = 0x11;
        hdr[17] = 0x2200;
        hdr[18] = 0x1000;
        let image = [0xff00_00ffu32, 0x00ff_00ff, 0x0000_ffff, 0xffff_ffff];
        command(&mut b, Grammar::Gl, DRAW_TEXELS_INTO_TEXTURE_MEMORY, &hdr);
        send(&mut b, Job::Data { grammar: Grammar::Gl, selector: DRAW_TEXELS_INTO_TEXTURE_MEMORY, args: &hdr, words: &image });
        command(&mut b, Grammar::Gl, TEXTURE_LOAD_BY_IDENTIFIER, &[0]);
        let mut rec = [0u32; 22];
        rec[0] = 0x44;
        rec[1] = 0x1010;
        rec[6] = 0x1000;
        rec[7] = 0xffff_ffff;
        command(&mut b, Grammar::Gl, BIND_TEXTURE_CURRENT_FORM, &[0xffff_ffff, 2 << 12, 0, 0x0011, 1]);
        command(&mut b, Grammar::Gl, BIND_2D_MIP_MAPPED_TEXTURE, &rec);
        command(&mut b, Grammar::Gl, SET_TEXTURE_ENVIRONMENT_PARAMETER_FLOAT_FORM, &[0x2200, (0x1e01 as f32).to_bits(), 0, 0, 0]);
        command(&mut b, Grammar::Gl, ENABLE_CAPABILITY, &[TEXTURE_2D]);
        // Clip coordinates: x = +-1 times w, so the screen's x is 0 and 64 whatever w is.
        let corner = |b: &mut KonaBackend, xn: f32, yn: f32, w: f32, s: f32| {
            command(b, Grammar::Gl, SET_CURRENT_TEXTURE_COORDINATE_2_COMPONENTS_SINGLE_FLOAT, &[s.to_bits(), 0.25f32.to_bits()]);
            command(b, Grammar::Gl, SUBMIT_VERTEX_4_COMPONENTS_SINGLE_FLOAT, &[(xn * w).to_bits(), (yn * w).to_bits(), 0.0f32.to_bits(), w.to_bits()]);
        };
        command(&mut b, Grammar::Gl, BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES, &[0]);
        for (xn, yn, w, s) in [(-1.0, -0.5, 1.0, 0.0), (1.0, -0.5, 2.0, 1.0), (1.0, 0.5, 2.0, 1.0), (-1.0, -0.5, 1.0, 0.0), (1.0, 0.5, 2.0, 1.0), (-1.0, 0.5, 1.0, 0.0)] {
            corner(&mut b, xn, yn, w, s);
        }
        command(&mut b, Grammar::Gl, END_OPEN_PRIMITIVE, &[0]);
        command(&mut b, Grammar::Kernel, WINDOW_SYSTEM_PIXEL_PATH_FINISH, &[0]);
        let at = |b: &KonaBackend, x| b.store.read(x, 32, 2);
        // The middle is a third across (red); the affine half would be green. Near the right edge, s nears 1 (green).
        assert_eq!(at(&b, 32), 0xffff_0000, "the divided coordinate: a third of the way across");
        assert_eq!(at(&b, 8), 0xffff_0000);
        assert_eq!(at(&b, 60), 0xff00_ff00, "near the far edge");
        // The coordinate is half at two thirds of the screen: s = 0.5 t / (1 - 0.5 t), not at the middle.
        let first_green = (0..64).find(|&x| at(&b, x) == 0xff00_ff00).expect("some green");
        assert!((40..46).contains(&first_green), "the texel edge is at {first_green}, two thirds of the screen, not the middle");
    }

    /// A point of one pixel lights exactly the pixel its vertex is in, even on its
    /// edge; one larger than `POINT_PIXEL_MAX` is a disc of about its area.
    #[test]
    fn a_point_lights_one_pixel_or_a_disc() {
        for (x, y, want) in [(10.5, 10.5, (10, 10)), (20.0, 30.0, (20, 30)), (7.9, 3.0, (7, 3))] {
            let mut b = gl_window();
            vertices(&mut b, BEGIN_PRIMITIVE_POINTS, &[(x, y)]);
            assert_eq!(lit_pixels(&mut b), [want], "a point at ({x}, {y})");
        }
        let mut b = gl_window();
        command(&mut b, Grammar::Gl, SET_POINT_SIZE, &[3.0f32.to_bits(), 0]);
        vertices(&mut b, BEGIN_PRIMITIVE_POINTS, &[(10.5, 10.5)]);
        assert_eq!(lit_pixels(&mut b), [(10, 10)], "size 3 is still one pixel");
        let mut b = gl_window();
        command(&mut b, Grammar::Gl, SET_POINT_SIZE, &[8.0f32.to_bits(), 0]);
        vertices(&mut b, BEGIN_PRIMITIVE_POINTS, &[(32.0, 32.0)]);
        let lit = lit_pixels(&mut b);
        assert!((44..=56).contains(&lit.len()), "a disc of diameter 8 holds about 50 pixels: {}", lit.len());
        assert!(lit.iter().all(|&(x, y)| (27..=36).contains(&x) && (27..=36).contains(&y)));
    }

    /// A line of width 1 lights one row, from just past its first end to its last pixel
    /// (the fill rule leaves the left edge out); a wide line lights its width.
    #[test]
    fn a_line_lights_its_width() {
        let mut b = gl_window();
        vertices(&mut b, BEGIN_PRIMITIVE_INDEPENDENT_LINES, &[(10.5, 20.5), (40.5, 20.5)]);
        let lit = lit_pixels(&mut b);
        assert!(lit.iter().all(|&(_, y)| y == 20), "one row");
        assert_eq!((lit.first().map(|p| p.0), lit.last().map(|p| p.0), lit.len()), (Some(11), Some(40), 30));
        let mut b = gl_window();
        command(&mut b, Grammar::Gl, SET_LINE_WIDTH, &[3.0f32.to_bits(), 0]);
        vertices(&mut b, BEGIN_PRIMITIVE_INDEPENDENT_LINES, &[(10.5, 20.5), (40.5, 20.5)]);
        let rows: std::collections::BTreeSet<u32> = lit_pixels(&mut b).iter().map(|p| p.1).collect();
        assert_eq!(rows, [19, 20, 21].into(), "width 3");
    }

    /// The alpha a pixel was written with: smoothing scales it, and with no blending the colour is
    /// the vertex's, so alpha reads the coverage back.
    fn alpha_at(b: &KonaBackend, x: u32, y: u32) -> u32 {
        b.store.read(x, y, 2) >> 24
    }

    /// A smoothed line of width 1 centred on a row is that row at full coverage, with each end a half
    /// pixel in (its end passes through the end pixels' centres); moved half a pixel up it shares two
    /// rows evenly.
    #[test]
    fn a_smoothed_line_ramps_its_ends_and_shares_a_row_boundary() {
        use crate::dev::kona::defs::gl::LINE_SMOOTH;
        let mut b = gl_window();
        command(&mut b, Grammar::Gl, ENABLE_CAPABILITY, &[LINE_SMOOTH]);
        vertices(&mut b, BEGIN_PRIMITIVE_INDEPENDENT_LINES, &[(10.5, 20.5), (40.5, 20.5)]);
        let lit = lit_pixels(&mut b);
        assert!(lit.iter().all(|&(_, y)| y == 20), "one row: {lit:?}");
        assert_eq!((lit.first().map(|p| p.0), lit.last().map(|p| p.0), lit.len()), (Some(10), Some(40), 31), "the end pixels, half covered, are drawn too");
        assert_eq!([alpha_at(&b, 10, 20), alpha_at(&b, 11, 20), alpha_at(&b, 25, 20), alpha_at(&b, 39, 20), alpha_at(&b, 40, 20)], [128, 255, 255, 255, 128]);
        let mut b = gl_window();
        command(&mut b, Grammar::Gl, ENABLE_CAPABILITY, &[LINE_SMOOTH]);
        vertices(&mut b, BEGIN_PRIMITIVE_INDEPENDENT_LINES, &[(10.5, 21.0), (40.5, 21.0)]);
        lit_pixels(&mut b);
        assert_eq!([alpha_at(&b, 25, 19), alpha_at(&b, 25, 20), alpha_at(&b, 25, 21), alpha_at(&b, 25, 22)], [0, 128, 128, 0]);
        // Off, the same line is a row of whole pixels from the second.
        let mut b = gl_window();
        vertices(&mut b, BEGIN_PRIMITIVE_INDEPENDENT_LINES, &[(10.5, 21.0), (40.5, 21.0)]);
        let lit = lit_pixels(&mut b);
        assert!(lit.iter().all(|&(_, y)| y == 21) && lit.iter().all(|&(x, y)| alpha_at(&b, x, y) == 255));
    }

    /// A diagonal smoothed line through pixel centres covers its own pixel by a little under a whole and
    /// its two neighbours across by about a quarter each: a strip one pixel wide at 45 degrees takes
    /// 0.914 of its pixel and 0.25 of each neighbour, 1.414 in all, the line's width times the length
    /// of a diagonal step. Set-up snaps the rectangle's corners to eighths of a pixel, which widens it
    /// a little, so the figures here are a little above those.
    #[test]
    fn a_smoothed_diagonal_line_shares_its_width_across_the_pixels_it_touches() {
        use crate::dev::kona::defs::gl::LINE_SMOOTH;
        let mut b = gl_window();
        command(&mut b, Grammar::Gl, ENABLE_CAPABILITY, &[LINE_SMOOTH]);
        vertices(&mut b, BEGIN_PRIMITIVE_INDEPENDENT_LINES, &[(10.5, 10.5), (40.5, 40.5)]);
        lit_pixels(&mut b);
        let a = |dx: u32, dy: u32| alpha_at(&b, dx, dy);
        assert!((235..=250).contains(&a(25, 25)), "{}", a(25, 25));
        assert_eq!((a(26, 25), a(25, 26)), (a(24, 25), a(25, 24)), "the neighbours either side are alike");
        assert!((60..=80).contains(&a(26, 25)), "{}", a(26, 25));
        let sum = (a(25, 25) + a(26, 25) + a(24, 25)) as f32 / 255.0;
        assert!((1.35..=1.6).contains(&sum), "{sum}");
        assert_eq!(a(27, 25), 0, "and none past them");
    }

    /// A smoothed polygon ramps its outer edges and not the seam of the fan it is drawn as: a square
    /// from pixel centre to pixel centre covers its edge pixels by half and its corners by a quarter,
    /// and the pixels along its diagonal are whole.
    #[test]
    fn a_smoothed_polygon_ramps_its_outline_and_not_its_seam() {
        use crate::dev::kona::defs::gl::POLYGON_SMOOTH;
        let square = [(10.5, 10.5), (40.5, 10.5), (40.5, 40.5), (10.5, 40.5)];
        let mut b = gl_window();
        command(&mut b, Grammar::Gl, ENABLE_CAPABILITY, &[POLYGON_SMOOTH]);
        vertices(&mut b, BEGIN_PRIMITIVE_POLYGON, &square);
        lit_pixels(&mut b);
        let a = |x, y| alpha_at(&b, x, y);
        assert_eq!([a(10, 25), a(25, 10), a(40, 25), a(25, 40)], [128; 4], "edge pixels, half covered");
        assert_eq!([a(10, 10), a(40, 10), a(40, 40), a(10, 40)], [64; 4], "corners, a quarter");
        assert_eq!([a(11, 11), a(25, 25), a(39, 39), a(20, 30)], [255; 4], "inside, seam included, whole");
        assert_eq!([a(9, 25), a(41, 25), a(25, 9), a(25, 41)], [0; 4], "and nothing past the ramp");
        // Off, the same polygon is whole pixels only.
        let mut b = gl_window();
        vertices(&mut b, BEGIN_PRIMITIVE_POLYGON, &square);
        let lit = lit_pixels(&mut b);
        assert!(lit.iter().all(|&(x, y)| alpha_at(&b, x, y) == 255));
        // A triangle's slanted edge covers its pixels in part, and the pixels a plain triangle lights
        // keep at least half a pixel's cover on their outer edge.
        let mut b = gl_window();
        command(&mut b, Grammar::Gl, ENABLE_CAPABILITY, &[POLYGON_SMOOTH]);
        vertices(&mut b, BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES, &[(10.0, 10.0), (50.0, 10.0), (10.0, 50.0)]);
        lit_pixels(&mut b);
        let mid = alpha_at(&b, 30, 29);
        assert!((1..255).contains(&mid), "a pixel on the hypotenuse is partly covered: {mid}");
    }

    /// A smoothed point covers the share of each pixel its disc does: the pixel about a size-1 point's
    /// centre about pi / 4, a size-3 point's pixels symmetric about it and summing to the disc's area.
    #[test]
    fn a_smoothed_point_covers_by_its_disc() {
        use crate::dev::kona::defs::gl::POINT_SMOOTH;
        let mut b = gl_window();
        command(&mut b, Grammar::Gl, ENABLE_CAPABILITY, &[POINT_SMOOTH]);
        vertices(&mut b, BEGIN_PRIMITIVE_POINTS, &[(20.5, 20.5)]);
        let lit = lit_pixels(&mut b);
        assert_eq!(lit, [(20, 20)]);
        // pi / 4 of full, from an 8 x 8 grid of sample positions: within a few percent.
        assert!((195..=212).contains(&alpha_at(&b, 20, 20)), "{}", alpha_at(&b, 20, 20));
        let mut b = gl_window();
        command(&mut b, Grammar::Gl, ENABLE_CAPABILITY, &[POINT_SMOOTH]);
        command(&mut b, Grammar::Gl, SET_POINT_SIZE, &[3.0f32.to_bits(), 0]);
        vertices(&mut b, BEGIN_PRIMITIVE_POINTS, &[(20.5, 20.5)]);
        lit_pixels(&mut b);
        let a = |dx: i32, dy: i32| alpha_at(&b, (20 + dx) as u32, (20 + dy) as u32);
        assert_eq!(a(0, 0), 255);
        assert!(a(-1, 0) == a(1, 0) && a(0, -1) == a(0, 1) && a(1, 0) == a(0, 1), "the four edge neighbours alike");
        assert!(a(1, 1) == a(-1, -1) && a(1, 1) < a(1, 0), "the corners, alike, are less than the edges");
        let sum: u32 = (-2..=2).flat_map(|dy| (-2..=2).map(move |dx| (dx, dy))).map(|(dx, dy)| a(dx, dy)).sum();
        let area = std::f32::consts::PI * 1.5 * 1.5;
        assert!((sum as f32 / 255.0 - area).abs() < 0.2, "coverage sums to {} against the disc's {area}", sum as f32 / 255.0);
    }

    /// The line stipple indexes its 16-bit pattern by the distance along the line over
    /// the repeat count; the polygon stipple, 32 x 32 bits, by window position.
    #[test]
    fn stipples_keep_the_pixels_their_patterns_name() {
        use crate::dev::kona::defs::gl::{LINE_STIPPLE, POLYGON_STIPPLE};
        let mut b = gl_window();
        command(&mut b, Grammar::Gl, ENABLE_CAPABILITY, &[LINE_STIPPLE]);
        command(&mut b, Grammar::Gl, SET_LINE_STIPPLE, &[1, 0x00ff]);
        vertices(&mut b, BEGIN_PRIMITIVE_INDEPENDENT_LINES, &[(10.5, 20.5), (40.5, 20.5)]);
        let xs: Vec<u32> = lit_pixels(&mut b).iter().map(|p| p.0).collect();
        let want: Vec<u32> = (11..=40).filter(|x| (x - 10) % 16 < 8).collect();
        assert_eq!(xs, want, "the low eight bits of the pattern: eight pixels on, eight off");
        // A repeat of 2 stretches each bit over two pixels.
        let mut b = gl_window();
        command(&mut b, Grammar::Gl, ENABLE_CAPABILITY, &[LINE_STIPPLE]);
        command(&mut b, Grammar::Gl, SET_LINE_STIPPLE, &[2, 0x00ff]);
        vertices(&mut b, BEGIN_PRIMITIVE_INDEPENDENT_LINES, &[(10.5, 20.5), (50.5, 20.5)]);
        let xs: Vec<u32> = lit_pixels(&mut b).iter().map(|p| p.0).collect();
        let want: Vec<u32> = (11..=50).filter(|x| (x - 10) / 2 % 16 < 8).collect();
        assert_eq!(xs, want);
        // Polygon stipple: a checkerboard, row 0 starting set at the leftmost column.
        let mut b = gl_window();
        let mut rows = [0xaaaa_aaaau32; 32];
        rows.iter_mut().skip(1).step_by(2).for_each(|r| *r = 0x5555_5555);
        assert_eq!(command(&mut b, Grammar::Gl, LOAD_POLYGON_STIPPLE_PATTERN, &[0, 0, 0]), Reply::Data(32));
        send(&mut b, Job::Data { grammar: Grammar::Gl, selector: LOAD_POLYGON_STIPPLE_PATTERN, args: &[0, 0, 0], words: &rows });
        command(&mut b, Grammar::Gl, ENABLE_CAPABILITY, &[POLYGON_STIPPLE]);
        vertices(&mut b, BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES, &[(0.5, 0.5), (60.5, 0.5), (0.5, 60.5)]);
        let lit = lit_pixels(&mut b);
        assert!(lit.len() > 500 && lit.iter().all(|&(x, y)| (x + y) % 2 == 0), "a checkerboard of {} pixels", lit.len());
        // With the stipple off the same triangle is solid.
        let mut b = gl_window();
        send(&mut b, Job::Data { grammar: Grammar::Gl, selector: LOAD_POLYGON_STIPPLE_PATTERN, args: &[0, 0, 0], words: &rows });
        vertices(&mut b, BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES, &[(0.5, 0.5), (60.5, 0.5), (0.5, 60.5)]);
        assert!(lit_pixels(&mut b).iter().any(|&(x, y)| (x + y) % 2 == 1));
    }

    /// A GL triangle, through the board: the kernel places the window and the
    /// buffer configuration lays the planes out; the triangle goes to the back
    /// buffer's word, the display shows nothing of it until a swap names the
    /// window's id with the buffer bit, and then shows it as direct colour.
    #[test]
    fn a_gl_triangle_draws_into_the_back_buffer_and_shows_after_the_swap() {
        use crate::dev::kona::back::display::compose;
        use crate::dev::kona::back::store::DID_PLANE;
        let mut b = KonaBackend::new(KonaConfig::default());
        let f = |v: f32| v.to_bits();
        // Window at (10, 20), 50 x 40, display id 13, not partly covered.
        command(&mut b, Grammar::Kernel, WINDOW_CLIP_SETUP, &[10, 20, 50, 40, 10, 20, 50, 40, 0, 13, 0]);
        command(&mut b, Grammar::Gl, 1, &[0, 8, 0, 1, 0, 0, 24, 0, 0, 0]);
        command(&mut b, Grammar::Gl, SET_VIEWPORT, &[0, 0, 50, 40]);
        command(&mut b, Grammar::Gl, SET_CURRENT_COLOUR_4_COMPONENTS_SINGLE_FLOAT, &[f(1.0), f(0.0), f(0.0), f(1.0)]);
        command(&mut b, Grammar::Gl, BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES, &[0]);
        for (x, y) in [(-1.0, -1.0), (1.0, -1.0), (0.0, 1.0)] {
            command(&mut b, Grammar::Gl, SUBMIT_VERTEX_3_COMPONENTS_SINGLE_FLOAT, &[f(x), f(y), f(0.0)]);
        }
        command(&mut b, Grammar::Gl, END_OPEN_PRIMITIVE, &[0]);
        // A job that touches pixels waits for the drawing first.
        command(&mut b, Grammar::Kernel, WINDOW_SYSTEM_PIXEL_PATH_FINISH, &[0]);
        let red = |w: usize| (0..1024).flat_map(|y| (0..1280u32).map(move |x| (x, y))).filter(|&(x, y)| b.store.read(x, y, w) == 0xffff_0000).count();
        assert!(red(1) == 0, "the front buffer is untouched");
        let back = red(2);
        assert!(back > 900 && back <= 1000, "about half of 50 x 40 pixels: {back}");
        assert!(b.store.read(30, 30, 2) == 0xffff_0000 && b.store.read(5, 30, 2) == 0, "inside the window and outside it");
        // The display: id 13 is direct colour, showing the back buffer once swapped.
        b.store.write(30, 30, DID_PLANE, 13 << 2);
        command(&mut b, Grammar::Kernel, DISPLAY_MODE_BYTE, &[0, 13, 4]);
        let shown = |b: &KonaBackend| {
            let mut out = vec![0u32; 1280 * 1024];
            compose(&b.store, 1, &b.map.lock(), &b.tables.lock().snapshot(), &mut out, 1280);
            out[(1023 - 30) * 1280 + 30]
        };
        assert_eq!(shown(&b), 0xff00_0000, "the front buffer is empty");
        command(&mut b, Grammar::Kernel, DISPLAY_MODE_BYTE, &[0x3c0001, 13, 1]);
        assert_eq!(shown(&b), 0xff00_00ff, "red, as 0xAABBGGRR, after the swap");
        let text = run(&mut b, &["status"]).1;
        assert!(text.contains("raster stage      1 triangles"), "{text}");
    }

    /// Each row of the drain table: what reads or writes the store drains the
    /// GL stage, and what does not leaves it alone. Counted, so the test does not
    /// depend on a worker being slow enough to lose a race.
    #[test]
    fn the_drain_table_row_by_row() {
        let mut b = KonaBackend::new(test_cfg().with_pool(2, 32));
        b.tiles();
        let drains = |b: &KonaBackend| b.tiles.as_ref().map_or(0, |t| t.stats().drains);
        let mut expect = 0;
        // Rows that do not drain.
        let quiet = |b: &mut KonaBackend, what: &str, job: Job<'_>| {
            send(b, job);
            assert_eq!(drains(b), expect, "{what} left the workers alone");
        };
        quiet(&mut b, "GL state", Job::Command { grammar: Grammar::Gl, selector: SET_VIEWPORT, args: &[0, 0, 50, 40], packed: 0, handled: false });
        quiet(&mut b, "GL colour", Job::Command { grammar: Grammar::Gl, selector: SET_CURRENT_COLOUR_4_COMPONENTS_SINGLE_FLOAT, args: &[1.0f32.to_bits(); 4], packed: 0, handled: false });
        quiet(&mut b, "GL begin", Job::Command { grammar: Grammar::Gl, selector: BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES, args: &[0], packed: 0, handled: false });
        quiet(&mut b, "GL end", Job::Command { grammar: Grammar::Gl, selector: END_OPEN_PRIMITIVE, args: &[0], packed: 0, handled: false });
        quiet(&mut b, "a GL clear", Job::Command { grammar: Grammar::Gl, selector: CLEAR_SELECTED_BUFFERS, args: &[0], packed: 0, handled: false });
        quiet(&mut b, "an X state load", Job::Command { grammar: Grammar::Kernel, selector: x::state::X_POSITION, args: &[0, 0, 8, 8], packed: 0, handled: false });
        quiet(&mut b, "an X data token", Job::Command { grammar: Grammar::Kernel, selector: x::fill::X_DATA2, args: &[0, 0], packed: 0, handled: false });
        quiet(&mut b, "a display-table load", Job::Command { grammar: Grammar::Kernel, selector: DISPLAY_MODE_BYTE, args: &[0, 13, 4], packed: 0, handled: false });
        quiet(&mut b, "an indirect store", Job::Indirect { selector: XMAP_CURSOR_MODE, value: 0 });
        quiet(&mut b, "a context event", Job::Context(super::super::job::ContextEvent::Activated { slot: 1, fresh: true }));
        quiet(&mut b, "dropped words", Job::Dropped { word: 7 });
        // Rows that drain, each once.
        let mut loud = |b: &mut KonaBackend, what: &str, job: Job<'_>| {
            send(b, job);
            expect += 1;
            assert_eq!(drains(b), expect, "{what} drains");
        };
        loud(&mut b, "a textport command", Job::Command { grammar: Grammar::Textport, selector: TEXTPORT_FILL_BOX, args: &[0, 0, 3, 3], packed: 0, handled: false });
        loud(&mut b, "an X box", Job::Command { grammar: Grammar::Kernel, selector: x::fill::X_BOX, args: &[0, 2 << 16 | 2], packed: 0, handled: false });
        loud(&mut b, "an X pixel finish", Job::Command { grammar: Grammar::Kernel, selector: x::pixel::X_PIXEL_FINISH, args: &[0], packed: 0, handled: false });
        loud(&mut b, "a swap", Job::Command { grammar: Grammar::Kernel, selector: DISPLAY_MODE_BYTE, args: &[0x3c0001, 13, 1], packed: 0, handled: false });
        loud(&mut b, "a region swap-out", Job::Command { grammar: Grammar::Kernel, selector: PIPE_MEMORY_SWAP_OUT, args: &[0; 8], packed: 0, handled: false });
        loud(&mut b, "a retrace request", Job::Retrace { tag: 1 });
        loud(&mut b, "a GL finish", Job::Command { grammar: Grammar::Gl, selector: FINISH_DRAIN_PIPE_AND_SIGNAL_COMPLETION, args: &[0], packed: 0, handled: false });
        b.quiesce();
        assert_eq!(drains(&b), expect + 1, "a quiesce drains");
        assert_eq!(b.tiles.as_ref().map_or(0, |t| t.stats().busy_drains), 0, "no worker had anything queued");
    }

    /// The region commands are counted with the bytes the swaps name, and the
    /// front end still moves the bytes.
    #[test]
    fn region_commands_are_counted() {
        let (mut b, mut r) = (backend(), Rig::new());
        r.x(&mut b, PIPE_MEMORY_RESERVE, &[2, 9, 0, 1]);
        r.x(&mut b, PIPE_MEMORY_COPY_DOWN, &[2, 0, 0, 0]);
        r.x(&mut b, PIPE_MEMORY_SWAP_OUT, &[3, 0, 4, 0x400, 1, 0, 0x9000, 0x102]);
        r.x(&mut b, PIPE_MEMORY_SWAP_IN, &[3, 0, 2, 0x400, 1, 0]);
        let text = run(&mut b, &["status"]).1;
        assert!(text.contains("1 placed, 1 copied down, 1 swapped out (4096 bytes), 1 swapped in (2048 bytes)"), "{text}");
    }

    /// A bracketed read whose host buffer is not mapped writes nothing, posts no
    /// completion and is counted, as the recorder's model does.
    #[test]
    fn a_bracketed_read_into_an_unmapped_buffer_posts_nothing() {
        use crate::traits::{Device, BUS_OK};
        let (host, kona) = board();
        let w = |v: u32| assert_eq!(kona.test_write(0xc10000, 4, v as u64), BUS_OK);
        let words = |ws: &[u32]| ws.iter().for_each(|&v| w(v));
        let phys = |a: u64| 0x1000_0000 | (a >> 12) as u32;
        words(&[0x1f1, phys(REASON_PAGE), 0, 0x4000]);
        words(&[0x1fd, phys(0x50_0000), 0, 0x10000]);
        words(&[0x1f2, phys(RETURN_PAGE), 0, 0x4000]);
        words(&[0x129]);
        words(&INDEX_GC);
        words(&[0x154, 0, 2 << 16 | 1]);
        words(&[0x136, 2, 1, 0, 8, 2, 1, 0]);
        // Device page 0x1c1 has no map RAM entry.
        let dev = 0x1c1u64 << 14;
        words(&[0x1f5, (dev >> 12) as u32, 0x9000 | (dev & 0xfff) as u32, 2, 2, 0, 0]);
        words(&[0x138, 0, 0]);
        words(&[0x1f6, 0x100, 0x77]);
        assert_eq!(host.word(REASON_PAGE), 0, "no completion");
        let text = monitor(&kona, &["status"]).1;
        assert!(text.contains("read brackets   0 refused, 1 with no completion posted"), "{text}");
        kona.stop();
    }

    /// A read inside a select-feed bracket goes to the host buffer at the
    /// bracket's finish, which then posts its completion with the second word
    /// the finish carried. Through the board's own command port.
    #[test]
    fn a_bracketed_read_reaches_the_host_buffer() {
        use crate::traits::{Device, BUS_OK};
        let (host, kona) = board();
        let w = |v: u32| assert_eq!(kona.test_write(0xc10000, 4, v as u64), BUS_OK);
        let words = |ws: &[u32]| ws.iter().for_each(|&v| w(v));
        let phys = |a: u64| 0x1000_0000 | (a >> 12) as u32;
        words(&[0x1f1, phys(REASON_PAGE), 0, 0x4000]);
        words(&[0x1fd, phys(0x50_0000), 0, 0x10000]);
        words(&[0x1f2, phys(RETURN_PAGE), 0, 0x4000]);
        // Device page 0x1c0 maps the host buffer.
        let buf = 0x70_0000u64;
        assert_eq!(kona.test_write(0x100000 + 4 * 0x1c0, 4, 0x2000_0000 | (buf >> 12) as u64), BUS_OK);
        let dev = 0x1c0u64 << 14;
        words(&[0x129]);
        words(&INDEX_GC);
        words(&[0x154, 0, 2 << 16 | 1]);
        words(&[0x136, 2, 1, 0, 8, 2, 1, 0]);
        words(&[0x1f5, (dev >> 12) as u32, 0x9000 | (dev & 0xfff) as u32, 2, 2, 0, 0]);
        words(&[0x138, 0, 0]);
        assert_eq!(host.word(RETURN_PAGE), 0, "inside a bracket the bytes wait");
        words(&[0x1f6, 0x100, 0x77]);
        assert_eq!(host.bytes(buf, 2), [0x5a, 0x5a]);
        assert_eq!([host.word(REASON_PAGE), host.word(REASON_PAGE + 4), host.word(REASON_PAGE + 8)], [0x100, 0x77, 0]);
        let text = monitor(&kona, &["status"]).1;
        assert!(text.contains("1 reads"), "{text}");
        kona.stop();
    }

    /// While someone holds the pipe's state, the pixel commands still answer
    /// and a wait for the state gives up instead of hanging.
    #[test]
    fn pixel_reads_do_not_wait_for_the_state() {
        use crate::traits::Device;
        let (_host, kona) = board();
        kona.with_backend(|_| {
            let out = Capture::default();
            kona.execute_command("kona", &["hash"], Box::new(out.clone())).unwrap();
            assert_eq!(out.text().trim().len(), 16);
            kona.execute_command("kona", &["pix", "1", "1"], Box::new(out.clone())).unwrap();
            assert!(out.text().contains("word arrays allocated"));
        });
    }

    /// A command that reads the back end's state gives up waiting for it and
    /// says so: the answer is "board busy", and the state is its again after.
    #[test]
    fn a_state_command_answers_busy_when_the_state_stays_out() {
        use crate::traits::Device;
        let cfg = test_cfg();
        let host = TestHost::new();
        let kona = Kona::new(cfg, host.clone(), host.clone(), crate::dev::kona::tests::test_status(), None).unwrap();
        kona.shared.monitor_wait.store(0, Relaxed);
        kona.with_backend(|_| {
            let out = Capture::default();
            assert_eq!(kona.execute_command("kona", &["status"], Box::new(out.clone())), Err("kona: board busy".into()));
            assert!(kona.shared.pipe.try_lock().is_none(), "the state is out");
        });
        assert!(kona.shared.pipe.try_lock().is_some());
        let out = Capture::default();
        kona.execute_command("kona", &["status"], Box::new(out.clone())).unwrap();
        assert!(out.text().contains("raster managers"));
    }

    /// A stop from the main thread leaves the renderer to it; any other stop
    /// has the display thread release it as its last act.
    #[test]
    fn a_stop_from_the_main_thread_leaves_the_renderer_alone() {
        use crate::dev::ng1::rex3::Renderer;
        use crate::traits::Device;
        use std::sync::atomic::AtomicU32;
        struct Spy(Arc<AtomicU32>);
        impl Renderer for Spy {
            fn present(
                &mut self,
                _: &mut crate::disp::Rex3Screen,
                _: &mut crate::debug_overlay::DebugOverlay,
                _: &mut crate::disp::StatusBar,
                _: &mut crate::disp::StatusBarTexture,
                _: &crate::disp::BarStats,
                _: bool,
                _: Option<&[u32]>,
                _: Option<&[u32]>,
            ) {
            }

            fn stop(&mut self) {
                self.0.fetch_add(1, Relaxed);
            }
        }
        let run = |name: &str| {
            let stops = Arc::new(AtomicU32::new(0));
            let spy = stops.clone();
            std::thread::Builder::new()
                .name(name.into())
                .spawn(move || {
                    let cfg = test_cfg();
                    let host = TestHost::new();
                    let kona = Kona::new(cfg, host.clone(), host.clone(), crate::dev::kona::tests::test_status(), None).unwrap();
                    *kona.with_backend(|b| b.display()).renderer_slot().lock() = Some(Box::new(Spy(spy)));
                    kona.start();
                    kona.stop();
                })
                .unwrap()
                .join()
                .unwrap();
            stops.load(Relaxed)
        };
        assert_eq!(run("main"), 0, "the main thread releases it");
        assert_eq!(run("other"), 1, "the display thread releases it");
    }

    /// The machine may take the window before or after wiring in its status
    /// source: it is the same window.
    #[test]
    fn the_window_is_the_same_before_and_after_the_status_is_wired() {
        let cfg = test_cfg();
        let backend = KonaBackend::new(cfg);
        let before = backend.display();
        backend.set_status(crate::disp::StatusSource::new(Arc::default(), Arc::default()));
        assert!(Arc::ptr_eq(&before, &backend.display()));
    }

    #[test]
    fn pix_shows_the_allocated_words_of_a_pixel() {
        let mut b = backend();
        let (known, text) = run(&mut b, &["pix", "3", "4"]);
        assert!(known && text.contains("0 word arrays allocated"), "{text}");
        b.store.write(3, 4, 2, 0xabcd);
        let text = run(&mut b, &["pix", "3", "4"]).1;
        assert!(text.contains("word  2  0x0000abcd") && text.contains("1 word arrays"), "{text}");
        assert!(run(&mut b, &["pix", "5000", "0"]).1.contains("outside"));
        assert!(!run(&mut b, &["pix", "x", "0"]).0);
        // A reset zeroes the store in place.
        b.reset();
        assert!(run(&mut b, &["pix", "3", "4"]).1.contains("word  2  0x00000000"));
    }

    #[test]
    fn attach_and_stop_hold_and_release_the_handle() {
        use crate::traits::Device;
        let (_host, kona) = board();
        assert!(monitor(&kona, &["status"]).1.contains("detached"));
        kona.start();
        assert!(monitor(&kona, &["status"]).1.contains("display handle    attached"));
        kona.stop();
        assert!(monitor(&kona, &["status"]).1.contains("detached"));
    }

    // ---- GL pixel operations ----

    /// A pixel draw: the header, then the image words the back end asked for.
    fn draw_pixels(b: &mut KonaBackend, w: u32, h: u32, format: u32, ty: u32, image: &[u32]) {
        let hdr = [w, h, format, ty, 0, 0, 0, 0];
        assert_eq!(command(b, Grammar::Gl, DRAW_PIXEL_RECTANGLE_FROM_HOST_DATA, &hdr), if image.is_empty() { Reply::Done } else { Reply::Data(image.len() as u32) });
        if !image.is_empty() {
            send(b, Job::Data { grammar: Grammar::Gl, selector: DRAW_PIXEL_RECTANGLE_FROM_HOST_DATA, args: &hdr, words: image });
        }
    }

    fn colour_at(b: &mut KonaBackend, x: u32, y: u32) -> u32 {
        command(b, Grammar::Kernel, WINDOW_SYSTEM_PIXEL_PATH_FINISH, &[0]);
        b.store.read(x, y, 2)
    }


    // ---- Queries, select and feedback ----


    // ---- Multisample ----

    /// `gl_window` reconfigured with `samples` samples and a 24-bit depth buffer.
    fn ms_window(samples: u32) -> KonaBackend {
        let mut b = gl_window();
        command(&mut b, Grammar::Gl, CONFIGURE_FRAME_BUFFER_LAYOUT, &[0, 8, 0, 1, 0, samples, 24, 0, 0, 0]);
        b
    }

    /// A rectangle of two triangles from (2, 2) to (`right`, 20).
    fn rectangle(b: &mut KonaBackend, right: f32) {
        vertices(b, BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES, &[(2.0, 2.0), (right, 2.0), (right, 20.0), (2.0, 2.0), (right, 20.0), (2.0, 20.0)]);
    }

    fn column(b: &mut KonaBackend, x: u32, y: u32) -> u32 {
        command(b, Grammar::Kernel, WINDOW_SYSTEM_PIXEL_PATH_FINISH, &[0]);
        b.store.read(x, y, 2)
    }

    /// In a multisampled window a smoothed line takes coverage (smoothing wins for lines) while a smoothed
    /// polygon multisamples (multisampling wins for polygons).
    #[test]
    fn a_multisampled_window_smooths_lines_and_multisamples_polygons() {
        use crate::dev::kona::defs::gl::{LINE_SMOOTH, POLYGON_SMOOTH};
        let mut b = ms_window(4);
        command(&mut b, Grammar::Gl, ENABLE_CAPABILITY, &[LINE_SMOOTH]);
        vertices(&mut b, BEGIN_PRIMITIVE_INDEPENDENT_LINES, &[(10.5, 21.0), (40.5, 21.0)]);
        command(&mut b, Grammar::Kernel, WINDOW_SYSTEM_PIXEL_PATH_FINISH, &[0]);
        assert_eq!([alpha_at(&b, 25, 19), alpha_at(&b, 25, 20), alpha_at(&b, 25, 21), alpha_at(&b, 25, 22)], [0, 128, 128, 0], "half coverage a row either side");
        let mut b = ms_window(4);
        command(&mut b, Grammar::Gl, ENABLE_CAPABILITY, &[POLYGON_SMOOTH]);
        rectangle(&mut b, 10.75);
        assert_eq!([column(&mut b, 9, 10), column(&mut b, 10, 10), column(&mut b, 11, 10)], [0xffff_ffff, 0xff80_8080, 0], "the multisampled edge, not a ramp");
    }

    /// The right edge of a rectangle at 10.75 (10.25 once the half pixel comes off) falls between samples of its
    /// column: the 4-sample pattern's samples at 0 and 2 eighths across are in (one on the edge, which a right edge
    /// keeps), those at 4 and 6 out, so half the pixel is white; with 8 samples (0, 6, 5, 1, 2, 7, 3, 4 eighths across)
    /// three are in. A column left of it is whole and one right of it empty; the shared diagonal of the two triangles
    /// leaves no seam.
    #[test]
    fn a_four_and_an_eight_sample_edge_share_the_pixel_by_samples() {
        let mut b = ms_window(4);
        rectangle(&mut b, 10.75);
        assert_eq!([column(&mut b, 9, 10), column(&mut b, 10, 10), column(&mut b, 11, 10)], [0xffff_ffff, 0xff80_8080, 0]);
        // Every pixel the diagonal crosses is whole: no seam between the triangles.
        let seams: Vec<_> = (3..10).flat_map(|x| (3..19).map(move |y| (x, y))).filter(|&(x, y)| column(&mut b, x, y) != 0xffff_ffff).collect();
        assert!(seams.is_empty(), "{seams:?}");
        let mut b = ms_window(8);
        rectangle(&mut b, 10.75);
        // Three of eight samples: (3 * 255 + 4) / 8 = 96.
        assert_eq!([column(&mut b, 9, 10), column(&mut b, 10, 10), column(&mut b, 11, 10)], [0xffff_ffff, 0xff60_6060, 0]);
        let seams: Vec<_> = (3..10).flat_map(|x| (3..19).map(move |y| (x, y))).filter(|&(x, y)| column(&mut b, x, y) != 0xffff_ffff).collect();
        assert!(seams.is_empty(), "{seams:?}");
    }

    /// With the enable off, or no multisample buffer, the same edge is the single-sample one: the pixel is whole or empty.
    #[test]
    fn without_multisampling_an_edge_is_whole_or_empty() {
        let mut b = ms_window(4);
        command(&mut b, Grammar::Gl, DISABLE_CAPABILITY, &[crate::dev::kona::defs::gl::MULTISAMPLE]);
        rectangle(&mut b, 10.875);
        // The sample at the pixel's corner: 10.0 < 10.375, inside.
        assert_eq!([column(&mut b, 10, 10), column(&mut b, 11, 10)], [0xffff_ffff, 0]);
        let mut b = ms_window(0);
        rectangle(&mut b, 10.875);
        assert_eq!([column(&mut b, 10, 10), column(&mut b, 11, 10)], [0xffff_ffff, 0]);
    }

    /// Depth is kept a sample: a nearer rectangle and a farther one that share a pixel's edge each win the samples
    /// they cover, and a farther fragment does not overwrite a nearer sample. A clear sets every sample.
    #[test]
    fn depth_and_clears_are_per_sample() {
        use crate::dev::kona::defs::gl::*;
        let mut b = ms_window(4);
        command(&mut b, Grammar::Gl, ENABLE_CAPABILITY, &[DEPTH_TEST]);
        command(&mut b, Grammar::Gl, SET_DEPTH_TEST, &[LESS]);
        command(&mut b, Grammar::Gl, SET_COLOUR_CLEAR_VALUE, &[0, 0, 0.5f32.to_bits(), 1.0f32.to_bits()]);
        command(&mut b, Grammar::Gl, CLEAR_SELECTED_BUFFERS, &[COLOR_BUFFER_BIT | DEPTH_BUFFER_BIT]);
        assert_eq!(column(&mut b, 5, 5), 0xff00_0080, "a clear sets the samples and the word");
        // A white rectangle at z = 0 (nearer), then a red one behind it at z = 0.5 that reaches further right.
        let rect = |b: &mut KonaBackend, right: f32, z: f32| {
            command(b, Grammar::Gl, BEGIN_PRIMITIVE_INDEPENDENT_TRIANGLES, &[0]);
            for (x, y) in [(2.0, 2.0), (right, 2.0), (right, 20.0), (2.0, 2.0), (right, 20.0), (2.0, 20.0)] {
                command(b, Grammar::Gl, SUBMIT_VERTEX_3_COMPONENTS_SINGLE_FLOAT, &[ndc(x), ndc(y), z.to_bits()]);
            }
            command(b, Grammar::Gl, END_OPEN_PRIMITIVE, &[0]);
        };
        rect(&mut b, 10.875, 0.0);
        command(&mut b, Grammar::Gl, SET_CURRENT_COLOUR_4_COMPONENTS_SINGLE_FLOAT, &[1.0f32.to_bits(), 0, 0, 1.0f32.to_bits()]);
        rect(&mut b, 14.875, 0.5);
        // Column 5 stays white (the red is behind); column 10 has two white samples and two red ones that the
        // far rectangle won where the near one is not (samples 2 and 3): mean red (2 * 255 + 2 * 255 + 2) / 4 = 255,
        // green and blue (2 * 255 + 2) / 4 = 128; column 12 is red.
        assert_eq!(column(&mut b, 5, 10), 0xffff_ffff);
        assert_eq!(column(&mut b, 10, 10) & 0xffffff, 0xff8080);
        assert_eq!(column(&mut b, 12, 10) & 0xffffff, 0xff0000);
    }

    /// The sample mask keeps the samples its coverage word names, by pixel column: the half-coverage mask on 4
    /// samples is 0x33333333, nibble 3 (samples 0 and 1) a column: half the pixel.
    #[test]
    fn the_sample_mask_and_alpha_to_mask_thin_the_coverage() {
        use crate::dev::kona::defs::gl::*;
        let mut b = ms_window(4);
        command(&mut b, Grammar::Gl, SET_MULTISAMPLE_COVERAGE_MASK, &[0.5f32.to_bits(), 0]);
        command(&mut b, Grammar::Gl, ENABLE_CAPABILITY, &[SAMPLE_MASK]);
        rectangle(&mut b, 10.875);
        // Mask 0.5 selects the 15-16 entry: 0x33333333, samples 0 and 1 of every pixel: half of a whole pixel.
        assert_eq!(column(&mut b, 5, 10), 0xff80_8080);
        // Inverted, the other half.
        let mut b = ms_window(4);
        command(&mut b, Grammar::Gl, SET_MULTISAMPLE_COVERAGE_MASK, &[0.5f32.to_bits(), 1]);
        command(&mut b, Grammar::Gl, ENABLE_CAPABILITY, &[SAMPLE_MASK]);
        rectangle(&mut b, 10.875);
        assert_eq!(column(&mut b, 5, 10), 0xff80_8080);
        // Alpha to mask with alpha 0.5 lets half the samples through; alpha to one then makes the fragment opaque.
        let mut b = ms_window(4);
        command(&mut b, Grammar::Gl, SET_CURRENT_COLOUR_4_COMPONENTS_SINGLE_FLOAT, &[1.0f32.to_bits(), 1.0f32.to_bits(), 1.0f32.to_bits(), 0.5f32.to_bits()]);
        command(&mut b, Grammar::Gl, ENABLE_CAPABILITY, &[SAMPLE_ALPHA_TO_MASK]);
        command(&mut b, Grammar::Gl, ENABLE_CAPABILITY, &[SAMPLE_ALPHA_TO_ONE]);
        rectangle(&mut b, 10.875);
        assert_eq!(column(&mut b, 5, 10), 0xff80_8080);
        // A bad pattern is an invalid-enumerant error; a good one is taken.
        command(&mut b, Grammar::Gl, SET_MULTISAMPLE_PATTERN, &[0x1234]);
        assert_eq!(b.contexts.select(None).gl.error, 0x500);
        command(&mut b, Grammar::Gl, SET_MULTISAMPLE_PATTERN, &[0x80a3]);
    }

}

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
