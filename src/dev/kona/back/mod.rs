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
//! - `x`, `textport`: the X server's integer path and the PROM's console;
//!   `context`: the state each rendering context keeps.
//! - `display`: the tables the scan-out reads and the frame's composition.

pub mod display;
pub mod raster;
pub mod tiles;
pub mod store;
pub mod context;
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
use tiles::Tiles;
use store::texture::TextureStore;
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
    /// Test hooks: a back end test double that may answer a job.
    #[cfg(test)]
    pub(super) hook: Option<TestHook>,
    /// The tile scheduler, made when the first raster job comes so a board that
    /// never draws GL has no worker threads, and what set-up counted.
    tiles: Option<Tiles>,
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
            #[cfg(test)]
            hook: None,
            tiles: None,
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
        self.last_layout = None;
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
            ["help"] | [] => writeln!(w, "kona: status|counters|hash|pix X Y|shot FILE.png")?,
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

impl KonaBackend {

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

    /// A DMA packet: the rows of a pixel write that announced them, read a
    /// line at a time into words (big-endian, as the programmed-I/O path
    /// sends them). A packet with no write waiting is unclaimed. A read that
    /// fails on the bus faults the packet and draws nothing.
    fn dma_rows(&mut self, port: &mut BackPort<'_>, io: u64, physical: bool, lines: u32, per_line: u32, pitch: u64) -> Reply {
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
            Job::Command { grammar: Grammar::Gl, selector: CONFIGURE_FRAME_BUFFER_LAYOUT, args, .. } => self.buffer_config(args),
            Job::Command { grammar: Grammar::Textport, selector, args, .. } => {
                let (tiles, store) = (&mut self.tiles, &self.store);
                self.textport.command(|| raster::surface(tiles, store, textport::COLOUR_WORD), selector, args);
            }
            Job::Command { grammar: Grammar::Kernel, selector, args, .. } => {
                self.regions.note(selector, args);
                match (selector, args) {
                    // A swap: the window's id, the display id and the mode. The low bit of
                    // the display's mode byte is the back buffer being shown (the X server's
                    // case split, established); that this command's third word is that same
                    // byte is inferred from the recorded swaps, which set it to 0 and 1
                    // alternately.
                    (DISPLAY_MODE_BYTE, &[window, id, mode]) if window != 0 => {
                        self.tables.lock().load_bank(id, (mode & 1) as u8);
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
            Job::Context(e) => self.contexts.event(e),
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


    // ---- Queries, select and feedback ----


    // ---- Multisample ----

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
