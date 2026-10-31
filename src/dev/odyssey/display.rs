//! What the board shows: the scanout's colour lookup and gamma, the retrace
//! with its swap latching, and the window iris presents the frame in.

use super::*;

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::cpu::mips_core::CyclesPtr;
use crate::dev::ng1::rex3::Renderer;
use crate::disp::{Rex3Screen, FB_STRIDE};

/// Idle frames between forced presents.
const IDLE_REFRESH: u32 = 6;

/// The window the board shows the guest: what `GfxDisplay` hands out.
pub struct Display {
    pub renderer: Mutex<Option<Box<dyn Renderer>>>,
    pub screen: Mutex<Rex3Screen>,
    pub cycles: Mutex<CyclesPtr>,
    /// The other CPUs' cycle counters and tick counts, for the status bar.
    pub(super) more_cpus: Mutex<Vec<(CyclesPtr, Arc<AtomicU64>)>>,
    pub screenshot: AtomicBool,
    pub frame: AtomicBool,
    local: Mutex<Option<Local>>,
    /// What the display keeps of the scanout: the presented frame, the
    /// retrace clock and the update in progress. Held for each frame.
    pub reader: Mutex<ScanoutReader>,
    /// Frames since the last present.
    idle: AtomicU32,
    /// The frame size last copied to the screen; (0, 0) before the first.
    shown: Mutex<(usize, usize)>,
    /// The machine's status-bar word: front-panel LEDs and the disk and
    /// network activity bits other devices set.
    heartbeat: Arc<AtomicU64>,
    /// The CPU's Compare matches, for the status bar's timer rate.
    fasttick: Arc<AtomicU64>,
    /// A frame runs with no board lock and no hold on the raster unit's
    /// planes (reader.rs). `IRIS_VPRO_DISPLAY=lockless` turns it on; the
    /// default is the locked frame, which runs the same code with the
    /// locks held around it.
    lockless: AtomicBool,
}

struct Local {
    overlay: crate::debug_overlay::DebugOverlay,
    status: crate::disp::StatusBar,
    texture: crate::disp::StatusBarTexture,
    size: (usize, usize),
}

/// The status-bar bits that stay lit until changed: the front-panel LEDs.
const PERSISTENT: u64 = crate::dev::ng1::rex3::Rex3::HB_LED_RED | crate::dev::ng1::rex3::Rex3::HB_LED_GREEN;

impl Display {
    /// The other CPUs' figures for the status bar, one per CPU after the first.
    fn more_cpu_stats(&self) -> [Option<(u64, u64)>; crate::disp::MORE_CPUS] {
        let more = self.more_cpus.lock();
        std::array::from_fn(|i| more.get(i).map(|(c, f)| (c.get(), f.load(Ordering::Relaxed))))
    }

    pub(super) fn new(heartbeat: Arc<AtomicU64>, fasttick: Arc<AtomicU64>) -> Self {
        Self {
            renderer: Mutex::new(None),
            screen: Mutex::new(Rex3Screen::new()),
            cycles: Mutex::new(CyclesPtr::dangling()),
            more_cpus: Mutex::new(Vec::new()),
            screenshot: AtomicBool::new(false),
            frame: AtomicBool::new(false),
            local: Mutex::new(None),
            reader: Mutex::new(ScanoutReader::default()),
            idle: AtomicU32::new(0),
            shown: Mutex::new((0, 0)),
            heartbeat,
            fasttick,
            lockless: AtomicBool::new(matches!(std::env::var("IRIS_VPRO_DISPLAY").as_deref(), Ok("lockless"))),
        }
    }

    pub fn lockless(&self) -> bool {
        self.lockless.load(Ordering::Relaxed)
    }

    pub fn set_lockless(&self, on: bool) {
        self.lockless.store(on, Ordering::Relaxed);
    }

    pub(super) fn shown(&self) -> (usize, usize) {
        *self.shown.lock()
    }

    /// Whether this frame is presented: when a line changed, when a
    /// screenshot or frame is wanted, or on every sixth idle frame, as IMPACT
    /// does, so a window or VNC client that missed a frame catches up.
    pub(super) fn due(&self, changed: bool) -> bool {
        let wanted = self.screenshot.load(Ordering::Acquire) || self.frame.load(Ordering::Acquire);
        let idle = self.idle.fetch_add(1, Ordering::Relaxed) + 1;
        if changed || wanted || idle >= IDLE_REFRESH {
            self.idle.store(0, Ordering::Relaxed);
            return true;
        }
        false
    }

    /// Convert `rows` (0x00RRGGBB lines of `width` pixels, the first being
    /// line `first` of a frame `height` lines tall) into the screen and
    /// present it. No rows: present the screen as it is.
    pub(super) fn present(&self, rows: &[u32], width: usize, height: usize, first: usize) {
        self.present_lines(rows.chunks_exact(width.max(1)).enumerate().map(|(i, row)| (first + i, row)), width, height);
    }

    /// As present(), for lines given with their numbers.
    pub(super) fn present_lines<'a>(&self, lines: impl Iterator<Item = (usize, &'a [u32])>, width: usize, height: usize) {
        let mut screen = self.screen.lock();
        let w = width.min(FB_STRIDE);
        let h = height.min(screen.rgba.len() / FB_STRIDE);
        screen.width = w;
        screen.height = h;
        screen.prebuilt = true;
        screen.status_bar_only = false;
        let planes = &mut *screen;
        for (y, src) in lines {
            if y >= h {
                continue;
            }
            // 0x00RRGGBB to opaque 0xAABBGGRR: a byte swap with alpha set,
            // written to both copies in one pass.
            let at = y * FB_STRIDE;
            let line = &mut planes.fb_rgb[at..at + w];
            let rgba = &mut planes.rgba[at..at + w];
            for ((d, e), &rgb) in line.iter_mut().zip(rgba.iter_mut()).zip(&src[..w]) {
                let v = 0xff00_0000 | rgb.swap_bytes() >> 8;
                *d = v;
                *e = v;
            }
        }
        *self.shown.lock() = (width, height);
        screen.frame_seq = screen.frame_seq.wrapping_add(1);
        let mut renderer = self.renderer.lock();
        if let Some(renderer) = renderer.as_mut() {
            let mut local = self.local.lock();
            let local = local.get_or_insert_with(|| Local {
                overlay: crate::debug_overlay::DebugOverlay::new(),
                status: crate::disp::StatusBar::new(),
                texture: crate::disp::StatusBarTexture::new(),
                size: (0, 0),
            });
            if local.size != (w, h) {
                renderer.resize(w, h);
                local.size = (w, h);
            }
            let stats = crate::disp::BarStats {
                now: std::time::Instant::now(),
                cycles: self.cycles.lock().get(),
                fasttick: self.fasttick.load(Ordering::Relaxed),
                more_cpus: self.more_cpu_stats(),
                decoded_delta: 0,
                l1i_hits: 0,
                l1i_fetches: 0,
                uncached: 0,
                // Activity bits clear once shown; the LEDs persist, as on IMPACT.
                hb: self.heartbeat.fetch_and(PERSISTENT, Ordering::Relaxed),
                count_hz: 0,
                gfifo_pending: display_fifo_pending(),
            };
            renderer.present(&mut screen, &mut local.overlay, &mut local.status, &mut local.texture, &stats, false, None, None);
        }
        self.screenshot.store(false, Ordering::Release);
        self.frame.store(false, Ordering::Release);
    }

    pub(super) fn stop(&self) {
        if let Some(renderer) = self.renderer.lock().as_mut() {
            renderer.stop();
        }
        self.local.lock().take();
    }
}

impl Board {
    /// Pixel (x, y)'s main drawable id, bits 7:3 of its id-plane byte.
    pub fn main_id(&mut self, x: u32, y: u32) -> u32 {
        ((self.did[(y * SGI_ODYSSEY_WIDTH + x) as usize] as u32) >> ODSY_DID_MAIN_SHIFT) & (ODSY_DISPMODE_COUNT - 1)
    }

    /// The colormap group main drawable main_id shows through: bits 8:4 of its
    /// display mode. Below ODSY_GAMMA_GROUP the plane word is an index.
    pub fn display_group(&mut self, main_id: u32) -> u32 {
        self.scanout.shared.regs.read().display_group(main_id)
    }

    /// A vertical retrace: the host interface latches the swaps armed and
    /// raises the retrace source (HostInterface::retrace()).
    #[cfg(test)]
    pub fn retrace(&mut self) {
        self.scanout.shared.retrace();
    }
}

/// The scanout resolve as the tests and the benchmark run it on a board
/// they hold, without a display thread: the same reader the display uses.
#[cfg(test)]
impl Board {
    fn test_reader(&mut self) -> &mut ScanoutReader {
        self.reader.attach(&self.scanout.shared);
        &mut self.reader
    }

    /// Stage, resolve and commit the dirty lines into `reader.frame`.
    pub fn scanout_update(&mut self) -> Option<(usize, usize)> {
        self.test_reader().update()
    }

    /// Take the raster unit's published rows into the test reader, holding
    /// the planes as the display does.
    pub fn test_take_rows(&mut self, h: &BoardLocks, displayed: u32) {
        let marks = self.scanout.marks.clone();
        let differ = &mut self.test_reader().differ;
        h.raster.planes().take_rows(displayed, differ, &marks);
    }

    pub fn scanout_stage(&mut self) {
        self.test_reader().stage();
    }

    pub fn scanout_resolve(&mut self) -> Option<(usize, usize)> {
        self.test_reader().resolve()
    }

    /// Resolve the colour buffers at scanout. Per pixel: the id plane's main
    /// drawable (bits 7:3) selects the buffer shown, by its displayed-buffer bit,
    /// and display mode 0x2900 + id, whose bits 8:4 are the group. Inferred: that
    /// the displayed-buffer bits index windows by main drawable id. A group at or
    /// above 26 is an RGB mode: each channel of the plane word goes through the
    /// gamma table. Below it the plane word is an index: lut[256 * group + (index
    /// & 0xff)], or lut[index & 0xfff] for group 0; no second gamma pass. Group 25
    /// resolves as an index table and is reported once as unverified. Over all of
    /// that, a pixel whose overlay id's mode is enabled and whose overlay pixel is
    /// not zero shows the overlay colormap entry instead.
    ///
    /// This is the per-pixel form, which any plane shape can use. resolve_row()
    /// is the same function for the colour planes the board allocates, and the
    /// tests hold the two to the same pixels.
    pub fn resolve_row_generic(&mut self, y: u32, dst: &mut [u32]) {
        let width = self.scanout.mode.width.min(SGI_ODYSSEY_WIDTH);
        let displayed = self.front.lock().displayed_buffer;
        let shared = self.scanout.shared.clone();

        self.raster_wait();
        let regs = shared.regs.read();
        if shared.planes.get().is_some_and(|view| view.resolve_generic(&regs, displayed, y, width, dst)) {
            shared.report_group25();
        }
    }

    pub(super) fn resolve_ctx(&mut self) -> ResolveCtx {
        let displayed = self.front.lock().displayed_buffer;

        ResolveCtx::new(&self.scanout.shared.regs.read(), displayed)
    }

    /// The bytes of the colour buffer b's line y, `width` pixels of four
    /// bytes, and whether they are stored big-endian. None unless the line is
    /// whole in a four-byte plane.
    fn colour_line(&self, b: u32, y: u32, width: usize) -> Option<(&[u8], bool)> {
        let p = self.buffer_plane(b)?;
        let at = usize::try_from(p.offset(0, y as i32, 4)).ok()?;

        if plane_bpp(p.format) != 4 || !p.contains(width as i32 - 1, y as i32) {
            return None;
        }
        Some((self.buffer_mem(b).bytes.get(at..)?.get(..width * 4)?, !p.is_le()))
    }

    /// resolve_row_generic() for the board's own colour planes, with the
    /// per-pixel decisions that depend only on a pixel's drawable id made once
    /// per update (ResolveCtx) and the planes read as lines of words.
    fn resolve_row_with(&mut self, ctx: &ResolveCtx, y: u32, dst: &mut [u32]) {
        let width = self.scanout.mode.width.min(SGI_ODYSSEY_WIDTH);
        let Some(line) = self.scanout_line(y, width as usize) else {
            return self.resolve_row_generic(y, dst);
        };
        let saw_unverified = resolve_line(ctx, line, dst);
        let shared = self.scanout.shared.clone();

        if saw_unverified {
            shared.report_group25();
        }
        shared.regs.read().cursor_row(y, dst, width);
    }

    /// The plane bytes a row's resolve reads, when the row takes
    /// resolve_line(): both colour buffers in one byte order, the id
    /// bytes and the overlay bytes, each `width` pixels.
    pub(super) fn scanout_line(&self, y: u32, w: usize) -> Option<ScanoutLine<'_>> {
        let at = y as usize * SGI_ODYSSEY_WIDTH as usize;
        let (l0, be0) = self.colour_line(0, y, w)?;
        let (l1, be1) = self.colour_line(1, y, w)?;
        if be0 != be1 {
            return None;
        }
        Some(ScanoutLine { colour: [l0, l1], be: be0, did: self.did.get(at..at + w)?, overlay: self.overlay.get(at..at + w)? })
    }

    /// Resolve the colour buffers at scanout; see resolve_row_generic().
    pub fn resolve_row(&mut self, y: u32, dst: &mut [u32]) {
        let ctx = self.resolve_ctx();

        self.resolve_row_with(&ctx, y, dst);
    }
}

/// The channel byte c of plane word 0xRRGGBBAA as RGB display mode `mode`
/// shows it: a 4- or 5-bit visual's channel keeps its top bits, repeated down
/// to fill the byte as a full-scale value. How the board itself stores a
/// narrow pixel is unknown.
pub fn display_channel(mode: u32, c: u32) -> u32 {
    match (mode >> ODSY_DISPMODE_WIDTH_SHIFT) & ODSY_DISPMODE_WIDTH_MASK {
        ODSY_DISPMODE_WIDTH_4 => (c & 0xf0) | (c >> 4),
        ODSY_DISPMODE_WIDTH_5 => (c & 0xf8) | (c >> 5),
        _ => c,
    }
}

/// What one main drawable id shows, decided once per update instead of per
/// pixel: which colour buffer, and how its plane word becomes a colour.
#[derive(Clone, Copy, Default)]
struct IdShow {
    /// The displayed colour buffer, 0 or 1.
    buf: usize,
    /// Whether the plane word is RGB through the gamma table.
    rgb: bool,
    /// An RGB mode's channel fold: (c & keep) | (c >> down).
    keep: u32,
    down: u32,
    /// An index mode: lut entry = base + (word & mask).
    base: u32,
    mask: u32,
    /// The mode is colour map group 25, reported once when shown.
    unverified: bool,
}

/// Everything a row's resolve reads outside the planes, gathered once from a
/// copy of the registers, so none of it can change meanwhile.
pub(super) struct ResolveCtx {
    ids: [IdShow; ODSY_DISPMODE_COUNT as usize],
    /// Per overlay id: whether it is enabled, and its entry base.
    overlay: [(bool, u32); ODSY_OVERLAY_IDS as usize],
    /// Every lookup-RAM entry as 0x00RRGGBB (DisplayRegs::lut_rgb()).
    rgb: Box<[u32; ODSY_LUT_SIZE as usize]>,
    /// DisplayRegs::gamma() of each 8-bit channel value, placed in its byte of
    /// the pixel: red, green, blue.
    gamma: Box<[[u32; 256]; 3]>,
}

impl ResolveCtx {
    /// What the registers `regs` make of each drawable, overlay id, lookup
    /// entry and gamma step, with colour buffer (displayed >> id) & 1
    /// showing for each drawable.
    pub(super) fn new(regs: &DisplayRegs, displayed: u32) -> ResolveCtx {
        let mut ids = [IdShow::default(); ODSY_DISPMODE_COUNT as usize];
        let mut overlay = [(false, 0u32); ODSY_OVERLAY_IDS as usize];
        let mut rgb = Box::new([0u32; ODSY_LUT_SIZE as usize]);
        let mut gamma = Box::new([[0u32; 256]; 3]);

        for (i, v) in rgb.iter_mut().enumerate() {
            *v = regs.lut_rgb(i as u32);
        }
        for c in 0..256u32 {
            gamma[0][c as usize] = regs.gamma(c, ODSY_LUT_RED_SHIFT) << 16;
            gamma[1][c as usize] = regs.gamma(c, ODSY_LUT_GREEN_SHIFT) << 8;
            gamma[2][c as usize] = regs.gamma(c, ODSY_LUT_BLUE_SHIFT);
        }
        for (id, show) in ids.iter_mut().enumerate() {
            let mode = regs.dispmode[id];
            let group = regs.display_group(id as u32);

            show.buf = ((displayed >> id) & 1) as usize;
            if group >= ODSY_GAMMA_GROUP {
                show.rgb = true;
                // display_channel() as a mask and a shift; a shift of 8 or
                // more leaves a byte's high part empty.
                (show.keep, show.down) = match (mode >> ODSY_DISPMODE_WIDTH_SHIFT) & ODSY_DISPMODE_WIDTH_MASK {
                    ODSY_DISPMODE_WIDTH_4 => (0xf0, 4),
                    ODSY_DISPMODE_WIDTH_5 => (0xf8, 5),
                    _ => (0xff, 8),
                };
            } else if group != 0 {
                show.base = 256 * group;
                show.mask = 0xff;
                show.unverified = group == ODSY_CMAP_UNVERIFIED_GROUP;
            } else {
                show.mask = ODSY_INDEX_MASK;
            }
        }
        for (id, o) in overlay.iter_mut().enumerate() {
            let mode = regs.overlay_mode[id];

            *o = (
                mode & ODSY_OVERLAY_MODE_ENABLE != 0,
                256 * ((mode >> ODSY_OVERLAY_MODE_GROUP_SHIFT) & ODSY_OVERLAY_MODE_GROUP_MASK),
            );
        }
        ResolveCtx { ids, overlay, rgb, gamma }
    }
}

/// A run of RGB pixels through the gamma table: `src` is their plane words
/// (`BE`: big-endian), `FOLD`: the mode has narrow channels to fold first.
#[inline(always)]
fn gamma_run<const BE: bool, const FOLD: bool>(ctx: &ResolveCtx, show: &IdShow, src: &[u8], out: &mut [u32]) {
    let [red, green, blue] = &*ctx.gamma;

    for (out, c) in out.iter_mut().zip(src.chunks_exact(4)) {
        let raw = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        let word = if BE { raw.swap_bytes() } else { raw };
        let (mut r, mut g, mut b) = (word >> 24, (word >> 16) & 0xff, (word >> 8) & 0xff);

        if FOLD {
            r = (r & show.keep) | (r >> show.down);
            g = (g & show.keep) | (g >> show.down);
            b = (b & show.keep) | (b >> show.down);
        }
        *out = red[r as usize & 0xff] | green[g as usize & 0xff] | blue[b as usize & 0xff];
    }
}

/// One row's plane bytes, as resolve_line() reads them.
pub(super) struct ScanoutLine<'a> {
    colour: [&'a [u8]; 2],
    /// The colour words are big-endian.
    be: bool,
    did: &'a [u8],
    overlay: &'a [u8],
}

/// resolve_row_generic() for the board's own colour planes, with the
/// per-pixel decisions that depend only on a pixel's drawable id made once
/// per update (ResolveCtx) and the planes read as lines of words. Whether a
/// pixel showed through colour map group 25.
pub(super) fn resolve_line(ctx: &ResolveCtx, line: ScanoutLine<'_>, dst: &mut [u32]) -> bool {
    let ScanoutLine { colour: lines, be: be0, did, overlay: ovl } = line;
    let w = did.len();
    let mut saw_unverified = false;
    let mut x = 0;

    // A window is a run of pixels with one id byte, so the lookups that
    // depend only on the id are made per run and the run is a plain loop.
    while x < w {
        let d = did[x];
        let end = x + 1 + did[x + 1..].iter().take_while(|&&v| v == d).count();
        let (enabled, obase) = ctx.overlay[(d as u32 & ODSY_DID_OVERLAY_MASK) as usize];
        let show = &ctx.ids[(d >> ODSY_DID_MAIN_SHIFT) as usize & (ODSY_DISPMODE_COUNT as usize - 1)];
        let out = &mut dst[x..end];
        let src = &lines[show.buf][x * 4..end * 4];
        let ov = &ovl[x..end];

        if show.rgb {
            match (be0, show.keep != 0xff) {
                (true, true) => gamma_run::<true, true>(ctx, show, src, out),
                (false, true) => gamma_run::<false, true>(ctx, show, src, out),
                (true, false) => gamma_run::<true, false>(ctx, show, src, out),
                (false, false) => gamma_run::<false, false>(ctx, show, src, out),
            }
        } else {
            for (out, c) in out.iter_mut().zip(src.chunks_exact(4)) {
                let raw = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                let word = if be0 { raw.swap_bytes() } else { raw };
                *out = ctx.rgb[(show.base + (word & show.mask)) as usize & (ODSY_LUT_SIZE as usize - 1)];
            }
            // Reported only if a pixel of the run is shown through it.
            saw_unverified |= show.unverified && (!enabled || ov.contains(&0));
        }
        if enabled {
            for (out, &o) in out.iter_mut().zip(ov) {
                if o != 0 {
                    *out = ctx.rgb[(obase + o as u32) as usize & (ODSY_LUT_SIZE as usize - 1)];
                }
            }
        }
        x = end;
    }
    saw_unverified
}

/// A frame's staged lines on their way to the presented frame
/// (ScanoutReader::take()).
pub struct ScanoutJob {
    pub(super) stage: ScanoutStage,
    pub(super) ctx: Option<ResolveCtx>,
    /// The frame's width, and the pixels of it the planes cover.
    pub(super) w: usize,
    pub(super) width: u32,
    /// The frame's size when the lines were staged.
    pub(super) frame: (u32, u32),
    /// The staged lines resolved, `w` pixels each.
    pub(super) rows: Vec<u32>,
    pub(super) saw_unverified: bool,
    /// ScanoutReader::commit() put the rows in the frame.
    pub(super) committed: bool,
}

impl ScanoutJob {
    /// The lines of a committed job, as (line, `w` pixels) in order; none
    /// before ScanoutReader::commit() or if it dropped them.
    pub fn lines(&self) -> impl Iterator<Item = (usize, &[u32])> {
        let committed = self.committed;
        self.stage.lines.iter().zip(self.rows.chunks_exact(self.w.max(1))).filter(move |_| committed).map(|(&(y, _), row)| (y as usize, row))
    }

    pub fn committed(&self) -> bool {
        self.committed
    }

    /// Resolve the staged lines. Needs nothing of the board: the display
    /// runs it with no lock held, while the front end and the raster unit
    /// go on.
    pub fn resolve(&mut self) {
        let (w, width) = (self.w, self.width as usize);
        if w == 0 {
            return;
        }
        // A row for every line, in order; a line not staged is filled in
        // by ScanoutReader::commit().
        self.rows.clear();
        self.rows.resize(self.stage.lines.len() * w, 0);
        let Some(ctx) = &self.ctx else { return };
        let mut at = 0;
        for (&(_, staged), out) in self.stage.lines.iter().zip(self.rows.chunks_exact_mut(w)) {
            if staged {
                let (line, next) = self.stage.line(at, width);
                at = next;
                self.saw_unverified |= resolve_line(ctx, line, out);
            }
        }
    }
}

/// The dirty lines of a frame's update, with the plane bytes of those that
/// resolve_line() resolves copied aside (ScanoutReader::stage()).
#[derive(Default)]
pub struct ScanoutStage {
    /// Each dirty line, and whether its bytes are staged; a line that is
    /// not was resolved at once.
    pub(super) lines: Vec<(u32, bool)>,
    /// Per staged line: colour buffer 0 and 1 words, id bytes, overlay
    /// bytes. Kept between frames for its allocation.
    pub(super) bytes: Vec<u8>,
    /// The staged lines' colour words are big-endian.
    pub(super) be: Vec<bool>,
}

impl ScanoutStage {
    /// Copy line y's plane bytes from the view, `w` pixels: both colour
    /// buffers, the id bytes and the overlay bytes. False, with nothing
    /// kept, unless the line is whole in four-byte planes of one byte order.
    pub(super) fn push_row(&mut self, view: &PlaneView, y: u32, w: usize) -> bool {
        let start = self.bytes.len();

        self.bytes.resize(start + 10 * w, 0);
        let (colour, rest) = self.bytes[start..].split_at_mut(8 * w);
        let (c0, c1) = colour.split_at_mut(4 * w);
        let (did, overlay) = rest.split_at_mut(w);
        let be = [view.copy_row(RowOf::Colour(0), y, c0), view.copy_row(RowOf::Colour(1), y, c1)];

        if let (Some(a), Some(b)) = (be[0], be[1]) {
            if a == b && view.copy_row(RowOf::Did, y, did).is_some() && view.copy_row(RowOf::Overlay, y, overlay).is_some() {
                self.be.push(a);
                return true;
            }
        }
        self.bytes.truncate(start);
        false
    }

    /// The staged line at byte `at`, `w` pixels wide, and where the next
    /// one starts.
    fn line(&self, at: usize, w: usize) -> (ScanoutLine<'_>, usize) {
        let b = &self.bytes[at..at + 10 * w];
        let index = at / (10 * w);
        let line = ScanoutLine {
            colour: [&b[..4 * w], &b[4 * w..8 * w]],
            be: self.be[index],
            did: &b[8 * w..9 * w],
            overlay: &b[9 * w..],
        };
        (line, at + 10 * w)
    }
}

impl crate::gfx_display::GfxDisplay for Display {
    fn renderer_slot(&self) -> &Mutex<Option<Box<dyn Renderer>>> {
        &self.renderer
    }
    fn screen(&self) -> &Mutex<Rex3Screen> {
        &self.screen
    }
    fn request_screenshot(&self) {
        self.screenshot.store(true, Ordering::Release);
    }
    fn screenshot_pending(&self) -> bool {
        self.screenshot.load(Ordering::Acquire)
    }
    fn cycles(&self) -> CyclesPtr {
        *self.cycles.lock()
    }
    fn request_frame(&self) {
        self.frame.store(true, Ordering::Release);
    }
    fn frame_pending(&self) -> bool {
        self.frame.load(Ordering::Acquire)
    }
}
