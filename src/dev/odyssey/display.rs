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
    /// Frames since the last present.
    idle: AtomicU32,
    /// The frame size last copied to the screen; (0, 0) before the first.
    shown: Mutex<(usize, usize)>,
    /// The machine's status-bar word: front-panel LEDs and the disk and
    /// network activity bits other devices set.
    heartbeat: Arc<AtomicU64>,
    /// The CPU's Compare matches, for the status bar's timer rate.
    fasttick: Arc<AtomicU64>,
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
            idle: AtomicU32::new(0),
            shown: Mutex::new((0, 0)),
            heartbeat,
            fasttick,
        }
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
        let mut screen = self.screen.lock();
        let w = width.min(FB_STRIDE);
        let h = height.min(screen.rgba.len() / FB_STRIDE);
        screen.width = w;
        screen.height = h;
        screen.prebuilt = true;
        screen.status_bar_only = false;
        for (i, src) in rows.chunks_exact(width).enumerate() {
            let y = first + i;
            if y >= h {
                break;
            }
            for (x, &rgb) in src[..w].iter().enumerate() {
                let rgba = 0xff00_0000 | (rgb >> 16) & 0xff | rgb & 0xff00 | (rgb & 0xff) << 16;
                screen.fb_rgb[y * FB_STRIDE + x] = rgba;
                screen.rgba[y * FB_STRIDE + x] = rgba;
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
                gfifo_pending: 0,
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
    /// The top eight bits of a lookup-RAM entry's ten-bit channels, as RGB.
    pub fn lut_rgb(&mut self, entry: u32) -> u32 {
        let word = if entry < ODSY_LUT_SIZE { self.lut[entry as usize] } else { 0 };

        (((word >> ODSY_LUT_RED_SHIFT) & 0xff) << 16) | (((word >> ODSY_LUT_GREEN_SHIFT) & 0xff) << 8) | ((word >> ODSY_LUT_BLUE_SHIFT) & 0xff)
    }

    /// One 8-bit RGB channel through the 1,536-entry gamma table: widened to 12
    /// bits as the X server widens it (c << 4 | c >> 4; the board's own widening
    /// is unknown), mapped to the table in three ranges -- v below 0x200, v / 2 +
    /// 0x100 below 0x400, else v / 4 + 0x200 -- and shown as the top eight bits of
    /// that entry's channel at shift. Inferred: that the table applies to RGB
    /// pixels at all (the blank zeroes it to blacken the screen).
    pub fn gamma(&mut self, c: u32, shift: u32) -> u32 {
        let v = (c << 4) | (c >> 4);
        let t = if v < 0x200 {
            v
        } else if v < 0x400 {
            v / 2 + 0x100
        } else {
            v / 4 + 0x200
        };

        (self.lut[(ODSY_GAMMA_BASE + t) as usize] >> shift) & 0xff
    }

    /// Pixel (x, y)'s main drawable id, bits 7:3 of its id-plane byte.
    pub fn main_id(&mut self, x: u32, y: u32) -> u32 {
        ((self.did[(y * SGI_ODYSSEY_WIDTH + x) as usize] as u32) >> ODSY_DID_MAIN_SHIFT) & (ODSY_DISPMODE_COUNT - 1)
    }

    /// The colormap group main drawable main_id shows through: bits 8:4 of its
    /// display mode. Below ODSY_GAMMA_GROUP the plane word is an index.
    pub fn display_group(&mut self, main_id: u32) -> u32 {
        let mode = self.dispmode[main_id as usize];

        (mode >> ODSY_DISPMODE_GROUP_SHIFT) & ODSY_DISPMODE_GROUP_MASK
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

impl Board {
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
    pub fn resolve_row(&mut self, y: u32, dst: &mut [u32]) {
        let width = self.scanout.mode.width.min(SGI_ODYSSEY_WIDTH);
        // Swaps latch under the board lock (retrace()), so the selection
        // holds for the row.
        let displayed = self.front.lock().displayed_buffer;

        for x in 0..width {
            let at = (y * SGI_ODYSSEY_WIDTH + x) as usize;
            let main_id = self.main_id(x, y);
            let word = self.buffer_load((displayed >> main_id) & 1, x as i64, y as i64);
            let group = self.display_group(main_id);

            let omode = self.overlay_mode[((self.did[at] as u32) & ODSY_DID_OVERLAY_MASK) as usize];
            let opixel = self.overlay[at];

            if (omode & ODSY_OVERLAY_MODE_ENABLE) != 0 && opixel != 0 {
                dst[x as usize] = self.lut_rgb(
                    256 * ((omode >> ODSY_OVERLAY_MODE_GROUP_SHIFT) & ODSY_OVERLAY_MODE_GROUP_MASK) + opixel as u32,
                );
                continue;
            }
            if group >= ODSY_GAMMA_GROUP {
                let mode = self.dispmode[main_id as usize];
                let r = display_channel(mode, word >> 24);
                let g = display_channel(mode, (word >> 16) & 0xff);
                let b = display_channel(mode, (word >> 8) & 0xff);

                dst[x as usize] = self.gamma(r, ODSY_LUT_RED_SHIFT) << 16
                    | self.gamma(g, ODSY_LUT_GREEN_SHIFT) << 8
                    | self.gamma(b, ODSY_LUT_BLUE_SHIFT);
                continue;
            }
            if group == ODSY_CMAP_UNVERIFIED_GROUP && !self.scanout_group25_reported {
                self.scanout_group25_reported = true;
                crate::dlog!(LogModule::Odyssey, "sgi-odyssey: colormap group 25 (0x1900) resolved as an index table; unverified");
            }
            dst[x as usize] = self.lut_rgb(if group != 0 { 256 * group + (word & 0xff) } else { word & ODSY_INDEX_MASK });
        }
        self.cursor_row(y, dst, width);
    }

    /// A vertical retrace, which Board::scanout_poll() runs at the mode's rate. The
    /// driver issues a swap by writing ODSY_SWAP_ARM, a toggle mask with one bit
    /// per window whose displayed buffer must flip at the next retrace, then waits
    /// for the retrace interrupt. At each retrace the board first latches the
    /// pending flips into the displayed-buffer bits and then raises the retrace
    /// source, whose vector is byte 23:16 of ODSY_ROUTE_VECTORS. The interrupt
    /// fires at every retrace once the routing is programmed, not only when a swap
    /// is armed; before the driver programs the routing nothing is delivered.
    /// Whether the board can mask this source is unknown and it is left unmasked.
    ///
    /// Approximation: the board's encoding of ODSY_DISPLAYED_BUFFER is inferred
    /// from the driver's compare, and the latch's timing within the retrace is
    /// unknown. The model latches at its retrace callback and repaints the whole
    /// frame from the new selection; a frame is never presented half from each
    /// buffer.
    pub fn retrace(&mut self) {
        let mut f = self.front.lock();
        let flips = f.swap_arm;
        let mut repaint = false;

        if f.swap_set != 0 || f.swap_clear != 0 {
            f.displayed_buffer = (f.displayed_buffer | f.swap_set) & !f.swap_clear;
            f.swap_set = 0;
            f.swap_clear = 0;
            repaint = true;
        }
        if flips != 0 {
            f.swap_arm = 0;
            f.displayed_buffer ^= flips;
            crate::dlog!(LogModule::Odyssey, "sgi_odyssey_swap_latch: arm {:#x} displayed {:#x}", flips, f.displayed_buffer);
            repaint = true;
        }
        crate::pf_instant!(crate::perfetto::events::ODSY_RETRACE, f.displayed_buffer);
        if f.routed() {
            f.deliver_routed(ODSY_ROUTE_VECTOR_RETRACE_SHIFT, "retrace", true);
        }
        drop(f);
        if repaint {
            self.scanout.invalidate_all();
        }
    }

    /// Every retrace due at `now_ns`, in turn (Scanout::next_retrace()).
    pub fn scanout_poll(&mut self, now_ns: i64) {
        while self.scanout.next_retrace(now_ns) {
            self.retrace();
        }
    }

    /// Resolve the dirty lines into the scanout's frame. Returns the lines
    /// resolved as a half-open span `first..end` covering every one of them,
    /// or None when nothing was dirty (the frame is unchanged).
    ///
    /// Scanout reads the planes under the board lock, which the board's thread
    /// holds for a whole slice, so no draw runs meanwhile. A processor's write
    /// of the displayed-buffer register reached only the front: repaint for it.
    pub fn scanout_update(&mut self) -> Option<(usize, usize)> {
        if std::mem::take(&mut self.front.lock().displayed_written) {
            self.scanout.invalidate_all();
        }
        if self.scanout.frame.is_empty() {
            return None;
        }
        let mut span: Option<(usize, usize)> = None;
        let h = self.scanout.mode.height as usize;
        let w = self.scanout.mode.width as usize;
        let mut row = vec![0u32; w];
        let mut y = 0usize;
        while y < h {
            if !self.scanout.dirty[y] {
                y += 1;
                continue;
            }
            let mut end = y;
            while end < h && self.scanout.dirty[end] {
                end += 1;
            }
            for line in y..end {
                self.resolve_row(line as u32, &mut row);
                self.scanout.frame[line * w..(line + 1) * w].copy_from_slice(&row);
            }
            self.scanout.dirty[y..end].fill(false);
            crate::dlog_dev!(LogModule::Odyssey, "sgi_scanout_update: lines {}-{}", y, end - 1);
            span = Some(span.map_or((y, end), |(a, b)| (a.min(y), b.max(end))));
            y = end;
        }
        span
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
