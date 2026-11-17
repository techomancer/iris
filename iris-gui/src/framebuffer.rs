//! Headless renderer that captures the composited REX3 framebuffer into a
//! shared buffer pool for egui to upload as a texture each frame.
//!
//! Default path: `CaptureRenderer` (CPU `SwCompositor`).
//! Set `IRIS_GUI_GL=1` for GPU `GlCompositor` + readback (faster for GL demos).
//!
//! ## Frame hand-off
//!
//! The REX3 refresh thread composes a frame and *publishes* it into the sink;
//! the egui thread *takes* it. A small pool of [`Frame`] buffers is recycled
//! between the two so a frame is never copied on the way through: the renderer
//! fills a free buffer and publishes it, the GUI takes that buffer, uploads it,
//! and hands it straight back. Before the pool, every frame was cloned out of a
//! mutex — a multi-megabyte memcpy per frame at 60 Hz, hundreds of MB/s of
//! copying that showed the GUI nothing it did not already have.
//!
//! ## Wake-ups
//!
//! Without a wake the GUI would have to poll for new frames. [`FrameSink::set_wake`]
//! installs an `egui::Context` that every publish calls `request_repaint()` on;
//! eframe proxies that call into the winit event loop, so the GUI sleeps until a
//! frame actually exists instead of running 60 layout+paint passes a second over
//! an unchanging screen. On a two-core host that is most of a core back.
//!
//! ## Damage tracking
//!
//! Each frame is diffed against the previous one (see [`RowDamage`]) and the
//! changed scanlines are reported as a band. egui uploads only that band through
//! `TextureHandle::set_partial`, which the GL backend turns into a
//! `tex_sub_image_2d` of just those rows. A settled IRIX desktop therefore costs
//! no texture upload at all; a blinking cursor costs one row.
//!
//! A band is only valid against whatever the GUI's texture currently holds,
//! which is always the last frame *published* — every published frame is taken
//! and uploaded before the next is composed. So [`RowDamage::remember`] runs only
//! on a successful publish, never when a frame is dropped because the GUI has
//! not taken the previous one yet.
//!
//! ## What the sink is not for
//!
//! A published frame holds only its damaged band, so the sink cannot answer
//! "give me the whole picture". Screenshots and VNC do not ask it: they read
//! `Rex3Screen::rgba`, which the renderer keeps current — band by band, and only
//! when `need_readback` says someone outside this process is looking (see
//! [`mirror_band`]). That is the difference between a frame that costs nothing
//! and one that costs a multi-megabyte copy.

use iris::compositor::{Compositor, SwCompositor};
use iris::debug_overlay::DebugOverlay;
use iris::disp::{BarStats, Rex3Screen, StatusBar, StatusBarTexture, FB_STRIDE};
use iris::gl_compositor::GlCompositor;
use iris::headless_gl::HeadlessGl;
use iris::dev::ng1::rex3::Renderer;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// One captured frame: tightly-packed RGBA bytes plus pixel dimensions.
///
/// A frame is only ever read by the GUI, whose texture already holds every row
/// the frame does not carry, so `rgba` is complete only when `dirty_h` covers
/// the whole picture (see [`pack_frame`]). A reader that needs the whole
/// picture must not use this — see the module's "What the sink is not for".
#[derive(Clone, Default)]
pub struct Frame {
    pub width: usize,
    pub height: usize,
    /// Length is `width * height * 4`. Pixel order: R, G, B, A.
    pub rgba: Vec<u8>,
    /// Bumped every time a new frame is captured; egui uses this to skip the
    /// texture upload when nothing has changed.
    pub seq: u64,
    /// First scanline included in this update (0 = full frame).
    pub dirty_y: u32,
    /// Number of scanlines updated (`0` or `height` = full frame).
    pub dirty_h: u32,
}

/// Buffers kept recycled between the renderer and the GUI. Two is enough — one
/// in the renderer's hands, one in the GUI's — with slack so a burst does not
/// force an allocation.
const POOL_BUFFERS: usize = 4;

/// Everything the two threads share, behind one lock. `published` is the frame
/// waiting for the GUI; `free` is everything else. The lock is never taken
/// recursively, so there is no order to get wrong.
#[derive(Default)]
struct Pool {
    free: Vec<Frame>,
    published: Option<Frame>,
}

struct FrameSinkInner {
    pool: Mutex<Pool>,
    seq: AtomicU64,
    /// egui context to wake when a frame lands. Installed once the `App` exists
    /// (the sink outlives it, being created with the worker thread).
    wake: Mutex<Option<egui::Context>>,
}

/// Shared latest-frame slot. The renderer publishes; the GUI takes.
#[derive(Clone)]
pub struct FrameSink {
    inner: Arc<FrameSinkInner>,
}

impl Default for FrameSink {
    fn default() -> Self { Self::new() }
}

impl FrameSink {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(FrameSinkInner {
                pool: Mutex::new(Pool::default()),
                seq: AtomicU64::new(0),
                wake: Mutex::new(None),
            }),
        }
    }

    pub fn seq(&self) -> u64 { self.inner.seq.load(Ordering::Acquire) }

    /// Install the context every publish wakes. Called once, from the GUI
    /// thread, as soon as the `App` has a context to give. Cheap to call again.
    pub fn set_wake(&self, ctx: &egui::Context) {
        *self.inner.wake.lock() = Some(ctx.clone());
    }

    /// Take the published frame, leaving the sink empty until the next publish.
    /// `None` when the GUI has already taken everything published so far — the
    /// cheap "is there anything new?" answer that avoids touching the pixels.
    pub fn take(&self) -> Option<Frame> {
        self.inner.pool.lock().published.take()
    }

    /// Hand a frame's buffer back for reuse. Its pixels are stale as far as the
    /// GUI is concerned; the renderer overwrites every row it publishes, so
    /// nothing has to be cleared here.
    pub fn recycle(&self, frame: Frame) {
        let mut pool = self.inner.pool.lock();
        if pool.free.len() < POOL_BUFFERS {
            pool.free.push(frame);
        }
    }

    /// Forget any frame in flight and rewind the sequence to 0, so the GUI's
    /// "have I seen this frame yet?" check fails and it shows the placeholder
    /// instead of the previous run's last screen. Called on `Cmd::Start`, where
    /// the old machine (and so the old renderer) is already gone.
    pub fn reset(&self) {
        let stale = {
            let mut pool = self.inner.pool.lock();
            let stale = pool.published.take();
            pool.free.clear();
            stale
        };
        // Drop the stale frame rather than recycling it: its buffer holds the
        // old run's picture, and there is no reason to carry megabytes of it
        // across a Stop/Start boundary.
        drop(stale);
        self.inner.seq.store(0, Ordering::Release);
    }

    /// A buffer to fill.
    fn take_free(&self) -> Frame {
        self.inner.pool.lock().free.pop().unwrap_or_default()
    }

    /// Publish a filled frame. If the GUI has not taken the previous one yet the
    /// new frame is stale by the time it would be shown, so it is dropped rather
    /// than queued: the display always wants the newest, and dropping keeps the
    /// GUI's texture in step with the last frame published (which is what makes a
    /// partial upload valid — see [`RowDamage`]). Returns whether the frame was
    /// published, so the caller knows whether to move its damage baseline.
    fn publish(&self, frame: Frame) -> bool {
        let seq = frame.seq;
        {
            let mut pool = self.inner.pool.lock();
            if pool.published.is_some() {
                // Hand the buffer straight back; only the band was written, and
                // every future fill rewrites the rows it publishes.
                if pool.free.len() < POOL_BUFFERS { pool.free.push(frame); }
                return false;
            }
            pool.published = Some(frame);
        }
        self.inner.seq.store(seq, Ordering::Release);
        // Wake the GUI: it is asleep until a frame exists. eframe forwards this
        // into the event loop, so it is safe (and cheap) from this thread.
        if let Some(ctx) = self.inner.wake.lock().clone() {
            ctx.request_repaint();
        }
        true
    }
}

/// Headless `Renderer` that captures the composited frame into a `FrameSink`.
pub struct CaptureRenderer {
    sink:       FrameSink,
    seq:        u64,
    compositor: SwCompositor,
    damage:     RowDamage,
}

impl CaptureRenderer {
    pub fn new(sink: FrameSink) -> Self {
        Self {
            sink,
            seq: 0,
            compositor: SwCompositor::new(),
            damage: RowDamage::new(),
        }
    }
}

impl Renderer for CaptureRenderer {
    fn present(
        &mut self,
        screen:         &mut Rex3Screen,
        _overlay:       &mut DebugOverlay,
        _status:        &mut StatusBar,
        _sbtex:         &mut StatusBarTexture,
        _stats:         &BarStats,
        need_readback:  bool,
        live_fb_rgb:    Option<&[u32]>,
        live_fb_aux:    Option<&[u32]>,
    ) {
        let _ = (live_fb_rgb, live_fb_aux);
        let width  = screen.width;
        let height = screen.height;
        if width == 0 || height == 0 { return; }

        // Heartbeat-only refresh: the compositor sets `status_bar_only` when
        // nothing in the guest display changed and only the emulator's own
        // status bar would. iris-gui shows its own status chrome and never bakes
        // that bar into the frame, so there's nothing to push here.
        if screen.status_bar_only && self.sink.seq() > 0 {
            return;
        }

        let src = screen.compositor_source_from(live_fb_rgb, live_fb_aux);
        self.compositor.compose_pixels(&src);
        drop(src);

        // Band of scanlines that changed since the frame the GUI is showing.
        // `None` = nothing moved at all: no pack, no publish, no wake. The GUI
        // keeps the texture it already has and goes back to sleep.
        let Some((dirty_y, dirty_h)) = self.damage.diff(self.compositor.pixels(), width, height)
            else { return };

        let mut frame = self.sink.take_free();
        pack_frame(&mut frame, self.compositor.pixels(), width, height, dirty_y, dirty_h);
        self.seq = self.seq.wrapping_add(1);
        frame.seq = self.seq;
        if self.sink.publish(frame) {
            // Only now does the GUI's texture come to match this frame.
            self.damage.remember(self.compositor.pixels(), width, height);
            if need_readback {
                mirror_band(screen, self.compositor.pixels(), width, height, dirty_y, dirty_h);
            }
        } else if need_readback {
            // The GUI is behind and this frame was dropped, so `screen.rgba`
            // still holds the last *published* frame — the same baseline the
            // band above was diffed against, which makes a banded mirror wrong.
            // Copy the whole frame instead: VNC must not stall behind a slow
            // GUI, and this is the rare path.
            mirror_band(screen, self.compositor.pixels(), width, height, 0, height as u32);
        }
    }

    fn resize(&mut self, _width: usize, _height: usize) {}

    #[cfg(feature = "wgpu")]
    fn takes_resident_readback_frames(&self) -> bool { true }
}

/// GPU compositor path: `GlCompositor` on a hidden GL context (REX3-Refresh thread).
pub struct GlCaptureRenderer {
    sink:       FrameSink,
    seq:        u64,
    headless:   HeadlessGl,
    compositor: GlCompositor,
    rgba:       Vec<u32>,
    damage:     RowDamage,
}

impl GlCaptureRenderer {
    pub fn new(sink: FrameSink) -> Option<Self> {
        let headless = HeadlessGl::new()?;
        eprintln!("iris-gui: OpenGL capture renderer enabled (IRIS_GUI_GL=1)");
        Some(Self {
            sink,
            seq: 0,
            headless,
            compositor: GlCompositor::new(),
            rgba: Vec::new(),
            damage: RowDamage::new(),
        })
    }
}

impl Renderer for GlCaptureRenderer {
    fn present(
        &mut self,
        screen:         &mut Rex3Screen,
        _overlay:       &mut DebugOverlay,
        _status:        &mut StatusBar,
        _sbtex:         &mut StatusBarTexture,
        _stats:         &BarStats,
        need_readback:  bool,
        live_fb_rgb:    Option<&[u32]>,
        live_fb_aux:    Option<&[u32]>,
    ) {
        let _ = (live_fb_rgb, live_fb_aux);
        let width  = screen.width;
        let height = screen.height;
        if width == 0 || height == 0 { return; }

        // See CaptureRenderer::present — no baked status bar, so a heartbeat-only
        // refresh has nothing to push.
        if screen.status_bar_only && self.sink.seq() > 0 {
            return;
        }

        let needed = FB_STRIDE * iris::disp::FB_MAX_H;
        if self.rgba.len() < needed {
            self.rgba.resize(needed, 0);
        }
        if screen.prebuilt {
            // GR2 hands over a finished frame (stride 2048, RGBA bytes):
            // no GPU round trip needed.
            self.rgba.copy_from_slice(&screen.fb_rgb[..needed]);
        } else {
            let src = screen.compositor_source_from(live_fb_rgb, live_fb_aux);
            let gl = &self.headless.gl;
            let _tex = self.compositor.compose(&src, gl);
            drop(src);
            self.compositor.readback_to_screen(&mut self.rgba, width, height, gl);
        }

        let Some((dirty_y, dirty_h)) = self.damage.diff(&self.rgba, width, height)
            else { return };

        let mut frame = self.sink.take_free();
        pack_frame(&mut frame, &self.rgba, width, height, dirty_y, dirty_h);
        self.seq = self.seq.wrapping_add(1);
        frame.seq = self.seq;
        if self.sink.publish(frame) {
            self.damage.remember(&self.rgba, width, height);
            if need_readback {
                mirror_band(screen, &self.rgba, width, height, dirty_y, dirty_h);
            }
        } else if need_readback {
            // Dropped: `screen.rgba` still holds the last published frame, so a
            // band diffed against that baseline would be wrong — mirror it all.
            mirror_band(screen, &self.rgba, width, height, 0, height as u32);
        }
    }

    fn resize(&mut self, _width: usize, _height: usize) {}

    #[cfg(feature = "wgpu")]
    fn takes_resident_readback_frames(&self) -> bool { true }
}

impl Drop for GlCaptureRenderer {
    fn drop(&mut self) {
        self.compositor.destroy(&self.headless.gl);
    }
}

/// Convert `pixels` (stride 2048, `0xFFBBGGRR`) into `frame.rgba` (tightly packed
/// RGBA bytes), filling only the band `dirty_y..dirty_y + dirty_h`.
///
/// Whether the band covers the whole picture decides what the GUI does with the
/// buffer: a full band is uploaded whole (and the buffer must be complete, which
/// it then is), a partial band goes through `set_partial` and the rows outside it
/// are never read. This mirrors `framebuffer_panel_single`'s own `partial` test
/// exactly, so the two cannot disagree about which rows matter.
fn pack_frame(
    frame:   &mut Frame,
    pixels:  &[u32],
    width:   usize,
    height:  usize,
    dirty_y: u32,
    dirty_h: u32,
) {
    let needed = width * height * 4;
    if frame.rgba.len() != needed {
        frame.rgba = vec![0u8; needed];
    }
    frame.width    = width;
    frame.height   = height;
    frame.dirty_y  = dirty_y;
    frame.dirty_h  = dirty_h;

    let first = dirty_y as usize;
    let last  = (first + dirty_h as usize).min(height);
    for y in first..last {
        let src_row = &pixels[y * FB_STRIDE..y * FB_STRIDE + width];
        let dst_row = &mut frame.rgba[y * width * 4..(y + 1) * width * 4];
        for (dst_px, &word) in dst_row.chunks_exact_mut(4).zip(src_row) {
            dst_px.copy_from_slice(&(word | 0xFF00_0000).to_le_bytes());
        }
    }
}

/// Copy the band `dirty_y..dirty_y + dirty_h` of a composed frame (stride
/// 2048, `0xAABBGGRR`) into `screen.rgba`, where everything outside this process
/// reads it: the VNC server's `grab_loop`, `fresh_frame`, and the screenshot
/// path. Same layout, same word format, so it is a row-wise copy — no
/// conversion, exactly like the `--ci` path in REX3's refresh loop.
///
/// Only run when `need_readback` says someone is actually reading. The GUI
/// never touches `screen.rgba`, so on the normal path this is the difference
/// between a free frame and a multi-megabyte copy per frame.
///
/// A band is only meaningful against whatever `screen.rgba` already holds, so
/// callers pass the band of the frame that was *published*: that is the frame
/// `screen.rgba` was last synced to, and the one the GUI's texture holds.
fn mirror_band(
    screen:   &mut Rex3Screen,
    pixels:   &[u32],
    width:    usize,
    height:   usize,
    dirty_y:  u32,
    dirty_h:  u32,
) {
    let first = dirty_y as usize;
    let last  = (first + dirty_h as usize).min(height);
    for y in first..last {
        let row = y * FB_STRIDE;
        screen.rgba[row..row + width].copy_from_slice(&pixels[row..row + width]);
    }
}

/// Row-damage tracker for one head's frame stream.
///
/// Holds the previous *published* frame and diffs each new one against it. The
/// GUI's texture always matches the last published frame — every published frame
/// is taken and uploaded before the next is composed — so a band of changed rows
/// is exactly the region the GUI still has to upload.
struct RowDamage {
    /// Previous frame, stride 2048 (same layout as `SwCompositor::pixels`).
    prev: Vec<u32>,
    w: usize,
    h: usize,
    /// `prev` holds a frame at `w`×`h`. False until the second frame, so the
    /// first comparison is a full-frame update rather than a bogus diff.
    primed: bool,
}

impl RowDamage {
    fn new() -> Self { Self { prev: Vec::new(), w: 0, h: 0, primed: false } }

    /// `(first_row, row_count)` of the band that differs from the previous
    /// frame, or `None` when nothing did. A full frame when there is no previous
    /// frame to compare against, or when the resolution changed.
    fn diff(&self, pixels: &[u32], w: usize, h: usize) -> Option<(u32, u32)> {
        if !self.primed || self.w != w || self.h != h {
            return Some((0, h as u32));
        }
        let mut first = usize::MAX;
        let mut last = 0usize;
        for y in 0..h {
            let a = &pixels[y * FB_STRIDE..y * FB_STRIDE + w];
            let b = &self.prev[y * FB_STRIDE..y * FB_STRIDE + w];
            if a != b {
                if first == usize::MAX { first = y; }
                last = y;
            }
        }
        if first == usize::MAX {
            None
        } else {
            Some((first as u32, (last - first + 1) as u32))
        }
    }

    /// Adopt `pixels` as the frame the GUI is now showing. Call only once the
    /// frame has actually been published: a dropped frame must not move the
    /// baseline, or the next band would be diffed against a screen the GUI never
    /// received.
    fn remember(&mut self, pixels: &[u32], w: usize, h: usize) {
        if self.prev.len() < pixels.len() {
            self.prev.resize(pixels.len(), 0);
        }
        self.prev.copy_from_slice(pixels);
        self.w = w;
        self.h = h;
        self.primed = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use iris::disp::{FB_MAX_H, MORE_CPUS};

    /// `screen.rgba` words are `0xAABBGGRR`, so R sits in the low byte — the
    /// opposite of the `0x00RRGGBB` a reader might expect.
    const RED:   u32 = 0x0000_00FF;
    const GREEN: u32 = 0x0000_FF00;
    const BLUE:  u32 = 0x00FF_0000;

    /// A screen whose composed frame is whatever the test writes into `fb_rgb`:
    /// `prebuilt` makes `compose_pixels` copy the rows straight through, so a
    /// test drives the picture without depending on mode decoding.
    fn prebuilt_screen(w: usize, h: usize) -> Rex3Screen {
        let mut s = Rex3Screen::new();
        s.width = w;
        s.height = h;
        s.prebuilt = true;
        s
    }

    fn fill_row(screen: &mut Rex3Screen, y: usize, word: u32) {
        let row = y * FB_STRIDE;
        for x in 0..screen.width {
            screen.fb_rgb[row + x] = word;
        }
    }

    fn fill_pixels_row(pixels: &mut [u32], y: usize, w: usize, word: u32) {
        let row = y * FB_STRIDE;
        for x in 0..w {
            pixels[row + x] = word;
        }
    }

    fn bar_stats() -> BarStats {
        BarStats {
            now:            std::time::Instant::now(),
            cycles:         0,
            fasttick:       0,
            more_cpus:      [None; MORE_CPUS],
            decoded_delta:  0,
            l1i_hits:       0,
            l1i_fetches:    0,
            uncached:       0,
            hb:             0,
            count_hz:       0,
            gfifo_pending:  0,
        }
    }

    /// `present` ignores the overlay, status bar and stats, but it wants them.
    struct Dummies {
        overlay: DebugOverlay,
        status:  StatusBar,
        sbtex:   StatusBarTexture,
        stats:   BarStats,
    }

    impl Dummies {
        fn new() -> Self {
            Self {
                overlay: DebugOverlay::new(),
                status:  StatusBar::new(),
                sbtex:   StatusBarTexture::new(),
                stats:   bar_stats(),
            }
        }
    }

    fn present(r: &mut CaptureRenderer, screen: &mut Rex3Screen, d: &mut Dummies) {
        r.present(screen, &mut d.overlay, &mut d.status, &mut d.sbtex, &d.stats, false, None, None);
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn capture_renderer_accepts_resident_readback_frames() {
        let renderer = CaptureRenderer::new(FrameSink::new());
        assert!(renderer.takes_resident_readback_frames());
    }

    /// Copy `w`×`h` of a strided picture, as the GUI's texture would end up
    /// holding after uploading a frame.
    fn copy_pic(dst: &mut [u32], src: &[u32], w: usize, h: usize) {
        for y in 0..h {
            let row = y * FB_STRIDE;
            dst[row..row + w].copy_from_slice(&src[row..row + w]);
        }
    }

    /// An independent row diff, so the assertions below are an oracle rather
    /// than a restatement of `RowDamage`.
    fn band_between(prev: &[u32], cur: &[u32], w: usize, h: usize) -> Option<(u32, u32)> {
        let mut first = None;
        let mut last = 0;
        for y in 0..h {
            let a = &cur[y * FB_STRIDE..y * FB_STRIDE + w];
            let b = &prev[y * FB_STRIDE..y * FB_STRIDE + w];
            if a != b {
                if first.is_none() { first = Some(y); }
                last = y;
            }
        }
        first.map(|f| (f as u32, (last - f + 1) as u32))
    }

    #[test]
    fn first_frame_is_full_then_a_band_then_nothing() {
        let sink = FrameSink::new();
        let mut r = CaptureRenderer::new(sink.clone());
        let mut d = Dummies::new();
        let mut screen = prebuilt_screen(64, 32);
        fill_row(&mut screen, 0, RED);

        present(&mut r, &mut screen, &mut d);
        let f = sink.take().expect("the first frame must publish");
        assert_eq!((f.width, f.height), (64, 32));
        // Nothing to diff against yet, so the whole picture has to go up.
        assert_eq!((f.dirty_y, f.dirty_h), (0, 32));
        assert_eq!(f.rgba.len(), 64 * 32 * 4);
        assert_eq!(&f.rgba[0..4], &[0xFF, 0x00, 0x00, 0xFF], "R in the low byte, A forced on");
        sink.recycle(f);

        // Two rows apart: one band covering both, not two uploads.
        fill_row(&mut screen, 3, GREEN);
        fill_row(&mut screen, 20, GREEN);
        present(&mut r, &mut screen, &mut d);
        let f = sink.take().expect("a changed row must publish");
        assert_eq!((f.dirty_y, f.dirty_h), (3, 18));
        sink.recycle(f);

        // Nothing moved: no pack, no publish, and the GUI goes back to sleep.
        present(&mut r, &mut screen, &mut d);
        assert!(sink.take().is_none(), "an unchanged frame must not publish");
        assert_eq!(sink.seq(), 2);
    }

    /// The invariant the whole band path rests on: a dropped frame must not
    /// move the damage baseline, or the rows it carried would never reach the
    /// GUI's texture.
    #[test]
    fn a_dropped_frame_does_not_move_the_baseline() {
        let sink = FrameSink::new();
        let mut r = CaptureRenderer::new(sink.clone());
        let mut d = Dummies::new();
        let mut screen = prebuilt_screen(64, 32);
        fill_row(&mut screen, 0, RED);

        present(&mut r, &mut screen, &mut d);

        // A second frame arrives before the GUI looks. The display always wants
        // the newest, so this one is dropped rather than queued — which leaves
        // the first frame waiting, not the second.
        fill_row(&mut screen, 5, GREEN);
        present(&mut r, &mut screen, &mut d);
        let held = sink.take().expect("the dropped frame leaves the previous one waiting");
        assert_eq!((held.dirty_y, held.dirty_h), (0, 32), "it is still the first, full frame");
        sink.recycle(held);

        // Row 7 changes too. The band has to cover both, because the GUI's
        // texture still holds the frame from before row 5 turned green: had the
        // dropped frame moved the baseline, row 5 would never be uploaded.
        fill_row(&mut screen, 7, BLUE);
        present(&mut r, &mut screen, &mut d);
        let f = sink.take().expect("the third frame must publish");
        assert_eq!((f.dirty_y, f.dirty_h), (5, 3));
    }

    #[test]
    fn a_resolution_change_forces_a_full_frame() {
        let sink = FrameSink::new();
        let mut r = CaptureRenderer::new(sink.clone());
        let mut d = Dummies::new();
        let mut screen = prebuilt_screen(64, 32);
        fill_row(&mut screen, 0, RED);

        present(&mut r, &mut screen, &mut d);
        assert_eq!(sink.take().map(|f| (f.dirty_y, f.dirty_h)), Some((0, 32)));

        // A band is only valid against a texture of the same size, so a resize
        // starts over rather than trying to patch rows that moved.
        screen.width = 128;
        screen.height = 64;
        present(&mut r, &mut screen, &mut d);
        let f = sink.take().expect("the first frame at the new size must publish");
        assert_eq!((f.dirty_y, f.dirty_h), (0, 64));
        assert_eq!(f.rgba.len(), 128 * 64 * 4);
    }

    #[test]
    fn a_heartbeat_refresh_publishes_nothing() {
        let sink = FrameSink::new();
        let mut r = CaptureRenderer::new(sink.clone());
        let mut d = Dummies::new();
        let mut screen = prebuilt_screen(64, 32);
        fill_row(&mut screen, 0, RED);

        present(&mut r, &mut screen, &mut d);
        assert!(sink.take().is_some());

        // The compositor says only its own status bar would change. iris-gui
        // draws its own chrome and never bakes that bar into a frame, so there
        // is nothing to push — even though the guest framebuffer did move.
        screen.status_bar_only = true;
        fill_row(&mut screen, 9, GREEN);
        present(&mut r, &mut screen, &mut d);
        assert!(sink.take().is_none(), "a heartbeat must not publish");
        assert_eq!(sink.seq(), 1);
    }

    /// Drives the renderer through a scripted mix of idle frames, damage and a
    /// publish the GUI is too slow to take, and checks every band against an
    /// independent diff of what the GUI's texture holds.
    #[test]
    fn every_band_covers_exactly_what_the_gui_still_needs() {
        let sink = FrameSink::new();
        let mut r = CaptureRenderer::new(sink.clone());
        let mut d = Dummies::new();
        let mut screen = prebuilt_screen(64, 32);
        let (w, h) = (64, 32);

        // `baseline` is the picture the renderer last published — the frame its
        // damage tracker diffs against. `shown` is the picture the GUI's texture
        // holds. They are equal whenever the GUI has taken everything published,
        // which is the only state in which a band means anything.
        let mut baseline = vec![0u32; FB_STRIDE * FB_MAX_H];
        let mut shown    = vec![0u32; FB_STRIDE * FB_MAX_H];
        let mut gui_primed = false;
        let mut dropped = 0;

        // The GUI takes nothing on this step, so the publish that follows it is
        // dropped rather than queued. The last entry re-fills a row with the
        // colour it already has: a step with nothing to do, so the GUI gets to
        // take the final frame without a new one overtaking it.
        let slow = 2usize;
        let edits: &[(usize, u32)] =
            &[(0, RED), (5, GREEN), (6, GREEN), (20, BLUE), (9, RED), (0, RED)];

        for (i, &(row, word)) in edits.iter().enumerate() {
            // The GUI takes the frame the renderer published, unless it is behind.
            if i != slow {
                if let Some(f) = sink.take() {
                    let want = if gui_primed {
                        band_between(&shown, &baseline, w, h)
                    } else {
                        Some((0, h as u32))
                    };
                    assert_eq!(
                        Some((f.dirty_y, f.dirty_h)), want,
                        "step {i}: the frame the GUI takes must cover what its texture still needs"
                    );
                    copy_pic(&mut shown, &baseline, w, h);
                    gui_primed = true;
                    sink.recycle(f);
                }
            }

            fill_row(&mut screen, row, word);
            let before = sink.seq();
            present(&mut r, &mut screen, &mut d);

            if sink.seq() != before {
                // Published, so this is now what every band is diffed against.
                copy_pic(&mut baseline, &screen.fb_rgb, w, h);
            } else if i == slow {
                // The GUI took nothing last step, so the sink still holds that
                // frame and this one had to be dropped — and the baseline must
                // not have moved with it.
                dropped += 1;
            }
        }

        assert!(dropped > 0, "the drop path must have been exercised");

        // Row 6 went green on a frame the GUI never took, and row 20 went blue
        // on a frame that was dropped outright. Both must have arrived.
        assert_eq!(
            band_between(&shown, &screen.fb_rgb, w, h), None,
            "the GUI's texture must end up holding the whole picture"
        );
        assert_eq!(
            band_between(&baseline, &screen.fb_rgb, w, h), None,
            "the renderer's baseline must end up matching the picture"
        );
    }

    #[test]
    fn pack_frame_writes_only_the_band() {
        let mut frame = Frame::default();
        // A recycled buffer still holds the previous picture.
        frame.rgba = vec![0xAA; 64 * 32 * 4];
        let mut pixels = vec![0u32; FB_STRIDE * FB_MAX_H];
        fill_pixels_row(&mut pixels, 4, 64, RED);
        fill_pixels_row(&mut pixels, 5, 64, RED);

        pack_frame(&mut frame, &pixels, 64, 32, 4, 2);

        assert_eq!((frame.dirty_y, frame.dirty_h), (4, 2));
        assert_eq!(frame.rgba.len(), 64 * 32 * 4, "the buffer is reused, not reallocated");
        assert!(frame.rgba[..4 * 64 * 4].iter().all(|&b| b == 0xAA), "rows above the band are left alone");
        assert!(frame.rgba[6 * 64 * 4..].iter().all(|&b| b == 0xAA), "rows below the band are left alone");
        assert_eq!(&frame.rgba[4 * 64 * 4..4 * 64 * 4 + 4], &[0xFF, 0x00, 0x00, 0xFF]);
    }

    #[test]
    fn mirror_band_writes_only_the_band() {
        let mut screen = Rex3Screen::new();
        screen.width = 64;
        screen.height = 32;
        let mut pixels = vec![0u32; FB_STRIDE * FB_MAX_H];
        fill_pixels_row(&mut pixels, 1, 64, RED);
        fill_pixels_row(&mut pixels, 2, 64, RED);

        // VNC reads `screen.rgba`, which the GUI never touches, so only the
        // published band may be rewritten.
        mirror_band(&mut screen, &pixels, 64, 32, 2, 1);
        assert_eq!(screen.rgba[2 * FB_STRIDE], RED);
        assert_eq!(screen.rgba[1 * FB_STRIDE], 0, "row 1 was not in the band");

        // The dropped-frame path mirrors the whole frame instead: a band would
        // be diffed against a screen the GUI never received.
        mirror_band(&mut screen, &pixels, 64, 32, 0, 32);
        assert_eq!(screen.rgba[1 * FB_STRIDE], RED, "a full mirror brings row 1 too");
        assert_eq!(screen.rgba[3 * FB_STRIDE], 0, "rows the frame does not carry stay as they were");
    }

    #[test]
    fn the_sink_drops_a_frame_rather_than_queueing_it() {
        let sink = FrameSink::new();
        assert!(sink.publish(Frame { seq: 1, ..Frame::default() }));
        assert!(!sink.publish(Frame { seq: 2, ..Frame::default() }),
                "the display wants the newest, so the older of two is dropped");
        assert_eq!(sink.seq(), 1, "a dropped frame does not bump the sequence");
        assert_eq!(sink.take().map(|f| f.seq), Some(1));
        assert!(sink.take().is_none());
    }

    #[test]
    fn reset_rewinds_the_sequence_and_forgets_the_frame() {
        let sink = FrameSink::new();
        assert!(sink.publish(Frame { seq: 7, ..Frame::default() }));
        assert_eq!(sink.seq(), 7);

        sink.reset();

        assert_eq!(sink.seq(), 0, "the GUI's \"have I seen this yet?\" check must fail again");
        assert!(sink.take().is_none(), "the old run's last screen must not survive a Start");
    }

    #[test]
    fn a_recycled_buffer_is_reused_rather_than_reallocated() {
        let sink = FrameSink::new();
        let mut f = Frame::default();
        f.rgba = vec![0u8; 4096];
        sink.recycle(f);

        // What the renderer calls to get a buffer to fill: it must be the one
        // that came back, or every frame would cost a fresh allocation.
        assert_eq!(sink.take_free().rgba.len(), 4096);
    }
}

/// Pick CPU or GL capture renderer based on `IRIS_GUI_GL=1`.
pub fn new_capture_renderer(sink: FrameSink) -> Box<dyn Renderer> {
    if std::env::var("IRIS_GUI_GL").map(|v| v == "1").unwrap_or(false) {
        if let Some(r) = GlCaptureRenderer::new(sink.clone()) {
            return Box::new(r);
        }
        eprintln!("iris-gui: IRIS_GUI_GL=1 but headless GL init failed — using CPU compositor");
    }
    Box::new(CaptureRenderer::new(sink))
}
