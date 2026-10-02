//! Headless renderer that captures the composited REX3 framebuffer into a
//! shared buffer for egui to upload as a texture each frame.
//!
//! Default path: `CaptureRenderer` (CPU `SwCompositor`).
//! Set `IRIS_GUI_GL=1` for GPU `GlCompositor` + readback (faster for GL demos).

use iris::compositor::{Compositor, SwCompositor};
use iris::debug_overlay::DebugOverlay;
use iris::disp::{BarStats, Rex3Screen, StatusBar, StatusBarTexture};
use iris::gl_compositor::GlCompositor;
use iris::headless_gl::HeadlessGl;
use iris::dev::ng1::rex3::Renderer;
use parking_lot::{Mutex, MutexGuard};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// One captured frame: tightly-packed RGBA bytes plus pixel dimensions.
#[derive(Clone, Default)]
pub struct Frame {
    pub width: usize,
    pub height: usize,
    /// Length is `width * height * 4`. Pixel order: R, G, B, A.
    pub rgba: Vec<u8>,
    /// Bumped every time a new frame is captured; egui uses this to skip the
    /// texture upload when nothing has changed.
    pub seq: u64,
    /// First scanline included in `rgba` update (0 = full frame).
    pub dirty_y: u32,
    /// Number of scanlines updated (`0` or `height` = full frame).
    pub dirty_h: u32,
}

/// Shared latest-frame slot. The renderer writes; the GUI reads.
#[derive(Default, Clone)]
pub struct FrameSink {
    frame: Arc<Mutex<Frame>>,
    seq:   Arc<AtomicU64>,
}

impl FrameSink {
    pub fn new() -> Self { Self::default() }

    pub fn seq(&self) -> u64 { self.seq.load(Ordering::Acquire) }

    pub fn snapshot(&self) -> Frame { self.frame.lock().clone() }

    pub fn reset(&self) {
        *self.frame.lock() = Frame::default();
        self.seq.store(0, Ordering::Release);
    }

    fn lock(&self) -> MutexGuard<'_, Frame> { self.frame.lock() }
}

/// Headless `Renderer` that captures the composited frame into a `FrameSink`.
pub struct CaptureRenderer {
    sink:        FrameSink,
    seq:         u64,
    compositor:  SwCompositor,
    last_pixels: Vec<u32>,
}

impl CaptureRenderer {
    pub fn new(sink: FrameSink) -> Self {
        Self {
            sink,
            seq: 0,
            compositor: SwCompositor::new(),
            last_pixels: Vec::new(),
        }
    }
}

impl Renderer for CaptureRenderer {
    fn present(
        &mut self,
        screen:        &mut Rex3Screen,
        _overlay:       &mut DebugOverlay,
        _status:        &mut StatusBar,
        _sbtex:         &mut StatusBarTexture,
        _stats:         &BarStats,
        _need_readback: bool,
        live_fb_rgb:   Option<&[u32]>,
        live_fb_aux:   Option<&[u32]>,
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

        let needed = 2048 * 1024;
        if self.last_pixels.len() < needed {
            self.last_pixels.resize(needed, 0);
        }

        let src = screen.compositor_source_from(live_fb_rgb, live_fb_aux);
        self.compositor.compose_pixels(&src);
        drop(src);
        self.last_pixels.copy_from_slice(self.compositor.pixels());

        pack_sw_frame(&self.sink, &self.last_pixels, width, height, &mut self.seq);
    }

    fn resize(&mut self, _width: usize, _height: usize) {}
}

/// GPU compositor path: `GlCompositor` on a hidden GL context (REX3-Refresh thread).
pub struct GlCaptureRenderer {
    sink:       FrameSink,
    seq:        u64,
    headless:   HeadlessGl,
    compositor: GlCompositor,
    rgba:       Vec<u32>,
    last_pixels: Vec<u32>,
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
            last_pixels: Vec::new(),
        })
    }
}

impl Renderer for GlCaptureRenderer {
    fn present(
        &mut self,
        screen:        &mut Rex3Screen,
        _overlay:       &mut DebugOverlay,
        _status:        &mut StatusBar,
        _sbtex:         &mut StatusBarTexture,
        _stats:         &BarStats,
        _need_readback: bool,
        live_fb_rgb:   Option<&[u32]>,
        live_fb_aux:   Option<&[u32]>,
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

        let needed = 2048 * 1024;
        if self.last_pixels.len() < needed {
            self.last_pixels.resize(needed, 0);
        }
        if self.rgba.len() < needed {
            self.rgba.resize(needed, 0);
        }
        if screen.prebuilt {
            // GR2 hands over a finished frame (stride 2048, RGBA bytes):
            // no GPU round trip needed.
            self.last_pixels.copy_from_slice(&screen.fb_rgb[..needed]);
            pack_sw_frame(&self.sink, &self.last_pixels, width, height, &mut self.seq);
            return;
        }
        let src = screen.compositor_source_from(live_fb_rgb, live_fb_aux);
        let gl = &self.headless.gl;
        let _tex = self.compositor.compose(&src, gl);
        drop(src);
        self.compositor.readback_to_screen(&mut self.rgba, width, height, gl);
        self.last_pixels.copy_from_slice(&self.rgba[..needed]);

        pack_sw_frame(&self.sink, &self.last_pixels, width, height, &mut self.seq);
    }

    fn resize(&mut self, _width: usize, _height: usize) {}
}

impl Drop for GlCaptureRenderer {
    fn drop(&mut self) {
        self.compositor.destroy(&self.headless.gl);
    }
}

fn pack_sw_frame(sink: &FrameSink, pixels: &[u32], width: usize, height: usize, seq: &mut u64) {
    let needed = width * height * 4;
    let mut frame = sink.lock();
    if frame.rgba.len() != needed {
        frame.rgba = vec![0u8; needed];
    }
    frame.width  = width;
    frame.height = height;
    frame.dirty_y = 0;
    frame.dirty_h = height as u32;

    for y in 0..height {
        let src_row = &pixels[y * 2048..y * 2048 + width];
        let dst_row = &mut frame.rgba[y * width * 4..(y + 1) * width * 4];
        for (dst_px, &word) in dst_row.chunks_exact_mut(4).zip(src_row) {
            dst_px.copy_from_slice(&(word | 0xFF00_0000).to_le_bytes());
        }
    }

    *seq = seq.wrapping_add(1);
    frame.seq = *seq;
    drop(frame);
    sink.seq.store(*seq, Ordering::Release);
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
