//! What a front-end installs into a head to put frames on the screen.

use crate::disp::Rex3Screen;
#[cfg(feature = "wgpu")]
use std::sync::Arc;

pub trait Renderer: Send {
    /// Composite and present one display frame.
    ///
    /// The renderer owns its compositor and drives the full pipeline:
    /// compose → overlay → status bar → swap. `bar` is the status bar,
    /// already drawn by the display layer (`crate::status::StatusBar`):
    /// `STATUS_BAR_HEIGHT` rows of `STRIP_STRIDE` 0xAABBGGRR pixels, the
    /// display's width of them drawn. The renderer only shows it.
    /// When `need_readback` is true the renderer must populate `screen.rgba`
    /// with the composited pixels (for screenshots) before returning.
    fn present(
        &mut self,
        screen:        &mut crate::disp::Rex3Screen,
        overlay:       &mut crate::debug_overlay::DebugOverlay,
        bar:           &[u32],
        need_readback: bool,
        live_fb_rgb:   Option<&[u32]>,
        live_fb_aux:   Option<&[u32]>,
    );

    fn resize(&mut self, _width: usize, _height: usize) {}
    fn stop(&mut self) {}
    /// Switch to GL or SW compositor at runtime. Returns the active compositor name.
    fn switch_compositor(&mut self, _use_gl: bool) -> &'static str { "sw" }
    /// Return a short status string: "compositor=gl shader=integer" etc.
    fn compositor_status(&self) -> String { "compositor=sw shader=n/a".to_string() }
    /// Whether the GL compositor is in use, or is about to be.
    fn gl_compositor(&self) -> bool { false }

    /// Whether `present_gpu_frame` presents frames that are already on the
    /// board's GPU (the wgpu renderer, on the board's own device). A board
    /// that composes on the GPU then skips its readback.
    #[cfg(feature = "wgpu")]
    fn takes_gpu_frames(&self) -> bool { false }

    /// Present a frame of 0x00RRGGBB words (rows of `width`) that is on the
    /// board's GPU, with `screen` giving the overlay's sources; the status
    /// bar and overlay as `present` draws them. `screen.rgba` is not updated.
    #[cfg(feature = "wgpu")]
    #[allow(clippy::too_many_arguments)]
    fn present_gpu_frame(
        &mut self,
        _frame:   &wgpu::Buffer,
        _width:   u32,
        _height:  u32,
        _screen:  &mut crate::disp::Rex3Screen,
        _overlay: &mut crate::debug_overlay::DebugOverlay,
        _bar:     &[u32],
    ) {}

    #[cfg(all(test, feature = "wgpu"))]
    fn resident_presented_frames(&self) -> Option<u64> { None }

    #[cfg(all(test, feature = "wgpu"))]
    fn resident_presented_texture(&self) -> Option<&wgpu::Texture> { None }
}
