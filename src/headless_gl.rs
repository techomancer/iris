//! Minimal hidden-window GL context for off-thread compositor capture (iris-gui).
//!
//! OpenGL calls run on the thread that creates this context (the display refresh thread).

use std::ffi::CString;
use std::num::NonZeroU32;

use glow::HasContext;
use glutin::config::{ConfigTemplateBuilder, GlConfig};
use glutin::context::{ContextAttributesBuilder, PossiblyCurrentContext};
use glutin::display::{Display, DisplayApiPreference, GetGlDisplay};
use glutin::prelude::*;
use glutin::surface::{GlSurface, PbufferSurface, Surface, SurfaceAttributesBuilder, SwapInterval};
use raw_window_handle::{
    AppKitDisplayHandle, RawDisplayHandle, WindowsDisplayHandle, XlibDisplayHandle,
};

/// Offscreen GL context for the compositor: a display, a pbuffer to anchor it
/// and the context itself. No window and no event loop — see [`HeadlessGl::new`]
/// for why the two are not available here.
pub struct HeadlessGl {
    pub gl: glow::Context,
    _display: Display,
    _context: PossiblyCurrentContext,
    _surface: Surface<PbufferSurface>,
}

impl HeadlessGl {
    /// Create an offscreen GL context. Returns None if the platform cannot init GL.
    ///
    /// This deliberately does **not** go through `glutin_winit::DisplayBuilder`,
    /// which needs a `winit::EventLoop`. iris-gui is an eframe app: eframe
    /// creates the one event loop the process is allowed, and winit enforces
    /// that with a process-global one-shot —
    ///
    /// ```ignore
    /// static EVENT_LOOP_CREATED: AtomicBool = AtomicBool::new(false);
    /// if EVENT_LOOP_CREATED.swap(true, Ordering::Relaxed) {
    ///     return Err(EventLoopError::RecreationAttempt);
    /// }
    /// ```
    ///
    /// so a second `EventLoop::new()` fails before GL is ever touched. That is
    /// why `IRIS_GUI_GL=1` always fell back to the CPU compositor here: it was
    /// never a GLX/EGL problem, and no amount of working drivers would have
    /// changed it.
    ///
    /// The builder only used the event loop for two things, and neither needs
    /// it: `glutin_display_handle()` to get a `RawDisplayHandle`, and
    /// `finalize_window()` to create a window. So ask for the platform's
    /// *default* display directly and render into a pbuffer instead — the
    /// compositor draws into its own framebuffer object and reads that back, so
    /// it never needed a window in the first place.
    pub fn new() -> Option<Self> {
        let display = unsafe { Display::new(default_display_handle(), api_preference()) }.ok()?;

        let template = ConfigTemplateBuilder::new()
            .with_alpha_size(8)
            .with_transparency(true)
            .build();

        let config = unsafe { display.find_configs(template) }
            .ok()?
            .reduce(|accum, config| {
                if config.num_samples() > accum.num_samples() { config } else { accum }
            })?;

        // The compositor renders into its own FBO, so the size of this pbuffer
        // is arbitrary; it exists only to have something to make the context
        // current on.
        let attrs = SurfaceAttributesBuilder::<PbufferSurface>::new()
            .build(NonZeroU32::new(16).unwrap(), NonZeroU32::new(16).unwrap());
        let surface = unsafe { display.create_pbuffer_surface(&config, &attrs) }.ok()?;

        let context_attributes = ContextAttributesBuilder::new().build(None);
        let not_current = unsafe { display.create_context(&config, &context_attributes) }.ok()?;
        let context = not_current.make_current(&surface).ok()?;
        let _ = surface.set_swap_interval(&context, SwapInterval::DontWait);

        let gl = unsafe {
            glow::Context::from_loader_function(|s| {
                display.get_proc_address(&CString::new(s).unwrap())
            })
        };

        Some(Self { gl, _display: display, _context: context, _surface: surface })
    }
}

impl Drop for HeadlessGl {
    fn drop(&mut self) {
        unsafe {
            self.gl.finish();
        }
    }
}

/// Ask for the platform's default display.
///
/// Leaving the connection out is a supported thing to do, not a shortcut:
/// `raw-window-handle` documents an Xlib handle with no `Display` pointer as
/// "request the default display when using EGL", which is exactly the offscreen
/// case. macOS and Windows need nothing at all — CGL has no display connection
/// and WGL loads `opengl32.dll` itself.
#[cfg(target_os = "linux")]
fn default_display_handle() -> RawDisplayHandle {
    RawDisplayHandle::Xlib(XlibDisplayHandle::new(None, 0))
}

#[cfg(target_os = "macos")]
fn default_display_handle() -> RawDisplayHandle {
    RawDisplayHandle::AppKit(AppKitDisplayHandle::new())
}

#[cfg(target_os = "windows")]
fn default_display_handle() -> RawDisplayHandle {
    RawDisplayHandle::Windows(WindowsDisplayHandle::new())
}

/// Which backend to build the display with.
///
/// EGL only on Linux: GLX would need a real Xlib `Display` pointer, which is
/// the thing we are deliberately not obtaining, and EGL is the one glutin
/// recommends there anyway.
#[cfg(target_os = "linux")]
fn api_preference() -> DisplayApiPreference {
    DisplayApiPreference::Egl
}

#[cfg(target_os = "macos")]
fn api_preference() -> DisplayApiPreference {
    DisplayApiPreference::Cgl
}

/// WGL without a window: enough for a pbuffer, though it will lack the
/// extensions that need one. If that ever proves too little, the CPU compositor
/// is still there behind `IRIS_GUI_GL`.
#[cfg(target_os = "windows")]
fn api_preference() -> DisplayApiPreference {
    DisplayApiPreference::Wgl(None)
}

// GL context is owned and used only on the creating thread (the display refresh thread).
unsafe impl Send for HeadlessGl {}

#[cfg(test)]
mod tests {
    use super::*;

    /// The regression this file exists for: `IRIS_GUI_GL=1` always fell back to
    /// the CPU compositor because the old code created a second winit
    /// `EventLoop`, which winit rejects process-globally. Building a context
    /// without one is the whole fix, so check that it actually happens.
    #[test]
    fn builds_a_context_without_an_event_loop() {
        let Some(hg) = HeadlessGl::new() else {
            // No GL on this host at all — that is a legitimate None, and the
            // caller already reports it. Skip rather than fail.
            if std::env::var("DISPLAY").is_err() && std::env::var("WAYLAND_DISPLAY").is_err() {
                return;
            }
            panic!("HeadlessGl::new() returned None on a host with a display");
        };
        unsafe {
            let version = hg.gl.get_parameter_string(glow::VERSION);
            assert!(!version.is_empty(), "no GL version string from the new context");
            println!("GL VERSION: {version}");
            println!("GL RENDERER: {}", hg.gl.get_parameter_string(glow::RENDERER));
        }
    }
}
