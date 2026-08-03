//! Minimal hidden-window GL context for off-thread compositor capture (iris-gui).
//!
//! OpenGL calls run on the thread that creates this context (REX3-Refresh).

use std::ffi::CString;

use glow::HasContext;
use glutin::config::{ConfigTemplateBuilder, GlConfig};
use glutin::context::{ContextAttributesBuilder, PossiblyCurrentContext};
use glutin::display::GetGlDisplay;
use glutin::prelude::*;
use glutin::surface::{GlSurface, Surface, SwapInterval, WindowSurface};
use glutin_winit::{DisplayBuilder, GlWindow};
use raw_window_handle::HasRawWindowHandle;
use winit::event_loop::EventLoop;
use winit::window::WindowAttributes;

/// Hidden 1×1 window + GL context for offscreen GlCompositor use.
pub struct HeadlessGl {
    pub gl: glow::Context,
    _event_loop: EventLoop<()>,
    _window: winit::window::Window,
    _context: PossiblyCurrentContext,
    _surface: Surface<WindowSurface>,
}

impl HeadlessGl {
    /// Create a hidden GL context. Returns None if the platform cannot init GL.
    pub fn new() -> Option<Self> {
        let event_loop = EventLoop::new().ok()?;
        let window_builder = WindowAttributes::default()
            .with_title("iris-headless-gl")
            .with_visible(false)
            .with_inner_size(winit::dpi::LogicalSize::new(1u32, 1u32));

        let template = ConfigTemplateBuilder::new()
            .with_alpha_size(8)
            .with_transparency(true);

        let display_builder = DisplayBuilder::new().with_window_attributes(Some(window_builder));
        let (window, gl_config) = display_builder
            .build(&event_loop, template, |configs| {
                configs.reduce(|accum, config| {
                    if config.num_samples() > accum.num_samples() { config } else { accum }
                }).unwrap()
            })
            .ok()?;

        let window = window?;
        let raw_window_handle = window.raw_window_handle().ok()?;
        let gl_display = gl_config.display();

        let context_attributes = ContextAttributesBuilder::new().build(Some(raw_window_handle));
        let not_current_gl_context = unsafe {
            gl_display.create_context(&gl_config, &context_attributes).ok()?
        };

        let attrs = window.build_surface_attributes(Default::default()).ok()?;
        let gl_surface = unsafe {
            gl_display.create_window_surface(&gl_config, &attrs).ok()?
        };

        let gl_context = not_current_gl_context.make_current(&gl_surface).ok()?;
        let _ = gl_surface.set_swap_interval(&gl_context, SwapInterval::DontWait);

        let gl = unsafe {
            glow::Context::from_loader_function(|s| {
                gl_display.get_proc_address(&CString::new(s).unwrap())
            })
        };

        Some(Self {
            gl,
            _event_loop: event_loop,
            _window: window,
            _context: gl_context,
            _surface: gl_surface,
        })
    }
}

impl Drop for HeadlessGl {
    fn drop(&mut self) {
        unsafe {
            self.gl.finish();
        }
    }
}

// GL context is owned and used only on the creating thread (REX3-Refresh).
unsafe impl Send for HeadlessGl {}
