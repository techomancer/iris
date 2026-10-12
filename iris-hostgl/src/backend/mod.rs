//! What depends on the platform's GL: making contexts, finding entry points,
//! and a render target the CPU can read without a readback.
//!
//! Everything else in the crate -- the call replay, drawables, the swap -- is
//! OpenGL itself (legacy/compatibility profile, EXT_framebuffer_object and
//! EXT_framebuffer_blit), so a backend implements this trait and nothing
//! more. Its GL entry points must be the ones the `gl` module links:
//! OpenGL.framework with CGL on the Mac, libGL (glvnd) with EGL on Linux.

use std::ffi::c_void;

#[cfg(target_os = "macos")]
pub mod cgl;
#[cfg(target_os = "linux")]
pub mod egl;

/// This platform's backend, or `None` if the host cannot make the contexts.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub fn native() -> Option<Box<dyn Backend>> {
    #[cfg(target_os = "macos")]
    return Some(Box::new(cgl::Cgl::new()));
    #[cfg(target_os = "linux")]
    return egl::EglBackend::new().map(|b| Box::new(b) as Box<dyn Backend>);
}

/// A GL context, opaque to everything but the backend that made it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ContextHandle(pub usize);

pub trait Backend: Send {
    /// A short name for logs.
    fn name(&self) -> &'static str;

    /// A new context with the fixed-function pipeline (legacy profile),
    /// sharing objects with `share` if given. Not made current.
    fn create_context(&mut self, share: Option<ContextHandle>) -> Option<ContextHandle>;

    /// Destroy a context. It is not current on this thread afterwards.
    fn destroy_context(&mut self, ctx: ContextHandle);

    /// Make `ctx` current on the calling thread (`None`: no context).
    fn make_current(&mut self, ctx: Option<ContextHandle>) -> bool;

    /// The context current on the calling thread.
    fn current(&self) -> Option<ContextHandle>;

    /// The address of GL entry point `name`, or null if the library has none.
    fn lookup(&self, name: &str) -> *const c_void;

    /// A `w` x `h` render target whose pixels the CPU can address directly,
    /// wrapped as a texture in the current context; `None` if the platform
    /// has no such thing (the swap then reads back with glReadPixels).
    fn new_surface(&mut self, w: i32, h: i32) -> Option<Box<dyn Surface>>;
}

/// A CPU-addressable render target (see [`Backend::new_surface`]).
pub trait Surface: Send + Sync {
    /// The texture that renders into it: (target, name).
    fn texture(&self) -> (u32, u32);

    /// Lend the pixels to `f`: BGRA bytes, `stride` bytes a row, rows in the
    /// texture's order (row 0 is texture row 0). False if they could not be
    /// mapped. The caller has flushed GL first.
    fn with_pixels(&self, f: &mut dyn FnMut(&[u8], usize)) -> bool;

    /// Delete the texture. A context of the share group that made it must be
    /// current; without one (the group is gone) just drop the surface.
    fn delete_texture(&mut self);
}
