//! Host OpenGL for IRIX programs under IRIS, answered through host call 3000.
//!
//! An IRIX OpenGL program is given a libGL of ours (the iris-guest-tools
//! project, github.com/atomchild411/iris-guest-tools, installed in place of
//! SGI's with the same SONAME). Its GL entry points do not trap one
//! at a time: each encodes itself into a command buffer, and the buffer comes
//! here in one host call ([`OP_BATCH`]) whenever a call has to reach the host
//! before it returns -- a result, a pointer written, pixel data or client
//! arrays read in place, a swap -- or the buffer is full. The wire format is
//! described in iris-guest-tools' `gl/glshim.h`; the per-entry-point decoder (`calls.rs`) and
//! the guest's encoders are generated from one description of IRIX's gl.h
//! (`tools/glshim.py`), so the two sides agree on every offset.
//!
//! Each call runs on the host's OpenGL: a legacy-profile context per GLX
//! context, drawing into a framebuffer object per drawable. A swap flips the
//! frame on the GPU and either presents it through [`iris_hostcall::display`]
//! -- the IMPACT board composites it into its framebuffer under the window --
//! or hands the pixels back for the program to put up with
//! XShmPutImage/XPutImage.
//!
//! The modules:
//! - `service`: the operations, who is calling, and how a call that needs a
//!   page is run again without running anything twice (read, run, write).
//! - `exec`: one command's arguments, byte order, pixel transfers, client
//!   arrays, feedback and selection.
//! - `emul`: the SGI extensions the host has not got, done in the fragment
//!   stage with shaders over the fixed-function state.
//! - `draw`: drawables and the swap.
//! - `threaded`: the service on its own thread, so the emulated CPU is not
//!   held while a command buffer runs.
//! - `backend`: what is platform-specific -- contexts, entry point lookup, a
//!   CPU-addressable render target. CGL on the Mac, EGL on Linux.
//! - `calls`, `ext`, `avail`: generated tables.
//!
//! macOS and Linux have backends. Elsewhere the crate builds and [`register`]
//! does nothing, so the host call keeps answering EINVAL and the library says
//! host GL is not available.

#[cfg(any(target_os = "macos", target_os = "linux"))]
mod accum;
#[cfg(target_os = "linux")]
mod avail;
mod backend;
#[cfg(any(target_os = "macos", target_os = "linux"))]
mod calls;
#[cfg(any(target_os = "macos", target_os = "linux"))]
mod draw;
#[cfg(any(target_os = "macos", target_os = "linux"))]
#[allow(dead_code)] // enumerants and entry points kept for the extensions they document
mod emul;
#[cfg(any(target_os = "macos", target_os = "linux"))]
mod exec;
#[cfg(any(target_os = "macos", target_os = "linux"))]
mod ext;
#[cfg(any(target_os = "macos", target_os = "linux"))]
mod gl;
#[cfg(any(target_os = "macos", target_os = "linux"))]
mod present;
pub mod qos;
#[cfg(any(target_os = "macos", target_os = "linux"))]
mod service;
#[cfg(any(target_os = "macos", target_os = "linux"))]
mod threaded;

#[cfg(all(test, any(target_os = "macos", target_os = "linux")))]
mod tests;

/// The generated decoder names its executor as `super::Exec`.
#[cfg(any(target_os = "macos", target_os = "linux"))]
use exec::Exec;

#[cfg(any(target_os = "macos", target_os = "linux"))]
pub use service::{
    GlService, OP_BATCH, OP_BATCH_ASYNC, OP_CREATE, OP_DESTROY, OP_DRAWABLE_GONE, OP_FINISH, OP_GETSTRING, OP_GLX_STRING, OP_GOODBYE, OP_HELLO,
    OP_MAKECURRENT, OP_PBUFFER, OP_RELEASE, OP_SWAP, OP_SWAP_INTERVAL, OP_VIDEO_SYNC, PROTOCOL,
};

/// Answer host call [`iris_hostcall::GL`] with a new host GL service.
///
/// Registering again replaces the service, destroying every context the old
/// one held -- which is what a guest reboot wants, since no process that
/// made them is left. The service runs on a thread of its own, which makes
/// and uses every context (threaded.rs); with `IRIS_HOSTGL_INLINE=1` it runs
/// on the thread that makes the calls (the CPU thread), so call this from that
/// thread or before it starts.
pub fn register() {
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    match backend::native() {
        Some(backend) => {
            let name = backend.name();
            let svc = GlService::new(backend);
            if std::env::var_os("IRIS_HOSTGL_INLINE").is_some() {
                iris_hostcall::register(iris_hostcall::GL, Box::new(svc));
                log::info!("host GL: service registered ({name} backend, on the CPU thread)");
            } else {
                iris_hostcall::register(iris_hostcall::GL, Box::new(threaded::Threaded::spawn(svc)));
                log::info!("host GL: service registered ({name} backend, on its own thread)");
            }
        }
        None => log::warn!("host GL: this host cannot make OpenGL contexts; host call {} stays unanswered", iris_hostcall::GL),
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    log::info!("host GL: no backend for this platform; host call {} stays unanswered", iris_hostcall::GL);
}

/// A texture name for this crate's own use -- the surface a frame is presented
/// through, the textures the SGI-extension emulation draws with.
///
/// Never from glGenTextures. An old GL program names its textures itself --
/// Quake binds 1, 2, 3 and so on without ever generating a name, which GL 1
/// allows -- and glGenTextures hands out exactly those small numbers to
/// whoever asks first. Names from a high range no program picks by hand cannot
/// collide, and a program that does generate names is still given ones the
/// host knows are free.
#[cfg(any(target_os = "macos", target_os = "linux"))]
pub(crate) fn internal_texture_name() -> u32 {
    static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0x7F00_0000);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}
