//! EGL with desktop OpenGL: headless contexts in the compatibility profile,
//! on Linux.
//!
//! The contexts have no surface at all (EGL_KHR_surfaceless_context): every
//! drawable is a framebuffer object, so the service never needs a window
//! system, and an X or Wayland session is not involved. The display is
//! Mesa's surfaceless platform where there is one, else the first EGL device,
//! else the default display.
//!
//! libEGL is opened at run time, so building needs nothing of it; the GL
//! entry points the crate calls by name come from libGL (glvnd), which
//! dispatches to whichever context EGL made current.
//!
//! Entry points looked up by name need care. glvnd and Mesa both answer
//! *any* `gl*` name with a callable stub, so a missing function never comes
//! back null, and the service depends on null to fall back from an SGI or EXT
//! name to the core one. So a name is only looked up once the context's
//! version or extensions say it exists (`avail.rs`, from the registry).
//!
//! No CPU-addressable surface: a swap reads the frame back with glReadPixels.

use std::collections::HashSet;
use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::{Mutex, OnceLock};

use super::{Backend, ContextHandle, Surface};

type EglDisplay = *mut c_void;
type EglConfig = *mut c_void;
type EglContext = *mut c_void;

const EGL_NONE: i32 = 0x3038;
const EGL_EXTENSIONS: i32 = 0x3055;
const EGL_SURFACE_TYPE: i32 = 0x3033;
const EGL_PBUFFER_BIT: i32 = 0x0001;
const EGL_RENDERABLE_TYPE: i32 = 0x3040;
const EGL_OPENGL_BIT: i32 = 0x0008;
const EGL_OPENGL_API: u32 = 0x30A2;
const EGL_PLATFORM_SURFACELESS_MESA: u32 = 0x31DD;
const EGL_PLATFORM_DEVICE_EXT: u32 = 0x313F;

const GL_VERSION: u32 = 0x1F02;
const GL_EXTENSIONS: u32 = 0x1F03;

/// libEGL's entry points, opened once for the process.
struct Egl {
    get_proc_address: unsafe extern "C" fn(*const c_char) -> *const c_void,
    get_display: unsafe extern "C" fn(*mut c_void) -> EglDisplay,
    initialize: unsafe extern "C" fn(EglDisplay, *mut i32, *mut i32) -> u32,
    query_string: unsafe extern "C" fn(EglDisplay, i32) -> *const c_char,
    bind_api: unsafe extern "C" fn(u32) -> u32,
    choose_config: unsafe extern "C" fn(EglDisplay, *const i32, *mut EglConfig, i32, *mut i32) -> u32,
    create_context: unsafe extern "C" fn(EglDisplay, EglConfig, EglContext, *const i32) -> EglContext,
    destroy_context: unsafe extern "C" fn(EglDisplay, EglContext) -> u32,
    make_current: unsafe extern "C" fn(EglDisplay, *mut c_void, *mut c_void, EglContext) -> u32,
    get_current_context: unsafe extern "C" fn() -> EglContext,
    get_error: unsafe extern "C" fn() -> i32,
}

/// The display and the one config every context is made with (contexts that
/// share objects must agree on it).
struct Display {
    egl: Egl,
    dpy: EglDisplay,
    config: EglConfig,
}

// SAFETY: an EGLDisplay and EGLConfig are process-wide handles, usable from
// any thread; the function pointers are plain C functions.
unsafe impl Send for Display {}
unsafe impl Sync for Display {}

fn has(list: *const c_char, ext: &str) -> bool {
    // SAFETY: EGL returns a NUL-terminated string or null.
    !list.is_null() && unsafe { CStr::from_ptr(list) }.to_bytes().split(|&b| b == b' ').any(|e| e == ext.as_bytes())
}

impl Display {
    fn open() -> Option<Display> {
        // SAFETY: dlopen/dlsym of the system's libEGL; each symbol is cast to
        // the signature the EGL specification gives it.
        let egl = unsafe {
            let lib = libc::dlopen(c"libEGL.so.1".as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL);
            if lib.is_null() {
                log::warn!("host GL: no libEGL.so.1");
                return None;
            }
            macro_rules! sym {
                ($n:literal) => {{
                    let p = libc::dlsym(lib, concat!($n, "\0").as_ptr() as *const c_char);
                    if p.is_null() {
                        log::warn!("host GL: libEGL has no {}", $n);
                        return None;
                    }
                    std::mem::transmute::<*mut c_void, _>(p)
                }};
            }
            Egl {
                get_proc_address: sym!("eglGetProcAddress"),
                get_display: sym!("eglGetDisplay"),
                initialize: sym!("eglInitialize"),
                query_string: sym!("eglQueryString"),
                bind_api: sym!("eglBindAPI"),
                choose_config: sym!("eglChooseConfig"),
                create_context: sym!("eglCreateContext"),
                destroy_context: sym!("eglDestroyContext"),
                make_current: sym!("eglMakeCurrent"),
                get_current_context: sym!("eglGetCurrentContext"),
                get_error: sym!("eglGetError"),
            }
        };
        let dpy = Self::display(&egl)?;
        let (mut major, mut minor) = (0, 0);
        // SAFETY: a display handle from EGL; out pointers are valid.
        if unsafe { (egl.initialize)(dpy, &mut major, &mut minor) } == 0 {
            log::warn!("host GL: eglInitialize failed ({:#x})", unsafe { (egl.get_error)() });
            return None;
        }
        // SAFETY: an initialized display.
        let exts = unsafe { (egl.query_string)(dpy, EGL_EXTENSIONS) };
        if !has(exts, "EGL_KHR_surfaceless_context") {
            log::warn!("host GL: EGL {major}.{minor} has no EGL_KHR_surfaceless_context");
            return None;
        }
        let config = if has(exts, "EGL_KHR_no_config_context") {
            std::ptr::null_mut()
        } else {
            let attrs = [EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_NONE];
            let mut config = std::ptr::null_mut();
            let mut n = 0;
            // SAFETY: the attribute list ends in EGL_NONE; out pointers are valid.
            if unsafe { (egl.choose_config)(dpy, attrs.as_ptr(), &mut config, 1, &mut n) } == 0 || n == 0 {
                log::warn!("host GL: no EGL config renders desktop OpenGL");
                return None;
            }
            config
        };
        log::info!("host GL: EGL {major}.{minor}");
        Some(Display { egl, dpy, config })
    }

    /// Mesa's surfaceless platform, else the first device, else the default.
    fn display(egl: &Egl) -> Option<EglDisplay> {
        // SAFETY: client extensions are queried with EGL_NO_DISPLAY; the
        // platform functions come from eglGetProcAddress with the signature
        // EGL_EXT_platform_base gives them.
        unsafe {
            let client = (egl.query_string)(std::ptr::null_mut(), EGL_EXTENSIONS);
            let platform = (egl.get_proc_address)(c"eglGetPlatformDisplayEXT".as_ptr());
            if has(client, "EGL_EXT_platform_base") && !platform.is_null() {
                let get_platform: unsafe extern "C" fn(u32, *mut c_void, *const i32) -> EglDisplay =
                    std::mem::transmute(platform);
                if has(client, "EGL_MESA_platform_surfaceless") {
                    let d = get_platform(EGL_PLATFORM_SURFACELESS_MESA, std::ptr::null_mut(), std::ptr::null());
                    if !d.is_null() {
                        return Some(d);
                    }
                }
                let query = (egl.get_proc_address)(c"eglQueryDevicesEXT".as_ptr());
                if has(client, "EGL_EXT_platform_device") && !query.is_null() {
                    let query: unsafe extern "C" fn(i32, *mut *mut c_void, *mut i32) -> u32 = std::mem::transmute(query);
                    let mut dev = std::ptr::null_mut();
                    let mut n = 0;
                    if query(1, &mut dev, &mut n) != 0 && n > 0 {
                        let d = get_platform(EGL_PLATFORM_DEVICE_EXT, dev, std::ptr::null());
                        if !d.is_null() {
                            return Some(d);
                        }
                    }
                }
            }
            let d = (egl.get_display)(std::ptr::null_mut());
            if d.is_null() {
                log::warn!("host GL: no EGL display");
                return None;
            }
            Some(d)
        }
    }

    /// The process's display, opened on first use.
    fn get() -> Option<&'static Display> {
        static DISPLAY: OnceLock<Option<Display>> = OnceLock::new();
        DISPLAY.get_or_init(Display::open).as_ref()
    }
}

/// What the driver's contexts provide: the GL version and extension names.
/// Read from the first context made current; every context comes from the
/// same display and config, so they all agree.
struct Provides {
    version: (u32, u32),
    extensions: HashSet<String>,
}

impl Provides {
    /// From the current context; `None` without one.
    fn read() -> Option<Provides> {
        // SAFETY: glGetString with a context current (null without one).
        let (v, e) = unsafe { (crate::gl::glGetString(GL_VERSION), crate::gl::glGetString(GL_EXTENSIONS)) };
        if v.is_null() {
            return None;
        }
        // SAFETY: NUL-terminated strings owned by the GL.
        let v = unsafe { CStr::from_ptr(v as *const c_char) }.to_string_lossy();
        let mut nums = v.split(|c: char| !c.is_ascii_digit()).filter_map(|n| n.parse().ok());
        let version = (nums.next().unwrap_or(1), nums.next().unwrap_or(0));
        let extensions = if e.is_null() {
            HashSet::new()
        } else {
            unsafe { CStr::from_ptr(e as *const c_char) }.to_string_lossy().split_whitespace().map(str::to_owned).collect()
        };
        log::info!("host GL: OpenGL {v}, {} extensions", extensions.len());
        Some(Provides { version, extensions })
    }

    fn provides(&self, name: &str) -> bool {
        let Ok(i) = crate::avail::AVAIL.binary_search_by(|(n, _)| (*n).cmp(name)) else { return false };
        crate::avail::AVAIL[i].1.iter().any(|r| match r.split_once('.') {
            Some((major, minor)) if r.as_bytes()[0].is_ascii_digit() => {
                (major.parse().unwrap_or(99), minor.parse().unwrap_or(99)) <= self.version
            }
            _ => self.extensions.contains(*r),
        })
    }
}

pub struct EglBackend {
    display: &'static Display,
    provides: Mutex<Option<Provides>>,
}

impl EglBackend {
    /// `None` if this host has no EGL that can make a headless desktop GL
    /// context.
    pub fn new() -> Option<EglBackend> {
        Some(EglBackend { display: Display::get()?, provides: Mutex::new(None) })
    }

    fn bind(&self) {
        // SAFETY: eglBindAPI sets per-thread state; desktop OpenGL is what
        // every call here means.
        unsafe { (self.display.egl.bind_api)(EGL_OPENGL_API) };
    }
}

impl Backend for EglBackend {
    fn name(&self) -> &'static str {
        "EGL"
    }

    fn create_context(&mut self, share: Option<ContextHandle>) -> Option<ContextHandle> {
        self.bind();
        let d = self.display;
        let share = share.map_or(std::ptr::null_mut(), |c| c.0 as EglContext);
        // No version asked for: the driver's highest compatibility-profile
        // context, which keeps glBegin, display lists and the matrix stack.
        let attrs = [EGL_NONE];
        // SAFETY: an initialized display, its config (or none), `share` null
        // or a live context; the attribute list ends in EGL_NONE.
        let ctx = unsafe { (d.egl.create_context)(d.dpy, d.config, share, attrs.as_ptr()) };
        if ctx.is_null() {
            log::warn!("host GL: eglCreateContext failed ({:#x})", unsafe { (d.egl.get_error)() });
            return None;
        }
        Some(ContextHandle(ctx as usize))
    }

    fn destroy_context(&mut self, ctx: ContextHandle) {
        self.bind();
        let d = self.display;
        // SAFETY: `ctx` came from create_context and is destroyed once.
        unsafe {
            if (d.egl.get_current_context)() as usize == ctx.0 {
                (d.egl.make_current)(d.dpy, std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut());
            }
            (d.egl.destroy_context)(d.dpy, ctx.0 as EglContext);
        }
    }

    fn make_current(&mut self, ctx: Option<ContextHandle>) -> bool {
        self.bind();
        let d = self.display;
        let p = ctx.map_or(std::ptr::null_mut(), |c| c.0 as EglContext);
        // SAFETY: null or a live context, with no surface (surfaceless).
        let ok = unsafe { (d.egl.make_current)(d.dpy, std::ptr::null_mut(), std::ptr::null_mut(), p) } != 0;
        if ok && ctx.is_some() {
            let mut provides = self.provides.lock().unwrap();
            if provides.is_none() {
                *provides = Provides::read();
            }
        }
        ok
    }

    fn current(&self) -> Option<ContextHandle> {
        self.bind();
        // SAFETY: a query.
        let p = unsafe { (self.display.egl.get_current_context)() };
        (!p.is_null()).then_some(ContextHandle(p as usize))
    }

    fn lookup(&self, name: &str) -> *const c_void {
        let provided = self.provides.lock().unwrap().as_ref().is_some_and(|p| p.provides(name));
        if !provided {
            return std::ptr::null();
        }
        let Ok(c) = CString::new(name) else { return std::ptr::null() };
        // SAFETY: a lookup; the result is only called through a signature
        // generated from the same entry point's declaration.
        unsafe { (self.display.egl.get_proc_address)(c.as_ptr()) }
    }

    fn new_surface(&mut self, _w: i32, _h: i32) -> Option<Box<dyn Surface>> {
        None
    }
}
