# Host GL on Linux: EGL, glvnd and Mesa

**Keywords:** hostgl,linux,egl,mesa,glvnd,surfaceless,avail,texture names,c_char

The Linux backend (`iris-hostgl/src/backend/egl.rs`) makes headless
compatibility-profile contexts with no surface (`EGL_KHR_surfaceless_context`)
on Mesa's surfaceless platform, else the first EGL device, else the default
display. Things that bit while bringing it up:

- **Every `gl*` name resolves.** glvnd and Mesa answer `eglGetProcAddress` /
  `dlsym` for any name with a callable do-nothing stub, so "is this entry
  point there?" cannot be asked by looking it up. The service relies on a
  null to fall back from `glFooSGI`/`glFooEXT` to `glFoo`, so the backend only
  looks a name up once the context's version or extensions provide it:
  `src/avail.rs`, generated from the pinned registry by `tools/gl_avail.py`.
  Rerun the generator whenever `calls.rs` changes.
- **Mesa picks texture names from the bottom regardless.** `glGenTextures`
  returns one more than the highest name in use, even when the program bound
  names of its own choosing, so it handed out names the service had taken
  for its own textures and the two collided. The service's internal names
  now count down from `u32::MAX`.
- **`c_char` is `u8` on arm64 Linux** (`i8` on macOS): shader source and
  string externs must use `c_char`, not `i8`.
- **Link vs open.** libEGL is `dlopen`ed (`libEGL.so.1`); the GL entry points
  the crate calls by name come from libGL, linked, which glvnd dispatches to
  the current context (and libGL brings glvnd's libGLX and libX11 with it).
  Building needs libGL's development link: `libgl-dev` (Debian/Ubuntu),
  `libglvnd-devel` (Fedora), `libglvnd` (Arch). Running needs libEGL and
  Mesa's EGL and DRI drivers. **On Debian/Ubuntu `libegl-mesa0` does not
  pull in libEGL itself**: install `libegl1` too, or iris logs "no
  libEGL.so.1" and host GL stays off (seen on Ubuntu 24.04). Fedora:
  `mesa-libEGL mesa-dri-drivers`; Arch: `mesa`. `hostgl` is opt-in and
  outside the workspace, so a default build links none of this.
- **No GPU needed for the tests.** Mesa's software renderer (llvmpipe) makes
  surfaceless contexts too: the hostgl tests pass in containers on Fedora 44,
  Arch and Ubuntu 24.04 with no device at all.
- **A VM's GL is slow to wake.** Under Parallels (virgl) a frame's readback
  costs more than on bare metal; the GL thread wants a host core of its own.
