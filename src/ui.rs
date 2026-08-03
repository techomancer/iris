use std::sync::Arc;
use std::time::{Duration, Instant};
use parking_lot::Mutex;
use winit::{
    event::{ElementState, Event, KeyEvent, WindowEvent, MouseButton},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowAttributes},
};
use glow::HasContext;
use crate::ps2::Ps2Controller;
use crate::rex3::{Rex3, Renderer};
use crate::disp::{Rex3Screen, StatusBar, StatusBarTexture, BarStats, STATUS_BAR_HEIGHT};
use crate::compositor::{Compositor, SwCompositor};
use crate::gl_compositor::GlCompositor;
use crate::debug_overlay::DebugOverlay;
use crate::hptimer::{TimerManager, TimerReturn};
use crate::wd33c93a::Wd33c93a;
use glutin::config::ConfigTemplateBuilder;
use glutin::context::{ContextApi, ContextAttributesBuilder, GlProfile, NotCurrentContext, PossiblyCurrentContext, Version};
use glutin::display::GetGlDisplay;
use glutin::prelude::*;
use glutin::surface::{GlSurface, SwapInterval, WindowSurface, Surface};
use glutin_winit::{DisplayBuilder, GlWindow};
use raw_window_handle::HasRawWindowHandle;
use std::num::NonZeroU32;
use std::ffi::CString;

// Vertex layout: [pos_x, pos_y, tex_u, tex_v] × 4 vertices = 64 bytes
const VBO_SIZE: i32 = 64;

struct GlState {
    gl: glow::Context,
    context: PossiblyCurrentContext,
    surface: Surface<WindowSurface>,
    // Integer texelFetch shader (used at 1x and 2x exact scales)
    integer_program:      glow::Program,
    viewport_info_loc:    Option<glow::UniformLocation>,
    scale_factor_loc:     Option<glow::UniformLocation>,
    // Fallback UV sampler shader with mipmap/trilinear (used at non-integer scales)
    fallback_program:     glow::Program,
    fallback_tex_loc:     Option<glow::UniformLocation>,
    fallback_ortho_loc:   Option<glow::UniformLocation>,
    // Shared VAO + two VBOs: emulator quad and status-bar quad
    vao:        glow::VertexArray,
    main_vbo:   glow::Buffer,
    status_vbo: glow::Buffer,
}

// Snap-to request from event thread to render thread
#[derive(Clone, Copy, PartialEq)]
enum ScaleSnap { Scale1x, Scale2x }

// Which context we actually got. GlCompositor and the texelFetch/integer-sampler
// shaders need GL 3.2 core (usampler2D, #version 150). If the driver can't give us
// that, we drop to a plain GL 2.1-compatible context and switch to SwCompositor +
// a legacy (#version 120) copy-only shader — no integer textures, no texelFetch.
#[derive(Clone, Copy, PartialEq, Debug)]
enum GlTier { Core32, Legacy }

struct GlRenderer {
    window:      Arc<Window>,
    gl_config:   glutin::config::Config,
    // Created on the main thread (where the window/display were set up) and
    // handed across to the refresh thread, which only calls make_current() on
    // it. Some drivers (notably proprietary NVIDIA) are picky about GLX calls
    // that validate against the drawable/FBConfig being issued from a thread
    // other than the one that created the X11 connection/window.
    not_current_context: Option<NotCurrentContext>,
    gl_tier: GlTier,
    window_size: Arc<Mutex<Option<(u32, u32)>>>,
    scale_snap:  Arc<Mutex<Option<ScaleSnap>>>,
    // Current emulated display resolution (width, height), published for the
    // event thread so it can lock the window to the right aspect ratio.
    display_res: Arc<Mutex<(u32, u32)>>,
    state:       Option<GlState>,
    compositor:  Box<dyn Compositor>,
    use_gl_compositor: bool,
    current_w:     usize,
    current_h:     usize,
    current_win_w: usize,
    current_win_h: usize,
}

// Safety: GlRenderer is sent to the refresh thread where it owns and uses the GL context.
// No other thread touches these fields.
unsafe impl Send for GlRenderer {}

impl GlRenderer {
    fn init_gl(&mut self) {
        let gl_display = self.gl_config.display();

        // The context itself was already created on the main thread in Ui::new()
        // (see comment on not_current_context); here we only make it current on
        // the refresh thread.
        let not_current_gl_context = self.not_current_context.take()
            .expect("GL context missing — init_gl() called more than once");

        let attrs = self.window.build_surface_attributes(Default::default()).expect("surface attributes");
        let gl_surface = unsafe {
            gl_display
                .create_window_surface(&self.gl_config, &attrs)
                .unwrap()
        };

        let gl_context = not_current_gl_context.make_current(&gl_surface).unwrap();

        let gl = unsafe {
            glow::Context::from_loader_function(|s| {
                gl_display.get_proc_address(&CString::new(s).unwrap())
            })
        };

        let _ = gl_surface.set_swap_interval(&gl_context, SwapInterval::Wait(NonZeroU32::new(1).unwrap()));

        // In the Legacy tier we don't have GL 3.2 core (no texelFetch, no usampler2D),
        // so both the vertex and fragment shaders are written as plain GLSL 120 —
        // varying/gl_FragColor instead of in/out — and there is no integer shader at all.
        let legacy = self.gl_tier == GlTier::Legacy;

        let (integer_program, viewport_info_loc, scale_factor_loc,
             fallback_program, fallback_tex_loc, fallback_ortho_loc,
             vao, main_vbo, status_vbo) = unsafe {

            // Shared vertex shader: pixel-coordinate ortho projection.
            // ortho = (win_w, win_h); (0,0)=top-left, y increases downward.
            let vs_src_150 = "
                #version 150
                in vec2 position;
                in vec2 tex_coord;
                out vec2 v_tex_coord;
                uniform vec2 ortho;
                void main() {
                    vec2 ndc = (position / ortho) * 2.0 - 1.0;
                    gl_Position = vec4(ndc.x, -ndc.y, 0.0, 1.0);
                    v_tex_coord = tex_coord;
                }
            ";
            let vs_src_120 = "
                #version 120
                attribute vec2 position;
                attribute vec2 tex_coord;
                varying vec2 v_tex_coord;
                uniform vec2 ortho;
                void main() {
                    vec2 ndc = (position / ortho) * 2.0 - 1.0;
                    gl_Position = vec4(ndc.x, -ndc.y, 0.0, 1.0);
                    v_tex_coord = tex_coord;
                }
            ";

            // ── Integer shader: texelFetch, no filtering. Used at 1x and 2x. ──
            // scale_factor must be an exact integer (1 or 2). GL 3.2 core only —
            // there is no legacy equivalent, Legacy tier always uses the fallback shader.
            let integer_fs_src = "
                #version 150
                in vec2 v_tex_coord;
                out vec4 color;
                uniform sampler2D tex;
                uniform ivec2 viewport_info[2];
                uniform int scale_factor;
                void main() {
                    int tex_h  = viewport_info[0].y;
                    int quad_y = viewport_info[1].y;
                    int scale  = max(scale_factor, 1);
                    int x = int(gl_FragCoord.x) / scale;
                    int y = (int(gl_FragCoord.y) - quad_y) / scale;
                    y = (tex_h - 1) - y;
                    color = texelFetch(tex, ivec2(x, y), 0);
                }
            ";

            // ── Fallback shader: UV sampler with trilinear filtering. ──
            // Caller sets TEXTURE_MIN_FILTER = LINEAR_MIPMAP_LINEAR and generates mipmaps.
            let fallback_fs_src_150 = "
                #version 150
                in vec2 v_tex_coord;
                out vec4 color;
                uniform sampler2D tex;
                void main() {
                    color = texture(tex, v_tex_coord);
                }
            ";
            let fallback_fs_src_120 = "
                #version 120
                varying vec2 v_tex_coord;
                uniform sampler2D tex;
                void main() {
                    gl_FragColor = texture2D(tex, v_tex_coord);
                }
            ";

            let vs_src = if legacy { vs_src_120 } else { vs_src_150 };
            let fallback_fs_src = if legacy { fallback_fs_src_120 } else { fallback_fs_src_150 };

            let compile_shader = |kind: u32, src: &str| -> Option<glow::Shader> {
                let s = gl.create_shader(kind).unwrap();
                gl.shader_source(s, src);
                gl.compile_shader(s);
                if gl.get_shader_compile_status(s) { Some(s) } else {
                    eprintln!("Shader compile error: {}", gl.get_shader_info_log(s));
                    gl.delete_shader(s);
                    None
                }
            };

            let link_program = |vs: glow::Shader, fs: glow::Shader| -> Option<glow::Program> {
                let p = gl.create_program().unwrap();
                gl.attach_shader(p, vs);
                gl.attach_shader(p, fs);
                gl.link_program(p);
                if gl.get_program_link_status(p) { Some(p) } else {
                    eprintln!("Program link error: {}", gl.get_program_info_log(p));
                    gl.delete_program(p);
                    None
                }
            };

            let vs = compile_shader(glow::VERTEX_SHADER, vs_src)
                .expect("vertex shader must compile");

            // Integer program is GL 3.2 core only (#version 150, texelFetch) and has no
            // legacy equivalent — linking it against the 120 vertex shader would be an
            // invalid mixed-version program, so skip it outright in Legacy tier and always
            // draw through fallback_program there. In Core32 tier, try to compile it;
            // fall back to fallback_program if it fails.
            let int_fs    = if legacy { None } else { compile_shader(glow::FRAGMENT_SHADER, integer_fs_src) };
            let int_prog  = int_fs.and_then(|fs| link_program(vs, fs));

            // Fallback program — must always compile.
            let fb_fs   = compile_shader(glow::FRAGMENT_SHADER, fallback_fs_src)
                .expect("fallback fragment shader must compile");
            let fb_prog = link_program(vs, fb_fs)
                .expect("fallback program must link");

            // If integer program failed, reuse fallback as integer_program too (same draw path,
            // just won't do texelFetch). compositor_status() will reflect this.
            let integer_program = int_prog.unwrap_or(fb_prog);

            let viewport_info_loc = gl.get_uniform_location(integer_program, "viewport_info");
            let scale_factor_loc  = gl.get_uniform_location(integer_program, "scale_factor");
            let fallback_tex_loc  = gl.get_uniform_location(fb_prog, "tex");
            let fallback_ortho_loc = gl.get_uniform_location(fb_prog, "ortho");

            let vao = gl.create_vertex_array().unwrap();
            gl.bind_vertex_array(Some(vao));

            let main_vbo = gl.create_buffer().unwrap();
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(main_vbo));
            gl.buffer_data_size(glow::ARRAY_BUFFER, VBO_SIZE, glow::DYNAMIC_DRAW);
            Self::bind_vbo_attribs(&gl, integer_program, main_vbo);

            let status_vbo = gl.create_buffer().unwrap();
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(status_vbo));
            gl.buffer_data_size(glow::ARRAY_BUFFER, VBO_SIZE, glow::DYNAMIC_DRAW);
            Self::bind_vbo_attribs(&gl, integer_program, status_vbo);

            (integer_program, viewport_info_loc, scale_factor_loc,
             fb_prog, fallback_tex_loc, fallback_ortho_loc,
             vao, main_vbo, status_vbo)
        };

        self.state = Some(GlState {
            gl,
            context: gl_context,
            surface: gl_surface,
            integer_program,
            viewport_info_loc,
            scale_factor_loc,
            fallback_program,
            fallback_tex_loc,
            fallback_ortho_loc,
            vao,
            main_vbo,
            status_vbo,
        });
    }

    // Upload a quad covering pixel rect [x0..x1] × [y0..y1] (top-left origin, y down).
    unsafe fn upload_quad(gl: &glow::Context, vbo: glow::Buffer,
        x0: f32, y0: f32, x1: f32, y1: f32,
        u0: f32, v0: f32, u1: f32, v1: f32)
    {
        let vertices: [f32; 16] = [
            x0, y0,  u0, v0,
            x1, y0,  u1, v0,
            x0, y1,  u0, v1,
            x1, y1,  u1, v1,
        ];
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
        let u8_slice = std::slice::from_raw_parts(vertices.as_ptr() as *const u8, vertices.len() * 4);
        gl.buffer_sub_data_u8_slice(glow::ARRAY_BUFFER, 0, u8_slice);
    }

    unsafe fn bind_vbo_attribs(gl: &glow::Context, program: glow::Program, vbo: glow::Buffer) {
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
        let pos_loc = gl.get_attrib_location(program, "position").unwrap();
        gl.enable_vertex_attrib_array(pos_loc);
        gl.vertex_attrib_pointer_f32(pos_loc, 2, glow::FLOAT, false, 16, 0);
        if let Some(tex_loc) = gl.get_attrib_location(program, "tex_coord") {
            gl.enable_vertex_attrib_array(tex_loc);
            gl.vertex_attrib_pointer_f32(tex_loc, 2, glow::FLOAT, false, 16, 8);
        }
    }

    // Set trilinear filtering and regenerate mipmaps for a bound TEXTURE_2D.
    // Call after binding the texture, before drawing.
    unsafe fn setup_trilinear(gl: &glow::Context) {
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR_MIPMAP_LINEAR as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
        gl.generate_mipmap(glow::TEXTURE_2D);
    }

    // Draw a texture using the integer texelFetch shader.
    // quad_y_bottom: bottom edge of the quad in GL pixels from bottom of window.
    // scale_i: exact integer scale (1 or 2).
    unsafe fn draw_tex_integer(
        gl: &glow::Context,
        state: &GlState,
        vbo: glow::Buffer,
        tex: glow::Texture,
        tex_w: i32, tex_h: i32,
        quad_y_bottom: i32,
        scale_i: i32,
        win_w: f32, win_h: f32,
    ) {
        gl.use_program(Some(state.integer_program));
        if let Some(loc) = gl.get_uniform_location(state.integer_program, "ortho") {
            gl.uniform_2_f32(Some(&loc), win_w, win_h);
        }
        gl.bind_vertex_array(Some(state.vao));
        gl.bind_texture(glow::TEXTURE_2D, Some(tex));
        Self::setup_trilinear(gl);
        Self::bind_vbo_attribs(gl, state.integer_program, vbo);
        // viewport_info[0] = (tex_w, tex_h), viewport_info[1] = (0, quad_y_bottom)
        let info = [tex_w, tex_h, 0, quad_y_bottom];
        gl.uniform_2_i32_slice(state.viewport_info_loc.as_ref(), &info);
        if let Some(loc) = &state.scale_factor_loc {
            gl.uniform_1_i32(Some(loc), scale_i);
        }
        gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
    }

    // Draw a texture using the fallback UV sampler shader with trilinear filtering.
    unsafe fn draw_tex_fallback(
        gl: &glow::Context,
        state: &GlState,
        vbo: glow::Buffer,
        tex: glow::Texture,
        win_w: f32, win_h: f32,
    ) {
        gl.use_program(Some(state.fallback_program));
        if let Some(loc) = &state.fallback_ortho_loc {
            gl.uniform_2_f32(Some(loc), win_w, win_h);
        }
        if let Some(loc) = &state.fallback_tex_loc {
            gl.uniform_1_i32(Some(loc), 0);
        }
        gl.bind_vertex_array(Some(state.vao));
        gl.active_texture(glow::TEXTURE0);
        gl.bind_texture(glow::TEXTURE_2D, Some(tex));
        Self::setup_trilinear(gl);
        Self::bind_vbo_attribs(gl, state.fallback_program, vbo);
        gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
    }

    // Compute letterboxed quad rect and float scale for a given display and window size.
    // Returns (x0, y0, x1, y1, scale_f) in window pixel coords (top-left origin).
    // The status bar scales with the display: total content height = indy_h + STATUS_BAR_HEIGHT.
    fn letterbox(indy_w: f32, indy_h: f32, win_w: f32, win_h: f32)
        -> (f32, f32, f32, f32, f32)
    {
        let total_h = indy_h + STATUS_BAR_HEIGHT as f32;
        let scale = (win_w / indy_w).min(win_h / total_h);
        let disp_w = indy_w * scale;
        let disp_h = indy_h * scale;
        let sb_h   = STATUS_BAR_HEIGHT as f32 * scale;
        let x0 = ((win_w - disp_w) * 0.5).floor();
        let y0 = ((win_h - disp_h - sb_h) * 0.5).floor();
        (x0, y0, x0 + disp_w, y0 + disp_h, scale)
    }
}

impl Renderer for GlRenderer {
    fn present(
        &mut self,
        screen:        &mut Rex3Screen,
        overlay:       &mut DebugOverlay,
        status:        &mut StatusBar,
        sbtex:         &mut StatusBarTexture,
        stats:         &BarStats,
        need_readback: bool,
        live_fb_rgb:   Option<&[u32]>,
        live_fb_aux:   Option<&[u32]>,
    ) {
        if self.state.is_none() {
            self.init_gl();
        }

        let width  = screen.width;
        let height = screen.height;
        if width == 0 || height == 0 { return; }

        let state = self.state.as_mut().unwrap();
        let gl    = &state.gl;

        // Handle pending scale-snap from keyboard hotkey (RCtrl+1 / RCtrl+2).
        if let Some(snap) = self.scale_snap.lock().take() {
            let s = match snap { ScaleSnap::Scale1x => 1u32, ScaleSnap::Scale2x => 2u32 };
            let _ = self.window.request_inner_size(winit::dpi::PhysicalSize::new(
                width as u32 * s,
                (height as u32 + STATUS_BAR_HEIGHT as u32) * s,
            ));
        }

        // Handle window resize — take the latest queued size.
        let (win_w, win_h) = if let Some((w, h)) = self.window_size.lock().take() {
            state.surface.resize(
                &state.context,
                NonZeroU32::new(w).unwrap(),
                NonZeroU32::new(h).unwrap(),
            );
            (w as usize, h as usize)
        } else if self.current_win_w > 0 {
            (self.current_win_w, self.current_win_h)
        } else {
            // First frame before any resize event: query actual window size.
            let s = self.window.inner_size();
            (s.width as usize, s.height as usize)
        };

        let win_w_f = win_w as f32;
        let win_h_f = win_h as f32;
        let win_w_i = win_w as i32;
        let win_h_i = win_h as i32;

        // Letterbox: status bar scales with display.
        let (qx0, qy0, qx1, qy1, scale_f) = Self::letterbox(
            width as f32, height as f32, win_w_f, win_h_f);
        let sb_h_px = STATUS_BAR_HEIGHT as f32 * scale_f;
        let sb_h_i  = sb_h_px as i32;

        // Use integer texelFetch shader only at exactly 1x or 2x, and only when we
        // actually have a Core32 context — Legacy tier has no texelFetch shader at all.
        let use_integer = self.gl_tier == GlTier::Core32 && (scale_f == 1.0 || scale_f == 2.0);
        let scale_i = scale_f.round() as i32;

        // Recompute quads when anything changes.
        if width != self.current_w || height != self.current_h
            || win_w != self.current_win_w || win_h != self.current_win_h
        {
            self.current_w     = width;
            self.current_h     = height;
            self.current_win_w = win_w;
            self.current_win_h = win_h;
            // Publish the display resolution for the event thread's aspect lock.
            *self.display_res.lock() = (width as u32, height as u32);
            // UV coords into 2048×1024 texture
            let max_u      = width  as f32 / 2048.0;
            let max_v_main = height as f32 / 1024.0;
            unsafe {
                // Main display quad (letterboxed).
                Self::upload_quad(gl, state.main_vbo,
                    qx0, qy0, qx1, qy1,
                    0.0, 0.0, max_u, max_v_main);
                // Status bar: same x extent as letterboxed display, scaled height, bottom of window.
                // Texture is 2048×STATUS_BAR_HEIGHT, so v goes 0..1.
                Self::upload_quad(gl, state.status_vbo,
                    qx0, qy1, qx1, qy1 + sb_h_px,
                    0.0, 0.0, max_u, 1.0);
            }
        }

        // quad_y_bottom for integer shader: how many GL pixels from bottom to bottom of main quad.
        let main_quad_y_bottom = (win_h_f - qy1) as i32;

        unsafe {
            // ── Compositor runs first — it sets its own FBO/viewport ─────────
            let src      = screen.compositor_source_from(live_fb_rgb, live_fb_aux);
            let main_tex = self.compositor.compose(&src, gl);

            // Restore full-window viewport/scissor — compositor FBO left them dirty.
            gl.viewport(0, 0, win_w_i, win_h_i);
            gl.enable(glow::SCISSOR_TEST);
            gl.scissor(0, 0, win_w_i, win_h_i);
            gl.clear_color(0.0, 0.0, 0.0, 1.0);
            gl.clear(glow::COLOR_BUFFER_BIT);

            if need_readback {
                if let Some(pixels) = self.compositor.read_pixels() {
                    screen.rgba.copy_from_slice(pixels);
                } else {
                    self.compositor.readback_to_screen(&mut screen.rgba, width, height, gl);
                }
            }

            // ── Pass 1: main display ─────────────────────────────────────────
            gl.scissor(0, 0, win_w_i, win_h_i);
            if use_integer {
                Self::draw_tex_integer(gl, state, state.main_vbo, main_tex,
                    width as i32, height as i32, main_quad_y_bottom, scale_i,
                    win_w_f, win_h_f);
            } else {
                Self::draw_tex_fallback(gl, state, state.main_vbo, main_tex,
                    win_w_f, win_h_f);
            }

            // ── Pass 2: debug overlay (alpha-blended) ────────────────────────
            if overlay.active() {
                gl.enable(glow::BLEND);
                gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);
                let ov_src = screen.overlay_source();
                let ov_tex = overlay.render(&ov_src, gl);
                if use_integer {
                    Self::draw_tex_integer(gl, state, state.main_vbo, ov_tex,
                        width as i32, height as i32, main_quad_y_bottom, scale_i,
                        win_w_f, win_h_f);
                } else {
                    Self::draw_tex_fallback(gl, state, state.main_vbo, ov_tex,
                        win_w_f, win_h_f);
                }
                gl.disable(glow::BLEND);
            }

            // ── Pass 3: status bar ───────────────────────────────────────────
            gl.scissor(0, 0, win_w_i, win_h_i);
            let sb_tex = sbtex.render_and_upload(status, stats, width, gl);
            Self::draw_tex_fallback(gl, state, state.status_vbo, sb_tex, win_w_f, win_h_f);

            gl.disable(glow::SCISSOR_TEST);

            state.surface.swap_buffers(&state.context).unwrap();
        }
    }

    fn resize(&mut self, width: usize, height: usize) {
        // Publish the new resolution *before* the snap below, so the event
        // thread's aspect lock sees it when it handles the resulting Resized
        // event (otherwise it would re-fit the window to the stale aspect).
        *self.display_res.lock() = (width as u32, height as u32);
        // On display resolution change, snap window to 1x of the new resolution.
        let _ = self.window.request_inner_size(winit::dpi::PhysicalSize::new(
            width as u32,
            (height + STATUS_BAR_HEIGHT) as u32,
        ));
    }

    fn stop(&mut self) {
        if let Some(state) = self.state.take() {
            self.compositor.destroy(&state.gl);
        }
        self.current_w     = 0;
        self.current_h     = 0;
        self.current_win_w = 0;
        self.current_win_h = 0;
    }

    fn compositor_status(&self) -> String {
        let comp = if self.use_gl_compositor { "gl" } else { "sw" };
        format!("compositor={} shader=integer+fallback", comp)
    }

    fn switch_compositor(&mut self, use_gl: bool) -> &'static str {
        if use_gl == self.use_gl_compositor {
            return if use_gl { "gl" } else { "sw" };
        }
        self.use_gl_compositor = use_gl;
        if let Some(state) = &self.state {
            self.compositor.destroy(&state.gl);
        }
        if use_gl {
            self.compositor = Box::new(GlCompositor::new());
            "gl"
        } else {
            self.compositor = Box::new(SwCompositor::new());
            "sw"
        }
    }

}

struct MouseDelta {
    accum: (f64, f64),
    wheel: f64,
    buttons: u8,
}

/// UI Manager handling Window, OpenGL context, and Input
pub struct Ui {
    ps2: Arc<Ps2Controller>,
    rex3: Arc<Rex3>,
    scsi: Arc<Wd33c93a>,
    window: Arc<Window>,
    window_size: Arc<Mutex<Option<(u32, u32)>>>,
    scale_snap:  Arc<Mutex<Option<ScaleSnap>>>,
    display_res: Arc<Mutex<(u32, u32)>>,
    timer_manager: Arc<TimerManager>,
    initial_scale: u32,
    scroll_pixels_per_line: f64,
    lock_aspect_ratio: bool,
}

impl Ui {
    pub fn new(ps2: Arc<Ps2Controller>, rex3: Arc<Rex3>, scsi: Arc<Wd33c93a>, timer_manager: Arc<TimerManager>, event_loop: &EventLoop<()>, scale: u32, scroll_pixels_per_line: f64, lock_aspect_ratio: bool) -> Self {
        // The Indy's default video mode is 1280×1024; open the window at that
        // size (plus the status bar). The renderer snaps to the real resolution
        // via resize() once the PROM/IRIX programs its actual mode.
        let w = 1280 * scale;
        let h = (1024 + STATUS_BAR_HEIGHT as u32) * scale;
        let window_builder = WindowAttributes::default()
            .with_title(crate::machine::emulator_name())
            .with_resizable(true)
            .with_inner_size(winit::dpi::PhysicalSize::new(w, h));

        let template = ConfigTemplateBuilder::new()
            .with_alpha_size(8)
            .with_transparency(false);

        let display_builder = DisplayBuilder::new().with_window_attributes(Some(window_builder));

        let (window, gl_config) = display_builder
            .build(event_loop, template, |configs| {
                configs
                    .reduce(|accum, config| {
                        if config.num_samples() > accum.num_samples() { config } else { accum }
                    })
                    .unwrap()
            })
            .unwrap();

        let window = Arc::new(window.unwrap());

        // Create the GL context here on the main thread, i.e. the same thread
        // that created the window/display above. The refresh thread only
        // makes it current later (in GlRenderer::init_gl); it never calls
        // create_context itself. See the not_current_context field comment.
        let raw_window_handle = window.raw_window_handle().expect("no raw window handle");
        let gl_display = gl_config.display();

        // Try, in order: (1) explicit GL 3.2 core — what GlCompositor and the
        // texelFetch/integer shaders need; (2) explicit GL 2.1 compatibility —
        // enough for SwCompositor + the legacy (#version 120) copy shader;
        // (3) whatever the driver considers default, as a last resort. Some
        // drivers (seen on NVIDIA proprietary) reject an unversioned/ambiguous
        // context request with BadValue but accept an explicit version, so this
        // also guards against the original bug independent of the tier we land on.
        let attempts: &[(GlTier, ContextApi, Option<GlProfile>)] = &[
            (GlTier::Core32, ContextApi::OpenGl(Some(Version::new(3, 2))), Some(GlProfile::Core)),
            (GlTier::Legacy, ContextApi::OpenGl(Some(Version::new(2, 1))), Some(GlProfile::Compatibility)),
            (GlTier::Legacy, ContextApi::OpenGl(None), None),
        ];

        let mut result = None;
        for (tier, api, profile) in attempts {
            let mut builder = ContextAttributesBuilder::new().with_context_api(*api);
            if let Some(p) = profile {
                builder = builder.with_profile(*p);
            }
            let context_attributes = builder.build(Some(raw_window_handle));
            match unsafe { gl_display.create_context(&gl_config, &context_attributes) } {
                Ok(ctx) => { result = Some((*tier, ctx)); break; }
                Err(e) => eprintln!("iris: GL context attempt {:?} failed: {e}", tier),
            }
        }
        let (gl_tier, not_current_context) = result
            .expect("failed to create a GL context under any requested version/profile");
        eprintln!("iris: using GL tier {:?}", gl_tier);

        let window_size = Arc::new(Mutex::new(None));
        let scale_snap  = Arc::new(Mutex::new(None));
        // Seed with the Indy's default 1280×1024; the render thread republishes
        // the real resolution on the first frame and on any mode change.
        let display_res = Arc::new(Mutex::new((1280u32, 1024u32)));

        // GlCompositor needs GL 3.2 core (usampler2D/texelFetch); Legacy tier falls
        // back to SwCompositor, which only ever uploads a plain RGBA texture.
        let use_gl_compositor = gl_tier == GlTier::Core32;
        let compositor: Box<dyn Compositor> = if use_gl_compositor {
            Box::new(GlCompositor::new())
        } else {
            Box::new(SwCompositor::new())
        };

        let renderer = GlRenderer {
            window:      window.clone(),
            gl_config,
            not_current_context: Some(not_current_context),
            gl_tier,
            window_size: window_size.clone(),
            scale_snap:  scale_snap.clone(),
            display_res: display_res.clone(),
            state:       None,
            compositor,
            use_gl_compositor,
            current_w:     0,
            current_h:     0,
            current_win_w: 0,
            current_win_h: 0,
        };

        *rex3.renderer.lock() = Some(Box::new(renderer));

        Self { ps2, rex3, scsi, window, window_size, scale_snap, display_res, timer_manager, initial_scale: scale, scroll_pixels_per_line, lock_aspect_ratio }
    }

    /// Run the UI event loop (blocks the current thread)
    pub fn run(self, event_loop: EventLoop<()>) {
        let Ui { ps2, rex3, scsi, window, window_size, scale_snap, display_res, timer_manager, initial_scale, scroll_pixels_per_line, lock_aspect_ratio } = self;
        let scale = initial_scale;

        let mut mouse_grabbed = false;
        let mut rctrl_held = false;
        // Last window size we accepted, used to tell which edge the user is
        // dragging when locking the aspect ratio.
        let mut last_win_size = {
            let s = window.inner_size();
            (s.width, s.height)
        };
        let mouse_delta = Arc::new(Mutex::new(MouseDelta { accum: (0.0, 0.0), wheel: 0.0, buttons: 0 }));

        {
            let ps2   = ps2.clone();
            let delta = mouse_delta.clone();
            timer_manager.add_recurring(
                Instant::now() + Duration::from_millis(10),
                Duration::from_millis(10),
                (ps2, delta),
                |(ps2, delta)| {
                    Self::flush_mouse_delta(ps2, delta, true);
                    TimerReturn::Continue
                },
            );
        }

        event_loop.set_control_flow(ControlFlow::Wait);
        event_loop.run(move |event, elwt| {
            match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => { elwt.exit() },
                    WindowEvent::Resized(size) => {
                        if size.width != 0 && size.height != 0 {
                            let mut new_size = (size.width, size.height);
                            // Lock the window to the display's aspect ratio so the
                            // picture fills it without letterbox bars. Skipped when
                            // fullscreen or maximized (aspect can't be honoured there)
                            // and when disabled by config.
                            if lock_aspect_ratio
                                && window.fullscreen().is_none()
                                && !window.is_maximized()
                            {
                                let (dw, dh) = *display_res.lock();
                                if let Some(fixed) = Self::aspect_fit(
                                    size.width, size.height, last_win_size, dw, dh)
                                {
                                    new_size = match window.request_inner_size(
                                        winit::dpi::PhysicalSize::new(fixed.0, fixed.1))
                                    {
                                        // Some => applied synchronously, no further
                                        // Resized event; use the actual granted size.
                                        Some(actual) => (actual.width, actual.height),
                                        None => fixed,
                                    };
                                }
                            }
                            last_win_size = new_size;
                            *window_size.lock() = Some(new_size);
                        }
                    }
                    WindowEvent::KeyboardInput { event, .. } => {
                        Self::handle_keyboard(&ps2, &rex3, &scsi, &scale_snap, event, &mut mouse_grabbed, &mut rctrl_held, &window);
                    }
                    WindowEvent::MouseInput { state, button, .. } => {
                        if mouse_grabbed {
                            let pressed = state == ElementState::Pressed;
                            let mask = match button {
                                MouseButton::Left   => 1,
                                MouseButton::Right  => 2,
                                MouseButton::Middle => 4,
                                _ => 0,
                            };
                            if mask != 0 {
                                let mut md = mouse_delta.lock();
                                if pressed { md.buttons |= mask; } else { md.buttons &= !mask; }
                                drop(md);
                                Self::flush_mouse_delta(&ps2, &mouse_delta, false);
                            }
                        } else if state == ElementState::Pressed && button == MouseButton::Left {
                            mouse_grabbed = true;
                            if window.set_cursor_grab(winit::window::CursorGrabMode::Locked).is_err() {
                                let _ = window.set_cursor_grab(winit::window::CursorGrabMode::Confined);
                            }
                            window.set_cursor_visible(false);
                            mouse_delta.lock().accum = (0.0, 0.0);
                        }
                    }
                    WindowEvent::MouseWheel { delta, .. } => {
                        if mouse_grabbed {
                            let lines = match delta {
                                winit::event::MouseScrollDelta::LineDelta(_, y) => y as f64,
                                winit::event::MouseScrollDelta::PixelDelta(p)  => p.y / scroll_pixels_per_line,
                            };
                            mouse_delta.lock().wheel += lines;
                        }
                    }
                    WindowEvent::Focused(false) => {
                        if mouse_grabbed {
                            mouse_grabbed = false;
                            let _ = window.set_cursor_grab(winit::window::CursorGrabMode::None);
                            window.set_cursor_visible(true);
                        }
                    }
                    WindowEvent::RedrawRequested => {
                        // Rendering is driven by the Rex3 refresh thread
                    }
                    _ => (),
                },
                Event::DeviceEvent { event: winit::event::DeviceEvent::MouseMotion { delta }, .. } => {
                    if mouse_grabbed {
                        let mut md = mouse_delta.lock();
                        md.accum.0 += delta.0 / scale as f64;
                        md.accum.1 += delta.1 / scale as f64;
                    }
                },
                _ => (),
            }
        }).unwrap();
    }

    fn flush_mouse_delta(ps2: &Ps2Controller, mouse_delta: &Mutex<MouseDelta>, require_motion: bool) {
        let mut md = mouse_delta.lock();
        let dx = md.accum.0 as i32;
        let dy = md.accum.1 as i32;
        let dz = md.wheel as i32;
        if require_motion && dx == 0 && dy == 0 && dz == 0 { return; }
        md.accum.0 -= dx as f64;
        md.accum.1 -= dy as f64;
        md.wheel   -= dz as f64;
        let buttons = md.buttons;
        drop(md);
        ps2.push_mouse_input(buttons, dx, dy, dz);
    }

    /// Adjust an incoming window size to match the emulated display's aspect
    /// ratio (display width : display height + status bar). Whichever axis the
    /// user is actively dragging — the one that moved most from `prev` — is
    /// kept, and the other is derived from it. Returns `None` when the size is
    /// already within 1 px of the target (no correction needed), which keeps
    /// the follow-up resize from oscillating.
    fn aspect_fit(win_w: u32, win_h: u32, prev: (u32, u32), disp_w: u32, disp_h: u32)
        -> Option<(u32, u32)>
    {
        if disp_w == 0 || disp_h == 0 { return None; }
        let content_h = disp_h + STATUS_BAR_HEIGHT as u32;
        // round(a * b / c) in u64 to avoid overflow/bias.
        let muldiv = |a: u32, b: u32, c: u32| -> u32 {
            ((a as u64 * b as u64 + c as u64 / 2) / c as u64) as u32
        };
        let (pw, ph) = prev;
        if win_w.abs_diff(pw) >= win_h.abs_diff(ph) {
            // Width is the driven axis: derive height from it.
            let target_h = muldiv(win_w, content_h, disp_w).max(1);
            if target_h.abs_diff(win_h) <= 1 { None } else { Some((win_w, target_h)) }
        } else {
            // Height is the driven axis: derive width from it.
            let target_w = muldiv(win_h, disp_w, content_h).max(1);
            if target_w.abs_diff(win_w) <= 1 { None } else { Some((target_w, win_h)) }
        }
    }

    fn handle_keyboard(ps2: &Ps2Controller, rex3: &Rex3, scsi: &Wd33c93a, scale_snap: &Mutex<Option<ScaleSnap>>,
        input: KeyEvent, grabbed: &mut bool, rctrl_held: &mut bool, window: &Window)
    {
        use std::sync::atomic::Ordering;
        if let PhysicalKey::Code(keycode) = input.physical_key {
            let pressed = input.state == ElementState::Pressed;

            if keycode == KeyCode::ControlRight {
                *rctrl_held = pressed;
                if pressed && !input.repeat && *grabbed {
                    *grabbed = false;
                    let _ = window.set_cursor_grab(winit::window::CursorGrabMode::None);
                    window.set_cursor_visible(true);
                }
                return;
            }

            if keycode == KeyCode::PrintScreen && pressed && !input.repeat && *rctrl_held {
                rex3.screenshot_pending.store(true, Ordering::Relaxed);
                return;
            }

            if keycode == KeyCode::F11 && pressed && !input.repeat && *rctrl_held {
                let new_mode = if window.fullscreen().is_some() {
                    None
                } else {
                    Some(winit::window::Fullscreen::Borderless(None))
                };
                window.set_fullscreen(new_mode);
                return;
            }

            // RCtrl+F12: hot-swap CD-ROM disc (open file picker and load into first CD-ROM device)
            if keycode == KeyCode::F12 && pressed && !input.repeat && *rctrl_held {
                // Find the first CD-ROM device (disc_status only lists CD-ROMs)
                let cdrom_id = scsi.disc_status().first().map(|(id, ..)| *id);

                if let Some(id) = cdrom_id {
                    // Open file picker (blocks the event loop but winit tolerates it on most platforms)
                    if let Some(path) = rfd::FileDialog::new()
                        .set_title("Load CD-ROM disc")
                        .add_filter("ISO images", &["iso", "chd"])
                        .add_filter("All", &["*"])
                        .pick_file()
                    {
                        let path_str = path.to_string_lossy().into_owned();
                        match scsi.load_disc(id, path_str.clone()) {
                            Ok(_) => {
                                let filename = path.file_name()
                                    .map(|n| n.to_string_lossy().into_owned())
                                    .unwrap_or_else(|| path_str.clone());
                                eprintln!("SCSI #{}: loaded {}", id, filename);
                            }
                            Err(e) => {
                                eprintln!("SCSI #{}: {}", id, e);
                            }
                        }
                    }
                } else {
                    eprintln!("No CD-ROM drive attached");
                }
                return;
            }

            // RCtrl+1 / RCtrl+2: snap window to 1x or 2x scale.
            if pressed && !input.repeat && *rctrl_held {
                let snap = match keycode {
                    KeyCode::Digit1 => Some(ScaleSnap::Scale1x),
                    KeyCode::Digit2 => Some(ScaleSnap::Scale2x),
                    _ => None,
                };
                if let Some(s) = snap {
                    *scale_snap.lock() = Some(s);
                    return;
                }
            }

            ps2.push_kb(keycode, pressed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Indy default mode. Content aspect = 1280 : (1024 + 16) = 1280 : 1040.
    const DW: u32 = 1280;
    const DH: u32 = 1024;

    #[test]
    fn dragging_width_derives_height() {
        // Width grows from 1280 to 2560; height untouched. Expect height locked
        // to the content ratio: 2560 * 1040 / 1280 = 2080.
        let fixed = Ui::aspect_fit(2560, 1040, (1280, 1040), DW, DH);
        assert_eq!(fixed, Some((2560, 2080)));
    }

    #[test]
    fn dragging_height_derives_width() {
        // Height grows from 1040 to 2080; width untouched. Expect width locked:
        // 2080 * 1280 / 1040 = 2560.
        let fixed = Ui::aspect_fit(1280, 2080, (1280, 1040), DW, DH);
        assert_eq!(fixed, Some((2560, 2080)));
    }

    #[test]
    fn already_locked_is_noop() {
        // A size already on-ratio needs no correction (prevents oscillation on
        // the follow-up Resized event after we apply a fix).
        assert_eq!(Ui::aspect_fit(2560, 2080, (2560, 2080), DW, DH), None);
        assert_eq!(Ui::aspect_fit(1280, 1040, (1280, 1040), DW, DH), None);
    }

    #[test]
    fn within_one_pixel_tolerance() {
        // 1 px off the exact ratio is accepted as-is (no visible letterbox).
        assert_eq!(Ui::aspect_fit(2560, 2079, (2560, 2079), DW, DH), None);
        assert_eq!(Ui::aspect_fit(2560, 2081, (2560, 2081), DW, DH), None);
    }

    #[test]
    fn zero_resolution_is_noop() {
        // Guard against a divide-by-zero before the first frame publishes a res.
        assert_eq!(Ui::aspect_fit(800, 600, (800, 600), 0, 0), None);
    }
}
