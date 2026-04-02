use std::sync::Arc;
use std::time::{Duration, Instant};
use parking_lot::Mutex;
use winit::{
    dpi::PhysicalPosition,
    event::{ElementState, Event, KeyEvent, WindowEvent, MouseButton}, event_loop::{ControlFlow, EventLoop}, keyboard::{KeyCode, PhysicalKey}, window::{Window, WindowBuilder}
};
use glow::HasContext;
use crate::ps2::Ps2Controller;
use crate::rex3::{Rex3, Renderer};
use crate::disp::STATUS_BAR_HEIGHT;
use crate::hptimer::{TimerManager, TimerReturn};
use glutin::config::ConfigTemplateBuilder;
use glutin::context::{ContextAttributesBuilder, PossiblyCurrentContext};
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
    // Main emulator framebuffer texture (2048 × 1024, opaque)
    main_tex: glow::Texture,
    // Debug overlay texture (2048 × 1024, alpha-blended over main)
    overlay_tex: glow::Texture,
    // Status bar texture (2048 × STATUS_BAR_HEIGHT, opaque)
    statusbar_tex: glow::Texture,
    // Single program used for all three passes (texelFetch y-flip via uniform)
    program: glow::Program,
    viewport_info_loc: Option<glow::UniformLocation>,
    // Shared VAO + two VBOs: one for the emulator quad, one for the statusbar quad
    vao: glow::VertexArray,
    main_vbo: glow::Buffer,    // quad covering top `height` px of window
    status_vbo: glow::Buffer,  // quad covering bottom STATUS_BAR_HEIGHT px of window
}

struct GlRenderer {
    window: Arc<Window>,
    gl_config: glutin::config::Config,
    window_size: Arc<Mutex<Option<(u32, u32)>>>,
    state: Option<GlState>,
    current_w: usize,
    current_h: usize,
}

// Safety: GlRenderer is sent to the refresh thread where it initializes and uses the GL context.
// The context is never accessed from multiple threads simultaneously or from a thread other than the one it was created on.
unsafe impl Send for GlRenderer {}

impl GlRenderer {
    fn init_gl(&mut self) {
        let raw_window_handle = self.window.raw_window_handle();
        let gl_display = self.gl_config.display();

        let context_attributes = ContextAttributesBuilder::new().build(Some(raw_window_handle));
        let not_current_gl_context = unsafe {
            gl_display
                .create_context(&self.gl_config, &context_attributes)
                .expect("failed to create context")
        };

        let attrs = self.window.build_surface_attributes(Default::default());
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

        // Enable VSync
        let _ = gl_surface.set_swap_interval(&gl_context, SwapInterval::Wait(NonZeroU32::new(1).unwrap()));

        let (main_tex, overlay_tex, statusbar_tex, program, viewport_info_loc, vao, main_vbo, status_vbo) = unsafe {
            // Helper: allocate a 2D texture with NEAREST filtering
            let make_tex = |w: i32, h: i32| -> glow::Texture {
                let t = gl.create_texture().unwrap();
                gl.bind_texture(glow::TEXTURE_2D, Some(t));
                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
                gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA as i32, w, h, 0, glow::RGBA, glow::UNSIGNED_BYTE, None);
                t
            };

            let main_tex      = make_tex(2048, 1024);
            let overlay_tex   = make_tex(2048, 1024);
            let statusbar_tex = make_tex(2048, STATUS_BAR_HEIGHT as i32);

            // Single shader program used for all passes.
            // viewport_info[0] = (tex_width, tex_height) — dimensions of the texture region being sampled
            // viewport_info[1] = (0, window_y_base) — bottom pixel row of this quad in window coords
            // Y flip: tex_y = (tex_height - 1) - (gl_FragCoord.y - window_y_base)
            let vs_src = "
                #version 150
                in vec2 position;
                in vec2 tex_coord;
                out vec2 v_tex_coord;
                void main() {
                    gl_Position = vec4(position, 0.0, 1.0);
                    v_tex_coord = tex_coord;
                }
            ";

            let program = gl.create_program().unwrap();
            let vs = gl.create_shader(glow::VERTEX_SHADER).unwrap();
            gl.shader_source(vs, vs_src);
            gl.compile_shader(vs);
            if !gl.get_shader_compile_status(vs) {
                panic!("Vertex shader compilation failed: {}", gl.get_shader_info_log(vs));
            }

            let fs = gl.create_shader(glow::FRAGMENT_SHADER).unwrap();
            // Try GLSL 1.50 first (texelFetch, Core Profile compatible)
            gl.shader_source(fs, "
                #version 150
                in vec2 v_tex_coord;
                out vec4 color;
                uniform sampler2D tex;
                uniform ivec2 viewport_info[2];
                void main() {
                    int tex_h      = viewport_info[0].y;
                    int win_y_base = viewport_info[1].y;
                    int y = (tex_h - 1) - (int(gl_FragCoord.y) - win_y_base);
                    color = texelFetch(tex, ivec2(int(gl_FragCoord.x), y), 0);
                }
            ");
            gl.compile_shader(fs);

            let mut linked = false;
            if gl.get_shader_compile_status(fs) {
                gl.attach_shader(program, vs);
                gl.attach_shader(program, fs);
                gl.link_program(program);
                if gl.get_program_link_status(program) {
                    linked = true;
                } else {
                    gl.detach_shader(program, fs);
                }
            }

            if !linked {
                // Fallback (UV-based, no texelFetch)
                gl.shader_source(fs, "
                    #version 150
                    in vec2 v_tex_coord;
                    out vec4 color;
                    uniform sampler2D tex;
                    void main() {
                        color = texture(tex, v_tex_coord);
                    }
                ");
                gl.compile_shader(fs);
                if !gl.get_shader_compile_status(fs) {
                    panic!("Fragment shader compilation failed: {}", gl.get_shader_info_log(fs));
                }
                gl.attach_shader(program, vs);
                gl.attach_shader(program, fs);
                gl.link_program(program);
                if !gl.get_program_link_status(program) {
                    panic!("Program linking failed: {}", gl.get_program_info_log(program));
                }
            }

            let viewport_info_loc = gl.get_uniform_location(program, "viewport_info");

            let vao = gl.create_vertex_array().unwrap();
            gl.bind_vertex_array(Some(vao));

            gl.use_program(Some(program));
            let pos_loc = gl.get_attrib_location(program, "position").unwrap();
            let tex_loc = gl.get_attrib_location(program, "tex_coord").unwrap();

            // main_vbo: emulator quad
            let main_vbo = gl.create_buffer().unwrap();
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(main_vbo));
            gl.buffer_data_size(glow::ARRAY_BUFFER, VBO_SIZE, glow::DYNAMIC_DRAW);
            gl.enable_vertex_attrib_array(pos_loc);
            gl.vertex_attrib_pointer_f32(pos_loc, 2, glow::FLOAT, false, 16, 0);
            gl.enable_vertex_attrib_array(tex_loc);
            gl.vertex_attrib_pointer_f32(tex_loc, 2, glow::FLOAT, false, 16, 8);

            // status_vbo: status bar quad (separate buffer, same attrib layout)
            let status_vbo = gl.create_buffer().unwrap();
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(status_vbo));
            gl.buffer_data_size(glow::ARRAY_BUFFER, VBO_SIZE, glow::DYNAMIC_DRAW);

            (main_tex, overlay_tex, statusbar_tex, program, viewport_info_loc, vao, main_vbo, status_vbo)
        };

        self.state = Some(GlState {
            gl,
            context: gl_context,
            surface: gl_surface,
            main_tex,
            overlay_tex,
            statusbar_tex,
            program,
            viewport_info_loc,
            vao,
            main_vbo,
            status_vbo,
        });
    }

    // Upload pixel data to a texture (row stride = 2048 in buffer).
    unsafe fn upload_tex(gl: &glow::Context, tex: glow::Texture, buf: &[u32], w: i32, h: i32) {
        gl.bind_texture(glow::TEXTURE_2D, Some(tex));
        gl.pixel_store_i32(glow::UNPACK_ROW_LENGTH, 2048);
        let u8_slice = std::slice::from_raw_parts(buf.as_ptr() as *const u8, buf.len() * 4);
        gl.tex_sub_image_2d(glow::TEXTURE_2D, 0, 0, 0, w, h, glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelUnpackData::Slice(u8_slice));
        gl.pixel_store_i32(glow::UNPACK_ROW_LENGTH, 0);
    }

    // Upload quad vertices covering NDC [x0..x1] × [y0..y1], sampling UV [u0..u1] × [v0..v1].
    unsafe fn upload_quad(gl: &glow::Context, vbo: glow::Buffer, x0: f32, y0: f32, x1: f32, y1: f32, u0: f32, v0: f32, u1: f32, v1: f32) {
        let vertices: [f32; 16] = [
            x0, y0,  u0, v1,
            x1, y0,  u1, v1,
            x0, y1,  u0, v0,
            x1, y1,  u1, v0,
        ];
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
        let u8_slice = std::slice::from_raw_parts(vertices.as_ptr() as *const u8, vertices.len() * 4);
        gl.buffer_sub_data_u8_slice(glow::ARRAY_BUFFER, 0, u8_slice);
    }

    // Set viewport_info uniform: tex dimensions + window y-base of this quad.
    unsafe fn set_viewport_info(gl: &glow::Context, loc: Option<&glow::UniformLocation>, tex_w: i32, tex_h: i32, win_y_base: i32) {
        let info = [tex_w, tex_h, 0, win_y_base];
        gl.uniform_2_i32_slice(loc, &info);
    }

    // Bind the VAO and set up attribs for a given VBO (both share the same layout).
    unsafe fn bind_vbo_attribs(gl: &glow::Context, program: glow::Program, vbo: glow::Buffer) {
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
        let pos_loc = gl.get_attrib_location(program, "position").unwrap();
        let tex_loc = gl.get_attrib_location(program, "tex_coord").unwrap();
        gl.enable_vertex_attrib_array(pos_loc);
        gl.vertex_attrib_pointer_f32(pos_loc, 2, glow::FLOAT, false, 16, 0);
        gl.enable_vertex_attrib_array(tex_loc);
        gl.vertex_attrib_pointer_f32(tex_loc, 2, glow::FLOAT, false, 16, 8);
    }
}

impl Renderer for GlRenderer {
    fn render(&mut self, buffer: &[u32], width: usize, height: usize) {
        if self.state.is_none() {
            self.init_gl();
        }

        let state = self.state.as_mut().unwrap();
        let gl = &state.gl;

        // Handle window resize
        let win_h = if let Some((w, h)) = self.window_size.lock().take() {
            state.surface.resize(
                &state.context,
                NonZeroU32::new(w).unwrap(),
                NonZeroU32::new(h).unwrap(),
            );
            unsafe { gl.viewport(0, 0, w as i32, h as i32); }
            h as usize
        } else {
            height + STATUS_BAR_HEIGHT
        };

        unsafe {
            gl.use_program(Some(state.program));
            gl.bind_vertex_array(Some(state.vao));

            // Recompute quads if emulator resolution changed
            if width != self.current_w || height != self.current_h {
                self.current_w = width;
                self.current_h = height;

                let max_u = width as f32 / 2048.0;
                let max_v_main = height as f32 / 1024.0;
                let max_v_sb   = 1.0f32; // status bar texture is exactly STATUS_BAR_HEIGHT tall

                // NDC y of the boundary between emulator display and status bar.
                // Window: y=0 (bottom) = status bar bottom, y=win_h (top) = display top.
                // Status bar: bottom STATUS_BAR_HEIGHT px → NDC [-1 .. sb_ndc_top]
                let sb_ndc_top = -1.0 + 2.0 * STATUS_BAR_HEIGHT as f32 / win_h as f32;

                // Emulator quad: NDC [sb_ndc_top .. +1], full width
                Self::upload_quad(gl, state.main_vbo,
                    -1.0, sb_ndc_top, 1.0, 1.0,
                    0.0, 0.0, max_u, max_v_main);

                // Status bar quad: NDC [-1 .. sb_ndc_top], full width
                Self::upload_quad(gl, state.status_vbo,
                    -1.0, -1.0, 1.0, sb_ndc_top,
                    0.0, 0.0, max_u, max_v_sb);
            }

            // --- Pass 1: main framebuffer (opaque) ---
            gl.disable(glow::BLEND);
            Self::upload_tex(gl, state.main_tex, buffer, width as i32, height as i32);
            Self::bind_vbo_attribs(gl, state.program, state.main_vbo);
            // win_y_base for main quad = STATUS_BAR_HEIGHT (bottom of emulator area in window coords)
            Self::set_viewport_info(gl, state.viewport_info_loc.as_ref(),
                width as i32, height as i32, STATUS_BAR_HEIGHT as i32);
            gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);

            // swap_buffers called in render_overlay after all passes
        }
    }

    fn render_overlay(&mut self, buffer: &[u32], width: usize, height: usize, _overlay_extra_rows: usize) {
        if self.state.is_none() { return; }
        let state = self.state.as_mut().unwrap();
        let gl = &state.gl;

        unsafe {
            gl.use_program(Some(state.program));
            gl.bind_vertex_array(Some(state.vao));

            // --- Pass 2: debug overlay (alpha-blended over main) ---
            gl.enable(glow::BLEND);
            gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);
            Self::upload_tex(gl, state.overlay_tex, buffer, width as i32, height as i32);
            Self::bind_vbo_attribs(gl, state.program, state.main_vbo);
            Self::set_viewport_info(gl, state.viewport_info_loc.as_ref(),
                width as i32, height as i32, STATUS_BAR_HEIGHT as i32);
            gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);

            gl.disable(glow::BLEND);
        }
    }

    fn render_statusbar(&mut self, buffer: &[u32], width: usize) {
        if self.state.is_none() { return; }
        let state = self.state.as_mut().unwrap();
        let gl = &state.gl;

        unsafe {
            gl.use_program(Some(state.program));
            gl.bind_vertex_array(Some(state.vao));

            // --- Pass 3: status bar (opaque, bottom of window) ---
            gl.disable(glow::BLEND);
            Self::upload_tex(gl, state.statusbar_tex, buffer, width as i32, STATUS_BAR_HEIGHT as i32);
            Self::bind_vbo_attribs(gl, state.program, state.status_vbo);
            // win_y_base = 0: status bar sits at the very bottom of the window
            Self::set_viewport_info(gl, state.viewport_info_loc.as_ref(),
                width as i32, STATUS_BAR_HEIGHT as i32, 0);
            gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);

            state.surface.swap_buffers(&state.context).unwrap();
        }
    }

    fn resize(&mut self, width: usize, height: usize) {
        // Resize window to display + status bar
        let _ = self.window.request_inner_size(winit::dpi::PhysicalSize::new(
            width as u32,
            (height + STATUS_BAR_HEIGHT) as u32,
        ));
    }

    fn stop(&mut self) {
        self.state = None;
        self.current_w = 0;
        self.current_h = 0;
    }
}

struct MouseDelta {
    accum: (f64, f64),
    buttons: u8,
}

/// UI Manager handling Window, OpenGL context, and Input
pub struct Ui {
    ps2: Arc<Ps2Controller>,
    window: Arc<Window>,
    window_size: Arc<Mutex<Option<(u32, u32)>>>,
    timer_manager: Arc<TimerManager>,
}

impl Ui {
    pub fn new(ps2: Arc<Ps2Controller>, rex3: Arc<Rex3>, timer_manager: Arc<TimerManager>, event_loop: &EventLoop<()>) -> Self {
        let window_builder = WindowBuilder::new()
            .with_title("Irresponsible Rust IRIX Simulator")
            .with_inner_size(winit::dpi::LogicalSize::new(1024.0, 768.0 + STATUS_BAR_HEIGHT as f64));

        let template = ConfigTemplateBuilder::new()
            .with_alpha_size(8)
            .with_transparency(false);

        let display_builder = DisplayBuilder::new().with_window_builder(Some(window_builder));

        let (window, gl_config) = display_builder
            .build(event_loop, template, |configs| {
                configs
                    .reduce(|accum, config| {
                        if config.num_samples() > accum.num_samples() {
                            config
                        } else {
                            accum
                        }
                    })
                    .unwrap()
            })
            .unwrap();

        let window = Arc::new(window.unwrap());
        let window_size = Arc::new(Mutex::new(None));

        let renderer = GlRenderer {
            window: window.clone(),
            gl_config,
            window_size: window_size.clone(),
            state: None,
            current_w: 0,
            current_h: 0,
        };

        *rex3.renderer.lock() = Some(Box::new(renderer));

        Self { ps2, window, window_size, timer_manager }
    }

    /// Run the UI event loop (blocks the current thread)
    pub fn run(self, event_loop: EventLoop<()>) {
        let Ui { ps2, window, window_size, timer_manager } = self;

        let mut mouse_grabbed = false;
        // Warp-to-center mouse handling: on each real CursorMoved, accumulate
        // delta into shared MouseDelta. A 10ms timer flushes it to PS/2.
        let mut mouse_last: Option<PhysicalPosition<f64>> = None;
        let mouse_delta = Arc::new(Mutex::new(MouseDelta { accum: (0.0, 0.0), buttons: 0 }));

        // 10ms recurring timer: flush accumulated mouse delta to PS/2.
        {
            let ps2 = ps2.clone();
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

        event_loop.set_control_flow(ControlFlow::Poll);
        event_loop.run(move |event, elwt| {
            match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => { elwt.exit() },
                    WindowEvent::Resized(size) => {
                        if size.width != 0 && size.height != 0 {
                            *window_size.lock() = Some((size.width, size.height));
                        }
                    }
                    WindowEvent::KeyboardInput { event, .. } => {
                        Self::handle_keyboard(&ps2, event, &mut mouse_grabbed, &window);
                    }
                    WindowEvent::MouseInput { state, button, .. } => {
                        if mouse_grabbed {
                            let pressed = state == ElementState::Pressed;
                            let mask = match button {
                                MouseButton::Left => 1,
                                MouseButton::Right => 2,
                                MouseButton::Middle => 4,
                                _ => 0,
                            };
                            if mask != 0 {
                                // Update button state first, then flush — the flush packet
                                // carries both the accumulated motion and the new button state.
                                let mut md = mouse_delta.lock();
                                if pressed { md.buttons |= mask; } else { md.buttons &= !mask; }
                                drop(md);
                                Self::flush_mouse_delta(&ps2, &mouse_delta, false);
                            }
                        }
                        else if state == ElementState::Pressed && button == MouseButton::Left {
                            mouse_grabbed = true;
                            if window.set_cursor_grab(winit::window::CursorGrabMode::Locked).is_err() {
                                let _ = window.set_cursor_grab(winit::window::CursorGrabMode::Confined);
                            }
                            window.set_cursor_visible(false);
                            // Reset warp and delta state on grab.
                            mouse_last = None;
                            mouse_delta.lock().accum = (0.0, 0.0);
                        }
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        if mouse_grabbed {
                            let size = window.inner_size();
                            let center = PhysicalPosition::new((size.width / 2) as f64, (size.height / 2) as f64);

                            // Skip events at exactly center — those are our own warps.
                            if position.x == center.x && position.y == center.y {
                                mouse_last = Some(center);
                            } else if let Some(last) = mouse_last {
                                let dx = position.x - last.x;
                                let dy = position.y - last.y;
                                mouse_delta.lock().accum.0 += dx;
                                mouse_delta.lock().accum.1 += dy;
                                let _ = window.set_cursor_position(center);
                                mouse_last = Some(center);
                            } else {
                                // First event after grab: warp to center, record it.
                                let _ = window.set_cursor_position(center);
                                mouse_last = Some(center);
                            }
                        }
                    }
                    WindowEvent::RedrawRequested => {
                        // Rendering is handled by Rex3 refresh thread
                    }
                    _ => (),
                },
                _ => (),
            }
        }).unwrap();
    }

    /// Flush accumulated delta to PS/2. If `require_motion` is true, skips
    /// sending when there is no accumulated movement (used by the timer).
    fn flush_mouse_delta(ps2: &Ps2Controller, mouse_delta: &Mutex<MouseDelta>, require_motion: bool) {
        let mut md = mouse_delta.lock();
        let dx = md.accum.0 as i32;
        let dy = md.accum.1 as i32;
        if require_motion && dx == 0 && dy == 0 {
            return;
        }
        md.accum.0 -= dx as f64;
        md.accum.1 -= dy as f64;
        let buttons = md.buttons;
        drop(md);
        Self::send_mouse_packet(ps2, (dx as f64, dy as f64), buttons);
    }

    fn handle_keyboard(ps2: &Ps2Controller, input: KeyEvent, grabbed: &mut bool, window: &Window) {
        if let PhysicalKey::Code(keycode) = input.physical_key {
            // Right Ctrl: ungrab mouse — consumed, not forwarded to PS/2
            if keycode == KeyCode::ControlRight {
                if input.state == ElementState::Pressed && !input.repeat && *grabbed {
                    *grabbed = false;
                    let _ = window.set_cursor_grab(winit::window::CursorGrabMode::None);
                    window.set_cursor_visible(true);
                }
                return;
            }

            // Pass to PS/2
            let pressed = input.state == ElementState::Pressed;
            ps2.push_kb(keycode, pressed);
        }
    }

    fn send_mouse_packet(ps2: &Ps2Controller, delta: (f64, f64), buttons: u8) {
        // PS/2 mouse packet (3 bytes):
        //   Byte 0: Yovfl Xovfl Ysign Xsign 1 M R L
        //   Byte 1: X movement (low 8 bits of 9-bit signed value)
        //   Byte 2: Y movement (low 8 bits of 9-bit signed value)
        // The sign bit in byte 0 is the 9th bit, giving range -256..=255.
        // Overflow bits are set when the value exceeds that range.

        let dx = delta.0 as i32;
        let dy = -(delta.1 as i32); // PS/2 Y is bottom-to-top

        // Split movements that exceed the 9-bit signed range (-256..=255).
        let max_axis = dx.unsigned_abs().max(dy.unsigned_abs()) as i32;
        let limit = 255i32;
        let steps = if max_axis > limit { (max_axis + limit - 1) / limit } else { 1 };
        let step_x = dx / steps;
        let step_y = dy / steps;
        let rem_x = dx - step_x * (steps - 1);
        let rem_y = dy - step_y * (steps - 1);

        let mut send = |sx: i32, sy: i32| {
            let mut b0 = 0x08u8 | (buttons & 0x07);
            if sx < 0 { b0 |= 0x10; }
            if sy < 0 { b0 |= 0x20; }
            if sx < -256 || sx > 255 { b0 |= 0x40; }
            if sy < -256 || sy > 255 { b0 |= 0x80; }
            ps2.push_mouse_packet(b0, sx as u8, sy as u8);
        };

        for _ in 0..steps - 1 {
            send(step_x, step_y);
        }
        send(rem_x, rem_y);
    }
}
