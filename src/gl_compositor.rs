use glow::HasContext;
use crate::compositor::{Compositor, CompositorSource};
use crate::vc2::{
    VC2_REG_CURRENT_CURSOR_X, VC2_REG_WORKING_CURSOR_Y, VC2_REG_CURSOR_ENTRY_PTR,
    VC2_REG_DISPLAY_CONTROL, VC2_CTRL_CURSOR_EN, VC2_CTRL_CURSOR_SIZE,
};

const FB_W: i32 = 2048;
const FB_H: i32 = 1024;
const CURSOR_TEX_SIZE: i32 = 64;

const VERT_SRC: &str = "
#version 150
void main() {
    vec2 pos[3];
    pos[0] = vec2(-1.0, -1.0);
    pos[1] = vec2( 3.0, -1.0);
    pos[2] = vec2(-1.0,  3.0);
    gl_Position = vec4(pos[gl_VertexID], 0.0, 1.0);
}
";

// Lookup tables (cmap/ramdac/xmap) are 2D textures with height=1 since
// glow does not expose tex_sub_image_1d. Use texelFetch(..., ivec2(addr, 0), 0).
const FRAG_SRC: &str = "
#version 150

uniform usampler2D u_rgb;
uniform usampler2D u_aux;
uniform usampler2D u_did;
uniform usampler2D u_cursor;
uniform usampler2D u_cmap;
uniform usampler2D u_ramdac;
uniform usampler2D u_xmap;

uniform int  u_topscan;
uniform int  u_cursor_x_hot;
uniform int  u_cursor_y_hot;
uniform bool u_cursor_en;
uniform bool u_cursor_size_64;
uniform uint u_cursor_cmap_msb;
uniform uint u_popup_cmap_msb;
uniform int  u_width;
uniform int  u_height;

out vec4 frag_color;

uint me_buf_sel(uint m)      { return (m >> 0u) & 1u; }
uint me_ovl_buf_sel(uint m)  { return (m >> 1u) & 1u; }
uint me_msb_cmap(uint m)     { return (m >> 3u) & 0x1Fu; }
uint me_pix_mode(uint m)     { return (m >> 8u) & 3u; }
uint me_pix_size(uint m)     { return (m >> 10u) & 3u; }
uint me_aux_pix_mode(uint m) { return (m >> 16u) & 7u; }
uint me_aux_msb_cmap(uint m) { return (m >> 19u) & 0x1Fu; }

uint expand_4(uint v) {
    uint b  = (v >> 3u) & 1u;
    uint g1 = (v >> 2u) & 1u;
    uint g0 = (v >> 1u) & 1u;
    uint r  =  v        & 1u;
    uint g  = (0xAAu * g1) | (0x55u * g0);
    return (b * 255u) << 16u | (g << 8u) | (r * 255u);
}

uint expand_8(uint v) {
    uint b1 = (v >> 7u) & 1u; uint b0 = (v >> 6u) & 1u;
    uint g2 = (v >> 5u) & 1u; uint g1 = (v >> 4u) & 1u; uint g0 = (v >> 3u) & 1u;
    uint r2 = (v >> 2u) & 1u; uint r1 = (v >> 1u) & 1u; uint r0 =  v        & 1u;
    uint b = (0xAAu * b1) | (0x55u * b0);
    uint g = (0x92u * g2) | (0x49u * g1) | (0x24u * g0);
    uint r = (0x92u * r2) | (0x49u * r1) | (0x24u * r0);
    return (b << 16u) | (g << 8u) | r;
}

uint expand_12(uint v) {
    uint b = (v >> 8u) & 0xFu;
    uint g = (v >> 4u) & 0xFu;
    uint r =  v        & 0xFu;
    return (b * 0x11u) << 16u | (g * 0x11u) << 8u | (r * 0x11u);
}

void main() {
    int x = int(gl_FragCoord.x);
    // No y-flip here: the screen shader (ui.rs) already flips Y when reading
    // this FBO texture, so store row 0 = top of image (y=0 = display row 0).
    int y = int(gl_FragCoord.y);

    if (x >= u_width || y >= u_height || x < 0 || y < 0) {
        frag_color = vec4(0.0);
        return;
    }

    int fb_y  = (u_topscan + 1 + y) & 0x3FF;
    int fb_x  = x;

    uint did5 = texelFetch(u_did, ivec2(fb_x, y), 0).r & 0x1Fu;
    uint mode = texelFetch(u_xmap, ivec2(int(did5), 0), 0).r;

    uint buf_sel      = me_buf_sel(mode);
    uint ovl_buf_sel  = me_ovl_buf_sel(mode);
    uint msb_cmap     = me_msb_cmap(mode);
    uint pix_mode     = me_pix_mode(mode);
    uint pix_size     = me_pix_size(mode);
    uint aux_pix_mode = me_aux_pix_mode(mode);
    uint aux_msb_cmap = me_aux_msb_cmap(mode);

    uint raw_rgb = texelFetch(u_rgb, ivec2(fb_x, fb_y), 0).r;
    uint raw_aux = texelFetch(u_aux, ivec2(fb_x, fb_y), 0).r;

    uint pup     = (raw_aux >> 2u) & 3u;
    uint overlay = (ovl_buf_sel != 0u) ? ((raw_aux >> 16u) & 0xFFu)
                                       : ((raw_aux >>  8u) & 0xFFu);

    uint shift;
    uint mask;
    if (pix_size == 0u)      { shift = (buf_sel != 0u) ?  4u : 0u; mask = 0xFu; }
    else if (pix_size == 1u) { shift = (buf_sel != 0u) ?  8u : 0u; mask = 0xFFu; }
    else if (pix_size == 2u) { shift = (buf_sel != 0u) ? 12u : 0u; mask = 0xFFFu; }
    else                     { shift = 0u; mask = 0xFFFFFFu; }
    uint pixel = (raw_rgb >> shift) & mask;

    // Cursor hit-test
    uint cursor_pixel = 0u;
    if (u_cursor_en) {
        int cx = x - u_cursor_x_hot;
        int cy = y - u_cursor_y_hot;
        if (cx >= 0 && cy >= 0) {
            if (u_cursor_size_64) {
                if (cx < 64 && cy < 64)
                    cursor_pixel = texelFetch(u_cursor, ivec2(cx, cy), 0).r;
            } else {
                if (cx < 32 && cy < 32)
                    cursor_pixel = texelFetch(u_cursor, ivec2(cx, cy), 0).r;
            }
        }
    }

    // Priority composition + palette lookup
    uint rgb;
    if (cursor_pixel != 0u) {
        uint addr = (u_cursor_cmap_msb << 5u) | cursor_pixel;
        rgb = texelFetch(u_cmap, ivec2(int(addr), 0), 0).r & 0xFFFFFFu;
    } else if (pup != 0u) {
        uint addr = (u_popup_cmap_msb << 5u) | pup;
        rgb = texelFetch(u_cmap, ivec2(int(addr), 0), 0).r & 0xFFFFFFu;
    } else if ((aux_pix_mode == 2u || aux_pix_mode == 6u || aux_pix_mode == 7u) && overlay != 0u) {
        uint addr = (aux_msb_cmap << 8u) | overlay;
        rgb = texelFetch(u_cmap, ivec2(int(addr), 0), 0).r & 0xFFFFFFu;
    } else if (pix_mode == 0u) {
        uint addr;
        if      (pix_size == 2u) addr = ((msb_cmap & 0x10u) << 8u) | pixel;
        else if (pix_size == 3u) addr = pixel;
        else                     addr = (msb_cmap << 8u) | pixel;
        rgb = texelFetch(u_cmap, ivec2(int(addr), 0), 0).r & 0xFFFFFFu;
    } else {
        if      (pix_size == 0u) rgb = expand_4(pixel);
        else if (pix_size == 1u) rgb = expand_8(pixel);
        else if (pix_size == 2u) rgb = expand_12(pixel);
        else                     rgb = pixel & 0xFFFFFFu;
    }

    // 1:1 with SW compositor: r_in = bits23:16, g_in = bits15:8, b_in = bits7:0
    uint r_in = (rgb >> 16u) & 0xFFu;
    uint g_in = (rgb >>  8u) & 0xFFu;
    uint b_in =  rgb         & 0xFFu;
    uint r_out = (texelFetch(u_ramdac, ivec2(int(r_in), 0), 0).r >> 16u) & 0xFFu;
    uint g_out = (texelFetch(u_ramdac, ivec2(int(g_in), 0), 0).r >>  8u) & 0xFFu;
    uint b_out =  texelFetch(u_ramdac, ivec2(int(b_in), 0), 0).r         & 0xFFu;

    // SW writes 0xFF000000|(r_out<<16)|(g_out<<8)|b_out; LE bytes = [b_out,g_out,r_out,FF].
    // GL RGBA8 FBO stores bytes [R,G,B,A]; screen shader reads them back as-is.
    // So output vec4(b_out, g_out, r_out, 1) to store [b_out,g_out,r_out,FF] = same layout.
    frag_color = vec4(float(b_out) / 255.0, float(g_out) / 255.0, float(r_out) / 255.0, 1.0);
}
";

pub struct GlCompositor {
    tex_rgb:    Option<glow::Texture>,
    tex_aux:    Option<glow::Texture>,
    tex_did:    Option<glow::Texture>,
    tex_cursor: Option<glow::Texture>,
    tex_cmap:   Option<glow::Texture>,
    tex_ramdac: Option<glow::Texture>,
    tex_xmap:   Option<glow::Texture>,
    tex_out:    Option<glow::Texture>,
    fbo:        Option<glow::Framebuffer>,
    program:    Option<glow::Program>,
    vao:        Option<glow::VertexArray>,

    last_cmap_hash:   u64,
    last_ramdac_hash: u64,
    last_xmap_hash:   u64,
    last_cursor_hash: u64,

    /// Reusable PBO for async readback (iris-gui GL capture path).
    readback_pbo: std::cell::RefCell<Option<(glow::Buffer, usize)>>,
}

impl GlCompositor {
    pub fn new() -> Self {
        Self {
            tex_rgb:    None,
            tex_aux:    None,
            tex_did:    None,
            tex_cursor: None,
            tex_cmap:   None,
            tex_ramdac: None,
            tex_xmap:   None,
            tex_out:    None,
            fbo:        None,
            program:    None,
            vao:        None,
            last_cmap_hash:   u64::MAX,
            last_ramdac_hash: u64::MAX,
            last_xmap_hash:   u64::MAX,
            last_cursor_hash: u64::MAX,
            readback_pbo: std::cell::RefCell::new(None),
        }
    }

    fn compile_program(gl: &glow::Context) -> Result<glow::Program, String> {
        unsafe {
            let program = gl.create_program().map_err(|e| e.to_string())?;

            let vs = gl.create_shader(glow::VERTEX_SHADER).map_err(|e| e.to_string())?;
            gl.shader_source(vs, VERT_SRC);
            gl.compile_shader(vs);
            if !gl.get_shader_compile_status(vs) {
                let log = gl.get_shader_info_log(vs);
                gl.delete_shader(vs);
                return Err(format!("GL compositor vertex shader: {}", log));
            }

            let fs = gl.create_shader(glow::FRAGMENT_SHADER).map_err(|e| e.to_string())?;
            gl.shader_source(fs, FRAG_SRC);
            gl.compile_shader(fs);
            if !gl.get_shader_compile_status(fs) {
                let log = gl.get_shader_info_log(fs);
                gl.delete_shader(vs);
                gl.delete_shader(fs);
                return Err(format!("GL compositor fragment shader: {}", log));
            }

            gl.attach_shader(program, vs);
            gl.attach_shader(program, fs);
            gl.link_program(program);
            gl.detach_shader(program, vs);
            gl.detach_shader(program, fs);
            gl.delete_shader(vs);
            gl.delete_shader(fs);

            if !gl.get_program_link_status(program) {
                let log = gl.get_program_info_log(program);
                gl.delete_program(program);
                return Err(format!("GL compositor program link: {}", log));
            }

            Ok(program)
        }
    }

    fn make_2d_tex_r32ui(gl: &glow::Context, w: i32, h: i32) -> glow::Texture {
        unsafe {
            let t = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(t));
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
            gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::R32UI as i32, w, h, 0,
                glow::RED_INTEGER, glow::UNSIGNED_INT, glow::PixelUnpackData::Slice(None));
            t
        }
    }

    fn make_2d_tex_r8ui(gl: &glow::Context, w: i32, h: i32) -> glow::Texture {
        unsafe {
            let t = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(t));
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
            gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::R8UI as i32, w, h, 0,
                glow::RED_INTEGER, glow::UNSIGNED_BYTE, glow::PixelUnpackData::Slice(None));
            t
        }
    }

    fn init_resources(&mut self, gl: &glow::Context) -> bool {
        if self.program.is_some() { return true; }

        match Self::compile_program(gl) {
            Ok(prog) => { self.program = Some(prog); }
            Err(e)   => { eprintln!("GlCompositor: shader compile failed, falling back to SW: {}", e); return false; }
        }

        unsafe {
            let vao = gl.create_vertex_array().unwrap();
            self.vao = Some(vao);

            self.tex_rgb    = Some(Self::make_2d_tex_r32ui(gl, FB_W, FB_H));
            self.tex_aux    = Some(Self::make_2d_tex_r32ui(gl, FB_W, FB_H));
            self.tex_did    = Some(Self::make_2d_tex_r8ui (gl, FB_W, FB_H));
            self.tex_cursor = Some(Self::make_2d_tex_r8ui (gl, CURSOR_TEX_SIZE, CURSOR_TEX_SIZE));
            // Lookup tables: height=1, width=N (since tex_sub_image_1d is not in glow)
            self.tex_cmap   = Some(Self::make_2d_tex_r32ui(gl, 8192, 1));
            self.tex_ramdac = Some(Self::make_2d_tex_r32ui(gl, 256,  1));
            self.tex_xmap   = Some(Self::make_2d_tex_r32ui(gl, 32,   1));

            // FBO output: RGBA8
            let tex_out = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(tex_out));
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
            gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA as i32,
                FB_W, FB_H, 0, glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelUnpackData::Slice(None));
            self.tex_out = Some(tex_out);

            let fbo = gl.create_framebuffer().unwrap();
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
            gl.framebuffer_texture_2d(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0,
                glow::TEXTURE_2D, self.tex_out, 0);
            let status = gl.check_framebuffer_status(glow::FRAMEBUFFER);
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            if status != glow::FRAMEBUFFER_COMPLETE {
                eprintln!("GlCompositor: FBO incomplete ({}), falling back to SW", status);
                self.destroy(gl);
                return false;
            }
            self.fbo = Some(fbo);
        }

        true
    }

    fn upload_fb_u32(gl: &glow::Context, tex: glow::Texture, data: &[u32], w: i32, h: i32) {
        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            gl.pixel_store_i32(glow::UNPACK_ROW_LENGTH, FB_W);
            let bytes = std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4);
            gl.tex_sub_image_2d(
                glow::TEXTURE_2D, 0, 0, 0, w, h,
                glow::RED_INTEGER, glow::UNSIGNED_INT,
                glow::PixelUnpackData::Slice(Some(bytes)),
            );
            gl.pixel_store_i32(glow::UNPACK_ROW_LENGTH, 0);
        }
    }

    fn upload_fb_u8(gl: &glow::Context, tex: glow::Texture, data: &[u8], w: i32, h: i32) {
        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            gl.pixel_store_i32(glow::UNPACK_ROW_LENGTH, FB_W);
            gl.tex_sub_image_2d(
                glow::TEXTURE_2D, 0, 0, 0, w, h,
                glow::RED_INTEGER, glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(data)),
            );
            gl.pixel_store_i32(glow::UNPACK_ROW_LENGTH, 0);
        }
    }

    fn upload_lut(gl: &glow::Context, tex: glow::Texture, data: &[u32]) {
        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            let bytes = std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4);
            gl.tex_sub_image_2d(
                glow::TEXTURE_2D, 0, 0, 0, data.len() as i32, 1,
                glow::RED_INTEGER, glow::UNSIGNED_INT,
                glow::PixelUnpackData::Slice(Some(bytes)),
            );
        }
    }

    fn bind_sampler(gl: &glow::Context, program: glow::Program, name: &str, unit: u32, tex: glow::Texture) {
        unsafe {
            gl.active_texture(glow::TEXTURE0 + unit);
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            if let Some(loc) = gl.get_uniform_location(program, name) {
                gl.uniform_1_i32(Some(&loc), unit as i32);
            }
        }
    }
}

fn hash_u32_slice(data: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &w in data {
        h ^= w as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn extract_cursor(vc2_ram: &[u16], entry: usize, size_64: bool) -> [u8; 64 * 64] {
    let mut out = [0u8; 64 * 64];
    if size_64 {
        for cy in 0..64usize {
            for cx in 0..64usize {
                let addr = entry + cy * 4 + cx / 16;
                if addr < vc2_ram.len() {
                    let shift = 15 - (cx % 16);
                    out[cy * 64 + cx] = ((vc2_ram[addr] >> shift) & 1) as u8;
                }
            }
        }
    } else {
        for cy in 0..32usize {
            for cx in 0..32usize {
                let addr = entry + cy * 2 + cx / 16;
                if addr + 64 < vc2_ram.len() {
                    let shift = 15 - (cx % 16);
                    let bit0 = (vc2_ram[addr]      >> shift) & 1;
                    let bit1 = (vc2_ram[addr + 64] >> shift) & 1;
                    out[cy * 64 + cx] = (bit0 | (bit1 << 1)) as u8;
                }
            }
        }
    }
    out
}

impl Compositor for GlCompositor {
    fn compose(&mut self, src: &CompositorSource<'_>, gl: &glow::Context) -> glow::Texture {
        if !self.init_resources(gl) {
            panic!("GlCompositor::compose called after init_resources failed");
        }

        let w = src.width as i32;
        let h = src.height as i32;

        let program    = self.program.unwrap();
        let tex_rgb    = self.tex_rgb.unwrap();
        let tex_aux    = self.tex_aux.unwrap();
        let tex_did    = self.tex_did.unwrap();
        let tex_cursor = self.tex_cursor.unwrap();
        let tex_cmap   = self.tex_cmap.unwrap();
        let tex_ramdac = self.tex_ramdac.unwrap();
        let tex_xmap   = self.tex_xmap.unwrap();
        let tex_out    = self.tex_out.unwrap();
        let fbo        = self.fbo.unwrap();
        let vao        = self.vao.unwrap();

        // ── Upload per-frame buffers ─────────────────────────────────────────────
        // tex_rgb/tex_aux must be uploaded in full: fb_y = (topscan+1+y)&0x3FF can
        // wrap to any framebuffer row regardless of the visible window height, so
        // uploading only [0,h) leaves rows [h,FB_H) stale whenever topscan != 0.
        Self::upload_fb_u32(gl, tex_rgb, src.fb_rgb, FB_W, FB_H);
        Self::upload_fb_u32(gl, tex_aux, src.fb_aux, FB_W, FB_H);
        if !src.status_bar_only {
            // tex_did is indexed by display row `y` directly (not fb_y), so the
            // visible w×h window is all that's ever sampled.
            Self::upload_fb_u8(gl, tex_did, src.did, w, h);
        }

        // ── Upload lookup tables (skip if unchanged) ───────────────────────────
        let cmap_hash = hash_u32_slice(src.cmap);
        if cmap_hash != self.last_cmap_hash {
            Self::upload_lut(gl, tex_cmap, src.cmap);
            self.last_cmap_hash = cmap_hash;
        }

        let ramdac_hash = hash_u32_slice(src.ramdac);
        if ramdac_hash != self.last_ramdac_hash {
            Self::upload_lut(gl, tex_ramdac, src.ramdac);
            self.last_ramdac_hash = ramdac_hash;
        }

        let xmap_hash = hash_u32_slice(src.xmap_mode);
        if xmap_hash != self.last_xmap_hash {
            Self::upload_lut(gl, tex_xmap, src.xmap_mode);
            self.last_xmap_hash = xmap_hash;
        }

        // ── Upload cursor glyph (skip if unchanged) ────────────────────────────
        let cursor_entry   = src.vc2_regs[VC2_REG_CURSOR_ENTRY_PTR   as usize] as usize;
        let display_ctrl   = src.vc2_regs[VC2_REG_DISPLAY_CONTROL    as usize];
        let cursor_en      = (display_ctrl & VC2_CTRL_CURSOR_EN)  != 0;
        let cursor_size_64 = (display_ctrl & VC2_CTRL_CURSOR_SIZE) != 0;

        let cursor_x_reg = src.vc2_regs[VC2_REG_CURRENT_CURSOR_X as usize];
        let cursor_y_reg = src.vc2_regs[VC2_REG_WORKING_CURSOR_Y as usize];
        let cursor_x_hot = (cursor_x_reg as i32) - 31 + src.cursor_x_adjust;
        let cursor_y_hot = (cursor_y_reg as i32) - 31;

        if cursor_en {
            let words_per_entry = if cursor_size_64 { 64 * 4 } else { 32 * 2 + 32 * 2 };
            let slice_start = cursor_entry.min(src.vc2_ram.len());
            let slice_end   = (cursor_entry + words_per_entry).min(src.vc2_ram.len());
            let cursor_slice = &src.vc2_ram[slice_start..slice_end];
            let mut ch: u64 = 0xcbf29ce484222325u64
                .wrapping_add(cursor_entry as u64)
                .wrapping_add(if cursor_size_64 { 1 } else { 0 });
            for &word in cursor_slice {
                ch ^= word as u64;
                ch = ch.wrapping_mul(0x100000001b3);
            }

            if ch != self.last_cursor_hash {
                let glyph = extract_cursor(src.vc2_ram, cursor_entry, cursor_size_64);
                unsafe {
                    gl.bind_texture(glow::TEXTURE_2D, Some(tex_cursor));
                    gl.tex_sub_image_2d(
                        glow::TEXTURE_2D, 0, 0, 0,
                        CURSOR_TEX_SIZE, CURSOR_TEX_SIZE,
                        glow::RED_INTEGER, glow::UNSIGNED_BYTE,
                        glow::PixelUnpackData::Slice(Some(&glyph)),
                    );
                }
                self.last_cursor_hash = ch;
            }
        }

        // ── Render into FBO ────────────────────────────────────────────────────
        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
            gl.viewport(0, 0, w, h);
            gl.use_program(Some(program));
            gl.bind_vertex_array(Some(vao));

            Self::bind_sampler(gl, program, "u_rgb",    0, tex_rgb);
            Self::bind_sampler(gl, program, "u_aux",    1, tex_aux);
            Self::bind_sampler(gl, program, "u_did",    2, tex_did);
            Self::bind_sampler(gl, program, "u_cursor", 3, tex_cursor);
            Self::bind_sampler(gl, program, "u_cmap",   4, tex_cmap);
            Self::bind_sampler(gl, program, "u_ramdac", 5, tex_ramdac);
            Self::bind_sampler(gl, program, "u_xmap",   6, tex_xmap);

            let unif = |name: &str| gl.get_uniform_location(program, name);

            gl.uniform_1_i32(unif("u_topscan").as_ref(),        src.topscan as i32);
            gl.uniform_1_i32(unif("u_cursor_x_hot").as_ref(),   cursor_x_hot);
            gl.uniform_1_i32(unif("u_cursor_y_hot").as_ref(),   cursor_y_hot);
            gl.uniform_1_i32(unif("u_cursor_en").as_ref(),      cursor_en as i32);
            gl.uniform_1_i32(unif("u_cursor_size_64").as_ref(), cursor_size_64 as i32);
            gl.uniform_1_u32(unif("u_cursor_cmap_msb").as_ref(), src.xmap_cursor_cmap as u32);
            gl.uniform_1_u32(unif("u_popup_cmap_msb").as_ref(),  src.xmap_popup_cmap as u32);
            gl.uniform_1_i32(unif("u_width").as_ref(),  w);
            gl.uniform_1_i32(unif("u_height").as_ref(), h);

            gl.disable(glow::BLEND);
            gl.draw_arrays(glow::TRIANGLES, 0, 3);

            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.active_texture(glow::TEXTURE0);
        }

        tex_out
    }

    fn destroy(&mut self, gl: &glow::Context) {
        unsafe {
            if let Some(t) = self.tex_rgb.take()    { gl.delete_texture(t); }
            if let Some(t) = self.tex_aux.take()    { gl.delete_texture(t); }
            if let Some(t) = self.tex_did.take()    { gl.delete_texture(t); }
            if let Some(t) = self.tex_cursor.take() { gl.delete_texture(t); }
            if let Some(t) = self.tex_cmap.take()   { gl.delete_texture(t); }
            if let Some(t) = self.tex_ramdac.take() { gl.delete_texture(t); }
            if let Some(t) = self.tex_xmap.take()   { gl.delete_texture(t); }
            if let Some(t) = self.tex_out.take()    { gl.delete_texture(t); }
            if let Some(f) = self.fbo.take()        { gl.delete_framebuffer(f); }
            if let Some(p) = self.program.take()    { gl.delete_program(p); }
            if let Some(v) = self.vao.take()        { gl.delete_vertex_array(v); }
            if let Some((pbo, _)) = self.readback_pbo.borrow_mut().take() {
                gl.delete_buffer(pbo);
            }
        }
        self.last_cmap_hash   = u64::MAX;
        self.last_ramdac_hash = u64::MAX;
        self.last_xmap_hash   = u64::MAX;
        self.last_cursor_hash = u64::MAX;
    }

    fn read_pixels(&self) -> Option<&[u32]> {
        None
    }

    fn readback_to_screen(&self, dst: &mut [u32], width: usize, height: usize, gl: &glow::Context) {
        let Some(fbo) = self.fbo else { return; };
        let row_bytes = width * 4;
        let total_bytes = row_bytes * height;
        let mut tight: Vec<u8> = vec![0u8; total_bytes];
        unsafe {
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(fbo));

            let mut pbo_guard = self.readback_pbo.borrow_mut();
            let use_pbo = if let Some((pbo, cap)) = pbo_guard.as_mut() {
                if *cap < total_bytes {
                    gl.delete_buffer(*pbo);
                    *pbo_guard = None;
                    false
                } else {
                    gl.bind_buffer(glow::PIXEL_PACK_BUFFER, Some(*pbo));
                    gl.read_pixels(
                        0,
                        0,
                        width as i32,
                        height as i32,
                        glow::RGBA,
                        glow::UNSIGNED_BYTE,
                        glow::PixelPackData::BufferOffset(0),
                    );
                    let mapped = gl.map_buffer_range(
                        glow::PIXEL_PACK_BUFFER,
                        0,
                        total_bytes as i32,
                        glow::MAP_READ_BIT,
                    );
                    if !mapped.is_null() {
                        std::ptr::copy_nonoverlapping(
                            mapped,
                            tight.as_mut_ptr(),
                            total_bytes,
                        );
                        let _ = gl.unmap_buffer(glow::PIXEL_PACK_BUFFER);
                    }
                    gl.bind_buffer(glow::PIXEL_PACK_BUFFER, None);
                    true
                }
            } else {
                false
            };

            if !use_pbo {
                gl.read_pixels(
                    0,
                    0,
                    width as i32,
                    height as i32,
                    glow::RGBA,
                    glow::UNSIGNED_BYTE,
                    glow::PixelPackData::Slice(Some(&mut tight)),
                );
                if pbo_guard.is_none() {
                    if let Ok(pbo) = gl.create_buffer() {
                        gl.bind_buffer(glow::PIXEL_PACK_BUFFER, Some(pbo));
                        gl.buffer_data_size(
                            glow::PIXEL_PACK_BUFFER,
                            total_bytes as i32,
                            glow::DYNAMIC_READ,
                        );
                        gl.bind_buffer(glow::PIXEL_PACK_BUFFER, None);
                        *pbo_guard = Some((pbo, total_bytes));
                    }
                }
            }

            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None);
        }
        // FBO gl_FragCoord.y=0 = display row 0 (top); glReadPixels row 0 = that same bottom.
        // No flip: src row y → dst row y.
        for y in 0..height {
            let src_start = y * row_bytes;
            let dst_start = y * 2048;
            for x in 0..width {
                let r = tight[src_start + x * 4    ] as u32;
                let g = tight[src_start + x * 4 + 1] as u32;
                let b = tight[src_start + x * 4 + 2] as u32;
                let a = tight[src_start + x * 4 + 3] as u32;
                dst[dst_start + x] = (a << 24) | (b << 16) | (g << 8) | r;
            }
        }
    }
}
