//! The SGI OpenGL extensions the Mac's OpenGL does not have.
//!
//! A high-end SGI's GL had texturing and fog features that no later hardware
//! kept: a colour table between the texel and the fragment, detail and sharpen
//! textures, a four-tap filter, fog by a table of the program's own points, a
//! depth taken from a plane. The Mac has none of them, and refuses even their
//! enumerants.
//!
//! They are done here in the fragment stage. IRIX's own fixed-function
//! pipeline is left alone -- matrices, lighting, the fog the program set up,
//! the vertex path -- and when one of these features is switched on, a
//! fragment shader replaces only the last step, reading that same fixed state
//! through GLSL's built-ins (`gl_TexCoord`, `gl_Color`, `gl_Fog`). When none
//! is on, no shader is bound and nothing about a program's rendering changes.
//!
//! Each shader is built for the state it must reproduce -- whether texturing
//! is on, the texture environment's mode, whether fog is on -- and kept, so a
//! program that turns a feature on and off pays for the build once.

use std::collections::HashMap;
use std::ffi::{c_char, c_void, CString};

#[cfg_attr(target_os = "macos", link(name = "OpenGL", kind = "framework"))]
#[cfg_attr(target_os = "linux", link(name = "GL"))]
extern "C" {
    fn glCreateShader(kind: u32) -> u32;
    fn glShaderSource(s: u32, count: i32, strings: *const *const c_char, lengths: *const i32);
    fn glCompileShader(s: u32);
    fn glGetShaderiv(s: u32, pname: u32, out: *mut i32);
    fn glGetShaderInfoLog(s: u32, max: i32, len: *mut i32, log: *mut c_char);
    fn glCreateProgram() -> u32;
    fn glAttachShader(p: u32, s: u32);
    fn glLinkProgram(p: u32);
    fn glGetProgramiv(p: u32, pname: u32, out: *mut i32);
    fn glGetProgramInfoLog(p: u32, max: i32, len: *mut i32, log: *mut c_char);
    fn glUseProgram(p: u32);
    fn glGetUniformLocation(p: u32, name: *const c_char) -> i32;
    fn glUniform1i(loc: i32, v: i32);
    fn glUniform4f(loc: i32, a: f32, b: f32, c: f32, d: f32);
    fn glUniform1f(loc: i32, v: f32);
    fn glActiveTexture(unit: u32);
    fn glGenTextures(n: i32, out: *mut u32);
    fn glBindTexture(target: u32, tex: u32);
    fn glTexImage1D(target: u32, level: i32, internal: i32, w: i32, border: i32, format: u32, ty: u32, data: *const c_void);
    fn glTexImage2D(target: u32, level: i32, internal: i32, w: i32, h: i32, border: i32, format: u32, ty: u32, data: *const c_void);
    fn glTexImage3D(target: u32, level: i32, internal: i32, w: i32, h: i32, d: i32, border: i32, format: u32, ty: u32, data: *const c_void);
    fn glTexSubImage2D(target: u32, level: i32, x: i32, y: i32, w: i32, h: i32, format: u32, ty: u32, data: *const c_void);
    fn glPushMatrix();
    fn glPopMatrix();
    fn glTranslatef(x: f32, y: f32, z: f32);
    fn glMultMatrixf(m: *const f32);
    fn glGetFloatv(pname: u32, out: *mut f32);
    fn glMatrixMode(mode: u32);
    fn glLoadIdentity();
    fn glOrtho(l: f64, r: f64, b: f64, t: f64, n: f64, f: f64);
    fn glBegin(mode: u32);
    fn glEnd();
    fn glTexCoord2f(s: f32, t: f32);
    fn glVertex2f(x: f32, y: f32);
    fn glTexParameteri(target: u32, pname: u32, v: i32);
    fn glGetTexLevelParameteriv(target: u32, level: i32, pname: u32, out: *mut i32);
    fn glUniform2f(loc: i32, a: f32, b: f32);
    fn glGetIntegerv(pname: u32, out: *mut i32);
    fn glGetDoublev(pname: u32, out: *mut f64);
}

const FRAGMENT_SHADER: u32 = 0x8B30;
const COMPILE_STATUS: u32 = 0x8B81;
const LINK_STATUS: u32 = 0x8B82;
const GL_TEXTURE_1D: u32 = 0x0DE0;
const GL_TEXTURE0: u32 = 0x84C0;
const GL_TEXTURE1: u32 = 0x84C1;
const GL_RGBA: u32 = 0x1908;
const GL_UNSIGNED_BYTE: u32 = 0x1401;
const GL_TEXTURE_MIN_FILTER: u32 = 0x2801;
const GL_TEXTURE_MAG_FILTER: u32 = 0x2800;
const GL_TEXTURE_WRAP_S: u32 = 0x2802;
const GL_LINEAR: i32 = 0x2601;
const GL_CLAMP_TO_EDGE: i32 = 0x812F;
const GL_ACTIVE_TEXTURE: u32 = 0x84E0;

/// The state the shader has to reproduce, and the features it adds.
pub const GL_TEXTURE_2D: u32 = 0x0DE1;
pub const GL_FOG: u32 = 0x0B60;
/// SGI_texture_color_table.
pub const GL_TEXTURE_COLOR_TABLE_SGI: u32 = 0x80BC;
/// SGIX_texture_scale_bias: what a texel is multiplied by and added to after
/// the filter, kept per texture as the extension defines it.
pub const GL_POST_TEXTURE_FILTER_SCALE_SGIX: u32 = 0x817A;
pub const GL_POST_TEXTURE_FILTER_BIAS_SGIX: u32 = 0x8179;
/// SGIS_fog_function: fog by a table of the program's own points, and
/// SGIX_fog_offset: fog shifted towards the eye.
pub const GL_FOG_FUNC_SGIS: u32 = 0x812A;
pub const GL_FOG_OFFSET_SGIX: u32 = 0x8198;
/// SGIX_reference_plane: a plane that gives the fragment its depth.
/// SGIS_texture_filter4: a four-tap filter with weights the program gives.
pub const GL_FILTER4_SGIS: u32 = 0x8146;
/// SGIS_sharpen_texture: magnification that extrapolates away from level 1.
const GL_LINEAR_SHARPEN_SGIS: u32 = 0x80AD;
const GL_LINEAR_SHARPEN_ALPHA_SGIS: u32 = 0x80AE;
const GL_LINEAR_SHARPEN_COLOR_SGIS: u32 = 0x80AF;
const GL_TEXTURE4: u32 = 0x84C4;
const GL_TEXTURE5: u32 = 0x84C5;
/// SGIS_detail_texture: a second texture added in by level-of-detail.
pub const GL_DETAIL_TEXTURE_2D_SGIS: u32 = 0x8095;
const GL_LINEAR_DETAIL_SGIS: u32 = 0x8097;
const GL_LINEAR_DETAIL_ALPHA_SGIS: u32 = 0x8098;
const GL_LINEAR_DETAIL_COLOR_SGIS: u32 = 0x8099;
const GL_DETAIL_TEXTURE_LEVEL_SGIS: u32 = 0x809A;
const GL_DETAIL_TEXTURE_MODE_SGIS: u32 = 0x809B;
const GL_ADD_ENUM: u32 = 0x0104;
/// SGIS_texture_select: internal formats that hold two or four textures'
/// worth of components, one group of which is selected per texture.
const GL_DUAL_ALPHA4_SGIS: u32 = 0x8110;
const GL_DUAL_LUMINANCE_ALPHA8_SGIS: u32 = 0x811D;
const GL_QUAD_ALPHA4_SGIS: u32 = 0x811E;
const GL_QUAD_INTENSITY8_SGIS: u32 = 0x8123;
const GL_DUAL_TEXTURE_SELECT_SGIS: u32 = 0x8124;
/// SGIS_texture4D: a fourth texture axis, kept as slabs of a 3D texture.
pub const GL_TEXTURE_4D_SGIS: u32 = 0x8134;
/// SGIX_sprite: geometry turned to face the eye before the modelview.
/// SGIX_pixel_texture: a transferred pixel's colour used as texture
/// coordinates, so an image can index a lookup table held as a texture.
/// SGIX_clipmap: a virtual texture far larger than memory, of which a window
/// around a moving centre is resident.
const GL_LINEAR_CLIPMAP_LINEAR_SGIX: u32 = 0x8170;
const GL_TEXTURE_CLIPMAP_CENTER_SGIX: u32 = 0x8171;
const GL_TEXTURE_CLIPMAP_FRAME_SGIX: u32 = 0x8172;
const GL_TEXTURE_CLIPMAP_OFFSET_SGIX: u32 = 0x8173;
const GL_TEXTURE_CLIPMAP_VIRTUAL_DEPTH_SGIX: u32 = 0x8174;
const GL_TEXTURE_CLIPMAP_LOD_OFFSET_SGIX: u32 = 0x8175;
const GL_TEXTURE_CLIPMAP_DEPTH_SGIX: u32 = 0x8176;
const GL_LINEAR_MIPMAP_LINEAR: i32 = 0x2703;
pub const GL_PIXEL_TEX_GEN_SGIX: u32 = 0x8139;
const GL_TEXTURE8: u32 = 0x84C8;
const GL_CURRENT_RASTER_POSITION: u32 = 0x0B07;
const GL_CURRENT_RASTER_COLOR: u32 = 0x0B04;
const GL_VIEWPORT_ENUM: u32 = 0x0BA2;
pub const GL_SPRITE_SGIX: u32 = 0x8148;
const GL_SPRITE_MODE_SGIX: u32 = 0x8149;
const GL_SPRITE_AXIS_SGIX: u32 = 0x814A;
const GL_SPRITE_TRANSLATION_SGIX: u32 = 0x814B;
const GL_SPRITE_AXIAL_SGIX: u32 = 0x814C;
const GL_SPRITE_EYE_ALIGNED_SGIX: u32 = 0x814E;
const GL_TEXTURE_3D: u32 = 0x806F;
const GL_TEXTURE7: u32 = 0x84C7;
const GL_QUAD_TEXTURE_SELECT_SGIS: u32 = 0x8125;
pub const GL_TEXTURE_FILTER4_SIZE_SGIS: u32 = 0x8147;
const GL_NEAREST: i32 = 0x2600;
const GL_LUMINANCE: u32 = 0x1909;
const GL_LUMINANCE32F_ARB: i32 = 0x8818;
const GL_FLOAT: u32 = 0x1406;
const GL_TEXTURE_WIDTH: u32 = 0x1000;
const GL_TEXTURE_HEIGHT: u32 = 0x1001;
const GL_TEXTURE3: u32 = 0x84C3;
pub const GL_REFERENCE_PLANE_SGIX: u32 = 0x817D;
pub const GL_REFERENCE_PLANE_EQUATION_SGIX: u32 = 0x817E;
const GL_MODELVIEW_MATRIX: u32 = 0x0BA6;
const GL_PROJECTION_MATRIX: u32 = 0x0BA7;
const GL_VIEWPORT: u32 = 0x0BA2;
pub const GL_FOG_OFFSET_VALUE_SGIX: u32 = 0x8199;
const GL_FOG_MODE: u32 = 0x0B65;
const GL_LINEAR_FOG: u32 = 0x2601;
const GL_EXP_FOG: u32 = 0x0800;
const GL_EXP2_FOG: u32 = 0x0801;
const GL_TEXTURE2: u32 = 0x84C2;

/// Texture environment modes the shader knows.
const GL_MODULATE: u32 = 0x2100;
const GL_REPLACE: u32 = 0x1E01;
const GL_DECAL: u32 = 0x2101;
const GL_BLEND: u32 = 0x0BE2;
const GL_ADD: u32 = 0x0104;

#[derive(Default)]
pub struct Emul {
    /// What the program has switched on, of the state the shader cares about.
    texture_2d: bool,
    fog: bool,
    color_table: bool,
    /// GL_TEXTURE_ENV_MODE, which the shader has to reproduce once it is the
    /// one deciding the fragment's colour.
    env_mode: u32,
    /// The colour table, as a 1D texture, and whether it holds anything.
    table_tex: u32,
    table_ready: bool,
    programs: HashMap<u64, u32>,
    current: u32,
    reported: bool,
    /// Scale and bias per texture (SGIX_texture_scale_bias), and which
    /// texture is bound to the unit the shader samples.
    scale_bias: HashMap<u32, ([f32; 4], [f32; 4])>,
    bound_texture: u32,
    /// The fog the program asked for. Once a shader owns the fragment stage
    /// the fixed fog is bypassed, so whichever mode is in force has to be
    /// worked out here.
    fog_mode: u32,
    fog_offset: bool,
    fog_offset_value: f32,
    /// SGIS_fog_function's points, resampled into a 1D texture, and the
    /// distances its ends stand for.
    fog_func_tex: u32,
    fog_func_range: (f32, f32),
    fog_func_ready: bool,
    /// SGIX_reference_plane: the plane in clip coordinates, and whether the
    /// program has switched it on. The equation the program gives is in object
    /// coordinates and is transformed once, when it is given -- as the
    /// specification says, and as glClipPlane does -- so a later change of
    /// matrix does not move a plane that was already placed.
    ref_plane: [f32; 4],
    ref_plane_on: bool,
    ref_plane_set: bool,
    /// SGIS_texture_filter4: which textures the program asked to filter this
    /// way, and the weights, resampled into a 1D float texture. The weights
    /// go in a float texture rather than the byte one the colour table uses
    /// because a sharpening kernel's are negative and greater than one.
    filter4: std::collections::HashSet<u32>,
    filter4_tex: u32,
    filter4_ready: bool,
    filter4_n: f32,
    /// SGIS_sharpen_texture: which textures the program asked to magnify this
    /// way and with which of the three filters, and the function F of
    /// level-of-detail, resampled into a 1D float texture with the range of
    /// LOD its ends stand for. F is signed and unbounded, like filter4's
    /// weights, so it needs the float texture too.
    sharpen: HashMap<u32, u32>,
    sharpen_tex: u32,
    sharpen_range: (f32, f32),
    sharpen_ready: bool,
    /// SGIS_detail_texture: which textures want it, the detail image itself
    /// (one texture object of ours, uploaded when the program names the
    /// DETAIL_TEXTURE_2D_SGIS target), the F-of-LOD table, how many levels
    /// finer the detail image is, and whether it adds or modulates.
    detail: HashMap<u32, u32>,
    detail_tex: u32,
    detail_ready: bool,
    detail_func_tex: u32,
    detail_range: (f32, f32),
    detail_func_ready: bool,
    detail_level: f32,
    detail_add: bool,
    /// SGIS_texture_select: per texture, which grouped format it was given
    /// and which group the program selected.
    select_fmt: HashMap<u32, u32>,
    select_group: HashMap<u32, u32>,
    /// SGIS_texture4D: the 3D texture the four-dimensional image is kept in,
    /// how deep one slab is, and how many slabs there are.
    tex4d: u32,
    tex4d_depth: f32,
    tex4d_size: f32,
    tex4d_on: bool,
    tex4d_ready: bool,
    /// SGIX_sprite: the mode, the axis it keeps upright and the offset from
    /// the object's origin, plus whether a sprite matrix is currently pushed.
    sprite_on: bool,
    sprite_mode: u32,
    sprite_axis: [f32; 3],
    sprite_translation: [f32; 3],
    sprite_pushed: bool,
    /// SGIX_pixel_texture: whether it is on, which components the fragment's
    /// colour comes from, the texture the transferred image is staged in, and
    /// the program that reads it.
    pixel_tex_on: bool,
    pixel_tex_mode: u32,
    pixel_tex_img: u32,
    pixel_tex_prog: HashMap<u32, u32>,
    /// SGIX_clipmap, as far as it goes here: the parameters are kept and
    /// answered, and the filter falls back to ordinary mipmapping. See
    /// `clipmap_faithful` for what that does and does not cover.
    clipmap: HashMap<u32, [f32; 10]>,
    /// The resident levels of the clipmap, one above the other in a single
    /// image: level P occupies rows [P*C, (P+1)*C) of a C wide, C*L tall
    /// texture. This is the whole trick -- an OpenGL mipmap chain halves every
    /// level and so cannot hold the equal-sized windows a clipmap is made of,
    /// but nothing stops us laying them out ourselves.
    clipmap_tex: u32,
    clipmap_c: i32,
    clipmap_levels: i32,
    clipmap_v: f32,
    clipmap_ready: bool,
}

impl Emul {
    /// Is `cap` one this layer answers for? The Mac refuses these enumerants,
    /// so they must not reach it.
    pub fn owns(cap: u32) -> bool {
        cap == GL_TEXTURE_COLOR_TABLE_SGI
            || cap == GL_FOG_OFFSET_SGIX
            || cap == GL_REFERENCE_PLANE_SGIX
            || cap == GL_TEXTURE_4D_SGIS
            || cap == GL_SPRITE_SGIX
            || cap == GL_PIXEL_TEX_GEN_SGIX
    }

    /// True when a shader is doing the fragment stage, so glGet of the
    /// features' state comes from here.
    pub fn is_enabled(&self, cap: u32) -> bool {
        match cap {
            GL_TEXTURE_COLOR_TABLE_SGI => self.color_table,
            GL_FOG_OFFSET_SGIX => self.fog_offset,
            GL_REFERENCE_PLANE_SGIX => self.ref_plane_on,
            GL_TEXTURE_4D_SGIS => self.tex4d_on,
            GL_SPRITE_SGIX => self.sprite_on,
            GL_PIXEL_TEX_GEN_SGIX => self.pixel_tex_on,
            _ => false,
        }
    }

    /// Record state the shader reproduces or adds. Returns whether the
    /// fragment stage has to be rebuilt.
    pub fn set_enable(&mut self, cap: u32, on: bool) -> bool {
        match cap {
            GL_TEXTURE_COLOR_TABLE_SGI => self.color_table = on,
            GL_FOG_OFFSET_SGIX => self.fog_offset = on,
            GL_REFERENCE_PLANE_SGIX => self.ref_plane_on = on,
            GL_TEXTURE_2D => self.texture_2d = on,
            GL_TEXTURE_4D_SGIS => self.tex4d_on = on,
            GL_SPRITE_SGIX => self.sprite_on = on,
            GL_PIXEL_TEX_GEN_SGIX => self.pixel_tex_on = on,
            GL_FOG => self.fog = on,
            _ => return false,
        }
        true
    }

    pub fn set_env_mode(&mut self, mode: u32) {
        self.env_mode = mode;
    }

    /// glFog*: the mode a shader would have to reproduce, and the two SGI
    /// parameters the Mac has not got. True when this layer took the call.
    pub fn set_fog(&mut self, pname: u32, values: &[f32]) -> bool {
        match pname {
            GL_FOG_MODE => {
                self.fog_mode = values.first().copied().unwrap_or(0.0) as u32;
                // SGI's table mode is ours; the rest the Mac knows as well.
                self.fog_mode == GL_FOG_FUNC_SGIS
            }
            GL_FOG_OFFSET_VALUE_SGIX => {
                // Four values: the offset's point and its distance.
                self.fog_offset_value = values.get(3).copied().unwrap_or(0.0);
                true
            }
            _ => false,
        }
    }

    /// SGIS_fog_function's points: (distance, fog factor), increasing. They
    /// become a 1D texture the shader reads by distance.
    pub fn set_fog_func(&mut self, points: &[f32]) {
        let n = points.len() / 2;
        if n < 2 {
            self.fog_func_ready = false;
            return;
        }
        let (x0, x1) = (points[0], points[(n - 1) * 2]);
        if !(x1 > x0) {
            self.fog_func_ready = false;
            return;
        }
        // One entry per step across the range, each the factor at that distance.
        const STEPS: usize = 256;
        let mut table = vec![0u8; STEPS * 4];
        for (i, out) in table.chunks_exact_mut(4).enumerate() {
            let x = x0 + (x1 - x0) * i as f32 / (STEPS - 1) as f32;
            let mut f = points[1];
            for k in 1..n {
                let (xa, ya) = (points[(k - 1) * 2], points[(k - 1) * 2 + 1]);
                let (xb, yb) = (points[k * 2], points[k * 2 + 1]);
                if x >= xa && x <= xb && xb > xa {
                    f = ya + (yb - ya) * (x - xa) / (xb - xa);
                    break;
                }
                f = yb;
            }
            let v = (f.clamp(0.0, 1.0) * 255.0) as u8;
            out[0] = v;
            out[1] = v;
            out[2] = v;
            out[3] = 255;
        }
        unsafe {
            let mut unit = 0i32;
            glGetIntegerv(GL_ACTIVE_TEXTURE, &mut unit);
            glActiveTexture(GL_TEXTURE2);
            if self.fog_func_tex == 0 {
                self.fog_func_tex = super::internal_texture_name();
            }
            glBindTexture(GL_TEXTURE_1D, self.fog_func_tex);
            glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA as i32, STEPS as i32, 0, GL_RGBA, GL_UNSIGNED_BYTE, table.as_ptr() as *const c_void);
            glActiveTexture(unit.max(GL_TEXTURE0 as i32) as u32);
        }
        self.fog_func_range = (x0, x1);
        self.fog_func_ready = true;
    }

    pub fn bound_texture(&self) -> u32 {
        self.bound_texture
    }

    pub fn bind_texture(&mut self, target: u32, texture: u32) {
        if target == GL_TEXTURE_2D {
            self.bound_texture = texture;
        }
    }

    /// A texture parameter the Mac has not got: scale or bias after the
    /// filter. True when it was ours to take.
    pub fn set_tex_parameter(&mut self, pname: u32, values: &[f32]) -> bool {
        // SGIS_texture_filter4 arrives as a filter mode, and the Mac would
        // refuse the enumerant. The shader does the taps itself, so the host
        // is left sampling single texels.
        if (pname == GL_TEXTURE_MIN_FILTER || pname == GL_TEXTURE_MAG_FILTER)
            && values.first().map(|v| *v as u32) == Some(GL_FILTER4_SGIS)
        {
            self.filter4.insert(self.bound_texture);
            unsafe { glTexParameteri(GL_TEXTURE_2D, pname, GL_NEAREST) };
            return true;
        }
        if self.set_clipmap(pname, values) {
            return true;
        }
        // SGIX_clipmap's filter, with no clipmap behind it (see set_clipmap):
        // the Mac would refuse the enumerant, and ordinary mipmapping is what
        // a clipmap reduces to when nothing is clipped.
        if pname == GL_TEXTURE_MIN_FILTER
            && values.first().map(|v| *v as u32) == Some(GL_LINEAR_CLIPMAP_LINEAR_SGIX)
        {
            unsafe { glTexParameteri(GL_TEXTURE_2D, pname, GL_LINEAR_MIPMAP_LINEAR) };
            return true;
        }
        // SGIS_detail_texture's parameters and its three filters.
        if self.set_select(pname, values.first().copied().unwrap_or(0.0) as u32) {
            return true;
        }
        if pname == GL_DETAIL_TEXTURE_LEVEL_SGIS {
            self.detail_level = values.first().copied().unwrap_or(0.0);
            return true;
        }
        if pname == GL_DETAIL_TEXTURE_MODE_SGIS {
            self.detail_add = values.first().map(|v| *v as u32) == Some(GL_ADD_ENUM);
            return true;
        }
        if pname == GL_TEXTURE_MAG_FILTER {
            match values.first().map(|v| *v as u32) {
                Some(f @ (GL_LINEAR_DETAIL_SGIS | GL_LINEAR_DETAIL_ALPHA_SGIS | GL_LINEAR_DETAIL_COLOR_SGIS)) => {
                    self.detail.insert(self.bound_texture, f);
                    unsafe { glTexParameteri(GL_TEXTURE_2D, pname, GL_LINEAR) };
                    return true;
                }
                _ => {
                    self.detail.remove(&self.bound_texture);
                }
            }
        }
        // SGIS_sharpen_texture's three magnification filters, likewise.
        if pname == GL_TEXTURE_MAG_FILTER {
            match values.first().map(|v| *v as u32) {
                Some(f @ (GL_LINEAR_SHARPEN_SGIS | GL_LINEAR_SHARPEN_ALPHA_SGIS | GL_LINEAR_SHARPEN_COLOR_SGIS)) => {
                    self.sharpen.insert(self.bound_texture, f);
                    unsafe { glTexParameteri(GL_TEXTURE_2D, pname, GL_LINEAR) };
                    return true;
                }
                _ => {
                    self.sharpen.remove(&self.bound_texture);
                }
            }
        }
        if (pname == GL_TEXTURE_MIN_FILTER || pname == GL_TEXTURE_MAG_FILTER)
            && self.filter4.contains(&self.bound_texture)
        {
            // Back to an ordinary filter: the program has taken it off again.
            self.filter4.remove(&self.bound_texture);
            return false;
        }
        let entry = self.scale_bias.entry(self.bound_texture).or_insert(([1.0; 4], [0.0; 4]));
        let slot = match pname {
            GL_POST_TEXTURE_FILTER_SCALE_SGIX => &mut entry.0,
            GL_POST_TEXTURE_FILTER_BIAS_SGIX => &mut entry.1,
            _ => return false,
        };
        for (d, s) in slot.iter_mut().zip(values.iter()) {
            *d = *s;
        }
        true
    }

    /// Which detail filter the bound texture wants: 0 none, 1 every
    /// component, 2 alpha alone, 3 colour alone. Needs both the image and the
    /// function, since either alone says nothing.
    fn detail_kind(&self) -> u32 {
        if !self.detail_ready || !self.detail_func_ready {
            return 0;
        }
        match self.detail.get(&self.bound_texture) {
            Some(&GL_LINEAR_DETAIL_SGIS) => 1,
            Some(&GL_LINEAR_DETAIL_ALPHA_SGIS) => 2,
            Some(&GL_LINEAR_DETAIL_COLOR_SGIS) => 3,
            _ => 0,
        }
    }

    /// Which sharpen filter the bound texture wants: 0 none, 1 every
    /// component, 2 alpha alone, 3 colour alone.
    fn sharpen_kind(&self) -> u32 {
        if !self.sharpen_ready {
            return 0;
        }
        match self.sharpen.get(&self.bound_texture) {
            Some(&GL_LINEAR_SHARPEN_SGIS) => 1,
            Some(&GL_LINEAR_SHARPEN_ALPHA_SGIS) => 2,
            Some(&GL_LINEAR_SHARPEN_COLOR_SGIS) => 3,
            _ => 0,
        }
    }

    /// The scale and bias in force for the texture now bound, if they are not
    /// the identity.
    fn active_scale_bias(&self) -> Option<([f32; 4], [f32; 4])> {
        let (scale, bias) = *self.scale_bias.get(&self.bound_texture)?;
        let plain = scale == [1.0; 4] && bias == [0.0; 4];
        (!plain).then_some((scale, bias))
    }

    /// Whether the fragment stage is ours at all: only then is a program bound.
    fn wanted(&self) -> bool {
        let texture_feature =
            self.texture_2d && ((self.color_table && self.table_ready) || self.active_scale_bias().is_some());
        let fog_feature = self.fog && (self.fog_offset || (self.fog_mode == GL_FOG_FUNC_SGIS && self.fog_func_ready));
        let filter4 = self.texture_2d && self.filter4_ready && self.filter4.contains(&self.bound_texture);
        let sharpen = self.texture_2d && self.sharpen_ready && self.sharpen.contains_key(&self.bound_texture);
        let detail = self.texture_2d && self.detail_kind() != 0;
        let select = self.texture_2d && self.select_kind() != 0;
        let tex4d = self.tex4d_on && self.tex4d_ready;
        let clipmap = self.texture_2d && self.clipmap_on();
        texture_feature || fog_feature || filter4 || sharpen || detail || select || tex4d || clipmap
            || (self.ref_plane_on && self.ref_plane_set)
    }

    /// glTexFilterFuncSGIS: the filter function f, sampled over [0, 2].
    ///
    /// The specification says weights[i] is f((2*i)/(n-1)), so the array is
    /// already f at evenly spaced distances and goes into the texture as it
    /// stands -- the shader reads it back with distance/2 as the coordinate.
    pub fn set_filter_func(&mut self, filter: u32, weights: &[f32]) -> bool {
        if filter != GL_FILTER4_SGIS || weights.len() < 2 {
            return false;
        }
        unsafe {
            if self.filter4_tex == 0 {
                self.filter4_tex = super::internal_texture_name();
            }
            let mut unit = 0i32;
            glGetIntegerv(GL_ACTIVE_TEXTURE, &mut unit);
            glActiveTexture(GL_TEXTURE3);
            glBindTexture(GL_TEXTURE_1D, self.filter4_tex);
            glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexImage1D(
                GL_TEXTURE_1D,
                0,
                GL_LUMINANCE32F_ARB,
                weights.len() as i32,
                0,
                GL_LUMINANCE,
                GL_FLOAT,
                weights.as_ptr() as *const c_void,
            );
            glActiveTexture(unit.max(GL_TEXTURE0 as i32) as u32);
        }
        self.filter4_n = weights.len() as f32;
        self.filter4_ready = true;
        true
    }

    /// glSharpenTexFuncSGIS's points: (LOD, F) pairs, sorted by LOD, with a
    /// straight line between neighbours -- which the specification allows
    /// ("this curve may be linear between adjacent points ... but it will pass
    /// exactly through the points").
    pub fn set_sharpen_func(&mut self, points: &[f32]) {
        let n = points.len() / 2;
        if n < 2 {
            self.sharpen_ready = false;
            return;
        }
        let (x0, x1) = (points[0], points[(n - 1) * 2]);
        if !(x1 > x0) {
            self.sharpen_ready = false;
            return;
        }
        const STEPS: usize = 256;
        let mut table = vec![0f32; STEPS];
        for (i, out) in table.iter_mut().enumerate() {
            let x = x0 + (x1 - x0) * i as f32 / (STEPS - 1) as f32;
            *out = piecewise(points, n, x);
        }
        unsafe {
            if self.sharpen_tex == 0 {
                self.sharpen_tex = super::internal_texture_name();
            }
            let mut unit = 0i32;
            glGetIntegerv(GL_ACTIVE_TEXTURE, &mut unit);
            glActiveTexture(GL_TEXTURE4);
            glBindTexture(GL_TEXTURE_1D, self.sharpen_tex);
            glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexImage1D(GL_TEXTURE_1D, 0, GL_LUMINANCE32F_ARB, STEPS as i32, 0,
                GL_LUMINANCE, GL_FLOAT, table.as_ptr() as *const c_void);
            glActiveTexture(unit.max(GL_TEXTURE0 as i32) as u32);
        }
        self.sharpen_range = (x0, x1);
        self.sharpen_ready = true;
    }

    /// SGIX_clipmap's parameters.
    ///
    /// **This keeps the state; it is not a clipmap.** A real one holds the
    /// finer levels as equal-sized windows into a much larger virtual texture
    /// and slides them as the centre moves -- and equal-sized levels are
    /// exactly what an OpenGL mipmap chain cannot hold, since every level is
    /// half the one before. Emulating it properly needs the clipped levels
    /// packed into one image as a strip, with the level chosen, wrapped
    /// toroidally and blended in the shader.
    ///
    /// So the parameters are accepted and answered, and
    /// LINEAR_CLIPMAP_LINEAR falls back to LINEAR_MIPMAP_LINEAR: a program
    /// that uses a clipmap no larger than its resident levels draws
    /// correctly, and one that relies on a moving centre does not.
    /// `GL_SGIX_clipmap` is deliberately **not advertised** for that reason.
    pub fn set_clipmap(&mut self, pname: u32, values: &[f32]) -> bool {
        // centre (2), offset (2), the (D, N+1, V+1) triple (3), frame, lod bias.
        let (slot, n) = match pname {
            GL_TEXTURE_CLIPMAP_CENTER_SGIX => (0, 2),
            GL_TEXTURE_CLIPMAP_OFFSET_SGIX => (2, 2),
            GL_TEXTURE_CLIPMAP_VIRTUAL_DEPTH_SGIX => (4, 3),
            GL_TEXTURE_CLIPMAP_FRAME_SGIX => (7, 1),
            GL_TEXTURE_CLIPMAP_LOD_OFFSET_SGIX => (8, 1),
            _ => return false,
        };
        let e = self.clipmap.entry(self.bound_texture).or_insert([0.0; 10]);
        for (i, v) in values.iter().take(n).enumerate() {
            e[slot + i] = *v;
        }
        true
    }

    /// Is the bound texture a clipmap with something in it?
    fn clipmap_on(&self) -> bool {
        self.clipmap_ready && self.clipmap.contains_key(&self.bound_texture)
    }

    /// One level of a clipmap, into its band of the strip.
    ///
    /// The clip size is taken from the first level that arrives, which is the
    /// finest and therefore the widest clipped one; the levels below it are
    /// the same size while they are clipped and smaller once the virtual level
    /// fits, which `exp2(V - P)` says without having to remember each.
    pub fn clipmap_image(
        &mut self,
        level: i32,
        w: i32,
        h: i32,
        format: u32,
        ty: u32,
        pixels: *const c_void,
    ) -> bool {
        let Some(&params) = self.clipmap.get(&self.bound_texture) else { return false };
        if w <= 0 || h <= 0 {
            return false;
        }
        // (D, N+1, V+1): the number of resident levels, and the virtual
        // pyramid's depth, from which the finest virtual size is 2^V.
        let levels = (params[5] as i32).max(1);
        self.clipmap_v = (params[6] - 1.0).max(0.0);
        if self.clipmap_c == 0 {
            self.clipmap_c = w.max(h);
            self.clipmap_levels = levels;
            unsafe {
                if self.clipmap_tex == 0 {
                    self.clipmap_tex = super::internal_texture_name();
                }
                let mut unit = 0i32;
                glGetIntegerv(GL_ACTIVE_TEXTURE, &mut unit);
                glActiveTexture(GL_TEXTURE8 + 1);
                glBindTexture(GL_TEXTURE_2D, self.clipmap_tex);
                // Sampled a texel at a time: the strip is filtered by hand,
                // because a band boundary and a toroidal seam are both places
                // where the hardware's filter would blend the wrong texels.
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, 0x2803, GL_CLAMP_TO_EDGE);
                glTexImage2D(GL_TEXTURE_2D, 0, 0x1908, self.clipmap_c,
                    self.clipmap_c * self.clipmap_levels, 0, 0x1908, GL_UNSIGNED_BYTE,
                    std::ptr::null());
                glActiveTexture(unit.max(GL_TEXTURE0 as i32) as u32);
            }
        }
        if level < 0 || level >= self.clipmap_levels {
            return true;   // beyond what is resident: the program said so itself
        }
        unsafe {
            let mut unit = 0i32;
            glGetIntegerv(GL_ACTIVE_TEXTURE, &mut unit);
            glActiveTexture(GL_TEXTURE8 + 1);
            glBindTexture(GL_TEXTURE_2D, self.clipmap_tex);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, level * self.clipmap_c, w, h, format, ty, pixels);
            glActiveTexture(unit.max(GL_TEXTURE0 as i32) as u32);
        }
        self.clipmap_ready = true;
        true
    }

    /// What `glGetTexParameter` should answer for a clipmap parameter.
    pub fn clipmap_get(&self, pname: u32, out: &mut [f32]) -> bool {
        let Some(e) = self.clipmap.get(&self.bound_texture) else { return false };
        let (slot, n) = match pname {
            GL_TEXTURE_CLIPMAP_CENTER_SGIX => (0, 2),
            GL_TEXTURE_CLIPMAP_OFFSET_SGIX => (2, 2),
            GL_TEXTURE_CLIPMAP_VIRTUAL_DEPTH_SGIX => (4, 3),
            GL_TEXTURE_CLIPMAP_DEPTH_SGIX => (5, 1),
            GL_TEXTURE_CLIPMAP_FRAME_SGIX => (7, 1),
            GL_TEXTURE_CLIPMAP_LOD_OFFSET_SGIX => (8, 1),
            _ => return false,
        };
        for i in 0..n.min(out.len()) {
            out[i] = e[slot + i];
        }
        true
    }

    /// glPixelTexGenSGIX: which components of the fragment's colour come from
    /// the pixel and which from the current raster position.
    pub fn set_pixel_tex_gen(&mut self, mode: u32) {
        self.pixel_tex_mode = mode;
    }

    /// glDrawPixels under SGIX_pixel_texture.
    ///
    /// "The pixel group's red becomes the fragment S texture coordinate", and
    /// so on for green, blue and alpha. There is no such stage on the Mac, so
    /// the transfer is done as a drawing instead: the image is staged in a
    /// texture, a screen-aligned quad is drawn over the destination rectangle,
    /// and a shader reads the staged pixel, uses it as a coordinate into the
    /// texture the program has bound, and composes the colour the mode asks
    /// for. True when this took the call.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_pixels_textured(&mut self, w: i32, h: i32, format: u32, ty: u32, pixels: *const c_void) -> bool {
        if !self.pixel_tex_on || w <= 0 || h <= 0 {
            return false;
        }
        let program = self.pixel_tex_program();
        if program == 0 {
            return false;
        }
        let (mut raster, mut colour, mut vp) = ([0f32; 4], [0f32; 4], [0i32; 4]);
        unsafe {
            glGetFloatv(GL_CURRENT_RASTER_POSITION, raster.as_mut_ptr());
            glGetFloatv(GL_CURRENT_RASTER_COLOR, colour.as_mut_ptr());
            glGetIntegerv(GL_VIEWPORT_ENUM, vp.as_mut_ptr());

            // The image goes into a texture of ours, sampled a texel at a time
            // so the pixels arrive at the shader exactly as the program wrote
            // them rather than filtered into each other.
            if self.pixel_tex_img == 0 {
                self.pixel_tex_img = super::internal_texture_name();
            }
            let mut unit = 0i32;
            glGetIntegerv(GL_ACTIVE_TEXTURE, &mut unit);
            glActiveTexture(GL_TEXTURE8);
            glBindTexture(GL_TEXTURE_2D, self.pixel_tex_img);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, 0x2803, GL_CLAMP_TO_EDGE);
            glTexImage2D(GL_TEXTURE_2D, 0, 0x1908, w, h, 0, format, ty, pixels);
            glActiveTexture(unit.max(GL_TEXTURE0 as i32) as u32);

            glUseProgram(program);
            for (name, v) in [("hgl_px_raster", colour)] {
                let n = CString::new(name).unwrap();
                let loc = glGetUniformLocation(program, n.as_ptr());
                if loc >= 0 {
                    glUniform4f(loc, v[0], v[1], v[2], v[3]);
                }
            }
            for (name, u) in [("hgl_px_image", 8i32), ("hgl_px_tex", 0)] {
                let n = CString::new(name).unwrap();
                let loc = glGetUniformLocation(program, n.as_ptr());
                if loc >= 0 {
                    glUniform1i(loc, u);
                }
            }

            // A quad over exactly the pixels glDrawPixels would have written,
            // in window coordinates, with the matrices put back afterwards.
            glMatrixMode(0x1701 /* PROJECTION */);
            glPushMatrix();
            glLoadIdentity();
            glOrtho(vp[0] as f64, (vp[0] + vp[2]) as f64, vp[1] as f64, (vp[1] + vp[3]) as f64, -1.0, 1.0);
            glMatrixMode(0x1700 /* MODELVIEW */);
            glPushMatrix();
            glLoadIdentity();
            let (x0, y0) = (raster[0], raster[1]);
            let (x1, y1) = (x0 + w as f32, y0 + h as f32);
            glBegin(7 /* GL_QUADS */);
            glTexCoord2f(0.0, 0.0);
            glVertex2f(x0, y0);
            glTexCoord2f(1.0, 0.0);
            glVertex2f(x1, y0);
            glTexCoord2f(1.0, 1.0);
            glVertex2f(x1, y1);
            glTexCoord2f(0.0, 1.0);
            glVertex2f(x0, y1);
            glEnd();
            glPopMatrix();
            glMatrixMode(0x1701);
            glPopMatrix();
            glMatrixMode(0x1700);
            glUseProgram(self.current);
        }
        true
    }

    fn pixel_tex_program(&mut self) -> u32 {
        let key = self.pixel_tex_mode;
        if let Some(&p) = self.pixel_tex_prog.get(&key) {
            return p;
        }
        // NONE takes the whole colour from the pixel, RGBA the whole of it
        // from the raster position, and the two in between split it.
        let colour = match key {
            0x1906 => "vec4(px.rgb, hgl_px_raster.a);",  // ALPHA
            0x1907 => "vec4(hgl_px_raster.rgb, px.a);",  // RGB
            0x1908 => "hgl_px_raster;",                  // RGBA
            _ => "px;",                                   // NONE
        };
        let src = format!(
            r#"#version 120
uniform sampler2D hgl_px_image;
uniform sampler2D hgl_px_tex;
uniform vec4 hgl_px_raster;
void main()
{{
    vec4 px = texture2D(hgl_px_image, gl_TexCoord[0].st);
    vec4 c = {colour}
    gl_FragColor = c * texture2D(hgl_px_tex, px.st);
}}
"#
        );
        let p = self.compile(&src);
        self.pixel_tex_prog.insert(key, p);
        p
    }

    /// glSpriteParameter*SGIX.
    pub fn set_sprite(&mut self, pname: u32, values: &[f32]) -> bool {
        match pname {
            GL_SPRITE_MODE_SGIX => self.sprite_mode = values.first().copied().unwrap_or(0.0) as u32,
            GL_SPRITE_AXIS_SGIX => {
                for (d, v) in self.sprite_axis.iter_mut().zip(values) {
                    *d = *v;
                }
            }
            GL_SPRITE_TRANSLATION_SGIX => {
                for (d, v) in self.sprite_translation.iter_mut().zip(values) {
                    *d = *v;
                }
            }
            _ => return false,
        }
        true
    }

    /// Put the sprite transformation in front of the modelview for the
    /// primitive about to be drawn, and take it away again afterwards.
    ///
    /// The specification's compound matrix is MM = M * T * A * R, so the
    /// sprite part multiplies *after* the modelview -- which is what
    /// glMultMatrix does to the current matrix, and means the fixed pipeline
    /// can do this with no vertex shader at all.
    pub fn sprite_begin(&mut self) {
        if !self.sprite_on || self.sprite_pushed {
            return;
        }
        let mut mv = [0f64; 16];
        unsafe { glGetDoublev(GL_MODELVIEW_MATRIX, mv.as_mut_ptr()) };
        let Some(inv) = invert4(&mv) else { return };
        // The eye is the origin of eye coordinates, so its object-space
        // position is the inverse modelview's fourth column.
        let w = inv[15] as f32;
        let mut eye = [inv[12] as f32, inv[13] as f32, inv[14] as f32];
        if w.abs() > 1e-9 {
            eye = [eye[0] / w, eye[1] / w, eye[2] / w];
        }
        // An eye-aligned sprite's axis is given in eye coordinates, so it
        // comes back to object coordinates as a direction.
        let axis = if self.sprite_mode == GL_SPRITE_EYE_ALIGNED_SGIX {
            let a = self.sprite_axis;
            [
                (inv[0] as f32) * a[0] + (inv[4] as f32) * a[1] + (inv[8] as f32) * a[2],
                (inv[1] as f32) * a[0] + (inv[5] as f32) * a[1] + (inv[9] as f32) * a[2],
                (inv[2] as f32) * a[0] + (inv[6] as f32) * a[1] + (inv[10] as f32) * a[2],
            ]
        } else {
            self.sprite_axis
        };
        let r = if self.sprite_mode == GL_SPRITE_AXIAL_SGIX {
            axial(axis, eye)
        } else {
            face(axis, eye)
        };
        unsafe {
            glPushMatrix();
            glTranslatef(
                self.sprite_translation[0],
                self.sprite_translation[1],
                self.sprite_translation[2],
            );
            glMultMatrixf(r.as_ptr());
        }
        self.sprite_pushed = true;
    }

    pub fn sprite_end(&mut self) {
        if self.sprite_pushed {
            unsafe { glPopMatrix() };
            self.sprite_pushed = false;
        }
    }

    /// glTexImage4DSGIS. There is no fourth texture axis on the Mac, so the
    /// image is stored as a 3D texture `depth * size4d` deep -- the slabs laid
    /// end to end, which is the order the guest's own pixels already arrive in
    /// -- and the shader interpolates between two slabs to filter the fourth
    /// coordinate. Mipmapping is not supported by the extension either, so
    /// only level zero exists and nothing is lost by that.
    #[allow(clippy::too_many_arguments)]
    pub fn set_texture_4d(
        &mut self,
        internal: i32,
        w: i32,
        h: i32,
        d: i32,
        size4d: i32,
        border: i32,
        format: u32,
        ty: u32,
        pixels: *const c_void,
    ) {
        if w <= 0 || h <= 0 || d <= 0 || size4d <= 0 {
            return;
        }
        unsafe {
            if self.tex4d == 0 {
                self.tex4d = super::internal_texture_name();
            }
            let mut unit = 0i32;
            glGetIntegerv(GL_ACTIVE_TEXTURE, &mut unit);
            glActiveTexture(GL_TEXTURE7);
            glBindTexture(GL_TEXTURE_3D, self.tex4d);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_3D, 0x2803 /* WRAP_T */, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_3D, 0x8072 /* WRAP_R */, GL_CLAMP_TO_EDGE);
            glTexImage3D(GL_TEXTURE_3D, 0, internal, w, h, d * size4d, border, format, ty, pixels);
            glActiveTexture(unit.max(GL_TEXTURE0 as i32) as u32);
        }
        self.tex4d_depth = d as f32;
        self.tex4d_size = size4d as f32;
        self.tex4d_ready = true;
    }

    /// SGIS_texture_select's internal formats, and how a group is read back.
    ///
    /// The formats hold several textures' components side by side in one
    /// image; the Mac has no such enumerant, so the image is stored as plain
    /// RGBA -- which is the layout the components are already in -- and the
    /// shader presents the selected group as if it were the whole texture.
    ///
    ///   DUAL_{ALPHA,LUMINANCE,INTENSITY}: group 0 is red, group 1 is alpha
    ///   DUAL_LUMINANCE_ALPHA:             group 0 is (red, green), 1 is (blue, alpha)
    ///   QUAD_*:                           group i is component i
    ///
    /// Returns the internal format to give the host, or None when this is not
    /// one of these formats.
    pub fn select_format(&mut self, texture: u32, internal: i32) -> Option<i32> {
        let f = internal as u32;
        let grouped = (GL_DUAL_ALPHA4_SGIS..=GL_QUAD_INTENSITY8_SGIS).contains(&f);
        if !grouped {
            self.select_fmt.remove(&texture);
            return None;
        }
        self.select_fmt.insert(texture, f);
        Some(0x1908) // GL_RGBA: every component kept, whatever the group wants
    }

    /// The group number a DUAL_/QUAD_ texture parameter selects.
    pub fn set_select(&mut self, pname: u32, value: u32) -> bool {
        if pname != GL_DUAL_TEXTURE_SELECT_SGIS && pname != GL_QUAD_TEXTURE_SELECT_SGIS {
            return false;
        }
        self.select_group.insert(self.bound_texture, value);
        true
    }

    /// How the bound texture's selected group is read: 0 none, otherwise
    /// 1 + the swizzle case. Kept small because it goes in the shader key.
    fn select_kind(&self) -> u32 {
        let Some(&f) = self.select_fmt.get(&self.bound_texture) else { return 0 };
        let g = *self.select_group.get(&self.bound_texture).unwrap_or(&0);
        let quad = f >= GL_QUAD_ALPHA4_SGIS;
        let la = f == GL_DUAL_LUMINANCE_ALPHA8_SGIS || f == GL_DUAL_LUMINANCE_ALPHA8_SGIS - 1;
        if quad {
            1 + (g & 3)
        } else if la {
            5 + (g & 1)
        } else {
            7 + (g & 1)
        }
    }

    /// glDetailTexFuncSGIS's points: the same (LOD, F) shape as sharpen's.
    pub fn set_detail_func(&mut self, points: &[f32]) {
        let n = points.len() / 2;
        if n < 2 {
            self.detail_func_ready = false;
            return;
        }
        let (x0, x1) = (points[0], points[(n - 1) * 2]);
        if !(x1 > x0) {
            self.detail_func_ready = false;
            return;
        }
        const STEPS: usize = 256;
        let mut table = vec![0f32; STEPS];
        for (i, out) in table.iter_mut().enumerate() {
            *out = piecewise(points, n, x0 + (x1 - x0) * i as f32 / (STEPS - 1) as f32);
        }
        unsafe {
            if self.detail_func_tex == 0 {
                self.detail_func_tex = super::internal_texture_name();
            }
            let mut unit = 0i32;
            glGetIntegerv(GL_ACTIVE_TEXTURE, &mut unit);
            glActiveTexture(GL_TEXTURE5);
            glBindTexture(GL_TEXTURE_1D, self.detail_func_tex);
            glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexImage1D(GL_TEXTURE_1D, 0, GL_LUMINANCE32F_ARB, STEPS as i32, 0,
                GL_LUMINANCE, GL_FLOAT, table.as_ptr() as *const c_void);
            glActiveTexture(unit.max(GL_TEXTURE0 as i32) as u32);
        }
        self.detail_range = (x0, x1);
        self.detail_func_ready = true;
    }

    /// The detail image itself. The program names it with a target of its own,
    /// DETAIL_TEXTURE_2D_SGIS, which the Mac would refuse -- so it goes into a
    /// texture object of ours on a unit the shader reads, and the binding the
    /// program had is put back so nothing else notices.
    pub fn set_detail_image(
        &mut self,
        level: i32,
        internal: i32,
        w: i32,
        h: i32,
        border: i32,
        format: u32,
        ty: u32,
        pixels: *const c_void,
    ) {
        unsafe {
            if self.detail_tex == 0 {
                self.detail_tex = super::internal_texture_name();
            }
            let mut unit = 0i32;
            glGetIntegerv(GL_ACTIVE_TEXTURE, &mut unit);
            glActiveTexture(GL_TEXTURE5 + 1);
            glBindTexture(GL_TEXTURE_2D, self.detail_tex);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexImage2D(GL_TEXTURE_2D, level, internal, w, h, border, format, ty, pixels);
            glActiveTexture(unit.max(GL_TEXTURE0 as i32) as u32);
        }
        self.detail_ready = true;
    }

    /// glReferencePlaneSGIX: the plane the fragment's depth comes from.
    ///
    /// The equation arrives in object coordinates and the specification says
    /// it is "transformed by the transpose-adjoint of a matrix that is the
    /// complete object-coordinate to clip-coordinate transformation" -- the
    /// same treatment glClipPlane gives its plane, and for the same reason:
    /// the plane is placed in the world when it is given, not when it is used.
    /// The adjoint's determinant scales the whole equation and so cancels out
    /// of `-(d + a x + b y) / c`, which leaves the inverse-transpose.
    pub fn set_reference_plane(&mut self, equation: &[f64]) {
        if equation.len() < 4 {
            return;
        }
        let (mut mv, mut pr) = ([0f64; 16], [0f64; 16]);
        unsafe {
            glGetDoublev(GL_MODELVIEW_MATRIX, mv.as_mut_ptr());
            glGetDoublev(GL_PROJECTION_MATRIX, pr.as_mut_ptr());
        }
        // Column-major, as GL hands them over: m[c * 4 + r].
        let mut mvp = [0f64; 16];
        for c in 0..4 {
            for r in 0..4 {
                mvp[c * 4 + r] = (0..4).map(|k| pr[k * 4 + r] * mv[c * 4 + k]).sum();
            }
        }
        let Some(inv) = invert4(&mvp) else {
            // A singular matrix places no plane; leave the last one alone.
            return;
        };
        // (M^-1)^T * p, with the transpose folded into the indexing.
        let mut out = [0f32; 4];
        for i in 0..4 {
            let v: f64 = (0..4).map(|k| inv[i * 4 + k] * equation[k]).sum();
            out[i] = v as f32;
        }
        self.ref_plane = out;
        self.ref_plane_set = true;
    }

    /// SGI_texture_color_table's table: kept as a 1D texture the shader reads,
    /// one entry per level of each component.
    pub fn set_color_table(&mut self, width: i32, rgba: &[u8]) {
        if width <= 0 || rgba.len() < width as usize * 4 {
            self.table_ready = false;
            return;
        }
        unsafe {
            let mut unit = 0i32;
            glGetIntegerv(GL_ACTIVE_TEXTURE, &mut unit);
            glActiveTexture(GL_TEXTURE1);
            if self.table_tex == 0 {
                self.table_tex = super::internal_texture_name();
            }
            glBindTexture(GL_TEXTURE_1D, self.table_tex);
            glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA as i32, width, 0, GL_RGBA, GL_UNSIGNED_BYTE, rgba.as_ptr() as *const c_void);
            glActiveTexture(unit.max(GL_TEXTURE0 as i32) as u32);
        }
        self.table_ready = true;
    }

    /// Bind the fragment stage for the state as it now is, or none of ours.
    pub fn update(&mut self) {
        if !self.wanted() {
            if self.current != 0 {
                unsafe { glUseProgram(0) };
                self.current = 0;
            }
            return;
        }
        let table = self.color_table && self.table_ready;
        let scaled = self.active_scale_bias().is_some();
        let fog_kind = match self.fog_mode {
            GL_EXP_FOG => 1,
            GL_EXP2_FOG => 2,
            GL_FOG_FUNC_SGIS if self.fog_func_ready => 3,
            _ => 0, // linear
        };
        let key = (self.env_mode as u64 & 0xffff)
            | (self.fog as u64) << 16
            | (self.texture_2d as u64) << 17
            | (table as u64) << 18
            | (scaled as u64) << 19
            | (self.fog_offset as u64) << 20
            | (fog_kind as u64) << 21
            | ((self.ref_plane_on && self.ref_plane_set) as u64) << 23
            | ((self.filter4_ready && self.filter4.contains(&self.bound_texture)) as u64) << 24
            | (self.sharpen_kind() as u64) << 25
            | (self.detail_kind() as u64) << 27
            | (self.detail_add as u64) << 29
            | (self.select_kind() as u64) << 32
            | ((self.tex4d_on && self.tex4d_ready) as u64) << 36
            | (self.clipmap_on() as u64) << 37;
        let program = match self.programs.get(&key) {
            Some(&p) => p,
            None => {
                let p = self.build(key);
                self.programs.insert(key, p);
                p
            }
        };
        if program == 0 {
            return;
        }
        if self.current != program {
            unsafe { glUseProgram(program) };
            self.current = program;
        }
        unsafe {
            let mut unit = 0i32;
            glGetIntegerv(GL_ACTIVE_TEXTURE, &mut unit);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_1D, self.table_tex);
            glActiveTexture(unit.max(GL_TEXTURE0 as i32) as u32);
            glActiveTexture(GL_TEXTURE2);
            glBindTexture(GL_TEXTURE_1D, self.fog_func_tex);
            glActiveTexture(unit.max(GL_TEXTURE0 as i32) as u32);
            for (name, v) in [
                ("hgl_fog_x0", self.fog_func_range.0),
                ("hgl_fog_x1", self.fog_func_range.1),
                ("hgl_fog_offset", self.fog_offset_value),
            ] {
                let n = CString::new(name).unwrap();
                let loc = glGetUniformLocation(program, n.as_ptr());
                if loc >= 0 {
                    glUniform1f(loc, v);
                }
            }
            if self.filter4_ready && self.filter4.contains(&self.bound_texture) {
                glActiveTexture(GL_TEXTURE3);
                glBindTexture(GL_TEXTURE_1D, self.filter4_tex);
                glActiveTexture(unit.max(GL_TEXTURE0 as i32) as u32);
                // The taps are a texel apart, so the shader needs the size in
                // texels. Asked of the host rather than remembered: the
                // program may have loaded the texture by any of a dozen calls.
                let (mut w, mut h) = (0i32, 0i32);
                glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &mut w);
                glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &mut h);
                let n = CString::new("hgl_texsize").unwrap();
                let loc = glGetUniformLocation(program, n.as_ptr());
                if loc >= 0 {
                    glUniform2f(loc, w.max(1) as f32, h.max(1) as f32);
                }
                let n = CString::new("hgl_f4_n").unwrap();
                let loc = glGetUniformLocation(program, n.as_ptr());
                if loc >= 0 {
                    glUniform1f(loc, self.filter4_n);
                }
            }
            if self.clipmap_on() {
                glActiveTexture(GL_TEXTURE8 + 1);
                glBindTexture(GL_TEXTURE_2D, self.clipmap_tex);
                glActiveTexture(unit.max(GL_TEXTURE0 as i32) as u32);
                let p = self.clipmap.get(&self.bound_texture).copied().unwrap_or([0.0; 10]);
                for (name, v) in [
                    ("hgl_cm_c", self.clipmap_c as f32),
                    ("hgl_cm_levels", self.clipmap_levels as f32),
                    ("hgl_cm_v", self.clipmap_v),
                ] {
                    let n = CString::new(name).unwrap();
                    let loc = glGetUniformLocation(program, n.as_ptr());
                    if loc >= 0 {
                        glUniform1f(loc, v);
                    }
                }
                for (name, a, b) in [
                    ("hgl_cm_centre", p[0], p[1]),
                    ("hgl_cm_offset", p[2], p[3]),
                ] {
                    let n = CString::new(name).unwrap();
                    let loc = glGetUniformLocation(program, n.as_ptr());
                    if loc >= 0 {
                        glUniform2f(loc, a, b);
                    }
                }
            }
            if self.tex4d_on && self.tex4d_ready {
                glActiveTexture(GL_TEXTURE7);
                glBindTexture(GL_TEXTURE_3D, self.tex4d);
                glActiveTexture(unit.max(GL_TEXTURE0 as i32) as u32);
                let n = CString::new("hgl_t4_size").unwrap();
                let loc = glGetUniformLocation(program, n.as_ptr());
                if loc >= 0 {
                    glUniform1f(loc, self.tex4d_size);
                }
            }
            if self.detail_kind() != 0 {
                glActiveTexture(GL_TEXTURE5);
                glBindTexture(GL_TEXTURE_1D, self.detail_func_tex);
                glActiveTexture(GL_TEXTURE5 + 1);
                glBindTexture(GL_TEXTURE_2D, self.detail_tex);
                glActiveTexture(unit.max(GL_TEXTURE0 as i32) as u32);
                let (mut w, mut h) = (0i32, 0i32);
                glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &mut w);
                glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &mut h);
                let n = CString::new("hgl_texsize").unwrap();
                let loc = glGetUniformLocation(program, n.as_ptr());
                if loc >= 0 {
                    glUniform2f(loc, w.max(1) as f32, h.max(1) as f32);
                }
                for (name, v) in [
                    ("hgl_detail_x0", self.detail_range.0),
                    ("hgl_detail_x1", self.detail_range.1),
                    ("hgl_detail_scale", (2.0f32).powf(self.detail_level)),
                ] {
                    let n = CString::new(name).unwrap();
                    let loc = glGetUniformLocation(program, n.as_ptr());
                    if loc >= 0 {
                        glUniform1f(loc, v);
                    }
                }
            }
            if self.sharpen_kind() != 0 {
                glActiveTexture(GL_TEXTURE4);
                glBindTexture(GL_TEXTURE_1D, self.sharpen_tex);
                glActiveTexture(unit.max(GL_TEXTURE0 as i32) as u32);
                let (mut w, mut h) = (0i32, 0i32);
                glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &mut w);
                glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &mut h);
                let n = CString::new("hgl_texsize").unwrap();
                let loc = glGetUniformLocation(program, n.as_ptr());
                if loc >= 0 {
                    glUniform2f(loc, w.max(1) as f32, h.max(1) as f32);
                }
                for (name, v) in [
                    ("hgl_sharp_x0", self.sharpen_range.0),
                    ("hgl_sharp_x1", self.sharpen_range.1),
                ] {
                    let n = CString::new(name).unwrap();
                    let loc = glGetUniformLocation(program, n.as_ptr());
                    if loc >= 0 {
                        glUniform1f(loc, v);
                    }
                }
            }
            if self.ref_plane_on && self.ref_plane_set {
                let mut vp = [0i32; 4];
                glGetIntegerv(GL_VIEWPORT, vp.as_mut_ptr());
                for (name, v) in [
                    ("hgl_refplane", self.ref_plane),
                    ("hgl_viewport", [vp[0] as f32, vp[1] as f32, vp[2] as f32, vp[3] as f32]),
                ] {
                    let n = CString::new(name).unwrap();
                    let loc = glGetUniformLocation(program, n.as_ptr());
                    if loc >= 0 {
                        glUniform4f(loc, v[0], v[1], v[2], v[3]);
                    }
                }
            }
            if let Some((scale, bias)) = self.active_scale_bias() {
                for (name, v) in [("hgl_scale", scale), ("hgl_bias", bias)] {
                    let n = CString::new(name).unwrap();
                    let loc = glGetUniformLocation(program, n.as_ptr());
                    if loc >= 0 {
                        glUniform4f(loc, v[0], v[1], v[2], v[3]);
                    }
                }
            }
        }
    }

    fn build(&mut self, key: u64) -> u32 {
        let env = match (key & 0xffff) as u32 {
            GL_REPLACE => "c = texel;",
            GL_DECAL => "c = vec4(mix(c.rgb, texel.rgb, texel.a), c.a);",
            GL_BLEND => "c = vec4(mix(c.rgb, gl_TextureEnvColor[0].rgb, texel.rgb), c.a * texel.a);",
            GL_ADD => "c = vec4(c.rgb + texel.rgb, c.a * texel.a);",
            _ => "c = c * texel;", // GL_MODULATE, the default
        };
        let offset = if key & (1 << 20) != 0 {
            // SGIX_fog_offset: the fog begins that much nearer the eye.
            "z = max(z - hgl_fog_offset, 0.0);"
        } else {
            ""
        };
        let fog = if key & (1 << 16) != 0 {
            let factor = match (key >> 21) & 3 {
                1 => "float f = exp(-gl_Fog.density * z);",
                2 => "float f = exp(-gl_Fog.density * gl_Fog.density * z * z);",
                // SGIS_fog_function: the program's own points, by distance.
                3 => "float f = texture1D(hgl_fogfunc, clamp((z - hgl_fog_x0) / max(hgl_fog_x1 - hgl_fog_x0, 1e-6), 0.0, 1.0)).r;",
                _ => "float f = (gl_Fog.end - z) * gl_Fog.scale;",
            };
            // The distance to the eye. gl_FogFragCoord would hold it, but a
            // driver running our fragment shader does not fill it from the
            // fixed pipeline -- it reads 0, and every fragment comes out
            // unfogged. It is recovered instead from the window depth and the
            // projection the program set, which leaves that fixed pipeline
            // alone: z_ndc = 2*depth - 1, and eye z follows from the two rows
            // of the projection matrix that carry it.
            format!(
                "float ndc = 2.0 * gl_FragCoord.z - 1.0;\n    \
                 float ez = (gl_ProjectionMatrix[3][3] * ndc - gl_ProjectionMatrix[3][2])\n        \
                 / (gl_ProjectionMatrix[2][2] - gl_ProjectionMatrix[2][3] * ndc);\n    \
                 float z = abs(ez);\n    {offset}\n    {factor}\n    f = clamp(f, 0.0, 1.0);\n    \
                 c = vec4(mix(gl_Fog.color.rgb, c.rgb, f), c.a);"
            )
        } else {
            String::new()
        };
        let table = if key & (1 << 18) != 0 {
            // SGI_texture_color_table: each component through its own table.
            "texel = vec4(texture1D(hgl_table, texel.r).r,
                 texture1D(hgl_table, texel.g).g,
                 texture1D(hgl_table, texel.b).b,
                 texture1D(hgl_table, texel.a).a);"
        } else {
            ""
        };
        let scale_bias = if key & (1 << 19) != 0 {
            // SGIX_texture_scale_bias, after the filter and the table.
            "texel = texel * hgl_scale + hgl_bias;"
        } else {
            ""
        };
        // SGIS_texture_filter4: four taps in each direction, weighted by the
        // program's own function f. The specification's 1D formula is
        //   T = f(1+A) T[i0] + f(A) T[i1] + f(1-A) T[i2] + f(2-A) T[i3]
        // and the 2D case is its outer product over the sixteen texels. f is
        // held sampled over [0, 2], so a distance d reads it at d/2.
        // SGIS_sharpen_texture: the magnified texel extrapolated away from
        // what level 1 would have given --
        //   T' = (1 + F(LOD)) T0 - F(LOD) T1
        // -- which sharpens exactly the detail that magnification blurred. The
        // level-of-detail comes from the derivatives of the texture
        // coordinate, as the fixed pipeline's own minification would compute
        // it, and sampling a named level needs ARB_shader_texture_lod.
        let sharpen_kind = (key >> 25) & 3;
        let sharpen = if sharpen_kind != 0 {
            let apply = match sharpen_kind {
                2 => "texel.a = clamp(sharp.a, 0.0, 1.0);",
                3 => "texel.rgb = clamp(sharp.rgb, 0.0, 1.0);",
                _ => "texel = clamp(sharp, 0.0, 1.0);",
            };
            format!(
                "vec2 sd = gl_TexCoord[0].st * hgl_texsize;\n    \
                 float rho = max(length(dFdx(sd)), length(dFdy(sd)));\n    \
                 float lod = log2(max(rho, 1e-6));\n    \
                 float sf = texture1D(hgl_sharpen,\n        \
                 clamp((lod - hgl_sharp_x0) / max(hgl_sharp_x1 - hgl_sharp_x0, 1e-6), 0.0, 1.0)).r;\n    \
                 vec4 t1 = texture2D(hgl_tex, gl_TexCoord[0].st, 1.0 - lod);\n    \
                 vec4 sharp = (1.0 + sf) * texel - sf * t1;\n    {apply}"
            )
        } else {
            String::new()
        };
        // SGIS_detail_texture: a second, finer texture blended in by
        // level-of-detail --
        //   ADD:      T' = T + F(LOD) * (2 Td - 1)
        //   MODULATE: T' = T * (1 + F(LOD) * (2 Td - 1))
        // -- so the detail shows only where magnification has room for it.
        // The detail image is `level` levels finer, so its coordinates are
        // scaled by 2^level.
        // SGIS_texture_select: the selected group presented as if it were the
        // whole texture. The base format decides what a group *means* -- a
        // luminance group fills rgb and leaves alpha opaque, an alpha group
        // does the opposite, an intensity group fills both.
        let select = match (key >> 32) & 15 {
            1 => "texel = vec4(texel.rrr, 1.0);",         // QUAD group 0
            2 => "texel = vec4(texel.ggg, 1.0);",
            3 => "texel = vec4(texel.bbb, 1.0);",
            4 => "texel = vec4(texel.aaa, 1.0);",
            5 => "texel = vec4(texel.rrr, texel.g);",     // DUAL_LUMINANCE_ALPHA 0
            6 => "texel = vec4(texel.bbb, texel.a);",     // DUAL_LUMINANCE_ALPHA 1
            7 => "texel = vec4(texel.rrr, 1.0);",         // DUAL red group
            8 => "texel = vec4(texel.aaa, 1.0);",         // DUAL alpha group
            _ => "",
        };
        let detail_kind = (key >> 27) & 3;
        let detail = if detail_kind != 0 {
            let combine = if key & (1 << 29) != 0 {
                "vec4 dv = texel + df * (2.0 * td - 1.0);"
            } else {
                "vec4 dv = texel * (1.0 + df * (2.0 * td - 1.0));"
            };
            let apply = match detail_kind {
                2 => "texel.a = clamp(dv.a, 0.0, 1.0);",
                3 => "texel.rgb = clamp(dv.rgb, 0.0, 1.0);",
                _ => "texel = clamp(dv, 0.0, 1.0);",
            };
            format!(
                "vec2 dd = gl_TexCoord[0].st * hgl_texsize;\n    \
                 float drho = max(length(dFdx(dd)), length(dFdy(dd)));\n    \
                 float dlod = log2(max(drho, 1e-6));\n    \
                 float df = texture1D(hgl_detailfunc,\n        \
                 clamp((dlod - hgl_detail_x0) / max(hgl_detail_x1 - hgl_detail_x0, 1e-6), 0.0, 1.0)).r;\n    \
                 vec4 td = texture2D(hgl_detail, gl_TexCoord[0].st * hgl_detail_scale);\n    \
                 {combine}\n    {apply}"
            )
        } else {
            String::new()
        };
        // SGIS_texture4D: the fourth coordinate picks between two slabs of
        // the 3D texture the image is kept in, and is filtered by blending
        // them -- which together with the 3D sampler's own filtering of s, t
        // and r is the full four-dimensional linear filter.
        // SGIX_clipmap. The resident levels are bands of one image (see
        // `clipmap_image`), so the level is chosen here rather than by the
        // hardware, addressed toroidally, and filtered by hand.
        //
        // Two things make it a clipmap rather than a mipmap. The window at
        // level P is only C texels wide around the centre, so a coordinate
        // further than C/2 from it is not resident and the level of detail has
        // to rise until it is -- "the level of detail is increased to the
        // nearest image level that does include the required texels". And the
        // window wraps: the program slides it by uploading into the far edge,
        // so an address is (v + offset) modulo the window.
        let clipmap = if key & (1 << 37) != 0 {
            Some(
                "vec2 v0 = gl_TexCoord[0].st * exp2(hgl_cm_v);\n    \
                 vec2 dx = dFdx(v0);\n    \
                 vec2 dy = dFdy(v0);\n    \
                 float lod = log2(max(max(length(dx), length(dy)), 1e-6));\n    \
                 vec2 dc = abs(v0 - hgl_cm_centre);\n    \
                 float need = log2(max(2.0 * max(dc.x, dc.y) / hgl_cm_c, 1.0));\n    \
                 lod = clamp(max(lod, need), 0.0, hgl_cm_levels - 1.0);\n    \
                 float p0 = floor(lod);\n    \
                 float p1 = min(p0 + 1.0, hgl_cm_levels - 1.0);\n    \
                 vec4 texel = mix(hgl_cm_level(v0, p0), hgl_cm_level(v0, p1), lod - p0);",
            )
        } else {
            None
        };
        let fetch4d = if key & (1 << 36) != 0 {
            Some(
                "float q4 = gl_TexCoord[0].q * hgl_t4_size - 0.5;\n    \
                 float q4f = floor(q4);\n    \
                 float q4a = q4 - q4f;\n    \
                 float slab = 1.0 / hgl_t4_size;\n    \
                 float r0 = (clamp(q4f, 0.0, hgl_t4_size - 1.0) + gl_TexCoord[0].p) * slab;\n    \
                 float r1 = (clamp(q4f + 1.0, 0.0, hgl_t4_size - 1.0) + gl_TexCoord[0].p) * slab;\n    \
                 vec4 texel = mix(texture3D(hgl_tex4d, vec3(gl_TexCoord[0].st, r0)),\n        \
                 texture3D(hgl_tex4d, vec3(gl_TexCoord[0].st, r1)), q4a);",
            )
        } else {
            None
        };
        let fetch = if key & (1 << 24) != 0 {
            "vec2 uv = gl_TexCoord[0].st * hgl_texsize - 0.5;\n    \
             vec2 base = floor(uv);\n    \
             vec2 a = uv - base;\n    \
             vec4 wx = vec4(hgl_f4(1.0 + a.x), hgl_f4(a.x), hgl_f4(1.0 - a.x), hgl_f4(2.0 - a.x));\n    \
             vec4 wy = vec4(hgl_f4(1.0 + a.y), hgl_f4(a.y), hgl_f4(1.0 - a.y), hgl_f4(2.0 - a.y));\n    \
             vec4 texel = vec4(0.0);\n    \
             for (int j = 0; j < 4; ++j)\n        \
             for (int i = 0; i < 4; ++i)\n            \
             texel += wx[i] * wy[j] * texture2D(hgl_tex,\n                \
             (base + vec2(float(i) - 1.0, float(j) - 1.0) + 0.5) / hgl_texsize);"
        } else {
            "vec4 texel = texture2D(hgl_tex, gl_TexCoord[0].st);"
        };
        let textured = if let Some(cm) = clipmap {
            format!("{cm}\n    {env}")
        } else if let Some(f4) = fetch4d {
            format!("{f4}\n    {env}")
        } else if key & (1 << 17) != 0 {
            format!("{fetch}\n    {select}\n    {sharpen}\n    {detail}\n    {table}\n    {scale_bias}\n    {env}")
        } else {
            String::new()
        };
        // SGIX_reference_plane: the fragment's depth comes from a plane
        // rather than from the geometry, so coplanar things -- a road on a
        // terrain, a decal on a wall -- meet the depth test exactly.
        //
        // The plane is held in clip coordinates, so the fragment's window x
        // and y go back to normalised device coordinates through the viewport,
        // the plane gives z there, and the depth range carries it to a depth.
        // `-(d + a x + b y) / c` is the specification's own formula, in the
        // space the plane was transformed into.
        let ref_plane = if key & (1 << 23) != 0 {
            "vec2 nd = (gl_FragCoord.xy - hgl_viewport.xy) / hgl_viewport.zw * 2.0 - 1.0;\n    \
             float pz = -(hgl_refplane.w + hgl_refplane.x * nd.x + hgl_refplane.y * nd.y)\n        \
             / hgl_refplane.z;\n    \
             gl_FragDepth = 0.5 * (pz * gl_DepthRange.diff + gl_DepthRange.near + gl_DepthRange.far);"
        } else {
            // Left alone deliberately: a shader that writes gl_FragDepth on
            // any path gives up early depth rejection for every fragment.
            ""
        };
        // texture2DLodARB is how a fragment shader names a mipmap level in
        // GLSL 1.20; without the extension line the compiler will not have it.
        // GLSL 1.20 rather than the default 1.10: the bias form of texture2D
        // that `sharpen` uses to reach the coarser mipmap level belongs to the
        // fragment stage in both, but 1.20 is what the rest of this assumes.
        //
        // The obvious way to name a level, texture2DLodARB, is not available:
        // the Mac's context advertises GL_ARB_shader_texture_lod, yet its
        // legacy GLSL compiler rejects the identifier. Biasing the level the
        // hardware computed reaches the same array, and does it from the same
        // derivatives, so the two agree by construction.
        let src = format!(
            r#"#version 120
uniform sampler2D hgl_tex;
uniform sampler1D hgl_table;
uniform sampler1D hgl_fogfunc;
uniform vec4 hgl_scale;
uniform vec4 hgl_bias;
uniform float hgl_fog_x0;
uniform float hgl_fog_x1;
uniform float hgl_fog_offset;
uniform vec4 hgl_refplane;
uniform vec4 hgl_viewport;
uniform sampler1D hgl_filter4;
uniform sampler1D hgl_sharpen;
uniform sampler1D hgl_detailfunc;
uniform sampler2D hgl_detail;
uniform sampler3D hgl_tex4d;
uniform float hgl_t4_size;
uniform sampler2D hgl_cm;
uniform vec2 hgl_cm_centre;
uniform vec2 hgl_cm_offset;
uniform float hgl_cm_c;
uniform float hgl_cm_levels;
uniform float hgl_cm_v;

/* One texel of a clipmap band, wrapped into the resident window. A level whose
   virtual size has fallen below the clip size is held whole, and wraps at that
   smaller size instead -- which exp2(v - p) gives without a table. */
vec4 hgl_cm_texel(vec2 a, float p, float win)
{{
    vec2 w = mod(a, vec2(win));
    return texture2D(hgl_cm, vec2((w.x + 0.5) / hgl_cm_c,
        (p * hgl_cm_c + w.y + 0.5) / (hgl_cm_c * hgl_cm_levels)));
}}

/* Level p, bilinear across four wrapped taps. */
vec4 hgl_cm_level(vec2 v0, float p)
{{
    float scale = exp2(p);
    float win = min(hgl_cm_c, exp2(hgl_cm_v - p));
    vec2 a = v0 / scale + floor(hgl_cm_offset / scale) - 0.5;
    vec2 base = floor(a);
    vec2 f = a - base;
    return mix(mix(hgl_cm_texel(base, p, win), hgl_cm_texel(base + vec2(1.0, 0.0), p, win), f.x),
               mix(hgl_cm_texel(base + vec2(0.0, 1.0), p, win),
                   hgl_cm_texel(base + vec2(1.0, 1.0), p, win), f.x), f.y);
}}
uniform float hgl_detail_x0;
uniform float hgl_detail_x1;
uniform float hgl_detail_scale;
uniform vec2 hgl_texsize;
uniform float hgl_f4_n;
uniform float hgl_sharp_x0;
uniform float hgl_sharp_x1;
float hgl_f4(float d)
{{
    float i = clamp(d, 0.0, 2.0) * 0.5 * (hgl_f4_n - 1.0);
    return texture1D(hgl_filter4, (i + 0.5) / hgl_f4_n).r;
}}
void main()
{{
    vec4 c = gl_Color;
    {textured}
    {fog}
    {ref_plane}
    gl_FragColor = c;
}}
"#
        );
        unsafe {
            let sh = glCreateShader(FRAGMENT_SHADER);
            let c = CString::new(src).unwrap();
            glShaderSource(sh, 1, &c.as_ptr(), std::ptr::null());
            glCompileShader(sh);
            let mut ok = 0;
            glGetShaderiv(sh, COMPILE_STATUS, &mut ok);
            if ok != 1 {
                self.complain("compile", sh, false);
                return 0;
            }
            let p = glCreateProgram();
            glAttachShader(p, sh);
            glLinkProgram(p);
            glGetProgramiv(p, LINK_STATUS, &mut ok);
            if ok != 1 {
                self.complain("link", p, true);
                return 0;
            }
            // The texture the program bound is on unit 0; our table on unit 1.
            glUseProgram(p);
            for (name, unit) in [("hgl_tex", 0), ("hgl_table", 1), ("hgl_fogfunc", 2), ("hgl_filter4", 3), ("hgl_sharpen", 4),
                ("hgl_detailfunc", 5), ("hgl_detail", 6), ("hgl_tex4d", 7),
                ("hgl_cm", 9)] {
                let n = CString::new(name).unwrap();
                let loc = glGetUniformLocation(p, n.as_ptr());
                if loc >= 0 {
                    glUniform1i(loc, unit);
                }
            }
            glUseProgram(self.current);
            p
        }
    }

    /// A fragment shader made into a linked program, or 0 with the reason
    /// logged once. Shared by the fragment stage and the pixel-texture draw.
    fn compile(&mut self, src: &str) -> u32 {
        unsafe {
            let sh = glCreateShader(FRAGMENT_SHADER);
            let c = CString::new(src).unwrap();
            glShaderSource(sh, 1, &c.as_ptr(), std::ptr::null());
            glCompileShader(sh);
            let mut ok = 0;
            glGetShaderiv(sh, COMPILE_STATUS, &mut ok);
            if ok != 1 {
                self.complain("compile", sh, false);
                return 0;
            }
            let p = glCreateProgram();
            glAttachShader(p, sh);
            glLinkProgram(p);
            glGetProgramiv(p, LINK_STATUS, &mut ok);
            if ok != 1 {
                self.complain("link", p, true);
                return 0;
            }
            p
        }
    }

    fn complain(&mut self, what: &str, id: u32, program: bool) {
        if self.reported {
            return;
        }
        self.reported = true;
        let mut buf: Vec<c_char> = vec![0; 2048];
        let mut n = 0;
        let log = unsafe {
            if program {
                glGetProgramInfoLog(id, 2048, &mut n, buf.as_mut_ptr());
            } else {
                glGetShaderInfoLog(id, 2048, &mut n, buf.as_mut_ptr());
            }
            let bytes: Vec<u8> = buf[..n.max(0) as usize].iter().map(|&c| c as u8).collect();
            String::from_utf8_lossy(&bytes).into_owned()
        };
        log::warn!("host GL: the emulated fragment stage failed to {what}: {}", log.trim());
    }
}

/// The inverse of a column-major 4x4, or `None` when it has none.
///
/// Written out rather than pulled in: this is the only linear algebra in the
/// crate, and a dependency for one matrix is a poor trade.
fn invert4(m: &[f64; 16]) -> Option<[f64; 16]> {
    let a = |r: usize, c: usize| m[c * 4 + r];
    // Cofactor expansion, via the six 2x2 determinants of each row pair.
    let s0 = a(0, 0) * a(1, 1) - a(1, 0) * a(0, 1);
    let s1 = a(0, 0) * a(1, 2) - a(1, 0) * a(0, 2);
    let s2 = a(0, 0) * a(1, 3) - a(1, 0) * a(0, 3);
    let s3 = a(0, 1) * a(1, 2) - a(1, 1) * a(0, 2);
    let s4 = a(0, 1) * a(1, 3) - a(1, 1) * a(0, 3);
    let s5 = a(0, 2) * a(1, 3) - a(1, 2) * a(0, 3);
    let c5 = a(2, 2) * a(3, 3) - a(3, 2) * a(2, 3);
    let c4 = a(2, 1) * a(3, 3) - a(3, 1) * a(2, 3);
    let c3 = a(2, 1) * a(3, 2) - a(3, 1) * a(2, 2);
    let c2 = a(2, 0) * a(3, 3) - a(3, 0) * a(2, 3);
    let c1 = a(2, 0) * a(3, 2) - a(3, 0) * a(2, 2);
    let c0 = a(2, 0) * a(3, 1) - a(3, 0) * a(2, 1);
    let det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
    if !det.is_finite() || det.abs() < 1e-30 {
        return None;
    }
    let d = 1.0 / det;
    let mut o = [0f64; 16];
    let mut set = |r: usize, c: usize, v: f64| o[c * 4 + r] = v * d;
    set(0, 0, a(1, 1) * c5 - a(1, 2) * c4 + a(1, 3) * c3);
    set(0, 1, -a(0, 1) * c5 + a(0, 2) * c4 - a(0, 3) * c3);
    set(0, 2, a(3, 1) * s5 - a(3, 2) * s4 + a(3, 3) * s3);
    set(0, 3, -a(2, 1) * s5 + a(2, 2) * s4 - a(2, 3) * s3);
    set(1, 0, -a(1, 0) * c5 + a(1, 2) * c2 - a(1, 3) * c1);
    set(1, 1, a(0, 0) * c5 - a(0, 2) * c2 + a(0, 3) * c1);
    set(1, 2, -a(3, 0) * s5 + a(3, 2) * s2 - a(3, 3) * s1);
    set(1, 3, a(2, 0) * s5 - a(2, 2) * s2 + a(2, 3) * s1);
    set(2, 0, a(1, 0) * c4 - a(1, 1) * c2 + a(1, 3) * c0);
    set(2, 1, -a(0, 0) * c4 + a(0, 1) * c2 - a(0, 3) * c0);
    set(2, 2, a(3, 0) * s4 - a(3, 1) * s2 + a(3, 3) * s0);
    set(2, 3, -a(2, 0) * s4 + a(2, 1) * s2 - a(2, 3) * s0);
    set(3, 0, -a(1, 0) * c3 + a(1, 1) * c1 - a(1, 2) * c0);
    set(3, 1, a(0, 0) * c3 - a(0, 1) * c1 + a(0, 2) * c0);
    set(3, 2, -a(3, 0) * s3 + a(3, 1) * s1 - a(3, 2) * s0);
    set(3, 3, a(2, 0) * s3 - a(2, 1) * s1 + a(2, 2) * s0);
    Some(o)
}

/// A function given as (x, y) pairs, read at `x`: a straight line between
/// neighbouring points, and the end values held beyond either end -- which is
/// what SGIS_fog_function, SGIS_sharpen_texture and SGIS_detail_texture all
/// say their curves do.
fn piecewise(points: &[f32], n: usize, x: f32) -> f32 {
    if x <= points[0] {
        return points[1];
    }
    for k in 1..n {
        let (xa, ya) = (points[(k - 1) * 2], points[(k - 1) * 2 + 1]);
        let (xb, yb) = (points[k * 2], points[k * 2 + 1]);
        if x <= xb {
            return if xb > xa { ya + (yb - ya) * (x - xa) / (xb - xa) } else { yb };
        }
    }
    points[(n - 1) * 2 + 1]
}

fn norm(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if l < 1e-9 { [0.0, 0.0, 1.0] } else { [v[0] / l, v[1] / l, v[2] / l] }
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// A column-major rotation whose columns are the given basis.
fn basis(right: [f32; 3], up: [f32; 3], fwd: [f32; 3]) -> [f32; 16] {
    [
        right[0], right[1], right[2], 0.0,
        up[0], up[1], up[2], 0.0,
        fwd[0], fwd[1], fwd[2], 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]
}

/// SPRITE_AXIAL: turn about the axis so the front faces the eye as far as the
/// axis allows -- a tree that turns to face you but stays upright.
fn axial(axis: [f32; 3], eye: [f32; 3]) -> [f32; 16] {
    let up = norm(axis);
    // The eye direction with its component along the axis removed: what is
    // left is as far as the front can be turned towards it.
    let d = eye[0] * up[0] + eye[1] * up[1] + eye[2] * up[2];
    let fwd = norm([eye[0] - up[0] * d, eye[1] - up[1] * d, eye[2] - up[2] * d]);
    let right = norm(cross(up, fwd));
    basis(right, up, fwd)
}

/// SPRITE_OBJECT_ALIGNED / SPRITE_EYE_ALIGNED: turn about a point so the front
/// faces the eye outright, the remaining freedom spent aligning the top with
/// the axis.
fn face(axis: [f32; 3], eye: [f32; 3]) -> [f32; 16] {
    let fwd = norm(eye);
    let up0 = norm(axis);
    let r = cross(up0, fwd);
    // An axis parallel to the view leaves the roll undecided: any
    // perpendicular will do, and this picks one rather than dividing by zero.
    let right = if r[0].abs() + r[1].abs() + r[2].abs() < 1e-6 {
        norm(cross([0.0, 0.0, 1.0], fwd))
    } else {
        norm(r)
    };
    basis(right, cross(fwd, right), fwd)
}
