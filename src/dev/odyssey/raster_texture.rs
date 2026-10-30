//! Texture sampling for the scalar rasteriser: the texture pyramid and its
//! checks, texel reads with wrap, borders and the texel cache, level
//! selection, filtering, sharpen and detail textures, and the texture
//! environment that combines a texel with the fragment colour.
//!
//! Ported from hw/display/sgi/rasterizer.c in jamesbraid/qemu-sgi-origami.

use super::*;

impl SoftDraw<'_> {
    /// Runtime errors are observable without logging once for every pixel.
    pub fn texture_report(&self, reason: u32, message: &str) {
        let old = self.reports.texture.fetch_or(reason, std::sync::atomic::Ordering::Relaxed);
        if old & reason == 0 {
            crate::dlog!(LogModule::Odyssey, "sgi-gl: texture {}", message);
        }
    }
}

pub fn soft_texture_format(format: u32) -> bool {
    format == SGI_GL_ALPHA || format == SGI_GL_LUMINANCE ||
        format == SGI_GL_LUMINANCE_ALPHA || format == SGI_GL_INTENSITY ||
        format == SGI_GL_RGB || format == SGI_GL_RGBA ||
        format == SGI_GL_DEPTH_COMPONENT
}

pub fn soft_texture_wrap(wrap: u32) -> bool {
    wrap == SGI_GL_REPEAT || wrap == SGI_GL_CLAMP ||
        wrap == SGI_GL_CLAMP_TO_EDGE || wrap == SGI_GL_CLAMP_TO_BORDER
}

/// OpenGL 1.2.1 table 3.16: exact identities remain distinct in a chain.
pub fn soft_texture_internal_format(format: u32) -> u32 {
    match format {
        SGI_GL_DEPTH_COMPONENT16_SGIX | SGI_GL_DEPTH_COMPONENT24_SGIX | SGI_GL_DEPTH_COMPONENT32_SGIX => SGI_GL_DEPTH_COMPONENT,
        SGI_GL_ALPHA4 | SGI_GL_ALPHA8 | SGI_GL_ALPHA12 | SGI_GL_ALPHA16 => SGI_GL_ALPHA,
        SGI_GL_LUMINANCE4 | SGI_GL_LUMINANCE8 | SGI_GL_LUMINANCE12 | SGI_GL_LUMINANCE16 => SGI_GL_LUMINANCE,
        SGI_GL_LUMINANCE4_ALPHA4 | SGI_GL_LUMINANCE6_ALPHA2 | SGI_GL_LUMINANCE8_ALPHA8 |
        SGI_GL_LUMINANCE12_ALPHA4 | SGI_GL_LUMINANCE12_ALPHA12 | SGI_GL_LUMINANCE16_ALPHA16 => SGI_GL_LUMINANCE_ALPHA,
        SGI_GL_INTENSITY4 | SGI_GL_INTENSITY8 | SGI_GL_INTENSITY12 | SGI_GL_INTENSITY16 => SGI_GL_INTENSITY,
        SGI_GL_R3_G3_B2 | SGI_GL_RGB4 | SGI_GL_RGB5 | SGI_GL_RGB8 | SGI_GL_RGB10 | SGI_GL_RGB12 | SGI_GL_RGB16 => SGI_GL_RGB,
        SGI_GL_RGBA2 | SGI_GL_RGBA4 | SGI_GL_RGB5_A1 | SGI_GL_RGBA8 | SGI_GL_RGB10_A2 | SGI_GL_RGBA12 | SGI_GL_RGBA16 => SGI_GL_RGBA,
        _ => if soft_texture_format(format) { format } else { 0 },
    }
}

pub fn soft_texture_mipmap(filter: u32) -> bool {
    (SGI_GL_NEAREST_MIPMAP_NEAREST..=SGI_GL_LINEAR_MIPMAP_LINEAR).contains(&filter)
}

pub fn soft_texture_sharpen(filter: u32) -> bool {
    filter == SGI_GL_LINEAR_SHARPEN_SGIS || filter == SGI_GL_LINEAR_SHARPEN_ALPHA_SGIS ||
        filter == SGI_GL_LINEAR_SHARPEN_COLOR_SGIS
}

pub fn soft_texture_detail(filter: u32) -> bool {
    filter == SGI_GL_LINEAR_DETAIL_SGIS || filter == SGI_GL_LINEAR_DETAIL_ALPHA_SGIS ||
        filter == SGI_GL_LINEAR_DETAIL_COLOR_SGIS
}

pub fn soft_texture_dimensions(w: u32, h: u32) -> bool {
    w != 0 && h != 0 && w <= i32::MAX as u32 && h <= i32::MAX as u32 &&
        (w as u64).is_power_of_two() && (h as u64).is_power_of_two()
}

impl<'a> GlTexPyramid<'a> {
    pub fn soft_texture_level(&self, level: u32) -> Option<&'a GlTexLevel> {
        self.levels.iter().find(|l| l.level == level)
    }
}

impl GlTexImage<'_> {
    pub fn soft_texture_border_supported(&self, level: Option<&GlTexLevel>) -> bool {
        let border = match level { Some(l) => l.border, None => self.border };
        let callback = if self.volume {
            match level {
                Some(l) => l.read_border_texel_3d,
                None => self.read_border_texel_3d,
            }
        } else {
            match level {
                Some(l) => l.read_border_texel,
                None => self.read_border_texel,
            }
        };

        border <= 1 && (border == 0 || callback)
    }

    pub fn soft_texture_shadow_supported(&self, format: u32) -> bool {
        if (self.shadow_compare != 0 &&
            self.shadow_compare != SGI_GL_TEXTURE_LEQUAL_R_SGIX &&
            self.shadow_compare != SGI_GL_TEXTURE_GEQUAL_R_SGIX) ||
            (self.shadow_compare != 0 &&
             (!self.shadow_ambient.is_finite() || self.shadow_ambient < 0.0 || self.shadow_ambient > 1.0))
        {
            return false;
        }
        // SGIX_depth_texture alone does not define a texture environment.
        format != SGI_GL_DEPTH_COMPONENT ||
            (self.shadow_compare != 0 && !soft_texture_sharpen(self.mag_filter) &&
             !soft_texture_detail(self.mag_filter))
    }
}

impl<'a> SoftDraw<'a> {
    pub fn texture_pyramid(&mut self, im: &'a GlTexImage<'a>, report: bool) -> bool {
        let p = im.pyramid.expect("an image with a pyramid");
        let mipmap = soft_texture_mipmap(im.min_filter);
        let mut border_finite = true;

        #[derive(PartialEq)]
        enum Out { Unsupported, Incomplete, Done }
        let res = (|| -> Out {
            border_finite &= im.border_color.iter().all(|c| c.is_finite());
            if p.levels.len() > SGI_GL_TEX_LEVELS_MAX as usize ||
                !p.min_lod.is_finite() || !p.max_lod.is_finite() ||
                p.min_lod > p.max_lod || !border_finite ||
                !soft_texture_wrap(im.wrap_s) ||
                (!im.one_dimensional && !soft_texture_wrap(im.wrap_t)) ||
                (im.volume && !soft_texture_wrap(im.wrap_r)) ||
                (im.min_filter != SGI_GL_NEAREST && im.min_filter != SGI_GL_LINEAR && !mipmap) ||
                (im.mag_filter != SGI_GL_NEAREST && im.mag_filter != SGI_GL_LINEAR &&
                 !soft_texture_sharpen(im.mag_filter) && !soft_texture_detail(im.mag_filter))
            {
                return Out::Unsupported;
            }
            if p.levels.windows(2).any(|w| w[0].level >= w[1].level) {
                return Out::Unsupported;
            }
            if mipmap && p.max_level < p.base_level {
                return Out::Incomplete;
            }
            let base = match p.soft_texture_level(p.base_level) {
                Some(b) => b,
                None => return Out::Incomplete,
            };
            if !im.soft_texture_shadow_supported(base.base_format) ||
                !im.soft_texture_border_supported(Some(base)) ||
                soft_texture_internal_format(base.internal_format) != base.base_format ||
                !soft_texture_format(base.base_format) ||
                (self.st.texture_env_mode == SGI_GL_DECAL &&
                 base.base_format != SGI_GL_RGB && base.base_format != SGI_GL_RGBA)
            {
                return Out::Unsupported;
            }
            if im.one_dimensional && base.height != 1 {
                return Out::Unsupported;
            }
            if !soft_texture_dimensions(base.width, base.height) ||
                (im.volume && (base.depth == 0 || base.depth > i32::MAX as u32 || !(base.depth as u64).is_power_of_two()))
            {
                return Out::Incomplete;
            }
            let mut last = p.base_level;
            if mipmap {
                let mut extent = std::cmp::max(base.width, base.height);

                if im.volume {
                    extent = std::cmp::max(extent, base.depth);
                }
                let k = 31 - extent.leading_zeros();
                last = std::cmp::min(p.base_level as u64 + k as u64, p.max_level as u64) as u32;
            }
            // Unsupported relevant forms cannot be concealed by missing images.
            for l in p.levels {
                if l.level < p.base_level || l.level > last {
                    continue;
                }
                if im.one_dimensional && l.height != 1 {
                    return Out::Unsupported;
                }
                if !im.soft_texture_shadow_supported(l.base_format) {
                    return Out::Unsupported;
                }
                if !im.soft_texture_border_supported(Some(l)) ||
                    !soft_texture_format(l.base_format) ||
                    soft_texture_internal_format(l.internal_format) != l.base_format
                {
                    return Out::Unsupported;
                }
            }
            let mut id = p.base_level as u64;
            while id <= last as u64 {
                let l = p.soft_texture_level(id as u32);
                let shift = (id - p.base_level as u64) as u32;

                let l = match l { Some(l) => l, None => return Out::Incomplete };
                if (if im.volume { !l.read_texel_3d } else { !l.read_texel }) ||
                    (im.volume && l.depth != std::cmp::max(1, base.depth.wrapping_shr(shift))) ||
                    l.width != std::cmp::max(1, base.width.wrapping_shr(shift)) ||
                    l.height != std::cmp::max(1, base.height.wrapping_shr(shift)) ||
                    l.internal_format != base.internal_format ||
                    l.border != base.border
                {
                    return Out::Incomplete;
                }
                id += 1;
            }
            self.texture = Some(im);
            self.texture_base = Some(base);
            self.texture_last = last;
            Out::Done
        })();
        match res {
            Out::Done => true,
            Out::Unsupported => {
                if report {
                    crate::dlog!(LogModule::Odyssey, "sgi-gl: unsupported texture state in pyramid, draw dropped");
                }
                false
            }
            Out::Incomplete => {
                if report {
                    crate::dlog!(LogModule::Odyssey, "sgi-gl: incomplete texture image pyramid, texturing disabled for draw");
                }
                true
            }
        }
    }

    /// Incomplete images disable texturing; unsupported states drop the draw.
    pub fn texture_setup(&mut self, report: bool) -> bool {
        let im = self.t.texture.as_ref();
        let st = self.st;

        // With several targets enabled the highest dimension wins: 3D, then
        // 2D, then 1D (OpenGL 1.2.1, 3.8.10). The family supplies the winning
        // target's image, so the image is checked against that target only.
        let target: u32 = if st.texture_3d != 0 { 3 } else if st.texture_2d != 0 { 2 } else if st.texture_1d != 0 { 1 } else { 0 };

        if target == 0 || st.index_bits != 0 {
            return true;
        }
        if let Some(im) = im {
            if (im.one_dimensional && im.volume) ||
                (target == 1) != im.one_dimensional ||
                (im.one_dimensional && im.pyramid.is_none() && im.height != 1)
            {
                if report {
                    crate::dlog!(LogModule::Odyssey, "sgi-gl: incompatible 1D texture image, draw dropped");
                }
                return false;
            }
        }
        // A volume image under the 2D enable alone stays accepted, as before
        // targets were ranked. With 1D enabled as well the 2D binding wins,
        // and a volume is not a 2D image.
        if target == 2 && im.is_some_and( |i| i.volume) && st.texture_1d != 0 {
            if report {
                crate::dlog!(LogModule::Odyssey, "sgi-gl: volume texture image for a 2D target, draw dropped");
            }
            return false;
        }
        if target == 3 && im.is_some_and( |i| !i.volume) {
            if report {
                crate::dlog!(LogModule::Odyssey, "sgi-gl: 3D texturing requires a volume image, draw dropped");
            }
            return false;
        }
        if st.texture_env_mode != SGI_GL_REPLACE && st.texture_env_mode != SGI_GL_MODULATE &&
            st.texture_env_mode != SGI_GL_DECAL && st.texture_env_mode != SGI_GL_BLEND
        {
            if report {
                crate::dlog!(LogModule::Odyssey, "sgi-gl: unsupported texture state environment {:#x}, draw dropped", st.texture_env_mode);
            }
            return false;
        }
        if let Some(im) = im {
            if im.max_clamp &&
                !(im.max_clamp_s >= 0.0 && im.max_clamp_s <= 1.0 &&
                  im.max_clamp_t >= 0.0 && im.max_clamp_t <= 1.0 &&
                  im.max_clamp_r >= 0.0 && im.max_clamp_r <= 1.0)
            {
                if report {
                    crate::dlog!(LogModule::Odyssey, "sgi-gl: invalid coordinate clamp, draw dropped");
                }
                return false;
            }
            if !im.lod_scale_s.is_finite() || im.lod_scale_s < 0.0 ||
                (!im.one_dimensional && (!im.lod_scale_t.is_finite() || im.lod_scale_t < 0.0)) ||
                (im.volume && (!im.lod_scale_r.is_finite() || im.lod_scale_r < 0.0))
            {
                if report {
                    crate::dlog!(LogModule::Odyssey, "sgi-gl: invalid derivative scales, draw dropped");
                }
                return false;
            }
            if im.pyramid.is_some() {
                return self.texture_pyramid(im, report);
            }
            let mut border_finite = true;

            border_finite &= im.border_color.iter().all(|c| c.is_finite());
            if !im.soft_texture_shadow_supported(im.base_format) ||
                !im.soft_texture_border_supported(None) ||
                !border_finite ||
                !soft_texture_format(im.base_format) ||
                !soft_texture_wrap(im.wrap_s) ||
                (!im.one_dimensional && !soft_texture_wrap(im.wrap_t)) ||
                (im.volume && !soft_texture_wrap(im.wrap_r)) ||
                im.min_filter != im.mag_filter ||
                (im.min_filter != SGI_GL_NEAREST && im.min_filter != SGI_GL_LINEAR) ||
                (st.texture_env_mode == SGI_GL_DECAL && im.base_format != SGI_GL_RGB && im.base_format != SGI_GL_RGBA)
            {
                if report {
                    crate::dlog!(
                        LogModule::Odyssey,
                        "sgi-gl: unsupported texture state format {:#x} border {} filters {:#x}/{:#x} wraps {:#x}/{:#x}/{:#x} environment {:#x}{}, draw dropped",
                        im.base_format, im.border, im.min_filter, im.mag_filter, im.wrap_s, im.wrap_t,
                        im.wrap_r, st.texture_env_mode,
                        if border_finite { "" } else { " nonfinite border colour" }
                    );
                }
                return false;
            }
        }
        let incomplete = match im {
            None => true,
            Some(im) => {
                (if im.volume { !im.read_texel_3d } else { !im.read_texel }) ||
                    im.width == 0 || im.height == 0 ||
                    im.width > i32::MAX as u32 || im.height > i32::MAX as u32 ||
                    !(im.width as u64).is_power_of_two() || !(im.height as u64).is_power_of_two() ||
                    (im.volume && (im.depth == 0 || im.depth > i32::MAX as u32 || !(im.depth as u64).is_power_of_two()))
            }
        };
        if incomplete {
            if report {
                crate::dlog!(LogModule::Odyssey, "sgi-gl: incomplete texture image, texturing disabled for draw");
            }
            return true;
        }
        self.texture = im;
        true
    }
}

impl GlTexImage<'_> {
    /// Whether a clipped vertex has a clip w and the texture coordinates im
    /// uses: s and q always, t unless the image is one-dimensional, and r for
    /// a shadow comparison against a depth image.
    pub fn soft_texcoord_valid(&self, base: Option<&GlTexLevel>, v: &VolumeVertex) -> bool {
        let mut valid = v.clip[3].is_finite() && v.clip[3] > 0.0 && v.attr[SGI_GL_V_TEXCOORD + 3] > 0.0;

        for k in 0..4usize {
            if !self.one_dimensional || k == 0 || k == 3 ||
                (k == 2 && self.shadow_compare != 0 &&
                 base.map_or(self.base_format, |b| b.base_format) == SGI_GL_DEPTH_COMPONENT)
            {
                valid &= v.attr[SGI_GL_V_TEXCOORD + k].is_finite();
            }
        }
        valid
    }
}

impl SoftDraw<'_> {
    pub fn texture_vertex_valid(&self, v: &VolumeVertex) -> bool {
        let Some(im) = self.texture else {
            return true;
        };
        let valid = im.soft_texcoord_valid(self.texture_base, v);
        if !valid {
            self.texture_report(4, "invalid clipped coordinates, primitive dropped");
        }
        valid
    }
}

/// Meaningful base components, including the border, before filtering.
pub fn soft_texture_components(format: u32, v: &mut [f64; 4]) {
    match format {
        SGI_GL_ALPHA => {
            v[0] = 1.0;
            v[1] = 1.0;
            v[2] = 1.0;
        }
        SGI_GL_LUMINANCE => {
            v[3] = 1.0;
            v[1] = v[0];
            v[2] = v[0];
        }
        SGI_GL_LUMINANCE_ALPHA => {
            v[1] = v[0];
            v[2] = v[0];
        }
        SGI_GL_INTENSITY => {
            v[1] = v[0];
            v[2] = v[0];
            v[3] = v[0];
        }
        SGI_GL_RGB => {
            v[3] = 1.0;
        }
        _ => {}
    }
}

impl GlTexImage<'_> {
    pub fn soft_texture_shadow(&self, depth: f64, reference: f64, out: &mut [f64; 4]) {
        let pass = if self.shadow_compare == SGI_GL_TEXTURE_LEQUAL_R_SGIX { depth <= reference } else { depth >= reference };

        let v = if pass { 1.0 } else { self.shadow_ambient };
        out[0] = v;
        out[1] = v;
        out[2] = v;
        out[3] = 1.0;
    }
}

/// GL_REPEAT's wrap of a texel index into [0, n). Image and level sizes are
/// powers of two wherever setup admits them, and there the mask gives the
/// remainder without two 64-bit divisions a tap.
pub fn soft_texture_repeat(i: i64, n: u32) -> i64 {
    if (n as u64).is_power_of_two() {
        i & (n as i64 - 1)
    } else {
        (i % n as i64 + n as i64) % n as i64
    }
}

/// An 8-bit channel normalized, k / 255.0 for each byte k.
pub fn soft_unorm8(k: u32) -> f64 {
    k as f64 / 255.0
}

pub const SGI_SOFT_TEXEL_CACHE_SIZE: usize = 4096;

#[derive(Clone, Copy, Default)]
pub struct SoftTexelEntry {
    pub serial: u64,
    /// SoftDraw::texel_key() of the level
    pub level: usize,
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub out: [f64; 4],
}

/// The texels one draw has read, by level and interior index, after
/// conversion to components in [0, 1]. A family's image stays unchanged
/// through a synchronous draw (GlTexImage), so a texel read twice in one
/// draw reads the same both times. A draw starts by taking a new serial,
/// which empties the table without clearing it. Border texels, depth
/// texels and failed reads always go to the family.
pub struct SoftTexelCache {
    pub serial: u64,
    /// Set for a fragment that minifies by more than two texels a pixel
    /// without mipmaps.
    pub bypass: bool,
    pub tag: u32,
    pub fixed: Vec<SoftFixedStore>,
    pub entry: Vec<SoftTexelEntry>,
}

impl SoftTexelCache {
    pub fn new() -> Self {
        SoftTexelCache {
            serial: 0,
            bypass: false,
            tag: 0,
            fixed: vec![SoftFixedStore::default(); SGI_SOFT_FIXED_LEVELS],
            entry: vec![SoftTexelEntry::default(); SGI_SOFT_TEXEL_CACHE_SIZE],
        }
    }

    /// Start a draw: a new serial, which empties the table.
    pub fn begin(&mut self) {
        self.serial += 1;
        self.bypass = false;
        self.tag = (self.serial & !(SGI_SOFT_FIXED_FAILED as u64)) as u32;
        if self.tag == 0 {
            // Tags have come round: cells from 2^31 draws ago would match.
            for store in self.fixed.iter_mut() {
                store.cell.fill(0);
            }
            self.serial += 1;
            self.tag = (self.serial & !(SGI_SOFT_FIXED_FAILED as u64)) as u32;
        }
    }
}

impl Default for SoftTexelCache {
    fn default() -> Self {
        Self::new()
    }
}

impl SoftTexelCache {
    pub fn texel_slot(&self, level: usize, x: u32, y: u32, z: u32) -> usize {
        let hash: u32 = x.wrapping_mul(0x9e3779b1)
            ^ y.wrapping_mul(0x85ebca77)
            ^ z.wrapping_mul(0xc2b2ae3d)
            ^ (level as u32).wrapping_mul(0x27d4eb2f);

        ((hash ^ (hash >> 15)) as usize) & (SGI_SOFT_TEXEL_CACHE_SIZE - 1)
    }
}

impl SoftDraw<'_> {
    /// None is the read-failure exit.
    pub fn texture_tap_inner(&self, level: Option<&GlTexLevel>, mut x: i64, mut y: i64, mut z: i64, reference: f64, out: &mut [f64; 4]) -> Option<bool> {
        let im = self.image();
        let width = level.map_or(im.width, |l| l.width);
        let height = level.map_or(im.height, |l| l.height);
        let depth = if im.volume { level.map_or(im.depth, |l| l.depth) } else { 1 };
        let format = level.map_or(im.base_format, |l| l.base_format);
        let mut slot: Option<usize> = None;
        // The cache's name for the level: 0 for the image itself.
        let key = level.map_or(0usize, |l| l.level as usize + 1);

        if im.wrap_s == SGI_GL_REPEAT {
            x = soft_texture_repeat(x, width);
        }
        if !im.one_dimensional && im.wrap_t == SGI_GL_REPEAT {
            y = soft_texture_repeat(y, height);
        }
        if im.volume && im.wrap_r == SGI_GL_REPEAT {
            z = soft_texture_repeat(z, depth);
        }
        let rgba = if x < 0 || y < 0 || z < 0 || x >= width as i64 || y >= height as i64 || z >= depth as i64 {
            let border = level.map_or(im.border, |l| l.border);

            if border == 0 {
                for k in 0..4usize {
                    out[k] = fmin(fmax(im.border_color[k] as f64, 0.0), 1.0);
                }
                if format == SGI_GL_DEPTH_COMPONENT {
                    let o0 = out[0];
                    im.soft_texture_shadow(o0, reference, out);
                } else {
                    soft_texture_components(format, out);
                }
                return Some(true);
            }
            // Signed indices cover the stored faces, edges and corners.
            if x < -1 || y < -1 || x > width as i64 || y > height as i64 ||
                (im.volume && (z < -1 || z > depth as i64))
            {
                return None;
            }
            let readable = if im.volume {
                level.map_or(im.read_border_texel_3d, |l| l.read_border_texel_3d)
            } else {
                level.map_or(im.read_border_texel, |l| l.read_border_texel)
            };
            if !readable {
                return None;
            }
            if im.volume {
                self.texels.border_texel_3d(level, x as i32, y as i32, z as i32)?
            } else {
                self.texels.border_texel(level, x as i32, y as i32)?
            }
        } else {
            if let Some(cache) = self.texels_cache {
                let c = cache.borrow();

                if !c.bypass && format != SGI_GL_DEPTH_COMPONENT {
                    let idx = c.texel_slot(key, x as u32, y as u32, z as u32);
                    let e = &c.entry[idx];
                    if e.serial == c.serial && e.level == key &&
                        e.x as i64 == x && e.y as i64 == y && e.z as i64 == z
                    {
                        *out = e.out;
                        return Some(true);
                    }
                    slot = Some(idx);
                }
            }
            if im.volume {
                self.texels.texel_3d(level, x as u32, y as u32, z as u32)?
            } else {
                self.texels.texel(level, x as u32, y as u32)?
            }
        };
        if format == SGI_GL_DEPTH_COMPONENT {
            im.soft_texture_shadow(rgba as f64 / u32::MAX as f64, reference, out);
            return Some(true);
        }
        for k in 0..4u32 {
            out[k as usize] = soft_unorm8((rgba >> (24 - k * 8)) & 255);
        }
        soft_texture_components(format, out);
        if let (Some(idx), Some(cache)) = (slot, self.texels_cache) {
            let mut c = cache.borrow_mut();
            let serial = c.serial;

            c.entry[idx] = SoftTexelEntry {
                serial,
                level: key,
                x: x as u32,
                y: y as u32,
                z: z as u32,
                out: *out,
            };
        }
        Some(true)
    }

    pub fn texture_tap(&self, level: Option<&GlTexLevel>, x: i64, y: i64, z: i64, reference: f64, out: &mut [f64; 4]) -> bool {
        match self.texture_tap_inner(level, x, y, z, reference, out) {
            Some(b) => b,
            None => {
                self.texture_report(1, "image-read failure, fragment dropped");
                false
            }
        }
    }
}

/// Reduce finite coordinates before any float-to-integer conversion.
pub fn soft_texture_axis(x: f64, n: u32, wrap: u32, upper: f64) -> f64 {
    let half = 0.5 / n as f64;

    match wrap {
        SGI_GL_REPEAT => x - x.floor(),
        SGI_GL_CLAMP_TO_EDGE => fmin(fmax(x, half), 1.0 - half),
        SGI_GL_CLAMP_TO_BORDER => fmin(fmax(x, -half), 1.0 + half),
        _ => fmin(fmax(x, 0.0), upper),
    }
}

impl SoftDraw<'_> {
    /// Sampling always reduces coordinates before any integer conversion.
    pub fn texture_sample(&self, level: Option<&GlTexLevel>, filter: u32, s: f64, t: f64, rr: f64, texel: &mut [f64; 4]) -> bool {
        let im = self.image();
        let width = level.map_or(im.width, |l| l.width);
        let height = level.map_or(im.height, |l| l.height);
        let depth = if im.volume { level.map_or(im.depth, |l| l.depth) } else { 1 };

        let reference = rr;
        let mut rr = rr;

        *texel = [0.0; 4];
        let s = soft_texture_axis(s, width, im.wrap_s, if im.max_clamp { im.max_clamp_s } else { 1.0 });
        let t = if im.one_dimensional {
            0.0
        } else {
            soft_texture_axis(t, height, im.wrap_t, if im.max_clamp { im.max_clamp_t } else { 1.0 })
        };
        if im.volume {
            rr = soft_texture_axis(rr, depth, im.wrap_r, if im.max_clamp { im.max_clamp_r } else { 1.0 });
        }
        if filter == SGI_GL_NEAREST {
            let mut x = (width as f64 * s).floor() as i64;
            let mut y = (height as f64 * t).floor() as i64;
            let mut z = if im.volume { (depth as f64 * rr).floor() as i64 } else { 0 };

            if s == 1.0 && im.wrap_s != SGI_GL_CLAMP_TO_BORDER {
                x = width as i64 - 1;
            }
            if t == 1.0 && im.wrap_t != SGI_GL_CLAMP_TO_BORDER {
                y = height as i64 - 1;
            }
            if rr == 1.0 && im.volume && im.wrap_r != SGI_GL_CLAMP_TO_BORDER {
                z = depth as i64 - 1;
            }
            return self.texture_tap(level, x, y, z, reference, texel);
        }
        let u = width as f64 * s - 0.5;
        let v = if im.one_dimensional { 0.0 } else { height as f64 * t - 0.5 };
        let x = u.floor() as i64;
        let y = v.floor() as i64;
        let w = depth as f64 * rr - 0.5;
        let z = if im.volume { w.floor() as i64 } else { 0 };
        let a = u - x as f64;
        let b = v - y as f64;
        let c = if im.volume { w - z as f64 } else { 0.0 };

        for k in 0..(if im.volume { 2 } else { 1 }) {
            for j in 0..(if im.one_dimensional { 1 } else { 2 }) {
                for i in 0..2 {
                    let weight = (if i != 0 { a } else { 1.0 - a }) * (if j != 0 { b } else { 1.0 - b })
                        * (if im.volume { if k != 0 { c } else { 1.0 - c } } else { 1.0 });
                    let mut tap = [0.0f64; 4];

                    if weight == 0.0 {
                        continue;
                    }
                    if !self.texture_tap(level, x + i as i64, y + j as i64, z + k as i64, reference, &mut tap) {
                        return false;
                    }
                    for channel in 0..4usize {
                        texel[channel] += weight * tap[channel];
                    }
                }
            }
        }
        true
    }
}

#[derive(Clone, Copy, Default)]
pub struct SoftTexSelection {
    pub first: u32,
    pub second: u32,
    pub filter: u32,
    pub alpha: f64,
    pub lambda: f64,
}

/// Relative levels avoid rounding a large absolute base in floating point.
pub fn soft_texture_select(min: u32, mag: u32, p: &GlTexPyramid, last: u32, rho: f64, out: &mut SoftTexSelection) -> bool {
    let enhanced = soft_texture_sharpen(mag) || soft_texture_detail(mag);
    let c: f64 = if enhanced {
        if min == SGI_GL_NEAREST_MIPMAP_NEAREST || min == SGI_GL_LINEAR_MIPMAP_NEAREST { 0.5 } else { 0.0 }
    } else if mag == SGI_GL_LINEAR && (min == SGI_GL_NEAREST_MIPMAP_NEAREST || min == SGI_GL_NEAREST_MIPMAP_LINEAR) {
        0.5
    } else {
        0.0
    };
    let range: u32 = last.wrapping_sub(p.base_level);

    if !rho.is_finite() || rho < 0.0 {
        return false;
    }
    let mut lambda = if rho != 0.0 { rho.log2() } else { p.min_lod };
    if !lambda.is_finite() {
        return false;
    }
    lambda = fmin(fmax(lambda, p.min_lod), p.max_lod);
    *out = SoftTexSelection {
        first: p.base_level,
        second: p.base_level,
        filter: if lambda <= c { mag } else { min },
        alpha: 0.0,
        lambda,
    };
    if lambda <= c || !soft_texture_mipmap(min) {
        return true;
    }
    out.filter = if min == SGI_GL_NEAREST_MIPMAP_NEAREST || min == SGI_GL_NEAREST_MIPMAP_LINEAR {
        SGI_GL_NEAREST
    } else {
        SGI_GL_LINEAR
    };
    if min == SGI_GL_NEAREST_MIPMAP_NEAREST || min == SGI_GL_LINEAR_MIPMAP_NEAREST {
        let mut relative: f64;

        if lambda <= 0.5 {
            relative = 0.0;
        } else if lambda > range as f64 + 0.5 {
            relative = range as f64;
        } else {
            relative = lambda.floor();
            // Adding 0.5 can round the next value above a half tie onto it.
            relative += (lambda - relative > 0.5) as i32 as f64;
        }
        out.first = out.first.wrapping_add(fmin(fmax(relative, 0.0), range as f64) as u32);
        out.second = out.first;
    } else {
        let relative = fmin(fmax(lambda, 0.0), range as f64);
        let n = relative.floor() as u32;

        out.first = out.first.wrapping_add(n);
        out.second = if out.first == last { last } else { out.first.wrapping_add(1) };
        out.alpha = relative - n as f64;
    }
    true
}

/// Projective gradients precede wrapping; D is Q/W, not geometry W.
pub fn soft_texture_lod_scale(scale: f64) -> f64 {
    if scale != 0.0 { scale } else { 1.0 }
}

pub fn soft_texture_derivatives(
    value: &[f64; 4],
    dx: &[f64; 4],
    dy: &[f64; 4],
    ns: f64,
    nt: f64,
    nr: f64,
    out: &mut [f64; 5],
) -> bool {
    let d = value[2];
    let d2 = d * d;

    if !(d > 0.0) || !d2.is_finite() || d2 == 0.0 {
        return false;
    }
    for k in 0..2usize {
        out[2 * k] = (dx[k] * d - value[k] * dx[2]) / d2;
        out[2 * k + 1] = (dy[k] * d - value[k] * dy[2]) / d2;
        if !(value[k] / d).is_finite() || !out[2 * k].is_finite() || !out[2 * k + 1].is_finite() {
            return false;
        }
    }
    out[4] = fmax((ns * out[0]).hypot(nt * out[2]), (ns * out[1]).hypot(nt * out[3]));
    if nr != 0.0 {
        let drdx = (dx[3] * d - value[3] * dx[2]) / d2;
        let drdy = (dy[3] * d - value[3] * dy[2]) / d2;

        if !(value[3] / d).is_finite() || !drdx.is_finite() || !drdy.is_finite() {
            return false;
        }
        out[4] = fmax(
            (ns * out[0]).hypot(nt * out[2]).hypot(nr * drdx),
            (ns * out[1]).hypot(nt * out[3]).hypot(nr * drdy),
        );
    }
    out[4].is_finite()
}

/// The last level selection a thread's draw made, by the footprint it was
/// made for.
#[derive(Clone, Copy, Default)]
pub struct SoftTexSelectionMemo {
    pub serial: u64,
    pub rho: u64,
    pub selected: SoftTexSelection,
}

impl SoftDraw<'_> {
    pub fn texture_select_memo(&self, rho: f64, out: &mut SoftTexSelection) -> bool {
        let im = self.image();
        let serial = self.texels_cache.map(|c| c.borrow().serial);
        let bits = rho.to_bits();

        if let Some(serial) = serial {
            let m = SOFT_TEXTURE_SELECTED.with(|c| c.get());
            if m.serial == serial && m.rho == bits {
                *out = m.selected;
                return true;
            }
        }
        if !soft_texture_select(im.min_filter, im.mag_filter, &im.pyramid.expect("a pyramid"), self.texture_last, rho, out) {
            return false;
        }
        if let Some(serial) = serial {
            SOFT_TEXTURE_SELECTED.with(|c| c.set(SoftTexSelectionMemo { serial, rho: bits, selected: *out }));
        }
        true
    }
}

pub fn soft_texture_mix(value: &mut [f64; 4], other: &[f64; 4], alpha: f64) {
    for k in 0..4usize {
        value[k] = (1.0 - alpha) * value[k] + alpha * other[k];
    }
}

impl SoftDraw<'_> {
    pub fn texture_weight(&self, lambda: f64) -> Option<f64> {
        let im = self.image();

        if im.read_filter_weight {
            match self.texels.filter_weight(lambda) {
                Some(w) if w.is_finite() => return Some(w),
                _ => {
                    self.texture_report(16, "invalid filter weight, fragment dropped");
                    return None;
                }
            }
        }
        Some(fmin(fmax(-lambda / 4.0, 0.0), 1.0))
    }

    pub fn texture_sharpen_sample(&self, filter: u32, lambda: f64, s: f64, t: f64, rr: f64, texel: &mut [f64; 4]) -> bool {
        let im = self.image();
        let p = im.pyramid.expect("a pyramid");
        let base = self.texture_base.expect("a base level");
        let mut extent = std::cmp::max(base.width, base.height);
        let mut other = [0.0f64; 4];

        if !self.texture_sample(Some(base), SGI_GL_LINEAR, s, t, rr, texel) {
            return false;
        }
        if im.volume {
            extent = std::cmp::max(extent, base.depth);
        }
        let coarse = if p.base_level < u32::MAX { p.soft_texture_level(p.base_level + 1) } else { None };
        let coarse = match coarse {
            Some(c) if !(extent <= 1 || p.max_level <= p.base_level ||
                c.width != std::cmp::max(1, base.width / 2) ||
                c.height != std::cmp::max(1, base.height / 2) ||
                (im.volume && c.depth != std::cmp::max(1, base.depth / 2)) ||
                c.internal_format != base.internal_format ||
                c.base_format != base.base_format ||
                c.border != base.border ||
                !im.soft_texture_border_supported(Some(c)) ||
                (if im.volume { !c.read_texel_3d } else { !c.read_texel })) => c,
            _ => return true,
        };
        let Some(weight) = self.texture_weight(lambda) else {
            return false;
        };
        if !self.texture_sample(Some(coarse), SGI_GL_LINEAR, s, t, rr, &mut other) {
            return false;
        }
        for k in 0..4usize {
            if (k == 3 && filter == SGI_GL_LINEAR_SHARPEN_COLOR_SGIS) ||
                (k != 3 && filter == SGI_GL_LINEAR_SHARPEN_ALPHA_SGIS)
            {
                continue;
            }
            texel[k] = fmin(fmax(texel[k] + weight * (texel[k] - other[k]), 0.0), 1.0);
        }
        true
    }
}

/// scalbn: x * 2^n, exact.
pub fn scalbn(x: f64, n: i32) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let mut x = x;
    let mut n = n.clamp(-3000, 3000);
    while n > 1000 {
        x *= 2f64.powi(1000);
        n -= 1000;
    }
    while n < -1000 {
        x *= 2f64.powi(-1000);
        n += 1000;
    }
    x * 2f64.powi(n)
}

impl SoftDraw<'_> {
    pub fn texture_detail_sample(&self, filter: u32, lambda: f64, s: f64, t: f64, rr: f64, texel: &mut [f64; 4]) -> bool {
        let im = self.image();
        let base = self.texture_base.expect("a base level");
        let mut repeat = *im;
        let mut other = [0.0f64; 4];
        let scale: i32 = std::cmp::min(-(im.detail_level as i64), i32::MAX as i64) as i32;

        if !self.texture_sample(Some(base), SGI_GL_LINEAR, s, t, rr, texel) {
            return false;
        }
        let detail = match im.detail_image {
            Some(d) if !(im.one_dimensional || im.volume || im.pyramid.is_none_or(|p| p.base_level != 0) ||
                d.level != 0 || d.border != 0 ||
                !d.read_texel ||
                !soft_texture_dimensions(d.width, d.height) ||
                d.internal_format != base.internal_format ||
                d.base_format != base.base_format || im.detail_level > 0) => d,
            _ => return true,
        };
        if im.detail_mode != SGI_GL_ADD && im.detail_mode != SGI_GL_MODULATE {
            self.texture_report(32, "unsupported detail mode, fragment dropped");
            return false;
        }
        // Scaling is in primary texels, independent of the detail dimensions.
        let ds = scalbn(s * base.width as f64 / detail.width as f64, scale);
        let dt = scalbn(t * base.height as f64 / detail.height as f64, scale);
        if !ds.is_finite() || !dt.is_finite() {
            self.texture_report(32, "invalid detail coordinates, fragment dropped");
            return false;
        }
        repeat.wrap_s = SGI_GL_REPEAT;
        repeat.wrap_t = SGI_GL_REPEAT;
        // The detail image is not the draw's image: its texels are not cached.
        let mut draw = *self;
        draw.texture = Some(&repeat);
        draw.texels_cache = None;
        let Some(weight) = self.texture_weight(lambda) else {
            return false;
        };
        if !draw.texture_sample(Some(&detail), SGI_GL_LINEAR, ds, dt, 0.0, &mut other) {
            return false;
        }
        for k in 0..4usize {
            let signed_detail = weight * (2.0 * other[k] - 1.0);

            if (k == 3 && filter == SGI_GL_LINEAR_DETAIL_COLOR_SGIS) ||
                (k != 3 && filter == SGI_GL_LINEAR_DETAIL_ALPHA_SGIS)
            {
                continue;
            }
            texel[k] = if im.detail_mode == SGI_GL_ADD {
                texel[k] + signed_detail
            } else {
                texel[k] * (1.0 + signed_detail)
            };
            texel[k] = fmin(fmax(texel[k], 0.0), 1.0);
        }
        true
    }

    /// A filtered texel through the image's post-filter, if it has one, and
    /// then by the texture environment env into a fragment's colour rgba,
    /// for an image of base format format (OpenGL 1.2.1, 3.8.9).
    #[inline(always)]
    pub fn texture_apply(&self, format: u32, env: u32, texel: &mut [f64; 4], rgba: &mut [f64; 4]) {
        let im = self.image();
        let mut format = format;

        if im.post_filter {
            self.texels.post_filter(texel);
        }
        if format == SGI_GL_DEPTH_COMPONENT {
            format = SGI_GL_LUMINANCE;
        }
        for k in 0..4usize {
            let alpha = k == 3;

            if (!alpha && format == SGI_GL_ALPHA) ||
                (alpha && (format == SGI_GL_RGB || format == SGI_GL_LUMINANCE))
            {
                continue;
            }
            match env {
                SGI_GL_REPLACE => rgba[k] = texel[k],
                SGI_GL_MODULATE => rgba[k] *= texel[k],
                SGI_GL_DECAL => {
                    if !alpha {
                        rgba[k] = if format == SGI_GL_RGB {
                            texel[k]
                        } else {
                            rgba[k] * (1.0 - texel[3]) + texel[k] * texel[3]
                        };
                    }
                }
                SGI_GL_BLEND => {
                    rgba[k] = if alpha && format != SGI_GL_INTENSITY {
                        rgba[k] * texel[k]
                    } else {
                        rgba[k] * (1.0 - texel[k])
                            + fmin(fmax(self.st.texture_env_color[k] as f64, 0.0), 1.0) * texel[k]
                    };
                }
                _ => {}
            }
        }
    }

    pub fn texture(&self, frag: &mut SoftFrag) -> bool {
        let im = self.image();
        let st: &GlDrawState = self.st;
        let base = self.texture_base;
        let format = base.map_or(im.base_format, |b| b.base_format);
        let mut texel = [0.0f64; 4];
        let mut rr = 0.0f64;
        let q = frag.texcoord[3];

        if !(q > 0.0) || !q.is_finite() {
            self.texture_report(2, "invalid projected coordinates, fragment dropped");
            return false;
        }
        let s = frag.texcoord[0] / q;
        let t = if im.one_dimensional { 0.0 } else { frag.texcoord[1] / q };
        if im.volume || (format == SGI_GL_DEPTH_COMPONENT && im.shadow_compare != 0) {
            rr = frag.texcoord[2] / q;
        }
        if !s.is_finite() || !t.is_finite() || !rr.is_finite() ||
            (format == SGI_GL_DEPTH_COMPONENT && (rr < 0.0 || rr > 1.0))
        {
            self.texture_report(2, "invalid projected coordinates, fragment dropped");
            return false;
        }
        if let Some(base) = base {
            let mut selected = SoftTexSelection::default();
            let mut other = [0.0f64; 4];

            if !self.texture_select_memo(frag.texture_rho, &mut selected) {
                self.texture_report(8, "invalid footprint, fragment dropped");
                return false;
            }
            let pyramid = im.pyramid.expect("a pyramid");
            let mut level = pyramid.soft_texture_level(selected.first);
            if let Some(cache) = self.texels_cache {
                cache.borrow_mut().bypass = selected.lambda > 1.0 && !soft_texture_mipmap(im.min_filter);
            }
            if soft_texture_sharpen(selected.filter) {
                if !self.texture_sharpen_sample(selected.filter, selected.lambda, s, t, rr, &mut texel) {
                    return false;
                }
            } else if soft_texture_detail(selected.filter) {
                if !self.texture_detail_sample(selected.filter, selected.lambda, s, t, rr, &mut texel) {
                    return false;
                }
            } else if !self.texture_sample(level, selected.filter, s, t, rr, &mut texel) {
                return false;
            }
            if selected.alpha != 0.0 && selected.first != selected.second {
                level = pyramid.soft_texture_level(selected.second);
                if !self.texture_sample(level, selected.filter, s, t, rr, &mut other) {
                    return false;
                }
                soft_texture_mix(&mut texel, &other, selected.alpha);
            }
            let _ = base;
        } else if !self.texture_sample(None, im.min_filter, s, t, rr, &mut texel) {
            return false;
        }
        self.texture_apply(format, st.texture_env_mode, &mut texel, &mut frag.rgba);
        true
    }
}

#[derive(Clone, Copy, Default)]
pub struct SoftTexPlane {
    /// S/W, T/W, Q/W and, for volumes, R/W.
    pub value: [f64; 4],
    pub dx: [f64; 4],
    pub dy: [f64; 4],
    pub x: f64,
    pub y: f64,
}
