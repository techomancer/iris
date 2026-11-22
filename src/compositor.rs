use crate::dev::ng1::rex3::Rex3;
use crate::dev::ng1::rex3_dirty::DamageRect;
use glow::HasContext;

/// All source data needed to composite one display frame.
/// Borrowed from Rex3Screen's cached copies — no allocations per frame.
pub struct CompositorSource<'a> {
    /// REX3 colour framebuffer, 2048×1024, stride 2048 (host u32, no byte-swap)
    pub fb_rgb:           &'a [u32],
    /// REX3 auxiliary planes (overlay/popup/CID), same dimensions
    pub fb_aux:           &'a [u32],
    /// Physical framebuffer rectangles refreshed in `fb_rgb` for this frame.
    pub(crate) fb_rgb_damage: &'a [DamageRect],
    /// Physical framebuffer rectangles refreshed in `fb_aux` for this frame.
    pub(crate) fb_aux_damage: &'a [DamageRect],
    /// Decoded DID buffer in *display* coordinates, 2048×1024, stride 2048.
    /// Rex3Screen::decode_did() fills this before calling compose().
    pub did:              &'a [u8],
    /// XMAP9 mode table (32 entries × 24-bit word)
    pub xmap_mode:        &'a [u32; 32],
    pub xmap_config:      u8,
    pub xmap_cursor_cmap: u8,
    pub xmap_popup_cmap:  u8,
    /// CMAP palette (8192 entries, 0x00RRGGBB)
    pub cmap:             &'a [u32; 8192],
    /// Bt445 RAMDAC gamma LUT (256 entries, 0x00RRGGBB)
    pub ramdac:           &'a [u32; 256],
    /// VC2 RAM (cursor bitmap, DID table, video timings)
    pub vc2_ram:          &'a [u16],
    /// VC2 register file (32 × u16)
    pub vc2_regs:         &'a [u16; 32],
    /// Framebuffer row that maps to display row 0 (before the +1 the compositor applies)
    pub topscan:          usize,
    /// Cursor X hot-spot correction derived from VT timing
    pub cursor_x_adjust:  i32,
    /// Visible display dimensions decoded from VC2 video timings
    pub width:            usize,
    pub height:           usize,
    /// The screen's picture did not change: what the window shows is still
    /// the picture the last frame was made of, and only the status bar may
    /// be new. A compositor whose texture holds that picture has nothing
    /// to upload and nothing to render again.
    pub status_bar_only:  bool,
    /// A finished frame (stride 2048, `0xAABBGGRR`, R in the low byte) from a board that does its
    /// own composition (GR2). When `Some`, compositors present it verbatim and
    /// ignore every Newport field above except `width`/`height`.
    pub prebuilt:         Option<&'a [u32]>,
}

/// Whether a compositor can leave its texture as it is: the screen says
/// the picture did not change, and the texture holds the picture of that
/// size already. A compositor that has never made one (`current` false)
/// has nothing to leave.
pub(crate) fn holds_current_picture(
    src: &CompositorSource<'_>,
    current: bool,
    size: (usize, usize),
) -> bool {
    current && src.status_bar_only && size == (src.width, src.height)
}

/// Compositor: maps hardware source state to a composited GL texture.
///
/// The compositor owns its output texture.  On each call to `compose()` it
/// either uploads a CPU buffer or renders into an FBO, then returns the
/// `glow::Texture` handle.  The caller (Renderer) binds and draws it.
///
/// `compose()` is always called with `src.width > 0` and `src.height > 0`.
pub trait Compositor: Send {
    /// Produce the composited frame and return the GL texture containing it.
    /// The returned texture is owned by the compositor and valid until the
    /// next call to `compose()` or until `destroy()`.
    fn compose(&mut self, src: &CompositorSource<'_>, gl: &glow::Context) -> glow::Texture;

    /// Whether the last `compose()` changed the picture the returned
    /// texture holds. False when the compositor found the picture unchanged
    /// and left its texture as it was.
    fn picture_changed(&self) -> bool { true }

    /// Called when the display resolution changes so the compositor can resize
    /// any internal GPU resources.  Default is a no-op.
    fn resize(&mut self, _width: usize, _height: usize, _gl: &glow::Context) {}

    /// Release all GL resources.  Called before the GL context is destroyed.
    fn destroy(&mut self, _gl: &glow::Context) {}

    /// Return a CPU copy of the last composited frame for screenshots.
    /// Default: not supported (returns None).
    fn read_pixels(&self) -> Option<&[u32]> { None }

    /// Read the composited frame back to CPU memory, writing into `dst` (stride-2048,
    /// layout matching `screen.rgba`).  Only called on screenshot requests.
    /// Default no-op (SwCompositor uses `read_pixels()` instead).
    fn readback_to_screen(&self, _dst: &mut [u32], _width: usize, _height: usize, _gl: &glow::Context) {}
}

// ── Software compositor ──────────────────────────────────────────────────────

/// Pure-CPU compositor.  Runs the per-pixel loop into a `Vec<u32>`, uploads to
/// a GL texture via `glTexSubImage2D`, and returns that texture.
pub struct SwCompositor {
    /// CPU pixel buffer: stride 2048, grown to the tallest frame composed.
    buf:    Vec<u32>,
    /// GL texture that holds the last uploaded frame.
    tex:    Option<glow::Texture>,
    /// The texture holds the picture the screen shows: nothing needs
    /// composing or uploading until the picture changes.
    picture_current: bool,
    /// The size the texture was made for.
    current_size: (usize, usize),
    /// The last compose changed the picture.
    picture_changed: bool,
}

impl SwCompositor {
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            tex: None,
            picture_current: false,
            current_size: (0, 0),
            picture_changed: true,
        }
    }

    /// Raw composited pixel buffer (stride 2048, 0xFFBBGGRR: R in the low byte).
    /// Valid after `compose_pixels()`. Note: the `r_*`/`b_*` names in
    /// `compose_pixels` follow Newport's CMAP layout, whose bits 23:16 are blue.
    pub fn pixels(&self) -> &[u32] { &self.buf }

    fn ensure_tex(&mut self, gl: &glow::Context) -> glow::Texture {
        if let Some(t) = self.tex { return t; }
        let t = unsafe {
            let t = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(t));
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
            gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA as i32,
                2048, crate::disp::FB_MAX_H as i32, 0, glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelUnpackData::Slice(None));
            t
        };
        self.tex = Some(t);
        t
    }
}

impl SwCompositor {
    /// Run the per-pixel composition loop into `self.buf`. No GL calls.
    /// After this, `self.buf` holds the composited frame; call `upload_gl` to
    /// push it to the GPU, or read `self.buf` directly for CPU readback.
    pub fn compose_pixels(&mut self, src: &CompositorSource<'_>) {
        use crate::dev::ng1::vc2::{
            VC2_REG_CURRENT_CURSOR_X, VC2_REG_WORKING_CURSOR_Y, VC2_REG_CURSOR_ENTRY_PTR,
            VC2_REG_DISPLAY_CONTROL, VC2_CTRL_CURSOR_EN, VC2_CTRL_CURSOR_SIZE,
        };
        use crate::dev::ng1::rex3::ModeEntry;

        let width  = src.width;
        let height = src.height;
        let needed = crate::disp::FB_STRIDE * height.min(crate::disp::FB_MAX_H);
        if self.buf.len() < needed {
            self.buf.resize(needed, 0);
        }

        if let Some(frame) = src.prebuilt {
            for y in 0..height {
                let row = y * 2048;
                self.buf[row..row + width].copy_from_slice(&frame[row..row + width]);
            }
            return;
        }

        let cursor_x_reg   = src.vc2_regs[VC2_REG_CURRENT_CURSOR_X   as usize];
        let cursor_y_reg   = src.vc2_regs[VC2_REG_WORKING_CURSOR_Y   as usize];
        let cursor_entry   = src.vc2_regs[VC2_REG_CURSOR_ENTRY_PTR   as usize];
        let display_ctrl   = src.vc2_regs[VC2_REG_DISPLAY_CONTROL    as usize];

        let cursor_en      = (display_ctrl & VC2_CTRL_CURSOR_EN)  != 0;
        let cursor_size_64 = (display_ctrl & VC2_CTRL_CURSOR_SIZE) != 0;

        let cursor_x_hot  = (cursor_x_reg as i32) - 31 + src.cursor_x_adjust;
        let cursor_y_hot  = (cursor_y_reg as i32) - 31;

        let cursor_cmap_msb = src.xmap_cursor_cmap;
        let popup_cmap_msb  = src.xmap_popup_cmap;

        let xmap_mode    = src.xmap_mode;
        let cmap         = src.cmap;
        let ramdac       = src.ramdac;
        let fb_rgb       = src.fb_rgb;
        let fb_aux       = src.fb_aux;
        let did_buf      = src.did;
        let vc2_ram      = src.vc2_ram;
        let buf          = &mut self.buf;

        let topscan       = src.topscan + 1;

        for y in 0..height {
            let fb_y = (topscan + y) & 0x3FF;
            for x in 0..width {
                let fb_x    = x;
                let idx     = fb_y * 2048 + fb_x;
                let out_idx = y    * 2048 + x;
                let did5    = did_buf[y * 2048 + fb_x];
                let mode_entry = ModeEntry(xmap_mode[did5 as usize]);

                let pix_mode     = mode_entry.pix_mode();
                let pix_size     = mode_entry.pix_size();
                let aux_pix_mode = mode_entry.aux_pix_mode();
                let aux_msb_cmap = mode_entry.aux_msb_cmap();
                let msb_cmap     = mode_entry.msb_cmap();
                let buf_sel      = mode_entry.buf_sel();
                let ovl_buf_sel  = mode_entry.ovl_buf_sel();

                let raw_rgb = fb_rgb[idx];
                let raw_aux = fb_aux[idx];

                let pup     = (raw_aux >> 2) & 3;
                let overlay = if ovl_buf_sel { (raw_aux >> 16) & 0xFF } else { (raw_aux >> 8) & 0xFF };

                let pixel = match pix_size {
                    0 => (raw_rgb >> (if buf_sel { 4  } else { 0 })) & 0xF,
                    1 => (raw_rgb >> (if buf_sel { 8  } else { 0 })) & 0xFF,
                    2 => (raw_rgb >> (if buf_sel { 12 } else { 0 })) & 0xFFF,
                    3 => raw_rgb & 0xFFFFFF,
                    _ => 0,
                };

                // Cursor hit-test
                let mut cursor_pixel = 0u32;
                if cursor_en {
                    let cx = (x as i32) - cursor_x_hot;
                    let cy = (y as i32) - cursor_y_hot;
                    if cx >= 0 && cy >= 0 {
                        let cx = cx as usize;
                        let cy = cy as usize;
                        let shift = 15 - (cx % 16);
                        if cursor_size_64 {
                            if cx < 64 && cy < 64 {
                                let addr = cursor_entry as usize + cy * 4 + cx / 16;
                                if addr < vc2_ram.len() {
                                    cursor_pixel = ((vc2_ram[addr] >> shift) & 1) as u32;
                                }
                            }
                        } else if cx < 32 && cy < 32 {
                            let addr = cursor_entry as usize + cy * 2 + cx / 16;
                            if addr + 64 < vc2_ram.len() {
                                let bit0 = (vc2_ram[addr]      >> shift) & 1;
                                let bit1 = (vc2_ram[addr + 64] >> shift) & 1;
                                cursor_pixel = (bit0 | (bit1 << 1)) as u32;
                            }
                        }
                    }
                }

                // Priority: cursor → popup → overlay → main pixel
                let mut final_color = 0xFF000000u32;
                if cursor_pixel != 0 {
                    let addr = ((cursor_cmap_msb as usize) << 5) | (cursor_pixel as usize);
                    if addr < cmap.len() { final_color = cmap[addr] | 0xFF000000; }
                } else if pup != 0 {
                    let addr = ((popup_cmap_msb as usize) << 5) | (pup as usize);
                    if addr < cmap.len() { final_color = cmap[addr] | 0xFF000000; }
                } else if (aux_pix_mode == 2 || aux_pix_mode == 6 || aux_pix_mode == 7) && overlay != 0 {
                    let addr = ((aux_msb_cmap as usize) << 8) | (overlay as usize);
                    if addr < cmap.len() { final_color = cmap[addr] | 0xFF000000; }
                } else if pix_mode == 0 {
                    let addr = match pix_size {
                        0 | 1 => ((msb_cmap as usize) << 8)        | (pixel as usize),
                        2     => ((msb_cmap as usize & 0x10) << 8) | (pixel as usize),
                        _     =>                                       pixel as usize,
                    };
                    if addr < cmap.len() { final_color = cmap[addr] | 0xFF000000; }
                } else {
                    final_color = match pix_size {
                        0 => Rex3::expand_4_rgb(pixel)  | 0xFF000000,
                        1 => Rex3::expand_8_rgb(pixel)  | 0xFF000000,
                        2 => Rex3::expand_12_rgb(pixel) | 0xFF000000,
                        _ => pixel | 0xFF000000,
                    };
                }

                // Bt445 RAMDAC gamma correction (per-channel 8-bit LUT)
                let r_in  = ((final_color >> 16) & 0xFF) as usize;
                let g_in  = ((final_color >>  8) & 0xFF) as usize;
                let b_in  = ( final_color         & 0xFF) as usize;
                let r_out = (ramdac[r_in] >> 16) & 0xFF;
                let g_out = (ramdac[g_in] >>  8) & 0xFF;
                let b_out =  ramdac[b_in]         & 0xFF;
                buf[out_idx] = 0xFF000000 | (r_out << 16) | (g_out << 8) | b_out;
            }
        }
    }
}

impl Compositor for SwCompositor {
    fn compose(&mut self, src: &CompositorSource<'_>, gl: &glow::Context) -> glow::Texture {
        let width  = src.width;
        let height = src.height;
        // The picture did not change: the texture holds it already, and
        // composing it again would only cost a pass over every pixel and
        // an upload of the whole frame.
        if holds_current_picture(src, self.picture_current, self.current_size) {
            if let Some(tex) = self.tex {
                self.picture_changed = false;
                return tex;
            }
        }
        crate::pf_counter!(crate::perfetto::events::REX3_UPLOAD_RGB_BYTES, 0);
        crate::pf_counter!(crate::perfetto::events::REX3_UPLOAD_AUX_BYTES, 0);
        crate::pf_counter!(crate::perfetto::events::REX3_UPLOAD_RGB_RECTS, 0);
        crate::pf_counter!(crate::perfetto::events::REX3_UPLOAD_AUX_RECTS, 0);
        {
            crate::pf_scope!(crate::perfetto::events::REX3_COMPOSE);
        self.compose_pixels(src);
        }

        // Upload to GL texture
        let tex = self.ensure_tex(gl);
        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(tex));
            gl.pixel_store_i32(glow::UNPACK_ROW_LENGTH, 2048);
            let u8_slice = std::slice::from_raw_parts(
                self.buf.as_ptr() as *const u8,
                self.buf.len() * 4,
            );
            gl.tex_sub_image_2d(
                glow::TEXTURE_2D, 0,
                0, 0, width as i32, height as i32,
                glow::RGBA, glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(u8_slice)),
            );
            gl.pixel_store_i32(glow::UNPACK_ROW_LENGTH, 0);
        }
        self.picture_current = true;
        self.current_size = (width, height);
        self.picture_changed = true;
        tex
    }

    fn picture_changed(&self) -> bool { self.picture_changed }

    fn destroy(&mut self, gl: &glow::Context) {
        if let Some(t) = self.tex.take() {
            unsafe { gl.delete_texture(t); }
        }
        self.picture_current = false;
        self.current_size = (0, 0);
    }

    fn read_pixels(&self) -> Option<&[u32]> {
        Some(&self.buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A source with the fields the keep rule looks at; the rest are empty.
    fn source(width: usize, height: usize, status_bar_only: bool) -> CompositorSource<'static> {
        CompositorSource {
            fb_rgb: &[],
            fb_aux: &[],
            fb_rgb_damage: &[],
            fb_aux_damage: &[],
            did: &[],
            xmap_mode: &[0; 32],
            xmap_config: 0,
            xmap_cursor_cmap: 0,
            xmap_popup_cmap: 0,
            cmap: &[0; 8192],
            ramdac: &[0; 256],
            vc2_ram: &[],
            vc2_regs: &[0; 32],
            topscan: 0,
            cursor_x_adjust: 0,
            width,
            height,
            status_bar_only,
            prebuilt: None,
        }
    }

    /// A compositor keeps its texture only when the screen says the picture
    /// did not change, it has a picture of its own, and that picture is of
    /// the size the screen shows.
    #[test]
    fn a_picture_is_kept_only_when_it_is_the_one_on_show() {
        let src = source(1280, 1024, true);
        assert!(holds_current_picture(&src, true, (1280, 1024)));
        // Nothing composed yet: there is no picture to keep.
        assert!(!holds_current_picture(&src, false, (1280, 1024)));
        // The screen has a new picture.
        let new = source(1280, 1024, false);
        assert!(!holds_current_picture(&new, true, (1280, 1024)));
        // Another size: the texture holds a picture of another size.
        assert!(!holds_current_picture(&src, true, (1280, 768)));
    }

    #[test]
    fn the_pixel_buffer_tracks_the_tallest_active_frame() {
        let tall = vec![0xff12_3456; crate::disp::FB_STRIDE * 1200];
        let mut comp = SwCompositor::new();
        comp.compose_pixels(&CompositorSource {
            prebuilt: Some(&tall),
            ..source(1600, 1200, false)
        });
        assert_eq!(comp.pixels().len(), crate::disp::FB_STRIDE * 1200);
        assert_eq!(comp.pixels()[1199 * crate::disp::FB_STRIDE], 0xff12_3456);

        let small = vec![0xff65_4321; crate::disp::FB_STRIDE * 768];
        comp.compose_pixels(&CompositorSource {
            prebuilt: Some(&small),
            ..source(1280, 768, false)
        });
        assert_eq!(comp.pixels().len(), crate::disp::FB_STRIDE * 1200);
        assert_eq!(comp.pixels()[0], 0xff65_4321);
    }
}
