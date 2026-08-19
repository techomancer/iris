use glow::HasContext;
use crate::rex3::Rex3;

/// All source data needed to composite one display frame.
/// Borrowed from Rex3Screen's cached copies — no allocations per frame.
pub struct CompositorSource<'a> {
    /// REX3 colour framebuffer, 2048×1024, stride 2048 (host u32, no byte-swap)
    pub fb_rgb:           &'a [u32],
    /// REX3 auxiliary planes (overlay/popup/CID), same dimensions
    pub fb_aux:           &'a [u32],
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
    /// Heartbeat: only status-bar rows changed — skip full DID upload.
    pub status_bar_only:  bool,
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
    /// CPU pixel buffer: stride-2048 array, exactly 2048×1024 words.
    buf:    Vec<u32>,
    /// GL texture that holds the last uploaded frame.
    tex:    Option<glow::Texture>,
}

impl SwCompositor {
    pub fn new() -> Self {
        Self {
            buf: vec![0u32; 2048 * 1024],
            tex: None,
        }
    }

    /// Raw composited pixel buffer (stride 2048, 0xFFBBGGRR). Valid after `compose_pixels()`.
    pub fn pixels(&self) -> &[u32] { &self.buf }

    fn ensure_tex(&mut self, gl: &glow::Context) -> glow::Texture {
        if let Some(t) = self.tex { return t; }
        let t = unsafe {
            let t = gl.create_texture().unwrap();
            gl.bind_texture(glow::TEXTURE_2D, Some(t));
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
            gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA as i32,
                2048, 1024, 0, glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelUnpackData::Slice(None));
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
        use crate::vc2::{
            VC2_REG_CURRENT_CURSOR_X, VC2_REG_WORKING_CURSOR_Y, VC2_REG_CURSOR_ENTRY_PTR,
            VC2_REG_DISPLAY_CONTROL, VC2_CTRL_CURSOR_EN, VC2_CTRL_CURSOR_SIZE,
        };
        use crate::rex3::ModeEntry;

        let width  = src.width;
        let height = src.height;

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
        self.compose_pixels(src);

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
        tex
    }

    fn destroy(&mut self, gl: &glow::Context) {
        if let Some(t) = self.tex.take() {
            unsafe { gl.delete_texture(t); }
        }
    }

    fn read_pixels(&self) -> Option<&[u32]> {
        Some(&self.buf)
    }
}
