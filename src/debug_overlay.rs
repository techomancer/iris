use glow::HasContext;
use crate::rex3::{DrawRecord, DrawMode0, DrawMode1, ModeEntry,
    DRAWMODE0_OPCODE_READ, DRAWMODE0_OPCODE_DRAW, DRAWMODE0_OPCODE_SCR2SCR};
use crate::vc2::{
    VC2_REG_CURRENT_CURSOR_X, VC2_REG_WORKING_CURSOR_Y, VC2_REG_DISPLAY_CONTROL,
    VC2_CTRL_CURSOR_EN,
};

/// Inputs the debug overlay needs from the compositor source snapshot.
pub struct OverlaySource<'a> {
    pub width:            usize,
    pub height:           usize,
    pub topscan:          usize,  // pre-increment value (overlay adds 1)
    pub cursor_x_adjust:  i32,
    pub vc2_regs:         &'a [u16; 32],
    pub xmap_mode:        &'a [u32; 32],
    pub xmap_cursor_cmap: u8,
    pub xmap_popup_cmap:  u8,
    pub cmap:             &'a [u32; 8192],
    pub fb_rgb:           &'a [u32],
    pub fb_aux:           &'a [u32],
    pub did:              &'a [u8],
}

/// Unique DID→mode_entry observations recorded during composition.
/// Populated by Rex3Screen after compose() when show_disp_debug is on.
#[derive(Clone, Copy, Default)]
pub struct SeenMode {
    pub did5:       u8,
    pub raw:        u32,
    pub pix_count:  u32,
}

/// Debug overlay: renders CMAP swatches, DID/XMAP mode info, and draw-record
/// annotations into a GL texture that is alpha-blended over the compositor output.
pub struct DebugOverlay {
    /// CPU pixel buffer for the overlay (transparent=0 where inactive), stride 2048.
    buf:         Vec<u32>,
    /// GL texture backing the overlay.
    tex:         Option<glow::Texture>,

    /// VGA 8×16 font (256 glyphs × 16 rows × 1 byte).
    font:        Vec<u8>,

    // ── State set by the refresh loop each frame ──
    pub show_cmap:        bool,
    pub show_disp_debug:  bool,
    pub show_draw_debug:  bool,

    /// Unique DID→mode entries seen during last compose() pass.
    pub seen_modes:       [SeenMode; 32],
    pub seen_modes_count: usize,

    /// Snapshot of the draw ring (newest-first), valid when show_draw_debug.
    pub draw_snapshot:    Vec<DrawRecord>,
}

impl DebugOverlay {
    pub fn new() -> Self {
        Self {
            buf:              vec![0u32; 2048 * 1024],
            tex:              None,
            font:             crate::vga_font::VGA_8X16.to_vec(),
            show_cmap:        false,
            show_disp_debug:  false,
            show_draw_debug:  false,
            seen_modes:       [SeenMode::default(); 32],
            seen_modes_count: 0,
            draw_snapshot:    Vec::new(),
        }
    }

    /// True if any overlay is enabled (caller can skip render() if false).
    pub fn active(&self) -> bool {
        self.show_cmap || self.show_disp_debug || self.show_draw_debug
    }

    /// Record a DID→mode observation. Called from render()'s did[]/xmap_mode scan
    /// when show_disp_debug is on — did[] is the shared source of truth read by
    /// both compositors, so this works regardless of which one rendered the frame.
    fn record_mode(&mut self, did5: u8, raw: u32) {
        let n = self.seen_modes_count;
        if let Some(e) = self.seen_modes[..n].iter_mut().find(|e| e.did5 == did5 && e.raw == raw) {
            e.pix_count += 1;
        } else if n < 32 {
            self.seen_modes[n] = SeenMode { did5, raw, pix_count: 1 };
            self.seen_modes_count += 1;
        }
    }

    /// Render overlay into the CPU buffer and upload to the GL texture.
    /// Returns the GL texture handle.
    pub fn render(&mut self, src: &OverlaySource<'_>, gl: &glow::Context) -> glow::Texture {
        let width  = src.width;
        let height = src.height;

        let display_ctrl = src.vc2_regs[VC2_REG_DISPLAY_CONTROL as usize];
        let cursor_en    = (display_ctrl & VC2_CTRL_CURSOR_EN) != 0;
        let cursor_x_reg = src.vc2_regs[VC2_REG_CURRENT_CURSOR_X   as usize];
        let cursor_y_reg = src.vc2_regs[VC2_REG_WORKING_CURSOR_Y   as usize];
        let cursor_x_hot = (cursor_x_reg as i32) - 31 + src.cursor_x_adjust;
        let cursor_y_hot = (cursor_y_reg as i32) - 31;
        let cursor_cmap_msb = src.xmap_cursor_cmap;
        let popup_cmap_msb  = src.xmap_popup_cmap;
        let _ = (cursor_en, cursor_cmap_msb, popup_cmap_msb);

        // Clear overlay (display area only)
        for row in 0..height {
            let base = row * 2048;
            let end  = base + width;
            if end <= self.buf.len() {
                self.buf[base..end].fill(0);
            }
        }

        // ── CMAP swatch grid (128 cols × 64 rows of 8×8 px swatches = 8192 entries) ──
        if self.show_cmap {
            const COLS: usize = 128;
            const SWATCH: usize = 8;
            let cmap   = src.cmap;
            let overlay = &mut self.buf;
            for entry in 0..8192usize {
                let col   = entry % COLS;
                let row   = entry / COLS;
                let color = cmap[entry] | 0xFF000000;
                let ox = col * SWATCH;
                let oy = row * SWATCH;
                for dy in 0..SWATCH {
                    let py = oy + dy;
                    if py >= height { break; }
                    let row_base = py * 2048;
                    for dx in 0..SWATCH {
                        let px = ox + dx;
                        if px < width { overlay[row_base + px] = color; }
                    }
                }
            }
        }

        // ── DID/XMAP mode table overlay ──────────────────────────────────────────
        if self.show_disp_debug && height > 0 {
            // did[] is the shared source of truth (both SwCompositor and GlCompositor's
            // shader read from it) — classify on-screen DIDs against xmap_mode directly
            // here instead of hooking per-pixel compose paths, so this works regardless
            // of which compositor rendered the frame.
            self.seen_modes_count = 0;
            for y in 0..height {
                let row_base = y * 2048;
                for x in 0..width {
                    let did5 = src.did[row_base + x];
                    let raw = src.xmap_mode[did5 as usize];
                    self.record_mode(did5, raw);
                }
            }

            let topscan = src.topscan + 1;
            let cx = cursor_x_hot.clamp(0, width as i32 - 1) as usize;
            let cy = cursor_y_hot.clamp(0, height as i32 - 1) as usize;
            let did_idx = cy * 2048 + cx;
            let fb_cy   = (topscan + cy) & 0x3FF;
            let fb_idx  = fb_cy * 2048 + cx;

            let cursor_did5 = if did_idx < src.did.len() {
                src.did[did_idx] as u32
            } else { 0xFF };
            let raw_rgb = if fb_idx < src.fb_rgb.len() { src.fb_rgb[fb_idx] } else { 0 };
            let raw_aux = if fb_idx < src.fb_aux.len() { src.fb_aux[fb_idx] } else { 0 };

            let mut lines: Vec<String> = Vec::new();
            lines.push(format!(
                "cursor_cmap_msb={:02x} (cmap[{:04x}..]) popup_cmap_msb={:02x} (cmap[{:04x}..])  RGB:{:06x} AUX:{:06x}",
                cursor_cmap_msb, (cursor_cmap_msb as usize) << 5,
                popup_cmap_msb,  (popup_cmap_msb  as usize) << 5,
                raw_rgb & 0xFFFFFF, raw_aux & 0xFFFFFF,
            ));
            lines.push("DID  raw      pix_mode pix_size msb cmap_base aux_mode aux_msb aux_base b o  pixels".to_string());

            let pix_mode_name = |m: u32| match m { 0=>"CI", 1=>"RGB0", 2=>"RGB1", 3=>"RGB2", _=>"?" };
            let pix_size_name = |s: u32| match s { 0=>"4bpp", 1=>"8bpp", 2=>"12bpp", 3=>"24bpp", _=>"?" };

            let n = self.seen_modes_count;
            let mut highlight_line: Option<usize> = None;
            for i in 0..n {
                let sm  = self.seen_modes[i];
                let me  = ModeEntry(sm.raw);
                let pix_mode   = me.pix_mode();
                let pix_size   = me.pix_size();
                let msb_cmap   = me.msb_cmap();
                let aux_pix    = me.aux_pix_mode();
                let aux_msb    = me.aux_msb_cmap();
                let buf_sel    = me.buf_sel();
                let ovl_sel    = me.ovl_buf_sel();
                let cmap_base  = if pix_mode == 0 {
                    if pix_size == 2 { (msb_cmap as usize & 0x10) << 8 }
                    else             { (msb_cmap as usize) << 8 }
                } else { 0 };
                let aux_base = (aux_msb as usize) << 8;
                if sm.did5 as u32 == cursor_did5 { highlight_line = Some(lines.len()); }
                lines.push(format!(
                    "{:3}  {:06x}   {:<5}    {:<6}  {:02x}  {:04x}     {:x}        {:02x}     {:04x}    {} {}  {}",
                    sm.did5, sm.raw,
                    pix_mode_name(pix_mode), pix_size_name(pix_size),
                    msb_cmap, cmap_base, aux_pix, aux_msb, aux_base,
                    buf_sel as u8, ovl_sel as u8, sm.pix_count,
                ));
            }

            let line_h = 16usize;
            let total_h = lines.len() * line_h;
            let start_y = if height > total_h + 4 { height - total_h - 4 } else { 0 };
            let bg = 0xC0101010u32;
            for row in 0..total_h {
                let py = start_y + row;
                if py >= height { break; }
                let base = py * 2048;
                for px in 0..width.min(840) {
                    if base + px < self.buf.len() { self.buf[base + px] = bg; }
                }
            }
            for (li, line) in lines.iter().enumerate() {
                let y0 = start_y + li * line_h;
                let fg = if highlight_line == Some(li) { 0xFF00FFFF } else { 0xFF00FF00 };
                self.draw_text_line(width, y0, height, line, fg, bg);
            }
        }

        // ── Draw-record debug overlay ─────────────────────────────────────────────
        if self.show_draw_debug && height > 0 {
            let topscan = src.topscan + 1;
            let cx = cursor_x_hot.clamp(0, width as i32 - 1);
            let cy = cursor_y_hot.clamp(0, height as i32 - 1);

            let hits: Vec<DrawRecord> = self.draw_snapshot.iter()
                .filter(|r| {
                    let (x0, x1) = if r.x0 <= r.x1 { (r.x0 as i32, r.x1 as i32) } else { (r.x1 as i32, r.x0 as i32) };
                    let (y0, y1) = if r.y0 <= r.y1 { (r.y0 as i32, r.y1 as i32) } else { (r.y1 as i32, r.y0 as i32) };
                    cx >= x0 && cx <= x1 && cy >= y0 && cy <= y1
                })
                .take(3)
                .copied()
                .collect();

            if !hits.is_empty() {
                // Red frame around newest hit rect
                let r0 = &hits[0];
                let fx0 = (r0.x0 as i32).min(r0.x1 as i32).clamp(0, width as i32 - 1) as usize;
                let fy0 = (r0.y0 as i32).min(r0.y1 as i32).clamp(0, height as i32 - 1) as usize;
                let fx1 = (r0.x0 as i32).max(r0.x1 as i32).clamp(0, width as i32 - 1) as usize;
                let fy1 = (r0.y0 as i32).max(r0.y1 as i32).clamp(0, height as i32 - 1) as usize;
                let red = 0xFF0000FFu32;
                for x in fx0..=fx1 {
                    if fy0 * 2048 + x < self.buf.len() { self.buf[fy0 * 2048 + x] = red; }
                    if fy1 * 2048 + x < self.buf.len() { self.buf[fy1 * 2048 + x] = red; }
                }
                for y in fy0..=fy1 {
                    if y * 2048 + fx0 < self.buf.len() { self.buf[y * 2048 + fx0] = red; }
                    if y * 2048 + fx1 < self.buf.len() { self.buf[y * 2048 + fx1] = red; }
                }

                let op_str  = |dm0: u32| match DrawMode0(dm0).opcode() {
                    DRAWMODE0_OPCODE_READ    => "READ",
                    DRAWMODE0_OPCODE_DRAW    => "DRAW",
                    DRAWMODE0_OPCODE_SCR2SCR => "S2S",
                    _                        => "NOP",
                };
                let adr_str = |dm0: u32| match DrawMode0(dm0).adrmode() {
                    0=>"SPAN",1=>"BLK",2=>"ILINE",3=>"FLINE",4=>"ALINE",_=>"?"
                };
                let pln_str = |dm1: u32| match DrawMode1(dm1).planes() {
                    crate::rex3::DRAWMODE1_PLANES_RGB  => "RGB",
                    crate::rex3::DRAWMODE1_PLANES_RGBA => "RGBA",
                    crate::rex3::DRAWMODE1_PLANES_OLAY => "OLAY",
                    crate::rex3::DRAWMODE1_PLANES_PUP  => "PUP",
                    crate::rex3::DRAWMODE1_PLANES_CID  => "CID",
                    _                                   => "NONE",
                };
                let lop_str = |dm1: u32| match DrawMode1(dm1).logicop() {
                    0=>"ZERO",1=>"AND",2=>"ANDR",3=>"SRC",4=>"ANDI",5=>"DST",6=>"XOR",7=>"OR",
                    8=>"NOR",9=>"XNOR",10=>"NDST",11=>"ORR",12=>"NSRC",13=>"ORI",14=>"NAND",_=>"ONE",
                };
                let dep_str = |dm1: u32| match DrawMode1(dm1).drawdepth() {
                    0=>"4b",1=>"8b",2=>"12b",_=>"24b"
                };

                let topscan_val = topscan;
                let fb_cy_dd = (topscan_val + cy as usize) & 0x3FF;
                let fb_idx_dd = fb_cy_dd * 2048 + cx as usize;
                let dd_raw_rgb = if fb_idx_dd < src.fb_rgb.len() { src.fb_rgb[fb_idx_dd] } else { 0 };
                let dd_raw_aux = if fb_idx_dd < src.fb_aux.len() { src.fb_aux[fb_idx_dd] } else { 0 };

                let mut lines: Vec<(String, u32)> = Vec::new();
                lines.push((format!(
                    "cursor=({},{})  RGB:{:06x}  AUX:{:06x} [ovl0={:02x} ovl1={:02x} pup0={} pup1={} cid={}]",
                    cx, cy,
                    dd_raw_rgb & 0xFFFFFF, dd_raw_aux & 0xFFFFFF,
                    (dd_raw_aux >> 8) & 0xFF, (dd_raw_aux >> 16) & 0xFF,
                    (dd_raw_aux >> 2) & 3, (dd_raw_aux >> 6) & 3,
                    dd_raw_aux & 3,
                ), 0xFF));

                for (age, r) in hits.iter().enumerate() {
                    let dm0       = DrawMode0(r.dm0);
                    let dm1       = DrawMode1(r.dm1);
                    let colorhost = dm0.colorhost();
                    let alphahost = dm0.alphahost();
                    let hostrw_info = if colorhost || alphahost {
                        let label = match (colorhost, alphahost) {
                            (true,  true)  => "CH+AH",
                            (true,  false) => "CH",
                            (false, true)  => "AH",
                            _              => "",
                        };
                        let dbl  = dm1.rwdouble();
                        let pk   = dm1.rwpacked();
                        let flags = match (dbl, pk) {
                            (true,  true)  => " DBL+PK",
                            (true,  false) => " DBL",
                            (false, true)  => " PK",
                            (false, false) => "",
                        };
                        let exp = if dbl { r.expected_doubles } else { r.expected_words };
                        let ok  = r.hostrw_writes == exp;
                        format!(" {}{}: {}/{}{}", label, flags, r.hostrw_writes, exp,
                            if ok { "" } else { " MISMATCH" })
                    } else if r.spurious_writes > 0 {
                        format!(" SPURIOUS:{}", r.spurious_writes)
                    } else { String::new() };

                    let is_s2s   = DrawMode0(r.dm0).opcode() == DRAWMODE0_OPCODE_SCR2SCR;
                    let src_info = if is_s2s {
                        format!(" src=({},{})→({},{})", r.sx0, r.sy0, r.sx1, r.sy1)
                    } else { String::new() };

                    lines.push((format!(
                        "#{} dst=({},{})→({},{}) {}x{}  {} {} {} {} logicop={}  wrmask={:06x}{}{}",
                        age + 1,
                        r.x0, r.y0, r.x1, r.y1,
                        (r.x1 as i32 - r.x0 as i32).unsigned_abs() + 1,
                        (r.y1 as i32 - r.y0 as i32).unsigned_abs() + 1,
                        pln_str(r.dm1), op_str(r.dm0), adr_str(r.dm0), dep_str(r.dm1), lop_str(r.dm1),
                        r.wrmask, src_info, hostrw_info,
                    ), age as u32));

                    let has_lspat = dm0.enlspattern();
                    let has_zpat  = dm0.enzpattern();
                    let mut pat_info = String::new();
                    if has_lspat {
                        let tag = if dm0.lsopaque() { "LO" } else { "L" };
                        pat_info.push_str(&format!(" {}={:08x}", tag, r.lspat));
                    }
                    if has_zpat {
                        let tag = if dm0.zpopaque() { "ZO" } else { "Z" };
                        pat_info.push_str(&format!(" {}={:08x}", tag, r.zpat));
                    }
                    lines.push((format!(
                        "   colori={:08x} back={:08x} DM0={:08x} DM1={:08x}{}",
                        r.colori, r.colorback, r.dm0, r.dm1, pat_info,
                    ), age as u32));
                }

                let line_h  = 16usize;
                let total_h = lines.len() * line_h;
                let disp_debug_h = if self.show_disp_debug {
                    (2 + self.seen_modes_count) * line_h + 4
                } else { 0 };
                let start_y = if height > total_h + disp_debug_h + 4 {
                    height - total_h - disp_debug_h - 4
                } else { 0 };
                let bg = 0xC0101828u32;
                for row in 0..total_h {
                    let py = start_y + row;
                    if py >= height { break; }
                    let base = py * 2048;
                    for px in 0..width.min(960) {
                        if base + px < self.buf.len() { self.buf[base + px] = bg; }
                    }
                }
                let fgs = [0xFF00FFFF_u32, 0xFF00D8FFu32, 0xFF00B8D8u32];
                for (li, (line, age)) in lines.iter().enumerate() {
                    let y0 = start_y + li * line_h;
                    let fg = if *age == 0xFF { 0xFFFFFFFF_u32 } else { fgs[(*age as usize).min(fgs.len() - 1)] };
                    self.draw_text_line(width, y0, height, line, fg, bg);
                }
            }
        }

        // Upload to GL texture and return handle
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
                0, 0, src.width as i32, src.height as i32,
                glow::RGBA, glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(u8_slice)),
            );
            gl.pixel_store_i32(glow::UNPACK_ROW_LENGTH, 0);
        }
        tex
    }

    pub fn destroy(&mut self, gl: &glow::Context) {
        if let Some(t) = self.tex.take() {
            unsafe { gl.delete_texture(t); }
        }
    }

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

    /// Draw a line of text into `self.buf` using the VGA 8×16 font.
    fn draw_text_line(&mut self, width: usize, y0: usize, height: usize, line: &str, fg: u32, bg: u32) {
        let font_ref: &[u8] = &self.font;
        let w = width;
        let mut tx = 4usize;
        for ch in line.chars() {
            if tx + 8 > w { break; }
            let glyph = ch as usize & 0xFF;
            let goff  = glyph * 16;
            for row in 0..16usize {
                let bi  = goff + row;
                if bi >= font_ref.len() { break; }
                let bits = font_ref[bi];
                let py   = y0 + row;
                if py >= height { break; }
                let rb   = py * 2048;
                for col in 0..8usize {
                    let px  = tx + col;
                    if px >= w { break; }
                    let idx = rb + px;
                    if idx < self.buf.len() {
                        self.buf[idx] = if (bits >> (7 - col)) & 1 != 0 { fg } else { bg };
                    }
                }
            }
            tx += 8;
        }
    }
}
