use std::sync::atomic::{AtomicU64, Ordering};
use parking_lot::Mutex;
use crate::vc2::{Vc2, VC2_REG_DISPLAY_CONTROL, VC2_CTRL_DID_EN, VC2_REG_DID_ENTRY_PTR, VC2_REG_SCANLINE_LEN, VC2_REG_CURRENT_CURSOR_X, VC2_REG_WORKING_CURSOR_Y, VC2_REG_CURSOR_ENTRY_PTR, VC2_CTRL_CURSOR_EN, VC2_CTRL_CURSOR_SIZE, VC2_REG_VIDEO_ENTRY_PTR, VC2_CTRL_BLACKOUT, VC2_CTRL_VIDEO_TIMING_EN, VT_VIS_LN_VC_N, VT_DSPLY_EN_RO_N, VT_CBLANK_XMAP_N, VT_HPOS_VC_N};
use crate::xmap9::Xmap9;
use crate::cmap::Cmap;
use crate::bt445::Bt445;
use crate::rex3::{Renderer, ModeEntry, Rex3, DrawRecord, DrawMode0, DrawMode1,
    DRAWMODE0_OPCODE_READ, DRAWMODE0_OPCODE_DRAW, DRAWMODE0_OPCODE_SCR2SCR,
    DRAWMODE1_PLANES_RGB, DRAWMODE1_PLANES_RGBA, DRAWMODE1_PLANES_OLAY, DRAWMODE1_PLANES_PUP, DRAWMODE1_PLANES_CID};

pub struct Rex3Screen {
    pub width: usize,
    pub height: usize,
    pub fb_rgb: Vec<u32>,
    pub fb_aux: Vec<u32>,
    pub did: Vec<u16>,
    /// Main framebuffer output — exactly `width × height`, row stride 2048.
    /// Contains only the emulated pixel data with no overlays.
    pub rgba: Vec<u32>,
    /// Debug overlay buffer — `width × height`, row stride 2048.
    /// Transparent (0) where no overlay; alpha-composited over `rgba` by the UI.
    pub overlay_rgba: Vec<u32>,
    /// Status bar buffer — `width × STATUS_BAR_HEIGHT`, row stride 2048.
    /// Rendered as a separate opaque texture at the bottom of the window.
    pub statusbar_rgba: Vec<u32>,

    // VC2 State
    pub vc2_ram: Vec<u16>,
    pub vc2_regs: [u16; 32],

    // CMAP
    pub cmap: [u32; 8192],

    // BT445 RAMDAC gamma palette (256 × 0x00RRGGBB, copied from Bt445 on dirty)
    pub ramdac_palette: [u32; 256],

    // XMAP
    pub xmap_mode: [u32; 32],
    pub xmap_config: u8,
    pub xmap_cursor_cmap: u8,
    pub xmap_popup_cmap: u8,

    // TOPSCAN: framebuffer row that maps to display row 0
    pub topscan: usize,

    // Cursor X correction: added to (cursor_x_reg - 31) to align hot spot with display pixels.
    // Derived from VT timing: hpos_to_visible_delta - 31 (VC2 doc: HPOS asserts 31px before
    // first visible cursor pixel).
    pub cursor_x_adjust: i32,

    // Debug: overlay all 8192 CMAP entries as 8x8 swatches in a 128x64 grid
    pub show_cmap: bool,

    // Debug: overlay decoded DID/XMAP mode info near bottom of screen
    pub show_disp_debug: bool,
    // Debug: overlay draw-record info for draws under cursor
    pub show_draw_debug: bool,
    pub draw_snapshot: Vec<DrawRecord>, // newest-first snapshot of DrawRingBuf
    // Unique DID→mode_entry values seen during last frame (DID index, raw 24-bit mode entry, pixel count)
    pub seen_modes: [(u8, u32, u32); 32],
    pub seen_modes_count: usize,
    // VGA 8x16 font for debug overlays
    pub debug_font: Vec<u8>,
}

impl Rex3Screen {
    fn decode_did(&mut self) {
        // Check DID Enable (Bit 3 of Display Control)
        let display_ctrl = self.vc2_regs[VC2_REG_DISPLAY_CONTROL as usize];
        if (display_ctrl & VC2_CTRL_DID_EN) == 0 {
            self.did.fill(0);
            return;
        }

        let did_ptr = self.vc2_regs[VC2_REG_DID_ENTRY_PTR as usize];
        let scan_len = (self.vc2_regs[VC2_REG_SCANLINE_LEN as usize] >> 5) as usize;
        
        let width = self.width;
        let height = self.height;
        let ram = &self.vc2_ram;
        let did_buf = &mut self.did;

        // Use scan_len directly; cap only against the buffer row stride (2048).
        // Do NOT cap at display width — DID may extend beyond display width to cover
        // fb columns shifted by fb_x_offset (e.g. scan_len=1028, display width=1024).
        let effective_len = if scan_len > 0 && scan_len <= 2048 { scan_len } else { width };

        let mut y = 0;
        let mut table_idx = did_ptr as usize;

        while y < height {
            if table_idx >= ram.len() { break; }
            
            let line_ptr = ram[table_idx];
            
            // Check for end of table marker
            if line_ptr == 0xFFFF { break; }
            
            let mut ptr = line_ptr as usize;

            // Read initial entry for the line
            if ptr >= ram.len() { 
                // No data for this line, fill with 0
                let start_idx = y * 2048;
                let end_idx = y * 2048 + effective_len;
                if end_idx <= did_buf.len() {
                    did_buf[start_idx..end_idx].fill(0);
                }
                y += 1; 
                table_idx += 1; 
                continue; 
            }

            let entry = ram[ptr];
            ptr += 1;

            let mut current_did = (entry & 0x1F) as u16;
            let mut current_x = 0; 
            
            // Process scanline RLE data
            loop {
                if current_x >= effective_len { break; }

                if ptr >= ram.len() { 
                    // End of RAM, fill rest of line
                    let start_idx = y * 2048 + current_x;
                    let end_idx = y * 2048 + effective_len;
                    if end_idx <= did_buf.len() {
                        did_buf[start_idx..end_idx].fill(current_did);
                    }
                    break; 
                }
                
                let next_entry = ram[ptr];
                ptr += 1;
                
                let next_x_raw = ((next_entry >> 5) & 0x7FF) as usize;
                let next_did = (next_entry & 0x1F) as u16;
                
                let is_eol = next_x_raw == 0x7FF;
                let next_x = if is_eol { effective_len } else { next_x_raw };
                let run_end = next_x.max(current_x).min(effective_len);
                
                if run_end > current_x {
                    let start_idx = y * 2048 + current_x;
                    let end_idx = y * 2048 + run_end;
                    
                    if end_idx <= did_buf.len() {
                        did_buf[start_idx..end_idx].fill(current_did);
                    }
                }
                
                current_x = run_end;
                current_did = next_did;
                
                if is_eol { break; }
            }
            
            y += 1;
            table_idx += 1;
        }
    }

    // Returns (width, height, cursor_x_adjust).
    // cursor_x_adjust is added to (cursor_x_reg - 31) to align the hot spot with display pixels.
    //
    // The VC2 doc states: "the assertion of HPOS must precede the first visible pixel cursor
    // by 31 pixels".  So the cursor register counts pixels from HPOS assertion, and display
    // column 0 is reached at register value (hpos_to_visible_delta).  The formula
    //   cursor_x_hot = cursor_x_reg - 31 + cursor_x_adjust
    // means cursor_x_adjust = hpos_to_visible_delta - 31.
    // We derive hpos_to_visible_delta by walking the first visible line's VT entries and
    // measuring the pixel distance from HPOS assertion to the first visible pixel.
    fn decode_video_timings(&self) -> (usize, usize, i32) {
        let frame_ptr = self.vc2_regs[VC2_REG_VIDEO_ENTRY_PTR as usize] as usize;
        let ram = &self.vc2_ram;

        let mut max_visible_width = 0usize;
        let mut total_visible_lines = 0usize;
        // Pixel distance from HPOS assertion to visible area start, from first visible line.
        let mut hpos_to_visible: Option<usize> = None;

        let mut curr_frame_ptr = frame_ptr;
        let mut loop_safety = 0usize;

        loop {
            if curr_frame_ptr + 1 >= ram.len() { break; }
            let line_seq_ptr = ram[curr_frame_ptr] as usize;
            let mut line_seq_len = ram[curr_frame_ptr + 1] as usize;
            if line_seq_len == 0 { break; }

            let mut curr_line_ptr = line_seq_ptr;

            while line_seq_len > 0 {
                let mut line_visible_width = 0usize;
                let mut eol = false;
                let mut line_loop_safety = 0usize;
                let mut state_c = 0u8;
                // Track HPOS→visible delta for this line.
                // VT_HPOS_VC_N is active-low: bit CLEAR = asserted (pin driven low).
                // We want the leading edge: first entry where bit is clear, after seeing it set.
                let mut pixel_offset = 0usize;
                let mut hpos_pixel: Option<usize> = None;
                let mut visible_pixel: Option<usize> = None;
                let mut hpos_seen_deasserted = false;

                while !eol {
                    if curr_line_ptr >= ram.len() { break; }
                    let w1 = ram[curr_line_ptr];
                    curr_line_ptr += 1;

                    let duration = ((w1 >> 8) & 0x7F) as usize;
                    let state_a = (w1 & 0x7F) as u8;
                    let sb_sc_absent = (w1 & 0x0080) != 0;
                    let mut eol_bit = (w1 & 0x8000) != 0;

                    if !sb_sc_absent {
                        if curr_line_ptr >= ram.len() { break; }
                        let w2 = ram[curr_line_ptr];
                        if (w2 & 0x8000) != 0 { eol_bit = true; }
                        curr_line_ptr += 1;
                        let _state_b = ((w2 >> 8) & 0x7F) as u8;
                        state_c = (w2 & 0x7F) as u8;
                    }

                    eol = eol_bit;
                    let pixels = duration * 2;

                    // HPOS is active-low (_N): bit set = deasserted, bit clear = asserted.
                    // Latch leading edge: first entry with bit clear after seeing it set.
                    if hpos_pixel.is_none() {
                        if (state_a & VT_HPOS_VC_N) != 0 {
                            hpos_seen_deasserted = true;
                        } else if hpos_seen_deasserted {
                            hpos_pixel = Some(pixel_offset);
                        }
                    }

                    let visible = (state_c & VT_CBLANK_XMAP_N) != 0 && (state_a & VT_VIS_LN_VC_N) == 0 && (state_a & VT_DSPLY_EN_RO_N) == 0;

                    if visible {
                        if visible_pixel.is_none() {
                            visible_pixel = Some(pixel_offset);
                        }
                        line_visible_width += pixels;
                    }

                    pixel_offset += pixels;
                    line_loop_safety += 1;
                    if line_loop_safety > 1000 { break; }
                }

                if line_visible_width > 0 {
                    total_visible_lines += 1;
                    if line_visible_width > max_visible_width {
                        max_visible_width = line_visible_width;
                    }
                    // Latch HPOS→visible delta from the first visible line where both are seen.
                    if hpos_to_visible.is_none() {
                        if let (Some(h), Some(v)) = (hpos_pixel, visible_pixel) {
                            if v >= h {
                                hpos_to_visible = Some(v - h);
                            }
                        }
                    }
                }

                line_seq_len -= 1;
                if curr_line_ptr >= ram.len() { break; }
                curr_line_ptr = ram[curr_line_ptr] as usize;
            }

            curr_frame_ptr += 2;
            loop_safety += 1;
            if loop_safety > 1000 { break; }
        }

        if max_visible_width > 0 && total_visible_lines > 0 {
            let w = max_visible_width.min(2048);
            let h = total_visible_lines.min(1024);
            // cursor_x_adjust = hpos_to_visible_delta - 31.
            // The VC2 doc says HPOS asserts 31 pixels before the first visible cursor pixel,
            // so the cursor register value at display column 0 is hpos_to_visible_delta,
            // and subtracting 31 (already in the hot-spot formula) leaves the remainder.
            // Sanity check: delta must be < 31 (HPOS must precede visible by exactly 31px
            // per spec); if it's >= 31 the edge detection misfired — fall back to empirical 11.
            let cursor_x_adjust = match hpos_to_visible {
                Some(d) => {
                    let adj = d as i32 - 31;
                    if adj < 0 || adj > 64 {
                        println!("Rex3: WARNING: hpos_to_visible={} gives cursor_x_adjust={}, out of range, falling back to 11", d, adj);
                        11
                    } else {
                        adj
                    }
                }
                None => {
                    println!("Rex3: WARNING: HPOS leading edge not found in VT, falling back to cursor_x_adjust=11");
                    11
                }
            };
            (w, h, cursor_x_adjust)
        } else {
            (0, 0, 0)
        }
    }

    /// Refresh the rgba buffer from the framebuffer/VC2/CMAP/XMAP state.
    /// Returns `true` if the display resolution changed (caller should resize the window).
    /// The caller is responsible for calling `renderer.render()` afterward.
    pub fn refresh(&mut self, fb_rgb: &[u32], fb_aux: &[u32], vc2: &Mutex<Vc2>, xmap: &Mutex<Xmap9>, cmap: &Mutex<Cmap>, bt445: &Mutex<Bt445>, renderer: &mut Option<Box<dyn Renderer>>, diag: &AtomicU64) -> bool {
        let mut resized = false;
        // 1. Copy State
        {
            // Copy Framebuffers
            diag.fetch_or(Rex3::DIAG_LOOP_FB_COPY, Ordering::Relaxed);
            self.fb_rgb.copy_from_slice(fb_rgb);
            self.fb_aux.copy_from_slice(fb_aux);
            diag.fetch_and(!Rex3::DIAG_LOOP_FB_COPY, Ordering::Relaxed);

            // Copy VC2
            diag.fetch_or(Rex3::DIAG_LOCK_VC2 | Rex3::DIAG_LOOP_VC2_COPY, Ordering::Relaxed);
            let mut vc2 = vc2.lock();
            if vc2.dirty {
                self.vc2_ram.copy_from_slice(&vc2.ram);
                self.vc2_regs.copy_from_slice(&vc2.regs);
                vc2.dirty = false;
            }
            drop(vc2);
            diag.fetch_and(!(Rex3::DIAG_LOCK_VC2 | Rex3::DIAG_LOOP_VC2_COPY), Ordering::Relaxed);

            // Copy CMAP
            diag.fetch_or(Rex3::DIAG_LOCK_CMAP0 | Rex3::DIAG_LOOP_CMAP_COPY, Ordering::Relaxed);
            let mut cmap = cmap.lock();
            if cmap.dirty {
                self.cmap.copy_from_slice(&cmap.palette);
                cmap.dirty = false;
            }
            drop(cmap);
            diag.fetch_and(!(Rex3::DIAG_LOCK_CMAP0 | Rex3::DIAG_LOOP_CMAP_COPY), Ordering::Relaxed);

            // Copy BT445 RAMDAC palette
            {
                let mut dac = bt445.lock();
                if dac.dirty {
                    self.ramdac_palette = dac.palette_as_rgb();
                    dac.dirty = false;
                }
                drop(dac);
            }

            // Copy XMAP
            diag.fetch_or(Rex3::DIAG_LOCK_XMAP0 | Rex3::DIAG_LOOP_XMAP_COPY, Ordering::Relaxed);
            let mut xmap = xmap.lock();
            if xmap.dirty {
                self.xmap_mode.copy_from_slice(&xmap.mode_table);
                self.xmap_config = xmap.config;
                self.xmap_cursor_cmap = xmap.cursor_cmap_msb;
                self.xmap_popup_cmap = xmap.popup_cmap_msb;
                xmap.dirty = false;
            }
            drop(xmap);
            diag.fetch_and(!(Rex3::DIAG_LOCK_XMAP0 | Rex3::DIAG_LOOP_XMAP_COPY), Ordering::Relaxed);
        }

        // 2. Processing (DID, RGBA conversion)
        let display_ctrl = self.vc2_regs[VC2_REG_DISPLAY_CONTROL as usize];
        let video_timing_en = (display_ctrl & VC2_CTRL_VIDEO_TIMING_EN) != 0;
        let display_en = (display_ctrl & VC2_CTRL_BLACKOUT) != 0;

        if !video_timing_en || !display_en {
            self.rgba.fill(0xFF000000);
            return resized;
        }

        diag.fetch_or(Rex3::DIAG_LOOP_VID_TIMINGS, Ordering::Relaxed);
        let (w, h, cursor_x_adjust) = self.decode_video_timings();
        diag.fetch_and(!Rex3::DIAG_LOOP_VID_TIMINGS, Ordering::Relaxed);
        self.cursor_x_adjust = cursor_x_adjust;
        if w > 0 && h > 0 && (w != self.width || h != self.height) {
            println!("Rex3: Resolution changed to {}x{} cursor_x_adjust={}", w, h, cursor_x_adjust);
            self.width = w;
            self.height = h;
            if let Some(renderer) = renderer {
                renderer.resize(w, h);
            }
            resized = true;
        }

        if !display_en {
            self.rgba.fill(0xFF000000);
            return resized;
        }

        diag.fetch_or(Rex3::DIAG_LOOP_DECODE_DID, Ordering::Relaxed);
        self.decode_did();
        diag.fetch_and(!Rex3::DIAG_LOOP_DECODE_DID, Ordering::Relaxed);

        // RGBA Conversion
        let width = self.width;
        let height = self.height;
        
        // VC2 Cursor Registers
        let cursor_x_reg = self.vc2_regs[VC2_REG_CURRENT_CURSOR_X as usize];
        let cursor_y_reg = self.vc2_regs[VC2_REG_WORKING_CURSOR_Y as usize];
        let cursor_entry = self.vc2_regs[VC2_REG_CURSOR_ENTRY_PTR as usize];
        let display_ctrl = self.vc2_regs[VC2_REG_DISPLAY_CONTROL as usize];
        
        let cursor_en = (display_ctrl & VC2_CTRL_CURSOR_EN) != 0;
        let cursor_size_64 = (display_ctrl & VC2_CTRL_CURSOR_SIZE) != 0;
        
        let cursor_w = if cursor_size_64 { 64 } else { 32 };
        let cursor_h = if cursor_size_64 { 64 } else { 32 };
        
        // XYWIN.x is typically 0x1002 at runtime (2 above the 0x1000 base), so IRIX
        // draws into fb columns starting at 2, leaving columns 0-1 blank.  Shift the
        // display read window right by 2 so display column 0 = fb column 2.
        // TODO: derive from XYWIN register rather than hardcoding.
        let fb_x_offset = 2i32;

        // Cursor X register is in screen coordinates (0 = first visible pixel).
        // cursor_x_adjust compensates for decoded-width overshoot vs nominal resolution.
        // Subtract fb_x_offset so the cursor tracks the shifted display window.
        let cursor_x_hot = (cursor_x_reg as i32) - 31 + self.cursor_x_adjust - fb_x_offset;
        let cursor_y_hot = (cursor_y_reg as i32) - 31;
        
        let cursor_cmap_msb = self.xmap_cursor_cmap;
        let popup_cmap_msb = self.xmap_popup_cmap;

        let xmap_mode = &self.xmap_mode;
        let cmap = &self.cmap;
        let ramdac = &self.ramdac_palette;
        let fb_rgb = &self.fb_rgb;
        let fb_aux = &self.fb_aux;
        let did_buf = &self.did;
        let vc2_ram = &self.vc2_ram;
        let rgba = &mut self.rgba;

        // Reset seen-mode tracking for this frame
        if self.show_disp_debug {
            self.seen_modes_count = 0;
        }

        let topscan = self.topscan + 1;
        let fb_x_offset_u = fb_x_offset as usize;
        diag.fetch_or(Rex3::DIAG_LOOP_PIXEL_CONV, Ordering::Relaxed);
        for y in 0..height {
            let fb_y = (topscan + y) & 0x3FF;
            for x in 0..width {
                let fb_x = x + fb_x_offset_u;
                let idx = fb_y * 2048 + fb_x;
                let out_idx = y * 2048 + x;
                let did = did_buf[y * 2048 + fb_x] as usize;
                let did5 = (did & 0x1F) as u8;
                let mode_entry = ModeEntry(xmap_mode[did5 as usize]); // 32 entries

                // Track unique DID→mode_entry pairs and pixel counts seen this frame
                if self.show_disp_debug {
                    let raw = mode_entry.0 & 0xFFFFFF;
                    let n = self.seen_modes_count;
                    if let Some(e) = self.seen_modes[..n].iter_mut().find(|e| e.0 == did5 && e.1 == raw) {
                        e.2 += 1;
                    } else if n < 32 {
                        self.seen_modes[n] = (did5, raw, 1);
                        self.seen_modes_count += 1;
                    }
                }
                
                // Decode Mode
                let pix_mode = mode_entry.pix_mode();
                let pix_size = mode_entry.pix_size();
                let aux_pix_mode = mode_entry.aux_pix_mode();
                let aux_msb_cmap = mode_entry.aux_msb_cmap();
                let msb_cmap = mode_entry.msb_cmap();
                let buf_sel = mode_entry.buf_sel();
                let ovl_buf_sel = mode_entry.ovl_buf_sel();
                
                let raw_rgb = fb_rgb[idx];
                let raw_aux = fb_aux[idx];
                
                // Extract Planes
                // PUP: Bits 3:2 (Buf0), 7:6 (Buf1) - Assuming Buf0 for now as global
                let pup = (raw_aux >> 2) & 3; 
                
                // Overlay: Bits 15:8 (Buf0), 23:16 (Buf1)
                let overlay = if ovl_buf_sel { (raw_aux >> 16) & 0xFF } else { (raw_aux >> 8) & 0xFF };
                
                // Pixel Data — extract from raw_rgb based on depth and buffer select
                let pixel = match pix_size {
                    0 => (raw_rgb >> (if buf_sel { 4 } else { 0 })) & 0xF,
                    1 => (raw_rgb >> (if buf_sel { 8 } else { 0 })) & 0xFF,
                    2 => (raw_rgb >> (if buf_sel { 12 } else { 0 })) & 0xFFF,
                    3 => raw_rgb & 0xFFFFFF,
                    _ => 0,
                };

                // Cursor Logic
                let mut cursor_pixel = 0u32;
                if cursor_en {
                    let cx = (x as i32) - cursor_x_hot;
                    let cy = (y as i32) - cursor_y_hot;
                    if cx >= 0 && cy >= 0 {
                        let cx = cx as usize;
                        let cy = cy as usize;
                        let shift = 15 - (cx % 16);
                        if cursor_size_64 {
                            // 64x64 monochrome: 4 words per line, 1 bit per pixel
                            if cx < 64 && cy < 64 {
                                let addr = cursor_entry as usize + cy * 4 + cx / 16;
                                if addr < vc2_ram.len() {
                                    cursor_pixel = ((vc2_ram[addr] >> shift) & 1) as u32;
                                }
                            }
                        } else {
                            // 32x32 two-plane: 2 words per line per plane
                            // plane0 at cursor_entry + y*2 + x/16
                            // plane1 at cursor_entry + y*2 + x/16 + 64
                            if cx < 32 && cy < 32 {
                                let addr = cursor_entry as usize + cy * 2 + cx / 16;
                                if addr + 64 < vc2_ram.len() {
                                    let bit0 = (vc2_ram[addr] >> shift) & 1;
                                    let bit1 = (vc2_ram[addr + 64] >> shift) & 1;
                                    cursor_pixel = (bit0 | (bit1 << 1)) as u32;
                                }
                            }
                        }
                    }
                }

                // Priority Selection & Color Lookup
                let mut final_color = 0xFF000000; // Alpha 255
                
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
                    // CI mode: index into CMAP. msb_cmap provides upper address bits.
                    let addr = match pix_size {
                        0 => ((msb_cmap as usize) << 8) | (pixel as usize),          // 4bpp: full msb_cmap
                        1 => ((msb_cmap as usize) << 8) | (pixel as usize),          // 8bpp: full msb_cmap
                        2 => ((msb_cmap as usize & 0x10) << 8) | (pixel as usize),   // 12bpp: only bit 4 of msb
                        3 => pixel as usize,                                           // 24bpp CI: direct index
                        _ => 0,
                    };
                    if addr < cmap.len() { final_color = cmap[addr] | 0xFF000000; }
                } else {
                    // RGB mode: expand packed pixel to 24bpp
                    final_color = match pix_size {
                        0 => Rex3::expand_4_rgb(pixel)  | 0xFF000000,
                        1 => Rex3::expand_8_rgb(pixel)  | 0xFF000000,
                        2 => Rex3::expand_12_rgb(pixel) | 0xFF000000,
                        _ => pixel | 0xFF000000,
                    };
                }
                // Apply RAMDAC gamma LUT (per-channel lookup).
                // ramdac_palette[i] = 0x00RRGGBB where R=G=B for the Bt445 on Newport
                // (SGI loads a shared gamma ramp into all three channels).
                // Each channel of final_color is used as an 8-bit index independently.
                let r_in = ((final_color >> 16) & 0xFF) as usize;
                let g_in = ((final_color >>  8) & 0xFF) as usize;
                let b_in = ( final_color        & 0xFF) as usize;
                let r_out = (ramdac[r_in] >> 16) & 0xFF;
                let g_out = (ramdac[g_in] >>  8) & 0xFF;
                let b_out =  ramdac[b_in]         & 0xFF;
                rgba[out_idx] = 0xFF000000 | (r_out << 16) | (g_out << 8) | b_out;
            }
        }
        diag.fetch_and(!Rex3::DIAG_LOOP_PIXEL_CONV, Ordering::Relaxed);

        resized
    }

    /// Render all overlays (debug info, status bar) into `self.overlay_rgba`.
    /// The overlay buffer uses the same 2048-pixel row stride as `rgba`.
    /// Transparent pixels (alpha=0) show through to the main framebuffer in the UI.
    /// The status bar occupies rows `height .. height + STATUS_BAR_HEIGHT` of the overlay.
    pub fn render_overlay(&mut self) {
        let width  = self.width;
        let height = self.height;
        let topscan = self.topscan + 1;

        // Cached cursor position from last refresh() call.
        let cursor_x_reg = self.vc2_regs[VC2_REG_CURRENT_CURSOR_X as usize];
        let cursor_y_reg = self.vc2_regs[VC2_REG_WORKING_CURSOR_Y as usize];
        let cursor_x_hot = (cursor_x_reg as i32) - 31 + self.cursor_x_adjust - 2;
        let cursor_y_hot = (cursor_y_reg as i32) - 31;
        let cursor_cmap_msb = self.xmap_cursor_cmap;
        let popup_cmap_msb  = self.xmap_popup_cmap;

        // Clear overlay to fully transparent (main framebuffer area only).
        for row in 0..height {
            let base = row * 2048;
            let end = base + width;
            if end <= self.overlay_rgba.len() {
                self.overlay_rgba[base..end].fill(0);
            }
        }

        // CMAP overlay: 128×64 grid of 8×8 swatches = 8192 CMAP entries, 1024×512 px total.
        if self.show_cmap {
            const COLS: usize = 128;
            const SWATCH: usize = 8;
            let cmap = &self.cmap;
            let overlay = &mut self.overlay_rgba;
            for entry in 0..8192usize {
                let col = entry % COLS;
                let row = entry / COLS;
                let color = cmap[entry] | 0xFF000000;
                let ox = col * SWATCH;
                let oy = row * SWATCH;
                for dy in 0..SWATCH {
                    let py = oy + dy;
                    if py >= height { break; }
                    let row_base = py * 2048;
                    for dx in 0..SWATCH {
                        let px = ox + dx;
                        if px < width {
                            overlay[row_base + px] = color;
                        }
                    }
                }
            }
        }

        // Display debug overlay: decoded DID/XMAP modes near the bottom of the screen.
        if self.show_disp_debug && height > 0 {
            // Find the DID and framebuffer values under the cursor hotspot.
            let cx = cursor_x_hot.clamp(0, width as i32 - 1) as usize;
            let cy = cursor_y_hot.clamp(0, height as i32 - 1) as usize;
            // DID buffer is in display coords (out_idx = y*2048+x); fb buffers use topscan-rotated fb_y.
            let did_idx = cy * 2048 + cx;
            let fb_cy = (topscan + cy) & 0x3FF;
            let fb_idx = fb_cy * 2048 + cx;
            let cursor_did5 = if did_idx < self.did.len() {
                (self.did[did_idx] as u32) & 0x1F
            } else { 0xFF };
            let raw_rgb = if fb_idx < self.fb_rgb.len() { self.fb_rgb[fb_idx] } else { 0 };
            let raw_aux = if fb_idx < self.fb_aux.len() { self.fb_aux[fb_idx] } else { 0 };

            // Build lines of text to render
            let mut lines: Vec<String> = Vec::new();

            // Cursor and popup CMAP pointers + raw buffer values under cursor
            lines.push(format!(
                "cursor_cmap_msb={:02x} (cmap[{:04x}..]) popup_cmap_msb={:02x} (cmap[{:04x}..])  RGB:{:06x} AUX:{:06x}",
                cursor_cmap_msb, (cursor_cmap_msb as usize) << 5,
                popup_cmap_msb,  (popup_cmap_msb  as usize) << 5,
                raw_rgb & 0xFFFFFF, raw_aux & 0xFFFFFF,
            ));

            // Column header
            lines.push("DID  raw      pix_mode pix_size msb cmap_base aux_mode aux_msb aux_base b o  pixels".to_string());

            let pix_mode_name = |m: u32| match m { 0=>"CI", 1=>"RGB0", 2=>"RGB1", 3=>"RGB2", _=>"?" };
            let pix_size_name = |s: u32| match s { 0=>"4bpp", 1=>"8bpp", 2=>"12bpp", 3=>"24bpp", _=>"?" };

            let n = self.seen_modes_count;
            let mut highlight_line: Option<usize> = None; // line index of cursor DID
            for i in 0..n {
                let (did5, raw, pix_count) = self.seen_modes[i];
                let me = ModeEntry(raw);
                let pix_mode   = me.pix_mode();
                let pix_size   = me.pix_size();
                let msb_cmap   = me.msb_cmap();
                let aux_pix    = me.aux_pix_mode();
                let aux_msb    = me.aux_msb_cmap();
                let buf_sel    = me.buf_sel();
                let ovl_sel    = me.ovl_buf_sel();

                let cmap_base = if pix_mode == 0 {
                    if pix_size == 2 { (msb_cmap as usize & 0x10) << 8 }
                    else             { (msb_cmap as usize) << 8 }
                } else { 0 };
                let aux_base = (aux_msb as usize) << 8;

                if did5 as u32 == cursor_did5 { highlight_line = Some(lines.len()); }

                lines.push(format!(
                    "{:3}  {:06x}   {:<5}    {:<6}  {:02x}  {:04x}     {:x}        {:02x}     {:04x}    {} {}  {}",
                    did5, raw,
                    pix_mode_name(pix_mode), pix_size_name(pix_size),
                    msb_cmap, cmap_base,
                    aux_pix, aux_msb, aux_base,
                    buf_sel as u8, ovl_sel as u8,
                    pix_count,
                ));
            }

            // Render lines bottom-up from (height - lines.len()*16 - 4) so they sit
            // above the status bar without overlapping the CMAP swatch grid.
            let line_h = 16usize;
            // Place block so its bottom edge is 4px above the status bar row
            let total_h = lines.len() * line_h;
            let start_y = if height > total_h + 4 { height - total_h - 4 } else { 0 };

            // Dark semi-opaque background strip
            let bg = 0xC0101010u32;
            for row in 0..total_h {
                let py = start_y + row;
                if py >= height { break; }
                let base = py * 2048;
                for px in 0..width.min(840) {
                    if base + px < self.overlay_rgba.len() {
                        self.overlay_rgba[base + px] = bg;
                    }
                }
            }

            // Render each line using the VGA font via a temporary StatusBar-style draw
            let font_ref: &[u8] = &self.debug_font;
            for (li, line) in lines.iter().enumerate() {
                let y0 = start_y + li * line_h;
                let fg = if highlight_line == Some(li) { 0xFF00FFFF } else { 0xFF00FF00 }; // cyan=cursor row, green=normal
                let mut cx = 4usize;
                for ch in line.chars() {
                    if cx + 8 > width { break; }
                    let glyph = ch as usize & 0xFF;
                    let glyph_offset = glyph * 16;
                    for row in 0..16usize {
                        let byte_idx = glyph_offset + row;
                        if byte_idx >= font_ref.len() { break; }
                        let bits = font_ref[byte_idx];
                        let py = y0 + row;
                        if py >= height { break; }
                        let row_base = py * 2048;
                        for col in 0..8usize {
                            let px = cx + col;
                            if px >= width { break; }
                            let idx = row_base + px;
                            if idx < self.overlay_rgba.len() {
                                self.overlay_rgba[idx] = if (bits >> (7 - col)) & 1 != 0 { fg } else { bg };
                            }
                        }
                    }
                    cx += 8;
                }
            }
        }

        // Draw-debug overlay: show last ≤3 draws whose rect intersects the cursor.
        if self.show_draw_debug && height > 0 {
            let cx = cursor_x_hot.clamp(0, width as i32 - 1);
            let cy = cursor_y_hot.clamp(0, height as i32 - 1);

            // Collect up to 3 draws intersecting cursor (newest-first).
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
                // Red frame around the newest hit rect.
                let r0 = &hits[0];
                let fx0 = (r0.x0 as i32).min(r0.x1 as i32).clamp(0, width as i32 - 1) as usize;
                let fy0 = (r0.y0 as i32).min(r0.y1 as i32).clamp(0, height as i32 - 1) as usize;
                let fx1 = (r0.x0 as i32).max(r0.x1 as i32).clamp(0, width as i32 - 1) as usize;
                let fy1 = (r0.y0 as i32).max(r0.y1 as i32).clamp(0, height as i32 - 1) as usize;
                let red = 0xFF0000FFu32; // ABGR little-endian → R=255,G=0,B=0
                for x in fx0..=fx1 {
                    if fy0 * 2048 + x < self.overlay_rgba.len() { self.overlay_rgba[fy0 * 2048 + x] = red; }
                    if fy1 * 2048 + x < self.overlay_rgba.len() { self.overlay_rgba[fy1 * 2048 + x] = red; }
                }
                for y in fy0..=fy1 {
                    if y * 2048 + fx0 < self.overlay_rgba.len() { self.overlay_rgba[y * 2048 + fx0] = red; }
                    if y * 2048 + fx1 < self.overlay_rgba.len() { self.overlay_rgba[y * 2048 + fx1] = red; }
                }

                let op_str   = |dm0: u32| match DrawMode0(dm0).opcode() {
                    DRAWMODE0_OPCODE_READ    => "READ",
                    DRAWMODE0_OPCODE_DRAW    => "DRAW",
                    DRAWMODE0_OPCODE_SCR2SCR => "S2S",
                    _                        => "NOP",
                };
                let adr_str  = |dm0: u32| match DrawMode0(dm0).adrmode() {
                    0 => "SPAN", 1 => "BLK", 2 => "ILINE", 3 => "FLINE", 4 => "ALINE", _ => "?"
                };
                let pln_str  = |dm1: u32| match DrawMode1(dm1).planes() {
                    DRAWMODE1_PLANES_RGB  => "RGB",
                    DRAWMODE1_PLANES_RGBA => "RGBA",
                    DRAWMODE1_PLANES_OLAY => "OLAY",
                    DRAWMODE1_PLANES_PUP  => "PUP",
                    DRAWMODE1_PLANES_CID  => "CID",
                    _                     => "NONE",
                };
                let lop_str  = |dm1: u32| match DrawMode1(dm1).logicop() {
                    0=>"ZERO",1=>"AND",2=>"ANDR",3=>"SRC",4=>"ANDI",5=>"DST",6=>"XOR",7=>"OR",
                    8=>"NOR",9=>"XNOR",10=>"NDST",11=>"ORR",12=>"NSRC",13=>"ORI",14=>"NAND",_=>"ONE",
                };
                let dep_str  = |dm1: u32| match DrawMode1(dm1).drawdepth() { 0=>"4b",1=>"8b",2=>"12b",_=>"24b" };

                // Pixel values under cursor from fb buffers.
                let fb_cy_dd = (topscan + cy as usize) & 0x3FF;
                let fb_idx_dd = fb_cy_dd * 2048 + cx as usize;
                let dd_raw_rgb = if fb_idx_dd < self.fb_rgb.len() { self.fb_rgb[fb_idx_dd] } else { 0 };
                let dd_raw_aux = if fb_idx_dd < self.fb_aux.len() { self.fb_aux[fb_idx_dd] } else { 0 };

                let mut lines: Vec<(String, u32)> = Vec::new(); // (text, age_index)
                // 0xFF = header color (white)
                lines.push((format!(
                    "cursor=({},{})  RGB:{:06x}  AUX:{:06x} [ovl0={:02x} ovl1={:02x} pup0={} pup1={} cid={}]",
                    cx, cy,
                    dd_raw_rgb & 0xFFFFFF,
                    dd_raw_aux & 0xFFFFFF,
                    (dd_raw_aux >> 8) & 0xFF, (dd_raw_aux >> 16) & 0xFF,
                    (dd_raw_aux >> 2) & 3, (dd_raw_aux >> 6) & 3,
                    dd_raw_aux & 3,
                ), 0xFF));
                for (age, r) in hits.iter().enumerate() {
                    let dm0 = DrawMode0(r.dm0);
                    let dm1 = DrawMode1(r.dm1);
                    let colorhost = dm0.colorhost();
                    let alphahost = dm0.alphahost();
                    // HOSTRW accounting string
                    let hostrw_info = if colorhost || alphahost {
                        let label = match (colorhost, alphahost) {
                            (true,  true)  => "CH+AH",
                            (true,  false) => "CH",
                            (false, true)  => "AH",
                            _              => "",
                        };
                        let dbl = dm1.rwdouble();
                        let pk  = dm1.rwpacked();
                        let flags = match (dbl, pk) {
                            (true,  true)  => " DBL+PK",
                            (true,  false) => " DBL",
                            (false, true)  => " PK",
                            (false, false) => "",
                        };
                        let exp = if dbl { r.expected_doubles } else { r.expected_words };
                        let ok = r.hostrw_writes == exp;
                        format!(" {}{}: {}/{}{}", label, flags, r.hostrw_writes, exp,
                            if ok { "" } else { " MISMATCH" })
                    } else if r.spurious_writes > 0 {
                        format!(" SPURIOUS:{}", r.spurious_writes)
                    } else { String::new() };
                    // Line 1: geometry + mode summary + HOSTRW counts
                    let is_s2s = DrawMode0(r.dm0).opcode() == DRAWMODE0_OPCODE_SCR2SCR;
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
                    // Line 2: color + pattern
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

                let line_h = 16usize;
                let total_h = lines.len() * line_h;
                // Sit just above the display-debug overlay if it's also shown, otherwise above bottom
                let disp_debug_h = if self.show_disp_debug {
                    let n_disp_lines = 2 + self.seen_modes_count; // header + column + rows
                    n_disp_lines * line_h + 4
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
                        if base + px < self.overlay_rgba.len() { self.overlay_rgba[base + px] = bg; }
                    }
                }

                let font_ref: &[u8] = &self.debug_font;
                // Color palette: 0xFF=white header, newest=bright cyan, 2nd=medium, 3rd=dim
                let fgs = [0xFF00FFFF_u32, 0xFF00D8FFu32, 0xFF00B8D8u32]; // cyan shades newest→oldest
                for (li, (line, age)) in lines.iter().enumerate() {
                    let y0 = start_y + li * line_h;
                    let fg = if *age == 0xFF { 0xFFFFFFFF_u32 } else { fgs[(*age as usize).min(fgs.len() - 1)] };
                    let mut tx = 4usize;
                    for ch in line.chars() {
                        if tx + 8 > width { break; }
                        let glyph = ch as usize & 0xFF;
                        let goff = glyph * 16;
                        for row in 0..16usize {
                            let bi = goff + row;
                            if bi >= font_ref.len() { break; }
                            let bits = font_ref[bi];
                            let py = y0 + row;
                            if py >= height { break; }
                            let rb = py * 2048;
                            for col in 0..8usize {
                                let px = tx + col;
                                if px >= width { break; }
                                let idx = rb + px;
                                if idx < self.overlay_rgba.len() {
                                    self.overlay_rgba[idx] = if (bits >> (7 - col)) & 1 != 0 { fg } else { bg };
                                }
                            }
                        }
                        tx += 8;
                    }
                }
            }
        }

    }

    /// Render the status bar into `statusbar_rgba` (16 rows, separate from the main overlay).
    pub fn render_status_bar(&mut self, bar: &mut StatusBar, stats: &BarStats) {
        bar.update(stats.hb);
        bar.render(&mut self.statusbar_rgba, self.width, 0, stats);
    }
}

/// Height of the status bar in pixels (one VGA glyph row = 16px)
pub const STATUS_BAR_HEIGHT: usize = 16;

/// Snapshot of CPU counters and wall-clock time captured once per refresh loop iteration.
pub struct BarStats {
    pub now:            std::time::Instant,
    pub cycles:         u64,
    pub fasttick:       u64,
    pub decoded_delta:  u64,
    pub l1i_hits:       u64,
    pub l1i_fetches:    u64,
    pub uncached:       u64,
    pub hb:             u64,
    pub count_step:     u64,
    pub gfifo_pending:  usize,
}

/// Fade duration in frames (~quarter second at 60 Hz)
const FADE_FRAMES: u8 = 15;

/// Colors used in the status bar
const BAR_BG:        u32 = 0xFF202020; // dark grey background
const BAR_FG:        u32 = 0xFF00CC00; // green text
const BAR_ACTIVE:    u32 = 0xFF00FFAA; // bright cyan-green when active
const BAR_DIM:       u32 = 0xFF004400; // dim when inactive
const LED_RED_ON:    u32 = 0xFF2020FF; // bright red LED on
const LED_RED_OFF:   u32 = 0xFF000030; // dim red LED off
const LED_GREEN_ON:  u32 = 0xFF20FF20; // bright green LED on
const LED_GREEN_OFF: u32 = 0xFF003000; // dim green LED off

pub struct StatusBar {
    font: Vec<u8>,          // 4096 bytes: 256 × 16 rows × 1 byte (8px wide)
    enet_tx_fade: u8,
    enet_rx_fade: u8,
    scsi_fade: [u8; 7],    // IDs 0-6 (7 is host, unused)
    led_red: bool,
    led_green: bool,
    prev_cycles: u64,
    prev_fasttick: u64,
    prev_time: std::time::Instant,
    mips: f64,
    fasthz: f64,
    decode_pct: f64,
    l1i_hit_pct: f64,
    uncached_pct: f64,
}

impl StatusBar {
    pub fn new() -> Self {
        Self {
            font: crate::vga_font::VGA_8X16.to_vec(),
            enet_tx_fade: 0,
            enet_rx_fade: 0,
            scsi_fade: [0; 7],
            led_red: false,
            led_green: false,
            prev_cycles: 0,
            prev_fasttick: 0,
            prev_time: std::time::Instant::now(),
            mips: 0.0,
            fasthz: 0.0,
            decode_pct: 0.0,
            l1i_hit_pct: 0.0,
            uncached_pct: 0.0,
        }
    }

    /// Decode the shared heartbeat atomic value (already fetch_and_cleared by caller)
    /// and update fade counters. LED bits are persistent (not cleared by fetch_and).
    pub fn update(&mut self, hb: u64) {
        use crate::rex3::Rex3;
        if hb & Rex3::HB_ENET_TX != 0 { self.enet_tx_fade = FADE_FRAMES; }
        if hb & Rex3::HB_ENET_RX != 0 { self.enet_rx_fade = FADE_FRAMES; }
        for i in 0..7usize {
            if hb & (1 << (Rex3::HB_SCSI_BASE as u64 + i as u64)) != 0 {
                self.scsi_fade[i] = FADE_FRAMES;
            }
        }
        // LED bits are persistent — read state directly from the (un-cleared) bits.
        self.led_red   = hb & Rex3::HB_LED_RED   != 0;
        self.led_green = hb & Rex3::HB_LED_GREEN  != 0;
        // Decay fade counters
        if self.enet_tx_fade > 0 { self.enet_tx_fade -= 1; }
        if self.enet_rx_fade > 0 { self.enet_rx_fade -= 1; }
        for f in self.scsi_fade.iter_mut() { if *f > 0 { *f -= 1; } }
    }

    /// Render STATUS_BAR_HEIGHT rows of status bar into `rgba` starting at row `bar_y`.
    pub fn render(&mut self, rgba: &mut Vec<u32>, width: usize, bar_y: usize, stats: &BarStats) {
        // Update MIPS and fasthz estimates from deltas and wall-clock delta
        let dt = stats.now.duration_since(self.prev_time).as_secs_f64();
        if dt >= 0.1 {
            let dc = stats.cycles.wrapping_sub(self.prev_cycles);
            let df = stats.fasttick.wrapping_sub(self.prev_fasttick);
            self.mips        = (dc as f64 / dt / 1_000_000.0 * 10.0).round() / 10.0;
            #[cfg(feature = "developer")] {
                let total_fetches = stats.l1i_fetches + stats.uncached;
                self.decode_pct   = if total_fetches > 0 { stats.decoded_delta as f64 / total_fetches as f64 * 100.0 } else { 0.0 };
                self.l1i_hit_pct  = if stats.l1i_fetches > 0 { stats.l1i_hits as f64 / stats.l1i_fetches as f64 * 100.0 } else { 0.0 };
                self.uncached_pct = if dc > 0 { stats.uncached as f64 / dc as f64 * 100.0 } else { 0.0 };
            }
            self.fasthz      = (df as f64 / dt).round();
            self.prev_cycles   = stats.cycles;
            self.prev_fasttick = stats.fasttick;
            self.prev_time = stats.now;
        }

        let tx_color = if self.enet_tx_fade > 0 { BAR_ACTIVE } else { BAR_DIM };
        let rx_color = if self.enet_rx_fade > 0 { BAR_ACTIVE } else { BAR_DIM };

        #[cfg(feature = "developer")]
        let line = format!(" {:5.1} MIPS D:{:3.0}% I$:{:3.0}% UC:{:3.0}% {:4.0}Hz cs:{:08x} g{:04X}  NET:", self.mips, self.decode_pct, self.l1i_hit_pct, self.uncached_pct, self.fasthz, stats.count_step >> 32, stats.gfifo_pending);
        #[cfg(not(feature = "developer"))]
        let line = format!(" {:5.1} MIPS {:4.0}Hz  NET:", self.mips, self.fasthz);

        let row_stride = 2048;

        // Fill background
        for row in 0..STATUS_BAR_HEIGHT {
            let base = (bar_y + row) * row_stride;
            if base + width <= rgba.len() {
                rgba[base..base + width].fill(BAR_BG);
            }
        }

        let mut cursor_x = 0;

        // Draw the fixed text prefix
        cursor_x = self.draw_text(rgba, &line, cursor_x, bar_y, width, BAR_FG);

        // Draw TX
        cursor_x = self.draw_text(rgba, " TX", cursor_x, bar_y, width, tx_color);
        // Draw RX
        cursor_x = self.draw_text(rgba, " RX", cursor_x, bar_y, width, rx_color);

        cursor_x = self.draw_text(rgba, "  SCSI:", cursor_x, bar_y, width, BAR_FG);

        // Draw SCSI IDs 0-6 (7 is host, not shown)
        for i in 0..7 {
            let color = if self.scsi_fade[i] > 0 { BAR_ACTIVE } else { BAR_DIM };
            let s = format!(" {}", i);
            cursor_x = self.draw_text(rgba, &s, cursor_x, bar_y, width, color);
        }

        // Draw front-panel LEDs as small filled squares
        cursor_x = self.draw_text(rgba, "  LED:", cursor_x, bar_y, width, BAR_FG);
        cursor_x = self.draw_square(rgba, cursor_x + 2, bar_y, width,
            if self.led_red   { LED_RED_ON   } else { LED_RED_OFF   });
        cursor_x = self.draw_square(rgba, cursor_x + 4, bar_y, width,
            if self.led_green { LED_GREEN_ON } else { LED_GREEN_OFF });

        let _ = cursor_x;
    }

    /// Draw a 10×10 filled square centered vertically in the status bar.
    /// Returns x position after the square (x + 14).
    fn draw_square(&self, rgba: &mut Vec<u32>, x: usize, bar_y: usize, width: usize, color: u32) -> usize {
        const SQ: usize = 10;
        const MARGIN: usize = (STATUS_BAR_HEIGHT - SQ) / 2;
        let row_stride = 2048;
        for row in 0..SQ {
            let py = bar_y + MARGIN + row;
            for col in 0..SQ {
                let px = x + col;
                if px < width {
                    let idx = py * row_stride + px;
                    if idx < rgba.len() {
                        rgba[idx] = color;
                    }
                }
            }
        }
        x + SQ + 4
    }

    /// Draw text at (x, y) using VGA 8×16 font, returns new x position.
    fn draw_text(&self, rgba: &mut Vec<u32>, text: &str, x: usize, y: usize, width: usize, color: u32) -> usize {
        let mut cx = x;
        for ch in text.chars() {
            if cx + 8 > width { break; }
            self.draw_glyph(rgba, ch as usize & 0xFF, cx, y, width, color);
            cx += 8;
        }
        cx
    }

    /// Draw a single 8×16 VGA glyph at pixel position (x, y).
    fn draw_glyph(&self, rgba: &mut Vec<u32>, glyph: usize, x: usize, y: usize, width: usize, color: u32) {
        let glyph_offset = glyph * 16;
        for row in 0..16usize {
            let byte_idx = glyph_offset + row;
            if byte_idx >= self.font.len() { break; }
            let bits = self.font[byte_idx];
            let row_base = (y + row) * 2048;
            for col in 0..8usize {
                let px = x + col;
                if px >= width { break; }
                let idx = row_base + px;
                if idx >= rgba.len() { break; }
                let lit = (bits >> (7 - col)) & 1;
                rgba[idx] = if lit != 0 { color } else { BAR_BG };
            }
        }
    }
}

/// Save the emulator rgba buffer as a PNG file.
/// `rgba` uses 0xFFBBGGRR (GL internal format); swap R and B when writing PNG.
pub fn save_screenshot(path: &str, rgba: &[u32], width: usize, height: usize) -> Result<(), String> {
    use std::io::BufWriter;
    let file = std::fs::File::create(path).map_err(|e| e.to_string())?;
    let mut enc = png::Encoder::new(BufWriter::new(file), width as u32, height as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header().map_err(|e| e.to_string())?;
    let mut rows: Vec<u8> = Vec::with_capacity(height * width * 3);
    for y in 0..height {
        for x in 0..width {
            let px = rgba[y * 2048 + x];
            rows.push(( px        & 0xFF) as u8); // R (was in blue lane)
            rows.push(((px >>  8) & 0xFF) as u8); // G
            rows.push(((px >> 16) & 0xFF) as u8); // B (was in red lane)
        }
    }
    writer.write_image_data(&rows).map_err(|e| e.to_string())
}
