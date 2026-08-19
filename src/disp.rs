use std::sync::atomic::{AtomicU64, Ordering};
use parking_lot::Mutex;
use glow::HasContext;
use crate::vc2::{Vc2, VC2_REG_DISPLAY_CONTROL, VC2_CTRL_DID_EN, VC2_REG_DID_ENTRY_PTR,
    VC2_REG_SCANLINE_LEN, VC2_REG_CURRENT_CURSOR_X, VC2_REG_WORKING_CURSOR_Y,
    VC2_CTRL_CURSOR_EN, VC2_CTRL_CURSOR_SIZE, VC2_REG_CURSOR_ENTRY_PTR,
    VC2_REG_VIDEO_ENTRY_PTR, VC2_CTRL_BLACKOUT, VC2_CTRL_VIDEO_TIMING_EN,
    VT_VIS_LN_VC_N, VT_DSPLY_EN_RO_N, VT_CBLANK_XMAP_N, VT_HPOS_VC_N};
use crate::xmap9::Xmap9;
use crate::cmap::Cmap;
use crate::bt445::Bt445;
use crate::rex3::Rex3;
use crate::compositor::CompositorSource;
use crate::debug_overlay::OverlaySource;

/// Cached device state for the display pipeline.
/// The refresh loop copies hardware state into this struct once per frame,
/// then builds a `CompositorSource` from it for the active compositor.
pub struct Rex3Screen {
    pub width:  usize,
    pub height: usize,

    // Framebuffer snapshots (copied from Rex3::fb_rgb / fb_aux each frame)
    pub fb_rgb: Vec<u32>,
    pub fb_aux: Vec<u32>,

    // Decoded DID buffer in display coordinates (2048×1024, stride 2048)
    pub did: Vec<u8>,

    /// Last composited frame in CPU memory (stride 2048), filled after each present().
    /// Available for screenshots and CI pixel reads.
    pub rgba: Vec<u32>,

    // VC2 state
    pub vc2_ram:  Vec<u16>,
    pub vc2_regs: [u16; 32],

    // CMAP palette (8192 × 0x00RRGGBB)
    pub cmap: [u32; 8192],

    // Bt445 RAMDAC gamma LUT (256 × 0x00RRGGBB)
    pub ramdac_palette: [u32; 256],

    // XMAP mode table + config
    pub xmap_mode:        [u32; 32],
    pub xmap_config:      u8,
    pub xmap_cursor_cmap: u8,
    pub xmap_popup_cmap:  u8,

    // Display geometry derived from VC2 video timings
    /// Framebuffer row that maps to display row 0 (before the compositor adds 1)
    pub topscan:         usize,
    /// Cursor X correction from VT timing decode
    pub cursor_x_adjust: i32,

    /// When true, renderers may skip full compositor work and refresh only the
    /// status bar (heartbeat frames with no FB/palette change).
    pub status_bar_only: bool,
    /// When true, `fb_rgb`/`fb_aux` snapshots were not copied — use live slices from refresh().
    pub fb_borrowed: bool,
}

impl Rex3Screen {
    pub fn new() -> Self {
        Self {
            width:            0,
            height:           0,
            fb_rgb:           vec![0u32; 2048 * 1024],
            fb_aux:           vec![0u32; 2048 * 1024],
            did:              vec![0u8; 2048 * 1024],
            rgba:             vec![0u32; 2048 * 1024],
            vc2_ram:          vec![0u16; 32768],
            vc2_regs:         [0u16; 32],
            cmap:             [0u32; 8192],
            ramdac_palette:   [0u32; 256],
            xmap_mode:        [0u32; 32],
            xmap_config:      0,
            xmap_cursor_cmap: 0,
            xmap_popup_cmap:  0,
            topscan:          0,
            cursor_x_adjust:  0,
            status_bar_only:  false,
            fb_borrowed:      false,
        }
    }

    fn decode_did(&mut self) {
        let display_ctrl = self.vc2_regs[VC2_REG_DISPLAY_CONTROL as usize];
        if (display_ctrl & VC2_CTRL_DID_EN) == 0 {
            self.did.fill(0);
            return;
        }

        let did_ptr  = self.vc2_regs[VC2_REG_DID_ENTRY_PTR as usize];
        let scan_len = (self.vc2_regs[VC2_REG_SCANLINE_LEN as usize] >> 5) as usize;
        let width    = self.width;
        let height   = self.height;
        let ram      = &self.vc2_ram;
        let did_buf  = &mut self.did;

        let effective_len = if scan_len > 0 && scan_len <= 2048 { scan_len } else { width };

        let mut y         = 0;
        let mut table_idx = did_ptr as usize;

        while y < height {
            if table_idx >= ram.len() { break; }
            let line_ptr = ram[table_idx];
            if line_ptr == 0xFFFF { break; }

            let mut ptr = line_ptr as usize;

            if ptr >= ram.len() {
                let start_idx = y * 2048;
                let end_idx   = y * 2048 + effective_len;
                if end_idx <= did_buf.len() { did_buf[start_idx..end_idx].fill(0); }
                y += 1; table_idx += 1; continue;
            }

            let entry = ram[ptr]; ptr += 1;
            let mut current_did = (entry & 0x1F) as u8;
            let mut current_x   = 0usize;

            loop {
                if current_x >= effective_len { break; }
                if ptr >= ram.len() {
                    let s = y * 2048 + current_x;
                    let e = y * 2048 + effective_len;
                    if e <= did_buf.len() { did_buf[s..e].fill(current_did); }
                    break;
                }
                let next_entry  = ram[ptr]; ptr += 1;
                let next_x_raw  = ((next_entry >> 5) & 0x7FF) as usize;
                let next_did    = (next_entry & 0x1F) as u8;
                let is_eol      = next_x_raw == 0x7FF;
                let next_x      = if is_eol { effective_len } else { next_x_raw };
                let run_end     = next_x.max(current_x).min(effective_len);
                if run_end > current_x {
                    let s = y * 2048 + current_x;
                    let e = y * 2048 + run_end;
                    if e <= did_buf.len() { did_buf[s..e].fill(current_did); }
                }
                current_x   = run_end;
                current_did = next_did;
                if is_eol { break; }
            }
            y += 1; table_idx += 1;
        }
    }

    // Returns (width, height, cursor_x_adjust).
    fn decode_video_timings(&self) -> (usize, usize, i32) {
        decode_vc2_timings(&self.vc2_regs, &self.vc2_ram)
    }

    /// Copy hardware device state into this cache, decode timings and DID.
    /// Returns `true` if the display resolution changed.
    ///
    /// When `copy_fb` is false, skip the ~16 MB RGB/aux snapshot (palette/cursor
    /// heartbeat refreshes that don't need new framebuffer pixels).
    pub fn refresh(
        &mut self,
        fb_rgb:   &[u32],
        fb_aux:   &[u32],
        copy_fb:  bool,
        vc2:      &Mutex<Vc2>,
        xmap:     &Mutex<Xmap9>,
        cmap:     &Mutex<Cmap>,
        bt445:    &Mutex<Bt445>,
        diag:     &AtomicU64,
    ) -> bool {
        let mut resized = false;

        // ── 1. Copy device state snapshots ──────────────────────────────────────
        if copy_fb {
            diag.fetch_or(Rex3::DIAG_LOOP_FB_COPY, Ordering::Relaxed);
            self.fb_rgb.copy_from_slice(fb_rgb);
            self.fb_aux.copy_from_slice(fb_aux);
            self.fb_borrowed = false;
            diag.fetch_and(!Rex3::DIAG_LOOP_FB_COPY, Ordering::Relaxed);
        } else {
            self.fb_borrowed = true;
        }

        diag.fetch_or(Rex3::DIAG_LOCK_VC2 | Rex3::DIAG_LOOP_VC2_COPY, Ordering::Relaxed);
        {
            let mut vc2 = vc2.lock();
            if vc2.dirty {
                self.vc2_ram.copy_from_slice(&vc2.ram);
                self.vc2_regs.copy_from_slice(&vc2.regs);
                vc2.dirty = false;
            }
        }
        diag.fetch_and(!(Rex3::DIAG_LOCK_VC2 | Rex3::DIAG_LOOP_VC2_COPY), Ordering::Relaxed);

        diag.fetch_or(Rex3::DIAG_LOCK_CMAP0 | Rex3::DIAG_LOOP_CMAP_COPY, Ordering::Relaxed);
        {
            let mut cmap = cmap.lock();
            if cmap.dirty { self.cmap.copy_from_slice(&cmap.palette); cmap.dirty = false; }
        }
        diag.fetch_and(!(Rex3::DIAG_LOCK_CMAP0 | Rex3::DIAG_LOOP_CMAP_COPY), Ordering::Relaxed);

        {
            let mut dac = bt445.lock();
            if dac.dirty { self.ramdac_palette = dac.palette_as_rgb(); dac.dirty = false; }
        }

        diag.fetch_or(Rex3::DIAG_LOCK_XMAP0 | Rex3::DIAG_LOOP_XMAP_COPY, Ordering::Relaxed);
        {
            let mut xmap = xmap.lock();
            if xmap.dirty {
                self.xmap_mode.copy_from_slice(&xmap.mode_table);
                self.xmap_config      = xmap.config;
                self.xmap_cursor_cmap = xmap.cursor_cmap_msb;
                self.xmap_popup_cmap  = xmap.popup_cmap_msb;
                xmap.dirty = false;
            }
        }
        diag.fetch_and(!(Rex3::DIAG_LOCK_XMAP0 | Rex3::DIAG_LOOP_XMAP_COPY), Ordering::Relaxed);

        // ── 2. Check display-enable bits ─────────────────────────────────────────
        let display_ctrl    = self.vc2_regs[VC2_REG_DISPLAY_CONTROL as usize];
        let video_timing_en = (display_ctrl & VC2_CTRL_VIDEO_TIMING_EN) != 0;
        let display_en      = (display_ctrl & VC2_CTRL_BLACKOUT)        != 0;

        if !video_timing_en || !display_en {
            return resized;
        }

        // ── 3. Decode video timings and DID ──────────────────────────────────────
        diag.fetch_or(Rex3::DIAG_LOOP_VID_TIMINGS, Ordering::Relaxed);
        let (w, h, cursor_x_adjust) = self.decode_video_timings();
        diag.fetch_and(!Rex3::DIAG_LOOP_VID_TIMINGS, Ordering::Relaxed);

        self.cursor_x_adjust = cursor_x_adjust;
        if w > 0 && h > 0 && (w != self.width || h != self.height) {
            println!("Rex3: Resolution changed to {}x{} cursor_x_adjust={}", w, h, cursor_x_adjust);
            self.width  = w;
            self.height = h;
            resized = true;
        }

        diag.fetch_or(Rex3::DIAG_LOOP_DECODE_DID, Ordering::Relaxed);
        self.decode_did();
        diag.fetch_and(!Rex3::DIAG_LOOP_DECODE_DID, Ordering::Relaxed);

        resized
    }

    pub fn compositor_source_from<'a>(
        &'a self,
        live_rgb: Option<&'a [u32]>,
        live_aux: Option<&'a [u32]>,
    ) -> CompositorSource<'a> {
        CompositorSource {
            fb_rgb:           if self.fb_borrowed { live_rgb.unwrap_or(&self.fb_rgb) } else { &self.fb_rgb },
            fb_aux:           if self.fb_borrowed { live_aux.unwrap_or(&self.fb_aux) } else { &self.fb_aux },
            did:              &self.did,
            xmap_mode:        &self.xmap_mode,
            xmap_config:      self.xmap_config,
            xmap_cursor_cmap: self.xmap_cursor_cmap,
            xmap_popup_cmap:  self.xmap_popup_cmap,
            cmap:             &self.cmap,
            ramdac:           &self.ramdac_palette,
            vc2_ram:          &self.vc2_ram,
            vc2_regs:         &self.vc2_regs,
            topscan:          self.topscan,
            cursor_x_adjust:  self.cursor_x_adjust,
            width:            self.width,
            height:           self.height,
            status_bar_only:  self.status_bar_only,
        }
    }

    pub fn compositor_source(&self) -> CompositorSource<'_> {
        self.compositor_source_from(None, None)
    }

    /// Build an `OverlaySource` borrowing from this struct's caches.
    pub fn overlay_source(&self) -> OverlaySource<'_> {
        OverlaySource {
            width:            self.width,
            height:           self.height,
            topscan:          self.topscan,
            cursor_x_adjust:  self.cursor_x_adjust,
            vc2_regs:         &self.vc2_regs,
            xmap_mode:        &self.xmap_mode,
            xmap_cursor_cmap: self.xmap_cursor_cmap,
            xmap_popup_cmap:  self.xmap_popup_cmap,
            cmap:             &self.cmap,
            fb_rgb:           &self.fb_rgb,
            fb_aux:           &self.fb_aux,
            did:              &self.did,
        }
    }
}

/// Decode visible display size from VC2 state (used by compositor and timing presets).
pub fn decode_vc2_timings(vc2_regs: &[u16; 32], vc2_ram: &[u16]) -> (usize, usize, i32) {
    let frame_ptr = vc2_regs[VC2_REG_VIDEO_ENTRY_PTR as usize] as usize;
    let ram = vc2_ram;

    let mut max_visible_width   = 0usize;
    let mut total_visible_lines = 0usize;
    let mut hpos_to_visible: Option<usize> = None;
    let mut curr_frame_ptr = frame_ptr;
    let mut loop_safety    = 0usize;

    loop {
        if curr_frame_ptr + 1 >= ram.len() { break; }
        let line_seq_ptr    = ram[curr_frame_ptr] as usize;
        let mut line_seq_len = ram[curr_frame_ptr + 1] as usize;
        if line_seq_len == 0 { break; }
        let mut curr_line_ptr = line_seq_ptr;

        while line_seq_len > 0 {
            let mut line_visible_width = 0usize;
            let mut eol  = false;
            let mut line_loop_safety = 0usize;
            let mut state_c = 0u8;
            let mut pixel_offset = 0usize;
            let mut hpos_pixel: Option<usize> = None;
            let mut visible_pixel: Option<usize> = None;
            let mut hpos_seen_deasserted = false;

            while !eol {
                if curr_line_ptr >= ram.len() { break; }
                let w1 = ram[curr_line_ptr]; curr_line_ptr += 1;
                let duration      = ((w1 >> 8) & 0x7F) as usize;
                let state_a       = (w1 & 0x7F) as u8;
                let sb_sc_absent  = (w1 & 0x0080) != 0;
                let mut eol_bit   = (w1 & 0x8000) != 0;

                if !sb_sc_absent {
                    if curr_line_ptr >= ram.len() { break; }
                    let w2 = ram[curr_line_ptr];
                    if (w2 & 0x8000) != 0 { eol_bit = true; }
                    curr_line_ptr += 1;
                    state_c = (w2 & 0x7F) as u8;
                }

                eol = eol_bit;
                let pixels = duration * 2;

                if hpos_pixel.is_none() {
                    if (state_a & VT_HPOS_VC_N) != 0 {
                        hpos_seen_deasserted = true;
                    } else if hpos_seen_deasserted {
                        hpos_pixel = Some(pixel_offset);
                    }
                }

                let visible = (state_c & VT_CBLANK_XMAP_N) != 0;
                   // && (state_a & VT_VIS_LN_VC_N)    == 0
                   // && (state_a & VT_DSPLY_EN_RO_N)   == 0;
                   // including these allow us for more precise 1024/1280 visible pixels
                   // but we need two more to account for IRIX keeping 2 pixels on the left blank

                if visible {
                    if visible_pixel.is_none() { visible_pixel = Some(pixel_offset); }
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
                if hpos_to_visible.is_none() {
                    if let (Some(h), Some(v)) = (hpos_pixel, visible_pixel) {
                        if v >= h { hpos_to_visible = Some(v - h); }
                    }
                }
            }

            line_seq_len -= 1;
            if curr_line_ptr >= ram.len() { break; }
            curr_line_ptr = ram[curr_line_ptr] as usize;
        }

        curr_frame_ptr += 2;
        loop_safety    += 1;
        if loop_safety > 1000 { break; }
    }

    if max_visible_width > 0 && total_visible_lines > 0 {
        let w = max_visible_width.min(2048);
        let h = total_visible_lines.min(1024);
        let cursor_x_adjust = match hpos_to_visible {
            Some(d) => {
                let adj = d as i32 - 31;
                if adj < 0 || adj > 64 {
                    println!("Rex3: WARNING: hpos_to_visible={} gives cursor_x_adjust={}, out of range, falling back to 11", d, adj);
                    11
                } else { adj }
            }
            None => {
                println!("Rex3: WARNING: HPOS leading edge not found in VT, falling back to cursor_x_adjust=11");
                11
            }
        };
        (w, h, cursor_x_adjust-2)
    } else {
        (0, 0, 0)
    }
}

// ── Status bar ───────────────────────────────────────────────────────────────

/// Height of the status bar in pixels (one VGA glyph row = 16px)
pub const STATUS_BAR_HEIGHT: usize = 16;

/// A status-bar texture: CPU pixel buffer + backing GL texture.
/// Owned separately from the display compositor.
pub struct StatusBarTexture {
    pub buf: Vec<u32>,
    tex:     Option<glow::Texture>,
}

impl StatusBarTexture {
    pub fn new() -> Self {
        Self {
            buf: vec![0u32; 2048 * STATUS_BAR_HEIGHT],
            tex: None,
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
                2048, STATUS_BAR_HEIGHT as i32, 0, glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelUnpackData::Slice(None));
            t
        };
        self.tex = Some(t);
        t
    }

    /// Render the status bar and upload to the GL texture; returns the texture handle.
    pub fn render_and_upload(&mut self, bar: &mut StatusBar, stats: &BarStats, width: usize, gl: &glow::Context) -> glow::Texture {
        bar.update(stats.hb);
        bar.render(&mut self.buf, width, 0, stats);
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
                0, 0, width as i32, STATUS_BAR_HEIGHT as i32,
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
}

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
    pub count_hz:       u64,
    pub gfifo_pending:  usize,
}

const FADE_FRAMES: u8 = 15;

const BAR_BG:        u32 = 0xFF202020;
const BAR_FG:        u32 = 0xFF00CC00;
const BAR_ACTIVE:    u32 = 0xFF00FFAA;
const BAR_DIM:       u32 = 0xFF004400;
const LED_RED_ON:    u32 = 0xFF2020FF;
const LED_RED_OFF:   u32 = 0xFF000030;
const LED_GREEN_ON:  u32 = 0xFF20FF20;
const LED_GREEN_OFF: u32 = 0xFF003000;
/// Flash color for the jitv2 rectangle right after a `mega_flush` — purple,
/// distinct from any steady-state green/red fill color so it reads as an
/// event, not just "100% full and red". 0xAABBGGRR (this file's rgba
/// convention — see `save_screenshot`'s doc comment).
#[cfg(feature = "jitv2")]
const JIT_FLUSH_FLASH: u32 = 0xFFFF20C0;

/// Green-at-0-to-red-at-255 interpolation for the jitv2 fill bars
/// (`StatusBar::draw_fill_bar`). `frac` is `JitFeedback`'s 0..=255
/// fixed-point fraction, not a 0..=100 percentage. 0xAABBGGRR, this file's
/// rgba convention: B fixed near 0, G falls from 0xE0 to 0 as R rises from
/// 0 to 0xE0, so the midpoint lands on a yellow-ish amber rather than
/// passing through a washed-out gray.
#[cfg(feature = "jitv2")]
fn lerp_green_red(frac: u8) -> u32 {
    let f = frac as u32;
    let r = f * 0xE0 / 255;
    let g = 0xE0 - r;
    0xFF000000 | (g << 8) | r
}

pub struct StatusBar {
    font: Vec<u8>,
    enet_tx_fade: u8,
    enet_rx_fade: u8,
    scsi_fade: [u8; 7],
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
    /// Live arena/queue fill fractions (0..=255) from `crate::jit_feedback`,
    /// sampled fresh every `update()` — unlike the fade counters above,
    /// these are gauges, not events, so there's nothing to remember between
    /// frames beyond "what they are right now". Feature-gated: with jitv2
    /// not built in, `crate::jit_feedback::JIT_FEEDBACK` never gets touched
    /// by anything and would just read as a permanently-empty rectangle —
    /// this hides it instead of showing a gauge for a JIT that isn't there.
    #[cfg(feature = "jitv2")]
    jit_arena_fill: u8,
    #[cfg(feature = "jitv2")]
    jit_queue_fill: u8,
    /// Countdown that keeps the JIT rectangle flashing for a few frames
    /// after a `mega_flush`, same fade pattern as `enet_tx_fade` above.
    #[cfg(feature = "jitv2")]
    jit_flush_fade: u8,
    /// Last `JitFeedback::flush_events` value seen — a flush is detected by
    /// this counter changing, not by a bool, so a flush that fires and a
    /// later one both landing between two `update()` samples still counts
    /// (see `JitFeedback::flush_events`'s own doc comment).
    #[cfg(feature = "jitv2")]
    jit_flush_seen: u32,
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
            #[cfg(feature = "jitv2")]
            jit_arena_fill: 0,
            #[cfg(feature = "jitv2")]
            jit_queue_fill: 0,
            #[cfg(feature = "jitv2")]
            jit_flush_fade: 0,
            #[cfg(feature = "jitv2")]
            jit_flush_seen: 0,
        }
    }

    pub fn update(&mut self, hb: u64) {
        use crate::rex3::Rex3;
        if hb & Rex3::HB_ENET_TX != 0 { self.enet_tx_fade = FADE_FRAMES; }
        if hb & Rex3::HB_ENET_RX != 0 { self.enet_rx_fade = FADE_FRAMES; }
        for i in 0..7usize {
            if hb & (1 << (Rex3::HB_SCSI_BASE as u64 + i as u64)) != 0 {
                self.scsi_fade[i] = FADE_FRAMES;
            }
        }
        self.led_red   = hb & Rex3::HB_LED_RED   != 0;
        self.led_green = hb & Rex3::HB_LED_GREEN  != 0;
        if self.enet_tx_fade > 0 { self.enet_tx_fade -= 1; }
        if self.enet_rx_fade > 0 { self.enet_rx_fade -= 1; }
        for f in self.scsi_fade.iter_mut() { if *f > 0 { *f -= 1; } }

        #[cfg(feature = "jitv2")]
        {
            use std::sync::atomic::Ordering as JitOrdering;
            let fb = &crate::jit_feedback::JIT_FEEDBACK;
            self.jit_arena_fill = fb.arena_fill.load(JitOrdering::Relaxed);
            self.jit_queue_fill = fb.queue_fill.load(JitOrdering::Relaxed);
            let flush_events = fb.flush_events.load(JitOrdering::Relaxed);
            if flush_events != self.jit_flush_seen {
                self.jit_flush_seen = flush_events;
                self.jit_flush_fade = FADE_FRAMES;
            }
            if self.jit_flush_fade > 0 { self.jit_flush_fade -= 1; }
        }
    }

    pub fn render(&mut self, rgba: &mut Vec<u32>, width: usize, bar_y: usize, stats: &BarStats) {
        let dt = stats.now.duration_since(self.prev_time).as_secs_f64();
        if dt >= 0.1 {
            let dc = stats.cycles.wrapping_sub(self.prev_cycles);
            let df = stats.fasttick.wrapping_sub(self.prev_fasttick);
            self.mips   = (dc as f64 / dt / 1_000_000.0 * 10.0).round() / 10.0;
            #[cfg(feature = "developer")] {
                let total_fetches = stats.l1i_fetches + stats.uncached;
                self.decode_pct   = if total_fetches > 0 { stats.decoded_delta as f64 / total_fetches as f64 * 100.0 } else { 0.0 };
                self.l1i_hit_pct  = if stats.l1i_fetches > 0 { stats.l1i_hits as f64 / stats.l1i_fetches as f64 * 100.0 } else { 0.0 };
                self.uncached_pct = if dc > 0 { stats.uncached as f64 / dc as f64 * 100.0 } else { 0.0 };
            }
            self.fasthz = (df as f64 / dt).round();
            self.prev_cycles   = stats.cycles;
            self.prev_fasttick = stats.fasttick;
            self.prev_time     = stats.now;
        }

        let tx_color = if self.enet_tx_fade > 0 { BAR_ACTIVE } else { BAR_DIM };
        let rx_color = if self.enet_rx_fade > 0 { BAR_ACTIVE } else { BAR_DIM };

        #[cfg(feature = "developer")]
        let line = format!(" {:5.1} MIPS D:{:3.0}% I$:{:3.0}% UC:{:3.0}% {:4.0}Hz cnt:{:5.1}MHz g{:04X}  NET:", self.mips, self.decode_pct, self.l1i_hit_pct, self.uncached_pct, self.fasthz, stats.count_hz as f64 / 1e6, stats.gfifo_pending);
        #[cfg(not(feature = "developer"))]
        let line = format!(" {:5.1} MIPS {:4.0}Hz  NET:", self.mips, self.fasthz);

        let row_stride = 2048;
        for row in 0..STATUS_BAR_HEIGHT {
            let base = (bar_y + row) * row_stride;
            if base + width <= rgba.len() { rgba[base..base + width].fill(BAR_BG); }
        }

        let mut cursor_x = 0;
        cursor_x = self.draw_text(rgba, &line,    cursor_x, bar_y, width, BAR_FG);
        cursor_x = self.draw_text(rgba, " TX",    cursor_x, bar_y, width, tx_color);
        cursor_x = self.draw_text(rgba, " RX",    cursor_x, bar_y, width, rx_color);
        cursor_x = self.draw_text(rgba, "  SCSI:", cursor_x, bar_y, width, BAR_FG);
        for i in 0..7 {
            let color = if self.scsi_fade[i] > 0 { BAR_ACTIVE } else { BAR_DIM };
            cursor_x = self.draw_text(rgba, &format!(" {}", i), cursor_x, bar_y, width, color);
        }
        cursor_x = self.draw_text(rgba, "  LED:", cursor_x, bar_y, width, BAR_FG);
        cursor_x = self.draw_square(rgba, cursor_x + 2, bar_y, width,
            if self.led_red   { LED_RED_ON   } else { LED_RED_OFF   });
        cursor_x = self.draw_square(rgba, cursor_x + 4, bar_y, width,
            if self.led_green { LED_GREEN_ON } else { LED_GREEN_OFF });
        #[cfg(feature = "jitv2")]
        {
            cursor_x = self.draw_text(rgba, "  JIT:", cursor_x, bar_y, width, BAR_FG);
            cursor_x = self.draw_jit_rect(rgba, cursor_x + 2, bar_y, width);
        }
        let _ = cursor_x;
    }

    /// 32x16 jitv2 feedback rectangle: top 32x8 is arena (page-pool) fill,
    /// bottom 32x8 is compile-queue fill. Each is a horizontal fill bar
    /// (dim background, filled left-to-right by the live percentage,
    /// like `draw_bar`'s convention elsewhere in this file) whose filled
    /// color interpolates green-at-0%-to-red-at-100% (`lerp_green_red`) —
    /// see `crate::jit_feedback::JitFeedback`'s own doc comment for what
    /// feeds the two percentages. While `jit_flush_fade` is counting down
    /// (a `mega_flush` happened recently), the whole rectangle flashes a
    /// single flat color instead, overriding both bars — a flush resets the
    /// gauges to near-zero anyway, so showing the bars mid-flash would just
    /// read as "everything's fine" for one frame right when it isn't.
    #[cfg(feature = "jitv2")]
    fn draw_jit_rect(&self, rgba: &mut Vec<u32>, x: usize, bar_y: usize, width: usize) -> usize {
        const RECT_W: usize = 32;
        const RECT_H: usize = 8;
        if self.jit_flush_fade > 0 {
            self.fill_rect(rgba, x, bar_y, RECT_W, RECT_H * 2, width, JIT_FLUSH_FLASH);
            return x + RECT_W + 4;
        }
        self.draw_fill_bar(rgba, x, bar_y,            RECT_W, RECT_H, width, self.jit_arena_fill);
        self.draw_fill_bar(rgba, x, bar_y + RECT_H,   RECT_W, RECT_H, width, self.jit_queue_fill);
        x + RECT_W + 4
    }

    /// One horizontal fill bar: `BAR_BG` background, filled left-to-right by
    /// `fill/255`, colored by `lerp_green_red(fill)`.
    #[cfg(feature = "jitv2")]
    fn draw_fill_bar(&self, rgba: &mut Vec<u32>, x: usize, y: usize, w: usize, h: usize, width: usize, fill: u8) {
        let filled_cols = (w * fill as usize) / 255;
        let color = lerp_green_red(fill);
        for col in 0..w {
            let px = x + col;
            if px >= width { break; }
            let c = if col < filled_cols { color } else { BAR_BG };
            for row in 0..h {
                let idx = (y + row) * 2048 + px;
                if idx < rgba.len() { rgba[idx] = c; }
            }
        }
    }

    #[cfg(feature = "jitv2")]
    fn fill_rect(&self, rgba: &mut Vec<u32>, x: usize, y: usize, w: usize, h: usize, width: usize, color: u32) {
        for row in 0..h {
            let py = y + row;
            for col in 0..w {
                let px = x + col;
                if px < width {
                    let idx = py * 2048 + px;
                    if idx < rgba.len() { rgba[idx] = color; }
                }
            }
        }
    }

    fn draw_square(&self, rgba: &mut Vec<u32>, x: usize, bar_y: usize, width: usize, color: u32) -> usize {
        const SQ: usize = 10;
        const MARGIN: usize = (STATUS_BAR_HEIGHT - SQ) / 2;
        for row in 0..SQ {
            let py = bar_y + MARGIN + row;
            for col in 0..SQ {
                let px = x + col;
                if px < width {
                    let idx = py * 2048 + px;
                    if idx < rgba.len() { rgba[idx] = color; }
                }
            }
        }
        x + SQ + 4
    }

    fn draw_text(&self, rgba: &mut Vec<u32>, text: &str, x: usize, y: usize, width: usize, color: u32) -> usize {
        let mut cx = x;
        for ch in text.chars() {
            if cx + 8 > width { break; }
            self.draw_glyph(rgba, ch as usize & 0xFF, cx, y, width, color);
            cx += 8;
        }
        cx
    }

    fn draw_glyph(&self, rgba: &mut Vec<u32>, glyph: usize, x: usize, y: usize, width: usize, color: u32) {
        let glyph_offset = glyph * 16;
        for row in 0..16usize {
            let byte_idx = glyph_offset + row;
            if byte_idx >= self.font.len() { break; }
            let bits     = self.font[byte_idx];
            let row_base = (y + row) * 2048;
            for col in 0..8usize {
                let px  = x + col;
                if px >= width { break; }
                let idx = row_base + px;
                if idx >= rgba.len() { break; }
                let lit = (bits >> (7 - col)) & 1;
                rgba[idx] = if lit != 0 { color } else { BAR_BG };
            }
        }
    }
}

/// Save the composited rgba buffer as a PNG file.
/// `rgba` uses 0xFFBBGGRR (GL-native); swap R↔B when writing PNG.
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
            rows.push(( px        & 0xFF) as u8);
            rows.push(((px >>  8) & 0xFF) as u8);
            rows.push(((px >> 16) & 0xFF) as u8);
        }
    }
    writer.write_image_data(&rows).map_err(|e| e.to_string())
}
