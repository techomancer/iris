use crate::compositor::CompositorSource;
use crate::debug_overlay::OverlaySource;
use crate::dev::ng1::bt445::Bt445;
use crate::dev::ng1::cmap::Cmap;
use crate::dev::ng1::rex3::Rex3;
#[cfg(feature = "rex-dirty-upload")]
use crate::dev::ng1::rex3_dirty::{copy_u32_rects, DamagePlane, DamageRect, DamageSnapshot};
use crate::dev::ng1::vc2::{
    Vc2, VC2_CTRL_BLACKOUT, VC2_CTRL_CURSOR_EN, VC2_CTRL_CURSOR_SIZE, VC2_CTRL_DID_EN,
    VC2_CTRL_VIDEO_TIMING_EN, VC2_REG_CURRENT_CURSOR_X, VC2_REG_CURSOR_ENTRY_PTR,
    VC2_REG_DID_ENTRY_PTR, VC2_REG_DISPLAY_CONTROL, VC2_REG_SCANLINE_LEN, VC2_REG_VIDEO_ENTRY_PTR,
    VC2_REG_WORKING_CURSOR_Y, VT_CBLANK_XMAP_N, VT_DSPLY_EN_RO_N, VT_HPOS_VC_N, VT_VIS_LN_VC_N,
};
use crate::dev::ng1::xmap9::Xmap9;
use glow::HasContext;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Cached device state for the display pipeline.
/// The refresh loop copies hardware state into this struct once per frame,
/// then builds a `CompositorSource` from it for the active compositor.
/// Row stride of every display buffer, in pixels.
pub const FB_STRIDE: usize = 2048;
/// Rows of the display buffers and output textures: the tallest frame a
/// board can hand over (Newport and GR2 use the first 1024; IMPACT's
/// 1600x1200 modes need more).
pub const FB_MAX_H: usize = 2048;

/// Independent revisions of the cached Newport display state. The source
/// identifies the screen that issued them, so replacing a screen cannot make
/// an old compositor cache appear current.
#[cfg(feature = "wgpu")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Ng1DisplayGenerations {
    pub(crate) source: u64,
    pub(crate) vc2_ram: u64,
    pub(crate) vc2_regs: u64,
    pub(crate) xmap: u64,
    pub(crate) cmap: u64,
    pub(crate) ramdac: u64,
    pub(crate) did: u64,
    pub(crate) geometry: u64,
}

#[cfg(feature = "wgpu")]
impl Ng1DisplayGenerations {
    fn new() -> Self {
        static NEXT_SOURCE: AtomicU64 = AtomicU64::new(1);
        let source = NEXT_SOURCE
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .expect("Newport display source identifiers exhausted");
        Self { source, vc2_ram: 0, vc2_regs: 0, xmap: 0, cmap: 0, ramdac: 0, did: 0, geometry: 0 }
    }

    fn bump(generation: &mut u64) {
        *generation = generation.checked_add(1).expect("Newport display generation exhausted");
    }
}

/// A Newport display snapshot with no access to either CPU framebuffer.
/// The generations change when the corresponding cached state changes.
#[cfg(feature = "wgpu")]
pub(crate) struct Ng1DisplaySource<'a> {
    pub(crate) generations: Ng1DisplayGenerations,
    pub(crate) did: &'a [u8],
    pub(crate) resident_did: &'a Mutex<Option<crate::dev::ng1::wgpu::direct::ResidentDid>>,
    pub(crate) xmap_mode: &'a [u32; 32],
    pub(crate) xmap_config: u8,
    pub(crate) xmap_cursor_cmap: u8,
    pub(crate) xmap_popup_cmap: u8,
    pub(crate) cmap: &'a [u32; 8192],
    pub(crate) ramdac: &'a [u32; 256],
    pub(crate) vc2_ram: &'a [u16],
    pub(crate) vc2_regs: &'a [u16; 32],
    pub(crate) topscan: usize,
    pub(crate) cursor_x_adjust: i32,
    pub(crate) width: usize,
    pub(crate) height: usize,
}

pub struct Rex3Screen {
    pub width:  usize,
    pub height: usize,

    // Framebuffer snapshots (copied from Rex3::fb_rgb / fb_aux each frame)
    pub fb_rgb: Vec<u32>,
    pub fb_aux: Vec<u32>,
    #[cfg(feature = "rex-dirty-upload")]
    fb_rgb_damage: Vec<DamageRect>,
    #[cfg(feature = "rex-dirty-upload")]
    fb_aux_damage: Vec<DamageRect>,
    #[cfg(feature = "rex-dirty-upload")]
    fb_rgb_tiles: usize,
    #[cfg(feature = "rex-dirty-upload")]
    fb_aux_tiles: usize,

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
    /// The cursor-gap fault last reported, so each table is reported once.
    cursor_gap_warned: Option<CursorGapFault>,

    /// When true, renderers may skip full compositor work and refresh only the
    /// status bar (heartbeat frames with no FB/palette change).
    pub status_bar_only: bool,
    /// When true, `fb_rgb`/`fb_aux` snapshots were not copied; use the live
    /// slices supplied by the refresh loop.
    #[cfg(not(feature = "rex-dirty-upload"))]
    pub fb_borrowed: bool,
    /// When true, `fb_rgb` already holds a finished frame (stride 2048,
    /// `0xAABBGGRR` with R in the low byte, the same format `SwCompositor`
    /// produces) built by a
    /// board's own compositor (GR2's `gr2comp`). The renderer's compositor
    /// then presents it as-is instead of decoding Newport state.
    pub prebuilt: bool,
    /// Bumped each time `rgba` is rewritten with a composed frame, so frame
    /// consumers (VNC) can skip unchanged ones without comparing pixels.
    pub frame_seq: u64,
    #[cfg(feature = "wgpu")]
    display_generations: Ng1DisplayGenerations,
    #[cfg(feature = "wgpu")]
    display_geometry: (usize, usize, usize, i32),
    #[cfg(feature = "wgpu")]
    resident_did: Mutex<Option<crate::dev::ng1::wgpu::direct::ResidentDid>>,
}

impl Rex3Screen {
    pub fn new() -> Self {
        Self {
            width:            0,
            height:           0,
            fb_rgb:           vec![0u32; FB_STRIDE * FB_MAX_H],
            fb_aux:           vec![0u32; FB_STRIDE * FB_MAX_H],
            #[cfg(feature = "rex-dirty-upload")]
            fb_rgb_damage: Vec::new(),
            #[cfg(feature = "rex-dirty-upload")]
            fb_aux_damage: Vec::new(),
            #[cfg(feature = "rex-dirty-upload")]
            fb_rgb_tiles: 0,
            #[cfg(feature = "rex-dirty-upload")]
            fb_aux_tiles: 0,
            did:              vec![0u8; FB_STRIDE * FB_MAX_H],
            rgba:             vec![0u32; FB_STRIDE * FB_MAX_H],
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
            cursor_gap_warned: None,
            status_bar_only:  false,
            #[cfg(not(feature = "rex-dirty-upload"))]
            fb_borrowed:      false,
            prebuilt:         false,
            frame_seq:        0,
            #[cfg(feature = "wgpu")]
            display_generations: Ng1DisplayGenerations::new(),
            #[cfg(feature = "wgpu")]
            display_geometry: (0, 0, 0, 0),
            #[cfg(feature = "wgpu")]
            resident_did: Mutex::new(None),
        }
    }

    fn decode_did_into(
        vc2_regs: &[u16; 32],
        vc2_ram: &[u16],
        width: usize,
        height: usize,
        did_buf: &mut [u8],
    ) -> bool {
        fn fill(dst: &mut [u8], value: u8) -> bool {
            #[cfg(feature = "wgpu")]
            let changed = dst.iter().any(|&old| old != value);
            #[cfg(not(feature = "wgpu"))]
            let changed = false;
            dst.fill(value);
            changed
        }

        let display_ctrl = vc2_regs[VC2_REG_DISPLAY_CONTROL as usize];
        if (display_ctrl & VC2_CTRL_DID_EN) == 0 {
            return fill(did_buf, 0);
        }

        let mut changed = false;
        let did_ptr  = vc2_regs[VC2_REG_DID_ENTRY_PTR as usize];
        let scan_len = (vc2_regs[VC2_REG_SCANLINE_LEN as usize] >> 5) as usize;
        let ram      = vc2_ram;

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
                if end_idx <= did_buf.len() { changed |= fill(&mut did_buf[start_idx..end_idx], 0); }
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
                    if e <= did_buf.len() { changed |= fill(&mut did_buf[s..e], current_did); }
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
                    if e <= did_buf.len() { changed |= fill(&mut did_buf[s..e], current_did); }
                }
                current_x   = run_end;
                current_did = next_did;
                if is_eol { break; }
            }
            y += 1; table_idx += 1;
        }
        changed
    }

    fn decode_did(&mut self) -> bool {
        Self::decode_did_into(
            &self.vc2_regs,
            &self.vc2_ram,
            self.width,
            self.height,
            &mut self.did,
        )
    }

    #[cfg(all(test, feature = "wgpu"))]
    pub(crate) fn decoded_did_oracle(&self) -> Vec<u8> {
        let mut did = self.did.clone();
        Self::decode_did_into(
            &self.vc2_regs,
            &self.vc2_ram,
            self.width,
            self.height,
            &mut did,
        );
        did
    }

    // Returns (width, height, cursor_x_adjust).
    fn decode_video_timings(&mut self) -> (usize, usize, i32) {
        let (w, h, adjust, fault) = decode_vc2_timings_gap(&self.vc2_regs, &self.vc2_ram);
        // Said once per table, and only while there is a cursor to misplace.
        let cursor_on = self.vc2_regs[VC2_REG_DISPLAY_CONTROL as usize] & VC2_CTRL_CURSOR_EN != 0;
        match fault {
            Some(gap) if cursor_on && self.cursor_gap_warned != Some(gap) => {
                self.cursor_gap_warned = Some(gap);
                match gap {
                    Some(d) => println!("Rex3: WARNING: hpos_to_visible={d} is out of the cursor adjustment's range, using 11"),
                    None => println!("Rex3: WARNING: HPOS leading edge not found in the video table, using cursor_x_adjust=11"),
                }
            }
            None => self.cursor_gap_warned = None,
            _ => {}
        }
        (w, h, adjust)
    }

    /// Copy framebuffer and display state into this cache.
    /// Returns `true` if the display resolution changed.
    ///
    /// With `rex-dirty-upload`, only rectangles named by `fb_damage` are
    /// copied into the cached planes. Otherwise `copy_fb` preserves the
    /// original full-copy/borrowed-frame behavior.
    pub(crate) fn refresh(
        &mut self,
        fb_rgb: &[u32],
        fb_aux: &[u32],
        #[cfg(feature = "rex-dirty-upload")] fb_damage: &DamageSnapshot,
        #[cfg(not(feature = "rex-dirty-upload"))] copy_fb: bool,
        vc2: &Mutex<Vc2>,
        xmap:     &Mutex<Xmap9>,
        cmap:     &Mutex<Cmap>,
        bt445:    &Mutex<Bt445>,
        diag:     &AtomicU64,
    ) -> bool {
        #[cfg(feature = "rex-dirty-upload")]
        {
            fb_damage.rectangles_into(DamagePlane::Rgb, &mut self.fb_rgb_damage);
            fb_damage.rectangles_into(DamagePlane::Aux, &mut self.fb_aux_damage);
            self.fb_rgb_tiles = fb_damage.tile_count(DamagePlane::Rgb);
            self.fb_aux_tiles = fb_damage.tile_count(DamagePlane::Aux);
            let rgb_bytes = self
                .fb_rgb_damage
                .iter()
                .map(|rect| rect.bytes())
                .sum::<usize>();
            let aux_bytes = self
                .fb_aux_damage
                .iter()
                .map(|rect| rect.bytes())
                .sum::<usize>();
            crate::pf_counter!(
                crate::perfetto::events::REX3_DIRTY_RGB_TILES,
                self.fb_rgb_tiles
            );
            crate::pf_counter!(
                crate::perfetto::events::REX3_DIRTY_AUX_TILES,
                self.fb_aux_tiles
            );
            crate::pf_counter!(crate::perfetto::events::REX3_COPY_RGB_BYTES, rgb_bytes);
            crate::pf_counter!(crate::perfetto::events::REX3_COPY_AUX_BYTES, aux_bytes);
            if rgb_bytes != 0 || aux_bytes != 0 {
                crate::pf_scope!(crate::perfetto::events::REX3_FB_COPY, rgb_bytes + aux_bytes);
                diag.fetch_or(Rex3::DIAG_LOOP_FB_COPY, Ordering::Relaxed);
                copy_u32_rects(&mut self.fb_rgb, fb_rgb, &self.fb_rgb_damage);
                copy_u32_rects(&mut self.fb_aux, fb_aux, &self.fb_aux_damage);
                diag.fetch_and(!Rex3::DIAG_LOOP_FB_COPY, Ordering::Relaxed);
            }
        }
        #[cfg(not(feature = "rex-dirty-upload"))]
        if copy_fb {
            diag.fetch_or(Rex3::DIAG_LOOP_FB_COPY, Ordering::Relaxed);
            self.fb_rgb[..fb_rgb.len()].copy_from_slice(fb_rgb);
            self.fb_aux[..fb_aux.len()].copy_from_slice(fb_aux);
            self.fb_borrowed = false;
            diag.fetch_and(!Rex3::DIAG_LOOP_FB_COPY, Ordering::Relaxed);
        } else {
            self.fb_borrowed = true;
        }

        self.refresh_display_state(vc2, xmap, cmap, bt445, diag)
    }

    /// Snapshot Newport display controls and tables without accessing either
    /// CPU framebuffer. Returns `true` if the display resolution changed.
    fn refresh_display_state_inner(
        &mut self,
        vc2: &Mutex<Vc2>,
        xmap: &Mutex<Xmap9>,
        cmap: &Mutex<Cmap>,
        bt445: &Mutex<Bt445>,
        diag: &AtomicU64,
        decode_did: bool,
    ) -> bool {
        let mut resized = false;

        diag.fetch_or(
            Rex3::DIAG_LOCK_VC2 | Rex3::DIAG_LOOP_VC2_COPY,
            Ordering::Relaxed,
        );
        {
            let mut vc2 = vc2.lock();
            if vc2.dirty {
                #[cfg(feature = "wgpu")]
                {
                    if self.vc2_ram != vc2.ram {
                        Ng1DisplayGenerations::bump(&mut self.display_generations.vc2_ram);
                    }
                    if self.vc2_regs != vc2.regs {
                        Ng1DisplayGenerations::bump(&mut self.display_generations.vc2_regs);
                    }
                }
                self.vc2_ram.copy_from_slice(&vc2.ram);
                self.vc2_regs.copy_from_slice(&vc2.regs);
                vc2.dirty = false;
            }
        }
        diag.fetch_and(!(Rex3::DIAG_LOCK_VC2 | Rex3::DIAG_LOOP_VC2_COPY), Ordering::Relaxed);

        diag.fetch_or(Rex3::DIAG_LOCK_CMAP0 | Rex3::DIAG_LOOP_CMAP_COPY, Ordering::Relaxed);
        {
            let mut cmap = cmap.lock();
            if cmap.dirty {
                #[cfg(feature = "wgpu")]
                if self.cmap != cmap.palette {
                    Ng1DisplayGenerations::bump(&mut self.display_generations.cmap);
                }
                self.cmap.copy_from_slice(&cmap.palette);
                cmap.dirty = false;
            }
        }
        diag.fetch_and(!(Rex3::DIAG_LOCK_CMAP0 | Rex3::DIAG_LOOP_CMAP_COPY), Ordering::Relaxed);

        {
            let mut dac = bt445.lock();
            if dac.dirty {
                let palette = dac.palette_as_rgb();
                #[cfg(feature = "wgpu")]
                if self.ramdac_palette != palette {
                    Ng1DisplayGenerations::bump(&mut self.display_generations.ramdac);
                }
                self.ramdac_palette = palette;
                dac.dirty = false;
            }
        }

        diag.fetch_or(Rex3::DIAG_LOCK_XMAP0 | Rex3::DIAG_LOOP_XMAP_COPY, Ordering::Relaxed);
        {
            let mut xmap = xmap.lock();
            if xmap.dirty {
                #[cfg(feature = "wgpu")]
                if self.xmap_mode != xmap.mode_table
                    || self.xmap_config != xmap.config
                    || self.xmap_cursor_cmap != xmap.cursor_cmap_msb
                    || self.xmap_popup_cmap != xmap.popup_cmap_msb
                {
                    Ng1DisplayGenerations::bump(&mut self.display_generations.xmap);
                }
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
            crate::dlog!(crate::devlog::LogModule::Rex3, "Rex3: Resolution changed to {}x{} cursor_x_adjust={}", w, h, cursor_x_adjust);
            self.width  = w;
            self.height = h;
            resized = true;
        }

        if decode_did {
            diag.fetch_or(Rex3::DIAG_LOOP_DECODE_DID, Ordering::Relaxed);
            let did_changed = self.decode_did();
            #[cfg(feature = "wgpu")]
            if did_changed {
                Ng1DisplayGenerations::bump(&mut self.display_generations.did);
            }
            #[cfg(not(feature = "wgpu"))]
            let _ = did_changed;
            diag.fetch_and(!Rex3::DIAG_LOOP_DECODE_DID, Ordering::Relaxed);
        }

        resized
    }

    pub(crate) fn refresh_display_state(
        &mut self,
        vc2: &Mutex<Vc2>,
        xmap: &Mutex<Xmap9>,
        cmap: &Mutex<Cmap>,
        bt445: &Mutex<Bt445>,
        diag: &AtomicU64,
    ) -> bool {
        self.refresh_display_state_inner(vc2, xmap, cmap, bt445, diag, true)
    }

    #[cfg(feature = "wgpu")]
    pub(crate) fn refresh_resident_display_state(
        &mut self,
        vc2: &Mutex<Vc2>,
        xmap: &Mutex<Xmap9>,
        cmap: &Mutex<Cmap>,
        bt445: &Mutex<Bt445>,
        diag: &AtomicU64,
    ) -> bool {
        self.refresh_display_state_inner(vc2, xmap, cmap, bt445, diag, false)
    }

    /// Borrow only display state. Geometry includes externally supplied
    /// topscan as well as the decoded dimensions and cursor adjustment.
    #[cfg(feature = "wgpu")]
    pub(crate) fn display_state(&mut self) -> Ng1DisplaySource<'_> {
        let geometry = (self.width, self.height, self.topscan, self.cursor_x_adjust);
        if self.display_geometry != geometry {
            Ng1DisplayGenerations::bump(&mut self.display_generations.geometry);
            self.display_geometry = geometry;
        }
        Ng1DisplaySource {
            generations: self.display_generations,
            did: &self.did,
            resident_did: &self.resident_did,
            xmap_mode: &self.xmap_mode,
            xmap_config: self.xmap_config,
            xmap_cursor_cmap: self.xmap_cursor_cmap,
            xmap_popup_cmap: self.xmap_popup_cmap,
            cmap: &self.cmap,
            ramdac: &self.ramdac_palette,
            vc2_ram: &self.vc2_ram,
            vc2_regs: &self.vc2_regs,
            topscan: self.topscan,
            cursor_x_adjust: self.cursor_x_adjust,
            width: self.width,
            height: self.height,
        }
    }

    pub fn compositor_source_from<'a>(
        &'a self,
        live_rgb: Option<&'a [u32]>,
        live_aux: Option<&'a [u32]>,
    ) -> CompositorSource<'a> {
        #[cfg(feature = "rex-dirty-upload")]
        let _ = (live_rgb, live_aux);
        CompositorSource {
            #[cfg(feature = "rex-dirty-upload")]
            fb_rgb: &self.fb_rgb,
            #[cfg(not(feature = "rex-dirty-upload"))]
            fb_rgb: if self.fb_borrowed {
                live_rgb.unwrap_or(&self.fb_rgb)
            } else {
                &self.fb_rgb
            },
            #[cfg(feature = "rex-dirty-upload")]
            fb_aux: &self.fb_aux,
            #[cfg(not(feature = "rex-dirty-upload"))]
            fb_aux: if self.fb_borrowed {
                live_aux.unwrap_or(&self.fb_aux)
            } else {
                &self.fb_aux
            },
            #[cfg(feature = "rex-dirty-upload")]
            fb_rgb_damage: &self.fb_rgb_damage,
            #[cfg(not(feature = "rex-dirty-upload"))]
            fb_rgb_damage: &[],
            #[cfg(feature = "rex-dirty-upload")]
            fb_aux_damage: &self.fb_aux_damage,
            #[cfg(not(feature = "rex-dirty-upload"))]
            fb_aux_damage: &[],
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
            prebuilt: if self.prebuilt {
                Some(&self.fb_rgb)
            } else {
                None
            },
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

/// What a VC2 or VC3 timing generator's tables add up to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Vc2Totals {
    /// Pixel clocks a line, blanking included.
    pub h_total: u32,
    /// Lines a frame, blanking included, over all of its fields.
    pub v_total: u32,
    /// Lines with any pixel out of blanking.
    pub visible_lines: u32,
    /// Runs of visible lines: 1, or 2 for an interlaced, a two-field or a
    /// field-sequential stereo format, each with its own vertical blank.
    pub fields: u32,
}

/// Total a VC2's video timing tables (the VC3's are laid out the same): the
/// frame table at the video entry pointer lists (line table, repeat count)
/// pairs up to a zero count; a line is runs of states, each lasting its
/// duration field in units of two pixel clocks, and ends with the address
/// of the line that follows. `visible` says whether a run's states (the
/// first word's seven A bits, and the C bits of the last second word) put
/// its line in the picture: the VC2's composite blank is a C bit, the VC3
/// marks its visible lines in an A bit. None while no table is loaded.
pub fn decode_vc2_totals(vc2_regs: &[u16; 32], ram: &[u16], visible_run: impl Fn(u8, u8) -> bool) -> Option<Vc2Totals> {
    let mut frame = vc2_regs[VC2_REG_VIDEO_ENTRY_PTR as usize] as usize;
    let (mut h_total, mut v_total, mut visible_lines, mut fields) = (0u32, 0u32, 0u32, 0u32);
    let mut in_picture = false;
    // The tables are the guest's: every walk is bounded.
    for _ in 0..1024 {
        let (mut line, mut repeat) = (*ram.get(frame)? as usize, *ram.get(frame + 1)? as usize);
        if repeat == 0 {
            break;
        }
        while repeat > 0 {
            let (mut clocks, mut visible, mut state_c) = (0u32, false, 0u8);
            for _ in 0..1024 {
                let w1 = *ram.get(line)?;
                line += 1;
                let mut eol = w1 & 0x8000 != 0;
                if w1 & 0x0080 == 0 {
                    let w2 = *ram.get(line)?;
                    line += 1;
                    eol |= w2 & 0x8000 != 0;
                    state_c = (w2 & 0x7F) as u8;
                }
                visible |= visible_run((w1 & 0x7F) as u8, state_c);
                clocks += 2 * ((w1 >> 8) & 0x7F) as u32;
                if eol {
                    break;
                }
            }
            h_total = h_total.max(clocks);
            v_total += 1;
            if visible {
                visible_lines += 1;
                fields += !in_picture as u32;
            }
            in_picture = visible;
            if v_total > 8192 {
                return None;
            }
            repeat -= 1;
            line = *ram.get(line)? as usize;
        }
        frame += 2;
    }
    (h_total != 0 && v_total != 0 && visible_lines != 0).then_some(Vc2Totals { h_total, v_total, visible_lines, fields })
}

/// Decode visible display size from VC2 state (used by compositor and timing presets).
pub fn decode_vc2_timings(vc2_regs: &[u16; 32], vc2_ram: &[u16]) -> (usize, usize, i32) {
    let (w, h, adjust, _) = decode_vc2_timings_gap(vc2_regs, vc2_ram);
    (w, h, adjust)
}

/// A cursor placement the video table does not give: the distance from the
/// HPOS edge to the first visible pixel was missing (`None`) or outside what
/// the cursor adjustment covers, so the default adjustment is used.
pub type CursorGapFault = Option<usize>;

/// [`decode_vc2_timings`], also saying when the cursor adjustment is the
/// fallback and why. It matters only while the cursor is shown.
pub fn decode_vc2_timings_gap(vc2_regs: &[u16; 32], vc2_ram: &[u16]) -> (usize, usize, i32, Option<CursorGapFault>) {
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
        let (cursor_x_adjust, fault) = match hpos_to_visible {
            // The NG2 table (a Newport in an expansion slot) has a 30-pixel
            // gap, so the adjustment goes slightly negative.
            Some(d) if (-16..=64).contains(&(d as i32 - 31)) => (d as i32 - 31, None),
            other => (11, Some(other)),
        };
        (w, h, cursor_x_adjust - 2, fault)
    } else {
        (0, 0, 0, None)
    }
}

/// Write `data`, tightly packed 8-bit `color` rows of `w` x `h`, as a PNG. It is
/// written to a temporary name and renamed, so a reader never sees half a file.
pub fn write_png(path: &std::path::Path, w: u32, h: u32, color: png::ColorType, data: &[u8]) -> std::io::Result<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    {
        let mut enc = png::Encoder::new(std::io::BufWriter::new(std::fs::File::create(&tmp)?), w, h);
        enc.set_color(color);
        enc.set_depth(png::BitDepth::Eight);
        enc.write_header()?.write_image_data(data)?;
    }
    std::fs::rename(&tmp, path)
}

/// Save the composited rgba buffer as a PNG file.
/// `rgba` uses 0xFFBBGGRR (GL-native); swap R↔B when writing PNG.
pub fn save_screenshot(path: &str, rgba: &[u32], width: usize, height: usize) -> Result<(), String> {
    let mut rows: Vec<u8> = Vec::with_capacity(height * width * 3);
    for y in 0..height {
        for x in 0..width {
            let px = rgba[y * 2048 + x];
            rows.push(( px        & 0xFF) as u8);
            rows.push(((px >>  8) & 0xFF) as u8);
            rows.push(((px >> 16) & 0xFF) as u8);
        }
    }
    write_png(std::path::Path::new(path), width as u32, height as u32, png::ColorType::Rgb, &rows).map_err(|e| e.to_string())
}
