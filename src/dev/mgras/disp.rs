//! Display side: the refresh thread (vertical retrace, then a frame
//! snapshot and composition, see `frame`), host GL frame compositing, and
//! PNG output.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use super::dcb::Dcb;
use super::rss::{self, Rss};
use super::frame::{self, Frame};
use super::{plain, Line, Mgras};

/// The window ID a frame for the window at (`x`, `y`), `w` x `h` (screen
/// coordinates, top-down) should be painted through: the commonest ID
/// over a grid of samples inside it that has an RGB display mode. None
/// if no sampled pixel is in an RGB window.
pub fn window_did(rss: &Rss, dcb: &Dcb, x: i32, y: i32, w: usize, h: usize) -> Option<u8> {
    let mut counts = [0u32; 32];
    let mut runs = Vec::new();
    let (screen_w, screen_h) = frame::display_size(dcb);
    for sy in 0..16 {
        let py = y + (h as i32 * (2 * sy + 1)) / 32;
        if !(0..screen_h as i32).contains(&py) {
            continue;
        }
        dcb.vc3.main_did_runs(py as usize, &mut runs);
        for sx in 0..16 {
            let px = x + (w as i32 * (2 * sx + 1)) / 32;
            if !(0..screen_w as i32).contains(&px) {
                continue;
            }
            let did = runs.iter().rev().find(|r| r.0 as i32 <= px).map_or(0, |r| r.1);
            if dcb.xmap.main_mode(did as u32) & 0x1F >= 4 {
                counts[did as usize & 31] += 1;
            }
        }
    }
    let (did, n) = counts.iter().enumerate().max_by_key(|&(_, n)| *n)?;
    (*n > 0).then_some(did as u8)
}

/// Paint a host GL frame (`bgra`: `h` rows, top first, `stride` bytes
/// each) into the framebuffer at screen position (`x`, `y`), wherever the
/// pixel belongs to the frame's window ID, so windows over it stay over it.
/// The pixels become part of the framebuffer, as GL's would on the board,
/// for anything that reads them back. False when no window ID fits.
pub fn composite(rss: &mut Rss, dcb: &Dcb, x: i32, y: i32, bgra: &[u8], stride: usize, w: usize, h: usize) -> bool {
    let Some(target) = window_did(rss, dcb, x, y, w, h) else { return false };
    let mut runs = Vec::new();
    let (screen_w, screen_h) = frame::display_size(dcb);
    let (main, _) = frame::scanout_buffers(rss, dcb);
    // A 12-bit pair window shows the frame in both buffers.
    let pair = frame::rgb12_format(dcb.xmap.main_mode(target as u32));
    for row in 0..h {
        let sy = y + row as i32;
        if !(0..screen_h as i32).contains(&sy) {
            continue;
        }
        dcb.vc3.main_did_runs(sy as usize, &mut runs);
        if runs.is_empty() || runs[0].0 != 0 {
            runs.insert(0, (0, 0));
        }
        let fb_y = (screen_h - 1 - sy as usize) as u32;
        for (k, &(x0, did)) in runs.iter().enumerate() {
            if did != target {
                continue;
            }
            let x1 = runs.get(k + 1).map_or(screen_w as i32, |r| r.0 as i32);
            let lo = (x0 as i32).max(x).max(0);
            let hi = x1.min(x + w as i32).min(screen_w as i32);
            for sx in lo..hi {
                let i = row * stride + (sx - x) as usize * 4;
                let Some(p) = bgra.get(i..i + 4) else { break };
                let v = p[2] as u32 | (p[1] as u32) << 8 | (p[0] as u32) << 16;
                let v = if pair { rss::to_rgb12(v) * 0x1001 } else { v };
                rss.mem.put(&main, sx as u32, fb_y, v as u64);
            }
        }
    }
    true
}

/// Write a scanned-out frame (`0xFF_BB_GG_RR`, stride `OUT_STRIDE`) as a PNG.
pub fn save_png(path: &str, frame: &[u32], width: usize, height: usize) -> Result<(), String> {
    let file = std::fs::File::create(path).map_err(|e| e.to_string())?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), width as u32, height as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let mut out = enc.write_header().map_err(|e| e.to_string())?;
    let mut rows = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for px in &frame[y * frame::OUT_STRIDE..y * frame::OUT_STRIDE + width] {
            rows.extend_from_slice(&[*px as u8, (px >> 8) as u8, (px >> 16) as u8]);
        }
    }
    out.write_image_data(&rows).map_err(|e| e.to_string())
}

impl Mgras {
    pub(super) fn refresh_loop(self: &Arc<Self>) {
        let frame = std::time::Duration::from_micros(16_667);
        {
            let mut screen = self.screen.lock();
            (screen.width, screen.height) = frame::DEFAULT_SIZE;
            screen.fb_rgb.fill(0xFF00_0000);
            screen.prebuilt = true;
        }
        let mut overlay = crate::debug_overlay::DebugOverlay::new();
        let mut status_bar = crate::disp::StatusBar::new();
        let mut sbtex = crate::disp::StatusBarTexture::new();
        // Size the renderer was last given; the guest can change modes.
        let mut shown = (0usize, 0usize);
        let mut idle_frames = 0u32;
        let mut shots = 0u32;
        let mut last_rec_flush = std::time::Instant::now();
        let mut snap = plain::boxed_zeroed::<Frame>();
        const PERSISTENT: u64 = crate::dev::ng1::rex3::Rex3::HB_LED_RED | crate::dev::ng1::rex3::Rex3::HB_LED_GREEN;

        while self.running.load(Ordering::Relaxed) {
            let start = std::time::Instant::now();
            let stats = crate::disp::BarStats {
                now: start,
                hb: self.heartbeat.fetch_and(PERSISTENT, Ordering::Relaxed),
                cycles: self.cycles.lock().get(),
                fasttick: self.fasttick.load(Ordering::Relaxed),
                decoded_delta: 0,
                l1i_hits: 0,
                l1i_fetches: 0,
                uncached: 0,
                count_hz: 0,
                gfifo_pending: 0,
            };
            if last_rec_flush.elapsed() >= std::time::Duration::from_millis(500) {
                if let Some(rec) = self.mem_rec.lock().clone() {
                    rec.lock().flush();
                }
                last_rec_flush = std::time::Instant::now();
            }
            self.trace_flush(false);
            // Video timing chip display control bit 0: retrace interrupts.
            let retrace = self.front_view().dcb.vc3.regs[0x1E] & 1 != 0;
            // Vertical retrace: a pulse per frame. The handler acknowledges
            // nothing on the board, so the line must drop again by itself.
            if retrace {
                self.set_line(Line::Retrace, true);
                std::thread::sleep(std::time::Duration::from_micros(500));
                self.set_line(Line::Retrace, false);
            }
            let dirty = self.dirty.swap(false, Ordering::AcqRel);
            let shot = self.screenshot_pending.swap(false, Ordering::Relaxed);
            idle_frames += 1;
            if dirty || shot || idle_frames >= 6 {
                idle_frames = 0;
                let mut screen = self.screen.lock();
                if dirty || shot {
                    let screen = &mut *screen;
                    // The snapshot follows the retrace pulse, so the guest's
                    // retrace-time updates (window-ID tables, swaps) are in.
                    // SAFETY: a read-only view of the framebuffer; tearing
                    // tolerated.
                    let rss = unsafe { &*self.rss.get() };
                    snap.snapshot(rss, &self.front_view().dcb);
                    snap.compose(&mut screen.fb_rgb);
                    // The frame is already final RGB, so keep `rgba` (what CI
                    // screenshots read) current without a renderer readback,
                    // as GR2 does. This also makes screenshots work headless.
                    let (w, h) = (snap.width, snap.height);
                    for y in 0..h {
                        let row = y * 2048;
                        screen.rgba[row..row + w].copy_from_slice(&screen.fb_rgb[row..row + w]);
                    }
                    screen.width = w;
                    screen.height = h;
                    screen.status_bar_only = false;
                } else {
                    screen.status_bar_only = true;
                }
                if let Some(r) = self.renderer.lock().as_mut() {
                    if shown != (screen.width, screen.height) {
                        shown = (screen.width, screen.height);
                        r.resize(shown.0, shown.1);
                    }
                    r.present(&mut screen, &mut overlay, &mut status_bar, &mut sbtex, &stats, shot, None, None);
                }
                // The screenshot hotkey (Right Ctrl + Print Screen), as REX3
                // and GR2 do: the frame just composed, written off-thread.
                if shot {
                    let (pixels, w, h) = (screen.fb_rgb.clone(), screen.width, screen.height);
                    let path = format!("screenshot_{shots:04}.png");
                    shots += 1;
                    std::thread::spawn(move || match save_png(&path, &pixels, w, h) {
                        Ok(()) => println!("iris: screenshot saved to {path}"),
                        Err(e) => println!("iris: screenshot failed: {e}"),
                    });
                }
            }
            if let Some(rest) = frame.checked_sub(start.elapsed()) {
                std::thread::sleep(rest);
            }
        }
        // The renderer's GL state belongs to this thread (its context is
        // current here), so it is torn down here and nowhere else.
        if let Some(r) = self.renderer.lock().as_mut() {
            r.stop();
        }
    }

    /// Start the engines and the display refresh thread.
    pub fn start_display(self: &Arc<Self>) {
        self.start_engines();
        if self.running.swap(true, Ordering::AcqRel) {
            return;
        }
        let me = Arc::clone(self);
        *self.refresh.lock() = Some(
            std::thread::Builder::new()
                .name("MGRAS-Refresh".into())
                .spawn(move || me.refresh_loop())
                .expect("spawn MGRAS refresh thread"),
        );
    }

    pub fn stop_display(&self) {
        self.running.store(false, Ordering::Release);
        if let Some(h) = self.refresh.lock().take() {
            let _ = h.join();
        }
        self.stop_engines();
    }
}
