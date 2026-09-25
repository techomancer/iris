//! GR2 display thread: vertical retrace, composition, presentation.
//!
//! Every ~16.7 ms the thread pulses the retrace line. The IOC latches it until
//! the host acknowledges: PORT_CONFIG on Indigo2, HPC3_GEN_CONTROL on Indy
//! (GR2.h "INTERRUPTS"). When something visible changed, or on a periodic
//! heartbeat for the status bar, it composes VRAM into `screen.fb_rgb`
//! (`prebuilt`) and presents through the installed renderer.

use std::sync::atomic::Ordering;
use std::thread;
use std::time::{Duration, Instant};

use super::gr2comp;
use super::re3::{FB_H, FB_W};
use super::Gr2;

/// Vertical blanking interval.
const VBLANK: Duration = Duration::from_micros(700);
/// Minimum refresh rate while idle, so the status bar keeps moving.
const IDLE_HEARTBEAT_FRAMES: u32 = 6;
/// Heartbeat bits that persist across frames (front-panel LEDs); same as REX3.
const HB_PERSISTENT: u64 = crate::rex3::Rex3::HB_PERSISTENT;

impl Gr2 {
    pub(super) fn display_loop(&self) {
        let frame = Duration::from_micros(16667);
        let mut status_bar = crate::disp::StatusBar::new();
        let mut overlay = crate::debug_overlay::DebugOverlay::new();
        let mut sbtex = crate::disp::StatusBarTexture::new();
        let mut frames_since_render = u32::MAX;
        let mut sized = false;

        while self.running.load(Ordering::Relaxed) {
            let start = Instant::now();

            // Vertical retrace. The interrupt stays latched in the IOC until
            // the guest clears it. The vertical status level (EXTIO SG_STAT_0)
            // is held for the blanking interval (~40 of 1066 lines at 60 Hz),
            // long enough for a polling CPU to see both edges.
            if let Some(cb) = self.retrace_cb.lock().clone() {
                cb(true);
                thread::sleep(VBLANK);
                cb(false);
            }

            let dirty = self.dirty.swap(false, Ordering::Acquire);
            let shot = self.screenshot_pending.load(Ordering::Relaxed);
            if dirty || shot || frames_since_render >= IDLE_HEARTBEAT_FRAMES {
                frames_since_render = 0;
                let stats = crate::disp::BarStats {
                    now: start,
                    hb: self.stats.heartbeat.fetch_and(HB_PERSISTENT, Ordering::Relaxed),
                    cycles: self.cycles.get().get(),
                    fasttick: self.stats.fasttick.load(Ordering::Relaxed),
                    decoded_delta: 0,
                    l1i_hits: 0,
                    l1i_fetches: 0,
                    uncached: 0,
                    count_hz: 0,
                    gfifo_pending: self.hq_fifo.len(),
                };

                let mut screen = self.screen.lock();
                if dirty || !sized {
                    let r = self.regs();
                    gr2comp::compose(self.vram(), &r.vc1, &r.xmap, &r.dac, &mut screen.fb_rgb);
                    // The frame is already final RGB, so keep `rgba` (what CI
                    // screenshots read) current without needing a renderer
                    // readback. This also makes screenshots work headless.
                    let screen = &mut *screen;
                    for y in 0..FB_H {
                        let row = y * gr2comp::OUT_STRIDE;
                        screen.rgba[row..row + FB_W].copy_from_slice(&screen.fb_rgb[row..row + FB_W]);
                    }
                }
                screen.prebuilt = true;
                screen.width = FB_W;
                screen.height = FB_H;
                screen.status_bar_only = !dirty && sized && !shot;

                let take_shot = self.screenshot_pending.swap(false, Ordering::Relaxed);
                let mut renderer = self.renderer.lock();
                if let Some(r) = renderer.as_mut() {
                    if !sized {
                        r.resize(FB_W, FB_H);
                    }
                    r.present(&mut screen, &mut overlay, &mut status_bar, &mut sbtex, &stats, take_shot, None, None);
                }
                sized = true;

                if take_shot {
                    let pixels = screen.rgba.clone();
                    let n = self.screenshot_counter.fetch_add(1, Ordering::Relaxed);
                    let path = format!("screenshot_{:04}.png", n);
                    thread::spawn(move || match crate::disp::save_screenshot(&path, &pixels, FB_W, FB_H) {
                        Ok(()) => println!("iris: screenshot saved to {}", path),
                        Err(e) => println!("iris: screenshot failed: {}", e),
                    });
                }
            } else {
                frames_since_render = frames_since_render.saturating_add(1);
            }

            self.trace_flush(false);

            let elapsed = start.elapsed();
            if elapsed < frame {
                thread::sleep(frame - elapsed);
            }
        }

        if let Some(r) = self.renderer.lock().as_mut() {
            r.stop();
        }
    }
}
