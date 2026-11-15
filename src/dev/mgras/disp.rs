//! Display side: the board on the display refresh thread (vertical
//! retrace, then a frame snapshot and composition, see `frame`), host GL
//! frame compositing, and PNG output.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use super::dcb::Dcb;
use super::rss::{self, Rss};
use super::frame::{self, Frame};
use super::{microcode, plain, Line, Mgras};

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
    let pair = frame::pair_format(dcb.xmap.main_mode(target as u32));
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
                let v = match pair {
                    Some(p) => p.pack_with(v | 0xFF << 24, None),
                    None => v,
                };
                rss.mem.put(&main, sx as u32, fb_y, v as u64);
            }
        }
    }
    true
}

/// The window ID a colour-index frame's window has, in the main planes (a
/// colour-index display mode) or the overlay planes (one that is on): the
/// commonest such ID over a grid of samples inside the window, as
/// `window_did` -- and only when it covers three quarters of the samples on
/// the screen. A window just mapped may not have its ID in the tables yet,
/// and the commonest ID around it is then someone else's: the frame went
/// there, the program was told it was shown, and gltest's overlay read back
/// 0s now and then. None, and the program puts the frame up itself.
fn index_window_did(dcb: &Dcb, x: i32, y: i32, w: usize, h: usize, overlay: bool) -> Option<u8> {
    let mut total = 0u32;
    let mut counts = [0u32; 32];
    let mut runs = Vec::new();
    let (screen_w, screen_h) = frame::display_size(dcb);
    for sy in 0..16 {
        let py = y + (h as i32 * (2 * sy + 1)) / 32;
        if !(0..screen_h as i32).contains(&py) {
            continue;
        }
        if overlay {
            dcb.vc3.overlay_did_runs(py as usize, &mut runs);
        } else {
            dcb.vc3.main_did_runs(py as usize, &mut runs);
        }
        for sx in 0..16 {
            let px = x + (w as i32 * (2 * sx + 1)) / 32;
            if !(0..screen_w as i32).contains(&px) {
                continue;
            }
            let did = runs.iter().rev().find(|r| r.0 as i32 <= px).map_or(0, |r| r.1) as u32;
            let index = if overlay { dcb.xmap.overlay_mode(did) != 0 } else { dcb.xmap.main_mode(did) & 0x1F < 4 };
            total += 1;
            if index {
                counts[did as usize & 31] += 1;
            }
        }
    }
    let (did, n) = counts.iter().enumerate().max_by_key(|&(_, n)| *n)?;
    (*n > 0 && *n * 4 >= total * 3).then_some(did as u8)
}

/// Paint a colour-index window's host GL frame into the framebuffer, as
/// `composite` does an RGB one, but as indices: each pixel's index is its
/// red byte and the low nibble of its green (`bgra`: B, G, R, A), and goes
/// where the board keeps that window's indices -- the low 8 or 12 main
/// planes (by the window ID's display mode), the high 12 when the ID shows
/// buffer B (`frame::Frame::snapshot`), or the overlay planes for an overlay
/// window -- for the colour map to show, as GL's drawing would on the board.
/// Before this, the program read every frame back, turned it into indices
/// and put it to the X server itself: SGI's IRIS GL on host OpenGL ran
/// flight's full-screen colour-index window at a frame every few seconds.
/// False when no window ID fits, or there are no overlay planes.
#[allow(clippy::too_many_arguments)]
pub fn composite_index(rss: &mut Rss, dcb: &Dcb, x: i32, y: i32, bgra: &[u8], stride: usize, w: usize, h: usize, overlay: bool) -> bool {
    let (main, over) = frame::scanout_buffers(rss, dcb);
    let buf = match (overlay, over) {
        (false, _) => main,
        (true, Some(o)) => o,
        (true, None) => return false,
    };
    let Some(target) = index_window_did(dcb, x, y, w, h, overlay) else { return false };
    let b_half = !overlay && dcb.xmap.buf_select() >> target & 1 != 0;
    let (mask, shift): (u64, u32) = if overlay {
        (0xFF, 0)
    } else if b_half {
        (0xFFF << 12, 12)
    } else if dcb.xmap.main_mode(target as u32) & 0x1F == 1 {
        (0xFF, 0)
    } else {
        (0xFFF, 0)
    };
    let mut runs = Vec::new();
    let (screen_w, screen_h) = frame::display_size(dcb);
    for row in 0..h {
        let sy = y + row as i32;
        if !(0..screen_h as i32).contains(&sy) {
            continue;
        }
        if overlay {
            dcb.vc3.overlay_did_runs(sy as usize, &mut runs);
        } else {
            dcb.vc3.main_did_runs(sy as usize, &mut runs);
        }
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
                let index = p[2] as u64 | (p[1] as u64 & 0xF) << 8;
                let old = rss.mem.get(&buf, sx as u32, fb_y);
                rss.mem.put(&buf, sx as u32, fb_y, (old & !mask) | (index << shift & mask));
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

/// What the refresh thread keeps between its callbacks.
pub(super) struct Scan {
    snap: Box<Frame>,
    /// The frame `capture` took, for `compose`.
    taken: Taken,
    /// The retrace line is up.
    retrace: bool,
    frames: u32,
    /// Microcode fingerprints: checked every two seconds, logged once the
    /// same value is seen twice in a row and differs from the last one
    /// logged.
    microcode_seen: microcode::Fingerprints,
    microcode_logged: microcode::Fingerprints,
    last_rec_flush: std::time::Instant,
}

enum Taken {
    None,
    /// The planes and display state, in `snap`.
    Snapshot,
    /// Composed on the GPU and read back into `screen.fb_rgb`.
    Read(usize, usize),
    /// Composed on the GPU, for a renderer on the same device.
    #[cfg(feature = "wgpu")]
    Gpu(::wgpu::Buffer, u32, u32),
}

impl Scan {
    pub(super) fn new() -> Self {
        Scan {
            snap: plain::boxed_zeroed::<Frame>(),
            taken: Taken::None,
            retrace: false,
            frames: 0,
            microcode_seen: Default::default(),
            microcode_logged: Default::default(),
            last_rec_flush: std::time::Instant::now(),
        }
    }
}

impl crate::display::Head for Mgras {
    /// With the wgpu window on the GPU unit's device the frame is composed
    /// and presented there every frame (cheap; the cursor and the last
    /// batches show without a dirty mark).
    fn changed(&self) -> bool {
        self.presents_on_gpu() && self.head.want().gpu
    }

    /// The frame is taken after the blank: the kernel's retrace handler
    /// has flipped buffers (XMAP buffer select) and updated the window-ID
    /// tables, and the client draws its next frame in the buffer that is
    /// not shown.
    fn capture(&self, screen: &mut crate::disp::Rex3Screen, want: crate::display::Want) -> bool {
        let mut scan = self.scan.lock();
        // SAFETY: a read-only view of the framebuffer; tearing tolerated.
        let rss = unsafe { &*self.rss.get() };
        crate::pf_begin!(crate::perfetto::events::MGRAS_CAPTURE, self.rss_fifo.len());
        scan.taken = Taken::None;
        #[cfg(feature = "wgpu")]
        if want.gpu && self.presents_on_gpu() {
            // SAFETY: the scan-out is the refresh thread's; the registers
            // and the DCB are read racily, as the CPU snapshot reads them.
            if let Some(sc) = (unsafe { &mut *self.gpu_scanout.get() }).as_mut() {
                match sc.compose(rss, &self.front_view().dcb) {
                    Ok((buffer, w, h)) => {
                        self.gpu_frames[0].fetch_add(1, Ordering::Relaxed);
                        scan.taken = Taken::Gpu(buffer, w, h);
                    }
                    Err(e) => eprintln!("mgras: GPU scan-out: {e}"),
                }
            }
        }
        let _ = want;
        if matches!(scan.taken, Taken::None) {
            scan.taken = match self.compose_gpu_into(rss, &mut screen.fb_rgb) {
                Some((w, h)) => Taken::Read(w, h),
                None => {
                    scan.snap.snapshot(rss, &self.front_view().dcb);
                    Taken::Snapshot
                }
            };
        }
        crate::pf_end!(crate::perfetto::events::MGRAS_CAPTURE);
        true
    }

    fn vblank_start(&self, _field: u32, _t_ns: u64) {
        self.begin_frame();
        // Video timing chip display control bit 0: retrace interrupts. The
        // handler acknowledges nothing on the board, so the line drops
        // again by itself, at the end of the blank.
        if self.front_view().dcb.vc3.regs[0x1E] & 1 != 0 {
            crate::pf_instant!(crate::perfetto::events::MGRAS_RETRACE);
            self.scan.lock().retrace = true;
            self.set_line(Line::Retrace, true);
        }
    }

    fn vblank_end(&self, _field: u32, _t_ns: u64) {
        let mut scan = self.scan.lock();
        if std::mem::take(&mut scan.retrace) {
            self.set_line(Line::Retrace, false);
        }
        if scan.last_rec_flush.elapsed() >= std::time::Duration::from_millis(500) {
            if let Some(rec) = self.mem_rec.lock().clone() {
                rec.lock().flush();
            }
            scan.last_rec_flush = std::time::Instant::now();
        }
        self.trace_flush(false);
        scan.frames = scan.frames.wrapping_add(1);
        if scan.frames % 120 == 0 {
            let fp = self.microcode_fingerprints();
            if fp.hq != 0 && fp == scan.microcode_seen && fp != scan.microcode_logged {
                eprintln!("impact: microcode {fp}");
                scan.microcode_logged = fp;
            }
            scan.microcode_seen = fp;
        }
    }

    fn compose(&self, screen: &mut crate::disp::Rex3Screen, _want: crate::display::Want) -> crate::display::Composed {
        use crate::display::Composed;
        let mut scan = self.scan.lock();
        screen.prebuilt = true;
        match std::mem::replace(&mut scan.taken, Taken::None) {
            Taken::None => Composed::Unchanged,
            Taken::Snapshot => {
                scan.snap.compose(&mut screen.fb_rgb);
                (screen.width, screen.height) = (scan.snap.width, scan.snap.height);
                Composed::Frame
            }
            Taken::Read(w, h) => {
                (screen.width, screen.height) = (w, h);
                Composed::Frame
            }
            #[cfg(feature = "wgpu")]
            Taken::Gpu(frame, width, height) => {
                (screen.width, screen.height) = (width as usize, height as usize);
                Composed::Gpu { frame, width, height }
            }
        }
    }

    /// Both queues: the HQ3's usually drains at once, and the backlog waits
    /// in front of the raster engines.
    fn gfifo_pending(&self) -> usize {
        self.hq_fifo.len() + self.rss_fifo.len()
    }
}

impl Mgras {
    /// With the GPU unit: compose the displayed frame on the GPU and read
    /// it back into `out` (`frame.rs`'s layout); its size. None on the CPU
    /// raster path, or if the scan-out fails (the planes are then composed
    /// on the CPU, current as of the last sync).
    fn compose_gpu_into(&self, rss: &Rss, out: &mut [u32]) -> Option<(usize, usize)> {
        #[cfg(feature = "wgpu")]
        {
            // SAFETY: the scan-out is the display thread's (and this runs
            // there, or from the monitor with the display stopped).
            let sc = unsafe { &mut *self.gpu_scanout.get() }.as_mut()?;
            match sc.read(rss, &self.front_view().dcb, out, frame::OUT_STRIDE) {
                Ok(size) => {
                    self.gpu_frames[1].fetch_add(1, Ordering::Relaxed);
                    return Some(size);
                }
                Err(e) => eprintln!("mgras: GPU scan-out: {e}"),
            }
        }
        let _ = (rss, out);
        None
    }

    /// Start the engines and the display.
    pub fn start_display(self: &Arc<Self>) {
        self.start_engines();
        if self.running.swap(true, Ordering::AcqRel) {
            return;
        }
        {
            let mut screen = self.head.screen().lock();
            (screen.width, screen.height) = frame::DEFAULT_SIZE;
            screen.fb_rgb.fill(0xFF00_0000);
            screen.prebuilt = true;
        }
        self.head.start();
    }

    pub fn stop_display(&self) {
        // The renderer's GL state belongs to the refresh thread (its
        // context is current there), which lets go of it here.
        self.head.stop();
        self.running.store(false, Ordering::Release);
        self.stop_engines();
    }
}
