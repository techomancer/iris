//! Display side: the board on the display refresh thread (vertical
//! retrace, then a frame snapshot and composition, see `frame`), host GL
//! frame compositing, and PNG output.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use super::dcb::Dcb;
use super::pixmem::{Buffer, Kind, TILE_H};
use super::rss::{self, Rss};
use super::frame::{self, Frame};
use crate::display::Mark;
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

/// A buffer as a draw mark's store: its page pointer and its kind.
pub(super) fn store_key(b: &Buffer) -> u64 {
    b.ptr as u64 | ((b.kind == Kind::Overlay) as u64) << 16
}

/// The planes of a pair word the display reads for buffer `b`: those its
/// colour depends on.
fn pair_planes(pair: rss::Pair, b: bool) -> u64 {
    let shown = |w: u32| pair.unpack(w, b) & 0xFF_FFFF;
    (0..32).filter(|i| shown(1 << i) != shown(0)).fold(0, |m, i| m | 1 << i)
}

/// Whether any drawing in `marks` (`Rss::drawn`) shows in the frame the
/// display state describes: a mark shows when some pixel in its box
/// belongs to a window ID whose displayed buffer is the page it drew in
/// and reads a plane it could change. A double-buffered window's drawing
/// goes to the buffer its window ID does not show (the other page, or the
/// other half of the pixel pair), so it does not; the kernel's flip at the
/// retrace changes the display state, which is damage of its own.
pub(super) fn marks_show(marks: &[Mark], rss: &Rss, dcb: &Dcb) -> bool {
    if marks.is_empty() {
        return false;
    }
    let (w, h) = frame::display_size(dcb);
    let (main, _) = frame::scanout_buffers(rss, dcb);
    let back = Buffer::new(dcb.xmap.dib_pointers() >> 10, Kind::Wide, rss.reg(rss::reg::DRBSIZE));
    let select = dcb.xmap.buf_select();
    // Pages a displayed buffer runs over from its pointer: a buffer drawn
    // at another pointer inside them shares memory with it.
    let span = ((h.div_ceil(1 << main.yshift)).div_ceil(TILE_H) * main.xtiles as usize) as u32;
    let near = |a: u32, b: u32| {
        let d = a.wrapping_sub(b) & 0x3FF;
        d < span || 0x400 - d < span
    };
    // Per window ID: the page its pixels are read from, and the planes.
    let mut reads: [Option<(u32, u64)>; 32] = [None; 32];
    let mut read = |did: usize| {
        *reads[did].get_or_insert_with(|| {
            let mode = frame::MainMode::of(dcb.xmap.main_mode(did as u32));
            let b = select >> did & 1 != 0;
            match mode.pair {
                Some(pair) => (main.ptr, pair_planes(pair, b)),
                None if !mode.rgb => (main.ptr, if b { 0xFFF000 } else { mode.index_mask as u64 }),
                None if b => (back.ptr, 0xFF_FFFF),
                None => (main.ptr, 0xFF_FFFF),
            }
        })
    };
    let mut runs = Vec::with_capacity(64);
    let mut row = [0u8; frame::W];
    let screen = crate::display::Rect::new(0, 0, w as i32 - 1, h as i32 - 1);
    for m in marks {
        if m.store >> 16 != 0 {
            // The overlay planes: shown wherever they are on.
            return true;
        }
        let ptr = m.store as u32 & 0x3FF;
        let (on_main, on_back) = (ptr == main.ptr, ptr == back.ptr);
        if (!on_main && near(ptr, main.ptr)) || (!on_back && select != 0 && near(ptr, back.ptr)) {
            return true;
        }
        if !on_main && !on_back {
            // Another buffer altogether (depth, a pbuffer).
            continue;
        }
        let r = m.rect.intersect(&screen);
        if r.is_empty() {
            continue;
        }
        for fb_y in r.y0..=r.y1 {
            // The screen's top row is framebuffer row height - 1.
            dcb.vc3.main_did_runs(h - 1 - fb_y as usize, &mut runs);
            frame::fill_did_row(&mut row, &mut runs);
            let mut last = usize::MAX;
            for &did in &row[r.x0 as usize..=r.x1 as usize] {
                let did = did as usize & 31;
                if did != last {
                    last = did;
                    let (page, planes) = read(did);
                    if page == ptr && planes & m.planes != 0 {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// What the refresh thread keeps between its callbacks.
pub(super) struct Scan {
    /// The draw marks of the frame, taken by `changed`.
    marks: Vec<Mark>,
    /// The raster FIFO's consumed count at the last frame.
    consumed: usize,
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
    /// Nothing to take: `compose` reads the pixel pages themselves.
    Pages,
    /// Composed on the GPU and read back into `screen.fb_rgb`.
    Read(usize, usize),
    /// Composed on the GPU, for a renderer on the same device.
    #[cfg(feature = "wgpu")]
    Gpu(::wgpu::Buffer, u32, u32),
}

impl Scan {
    pub(super) fn new() -> Self {
        Scan {
            marks: Vec::new(),
            consumed: 0,
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
    /// Whether the raster engines drew into something the screen shows
    /// (`marks_show`), or are in the middle of something long; the display
    /// state's own changes are the head's damage. With the wgpu window on the GPU unit's device the frame is
    /// composed and presented there every frame (cheap; the cursor and the
    /// last batches show without a mark).
    fn changed(&self) -> bool {
        let mut scan = self.scan.lock();
        let any = self.marks.take(&mut scan.marks);
        // A command marks when it is done. One that has kept the engines
        // for a whole frame (the X server's full-screen fills take several)
        // is shown as far as it has got, as the board's scan-out would.
        let consumed = self.rss_fifo.consumed();
        let held = consumed == scan.consumed && self.rss_fifo.len() != 0;
        scan.consumed = consumed;
        if any || held || (self.presents_on_gpu() && self.head.want().gpu) {
            return true;
        }
        // SAFETY: a read of one register (the buffers' size), racing the
        // engines as the frame's snapshot does.
        let rss = unsafe { &*self.rss.get() };
        marks_show(&scan.marks, rss, &self.front_view().dcb)
    }

    /// The frame is made after the blank: the kernel's retrace handler has
    /// flipped buffers (XMAP buffer select) and updated the window-ID
    /// tables, and the client draws its next frame in the buffer that is
    /// not shown. On the CPU there is nothing to take first: `compose`
    /// reads the pixel pages and the display state as they are, before the
    /// next retrace can change either. With the GPU unit the frame is
    /// composed there, here.
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
                None => Taken::Pages,
            };
        }
        crate::pf_end!(crate::perfetto::events::MGRAS_CAPTURE);
        true
    }

    fn vblank_start(&self, _field: u32, _t_ns: u64) {
        self.begin_frame();
        // The refresh rate is the guest's: the VC3's totals and the DAC's
        // pixel clock, decoded when the display bus was written. A timing
        // the thread cannot run (tables half loaded) leaves the last one.
        if self.timing_stale.swap(false, Ordering::AcqRel) {
            if let Some(timing) = self.front_view().dcb.video_timing() {
                self.head.set_timing(timing);
            }
        }
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
            Taken::Pages => {
                // SAFETY: a read-only view of the framebuffer, read while
                // the engines draw, as the board's scan-out reads it.
                let rss = unsafe { &*self.rss.get() };
                (screen.width, screen.height) = frame::compose_direct(rss, &self.front_view().dcb, &mut screen.fb_rgb);
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

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::rss::reg;
    use crate::display::{Marks, Rect};

    /// A 1024x768 screen (two raster engines, a buffer 144 pages) of
    /// window ID 0 with window ID `did` at (100, 100)-(299, 299), buffer A
    /// at page 0x1C0 and B at 0x2E0.
    fn screen(did: u8, mode: u32, select: u32) -> (Box<Rss>, Box<Dcb>) {
        let mut r = Rss::new_boxed();
        r.write(reg::DRBSIZE, 0x9000_025A, false);
        r.write(reg::DRBPOINTERS, 0x1C0 | 0x2E0 << 10, false);
        let mut dcb = plain::boxed_zeroed::<Dcb>();
        dcb.xmap.set_modes(0, 0x15, 0);
        dcb.xmap.set_modes(did as u32, mode, 0);
        dcb.xmap.set_scanout(select, 0x1C0 | 0x2E0 << 10 | 0x40 << 20, 0);
        let vc3 = &mut dcb.vc3;
        vc3.regs[4] = 0;
        // Line tables: the desktop alone, and with the window across it.
        vc3.sram[0x2000] = 0;
        vc3.sram[0x2001] = 0x7FF << 5;
        vc3.sram[0x2040] = 0;
        vc3.sram[0x2041] = 100 << 5 | did as u16;
        vc3.sram[0x2042] = 300 << 5;
        vc3.sram[0x2043] = 0x7FF << 5;
        for y in 0..768 {
            vc3.sram[y] = if (100..300).contains(&y) { 0x2040 } else { 0x2000 };
        }
        vc3.sram[768] = 0xFFFF;
        (r, dcb)
    }

    /// The marks of a primitive under fill mode `pp1` clipped to display
    /// box (x0, y0)-(x1, y1), and whether they show.
    fn drawn_shows(r: &mut Rss, dcb: &Dcb, pp1: u32, (x0, y0, x1, y1): (u32, u32, u32, u32)) -> bool {
        r.write(reg::PP1FILLMODE, pp1, false);
        r.write(reg::CLIP_MODE, 0x11, false);
        r.write(reg::CLIP_X, x0 << 16 | x1, false);
        // The screen's top row is framebuffer row 767.
        r.write(reg::CLIP_Y, (767 - y1) << 16 | (767 - y0), false);
        let marks = Marks::new();
        r.drawn(|b, planes, rect| marks.mark(store_key(&b), planes, rect));
        let mut taken = Vec::new();
        assert!(!marks.take(&mut taken));
        marks_show(&taken, r, dcb)
    }

    const WINDOW: (u32, u32, u32, u32) = (100, 100, 299, 299);

    #[test]
    fn a_24_bit_window_s_back_page_does_not_show() {
        // RGBA8888 through all planes; draw field 1 is page A, 4 page B.
        let (a, b, both) = (0x200 | 1 << 14, 0x200 | 4 << 14, 0x200 | 5 << 14);
        let (mut r, dcb) = screen(5, 0x15, 0);
        r.write(reg::COLORMASKLSBSA, u32::MAX, false);
        r.write(reg::COLORMASKLSBSB, u32::MAX, false);
        assert!(drawn_shows(&mut r, &dcb, a, WINDOW), "the window shows A");
        assert!(!drawn_shows(&mut r, &dcb, b, WINDOW), "B is its back buffer");
        assert!(drawn_shows(&mut r, &dcb, both, WINDOW));
        // A box one pixel into the desktop is still only B there.
        assert!(!drawn_shows(&mut r, &dcb, b, (99, 100, 300, 300)));
        let (mut r, dcb) = screen(5, 0x15, 1 << 5);
        r.write(reg::COLORMASKLSBSA, u32::MAX, false);
        r.write(reg::COLORMASKLSBSB, u32::MAX, false);
        assert!(drawn_shows(&mut r, &dcb, b, WINDOW), "after the flip the window shows B");
        assert!(!drawn_shows(&mut r, &dcb, a, WINDOW));
        assert!(drawn_shows(&mut r, &dcb, a, (99, 100, 299, 299)), "the desktop beside it shows A");
        assert!(!drawn_shows(&mut r, &dcb, b, (400, 400, 500, 500)), "the desktop never shows B");
    }

    #[test]
    fn a_pair_window_s_other_half_does_not_show() {
        // 5:5:5 pairs on one page: the plane mask picks the half.
        let (a, b) = (0x100 | 1 << 14, 0x100 | 2 << 14);
        for select in [0, 1 << 7] {
            let (mut r, mut dcb) = screen(7, 7, select);
            // One page for both, as a HighImpact at 1280x1024 has it.
            r.write(reg::DRBSIZE, 0x9000_031A, false);
            dcb.xmap.set_scanout(select, 0x240 | 0x240 << 10 | 0x40 << 20, 0);
            r.write(reg::DRBPOINTERS, 0x240 | 0x240 << 10, false);
            r.write(reg::COLORMASKLSBSA, 0x4000_7FFF, false);
            r.write(reg::COLORMASKLSBSB, 0xBFFF_8000, false);
            let shows_b = select != 0;
            assert_eq!(drawn_shows(&mut r, &dcb, a, WINDOW), !shows_b, "select {select:#x}: half A");
            assert_eq!(drawn_shows(&mut r, &dcb, b, WINDOW), shows_b, "select {select:#x}: half B");
            // Alpha alone is in neither half's colour.
            r.write(reg::COLORMASKLSBSA, 0x4000_0000, false);
            assert!(!drawn_shows(&mut r, &dcb, a, WINDOW));
        }
    }

    #[test]
    fn overlay_drawing_and_shared_pages_always_show() {
        let (mut r, dcb) = screen(5, 0x15, 0);
        r.write(reg::COLORMASKLSBSA, u32::MAX, false);
        assert!(drawn_shows(&mut r, &dcb, 0x200 | 0x4F << 14, WINDOW), "the overlay planes");
        // A buffer three pages into A shares its memory.
        r.write(reg::DRBPOINTERS, 0x1C3, false);
        assert!(drawn_shows(&mut r, &dcb, 0x200 | 1 << 14, WINDOW));
        // One clear of both (a depth buffer's pages) does not.
        r.write(reg::DRBPOINTERS, 0x100, false);
        assert!(!drawn_shows(&mut r, &dcb, 0x200 | 1 << 14, WINDOW));
        // Clip-ID drawing is in no frame.
        r.write(reg::DRBPOINTERS, 0x1C0, false);
        assert!(!drawn_shows(&mut r, &dcb, 0x200 | 0x50 << 14, WINDOW));
        // No screen mask: the whole screen, where the desktop shows A.
        r.write(reg::PP1FILLMODE, 0x200 | 1 << 14, false);
        r.write(reg::CLIP_MODE, 0, false);
        let marks = Marks::new();
        r.drawn(|b, planes, rect| marks.mark(store_key(&b), planes, rect));
        let mut taken = Vec::new();
        marks.take(&mut taken);
        assert_eq!(taken[0].rect, Rect::new(0, 0, rss::WIDTH as i32 - 1, rss::HEIGHT as i32 - 1));
        assert!(marks_show(&taken, &r, &dcb));
    }
}
