//! Live host camera capture as a VINO video source.
//!
//! Platform backends:
//!   Linux   — `camera_v4l`   (V4L2 via the `v4l` crate; requires libv4l-dev)
//!   macOS   — `camera_nokhwa` (AVFoundation via nokhwa)
//!   Windows — `camera_nokhwa` (MediaFoundation via nokhwa)
//!
//! A worker thread opens the camera, and on every frame: area-average
//! downscales to the standard's frame size (640×486 NTSC, 768×576 PAL),
//! converts YUYV→UYVY in the same pass, splits the resulting frame into its
//! even/odd fields, and parks them in a shared slot.
//!
//! `next_field()` paces itself to the field rate and returns the latest field
//! of the matching parity.  If the camera hasn't produced a frame yet it
//! returns a solid black field so VINO DMA still gets coherent bytes.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Instant;

use parking_lot::Mutex;

use crate::video_source::{Field, FieldParity, VideoSource, VideoStandard};

#[cfg(target_os = "linux")]
#[path = "camera_v4l.rs"]
mod backend;
#[cfg(not(target_os = "linux"))]
#[path = "camera_nokhwa.rs"]
mod backend;

// ─── Shared frame slot ───────────────────────────────────────────────────────

pub(crate) struct Shared {
    pub even:        Option<Arc<[u8]>>,
    pub odd:         Option<Arc<[u8]>>,
    pub next_due:    Instant,
    pub next_parity: FieldParity,
    /// Frames received from the capture backend (incremented by backend).
    pub frame_count: u64,
    /// Capture resolution reported by the backend once streaming starts.
    pub capture_res: Option<(u32, u32)>,
}

// ─── CameraSource ────────────────────────────────────────────────────────────

pub struct CameraSource {
    standard: VideoStandard,
    shared:   Arc<Mutex<Shared>>,
    /// Cleared on drop to stop the capture loop so the host camera is released
    /// (its indicator light turns off) rather than the worker running until the
    /// process exits. The backend polls this between frames.
    running:  Arc<AtomicBool>,
    worker:   Option<thread::JoinHandle<()>>,
}

impl CameraSource {
    pub fn new(standard: VideoStandard) -> Result<Self, String> {
        Self::new_with_index(standard, 0)
    }

    pub fn new_with_index(standard: VideoStandard, camera_index: u32) -> Result<Self, String> {
        let (field_w, field_h) = standard.field_size();
        let frame_w = field_w;
        let frame_h = field_h * 2;

        let shared = Arc::new(Mutex::new(Shared {
            even:        None,
            odd:         None,
            next_due:    Instant::now(),
            next_parity: FieldParity::Even,
            frame_count: 0,
            capture_res: None,
        }));
        let s2 = shared.clone();
        let running = Arc::new(AtomicBool::new(true));
        let r2 = running.clone();

        let worker = thread::Builder::new()
            .name("iris-camera".into())
            .spawn(move || backend::capture_loop(s2, frame_w, frame_h, camera_index, r2))
            .map_err(|e| format!("camera worker spawn failed: {}", e))?;

        Ok(Self { standard, shared, running, worker: Some(worker) })
    }
}

impl Drop for CameraSource {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(w) = self.worker.take() {
            // The backend exits within one frame interval of seeing `running`
            // clear, then closes the camera stream. Join so the device is fully
            // released before we return (e.g. before a re-open).
            let _ = w.join();
        }
    }
}

impl VideoSource for CameraSource {
    fn standard(&self) -> VideoStandard { self.standard }

    fn status(&self) -> String {
        let s = self.shared.lock();
        let res = match s.capture_res {
            Some((w, h)) => format!("{}×{}", w, h),
            None         => "pending".to_string(),
        };
        let (fw, fh) = self.standard.field_size();
        format!("frames={} capture={} output={}×{} standard={:?}",
            s.frame_count, res, fw, fh * 2, self.standard)
    }

    fn next_field(&self) -> Field {
        let period     = self.standard.field_period();
        let (w, h)     = self.standard.field_size();

        let (parity, pixels, due) = {
            let mut s = self.shared.lock();
            let due  = s.next_due;
            s.next_due = due + period;
            let parity = s.next_parity;
            s.next_parity = match parity {
                FieldParity::Even => FieldParity::Odd,
                FieldParity::Odd  => FieldParity::Even,
            };
            let pix = match parity {
                FieldParity::Even => s.even.clone(),
                FieldParity::Odd  => s.odd.clone(),
            };
            (parity, pix, due)
        };

        let now = Instant::now();
        if now < due {
            std::thread::sleep(due - now);
        } else if now > due + period * 4 {
            self.shared.lock().next_due = now + period;
        }

        let pixels = pixels.unwrap_or_else(|| black_field(w, h));
        Field { parity, width: w, height: h, pixels }
    }
}

// ─── Shared helpers (used by both backends) ──────────────────────────────────

pub(crate) fn black_field(w: u32, h: u32) -> Arc<[u8]> {
    let mut buf = vec![0u8; (w * h * 2) as usize];
    for c in buf.chunks_mut(4) {
        c[0] = 128; c[1] = 16; c[2] = 128; c[3] = 16;
    }
    Arc::from(buf)
}

/// One-pass area-averaging downscale + YUYV→UYVY reorder.
///
/// Input is standard YUYV (Y0 Cb Y1 Cr); output is canonical UYVY (U Y0 V Y1).
/// See the long comment in the original camera.rs for why U/V are read straight
/// without swapping.
pub(crate) fn downscale_yuyv_to_uyvy(src: &[u8], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<u8> {
    let mut dst = vec![0u8; (dw * dh * 2) as usize];
    let dw_pairs = dw / 2;

    for dy in 0..dh {
        let sy0 = (dy * sh) / dh;
        let sy1 = (((dy + 1) * sh) / dh).max(sy0 + 1).min(sh);

        for dx_pair in 0..dw_pairs {
            let dx0 = dx_pair * 2;
            let sx0 = ((dx0 * sw) / dw) & !1;        // align to YUYV pair
            let sx2 = ((((dx0 + 2) * sw) / dw).max(sx0 + 2)).min(sw) & !1;

            let (mut y0_s, mut u_s, mut y1_s, mut v_s, mut n) = (0u32, 0u32, 0u32, 0u32, 0u32);

            for sy in sy0..sy1 {
                let row = (sy * sw * 2) as usize;
                let mut sx = sx0;
                while sx < sx2 {
                    let i = row + (sx as usize) * 2;
                    if i + 3 >= src.len() { break; }
                    y0_s += src[i    ] as u32; // Y0
                    u_s  += src[i + 1] as u32; // Cb (U) — standard YUYV
                    y1_s += src[i + 2] as u32; // Y1
                    v_s  += src[i + 3] as u32; // Cr (V) — standard YUYV
                    n += 1;
                    sx += 2;
                }
            }
            if n == 0 { n = 1; }

            let di = ((dy * dw + dx0) * 2) as usize;
            dst[di    ] = (u_s  / n) as u8;
            dst[di + 1] = (y0_s / n) as u8;
            dst[di + 2] = (v_s  / n) as u8;
            dst[di + 3] = (y1_s / n) as u8;
        }
    }
    dst
}

/// Split an interlaced frame into even (rows 0,2,…) and odd (rows 1,3,…) fields.
pub(crate) fn split_fields(frame: &[u8], w: u32, h: u32) -> (Vec<u8>, Vec<u8>) {
    let row_bytes = (w * 2) as usize;
    let field_h   = (h / 2) as usize;
    let mut even  = Vec::with_capacity(row_bytes * field_h);
    let mut odd   = Vec::with_capacity(row_bytes * field_h);
    for y in 0..h as usize {
        let row = &frame[y * row_bytes .. (y + 1) * row_bytes];
        if y & 1 == 0 { even.extend_from_slice(row); }
        else          { odd .extend_from_slice(row); }
    }
    (even, odd)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn black_field_is_neutral_uyvy() {
        let b = black_field(2, 1);
        assert_eq!(&b[..], &[128, 16, 128, 16]);
    }

    #[test]
    fn split_fields_partitions_rows_by_parity() {
        // Two-row frame, each row is 4 bytes (2 pixels UYVY).
        let frame = vec![
            0x01, 0x02, 0x03, 0x04,  // row 0 → even
            0x11, 0x12, 0x13, 0x14,  // row 1 → odd
        ];
        let (e, o) = split_fields(&frame, 2, 2);
        assert_eq!(e, vec![0x01, 0x02, 0x03, 0x04]);
        assert_eq!(o, vec![0x11, 0x12, 0x13, 0x14]);
    }

    #[test]
    fn downscale_byte_swap_is_correct_for_1to1_2x1() {
        // nokhwa delivers standard YUYV: Y0 Cb Y1 Cr (Cb/U at byte 1, Cr/V at
        // byte 3). The downscale function re-orders to canonical UYVY (U Y0 V Y1).
        // Input bytes (standard YUYV): Y0=0xA0 Cb=0x40 Y1=0xA1 Cr=0x80
        let src = vec![0xA0, 0x40, 0xA1, 0x80];
        let dst = downscale_yuyv_to_uyvy(&src, 2, 1, 2, 1);
        // Output UYVY: U(=Cb) Y0 V(=Cr) Y1
        assert_eq!(dst, vec![0x40, 0xA0, 0x80, 0xA1]);
    }

    #[test]
    fn downscale_averages_4x1_to_2x1() {
        // Two standard-YUYV pairs at 4×1 → one canonical-UYVY pair at 2×1,
        // averaged. Input byte order per pair: Y0 Cb Y1 Cr (Cb at 1, Cr at 3).
        let src = vec![
            0x10, 0x40, 0x20, 0x80,   // pair 0: Y0=0x10 Cb=0x40 Y1=0x20 Cr=0x80
            0x30, 0x60, 0x40, 0xA0,   // pair 1: Y0=0x30 Cb=0x60 Y1=0x40 Cr=0xA0
        ];
        let dst = downscale_yuyv_to_uyvy(&src, 4, 1, 2, 1);
        // Expected output pair has averaged channels (canonical UYVY):
        //   U  = (0x40 + 0x60) / 2 = 0x50
        //   Y0 = (0x10 + 0x30) / 2 = 0x20
        //   V  = (0x80 + 0xA0) / 2 = 0x90
        //   Y1 = (0x20 + 0x40) / 2 = 0x30
        assert_eq!(dst, vec![0x50, 0x20, 0x90, 0x30]);
    }
}
