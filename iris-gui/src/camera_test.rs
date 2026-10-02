//! "Test Camera" support.
//!
//! Opens the host camera through the same [`iris::dev::camera::CameraSource`] the
//! VINO / IndyCam emulation uses, on a background thread, and parks the latest
//! frame (converted to RGBA) plus a status line for the GUI to display. This
//! gives the user — and an App Review tester — a way to confirm the host-camera
//! capability works (it triggers the macOS camera-permission prompt and shows a
//! live preview) without booting IRIX and configuring the VINO video source.
//!
//! Dropping `CameraTest` stops the worker, which drops the `CameraSource` and
//! releases the camera (indicator light off).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

use parking_lot::Mutex;

use iris::dev::camera::CameraSource;
use iris::video_source::{Field, VideoSource, VideoStandard};

#[derive(Default)]
struct Shared {
    /// One-line capture status (frame count, capture resolution, …).
    status: String,
    /// Set if the camera could not be opened (permission denied / no device).
    error: Option<String>,
    /// Latest preview frame: (width, height, RGBA bytes).
    frame: Option<(u32, u32, Vec<u8>)>,
    /// Bumped on each new frame so the GUI can skip redundant texture uploads.
    seq: u64,
}

pub struct CameraTest {
    shared: Arc<Mutex<Shared>>,
    running: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

impl CameraTest {
    /// Start capturing from host camera `index` using `standard`'s field size.
    pub fn start(standard: VideoStandard, index: u32) -> Self {
        let shared = Arc::new(Mutex::new(Shared {
            status: "opening camera…".into(),
            ..Shared::default()
        }));
        let running = Arc::new(AtomicBool::new(true));
        let s2 = shared.clone();
        let r2 = running.clone();

        let worker = std::thread::Builder::new()
            .name("iris-gui-camtest".into())
            .spawn(move || {
                let cam = match CameraSource::new_with_index(standard, index) {
                    Ok(c) => c,
                    Err(e) => {
                        s2.lock().error = Some(e);
                        return;
                    }
                };
                // next_field() paces itself to the field rate, so this loop
                // runs at ~50–60 Hz without a manual sleep.
                while r2.load(Ordering::Relaxed) {
                    let field = cam.next_field();
                    let rgba = uyvy_field_to_rgba(&field);
                    let status = cam.status();
                    let mut g = s2.lock();
                    g.status = status;
                    g.frame = Some((field.width, field.height, rgba));
                    g.seq = g.seq.wrapping_add(1);
                    g.error = None;
                }
                // `cam` drops here → camera stream closed, device released.
            })
            .expect("spawn camera-test worker");

        Self { shared, running, worker: Some(worker) }
    }

    pub fn status(&self) -> String {
        self.shared.lock().status.clone()
    }

    pub fn error(&self) -> Option<String> {
        self.shared.lock().error.clone()
    }

    /// Return the latest frame if it is newer than `last_seq` (which is then
    /// advanced). `None` when there is nothing new to upload.
    pub fn take_new_frame(&self, last_seq: &mut u64) -> Option<(u32, u32, Vec<u8>)> {
        let g = self.shared.lock();
        if g.seq != *last_seq {
            *last_seq = g.seq;
            g.frame.clone()
        } else {
            None
        }
    }
}

impl Drop for CameraTest {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

/// Convert one packed UYVY 4:2:2 field to RGBA8 (BT.601 limited range).
/// Each 4-byte group `U Y0 V Y1` yields two pixels sharing the U/V chroma.
fn uyvy_field_to_rgba(field: &Field) -> Vec<u8> {
    let w = field.width as usize;
    let h = field.height as usize;
    let src = &field.pixels;
    let mut out = vec![0u8; w * h * 4];

    for y in 0..h {
        let row = y * w * 2;
        for pair in 0..(w / 2) {
            let i = row + pair * 4;
            if i + 3 >= src.len() {
                break;
            }
            let u = src[i] as i32;
            let y0 = src[i + 1] as i32;
            let v = src[i + 2] as i32;
            let y1 = src[i + 3] as i32;

            let o = (y * w + pair * 2) * 4;
            yuv_to_rgba(y0, u, v, &mut out[o..o + 4]);
            yuv_to_rgba(y1, u, v, &mut out[o + 4..o + 8]);
        }
    }
    out
}

#[inline]
fn yuv_to_rgba(y: i32, u: i32, v: i32, out: &mut [u8]) {
    let c = y - 16;
    let d = u - 128;
    let e = v - 128;
    let r = (298 * c + 409 * e + 128) >> 8;
    let g = (298 * c - 100 * d - 208 * e + 128) >> 8;
    let b = (298 * c + 516 * d + 128) >> 8;
    out[0] = r.clamp(0, 255) as u8;
    out[1] = g.clamp(0, 255) as u8;
    out[2] = b.clamp(0, 255) as u8;
    out[3] = 255;
}
