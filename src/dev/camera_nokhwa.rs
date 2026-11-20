//! nokhwa-based capture loop (macOS AVFoundation, Windows MediaFoundation).
//!
//! nokhwa hands frames over in the camera's own format: `Buffer::buffer()` is
//! YUYV only if YUYV was negotiated. Windows MediaFoundation in particular
//! offers high-resolution cameras mostly as MJPEG and NV12, so we list the
//! modes, choose one ourselves (`format::choose_mode`), and convert each frame
//! from the format and size the buffer says it has.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use nokhwa::utils::{CameraFormat, FrameFormat};
use parking_lot::Mutex;

use super::format::{self, CaptureMode, PixelFormat};
use super::{FrameReporter, Shared, publish_frame};

/// Every format nokhwa knows; we can convert all of them.
const ALL_FORMATS: &[FrameFormat] = &[
    FrameFormat::YUYV, FrameFormat::NV12, FrameFormat::RAWRGB, FrameFormat::RAWBGR,
    FrameFormat::GRAY, FrameFormat::MJPEG,
];

fn pixel_format(f: FrameFormat) -> PixelFormat {
    match f {
        FrameFormat::YUYV   => PixelFormat::Yuyv,
        FrameFormat::NV12   => PixelFormat::Nv12,
        FrameFormat::RAWRGB => PixelFormat::Rgb24,
        FrameFormat::RAWBGR => PixelFormat::Bgr24,
        FrameFormat::GRAY   => PixelFormat::Gray,
        FrameFormat::MJPEG  => PixelFormat::Mjpeg,
    }
}

fn capture_mode(f: &CameraFormat) -> CaptureMode {
    CaptureMode::new(pixel_format(f.format()), f.width(), f.height(), f.frame_rate())
}

pub(super) fn capture_loop(shared: Arc<Mutex<Shared>>, frame_w: u32, frame_h: u32,
                            camera_index: u32, running: Arc<AtomicBool>) {
    use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
    use nokhwa::Camera;

    // Opening needs some request nokhwa can fulfil before we can list the
    // modes; any format will do, we replace it below.
    let req = RequestedFormat::with_formats(RequestedFormatType::AbsoluteHighestFrameRate,
                                            ALL_FORMATS);
    let mut cam = match Camera::new(CameraIndex::Index(camera_index), req) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("camera: open failed: {} — falling back to black source", e);
            return;
        }
    };

    match cam.compatible_camera_formats() {
        Ok(list) => {
            let modes: Vec<CaptureMode> = list.iter().map(capture_mode).collect();
            match format::choose_mode(&modes, frame_w, frame_h) {
                Some(want) => {
                    let pick = list.iter().copied().find(|f| capture_mode(f) == want).unwrap();
                    if pick != cam.camera_format() {
                        let exact = RequestedFormat::with_formats(
                            RequestedFormatType::Exact(pick), ALL_FORMATS);
                        if let Err(e) = cam.set_camera_requset(exact) {
                            eprintln!("camera: could not switch to {}: {}; staying at {}",
                                want, e, capture_mode(&cam.camera_format()));
                        }
                    }
                }
                None => eprintln!("camera: none of the camera's {} modes is usable; \
                                   staying at {}", modes.len(), capture_mode(&cam.camera_format())),
            }
        }
        Err(e) => eprintln!("camera: could not list modes ({}); staying at {}",
                            e, capture_mode(&cam.camera_format())),
    }

    if let Err(e) = cam.open_stream() {
        eprintln!("camera: open_stream failed: {} — falling back to black source", e);
        return;
    }

    let negotiated = capture_mode(&cam.camera_format());
    eprintln!("camera: streaming {} → downscale to {}×{}", negotiated, frame_w, frame_h);
    shared.lock().capture_res = Some((negotiated.width, negotiated.height));
    let mut report = FrameReporter::new(negotiated);

    while running.load(Ordering::Relaxed) {
        let buf = match cam.frame() {
            Ok(b) => b,
            Err(e) => {
                eprintln!("camera: frame error: {} — retrying", e);
                thread::sleep(Duration::from_millis(33));
                continue;
            }
        };

        // The buffer carries the format and size the backend negotiated for
        // this frame; trust those, not what we asked for.
        let fmt = pixel_format(buf.source_frame_format());
        let res = buf.resolution();
        let data = buf.buffer();
        let stride = format::infer_stride(fmt, res.width(), res.height(), data.len());
        match format::to_yuyv(fmt, res.width(), res.height(), stride, data) {
            Ok(f) => {
                report.good_frame(&f);
                if publish_frame(&shared, &f, frame_w, frame_h) == 1 {
                    eprintln!("camera: first frame received — capture is live");
                }
            }
            Err(e) => report.bad_frame(&e),
        }
    }
}
