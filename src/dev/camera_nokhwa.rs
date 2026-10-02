//! nokhwa-based capture loop (macOS AVFoundation, Windows MediaFoundation).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use parking_lot::Mutex;

use super::{Shared, downscale_yuyv_to_uyvy, split_fields};

pub(super) fn capture_loop(shared: Arc<Mutex<Shared>>, frame_w: u32, frame_h: u32,
                            camera_index: u32, running: Arc<AtomicBool>) {
    use nokhwa::pixel_format::YuyvFormat;
    use nokhwa::utils::{
        CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType, Resolution,
    };
    use nokhwa::Camera;

    // Most cameras don't advertise 640×486 (NTSC native isn't a standard
    // webcam mode), and many only expose NV12 / MJPEG at their native
    // resolutions.  Ask for the highest-framerate format the camera offers
    // and let YuyvFormat handle the decode to YUV422 inside nokhwa.  We
    // downscale to (frame_w × frame_h) ourselves regardless.
    let _ = CameraFormat::new(
        Resolution::new(frame_w, frame_h),
        FrameFormat::YUYV,
        30,
    ); // (kept for future fine-grained requests)

    let req = RequestedFormat::new::<YuyvFormat>(
        RequestedFormatType::AbsoluteHighestFrameRate
    );

    let mut cam = match Camera::new(CameraIndex::Index(camera_index), req) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("camera: open failed: {} — falling back to black source", e);
            return;
        }
    };
    if let Err(e) = cam.open_stream() {
        eprintln!("camera: open_stream failed: {} — falling back to black source", e);
        return;
    }

    let res = cam.resolution();
    let sw  = res.width_x;
    let sh  = res.height_y;
    eprintln!("camera: streaming at {}×{} → downscale to {}×{}", sw, sh, frame_w, frame_h);
    shared.lock().capture_res = Some((sw, sh));

    while running.load(Ordering::Relaxed) {
        let buf = match cam.frame() {
            Ok(b) => b,
            Err(e) => {
                eprintln!("camera: frame error: {} — retrying", e);
                thread::sleep(Duration::from_millis(33));
                continue;
            }
        };

        let yuyv = buf.buffer();
        let expected = (sw * sh * 2) as usize;
        if yuyv.len() < expected {
            eprintln!("camera: short frame ({} bytes for {}×{}), skipping",
                yuyv.len(), sw, sh);
            continue;
        }

        let frame = downscale_yuyv_to_uyvy(yuyv, sw, sh, frame_w, frame_h);
        let (even, odd) = split_fields(&frame, frame_w, frame_h);

        let mut s = shared.lock();
        s.even = Some(Arc::from(even));
        s.odd  = Some(Arc::from(odd));
        s.frame_count += 1;
    }
}
