//! Linux V4L2 capture loop via the `v4l` crate.
//!
//! nokhwa's V4L2 backend has two bugs that prevent it from working on cameras
//! that report frame intervals with numerator != 1 (e.g. 1001/30000 for 29.97
//! fps): the format-enumeration loop silently drops those entries, so
//! `fulfill()` sees an empty list and fails.  Even when the format is forced
//! via `RequestedFormatType::Exact`, `open_stream()` never calls
//! VIDIOC_STREAMON unless the `no-arena-buffer` feature is enabled, so every
//! subsequent `frame()` returns ENODEV.
//!
//! We bypass nokhwa for capture entirely on Linux: enumerate devices ourselves
//! to skip metadata/overview nodes, open the first real YUYV capture node,
//! start an mmap stream, and pull frames directly.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use parking_lot::Mutex;
use v4l::buffer::Type;
use v4l::framesize::FrameSizeEnum;
use v4l::io::traits::CaptureStream;
use v4l::prelude::MmapStream;
use v4l::video::Capture;
use v4l::{Device, FourCC};

use super::{Shared, downscale_yuyv_to_uyvy, split_fields};

// Substrings in a V4L2 card name that identify non-capture auxiliary nodes.
const NAME_DENYLIST: &[&str] = &["overview", "metadata", "statistics"];

/// Resolve a logical camera index (0 = first usable device) to the `/dev/videoN`
/// index, skipping denylist names and nodes that don't support VIDIOC_ENUM_FRAMESIZES.
fn resolve_device_index(logical: u32) -> Option<usize> {
    // Walk /sys/class/video4linux in index order.
    let mut candidates: Vec<(usize, String)> = std::fs::read_dir("/sys/class/video4linux")
        .ok()?
        .filter_map(|e| {
            let e = e.ok()?;
            let name = e.file_name().into_string().ok()?; // "videoN"
            let idx: usize = name.strip_prefix("video")?.parse().ok()?;
            let card = std::fs::read_to_string(e.path().join("name")).ok()?;
            let card = card.trim().to_string();
            Some((idx, card))
        })
        .collect();
    candidates.sort_by_key(|(idx, _)| *idx);

    let yuyv = FourCC::new(b"YUYV");
    let mut usable = 0u32;
    for (idx, card) in &candidates {
        let card_lc = card.to_lowercase();
        if NAME_DENYLIST.iter().any(|d| card_lc.contains(d)) {
            continue;
        }
        // Check it's a real capture node by trying VIDIOC_ENUM_FRAMESIZES.
        let Ok(dev) = Device::new(*idx) else { continue };
        if dev.enum_framesizes(yuyv).is_err() {
            continue;
        }
        if usable == logical {
            eprintln!("camera: logical {} → /dev/video{} (\"{}\")", logical, idx, card);
            return Some(*idx);
        }
        usable += 1;
    }
    eprintln!("camera: logical index {} out of range ({} usable devices)", logical, usable);
    None
}

/// Pick the largest YUYV resolution the device offers.
fn best_yuyv_resolution(dev: &Device) -> Option<(u32, u32)> {
    let yuyv = FourCC::new(b"YUYV");
    dev.enum_framesizes(yuyv).ok()?
        .iter()
        .filter_map(|s| match &s.size {
            FrameSizeEnum::Discrete(d)   => Some((d.width, d.height)),
            FrameSizeEnum::Stepwise(s)   => Some((s.max_width, s.max_height)),
        })
        .max_by_key(|&(w, h)| w * h)
}

pub(super) fn capture_loop(shared: Arc<Mutex<Shared>>, frame_w: u32, frame_h: u32,
                            camera_index: u32, running: Arc<AtomicBool>) {
    // Outer retry loop: reopens the device from scratch after stream loss or
    // hot-plug. Re-resolves the device index each time so a swap (e.g. Logitech
    // → Huddly GO) picks up the new camera's node without restarting iris.
    let mut open_attempts = 0u32;
    'outer: loop {
        if !running.load(Ordering::Relaxed) { return; }
        let cur_idx = match resolve_device_index(camera_index) {
            Some(i) => i,
            None => {
                let delay = (open_attempts * 2).min(30);
                if open_attempts == 0 {
                    eprintln!("camera: no usable V4L2 device — waiting for hot-plug");
                }
                thread::sleep(Duration::from_secs(delay.max(2) as u64));
                open_attempts += 1;
                continue 'outer;
            }
        };

        if open_attempts > 0 {
            let delay = (open_attempts * 2).min(30);
            eprintln!("camera: reopening /dev/video{} in {}s (attempt {})", cur_idx, delay, open_attempts + 1);
            thread::sleep(Duration::from_secs(delay as u64));
        }
        open_attempts += 1;

        let dev = match Device::new(cur_idx) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("camera: open /dev/video{} failed: {}", cur_idx, e);
                continue 'outer;
            }
        };

        let (sw, sh) = match best_yuyv_resolution(&dev) {
            Some(r) => r,
            None => {
                eprintln!("camera: no YUYV format on /dev/video{} — retrying", cur_idx);
                continue 'outer;
            }
        };

        let mut fmt = dev.format().unwrap_or_else(|_| v4l::Format::new(sw, sh, FourCC::new(b"YUYV")));
        fmt.width  = sw;
        fmt.height = sh;
        fmt.fourcc = FourCC::new(b"YUYV");
        if let Err(e) = dev.set_format(&fmt) {
            eprintln!("camera: set_format {}×{} YUYV failed: {}", sw, sh, e);
            continue 'outer;
        }

        eprintln!("camera: streaming at {}×{} → downscale to {}×{}", sw, sh, frame_w, frame_h);
        if open_attempts == 1 {
            eprintln!("camera: OK — run 'vino status' in the monitor to check DMA state");
        }
        shared.lock().capture_res = Some((sw, sh));

        let mut stream = match MmapStream::with_buffers(&dev, Type::VideoCapture, 4) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("camera: mmap stream failed: {}", e);
                continue 'outer;
            }
        };

        // Reset open_attempts on a successful stream start.
        open_attempts = 0;

        loop {
        if !running.load(Ordering::Relaxed) { return; }
        let (data, _meta) = match stream.next() {
            Ok(f) => f,
            Err(e) => {
                eprintln!("camera: stream lost: {} — reopening", e);
                continue 'outer;
            }
        };

        let expected = (sw * sh * 2) as usize;
        if data.len() < expected {
            eprintln!("camera: short frame ({} bytes for {}×{}), skipping", data.len(), sw, sh);
            continue;
        }

        let frame = downscale_yuyv_to_uyvy(data, sw, sh, frame_w, frame_h);
        let (even, odd) = split_fields(&frame, frame_w, frame_h);

        let mut s = shared.lock();
        s.even = Some(Arc::from(even));
        s.odd  = Some(Arc::from(odd));
        s.frame_count += 1;
        if s.frame_count == 1 {
            eprintln!("camera: first frame received — capture is live");
        }
        } // inner frame loop
    } // outer reopen loop
}
