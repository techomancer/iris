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
//! to skip metadata/overview nodes, open the first real capture node, choose
//! a mode with `format::choose_mode` from every format, size and rate it
//! offers (YUYV, UYVY, NV12, RGB, gray or MJPEG), start an mmap stream, and
//! convert each frame from the format the driver actually set.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use parking_lot::Mutex;
use v4l::buffer::Type;
use v4l::frameinterval::FrameIntervalEnum;
use v4l::framesize::FrameSizeEnum;
use v4l::io::traits::CaptureStream;
use v4l::prelude::MmapStream;
use v4l::video::Capture;
use v4l::{Device, FourCC};

use super::format::{self, CaptureMode, PixelFormat};
use super::{FrameReporter, Shared, publish_frame};

// Substrings in a V4L2 card name that identify non-capture auxiliary nodes.
const NAME_DENYLIST: &[&str] = &["overview", "metadata", "statistics"];

fn pixel_format(fourcc: FourCC) -> Option<PixelFormat> {
    Some(match &fourcc.repr {
        b"YUYV" => PixelFormat::Yuyv,
        b"UYVY" => PixelFormat::Uyvy,
        b"NV12" => PixelFormat::Nv12,
        b"RGB3" => PixelFormat::Rgb24,
        b"BGR3" => PixelFormat::Bgr24,
        b"GREY" => PixelFormat::Gray,
        b"MJPG" | b"JPEG" => PixelFormat::Mjpeg,
        _ => return None,
    })
}

/// Formats of this node that we can convert, with their FourCCs.
fn usable_formats(dev: &Device) -> Vec<(FourCC, PixelFormat)> {
    dev.enum_formats()
        .map(|list| list.iter().filter_map(|d| Some((d.fourcc, pixel_format(d.fourcc)?))).collect())
        .unwrap_or_default()
}

/// Resolve a logical camera index (0 = first usable device) to the `/dev/videoN`
/// index, skipping denylist names and nodes with no capture format we can use.
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

    let mut usable = 0u32;
    for (idx, card) in &candidates {
        let card_lc = card.to_lowercase();
        if NAME_DENYLIST.iter().any(|d| card_lc.contains(d)) {
            continue;
        }
        // A real capture node lists capture formats; metadata nodes do not.
        let Ok(dev) = Device::new(*idx) else { continue };
        if usable_formats(&dev).is_empty() {
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

/// Highest frame rate the device offers for one format and size (0 = unknown).
fn best_fps(dev: &Device, fourcc: FourCC, w: u32, h: u32) -> u32 {
    dev.enum_frameintervals(fourcc, w, h).ok().into_iter().flatten()
        .map(|i| match i.interval {
            FrameIntervalEnum::Discrete(f) => format::fps_from_interval(f.numerator, f.denominator),
            FrameIntervalEnum::Stepwise(s) => format::fps_from_interval(s.min.numerator, s.min.denominator),
        })
        .max()
        .unwrap_or(0)
}

/// Every mode the device offers in a format we can convert, with its FourCC.
fn list_modes(dev: &Device, frame_w: u32, frame_h: u32) -> Vec<(FourCC, CaptureMode)> {
    let mut modes = Vec::new();
    for (fourcc, pf) in usable_formats(dev) {
        let Ok(sizes) = dev.enum_framesizes(fourcc) else { continue };
        for s in sizes {
            let wh = match s.size {
                FrameSizeEnum::Discrete(d) => vec![(d.width, d.height)],
                FrameSizeEnum::Stepwise(s) => format::stepwise_sizes(
                    (s.min_width, s.min_height), (s.max_width, s.max_height),
                    (s.step_width, s.step_height), frame_w, frame_h),
            };
            for (w, h) in wh {
                modes.push((fourcc, CaptureMode::new(pf, w, h, best_fps(dev, fourcc, w, h))));
            }
        }
    }
    modes
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

        let modes = list_modes(&dev, frame_w, frame_h);
        let plain: Vec<CaptureMode> = modes.iter().map(|(_, m)| *m).collect();
        let Some(want) = format::choose_mode(&plain, frame_w, frame_h) else {
            eprintln!("camera: no usable mode on /dev/video{} ({} listed) — retrying",
                      cur_idx, modes.len());
            continue 'outer;
        };
        let fourcc = modes.iter().find(|(_, m)| *m == want).unwrap().0;

        let mut fmt = dev.format().unwrap_or_else(|_| v4l::Format::new(want.width, want.height, fourcc));
        fmt.width  = want.width;
        fmt.height = want.height;
        fmt.fourcc = fourcc;
        // The driver may adjust any of these; what it returns is what frames
        // will have.
        let got = match dev.set_format(&fmt) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("camera: set_format {} failed: {}", want, e);
                continue 'outer;
            }
        };
        let Some(got_pf) = pixel_format(got.fourcc) else {
            eprintln!("camera: asked for {}, driver set unusable format {}", want, got.fourcc);
            continue 'outer;
        };
        let negotiated = CaptureMode::new(got_pf, got.width, got.height, want.fps);
        if negotiated != want {
            eprintln!("camera: asked for {}, driver set {}", want, negotiated);
        }
        let stride = (got.stride > 0).then_some(got.stride as usize);

        eprintln!("camera: streaming {} → downscale to {}×{}", negotiated, frame_w, frame_h);
        if open_attempts == 1 {
            eprintln!("camera: OK — run 'vino status' in the monitor to check DMA state");
        }
        shared.lock().capture_res = Some((negotiated.width, negotiated.height));
        let mut report = FrameReporter::new(negotiated);

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
        let (data, meta) = match stream.next() {
            Ok(f) => f,
            Err(e) => {
                eprintln!("camera: stream lost: {} — reopening", e);
                continue 'outer;
            }
        };
        let used = meta.bytesused as usize;
        let data = if used > 0 && used <= data.len() { &data[..used] } else { data };

        match format::to_yuyv(got_pf, got.width, got.height, stride, data) {
            Ok(f) => {
                report.good_frame(&f);
                if publish_frame(&shared, &f, frame_w, frame_h) == 1 {
                    eprintln!("camera: first frame received — capture is live");
                }
            }
            Err(e) => report.bad_frame(&e),
        }
        } // inner frame loop
    } // outer reopen loop
}
