# Host camera frames are in the format negotiated, not YUYV

**Keywords:** camera, nokhwa, v4l2, mjpeg, nv12, yuyv, format, resolution, issue 70
**Category:** video
**Status:** rule, 2026-11-19. Fix for techomancer/iris#70 (GPD Win Max 2 2023,
Windows). Unit-tested on macOS; the Windows and Linux backends are only
type-checked, not run against a camera.

## Symptom

On a Windows laptop with a very high-resolution webcam, `source = "camera"`
gave a black IndyCam and the log filled with
`camera: short frame (N bytes for W×H), skipping`.

## Cause

`camera_nokhwa.rs` asked nokhwa for `AbsoluteHighestFrameRate` with a
`YuyvFormat` decoder and then read `Buffer::buffer()` as YUYV. The decoder
type only filters which formats `fulfill()` may pick, and its
`decode_image` output is I420, not YUYV; `buffer()` is always the camera's
native bytes. MediaFoundation offers high-resolution cameras mostly as MJPEG
and NV12, so the highest-frame-rate mode was MJPEG (or NV12, 1.5 bytes a
pixel) at the full sensor size, and every frame failed the `w*h*2` check.
The V4L2 backend had the mirror problem: it insisted on YUYV and took the
largest YUYV size, which on most UVC cameras runs at a few frames a second,
and it ignored MJPEG-only cameras entirely.

## How negotiation works now

`src/dev/camera_format.rs` is platform-free and holds the policy:

- Each backend lists every mode the camera offers as a `CaptureMode`
  (format, width, height, fps). nokhwa: `compatible_camera_formats()`. V4L2:
  `enum_formats` × `enum_framesizes` × `enum_frameintervals`; a stepwise size
  range contributes the step nearest above VINO's size and its maximum.
- `choose_mode` ranks them: fps ≥ 24 (or unknown) first, then uncompressed
  before MJPEG, then the smallest size covering 640×486 / 768×576 (else the
  largest below), then YUYV > UYVY > NV12 > RGB > gray, then higher fps.
  MJPEG is used only when no smooth uncompressed mode exists.
- nokhwa opens with any request, then switches with
  `set_camera_requset(Exact(..))`. V4L2 sets the format and keeps what
  `VIDIOC_S_FMT` returned (the driver may change size or format).
- Each frame goes through `to_yuyv(format, w, h, stride, data)` using the
  format and size the frame itself carries (`Buffer::source_frame_format()`
  and `resolution()`; V4L2's returned `Format` and `bytesused`), then the
  usual downscale.

`to_yuyv` checks the length against the declared format and returns
`FrameError::Short` naming the format and both sizes. `FrameReporter` in
`camera.rs` prints the first bad frame and every 300th after, with the
negotiated mode. A frame labelled uncompressed that starts with a JPEG SOI is
decoded as MJPEG (noted once). MJPEG frames are scaled from the JPEG's own
size; a size other than the negotiated one is noted once.

## MJPEG decoding

`zune-jpeg`, not nokhwa's `decoding` feature. mozjpeg reports libjpeg errors
with `resume_unwind`, and `[profile.release]` has `panic = "abort"`, so one
corrupt USB frame would kill the emulator. UVC cameras often leave out the
Huffman tables; zune-jpeg fills in the standard ones only when the frame has
an `AVI1` APP0 segment, so `with_mjpeg_marker` inserts one after SOI when
missing (tables the frame does carry still win). The test fixture
`src/dev/testdata/camera-16x8-422.jpg` (cjpeg, 4:2:2, standard tables) is
decoded both whole and with its DHT segments stripped.

## Gotchas

- macOS AVFoundation reports its NV12 modes (`420v`/`420f`) as YUYV, and
  asking for YUYV makes AVFoundation convert to `yuvs`, so on a Mac the
  chosen mode is always YUYV and the bytes really are YUYV. Do not ask
  nokhwa's AVFoundation backend for NV12: it maps that to a 10-bit format.
- On this MacBook Pro (2026-11-19), nokhwa's AVFoundation
  `compatible_camera_formats()` returned an empty list, so the backend kept
  the mode it opened with (1920×1080 YUYV @30). That path is the old
  behaviour and is fine on macOS for the reason above.
- A process without macOS camera permission opens the camera but never gets
  a frame (ffmpeg hangs the same way), and dropping `CameraSource` then
  joins a worker blocked in `frame()`. The ignored `live_host_camera` test
  leaks the source in that case so it fails instead of hanging.
- nokhwa's MediaFoundation backend reports a rate of 0 for intervals whose
  denominator is not 1; `choose_mode` treats 0 as "unknown, acceptable".
- nokhwa does not report row stride; `infer_stride` accepts rows padded by
  under 256 bytes when the length is a whole number of equal rows.
