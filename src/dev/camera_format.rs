//! Host camera formats: choosing a capture mode, and turning the frames the
//! host delivers into packed YUYV for the VINO downscaler.
//!
//! Both camera backends (`camera_nokhwa`, `camera_v4l`) list the modes the
//! camera offers as [`CaptureMode`]s, pick one with [`choose_mode`], and then
//! pass every frame through [`to_yuyv`] together with the format and size the
//! host says that frame has. Nothing here assumes YUYV: a camera that only
//! offers NV12, RGB or MJPEG at the size we want works too, and a frame that
//! does not match its declared format is reported instead of being read as
//! garbage.
//!
//! MJPEG is decoded with `zune-jpeg`, which reports bad data as an error.
//! nokhwa's own decoder (mozjpeg) reports libjpeg errors by unwinding, and
//! release builds abort on panic, so one corrupt USB frame would end the
//! emulator.

use std::borrow::Cow;
use std::fmt;

/// Pixel formats a host camera may deliver.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum PixelFormat {
    /// Packed 4:2:2, Y0 Cb Y1 Cr (YUY2).
    Yuyv,
    /// Packed 4:2:2, Cb Y0 Cr Y1.
    Uyvy,
    /// Planar Y, then interleaved Cb/Cr at half resolution in both directions.
    Nv12,
    /// Packed R G B.
    Rgb24,
    /// Packed B G R.
    Bgr24,
    /// 8-bit luma only.
    Gray,
    /// One JPEG image per frame; the length varies.
    Mjpeg,
}

impl PixelFormat {
    pub(crate) fn is_compressed(self) -> bool {
        matches!(self, PixelFormat::Mjpeg)
    }

    /// Lower is better among modes that are otherwise equal: YUYV needs no
    /// conversion, NV12 and RGB a cheap one, MJPEG a decode.
    fn preference(self) -> u8 {
        match self {
            PixelFormat::Yuyv  => 0,
            PixelFormat::Uyvy  => 1,
            PixelFormat::Nv12  => 2,
            PixelFormat::Rgb24 | PixelFormat::Bgr24 => 3,
            PixelFormat::Gray  => 4,
            PixelFormat::Mjpeg => 5,
        }
    }

    /// Bytes in one tightly packed `w`×`h` frame, or `None` for MJPEG.
    pub(crate) fn frame_len(self, w: u32, h: u32) -> Option<usize> {
        let (w, h) = (w as usize, h as usize);
        match self {
            PixelFormat::Yuyv | PixelFormat::Uyvy => Some(w * h * 2),
            PixelFormat::Nv12 => Some(w * h + w * h.div_ceil(2)),
            PixelFormat::Rgb24 | PixelFormat::Bgr24 => Some(w * h * 3),
            PixelFormat::Gray => Some(w * h),
            PixelFormat::Mjpeg => None,
        }
    }

    /// Bytes per pixel of the packed formats; `None` for NV12 and MJPEG.
    fn packed_bpp(self) -> Option<usize> {
        match self {
            PixelFormat::Yuyv | PixelFormat::Uyvy => Some(2),
            PixelFormat::Rgb24 | PixelFormat::Bgr24 => Some(3),
            PixelFormat::Gray => Some(1),
            PixelFormat::Nv12 | PixelFormat::Mjpeg => None,
        }
    }
}

impl fmt::Display for PixelFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            PixelFormat::Yuyv  => "YUYV",
            PixelFormat::Uyvy  => "UYVY",
            PixelFormat::Nv12  => "NV12",
            PixelFormat::Rgb24 => "RGB24",
            PixelFormat::Bgr24 => "BGR24",
            PixelFormat::Gray  => "GRAY",
            PixelFormat::Mjpeg => "MJPEG",
        })
    }
}

/// One mode a camera offers: a format, a size and a frame rate (0 = unknown).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CaptureMode {
    pub format: PixelFormat,
    pub width:  u32,
    pub height: u32,
    pub fps:    u32,
}

impl CaptureMode {
    pub(crate) fn new(format: PixelFormat, width: u32, height: u32, fps: u32) -> Self {
        Self { format, width, height, fps }
    }
}

impl fmt::Display for CaptureMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}×{} {}", self.width, self.height, self.format)?;
        if self.fps != 0 { write!(f, " @{} fps", self.fps) } else { Ok(()) }
    }
}

/// Below this rate the captured video visibly stutters; VINO wants 25 or 30
/// frames a second.
pub(crate) const MIN_SMOOTH_FPS: u32 = 24;

/// Pick the mode to capture in for a `target_w`×`target_h` VINO frame.
///
/// In order of importance:
/// 1. a frame rate of at least [`MIN_SMOOTH_FPS`] (or an unknown rate);
/// 2. an uncompressed format over MJPEG;
/// 3. a size that covers the target, the smallest such; failing that, the
///    largest the camera has (we downscale, so more pixels than needed only
///    cost bandwidth, and a high-resolution camera's full size is wasteful);
/// 4. the cheapest format to convert (YUYV, UYVY, NV12, RGB, gray);
/// 5. the higher frame rate.
///
/// Modes with an odd width or a zero size are skipped: 4:2:2 output needs
/// pixel pairs.
pub(crate) fn choose_mode(modes: &[CaptureMode], target_w: u32, target_h: u32)
    -> Option<CaptureMode>
{
    let target_area = target_w as u64 * target_h as u64;
    modes.iter()
        .copied()
        .filter(|m| m.width >= 2 && m.height >= 1 && m.width % 2 == 0)
        .min_by_key(|m| {
            let slow  = m.fps != 0 && m.fps < MIN_SMOOTH_FPS;
            let area  = m.width as u64 * m.height as u64;
            let short = m.width < target_w || m.height < target_h;
            let size_cost = if short { target_area.saturating_sub(area) } else { area - target_area };
            (slow, m.format.is_compressed(), short, size_cost,
             m.format.preference(), std::cmp::Reverse(m.fps))
        })
}

/// Sizes worth trying from a stepwise (continuous) size range: the smallest
/// step at or above the target, clamped to the range, and the largest.
pub(crate) fn stepwise_sizes(min: (u32, u32), max: (u32, u32), step: (u32, u32),
                             target_w: u32, target_h: u32) -> Vec<(u32, u32)> {
    let fit = |t: u32, lo: u32, hi: u32, st: u32| {
        let st = st.max(1);
        let t = t.clamp(lo, hi);
        (lo + (t - lo).div_ceil(st) * st).min(hi)
    };
    let near = (fit(target_w, min.0, max.0, step.0), fit(target_h, min.1, max.1, step.1));
    if near == max { vec![max] } else { vec![near, max] }
}

/// Frames per second for a frame interval of `num`/`den` seconds, rounded;
/// 0 if unknown.
pub(crate) fn fps_from_interval(num: u32, den: u32) -> u32 {
    if num == 0 { 0 } else { (den + num / 2) / num }
}

/// A frame converted to packed YUYV, with the size it really has.
#[derive(Debug)]
pub(crate) struct YuyvFrame<'a> {
    pub data:   Cow<'a, [u8]>,
    pub width:  u32,
    pub height: u32,
    /// The frame was declared uncompressed but held a JPEG, and was decoded
    /// as one.
    pub was_jpeg: bool,
}

/// Why a frame could not be used.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum FrameError {
    /// Fewer bytes than the declared format and size need.
    Short { format: PixelFormat, width: u32, height: u32, got: usize, expected: usize },
    /// The MJPEG data did not decode.
    Mjpeg(String),
    /// A zero size or an odd width for a 4:2:2 or NV12 frame.
    BadSize { format: PixelFormat, width: u32, height: u32 },
}

impl fmt::Display for FrameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FrameError::Short { format, width, height, got, expected } => write!(f,
                "{} bytes for a {}×{} {} frame, which needs {}; the camera is not \
                 delivering the format it negotiated", got, width, height, format, expected),
            FrameError::Mjpeg(e) => write!(f, "MJPEG frame did not decode: {}", e),
            FrameError::BadSize { format, width, height } => write!(f,
                "unusable {}×{} {} frame", width, height, format),
        }
    }
}

/// Convert one host frame of the declared `format` and size to packed YUYV.
///
/// `stride` is the bytes per row of a packed format when the host pads rows
/// (V4L2 `bytesperline`); `None` means tightly packed. Extra bytes after the
/// frame are ignored. MJPEG frames report the size the JPEG itself has, which
/// the caller should compare with the negotiated one.
pub(crate) fn to_yuyv<'a>(format: PixelFormat, width: u32, height: u32,
                          stride: Option<usize>, data: &'a [u8])
    -> Result<YuyvFrame<'a>, FrameError>
{
    if format == PixelFormat::Mjpeg {
        return decode_mjpeg(data).map(|f| YuyvFrame { was_jpeg: false, ..f });
    }
    let needs_pairs = !matches!(format, PixelFormat::Rgb24 | PixelFormat::Bgr24 | PixelFormat::Gray);
    if width == 0 || height == 0 || (needs_pairs && width % 2 != 0) {
        return Err(FrameError::BadSize { format, width, height });
    }

    let (w, h) = (width as usize, height as usize);
    let stride = match (format.packed_bpp(), stride) {
        (Some(bpp), Some(s)) if s > w * bpp => s,
        (Some(bpp), _) => w * bpp,
        (None, _) => w,
    };
    let expected = match format {
        PixelFormat::Nv12 => format.frame_len(width, height).unwrap(),
        _ => stride * (h - 1) + w * format.packed_bpp().unwrap(),
    };
    if data.len() < expected {
        // A camera that claims an uncompressed format but sends JPEG: decode
        // it rather than drop every frame.
        if data.starts_with(&[0xFF, 0xD8]) {
            if let Ok(f) = decode_mjpeg(data) {
                return Ok(YuyvFrame { was_jpeg: true, ..f });
            }
        }
        return Err(FrameError::Short { format, width, height, got: data.len(), expected });
    }

    let out_w = width & !1;
    let data: Cow<'a, [u8]> = match format {
        PixelFormat::Yuyv if stride == w * 2 => Cow::Borrowed(&data[..w * h * 2]),
        PixelFormat::Yuyv => Cow::Owned(compact_rows(data, w * 2, h, stride)),
        PixelFormat::Uyvy => Cow::Owned(uyvy_to_yuyv(data, w, h, stride)),
        PixelFormat::Nv12 => Cow::Owned(nv12_to_yuyv(data, w, h)),
        PixelFormat::Rgb24 => Cow::Owned(rgb_to_yuyv(data, w, h, stride, false)),
        PixelFormat::Bgr24 => Cow::Owned(rgb_to_yuyv(data, w, h, stride, true)),
        PixelFormat::Gray  => Cow::Owned(gray_to_yuyv(data, w, h, stride)),
        PixelFormat::Mjpeg => unreachable!(),
    };
    Ok(YuyvFrame { data, width: out_w, height, was_jpeg: false })
}

/// Guess the row stride of a packed frame from its length, for hosts that
/// pad rows without saying so: a frame of exactly `height` equal rows, each
/// longer than the pixels need, by less than 256 bytes. Anything else is
/// read as tightly packed.
pub(crate) fn infer_stride(format: PixelFormat, width: u32, height: u32, len: usize)
    -> Option<usize>
{
    let bpp = format.packed_bpp()?;
    let (row, h) = (width as usize * bpp, height as usize);
    if h == 0 || len % h != 0 { return None; }
    let stride = len / h;
    (stride > row && stride - row < 256).then_some(stride)
}

// ─── Conversions ─────────────────────────────────────────────────────────────

fn compact_rows(src: &[u8], row: usize, h: usize, stride: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(row * h);
    for y in 0..h {
        out.extend_from_slice(&src[y * stride..y * stride + row]);
    }
    out
}

fn uyvy_to_yuyv(src: &[u8], w: usize, h: usize, stride: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(w * h * 2);
    for y in 0..h {
        for q in src[y * stride..y * stride + w * 2].chunks_exact(4) {
            out.extend_from_slice(&[q[1], q[0], q[3], q[2]]);
        }
    }
    out
}

fn nv12_to_yuyv(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    let (luma, chroma) = src.split_at(w * h);
    let mut out = Vec::with_capacity(w * h * 2);
    for y in 0..h {
        let yrow = &luma[y * w..(y + 1) * w];
        let crow = &chroma[(y / 2) * w..(y / 2 + 1) * w];
        for x in (0..w).step_by(2) {
            out.extend_from_slice(&[yrow[x], crow[x], yrow[x + 1], crow[x + 1]]);
        }
    }
    out
}

/// BT.601 studio-range Y'CbCr from full-range RGB, in 8.8 fixed point.
fn rgb_to_ycbcr(r: i32, g: i32, b: i32) -> (u8, i32, i32) {
    let y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
    let u = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
    let v = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
    (y.clamp(16, 235) as u8, u, v)
}

/// Packed RGB (or BGR) to YUYV. An odd width loses its last column.
fn rgb_to_yuyv(src: &[u8], w: usize, h: usize, stride: usize, bgr: bool) -> Vec<u8> {
    let pairs = w / 2;
    let mut out = Vec::with_capacity(pairs * 4 * h);
    for y in 0..h {
        let row = &src[y * stride..y * stride + w * 3];
        for p in 0..pairs {
            let px = |i: usize| {
                let c = &row[i * 3..i * 3 + 3];
                if bgr { (c[2] as i32, c[1] as i32, c[0] as i32) }
                else   { (c[0] as i32, c[1] as i32, c[2] as i32) }
            };
            let (r0, g0, b0) = px(p * 2);
            let (r1, g1, b1) = px(p * 2 + 1);
            let (y0, u0, v0) = rgb_to_ycbcr(r0, g0, b0);
            let (y1, u1, v1) = rgb_to_ycbcr(r1, g1, b1);
            let u = ((u0 + u1 + 1) / 2).clamp(16, 240) as u8;
            let v = ((v0 + v1 + 1) / 2).clamp(16, 240) as u8;
            out.extend_from_slice(&[y0, u, y1, v]);
        }
    }
    out
}

/// Full-range 8-bit luma to YUYV with neutral chroma. An odd width loses its
/// last column.
fn gray_to_yuyv(src: &[u8], w: usize, h: usize, stride: usize) -> Vec<u8> {
    let pairs = w / 2;
    let luma = |g: u8| (16 + (g as u32 * 219 + 127) / 255) as u8;
    let mut out = Vec::with_capacity(pairs * 4 * h);
    for y in 0..h {
        let row = &src[y * stride..y * stride + w];
        for p in 0..pairs {
            out.extend_from_slice(&[luma(row[p * 2]), 128, luma(row[p * 2 + 1]), 128]);
        }
    }
    out
}

// ─── MJPEG ───────────────────────────────────────────────────────────────────

/// An APP0 segment tagged "AVI1", as Motion-JPEG streams carry it. It tells
/// the decoder to use the standard Huffman tables (ITU T.81 Annex K) for any
/// the frame leaves out, which UVC cameras commonly do.
const AVI1_APP0: [u8; 9] = [0xFF, 0xE0, 0x00, 0x07, b'A', b'V', b'I', b'1', 0x00];

/// Mark a JPEG as Motion-JPEG, unless it is already, so missing Huffman
/// tables fall back to the standard ones. Tables the frame does carry win.
fn with_mjpeg_marker(data: &[u8]) -> Cow<'_, [u8]> {
    if data.len() < 2 || data[..2] != [0xFF, 0xD8] {
        return Cow::Borrowed(data);
    }
    if data[..data.len().min(64)].windows(4).any(|w| w == b"AVI1") {
        return Cow::Borrowed(data);
    }
    let mut out = Vec::with_capacity(data.len() + AVI1_APP0.len());
    out.extend_from_slice(&data[..2]);
    out.extend_from_slice(&AVI1_APP0);
    out.extend_from_slice(&data[2..]);
    Cow::Owned(out)
}

fn decode_mjpeg(data: &[u8]) -> Result<YuyvFrame<'static>, FrameError> {
    use zune_core::bytestream::ZCursor;
    use zune_core::colorspace::ColorSpace;
    use zune_core::options::DecoderOptions;
    use zune_jpeg::JpegDecoder;

    if data.is_empty() {
        return Err(FrameError::Mjpeg("empty frame".into()));
    }
    let jpeg = with_mjpeg_marker(data);
    let opts = DecoderOptions::default()
        .jpeg_set_out_colorspace(ColorSpace::RGB)
        .set_strict_mode(false);
    let mut dec = JpegDecoder::new_with_options(ZCursor::new(&jpeg[..]), opts);
    let pixels = dec.decode().map_err(|e| FrameError::Mjpeg(format!("{e:?}")))?;
    let info = dec.info().ok_or_else(|| FrameError::Mjpeg("no image header".into()))?;
    let (w, h) = (info.width as usize, info.height as usize);
    if w < 2 || h == 0 {
        return Err(FrameError::Mjpeg(format!("{}×{} image", w, h)));
    }
    let channels = match dec.output_colorspace() {
        Some(ColorSpace::Luma) => 1,
        Some(ColorSpace::RGB) | None => 3,
        Some(cs) => return Err(FrameError::Mjpeg(format!("decoded to {cs:?}"))),
    };
    if pixels.len() < w * h * channels {
        return Err(FrameError::Mjpeg(format!(
            "{} bytes decoded for a {}×{} image", pixels.len(), w, h)));
    }
    let yuyv = if channels == 1 { gray_to_yuyv(&pixels, w, h, w) }
               else             { rgb_to_yuyv(&pixels, w, h, w * 3, false) };
    Ok(YuyvFrame { data: Cow::Owned(yuyv), width: (w & !1) as u32, height: h as u32,
                   was_jpeg: false })
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use PixelFormat::*;

    const NTSC: (u32, u32) = (640, 486);
    const PAL:  (u32, u32) = (768, 576);

    fn m(f: PixelFormat, w: u32, h: u32, fps: u32) -> CaptureMode { CaptureMode::new(f, w, h, fps) }

    #[test]
    fn prefers_uncompressed_over_mjpeg_at_the_same_size() {
        let modes = [m(Mjpeg, 1280, 720, 30), m(Yuyv, 1280, 720, 30), m(Nv12, 1280, 720, 30)];
        assert_eq!(choose_mode(&modes, NTSC.0, NTSC.1), Some(m(Yuyv, 1280, 720, 30)));
    }

    #[test]
    fn picks_the_smallest_size_that_covers_the_target() {
        // A very high-resolution camera: its full size is wasted on VINO.
        let modes = [m(Yuyv, 2560, 1920, 30), m(Yuyv, 1280, 720, 30),
                     m(Yuyv, 800, 600, 30), m(Yuyv, 640, 480, 30)];
        assert_eq!(choose_mode(&modes, NTSC.0, NTSC.1), Some(m(Yuyv, 800, 600, 30)));
        assert_eq!(choose_mode(&modes, PAL.0, PAL.1), Some(m(Yuyv, 800, 600, 30)));
    }

    #[test]
    fn falls_back_to_the_largest_size_below_the_target() {
        let modes = [m(Yuyv, 320, 240, 30), m(Yuyv, 640, 480, 30), m(Yuyv, 160, 120, 30)];
        assert_eq!(choose_mode(&modes, NTSC.0, NTSC.1), Some(m(Yuyv, 640, 480, 30)));
    }

    #[test]
    fn a_smooth_rate_beats_size() {
        // Typical UVC camera: YUYV above VGA only at a few frames a second.
        let modes = [m(Yuyv, 1280, 720, 10), m(Yuyv, 640, 480, 30), m(Mjpeg, 1280, 720, 30)];
        assert_eq!(choose_mode(&modes, NTSC.0, NTSC.1), Some(m(Yuyv, 640, 480, 30)));
    }

    #[test]
    fn mjpeg_only_when_nothing_uncompressed_is_smooth() {
        // A camera that offers YUYV only at its slow full size and MJPEG
        // everywhere: MJPEG near the target wins.
        let modes = [m(Yuyv, 2592, 1944, 5), m(Mjpeg, 2592, 1944, 30),
                     m(Mjpeg, 1280, 720, 30), m(Mjpeg, 640, 480, 30)];
        assert_eq!(choose_mode(&modes, NTSC.0, NTSC.1), Some(m(Mjpeg, 1280, 720, 30)));
        // Only MJPEG at all.
        assert_eq!(choose_mode(&[m(Mjpeg, 3840, 2160, 30)], NTSC.0, NTSC.1),
                   Some(m(Mjpeg, 3840, 2160, 30)));
    }

    #[test]
    fn slow_modes_are_still_used_when_nothing_else_exists() {
        let modes = [m(Yuyv, 1920, 1080, 5), m(Mjpeg, 1920, 1080, 15)];
        assert_eq!(choose_mode(&modes, NTSC.0, NTSC.1), Some(m(Yuyv, 1920, 1080, 5)));
    }

    #[test]
    fn unknown_rate_counts_as_smooth_and_higher_rate_breaks_ties() {
        let modes = [m(Yuyv, 640, 480, 15), m(Nv12, 640, 480, 0)];
        assert_eq!(choose_mode(&modes, NTSC.0, NTSC.1), Some(m(Nv12, 640, 480, 0)));
        let modes = [m(Yuyv, 640, 480, 25), m(Yuyv, 640, 480, 30)];
        assert_eq!(choose_mode(&modes, NTSC.0, NTSC.1), Some(m(Yuyv, 640, 480, 30)));
    }

    #[test]
    fn odd_widths_and_empty_lists_are_skipped() {
        assert_eq!(choose_mode(&[], NTSC.0, NTSC.1), None);
        assert_eq!(choose_mode(&[m(Yuyv, 641, 480, 30)], NTSC.0, NTSC.1), None);
    }

    #[test]
    fn stepwise_ranges_offer_the_target_and_the_maximum() {
        assert_eq!(stepwise_sizes((160, 120), (1920, 1080), (16, 8), 640, 486),
                   vec![(640, 488), (1920, 1080)]);
        // Target above the range: only the maximum.
        assert_eq!(stepwise_sizes((160, 120), (320, 240), (2, 2), 640, 486), vec![(320, 240)]);
        // Zero step is treated as 1.
        assert_eq!(stepwise_sizes((2, 2), (4000, 3000), (0, 0), 768, 576),
                   vec![(768, 576), (4000, 3000)]);
    }

    #[test]
    fn frame_intervals_to_rates() {
        assert_eq!(fps_from_interval(1, 30), 30);
        assert_eq!(fps_from_interval(1001, 30000), 30);
        assert_eq!(fps_from_interval(1, 5), 5);
        assert_eq!(fps_from_interval(0, 0), 0);
    }

    #[test]
    fn frame_lengths() {
        assert_eq!(Yuyv.frame_len(640, 480), Some(614_400));
        assert_eq!(Nv12.frame_len(640, 480), Some(460_800));
        assert_eq!(Nv12.frame_len(4, 3), Some(12 + 8));
        assert_eq!(Rgb24.frame_len(2, 2), Some(12));
        assert_eq!(Gray.frame_len(2, 2), Some(4));
        assert_eq!(Mjpeg.frame_len(640, 480), None);
    }

    #[test]
    fn yuyv_passes_through_without_copying() {
        let src = [1, 2, 3, 4, 5, 6, 7, 8, 99];
        let f = to_yuyv(Yuyv, 2, 2, None, &src).unwrap();
        assert!(matches!(f.data, Cow::Borrowed(_)));
        assert_eq!(&f.data[..], &src[..8]);
        assert_eq!((f.width, f.height), (2, 2));
    }

    #[test]
    fn padded_rows_are_compacted() {
        let src = [1, 2, 3, 4, 0xEE, 0xEE, 5, 6, 7, 8];
        let f = to_yuyv(Yuyv, 2, 2, Some(6), &src).unwrap();
        assert_eq!(&f.data[..], &[1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn stride_is_inferred_only_from_padded_rows() {
        // 2 px YUYV rows padded to 8 bytes.
        assert_eq!(infer_stride(Yuyv, 2, 3, 24), Some(8));
        assert_eq!(infer_stride(Yuyv, 2, 3, 12), None);       // tightly packed
        assert_eq!(infer_stride(Yuyv, 2, 3, 13), None);       // not whole rows
        assert_eq!(infer_stride(Yuyv, 2, 1, 4096), None);     // far too long a row
        assert_eq!(infer_stride(Nv12, 2, 2, 12), None);       // planar
        assert_eq!(infer_stride(Mjpeg, 2, 2, 100), None);
    }

    #[test]
    fn short_frames_are_reported_with_the_negotiated_format() {
        // What issue #70 looked like: NV12 (1.5 bytes a pixel) read as YUYV.
        let src = vec![0u8; Nv12.frame_len(640, 480).unwrap()];
        let err = to_yuyv(Yuyv, 640, 480, None, &src).unwrap_err();
        assert_eq!(err, FrameError::Short { format: Yuyv, width: 640, height: 480,
                                             got: 460_800, expected: 614_400 });
        assert!(err.to_string().contains("640×480 YUYV"));
        // Read as what it is, the same bytes are a whole frame.
        assert!(to_yuyv(Nv12, 640, 480, None, &src).is_ok());
    }

    #[test]
    fn bad_sizes_are_rejected() {
        assert!(matches!(to_yuyv(Yuyv, 3, 2, None, &[0; 12]), Err(FrameError::BadSize { .. })));
        assert!(matches!(to_yuyv(Nv12, 0, 2, None, &[]), Err(FrameError::BadSize { .. })));
    }

    #[test]
    fn uyvy_is_reordered() {
        let f = to_yuyv(Uyvy, 2, 1, None, &[0x40, 0xA0, 0x80, 0xA1]).unwrap();
        assert_eq!(&f.data[..], &[0xA0, 0x40, 0xA1, 0x80]);
    }

    #[test]
    fn nv12_shares_chroma_between_row_pairs() {
        // 2×2: Y = 10 20 / 30 40, one Cb/Cr pair 0x40 0x80.
        let src = [10, 20, 30, 40, 0x40, 0x80];
        let f = to_yuyv(Nv12, 2, 2, None, &src).unwrap();
        assert_eq!(&f.data[..], &[10, 0x40, 20, 0x80, 30, 0x40, 40, 0x80]);
        // Odd height: the last luma row uses the last chroma row.
        let src = [1, 2, 3, 4, 5, 6, 0x10, 0x20, 0x30, 0x40];
        let f = to_yuyv(Nv12, 2, 3, None, &src).unwrap();
        assert_eq!(&f.data[8..], &[5, 0x30, 6, 0x40]);
    }

    #[test]
    fn rgb_uses_bt601_studio_range() {
        let white_black = [255, 255, 255, 0, 0, 0];
        let f = to_yuyv(Rgb24, 2, 1, None, &white_black).unwrap();
        assert_eq!(&f.data[..], &[235, 128, 16, 128]);
        // Pure red and pure blue, as RGB and as BGR.
        let red = [255, 0, 0, 255, 0, 0];
        let f = to_yuyv(Rgb24, 2, 1, None, &red).unwrap();
        assert_eq!(&f.data[..], &[82, 90, 82, 240]);
        let red_bgr = [0, 0, 255, 0, 0, 255];
        let f = to_yuyv(Bgr24, 2, 1, None, &red_bgr).unwrap();
        assert_eq!(&f.data[..], &[82, 90, 82, 240]);
        // An odd width drops the last column.
        let f = to_yuyv(Rgb24, 3, 1, None, &[0; 9]).unwrap();
        assert_eq!((f.width, f.data.len()), (2, 4));
    }

    #[test]
    fn gray_maps_to_studio_luma_with_neutral_chroma() {
        let f = to_yuyv(Gray, 2, 1, None, &[0, 255]).unwrap();
        assert_eq!(&f.data[..], &[16, 128, 235, 128]);
    }

    /// 16×8, baseline 4:2:2, left half red (220,20,20), right half blue
    /// (20,20,220), made with libjpeg-turbo's cjpeg (standard Huffman tables).
    const JPEG_16X8: &[u8] = include_bytes!("testdata/camera-16x8-422.jpg");

    fn check_red_blue(f: &YuyvFrame) {
        assert_eq!((f.width, f.height), (16, 8));
        assert_eq!(f.data.len(), 16 * 8 * 2);
        // Exact conversion: red Y 85 Cb 98 Cr 216, blue Y 53 Cb 216 Cr 114;
        // JPEG quantisation moves them a little.
        let row = &f.data[4 * 32..5 * 32];
        let (y, u, v) = (row[4] as i32, row[5] as i32, row[7] as i32);
        assert!((y - 85).abs() <= 6 && (u - 98).abs() <= 8 && (v - 216).abs() <= 8,
                "red pixel decoded as Y {y} Cb {u} Cr {v}");
        let (y, u, v) = (row[28] as i32, row[29] as i32, row[31] as i32);
        assert!((y - 53).abs() <= 6 && (u - 216).abs() <= 8 && (v - 114).abs() <= 8,
                "blue pixel decoded as Y {y} Cb {u} Cr {v}");
    }

    #[test]
    fn mjpeg_decodes_to_yuyv_at_its_own_size() {
        // Negotiated size is ignored for MJPEG: the JPEG header is the truth.
        let f = to_yuyv(Mjpeg, 1920, 1080, None, JPEG_16X8).unwrap();
        check_red_blue(&f);
        assert!(!f.was_jpeg);
    }

    /// Drop every DHT segment, the way UVC cameras send Motion-JPEG.
    fn strip_dht(jpeg: &[u8]) -> Vec<u8> {
        let mut out = jpeg[..2].to_vec();
        let mut i = 2;
        while i + 4 <= jpeg.len() && jpeg[i] == 0xFF {
            let marker = jpeg[i + 1];
            if marker == 0xDA { break; }
            let len = u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]) as usize;
            if marker != 0xC4 { out.extend_from_slice(&jpeg[i..i + 2 + len]); }
            i += 2 + len;
        }
        out.extend_from_slice(&jpeg[i..]);
        out
    }

    #[test]
    fn mjpeg_without_huffman_tables_uses_the_standard_ones() {
        let bare = strip_dht(JPEG_16X8);
        assert!(bare.len() < JPEG_16X8.len() - 400, "fixture had no DHT to strip");
        check_red_blue(&to_yuyv(Mjpeg, 16, 8, None, &bare).unwrap());
    }

    #[test]
    fn jpeg_data_under_an_uncompressed_label_is_decoded() {
        let f = to_yuyv(Yuyv, 1920, 1080, None, JPEG_16X8).unwrap();
        assert!(f.was_jpeg);
        check_red_blue(&f);
    }

    #[test]
    fn corrupt_mjpeg_is_an_error_not_a_crash() {
        assert!(matches!(to_yuyv(Mjpeg, 16, 8, None, &[]), Err(FrameError::Mjpeg(_))));
        assert!(matches!(to_yuyv(Mjpeg, 16, 8, None, &[0xFF, 0xD8, 0xFF]), Err(FrameError::Mjpeg(_))));
        let truncated = &JPEG_16X8[..JPEG_16X8.len() / 2];
        // Truncated scan data may decode to a partial image or fail; it must
        // not panic.
        let _ = to_yuyv(Mjpeg, 16, 8, None, truncated);
        let mut garbage = JPEG_16X8.to_vec();
        for b in garbage.iter_mut().skip(300) { *b = 0xA5; }
        let _ = to_yuyv(Mjpeg, 16, 8, None, &garbage);
    }
}
