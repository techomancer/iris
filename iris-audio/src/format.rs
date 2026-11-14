//! Sample types and the format a device runs at.

/// The widest frame a stream carries.
pub const MAX_CHANNELS: usize = 8;

/// A sample type a guest can write or read. Everything is `f32` in
/// `-1.0..1.0` inside the library; this converts at the edge.
pub trait Sample: Copy + Send + 'static {
    fn to_f32(self) -> f32;
    fn from_f32(v: f32) -> Self;
}

impl Sample for f32 {
    #[inline]
    fn to_f32(self) -> f32 { self }
    #[inline]
    fn from_f32(v: f32) -> Self { v }
}

impl Sample for i16 {
    #[inline]
    fn to_f32(self) -> f32 { f32::from(self) / 32768.0 }
    #[inline]
    fn from_f32(v: f32) -> Self {
        (v * 32768.0).round().clamp(-32768.0, 32767.0) as i16
    }
}

/// Full-scale 32-bit samples. A 24-bit codec's samples go in shifted left
/// by eight.
impl Sample for i32 {
    #[inline]
    fn to_f32(self) -> f32 { (f64::from(self) / 2147483648.0) as f32 }
    #[inline]
    fn from_f32(v: f32) -> Self {
        (f64::from(v) * 2147483648.0).round().clamp(-2147483648.0, 2147483647.0) as i32
    }
}

/// What a backend's device runs at. A backend tells its [`Render`] or
/// [`Capture`] this before the first buffer and again whenever it changes
/// (a new default device, a reconnect).
///
/// [`Render`]: crate::Render
/// [`Capture`]: crate::Capture
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceFormat {
    /// Frames per second.
    pub rate: u32,
    /// Samples per interleaved frame.
    pub channels: usize,
}
