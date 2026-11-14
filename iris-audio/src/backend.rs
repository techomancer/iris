//! What a host audio backend implements, and what it calls.
//!
//! A backend owns the device and the thread that serves it. The library
//! hands it a [`Render`] (for output) or a [`Capture`] (for input) and gets
//! back a [`StreamHandle`]; dropping the handle stops the stream.
//!
//! The backend pulls. It calls [`Render::render`] from its own thread
//! whenever the device wants frames, with a buffer of exactly the size the
//! device asked for, and the renderer always fills all of it. All buffers
//! are interleaved `f32`: a backend whose device wants something else
//! converts on its side.
//!
//! Following the device is the backend's job. If the default device
//! changes, or the server restarts, the backend reconnects by itself and
//! calls [`Render::configure`] with the new format before the next buffer.
//! Nothing above the backend, and nothing in the guest, takes part.

use std::time::Duration;

use crate::Error;
use crate::format::DeviceFormat;

#[cfg(all(target_os = "macos", feature = "coreaudio"))]
pub mod coreaudio;
pub mod file;
pub mod null;
#[cfg(all(unix, feature = "pulse"))]
pub mod pulse;
#[cfg(all(windows, feature = "wasapi"))]
pub mod wasapi;

/// What the library asks of a backend when it opens an output.
#[derive(Clone, Debug)]
pub struct OutputRequest {
    /// The stream's name, for the host's mixer.
    pub name: String,
    /// The guest's frame width. A backend opens this many channels if the
    /// host can take them (and lets the host mix them down if it has to),
    /// or the nearest layout it has; the renderer maps to whatever
    /// [`Render::configure`] reports.
    pub channels: usize,
    /// The guest's rate when the stream was opened: a hint for backends
    /// that can pick a device rate. The guest may change its rate later.
    pub rate: u32,
    /// How much the device should buffer beyond the library's own queue,
    /// or `None` for the backend's low-latency default.
    pub latency: Option<Duration>,
    /// Whether a stream is wanted even if no device can be opened now. A
    /// backend that honours this starts the stream on a clock of its own,
    /// discarding the frames ([`null`]), and moves it to a device when
    /// there is one; a failure that no device arriving would cure is still
    /// returned. A backend that does not honour it fails as usual.
    pub null_fallback: bool,
}

/// What the library asks of a backend when it opens an input.
#[derive(Clone, Debug)]
pub struct InputRequest {
    pub name: String,
    pub channels: usize,
    pub rate: u32,
    pub latency: Option<Duration>,
}

/// What a backend knows about the device at the time of one buffer.
#[derive(Clone, Copy, Debug, Default)]
pub struct RenderInfo {
    /// How long after this call the first frame of the buffer reaches the
    /// speaker, if the backend can tell.
    pub device_delay: Option<Duration>,
    /// Times the device ran dry because the backend's thread was late,
    /// since the stream opened. Not the guest's fault; see
    /// [`Stats::underruns`](crate::Stats::underruns) for that.
    pub device_underruns: u64,
}

/// What a backend knows about the device at the time of one captured buffer.
#[derive(Clone, Copy, Debug, Default)]
pub struct CaptureInfo {
    /// How long ago the first frame of the buffer was at the microphone.
    pub device_delay: Option<Duration>,
    /// Times the device overran because the backend's thread was late.
    pub device_overruns: u64,
}

/// The source of an output stream's frames, called on the backend's thread.
///
/// Neither method may block, lock or allocate: on a real device they run
/// on its real-time thread.
pub trait Render: Send + 'static {
    /// The device's format. Called before the first [`Render::render`] and
    /// before the first one after any change.
    fn configure(&mut self, format: DeviceFormat);

    /// Fill all of `out` with interleaved frames in the configured format.
    fn render(&mut self, out: &mut [f32], info: &RenderInfo);
}

/// The destination of an input stream's frames, called on the backend's
/// thread under the same rules as [`Render`].
pub trait Capture: Send + 'static {
    fn configure(&mut self, format: DeviceFormat);

    /// Take all of `input`, interleaved frames in the configured format.
    fn capture(&mut self, input: &[f32], info: &CaptureInfo);
}

/// A running stream. Dropping it stops the stream and joins its thread.
pub trait StreamHandle: Send {
    /// What the stream plays to or records from, for messages.
    fn device(&self) -> String;
}

/// A host audio system.
pub trait Backend: Send + Sync {
    /// A short name: `"pulse"`, `"file"`.
    fn name(&self) -> &'static str;

    /// Open an output stream on the default device and start it. The
    /// backend calls `render` until the handle is dropped.
    fn open_output(&self, request: &OutputRequest, render: Box<dyn Render>) -> Result<Box<dyn StreamHandle>, Error>;

    /// Open an input stream on the default device and start it.
    fn open_input(&self, _request: &InputRequest, _capture: Box<dyn Capture>) -> Result<Box<dyn StreamHandle>, Error> {
        Err(Error::Unsupported("audio input"))
    }
}
