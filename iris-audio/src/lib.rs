//! Low-latency audio output for emulators.
//!
//! An emulated sound device produces frames on the guest's schedule: at
//! whatever rate the guest programmed, a few at a time from a timer
//! interrupt or in blocks from a DMA buffer, and by a clock that is not
//! the host's. A host sound card wants frames on its own schedule, at its
//! own rate. This crate sits between the two:
//!
//! ```no_run
//! use iris_audio::{Audio, BackendChoice, OutputConfig};
//! use std::time::Duration;
//!
//! let audio = Audio::new(BackendChoice::Default)?;
//! let mut out = audio.open_output(OutputConfig {
//!     name: "HAL2".into(),
//!     rate: 44100,
//!     prebuffer: Duration::from_millis(20),
//!     ..Default::default()
//! })?;
//!
//! // From the device's thread, whenever the guest has frames:
//! let frames: [i16; 8] = [0; 8]; // four stereo frames
//! out.write(&frames);
//! // When the guest reprograms its codec:
//! out.set_rate(48000)?;
//! # Ok::<(), iris_audio::Error>(())
//! ```
//!
//! [`Output::write`] never blocks. Rate conversion, following the guest's
//! clock, prebuffering and following the host's default device all happen
//! on the other side of a lock-free ring; see [`stream`] for how, and
//! [`backend`] for what a host audio system has to provide.
//!
//! Backends so far: PulseAudio's native protocol (which PipeWire also
//! serves), WASAPI on Windows, CoreAudio on macOS, a WAV file played in
//! real time, and one that plays to nothing. Input has its interface
//! ([`Capture`], [`Backend::open_input`], [`Input`]) but no implementation.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

pub mod backend;
pub mod format;
pub mod input;
pub mod mix;
pub mod resample;
pub mod stream;
pub mod tee;
pub mod wav;

#[cfg(all(target_os = "macos", feature = "coreaudio"))]
pub use backend::coreaudio::CoreAudioBackend;
pub use backend::file::FileBackend;
pub use backend::null::NullBackend;
#[cfg(all(unix, feature = "pulse"))]
pub use backend::pulse::PulseBackend;
#[cfg(all(windows, feature = "wasapi"))]
pub use backend::wasapi::WasapiBackend;
pub use backend::{Backend, Capture, CaptureInfo, InputRequest, OutputRequest, Render, RenderInfo, StreamHandle};
pub use format::{DeviceFormat, MAX_CHANNELS, Sample};
pub use input::{Input, InputConfig};
pub use mix::Mixer;
pub use stream::{Output, OutputConfig, Renderer, Stats};

/// Why something could not be opened or set.
#[derive(Debug)]
pub enum Error {
    /// The backend, or this build, cannot do that at all.
    Unsupported(&'static str),
    /// A configuration value is out of range.
    Config(String),
    /// The host's audio system refused or failed.
    Backend { backend: &'static str, message: String },
    Io(std::io::Error),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Unsupported(what) => write!(f, "{what} is not supported"),
            Error::Config(message) => write!(f, "bad audio configuration: {message}"),
            Error::Backend { backend, message } => write!(f, "{backend}: {message}"),
            Error::Io(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self { Error::Io(e) }
}

/// Which host audio system to use.
#[derive(Clone, Debug, Default)]
pub enum BackendChoice {
    /// The platform's usual one.
    #[default]
    Default,
    /// PulseAudio or PipeWire's pipewire-pulse.
    Pulse,
    /// WASAPI, on Windows.
    Wasapi,
    /// CoreAudio, on macOS.
    CoreAudio,
    /// Nothing: frames are taken in real time and discarded; see
    /// [`NullBackend`].
    Null,
    /// A WAV file played in real time; see [`FileBackend`].
    File(PathBuf),
}

/// A host audio system, ready to open streams.
#[derive(Clone)]
pub struct Audio {
    backend: Arc<dyn Backend>,
    app: String,
}

impl Audio {
    /// Open a backend. Streams appear in the host's mixer under the
    /// running program's name; [`Audio::named`] sets another.
    pub fn new(choice: BackendChoice) -> Result<Self, Error> {
        let app = std::env::current_exe()
            .ok()
            .and_then(|p| p.file_stem().map(|s| s.to_string_lossy().into_owned()))
            .unwrap_or_else(|| "iris-audio".into());
        Self::named(choice, app)
    }

    /// [`Audio::new`] with the application's name given.
    pub fn named(choice: BackendChoice, app: impl Into<String>) -> Result<Self, Error> {
        let app = app.into();
        let backend: Arc<dyn Backend> = match choice {
            BackendChoice::File(path) => Arc::new(FileBackend::new(path)),
            BackendChoice::Null => Arc::new(NullBackend::new()),
            #[cfg(all(unix, feature = "pulse"))]
            BackendChoice::Pulse => Arc::new(PulseBackend::new(app.clone())),
            #[cfg(not(all(unix, feature = "pulse")))]
            BackendChoice::Pulse => return Err(Error::Unsupported("the PulseAudio backend")),
            #[cfg(all(windows, feature = "wasapi"))]
            BackendChoice::Wasapi | BackendChoice::Default => Arc::new(WasapiBackend::new()),
            #[cfg(not(all(windows, feature = "wasapi")))]
            BackendChoice::Wasapi => return Err(Error::Unsupported("the WASAPI backend")),
            // Every Unix but macOS: Linux, and the BSDs until they have a
            // backend for their own sound.
            #[cfg(all(unix, not(target_os = "macos"), feature = "pulse"))]
            BackendChoice::Default => Arc::new(PulseBackend::new(app.clone())),
            #[cfg(all(target_os = "macos", feature = "coreaudio"))]
            BackendChoice::CoreAudio | BackendChoice::Default => Arc::new(CoreAudioBackend::new()),
            #[cfg(not(all(target_os = "macos", feature = "coreaudio")))]
            BackendChoice::CoreAudio => return Err(Error::Unsupported("the CoreAudio backend")),
            #[cfg(not(any(
                all(unix, not(target_os = "macos"), feature = "pulse"),
                all(windows, feature = "wasapi"),
                all(target_os = "macos", feature = "coreaudio")
            )))]
            BackendChoice::Default => return Err(Error::Unsupported("audio on this platform (no backend yet)")),
        };
        Ok(Self { backend, app })
    }

    /// Use a backend built by the caller.
    pub fn with_backend(backend: Arc<dyn Backend>) -> Self {
        Self { backend, app: "iris-audio".into() }
    }

    /// The backend's short name.
    pub fn backend(&self) -> &'static str { self.backend.name() }

    /// The application name streams are opened under.
    pub fn app(&self) -> &str { &self.app }

    /// Open an output stream and start it. The stream plays silence until
    /// the guest has written its prebuffer.
    pub fn open_output(&self, config: OutputConfig) -> Result<Output, Error> {
        Ok(self.open_outputs(&[config])?.remove(0))
    }

    /// Open several outputs that play mixed into one stream on the host:
    /// one connection and one entry in its mixer. Each is a stream of its
    /// own in every other respect, with its own rate, prebuffer, clock
    /// follower and counters, for a device with several DACs that run at
    /// different rates into one jack. The outputs come back in the order
    /// of `configs`; the stream stops when the last of them is dropped,
    /// and one dropped earlier just falls silent.
    ///
    /// What belongs to the stream and not to a voice is taken from the
    /// first configuration (`name`, `device_latency`, `tee`, and the rate
    /// suggested to the backend), except that the stream is as wide as
    /// the widest voice and `null_fallback` holds if any asks for it.
    pub fn open_outputs(&self, configs: &[OutputConfig]) -> Result<Vec<Output>, Error> {
        let Some(first) = configs.first() else {
            return Err(Error::Config("no outputs to open".into()));
        };
        let mut outputs = Vec::with_capacity(configs.len());
        let mut renderers = Vec::with_capacity(configs.len());
        for config in configs {
            let (output, renderer) = Output::detached(config)?;
            outputs.push(output);
            renderers.push(renderer);
        }
        let voices: Box<dyn Render> = if renderers.len() == 1 {
            Box::new(renderers.remove(0))
        } else {
            Box::new(Mixer::new(renderers))
        };
        let render: Box<dyn Render> = match &first.tee {
            Some(path) => Box::new(tee::Tee::new(voices, path)?),
            None => voices,
        };
        let request = OutputRequest {
            name: first.name.clone(),
            channels: configs.iter().map(|c| c.channels).max().unwrap_or(first.channels),
            rate: first.rate,
            latency: first.device_latency,
            null_fallback: configs.iter().any(|c| c.null_fallback),
        };
        let handle = Arc::new(Mutex::new(self.backend.open_output(&request, render)?));
        for output in &mut outputs {
            output.attach(handle.clone());
        }
        Ok(outputs)
    }

    /// Open an input stream. No backend implements input yet, so this
    /// fails with [`Error::Unsupported`]; the interface is here so that
    /// devices and backends can be written against it.
    pub fn open_input(&self, config: InputConfig) -> Result<Input, Error> {
        input::open(&*self.backend, config)
    }
}
