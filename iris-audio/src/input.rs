//! Input streams: the interface, with nothing behind it yet.
//!
//! Input is output turned round. The backend's thread hands captured
//! device-rate frames to a [`Capture`](crate::Capture); that end converts
//! them to the guest's rate, following the guest's clock the same way, and
//! queues them in a ring; the guest's thread takes them with
//! [`Input::read`] and gets silence when the ring is empty. The types
//! below fix that shape so that devices and backends can be written
//! against it. Until a backend implements
//! [`Backend::open_input`], no [`Input`] can
//! exist.

use std::convert::Infallible;
use std::time::Duration;

use crate::Error;
use crate::backend::Backend;
use crate::format::{MAX_CHANNELS, Sample};

/// How to open an input stream.
#[derive(Clone, Debug)]
pub struct InputConfig {
    /// The stream's name in the host's mixer.
    pub name: String,
    /// Samples per guest frame.
    pub channels: usize,
    /// The rate the guest's ADC runs at now.
    pub rate: u32,
    /// The highest rate the guest will ever set.
    pub max_rate: u32,
    /// How much captured sound to hold for the guest: the margin against
    /// the guest reading in bursts, and the latency the library adds.
    pub buffer: Duration,
    /// How much the device should buffer, or `None` for the backend's
    /// low-latency default.
    pub device_latency: Option<Duration>,
}

impl Default for InputConfig {
    fn default() -> Self {
        Self {
            name: "iris-audio".into(),
            channels: 2,
            rate: 48000,
            max_rate: 96000,
            buffer: Duration::from_millis(10),
            device_latency: None,
        }
    }
}

/// The guest's end of an input stream. Cannot be constructed yet.
pub struct Input {
    never: Infallible,
}

impl Input {
    /// Fill `frames` with captured interleaved frames at the current rate.
    /// Never blocks: what has not been captured yet is silence. Returns
    /// the frames that were real.
    pub fn read<S: Sample>(&mut self, _frames: &mut [S]) -> usize {
        match self.never {}
    }

    /// The guest reprogrammed its ADC.
    pub fn set_rate(&mut self, _rate: u32) -> Result<(), Error> {
        match self.never {}
    }

    pub fn rate(&self) -> u32 {
        match self.never {}
    }

    pub fn channels(&self) -> usize {
        match self.never {}
    }
}

pub(crate) fn open(_backend: &dyn Backend, config: InputConfig) -> Result<Input, Error> {
    if !(1..=MAX_CHANNELS).contains(&config.channels) {
        return Err(Error::Config(format!("{} channels; 1 to {MAX_CHANNELS} are supported", config.channels)));
    }
    // When there is a recorder to hand over, this becomes
    // `backend.open_input(&request, Box::new(recorder))`.
    Err(Error::Unsupported("audio input"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Audio, FileBackend};
    use std::sync::Arc;

    #[test]
    fn input_is_declared_but_not_there_yet() {
        let audio = Audio::with_backend(Arc::new(FileBackend::new("unused.wav")));
        assert!(matches!(audio.open_input(InputConfig::default()), Err(Error::Unsupported(_))));
    }
}
