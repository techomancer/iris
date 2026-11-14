//! A backend that plays to nothing, in real time.
//!
//! A thread stands in for the sound card: every [`NullBackend::period`] it
//! takes the frames that are due by the host's clock and throws them away.
//! To the guest it is a device like any other: frames are consumed at the
//! device's pace, the queue holds its level and nothing is dropped. It is
//! for running an emulator with its sound switched off, and for a machine
//! with no output at all, where a guest that waits for its samples to be
//! played would otherwise wait for ever.
//!
//! The clock behind it is also what a backend for a real device falls back
//! on while its device is gone.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::Error;
use crate::backend::{Backend, OutputRequest, Render, RenderInfo, StreamHandle};
use crate::format::DeviceFormat;

/// The most a [`Pacer`] takes in one step. A thread that was away for
/// longer does not make the time up: the frames it missed are the guest's
/// to keep or drop, as with a real device that stalled.
const LONGEST_STEP: Duration = Duration::from_millis(100);

/// The clock of a device that is not there: renders the frames that are due
/// by the host's clock and discards them.
pub(crate) struct Pacer {
    format: DeviceFormat,
    start: Instant,
    /// Frames due at the last step.
    played: u64,
    scratch: Vec<f32>,
}

impl Pacer {
    /// A clock that starts now. `format` is what the renderer was last
    /// configured with. Allocates, so not for a real-time thread.
    pub(crate) fn new(format: DeviceFormat) -> Self {
        let frames = (LONGEST_STEP.as_secs_f64() * f64::from(format.rate)).ceil() as usize;
        Self { format, start: Instant::now(), played: 0, scratch: vec![0.0; frames.max(1) * format.channels] }
    }

    #[allow(dead_code)] // Not every backend asks.
    pub(crate) fn format(&self) -> DeviceFormat { self.format }

    /// Render what has become due since the last step, and discard it.
    pub(crate) fn step(&mut self, render: &mut dyn Render, device_underruns: u64) {
        let due = (self.start.elapsed().as_nanos() * u128::from(self.format.rate) / 1_000_000_000) as u64;
        let frames = ((due - self.played) as usize).min(self.scratch.len() / self.format.channels);
        self.played = due;
        if frames > 0 {
            // Nothing reaches a speaker, so there is no delay to report.
            let info = RenderInfo { device_delay: None, device_underruns };
            render.render(&mut self.scratch[..frames * self.format.channels], &info);
        }
    }
}

pub struct NullBackend {
    rate: u32,
    period: Duration,
}

impl NullBackend {
    /// Discard at 48 kHz in 5 ms periods.
    pub fn new() -> Self {
        Self { rate: 48000, period: Duration::from_millis(5) }
    }

    /// The rate of the device that is not there.
    pub fn rate(mut self, rate: u32) -> Self {
        self.rate = rate;
        self
    }

    /// How often the "card" takes frames.
    pub fn period(mut self, period: Duration) -> Self {
        self.period = period;
        self
    }
}

impl Default for NullBackend {
    fn default() -> Self { Self::new() }
}

struct NullStream {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl StreamHandle for NullStream {
    fn device(&self) -> String { "null".into() }
}

impl Drop for NullStream {
    fn drop(&mut self) {
        self.stop.store(true, Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Backend for NullBackend {
    fn name(&self) -> &'static str { "null" }

    fn open_output(&self, request: &OutputRequest, mut render: Box<dyn Render>) -> Result<Box<dyn StreamHandle>, Error> {
        if self.rate == 0 {
            return Err(Error::Config("the null backend's rate is 0".into()));
        }
        let format = DeviceFormat { rate: self.rate, channels: request.channels };
        let stop = Arc::new(AtomicBool::new(false));
        let (stop_t, period) = (stop.clone(), self.period);
        let thread = std::thread::Builder::new().name("iris-audio null".into()).spawn(move || {
            render.configure(format);
            let mut pacer = Pacer::new(format);
            while !stop_t.load(Relaxed) {
                std::thread::sleep(period);
                pacer.step(&mut *render, 0);
            }
        })?;
        Ok(Box::new(NullStream { stop, thread: Some(thread) }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Audio, OutputConfig};

    #[test]
    fn a_guest_playing_to_nothing_is_consumed_at_the_devices_pace() {
        let audio = Audio::with_backend(Arc::new(NullBackend::new()));
        let cfg = OutputConfig { rate: 22050, prebuffer: Duration::from_millis(20), ..Default::default() };
        let mut out = audio.open_output(cfg).unwrap();
        assert_eq!(out.device().unwrap(), "null");

        // A guest that writes what is due every millisecond or so.
        let start = Instant::now();
        let mut written = 0u64;
        while start.elapsed() < Duration::from_millis(1000) {
            let due = (start.elapsed().as_secs_f64() * 22050.0) as u64;
            out.write(&vec![0.25f32; (due - written) as usize * 2]);
            written = due;
            std::thread::sleep(Duration::from_millis(1));
        }
        let stats = out.stats();
        assert_eq!(stats.device, Some(DeviceFormat { rate: 48000, channels: 2 }));
        assert_eq!(stats.dropped, 0);
        // All but what is still queued has been taken, counted in the
        // device's frames; a loaded test machine may have left a little
        // more behind.
        let taken = stats.played * 22050 / 48000;
        assert!(taken + 22050 / 5 >= stats.written, "{taken} of {} frames", stats.written);
    }

    #[test]
    fn a_pacer_that_was_away_takes_no_more_than_its_longest_step() {
        struct Count(Arc<std::sync::atomic::AtomicUsize>);
        impl Render for Count {
            fn configure(&mut self, _: DeviceFormat) {}
            fn render(&mut self, out: &mut [f32], _: &RenderInfo) {
                self.0.fetch_add(out.len() / 2, Relaxed);
            }
        }
        let frames = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut render = Count(frames.clone());
        let mut pacer = Pacer::new(DeviceFormat { rate: 48000, channels: 2 });
        std::thread::sleep(Duration::from_millis(250));
        pacer.step(&mut render, 0);
        assert_eq!(frames.load(Relaxed), 4800);
        // And the time it missed is not owed afterwards.
        pacer.step(&mut render, 0);
        assert!(frames.load(Relaxed) < 4800 + 480);
    }
}
