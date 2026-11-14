//! A backend that plays into a WAV file in real time.
//!
//! A thread stands in for the sound card: every [`FileBackend::period`] it
//! takes the frames that are due by the host's clock and appends them to
//! the file. A stream behaves exactly as it would on a card at that rate
//! and period, underruns included, which makes this the backend for
//! machines with no sound and for tests that have to hear the result.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::Error;
use crate::backend::{Backend, OutputRequest, Render, RenderInfo, StreamHandle};
use crate::format::DeviceFormat;
use crate::wav::WavWriter;

pub struct FileBackend {
    path: PathBuf,
    rate: u32,
    period: Duration,
    /// One stream at a time: a second one would overwrite the first's file.
    busy: Arc<AtomicBool>,
}

impl FileBackend {
    /// Play into `path` at 48 kHz in 5 ms periods.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into(), rate: 48000, period: Duration::from_millis(5), busy: Arc::new(AtomicBool::new(false)) }
    }

    /// The file's rate.
    pub fn rate(mut self, rate: u32) -> Self {
        self.rate = rate;
        self
    }

    /// How often the "card" takes frames.
    pub fn period(mut self, period: Duration) -> Self {
        self.period = period;
        self
    }

    pub fn path(&self) -> &Path { &self.path }
}

struct FileStream {
    path: PathBuf,
    stop: Arc<AtomicBool>,
    busy: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl StreamHandle for FileStream {
    fn device(&self) -> String { format!("file {}", self.path.display()) }
}

impl Drop for FileStream {
    fn drop(&mut self) {
        self.stop.store(true, Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        self.busy.store(false, Relaxed);
    }
}

impl Backend for FileBackend {
    fn name(&self) -> &'static str { "file" }

    fn open_output(&self, request: &OutputRequest, mut render: Box<dyn Render>) -> Result<Box<dyn StreamHandle>, Error> {
        if self.busy.swap(true, Relaxed) {
            return Err(Error::Backend { backend: "file", message: format!("{} is already being played into", self.path.display()) });
        }
        let format = DeviceFormat { rate: self.rate, channels: request.channels };
        let spawned = WavWriter::create(&self.path, format.rate, format.channels).map_err(Error::from).and_then(|mut wav| {
            let stop = Arc::new(AtomicBool::new(false));
            let (stop_t, period) = (stop.clone(), self.period);
            let thread = std::thread::Builder::new().name("iris-audio file".into()).spawn(move || {
                render.configure(format);
                let start = Instant::now();
                let mut played = 0u64;
                let mut buf = Vec::new();
                while !stop_t.load(Relaxed) {
                    std::thread::sleep(period);
                    let due = (start.elapsed().as_nanos() * u128::from(format.rate) / 1_000_000_000) as u64;
                    buf.clear();
                    buf.resize((due - played) as usize * format.channels, 0.0);
                    render.render(&mut buf, &RenderInfo { device_delay: Some(Duration::ZERO), device_underruns: 0 });
                    played = due;
                    // A full disk ends the stream; the guest sees a device
                    // that stopped taking frames.
                    if wav.append(&buf).is_err() {
                        return;
                    }
                }
            })?;
            Ok((stop, thread))
        });
        match spawned {
            Ok((stop, thread)) => {
                Ok(Box::new(FileStream { path: self.path.clone(), stop, busy: self.busy.clone(), thread: Some(thread) }))
            }
            Err(e) => {
                self.busy.store(false, Relaxed);
                Err(e)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resample::tests::rms;
    use crate::wav::tests::Temp;
    use crate::{Audio, OutputConfig};

    #[test]
    fn a_tone_trickled_in_real_time_arrives_whole() {
        let tmp = Temp::new("file-backend.wav");
        let audio = Audio::with_backend(Arc::new(FileBackend::new(&tmp.0)));
        let cfg = OutputConfig { rate: 22050, prebuffer: Duration::from_millis(20), ..Default::default() };
        let mut out = audio.open_output(cfg).unwrap();
        assert!(out.device().unwrap().starts_with("file "));

        // A guest that writes what is due every millisecond or so.
        let start = Instant::now();
        let (mut written, mut phase) = (0u64, 0.0f64);
        while start.elapsed() < Duration::from_millis(1500) {
            let due = (start.elapsed().as_secs_f64() * 22050.0) as u64;
            let block: Vec<f32> = (written..due)
                .flat_map(|_| {
                    phase += 2.0 * std::f64::consts::PI * 440.0 / 22050.0;
                    let v = (phase.sin() * 0.5) as f32;
                    [v, v]
                })
                .collect();
            out.write(&block);
            written = due;
            std::thread::sleep(Duration::from_millis(1));
        }
        let stats = out.stats();
        drop(out);

        assert_eq!(stats.device, Some(DeviceFormat { rate: 48000, channels: 2 }));
        let (rate, channels, pcm) = crate::wav::read(&tmp.0).unwrap();
        assert_eq!((rate, channels), (48000, 2));
        // About a second and a half of file, by the host's clock.
        let frames = pcm.len() / 2;
        assert!((60000..=84000).contains(&frames), "{frames} frames");
        // A loaded test machine may stall the writer; only a clean run
        // says anything about the tone.
        if stats.underruns == 0 {
            let left: Vec<f32> = pcm.chunks(2).skip(9600).take(48000).map(|f| f32::from(f[0]) / 32768.0).collect();
            let tone = rms(&left) * std::f64::consts::SQRT_2;
            assert!((tone - 0.5).abs() < 0.02, "tone level {tone}");
        }
    }

    #[test]
    fn a_second_stream_on_the_same_file_is_refused_until_the_first_ends() {
        let tmp = Temp::new("file-busy.wav");
        let audio = Audio::with_backend(Arc::new(FileBackend::new(&tmp.0)));
        let first = audio.open_output(OutputConfig::default()).unwrap();
        assert!(audio.open_output(OutputConfig::default()).is_err());
        drop(first);
        assert!(audio.open_output(OutputConfig::default()).is_ok());
    }

    #[test]
    fn an_unwritable_file_opens_no_stream() {
        let audio = Audio::with_backend(Arc::new(FileBackend::new("/nonexistent-dir/out.wav")));
        assert!(audio.open_output(OutputConfig::default()).is_err());
        // And the failure does not leave the backend marked busy.
        assert!(matches!(audio.open_output(OutputConfig::default()), Err(Error::Io(_))));
    }
}
