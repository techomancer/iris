//! A [`Render`] that also records what it renders to a WAV file.
//!
//! The file is written by a thread of its own; the audio thread only
//! copies into a ring, and drops (and counts) what the ring cannot take
//! rather than wait for a disk.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::thread::JoinHandle;
use std::time::Duration;

use rtrb::{Consumer, Producer, RingBuffer};

use crate::Error;
use crate::backend::{Render, RenderInfo};
use crate::format::DeviceFormat;
use crate::wav::WavWriter;

/// Samples the ring holds: 2.7 s of 48 kHz stereo.
const RING_SAMPLES: usize = 1 << 18;
/// How often the writer thread empties the ring.
const TICK: Duration = Duration::from_millis(20);

pub struct Tee {
    inner: Box<dyn Render>,
    tx: Producer<f32>,
    /// The format the writer should be writing: a generation count above
    /// the rate (24 bits) and channels (8 bits).
    format: Arc<AtomicU64>,
    /// Samples put in the ring before the format last changed: where the
    /// writer has to switch files.
    boundary: Arc<AtomicU64>,
    sent: u64,
    generation: u64,
    lost: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

fn pack(generation: u64, f: DeviceFormat) -> u64 {
    generation << 32 | u64::from(f.rate & 0xff_ffff) << 8 | f.channels as u64
}

/// The file for the `n`th format: the path itself, then `name.2.wav`, ...
fn nth_path(path: &Path, n: u64) -> PathBuf {
    if n <= 1 {
        return path.to_path_buf();
    }
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("wav");
    path.with_extension(format!("{n}.{ext}"))
}

impl Tee {
    /// Wrap `inner`. The file is created now, so a path that cannot be
    /// written is an error here and not a silent recording later; it gets
    /// its real header when the backend reports the device's format. If
    /// the format changes later (another device), the recording carries on
    /// in `name.2.wav` and so on. Changes are expected seconds apart: of
    /// two within one writer tick (20 ms) only the second gets a file.
    pub fn new(inner: Box<dyn Render>, path: &Path) -> Result<Self, Error> {
        std::fs::File::create(path)?;
        let (tx, rx) = RingBuffer::new(RING_SAMPLES);
        let format = Arc::new(AtomicU64::new(0));
        let boundary = Arc::new(AtomicU64::new(0));
        let lost = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let thread = {
            let (format, boundary, stop, path) = (format.clone(), boundary.clone(), stop.clone(), path.to_path_buf());
            std::thread::Builder::new().name("iris-audio tee".into()).spawn(move || writer(rx, format, boundary, stop, path))?
        };
        Ok(Self { inner, tx, format, boundary, sent: 0, generation: 0, lost, stop, thread: Some(thread) })
    }
}

fn writer(mut rx: Consumer<f32>, format: Arc<AtomicU64>, boundary: Arc<AtomicU64>, stop: Arc<AtomicBool>, path: PathBuf) {
    let mut current = 0u64;
    let mut received = 0u64;
    let mut wav: Option<WavWriter> = None;
    let mut buf = Vec::new();
    loop {
        let stopping = stop.load(Relaxed);
        let want = format.load(Relaxed);
        if want != current {
            // What was rendered before the change belongs to the old file.
            let old = boundary.load(Relaxed).saturating_sub(received);
            received += drain(&mut rx, old as usize, &mut buf, &mut wav);
            current = want;
            let (rate, channels) = ((want >> 8 & 0xff_ffff) as u32, (want & 0xff) as usize);
            wav = WavWriter::create(&nth_path(&path, want >> 32), rate, channels).ok();
        }
        received += drain(&mut rx, usize::MAX, &mut buf, &mut wav);
        if stopping {
            return;
        }
        std::thread::sleep(TICK);
    }
}

/// Move up to `limit` samples from the ring to the file; returns how many.
fn drain(rx: &mut Consumer<f32>, limit: usize, buf: &mut Vec<f32>, wav: &mut Option<WavWriter>) -> u64 {
    let n = rx.slots().min(limit);
    let Ok(chunk) = rx.read_chunk(n) else { return 0 };
    let (a, b) = chunk.as_slices();
    buf.clear();
    buf.extend_from_slice(a);
    buf.extend_from_slice(b);
    chunk.commit_all();
    if let Some(w) = wav {
        // A full disk or a full WAV ends the recording, not the sound.
        if w.append(buf).is_err() {
            *wav = None;
        }
    }
    n as u64
}

impl Render for Tee {
    fn configure(&mut self, format: DeviceFormat) {
        self.inner.configure(format);
        self.generation += 1;
        // The boundary first: the writer reads it after it sees the format.
        self.boundary.store(self.sent, Relaxed);
        self.format.store(pack(self.generation, format), Relaxed);
    }

    fn render(&mut self, out: &mut [f32], info: &RenderInfo) {
        self.inner.render(out, info);
        match self.tx.write_chunk_uninit(out.len()) {
            Ok(chunk) => {
                chunk.fill_from_iter(out.iter().copied());
                self.sent += out.len() as u64;
            }
            Err(_) => {
                self.lost.fetch_add(out.len() as u64, Relaxed);
            }
        }
    }
}

impl Drop for Tee {
    fn drop(&mut self) {
        self.stop.store(true, Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wav::tests::Temp;

    /// Renders a rising count, so a recording can be checked sample for sample.
    struct Count(i16);

    impl Render for Count {
        fn configure(&mut self, _: DeviceFormat) {}
        fn render(&mut self, out: &mut [f32], _: &RenderInfo) {
            for v in out {
                *v = f32::from(self.0) / 32768.0;
                self.0 += 1;
            }
        }
    }

    #[test]
    fn the_file_holds_exactly_what_was_rendered() {
        let tmp = Temp::new("tee.wav");
        let mut tee = Tee::new(Box::new(Count(0)), &tmp.0).unwrap();
        tee.configure(DeviceFormat { rate: 44100, channels: 2 });
        let mut buf = [0.0f32; 256];
        for _ in 0..10 {
            tee.render(&mut buf, &RenderInfo::default());
        }
        assert_eq!(buf[255], 2559.0 / 32768.0);
        drop(tee);
        let (rate, channels, pcm) = crate::wav::read(&tmp.0).unwrap();
        assert_eq!((rate, channels), (44100, 2));
        assert_eq!(pcm, (0..2560).collect::<Vec<i16>>());
    }

    #[test]
    fn a_new_format_starts_a_new_file() {
        let tmp = Temp::new("tee-move.wav");
        let second = Temp(nth_path(&tmp.0, 2));
        let mut tee = Tee::new(Box::new(Count(0)), &tmp.0).unwrap();
        let mut buf = [0.0f32; 96];
        tee.configure(DeviceFormat { rate: 48000, channels: 2 });
        tee.render(&mut buf, &RenderInfo::default());
        // Let the writer see the first format before the second.
        std::thread::sleep(TICK * 3);
        tee.configure(DeviceFormat { rate: 44100, channels: 6 });
        tee.render(&mut buf, &RenderInfo::default());
        drop(tee);
        let (rate, channels, pcm) = crate::wav::read(&tmp.0).unwrap();
        assert_eq!((rate, channels, pcm.len()), (48000, 2, 96));
        let (rate, channels, pcm) = crate::wav::read(&second.0).unwrap();
        assert_eq!((rate, channels, pcm.len()), (44100, 6, 96));
    }

    #[test]
    fn an_unwritable_path_is_an_error() {
        assert!(Tee::new(Box::new(Count(0)), Path::new("/nonexistent-dir/tee.wav")).is_err());
    }
}
