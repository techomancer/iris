//! Host audio output shared by the machines' sound devices: a persistent
//! cpal stream fed through a lock-free ring, and the resampler from a
//! device's own rate to the stream's. HAL2 (Indy, Indigo2) and MACE audio
//! (O2) both play through it.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use crate::config::{AudioConfig, AudioHost};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rtrb::{Consumer, Producer, RingBuffer};

// Sample rates to try when opening the persistent output stream, in order.
const PREFERRED_RATES: &[u32] = &[48000, 44100, 22050];
/// The PulseAudio stream's period, in frames, unless `[audio]
/// cpal_buffer_frames` says otherwise (see `output_buffer`).
#[cfg(target_os = "linux")]
const PULSE_FRAMES: u32 = 512;

/// A host output stream: i16 stereo pairs pushed into `producer` play at
/// `stream_rate`, and the stream plays silence when the ring is empty.
pub struct HostOut {
    pub stream_rate: u32,
    pub producer: Producer<i16>,
    pub underruns: Arc<AtomicU64>,
    // Set once prebuffering finishes and real samples are flowing; cleared when
    // the codec is disarmed/reset. Gates underrun counting so idle silence
    // (stream open, nothing enabled yet) isn't reported as an underrun.
    pub playing: Arc<AtomicBool>,
    // What plays the ring -- a host stream or an `[audio] file` -- dropped
    // with the HostOut. Neither for a detached one.
    _stream: Option<cpal::Stream>,
    _file: Option<FilePlayer>,
}

impl HostOut {
    /// An output with no host stream behind it: whoever holds the ring's
    /// consumer plays it. For tests of the devices that feed one.
    #[cfg(test)]
    pub fn detached(stream_rate: u32, producer: Producer<i16>) -> Self {
        Self {
            stream_rate,
            producer,
            underruns: Arc::new(AtomicU64::new(0)),
            playing: Arc::new(AtomicBool::new(true)),
            _stream: None,
            _file: None,
        }
    }
}

// cpal::Stream is !Send/!Sync on some platforms (ALSA uses raw pointers internally),
// but it is safe to hold inside a Mutex.
unsafe impl Send for HostOut {}
unsafe impl Sync for HostOut {}

// Simple skip/repeat resampler using a fixed-point accumulator.
// Produces output at `out_rate` from input at `in_rate`.
// Call `push_sample` for every input sample pair; it pushes 0, 1, or 2 pairs to the ring.
/// Converts the codec's rate to the host stream's by Catmull-Rom interpolation.
///
/// It used to repeat the last sample (zero-order hold). From 44.1 or 48 kHz
/// to 48 kHz that is nearly harmless, but Quake plays at 11025 Hz, where each
/// sample became a 4-or-5-sample step: the staircase put loud images of every
/// sound around 11 kHz and its multiples, and the games sounded metallic and
/// robotic. A real DAC at 11025 Hz filters those images out; interpolating
/// through the samples does most of the same.
pub struct Resampler {
    pub in_rate: u32,
    pub out_rate: u32,
    // Position of the next output between h[1] and h[2], in 1/out_rate
    // input-sample units: an exact rational step, so no drift.
    acc: u64,
    // The last four input frames, oldest first; output is interpolated
    // between h[1] and h[2] (two input samples of latency).
    h: [[f32; 2]; 4],
}

impl Resampler {
    pub fn new(in_rate: u32, out_rate: u32) -> Self {
        Self { in_rate, out_rate, acc: 0, h: [[0.0; 2]; 4] }
    }

    pub fn passthrough(&self) -> bool { self.in_rate == self.out_rate }

    /// Push one input stereo pair; emits the output pairs it completes. A
    /// pair the ring has no room for is dropped whole, or the channels of
    /// every later pair would swap.
    pub fn push(&mut self, l: i16, r: i16, prod: &mut Producer<i16>) {
        if self.passthrough() {
            push_frame(prod, &[l, r]);
            return;
        }
        self.h = [self.h[1], self.h[2], self.h[3], [l as f32, r as f32]];
        let out = self.out_rate as u64;
        while self.acc < out {
            let t = self.acc as f32 / out as f32;
            let [l, r] = [0, 1].map(|c| {
                let v = catmull_rom(self.h[0][c], self.h[1][c], self.h[2][c], self.h[3][c], t);
                v.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16
            });
            push_frame(prod, &[l, r]);
            self.acc += self.in_rate as u64;
        }
        self.acc -= out;
    }
}

/// Push a whole frame, or none of it if the ring lacks room, and publish it
/// with one commit so the consumer never sees part of a frame. Returns
/// whether it fit.
pub fn push_frame(prod: &mut Producer<i16>, frame: &[i16]) -> bool {
    match prod.write_chunk_uninit(frame.len()) {
        Ok(chunk) => {
            chunk.fill_from_iter(frame.iter().copied());
            true
        }
        Err(_) => false,
    }
}

/// Fill `out` from the ring a whole frame of `channels` samples at a time,
/// each sample through `conv`. A frame the ring cannot supply whole is
/// silence, and what it holds stays for the next call: an underrun in the
/// middle of a frame would otherwise play the rest of the frame as zeros
/// and start the next one a channel late for good. Returns the samples
/// short.
pub fn pop_frames<T>(c: &mut Consumer<i16>, channels: usize, out: &mut [T], mut conv: impl FnMut(i16) -> T) -> usize {
    let mut missed = 0;
    let mut frames = out.chunks_exact_mut(channels);
    for frame in frames.by_ref() {
        match c.read_chunk(channels) {
            Ok(chunk) => {
                let (a, b) = chunk.as_slices();
                for (o, &v) in frame.iter_mut().zip(a.iter().chain(b)) {
                    *o = conv(v);
                }
                chunk.commit_all();
            }
            Err(_) => {
                frame.iter_mut().for_each(|o| *o = conv(0));
                missed += channels;
            }
        }
    }
    // A buffer that is not a whole number of frames: the tail is silence,
    // not what the host left in it.
    for o in frames.into_remainder() {
        *o = conv(0);
    }
    missed
}

/// The Catmull-Rom spline through p1 and p2 at t in [0, 1).
fn catmull_rom(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * (2.0 * p1
        + (p2 - p0) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (3.0 * p1 - p0 - 3.0 * p2 + p3) * t3)
}


/// The host `[audio] host` asks for. cpal's `pulseaudio` feature makes its
/// own `default_host()` prefer PulseAudio whenever a server runs, so the
/// default is chosen here instead: the platform's usual host (ALSA on Linux).
fn output_host(want: AudioHost) -> cpal::Host {
    #[cfg(target_os = "linux")]
    {
        if want == AudioHost::Pulseaudio {
            match cpal::host_from_id(cpal::HostId::PulseAudio) {
                Ok(h) => return h,
                Err(e) => eprintln!("audio: [audio] host = \"pulseaudio\", but no PulseAudio server answers ({e}); using ALSA"),
            }
        }
        if let Ok(h) = cpal::host_from_id(cpal::HostId::Alsa) {
            return h;
        }
    }
    #[cfg(not(target_os = "linux"))]
    if want != AudioHost::Default {
        eprintln!("audio: [audio] host = {want:?} is for Linux; using this platform's default");
    }
    cpal::default_host()
}

/// How much the host buffers: `[audio] cpal_buffer_frames` if set, else the
/// host's default -- except PulseAudio's, a target of about two seconds
/// (pipewire-pulse's too), which made IRIX's sound that late: there a fixed
/// period of PULSE_FRAMES asks for 2 x 512 frames, ~21 ms at 48 kHz.
fn output_buffer(audio: &AudioConfig, host: cpal::HostId) -> cpal::BufferSize {
    match (audio.cpal_buffer_frames, host) {
        (Some(frames), _) => cpal::BufferSize::Fixed(frames),
        #[cfg(target_os = "linux")]
        (None, cpal::HostId::PulseAudio) => cpal::BufferSize::Fixed(PULSE_FRAMES),
        _ => cpal::BufferSize::Default,
    }
}

/// Count the samples a player had to make up with silence, once real
/// samples are flowing.
#[inline]
fn count_underrun(missed: usize, playing: &AtomicBool, underruns: &AtomicU64) {
    if missed > 0 && playing.load(Ordering::Relaxed) {
        underruns.fetch_add(missed as u64, Ordering::Relaxed);
        crate::pf_instant!(crate::perfetto::events::HAL2_UNDERRUN, underruns.load(Ordering::Relaxed));
    }
}

/// Open a persistent stereo i16 cpal output stream, trying PREFERRED_RATES in
/// order, with a ring of `ring_samples(rate)` samples. `who` prefixes the
/// messages; the variable `capture_var`, if set to a file, also records what
/// the host plays (raw interleaved 16-bit little-endian stereo at the stream
/// rate). The stream plays silence when the ring buffer is empty.
pub fn open_output(
    audio: &AudioConfig,
    underruns: Arc<AtomicU64>,
    playing: Arc<AtomicBool>,
    ring_samples: impl Fn(u32) -> usize,
    who: &str,
    capture_var: &str,
) -> Option<HostOut> {
    if let Some(path) = &audio.file {
        return open_file_output(path, underruns, playing, ring_samples, who);
    }
    let host = output_host(audio.host);
    let device = host.default_output_device()?;
    // A fixed size the device refuses falls back to the host's default.
    let mut sizes = vec![output_buffer(audio, host.id())];
    if sizes[0] != cpal::BufferSize::Default {
        sizes.push(cpal::BufferSize::Default);
    }

    for buffer_size in sizes {
        for &rate in PREFERRED_RATES {
            let config = cpal::StreamConfig {
                channels: 2,
                sample_rate: rate,
                buffer_size,
            };
            let ring_size = ring_samples(rate);
            let who_cb = who.to_string();
            let err_fn = move |err: cpal::Error| { eprintln!("{who_cb}: cpal stream error: {:?}", err); };

            // Try f32 first (macOS CoreAudio native), then i16 (Linux ALSA).
            let (producer, stream) = {
                let (p, mut c) = RingBuffer::<i16>::new(ring_size);
                let underruns_cb = underruns.clone();
                let playing_cb = playing.clone();
                // `capture_var`=<file>: also write what the host plays -- the
                // end of the whole chain, to check or listen to offline.
                let mut capture = std::env::var_os(capture_var)
                    .and_then(|p| std::fs::File::create(p).ok())
                    .map(std::io::BufWriter::new);
                let data_fn = move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    let missed = pop_frames(&mut c, 2, data, |v| {
                        if let Some(w) = capture.as_mut() {
                            use std::io::Write;
                            let _ = w.write_all(&v.to_le_bytes());
                        }
                        v as f32 / 32768.0
                    });
                    count_underrun(missed, &playing_cb, &underruns_cb);
                };
                match device.build_output_stream(config, data_fn, err_fn.clone(), None) {
                    Ok(s) => (p, s),
                    Err(_) => {
                        // f32 failed, try i16
                        let (p, mut c) = RingBuffer::<i16>::new(ring_size);
                        let underruns_cb = underruns.clone();
                        let playing_cb = playing.clone();
                        let data_fn = move |data: &mut [i16], _: &cpal::OutputCallbackInfo| {
                            let missed = pop_frames(&mut c, 2, data, |v| v);
                            count_underrun(missed, &playing_cb, &underruns_cb);
                        };
                        match device.build_output_stream(config, data_fn, err_fn.clone(), None) {
                            Ok(s) => (p, s),
                            Err(e) => {
                                eprintln!("{who}: cpal build_output_stream failed at {}Hz: {:?}", rate, e);
                                continue;
                            }
                        }
                    }
                }
            };
            if stream.play().is_err() { continue; }
            // cpal 0.18 dropped DeviceTrait::name(); a Device's Display impl is its name.
            println!("{who}: audio output: {} via {:?} at {}Hz", device, host.id(), rate);
            return Some(HostOut {
                stream_rate: rate,
                producer,
                underruns,
                playing,
                _stream: Some(stream),
                _file: None,
            });
        }

        if buffer_size != cpal::BufferSize::Default {
            eprintln!("{who}: no output stream with a {:?} buffer; trying the host's default", buffer_size);
        }
    }

    eprintln!("{who}: failed to open audio output at any rate (tried {:?})", PREFERRED_RATES);
    None
}

/// A 16-bit PCM WAV file whose sizes are rewritten after every append, so
/// a file from a killed run still plays.
pub struct WavWriter {
    file: std::fs::File,
    rate: u32,
    channels: u16,
    data_bytes: u32,
}

impl WavWriter {
    pub fn create(path: &Path, rate: u32, channels: u16) -> std::io::Result<Self> {
        let file = std::fs::File::create(path)?;
        let mut w = Self { file, rate, channels, data_bytes: 0 };
        w.header()?;
        Ok(w)
    }

    fn header(&mut self) -> std::io::Result<()> {
        use std::io::{Seek, SeekFrom, Write};
        let align = self.channels * 2;
        let mut h = Vec::with_capacity(44);
        h.extend_from_slice(b"RIFF");
        h.extend_from_slice(&(36 + self.data_bytes).to_le_bytes());
        h.extend_from_slice(b"WAVEfmt ");
        h.extend_from_slice(&16u32.to_le_bytes());
        h.extend_from_slice(&1u16.to_le_bytes());
        h.extend_from_slice(&self.channels.to_le_bytes());
        h.extend_from_slice(&self.rate.to_le_bytes());
        h.extend_from_slice(&(self.rate * u32::from(align)).to_le_bytes());
        h.extend_from_slice(&align.to_le_bytes());
        h.extend_from_slice(&16u16.to_le_bytes());
        h.extend_from_slice(b"data");
        h.extend_from_slice(&self.data_bytes.to_le_bytes());
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&h)?;
        self.file.seek(SeekFrom::End(0))?;
        Ok(())
    }

    /// Append interleaved samples, little-endian. A WAV's sizes are 32-bit:
    /// past about 4 GiB (six hours of 48 kHz stereo) it is full, and this
    /// fails rather than write sizes that wrap.
    pub fn append(&mut self, pcm: &[i16]) -> std::io::Result<()> {
        use std::io::Write;
        let bytes: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
        let total = u32::try_from(bytes.len()).ok().and_then(|n| self.data_bytes.checked_add(n)).filter(|&t| t <= u32::MAX - 36);
        let Some(total) = total else {
            return Err(std::io::Error::other("the WAV file is full (4 GiB)"));
        };
        self.file.write_all(&bytes)?;
        self.data_bytes = total;
        self.header()
    }
}

/// The rate of a file output, the first a sound card is asked for.
pub const FILE_RATE: u32 = PREFERRED_RATES[0];
/// How often a file output takes what is due from the ring.
const FILE_TICK: Duration = Duration::from_millis(5);

/// Plays a ring into a WAV file on its own thread, at [`FILE_RATE`] by the
/// host's clock, as a sound card would: a frame the ring cannot supply is
/// silence and counts as an underrun. Stops and joins when dropped.
struct FilePlayer {
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Drop for FilePlayer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// `[audio] file`: a stereo output at [`FILE_RATE`] that a thread plays
/// into the WAV at `path` in real time. Lets a machine with no sound card,
/// or a test, hear exactly what the host stream would have played.
fn open_file_output(
    path: &Path,
    underruns: Arc<AtomicU64>,
    playing: Arc<AtomicBool>,
    ring_samples: impl Fn(u32) -> usize,
    who: &str,
) -> Option<HostOut> {
    const CHANNELS: usize = 2;
    let mut wav = match WavWriter::create(path, FILE_RATE, CHANNELS as u16) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("{who}: cannot write {}: {e}", path.display());
            return None;
        }
    };
    let (producer, mut c) = RingBuffer::<i16>::new(ring_samples(FILE_RATE));
    let stop = Arc::new(AtomicBool::new(false));
    let thread = {
        let (stop, underruns, playing) = (stop.clone(), underruns.clone(), playing.clone());
        let who = who.to_string();
        std::thread::Builder::new().name(format!("{who} file")).spawn(move || {
            let start = Instant::now();
            let mut played = 0u64;
            let mut buf = Vec::new();
            while !stop.load(Ordering::Relaxed) {
                std::thread::sleep(FILE_TICK);
                let due = (start.elapsed().as_nanos() * u128::from(FILE_RATE) / 1_000_000_000) as u64;
                buf.clear();
                buf.resize((due - played) as usize * CHANNELS, 0);
                count_underrun(pop_frames(&mut c, CHANNELS, &mut buf, |v| v), &playing, &underruns);
                if let Err(e) = wav.append(&buf) {
                    eprintln!("{who}: writing the audio file failed: {e}");
                    return;
                }
                played = due;
            }
        })
    };
    let thread = match thread {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{who}: cannot start the audio file thread: {e}");
            return None;
        }
    };
    println!("{who}: audio output: file {} at {FILE_RATE}Hz, {CHANNELS} channels", path.display());
    Some(HostOut {
        stream_rate: FILE_RATE,
        producer,
        underruns,
        playing,
        _stream: None,
        _file: Some(FilePlayer { stop, thread: Some(thread) }),
    })
}

/// Known signals and what a test can measure of a played one: the same for
/// every device that plays to the host.
#[cfg(test)]
pub(crate) mod test_signal {
    use std::path::{Path, PathBuf};
    use crate::config::AudioConfig;

    /// A test's WAV file, removed when the test ends, passed or not.
    pub(crate) struct TempWav(pub(crate) PathBuf);

    impl Drop for TempWav {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    /// `[audio]` settings that play into a WAV file of the test's own, and
    /// the file.
    pub(crate) fn file_config(name: &str) -> (AudioConfig, TempWav) {
        let path = std::env::temp_dir().join(format!("iris-{name}-{}.wav", std::process::id()));
        (AudioConfig { file: Some(path.clone()), ..AudioConfig::default() }, TempWav(path))
    }


    /// `frames` of a stereo sine pair at `rate`, `left` Hz on the left and
    /// `right` Hz on the right, at `amp`.
    pub(crate) fn tones(frames: usize, rate: u32, left: f64, right: f64, amp: f64) -> Vec<i16> {
        let at = |hz: f64, n: usize| (amp * (std::f64::consts::TAU * hz * n as f64 / f64::from(rate)).sin()).round() as i16;
        (0..frames).flat_map(|n| [at(left, n), at(right, n)]).collect()
    }

    /// A WAV's rate, channels and samples. Only the 16-bit PCM this crate
    /// writes.
    pub(crate) fn read_wav(path: &Path) -> (u32, usize, Vec<i16>) {
        let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert!(b.len() >= 44 && &b[..4] == b"RIFF" && &b[8..16] == b"WAVEfmt ", "{}: not a WAV", path.display());
        let u16_at = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]);
        let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        assert_eq!(u16_at(20), 1, "PCM");
        assert_eq!(u16_at(34), 16, "16-bit");
        let data = u32_at(40) as usize;
        assert_eq!(b.len(), 44 + data, "the data size covers the file");
        let pcm = b[44..].as_chunks::<2>().0.iter().map(|c| i16::from_le_bytes(*c)).collect();
        (u32_at(24), usize::from(u16_at(22)), pcm)
    }

    /// One channel of interleaved frames.
    pub(crate) fn channel(pcm: &[i16], channels: usize, c: usize) -> Vec<i16> {
        pcm.chunks_exact(channels).map(|f| f[c]).collect()
    }

    /// The frames from the first one with a sample other than zero.
    pub(crate) fn from_first_sound(pcm: &[i16], channels: usize) -> &[i16] {
        let first = pcm.chunks_exact(channels).position(|f| f.iter().any(|&v| v != 0)).unwrap_or(pcm.len() / channels);
        &pcm[first * channels..]
    }

    /// The frequency of a tone from its rising zero crossings: the cycles
    /// between the first and the last, over the time between them.
    pub(crate) fn frequency(s: &[i16], rate: u32) -> f64 {
        let ups: Vec<usize> = s.windows(2).enumerate().filter(|(_, w)| w[0] < 0 && w[1] >= 0).map(|(i, _)| i).collect();
        match (ups.first(), ups.last()) {
            (Some(&a), Some(&b)) if b > a => (ups.len() - 1) as f64 * f64::from(rate) / (b - a) as f64,
            _ => 0.0,
        }
    }

    pub(crate) fn peak(s: &[i16]) -> i32 {
        s.iter().map(|&v| i32::from(v).abs()).max().unwrap_or(0)
    }

    /// Assert the left channel is a `left` Hz tone and the right a `right`
    /// Hz one, each within 1%, and both louder than `min_peak`.
    pub(crate) fn assert_tones(pcm: &[i16], channels: usize, rate: u32, left: f64, right: f64, min_peak: i32) {
        for (c, want) in [(0, left), (1, right)] {
            let s = channel(pcm, channels, c);
            let (hz, p) = (frequency(&s, rate), peak(&s));
            assert!((hz - want).abs() <= want / 100.0, "channel {c}: {hz:.1} Hz, wanted {want} Hz");
            assert!(p >= min_peak, "channel {c}: peak {p}, wanted at least {min_peak}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::test_signal::*;

    fn drain(c: &mut rtrb::Consumer<i16>) -> Vec<i16> {
        std::iter::from_fn(|| c.pop().ok()).collect()
    }

    #[test]
    fn a_full_ring_drops_whole_pairs_and_keeps_the_channels() {
        for (in_rate, out_rate) in [(44100, 44100), (22050, 44100)] {
            // Odd capacity: the ring fills with one slot left over.
            let (mut p, mut c) = RingBuffer::<i16>::new(5);
            let mut rs = Resampler::new(in_rate, out_rate);
            for _ in 0..2 {
                for _ in 0..8 {
                    rs.push(1, -1, &mut p);
                }
                let got = drain(&mut c);
                assert!(got.len() >= 2 && got.len() % 2 == 0, "{in_rate}: whole pairs only, got {got:?}");
                assert!(got.chunks(2).all(|q| q == [0, 0] || q[0] > 0 && q[1] < 0), "{in_rate}: channels swapped in {got:?}");
            }
        }
    }

    #[test]
    fn an_underrun_in_the_middle_of_a_frame_keeps_the_channels_aligned() {
        let (mut p, mut c) = RingBuffer::<i16>::new(64);
        // One whole four-channel frame and half of the next, as a producer
        // that pushes sample by sample leaves the ring between two pushes.
        for v in [1, 2, 3, 4, 5, 6] {
            p.push(v).unwrap();
        }
        let mut out = [9i16; 12];
        let missed = pop_frames(&mut c, 4, &mut out, |v| v);
        assert_eq!(out, [1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(missed, 8);
        // The rest of the frame arrives: it plays whole, first.
        p.push(7).unwrap();
        p.push(8).unwrap();
        let mut out = [9i16; 8];
        assert_eq!(pop_frames(&mut c, 4, &mut out, |v| v), 4);
        assert_eq!(out, [5, 6, 7, 8, 0, 0, 0, 0]);
    }

    #[test]
    fn a_frame_goes_into_the_ring_whole_or_not_at_all() {
        let (mut p, mut c) = RingBuffer::<i16>::new(6);
        assert!(push_frame(&mut p, &[1, 2, 3, 4]));
        // Two slots left, a four-sample frame does not fit: none of it is
        // written.
        assert!(!push_frame(&mut p, &[5, 6, 7, 8]));
        assert_eq!(c.slots(), 4);
        let mut out = [0i16; 4];
        assert_eq!(pop_frames(&mut c, 4, &mut out, |v| v), 0);
        assert_eq!(out, [1, 2, 3, 4]);
    }

    /// `[audio] file` plays the ring in real time: what was pushed comes out
    /// in the WAV in order, sample for sample and channel for channel, the
    /// file grows with the host's clock whether or not anything is pushed,
    /// and the silence after the end counts as underruns.
    #[test]
    fn a_file_output_plays_the_ring_in_real_time() {
        let (cfg, path) = file_config("file-output");
        let underruns = Arc::new(AtomicU64::new(0));
        let start = Instant::now();
        let mut out = open_output(&cfg, underruns.clone(), Arc::new(AtomicBool::new(true)), |r| r as usize * 2, "test", "IRIS_TEST_CAPTURE")
            .expect("a file output opens without a sound card");
        assert_eq!(out.stream_rate, FILE_RATE, "the file rate");
        // A quarter second of tone, as a device would push it.
        let sent = tones(FILE_RATE as usize / 4, FILE_RATE, 440.0, 880.0, 12000.0);
        for f in sent.chunks_exact(2) {
            assert!(push_frame(&mut out.producer, f));
        }
        std::thread::sleep(Duration::from_millis(500));
        drop(out);
        let elapsed = start.elapsed().as_secs_f64();
        let (rate, channels, pcm) = read_wav(&path.0);
        assert_eq!((rate, channels), (FILE_RATE, 2));
        let frames = (pcm.len() / 2) as f64;
        // The player starts after the open and stops a tick short of the drop.
        assert!(frames <= elapsed * f64::from(FILE_RATE) && frames >= 0.4 * f64::from(FILE_RATE), "{frames} frames in {elapsed:.3} s");
        // The tone's first frame is zero on both sides.
        let (played, sent) = (from_first_sound(&pcm, 2), from_first_sound(&sent, 2));
        assert_eq!(&played[..sent.len()], sent, "the pushed frames, unchanged and in order");
        assert_tones(&played[..sent.len()], 2, FILE_RATE, 440.0, 880.0, 11000);
        assert!(played[sent.len()..].iter().all(|&v| v == 0), "silence once the ring is empty");
        assert!(underruns.load(Ordering::Relaxed) > 0, "the silence counted as underruns");
    }

    /// A file that cannot be written opens no output, as a sound card that
    /// refuses does.
    #[test]
    fn an_unwritable_file_opens_no_output() {
        let cfg = AudioConfig { file: Some("/nonexistent-dir/x.wav".into()), ..AudioConfig::default() };
        assert!(open_output(&cfg, Arc::new(AtomicU64::new(0)), Arc::new(AtomicBool::new(true)), |r| r as usize, "test", "IRIS_TEST_CAPTURE").is_none());
    }

    #[test]
    fn the_tone_measures_find_frequency_and_channel() {
        let pcm = tones(48000, 48000, 440.0, 880.0, 10000.0);
        assert_tones(&pcm, 2, 48000, 440.0, 880.0, 9900);
        // Swapped channels are caught.
        let swapped: Vec<i16> = pcm.chunks_exact(2).flat_map(|f| [f[1], f[0]]).collect();
        assert!(std::panic::catch_unwind(|| assert_tones(&swapped, 2, 48000, 440.0, 880.0, 9900)).is_err());
    }
}
