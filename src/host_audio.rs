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

    /// The variable naming the loopback a real-device test records from:
    /// `<output host>:<capture source>`. The host is the `[audio] host` the
    /// card plays through (`alsa`, `pulseaudio` or `default`); the source is
    /// the PulseAudio source that hears the default output, such as a null
    /// sink's monitor: `alsa:iris_sink.monitor`.
    pub(crate) const LOOPBACK_VAR: &str = "IRIS_AUDIO_LOOPBACK";

    /// Where a test hears what a card sent to the host.
    pub(crate) enum Backend {
        /// An `[audio] file` of the test's own. Needs no sound card.
        File,
        /// The host's real default output, recorded back through the
        /// loopback [`LOOPBACK_VAR`] names: cpal, the operating system's
        /// sound server and its mixer, none of which the file goes through.
        RealDevice,
    }

    /// What a test heard: interleaved samples at the capture's own rate.
    pub(crate) struct Heard {
        pub(crate) rate: u32,
        pub(crate) channels: usize,
        pub(crate) pcm: Vec<i16>,
    }

    /// The `[audio]` settings a card under test plays with, and what records
    /// it. A real-device listener holds a lock while it lives: two cards on
    /// one default output would mix.
    pub(crate) struct Listener {
        pub(crate) cfg: AudioConfig,
        ear: Ear,
    }

    enum Ear {
        File(TempWav),
        #[cfg(any(target_os = "linux", windows, target_os = "macos"))]
        Loopback(loopback::Recorder, std::sync::MutexGuard<'static, ()>),
    }

    impl Listener {
        /// A listener on `backend`, or `None` when a real device is asked
        /// for and [`LOOPBACK_VAR`] is unset: the test prints why it skipped
        /// and passes, so a plain `cargo test`, and a machine whose speakers
        /// nothing records, stay green. With the variable set, a loopback
        /// that cannot be opened fails the test.
        pub(crate) fn new(backend: Backend, name: &str) -> Option<Self> {
            match backend {
                Backend::File => {
                    let (cfg, path) = file_config(name);
                    Some(Self { cfg, ear: Ear::File(path) })
                }
                Backend::RealDevice => {
                    let Some(spec) = std::env::var_os(LOOPBACK_VAR) else {
                        eprintln!("SKIPPED: {LOOPBACK_VAR} not set (no loopback audio device on this host)");
                        return None;
                    };
                    let spec = spec.into_string().unwrap_or_else(|s| panic!("{LOOPBACK_VAR}={s:?}: not UTF-8"));
                    Some(Self::loopback(&spec))
                }
            }
        }

        #[cfg(any(target_os = "linux", windows, target_os = "macos"))]
        fn loopback(spec: &str) -> Self {
            use crate::config::AudioHost;
            static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());
            let serial = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
            let (host, source) = spec
                .split_once(':')
                .unwrap_or_else(|| panic!("{LOOPBACK_VAR}={spec}: want <output host>:<capture source>"));
            let host = match host {
                "default" => AudioHost::Default,
                "alsa" => AudioHost::Alsa,
                "pulseaudio" => AudioHost::Pulseaudio,
                _ => panic!("{LOOPBACK_VAR}={spec}: the output host is default, alsa or pulseaudio"),
            };
            let rec = loopback::Recorder::open(source).unwrap_or_else(|e| panic!("{LOOPBACK_VAR}={spec}: {e}"));
            Self { cfg: AudioConfig { host, ..AudioConfig::default() }, ear: Ear::Loopback(rec, serial) }
        }

        #[cfg(not(any(target_os = "linux", windows, target_os = "macos")))]
        fn loopback(spec: &str) -> Self {
            panic!("{LOOPBACK_VAR}={spec}: recording a loopback is not built for this platform");
        }

        /// Stop listening and give what was heard. A file is removed.
        pub(crate) fn heard(self) -> Heard {
            match self.ear {
                Ear::File(path) => {
                    let (rate, channels, pcm) = read_wav(&path.0);
                    Heard { rate, channels, pcm }
                }
                #[cfg(any(target_os = "linux", windows, target_os = "macos"))]
                Ear::Loopback(rec, _serial) => rec.finish(),
            }
        }
    }

    /// Records a PulseAudio source with `parec` (pulseaudio-utils, which the
    /// loopback's setup needs anyway for `pactl`). cpal's own PulseAudio
    /// client was tried first: its record stream did not start until a
    /// default fragment of about two seconds had filled, and with a short
    /// fixed buffer it never started at all.
    #[cfg(target_os = "linux")]
    mod loopback {
        use std::io::Read;
        use std::process::{Child, Command, Stdio};
        use std::sync::{Arc, Mutex};
        use std::time::{Duration, Instant};
        use super::Heard;

        const RATE: u32 = 48000;
        /// How long a recording runs on after the card has stopped: what the
        /// server still held for the sink.
        const TAIL: Duration = Duration::from_millis(300);

        pub(super) struct Recorder {
            child: Child,
            reader: Option<std::thread::JoinHandle<()>>,
            bytes: Arc<Mutex<Vec<u8>>>,
        }

        impl Recorder {
            pub(super) fn open(source: &str) -> Result<Self, String> {
                let mut child = Command::new("parec")
                    .args(["-d", source, "--raw", "--format=s16le", "--channels=2", "--latency-msec=20"])
                    .arg(format!("--rate={RATE}"))
                    .stdout(Stdio::piped())
                    .spawn()
                    .map_err(|e| format!("cannot run parec: {e}"))?;
                let mut out = child.stdout.take().unwrap();
                let bytes = Arc::new(Mutex::new(Vec::new()));
                let reader = {
                    let bytes = bytes.clone();
                    std::thread::spawn(move || {
                        let mut buf = [0u8; 4096];
                        while let Ok(n @ 1..) = out.read(&mut buf) {
                            bytes.lock().unwrap().extend_from_slice(&buf[..n]);
                        }
                    })
                };
                // Recording has started once the monitor gives anything.
                let start = Instant::now();
                while bytes.lock().unwrap().is_empty() {
                    if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
                        return Err(format!("parec -d {source} ended: {status}"));
                    }
                    if start.elapsed() > Duration::from_secs(5) {
                        let _ = child.kill();
                        return Err(format!("parec -d {source} recorded nothing in 5 s"));
                    }
                    std::thread::sleep(Duration::from_millis(5));
                }
                Ok(Self { child, reader: Some(reader), bytes })
            }

            /// Stop once the monitor has been quiet for 100 ms after the
            /// tail, or after two seconds: on a loaded host the sink's
            /// latency can outlast a fixed tail and cut the tone short.
            pub(super) fn finish(mut self) -> Heard {
                std::thread::sleep(TAIL);
                let quiet = (RATE / 10) as usize * 4;
                let end = Instant::now() + Duration::from_secs(2);
                while Instant::now() < end {
                    let b = self.bytes.lock().unwrap();
                    if b.len() >= quiet && b[b.len() - quiet..].iter().all(|&x| x == 0) {
                        break;
                    }
                    drop(b);
                    std::thread::sleep(Duration::from_millis(20));
                }
                let _ = self.child.kill();
                let _ = self.child.wait();
                if let Some(r) = self.reader.take() {
                    let _ = r.join();
                }
                let bytes = std::mem::take(&mut *self.bytes.lock().unwrap());
                let pcm = bytes.as_chunks::<2>().0.iter().map(|c| i16::from_le_bytes(*c)).collect();
                Heard { rate: RATE, channels: 2, pcm }
            }
        }

        /// A test that fails before `finish` must not leave parec recording.
        impl Drop for Recorder {
            fn drop(&mut self) {
                let _ = self.child.kill();
                let _ = self.child.wait();
            }
        }
    }

    /// Records the host's own sound stack through cpal, where there is no
    /// `parec`. The source is a device's name, as cpal shows it: on macOS
    /// the BlackHole input that the default output feeds ("BlackHole 2ch"),
    /// on Windows `default-render` for the default output's WASAPI loopback
    /// (cpal turns an input stream on a render device into one), or a render
    /// device's name.
    #[cfg(any(windows, target_os = "macos"))]
    mod loopback {
        use std::sync::{Arc, Mutex};
        use std::time::{Duration, Instant};
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
        use super::Heard;

        /// How long a recording runs on after the card has stopped: what
        /// the mixer still held.
        const TAIL: Duration = Duration::from_millis(300);

        #[derive(Default)]
        struct Capture {
            pcm: Vec<i16>,
            /// When the first buffer arrived, and when the latest did.
            first: Option<Instant>,
            last: Option<Instant>,
        }

        pub(super) struct Recorder {
            stream: Option<cpal::Stream>,
            heard: Arc<Mutex<Capture>>,
            rate: u32,
            channels: usize,
        }

        /// Adds a buffer. A Windows loopback delivers nothing while nothing
        /// renders, so a pause in the card's output would leave no gap in
        /// what was recorded, and the tests look for that gap. The wall
        /// clock fills what a late buffer left out with silence.
        fn push<T: Copy>(heard: &Mutex<Capture>, rate: u32, channels: usize, data: &[T], conv: impl Fn(T) -> i16) {
            let mut c = heard.lock().unwrap();
            let now = Instant::now();
            // The first buffer holds audio that ended now, so the clock
            // starts one buffer before it arrived.
            let first = *c.first.get_or_insert(now - Duration::from_secs_f64((data.len() / channels) as f64 / f64::from(rate)));
            let due = (now - first).as_secs_f64() * f64::from(rate);
            let have = (c.pcm.len() / channels + data.len() / channels) as f64;
            if due - have > f64::from(rate) / 20.0 {
                let pad = (due - have) as usize * channels;
                let len = c.pcm.len();
                c.pcm.resize(len + pad, 0);
            }
            c.pcm.extend(data.iter().map(|&v| conv(v)));
            c.last = Some(now);
        }

        fn device(host: &cpal::Host, source: &str) -> Result<cpal::Device, String> {
            #[cfg(windows)]
            if source == "default-render" {
                return host.default_output_device().ok_or_else(|| "no default output device to loop back".to_string());
            }
            let devices = host.devices().map_err(|e| format!("cannot list audio devices: {e}"))?;
            let mut names = Vec::new();
            for d in devices {
                // An endpoint that cannot say its name (an unplugged one)
                // must not stop the search for the one wanted.
                let Ok(desc) = d.description() else { continue };
                if desc.name() == source {
                    return Ok(d);
                }
                names.push(desc.name().to_owned());
            }
            Err(format!("no audio device named {source:?}; there are {names:?}"))
        }

        impl Recorder {
            pub(super) fn open(source: &str) -> Result<Self, String> {
                let host = cpal::default_host();
                let dev = device(&host, source)?;
                // WASAPI hands out a loopback's mix format only as the
                // output side's. CoreAudio's device rate is the device's
                // alone, so ask for the one the card plays at first.
                #[cfg(windows)]
                let supported = dev.default_output_config();
                #[cfg(target_os = "macos")]
                let supported = dev.default_input_config();
                let supported = supported.map_err(|e| format!("{source}: no default config: {e}"))?;
                let format = supported.sample_format();
                #[cfg(windows)]
                let config = supported.config();
                #[cfg(target_os = "macos")]
                let config = cpal::StreamConfig { channels: 2, sample_rate: 48000, ..supported.config() };
                let (rate, channels) = (config.sample_rate, usize::from(config.channels));
                let heard = Arc::new(Mutex::new(Capture::default()));
                let err_fn = |e: cpal::Error| eprintln!("loopback capture: cpal stream error: {e:?}");
                let h = heard.clone();
                let stream = match format {
                    cpal::SampleFormat::F32 => dev.build_input_stream(
                        config,
                        move |d: &[f32], _: &cpal::InputCallbackInfo| {
                            push(&h, rate, channels, d, |v| (v * 32767.0).clamp(-32768.0, 32767.0) as i16)
                        },
                        err_fn,
                        None,
                    ),
                    cpal::SampleFormat::I16 => dev.build_input_stream(
                        config,
                        move |d: &[i16], _: &cpal::InputCallbackInfo| push(&h, rate, channels, d, |v| v),
                        err_fn,
                        None,
                    ),
                    cpal::SampleFormat::I32 => dev.build_input_stream(
                        config,
                        move |d: &[i32], _: &cpal::InputCallbackInfo| push(&h, rate, channels, d, |v| (v >> 16) as i16),
                        err_fn,
                        None,
                    ),
                    other => return Err(format!("{source}: records {other:?}, which the test does not read")),
                }
                .map_err(|e| format!("{source}: cannot record at {rate} Hz, {channels} channels: {e}"))?;
                stream.play().map_err(|e| format!("{source}: cannot start recording: {e}"))?;
                println!("loopback capture: recording {source} at {rate} Hz, {channels} channels, {format:?}");
                // A CoreAudio input delivers silence from the start, so
                // recording has started once a buffer has come. A Windows
                // loopback delivers only once something plays.
                #[cfg(target_os = "macos")]
                {
                    let start = Instant::now();
                    while heard.lock().unwrap().first.is_none() {
                        if start.elapsed() > Duration::from_secs(5) {
                            return Err(format!("{source} recorded nothing in 5 s"));
                        }
                        std::thread::sleep(Duration::from_millis(5));
                    }
                }
                Ok(Self { stream: Some(stream), heard, rate, channels })
            }

            /// Stop once nothing new has come, or only silence, for 100 ms
            /// after the tail, or after two seconds.
            pub(super) fn finish(mut self) -> Heard {
                std::thread::sleep(TAIL);
                let quiet = (self.rate / 10) as usize * self.channels;
                let end = Instant::now() + Duration::from_secs(2);
                while Instant::now() < end {
                    let c = self.heard.lock().unwrap();
                    let idle = c.last.is_none_or(|t| t.elapsed() > Duration::from_millis(100));
                    if idle || (c.pcm.len() >= quiet && c.pcm[c.pcm.len() - quiet..].iter().all(|&x| x == 0)) {
                        break;
                    }
                    drop(c);
                    std::thread::sleep(Duration::from_millis(20));
                }
                drop(self.stream.take());
                let pcm = std::mem::take(&mut self.heard.lock().unwrap().pcm);
                Heard { rate: self.rate, channels: self.channels, pcm }
            }
        }
    }

    /// The stretches of sound in `pcm`, as ranges of frames: runs of 5 ms
    /// blocks whose peak passes `floor`.
    pub(crate) fn loud_runs(pcm: &[i16], channels: usize, rate: u32, floor: i32) -> Vec<std::ops::Range<usize>> {
        let block = (rate / 200) as usize;
        let frames = pcm.len() / channels;
        let mut runs: Vec<std::ops::Range<usize>> = Vec::new();
        for (i, b) in pcm.chunks(block * channels).enumerate() {
            if peak(b) > floor {
                let r = i * block..((i + 1) * block).min(frames);
                match runs.last_mut() {
                    Some(last) if last.end == r.start => last.end = r.end,
                    _ => runs.push(r),
                }
            }
        }
        runs
    }

    /// The frequency of a tone from the samples between its rising zero
    /// crossings: their mean, leaving out the few that a lost or repeated
    /// stretch, or the edge of a gap, makes much shorter or longer than the
    /// median. Those are for the duration check to catch; this measures
    /// the pitch.
    fn tone_frequency(periods: &mut [usize], rate: u32) -> f64 {
        if periods.is_empty() {
            return 0.0;
        }
        periods.sort_unstable();
        let median = periods[periods.len() / 2] as f64;
        let near: Vec<f64> = periods.iter().map(|&n| n as f64).filter(|&n| n >= 0.8 * median && n <= 1.25 * median).collect();
        f64::from(rate) * near.len() as f64 / near.iter().sum::<f64>()
    }

    /// Assert that `heard` holds a tone, a silence where its producer
    /// paused, and the tone again. The pause is the longest silence; the
    /// shorter ones a busy host leaves are allowed. On each side the left
    /// channel must be at `left` Hz and the right at `right` Hz within 1%,
    /// so channels that swap after the pause fail; the peaks must reach
    /// `min_peak`; and the sound must last between half and 110% of what
    /// the card played, `played` seconds, at the capture's rate, so
    /// repeated frames or a stream that lost most of its frames fail.
    pub(crate) fn assert_a_tone_either_side_of_a_pause(heard: &Heard, left: f64, right: f64, min_peak: i32, played: [f64; 2]) {
        let (ch, rate) = (heard.channels, heard.rate);
        let runs = loud_runs(&heard.pcm, ch, rate, min_peak / 4);
        assert!(!runs.is_empty(), "heard nothing: no sound reached the capture");
        let (pause, gap) = runs.windows(2).enumerate().map(|(i, w)| (i + 1, w[1].start - w[0].end)).max_by_key(|&(_, g)| g).unwrap_or((runs.len(), 0));
        let gap = gap as f64 / f64::from(rate);
        assert!(gap >= 0.1, "no pause heard: the longest silence is {gap:.3} s");
        let (before, after) = runs.split_at(pause);
        for (name, runs, want) in [("before the pause", before, played[0]), ("after the pause", after, played[1])] {
            let secs = runs.iter().map(|r| r.len()).sum::<usize>() as f64 / f64::from(rate);
            // Not more than the card played, within 10%: repeated frames
            // or a stream played too slowly fail. At least half: a busy
            // host can delay the card's service pass until the burst it
            // catches up with overflows the host ring, and those frames
            // are rightly dropped. A wrong rate fails the frequency check.
            assert!(secs <= want * 1.1 && secs >= want / 2.0, "{name}: {secs:.3} s of sound at {rate} Hz, the card played {want:.3} s");
            for (c, hz) in [(0, left), (1, right)] {
                let (mut periods, mut p) = (Vec::new(), 0);
                for r in runs {
                    let s = channel(&heard.pcm[r.start * ch..r.end * ch], ch, c);
                    let ups: Vec<usize> = s.windows(2).enumerate().filter(|(_, w)| w[0] < 0 && w[1] >= 0).map(|(i, _)| i).collect();
                    periods.extend(ups.windows(2).map(|w| w[1] - w[0]));
                    p = p.max(peak(&s));
                }
                let got = tone_frequency(&mut periods, rate);
                assert!((got - hz).abs() <= hz / 100.0, "{name}: channel {c} at {got:.1} Hz, wanted {hz:.1} Hz");
                assert!(p >= min_peak, "{name}: channel {c} peak {p}, wanted at least {min_peak}");
            }
        }
    }

    /// The frames with any sample other than zero: what a device played,
    /// without the silence a host stream fills underruns with.
    pub(crate) fn sounding(pcm: &[i16], channels: usize) -> Vec<i16> {
        pcm.chunks_exact(channels).filter(|f| f.iter().any(|&v| v != 0)).flatten().copied().collect()
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

    /// Frames pushed into an output as a card pushes them, in real time: a
    /// 440 Hz tone on the left and 880 Hz on the right, nothing for 0.3 s so
    /// the stream runs dry, then the tone again. Both sides of the gap must
    /// be the tone on its own channels, as long as was pushed, and the gap
    /// must count as underruns. The host end alone, with no card.
    fn a_tone_survives_an_underrun(backend: Backend) {
        let Some(ear) = Listener::new(backend, "host-underrun") else { return };
        let underruns = Arc::new(AtomicU64::new(0));
        let mut out = open_output(&ear.cfg, underruns.clone(), Arc::new(AtomicBool::new(true)), |r| r as usize * 2 / 5, "test", "IRIS_TEST_CAPTURE")
            .expect("the output opens");
        let rate = out.stream_rate;
        // One second holds whole cycles of both at any rate.
        let tone = tones(rate as usize, rate, 440.0, 880.0, 16000.0);
        let mut at = 0usize;
        // Push what is due every 2 ms for `ms`; returns the frames pushed.
        let mut play = |out: &mut HostOut, ms: u64| {
            let (start, mut pushed) = (Instant::now(), 0u64);
            while start.elapsed() < Duration::from_millis(ms) {
                let due = (start.elapsed().as_nanos() * u128::from(rate) / 1_000_000_000) as u64;
                while pushed < due && push_frame(&mut out.producer, &tone[at * 2..at * 2 + 2]) {
                    at = (at + 1) % (rate as usize);
                    pushed += 1;
                }
                std::thread::sleep(Duration::from_millis(2));
            }
            pushed as f64 / f64::from(rate)
        };
        let first = play(&mut out, 600);
        let before = underruns.load(Ordering::Relaxed);
        std::thread::sleep(Duration::from_millis(300));
        let underran = underruns.load(Ordering::Relaxed) - before;
        let second = play(&mut out, 600);
        // What the ring still holds plays out.
        std::thread::sleep(Duration::from_millis(300));
        drop(out);
        let heard = ear.heard();
        assert_a_tone_either_side_of_a_pause(&heard, 440.0, 880.0, 8000, [first, second]);
        assert!(underran > 0, "the stream ran dry while nothing was pushed");
    }

    #[test]
    fn a_tone_survives_an_underrun_into_a_file() {
        a_tone_survives_an_underrun(Backend::File);
    }

    #[test]
    fn a_tone_survives_an_underrun_on_the_real_device() {
        a_tone_survives_an_underrun(Backend::RealDevice);
    }
}
