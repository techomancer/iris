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

/// What a card's output has done, counted the same way for every card from
/// the start of the machine, with or without a host stream open: frames each
/// output moved, frames the host stream played as silence while the card was
/// playing, and frames the host ring refused.
pub struct Meter {
    card: &'static str,
    outputs: &'static [&'static str],
    moved: Box<[AtomicU64]>,
    underruns: AtomicU64,
    dropped: AtomicU64,
    /// Set by the first frame the card writes to the host ring, cleared by
    /// [`Meter::stopped`]: an idle card sends nothing, so the host's silence
    /// is an underrun only while it is set.
    playing: AtomicBool,
}

impl Meter {
    /// A meter for `card` with `outputs`, in the order the card reports them.
    /// Every card names its outputs `analog in`, `analog out` (`analog out 1`
    /// and `2` for two), `S/PDIF out`, in that order, for those it has.
    pub fn new(card: &'static str, outputs: &'static [&'static str]) -> Arc<Self> {
        let moved = outputs.iter().map(|_| AtomicU64::new(0)).collect();
        Arc::new(Self { card, outputs, moved, underruns: AtomicU64::new(0), dropped: AtomicU64::new(0), playing: AtomicBool::new(false) })
    }

    /// `output` has moved `frames` more frames, to the host or not.
    pub fn moved(&self, output: usize, frames: u64) { self.moved[output].fetch_add(frames, Ordering::Relaxed); }
    /// The card has no output running, or the machine stopped.
    pub fn stopped(&self) { self.playing.store(false, Ordering::Relaxed); }
    /// Whether a frame has been written since the card last stopped.
    pub fn playing(&self) -> bool { self.playing.load(Ordering::Relaxed) }
    /// A frame has reached the host ring, or would have, with none open.
    pub fn started(&self) { if !self.playing.load(Ordering::Relaxed) { self.playing.store(true, Ordering::Relaxed); } }
    pub fn frames(&self, output: usize) -> u64 { self.moved[output].load(Ordering::Relaxed) }
    pub fn underruns(&self) -> u64 { self.underruns.load(Ordering::Relaxed) }
    pub fn dropped(&self) -> u64 { self.dropped.load(Ordering::Relaxed) }

    /// What `audio status` prints: the counts, the guest's rate for each
    /// output (`rates`, `None` while off) and where the output goes. `out` is
    /// the open host stream, `wanted` whether host output is on (`no_audio`
    /// is not set).
    pub fn report(&self, rates: &[Option<u32>], out: Option<&HostOut>, wanted: bool) -> String {
        use std::fmt::Write;
        let frames = |n: u64| format!("{n} frame{}", if n == 1 { "" } else { "s" });
        let mut s = format!("audio: {}\n", self.card);
        for (i, (name, rate)) in self.outputs.iter().zip(rates).enumerate() {
            let rate = rate.map_or("off".to_string(), |r| format!("{r} Hz"));
            writeln!(s, "  {name}: {} moved, guest rate {rate}", frames(self.frames(i))).unwrap();
        }
        writeln!(s, "  host underruns: {}\n  dropped by the host: {}", frames(self.underruns()), frames(self.dropped())).unwrap();
        match (out, wanted) {
            (Some(o), _) => writeln!(s, "  host output: {} at {} Hz, {} channels", o.target, o.stream_rate, o.channels),
            (None, true) => writeln!(s, "  host output: none (not open)"),
            (None, false) => writeln!(s, "  host output: none (no_audio)"),
        }
        .unwrap();
        s
    }
}

/// A host output stream: i16 stereo pairs pushed into `producer` play at
/// `stream_rate`, and the stream plays silence when the ring is empty.
pub struct HostOut {
    pub stream_rate: u32,
    /// Channels per frame in the ring and the stream: 2 unless the device
    /// asked for more (see [`open_output_channels`]).
    pub channels: usize,
    producer: Producer<i16>,
    /// The card's counters: the stream counts underruns, [`HostOut::push`]
    /// refusals.
    pub meter: Arc<Meter>,
    /// What plays the ring: the host device and API, or the `[audio] file`.
    pub target: String,
    // What plays the ring -- a host stream or an `[audio] file` -- dropped
    // with the HostOut. Neither for a detached one.
    _stream: Option<cpal::Stream>,
    _file: Option<FilePlayer>,
}

impl HostOut {
    /// Push one input pair through `rs`, counting the frames the ring refused.
    pub fn push(&mut self, rs: &mut Resampler, l: i16, r: i16) {
        self.meter.started();
        let refused = rs.push(l, r, &mut self.producer);
        if refused != 0 {
            self.meter.dropped.fetch_add(refused as u64, Ordering::Relaxed);
        }
    }

    /// Push one whole frame, counting it if the ring refused it.
    pub fn push_frame(&mut self, frame: &[i16]) {
        self.meter.started();
        if !push_frame(&mut self.producer, frame) {
            self.meter.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// An output with no host stream behind it: whoever holds the ring's
    /// consumer plays it. For tests of the devices that feed one.
    #[cfg(test)]
    pub fn detached(meter: Arc<Meter>, stream_rate: u32, producer: Producer<i16>) -> Self {
        Self::detached_channels(meter, stream_rate, 2, producer)
    }

    /// [`HostOut::detached`] with `channels` samples to a frame.
    #[cfg(test)]
    pub fn detached_channels(meter: Arc<Meter>, stream_rate: u32, channels: usize, producer: Producer<i16>) -> Self {
        Self {
            stream_rate,
            channels,
            producer,
            meter,
            target: "test".to_string(),
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
    /// every later pair would swap. Returns how many pairs were dropped.
    pub fn push(&mut self, l: i16, r: i16, prod: &mut Producer<i16>) -> usize {
        if self.passthrough() {
            return usize::from(!push_frame(prod, &[l, r]));
        }
        let mut dropped = 0;
        self.h = [self.h[1], self.h[2], self.h[3], [l as f32, r as f32]];
        let out = self.out_rate as u64;
        while self.acc < out {
            let t = self.acc as f32 / out as f32;
            let [l, r] = [0, 1].map(|c| {
                let v = catmull_rom(self.h[0][c], self.h[1][c], self.h[2][c], self.h[3][c], t);
                v.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16
            });
            dropped += usize::from(!push_frame(prod, &[l, r]));
            self.acc += self.in_rate as u64;
        }
        self.acc -= out;
        dropped
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
/// and start the next one a channel late for good. Returns the frames
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
                missed += 1;
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

/// [`Resampler`] for frames of up to [`FrameResampler::MAX_CHANNELS`]
/// channels, appending to a vector instead of a ring so the caller can mix
/// before it writes. The same Catmull-Rom interpolation and exact rational
/// step.
pub struct FrameResampler {
    pub in_rate: u32,
    pub out_rate: u32,
    channels: usize,
    acc: u64,
    h: [[f32; Self::MAX_CHANNELS]; 4],
}

impl FrameResampler {
    pub const MAX_CHANNELS: usize = 8;

    pub fn new(in_rate: u32, out_rate: u32, channels: usize) -> Self {
        debug_assert!((1..=Self::MAX_CHANNELS).contains(&channels));
        Self { in_rate, out_rate, channels, acc: 0, h: [[0.0; Self::MAX_CHANNELS]; 4] }
    }

    pub fn passthrough(&self) -> bool { self.in_rate == self.out_rate }

    /// Push one input frame; appends the output frames it completes.
    pub fn push(&mut self, frame: &[i16], out: &mut Vec<i16>) {
        let ch = self.channels;
        debug_assert_eq!(frame.len(), ch);
        if self.passthrough() {
            out.extend_from_slice(frame);
            return;
        }
        let mut last = [0.0; Self::MAX_CHANNELS];
        for (d, &v) in last.iter_mut().zip(frame) {
            *d = v as f32;
        }
        self.h = [self.h[1], self.h[2], self.h[3], last];
        let outr = self.out_rate as u64;
        while self.acc < outr {
            let t = self.acc as f32 / outr as f32;
            for c in 0..ch {
                let v = catmull_rom(self.h[0][c], self.h[1][c], self.h[2][c], self.h[3][c], t);
                out.push(v.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16);
            }
            self.acc += self.in_rate as u64;
        }
        self.acc -= outr;
    }
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

/// Open a persistent stereo i16 cpal output stream, trying PREFERRED_RATES in
/// order, with a ring of `ring_samples(rate)` samples. `who` prefixes the
/// messages. The stream plays silence when the ring buffer is empty.
pub fn open_output(
    audio: &AudioConfig,
    meter: Arc<Meter>,
    ring_samples: impl Fn(u32) -> usize,
    who: &str,
) -> Option<HostOut> {
    open_output_channels(audio, 2, meter, ring_samples, who)
}

/// The standard speaker counts a stream is opened with: 7.1, 5.1, quad and
/// stereo.
const CHANNEL_LAYOUTS: [u16; 4] = [8, 6, 4, 2];

/// The (channels, rate) pairs to try, best first: every layout at the
/// preferred rate before any at the next one, so a wide layout at an odd rate
/// never beats stereo at 48 kHz.
fn open_attempts(layouts: &[u16]) -> impl Iterator<Item = (u16, u32)> + '_ {
    PREFERRED_RATES.iter().flat_map(move |&r| layouts.iter().map(move |&c| (c, r)))
}

/// Count the frames a player had to make up with silence, while the card is
/// playing.
#[inline]
pub(crate) fn count_underrun(missed: usize, meter: &Meter) {
    if missed > 0 && meter.playing.load(Ordering::Relaxed) {
        let total = meter.underruns.fetch_add(missed as u64, Ordering::Relaxed) + missed as u64;
        crate::pf_instant!(crate::perfetto::events::HOST_UNDERRUN, total);
    }
}

/// [`open_output`] with as many channels as the device supports, up to
/// `max_channels` (2 to 8). Of the layouts 7.1, 5.1, quad and stereo the
/// largest the device reports and `max_channels` allow is tried first, then
/// the smaller ones. The ring and the stream carry interleaved frames of
/// [`HostOut::channels`] samples; `ring_samples` is the ring's size as for a
/// stereo stream, and a wider one gets proportionally more. `[audio] file`
/// is always a stereo file, whatever the card asks for: [`HostOut::channels`]
/// says so and the card mixes down to it.
pub fn open_output_channels(
    audio: &AudioConfig,
    max_channels: usize,
    meter: Arc<Meter>,
    ring_samples: impl Fn(u32) -> usize,
    who: &str,
) -> Option<HostOut> {
    if let Some(path) = &audio.file {
        return open_file_output(path, meter, ring_samples, who);
    }
    let host = output_host(audio.host);
    let device = host.default_output_device()?;
    let device_max = if max_channels > 2 {
        device
            .supported_output_configs()
            .ok()
            .and_then(|c| c.map(|c| c.channels()).max())
            .unwrap_or(2)
    } else {
        2
    };
    let widest = max_channels.max(2).min(usize::from(device_max.max(2)));
    let layouts: Vec<u16> = CHANNEL_LAYOUTS.into_iter().filter(|&n| usize::from(n) <= widest).collect();
    // A fixed size the device refuses falls back to the host's default.
    let mut sizes = vec![output_buffer(audio, host.id())];
    if sizes[0] != cpal::BufferSize::Default {
        sizes.push(cpal::BufferSize::Default);
    }

    for buffer_size in sizes {
        for (channels, rate) in open_attempts(&layouts) {
            let config = cpal::StreamConfig {
                channels,
                sample_rate: rate,
                buffer_size,
            };
            let ring_size = ring_samples(rate) * usize::from(channels) / 2;
            let who_cb = who.to_string();
            let err_fn = move |err: cpal::Error| { eprintln!("{who_cb}: cpal stream error: {:?}", err); };

            // Try f32 first (macOS CoreAudio native), then i16 (Linux ALSA).
            let (producer, stream) = {
                let (p, mut c) = RingBuffer::<i16>::new(ring_size);
                let meter_cb = meter.clone();
                let data_fn = move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    let missed = pop_frames(&mut c, usize::from(channels), data, |v| v as f32 / 32768.0);
                    count_underrun(missed, &meter_cb);
                };
                match device.build_output_stream(config, data_fn, err_fn.clone(), None) {
                    Ok(s) => (p, s),
                    Err(_) => {
                        // f32 failed, try i16
                        let (p, mut c) = RingBuffer::<i16>::new(ring_size);
                        let meter_cb = meter.clone();
                        let data_fn = move |data: &mut [i16], _: &cpal::OutputCallbackInfo| {
                            let missed = pop_frames(&mut c, usize::from(channels), data, |v| v);
                            count_underrun(missed, &meter_cb);
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
            let target = format!("{device} via {:?}", host.id());
            println!("{who}: audio output: {target} at {rate}Hz, {channels} channels");
            return Some(HostOut {
                stream_rate: rate,
                channels: channels as usize,
                producer,
                meter,
                target,
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
    meter: Arc<Meter>,
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
        let (stop, meter) = (stop.clone(), meter.clone());
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
                count_underrun(pop_frames(&mut c, CHANNELS, &mut buf, |v| v), &meter);
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
        channels: CHANNELS,
        producer,
        meter,
        target: format!("file {}", path.display()),
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
    fn tone_frequency(periods: &mut [f64], rate: u32) -> f64 {
        if periods.is_empty() {
            return 0.0;
        }
        periods.sort_unstable_by(f64::total_cmp);
        let median = periods[periods.len() / 2];
        let near: Vec<f64> = periods.iter().copied().filter(|&n| n >= 0.8 * median && n <= 1.25 * median).collect();
        f64::from(rate) * near.len() as f64 / near.iter().sum::<f64>()
    }

    /// A card as the host path sees it: what the shared tone test needs of
    /// each to play a known tone through its own DMA, pause it, and read what
    /// it moved.
    pub(crate) trait CardUnderTest: Sized {
        /// Names the run's capture file.
        const NAME: &'static str;
        /// Whether a pause leaves the output running with nothing to play (a
        /// guest that fell behind), so the host's silence is an underrun; or
        /// stops the output, so it is idle and counts none.
        const PAUSE_IS_STALL: bool;
        /// The card, started on the real clock and playing to `cfg`, with the
        /// tone in guest memory and the output not yet running.
        fn build(cfg: &AudioConfig) -> Self;
        /// The tone's left and right frequencies in Hz.
        fn tone(&self) -> (f64, f64);
        /// The rate the card plays at, in Hz.
        fn rate(&self) -> f64;
        /// Start, or resume, the output.
        fn play(&self);
        /// Pause the output, as [`CardUnderTest::PAUSE_IS_STALL`] says.
        fn pause(&self);
        /// Let the card run for `ms`, servicing it if it needs it.
        fn wait(&self, ms: u64);
        /// Frames the card has moved so far.
        fn frames(&self) -> f64;
        /// The card's counters.
        fn meter(&self) -> &super::Meter;
        /// Stop the machine and let go of the host output.
        fn finish(&self);
        /// Checks only this card can make of what the host heard, with the
        /// frames it moved in all; `exact` when the host output is a file,
        /// which holds every frame.
        fn check(&self, _heard: &Heard, _moved: f64, _exact: bool) {}
    }

    /// Any card playing a tone for 0.6 s, pausing for 0.3 s so the host
    /// stream runs dry, then playing again. Both sides of the gap must be the
    /// tone with each channel on its own side, as long as the card moved;
    /// the gap counts as underruns if the pause is a stall and not if the
    /// output is stopped. On an `[audio] file` this runs everywhere; on the
    /// real device it checks cpal and the host's sound server too.
    pub(crate) fn a_tone_survives_a_pause<C: CardUnderTest>(backend: Backend) {
        let exact = matches!(backend, Backend::File);
        let Some(ear) = Listener::new(backend, C::NAME) else { return };
        let card = C::build(&ear.cfg);
        let (left, right) = card.tone();
        card.play();
        card.wait(600);
        card.pause();
        let (first, before) = (card.frames(), card.meter().underruns());
        let dropped_first = card.meter().dropped();
        card.wait(300);
        let (paused, underran) = (card.frames(), card.meter().underruns() - before);
        card.play();
        card.wait(600);
        card.pause();
        let second = card.frames() - paused;
        let dropped_second = card.meter().dropped() - dropped_first;
        // What the host still holds plays out.
        card.wait(300);
        let rate = card.rate();
        card.finish();
        let heard = ear.heard();
        // A stall past the host ring's 200 ms makes the card's catch-up
        // burst overflow it. The meter counts the frames the ring refused,
        // at the host's rate; they never reached the host.
        let refused = [dropped_first, dropped_second].map(|d| d as f64 / f64::from(heard.rate));
        assert_a_tone_either_side_of_a_pause(&heard, left, right, 8000, [first / rate, second / rate], refused);
        if C::PAUSE_IS_STALL {
            assert!(underran > 0, "the host stream ran dry while the card had nothing to play");
        } else {
            assert_eq!(underran, 0, "an output the guest stopped is idle: the host's silence is not an underrun");
            assert_eq!(paused, first, "the card moved frames while its output was stopped");
        }
        card.check(&heard, first + second, exact);
    }

    /// Assert that `heard` holds a tone, a silence where its producer
    /// paused, and the tone again. The pause is the silence of at least
    /// 0.1 s after which the sound so far is closest to what the card played
    /// before it, less half of what the ring refused: a host stall leaves
    /// silences too, some as long as the pause, which the longest silence
    /// would take for it. On
    /// each side the left channel must be at `left` Hz and the right at
    /// `right` Hz within 1%, so channels that swap after the pause fail;
    /// the peaks must reach `min_peak`; and the sound must last between
    /// half of what the card played less what the host ring refused
    /// (`refused` seconds) and 110% of what it played, `played` seconds, at
    /// the capture's rate, so repeated frames or a stream that lost most of
    /// its frames fail.
    pub(crate) fn assert_a_tone_either_side_of_a_pause(heard: &Heard, left: f64, right: f64, min_peak: i32, played: [f64; 2], refused: [f64; 2]) {
        let (ch, rate) = (heard.channels, heard.rate);
        let runs = loud_runs(&heard.pcm, ch, rate, min_peak / 4);
        assert!(!runs.is_empty(), "heard nothing: no sound reached the capture");
        // The frames with sound in them, not whole blocks: a busy host
        // leaves dropouts shorter than a block inside a run.
        let sound: Vec<f64> = runs.iter().map(|r| (sounding(&heard.pcm[r.start * ch..r.end * ch], ch).len() / ch) as f64 / f64::from(rate)).collect();
        let silence = |k: usize| (runs[k].start - runs[k - 1].end) as f64 / f64::from(rate);
        let longest = (1..runs.len()).map(silence).fold(0.0, f64::max);
        let pause = (1..runs.len())
            .filter(|&k| silence(k) >= 0.1)
            .min_by(|&a, &b| {
                let off = |k: usize| (sound[..k].iter().sum::<f64>() - (played[0] - refused[0] / 2.0)).abs();
                off(a).total_cmp(&off(b))
            });
        let Some(pause) = pause else { panic!("no pause heard: the longest silence is {longest:.3} s") };
        let (before, after) = runs.split_at(pause);
        let (sound_before, sound_after) = (sound[..pause].iter().sum::<f64>(), sound[pause..].iter().sum::<f64>());
        for (name, runs, secs, want, lost) in [("before the pause", before, sound_before, played[0], refused[0]), ("after the pause", after, sound_after, played[1], refused[1])] {
            // Not more than the card played, within 10%: repeated frames
            // or a stream played too slowly fail. At least half of it less
            // the frames the host ring refused: a card behind its host
            // leaves short dropouts of its own. A wrong rate fails the
            // frequency check.
            assert!(
                secs <= want * 1.1 && secs >= (want - lost) / 2.0,
                "{name}: {secs:.3} s of sound at {rate} Hz, the card played {want:.3} s and the host ring refused {lost:.3} s"
            );
            for (c, hz) in [(0, left), (1, right)] {
                let (mut periods, mut p) = (Vec::new(), 0);
                for r in runs {
                    // Each unbroken stretch on its own: a sample falling
                    // from below zero into the zeros of a dropout is no
                    // rising crossing of the tone, and counting it cuts a
                    // period in two. A few of those pull the mean of the
                    // periods that stay in range past 1%.
                    for stretch in unbroken(&heard.pcm[r.start * ch..r.end * ch], ch) {
                        let s = channel(stretch, ch, c);
                        // Where each rising crossing falls between its two
                        // samples, so a stretch cut short at the tone's own
                        // zero frames gives the true period, not a whole
                        // number of samples.
                        let ups: Vec<f64> = s
                            .windows(2)
                            .enumerate()
                            .filter(|(_, w)| w[0] < 0 && w[1] >= 0)
                            .map(|(i, w)| i as f64 - f64::from(w[0]) / f64::from(i32::from(w[1]) - i32::from(w[0])))
                            .collect();
                        periods.extend(ups.windows(2).map(|w| w[1] - w[0]));
                        p = p.max(peak(&s));
                    }
                }
                let got = tone_frequency(&mut periods, rate);
                // Frames the host ring refused leave jumps in the tone that
                // no stretch shows, and each moves the mean period: its
                // pitch cannot be told then.
                if lost == 0.0 {
                    assert!((got - hz).abs() <= hz / 100.0, "{name}: channel {c} at {got:.1} Hz, wanted {hz:.1} Hz");
                }
                assert!(p >= min_peak, "{name}: channel {c} peak {p}, wanted at least {min_peak}");
            }
        }
    }

    /// The stretches of `pcm` between frames that are zero in every channel:
    /// a dropout is frames of zeros, and a tone has such a frame only where
    /// both channels cross zero together.
    fn unbroken(pcm: &[i16], channels: usize) -> Vec<&[i16]> {
        let (mut stretches, mut start) = (Vec::new(), None);
        for (i, f) in pcm.chunks_exact(channels).enumerate() {
            match (f.iter().any(|&v| v != 0), start) {
                (true, None) => start = Some(i),
                (false, Some(s)) => {
                    stretches.push(&pcm[s * channels..i * channels]);
                    start = None;
                }
                _ => {}
            }
        }
        if let Some(s) = start {
            stretches.push(&pcm[s * channels..]);
        }
        stretches
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

    /// A meter for a card with a running output.
    fn playing_meter() -> Arc<Meter> {
        let m = Meter::new("test", &[]);
        m.started();
        m
    }

    /// The report, whatever the card: its outputs with their rates, the host
    /// stream's underruns, the refusals, and where the output goes.
    #[test]
    fn the_report_has_one_layout_for_every_card() {
        let meter = Meter::new("Example card", &["left", "right"]);
        meter.moved(0, 441);
        let (prod, mut cons) = RingBuffer::<i16>::new(2);
        let mut out = HostOut::detached(meter.clone(), 48000, prod);
        out.push_frame(&[1, 2]);
        out.push_frame(&[3, 4]);
        let mut missed = [0i16; 2];
        count_underrun(pop_frames(&mut cons, 2, &mut missed, |v| v), &meter);
        count_underrun(pop_frames(&mut cons, 2, &mut missed, |v| v), &meter);
        assert_eq!(
            meter.report(&[Some(44100), None], Some(&out), true),
            "audio: Example card\n\
             \x20 left: 441 frames moved, guest rate 44100 Hz\n\
             \x20 right: 0 frames moved, guest rate off\n\
             \x20 host underruns: 1 frame\n\
             \x20 dropped by the host: 1 frame\n\
             \x20 host output: test at 48000 Hz, 2 channels\n"
        );
        assert!(meter.report(&[None, None], None, true).ends_with("host output: none (not open)\n"));
        assert!(meter.report(&[None, None], None, false).ends_with("host output: none (no_audio)\n"));
    }

    /// Silence from the host while no output runs is not an underrun.
    #[test]
    fn an_idle_card_counts_no_underruns() {
        let meter = Meter::new("test", &[]);
        let (prod, mut cons) = RingBuffer::<i16>::new(8);
        let mut out = HostOut::detached(meter.clone(), 48000, prod);
        let mut buf = [0i16; 8];
        count_underrun(pop_frames(&mut cons, 2, &mut buf, |v| v), &meter);
        assert_eq!(meter.underruns(), 0, "idle");
        out.push_frame(&[1, 2]);
        count_underrun(pop_frames(&mut cons, 2, &mut buf, |v| v), &meter);
        assert_eq!(meter.underruns(), 3, "three frames short once a frame has been written");
        meter.stopped();
        count_underrun(pop_frames(&mut cons, 2, &mut buf, |v| v), &meter);
        assert_eq!(meter.underruns(), 3, "stopped");
        out.push_frame(&[3, 4]);
        count_underrun(pop_frames(&mut cons, 2, &mut buf, |v| v), &meter);
        assert_eq!(meter.underruns(), 6, "playing again with the next frame");
    }

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
    fn stereo_at_the_first_rate_comes_before_a_wide_layout_at_a_later_one() {
        let order: Vec<_> = open_attempts(&[8, 2]).collect();
        let at = |c, r| order.iter().position(|&a| a == (c, r)).unwrap();
        assert_eq!(order[0], (8, PREFERRED_RATES[0]));
        assert!(at(2, PREFERRED_RATES[0]) < at(8, PREFERRED_RATES[1]));
    }

    #[test]
    fn the_frame_resampler_matches_the_stereo_one_on_two_channels() {
        for (inr, outr) in [(44100u32, 48000u32), (48000, 44100), (11025, 44100), (8000, 48000)] {
            let (mut prod, mut cons) = RingBuffer::<i16>::new(1 << 16);
            let mut stereo = Resampler::new(inr, outr);
            let mut frames = FrameResampler::new(inr, outr, 2);
            let mut want = Vec::new();
            for i in 0..400i32 {
                let (l, r) = ((i * 97 % 20000 - 10000) as i16, (i * 53 % 30000 - 15000) as i16);
                stereo.push(l, r, &mut prod);
                frames.push(&[l, r], &mut want);
            }
            let got: Vec<i16> = std::iter::from_fn(|| cons.pop().ok()).collect();
            assert_eq!(got, want, "{inr}->{outr}");
        }
    }

    #[test]
    fn frames_keep_their_channels_through_the_frame_resampler() {
        for (inr, outr, per_in) in [(48000u32, 48000u32, 1usize), (24000, 48000, 2), (48000, 24000, 0)] {
            let mut rs = FrameResampler::new(inr, outr, 6);
            let mut out = Vec::new();
            let frame = [100i16, -200, 300, -400, 500, -600];
            for _ in 0..64 {
                rs.push(&frame, &mut out);
            }
            assert_eq!(out.len() % 6, 0);
            let frames = out.len() / 6;
            if per_in > 0 {
                assert_eq!(frames, 64 * per_in);
            } else {
                assert_eq!(frames, 32);
            }
            // Past the warm-up each channel is constant at its own level.
            for f in out.chunks(6).skip(8) {
                assert_eq!(f, frame, "{inr}->{outr}");
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
        assert_eq!(missed, 2, "two frames short");
        // The rest of the frame arrives: it plays whole, first.
        p.push(7).unwrap();
        p.push(8).unwrap();
        let mut out = [9i16; 8];
        assert_eq!(pop_frames(&mut c, 4, &mut out, |v| v), 1);
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
        let meter = playing_meter();
        let start = Instant::now();
        let mut out = open_output(&cfg, meter.clone(), |r| r as usize * 2, "test")
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
        assert!(meter.underruns() > 0, "the silence counted as underruns");
    }

    /// A file that cannot be written opens no output, as a sound card that
    /// refuses does.
    #[test]
    fn an_unwritable_file_opens_no_output() {
        let cfg = AudioConfig { file: Some("/nonexistent-dir/x.wav".into()), ..AudioConfig::default() };
        assert!(open_output(&cfg, playing_meter(), |r| r as usize, "test").is_none());
    }

    /// A card that asks for eight channels still gets a stereo file, and
    /// `HostOut::channels` says so: the card mixes down to what it is given.
    #[test]
    fn a_file_output_asked_for_eight_channels_is_stereo() {
        let (cfg, path) = file_config("file-wide");
        let out = open_output_channels(&cfg, 8, playing_meter(), |r| r as usize, "test")
            .expect("a file output opens without a sound card");
        assert_eq!(out.channels, 2);
        drop(out);
        let (rate, channels, _) = read_wav(&path.0);
        assert_eq!((rate, channels), (FILE_RATE, 2));
    }

    #[test]
    fn the_tone_measures_find_frequency_and_channel() {
        let pcm = tones(48000, 48000, 440.0, 880.0, 10000.0);
        assert_tones(&pcm, 2, 48000, 440.0, 880.0, 9900);
        // Swapped channels are caught.
        let swapped: Vec<i16> = pcm.chunks_exact(2).flat_map(|f| [f[1], f[0]]).collect();
        assert!(std::panic::catch_unwind(|| assert_tones(&swapped, 2, 48000, 440.0, 880.0, 9900)).is_err());
    }

    /// The host output with no card: frames pushed as a card pushes them, in
    /// real time. A pause is a stall: nothing is pushed.
    struct RawCard {
        out: std::sync::Mutex<Option<HostOut>>,
        meter: Arc<Meter>,
        rate: u32,
        tone: Vec<i16>,
        at: std::sync::atomic::AtomicUsize,
        pushed: AtomicU64,
        running: AtomicBool,
    }

    impl CardUnderTest for RawCard {
        const NAME: &'static str = "host-underrun";
        const PAUSE_IS_STALL: bool = true;
        fn build(cfg: &AudioConfig) -> Self {
            let meter = Meter::new("test", &[]);
            let out = open_output(cfg, meter.clone(), |r| r as usize * 2 / 5, "test").expect("the output opens");
            // One second holds whole cycles of both at any rate.
            let tone = tones(out.stream_rate as usize, out.stream_rate, 440.0, 880.0, 16000.0);
            Self { rate: out.stream_rate, out: std::sync::Mutex::new(Some(out)), meter, tone, at: Default::default(), pushed: Default::default(), running: AtomicBool::new(false) }
        }
        fn tone(&self) -> (f64, f64) {
            (440.0, 880.0)
        }
        fn rate(&self) -> f64 {
            f64::from(self.rate)
        }
        fn play(&self) {
            self.running.store(true, Ordering::Relaxed);
        }
        fn pause(&self) {
            self.running.store(false, Ordering::Relaxed);
        }
        /// Push what is due every 2 ms.
        fn wait(&self, ms: u64) {
            let rate = self.rate() as u64;
            let (start, mut pushed) = (Instant::now(), 0u64);
            while start.elapsed() < Duration::from_millis(ms) {
                let due = (start.elapsed().as_nanos() * u128::from(rate) / 1_000_000_000) as u64;
                if let Some(out) = self.out.lock().unwrap().as_mut() {
                    while self.running.load(Ordering::Relaxed) && pushed < due && out.producer.slots() >= 2 {
                        let at = self.at.load(Ordering::Relaxed);
                        out.push_frame(&self.tone[at * 2..at * 2 + 2]);
                        self.at.store((at + 1) % rate as usize, Ordering::Relaxed);
                        pushed += 1;
                        self.pushed.fetch_add(1, Ordering::Relaxed);
                    }
                }
                std::thread::sleep(Duration::from_millis(2));
            }
        }
        fn frames(&self) -> f64 {
            self.pushed.load(Ordering::Relaxed) as f64
        }
        fn meter(&self) -> &Meter {
            &self.meter
        }
        /// Let go of the output, so its file is complete.
        fn finish(&self) {
            self.out.lock().unwrap().take();
        }
    }

    #[test]
    fn a_tone_survives_a_pause_into_a_file() {
        a_tone_survives_a_pause::<RawCard>(Backend::File);
    }

    #[test]
    fn a_tone_survives_a_pause_on_the_real_device() {
        a_tone_survives_a_pause::<RawCard>(Backend::RealDevice);
    }
}
