//! Host audio output shared by the machines' sound devices: a persistent
//! cpal stream fed through a lock-free ring, and the resampler from a
//! device's own rate to the stream's. HAL2 (Indy, Indigo2) and MACE audio
//! (O2) both play through it.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use crate::config::{AudioConfig, AudioHost};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rtrb::{RingBuffer, Producer};

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
    // Keep stream alive; dropped with the HostOut. None for a detached one.
    _stream: Option<cpal::Stream>,
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
            push_pair(prod, l, r);
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
            push_pair(prod, l, r);
            self.acc += self.in_rate as u64;
        }
        self.acc -= out;
    }
}

/// Push a stereo pair only if both samples fit; the consumer pops single
/// samples, so the free room can be odd.
fn push_pair(prod: &mut Producer<i16>, l: i16, r: i16) {
    if prod.slots() >= 2 {
        let _ = prod.push(l);
        let _ = prod.push(r);
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
                    let mut short = false;
                    for sample in data.iter_mut() {
                        let v = match c.pop() {
                            Ok(v) => v,
                            Err(_) => {
                                if playing_cb.load(Ordering::Relaxed) {
                                    underruns_cb.fetch_add(1, Ordering::Relaxed);
                                    short = true;
                                }
                                0
                            }
                        };
                        if let Some(w) = capture.as_mut() {
                            use std::io::Write;
                            let _ = w.write_all(&v.to_le_bytes());
                        }
                        *sample = v as f32 / 32768.0;
                    }
                    if short {
                        crate::pf_instant!(crate::perfetto::events::HAL2_UNDERRUN, underruns_cb.load(Ordering::Relaxed));
                    }
                };
                match device.build_output_stream(config, data_fn, err_fn.clone(), None) {
                    Ok(s) => (p, s),
                    Err(_) => {
                        // f32 failed, try i16
                        let (p, mut c) = RingBuffer::<i16>::new(ring_size);
                        let underruns_cb = underruns.clone();
                        let playing_cb = playing.clone();
                        let data_fn = move |data: &mut [i16], _: &cpal::OutputCallbackInfo| {
                            let mut short = false;
                            for sample in data.iter_mut() {
                                *sample = match c.pop() {
                                    Ok(v) => v,
                                    Err(_) => {
                                        if playing_cb.load(Ordering::Relaxed) {
                                            underruns_cb.fetch_add(1, Ordering::Relaxed);
                                            short = true;
                                        }
                                        0
                                    }
                                };
                            }
                            if short {
                                crate::pf_instant!(crate::perfetto::events::HAL2_UNDERRUN, underruns_cb.load(Ordering::Relaxed));
                            }
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
            });
        }

        if buffer_size != cpal::BufferSize::Default {
            eprintln!("{who}: no output stream with a {:?} buffer; trying the host's default", buffer_size);
        }
    }

    eprintln!("{who}: failed to open audio output at any rate (tried {:?})", PREFERRED_RATES);
    None
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
