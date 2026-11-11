//! What a PCI audio card with a host-timed sample clock shares with the
//! others: the host output sink, a WAV capture of it, and the clock that
//! stands still while the machine is stopped.
//!
//! RAD1 was the first such card; a second one can use the same parts so the
//! two do not drift apart.

use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use crate::host_audio::{HostOut, Resampler, WavWriter};

/// Words a host frame carries: the mixer takes two channels.
pub const PCM_CHANNELS: usize = 2;
/// The WAV capture's rate.
const WAV_RATE: u32 = 44100;
/// Most of the card's clock held back for hold-ups, and the share of an
/// interval won back per on-time completion (see [`CardClock::absorb_stall`]).
const STALL_MAX_NS: i64 = 200_000_000;
const STALL_REPAY_DIV: i64 = 32;

/// Where decoded output goes: the host stream (through the shared resampler),
/// an optional 44.1 kHz WAV file of what the card played, and for tests a
/// plain buffer of the card's own frames.
pub struct Sink {
    pub host: Option<(HostOut, Resampler)>,
    /// When the host stream was last tried and failed to open.
    pub host_failed_at: Option<Instant>,
    pub wav: Option<Wav>,
    pub tap: Option<Arc<Mutex<Vec<i16>>>>,
}

/// How long to wait before trying to open the host stream again.
pub const HOST_RETRY: Duration = Duration::from_secs(1);

impl Sink {
    /// Whether the host stream, wanted but not open, is due another try.
    pub fn host_retry_due(&self, now: Instant) -> bool {
        self.host.is_none() && self.host_failed_at.is_none_or(|t| now.duration_since(t) >= HOST_RETRY)
    }

    /// Play stereo frames the card produced at `rate` Hz. A muted output
    /// still moves the ring; the host and the WAV get silence. Returns the
    /// host ring's free and total samples.
    pub fn play(&mut self, pcm: &[i16], rate: u32, muted: bool) -> Option<(usize, usize)> {
        let silence;
        let host_pcm = if muted {
            silence = vec![0i16; pcm.len()];
            &silence[..]
        } else {
            pcm
        };
        if let Some((out, rs)) = self.host.as_mut() {
            if rs.in_rate != rate {
                *rs = Resampler::new(rate, out.stream_rate);
            }
            for p in host_pcm.chunks_exact(PCM_CHANNELS) {
                rs.push(p[0], p[1], &mut out.producer);
            }
        }
        if let Some(w) = self.wav.as_mut() {
            w.play(host_pcm, rate);
        }
        if let Some(t) = self.tap.as_ref() {
            t.lock().extend_from_slice(pcm);
        }
        self.host.as_ref().map(|(out, _)| (out.producer.slots(), out.producer.buffer().capacity()))
    }
}

/// A 44.1 kHz stereo 16-bit WAV of the output, converted from the card's
/// rate, its sizes rewritten after every append so a file from a killed run
/// still plays. A debugging aid, set by the card that owns the sink.
pub struct Wav {
    out: WavWriter,
    rs: Resampler,
}

impl Wav {
    pub fn create(path: &std::ffi::OsStr) -> Option<Self> {
        let out = WavWriter::create(std::path::Path::new(path), WAV_RATE, PCM_CHANNELS as u16).ok()?;
        Some(Self { out, rs: Resampler::new(WAV_RATE, WAV_RATE) })
    }

    /// Append frames the card produced at `rate` Hz.
    fn play(&mut self, pcm: &[i16], rate: u32) {
        if self.rs.in_rate != rate {
            self.rs = Resampler::new(rate, WAV_RATE);
        }
        if self.rs.passthrough() {
            self.append(pcm);
            return;
        }
        // The resampler writes into a ring; take its output back out.
        let frames = pcm.len() / PCM_CHANNELS;
        let room = (frames * WAV_RATE as usize / rate.max(1) as usize + 2) * PCM_CHANNELS;
        let (mut prod, mut cons) = rtrb::RingBuffer::<i16>::new(room);
        for p in pcm.chunks_exact(PCM_CHANNELS) {
            self.rs.push(p[0], p[1], &mut prod);
        }
        let out: Vec<i16> = std::iter::from_fn(|| cons.pop().ok()).collect();
        self.append(&out);
    }

    pub fn append(&mut self, pcm: &[i16]) {
        let _ = self.out.append(pcm);
    }
}

/// A card's sample clock: host time less the time the machine was stopped
/// and less the time held back after a late timer. Lock order: `tick_pass`,
/// then `stopped_at` (a leaf, taken by `now`).
pub struct CardClock {
    pub start: Instant,
    /// In tests, the clock in nanoseconds once a test sets it; `i64::MIN`
    /// follows the host.
    pub(crate) fake_ns: AtomicI64,
    /// Nanoseconds the clock stood still while the machine was stopped, and
    /// when the current stop began: the sample clock and whatever is
    /// scheduled from it do not run on while the guest cannot see them.
    pub stopped_ns: AtomicI64,
    pub stopped_at: Mutex<Option<Instant>>,
    /// Nanoseconds taken off the clock for timer hold-ups (see
    /// [`CardClock::absorb_stall`]), and the latest time it has handed out.
    pub stalled_ns: AtomicI64,
    pub seen_ns: AtomicI64,
    /// Held across each timer callback's work, so a stop can wait one out.
    pub tick_pass: Mutex<()>,
    /// Between `Device::start` and `Device::stop`: timers are armed only
    /// then.
    pub armed: AtomicBool,
}

impl Default for CardClock {
    fn default() -> Self {
        Self::new()
    }
}

impl CardClock {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
            fake_ns: AtomicI64::new(i64::MIN),
            stopped_ns: AtomicI64::new(0),
            stalled_ns: AtomicI64::new(0),
            seen_ns: AtomicI64::new(i64::MIN),
            stopped_at: Mutex::new(None),
            tick_pass: Mutex::new(()),
            armed: AtomicBool::new(false),
        }
    }

    pub fn now(&self) -> i64 {
        let at = self.stopped_at.lock();
        let t = self.clock_locked(&at);
        self.seen_ns.fetch_max(t, Ordering::SeqCst);
        t
    }

    /// The card's clock, with `stopped_at` held; not recorded as seen.
    pub fn clock_locked(&self, at: &Option<Instant>) -> i64 {
        let at = at.unwrap_or_else(Instant::now);
        let host = at.duration_since(self.start).as_nanos() as i64 - self.stopped_ns.load(Ordering::SeqCst);
        let host = match self.fake_ns.load(Ordering::Relaxed) {
            i64::MIN => host,
            t => t,
        };
        host - self.stalled_ns.load(Ordering::SeqCst)
    }

    /// Whether time passes by itself, so timers may be armed.
    pub fn runs_by_itself(&self) -> bool {
        self.fake_ns.load(Ordering::Relaxed) == i64::MIN
    }

    /// Set the clock to read `now` with `stalled` held back, as a restored
    /// card's clock does.
    pub fn restore(&self, now: i64, stalled: i64) {
        self.stalled_ns.store(stalled, Ordering::SeqCst);
        if self.fake_ns.load(Ordering::Relaxed) != i64::MIN {
            self.fake_ns.store(now + stalled, Ordering::Relaxed);
            self.seen_ns.store(now, Ordering::SeqCst);
            return;
        }
        let at = self.stopped_at.lock().unwrap_or_else(Instant::now);
        let since_start = at.duration_since(self.start).as_nanos() as i64;
        self.stopped_ns.store(since_start - now - stalled, Ordering::SeqCst);
        self.seen_ns.store(now, Ordering::SeqCst);
    }

    /// The card's clock as a completion due at `due` sees it. One more than
    /// `interval` late means the host held up the timer thread (and the
    /// guest's clock interrupt with it); crediting that would replay the
    /// time in one burst. So the clock goes back to `due`, at most
    /// [`STALL_MAX_NS`] in all and never behind a time already handed out.
    /// An on-time one wins back `interval / STALL_REPAY_DIV` of what is held
    /// back. Emulator approximation; see the cards' module notes.
    pub fn absorb_stall(&self, due: i64, interval: i64) -> i64 {
        let at = self.stopped_at.lock();
        let t = self.clock_locked(&at);
        let held = self.stalled_ns.load(Ordering::SeqCst);
        let late = t - due;
        if late > interval {
            crate::pf_instant!(crate::perfetto::events::RAD_LATE, late / 1000);
        }
        let shift = if late > interval {
            let floor = self.seen_ns.load(Ordering::SeqCst);
            -late.min(STALL_MAX_NS - held).min(t.saturating_sub(floor)).max(0)
        } else {
            held.min(interval / STALL_REPAY_DIV)
        };
        self.stalled_ns.fetch_sub(shift, Ordering::SeqCst);
        let t = t + shift;
        self.seen_ns.fetch_max(t, Ordering::SeqCst);
        t
    }

    /// Stop the clock where it is.
    pub fn stop_clock(&self) {
        let mut at = self.stopped_at.lock();
        if at.is_none() {
            *at = Some(Instant::now());
        }
    }

    /// Run the clock again from where it stopped.
    pub fn start_clock(&self) {
        let mut at = self.stopped_at.lock();
        if let Some(t) = at.take() {
            self.stopped_ns.fetch_add(t.elapsed().as_nanos() as i64, Ordering::SeqCst);
        }
    }

    pub fn stopped(&self) -> bool {
        self.stopped_at.lock().is_some()
    }
}

/// The test clock's value, to read and set by hand.
#[cfg(test)]
pub(crate) mod test_time {
    use super::*;

    pub(crate) struct TestTime;

    impl TestTime {
        /// A clock that follows the host until a test sets `fake_ns`.
        pub(crate) fn clock() -> CardClock {
            CardClock::new()
        }
    }

    pub(crate) trait FakeNs {
        fn fake_ns(&self) -> &AtomicI64;
    }

    impl FakeNs for CardClock {
        fn fake_ns(&self) -> &AtomicI64 {
            &self.fake_ns
        }
    }
}
