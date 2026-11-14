//! Host audio output shared by the machines' sound devices: HAL2 (Indy,
//! Indigo2), MACE audio (O2), RAD1 (Octane, Onyx2) and the ICE1724 (Tezro).
//!
//! The work is done by the `iris-audio` crate. A device opens an
//! [`Output`] at its own rate and writes frames as the guest produces
//! them; rate conversion, following the guest's clock, prebuffering,
//! following the host's default device and surviving its absence all
//! happen on the other side of the stream. This module only turns the
//! `[audio]` section into a stream.

use std::time::Duration;

use iris_audio::{Audio, BackendChoice, OutputConfig};
pub use iris_audio::Output;

use crate::config::{AudioConfig, AudioHost};

/// The rate of an `[audio] file`.
pub const FILE_RATE: u32 = 48000;
/// The highest rate a device may set on its stream.
const MAX_RATE: u32 = 96000;
/// The least the stream may queue before it skips ahead; more when the
/// prebuffer asks for it (see [`open_output`]).
const MAX_LATENCY: Duration = Duration::from_millis(100);

/// Open and start a host output stream for the device `who`, playing
/// `channels`-wide frames at `rate` Hz (which [`Output::set_rate`] changes
/// later) as `[audio]` asks. The stream opens even when the host has no
/// sound device or server just now, and moves to one when it appears, so a
/// device opens once and never retries. `None`, with a message, only when
/// the configuration cannot work: an unwritable `file`, a host this
/// platform does not have.
///
/// The variable `capture_var`, if set to a file, also records what the
/// host is given, as a WAV at the host device's rate.
pub fn open_output(audio: &AudioConfig, who: &str, channels: usize, rate: u32, capture_var: &str) -> Option<Output> {
    Some(open_outputs(audio, who, channels, &[rate], capture_var)?.remove(0))
}

/// [`open_output`] for a device with several DACs: an output for each of
/// `rates`, in that order, each with a rate of its own to change, all
/// playing mixed into one host stream.
pub fn open_outputs(audio: &AudioConfig, who: &str, channels: usize, rates: &[u32], capture_var: &str) -> Option<Vec<Output>> {
    let choice = match (&audio.file, audio.host) {
        (Some(path), _) => BackendChoice::File(path.clone()),
        (None, AudioHost::Default) => BackendChoice::Default,
        (None, AudioHost::Pulse) => BackendChoice::Pulse,
        (None, AudioHost::Wasapi) => BackendChoice::Wasapi,
        (None, AudioHost::Coreaudio) => BackendChoice::CoreAudio,
        (None, AudioHost::Null) => BackendChoice::Null,
    };
    let prebuffer = Duration::from_millis(audio.prebuf_ms);
    let configs: Vec<OutputConfig> = rates.iter().map(|&rate| OutputConfig {
        name: who.to_string(),
        channels,
        rate,
        max_rate: MAX_RATE.max(rate),
        prebuffer,
        max_latency: MAX_LATENCY.max(prebuffer * 3),
        device_latency: audio.latency_ms.map(Duration::from_millis),
        null_fallback: true,
        tee: std::env::var_os(capture_var).map(Into::into),
        ..Default::default()
    }).collect();
    let opened = Audio::named(choice, "IRIS").and_then(|a| Ok((a.backend(), a.open_outputs(&configs)?)));
    match opened {
        Ok((backend, outs)) => {
            let device = outs[0].device().filter(|d| !d.is_empty()).unwrap_or_else(|| "no device yet".into());
            println!("{who}: audio output: {device} via {backend}, {channels} channels");
            Some(outs)
        }
        Err(e) => {
            eprintln!("{who}: no audio output: {e}");
            None
        }
    }
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
    /// card plays through (`default` or `pulse`); the source is the
    /// PulseAudio source that hears the default output, such as a null
    /// sink's monitor: `default:iris_sink.monitor`. Linux only: recording
    /// is done with `parec`.
    pub(crate) const LOOPBACK_VAR: &str = "IRIS_AUDIO_LOOPBACK";

    /// Where a test hears what a card sent to the host.
    pub(crate) enum Backend {
        /// An `[audio] file` of the test's own. Needs no sound card.
        File,
        /// The host's real default output, recorded back through the
        /// loopback [`LOOPBACK_VAR`] names: the host backend, the sound
        /// server and its mixer, none of which the file goes through.
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
        #[cfg(target_os = "linux")]
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

        #[cfg(target_os = "linux")]
        fn loopback(spec: &str) -> Self {
            use crate::config::AudioHost;
            static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());
            let serial = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
            let (host, source) = spec
                .split_once(':')
                .unwrap_or_else(|| panic!("{LOOPBACK_VAR}={spec}: want <output host>:<capture source>"));
            let host = match host {
                "default" => AudioHost::Default,
                "pulse" | "pulseaudio" => AudioHost::Pulse,
                _ => panic!("{LOOPBACK_VAR}={spec}: the output host is default or pulse"),
            };
            let rec = loopback::Recorder::open(source).unwrap_or_else(|e| panic!("{LOOPBACK_VAR}={spec}: {e}"));
            Self { cfg: AudioConfig { host, ..AudioConfig::default() }, ear: Ear::Loopback(rec, serial) }
        }

        #[cfg(not(target_os = "linux"))]
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
                #[cfg(target_os = "linux")]
                Ear::Loopback(rec, _serial) => rec.finish(),
            }
        }
    }

    /// Records a PulseAudio source with `parec` (pulseaudio-utils, which the
    /// loopback's setup needs anyway for `pactl`).
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
            // The frames with sound in them, not whole blocks: a busy host
            // leaves dropouts shorter than a block inside a run.
            let secs = runs.iter().map(|r| sounding(&heard.pcm[r.start * ch..r.end * ch], ch).len() / ch).sum::<usize>() as f64 / f64::from(rate);
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

    /// A WAV's rate, channels and samples.
    pub(crate) fn read_wav(path: &Path) -> (u32, usize, Vec<i16>) {
        iris_audio::wav::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
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
    /// between the first and the last, over the time between them. A
    /// crossing counts only once the signal has been a quarter of its peak
    /// below zero: the host stream's rate converter rings faintly around
    /// zero just before a tone starts and after it stops.
    pub(crate) fn frequency(s: &[i16], rate: u32) -> f64 {
        let low = -(peak(s) / 4).max(1);
        let (mut ups, mut armed) = (Vec::new(), false);
        for (i, &v) in s.iter().enumerate() {
            if i32::from(v) <= low {
                armed = true;
            } else if armed && v >= 0 {
                armed = false;
                ups.push(i);
            }
        }
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

    /// The device end of a stream with no host behind it, for tests of the
    /// cards that feed one: it takes exactly the frames the card has
    /// written, at `rate`, with no prebuffer and no clock following, so a
    /// card playing at `rate` is heard sample for sample.
    pub(crate) struct TestHost {
        renderer: iris_audio::Renderer,
        channels: usize,
        taken: u64,
    }

    /// A detached output of `channels` at `rate` and the host that hears it.
    pub(crate) fn detached(rate: u32, channels: usize) -> (iris_audio::Output, TestHost) {
        use iris_audio::Render;
        let cfg = iris_audio::OutputConfig {
            channels,
            rate,
            prebuffer: std::time::Duration::ZERO,
            max_latency: std::time::Duration::from_secs(2),
            max_adjust: 0.0,
            ..Default::default()
        };
        let (out, mut renderer) = iris_audio::Output::detached(&cfg).expect("a detached output");
        renderer.configure(iris_audio::DeviceFormat { rate, channels });
        (out, TestHost { renderer, channels, taken: 0 })
    }

    impl TestHost {
        /// Everything `out` has been given since the last call.
        pub(crate) fn drain(&mut self, out: &iris_audio::Output) -> Vec<i16> {
            use iris_audio::{Render, Sample};
            let written = out.stats().written;
            let mut buf = vec![0f32; (written - self.taken) as usize * self.channels];
            self.taken = written;
            if !buf.is_empty() {
                self.renderer.render(&mut buf, &iris_audio::RenderInfo::default());
            }
            buf.into_iter().map(i16::from_f32).collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::test_signal::*;
    use std::time::Instant;

    fn open(cfg: &AudioConfig, channels: usize) -> Option<Output> {
        open_output(cfg, "test", channels, FILE_RATE, "IRIS_TEST_CAPTURE")
    }

    /// A file that cannot be written opens no output.
    #[test]
    fn an_unwritable_file_opens_no_output() {
        let cfg = AudioConfig { file: Some("/nonexistent-dir/x.wav".into()), ..AudioConfig::default() };
        assert!(open(&cfg, 2).is_none());
    }

    /// The null host takes a stream anywhere, with no sound device.
    #[test]
    fn the_null_host_opens_everywhere() {
        let cfg = AudioConfig { host: AudioHost::Null, ..AudioConfig::default() };
        let mut out = open(&cfg, 2).expect("the null host opens");
        assert_eq!(out.write(&[0i16; 8]), 4);
    }

    #[test]
    fn a_file_output_is_as_wide_as_the_stream_at_the_file_rate() {
        let (cfg, path) = file_config("file-wide");
        drop(open(&cfg, 6).expect("a file output opens without a sound card"));
        let (rate, channels, _) = read_wav(&path.0);
        assert_eq!((rate, channels), (FILE_RATE, 6));
    }

    #[test]
    fn a_detached_host_hears_what_was_written_sample_for_sample() {
        let (mut out, mut host) = detached(32000, 2);
        let sent = tones(1000, 32000, 440.0, 880.0, 12000.0);
        out.write(&sent[..600]);
        assert_eq!(host.drain(&out), &sent[..600]);
        assert!(host.drain(&out).is_empty());
        out.write(&sent[600..]);
        assert_eq!(host.drain(&out), &sent[600..]);
    }

    #[test]
    fn the_tone_measures_find_frequency_and_channel() {
        let pcm = tones(48000, 48000, 440.0, 880.0, 10000.0);
        assert_tones(&pcm, 2, 48000, 440.0, 880.0, 9900);
        // Swapped channels are caught.
        let swapped: Vec<i16> = pcm.chunks_exact(2).flat_map(|f| [f[1], f[0]]).collect();
        assert!(std::panic::catch_unwind(|| assert_tones(&swapped, 2, 48000, 440.0, 880.0, 9900)).is_err());
    }

    /// Frames written to an output as a card writes them, in real time: a
    /// 440 Hz tone on the left and 880 Hz on the right, nothing for 0.3 s so
    /// the stream runs dry, then the tone again. Both sides of the gap must
    /// be the tone on its own channels, as long as was written, and the gap
    /// must count as an underrun. The host end alone, with no card.
    fn a_tone_survives_an_underrun(backend: Backend) {
        let Some(ear) = Listener::new(backend, "host-underrun") else { return };
        let rate = 44100;
        let mut out = open_output(&ear.cfg, "test", 2, rate, "IRIS_TEST_CAPTURE").expect("the output opens");
        // One second holds whole cycles of both.
        let tone = tones(rate as usize, rate, 440.0, 880.0, 16000.0);
        let mut at = 0usize;
        // Write what is due every 2 ms for `ms`; returns the seconds written.
        let mut play = |out: &mut Output, ms: u64| {
            let (start, mut pushed) = (Instant::now(), 0u64);
            while start.elapsed() < Duration::from_millis(ms) {
                let due = (start.elapsed().as_nanos() * u128::from(rate) / 1_000_000_000) as u64;
                while pushed < due && out.write(&tone[at * 2..at * 2 + 2]) == 1 {
                    at = (at + 1) % (rate as usize);
                    pushed += 1;
                }
                std::thread::sleep(Duration::from_millis(2));
            }
            pushed as f64 / f64::from(rate)
        };
        let first = play(&mut out, 600);
        let before = out.stats().underruns;
        std::thread::sleep(Duration::from_millis(300));
        let underran = out.stats().underruns - before;
        let second = play(&mut out, 600);
        // What the stream still holds plays out.
        std::thread::sleep(Duration::from_millis(300));
        drop(out);
        let heard = ear.heard();
        assert_a_tone_either_side_of_a_pause(&heard, 440.0, 880.0, 8000, [first, second]);
        assert!(underran > 0, "the stream ran dry while nothing was written");
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
