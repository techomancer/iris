//! One recorder for what a device's audio output plays, set by
//! `IRIS_AUDIO_DUMP=<file>`. The file's extension picks the layout.
//!
//! Anything but `.wav` is the raw record layout, for comparing a PROM's tune
//! sample for sample: a run of records, each a sample rate and a frame count
//! (u32 each) followed by that many left, right pairs of i16, all
//! little-endian. Frames are recorded as the device produces them, at its own
//! rate and before any resampling, with or without a host sound device, so a
//! muted run can still be checked. A new record starts every time the device
//! plays a burst, so a rate change shows up as a record with a different
//! rate. The O2 has two output channels and prefixes each record with the
//! channel number (u8); the other machines have one stream and no prefix.
//!
//! `.wav` is a 44.1 kHz stereo 16-bit WAV of the same frames, converted from
//! the device's rate by the resampler the host stream uses, for listening to
//! or opening in an editor. Its sizes are rewritten after every append, so a
//! file from a killed run still plays. A WAV holds one stream: of the O2's two
//! outputs only output 0 goes in.
//!
//! A device holds an `Option<AudioDump>` built once when it is created, so a
//! run without the variable pays one branch per burst and nothing else.

use crate::host_audio::Resampler;
use parking_lot::Mutex;
use std::ffi::OsStr;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

/// The environment variable that names the dump file.
pub const ENV_VAR: &str = "IRIS_AUDIO_DUMP";

/// The rate of a `.wav` dump.
pub const WAV_RATE: u32 = 44100;

pub struct AudioDump {
    out: Mutex<Out>,
}

enum Out {
    Raw(BufWriter<std::fs::File>),
    Wav(Wav),
}

impl AudioDump {
    /// The dump `IRIS_AUDIO_DUMP` asks for, if it is set and the file opens.
    pub fn from_env() -> Option<Self> {
        Self::from_var(std::env::var_os(ENV_VAR).as_deref())
    }

    fn from_var(path: Option<&OsStr>) -> Option<Self> {
        Self::create(path?)
    }

    /// Open `path`: a WAV when it ends in `.wav` (any case), raw records otherwise.
    pub fn create(path: &OsStr) -> Option<Self> {
        let wav = Path::new(path).extension().is_some_and(|e| e.eq_ignore_ascii_case("wav"));
        let file = std::fs::File::create(path).ok()?;
        let out = if wav { Out::Wav(Wav::new(file)?) } else { Out::Raw(BufWriter::new(file)) };
        Some(Self { out: Mutex::new(out) })
    }

    /// Append `frames` played at `rate` Hz; `channel` is the O2's output
    /// number and `None` everywhere else. An empty burst leaves no record.
    /// Every burst is flushed, so a killed run keeps its tune.
    pub fn record(&self, channel: Option<u8>, rate: u32, frames: impl ExactSizeIterator<Item = (i16, i16)>) {
        if frames.len() == 0 {
            return;
        }
        match &mut *self.out.lock() {
            Out::Raw(w) => {
                let mut rec = Vec::with_capacity(9 + frames.len() * 4);
                rec.extend(channel);
                rec.extend_from_slice(&rate.to_le_bytes());
                rec.extend_from_slice(&(frames.len() as u32).to_le_bytes());
                for (l, r) in frames {
                    rec.extend_from_slice(&l.to_le_bytes());
                    rec.extend_from_slice(&r.to_le_bytes());
                }
                let _ = w.write_all(&rec);
                let _ = w.flush();
            }
            Out::Wav(w) => {
                if channel.is_none_or(|c| c == 0) {
                    w.play(rate, frames);
                }
            }
        }
    }
}

/// A 44.1 kHz stereo 16-bit WAV, converted from the device's rate.
struct Wav {
    file: std::fs::File,
    data_bytes: u32,
    rs: Resampler,
}

impl Wav {
    fn new(file: std::fs::File) -> Option<Self> {
        let mut w = Self { file, data_bytes: 0, rs: Resampler::new(WAV_RATE, WAV_RATE) };
        w.header().ok()?;
        Some(w)
    }

    fn play(&mut self, rate: u32, frames: impl ExactSizeIterator<Item = (i16, i16)>) {
        if self.rs.in_rate != rate {
            self.rs = Resampler::new(rate, WAV_RATE);
        }
        // The resampler writes into a ring; take its output back out.
        let room = (frames.len() * WAV_RATE as usize / rate.max(1) as usize + 2) * 2;
        let (mut prod, mut cons) = rtrb::RingBuffer::<i16>::new(room);
        for (l, r) in frames {
            self.rs.push(l, r, &mut prod);
        }
        let out: Vec<i16> = std::iter::from_fn(|| cons.pop().ok()).collect();
        self.append(&out);
    }

    /// (Re)write the 44-byte header for the data so far.
    fn header(&mut self) -> std::io::Result<()> {
        let mut h = Vec::with_capacity(44);
        h.extend_from_slice(b"RIFF");
        h.extend_from_slice(&(36 + self.data_bytes).to_le_bytes());
        h.extend_from_slice(b"WAVEfmt ");
        h.extend_from_slice(&16u32.to_le_bytes());
        h.extend_from_slice(&1u16.to_le_bytes());
        h.extend_from_slice(&2u16.to_le_bytes());
        h.extend_from_slice(&WAV_RATE.to_le_bytes());
        h.extend_from_slice(&(WAV_RATE * 4).to_le_bytes());
        h.extend_from_slice(&4u16.to_le_bytes());
        h.extend_from_slice(&16u16.to_le_bytes());
        h.extend_from_slice(b"data");
        h.extend_from_slice(&self.data_bytes.to_le_bytes());
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&h)?;
        self.file.seek(SeekFrom::End(0))?;
        Ok(())
    }

    fn append(&mut self, pcm: &[i16]) {
        let bytes: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
        if self.file.write_all(&bytes).is_ok() {
            self.data_bytes = self.data_bytes.wrapping_add(bytes.len() as u32);
            let _ = self.header();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("iris-audio-dump-{}-{name}", std::process::id()))
    }

    fn record(rate: u32, pairs: &[(i16, i16)]) -> Vec<u8> {
        let mut v = rate.to_le_bytes().to_vec();
        v.extend_from_slice(&(pairs.len() as u32).to_le_bytes());
        for (l, r) in pairs {
            v.extend_from_slice(&l.to_le_bytes());
            v.extend_from_slice(&r.to_le_bytes());
        }
        v
    }

    /// Each burst is its own record at its own rate, an empty burst leaves
    /// none, and the bytes are on disk while the dump is still open.
    #[test]
    fn a_rate_change_starts_a_new_record_and_every_record_is_flushed() {
        let path = temp("rate");
        let dump = AudioDump::create(path.as_os_str()).unwrap();
        dump.record(None, 44100, [(1, -2), (300, -400)].into_iter());
        dump.record(None, 44100, std::iter::empty());
        dump.record(None, 32000, [(i16::MIN, i16::MAX)].into_iter());
        let bytes = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let mut want = record(44100, &[(1, -2), (300, -400)]);
        want.extend(record(32000, &[(i16::MIN, i16::MAX)]));
        assert_eq!(bytes, want);
    }

    /// The O2's two outputs share a file, told apart by a leading byte.
    #[test]
    fn a_channel_number_prefixes_the_record() {
        let path = temp("channel");
        let dump = AudioDump::create(path.as_os_str()).unwrap();
        dump.record(Some(1), 8000, [(5, 6)].into_iter());
        let bytes = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let mut want = vec![1];
        want.extend(record(8000, &[(5, 6)]));
        assert_eq!(bytes, want);
    }

    #[test]
    fn without_the_variable_there_is_no_recorder_and_no_file() {
        assert!(AudioDump::from_var(None).is_none());
        let missing = temp("no-such-dir").join("x");
        assert!(AudioDump::from_var(Some(missing.as_os_str())).is_none());
    }

    /// The extension picks the layout, whatever its case; no extension, or
    /// any other, is the raw one.
    #[test]
    fn the_extension_picks_the_layout() {
        for (name, wav) in [("wav", true), ("WAV", true), ("dump", false), ("bin", false)] {
            let path = temp(&format!("ext.{name}"));
            let dump = AudioDump::create(path.as_os_str()).unwrap();
            let b = std::fs::read(&path).unwrap();
            let _ = std::fs::remove_file(&path);
            assert_eq!(b.starts_with(b"RIFF"), wav, "{name}");
            assert_eq!(matches!(*dump.out.lock(), Out::Wav(_)), wav, "{name}");
        }
    }

    /// A WAV has a valid header after every append, so a killed run still
    /// plays: the sizes track the data and the file is only ever header plus
    /// whole frames.
    #[test]
    fn a_wav_has_valid_sizes_after_each_append() {
        let path = temp("sizes.wav");
        let dump = AudioDump::create(path.as_os_str()).unwrap();
        let b = std::fs::read(&path).unwrap();
        assert_eq!((b.len(), u32::from_le_bytes(b[40..44].try_into().unwrap())), (44, 0));
        dump.record(None, WAV_RATE, [(1, -1), (2, -2)].into_iter());
        let b = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(&b[..4], b"RIFF");
        assert_eq!(u32::from_le_bytes(b[4..8].try_into().unwrap()), 36 + 8);
        assert_eq!(&b[8..16], b"WAVEfmt ");
        assert_eq!(u16::from_le_bytes(b[22..24].try_into().unwrap()), 2, "channels");
        assert_eq!(u32::from_le_bytes(b[24..28].try_into().unwrap()), 44100);
        assert_eq!(&b[36..40], b"data");
        assert_eq!(u32::from_le_bytes(b[40..44].try_into().unwrap()), 8);
        assert_eq!(i16::from_le_bytes(b[46..48].try_into().unwrap()), -1);
        assert_eq!(b.len(), 52);
    }

    /// The WAV stays at 44.1 kHz whatever the device's rate, and the O2's
    /// second output is left out of it.
    #[test]
    fn a_wav_converts_to_44100_and_keeps_one_stream() {
        let path = temp("rate.wav");
        let dump = AudioDump::create(path.as_os_str()).unwrap();
        dump.record(None, 22050, std::iter::repeat((0, 0)).take(2205));
        dump.record(None, 44100, std::iter::repeat((0, 0)).take(4410));
        dump.record(Some(1), 44100, std::iter::repeat((9, 9)).take(4410));
        let b = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(u32::from_le_bytes(b[24..28].try_into().unwrap()), 44100);
        let frames = u32::from_le_bytes(b[40..44].try_into().unwrap()) / 4;
        assert!((8818..=8822).contains(&frames), "{frames} frames for 200 ms");
        assert!(b[44..].iter().all(|&x| x == 0), "output 1 is not in the WAV");
    }
}
