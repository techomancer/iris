//! One recorder for what a device's audio output plays, set by
//! `IRIS_AUDIO_DUMP=<file>`.
//!
//! The file is a run of records, each a sample rate and a frame count (u32
//! each) followed by that many left, right pairs of i16, all little-endian.
//! Frames are recorded as the device produces them, at its own rate and
//! before any resampling, with or without a host sound device, so a muted
//! run can still be checked. A new record starts every time the device plays
//! a burst, so a rate change shows up as a record with a different rate. The
//! O2 has two output channels and prefixes each record with the channel
//! number (u8); the other machines have one stream and no prefix.
//!
//! A device holds an `Option<AudioDump>` built once when it is created, so a
//! run without the variable pays one branch per burst and nothing else.

use parking_lot::Mutex;
use std::ffi::OsStr;
use std::io::{BufWriter, Write};

/// The environment variable that names the dump file.
pub const ENV_VAR: &str = "IRIS_AUDIO_DUMP";

pub struct AudioDump {
    out: Mutex<BufWriter<std::fs::File>>,
}

impl AudioDump {
    /// The dump `IRIS_AUDIO_DUMP` asks for, if it is set and the file opens.
    pub fn from_env() -> Option<Self> {
        Self::from_var(std::env::var_os(ENV_VAR).as_deref())
    }

    fn from_var(path: Option<&OsStr>) -> Option<Self> {
        Self::create(path?)
    }

    pub fn create(path: &OsStr) -> Option<Self> {
        Some(Self { out: Mutex::new(BufWriter::new(std::fs::File::create(path).ok()?)) })
    }

    /// Append one record of `frames` played at `rate` Hz; `channel` is the
    /// O2's output number and `None` everywhere else. An empty burst leaves
    /// no record. Every record is flushed, so a killed run keeps its tune.
    pub fn record(&self, channel: Option<u8>, rate: u32, frames: impl ExactSizeIterator<Item = (i16, i16)>) {
        if frames.len() == 0 {
            return;
        }
        let mut rec = Vec::with_capacity(9 + frames.len() * 4);
        rec.extend(channel);
        rec.extend_from_slice(&rate.to_le_bytes());
        rec.extend_from_slice(&(frames.len() as u32).to_le_bytes());
        for (l, r) in frames {
            rec.extend_from_slice(&l.to_le_bytes());
            rec.extend_from_slice(&r.to_le_bytes());
        }
        let mut out = self.out.lock();
        let _ = out.write_all(&rec);
        let _ = out.flush();
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
}
