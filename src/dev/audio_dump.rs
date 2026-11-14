//! One recorder for what every audio card plays, set by
//! `IRIS_AUDIO_DUMP=<file>`.
//!
//! The file is a run of records, each a sample rate and a frame count (u32
//! each) followed by that many left, right pairs of i16, all little-endian.
//! Frames are recorded as the card produces them, at its own rate and before
//! any resampling or host mixing, with or without a host sound device, so a
//! muted run can be checked sample for sample against what the guest
//! computed. A new record starts every time the card plays a burst, so a rate
//! change shows up as a record with a different rate. The O2 has two outputs
//! and prefixes each record with the output number (u8); the other cards have
//! one stream and no prefix. The ICE1724 records the front pair of its mix.
//! For a WAV of what the host plays, use `[audio] output = { file = ... }` instead.
//!
//! Each card records where it makes its frames, which is not the same point
//! everywhere: HAL2 records what it reads from DMA, before its output gain;
//! RAD1 records after its mute words (a muted card records silence); MACE
//! records after the DAC gain.
//!
//! A card holds an `Option<Arc<AudioDump>>` built once when it is created, so
//! a run without the variable pays one branch per burst and nothing else.
//! Cards in one process share one file, so a second card does not truncate
//! the first one's records; records of two cards then interleave and the
//! format does not say which card wrote which.

use parking_lot::Mutex;
use std::ffi::OsStr;
use std::io::{BufWriter, Write};
use std::sync::{Arc, OnceLock};

/// The environment variable that names the dump file.
pub const ENV_VAR: &str = "IRIS_AUDIO_DUMP";

pub struct AudioDump {
    out: Mutex<BufWriter<std::fs::File>>,
}

impl AudioDump {
    /// The dump `IRIS_AUDIO_DUMP` asks for, if it is set and the file opens.
    /// A file that cannot be created is reported once and the recorder stays
    /// off for the process: the path will not get better, and retrying per
    /// card would only repeat the message.
    pub fn from_env() -> Option<Arc<Self>> {
        static SHARED: OnceLock<Option<Arc<AudioDump>>> = OnceLock::new();
        SHARED
            .get_or_init(|| {
                let path = std::env::var_os(ENV_VAR);
                let dump = Self::from_var(path.as_deref());
                if let (None, Some(p)) = (&dump, &path) {
                    eprintln!("{ENV_VAR}: cannot create {}: audio is not recorded", p.to_string_lossy());
                }
                dump.map(Arc::new)
            })
            .clone()
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
pub(crate) mod tests {
    use super::*;

    pub(crate) fn temp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("iris-audio-dump-{}-{name}", std::process::id()))
    }

    /// The records of a dump file without output numbers, as (rate, frames).
    pub(crate) fn records(bytes: &[u8]) -> Vec<(u32, Vec<(i16, i16)>)> {
        let mut out = Vec::new();
        let mut at = 0;
        while at < bytes.len() {
            let rate = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
            let n = u32::from_le_bytes(bytes[at + 4..at + 8].try_into().unwrap()) as usize;
            at += 8;
            let frames = bytes[at..at + n * 4]
                .chunks_exact(4)
                .map(|f| (i16::from_le_bytes([f[0], f[1]]), i16::from_le_bytes([f[2], f[3]])))
                .collect();
            at += n * 4;
            out.push((rate, frames));
        }
        out
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
        assert_eq!(records(&bytes).len(), 2);
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
