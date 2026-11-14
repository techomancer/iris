//! The one recorder for what a sound card plays, set by
//! `IRIS_AUDIO_DUMP=<file>`.
//!
//! The file is a run of records: an output number (u8; 0 except for the
//! O2's second DAC), a sample rate and a frame count (u32 each), then that
//! many left, right pairs of i16, all little-endian. Each burst a card plays
//! is its own record, at the card's rate, before any conversion and with or
//! without a host sound device. Cards in one process share one file, created
//! once per process.

use parking_lot::Mutex;
use std::ffi::OsStr;
use std::io::Write;
use std::sync::{Arc, OnceLock};

/// The environment variable that names the dump file.
pub const ENV_VAR: &str = "IRIS_AUDIO_DUMP";

pub struct AudioDump {
    out: Mutex<Box<dyn Write + Send>>,
}

impl AudioDump {
    /// The dump `IRIS_AUDIO_DUMP` asks for, if it is set; a file that cannot
    /// be created is reported once and the recorder stays off.
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
        Some(Self::to(std::fs::File::create(path?).ok()?))
    }

    /// A dump into `out`, which should not buffer: every record is one
    /// `write_all`, so a killed run keeps what was played.
    pub fn to(out: impl Write + Send + 'static) -> Self {
        Self { out: Mutex::new(Box::new(out)) }
    }

    /// Append one record of `frames` played at `rate` Hz on output `out`.
    /// An empty burst leaves no record.
    pub fn record(&self, out: u8, rate: u32, frames: impl ExactSizeIterator<Item = (i16, i16)>) {
        if frames.len() == 0 {
            return;
        }
        let mut rec = Vec::with_capacity(9 + frames.len() * 4);
        rec.push(out);
        rec.extend_from_slice(&rate.to_le_bytes());
        rec.extend_from_slice(&(frames.len() as u32).to_le_bytes());
        for (l, r) in frames {
            rec.extend_from_slice(&l.to_le_bytes());
            rec.extend_from_slice(&r.to_le_bytes());
        }
        let _ = self.out.lock().write_all(&rec);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// What a dump holds, readable while the dump is still open.
    #[derive(Clone, Default)]
    pub(crate) struct Bytes(Arc<Mutex<Vec<u8>>>);

    impl Write for Bytes {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.0.lock().extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl Bytes {
        pub(crate) fn get(&self) -> Vec<u8> {
            self.0.lock().clone()
        }

        /// The records of a dump, as (rate, frames).
        pub(crate) fn records(&self) -> Vec<(u32, Vec<(i16, i16)>)> {
            let bytes = self.get();
            let mut out = Vec::new();
            let mut at = 0;
            while at < bytes.len() {
                let rate = u32::from_le_bytes(bytes[at + 1..at + 5].try_into().unwrap());
                let n = u32::from_le_bytes(bytes[at + 5..at + 9].try_into().unwrap()) as usize;
                at += 9;
                let frames = bytes[at..at + n * 4]
                    .chunks_exact(4)
                    .map(|f| (i16::from_le_bytes([f[0], f[1]]), i16::from_le_bytes([f[2], f[3]])))
                    .collect();
                at += n * 4;
                out.push((rate, frames));
            }
            out
        }
    }

    pub(crate) fn dump() -> (Arc<AudioDump>, Bytes) {
        let bytes = Bytes::default();
        (Arc::new(AudioDump::to(bytes.clone())), bytes)
    }

    /// The layout, byte for byte: each burst is its own record at its own
    /// rate, and an empty burst leaves none.
    #[test]
    fn a_record_is_output_rate_count_and_pairs() {
        let (d, bytes) = dump();
        d.record(0, 44100, [(1, -2), (300, -400)].into_iter());
        d.record(0, 44100, std::iter::empty());
        d.record(1, 8000, [(i16::MIN, i16::MAX)].into_iter());
        let want: Vec<u8> = [
            &[0, 0x44, 0xac, 0, 0][..], // output 0, 44100
            &[2, 0, 0, 0],
            &[1, 0, 0xfe, 0xff, 0x2c, 0x01, 0x70, 0xfe],
            &[1, 0x40, 0x1f, 0, 0], // output 1, 8000
            &[1, 0, 0, 0],
            &[0x00, 0x80, 0xff, 0x7f],
        ]
        .concat();
        assert_eq!(bytes.get(), want);
    }

    /// Without the variable there is no recorder and no file, and a path
    /// that cannot be created leaves the recorder off.
    #[test]
    fn without_the_variable_there_is_no_recorder_and_no_file() {
        assert!(AudioDump::from_var(None).is_none());
        let dir = std::env::temp_dir().join(format!("iris-audio-dump-{}", std::process::id()));
        assert!(AudioDump::from_var(Some(dir.join("no-such-dir").join("x").as_os_str())).is_none());
        assert!(!dir.exists());
    }
}
