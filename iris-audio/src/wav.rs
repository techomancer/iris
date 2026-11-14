//! A 16-bit PCM WAV writer for the file backend and the tee.

use std::fs::File;
use std::io::{self, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::{Duration, Instant};

use crate::format::Sample;

/// How stale the sizes in the header may get between appends. A file from
/// a killed run is short by at most this much.
const HEADER_INTERVAL: Duration = Duration::from_millis(250);

/// Writes interleaved samples to a WAV file. The header's sizes are kept
/// nearly current while writing and made exact when the writer is dropped.
pub struct WavWriter {
    file: File,
    rate: u32,
    channels: u16,
    data_bytes: u32,
    header_at: Instant,
    scratch: Vec<u8>,
}

impl WavWriter {
    pub fn create(path: &Path, rate: u32, channels: usize) -> io::Result<Self> {
        let mut w = Self {
            file: File::create(path)?,
            rate,
            channels: channels as u16,
            data_bytes: 0,
            header_at: Instant::now(),
            scratch: Vec::new(),
        };
        w.header()?;
        Ok(w)
    }

    fn header(&mut self) -> io::Result<()> {
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
        self.header_at = Instant::now();
        Ok(())
    }

    /// Append interleaved samples. A WAV's sizes are 32-bit: past about
    /// 4 GiB (six hours of 48 kHz stereo) the file is full and this fails
    /// instead of writing sizes that wrap.
    pub fn append(&mut self, samples: &[f32]) -> io::Result<()> {
        let total = u32::try_from(samples.len() * 2).ok().and_then(|n| self.data_bytes.checked_add(n)).filter(|&t| t <= u32::MAX - 36);
        let Some(total) = total else {
            return Err(io::Error::other("the WAV file is full (4 GiB)"));
        };
        self.scratch.clear();
        self.scratch.extend(samples.iter().flat_map(|&s| i16::from_f32(s).to_le_bytes()));
        self.file.write_all(&self.scratch)?;
        self.data_bytes = total;
        if self.header_at.elapsed() >= HEADER_INTERVAL {
            self.header()?;
        }
        Ok(())
    }
}

impl Drop for WavWriter {
    fn drop(&mut self) {
        let _ = self.header();
    }
}

/// Read a 16-bit PCM WAV written by [`WavWriter`]: its rate, channels and
/// interleaved samples.
pub fn read(path: &Path) -> io::Result<(u32, usize, Vec<i16>)> {
    let bytes = std::fs::read(path)?;
    if bytes.len() < 44 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(io::Error::other("not a WAV file"));
    }
    let channels = u16::from_le_bytes([bytes[22], bytes[23]]) as usize;
    let rate = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
    let len = (u32::from_le_bytes([bytes[40], bytes[41], bytes[42], bytes[43]]) as usize).min(bytes.len() - 44);
    let pcm = bytes[44..44 + len].chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]])).collect();
    Ok((rate, channels, pcm))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::path::PathBuf;

    /// A test's file, removed when the test ends, passed or not.
    pub(crate) struct Temp(pub(crate) PathBuf);

    impl Temp {
        pub(crate) fn new(name: &str) -> Self {
            Self(std::env::temp_dir().join(format!("iris-audio-{}-{name}", std::process::id())))
        }
    }

    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn what_is_written_reads_back() {
        let tmp = Temp::new("roundtrip.wav");
        let mut w = WavWriter::create(&tmp.0, 44100, 2).unwrap();
        w.append(&[0.5, -0.5, 1.0, -1.0]).unwrap();
        w.append(&[0.0, 0.25]).unwrap();
        drop(w);
        let (rate, channels, pcm) = read(&tmp.0).unwrap();
        assert_eq!((rate, channels), (44100, 2));
        assert_eq!(pcm, [16384, -16384, 32767, -32768, 0, 8192]);
    }
}
