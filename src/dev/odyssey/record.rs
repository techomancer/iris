//! A replayable recording of what reaches the VPro board, for regression
//! tests (`golden/replay.rs`): every bus store the processors make, the
//! loads that are not plain repeats, and the guest memory the board's DMA
//! reads. Replaying the stores from power-on through a fresh board, with the
//! DMA reads answered from the recording, must reproduce the planes the live
//! run drew; the replay checks them against hash checkpoints kept beside the
//! recording.
//!
//! Started with `IRIS_ODSY_REC=<file>` when the board is built (power-on is
//! what a replay starts from). The recording is gzip-compressed whatever the
//! name. It stays out of git, as IMPACT's `.rec` files do; the checkpoints
//! are what the tree keeps (`src/dev/odyssey/testdata/replay/`).
//!
//! File: `ODSYREC` + NUL + u32 version, then records, little-endian:
//!
//! | tag | fields                         | what                                  |
//! |-----|--------------------------------|---------------------------------------|
//! | `W` | size u8, off u32, val u64      | bus store, `off` within the aperture  |
//! | `R` | size u8, off u32, val u64      | bus load and what it answered; an access identical to the one before it is not kept |
//! | `B` | addr u64, n u32, n x u64       | DMA read of n qwords from system memory |
//! | `b` | addr u64, val u8               | DMA read of one byte                  |
//!
//! Loads the board answered busy are not kept: the processor retries them.

use std::io::{Read, Write};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::traits::{BusDevice, BusRead8, BUS_OK};

const MAGIC: &[u8; 8] = b"ODSYREC\0";
const VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Rec {
    Write { size: u8, off: u32, val: u64 },
    Read { size: u8, off: u32, val: u64 },
    DmaBlock { addr: u64, words: Vec<u64> },
    DmaByte { addr: u64, val: u8 },
}

pub(super) struct Recorder {
    out: flate2::write::GzEncoder<std::io::BufWriter<std::fs::File>>,
    pub records: u64,
    last_read: Option<(u8, u32, u64)>,
}

/// Shared so the DMA path can record through `RecordingMem` while the bus
/// path records through the widget.
pub(super) type RecHandle = Arc<Mutex<Recorder>>;

impl Recorder {
    pub fn create(path: &str) -> std::io::Result<RecHandle> {
        let file = std::io::BufWriter::with_capacity(1 << 20, std::fs::File::create(path)?);
        let mut out = flate2::write::GzEncoder::new(file, flate2::Compression::fast());
        out.write_all(MAGIC)?;
        out.write_all(&VERSION.to_le_bytes())?;
        Ok(Arc::new(Mutex::new(Recorder { out, records: 0, last_read: None })))
    }

    /// Sync-flush now and then, so a recording cut off by a kill still reads.
    fn tick(&mut self) {
        if self.records % 4096 == 0 {
            let _ = self.out.flush();
        }
    }

    pub fn write(&mut self, size: u8, off: u32, val: u64) {
        self.records += 1;
        self.tick();
        self.last_read = None;
        let _ = self.out.write_all(&[b'W', size]).and_then(|_| self.out.write_all(&off.to_le_bytes())).and_then(|_| self.out.write_all(&val.to_le_bytes()));
    }

    pub fn read(&mut self, size: u8, off: u32, val: u64) {
        if self.last_read == Some((size, off, val)) {
            return;
        }
        self.last_read = Some((size, off, val));
        self.records += 1;
        self.tick();
        let _ = self.out.write_all(&[b'R', size]).and_then(|_| self.out.write_all(&off.to_le_bytes())).and_then(|_| self.out.write_all(&val.to_le_bytes()));
    }

    fn dma_block(&mut self, addr: u64, words: &[u64]) {
        self.records += 1;
        let _ = self.out.write_all(b"B").and_then(|_| self.out.write_all(&addr.to_le_bytes())).and_then(|_| self.out.write_all(&(words.len() as u32).to_le_bytes()));
        for w in words {
            let _ = self.out.write_all(&w.to_le_bytes());
        }
    }

    fn dma_byte(&mut self, addr: u64, val: u8) {
        self.records += 1;
        let _ = self.out.write_all(b"b").and_then(|_| self.out.write_all(&addr.to_le_bytes())).and_then(|_| self.out.write_all(&[val]));
    }

    pub fn flush(&mut self) {
        let _ = self.out.flush();
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        let _ = self.out.try_finish();
    }
}

/// System memory as the board's DMA sees it, recording what it reads.
pub(super) struct RecordingMem {
    pub inner: Arc<dyn BusDevice>,
    pub rec: RecHandle,
}

impl BusDevice for RecordingMem {
    fn read8(&self, addr: u64) -> BusRead8 {
        let r = self.inner.read8(addr);
        if r.is_ok() {
            self.rec.lock().dma_byte(addr, r.data);
        }
        r
    }
    fn write8(&self, addr: u64, val: u8) -> u32 {
        self.inner.write8(addr, val)
    }
    fn read_block(&self, addr: u64, buf: &mut [u64]) -> u32 {
        let st = self.inner.read_block(addr, buf);
        if st == BUS_OK {
            self.rec.lock().dma_block(addr, buf);
        }
        st
    }
    fn write_block(&self, addr: u64, buf: &[u64]) -> u32 {
        self.inner.write_block(addr, buf)
    }
}

/// Parse a whole recording.
pub(super) fn load(path: &str) -> std::io::Result<Vec<Rec>> {
    let bad = |what: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, what.to_string());
    let mut data = Vec::new();
    // A recording the emulator was killed in the middle of ends without the
    // gzip trailer: keep what decoded.
    match std::io::BufReader::new(flate2::read::MultiGzDecoder::new(std::fs::File::open(path)?)).read_to_end(&mut data) {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof && !data.is_empty() => {}
        Err(e) => return Err(e),
    }
    if data.len() < 12 || &data[..8] != MAGIC {
        return Err(bad("not a VPro recording"));
    }
    if u32::from_le_bytes(data[8..12].try_into().unwrap()) != VERSION {
        return Err(bad("unsupported recording version"));
    }
    let mut out = Vec::new();
    let mut i = 12;
    let u32_at = |i: usize| u32::from_le_bytes(data[i..i + 4].try_into().unwrap());
    let u64_at = |i: usize| u64::from_le_bytes(data[i..i + 8].try_into().unwrap());
    while i < data.len() {
        let need = match data[i] {
            b'W' | b'R' => 14,
            b'b' => 10,
            b'B' => {
                if i + 13 > data.len() {
                    break;
                }
                13 + 8 * u32_at(i + 9) as usize
            }
            t => return Err(bad(&format!("bad record tag {t:#x} at byte {i}"))),
        };
        if i + need > data.len() {
            // Cut off mid-record (the emulator was killed): keep what is complete.
            break;
        }
        out.push(match data[i] {
            b'W' => Rec::Write { size: data[i + 1], off: u32_at(i + 2), val: u64_at(i + 6) },
            b'R' => Rec::Read { size: data[i + 1], off: u32_at(i + 2), val: u64_at(i + 6) },
            b'b' => Rec::DmaByte { addr: u64_at(i + 1), val: data[i + 9] },
            _ => Rec::DmaBlock { addr: u64_at(i + 1), words: (0..u32_at(i + 9) as usize).map(|k| u64_at(i + 13 + 8 * k)).collect() },
        });
        i += need;
    }
    Ok(out)
}
