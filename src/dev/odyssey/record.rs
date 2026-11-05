//! A replayable recording of what reaches the VPro board, for regression
//! tests (`golden/replay.rs`): every bus store the processors make, the
//! loads that are not plain repeats, and the guest memory the board's DMA
//! reads. Replaying the stores from power-on through a fresh board, with the
//! DMA reads answered from the recording, must reproduce the planes the live
//! run drew; the replay checks them against hash checkpoints kept beside the
//! recording.
//!
//! Started with `IRIS_ODSY_REC=<file>` when the board is built (power-on is
//! what a replay starts from) or from the monitor (`vpro rec <file>`, `vpro
//! rec mark`, `vpro rec off`; debug.rs). The recording is gzip-compressed
//! whatever the name. It stays out of git, as IMPACT's `.rec` files do; the
//! checkpoints are what the tree keeps (`src/dev/odyssey/testdata/replay/`).
//!
//! File: `ODSYREC` + NUL + u32 version, then records, little-endian:
//!
//! | tag | fields                         | what                                  |
//! |-----|--------------------------------|---------------------------------------|
//! | `W` | size u8, off u32, val u64      | bus store, `off` within the aperture  |
//! | `R` | size u8, off u32, val u64      | bus load and what it answered; an access identical to the one before it is not kept |
//! | `B` | addr u64, n u32, n x u64       | DMA read of n qwords from system memory |
//! | `b` | addr u64, val u8               | DMA read of one byte                  |
//! | `H` | blake3 [u8; 32]                | mark: the plane hash (`debug::memory_hash`) with the board idle (version 2) |
//!
//! Loads the board answered busy are not kept: the processor retries them.
//! Every recording opens with a mark of the state it starts from (at
//! power-on, the board after `reset`), so a replay can tell a power-on
//! recording from one started mid-session; `vpro rec mark` adds one, `vpro
//! rec off` closes with one.
//!
//! The recorder sits in a `RecSlot` the bus path (`BoardLocks`, under the
//! processors' lock), the DMA path (`RecordingMem`, on the board's thread)
//! and the monitor share. Lock order: the processors' lock, then the slot;
//! the DMA path takes the slot alone.

use std::io::Write;
#[cfg(test)]
use std::io::Read;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::traits::{BusDevice, BusRead8, BUS_OK};

const MAGIC: &[u8; 8] = b"ODSYREC\0";
/// Version 2 added the `H` mark; a version 1 file is the same records
/// without it.
const VERSION: u32 = 2;
#[cfg(test)]
const VERSION_MIN: u32 = 1;

#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Rec {
    Write { size: u8, off: u32, val: u64 },
    Read { size: u8, off: u32, val: u64 },
    DmaBlock { addr: u64, words: Vec<u64> },
    DmaByte { addr: u64, val: u8 },
    Mark([u8; 32]),
}

pub(super) struct Recorder {
    out: flate2::write::GzEncoder<std::io::BufWriter<std::fs::File>>,
    pub path: String,
    pub records: u64,
    pub marks: u32,
    last_read: Option<(u8, u32, u64)>,
}

/// The recorder's place, shared by the bus path, the DMA path and the
/// monitor, which starts and stops it. `active` is the cheap check the hot
/// paths make before taking the lock.
pub(super) struct RecSlot {
    on: AtomicBool,
    rec: Mutex<Option<Recorder>>,
}

pub(super) type RecHandle = Arc<RecSlot>;

impl RecSlot {
    pub fn new(rec: Option<Recorder>) -> RecHandle {
        Arc::new(RecSlot { on: AtomicBool::new(rec.is_some()), rec: Mutex::new(rec) })
    }

    /// Whether a recorder is in the slot (relaxed: a path that sees false
    /// around a start or stop skips one access, see `set`).
    #[inline]
    pub fn active(&self) -> bool {
        self.on.load(Ordering::Relaxed)
    }

    /// The recorder, if one runs. Hold it across an access and its record,
    /// as the bus path does, when the record's order matters.
    pub fn lock(&self) -> parking_lot::MutexGuard<'_, Option<Recorder>> {
        self.rec.lock()
    }

    /// Put a recorder in (or take it out with None); returns the one that
    /// was there, which finishes its file when dropped. Call with the
    /// processors held off, so no board store lands between the state a
    /// start or stop marks and the first or last record.
    pub fn set(&self, rec: Option<Recorder>) -> Option<Recorder> {
        let mut slot = self.rec.lock();
        self.on.store(rec.is_some(), Ordering::SeqCst);
        std::mem::replace(&mut *slot, rec)
    }
}

impl Recorder {
    pub fn create(path: &str) -> std::io::Result<Recorder> {
        let file = std::io::BufWriter::with_capacity(1 << 20, std::fs::File::create(path)?);
        let mut out = flate2::write::GzEncoder::new(file, flate2::Compression::fast());
        out.write_all(MAGIC)?;
        out.write_all(&VERSION.to_le_bytes())?;
        Ok(Recorder { out, path: path.to_string(), records: 0, marks: 0, last_read: None })
    }

    /// A mark: the plane hash of the board, idle, now. Flushed, so a
    /// recording cut off later still carries it.
    pub fn mark(&mut self, hash: [u8; 32]) {
        self.records += 1;
        self.marks += 1;
        self.last_read = None;
        let _ = self.out.write_all(b"H").and_then(|_| self.out.write_all(&hash));
        self.flush();
    }

    /// Sync-flush now and then, so a recording cut off by a kill still reads.
    fn tick(&mut self) {
        if self.records.is_multiple_of(4096) {
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
        if r.is_ok() && self.rec.active() {
            if let Some(rec) = self.rec.lock().as_mut() {
                rec.dma_byte(addr, r.data);
            }
        }
        r
    }
    fn write8(&self, addr: u64, val: u8) -> u32 {
        self.inner.write8(addr, val)
    }
    fn read_block(&self, addr: u64, buf: &mut [u64]) -> u32 {
        let st = self.inner.read_block(addr, buf);
        if st == BUS_OK && self.rec.active() {
            if let Some(rec) = self.rec.lock().as_mut() {
                rec.dma_block(addr, buf);
            }
        }
        st
    }
    fn write_block(&self, addr: u64, buf: &[u64]) -> u32 {
        self.inner.write_block(addr, buf)
    }
}

#[cfg(test)]
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
    if !(VERSION_MIN..=VERSION).contains(&u32::from_le_bytes(data[8..12].try_into().unwrap())) {
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
            b'H' => 33,
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
            b'H' => Rec::Mark(data[i + 1..i + 33].try_into().unwrap()),
            _ => Rec::DmaBlock { addr: u64_at(i + 1), words: (0..u32_at(i + 9) as usize).map(|k| u64_at(i + 13 + 8 * k)).collect() },
        });
        i += need;
    }
    Ok(out)
}
