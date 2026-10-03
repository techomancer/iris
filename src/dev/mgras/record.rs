//! Replayable recording of everything that changes the board, for regression
//! tests: every bus access, the system memory the board's DMA reads and
//! writes, and board-internal events that change drawing (a stale block
//! flushed by the display thread). Replaying a recording from power-on through
//! the device's bus interface must reproduce the board state the live run
//! reached, checked against hash checkpoints taken while recording.
//!
//! Started from `IRIS_MGRAS_REC=<file>` (from power-on, which is what replay
//! needs) or the monitor (`mgras rec <file>`, `mgras rec mark`, `mgras rec
//! off`). Replay: `MGRAS_REPLAY=<file> cargo test --release mgras_replay_file
//! -- --nocapture` (`MGRAS_REPLAY_OUT=<file.png>` saves the final screen,
//! `MGRAS_REPLAY_TRACE=<file>` writes the annotated trace of the replay).
//!
//! File: `MGRASREC` + u32 version, then records, little-endian:
//!
//! | tag | fields                         | what                                |
//! |-----|--------------------------------|-------------------------------------|
//! | `W` | bits u8, off u32, val u64      | bus write, `off` within the board   |
//! | `R` | bits u8, off u32, val u64      | bus read and the value it returned  |
//! | `M` | bits u8, addr u32, val u32     | DMA read of system memory           |
//! | `m` | bits u8, addr u32, val u32     | DMA write to system memory          |
//! | `F` |                                | display thread flushed a stale block|
//! | `T` |                                | display frame tick (stale-block check)|
//! | `H` | blake3 [u8; 32]                | checkpoint: hash of `state_bytes`   |
//! | `C` |                                | host GL composite (not replayable)  |
//!
//! A path ending in `.txt` records the same records as text, one per line
//! (`W 64 70080 1000400000001`, `M 8 8001000 ab`, `T`, `H <hex>`; numbers
//! in hex, `#` starts a comment). Text recordings are for small traces kept
//! in `testdata/`; `load` reads either form.

use std::io::{BufWriter, Read, Write};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::traits::{BusDevice, BusRead32, BusRead8, BUS_OK};

const MAGIC: &[u8; 8] = b"MGRASREC";
const VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rec {
    Write { bits: u8, off: u32, val: u64 },
    Read { bits: u8, off: u32, val: u64 },
    MemRead { bits: u8, addr: u32, val: u32 },
    MemWrite { bits: u8, addr: u32, val: u32 },
    FlushStale,
    Tick,
    Hash([u8; 32]),
    Composite,
}

pub struct Recorder {
    out: BufWriter<std::fs::File>,
    text: bool,
    pub path: String,
    pub records: u64,
    /// Records since the last checkpoint.
    since_hash: u64,
}

/// Records between automatic checkpoints, so a divergence is caught near
/// where it happens, not only at the next scripted mark.
const CHECKPOINT_EVERY: u64 = 250_000;

/// Shared so the DMA path can record through `RecordingMem` while the board
/// is borrowed.
pub type RecHandle = Arc<Mutex<Recorder>>;

impl Recorder {
    pub fn create(path: &str) -> std::io::Result<RecHandle> {
        let mut out = BufWriter::with_capacity(1 << 20, std::fs::File::create(path)?);
        let text = path.ends_with(".txt");
        if text {
            writeln!(out, "# MGRAS recording (record.rs text form)")?;
        } else {
            out.write_all(MAGIC)?;
            out.write_all(&VERSION.to_le_bytes())?;
        }
        Ok(Arc::new(Mutex::new(Recorder { out, text, path: path.to_string(), records: 0, since_hash: 0 })))
    }

    pub fn put(&mut self, r: Rec) {
        self.records += 1;
        self.since_hash = if matches!(r, Rec::Hash(_)) { 0 } else { self.since_hash + 1 };
        let o = &mut self.out;
        if self.text {
            let _ = writeln!(o, "{}", text_line(r));
            return;
        }
        let _ = match r {
            Rec::Write { bits, off, val } => put_access(o, b'W', bits, off, val),
            Rec::Read { bits, off, val } => put_access(o, b'R', bits, off, val),
            Rec::MemRead { bits, addr, val } => put_mem(o, b'M', bits, addr, val),
            Rec::MemWrite { bits, addr, val } => put_mem(o, b'm', bits, addr, val),
            Rec::FlushStale => o.write_all(b"F"),
            Rec::Tick => o.write_all(b"T"),
            Rec::Hash(h) => o.write_all(b"H").and_then(|_| o.write_all(&h)),
            Rec::Composite => o.write_all(b"C"),
        };
    }

    /// Time for an automatic checkpoint.
    pub fn checkpoint_due(&self) -> bool {
        self.since_hash >= CHECKPOINT_EVERY
    }

    pub fn flush(&mut self) {
        let _ = self.out.flush();
    }
}

fn put_access(o: &mut impl Write, tag: u8, bits: u8, off: u32, val: u64) -> std::io::Result<()> {
    o.write_all(&[tag, bits])?;
    o.write_all(&off.to_le_bytes())?;
    o.write_all(&val.to_le_bytes())
}

fn put_mem(o: &mut impl Write, tag: u8, bits: u8, addr: u32, val: u32) -> std::io::Result<()> {
    o.write_all(&[tag, bits])?;
    o.write_all(&addr.to_le_bytes())?;
    o.write_all(&val.to_le_bytes())
}

/// One record in the text form.
fn text_line(r: Rec) -> String {
    match r {
        Rec::Write { bits, off, val } => format!("W {bits} {off:x} {val:x}"),
        Rec::Read { bits, off, val } => format!("R {bits} {off:x} {val:x}"),
        Rec::MemRead { bits, addr, val } => format!("M {bits} {addr:x} {val:x}"),
        Rec::MemWrite { bits, addr, val } => format!("m {bits} {addr:x} {val:x}"),
        Rec::FlushStale => "F".into(),
        Rec::Tick => "T".into(),
        Rec::Hash(h) => format!("H {}", blake3::Hash::from(h).to_hex()),
        Rec::Composite => "C".into(),
    }
}

/// Parse a text recording.
pub fn load_text(text: &str) -> Result<Vec<Rec>, String> {
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        let bad = || format!("line {}: cannot parse {line:?}", n + 1);
        let hex = |i: usize| f.get(i).and_then(|s| u64::from_str_radix(s, 16).ok()).ok_or_else(bad);
        let bits = || f.get(1).and_then(|s| s.parse::<u8>().ok()).ok_or_else(bad);
        out.push(match f[0] {
            "W" => Rec::Write { bits: bits()?, off: hex(2)? as u32, val: hex(3)? },
            "R" => Rec::Read { bits: bits()?, off: hex(2)? as u32, val: hex(3)? },
            "M" => Rec::MemRead { bits: bits()?, addr: hex(2)? as u32, val: hex(3)? as u32 },
            "m" => Rec::MemWrite { bits: bits()?, addr: hex(2)? as u32, val: hex(3)? as u32 },
            "F" => Rec::FlushStale,
            "T" => Rec::Tick,
            "C" => Rec::Composite,
            "H" => {
                let h = f.get(1).and_then(|s| blake3::Hash::from_hex(s).ok()).ok_or_else(bad)?;
                Rec::Hash(*h.as_bytes())
            }
            _ => return Err(bad()),
        });
    }
    Ok(out)
}

/// Parse a whole recording, binary or text.
pub fn load(path: &str) -> std::io::Result<Vec<Rec>> {
    let mut data = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut data)?;
    let bad = |what: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, what.to_string());
    if data.len() < 12 || &data[..8] != MAGIC {
        let text = std::str::from_utf8(&data).map_err(|_| bad("not an MGRAS recording"))?;
        return load_text(text).map_err(|e| bad(&e));
    }
    if u32::from_le_bytes(data[8..12].try_into().unwrap()) != VERSION {
        return Err(bad("unsupported recording version"));
    }
    let mut out = Vec::with_capacity(data.len() / 14);
    let mut i = 12;
    let u32_at = |i: usize| u32::from_le_bytes(data[i..i + 4].try_into().unwrap());
    let u64_at = |i: usize| u64::from_le_bytes(data[i..i + 8].try_into().unwrap());
    while i < data.len() {
        let need = match data[i] {
            b'W' | b'R' => 14,
            b'M' | b'm' => 10,
            b'H' => 33,
            b'F' | b'T' | b'C' => 1,
            t => return Err(bad(&format!("bad record tag {t:#x} at byte {i}"))),
        };
        if i + need > data.len() {
            // A recording cut off mid-record (the emulator was killed): keep
            // what is complete.
            break;
        }
        let r = match data[i] {
            b'W' => Rec::Write { bits: data[i + 1], off: u32_at(i + 2), val: u64_at(i + 6) },
            b'R' => Rec::Read { bits: data[i + 1], off: u32_at(i + 2), val: u64_at(i + 6) },
            b'M' => Rec::MemRead { bits: data[i + 1], addr: u32_at(i + 2), val: u32_at(i + 6) },
            b'm' => Rec::MemWrite { bits: data[i + 1], addr: u32_at(i + 2), val: u32_at(i + 6) },
            b'H' => Rec::Hash(data[i + 1..i + 33].try_into().unwrap()),
            b'F' => Rec::FlushStale,
            b'T' => Rec::Tick,
            _ => Rec::Composite,
        };
        out.push(r);
        i += need;
    }
    Ok(out)
}

/// System memory as the board's DMA sees it, recording every access.
pub struct RecordingMem {
    pub inner: Arc<dyn BusDevice>,
    pub rec: RecHandle,
}

impl BusDevice for RecordingMem {
    fn read8(&self, addr: u32) -> BusRead8 {
        let r = self.inner.read8(addr);
        self.rec.lock().put(Rec::MemRead { bits: 8, addr, val: if r.is_ok() { r.data as u32 } else { 0 } });
        if r.is_ok() { r } else { BusRead8::ok(0) }
    }
    fn read32(&self, addr: u32) -> BusRead32 {
        let r = self.inner.read32(addr);
        self.rec.lock().put(Rec::MemRead { bits: 32, addr, val: if r.is_ok() { r.data } else { 0 } });
        if r.is_ok() { r } else { BusRead32::ok(0) }
    }
    fn write8(&self, addr: u32, val: u8) -> u32 {
        self.rec.lock().put(Rec::MemWrite { bits: 8, addr, val: val as u32 });
        self.inner.write8(addr, val)
    }
    fn write32(&self, addr: u32, val: u32) -> u32 {
        self.rec.lock().put(Rec::MemWrite { bits: 32, addr, val });
        self.inner.write32(addr, val)
    }
}

/// System memory for replay: answers the board's DMA reads with the recorded
/// values, in order, and checks each access hits the recorded address.
pub struct ReplayMem {
    reads: Vec<(u8, u32, u32)>,
    next: Mutex<usize>,
    pub mismatches: Mutex<u64>,
}

impl ReplayMem {
    pub fn new(recs: &[Rec]) -> Self {
        let reads = recs
            .iter()
            .filter_map(|r| match *r {
                Rec::MemRead { bits, addr, val } => Some((bits, addr, val)),
                _ => None,
            })
            .collect();
        ReplayMem { reads, next: Mutex::new(0), mismatches: Mutex::new(0) }
    }

    fn take(&self, bits: u8, addr: u32) -> u32 {
        let mut n = self.next.lock();
        let Some(&(b, a, v)) = self.reads.get(*n) else {
            *self.mismatches.lock() += 1;
            return 0;
        };
        *n += 1;
        if (b, a) != (bits, addr) {
            *self.mismatches.lock() += 1;
        }
        v
    }

    pub fn consumed(&self) -> (usize, usize) {
        (*self.next.lock(), self.reads.len())
    }
}

impl BusDevice for ReplayMem {
    fn read8(&self, addr: u32) -> BusRead8 { BusRead8::ok(self.take(8, addr) as u8) }
    fn read32(&self, addr: u32) -> BusRead32 { BusRead32::ok(self.take(32, addr)) }
    fn write8(&self, _addr: u32, _val: u8) -> u32 { BUS_OK }
    fn write32(&self, _addr: u32, _val: u32) -> u32 { BUS_OK }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::config::GraphicsBoard;
    use crate::dev::mgras::{Mgras, MGRAS_SLOT_GFX_BASE};
    use crate::traits::BUS_BUSY;

    /// A High IMPACT board with its engine threads running (no display).
    pub fn live_board() -> Arc<Mgras> {
        let hb = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let m = Mgras::new(GraphicsBoard::HighImpact, crate::dev::ioc::Ioc::new(false), hb.clone(), hb);
        m.start_engines();
        m
    }

    /// Store, retrying on bus back-pressure like the CPU does.
    pub fn write(m: &Mgras, bits: u8, off: u32, val: u64) {
        let a = MGRAS_SLOT_GFX_BASE + off;
        loop {
            let st = match bits {
                8 => m.write8(a, val as u8),
                16 => m.write16(a, val as u16),
                32 => m.write32(a, val as u32),
                _ => m.write64(a, val),
            };
            if st != BUS_BUSY {
                return;
            }
            std::hint::spin_loop();
        }
    }

    /// Load, retrying while the board answers busy.
    pub fn read(m: &Mgras, bits: u8, off: u32) -> u64 {
        let a = MGRAS_SLOT_GFX_BASE + off;
        loop {
            let (st, v) = match bits {
                8 => { let r = m.read8(a); (r.status, r.data as u64) }
                16 => { let r = m.read16(a); (r.status, r.data as u64) }
                32 => { let r = m.read32(a); (r.status, r.data as u64) }
                _ => { let r = m.read64(a); (r.status, r.data) }
            };
            if st != BUS_BUSY {
                return v;
            }
            std::hint::spin_loop();
        }
    }

    /// What a replay saw.
    #[derive(Debug, Default)]
    pub struct Report {
        pub records: usize,
        pub checked: usize,
        /// Record indices of failed checkpoints.
        pub failed: Vec<usize>,
        pub dma_used: usize,
        pub dma_total: usize,
        pub dma_mismatched: u64,
        pub reads_differ: u64,
        pub differ_at: std::collections::BTreeMap<u32, u32>,
    }

    impl Report {
        pub fn diverged(&self) -> bool {
            !self.failed.is_empty() || self.dma_mismatched != 0 || self.dma_used != self.dma_total
        }
    }

    /// Feed `recs` through `m`'s bus interface, serving its DMA reads from
    /// the recorded values; checkpoints are checked when `check_hashes` (a
    /// recording that starts mid-session cannot match a fresh board's).
    /// Returns once the board has run everything.
    pub fn replay_recs(m: &Mgras, recs: &[Rec], check_hashes: bool) -> Result<Report, String> {
        replay_recs_with(m, recs, check_hashes, &mut |_| {})
    }

    /// `replay_recs`, calling `at_checkpoint` at every checkpoint record
    /// (whether or not hashes are checked) with the board idle.
    pub fn replay_recs_with(m: &Mgras, recs: &[Rec], check_hashes: bool, at_checkpoint: &mut dyn FnMut(&Mgras)) -> Result<Report, String> {
        let mem = Arc::new(ReplayMem::new(recs));
        m.set_phys(mem.clone());
        let mut rep = Report { records: recs.len(), ..Default::default() };
        for (i, r) in recs.iter().enumerate() {
            match *r {
                Rec::Write { bits, off, val } => write(m, bits, off, val),
                Rec::Read { bits, off, val } => {
                    // Compare at the access width (early recordings kept
                    // the whole register value for narrow reads).
                    let mask = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
                    if read(m, bits, off) & mask != val & mask {
                        rep.reads_differ += 1;
                        *rep.differ_at.entry(off).or_default() += 1;
                    }
                }
                // Recordings from before block type 0 was understood as a
                // fill carry these; the board no longer needs them.
                Rec::FlushStale | Rec::Tick => {}
                Rec::Hash(h) => {
                    if check_hashes {
                        rep.checked += 1;
                        if m.state_hash() != h {
                            rep.failed.push(i);
                        }
                    }
                    at_checkpoint(m);
                }
                Rec::Composite => return Err(format!("record {i}: host GL composite cannot be replayed")),
                Rec::MemRead { .. } | Rec::MemWrite { .. } => {}
            }
        }
        m.state_hash(); // waits for the board to finish
        (rep.dma_used, rep.dma_total) = mem.consumed();
        rep.dma_mismatched = *mem.mismatches.lock();
        Ok(rep)
    }

    /// Replay a recording file through a fresh board and check every
    /// checkpoint.
    pub fn replay(path: &str, png: Option<&str>) -> Result<(), String> {
        let recs = load(path).map_err(|e| format!("{path}: {e}"))?;
        let m = live_board();
        // MGRAS_REPLAY_TRACE=<file>: the annotated trace of the replay
        // (`mgras trace`), all categories.
        if let Ok(t) = std::env::var("MGRAS_REPLAY_TRACE") {
            m.start_trace(&t, crate::dev::mgras::debug::TRACE_ALL).map_err(|e| format!("{t}: {e}"))?;
        }
        let rep = replay_recs(&m, &recs, true)?;
        m.trace_flush(true);
        if let Some(p) = png {
            m.save_shot(p)?;
        }
        m.stop_engines();
        eprintln!(
            "{path}: {} records, {} checkpoints, {} failed {:?}, DMA reads {}/{} ({} mismatched), {} bus reads differ",
            rep.records, rep.checked, rep.failed.len(), &rep.failed[..rep.failed.len().min(8)],
            rep.dma_used, rep.dma_total, rep.dma_mismatched, rep.reads_differ
        );
        if !rep.differ_at.is_empty() {
            eprintln!("  bus reads that differ, by offset: {:x?}", rep.differ_at);
        }
        if rep.checked == 0 {
            return Err("no checkpoints in recording".into());
        }
        if rep.diverged() {
            return Err(format!("replay diverged: {} of {} checkpoints failed", rep.failed.len(), rep.checked));
        }
        Ok(())
    }

    #[test]
    fn mgras_replay_file() {
        let Ok(path) = std::env::var("MGRAS_REPLAY") else { return };
        let png = std::env::var("MGRAS_REPLAY_OUT").ok();
        replay(&path, png.as_deref()).unwrap();
    }

    const ALL: [Rec; 8] = [
        Rec::Write { bits: 64, off: 0x70080, val: 0x0010_0004_0000_0001 },
        Rec::Read { bits: 32, off: 0x70000, val: 0x53 },
        Rec::MemRead { bits: 8, addr: 0x0800_1000, val: 0xAB },
        Rec::MemWrite { bits: 32, addr: 0x0800_2000, val: 0xDEAD_BEEF },
        Rec::FlushStale,
        Rec::Tick,
        Rec::Hash([7; 32]),
        Rec::Composite,
    ];

    fn round_trip(ext: &str) {
        let path = std::env::temp_dir().join(format!("mgras_rec_{}{ext}", std::process::id()));
        let path = path.to_str().unwrap().to_string();
        {
            let h = Recorder::create(&path).unwrap();
            let mut r = h.lock();
            for x in ALL {
                r.put(x);
            }
            r.flush();
        }
        assert_eq!(load(&path).unwrap(), ALL);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn record_round_trip_binary() {
        round_trip(".rec");
    }

    #[test]
    fn record_round_trip_text() {
        round_trip(".txt");
    }
}
