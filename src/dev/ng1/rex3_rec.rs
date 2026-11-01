//! Replayable recording of everything that changes the REX3 board, for
//! regression tests and for taking a live bug to a unit test: every bus
//! access the CPU and the MC DMA engine make, with the state the board
//! started from and hash checkpoints. Replaying a recording through the
//! board's bus interface must reproduce every checkpoint. Same idea as
//! IMPACT's `mgras/record.rs`, and the same monitor syntax.
//!
//! Started from the monitor (`rex rec <file>`, `rex rec mark`, `rex rec off`)
//! at any time: the recording opens with the board's state (`S`), so it does
//! not have to start at power-on. `IRIS_REX3_REC=<file>` starts one when the
//! board starts. Replay:
//! `REX3_REPLAY=<file> cargo test --release rex3_replay_file -- --nocapture`
//! (`REX3_REPLAY_OUT=<dir>` saves the final framebuffers there,
//! `REX3_REPLAY_TRACE=<file>` writes the annotated trace of the replay,
//! `REX3_REPLAY_ENGINE=interp|jit` picks the draw path in `rex-jit` builds,
//! `REX3_REPLAY_NOCHECK=1` skips the checkpoints, `REX3_REPLAY_LIMIT=<n>`
//! replays only the first n records, `REX3_REPLAY_STATE=<file>` writes the
//! final register state as TOML).
//!
//! File: `REX3REC\0` + u32 version, then records, little-endian. Offsets are
//! within the REX3's 8 KB window, GO bit (0x800) included:
//!
//! | tag | fields                          | what                                  |
//! |-----|---------------------------------|---------------------------------------|
//! | `W` | bits u8, off u16, val u32/u64   | bus write (val u64 only for 64 bits)  |
//! | `R` | bits u8, off u16, val u32/u64   | bus read and the value it returned    |
//! | `D` | off u16, val u64                | `dma_read64` (primed HOSTRW read)     |
//! | `B` | off u16, n u32, n x u64         | `dma_write64_bulk`                    |
//! | `b` | off u16, n u32, n x u64         | `dma_read64_bulk` and what it returned|
//! | `H` | blake3 [u8; 32]                 | checkpoint: `Rex3::rec_state_hash`    |
//! | `S` | len u32, len bytes              | board state the recording starts from |
//!
//! Plain STATUS polls are not recorded (they change nothing that is hashed),
//! nor are accesses the board answered busy (the CPU retries them).
//!
//! A path ending in `.txt` records the same records as text, one per line
//! (`W 32 1154 3ff`, `B a30 2 1122 3344`, `H <hex>`, `S <hex>`; numbers in
//! hex, `#` starts a comment). Text recordings are for small traces kept in
//! `testdata/`; one without an `S` line replays on a power-on board.
//! `load` reads either form.
//!
//! The recording lock is held across the access and its record, so records
//! come out in the order the GFIFO received them even with the CPU and the
//! DMA engine both pushing.

use std::io::Read;
use std::sync::atomic::Ordering;

use crate::trace_writer::TraceWriter;
use crate::traits::Saveable;

use super::{hex_u32, Rex3};

const MAGIC: &[u8; 8] = b"REX3REC\0";
const VERSION: u32 = 1;

/// Records between automatic checkpoints. A checkpoint drains the GFIFO and
/// hashes 16 MB of VRAM, so not too often.
const CHECKPOINT_EVERY: u64 = 1_000_000;

/// `Rex3::tap` bit: a recording is running (the trace categories use the
/// low bits).
pub const TAP_REC: u32 = 1 << 31;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Rec {
    Write { bits: u8, off: u32, val: u64 },
    Read { bits: u8, off: u32, val: u64 },
    DmaRead { off: u32, val: u64 },
    BulkWrite { off: u32, vals: Vec<u64> },
    BulkRead { off: u32, vals: Vec<u64> },
    Hash([u8; 32]),
    State(Vec<u8>),
}

pub struct Recorder {
    out: TraceWriter,
    text: bool,
    pub records: u64,
    /// Records since the last checkpoint.
    since_hash: u64,
}

impl Recorder {
    pub fn create(path: &str) -> std::io::Result<Self> {
        let mut out = TraceWriter::create(path)?;
        let text = path.ends_with(".txt");
        if text {
            out.line(format_args!("# REX3 recording (rex3_rec.rs text form)"));
        } else {
            out.put(MAGIC);
            out.put(&VERSION.to_le_bytes());
        }
        Ok(Recorder { out, text, records: 0, since_hash: 0 })
    }

    pub fn path(&self) -> &str {
        &self.out.path
    }

    fn count(&mut self) {
        self.records += 1;
        self.since_hash += 1;
    }

    /// A bus write (`b'W'`) or read (`b'R'`).
    #[inline]
    pub fn access(&mut self, tag: u8, bits: u8, off: u32, val: u64) {
        self.count();
        if self.text {
            self.out.line(format_args!("{} {bits} {off:x} {val:x}", tag as char));
            return;
        }
        let o = (off as u16).to_le_bytes();
        if bits > 32 {
            let v = val.to_le_bytes();
            self.out.put(&[tag, bits, o[0], o[1], v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]]);
        } else {
            let v = (val as u32).to_le_bytes();
            self.out.put(&[tag, bits, o[0], o[1], v[0], v[1], v[2], v[3]]);
        }
    }

    pub fn dma_read(&mut self, off: u32, val: u64) {
        self.count();
        if self.text {
            self.out.line(format_args!("D {off:x} {val:x}"));
            return;
        }
        self.out.put(&[b'D']);
        self.out.put(&(off as u16).to_le_bytes());
        self.out.put(&val.to_le_bytes());
    }

    /// A bulk DMA write (`b'B'`) or read (`b'b'`).
    pub fn bulk(&mut self, tag: u8, off: u32, vals: &[u64]) {
        self.count();
        if self.text {
            let words: Vec<String> = vals.iter().map(|v| format!("{v:x}")).collect();
            self.out.line(format_args!("{} {off:x} {:x} {}", tag as char, vals.len(), words.join(" ")));
            return;
        }
        self.out.put(&[tag]);
        self.out.put(&(off as u16).to_le_bytes());
        self.out.put(&(vals.len() as u32).to_le_bytes());
        for v in vals {
            self.out.put(&v.to_le_bytes());
        }
    }

    pub fn hash(&mut self, h: [u8; 32]) {
        self.records += 1;
        self.since_hash = 0;
        if self.text {
            self.out.line(format_args!("H {}", blake3::Hash::from(h).to_hex()));
            return;
        }
        self.out.put(b"H");
        self.out.put(&h);
    }

    pub fn state(&mut self, bytes: &[u8]) {
        self.records += 1;
        if self.text {
            let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
            self.out.line(format_args!("S {hex}"));
            return;
        }
        self.out.put(b"S");
        self.out.put(&(bytes.len() as u32).to_le_bytes());
        self.out.put(bytes);
    }

    pub fn put(&mut self, r: &Rec) {
        match r {
            Rec::Write { bits, off, val } => self.access(b'W', *bits, *off, *val),
            Rec::Read { bits, off, val } => self.access(b'R', *bits, *off, *val),
            Rec::DmaRead { off, val } => self.dma_read(*off, *val),
            Rec::BulkWrite { off, vals } => self.bulk(b'B', *off, vals),
            Rec::BulkRead { off, vals } => self.bulk(b'b', *off, vals),
            Rec::Hash(h) => self.hash(*h),
            Rec::State(s) => self.state(s),
        }
    }

    /// Time for an automatic checkpoint.
    #[inline]
    pub fn checkpoint_due(&self) -> bool {
        self.since_hash >= CHECKPOINT_EVERY
    }

    /// Make everything recorded so far readable in the file.
    pub fn sync(&mut self) {
        self.out.sync();
    }

    pub fn finish(self) -> std::io::Result<u64> {
        self.out.finish()
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
        let words = || -> Result<Vec<u64>, String> {
            let n = hex(2)? as usize;
            let v = (0..n).map(|i| hex(3 + i)).collect::<Result<Vec<_>, _>>()?;
            if f.len() != 3 + n {
                return Err(bad());
            }
            Ok(v)
        };
        out.push(match f[0] {
            "W" => Rec::Write { bits: bits()?, off: hex(2)? as u32, val: hex(3)? },
            "R" => Rec::Read { bits: bits()?, off: hex(2)? as u32, val: hex(3)? },
            "D" => Rec::DmaRead { off: hex(1)? as u32, val: hex(2)? },
            "B" => Rec::BulkWrite { off: hex(1)? as u32, vals: words()? },
            "b" => Rec::BulkRead { off: hex(1)? as u32, vals: words()? },
            "H" => {
                let h = f.get(1).and_then(|s| blake3::Hash::from_hex(s).ok()).ok_or_else(bad)?;
                Rec::Hash(*h.as_bytes())
            }
            "S" => {
                let s = f.get(1).ok_or_else(bad)?;
                if s.len() % 2 != 0 {
                    return Err(bad());
                }
                let bytes = (0..s.len() / 2)
                    .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).map_err(|_| bad()))
                    .collect::<Result<Vec<_>, _>>()?;
                Rec::State(bytes)
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
        let text = std::str::from_utf8(&data).map_err(|_| bad("not a REX3 recording"))?;
        return load_text(text).map_err(|e| bad(&e));
    }
    if u32::from_le_bytes(data[8..12].try_into().unwrap()) != VERSION {
        return Err(bad("unsupported recording version"));
    }
    let mut out = Vec::with_capacity(data.len() / 8);
    let mut i = 12;
    let u16_at = |i: usize| u16::from_le_bytes(data[i..i + 2].try_into().unwrap()) as u32;
    let u32_at = |i: usize| u32::from_le_bytes(data[i..i + 4].try_into().unwrap());
    let u64_at = |i: usize| u64::from_le_bytes(data[i..i + 8].try_into().unwrap());
    while i < data.len() {
        // Size of the record at `i`, or None when it runs past the end: a
        // recording cut off mid-record (the emulator was killed) keeps what
        // is complete.
        let avail = data.len() - i;
        let need = match data[i] {
            b'W' | b'R' => match data.get(i + 1) {
                Some(&bits) => if bits > 32 { 12 } else { 8 },
                None => break,
            },
            b'D' => 11,
            b'B' | b'b' => {
                if avail < 7 {
                    break;
                }
                7 + 8 * u32_at(i + 3) as usize
            }
            b'H' => 33,
            b'S' => {
                if avail < 5 {
                    break;
                }
                5 + u32_at(i + 1) as usize
            }
            t => return Err(bad(&format!("bad record tag {t:#x} at byte {i}"))),
        };
        if need > avail {
            break;
        }
        let words = |at: usize, n: usize| (0..n).map(|k| u64_at(at + 8 * k)).collect::<Vec<_>>();
        out.push(match data[i] {
            b'W' | b'R' => {
                let (bits, off) = (data[i + 1], u16_at(i + 2));
                let val = if bits > 32 { u64_at(i + 4) } else { u32_at(i + 4) as u64 };
                if data[i] == b'W' { Rec::Write { bits, off, val } } else { Rec::Read { bits, off, val } }
            }
            b'D' => Rec::DmaRead { off: u16_at(i + 1), val: u64_at(i + 3) },
            b'B' => Rec::BulkWrite { off: u16_at(i + 1), vals: words(i + 7, u32_at(i + 3) as usize) },
            b'b' => Rec::BulkRead { off: u16_at(i + 1), vals: words(i + 7, u32_at(i + 3) as usize) },
            b'H' => Rec::Hash(data[i + 1..i + 33].try_into().unwrap()),
            _ => Rec::State(data[i + 5..i + need].to_vec()),
        });
        i += need;
    }
    Ok(out)
}

// ── board state ─────────────────────────────────────────────────────────────

/// Run-length coding of a framebuffer: (count u32, value u32) pairs. A
/// desktop is mostly long runs, so this keeps the start state small.
fn rle_put(out: &mut Vec<u8>, words: &[u32]) {
    let mut runs: Vec<(u32, u32)> = Vec::new();
    for &w in words {
        match runs.last_mut() {
            Some((n, v)) if *v == w => *n += 1,
            _ => runs.push((1, w)),
        }
    }
    out.extend_from_slice(&(runs.len() as u32).to_le_bytes());
    for (n, v) in runs {
        out.extend_from_slice(&n.to_le_bytes());
        out.extend_from_slice(&v.to_le_bytes());
    }
}

fn rle_get(data: &[u8], at: &mut usize) -> Result<Vec<u32>, String> {
    let u32_at = |i: usize| data.get(i..i + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).ok_or("state record truncated");
    let runs = u32_at(*at)? as usize;
    *at += 4;
    let mut out = Vec::new();
    for _ in 0..runs {
        let (n, v) = (u32_at(*at)?, u32_at(*at + 4)?);
        *at += 8;
        out.extend(std::iter::repeat_n(v, n as usize));
    }
    Ok(out)
}

/// The VC2 register the refresh thread rewrites every frame (the cursor Y
/// it latches at retrace), so its value at a checkpoint depends on timing.
const VC2_VOLATILE_REG: usize = crate::dev::ng1::vc2::VC2_REG_WORKING_CURSOR_Y as usize;

impl Rex3 {
    /// Drawing-context state a snapshot leaves out because no snapshot can
    /// see it mid-primitive, but a recording started between two GOs of one
    /// primitive needs.
    fn rec_extra(&self) -> toml::Value {
        // SAFETY: called with the board idle (see callers).
        let ctx = unsafe { &*self.context.get() };
        let mut t = toml::map::Map::new();
        t.insert("pat_bit".into(), hex_u32(ctx.pat_bit as u32));
        t.insert("zpat_bit".into(), hex_u32(ctx.zpat_bit as u32));
        t.insert("mid_primitive".into(), hex_u32(ctx.mid_primitive as u32));
        t.insert("hostrw0".into(), super::hex_u64(ctx.hostrw[0]));
        toml::Value::Table(t)
    }

    /// The board state as hashed at a checkpoint: what `save_state` keeps,
    /// minus the bits that move with time rather than with bus traffic
    /// (STATUS: retrace and FIFO interrupt flags; VC2's working cursor Y),
    /// plus `rec_extra`. Call with the board idle.
    pub(crate) fn rec_state_hash(&self) -> [u8; 32] {
        let mut v = self.save_state();
        if let Some(t) = v.get_mut("config_regs").and_then(|t| t.as_table_mut()) {
            t.remove("status");
        }
        if let Some(r) = v.get_mut("vc2").and_then(|t| t.get_mut("regs")).and_then(|a| a.as_array_mut()) {
            if let Some(x) = r.get_mut(VC2_VOLATILE_REG) {
                *x = hex_u32(0);
            }
        }
        if let Some(t) = v.as_table_mut() {
            t.insert("rec_extra".into(), self.rec_extra());
        }
        let mut h = blake3::Hasher::new();
        h.update(toml::to_string(&v).unwrap_or_default().as_bytes());
        // SAFETY: idle; the display thread only reads VRAM.
        for fb in unsafe { [&*self.fb_rgb.get(), &*self.fb_aux.get()] } {
            let words: &[u32] = fb;
            // SAFETY: plain u32 words viewed as bytes.
            h.update(unsafe { std::slice::from_raw_parts(words.as_ptr() as *const u8, words.len() * 4) });
        }
        *h.finalize().as_bytes()
    }

    /// The `S` record: `save_state` plus `rec_extra` as TOML, then both
    /// framebuffers run-length coded. Call with the board idle.
    pub(crate) fn rec_state_encode(&self) -> Vec<u8> {
        let mut v = self.save_state();
        if let Some(t) = v.as_table_mut() {
            t.insert("rec_extra".into(), self.rec_extra());
        }
        let text = toml::to_string(&v).unwrap_or_default();
        let mut out = Vec::new();
        out.extend_from_slice(&(text.len() as u32).to_le_bytes());
        out.extend_from_slice(text.as_bytes());
        // SAFETY: idle; the display thread only reads VRAM.
        unsafe {
            rle_put(&mut out, &*self.fb_rgb.get());
            rle_put(&mut out, &*self.fb_aux.get());
        }
        out
    }

    /// Load an `S` record. The processor thread must not be running
    /// (`load_state`'s rule: it resets the GFIFO).
    pub(crate) fn rec_state_decode(&self, data: &[u8]) -> Result<(), String> {
        let len = data.get(..4).map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize).ok_or("state record truncated")?;
        let text = data.get(4..4 + len).ok_or("state record truncated")?;
        let text = std::str::from_utf8(text).map_err(|e| e.to_string())?;
        let v: toml::Value = toml::from_str(text).map_err(|e| e.to_string())?;
        self.load_state(&v)?;
        if let Some(x) = v.get("rec_extra") {
            // SAFETY: the processor thread is not running.
            let ctx = unsafe { &mut *self.context.get() };
            let f = |k: &str| x.get(k).and_then(super::toml_u64).unwrap_or(0);
            ctx.pat_bit = f("pat_bit") as u8;
            ctx.zpat_bit = f("zpat_bit") as u8;
            ctx.mid_primitive = f("mid_primitive") != 0;
            ctx.hostrw[0] = f("hostrw0");
        }
        let mut at = 4 + len;
        let rgb = rle_get(data, &mut at)?;
        let aux = rle_get(data, &mut at)?;
        self.restore_framebuffers_inmem(&rgb, &aux);
        Ok(())
    }

    /// Lock the recording when one is running (`tap` has `TAP_REC`). The
    /// guard is held across the access and its record.
    #[inline]
    pub(crate) fn rec_guard(&self, tap: u32) -> Option<parking_lot::MutexGuard<'_, Option<Recorder>>> {
        (tap & TAP_REC != 0).then(|| self.rec.lock())
    }

    /// After a recorded write: checkpoint when due. Called with the recording
    /// locked, so nothing can push while the board drains.
    pub(crate) fn rec_checkpoint_if_due(&self, r: &mut Recorder) {
        if r.checkpoint_due() {
            self.wait_idle();
            r.hash(self.rec_state_hash());
        }
    }

    /// Start a recording at `path`, opening with the board state (`state`;
    /// a recording without it replays on a power-on board) and a checkpoint,
    /// or stop the running one with `None` (checkpointing where it ends).
    /// Returns the stopped recording's path and record count.
    pub fn set_recording(&self, path: Option<&str>, state: bool) -> std::io::Result<Option<(String, u64)>> {
        // Raise the tap first so every producer that starts an access from
        // here on blocks on the lock below; whatever slipped in before is
        // drained by wait_idle and lands in the start state.
        if path.is_some() {
            self.tap.fetch_or(TAP_REC, Ordering::SeqCst);
        }
        let mut g = self.rec.lock();
        self.wait_idle();
        let mut stopped = None;
        if let Some(mut r) = g.take() {
            r.hash(self.rec_state_hash());
            let path = r.path().to_string();
            let n = r.records;
            r.finish()?;
            stopped = Some((path, n));
        }
        match path {
            Some(p) => {
                let mut r = match Recorder::create(p) {
                    Ok(r) => r,
                    Err(e) => {
                        self.tap.fetch_and(!TAP_REC, Ordering::SeqCst);
                        return Err(e);
                    }
                };
                if state {
                    r.state(&self.rec_state_encode());
                }
                r.hash(self.rec_state_hash());
                *g = Some(r);
            }
            None => {
                self.tap.fetch_and(!TAP_REC, Ordering::SeqCst);
            }
        }
        Ok(stopped)
    }

    /// Checkpoint the running recording now and make it readable on disk.
    /// Returns the hash and record count, or None when not recording.
    pub fn record_mark(&self) -> Option<([u8; 32], u64)> {
        let mut g = self.rec.lock();
        let r = g.as_mut()?;
        self.wait_idle();
        let h = self.rec_state_hash();
        r.hash(h);
        r.sync();
        Some((h, r.records))
    }

    /// `rex rec ...`.
    pub(super) fn cmd_rec(&self, args: &[&str], w: &mut dyn std::io::Write) -> Result<(), String> {
        let e = |r: std::io::Result<()>| r.map_err(|e| e.to_string());
        match args {
            ["off"] => match self.set_recording(None, false).map_err(|e| format!("rex rec: {e}"))? {
                Some((p, n)) => e(writeln!(w, "recording stopped: {p} ({n} records)")),
                None => e(writeln!(w, "no recording running")),
            },
            ["mark"] => {
                let (h, n) = self.record_mark().ok_or("rex rec: no recording running")?;
                e(writeln!(w, "checkpoint {} at record {n}", blake3::Hash::from(h).to_hex()))
            }
            [path] => {
                self.set_recording(Some(path), true).map_err(|e| format!("rex rec: {e}"))?;
                e(writeln!(w, "recording to {path}"))
            }
            _ => Err("Usage: rex rec <file>[.txt] | rex rec mark | rex rec off".into()),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::dev::ng1::rex3::{REX3_BASE, REX3_SIZE};
    use crate::traits::{BusDevice, Device, BUS_BUSY};

    /// A REX3 with `state` loaded (power-on otherwise) and its threads
    /// running. Built on a big stack (see rex3_tests' `make_rex3`) and
    /// leaked, as the processor thread holds a 'static reference.
    pub fn board(state: Option<&[u8]>, jit: bool) -> &'static Rex3 {
        let state = state.map(|s| s.to_vec());
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                let rex: &'static Rex3 = Box::leak(Box::new(Rex3::new(crate::disp::StatusSource::detached())));
                unsafe {
                    (*rex.fb_rgb.get()).fill(0);
                    (*rex.fb_aux.get()).fill(0);
                }
                if let Some(s) = &state {
                    rex.rec_state_decode(s).expect("bad state record");
                }
                #[cfg(feature = "rex-jit")]
                rex.jit_enabled.store(jit, Ordering::Relaxed);
                let _ = jit;
                rex.start();
                rex
            })
            .expect("spawn")
            .join()
            .expect("board thread panicked")
    }

    /// What a replay saw.
    #[derive(Debug, Default)]
    pub struct Report {
        pub records: usize,
        pub checked: usize,
        /// Record indices of failed checkpoints.
        pub failed: Vec<usize>,
        pub reads_differ: u64,
        pub differ_at: std::collections::BTreeMap<u32, u32>,
    }

    impl Report {
        pub fn diverged(&self) -> bool {
            !self.failed.is_empty()
        }
    }

    /// The `S` record a recording starts from, if any.
    pub fn start_state(recs: &[Rec]) -> Option<&[u8]> {
        match recs.first() {
            Some(Rec::State(s)) => Some(s),
            _ => None,
        }
    }

    /// Feed `recs` through `rex`'s bus interface the way the CPU and the DMA
    /// engine did (retrying while the board answers busy); checkpoints are
    /// checked when `check_hashes`. An `S` record must have been loaded into
    /// the board already (`board`).
    pub fn replay_recs(rex: &Rex3, recs: &[Rec], check_hashes: bool) -> Report {
        let mut rep = Report { records: recs.len(), ..Default::default() };
        let addr = |off: u32| (REX3_BASE | (off & (REX3_SIZE - 1))) as u64;
        let differ = |rep: &mut Report, off: u32| {
            rep.reads_differ += 1;
            *rep.differ_at.entry(off).or_default() += 1;
        };
        for (i, r) in recs.iter().enumerate() {
            match r {
                Rec::Write { bits, off, val } => {
                    let a = addr(*off);
                    while match bits {
                        8 => rex.write8(a, *val as u8),
                        16 => rex.write16(a, *val as u16),
                        32 => rex.write32(a, *val as u32),
                        _ => rex.write64(a, *val),
                    } == BUS_BUSY
                    {
                        std::hint::spin_loop();
                    }
                }
                Rec::Read { bits, off, val } => {
                    let a = addr(*off);
                    let got = loop {
                        let (st, v) = match bits {
                            8 => { let r = rex.read8(a); (r.status, r.data as u64) }
                            16 => { let r = rex.read16(a); (r.status, r.data as u64) }
                            32 => { let r = rex.read32(a); (r.status, r.data as u64) }
                            _ => { let r = rex.read64(a); (r.status, r.data) }
                        };
                        if st != BUS_BUSY {
                            break v;
                        }
                        std::hint::spin_loop();
                    };
                    if got != *val {
                        differ(&mut rep, *off);
                    }
                }
                Rec::DmaRead { off, val } => {
                    if rex.dma_read64(addr(*off)).data != *val {
                        differ(&mut rep, *off);
                    }
                }
                Rec::BulkWrite { off, vals } => {
                    while rex.dma_write64_bulk(addr(*off), vals) == BUS_BUSY {
                        std::hint::spin_loop();
                    }
                }
                Rec::BulkRead { off, vals } => {
                    let mut got = vec![0u64; vals.len()];
                    rex.dma_read64_bulk(addr(*off), &mut got);
                    if got != *vals {
                        differ(&mut rep, *off);
                    }
                }
                Rec::Hash(h) => {
                    if check_hashes {
                        rex.wait_idle();
                        rep.checked += 1;
                        if rex.rec_state_hash() != *h {
                            rep.failed.push(i);
                        }
                    }
                }
                Rec::State(_) => {}
            }
        }
        rex.wait_idle();
        rep
    }

    /// Replay a recording file on a fresh board and check every checkpoint.
    pub fn replay(path: &str, png: Option<&str>) -> Result<(), String> {
        let mut recs = load(path).map_err(|e| format!("{path}: {e}"))?;
        // REX3_REPLAY_LIMIT=<n>: only the first n records (bisecting).
        if let Some(n) = std::env::var("REX3_REPLAY_LIMIT").ok().and_then(|v| v.parse::<usize>().ok()) {
            recs.truncate(n);
        }
        let jit = match std::env::var("REX3_REPLAY_ENGINE").as_deref() {
            Ok("interp") => false,
            Ok("jit") | Err(_) => true,
            Ok(other) => return Err(format!("REX3_REPLAY_ENGINE={other}: want interp or jit")),
        };
        let rex = board(start_state(&recs), jit);
        if let Ok(t) = std::env::var("REX3_REPLAY_TRACE") {
            rex.start_trace(&t, super::super::trace::TRACE_ALL).map_err(|e| format!("{t}: {e}"))?;
        }
        let check = std::env::var("REX3_REPLAY_NOCHECK").is_err();
        let t0 = std::time::Instant::now();
        let rep = replay_recs(rex, &recs, check);
        let took = t0.elapsed();
        rex.stop_trace();
        if let Some(p) = png {
            rex.save_framebuffers_named(std::path::Path::new(p), "replay").map_err(|e| format!("{p}: {e}"))?;
        }
        // REX3_REPLAY_STATE=<file>: the final register state as TOML, to diff
        // two replays (engines, or before and after a change).
        if let Ok(f) = std::env::var("REX3_REPLAY_STATE") {
            let mut st = rex.save_state();
            if let Some(t) = st.as_table_mut() {
                t.insert("rec_extra".into(), rex.rec_extra());
            }
            std::fs::write(&f, toml::to_string(&st).unwrap_or_default()).map_err(|e| format!("{f}: {e}"))?;
        }
        let h = rex.rec_state_hash();
        rex.stop();
        eprintln!(
            "{path}: {} records in {:.3} s ({}), final state {}",
            rep.records,
            took.as_secs_f64(),
            if jit { "compiled shaders" } else { "interpreter" },
            h.iter().take(8).map(|b| format!("{b:02x}")).collect::<String>()
        );
        eprintln!(
            "{path}: {} checkpoints, {} failed {:?}, {} bus reads differ",
            rep.checked, rep.failed.len(), &rep.failed[..rep.failed.len().min(8)], rep.reads_differ
        );
        if !rep.differ_at.is_empty() {
            eprintln!("  bus reads that differ, by offset: {:x?}", rep.differ_at);
        }
        if check && rep.checked == 0 {
            return Err("no checkpoints in recording".into());
        }
        if rep.diverged() {
            return Err(format!("replay diverged: {} of {} checkpoints failed", rep.failed.len(), rep.checked));
        }
        Ok(())
    }

    #[test]
    fn rex3_replay_file() {
        let Ok(path) = std::env::var("REX3_REPLAY") else { return };
        let png = std::env::var("REX3_REPLAY_OUT").ok();
        replay(&path, png.as_deref()).unwrap();
    }

    /// Every recording in `testdata/` replays with its checkpoints intact.
    #[test]
    fn rex3_replay_testdata() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/dev/ng1/testdata");
        let Ok(entries) = std::fs::read_dir(&dir) else { return };
        let mut n = 0;
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().is_some_and(|x| x == "txt" || x == "rec") {
                replay(p.to_str().unwrap(), None).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
                n += 1;
            }
        }
        eprintln!("{n} REX3 recordings replayed");
    }

    fn all() -> Vec<Rec> {
        vec![
            Rec::State(vec![1, 2, 3, 0xFF]),
            Rec::Write { bits: 32, off: 0x954, val: 0x1234_5678 },
            Rec::Write { bits: 64, off: 0xA30, val: 0x0102_0304_0506_0708 },
            Rec::Write { bits: 8, off: 0x240, val: 0xAB },
            Rec::Read { bits: 32, off: 0x1338, val: 0x53 },
            Rec::Read { bits: 64, off: 0xA30, val: u64::MAX },
            Rec::DmaRead { off: 0xA30, val: 0xDEAD_BEEF_0000_0001 },
            Rec::BulkWrite { off: 0xA30, vals: vec![1, 2, 0xFFFF_FFFF_FFFF] },
            Rec::BulkRead { off: 0xA30, vals: vec![] },
            Rec::Hash([7; 32]),
        ]
    }

    fn round_trip(ext: &str) {
        let path = std::env::temp_dir().join(format!("rex3_rec_{}{ext}", std::process::id()));
        let path = path.to_str().unwrap().to_string();
        let mut r = Recorder::create(&path).unwrap();
        for x in all() {
            r.put(&x);
        }
        r.finish().unwrap();
        assert_eq!(load(&path).unwrap(), all());
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

    #[test]
    fn truncated_binary_recording_keeps_complete_records() {
        let path = std::env::temp_dir().join(format!("rex3_rec_cut_{}.rec", std::process::id()));
        let path = path.to_str().unwrap().to_string();
        let mut r = Recorder::create(&path).unwrap();
        for x in all() {
            r.put(&x);
        }
        r.finish().unwrap();
        let data = std::fs::read(&path).unwrap();
        std::fs::write(&path, &data[..data.len() - 5]).unwrap();
        let mut want = all();
        want.pop();
        assert_eq!(load(&path).unwrap(), want);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn rle_round_trip() {
        let words: Vec<u32> = (0..1000u32).map(|i| if i % 97 < 50 { 7 } else { i }).collect();
        let mut out = Vec::new();
        rle_put(&mut out, &words);
        let mut at = 0;
        assert_eq!(rle_get(&out, &mut at).unwrap(), words);
        assert_eq!(at, out.len());
    }
}
