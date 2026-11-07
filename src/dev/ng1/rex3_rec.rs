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
//! `REX3_REPLAY=<file> cargo test -p iris --lib --features gputrace rex3_replay_file -- --ignored --nocapture`
//! (`REX3_REPLAY_OUT=<dir>` saves the final framebuffers there,
//! `REX3_REPLAY_TRACE=<file>` writes the annotated trace of the replay,
//! `REX3_REPLAY_ENGINE=interp|jit|gpu` picks the draw path (`gpu` requires the
//! opt-in `rex-gpu` feature and is replay-only),
//! `REX3_GPU_BACKEND=auto|vulkan|metal|dx12` selects its wgpu backend,
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
    use crate::dev::ng1::rex3::{
        REX3_BASE, REX3_SIZE, REX3_STATUS, REX3_USER_STATUS, STATUS_BACKBUSY,
        STATUS_GFXBUSY, STATUS_GFIFOLEVEL_MASK, STATUS_GFIFO_INT, STATUS_VRINT,
    };
    use crate::traits::{BusDevice, Device, BUS_BUSY, BUS_OK};
    use std::sync::Arc;
    use std::time::Duration;

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

    type CleanupTask = Box<dyn FnOnce() + Send + 'static>;

    fn spawn_cleanup(task: CleanupTask) -> std::io::Result<()> {
        std::thread::Builder::new().name("REX3-ReplayCleanup".into()).spawn(task).map(|_| ())
    }

    // start() gives the processor and refresh threads a 'static borrow. Keep
    // the board alive until both have joined, even if spawning the reaper
    // fails or the reaper itself panics.
    struct ReplayCleanup {
        rex: Option<Arc<Rex3>>,
        exit_queued: bool,
    }

    impl ReplayCleanup {
        fn run(mut self) {
            let rex = self.rex.as_ref().unwrap();
            let processor = rex.processor_thread.lock().take();
            if let Some(processor) = processor {
                let backoff = crossbeam_utils::Backoff::new();
                while !self.exit_queued && !processor.is_finished() {
                    if rex.gfifo_try_push(super::super::GFIFO_EXIT, 0) {
                        self.exit_queued = true;
                    } else if backoff.is_completed() {
                        std::thread::park_timeout(Duration::from_millis(2));
                    } else {
                        backoff.snooze();
                    }
                }
                if processor.join().is_err() {
                    eprintln!("REX3: replay processor panicked during shutdown");
                }
            }
            let refresh = rex.refresh_thread.lock().take();
            if let Some(refresh) = refresh {
                if refresh.join().is_err() {
                    eprintln!("REX3: replay refresh thread panicked during shutdown");
                }
            }
            if let Some(summary) = rex.stop_trace() {
                eprintln!("REX3: replay trace closed: {summary}");
            }
            if let Err(e) = rex.set_recording(None, false) {
                eprintln!("REX3: closing replay recording: {e}");
            }
            drop(self.rex.take());
        }
    }

    impl Drop for ReplayCleanup {
        fn drop(&mut self) {
            if let Some(rex) = self.rex.take() {
                std::mem::forget(rex);
            }
        }
    }

    fn request_cleanup(
        rex: Arc<Rex3>, spawn: impl FnOnce(CleanupTask) -> std::io::Result<()>,
    ) -> Result<(), String> {
        rex.display_stop.request();
        rex.running.store(false, Ordering::SeqCst);
        rex.tap.fetch_and(!super::super::trace::TRACE_ALL, Ordering::SeqCst);
        let exit_queued = rex.gfifo_try_push(super::super::GFIFO_EXIT, 0);
        let cleanup = ReplayCleanup { rex: Some(rex), exit_queued };
        spawn(Box::new(move || cleanup.run()))
            .map_err(|e| format!("could not start REX3 replay shutdown worker: {e}"))
    }

    struct ReplayBoard(Option<Arc<Rex3>>);

    impl ReplayBoard {
        fn rex(&self) -> &Rex3 {
            self.0.as_deref().expect("replay board already stopped")
        }

        fn abort(
            mut self, error: String, spawn: impl FnOnce(CleanupTask) -> std::io::Result<()>,
        ) -> String {
            let rex = self.0.take().expect("replay board already stopped");
            match request_cleanup(rex, spawn) {
                Ok(()) => error,
                Err(shutdown) => format!("{error}; additionally, {shutdown}"),
            }
        }

        fn stopped(&mut self) {
            drop(self.0.take());
        }
    }

    impl Drop for ReplayBoard {
        fn drop(&mut self) {
            if let Some(rex) = self.0.take() {
                if let Err(e) = request_cleanup(rex, spawn_cleanup) {
                    eprintln!("REX3: {e}");
                }
            }
        }
    }

    fn replay_board(state: Option<&[u8]>, jit: bool) -> Result<ReplayBoard, String> {
        let state = state.map(|s| s.to_vec());
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                let rex = Arc::new(Rex3::new(crate::disp::StatusSource::detached()));
                unsafe {
                    (*rex.fb_rgb.get()).fill(0);
                    (*rex.fb_aux.get()).fill(0);
                }
                if let Some(s) = &state {
                    rex.rec_state_decode(s).map_err(|e| format!("bad state record: {e}"))?;
                }
                #[cfg(feature = "rex-jit")]
                rex.jit_enabled.store(jit, Ordering::Relaxed);
                let _ = jit;
                let board = ReplayBoard(Some(rex));
                board.rex().start();
                Ok(board)
            })
            .map_err(|e| format!("cannot start REX3 replay setup: {e}"))?
            .join()
            .map_err(|_| "REX3 replay setup thread panicked".to_string())?
    }

    /// What a replay saw.
    #[derive(Debug, Default)]
    pub struct Report {
        pub records: usize,
        /// Records actually executed, excluding any after an access error.
        pub executed: usize,
        /// Successful read transactions (including DMA and bulk reads).
        pub reads_checked: usize,
        pub checked: usize,
        /// Record indices of failed checkpoints.
        pub failed: Vec<usize>,
        pub reads_differ: u64,
        pub differ_at: std::collections::BTreeMap<u32, u32>,
        pub access_error: Option<String>,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum ReplayEngine {
        Interp,
        Jit,
        Gpu,
    }

    impl ReplayEngine {
        fn label(self) -> &'static str {
            match self {
                Self::Interp => "interpreter",
                Self::Jit => "compiled shaders",
                Self::Gpu => "GPU replay",
            }
        }
    }

    impl Report {
        pub fn diverged(&self) -> bool {
            !self.failed.is_empty() || self.reads_differ != 0 || self.access_error.is_some()
        }
    }

    // Queue depth/GFXBUSY and GFIFO_INT depend on consumer scheduling, VRINT
    // on the refresh thread, and STATUS's BACKBUSY on the DCB wall-clock
    // deadline. USER_STATUS does not sample that deadline. All other bits,
    // including VERSION, VIDEOINT and BFIFO fields, remain strict. A 64-bit
    // read combines two 32-bit registers, so mask each half independently.
    // The bus read executes first, preserving read-to-clear and GO effects.
    fn volatile_status_bits(off: u32) -> u32 {
        let common = STATUS_GFXBUSY | STATUS_GFIFOLEVEL_MASK | STATUS_GFIFO_INT | STATUS_VRINT;
        match (off & (REX3_SIZE - 1)) & !0x800 {
            REX3_STATUS => common | STATUS_BACKBUSY,
            REX3_USER_STATUS => common,
            _ => 0,
        }
    }

    fn read_matches(bits: u8, off: u32, got: u64, expected: u64) -> bool {
        let volatile = match bits {
            32 => volatile_status_bits(off) as u64,
            64 => ((volatile_status_bits(off) as u64) << 32)
                | volatile_status_bits(off.wrapping_add(4)) as u64,
            _ => 0,
        };
        (got ^ expected) & !volatile == 0
    }

    // VDMA retries the same operation on BUS_BUSY; read data is unspecified
    // until BUS_OK. Back off and park as the GFIFO consumer does so a slow
    // draw cannot make the replay burn a core; parking is not a deadline.
    fn retry_dma<T>(mut access: impl FnMut() -> (u32, T)) -> Result<T, u32> {
        let backoff = crossbeam_utils::Backoff::new();
        loop {
            let (status, data) = access();
            if status == BUS_OK {
                return Ok(data);
            }
            if status != BUS_BUSY {
                return Err(status);
            }
            if backoff.is_completed() {
                std::thread::park_timeout(Duration::from_millis(2));
            } else {
                backoff.snooze();
            }
        }
    }

    fn access_failure(i: usize, off: u32, operation: &str, status: u32) -> String {
        format!("record {i} {operation} at {off:#x} failed with status {status:#x}")
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
        replay_recs_with_bus(rex, rex, recs, check_hashes)
    }

    fn replay_recs_with_bus(
        rex: &Rex3, bus: &dyn BusDevice, recs: &[Rec], check_hashes: bool,
    ) -> Report {
        replay_recs_with_hook(rex, bus, recs, check_hashes, |_, _| {})
    }

    pub(super) fn replay_recs_with_hook(
        rex: &Rex3, bus: &dyn BusDevice, recs: &[Rec], check_hashes: bool,
        mut before_record: impl FnMut(usize, &Rec),
    ) -> Report {
        let mut rep = Report { records: recs.len(), ..Default::default() };
        let addr = |off: u32| (REX3_BASE | (off & (REX3_SIZE - 1))) as u64;
        let differ = |rep: &mut Report, off: u32| {
            rep.reads_differ += 1;
            *rep.differ_at.entry(off).or_default() += 1;
        };
        for (i, r) in recs.iter().enumerate() {
            before_record(i, r);
            match r {
                Rec::Write { bits, off, val } => {
                    let a = addr(*off);
                    let status = loop {
                        let status = match bits {
                            8 => bus.write8(a, *val as u8),
                            16 => bus.write16(a, *val as u16),
                            32 => bus.write32(a, *val as u32),
                            _ => bus.write64(a, *val),
                        };
                        if status != BUS_BUSY {
                            break status;
                        }
                        std::hint::spin_loop();
                    };
                    if status != BUS_OK {
                        rep.access_error = Some(access_failure(i, *off, "write", status));
                        break;
                    }
                }
                Rec::Read { bits, off, val } => {
                    let a = addr(*off);
                    let got = loop {
                        let (st, v) = match bits {
                            8 => { let r = bus.read8(a); (r.status, r.data as u64) }
                            16 => { let r = bus.read16(a); (r.status, r.data as u64) }
                            32 => { let r = bus.read32(a); (r.status, r.data as u64) }
                            _ => { let r = bus.read64(a); (r.status, r.data) }
                        };
                        if st == BUS_OK {
                            break Some(v);
                        }
                        if st != BUS_BUSY {
                            rep.access_error = Some(access_failure(i, *off, "read", st));
                            break None;
                        }
                        std::hint::spin_loop();
                    };
                    let Some(got) = got else { break };
                    rep.reads_checked += 1;
                    if !read_matches(*bits, *off, got, *val) {
                        differ(&mut rep, *off);
                    }
                }
                Rec::DmaRead { off, val } => {
                    match retry_dma(|| {
                        let r = bus.dma_read64(addr(*off));
                        (r.status, r.data)
                    }) {
                        Ok(got) => {
                            rep.reads_checked += 1;
                            if !read_matches(64, *off, got, *val) {
                                differ(&mut rep, *off);
                            }
                        }
                        Err(status) => {
                            rep.access_error = Some(access_failure(i, *off, "dma_read64", status));
                            break;
                        }
                    }
                }
                Rec::BulkWrite { off, vals } => {
                    if let Err(status) = retry_dma(|| (bus.dma_write64_bulk(addr(*off), vals), ())) {
                        rep.access_error = Some(access_failure(i, *off, "dma_write64_bulk", status));
                        break;
                    }
                }
                Rec::BulkRead { off, vals } => {
                    let mut got = vec![0u64; vals.len()];
                    if let Err(status) = retry_dma(|| (bus.dma_read64_bulk(addr(*off), &mut got), ())) {
                        rep.access_error = Some(access_failure(i, *off, "dma_read64_bulk", status));
                        break;
                    }
                    rep.reads_checked += 1;
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
            rep.executed += 1;
        }
        // A failed access can leave the processor blocked or already dead.
        // Only a completed replay may synchronously drain it.
        if rep.access_error.is_none() {
            rex.wait_idle();
        }
        rep
    }

    /// Replay a recording file on a fresh board and check every checkpoint.
    pub fn replay(path: &str, png: Option<&str>) -> Result<(), String> {
        replay_with_board(path, png, replay_board, spawn_cleanup)
    }

    fn replay_with_board(
        path: &str, png: Option<&str>,
        make_board: impl FnOnce(Option<&[u8]>, bool) -> Result<ReplayBoard, String>,
        spawn: impl FnOnce(CleanupTask) -> std::io::Result<()>,
    ) -> Result<(), String> {
        let mut recs = load(path).map_err(|e| format!("{path}: {e}"))?;
        // REX3_REPLAY_LIMIT=<n>: only the first n records (bisecting).
        if let Some(n) = std::env::var("REX3_REPLAY_LIMIT").ok().and_then(|v| v.parse::<usize>().ok()) {
            recs.truncate(n);
        }
        let engine = match std::env::var("REX3_REPLAY_ENGINE").as_deref() {
            Ok("interp") => ReplayEngine::Interp,
            Ok("jit") | Err(_) => ReplayEngine::Jit,
            Ok("gpu") => {
                #[cfg(feature = "rex-gpu")]
                {
                    ReplayEngine::Gpu
                }
                #[cfg(not(feature = "rex-gpu"))]
                {
                    return Err("REX3_REPLAY_ENGINE=gpu requires cargo --features rex-gpu".into());
                }
            }
            Ok(other) => {
                return Err(format!(
                    "REX3_REPLAY_ENGINE={other}: want interp, jit, or gpu"
                ))
            }
        };
        let mut board = make_board(start_state(&recs), engine == ReplayEngine::Jit)?;
        let result = (|| {
            let rex = board.rex();
            #[cfg(feature = "rex-gpu")]
            if engine == ReplayEngine::Gpu {
                let backend = std::env::var("REX3_GPU_BACKEND").unwrap_or_else(|_| "auto".into());
                let diagnostics = rex.enable_gpu_replay(&backend)?;
                eprintln!("{path}: REX3 GPU replay {diagnostics}");
            }
            if let Ok(t) = std::env::var("REX3_REPLAY_TRACE") {
                rex.start_trace(&t, super::super::trace::TRACE_ALL).map_err(|e| format!("{t}: {e}"))?;
            }
            let check = std::env::var("REX3_REPLAY_NOCHECK").is_err();
            let t0 = std::time::Instant::now();
            let rep = replay_recs(rex, &recs, check);
            let took = t0.elapsed();
            if let Some(error) = &rep.access_error {
                return Err(format!("replay failed: {error}"));
            }
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
            Ok((rep, took, rex.rec_state_hash(), check))
        })();
        let (rep, took, h, check) = match result {
            Ok(result) => result,
            Err(error) => return Err(board.abort(error, spawn)),
        };
        let rex = board.rex();
        rex.stop();
        #[cfg(feature = "rex-gpu")]
        let gpu_report = if engine == ReplayEngine::Gpu {
            rex.gpu_replay_report()
        } else {
            None
        };
        board.stopped();
        eprintln!(
            "{path}: {} records in {:.3} s ({}), final state {}",
            rep.records,
            took.as_secs_f64(),
            engine.label(),
            h.iter().take(8).map(|b| format!("{b:02x}")).collect::<String>()
        );
        eprintln!(
            "{path}: {} checkpoints, {} failed {:?}, {} bus reads differ",
            rep.checked, rep.failed.len(), &rep.failed[..rep.failed.len().min(8)], rep.reads_differ
        );
        if !rep.differ_at.is_empty() {
            eprintln!("  bus reads that differ, by offset: {:x?}", rep.differ_at);
        }
        #[cfg(feature = "rex-gpu")]
        if let Some(report) = &gpu_report {
            eprintln!(
                "{path}: GPU handled={} fallback={} submissions={} fences={} \
                 upload_bytes={} readback_bytes={} fallback_reasons={:?}",
                report.handled_draws,
                report.fallback_draws,
                report.submissions,
                report.fences,
                report.bytes_uploaded,
                report.bytes_read_back,
                report.fallback_reasons,
            );
            if let Some(telemetry) = &report.telemetry {
                eprintln!("{path}: GPU telemetry={telemetry:?}");
            }
            if let Some(error) = &report.error {
                return Err(format!("GPU replay failed: {error}"));
            }
            if let Some(error) = &report.timestamp_error {
                return Err(format!("GPU timestamp profiling failed: {error}"));
            }
            if report.handled_draws == 0 || report.submissions == 0 {
                return Err(
                    "GPU replay submitted no draws; the recording does not exercise the initial flat-fill subset"
                        .into(),
                );
            }
        }
        if check && rep.checked == 0 {
            return Err("no checkpoints in recording".into());
        }
        if rep.diverged() {
            return Err(format!("replay diverged: {} of {} checkpoints failed, {} bus reads differ",
                rep.failed.len(), rep.checked, rep.reads_differ));
        }
        Ok(())
    }

    #[test]
    fn replay_rejects_mismatched_bus_reads() {
        let rex = board(None, false);
        let off = super::super::REX3_DRAWMODE0;
        let actual = rex.read32((REX3_BASE | off) as u64).data;
        let rep = replay_recs(rex, &[Rec::Read {
            bits: 32, off, val: (actual ^ 1) as u64,
        }], true);
        rex.stop();
        assert_eq!(rep.checked, 0);
        assert!(rep.failed.is_empty());
        assert_eq!(rep.reads_differ, 1);
        assert_eq!(rep.differ_at.get(&off), Some(&1));
        assert!(rep.diverged());
    }

    #[test]
    fn review_replay_status_go_volatile_bits_and_side_effects() {
        use super::super::*;
        use std::sync::{Arc, Mutex};

        #[derive(Default)]
        struct Irqs(Mutex<Vec<bool>>);
        impl crate::irq::InterruptController for Irqs {
            fn set_line(&self, _: u32, level: bool) {
                self.0.lock().unwrap().push(level);
            }
        }

        let rex = board(None, false);
        let irqs = Arc::new(Irqs::default());
        *rex.vblank_cb.lock() = Some(crate::irq::IrqLine::new(irqs.clone(), 0));
        rex.config.status.fetch_or(STATUS_VRINT, Ordering::Relaxed);

        let xy = |x: u32, y: u32| ((x + REX3_COORD_BIAS as u32) << 16)
            | (y + REX3_COORD_BIAS as u32);
        let dm1 = DRAWMODE1_PLANES_RGB | (1 << 3)
            | DRAWMODE1_COMPARE_DISABLE_SH | DRAWMODE1_LOGICOP_SRC_SH;
        let mut recs: Vec<_> = [
            (REX3_DRAWMODE1, dm1),
            (REX3_WRMASK, 0xff),
            (REX3_CLIPMODE, 0xF << CLIPMODE_CIDMATCH_SHIFT),
            (REX3_TOPSCAN, 0x3ff),
            (REX3_COLORI, 0x55),
            (REX3_XYENDI, xy(17, 33)),
            (REX3_XYSTARTI, xy(17, 30)),
            (REX3_DRAWMODE0, 0x206),
        ].into_iter().map(|(off, val)| Rec::Write {
            bits: 32, off, val: val as u64,
        }).collect();
        let volatile = STATUS_GFXBUSY | STATUS_VRINT | STATUS_GFIFOLEVEL_MASK | STATUS_GFIFO_INT;
        let status = 3 | volatile | STATUS_BACKBUSY;
        let user_status = 3 | volatile;
        recs.extend([
            Rec::Read { bits: 32, off: REX3_STATUS | 0x800, val: status as u64 },
            Rec::Read { bits: 32, off: REX3_USER_STATUS | 0x800, val: user_status as u64 },
            Rec::Read { bits: 64, off: REX3_STATUS | 0x800,
                val: ((status as u64) << 32) | user_status as u64 },
            Rec::DmaRead { off: REX3_STATUS | 0x800,
                val: ((status as u64) << 32) | user_status as u64 },
        ]);
        let rep = replay_recs(rex, &recs, true);
        let pixels: Vec<_> = (30..=33).map(|y| unsafe {
            (*rex.fb_rgb.get())[y * 2048 + 17] & 0xff
        }).collect();
        let irq_cleared = irqs.0.lock().unwrap().iter().any(|&level| !level);
        rex.stop();

        assert_eq!(pixels, [0x55, 0x55, 0x55, 0x55], "each STATUS GO must execute");
        assert!(irq_cleared, "the STATUS read must clear the vblank interrupt");
        assert_eq!(rep.reads_differ, 0, "{rep:?}");
        assert!(!rep.diverged(), "{rep:?}");
    }

    #[test]
    fn review_replay_status_deterministic_bits_and_other_reads_are_strict() {
        use super::super::*;
        let rex = board(None, false);
        let recs = [
            Rec::Read { bits: 32, off: REX3_STATUS | 0x800, val: 2 },
            Rec::Read { bits: 32, off: REX3_USER_STATUS | 0x800,
                val: (3 | STATUS_VIDEOINT) as u64 },
            Rec::Read { bits: 32, off: REX3_USER_STATUS | 0x800,
                val: (3 | STATUS_BACKBUSY) as u64 },
            Rec::Read { bits: 64, off: REX3_STATUS | 0x800,
                val: (((3 | STATUS_BFIFO_INT) as u64) << 32) | 3 },
            Rec::DmaRead { off: REX3_STATUS | 0x800,
                val: (((3 | STATUS_VIDEOINT) as u64) << 32) | 3 },
            Rec::Read { bits: 32, off: REX3_DRAWMODE0, val: STATUS_GFXBUSY as u64 },
        ];
        let rep = replay_recs(rex, &recs, true);
        rex.stop();
        assert_eq!(rep.reads_differ, 6, "{rep:?}");
        assert!(rep.diverged());
    }

    #[test]
    fn review_replay_status_mask_is_limited_to_volatile_fields() {
        let common = STATUS_GFXBUSY | STATUS_GFIFOLEVEL_MASK | STATUS_GFIFO_INT | STATUS_VRINT;
        let status = common | STATUS_BACKBUSY;
        for (off, mask) in [(REX3_STATUS, status), (REX3_USER_STATUS, common)] {
            for bit in 0..32 {
                let changed = 1u32 << bit;
                assert_eq!(read_matches(32, off | 0x800, 0, changed as u64), mask & changed != 0,
                    "offset {off:#x}, bit {bit}");
            }
        }
        for (off, mask) in [
            (REX3_STATUS, ((status as u64) << 32) | common as u64),
            (REX3_STATUS - 4, status as u64),
            (REX3_USER_STATUS, (common as u64) << 32),
        ] {
            for bit in 0..64 {
                let changed = 1u64 << bit;
                assert_eq!(read_matches(64, off | 0x800, 0, changed), mask & changed != 0,
                    "offset {off:#x}, bit {bit}");
            }
        }
        assert!(!read_matches(8, REX3_STATUS, 0, STATUS_GFXBUSY as u64));
        assert!(!read_matches(16, REX3_STATUS, 0, STATUS_GFXBUSY as u64));
        assert!(!read_matches(32, REX3_STATUS - 4, 0, STATUS_GFXBUSY as u64));
    }

    #[test]
    fn review_replay_status_mask_does_not_waive_checkpoint_hashes() {
        let rex = board(None, false);
        let recs = [
            Rec::Read { bits: 32, off: REX3_STATUS | 0x800,
                val: (3 | STATUS_GFXBUSY | STATUS_GFIFOLEVEL_MASK) as u64 },
            Rec::Hash([0; 32]),
        ];
        let rep = replay_recs(rex, &recs, true);
        rex.stop();
        assert_eq!(rep.reads_differ, 0, "{rep:?}");
        assert_eq!(rep.checked, 1);
        assert_eq!(rep.failed, [1]);
        assert!(rep.diverged());
    }

    #[derive(Default)]
    struct DmaBusyThenReady {
        busy_replies: usize,
        scalar_busy_delay: Duration,
        scalar_calls: std::sync::atomic::AtomicUsize,
        bulk_calls: std::sync::atomic::AtomicUsize,
        write_calls: std::sync::atomic::AtomicUsize,
    }

    impl BusDevice for DmaBusyThenReady {
        fn dma_read64(&self, addr: u64) -> crate::traits::BusRead64 {
            assert_eq!(addr, (REX3_BASE | 0xa30) as u64);
            if self.scalar_calls.fetch_add(1, Ordering::Relaxed) < self.busy_replies {
                if !self.scalar_busy_delay.is_zero() {
                    std::thread::sleep(self.scalar_busy_delay);
                }
                crate::traits::BusRead64::busy()
            } else {
                crate::traits::BusRead64::ok(0x1234)
            }
        }

        fn dma_read64_bulk(&self, addr: u64, out: &mut [u64]) -> u32 {
            assert_eq!(addr, (REX3_BASE | 0xa30) as u64);
            if self.bulk_calls.fetch_add(1, Ordering::Relaxed) < self.busy_replies {
                out.fill(0xdead);
                BUS_BUSY
            } else {
                out.copy_from_slice(&[0x5678, 0x9abc]);
                crate::traits::BUS_OK
            }
        }

        fn dma_write64_bulk(&self, addr: u64, vals: &[u64]) -> u32 {
            assert_eq!(addr, (REX3_BASE | 0xa30) as u64);
            assert_eq!(vals, [0x55]);
            if self.write_calls.fetch_add(1, Ordering::Relaxed) < self.busy_replies {
                BUS_BUSY
            } else {
                BUS_OK
            }
        }
    }

    #[test]
    fn review_replay_dma_busy_then_success() {
        let rex = board(None, false);
        let bus = DmaBusyThenReady { busy_replies: 1, ..Default::default() };
        let recs = [
            Rec::BulkWrite { off: 0xa30, vals: vec![0x55] },
            Rec::DmaRead { off: 0xa30, val: 0x1234 },
            Rec::BulkRead { off: 0xa30, vals: vec![0x5678, 0x9abc] },
        ];
        let rep = replay_recs_with_bus(rex, &bus, &recs, true);
        rex.stop();
        assert_eq!(bus.write_calls.load(Ordering::Relaxed), 2);
        assert_eq!(bus.scalar_calls.load(Ordering::Relaxed), 2);
        assert_eq!(bus.bulk_calls.load(Ordering::Relaxed), 2);
        assert_eq!(rep.reads_differ, 0, "{rep:?}");
        assert!(!rep.diverged(), "{rep:?}");
    }

    #[test]
    fn replay_dma_retries_many_busy_responses_past_previous_deadline() {
        const BUSY_REPLIES: usize = 64;
        let rex = board(None, false);
        rex.wait_idle();
        let bus = DmaBusyThenReady {
            busy_replies: BUSY_REPLIES,
            // The busy responses span longer than the old five-second cutoff.
            scalar_busy_delay: Duration::from_millis(85),
            ..Default::default()
        };
        let recs = [
            Rec::BulkWrite { off: 0xa30, vals: vec![0x55] },
            Rec::DmaRead { off: 0xa30, val: 0x1234 },
            Rec::BulkRead { off: 0xa30, vals: vec![0x5678, 0x9abc] },
            Rec::Hash(rex.rec_state_hash()),
        ];
        let rep = replay_recs_with_bus(rex, &bus, &recs, true);
        rex.stop();
        assert_eq!(bus.write_calls.load(Ordering::Relaxed), BUSY_REPLIES + 1);
        assert_eq!(bus.scalar_calls.load(Ordering::Relaxed), BUSY_REPLIES + 1, "{rep:?}");
        assert_eq!(bus.bulk_calls.load(Ordering::Relaxed), BUSY_REPLIES + 1);
        assert_eq!(rep.checked, 1, "{rep:?}");
        assert_eq!(rep.reads_differ, 0, "{rep:?}");
        assert!(!rep.diverged(), "{rep:?}");
    }

    #[test]
    fn review_replay_dma_success_data_remains_strict() {
        let rex = board(None, false);
        let bus = DmaBusyThenReady { busy_replies: 16, ..Default::default() };
        let recs = [
            Rec::DmaRead { off: 0xa30, val: 0x1235 },
            Rec::BulkRead { off: 0xa30, vals: vec![0x5678, 0x9abd] },
        ];
        let rep = replay_recs_with_bus(rex, &bus, &recs, true);
        rex.stop();
        assert_eq!(bus.scalar_calls.load(Ordering::Relaxed), 17);
        assert_eq!(bus.bulk_calls.load(Ordering::Relaxed), 17);
        assert_eq!(rep.reads_differ, 2, "{rep:?}");
        assert_eq!(rep.differ_at.get(&0xa30), Some(&2));
        assert!(rep.access_error.is_none());
        assert!(rep.diverged());
    }

    struct DmaFailure {
        status: u32,
        busy_replies: usize,
        calls: std::sync::atomic::AtomicUsize,
    }

    impl DmaFailure {
        fn next_status(&self) -> u32 {
            if self.calls.fetch_add(1, Ordering::Relaxed) < self.busy_replies {
                BUS_BUSY
            } else {
                self.status
            }
        }
    }

    impl BusDevice for DmaFailure {
        fn dma_read64(&self, _: u64) -> crate::traits::BusRead64 {
            crate::traits::BusRead64 { status: self.next_status(), data: 0 }
        }

        fn dma_read64_bulk(&self, _: u64, out: &mut [u64]) -> u32 {
            out.fill(0xdead);
            self.next_status()
        }

        fn dma_write64_bulk(&self, _: u64, _: &[u64]) -> u32 {
            self.next_status()
        }
    }

    #[test]
    fn review_replay_dma_nonbusy_error_is_failure() {
        let rex = board(None, false);
        for rec in [
            Rec::DmaRead { off: 0xa30, val: 0 },
            Rec::BulkRead { off: 0xa30, vals: vec![0] },
            Rec::BulkWrite { off: 0xa30, vals: vec![0] },
        ] {
            for busy_replies in [0, 5] {
                let status = crate::traits::BUS_ERR;
                let bus = DmaFailure { status, busy_replies,
                    calls: std::sync::atomic::AtomicUsize::new(0) };
                let rep = replay_recs_with_bus(rex, &bus, &[rec.clone(), Rec::Hash([0; 32])], true);
                assert!(rep.diverged(), "{rec:?}: {status:#x} was accepted as success");
                let error = rep.access_error.as_deref().expect("explicit DMA failure");
                assert!(error.contains("record 0") && error.contains("0xa30"), "{error}");
                assert!(error.contains(&format!("{status:#x}")), "{error}");
                assert_eq!(rep.reads_differ, 0, "failed reads have no valid data");
                assert_eq!(rep.checked, 0, "replay must stop after an access failure");
                assert_eq!(bus.calls.load(Ordering::Relaxed), busy_replies + 1);
            }
        }
        rex.stop();
    }

    #[test]
    fn review_replay_pio_read_error_is_not_valid_zero_data() {
        let rex = board(None, false);
        let bus = DmaFailure { status: crate::traits::BUS_ERR, busy_replies: 0,
            calls: std::sync::atomic::AtomicUsize::new(0) };
        let rec = Rec::Read { bits: 32, off: super::super::REX3_DRAWMODE0, val: 0 };
        let rep = replay_recs_with_bus(rex, &bus, &[rec], true);
        rex.stop();
        assert!(rep.access_error.as_deref().is_some_and(|e| e.contains("read") && e.contains("0x800001c")));
        assert_eq!(rep.reads_differ, 0);
        assert!(rep.diverged());
    }

    #[test]
    fn replay_pio_write_error_is_terminal() {
        let rex = board(None, false);
        let bus = DmaFailure { status: crate::traits::BUS_ERR, busy_replies: 0,
            calls: std::sync::atomic::AtomicUsize::new(0) };
        let rec = Rec::Write { bits: 32, off: super::super::REX3_DRAWMODE0, val: 0 };
        let rep = replay_recs_with_bus(rex, &bus, &[rec, Rec::Hash([0; 32])], true);
        rex.stop();
        assert!(rep.access_error.as_deref().is_some_and(|e| e.contains("write") && e.contains("0x800001c")), "{rep:?}");
        assert_eq!(rep.checked, 0, "a failed write must stop replay before the checkpoint");
        assert!(rep.diverged());
    }

    fn replay_recording(name: &str, records: &[Rec]) -> std::path::PathBuf {
        let path = std::path::PathBuf::from(format!("rex3-replay-{name}-{}.rec", std::process::id()));
        let mut recorder = Recorder::create(path.to_str().unwrap()).unwrap();
        for record in records {
            recorder.put(record);
        }
        recorder.finish().unwrap();
        path
    }

    fn invalid_dma_recording(name: &str) -> std::path::PathBuf {
        replay_recording(name, &[Rec::BulkRead { off: super::super::REX3_HOSTRW0, vals: vec![0] }])
    }

    fn assert_dma_replay_error(result: Result<(), String>) {
        let error = result.expect_err("an invalid DMA port must not succeed");
        assert!(error.contains("record 0") && error.contains("dma_read64_bulk")
            && error.contains("status 0x800001c"), "{error}");
    }

    fn assert_reaped(weak: std::sync::Weak<Rex3>) {
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while weak.strong_count() != 0 && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(weak.strong_count(), 0, "replay cleanup retained an idle board");
    }

    #[test]
    fn replay_dma_error_returns_while_processor_is_active() {
        let board = replay_board(None, false).unwrap();
        let rex = Arc::clone(board.0.as_ref().unwrap());
        let weak = Arc::downgrade(&rex);
        rex.wait_idle();
        let shape_guard = rex.seen_shapes.lock();
        let trace_guard = rex.trace.lock();
        let go = (REX3_BASE | super::super::REX3_DRAWMODE0 | 0x800) as u64;
        assert_eq!(rex.write32(go, 0x6), BUS_OK);
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while !rex.gfxbusy.load(Ordering::Acquire) && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(rex.gfxbusy.load(Ordering::Acquire), "processor did not start");
        let path = invalid_dma_recording("active-error");
        let worker_path = path.clone();
        let (sent, received) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let result = replay_with_board(worker_path.to_str().unwrap(), None, |_, _| Ok(board), spawn_cleanup);
            let _ = sent.send(result);
        });
        let timely = received.recv_timeout(Duration::from_secs(2));
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        let mut handle_lock_available = false;
        while std::time::Instant::now() < deadline {
            if rex.processor_thread.try_lock().is_some_and(|handle| handle.is_none()) {
                handle_lock_available = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        drop(shape_guard);
        drop(trace_guard);
        let prompt = timely.is_ok();
        let result = timely.unwrap_or_else(|_| received.recv_timeout(Duration::from_secs(10)).unwrap());
        worker.join().unwrap();
        std::fs::remove_file(path).unwrap();
        drop(rex);
        assert_reaped(weak);
        assert!(prompt, "replay waited for the active processor before returning the DMA error");
        assert!(handle_lock_available, "cleanup held the processor handle lock while joining");
        assert_dma_replay_error(result);
    }

    #[test]
    fn replay_dma_error_reaps_active_processor_after_fifo_drains() {
        let board = replay_board(None, false).unwrap();
        let rex = Arc::clone(board.0.as_ref().unwrap());
        let weak = Arc::downgrade(&rex);
        rex.wait_idle();
        let shape_guard = rex.seen_shapes.lock();
        let go = (REX3_BASE | super::super::REX3_DRAWMODE0 | 0x800) as u64;
        assert_eq!(rex.write32(go, 0x6), BUS_OK);
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while !rex.gfxbusy.load(Ordering::Acquire) && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(rex.gfxbusy.load(Ordering::Acquire), "processor did not start");
        let mut queued = 0;
        while rex.gfifo.try_push(super::super::REX3_DRAWMODE0, 0) {
            queued += 1;
        }
        assert!(queued >= super::super::GFIFO_DEPTH - 2, "FIFO was not full");
        let path = invalid_dma_recording("active-full-fifo");
        let worker_path = path.clone();
        let (sent, received) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let result = replay_with_board(worker_path.to_str().unwrap(), None, |_, _| Ok(board), spawn_cleanup);
            let _ = sent.send(result);
        });
        let timely = received.recv_timeout(Duration::from_secs(2));
        drop(shape_guard);
        let result = timely.expect("replay blocked while requesting exit on a full FIFO");
        worker.join().unwrap();
        std::fs::remove_file(path).unwrap();
        assert_dma_replay_error(result);
        drop(rex);
        assert_reaped(weak);
    }

    #[test]
    fn replay_dma_error_returns_after_processor_failure() {
        let board = replay_board(None, false).unwrap();
        let rex = Arc::clone(board.0.as_ref().unwrap());
        let weak = Arc::downgrade(&rex);
        rex.wait_idle();
        rex.gfifo_push(super::super::GFIFO_EXIT, 0);
        rex.processor_thread.lock().take().unwrap().join().unwrap();
        rex.gfxbusy.store(true, Ordering::Release);
        let failed = std::thread::spawn(|| panic!("injected processor failure"));
        while !failed.is_finished() {
            std::thread::yield_now();
        }
        *rex.processor_thread.lock() = Some(failed);
        let path = invalid_dma_recording("failed-processor");
        let worker_path = path.clone();
        let (sent, received) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let result = replay_with_board(worker_path.to_str().unwrap(), None, |_, _| Ok(board), spawn_cleanup);
            let _ = sent.send(result);
        });
        let timely = received.recv_timeout(Duration::from_secs(2));
        let prompt = timely.is_ok();
        if !prompt {
            rex.gfxbusy.store(false, Ordering::Release);
        }
        let result = timely.unwrap_or_else(|_| received.recv_timeout(Duration::from_secs(10)).unwrap());
        worker.join().unwrap();
        std::fs::remove_file(path).unwrap();
        drop(rex);
        assert_reaped(weak);
        assert!(prompt, "replay waited for a failed processor before returning the DMA error");
        assert_dma_replay_error(result);
    }

    #[test]
    fn replay_dma_error_reaps_failed_processor_with_full_fifo() {
        let board = replay_board(None, false).unwrap();
        let rex = Arc::clone(board.0.as_ref().unwrap());
        let weak = Arc::downgrade(&rex);
        rex.wait_idle();
        rex.gfifo_push(super::super::GFIFO_EXIT, 0);
        rex.processor_thread.lock().take().unwrap().join().unwrap();
        let failed = std::thread::spawn(|| panic!("injected processor failure with a full FIFO"));
        while !failed.is_finished() {
            std::thread::yield_now();
        }
        *rex.processor_thread.lock() = Some(failed);
        for _ in 0..super::super::GFIFO_DEPTH - 1 {
            assert!(rex.gfifo.try_push(super::super::REX3_DRAWMODE0, 0));
        }
        assert!(!rex.gfifo.try_push(super::super::REX3_DRAWMODE0, 0));
        rex.gfxbusy.store(true, Ordering::Release);
        let path = invalid_dma_recording("failed-full-fifo");
        let worker_path = path.clone();
        let (sent, received) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let result = replay_with_board(worker_path.to_str().unwrap(), None, |_, _| Ok(board), spawn_cleanup);
            let _ = sent.send(result);
        });
        let result = received.recv_timeout(Duration::from_secs(2))
            .expect("replay blocked on the failed processor's full FIFO");
        worker.join().unwrap();
        std::fs::remove_file(path).unwrap();
        assert_dma_replay_error(result);
        drop(rex);
        assert_reaped(weak);
    }

    #[test]
    fn replay_reports_both_access_and_shutdown_spawn_errors() {
        let board = replay_board(None, false).unwrap();
        let weak = Arc::downgrade(board.0.as_ref().unwrap());
        let path = invalid_dma_recording("spawn-error");
        let (tasks, receive_task) = std::sync::mpsc::channel();
        let result = replay_with_board(path.to_str().unwrap(), None, |_, _| Ok(board), move |task| {
            tasks.send(task).unwrap();
            Err(std::io::Error::other("injected shutdown spawn failure"))
        });
        let error = result.expect_err("an invalid DMA port must not succeed");
        assert!(error.contains("record 0") && error.contains("dma_read64_bulk")
            && error.contains("status 0x800001c"), "{error}");
        assert!(error.contains("shutdown") && error.contains("injected shutdown spawn failure"), "{error}");
        let cleanup = receive_task.recv_timeout(Duration::from_secs(2)).unwrap();
        std::thread::spawn(cleanup).join().unwrap();
        std::fs::remove_file(path).unwrap();
        assert_reaped(weak);
    }

    #[test]
    fn replay_pio_access_errors_are_terminal_and_reaped() {
        for (name, record, operation) in [
            ("pio-read", Rec::Read { bits: 8, off: super::super::REX3_DRAWMODE0, val: 0 }, "read"),
            ("pio-write", Rec::Write { bits: 8, off: super::super::REX3_DRAWMODE0, val: 0 }, "write"),
        ] {
            let board = replay_board(None, false).unwrap();
            let weak = Arc::downgrade(board.0.as_ref().unwrap());
            let path = replay_recording(name, &[record, Rec::Hash([0; 32])]);
            let error = replay_with_board(path.to_str().unwrap(), None, |_, _| Ok(board), spawn_cleanup)
                .expect_err("a failed PIO access must not succeed");
            assert!(error.contains(&format!("record 0 {operation}")) && error.contains("status 0x800001c"), "{error}");
            std::fs::remove_file(path).unwrap();
            assert_reaped(weak);
        }
    }

    #[test]
    fn replay_error_finishes_trace_during_background_cleanup() {
        let board = replay_board(None, false).unwrap();
        let weak = Arc::downgrade(board.0.as_ref().unwrap());
        let path = invalid_dma_recording("trace-error");
        let trace_path = path.with_extension("trace");
        board.rex().start_trace(trace_path.to_str().unwrap(), super::super::trace::TRACE_ALL).unwrap();
        let result = replay_with_board(path.to_str().unwrap(), None, |_, _| Ok(board), spawn_cleanup);
        assert_dma_replay_error(result);
        assert_reaped(weak);
        assert!(std::fs::read_to_string(&trace_path).unwrap().contains("REX3 trace started"));
        std::fs::remove_file(path).unwrap();
        std::fs::remove_file(trace_path).unwrap();
    }

    #[test]
    fn replay_output_error_is_preserved_and_reaped() {
        let board = replay_board(None, false).unwrap();
        let weak = Arc::downgrade(board.0.as_ref().unwrap());
        board.rex().wait_idle();
        let path = replay_recording("output-error", &[Rec::Hash(board.rex().rec_state_hash())]);
        let output = path.to_str().unwrap();
        let error = replay_with_board(output, Some(output), |_, _| Ok(board), spawn_cleanup)
            .expect_err("a recording file cannot also be an output directory");
        assert!(error.contains(output), "{error}");
        std::fs::remove_file(path).unwrap();
        assert_reaped(weak);
    }

    #[test]
    fn replay_read_mismatch_remains_strict_after_synchronous_stop() {
        let board = replay_board(None, false).unwrap();
        let weak = Arc::downgrade(board.0.as_ref().unwrap());
        board.rex().wait_idle();
        let off = super::super::REX3_DRAWMODE0;
        let actual = board.rex().read32((REX3_BASE | off) as u64);
        assert_eq!(actual.status, BUS_OK);
        let hash = board.rex().rec_state_hash();
        let path = replay_recording("read-mismatch", &[
            Rec::Read { bits: 32, off, val: (actual.data ^ 1) as u64 },
            Rec::Hash(hash),
        ]);
        let error = replay_with_board(path.to_str().unwrap(), None, |_, _| Ok(board), |_| {
            panic!("a completed replay must stop synchronously")
        }).expect_err("a deterministic bus-read mismatch must fail");
        assert!(error.contains("0 of 1 checkpoints failed") && error.contains("1 bus reads differ"), "{error}");
        std::fs::remove_file(path).unwrap();
        assert_eq!(weak.strong_count(), 0, "completed replay retained its board");
    }

    #[test]
    fn replay_success_drains_checks_and_stops_synchronously() {
        let mut source = replay_board(None, false).unwrap();
        let go = (REX3_BASE | super::super::REX3_DRAWMODE0 | 0x800) as u64;
        assert_eq!(source.rex().write32(go, 0x6), BUS_OK);
        source.rex().wait_idle();
        let hash = source.rex().rec_state_hash();
        source.rex().stop();
        source.stopped();

        let board = replay_board(None, false).unwrap();
        let rex = Arc::clone(board.0.as_ref().unwrap());
        let weak = Arc::downgrade(&rex);
        let shape_guard = rex.seen_shapes.lock();
        assert_eq!(rex.write32(go, 0x6), BUS_OK);
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while !rex.gfxbusy.load(Ordering::Acquire) && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(rex.gfxbusy.load(Ordering::Acquire), "processor did not start");
        let path = replay_recording("successful-drain", &[Rec::Hash(hash)]);
        let worker_path = path.clone();
        let (sent, received) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let result = replay_with_board(worker_path.to_str().unwrap(), None, |_, _| Ok(board), |_| {
                panic!("successful replay must not start asynchronous cleanup")
            });
            let _ = sent.send(result);
        });
        let early = received.recv_timeout(Duration::from_millis(100));
        drop(shape_guard);
        assert!(early.is_err(), "successful replay returned without draining the processor");
        let result = received.recv_timeout(Duration::from_secs(10)).unwrap();
        worker.join().unwrap();
        std::fs::remove_file(path).unwrap();
        result.expect("strict checkpoint replay should succeed");
        assert!(!rex.is_running());
        assert!(rex.processor_thread.lock().is_none());
        assert!(rex.refresh_thread.lock().is_none());
        drop(rex);
        assert_eq!(weak.strong_count(), 0, "successful replay did not release its board synchronously");
    }

    #[test]
    fn review_replay_file_reports_dma_access_error_before_missing_hash() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("rex3-replay-dma-error-{}.rec", std::process::id()));
        let path = path.to_str().unwrap();
        let mut recorder = Recorder::create(path).unwrap();
        recorder.put(&Rec::BulkRead { off: super::super::REX3_HOSTRW0, vals: vec![0] });
        recorder.finish().unwrap();
        let result = replay(path, None);
        let _ = std::fs::remove_file(path);
        let error = result.expect_err("an invalid DMA port must fail");
        assert!(error.contains("record 0") && error.contains("dma_read64_bulk")
            && error.contains("status 0x800001c"), "{error}");
    }

    #[test]
    #[ignore = "optional one-file replay; set REX3_REPLAY and run with --ignored"]
    fn rex3_replay_file() {
        let path = std::env::var("REX3_REPLAY").expect("set REX3_REPLAY to a recording");
        let png = std::env::var("REX3_REPLAY_OUT").ok();
        replay(&path, png.as_deref()).unwrap();
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

#[cfg(test)]
#[path = "rex3_corpus.rs"]
mod corpus;

#[cfg(test)]
#[path = "rex3_corpus_io.rs"]
mod corpus_io;
