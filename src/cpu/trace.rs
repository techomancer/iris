//! Per-instruction execution trace capture and replay.
//!
//! Records exactly what the interpreter actually executed — pc, raw
//! instruction word, and full pre-execution architectural state — for every
//! instruction, so a JIT (or any other alternate execution engine) can be
//! verified offline afterward: replay each recorded instruction against its
//! recorded pre-state, run it through the JIT, and diff the result against
//! the trace's *next* record (which is the authoritative post-state, since
//! it's the pre-state of whatever the interpreter executed next).
//!
//! This sidesteps the double-execution problem live lockstep would have:
//! MMIO devices (SCSI, REX3, HPC3, ...) have real, stateful side effects on
//! loads/stores, so running the JIT a second time immediately after the
//! interpreter already executed the same instruction would fire those side
//! effects twice. Trace-then-replay only ever executes each instruction once
//! against real hardware state; the JIT verification pass runs later,
//! entirely offline, against snapshotted register state with no bus access
//! at all.
//!
//! Deliberately a flat, fixed-size, `#[repr(C)]` record with no compression
//! or serde — this is a debugging tool, not a shipped artifact, and a raw
//! memcpy read/write is the simplest thing that lets a boot trace (tens of
//! millions of instructions) stream to/from disk without going through an
//! allocator per record.

use std::io::{self, BufReader, BufWriter, Read, Write};

/// One instruction's pre-execution state and identity. `pc`/`raw` are
/// carried outside `state` (rather than only trusting `state.pc`) so a
/// reader can filter/seek by pc or opcode without decoding the full state
/// blob — `state.pc` is always equal to `pc` by construction (see
/// `TraceRecord::capture`).
///
/// A record's *post*-state is not stored: it's the next record's `state`
/// (the interpreter's `step()` produced exactly the state the following
/// instruction sees as its pre-state). The last record in a trace has no
/// recorded post-state — verification only compares complete
/// (record, next_record) pairs.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TraceRecord {
    pub pc: u64,
    pub raw: u32,
    _pad: u32,
    pub state: CoreState,
}

/// Snapshot of the architectural fields an alternate execution engine
/// (jitv2's `Codegen`-compiled functions today) can observe or mutate.
/// Mirrors `jitv2::equiv_test`'s test-only `CoreSnapshot` — this is that
/// same set of fields, promoted out of `#[cfg(test)]` so both the live
/// recorder and the offline verifier can use it. Extend alongside new
/// emitter coverage (CP0 regs, HI/LO already covered, ...), matching
/// whatever `jitv2/codegen.rs`'s emitters can actually touch.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoreState {
    pub gpr: [u64; 32],
    pub pc: u64,
    pub hi: u64,
    pub lo: u64,
    pub cp0_epc: u64,
    pub cp0_badvaddr: u64,
    pub cp0_cause: u32,
    pub cp0_status: u32,
    pub fpr: [u64; 32],
    pub fpu_fcsr: u32,
    pub fpu_fccr: u32,
    pub fpu_fexr: u32,
    pub fpu_fenr: u32,
    /// Whether the recorded instruction is a branch/jump's delay slot —
    /// `MipsCore::in_delay_slot` at capture time. Needed by
    /// `jitv2-verify`'s `deliver_exception` call for a record whose
    /// instruction traps: without it, every trap looks like it happened
    /// outside a delay slot (EPC/Cause.BD both computed wrong for the
    /// records that actually were slots) — `MipsCore::new()`'s default
    /// (`false`) is silently wrong for those, not just "unknown".
    pub in_delay_slot: bool,
}

impl CoreState {
    pub fn capture(core: &crate::cpu::mips_core::MipsCore) -> Self {
        Self {
            gpr: core.gpr, pc: core.pc, hi: core.hi, lo: core.lo,
            cp0_epc: core.cp0_epc, cp0_badvaddr: core.cp0_badvaddr,
            cp0_cause: core.cp0_cause, cp0_status: core.cp0_status,
            fpr: core.fpr, fpu_fcsr: core.fpu_fcsr,
            fpu_fccr: core.fpu_fccr, fpu_fexr: core.fpu_fexr, fpu_fenr: core.fpu_fenr,
            in_delay_slot: core.in_delay_slot,
        }
    }
}

impl TraceRecord {
    pub fn capture(pc: u64, raw: u32, core: &crate::cpu::mips_core::MipsCore) -> Self {
        Self { pc, raw, _pad: 0, state: CoreState::capture(core) }
    }

    /// Build a record directly from a `pc`/`raw`/`state` triple — used by
    /// tests and tools (e.g. `jitv2-verify`'s own test suite) that construct
    /// trace records synthetically rather than capturing from a live
    /// `MipsCore`. `state.pc` should normally equal `pc`.
    pub fn new(pc: u64, raw: u32, state: CoreState) -> Self {
        Self { pc, raw, _pad: 0, state }
    }

    const SIZE: usize = std::mem::size_of::<TraceRecord>();

    fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self as *const Self as *const u8, Self::SIZE) }
    }

    fn from_bytes(buf: &[u8; Self::SIZE]) -> Self {
        unsafe { std::ptr::read_unaligned(buf.as_ptr() as *const Self) }
    }
}

/// Sequential trace writer. Buffered — `TraceRecord` is ~600 bytes and a
/// boot-length trace is tens of millions of records, so unbuffered
/// one-syscall-per-record writes would dominate capture overhead.
pub struct TraceWriter {
    out: BufWriter<std::fs::File>,
    count: u64,
}

impl TraceWriter {
    pub fn create(path: &std::path::Path) -> io::Result<Self> {
        let file = std::fs::File::create(path)?;
        Ok(Self { out: BufWriter::with_capacity(1 << 20, file), count: 0 })
    }

    pub fn push(&mut self, record: &TraceRecord) -> io::Result<()> {
        self.out.write_all(record.as_bytes())?;
        self.count += 1;
        Ok(())
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    pub fn flush(&mut self) -> io::Result<()> {
        self.out.flush()
    }
}

/// Sequential trace reader. Yields records in capture order via `next()`
/// (an `Iterator`-shaped API, not literally `Iterator` since reads are
/// fallible — `io::Result` per record).
pub struct TraceReader {
    input: BufReader<std::fs::File>,
}

impl TraceReader {
    pub fn open(path: &std::path::Path) -> io::Result<Self> {
        let file = std::fs::File::open(path)?;
        Ok(Self { input: BufReader::with_capacity(1 << 20, file) })
    }

    /// Read the next record, or `Ok(None)` at a clean end-of-file. A
    /// truncated final record (partial write from a killed capture) also
    /// reads as `Ok(None)` — `read_exact`'s `UnexpectedEof` is treated as
    /// "no more complete records" rather than propagated, since a trace
    /// file's whole point is being safe to read after the writer died
    /// mid-capture (e.g. the guest was still running when the emulator was
    /// killed).
    pub fn next(&mut self) -> io::Result<Option<TraceRecord>> {
        let mut buf = [0u8; TraceRecord::SIZE];
        match self.input.read_exact(&mut buf) {
            Ok(()) => Ok(Some(TraceRecord::from_bytes(&buf))),
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Fast-forward past `n` records without reading them — an O(1) file
    /// seek (records are fixed-size), not `n` calls to `next()`. Lets a
    /// caller jump into the middle of a multi-GB trace (e.g. to inspect the
    /// records around a specific pc found via a first pass, or to resume a
    /// long verify run partway through) without paying for every record on
    /// the way there.
    pub fn skip_records(&mut self, n: u64) -> io::Result<()> {
        use std::io::Seek;
        self.input.seek_relative((n * TraceRecord::SIZE as u64) as i64)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_state(seed: u64) -> CoreState {
        let mut gpr = [0u64; 32];
        for (i, g) in gpr.iter_mut().enumerate() { *g = seed.wrapping_add(i as u64); }
        CoreState {
            gpr, pc: seed, hi: seed + 1, lo: seed + 2,
            cp0_epc: seed + 3, cp0_badvaddr: seed + 4,
            cp0_cause: seed as u32, cp0_status: seed as u32,
            fpr: [seed; 32], fpu_fcsr: seed as u32,
            fpu_fccr: seed as u32, fpu_fexr: seed as u32, fpu_fenr: seed as u32,
            in_delay_slot: seed % 2 == 0,
        }
    }

    #[test]
    fn round_trips_records_in_order() {
        let dir = std::env::temp_dir().join(format!("iris_trace_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("round_trip.trace");

        let mut w = TraceWriter::create(&path).unwrap();
        let records: Vec<TraceRecord> = (0..1000u64)
            .map(|i| TraceRecord { pc: 0x8000_0000 + i * 4, raw: i as u32, _pad: 0, state: sample_state(i) })
            .collect();
        for r in &records { w.push(r).unwrap(); }
        assert_eq!(w.count(), 1000);
        w.flush().unwrap();
        drop(w);

        let mut r = TraceReader::open(&path).unwrap();
        for expected in &records {
            let got = r.next().unwrap().expect("record must be present");
            assert_eq!(got, *expected);
        }
        assert!(r.next().unwrap().is_none(), "reader must report clean EOF after the last record");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn truncated_final_record_reads_as_clean_eof() {
        let dir = std::env::temp_dir().join(format!("iris_trace_test_trunc_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("truncated.trace");

        {
            let mut w = TraceWriter::create(&path).unwrap();
            w.push(&TraceRecord { pc: 0x8000_0000, raw: 0, _pad: 0, state: sample_state(1) }).unwrap();
            w.flush().unwrap();
        }
        // Truncate mid-record to simulate a capture killed mid-write.
        let full_len = std::fs::metadata(&path).unwrap().len();
        let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        f.set_len(full_len - 10).unwrap();

        let mut r = TraceReader::open(&path).unwrap();
        assert!(r.next().unwrap().is_none(), "a partial trailing record must read as EOF, not an error");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
