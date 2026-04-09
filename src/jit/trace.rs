//! Lightweight binary state trace for differential JIT debugging.
//!
//! Enable with `IRIS_JIT_TRACE=<path>`. After every BATCH_SIZE instructions,
//! writes a 48-byte record of key architectural state. Run once with JIT,
//! once interpreter-only, diff with tools/diff-trace.py to find the first
//! state divergence.
//!
//! Read-only access to executor state. No interaction with sync/snapshot/verify.

use std::fs::File;
use std::io::{self, BufWriter, Write};

/// Fixed-size trace record. 48 bytes, written as raw bytes (little-endian).
#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct TraceRecord {
    pub insn_count: u64,
    pub pc: u64,
    pub cp0_count: u64,
    pub cp0_status: u32,
    pub cp0_cause: u32,
    pub in_delay_slot: u8,
    pub _pad: [u8; 7],
    pub gpr_hash: u64,
}

const _: () = assert!(std::mem::size_of::<TraceRecord>() == 48);

impl TraceRecord {
    /// XOR-fold all 32 GPRs into a single u64 fingerprint.
    pub fn hash_gprs(gpr: &[u64; 32]) -> u64 {
        let mut h: u64 = 0;
        for &v in gpr.iter() {
            h ^= v;
        }
        // Rotate between XORs to reduce collision on symmetric patterns
        h ^= h.rotate_left(17);
        h
    }
}

/// Buffered binary trace writer. Call `write_record` after each batch.
pub struct TraceWriter {
    out: BufWriter<File>,
    records: u64,
}

impl TraceWriter {
    /// Open a trace file. Returns None if the path is empty or open fails.
    pub fn new(path: &str) -> io::Result<Self> {
        let file = File::create(path)?;
        eprintln!("iris: JIT trace enabled, writing to {}", path);
        Ok(Self {
            out: BufWriter::with_capacity(64 * 1024, file),
            records: 0,
        })
    }

    /// Try to create from the IRIS_JIT_TRACE env var. Returns None if not set.
    pub fn from_env() -> Option<Self> {
        let path = std::env::var("IRIS_JIT_TRACE").ok()?;
        if path.is_empty() { return None; }
        match Self::new(&path) {
            Ok(w) => Some(w),
            Err(e) => {
                eprintln!("iris: failed to open trace file '{}': {}", path, e);
                None
            }
        }
    }

    /// Write a trace record. This is a raw byte write, no serialization overhead.
    pub fn write_record(&mut self, rec: &TraceRecord) {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                rec as *const TraceRecord as *const u8,
                std::mem::size_of::<TraceRecord>(),
            )
        };
        // Ignore write errors (trace is best-effort, don't crash the emulator)
        let _ = self.out.write_all(bytes);
        self.records += 1;
    }

    /// Flush and report stats.
    pub fn finish(&mut self) {
        let _ = self.out.flush();
        eprintln!("iris: JIT trace finished, {} records written", self.records);
    }
}

impl Drop for TraceWriter {
    fn drop(&mut self) {
        self.finish();
    }
}
