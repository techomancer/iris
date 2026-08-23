//! `j2 dump-pcp`: serialize a `PhysicalCodePage`'s full material state plus
//! its backing 4KB physical memory to a single file, for later offline
//! analysis by the standalone `jitv2_pcp_dump` tool (`src/bin/jitv2_pcp_dump.rs`).
//!
//! Distinct from the `jitv2_corpus_dump` feature's `dump_corpus_snapshot`
//! (`comp.rs`): that one is a raw-memory-only capture, built to feed the
//! analyzer/codegen offline during development against real captured pages,
//! and is compiled out entirely unless the `jitv2_corpus_dump` feature is
//! on. This dump is unconditional (always compiled in — a monitor-console
//! diagnostic, not a dev-build-only tool) and captures the full
//! `PhysicalCodePage` state (bitmaps, gen, pinned FR mode) alongside the
//! memory, specifically for debugging a live divergence/panic where the
//! *page's own bookkeeping*, not just its bytes, might be the story (e.g.
//! "was this offset actually compiled/denied/requested at the moment of
//! the panic, and against which generation").
//!
//! File format (little-endian, fixed layout, no compression — a handful of
//! these will ever exist on disk at once):
//!
//! ```text
//! offset  size  field
//! 0       8     magic = b"IRISPCP1" (also encodes format version 1)
//! 8       4     pfn (u32)
//! 12      8     current_gen (u64) — the page's live generation at dump time
//! 20      8     entry_gen (u64) — generation `func`/`compiled` were last published against
//! 28      1     fr1 (u8, 0 or 1) — pinned FR mode (PhysicalCodePage::fr1)
//! 29      7     padding (zero)
//! 36      128   requested bitmap (16 x u64 LE)
//! 164     128   compiled bitmap (16 x u64 LE)
//! 292     128   denied bitmap, RAW/inverted sense (16 x u64 LE) — 1 = allowed
//! 420     4096  raw page words (1024 x u32 LE) — ENTRIES_PER_PAGE words, as read off the bus
//! ------
//! 4516  total
//! ```

use crate::jitv2::{PhysicalCodePage, BITMAP_WORDS, ENTRIES_PER_PAGE, PAGE_SIZE};
use crate::traits::BusDevice;

pub const MAGIC: &[u8; 8] = b"IRISPCP1";

/// Everything a `.pcp` dump file carries, in parsed form — shared between
/// the writer here and the standalone tool's reader (`jitv2_pcp_dump.rs`
/// duplicates this struct rather than depending on the `iris` lib's private
/// `mips_exec`-adjacent glue; this module itself is `pub` and reused as a
/// library dependency by that binary, same pattern as `jitv2_analyze`
/// depending on `iris::jitv2::analyzer`).
pub struct PcpDump {
    pub pfn: u32,
    pub current_gen: u64,
    pub entry_gen: u64,
    pub fr1: bool,
    pub requested: [u64; BITMAP_WORDS],
    pub compiled: [u64; BITMAP_WORDS],
    pub denied_raw: [u64; BITMAP_WORDS],
    pub words: [u32; ENTRIES_PER_PAGE],
}

pub const FILE_LEN: usize = 8 + 4 + 8 + 8 + 1 + 7 + 128 * 3 + (ENTRIES_PER_PAGE * 4);

fn push_bitmap(buf: &mut Vec<u8>, bm: &[u64; BITMAP_WORDS]) {
    for &w in bm {
        buf.extend_from_slice(&w.to_le_bytes());
    }
}

impl PcpDump {
    /// Capture `page`'s current state plus a fresh read of its backing 4KB
    /// physical page off `bus`. Not a seqlock snapshot (§13.3's compile-time
    /// discipline doesn't apply here — this is a best-effort diagnostic
    /// capture of "what does this look like right now," torn reads against
    /// a live SMC race are an acceptable, clearly-labeled-by-context
    /// possibility for a debugging dump, not a correctness-critical path).
    /// Returns `Err` with the failing physical address if any word read fails.
    pub fn capture(page: &PhysicalCodePage, bus: &dyn BusDevice) -> Result<Self, u32> {
        let phys_base = page.pfn * PAGE_SIZE;
        let mut words = [0u32; ENTRIES_PER_PAGE];
        for (i, w) in words.iter_mut().enumerate() {
            let addr = phys_base + (i as u32) * 4;
            let r = bus.read32(addr);
            if !r.is_ok() {
                return Err(addr);
            }
            *w = r.data;
        }
        Ok(Self {
            pfn: page.pfn,
            current_gen: page.current_gen(),
            entry_gen: page.entry_gen(),
            fr1: page.is_fr1(),
            requested: page.snapshot_requested(),
            compiled: page.snapshot_compiled(),
            denied_raw: page.snapshot_denied_raw(),
            words,
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(FILE_LEN);
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&self.pfn.to_le_bytes());
        buf.extend_from_slice(&self.current_gen.to_le_bytes());
        buf.extend_from_slice(&self.entry_gen.to_le_bytes());
        buf.push(self.fr1 as u8);
        buf.extend_from_slice(&[0u8; 7]);
        push_bitmap(&mut buf, &self.requested);
        push_bitmap(&mut buf, &self.compiled);
        push_bitmap(&mut buf, &self.denied_raw);
        for &w in &self.words {
            buf.extend_from_slice(&w.to_le_bytes());
        }
        debug_assert_eq!(buf.len(), FILE_LEN);
        buf
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < FILE_LEN {
            return Err(format!("truncated dump: got {} bytes, expected {}", bytes.len(), FILE_LEN));
        }
        if &bytes[0..8] != MAGIC {
            return Err(format!("bad magic: expected {:?}, got {:?}", MAGIC, &bytes[0..8]));
        }
        let mut off = 8usize;
        let read_u32 = |off: &mut usize| -> u32 {
            let v = u32::from_le_bytes(bytes[*off..*off + 4].try_into().unwrap());
            *off += 4;
            v
        };
        let read_u64 = |off: &mut usize| -> u64 {
            let v = u64::from_le_bytes(bytes[*off..*off + 8].try_into().unwrap());
            *off += 8;
            v
        };
        let pfn = read_u32(&mut off);
        let current_gen = read_u64(&mut off);
        let entry_gen = read_u64(&mut off);
        let fr1 = bytes[off] != 0;
        off += 1 + 7; // fr1 byte + padding
        let read_bitmap = |off: &mut usize| -> [u64; BITMAP_WORDS] {
            std::array::from_fn(|_| read_u64(off))
        };
        let requested = read_bitmap(&mut off);
        let compiled = read_bitmap(&mut off);
        let denied_raw = read_bitmap(&mut off);
        let mut words = [0u32; ENTRIES_PER_PAGE];
        for w in words.iter_mut() {
            *w = read_u32(&mut off);
        }
        debug_assert_eq!(off, FILE_LEN);
        Ok(Self { pfn, current_gen, entry_gen, fr1, requested, compiled, denied_raw, words })
    }

    /// Bit test helper, mirroring `PhysicalCodePage`'s own accessor shapes —
    /// `denied_raw` is inverted (1 = allowed), same as the live field.
    pub fn is_requested(&self, offset: usize) -> bool {
        self.requested[offset >> 6] & (1u64 << (offset & 63)) != 0
    }
    pub fn is_compiled(&self, offset: usize) -> bool {
        self.compiled[offset >> 6] & (1u64 << (offset & 63)) != 0
    }
    pub fn is_denylisted(&self, offset: usize) -> bool {
        self.denied_raw[offset >> 6] & (1u64 << (offset & 63)) == 0
    }
}

/// Default output path for `j2 dump-pcp` when no explicit path is given —
/// timestamped so repeated dumps during one debugging session never
/// silently clobber each other.
pub fn default_dump_path(pfn: u32) -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("pcp_dump_pfn_{:08x}_{}.bin", pfn, now)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_preserves_every_field() {
        let mut requested = [0u64; BITMAP_WORDS];
        requested[0] = 0x1234;
        let mut compiled = [0u64; BITMAP_WORDS];
        compiled[1] = 0xABCD;
        let mut denied_raw = [u64::MAX; BITMAP_WORDS];
        denied_raw[2] = 0xFFFF_FFFF_FFFF_FFFE;
        let mut words = [0u32; ENTRIES_PER_PAGE];
        words[0] = 0xDEAD_BEEF;
        words[1023] = 0x1;

        let dump = PcpDump {
            pfn: 0x1234,
            current_gen: 42,
            entry_gen: 41,
            fr1: true,
            requested,
            compiled,
            denied_raw,
            words,
        };

        let bytes = dump.to_bytes();
        assert_eq!(bytes.len(), FILE_LEN);
        let back = PcpDump::from_bytes(&bytes).unwrap();

        assert_eq!(back.pfn, dump.pfn);
        assert_eq!(back.current_gen, dump.current_gen);
        assert_eq!(back.entry_gen, dump.entry_gen);
        assert_eq!(back.fr1, dump.fr1);
        assert_eq!(back.requested, dump.requested);
        assert_eq!(back.compiled, dump.compiled);
        assert_eq!(back.denied_raw, dump.denied_raw);
        assert_eq!(back.words, dump.words);
    }

    #[test]
    fn from_bytes_rejects_bad_magic() {
        let mut bytes = vec![0u8; FILE_LEN];
        bytes[0..8].copy_from_slice(b"NOTAPCP!");
        assert!(PcpDump::from_bytes(&bytes).is_err());
    }

    #[test]
    fn from_bytes_rejects_truncated_input() {
        let bytes = vec![0u8; 10];
        assert!(PcpDump::from_bytes(&bytes).is_err());
    }
}
