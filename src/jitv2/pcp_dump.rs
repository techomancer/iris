//! `j2 dumppcp` / `j2 corpus`: serialize a `PhysicalCodePage`'s full
//! material state plus its backing 4KB physical memory to a single file, for
//! later offline analysis by the standalone `jitv2_pcp_dump` tool
//! (`src/bin/jitv2_pcp_dump.rs`) and as the corpus format that
//! `zz_corpus_sizes` (`jitv2/mod.rs`) measures emitted code volume against.
//!
//! Always compiled in — a monitor-console diagnostic, not a dev-build-only
//! tool — and available under **both** `comp.rs` implementations (`j2wp` and
//! the default one-function-per-entry protocol). It captures the full
//! `PhysicalCodePage` state (bitmaps, gen, pinned FR mode) alongside the
//! memory, for debugging a live divergence/panic where the *page's own
//! bookkeeping*, not just its bytes, might be the story ("was this offset
//! actually compiled/denied/requested at the moment of the panic, and
//! against which generation").
//!
//! This replaced the old `jitv2_corpus_dump` Cargo feature, which wrote raw
//! 4KB `pfn_XXXXXXXX_off_XXXX.bin` snapshots from inside `handle_request`
//! (plus a `PhysicalCodePage::saved_bits` bitmap to dedup them) and encoded
//! the single entry offset in the *filename*. Three things were wrong with
//! that: it needed a non-default feature compiled in before the run that
//! would produce the interesting pages, so a corpus could only ever be
//! captured by knowing in advance you wanted one; it recorded one entry
//! offset per file, so a page with many entry points became many near-
//! duplicate 4KB files; and it wrote to the filesystem from the compile
//! worker. `j2 corpus` instead dumps on demand, after the fact, from the
//! pages already sitting in the live pcp cache — which is where this data
//! has been cached all along — and the `requested` bitmap below carries
//! *every* entry offset for a page in the one file.
//!
//! # Format version 2 (`IRISPCP2`)
//!
//! Version 1 (`IRISPCP1`) is still accepted by [`PcpDump::from_bytes`]; its
//! files are shorter and simply have the v2-only fields defaulted (see
//! [`V1_FILE_LEN`]). Writers always emit v2.
//!
//! ```text
//! offset  size  field
//! 0       8     magic = b"IRISPCP2" (also encodes the format version)
//! 8       4     pfn (u32)
//! 12      8     current_gen (u64) — the page's live generation at dump time
//! 20      8     entry_gen (u64) — generation `func`/`compiled` were last published against
//! 28      1     fr1 (u8, 0 or 1) — pinned FR mode (PhysicalCodePage::fr1)
//! 29      7     padding (zero)
//! 36      128   requested bitmap (16 x u64 LE)
//! 164     128   compiled bitmap (16 x u64 LE)
//! 292     128   denied bitmap, RAW/inverted sense (16 x u64 LE) — 1 = allowed
//! 420     4096  raw page words (1024 x u32 LE) — ENTRIES_PER_PAGE words, as read off the bus
//!       ---- v2 additions start here (a v1 file ends at 4516) ----
//! 4516    8     call_count (u64) — dispatches into this page's compiled code
//! ------
//! 4524  total
//! ```
//!
//! ## Why `call_count`
//!
//! It weights the corpus, and it is free at capture time (an
//! already-maintained counter, no new bookkeeping). Emitted-code measurements
//! over a corpus (`zz_corpus_sizes`, and every codegen size claim in
//! `rules/jitv2/`) otherwise treat all pages equally, but a page dispatched
//! two million times and one dispatched twice contribute the same one page of
//! bytes to a flat total — which makes a flat byte count a poor proxy for
//! what a change does to the code that actually *runs*. Recording the count
//! lets an offline consumer weight by it, or filter to the hot tail, without
//! re-running the guest.
//!
//! ## Why there is no virtual address here
//!
//! Deliberately: **everything in this format is physical, and must stay that
//! way.** jitv2 compiles position-independent code precisely because one
//! physical page is reused across many processes — the same page can be
//! mapped at many virtual addresses at once, at different ones over time, and
//! at none after a mapping is torn down. There is no "the" virtual address
//! for a page, so recording one would be a value that looks authoritative,
//! is unverifiable offline, and is wrong for every mapping but the one that
//! happened to be live at capture. A `Pfn` is what the JIT keys on and it is
//! what this file carries.
//!
//! (A v2 draft did carry a `vaddr_hint` for symbolization. It was removed
//! before it shipped: `j2 corpus`, the bulk path this format exists to serve,
//! never had an address to record — 0 of 1427 pages in the first real corpus
//! had one — so it was empty exactly where it was supposed to help, and
//! misleading in the single-page case where it was set.)

use crate::jitv2::{PhysicalCodePage, BITMAP_WORDS, ENTRIES_PER_PAGE, PAGE_SIZE};
use crate::traits::BusDevice;

/// Format version 2 magic. Writers always emit this.
pub const MAGIC: &[u8; 8] = b"IRISPCP2";

/// Format version 1 magic, still accepted on read (see [`PcpDump::from_bytes`]).
/// A v1 file carries no `call_count`; it reads back as 0, the same "unknown"
/// value a v2 writer uses when the counter is compiled out.
pub const MAGIC_V1: &[u8; 8] = b"IRISPCP1";

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
    /// Dispatches into this page's compiled code, for weighting a corpus by
    /// how hot the page actually was (see the module doc). 0 = unknown: a v1
    /// file, or a build where the counter is `developer`-gated away.
    pub call_count: u64,
}

/// Length of a v1 (`IRISPCP1`) file — also the offset at which v2's own
/// fields begin.
pub const V1_FILE_LEN: usize = 8 + 4 + 8 + 8 + 1 + 7 + 128 * 3 + (ENTRIES_PER_PAGE * 4);

/// Length of a v2 (`IRISPCP2`) file: v1's layout plus `call_count`.
pub const FILE_LEN: usize = V1_FILE_LEN + 8;

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
            entry_gen: page.dump_entry_gen(),
            fr1: page.dump_fr1(),
            requested: page.dump_requested(),
            compiled: page.dump_compiled(),
            denied_raw: page.dump_denied_raw(),
            words,
            call_count: page.dump_call_count(),
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
        debug_assert_eq!(buf.len(), V1_FILE_LEN);
        buf.extend_from_slice(&self.call_count.to_le_bytes());
        debug_assert_eq!(buf.len(), FILE_LEN);
        buf
    }

    /// Parse a dump file. Accepts both `IRISPCP2` and the older `IRISPCP1`
    /// (whose shorter layout ends at [`V1_FILE_LEN`]); a v1 file's missing
    /// `call_count` reads back as 0, the same "unknown" value a
    /// v2 writer uses when it has nothing to record, so consumers need no
    /// version check of their own.
    ///
    /// **Trailing bytes past the expected length are ignored, by design.**
    /// Every field has a fixed offset from the start, so a longer file parses
    /// identically to a correct one. This matters because `IRISPCP2` briefly
    /// carried an extra trailing `vaddr_hint` u64 during development (removed
    /// before release — see the module doc), and corpora captured in that
    /// window are 8 bytes longer while being bit-identical in every field
    /// this reader consumes. Rejecting on exact length would have thrown away
    /// hours of capture for no benefit. The magic still pins the layout, so a
    /// genuinely different format cannot slip through this way.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 8 {
            return Err(format!("truncated dump: got {} bytes, too short for a magic", bytes.len()));
        }
        let is_v2 = &bytes[0..8] == MAGIC;
        if !is_v2 && &bytes[0..8] != MAGIC_V1 {
            return Err(format!("bad magic: expected {:?} or {:?}, got {:?}", MAGIC, MAGIC_V1, &bytes[0..8]));
        }
        let want = if is_v2 { FILE_LEN } else { V1_FILE_LEN };
        if bytes.len() < want {
            return Err(format!("truncated dump: got {} bytes, expected {}", bytes.len(), want));
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
        debug_assert_eq!(off, V1_FILE_LEN);
        // v1 stops here; its absent `call_count` defaults to 0 ("unknown").
        let call_count = if is_v2 {
            let c = read_u64(&mut off);
            debug_assert_eq!(off, FILE_LEN);
            c
        } else {
            0
        };
        Ok(Self { pfn, current_gen, entry_gen, fr1, requested, compiled, denied_raw, words, call_count })
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

/// Default directory `j2 corpus` writes its per-page `.pcp` files into.
/// Same name the removed `jitv2_corpus_dump` feature used, so existing notes
/// in `rules/jitv2/` that refer to `jitv2_corpus/` still point at the right
/// place.
pub const CORPUS_DIR: &str = "jitv2_corpus";

/// Default output path for `j2 dumppcp` when no explicit path is given —
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
            call_count: 2_177_413,
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
        assert_eq!(back.call_count, dump.call_count);
    }

    /// A v1 file must still parse, with the v2-only field defaulting to 0
    /// ("unknown") rather than the read failing or reading past the end of
    /// the shorter buffer. Synthesized by taking a v2 encoding, truncating it
    /// at `V1_FILE_LEN` and stamping the v1 magic — which is exactly the
    /// layout v1 wrote, since v2 only appends.
    #[test]
    fn v1_files_still_parse_with_defaulted_extras() {
        let dump = PcpDump {
            pfn: 0x99,
            current_gen: 7,
            entry_gen: 7,
            fr1: false,
            requested: [0u64; BITMAP_WORDS],
            compiled: [0u64; BITMAP_WORDS],
            denied_raw: [u64::MAX; BITMAP_WORDS],
            words: [0xABCD_1234u32; ENTRIES_PER_PAGE],
            call_count: 12345,
        };
        let mut bytes = dump.to_bytes();
        bytes.truncate(V1_FILE_LEN);
        bytes[0..8].copy_from_slice(MAGIC_V1);

        let back = PcpDump::from_bytes(&bytes).expect("v1 file must still parse");
        assert_eq!(back.pfn, dump.pfn);
        assert_eq!(back.words, dump.words);
        assert_eq!(back.denied_raw, dump.denied_raw);
        // The v2-only field is absent from a v1 file, so it reads as the
        // documented "unknown" value, NOT the value encoded above.
        assert_eq!(back.call_count, 0);
    }

    /// A file with extra trailing bytes past `FILE_LEN` must parse exactly
    /// like a correct one — the real case being corpora captured while
    /// `IRISPCP2` briefly carried a trailing `vaddr_hint` u64 (see
    /// `from_bytes`' doc). Every field is at a fixed offset from the start,
    /// so the extra bytes are simply never read.
    #[test]
    fn trailing_bytes_are_ignored() {
        let dump = PcpDump {
            pfn: 0x8004,
            current_gen: 2050,
            entry_gen: 2050,
            fr1: false,
            requested: [0u64; BITMAP_WORDS],
            compiled: [0x5555u64; BITMAP_WORDS],
            denied_raw: [u64::MAX; BITMAP_WORDS],
            words: [0xDEAD_BEEFu32; ENTRIES_PER_PAGE],
            call_count: 2_177_413,
        };
        let mut bytes = dump.to_bytes();
        // Exactly what the withdrawn draft appended.
        bytes.extend_from_slice(&0x8000_0000_0800_4000u64.to_le_bytes());
        assert_eq!(bytes.len(), FILE_LEN + 8);

        let back = PcpDump::from_bytes(&bytes).expect("longer file must still parse");
        assert_eq!(back.pfn, dump.pfn);
        assert_eq!(back.call_count, dump.call_count);
        assert_eq!(back.compiled, dump.compiled);
        assert_eq!(back.words, dump.words);
    }

    /// A v2 file truncated to less than its full length must be rejected, not
    /// silently treated as a v1 file: v1's magic is what makes a short file
    /// legal, and a v2 magic promises v2's length.
    #[test]
    fn truncated_v2_is_rejected_not_read_as_v1() {
        let dump = PcpDump {
            pfn: 1, current_gen: 1, entry_gen: 1, fr1: false,
            requested: [0u64; BITMAP_WORDS],
            compiled: [0u64; BITMAP_WORDS],
            denied_raw: [0u64; BITMAP_WORDS],
            words: [0u32; ENTRIES_PER_PAGE],
            call_count: 1,
        };
        let mut bytes = dump.to_bytes();
        bytes.truncate(V1_FILE_LEN);
        assert!(PcpDump::from_bytes(&bytes).is_err());
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
