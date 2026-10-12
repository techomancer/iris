# Cache tags: index ops index by address; check bits are generated, not stored

Both from `docs/R4000_um2.pdf` and `docs/VR5000 User Manual.pdf`, now in the tree.

## Index_Store_Tag(SD) indexes by the CACHE address

An index op takes its line from the CACHE instruction's own address (R4000 UM
A-42: `pAddr[CACHEBITS:LINEBITS]` for the secondary; VR5000 UM §3.4 the same).
MAME's `r4000.cpp` agrees.

`mips_exec` passes TagHi:TagLo in `cache_op`'s `phys_addr` argument for C_IST,
and the functional model computed the L2 index from `phys_addr` for every op.
So a secondary `Index_Store_Tag` wrote whichever set TagLo[19:7] named. Fixed
in `CpuCache::cache_op`; regression test
`l2_index_store_tag_indexes_by_address_not_by_taglo`.

It went unnoticed because:

- PROM/IRIX cache init stores TagLo = 0, so every store hit set 0, and the
  other sets were already invalid from power-on.
- `cache_op_ilt_ist_l2` stores back the tag it just read. The store goes to the
  wrong set, and the read-back finds the untouched original: a round trip that
  passes with the bug. Same trap as `r5000-taglo-is-the-r4400-pstate.md`.

## P and tag ECC are computed on Index_Load_Tag

"The P and ECC fields of these registers are ignored on Index Store Tag
operations. Parity and ECC are computed by the store operation." (R4000 UM p. 93,
VR5000 UM §6.5.10.) Appendix A's wording ("the TagLo register in the case of
the secondary cache") reads the other way, but p. 93 and Chapter 16 ("the
processor generates check bits for the tag when it is written") both say
computed.

Since they are a pure function of the stored fields, the model computes them
when the tag is read back and stores nothing:

| | TagLo[0] P | covers |
|---|---|---|
| R4400 L1I | even parity | PTag, V (Fig. 11-3) |
| R4400 L1D | even parity | PTag, CS (Fig. 11-4) |
| R5000 L1I/L1D | even parity | PTag only (Fig. 12-3/12-4) |
| R4400 L2 | TagLo[6:0] ECC | CS:PIdx:STag via Fig. 16-4 (`l2_tag_ecc`) |

The Fig. 16-4 transcription is pinned by `l2_tag_ecc_matrix_is_the_manuals`
(row weights, weight-3 distinct syndromes). The text extraction of that figure
is garbled; render the page (`pdftoppm -f 455 -l 455`) to read it.

Not modelled: the CP0 ECC register load on Index_Load_Tag (data parity/ECC of
the addressed doubleword), and planted bad check bits (CE bit).

## Hit_Writeback(I) writes the I-cache's own copy back

Op 6 on the I-cache writes the line back unconditionally (both manuals). IDE
`icache2` (`icache_fill_writeback`) fills the I-cache, changes memory under it
through the D-cache, and expects the write-back to restore the I-cache's
words. It was a no-op here, so every word failed with `Actual:0xffffffff`.
(The `Expect:` value IDE prints there is an uninitialised variable.)

The I-cache has to keep what it was filled with:
- **R5000:** `ic_instrs` already is that copy.
- **R4400:** fetch is served from the L2 decode slots, which follow L2 data
  after the fill. `ic_raw` snapshots the words at fill time, for the write-back
  only.

Under tcache a cached store goes straight to RAM, so the fill already sees the
new words, and this test cannot pass (same trade as `cache/hit_inv_discards`).

Snapshots carry I-cache tags but not the words. `load_cache_state` now
rebuilds them (`rebuild_l1i_words`). Before, a restored valid R5000 line was
served from empty slots (zeros, i.e. NOPs) until it was evicted.

## Parity/ECC injection (`cache-ecc` feature)

IDE `dcache1`, `icache1` and `scache1` plant bad check bits and expect a Cache
Error. That machinery is behind the `cache-ecc` Cargo feature, off by
default: nothing but a diagnostic ever observes it. With it, every IP22 IDE
(field, 6.5) cache test passes on R4400 and R5000: icache1, icache2, dcache1,
scache1.

- **Cache Error:** `EXEC_CACHE_ERR` (ExecStatus bit 30). Delivery sets
  ErrorEPC and ERL and jumps to `0xA0000100`, or `0xBFC00300` with BEV; Cause,
  EPC, EXL and BadVAddr are untouched. CacheErr comes from
  `MipsCache::take_cache_err`.
- **Check bits are stored only when forced.** Otherwise they always match the
  data, so they are computed on demand. D-cache: per doubleword. L2:
  per-doubleword XOR in `ParityState::l2_xor`. I-cache: per line.
- **Zero hit-path cost:** a line that needs checking carries `L1DTag::SLOW`
  (ptag bit 1), so it fails the exact compare in `matches_phys`, in the
  interpreter and in jitv2's inline guard alike. The miss path resolves it and
  returns `L1D_SLOW` (2) inside the existing `r > 1` branch. Cold paths that
  must find a marked line use `matches_line`.
- **CE semantics** (R4000 UM ch. 5, VR5000 UM 7.2.9):
  - a store takes the ECC register as the check bits for the bytes it writes;
  - Fill(I) takes it for the whole line;
  - a D-cache write-back by a CACHE op XORs it into the L2 ECC of the
    doubleword that op addresses.
- **Detection:**
  - a D-cache load checks its bytes;
  - a primary refill from L2 checks the transferred line (EC set);
  - Hit_Writeback(I) checks the line and raises the error from the CACHE
    instruction (`take_op_status`).
- **Index_Load_Tag** loads CP0 ECC with the selected doubleword's check bits:
  - L2: data ECC via Fig. 16-1, transcribed in `L2_DATA_ECC_MASKS`;
  - D-cache: byte parity;
  - I-cache: byte parity on R4400, per word on R5000.
- **Fill(I) on a 2-way cache** refills the way already holding the line. IDE
  pins lines into the way VA[14] selects with Index_Store_Tag first.

Measured before gating (interpreter, `--quick`): no cost distinguishable from
build-layout noise, which is ±10% on single `sys/` kernels. The gate is there
so the cost is zero by construction.

Not modelled: snapshots do not carry forced bits; I-cache fetch does not check
parity; CacheErr's EB/EI/EW and secondary tag errors.
