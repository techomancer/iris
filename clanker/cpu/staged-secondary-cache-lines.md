# Staged cache lines: a hit until an explicit operation ends them

The R10000-family shadow cache (`src/cpu/mips_cache_shadow.rs`) serves a cached
load, store or fetch from a secondary line the guest staged with Index Store Tag
(S) and Index Store Data (S): a valid state in TagLo[11:10] and a tag equal to
PA[39:18] of the access. A guest that never stages a line pays one load and a
not-taken branch per access.

## The limit: no replacement

An ordinary cached miss never fills the shadow, so it never evicts. A staged
line stays until Index Writeback Invalidate, Hit Invalidate, Hit Writeback
Invalidate, an Index Store Tag that overwrites it, or power-on. A guest that
reuses the memory without invalidating reads the stale line and its stores stay
in the line, where hardware would have evicted it, writing a dirty one back to
the tag's address.

Staged lines left at the login prompt, `l2 probe 0` in the monitor, CPU 0:
Octane (IP30 PROM, IRIX 6.5.30) 0, Fuel (PROM 6.210, IRIX 6.5.30, jitv2) 0,
O2 R10000 (IRIX 6.5.30) 0. The shipped PROMs invalidate what they stage, so the
`staged_any()` branch is not taken once IRIX runs.

## Primary instruction-cache lines

Index Store Tag (I) and Index Store Data (I) stage a primary I line the same
way: tag TagLo[31:8] = PA[35:12], TagLo[6] valid, TagHi[3:0] = PA[39:36]; index
VA[13:6], way VA[0], one word per Index Store Data at VA[13:2], 16 per block. A
cached fetch checks both ways at its VA[13:6] against its PA, before the
secondary cache. Index Invalidate (I), Hit Invalidate (I), an Index Store Tag
with the valid bit clear, and power-on end the line. Loads are not served by it.
Model choices: the compare on a fetch is inferred, a hit does not touch LRU, and
the instruction is TagLo[31:0] with TagHi[3:0] ignored.

**Serving is off unless `IRIS_STAGED_L1I` is set.** The predecoded form is
unknown, and the plain-instruction assumption is wrong for the Fuel PROM: it
stages primary I lines for its own code with words taken from Index Load Data of
lines the hardware had refilled and predecoded (36-bit values with extra bits
inside the word, not an instruction in TagLo). With serving on, the Fuel PROM
executes them and never gets past "Copying PROM code to memory". iris refills and
predecodes nothing, so it cannot produce those words. Serving needs the
predecoded form, or a guest that stages plain instructions. The primary data
cache is not served: its dirty lines write back to the secondary cache and the
state-modifier field has no model.

## Why the fast paths must be told

Translations are cached per page. The interpreter's and the JIT's inline loads
and stores reach RAM through the micro-TLB and ppmem's window without calling the
cache, and the JIT compiles code from memory. So:

- `nutlb_fill` declines a page that holds a staged line (`MipsCache::page_staged`);
- `jitv2_track_pcp` leaves such a page to the interpreter;
- a CACHE instruction that changes the set of staged pages calls
  `nanotlb_invalidate` (`take_staged_changed`), because entries made before it
  do not know;
- the decode cache reads such a page word by word (`DecodeCache::set_bus_only`).

Dropping any of the first three fails `a_compiled_load_hits_a_staged_line_not_the_window`,
`a_compiled_store_to_a_staged_line_stays_in_the_line` or
`a_staged_line_is_fetched_as_code_instead_of_the_compiled_page`; those run under
`jitv2` and under `lightning,jitv2`.

## Inferred or absent

- The TagLo word of a staged doubleword lands at the lower address
  (`STAGED_TAGLO_AT_LOWER_ADDRESS`). Inferred from software, not measured.
- A store to a staged line makes it DirtyExclusive. NEC U10278EJ4V0UM p.246
  counts such requests; the manual does not state the transition.
- Every CACHE operand is translated first (exceptions: TLB Refill/Invalid, privilege Address Error, never TLB Modified). The secondary set, way and quadword come from the operand's physical address (`set_cache_op_pa`), the primaries' from its virtual address. An operand on an uncached page is undefined on the R10000 and goes ahead as if cached.
- Only the R10000 is documented. The R12000, R14000 and R16000 use the same
  tag layout here, which is an inference.
- Not modelled: predecode, primary data cache staged hits, ECC and check bits, the
  virtual-index purge, the BadVAddr/Context/EntryHi values after a CACHE-raised TLB exception, Status.CH, the
  tag rewrite on invalidate (only the state clears), coherence with other CPUs
  and DMA.
- Staged lines are saved in snapshots (`save_cache_state`). The rest of the
  shadow arrays, which only diagnostics read, are not.
