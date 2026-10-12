# R5000 L1 TagLo uses the R4400 PState encoding

Addresses `cpucritique.md` CACHE-1 (twice: see below), and cleans up
CACHE-2/CACHE-3/CACHE-4 alongside it.

## CACHE-1 — settled by the software SGI shipped (2026-10-11)

The R5000 primary TagLo is the R4000-family layout: `[31:8]` PTag, `[7:6]`
PState, `[0]` P. A valid I-cache line reads 2; a D-cache line reads 2 when
clean, 3 when dirty. Shared (1) never occurs on an R5000, so a stored 1 is
taken as CleanExclusive there. The only R5000-specific parts are P (over PTag
alone) and the way select (VA[14]).

Evidence, from `~/gits/irixhistory` (IP22 builds use one R4000-family binary
for R4400, R4600 and R5000):

- **IDE `icachetag`** (`stand/arcs/ide/fforward/cache/icache.c`) fails unless
  every filled I-cache line reads PState == 2. The Indy script
  (`stand/arcs/ide/IP22/ip24`) runs `icache2` with no CPU-type check.
- **IDE `set_tag`** (`.../cache/cachesubs.c`) writes PCLEANEXCL `0x80` to make
  an I-cache line valid and PDIRTYEXCL `0xC0` for a dirty D-cache line, then
  sets W with a load and store; W is not in TagLo.
- **IDE `icache_fill_writeback`** pins a line into a way on R4600/R4700/R5000
  by storing back exactly what Index_Load_Tag returned.
- **IRIX `mcparity.c`** (IP22 parity recovery) reads primary tags on any IP22
  CPU: PState != 0 is valid, PState > PCLEANEXCL is dirty.

### How it went wrong first

The first CACHE-1 fix decided the R5000 used discrete bits, D = 7 and V = 6,
from a code critique with no source behind it. It read a clean line back as
`0x40` (fails `icachetag`) and took IDE's `0x80` for *invalid*. Its tests were
written to the same assumption, so they passed.

The NEC VR5000 manual (Table 6-13: 0 = Invalid, 3 = Valid) doesn't match the
shipped software either. Trust what IRIX and IDE do on the real part over a
datasheet table, and never over a critique.

Tests: `taglo_filled_lines_read_back_the_r4000_pstates_on_both_models`,
`taglo_store_decodes_what_ide_and_irix_write`, and the symmetry check
`taglo_store_load_round_trips_on_both_models`. See also
`cache-tag-index-and-check-bits.md`.

## CACHE-3 — the comment was simply false

The `write` path carried:

> *"tcache is **not implemented here** — the 2-way paths always take the real-cache route,
> so no transparent lines are ever created on R5000 and the feature is silently a no-op for
> that model."*

Contradicted five lines below it, where `tc_write` runs unconditionally under `cfg(tcache)`.
`fill_l1d_line` likewise sets `transparent = true` purely on `cfg(tcache)` with no `IS_R5K`
guard. Comment rewritten to describe what the code does.

## CACHE-4 — not a bug

The critique claims R5000 `tcache` leaves stale instructions in `ic_instrs` because
`tc_invalidate_l2_code` early-returns when `!HAS_L2`.

That early return is correct. `has_code` is **tcache bookkeeping for the decoded-instruction
slots in L2** (`l2.instrs`), not architectural I-cache coherency — see the `l2_hit` guard in
the L1-I fill path, which refuses an L2 line that was filled for data. With no L2 there are
no such slots to invalidate.

R5000's own decode slots (`ic_instrs`) are populated **only** on an L1-I fill, which happens
only after an L1-I miss. A guest that modifies code must issue a `CACHE` op to retire the
L1-I line — that is the architectural contract on every MIPS, and hardware does not snoop
L1-I against data stores either. Retiring the line retires its decode slots with it.

Adding write-side L1-I invalidation would impose a cost on the *primary* R4400 target to
"fix" behaviour that matches the architecture. Not done.

## CACHE-2 — nothing to store

TagHi carries ECC/parity for the primary caches on both R4400 and R5000, and this emulator
models no ECC. The physical tag fits entirely in TagLo: `L1_PTAG_MASK` is 24 bits at
`L1_PTAG_SHIFT = 12`, i.e. PA[35:12] — the full 36-bit physical address. Nothing spills into
TagHi, so zeroing it on `C_ILT` is right. Left as-is.

## Tests, and what each actually proves

- `taglo_filled_lines_read_back_the_r4000_pstates_on_both_models` — fills I and
  D lines (clean, then dirty) on both models and checks PState 2/2/3.
  **Catches the load side.**
- `taglo_store_decodes_what_ide_and_irix_write` — feeds the R5000 the patterns
  IDE writes (`0x00`, `0x40`, `0x80`, `0xC0`). **Catches the store side.**
- `taglo_store_load_round_trips_on_both_models` — symmetry only.

The round trip **passes even with both halves wrong**: an encode/decode pair
that shares a mistake is still self-inverse. Keep it (an *asymmetric* pair is
its own bug class), but never treat a round trip as evidence that a format is
right. The first two only help if their expected values come from a source,
not from the encoder under test. The D/V-era tests failed exactly there: they
asserted the D/V layout itself.

## Resolved: `C_IST` writes back a dirty L1D victim (cpucritique §5.1)

**Resolved since:** the L1D arm no longer writes back (see the comment in
`cache_op`'s `C_IST` arm), pinned by the test asserting "Index_Store_Tag wrote
the dirty line back to memory". The history below is kept for the reasoning.

Investigated 2026-09-12, **deliberately not changed** — recorded so the next
session starts from the finding rather than re-deriving it.

`cache_op`'s `C_IST` arm is **asymmetric between levels**:

- **L2** calls `invalidate_l2_line(idx)` with the comment *"C_IST does not
  writeback (it's used for cache init/invalidation)."* Correct.
- **L1D** calls `writeback_l1d_line(idx, cascade)` before
  `invalidate_l1d_line`, commented *"Writeback dirty data before overwriting
  the tag."* This is what §5.1 flags.

The architectural argument against the writeback: `Index_Store_Tag` is the
cache **initialization** primitive. It runs when tags hold power-on garbage, so
a line that merely *looks* Dirty-Exclusive has no valid ptag — writing it back
scribbles arbitrary data at an arbitrary physical address. Hardware does not
do this.

**Why it was left alone:**

1. The plan called for checking the **VR5000 manual** first; it is not
   available locally, and R4400-vs-R5000 cache behaviour is exactly where this
   file already documents a divergence that was gotten wrong once.
2. It is on the boot-critical cache-init path.
3. Changing writeback semantics on reasoning alone, with no failing test and
   no observed symptom, is the wrong risk trade against everything else in
   this batch, which had tests that failed before the fix.

**To pick this up:** confirm against the VR5000 (and R4400 §11) manuals, then
drop the `writeback_l1d_line` call so L1D matches L2. A test would seed a dirty
L1D line with a ptag pointing at known memory, issue `C_IST` with a TagLo whose
state bits read Dirty-Exclusive, and assert memory is **unchanged**. That test
fails today.
