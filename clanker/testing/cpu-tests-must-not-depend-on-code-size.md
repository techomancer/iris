# cpu-tests must not depend on where the linker puts the stack

The stack sits right after `.bss` (`cpu-tests/harness/link.ld`), so its address
moves whenever the suite's code or data changes size. Two tests depended on it
not landing in particular places, and a compiler change exposed both. Building
with GCC 16.2.1 instead of GCC 12.4 shrank the code by about 0x7a0 bytes and
moved `_stack_top` from 0x88254890 to 0x882540f0. On the R4400 interpreter
(the full cache model; jitv2's write-through shadow cache is unaffected) two
tests then went wrong. Both were test bugs; IRIS did what the R4400 does.

## `cache/index_tag_rt` destroyed the suite's own state

The test stores zero tags into D-cache lines 0-3 with Index_Store_Tag. That
drops each line without writing it back. In the R4000 User's Manual, 2nd ed.,
the CACHE instruction (pp. A-43 to A-45) has its operation table on A-44: Index
Writeback Invalidate writes a dirty primary data line back, while Index Store
Tag only writes the tag from TagLo and TagHi (described on p. 93). With the
stack on those lines, dirty stack frames vanished, and the run derailed: the
rest of the cache group never ran and the MIPS IV tests misreported the CPU.
Moving only the stack reproduced or removed it, in both directions.

Fix: write each line back with Index_Writeback_Invalidate_D in the same asm
block as the tag store, so nothing can dirty it in between. A linker fix cannot
work: with a direct-mapped 16 KB D-cache and a 600 KB image, some live data
always shares those indexes.

## `umode/load_miss_third` took a silent Virtual Coherency exception

The test mapped scratch page `_scratch_start + 0x4000` (VA[14:12] = 4 through
KSEG0) at user address 0x00800000 (VA[14:12] = 0). Bit 14 is part of the R4400
secondary cache's PIdx, so a primary miss that hits the secondary through the
other address raises a Virtual Coherency exception. The manual lists the
exception's conditions on p. 133; p. 252 defines the secondary tag's PIdx as
VA[14:12], p. 117 gives CacheErr's PIdx the same bits, and p. 253 says a PIdx
mismatch on a secondary hit signals the exception. The suite's default
exception handler steps over the faulting instruction, so the store was lost
and the test read 0x5ec0de02 instead of 0x5ec0de00. Caught in the monitor with
`bp add 0x80000180 if (c0_cause & 0x7c) == 0x7c`: Cause 0x0000807c (VCED), EPC
on an `swl`, BadVAddr 0x8825c000. With the GCC 12 layout the primary line
happened to survive, so no miss and no exception.

Fix: every mapped scratch page keeps VA[14:12] equal to its KSEG0 alias. The
second data page moved to 0x10000, and `_scratch_start` is aligned to 32 KB;
the tlb and excep tests also map it and only worked because bit 14 happened to
be clear.

## Rules

- A test that clobbers cache state by index must clean those lines first.
- Every page a test maps must have the same VA[14:12] as every other address
  the suite uses for it, or the R4400 raises a VCE.
- No test expects a VCE, and the handler hides one. If a mapped-memory test
  reads a stale value, check for a VCE before suspecting the cache model.
- CI fails a run that stops short of its TAP plan (`make check-run`): a
  derailed run can report no more failures than a healthy one, which a
  count-only gate reads as no change or progress.
