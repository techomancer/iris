# L1D cache tag must match one manual layout, not two

**Keywords:** snapshot,l1d,cache,taglo,r4400,dirty,writeback,round-trip
**Category:** snapshot

The R4400 manual describes the primary D-cache tag twice, and the two are not
interchangeable:

- Figure 4-18, the CP0 TagLo register: `[31:8] PTagLo`, `[7:6] PState`, `[5:1]`
  written as zero, `[0]` parity. No dirty bit.
- Figure 11-4, the physical cache line: `[28] W'`, `[27] W`, `[26] P`,
  `[25:24] CS`, `[23:0] PTag`. `W` is the write-back bit.

`impl From<L1DTag> for u32` used to mix them, taking `PTag` at `[31:8]` from the
register and the dirty bit at 27 from the cache line. In the register layout bit
27 is `PTag` bit 19, so it aliased physical address bit 31. Every dirty L1D line
deserialized 2 GB above its own address, `matches_phys` never matched it again,
and the stores it held were never written back.

Serialization now uses Figure 11-4 throughout, which is also what `L2Tag`
follows. `L1ITag` still uses the register layout; it has no dirty bit, so
nothing aliases.

A `save -> load -> save` round-trip cannot catch this class of bug. `u32 ->
L1DTag -> u32` is stable, because the second save re-derives bit 27 from the
address the load already corrupted, so both words agree. Only `L1DTag -> u32 ->
L1DTag` loses information. Test the direction that starts from the in-memory
tag.

`iris-ci validate` passed before the fix for the same reason: it loads one
snapshot twice, so both sides carry the identical corruption.
