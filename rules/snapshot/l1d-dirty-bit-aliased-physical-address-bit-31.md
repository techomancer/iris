# L1D dirty bit aliased physical address bit 31, so restore lost every dirty line

**Keywords:** snapshot,restore,cache,mips_cache_v2,l1dtag,dirty,panic,writeback,corruption
**Category:** snapshot

Every `iris-ci restore` of a booted IRIX 6.5 guest ended in a kernel panic. This
was the cause. Fixed.

## The bug

`impl From<L1DTag> for u32` (`src/mips_cache_v2.rs:277`) packed the `dirty` flag
into bit 27:

```rust
(raw_ptag << 8) | ((t.cs as u32 & 0x3) << 6) | (if t.dirty { 1 << 27 } else { 0 })
```

`raw_ptag` occupies bits [31:8] of that word, so bit 27 is `raw_ptag` bit 19.
With `L1_PTAG_SHIFT = 12` that is **physical address bit 31**.

So every dirty L1 D-cache line deserialized with its tag `0x8000_0000` too high.
`matches_phys` then never matched it, the line was never written back, and every
store still sitting in L1D at snapshot time silently reverted to the value in
RAM.

The corruption is one-directional, but the margin is smaller than it looks.
Setting physical bit 31 needs 2 GB, and the highest mapped physical address in
`src/physical.rs` is `HIMEM_END` at `0x3000_0000`, so no real line can carry that
bit. Note that RAM banks 2 and 3 do live at `0x2000_0000..0x3000_0000`, so the
bound is `0x8000_0000` and not, as an earlier draft of this note said,
`0x2000_0000`.

The aliasing is exact rather than approximate, which is worth seeing:
`(0x8000_0000 >> 12) << 8 == 0x0800_0000 == 1 << 27`. The old dirty flag and
physical address bit 31 were literally the same bit. A clean line at
`0x8000_0000` would have deserialized **dirty** and been written back over RAM,
so the other direction of the corruption existed too and was unreachable only by
accident of the address map.

The fix moves `dirty` to bit 0. Bits [5:0] are outside both the ptag field and
`cs` at [7:6].

## Why no test caught it

Two separate gaps, and the first is the plain one.

**The cache was exempt from the regression net.** `mips_cache_v2.rs` had no
serialization test at all. `save_load_round_trip` exists for ds1x86,
eeprom_93c56, ioc, mc, mips_tlb, pit8254, ps2, seeq8003, wd33c93a and z85c30, and
`rules/snapshot/per-device-saveloadsave-round-trip-is-the-regression-net.md` does
not list the cache in either its covered or its not-covered section. It was
simply missed.

**And the mandated shape would not have caught it anyway.** `u32 -> tag -> u32`
is **stable**, because the second save re-derives bit 27 from the already
corrupted address and ORs the same bit back. Only `tag -> u32 -> tag` loses
information, and that is the direction a round-trip test does not exercise.

So: **a round-trip test proves serialization is self-consistent, not that it is
correct.** When a field is packed into a word, test the direction that starts
from the struct. `dirty_l1d_tag_keeps_its_address` does, and it fails on the old
code with `left: 2147483648, right: 0`.

## Evidence

| condition | restores survived |
|---|---|
| unpatched, 4 arms, 2 snapshots | **0 of 22** |
| `dirty` moved to bit 0, 2 independent guests | **20 of 20** |

Survival means `iris-ci run --shell sh "echo ALIVE"` returning rc 0 after the
restore. Independently reproduced on a second harness: 0 clean survivors
unpatched against 7 of 7 patched.

Baseline panic:

```
PANIC: KERNEL FAULT
PC: 0x1 ep: 0x8d6a3150
EXC code:128, `Software detected SEGV '
Bad addr: 0x0, cause: 0x10008008
```

Wild `PC` and `BadVAddr` values are the expected signature: kernel stores that
were still in L1D reverted, so pointers read back as whatever the underlying RAM
held. One observed fault address was `0x54524152`, which is ASCII `"TRAR"`.

## The format changes, and old snapshots are migrated

The fix changes the on-disk `dc_tags` encoding, so `save_cache_state` now writes
a `dc_tag_format` key and `load_cache_state` treats its absence as format 1.

The migration is lossless. In an old word, bit 27 can only be the dirty flag,
never part of a real address, because a real address would need physical bit 31
and the highest mapped address is `0x3000_0000`. So `migrate_dc_tag_word` moves
bit 27 down to bit 0 and clears it, recovering both the address and the flag.

Without that migration the failure would have been silent and would have looked
exactly like the original bug: a patched binary reading an old word computes
`raw_ptag = (word >> 8) & L1_PTAG_MASK`, which still includes the spurious bit 19,
so the address comes back `P | 0x8000_0000` and `dirty` reads as 0 from bit 0.

The reverse direction, an old binary reading a new snapshot, is not symmetric: it
never inspects bit 0, so addresses are intact and only the dirty flags are lost.
It is still wrong, and old binaries have no way to detect it.

`dc_tag_format` is versioned separately from `SCHEMA_VERSION` because the
migration is local to `mips_cache_v2.rs`. Nothing outside that file reads these
words: the guest-visible CP0 TagLo paths `C_ILT` and `C_IST` hand-roll their own
encoding and never touch the `From`/`Into` conversions.

## Which oracle to use for restore correctness

Learned while confirming this, and worth knowing before writing a regression
gate:

**Nothing available today would have caught this.** Say that plainly before
reaching for one of these.

`iris-ci validate` (`src/validate.rs`) loads the same snapshot twice and diffs
CPU digests, so it *is* a restore-versus-restore comparison. A deterministic
corruption applied identically to both loads cancels exactly, so `validate` was
clean before the fix as well as after. It catches non-determinism, not
incorrectness, and it is not a regression gate for this class of bug.

A **live-versus-restored** trace comparison does detect a difference, but it
over-triggers so badly that it cannot serve as an oracle either: with the fix
applied and liveness at 7 of 7, live-versus-restored still diverges within tens
of instructions. `cp0_random_cycle` and the `cycles` atomic are unserialized
while `cp0_random` is saved, so `update_random` steps Random differently after a
load and `TLBWR` picks a different victim. Random is architecturally
pseudo-random, so that divergence is expected on a correct implementation.
Serializing both moves the divergence later without removing it.

What actually detected this was **guest liveness**, `iris-ci run --shell sh
"echo ALIVE"` over repeated trials, plus a unit test asserting the
`tag -> u32 -> tag` direction. Use those.

Use `ci_clock` for any restore comparison; the default build is
non-deterministic across restores (`count_step: A=1797737952 B=1752844598`).

## Dead ends, so nobody repeats them

- **Component bisection does not work on this.** Neither "restore only X" nor
  "restore everything except X" discriminates, because every component is
  load-bearing and any partial restore is an inconsistent machine that dies for
  the partiality. Excluding `cpu` or `pc` is worse than useless: `power_on_devices`
  reaches `reset_registers`, which sets `pc` to the PROM reset vector, so those
  arms just park the machine at the maintenance menu.
- **Excluding `cache_ic_tags` suppresses the symptom without fixing anything.**
  It keeps the L1-I cache cold so fetches bypass the incoherent path. It reads as
  a clean collapse to the noise floor and gives 0 of 6 survivors.
- **The CP0 timer anchor is not involved.** Its state round-trips bit-exact,
  there is no post-restore rate transient (975 Hz after load against 970
  before), and a guest failed with a healthy 979 Hz timer, `EXL` clear and the
  kernel scheduling normally. Failing guests sit in the exception vector having
  written zero Compares over 10 to 24 billion instructions, so the timer is
  starved by the fault rather than causing it.
- The CHD disk-capture hole, the JIT, `power_on_devices` on restore, CPU dispatch
  re-derivation, `wd33c93a` in-flight state, `L1ITag`/`L2Tag` conversions, and
  LL/SC link state (`llbit` and `lladdr` are both saved) were each ruled out with
  a measurement.

