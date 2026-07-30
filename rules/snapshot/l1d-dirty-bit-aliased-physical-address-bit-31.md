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
RAM. Indy physical addresses stay below `0x2000_0000`, so the corruption is
one-directional: no clean line is falsely marked dirty, but every dirty line is
lost.

The fix moves `dirty` to bit 0. Bits [5:0] are outside both the ptag field and
`cs` at [7:6].

## Why the round-trip regression net was blind to it

`rules/snapshot/per-device-saveloadsave-round-trip-is-the-regression-net.md`
mandates a `save_load_round_trip` test per device, and the cache had one. It
could never fail:

- `u32 -> tag -> u32` is **stable**. The second save re-derives bit 27 from the
  already-corrupted address and ORs the same bit back.
- `tag -> u32 -> tag` is what loses information, and nothing tested that
  direction.

That is the general lesson, and it is the second time this shape has bitten this
codebase. **A round-trip test proves serialization is self-consistent, not that
it is correct.** When a field is packed into a word, test the direction that
starts from the struct: `dirty_l1d_tag_keeps_its_address` asserts that a
serialized dirty line deserializes to the address it came from, and it fails on
the old code with `left: 2147483648, right: 0`.

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

## The format changes

The fix changes the on-disk `dc_tags` encoding. **A snapshot written by an
unpatched binary and read by a patched one carries corrupted tags**, and vice
versa. Existing snapshots are not repaired and must be retaken. Nothing outside
`mips_cache_v2.rs` reads those words.

## Which oracle to use for restore correctness

Learned while confirming this, and worth knowing before writing a regression
gate:

- **`iris-ci validate` and a restore-versus-restore trace comparison are valid
  oracles.** Both are clean with the fix.
- **A live-versus-restored trace comparison is not.** It over-triggers. With the
  fix applied, liveness is 7 of 7 while a live-versus-restored trace still
  diverges within tens of instructions, because `cp0_random_cycle` and the
  `cycles` atomic are not serialized while `cp0_random` is, so `update_random`
  steps Random differently after a load and `TLBWR` picks a different victim.
  Random is architecturally pseudo-random, so that divergence is expected even
  on a correct implementation. Serializing both moves the divergence later
  without removing it.

Note also that a deterministic corruption cancels exactly in a
restore-versus-restore control, so a zero noise floor there is not evidence
against a deterministic bug.

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

## Determinism

Use the `ci_clock` feature for any restore comparison. The default build is
non-deterministic across restores (`count_step: A=1797737952 B=1752844598`).
