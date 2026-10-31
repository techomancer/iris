# MP LL/SC: SC is a host compare-exchange on the data, not a line-ownership tracker

Status: implemented 2026-10-30, for machines with `MipsCore::smp` set
(`exec_sc`/`exec_scd`, `MipsCache::cas_ram32`/`cas_ram64`). One CPU still tests
the LLbit and stores. An SC to a word other than the LL's, or to memory outside
the ppmem window, stores the ordinary way. `write64_masked` became a CAS loop.

## The decision

When IRIS runs more than one CPU thread:

- `LL`/`LLD` record the physical address **and the value loaded**, and set the
  LLbit as today.
- `SC`/`SCD` succeed iff the LLbit is set **and** a host atomic
  `compare_exchange` of the word at that physical address, from the LL'd value
  to the new value, succeeds. The store happens only inside the CAS.
- Plain stores from other CPUs, and DMA, stay what they are now: aligned host
  stores of exactly their own width. They need no tracking and no locking.
- Exceptions and `ERET` keep clearing the LLbit (see
  `sc-succeeds-on-llbit-alone.md`); that part does not change.
- `LL`/`SC` stay `Classify::Excluded` in jitv2, so the CAS lives in one place,
  the interpreter.

This is what QEMU does for SMP MIPS, ARM and PowerPC guests.

## Why not track cache-line ownership

The tempting design is one atomic byte per guest cache line, every CPU access
ORs its bit in, `LL` resets the byte to the issuing CPU, `SC` checks that no
other bit appeared. It was considered and rejected for four reasons:

1. **It is the global lock in disguise.** A locked RMW on every guest load and
   store is a full fence per access on x86, and the tracking bytes false-share:
   64 guest lines' bytes sit in one host line, so two CPUs working in unrelated
   8 KB regions bounce the same host line on every access. It also puts a
   store on the load path, which the inline-memory JIT and tcache fast paths
   cannot absorb.
2. **Reads must not be tracked.** The architecture requires `SC` to fail only
   when another processor or coherent I/O agent *stores* into the block.
   Remote loads downgrade the line to shared and are allowed to leave the
   LLbit alone. Tracking them only adds spurious failures.
3. **`LL` resetting the byte clobbers other CPUs' reservations.** Two CPUs
   with `LL` outstanding on different words of one line (adjacent lock words,
   which IRIX has) each cancel the other; both `SC`s fail, both retry, and
   they livelock with no store ever happening.
4. **Check-then-store is not atomic.** Another CPU's store lands between the
   check and the store and two `SC`s both report success. Checking the byte
   again *after* the store does not fix it: a failed `SC` must leave memory
   untouched, and by then the store has landed. Concretely: A checks (clean),
   B stores, A stores 1 into a lock word, A re-checks, sees B, reports
   failure. Memory holds A's 1, nobody owns the lock, everyone spins forever.
   Restoring the old value destroys B's store instead.

Validate-before-and-after works for operations whose effect can be discarded
(seqlock readers). A store cannot be discarded. The only correct form of the
tracking design is a per-line lock that every plain store from every CPU must
take, which is strictly more expensive than the CAS and buys only exact ABA.

## Why the CAS is enough

Hardware makes "nobody intervened" and "the store happens" one event by holding
the line exclusive across both. The host CAS is the same mechanism one level
down: the host core takes the host cache line exclusive for the duration of the
instruction, and every other emulated CPU's plain store to that word is
serialised against it by the host's own coherence protocol. The guest word is a
single aligned host `u32`/`u64` thanks to the `^3` byte and `rotate_left(32)`
qword layout in `src/dev/mem.rs`, so there is nothing to tear.

The one deviation from hardware is **ABA**: if the word changed and changed
back between `LL` and `SC`, the CAS succeeds where hardware would fail.
Spurious *failure* is architecturally allowed (software must loop), so spurious
*success* is the only thing that can go wrong, and ABA is all of it. Every
primitive in the IRIX kernel's `atomic_ops.h` is value-based (add, compare-and-
swap, test-and-set, bit set/clear, the `k_flags` bitlock), so none of them can
observe it. Nothing in IRIX pops a lock-free stack with `LL`/`SC`.

Width interactions are consistent with hardware: a byte store by another CPU
into a different byte of the LL'd word fails the CAS, and hardware fails that
`SC` too because the byte store is a store into the block.

## Prerequisites that matter more than the CAS itself

These are the real MP work; the CAS is a dozen lines.

- **The cache model must be data-less.** `mips_cache_v2` keeps real L1D and L2
  line contents; with two CPUs that is two private copies of RAM. An MP build
  runs tags-only with every access going to shared RAM. Under tcache the L2
  data is already bypassed; L1D must follow.
- **`write64_masked` is a read-modify-write** of a host qword and will lose
  bytes when two CPUs hit one qword. It must become per-byte stores or a CAS
  loop. `write8`/`write16`/`write32`/`write64` are already exact-width single
  stores; keep them that way and audit device DMA paths for merges.
- **RAM backing must be 8-byte aligned** for `SCD`'s CAS and for torn-free
  `write64`. `Vec<u32>` guarantees 4; allocators give 16 in practice; make it
  explicit.
- **Memory ordering per host.** x86 TSO is stronger than MIPS, so guest `SYNC`
  can be a no-op there. On aarch64 `SYNC` needs a real fence and the `SC` CAS
  needs acquire-release. Pick the `Ordering` per host arch from day one.
- **Cross-CPU code invalidation.** The per-page generation counters in
  `Memory` are atomic already, but a compiled page running on CPU B must see
  CPU A's write before B's `CACHE` flush completes. IRIX flushes explicitly
  after writing code, so a check at the flush point on the executing CPU is
  enough, but it has to be cross-thread.
- **Count vs. the HEART timer.** CPU threads drift relative to each other;
  per-CPU Count must stay monotonic and roughly in step
  (`cpu-count-is-wallclock-derived.md`).

## Instrument

`--features llstats` (`ll stats`, see `sc-succeeds-on-llbit-alone.md`) already
reports per-address `sc_fail` runs. Under MP it is the first thing to look at
when a lock appears stuck: a `max_run` in the hundreds on one address means the
CAS is failing where it should not, or a plain store is reaching the word
through a path that is not width-exact.
