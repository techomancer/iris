# An aliased physical page must share its generation counter

Two independent bugs, one per memory path, both with the same effect: a page
reachable at two physical addresses had **two different** JIT generation
counters, so writes through one address never invalidated code compiled through
the other.

## Symptom

```
> j2 pcp
pfn=0x00000000  gen=0  entry_gen=0  ...
```

`gen=0` on the MIPS TLB refill vector page — which the IRIX kernel patches
repeatedly during boot. The `fetchverify` detector then reported stale compiled
code there:

```
=== STALE COMPILED CODE at 0xffffffff80000048 phys=0x00000048 ===
  compiled-in: 03400008  jr k0
  now in memory: 42000006  tlbwr
```

The JIT compiled the page, the kernel rewrote it, no generation bump was ever
observed on the counter the JIT was watching, and the stale compilation kept
running.

## The aliasing

IRIS maps bank 0's first 512KB twice:

| view | physical | gen counter index |
|---|---|---|
| LOMEM | `0x08000000` | `gen_base + 0x8000` |
| low alias | `0x00000000` | `gen_base + 0` |

`gen_ptr` is documented as *"a pure shift off the physical address, no bank
lookup, no masking"*. That is precisely why two addresses aliasing one page
index two different counters **unless the alias is mapped onto the same gen
object**.

## Bug 1 — ppmem window: `PpMemSpace::map_alias`

Skipped the gen mapping whenever the alias's gen range was below host
granularity. 512KB of data needs 1KB of counters (`size / GEN_RATIO`), which is
under the 4KB granule, so it was **always** skipped. The comment justifying it:

> harmless, since an alias is by definition the same physical pages as the
> region it mirrors, hence the same counters

True of **data** — one mmap object, two views. False of the **gen window**,
which is a separate parallel mapping.

**Fix:** round the gen length up to one granule (`map::align_up`). Safe because
an alias always maps a bank's *leading* `size` bytes from object offset 0, so
rounding up can only pull in gen pages belonging to that same bank. The
remaining unmappable case is now a `debug_assert!` rather than a silent skip.

## Bug 2 — bus path: `AliasBus`

Forwarded all eight read/write methods with `addr + offset`, but **not
`gen_ptr`** — so it fell through to the `BusDevice` trait default, which returns
null. A null `gen_ptr` makes `PhysicalCodePage::claim` point the page at the
shared `NEVER_COMPILABLE_GEN`, *"initialised to 0 and never bumped"*.

**Fix:** forward `gen_ptr` with the same `+offset` translation as every other
method.

## Both had to be fixed

`Physical::gen_ptr` tries ppmem first and falls through to `device_map`:

```rust
if let Some(p) = self.ppmem_gen_ptr(addr) { return p; }
unsafe { (*self.device_map[(addr >> 16) as usize]).gen_ptr(addr) }
```

So a `ppmem` build hit bug 1 and a non-`ppmem` build hit bug 2. Fixing one alone
leaves the other configuration broken — which is why it is worth checking *both*
paths whenever aliasing is involved.

## Tests

- `ppmem::tests::low_alias_and_lomem_share_generation_counters` — a bump at one
  index must be visible at the other, both directions.
- `physical::ppmem_tests::low_alias_gen_counter_agrees_with_lomem_on_both_paths`
  — `AliasBus::gen_ptr` forwards *and* translates (a page one granule past the
  alias base must not resolve to the first counter).

Both verified to fail with their own fix reverted.

**Test by behaviour, not pointer identity.** A first attempt compared the two
counter addresses and failed even with the fix applied: with ppmem they are two
window offsets backed by one physical page, so the addresses legitimately
differ while the contents must not.

## How it was found

`--features fetchverify`, which verifies per instruction that the word an engine
is about to execute still matches memory. This is the class `jitv2_lockstep`
structurally cannot see: lockstep compares the JIT against the interpreter, and
if both run the same stale bytes they agree perfectly. A full lockstep boot of
IRIX reaching a working desktop says instruction *emulation* is correct — it
says nothing about whether the bytes were the right ones.

The pre-existing `low_alias_shares_pages_with_bank0` test checked that the alias
mirrors **data** correctly. It does. Nothing checked the counters.
