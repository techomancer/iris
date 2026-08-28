# nutlb — direct-mapped data-side micro-TLB

Status: **shipped, unconditional (no feature flag).** Replaces the Read/Write
slots of the old 3-entry `nanotlb`; the Fetch slot survives as a separate,
specialised one-entry structure.

Companion reading: `HACKING.md` (data path), `rules/testing/` (TLB findings),
`docs/nanotlb_associativity.md`-adjacent memory note — a previous 4-way/AVX2
experiment **lost** to the 1-way nanotlb, so any new structure has to justify
itself against a very cheap incumbent.

---

## 0. How this evolved — read this first

This document was written as a design proposal, then kept as a results log, and
the design it originally specified is **not** the one that shipped. Two
generations exist, and sections written for the first are marked as historical
rather than deleted, because the reasoning that killed it is the most useful
thing here.

**Generation 1 — self-describing tags (§2–§6, historical).** Each entry carried
ASID + a permission set + a TLB generation counter, so a mode switch, exception
or ERET did not have to invalidate anything: a stale entry simply failed to
match. Only a real TLB mutation retired entries, via `tlb_gen`. This measured
**+10–13% on live IRIX** against the nanotlb (§9) and shipped behind
`--features nutlb`.

**Generation 2 — validity bitmask (§11, current).** The tag was stripped to a
bare page number and validity moved to a 4-word-per-array bitmask, with
coherency restored the way the nanotlb does it: flush at every barrier. This
trades ~3–4 ALU ops off *every probe* — including every inline JIT load and
store — against discarding the whole 256-entry working set on every exception,
ERET, mode switch and ASID change.

That trade was expected to lose. §1.2 and §3 of this document argue at length
that not-flushing is "the bulk of the win", and the prediction going into the
experiment was that 256 refills per syscall would swamp a few ops per access.
**It lost the prediction and won the benchmark** (§12): the probe savings, paid
on every load and store, outweigh the refills. The flush cost scales with how
much of the working set is *re-touched before the next flush*, and on real IRIX
that is far less than capacity suggests.

Generation 1 also turned out to carry a **latent correctness bug** that its own
design section did not catch — the permission test never rejected anything
(§11.3). That is not why it was replaced (the benchmark decided that), but it
does mean the +10–13% in §9 was measured against a permission check doing no
work, so Generation 2's margin over it is slightly understated.

The lesson worth carrying forward: **on this hot path, per-access cost beat
per-event cost, twice.** The 4-way nanotlb lost to 1-way for the same reason,
and now tagged nutlb has lost to untagged. Two data points pointing the same
direction is worth remembering before adding anything to the probe.

---

## 1. What's wrong with the nanotlb *(the original motivation)*

> Written when the nanotlb still served Read/Write. It no longer does; the
> Fetch slot is all that remains of it. Both problems below are still the
> reason a data-side micro-TLB exists — but note that Generation 2 chose to
> take problem 2 *back* (it flushes on every transition) and still won, which
> §12 unpacks.

Three single-entry slots indexed by `AccessType` (Fetch=0, Read=1, Write=2):

```rust
pub struct NanoTlbEntry { pub va_tag: u64, pub pa_encoded: u64 }
// hit test: self.va_tag == (va & !0xFFF) | 1
```

Two structural problems, both costing hit rate rather than correctness:

1. **Capacity one, per access type.** Any load stream touching two pages
   alternately (memcpy src/dst, stack + heap, a struct spanning a page
   boundary) misses on *every* access and pays a full `translate_fn` call plus
   a `MipsTlb::translate` vmap probe. This is the whole motivation.

   The slow path also got *more* expensive recently, which raises the value of
   hitting in nutlb: since `VMAP_MULTI` (`mips_tlb.rs:356`) fixed the phantom-miss
   bug, a vmap slot with more than one claimant no longer short-circuits to
   `TlbResult::Miss` — it correctly falls through to the linear MRU scan over 48
   entries. Correct, but it means the tail of the miss-cost distribution is
   heavier than it was. Baselines taken before that fix understate the win here.

2. **Context changes must nuke it.** Because the tag carries no ASID and no
   privilege, every event that could change the meaning of a VA has to
   invalidate all three slots. `nanotlb_invalidate()` is called from
   `on_cp0_status_changed`, both `handle_exception` paths, `exec_tlbwi`,
   `exec_tlbwr`, `exec_eret`, and `MipsCpu::stop()`. Exceptions and ERET are
   *hot* in IRIX — every syscall and every TLB refill flushes the data-side
   translation cache even though the TLB contents never changed.

### 1a. ASID is absent from the tag *by design* — and what that costs

It would be easy to read the missing ASID as an oversight. It isn't. The
incumbent deliberately trades tag width for invalidation: because
`on_cp0_status_changed`, both `handle_exception` paths and `exec_eret` all
flush, **any privilege transition that could carry an ASID switch is itself the
barrier**. A kernel that changes ASID does so inside EXL and returns through
`ERET`, and the flush on that transition is what makes the untagged slots safe.
The slots are only ever consulted within one privilege epoch. That is a
coherent design, and it is why the 3-entry nanotlb can get away with a
one-compare hit test.

The cost is the flush frequency in §1.2, not correctness — with one caveat
worth stating precisely, because it decides §7.

**The barrier is transition-shaped, not ASID-shaped.** It contains any ASID
change that is *accompanied by* a privilege transition. An ASID change with no
transition around it is not covered:

- `MTC0 EntryHi` (`write_cp0` reg 10) writes ASID; `handle_cp0_side_effects`
  has no reg-10 case.
- `TLBR` (`exec_tlbr`, `mips_exec.rs:5436`) overwrites `cp0_entryhi` wholesale
  — ASID included — from the indexed entry. `exec_tlb`'s own comment justifies
  the absent flush with "they don't mutate the TLB", which is true of the TLB
  and false of the current ASID.

Neither is reachable outside kernel mode, and in kernel mode the VAs in flight
are overwhelmingly KSEG0/KSEG1 — unmapped, ASID-irrelevant. So the exposure is
narrow: it needs a kernel-mode access to a *TLB-mapped* VA, on a page cached in
a slot, after a bare ASID change with no intervening transition. Whether IRIX
ever does that is an empirical question (§10), and the honest answer today is
that it apparently doesn't.

**Status: closed, ahead of any nutlb work.** `handle_cp0_side_effects` now
flushes on reg 10, and `exec_tlbr` flushes itself; `exec_tlb`'s misleading
"they don't mutate the TLB" comment is corrected in place. Two regression tests
(`test_mtc0_entryhi_asid_change_invalidates_nanotlb`,
`test_tlbr_asid_change_invalidates_nanotlb` in `mips_exec_test.rs`) map one VA
to different PFNs under ASIDs 10 and 11 and assert the second read sees the
second page. Both were verified to **fail** with the flushes disabled —
returning `0xAAAAAAAA`, the stale ASID-10 page — so the hazard is real and
reachable, not theoretical.

Snapshot restore, digest restore and the TOML load path all already route
through `on_cp0_status_changed`, so they were covered; reset zeroes EntryHi
before anything can fill a slot. Those two instructions were the entire gap.

nutlb would close this differently — ASID in the tag, so a stale entry simply
fails to match rather than needing a flush. That remains a nice property of the
redesign, but with the hazard already fixed it is **not part of nutlb's
justification**. That stays performance: §1.1 capacity and §1.2 flush frequency.

## 2. Shape *(Generation 1 — historical, see §0 and §11)*

> **Superseded.** The shipped entry is 16 B — `virttag` (bare page number) +
> `phys` — with no `tlbgen` and no `asid_mask`, and validity held in a separate
> bitmask. The Read/Write split and the `phys` encoding below *did* survive
> unchanged; everything about the tag did not.


```rust
pub const NUTLB_BITS: usize = 6;                    // tunable, see §8
pub const NUM_NUTLB_ENTRIES: usize = 1 << NUTLB_BITS;

#[derive(Clone, Copy, Default)]
pub struct NuTlbEntry {
    pub virttag:   u64,   // see §3
    pub phys:      u64,   // PA[63:12] in bits [63:12]; C-field in bits [2:0]
    pub tlbgen:    u32,   // must equal core.tlb_gen
    pub asid_mask: u8,    // 0x00 if global, 0xFF if ASID-qualified
}

// [0] = Read, [1] = Write
pub nutlb: [[NuTlbEntry; NUM_NUTLB_ENTRIES]; 2],
```

Direct-mapped, indexed by `(va >> 12) & (NUM_NUTLB_ENTRIES - 1)`.

**Keep the Read/Write split** (rather than one shared array). The two arrays are
not redundant: the Write array holds only pages that passed the **dirty-bit
check** in `tlb_translate_impl`. A page mapped V=1/D=0 is a legitimate Read hit
and must trap `EXC_MOD` on write. Merging the arrays would require an extra
"writable" bit in the tag and a per-access mask of it — more work in the hot
path than just having two arrays, and it would halve effective read capacity
for write-heavy code. Two arrays also means a write never evicts a read entry
for the same page, which matters for read-modify-write loops.

`phys` reuses the incumbent's proven encoding: C-field (2/3/5) in bits [2:0]
straight out of `TranslateResult.status & 0x7`, so `cache_attr_raw()` stays a
mask and `phys_addr(va)` stays `(phys & !0xFFF) | (va & 0xFFF)`.

---

## 3. The virttag, and a correction to the proposed hit test *(Generation 1 — historical)*

> **Superseded.** No permission bits, ASID or generation are stored today.
> Retained because §3a's analysis of *why* the permission test is hard to fold
> into one compare is still correct, and because the test it settled on turned
> out to be broken anyway (§11.3) — a worked example of a design review that
> checked the algebra and missed the bit assignment.


Proposed layout, which I'm keeping:

| bits | meaning |
|---|---|
| 63:12 | VA[63:12] — page number |
| 11 | *reserved for ERL, see §5* |
| 10 | page is reachable from **kernel** |
| 9 | page is reachable from **supervisor** |
| 8 | page is reachable from **user** |
| 7:0 | ASID (0 if global) |

Note the polarity: bits 10/9/8 form a **permission set** — "which modes may
touch this page" — not a single "security level of the page". A KUSEG page is
reachable from all three modes, so it stores `0b111`; a KSSEG page stores
`0b110`; a KSEG3/KSEG0/KSEG1 page stores `0b100`. This is the correct
orientation, because MIPS segment permission is downward-inclusive (kernel can
reach everything), and it is what makes the mask test below work.

The current privilege mask held in `MipsCore`:

| mode | `cur_sec_mask` |
|---|---|
| User | `0x0100` |
| Supervisor | `0x0200` |
| Kernel | `0x0400` |

### 3a. The proposed test is wrong; here is the fix

You proposed:

```
((incomingva & !0xFFF) | (cur_sec & (virttag & 0xF000)) | ASID)
    == (virttag & (!0xFFFF | asid_mask))
```

Three defects:

1. `virttag & 0xF000` masks bits 15:12, but the security bits are at 11:8 —
   it should be `0x0F00`. Bits 15:12 are part of the page number.
2. Masking the security bits *into the left-hand side from the tag itself*
   makes the test self-satisfying: whatever the tag holds is what gets compared
   against the tag. A user-mode access to a kernel-only page produces
   `0x0400 & 0x0400`… no wait — `cur_sec(0x0100) & tag(0x0400) == 0`, and the
   RHS has `0x0400` set, so it happens to *fail*. It fails for the right reason
   only by accident, and it also fails **correctly-permitted** cases: kernel
   (`0x0400`) accessing a KUSEG page (tag bits `0x0700`) yields `0x0400` on the
   left but `0x0700` on the right → spurious miss on every kernel access to
   user memory, which is exactly the `copyin`/`copyout` path. Unacceptable.
3. `!0xFFFF` on the RHS clears bits 15:12 of the page number, so pages 16 KiB
   apart alias. Must be `!0xFFF`.

The permission check is a **subset test**, and the natural instinct — OR the
mode mask into both sides — makes the bits cancel so nothing is checked.

Folding it in properly is constructible (pre-derive per-mode tag/mask pairs so a
permitted page compares equal), but "any of three bits set" is not an equality
predicate, so it needs an extra isolate-AND, an inject-OR, and `cur_sec_mask`
kept in two forms: 3-4 added ops to remove one `test`+`setne`. Not worth it.

Split it instead — one compare for identity, one bit test for permission:

```rust
#[inline(always)]
fn probe<const W: bool>(&mut self, va: u64) -> Option<&NuTlbEntry> {
    let idx = ((va >> 12) as usize) & (NUM_NUTLB_ENTRIES - 1);
    let e   = &self.core.nutlb[W as usize][idx];

    // one 64-bit compare: page number + ASID (masked out when global)
    let want = (va & !0xFFF) | (self.core.cur_asid as u64 & e.asid_mask as u64);
    let have = e.virttag & (!0xFFF | e.asid_mask as u64);

    if want == have
        && e.tlbgen == self.core.tlb_gen
        && (e.virttag & self.core.cur_sec_mask) != 0   // permission: subset test
    { Some(e) } else { None }
}
```

`(virttag & cur_sec_mask) != 0` reads as "is the current mode in this page's
permitted set". Kernel(`0x400`) ∩ KUSEG(`0x700`) ≠ 0 → hit. User(`0x100`) ∩
KSEG3(`0x400`) = 0 → miss → falls into the slow path, which raises the correct
`EXC_ADEL`/`EXC_ADES`. Permission failures **must** route through the slow path
rather than being decided here, so that `cp0_badvaddr` and friends are set by
the existing `translate_*bit_impl` code — nutlb never synthesises an exception.

Note `virttag` is never zero for a valid entry (some permission bit is always
set), so a zeroed array is naturally invalid — no separate valid bit needed,
same trick as the incumbent's `|1`. But do **not** rely on that alone: `tlbgen`
must also match, and a fresh core starts at `tlb_gen = 1` so gen 0 entries are
dead too. Belt and braces, both free.

An important consequence: because the tag carries ASID and permissions, a
**mode switch or ERET no longer needs to invalidate anything.** Only a real TLB
mutation does, via the generation counter. That is the bulk of the win in §1.2.

---

## 4. Generation counter *(Generation 1 — historical)*

> **Superseded.** There is no `tlb_gen`. TLB mutations retire entries by
> flushing the validity bitmask, like every other barrier.


```rust
pub tlb_gen: u32,   // in MipsCore, bumped on every TLB mutation
```

Bump in exactly the places that today call `nanotlb_invalidate()` *because the
TLB changed*: `exec_tlbwi`, `exec_tlbwr`. Also bump on snapshot restore and on
any debug/monitor path that writes TLB entries directly.

Do **not** bump on: exception delivery, ERET, `on_cp0_status_changed`,
`MipsCpu::stop()`. Those are the calls the redesign is trying to eliminate. Two
of them need separate treatment:

- `on_cp0_status_changed` must instead recompute `cur_sec_mask` (and keep
  calling `update_translate_fn`). Cheap, no flush.
- `MipsCpu::stop()`'s call exists to null `self.pcp` for jitv2, which is a
  **Fetch**-side concern — it stays with the fetch slot (§7), and no longer
  drags the data side down with it.

`u32` wrap: at one bump per TLBWR, a 4-billion-write wrap is ~hours of guest
time and the failure mode is an entry that is stale *and* happens to land on
the exact wrapped generation — vanishingly unlikely but not impossible. Handle
it explicitly: on wrap to 0, zero both nutlb arrays and restart at 1. That's a
~1 KiB memset once per 2^32 TLB writes; the branch is predicted-not-taken and
free.

Do **not** bump `tlb_gen` on `MTC0 EntryHi` or `TLBR`. Those change the current
ASID, not the TLB, and the whole point of putting ASID in the tag is that an
ASID change costs nothing on the data side — entries for the old ASID simply
stop matching and are reclaimed by ordinary replacement. Bumping the generation
there would throw away every global-page entry too, which is exactly the
over-invalidation nutlb exists to avoid.

The fetch entry is the opposite case: it is untagged by choice (§7) and *does*
get flushed at those two sites. Same event, deliberately different response —
see §7's closing note on the asymmetry.

---

## 5. ERL — recommendation *(partly historical)*

> **Partly superseded.** The conclusion — *do not cache ERL translations at
> all* — survives, and `nutlb_fill` still declines the fill while
> `Status.ERL=1`. The mechanism does not: there is no `cur_sec_mask` and no
> ERL guard bit, so the test is a direct `cp0_status & STATUS_ERL` check on the
> cold fill path rather than a bit folded into the hot-path AND. That folding
> is precisely what made the permission test defective (§11.3).


This is the part of your sketch that was left open, and I think the ERL-bit
idea should be **dropped** in favour of not caching ERL translations at all.

What ERL actually does (`translate_32bit_impl` segment 0..=3, and
`translate_64bit_impl` top_bits 0): when `Status.ERL=1`, KUSEG/xuseg stop being
TLB-mapped and become **unmapped, uncached identity** — `TranslateResult::ok(va,
TR_UNCACHED)`. Every other segment is unaffected.

Adding an ERL bit to the tag (your bit 11, level `0xE`) works, but it buys
nothing:

- ERL=1 is a *narrow, rare* window. It is set at reset (`cp0_status = BEV|ERL`)
  and by cache-error exceptions, and cleared by the first `ERET`. IRIX spends
  effectively none of its runtime there.
- The translation it produces is **arithmetically free** — identity, uncached.
  There is nothing to cache. A nutlb hit would save less work than the tag
  comparison costs.
- It doubles the state space of the permission bits and makes the mask test in
  §3a harder to keep to one AND.

**Recommendation:** treat ERL as a *do-not-cache* condition for the KUSEG/xuseg
identity case only.

- On **fill**: the slow path knows it took the ERL branch. Skip the fill. One
  predictable branch on an already-cold path.
- On **probe**: nothing special needed *provided* the entries that could be
  wrong are unreachable. They are: an entry filled while ERL=0 for a KUSEG page
  carries permission bits `0b111` and would still hit under ERL=1 and return a
  *mapped* translation where the architecture demands identity/uncached. So a
  probe guard **is** required.

Cheapest correct guard: fold ERL into `cur_sec_mask` as a **separate bit that
no cached entry ever sets**. Reserve bit 11 as "entry is valid only when
ERL=0", and have every fill set it:

| | `cur_sec_mask` |
|---|---|
| User, ERL=0 | `0x0900` (bit 11 + user) |
| Supervisor, ERL=0 | `0x0A00` |
| Kernel, ERL=0 | `0x0C00` |
| any mode, ERL=1 | `0x0000` |

With `cur_sec_mask = 0` under ERL, `(virttag & cur_sec_mask) != 0` is false for
every entry, so **the whole nutlb transparently disables itself while ERL=1**
and every access goes down the slow path, which already implements ERL exactly
right. Zero extra hot-path instructions (the AND was already there), zero
special cases, and the ERL window is too rare for the lost hit rate to matter.

Note this uses bit 11 with the *opposite* sense to your sketch — it marks
"cached entries are valid", not "this is an ERL page" — which is what makes the
single-AND trick work. Entries always set it; the mask clears it. So the
`0xF00`-style masks in §3a become `0x0F00` covering bits 11:8 as intended, with
bit 11 always set in tags.

`STATUS_EXL` needs no such treatment: EXL only forces *privilege* to Kernel,
which `get_privilege_mode` already folds into `cur_sec_mask` at
`on_cp0_status_changed` time.

---

## 6. Cached current-context values *(Generation 1 — historical)*

> **Superseded.** `cur_sec_mask`, `cur_asid` and `tlb_gen` are all gone, along
> with `refresh_nutlb_context()`. Nothing per-context is cached, because
> nothing in the tag depends on context.


`MipsCore` gains, all recomputed in `on_cp0_status_changed` / EntryHi writes:

```rust
pub cur_sec_mask: u64,   // 0x0900 / 0x0A00 / 0x0C00, or 0 when ERL=1  (§5)
pub cur_asid:     u8,    // cp0_entryhi & 0xFF
pub tlb_gen:      u32,
```

`cur_asid` must be refreshed wherever `cp0_entryhi` is written: `write_cp0` reg
10, `exec_tlbr`, and `update_tlb_exception_registers` (which rewrites EntryHi
on every TLB exception but *preserves* the ASID — so it can't actually change
it; refresh anyway for uniformity, or assert). Keeping `cur_asid` as a mirror
avoids a dependent load of `cp0_entryhi` + mask in the hot path.

Alternative considered and rejected: read `cp0_entryhi & 0xFF` inline on every
probe. It's one extra AND on a field that's already in the same cache line as
the rest of the CP0 block, so the cost is near zero — but the mirror is
strictly cheaper and the refresh sites are few and already well-defined.

---

## 7. The fetch slot: tag it, or invalidate on ASID write?

Fetch keeps a **single specialised entry**, not an array — instruction fetch is
overwhelmingly sequential-within-a-page, and the 4-way experiment in
`nanotlb_associativity.md` found extra associativity actively harmful, a finding
that applies most strongly to the fetch stream.

That leaves a genuine choice for how it stays coherent, and the two options are
not obviously separable in value:

**Option A — full tag discipline.** Give the fetch entry the same
ASID + permission + `tlbgen` tag as the data arrays. Exceptions and ERET stop
invalidating it entirely.

**Option B — keep the untagged one-compare entry, add an ASID-write flush.**
Leave the incumbent's `va_tag == va_page | 1` test exactly as-is, and simply
extend the existing barrier to cover the two uncontained sites from §1a: flush
the fetch entry from `handle_cp0_side_effects` on reg 10, and from `exec_tlbr`.

**Recommendation: Option B, and it is not a consolation prize.** The reasoning:

- The fetch entry's hit test is its whole value. It is one 64-bit compare
  against a single word, on the hottest path in the emulator. Option A adds a
  gen compare and a permission AND to *every instruction fetch* to buy coherence
  across an event — bare ASID change — that a kernel performs a handful of
  times per context switch. That is a bad trade at fetch frequency even if the
  hit rate improves slightly.
- The ERET/exception flushes that Option A eliminates are worth much less on
  the fetch side than the data side. After an ERET the PC has, by construction,
  just moved to a different page (the return target), so the entry was going to
  miss on the next fetch regardless. The flush is mostly free because it
  discards something already dead. On the data side the opposite holds — the
  working set usually *survives* the transition, which is precisely why §1.2 is
  the bulk of the win, and why that argument does not transfer here.
- Two flush sites, both cold (`MTC0 EntryHi`, `TLBR`), close the §1a gap
  completely for both the fetch entry and — during the transition period —
  the incumbent data slots too. It is a two-line change that can land
  independently of, and before, all the nutlb work.

So: **add the ASID-write invalidation regardless** — it is cheap, it is correct,
and it stands on its own. Then let the fetch entry keep its one-compare test.
If fetch-side measurement later shows the ERET flushes actually costing
something, Option A remains available as a follow-up; nothing here forecloses it.

Note this means the fetch entry and the data arrays end up with *different*
coherence strategies — barrier-based and tag-based respectively. That asymmetry
is intentional and should be commented at both definitions, or a future reader
will "fix" the inconsistency.

### jitv2 coupling

Unchanged by Option B, which is a further argument for it: `self.pcp` is
re-derived only on a fetch **miss** via `jitv2_track_pcp`, and the fetch-only
`nanotlb_invalidate()` still nulls `self.pcp`. `MipsCpu::stop()` keeps calling
it, and the ERET/exception flushes stay, so the invalidation cadence
`jitv2_track_pcp` sees is exactly what it sees today.

Option A would have *reduced* that cadence (fetch entry surviving ERET), which
is the single riskiest interaction in the change and would need validating
against `rules/jitv2/` first. Option B sidesteps that risk entirely. The two new
flush sites only *add* invalidations, which is always safe for `pcp`.

## 8. Sizing — measured

`NUTLB_BITS = 8` (256 entries/array) — unchanged in Generation 2, though the
entry is now 16 B, so the total is 8 KiB rather than 16 KiB. The sweep was run
on Generation 1 (`nutlb,tlbstats`):

| bits | entries | read hit | MIPS | DMIPS |
|---|---|---|---|---|
| **8** | 256 | 97.3% | 59.5 | 73.3 |
| 9 | 512 | 98.0% | 62.3 | 75.2 |
| 10 | 1024 | 98.2% | 62.1 | 73.5 |

(Absolute MIPS is depressed across all three by the `tlbstats` counters; only
the *comparison between rows* is meaningful.)

More sets buy 0.7–0.9pp of hit rate for 2–4× the footprint, and MIPS is flat
within noise. **256 stays.**

The reason more sets don't help is in §9's conflict breakdown: 95–100% of
misses are same-set collisions, so the binding constraint is direct-mapping,
not capacity.

**And capacity is not the lever, because the workloads that matter don't fit
and are structured to defeat set-indexing anyway.** `test/ib/iris_bench.c` uses
~4 MB (~1024 pages vs 256 entries) and sweeps it *linearly* in 32 KB chunks —
the pathological pattern for a direct-mapped array, since it touches every set
in order and wraps having evicted the lot. The guest-level suite
(`bench/irix/steps.toml`) is worse still: a 16 MB file (4096 pages) and a 256 MB
stream in `mem/syscall_copy`. Going to 1024 entries does not rescue any of them.

This also revises the conflict-miss reading below: a linear sweep produces a
~100% conflict signature with ordinary 4 KB pages, so those numbers are **not**
by themselves evidence of large-page aliasing (§10).

The obvious next experiment is therefore **2-way associativity**, which attacks
conflicts directly at 2× the tag comparisons — not a bigger array.

Treat that as unproven: `nanotlb_associativity.md` records a 4-way/AVX2 nanotlb
that *lost* to the 1-way version. That result was on a 3-entry structure where
associativity fought a nearly-free hit test, so it does not transfer
automatically — but it is a standing warning that this exact idea has already
failed once here. Measure on live IRIX before believing it.

32 B/entry (padded from 24 B) was also never validated on its own: it made
indexing a shift instead of a multiply and kept entries off split cache lines,
but the packed variant was never benchmarked *against it at equal tag width*.
Generation 2 does use 16 B entries — and got faster — but changed the tag at
the same time, so this remains unattributed (§12).

## 9. Results *(Generation 1, vs the nanotlb)*

> These are Generation 1's numbers against the **nanotlb** baseline, and they
> remain the justification for having a data-side micro-TLB at all. They are
> *not* a comparison between the two nutlb generations — that is §12.
>
> One caveat, discovered later: the tagged build's permission test never
> actually rejected anything (§11.3), so these figures were produced by a probe
> doing slightly less work than intended.

### Live IRIX (the number that matters)

| benchmark | baseline | nutlb | delta |
|---|---|---|---|
| whetstone | 125 | 137.9 | **+10.3%** |
| dhrystone | 71428 | 78431 | **+9.8%** |
| ssl / irisbench | 0.61 | 0.69 | **+13.1%** |

### Bare-metal `iris-bench run` (4 runs each, `lightning,rex-jit`, no tlbstats)

| | MIPS | DMIPS | accuracy |
|---|---|---|---|
| baseline (nanotlb) | 62.4–63.4 | 69.6–70.9 | 100% |
| **nutlb** | **64.2–65.7** | **74.1–80.7** | 100% |

**The bare-metal suite understates the win, and the reason matters.** `bench/`
runs bare-metal: almost no syscalls, no context switches, few TLB refills — so
it barely exercises the thing nutlb actually fixes (§1.2, not flushing the data
side on every privilege transition). The one kernel that moved sharply there
was `sys/exception`, 258M ns → 164M ns. Live IRIX is that same effect applied
across the whole workload, which is why it shows ~10-13% against bare-metal's
~3% MIPS. DMIPS (+12%) tracked the real result far better than MIPS did.

**The win is flush-frequency, not residency — and the workloads prove it from
both ends.**

`test/ib/iris_bench.c` (the "ssl" row, openssl+zlib on IRIX) is the strongest
case *against* a residency explanation. Its working set is a 2 MB input plus a
2 MB output buffer — ~1024 pages against 256 entries, 4× oversubscribed — and
it was **deliberately written to be hostile to caches like this one**: it sweeps
the buffer linearly in 32 KB chunks, rotating through 8 different
cipher/hash/compress algorithms to keep L1I under constant pressure. A linear
sweep across 4 MB is the pathological input for a direct-mapped array: it walks
every set in order, wraps, and has evicted everything by the time it returns.
That benchmark still posted the **largest** gain of the three (+13.1%).

From the other end, whetstone and dhrystone have working sets of a few KB —
they fit comfortably in the old 1-entry nanotlb *and* the new array, so neither
can be gaining from residency — and they still gained ~10%.

Something that helps both a workload too large to cache and workloads small
enough not to need caching is not helping by holding data. The common factor is
that all of them trap frequently, and every trap used to flush the data-side
translation twice: on entry, and again on ERET. That is what was removed.

Anyone re-measuring this should treat **live IRIX as the primary signal** and
`iris-bench` as a regression guard, not as the headline.

### Correctness

- `cpu-tests`: identical **2101 passed / 61 failed** with and without the
  feature — the same 15 FPU-flag tests, which are known-deferred and unrelated
  to translation. Zero ISA-level regressions across 240 tests.
- `cargo test --lib` green in all of: default, `nutlb`, `nutlb,tlbstats`,
  `tlbstats`, `nutlb,tlbcheck`.

### Hit rates (`nutlb,tlbstats`, whole bench run)

```
[Read ]  nutlb: total=496032061  hit=97.3%  miss=2.7%  (conflict=95.0% of misses)
[Write]  nutlb: total=155845715  hit=96.4%  miss=3.6%  (conflict=100%  of misses)
[Read ]  translate: calls=2051876          <- vs 496M reads: slow path all but gone
```

**Nearly every miss is a conflict, not a capacity miss.** That is the single
most actionable number here, and it says the limit is the direct-mapped
organization, not the size — see §8.

### The hit test is branchless on purpose *(still true in Generation 2)*

Measured on an isolated 200M-iteration microbenchmark of exactly this test:

| miss rate | `&&` | `&` |
|---|---|---|
| 0% | 3.60 ns | 3.76 ns |
| 2% | 3.78 ns | 3.76 ns |
| 10% | 4.40 ns | **3.74 ns** |
| 50% | 7.56 ns | **3.86 ns** |

`&&` emits two conditional jumps and degrades as misses rise; `&` costs two
extra ALU ops and is flat. Only a ~100% hit rate favours branching. Shipped
with `&`.

## 10. Open questions

Ranked by expected value. Revised for Generation 2 — two entries that referred
to Generation 1 machinery are now moot and marked so.

- **2-way associativity.** Still the obvious next move, and cheaper to try
  now: with a 16 B entry a 2-way set is 32 B, one cache line, and the probe it
  competes against is only two operations. But note that Generation 2 is the
  *second* time a cheaper hit test beat a smarter one here (§0), so weight the
  extra tag comparison accordingly. Original reasoning: 95–100% of misses are
  same-set conflicts, and more sets demonstrably don't fix that (§8). Caveat
  in §8 — the 4-way nanotlb experiment already lost once. Note also that a
  *linear sweep* (what `test/ib/iris_bench.c` does) is not helped much by
  2-way either: it evicts on the wrap regardless of associativity. Associativity
  pays against interleaved streams (memcpy src/dst, stack+heap), which is a
  different access pattern than the one currently being measured — so pick a
  benchmark that actually exhibits it before drawing conclusions. Measure on
  live IRIX, not `iris-bench`.

- **Page size.** Everything here assumes 4 KiB. The TLB supports `PageMask`,
  and IRIX 6.5 does use larger pages; a large-page translation cached at 4 KiB
  granularity is *correct* but occupies one entry per 4 KiB sub-page, so a
  16 KiB page burns four consecutive sets. That is a conflict-miss generator —
  **but do not read §9's ~100% conflict rate as evidence for it.** The
  benchmark that produced those counters sweeps ~4 MB linearly, which yields
  the same signature with plain 4 KiB pages. Distinguishing the two needs
  direct instrumentation of what page sizes IRIX actually maps, not inference
  from the conflict counter.

- **~~`nutlb_perm_bits` must track the segment decode.~~** *Moot in
  Generation 2* — the function is gone, along with every permission bit. What
  replaced it, `nutlb_cacheable_segment`, only rejects VA shapes that are not
  valid addresses at all, so drifting out of sync with the segment decode
  costs at worst a spurious miss, never an unsafe cache. The hazard this entry
  warned about is structurally absent.

- **Entry size — still open, and now a confound.** 32 B padded vs 16 B packed
  was never benchmarked in isolation (§8), and Generation 2 changed it at the
  same time as the tag, so §12's win cannot be attributed between the two. A
  16 B *tagged* build would separate them.

- **jitv2 fetch-slot longevity (§7).** Still moot, and more firmly so: the
  fetch slot stays barrier-flushed, and Generation 2 removed the only tagged
  structure in the emulator, so `jitv2_track_pcp`'s invalidation cadence is
  unchanged. Only becomes live if the fetch entry is ever tagged — for which
  §0's "per-access cost beat per-event cost, twice" is the standing prior.

- **Did the uncontained ASID change in §1a ever fire on real IRIX?** Moot for
  correctness (fixed), still mildly interesting for sizing the §7 flush cost.
  Lowest priority here.

---

## 11. Generation 2 — the shipped design

### 11.1 Shape

```rust
pub const NUTLB_BITS:  usize = 8;                       // 256 entries/array
pub const NUM_NUTLB_WORDS: usize = NUM_NUTLB_ENTRIES / 64;   // 4

#[derive(Clone, Copy, Default)]
#[repr(C, align(16))]
pub struct NuTlbEntry {
    pub virttag: u64,   // VA[63:12] — bare page number, nothing else
    pub phys:    u64,   // PA[63:12] in [63:12]; C-field in [2:0]
}

pub nutlb:       [[NuTlbEntry; NUM_NUTLB_ENTRIES]; 2],   // [0]=Read [1]=Write
pub nutlb_valid: [[u64; NUM_NUTLB_WORDS]; 2],            // one bit per set
```

16 B/entry, so both arrays together are **8 KiB** rather than Generation 1's
16 KiB. The Read/Write split (§2) and the `phys` encoding are unchanged.

**A zero tag is no longer self-evidently invalid.** VA 0 is a legal page
number, so validity comes *only* from the bitmask. Never test `virttag` without
first testing its bit — including in diagnostics: the `tlbstats` conflict
counter has to ask the bitmask, or every post-flush refill is miscounted as a
conflict.

### 11.2 The hit test, and the flush that pays for it

```rust
if (e.virttag == (va & !0xFFF)) & self.core.nutlb_is_valid(arr, idx) {
    return TranslateResult::ok(e.phys_addr(va), e.cache_attr_raw());
}
```

One page compare, one bitmask probe, `&` not `&&` for the reasons in §9. The
bitmask word load is independent of the entry load, so the two issue in
parallel; what Generation 2 actually removes from the *critical path* is the
`asid_mask` load that Generation 1's `want`/`have` computation depended on.

Coherency comes back as flushing. `MipsCore::nutlb_clear()` zeroes the bitmask —
8 stores, independent of how many entries were live — and it is called from
**`MipsExecutor::nanotlb_invalidate()`**, deliberately, rather than from the
~10 individual barrier sites. That is the single most important structural
decision in this generation: the nutlb and the nanotlb now share one barrier,
so a future barrier cannot cover one structure and silently miss the other.
Generation 1's call sites each needed their own `refresh`/`bump`/nothing
decision, and getting one wrong was a silent correctness bug.

Sites that flush (all via `nanotlb_invalidate`): exception delivery, ERET,
`on_cp0_status_changed`, `MTC0 EntryHi`, `TLBR`, `TLBWI`, `TLBWR`,
`MipsCpu::stop()`, snapshot restore.

The JIT's inline probe in `jitv2/codegen.rs` mirrors this exactly and dropped
from ~15 IR ops to 6. Any divergence between the two is a correctness bug, not
a performance one.

### 11.3 Why Generation 1's permission test never worked

Worth recording in full, because the flaw is invisible in the algebra §3a
checks and was carried through review, implementation and a full benchmark run.

`NUTLB_TAG_ERL_OK` (bit 11) was set in **every filled tag** *and* in **every
non-ERL `cur_sec_mask`**:

```
NUTLB_SEC_USER = ERL_OK | USER   = 0x900
virttag(kseg0) = ERL_OK | KERNEL = 0xC00
0xC00 & 0x900  = 0x800           -> nonzero -> "permitted"
```

The subset test `(virttag & cur_sec_mask) != 0` was therefore satisfied by the
**shared ERL bit alone**, with no privilege bit in common. A kernel-filled
KSEG0 entry stayed readable from user mode. Observed directly:

```
virttag=0xffffffff80004c00 sec_mask=0x900 perm_and=0x800  -> hit
```

§5's own claim is where it went wrong: folding ERL into the mask costs "zero
extra hot-path instructions (the AND was already there)". That is exactly the
problem — **one AND cannot carry both an always-set flag and a subset test**,
because the always-set bit alone satisfies `!= 0`. The fix would have been to
mask the privilege bits separately (`& NUTLB_PERM_MASK`), i.e. the extra op
§3a talked itself out of.

Never reachable in practice on IRIX — it needs a user access to a VA a
kernel access cached in the same set, with no intervening flush — but it was a
real hole, not a theoretical one. Generation 2 does not share it: no
permission bits exist, and the flush at every privilege transition is what
enforces the property. `test_nutlb_kernel_entry_unreachable_from_user`
(`mips_exec_test.rs`) is the regression test, verified to fail when the flush
is removed. Full writeup:
`rules/testing/nutlb-erl-guard-defeats-permission-test.md`.

---

## 12. Generation 2 vs Generation 1 — measured

Live IRIX, `lightning,rex-jit,j2wp,tcache`:

| benchmark | Gen 1 (tagged) | Gen 2 (bitmask) | delta |
|---|---|---|---|
| whetstone | 416 | 400 | **-3.8%** (lower is better) |
| dhrystone | 263157 | 277777 | **+5.6%** |
| ssl / irisbench | 2.98 | 3.27 | **+9.7%** |

**The prediction was wrong, and it is worth being precise about how.** The
argument against Generation 2 (§0, and §1.2/§3 throughout) was that flushing
256 entries on every syscall would cost far more than 3–4 ALU ops per access
could save. The error was costing the flush as "discards 256 live
translations". What a flush actually costs is the refill of however many sets
are **re-touched before the next flush** — and between two syscalls, IRIX
touches far fewer than capacity. Meanwhile the probe savings are paid on
*every single load and store*, which is several orders of magnitude more
frequent than the barriers.

Flush-based coherency scaling badly with structure size is still true in
general; it just isn't the binding constraint at this structure's size and this
barrier frequency.

**Confound, unresolved.** Generation 2 changed two things at once: the tag
*and* the entry size (32 B → 16 B, arrays 16 KiB → 8 KiB). Some unknown share
of the win is host D-cache footprint rather than the shorter probe. Separating
them needs a third build — 16 B entries with tags retained — which was never
run. §8 already flagged packed-vs-padded as unbenchmarked; it still is. This
only matters if someone wants tagged entries back, in which case a packed
tagged entry might recover most of the difference while restoring flush-free
transitions.

---
