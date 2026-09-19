# Compile churn avoidance: skip a recompile whose inputs did not change

`j2wp` only (the whole-page design). Measured on an IRIX 6.5 boot:
**~286k compiles skipped, 98.0% of everything checked.**

## The problem

Under `j2wp` a page's compiled function is invalidated by its *generation
counter*, which the bus bumps on any write anywhere in the 4KB page. Code and
data share pages constantly, so a page gets its generation bumped for reasons
that have nothing to do with the instructions on it — the guest writing a
neighbouring struct, DMA landing in the page, a data constant being updated.

Every such bump used to cost a full analyze + codegen + finalize pass that
emitted **byte-identical code**.

`prepare_multi_entry_compile`'s §13.3 step 4 subsumption check cannot help:
it gates on `same_gen` (`entry_gen == gen_snap == current_gen()`), and a gen
bump is precisely what makes that false. It only catches the "two compiles
raced for the same generation" case.

## The mechanism

`PhysicalCodePage` keeps a `CompileSnapshot` of what its last successful
compile actually looked at:

- `words` — the full 4KB seqlock byte snapshot that compile analyzed
- `used` — one bit per word the walk **decoded** (from `instrs_linear`,
  delay slots included)
- `entries` — the entry-point bitmap it published
- `fr1` — the FR mode it was specialized for

On the next request, `try_skip_redundant_compile` skips the compile iff all of:

1. the wanted entry points are a subset of what's published (the installed
   function's dispatch switch has cases for nothing else),
2. every word in `used` is byte-identical, and
3. the FR mode matches.

On success it advances `entry_gen` to the new `gen_snap` under `publish_lock`
— the same single field write `publish` would have done — which is what makes
`is_runnable` start returning true again. `func` and `compiled` are untouched.

**Comparing only `used`, not the whole page, is what makes this fire.** The
data that changed was by definition never decoded, so it cannot change what
codegen would emit.

## The skip must end in the same state a real publish would

**This is the part that is easy to get wrong, and getting it wrong looks like
the optimization working while throughput drops.** Measured live: R4K skipped
a great many compiles and ran *slower*.

After a gen bump `same_gen` is false, so
`snapshot_compile_candidates(false)` drops `compiled` and the candidate set is
just `requested & denied` — a handful of freshly-requested offsets, not the
page's full coverage. A real compile walks that set, and `publish` sees
`snap_gen > entry_gen` (`is_real_invalidation`) and **replaces** `compiled`
with it wholesale.

Two bugs came out of not mirroring that:

1. **A skip that only bumped `entry_gen`** left the older, wider `compiled`
   published. `is_runnable` then returns true for offsets the check never
   verified; dispatch calls `func`, the switch `compile_region` built has no
   case for that offset, and it returns **`EXEC_FALLBACK`** — every arrival,
   forever, each one re-requesting a compile. Fix: replace/union `compiled`
   exactly as `publish` does.

2. **Narrowing `compiled` then requiring `wanted_entries ⊆ compiled`** on the
   next request refused to re-cover an entry the previous skip had dropped,
   even though the *same installed function* still has a case for it —
   forcing a pointless recompile. Fix: check against `snap.entries` (what the
   function can serve, fixed while it stays installed), never `compiled`
   (what is currently advertised, which a narrowing skip changes).

The separation to hold onto:

| | meaning | changes when |
|---|---|---|
| `snap.entries` | what the installed `func` *can* serve (its switch cases) | only on a real compile |
| `compiled` | what is currently *advertised* to dispatch | also on a narrowing skip |

A skip also **folds the entries it drops back into `requested`**. Unlike
`publish`'s equivalent branch the bytes here are provably unchanged, so a
dropped entry is still legitimately wanted — nothing re-requested it only
because it was already covered. Without the fold it is stranded on the
interpreter with no automatic path back, the same failure
`reset_for_flush_survivor` documents.

Why R4K showed it worst: the 1-way L1 means more conflict misses and
writebacks, so more generation bumps per unit work — more trips through
exactly the broken path.

## `kill` needs its own flag

A `kill` means "stop dispatching into `func` here", so a skip must not
re-advertise that offset. `compiled` cannot express this any more, because a
narrowing skip clears bits there for an unrelated reason — hence
`killed_since_snapshot`, set by `kill` and cleared by
`stage_compile_snapshot`.

In production every `kill` is already paired with an FR repin (caught by the
`fr1` check) or a `denylist` (caught by `denied`, which the caller folds into
`wanted_entries`), so the flag is defence in depth — but it is the one that
states the actual invariant, and a unit test killing an entry directly caught
its absence.

## Three things that look optional and are not

### The FR mode must be part of the comparison

FR mode is baked into every FPR-access emitter at compile time and is
**invisible in the page bytes**. Identical bytes compiled for the other mode
are genuinely different code. Skipping on a mismatch leaves the wrong mode's
function installed, and `emit_fr_mode_guard` is not a backstop — it
deliberately suppresses itself when CU1 is clear (see `fr1`'s field doc), so
the failure mode is `ldc1`/`lwc1` packing an FP value into the wrong register
half: silent corruption, not a fault.

### A skip must NOT reset the denylist — but `used` must cover every denial

`publish` clears `denied` on a real invalidation (§13.6: new bytes deserve a
fresh chance). A skip proves the *opposite* — the decoded bytes are identical
— so an offset denied against them is still correctly denied. Clearing it
would re-walk and re-reject it on every generation bump, which is the exact
churn this exists to remove.

That only holds if every denial's *evidence* is inside `used`, so a rewrite of
the code that earned it fails the byte compare. **Every production denylist is
a compiler verdict** (nothing in CPU execution denylists; the only other site
is the `j2 deny` monitor command), with three reasons:

| Reason | Denies | Evidence in `used`? |
|---|---|---|
| Entry excluded at its own word (`EntryExcluded`) | that candidate | **No — needs an explicit fold** |
| Merged region below the floor (`TooShort`) | every covered offset | Yes, covered ⇒ visited |
| Cranelift declined / verifier error | every covered offset | Yes, the un-emittable instruction was visited |

The first is the odd one out: `visit` returns `false`, so `walk_multi_entry`
never marks that word visited and `instrs_linear` never reports it.

**Per-reason reasoning is not enough, though.** `denied` is sticky page state
that outlives the compile that set it, so "this compile's denials are covered
by this compile's `used`" fails on a later **same-generation** compile: a
pure-added-coverage publish does *not* reset `denied`, it re-stages the
snapshot, and its own walk has no reason to touch a word denied long ago
(`denied` masks that offset out of the candidate set, so nothing walks it).
The denial stays live while its evidence drops out of the snapshot —
permanently, since no skip can then notice the guest rewriting that
instruction into something compilable.

So `prepare_multi_entry_compile` folds the **whole `denied` set** into `used`
when it stages, not just the offsets it denied itself:

```rust
for (i, w) in page.snapshot_denied_raw().iter().enumerate() {
    used[i] |= !*w; // `denied` is inverted: 0 = denied
}
```

**At stage time, not skip time**, deliberately: it makes the snapshot
self-contained (bytes plus every word that justified a carried-forward
decision), keeps the skip a pure comparison against a fixed mask with no
second source of truth, and pays the cost once per compile rather than once
per skip. Pinned by
`a_denied_offsets_word_stays_compared_across_a_later_same_gen_compile`.

### Denials from the CPU thread need no handling

The CPU thread never denylists — it only `kill`s. Two independent reasons a
CPU-side denial could not break the fold anyway:

- **Structural:** the CPU only reaches an offset it was executing, so that
  offset was published, hence in `covered` at its last compile, hence already
  in `used`.
- **Mechanical:** every CPU-side path that touches denial state (`j2 deny`,
  and the FR guard's `jit_kill_entry`) also calls `kill()`, which sets
  `killed_since_snapshot` and refuses the next skip outright. The FR guard
  additionally sets `fr_repin`, caught by the `fr1` check.

Pinned by `every_cpu_thread_denial_path_also_kills_which_blocks_the_skip`.

The residual, accepted gap: a denial acquired *after* staging isn't in the
mask until the next real compile re-stages. Since denials come only from
compiles, and a page is never compiled twice at once, the only way to hit this
is the manual `j2 deny` debug command — which kills too, so the skip is
blocked regardless.

### Stage at compile time, commit at publish time

A snapshot must never be consultable before the code it describes is
installed. The deferred path (`handle_request_deferred`) hands its compile to
the shared arena's seal queue, and the matching publish can happen later on
another worker's `finalize` call with only a `PublishInfo` in hand — far too
late and too small to carry 4KB. So `stage_compile_snapshot` records it
*invalid*, and `publish_all`/`handle_request` call `commit_compile_snapshot`
once the publish lands.

The commit is matched on `gen_snap` + `entries` so it can only validate its
own record. The case that guards is a **reset landing under an in-flight
compile**: `invalidate_compile_snapshot` poisons `staged_gen` to `u64::MAX`
(never a real generation), so a compile in flight across a flush cannot come
back and re-validate a snapshot for a function the flush threw away.

## A page is never compiled twice at once

`compile_snapshot` is a plain `UnsafeCell`, not a `Mutex`, and this is why.

`page_scheduled` is a test-and-set taken by the dispatch gate *before* a
`CompileRequest` is sent, and released only by
`handle_request`/`handle_request_deferred`'s **exit** scope guard — so it is
held for the whole compile, not just the enqueue. A second request for the
same page therefore cannot be sent, let alone dequeued, while the first is
still running, **at any `thread_count`**. Same invariant that lets `Analyzer`
reuse its 1024-entry buffer across compiles without locking.

Don't be misled by `publish_lock`'s field doc ("compiles proceed unlocked and
in parallel, including two compiles of the same page") — that describes what
that lock does not *assume*, not a reachable state. `publish_lock` is still
taken on the skip path, but for a different reason: a dispatch-side `kill()`
can land concurrently, and the `entry_gen` write must not interleave with the
coverage re-check.

## Every path that drops `func` must invalidate the snapshot

`reset_to_unclaimed` and `reset_for_flush_survivor` both null `func`, so both
call `invalidate_compile_snapshot`. A snapshot outliving its function would let
a skip advance `entry_gen` over a null pointer and send dispatch into nothing.

Note `reset_for_flush_survivor` cannot preserve the snapshot the way it
preserves `requested`/`denied`: the arena holding that `func` is being flushed.

`kill()` is different and needs no special handling — it clears a `compiled`
bit while leaving `func` valid for the other entries, and the skip path
re-checks coverage against the live `compiled` bitmap under the lock, so a
killed entry correctly forces a real compile.

## Cost

`PhysicalCodePage` went **720 → 4976 bytes**, so the default 4096-slot pool
is ~20MB instead of ~3.4MB. Inline rather than `Option<Box<..>>` on purpose
(matching the same call made for `entries`): the pool array is allocated once
up front, so boxing would buy only a pointer chase plus an allocation on every
page's first compile. Slots that never compile never touch their copy, so it
stays untouched zero pages.

## Reading the counters

`redundant_skipped` / `redundant_rejected` are per-page, **not**
`developer`-gated (deliberately — `prepare_multi_entry_compile` has no
`JitStats` outside a `developer` build, so routing through `JitStats` would
have made the numbers invisible in exactly the `lightning` build you want to
measure).

- `j2 status` → pool total + percentage
- `j2 pcp` → per-page breakdown

**Both are since-last-flush, not lifetime.** A boot storm's ~286k skips would
otherwise sit in `j2 status` forever and drown out current behaviour; the
flush is the honest epoch boundary anyway, since it invalidates the very
snapshots those counts describe. `rejected_compiles`/`prepare_bounced` next
door are the opposite (lifetime, flush-surviving) — don't "tidy up" the
difference. A request with no snapshot to compare against (first compile, or
post-flush) counts in neither.
