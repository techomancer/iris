# A page's pinned FR mode must be re-derived, not pinned for life

`j2wp` compiles one function per physical page, with FR mode baked into every
FPR-access emitter at compile time. `PhysicalCodePage::fr1` records which mode
that function was built for. It used to be stored once at `claim` and **never
reset by anything** — verified across `page_for` (ignores its `fr1` argument on
a lookup hit), `publish`, `reset_for_flush_survivor` and `reset_to_unclaimed`.

## Why that was wrong

`publish` already reasons about this correctly for its neighbours. On a real
invalidation (`snap_gen > prev_entry_gen` — the guest replaced the page's bytes)
it *replaces* rather than unions `compiled`, and resets `denied`, with the
comment:

> the bytes changed, so whatever got sticky-denied against the OLD bytes
> deserves a fresh chance against the new ones

**`fr1` has identical justification and was the one field left behind.** A
physical page freed by an o32 (FR=0) process and reallocated to an n32 (FR=1)
one arrived with entirely new bytes but kept the previous tenant's pin.

## Why the guard did not catch it

`emit_fr_mode_guard` computes:

```rust
let fr_mismatch = ctx.builder.ins().band(cu1_set, fr_mismatch_if_cu1_set);
```

The FR check is **deliberately suppressed when CU1 is clear**, so it cannot
pre-empt the real per-instruction CU1 fault (a live bug its own comment
documents). Correct for that case — but it means a stale pin is *not* reliably
caught. A region entered with CU1 clear and FR already flipped runs unguarded,
and the per-instruction CU1 check only fires if control flow actually reaches a
CP1 instruction.

Wrong FR packing on `ldc1`/`lwc1` puts an FP value in the wrong register half:
**silent data corruption, not a fault.**

## Symptom

4Dwm / background / fm core-dumping at session start — a burst of process
creation, which is exactly a burst of page recycling — **cleared by `j2 flush`**.

`j2 flush` curing a symptom is a reliable fingerprint for mis-specialized
compiled code: a flush discards compiled code and forces recompilation from
current memory, so if it helps, the compiled function disagreed with what the
guest would actually execute. That rules out the whole interpreter side.

## The fix — two halves, neither subsumes the other

1. **`publish` drops the pin on a real invalidation** (`fr_unpinned`). New bytes
   ⇒ the old mode says nothing about what they want. Don't pin a mode here; the
   compiling thread's notion of "live" is the one that matters and is read at
   compile-request time. This is the half that addresses page recycling.
2. **`request_fr_repin` records the demanded mode on an FR-guard bail**
   (`fr_repin`), consumed by `take_fr_repin` at the next compile. Handles what
   (1) cannot see: the *same* bytes arriving in a new mode — the shared
   `checkfp.s` page, which runs `lwc1`×32 under FR0 and `ldc1`×32 under FR1.

### Store the demanded mode, not a "flip me" bit

Several bails normally land before the next compile: a page has many published
entries, the guard kills them **one at a time as each is dispatched**, and each
kill records a demand. A "flip the pin" bit would therefore toggle once per
killed entry — an even number of kills settles back on the *wrong* mode.

Recording the observed live FR is idempotent instead: N bails demanding FR1 all
say "FR1", and the recompile reads a value rather than applying a delta.
Regression test: `repeated_repin_requests_are_idempotent`.

This is about repeated bails, **not** a compile-vs-compile race: `try_schedule_page`
is a test-and-set (`page_scheduled.swap(true)`), so there is only ever **one
outstanding compile request per page** — every other caller sees `false` and
skips. `take_fr_repin` consumes the request with a `swap` anyway, which is cheap
and keeps the one-shot semantics local rather than resting on that invariant.

### Live FR at `jit_kill_entry` *is* the demanded mode

`emit_fr_mode_guard` is the **only** caller of `emit_kill_entry` (verified), and
it only fires when CU1 is set and live FR disagrees with the compiled mode. So
reading `core.cp0_status` in `jit_kill_entry` gives the mode the dispatch
actually wanted. No codegen or ABI change was needed — that callback already
held both the executor and the page.

### Already-published entries are retired lazily

Deliberate: the per-entry guard kills wrong-mode entries one at a time and
`mega_flush` recycles the page eventually. A page-wide kill on re-pin would
discard correct code — non-FP entries on the same page are mode-independent —
to save a bounded number of guard failures.

### A mode flip must also reset `denied`

`reset_for_flush_survivor` deliberately **preserves** `denied` (that is its
churn-reduction purpose), so the flush-survivor path clears it explicitly when
the mode actually changes. The denylist records verdicts reached while compiling
for the old mode; those no longer describe the code that will be generated.

## What the old TODO got wrong

The field's TODO justified never re-pinning as *"measured at ~2 occurrences per
real IRIX boot (rare)"*. That measured a **different scenario** — a page whose
entries all get killed — and said nothing about page recycling at process
startup. A measurement of the wrong population is not evidence about this one.
