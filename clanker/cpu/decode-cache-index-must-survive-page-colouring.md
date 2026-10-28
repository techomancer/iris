# The decode cache index must survive page colouring

Status: fixed 2026-10-27 (hashed index in `src/cpu/decode_cache.rs`).

## Symptom

O2 R10000 interpreter: Dhrystone ran at ~74% of the IP28's, after the
bitmap bug below was fixed, even though both use the same shadow cache and
the same `step_int`. The O2 profile had `DecodeCache::fetch_slow` at ~4%
(0.04% on IP28) and a lot of re-decoding inside `step_int`.

## Cause

The decode cache keeps 64 pages, direct-mapped. It was indexed by
`pfn % 64`. The O2's IRIX colours pages: a page gets the physical colour of
its virtual address, and program and DSO text starts on aligned virtual
addresses. Dhrystone's and libc's text pages came out at PFNs 0x50e80,
0x51180, 0x51200 — all multiples of 64 — and evicted each other on every
call (≈96 M evictions in one 4 M-loop Dhrystone run). Every eviction drops
the page's decoded slots, so each instruction re-decodes.

The IP28 kernel does not hand out pages that way, which is why the same code
looked fine there.

## Rule

Index any physically-keyed, direct-mapped host-side cache by a hash of the
page number, not its low bits. Guest kernels align physical pages on
purpose. `index()` is one multiply and a shift.

## Found with

A throwaway counter in `fetch_slow` keyed by (evicted pfn, new pfn), dumped
every 500 k evictions under an env var. The pairs made the alignment obvious.

## Related: the O2 shadow model ran without RAM (fixed the same day)

`HandoverCache::tcache_bitmap_ptr` returned the full model's bitmap, which
is null without the `tcache` feature (interpreter builds). The memory space
then had nothing to publish into, `hand_over` copied nothing, and the shadow
model's region bitmap stayed 0: every load, store and instruction fetch went
through `Ip32Bus` (the O2 ran at ~15 M instr/s). It now falls back to the
shadow model's own bitmap. Also: `HandoverCache` forwarders must inline the
shadow arm and keep the full (PROM-only) model `#[cold]` out of line, or every
fetch pays an outlined call.
