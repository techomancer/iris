# intrfast (0x08016xxx) gets overwritten with data mid-boot — investigation in progress

## Status: root mechanism found, root CAUSE (what issues the write, and why) not yet found.

## Symptom (how this was found)

Testing `j2 fallback` (interp-fallback-in-region), `dt` showed the same
addresses tagged both `[jit:entry]` and `[fallback]` for consecutive NOPs
inside `intrfast`. Chased via `d` (disassemble) showing `nop` at
`0xffffffff88022ab0`/`+4`, while `j2 analyze 0x08022ab0` (reads physical
memory directly via `sysad.read32`, bypassing cache) showed the REAL
instructions there: `mfc0 k0, Cause` / `mfc0 k1, Status` — correctly
classified `Excluded`+fallback, not a bug. `d`'s disassembly was reading
stale/wrong CACHE content, not physical memory.

Second, more precise repro at a different address confirmed this is a real
cache-content bug, not a monitor display bug:

```
> l1i probe 0xffffffff880165d4
  ... set=0x25d: HIT  Way0: tag=0x0008016000 valid=true <-- HIT
> l1d probe 0xffffffff880165d4
  ... set=0x25d: MISS Way0: tag=0x0008012000 (unrelated line)
> l2 probe 0xffffffff880165d4
  ... MISS at index (real backing memory at this phys addr = 0x401a4000, mfc0)
```

L1I reports a valid, tag-matching HIT for cached content that reads as
`0x00000000`, while the real backing physical memory (read directly, bypassing
cache) holds `0x401a4000` (`mfc0 k0, Cause` again — this is `intrfast`'s
entry, reached a second time in the boot at a different point).

## Mechanism, confirmed via `debug_cache` feature live trace

`DEBUG_TRACK_ADDR` in `src/mips_cache_v2.rs` was set to `0x080165d4` to get
per-line tracked debug prints (see the (also fixed this session) hex-format
cleanup of these prints — was decimal `idx={}`/`l1_idx={}`, now `0x{:x}`).

The real sequence, captured live:

```
fetch 0x080165d0..0x080165fc  -> ALL correct (real mfc0/lui/etc instructions)
...
fill_l1d_line: TARGET virt 0xffffffff880165d0 phys 0x080165d0 -> L1D eidx=0x25d
    [0] 0x00000000401a4000   <- correct code bytes, legitimately D-CACHE-FILLED
    [1] 0x3c01c000001ad280
...
write64: TARGET virt_addr 0xffffffff880165d0, phys_addr 0x080165d0, val 0x0000000000000000
...
writeback_l1d_line: TARGET l1_idx=0x25d phys_addr=0x080165d0 DIRTY -> L2
    wrote 2 chunks to L2 idx=0x2cb offset=0xa
    [0] addr=0x080165d0 val=0x0000000000000000   <- clobbers the L2 line
...
fetch 0x080165d4 -> raw=0x00000000   <- CORRUPTED, matches the guard firing
```

**This is not a cache-coherency/inclusion bug.** The cache model behaved
exactly as instructed: a real CPU `write64` (a genuine 8-byte store,
`SD`-shaped) targeted **`virt_addr=0xffffffff880165d0`** — the *exact same
virtual address* (same kseg0 alias) that the code fetches used — and stored
`0`. The D-cache correctly cached it, correctly wrote it back to L2 on
eviction, correctly clobbering the 8-byte chunk that also held the real
`mfc0` instruction at `+4`. Physical memory is presumably ALSO now `0`
in reality (the writeback is exactly what a real R4400 would do) — the
`j2 analyze`/bus-read showing `0x401a4000` was from BEFORE this store
happened (a snapshot at a different point in boot, not a live disagreement
with a "correct" ground truth).

## What's NOT yet known — pick this up next

**Who issues the `write64` to `0xffffffff880165d0`, and why?** This is the
open question. Two live hypotheses, unresolved:

1. This physical/virtual address isn't actually `intrfast`'s *permanent*
   home — PROM or early boot code used it as scratch/bss before the kernel
   proper relocated/copied the real `intrfast` handler elsewhere, and this
   store is expected, harmless overwrite of now-dead scratch space (i.e.
   `j2 analyze`'s "correct" reading was itself just an earlier, transient
   state, not the final one).
2. A genuine kernel/PROM bug (real IRIX bug, faithfully emulated) or —
   less likely but not ruled out — an IRIS emulation bug elsewhere (wrong
   pointer computed, wrong struct offset, use-after-free-style stray write)
   that corrupts what should remain live interrupt-vector code.

## Next steps

- Capture the PC of the `write64` instruction (extend `write_data_impl`
  or `cache.write` with a guard printing `self.core.pc` when `phys_addr`
  falls in this line — the interp_dispatch_one fetch-vs-bus guard added
  this session, still present, catches the SYMPTOM at fetch time but not
  the WRITE's origin).
- Symbol-lookup that PC (`j2 analyze`/`dt`/the symbols table) to identify
  which kernel routine issues it.
- Once identified: check whether this is IRIX's own boot-time behavior
  (compare against real hardware docs / other MIPS emulators if uncertain)
  before concluding it's an IRIS bug.

## Debugging infrastructure added/changed this session (still in tree)

- `interp_dispatch_one` (mips_exec.rs): a `SCRATCH` cache-vs-bus-read guard
  that fires `EXEC_BREAKPOINT` with a diagnostic print on any mismatch
  between the cached fetch and a direct physical read. **Should probably be
  removed or made permanent/gated once this investigation concludes** — it's
  currently unconditional in the `interp_dispatch_one` path (fallback
  dispatch only), not on the hot `fetch()` path.
- `mips_cache_v2.rs`: all `[CACHE DEBUG]` `println!` index/offset fields
  changed from decimal (`idx={}`) to hex (`idx=0x{:x}`) for readability —
  this is a real, permanent improvement, keep it.
- `mips_cache_v2.rs` `fetch()` (non-r5k path): now prints the actually
  fetched `raw` instruction word alongside the address, not just the
  address — real, permanent improvement, keep it.
- `mips_cache_v2.rs` `debug_probe`'s `l1i`/`l2` arms: now print an explicit
  `HIT`/`MISS` verdict up front, not just a per-way `<-- HIT` marker that
  said nothing on a miss — real, permanent improvement, keep it. (l1d's
  verdict line was proposed but not yet confirmed applied — check before
  relying on it.)
- `DEBUG_TRACK_ADDR` (mips_cache_v2.rs, `debug_cache` feature only): set to
  `0x080165d4` for this investigation. Was `0x17fa5ee0` before — restore or
  repurpose as needed; it's a single hardcoded const, not a runtime knob.

## Related

- [[jitv2_lockstep_region_ending_double_page_jump]] — the unrelated bug this
  session started with (fixed, unrelated to this cache finding).

---

## Update (2026-09-10): the SCRATCH guard has been removed

The cache-vs-bus guard this investigation left behind in `interp_dispatch_one`
(`mips_exec.rs`, ~lines 8109-8131) is **gone**. It compared the cache model's
`d.raw` against a direct `sysad.read32` and, on mismatch, printed
`=== CACHE/MEMORY MISMATCH ... ===`, rewound `self.core.pc = pc`, and returned
`EXEC_BREAKPOINT`.

Removed because it was actively harmful, not merely stale:

- **Under `lightning` (the production build) it livelocked.** The run loop
  discards `step_jit`'s return value entirely (`mips_exec.rs`, the 10x-unrolled
  batch), so `EXEC_BREAKPOINT` was ignored — and because the guard *also*
  rewound `core.pc`, the next dispatch re-fetched the same stale line and
  re-triggered. Infinite loop, unbounded stderr spam. `EXEC_BREAKPOINT` is
  supposed to be impossible in a `lightning` build at all.
- **Without `lightning` it stopped the CPU thread outright** (`running.store(false)`),
  which presents as a hang with one line on stderr.
- A mismatch here is usually a **legitimate** stale-I$ window — the guest wrote
  code and has not yet issued `Hit_Invalidate_I` — which on real hardware would
  also execute stale. That is the other reason it must never stop the machine.
- It cost a `debug_translate` + uncached bus read on **every** fallback dispatch.

If the intrfast symptom recurs, recover the guard from git history — but
re-introduce it as a `developer`-gated, rate-limited **log line only**: no
`EXEC_BREAKPOINT`, no `core.pc` rewind.
