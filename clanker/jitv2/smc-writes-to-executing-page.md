# Self-modifying writes onto the page being executed (`jitv2_smc_check`)

## What the feature does

`jitv2_smc_check` (Cargo feature, off by default) reports every data write
whose **translated physical address** lands inside the physical frame the CPU is
currently fetching from.

- `MipsCore::cur_code_pfn` mirrors `MipsExecutor::pcp`'s `pfn`. `u32::MAX` =
  nothing tracked.
- Written only via `set_pcp`/`clear_pcp` — six assignment sites previously, and
  a mirror that can drift is worse than no mirror.
- Checked in `write_data_impl`, **after** translation, **before** the write
  commits. A code page is identified by physical frame; the writer's virtual
  alias is usually not the fetcher's, so comparing VAs would miss everything.
- Deduplicated on `(pc, target pfn)`; `SMC_HITS` counts pre-dedup and prints at
  shutdown.

## Scope: interpreter path only

The check lives in `write_data_impl`, which is the **interpreter's** store path.
JIT stores reach it only through the `jit_write*_fn` callouts. The inline store
fast path (`codegen::emit_mem_write_split`) writes the L1-D data array directly
and never enters `write_data_impl`.

**So: run with `j2 inline_mem off`.** (There is no longer a `nutlb` feature to
build without — it is unconditional.) Otherwise a
clean result means "no SMC among writes that took the slow path", which is much
weaker than "no SMC". Pair any zero with the `jitv2 inline memory: emitted=…`
line before concluding anything.

## Observed on a real run

Full IRIX startup + SoftWindows 95 + shutdown produced only **8** distinct
reports. Three distinct shapes:

```
pc=0x0000000010000038  va=0x10000200  phys=0x08a0d200  pfn 0x8a0d  word 128
pc=0xffffffffc00023b8  va=0xc0002620  phys=0x0cd02620  pfn 0xcd02  word 392
pc=0xffffffffc00023cc  va=0xc000261c  phys=0x0cd0261c  pfn 0xcd02  word 391
pc=0xffffffffc000246c  va=0xc000261c  phys=0x0cd0261c  pfn 0xcd02  word 391
pc=0xffffffffc0002478  va=0xc0002620  phys=0x0cd02620  pfn 0xcd02  word 392
pc=0xffffffffc04a6b90  va=0xc04a6d80  phys=0x148a6d80  pfn 0x148a6 word 864 (8B)
pc=0xffffffffc0084b90  va=0xc0084d80  phys=0x12284d80  pfn 0x12284 word 864 (8B)
pc=0xffffffffc02b2b90  va=0xc02b2d80  phys=0x140b2d80  pfn 0x140b2 word 864 (8B)
```

1. **User text (`0x10000000` base), word 128** — one shot, looks like a
   loader/startup fixup.
2. **pfn `0xcd02`, words 391/392, four writes** — a small code region writing
   two fixed words just past itself, each twice.
3. **word 864, 8 bytes, three different pfns** — note `pc & 0xFFF == 0xb90` and
   `target & 0xFFF == 0xd80` in all three: the *same routine* mapped at three
   addresses, writing 8 bytes 0x1F0 past itself.

## The open question — do not skip this

**A write onto a code page is not necessarily a write to executed code.** In
every sample above the target word is far from the pc (word 864 vs pc-word 748;
words 391/392 vs pc-words 142–147). These are very plausibly *data* — literal
pools, GOT-style slots, lock/counter words — that merely share a 4KB page with
code. Shape 3's fixed `pc→target` delta across three mappings is exactly what a
per-instance data slot in a shared text page looks like.

It only matters if the JIT **compiled a region covering the written word**. To
answer that, correlate each report's `(pfn, word)` against the compiled region
for that pfn (`j2 pcp`) — is the written word inside a published/runnable
region, or in a part of the page nothing ever compiled?

Until that correlation is done this is a **lead, not a diagnosis**. 8 events
across a whole boot is low enough that it may be entirely benign, and the
SoftWindows shutdown misbehaviour may well have an unrelated cause — a plain
jitv2 correctness gap remains at least as likely.

## Testing note

`smc_check_fires_on_write_to_executing_page_and_not_elsewhere` asserts **both**
directions — fires on same-page, silent on a different page. The negative half
is what makes a clean live run meaningful; a check that fired on everything
would also "detect" SMC.

Two traps hit while writing it, both worth remembering:

- `MipsExecutor::exec()` is the debug/single-injection entry point and feeds
  `core.pc` to `jitv2_track_pcp` **as though it were already physical**. For a
  kseg0 pc that yields a virtual pfn which can never equal a store's physical
  pfn, so the check is permanently silent under `exec()`. Drive it with
  `step_jit()`.
- `MockMemory::set_word(pc, …)` writes the virtual address only; the fetch path
  reads the translated physical one. Seed both aliases or the CPU fetches zeros
  and executes a nop instead of the store under test.

`SMC_HITS` is a process global and the suite runs in parallel, so the test
serializes on `FALLBACK_TEST_LOCK` — without it the negative assertion is flaky
(observed once).
