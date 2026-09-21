# Corpus capture: `j2 corpus`, and the two bitmaps that look interchangeable

Replaces the old `jitv2_corpus_dump` Cargo feature (removed 2026-09-19).

## Why the old capture went away

`jitv2_corpus_dump` wrote a raw 4KB `pfn_XXXXXXXX_off_XXXX.bin` per compile
request, from inside `comp::handle_request`, deduped by a
`PhysicalCodePage::saved_bits` bitmap that existed for no other purpose.
Three problems, and the third is the one that actually bit:

1. It had to be **compiled in before the run**, so you could only get a corpus
   by knowing in advance you'd want one. The 300-page corpus every measurement
   in `rules/jitv2/` cites was a local artifact that was never in the tree, and
   when it was lost there was no way to reproduce it without another
   instrumented boot.
2. **One entry offset per file**, encoded in the filename. A page with 85 entry
   points became 85 near-identical 4KB files.
3. It wrote to the filesystem **from the compile worker**.

`j2 corpus [dir]` instead walks `Jitv2::claimed_pages()` on demand and writes
one `.pcp` per page. The data was always there — the pcp cache holds every
page the JIT touched, with its bitmaps — it just had no way out.

## The trap: `requested` is not a cumulative record

The obvious reading of `PhysicalCodePage::requested` is "every entry offset
anyone ever asked for," which would make it exactly the entry set a corpus
consumer wants. **It isn't.** Under `j2wp` a requested bit is cleared once a
compile covers it, so on a page whose compiles have all landed, `requested`
reads *empty* and `compiled` holds the real entry set.

Measured on the first real 1427-page IRIX corpus:

```
sum(requested)  =    381
sum(compiled)   = 77,309
sum(union)      = 77,690
pages with requested==0 but compiled>0: 1337 of 1427
```

Keying the walk off `requested` alone discards **99.5%** of the corpus — and
it fails silently, reporting a smaller-but-plausible `total_bytes` rather than
an error. Use the union, which is what `jitv2_pcp_dump`'s offline walk already
did:

```rust
(0..ENTRIES_PER_PAGE).filter(|&o| dump.is_requested(o) || dump.is_compiled(o))
```

Do **not** subtract denylisted offsets. A declined region costs a
`compile_region` call returning `None`, counted under `declined=` — a build
that starts declining regions it used to compile is a regression the
measurement should surface, not hide.

## `last_code_size` was gated in the build that could use it

`Codegen::last_code_size` was `#[cfg(feature = "developer")]`. That made the
one number a codegen-size measurement needs available **only** in the build
that invalidates such a measurement: `developer` forces `opt_level=none` and
injects a per-instruction `emit_dev_trace_bp` callout (71% of all callouts in
one measurement). So `zz_corpus_sizes` reported `total_bytes=0` in exactly the
build it was meant to measure, and the useful number only in the build whose
output is not production's.

It is now ungated — one `u32` written once per compile, off the hot path. The
test prints a warning when `developer` is on. See
[[block-fragmentation-blocks-cse]] for the wrong conclusion this once caused.

## Format

`.pcp` is now `IRISPCP2` (`src/jitv2/pcp_dump.rs`), v1 still readable. v2
appends one field, free at capture time: `call_count`, dispatches into the
page, for weighting. A flat byte total counts a page dispatched two million
times and one dispatched twice equally, which makes it a poor proxy for the
code that actually runs.

## No virtual address in this format, ever

A draft of v2 also carried a `vaddr_hint` for symbolization. It was cut before
release, and the reason is worth keeping: **jitv2 compiles PIC precisely
because one physical page is reused across many processes.** The same page is
mapped at many virtual addresses at once, at different ones over time, and at
none once a mapping is torn down. There is no "the" VA for a page, so any
recorded one is authoritative-looking, unverifiable offline, and wrong for
every mapping but the one live at capture.

It also failed on its own terms: `j2 corpus`, the bulk path the format exists
to serve, never has an address to record. In the first real 1427-page corpus,
**0 pages had a non-zero hint** — empty exactly where it was meant to help,
and misleading in the single-page `j2 dumppcp` case where it was set.

Corpora captured during that window are 8 bytes longer than `FILE_LEN` and
still parse correctly: every field sits at a fixed offset from the start, so
`from_bytes` ignores trailing bytes by design (pinned by
`trailing_bytes_are_ignored`).

Both `call_count` and the reconstruction below are available under **both** `comp.rs` implementations. The default
(non-`j2wp`) impl has no `requested`/`compiled`/`denied` fields at all — its
per-offset state lives in each `JitEntry`'s `flags`/`gen` — so it reconstructs
the same logical view behind `PhysicalCodePage::dump_*`. That reconstruction's
`requested` is approximate in one direction (an offset requested, satisfied,
then invalidated without being re-requested reads as not-requested); see the
method's own doc comment.
