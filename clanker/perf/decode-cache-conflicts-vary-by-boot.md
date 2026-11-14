# The shadow cache's 64-page decode cache conflicts differently on every boot

Status: measured 2026-11-11 on a Tezro (4 × R16000 1 GHz, IRIX 6.5.30),
interpreter (`IRIS_JIT_DISPATCH=off`), on a downstream tree whose
`decode_cache.rs` had the same `PAGES = 64` and page hash as this one. Not
re-measured here. Not fixed: `PAGES` is still 64.

## Symptom

A guest program's interpreter speed changes from one boot to the next by up
to 25% with the same build, disk and workload. A single-process `awk` over
a 17 MB file took 64–67 s in some boots and 74–110 s in others. A paired
comparison of two builds then looks like a regression in whichever build had
the unlucky boot.

Host profiles (`sample`) of the slow boots show `decode_into` and
`DecodeCache::fetch_slow` at 3–8% of busy samples. Fast boots show under 1%.

## Cause

`src/cpu/decode_cache.rs` keeps decoded instructions for `PAGES = 64`
physical pages, direct-mapped on a multiplicative hash of the page number.
Which text pages collide depends on where IRIX placed the program's, libc's
and its own text in physical memory, and that changes with boot timing on an
SMP guest. A working set that maps two hot pages to one entry re-decodes
both on every alternation.

## Evidence

Three boots each, the same build with `PAGES` 64 and 256, run
concurrently in alternating slots:

| boot | 64 pages: awk on file | re-decode share | 256 pages: awk on file | re-decode share |
|---|---|---|---|---|
| d1 | 84.1 s | 3.3% | 64.3 s | 0.2% |
| d2 | 74.1 s | 6.1% | 66.5 s | 0.1% |
| d3 | 67.8 s | 1.2% | 67.2 s | 0.2% |

## Rule

- Compare interpreter builds over several boots, never one boot each, and
  check the `decode_into` share in a profile before calling a difference a
  regression.
- A larger table (256 pages) removed the variance in this workload. Each
  page costs about 1024 decoded slots per CPU, so measure the bare-metal
  suite (`iris-bench`) on the small-memory machines before changing it.
