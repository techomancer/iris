# R5000 secondary cache: no working configuration currently exists

## Status: open, not investigated — all R5000 L1I/L2 configs are broken

Discovered while building `run/matrix.sh`'s R5000 cell for the cpu-tests
findings pass ([[project_cpu_tests_findings]]). Not one of dani's findings —
this is in `src/mips_cache_v2.rs`'s unit tests, found by running `cargo test`
under different R5000 feature combinations.

## What's broken

`cargo test --features r5k` (R5000 CPU, **no** secondary cache feature at
all) fails two tests:

```
mips_cache_v2::tests::l1i_fetch_stress
  L1I mismatch phys=0x0006be50 op=0: got=0x84a2655a want=0x1649a0db

mips_cache_v2::tests::l1i_l1d_coherence
  L1I coherence mismatch phys=0x0016799c op=2203: got=0x00000000 want=0x719b4a09
```

`cargo test --features r5k,r5ksc_triton` (R5000 + Triton on-die L2, the O2
config — not a machine IRIS targets anyway) additionally fails:

```
mips_cache_v2::tests::cache_op_index_inv_l1i
```

**`r5ksc` (external R4600SC-style secondary cache — what a real Indy R5000
board actually has) does not work either.** An earlier version of this note
claimed `r5k,r5ksc` passed clean; that was wrong — confirmed directly by the
user, not re-verified in detail here. Do not trust that combination without
re-testing.

**There is currently no known-working R5000 secondary-cache configuration.**
`src/lib.rs` has `compile_error!`s refusing to build both `r5ksc` and
`r5ksc_triton` so this can't be silently shipped broken. Plain `r5k` (R5000
CPU/FPU semantics, no secondary cache) still builds — its own L1I model has
the two failures above, but nothing depends on a working secondary cache for
CPU/FPU correctness work, which is what `cpu-tests` and `run/matrix.sh`'s
R5000 cell actually need. `matrix.sh` builds the R5000 cell with plain `r5k`.

## The actual bug

For `r5ksc`: not a subtle bug, it's unimplemented functionality — per the
user directly, the external R4600SC-style secondary cache is controlled via
memory-mapped/special-transaction L2 operations ("cursed special memory
transactions") that this codebase doesn't implement. Deliberately so: those
transactions would need to be checked on every memory access, which is a
real hot-path cost this emulator isn't willing to pay just to model a cache
tier that doesn't change guest-visible behavior for anything IRIX does.
`r5ksc` can't work until that's built, not just debugged.

For plain `r5k` (no secondary cache at all): a separate, actually-a-bug
failure in the R5000 2-way-associative L1I model itself, independent of L2.
Not investigated — `mips_cache_v2.rs:3047` (`l1i_fetch_stress`) and the
coherence assertion near it are the two failing sites to start from.

## Practical impact

No R5000 secondary-cache build exists, and won't until the R4600SC memory-
mapped L2 control transactions are implemented — this is missing
functionality, not a bug to hunt. CPU/FPU correctness work (the cpu-tests
findings pass) doesn't depend on the secondary cache working, so this
doesn't block that. It does block any work that needs an accurate Indy
R5000 L2 model.
