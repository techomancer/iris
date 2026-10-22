# VDMA image transfers are always address-translated

**Do not gate a VDMA fast path on `!xlate`.** It would be dead code.

`MCdma()` in `irix/kern/io/vdma.c` — the only entry point IRIX uses for REX3
image up/download — sets `VDMA_C_XLATE` unconditionally. Both arms of its only
branch OR it in; just the interrupt-enable bit differs:

```c
if ((ena_int) && (ena_int != REX_BUG))
        VDMAREG (DMA_CTL) = VDMAREG(DMA_CTL) | VDMA_C_XLATE | VDMA_C_IE;
else
        VDMAREG (DMA_CTL) = (VDMAREG(DMA_CTL)|VDMA_C_XLATE )&~VDMA_C_IE;
```

`vdma_set_tlb()` exists precisely to populate the 4-entry µTLB before each
transfer — the whole apparatus only makes sense because translation is on.

## Who runs untranslated

Only two callers clear the bit:

| caller | why |
|---|---|
| `MCdma_desc()` (`vdma.c`, the `~VDMA_C_XLATE` write) | descriptor lists already carry physical addresses |
| PROM memory fill | runs before the MMU is up |

So untranslated VDMA is the *exception*: PROM-era memory clearing and the
descriptor path. Everything X11 does — every image upload, every readback — is
translated.

## Why this is easy to get wrong

The pre-split `dma_worker()` in `mc.rs` handled both with an inline
`if xlate { translate_addr(..) } else { mem_vaddr }` at each of its three access
sites, so nothing in the emulator source *says* which case dominates. Reading
only the emulator, "untranslated qword transfer" looks like the obvious fast
path to build. It never executes.

The answer is in `ignore/irix/irix/kern/io/vdma.c`, which is in the tree. Check
it before assuming what the guest does.

## Cost note

The generic byte engine calls `translate_addr()` **per byte** — a µTLB walk, a
`giodma.state` lock acquisition and a PTE bus read for each of the 8 bytes in a
qword. That is how it has always worked and it is correct, just wasteful.
`mc_vdma.rs`'s flat 64-bit paths translate once per qword (safe: `qword_flat()`
requires an 8-byte-aligned vaddr, and pages are 4K/16K, so no qword straddles a
page). Hoisting translation to once per *page* is the next optimisation and is
not done yet.

## Test

`mc_vdma::tests::translated_qword_path_reads_through_the_page_table` builds a
real µTLB entry + page table and asserts the fast path resolves through it.
Verified non-vacuous: stubbing `dma_xlate_qword()` to the identity fails it.
