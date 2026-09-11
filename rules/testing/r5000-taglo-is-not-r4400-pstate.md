# R5000 L1 TagLo is D/V bits, not R4400 MESI PState

Addresses `cpucritique.md` CACHE-1, and cleans up CACHE-2/CACHE-3/CACHE-4 alongside it.

## CACHE-1 — the real bug

`C_ILT` / `C_IST` in `mips_cache_v2.rs` applied the R4400 tag layout to every model:

| | R4400 | R5000 |
|---|---|---|
| L1-D `[7:6]` | 2-bit MESI PState: 0=Inv, 1=Shared, 2=CleanExcl, 3=DirtyExcl | bit 7 = **D**, bit 6 = **V** |
| L1-I `[7:6]` | same PState field (2 = valid) | bit 7 = **L** (lock), bit 6 = **V** |

The R5000 L1 has no bus snooping and therefore no Shared state at all, so the two encodings
are not reconcilable. Consequences before the fix:

- A valid **clean** R5000 line was emitted as PState=2 → `D=1, V=0` → reads back as
  **invalid** to R5000 software.
- Firmware writing `TagLo = 0x40` ("valid, clean") was decoded as PState=1 → **Shared**, a
  state R5000 does not have.

Fixed by branching on `Self::IS_R5K` in both directions. `IS_R5K` is `const IC_WAYS == 2`,
so this is compile-time — no runtime branch on the cache path.

We model no lock bit, so R5000 L1-I emits L=0 always.

## CACHE-3 — the comment was simply false

The `write` path carried:

> *"tcache is **not implemented here** — the 2-way paths always take the real-cache route,
> so no transparent lines are ever created on R5000 and the feature is silently a no-op for
> that model."*

Contradicted five lines below it, where `tc_write` runs unconditionally under `cfg(tcache)`.
`fill_l1d_line` likewise sets `transparent = true` purely on `cfg(tcache)` with no `IS_R5K`
guard. Comment rewritten to describe what the code does.

## CACHE-4 — not a bug

The critique claims R5000 `tcache` leaves stale instructions in `ic_instrs` because
`tc_invalidate_l2_code` early-returns when `!HAS_L2`.

That early return is correct. `has_code` is **tcache bookkeeping for the decoded-instruction
slots in L2** (`l2.instrs`), not architectural I-cache coherency — see the `l2_hit` guard in
the L1-I fill path, which refuses an L2 line that was filled for data. With no L2 there are
no such slots to invalidate.

R5000's own decode slots (`ic_instrs`) are populated **only** on an L1-I fill, which happens
only after an L1-I miss. A guest that modifies code must issue a `CACHE` op to retire the
L1-I line — that is the architectural contract on every MIPS, and hardware does not snoop
L1-I against data stores either. Retiring the line retires its decode slots with it.

Adding write-side L1-I invalidation would impose a cost on the *primary* R4400 target to
"fix" behaviour that matches the architecture. Not done.

## CACHE-2 — nothing to store

TagHi carries ECC/parity for the primary caches on both R4400 and R5000, and this emulator
models no ECC. The physical tag fits entirely in TagLo: `L1_PTAG_MASK` is 24 bits at
`L1_PTAG_SHIFT = 12`, i.e. PA[35:12] — the full 36-bit physical address. Nothing spills into
TagHi, so zeroing it on `C_ILT` is right. Left as-is.

## Tests, and what each actually proves

- `taglo_valid_clean_line_reads_back_valid_on_both_models` — fills a clean line and checks
  R4400 reports PState=2 while R5000 reports V=1/D=0. **Catches the load side.**
- `taglo_r5000_decodes_hardware_bit_patterns` — feeds R5000 the literal patterns firmware
  writes (`0x80`, `0x40`, `0xC0`) and checks how they land. **Catches the store side.**
- `taglo_store_load_round_trips_on_both_models` — symmetry only.

That third one is worth a warning: **it passes even with both halves wrong**, because an
encode/decode pair that shares a mistake is still self-inverse. It was written first and
gave a false green; the other two exist because of it. Keep it (an *asymmetric* pair is its
own bug class) but never treat a round-trip test as evidence that a wire format is correct.

Both fixes were verified by reverting each half independently and confirming the
corresponding test fails.
