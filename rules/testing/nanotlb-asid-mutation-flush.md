# nanotlb must be flushed on ASID mutation, not just on privilege transition

## Status: fixed (`handle_cp0_side_effects` reg 10, `exec_tlbr`), regression-tested

## The design being corrected

`NanoTlbEntry` tags a translation by **VA only** (`va_tag == (va & !0xFFF) | 1`).
Its entries are therefore valid only for the ASID that was live when they were
filled. This is deliberate — a one-compare hit test is the whole point — and the
coherency strategy is *barrier-based*: every privilege transition flushes
(`on_cp0_status_changed`, both `handle_exception` paths, `exec_eret`,
`MipsCpu::stop`), so the slots are only ever consulted within one privilege
epoch. A kernel that changes ASID does so inside EXL and returns via ERET, and
that transition is what makes the untagged tag safe.

**The gap: the barrier is transition-shaped, not ASID-shaped.** It contains any
ASID change *accompanied by* a transition. Two instructions change the current
ASID with no transition around them:

- **`MTC0`/`DMTC0 EntryHi`** — `write_cp0` reg 10 writes ASID in bits [7:0];
  `handle_cp0_side_effects` had no reg-10 case.
- **`TLBR`** — `exec_tlbr` overwrites `cp0_entryhi` *wholesale* from the indexed
  entry, ASID included. `exec_tlb`'s dispatch comment justified the absent flush
  with "they don't mutate the TLB" — true of the TLB, **false of the current
  ASID**. That comment is the trap: it reads as a completed safety argument and
  it is only half of one.

Result: an access to a TLB-mapped VA after a bare ASID change could hit a slot
filled under the previous ASID and return **the wrong address space's physical
page**.

## Why it never showed up in IRIX

Both sites are kernel-only, and kernel-mode VAs in that window are
overwhelmingly KSEG0/KSEG1 — unmapped, ASID-irrelevant. Triggering it needs a
kernel-mode access to a *mapped* VA, on a page already resident in a slot, after
an ASID change with no intervening transition. Apparently IRIX never does that.
Fixed anyway: the flush is two cold-path instructions and the reasoning required
to stay convinced it's unreachable is worth more than the flush costs.

## What was already covered (checked, don't re-derive)

- Snapshot restore, `restore_state_digest`, and the TOML load path all call
  `on_cp0_status_changed`, which flushes.
- Reset zeroes `cp0_entryhi` before any slot can be filled.
- `update_tlb_exception_registers` rewrites EntryHi on every TLB exception but
  **preserves** the ASID, so it cannot change it.

Those two instructions were the entire gap.

## Regression tests

`test_mtc0_entryhi_asid_change_invalidates_nanotlb` and
`test_tlbr_asid_change_invalidates_nanotlb` (`mips_exec_test.rs`): map VA 0x1000
to PFN 0x50 under ASID 10 and PFN 0x60 under ASID 11, prime the read slot under
ASID 10, switch ASID via the instruction under test, assert the read now sees
PFN 0x60's data.

**Verified to fail with the flushes disabled** (returning `0xAAAAAAAA`, the
stale ASID-10 page). A test for a "can't happen" hazard is worthless unless you
have watched it fail — do that before trusting either of these.

### Gotcha when writing TLB tests: reset leaves `Status.ERL=1`

`MipsCore` resets with `cp0_status = STATUS_BEV | STATUS_ERL`. Under ERL=1,
KUSEG/xuseg become **unmapped, uncached identity** and the TLB is bypassed
entirely (`translate_32bit_impl` segment 0..=3). A test that writes TLB entries
and then reads a KUSEG VA silently gets identity translation and reads zeroes —
the TLB entries are never consulted and nothing errors.

Clear it first:

```rust
exec.core.cp0_status &= !crate::mips_core::STATUS_ERL;
exec.update_translate_fn();   // translate_fn is cached per priv/mode — must re-derive
```

Forgetting `update_translate_fn()` leaves the stale ERL-era function pointer
installed, with the same silent-identity symptom.

Also: `MockMemory` seeds via `set_word`/`set_byte`, not `write32`.

## Consequence for nutlb

`docs/nutlb-design.md` proposes putting ASID *in* the tag, which closes this by
construction. Since the hazard is now fixed independently, that is a property of
the redesign but **not an argument for it** — nutlb has to justify itself on
capacity and flush frequency alone. Note the intended end state is deliberately
asymmetric: the fetch entry stays untagged and barrier-flushed (these two sites
included), while the data-side arrays go tag-based and stop flushing on
transitions. Expect someone to try to "fix" that inconsistency.
