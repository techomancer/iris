# The XTLB refill vector follows the MODE, not the address shape

*cpucritique.md T1. Fixed 2026-09-12.*

## The rule

On R4400/R5000 the TLB-refill vector is chosen by the **X bit for the privilege
mode the miss was taken in** — `Status.KX` in kernel mode, `SX` in supervisor,
`UX` in user:

- X bit clear → 32-bit TLB refill, offset **0x000**
- X bit set   → XTLB refill, offset **0x080**

It is **not** chosen from the shape of the faulting virtual address. A
sign-extended 32-bit-looking address such as `0xFFFF_FFFF_C000_0000` (kseg3
compat) taken by a kernel running KX=1 vectors to **XTLB**, because the mode is
64-bit. Its "32-bit shape" is irrelevant.

## What was wrong

`translate_64bit_impl`'s kseg3/ksseg compat arm passed `XTLB=0`:

```rust
if (virt_addr >> 32) == 0xFFFFFFFF {
    match (addr_32 >> 29) & 0x7 {
        4 => /* kseg0 */, 5 => /* kseg1 */,
        _ => return self.tlb_translate_impl::<DEBUG, 0>(...),  // WRONG
    }
}
```

That arm is only reachable when `is_64bit_mode()` is already true, so `XTLB=0`
there is unconditionally wrong. Fixed to `1`.

`is_xtlb_address` had the mirror-image bug and collapsed to:

```rust
self.core.is_64bit_mode() && (virt_addr >> 62) != 2   // != xkphys
```

## Two distinct effects, one const param

`XTLB` has exactly two live uses in `tlb_translate_impl`, and the bug broke
both. This is why it is worth more than the vector number suggests:

1. **Vector choice** — `exec_xtlb_miss` vs `exec_tlb_miss`. A KX=1 kernel's
   cksseg miss entered the 32-bit refill handler, which reads `Context`
   (32-bit BadVPN2) rather than `XContext`.

2. **Compare width** — `tlb.translate::<XTLB>` selects `vcmp64`
   (`0xC000_00FF_FFFF_E000`, **includes the Region field**) or `vcmp32`
   (`0x0000_0000_FFFF_E000`, **drops it**). With `vcmp32`, a cksseg VA
   (Region `11`) matched an xuseg entry (Region `00`) whose VPN2[31:13]
   happened to collide — **a spurious hit returning another page's data**, not
   merely a wrong vector.

Effect 2 is the severe one. The test
`test_xtlb_compare_width_disambiguates_region_field` demonstrates it directly:
with the fix reverted it reads `0x12345678` from a page that VA does not map.

A third use, `update_tlb_exception_registers::<XTLB>`, is **dead** — the mask
was unified to `EH_VPN_MASK` in an earlier round and the function no longer
branches on the param. Left in place only so the call sites read consistently.

## Why the compare-width half is safe

Widening the compare works *only* because the EntryHi unification already
landed. `update_tlb_exception_registers` commits
`virt_addr & 0xC000_00FF_FFFF_E000`, so a refill handler's `TLBWI` for a cksseg
miss writes **R=11 and bits[39:32]=0xFF** — exactly what `vcmp64` will later
compare a cksseg VA against. Had EntryHi still been masked with the narrow
32-bit mask, widening the compare would have turned every cksseg hit into a
permanent miss. **If you ever revisit the EntryHi mask, these two must move
together.**

## Dormancy — read this before deleting the tests

IRIX 6.5 on IP22 runs with **KX=0** (`trap.c`). Nothing IRIS boots exercises
any of this. There is no boot, benchmark, or cpu-test that would have caught
the bug and none that will catch a regression.

The three unit tests in `mips_exec_test.rs` are therefore **the entire
coverage**, not a supplement to it:

- `test_xtlb_vector_chosen_by_kx_not_address_shape` — sweeps KX on/off over the
  same compat address; also pins that true xkseg was already correct, narrowing
  the change to the compat arm.
- `test_xtlb_compare_width_disambiguates_region_field` — the Region-alias
  spurious hit.
- `test_tlbp_region_field_symmetry_under_kx` — consistency between `exec_tlbp`
  and a real access.

Both of the first two were verified to **fail** with the fix reverted.

## A note on the TLBP path

`exec_tlbp` was never broken, despite calling the same `is_xtlb_address`. It
masks EntryHi with `EH_VPN_MASK` first, so `virt_addr >> 32` is `0xC00000FF` —
never the `0xFFFFFFFF` the old address-shape test looked for. The old code's
compat arm was simply unreachable from a masked EntryHi, so TLBP always used
the 64-bit compare. Worth knowing: the third test is a consistency assertion,
not a regression test, and it is labelled as such in the source.

## Related

- `rules/testing/cpu-tests-known-failure-baseline.md`
- EntryHi unification: cpucritique TLB-4, `update_tlb_exception_registers`
