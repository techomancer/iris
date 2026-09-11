# EntryHi.Region on a 32-bit TLB miss comes free from sign extension

Addresses `cpucritique.md` finding TLB-4.

## The bug

`update_tlb_exception_registers` built EntryHi with:

```rust
let vpn_mask = if XTLB != 0 { EH_REGION | EH_VPN2_64 } else { EH_VPN2_32 };
```

`EH_VPN2_32` is `0x0000_0000_FFFF_E000`, so the 32-bit arm wiped bits [63:32] entirely and
left `EntryHi.Region` = `00` (User) even for a KSSEG/KSEG3 miss. A refill handler that read
EntryHi back and issued TLBWI committed the entry under the wrong region.

## The fix — no branch needed

```rust
let vpn_mask = if XTLB != 0 { EH_REGION | EH_VPN2_64 } else { EH_REGION | EH_VPN2_32 };
```

Keep `EH_REGION` in **both** arms. The critique proposed branching on bit 31 to decide
whether to OR in the region; that is unnecessary, because in 32-bit mode every address that
reaches the TLB is already **sign-extended**:

| Segment | VA | bits [63:32] | `& EH_REGION` >> 62 |
|---|---|---|---|
| KUSEG | `0x0000_0000_xxxx_xxxx` | all 0 | `00` — User |
| KSSEG | `0xFFFF_FFFF_C000_0000` | all 1 | `11` — Kernel |
| KSEG3 | `0xFFFF_FFFF_E000_0000` | all 1 | `11` — Kernel |

The sign extension *is* the region encoding, so masking it through produces the
architecturally required value for both directions with one constant and no conditional.

This convention is relied on elsewhere in the tree — e.g. `is_xtlb_address` distinguishes
32-bit-compat kseg by testing `(virt_addr >> 32) != 0xFFFFFFFF`, and `mips_dis.rs` names
CKSSEG/CKSEG3 at `0xFFFF_FFFF_C000_0000` / `_E000_0000`.

## Test

`test_tlb_miss_preserves_entryhi_region_32bit` drives real misses on an empty TLB and
asserts Region is `11` for KSEG3 and KSSEG and `00` for KUSEG, plus that the ASID survives
and BadVAddr is set.

Pinning the KUSEG `00` case matters as much as the kernel ones: the whole fix is "let the
sign extension through", so the test has to prove that does not also promote *user* misses
to kernel region. Verified to fail (`left: 0, right: 3`) with the mask reverted.
