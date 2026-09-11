# CP0 `Index` selects a TLB slot by masking, never modulo

Addresses `cpucritique.md` TLB-2 / `cpucritique2.md` TLB-A.

## The bug

`exec_tlbr` and `exec_tlbwi` both did:

```rust
let index = (self.core.cp0_index as usize) % self.tlb.num_entries();
```

`%` maps out-of-range values back onto **valid** slots, so an illegal Index silently
corrupts a live entry instead of doing nothing:

| Index | `% 48` | Effect |
|---|---|---|
| `0x8000_0000` (failed TLBP) | **32** | overwrites live entry 32 |
| 49 | **1** | overwrites wired entry 1 |
| 63 | **15** | overwrites wired entry 15 |

`exec_tlbp` stores the probe result — bit 31 included — straight into `cp0_index`, so the
first row is reachable by any guest that issues TLBWI after a missed probe.

## The fix

Mask the slot field, then bounds-check:

```rust
let index = (self.core.cp0_index & CP0_INDEX_SLOT_MASK) as usize;  // 0x3F
if index >= self.tlb.num_entries() { return ...; }
```

`exec_tlbwr` is left alone — it indexes with `cp0_random`, which is generated in range.

**R4400 leaves TLBWI/TLBR with Index >= num_entries architecturally undefined.** Skipping is
our choice, not a fidelity claim.

## Cross-checked against MAME

Both MAME MIPS cores do exactly this, independently:

```cpp
// r4000.cpp
cp0_tlbwi(m_cp0[CP0_Index] & 0x3f);
void r4000_base_device::cp0_tlbr() {
    u8 const index = m_cp0[CP0_Index] & 0x3f;
    if (index < std::size(m_tlb)) { ... }      // no else
}

// mips3com.cpp
tlb_write_common(m_core->cpr[0][COP0_Index] & 0x3f);
uint32_t tlbindex = m_core->cpr[0][COP0_Index] & 0x3f;
if (tlbindex < m_tlbentries) { ... }           // no else
```

Mask + bounds check + silent skip, in both.

## Where we deliberately differ from MAME: the P bit on MTC0

| | `write_cp0(Index)` |
|---|---|
| MAME r4000.cpp | `data & 0x3f` — **drops** the probe-failure bit |
| MAME mips3.cpp | no mask at all; masks only at use |
| IRIS | `& (CP0_INDEX_P \| CP0_INDEX_SLOT_MASK)` — keeps P, drops reserved [30:6] |

We keep bit 31 because software reads Index back with MFC0 to test whether a TLBP missed,
and a context switch that saves and restores Index must round-trip that bit. Bounding the
*slot* field is what prevents the out-of-range write; discarding P is not needed for that
and would lose architectural state. r4000.cpp gets away with it because hardware writes the
register directly rather than through MTC0.

## What the fix does NOT do

The failed-probe case now writes **entry 0** (`0x8000_0000 & 0x3f`) instead of entry 32.
Entry 0 is typically wired, so a guest doing TLBWI after a missed TLBP still corrupts
something — it just corrupts what real hardware would. This is conformance, not a safety
net; do not read the fix as making that sequence harmless.

## Tests

`test_tlb_index_masks_rather_than_wraps` covers four things:
- TLBWI with Index = `0x8000_0000` leaves entry 32 untouched (fails before the fix);
- MTC0 Index=49 then TLBWI is a no-op, not a write to wired entry 1;
- reserved bits [30:6] read back as zero after MTC0;
- the P bit survives MTC0, so save/restore round-trips.

Two pre-existing tests (`test_mfc0_mtc0`, `test_cop0_64bit`) asserted the *unmasked*
behaviour — `MTC0 0x12345678` reading back verbatim. They were pinning "Index is a dumb
u32", which is the thing being fixed, and were updated to the narrow-register semantics
(`0x12345678 -> 0x38`, `0x90ABCDEF -> P|0x2F`).
