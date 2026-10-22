# MEMCFG: `limit` is the slot, `addr_mask+1` is the mirror period — they differ

## Status: confirmed by test, 2026-08-24

`MemoryController::memcfg_bank_info(half, size_mb)` returns
`(base, addr_mask, limit)`. It is tempting to assume `limit` is how much RAM the
bank contributes, or that it is a whole multiple of the bank size. **Both are
wrong**, and ppmem's `map_bank` shipped with that bug until a `remap_banks`
integration test caught it.

Actual decode for a bank at `0x08000000`:

| SIMM | `addr_mask` | `addr_mask+1` (mirror period) | `limit` (slot) |
|---|---|---|---|
| 8MB | `0x7fffff` | 8MB | **4MB** |
| 16MB | `0xffffff` | 16MB | 16MB |
| 32MB | `0x1ffffff` | 32MB | **16MB** |
| 64MB | `0x3ffffff` | 64MB | 64MB |
| 128MB | `0x7ffffff` | 128MB | **64MB** |

## What the two numbers mean

- **`addr_mask + 1` — the mirror period.** How far you travel before the same
  physical storage reappears. This is what `Memory::read32` enforces with
  `addr & addr_mask`.
- **`limit` — the configured slot.** How many bytes `remap_banks` installs in
  `device_map`. Reads past it hit `UnmappedRam` and return 0, matching a real
  SIMM's boundary.

For **dual-rank** SIMMs (8/32/128MB — the ones where `simm_rank == 1`) the slot
is *half* the physical bank, because the two ranks are placed at separate
addresses: rank 0 at `base`, rank 1 at `base + conf_size_per_rank`. So `limit`
being smaller than the bank is normal, not a misconfiguration.

## Consequences for anything that maps banks

A region can be:

- **smaller than one period** — map a prefix, do not repeat (the dual-rank case);
- **exactly one period** — one view;
- **several periods** — repeat to fill (an undersized SIMM in a bigger slot).

So a mapper must take the period as an explicit parameter and fill
`[offset, offset+size)` with `min(period, remaining)`-sized views. Inferring the
repeat count as `size / bank_size` asserts on a stock `banks = [128, 128, 0, 0]`
config, since `0x4000000 % 0x8000000 != 0`.

`src/ppmem/ppmem.rs`'s `MappedMemory::map_bank(bank, offset, size, period)`
takes `period` for exactly this reason; `physical.rs` passes
`addr_mask.wrapping_add(1)`.

## Test

`physical::ppmem_tests::remap_makes_window_agree_with_bus` drives the real
`memcfg_bank_info` decode for the default two-128MB-SIMM config and asserts the
mapped window and the bus return identical bytes. It failed loudly on the
`size % bank_size == 0` assumption — keep it.
