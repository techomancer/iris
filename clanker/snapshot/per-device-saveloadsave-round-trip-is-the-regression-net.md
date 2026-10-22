# Per-device save→load→save round-trip is the regression net

**Keywords:** snapshot,round-trip,save_state,load_state,regression,test,convention
**Category:** snapshot

# Round-Trip Test Convention

Every device with a `Saveable` impl gets a `save_load_round_trip` test in its `#[cfg(test)] mod tests`. Catches save/load asymmetries that would otherwise corrupt snapshots silently.

## Pattern

```rust
#[test]
fn save_load_round_trip() {
    let src = Device::new(...);
    // 1. Mutate to non-default state.
    {
        let mut s = src.state.lock();
        // ... touch fields that save_state serializes
    }
    let v1 = src.save_state();

    let dst = Device::new(...);
    dst.load_state(&v1).expect("load_state");
    let v2 = dst.save_state();

    assert_eq!(v1, v2, "Device save_state mismatch after load_state round-trip");
}
```

## Conventions

- **Mutate first.** Saving an all-default state proves nothing — a load that no-ops on every field will pass.
- **Use null/CI constructors when devices bind ports.** Z85c30::new_null avoids TCP 8880/8881; Ioc::new_ci uses null backends.
- **If load_state has a side-effect that derives state from other fields, call it on src before saving.** Example: IOC update_interrupts re-derives MAP_INT0/MAP_INT1 cascade bits in l0_stat/l1_stat from (map_stat & map_mask{0,1}). Save the post-derive state so v1 already includes the cascade — otherwise v2 differs by the cascade bits.
- **Disable wall-clock-driven side effects.** RTC: clear TE_BIT before saving so save_state doesn't tick the host clock between v1 and v2.

## What's covered

eeprom_93c56, ds1x86, ioc, pit8254, mc, mips_tlb, ps2, z85c30, wd33c93a, seeq8003,
testdev.

## What's not (yet) covered

- hpc3 — composite of nested devices. Round-trip indirectly tested via end-to-end snapshot/restore.
- rex3 — 16 MB framebuffers + massive VC2/CMAP/XMAP state.
- mips_exec — needs Tlb+Cache type params + Bus integration.

These are exercised by the end-to-end snapshot/restore validation in the CI socket workflow.

