# Snapshot v2 device state is postcard-encoded BinValue *.bin

**Keywords:** snapshot,binvalue,postcard,schema_version,binary,device,state
**Category:** snapshot

# Snapshot v2: Postcard BinValue Device State

For schema_version=2 snapshots, every per-device save_state lives in `<base>.bin` (postcard-encoded) instead of `<base>.toml`.

## Why a tagged enum

Postcard is non-self-describing — it cannot deserialize directly into `toml::Value`, whose Deserialize impl uses `deserialize_any`. To round-trip toml::Value we mirror it as `BinValue` (tagged) in `src/snapshot.rs`:

```rust
pub enum BinValue {
    String(String), Integer(i64), Float(f64), Boolean(bool),
    Array(Vec<BinValue>), Table(Vec<(String, BinValue)>),
    Datetime(String),  // ISO-8601, falls back to String on parse error
}
```

The conversion `toml::Value` <-> `BinValue` is a single tree walk; sub-millisecond for typical device tables.

## What's TOML, what's binary

- TOML (text): `snapshot.toml` (manifest), `cow.toml` (overlay dirty sectors).
- Binary (postcard): `cpu.bin`, `mc.bin`, `ioc.bin`, `scc.bin`, `pit.bin`, `ps2.bin`, `rtc.bin`, `eeprom.bin`, `scsi.bin`, `seeq.bin`, `hpc3.bin`, `rex3.bin` (when REX3 present).
- Raw (untouched): `bank0..3.bin` (RAM), `rex3_rgb/aux.bin` (framebuffers), `scsi*.overlay` (COW).

## Save/load helpers

`Snapshot::write_state(base, &Value, schema_version)` picks `<base>.bin` for v2+ and `<base>.toml` for legacy. Mirror: `read_state(base, schema_version)`. v2 read also falls back to .toml when .bin is missing — half-migrated snapshots from external tooling still load.

## Performance

cpu state on M2 (3.6 MB cpu.toml legacy):
- TOML parse: 19.7 ms avg
- Postcard decode + BinValue->Value: 5.8 ms avg
- 3.4x speedup, 24% size reduction (2.79 MB bin vs 3.65 MB toml)

End-to-end cold restore: 189 ms (v1) -> 145 ms (v2), ~23%.

## Backward compatibility

Manifest read first. Missing manifest -> schema_version=0, loads `*.toml`. Manifest version > SCHEMA_VERSION -> hard refuse. host_arch mismatch -> hard refuse (FPU bit-layout differs cross-arch).

