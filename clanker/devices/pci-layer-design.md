# The PCI layer: invariants, hazards, and what v1 got wrong (design)

Status: in place for MACE and the Bridge, 2026-11. The user-facing description
is `docs/pci.md`.

## Invariants

- **Headers change only through the bus** (`config_write`, `reset`, snapshot
  load). Each updates the decode table and the BAR mappings. A function that
  resets its own header leaves a stale BAR mapping on the physical bus.
- **Byte orders are per-window values**, `Copy`, chosen by the host and held by
  value in each `BarDevice`. The bus never sees byte order.
- **`pci/` imports no card, no SCSI and no host.** The one thing it takes from
  outside is the physical bus's page size for the run logic.
- **Snapshot restore is not RST#.** It loads over what is there and ends
  with `PciBus::redrive`. A function's load must overwrite what the
  snapshot does not name (the AIC-7880 puts its SCSI targets back with
  `reset_targets`).
- **The combiner's handshake is one atomic word per output** (`DIRTY`,
  `HELD`, `FELL`, `FORCE`). Every step is a read-modify-write on that word, so
  acquire-release is enough; a store-then-load handshake would need SeqCst and
  cannot be model-checked. `loom_tests` in `src/pci/intx.rs` checks it
  (`cargo loom`).

## Hazards

- A stale run table: anything that changes a BAR, a command register, a bridge
  window or a host window must update the mappings. A change behind a bridge
  updates the buses in front of it. A configuration write to any other register
  does not, because each update walks the whole table.
- The physical bus keeps every device it is given, for the life of the
  machine. Hosts make their `BarDevice`s through `BarDevices`, which returns the
  same device for the same function, BAR, base and order.
- Delivery ordering: the host's own lock must not be taken by anything that
  calls into the combiner with that lock held. The Bridge decides a packet
  under `st` and sends it after dropping `st`.
- Lock order: publish mutex, then the host's lock (`refresh`'s `make`). The
  Bridge drops `st` before `refresh`. A bridge's secondary holds its parent
  weakly; the parent holds the bridge function, which holds the bus.
- The slow BAR table is scanned without a lock while an update rewrites it. A
  reader can briefly see a mix of old and new regions; every word is valid and
  no slot is freed, so it is not unsafe.

## What v1 (`pci-addin`) got wrong

- Cards, SCSI and the snapshot format leaked into the core (`CardRef`,
  `Origin`, `attach_card`, `ScsiController`).
- Three copies of the level state in the Bridge and a lock held across the
  host call in MACE's router.
- A byte-order trait object per host and a duplicate swizzle in each.
- Reset was whatever each function's power-on did, and the O2 had none.

## Open

See [pci-layer-open.md](pci-layer-open.md).
