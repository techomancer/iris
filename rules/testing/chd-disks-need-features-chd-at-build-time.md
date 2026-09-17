# CHD disk images require `--features chd` at build time

If a `.chd` is configured for any `[scsi.N]` in the toml but the binary was
built **without** `--features chd` (e.g. a bare `cargo build --release --bin
iris`), the disk **cannot be attached**. `ChdHd::open` returns
`"CHD image support not compiled in (rebuild with --features chd)"`.

The symptom is several steps removed from the cause:

- That attach error now **aborts startup** with a fatal message
  (`machine.rs`, the `add_scsi_device` result check). It used to print only a
  skippable `Note:` and continue, so the device was silently absent and the
  *only* visible failure was much later: the PROM's `OSLoadPartition=disk(N)`
  finding nothing → `Unable to load bootfile: no such device` /
  `Autoboot failed`, and `hinv` listing every disk **except** the CHD.
- Confirmation when debugging a "missing disk": `lsof -p <iris-pid>` shows the
  `.raw`/`.iso` images open but **not** the `.chd` → it was never attached.

**Always build with the feature when any CHD is in the config:**

```
cargo build --release --bin iris --features lightning,rex-jit,tlbvmap,chd
```

(`chd = ["dep:libchdman-rs"]` in `Cargo.toml`.) The docs mandate CHD over raw
for install disks (see `rules/irix/irix-install.md`), so `chd` is effectively
required for normal use — don't drop it from the feature list when rebuilding.
