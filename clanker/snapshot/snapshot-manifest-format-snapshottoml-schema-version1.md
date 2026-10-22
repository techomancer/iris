# Snapshot manifest format (snapshot.toml schema_version=1)

**Keywords:** snapshot,manifest,schema,version,parent,host_arch,iris_git_rev,saves
**Category:** snapshot

# Snapshot Manifest (snapshot.toml)

Every snapshot saved by Phase 1+ writes a `snapshot.toml` at the top of `saves/<name>/`. It is read FIRST on load so format mismatches fail fast with a clear error, before any device state is touched.

## Schema
```toml
schema_version = 1            # u32, current = 1
host_arch = "aarch64"         # std::env::consts::ARCH at save time
created_at_unix = 1777764190  # u64 unix seconds
installed_bundles = []        # Vec<String>; names of software bundles installed in this image
# optional:
iris_git_rev = "abc123"       # from option_env!("IRIS_GIT_REV") at build time
parent = "base/desktop"       # name of the snapshot we restored from before this save
description = "post-install"  # free-form note
```

## Load behavior (`src/machine.rs:633` `load_snapshot`)
- **No manifest** → treated as legacy v0 with a warning. Best-effort load. (Old `saves/working*` snapshots are v0.)
- **`schema_version > 1`** → refuse: "snapshot schema_version N is newer than this iris build supports (1)".
- **`host_arch` mismatch** → refuse. FPU bit-layout differs cross-arch and there's no migration plumbing yet.
- **`iris_git_rev` mismatch with current build** → warn but proceed. Snapshots are not pinned to commits.

## Where it's defined
- `src/snapshot.rs` — `Manifest` struct, `to_toml`/`from_toml`, `Snapshot::write_manifest`/`read_manifest`, `SCHEMA_VERSION` const.
- `src/machine.rs:594` writes the manifest first thing in `save_snapshot`. `parent` is auto-set to `self.last_restore`.
- `src/machine.rs:643-672` validates on load.

## CI inspection
- `info <name>` socket command returns the manifest plus `bytes_on_disk` for any snapshot. Legacy snapshots return `{"schema_version":0,"legacy":true}`.

## Future bumps
When a device's `save_state` format changes incompatibly, increment `SCHEMA_VERSION` and add migration logic keyed off the old version number. Don't silently break v1 readers.
