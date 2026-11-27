# One save, load and rollback for every machine; machines supply only their state

Status: current (2026-11-27). Read `src/snapshot.rs` and the `Workstation` snapshot hooks before relying on details. Replaces the note that described three separate rollbacks.

**Why.** The Indy and Indigo2, the Octane and the O2 each hand-wrote save, load,
CI restore and CI rollback: the same manifest, load checks, chunk store,
overlays, stop and restart and parent name, three ways. The copies drifted into
bugs: a wrong-machine load wiped a running Indy before checking, a failed Indy
save left the guest frozen, the O2 never recorded a parent. Only the device list
and the order it loads in differ, so the steps live once, in `src/snapshot.rs`.

**What to know.**

- The shared steps are plain functions, generic over `Workstation` so the
  trait's default methods call them: `save_in`, `read`, `restore`, `load_in`,
  `ci_restore`, `ci_rollback`. A machine supplies six defaulted hooks:
  `snapshot_refusal`, `snapshot_layout` (what a snapshot must hold to fit this
  machine), `capture` (an `Image` of the stopped machine), `apply_image` (power
  on, then load), `with_disk_buses` and `snapshots` (its `Restored` slot). The
  default refusal is "not supported on this machine", and the SN machines only
  override the message.
- `Image` is the state in load order: named device tables, raw files
  (`flash.bin`, the Indy's second head's planes), memory as `(bank_chunks index,
  words)`, the first head's framebuffer and graphics `Board`s. The IMPACT, VPro
  and GR2 chunk lists share one `NamedChunk` type, which postcard writes as the
  old three did.
- **Save** refuses first, notes the lifecycle and whether the CPU ran, stops,
  writes `snapshot_incomplete`, the manifest, states, files, chunks and
  overlays, removes the marker, then restarts only what was running, on error
  paths too. A load refuses a directory that still has the marker (or the older
  `resident_snapshot_incomplete`).
- **Load reads and checks before it touches the machine**: marker, machine, host
  arch, schema, provenance, slots, every state and file the layout names, each
  memory list's length, each board present on both sides or neither, and
  `cow.toml`. `restore` then stops, applies the image, imports the overlays and
  restarts. If the apply fails the machine stays stopped, wanting a reset or
  another load: restarting a board whose resident raster has faulted panics.
- **Rollback is the same everywhere.** A CI restore keeps the image it read and
  `ci_rollback` applies it again. Any load records its name, as the next save's
  `parent` and the rollback target; a plain load keeps no image, so a rollback
  after one reads the snapshot again.
- Every machine imports the COW overlays from `saves/<name>/` again on a
  rollback. That is safe because guest writes after the import go to the live
  overlay, never back into the snapshot directory.
- The Indy's live checkpoints (`jitcheck`, `j2 replay`) are an `Image` too. They
  include the SCSI controllers' registers but not the disks' contents.

**What stays per machine** (hardware, or the schema-11 file layout): the
manifest's `machine` value, the device tables and their load order (the
Octane's HEART last and `sync_links`; the O2's boot state), memory as four lists
on the Indy and one on the others, the graphics boards and their state names
(`mgras` on the Indy, `impact` on the Octane), the NVRAM and flash files, the
Octane's refusal of two CPUs. The Indy lists its device tables once, in
`each_saved_device`, for capture, apply and the layout.

**Risks.**

- Schema 11 must still load, byte for byte: file names, state names,
  chunk-list placement, manifest keys and the overlay layout do not change.
  `tests/snapshot_schemas.rs` loads the three schema-11 fixtures through the
  binary and refuses schema 10.
- A device whose `load_state` then `save_state` is not the identity would
  behave differently on a second rollback, now that the Indy rolls back from the
  image it read and not from a re-capture. The head 1 lifecycle test, the GR2
  checkpoint tests and the rollback tests on each machine cover it.
- The Octane imported overlays before loading its devices ("an import can take
  minutes"); the shared restore imports after. Its clock is stopped during the
  import, and the Octane snapshot tests pass.
- The Octane now holds a RAM-sized image while a CI restore is active, as the
  Indy and O2 already do. The `iris-ci` help's restore and rollback times are
  not remeasured.

**Other notes that go with it.** A CD change (`cdrom-load`/`cdrom-eject`) needs
nothing per controller: the `DiscChanger` blanket impl over `ScsiTargets` reaches
the shared `ScsiDevice`, which arms unit attention 06/28, and the AIC-7880 passes
it to the HIM's auto-sense like any other check condition
(`a_disc_change_reaches_the_him_as_unit_attention`). Snapshots are written under
the working directory's `saves/`; the shared functions take that directory, so
tests use a temporary one instead of running themselves again as a child
process. The existing child-process tests keep working.
