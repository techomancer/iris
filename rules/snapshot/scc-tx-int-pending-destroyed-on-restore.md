# SCC restore drops the TX latch and never redrives the IRQ line

**Keywords:** snapshot,restore,scc,z85c30,tx_int_pending,update_ip,rr0,serial,console
**Category:** snapshot

## What `channel_from_toml` got wrong

`channel_from_toml` in `src/z85c30.rs` restores `regs`, `reg_ptr` and `status`,
then clears both FIFOs. Three things did not follow from that:

- `tx_int_pending` was forced false. `Channel::get_ip` gates the TX bit on it
  and the only setter is `notify_tx_empty`, called by the TX thread after it
  drains a queued char. Clearing `tx_queue` removes the char whose completion
  would have set the latch, so nothing sets it again.
- `status` was taken from the snapshot verbatim. A save with the TX FIFO full
  restored `TX_BUFFER_EMPTY` clear over an empty `tx_queue`; a save with a byte
  in the RX FIFO restored `RX_CHAR_AVAILABLE` set over an empty `rx_queue`.
  The RX case does not recover on its own: `read_data` only drops the bit when
  it pops a byte, and there is nothing to pop.
- Nothing called `update_ip`. That is the only thing that publishes `ip_num`
  (what `read_a_control` reports as RR3 for the other channel) and calls
  `IrqCallback::set_level` to drive the IOC's `map_stat` SERIAL bit. Arming the
  latch without it changes a field nobody reads.

Ordering matters for the `update_ip` call: `Machine::load_snapshot` runs
`Ioc::load_state` before `scc().load_state`, and `Ioc::load_state` restores
`map_stat` wholesale, so the SCC has to drive the line afterwards. Lock order
is channel then IOC state, same as the RX thread at `src/z85c30.rs:982`.

## Measured on an IRIX 6.5 Indy guest

Boot to a root shell, `iris-ci save`, `iris-ci restore`, then sit for 45 s
sending nothing. Fresh `.diff.chd` overlay for each run, one run per tree.

- Unfixed tree: no bytes at all. Injecting a newline and waiting 15 s produced
  no bytes either.
- Fixed tree: the console immediately carried `PANIC: KERNEL FAULT / EXC
  code:128, 'Software detected SEGV'` and the whole crash-dump progress.

So the fix does restore console output across a snapshot load. It does not make
restore usable, because of the next section.

## Restore corrupts the guest, and that is a different bug

Every restore observed on this image left IRIX damaged, on both trees.
`iris-ci run` times out afterwards because the shell it is talking to is gone,
not because the SCC is wedged. Symptoms seen across runs:

- `PANIC: KERNEL FAULT ... Software detected SEGV`
- `ALERT: Process [sysevent] ... process or stack limit exceeded`
- `ALERT: XFS internal error XFS_WANT_CORRUPTED_GOTO at ... xfs_alloc.c`
- `NOTICE - cpu 0 has duplicate tlb entries (13, 24)`

Two further facts worth knowing before debugging that:

- Restoring into a process that was never `start`ed gives total silence and a
  CPU thread spinning at 100 %. `iris-ci start` first, even only as far as the
  PROM `Option?` menu, and the same snapshot comes back to a working shell.
  Identical on both trees.
- The guest state is sticky across runs. With CHD images the COW overlay is
  `<base>.chd.diff.chd` beside the image, not the `/tmp/iris-ci-<pid>-scsi<id>`
  path the raw-image rule describes, and it is not per-pid. A guest that
  panicked in one run boots into `savecore` in the next and behaves differently.
  Delete the `.diff.chd` between A/B runs or the comparison is worthless.

## Why the round-trip test could not catch any of it

`save_load_round_trip` asserts `save_state == load_state -> save_state`.
`tx_int_pending`, `rx_queue` and `tx_queue` are not in `save_state` at all, so
destroying them on load is invisible to it. Nor does the shape reach anything
outside the device: `ip_num` and the IRQ callback are not serialized either.

A round-trip test proves serialization is self-consistent. Proving the restored
device still works needs a functional test: load into a device built with a
recording `IrqCallback` and assert on delivery, not on field equality. See
`restore_drives_irq_line_to_parent` in `src/z85c30.rs`.
