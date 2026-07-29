# SCC restore leaves rr0 contradicting the emptied FIFOs

**Keywords:** snapshot,restore,scc,z85c30,rr0,tx_buffer_empty,tx_int_pending,update_ip,serial,console
**Category:** snapshot

## Symptom

After `iris-ci restore` the guest is alive but the serial console emits nothing.
`serial-wait` times out on every pattern while the emulator answers `ping`.

## What `channel_from_toml` got wrong

It restores `regs`, `reg_ptr` and `status`, then clears both FIFOs. Three things
did not follow from that:

- `status` was taken from the snapshot verbatim, so it can contradict the queues
  it restores into. Neither bit heals itself: `write_data` clears
  `TX_BUFFER_EMPTY` only when the FIFO reaches 4 entries, and `read_data` clears
  `RX_CHAR_AVAILABLE` only when a pop empties the queue.
- `tx_int_pending` was forced false. `Channel::get_ip` gates the TX bit on it and
  the only setter is `notify_tx_empty`, which the TX thread calls after a
  character finishes transmitting. Clearing `tx_queue` removes the character
  whose completion would have set the latch, so nothing sets it again.
- Nothing called `update_ip`, the only thing that publishes `ip_num` and calls
  `IrqCallback::set_level` to drive the IOC's `map_stat` SERIAL bit.

## rr0 is what unblocks the console, not the interrupt

Measured by decomposing the change and restoring one doctored snapshot on each
variant build, fresh overlay per run:

| tree | rr0 at restore | IRQ raised | guest's first act | console |
|---|---|---|---|---|
| unfixed | `0x28` | no | `rd RR0 -> 28`, stop | silent |
| unfixed | `0x2c` | no | `write_data` | alive |
| latch and `update_ip` only | `0x28` | yes | `rd L0_STAT`, `rd RR0 -> 28`, stop | silent |
| full fix | `0x2c` | yes | `rd L0_STAT`, `write_data` | alive |

A build carrying only the latch re-arm and the `update_ip` call raises the
interrupt and the guest ISR takes it, reads RR0, sees the stale busy bit and
gives up. rr0 is necessary and sufficient. The interrupt is neither.

IRIX drives the console two ways, which is why. The PROM console and the kernel
panic printer are **polled**, `wr1` being `0x00` and `0x11` respectively. Only
the normal tty path enables TX interrupts, with `wr1 = 0x13`.

## The mid-write window is rare in practice

`save_snapshot` and `capture_rollback_checkpoint` both open with
`Machine::stop()`, which joins the TX thread and lets the driver's ISR settle.
Four snapshots taken organically with `iris-ci save`, two at a login prompt, one
at an idle root shell, one two seconds into a console burst, all captured
channel B as `wr1 = 0x11`, `status = 0x2c`. TX interrupts disabled,
`TX_BUFFER_EMPTY` already set, `RX_CHAR_AVAILABLE` already clear: **all three
arms of the fix are no-ops there**, and the fixed and unfixed trees produced
byte-identical `scc.bin`. Reaching the state the fix targets required doctoring
a real snapshot to `wr1 = 0x13` / `status = 0x28`.

## Why `update_ip` is still right

`load_snapshot_inner` calls `power_on_devices()`, which reaches `Channel::reset()`
and drops `map_stat` SERIAL. `Ioc::load_state` then restores `map_stat` wholesale
and can re-assert it, leaving the IOC holding the line while both channels report
`get_ip() == 0`.

That disagreement was tested directly by doctoring `ioc.bin` `map_stat` to `0x20`
with `scc.bin` untouched. It yields **one** spurious interrupt, not a storm: the
guest's first SCC register write calls `update_ip`, which calls `set_level(false)`
and clears SERIAL by itself. `rd RR3` after restore was 0 on every unfixed-tree
run traced.

Ordering therefore matters. `Machine::load_snapshot` runs `Ioc::load_state`
before `scc().load_state`, so the SCC drives the line afterwards. Lock order is
channel then IOC state, matching `write_a_control`, which holds both channel
locks across `update_ip`.

## Why the round-trip test could not catch any of it

`save_load_round_trip` asserts `save_state == load_state -> save_state`.
`channel_to_toml` serializes only `regs`, `reg_ptr` and `status`, and the v2
`.bin` path is postcard over the same `toml::Value`, so no second serialization
carries the queues. Clearing a field on load is symmetric under that comparison,
and a status bit contradicting a queue is invisible because the queues are never
compared.

**That shape cannot catch a cleared-field bug or a state-consistency bug.** Use a
functional restore test that asserts observable behaviour after load: RR3 as a
guest ISR would read it, or delivery into a recording `IrqCallback`. See
`restore_rearms_tx_interrupt` and `restore_drives_irq_line_to_parent`.

## Restore is still unusable on IRIX 6.5

Fixing this does not make snapshots work. Every restore traced left the guest
kernel dead on both trees:

- `PANIC: KERNEL FAULT`, variously `Software detected SEGV`, `Read Address Error`
  and `Read TLB Miss`
- `ALERT: XFS internal error XFS_WANT_CORRUPTED_GOTO`
- `NOTICE - cpu 0 has duplicate tlb entries`

The panic PC is stable for a given guest state and varies between states, so it
tracks what was running rather than being random. Quiescence does not help: an
idle root shell snapshotted after `sync; sync` and 10 s of quiet panicked like
the rest. The fix makes the panic audible rather than silent.

One contributing cause is recorded separately in
`chd-snapshots-do-not-capture-the-disk.md`. A raw-image run panicked too, so
that is not the whole story.

## Two things that will waste your time

- Restoring into a process that was never `start`ed gives silence and a pinned
  CPU thread, with **zero** SCC and IOC register accesses in the window. It is
  not an ISR storm, it is spinning somewhere else. Note also that `idle-pause` is
  off by default, so `MIPS-CPU` sits at 100% at a healthy idle shell too: use
  register-access counts, not CPU percentage, to tell a wedged guest from a live
  one.
- `iris-ci` flakes under load with `connect ...: Resource temporarily unavailable
  (os error 11)`, which aborts `boot` early and looks like a boot failure. Retry.
