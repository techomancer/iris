# SCC restore leaves rr0 contradicting the emptied FIFOs

**Keywords:** snapshot,restore,scc,z85c30,rr0,tx_buffer_empty,serial,console,tx_int_pending,update_ip
**Category:** snapshot

## Scope

This is a latent defect found while chasing a different failure. The reported
symptom, a guest that survives `iris-ci restore` alive but silent, is **not**
cured by anything here: that has a separate and still undiagnosed cause, see the
last section. What is recorded here is a real state-consistency bug in
`channel_from_toml`, reproducible only from a doctored snapshot.

## What `channel_from_toml` got wrong

It restores `regs`, `reg_ptr` and `status`, then clears both FIFOs. Three things
did not follow from that:

- `status` was taken from the snapshot verbatim, so it can contradict the queues
  it restores into.
- `tx_int_pending` was forced false. `Channel::get_ip` gates the TX bit on it and
  the only setter is `notify_tx_empty`, which the TX thread calls after a
  character finishes transmitting. Clearing `tx_queue` removes the character
  whose completion would have set the latch, so nothing sets it again.
- Nothing called `update_ip`, the only thing that publishes `ip_num` and calls
  `IrqCallback::set_level` to drive the IOC's `map_stat` SERIAL bit.

The two rr0 bits are not symmetric, which matters when reading the fix:

- **`TX_BUFFER_EMPTY` restored clear is a guest deadlock, not a model dead end.**
  The TX thread sets the bit as soon as anything reaches `tx_queue`, so the
  model recovers on the first write. IRIX gates its write on the bit, so the
  write never comes.
- **`RX_CHAR_AVAILABLE` restored set is genuinely unrecoverable.** `read_data`
  returns 0 without touching the bit when the queue is empty, so a guest polling
  it spins on zeroes with no setter reachable.

The re-armed latch is gated on `WR5 TX_ENABLE` as well as `WR1 TX_INT_EN`,
because the TX thread's own predicate is `!tx_queue.is_empty() && tx_enabled`.
Gating on WR1 alone would synthesize a latched interrupt on a disabled
transmitter, which the running model can never produce, and hold the IOC line
asserted until the guest issues `RES_Tx_P`.

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

`wr1` was `0x00` during PROM output and `0x11` on every organic sample taken
under IRIX, including one two seconds into a console burst. Both have
`TX_INT_EN` clear, so the console traffic actually observed was polled, and the
interrupt arm of the fix never mattered to it. `0x13`, which would make the
console interrupt-driven, was seen only after doctoring a snapshot. Do not read
the two values as "panic printer" versus "normal tty": the data does not support
that split.

## How often it bites: rarely, on four samples, with no mechanism

Four snapshots taken organically with `iris-ci save`, two at a login prompt, one
at an idle root shell, one two seconds into a console burst, all captured
channel B as `wr1 = 0x11`, `status = 0x2c`. TX interrupts disabled,
`TX_BUFFER_EMPTY` already set, `RX_CHAR_AVAILABLE` already clear: **all three
arms of the fix are no-ops there**, and the fixed and unfixed trees produced
byte-identical `scc.bin`. Reaching the state the fix targets required doctoring
a real snapshot to `wr1 = 0x13` / `status = 0x28`.

Do not explain that rarity by saving being quiescent. `save_snapshot` and
`capture_rollback_checkpoint` do both open with `Machine::stop()`, and no save
path skips it, but `stop()` **freezes** the FIFO rather than draining it: the TX
thread breaks out of its wait and exits without popping, so a queue of 4 with
`TX_BUFFER_EMPTY` clear is preserved verbatim. Stopping also kills the one
thread that would have set the bit back. The FIFO is 4 deep and the TX thread
paces at `tx_delay` per character while the CPU fills it in nanoseconds, so
during sustained output the bit is clear for a real fraction of each character
time. Four samples is the whole of the evidence for rarity.

## Why `update_ip` is still right

`load_snapshot_inner` calls `power_on_devices()`, which reaches `Channel::reset()`
and drops `map_stat` SERIAL. `Ioc::load_state` then restores `map_stat` wholesale
and can re-assert it, leaving the IOC holding the line while both channels report
`get_ip() == 0`.

That disagreement was tested directly by doctoring `ioc.bin` `map_stat` to `0x20`
with `scc.bin` untouched, and it produces **one** spurious interrupt rather than
a storm: the guest's first SCC register write calls `update_ip`, which calls
`set_level(false)` and clears SERIAL. `rd RR3` after restore was 0 on every
unfixed-tree run traced, so the storm signature never appeared.

That measurement was taken against a tree where `tx_int_pending` was false.
**It does not carry over to the fixed code**: with the latch armed, `get_ip`
returns the TX IP, so the same guest write calls `set_level(true)` and SERIAL
stays asserted until the guest issues `RES_Tx_P` or writes data. That is a
normal pending interrupt rather than a stuck line, but it is not the "clears
itself" behaviour the experiment showed.

Ordering therefore matters. `load_snapshot_inner` runs `Ioc::load_state` before
`scc().load_state`, and `apply_rollback_checkpoint` does the same, so the SCC
drives the line afterwards on both paths. Lock order is channel then IOC state,
matching `write_a_control`, which holds both channel locks across `update_ip`.

## Why the round-trip test could not catch any of it

`save_load_round_trip` asserts `save_state == load_state -> save_state`.
`channel_to_toml` serializes only `regs`, `reg_ptr` and `status`, and the v2
`.bin` path is postcard over the same `toml::Value`, so no second serialization
carries the queues. Clearing a field on load is symmetric under that comparison,
and a status bit contradicting a queue is invisible because the queues are never
compared.

**That shape cannot catch a cleared-field bug or a state-consistency bug.** Use a
functional restore test that asserts observable behaviour after load: RR3 as a
guest ISR would read it (`restore_rearms_tx_interrupt`), or delivery into a
recording `IrqCallback` (`restore_drives_irq_line_to_parent`, which is the only
one that covers the `update_ip` arm, since RR3 recomputes channel A's IP live).

Worth saying plainly, since this note is the durable record: the underlying
defect is a serialization gap, and the fix works around it rather than closing
it. Dropping the queues is defensible, because in-flight console bytes are not
worth persisting. Deriving `tx_int_pending` from WR1 and WR5 is a guess at a
field that could simply have been saved.

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
the rest. On a doctored mid-write snapshot the fix makes that panic audible
instead of silent, but on organic snapshots both trees emit the same bytes, so
it changes nothing about how the failure presents in practice.

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
