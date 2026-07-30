# SCC restore leaves rr0 contradicting the emptied FIFOs

**Keywords:** snapshot,restore,scc,z85c30,rr0,tx_buffer_empty,serial,console,tx_int_pending,update_ip
**Category:** snapshot

## Scope

Two rr0 bits, and their reachability is not the same. **`RX_CHAR_AVAILABLE` is
hit by ordinary use, in roughly 1 save in 10.** `TX_BUFFER_EMPTY` cannot be
reached through `save_snapshot` at all; forcing it is defensive hardening.
Measurements for both are below.

Neither fixes the failure this was found while chasing, a guest that survives
`iris-ci restore` alive but silent. That had a separate cause, since fixed and
recorded in `l1d-dirty-bit-aliased-physical-address-bit-31.md`. It was never the
SCC: the panic was byte-identical on the fixed and unfixed trees.

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

## Reachability: RX yes, TX no

**RX, about 1 save in 10.** `iris-ci serial-send` immediately followed by
`iris-ci save`, at a root shell, nothing doctored:

| sequence | `ch_b status` |
|---|---|
| no injection (control) | `0x2c` |
| 2000 chars, `--no-cr`, then save | `0x2d` |
| 100 chars, `--no-cr`, then save | `0x2d` |
| 10 trials, 100 chars no-CR | `0x2d` in 1 |
| 10 trials, 30 chars with CR | `0x2d` in 1 |

Bit 0 is `RX_CHAR_AVAILABLE`. Nothing drains `rx_queue` except the guest calling
`read_data`, and `Machine::stop()` halts the CPU first, so any host input the
guest has not consumed is queued at save time with the bit set. `iris-ci login`
and `iris-ci run` both inject through `serial-send`, so a script that runs a
command and then snapshots is exactly this shape.

**TX, never, and there is a mechanism.** `Machine::stop()` calls `cpu.stop()`
first and only reaches the SCC join several milliseconds later, and in that
window the TX thread drains with nothing refilling it. Instrumented over five
`save` calls:

```
cpu.stop() done at 115 us, scc joined by 3818 us; window = 3703 us
cpu.stop() done at  81 us, scc joined by 8452 us; window = 8371 us
cpu.stop() done at  21 us, scc joined by 8046 us; window = 8025 us
cpu.stop() done at  37 us, scc joined by 7560 us; window = 7523 us
cpu.stop() done at  42 us, scc joined by 8204 us; window = 8162 us
tx_delay: channel A 192 us, channel B 384 us
```

A full 4-deep FIFO drains in 4 x 384 = 1536 us and the bit is set on the first
pop, so the narrowest observed window still clears it with 2.4x margin. Reaching
the TX state needed a snapshot edited by hand to `wr1 = 0x13` / `status = 0x28`.

Earlier drafts of this note explained the TX rarity by saving being quiescent.
That was wrong twice over: `stop()` freezes the FIFO rather than draining it *at
the moment it runs*, and the reason the state does not survive is the join delay
afterwards, not quiescence.

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

## The panic that led here was something else, and it is fixed

**Nothing in this note is the cause of the post-restore kernel panic.** That was
`dirty` serialized into bit 27 of the L1D tag word, which aliased physical
address bit 31. See
`l1d-dirty-bit-aliased-physical-address-bit-31.md`, which also carries the dead
ends and the oracle guidance from this investigation.

Two results from here are worth keeping because they bound the search:

- **The SCC is not the cause.** One snapshot restored on three binaries, `main`,
  the fixed tree, and the fixed tree plus a CPU re-derivation patch, produced a
  byte-identical panic each time. That snapshot's channel B has `wr1 = 0x11` and
  `status = 0x2c`, so every arm of the SCC fix is an arithmetic no-op on it.
- **The restore machinery works below the OS.** At the PROM `Option?` menu,
  `save prom1` then `restore prom1` then `serial-send 5` gives a live Command
  Monitor prompt, so whatever was broken was specific to a booted IRIX.

An earlier revision of this note listed "cache tag serialization, lossless at
this size" among the things ruled out with a measurement. That was wrong: the
argument was static, about the tag types not being lossy, and it missed the bit
aliasing entirely. Left recorded here because a confident static exclusion that
turns out to be the bug is worth remembering.

## Two things that will waste your time

- Restoring into a process that was never `start`ed gives silence and a pinned
  CPU thread, with **zero** SCC and IOC register accesses in the window. It is
  not an ISR storm, it is spinning somewhere else. Note also that `idle-pause` is
  off by default, so `MIPS-CPU` sits at 100% at a healthy idle shell too: use
  register-access counts, not CPU percentage, to tell a wedged guest from a live
  one.
- `iris-ci` flakes under load with `connect ...: Resource temporarily unavailable
  (os error 11)`, which aborts `boot` early and looks like a boot failure. Retry.
- `PANIC: stack underflow/overflow` after a restore has been blamed on having an
  NFS export mounted in the guest. It is not NFS. The same panic appeared on a
  raw image with no NFS mount anywhere, so it was the L1D bug wearing a
  different hat.
- A sibling `<base>.chd.diff.chd` was observed appearing during an extraction
  run, which silently changes the next run's guest state. An earlier draft
  blamed `chd_extract` for ignoring `IRIS_CHD_DIFF_DIR`, and that is wrong:
  `diff_path_for` (`src/chd_disk.rs:305`) does honour it, and `chd_extract`
  opens with COW off so it creates no diff at all. Cause not established. Check
  for a sibling before trusting an A/B.
- `cp -a` of a live `.diff.chd` gives a torn copy. One reused disk state booted
  straight into `Error during dump: i/o error`. Stop the emulator first.
