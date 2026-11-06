# A refused command store answers busy; it never spins on the host

Status: fixed 2026-11-06.

**Symptom.** `/usr/sgiquake1/quake.gl` on a jitv2 build logged "RAD thread
held off for more than 32 mS" every few seconds, with audio drop-outs. The
interpreter build rarely did. A perfetto trace (`tools/rad-holdoff.py`)
showed IP7 raised and unmasked for 40-55 ms with no instruction boundary
examining it at all.

**Cause.** `bus_write` asked `CommandFifo::has_room`, which counted the
stores the board's thread had retired, then pushed. The `GFifo` consumer
publishes its head only every 64 entries or when it drains to empty, so the
ring could still look full to `try_push` when the counts said there was room.
`write_command` then looped `try_push` + `yield_now` until the board's thread
next published, which could take tens of ms while it waited on the raster
unit. That loop is on the CPU thread and takes no interrupts. `perf` showed
the CPU thread in `do_sched_yield` under `CommandFifo::push`.

**Fix.** `has_room` also asks the ring (`GFifo::has_space`, the same test
`try_push` makes), so the bus answers a full ring busy and the CPU takes
interrupts between retries. `retire` publishes the head after each run, so
the pessimistic view costs no false refusals. `submit` serialises the
producers, so a push the room check allowed goes in; `write_command`'s wait
loop is only for callers off the bus.

**Rule.** For any board: a CPU store into a `GFifo` must return `BUS_BUSY`
when `try_push` fails, never wait in the device. Any check made before the
push must use the ring's published view, not a count of what the consumer
has taken. REX3, GR2, IMPACT and the O2 already return busy directly.

A second hole made it worse under jitv2: compiled code skips the entry
word's interrupt check (`skip_entry_preamble`), and `step_jit` does not run
the interpreter's preamble, so a store retried at the same entry PC never
looked at a pending interrupt.
