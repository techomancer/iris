# VPro threads, the two locks and the command ring

Status: design (iris `src/dev/odyssey/{interface,fifo,worker,mod}.rs`,
2026-10-30). Check against the code before relying on details.

**Why it is shaped this way.** The board decodes and draws behind the
processor: the driver stores more commands and polls registers while a
primitive is drawn. The model must let the processor go on, yet never let a
draw and a processor access touch the same board state.

## Threads

- The CPU thread: widget accesses.
- `odyssey-work`: runs queued command-FIFO stores.
- `odyssey-display`: counts the 60 Hz retrace and presents the frame.
- Rasteriser bands: large filled draws and clears run in bands of rows on a
  small pool. Each band borrows only its own rows of the planes
  (`RowsMut::split` in memory.rs), so bands never share a pixel.

## Two locks

`Board` (everything but the host registers) is behind the board lock.
`HostInterface` is behind the front lock and holds the registers that do not
wait for drawing: identity, control, the interrupt bank, the routing, the
swap bank and the FIFO status, plus the interrupt state and the credit
return.

- Lock order: board, then front. Nothing takes the board lock while holding
  the front lock.
- A front access (those registers, and command stores) takes only the front
  lock and never waits for a draw.
- Every other access takes the board lock, which waits out the current
  slice, then runs whatever is still queued before it (`work_drain`), so it
  sees all earlier stores.
- The display thread reads the planes under the board lock, so no draw runs
  while a frame is resolved. The retrace is counted under the board lock and
  can be up to one slice late.

## The 1 ms slice

The worker sleeps on a condvar until a store joins an empty queue, waits
`GAP` (100 us, both locks free), then holds the board lock for a slice of
at most `SLICE` (1 ms of host time, checked every 16 stores) and releases it
with `unlock_fair`. The gap exists because a thread that drops a lock and
takes it straight back usually wins it again; without the gap a processor
waiting for the board could wait out many slices. A slice never lets go
part-way: a draw holds `&mut` into the board. 1 ms stays well under the
guest's 10 ms clock tick. A slice always runs at least one store.

## The command ring

The PROM's and IRIX's textports send hundreds of thousands of FIFO words per
scrolled line, so a store must cost little. Stores go into an `rtrb`
single-producer single-consumer ring (`fifo.rs`). The processor pushes
without the front lock and wakes the worker only when it sleeps. The
consumer, whoever holds the board lock (worker or a draining access), pops.
While the worker runs every command store queues, so decoding and drawing
never run on the processor's thread.

Counted stores (machines whose processor interface counts graphics credits;
the Octane has none and the calls are no-ops) are credited as they enter the
FIFO's first 64 places: at the store when there is room, or when the stores
ahead have run, with the node holding the credit as owed meanwhile.

## The six FIFO ordering rules

These are guest-visible; any change to the queue must keep them.

1. One in-order queue: stores run in arrival order.
2. An access that waits drains the queue first.
3. STATUS0's empty and low-water bits follow the number of stores pushed
   and not yet taken out.
4. The low-water flag latches as a store retires below the mark (16),
   whatever the enable mask says; it reaches the driver only while enabled.
5. Completion words are written when the command runs, not when it queues.
6. The queue is drained when the machine stops and cleared at reset.

## Tracing it

`docs/perfetto.md`: `VPro cfifo level` is sampled at slice edges and STATUS0
reads, not per store. The `slice` slice on `odyssey-work` against `drain` on
the CPU thread shows who is running the draws. A long `slice` with the CPU
blocked on a read points at a draw holding the board lock.
